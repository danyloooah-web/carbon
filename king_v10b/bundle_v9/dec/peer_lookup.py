from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from functools import lru_cache

import torch

CHECK = False
PREFIX = "bundle-v9-check" if CHECK else "bundle-v9"
MAX_REQUESTS = 48
CHECK_EAGER = 8
WORLD, COLS, DIM, BLOCK, HANDLE = 4, 24, 256, 32, 80
SELFTEST_IDS = 2048

_VIEWS: dict = {}
_TABLES: dict = {}
_STATE = {"on": False, "max_rows": 0, "detail": "not prepared"}
_MIRROR: dict = {}
_paths: dict = {}


def _write(path: str, data: bytes) -> None:
    staging = f"{path}.{os.getpid()}.tmp"
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(staging, path)


def _header() -> str:
    path = _paths.get("peer_lookup")
    if path is None:
        from sglang.kernels.jit.utils.compile.cache import cache_root

        from bundle_v9.dec import cuda_peer_lookup

        data = cuda_peer_lookup.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-peer-lookup",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cuda_peer_lookup.NAME)
        if not os.path.isfile(path):
            _write(path, data)
        _paths["peer_lookup"] = path
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from bundle_v9.dec import cuda_peer_lookup

    return load_jit("dec_peer_lookup", cuda_peer_lookup.TAG, cuda_files=[_header()],
                    cuda_wrappers=[("export_handle", "DecPeerLookup::export_handle"),
                                   ("open_handle", "DecPeerLookup::open_handle"),
                                   ("host_device_ptr", "DecPeerLookup::host_device_ptr"),
                                   ("mirror", "DecPeerLookup::mirror")])


def _views(model_view) -> list:
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang.srt.layers.engram import EngramEmbedding

    found = []
    for layer in model_view.layers:
        engram = getattr(layer, "engram", None)
        if engram is None:
            continue
        embed = engram.embed
        if not isinstance(embed, EngramEmbedding) or getattr(embed, "_col_views", None) is None:
            raise TypeError("bundle-v9 PEER_LOOKUP requires COL_EXCHANGE's Engram embedding views (col_exchange.engram_view)")
        if embed.peer is not None or embed.host_table is not None or embed.staged_by_scheduler:
            raise RuntimeError("bundle-v9 PEER_LOOKUP requires device-resident shards (no peer, host table or staging)")
        w, s = embed.weight, embed.scale
        if (FP8_BLOCK_SIZE != BLOCK or embed.dim != DIM or w.dtype != torch.float8_e4m3fn
                or s.dtype != torch.float8_e8m0fnu or tuple(w.shape) != (embed.rows, DIM)
                or tuple(s.shape) != (embed.rows, DIM // BLOCK) or not w.is_contiguous() or not s.is_contiguous()
                or not w.is_cuda or not s.is_cuda):
            raise RuntimeError(f"bundle-v9 PEER_LOOKUP requires contiguous device e4m3 [rows, {DIM}] tables with "
                               f"[rows, {DIM // BLOCK}] e8m0 scales")
        found.append((engram.layer_hash_index, embed))
    return found


def _verify_width() -> int:
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    algorithm = getattr(spec, "speculative_algorithm", None)
    if algorithm is None:
        return 1
    width = getattr(spec, "speculative_num_draft_tokens", None)
    if width is None and str(algorithm).upper().endswith("DSPARK"):
        block = getattr(spec, "speculative_dspark_block_size", None)
        width = int(block) + 1 if isinstance(block, int) else None
    if not isinstance(width, int) or width < 1:
        raise RuntimeError(f"bundle-v9 PEER_LOOKUP: speculative decoding ({algorithm}) without a verify width ({width!r})")
    return width


def _precondition(found) -> str:
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    counts = torch.zeros(3, dtype=torch.int64, device=found[0][1].weight.device)
    chunk = 1 << 20
    for _, embed in found:
        w8, s8 = embed.weight.view(torch.uint8), embed.scale.view(torch.uint8)
        for lo in range(0, embed.rows, chunk):
            w, sc = w8[lo:lo + chunk], s8[lo:lo + chunk]
            counts[0] += ((w & 0x7F) == 0x7F).sum()
            counts[1] += (sc == 255).sum()
            low = sc < 10
            if bool(low.any()):
                blocks = w.view(-1, DIM // BLOCK, BLOCK)[low]
                counts[2] += ((blocks & 0x7F) != 0).any(-1).sum()
    dist.all_reduce(counts, group=get_parallel().tp_group.device_group)
    nan_codes, nan_scales, tiny = (int(v) for v in counts.tolist())
    if nan_codes or nan_scales or tiny:
        raise RuntimeError(f"bundle-v9 PEER_LOOKUP: the Engram tables hold {nan_codes} NaN codes, {nan_scales} NaN scales and "
                           f"{tiny} non-zero blocks under scale byte 10: the peer lookup is not provably stock's bytes")
    return "tables: no NaN code / scale, no non-zero block under scale byte 10"


def _map(found) -> None:
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    rank, world = group.rank_in_group, group.world_size
    lib = module()
    local = torch.zeros((len(found), 2, HANDLE), dtype=torch.uint8)
    ranges = torch.zeros((len(found), 2), dtype=torch.int64)
    for i, (_, embed) in enumerate(found):
        lib.export_handle(embed.weight.data_ptr(), local[i, 0])
        lib.export_handle(embed.scale.data_ptr(), local[i, 1])
        ranges[i, 0], ranges[i, 1] = embed.row_start, embed.row_start + embed.rows
    handles = [torch.empty_like(local) for _ in range(world)]
    spans = [torch.empty_like(ranges) for _ in range(world)]
    dist.all_gather(handles, local, group=group.cpu_group)
    dist.all_gather(spans, ranges, group=group.cpu_group)
    device = found[0][1].weight.device
    opened: dict = {}
    for i, (index, embed) in enumerate(found):
        starts, w_bias, s_bias = [], [], []
        for r in range(world):
            lo, hi = int(spans[r][i, 0]), int(spans[r][i, 1])
            if r and lo != int(spans[r - 1][i, 1]):
                raise RuntimeError(f"bundle-v9 PEER_LOOKUP: layer {index} shards are not contiguous at rank {r}")
            addresses = []
            for j, row_bytes in enumerate((DIM, DIM // BLOCK)):
                blob = handles[r][i, j]
                offset, size = (int(v) for v in blob[64:80].contiguous().view(torch.int64))
                if offset + (hi - lo) * row_bytes > size:
                    raise RuntimeError(f"bundle-v9 PEER_LOOKUP: rank {r}'s allocation ({size} B from offset {offset}) does "
                                       f"not hold its {hi - lo}-row shard")
                if r == rank:
                    addresses.append((embed.weight if j == 0 else embed.scale).data_ptr())
                else:
                    key = bytes(blob[:64].tolist())
                    if key not in opened:
                        out = torch.zeros(1, dtype=torch.int64)
                        lib.open_handle(blob.contiguous(), out)
                        opened[key] = int(out[0])
                    addresses.append(opened[key] + offset)
            starts.append(lo)
            w_bias.append(addresses[0] - lo * DIM)
            s_bias.append(addresses[1] - lo * (DIM // BLOCK))
        if starts[0] != 0 or starts[rank] != embed.row_start:
            raise RuntimeError(f"bundle-v9 PEER_LOOKUP: layer {index} shard starts {starts} disagree with rank {rank}'s "
                               f"row_start {embed.row_start}")
        _TABLES[id(embed)] = tuple(torch.tensor(v, dtype=torch.int64, device=device) for v in (w_bias, s_bias, starts))


def _self_test(model, found) -> int:
    import torch.distributed as dist
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather, engram_gather_peer
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group.device_group
    gen = torch.Generator().manual_seed(20261030)
    compared = 0
    for index, embed in found:
        bounds = model.engram_layout.column_bounds(index)
        if len(bounds) - 1 != COLS:
            raise RuntimeError(f"bundle-v9 PEER_LOOKUP: layer {index} has {len(bounds) - 1} hash columns, not {COLS}")
        cols = []
        for c in range(COLS):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            rand = torch.randint(lo, hi, (SELFTEST_IDS,), generator=gen, dtype=torch.int64)
            cols.append(torch.cat((torch.tensor([lo, hi - 1], dtype=torch.int64), rand)))
        ids = torch.stack(cols, dim=1).to(embed.weight.device)
        ref = torch.empty((ids.shape[0], COLS, DIM), dtype=torch.bfloat16, device=ids.device)
        engram_gather(*embed._table_ptrs(), ids.view(-1), ref.view(-1, DIM), DIM, BLOCK,
                      row_lo=embed.row_start, row_hi=embed.row_start + embed.rows)
        dist.all_reduce(ref, group=group)
        got = torch.empty_like(ref)
        engram_gather_peer(ids, got, *_TABLES[id(embed)], DIM, BLOCK)
        bad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        torch.cuda.synchronize()
        if bad:
            raise RuntimeError(f"bundle-v9 PEER_LOOKUP: layer {index} self-test: {bad} of {ref.numel()} elements differ "
                               f"from stock's gather + sum")
        compared += ids.shape[0]
    return compared


def prepare(model_view, model) -> None:
    from sglang.srt.runtime_context import get_disagg, get_parallel

    mode = get_disagg().disaggregation_mode
    if mode == "prefill":
        _STATE["detail"] = "off (prefill-role engine)"
        print(f"{PREFIX} PEER_LOOKUP {_STATE['detail']}", file=sys.stderr, flush=True)
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("bundle-v9 PEER_LOOKUP: prepare runs inside a CUDA graph capture (the peer mapping needs an "
                           "eager first call)")
    parallel = get_parallel()
    facts = (parallel.tp_size, parallel.attn_dp_size, parallel.attn_tp_size, parallel.attn_cp_size)
    if facts != (WORLD, WORLD, 1, 1) or parallel.tp_rank != parallel.attn_dp_rank:
        raise RuntimeError(f"bundle-v9 PEER_LOOKUP requires TP4 = DP-attention 4 (attention TP 1, no CP) with TP rank = DP "
                           f"rank, got (tp, dp, attn_tp, attn_cp) = {facts}")
    found = _views(model_view)
    if not found:
        raise RuntimeError("bundle-v9 PEER_LOOKUP: no Engram layer view")
    t0 = time.perf_counter()
    tables = _precondition(found)
    _map(found)
    rows = _self_test(model, found)
    for index, embed in found:
        _VIEWS[id(embed)] = (index, embed)
    width = _verify_width()
    _STATE.update(on=True, max_rows=MAX_REQUESTS * width,
                  detail=f"{len(found)} Engram layers mapped ({mode} engine), {tables}, self-test {rows} rows x "
                         f"{COLS} columns bitwise, serving <= {MAX_REQUESTS} requests a rank (<= {MAX_REQUESTS * width} "
                         f"rows)")
    if CHECK:
        _mirror_prepare(len(found), found[0][1].weight.device)
    print(f"{PREFIX} PEER_LOOKUP {_STATE['detail']} in {1e3 * (time.perf_counter() - t0):.0f} ms", file=sys.stderr,
          flush=True)


def _served(forward_batch, rows: int, n_local: int) -> bool:
    from sglang.srt.layers.dp_attention import is_dp_gatherv_active

    if not _STATE["on"] or forward_batch is None:
        return False
    padding = forward_batch.dp_padding_mode
    if forward_batch.is_extend_in_batch or padding is None or not padding.is_max_len() or is_dp_gatherv_active():
        return False
    if rows > WORLD * _STATE["max_rows"]:
        return False
    if rows != WORLD * n_local:
        raise RuntimeError(f"bundle-v9 PEER_LOOKUP: a MAX_LEN buffer of {rows} rows with {n_local} local rows (the exchange's "
                           f"reduce-scatter needs a quarter on every rank)")
    return True


def lookup(embed, indices: torch.Tensor, forward_batch):
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    entry = _VIEWS.get(id(embed))
    if entry is None or not _served(forward_batch, get_global_dp_buffer_len(), indices.shape[0]):
        return None
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather_peer

    flat = indices.reshape(-1, indices.shape[-1])
    if flat.shape[-1] != COLS:
        raise RuntimeError(f"bundle-v9 PEER_LOOKUP: {flat.shape[-1]} hash columns, not {COLS}")
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    out = torch.empty((*flat.shape, DIM), dtype=torch.bfloat16, device=indices.device)
    if flat.shape[0]:
        engram_gather_peer(flat, out, *_TABLES[id(embed)], DIM, BLOCK)
    out = out.view(*indices.shape, DIM)
    if CHECK:
        _mirror(entry[0], embed, indices, forward_batch, out)
    return out


def _mirror_prepare(layers: int, device) -> None:
    _MIRROR["dev"] = torch.zeros((layers, 2), dtype=torch.int64, device=device)
    _MIRROR["host"] = torch.zeros((layers, 2), dtype=torch.int64, pin_memory=True)
    address = torch.zeros(1, dtype=torch.int64)
    module().host_device_ptr(_MIRROR["host"].data_ptr(), address)
    _MIRROR["host_dev"] = int(address[0])
    _MIRROR["eager"] = 0
    rank = device.index

    def monitor():
        last = None
        while True:
            time.sleep(15)
            now = _MIRROR["host"].tolist()
            if now != last:
                last = now
                for layer, (calls, bad) in enumerate(now):
                    print(f"{PREFIX} PEER_LOOKUP graph cuda:{rank} engram {layer}: replays {calls} mismatched-elements {bad} "
                          f"{'OK' if bad == 0 else 'MISMATCH'}", file=sys.stderr, flush=True)

    threading.Thread(target=monitor, name="peer-lookup-mirror", daemon=True).start()


def _mirror(index: int, embed, indices, forward_batch, out) -> None:
    from bundle_v9.dec import col_exchange

    ref = col_exchange._stock_lookup()(embed, indices, forward_batch)
    bad = (out.view(torch.int16) != ref.view(torch.int16)).sum()
    if torch.cuda.is_current_stream_capturing():
        dev = _MIRROR["dev"]
        dev[index, 0] += 1
        dev[index, 1] += bad
        module().mirror(dev, _MIRROR["host_dev"])
        return
    _MIRROR["eager"] += 1
    if _MIRROR["eager"] <= CHECK_EAGER:
        n = int(bad)
        print(f"{PREFIX} PEER_LOOKUP eager {indices.device} engram {index} rows {indices.shape[0]}: "
              f"{'OK' if n == 0 else 'MISMATCH'} elements-differ {n}", file=sys.stderr, flush=True)
