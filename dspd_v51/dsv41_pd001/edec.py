"""dsv41-pd071 EDEC: the Engram layers' decode lookup without collectives. Bitwise pd065.

pd065 (stock's EngramEmbedding._dp_sharded_lookup; engram_cols serves only big SUM_LEN prefills): in every decode /
verify step each of the Engram layers 1 and 14 all-gathers every DP rank's [rows, 24] hash ids (NCCL), looks all of
them up in its own column-aligned table shard (stock's engram_gather: zero where another rank owns the row) and
reduce-scatters the [4 x rows, 6144] bf16 buffer back (NCCL); a rank's row is the one owner's value plus three zeros.
In the PD decode engine's verify graphs that costs 45-70 us a layer on the main stream (all-gather 10-11.5, gather 2.4,
a ~16 us gap, reduce-scatter 13-14; pd040 traces), every step.

EDEC: each rank reads its own rows' 24 table rows in place from the owners' shards over NVLink with the fork's own
kernel sglang.kernels.ops.embeddings.engram_gather.engram_gather_peer (fp32(e4m3) x 2^(e - 127), one bf16 rounding,
-0 -> +0: the fork's SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS lookup, documented there as bit-equal to the zero-fill + sum
lookup), on legacy CUDA IPC mappings of the existing device shards (nothing copied or allocated) opened once at
prepare. prepare also proves the claim on this engine's tables: ids spanning every hash column's range, looked up by
stock's gather + an NCCL sum over the TP group and by EDEC, every byte equal on every rank, or prepare raises.

Domain (every input is shared by all DP ranks, IDLE ranks included, so all ranks skip the collectives together):
not a prefill-role engine (prefill engines never map or serve EDEC); the batch has no extend (all-gathered flag); MAX_LEN
padding, no gatherv (the forward's padding mode); global DP buffer <= WORLD x MAX_REQUESTS x the verify width (a
captured graph's bucket, the same on every rank under DP-max replay). There stock's lookup is exactly gather -> owned
rows -> reduce-scatter, which needs every rank's local rows to be a quarter of the buffer; EDEC raises if not. Every
other call (prefills, bigger decode batches, SUM_LEN) is pd065's lookup.

CHECK (the twin, never submitted): every served call also runs pd065's lookup (its collectives, on every rank in the
same calls) and compares the bytes: eager calls print OK / MISMATCH; captured graphs keep per-layer device counters
(calls, mismatched elements) mirrored into pinned host memory every replay, printed by a monitor thread."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from functools import lru_cache

import torch

CHECK = False  # gen_pd071 --check: the twin's comparisons (measurement only, never submit)
PREFIX = "dsv41-pd071chk" if CHECK else "dsv41-pd071"
MAX_REQUESTS = 48  # dsv41pd.v41f EDEC48: every decode graph bucket (cuda_graph_max_bs_decode 48); pd071 stopped at 24
CHECK_EAGER = 8  # eager twin comparisons printed (every rank serves the same calls)
WORLD, COLS, DIM, BLOCK, HANDLE = 4, 24, 256, 32, 80
SELFTEST_IDS = 2048  # random ids a hash column in prepare's bitwise self-test (plus each column's first / last id)

_VIEWS: dict = {}  # id(EngramEmbedding view) -> (layer_hash_index, view)
_TABLES: dict = {}  # id(EngramEmbedding view) -> (w_bias, s_bias, starts) CUDA int64 [WORLD]
_STATE = {"on": False, "max_rows": 0, "detail": "not prepared"}
_TWIN: dict = {}  # CHECK: "dev" CUDA int64 [layers, 2], "host" pinned int64 [layers, 2] (+ its device address), "eager"
_paths: dict = {}


def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial header."""
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
    path = _paths.get("edec")
    if path is None:
        from sglang.kernels.jit.utils.compile.cache import cache_root

        from dsv41_pd001 import cuda_edec

        data = cuda_edec.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd071",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cuda_edec.NAME)
        if not os.path.isfile(path):
            _write(path, data)
        _paths["edec"] = path
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from dsv41_pd001 import cuda_edec

    return load_jit("pd071_edec", cuda_edec.TAG, cuda_files=[_header()],
                    cuda_wrappers=[("export_handle", "Pd071Edec::export_handle"),
                                   ("open_handle", "Pd071Edec::open_handle"),
                                   ("host_device_ptr", "Pd071Edec::host_device_ptr"),
                                   ("mirror", "Pd071Edec::mirror")])


def _views(model_view) -> list:
    """(layer_hash_index, EngramEmbedding view) of every Engram layer view, checked for in-place peer reads."""
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang.srt.layers.engram import EngramEmbedding

    found = []
    for layer in model_view.layers:
        engram = getattr(layer, "engram", None)
        if engram is None:
            continue
        embed = engram.embed
        if not isinstance(embed, EngramEmbedding) or getattr(embed, "_pd034_cols", None) is None:
            raise TypeError("dsv41-pd071 EDEC requires pd034's Engram embedding views (engram_cols.engram_view)")
        if embed.peer is not None or embed.host_table is not None or embed.staged_by_scheduler:
            raise RuntimeError("dsv41-pd071 EDEC requires device-resident shards (no peer, host table or staging)")
        w, s = embed.weight, embed.scale
        if (FP8_BLOCK_SIZE != BLOCK or embed.dim != DIM or w.dtype != torch.float8_e4m3fn
                or s.dtype != torch.float8_e8m0fnu or tuple(w.shape) != (embed.rows, DIM)
                or tuple(s.shape) != (embed.rows, DIM // BLOCK) or not w.is_contiguous() or not s.is_contiguous()
                or not w.is_cuda or not s.is_cuda):
            raise RuntimeError(f"dsv41-pd071 EDEC requires contiguous device e4m3 [rows, {DIM}] tables with "
                               f"[rows, {DIM // BLOCK}] e8m0 scales")
        found.append((engram.layer_hash_index, embed))
    return found


def _verify_width() -> int:
    """Verify rows a request: the resolved speculative_num_draft_tokens, or for DSpark the fork's own rule
    (arg_groups/speculative_hook._handle_dspark: verify window = block size + 1) when the bag still holds the raw
    None; 1 without speculative decoding."""
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
        raise RuntimeError(f"dsv41-pd071 EDEC: speculative decoding ({algorithm}) without a verify width ({width!r})")
    return width


def _precondition(found) -> str:
    """The tables' bytes for which engram_gather_peer's result provably equals stock's gather + sum, all ranks agreed:
    no NaN e4m3 code (0x7F / 0xFF) and no NaN e8m0 scale (255) (NaN payloads could differ), and no non-zero code in a
    block whose scale byte is below 10 (only there can a non-zero fp32 product, |v| >= 2^-9 x 2^(e - 127), round to
    a signed bf16 zero that stock's sum would turn into +0). Scanned in row chunks of this rank's shards; raises
    otherwise."""
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
        raise RuntimeError(f"dsv41-pd071 EDEC: the Engram tables hold {nan_codes} NaN codes, {nan_scales} NaN scales and "
                           f"{tiny} non-zero blocks under scale byte 10: the peer lookup is not provably stock's bytes")
    return "tables: no NaN code / scale, no non-zero block under scale byte 10"


def _map(found) -> None:
    """Collective over the TP group: every rank exports its shards' legacy IPC handles and opens its peers'. The
    tables give each rank's (fp8 row, e8m0 scale) base biased by its first row, as engram_gather_peer expects."""
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
    opened: dict = {}  # a peer allocation's handle bytes -> its base here (each handle is opened once a process)
    for i, (index, embed) in enumerate(found):
        starts, w_bias, s_bias = [], [], []
        for r in range(world):
            lo, hi = int(spans[r][i, 0]), int(spans[r][i, 1])
            if r and lo != int(spans[r - 1][i, 1]):
                raise RuntimeError(f"dsv41-pd071 EDEC: layer {index} shards are not contiguous at rank {r}")
            addresses = []
            for j, row_bytes in enumerate((DIM, DIM // BLOCK)):
                blob = handles[r][i, j]
                offset, size = (int(v) for v in blob[64:80].contiguous().view(torch.int64))
                if offset + (hi - lo) * row_bytes > size:
                    raise RuntimeError(f"dsv41-pd071 EDEC: rank {r}'s allocation ({size} B from offset {offset}) does "
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
            raise RuntimeError(f"dsv41-pd071 EDEC: layer {index} shard starts {starts} disagree with rank {rank}'s "
                               f"row_start {embed.row_start}")
        _TABLES[id(embed)] = tuple(torch.tensor(v, dtype=torch.int64, device=device) for v in (w_bias, s_bias, starts))


def _self_test(model, found) -> int:
    """Stock's arithmetic against EDEC's on this engine's tables, every byte: ids spanning every hash column's subtable
    (each column's first and last id and SELFTEST_IDS random ones, the same on every rank) through stock's gather of
    the owned rows + an NCCL sum over the TP group, and through engram_gather_peer. Returns the rows compared."""
    import torch.distributed as dist
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather, engram_gather_peer
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group.device_group
    gen = torch.Generator().manual_seed(20261030)
    compared = 0
    for index, embed in found:
        bounds = model.engram_layout.column_bounds(index)
        if len(bounds) - 1 != COLS:
            raise RuntimeError(f"dsv41-pd071 EDEC: layer {index} has {len(bounds) - 1} hash columns, not {COLS}")
        cols = []
        for c in range(COLS):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            rand = torch.randint(lo, hi, (SELFTEST_IDS,), generator=gen, dtype=torch.int64)
            cols.append(torch.cat((torch.tensor([lo, hi - 1], dtype=torch.int64), rand)))
        ids = torch.stack(cols, dim=1).to(embed.weight.device)  # [rows, COLS], column c inside column c's subtable
        ref = torch.empty((ids.shape[0], COLS, DIM), dtype=torch.bfloat16, device=ids.device)
        engram_gather(*embed._table_ptrs(), ids.view(-1), ref.view(-1, DIM), DIM, BLOCK,
                      row_lo=embed.row_start, row_hi=embed.row_start + embed.rows)
        dist.all_reduce(ref, group=group)
        got = torch.empty_like(ref)
        engram_gather_peer(ids, got, *_TABLES[id(embed)], DIM, BLOCK)
        bad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        torch.cuda.synchronize()
        if bad:
            raise RuntimeError(f"dsv41-pd071 EDEC: layer {index} self-test: {bad} of {ref.numel()} elements differ "
                               f"from stock's gather + sum")
        compared += ids.shape[0]
    return compared


def prepare(model_view, model) -> None:
    """Map every rank's shards, prove EDEC bitwise on them, then serve; a prefill-role engine keeps pd065's lookup."""
    from sglang.srt.runtime_context import get_disagg, get_parallel

    mode = get_disagg().disaggregation_mode
    if mode == "prefill":
        _STATE["detail"] = "off (prefill-role engine)"
        print(f"{PREFIX} EDEC {_STATE['detail']}", file=sys.stderr, flush=True)
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("dsv41-pd071 EDEC: prepare runs inside a CUDA graph capture (the peer mapping needs an "
                           "eager first call)")
    parallel = get_parallel()
    facts = (parallel.tp_size, parallel.attn_dp_size, parallel.attn_tp_size, parallel.attn_cp_size)
    if facts != (WORLD, WORLD, 1, 1) or parallel.tp_rank != parallel.attn_dp_rank:
        raise RuntimeError(f"dsv41-pd071 EDEC requires TP4 = DP-attention 4 (attention TP 1, no CP) with TP rank = DP "
                           f"rank, got (tp, dp, attn_tp, attn_cp) = {facts}")
    found = _views(model_view)
    if not found:
        raise RuntimeError("dsv41-pd071 EDEC: no Engram layer view")
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
        _twin_prepare(len(found), found[0][1].weight.device)
    print(f"{PREFIX} EDEC {_STATE['detail']} in {1e3 * (time.perf_counter() - t0):.0f} ms", file=sys.stderr,
          flush=True)


def _served(forward_batch, rows: int, n_local: int) -> bool:
    """pd065's lookup is the MAX_LEN gather -> owned rows -> reduce-scatter of a no-extend batch within MAX_REQUESTS
    requests a rank. Every input is the same on all DP ranks (all-gathered or captured), never the rank's own mode."""
    from sglang.srt.layers.dp_attention import is_dp_gatherv_active

    if not _STATE["on"] or forward_batch is None:
        return False
    padding = forward_batch.dp_padding_mode
    if forward_batch.is_extend_in_batch or padding is None or not padding.is_max_len() or is_dp_gatherv_active():
        return False
    if rows > WORLD * _STATE["max_rows"]:
        return False
    if rows != WORLD * n_local:
        raise RuntimeError(f"dsv41-pd071 EDEC: a MAX_LEN buffer of {rows} rows with {n_local} local rows (pd065's "
                           f"reduce-scatter needs a quarter on every rank)")
    return True


def lookup(embed, indices: torch.Tensor, forward_batch):
    """dp_sharded_lookup's [*indices.shape, DIM] bf16 rows read in place from the owners; None outside the domain."""
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    entry = _VIEWS.get(id(embed))
    if entry is None or not _served(forward_batch, get_global_dp_buffer_len(), indices.shape[0]):
        return None
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather_peer

    flat = indices.reshape(-1, indices.shape[-1])
    if flat.shape[-1] != COLS:
        raise RuntimeError(f"dsv41-pd071 EDEC: {flat.shape[-1]} hash columns, not {COLS}")
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    out = torch.empty((*flat.shape, DIM), dtype=torch.bfloat16, device=indices.device)
    if flat.shape[0]:
        engram_gather_peer(flat, out, *_TABLES[id(embed)], DIM, BLOCK)
    out = out.view(*indices.shape, DIM)
    if CHECK:
        _twin(entry[0], embed, indices, forward_batch, out)
    return out


# ---- CHECK (the twin) ------------------------------------------------------------------------------------------------
def _twin_prepare(layers: int, device) -> None:
    _TWIN["dev"] = torch.zeros((layers, 2), dtype=torch.int64, device=device)
    _TWIN["host"] = torch.zeros((layers, 2), dtype=torch.int64, pin_memory=True)
    address = torch.zeros(1, dtype=torch.int64)
    module().host_device_ptr(_TWIN["host"].data_ptr(), address)
    _TWIN["host_dev"] = int(address[0])
    _TWIN["eager"] = 0
    rank = device.index

    def monitor():
        last = None
        while True:
            time.sleep(15)
            now = _TWIN["host"].tolist()
            if now != last:
                last = now
                for layer, (calls, bad) in enumerate(now):
                    print(f"{PREFIX} EDEC graph cuda:{rank} engram {layer}: replays {calls} mismatched-elements {bad} "
                          f"{'OK' if bad == 0 else 'MISMATCH'}", file=sys.stderr, flush=True)

    threading.Thread(target=monitor, name="pd071-edec-twin", daemon=True).start()


def _twin(index: int, embed, indices, forward_batch, out) -> None:
    """pd065's lookup for the same ids (its collectives: every rank serves the same calls), every byte compared."""
    from dsv41_pd001 import engram_cols

    ref = engram_cols._stock_lookup()(embed, indices, forward_batch)
    bad = (out.view(torch.int16) != ref.view(torch.int16)).sum()
    if torch.cuda.is_current_stream_capturing():
        dev = _TWIN["dev"]
        dev[index, 0] += 1
        dev[index, 1] += bad
        module().mirror(dev, _TWIN["host_dev"])
        return
    _TWIN["eager"] += 1
    if _TWIN["eager"] <= CHECK_EAGER:
        n = int(bad)
        print(f"{PREFIX} EDEC eager {indices.device} engram {index} rows {indices.shape[0]}: "
              f"{'OK' if n == 0 else 'MISMATCH'} elements-differ {n}", file=sys.stderr, flush=True)
