from __future__ import annotations

import hashlib
import os
import sys
import time
from functools import lru_cache

import torch

CHECK = False
CHECK_CALLS = 4
PREFIX = "bundle-v9-check" if CHECK else "bundle-v9"
WORLD, COLS, DIM, BLOCK, HANDLE = 4, 24, 256, 32, 80

_EMBEDS: dict = {}
_TABLES: dict = {}
_LOCAL: list = []
_CALLS = [0]
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
    path = _paths.get("peer_read")
    if path is None:
        from sglang.kernels.jit.utils.compile.cache import cache_root

        from bundle_v9.dec import cuda_peer_read

        data = cuda_peer_read.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-peer-read",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cuda_peer_read.NAME)
        if not os.path.isfile(path):
            _write(path, data)
        _paths["peer_read"] = path
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from bundle_v9.dec import cuda_peer_read

    return load_jit("dec_peer_read", cuda_peer_read.TAG, cuda_files=[_header()],
                    cuda_wrappers=[("export_handle", "DecPeerRead::export_handle"),
                                   ("open_handle", "DecPeerRead::open_handle"),
                                   ("peer_place", "DecPeerRead::peer_place"),
                                   ("peer_lookup_bf16", "DecPeerRead::peer_lookup_bf16")])


def _map_all() -> None:
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    t0 = time.perf_counter()
    group = get_parallel().tp_group
    rank, world = group.rank_in_group, group.world_size
    if world != WORLD:
        raise RuntimeError(f"bundle-v9 PEER_READ requires a {WORLD}-rank TP group, got {world}")
    keys, local = _LOCAL
    lib = module()
    ranges = torch.zeros((len(keys), 2), dtype=torch.int64)
    for i, key in enumerate(keys):
        _, embed, owned = _EMBEDS[key]
        ranges[i, 0], ranges[i, 1] = embed.row_start, embed.row_start + embed.rows
        if tuple(owned[rank]) != (embed.row_start, embed.row_start + embed.rows):
            raise RuntimeError(f"bundle-v9 PEER_READ: rank {rank} holds rows {(embed.row_start, embed.row_start + embed.rows)}"
                               f", the layout gives {owned[rank]}")
    handles = [torch.empty_like(local) for _ in range(world)]
    spans = [torch.empty_like(ranges) for _ in range(world)]
    dist.all_gather(handles, local, group=group.cpu_group)
    dist.all_gather(spans, ranges, group=group.cpu_group)
    tables = {}
    for i, key in enumerate(keys):
        _, embed, owned = _EMBEDS[key]
        table = torch.zeros((WORLD, 4), dtype=torch.int64)
        for r in range(world):
            lo, hi = int(spans[r][i, 0]), int(spans[r][i, 1])
            if (lo, hi) != tuple(owned[r]):
                raise RuntimeError(f"bundle-v9 PEER_READ: rank {r} reports rows {(lo, hi)}, the layout gives {owned[r]}")
            for j, row_bytes in enumerate((DIM, DIM // BLOCK)):
                blob = handles[r][i, j]
                offset, size = (int(v) for v in blob[64:80].view(torch.int64))
                if offset + (hi - lo) * row_bytes > size:
                    raise RuntimeError(f"bundle-v9 PEER_READ: rank {r}'s allocation ({size} B from offset {offset}) "
                                       f"does not hold its {hi - lo}-row shard")
                if r == rank:
                    address = (embed.weight if j == 0 else embed.scale).data_ptr()
                else:
                    out = torch.zeros(1, dtype=torch.int64)
                    lib.open_handle(blob.contiguous(), out)
                    address = int(out[0])
                table[r, j] = address
            table[r, 2], table[r, 3] = lo, hi
        tables[key] = table
    _TABLES.update(tables)
    print(f"{PREFIX} PEER_READ mapped {len(keys)} Engram layers x {world - 1} peers in "
          f"{1e3 * (time.perf_counter() - t0):.1f} ms", file=sys.stderr, flush=True)


def _counts(embed, forward_batch):
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    from bundle_v9.dec import col_exchange

    if id(embed) not in _EMBEDS:
        return None
    counts = col_exchange._cols_counts(forward_batch, get_global_dp_buffer_len())
    if counts is not None and id(embed) not in _TABLES:
        _map_all()
    return counts


def _ids(hash_ids: torch.Tensor, counts, n_local: int):
    from sglang.srt.runtime_context import get_parallel

    mine = counts[get_parallel().attn_dp_rank]
    if hash_ids.shape[0] != n_local or n_local < mine:
        raise RuntimeError(f"bundle-v9 PEER_READ: {hash_ids.shape[0]} hash-id rows for {n_local} local rows, DP count "
                           f"{mine}")
    return (hash_ids if hash_ids.stride(-1) == 1 else hash_ids.contiguous()), int(mine)


def lookup(engram, hash_ids: torch.Tensor, forward_batch, n_local: int):
    embed = engram.embed
    counts = _counts(embed, forward_batch)
    if counts is None:
        return None
    ids, mine = _ids(hash_ids, counts, n_local)
    q = torch.empty((n_local, COLS * DIM), dtype=torch.float8_e4m3fn, device=hash_ids.device)
    s = torch.empty(((n_local + 127) // 128) * 128 * (COLS * DIM // BLOCK), dtype=torch.uint8, device=hash_ids.device)
    module().peer_place(ids, _TABLES[id(embed)], q, s, mine)
    if CHECK:
        _check(embed, hash_ids, forward_batch, (q, s), engram)
    return q, s


def lookup_bf16(embed, indices: torch.Tensor, forward_batch):
    counts = _counts(embed, forward_batch)
    if counts is None:
        return None
    n_local = indices.shape[0]
    ids, mine = _ids(indices, counts, n_local)
    out = torch.empty((n_local, COLS * DIM), dtype=torch.bfloat16, device=indices.device)
    module().peer_lookup_bf16(ids, _TABLES[id(embed)], out, mine)
    if CHECK:
        _check(embed, indices, forward_batch, (out,), None)
    return out


def _check(embed, hash_ids, forward_batch, got, engram) -> None:
    _CALLS[0] += 1
    if _CALLS[0] > CHECK_CALLS:
        return
    from bundle_v9.dec import col_exchange, prefill_fuse, exchange_fuse

    exchanged = col_exchange._exchange(embed, hash_ids, forward_batch, engram=engram)
    if exchanged is None:
        raise RuntimeError("bundle-v9-check PEER_READ: the exchange's exchange declined a call PEER_READ served")
    received, mine, n_local, width, fp8 = exchanged
    if engram is not None:
        ref = exchange_fuse.place(received, mine, n_local) if fp8 else prefill_fuse.engram_quant(received, mine, n_local)
    else:
        local = received.new_empty((n_local, WORLD * width))
        prefill_fuse.engram_place(received, local, mine)
        if n_local > mine:
            local[mine:].zero_()
        ref = (local,)
    diffs = [int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(got, ref)]
    print(f"{PREFIX} PEER_READ{'' if engram is not None else ' bf16'} {hash_ids.device} rows {n_local} mine {mine} "
          f"{'fp8' if fp8 else 'bf16'}-exchange: {'OK' if not any(diffs) else 'MISMATCH'} bytes-differ {diffs}",
          file=sys.stderr, flush=True)


def register(engram_view, model) -> None:
    layout = model.engram_layout
    bounds = layout.column_bounds(engram_view.layer_hash_index)
    per = (len(bounds) - 1) // WORLD
    if per * WORLD != COLS or len(bounds) - 1 != COLS:
        raise RuntimeError(f"bundle-v9 PEER_READ requires {COLS} hash columns over {WORLD} ranks, got {len(bounds) - 1}")
    owned = [(bounds[r * per], bounds[(r + 1) * per]) for r in range(WORLD)]
    embed = engram_view.embed
    if embed.weight.dtype != torch.float8_e4m3fn or tuple(embed.weight.shape) != (embed.rows, DIM) or \
            tuple(embed.scale.shape) != (embed.rows, DIM // BLOCK) or not embed.weight.is_contiguous() or \
            not embed.scale.is_contiguous():
        raise RuntimeError("bundle-v9 PEER_READ requires contiguous e4m3 [rows, 256] tables with [rows, 8] e8m0 scales")
    _EMBEDS[id(embed)] = (engram_view.layer_hash_index, embed, owned)


def export_local() -> None:
    keys = sorted(_EMBEDS, key=lambda k: _EMBEDS[k][0])
    lib = module()
    local = torch.zeros((len(keys), 2, HANDLE), dtype=torch.uint8)
    for i, key in enumerate(keys):
        embed = _EMBEDS[key][1]
        lib.export_handle(embed.weight.data_ptr(), local[i, 0])
        lib.export_handle(embed.scale.data_ptr(), local[i, 1])
    _LOCAL[:] = [keys, local]


def prepare(model) -> None:
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        return
    if not _EMBEDS:
        raise RuntimeError("bundle-v9 PEER_READ: no Engram view registered (col_exchange.engram_view registers each)")
    export_local()
    print(f"{PREFIX} PEER_READ prepared: {len(_EMBEDS)} Engram layers exported", file=sys.stderr, flush=True)
