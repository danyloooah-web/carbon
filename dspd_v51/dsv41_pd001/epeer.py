"""dsv41-pd068 EPEER (sglang#47033's peer_read, on pd034's column exchange): each rank reads its own rows' Engram table
rows in place from the owning GPUs over NVLink, instead of IDGATHER + the own-column gather + all_to_all_single + the
placement. A rank PGEMM routes (engram_cols.engram_forward) gets the wkv GEMM's MXFP8 pair in one kernel with
q8_gather_pack's arithmetic and q8_place's layout (pd054 EQ8's bytes, which are pd052 EQUANT's); any other rank
(engram_cols.dp_sharded_lookup) gets stock's placed bf16 rows (EPLACE's bytes). cuda_epeer.SOURCE holds both kernels.

Domain: exactly pd034's column-exchange domain (engram_cols._cols_counts: all-gathered token counts, the batch-wide
extend flag, the padding mode, capture), so every DP rank takes EPEER in the same forwards, IDLE ranks included, and
none of them enters the exchange's collectives there; outside it engram_cols runs unchanged. EPEER has no collective
in a served forward (the tables are read-only weights, written once at load).

Mapping: prepare exports this rank's shards (legacy CUDA IPC handles of the existing allocations; it raises unless the
allocation is legacy-IPC capable, so a bad allocator fails at load, not mid-forward); the first served forward
exchanges the handles over the TP group's CPU group and opens every peer's (nothing is copied or allocated). Every rank
reaches the exchange in the same forward, as the domain is decided from all-gathered facts. The peers' row ranges must
equal the ranges this rank derives from the Engram layout (engram_cols.check_domain's column-aligned shards), and each
mapped allocation must hold its shard.

CHECK (the twin, never submitted): the first CHECK_CALLS served calls (one an Engram layer on every rank, so the counts
advance alike) also run pd065's exchange and placement for the same ids and compare every byte under
"dsv41-pd068chk"."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from functools import lru_cache

import torch

CHECK = False  # gen_pd068 --check: the twin's comparisons (measurement only, never submit)
CHECK_CALLS = 4  # counted over the process's served calls (every rank advances alike)
PREFIX = "dsv41-pd068chk" if CHECK else "dsv41-pd068"
WORLD, COLS, DIM, BLOCK, HANDLE = 4, 24, 256, 32, 80

_EMBEDS: dict = {}  # id(EngramEmbedding view) -> (layer_hash_index, the view, [(lo, hi)] a rank)
_TABLES: dict = {}  # id(EngramEmbedding view) -> CPU int64 [4, 4]: (weight address, scale address, lo, hi) a rank
_LOCAL: list = []  # [sorted view ids, CPU u8 [layers, 2, 80] of this rank's exported handles], made by prepare
_CALLS = [0]
_paths: dict = {}


def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial header (as skfuse2._write)."""
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
    path = _paths.get("epeer")
    if path is None:
        from sglang.kernels.jit.utils.compile.cache import cache_root

        from dsv41_pd001 import cuda_epeer

        data = cuda_epeer.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd068",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cuda_epeer.NAME)
        if not os.path.isfile(path):
            _write(path, data)
        _paths["epeer"] = path
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from dsv41_pd001 import cuda_epeer

    return load_jit("pd068_epeer", cuda_epeer.TAG, cuda_files=[_header()],
                    cuda_wrappers=[("export_handle", "Pd068Epeer::export_handle"),
                                   ("open_handle", "Pd068Epeer::open_handle"),
                                   ("peer_place", "Pd068Epeer::peer_place"),
                                   ("peer_lookup_bf16", "Pd068Epeer::peer_lookup_bf16")])


def _map_all() -> None:
    """Collective over the TP group's CPU group, once a process: every registered layer's shards exported, exchanged
    and the peers' opened. Raises on any mismatch (no partial mapping is kept)."""
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    t0 = time.perf_counter()
    group = get_parallel().tp_group
    rank, world = group.rank_in_group, group.world_size
    if world != WORLD:
        raise RuntimeError(f"dsv41-pd068 EPEER requires a {WORLD}-rank TP group, got {world}")
    keys, local = _LOCAL
    lib = module()
    ranges = torch.zeros((len(keys), 2), dtype=torch.int64)
    for i, key in enumerate(keys):
        _, embed, owned = _EMBEDS[key]
        ranges[i, 0], ranges[i, 1] = embed.row_start, embed.row_start + embed.rows
        if tuple(owned[rank]) != (embed.row_start, embed.row_start + embed.rows):
            raise RuntimeError(f"dsv41-pd068 EPEER: rank {rank} holds rows {(embed.row_start, embed.row_start + embed.rows)}"
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
                raise RuntimeError(f"dsv41-pd068 EPEER: rank {r} reports rows {(lo, hi)}, the layout gives {owned[r]}")
            for j, row_bytes in enumerate((DIM, DIM // BLOCK)):
                blob = handles[r][i, j]
                offset, size = (int(v) for v in blob[64:80].view(torch.int64))
                if offset + (hi - lo) * row_bytes > size:
                    raise RuntimeError(f"dsv41-pd068 EPEER: rank {r}'s allocation ({size} B from offset {offset}) "
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
    print(f"{PREFIX} EPEER mapped {len(keys)} Engram layers x {world - 1} peers in "
          f"{1e3 * (time.perf_counter() - t0):.1f} ms", file=sys.stderr, flush=True)


def _counts(embed, forward_batch):
    """This forward's DP token counts when EPEER serves it (pd034's column-exchange domain: all-gathered counts, the
    batch-wide extend flag, the padding mode, capture), else None; the first served forward maps the peers' shards."""
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    from dsv41_pd001 import engram_cols

    if id(embed) not in _EMBEDS:
        return None
    counts = engram_cols._cols_counts(forward_batch, get_global_dp_buffer_len())
    if counts is not None and id(embed) not in _TABLES:
        _map_all()
    return counts


def _ids(hash_ids: torch.Tensor, counts, n_local: int):
    from sglang.srt.runtime_context import get_parallel

    mine = counts[get_parallel().attn_dp_rank]
    if hash_ids.shape[0] != n_local or n_local < mine:
        raise RuntimeError(f"dsv41-pd068 EPEER: {hash_ids.shape[0]} hash-id rows for {n_local} local rows, DP count "
                           f"{mine}")
    return (hash_ids if hash_ids.stride(-1) == 1 else hash_ids.contiguous()), int(mine)


def lookup(engram, hash_ids: torch.Tensor, forward_batch, n_local: int):
    """engram_forward's wkv MXFP8 pair (q, s) for this rank's rows (a rank PGEMM routes), read in place; None outside
    the domain."""
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
    """engram_cols.dp_sharded_lookup's [n_local, 6144] bf16 rows for a rank PGEMM does not route, read in place; None
    outside the domain."""
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
    """Twin: pd065's exchange for the same ids (its collectives: every rank serves one EPEER call an Engram layer, so
    the counts advance alike) and its result -- EQ8's place / EQUANT for a routed rank's pair, EPLACE for the bf16
    rows -- every byte compared."""
    _CALLS[0] += 1
    if _CALLS[0] > CHECK_CALLS:
        return
    from dsv41_pd001 import engram_cols, skfuse, skfuse2

    exchanged = engram_cols._exchange(embed, hash_ids, forward_batch, engram=engram)
    if exchanged is None:
        raise RuntimeError("dsv41-pd068chk EPEER: pd065's exchange declined a call EPEER served")
    received, mine, n_local, width, fp8 = exchanged
    if engram is not None:
        ref = skfuse2.place(received, mine, n_local) if fp8 else skfuse.engram_quant(received, mine, n_local)
    else:
        local = received.new_empty((n_local, WORLD * width))
        skfuse.engram_place(received, local, mine)
        if n_local > mine:
            local[mine:].zero_()
        ref = (local,)
    diffs = [int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(got, ref)]
    print(f"{PREFIX} EPEER{'' if engram is not None else ' bf16'} {hash_ids.device} rows {n_local} mine {mine} "
          f"{'fp8' if fp8 else 'bf16'}-exchange: {'OK' if not any(diffs) else 'MISMATCH'} bytes-differ {diffs}",
          file=sys.stderr, flush=True)


def register(engram_view, model) -> None:
    """At prepare: an Engram view EPEER may serve, with every rank's shard rows from the layout (the column-aligned
    shards engram_cols.check_domain verified when it built the view)."""
    layout = model.engram_layout
    bounds = layout.column_bounds(engram_view.layer_hash_index)
    per = (len(bounds) - 1) // WORLD
    if per * WORLD != COLS or len(bounds) - 1 != COLS:
        raise RuntimeError(f"dsv41-pd068 EPEER requires {COLS} hash columns over {WORLD} ranks, got {len(bounds) - 1}")
    owned = [(bounds[r * per], bounds[(r + 1) * per]) for r in range(WORLD)]
    embed = engram_view.embed
    if embed.weight.dtype != torch.float8_e4m3fn or tuple(embed.weight.shape) != (embed.rows, DIM) or \
            tuple(embed.scale.shape) != (embed.rows, DIM // BLOCK) or not embed.weight.is_contiguous() or \
            not embed.scale.is_contiguous():
        raise RuntimeError("dsv41-pd068 EPEER requires contiguous e4m3 [rows, 256] tables with [rows, 8] e8m0 scales")
    _EMBEDS[id(embed)] = (engram_view.layer_hash_index, embed, owned)


def export_local() -> None:
    """This rank's handles of every registered layer's weight and scale shards (raises unless IPC capable)."""
    keys = sorted(_EMBEDS, key=lambda k: _EMBEDS[k][0])
    lib = module()
    local = torch.zeros((len(keys), 2, HANDLE), dtype=torch.uint8)
    for i, key in enumerate(keys):
        embed = _EMBEDS[key][1]
        lib.export_handle(embed.weight.data_ptr(), local[i, 0])
        lib.export_handle(embed.scale.data_ptr(), local[i, 1])
    _LOCAL[:] = [keys, local]


def prepare(model) -> None:
    """Build the kernel and export this rank's shards now (either failure fails prepare); the peers' shards are opened
    by the first served forward."""
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        return
    if not _EMBEDS:
        raise RuntimeError("dsv41-pd068 EPEER: no Engram view registered (engram_cols.engram_view registers each)")
    export_local()
    print(f"{PREFIX} EPEER prepared: {len(_EMBEDS)} Engram layers exported", file=sys.stderr, flush=True)
