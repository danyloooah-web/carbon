
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
# king-v6 PEER-SMALL (RANKED #39, VERIFY_prefill section 4.1): the bf16 PEER-READ lookup (lookup_bf16) also serves
# the eager sum-len extend forwards below col_exchange.MIN_ROWS DP-global rows. Stock's dp_gather_replicate, R-row
# gather, bf16 all-reduce and dp_scatter go; the kernel, the -0 -> +0 rule and the bytes are those of its >= MIN_ROWS
# use. The FP8 place path (lookup) and the NCCL exchange keep MIN_ROWS. A rank serves the peer path iff every rank
# does: the floor is a constant and every other input of col_exchange._cols_counts is global (counts, extend in
# batch, padding mode), so no vote is needed. Decided once at prepare from these constants and the engine role.
PEER_SMALL = False  # king-v7s GPU test copy: off (king_v6t prefill OOM)
PEER_SMALL_MIN_ROWS = 1
PEER_SMALL_ROLES = ("prefill", "null")
# review R1 (VPRE correction 1): with PEER-SMALL on, PEER-READ's mapping (the king's lazy _map_all: a cpu_group
# all-gather plus IPC opens inside the first served forward) runs at prepare instead (_map_prepare), followed by a probe
# of PEER_SMALL_PROBE_ROWS rows per Engram layer: every rank's bf16 peer reads must equal, bitwise, each owner's stock
# _owned_rows (+0.0, stock's all-reduce rule). Every rank reaches every collective whatever fails locally, and the
# group decides together: on any failure PEER-SMALL is off on every rank and _TABLES stays empty, so the king's lazy
# >= MIN_ROWS mapping is unchanged. No collective or IPC open then runs inside a PEER-SMALL forward.
PEER_SMALL_PROBE_ROWS = 16

_EMBEDS: dict = {}
_BF16_MIN: list = []
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

        from bundle_v9 import cuda_peer_read

        data = cuda_peer_read.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-peer-read",
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

    from bundle_v9 import cuda_peer_read

    return load_jit("v9_peer_read", cuda_peer_read.TAG, cuda_files=[_header()],
                    cuda_wrappers=[("export_handle", "V9PeerRead::export_handle"),
                                   ("open_handle", "V9PeerRead::open_handle"),
                                   ("peer_place", "V9PeerRead::peer_place"),
                                   ("peer_lookup_bf16", "V9PeerRead::peer_lookup_bf16")])


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


def _counts(embed, forward_batch, small: bool = False):
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    from bundle_v9 import col_exchange as col_exchange

    if id(embed) not in _EMBEDS:
        return None
    floor = _BF16_MIN[0] if small and _BF16_MIN else col_exchange.MIN_ROWS
    counts = col_exchange._cols_counts(forward_batch, get_global_dp_buffer_len(), floor)
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
    counts = _counts(embed, forward_batch, small=True)
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
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

    from bundle_v9 import col_exchange as col_exchange, exchange_fuse as exchange_fuse, prefill_fuse as prefill_fuse

    if engram is None and col_exchange._cols_counts(forward_batch, get_global_dp_buffer_len()) is None:
        # king-v6 PEER-SMALL: below MIN_ROWS the reference is stock's own lookup, a collective every rank reaches
        # here because every rank served the peer path.
        ref = col_exchange._stock_lookup()(embed, hash_ids, forward_batch).view(got[0].shape)
        diffs = [int((got[0].view(torch.uint8) != ref.view(torch.uint8)).sum())]
        print(f"{PREFIX} PEER_READ bf16 {hash_ids.device} rows {hash_ids.shape[0]} (PEER-SMALL) stock-lookup: "
              f"{'OK' if not any(diffs) else 'MISMATCH'} bytes-differ {diffs}", file=sys.stderr, flush=True)
        return
    exchanged = col_exchange._exchange(embed, hash_ids, forward_batch, engram=engram)
    if exchanged is None:
        raise RuntimeError("bundle-v9-check PEER_READ: v9's exchange declined a call PEER_READ served")
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


def _probe_ids(table: torch.Tensor, rows: int) -> torch.Tensor:
    """[rows, COLS] int64 hash ids, column c inside its owner's [lo, hi): row 0 hits lo and the last row hi - 1."""
    per = COLS // WORLD
    ids = torch.empty((rows, COLS), dtype=torch.int64)
    for c in range(COLS):
        lo, hi = int(table[c // per, 2]), int(table[c // per, 3])
        if hi <= lo:
            raise RuntimeError(f"owner {c // per} holds no rows ({lo}, {hi})")
        for i in range(rows):
            ids[i, c] = lo + ((i * (hi - lo - 1)) // max(rows - 1, 1) + (c % per) * 7919) % (hi - lo)
    return ids


def _map_prepare() -> str:
    """PEER-SMALL's mapping and probe at prepare (review R1). Returns "" with _TABLES set on every rank, or the cause
    with _TABLES untouched on every rank. Collectives: three all_gather_object calls on the TP cpu_group, each reached
    by every rank (a rank-local failure is carried as a message, never raised past a collective)."""
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    t0 = time.perf_counter()
    group = get_parallel().tp_group
    rank, world = int(group.rank_in_group), int(group.world_size)
    if world != WORLD:
        return f"a {world}-rank TP group, PEER_READ needs {WORLD}"
    per, rows = COLS // WORLD, int(PEER_SMALL_PROBE_ROWS)
    keys, local, ranges, err = [], None, None, ""
    try:
        if rows < 2:
            raise RuntimeError(f"PEER_SMALL_PROBE_ROWS {rows} < 2")
        keys, local = _LOCAL
        ranges = torch.zeros((len(keys), 2), dtype=torch.int64)
        for i, key in enumerate(keys):
            _, embed, owned = _EMBEDS[key]
            ranges[i, 0], ranges[i, 1] = embed.row_start, embed.row_start + embed.rows
            if tuple(owned[rank]) != (int(ranges[i, 0]), int(ranges[i, 1])):
                raise RuntimeError(f"rank {rank} holds rows {tuple(ranges[i].tolist())}, the layout gives "
                                   f"{owned[rank]}")
        module()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001  (carried to the vote)
        err = f"rank {rank} export: {type(error).__name__}: {str(error)[:200]}"
    meta = [None] * world
    dist.all_gather_object(meta, (err, local, ranges), group=group.cpu_group)
    bad = [m[0] for m in meta if m[0]]
    if bad:
        return " | ".join(bad)[:400]
    tables, probes, err = {}, {}, ""
    try:
        lib = module()
        for i, key in enumerate(keys):
            _, embed, owned = _EMBEDS[key]
            table = torch.zeros((WORLD, 4), dtype=torch.int64)
            for r in range(world):
                handles, spans = meta[r][1], meta[r][2]
                if tuple(spans.shape) != tuple(ranges.shape) or tuple(handles.shape) != tuple(local.shape):
                    raise RuntimeError(f"rank {r} exported {tuple(handles.shape)} handles / {tuple(spans.shape)} spans")
                lo, hi = int(spans[i, 0]), int(spans[i, 1])
                if (lo, hi) != tuple(owned[r]):
                    raise RuntimeError(f"rank {r} reports rows {(lo, hi)}, the layout gives {owned[r]}")
                for j, row_bytes in enumerate((DIM, DIM // BLOCK)):
                    blob = handles[i, j].contiguous()
                    offset, size = (int(v) for v in blob[64:80].view(torch.int64))
                    if offset + (hi - lo) * row_bytes > size:
                        raise RuntimeError(f"rank {r}'s allocation ({size} B from offset {offset}) does not hold its "
                                           f"{hi - lo}-row shard")
                    if r == rank:
                        address = (embed.weight if j == 0 else embed.scale).data_ptr()
                    else:
                        out = torch.zeros(1, dtype=torch.int64)
                        lib.open_handle(blob, out)
                        address = int(out[0])
                    table[r, j] = address
                table[r, 2], table[r, 3] = lo, hi
            tables[key] = table
            device = embed.weight.device
            ids = _probe_ids(table, rows).to(device)
            got = torch.full((rows, COLS * DIM), float("nan"), dtype=torch.bfloat16, device=device)
            lib.peer_lookup_bf16(ids, table, got, rows)
            ref = (embed._owned_rows(ids) + 0.0).view(rows, COLS, DIM)
            others = torch.ones(COLS, dtype=torch.bool, device=device)
            others[rank * per:(rank + 1) * per] = False
            if bool((ref[:, others].view(torch.int16) != 0).any()):
                raise RuntimeError(f"stock _owned_rows is non-zero outside rank {rank}'s columns")
            probes[i] = (got.view(rows, COLS, DIM).cpu(), ref[:, rank * per:(rank + 1) * per].contiguous().cpu())
    except Exception as error:  # noqa: BLE001  (carried to the vote)
        err = f"rank {rank} map / probe: {type(error).__name__}: {str(error)[:200]}"
    shared = [None] * world
    dist.all_gather_object(shared, (err, {i: own for i, (_, own) in probes.items()} if not err else None),
                           group=group.cpu_group)
    bad = [m[0] for m in shared if m[0]]
    if bad:
        return " | ".join(bad)[:400]
    err = ""
    try:
        for i in range(len(keys)):
            got = probes[i][0]
            for r in range(world):
                want = shared[r][1][i]
                if not torch.equal(got[:, r * per:(r + 1) * per].view(torch.int16), want.view(torch.int16)):
                    diff = int((got[:, r * per:(r + 1) * per].view(torch.int16) != want.view(torch.int16)).sum())
                    raise RuntimeError(f"Engram layer {i}: rank {rank}'s reads of rank {r}'s rows differ from its "
                                       f"stock _owned_rows in {diff} bf16 words")
    except Exception as error:  # noqa: BLE001  (carried to the vote)
        err = f"rank {rank} compare: {type(error).__name__}: {str(error)[:200]}"
    verdict = [None] * world
    dist.all_gather_object(verdict, err, group=group.cpu_group)
    bad = [v for v in verdict if v]
    if bad:
        return " | ".join(bad)[:400]
    _TABLES.update(tables)
    print(f"{PREFIX} PEER_READ mapped {len(keys)} Engram layers x {world - 1} peers in "
          f"{1e3 * (time.perf_counter() - t0):.1f} ms (at prepare, king-v6 PEER-SMALL; peer reads proven against "
          f"stock _owned_rows on {rows} rows x {len(keys)} layers)", file=sys.stderr, flush=True)
    return ""


def _peer_small(role: str) -> str:
    from bundle_v9 import col_exchange

    _BF16_MIN.clear()
    if not PEER_SMALL:
        return "off (switched off by peer_read.PEER_SMALL; bf16 PEER-READ from col_exchange.MIN_ROWS)"
    if role not in PEER_SMALL_ROLES:
        return f"off (role {role} is not in peer_read.PEER_SMALL_ROLES {PEER_SMALL_ROLES})"
    if not 1 <= PEER_SMALL_MIN_ROWS < col_exchange.MIN_ROWS:
        return f"off (FAIL, PEER_SMALL_MIN_ROWS {PEER_SMALL_MIN_ROWS} is outside [1, {col_exchange.MIN_ROWS}))"
    _BF16_MIN.append(PEER_SMALL_MIN_ROWS)
    return (f"ON (role {role}: bf16 PEER-READ Engram lookup from {PEER_SMALL_MIN_ROWS} DP-global rows of an "
            f"uncaptured sum-len forward with extend in batch; FP8 place and NCCL exchange keep "
            f"{col_exchange.MIN_ROWS})")


def prepare(model) -> None:
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        return
    if not _EMBEDS:
        raise RuntimeError("bundle-v9 PEER_READ: no Engram view registered (col_exchange.engram_view registers each)")
    export_local()
    print(f"{PREFIX} PEER_READ prepared: {len(_EMBEDS)} Engram layers exported", file=sys.stderr, flush=True)
    state = _peer_small(str(get_disagg().disaggregation_mode))
    if _BF16_MIN:  # the decision so far is global (constants and the engine role), so every rank is here or none is
        cause = _map_prepare()
        if cause:
            _BF16_MIN.clear()
            state = (f"off (FAIL, PEER-READ mapping / probe at prepare: {cause}; bf16 PEER-READ from "
                     f"col_exchange.MIN_ROWS, mapped lazily as the king does)")
        else:
            state = state[:-1] + f"; mapped at prepare, peer reads proven on {PEER_SMALL_PROBE_ROWS} rows a layer)"
    print(f"bundle-v9 king-v6 PEER-SMALL: {state} gpu{torch.cuda.current_device()}", file=sys.stderr, flush=True)
