
from __future__ import annotations

import logging
import sys
import time
from types import MethodType

import torch
import torch.distributed as dist

from bundle_v9 import rehome_plan, row_exchange, tactic_pins

logger = logging.getLogger(__name__)

PEER_ROLES = ("decode",)
CHECK_ROWS = 512
_TABLES: dict = {}

# PEER-RH (king_v5 lever C). On REHOME-planned steps each rank reads only its post-plan Engram rows from the
# owning shards (the image's engram_gather_peer over PEER_READ's IPC mappings) instead of row_exchange.rehome_lookup's
# dense gather + NCCL all-to-all (or all-reduce) + 4-way sum. Fail-closed: prepare_rh proves bitwise equality against
# the king's path on the real tables and attaches the tables only if every rank passes; otherwise the king's path runs.
RH_ON = True
RH_ROLES = ("prefill", "null")
RH_PLANS = (
    ((16384, 0, 0, 0), ((0, 1, 8192, 16384, 0),)),
    ((16384, 2112, 4500, 65), ((0, 3, 8192, 12288, 65), (0, 1, 12288, 16384, 2112))),
    ((12288, 9000, 4096, 129), ((0, 3, 8192, 12288, 129), (1, 2, 4904, 9000, 4096))),
    ((16384, 4096, 2048, 1), ((0, 3, 4096, 8192, 1), (0, 2, 8192, 12288, 2048), (0, 1, 12288, 16384, 4096))),
)
RH_SEED = 20261106
RH_SCALE_OK = (10, 240)
RH_SCAN_ROWS = 1 << 18
RH_MEM_MARGIN = 1 << 30
RH_CHECK = 0
# Source pins of the king-side code the all-ids exactness argument also rests on (the dense gather wrapper and kernel
# behind EngramEmbedding._owned_rows, and the owner all-reduce), on top of tactic_pins.verify_egp. Checked inside
# prepare_rh's first try, so a mismatch turns PEER-RH off on every rank; it never raises.
RH_PINS = (
    ("engram", "EngramEmbedding._reduce_owned_rows", "68b957523c4c6466"),
    ("egk", "engram_gather", "9488014cf90aa632"),
    ("egk", "_engram_gather_kernel", "50eef55d019f2c0f"),
)
_RH = {"state": "not prepared", "calls": 0}


def _open_peer(handle, device_index: int):
    _, h, size, offset, rc_handle, rc_offset, ev_handle, _ = handle
    return torch.UntypedStorage._new_shared_cuda(
        device_index, h, size, offset, rc_handle, rc_offset, ev_handle, False
    )


def _check_static_domain(lid: int, engram) -> int:
    from sglang.srt.layers import engram as E
    from sglang.srt.runtime_context import get_parallel

    if type(engram) is not E.Engram or type(engram.embed) is not E.EngramEmbedding:
        raise TypeError(
            f"bundle-v9 PEER L{lid}: expected stock Engram / EngramEmbedding, got "
            f"{type(engram).__name__} / {type(engram.embed).__name__}"
        )
    embed = engram.embed
    par = get_parallel()
    attn_cp = int(getattr(par, "attn_cp_size", 1))
    wkv_w = engram.wkv.weight
    ok = (
        embed.peer is None
        and not embed._shared
        and embed.host_table is None
        and not embed.staged_by_scheduler
        and (int(par.tp_size), int(embed.tp_size), int(par.attn_dp_size), int(par.attn_tp_size), attn_cp)
        == (4, 4, 4, 1, 1)
        and int(par.tp_rank) == int(par.attn_dp_rank)
        and embed.weight.dtype == torch.float8_e4m3fn
        and embed.weight.is_contiguous()
        and embed.scale.is_contiguous()
        and int(embed.dim) > 0
        and int(embed.dim) % int(E.FP8_BLOCK_SIZE) == 0
        and int(wkv_w.shape[1]) > 0
        and int(wkv_w.shape[1]) % int(embed.dim) == 0
    )
    if not ok:
        raise RuntimeError(f"bundle-v9 PEER L{lid}: unsupported Engram / DP domain")
    return int(wkv_w.shape[1]) // int(embed.dim)


def peer_lookup(embed, table: dict, indices: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.embeddings import engram_gather as EGK
    from sglang.srt.layers import engram as E

    if indices.numel() == 0:
        return torch.empty(*indices.shape, embed.dim, dtype=torch.bfloat16, device=indices.device)
    flat = indices if indices.stride(-1) == 1 else indices.contiguous()
    out = torch.empty(*flat.shape, embed.dim, dtype=torch.bfloat16, device=indices.device)
    EGK.engram_gather_peer(flat, out, table["w_bias"], table["s_bias"], table["starts"], embed.dim, E.FP8_BLOCK_SIZE)
    return out


def _bits_sums(t: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> list[int]:
    b = t.view(torch.int16).to(torch.int64) * mask
    return [int(b.sum().item()), int((b * weight).sum().item()), int((b * b).sum().item())]


@torch.no_grad()
def _connect_and_verify(lid: int, engram, cols: int) -> dict:
    from sglang.srt.layers import engram as E
    from sglang.srt.runtime_context import get_parallel

    embed = engram.embed
    group = get_parallel().tp_group
    rank, world = int(group.rank_in_group), int(group.world_size)
    device = embed.weight.device

    mine, exp_err = None, ""
    try:
        mine = (
            int(embed.row_start),
            int(embed.rows),
            embed.weight.untyped_storage()._share_cuda_(),
            int(embed.weight.storage_offset()),
            embed.scale.untyped_storage()._share_cuda_(),
            int(embed.scale.storage_offset()),
        )
    except Exception as exc:
        exp_err = f"rank {rank} export: {type(exc).__name__}: {exc}"

    meta = [None] * world
    dist.all_gather_object(meta, (mine, exp_err), group=group.cpu_group)
    exp_failures = [err or f"rank {r} missing export" for r, (m, err) in enumerate(meta) if m is None]
    if exp_failures:
        raise RuntimeError(f"bundle-v9 PEER L{lid} IPC export failed: {' | '.join(exp_failures)}")

    table = None
    map_err = ""
    keep: list = []
    try:
        shards = [m for m, _ in meta]
        n_scales = int(embed.dim) // int(E.FP8_BLOCK_SIZE)
        starts, w_bias, s_bias = [], [], []
        for r, m in enumerate(shards):
            if r != rank and not torch.cuda.can_device_access_peer(device, m[2][0]):
                raise RuntimeError(f"no NVLink peer access from cuda:{device.index} to cuda:{m[2][0]}")
        for r, (start, _rows, wh, w_off, sh, s_off) in enumerate(shards):
            if r == rank:
                wp, sp = int(embed.weight.data_ptr()), int(embed.scale.data_ptr())
            else:
                ws, ss = _open_peer(wh, device.index), _open_peer(sh, device.index)
                keep.extend((ws, ss))
                wp, sp = int(ws.data_ptr()) + w_off, int(ss.data_ptr()) + s_off
            starts.append(start)
            w_bias.append(wp - start * int(embed.dim))
            s_bias.append(sp - start * n_scales)
        if starts[0] != 0 or starts != sorted(starts):
            raise RuntimeError(f"unordered shard starts {starts}")
        padded = 1 << max(0, (world - 1).bit_length())
        for _ in range(world, padded):
            starts.append(torch.iinfo(torch.int64).max)
            w_bias.append(0)
            s_bias.append(0)
        total = shards[-1][0] + shards[-1][1]
        table = {
            "w_bias": torch.tensor(w_bias, dtype=torch.int64, device=device),
            "s_bias": torch.tensor(s_bias, dtype=torch.int64, device=device),
            "starts": torch.tensor(starts, dtype=torch.int64, device=device),
            "keep": keep,
            "total": total,
            "meta": [(m[0], m[1]) for m in shards],
        }
    except Exception as exc:
        map_err = f"rank {rank} map: {type(exc).__name__}: {exc}"

    probe = None
    if table is not None:
        try:
            gen = torch.Generator(device=device).manual_seed(20261027 + int(engram.layer_hash_index))
            ids = torch.randint(0, table["total"], (CHECK_ROWS, cols), device=device, generator=gen, dtype=torch.int64)
            edges = []
            for start, rows in table["meta"]:
                edges.extend((start, start + rows - 1))
            edge_t = torch.tensor(edges, dtype=torch.int64, device=device)
            k_edge = min(cols, int(edge_t.numel()))
            ids[0, :k_edge] = edge_t[:k_edge]
            starts_t = torch.tensor([m[0] for m in table["meta"]], dtype=torch.int64, device=device)
            owner = (torch.searchsorted(starts_t, ids, right=True) - 1).unsqueeze(-1)
            got = peer_lookup(embed, table, ids)
            ref = embed._owned_rows(ids).contiguous() + 0.0
            weight = (torch.arange(got.numel(), device=device, dtype=torch.int64) % 1000003 + 1).view(got.shape)
            probe = {
                "got": [_bits_sums(got, owner == r, weight) for r in range(world)],
                "ref": _bits_sums(ref, owner == rank, weight),
                "unowned_zero": bool(((ref.view(torch.int16) != 0) & (owner != rank)).sum().item() == 0),
            }
            del ids, got, ref, weight, owner
        except Exception as exc:
            map_err = f"rank {rank} probe: {type(exc).__name__}: {exc}"
            probe = None

    allv = [None] * world
    dist.all_gather_object(allv, (probe, map_err), group=group.cpu_group)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bad = []
    for r, (p, err) in enumerate(allv):
        if p is None:
            bad.append(err or f"rank {r} probe missing")
        elif not p["unowned_zero"]:
            bad.append(f"rank {r} unowned shard rows non-zero")
    if not bad:
        for r in range(world):
            if allv[rank][0]["got"][r] != allv[r][0]["ref"]:
                bad.append(f"rank {rank} vs owner {r} bit-fingerprint mismatch")
    if bad:
        raise RuntimeError(f"bundle-v9 PEER L{lid} STRICT peer byte-probe failed: {' | '.join(bad)}")
    return table


def rehome_spans_for_rank(plan, rank: int) -> list[tuple[int, int]]:
    starts = [sum(plan.counts[:r]) for r in range(len(plan.counts))]
    spans = [(starts[rank], starts[rank] + plan.kept(rank))]
    at = plan.kept(rank)
    for mv in sorted(plan.incoming(rank), key=lambda mv: mv.at):
        if mv.at != at:
            raise RuntimeError(f"bundle-v9 PEER REHOME: rank {rank} receives rows at {mv.at}, expected {at}")
        spans.append((starts[mv.owner] + mv.lo, starts[mv.owner] + mv.hi))
        at += mv.hi - mv.lo
    return spans


def _verify_rehome_spans_contract() -> None:
    for counts in ((32768, 4096, 0, 0), (16384, 16384, 0, 0)):
        p = rehome_plan.plan(counts, (1, 1, 0, 0))
        if p is None or [span for r in range(4) for span in rehome_spans_for_rank(p, r)] != row_exchange.rehome_spans(p):
            raise RuntimeError(f"bundle-v9 PEER: rehome_spans_for_rank disagrees with row_exchange.rehome_spans on {counts}")


def prepare(module, role: str) -> bool:
    _TABLES.clear()
    if role not in PEER_ROLES:
        return False
    tactic_pins.verify_egp()
    _verify_rehome_spans_contract()
    engrams = [(i, layer.engram) for i, layer in enumerate(module.layers) if layer.engram is not None]
    if not engrams:
        raise RuntimeError("bundle-v9 PEER: model carries no Engram layers")
    for lid, engram in engrams:
        cols = _check_static_domain(lid, engram)
        _TABLES[id(engram.embed)] = _connect_and_verify(lid, engram, cols)
    return True


def dp_sharded_lookup(self, indices: torch.Tensor, forward_batch):
    out = peer_lookup(self, self._egp_table, indices)
    if (
        forward_batch is not None
        and not torch.cuda.is_current_stream_capturing()
        and forward_batch.global_num_tokens_cpu is not None
    ):
        from sglang.srt.runtime_context import get_parallel

        mine = int(forward_batch.global_num_tokens_cpu[get_parallel().attn_dp_rank])
        if 0 <= mine < int(indices.shape[0]):
            out[mine:].zero_()
    return out


def engram_view(engram, view_fn):
    table = _TABLES.get(id(engram.embed))
    if table is None:
        raise RuntimeError("bundle-v9 PEER: engram_view requested before prepare populated _TABLES")
    engram_v = view_fn(engram)
    embed_v = view_fn(engram.embed)
    embed_v._egp_table = table
    embed_v._dp_sharded_lookup = MethodType(dp_sharded_lookup, embed_v)
    engram_v.embed = embed_v
    return engram_v


def rehome_lookup(embed, table: dict, indices: torch.Tensor, forward_batch, plan, rank: int) -> torch.Tensor:
    from sglang.srt.layers.dp_attention import dp_gather_replicate, get_global_dp_buffer_len

    rows = get_global_dp_buffer_len()
    counts = row_exchange._a2a_shared(forward_batch, rows)
    if counts is None or tuple(counts) != tuple(plan.counts) or int(indices.shape[0]) != plan.counts[rank]:
        raise RuntimeError(
            f"bundle-v9 PEER REHOME: shared counts {counts} / local rows {int(indices.shape[0])} "
            f"do not match plan {plan.counts} on rank {rank}"
        )
    ids_global = torch.empty((rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device)
    dp_gather_replicate(ids_global, indices.clone(), forward_batch)
    ids_mine = torch.cat([ids_global[a:b] for a, b in rehome_spans_for_rank(plan, rank)], dim=0)
    return peer_lookup(embed, table, ids_mine)


def _rh_log(on: bool, detail: str) -> str:
    from sglang.srt.runtime_context import get_parallel

    _RH["state"] = f"{'ON' if on else 'off'} ({detail})"
    if int(get_parallel().tp_group.rank_in_group) == 0:
        logger.warning("[v5] bundle-v9 PEER-RH: %s", _RH["state"])
    return _RH["state"]


def _rh_round(group, ok: bool, msg: str, extra=None):
    got = [None] * int(group.world_size)
    dist.all_gather_object(got, (bool(ok), str(msg), extra), group=group.cpu_group)
    return [m or f"rank {r} failed" for r, (k, m, _) in enumerate(got) if not k], [e for _, _, e in got]


def _rh_pins() -> None:
    import sglang.kernels.ops.embeddings.engram_gather as egk
    import sglang.srt.layers.engram as engram

    tactic_pins._verify_segments("PEER-RH", {"engram": engram, "egk": egk}, RH_PINS)


def _rh_plans() -> list:
    plans, branches = [], set()
    for counts, moves in RH_PLANS:
        counts = tuple(int(c) for c in counts)
        mv = tuple(rehome_plan.Move(*(int(v) for v in m)) for m in moves)
        if len(counts) != rehome_plan.RANKS or not rehome_plan._valid(counts, mv):
            raise RuntimeError(f"proof plan {counts} {moves} is not a valid REHOME plan")
        plan = rehome_plan.Plan(counts, (0,) * rehome_plan.RANKS, mv, 1)
        spans = [s for r in range(rehome_plan.RANKS) for s in rehome_spans_for_rank(plan, r)]
        if spans != row_exchange.rehome_spans(plan):
            raise RuntimeError(f"rehome_spans_for_rank disagrees with row_exchange.rehome_spans on {counts}")
        tl = [plan.rows(r) for r in range(rehome_plan.RANKS)]
        branches.add(2 * max(tl) >= sum(counts))
        plans.append(plan)
    if branches != {False, True}:
        raise RuntimeError("proof plans do not cover both rehome_lookup branches (all-reduce and all-to-all)")
    return plans


def _rh_need(plan, rank: int, row_bytes: int, cols: int) -> int:
    rows = sum(plan.counts)
    tl = [plan.rows(r) for r in range(rehome_plan.RANKS)]
    reduce_rows = rows if 2 * max(tl) >= rows else rehome_plan.RANKS * tl[rank]
    return (rows + reduce_rows + 2 * tl[rank]) * row_bytes + 3 * rows * cols * 8


def _rh_scan(embed) -> int:
    # Blocks whose dequant could leave the exact domain: an e4m3 NaN byte, an e8m0 255 byte, or a non-zero weight
    # under a scale byte outside RH_SCALE_OK. With none, every looked-up value is +-0 or a normal fp32 >= 2**-126
    # (finite), so x + 0 + 0 + 0 == x and -0 + 0 == +0 in every reduction order: both paths give the same bits.
    from sglang.srt.layers import engram as E

    blk = int(E.FP8_BLOCK_SIZE)
    w, s = embed.weight.view(torch.uint8), embed.scale.view(torch.uint8)
    if w.dim() != 2 or s.dim() != 2 or int(s.shape[0]) != int(w.shape[0]) or int(w.shape[1]) != int(s.shape[1]) * blk:
        raise RuntimeError(f"shard shapes {tuple(w.shape)} / {tuple(s.shape)} are not [rows, {blk} x blocks]")
    rows, nblk = int(w.shape[0]), int(s.shape[1])
    lo, hi = RH_SCALE_OK
    bad = torch.zeros((), dtype=torch.int64, device=w.device)
    for r0 in range(0, rows, RH_SCAN_ROWS):
        mag = w[r0 : r0 + RH_SCAN_ROWS].reshape(-1, nblk, blk) & 0x7F
        sc = s[r0 : r0 + RH_SCAN_ROWS]
        nonzero = (mag != 0).any(dim=-1)
        nan = (mag == 0x7F).any(dim=-1)
        bad += (nan | (sc == 0xFF) | (nonzero & ((sc < lo) | (sc > hi)))).sum()
        del mag, sc, nonzero, nan
    return int(bad.item())


def _rh_ids(bounds, rows: int, seed: int, meta) -> torch.Tensor:
    cols = len(bounds) - 1
    gen = torch.Generator().manual_seed(seed)
    ids = torch.empty((rows, cols), dtype=torch.int64)
    for c in range(cols):
        ids[:, c] = torch.randint(bounds[c], bounds[c + 1], (rows,), generator=gen, dtype=torch.int64)
    mixed = ids[2::5]
    mixed.copy_(torch.randint(0, bounds[-1], tuple(mixed.shape), generator=gen, dtype=torch.int64))
    edges = torch.tensor([v for lo, hi in meta for v in (lo, lo + 1, hi - 2, hi - 1)], dtype=torch.int64)
    ids.view(-1)[: edges.numel()] = edges
    return ids


def _rh_table(embed, entry, rank: int, bounds, cols: int) -> dict:
    from sglang.srt.layers import engram as E

    rows = [[int(v) for v in r] for r in entry.tolist()]
    dim = int(embed.dim)
    n_scales = dim // int(E.FP8_BLOCK_SIZE)
    lo, hi = [r[2] for r in rows], [r[3] for r in rows]
    if (
        len(rows) != 4
        or lo[0] != 0
        or hi[-1] != bounds[-1]
        or any(a >= b for a, b in zip(lo, hi))
        or any(hi[r] != lo[r + 1] for r in range(len(rows) - 1))
    ):
        raise RuntimeError(f"PEER_READ spans {list(zip(lo, hi))} do not tile the table [0, {bounds[-1]})")
    if (
        (lo[rank], hi[rank]) != (int(embed.row_start), int(embed.row_start) + int(embed.rows))
        or rows[rank][0] != int(embed.weight.data_ptr())
        or rows[rank][1] != int(embed.scale.data_ptr())
    ):
        raise RuntimeError("PEER_READ's own-rank entry is not this rank's shard")
    device = embed.weight.device
    return {
        "w_bias": torch.tensor([r[0] - r[2] * dim for r in rows], dtype=torch.int64, device=device),
        "s_bias": torch.tensor([r[1] - r[2] * n_scales for r in rows], dtype=torch.int64, device=device),
        "starts": torch.tensor(lo, dtype=torch.int64, device=device),
        "cols": int(cols),
        "meta": list(zip(lo, hi)),
    }


def _rh_values(embed, table: dict, ids_global: torch.Tensor, plan, rank: int) -> torch.Tensor:
    ids_mine = torch.cat([ids_global[a:b] for a, b in rehome_spans_for_rank(plan, rank)], dim=0)
    return peer_lookup(embed, table, ids_mine)


@torch.no_grad()
def prepare_rh(view, module, role: str, rehome_on: bool) -> str:
    _RH["calls"] = 0
    if not RH_ON:
        return _rh_log(False, "switch RH_ON is False")
    if role not in RH_ROLES:
        return _rh_log(False, f"role {role!r} is not in {RH_ROLES}")
    if not rehome_on:
        return _rh_log(False, "REHOME is off")
    from sglang.srt.runtime_context import get_parallel

    from bundle_v9 import peer_read

    t0 = time.perf_counter()
    group = get_parallel().tp_group
    rank = int(group.rank_in_group)
    layers, plans, err, fresh, scanned, device = [], [], "", None, 0, None
    try:
        tactic_pins.verify_egp()
        _rh_pins()
        plans = _rh_plans()
        for lid, layer_v in enumerate(view.layers):
            engram_v = getattr(layer_v, "engram", None)
            if engram_v is None:
                continue
            cols = _check_static_domain(lid, module.layers[lid].engram)
            embed_v = engram_v.embed
            bounds = tuple(int(b) for b in module.engram_layout.column_bounds(engram_v.layer_hash_index))
            reg = peer_read._EMBEDS.get(id(embed_v))
            if (
                reg is None
                or reg[1] is not embed_v
                or len(bounds) - 1 != cols
                or tuple(reg[2][rank]) != (int(embed_v.row_start), int(embed_v.row_start) + int(embed_v.rows))
            ):
                raise RuntimeError(f"L{lid}: the Engram view is not PEER_READ's registered shard")
            layers.append((lid, embed_v, bounds, cols))
        if not layers:
            raise RuntimeError("the model carries no Engram layers")
        device = layers[0][1].weight.device
        with torch.cuda.device(device):
            for lid, embed_v, _, _ in layers:
                n = _rh_scan(embed_v)
                if n:
                    raise RuntimeError(f"L{lid}: {n} shard blocks outside the exact-dequant domain")
                scanned += int(embed_v.rows)
            cols0 = layers[0][3]
            need = max(_rh_need(p, rank, cols0 * int(layers[0][1].dim) * 2, cols0) for p in plans) + RH_MEM_MARGIN
            free = (
                torch.cuda.mem_get_info(device)[0]
                + torch.cuda.memory_reserved(device)
                - torch.cuda.memory_allocated(device)
            )
            if free < need:
                raise RuntimeError(f"{free >> 20} MiB free < {need >> 20} MiB the proof needs")
        fresh = all(id(embed_v) not in peer_read._TABLES for _, embed_v, _, _ in layers)
    except Exception as exc:
        err = f"rank {rank} checks: {type(exc).__name__}: {exc}"
    bad, fresh_all = _rh_round(group, not err, err, fresh)
    if bad:
        return _rh_log(False, " | ".join(bad))
    if len(set(fresh_all)) != 1:
        return _rh_log(False, f"PEER_READ mapping state differs across ranks {fresh_all}")

    tables, err = [], ""
    try:
        with torch.cuda.device(device):
            if fresh_all[0]:
                peer_read._map_all()
            tables = [_rh_table(e, peer_read._TABLES[id(e)], rank, b, c) for _, e, b, c in layers]
    except Exception as exc:
        err = f"rank {rank} map: {type(exc).__name__}: {exc}"
    bad, mapped = _rh_round(group, not err, err, all(id(e) in peer_read._TABLES for _, e, _, _ in layers))
    if bad:
        if fresh_all[0] and not all(mapped):
            for _, embed_v, _, _ in layers:
                peer_read._TABLES.pop(id(embed_v), None)
        return _rh_log(False, " | ".join(bad))

    failures, checks = [], 0
    with torch.cuda.device(device):
        for (lid, embed_v, bounds, cols), table in zip(layers, tables):
            for k, plan in enumerate(plans):
                ids_g = values = tl = ref = got = None
                err, csum = "", None
                try:
                    ids_cpu = _rh_ids(bounds, sum(plan.counts), RH_SEED + 97 * lid + k, table["meta"])
                    csum = int(ids_cpu.sum())
                    ids_g = ids_cpu.to(device)
                    values, tl = row_exchange.rehome_gather(embed_v, ids_g, plan)
                except Exception as exc:
                    err = f"rank {rank} L{lid} plan {k} gather: {type(exc).__name__}: {exc}"
                bad, sums = _rh_round(group, not err, err, csum)
                if not bad and len(set(sums)) != 1:
                    bad = [f"L{lid} plan {k}: the proof ids differ across ranks"]
                if bad:
                    del ids_g, values
                    torch.cuda.empty_cache()
                    return _rh_log(False, " | ".join(bad))
                ref = row_exchange.rehome_reduce(embed_v, values, tl, rank)
                try:
                    ref = ref.view(tl[rank], cols, int(embed_v.dim))
                    got = _rh_values(embed_v, table, ids_g, plan, rank)
                    if tuple(got.shape) != tuple(ref.shape) or got.dtype != ref.dtype:
                        failures.append(
                            f"rank {rank} L{lid} plan {k}: {tuple(got.shape)} {got.dtype} vs {tuple(ref.shape)} {ref.dtype}"
                        )
                    else:
                        n = int((got.view(torch.int16) != ref.view(torch.int16)).sum().item())
                        if n:
                            failures.append(f"rank {rank} L{lid} plan {k} rows {tl[rank]}: {n} bf16 elements differ")
                        checks += int(got.numel())
                except Exception as exc:
                    failures.append(f"rank {rank} L{lid} plan {k} peer: {type(exc).__name__}: {exc}")
                del ids_g, values, ref, got
        torch.cuda.synchronize(device)
    bad, _ = _rh_round(group, not failures, " | ".join(failures))
    torch.cuda.empty_cache()
    if bad:
        return _rh_log(False, "proof mismatch, king path kept: " + " | ".join(bad))
    for (_, embed_v, _, _), table in zip(layers, tables):
        embed_v._v5_rh = table
    return _rh_log(
        True,
        f"role={role}, {len(layers)} Engram layers x {len(plans)} REHOME plans bitwise = king rehome_lookup "
        f"(all-reduce + all-to-all branches, {checks} bf16/rank), shard scan clean ({scanned} rows), "
        f"{time.perf_counter() - t0:.2f}s",
    )


def rh_lookup(embed, indices: torch.Tensor, forward_batch, plan, rank: int):
    table = getattr(embed, "_v5_rh", None)
    if (
        table is None
        or indices.dim() != 2
        or indices.dtype != torch.int64
        or int(indices.shape[1]) != table["cols"]
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    from sglang.srt.layers.dp_attention import dp_gather_replicate, get_global_dp_buffer_len

    rows = get_global_dp_buffer_len()
    counts = row_exchange._a2a_shared(forward_batch, rows)
    if counts is None or tuple(counts) != tuple(plan.counts) or int(indices.shape[0]) != plan.counts[rank]:
        return None
    ids_global = torch.empty((rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device)
    dp_gather_replicate(ids_global, indices.clone(), forward_batch)
    got = _rh_values(embed, table, ids_global, plan, rank)
    if _RH["calls"] < RH_CHECK:
        _RH["calls"] += 1
        values, tl = row_exchange.rehome_gather(embed, ids_global, plan)
        ref = row_exchange.rehome_reduce(embed, values, tl, rank).view(got.shape)
        n = int((ref.view(torch.int16) != got.view(torch.int16)).sum().item())
        print(
            f"bundle-v9-check PEER-RH {indices.device} plan {tuple(plan.counts)} rows {tl[rank]}: "
            f"{'OK' if n == 0 else 'MISMATCH'} elements-differ {n}",
            file=sys.stderr,
            flush=True,
        )
        return ref
    return got
