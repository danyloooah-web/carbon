
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from bundle_v9 import rehome_transport as T

_FIELDS = ("streams", "input", "pre", "s_pre", "s_post", "s_comb")
_SWA_WINDOW = 128
_COMPRESS_RATIO = 2


class OwnerKV(NamedTuple):
    n_compressed: int
    n_swa: int
    c_off_min: int
    s_off_min: int


class HelperKV(NamedTuple):
    n_compressed: int
    n_swa: int
    c_off_min: int
    s_off_min: int
    c_off: int
    c_hi: int
    s_off: int
    s_hi: int
    rel0: int


def _to_int_list(vals) -> List[int]:
    if vals is None:
        return []
    if hasattr(vals, "tolist"):
        vals = vals.tolist()
    return [int(v) for v in vals]


def l0_kv_exchange(
    plan, rank: int, seq_lens_cpu, extend_lens_cpu
) -> Tuple[Optional[OwnerKV], Optional[HelperKV], bool]:
    group = T.link().group
    ws = int(group.world_size)
    cols = 5 + 5 * ws
    tbl = torch.zeros((ws, cols), dtype=torch.int64)
    out = tuple(sorted(plan.outgoing(rank), key=lambda m: m.lo))
    err: Optional[str] = None
    if out:
        try:
            seq_lens = _to_int_list(seq_lens_cpu)
            ext_lens = _to_int_list(extend_lens_cpu)
            if not seq_lens or len(seq_lens) != len(ext_lens) or sum(ext_lens) != plan.counts[rank]:
                raise RuntimeError(
                    f"rank {rank} seq/extend lens ({len(seq_lens)}/{len(ext_lens)}, sum={sum(ext_lens)}) "
                    f"disagree with plan count {plan.counts[rank]}"
                )
            q_start, q_end, c_base, c_len, s_base, s_len = [], [], [], [], [], []
            q_cur = c_cur = s_cur = 0
            for s, e in zip(seq_lens, ext_lens):
                if s < 0 or e < 0 or e > s:
                    raise RuntimeError(f"rank {rank} invalid (seq_len={s}, extend_len={e})")
                cl = s // _COMPRESS_RATIO
                sl = min(s, e + (_SWA_WINDOW - 1))
                q_start.append(q_cur)
                q_cur += e
                q_end.append(q_cur)
                c_base.append(c_cur)
                c_len.append(cl)
                c_cur += cl
                s_base.append(s_cur)
                s_len.append(sl)
                s_cur += sl
            n_compressed, n_swa = c_cur, s_cur
            kept = plan.kept(rank)
            c_offs, s_offs = [], []
            for mv in out:
                reqs = [r for r in range(len(ext_lens)) if ext_lens[r] > 0 and q_end[r] > mv.lo and q_start[r] < mv.hi]
                if not reqs:
                    raise RuntimeError(f"rank {rank} move {mv} intersects no extend request")
                r_lo, r_hi = reqs[0], reqs[-1]
                co, ch = c_base[r_lo], c_base[r_hi] + c_len[r_hi]
                so, sh = s_base[r_lo], s_base[r_hi] + s_len[r_hi]
                c_offs.append(co)
                s_offs.append(so)
                base_col = 5 + 5 * int(mv.helper)
                tbl[rank, base_col] = 1 + co
                tbl[rank, base_col + 1] = ch
                tbl[rank, base_col + 2] = so
                tbl[rank, base_col + 3] = sh
                tbl[rank, base_col + 4] = mv.lo - kept
            c_off_min = min(c_offs)
            s_off_min = min(s_offs)
            over = (n_compressed - c_off_min > T.KVC_CAP or n_swa - s_off_min > T.KVS_CAP
                    or n_compressed + n_swa > T.KVL_CAP)
            tbl[rank, 0] = 3 if over else 2
            tbl[rank, 1] = n_compressed
            tbl[rank, 2] = n_swa
            tbl[rank, 3] = c_off_min
            tbl[rank, 4] = s_off_min
        except BaseException as exc:
            tbl[rank].zero_()
            tbl[rank, 0] = 1
            err = f"rank {rank}: {type(exc).__name__}: {exc}"
    dist.all_reduce(tbl, op=dist.ReduceOp.SUM, group=group.cpu_group)
    bad_ranks = [r for r in range(ws) if int(tbl[r, 0].item()) == 1]
    if bad_ranks:
        detail = err if err is not None else f"remote rank(s) {bad_ranks} failed"
        raise RuntimeError(f"v9-rehome L0 KV exchange failed: {detail}")
    if any(int(tbl[r, 0].item()) == 3 for r in range(ws)):
        return None, None, True
    owner_kv = None
    if out:
        owner_kv = OwnerKV(
            int(tbl[rank, 1].item()),
            int(tbl[rank, 2].item()),
            int(tbl[rank, 3].item()),
            int(tbl[rank, 4].item()),
        )
    helper_kv = None
    inc = plan.incoming(rank)
    if inc:
        (mv,) = inc
        ow = int(mv.owner)
        base_col = 5 + 5 * rank
        co1 = int(tbl[ow, base_col].item())
        if int(tbl[ow, 0].item()) != 2 or co1 <= 0:
            raise RuntimeError(f"v9-rehome L0 KV exchange: helper {rank} missing slice from owner {ow}")
        helper_kv = HelperKV(
            int(tbl[ow, 1].item()),
            int(tbl[ow, 2].item()),
            int(tbl[ow, 3].item()),
            int(tbl[ow, 4].item()),
            co1 - 1,
            int(tbl[ow, base_col + 1].item()),
            int(tbl[ow, base_col + 2].item()),
            int(tbl[ow, base_col + 3].item()),
            int(tbl[ow, base_col + 4].item()),
        )
    return owner_kv, helper_kv, False


def _tail_spans(extend_lens: Sequence[int], tail_lens: Sequence[int]) -> List[Tuple[int, int]]:
    spans, end = [], 0
    for e, t in zip(extend_lens, tail_lens):
        end += int(e)
        spans.append((end - int(t), end))
    return spans


def _count(spans, lo: int, hi: int) -> int:
    return sum(max(0, min(b, hi) - max(a, lo)) for a, b in spans)


def _slot(plan, move) -> int:
    return sorted(m.lo for m in plan.outgoing(move.owner)).index(move.lo)


def _bound(plan, move) -> int:
    return min(move.hi - move.lo, plan.late[move.owner])


def l0_send(plan, rank: int, input_ids: torch.Tensor, extend_lens: Sequence[int], tail_lens: Sequence[int]) -> None:
    from sglang.srt.utils.common import async_h2d

    spans = _tail_spans(extend_lens, tail_lens)
    if spans[-1][1] != plan.counts[rank] or sum(b - a for a, b in spans) != plan.late[rank]:
        raise RuntimeError(f"v9-rehome L0: rank {rank} extend/tail lengths disagree with the plan "
                           f"({spans[-1][1]} rows, {sum(b - a for a, b in spans)} tail rows vs {plan.counts[rank]}, "
                           f"{plan.late[rank]})")
    for mv in plan.outgoing(rank):
        T.push(mv.helper, "ids", input_ids[mv.lo:mv.hi].to(torch.int64).contiguous())
        k = _bound(plan, mv)
        if k:
            rel = [r - mv.lo for a, b in spans for r in range(max(a, mv.lo), min(b, mv.hi))]
            rel += [0] * (k - len(rel))
            T.push(mv.helper, "tidx", async_h2d(rel, dtype=torch.int64, device=input_ids.device))
        T.signal(mv.helper, T.IDS)


def l0_receive(plan, rank: int, model) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.layernorm.mhc import hc_broadcast

    (mv,) = plan.incoming(rank)
    T.wait(mv.owner, T.IDS)
    ids = T.view("ids", mv.hi - mv.lo)
    return ids, hc_broadcast(model.embed_tokens(ids), model.hc_mult)


def _fields(state) -> dict:
    from sglang.srt.models.deepseek_v4_mhc import HcNormed

    out = {"streams": state.residual}
    if isinstance(state.input, HcNormed):
        out["input"] = state.input.rows
    if state.pre is not None:
        out["pre"] = state.pre
    if state.stats is not None:
        out["s_pre"], out["s_post"], out["s_comb"] = state.stats
    return out


def l20_return(plan, rank: int, state, attn_hc) -> None:
    from sglang.srt.models import deepseek_v4_mhc as mhc

    (mv,) = plan.incoming(rank)
    T.push(mv.owner, "x", mhc.combine(attn_hc, state).contiguous(), row0=mv.lo)
    k = _bound(plan, mv)
    if k:
        idx = T.view("tidx", k)
        slot = _slot(plan, mv)
        for name, t in _fields(state).items():
            T.push(mv.owner, f"ts{slot}_{name}", t.index_select(0, idx).contiguous())
    T.signal(mv.owner, T.RET)


def l20_compose(plan, rank: int, state, attn_hc, tail, extend_lens: Sequence[int]):
    from sglang.srt.models import deepseek_v4_mhc as mhc

    kept, n = plan.kept(rank), plan.counts[rank]
    T.view("x", kept).copy_(mhc.combine(attn_hc, state))
    out = sorted(plan.outgoing(rank), key=lambda m: m.lo)
    for mv in out:
        T.wait(mv.helper, T.RET)
    spans = _tail_spans(extend_lens, tail.extend_seq_lens_cpu)
    k_pre = _count(spans, 0, kept)
    if tail.contiguous_start is not None:
        start = tail.contiguous_start
        part = state.take_rows(lambda t: t[start:start + k_pre])
    else:
        idx = tail.token_indices[:k_pre]
        part = state.take_rows(lambda t: t[idx])
    pieces = _fields(part)
    for slot, mv in enumerate(out):
        k = _count(spans, mv.lo, mv.hi)
        for name in pieces:
            moved = T.view(f"ts{slot}_{name}", k).view(k, *pieces[name].shape[1:])
            pieces[name] = torch.cat([pieces[name], moved])
    return T.view("x", n), _rebuild(part, pieces)


def _rebuild(part, pieces: dict):
    from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcState

    stats = None
    if "s_pre" in pieces:
        stats = (pieces["s_pre"], pieces["s_post"], pieces["s_comb"])
    return HcState(pieces["streams"], pieces.get("pre"), HcNormed(pieces["input"]) if "input" in pieces else None,
                   stats)
