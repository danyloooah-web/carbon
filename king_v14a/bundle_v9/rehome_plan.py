
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

RANKS = 4
CAP = 32768
HELP = 16384
MAX_FAN = 3
MIN_BLOCK = 4096
ALIGN = 128
COARSE = 8 * ALIGN
TS_ROWS = 2048
LAYERS = 20
SRC_LAYERS = 3

HOME_NS, MOVE_NS = 1200, 4700
OWNER_ROW_NS, HELPER_ROW_NS = 35, 95
FIXED_NS, FAN_NS, FAN3_NS = 600_000, 300_000, 650_000
PAIR_PHI = 55
FAN_PHI = 70
FAN3_PHI = 75
SCALE = (566, 590)
SPLIT_IN = (109, 100)


class Move(NamedTuple):
    owner: int
    helper: int
    lo: int
    hi: int
    at: int


class Plan(NamedTuple):
    counts: Tuple[int, ...]
    late: Tuple[int, ...]
    moves: Tuple[Move, ...]
    saving_ns: int

    def outgoing(self, rank: int) -> Tuple[Move, ...]:
        return tuple(m for m in self.moves if m.owner == rank)

    def incoming(self, rank: int) -> Tuple[Move, ...]:
        return tuple(m for m in self.moves if m.helper == rank)

    def kept(self, rank: int) -> int:
        out = self.outgoing(rank)
        return min(m.lo for m in out) if out else self.counts[rank]

    def rows(self, rank: int) -> int:
        return self.kept(rank) + sum(m.hi - m.lo for m in self.incoming(rank))


def _split_ns(rows: int) -> int:
    if rows <= 0 or rows >= 32768:
        return 0
    if rows <= 9830:
        if rows <= 4096:
            return 66200 * rows // 4096
        if rows <= 8192:
            return 66200 + (-2800) * (rows - 4096) // 4096
        return 63400 + (-4200) * (rows - 8192) // 1638
    if rows <= 16384:
        if rows <= 12288:
            return 59200 + (-6400) * (rows - 9830) // 2458
        return 52800 + (-26600) * (rows - 12288) // 4096
    if rows <= 24576:
        return 26200 + (-11000) * (rows - 16384) // 8192
    return 15200 + (-15200) * (rows - 24576) // 8192


def _split_step_ns(rows: int) -> int:
    return LAYERS * SPLIT_IN[0] * _split_ns(rows) // SPLIT_IN[1]


def _xq_overrun_step_ns(kept: int, blocks: Sequence[int]) -> int:
    t_owner_pre = 65_000 + 825 * kept // 10
    if len(blocks) == 1:
        m0 = blocks[0]
        t_xq = 95_000 + 318 * m0 // 10 + 208 * m0 // 10 + 10_000
    else:
        m_tot = 0
        t_single = 0
        min_prod = 1 << 60
        for m in blocks:
            p = 95_000 + 318 * m // 10
            if p < min_prod:
                min_prod = p
            ts = p + 208 * m // 10
            if ts > t_single:
                t_single = ts
            m_tot += m
        t_agg = min_prod + 165 * m_tot // 10 + 25_000 * (len(blocks) - 1)
        t_xq = (t_single if t_single >= t_agg else t_agg) + 12_000
    over = t_xq - t_owner_pre if t_xq > t_owner_pre else 0
    return SRC_LAYERS * over


def _owner_ns(nh: int, blocks: Sequence[int]) -> int:
    m_tot = sum(blocks)
    kept = nh - m_tot
    fan = len(blocks)
    kvr_pub = 250_000 * fan + (55 + 15 * (fan - 1)) * m_tot
    return (
        HOME_NS * nh
        + MOVE_NS * kept
        + OWNER_ROW_NS * m_tot
        + _split_step_ns(kept)
        + _xq_overrun_step_ns(kept, blocks)
        + kvr_pub
    )


def _helper_ns(nl: int, mk: int, nh: int, kept: int, fan: int) -> int:
    xq_ns = 520_000 + (98 if fan == 1 else 118) * mk
    kv_17 = 450_000 + 28 * mk
    h_ready_17 = 17 * (285 * kept // 10 + 38 * nh // 100) + kv_17
    g_ready_17 = 17 * (2650 * nl // 10 + 740 * mk // 10)
    diff_17 = h_ready_17 - g_ready_17
    stall_17 = diff_17 if diff_17 > kv_17 else kv_17

    kv_3 = 900_000 + 62 * mk
    h_ready_3 = 3 * (285 * kept // 10 + 760 * nh // 10) + kv_3
    g_ready_3 = 3 * ((3350 * nl // 10 if nl > 0 else 0) + 740 * mk // 10)
    diff_3 = h_ready_3 - g_ready_3
    stall_3 = diff_3 if diff_3 > kv_3 else kv_3

    return (
        (HOME_NS + MOVE_NS) * nl
        + MOVE_NS * mk
        + _split_step_ns(mk)
        + xq_ns
        + stall_17
        + stall_3
        + (300_000 if nl > 0 else 0)
    )


def _saving(stock_max: int, wmax: int, fixed: int) -> int:
    return SCALE[0] * (stock_max - wmax) // SCALE[1] - fixed


def _zoom_m(lo: int, hi: int, eval_fn):
    best_m = lo
    best_v = eval_fn(lo)
    seen = {lo: best_v}
    m = lo + COARSE
    while m < hi:
        v = eval_fn(m)
        seen[m] = v
        if v[0] > best_v[0]:
            best_v, best_m = v, m
        m += COARSE
    if hi != lo:
        v = eval_fn(hi)
        seen[hi] = v
        if v[0] > best_v[0]:
            best_v, best_m = v, hi
    for step, rad in ((2 * ALIGN, 6 * ALIGN), (ALIGN, ALIGN)):
        m_lo = lo if best_m - rad < lo else best_m - rad
        m_hi = hi if best_m + rad > hi else best_m + rad
        for m in range(m_lo, m_hi + 1, step):
            if m in seen:
                continue
            v = eval_fn(m)
            seen[m] = v
            if v[0] > best_v[0]:
                best_v, best_m = v, m
    return best_m, best_v


def _pairs(counts: Tuple[int, ...], order: Tuple[int, ...], late_over: Tuple[bool, ...]):
    work = [(HOME_NS + MOVE_NS) * n for n in counts]
    moves = []
    for h, l in ((order[0], order[3]), (order[1], order[2])):
        nh, nl = counts[h], counts[l]
        if nh <= nl or late_over[h]:
            continue
        hi = min(PAIR_PHI * nh // 100, nh - MIN_BLOCK, CAP - nl, HELP) // ALIGN * ALIGN
        if hi < MIN_BLOCK:
            continue
        base_pair = work[h] if work[h] >= work[l] else work[l]

        def eval_pair(m: int):
            wh = _owner_ns(nh, (m,))
            wl = _helper_ns(nl, m, nh, nh - m, 1)
            w = wh if wh >= wl else wl
            return (-w, wh, wl)

        m_best, (neg_w, wh, wl) = _zoom_m(MIN_BLOCK, hi, eval_pair)
        if -neg_w < base_pair:
            work[h], work[l] = wh, wl
            moves.append(Move(h, l, nh - m_best, nh, nl))
    return moves, work


def _eval_fan2_m(nh: int, nl1: int, nl2: int, bystander_w: int, stock_max: int, m: int, fixed: int):
    c2 = HELP if CAP - nl2 >= HELP else CAP - nl2
    c1 = HELP if CAP - nl1 >= HELP else CAP - nl1
    lo1 = MIN_BLOCK if m - c2 <= MIN_BLOCK else -(-(m - c2) // ALIGN) * ALIGN
    hi1_raw = c1 if c1 <= m - MIN_BLOCK else m - MIN_BLOCK
    hi1 = hi1_raw // ALIGN * ALIGN
    if lo1 > hi1:
        return (-(1 << 60), None)
    m1_eq = (m // 2) // ALIGN * ALIGN
    if m1_eq < lo1:
        m1_eq = lo1
    elif m1_eq > hi1:
        m1_eq = hi1
    m1_wf = (49 * m + 59 * (nl2 - nl1) + 49 * ALIGN) // (98 * ALIGN) * ALIGN
    cap80 = (80 * m // 100) // ALIGN * ALIGN
    if m1_wf > cap80:
        m1_wf = cap80
    if m1_wf < lo1:
        m1_wf = lo1
    elif m1_wf > hi1:
        m1_wf = hi1
    kept = nh - m
    best_s = -(1 << 60)
    best_sp = None
    for m1 in (m1_eq,) if m1_wf == m1_eq else (m1_eq, m1_wf):
        m2 = m - m1
        wh = _owner_ns(nh, (m1, m2))
        wl1 = _helper_ns(nl1, m1, nh, kept, 2)
        wl2 = _helper_ns(nl2, m2, nh, kept, 2)
        wmax = wh
        if wl1 > wmax:
            wmax = wl1
        if wl2 > wmax:
            wmax = wl2
        if bystander_w > wmax:
            wmax = bystander_w
        s = _saving(stock_max, wmax, fixed)
        if s > best_s:
            best_s = s
            best_sp = (m1, m2)
    return (best_s, best_sp)


def _fan2(counts: Tuple[int, ...], order: Tuple[int, ...], late_over: Tuple[bool, ...], stock_max: int):
    h = order[0]
    nh = counts[h]
    l1, l2 = order[3], order[2]
    nl1, nl2 = counts[l1], counts[l2]
    if late_over[h] or not (nl1 < nh and nl2 < nh):
        return None, None
    cap1 = HELP if CAP - nl1 >= HELP else CAP - nl1
    cap2 = HELP if CAP - nl2 >= HELP else CAP - nl2
    hi = min(FAN_PHI * nh // 100, nh - MIN_BLOCK, cap1 + cap2) // ALIGN * ALIGN
    if hi < 2 * MIN_BLOCK:
        return None, None
    bystander_w = (HOME_NS + MOVE_NS) * counts[order[1]]
    fixed = FIXED_NS + FAN_NS
    m_best, (s, sp) = _zoom_m(2 * MIN_BLOCK, hi, lambda m: _eval_fan2_m(nh, nl1, nl2, bystander_w, stock_max, m, fixed))
    if sp is None:
        return None, None
    m1, m2 = sp
    return (Move(h, l1, nh - m_best, nh - m2, nl1), Move(h, l2, nh - m2, nh, nl2)), s


def _fill3(m: int, nls: Tuple[int, int, int], caps: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
    if m < 3 * MIN_BLOCK or m > sum(caps):
        return None

    def raw(level: int):
        out = []
        for nl, cap in zip(nls, caps):
            b = (level - 59 * nl) // 49
            out.append(MIN_BLOCK if b < MIN_BLOCK else cap if b > cap else b)
        return out

    lo, hi = 0, 59 * max(nls) + 49 * (m + MIN_BLOCK)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(raw(mid)) >= m:
            hi = mid
        else:
            lo = mid + 1
    blocks = [b // ALIGN * ALIGN for b in raw(lo)]
    deficit = m - sum(blocks)
    while deficit > 0:
        room = [k for k in range(3) if blocks[k] + ALIGN <= caps[k]]
        if not room:
            return None
        k = min(room, key=lambda j: (59 * nls[j] + 49 * blocks[j], j))
        blocks[k] += ALIGN
        deficit -= ALIGN
    if deficit or any(b < MIN_BLOCK or b > cap for b, cap in zip(blocks, caps)):
        return None
    return blocks[0], blocks[1], blocks[2]


def _fan3(counts: Tuple[int, ...], order: Tuple[int, ...], late_over: Tuple[bool, ...], stock_max: int):
    h = order[0]
    nh = counts[h]
    helpers = (order[3], order[2], order[1])
    nls = tuple(counts[x] for x in helpers)
    if FAN3_PHI <= 0 or late_over[h] or any(nl >= nh for nl in nls):
        return None, None
    caps = tuple((HELP if CAP - nl >= HELP else CAP - nl) // ALIGN * ALIGN for nl in nls)
    hi = min(FAN3_PHI * nh // 100, nh - MIN_BLOCK, sum(caps)) // ALIGN * ALIGN
    if hi < 3 * MIN_BLOCK:
        return None, None
    fixed = FIXED_NS + FAN3_NS

    def eval_m(m: int):
        blocks = _fill3(m, nls, caps)
        if blocks is None:
            return (-(1 << 60), None)
        kept = nh - m
        wmax = _owner_ns(nh, blocks)
        for nl, b in zip(nls, blocks):
            w = _helper_ns(nl, b, nh, kept, 3)
            if w > wmax:
                wmax = w
        return (_saving(stock_max, wmax, fixed), blocks)

    m_best, (s, blocks) = _zoom_m(3 * MIN_BLOCK, hi, eval_m)
    if blocks is None or sum(blocks) != m_best:
        return None, None
    moves, lo = [], nh - m_best
    for helper, nl, b in zip(helpers, nls, blocks):
        moves.append(Move(h, helper, lo, lo + b, nl))
        lo += b
    return tuple(moves), s


def _valid(c: Tuple[int, ...], moves: Tuple[Move, ...]) -> bool:
    owners, helpers = {mv.owner for mv in moves}, [mv.helper for mv in moves]
    if owners & set(helpers) or len(set(helpers)) != len(helpers):
        return False
    for owner in owners:
        out = sorted((mv for mv in moves if mv.owner == owner), key=lambda mv: mv.lo)
        if len(out) > MAX_FAN or out[-1].hi != c[owner] or out[0].lo < MIN_BLOCK:
            return False
        for a, b in zip(out, out[1:]):
            if a.hi != b.lo:
                return False
    for mv in moves:
        rows = mv.hi - mv.lo
        if rows < MIN_BLOCK or rows > HELP or rows % ALIGN or mv.at != c[mv.helper] or c[mv.helper] + rows > CAP:
            return False
    return True


def plan(counts: Sequence[int], late: Optional[Sequence[int]]) -> Optional[Plan]:
    if late is None or len(counts) != RANKS or len(late) != RANKS:
        return None
    c0, c1, c2, c3 = int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])
    if (c0 | c1 | c2 | c3) < 0:
        return None
    cmax = max(c0, c1, c2, c3)
    if cmax < 2 * MIN_BLOCK or cmax > CAP:
        return None
    c = (c0, c1, c2, c3)
    if c0 == cmax and c1 == cmax and c2 == cmax and c3 == cmax:
        return None
    t0, t1, t2, t3 = int(late[0]), int(late[1]), int(late[2]), int(late[3])
    late_over = (t0 > TS_ROWS, t1 > TS_ROWS, t2 > TS_ROWS, t3 > TS_ROWS)
    order = tuple(sorted(range(RANKS), key=lambda r: (-c[r], r)))
    stock_max = (HOME_NS + MOVE_NS) * cmax
    moves, work = _pairs(c, order, late_over)
    best = (_saving(stock_max, max(work), FIXED_NS), tuple(moves)) if moves else (0, ())
    fan2, fan2_s = _fan2(c, order, late_over, stock_max)
    if fan2 is not None and fan2_s > best[0]:
        best = (fan2_s, fan2)
    fan3, fan3_s = _fan3(c, order, late_over, stock_max)
    if fan3 is not None and fan3_s > best[0]:
        best = (fan3_s, fan3)
    if best[0] <= 0 or not best[1] or not _valid(c, best[1]):
        return None
    return Plan(c, (t0, t1, t2, t3), best[1], best[0])


def step_plan(forward_batch) -> Optional[Plan]:
    import torch

    if forward_batch is None or torch.cuda.is_current_stream_capturing() or not forward_batch.is_extend_in_batch:
        return None
    padding = forward_batch.dp_padding_mode
    if padding is None or not padding.is_sum_len() or forward_batch.dp_spec_prefill_coordination_applied:
        return None
    counts = forward_batch.global_num_tokens_cpu
    if counts is None or len(counts) != RANKS:
        return None
    return plan(counts, forward_batch.global_late_layer_num_tokens_cpu)
