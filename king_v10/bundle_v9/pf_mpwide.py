"""king-v7 MP-WIDE (RANKED #49, PRE P6): the king's MOE-PREFILL variant also serves the eager prefill MegaMoE calls
below its 49,152-token DP-global floor (moe_prefill.MIN_GLOBAL_TOKENS) wherever the image's own heuristic picks
BLOCK_M 240. On this model that is DP-global 11,433-14,400, 23,354-29,349 (where the king's 20,480-28,671 tile band
is not set: above it, and in REHOME flights, which stand the band down), 35,360-44,394 and 47,411-49,151; the bands
are scanned at prepare, not written in.

Why the bytes are stock's. moe_prefill.serves already requires, below the floor as above it, the heuristic's BLOCK_M
to be 240, no tiles env, an eager call and the prepared facts; MP-WIDE only moves the floor. A MegaMoE config is a
function of BLOCK_M, the knobs and the model facts (cuda_moe_prefill heuristics/mega_moe.hpp get_mega_moe_config), and
the kernel template carries no token count, so every such call runs the one prepared MOE-PREFILL binary against
stock's one BLOCK_M-240 binary, the same pair the king serves at >= 49,152 on its own exactness argument (regrouped
epilogue rows and reordered waits only). The engine lays the shared expert's scale factors out for the same BLOCK_M.

Proof (prepare; host only). (1) Every DP-global count below the floor goes through the same stock heuristic serves()
asks (deep_gemm.get_block_m_for_mega_moe); its BLOCK_M-240 runs are the bands. (2) At both edges of every band the
library's compile_only, the call the king's own prepare makes, given the band's token counts, must return the
prepared config tuple: the same code, so no new compile and no 'unprepared MegaMoE kernel config' line can follow.
(3) MOE-PREFILL is a cross-rank kernel, so the EP group votes (one all_gather_object on its CPU group, reached by
every rank whatever failed locally): MP-WIDE serves only if every rank passed (2) with the same bands.

Off (switch, role, or any failure on any rank): the king's floor holds on every rank, with one line. The sign of the
speed change below 49,152 is unmeasured (VERIFY_prefill P6): read the MegaMoE class per DP-global band on vs off.

SHIPS OFF (king-v7 fixer, review F1). This serves the MOE-PREFILL variant on the AUDITED null role at 11.4K-49K
DP-global tokens, a regime the king never ran (about one partly filled BLOCK_M-240 block per expert, and busy-variant
/ idle-stock rank mixes far more often), and byte equality there rests on the structural argument and the config
identity above, not on a measurement; the value is +-0.005 with an unknown sign. Gate before MP_WIDE = True: one dev
boot with moe_prefill.CHECK = True showing 'OK bytes-differ 0' at about 12K / 28.7K / 40K / 48K DP-global tokens.
CHECK runs stock and then the variant on each checked call, i.e. TWO cross-rank MegaMoE launches, so every EP rank
must be busy (not IDLE) in each checked forward: an IDLE rank runs stock once and the variant launch would wait on it.
Drive the check with >= 4 concurrent prompts spread over the DP ranks; a one-busy-rank case cannot be checked this way
(that case is covered only by the wire argument: SM 0's (kNumSMs << 32) | count pushes equal stock's)."""

from __future__ import annotations

import os
import sys
import time

import torch

MP_WIDE = False   # king-v7 fixer (review F1): off until the CHECK dev boot above shows bytes-differ 0 in every band
ROLES = ("prefill", "null")

_STATE = {"on": False, "why": "not prepared"}


def mark(on_: bool, why: str) -> None:
    _STATE["on"] = bool(on_)
    _STATE["why"] = why


def line() -> str:
    state = "ON" if _STATE["on"] else "off"
    return f"bundle-v9 king-v7 MP-WIDE: {state} ({_STATE['why']}) gpu{torch.cuda.current_device()}"


def _bands(facts: dict, floor: int, block_m: int) -> list:
    """[(lo, hi)] DP-global runs below `floor` where the stock heuristic, as serves() calls it, picks `block_m`."""
    import deep_gemm

    ranks, cap = facts["ranks"], facts["cap"]
    top = min(floor, ranks * cap + 1)
    bands, lo = [], None
    for g in range(1, top):
        hit = deep_gemm.get_block_m_for_mega_moe(
            num_ranks=ranks, num_experts=facts["experts"], num_max_tokens_per_rank=cap, num_tokens=-(-g // ranks),
            num_topk=facts["topk"], mma_type="fp8xfp4", global_num_tokens=g) == block_m
        if hit and lo is None:
            lo = g
        elif not hit and lo is not None:
            bands.append((lo, g - 1))
            lo = None
    if lo is not None:
        bands.append((lo, top - 1))
    return bands


def _configs(lib, facts: dict, knobs, got: tuple, bands: list) -> str:
    """'' when compile_only returns the prepared config at both edges of every band, else the first miss."""
    ranks, cap = facts["ranks"], facts["cap"]
    for lo, hi in bands:
        for g in (lo, hi):
            cfg = tuple(int(v) for v in lib.compile_only(
                list(knobs) + [facts["num_sms"]], ranks, facts["experts"], cap, facts["topk"], facts["hidden"],
                facts["inter"], facts["shared"], facts["weight_ref"], facts["clamp"], True,
                min(cap, -(-g // ranks)), g))
            if cfg != tuple(got):
                return (f"band {lo}-{hi}: compile_only at {g} DP-global tokens gives config {cfg}, not the prepared "
                        f"MOE-PREFILL config {tuple(got)}")
    return ""


def prepare(lib, facts: dict, got: tuple, knobs, floor: int, block_m: int, role: str) -> list:
    """The proven [(lo, hi)] DP-global bands below `floor` that moe_prefill.serves admits from now on; [] keeps the
    king's floor alone. Never raises; every rank reaches the vote unless the switch or the role (both rank-uniform)
    is off."""
    if not MP_WIDE:
        mark(False, f"switched off by pf_mpwide.MP_WIDE; MOE-PREFILL from {floor} DP-global tokens (the king's)")
        print(line(), file=sys.stderr, flush=True)
        return []
    if role not in ROLES:
        mark(False, f"role {role} is not in pf_mpwide.ROLES {ROLES}; the king's floor {floor}")
        print(line(), file=sys.stderr, flush=True)
        return []
    t0 = time.perf_counter()
    err, bands = "", []
    try:
        if "DG_MEGA_MOE_TILES" in os.environ:
            err = "DG_MEGA_MOE_TILES is set at prepare"
        else:
            bands = _bands(facts, floor, block_m)
            err = _configs(lib, facts, knobs, got, bands) if bands else f"no BLOCK_M-{block_m} band below {floor}"
    except Exception as error:  # noqa: BLE001  (reported through the vote)
        err = f"{type(error).__name__}: {str(error)[:200]}"
    try:
        import torch.distributed as dist
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().moe_ep_group
        votes = [None] * int(group.world_size)
        dist.all_gather_object(votes, (err, bands), group=group.cpu_group)
    except Exception as error:  # noqa: BLE001
        mark(False, f"FAIL, the king's floor {floor}: the EP vote raised {type(error).__name__}: {str(error)[:200]}")
        print(line(), file=sys.stderr, flush=True)
        return []
    bad = [(rank, e) for rank, (e, _) in enumerate(votes) if e]
    if bad:
        mark(False, f"FAIL, the king's floor {floor} on every rank: rank {bad[0][0]}: {bad[0][1]}")
    elif any(list(b) != list(bands) for _, b in votes):
        mark(False, f"FAIL, the king's floor {floor} on every rank: the ranks scanned different bands")
    else:
        mark(True, f"MOE-PREFILL at every eager heuristic BLOCK_M-{block_m} call from {bands[0][0]} DP-global tokens "
                   f"(the king's floor {floor}); new bands {bands}; compile_only at every band edge = the prepared "
                   f"config {tuple(got)}; voted on {len(votes)} EP ranks; {time.perf_counter() - t0:.2f} s")
    print(line(), file=sys.stderr, flush=True)
    return list(bands) if _STATE["on"] else []
