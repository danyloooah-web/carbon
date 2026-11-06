"""king-v6 WTILE (RANKED #29, PRE P3 = sglang #47684 under our stricter rule): the eager prefill dense indexer's
index-logits row tiles sized to whole SM rounds of DeepGEMM's logits kernel. Host only, no kernel.

Scope. Only the full (dense) top-k of the index layers that are neither the candidate source nor a consumer (layers
2, 8 and 14 on this model). routed() swaps v41_indexer.full_topk.dense_prefill_topk for one eager EXTEND forward (not
TARGET_VERIFY, not capture); that wrapper swaps index_logits.flat_index_logits_rows_per_tile for its own call only,
outside graph scope (capture, breakable graph, replay tail). The candidate source and consumers (sparse_table,
dense_blocks) keep stock's tiles. Rank-local: no collective, each rank decides alone.

Rule (pick). Stock cuts a call's rows into tiles of t_s rows (2 GiB of fp32-sized logits, a multiple of q = 128 //
heads rows). The logits kernel runs one CTA per SM over q-row blocks, so a tile of b blocks costs ceil(b / S) rounds
and a tile can end in a mostly idle round. Among multiples of q no larger than t_s: the fewest rounds, then the fewest
tiles (the largest such tile); equal rounds keep t_s.

Why the bytes are stock's.
  logits  rows are a run-time argument of fp8_fp4_mqa_logits; tile offsets stay multiples of q and the width is per
          call, not per tile, so every row's logits are the same element computation in any tile.
  top-k   every tile goes to scoring.topk_ragged: the king's TOPK (one CTA a row) on DeepSelect's NORMAL class at
          >= 49,152 columns, stock DeepSelect otherwise. NORMAL is row-local. DeepSelect's CLUSTER tiers (<= the
          largest tier's rows at width > 16,384) and NORMAL's one-wave config (<= SM count rows) are other classes,
          so WTILE keeps t_s unless both stock's and its own last tile exceed max(SM count, the largest tier's rows):
          every tile of the call then runs NORMAL's waves class, as every stock tile does.
  proof   the first FIRST_USE calls whose tile WTILE changes (outside a REHOME step, whose transport must not meet a
          host sync) run stock's tiles, which serve, and then WTILE's tiles into scratch; the two top-k outputs are
          compared bitwise. Until FIRST_USE compares pass only stock's tiles serve; a mismatch turns WTILE off.
  class   (review R2) the image's DeepGEMM 0.2.3.post2 _C.so carries the 0.2.3 source's (dg023) host asserts and its
          sm100_bf16_mqa_logits JIT template (no rows argument), and dg023's sm100 dispatch compares seq_len only
          with block_q x SMs and 2048, both under clean_logits (False here), and seq_len_kv with 16,384 / 65,536. A
          binary is not source, so each changed call also has a class: the sides of every tile height of stock's and
          of WTILE's tiling against block_q x SMs and 2048, and the sides of the call's width against 16,384 /
          49,152 (the king's TOPK) / 65,536. A call serves WTILE's tiles only once FIRST_USE compares have passed and
          one has passed in its class; a call of an unproven class serves stock's tiles and is compared
          ("first-use k/FIRST_USE (new class)"). The scratch top-k is released before the line prints (review R6)."""

from __future__ import annotations

import contextlib
import sys

import torch

WTILE = False  # king-v7s GPU test copy: off (king_v6t prefill OOM)
ROLES = ("prefill", "null")
FIRST_USE = 4

_STOCK: dict = {}
_STATE = {"on": False, "why": "not prepared", "sms": 0, "thr": 0, "checked": 0}
# _SERVE[0]: 0 stock's tiles; 1 WTILE's tiles if the call's class is proven (serving); 2 WTILE's tiles (scratch proof)
_SERVE = [0]
_SEEN: list = []
_PROVEN: set = set()
_ROWS_EDGES = (2048,)
_WIDTH_EDGES = (16384, 49152, 65536)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pick(rows: int, heads: int, t_s: int, sms: int, thr: int) -> int:
    """WTILE's tile for one call: t_s unless a multiple of q below it saves a round and both last tiles exceed thr."""
    q = 128 // heads if heads > 0 and 128 % heads == 0 else 0
    if q <= 0 or sms <= 0 or t_s <= 0 or t_s >= rows or t_s % q:
        return t_s

    def rounds(t: int) -> int:
        return rows // t * _cdiv(_cdiv(t, q), sms) + _cdiv(_cdiv(rows % t, q), sms)

    # A tile of whole rounds reaches the least rounds; the first hit scanning down from t_s is the largest tile.
    best = _cdiv(_cdiv(rows, q), sms) if t_s >= q * sms else rounds(t_s)
    if rounds(t_s) <= best:
        return t_s
    t = t_s - q
    while t >= q and rounds(t) != best:
        t -= q
    if t < q:
        return t_s

    def last(n: int) -> int:
        return rows - (_cdiv(rows, n) - 1) * n

    return t if last(t_s) > thr and last(t) > thr else t_s


def _heights(rows: int, t: int) -> set:
    n = _cdiv(rows, t) if t > 0 else 1
    return {t, rows - (n - 1) * t} if n > 1 else {rows}


def klass(rows: int, width: int, heads: int, t_s: int, t: int, sms: int) -> tuple:
    """The proof class of a changed call (review R2): per tiling, the set of sides of every tile height against
    block_q x SMs and _ROWS_EDGES; plus the call width's sides against _WIDTH_EDGES."""
    q = 128 // heads if heads > 0 and 128 % heads == 0 else 0
    edges = (q * sms,) + _ROWS_EDGES

    def sides(tile: int) -> tuple:
        return tuple(sorted({tuple(h >= e for e in edges) for h in _heights(rows, tile)}))

    return sides(t_s), sides(t), tuple(width >= e for e in _WIDTH_EDGES)


def _rule(rows, width, *, heads, budget_bytes):
    t_s = _STOCK["rule"](rows, width, heads=heads, budget_bytes=budget_bytes)
    try:
        t = pick(int(rows), int(heads), int(t_s), _STATE["sms"], _STATE["thr"])
        key = klass(int(rows), int(width), int(heads), int(t_s), int(t), _STATE["sms"]) if t != t_s else None
    except Exception:  # noqa: BLE001  (stock's tile)
        t, key = t_s, None
    mode = _SERVE[0]
    served = t != t_s and (mode == 2 or (mode == 1 and key in _PROVEN))
    _SEEN[:] = [int(rows), int(width), int(t_s), int(t), key, served]
    return t if served else t_s


def mark(on: bool, why: str) -> None:
    _STATE["on"] = bool(on)
    _STATE["why"] = why


def line() -> str:
    state = "ON" if _STATE["on"] else "off"
    return f"bundle-v9 king-v6 WTILE: {state} ({_STATE['why']}) gpu{torch.cuda.current_device()}"


def _first_use(stock, data, kv, out) -> None:
    rows, width, t_s, t, key, _ = _SEEN
    mine = None
    try:
        mine = torch.empty_like(out)
        _SERVE[0] = 2
        stock(data, kv, out=mine)
        _SERVE[0] = 0
        bad = int((mine != out).sum())
    except Exception as error:  # noqa: BLE001  (stock's result is already in out; an OOM here lands here too)
        _SERVE[0] = 0
        mine = None
        mark(False, f"FAIL, stock tiles serve: first-use run at rows {rows} width {width} tile {t_s} -> {t} raised "
                    f"{type(error).__name__}: {str(error)[:200]}")
        print(line(), file=sys.stderr, flush=True)
        return
    mine = None  # review R6: release the scratch top-k before anything else
    _STATE["checked"] += 1
    k = _STATE["checked"]
    print(f"bundle-v9 king-v6 WTILE first-use {k}/{FIRST_USE}{'' if k <= FIRST_USE else ' (new class)'} "
          f"gpu{torch.cuda.current_device()}: rows {rows} width {width} tile {t_s} -> {t} class {key}: "
          f"{'BITWISE' if bad == 0 else f'MISMATCH {bad}'}", file=sys.stderr, flush=True)
    if bad:
        mark(False, f"FAIL, stock tiles serve: first-use compare at rows {rows} width {width} tile {t_s} -> {t}: "
                    f"{bad} top-k indices differ")
        print(line(), file=sys.stderr, flush=True)
        return
    _PROVEN.add(key)


def dense_prefill_topk(data, kv, *, out):
    from sglang.kernels.ops.attention.dsv4 import index_logits

    from bundle_v9 import prefill_attn, rehome_block

    stock = _STOCK["topk"]
    if not _STATE["on"] or prefill_attn._graph_scope():
        return stock(data, kv, out=out)
    _SEEN.clear()
    _SERVE[0] = 1 if _STATE["checked"] >= FIRST_USE else 0
    index_logits.flat_index_logits_rows_per_tile = _rule
    try:
        try:
            stock(data, kv, out=out)
        except Exception as error:  # noqa: BLE001  (WTILE's tiles raised: off, stock's tiles redo the whole call)
            if not (len(_SEEN) == 6 and _SEEN[5]):
                raise
            _SERVE[0] = 0
            mark(False, f"FAIL, stock tiles serve: a served call at {_SEEN[:4]} raised {type(error).__name__}: "
                        f"{str(error)[:200]}")
            print(line(), file=sys.stderr, flush=True)
            stock(data, kv, out=out)
            return None
        # A changed call that served stock's tiles (still proving, or a class not proven yet) is compared now, unless
        # a REHOME step is in flight: its transport must not meet the compare's host sync.
        if len(_SEEN) == 6 and _SEEN[2] != _SEEN[3] and not _SEEN[5] and _STATE["on"] and \
                not rehome_block.in_flight():
            _first_use(stock, data, kv, out)
    finally:
        _SERVE[0] = 0
        index_logits.flat_index_logits_rows_per_tile = _STOCK["rule"]
    return None


def _install(model) -> str:
    from sglang.kernels.ops.attention import deep_select
    from sglang.kernels.ops.attention.dsv4 import index_logits
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, scoring

    rule, topk = index_logits.flat_index_logits_rows_per_tile, full_topk.dense_prefill_topk
    for fn, owner, name in ((rule, index_logits, "flat_index_logits_rows_per_tile"),
                            (topk, scoring, "dense_prefill_topk")):
        if fn.__module__ != owner.__name__ or fn.__qualname__ != name:
            raise RuntimeError(f"{owner.__name__}.{name} is not stock's ({fn.__module__}.{fn.__qualname__})")
    if index_logits.index_logits_dtype() != torch.bfloat16:
        raise RuntimeError("index logits are not bf16 (the fp32 top-k is topk_transform_ragged_v2, not DeepSelect)")
    sms = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    thr = max([sms] + [int(rows) for rows, _ in deep_select._cluster_tiers()])
    dense = []
    for index, layer in enumerate(model.layers):
        ix = getattr(layer.self_attn, "indexer", None)
        if ix is not None and not ix.is_candidate_source and not ix.uses_candidates:
            dense.append(index)
    _STOCK.update(rule=rule, topk=topk)
    _STATE.update(sms=sms, thr=thr, checked=0)
    _PROVEN.clear()
    return (f"eager-extend dense top-k index-logits tiles in whole {sms}-SM rounds, layers {dense}; both last tiles "
            f"> {thr} rows; stock tiles serve until {FIRST_USE} changed calls compare bitwise, then a call serves "
            f"WTILE's tiles once a compare of its class has passed (tile heights vs block_q x {sms} and "
            f"{', '.join(map(str, _ROWS_EDGES))} rows, width vs {', '.join(map(str, _WIDTH_EDGES))})")


def prepare(model, role: str) -> None:
    _STOCK.clear()
    _PROVEN.clear()
    if not WTILE:
        mark(False, "switched off by wtile.WTILE; stock index-logits tiles")
    elif role not in ROLES:
        mark(False, f"role {role} is not in wtile.ROLES {ROLES}; stock index-logits tiles")
    else:
        try:
            mark(True, _install(model))
        except Exception as error:  # noqa: BLE001  (off; prepare goes on)
            _STOCK.clear()
            mark(False, f"FAIL, stock index-logits tiles: {type(error).__name__}: {str(error)[:300]}")
    print(line(), file=sys.stderr, flush=True)


@contextlib.contextmanager
def routed(view, forward_batch):
    mode = getattr(forward_batch, "forward_mode", None)
    if (not _STATE["on"] or mode is None or not mode.is_extend_without_speculative()
            or torch.cuda.is_current_stream_capturing()):
        yield
        return
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk

    full_topk.dense_prefill_topk = dense_prefill_topk
    try:
        yield
    finally:
        full_topk.dense_prefill_topk = _STOCK["topk"]
