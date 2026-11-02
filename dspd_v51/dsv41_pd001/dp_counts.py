"""dsv41-pd048 DPCNT: the replay-tail planner's DP counts from the counts this forward batch already carries.

Stock's DecoderReplayTail.plan calls deepseek_v4_replay_tail.scattered_dp_counts before layer 0 of every extend forward
under DP attention with an a2a MoE: a blocking gloo all-gather of (input_ids rows, late-layer tail rows) over the TP CPU
group. The host stops mid-forward while the GPU drains (trace p46: 2.6 ms host, ~3.5 ms GPU idle a cold forward).
Both values were exchanged before the forward on every rank:
  - each rank's padded token count by the DP sync: ForwardBatch.prepare_mlp_sync_batch stores the padded list in
    global_num_tokens_cpu and pads input_ids to this rank's entry;
  - each rank's tail rows by the same function's late-row all-gather (global_late_layer_num_tokens_cpu), the counts the
    stock graph path already reads (DecoderReplayTail.run_in_prefill_graph, late_layer_dp_counts).
Inside forward_model, scattered_dp_counts returns (max tokens, rows) from those lists after checking this rank's own
entries against the values stock would have sent; a mismatch raises. Every rank takes the same branch (the lists are
set by collectives), so no rank is left inside a gather. A batch without both lists keeps stock's gather: that is
the domain edge (no late-row exchange happened), not a fallback.

CHECK (the twin, never submitted) also runs stock's gather on every derived call, returns stock's result, and prints
the comparison: the first 20 calls, every 500th, and every mismatch (up to 50)."""

import contextlib
import sys

CHECK = False  # gen_pd048 check: run stock's gather too, compare, serve stock's result (twin, never submit)
PREFIX = "dsv41-pd048chk" if CHECK else "dsv41-pd048"
_STOCK = []
_BATCH = [None]
_N = {"derived": 0, "stock": 0, "mismatch": 0}


def prepare() -> None:
    """Capture stock's scattered_dp_counts (before any candidate forward swaps it) and check the rank layout."""
    from sglang.srt.models import deepseek_v4_replay_tail as replay_tail
    from sglang.srt.runtime_context import get_parallel

    fn = replay_tail.scattered_dp_counts
    if fn.__module__ != "sglang.srt.models.deepseek_v4_replay_tail" or fn.__qualname__ != "scattered_dp_counts":
        raise RuntimeError(f"dsv41-pd048 DPCNT: scattered_dp_counts is not stock's ({fn.__module__}.{fn.__qualname__})")
    parallel = get_parallel()
    if parallel.attn_tp_size != 1 or parallel.attn_dp_size != parallel.tp_size:
        # stock's gather is indexed by TP rank, the batch's lists by DP rank: equal only when attention TP is 1
        raise RuntimeError(f"dsv41-pd048 DPCNT requires attention TP 1 with DP = TP, got attn_tp "
                           f"{parallel.attn_tp_size}, attn_dp {parallel.attn_dp_size}, tp {parallel.tp_size}")
    if not _STOCK:
        _STOCK.append(fn)


@contextlib.contextmanager
def routed(batch):
    """Serve scattered_dp_counts from `batch` for this forward only."""
    from sglang.srt.models import deepseek_v4_replay_tail as replay_tail

    _BATCH[0] = batch
    replay_tail.scattered_dp_counts = scattered_dp_counts
    try:
        yield
    finally:
        replay_tail.scattered_dp_counts = _STOCK[0]
        _BATCH[0] = None


def scattered_dp_counts(num_tokens: int, num_rows: int):
    batch = _BATCH[0]
    tokens = None if batch is None else batch.global_num_tokens_cpu
    rows = None if batch is None else batch.global_late_layer_num_tokens_cpu
    if tokens is None or rows is None:
        _N["stock"] += 1
        return _STOCK[0](num_tokens, num_rows)
    from sglang.srt.runtime_context import get_parallel

    rank = get_parallel().attn_dp_rank
    if len(tokens) != len(rows) or tokens[rank] != num_tokens or rows[rank] != num_rows:
        raise RuntimeError(f"dsv41-pd048 DPCNT: rank {rank} would send ({num_tokens}, {num_rows}) but the batch "
                           f"carries tokens {list(tokens)} rows {list(rows)}")
    _N["derived"] += 1
    derived = (max(tokens), [int(r) for r in rows])
    if not CHECK:
        return derived
    ref = _STOCK[0](num_tokens, num_rows)
    same = (int(ref[0]), [int(r) for r in ref[1]]) == derived
    if not same:
        _N["mismatch"] += 1
    n = _N["derived"]
    if not same and _N["mismatch"] <= 50 or same and (n <= 20 or n % 500 == 0):
        print(f"{PREFIX} rank {rank} call {n} {'OK' if same else 'MISMATCH'} derived {derived} stock "
              f"({int(ref[0])}, {list(ref[1])}) local ({num_tokens}, {num_rows}) stock-path {_N['stock']} "
              f"mismatches {_N['mismatch']}", file=sys.stderr, flush=True)
    return ref
