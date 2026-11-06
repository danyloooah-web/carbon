
import contextlib
import sys

CHECK = False
PREFIX = "v4hk" if CHECK else "v4"
_STOCK = []
_BATCH = [None]
_N = {"derived": 0, "stock": 0, "mismatch": 0}


def prepare() -> None:
    from sglang.srt.models import deepseek_v4_replay_tail as replay_tail
    from sglang.srt.runtime_context import get_parallel

    fn = replay_tail.scattered_dp_counts
    if fn.__module__ != "sglang.srt.models.deepseek_v4_replay_tail" or fn.__qualname__ != "scattered_dp_counts":
        raise RuntimeError(f"v9 COUNTS: scattered_dp_counts is not stock's ({fn.__module__}.{fn.__qualname__})")
    parallel = get_parallel()
    if parallel.attn_tp_size != 1 or parallel.attn_dp_size != parallel.tp_size:
        raise RuntimeError(f"v9 COUNTS requires attention TP 1 with DP = TP, got attn_tp "
                           f"{parallel.attn_tp_size}, attn_dp {parallel.attn_dp_size}, tp {parallel.tp_size}")
    if not _STOCK:
        _STOCK.append(fn)


@contextlib.contextmanager
def routed(batch):
    from sglang.srt.models import deepseek_v4_replay_tail as replay_tail

    _BATCH[0] = batch
    replay_tail.scattered_dp_counts = scattered_replay_counts
    try:
        yield
    finally:
        replay_tail.scattered_dp_counts = _STOCK[0]
        _BATCH[0] = None


def scattered_replay_counts(num_tokens: int, num_rows: int):
    batch = _BATCH[0]
    tokens = None if batch is None else batch.global_num_tokens_cpu
    rows = None if batch is None else batch.global_late_layer_num_tokens_cpu
    if tokens is None or rows is None:
        _N["stock"] += 1
        return _STOCK[0](num_tokens, num_rows)
    from sglang.srt.runtime_context import get_parallel

    rank = get_parallel().attn_dp_rank
    if len(tokens) != len(rows) or tokens[rank] != num_tokens or rows[rank] != num_rows:
        raise RuntimeError(f"v9 COUNTS: rank {rank} would send ({num_tokens}, {num_rows}) but the batch "
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
