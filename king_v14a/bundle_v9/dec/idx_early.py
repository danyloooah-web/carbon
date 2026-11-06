"""king-v13s IDX-EARLY = IDX-WP-EARLY + IDX-Q-EARLY (struct/king_lane/timeline_v10/decode_path.md #1; v13s/S2.md).

Two stream-wiring levers on the decode role's heavy index layers (the ratio 1 / 2 indexers IDXSTREAM serves, 2 8 14 20
24 28 32 36). The runtime hooks live in index_streams.py; this module holds the switches, the engine pins, the
prepare-time proof and the boot lines.

IDX-WP-EARLY (index-only layers, compressor None: 24 28 32 36). The indexer's weights_proj (head_weights_raw(x): the
cuBLAS split-K nvjet + splitKreduce, 12.7 us) reads only x and its weight. x is the attention input, ready at the
attention-side boundary end (~7.4 us after the MegaMoE end), but king_v10b forks the call after the q-norm event
(index_streams._sources, elif branch, reached only after the attention wq_b is issued), so it starts at ~19 and its
reduce (31.3-31.8) binds index_q, whose other input (idx wq_b) is done at 23.5. Here index_streams.routed() swaps
deepseek_v4.MQALayer._forward_prepare_low_ratio_multi_stream (class attribute, decode / target-verify forwards only,
restored in finally; identity-checked at swap-in, source-pinned below) for index_streams._prepare_wp, which issues the
same call on the same side stream (_SIDE[0]) after _SIDE[0].wait_stream(current) and then runs stock's prepare
unchanged. That is the fork the compressor layers already take (their first sources call). _sources finds "w" and
skips its own fork; _index_q_and_weights still checks that "w" was computed on the very x object it scores.

IDX-Q-EARLY (all 8 heavy layers). index_q (king-v6 IDX-W16's PDL copy, or stock's index_q_rope_pack_weights when that
lever is off) reads idx wq_b's output, weights_proj's output, layer.freqs_cis, the positions scoring receives and the
head-weight scale: none of the compressor's outputs (only the logits read the new index-K entries). The positions are
the ones stock's forward_low_ratio_sources hands to scoring: in DECODE the backend's hoisted low_ratio_pos_i64[:rows];
in TARGET_VERIFY (every served graph) that field is None (make_forward_metadata_from_raw_verify never sets it and
init_forward_metadata_in_graph refreshes it only for DECODE), so stock passes `pos = positions`, the layer prepare's
own positions argument (a graph input), which the prepare hook stashes (fixer, reviews F1: the first build keyed only
on the hoisted field and never fired in verify).
king_v10b issues it inside stock scoring on the sources stream, behind the compressor chain, so it inherits that
chain's join (8-10 us late on L2 / L8 / L20). Here the q-norm hook issues it on the indexer's wq_b stream (_SIDE[1],
the device's top priority) right after idx wq_b and after _SIDE[1].wait_stream(_SIDE[0]) (weights_proj), with the
positions view stock would hand to scoring (index_streams._q_positions). The attention wq_b still waits only
for idx wq_b (current.wait_stream(side) is recorded before the index_q launch). _index_q_and_weights hands the stash
out after current.wait_event(ready) only when every argument of the late call is the stash's: freqs_cis is the same
object, positions has the same data_ptr / shape / stride / dtype / device, rows match, and x / q_lora pass the IDXSTREAM
identity checks. Anything else (no positions view, a different view) recomputes on the current stream exactly as
king_v10b does; the first such miss prints one line.

Exactness: the same kernels run on the same tensors with the same arguments; only streams and events change. Every
producer -> consumer pair keeps a stream-order or event edge: x (boundary) -> weights_proj (fork after the boundary);
idx wq_b and weights_proj -> index_q (same stream / event); index_q -> logits (event before the stash is returned);
the verify positions are a graph input (written before the replay); the decode-mode hoisted positions are written by
the engine's in-graph metadata prep before layer 1 (with ROOT-FORK too: its forward stream waits on the prep before
layer 1) and read only by heavy layers >= 2. Memory: the early outputs are allocated on _SIDE[1]; that stream's next
use is the next heavy layer's idx wq_b, issued after
_SIDE[1].wait_stream(current) on a main stream that has joined the sources stream (stock's prepare end), i.e. after
the logits finished reading them; weights_proj's buffers on _SIDE[0] follow the same join chain (as in king_v10b).
The prepare-time proof below runs the served wiring (WP fork, idx wq_b, early index_q, the hook's matching path) on
every served layer at PROBE_ROWS, once per positions source (the verify stash and the decode hoisted view, both chosen
by the serving function index_streams._q_positions), and compares every byte against king_v10b's late path (idx wq_b
on the reused pair, weights_proj, index_q on the current stream). The kernels allocate their own outputs, so there is
no NaN pre-fill to do; all output bytes are compared. Bounded (BUDGET_S), no collective, eager only. Any failure turns
both halves off: king_v10b's IDXSTREAM path serves, unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import time

import torch

# The lead's switches. IDX_EARLY_ON False: both halves off, exactly king_v10b's IDXSTREAM. The two sub-switches turn
# one half off each (A/B arms).
IDX_EARLY_ON = True
WP_EARLY_ON = True
Q_EARLY_ON = False  # king-v14a (v13s3: firing index_q early cost ~+30 us heavy pre-attn vs v13s2's late path)
MAX_ROWS = 288  # decode / target-verify graph rows; larger eager calls take king_v10b's path
PROBE_ROWS = (6, 96, 192, 240, 288)
BUDGET_S = 90.0
SEED = 20261113
PREFIX = "bundle-v9 king-v13s"
# The engine text the WP fork's argument rests on (live_1106/sg_cacheon_dsv41-pd-tee; the top tree's IHX pins the same
# prepare segment). Segment sha256[:16], decorator line to the block end, as tactic_pins computes them.
WP_PINS = (
    ("deepseek_v4", "MQALayer._forward_prepare_low_ratio_multi_stream", "bc455a7c3e96a5e1"),
    ("deepseek_v4", "MQALayer._compute_q_a", "30fa4f1c450502b2"),
)
# The positions source IDX-Q-EARLY mirrors (fixer, reviews F1): forward_low_ratio_sources' pos choice, its pass-through
# into DecodeInputs.positions, and scoring's hand-off to _index_q_and_weights. A mismatch turns IDX-Q-EARLY off only.
Q_PINS = (
    ("deepseek_v4", "MQALayer._forward_prepare_low_ratio_multi_stream", "bc455a7c3e96a5e1"),
    ("backend", "DeepseekV4AttnBackend.forward_low_ratio_sources", "561930f7b852ab90"),
    ("backend", "DeepseekV4AttnBackend._low_ratio_index_topk", "15554df6cf92120b"),
    ("backend", "DeepseekV4AttnBackend._make_low_ratio_decode_indexer_inputs", "9365b7a60b9bbac7"),
    ("backend", "DeepseekV4AttnBackend.make_forward_metadata_from_raw_verify", "019d1f789b7441d9"),
    ("backend", "DeepseekV4AttnBackend.init_forward_metadata_in_graph", "13a897583ee4aadf"),
    ("scoring", "_index_q_and_weights", "1bd9f59622d845d6"),
    ("scoring", "get_deep_gemm_decode_data", "6f2511457ad5f7c9"),
)

_STATE: dict = {name: {"on": False, "why": "not prepared"} for name in ("IDX-WP-EARLY", "IDX-Q-EARLY")}
_WP: list = [False]
_Q: list = [False]
_STOCK_PREP: list = []
_COUNT: dict = {"wp": 0, "wp_skip": 0, "q_issued": 0, "q_not_issued": 0, "q_hit": 0, "q_miss": 0}
_SAID: set = set()
_REPORTED: list = [False]


def mark(name: str, on: bool, why: str) -> None:
    _STATE[name]["on"] = bool(on)
    _STATE[name]["why"] = why


def rank_tag() -> str:
    """'rank R; ' for the boot lines (fixer, review F6), '' when the parallel state is not up."""
    try:
        from sglang.srt.runtime_context import get_parallel

        return f"rank {get_parallel().tp_rank}; "
    except Exception:  # noqa: BLE001
        return ""


def line(name: str) -> str:
    state = _STATE[name]
    return f"{PREFIX} {name}: {'ON' if state['on'] else 'off'} ({rank_tag()}{state['why']})"


def say(name: str) -> None:
    print(line(name), file=sys.stderr, flush=True)


def _once(key: str, text: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(f"{PREFIX} {text}", file=sys.stderr, flush=True)


def wp_on() -> bool:
    return _WP[0]


def q_on() -> bool:
    return _Q[0]


_FIRST = {"wp": "IDX-WP-EARLY: first weights_proj forked at the attention-side boundary (shown once)",
          "q_hit": "IDX-Q-EARLY: first early index_q taken by its indexer (shown once)"}


def count(key: str, note: str = "") -> None:
    """Host-side counters (capture and eager forwards); the first event of each kind prints one line."""
    _COUNT[key] += 1
    if _COUNT[key] == 1 and (note or key in _FIRST):
        _once(key, note or _FIRST[key])


def first_capture_snapshot():
    """At index_streams.routed's entry: the counters, if this forward is the first captured one with a lever ON."""
    if _REPORTED[0] or not (_WP[0] or _Q[0] or _tail_on()) or not torch.cuda.is_current_stream_capturing():
        return None
    from bundle_v9.dec import tail_norm

    return dict(_COUNT), dict(tail_norm._COUNT)


def _tail_on() -> bool:
    from bundle_v9.dec import tail_norm

    return bool(tail_norm._ON[0])


def first_capture_report(snapshot, forward_batch) -> None:
    """One line per lever after the first captured forward (fixer, reviews F1 / F6): how often each lever engaged in
    that graph, so a lever that is ON at boot but never fires is visible (0 issued / 0 fused)."""
    if snapshot is None or _REPORTED[0]:
        return
    _REPORTED[0] = True
    from bundle_v9.dec import tail_norm

    idx0, tail0 = snapshot
    d = {key: _COUNT[key] - idx0.get(key, 0) for key in _COUNT}
    t = {key: tail_norm._COUNT[key] - tail0.get(key, 0) for key in tail_norm._COUNT}
    ids = getattr(forward_batch, "input_ids", None)
    rows = None if ids is None else ids.shape[0]
    tag = rank_tag()
    if _WP[0] or _Q[0]:
        print(f"{PREFIX} IDX-EARLY: first captured forward ({tag}{rows} rows): weights_proj forked early {d['wp']} "
              f"(index-only layers), index_q issued early {d['q_issued']} / not issued {d['q_not_issued']} / taken "
              f"{d['q_hit']} / missed {d['q_miss']} (heavy layers)", file=sys.stderr, flush=True)
    if tail_norm._ON[0]:
        print(f"{PREFIX} TAIL-NORM: first captured forward ({tag}{rows} rows): fused {t['fused']}, combine handoff "
              f"{t['combine_hit']} (miss {t['combine_miss']}), norm handoff {t['norm_hit']} (miss {t['norm_miss']}), "
              f"skipped {t['skip']}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------- engine source pins
def _definition(tree, qualname: str):
    node = tree
    for part in qualname.split("."):
        found = [child for child in getattr(node, "body", ())
                 if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part]
        if not found:
            return None
        node = found[-1]
    return node


def segment_sha16(module, qualname: str, cache: dict) -> str:
    hit = cache.get(module.__name__)
    if hit is None:
        src = inspect.getsource(module)
        hit = cache[module.__name__] = (src, ast.parse(src))
    src, tree = hit
    node = _definition(tree, qualname)
    if node is None:
        return "missing"
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    seg = "".join(src.splitlines(keepends=True)[start - 1: node.end_lineno])
    return hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]


def pin_mismatches(modules: dict, specs) -> list:
    """[] when every (module key, qualname, sha16) matches the loaded engine text; else 'mod:qual=got!=want' items."""
    cache: dict = {}
    bad = []
    for key, qualname, want in specs:
        try:
            got = segment_sha16(modules[key], qualname, cache)
        except (OSError, TypeError, SyntaxError, KeyError) as error:
            got = f"err:{type(error).__name__}"
        if got != want:
            bad.append(f"{key}:{qualname}={got}!={want}")
    return bad


# ------------------------------------------------------------------------------- the class-level prepare swap (WP)
def swap_in(hook) -> bool:
    """Install the prepare hook (IDX-WP-EARLY's fork, IDX-Q-EARLY's positions stash) for one decode / verify forward.
    False (nothing changed) unless the class attribute is exactly stock's function, the one the pins and the proof
    cover."""
    if not (_WP[0] or _Q[0]) or not _STOCK_PREP:
        return False
    try:
        from sglang.srt.models import deepseek_v4

        cls = deepseek_v4.MQALayer
        if inspect.getattr_static(cls, "_forward_prepare_low_ratio_multi_stream", None) is not _STOCK_PREP[0]:
            count("wp_skip", "IDX-EARLY: MQALayer._forward_prepare_low_ratio_multi_stream is not stock's at this "
                             "forward's entry (another swap holds it); king_v10b's late weights_proj and index_q serve "
                             "it")
            return False
        cls._forward_prepare_low_ratio_multi_stream = hook
        return True
    except Exception:
        return False


def swap_out(installed: bool) -> None:
    if installed:
        from sglang.srt.models import deepseek_v4

        deepseek_v4.MQALayer._forward_prepare_low_ratio_multi_stream = _STOCK_PREP[0]


def stock_prepare():
    return _STOCK_PREP[0]


# -------------------------------------------------------------------------------------------------------- the proof
def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool(torch.equal(_bits(a), _bits(b)))


def _prove(ix, deadline: float) -> int:
    """The served wiring against king_v10b's late path, byte for byte, on every served layer at PROBE_ROWS, once per
    positions source: TARGET_VERIFY (the prepare's positions argument, stashed as index_streams._prepare_wp stashes it,
    no hoisted field) and DECODE (the backend's hoisted view); both chosen by the serving index_streams._q_positions."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    cur = torch.cuda.current_stream()
    side_w, side_q = ix._SIDE[0], ix._SIDE[1]
    cases = 0
    gen = None
    for attn in ix._ATTN:
        indexer, norm = attn.indexer, attn.q_norm
        device = norm.weight.device
        gen = gen or torch.Generator(device=device).manual_seed(SEED)
        hidden = indexer.weights_proj.weight.shape[1]
        table = attn.freqs_cis.shape[0]
        for rows in PROBE_ROWS:
            if time.monotonic() > deadline:
                raise RuntimeError(f"proof budget of {BUDGET_S:.0f} s exhausted at layer {attn.layer_id}, {rows} rows")
            x = (torch.randn((rows, hidden), device=device, generator=gen) * 2).to(torch.bfloat16)
            qa = (torch.randn((rows, norm.weight.shape[0]), device=device, generator=gen) * 3).to(torch.bfloat16)
            # int64, a [:rows] view of a longer buffer: the graph's positions input (buffers.positions[:num_tokens])
            # and the hoisted low_ratio_pos_i64 have this form.
            pbuf = torch.randint(0, table, (rows + 13,), device=device, generator=gen, dtype=torch.int64)
            late_positions = pbuf[:rows]
            y, y_q, y_sf = ix._STOCK_NORM(qa, norm.weight, norm.variance_epsilon)
            # king_v10b's late path, serially on the current stream.
            q_ref, _ = indexer.wq_b(Mxfp8SwizzledInput(y_q, y_sf))
            w_ref = indexer.head_weights_raw(x)
            want = ix._late_q(indexer, q_ref, w_ref, attn.freqs_cis, late_positions, rows)
            torch.cuda.synchronize(device)
            for source in ("verify", "decode"):
                # The served wiring: the WP fork, idx wq_b on the top-priority stream, the early index_q, the match.
                pending: dict = {}
                side_w.wait_stream(cur)
                with torch.cuda.stream(side_w):
                    pending["w"] = (x, indexer.head_weights_raw(x))
                side_q.wait_stream(cur)
                with torch.cuda.stream(side_q):
                    q, _ = indexer.wq_b(Mxfp8SwizzledInput(y_q, y_sf))
                cur.wait_stream(side_q)
                pending["q"] = (y, q, y_q, y_sf)
                if source == "verify":
                    ix._POS[id(indexer)] = late_positions  # what _prepare_wp stashes: the prepare's positions argument
                    positions = ix._q_positions(indexer, rows, None)
                else:
                    ix._POS.pop(id(indexer), None)
                    positions = ix._q_positions(indexer, rows, pbuf)
                if positions is None or not ix._issue_q(indexer, attn, pending, q, rows, positions):
                    raise RuntimeError(f"layer {attn.layer_id}: the early index_q was not issued at {rows} rows "
                                       f"({source} positions)")
                got = ix._take_q(pending, x, attn.freqs_cis, late_positions)
                if got is None:
                    raise RuntimeError(f"layer {attn.layer_id}: the hook did not match its own stash at {rows} rows "
                                       f"({source} positions)")
                cur.wait_stream(side_w)
                torch.cuda.synchronize(device)
                pairs = [("idx wq_b", q_ref, q), ("weights_proj", w_ref, pending["w"][1]), ("q_fp4", want[0], got[0]),
                         ("q_sf", want[1], got[1]), ("weights", want[2], got[2])]
                bad = [name for name, a, b in pairs if not _same(a, b)]
                if bad:
                    raise RuntimeError(f"layer {attn.layer_id}, {rows} rows, {source} positions: {bad} differ from "
                                       "king_v10b's late path")
                # Arguments that are not the stash's must miss (the hook then recomputes as king_v10b does).
                if (ix._take_q(pending, x, attn.freqs_cis, pbuf[1:rows + 1]) is not None
                        or ix._take_q(pending, x, attn.freqs_cis[:], late_positions) is not None
                        or ix._take_q(pending, x[:rows - 1] if rows > 1 else x[:0], attn.freqs_cis, late_positions)
                        is not None):
                    raise RuntimeError(f"layer {attn.layer_id}: the hook matched arguments its stash was not built for")
                cases += 1
            # No stash, or a stash of another length, issues nothing (the late path serves).
            ix._POS.pop(id(indexer), None)
            if ix._q_positions(indexer, rows, None) is not None:
                raise RuntimeError(f"layer {attn.layer_id}: positions were chosen without a stash")
            ix._POS[id(indexer)] = pbuf
            if ix._q_positions(indexer, rows, None) is not None:
                raise RuntimeError(f"layer {attn.layer_id}: a stash longer than the rows was used")
    ix._POS.clear()
    return cases


def prepare(model, ix) -> None:
    """IDX-EARLY prepare, from index_streams.prepare after the side streams, IDX-W16 and the paged kernels. Never
    raises: any failure leaves both halves off (king_v10b's IDXSTREAM path)."""
    _WP[0] = _Q[0] = False
    _REPORTED[0] = False
    for key in _COUNT:
        _COUNT[key] = 0
    if not IDX_EARLY_ON:
        for name in _STATE:
            mark(name, False, "switched off by idx_early.IDX_EARLY_ON; king_v10b's IDXSTREAM wiring")
            say(name)
        return
    try:
        from sglang.srt.layers.attention import deepseek_v4_backend
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
        from sglang.srt.models import deepseek_v4

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare requires an eager stream")
        if len(ix._SIDE) != 3 or not ix._ATTN:
            raise RuntimeError("IDXSTREAM's side streams or served layers are missing")
        stock = inspect.getattr_static(deepseek_v4.MQALayer, "_forward_prepare_low_ratio_multi_stream", None)
        if stock is None or getattr(stock, "__qualname__", "") != "MQALayer._forward_prepare_low_ratio_multi_stream":
            raise RuntimeError("MQALayer._forward_prepare_low_ratio_multi_stream is not stock's function")
        _STOCK_PREP[:] = [stock]
        for attn in ix._ATTN:
            if type(attn) is not deepseek_v4.MQALayer:
                raise RuntimeError(f"layer {attn.layer_id}'s attention is {type(attn).__name__}, not MQALayer")
            if inspect.getattr_static(attn, "_forward_prepare_low_ratio_multi_stream", None) is not stock:
                raise RuntimeError(f"layer {attn.layer_id} carries an instance-level low-ratio prepare")
        start = time.monotonic()
        cases = _prove(ix, start + BUDGET_S)
        elapsed = time.monotonic() - start
        for key in _COUNT:  # the proof's own issues are not served forwards
            _COUNT[key] = 0
    except Exception as error:
        for name in _STATE:
            mark(name, False, f"FAIL, king_v10b's IDXSTREAM wiring: {type(error).__name__}: {str(error)[:300]}")
            say(name)
        return
    compressor = sorted(a.layer_id for a in ix._ATTN if a.compressor is not None)
    index_only = sorted(a.layer_id for a in ix._ATTN if a.compressor is None)
    q_bad = pin_mismatches({"deepseek_v4": deepseek_v4, "backend": deepseek_v4_backend, "scoring": scoring}, Q_PINS)
    if not Q_EARLY_ON:
        mark("IDX-Q-EARLY", False, "switched off by idx_early.Q_EARLY_ON; index_q inside stock scoring")
    elif q_bad:
        mark("IDX-Q-EARLY", False, f"engine source pins differ ({', '.join(q_bad)}); index_q inside stock scoring "
                                   f"as in king_v10b")
    else:
        _Q[0] = True
        layers = sorted(compressor + index_only) if WP_EARLY_ON else compressor
        mark("IDX-Q-EARLY", True, f"index_q issued on the indexer's wq_b stream right after idx wq_b (waits only for "
                                  f"weights_proj) on layers {layers}, rows <= {MAX_ROWS}, with the positions scoring "
                                  f"receives (verify: the prepare's positions argument; decode: the hoisted view); the "
                                  f"hook takes the stash only for its exact arguments; {len(Q_PINS)} engine segments "
                                  f"pinned; bytes proven against king_v10b's late path on {cases} cases, both "
                                  f"positions sources ({elapsed:.1f} s)")
    if not WP_EARLY_ON:
        mark("IDX-WP-EARLY", False, "switched off by idx_early.WP_EARLY_ON; weights_proj forks after the q-norm")
    elif not index_only:
        mark("IDX-WP-EARLY", False, "no index-only served layer")
    else:
        bad = pin_mismatches({"deepseek_v4": deepseek_v4}, WP_PINS)
        if bad:
            mark("IDX-WP-EARLY", False, f"engine source pins differ ({', '.join(bad)}); weights_proj forks after the "
                                        f"q-norm as in king_v10b")
        else:
            _WP[0] = True
            mark("IDX-WP-EARLY", True, f"weights_proj forked at the attention-side boundary (before wqkv_a) on "
                                       f"index-only layers {index_only}, by a class-level swap of MQALayer's low-ratio "
                                       f"prepare in decode / verify forwards (pinned, restored per forward); bytes "
                                       f"proven with IDX-Q-EARLY's cases")
    say("IDX-WP-EARLY")
    say("IDX-Q-EARLY")
