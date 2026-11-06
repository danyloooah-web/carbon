"""king-v7 AWS (dec_rest DR1, re-priced by its verifier and re-designed on our measured traces): the target-verify
attention split into its two waves, with the first wave's wo_a overlapping the second wave (decode role only).

Today (VATTN, one launch over R rows with CLC work stealing, one 1-row job per SM): above ~148 rows the attention runs
two waves, then wo_a (DeepGEMM fp8_einsum "bhr,hdr->bhd", 8 x 1024 x 4096, fp8 + packed-UE8M0 output) streams its
33.5 MB of weights for 13-15.5 us after the attention ends, at every bucket and with or without VATTN-T's early
trigger (critpath traces king1 / v2e / v5: king_lane/v7/attn/tools/attn_tail.out). AWS, at the graph buckets of
AWS_BUCKETS that pass the install gates, on the layers with the extra (compressed) cache:

  main:  A1 = VATTN(rows [0, n1)) -> event -> A2 = VATTN(rows [n1, R)) -> wo_a(rows [n1, R)) -> SF copy -> join
  side:                      wait(event) -> wo_a(rows [0, n1)) on AWS side SMs -> SF copy -> record

n1 = SMs - AWS_RESERVE_SMS, because the stats stream's prenorm_pair still holds ~16 SMs when the attention starts
(attn_head.out), so A1 is one wave. The side wo_a runs under a scoped deep_gemm.set_num_sms (the engine's own
configure_deep_gemm_num_sms pattern) on the SMs that A2 and the stats stream leave free, while A2 runs; the main
stream is then left with only the tail rows' wo_a. VATTN-T stays on both launches (the side wo_a's CTAs are resident
before A2's trigger lets the tail wo_a land), and ATTN-PDL-IN applies to both launches as to any VATTN call.

Exactness (bitwise by construction; proven at install and at first use):
- attention rows are independent jobs (enable_split_kv = 0, width <= 640), so a row's bytes do not depend on which
  launch computes it; out_fp8 and lse are row slices of one buffer each, out_sf is per slice (DeepGEMM's
  check_sf_layout takes an SF whose M stride is align4 of its OWN rows, so a row-offset view of one SF is refused);
- wo_a has no split-K: each output element is one K-ordered block-scaled MMA chain whatever the tile config or SM
  count, and its fp8 output SF is per row; the output rows are row slices of the engine's y, and the output SF of each
  slice is written in DeepGEMM's layout for that slice and copied into the engine's y_sf (the engine's own prefill path
  does the same restride copy: deepseek_v4_mega_attention_with_mxfp8_output);
- install, per bucket: VATTN against the engine's kernel at that row count, then the split chain against the unsplit
  chain for out_fp8, out_sf, y and y_sf, byte for byte, eager and from a captured graph (the served launch form), on
  AWS_ROTATE rotated synthetic inputs (VATTN's _synth, its edge case first) with the engine's own wo_a quantizer; then a
  speed gate on the same graphs: the split chain must beat the unsplit one by AWS_MARGIN_US a layer, with L2 flushed
  between layers (in serving the MoE streams GBs between two attentions);
- first use (eager warm-up, per key): the split attention against the engine's kernel (vattn._mega_decode) and the
  split wo_a against the unsplit wo_a on the live inputs (_aws_mega_core); a difference serves the reference bytes and
  turns that key off, and a key with no verdict is never captured split (P5-V).

Off (the engine's MQALayer._mega_core and VATTN's unsplit call serve) on: AWS_ON = False, VATTN off, a role other
than decode, an engine pin mismatch, no DeepGEMM SM setter, too few SMs, any install raise or difference, a bucket
whose speed gate fails, a key whose first-use verdict is False, a row count with no served plan, a caller that passes
its own output buffers, a layer without the extra cache (layers 0-1), a layer of another shape, and any raise in the
split wo_a (the side stream is joined first; the unsplit wo_a then runs on the assembled SF and AWS turns off for the
process). Scope: vattn.enter / leave swap MQALayer._mega_core for our node's target-verify forwards only, as VATTN
swaps the backend's method, and restore it in `finally`."""
import collections
import contextvars
import time

import torch

AWS_ON = True               # False: install returns at once; nothing is swapped, built or compiled
AWS_BUCKETS = (156, 168, 180, 192)  # graph rows served split if they pass install; 240 / 264 / 288 leave the side
                            # wo_a too few SMs (A2 holds R - n1 CTAs), and <= SMs rows run one wave unsplit
AWS_RESERVE_SMS = 16        # SMs left to the stats stream (prenorm_pair / reduce_sinkhorn, 16 CTAs each); n1 = SMs - this
AWS_MIN_TAIL = 8            # A2 must hold at least this many rows
AWS_MIN_SIDE_SMS = 64       # wo_a at M > 128 rows runs 64 tiles (128 x 256, 2-CTA clusters): never fewer SMs than that
AWS_MARGIN_US = 1.0         # install speed gate: split chain <= unsplit chain - this, per layer
AWS_ROTATE = 3              # rotated input sets (and wo_a weight sets) per install graph
AWS_FLUSH_BYTES = 160 << 20  # L2 flush before each chain of the install graphs (B300 L2 is ~126 MB)
AWS_BUDGET_S = 120.0        # install starts no further bucket after this long (first boot compiles DeepGEMM configs)
AWS_ROLES = ("decode",)     # the null (audit) role keeps VATTN's unsplit call
# MQALayer._mega_core (its tail is re-run here when the call is not split) and _fp8_wo_a_einsum (its out-tuple form is
# what both slices call), sg_real 15338a8e / live 11-06 copy (inspect.getsource digests, as vattn.pins_match takes)
PINS = {"_mega_core": "993a18ee096ebfa8", "_fp8_wo_a_einsum": "9ee601564ebe8e79"}
_GROUPS, _RANK, _K = 8, 1024, 4096  # n_local_groups, o_lora_rank, the attention output width per group
Plan = collections.namedtuple("Plan", "rows n1 side_sms")
_ARM = contextvars.ContextVar("king_v7_aws_arm", default=None)
_STASH = []
_g = {"state": None, "plans": {}, "woa": {}, "cls": None, "orig": None, "swapped": False, "side": {}, "dv4": None,
      "mhc": None, "dg": None, "report": {}}


def _say(what):
    from bundle_v9.dec import vattn

    vattn.say("aws", what)


# ------------------------------------------------------------------------------------------------- vattn's hooks
def serving():
    return _g["state"] == "on"


def armed(rows):
    """The plan _aws_mega_core armed for its own attention call, if it is for exactly `rows` rows (vattn)."""
    plan = _ARM.get()
    return plan if plan is not None and plan.rows == rows else None


def stash(res):
    """vattn._mega_decode hands its split launch's buffers and event to the _aws_mega_core that armed it."""
    _STASH.append(res)


def enter():
    """Outermost entry of VATTN's scope (a target-verify forward of our node): MQALayer._mega_core swapped."""
    if _g["state"] == "on" and not _g["swapped"]:
        setattr(_g["cls"], "_mega_core", _aws_mega_core)
        _g["swapped"] = True


def leave():
    if _g["swapped"]:
        setattr(_g["cls"], "_mega_core", _g["orig"])
        _g["swapped"] = False


def _off(why):
    if _g["state"] == "on":
        _say(f"AWS off for this process: {why}; the engine's _mega_core and VATTN's unsplit call serve")
    _g["state"], _g["plans"] = "off", {}


# ------------------------------------------------------------------------------------------------- the split wo_a
def _fill(st):
    """The engine-visible SF of all rows from the two slices (fallback and first-use reference only)."""
    n1 = st["n1"]
    st["full"][:n1].copy_(st["sf1"])
    st["full"][n1:].copy_(st["sf2"])


def _split_woa(st, wo_a, y, y_sf, plan, dev):
    """Side: wait(A1's event) -> wo_a(rows [0, n1)) at plan.side_sms SMs -> its SF into y_sf[:n1] -> record. Main:
    wo_a(rows [n1, R)) -> its SF into y_sf[n1:] -> wait(side). The side stream is joined back in every case, also when
    a call raises (a capture never ends with an unjoined stream)."""
    woa = _g["dv4"]._fp8_wo_a_einsum   # looked up per call, as _mega_core does: composes with any later swap
    dg, mhc = _g["dg"], _g["mhc"]
    rows, n1 = plan.rows, plan.n1
    m2, cols, out = rows - n1, _GROUPS * _RANK, st["out"]
    ysf1 = mhc.empty_deepgemm_mxfp8_scales(n1, cols, dev)    # allocated on the main stream before the fork
    ysf2 = mhc.empty_deepgemm_mxfp8_scales(m2, cols, dev)
    cur = torch.cuda.current_stream(dev)
    side = _g["side"][dev.index]
    done = torch.cuda.Event()
    side.wait_event(st["ev"])
    try:
        with torch.cuda.stream(side):
            prev = int(dg.get_num_sms())
            dg.set_num_sms(min(plan.side_sms, prev))
            try:
                woa(out[:n1, :_GROUPS], st["sf1"][:, :_GROUPS], *wo_a, recipe=(1, 1, 32),
                    out=(y[:n1].view(n1, _GROUPS, -1), ysf1))
            finally:
                dg.set_num_sms(prev)
            y_sf[:n1].copy_(ysf1)
        woa(out[n1:, :_GROUPS], st["sf2"][:, :_GROUPS], *wo_a, recipe=(1, 1, 32),
            out=(y[n1:].view(m2, _GROUPS, -1), ysf2))
        y_sf[n1:].copy_(ysf2)
    finally:
        done.record(side)
        cur.wait_event(done)


def _layer_ok(layer):
    wo_a = getattr(layer, "mega_wo_a", None)
    return (getattr(layer, "use_mega_attention", False) and getattr(layer, "n_local_groups", 0) == _GROUPS
            and getattr(layer, "o_lora_rank", 0) == _RANK and isinstance(wo_a, tuple) and len(wo_a) == 2)


def _aws_mega_core(self, q, attn_sink, forward_batch, attn_backend, allow_mxfp8_output=False, out=None):
    """MQALayer._mega_core while AWS serves. Anything but a served bucket with the engine's own fp8-pair output is the
    engine's method, untouched. A served call allocates that pair exactly as _mega_core would, runs the engine's
    attention call with this call's plan armed (vattn._mega_decode splits it, or not), and then runs either the
    engine's own tail (not split) or the split wo_a."""
    from bundle_v9.dec import vattn

    orig = _g["orig"]
    rows = int(q.shape[0])
    plan = _g["plans"].get(rows) if (_g["state"] == "on" and out is None and allow_mxfp8_output) else None
    wkey, verdict = None, None
    if plan is not None:
        wkey = (plan.rows, plan.n1)
        verdict = _g["woa"].get(wkey)
        if (verdict is False or (verdict is None and torch.cuda.is_current_stream_capturing())
                or not vattn.context.get() or not _layer_ok(self) or not self._mega_mxfp8_output(rows)):
            plan = None
    if plan is None:
        return orig(self, q, attn_sink, forward_batch, attn_backend, allow_mxfp8_output=allow_mxfp8_output, out=out)
    groups = self.n_local_groups
    y, y_sf = _g["mhc"].empty_deepgemm_mxfp8_pair(rows, groups * self.o_lora_rank, q.device)
    _STASH.clear()
    token = _ARM.set(plan)
    try:
        data, scale = attn_backend.forward(q=q, k=q, v=q, layer=self.attn_mqa, forward_batch=forward_batch,
                                           compress_ratio=self.compress_ratio, attn_sink=attn_sink,
                                           save_kv_cache=False, cos_sin_cache=self.mega_cos_sin_cache)
    finally:
        _ARM.reset(token)
    st = _STASH.pop() if _STASH else None
    _STASH.clear()
    woa = _g["dv4"]._fp8_wo_a_einsum
    if st is None:   # not split (SWA-only layer, unverified or refused key): _mega_core's own tail, line for line
        n = data.shape[0]
        woa(data[:, :groups], scale[:, :groups], *self.mega_wo_a, recipe=(1, 1, 32), out=(y.view(n, groups, -1), y_sf))
        return y, y_sf
    if data.data_ptr() != st["out"].data_ptr() or scale.data_ptr() != st["full"].data_ptr() or data.shape[0] != rows:
        _off("the attention backend did not hand back the split call's own buffers")
        _fill(st)
        n = data.shape[0]
        woa(data[:, :groups], scale[:, :groups], *self.mega_wo_a, recipe=(1, 1, 32), out=(y.view(n, groups, -1), y_sf))
        return y, y_sf
    try:
        _split_woa(st, self.mega_wo_a, y, y_sf, plan, q.device)
    except Exception as e:  # noqa: BLE001  (joined already; the unsplit wo_a rewrites every byte of y and y_sf)
        _off(f"the split wo_a raised ({type(e).__name__}: {str(e)[:200]})")
        _fill(st)
        woa(data[:, :groups], st["full"][:, :groups], *self.mega_wo_a, recipe=(1, 1, 32),
            out=(y.view(rows, groups, -1), y_sf))
        return y, y_sf
    if verdict is None:   # eager first use of this bucket: the unsplit wo_a on the same inputs
        _fill(st)
        ry, rsf = _g["mhc"].empty_deepgemm_mxfp8_pair(rows, groups * self.o_lora_rank, q.device)
        woa(data[:, :groups], st["full"][:, :groups], *self.mega_wo_a, recipe=(1, 1, 32),
            out=(ry.view(rows, groups, -1), rsf))
        same = bool(torch.equal(y.view(torch.uint8), ry.view(torch.uint8))) and bool(torch.equal(y_sf, rsf))
        _g["woa"][wkey] = same
        _say(f"AWS wo_a first-use check {'OK' if same else 'DIFFERENT (unsplit)'}: rows {rows}, split at {plan.n1}, "
             f"side {plan.side_sms} SMs")
        if not same:
            return ry, rsf
    return y, y_sf


# ------------------------------------------------------------------------------------------------- install
def _weights(dev, interleave_output_columns, quantize_wo_a):
    """AWS_ROTATE synthetic wo_a weights in the served form: the engine's own column interleave and fp8 quantizer,
    exactly as MQALayer builds mega_wo_a from its bf16 wo_a."""
    out = []
    for i in range(AWS_ROTATE):
        g = torch.Generator(device=dev)
        g.manual_seed(4100 + i)
        w = torch.randn((_GROUPS, _RANK, _K), generator=g, device=dev, dtype=torch.bfloat16) * 0.05
        out.append(tuple(quantize_wo_a(interleave_output_columns(w))))
        del w
    return out


def _unsplit_chain(vattn, a, wo_a, rows, dev):
    """The served unsplit form: VATTN over all rows, then the engine's wo_a into the engine's fp8 pair."""
    got = vattn._launch(*a)
    if got is None:
        raise RuntimeError("VATTN refused the unsplit call")
    out, sf = got
    y, y_sf = _g["mhc"].empty_deepgemm_mxfp8_pair(rows, _GROUPS * _RANK, dev)
    _g["dv4"]._fp8_wo_a_einsum(out[:, :_GROUPS], sf[:, :_GROUPS], *wo_a, recipe=(1, 1, 32),
                               out=(y.view(rows, _GROUPS, -1), y_sf))
    return {"out": out, "sf": sf, "y": y, "ysf": y_sf}


def _split_chain(vattn, a, wo_a, plan, dev):
    """The served split form: vattn._split_launch, then _split_woa (the very functions the served path calls)."""
    st = vattn._split_launch(*a, plan.n1)
    if not isinstance(st, dict):
        raise RuntimeError(f"VATTN refused the split call ({st})")
    y, y_sf = _g["mhc"].empty_deepgemm_mxfp8_pair(plan.rows, _GROUPS * _RANK, dev)
    _split_woa(st, wo_a, y, y_sf, plan, dev)
    st.update(y=y, ysf=y_sf)
    return st


def _diff(u, s, n1):
    """The names of the outputs whose bytes differ between the unsplit (u) and split (s) chains."""
    bad = []
    if not torch.equal(u["out"].view(torch.uint8), s["out"].view(torch.uint8)):
        bad.append("out_fp8")
    if not (torch.equal(u["sf"][:n1], s["sf1"]) and torch.equal(u["sf"][n1:], s["sf2"])):
        bad.append("out_sf")
    if not torch.equal(u["y"].view(torch.uint8), s["y"].view(torch.uint8)):
        bad.append("y")
    if not torch.equal(u["ysf"], s["ysf"]):
        bad.append("y_sf")
    return bad


def _graph(fn):
    """A private CUDA graph of fn() (one eager run on a side stream first, as VATTN's speed check does); returns the
    graph and the outputs of the captured run, which every replay rewrites."""
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        fn()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        outs = fn()
    return graph, outs


def _timed(graph):
    graph.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    graph.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _prove(vattn, plan, dev, weights):
    """(ok, what) for one bucket: VATTN vs the engine's kernel at plan.rows, split vs unsplit chains eager and from
    captured graphs (bitwise), then the speed gate on those graphs."""
    rows, n1 = plan.rows, plan.n1
    scale = 512 ** -0.5
    sets = [vattn._synth(rows, 1, True, dev, seed=rows * 13 + i, edge=(i == 0)) for i in range(AWS_ROTATE)]
    args = []
    for x in sets:
        if not vattn._gates(x["q"], x["kv"], x["idx"], x["pos"], x["cs"], x["sink"], x["tl"], x["ekv"], x["eidx"],
                            x["etl"]):
            return False, "a synthetic input is outside VATTN's gates"
        args.append((x["q"], x["kv"], x["idx"], x["pos"], x["cs"], x["sink"], x["tl"], x["ekv"], x["eidx"], x["etl"],
                     scale))
    ref = vattn._stock_call(sets[0], scale)
    got = vattn._launch(*args[0])
    if got is None or not (torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8))
                           and torch.equal(got[1], ref[1])):
        return False, f"VATTN differs from the engine's kernel at {rows} rows (edge set)"
    del ref, got
    for i, a in enumerate(args):
        bad = _diff(_unsplit_chain(vattn, a, weights[i], rows, dev), _split_chain(vattn, a, weights[i], plan, dev), n1)
        if bad:
            return False, f"eager set {i}: {', '.join(bad)} differ"
    flush = torch.empty(AWS_FLUSH_BYTES, dtype=torch.uint8, device=dev)

    def chains(split):
        outs = []
        for i, a in enumerate(args):
            flush.zero_()
            outs.append(_split_chain(vattn, a, weights[i], plan, dev) if split
                        else _unsplit_chain(vattn, a, weights[i], rows, dev))
        return outs

    g_u, o_u = _graph(lambda: chains(False))
    g_s, o_s = _graph(lambda: chains(True))
    t_u, t_s = float("inf"), float("inf")
    for _ in range(7):                                # interleaved rounds, best of each
        t_u = min(t_u, _timed(g_u))
        t_s = min(t_s, _timed(g_s))
    g_u.replay()
    g_s.replay()
    torch.cuda.synchronize(dev)
    for i in range(len(args)):
        bad = _diff(o_u[i], o_s[i], n1)
        if bad:
            return False, f"captured set {i}: {', '.join(bad)} differ"
    n = len(args)
    saving = (t_u - t_s) / n
    what = (f"{saving:+.1f} us/layer (captured, L2 flushed: split {t_s / n:.1f} vs unsplit {t_u / n:.1f} us a layer "
            f"incl. the flush)")
    del g_u, g_s, o_u, o_s, flush, sets, args
    return saving >= AWS_MARGIN_US, what


def _install():
    from bundle_v9.dec import vattn

    if vattn._g["state"] != "on":
        return "VATTN is off (AWS splits VATTN's own launch)"
    if torch.cuda.is_current_stream_capturing() or not torch.cuda.is_available():
        return "install ran inside a capture or without CUDA"
    from sglang.srt.runtime_context import get_disagg

    try:
        role = get_disagg().disaggregation_mode
    except ValueError:                                # no engine context published (the 1-GPU rig): decode-like
        role = None
    if role is not None and role not in AWS_ROLES:
        return f"the {role} role (AWS serves {AWS_ROLES} only)"
    import deep_gemm
    from sglang.srt.layers.attention.deepseek_v4_mega_backend import interleave_output_columns, quantize_wo_a
    from sglang.srt.models import deepseek_v4 as dv4
    from sglang.srt.models import deepseek_v4_mhc as mhc

    if not (callable(getattr(deep_gemm, "get_num_sms", None)) and callable(getattr(deep_gemm, "set_num_sms", None))):
        return "deep_gemm has no get_num_sms / set_num_sms"
    if not callable(getattr(mhc, "empty_deepgemm_mxfp8_scales", None)):
        return "deepseek_v4_mhc has no empty_deepgemm_mxfp8_scales"
    if not vattn.pins_match("aws", PINS, {"_mega_core": dv4.MQALayer._mega_core,
                                          "_fp8_wo_a_einsum": dv4._fp8_wo_a_einsum}):
        return "engine pins mismatch"
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = int(torch.cuda.get_device_properties(dev).multi_processor_count)
    dg_sms = int(deep_gemm.get_num_sms())
    n1 = sms - AWS_RESERVE_SMS
    buckets = sorted({int(r) for r in AWS_BUCKETS if int(r) > sms and int(r) - n1 >= AWS_MIN_TAIL})
    if not buckets:
        return f"no bucket of {AWS_BUCKETS} is above {sms} rows with >= {AWS_MIN_TAIL} rows past n1 {n1}"
    side_sms = min(min(sms - (r - n1) - AWS_RESERVE_SMS for r in buckets), dg_sms)
    side_sms -= side_sms % 2
    if side_sms < AWS_MIN_SIDE_SMS:
        return f"the side wo_a would get {side_sms} SMs (< {AWS_MIN_SIDE_SMS}) at {max(buckets)} rows"
    _g.update(dv4=dv4, mhc=mhc, dg=deep_gemm, cls=dv4.MQALayer, orig=dv4.MQALayer._mega_core)
    _g["side"][dev.index] = torch.cuda.Stream(dev)
    began = time.monotonic()
    weights = _weights(dev, interleave_output_columns, quantize_wo_a)
    for rows in buckets:
        if time.monotonic() - began > AWS_BUDGET_S:
            _g["report"][rows] = f"not tried (install past {AWS_BUDGET_S:.0f} s)"
            continue
        plan = Plan(rows, n1, side_sms)
        try:
            ok, what = _prove(vattn, plan, dev, weights)
        except Exception as e:  # noqa: BLE001
            ok, what = False, f"raised ({type(e).__name__}: {str(e)[:200]})"
        _g["report"][rows] = ("ON " if ok else "off ") + what
        if ok:
            _g["plans"][rows] = plan
    del weights
    elapsed = time.monotonic() - began
    return f"n1 {n1}, side {side_sms} SMs (device {sms}, DeepGEMM {dg_sms}), install {elapsed:.1f} s"


def install():
    """Pins, plans, per-bucket proof and speed gate; once per process, after VATTN's install. Never raises."""
    if _g["state"] is not None:
        return
    _g["state"] = "off"
    try:
        why = _install() if AWS_ON else "AWS_ON = False"
    except Exception as e:  # noqa: BLE001  (any failure: AWS off, VATTN's unsplit call serves)
        why = f"install raised ({type(e).__name__}: {str(e)[:300]})"
        _g["plans"] = {}
    finally:
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    report = "; ".join(f"{rows}: {what}" for rows, what in sorted(_g["report"].items()))
    if _g["plans"]:
        _g["state"] = "on"
        _say(f"AWS ON: target-verify buckets {sorted(_g['plans'])} split at row {next(iter(_g['plans'].values())).n1} "
             f"with the first wave's wo_a on a side stream; {why}; per bucket: {report}")
    else:
        _say(f"AWS off: {why}" + (f"; per bucket: {report}" if report else "") +
             "; VATTN's unsplit call and the engine's _mega_core serve")
