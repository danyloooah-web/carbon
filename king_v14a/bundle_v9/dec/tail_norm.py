"""king-v13s TAIL-NORM (struct/king_lane/timeline_v10/decode_path.md #5; v13s/S2.md).

The decode step's tail before the LM head runs three kernels: the final seam post (SEAM-POST at 149..288 rows, stock's
mhc_post_split_h below; 4.64 us at 240 rows), stock's Triton _mhc_combine_kernel (2.42 us) and flashinfer's CuTe-DSL
RMSNorm (3.33 us), plus three launch gaps. cuda_tail_norm.py computes all three in one launch, one CTA a row:
- tier A: R' (pre_hc_head), h = mhc_combine(R', pre) and y = RMSNorm(h) (the norm's reduction tree reproduced);
- tier B: R' and h; flashinfer's RMSNorm still runs on h.
Both write R' and h, so whatever consumes them gets valid bytes even when a later handoff does not match.

Wiring (decode / target-verify forwards only, inside index_streams.routed(), restored in finally):
- deepseek_v4_mhc._post_fusion (forward_model sets it to mhc_floor._post_fusion) is wrapped: the call with
  `next is None` and `hc is model.layers[-1].ffn_hc` (the final read's FFN post) runs the fused kernel on the
  coefficients' pre / post / comb and returns HcState(R', pre), what mhc_floor's seam branch returns; every other call
  goes to mhc_floor unchanged.
- sglang.kernels.ops.layernorm.mhc.mhc_combine (DeepseekV4Model._forward_impl imports it at call time) returns the
  stashed h only for x_flat = R'.flatten(1) (same data_ptr, shape, strides, dtype), the same `pre` object, hc 4 and a
  bf16 out dtype; anything else runs stock.
- tier A: sglang.srt.layers.layernorm.rmsnorm (what RMSNorm.forward_cuda calls) returns the stashed y only for
  (h itself, the final norm's weight by data_ptr, its eps), positional, no out / pdl argument; anything else runs stock.
Exactness: prepare proves R', h and y byte for byte against stock's chain (mhc_post_split_h, which SEAM-POST equals;
stock mhc_combine; the model's own norm module) on NaN-filled outputs, every decode graph row count 6..288 plus odd
counts, two input families (ordinary, and scaled with zeros / -0 / mixed signs), the final norm's real weight and eps.
The combine form and the norm's (reduction, division) form are chosen by that proof among 3 and 6 compiled forms; a
form is served only if it matches every case; no match turns the norm (or the lever) off. A probe checks that the
norm module really calls layernorm.rmsnorm(h, weight, eps) once (else tier A is off). Engine segments pinned below.
Bounded (BUDGET_S), eager, no collective. Any failure: king_v10b's tail, unchanged.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
from functools import lru_cache, partial

import torch

from bundle_v9.dec import cuda_tail_norm

# The lead's switches. TAIL_NORM_ON False: king_v10b's tail. TAIL_NORM_FUSE_NORM False: at most tier B.
TAIL_NORM_ON = True
TAIL_NORM_FUSE_NORM = True
ROWS = (1, 288)
PROOF_ROWS = tuple(sorted(set(range(6, 289, 6)) | {1, 2, 3, 5, 7, 148, 149, 150, 287}))
SENSE_ROWS = 512  # one extra norm-sensitivity case (family 2); the kernel is one CTA a row, any count runs
HIDDEN, MULT = 5120, 4
BUDGET_S = 120.0
SEED = 20261114
PREFIX = "bundle-v9 king-v13s"
PINS = (
    ("v4", "DeepseekV4Model._forward_impl", "d026453719fc934b"),
    ("mhc", "run_moe_post", "62431f4cc899ea63"),
    ("mhc", "post", "5873c3f00fb90418"),
    ("mhc", "HcState.materialized", "22e2501ca0ef9145"),
    ("boundary", "FinalRead._complete", "e45bb4a5d850303b"),
    ("mhc_ops", "mhc_combine", "d8f1c2884648d1de"),
    ("mhc_ops", "_mhc_combine_kernel", "501e5cc86b05058b"),
    ("split_h", "mhc_post_split_h", "77b7c7a5f2b81cd7"),
    ("split_h", "_mhc_post_split_h_kernel", "ddb54b08fa9edda6"),
    ("layernorm", "RMSNorm.forward_cuda", "1d2abfb88cc802f9"),
)

_ON: list = [False]
_CFG: dict = {}  # comb, red, div, norm (0 / 1), final_fn, weight, w_ptr, eps, probe_ok
_STOCK: dict = {}
_STASH: list = [None]
_NORM: list = [None]
_STATE = {"on": False, "why": "not prepared"}
_COUNT = {"fused": 0, "combine_hit": 0, "norm_hit": 0, "combine_miss": 0, "norm_miss": 0, "skip": 0}
_SAID: set = set()


def _mark(on: bool, why: str) -> None:
    _STATE["on"] = bool(on)
    _STATE["why"] = why


def line() -> str:
    from bundle_v9.dec import idx_early

    return f"{PREFIX} TAIL-NORM: {'ON' if _STATE['on'] else 'off'} ({idx_early.rank_tag()}{_STATE['why']})"


def say() -> None:
    print(line(), file=sys.stderr, flush=True)


def _once(key: str, text: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(f"{PREFIX} TAIL-NORM: {text}", file=sys.stderr, flush=True)


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


@lru_cache(maxsize=None)
def _module():
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data = cuda_tail_norm.SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-tail-norm", digest)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_tail_norm.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return load_jit("dec_tail_norm", digest, cuda_files=[path], cuda_wrappers=[("run", cuda_tail_norm.ENTRY)])


# ------------------------------------------------------------------------------------------------------- runtime
def _launch(y, residual, post_mix, comb, pre, comb_form: int, red: int, div: int, norm: int):
    updated = torch.empty_like(residual)
    h = torch.empty((y.shape[0], HIDDEN), dtype=torch.bfloat16, device=y.device)
    out = torch.empty_like(h) if norm else h
    _module().run(y, residual, post_mix, comb, pre, _CFG["weight"], updated, h, out, _CFG["eps"], comb_form, red, div,
                  norm)
    return updated, h, (out if norm else None)


def _domain(y, residual, coefficients) -> bool:
    try:
        pre, post_mix, comb = coefficients
        rows = y.shape[0]
        return (ROWS[0] <= rows <= ROWS[1] and y.dim() == 2 and tuple(y.shape) == (rows, HIDDEN)
                and y.dtype == residual.dtype == torch.bfloat16 and tuple(residual.shape) == (rows, MULT, HIDDEN)
                and pre.dtype == post_mix.dtype == comb.dtype == torch.float32
                and tuple(pre.shape) == tuple(post_mix.shape) == (rows, MULT) and tuple(comb.shape) == (rows, MULT, MULT)
                and all(t.is_cuda and t.is_contiguous() and t.device == y.device for t in (y, residual, pre, post_mix, comb))
                and y.data_ptr() % 16 == 0 and residual.data_ptr() % 16 == 0
                and _CFG["weight"].device == y.device)
    except Exception:
        return False


def _is_final(hc) -> bool:
    """The last layer's FFN hyper-connection (the final read's post), by its fn parameter."""
    fn = getattr(hc, "fn", None)
    return isinstance(fn, torch.Tensor) and fn.data_ptr() == _CFG.get("final_fn")


def _post_fusion(inner, hc, y, residual, coefficients, next):
    if next is None and _ON[0] and _is_final(hc) and _domain(y, residual, coefficients):
        from sglang.srt.models.deepseek_v4_mhc import HcState

        pre, post_mix, comb = coefficients
        try:
            updated, h, out = _launch(y, residual, post_mix, comb, pre, _CFG["comb"], _CFG["red"], _CFG["div"],
                                      _CFG["norm"])
        except Exception as error:
            _ON[0] = False
            _mark(False, f"disabled at run time after a launch error, king_v10b's tail from here on: "
                         f"{type(error).__name__}: {str(error)[:300]}")
            say()
            return inner(hc, y, residual, coefficients, next)
        _COUNT["fused"] += 1
        _STASH[0] = (updated, pre, h, out)
        _NORM[0] = None
        return HcState(updated, pre)
    return inner(hc, y, residual, coefficients, next)


def _combine(x_flat, pre, hc, out_dtype):
    stash, _STASH[0] = _STASH[0], None
    if stash is not None:
        updated, spre, h, out = stash
        rows = updated.shape[0]
        if (pre is spre and hc == MULT and out_dtype == torch.bfloat16 and x_flat.dtype == torch.bfloat16
                and x_flat.data_ptr() == updated.data_ptr() and tuple(x_flat.shape) == (rows, MULT * HIDDEN)
                and tuple(x_flat.stride()) == (MULT * HIDDEN, 1) and x_flat.device == updated.device):
            _COUNT["combine_hit"] += 1
            _NORM[0] = (h, out) if out is not None else None
            return h
        _COUNT["combine_miss"] += 1
        _once("combine_miss", "the final mhc_combine was not called on the fused R' / pre; stock combine served it "
                              "(shown once)")
    return _STOCK["combine"](x_flat, pre, hc, out_dtype)


def _rmsnorm(*args, **kwargs):
    stash, _NORM[0] = _NORM[0], None
    if stash is not None:
        h, out = stash
        if (len(args) == 3 and not kwargs and args[0] is h and args[1].dtype == torch.bfloat16
                and args[1].data_ptr() == _CFG["w_ptr"] and tuple(args[1].shape) == (HIDDEN,)
                and float(args[2]) == _CFG["eps"]):
            _COUNT["norm_hit"] += 1
            return out
        _COUNT["norm_miss"] += 1
        _once("norm_miss", "the final norm was not called on the fused h with its weight / eps; stock rmsnorm served "
                           "it (shown once)")
    return _STOCK["rmsnorm"](*args, **kwargs)


@contextlib.contextmanager
def routed():
    """TAIL-NORM's swaps for one decode / verify forward (entered by index_streams.routed). A no-op when off."""
    if not _ON[0]:
        yield
        return
    held = None  # (deepseek_v4_mhc, mhc_ops, layernorm, inner _post_fusion, rmsnorm swapped) while the swaps stand
    try:
        from sglang.kernels.ops.layernorm import mhc as mhc_ops
        from sglang.srt.layers import layernorm
        from sglang.srt.models import deepseek_v4_mhc as mhc

        from bundle_v9.dec import mhc_floor

        inner = mhc._post_fusion
        if (inner is mhc_floor._post_fusion and mhc_ops.mhc_combine is _STOCK["combine"]
                and (not _CFG["norm"] or layernorm.rmsnorm is _STOCK["rmsnorm"])):
            held = (mhc, mhc_ops, layernorm, inner, bool(_CFG["norm"]))
            mhc._post_fusion = partial(_post_fusion, inner)
            mhc_ops.mhc_combine = _combine
            if _CFG["norm"]:
                layernorm.rmsnorm = _rmsnorm
        else:
            _COUNT["skip"] += 1
            _once("skip", "a forward entered with _post_fusion / mhc_combine / rmsnorm not as proven (another swap "
                          "holds one); king_v10b's tail served it (shown once)")
    except Exception as error:
        _restore(held)
        held = None
        _once("enter", f"entry failed, king_v10b's tail for this forward: {type(error).__name__}: {str(error)[:200]}")
    try:
        yield
    finally:
        _restore(held)
        _STASH[0] = _NORM[0] = None


def _restore(held) -> None:
    if held is None:
        return
    mhc, mhc_ops, layernorm, inner, norm = held
    mhc._post_fusion = inner
    mhc_ops.mhc_combine = _STOCK["combine"]
    if norm:
        layernorm.rmsnorm = _STOCK["rmsnorm"]


# ------------------------------------------------------------------------------------------------------- the proof
def _case(rows: int, family: int, gen, device):
    if family == 2:
        # Norm sensitivity: post = pre = e0, comb = 0, R = 0, so R'[0] = h = y exactly; y spans 6 decades inside each
        # row, so any other order of the 5120-term sum of squares (or another division) moves rstd by ulps and flips
        # bf16 roundings of y in most rows.
        y = torch.randn((rows, HIDDEN), device=device, generator=gen) * torch.pow(
            10.0, torch.rand((rows, HIDDEN), device=device, generator=gen) * 6 - 3)
        residual = torch.zeros((rows, MULT, HIDDEN), device=device, dtype=torch.bfloat16)
        post = torch.zeros((rows, MULT), device=device)
        post[:, 0] = 1.0
        comb = torch.zeros((rows, MULT, MULT), device=device)
        pre = post.clone()
        return y.bfloat16().contiguous(), residual, post, comb, pre
    y = torch.randn((rows, HIDDEN), device=device, generator=gen) * 2
    residual = torch.randn((rows, MULT, HIDDEN), device=device, generator=gen)
    post = torch.rand((rows, MULT), device=device, generator=gen) * 2
    comb = torch.rand((rows, MULT, MULT), device=device, generator=gen)
    pre = torch.rand((rows, MULT), device=device, generator=gen)
    if family == 1:
        y = y * torch.pow(10.0, torch.rand((rows, 1), device=device, generator=gen) * 8 - 4)
        residual = residual * torch.pow(10.0, torch.rand((rows, MULT, 1), device=device, generator=gen) * 8 - 4)
        comb = comb * 2 - 1
        pre = pre * 2 - 1
        y[rows // 2].zero_()
        residual[rows // 2].zero_()
        residual[rows // 3, 1].fill_(-0.0)
        post[rows // 4].zero_()
        pre[(rows * 3) // 4].fill_(-0.0)
        if rows > 1:
            pre[rows - 1, 0] = 0.0
            y[rows - 2, :64] = -0.0
    return y.bfloat16().contiguous(), residual.bfloat16().contiguous(), post.contiguous(), comb.contiguous(), \
        pre.contiguous()


def _same16(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a.contiguous().view(torch.int16),
                                                                 b.contiguous().view(torch.int16)))


def _families(rows: int) -> tuple:
    return (0, 1) if rows in (1, 6, 144, 148, 149, 192, 240, 264, 288) else (rows % 2,)


def _prove(norm_module, device, deadline: float):
    """(comb form, (red, div) or None, cases): every case's stock reference is computed once, then each surviving form
    runs on NaN-filled outputs and is dropped at its first differing byte. Raises when no combine form survives."""
    split_h, combine = _STOCK["split_h"], _STOCK["combine"]
    gen = torch.Generator(device=device).manual_seed(SEED)
    combs = [0, 1, 2]
    norms = [(c, red, div) for c in combs for red in (0, 1) for div in (0, 1, 2, 3)] if (
        TAIL_NORM_FUSE_NORM and _CFG.get("probe_ok")) else []
    cases = 0

    def ok(y, residual, post, comb, pre, refs, comb_form: int, red: int, div: int, norm: int) -> bool:
        rows = y.shape[0]
        updated = torch.full_like(residual, float("nan"))
        h = torch.full((rows, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device)
        out = torch.full_like(h, float("nan")) if norm else h
        _module().run(y, residual, post, comb, pre, _CFG["weight"], updated, h, out, _CFG["eps"], comb_form, red, div,
                      norm)
        torch.cuda.synchronize(device)
        return _same16(updated, refs[0]) and _same16(h, refs[1]) and (not norm or _same16(out, refs[2]))

    plan = [(rows, family) for rows in PROOF_ROWS for family in _families(rows)] + [(SENSE_ROWS, 2)]
    for rows, family in plan:
        if time.monotonic() > deadline:
            raise RuntimeError(f"proof budget of {BUDGET_S:.0f} s exhausted at {rows} rows")
        y, residual, post, comb, pre = _case(rows, family, gen, device)
        r_ref = split_h(y, residual, post, comb)
        h_ref = combine(r_ref.flatten(1), pre, MULT, torch.bfloat16)
        refs = (r_ref, h_ref, norm_module(h_ref) if norms else None)
        combs = [c for c in combs if ok(y, residual, post, comb, pre, refs, c, 0, 0, 0)]
        if not combs:
            raise RuntimeError(f"no combine form matches stock's seam post + mhc_combine ({rows} rows, family "
                               f"{family})")
        norms = [v for v in norms if v[0] in combs and ok(y, residual, post, comb, pre, refs, *v, 1)]
        cases += 1
    comb_form = combs[0]
    norm_form = next(((red, div) for c, red, div in norms if c == comb_form), None)
    return comb_form, norm_form, cases


def _probe_norm(norm_module, device) -> str:
    """'' when the model's final norm calls layernorm.rmsnorm(h, weight, eps) once, positionally; else why not."""
    from sglang.srt.layers import layernorm

    calls = []
    stock = layernorm.rmsnorm

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return stock(*args, **kwargs)

    h = torch.randn((6, HIDDEN), device=device).to(torch.bfloat16)
    layernorm.rmsnorm = spy
    try:
        out = norm_module(h)
    finally:
        layernorm.rmsnorm = stock
    torch.cuda.synchronize(device)
    if len(calls) != 1:
        return f"the final norm made {len(calls)} layernorm.rmsnorm calls"
    args, kwargs = calls[0]
    if kwargs or len(args) != 3 or args[0] is not h or args[1].data_ptr() != _CFG["w_ptr"] or float(args[2]) != _CFG["eps"]:
        return "the final norm's layernorm.rmsnorm call is not (h, weight, eps)"
    if not isinstance(out, torch.Tensor) or out.dtype != torch.bfloat16 or tuple(out.shape) != (6, HIDDEN):
        return "the final norm returned no bf16 [rows, hidden] tensor"
    return ""


def prepare(model) -> None:
    """TAIL-NORM prepare (decode role, from index_streams.prepare). Never raises: king_v10b's tail on any failure."""
    _ON[0] = False
    _CFG.clear()
    if not TAIL_NORM_ON:
        _mark(False, "switched off by tail_norm.TAIL_NORM_ON; king_v10b's seam post + mhc_combine + rmsnorm")
        say()
        return
    try:
        from sglang.kernels.ops.layernorm import mhc as mhc_ops
        from sglang.kernels.ops.layernorm import mhc_post_split_h as split_h_mod
        from sglang.srt.layers import layernorm
        from sglang.srt.models import deepseek_v4, deepseek_v4_boundary, deepseek_v4_mhc

        from bundle_v9.dec import idx_early

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare requires an eager stream")
        bad = idx_early.pin_mismatches({"v4": deepseek_v4, "mhc": deepseek_v4_mhc, "boundary": deepseek_v4_boundary,
                                        "mhc_ops": mhc_ops, "split_h": split_h_mod, "layernorm": layernorm}, PINS)
        if bad:
            raise RuntimeError(f"engine source pins differ: {', '.join(bad)}")
        combine, rmsnorm, split_h = mhc_ops.mhc_combine, layernorm.rmsnorm, split_h_mod.mhc_post_split_h
        if getattr(combine, "__qualname__", "") != "mhc_combine" or getattr(split_h, "__qualname__", "") != \
                "mhc_post_split_h":
            raise RuntimeError("mhc_combine or mhc_post_split_h is not stock's")
        if not model.hc_pre_from_prev_sublayer or int(model.hc_mult) != MULT:
            raise RuntimeError("the model is not DeepSeek-V4.1's pre-from-prev mHC with 4 streams")
        norm = model.norm
        weight = norm.weight.data
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (HIDDEN,) or not weight.is_contiguous() or \
                weight.data_ptr() % 16:
            raise RuntimeError(f"the final norm weight is {weight.dtype} {tuple(weight.shape)}")
        final_hc = model.layers[-1].ffn_hc
        cfg = getattr(final_hc, "cfg", None)
        if final_hc is None or cfg is None or int(cfg.mult) != MULT or int(cfg.hidden) != HIDDEN or not cfg.pre_from_prev:
            raise RuntimeError("the last layer's FFN hyper-connection is not mult 4 / hidden 5120 / pre_from_prev")
        device = weight.device
        _STOCK.update(combine=combine, rmsnorm=rmsnorm, split_h=split_h)
        _CFG.update(weight=weight, w_ptr=weight.data_ptr(), eps=float(norm.variance_epsilon),
                    final_fn=final_hc.fn.data_ptr())
        why_not = _probe_norm(norm, device)
        _CFG["probe_ok"] = not why_not
        start = time.monotonic()
        comb_form, norm_form, cases = _prove(norm, device, start + BUDGET_S)
        elapsed = time.monotonic() - start
    except Exception as error:
        _CFG.clear()
        _mark(False, f"FAIL, king_v10b's tail: {type(error).__name__}: {str(error)[:300]}")
        say()
        return
    red, div = norm_form if norm_form is not None else (0, 0)
    _CFG.update(comb=comb_form, red=red, div=div, norm=1 if norm_form is not None else 0)
    _ON[0] = True
    if norm_form is not None:
        tier = (f"tier A: R', mhc_combine and the final RMSNorm in one launch (combine form {comb_form}, norm reduction "
                f"{('mul+add', 'fma')[red]}, division {('div.rn', 'div.full', 'div.approx', 'rcp.rn mul')[div]})")
    else:
        reason = ("switched off by TAIL_NORM_FUSE_NORM" if not TAIL_NORM_FUSE_NORM else why_not if why_not
                  else "no norm form matched flashinfer's RMSNorm on every case")
        tier = f"tier B: R' and mhc_combine in one launch (combine form {comb_form}), flashinfer's RMSNorm kept ({reason})"
    _mark(True, f"{tier} at the final seam post, {ROWS[0]}..{ROWS[1]} rows; R', h{', y' if norm_form else ''} proven "
                f"against stock's split_h + mhc_combine + the model's norm on {cases} NaN-filled cases ({elapsed:.1f} s)")
    say()
