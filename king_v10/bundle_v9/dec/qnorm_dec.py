"""king-v2e QNORM-DEC: the decode / verify attention q-norm + MXFP8 quantization on the king's own PFUSE CUDA kernel.

Stock (and the king) run every decode q-norm through SGLang's Triton `_rmsnorm_mxfp8_kernel` (one 256-thread program per
row plus ten pad-scale programs; ~2.6 us a call, 40 calls a verify step, on the main stream before wq_b). The king
already ships a CUDA replacement, `pfuse::qnorm_mxfp8_kernel` (cuda_prefill_fuse.py; a warp per row, the Triton
reduction tree replayed lane for lane), but serves it only on uncaptured extend forwards (prefill_fuse.rmsnorm_mxfp8).
Here index_streams._rmsnorm_mxfp8, the hook the king installs for decode / verify forwards, calls that same compiled
kernel (`prefill_fuse.module().qnorm`, the king's launcher, two rows a warp; no new C++ is compiled) for the layers'
own q_norm weights whose bytes were proven at prepare, and stock's Triton kernel for everything else.

Exactness. prepare() compares y (bf16), y_q (e4m3) and y_sf (the 128x4-swizzled UE8M0 bytes, pad rows included) byte
for byte against the stock function itself, with every output buffer pre-filled with a non-zero pattern, on every
layer's q_norm weight and eps: layer 0 at 37 row counts (1..320, every 32/64/128 edge), layers 1..39 at 6, 48, 240 and
288 rows, each in three input families (gaussian x3, gaussian x3 in a 1856-wide strided buffer, per-row magnitudes
1e-3..1e3 with zero, -0 and single-outlier rows). Only a full pass publishes the weights; any mismatch, exception or
missing fact leaves the stock call everywhere. The proven domain is finite inputs.

Capture. The served call allocates its three outputs with torch.empty exactly like stock (the graph pool under capture),
does no host sync, and is guarded on the Python side so the C++ RuntimeChecks cannot fire; a raise inside the call
(not expected) clears the proven set and falls back to stock for this and every later call.

king-v6 QNORM-PDL (IDEAS_dec_rest DR3, glue_pdl.QNORM_PDL_ON). The served kernel is first the PDL copy of the same PFUSE
kernel (cuda_glue_pdl.py: weight before griddepcontrol.wait, x after it, wq_b released after the wait, programmatic
launch, glue_pdl.QNORM_PDL_ROWS rows a warp), proven by the same 579-case comparison above at that launch form. If the
glue module or that proof fails, the king's PFUSE kernel is proven and serves as in king-v2e. On the 8 indexer layers
the main stream joins the side stream's indexer wq_b right after this call (index_streams._rmsnorm_mxfp8), so the main
wq_b has no programmatic q-norm edge there (VERIFY_dec_rest DR3); the in-edge from wqkv_a and the side-stream edge stay.
"""
from __future__ import annotations

import sys

import torch

# The lead's switch. True: decode / verify q-norms run the king's PFUSE kernel where proven. False: stock Triton.
QNORM_DEC_ON = True

Q_LORA = 1280
MAX_ROWS = 320
FULL_ROWS = (1, 2, 3, 5, 6, 8, 16, 24, 31, 32, 33, 48, 63, 64, 65, 96, 120, 127, 128, 129, 144, 160, 191, 192, 193,
             200, 216, 240, 255, 256, 257, 264, 280, 287, 288, 300, 320)
LAYER_ROWS = (6, 48, 240, 288)
FAMILIES = 3
STRIDED_WIDTH = 1856
SEED = 20261106

_PROVEN: dict = {}
_KERNEL: list = []  # king-v6: [the served qnorm launcher] (the PDL copy's or the king's PFUSE module's)
_STATE = {"on": False, "why": "not prepared"}
_WARNED = [False]


def serves(x, weight, eps) -> bool:
    if not _PROVEN or type(eps) is not float:
        return False
    proven = _PROVEN.get(weight.data_ptr())
    return (proven is not None and proven == eps
            and isinstance(x, torch.Tensor) and x.dim() == 2 and x.is_cuda and x.device == weight.device
            and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= MAX_ROWS and x.shape[1] == Q_LORA
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and (x.shape[0] == 1 or x.stride(0) >= Q_LORA)
            and x.data_ptr() % 16 == 0
            and weight.dtype == torch.bfloat16 and weight.dim() == 1 and weight.shape[0] == Q_LORA
            and weight.is_contiguous() and weight.data_ptr() % 16 == 0)


def norm(x, weight, eps):
    """(y, y_q, y_sf) from the served PFUSE qnorm kernel (the PDL copy or the king's) when it serves this call;
    None means: call stock."""
    try:
        if not serves(x, weight, eps):
            return None
    except Exception:
        return None
    try:
        kernel = _KERNEL[0]
        m = x.shape[0]
        y = torch.empty((m, Q_LORA), dtype=torch.bfloat16, device=x.device)
        q = torch.empty((m, Q_LORA), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((Q_LORA // 32) * ((m + 127) // 128) * 128, dtype=torch.uint8, device=x.device)
        kernel(x, weight, y, q, s, eps)
    except Exception as error:
        _PROVEN.clear()
        mark(False, f"disabled at run time, stock Triton q-norm from here on: {type(error).__name__}: {str(error)[:300]}")
        if not _WARNED[0]:
            _WARNED[0] = True
            print(line(), file=sys.stderr, flush=True)
        return None
    return y, q, s


def _input(rows: int, family: int, gen, device) -> torch.Tensor:
    if family == 1:
        wide = torch.randn((rows, STRIDED_WIDTH), device=device, generator=gen, dtype=torch.float32) * 3
        return wide.to(torch.bfloat16)[:, :Q_LORA]
    x = torch.randn((rows, Q_LORA), device=device, generator=gen, dtype=torch.float32) * 3
    if family == 2:
        exponent = torch.rand((rows, 1), device=device, generator=gen, dtype=torch.float32) * 6 - 3
        x = x * torch.pow(10.0, exponent)
        x[rows // 2].zero_()
        if rows > 2:
            x[rows // 3].fill_(-0.0)
        if rows > 3:
            x[rows - 1, rows % Q_LORA] = 30000.0
    return x.to(torch.bfloat16)


def _compare(module, stock, x, weight, eps: float, where: str) -> None:
    rows = x.shape[0]
    want = stock(x, weight, eps)
    y = torch.full((rows, Q_LORA), float("nan"), dtype=torch.bfloat16, device=x.device)
    q = torch.empty((rows, Q_LORA), dtype=torch.float8_e4m3fn, device=x.device)
    q.view(torch.uint8).fill_(0x7F)
    s = torch.full(((Q_LORA // 32) * ((rows + 127) // 128) * 128,), 0xA5, dtype=torch.uint8, device=x.device)
    module.qnorm(x, weight, y, q, s, eps)
    if len(want) != 3:
        raise RuntimeError(f"stock rmsnorm_mxfp8 returned {len(want)} outputs, not 3")
    for name, a, b in zip(("y", "y_q", "y_sf"), (y, q, s), want):
        if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride():
            raise RuntimeError(f"{where}: {name} is {tuple(a.shape)} {a.dtype} {a.stride()}, stock "
                               f"{tuple(b.shape)} {b.dtype} {b.stride()}")
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise RuntimeError(f"{where}: {name} bytes differ from stock")


def _prove(model, module) -> int:
    from bundle_v9.dec import index_streams

    stock = index_streams._STOCK_NORM
    if stock.__module__ != "sglang.kernels.ops.layernorm.rmsnorm_mxfp8" or stock.__qualname__ != "rmsnorm_mxfp8":
        raise RuntimeError(f"the reference is not stock's rmsnorm_mxfp8 ({stock.__module__}.{stock.__qualname__})")
    norms = []
    for index, layer in enumerate(model.layers):
        q_norm = getattr(layer.self_attn, "q_norm", None)
        weight = getattr(q_norm, "weight", None)
        eps = getattr(q_norm, "variance_epsilon", None)
        if not (isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16 and tuple(weight.shape) == (Q_LORA,)
                and weight.is_contiguous() and weight.is_cuda and weight.data_ptr() % 16 == 0 and type(eps) is float):
            raise RuntimeError(f"layer {index}'s attention q_norm is not a contiguous bf16 ({Q_LORA},) weight with a "
                               f"float eps")
        norms.append((index, weight, eps))
    if not norms:
        raise RuntimeError("the model has no layers")
    device = norms[0][1].device
    gen = torch.Generator(device=device).manual_seed(SEED)
    found: dict = {}
    cases = 0
    with torch.no_grad():
        for index, weight, eps in norms:
            if found.get(weight.data_ptr(), eps) != eps:
                raise RuntimeError(f"layer {index} shares a q_norm weight with another eps")
            for rows in (FULL_ROWS if index == 0 else LAYER_ROWS):
                for family in range(FAMILIES):
                    x = _input(rows, family, gen, device)
                    _compare(module, stock, x, weight, eps, f"layer {index} rows {rows} family {family}")
                    cases += 1
            found[weight.data_ptr()] = eps
        torch.cuda.synchronize(device)
    _PROVEN.clear()
    _PROVEN.update(found)
    return cases


def mark(on: bool, why: str) -> None:
    _STATE["on"] = bool(on)
    _STATE["why"] = why


def line() -> str:
    return f"bundle-v9 king-v2e QNORM-DEC: {'ON' if _STATE['on'] else 'off'} ({_STATE['why']})"


def _prepare_pdl(model) -> bool:
    """king-v6 QNORM-PDL: prove and pick the PDL copy. Never raises; False leaves the king-v2e kernel to be proven."""
    from bundle_v9.dec import glue_pdl

    if not glue_pdl.QNORM_PDL_ON:
        glue_pdl.mark("QNORM-PDL", False, "switched off by glue_pdl.QNORM_PDL_ON; QNORM-DEC on the king's PFUSE kernel")
        return False
    try:
        module = glue_pdl.module()
        cases = _prove(model, module)
    except Exception as error:
        _PROVEN.clear()
        glue_pdl.mark("QNORM-PDL", False, f"FAIL, QNORM-DEC on the king's PFUSE kernel: {type(error).__name__}: "
                                          f"{str(error)[:300]}")
        return False
    _KERNEL[:] = [module.qnorm]
    glue_pdl.mark("QNORM-PDL", True, f"decode / verify q-norm on the PDL copy of the king's PFUSE qnorm "
                                     f"(kRows {glue_pdl.QNORM_PDL_ROWS}; weight before griddepcontrol.wait, x after it, "
                                     f"wq_b released after the wait); bytes proven against stock on {cases} cases over "
                                     f"{len(_PROVEN)} q_norm weights")
    mark(True, f"decode / verify q-norm at 1..{MAX_ROWS} rows on the PDL copy of the king's PFUSE qnorm kernel "
               f"(QNORM-PDL); bytes proven against stock on {cases} cases over {len(_PROVEN)} q_norm weights")
    return True


def prepare(model) -> None:
    from bundle_v9.dec import glue_pdl

    _PROVEN.clear()
    _KERNEL.clear()
    if not QNORM_DEC_ON:
        mark(False, "switched off by qnorm_dec.QNORM_DEC_ON; stock Triton q-norm")
        glue_pdl.mark("QNORM-PDL", False, "QNORM-DEC is switched off; stock Triton q-norm")
    elif not _prepare_pdl(model):
        try:
            from bundle_v9.dec import prefill_fuse

            module = prefill_fuse.module()
            cases = _prove(model, module)
            _KERNEL[:] = [module.qnorm]
            mark(True, f"decode / verify q-norm on the king's PFUSE qnorm kernel at 1..{MAX_ROWS} rows; bytes proven "
                       f"against stock on {cases} cases over {len(_PROVEN)} q_norm weights")
        except Exception as error:
            _PROVEN.clear()
            _KERNEL.clear()
            mark(False, f"FAIL, stock Triton q-norm: {type(error).__name__}: {str(error)[:300]}")
    print(line(), file=sys.stderr, flush=True)
    glue_pdl.say("QNORM-PDL")
