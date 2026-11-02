"""dsv41-pd046 PGEMM v3: exact prefill GEMM routes for every dense MXFP8 linear the sweep found slow, and the MegaMoE
BLOCK_M band.

The engine serves prefill MXFP8 linears on FlashInfer's CuTe-DSL runner, tuned at startup over CuTe-DSL tactics only
(autotune_prefill_kernels), or on DeepGEMM for DeepGEMM-input linears. The g8c sweep twin (10-29, results/pd043/
g8c_pd043chk_lines.txt) timed each linear's served call against every CUTLASS tactic and four CuTe-DSL tactics, in-engine
on c29498. Every candidate was bitwise in every bucket: no tactic here splits K, so the fp32 reduction runs in K order.
ROUTES keeps one tactic a linear, the best across row bands rather than per bucket:
  - wq_b (32768 x 1280): CUTLASS 18 at >= 6144 rows, the dspd CuTe-DSL tactic at 384-6143 (pd042chk, g8c route checks;
    x0.45-0.91 of stock's time);
  - wo_b (5120 x 8192, DeepGEMM input): CUTLASS 19 at >= 16384 (x0.90-0.94); below that no tactic wins;
  - wqkv_a (1792 x 5120, DeepGEMM input): CuTe-DSL (256, 192) (2, 1) swapped at >= 1024 (x0.49 at 1-3K rows .. x0.82 at
    17K+; worst bucket x0.92);
  - the indexer's wq_b (4096 x 1280): the same tactic (x0.44-0.66; worst x0.72);
  - Engram wkv (25600 x 6144, BF16 input): CuTe-DSL (256, 256) (2, 2) (x0.80-0.94 by band; worst x0.97).
dsv41-pd062 CUBLT (generator/pd061-pd062): cuBLASLt MXFP8 (torch._scaled_mm) serves Engram wkv and the indexer's wq_b
from 1024 rows and wqkv_a at 1024-24575 rows: bitwise with the tactics above on every measured shape, x1.03-1.31.
DeepGEMM-packed scale words (4 UE8M0 bytes a word along K, column-major) are reordered into FlashInfer's 128x4-swizzled
bytes by `block_scale_interleave`; BF16 inputs are quantized by `flashinfer_mxfp8_quantize`, as stock's own path does.
MegaMoE BLOCK_M 128 (DG_MEGA_MOE_TILES) runs at 20-28K DP-global tokens, where DeepGEMM picks 240 (dsv41pd.v28's probe).

The routes bind only inside the candidate's forward (`routed`, next to v16's): the registered linears' quant methods and
the module-level DeepGEMM linear are swapped and restored in `finally`, so the audit's stock references never see them.
CHECK (the twin) compares each route's first calls a row bucket with stock's result bitwise and times both, printing to
stderr (bundle loggers never reach the engine logs).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from functools import partial

import torch

MXT = ((256, 256), (2, 1), False, False, 1)  # the dspd bundles' eager-prefill tactic
SWAP = ((256, 192), (2, 1), True, False, 1)
WIDE = ((256, 256), (2, 2), False, False, 1)
# weight (N, K) -> ((min rows, family, tactic), ...): the first entry whose min rows <= the call's rows serves it
ROUTES = {
    (32768, 1280): ((6144, "cutlass", 18), (384, "cute", MXT)),  # wq_b
    (5120, 8192): ((16384, "cutlass", 19),),  # wo_b
    (1792, 5120): ((24576, "cute", SWAP), (1024, "cublaslt", None)),  # wqkv_a (dsv41-pd062 CUBLT below 24576)
    (4096, 1280): ((1024, "cublaslt", None),),  # the indexer's wq_b (dsv41-pd062 CUBLT)
    (25600, 6144): ((1024, "cublaslt", None),),  # Engram wkv (dsv41-pd062 CUBLT)
}
MOE_TILES, MOE_TILE_TOKENS = "2,128,32,128,2", (20480, 28672)
CHECK = False  # build-time (gen_pd046 --check): the twin's bitwise + timing checks
CHECK_CALLS = 3  # per (linear, row bucket)
HIDDEN = 5120

_CUTLASS: list = []
_CUTE: list = []
_REG: dict = {}  # weight data_ptr -> (name, weight, FlashInfer-swizzled scale, routes)
_METHODS: dict = {}  # id -> the quant method of every registered linear
_STOCK_LINEAR: list = []  # [fp8.deepgemm_w8a8_mxfp8_linear_with_fallback as loaded]
_CHK: dict = {}
_FIRST: set = set()


def _say(msg: str) -> None:
    print(f"dsv41-pd046chk {msg}", file=sys.stderr, flush=True)


def _gemm(family, tactic, q, q_scale, weight, weight_scale):
    """FlashInfer's CUTLASS or CuTe-DSL MXFP8 GEMM at one tactic (mm_mxfp8's launch without its autotuner), BF16 out."""
    from flashinfer.gemm import gemm_base

    if family == "cublaslt":  # dsv41-pd062 CUBLT: cuBLASLt MXFP8 on the same swizzled scale bytes, in memory order
        return torch._scaled_mm(q, weight.t(), scale_a=_flat(q_scale).view(torch.float8_e8m0fnu),
                                scale_b=_flat(weight_scale).view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)
    if family == "cutlass":
        if not _CUTLASS:
            major, _ = torch.cuda.get_device_capability(q.device)
            _CUTLASS.append(gemm_base.get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner())
        runner = _CUTLASS[0]
    else:
        if not _CUTE:
            major, minor = torch.cuda.get_device_capability(q.device)
            _CUTE.append(gemm_base._cute_dsl_gemm_mxfp8_runner(major, minor, True, torch.bfloat16))
        runner = _CUTE[0]
    out = torch.empty((q.shape[0], weight.shape[0]), device=q.device, dtype=torch.bfloat16)
    work = gemm_base._get_cache_buf("mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, q.device)
    runner(inputs=[q, weight.t(), q_scale, weight_scale, torch.bfloat16, out, work], tactic=tactic)
    return out


def _flat(scale):
    """dsv41-pd062 CUBLT: a scale tensor's bytes in memory order (the order the CUTLASS / CuTe runners read through its
    pointer), as a 1-D uint8 view; raises if its elements do not cover one dense block."""
    s = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    if s.is_contiguous():
        return s.reshape(-1)
    span = 1 + sum((n - 1) * st for n, st in zip(s.shape, s.stride()))
    if span != s.numel():
        raise ValueError(f"dsv41-pd062 CUBLT: scale {tuple(s.shape)} / {s.stride()} is not one dense block")
    return torch.as_strided(s, (s.numel(),), (1,), s.storage_offset())


def _route(entry, rows):
    for min_rows, family, tactic in entry[3]:
        if rows >= min_rows:
            return family, tactic
    return None


def _timed(fn, reps=3):
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e) * 1e3
        best = t if best is None else min(best, t)
    return out, best


def _check(name, rows, stock_fn, route_fn, label):
    """Twin: the first CHECK_CALLS calls of (linear, row bucket): the route's output against stock's, bitwise, and the
    time of each (the first call's wall time too: a JIT inside the window shows there). Returns the route's output, or
    None once the bucket is done."""
    key = (name, rows.bit_length())
    n = _CHK.get(key, 0)
    if n >= CHECK_CALLS or torch.cuda.is_current_stream_capturing():
        return None
    _CHK[key] = n + 1
    first = ""
    if label not in _FIRST:
        _FIRST.add(label)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        route_fn()
        torch.cuda.synchronize()
        first = f" first {1e3 * (time.perf_counter() - t0):.1f} ms"
    ref, t_stock = _timed(stock_fn)
    out, t = _timed(route_fn)
    same = out.numel() == ref.numel() and torch.equal(out.view(ref.shape), ref)
    _say(f"{name} rows {rows}: stock {t_stock:.0f} us | {label} {t:.0f} us {'BITWISE' if same else 'MISMATCH'}{first}")
    return out


def qm_prefill(stock, input, weight, weight_scale, input_scale=None, bias=None, pin_tactic=None, **kwargs):
    """A registered linear's MXFP8 quant-method call in an extend forward (its w8a8_mxfp8_linear: the arguments and
    result of `stock`): on its route when one serves the row count, else `stock`. A BF16 input is quantized exactly as
    stock's flashinfer_mxfp8_blockscaled_linear does; a swizzled pair is used as given."""
    entry = _REG.get(weight.data_ptr())
    rows = input.numel() // input.shape[-1] if input.dim() else 0
    route = _route(entry, rows) if entry is not None and bias is None and not kwargs and rows else None
    if route is None:
        if pin_tactic is not None:
            kwargs["pin_tactic"] = pin_tactic
        return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias,
                     **kwargs)
    two_d = input.reshape(rows, input.shape[-1])
    if input_scale is None:
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        def run():
            q, q_scale = flashinfer_mxfp8_quantize(two_d, is_sf_swizzled_layout=True, alignment=32)
            return _gemm(*route, q, q_scale, entry[1], entry[2])
    else:
        run = partial(_gemm, *route, two_d, input_scale.view(torch.uint8), entry[1], entry[2])
    if CHECK:
        out = _check(entry[0], rows, lambda: stock(input=input, weight=weight, weight_scale=weight_scale,
                                                   input_scale=input_scale, bias=bias, pin_tactic=pin_tactic),
                     run, f"{route[0]}:{route[1]}")
        if out is not None:
            return out.view(*input.shape[:-1], weight.shape[0])
    return run().view(*input.shape[:-1], weight.shape[0])


def dg_prefill(input, weight, weight_scale, input_scale=None, bias=None, weight_scale_swizzled=None):
    """deepgemm_w8a8_mxfp8_linear_with_fallback in an extend forward: a registered linear's DeepGEMM-input pair on its
    route, the packed scale words reordered into FlashInfer's swizzled bytes; anything else on stock's."""
    from dsv41_pd001 import skfuse

    stock = partial(_STOCK_LINEAR[0], input, weight, weight_scale, input_scale=input_scale, bias=bias,
                    weight_scale_swizzled=weight_scale_swizzled)
    entry = _REG.get(weight.data_ptr())
    rows = input.numel() // input.shape[-1] if input.dim() else 0
    packed = (input_scale is not None and bias is None and input_scale.dtype == torch.int32
              and tuple(input_scale.shape) == (rows, input.shape[-1] // 128))
    route = _route(entry, rows) if entry is not None and packed and rows else None
    if route is None:
        return stock()
    two_d = input.reshape(rows, input.shape[-1])

    def run():
        q_scale = skfuse.wob_scale(input_scale)  # dsv41-pd052 WOBSF: block_scale_interleave(.contiguous()) bytes
        return _gemm(*route, two_d, q_scale, entry[1], entry[2])
    if CHECK:
        out = _check(entry[0], rows, stock, run, f"{route[0]}:{route[1]}")
        if out is not None:
            return out.view(*input.shape[:-1], weight.shape[0])
    return run().view(*input.shape[:-1], weight.shape[0])


def _in_band(forward_batch) -> bool:
    """The MoE tile band, from the all-gathered token counts every rank shares; never inside a capture."""
    from sglang.srt.layers.moe import mega_moe

    if forward_batch is None or torch.cuda.is_current_stream_capturing() or "DG_MEGA_MOE_TILES" in os.environ:
        return False
    counts = mega_moe.get_dp_global_num_tokens()
    return bool(counts) and MOE_TILE_TOKENS[0] <= int(sum(counts)) < MOE_TILE_TOKENS[1]


@contextlib.contextmanager
def moe_tiles(forward_batch):
    """DG_MEGA_MOE_TILES around one MoE call of a forward in the band (DeepGEMM reads it at the launch and in
    get_block_m_for_mega_moe, so the front's shared-expert paging agrees with the kernel)."""
    if not _in_band(forward_batch):
        yield
        return
    os.environ["DG_MEGA_MOE_TILES"] = MOE_TILES
    try:
        yield
    finally:
        os.environ.pop("DG_MEGA_MOE_TILES", None)


def _moe_wrapped(stock_forward_mega_moe):
    """mega_forward.forward_mega_moe (moe, hidden_states, forward_batch, input_ids_global) under the tile band; the MoE
    view's forward imports it from mega_forward at each call, so the module attribute is the seam."""
    def forward_mega_moe(moe, hidden_states, forward_batch=None, input_ids_global=None):
        with moe_tiles(forward_batch):
            return stock_forward_mega_moe(moe, hidden_states, forward_batch, input_ids_global=input_ids_global)
    return forward_mega_moe


@contextlib.contextmanager
def routed(view, forward_batch):
    """One extend forward: the module-level DeepGEMM linear, every registered linear's quant method and mega_forward's
    MoE entry swapped to the routes above, all restored on exit; other forwards run untouched. LEAN's MoE replica
    bypasses the entry and scopes `moe_tiles` itself (gen_pd042's lean_layers edit)."""
    from sglang.srt.layers.quantization import fp8 as fp8_linear

    from dsv41_pd001 import mega_forward

    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend() or torch.cuda.is_current_stream_capturing() or not _STOCK_LINEAR:
        yield
        return
    methods = [(method, method.w8a8_mxfp8_linear) for method in _METHODS.values()]
    stock_moe = mega_forward.forward_mega_moe
    fp8_linear.deepgemm_w8a8_mxfp8_linear_with_fallback = dg_prefill
    mega_forward.forward_mega_moe = _moe_wrapped(stock_moe)
    try:
        for method, stock in methods:
            method.w8a8_mxfp8_linear = partial(qm_prefill, stock)
        yield
    finally:
        fp8_linear.deepgemm_w8a8_mxfp8_linear_with_fallback = _STOCK_LINEAR[0]
        mega_forward.forward_mega_moe = stock_moe
        for method, stock in methods:
            method.w8a8_mxfp8_linear = stock


def _register(name, linear, required: bool) -> None:
    weight = getattr(linear, "weight", None) if linear is not None else None
    scale = getattr(linear, "weight_scale_inv_swizzled", None) if linear is not None else None
    ok = (weight is not None and scale is not None and weight.dtype == torch.float8_e4m3fn and weight.dim() == 2
          and tuple(weight.shape) in ROUTES)
    if not ok:
        if required:
            raise TypeError(f"dsv41-pd046 PGEMM: {name} is not an MXFP8 linear of a routed shape with FlashInfer scales")
        return
    _REG[weight.data_ptr()] = (name, weight, scale.t() if scale.ndim == 2 else scale, ROUTES[tuple(weight.shape)])
    method = getattr(linear, "quant_method", None)
    if callable(getattr(method, "w8a8_mxfp8_linear", None)):
        _METHODS.setdefault(id(method), method)


def prepare(module) -> None:
    """Capture stock's DeepGEMM linear, register every layer's wq_b, wo_b, wqkv_a and (where present) the indexer's wq_b
    and the Engram wkv, and run each route's tactic once before any graph capture. Raises otherwise."""
    from sglang.srt.layers.quantization import fp8 as fp8_linear

    from dsv41_pd001 import mega_forward

    if not _STOCK_LINEAR:
        fn = fp8_linear.deepgemm_w8a8_mxfp8_linear_with_fallback
        if fn.__module__ != "sglang.srt.layers.quantization.fp8_utils" or fn.__qualname__ != \
                "deepgemm_w8a8_mxfp8_linear_with_fallback":
            raise RuntimeError(f"dsv41-pd046 PGEMM: deepgemm_w8a8_mxfp8_linear_with_fallback is not stock's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK_LINEAR.append(fn)
    if getattr(mega_forward.forward_mega_moe, "__module__", "") != mega_forward.__name__:  # dsv41-pd070 LFIX: engine tree
        raise RuntimeError("dsv41-pd046 PGEMM: mega_forward.forward_mega_moe is not the generated entry")
    for index, layer in enumerate(module.layers):
        attn = layer.self_attn
        _register("wq_b", attn.wq_b, True)
        _register("wo_b", attn.wo_b, True)
        _register("wqkv_a", getattr(attn, "wqkv_a", None), True)
        indexer = getattr(attn, "indexer", None)
        _register("indexer.wq_b", getattr(indexer, "wq_b", None), False)
        engram = getattr(layer, "engram", None)
        _register("engram.wkv", getattr(engram, "wkv", None), False)
    device = module.layers[0].self_attn.wq_b.weight.device
    seen = set()
    for name, weight, scale, routes in list(_REG.values()):
        for min_rows, family, tactic in routes:
            if (family, str(tactic)) in seen:
                continue
            seen.add((family, str(tactic)))
            rows = -(-min_rows // 128) * 128
            k = weight.shape[1]
            q = torch.zeros((rows, k), device=device, dtype=torch.float8_e4m3fn)
            q_scale = torch.zeros(rows * (k // 32), device=device, dtype=torch.uint8)
            _gemm(family, tactic, q, q_scale, weight, scale)
    torch.cuda.synchronize(device)
    gen = torch.Generator(device=device).manual_seed(62)  # dsv41-pd062 CUBLT: each cuBLASLt route bitwise vs CuTe SWAP
    checked = set()
    for name, weight, scale, routes in list(_REG.values()):
        if tuple(weight.shape) in checked or not any(family == "cublaslt" for _, family, _ in routes):
            continue
        checked.add(tuple(weight.shape))
        rows, k = 1152, weight.shape[1]
        q = (torch.randn((rows, k), device=device, generator=gen) * 4).to(torch.float8_e4m3fn)
        q_scale = torch.randint(120, 134, (rows * (k // 32),), device=device, generator=gen, dtype=torch.uint8)
        ref, out = _gemm("cute", SWAP, q, q_scale, weight, scale), _gemm("cublaslt", None, q, q_scale, weight, scale)
        if not torch.equal(ref.view(torch.int16), out.view(torch.int16)):
            raise RuntimeError(f"dsv41-pd062 CUBLT: cuBLASLt differs from the CuTe SWAP tactic on {name} {tuple(weight.shape)}")
    torch.cuda.synchronize(device)
    if CHECK:
        _say(f"registered {len(_REG)} linears, {len(_METHODS)} quant methods, warmed {sorted(seen)}, "
             f"cuBLASLt bitwise vs CuTe SWAP on {sorted(checked)}")
