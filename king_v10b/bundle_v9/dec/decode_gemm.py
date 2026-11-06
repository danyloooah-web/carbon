from __future__ import annotations

import contextlib
import sys
from functools import partial

import torch

T8 = ((128, 8), (1, 1), True, False, 1)
T16 = ((128, 16), (1, 1), True, False, 1)
C42 = ((128, 64), (4, 2), True, True, 1)
C14 = ((128, 64), (1, 4), False, True, 1)
S21 = ((256, 64), (2, 1), True, False, 1)
W24 = ((256, 128), (2, 4), False, True, 1)
PK21 = ((256, 128), (2, 1), True, False, 1)  # king-v1c WOBPEAK (264 / 288-row decode wo_b)
# king-v7 WOBPEAK-DROP (critpath CRITPATH.md #2, VERIFY.md §5; IDEAS_dec_rest DR11): False = the crowned king's wo_b
# band (CuTe + swizzle to 256 rows, stock DeepGEMM wo_b on the DeepGEMM pair at 257..288) and its proof rows. The
# traces show WOBPEAK's CuTe 18.2 us + 2.5 us swizzle against stock DeepGEMM's 17.9 us at 264 rows (+96..+132 us/step
# at 264 / 288-row steps, out/wobcheck.txt). True = king-v1c WOBPEAK's (288, PK21) band, as in king_v2b..v6.
WOBPEAK_ON = False
WOB = ((12, T8), (24, T16), (36, C42), (72, C14), (96, C42), (128, C14), (192, S21), (256, W24)) + (
    ((288, PK21),) if WOBPEAK_ON else ())
WQKV = ((24, T8), (216, S21))
WOB_SHAPE, WQKV_SHAPE = (5120, 8192), (1792, 5120)
PROOF_ROWS = (1, 6, 12, 13, 24, 25, 36, 37, 64, 72, 96, 128, 148, 149, 160, 192, 216, 256) + (
    (257, 264, 288) if WOBPEAK_ON else ())
WKV = ((126, ((128, 192), (1, 1), False, False, 1)), (240, ((256, 192), (2, 1), False, False, 1)), (288, "deepgemm"))
WKV_SHAPE = (25600, 6144)
WKV_PROOF_ROWS = (1, 6, 48, 96, 126, 127, 144, 192, 240) + tuple(range(241, 289))
TRACE = True
# king-v7 WQKV288 (IDEAS_dec_rest DR7): CuTe wqkv_a at 217..288 rows (the king stops at 216, where HOIST mode 2 hands
# DeepGEMM pairs). Extends the WQKV band only after the whole served substitution (HOIST mode 1 + CuTe) is proven
# bitwise against the king's (HOIST mode 2 + stock DeepGEMM) at every row count in WQKV288_ROWS; off = the king's table.
# The tactic is the king's 25..216-row one (no rig read yet: the G0(c) rig decides; RANKED #18).
WQKV288_ON = True
WQKV288_TACTIC = S21
WQKV288_ROWS = tuple(range(217, 289))

_ON = [False]
_STOCK_LINEAR: list = []
_WOB: dict = {}
_WQKV: dict = {}
_METHODS: dict = {}
_WKV: dict = {}
_PERM: dict = {}
_WKV_CALLS = [0]
_SEEN: set = set()
_WQKV_LIVE = [WQKV]  # the served wqkv_a band: the king's WQKV, or WQKV + WQKV288's band once proven
_W288 = {"on": False, "why": "not prepared"}


def _trace(name, rows, tactic) -> None:
    if TRACE and (name, tactic) not in _SEEN:
        _SEEN.add((name, tactic))
        print(f"bundle-v9 DECODE_GEMM {name} rows {rows} tactic {tactic}", file=sys.stderr, flush=True)


def _tactic(table, rows):
    return next((tactic for bound, tactic in table if rows <= bound), None)


def serves_wqkv(rows: int) -> bool:
    return _ON[0] and 0 < rows <= _WQKV_LIVE[0][-1][0]


def dg_decode(input, weight, weight_scale, input_scale=None, bias=None, weight_scale_swizzled=None):
    from bundle_v9.dec import gemm_tables, prefill_gemm, prefill_fuse

    scale = _WOB.get(weight.data_ptr())
    rows = input.numel() // input.shape[-1] if input.dim() else 0
    served = gemm_tables.wob(input, weight, weight_scale, input_scale, bias, weight_scale_swizzled, rows) \
        if scale is not None and rows else None  # king-v10 WOB-T (None: king_v7f's path below)
    if served is not None:
        return served
    tactic = _tactic(WOB, rows) if scale is not None and rows else None
    if (tactic is None or bias is not None or input_scale is None or input_scale.dtype != torch.int32
            or tuple(input_scale.shape) != (rows, input.shape[-1] // 128)):
        return _STOCK_LINEAR[0](input, weight, weight_scale, input_scale=input_scale, bias=bias,
                                weight_scale_swizzled=weight_scale_swizzled)
    _trace("wo_b", rows, tactic)
    from bundle_v9.dec import glue_pdl

    swizzled = glue_pdl.wob_scale(input_scale)  # king-v6 WOBSF-PDL; None = the king's swizzle
    if swizzled is None:
        swizzled = prefill_fuse.wob_scale(input_scale)
    out = prefill_gemm._gemm("cute", tactic, input.reshape(rows, input.shape[-1]), swizzled, weight, scale)
    return out.view(*input.shape[:-1], weight.shape[0])


def _perm(rows, k, device):
    r = torch.arange(-(-rows // 4) * 4, device=device).clamp(max=rows - 1)
    c = torch.arange(k // 128, device=device)[:, None]
    return ((r // 128 * (k // 128) + c) * 128 + r % 32 * 4 + r % 128 // 32).reshape(-1)


def _wkv(input, weight, entry, rows, route):
    from sglang.srt.layers.quantization import fp8_utils

    from bundle_v9.dec import prefill_gemm, prefill_fuse

    x = input.reshape(rows, input.shape[-1])
    _trace("wkv", rows, route)
    _WKV_CALLS[0] += 1
    if route == "deepgemm":
        q, words = prefill_fuse.wkv_quant(x)
        out = _STOCK_LINEAR[0](q, weight, entry[1], input_scale=words.view(x.shape[1] // 128, -1)[:, :rows].t())
    else:
        q, sf = fp8_utils.flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        out = prefill_gemm._gemm("cute", route, q, sf, weight, entry[0])
    return out.view(*input.shape[:-1], weight.shape[0])


def qm_decode(stock, input, weight, weight_scale, input_scale=None, bias=None, pin_tactic=None, **kwargs):
    from bundle_v9.dec import gemm_tables, prefill_gemm

    rows = input.numel() // input.shape[-1] if input.dim() else 0
    wkv = _WKV.get(weight.data_ptr())
    route = _tactic(WKV, rows) if wkv is not None and rows else None
    if (route is not None and input_scale is None and bias is None and not kwargs
            and input.dtype == torch.bfloat16):
        return _wkv(input, weight, wkv, rows, route)
    scale = _WQKV.get(weight.data_ptr())
    tactic = gemm_tables.wqkv_tactic(rows, _tactic(_WQKV_LIVE[0], rows)) if scale is not None and rows else None  # WQKV-T
    if tactic is None or input_scale is None or input_scale.dtype != torch.uint8 or bias is not None or kwargs:
        if pin_tactic is not None:
            kwargs["pin_tactic"] = pin_tactic
        return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias,
                     **kwargs)
    _trace("wqkv_a(T)" if rows in gemm_tables._SERVE["wqkv_a"] else "wqkv_a", rows, tactic)  # WQKV-T: CuTe taken
    out = prefill_gemm._gemm("cute", tactic, input.reshape(rows, input.shape[-1]), input_scale, weight, scale)
    return out.view(*input.shape[:-1], weight.shape[0])


@contextlib.contextmanager
def routed(forward_batch):
    from sglang.srt.layers.quantization import fp8 as fp8_linear

    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not (mode.is_decode() or mode.is_target_verify()) or not _STOCK_LINEAR:
        yield
        return
    methods = [(method, method.w8a8_mxfp8_linear) for method in _METHODS.values()]
    from bundle_v9.dec import dgt, gemm_tables

    dgt_swap = dgt.on()  # king-v7 DGT: wo_a's fp8_einsum through the triggered 1d1d build where proven
    if dgt_swap:
        import deep_gemm
    fp8_linear.deepgemm_w8a8_mxfp8_linear_with_fallback = dg_decode
    _ON[0] = True
    try:
        for method, stock in methods:
            method.w8a8_mxfp8_linear = partial(qm_decode, stock)
        if dgt_swap:
            dgt._ON[0] = True
            deep_gemm.fp8_einsum = dgt.fp8_einsum
        with gemm_tables.routed():  # king-v10 WOA-T: deepseek_v4._fp8_wo_a_einsum, restored on exit (never DGT's rows)
            yield
    finally:
        _ON[0] = False
        fp8_linear.deepgemm_w8a8_mxfp8_linear_with_fallback = _STOCK_LINEAR[0]
        for method, stock in methods:
            method.w8a8_mxfp8_linear = stock
        if dgt_swap:
            dgt._ON[0] = False
            deep_gemm.fp8_einsum = dgt._STOCK[0]


def _random_pair(rows, k, device, gen):
    from sglang.srt.models import deepseek_v4_mhc as mhc

    data = (torch.randn((rows, k), device=device, generator=gen) * 4).to(torch.float8_e4m3fn)
    scale = mhc.empty_deepgemm_mxfp8_scales(rows, k, device)
    span = 1 + sum((n - 1) * st for n, st in zip(scale.shape, scale.stride()))
    raw = torch.randint(120, 134, (span * 4,), device=device, generator=gen, dtype=torch.uint8)
    torch.as_strided(scale, (span,), (1,), scale.storage_offset()).copy_(raw.view(torch.int32))
    return data, scale


def _prove(attn, device) -> None:
    from bundle_v9.dec import mhc_floor
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import mhc_post_combine_norm_decode

    gen = torch.Generator(device=device).manual_seed(63)
    wo_b, wqkv_a = attn.wo_b, attn.wqkv_a
    for rows in PROOF_ROWS:
        data, packed = _random_pair(rows, WOB_SHAPE[1], device, gen)
        if packed.dtype != torch.int32 or tuple(packed.shape) != (rows, WOB_SHAPE[1] // 128) or _tactic(WOB, rows) is None:
            raise RuntimeError(f"bundle-v9 DECODE_GEMM: the proof's wo_b pair at {rows} rows is outside the route's domain")
        ref = _STOCK_LINEAR[0](data, wo_b.weight, wo_b.weight_scale_inv_deepgemm, input_scale=packed)
        out = dg_decode(data, wo_b.weight, wo_b.weight_scale_inv_deepgemm, input_scale=packed)
        if not torch.equal(ref.view(torch.int16), out.view(torch.int16)):
            raise RuntimeError(f"bundle-v9 DECODE_GEMM: wo_b at {rows} rows differs from stock's DeepGEMM linear")
        if rows > WQKV[-1][0]:
            continue
        y = (torch.randn((rows, 5120), device=device, generator=gen) * 2).bfloat16()
        residual = torch.randn((rows, 4, 5120), device=device, generator=gen).bfloat16()
        post = torch.rand((rows, 4), device=device, generator=gen) * 2
        comb = torch.rand((rows, 4, 4), device=device, generator=gen)
        pre = torch.rand((rows, 4), device=device, generator=gen)
        norm = (torch.rand(5120, device=device, generator=gen) + 0.5).bfloat16()
        eps = 1e-6
        upd_dg, nrm_dg, (q_dg, s_dg) = mhc_floor.mhc_post_combine_norm_decode_dg(y, residual, post, comb, pre,
                                                                                      norm, eps)
        upd, nrm, q, sf = mhc_post_combine_norm_decode(y, residual, post, comb, pre, norm, eps, quant=True,
                                                       threads=mhc_floor._decode_threads(rows, device))
        if not (torch.equal(upd.view(torch.int16), upd_dg.view(torch.int16))
                and torch.equal(nrm.view(torch.int16), nrm_dg.view(torch.int16))
                and torch.equal(q.view(torch.uint8), q_dg.view(torch.uint8))):
            raise RuntimeError(f"bundle-v9 DECODE_GEMM: the quantizing boundary differs from the DeepGEMM one at {rows} rows")
        ref = _STOCK_LINEAR[0](q_dg, wqkv_a.weight, wqkv_a.weight_scale_inv_deepgemm, input_scale=s_dg)
        out = qm_decode(None, q, wqkv_a.weight, None, input_scale=sf)
        if not torch.equal(ref.view(torch.int16), out.view(torch.int16)):
            raise RuntimeError(f"bundle-v9 DECODE_GEMM: wqkv_a at {rows} rows differs from stock's DeepGEMM path")
    torch.cuda.synchronize(device)


def _w288_line() -> str:
    return f"bundle-v9 king-v7 WQKV288: {'ON' if _W288['on'] else 'off'} ({_W288['why']})"


def _prove_wqkv288(attn, device) -> int:
    """The served 217..288-row substitution against the king's: HOIST mode 1 (FlashInfer-layout pair) + CuTe wqkv_a
    against HOIST mode 2 (DeepGEMM pair) + stock's DeepGEMM linear, as mhc_floor._post_fusion picks them."""
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import mhc_post_combine_norm_decode

    from bundle_v9.dec import mhc_floor, mhc_prefetch

    gen = torch.Generator(device=device).manual_seed(66)
    wqkv_a = attn.wqkv_a
    eps = 1e-6
    for rows in WQKV288_ROWS:
        if _tactic(_WQKV_LIVE[0], rows) is None:
            raise RuntimeError(f"no wqkv_a tactic at {rows} rows in the candidate band")
        y = (torch.randn((rows, 5120), device=device, generator=gen) * 2).bfloat16()
        y[rows // 2].zero_()
        residual = torch.randn((rows, 4, 5120), device=device, generator=gen).bfloat16()
        residual[rows // 3, 1].fill_(-0.0)
        post = torch.rand((rows, 4), device=device, generator=gen) * 2
        comb = torch.rand((rows, 4, 4), device=device, generator=gen)
        pre = torch.rand((rows, 4), device=device, generator=gen)
        norm = (torch.rand(5120, device=device, generator=gen) + 0.5).bfloat16()
        upd_dg, nrm_dg, (q_dg, s_dg) = mhc_floor.mhc_post_combine_norm_decode_dg(y, residual, post, comb, pre, norm,
                                                                                  eps)
        threads = mhc_floor._decode_threads(rows, device)
        if mhc_prefetch.serves(threads):
            upd, nrm, q, sf = mhc_prefetch.post_combine(y, residual, post, comb, pre, norm, eps, True)
        else:
            upd, nrm, q, sf = mhc_post_combine_norm_decode(y, residual, post, comb, pre, norm, eps, quant=True,
                                                           threads=threads)
        if not (torch.equal(upd.view(torch.int16), upd_dg.view(torch.int16))
                and torch.equal(nrm.view(torch.int16), nrm_dg.view(torch.int16))
                and torch.equal(q.view(torch.uint8), q_dg.view(torch.uint8))):
            raise RuntimeError(f"the FlashInfer-layout boundary differs from the DeepGEMM-pair one at {rows} rows")
        ref = _STOCK_LINEAR[0](q_dg, wqkv_a.weight, wqkv_a.weight_scale_inv_deepgemm, input_scale=s_dg)
        out = qm_decode(None, q, wqkv_a.weight, None, input_scale=sf)
        if not torch.equal(ref.view(torch.int16), out.view(torch.int16)):
            raise RuntimeError(f"CuTe wqkv_a at {rows} rows differs from stock's DeepGEMM path")
    torch.cuda.synchronize(device)
    return len(WQKV288_ROWS)


def _prepare_wqkv288(attn, device) -> None:
    """Never raises: on any failure the king's WQKV band (<= 216 rows) serves."""
    _WQKV_LIVE[0] = WQKV
    _W288.update(on=False, why="not prepared")
    if not WQKV288_ON:
        _W288["why"] = "switched off by decode_gemm.WQKV288_ON; DeepGEMM wqkv_a at 217..288 rows as in the king"
    else:
        try:
            from bundle_v9.dec import mhc_floor

            top = WQKV288_ROWS[-1] if WQKV288_ROWS else 0
            if (tuple(WQKV288_ROWS) != tuple(range(WQKV[-1][0] + 1, top + 1))
                    or top > mhc_floor.MEGA_MHC_DECODE_MAX_ROWS):
                raise RuntimeError(f"WQKV288_ROWS must be every row count from {WQKV[-1][0] + 1} up to at most "
                                   f"{mhc_floor.MEGA_MHC_DECODE_MAX_ROWS}")
            _WQKV_LIVE[0] = WQKV + ((WQKV288_ROWS[-1], WQKV288_TACTIC),)
            cases = _prove_wqkv288(attn, device)
            _W288.update(on=True, why=f"CuTe wqkv_a tactic {WQKV288_TACTIC} at {WQKV288_ROWS[0]}..{WQKV288_ROWS[-1]} "
                                      f"rows (HOIST mode 1 pairs); bytes equal to HOIST mode 2 + DeepGEMM at {cases} "
                                      f"row counts")
        except Exception as error:
            _WQKV_LIVE[0] = WQKV
            _W288["why"] = f"FAIL, DeepGEMM wqkv_a above 216 rows: {type(error).__name__}: {str(error)[:400]}"
    print(_w288_line(), file=sys.stderr, flush=True)


def _register_wkv(index, wkv) -> None:
    from sglang.srt.layers.quantization import fp8_utils

    from bundle_v9.dec import prefill_gemm

    method = wkv.quant_method
    entry = prefill_gemm._REG.get(wkv.weight.data_ptr())
    if (entry is None or entry[0] != "engram.wkv" or tuple(wkv.weight.shape) != WKV_SHAPE
            or not getattr(method, "block_fp8_as_mxfp8", False) or not getattr(wkv, "block_fp8_mxfp8_ready", False)
            or not callable(getattr(method, "w8a8_mxfp8_linear", None))):
        raise TypeError(f"bundle-v9 WKV: layer {index}'s Engram wkv is not a registered {WKV_SHAPE} block-FP8 MXFP8 linear")
    n, k = WKV_SHAPE
    scale_u8 = fp8_utils.block_fp8_scale_to_mxfp8_e8m0(wkv.weight_scale_inv.data, (n, k), method.weight_block_size)
    _WKV[wkv.weight.data_ptr()] = (entry[2], fp8_utils.pack_mxfp8_weight_scale_for_deepgemm(scale_u8, n, k), wkv)
    _METHODS.setdefault(id(method), method)


def _prove_wkv(device) -> None:
    gen = torch.Generator(device=device).manual_seed(64)
    k = WKV_SHAPE[1]
    for _, _, wkv in _WKV.values():
        method = wkv.quant_method
        stock = method.w8a8_mxfp8_linear
        for rows in WKV_PROOF_ROWS:
            x = torch.randn((rows, k), device=device, generator=gen)
            row = torch.randint(-7, 8, (rows, 1), device=device, generator=gen).float().exp2()
            group = torch.randint(-7, 8, (1, k // 32), device=device, generator=gen).float().exp2()
            x = (x * row * group.repeat_interleave(32, dim=1)).bfloat16()
            ref = method.apply(wkv, x)
            calls = _WKV_CALLS[0]
            method.w8a8_mxfp8_linear = partial(qm_decode, stock)
            try:
                out = method.apply(wkv, x)
            finally:
                method.w8a8_mxfp8_linear = stock
            if _WKV_CALLS[0] != calls + 1:
                raise RuntimeError(f"bundle-v9 WKV: stock's apply of the Engram wkv at {rows} rows did not reach the route")
            if not torch.equal(ref.view(torch.int16), out.view(torch.int16)):
                raise RuntimeError(f"bundle-v9 WKV: the Engram wkv at {rows} rows differs from stock's call")
    torch.cuda.synchronize(device)


def _prove_wkvq(device) -> None:
    from sglang.srt.layers.quantization import fp8_utils

    from bundle_v9.dec import prefill_fuse

    gen = torch.Generator(device=device).manual_seed(65)
    k = WKV_SHAPE[1]
    for rows in range(WKV[-2][0] + 1, WKV[-1][0] + 1):
        x = torch.randn((rows, k), device=device, generator=gen)
        row = torch.randint(-7, 8, (rows, 1), device=device, generator=gen).float().exp2()
        group = torch.randint(-7, 8, (1, k // 32), device=device, generator=gen).float().exp2()
        x = (x * row * group.repeat_interleave(32, dim=1)).bfloat16()
        flat = x.view(-1)
        picks = torch.randint(0, flat.numel(), (6, 64), device=device, generator=gen)
        for i, value in enumerate((-0.0, float("inf"), float("-inf"), float("nan"), 1e-40, 3e38)):
            flat[picks[i]] = value
        x[0, :32] = 0.0
        x[rows - 1, 32:64] = float("nan")
        x[rows // 2, 64:96] = -0.0
        q_ref, sf_ref = fp8_utils.flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        words_ref = sf_ref.reshape(-1).view(torch.int32).index_select(0, _PERM[rows])
        q, words = prefill_fuse.wkv_quant(x)
        if not (torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(words, words_ref)):
            raise RuntimeError(f"WKVQ: the quantized pair at {rows} rows differs from stock's quantize and gather")
    torch.cuda.synchronize(device)


def enabled() -> bool:
    from sglang.srt.runtime_context import get_disagg

    return get_disagg().disaggregation_mode != "prefill"


def prepare(module) -> None:
    from sglang.srt.layers.quantization import fp8 as fp8_linear

    from bundle_v9.dec import prefill_gemm

    if not enabled():
        return
    _WQKV_LIVE[0] = WQKV
    if not _STOCK_LINEAR:
        fn = fp8_linear.deepgemm_w8a8_mxfp8_linear_with_fallback
        if fn.__module__ != "sglang.srt.layers.quantization.fp8_utils" or fn.__qualname__ != \
                "deepgemm_w8a8_mxfp8_linear_with_fallback":
            raise RuntimeError(f"bundle-v9 DECODE_GEMM: deepgemm_w8a8_mxfp8_linear_with_fallback is not stock's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK_LINEAR.append(fn)
    for index, layer in enumerate(module.layers):
        attn = layer.self_attn
        for name, linear, registry, shape in (("wo_b", attn.wo_b, _WOB, WOB_SHAPE),
                                              ("wqkv_a", getattr(attn, "wqkv_a", None), _WQKV, WQKV_SHAPE)):
            entry = prefill_gemm._REG.get(linear.weight.data_ptr()) if linear is not None else None
            if entry is None or entry[0] != name or tuple(linear.weight.shape) != shape \
                    or not getattr(linear, "deepgemm_mxfp8_input", False) \
                    or getattr(linear, "weight_scale_inv_deepgemm", None) is None:
                raise TypeError(f"bundle-v9 DECODE_GEMM: layer {index}'s {name} is not a registered {shape} MXFP8 linear "
                                "taking DeepGEMM pairs")
            registry[linear.weight.data_ptr()] = entry[2]
        method = attn.wqkv_a.quant_method
        if not callable(getattr(method, "w8a8_mxfp8_linear", None)):
            raise TypeError(f"bundle-v9 DECODE_GEMM: layer {index}'s wqkv_a has no w8a8_mxfp8_linear")
        _METHODS.setdefault(id(method), method)
        if getattr(layer, "engram", None) is not None:
            _register_wkv(index, layer.engram.wkv)
    if not _WKV:
        raise TypeError("bundle-v9 WKV: the model has no Engram wkv")
    attn = module.layers[0].self_attn
    device = attn.wo_b.weight.device
    for table, shape, registry in ((WOB, WOB_SHAPE, _WOB), (WQKV, WQKV_SHAPE, _WQKV)):
        weight = attn.wo_b.weight if shape == WOB_SHAPE else attn.wqkv_a.weight
        for bound, tactic in table:
            rows = -(-bound // 128) * 128
            q = torch.zeros((bound, shape[1]), device=device, dtype=torch.float8_e4m3fn)
            q_scale = torch.zeros(rows * (shape[1] // 32), device=device, dtype=torch.uint8)
            prefill_gemm._gemm("cute", tactic, q, q_scale, weight, registry[weight.data_ptr()])
    torch.cuda.synchronize(device)
    from bundle_v9.dec import glue_pdl

    glue_pdl.prepare_wobsf(device)  # king-v6 WOBSF-PDL; never raises, the king's swizzle on any failure
    try:
        _prove(attn, device)  # with WOBSF-PDL serving: the whole wo_b substitution at its served launch form
    except Exception as error:
        if not glue_pdl.wobsf_on():
            raise
        glue_pdl.fail_wobsf(f"decode_gemm's wo_b proof failed with the PDL swizzle serving: {type(error).__name__}: "
                            f"{str(error)[:300]}")
        _prove(attn, device)  # the king's path, which raises as in the king
    else:
        glue_pdl.report_wobsf()
    drop = ("off (king-v1c WOBPEAK serves: CuTe wo_b + scale swizzle at 257..288 rows)" if WOBPEAK_ON else
            "ON (the king's wo_b band: stock DeepGEMM wo_b on the DeepGEMM pair at 257..288 rows)")
    print(f"bundle-v9 king-v7 WOBPEAK-DROP: {drop}; wo_b bytes proven at {len(PROOF_ROWS)} row counts",
          file=sys.stderr, flush=True)
    _prepare_wqkv288(attn, device)  # king-v7 WQKV288; never raises, the king's band on any failure
    for rows in range(WKV[-2][0] + 1, WKV[-1][0] + 1):
        _PERM[rows] = _perm(rows, WKV_SHAPE[1], device)
    _prove_wkvq(device)
    _prove_wkv(device)
    from bundle_v9.dec import dgt, xgf

    xgf.prepare(module)  # king-v7 XG-F (MoE front PDL); never raises, the king's launch on any failure
    dgt.prepare(module)  # king-v7 DGT (wo_a trigger); never raises, stock's wo_a on any failure
    from bundle_v9.dec import gemm_tables

    # king-v10 WOB-T / WQKV-T / WOA-T: proven last, against king_v7f's served paths (WOBSF-PDL, WQKV288, DGT all set);
    # never raises but on a CUDA context fault, king_v7f's path per band on any failure
    gemm_tables.prepare(module)
