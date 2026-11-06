from __future__ import annotations

import contextlib
from functools import partial

import torch
from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as mmhc
from flashinfer.mega_mhc import prepare_mega_mhc

HIDDEN, MULT, MIX = 5120, 4, 24
# king-v1b WQB8 (dspd-v52p): eight decode wq_b bands, every one split-K 1
WQB_TACTICS = ((24, ((128, 32), (4, 1), True, False, 1)), (60, ((128, 64), (1, 1), True, False, 1)),
               (88, ((128, 128), (1, 4), False, False, 1)), (108, ((256, 128), (4, 1), True, False, 1)),
               (126, ((128, 128), (1, 4), False, False, 1)), (186, ((256, 192), (2, 1), True, False, 1)),
               (256, ((256, 256), (2, 2), False, False, 1)), (288, ((256, 192), (2, 2), True, False, 1)))
_WQB: dict = {}
_RUNNER: list = []
_MMHC: dict = {}
_STOCK: list = []


def _mmhc_served(x, residual, pre, post, comb, fn, scale, base, weight, y_fp8, y_gemm_sf, moe_input) -> bool:
    rows = x.shape[0]
    facts = _MMHC.get(x.device.index)
    if facts is None:
        return False
    sms = facts[2]
    return (
        y_fp8 is not None and y_gemm_sf is not None and moe_input is None
        and 1 <= rows <= mmhc.MAX_TOKENS
        and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
        and tuple(x.shape) == (rows, HIDDEN)
        and tuple(residual.shape) == (rows, MULT, HIDDEN)
        and tuple(fn.shape) == (MIX, MULT * HIDDEN)
        and tuple(scale.shape) == (3,) and tuple(base.shape) == (MIX,) and tuple(weight.shape) == (HIDDEN,)
        and tuple(pre.shape) == tuple(post.shape) == (rows, MULT)
        and tuple(comb.shape) == (rows, MULT, MULT)
        and all(t.dtype == torch.float32 for t in (pre, post, comb, fn, scale, base))
        and all(t.is_contiguous() for t in (x, residual, pre, post, comb, fn, scale, base, weight))
        and y_fp8.dtype == torch.float8_e4m3fn and tuple(y_fp8.shape) == (rows, HIDDEN) and y_fp8.is_contiguous()
        and y_gemm_sf.dtype == torch.int32 and tuple(y_gemm_sf.shape) == (rows, HIDDEN // 128)
        and y_gemm_sf.stride() == (1, -(-rows // 4) * 4)
        and mmhc.num_launch_sms(rows, HIDDEN, sms, mmhc.num_splits(rows, HIDDEN, sms), True, True, True, False) == sms
    )


def _mmhc(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters, y_fp8,
          y_gemm_sf):
    rows = x.shape[0]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, MULT, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    barriers, epochs, _ = _MMHC[x.device.index]
    out = dict(new_residual=updated, new_prev_mix=next_pre, new_post_mix=next_post, new_comb_res_mix=next_comb,
               y_bf16=normalized, y_fp8=y_fp8, y_gemm_sf=y_gemm_sf)
    prepare_mega_mhc(x=x, residual=residual, post_mix=post.unsqueeze(-1), comb_res_mix=comb,
                     shifted_prev_mix=pre.unsqueeze(-1), fn=fn, mix_scales=scale, mix_bases=base,
                     rmsnorm_weight=weight, hc_mult=MULT, hc_norm_eps=rms_eps, hc_pre_eps=hc_eps, hc_post_scale=2.0,
                     sinkhorn_eps=hc_eps, num_sinkhorn_iters=sinkhorn_iters, rmsnorm_eps=norm_eps, rmsnorm_scale=1.0,
                     sf_layout="col", out=out, split_barriers=barriers, launch_epochs=epochs).run()
    return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)


def exact_boundary(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters,
                   y_fp8=None, y_gemm_sf=None, moe_input=None):
    if _mmhc_served(x, residual, pre, post, comb, fn, scale, base, weight, y_fp8, y_gemm_sf, moe_input):
        return _mmhc(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                     sinkhorn_iters, y_fp8, y_gemm_sf)
    return _STOCK[0](x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                     sinkhorn_iters, y_fp8=y_fp8, y_gemm_sf=y_gemm_sf, moe_input=moe_input)


def _cute_gemm(q, q_scale, weight, weight_scale, tactic):
    from flashinfer.gemm import gemm_base

    if not _RUNNER:
        major, minor = torch.cuda.get_device_capability(q.device)
        _RUNNER.append(gemm_base._cute_dsl_gemm_mxfp8_runner(major, minor, True, torch.bfloat16))
    out = torch.empty((q.shape[0], weight.shape[0]), device=q.device, dtype=torch.bfloat16)
    work = gemm_base._get_cache_buf("mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, q.device)
    _RUNNER[0](inputs=[q, weight.t(), q_scale, weight_scale, torch.bfloat16, out, work], tactic=tactic)
    return out


def wq_b_linear(stock, input, weight, weight_scale, input_scale=None, bias=None, pin_tactic=None, **kwargs):
    entry = _WQB.get(weight.data_ptr())
    rows = input.numel() // input.shape[-1] if input.dim() else 0
    if entry is None or input_scale is None or bias is not None or kwargs or not 1 <= rows <= WQB_TACTICS[-1][0]:
        if pin_tactic is not None:
            kwargs["pin_tactic"] = pin_tactic
        return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias,
                     **kwargs)
    tactic = next(t for bound, t in WQB_TACTICS if rows <= bound)
    out = _cute_gemm(input.reshape(rows, input.shape[-1]), input_scale.view(torch.uint8), *entry, tactic)
    return out.view(*input.shape[:-1], weight.shape[0])


def _mmhc_setup(device) -> None:
    from sglang.srt.models import deepseek_v4_mhc as mhc

    if device.index in _MMHC:
        return
    _, sms = mmhc.device_facts(device.index)
    _MMHC[device.index] = (torch.zeros((mmhc.SPLIT_BARRIER_WORDS,), device=device, dtype=torch.uint64),
                           torch.zeros((sms,), device=device, dtype=torch.uint64), sms)
    first = {}
    for blocks in range(1, mmhc.MAX_TOKENS // 64 + 1):
        first.setdefault(mmhc.num_splits(64 * blocks - 2, HIDDEN, sms), 64 * blocks - 2)
    fn = torch.zeros((MIX, MULT * HIDDEN), device=device, dtype=torch.float32)
    scale, base = torch.ones(3, device=device), torch.zeros(MIX, device=device)
    weight = torch.ones(HIDDEN, device=device, dtype=torch.bfloat16)
    for rows in first.values():
        x = torch.zeros((rows, HIDDEN), device=device, dtype=torch.bfloat16)
        mix = torch.full((rows, MULT), 0.25, device=device)
        y_fp8, y_gemm_sf = mhc.empty_deepgemm_mxfp8_pair(rows, HIDDEN, device)
        _mmhc(x, torch.zeros((rows, MULT, HIDDEN), device=device, dtype=torch.bfloat16), mix, mix,
              torch.full((rows, MULT, MULT), 0.25, device=device), fn, scale, base, weight, 1e-6, 1e-6, 1e-6, 20,
              y_fp8, y_gemm_sf)
    torch.cuda.synchronize(device)


def prepare(module) -> None:
    from bundle_v9.dec import mhc_floor
    from sglang.kernels.ops.layernorm import mhc_mega

    if not _STOCK:
        fn = mhc_floor.mhc_mega_boundary
        if fn is not mhc_mega.mhc_mega_boundary:
            raise RuntimeError("bundle-v9: the MHC floor's mhc_mega_boundary is not stock's")
        _STOCK.append(fn)
    for index, layer in enumerate(module.layers):
        wq_b = layer.self_attn.wq_b
        scale = getattr(wq_b, "weight_scale_inv_swizzled", None)
        method = wq_b.quant_method
        backend = getattr(method, "mxfp8_dense_backend", None)
        if not (wq_b.weight.dtype == torch.float8_e4m3fn and tuple(wq_b.weight.shape) == (32768, 1280)
                and scale is not None and callable(getattr(method, "w8a8_mxfp8_linear", None))
                and backend is not None and backend.is_flashinfer_cutedsl()):
            raise TypeError(f"bundle-v9: layer {index}'s wq_b is not an MXFP8 (32768 x 1280) linear on FlashInfer's "
                            "CuTe-DSL backend with swizzled scales")
        _WQB[wq_b.weight.data_ptr()] = (wq_b.weight, scale.t() if scale.ndim == 2 else scale)
    device = module.layers[0].self_attn.wq_b.weight.device
    _mmhc_setup(device)
    entry = next(iter(_WQB.values()))
    for bound, tactic in WQB_TACTICS:
        q = torch.zeros((bound, 1280), device=device, dtype=torch.float8_e4m3fn)
        q_scale = torch.zeros(-(-bound // 128) * 128 * (1280 // 32), device=device, dtype=torch.uint8)
        _cute_gemm(q, q_scale, *entry, tactic)
    torch.cuda.synchronize(device)


@contextlib.contextmanager
def routed(view, forward_batch):
    from bundle_v9.dec import mhc_floor

    mode = getattr(forward_batch, "forward_mode", None)
    decode = mode is not None and (mode.is_decode() or mode.is_target_verify())
    methods = {}
    if decode:
        for layer in view.layers:
            method = layer.self_attn.wq_b.quant_method
            methods.setdefault(id(method), (method, method.w8a8_mxfp8_linear))
    mhc_floor.mhc_mega_boundary = exact_boundary
    try:
        for method, stock in methods.values():
            method.w8a8_mxfp8_linear = partial(wq_b_linear, stock)
        yield
    finally:
        mhc_floor.mhc_mega_boundary = _STOCK[0]
        for method, stock in methods.values():
            method.w8a8_mxfp8_linear = stock
