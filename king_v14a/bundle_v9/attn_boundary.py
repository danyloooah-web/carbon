
from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as mmhc
from flashinfer.mega_mhc import prepare_mega_mhc

from bundle_v9 import mhc_sched

HIDDEN, MULT, MIX = 5120, 4, 24
_MMHC: dict = {}
_STOCK: list = []
_REHOME_Y_BF16: list = []


def _mmhc_served(x, residual, pre, post, comb, fn, scale, base, weight, y_fp8, y_gemm_sf, moe_input, *,
                 ffn_ok: bool = False) -> bool:
    rows = x.shape[0]
    facts = _MMHC.get(x.device.index)
    if facts is None:
        return False
    sms = facts[2]
    pair = y_fp8 is not None and y_gemm_sf is not None
    ffn = ffn_ok and y_fp8 is None and y_gemm_sf is None
    return (
        (pair or ffn) and moe_input is None
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
        and (ffn or (y_fp8.dtype == torch.float8_e4m3fn and tuple(y_fp8.shape) == (rows, HIDDEN)
                     and y_fp8.is_contiguous()
                     and y_gemm_sf.dtype == torch.int32 and tuple(y_gemm_sf.shape) == (rows, HIDDEN // 128)
                     and y_gemm_sf.stride() == (1, -(-rows // 4) * 4)))
        and mmhc.num_launch_sms(rows, HIDDEN, sms, mmhc.num_splits(rows, HIDDEN, sms), True, True, True, False) == sms
    )


def _rehome_normalized(x: torch.Tensor) -> torch.Tensor:
    if not _REHOME_Y_BF16:
        return torch.empty_like(x)
    out = _REHOME_Y_BF16.pop()
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise RuntimeError(f"bundle-v9: armed y_bf16 {tuple(out.shape)} {out.dtype} does not fit seam {tuple(x.shape)}")
    return out


def _run_mmhc(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters,
              y_fp8, y_gemm_sf, normalized: torch.Tensor):
    rows = x.shape[0]
    updated = torch.empty_like(residual)
    next_pre = torch.empty((rows, MULT, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    barriers, epochs, _ = _MMHC[x.device.index]
    if y_fp8 is None:
        from sglang.srt.models import deepseek_v4_mhc as mhc

        y_fp8, y_gemm_sf = mhc.empty_deepgemm_mxfp8_pair(rows, HIDDEN, x.device)
    out = dict(new_residual=updated, new_prev_mix=next_pre, new_post_mix=next_post, new_comb_res_mix=next_comb,
               y_bf16=normalized, y_fp8=y_fp8, y_gemm_sf=y_gemm_sf)
    with mhc_sched.scheduled():
        prepare_mega_mhc(x=x, residual=residual, post_mix=post.unsqueeze(-1), comb_res_mix=comb,
                         shifted_prev_mix=pre.unsqueeze(-1), fn=fn, mix_scales=scale, mix_bases=base,
                         rmsnorm_weight=weight, hc_mult=MULT, hc_norm_eps=rms_eps, hc_pre_eps=hc_eps,
                         hc_post_scale=2.0, sinkhorn_eps=hc_eps, num_sinkhorn_iters=sinkhorn_iters,
                         rmsnorm_eps=norm_eps, rmsnorm_scale=1.0, sf_layout="col", out=out, split_barriers=barriers,
                         launch_epochs=epochs).run()
    return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)


def exact_boundary(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps, sinkhorn_iters,
                   y_fp8=None, y_gemm_sf=None, moe_input=None):
    if _mmhc_served(x, residual, pre, post, comb, fn, scale, base, weight, y_fp8, y_gemm_sf, moe_input):
        return _run_mmhc(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                         sinkhorn_iters, y_fp8, y_gemm_sf, _rehome_normalized(x))
    if y_fp8 is None and y_gemm_sf is None and moe_input is None and mhc_sched.dg_serves(x):
        return mhc_sched.dg_boundary(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                                     sinkhorn_iters)
    return _STOCK[0](x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                     sinkhorn_iters, y_fp8=y_fp8, y_gemm_sf=y_gemm_sf, moe_input=moe_input)


def exact_boundary_decode(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                          sinkhorn_iters, y_fp8=None, y_gemm_sf=None, moe_input=None):
    if _mmhc_served(x, residual, pre, post, comb, fn, scale, base, weight, y_fp8, y_gemm_sf, moe_input, ffn_ok=True):
        return _run_mmhc(x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                         sinkhorn_iters, y_fp8, y_gemm_sf, torch.empty_like(x))
    return _STOCK[0](x, residual, pre, post, comb, fn, scale, base, weight, rms_eps, hc_eps, norm_eps,
                     sinkhorn_iters, y_fp8=y_fp8, y_gemm_sf=y_gemm_sf, moe_input=moe_input)


def _mmhc_setup(device: torch.device) -> None:
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
        _run_mmhc(x, torch.zeros((rows, MULT, HIDDEN), device=device, dtype=torch.bfloat16), mix, mix,
                  torch.full((rows, MULT, MULT), 0.25, device=device), fn, scale, base, weight, 1e-6, 1e-6, 1e-6, 20,
                  y_fp8, y_gemm_sf, torch.empty_like(x))
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _bits_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


@torch.no_grad()
def prepare(module, device: torch.device) -> int:
    from sglang.kernels.ops.layernorm import mhc_mega
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.runtime_context import get_disagg, get_parallel

    from bundle_v9 import mhc_floor

    if not _STOCK:
        fn = mhc_floor.mhc_mega_boundary
        if fn is not mhc_mega.mhc_mega_boundary:
            raise RuntimeError("bundle-v9: mhc_floor.mhc_mega_boundary is not stock's")
        _STOCK.append(fn)
    mhc_sched.prepare()
    _mmhc_setup(device)
    mhc_sched.verify()
    if get_disagg().disaggregation_mode != "decode":
        mhc_sched.dg_prepare()

    group = get_parallel().tp_group
    layer = module.layers[min(2, len(module.layers) - 1)]
    ahc, cfg, norm = layer.attn_hc, layer.ffn_hc.cfg, layer.input_layernorm
    bad: list[str] = []
    served_checks = 0
    try:
        for rows in (62, 190, 288, 1024, 4096):
            g = torch.Generator(device=device).manual_seed(20261031 + rows)
            x = (torch.randn((rows, HIDDEN), generator=g, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            res = (torch.randn((rows, MULT, HIDDEN), generator=g, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            pre = torch.rand((rows, MULT), generator=g, device=device, dtype=torch.float32)
            post = torch.rand((rows, MULT), generator=g, device=device, dtype=torch.float32)
            comb = torch.full((rows, MULT, MULT), 0.25, device=device, dtype=torch.float32)
            rfp8, rsf = mhc.empty_deepgemm_mxfp8_pair(rows, HIDDEN, device)
            if not _mmhc_served(x, res, pre, post, comb, ahc.fn, ahc.scale, ahc.base, norm.weight, rfp8, rsf, None):
                continue
            rupd, rnorm, (rpre, rpost, rcomb) = _STOCK[0](
                x, res, pre, post, comb, ahc.fn, ahc.scale, ahc.base, norm.weight,
                cfg.rms_eps, cfg.eps, norm.variance_epsilon, cfg.sinkhorn_iters, y_fp8=rfp8, y_gemm_sf=rsf,
            )
            dfp8, dsf = mhc.empty_deepgemm_mxfp8_pair(rows, HIDDEN, device)
            dupd, dnorm, (dpre, dpost, dcomb) = exact_boundary_decode(
                x, res, pre, post, comb, ahc.fn, ahc.scale, ahc.base, norm.weight,
                cfg.rms_eps, cfg.eps, norm.variance_epsilon, cfg.sinkhorn_iters, y_fp8=dfp8, y_gemm_sf=dsf,
            )
            inplace = torch.empty_like(x)
            pfp8, psf = mhc.empty_deepgemm_mxfp8_pair(rows, HIDDEN, device)
            _REHOME_Y_BF16[:] = [inplace]
            try:
                pupd, pnorm, (ppre, ppost, pcomb) = exact_boundary(
                    x, res, pre, post, comb, ahc.fn, ahc.scale, ahc.base, norm.weight,
                    cfg.rms_eps, cfg.eps, norm.variance_epsilon, cfg.sinkhorn_iters, y_fp8=pfp8, y_gemm_sf=psf,
                )
            finally:
                unconsumed = len(_REHOME_Y_BF16)
                _REHOME_Y_BF16.clear()
            if unconsumed or pnorm.data_ptr() != inplace.data_ptr():
                bad.append(f"r={rows}:inplace")
                break
            pairs = (
                ("upd", rupd, dupd, pupd),
                ("norm", rnorm, dnorm, pnorm),
                ("pre", rpre, dpre, ppre),
                ("post", rpost, dpost, ppost),
                ("comb", rcomb, dcomb, pcomb),
                ("fp8", rfp8, dfp8, pfp8),
                ("sf", rsf, dsf, psf),
            )
            for label, ref_t, dec_t, pf_t in pairs:
                if not (_bits_eq(ref_t, dec_t) and _bits_eq(ref_t, pf_t)):
                    bad.append(f"r={rows}:{label}")
                    break
            served_checks += 1
            if bad:
                break
        if served_checks == 0 and not bad:
            bad.append("no served row bucket matched mmhc")
        last = len(module.layers) - 1
        for index, rows in ((i, r) for i in (0, last) for r in (240, 264, 288)):
            if bad:
                break
            fhc = module.layers[index].ffn_hc
            g = torch.Generator(device=device).manual_seed(20261101 + 1000 * index + rows)
            x = (torch.randn((rows, HIDDEN), generator=g, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            res = (torch.randn((rows, MULT, HIDDEN), generator=g, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            pre = torch.rand((rows, MULT), generator=g, device=device, dtype=torch.float32)
            post = torch.rand((rows, MULT), generator=g, device=device, dtype=torch.float32)
            comb = torch.full((rows, MULT, MULT), 0.25, device=device, dtype=torch.float32)
            args = (x, res, pre, post, comb, fhc.fn, fhc.scale, fhc.base, fhc.norm.weight,
                    fhc.cfg.rms_eps, fhc.cfg.eps, fhc.norm.variance_epsilon, fhc.cfg.sinkhorn_iters)
            if not _mmhc_served(*args[:9], None, None, None, ffn_ok=True):
                bad.append(f"ffn L{index} r={rows}:unserved")
                break
            form = dict(y_fp8=None, y_gemm_sf=None, moe_input=None)
            rupd, rnorm, (rpre, rpost, rcomb) = _STOCK[0](*args, **form)
            dupd, dnorm, (dpre, dpost, dcomb) = exact_boundary_decode(*args, **form)
            for label, ref_t, dec_t in (("upd", rupd, dupd), ("norm", rnorm, dnorm), ("pre", rpre, dpre),
                                        ("post", rpost, dpost), ("comb", rcomb, dcomb)):
                if not _bits_eq(ref_t, dec_t):
                    bad.append(f"ffn L{index} r={rows}:{label}")
                    break
            served_checks += 1
        for index, rows in ((i, r) for i in (0, last) for r in (mhc_sched.DG_ROWS + 2, 4094) if mhc_sched._DG):
            if bad:
                break
            fhc = module.layers[index].ffn_hc
            g = torch.Generator(device=device).manual_seed(20261102 + 1000 * index + rows)
            x = (torch.randn((rows, HIDDEN), generator=g, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            res = (torch.randn((rows, MULT, HIDDEN), generator=g, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            pre = torch.rand((rows, MULT), generator=g, device=device, dtype=torch.float32)
            post = torch.rand((rows, MULT), generator=g, device=device, dtype=torch.float32)
            comb = torch.full((rows, MULT, MULT), 0.25, device=device, dtype=torch.float32)
            args = (x, res, pre, post, comb, fhc.fn, fhc.scale, fhc.base, fhc.norm.weight,
                    fhc.cfg.rms_eps, fhc.cfg.eps, fhc.norm.variance_epsilon, fhc.cfg.sinkhorn_iters)
            form = dict(y_fp8=None, y_gemm_sf=None, moe_input=None)
            calls = mhc_sched._DG_CALLS[0]
            rupd, rnorm, (rpre, rpost, rcomb) = _STOCK[0](*args, **form)
            pupd, pnorm, (ppre, ppost, pcomb) = exact_boundary(*args, **form)
            if mhc_sched._DG_CALLS[0] != calls + 1:
                bad.append(f"dg L{index} r={rows}:unserved")
                break
            for label, ref_t, pf_t in (("upd", rupd, pupd), ("norm", rnorm, pnorm), ("pre", rpre, ppre),
                                       ("post", rpost, ppost), ("comb", rcomb, pcomb)):
                if not _bits_eq(ref_t, pf_t):
                    bad.append(f"dg L{index} r={rows}:{label}")
                    break
            served_checks += 1
    except Exception as exc:
        bad.append(f"{type(exc).__name__}: {exc}")

    gathered = [None] * group.world_size
    dist.all_gather_object(gathered, bad[:3], group=group.cpu_group)
    failures = [f"rank {r}: {', '.join(errs)}" for r, errs in enumerate(gathered) if errs]
    if failures:
        raise RuntimeError(f"bundle-v9 MMHC STRICT byte-probe failed ({' | '.join(failures)})")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return served_checks


@contextlib.contextmanager
def routed(view, forward_batch):
    from bundle_v9 import mhc_floor

    if mhc_floor.mhc_mega_boundary is not _STOCK[0]:
        raise RuntimeError("bundle-v9: mhc_floor.mhc_mega_boundary changed since prepare")
    rehome_on = bool(getattr(view, "_v4_rehome_on", False))
    mhc_floor.mhc_mega_boundary = exact_boundary if rehome_on else exact_boundary_decode
    try:
        yield
    finally:
        mhc_floor.mhc_mega_boundary = _STOCK[0]
