
from __future__ import annotations

import os
from functools import lru_cache

import torch

_K = 20480
_MIX = 24
_MAX_ROWS = 4095


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
def _header() -> str:
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9 import cuda_sinkhorn

    data = cuda_sinkhorn.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_sinkhorn.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "v4_hc_sinkhorn_reduce",
        cuda_files=[_header()],
        cuda_wrappers=[
            ("run_triton", "HcskReduceSinkhorn<80, false>::run"),
            ("run_deepgemm", "HcskReduceSinkhorn<16, true>::run"),
        ],
    )


def _reduce(part_mix, residual, part_sq, scale, base, m, sinkhorn_iters, rms_eps, hc_eps):
    dev = part_mix.device
    pre = torch.empty(m, 4, dtype=torch.float32, device=dev)
    post = torch.empty(m, 4, dtype=torch.float32, device=dev)
    comb = torch.empty(m, 4, 4, dtype=torch.float32, device=dev)
    mod = _module()
    if residual is None:
        mod.run_triton(part_mix, part_mix, part_sq, scale, base, pre, post, comb, 1.0 / _K, float(rms_eps),
                       int(sinkhorn_iters), float(hc_eps))
    else:
        mod.run_deepgemm(part_mix, residual, part_sq, scale, base, pre, post, comb, 1.0 / _K, float(rms_eps),
                         int(sinkhorn_iters), float(hc_eps))
    return pre, post, comb


def _in_domain(hc, x_flat) -> bool:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_platform

    m = x_flat.shape[0]
    return (
        x_flat.is_cuda
        and torch.version.cuda is not None
        and get_platform().is_sm100
        and x_flat.dtype == torch.bfloat16
        and 1 <= m <= _MAX_ROWS
        and x_flat.shape[1] == _K
        and x_flat.stride(1) == 1
        and hc.cfg.mult == 4
        and hc.cfg.sinkhorn_iters >= 1
        and tuple(hc.fn.shape) == (_MIX, _K)
        and hc.fn.dtype == torch.float32
        and hc.fn.is_contiguous()
        and hc.scale.dtype == torch.float32
        and hc.base.dtype == torch.float32
        and hc.scale.is_contiguous()
        and hc.base.is_contiguous()
        and hc.scale.numel() == 3
        and hc.base.numel() == _MIX
        and not is_batch_invariant_mode_enabled()
    )


def make_mix_stats_impl(stock_impl):

    def mix_stats_impl(hc, x: torch.Tensor):
        from sglang.kernels.ops.layernorm import mhc as ops
        from sglang.srt.environ import envs

        x_flat = x.flatten(1)
        if not _in_domain(hc, x_flat):
            return stock_impl(hc, x)
        cfg = hc.cfg
        m, k = x_flat.shape
        parts = (
            hc.tf32_parts
            if m >= 32 and x_flat.is_contiguous() and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
            else None
        )
        if parts is not None:
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

            slices = ops._HC_MIX_COMPENSATED_SLICES
            high, low = parts
            mix_hi = torch.empty((slices, m, _MIX), dtype=torch.float32, device=x.device)
            mix_lo = torch.empty_like(mix_hi)
            sq = torch.empty((slices, m), dtype=torch.float32, device=x.device)
            tf32_hc_prenorm_gemm(x_flat, high, mix_hi, sq, slices)
            tf32_hc_prenorm_gemm(x_flat, low, mix_lo, sq, slices)
            return _reduce(mix_hi, mix_lo, sq, hc.scale, hc.base, m, cfg.sinkhorn_iters, cfg.rms_eps, cfg.eps)
        import triton

        num_slices = ops._num_slices_for(k)
        part_mix = torch.empty((num_slices, m, _MIX), dtype=torch.float32, device=x.device)
        part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=x.device)
        block_m = ops._block_m_for(m)
        ops._hc_mix_stats_partial_kernel[(triton.cdiv(m, block_m), num_slices)](
            x_flat,
            hc.fn,
            part_mix,
            part_sq,
            m,
            k,
            x_flat.stride(0),
            hc.fn.stride(0),
            MIX=_MIX,
            MIX_PAD=max(16, triton.next_power_of_2(_MIX)),
            NUM_SLICES=num_slices,
            BLOCK_M=block_m,
            BLOCK_K=ops._HC_MIX_BLOCK_K,
            DOT_PRECISION=ops._HC_MIX_DOT_PRECISION,
            num_warps=ops._HC_MIX_NUM_WARPS,
            num_stages=ops._HC_MIX_NUM_STAGES,
        )
        return _reduce(part_mix, None, part_sq, hc.scale, hc.base, m, cfg.sinkhorn_iters, cfg.rms_eps, cfg.eps)

    return mix_stats_impl


def make_mix_stats(mix_stats_impl):

    def mix_stats(hc, x: torch.Tensor, stats_stream=None):
        if stats_stream is None:
            return mix_stats_impl(hc, x)
        main_stream = torch.cuda.current_stream()
        x.record_stream(stats_stream)
        with torch.cuda.stream(stats_stream):
            coefficients = mix_stats_impl(hc, x)
        for coefficient in coefficients:
            coefficient.record_stream(main_stream)
        return coefficients

    return mix_stats
