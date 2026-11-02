"""pd002 sk_jit: the sealed decode / verify mHC stats with pd001's SK reduce (hc_sinkhorn_reduce.cuh) built by SGLang's
own `load_jit` at first use, in place of the pybind extension (`hcsk_pd001`) that cacheon's check never builds.

`make_mix_stats_impl(stock_impl)` returns a drop-in for the sealed `deepseek_v4_mhc._mix_stats_impl(hc, x)` (d3a157a2,
identical to 5884e4fe here): stock's own projection launches -- Triton `_hc_mix_stats_partial_kernel` (tf32x3, 80
slices) below 32 rows, DeepGEMM `tf32_hc_prenorm_gemm` on the tf32 hi / lo parts (16 slices) from 32 rows under
stock's gate -- then the JIT reduce in place of Triton `_hc_mix_reduce_sinkhorn_kernel`; the triplet is stock's bit
for bit. Outside the domain it calls the `stock_impl` it was built with (the sealed function object captured by the
caller, never looked up through the module global, which a bundle may have rebound to this drop-in).
`make_mix_stats(mix_stats_impl)` wraps it in `mix_stats`' stream discipline.

pd001's CAP (a capped Triton projection at 9..31 rows) is not carried: the projection launches are stock's.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

_K = 20480
_MIX = 24
_MAX_ROWS = 4095  # below stock's bf16x3 projection (4096..65536 rows)


def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial header (as mega_front._write)."""
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
    """dsv41-pd006: cuda_hcsk.SOURCE under SGLang's JIT cache root, content-addressed, written on first use."""
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from dsv41_pd001 import cuda_hcsk

    data = cuda_hcsk.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd006",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_hcsk.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _module():
    """One load_jit module, two entries: `run_triton` (<= 80 slices) and `run_deepgemm` (<= 16 slices + lo partials).
    Built with load_jit's default flags for the local GPU into $SGLANG_JIT_CACHE_DIR (default ~/.cache/sglang/jit)."""
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "pd002_hc_sinkhorn_reduce",
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
    """pd001's domain: the SM10x decode / verify shapes the sealed `_mix_stats_impl` sends to its Triton or DeepGEMM
    projection plus the Triton reduce (bf16 rows, mult 4, K 20480, 24 mixes, 1..4095 rows, batch invariance off)."""
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
    """A `_mix_stats_impl(hc, x)` drop-in bound to `stock_impl` (pass the sealed function object itself)."""

    def mix_stats_impl(hc, x: torch.Tensor):
        from sglang.kernels.ops.layernorm import mhc as ops
        from sglang.srt.environ import envs

        x_flat = x.flatten(1)
        if not _in_domain(hc, x_flat):
            return stock_impl(hc, x)
        cfg = hc.cfg
        m, k = x_flat.shape
        # stock's DeepGEMM gate (SM10x, < 4096 rows: the bf16x3 branch cannot apply)
        parts = (
            hc.tf32_parts
            if m >= 32 and x_flat.is_contiguous() and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
            else None
        )
        if parts is not None:
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

            # hc_mix_stats_sinkhorn_deepgemm's projection, verbatim
            slices = ops._HC_MIX_COMPENSATED_SLICES
            high, low = parts
            mix_hi = torch.empty((slices, m, _MIX), dtype=torch.float32, device=x.device)
            mix_lo = torch.empty_like(mix_hi)
            sq = torch.empty((slices, m), dtype=torch.float32, device=x.device)
            tf32_hc_prenorm_gemm(x_flat, high, mix_hi, sq, slices)
            tf32_hc_prenorm_gemm(x_flat, low, mix_lo, sq, slices)
            return _reduce(mix_hi, mix_lo, sq, hc.scale, hc.base, m, cfg.sinkhorn_iters, cfg.rms_eps, cfg.eps)
        # hc_mix_stats_sinkhorn's projection, verbatim (stock's kernel, grid and meta-parameters)
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
    """A `mix_stats(hc, x, stats_stream=None)` drop-in: the sealed function's stream discipline around `mix_stats_impl`."""

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
