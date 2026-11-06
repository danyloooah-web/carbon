
import os
from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.models.deepseek_v4_mhc import (
    HcNextBoundary,
    HcNormed,
    HcQuantized,
    HcState,
    HcSubLayer,
    HcTriplet,
    Mxfp8SwizzledInput,
    _can_use_decode_post_combine,
    _can_use_mega_mhc,
    empty_deepgemm_mxfp8_pair,
    envs,
    fused_mxfp8_rows,
    get_forward,
    get_parallel,
    get_platform,
    is_in_prefill_graph_scope,
    mhc_mega_boundary,
    post,
)

MEGA_MHC_DECODE_MAX_ROWS = 216

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
def _dg_header() -> str:
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9 import cuda_mhc_packed

    data = cuda_mhc_packed.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_mhc_packed.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _dg_module(threads: int):
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "v4_mhc_post_combine_norm_decode_dg",
        str(threads),
        cuda_files=[_dg_header()],
        cuda_wrappers=[("run", f"MhcPostCombineNormDecodeDg<{threads}>::run")],
    )


@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _decode_threads(rows: int, device: torch.device) -> int:
    return 640 if rows <= _num_sms(device.index) else 320


def mhc_post_combine_norm_decode_dg(y, residual, post_mix, comb, pre, weight, eps):
    updated, normalized = torch.empty_like(residual), torch.empty_like(y)
    y_fp8, y_sf = empty_deepgemm_mxfp8_pair(*y.shape, y.device)
    _dg_module(_decode_threads(y.shape[0], y.device)).run(
        y, residual, post_mix, comb, pre, weight, updated, normalized, y_fp8, y_sf, eps
    )
    return updated, normalized, (y_fp8, y_sf)


def _decode_like() -> bool:
    fwd = get_forward()
    return envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and not (
        fwd.sp_active or fwd.is_extend_in_batch or is_in_prefill_graph_scope()
    )


def _post_fusion(
    hc: HcSubLayer,
    y: torch.Tensor,
    residual: torch.Tensor,
    coefficients: HcTriplet,
    next: Optional[HcNextBoundary],
) -> HcState:
    cfg = hc.cfg
    pre, post_mix, comb = coefficients
    mega = (
        next is not None
        and next.norm_fusable
        and next.hc is not None
        and _can_use_mega_mhc(cfg)
        and y.shape[0] > 0
    )
    skip_mega = (
        mega
        and y.shape[0] <= MEGA_MHC_DECODE_MAX_ROWS
        and _decode_like()
        and _can_use_decode_post_combine(cfg, y, residual, coefficients)
    )
    if mega and not skip_mega:
        next_hc = next.hc
        y_fp8 = y_sf = moe_input = None
        if next.accepts_mxfp8_deepgemm and fused_mxfp8_rows(y.shape[0]):
            y_fp8, y_sf = empty_deepgemm_mxfp8_pair(*y.shape, y.device)
        elif next.mega_moe_input is not None:
            moe_input = next.mega_moe_input(y)
        updated, normalized, stats = mhc_mega_boundary(
            y,
            residual,
            pre,
            post_mix,
            comb,
            next_hc.fn,
            next_hc.scale,
            next_hc.base,
            next_hc.norm.weight,
            next_hc.cfg.rms_eps,
            next_hc.cfg.eps,
            next_hc.norm.variance_epsilon,
            next_hc.cfg.sinkhorn_iters,
            y_fp8=y_fp8,
            y_gemm_sf=y_sf,
            moe_input=None if moe_input is None else moe_input.views(),
        )
        if y_fp8 is not None:
            return HcState(updated, pre, HcQuantized(normalized, (y_fp8, y_sf)), stats)
        if moe_input is not None:
            moe_input.mark(normalized)
        return HcState(updated, pre, HcNormed(normalized), stats)
    if skip_mega and next.accepts_mxfp8_deepgemm and fused_mxfp8_rows(y.shape[0]):
        updated, normalized, pair = mhc_post_combine_norm_decode_dg(
            y, residual, post_mix, comb, pre, next.norm.weight, next.norm.variance_epsilon
        )
        return HcState(updated, pre, HcQuantized(normalized, pair))
    if (
        next is not None
        and next.norm_fusable
        and (skip_mega or envs.SGLANG_OPT_DSV41_MHC_POST_COMBINE_DECODE.get())
        and _can_use_decode_post_combine(cfg, y, residual, coefficients)
    ):
        from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import (
            mhc_post_combine_norm_decode,
        )

        quant = next.accepts_mxfp8
        updated, normalized, q, sf = mhc_post_combine_norm_decode(
            y,
            residual,
            post_mix,
            comb,
            pre,
            next.norm.weight,
            next.norm.variance_epsilon,
            quant=quant,
            threads=_decode_threads(y.shape[0], y.device) if skip_mega else None,
        )
        if quant:
            return HcState(
                updated, pre, HcQuantized(normalized, Mxfp8SwizzledInput(q, sf))
            )
        return HcState(updated, pre, HcNormed(normalized))
    if (
        next is not None
        and next.norm_fusable
        and cfg.pre_from_prev
        and get_platform().is_blackwell
        and 4096 <= y.shape[0] <= 65536
        and cfg.hidden == 5120
        and cfg.mult == 4
        and get_parallel().attn_dp_size == 1
        and not cfg.cp_prefill
    ):
        from sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill import (
            mhc_post_combine_norm_prefill,
        )

        updated, normalized = mhc_post_combine_norm_prefill(
            y,
            residual,
            post_mix,
            comb,
            pre,
            next.norm.weight,
            next.norm.variance_epsilon,
        )
        return HcState(updated, pre, HcNormed(normalized))
    return HcState(post(cfg, y, residual, post_mix, comb), pre)
