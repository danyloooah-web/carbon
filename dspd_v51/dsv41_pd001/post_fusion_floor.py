"""pd002 MHC: a copy of the sealed `sglang.srt.models.deepseek_v4_mhc._post_fusion` (d3a157a2 + ports, image
cacheon-dsv41pd-proxy:d3a157a2-ca8e5b23-v3) in which decode-sized, decode-like boundaries skip the Mega-mHC kernel.

Under the floor (rows <= MEGA_MHC_DECODE_MAX_ROWS, a decode / target-verify forward, every gate stock's decode
post-combine branch checks), a boundary that stock would hand to `mhc_mega_boundary` instead runs
`mhc_post_combine_norm_decode` -- the SGLANG_OPT_DSV41_MHC_POST_COMBINE_DECODE branch, taken without reading that env
var -- and returns no triplet, so the next `_compute_triplet` runs `mix_stats` on the stats stream beside the sublayer
(`use_stats_stream` is true for exactly these forwards). For a next sublayer that takes DeepGEMM's MXFP8 pair
(`accepts_mxfp8_deepgemm`: wqkv_a under flashmla_mega), the same kernel writes the pair in DeepGEMM's packed scale
layout (mhc_post_combine_norm_decode_dg.cuh), so wqkv_a keeps stock's DeepGEMM consumer and input format. Everything
else -- extend / prefill-graph forwards, rows above the floor, no `next`, gates stock's branch rejects -- is stock's
code path unchanged.

Anchored edits against the stock body (marked `# pd002`): the `mega` predicate is named and gains `not skip_mega`;
the decode branch is also entered for `skip_mega`; under `skip_mega` a DeepGEMM-pair consumer gets the DG-scale kernel
and both decode kernels take 320 threads above one wave of 640-thread CTAs (`_decode_threads`).
"""

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

# pd002: decode-like boundaries at or below this many rows a rank skip Mega-mHC. Layer proxy (graph replay, B300):
# skipping wins 10-15 us a layer at 8-512 rows and breaks even at 1024 (generator/pd002/mhc/NOTES.md).
MEGA_MHC_DECODE_MAX_ROWS = 216  # dsv41-pd021 (as pd013): the replay's tail (verify buckets <= 36 requests a rank); the steady state's 264-288 rows keep Mega-mHC

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
def _dg_header() -> str:
    """dsv41-pd004: cuda_mhc_dg.SOURCE under SGLang's JIT cache root, content-addressed, written on first use."""
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from dsv41_pd001 import cuda_mhc_dg

    data = cuda_mhc_dg.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd004",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_mhc_dg.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _dg_module(threads: int):
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "pd002_mhc_post_combine_norm_decode_dg",
        str(threads),
        cuda_files=[_dg_header()],
        cuda_wrappers=[("run", f"MhcPostCombineNormDecodeDg<{threads}>::run")],
    )


@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _decode_threads(rows: int, device: torch.device) -> int:
    """640 threads a row while one CTA an SM covers the rows, else 320. A 640-thread CTA of this kernel holds 62
    registers a thread (39,680 of an SM's 65,536), so only one fits an SM and row 149 starts a second wave; stock's
    `_pick_threads` counts threads only and keeps 640 to 444 rows (measured: 7.2-7.5 us vs 5.6-5.7 us a call at
    160-288 rows; equal at <= 144)."""
    return 640 if rows <= _num_sms(device.index) else 320


def mhc_post_combine_norm_decode_dg(y, residual, post_mix, comb, pre, weight, eps):
    """``(updated, normalized, (fp8, DeepGEMM-packed UE8M0 scales))``: stock's decode post-combine arithmetic with the
    pair Mega-mHC hands a `deepgemm_mxfp8_input` linear."""
    updated, normalized = torch.empty_like(residual), torch.empty_like(y)
    y_fp8, y_sf = empty_deepgemm_mxfp8_pair(*y.shape, y.device)
    _dg_module(_decode_threads(y.shape[0], y.device)).run(
        y, residual, post_mix, comb, pre, weight, updated, normalized, y_fp8, y_sf, eps
    )
    return updated, normalized, (y_fp8, y_sf)


def _decode_like() -> bool:
    """Decode / target-verify forwards with a stats stream: the ones whose next triplet `use_stats_stream` puts on
    `hc_stats_stream` (created only under SGLANG_OPT_USE_MULTI_STREAM_OVERLAP). Mirrors `fused_mxfp8_rows`' extend
    test (graph runners clear `is_extend_in_batch` around decode capture)."""
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
    """Step 4 without a collective: the wide-tile kernel folds the next combine +
    norm in where it serves this seam, otherwise the pure post runs and the next
    combine computes itself."""
    cfg = hc.cfg
    pre, post_mix, comb = coefficients
    mega = (  # pd002: stock's mega predicate, named
        next is not None
        and next.norm_fusable
        and next.hc is not None
        and _can_use_mega_mhc(cfg)
        and y.shape[0] > 0
    )
    skip_mega = (  # pd002: decode-sized, decode-like boundaries take the decode post-combine branch
        mega
        and y.shape[0] <= MEGA_MHC_DECODE_MAX_ROWS
        and _decode_like()
        and _can_use_decode_post_combine(cfg, y, residual, coefficients)
    )
    if mega and not skip_mega:  # pd002: `and not skip_mega`
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
        # pd002: the pair stock's mega branch would have produced, from stock's decode post-combine arithmetic
        updated, normalized, pair = mhc_post_combine_norm_decode_dg(
            y, residual, post_mix, comb, pre, next.norm.weight, next.norm.variance_epsilon
        )
        return HcState(updated, pre, HcQuantized(normalized, pair))
    if (
        next is not None
        and next.norm_fusable
        and (skip_mega or envs.SGLANG_OPT_DSV41_MHC_POST_COMBINE_DECODE.get())  # pd002: `skip_mega or`
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
            threads=_decode_threads(y.shape[0], y.device) if skip_mega else None,  # pd002
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
