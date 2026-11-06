from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

from bundle_v9.dec import prenorm_pair

_K = 20480
_MIX = 24
_MAX_ROWS = 4095
# king-v6 RSK-4ROW (RANKED #38, ships with FFN-PAIR #9; review R3). The residual (DeepGEMM-prenorm) reduce_sinkhorn of
# a call with RSK4_ROWS[0]..RSK4_ROWS[1] rows launches the king's kernel at kRows 4 (cuda_sinkhorn.rsk4_source(), grid
# ceil(rows / 4)) instead of kRows 1 (grid = rows, 3 of 4 warps idle): a quarter of the CTAs, so the FFN-side reduce
# that follows a one-CTA FFN-PAIR chain needs fewer free SMs beside MegaMoE. Bitwise by construction and proven at
# prepare against the king's launch (prepare_rsk4, called from prenorm_pair.prepare). Rank-local; no allocation.
# The lead's switch. True: the band below. False: the king's kRows-1 launch everywhere, exactly the king_v2e path.
RSK4_ON = True
RSK4_ROWS = (32, 320)
_RSK4: list = []
_RSK4_STATE = {"on": False, "why": "not prepared"}
_RSK4_WARNED = [False]


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

    from bundle_v9.dec import cuda_sinkhorn

    data = cuda_sinkhorn.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-mhc-stats",
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
        "dec_hc_sinkhorn_reduce",
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
    elif not _rsk4(part_mix, residual, part_sq, scale, base, pre, post, comb, m, sinkhorn_iters, rms_eps, hc_eps):
        mod.run_deepgemm(part_mix, residual, part_sq, scale, base, pre, post, comb, 1.0 / _K, float(rms_eps),
                         int(sinkhorn_iters), float(hc_eps))
    return pre, post, comb


@lru_cache(maxsize=None)
def _rsk4_text() -> tuple:
    import hashlib

    from bundle_v9.dec import cuda_sinkhorn

    data = cuda_sinkhorn.rsk4_source().encode()
    return data, hashlib.sha256(data).hexdigest()[:16]


@lru_cache(maxsize=None)
def _rsk4_header() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9.dec import cuda_sinkhorn

    data, digest = _rsk4_text()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-mhc-stats-rsk4", digest)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_sinkhorn.RSK4_NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _rsk4_module():
    from sglang.kernels.jit.utils import load_jit

    from bundle_v9.dec import cuda_sinkhorn

    return load_jit("dec_hc_sinkhorn_reduce4", _rsk4_text()[1], cuda_files=[_rsk4_header()],
                    cuda_wrappers=[("run_deepgemm4", cuda_sinkhorn.RSK4_ENTRY)])


def _rsk4_mark(on: bool, why: str) -> None:
    _RSK4_STATE["on"] = bool(on)
    _RSK4_STATE["why"] = why


def rsk4_line() -> str:
    return f"bundle-v9 king-v6 RSK-4ROW: {'ON' if _RSK4_STATE['on'] else 'off'} ({_RSK4_STATE['why']})"


def _rsk4(part_mix, residual, part_sq, scale, base, pre, post, comb, m, sinkhorn_iters, rms_eps, hc_eps) -> bool:
    """True when the kRows-4 launch wrote pre, post and comb; False means: run the king's kRows-1 launch."""
    if not _RSK4 or not _RSK4[0] <= m <= _RSK4[1]:
        return False
    try:
        _rsk4_module().run_deepgemm4(part_mix, residual, part_sq, scale, base, pre, post, comb, 1.0 / _K,
                                     float(rms_eps), int(sinkhorn_iters), float(hc_eps))
    except Exception as error:  # fail-closed: the king's launch rewrites all three outputs
        _RSK4.clear()
        _rsk4_mark(False, f"disabled at run time after a launch error, the king's kRows-1 launch from here on: "
                          f"{type(error).__name__}: {str(error)[:300]}")
        if not _RSK4_WARNED[0]:
            _RSK4_WARNED[0] = True
            print(rsk4_line(), file=sys.stderr, flush=True)
        return False
    return True


def _same(a, b) -> bool:
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _rsk4_rows(first: int, last: int) -> list:
    mid = (first + last) // 2
    mid += (2 - mid % 4) % 4
    return sorted({r for r in (first + 1, mid, last) if first <= r <= last} or {first})


def _prove_rsk4(hcs, first: int, last: int) -> int:
    """kRows 4 against the king's kRows 1 on NaN-filled outputs, int32 compare: every row count of the band with the
    first constants set, then 3 row counts (row % 4 = 1, 2, 0) with every layer's own attention- and FFN-side
    scale / base / rms_eps / eps / sinkhorn iters. Inputs are stock-shaped: [16, m, 24] hi / lo parts, [16, m] sums."""
    from sglang.kernels.ops.layernorm import mhc as ops

    slices = int(ops._HC_MIX_COMPENSATED_SLICES)
    if slices != 16:
        raise RuntimeError(f"stock projects {slices} prenorm slices, the kRows-4 launcher is built for 16")
    device = hcs[0][2].scale.device
    gen = torch.Generator(device=device).manual_seed(20261106 + 38)
    mod, mod4 = _module(), _rsk4_module()
    plan = [(hcs[0], rows, 0) for rows in range(first, last + 1)]
    plan += [(entry, rows, 1) for entry in hcs for rows in _rsk4_rows(first, last)]
    for (index, side, hc), rows, family in plan:
        cfg = hc.cfg
        pm = torch.randn((slices, rows, _MIX), device=device, generator=gen, dtype=torch.float32) * 4.0
        lo = torch.randn((slices, rows, _MIX), device=device, generator=gen, dtype=torch.float32) * 1e-3
        sq = torch.rand((slices, rows), device=device, generator=gen, dtype=torch.float32) * (_K / slices) * 0.5
        if family == 1:
            pm[:, rows // 2].zero_()
            lo[:, rows // 2].fill_(-0.0)
            sq[:, rows // 2].zero_()
            pm[:, rows // 3] *= 1e4
        want = (torch.full((rows, 4), float("nan"), device=device), torch.full((rows, 4), float("nan"), device=device),
                torch.full((rows, 4, 4), float("nan"), device=device))
        got = tuple(torch.full_like(t, float("nan")) for t in want)
        args = (1.0 / _K, float(cfg.rms_eps), int(cfg.sinkhorn_iters), float(cfg.eps))
        mod.run_deepgemm(pm, lo, sq, hc.scale, hc.base, *want, *args)
        mod4.run_deepgemm4(pm, lo, sq, hc.scale, hc.base, *got, *args)
        torch.cuda.synchronize(device)
        if not all(_same(a, b) for a, b in zip(got, want)):
            raise RuntimeError(f"layer {index} {side} kRows 4 differs from the king's kRows 1 at {rows} rows")
    return len(plan)


def prepare_rsk4(model) -> None:
    """king-v6 RSK-4ROW prepare (decode server; called from prenorm_pair.prepare). Never raises: the king's kRows-1
    launch serves on any failure."""
    _RSK4.clear()
    if not RSK4_ON:
        _rsk4_mark(False, "switched off by mhc_stats.RSK4_ON; the king's kRows-1 reduce_sinkhorn launch")
        print(rsk4_line(), file=sys.stderr, flush=True)
        return
    try:
        first, last = (int(v) for v in RSK4_ROWS)
        if not 32 <= first <= last <= _MAX_ROWS:
            raise RuntimeError(f"RSK4_ROWS {RSK4_ROWS!r} is not a band inside 32..{_MAX_ROWS}")
        hcs = []
        for index, layer in enumerate(model.layers):
            for side, hc in (("attn_hc", layer.attn_hc), ("ffn_hc", layer.ffn_hc)):
                if hc is None or hc.scale.dtype != torch.float32 or hc.base.dtype != torch.float32 or \
                        not hc.scale.is_contiguous() or not hc.base.is_contiguous() or hc.scale.numel() != 3 or \
                        hc.base.numel() != _MIX:
                    raise RuntimeError(f"layer {index} {side} has no float32 [3] scale / [24] base")
                hcs.append((index, side, hc))
        if not hcs:
            raise RuntimeError("the model has no layers")
        cases = _prove_rsk4(hcs, first, last)
    except Exception as error:  # fail-closed: the king's kRows-1 launch everywhere
        _rsk4_mark(False, f"FAIL, the king's kRows-1 reduce_sinkhorn launch: {type(error).__name__}: "
                          f"{str(error)[:300]}")
        print(rsk4_line(), file=sys.stderr, flush=True)
        return
    _RSK4.extend((first, last))
    _rsk4_mark(True, f"decode residual reduce_sinkhorn at kRows 4 (grid ceil(rows / 4)) at {first}..{last} rows, the "
                     f"king's kRows-1 launch elsewhere; bytes proven against the king's launch on {cases} cases over "
                     f"{len(hcs)} attention / FFN constant sets")
    print(rsk4_line(), file=sys.stderr, flush=True)


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
            if prenorm_pair.serves(hc, m):
                prenorm_pair.run(x_flat, high, low, mix_hi, mix_lo, sq)
            elif (tail := prenorm_pair.run_ffn_tail(hc, x_flat, high, low, mix_hi, mix_lo, sq, cfg.sinkhorn_iters,
                                                    cfg.rms_eps, cfg.eps)) is not None:
                return tail  # king-v7 STATS-TAIL: FFN-PAIR's launch also wrote pre, post and comb (no reduce launch)
            elif prenorm_pair.run_ffn(hc, x_flat, high, low, mix_hi, mix_lo, sq):
                pass  # king-v6 FFN-PAIR: hi, lo and the square sums in one launch on the stats stream
            else:
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
