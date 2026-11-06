from __future__ import annotations

import hashlib
import os
import sys
import time
from functools import lru_cache

import torch

from bundle_v9.dec import cuda_prenorm_pair

_K = 20480
_MIX = 24
_SPLITS = 16
_CTAS = 16
_MIN_ROWS = 32
_MAX_ROWS = 320
_QN: set = set()
_ATTN: set = set()
_ARM: list = [None]
# king-v6 FFN-PAIR (IDEAS_dec_mhc M1 via king_v2d; RANKED F-F / #9). The FFN-side mHC stats of a decode / verify call run
# the king's pair kernel above on the model's stats stream instead of stock's two DeepGEMM launches, with the CTA count
# of the call's row band. king_v2d served one CTA at 32..320 rows; the mHC verifier (M1 corrections 1-2) puts one CTA at
# 125-250 us at 257-288 rows against ~279 us of MegaMoE, and an overrun costs milliseconds a step, so the served range is
# capped at _FFN_MAX_ROWS = 192 and rows above it run stock until the pair microbench (RANKED T3) sets ctas per band.
# The king's attention-side range (_MIN_ROWS.._MAX_ROWS, ctas _CTAS) is unchanged.
# The lead's switch. True: FFN-side stats in the bands below. False: stock's two launches, exactly the king_v2e path.
FFN_PAIR_ON = True
_FFN_MAX_ROWS = 192
# (first row, last row, ctas) per band, within _MIN_ROWS.._FFN_MAX_ROWS. ctas 1: the king's compiled module on one CTA
# (one SM). ctas 2: one 2-CTA cluster (one TPC), cuda_prenorm_pair.cluster_source(), compiled only if a band selects it.
FFN_PAIR_BANDS = ((32, 192, 1),)
_FFN: set = set()
_FFN_BANDS: list = []
_STATS: list = [None]
_FFN_STATE = {"on": False, "why": "not prepared"}
_FFN_WARNED = [False]
# review R5: the ctas-1 (16 splits in one CTA) path first runs in _prove_ffn. A hang there cannot fail closed, so the
# proof prints a start marker and, if one launch is still running after this many seconds, one WATCHDOG line.
_FFN_WATCHDOG_S = 60.0
_FFN_HUNG = [False]
# king-v7 FFN-PAIR-W (struct/king_lane/v7/mhc/NOTES.md; CRITPATH #9; VERIFY_dec_mhc M1 correction 2). v6u's decode trace
# (rank 0): the one-CTA FFN pair takes 155 us at 65..128 rows and 222..230 us at 129..192 rows beside MegaMoE and ends
# after MegaMoE in 1.0..3.3 layers a replay at 129..192 rows; above 192 rows stock's second prenorm GEMM, starved on the
# TPC MegaMoE leaves free, ends after MegaMoE in 16.8 layers a replay at 264 rows (135 us a replay). FFN-PAIR-W serves the
# FFN side at FFN_PAIR_W_BANDS instead of FFN_PAIR_BANDS: ctas 1 on the king's module, ctas 2 as one 2-CTA cluster on the
# dec_prenorm_pair_x module (device code = the king's, PTX-identical), whose two CTAs share the one TPC MegaMoE's 73
# two-CTA clusters leave free (a %smid probe at prepare checks that every 2-CTA cluster lands on one TPC). The proof covers
# the served forms at every 64-row block edge and the 240 / 264 / 288-row buckets. Any FFN-PAIR-W failure (or
# FFN_PAIR_W_ON = False) leaves v6t's FFN_PAIR_BANDS serving. That is exactly the king_v6t path only with
# STATS_TAIL_ON = False as well (king-v7 fixer, review F6): STATS-TAIL is its own lever and still serves v6t's 32..192
# band through the dec_prenorm_pair_x kTail kernel at ctas 1, with no separate reduce. The lead's switch.
FFN_PAIR_W_ON = True
FFN_PAIR_W_BANDS = ((32, 128, 1), (129, 320, 2))
_X_CLUSTER = [False]
_W_STATE = {"on": False, "why": "not prepared"}
# king-v7 STATS-TAIL (IDEAS_dec_mhc M2, VERIFY_dec_mhc M2). An FFN-side call that FFN-PAIR serves runs the reduce +
# Sinkhorn in the pair kernel's tail (dec_prenorm_pair_x, kTail) instead of a separate reduce_sinkhorn launch: the chain
# beside MegaMoE is one kernel, ~20-27 us shorter (v6u: the FFN reduce takes 14..30 us after the pair). Bitwise by
# replicating dec_hcsk's op sequence lane by lane; proven at prepare against stock's two DeepGEMM launches + the king's
# kRows-1 reduce (all six outputs, NaN-filled) for every layer's FFN constants at every served CTA count. Attention side
# not served (the verifier: a fused tail needs ctas <= 2 there, and the attention chain has >= 34 us of slack). Any
# failure: the FFN-PAIR launch + mhc_stats._reduce, exactly the path without it. The lead's switch.
STATS_TAIL_ON = True
_TAIL: set = set()
_TAIL_KEYS: set = set()
_TAIL_STATE = {"on": False, "why": "not prepared"}
_TAIL_WARNED = [False]
# king-v7 ATTN-PAIR-WIDTH (IDEAS_dec_mhc M3, VERIFY_dec_mhc M3). The attention-side pair's CTA count per row band
# (ATTN_PAIR_BANDS, ctas 1/2/4/8/16; rows outside every band keep the king's _CTAS) with a bitwise proof at each served
# ctas (nsplit > 1 is new on the attention side). Ships OFF: in v6u's trace at 240 rows the 16-CTA pair (23 us) overlaps
# wq_b (launched first, 132 CTAs, keeps its SMs) and VATTN, whose duration does not depend on the overlap (slope 0.02 us
# per us), and the attention chain ends >= 34 us (p10) before the attention post; a narrower pair only stretches the
# same SM-time over VATTN. The lead's switch, for an A/B arm.
ATTN_WIDTH_ON = False
ATTN_PAIR_BANDS = ((32, 320, 8),)
_ATTN_BANDS: list = []
_AW_STATE = {"on": False, "why": "not prepared"}


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


def _digest() -> str:
    return hashlib.sha256(cuda_prenorm_pair.SOURCE.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def _header() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-prenorm-pair", _digest())
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, os.path.basename(cuda_prenorm_pair.NAME))
    if not os.path.isfile(path):
        _write(path, cuda_prenorm_pair.SOURCE.encode())
    return path


@lru_cache(maxsize=None)
def _module():
    import deep_gemm
    from sglang.kernels.jit.utils import load_jit

    include = os.path.join(os.path.dirname(os.path.abspath(deep_gemm.__file__)), "include")
    return load_jit(
        "dec_prenorm_pair",
        _digest(),
        cuda_files=[_header()],
        cuda_wrappers=[("run", "dec_prenorm_pair::PrenormPair::run")],
        extra_cuda_cflags=["--use_fast_math"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[include],
        extra_dependencies=["cutlass"],
    )


def mark(weight: torch.Tensor) -> None:
    if weight.data_ptr() in _QN:
        stream = torch.cuda.current_stream()
        _ARM[0] = (stream, stream.record_event())


def serves(hc, rows: int) -> bool:
    if hc.fn.data_ptr() not in _ATTN:
        return False
    armed, _ARM[0] = _ARM[0], None
    if armed is None or not _MIN_ROWS <= rows <= _MAX_ROWS:
        return False
    stream = torch.cuda.current_stream()
    if stream == armed[0]:
        return False
    stream.wait_event(armed[1])
    return True


def _attn_ctas(rows: int) -> int:
    for first, last, ctas in _ATTN_BANDS:
        if first <= rows <= last:
            return ctas
    return _CTAS


def run(x_flat, high, low, mix_hi, mix_lo, sq) -> None:
    _module().run(x_flat, high, low, mix_hi, mix_lo, sq, _attn_ctas(x_flat.shape[0]) if _ATTN_BANDS else _CTAS)


@lru_cache(maxsize=None)
def _cluster_header() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data = cuda_prenorm_pair.cluster_source().encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-prenorm-pair-cluster", digest)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, os.path.basename(cuda_prenorm_pair.CLUSTER_NAME))
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _cluster_module():
    import deep_gemm
    from sglang.kernels.jit.utils import load_jit

    include = os.path.join(os.path.dirname(os.path.abspath(deep_gemm.__file__)), "include")
    digest = hashlib.sha256(cuda_prenorm_pair.cluster_source().encode()).hexdigest()[:16]
    return load_jit(
        "dec_prenorm_pair_cluster",
        digest,
        cuda_files=[_cluster_header()],
        cuda_wrappers=[("run", cuda_prenorm_pair.CLUSTER_ENTRY)],
        extra_cuda_cflags=["--use_fast_math"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[include],
        extra_dependencies=["cutlass"],
    )


@lru_cache(maxsize=None)
def _x_text() -> tuple:
    data = cuda_prenorm_pair.x_source().encode()
    return data, hashlib.sha256(data).hexdigest()[:16]


@lru_cache(maxsize=None)
def _x_header() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data, digest = _x_text()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-prenorm-pair-x", digest)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, os.path.basename(cuda_prenorm_pair.X_NAME))
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def _x_module():
    import deep_gemm
    from sglang.kernels.jit.utils import load_jit

    include = os.path.join(os.path.dirname(os.path.abspath(deep_gemm.__file__)), "include")
    ns = cuda_prenorm_pair.X_NAMESPACE
    return load_jit(
        "dec_prenorm_pair_x",
        _x_text()[1],
        cuda_files=[_x_header()],
        cuda_wrappers=[("run", f"{ns}::PrenormPair::run"), ("run_tail", f"{ns}::PrenormPair::run_tail"),
                       ("probe", f"{ns}::TpcProbe::run")],
        extra_cuda_cflags=["--use_fast_math"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[include],
        extra_dependencies=["cutlass"],
    )


def _run_ffn_kernel(x_flat, high, low, mix_hi, mix_lo, sq, ctas: int) -> None:
    if ctas == 2:
        if _X_CLUSTER[0]:  # king-v7 FFN-PAIR-W: the cluster form of the dec_prenorm_pair_x module
            _x_module().run(x_flat, high, low, mix_hi, mix_lo, sq, 2)
        else:
            _cluster_module().run(x_flat, high, low, mix_hi, mix_lo, sq, 2)
    else:
        _module().run(x_flat, high, low, mix_hi, mix_lo, sq, ctas)


def _ffn_ctas(rows: int) -> int:
    for first, last, ctas in _FFN_BANDS:
        if first <= rows <= last:
            return ctas
    return 0


def _aligned(t) -> bool:
    return t.data_ptr() % 16 == 0


def _serves_ffn(hc, x_flat, high, low) -> int:
    """The served CTA count of an FFN-side call, or 0. Only on the model's stats stream: the engine's mix_stats runs the
    stats there (`with torch.cuda.stream(stats_stream)`), never on the forward's own stream, so a one-CTA pair never sits
    on the main stream. The Python checks make the C++ RuntimeChecks unreachable."""
    if not _FFN or hc.fn.data_ptr() not in _FFN:
        return 0
    ctas = _ffn_ctas(x_flat.shape[0])
    stats = _STATS[0]
    if not ctas or stats is None or not torch.cuda.current_stream() == stats:
        return 0
    if not (x_flat.is_contiguous() and x_flat.dtype == torch.bfloat16 and _aligned(x_flat) and _aligned(high)
            and _aligned(low) and tuple(high.shape) == (_MIX, _K) and tuple(low.shape) == (_MIX, _K)
            and high.dtype == torch.float32 and low.dtype == torch.float32
            and high.is_contiguous() and low.is_contiguous()):
        return 0
    return ctas


def run_ffn(hc, x_flat, high, low, mix_hi, mix_lo, sq) -> bool:
    """True when the FFN pair wrote hi, lo and the square sums; False means: run stock's two launches."""
    try:
        ctas = _serves_ffn(hc, x_flat, high, low)
    except Exception:
        return False
    if not ctas:
        return False
    try:
        _run_ffn_kernel(x_flat, high, low, mix_hi, mix_lo, sq, ctas)
    except Exception as error:  # fail-closed: stock rewrites all three outputs
        _FFN.clear()
        _FFN_BANDS.clear()
        _ffn_mark(False, f"disabled at run time after a launch error, stock prenorm from here on: "
                         f"{type(error).__name__}: {str(error)[:300]}")
        if not _FFN_WARNED[0]:
            _FFN_WARNED[0] = True
            print(ffn_line(), file=sys.stderr, flush=True)
        return False
    return True


def run_ffn_tail(hc, x_flat, high, low, mix_hi, mix_lo, sq, sinkhorn_iters, rms_eps, hc_eps):
    """king-v7 STATS-TAIL: (pre, post, comb) of an FFN-side call that FFN-PAIR serves, written by the pair kernel's
    tail (hi, lo and the square sums are written too); None means: run_ffn, then mhc_stats._reduce."""
    if not _TAIL:
        return None
    try:
        ctas = _serves_ffn(hc, x_flat, high, low)
        key = (int(sinkhorn_iters), float(rms_eps), float(hc_eps))
        if not ctas or ctas not in _TAIL or key not in _TAIL_KEYS:
            return None
        if hc.scale.data_ptr() % 16 or hc.base.data_ptr() % 16 or not hc.scale.is_contiguous() or \
                not hc.base.is_contiguous():
            return None
    except Exception:
        return None
    m = x_flat.shape[0]
    pre = torch.empty(m, 4, dtype=torch.float32, device=x_flat.device)
    post = torch.empty(m, 4, dtype=torch.float32, device=x_flat.device)
    comb = torch.empty(m, 4, 4, dtype=torch.float32, device=x_flat.device)
    try:
        _x_module().run_tail(x_flat, high, low, mix_hi, mix_lo, sq, hc.scale, hc.base, pre, post, comb, 1.0 / _K,
                             float(rms_eps), int(sinkhorn_iters), float(hc_eps), ctas)
    except Exception as error:  # fail-closed: run_ffn + the reduce rewrite every output
        _TAIL.clear()
        _tail_mark(False, f"disabled at run time after a launch error, FFN-PAIR + reduce_sinkhorn from here on: "
                          f"{type(error).__name__}: {str(error)[:300]}")
        if not _TAIL_WARNED[0]:
            _TAIL_WARNED[0] = True
            print(tail_line(), file=sys.stderr, flush=True)
        return None
    return pre, post, comb


def _stock(x_flat, high, low):
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    m = x_flat.shape[0]
    mix_hi = torch.empty((_SPLITS, m, _MIX), dtype=torch.float32, device=x_flat.device)
    mix_lo = torch.empty_like(mix_hi)
    sq = torch.empty((_SPLITS, m), dtype=torch.float32, device=x_flat.device)
    tf32_hc_prenorm_gemm(x_flat, high, mix_hi, sq, _SPLITS)
    tf32_hc_prenorm_gemm(x_flat, low, mix_lo, sq, _SPLITS)
    return mix_hi, mix_lo, sq


def _same(a, b) -> bool:
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _prove(parts, device) -> None:
    gen = torch.Generator(device=device).manual_seed(20261103)
    for index, (high, low) in enumerate(parts):
        for rows in ((32, 36, 64, 66, 96, 128, 144, 192, 216, 240, 256, 264, 288, 300, 318, 320) if index == 0
                     else (48, 288)):
            x = (torch.randn((rows, _K), device=device, generator=gen, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            want = _stock(x, high, low)
            got = (torch.full_like(want[0], float("nan")), torch.full_like(want[1], float("nan")),
                   torch.full_like(want[2], float("nan")))
            run(x, high, low, *got)
            torch.cuda.synchronize(device)
            if not all(_same(a, b) for a, b in zip(got, want)):
                raise RuntimeError(f"bundle-v9 PRENORM_PAIR: attention parts {index} differ from stock's prenorm at {rows} rows")


def prepare(model) -> None:
    from sglang.kernels.ops.layernorm import mhc as ops

    if ops._HC_MIX_COMPENSATED_SLICES != _SPLITS:
        raise RuntimeError(f"bundle-v9 PRENORM_PAIR: stock projects {ops._HC_MIX_COMPENSATED_SLICES} slices, not {_SPLITS}")
    _ATTN_BANDS.clear()  # king-v7: the king's proof below always runs the king's _CTAS
    parts, device = [], None
    for index, layer in enumerate(model.layers):
        hc = layer.attn_hc
        if hc is None or hc.tf32_parts is None:
            raise RuntimeError(f"bundle-v9 PRENORM_PAIR: layer {index} has no attention tf32 parts")
        high, low = hc.tf32_parts
        for t in (high, low):
            if tuple(t.shape) != (_MIX, _K) or t.dtype != torch.float32 or not t.is_contiguous():
                raise RuntimeError(f"bundle-v9 PRENORM_PAIR: layer {index} parts are {tuple(t.shape)} {t.dtype}")
        _QN.add(layer.self_attn.q_norm.weight.data_ptr())
        _ATTN.add(hc.fn.data_ptr())
        parts.append((high, low))
        device = high.device
    _prove(parts, device)
    print(f"bundle-v9 PRENORM_PAIR: attention prenorm on {len(parts)} layers", flush=True)
    _prepare_attn_width(parts, device)  # king-v7 ATTN-PAIR-WIDTH; never raises, the king's 16 CTAs on any failure
    _prepare_ffn(model)
    from bundle_v9.dec import mhc_stats

    mhc_stats.prepare_rsk4(model)  # king-v6 RSK-4ROW (review R3), FFN-PAIR's margin; never raises


def _ffn_mark(on: bool, why: str) -> None:
    _FFN_STATE["on"] = bool(on)
    _FFN_STATE["why"] = why


def ffn_line() -> str:
    return f"bundle-v9 king-v6 FFN-PAIR: {'ON' if _FFN_STATE['on'] else 'off'} ({_FFN_STATE['why']})"


def _check_bands(bands, limit: int = _FFN_MAX_ROWS, allowed=(1, 2), name: str = "FFN_PAIR_BANDS") -> list:
    out, last_row = [], _MIN_ROWS - 1
    for band in bands:
        first, last, ctas = band
        if not (type(first) is int and type(last) is int and type(ctas) is int and last_row < first <= last
                and last <= limit and ctas in allowed):
            raise RuntimeError(f"{name} entry {band!r} is not an ordered, disjoint band inside "
                               f"{_MIN_ROWS}..{limit} with ctas in {allowed}")
        out.append((first, last, ctas))
        last_row = last
    if not out:
        raise RuntimeError(f"{name} is empty")
    return out


def _band_rows(first: int, last: int) -> list:
    rows = {first, min(first + 1, last), max(last - 1, first), last, (first + last) // 2}
    for edge in (64, 128, 192, 256, 320):
        rows.update(r for r in (edge - 1, edge, edge + 1) if first <= r <= last)
    rows.update(r for r in (36, 48, 96, 144, 160, 176, 240, 264, 288) if first <= r <= last)
    return sorted(rows)


def _ffn_input(rows: int, family: int, gen, device):
    x = torch.randn((rows, _K), device=device, generator=gen, dtype=torch.float32) * 0.5
    if family == 1:
        x = x * torch.pow(10.0, torch.rand((rows, 1), device=device, generator=gen, dtype=torch.float32) * 6 - 3)
        x[rows // 2].zero_()
        x[rows // 3].fill_(-0.0)
    return x.to(torch.bfloat16)


def _wait(device, rows: int, ctas: int, label: str = "king-v6 FFN-PAIR") -> None:
    """torch.cuda.synchronize(device) plus review R5's watchdog line; it still waits (and raises) as synchronize."""
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    start = time.perf_counter()
    while not event.query():
        if not _FFN_HUNG[0] and time.perf_counter() - start > _FFN_WATCHDOG_S:
            _FFN_HUNG[0] = True
            print(f"bundle-v9 {label}: WATCHDOG, the proof launch at {rows} rows ctas {ctas} is not done "
                  f"after {_FFN_WATCHDOG_S:.0f} s (a prenorm_pair_kernel hang is suspected); still waiting",
                  file=sys.stderr, flush=True)
        time.sleep(0.0002)
    torch.cuda.synchronize(device)


def _prove_ffn(parts, device, bands, label: str = "king-v6 FFN-PAIR") -> int:
    gen = torch.Generator(device=device).manual_seed(20261106)
    cases = 0
    print(f"bundle-v9 {label}: proof launches start (ctas {sorted({c for _, _, c in bands})}; the first eager "
          f"run of this pair launch form: a boot that stops here hung in prenorm_pair_kernel)", file=sys.stderr,
          flush=True)
    for index, (high, low) in enumerate(parts):
        for first, last, ctas in bands:
            rows_list = _band_rows(first, last) if index == 0 else sorted({first, (first + last) // 2, last})
            for rows in rows_list:
                for family in ((0, 1) if index == 0 else (0,)):
                    x = _ffn_input(rows, family, gen, device)
                    want = _stock(x, high, low)
                    for _ in range(2 if index == 0 else 1):
                        got = (torch.full_like(want[0], float("nan")), torch.full_like(want[1], float("nan")),
                               torch.full_like(want[2], float("nan")))
                        _run_ffn_kernel(x, high, low, *got, ctas)
                        _wait(device, rows, ctas, label)
                        if not all(_same(a, b) for a, b in zip(got, want)):
                            raise RuntimeError(f"FFN parts {index} differ from stock's prenorm at {rows} rows, ctas {ctas}")
                        cases += 1
    return cases


def _ffn_parts(model):
    """v6t's FFN-side checks: (stats stream, [(high, low)] per layer, fn pointers, device, [(layer, ffn_hc)])."""
    stats = getattr(model, "hc_stats_stream", None)
    if stats is None:
        raise RuntimeError("the model has no mHC stats stream")
    parts, ptrs, device, hcs = [], set(), None, []
    for index, layer in enumerate(model.layers):
        hc = layer.ffn_hc
        if hc is None or hc.tf32_parts is None:
            raise RuntimeError(f"layer {index} has no FFN tf32 parts")
        high, low = hc.tf32_parts
        for t in (high, low):
            if tuple(t.shape) != (_MIX, _K) or t.dtype != torch.float32 or not t.is_contiguous():
                raise RuntimeError(f"layer {index} FFN parts are {tuple(t.shape)} {t.dtype}")
        if hc.fn.data_ptr() in _ATTN:
            raise RuntimeError(f"layer {index} FFN fn shares an attention fn pointer")
        ptrs.add(hc.fn.data_ptr())
        parts.append((high, low))
        hcs.append((index, hc))
        device = high.device
    if not parts:
        raise RuntimeError("the model has no layers")
    return stats, parts, ptrs, device, hcs


def _prepare_ffn(model) -> None:
    _FFN.clear()
    _FFN_BANDS.clear()
    _STATS[0] = None
    _X_CLUSTER[0] = False
    _TAIL.clear()
    _TAIL_KEYS.clear()
    if not FFN_PAIR_ON:
        _ffn_mark(False, "switched off by prenorm_pair.FFN_PAIR_ON; FFN prenorm on stock's two DeepGEMM launches")
        print(ffn_line(), file=sys.stderr, flush=True)
        _w_mark(False, "FFN-PAIR is switched off (prenorm_pair.FFN_PAIR_ON)")
        print(w_line(), file=sys.stderr, flush=True)
        _tail_mark(False, "FFN-PAIR is switched off (prenorm_pair.FFN_PAIR_ON); the king's reduce_sinkhorn launch")
        print(tail_line(), file=sys.stderr, flush=True)
        return
    try:
        stats, parts, ptrs, device, hcs = _ffn_parts(model)
    except Exception as error:  # fail-closed: the FFN side stays on stock's two DeepGEMM launches
        _ffn_mark(False, f"FAIL, FFN prenorm on stock: {type(error).__name__}: {str(error)[:300]}")
        print(ffn_line(), file=sys.stderr, flush=True)
        _w_mark(False, "FFN-PAIR failed its checks; FFN prenorm on stock")
        print(w_line(), file=sys.stderr, flush=True)
        _tail_mark(False, "FFN-PAIR failed its checks; the king's reduce_sinkhorn launch")
        print(tail_line(), file=sys.stderr, flush=True)
        return
    bands, cases = None, 0
    if FFN_PAIR_W_ON:
        try:
            wide = _check_bands(FFN_PAIR_W_BANDS, _MAX_ROWS, (1, 2), "FFN_PAIR_W_BANDS")
            probe = "no 2-CTA band"
            if any(c == 2 for _, _, c in wide):
                probe = _tpc_probe(device)
                _X_CLUSTER[0] = True
            cases = _prove_ffn(parts, device, wide, "king-v7 FFN-PAIR-W")
            bands = wide
            shown = ", ".join(f"{a}..{b} rows ctas {c}{' (one 2-CTA cluster)' if c == 2 else ''}" for a, b, c in wide)
            _w_mark(True, f"FFN-side mHC prenorm pair at {shown} in place of FFN_PAIR_BANDS; {probe}; bytes proven "
                          f"against stock's two DeepGEMM launches on {cases} cases over {len(parts)} layers' FFN parts")
        except Exception as error:  # fail-closed: v6t's FFN_PAIR_BANDS below
            _X_CLUSTER[0] = False
            _w_mark(False, f"FAIL, v6t's FFN_PAIR_BANDS serve: {type(error).__name__}: {str(error)[:300]}")
    else:
        _w_mark(False, "switched off by prenorm_pair.FFN_PAIR_W_ON; v6t's FFN_PAIR_BANDS serve")
    print(w_line(), file=sys.stderr, flush=True)
    if bands is None:
        try:
            bands = _check_bands(FFN_PAIR_BANDS)
            cases = _prove_ffn(parts, device, bands)
        except Exception as error:  # fail-closed: the FFN side stays on stock's two DeepGEMM launches
            _ffn_mark(False, f"FAIL, FFN prenorm on stock: {type(error).__name__}: {str(error)[:300]}")
            print(ffn_line(), file=sys.stderr, flush=True)
            _tail_mark(False, "FFN-PAIR failed its proof; the king's reduce_sinkhorn launch")
            print(tail_line(), file=sys.stderr, flush=True)
            return
    _STATS[0] = stats
    _FFN_BANDS.extend(bands)
    _FFN.update(ptrs)
    shown = ", ".join(f"{a}..{b} rows ctas {c}{' (2-CTA cluster)' if c == 2 else ''}" for a, b, c in bands)
    _ffn_mark(True, f"FFN-side mHC prenorm pair on the stats stream at {shown}, stock above {bands[-1][1]} rows; bytes "
                    f"proven against stock's two DeepGEMM launches on {cases} cases over {len(parts)} layers' FFN parts")
    print(ffn_line(), file=sys.stderr, flush=True)
    _prepare_tail(hcs, parts, device, bands)


def _w_mark(on: bool, why: str) -> None:
    _W_STATE["on"] = bool(on)
    _W_STATE["why"] = why


def w_line() -> str:
    return f"bundle-v9 king-v7 FFN-PAIR-W: {'ON' if _W_STATE['on'] else 'off'} ({_W_STATE['why']})"


def _tpc_probe(device) -> str:
    """The x module's 2-CTA cluster form may serve beside MegaMoE only if (a) every 2-CTA cluster runs on one TPC
    (sibling SMs 2t, 2t+1) and (b) the device holds SMs / 2 such clusters at once, i.e. MegaMoE's 73 two-CTA clusters
    plus this one (useful_ideas MegaMoE note: a side kernel on two TPCs, or on a TPC MegaMoE needs, blocks its 73rd
    cluster). 160 KB of dynamic shared memory a CTA keeps one CTA an SM, as the pair's 168..200 KB do. The probe CTAs
    spin on a counter for at most 5 ms (no hang). Raises on any other placement."""
    mod = _x_module()
    n_sm = int(torch.cuda.get_device_properties(device).multi_processor_count)
    full = n_sm // 2
    seen = 0
    for clusters, reps, together in ((1, 32, True), (full // 2, 4, True), (full, 4, True), (2 * full, 1, False)):
        for _ in range(reps):
            out = torch.full((clusters * 8,), -1, dtype=torch.int32, device=device)
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            mod.probe(out, counter, 160 * 1024, 5_000_000)
            torch.cuda.synchronize(device)
            for k, ((s0, r0, n0, c0), (s1, r1, n1, c1)) in enumerate(out.view(-1, 2, 4).tolist()):
                if not (n0 == 2 and n1 == 2 and r0 == 0 and r1 == 1 and s0 >= 0 and s1 >= 0 and s0 != s1
                        and s0 // 2 == s1 // 2):
                    raise RuntimeError(f"TPC probe: cluster {k} of a {clusters}-cluster launch ran on SMs {s0} / {s1} "
                                       f"(ranks {r0} / {r1}, cluster sizes {n0} / {n1}), not on one TPC")
                if together and not (c0 == 1 and c1 == 1):
                    raise RuntimeError(f"TPC probe: {clusters} two-CTA clusters of 160 KB were not resident at once "
                                       f"on {n_sm} SMs (cluster {k} timed out)")
                seen += 1
    return f"TPC probe: {seen} 2-CTA clusters each on one TPC, {full} resident at once on {n_sm} SMs"


def _tail_mark(on: bool, why: str) -> None:
    _TAIL_STATE["on"] = bool(on)
    _TAIL_STATE["why"] = why


def tail_line() -> str:
    return f"bundle-v9 king-v7 STATS-TAIL: {'ON' if _TAIL_STATE['on'] else 'off'} ({_TAIL_STATE['why']})"


def _prove_tail(hcs, parts, device, bands) -> int:
    """The x module's kTail launch at every served CTA count against stock's two DeepGEMM launches (hi, lo, square
    sums) and the king's kRows-1 reduce_sinkhorn on those partials (pre, post, comb), NaN-filled, int32 compare. Layer
    0: every band row of _band_rows in two input families; every other layer (its own parts and FFN constants): the band
    edges and one middle row."""
    from bundle_v9.dec import mhc_stats

    gen = torch.Generator(device=device).manual_seed(20261106 + 43)
    red, xmod = mhc_stats._module(), _x_module()
    cases = 0
    print(f"bundle-v9 king-v7 STATS-TAIL: proof launches start (ctas {sorted({c for _, _, c in bands})}; the first run "
          f"of the fused tail: a boot that stops here hung in dec_prenorm_pair_x)", file=sys.stderr, flush=True)
    for index, hc in hcs:
        high, low = parts[index]
        cfg = hc.cfg
        args = (1.0 / _K, float(cfg.rms_eps), int(cfg.sinkhorn_iters), float(cfg.eps))
        for first, last, ctas in bands:
            rows_list = _band_rows(first, last) if index == 0 else sorted({first, (first + last) // 2, last})
            for rows in rows_list:
                for family in ((0, 1) if index == 0 else (0,)):
                    x = _ffn_input(rows, family, gen, device)
                    want = list(_stock(x, high, low))
                    want += [torch.full((rows, 4), float("nan"), device=device),
                             torch.full((rows, 4), float("nan"), device=device),
                             torch.full((rows, 4, 4), float("nan"), device=device)]
                    red.run_deepgemm(want[0], want[1], want[2], hc.scale, hc.base, want[3], want[4], want[5], *args)
                    got = [torch.full_like(t, float("nan")) for t in want]
                    xmod.run_tail(x, high, low, got[0], got[1], got[2], hc.scale, hc.base, got[3], got[4], got[5],
                                  *args, ctas)
                    _wait(device, rows, ctas, "king-v7 STATS-TAIL")
                    for name, a, b in zip(("hi", "lo", "sq", "pre", "post", "comb"), got, want):
                        if not _same(a, b):
                            raise RuntimeError(f"layer {index} {name} differs from stock + the king's reduce at {rows} "
                                               f"rows, ctas {ctas}")
                    cases += 1
    return cases


def _prepare_tail(hcs, parts, device, bands) -> None:
    """king-v7 STATS-TAIL prepare (after FFN-PAIR serves). Never raises: FFN-PAIR + the reduce serve on any failure."""
    _TAIL.clear()
    _TAIL_KEYS.clear()
    if not STATS_TAIL_ON:
        _tail_mark(False, "switched off by prenorm_pair.STATS_TAIL_ON; FFN-PAIR + the reduce_sinkhorn launch")
        print(tail_line(), file=sys.stderr, flush=True)
        return
    try:
        keys = set()
        for index, hc in hcs:
            cfg = hc.cfg
            if hc.scale.dtype != torch.float32 or hc.base.dtype != torch.float32 or not hc.scale.is_contiguous() or \
                    not hc.base.is_contiguous() or hc.scale.numel() != 3 or hc.base.numel() != _MIX or \
                    hc.scale.data_ptr() % 16 or hc.base.data_ptr() % 16 or int(cfg.sinkhorn_iters) < 1:
                raise RuntimeError(f"layer {index} FFN constants are not float32 [3] / [24], 16-byte aligned")
            keys.add((int(cfg.sinkhorn_iters), float(cfg.rms_eps), float(cfg.eps)))
        cases = _prove_tail(hcs, parts, device, bands)
    except Exception as error:  # fail-closed: FFN-PAIR + the reduce
        _tail_mark(False, f"FAIL, FFN-PAIR + the reduce_sinkhorn launch: {type(error).__name__}: {str(error)[:300]}")
        print(tail_line(), file=sys.stderr, flush=True)
        return
    _TAIL_KEYS.update(keys)
    _TAIL.update(c for _, _, c in bands)
    _tail_mark(True, f"FFN-side reduce + Sinkhorn in the pair kernel's tail at every FFN-PAIR band (ctas "
                     f"{sorted(_TAIL)}), no separate reduce_sinkhorn launch there; all six outputs proven against stock's "
                     f"two DeepGEMM launches + the king's kRows-1 reduce on {cases} cases over {len(hcs)} layers' FFN "
                     f"parts and constants")
    print(tail_line(), file=sys.stderr, flush=True)


def _aw_mark(on: bool, why: str) -> None:
    _AW_STATE["on"] = bool(on)
    _AW_STATE["why"] = why


def aw_line() -> str:
    return f"bundle-v9 king-v7 ATTN-PAIR-WIDTH: {'ON' if _AW_STATE['on'] else 'off'} ({_AW_STATE['why']})"


def _prove_attn_width(parts, device, bands) -> int:
    gen = torch.Generator(device=device).manual_seed(20261106 + 32)
    cases = 0
    print(f"bundle-v9 king-v7 ATTN-PAIR-WIDTH: proof launches start (ctas {sorted({c for _, _, c in bands})}; "
          f"multi-split attention-side launches: a boot that stops here hung in prenorm_pair_kernel)", file=sys.stderr,
          flush=True)
    for index, (high, low) in enumerate(parts):
        for first, last, ctas in bands:
            if ctas == _CTAS:
                continue  # the king's own form, proven by _prove
            rows_list = _band_rows(first, last) if index == 0 else sorted({first, (first + last) // 2, last})
            for rows in rows_list:
                for family in ((0, 1) if index == 0 else (0,)):
                    x = _ffn_input(rows, family, gen, device)
                    want = _stock(x, high, low)
                    got = (torch.full_like(want[0], float("nan")), torch.full_like(want[1], float("nan")),
                           torch.full_like(want[2], float("nan")))
                    _module().run(x, high, low, *got, ctas)
                    _wait(device, rows, ctas, "king-v7 ATTN-PAIR-WIDTH")
                    if not all(_same(a, b) for a, b in zip(got, want)):
                        raise RuntimeError(f"attention parts {index} differ from stock's prenorm at {rows} rows, "
                                           f"ctas {ctas}")
                    cases += 1
    return cases


def _prepare_attn_width(parts, device) -> None:
    """king-v7 ATTN-PAIR-WIDTH prepare (after the king's attention proof). Never raises: the king's 16 CTAs serve."""
    _ATTN_BANDS.clear()
    if not ATTN_WIDTH_ON:
        _aw_mark(False, f"switched off by prenorm_pair.ATTN_WIDTH_ON; the king's {_CTAS}-CTA attention pair at "
                        f"{_MIN_ROWS}..{_MAX_ROWS} rows")
        print(aw_line(), file=sys.stderr, flush=True)
        return
    try:
        bands = _check_bands(ATTN_PAIR_BANDS, _MAX_ROWS, (1, 2, 4, 8, 16), "ATTN_PAIR_BANDS")
        cases = _prove_attn_width(parts, device, bands)
    except Exception as error:  # fail-closed: the king's 16 CTAs
        _aw_mark(False, f"FAIL, the king's {_CTAS}-CTA attention pair: {type(error).__name__}: {str(error)[:300]}")
        print(aw_line(), file=sys.stderr, flush=True)
        return
    _ATTN_BANDS.extend(bands)
    shown = ", ".join(f"{a}..{b} rows ctas {c}" for a, b, c in bands)
    _aw_mark(True, f"attention-side prenorm pair at {shown} (the king's {_CTAS} CTAs elsewhere); bytes proven against "
                   f"stock on {cases} cases over {len(parts)} layers' attention parts")
    print(aw_line(), file=sys.stderr, flush=True)
