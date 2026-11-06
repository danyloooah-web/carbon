"""king-v6 HOIST-PDL (IDEAS_dec_rest M4, IDEAS_dec_mhc M8 / D2; RANKED #16) and SEAM-POST (IDEAS_dec_mhc M7; RANKED #21).

HOIST-T. The king's 320-thread decode boundary kernel (cuda_mhc_prefetch.py, modes 0 / 1 / 2 at 149..288 rows) issues
`griddepcontrol.launch_dependents` only in mode 2, as its last instruction, which is the implicit trigger at CTA exit
(VERIFY_dec_mhc M8). Here every mode triggers early: after its norm-weight and coefficient loads, before its x / R loads
and its first store (one CTA a row, all CTAs resident in one wave at <= 288 rows), so a PDL-launched successor (CuTe
wqkv_a after mode 1, DeepGEMM wqkv_a after mode 2) lands its CTAs and runs its prologue while HOIST works. Mode 2's end trigger is removed (one trigger per mode). Scheduling only: a programmatic
dependent's griddepcontrol.wait waits for this grid's completion and memory flush before it reads (the king's W4 and the
CuTe GEMMs already trigger before their stores).

HOIST-PDL-IN (switch off by default). The three HOIST launches take the programmatic-serialization attribute and the
kernel waits right after loading the next sublayer's norm weight (a model parameter), before reading x, R, post, comb or
pre; those three coefficient loads become plain coherent loads. HOIST follows the stats-stream join, and a PDL launch
after a join makes every in-edge programmatic (UI PDLX), which is safe here only because nothing but the weight is read
before the wait. Off by default because its value is small (D2: central 20 us a step) and the trigger half is the part
the ranked plan builds first; flip HOIST_PDL_IN_ON to arm it.

Both edits are anchored text edits on the king's HOIST text, applied after king-v2e's R-L2 edits (mhc_prefetch composes
them); every anchor must match exactly once. The edited module is proven by the king's own HOIST proof (mhc_prefetch._prove:
15 row counts, both quant modes against stock, the DG mode against the king's packed kernel) before it serves.

SEAM-POST. At the three closed seams of a step (the FFN posts before the Engram layers L1 and L14, and the final post)
`_post_fusion` has no next boundary and calls stock's `post()`, which runs the Triton `_mhc_post_split_h_kernel` (grid
rows x 5, ~12.6 us a call). This kernel is HOIST's `updated` computation alone, one 320-thread CTA a row: the same
fma(post, x, comb0 * r0), then fma(comb_j, r_j, acc) in j order, RNE bf16 store, which is split_h's sequence with fp
fusion off (VERIFY_dec_mhc M7). Served only where stock's post() itself would pick split_h (Blackwell, pre_from_prev,
mult 4, hidden 5120, contiguous bf16 / fp32 operands) and only at SEAM_POST_ROWS (HOIST's one-CTA-a-row band). prepare
proves it against stock's mhc_post_split_h at every row count of that band, output pre-filled with NaN.
"""
from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache

import torch

# The lead's switches. True: the HOIST text carries the edit / the seam posts run the kernel below. False: the king's.
HOIST_T_ON = True
HOIST_PDL_IN_ON = False
SEAM_POST_ON = True
SEAM_POST_ROWS = (149, 288)
SEAM_POST_THREADS = 320
SEAM_SEED = 20261107

# HOIST-T: one trigger a mode, after the norm-weight and coefficient loads and before the x / R block, none at the end.
# (Placed after the x / R loads, the trigger splits ptxas's schedule and the kernel goes from the king's 64 to 96
# registers, local nvcc 13.0 sm_103a; here it keeps 64, so HOIST's residency and its successor's co-residency are the
# king's.)
_T_EDITS = (
    ("\n  V xh[kVPT], rh[kVPT][4];\n",
     "\n  // king-v6 HOIST-T: release the PDL successor (its wait still covers this grid's completion) before the x / R\n"
     "  // loads and every store.\n"
     '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
     "  V xh[kVPT], rh[kVPT][4];\n"),
    ('  if constexpr (Mode == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n', ""),
)
# HOIST-PDL-IN: wait after the norm-weight loads; coherent coefficient loads; programmatic launches.
_PDL_IN_EDITS = (
    ("  for (int j = 0; j < kVPT; ++j) wv[j].load(p.weight, tid + j * Threads);\n",
     "  for (int j = 0; j < kVPT; ++j) wv[j].load(p.weight, tid + j * Threads);\n"
     "  // king-v6 HOIST-PDL-IN: only the norm weight (a parameter) is read before the wait.\n"
     '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'),
    ("    comb[i] = __ldg(p.comb + row * 16 + i);\n", "    comb[i] = p.comb[row * 16 + i];\n"),
    ("    post[i] = __ldg(p.post + row * 4 + i);\n    pre[i] = __ldg(p.pre + row * 4 + i);\n",
     "    post[i] = p.post[row * 4 + i];\n    pre[i] = p.pre[row * 4 + i];\n"),
    ("LaunchKernel(dim3(rows), Threads, dev.unwrap()).launch(mhc_prefetch_kernel<Threads, 1>, params);",
     "LaunchKernel(dim3(rows), Threads, dev.unwrap()).enable_pdl(true).launch(mhc_prefetch_kernel<Threads, 1>, params);"),
    ("LaunchKernel(dim3(rows), Threads, dev.unwrap()).launch(mhc_prefetch_kernel<Threads, 0>, params);",
     "LaunchKernel(dim3(rows), Threads, dev.unwrap()).enable_pdl(true).launch(mhc_prefetch_kernel<Threads, 0>, params);"),
    ("LaunchKernel(dim3(rows), Threads, dev.unwrap()).launch(mhc_prefetch_kernel<Threads, 2>, params);",
     "LaunchKernel(dim3(rows), Threads, dev.unwrap()).enable_pdl(true).launch(mhc_prefetch_kernel<Threads, 2>, params);"),
)

_STATE: dict = {name: {"on": False, "why": "not prepared"} for name in ("HOIST-T", "HOIST-PDL-IN", "SEAM-POST")}
_SEAM: list = [False]
_WARNED = [False]


def mark(name: str, on: bool, why: str) -> None:
    _STATE[name]["on"] = bool(on)
    _STATE[name]["why"] = why


def line(name: str) -> str:
    state = _STATE[name]
    return f"bundle-v9 king-v6 {name}: {'ON' if state['on'] else 'off'} ({state['why']})"


def say(name: str) -> None:
    print(line(name), file=sys.stderr, flush=True)


def _apply(text: str, edits) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"a HOIST-PDL anchor matches {count} times in the HOIST source, not once: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def hoist_edits() -> tuple:
    """(names, edits) of the switched-on HOIST text levers, in application order."""
    names, edits = [], []
    if HOIST_T_ON:
        names.append("HOIST-T")
        edits.extend(_T_EDITS)
    if HOIST_PDL_IN_ON:
        names.append("HOIST-PDL-IN")
        edits.extend(_PDL_IN_EDITS)
    return tuple(names), tuple(edits)


def hoist_source(text: str) -> str:
    """The HOIST text with the switched-on edits applied. Raises unless every anchor matches once."""
    return _apply(text, hoist_edits()[1])


def mark_hoist(on: bool, why: str) -> None:
    """States of the HOIST text levers (mhc_prefetch sets them once the composed text served or failed)."""
    for name, switch in (("HOIST-T", HOIST_T_ON), ("HOIST-PDL-IN", HOIST_PDL_IN_ON)):
        if not switch:
            mark(name, False, f"switched off by hoist_pdl.{name.replace('-', '_')}_ON; the king's HOIST "
                              f"{'trigger (mode 2 only, at its end)' if name == 'HOIST-T' else 'launch (no PDL attribute)'}")
        else:
            mark(name, on, why)


def say_hoist() -> None:
    say("HOIST-T")
    say("HOIST-PDL-IN")


# ---------------------------------------------------------------- SEAM-POST

SEAM_NAME = "mhc_seam_post.cuh"
SEAM_SOURCE = r"""#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace seam_post {

// king-v6 SEAM-POST: HOIST's R' computation (cuda_mhc_prefetch.py, the `updated` loop) alone, one CTA a row.
// R'[i] = fma(comb[12+i], r3, fma(comb[8+i], r2, fma(comb[4+i], r1, fma(post[i], x, comb[i] * r0)))), RNE bf16:
// the sequence of stock's _mhc_post_split_h_kernel (enable_fp_fusion=False).
template <int Threads>
__global__ __launch_bounds__(Threads) void mhc_seam_post_kernel(const bf16_t* x_in, const bf16_t* r_in,
                                                                const float* post_in, const float* comb_in,
                                                                bf16_t* r_out) {
  using namespace device;
  using V = AlignedVector<bf16x2_t, 4>;
  constexpr int kVectors = 640;
  constexpr int kVPT = kVectors / Threads;
  static_assert(kVPT * Threads == kVectors && Threads % 32 == 0);

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;

  float comb[16], post[4];
#pragma unroll
  for (int i = 0; i < 16; ++i) comb[i] = comb_in[row * 16 + i];
#pragma unroll
  for (int i = 0; i < 4; ++i) post[i] = post_in[row * 4 + i];

  V xh[kVPT], rh[kVPT][4];
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    xh[j].load(x_in + row * 5120, tid + j * Threads);
#pragma unroll
    for (int r = 0; r < 4; ++r) rh[j][r].load(r_in + row * 20480 + r * 5120, tid + j * Threads);
  }

#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = tid + j * Threads;
    V x = xh[j], residual[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) residual[r] = rh[j][r];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      V updated;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto xx = cast<fp32x2_t>(x[k]);
        const auto r0 = cast<fp32x2_t>(residual[0][k]);
        const auto r1 = cast<fp32x2_t>(residual[1][k]);
        const auto r2 = cast<fp32x2_t>(residual[2][k]);
        const auto r3 = cast<fp32x2_t>(residual[3][k]);
        float a = __fmaf_rn(post[i], xx.x, __fmul_rn(comb[i], r0.x));
        float b = __fmaf_rn(post[i], xx.y, __fmul_rn(comb[i], r0.y));
        a = __fmaf_rn(comb[4 + i], r1.x, a);
        b = __fmaf_rn(comb[4 + i], r1.y, b);
        a = __fmaf_rn(comb[8 + i], r2.x, a);
        b = __fmaf_rn(comb[8 + i], r2.y, b);
        a = __fmaf_rn(comb[12 + i], r3.x, a);
        b = __fmaf_rn(comb[12 + i], r3.y, b);
        updated[k] = cast<bf16x2_t>(fp32x2_t{a, b});
      }
      updated.store(r_out + row * 20480 + i * 5120, vid);
    }
  }
}

template <int Threads>
struct MhcSeamPost {
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView residual, tvm::ffi::TensorView post,
                  tvm::ffi::TensorView comb, tvm::ffi::TensorView out) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({m, 4, 5120}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(out);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(post);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    for (const auto* ptr : {x.data_ptr(), residual.data_ptr(), out.data_ptr()}) {
      CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % 16 == 0) << "BF16 pointers must be 16-byte aligned";
    }
    const int64_t rows = m.unwrap();
    if (rows == 0) return;
    LaunchKernel(dim3(rows), Threads, dev.unwrap())
        .launch(mhc_seam_post_kernel<Threads>, static_cast<const bf16_t*>(x.data_ptr()),
                static_cast<const bf16_t*>(residual.data_ptr()), static_cast<const float*>(post.data_ptr()),
                static_cast<const float*>(comb.data_ptr()), static_cast<bf16_t*>(out.data_ptr()));
  }
};

}  // namespace seam_post
}  // namespace sglang
"""


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
def _seam_module():
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data = SEAM_SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-mhc-seam-post", digest)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, SEAM_NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return load_jit("dec_mhc_seam_post", str(SEAM_POST_THREADS), digest, cuda_files=[path],
                    cuda_wrappers=[("run", f"seam_post::MhcSeamPost<{SEAM_POST_THREADS}>::run")])


def _seam_domain(cfg, y, residual, post_mix, comb) -> bool:
    """Where stock's post() picks mhc_post_split_h, intersected with the served band."""
    from sglang.srt.runtime_context import get_platform

    rows = y.shape[0]
    return (SEAM_POST_ROWS[0] <= rows <= SEAM_POST_ROWS[1] and get_platform().is_blackwell and cfg.pre_from_prev
            and cfg.mult == 4 and cfg.hidden == 5120 and y.dim() == 2 and tuple(y.shape) == (rows, 5120)
            and y.dtype == residual.dtype == torch.bfloat16 and tuple(residual.shape) == (rows, 4, 5120)
            and post_mix.dtype == comb.dtype == torch.float32 and tuple(post_mix.shape) == (rows, 4)
            and tuple(comb.shape) == (rows, 4, 4)
            and all(t.is_contiguous() and t.is_cuda and t.device == y.device for t in (y, residual, post_mix, comb))
            and y.data_ptr() % 16 == 0 and residual.data_ptr() % 16 == 0)


def seam_post(cfg, y, residual, post_mix, comb, decode_like):
    """R' for a seam post from the SEAM-POST kernel, or None: call stock's post(). `decode_like` is mhc_floor's gate
    (multi-stream overlap on, no extend in the batch, not a prefill graph), the one HOIST's own path uses."""
    if not _SEAM[0]:
        return None
    try:
        if not (_seam_domain(cfg, y, residual, post_mix, comb) and decode_like()):
            return None
    except Exception:
        return None
    try:
        out = torch.empty_like(residual)
        _seam_module().run(y, residual, post_mix, comb, out)
    except Exception as error:
        _SEAM[0] = False
        mark("SEAM-POST", False, f"disabled at run time, stock's mhc_post_split_h from here on: {type(error).__name__}: "
                                 f"{str(error)[:300]}")
        if not _WARNED[0]:
            _WARNED[0] = True
            say("SEAM-POST")
        return None
    return out


def _seam_case(rows: int, family: int, gen, device):
    y = torch.randn((rows, 5120), device=device, generator=gen) * 2
    residual = torch.randn((rows, 4, 5120), device=device, generator=gen)
    post = torch.rand((rows, 4), device=device, generator=gen) * 2
    comb = torch.rand((rows, 4, 4), device=device, generator=gen)
    if family == 1:
        y = y * torch.pow(10.0, torch.rand((rows, 1), device=device, generator=gen) * 12 - 6)
        residual = residual * torch.pow(10.0, torch.rand((rows, 4, 1), device=device, generator=gen) * 12 - 6)
        comb = comb * 2 - 1
        y[rows // 2].zero_()
        residual[rows // 3, 1].fill_(-0.0)
        post[rows // 4].zero_()
    return y.bfloat16(), residual.bfloat16(), post.contiguous(), comb.contiguous()


def prepare_seam(device) -> None:
    """Prove SEAM-POST against stock's mhc_post_split_h at every row count of its band. Never raises."""
    _SEAM[0] = False
    if not SEAM_POST_ON:
        mark("SEAM-POST", False, "switched off by hoist_pdl.SEAM_POST_ON; stock's mhc_post_split_h")
        say("SEAM-POST")
        return
    try:
        from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

        first, last = SEAM_POST_ROWS
        if not (type(first) is int and type(last) is int and 1 <= first <= last <= 384):
            raise RuntimeError(f"SEAM_POST_ROWS {SEAM_POST_ROWS!r} is not a band inside stock's split_h range 1..384")
        module = _seam_module()
        gen = torch.Generator(device=device).manual_seed(SEAM_SEED)
        cases = 0
        for rows in range(first, last + 1):
            for family in ((0, 1) if rows in (first, last) or rows % 16 == 0 else (rows % 2,)):
                y, residual, post, comb = _seam_case(rows, family, gen, device)
                want = mhc_post_split_h(y, residual, post, comb)
                got = torch.full_like(residual, float("nan"))
                module.run(y, residual, post, comb, got)
                if want.shape != got.shape or not torch.equal(want.view(torch.int16), got.view(torch.int16)):
                    raise RuntimeError(f"the seam post at {rows} rows (family {family}) differs from stock's split_h")
                cases += 1
        torch.cuda.synchronize(device)
    except Exception as error:
        mark("SEAM-POST", False, f"FAIL, stock's mhc_post_split_h: {type(error).__name__}: {str(error)[:300]}")
        say("SEAM-POST")
        return
    _SEAM[0] = True
    mark("SEAM-POST", True, f"seam posts (no next boundary) at {first}..{last} rows on the one-CTA-a-row HOIST R' kernel "
                            f"in place of stock's mhc_post_split_h; bytes proven against it at every row count of the "
                            f"band ({cases} cases)")
    say("SEAM-POST")
