"""king-v6 glue PDL kernels: anchored copies of three king PFUSE kernels (cuda_prefill_fuse.SOURCE), built as one JIT module.

- QNORM-PDL (IDEAS_dec_rest DR3): `qnorm_mxfp8_kernel`, the q-norm + MXFP8 pair that king-v2e's QNORM-DEC serves on decode.
  The static norm weight is loaded before `griddepcontrol.wait`, x (the wqkv_a output) only after it, and the dependents
  (wq_b) are released right after the wait. Launched with the programmatic-serialization attribute; `kRows` is a parameter
  (rows are independent, one warp each, so it changes only the grid).
- WOBSF-PDL (DR4): `wob_scale_kernel`, the wo_b input-scale swizzle. Wait at entry, before its only global read; release
  right after it; launched with the attribute.
- IDX-W16 (DR9, glue half): `index_q_tok_kernel`, the king's replica of stock's indexer index_q, writing the head weights as
  their bf16 bits instead of the bf16 value widened to fp32 (the two hold the same value), so the consumers'
  `weights.to(torch.bfloat16)` copy kernels become no-ops. Wait right after the token-range exit (stock's own index_q
  waits too); release right after the wait.

In every copy each global read of data a predecessor may still be writing (x, the input scale, the indexer query, the head
weights, positions, the rope table) is issued after the wait as a plain coherent load (no `__ldg`, no `__restrict__`): with
PDL the producer can still be running while this grid is resident, and `ld.global.nc` is defined only for data that is
read-only for the whole lifetime of the kernel. Only model parameters (the q-norm weight) are read before the wait.
Arithmetic, stores, store order and addresses are the king's text. The namespace and host structs are renamed
(`pfuse_pdl`, `GluePdl`, `GluePdlIdx`) so no symbol of this module can interpose the king's `dec_prefill_fuse` module.
Every anchor must match exactly once, or `source()` raises (glue_pdl turns the three levers off).
"""
from __future__ import annotations

import re

from bundle_v9.dec import cuda_prefill_fuse

NAME = "glue_pdl.cuh"

_QNORM_START = ("template <int kRows, int kMinBlocks>\n"
                "__global__ __launch_bounds__(kQThreads, kMinBlocks) void qnorm_mxfp8_kernel(")
_WOB_START = "__global__ __launch_bounds__(256) void wob_scale_kernel("
_WOB_END = "__device__ __forceinline__ uint32_t plus_zero("
_HOST_QNORM = "  static void qnorm(tvm::ffi::TensorView x,"
_HOST_WOB = "  static void wob_scale(tvm::ffi::TensorView sf,"
_HOST_END = "  static void engram_place("
_IDX_START = "#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>\n"

# Every edit table below is a literal (no names, no expressions), one (old, new) pair per anchored edit.
_QNORM_EDITS = (
    ("const __nv_bfloat16* __restrict__ x, int64_t sx,", "const __nv_bfloat16* x, int64_t sx,"),
    ("  uint4 wv[kQChunks];\n#pragma unroll\n"
     "  for (int c = 0; c < kQChunks; ++c) wv[c] = __ldg(reinterpret_cast<const uint4*>(w + 256 * c + 8 * lane));\n",
     ""),
    ("  if (row0 >= m_pad) return;\n  uint4 xv[kRows][kQChunks];\n",
     "  if (row0 >= m_pad) return;\n"
     "  // king-v6 QNORM-PDL: the static norm weight before the wait; x only after it; wq_b released after the wait.\n"
     "  uint4 wv[kQChunks];\n#pragma unroll\n"
     "  for (int c = 0; c < kQChunks; ++c) wv[c] = __ldg(reinterpret_cast<const uint4*>(w + 256 * c + 8 * lane));\n"
     '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
     '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
     "  uint4 xv[kRows][kQChunks];\n"),
    ("xv[r][c] = __ldg(reinterpret_cast<const uint4*>(src + 256 * c));",
     "xv[r][c] = *reinterpret_cast<const uint4*>(src + 256 * c);"),
)

_QNORM_HOST_EDITS = (
    ("      LaunchKernel(dim3(blocks), pfuse::kQThreads, device)(\n",
     "      LaunchKernel(dim3(blocks), pfuse::kQThreads, device).enable_pdl(true)(\n"),
)
# The q-norm launcher line; `{rows}` is QNORM_PDL_ROWS (1 or 2).
_QNORM_LAUNCH = ("    launch(pfuse::qnorm_mxfp8_kernel<2, 2>, 2);\n",
                 "    launch(pfuse::qnorm_mxfp8_kernel<{rows}, 2>, {rows});\n")

_WOB_EDITS = (
    ("    const int32_t* __restrict__ in, int64_t s0, int64_t s1, int rows, int kw_count, int64_t units, "
     "int4* __restrict__ out) {\n",
     "    const int32_t* in, int64_t s0, int64_t s1, int rows, int kw_count, int64_t units, int4* __restrict__ out) {\n"
     "  // king-v6 WOBSF-PDL: wait before the first read of wo_a's scale words; wo_b released right after the wait.\n"
     '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
     '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'),
    ("v[j] = mrow < rows ? __ldg(in + mrow * s0 + kw * s1) : 0;", "v[j] = mrow < rows ? in[mrow * s0 + kw * s1] : 0;"),
)

_WOB_HOST_EDITS = (
    ("    LaunchKernel(dim3(grid), 256, dev.unwrap())(\n        pfuse::wob_scale_kernel,",
     "    LaunchKernel(dim3(grid), 256, dev.unwrap()).enable_pdl(true)(\n        pfuse::wob_scale_kernel,"),
)

_IDX_EDITS = (
    ("  const uint4* __restrict__ input;\n  const float* __restrict__ freqs;\n  const void* __restrict__ positions;\n",
     "  const uint4* input;\n  const float* freqs;\n  const void* positions;\n"),
    ("  const uint16_t* __restrict__ head_weights;\n  float* __restrict__ weights_out;\n",
     "  const uint16_t* head_weights;\n  uint16_t* __restrict__ weights_out;  // king-v6 IDX-W16: the bf16 bits\n"),
    ("  if (token >= num_tokens) return;\n  float fr[8];\n",
     "  if (token >= num_tokens) return;\n"
     "  // king-v6 IDX-W16: every read after the wait (stock's index_q is PDL-launched and waits too).\n"
     '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
     '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
     "  float fr[8];\n"),
    ("    const float4 f0 = __ldg(f), f1 = __ldg(f + 1);\n", "    const float4 f0 = f[0], f1 = f[1];\n"),
    ("xin[it] = __ldg(p.input + (row_base + 2 * it + sub) * 16 + hl);",
     "xin[it] = p.input[(row_base + 2 * it + sub) * 16 + hl];"),
    ("__ldg(p.head_weights + row)", "p.head_weights[row]"),
    ("      p.weights_out[row] = bf16_lo(cvt_bf16x2(wv, 0.0f));\n",
     "      p.weights_out[row] = static_cast<uint16_t>(cvt_bf16x2(wv, 0.0f) & 0xFFFFu);\n"),
    ("    TensorMatcher({N, H}).with_dtype<fp32_t>().with_device(device_).verify(weights_out);\n",
     "    TensorMatcher({N, H}).with_dtype<bf16_t>().with_device(device_).verify(weights_out);\n"),
    ("        .weights_out = static_cast<float*>(weights_out.data_ptr()),\n",
     "        .weights_out = static_cast<uint16_t*>(weights_out.data_ptr()),\n"),
    ("LaunchKernel(blocks, 256, device_.unwrap())(pfuse::index_q_tok_kernel<16, int32_t>, params, tokens);",
     "LaunchKernel(blocks, 256, device_.unwrap()).enable_pdl(true)(pfuse::index_q_tok_kernel<16, int32_t>, params, tokens);"),
    ("LaunchKernel(blocks, 256, device_.unwrap())(pfuse::index_q_tok_kernel<16, int64_t>, params, tokens);",
     "LaunchKernel(blocks, 256, device_.unwrap()).enable_pdl(true)(pfuse::index_q_tok_kernel<16, int64_t>, params, tokens);"),
)


def _once(text: str, needle: str, where: str) -> int:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"glue PDL: the {where} anchor {needle[:60]!r} matches {count} times, not once")
    return text.find(needle)


def _cut(text: str, start: str, end: str, where: str) -> str:
    i = _once(text, start, where)
    j = _once(text, end, where) if end else len(text)
    if j <= i:
        raise RuntimeError(f"glue PDL: the {where} segment is empty")
    return text[i:j]


def _apply(text: str, edits, where: str) -> str:
    for old, new in edits:
        _once(text, old, where)
        text = text.replace(old, new)
    return text


def source(*, qnorm: bool, wob: bool, idx: bool, qnorm_rows: int) -> str:
    """The module text holding the switched-on copies. Raises unless every anchor matches once."""
    if not (qnorm or wob or idx):
        raise RuntimeError("glue PDL: no lever is switched on")
    if qnorm_rows not in (1, 2):
        raise RuntimeError(f"glue PDL: q-norm rows a warp must be 1 or 2, not {qnorm_rows!r}")
    king = cuda_prefill_fuse.SOURCE
    prelude = king[:_once(king, _QNORM_START, "prelude")]
    if prelude.count("namespace pfuse {") != 1 or "struct PFuse" in prelude:
        raise RuntimeError("glue PDL: the PFUSE prelude is not the helper namespace alone")
    text = prelude
    if qnorm:
        text += _apply(_cut(king, _QNORM_START, _WOB_START, "q-norm kernel"), _QNORM_EDITS, "q-norm kernel")
    if wob:
        text += _apply(_cut(king, _WOB_START, _WOB_END, "wo_b swizzle kernel"), _WOB_EDITS, "wo_b swizzle kernel")
    text += "\n}  // namespace pfuse\n\n"
    if qnorm or wob:
        text += "struct GluePdl {\n\n"
        if qnorm:
            host = _cut(king, _HOST_QNORM, _HOST_WOB, "q-norm host")
            launch = (_QNORM_LAUNCH[0], _QNORM_LAUNCH[1].replace("{rows}", str(qnorm_rows)))
            host = _apply(host, _QNORM_HOST_EDITS + (launch,), "q-norm host")
            text += host
        if wob:
            text += _apply(_cut(king, _HOST_WOB, _HOST_END, "wo_b swizzle host"), _WOB_HOST_EDITS, "wo_b swizzle host")
        text += "};\n\n"
    if idx:
        text += _apply(_cut(king, _IDX_START, "", "index_q"), _IDX_EDITS, "index_q")
    count = text.count("namespace pfuse {")
    text = text.replace("namespace pfuse {", "namespace pfuse_pdl {").replace("pfuse::", "pfuse_pdl::")
    text = text.replace("}  // namespace pfuse\n", "}  // namespace pfuse_pdl\n")
    text = text.replace("struct PFuseIdx {", "struct GluePdlIdx {")
    if count != (2 if idx else 1) or "PFuse" in text or re.search(r"\bpfuse\b(?!_pdl)", text):
        raise RuntimeError("glue PDL: the renamed text still names the king's PFUSE symbols")
    return text


def wrappers(*, qnorm: bool, wob: bool, idx: bool) -> list:
    out = []
    if qnorm:
        out.append(("qnorm", "GluePdl::qnorm"))
    if wob:
        out.append(("wob_scale", "GluePdl::wob_scale"))
    if idx:
        out.append(("index_q", "GluePdlIdx::index_q"))
    return out
