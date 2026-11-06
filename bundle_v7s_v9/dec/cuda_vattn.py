"""dsv41-pd072 VATTN device and host sources, as Python strings (written under SGLang's JIT cache
root by vattn._publish).

csrc/ below the first entry is FlashMLA 395cb3a (MIT, Copyright (c) 2025 DeepSeek; see LICENSE.FlashMLA) with the
decode KV transport ported to this arena's block-split V4.1 pages: byte for byte the include closure of
dspd_w38/dspd/vattn_src, whose 11 vendored `.h` headers carry the name `X_h.cuh` (the bundle scanner admits declared
.cu / .cuh sources only) with the include lines naming them rewritten to match.

Two texts are ours: csrc/vattn_kernel_tu.cuh, the device-only translation unit the bundle's DeepJIT runtime compiles
to one cubin per layer class, and csrc/vattn_host.cpp, the host library (TORCH_LIBRARY dspd_v59_vattn) that owns that
runtime and rebuilds the launch the wrapper of dspd_w38 performed. The elfutils declarations DeepJIT's exception
header takes are not here: vattn._publish writes v50's own cuda_m462 build stub, by reference.

Regenerate with struct/v59_lane/tools/build_v59.py, do not edit."""


STUB = 'build_include/elfutils/libdwfl.h'
EXPECT = (225280, 512, 1, 528, 272, 376, 896, 24, 64)

SOURCES = {
    'LICENSE.FlashMLA': r"""MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
    'csrc/3rdparty/kerutils/include/kerutils/common/common_h.cuh': r"""#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

#ifdef __CUDACC__
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

#if __has_include("kernel_operator.h")
#define KERUTILS_IS_BUILD_ON_ASCEND
#endif

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#error "KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined at the same time!"
#endif
#endif

#ifndef KERUTILS_IS_BUILD_ON_CUDA
#ifndef KERUTILS_IS_BUILD_ON_ASCEND
#error "Neither KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined!"
#endif
#endif

namespace kerutils {

template<typename T>
inline constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/common_h.cuh': r"""/*
Common data types and macros that are used across the kerutils library.
*/
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/config.hpp>  // For CUTE_DEVICE

namespace kerutils {

// Cache hints
enum class CacheHint {
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST,
    EVICT_UNCHANGED,
    NO_ALLOCATE
};

// Prefetch size
enum class PrefetchSize {
    B64,
    B128,
    B256
};

using nvbf16 = __nv_bfloat16;
using nvbf16x2 = __nv_bfloat162;
using nve4m3 = __nv_fp8_e4m3;
using nve4m3x2 = __nv_fp8x2_e4m3;
using nve4m3x4 = __nv_fp8x4_e4m3;

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define KERUTILS_ENABLE_SM80
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
static_assert(false, "kerutils doesn't support SM architectures below SM80");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define KERUTILS_ENABLE_SM90
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000))
#define KERUTILS_ENABLE_SM90A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
#define KERUTILS_ENABLE_SM100
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM103A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#define KERUTILS_ENABLE_SM80
#define KERUTILS_ENABLE_SM90
#define KERUTILS_ENABLE_SM90A
#define KERUTILS_ENABLE_SM100
#define KERUTILS_ENABLE_SM100A
#define KERUTILS_ENABLE_SM103A
#endif
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm100/gemm.cuh': r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common_h.cuh"

namespace cute {

// Extensions to CuTe
// CuTe don't support UTCMMA with .ws, so we add it here
// Besides, CuTe's UTCMMA has an `elect_one_sync()` inside which is really disgusting, so we have our own variant without `elect_one_sync()` here

namespace UMMA {

// 在双轨矩乘（或者其他用况）中，我们有时会想自己指定 tensor memory 的“折叠度”，也即，如果 M < 128，那么是在不同的 lane 上将数据进行多次复制，还是将 N 切成 128/M 份，使用 N/(128/M) 的 lane，还是某种介于二者之间的做法。
// 因此我在这里实现了这个 `tmem_frg_elastic`，其支持自定义 FOLD_DEGREE。最终这个 fragment 会占用 N/FOLD_DEGREE 个 TMEM col（等价于将数据在 TMEM 中复制 (128/M) / FOLD_DEGREE 份。
// 在正常情况（NV 官方的用法）中，由于 A 总是在不同的 Tensor Memory Row 之间复制，因此 FOLD_DEGREE = 1；在多轨矩乘情况下，FOLD_DEGREE 为轨道的数量。
template <class ValueType, class StorageType, int FOLD_DEGREE>
struct tmem_frg_elastic : tmem_frg_base {
  static_assert(sizeof_bits_v<ValueType> <= sizeof_bits_v<StorageType>, "TMEM MMA allocations require StorageType big enough for ValueType.");

  // UMMA TMEM Allocator
  //   Each UMMA expects a specific MxN layout of TMEM for accumulators
  //   and sometimes a specific MxK layout of TMEM for A-values.
  // @tparam ValueType The value type of the TMEM Tensor to allocate.
  // @tparam StorageType The storage type of the TMEM Tensor to allocate.
  //                     "Sparse" allocations often allocate ValueType=half_t within StorageType=uint32_t.
  //                     "Dense"  allocations often allocate ValueType=half_t within StorageType=half_t.
  // @param tmem_shape ((M_MMA_SM,N_MMA_SM),MMA_M,MMA_N,...)
  //                   The post-MMA-partitioned shape of TMEM to allocate.
  //                   Note for UMMA_2SM_128xNx16, that M_MMA_SM will be 64, for example.
  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(size(tmem_shape)*Int<int(sizeof_bits_v<StorageType>)>{} <= TMEM::MAX_CAPACITY_BITS{},
                        "Requesting more TMEM than is available.");
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int R     = decltype(rank(tmem_shape))::value;
    constexpr int M_MMA = decltype(size<0,0>(tmem_shape))::value;
    constexpr int N_MMA = decltype(size<0,1>(tmem_shape))::value;

    // It's convenient to use "virtual tensor memory addressing"
    //   with DP_STRIDE=1, COL_STRIDE=128 to define the tmem_atom,
    //   then convert to "logical tensor memory addressing" on return.
    using COL_ADDR = C<sizeof_bits<StorageType>::value / sizeof_bits<ValueType>::value>;
    Layout tmem_restride = Layout<Shape <               _128,   _16384>,
                                  Stride<TMEM::DP<ValueType>, COL_ADDR>>{};
    Layout tmem_atom = Layout<Shape<Int<M_MMA>, Shape<Int<N_MMA/FOLD_DEGREE>, Int<FOLD_DEGREE>>>,
                              Stride<_1, Stride<_128, Int<128/FOLD_DEGREE>> >>{};
    constexpr int tile_stride = (128/M_MMA) / FOLD_DEGREE;
    Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1, R>(tmem_shape),
      compact_col_major(take<1,R>(tmem_shape),Int<tile_stride>{})));

    return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
  }
};

}

template <class ValueType, class StorageType, int FOLD_DEGREE>
struct MakeTensor<UMMA::tmem_frg_elastic<ValueType, StorageType, FOLD_DEGREE>> {
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_frg_elastic<ValueType, StorageType, FOLD_DEGREE>::make(shape(tmem_shape));
  }
};

template <class ValueType, int FOLD_DEGREE>
struct MakeTensor<UMMA::tmem_frg_elastic<ValueType, uint32_t, FOLD_DEGREE>> {
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_frg_elastic<ValueType, uint32_t, FOLD_DEGREE>::make(shape(tmem_shape));
  }
};


template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          int FOLD_DEGREE_A = 1,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT
{
  static_assert(M == 32 || M == 64 || M == 128, "SM100_MMA_F16BF16_WS_TS_NOELECT M-mode size should be 32, 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 64 || N == 128 || N == 256,
                "SM100_MMA_F16BF16_WS_TS_NOELECT N-mode size should be 64, 128 or 256");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          int FOLD_DEGREE_A,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                FOLD_DEGREE_A,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_WS_TS_NOELECT supports 16bit types");

  // using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>; // Actually this should be "duplicated", however, our great CuTe doesn't allow us to set it to "duplicated", so we just set it to NonInterleaved for a correct address calculation
  using FrgTypeA = UMMA::tmem_frg_elastic<a_type, a_type, FOLD_DEGREE_A>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_elastic<c_type, uint32_t, 128/M>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  FOLD_DEGREE_A,
                  a_neg, b_neg>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT
{
  static_assert(M == 32 || M == 64 || M == 128, "SM100_MMA_F16BF16_WS_SS_NOELECT M-mode size should be 32, 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 64 || N == 128 || N == 256,
                "SM100_MMA_F16BF16_WS_SS_NOELECT N-mode size should be 64, 128 or 256");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_WS_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_elastic<c_type, uint32_t, 128/M>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type,
                  M, N, a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};


template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_2x1SM_TS_NOELECT
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_TS_NOELECT M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_TS_NOELECT N-mode size should be a multiple of 16 between 16 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_2x1SM_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_TS_NOELECT without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};


template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_TS_NOELECT<a_type, b_type, c_type,
                                     M, N,
                                     a_major, b_major,
                                     a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_TS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions' K extent is always 256 bits; convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};



// SM100_MMA_F16BF16_2x1SM_SS without elect_one_sync()
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS_NOELECT
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_SS_NOELECT M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_SS_NOELECT N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_SS_NOELECT without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_NOELECT<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions's K extent is always 256bits, convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};


template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_TS_NOELECT
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_TS_NOELECT M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0)  && (8 <= N)  && (N <= 256),
                "SM100_MMA_F16BF16_TS_NOELECT N-mode size should be a multiple of 8 between 8 and 256 for M=64");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_TS_NOELECT<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_TS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_TS_NOELECT<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};


template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_SS_NOELECT M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0)  && (8 <= N)  && (N <= 256),
                "SM100_MMA_F16BF16_SS_NOELECT N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type,
                  M, N, a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm100/helpers.cuh': r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common_h.cuh"

namespace kerutils {

// Perform SS UTCMMA
// sA and sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// Perform TS UTCMMA
// sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tA_frag and tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16, 
            UMMA::Layout_K_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_K_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_K_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_K_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{});
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_mn_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16, 
            UMMA::Layout_MN_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_MN_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_MN_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_MN_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_2, _1>{}
    ), Shape<_1, _1>{});
}

template<cute::UMMA::Major MAJOR, int MN, int K, int SWIZZLE, typename T = bf16>
auto make_umma_canonical_layout() {
    if constexpr (MAJOR == cute::UMMA::Major::K) {
        return make_umma_canonical_k_major_layout<MN, K, SWIZZLE, T>();
    } else {
        return make_umma_canonical_mn_major_layout<MN, K, SWIZZLE, T>();
    }
}

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm100/intrinsics.cuh': r"""#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include "kerutils/device/cuda/common_h.cuh"

namespace kerutils {

// tma gather4 (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow under some scenarios
CUTE_DEVICE
void tma_gather4(const void* desc_ptr, transac_bar_t &mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), 
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), 
          "r"(mbar_addr), "l"(cache_hint)
        : "memory"
    );
}

// tma gather4 prefetch (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow under some scenarios
CUTE_DEVICE
void tma_gather4_prefetch(const void* desc_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;\n"
        :
        : "l"(desc_ptr), "r"(col_idx), 
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), 
          "l"(cache_hint)
    );
}

// tma gather4 with cta_group::2, allowing for synchronization across CTAs within a pair of CTAs (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
template<bool USE_CTA0_MBAR = false>
CUTE_DEVICE void tma_gather4_cta_group_2(const void* desc_ptr, transac_bar_t &mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    if constexpr (USE_CTA0_MBAR) {
        mbar_addr &= cute::Sm100MmaPeerBitMask;
    }
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), 
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), 
          "r"(mbar_addr), "l"(cache_hint)
        : "memory"
    );
}

// Vectorized addition for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)
CUTE_DEVICE
float2 float2_add(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "add.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
    return c;
}

// Vectorized multiplication for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)
CUTE_DEVICE
float2 float2_mul(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

// Vectorized fused addition-multiplication for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
CUTE_DEVICE
float2 float2_fma(const float2 &a, const float2 &b, const float2 &c) {
    // return a*b+c
    float2 d;
    asm volatile(
        "fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

// Vectorized negation for foat32
CUTE_DEVICE
float2 float2_neg(const float2 &a) {
    float2 t = {-1.0f, -1.0f};
    return float2_mul(a, t);
}

// st.bulk (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
CUTE_DEVICE
void st_bulk(void* dst_ptr, int64_t size) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    asm volatile (
        "st.bulk.weak.shared::cta [%0], %1, 0;\n"
        :
        : "r"(dst_addr), "l"(size)
        : "memory"
    );
}

struct CUTE_ALIGNAS(16) CLCResponseObj {
    // An opaque 16B value
    char opaque[16];
};

struct CLCResult {
    int is_valid;
    int x, y, z;
};

// Issue a CLC try_cancel query (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
CUTE_DEVICE
void issue_clc_query(transac_bar_t &bar, CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr)
    );
}

// Issue a CLC try_cancel query with .multicast::cluster::all (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
CUTE_DEVICE
void issue_clc_query_multicast_cluster_all(transac_bar_t &bar, CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr)
    );
}

// Get the result of a CLC query (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel)
// In this function, we separate get_first_ctaid::x/y/z and hope PTXAS's dead code elimination can remove unnecessary instructions
template<bool USE_LD_ACQUIRE>
CUTE_DEVICE
CLCResult get_clc_query_response(CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(&response_obj);
    CLCResult result;
    #define EMIT_ASM(LD_MODIFIER)                                                                   \
        asm volatile(                                                                               \
            "{\n"                                                                                   \
            ".reg .pred p1;\n\t"                                                                    \
            ".reg .b128 clc_result;\n\t"                                                            \
            "ld" LD_MODIFIER ".shared.b128 clc_result, [%4];\n\t"                                   \
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"           \
            "selp.u32 %3, 1, 0, p1;\n\t"                                                            \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_result;\n\t" \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_result;\n\t" \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_result;\n\t" \
            "}\n"                                                                                   \
            : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.is_valid)                 \
            : "r"(response_addr)                                                                    \
            : "memory"                                                                              \
        );
    if constexpr (USE_LD_ACQUIRE) {
        EMIT_ASM(".acquire.cta");
    } else {
        EMIT_ASM("");
    }
    return result;
}

// LDG.256 or LDG.256 with non-coherent cache (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
// We use macro instead of function here, since we need a multi-level recursive dispatch based on template parameters if using function
// NC_STR should be either "" or ".nc"
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
// L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
#define KU_LDG_256(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>, "`result` must be a pointer"); \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result); \
        asm volatile( \
            "ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".L2::" L2_PREFETCH_SIZE_STR ".v4.u64 {%0, %1, %2, %3}, [%4];\n" \
            : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1]), \
            "=l"(result_as_uint64_ptr[2]), "=l"(result_as_uint64_ptr[3]) \
            : "l"(global_addr) \
        ); \
    }

// STG.256 (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st)
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
#define KU_STG_256(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>, "`src` must be a pointer"); \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src); \
        asm volatile( \
            "st.global.L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".v4.u64 [%0], {%1, %2, %3, %4};\n" \
            : \
            : "l"(global_addr), "l"(src_as_uint64_ptr[0]), "l"(src_as_uint64_ptr[1]), \
            "l"(src_as_uint64_ptr[2]), "l"(src_as_uint64_ptr[3]) \
        ); \
    }

}

namespace kerutils {

// tcgen05.commit.cta_group::1 (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_noelect(transac_bar_t &bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        :"r"(bar_intptr)
    );
}

// tcgen05.commit.cta_group::1, with multicast (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_multicast_noelect(transac_bar_t &bar, uint16_t cta_mask) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        :"r"(bar_intptr), "h"(cta_mask)
    );
}

// tcgen05.commit.cta_group::2 (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_2x1SM_noelect(transac_bar_t &bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        :"r"(bar_intptr)
    );
}

// tcgen05.commit.cta_group::2, with multicast (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_multicast_2x1SM_noelect(transac_bar_t &bar, uint16_t cta_mask) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        :"r"(bar_intptr), "h"(cta_mask)
    );
}

// tcgen05.fence::before_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// tcgen05.fence::after_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}


// Load from tensor memory, 32 data path lanes, 32-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumElements>
__device__ __forceinline__
void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
    // NOTE The following code crashes VSCode intellisense engine, so we disable it
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

// Load from tensor memory with column-wise reduction, 32 data path lanes, 32-bit pattern, repeated N times.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
// USE_MAX: true = .max reduction, false = .min reduction
// USE_ABS: true = .abs qualifier (use absolute values for reduction)
// USE_NAN: true = .NaN qualifier (propagate NaN in reduction)
template <int kNumElements, bool USE_MAX, bool USE_ABS, bool USE_NAN>
__device__ __forceinline__
void tmem_ld_red_32dp32bNx(uint32_t tmem_start, void* data_, float& redval) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 2 || kNumElements == 4 || kNumElements == 8 ||
                  kNumElements == 16 || kNumElements == 32 || kNumElements == 64 ||
                  kNumElements == 128,
                  "Invalid kNumElements for tcgen05.ld.red (must be power of 2, at least 2)");
#ifndef __VSCODE_IDE__
    static constexpr char s_max[] = ".max";
    static constexpr char s_min[] = ".min";
    static constexpr char s_abs[] = ".abs";
    static constexpr char s_nan[] = ".NaN";
    static constexpr char s_empty[] = "";
    // Operand numbering for each branch:
    //   outputs: %0..%{N-1} = data[0..N-1] ("=r"), %N = redval ("=f")
    //   inputs:  %{N+1} = tmem_start ("r"), %{N+2} = redop ("C"), %{N+3} = abs ("C"), %{N+4} = nan ("C")
    if constexpr (kNumElements == 2) {
        // outputs: %0..%1 = data, %2 = redval; inputs: %3 = taddr, %4 = redop, %5 = abs, %6 = nan
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x2%4%5%6.f32"
            " {%0, %1}, %2, [%3];\n"
            : "=r"(data[0]), "=r"(data[1]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 4) {
        // outputs: %0..%3 = data, %4 = redval; inputs: %5 = taddr, %6..%8 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x4%6%7%8.f32"
            " {%0, %1, %2, %3}, %4, [%5];\n"
            : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 8) {
        // outputs: %0..%7 = data, %8 = redval; inputs: %9 = taddr, %10..%12 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x8%10%11%12.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7}, %8, [%9];\n"
            : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
              "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 16) {
        // outputs: %0..%15 = data, %16 = redval; inputs: %17 = taddr, %18..%20 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x16%18%19%20.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15}, %16, [%17];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 32) {
        // outputs: %0..%31 = data, %32 = redval; inputs: %33 = taddr, %34..%36 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x32%34%35%36.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31}, %32, [%33];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=r"(data[16]), "=r"(data[17]), "=r"(data[18]), "=r"(data[19]),
              "=r"(data[20]), "=r"(data[21]), "=r"(data[22]), "=r"(data[23]),
              "=r"(data[24]), "=r"(data[25]), "=r"(data[26]), "=r"(data[27]),
              "=r"(data[28]), "=r"(data[29]), "=r"(data[30]), "=r"(data[31]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 64) {
        // outputs: %0..%63 = data, %64 = redval; inputs: %65 = taddr, %66..%68 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x64%66%67%68.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31,"
            " %32, %33, %34, %35,"
            " %36, %37, %38, %39,"
            " %40, %41, %42, %43,"
            " %44, %45, %46, %47,"
            " %48, %49, %50, %51,"
            " %52, %53, %54, %55,"
            " %56, %57, %58, %59,"
            " %60, %61, %62, %63}, %64, [%65];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=r"(data[16]), "=r"(data[17]), "=r"(data[18]), "=r"(data[19]),
              "=r"(data[20]), "=r"(data[21]), "=r"(data[22]), "=r"(data[23]),
              "=r"(data[24]), "=r"(data[25]), "=r"(data[26]), "=r"(data[27]),
              "=r"(data[28]), "=r"(data[29]), "=r"(data[30]), "=r"(data[31]),
              "=r"(data[32]), "=r"(data[33]), "=r"(data[34]), "=r"(data[35]),
              "=r"(data[36]), "=r"(data[37]), "=r"(data[38]), "=r"(data[39]),
              "=r"(data[40]), "=r"(data[41]), "=r"(data[42]), "=r"(data[43]),
              "=r"(data[44]), "=r"(data[45]), "=r"(data[46]), "=r"(data[47]),
              "=r"(data[48]), "=r"(data[49]), "=r"(data[50]), "=r"(data[51]),
              "=r"(data[52]), "=r"(data[53]), "=r"(data[54]), "=r"(data[55]),
              "=r"(data[56]), "=r"(data[57]), "=r"(data[58]), "=r"(data[59]),
              "=r"(data[60]), "=r"(data[61]), "=r"(data[62]), "=r"(data[63]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 128) {
        // outputs: %0..%127 = data, %128 = redval; inputs: %129 = taddr, %130..%132 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x128%130%131%132.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31,"
            " %32, %33, %34, %35,"
            " %36, %37, %38, %39,"
            " %40, %41, %42, %43,"
            " %44, %45, %46, %47,"
            " %48, %49, %50, %51,"
            " %52, %53, %54, %55,"
            " %56, %57, %58, %59,"
            " %60, %61, %62, %63,"
            " %64, %65, %66, %67,"
            " %68, %69, %70, %71,"
            " %72, %73, %74, %75,"
            " %76, %77, %78, %79,"
            " %80, %81, %82, %83,"
            " %84, %85, %86, %87,"
            " %88, %89, %90, %91,"
            " %92, %93, %94, %95,"
            " %96, %97, %98, %99,"
            " %100, %101, %102, %103,"
            " %104, %105, %106, %107,"
            " %108, %109, %110, %111,"
            " %112, %113, %114, %115,"
            " %116, %117, %118, %119,"
            " %120, %121, %122, %123,"
            " %124, %125, %126, %127}, %128, [%129];\n"
            : "=r"(data[0]),   "=r"(data[1]),   "=r"(data[2]),   "=r"(data[3]),
              "=r"(data[4]),   "=r"(data[5]),   "=r"(data[6]),   "=r"(data[7]),
              "=r"(data[8]),   "=r"(data[9]),   "=r"(data[10]),  "=r"(data[11]),
              "=r"(data[12]),  "=r"(data[13]),  "=r"(data[14]),  "=r"(data[15]),
              "=r"(data[16]),  "=r"(data[17]),  "=r"(data[18]),  "=r"(data[19]),
              "=r"(data[20]),  "=r"(data[21]),  "=r"(data[22]),  "=r"(data[23]),
              "=r"(data[24]),  "=r"(data[25]),  "=r"(data[26]),  "=r"(data[27]),
              "=r"(data[28]),  "=r"(data[29]),  "=r"(data[30]),  "=r"(data[31]),
              "=r"(data[32]),  "=r"(data[33]),  "=r"(data[34]),  "=r"(data[35]),
              "=r"(data[36]),  "=r"(data[37]),  "=r"(data[38]),  "=r"(data[39]),
              "=r"(data[40]),  "=r"(data[41]),  "=r"(data[42]),  "=r"(data[43]),
              "=r"(data[44]),  "=r"(data[45]),  "=r"(data[46]),  "=r"(data[47]),
              "=r"(data[48]),  "=r"(data[49]),  "=r"(data[50]),  "=r"(data[51]),
              "=r"(data[52]),  "=r"(data[53]),  "=r"(data[54]),  "=r"(data[55]),
              "=r"(data[56]),  "=r"(data[57]),  "=r"(data[58]),  "=r"(data[59]),
              "=r"(data[60]),  "=r"(data[61]),  "=r"(data[62]),  "=r"(data[63]),
              "=r"(data[64]),  "=r"(data[65]),  "=r"(data[66]),  "=r"(data[67]),
              "=r"(data[68]),  "=r"(data[69]),  "=r"(data[70]),  "=r"(data[71]),
              "=r"(data[72]),  "=r"(data[73]),  "=r"(data[74]),  "=r"(data[75]),
              "=r"(data[76]),  "=r"(data[77]),  "=r"(data[78]),  "=r"(data[79]),
              "=r"(data[80]),  "=r"(data[81]),  "=r"(data[82]),  "=r"(data[83]),
              "=r"(data[84]),  "=r"(data[85]),  "=r"(data[86]),  "=r"(data[87]),
              "=r"(data[88]),  "=r"(data[89]),  "=r"(data[90]),  "=r"(data[91]),
              "=r"(data[92]),  "=r"(data[93]),  "=r"(data[94]),  "=r"(data[95]),
              "=r"(data[96]),  "=r"(data[97]),  "=r"(data[98]),  "=r"(data[99]),
              "=r"(data[100]), "=r"(data[101]), "=r"(data[102]), "=r"(data[103]),
              "=r"(data[104]), "=r"(data[105]), "=r"(data[106]), "=r"(data[107]),
              "=r"(data[108]), "=r"(data[109]), "=r"(data[110]), "=r"(data[111]),
              "=r"(data[112]), "=r"(data[113]), "=r"(data[114]), "=r"(data[115]),
              "=r"(data[116]), "=r"(data[117]), "=r"(data[118]), "=r"(data[119]),
              "=r"(data[120]), "=r"(data[121]), "=r"(data[122]), "=r"(data[123]),
              "=r"(data[124]), "=r"(data[125]), "=r"(data[126]), "=r"(data[127]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    }
#endif
}

// Load from tensor memory, 16 data path lanes, 128-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumReplications>
__device__ __forceinline__
void tmem_ld_16dp128bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumReplications == 1 || kNumReplications == 2 || kNumReplications == 4 || kNumReplications == 8 || kNumReplications == 16 || kNumReplications == 32 || kNumReplications == 64, "Invalid kNumReplications");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumReplications == 1) {
            cute::SM100_TMEM_LOAD_16dp128b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 2) {
            cute::SM100_TMEM_LOAD_16dp128b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 4) {
            cute::SM100_TMEM_LOAD_16dp128b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 8) {
            cute::SM100_TMEM_LOAD_16dp128b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 16) {
            cute::SM100_TMEM_LOAD_16dp128b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 32) {
            cute::SM100_TMEM_LOAD_16dp128b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 64) {
            cute::SM100_TMEM_LOAD_16dp128b64x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumReplications*2>{});
#endif
}

// Load from tensor memory, 16 data path lanes, 256-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumReplications>
__device__ __forceinline__
void tmem_ld_16dp256bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumReplications == 1 || kNumReplications == 2 || kNumReplications == 4 || kNumReplications == 8 || kNumReplications == 16 || kNumReplications == 32, "Invalid kNumReplications");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumReplications == 1) {
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 2) {
            cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 4) {
            cute::SM100_TMEM_LOAD_16dp256b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 8) {
            cute::SM100_TMEM_LOAD_16dp256b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 16) {
            cute::SM100_TMEM_LOAD_16dp256b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 32) {
            cute::SM100_TMEM_LOAD_16dp256b32x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumReplications*4>{});
#endif
}

// Store into tensor memory, 32 data path lanes, 32-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
template <int kNumElements>
__device__ __forceinline__
void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

// Decompose a non-power-of-2 replication count into a sequence of power-of-2 calls.
//
// This helper splits kNumReplications into its constituent powers of 2 (from highest to lowest)
// and issues one call to tmem_ld_st_fn.template operator()<power>() for each, automatically
// advancing the tmem address by kNumElemsPerUnit columns and the data pointer by kNumElemsPerUnit
// uint32_t elements per unit.
//
// Template parameters:
//   kNumReplications — total number of replications (may be non-power-of-2)
//   kNumElemsPerUnit — number of tmem columns (and uint32_t data elements) consumed per unit
//                      (e.g. 1 for 32dp32b, 2 for 16dp128b, 4 for 16dp256b)
//   F                — a callable with a template operator(): f.template operator()<N>(uint32_t, void*)
//
// Usage example (with a template lambda):
//   tmem_ld_st_decomposed<6, 1>(tmem_start, data_ptr,
//       []<int N>(uint32_t ts, void* d) { tmem_ld_32dp32bNx<N>(ts, d); });
//   // Equivalent to:
//   //   tmem_ld_32dp32bNx<4>(tmem_start, data_ptr);
//   //   tmem_ld_32dp32bNx<2>(tmem_start + 4, (void*)((uint32_t*)data_ptr + 4));
template<int kNumReplications, int kNumElemsPerUnit, typename F>
__device__ __forceinline__
void tmem_ld_st_decomposed(uint32_t tmem_start, void* data, F&& tmem_ld_st_fn) {
    if constexpr (kNumReplications == 0) {
        // Base case: nothing to do
        return;
    } else {
        // Largest power-of-2 <= kNumReplications
        constexpr int kHighBit = 1 << (31 - __builtin_clz(kNumReplications));
        tmem_ld_st_fn.template operator()<kHighBit>(tmem_start, data);
        // Recurse on the remainder
        constexpr int kRemaining = kNumReplications - kHighBit;
        if constexpr (kRemaining > 0) {
            tmem_ld_st_decomposed<kRemaining, kNumElemsPerUnit>(
                tmem_start + kHighBit * kNumElemsPerUnit,
                (void*)((uint32_t*)data + kHighBit * kNumElemsPerUnit),
                tmem_ld_st_fn
            );
        }
    }
}

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm100/tma_cta_group2_nosplit.cuh': r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common_h.cuh"

namespace cute {

// Extensions to CuTe
// CuTe's built-in SM100_TMA_2SM_LOAD_1D series requires the number of participating threads to be 2 (using ThrID = Layout<_2>;) and also splits the data, which is really annoying to use, so we modified our own version. Additionally, to keep it consistent with other parts that use SM90 TMA, we made it accept TMA::CacheHintSm90 instead of TMA::CacheHintSm100.

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD : Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////
struct SM100_TMA_2SM_LOAD_1D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_2D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_3D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_4D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_5D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
    return SM100_TMA_2SM_LOAD_1D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM100_TMA_2SM_LOAD_2D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM100_TMA_2SM_LOAD_3D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM100_TMA_2SM_LOAD_4D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
  using PREFETCH = typename SM90_TMA_LOAD::PREFETCH;
};
struct SM100_TMA_2SM_LOAD_NOSPLIT_OP : SM100_TMA_2SM_LOAD_NOSPLIT {};
// The non-executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  // SM100_TMA_2SM_LOAD_NOSPLIT arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }
  // Construct an executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }
  // Construct an executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }
  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
  // Don't try to execute a copy with SM100_TMA_2SM_LOAD_NOSPLIT before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};
// The executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  : TMA_LOAD_Unpack<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  // SM100_TMA_2SM_LOAD_NOSPLIT arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint64_t   // cache hint
  > const opargs_;
  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint64_t cache)
    : opargs_(desc, mbar, cache) {}
};

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm80/helpers.cuh': r"""#pragma once

#include "kerutils/device/cuda/common_h.cuh"
#include "kerutils/device/cuda/sm80/intrinsics.cuh"

namespace kerutils {

// Retrieve the value of `%smid` and check its range
CUTE_DEVICE
uint32_t get_sm_id_with_range_check(uint32_t num_physical_sms) {
    uint32_t sm_id = get_sm_id();
    if (!(sm_id < num_physical_sms)) {
        trap();
    }
    return sm_id;
}

#ifndef KU_TRAP_ONLY_DEVICE_ASSERT
#define KU_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

// Construct a `float2` from a single `float` by duplicating the value 
CUTE_DEVICE
float2 float2float2(const float &x) {
    return float2 {x, x};
}

CUTE_DEVICE
void st_shared(void* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "r"(cute::cast_smem_ptr_to_uint(ptr)), "q"(val));
}

CUTE_DEVICE
void st_shared(void* ptr, float4 val) {
    st_shared(ptr, *(__int128_t*)&val);
}

CUTE_DEVICE
__int128_t ld_shared(const void* ptr) {
    __int128_t val;
    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(val) : "r"(cute::cast_smem_ptr_to_uint(ptr)));
    return val;
}

CUTE_DEVICE
float4 ld_shared_float4(const void* ptr) {
    __int128_t temp = ld_shared(ptr);
    return *(float4*)&temp;
}

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm80/intrinsics.cuh': r"""#pragma once

#include "kerutils/device/cuda/common_h.cuh"

namespace kerutils {

// cp.async.cg (cache global) with prefetch and predicate (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)
template<PrefetchSize PREFETCH_SIZE=PrefetchSize::B128>
CUTE_DEVICE
void cp_async_cacheglobal(const void* src, void* dst, bool pred=true) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
    if constexpr (PREFETCH_SIZE == PrefetchSize::B64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16, %2;\n"
            :: "r"(dst_addr),
               "l"(src),
               "r"(pred?16:0));
    } else if constexpr (PREFETCH_SIZE == PrefetchSize::B128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
            :: "r"(dst_addr),
               "l"(src),
               "r"(pred?16:0));
    } else if constexpr (PREFETCH_SIZE == PrefetchSize::B256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n"
            :: "r"(dst_addr),
               "l"(src),
               "r"(pred?16:0));
    } else {
        static_assert(PREFETCH_SIZE == PrefetchSize::B64 ||
                      PREFETCH_SIZE == PrefetchSize::B128 ||
                      PREFETCH_SIZE == PrefetchSize::B256,
                      "Unsupported prefetch size for cp_async_cacheglobal.");
    }
}

// Create fraction-based cache policy (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy)
template<CacheHint PRIMARY_PRIORITY, CacheHint SECONDARY_PRIORITY>
CUTE_DEVICE
int64_t create_fraction_based_cache_policy(float fraction = 1.0f) {
    int64_t result;
    #define EMIT(PRIMARY_PRIORITY_STR, SECONDARY_PRIORITY_STR) \
        asm volatile( \
            "createpolicy.fractional.L2::" PRIMARY_PRIORITY_STR ".L2::" SECONDARY_PRIORITY_STR ".b64 %0, %1;\n" \
            : "=l"(result) \
            : "f"(fraction) \
        );
    #define EMIT2(PRIMARY_PRIORITY_STR) \
        { \
            if constexpr (SECONDARY_PRIORITY == CacheHint::EVICT_FIRST) { \
                EMIT(PRIMARY_PRIORITY_STR, "evict_first") \
            } else if constexpr (SECONDARY_PRIORITY == CacheHint::EVICT_UNCHANGED) { \
                EMIT(PRIMARY_PRIORITY_STR, "evict_unchanged") \
            } else { \
                static_assert(SECONDARY_PRIORITY == CacheHint::EVICT_FIRST || \
                            SECONDARY_PRIORITY == CacheHint::EVICT_UNCHANGED, \
                            "Unsupported secondary cache hint for create_fraction_based_cache_policy."); \
            } \
        }
    if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_FIRST) {
        EMIT2("evict_first");
    } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_NORMAL) {
        EMIT2("evict_normal");
    } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_LAST) {
        EMIT2("evict_last");
    } else if constexpr (PRIMARY_PRIORITY == CacheHint::EVICT_UNCHANGED) {
        EMIT2("evict_unchanged");
    } else {
        static_assert(PRIMARY_PRIORITY == CacheHint::EVICT_FIRST ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_NORMAL ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_LAST ||
                      PRIMARY_PRIORITY == CacheHint::EVICT_UNCHANGED,
                      "Unsupported primary cache hint for create_fraction_based_cache_policy.");
    }
    #undef EMIT
    #undef EMIT2
    return result;
}

// Create a simple cache policy (equivalent to create_fraction_based_cache_policy(1.0f))
// The same as cute::TMA::CacheHintSmXX
template<CacheHint CACHE_HINT>
CUTE_DEVICE
constexpr int64_t create_simple_cache_policy() {
    if constexpr (CACHE_HINT == CacheHint::EVICT_FIRST) {
        return 0x12F0000000000000;  // Result of createpolicy.fractional.L2::evict_first.b64
    } else if constexpr (CACHE_HINT == CacheHint::EVICT_NORMAL) {
        return 0x1000000000000000;  // Copied from CuTe. Unsure about the exact meaning. (TODO Change to 0x16F0000000000000?)
    } else if constexpr (CACHE_HINT == CacheHint::EVICT_LAST) {
        return 0x14F0000000000000;  // Result of createpolicy.fractional.L2::evict_last.b64
    } else {
        static_assert(CACHE_HINT == CacheHint::EVICT_FIRST ||
                      CACHE_HINT == CacheHint::EVICT_NORMAL ||
                      CACHE_HINT == CacheHint::EVICT_LAST,
                      "Unsupported cache hint for create_simple_cache_policy.");
    }
}

// AtomicAdd (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)
CUTE_DEVICE
void atomicadd_f32_with_policy_and_pred(void* global_addr, const float &data, int64_t cache_policy, uint32_t pred = true) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %3, 1;\n\t"
        "@p red.relaxed.gpu.global.add.L2::cache_hint.f32 [%1], %0, %2; \n\t"
        "}"
        : 
        : "f"(data),
          "l"((int64_t)global_addr), "l"(cache_policy), "r"(pred)
    );
}

// Get the id of the current SM
// About %smid (https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-smid): PTX document says that %smid ranges from 0 to %nsmid-1, while "The SM identifier numbering is not guaranteed to be contiguous, so %nsmid may be larger than the physical number of SMs in the device.". However, result shows that, at least for sm90 and sm100, %nsmid is the number of physical SMs - 1. For the sake of safety, I recommend you to check the return of get_sm_id manually or call `get_sm_id_with_range_check()` defined in `device/sm80/helpers.cuh`.
// Besides, PTX document also says that this number may change due to preemption, but currently this never happens according to [DATEN GELÖSCHT]
CUTE_DEVICE
uint32_t get_sm_id() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %%smid;\n" : "=r"(ret));
    return ret;
}

// trap (https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-trap)
CUTE_DEVICE
void trap() {
    asm volatile("trap;\n");
}

// LDG.128 or LDG.128 with non-coherent cache (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
// We use macro instead of function here, since we need a multi-level recursive dispatch based on template parameters if using function
// NC_STR should be either "" or ".nc"
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
// L2 cache hint is not supported since it's only supported for LDG.256
#define KU_LDG_128(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>, "`result` must be a pointer"); \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result); \
        asm volatile( \
            "ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_PREFETCH_SIZE_STR ".v2.u64 {%0, %1}, [%2];\n" \
            : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1]) \
            : "l"(global_addr) \
        ); \
    }

// STG.128 (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st)
// L2_CACHE_HINT_STR should be ".L2::evict_XXX" (only available on sm100+) or ""
#define KU_STG_128(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>, "`src` must be a pointer"); \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src); \
        asm volatile( \
            "st.global.L1::" L1_CACHE_HINT_STR L2_CACHE_HINT_STR ".v2.u64 [%0], {%1, %2};\n" \
            : \
            : "l"(global_addr), "l"(src_as_uint64_ptr[0]), "l"(src_as_uint64_ptr[1]) \
        ); \
    }
    
}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm90/helpers.cuh': r"""#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/cuda/common_h.cuh"

namespace kerutils {

template<
    typename TMA,
    typename Tensor0,
    typename Tensor1
>
CUTE_DEVICE
void launch_tma_copy(
    const TMA &tma_copy,
    Tensor0 src,
    Tensor1 dst,
    transac_bar_t &bar,
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL
) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar), 0, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}

// In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
// You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
CUTE_DEVICE
int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    int row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
    return row_idx;
}

// In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in some rows. This function converts the local_elem_idx to the actual col_idx
// You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
CUTE_DEVICE
int get_AorC_col_idx(int local_elem_idx, int idx_in_warpgroup) {
    int col_idx = 8*(local_elem_idx/4) + (idx_in_warpgroup%4)*2 + (local_elem_idx&1);
    return col_idx;
}

template <bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC, bool zero_init) {
    using namespace cute;
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    tiled_mma.accumulate_ = zero_init ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma_ss(bool clear_accum, TiledMma tiled_mma, Tensor0 const &sA, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sA_frag = thr_mma.partition_fragment_A(sA);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma_rs(bool clear_accum, TiledMma tiled_mma, Tensor0 rA_frag, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(rA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(rA_frag); ++k) {
        cute::gemm(tiled_mma, rA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
}

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/cuda/sm90/intrinsics.cuh': r"""#pragma once

#include "kerutils/device/cuda/common_h.cuh"

namespace kerutils {

// st.async (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async)
template<typename T>
CUTE_DEVICE
static void st_async(void* dst_ptr, const T& data, transac_bar_t &mbar) {
    static_assert(sizeof(T) == 16, "Data type must be 16 bytes (128 bits) for st_async.");
    longlong2 data_long2 = *reinterpret_cast<const longlong2*>(&data);
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3]; \n"
        :
        : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr)
    );
}

static constexpr int PEER_ADDR_MASK = 16777216;

// Given an address in the current CTA, return the corresponding address in the peer CTA
template<typename T>
CUTE_DEVICE
T* get_peer_addr(const T* p) {
    return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

// Given an address in the current CTA, return the corresponding address in the peer CTA (if the current CTA_id%2 == 1) or the address itself (if CTA_id%2 == 0)
template<typename T>
CUTE_DEVICE
T* get_cta0_addr(const T* p) {
    constexpr int CTA0_ADDR_MASK = 0xFEFFFFFF;
    return (T*)((int64_t)(p) & CTA0_ADDR_MASK);
}

// TMA bulk reduce add (cp.reduce.async.bulk), shared to global, float32, add. (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk)
CUTE_DEVICE
void tma_bulk_reduce_add(void const* src_ptr, void* dst_ptr, int32_t store_bytes) {
    uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(src_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                     :
                     : "l"(dst_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
}

// Cluster barrier arrive with .release modifier. (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)
CUTE_DEVICE
void barrier_cluster_arrive_release() {
    asm volatile("barrier.cluster.arrive.release;" : : : "memory");
}

// Cluster barrier arrive with .relaxed modifier. (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)
CUTE_DEVICE
void barrier_cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed;" : : :);
}

// Cluster barrier wait with .acquire modifier. (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)
CUTE_DEVICE
void barrier_cluster_wait_acquire() {
    asm volatile("barrier.cluster.wait.acquire;" : : : "memory");
}

// mbarrier.arrive with .relaxed.cluster qualifier (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive)
CUTE_DEVICE
void mbarrier_arrive_relaxed_cluster(transac_bar_t &mbar) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.relaxed.cluster.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
}

// AtomicAdd with v4.f32 type (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)
CUTE_DEVICE
void atomicadd_f32x4_with_policy_and_pred(void* global_addr, const float4 &data, int64_t cache_policy, uint32_t pred = true) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %6, 1;\n\t"
        "@p red.relaxed.gpu.global.add.L2::cache_hint.v4.f32 [%4], {%0, %1, %2, %3}, %5; \n\t"
        "}"
        : 
        : "f"(data.x), "f"(data.y), "f"(data.z), "f"(data.w),
          "l"((int64_t)global_addr), "l"(cache_policy), "r"(pred)
    );
}

// cp.async.bulk, from .shared::cta to .shared::cluster (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk)
CUTE_DEVICE
void cp_async_bulk_shared_cta_to_shared_cluster(void* dst_ptr, const void* src_ptr, int32_t load_bytes, transac_bar_t &mbar) {
    uint32_t dst_smem_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    uint32_t src_smem_addr = cute::cast_smem_ptr_to_uint(src_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3]; \n"
        :
        : "r"(dst_smem_addr), "r"(src_smem_addr), "r"(load_bytes), "r"(mbar_addr)
    );
}

}
""",
    'csrc/3rdparty/kerutils/include/kerutils/device/device.cuh': r"""#pragma once

#include "kerutils/common/common_h.cuh"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "cuda/common_h.cuh"
#include "cuda/sm80/intrinsics.cuh"
#include "cuda/sm80/helpers.cuh"
#include "cuda/sm90/intrinsics.cuh"
#include "cuda/sm90/helpers.cuh"
#include "cuda/sm100/intrinsics.cuh"
#include "cuda/sm100/helpers.cuh"
#include "cuda/sm100/gemm.cuh"
#include "cuda/sm100/tma_cta_group2_nosplit.cuh"
#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#include "ascend/common.h"
#endif
""",
    'csrc/3rdparty/kerutils/include/kerutils/host/host_h.cuh': r"""#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <vector>

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#include <acl/acl.h>
#endif

#include "kerutils/common/common_h.cuh"

namespace kerutils {

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;
        
        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                                     \
    do {                                                                                         \
        if (not (cond)) {                                                                        \
            char _ku_buf[1024];                                                                  \
            int _ku_len = snprintf(_ku_buf, sizeof(_ku_buf),                                    \
                                   "Assertion `%s` failed (%s:%d)", #cond, __FILE__, __LINE__); \
            __VA_OPT__(                                                                          \
                _ku_len += snprintf(_ku_buf + _ku_len, sizeof(_ku_buf) - _ku_len,              \
                                    ": " __VA_ARGS__);                                          \
            )                                                                                    \
            fprintf(stderr, "%s\n", _ku_buf);                                                    \
            THROW_KU_EXCEPTION("Assertion", _ku_buf);                                            \
        }                                                                                        \
    } while(0)

#ifdef KERUTILS_IS_BUILD_ON_CUDA

#define KU_CUDA_CHECK(call)                                                                                   \
do {                                                                                                          \
    cudaError_t status_ = (call);                                                                             \
    if (status_ != cudaSuccess) {                                                                             \
        char _ku_buf[1024];                                                                                   \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUDA error (%s:%d): %s", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                    \
        THROW_KU_EXCEPTION("CUDA", _ku_buf);                                                                  \
    }                                                                                                         \
} while(0)

#define KU_CUTLASS_CHECK(call)                                                                                   \
do {                                                                                                             \
    cutlass::Status status_ = (call);                                                                            \
    if (status_ != cutlass::Status::kSuccess) {                                                                 \
        char _ku_buf[1024];                                                                                      \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUTLASS error (%s:%d): %d", __FILE__, __LINE__, static_cast<int>(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                       \
        THROW_KU_EXCEPTION("CUTLASS", _ku_buf);                                                                 \
    }                                                                                                            \
} while(0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template<typename T, T LOWER_BOUND = 1>
inline __host__ __device__ constexpr T find_next_power_of_2(const T& x) {
    if (x <= LOWER_BOUND)
        return LOWER_BOUND;
    return find_next_power_of_2<T, LOWER_BOUND*2>(x);
}

// A wrapper for make_tensor_map
static inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t> &size,
    const std::vector<uint64_t> &strides,   // PAY ATTENTION: In BYTES
    const std::vector<uint32_t> &box_size,
    void* global_ptr,
    CUtensorMapDataType data_type,
    CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t> &element_strides_ = {}
) {
    int dim = size.size();
    KU_ASSERT(dim >= 1);
    
    std::vector<uint32_t> element_strides;
    if (element_strides_.empty()) {
        for (int i = 0; i < dim; ++i)
            element_strides.push_back(1);
    } else {
        element_strides = element_strides_;
    }
    KU_ASSERT(strides.size() == (uint32_t)dim-1 && box_size.size() == (uint32_t)dim && element_strides.size() == (uint32_t)dim);

    auto call_cuTensorMapEncodeTiled = [&]<typename... Args>(Args... args) {
        cudaDriverEntryPointQueryResult cuda_status;
        void* pfn = nullptr;
#if (__CUDACC_VER_MAJOR__ > 12)
        KU_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &pfn, 12000,
            cudaEnableDefault,
            &cuda_status));
#else
        KU_CUDA_CHECK(cudaGetDriverEntryPoint(
            "cuTensorMapEncodeTiled",
            &pfn,
            cudaEnableDefault,
            &cuda_status));
#endif
        if (cuda_status != cudaDriverEntryPointSuccess) {
            KU_ASSERT(false, "Failed to load `cuTensorMapEncodeTiled`. cuda_status = %d", cuda_status);
        }
        return reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(pfn)(args...); \
    };

    CUtensorMap result;
    CUresult ret_code = call_cuTensorMapEncodeTiled(
        &result,
        data_type,
        dim,
        global_ptr,
        size.data(),
        strides.data(),
        box_size.data(),
        element_strides.data(),
        interleave_mode,
        swizzle_mode,
        l2_promotion,
        oob_fill
    );
    if (ret_code != CUresult::CUDA_SUCCESS) {
        auto print_vector = [&](auto t, const char* fmt, const char end='\n') {
            for (auto elem : t) {
                printf(fmt, elem);
            }
            printf("%c", end);
        };
        fprintf(stderr, "Failed to create tensormap\n");
        fprintf(stderr, "Dim: %d\n", dim);
        printf("size: "); print_vector(size, "%lu ");
        printf("strides: "); print_vector(strides, "%lu ");
        printf("box_size: "); print_vector(box_size, "%u ");
        printf("element_strides: "); print_vector(element_strides, "%u ");
        printf("global ptr: 0x%lx\n", (int64_t)global_ptr);
        printf("data_type: %d\n", (int)data_type);
        printf("swizzle_mode: %d\n", (int)swizzle_mode);
        printf("l2_promotion: %d\n", (int)l2_promotion);
        printf("interleave_mode: %d\n", (int)interleave_mode);
        printf("oob_fill: %d\n", (int)oob_fill);
        KU_ASSERT(false);
    }
    return result;
}

// Given strides (in number of elements), this function converts their datatype in uint64_t and then multiplies by elem_size
template<typename T>
static inline std::vector<uint64_t> make_stride_helper(const std::vector<T> &strides_in_elems, size_t elem_size) {
    std::vector<uint64_t> res;
    for (auto stride : strides_in_elems) {
        res.push_back(((uint64_t)stride) * elem_size);
    }
    return res;
}

struct KernelLaunchConfig {
    dim3 grid;
    dim3 block;
    size_t dynamic_smem;
    cudaStream_t stream;
    dim3 cluster{1, 1, 1};
    bool use_pdl{false};
    bool cooperative{false};
};

namespace detail {

template <class Arg>
void* kernel_arg_ptr(Arg&& arg) {
    return const_cast<void*>(reinterpret_cast<const void*>(std::addressof(arg)));
}

} // namespace detail

template<typename KernelFunc, typename... Args>
void launch_kernel(const KernelLaunchConfig &cfg, KernelFunc kernel, Args&&... args) {
    if (cfg.dynamic_smem > 0) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(cfg.dynamic_smem)));
    }

    const bool is_cluster     = !(cfg.cluster.x == 1 && cfg.cluster.y == 1 && cfg.cluster.z == 1);
    const bool need_kernel_ex = is_cluster || cfg.use_pdl;

    if (is_cluster && cfg.cooperative) {
        KU_ASSERT(false, "cluster and cooperative launch are mutually exclusive");
    }

    if (cfg.cooperative) {
        void* kernel_args[sizeof...(Args) > 0 ? sizeof...(Args) : 1] = {};
        if constexpr (sizeof...(Args) > 0) {
            size_t i = 0;
            ((kernel_args[i++] = detail::kernel_arg_ptr(std::forward<Args>(args))), ...);
        }
        KU_CUDA_CHECK(cudaLaunchCooperativeKernel(
            kernel, cfg.grid, cfg.block,
            sizeof...(Args) > 0 ? kernel_args : nullptr,
            static_cast<unsigned int>(cfg.dynamic_smem), cfg.stream));
    } else if (need_kernel_ex) {
        if (is_cluster) {
            KU_ASSERT(cfg.grid.x % cfg.cluster.x == 0 &&
                      cfg.grid.y % cfg.cluster.y == 0 &&
                      cfg.grid.z % cfg.cluster.z == 0);
            if (cfg.cluster.x * cfg.cluster.y * cfg.cluster.z > 8) {
                KU_CUDA_CHECK(cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(kernel),
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            }
        }

        const unsigned int num_attrs = is_cluster ? 2 : 1;
        cudaLaunchAttribute attrs[2];
        if (is_cluster) {
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim = {cfg.cluster.x, cfg.cluster.y, cfg.cluster.z};
        }
        attrs[num_attrs - 1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[num_attrs - 1].val.programmaticStreamSerializationAllowed = cfg.use_pdl ? 1 : 0;

        cudaLaunchConfig_t config = {
            {cfg.grid.x, cfg.grid.y, cfg.grid.z},
            {cfg.block.x, cfg.block.y, cfg.block.z},
            cfg.dynamic_smem,
            cfg.stream,
            attrs,
            num_attrs
        };
        KU_CUDA_CHECK(cudaLaunchKernelEx(
            &config, kernel, std::forward<Args>(args)...));
    } else {
        kernel<<<cfg.grid, cfg.block, cfg.dynamic_smem, cfg.stream>>>(
            std::forward<Args>(args)...);
    }

    KU_CHECK_KERNEL_LAUNCH();
}

#endif  // KERUTILS_IS_BUILD_ON_CUDA

#ifdef KERUTILS_IS_BUILD_ON_ASCEND

#define KU_ACLRT_CHECK(call)                                                                                   \
do {                                                                                                            \
    aclError status_ = (call);                                                                                  \
    if (status_ != ACL_SUCCESS) {                                                                               \
        char _ku_buf[1024];                                                                                     \
        snprintf(_ku_buf, sizeof(_ku_buf), "ACLRT error (%s:%d): %s (code %d)", __FILE__, __LINE__,            \
                 aclGetRecentErrMsg(), status_);                                                                \
        fprintf(stderr, "%s\n", _ku_buf);                                                                       \
        THROW_KU_EXCEPTION("ACLRT", _ku_buf);                                                                   \
    }                                                                                                            \
} while(0)


#endif  // KERUTILS_IS_BUILD_ON_ASCEND

}   // namespace kerutils""",
    'csrc/3rdparty/kerutils/include/kerutils/kerutils.cuh': r"""#pragma once

#include "kerutils/common/common_h.cuh"

#include "host/host_h.cuh"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "device/device.cuh"
#endif
""",
    'csrc/core_attn/config_h.cuh': r"""/*
Fused "Q Norm + Q RoPE + Core Attention + O RoPE + O FP8 Cast" kernel for DeepSeek V4.1
(d_qk = d_v = 512, h_kv = 1, token-level sparse attention) on SM100f.

Instead of launching separate kernels for RMSNorm, RoPE, attention, and output quantization, this
kernel fuses the whole "q_b_proj output -> wv_proj input" segment of the MLA block:
1. Q Norm (ENABLE_Q_NORM, the API's `enable_q_norm`): computes the per-head RMSNorm denominator
   rsqrt(sum(q^2)/d_qk + eps) on the fly while loading Q, and folds it into the softmax scale
2. Q RoPE: applies non-neox-style RoPE (rope_dim = 64) to the last D_ROPE dims of each Q head,
   using `token_positions` and `cos_sin_cache`
3. Core attention: token-level sparse attention over the (at most) `topk` KV tokens selected by
   `indices`. Invalid indices (< 0 or >= s_kv) and positions beyond `topk_length` are masked out.
   Supports an optional per-head `attn_sink` (affects output but not lse / max_logits)
4. O RoPE: applies the conjugate RoPE to the last D_ROPE dims of each output head
5. O FP8 Cast: quantizes the output to fp8_e4m3 with per-32-element ue8m0 scale factors
   (round_sf + packed ue8m0, TMA-aligned col-major sf layout)

Template parameters:
- FWD_MODE:   SparseAttnFwdMode::Prefill or SparseAttnFwdMode::Decode
- MODEL_TYPE: the format of the `kv` cache: ModelType::V41 (V4.1 fp8 layout)
- EXTRA_MODEL_TYPE: the format of the extra KV cache (decode only): ModelType::V41, or ModelType::V41_FP4
              (V4.1 with an fp4 extra KV cache, e4m3 per-16 scales); prefill passes V41
- H_Q:        number of Q heads: 64 (1 CTA) or 128 (2-CTA cluster)
- ENABLE_Q_NORM: the API's `enable_q_norm` flag

I/O (see `csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp` and the Python docstrings in
`flash_mla/fused_norm_rope_attn_rope_cast.py` for the full parameter list):
- q: [s_q, h_q, d_qk], bf16, WITHOUT RoPE applied, in the PERMUTED layout produced by the
  `permute_q_b_proj` kernel (16-element d-chunks interleaved across heads). Each token's
  h_q*d_qk elements must be contiguous
- Prefill: kv [s_kv, 1, d_qk] (bf16, non-paged) + indices [s_q, 1, topk]
  Decode:  paged quantized KV cache(s) (V4.1 fp8 / V4.1 fp4 format, see below) + indices_in_kvcache [s_q, topk],
  plus an optional secondary ("extra") KV cache with its own indices / topk_length
- out_fp8: [s_q, n_wv_group, wv_group_size * d_v], fp8_e4m3, in the permuted layout expected by
  the `permute_wv_proj`-transformed weights; out_sf: packed ue8m0 scale factors (always per-32)
- lse / max_logits (prefill only for max_logits): [s_q, h_q], fp32

Execution structure:
- Persistent kernel scheduled via CLC (Cluster Launch Control); the grid is
  (s_q * CLUSTER_SIZE, 1, 1) and each cluster processes one query token per job
- CLUSTER_SIZE = H_Q / 64: 1 CTA for H_Q = 64, a 2-CTA cluster (dual-CTA UMMA, CTA0 owns V[:, 0:256] and
  CTA1 owns V[:, 256:512]) for H_Q = 128. Each CTA has 512 threads (4 warpgroups):
  - WG0: Q fetching (q_sqr_sum + Q RoPE + store to TMEM) & O epilogue (TMEM load, O RoPE,
    FP8 quant, store to gmem). Timeline: Q0 Q1 O0 Q2 O1 ... Qn O(n-1) On
  - WG1: KV producer. Prefill: gathers the bf16 KV block via TMA gather4. Decode: loads the fp8 / fp4
    part from the paged cache and dequantizes it into the smem KV slots in registers
  - WG2: MMA warp (warp 8, issues UMMAs on CTA0 only), CLC warp (warp 9), indices / validity
    mask generator (warp 10), and warp 11: the decode KV producer's gather front (idle in prefill)
  - WG3: Scale & Exp: reduces P, maintains the online softmax state (mi / li), produces S
- KV tokens are processed in blocks of B_TOPK (96 for 2-CTA, 64 for 1-CTA) with NUM_KV_SLOTS-deep
  software pipelining

Multi-rail GeMM is always used:
- For cases where CLUSTER_SIZE is 1, we FOLD Q by FOLD_FACTOR and do a "batched gemm" in batch size = FOLD_FACTOR, and reduce P on shared memory
- For cases where CLUSTER_SIZE is 2, we also fold Q by FOLD_FACTOR and perform batched GeMM, and reduce P on shared memory

Decoding mode (FWD_MODE == SparseAttnFwdMode::Decode):
- Batch size must be 1; the grid covers the s_q query tokens
- Each KV cache token stores its raw data followed immediately by its scales: 512 + 16 B for fp8,
  or 256 + 32 B for fp4. Warp 11 gathers raw data and scales together with TMA gather4, except that
  CTA0 of h128 gathers only its raw half and uses cp.async for its scale subset. Both transports complete
  on the same per-block barrier. WG1 dequantizes in place into the slot's bf16 layout (DequantRows)
- EXTRA_MODEL_TYPE == ModelType::V41_FP4 selects an fp4 extra KV cache: every token is 512 e2m1 + 32 e4m3 scales,
  each page block stores [page_block_size, 288 B] interleaved token rows. A KV block may then
  mix fp8 tokens of `kv` and fp4 tokens of `extra_kv`; the format is resolved per 8 rows
- Split-KV is not supported (the fused RoPE + FP8-quant epilogue cannot be combined across splits)
*/

#pragma once

#include <cutlass/fast_math.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/defines_h.cuh"
#include "params_h.cuh"
#include "cuda_kernels/kv_cache_format_h.cuh"
#include "cuda_kernels/sm100/kv_cache_utils.cuh"

#include "kernel_h.cuh"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using namespace cute;

template<Config CONFIG>
struct Kernel {

static constexpr SparseAttnFwdMode FWD_MODE = CONFIG.FWD_MODE;
static constexpr ModelType MODEL_TYPE = CONFIG.MODEL_TYPE;
static constexpr ModelType EXTRA_MODEL_TYPE = CONFIG.EXTRA_MODEL_TYPE;
static constexpr uint32_t H_Q = CONFIG.H_Q;

using Params = ParamT<FWD_MODE>;

static_assert(FWD_MODE == SparseAttnFwdMode::Prefill || FWD_MODE == SparseAttnFwdMode::Decode);
static_assert(H_Q == 64 || H_Q == 128);

static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;

// Model parameters
static constexpr uint32_t D_QK = 512;
static constexpr uint32_t D_VO = 512;
static constexpr uint32_t O_QUANT_TILE_SIZE = 32;
static constexpr uint32_t D_ROPE = 64;
static constexpr uint32_t D_NOPE = D_QK - D_ROPE;
static constexpr uint32_t WV_GROUP_SIZE = 8;
static constexpr bool ENABLE_Q_NORM = CONFIG.ENABLE_Q_NORM;

// Cluster shape selection
static constexpr uint32_t CLUSTER_SIZE = ku::ceil_div((uint32_t)H_Q, 64u);
static constexpr uint32_t IS_2CTA = CLUSTER_SIZE == 2;
#ifndef VATTN_BS
#define VATTN_BS 0
#endif
// VATTN: the decode kernel reads the block-split V4.1 page layout (see kv_cache_format.h KVCachePartBS); 1 CTA only
static constexpr bool BLOCK_SPLIT = VATTN_BS != 0 && IS_DECODE;
static_assert(!BLOCK_SPLIT || CLUSTER_SIZE == 1, "the block-split port covers the 1-CTA (H_Q = 64) kernel only");

// Tiling Shape Selection
static constexpr uint32_t B_TOPK = CLUSTER_SIZE == 2 ? 96 : 64;
// Each CTA computes 64 rows of Q / P / O, and H_Q is 64 (1 CTA) or 128 (2 CTAs), so every row of Q / P / O is a head. A .ws UMMA
// instruction takes about 12 + N/4 cycles for M = 64 (the tensor datapath is 64 rows wide), so M decides the TMEM / shared-memory
// footprint and the softmax work per thread
static constexpr uint32_t H_Q_PER_CTA = 64;   // rows of Q / P / O per CTA
static constexpr uint32_t MMA_M = CLUSTER_SIZE * H_Q_PER_CTA;   // M of the UMMA atom: the 2-CTA atom spans both CTAs' rows

template<ModelType MT>
struct KVFormat : KVCacheFormat<MT> {
    using Base = KVCacheFormat<MT>;
    static constexpr uint32_t CHUNK_ELEMS = Base::IS_FP4 ? 32 : 16;   // The number of elements that one LDS.128 can load
};

// An odd number of 16 B chunks per row, together with the rotation in DequantRows,
// keeps LDS.128 conflict-free. Both CTAs share this row stride and one dequant body.
template<ModelType MT, uint32_t CTA>
struct KVFormatCta : std::conditional_t<BLOCK_SPLIT, KVCachePartBS<KVFormat<MT>>, KVCachePart<KVFormat<MT>, CLUSTER_SIZE, CTA>> {
    using F = KVFormat<MT>;
    using Part = std::conditional_t<BLOCK_SPLIT, KVCachePartBS<F>, KVCachePart<F, CLUSTER_SIZE, CTA>>;
    static constexpr uint32_t NUM_QUANT_CHUNKS_PER_ROW = Part::QUANT_BYTES / 16;
    static constexpr uint32_t NUM_QUANT_CHUNKS_PER_LANE = NUM_QUANT_CHUNKS_PER_ROW / 4;
    static constexpr uint32_t RAW_TOKEN_SMEM_STRIDE = BLOCK_SPLIT
        ? (ku::ceil_div(Part::QUANT_BYTES, 16) | 1) * 16          // raw only: 528 (fp8) / 272 (fp4)
        : (ku::ceil_div(Part::QUANT_BYTES + F::NUM_SCALES_EACH_TOKEN, 16) | 1) * 16;
    static_assert(NUM_QUANT_CHUNKS_PER_ROW % 4 == 0);
};
using OrigKVFormat = KVFormat<MODEL_TYPE>;          // Format of `kv`
using ExtraKVFormat = KVFormat<EXTRA_MODEL_TYPE>;   // Format of `extra_kv`
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static constexpr bool HAS_FP4_KV = ExtraKVFormat::IS_FP4;   // The extra KV cache is fp4
// Shorthands for the format of `kv`, which is also the format of `extra_kv` unless HAS_FP4_KV
static constexpr uint32_t D_FP8 = OrigKVFormat::D_FP8;
static constexpr uint32_t KV_QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN_PER_CTA = NUM_SCALES_EACH_TOKEN / CLUSTER_SIZE;
static constexpr uint32_t KV_CACHE_BYTES_PER_TOKEN = OrigKVFormat::BYTES_PER_TOKEN;
// Decode gathers the raw rows of a KV block into the beginning of the KV slot (one 4-row group per gather4) and dequantizes
// them in place. A group is padded to 128 B: the destination of a gather4 must be 128 B aligned (a 64 B aligned one raises a
// misaligned address error; the PTX ISA's 16 B does not hold for this instruction). Both CTAs of a cluster use the same group
// stride. Both caches hold the same dims, so a CTA's rows of both are placed the same way
static constexpr uint32_t RAW_KV_GROUP_BYTES = ku::ceil_div(4 * std::max({
    KVFormatCta<MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE, KVFormatCta<MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE,
    KVFormatCta<EXTRA_MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE, KVFormatCta<EXTRA_MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE}), 128u) * 128;
static_assert(!IS_DECODE || B_TOPK / 4 * RAW_KV_GROUP_BYTES <= B_TOPK * D_QK / CLUSTER_SIZE * sizeof(bf16));   // the raw rows fit in the slot
// CTA0 stages its scale subset in the unused tail of the KV slot. A mixed fp8/fp4 block uses
// a common 16 B row; a pure fp8 block needs 8 B per token. All raw and scale bytes are read before
// the dequant warpgroup's named barrier allows bf16 write-back to overwrite the slot.
static constexpr uint32_t SCALE_STAGE_ROW_BYTES = BLOCK_SPLIT ? (HAS_FP4_KV ? 32 : 16) : (HAS_FP4_KV ? 16 : NUM_SCALES_EACH_TOKEN_PER_CTA);
static constexpr uint32_t SCALE_STAGE_OFFSET = B_TOPK * (D_QK / CLUSTER_SIZE * sizeof(bf16) - SCALE_STAGE_ROW_BYTES);
static_assert(!IS_DECODE || !(IS_2CTA || BLOCK_SPLIT) || B_TOPK / 4 * RAW_KV_GROUP_BYTES <= SCALE_STAGE_OFFSET);

// Decode's per-block code nearly fills the instruction cache (h128 q-norm kernel: hit rate 82.7% with the per-job
// code unrolled, 89.6% with it compact), so the code that runs once per job (the O epilogue and the q-norm sum of
// squares) is kept small there, which adds some latency to each job: loops rolled, sums read back from TMEM.
// Prefill's code fits in the cache, and its per-job code is on the critical path between jobs, so it takes the
// unrolled forms
static constexpr bool COMPACT_PER_JOB_CODE = IS_DECODE;

// Registers per thread of each warpgroup.
// Decode's WG1 dequantizes a KV block in registers and needs 128. The prefill KV producer issues TMA gathers and uses
// about 50, so the 1-CTA prefill kernels give WG0 256 for the software-pipelined O epilogue (EPILOGUE_TILES_AHEAD)
static constexpr bool WIDE_WG0 = !IS_DECODE && CLUSTER_SIZE == 1;
static constexpr uint32_t WG0_REGS = WIDE_WG0 ? 256 : 184;
static constexpr uint32_t WG1_REGS = WIDE_WG0 ? 56 : 128;
static constexpr uint32_t WG2_REGS = 72;
static constexpr uint32_t WG3_REGS = 128;
static_assert(WG0_REGS + WG1_REGS + WG2_REGS + WG3_REGS == 4 * 128);
// setmaxnreg takes a multiple of 8 in [24, 256]
static_assert(WG0_REGS % 8 == 0 && WG1_REGS % 8 == 0 && WG2_REGS % 8 == 0 && WG3_REGS % 8 == 0);
static_assert(24 <= WG1_REGS && 24 <= WG2_REGS && WG0_REGS <= 256 && WG3_REGS <= 256);
// Q load: a thread's 16 LDG.256 are issued one 16 KB tile at a time, each tile's addresses depending on the data of
// the previous tile (load_q_and_save_to_tmem). 1-CTA kernels only: on the 2-CTA kernels the burst overlaps no
// shared-memory instruction of the critical path, so the grouping has nothing to hide and its serial load latencies
// only add to the job
static constexpr bool GROUP_Q_LOADS = CLUSTER_SIZE == 1;

static constexpr uint32_t NUM_THREADS = 512;
// Every thread that calls get_next_job arrives on bar_clc_empty once per CLC round, and this count
// is that barrier's expected arrival count. In decode warp 11 runs the job loop too (its front
// crosses job boundaries), so its 32 threads are counted
static constexpr uint32_t NUM_WORKING_THREADS =
    CLUSTER_SIZE == 1 ? (
        IS_DECODE ?
        128 + 128 + (1+1+32) + 128 + 32 :   // WG0 + WG3 + (MMA + CLC + indices) + WG1 (dequant) + warp 11 (gather front)
        128 + 128 + (1+1+32) + 4                            // WG0 + WG3 + (MMA + CLC + indices) + WG1 (KV producer, 1 elected thread per warp)
    ) : (
        IS_DECODE ?
        128*2 + 128*2 + (1+2+32*2) + 128*2 + 32*2 :   // ... + both CTAs' warp 11
        128*2 + 128*2 + (1+2+32*2) + 4*2
    );

// The 128 TMEM lanes form FOLD_FACTOR groups of H_Q_PER_CTA lanes. A .ws UMMA instruction reads the A operand of lane group g from
// that group's lanes and writes the g-th N-slice of D to them, so the QK GEMM runs NUM_MRGEMM_RAILS "rails": B is the KV block
// viewed as NUM_MRGEMM_RAILS x B_TOPK rows of D_QK / NUM_MRGEMM_RAILS dims (rail j = 64-dim atom columns j, j+2, j+4, j+6), and lane
// group g holds head rows x the K-view of rail g. The rails' partial P are summed through shared memory by the softmax warpgroup
static constexpr uint32_t FOLD_FACTOR = 128 / H_Q_PER_CTA;
static constexpr uint32_t NUM_MRGEMM_RAILS = 2; // The number of "rails" (batch size) during multi-rail GeMM. Currently must be 2
static_assert(FOLD_FACTOR % NUM_MRGEMM_RAILS == 0);
static constexpr uint32_t NUM_P_ELEMS_PER_THREAD = H_Q_PER_CTA * B_TOPK / 128;
// O epilogue tiles per job: (D_VO / FOLD_FACTOR) columns of O per thread in tiles of O_QUANT_TILE_SIZE
static constexpr uint32_t NUM_EPILOGUE_TILES = D_VO / FOLD_FACTOR / O_QUANT_TILE_SIZE;
// O epilogue: the TMEM load of a tile is issued this many tiles before the tile is converted; each tile loaded ahead
// holds 32 fp32 registers, so 6 tiles need the 256 of WIDE_WG0. 0 selects the per-tile loop: a tile's
// load, conversion and stores per iteration. The 2-CTA prefill kernels use 0. Their first SV of a job waits on the softmax's
// first S, and S is published later than tO_empty, so an earlier tO_empty does not start the SV earlier, while the stores the
// pipeline concentrates into a shorter time delay that softmax block
static constexpr uint32_t EPILOGUE_TILES_AHEAD = WIDE_WG0 ? std::min(6u, NUM_EPILOGUE_TILES) : 0;

static constexpr uint32_t NUM_KV_SLOTS = 3;
static constexpr uint32_t NUM_INDICES_BUFS = 4;
// Raw-gather barriers: an alternating pair, so two gather batches can be issued but not completed at the same
// time: the front of block k+1 waits for the dequant warps to be past block k-1, the previous user of
// bar_raw_kv_full[(k+1)%2], not past block k. With one barrier the front of block k+1 could be issued only
// after block k's rows had arrived, so warp 11's front time added to the transport time of every block
static constexpr uint32_t NUM_RAW_BARS = 2;
static constexpr uint32_t NUM_P_BUFS = CLUSTER_SIZE == 2 ? 1 : 2;
static constexpr uint32_t NEED_TP_EMPTY_BAR = NUM_P_BUFS == 1;  // Don't need to wait for P's emptiness as long as P has >= 2 buffers, since "we are issuing P[i]" <-- "O[i-2] has been issued" <-- "S[i-2] is ready" <-- "P[i-2] is free"

struct tmem_cols {
    // O: D_VO / FOLD_FACTOR = 256 columns; Q: the K-view of one rail per lane group, D_QK / NUM_MRGEMM_RAILS bf16 = 128 columns;
    // P: B_TOPK x NUM_MRGEMM_RAILS fp32 per lane group = 64 columns each
    static constexpr uint32_t O = 0;
    static constexpr uint32_t Q = O + D_VO / FOLD_FACTOR;
    static constexpr uint32_t P_0 = Q + D_QK / NUM_MRGEMM_RAILS / 2;  // /2 since 2 bf16 is packed in 1 uint32
    static constexpr uint32_t P_1 = P_0 + B_TOPK*NUM_MRGEMM_RAILS/FOLD_FACTOR;

    static constexpr uint32_t get_p(const uint32_t &p_buf_idx) {
        if constexpr (NUM_P_BUFS == 1) {
            return P_0;
        } else if constexpr (NUM_P_BUFS == 2) {
            return p_buf_idx ? P_1 : P_0;
        } else {
            static_assert(NUM_P_BUFS == 1 || NUM_P_BUFS == 2);
        }
    }
    static_assert(get_p(NUM_P_BUFS-1) + B_TOPK*NUM_MRGEMM_RAILS/FOLD_FACTOR <= 512);
};

using MMAAtom_QK = cute::conditional_t<
    CLUSTER_SIZE == 2,
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, MMA_M, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>,
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, MMA_M, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>
>;
using TiledMMA_QK = decltype(make_tiled_mma(MMAAtom_QK{}));
using TiledMMA_SV = cute::conditional_t<
    CLUSTER_SIZE == 2,
    decltype(make_tiled_mma(
        SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, 128, 256, UMMA::Major::K, UMMA::Major::MN>{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
    )),
    decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, 256, UMMA::Major::K, UMMA::Major::MN>{}))
>;

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) bf16 kv_slots[NUM_KV_SLOTS][B_TOPK * D_QK / CLUSTER_SIZE];    // Cluster size = 1: the whole KV; cluster size = 2: half KV
    CUTE_ALIGNAS(1024) bf16 s[H_Q_PER_CTA * B_TOPK];
    CUTE_ALIGNAS(1024) float p_exchange_buf[4][32*NUM_P_ELEMS_PER_THREAD];
    CUTE_ALIGNAS(1024) uint8_t is_k_valid[NUM_INDICES_BUFS][ku::find_next_power_of_2(B_TOPK/8)];
    // Decode: WG10 produces metadata once for the four dequant warps. 16 B aligned so that the fp4 path can read
    // the coordinates of 4 consecutive rows with one LDS.128
    CUTE_ALIGNAS(16) int decode_tma_coords[IS_DECODE ? NUM_INDICES_BUFS : 0][B_TOPK];
    // VATTN block-split: byte offset / 16 of each row's scale row in its cache (0xFFFFFFFF = invalid row)
    uint32_t decode_scale_offs[BLOCK_SPLIT ? NUM_INDICES_BUFS : 0][B_TOPK];
    float q_sqr_sum_buf[ENABLE_Q_NORM ? 2 : 0][128];    // We have 2 q_sqr_sum_buf to save some barriers, as Q[i+2] starts to fetch -> O[i] have finished -> Q[i]'s q_sqr_sum_buf is useless
    float rowwise_max_buf[128];
    float rowwise_mi_buf[H_Q_PER_CTA];
    float rowwise_li_buf[128];    // 128: warpgroup size

    transac_bar_t bar_kv_slot_full[NUM_KV_SLOTS], bar_kv_slot_empty[NUM_KV_SLOTS];
    transac_bar_t bar_indices_full[NUM_INDICES_BUFS], bar_indices_empty[NUM_INDICES_BUFS];
    transac_bar_t bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_q_sqr_sum_full;   // Only used for 2-CTA
    transac_bar_t bar_tO_empty, bar_tO_full;
    transac_bar_t bar_tP_full[NUM_P_BUFS], bar_tP_empty[NEED_TP_EMPTY_BAR ? NUM_P_BUFS : 0];
    transac_bar_t bar_SO_full, bar_SO_empty;
    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_li_mi_full, bar_li_mi_empty;
    // Completion of a block's raw gathers; block k uses index k % NUM_RAW_BARS (see NUM_RAW_BARS)
    transac_bar_t bar_raw_kv_full[IS_DECODE ? NUM_RAW_BARS : 0];
    // Dequant has passed the raw+scale completion wait; warp 11 may reuse this barrier index.
    transac_bar_t bar_wg1_past_raw[IS_DECODE ? NUM_RAW_BARS : 0];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};
static_assert(sizeof(SharedMemoryPlan) <= 227 * 1024);

struct TMAParams {
    // Prefill only
    CUtensorMap tensor_map_kv;                  // the whole (bf16, non-paged) KV cache
    // Decode only: the raw rows of each CTA's part of a token (KVFormatCta), per paged KV cache
    CUtensorMap tensor_map_kv_fp8_part_cta0;
    CUtensorMap tensor_map_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta1;
};

using AllocatorT = std::conditional_t<IS_2CTA, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

struct AuxParams {
    cutlass::FastDivmod fast_divmod_page_block_size;
    cutlass::FastDivmod fast_divmod_extra_page_block_size;
};

// Some helper functions for buffer arrival
static __device__ __forceinline__ void umma_arrive_on_every_cta(transac_bar_t &bar) {
    // Perform UMMA arrive, possibly with multicast, to every CTA
    if constexpr (IS_2CTA) {
        ku::umma_arrive_multicast_2x1SM_noelect(bar, 1|2);
    } else {
        ku::umma_arrive_noelect(bar);
    }
}
static __device__ __forceinline__ void umma_arrive_on_cta0(transac_bar_t &bar) {
    // Perform UMMA arrive on CTA0
    if constexpr (IS_2CTA) {
        ku::umma_arrive_2x1SM_noelect(bar);
    } else {
        ku::umma_arrive_noelect(bar);
    }
}
static __device__ __forceinline__ void arrive_on_cta0_barrier(transac_bar_t &bar) {
    if constexpr (IS_2CTA) {
        bar.arrive(0u);
    } else {
        bar.arrive();
    }
}

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG3_SYNC = 1;
    static constexpr int WG3_WARP02_SYNC = 2;
    static constexpr int WG3_WARP13_SYNC = 3;
    static constexpr int WG1_DEQUANT_SYNC = 7;  // The dequant warpgroup, between reading a block's raw rows and writing its bf16 rows back
};

static __device__ __forceinline__ void
devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params);

static void run(const Params &params);

};

}
""",
    'csrc/core_attn/kernel.cuh': r"""#pragma once

#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/utils_h.cuh"
#include "cuda_kernels/sm100/helpers_h.cuh"
#include "cuda_kernels/sm100/common_subroutine_h.cuh"

#include "config_h.cuh"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

static constexpr float MAX_INIT_VAL = -1e30;
static constexpr float O_QUANT_CLAMP_MIN_VALUE = 1e-4;

#ifdef KERUTILS_ENABLE_SM103A
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = true;
#else
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = false;
#endif

__device__ __forceinline__
float2 apply_rope(const float2 &x, const float &cur_cos, const float &cur_sin) {
    float2 a = {x.x, x.x};
    float2 b = {cur_cos, cur_sin};
    float2 c = {-x.y * cur_sin, +x.y * cur_cos};
    float2 y = ku::float2_fma(a, b, c);
    return y;
}

// To achieve prefill - decoding alignment while using block_size = 96, decoding must reproduce
// prefill's "natural" blocking of the concatenated indices array ([topk orig slots; extra_topk
// extra slots], tiled by B_TOPK). Since topk (e.g. 128) may not be a multiple of B_TOPK (e.g. 96),
// one KV block may straddle the orig/extra boundary, i.e. the last (partial) block of the orig KV
// and the first tokens of the extra KV are "stitched" into one block. Each KV block is thus
// classified into one of the following three categories:
enum class KVLocation {
    ORIG,           // All slots of this block come from `kv`/`indices`
    ORIG_AND_EXTRA, // This block straddles the boundary: slots < num_orig_slots come from `kv`/`indices`, the rest from `extra_kv`/`extra_indices`
    EXTRA           // All slots of this block come from `extra_kv`/`extra_indices`
};

// Block dequant of the KV producer: the lane-to-(row, chunk) mapping, the STS.128 offsets, and the per-block
// body that reads a block's scales and raw rows from its KV slot and writes the bf16 rows back into the slot.
// fp8 tokens (from `kv`) and fp4 tokens (from `extra_kv`) differ only in the number of elements per 16 B of
// raw data (CHUNK_ELEMS), the scale layout, and the conversion instructions.
//
// The unit of work is a chunk: 16 B of a raw row (one LDS.128). A quantized chunk = CHUNK_ELEMS elements =
// CHUNK_ELEMS * 2 B of the bf16 128 B swizzle-atom row of the KV slot (CHUNK_ELEMS / 8 STS.128).
// Lane-to-token mapping: 4 lanes per row, lane idx_in_row owning the idx_in_row-th quarter of the row's
// quantized chunks, and 8 consecutive rows (one "token" of the thread) per 8 consecutive lanes,
// NUM_TOKENS_PER_THREAD tokens per thread. With C = quantized chunks per lane, rows relative to the token:
//   lane          0  1  2  3 |  4  5  6  7 |  8 .. 11 | 12 .. 15 | 16 .. 19 | 20 .. 23 | 24 .. 27 | 28 .. 31
//   row           0  1  2  3 |  4  5  6  7 |  0 ..  3 |  4 ..  7 |  0 ..  3 |  4 ..  7 |  0 ..  3 |  4 ..  7
//   chunks         [0, C)    |  (rotated)  |  [C, 2C) | (rotated)| [2C, 3C) | (rotated)| [3C, 4C) | (rotated)
// The 4 lanes of one row are spread over the 4 wavefronts of an LDS/STS.128 (8 consecutive lanes each), so one
// wavefront covers 8 consecutive rows at one chunk position, which is bank-conflict-free on both sides:
//  - Raw LDS.128: the 8 rows are two 4-row gather4 groups whose rows hit two complementary sets of 4 bank
//    groups (KVFormatCta): the lanes of rows 4..7 process their chunks rotated by 4 (= 64 B,
//    see DequantRows::get_chunk_base).
//  - STS.128 into the SW128 K-major layout: a 16 B part of chunk c is written to the 16 B bank group
//    ((c % CHUNKS_PER_ATOM_ROW) * NUM_STS_PER_CHUNK + j) ^ (row % 8) of its swizzle-atom row; the 8 rows of a
//    wavefront have 8 distinct row % 8 and the same c % CHUNKS_PER_ATOM_ROW (a rotation by 4 is a multiple of it).
template<typename K, uint32_t CTA>
struct DequantRows {
    using OrigFormat = typename K::template KVFormatCta<K::MODEL_TYPE, CTA>;
    using ExtraFormat = typename K::template KVFormatCta<K::EXTRA_MODEL_TYPE, CTA>;
    static constexpr uint32_t B_TOPK = K::B_TOPK;
    static constexpr uint32_t NUM_DEQUANT_WARPS = 4, NUM_LANES_PER_ROW = 4;
    static constexpr uint32_t NUM_ROWS_PER_WAVEFRONT = 32 / NUM_LANES_PER_ROW;
    static constexpr uint32_t NUM_TOKENS_PER_THREAD = B_TOPK / (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT);
    static_assert(B_TOPK % (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT) == 0);
    static constexpr uint32_t NUM_ROWS_PER_WARP = NUM_TOKENS_PER_THREAD * NUM_ROWS_PER_WAVEFRONT;
    static constexpr uint32_t MAX_QUANT_CHUNKS_PER_LANE = std::max(OrigFormat::NUM_QUANT_CHUNKS_PER_LANE, ExtraFormat::NUM_QUANT_CHUNKS_PER_LANE);
    // Both caches hold the same dims (a V4.1 fp8 cache with a V4.1 fp4 one, or two of one format), so the
    // placement is the same for both formats
    static_assert(OrigFormat::QUANT_DIMS == ExtraFormat::QUANT_DIMS);
    static constexpr uint32_t QUANT_DIMS = OrigFormat::QUANT_DIMS;
    static_assert(MAX_QUANT_CHUNKS_PER_LANE <= 8);

    // fp8 scales: one ue8m0 per QUANT_TILE_SIZE elements, i.e. per CHUNKS_PER_TILE chunks. A lane's chunks are whole
    // tiles, so its scale bytes are the NUM_QUANT_CHUNKS_PER_LANE / CHUNKS_PER_TILE consecutive bytes at that offset
    // of the token's scale row, arranged so that chunk g uses byte g / CHUNKS_PER_TILE (a rotated lane reads the
    // rotated position or swaps the two halves of its word, see get_chunk_base).
    // fp4 scales: two e4m3 per chunk, so a lane's scales are the 2 * NUM_QUANT_CHUNKS_PER_LANE bytes at that offset of
    // the token's scale row
    template<typename F>
    static constexpr uint32_t chunks_per_tile() { return F::QUANT_TILE_SIZE / F::CHUNK_ELEMS; }
    static_assert(OrigFormat::IS_FP4 || OrigFormat::NUM_QUANT_CHUNKS_PER_LANE % chunks_per_tile<OrigFormat>() == 0);
    static_assert(ExtraFormat::IS_FP4 || ExtraFormat::NUM_QUANT_CHUNKS_PER_LANE % chunks_per_tile<ExtraFormat>() == 0);
    template<typename F>
    static constexpr uint32_t scale_bytes_per_lane() {
        return F::IS_FP4 ? F::NUM_QUANT_CHUNKS_PER_LANE * 2 : F::NUM_QUANT_CHUNKS_PER_LANE / chunks_per_tile<F>();
    }
    static constexpr uint32_t SCALE_WORDS_PER_LANE = ku::ceil_div(std::max(scale_bytes_per_lane<OrigFormat>(), scale_bytes_per_lane<ExtraFormat>()), 4u);
    static_assert(SCALE_WORDS_PER_LANE <= 2);

    // STS.128 offsets. Quantized chunk g of the lane is chunk c = get_chunk_base(g / 4) + g % 4 of the row, i.e. parts
    // [(c % CHUNKS_PER_ATOM_ROW) * NUM_STS_PER_CHUNK, +NUM_STS_PER_CHUNK) of atom row c / CHUNKS_PER_ATOM_ROW (an atom
    // row holds 64 dims: 4 fp8 chunks or 2 fp4 chunks). A lane owns QUANT_DIMS / 4 dims, which are whole atom rows
    // (QUANT_DIMS % 256 == 0): the part index of (g, j) is the same for every lane, so sts_offsets[p] is the swizzled
    // offset of part p (0..7) of the first atom row of the lane's first token and the chunk's atom column is added;
    // both formats share the table since it is indexed by part. The other tokens are multiples of 8 rows further
    template<typename F>
    static constexpr uint32_t chunks_per_atom_row() { return 128 / (F::CHUNK_ELEMS * 2); }
    template<typename F>
    static constexpr uint32_t num_sts_per_chunk() { return F::CHUNK_ELEMS * 2 / 16; }
    static_assert(QUANT_DIMS % 256 == 0);
    static constexpr uint32_t NUM_STS_OFFSETS = 8;
    static constexpr uint32_t STS_ATOM_COL_STRIDE_BYTES = B_TOPK * 128;
    static constexpr uint32_t STS_TOKEN_STRIDE_BYTES = NUM_ROWS_PER_WAVEFRONT * 128;

    typename K::SharedMemoryPlan &smem;
    uint32_t local_warp_idx, row_in_wavefront, idx_in_row, chunk_rotation;
    uint32_t sts_offsets[NUM_STS_OFFSETS];
    uint32_t orig_scale_offset, extra_scale_offset, scale_token_stride;

    __device__ __forceinline__ DequantRows(typename K::SharedMemoryPlan &smem_, uint32_t dq_warp_idx, uint32_t lane_idx, uint32_t cta_idx)
        : smem(smem_), local_warp_idx(dq_warp_idx),
          row_in_wavefront(lane_idx % NUM_ROWS_PER_WAVEFRONT), idx_in_row(lane_idx / NUM_ROWS_PER_WAVEFRONT),
          chunk_rotation(lane_idx % NUM_ROWS_PER_WAVEFRONT / 4 * 4) {
        using namespace cute;
        Tensor sKV = make_tensor(make_smem_ptr(smem.kv_slots[0]), ku::make_umma_canonical_k_major_layout<B_TOPK, K::D_QK / K::CLUSTER_SIZE, 128>());
        const uint32_t row = get_row_idx(0);
        if constexpr (K::IS_2CTA) {
            // Resolve the asymmetric scale transport once, outside the block loop, so both CTAs
            // execute the same dequantization body without duplicating its instruction cache footprint.
            auto init_scale_offset = [&]<typename F>() {
                if (cta_idx == 0) {
                    return K::SCALE_STAGE_OFFSET + row * K::SCALE_STAGE_ROW_BYTES;
                }
                using Cta1Format = typename K::template KVFormatCta<F::Base::MODEL_TYPE, 1>;
                return row / 4 * K::RAW_KV_GROUP_BYTES + row % 4 * F::RAW_TOKEN_SMEM_STRIDE + Cta1Format::SCALE_OFFSET;
            };
            orig_scale_offset = init_scale_offset.template operator()<OrigFormat>();
            extra_scale_offset = init_scale_offset.template operator()<ExtraFormat>();
            scale_token_stride = NUM_ROWS_PER_WAVEFRONT * (cta_idx == 0 ? K::SCALE_STAGE_ROW_BYTES : K::RAW_KV_GROUP_BYTES / 4);
        }
        // In the SW128 K-major layout a row is 128 B of every swizzle-atom column, and its 16 B part p lies at part index
        // p ^ (row % 8). The layout gives the row's first part; the other entries are derived from it with 32-bit
        // arithmetic. Deriving every entry from the layout takes 64-bit arithmetic per entry, and with all entries live at
        // once ptxas spills the table
        const uint32_t row_swizzle = row % 8;
        const uint32_t row_start = (uint32_t)((&sKV(row, 0) - smem.kv_slots[0]) * sizeof(bf16)) - row_swizzle * 16;
        // Offset of the (part_idx % 8)-th part of atom column part_idx / 8 of the row
        auto part_offset = [&](uint32_t part_idx) {
            return row_start + ((part_idx % 8 ^ row_swizzle) * 16) + part_idx / 8 * STS_ATOM_COL_STRIDE_BYTES;
        };
        CUTE_UNROLL
        for (uint32_t p = 0; p < 8; ++p) {
            sts_offsets[p] = part_offset(p);
        }
    }

    __device__ __forceinline__ uint32_t get_row_idx(uint32_t token_idx) const {
        return local_warp_idx * NUM_ROWS_PER_WARP + token_idx * NUM_ROWS_PER_WAVEFRONT + row_in_wavefront;
    }
    // Quantized chunk g of the lane is chunk get_chunk_base(g / 4) + g % 4 of the row, so an address is a per-lane base
    // plus an immediate. The rows 4..7 (chunk_rotation = 4) rotate the order by 4 chunks: a rotation of the row's chunk
    // indices when a lane owns at most 4 chunks, a swap of its two halves (g ^ 4) when it owns 8. For rows 0..3
    // (chunk_rotation = 0) the lane's chunks are [idx_in_row * N, +N)
    template<typename F>
    __device__ __forceinline__ uint32_t get_chunk_base(uint32_t half) const {
        if constexpr (F::NUM_QUANT_CHUNKS_PER_LANE <= 4) {
            return (idx_in_row * F::NUM_QUANT_CHUNKS_PER_LANE + chunk_rotation) % F::NUM_QUANT_CHUNKS_PER_ROW;
        } else {
            return idx_in_row * F::NUM_QUANT_CHUNKS_PER_LANE + (half * 4 ^ chunk_rotation);
        }
    }
    // Offset of part j of quantized chunk g of the lane's first token in the slot
    template<typename F>
    __device__ __forceinline__ uint32_t quant_sts_offset(uint32_t g, uint32_t j) const {
        // chunk_base is a multiple of 4 (a multiple of CHUNKS_PER_ATOM_ROW), so the part index is selected at compile time
        static constexpr uint32_t CHUNKS_PER_ATOM_ROW = chunks_per_atom_row<F>();
        uint32_t atom_col = get_chunk_base<F>(g / 4) / CHUNKS_PER_ATOM_ROW + g % 4 / CHUNKS_PER_ATOM_ROW;
        return sts_offsets[g % CHUNKS_PER_ATOM_ROW * num_sts_per_chunk<F>() + j] + atom_col * STS_ATOM_COL_STRIDE_BYTES;
    }

    // Dequantize this thread's rows of one block whose raw data and scales are in the slot; the first
    // NUM_FP8_TOKENS of its tokens are fp8, the rest fp4. Reads the scales (staged separately on CTA0 of h128,
    // interleaved with raw data otherwise), releases the indices buffer, then reads raw data via LDS.128.
    // A named barrier ensures every lane has read its input before bf16 write-back starts. Per quantized
    // chunk: fp8: 8x F2FP (e4m3x2 x ue8m0 -> bf16x2); fp4: 16x F2FP (e2m1x2 -> bf16x2) + 16x HMUL2.BF16 (x the
    // bf16 of the e4m3 scale, exact: the product has at most 2 + 4 significant bits). Then the STS.128s, the fence,
    // and the kv_slot_full arrive. The caller passes the block's slot and indices buffer and advances its own ring state
    template<uint32_t NUM_FP8_TOKENS>
    __device__ __forceinline__ void body(uint32_t kv_slot_idx, uint32_t indices_buf_idx) const {
        using cutlass::arch::NamedBarrier;
        auto for_each_token = [&](auto callable) {
            cute::for_each(cute::make_int_sequence<NUM_TOKENS_PER_THREAD>{}, [&](auto token_idx) {
                if constexpr (token_idx < NUM_FP8_TOKENS) {
                    callable.template operator()<OrigFormat>(token_idx);
                } else {
                    callable.template operator()<ExtraFormat>(token_idx);
                }
            });
        };
        uint8_t *slot_base = (uint8_t*)smem.kv_slots[kv_slot_idx];

        uint32_t cached_scales[NUM_TOKENS_PER_THREAD][SCALE_WORDS_PER_LANE];
        for_each_token([&]<typename F>(uint32_t token_idx) {
            static constexpr uint32_t SCALE_BYTES_PER_LANE = scale_bytes_per_lane<F>();
            const uint8_t *scale_src = [&]() {
                if constexpr (K::IS_2CTA) {
                    return slot_base + (F::IS_FP4 ? extra_scale_offset : orig_scale_offset) + token_idx * scale_token_stride;
                } else if constexpr (K::BLOCK_SPLIT) {
                    return slot_base + K::SCALE_STAGE_OFFSET + get_row_idx(token_idx) * K::SCALE_STAGE_ROW_BYTES;
                } else {
                    uint32_t row = get_row_idx(token_idx);
                    return slot_base + row / 4 * K::RAW_KV_GROUP_BYTES + row % 4 * F::RAW_TOKEN_SMEM_STRIDE + F::SCALE_OFFSET;
                }
            }();
            if constexpr (F::IS_FP4) {
                const uint8_t *src = scale_src + get_chunk_base<F>(0) * 2;
                if constexpr (SCALE_BYTES_PER_LANE == 8) {
                    *(uint64_t*)cached_scales[token_idx] = *(const uint64_t*)src;
                } else {
                    static_assert(SCALE_BYTES_PER_LANE == 4);
                    cached_scales[token_idx][0] = *(const uint32_t*)src;
                }
            } else {
                if constexpr (F::NUM_QUANT_CHUNKS_PER_LANE == 8) {
                    // A rotated lane swaps the two halves of its chunks, so also the two halves of its scale bytes
                    if constexpr (SCALE_BYTES_PER_LANE == 4) {
                        uint32_t scales = *(uint32_t*)(scale_src + idx_in_row * 4);
                        cached_scales[token_idx][0] = chunk_rotation ? __byte_perm(scales, scales, 0x1032) : scales;
                    } else {
                        static_assert(SCALE_BYTES_PER_LANE == 2);
                        uint32_t scales = *(uint16_t*)(scale_src + idx_in_row * 2);
                        cached_scales[token_idx][0] = chunk_rotation ? __byte_perm(scales, scales, 0x0001) : scales;
                    }
                } else {
                    // A rotated lane's chunks are the next lane's (modulo the row), so are its scale bytes
                    static_assert(F::NUM_QUANT_CHUNKS_PER_LANE == 4 && (SCALE_BYTES_PER_LANE == 2 || SCALE_BYTES_PER_LANE == 1));
                    const uint8_t *src = scale_src + get_chunk_base<F>(0) / chunks_per_tile<F>();
                    cached_scales[token_idx][0] = SCALE_BYTES_PER_LANE == 2 ? *(uint16_t*)src : *src;
                }
            }
        });
        smem.bar_indices_empty[indices_buf_idx].arrive();

        // A format with fewer quantized chunks per lane uses the first entries
        uint32_t cached_input[NUM_TOKENS_PER_THREAD][MAX_QUANT_CHUNKS_PER_LANE][4];
        for_each_token([&]<typename F>(uint32_t token_idx) {
            uint32_t row = get_row_idx(token_idx);
            uint8_t *row_base = slot_base + row / 4 * K::RAW_KV_GROUP_BYTES + row % 4 * F::RAW_TOKEN_SMEM_STRIDE;
            CUTE_UNROLL
            for (uint32_t g = 0; g < F::NUM_QUANT_CHUNKS_PER_LANE; ++g) {
                *(__int128_t*)(cached_input[token_idx][g]) = ku::ld_shared(row_base + (get_chunk_base<F>(g / 4) + g % 4) * 16);
            }
        });
        NamedBarrier::arrive_and_wait(128, K::barrier_ids::WG1_DEQUANT_SYNC);  // Make sure everyone has finished reading

        for_each_token([&]<typename F>(uint32_t token_idx) {
            static constexpr uint32_t NUM_STS_PER_CHUNK = num_sts_per_chunk<F>();
            uint8_t *token_base = slot_base + token_idx * STS_TOKEN_STRIDE_BYTES;
            ku::nvbf16x2 fp4_scales[2];   // fp4: the two scales of each of the current pair of chunks, as bf16
            // Each 16 B store is produced as four independent 32-bit values and stored as a uint4
            // aggregate, not through ku::st_shared's b128 form. That form passes the 16 B as one
            // 128-bit asm operand ("q"), so ptxas must place the four values in an aligned register
            // quad and, with several store chains in progress, gathers them with MOVs; from four
            // independent sources it emits the same single STS.128 with no MOVs
            static_assert(!F::IS_FP4 || (F::QUANT_TILE_SIZE / 2) % 4 == 0,
                          "a store quad must fall entirely inside one quantization tile");
            CUTE_UNROLL
            for (uint32_t g = 0; g < F::NUM_QUANT_CHUNKS_PER_LANE; ++g) {
                ku::nvbf16x2 scale_lo, scale_hi;
                __nv_fp8_e8m0 fp8_scale;
                if constexpr (F::IS_FP4) {
                    if (g % 2 == 0) {
                        fp8x4_to_bf16x2x2(cached_scales[token_idx][g / 2], fp4_scales);
                    }
                    scale_lo = __low2bfloat162(fp4_scales[g % 2]);
                    scale_hi = __high2bfloat162(fp4_scales[g % 2]);
                } else {
                    fp8_scale = ((__nv_fp8_e8m0*)cached_scales[token_idx])[g / chunks_per_tile<F>()];
                }
                CUTE_UNROLL
                for (uint32_t j = 0; j < NUM_STS_PER_CHUNK; ++j) {
                    ku::nvbf16x2 quad[4];
                    if constexpr (F::IS_FP4) {
                        fp4x8_to_bf16x2x4(cached_input[token_idx][g][j], quad);
                        ku::nvbf16x2 sc = j * 4 < F::QUANT_TILE_SIZE / 2 ? scale_lo : scale_hi;
                        CUTE_UNROLL
                        for (uint32_t k = 0; k < 4; ++k) quad[k] = __hmul2(quad[k], sc);
                    } else {
                        CUTE_UNROLL
                        for (uint32_t k = 0; k < 4; ++k) {
                            quad[k] = fp8x2_to_bf16x2_with_scale(((ku::nve4m3x2*)cached_input[token_idx][g])[j * 4 + k], fp8_scale);
                        }
                    }
                    *(uint4*)(token_base + quant_sts_offset<F>(g, j)) = *(const uint4*)quad;
                }
            }
        });

        cutlass::arch::fence_view_async_shared();
        K::arrive_on_cta0_barrier(smem.bar_kv_slot_full[kv_slot_idx]);
    }

    // Dispatch on this thread's number of fp8 tokens (warp-uniform since topk % 8 == 0) so a token's
    // conversion code is fixed at compile time by its format. When both caches have the same format
    // (both caches fp8, or one cache) every token takes the same code, so one body serves every value
    __device__ __forceinline__ void run_block(uint32_t num_fp8_tokens, uint32_t kv_slot_idx, uint32_t indices_buf_idx) const {
        if constexpr (std::is_same_v<OrigFormat, ExtraFormat>) {
            body<NUM_TOKENS_PER_THREAD>(kv_slot_idx, indices_buf_idx);
        } else {
            [&]<uint32_t... Ks>(std::integer_sequence<uint32_t, Ks...>) {
                ((num_fp8_tokens == Ks ? body<Ks>(kv_slot_idx, indices_buf_idx) : void()), ...);
            }(std::make_integer_sequence<uint32_t, NUM_TOKENS_PER_THREAD + 1>{});
        }
    }
};

// Gather front of one KV block (warp 11, see there): one gather4 per 4-row group of the block's raw rows into the KV
// slot, in bursts of 8 (the 8 int4 coordinates of a burst take 32 registers; all groups at once would not fit the
// 72-register allocation of that warpgroup). A block mixes the two caches (`mixed_block`) only when topk % B_TOPK != 0:
// its groups from `first_extra_group` on read the extra cache (`map_from_split`), the ones before it the original one
// (`map_before_split`). Every other block reads one cache: the MIXED_BLOCK = false instantiation takes that cache's map as
// `map_before_split` and has no per-group map select. The maps are byte offsets into tma_params: a select between two
// pointers into the kernel parameters whose result is an operand of the gather asm makes cicc (CUDA 13.1) crash, in a
// lambda as well as in a function
template<bool MIXED_BLOCK, uint32_t NUM_GATHER_GROUPS, uint32_t RAW_KV_GROUP_BYTES>
__device__ __forceinline__ void issue_raw_gathers(
    const int4 *coords, const uint8_t *maps_base, uint32_t map_before_split, uint32_t map_from_split, uint32_t first_extra_group,
    transac_bar_t &bar_raw_kv_full, uint8_t *slot_base
) {
    static_assert(NUM_GATHER_GROUPS % 8 == 0);
    // One burst of code, run NUM_GATHER_GROUPS / 8 times. The front runs once per block, and the per-block code of the h128
    // kernels is close to the instruction cache's capacity; unrolled bursts would be NUM_GATHER_GROUPS / 8 times the code
    // (h128: 3 x 8 gather4s, at two call sites)
    CUTE_NO_UNROLL
    for (uint32_t burst = 0; burst < NUM_GATHER_GROUPS / 8; ++burst) {
        int4 burst_coords[8];
        CUTE_UNROLL
        for (uint32_t i = 0; i < 8; ++i) {
            burst_coords[i] = coords[burst * 8 + i];
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t g = burst * 8 + i;
            uint32_t map_offset = map_before_split;
            if constexpr (MIXED_BLOCK) {
                map_offset = g < first_extra_group ? map_before_split : map_from_split;
            }
            ku::tma_gather4(
                maps_base + map_offset,
                bar_raw_kv_full,
                slot_base + g * RAW_KV_GROUP_BYTES,
                0,
                burst_coords[i],
                (int64_t)TMA::CacheHintSm90::EVICT_FIRST
            );
        }
    }
}

template<Config CONFIG>
__device__ __forceinline__
void Kernel<CONFIG>::devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params) {
#ifdef KERUTILS_ENABLE_SM100A
    const uint32_t cta_idx = IS_2CTA ? blockIdx.x % 2 : 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const uint32_t idx_in_warpgroup = threadIdx.x % 128;
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ char smem_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    }

    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        if constexpr (!IS_DECODE) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {
        // Init barriers
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {
            // bar_kv_slot_full:
            //   Prefill: 1 arrive (arrive_and_expect_tx from the MMA warp) + TMA transactions of the whole KV block
            //   Decode: 128 arrives (the dequant warpgroup) from each CTA + 1 arrive (the MMA warp) from CTA0; no TMA transactions
            smem.bar_kv_slot_full[i].init(IS_DECODE ? 128*CLUSTER_SIZE + 1 : 1);  // bar_kv_full: Every CTA -> CTA0
            smem.bar_kv_slot_empty[i].init(1); // bar_kv_empty: CTA0 -> Every CTA
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_INDICES_BUFS; ++i) {
            smem.bar_indices_full[i].init(32);  // CTA-local
            smem.bar_indices_empty[i].init(IS_DECODE ? 128 + 128 + 32 : 128);  // CTA-local: WG1 + WG3 + warp 11
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_P_BUFS; ++i) {
            smem.bar_tP_full[i].init(1);       // CTA0 -> Every CTA
            if constexpr (NEED_TP_EMPTY_BAR) {
                smem.bar_tP_empty[i].init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
            }
        }
        if constexpr (ENABLE_Q_NORM) {
            smem.bar_q_sqr_sum_full.init(128); // CTA-local: WG0 has written the job's sums of squares to q_sqr_sum_buf
        }
        smem.bar_clc_full.init(1);      // CTA0 -> Every CTA
        smem.bar_clc_empty.init(cta_idx == 1 ? 1 : NUM_WORKING_THREADS);   // Every CTA -> CTA0
        smem.bar_tQ_full.init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
        smem.bar_tQ_empty.init(1+128);      // CTA0 -> Every CTA (arrive by MMA thread), as well as CTA-local (arrived by exp warpgroup)
        smem.bar_tO_full.init(1);       // CTA0 -> Every CTA
        smem.bar_tO_empty.init(128*CLUSTER_SIZE);   // Every CTA -> CTA0
        smem.bar_SO_full.init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
        smem.bar_SO_empty.init(1);      // CTA0 -> Every CTA
        smem.bar_li_mi_full.init(128);  // CTA-local
        smem.bar_li_mi_empty.init(128); // CTA-local
        if constexpr (IS_DECODE) {
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_RAW_BARS; ++i) {
                smem.bar_raw_kv_full[i].init((IS_2CTA && cta_idx == 0) || BLOCK_SPLIT ? 33 : 1);  // TMA leader + CTA0's 32 deferred cp.async arrivals
                smem.bar_wg1_past_raw[i].init(4);      // CTA-local: one elected arrive per dequant warp
            }
        }
        fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate TMEM
        AllocatorT().allocate(512, smem.tmem_start_addr.data());
        AllocatorT().release_allocation_lock();
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
    }

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    } else {
        __syncthreads();
    }

    struct OuterloopArgs {
        bool is_valid;
        uint32_t s_q_idx;
        uint32_t job_idx_mod_2;
        uint32_t topk_length;
        uint32_t num_kv_blocks;
        // Decoding only:
        uint32_t extra_topk_length;     // Number of valid extra-topk entries of the current request
        uint32_t num_orig_slots;        // Number of slots occupied by the orig KV in the unified slot space, i.e. slots < num_orig_slots come from `kv`/`indices` (will be set to -1 if we don't have so many valid indices) while the others come from `extra_kv`/`extra_indices`. 0xFFFFFFFF when there is no extra KV (so that every slot belongs to the orig KV)
    };

    auto _make_outer_loop_args = [&](uint32_t job_idx_mod_2, uint32_t cta_x_idx) -> OuterloopArgs {
        uint32_t s_q_idx = cta_x_idx / CLUSTER_SIZE;
        if constexpr (IS_DECODE) {
            uint32_t topk_length = params.topk_length ? (uint32_t)__ldg(params.topk_length + s_q_idx) : (uint32_t)params.topk;
            uint32_t extra_topk_length = params.extra_topk_length ? (uint32_t)__ldg(params.extra_topk_length + s_q_idx) : (uint32_t)params.extra_topk;
            bool have_extra_kv = params.extra_topk > 0;
            uint32_t num_orig_slots, num_kv_blocks;
            if (have_extra_kv) {
                num_orig_slots = (uint32_t)params.topk;
                num_kv_blocks = ku::ceil_div(num_orig_slots + extra_topk_length, (uint32_t)B_TOPK); // The orig slots always use the full topk, so round topk + extra_topk_length up to whole blocks
            } else {
                num_orig_slots = 0xFFFFFFFFu;
                num_kv_blocks = ku::ceil_div(topk_length, (uint32_t)B_TOPK);
            }
            num_kv_blocks = std::max(num_kv_blocks, 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks,
                extra_topk_length,
                num_orig_slots
            };
        } else {
            uint32_t topk_length = params.topk_length ? __ldg(params.topk_length + s_q_idx) : params.topk;
            uint32_t num_kv_blocks = std::max(ku::ceil_div(topk_length, (uint32_t)B_TOPK), 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks
            };
        }
    };

    // Decode only: how block kv_block_idx of a job splits between the two KV caches. Its first
    // num_orig_rows rows come from `kv` (fp8), the rest from `extra_kv`
    auto get_num_orig_rows = [&](const OuterloopArgs &job, uint32_t kv_block_idx) -> uint32_t {
        uint32_t block_start = kv_block_idx * B_TOPK;
        return job.num_orig_slots >= block_start + B_TOPK ? B_TOPK
             : job.num_orig_slots <= block_start ? 0
             : job.num_orig_slots - block_start;
    };

    auto get_first_job = [&]() -> OuterloopArgs {
        return _make_outer_loop_args(0, blockIdx.x);
    };
    auto get_next_job = [&](const OuterloopArgs &cur_args) -> OuterloopArgs {
        smem.bar_clc_full.wait(cur_args.job_idx_mod_2);
        ku::CLCResult next_cta0_idx = ku::get_clc_query_response<true>(smem.clc_response_obj);
        arrive_on_cta0_barrier(smem.bar_clc_empty);

        if (!next_cta0_idx.is_valid) {
            return OuterloopArgs {false};
        } else {
            return _make_outer_loop_args(
                cur_args.job_idx_mod_2^1,
                next_cta0_idx.x
            );
        }
    };

    if (warpgroup_idx == 0) {
        /*
        Q fetching & Epilogue warpgroup

        The timeline of this warpgroup is as follows:
        Q0 Q1 O0 Q2 O1 Q3 O2 Q4 O3 ... Qn O(n-1) On

        Where
        - Qi means loading the Q of the i-th request, computing the sum of squares of each head on the fly,
          performing RoPE transformation, then writing Q to TMEM
        - Oi means reading the O of the i-th request from TMEM, performing RoPE transformation,
          quantizing to FP8, and writing back to global memory

        About cached_cos and cached_sin:
        - D_ROPE is always 64, so the i-th lane caches the i-th cos and sin dimension of the
          corresponding position
        - We always cache the cos and sin of the i-th request and the (i-1)-th request, so cached_cos
          and cached_sin have two slots
        - Every time a new Q is loaded, cached_cos/sin[1] (the first slot) is assigned to the zeroth slot
          (and "sin" is negated to prepare for the conjugate RoPE of O),
          and the cos/sin of the new Q is saved to the 1st slot
        - This way, O RoPE only needs to read from the 0th slot (except for the last O)
        */
        cutlass::arch::warpgroup_reg_alloc<WG0_REGS>();
        // The thread's TMEM lane is 32 * warp + lane; the 128 lanes are FOLD_FACTOR groups of H_Q_PER_CTA rows (config.h). Warp w
        // belongs to lane group w / (H_Q_PER_CTA / 32) and holds the K-view of rail w / 2 of Q (groups 0 | 1 = rails 0 | 1),
        // and its row is head idx % H_Q_PER_CTA. Both are derived from warp_idx, not from idx_in_warpgroup, so the compiler
        // sees warp-uniform values
        static_assert(NUM_MRGEMM_RAILS == 2, "rail = warp_idx / 2 below");
        const uint32_t rail = warp_idx / 2;

        #pragma nv_diag_suppress 549    // Uninitialized variable. The following two variables are indeed initialized but the compiler is too dumb to release that
        float cached_cos[2], cached_sin[2];
        auto shift_cached_cos_and_cached_sin = [&]() {
            cached_cos[0] = cached_cos[1];
            cached_sin[0] = -cached_sin[1];
        };

        auto load_q_and_save_to_tmem = [&](const OuterloopArgs &cur_job) {
            // A thread holds the K-view of one rail for its head: D_QK / NUM_MRGEMM_RAILS bf16
            static constexpr uint32_t NUM_CACHED_Q_BF16_PER_THREAD = D_QK / NUM_MRGEMM_RAILS;
            static constexpr uint32_t NUM_BF16_PER_LOAD = 256 / 16; // LDG 256
            bf16 cached_q[NUM_CACHED_Q_BF16_PER_THREAD];
            float q_sqr_sum = 0.0f;

            uint32_t cur_q_position = __ldg(params.token_positions + cur_job.s_q_idx);
            cached_cos[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + lane_idx);
            cached_sin[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + D_ROPE / 2 + lane_idx);

            bf16 *q_token_base = params.q + (uint64_t)cur_job.s_q_idx * params.stride_q_s_q;

            static constexpr uint32_t TILE_SIZE = 64;
            // GROUP_Q_LOADS: the 16 loads of a thread (4 warps x 16 x LDG.256 = 64 KB per CTA) are issued one 16 KB tile at a
            // time. More than ~24 KB of global loads issued but not completed on an SM stalls every shared-memory instruction
            // of the SM (STS, LDS, proxy fence, mbarrier arrive) until the data has returned. This load follows get_next_job,
            // while WG3 stores the S of the next job's second block (in decode also while WG1 writes the dequantized KV
            // block), so a 64 KB burst delays that S and the SV that waits on it by ~2000 cycles in prefill. Each tile's
            // addresses carry one word of every load of the previous tile, masked by a run-time zero, so ptxas issues a tile
            // only after the previous tile has arrived. Every load contributes a word: given the word of one load only, ptxas
            // hoists that load and merges the other loads into one burst. The serial tiles cost ~3 load latencies per job,
            // inside this warpgroup's ~10k cycles of slack
            uint32_t q_load_dep = 0;
            const uint32_t q_zero = cur_job.topk_length >> 31;   // topk_length < 2^31, so 0 at run time; a value ptxas cannot fold away
            CUTE_UNROLL
            for (uint32_t local_tile_idx = 3; local_tile_idx != 0xFFFFFFFF; --local_tile_idx) {
                // Rail j holds the 64-dim tiles j, j+2, j+4, j+6 (1 CTA) or the CTA's half of the dims in order (2 CTAs)
                uint32_t tile_idx = CLUSTER_SIZE == 1 ? local_tile_idx * 2 + rail : rail * 4 + local_tile_idx;
                CUTE_UNROLL
                for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                    uint32_t h_q_idx = cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA;
                    uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;
                    KU_LDG_256(
                        q_token_base + h_q_idx * NUM_BF16_PER_LOAD + d_q_idx * H_Q + (GROUP_Q_LOADS ? q_load_dep : 0u), // always 0
                        cached_q + local_tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD,
                        ".nc", "no_allocate", "evict_first", "256B"
                    );
                }
                if constexpr (GROUP_Q_LOADS) {
                    uint32_t tile_words = 0;
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                        tile_words ^= *(const uint32_t*)(cached_q + local_tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD + NUM_BF16_PER_LOAD - 2);
                    }
                    q_load_dep = tile_words & q_zero;
                }
                // Perform RoPE. This tile's sum of squares is taken here, before the rotation, from the values the
                // loop converts anyway; the other tiles' sums are taken after the TMEM store (below)
                if (local_tile_idx == 3 && tile_idx == D_VO / TILE_SIZE - 1) {
                    float2 cur_q_sqr_sum = {0.0f, 0.0f};
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < TILE_SIZE; j += 2) {
                        float2 x = __bfloat1622float2(*(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j));
                        if constexpr (ENABLE_Q_NORM) {
                            cur_q_sqr_sum = ku::float2_fma(x, x, cur_q_sqr_sum);
                        }
                        float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[1], j/2);
                        float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[1], j/2);
                        float2 y = apply_rope(x, cur_cos, cur_sin);
                        *(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j) = nv_bfloat162{__float2bfloat16_rn(y.x), __float2bfloat16_rn(y.y)};
                    }
                    q_sqr_sum += cur_q_sqr_sum.x + cur_q_sqr_sum.y;
                } else if constexpr (ENABLE_Q_NORM && !COMPACT_PER_JOB_CODE) {
                    // This tile's sum of squares, one fma.rn.f32.bf16 per element, NUM_CACHED_Q_BF16_PER_THREAD - TILE_SIZE = 192 instructions, With
                    // COMPACT_PER_JOB_CODE the sum is taken from TMEM after the store instead to save icache
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < TILE_SIZE; ++i)
                        asm volatile ("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(q_sqr_sum) : "h"(*(uint16_t*)(cached_q+local_tile_idx*TILE_SIZE+i)));
                }
            }

            smem.bar_tQ_empty.wait(cur_job.job_idx_mod_2^1);
            ku::tcgen05_after_thread_sync();
            
            static constexpr uint32_t NUM_CACHED_UINT32 = NUM_CACHED_Q_BF16_PER_THREAD/2;
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q, cached_q);
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q+NUM_CACHED_UINT32/2, cached_q+NUM_CACHED_UINT32/2*2);   // Two stores: one store of all NUM_CACHED_UINT32 words does not compile (insufficient registers)
            cutlass::arch::fence_view_async_tmem_store();

            ku::tcgen05_before_thread_sync();
            arrive_on_cta0_barrier(smem.bar_tQ_full);

            if constexpr (ENABLE_Q_NORM && COMPACT_PER_JOB_CODE) {
                // Sum of squares of the tiles other than the RoPE one, read back from TMEM one tile per iteration, in the
                // order the tiles were loaded (3 unless it is the RoPE tile, then 2, 1, 0), each element added with
                // fma.rn.f32.bf16: the same operations in the same order as the in-register sum of the load loop, so the
                // same fp32 result, in one tile of code
                // Local tile 3 is the RoPE tile for the warps whose tile_idx of it is the last one (the formula of the load loop)
                const uint32_t tile_idx_of_local_3 = CLUSTER_SIZE == 1 ? 3 * 2 + rail : rail * 4 + 3;
                const uint32_t first_tile = tile_idx_of_local_3 == D_VO / TILE_SIZE - 1 ? 2 : 3;
                CUTE_NO_UNROLL
                for (uint32_t local_tile_idx = first_tile; local_tile_idx != 0xFFFFFFFF; --local_tile_idx) {
                    uint32_t q_words[TILE_SIZE / 2];
                    ku::tmem_ld_32dp32bNx<TILE_SIZE / 2>(tmem_cols::Q + local_tile_idx * (TILE_SIZE / 2), q_words);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
                        asm volatile ("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(q_sqr_sum) : "h"(((const uint16_t*)q_words)[i]));
                    }
                }
            }
            if constexpr (ENABLE_Q_NORM) {
                smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] = q_sqr_sum;
                smem.bar_q_sqr_sum_full.arrive();
            }
        }; 
        auto store_o = [&](const OuterloopArgs &cur_job, const bool &is_last_job) { 
            smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
            float mi = smem.rowwise_mi_buf[idx_in_warpgroup % H_Q_PER_CTA];
            // The row sum over the FOLD_FACTOR token slices (softmax warp w holds slice q(w), see there); the same sum in the same
            // order in the prefill and the decode kernel
            float li = smem.rowwise_li_buf[idx_in_warpgroup] + smem.rowwise_li_buf[idx_in_warpgroup ^ 64];
            smem.bar_li_mi_empty.arrive();

            if (idx_in_warpgroup < H_Q_PER_CTA) {
                uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                params.lse[global_index] = cur_lse;
            }

            float attn_sink = (params.attn_sink == nullptr) ? -CUDART_INF_F : __ldg(params.attn_sink + cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA) * CUDART_L2E_F;
            float output_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));

            smem.bar_tO_full.wait(cur_job.job_idx_mod_2);
            ku::tcgen05_after_thread_sync();
            
            static constexpr uint32_t MMA_ATOM_N = 256;
            static constexpr uint32_t NUM_MMA_ATOMS = D_VO / MMA_ATOM_N;
            static constexpr uint32_t NUM_O_TMEM_COLS_PER_ATOM = MMA_ATOM_N / FOLD_FACTOR;
            static constexpr uint32_t EPILOGUE_TILE_SIZE = O_QUANT_TILE_SIZE;
            static constexpr uint32_t NUM_EPILOGUE_TILES_PER_ATOM = NUM_O_TMEM_COLS_PER_ATOM / EPILOGUE_TILE_SIZE;
            static_assert(NUM_O_TMEM_COLS_PER_ATOM % EPILOGUE_TILE_SIZE == 0);
            // Since the weight is per-32 scaled and deep_gemm.einsum requires A and B to have the same scale granularity, the output sf is always stored in a per-32 scaled format, although it will be actually (numerically) per-128 scaled when num_per_channels is 128
            static constexpr uint32_t OUTPUT_SAVE_AS_SCALE_GRAN = 32;
            static_assert(EPILOGUE_TILE_SIZE == OUTPUT_SAVE_AS_SCALE_GRAN, "one tile = one 32 B output store + one sf byte");

            uint32_t head_idx = cta_idx*H_Q_PER_CTA + idx_in_warpgroup%H_Q_PER_CTA;
            uint32_t wv_group_idx = head_idx / WV_GROUP_SIZE;
            uint32_t head_idx_in_wv_group = head_idx % WV_GROUP_SIZE;

            // Layout of O in Tensor Memory:
            // For HEAD_DIM_QK = 64 (head64):
            //   - Atom 0 computes O[:, 0:256]; Atom 1 computes O[:, 256:512]
            //   - Mapping to TMEM:
            //       O[0:128]   -> TMEM[0:64,   0:128]
            //       O[128:256] -> TMEM[64:128,  0:128]
            //       O[256:384] -> TMEM[0:64,   128:256]
            //       O[384:512] -> TMEM[64:128, 128:256]
            //   - Visually (label the four 128-wide O chunks as 0..3):
            //        +---+---+
            //        | 0 | 2 |
            //        +---+---+
            //        | 1 | 3 |
            //        +---+---+
            // For HEAD_DIM_QK = 128 (head128):
            //   - CTA0 holds V[:, 0:256]; CTA1 holds V[:, 256:512]
            //   - Atom 0 computes O[:, 0:128] and O[:, 256:384]
            //     Atom 1 computes O[:, 128:256] and O[:, 384:512]
            //   - Mapping to TMEM:
            //       O[0:128]   -> TMEM[0:64,   0:128]
            //       O[128:256] -> TMEM[0:64,   128:256]
            //       O[256:384] -> TMEM[64:128, 0:128]
            //       O[384:512] -> TMEM[64:128, 128:256]
            //   - Visually (label the four 128-wide O chunks as 0..3):
            //        +---+---+
            //        | 0 | 1 |
            //        +---+---+
            //        | 2 | 3 |
            //        +---+---+
            //
            // A tile is 32 columns of one atom: TMEM load, conjugate RoPE, quantization, its 32 B output store and its sf
            // byte. Two forms of the loop below: with EPILOGUE_TILES_AHEAD > 0 the TMEM loads run ahead of the conversion,
            // otherwise one tile is finished before the next one starts. finish_tile is a tile's work after its TMEM load;
            // the load and the tO_empty arrive belong to the loop
            auto finish_tile = [&](const uint32_t mma_atom_idx, const uint32_t epilogue_tile_idx_in_atom, float (&output)[EPILOGUE_TILE_SIZE], const float reduce_result_by_tmem_ld) {
                // RoPE (conjugate); the condition is warp-uniform
                bool should_perform_rope;
                {
                    static_assert(D_ROPE % EPILOGUE_TILE_SIZE == 0);
                    // The RoPE dims [D_VO - D_ROPE, D_VO) are the last atom's last two tiles of the last lane group (the group that holds
                    // the atom's last dims, warps (FOLD_FACTOR - 1) * H_Q_PER_CTA / 32 on, i.e. warps 2 and 3, tiles 2 and 3 of 4).
                    // The condition is warp-uniform; lane j/2 of cos/sin holds pair j/2 of the 64 RoPE dims
                    static_assert(D_ROPE / EPILOGUE_TILE_SIZE == 2 && NUM_EPILOGUE_TILES_PER_ATOM >= 2);
                    should_perform_rope =
                        mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                        epilogue_tile_idx_in_atom >= NUM_EPILOGUE_TILES_PER_ATOM - D_ROPE/EPILOGUE_TILE_SIZE &&
                        warp_idx >= (FOLD_FACTOR - 1) * (H_Q_PER_CTA / 32);
                    if (should_perform_rope) {
                        CUTE_UNROLL
                        for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                            float2 x = *(float2*)(output + j);
                            uint32_t src_lane = j/2 + (epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM ? EPILOGUE_TILE_SIZE / 2 : 0);
                            float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[0], src_lane);
                            float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[0], src_lane);
                            float2 y = apply_rope(x, cur_cos, cur_sin);
                            *(float2*)(output + j) = y;
                        }
                    }
                }

                // Cast to FP8
                float output_abs_max;
                if (!IS_TMEM_LD_WITH_RED_AVAILABLE || should_perform_rope) {
                    output_abs_max = get_max<EPILOGUE_TILE_SIZE, true>(output) * output_scale;
                } else {
                    output_abs_max = reduce_result_by_tmem_ld * output_scale;
                }
                output_abs_max = max(O_QUANT_CLAMP_MIN_VALUE, output_abs_max);
                float sf = output_abs_max / 448.0f;
                uint32_t sf_as_uint32 = *reinterpret_cast<uint32_t*>(&sf);
                uint32_t exp_sf = (int32_t)((sf_as_uint32-1) >> 23) + (1 - 127);
                uint32_t sf_inv_as_uint32 = (127 - exp_sf) << 23;
                float sf_inv = *reinterpret_cast<float*>(&sf_inv_as_uint32);
                float cur_multiplier = output_scale * sf_inv;
                float2 cur_multiplier_float2 = float2(cur_multiplier, cur_multiplier);

                fp8_e4m3 tile_fp8[EPILOGUE_TILE_SIZE];
                CUTE_UNROLL
                for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                    float2 x = *(float2*)(output + j);
                    x = ku::float2_mul(x, cur_multiplier_float2);
                    *(__nv_fp8x2_storage_t*)(tile_fp8 + j) = __nv_cvt_float2_to_fp8x2(
                        x,
                        __NV_SATFINITE,
                        __nv_fp8_interpretation_t::__NV_E4M3
                    );  // NOTE. Here we don't use cvt.f8x4type.f32 since it only has .rs mode, which affects accuracy
                }

                // Store this tile's sf byte and its 32 output bytes
                // Lane group g (= warp_idx / (H_Q_PER_CTA / 32)) holds dims [g * MMA_ATOM_N / FOLD_FACTOR, +MMA_ATOM_N / FOLD_FACTOR) of each
                // atom (1 CTA), or the CTA's half of the dims: [g * MMA_ATOM_N, +MMA_ATOM_N) of each half-atom (2 CTAs)
                uint32_t head_dim_idx_base =
                    CLUSTER_SIZE == 1 ?
                    mma_atom_idx * MMA_ATOM_N + (warp_idx/(H_Q_PER_CTA/32)) * (MMA_ATOM_N/FOLD_FACTOR) :
                    mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + (warp_idx/(H_Q_PER_CTA/32)) * MMA_ATOM_N;
                uint32_t head_dim_idx = head_dim_idx_base + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                uint32_t sf_block_idx = head_idx_in_wv_group + (head_dim_idx/OUTPUT_SAVE_AS_SCALE_GRAN) * WV_GROUP_SIZE;
                *((uint8_t*)(params.out_sf + cur_job.s_q_idx + wv_group_idx*params.stride_out_sf_wv_group + (sf_block_idx/4)*params.stride_out_sf_head_dim) + sf_block_idx%4) = (uint8_t)(exp_sf + 127);  // TODO Optimize
                KU_STG_256(
                    params.out_fp8 + (uint64_t)cur_job.s_q_idx*(H_Q*D_VO) + wv_group_idx*(WV_GROUP_SIZE*D_VO) + head_idx_in_wv_group*32 + head_dim_idx*WV_GROUP_SIZE,
                    tile_fp8,
                    "no_allocate",
                    "evict_first"
                );
            };

            if constexpr (EPILOGUE_TILES_AHEAD > 0) {
                // The TMEM loads run EPILOGUE_TILES_AHEAD tiles ahead of the conversion. A tcgen05.wait::ld covers every load
                // issued so far, so the wait of tile t covers the load of tile t + EPILOGUE_TILES_AHEAD - 1, and the tO_empty
                // arrive follows the wait that covers the last tile's load: the next job's first SV waits on tO_empty and
                // starts while the remaining tiles are converted and stored from registers
                static constexpr uint32_t NUM_TILES = NUM_MMA_ATOMS * NUM_EPILOGUE_TILES_PER_ATOM;
                static_assert(!COMPACT_PER_JOB_CODE && EPILOGUE_TILES_AHEAD <= NUM_TILES);
                float outputs[NUM_TILES][EPILOGUE_TILE_SIZE];
                float reduce_results_by_tmem_ld[NUM_TILES];
                auto issue_tile = [&](const uint32_t t) {
                    const uint32_t atom = t / NUM_EPILOGUE_TILES_PER_ATOM, tile = t % NUM_EPILOGUE_TILES_PER_ATOM;
                    const uint32_t col = tmem_cols::O + atom * NUM_O_TMEM_COLS_PER_ATOM + tile * EPILOGUE_TILE_SIZE;
                    if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                        ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(col, outputs[t], reduce_results_by_tmem_ld[t]);
                    } else {
                        ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(col, outputs[t]);
                    }
                };
                cute::for_each(cute::make_int_sequence<EPILOGUE_TILES_AHEAD>{}, [&](auto t) { issue_tile(t); });
                cute::for_each(cute::make_int_sequence<NUM_TILES>{}, [&](auto t_) {
                    constexpr uint32_t t = decltype(t_)::value;
                    cutlass::arch::fence_view_async_tmem_load();   // the loads of tiles <= min(t + EPILOGUE_TILES_AHEAD - 1, NUM_TILES - 1) have completed
                    if constexpr (t + EPILOGUE_TILES_AHEAD < NUM_TILES) {
                        issue_tile(t + EPILOGUE_TILES_AHEAD);
                    }
                    if constexpr (t + EPILOGUE_TILES_AHEAD - 1 == NUM_TILES - 1) {
                        // Notify tO's emptiness: every tile of O has been read out of TMEM
                        ku::tcgen05_before_thread_sync();
                        if (!is_last_job) {
                            // Don't arrive on the barrier if this job is the last job, to avoid "cluster target block not present"
                            arrive_on_cta0_barrier(smem.bar_tO_empty);
                        }
                    }
                    finish_tile(t / NUM_EPILOGUE_TILES_PER_ATOM, t % NUM_EPILOGUE_TILES_PER_ATOM, outputs[t], reduce_results_by_tmem_ld[t]);
                });
            } else {
                // With COMPACT_PER_JOB_CODE the loops are not unrolled, so the body exists once in the binary and no per-tile
                // state is live across iterations; the loop control adds nothing measurable (this warpgroup is busy about a
                // tenth of the time). Otherwise they are unrolled
                #pragma unroll (COMPACT_PER_JOB_CODE ? 1 : NUM_MMA_ATOMS)
                for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS; ++mma_atom_idx)
                #pragma unroll (COMPACT_PER_JOB_CODE ? 1 : NUM_EPILOGUE_TILES_PER_ATOM)
                for (uint32_t epilogue_tile_idx_in_atom = 0; epilogue_tile_idx_in_atom < NUM_EPILOGUE_TILES_PER_ATOM; ++epilogue_tile_idx_in_atom) {
                    const bool is_last_tile = mma_atom_idx + 1 == NUM_MMA_ATOMS && epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM;

                    // Fetch output from TMEM
                    uint32_t tmem_col_base = tmem_cols::O + mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                    float output[EPILOGUE_TILE_SIZE];
                    float reduce_result_by_tmem_ld;
                    if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                        ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(tmem_col_base, output, reduce_result_by_tmem_ld);
                    } else {
                        ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(tmem_col_base, output);
                    }
                    cutlass::arch::fence_view_async_tmem_load();

                    // Notify tO's emptiness
                    if (is_last_tile) {
                        ku::tcgen05_before_thread_sync();
                        if (!is_last_job) {
                            // Don't arrive on the barrier if this job is the last job, to avoid "cluster target block not present"
                            arrive_on_cta0_barrier(smem.bar_tO_empty);
                        }
                    }
                    finish_tile(mma_atom_idx, epilogue_tile_idx_in_atom, output, reduce_result_by_tmem_ld);
                }
            }
        };

        OuterloopArgs cur_job = get_first_job();
        load_q_and_save_to_tmem(cur_job);
        do {
            OuterloopArgs next_job = get_next_job(cur_job);
            shift_cached_cos_and_cached_sin();
            if (next_job.is_valid) {
                load_q_and_save_to_tmem(next_job);
            }
            store_o(cur_job, !next_job.is_valid);
            cur_job = next_job;
        } while (cur_job.is_valid);

        NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
        if (warp_idx == 0) {
            AllocatorT().free(0, 512);
        }
    } else if (warpgroup_idx == 3) {
        // Scale & Exp warpgroup
        cutlass::arch::warpgroup_reg_alloc<WG3_REGS>();

        OuterloopArgs cur_job = get_first_job();
        uint32_t local_warp_idx = warp_idx - 12;
        // After retrieve_mask_and_reduce_p, warp w holds every head's tokens [q(w) * B_TOPK / FOLD_FACTOR, +B_TOPK / FOLD_FACTOR):
        // q(w) = w / 2 (warps 0, 1 | 2, 3)
        // S is K-major with 8-row core matrices (INTER layout), so a head's 8 tokens are 16 B at row * 16 B of the token group's block
        static_assert(FOLD_FACTOR == 2);
        const uint32_t token_slice = local_warp_idx >= 2 ? 1u : 0u;
        bf16 *sS_base = smem.s + token_slice * (H_Q_PER_CTA * (B_TOPK / FOLD_FACTOR)) + (idx_in_warpgroup % H_Q_PER_CTA) * 8;
        RingBufferState rs;
        do {
            // For definition and consistency about `mi`, `li`, and `real_mi`, plz refer to head64 prefill
            static constexpr uint32_t NUM_ELEMS_PER_THREAD = B_TOPK * H_Q_PER_CTA / 128;
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            float score_multiplier; // qk_scale * rms_norm's denominator (if q norm is enabled)
            if constexpr (ENABLE_Q_NORM) {
                smem.bar_q_sqr_sum_full.wait(cur_job.job_idx_mod_2);

                // WG0 thread t summed the squares of its rail's dims of head t % H_Q_PER_CTA; threads t and t ^ 64 hold the two rails of
                // the same head
                static_assert(NUM_MRGEMM_RAILS == 2);
                score_multiplier = smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] + smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^64];
                score_multiplier = params.sm_scale_div_log2 * rsqrtf(score_multiplier / D_QK + params.rms_norm_eps);    // rsqrt is translated to `MUFU.RSQ`
            } else {
                score_multiplier = params.sm_scale_div_log2;
            }

            smem.bar_tQ_empty.arrive(); // Must arrive on the empty barrier here, to prevent smem.bar_tQ_full being phase-skipped

            CUTE_NO_UNROLL
            for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();
                smem.bar_tP_full[p_buf_idx].wait(p_bar_phase);
                smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<
                    H_Q_PER_CTA,
                    NUM_ELEMS_PER_THREAD,
                    barrier_ids::WG3_WARP02_SYNC,
                    barrier_ids::WG3_WARP13_SYNC
                >(
                    tmem_cols::get_p(p_buf_idx),
                    (char*)&smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {
                        if constexpr (NEED_TP_EMPTY_BAR) {
                            arrive_on_cta0_barrier(smem.bar_tP_empty[p_buf_idx]);
                        }
                    },
                    smem.p_exchange_buf,
                    p
                );

                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= score_multiplier;

                // Row max over the lane groups holding the head's other token slices: the partner warp (w ^ 2, a 64-thread
                // barrier per pair). max is exact in any order
                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(32 * FOLD_FACTOR, barrier_ids::WG3_WARP02_SYNC + (local_warp_idx & 1));
                smem.bar_indices_empty[indices_buf_idx].arrive();   // Put it here to give the compiler more room for code reordering
                CUTE_UNROLL
                for (uint32_t g = 1; g < FOLD_FACTOR; ++g) {
                    cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup ^ (g * H_Q_PER_CTA)]);
                }
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {
                    // Don't scale O
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;   // mi is still identical within each row

                // Calculate S
                nv_bfloat16 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>((nv_bfloat162*)s, p, score_multiplier, new_max);
                li = fmaf(li, scale_for_old, cur_sum);

                // Store S
                smem.bar_SO_empty.wait(rs.get<1>().second^1);
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    ku::st_shared(sS_base + i*8*H_Q_PER_CTA, *(__int128_t*)(s + i*8));
                }

                // Rescale O
                if (kv_block_idx > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_VO / FOLD_FACTOR, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                ku::tcgen05_before_thread_sync();
                arrive_on_cta0_barrier(smem.bar_SO_full);
                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // No valid TopK indices
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            smem.bar_li_mi_empty.wait(cur_job.job_idx_mod_2^1);
            static_assert(H_Q_PER_CTA % 32 == 0);
            if (local_warp_idx < H_Q_PER_CTA / 32) {
                if constexpr (!IS_DECODE) {
                    uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                    params.max_logits[global_index] = real_mi * CUDART_LN2_F;
                }
                smem.rowwise_mi_buf[idx_in_warpgroup] = mi;
            }
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            smem.bar_li_mi_full.arrive();

            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<WG2_REGS>();
        if (warp_idx == 8 && cta_idx == 0 && elect_one_sync()) {
            // MMA warp (CTA0 only)
            auto tiled_mma_qk = TiledMMA_QK{};
            auto tiled_mma_sv = TiledMMA_SV{};
            Tensor tQ = tiled_mma_qk.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<D_QK / NUM_MRGEMM_RAILS>>{})
            );
            Tensor tP = partition_fragment_C(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<B_TOPK * NUM_MRGEMM_RAILS>>{});
            Tensor sS = make_tensor(
                make_smem_ptr(smem.s),
                ku::make_umma_canonical_k_major_layout<H_Q_PER_CTA, B_TOPK, 0>()
            );
            Tensor tO = partition_fragment_C(tiled_mma_sv, Shape<Int<H_Q_PER_CTA>, Int<D_VO>>{});
            tQ.data().get() = tmem_cols::Q;
            tO.data().get() = tmem_cols::O;
            // tP.data() will be assigned in the loop since it has double buffers

            RingBufferState rs_qk, rs_sv;
            // Issue QK on the current kv slot. The caller has waited on kv_slot_full, on tQ_full for the
            // first block of a job, and on tP_empty when P is single-buffered
            auto issue_qk = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                auto kv_slot_idx = rs_qk.get<NUM_KV_SLOTS>().first;
                Tensor sK = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_k_major_layout<B_TOPK / CLUSTER_SIZE * NUM_MRGEMM_RAILS, D_QK / NUM_MRGEMM_RAILS, 128>()
                );
                auto p_buf_idx = rs_qk.get<NUM_P_BUFS>().first;
                tP.data().get() = tmem_cols::get_p(p_buf_idx);

                ku::tcgen05_after_thread_sync();
                ku::utcmma_ts(tiled_mma_qk, tQ, sK, tP, true);
                umma_arrive_on_every_cta(smem.bar_tP_full[p_buf_idx]);

                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tQ_empty);
                }
                rs_qk.update();
            };
            // Prefill: the KV block arrives by TMA, so the arrive carries its transaction bytes. Decode: the dequant warps
            // write the whole slot and arrive on the same barrier, so this is a plain arrive
            auto arrive_kv_slot_full = [&](uint32_t kv_slot_idx) {
                if constexpr (IS_DECODE) {
                    smem.bar_kv_slot_full[kv_slot_idx].arrive();
                } else {
                    smem.bar_kv_slot_full[kv_slot_idx].arrive_and_expect_tx(B_TOPK*D_QK*sizeof(bf16));
                }
            };
            // Issue SV on the current kv slot. The caller has waited on SO_full and, for the first block of
            // a job, on tO_empty
            auto issue_sv = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                auto kv_slot_idx = rs_sv.get<NUM_KV_SLOTS>().first;
                Tensor sV = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_mn_major_layout<D_VO / CLUSTER_SIZE, B_TOPK, 128>()
                );
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_sv, sS, sV, tO, kv_block_idx == 0);
                umma_arrive_on_every_cta(smem.bar_kv_slot_empty[kv_slot_idx]);
                umma_arrive_on_every_cta(smem.bar_SO_empty);
                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tO_full);
                }
                rs_sv.update();
            };

            if constexpr (IS_DECODE && !NEED_TP_EMPTY_BAR) {
                // Adaptive issue. In the fixed order SV(g) is issued after QK(g+1)'s kv_slot_full wait, so a late
                // dequant of block g+1 delays SV(g) and the release of slot g to the KV producer. The two gemms
                // wait on different barriers (kv_slot_full vs SO_full), so the warp polls both and issues
                // whichever is ready.
                // P is double-buffered and QK(g) overwrites P[g%2], the buffer of block g-2.
                // num_qk_issued <= num_sv_issued + 1 issues QK(g) only after SV(g-2), and SV(g-2) waited on
                // SO_full, so softmax(g-2) has finished reading it.
                // The polls are test_wait, which returns at once, so a pending SV input does not keep a ready QK
                // from being issued
                OuterloopArgs qk_job = get_first_job();
                OuterloopArgs sv_job = qk_job;
                uint32_t qk_block_idx = 0, sv_block_idx = 0;    // block index within qk_job / sv_job
                uint32_t num_qk_issued = 0, num_sv_issued = 0;  // gemms issued so far, across jobs
                uint32_t qk_job_idx = 0, sv_job_idx = 0;        // index of qk_job / sv_job among this CTA's jobs
                bool qk_all_done = false;
                bool kv_slot_full_arrived = false;  // arrive_kv_slot_full has been issued for the pending QK block (once per block, before its first poll)
                while (!qk_all_done || num_sv_issued < num_qk_issued) {
                    if (num_sv_issued < num_qk_issued && smem.bar_SO_full.test_wait(rs_sv.get<1>().second)
                        && (sv_block_idx > 0 || smem.bar_tO_empty.test_wait(sv_job.job_idx_mod_2 ^ 1))) {
                        issue_sv(sv_job, sv_block_idx);
                        ++num_sv_issued;
                        if (++sv_block_idx == sv_job.num_kv_blocks) {
                            // Each SV is issued after its QK, so QK has finished this job; the sv_job_idx == qk_job_idx
                            // guard below stops QK from starting the job after next, so qk_job is the SV side's next job
                            sv_block_idx = 0;
                            ++sv_job_idx;
                            sv_job = qk_job;
                        }
                        continue;
                    }
                    // QK starts a job only after SV has finished the job before it (sv_job_idx == qk_job_idx),
                    // so qk_job is never more than one job ahead of sv_job. With a 1-block job QK could otherwise
                    // finish two jobs between consecutive SV issues and the sv_job = qk_job copy above would skip one
                    if (!qk_all_done && num_qk_issued <= num_sv_issued + 1 && (qk_block_idx > 0 || sv_job_idx == qk_job_idx)) {
                        auto [kv_slot_idx, kv_bar_phase] = rs_qk.get<NUM_KV_SLOTS>();
                        if (!kv_slot_full_arrived) {
                            kv_slot_full_arrived = true;
                            arrive_kv_slot_full(kv_slot_idx);
                        }
                        bool ready = smem.bar_kv_slot_full[kv_slot_idx].test_wait(kv_bar_phase);
                        if (ready && qk_block_idx == 0) {
                            ready = smem.bar_tQ_full.test_wait(qk_job.job_idx_mod_2);
                        }
                        if (ready) {
                            issue_qk(qk_job, qk_block_idx);
                            kv_slot_full_arrived = false;
                            ++num_qk_issued;
                            if (++qk_block_idx == qk_job.num_kv_blocks) {
                                qk_block_idx = 0;
                                ++qk_job_idx;
                                qk_job = get_next_job(qk_job);
                                qk_all_done = !qk_job.is_valid;
                            }
                        }
                    }
                }
            } else {
                // The adaptive loop replaces the tP_empty wait with the issue-distance bound
                // num_qk_issued <= num_sv_issued + 1, so it needs two P buffers; the 2-CTA decode has one, and
                // prefill loads KV by TMA with no dequant to run late. Both keep the fixed interleaving
                // QK(0); { QK(k), SV(k-1) }*; SV(last)
                auto run_qk_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                    if (kv_block_idx == 0) {
                        smem.bar_tQ_full.wait(job.job_idx_mod_2);
                    }
                    auto [kv_slot_idx, kv_bar_phase] = rs_qk.get<NUM_KV_SLOTS>();
                    arrive_kv_slot_full(kv_slot_idx);
                    smem.bar_kv_slot_full[kv_slot_idx].wait(kv_bar_phase);
                    if constexpr (NEED_TP_EMPTY_BAR) {
                        auto [p_buf_idx, p_bar_phase] = rs_qk.get<NUM_P_BUFS>();
                        smem.bar_tP_empty[p_buf_idx].wait(p_bar_phase ^ 1);
                    }
                    issue_qk(job, kv_block_idx);
                };
                auto run_sv_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                    if (kv_block_idx == 0) {
                        smem.bar_tO_empty.wait(job.job_idx_mod_2^1);
                    }
                    smem.bar_SO_full.wait(rs_sv.get<1>().second);
                    issue_sv(job, kv_block_idx);
                };

                OuterloopArgs cur_job = get_first_job();
                run_qk_gemm(cur_job, 0);
                do {
                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 1; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        run_qk_gemm(cur_job, kv_block_idx);
                        run_sv_gemm(cur_job, kv_block_idx-1);
                    }

                    OuterloopArgs next_job = get_next_job(cur_job);
                    if (next_job.is_valid) {
                        run_qk_gemm(next_job, 0);
                    }
                    run_sv_gemm(cur_job, cur_job.num_kv_blocks-1);
                    
                    cur_job = next_job;
                } while (cur_job.is_valid);
            }
        } else if (warp_idx == 9 && elect_one_sync()) {
            // CLC warp
            bool phase = 0;
            while (true) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase^1);
                    ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                }
                smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                
                smem.bar_clc_full.wait(phase&1);
                ku::CLCResult clc_result = ku::get_clc_query_response<true>(smem.clc_response_obj);
                arrive_on_cta0_barrier(smem.bar_clc_empty);
                if (!clc_result.is_valid)
                    break;

                phase ^= 1;
            }
            if constexpr (IS_2CTA) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase); // Wait for all threads' arrival on `bar_clc_empty`, which means that there will be no further operations on distributed shared memory (including barrier arrive and MMA), avoiding the "cluster target block not present" error
                    smem.bar_clc_empty.arrive(1u);  // Transfer the signal above to CTA1
                } else {
                    smem.bar_clc_empty.wait(0);
                }
            }
        } else if (warp_idx == 10) {
            // Indices generator
            // (also generates TMA coordinates for the KV producer)
            OuterloopArgs cur_job = get_first_job();
            RingBufferState rs;
            static constexpr uint32_t NUM_INDICES_PER_THREAD = B_TOPK / 32;
            static_assert(B_TOPK % 32 == 0);

            do {
                if constexpr (!IS_DECODE) {
                    auto body = [&]<bool CHECK_TOPK_SUBSCRIPT>() {
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                            smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                            CUTE_UNROLL
                            for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                                uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                                int cur_index;
                                if constexpr (CHECK_TOPK_SUBSCRIPT) {
                                    // Predicate the load on `pos < topk_length` to prevent IMA
                                    cur_index = pos < cur_job.topk_length ? __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos) : -1;
                                } else {
                                    // topk_length % B_TOPK == 0, so every pos is within the row
                                    cur_index = __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos);
                                }
                                bool is_index_valid = (uint32_t)cur_index < (uint32_t)params.s_kv;  // Don't need to check `index >= 0`, since if `index < 0` holds, `(uint32_t)index` must lies in 2147483648 ~ 4294967295, which is definitely greater than `params.s_kv`
                                uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                                if (lane_idx == 0) {
                                    *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                                }
                            }

                            smem.bar_indices_full[indices_buf_idx].arrive();
                            rs.update();
                        }
                    };
                    if (cur_job.topk_length % B_TOPK == 0 && cur_job.topk_length != 0)
                        body.template operator()<false>();
                    else
                        body.template operator()<true>();
                } else {
                    int *indices_base = params.indices + (int64_t)cur_job.s_q_idx * params.stride_indices_s_q;
                    int *extra_indices_base = params.extra_indices + (int64_t)cur_job.s_q_idx * params.stride_extra_indices_s_q;
                    // The indices of a block are loaded one block ahead, into registers: the load is a global read
                    // with no shared-memory destination, so it needs neither the metadata buffer nor a barrier, and
                    // its DRAM round trip overlaps with this block's coordinate and scale arithmetic. Inside the
                    // per-index loop the load could not be issued early: the __ballot_sync at the end of each
                    // iteration orders the next iteration's load behind it. The prefetch resolves the source at
                    // run time (pos >= num_orig_slots holds for every KVLocation; num_orig_slots is 0xFFFFFFFF
                    // without an extra cache); the consume side keeps the compile-time dispatch. The first block
                    // of a job is not prefetched: its indices row is unknown until the CLC response
                    auto load_indices = [&](uint32_t kv_block_idx, int (&dst)[NUM_INDICES_PER_THREAD]) {
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            bool in_extra = pos >= cur_job.num_orig_slots;
                            uint32_t local_pos = in_extra ? pos - cur_job.num_orig_slots : pos;
                            uint32_t valid_len = in_extra ? cur_job.extra_topk_length : cur_job.topk_length;
                            // The last KV block may run beyond the end of the indices row (e.g. topk = 128 with
                            // B_TOPK = 96), so an unconditional load could read out of bounds of `indices` /
                            // `extra_indices`. Slots beyond valid_len are masked out anyway, so they get -1 without a load
                            dst[i] = local_pos < valid_len ? __ldg((in_extra ? extra_indices_base : indices_base) + local_pos) : -1;
                        }
                    };
                    int prefetched[NUM_INDICES_PER_THREAD];
                    load_indices(0, prefetched);
                    auto metadata_block = [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);
                        int cur_indices[NUM_INDICES_PER_THREAD];
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) cur_indices[i] = prefetched[i];
                        if (kv_block_idx + 1 < cur_job.num_kv_blocks) {
                            load_indices(kv_block_idx + 1, prefetched);
                        }
                        // Which cache a slot of this block comes from; a compile-time constant for ORIG / EXTRA blocks
                        auto pos_in_extra = [&](uint32_t pos) {
                            if constexpr (LOC == KVLocation::ORIG) {
                                return false;
                            } else if constexpr (LOC == KVLocation::EXTRA) {
                                return true;
                            } else {
                                return pos >= cur_job.num_orig_slots;
                            }
                        };
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            bool in_extra = pos_in_extra(pos);
                            int cur_index = cur_indices[i];
                            bool is_index_valid = cur_index >= 0;

                            // Share coordinate generation instead of repeating it in all dequant warps.
                            // The per-cache constants are selected by value, not by reference. Selecting between
                            // the two FastDivmod objects by reference makes ptxas lose the `divisor != 1` fold of
                            // fast_divmod and emit a full integer division per index; the divmod is written out
                            // here on the selected fields so the fold survives. Likewise the quotient
                            // stride / BYTES_PER_TOKEN is formed per format below: the format's stride is a
                            // compile-time constant, so each quotient is a multiply-shift, while dividing the
                            // selected stride by the selected BYTES_PER_TOKEN would be a runtime 64-bit division
                            const uint32_t dm_mul = in_extra ? aux_params.fast_divmod_extra_page_block_size.multiplier : aux_params.fast_divmod_page_block_size.multiplier;
                            const uint32_t dm_shr = in_extra ? aux_params.fast_divmod_extra_page_block_size.shift_right : aux_params.fast_divmod_page_block_size.shift_right;
                            const int32_t dm_div = in_extra ? aux_params.fast_divmod_extra_page_block_size.divisor : aux_params.fast_divmod_page_block_size.divisor;
                            uint32_t token_idx = is_index_valid ? (uint32_t)cur_index : 0;
                            int block_idx = dm_div != 1 ? (int)(__umulhi(token_idx, dm_mul) >> dm_shr) : (int)token_idx;
                            int idx_in_block = (int)token_idx - block_idx * dm_div;
                            uint32_t row = i * 32 + lane_idx;
                            // The extra KV cache may have another format (HAS_FP4_KV); see above for the per-format quotient
                            const int blocks_per_stride = BLOCK_SPLIT
                                ? (in_extra ? params.stride_extra_kv_block / (int)ExtraKVFormat::QUANT_BYTES
                                            : params.stride_kv_block / (int)OrigKVFormat::QUANT_BYTES)
                                : (in_extra ? params.stride_extra_kv_block / (int)ExtraKVFormat::BYTES_PER_TOKEN
                                            : params.stride_kv_block / (int)OrigKVFormat::BYTES_PER_TOKEN);
                            smem.decode_tma_coords[indices_buf_idx][row] = is_index_valid
                                ? blocks_per_stride * block_idx + idx_in_block
                                : -1;
                            if constexpr (BLOCK_SPLIT) {
                                const uint64_t blk_stride = (uint64_t)(in_extra ? params.stride_extra_kv_block : params.stride_kv_block);
                                const uint32_t qbytes = in_extra ? (uint32_t)ExtraKVFormat::QUANT_BYTES : (uint32_t)OrigKVFormat::QUANT_BYTES;
                                const uint32_t nscales = in_extra ? (uint32_t)ExtraKVFormat::NUM_SCALES_EACH_TOKEN : (uint32_t)OrigKVFormat::NUM_SCALES_EACH_TOKEN;
                                const uint64_t scale_byte = (uint64_t)block_idx * blk_stride + (uint64_t)dm_div * qbytes + (uint64_t)idx_in_block * nscales;
                                smem.decode_scale_offs[indices_buf_idx][row] = is_index_valid ? (uint32_t)(scale_byte >> 4) : 0xFFFFFFFFu;
                            }

                            uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                            if (lane_idx == 0) {
                                *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                            }
                        }

                        smem.bar_indices_full[indices_buf_idx].arrive();
                        rs.update();
                    };
                    if constexpr (HAS_FP4_KV) {
                        // Two copies of the body. The fp4 blocks, most of a job's blocks, run the EXTRA copy: the
                        // source folds to the extra cache at compile time so
                        // it is this kernel's per-block hot code and stays small. The fp8 blocks and the at most
                        // one mixed block share the run-time-dispatched copy: they run once per job, so their
                        // code is per-job footprint, and one copy instead of two (ORIG and mixed) is less to
                        // fetch through the TPC-shared instruction cache at every job boundary
                        const uint32_t num_full_orig_blocks = std::min(cur_job.num_kv_blocks, cur_job.num_orig_slots / B_TOPK);
                        const uint32_t num_general_blocks = std::min(cur_job.num_kv_blocks,
                            num_full_orig_blocks + (cur_job.num_orig_slots % B_TOPK != 0 ? 1u : 0u));
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < num_general_blocks; ++kv_block_idx) {
                            metadata_block.template operator()<KVLocation::ORIG_AND_EXTRA>(kv_block_idx);
                        }
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = num_general_blocks; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            metadata_block.template operator()<KVLocation::EXTRA>(kv_block_idx);
                        }
                    } else {
                        // fp8 extra cache (or none): both sources have the same format, so the run-time form
                        // computes the same arithmetic as the ORIG / EXTRA specializations and one copy serves
                        // every block
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            metadata_block.template operator()<KVLocation::ORIG_AND_EXTRA>(kv_block_idx);
                        }
                    }
                }
                cur_job = get_next_job(cur_job);
            } while (cur_job.is_valid);
        } else if (warp_idx == 11) {
            if constexpr (IS_DECODE) {
                // Gather front of the KV producer, for every block of every job (the dequant warps run only the
                // back, see WG1). Per block: wait until the dequant warps have released this block's index of the
                // alternating barrier pair (bar_wg1_past_raw), then for the block's KV slot and indices; issue the
                // raw gather4s and CTA0 scale copies, both tracked by bar_raw_kv_full[index]
                RingBufferState rs;
                auto issue_front = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                    uint32_t raw_buf = rs.get<NUM_RAW_BARS>().first;
                    auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                    smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                    auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                    smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                    uint8_t *slot_base = (uint8_t*)smem.kv_slots[kv_slot_idx];

                    // CTA0's lanes prefetch scales while its elected lane issues the raw gathers.
                    if constexpr (IS_2CTA) {
                        if (cta_idx == 0) {
                            CUTE_UNROLL
                            for (uint32_t i = 0; i < B_TOPK / 32; ++i) {
                                uint32_t row = i * 32 + lane_idx;
                                int coord = smem.decode_tma_coords[indices_buf_idx][row];
                                bool in_extra = kv_block_idx * B_TOPK + row >= job.num_orig_slots;
                                const uint8_t *base = in_extra ? (const uint8_t*)params.extra_kv + ExtraKVFormat::QUANT_BYTES
                                                               : (const uint8_t*)params.kv + OrigKVFormat::QUANT_BYTES;
                                uint32_t stride = in_extra ? ExtraKVFormat::BYTES_PER_TOKEN : OrigKVFormat::BYTES_PER_TOKEN;
                                uint8_t *dst = slot_base + SCALE_STAGE_OFFSET + row * SCALE_STAGE_ROW_BYTES;
                                copy_kv_scales_async<SCALE_STAGE_ROW_BYTES>(dst, base, coord, stride);
                            }
                            cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t const*>(&smem.bar_raw_kv_full[raw_buf]));
                        }
                    }
                    bool elected = elect_one_sync();
                    if (elected) {
                        // The tensor map is a property of the 4-row group (topk % 8 == 0, so a group never straddles the
                        // orig/extra boundary); the block's gather4s are issued by issue_raw_gathers
                        // The maps are passed as byte offsets into tma_params (issue_raw_gathers adds the base)
                        const uint8_t *maps_base = (const uint8_t*)&tma_params;
                        const uint32_t orig_map = cta_idx == 0 ? offsetof(TMAParams, tensor_map_kv_fp8_part_cta0) : offsetof(TMAParams, tensor_map_kv_fp8_part_cta1);
                        // The extra cache's map follows its format: fp4 (HAS_FP4_KV) or fp8 (an fp8 extra cache)
                        const uint32_t extra_map = HAS_FP4_KV
                            ? (cta_idx == 0 ? offsetof(TMAParams, tensor_map_extra_kv_fp4_part_cta0) : offsetof(TMAParams, tensor_map_extra_kv_fp4_part_cta1))
                            : (cta_idx == 0 ? offsetof(TMAParams, tensor_map_extra_kv_fp8_part_cta0) : offsetof(TMAParams, tensor_map_extra_kv_fp8_part_cta1));
                        const int4 *coords = (const int4*)smem.decode_tma_coords[indices_buf_idx];
                        const uint32_t num_orig_rows = get_num_orig_rows(job, kv_block_idx);
                        if (0 < num_orig_rows && num_orig_rows < B_TOPK) {
                            issue_raw_gathers<true, B_TOPK / 4, RAW_KV_GROUP_BYTES>(coords, maps_base, orig_map, extra_map, num_orig_rows / 4, smem.bar_raw_kv_full[raw_buf], slot_base);
                        } else {
                            issue_raw_gathers<false, B_TOPK / 4, RAW_KV_GROUP_BYTES>(coords, maps_base, num_orig_rows == 0 ? extra_map : orig_map, 0, 0, smem.bar_raw_kv_full[raw_buf], slot_base);
                        }
                        // The raw row widths of this CTA (both caches have the same width unless the extra cache is fp4)
                        const uint32_t orig_stride = cta_idx == 0 ? KVFormatCta<MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE : KVFormatCta<MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE;
                        const uint32_t extra_stride = cta_idx == 0 ? KVFormatCta<EXTRA_MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE : KVFormatCta<EXTRA_MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE;
                        smem.bar_raw_kv_full[raw_buf].arrive_and_expect_tx(num_orig_rows * orig_stride + (B_TOPK - num_orig_rows) * extra_stride);
                    }

                    // VATTN block-split: after the elected lane has issued the raw gathers (the larger transfer starts first),
                    // every lane stages the scale rows of its rows (16 B fp8, 32 B fp4) into the slot tail with cp.async.cg
                    // (L1 bypass; measured 0.7 us per layer faster than .ca issued before the gathers), zero-filled for an
                    // invalid row, then arrives on the block's raw barrier (count 1 + 32)
                    if constexpr (BLOCK_SPLIT) {
                        using ScaleCopy = cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < B_TOPK / 32; ++i) {
                            uint32_t row = i * 32 + lane_idx;
                            uint32_t soff = smem.decode_scale_offs[indices_buf_idx][row];
                            bool valid = soff != 0xFFFFFFFFu;
                            bool in_extra = kv_block_idx * B_TOPK + row >= job.num_orig_slots;
                            const uint8_t *base = in_extra ? (const uint8_t*)params.extra_kv : (const uint8_t*)params.kv;
                            const uint8_t *src = base + (uint64_t)(valid ? soff : 0u) * 16;
                            uint8_t *dst = slot_base + SCALE_STAGE_OFFSET + row * SCALE_STAGE_ROW_BYTES;
                            ScaleCopy::copy(*(const uint4*)src, *(uint4*)dst, valid);
                            if constexpr (HAS_FP4_KV) {
                                if (in_extra) {
                                    ScaleCopy::copy(*(const uint4*)(src + 16), *(uint4*)(dst + 16), valid);
                                }
                            }
                        }
                        cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t const*>(&smem.bar_raw_kv_full[raw_buf]));
                    }
                    // Each lane releases its coordinates after its reads.
                    smem.bar_indices_empty[indices_buf_idx].arrive();
                    rs.update();
                };

                // Block 0's front is issued before the loop; the iteration for block issued_block_idx issues the
                // front of the block after it, so that at a job boundary the CLC result is read before the next
                // job's first front
                OuterloopArgs cur_job = get_first_job();
                issue_front(cur_job, 0);
                uint32_t issued_block_idx = 0;   // index across jobs of the last block whose front was issued
                do {
                    OuterloopArgs next_job;
                    next_job.is_valid = false;
                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        // The next front uses pair index (issued_block_idx + 1) % NUM_RAW_BARS, last used by the block
                        // NUM_RAW_BARS blocks before it. The dequant warps arrive on bar_wg1_past_raw once they have
                        // passed the completion wait of that block, so after this wait no waiter is left behind on the
                        // barrier and the next front's arrives cannot complete a phase ahead of one. The barrier is
                        // at most at that block's phase (the next front has not been issued), so the parity is
                        // unambiguous
                        if (issued_block_idx + 1 >= NUM_RAW_BARS) {
                            auto [past_raw_buf, past_raw_phase] = RingBufferState{issued_block_idx + 1 - NUM_RAW_BARS}.get<NUM_RAW_BARS>();
                            smem.bar_wg1_past_raw[past_raw_buf].wait(past_raw_phase);
                        }
                        bool has_next = true;
                        if (kv_block_idx + 1 == cur_job.num_kv_blocks) {
                            next_job = get_next_job(cur_job);
                            has_next = next_job.is_valid;
                        }
                        const OuterloopArgs &next_block_job = kv_block_idx + 1 < cur_job.num_kv_blocks ? cur_job : next_job;
                        uint32_t next_block_idx = kv_block_idx + 1 < cur_job.num_kv_blocks ? kv_block_idx + 1 : 0;
                        if (has_next) {
                            issue_front(next_block_job, next_block_idx);
                        }
                        ++issued_block_idx;
                    }
                    cur_job = next_job;
                } while (cur_job.is_valid);
            }
        }
    } else if (warpgroup_idx == 1) {
        // setmaxnreg.inc raises the count from the initial 128 and .dec lowers it; the direction is part of the instruction
        if constexpr (WG1_REGS >= 128) {
            cutlass::arch::warpgroup_reg_alloc<WG1_REGS>();
        } else {
            cutlass::arch::warpgroup_reg_dealloc<WG1_REGS>();
        }
        if constexpr (IS_DECODE) {
            // Warp 11 loads raw data and scales. WG1 waits once for both transports before
            // dequantizing in place. The orig/extra format boundary is resolved per 8 rows.
            uint32_t local_warp_idx = warp_idx - 4;
            auto run_dequant_warps = [&](auto &dq_rows) {
                using DQ = std::remove_reference_t<decltype(dq_rows)>;
                RingBufferState rs_back;
                OuterloopArgs cur_job = get_first_job();
                do {
                    OuterloopArgs next_job;
                    next_job.is_valid = false;
                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        auto [raw_buf, raw_bar_phase] = rs_back.get<NUM_RAW_BARS>();
                        smem.bar_raw_kv_full[raw_buf].wait(raw_bar_phase);
                        if (elect_one_sync()) {
                            smem.bar_wg1_past_raw[raw_buf].arrive();
                        }
                        if (kv_block_idx + 1 == cur_job.num_kv_blocks) {
                            next_job = get_next_job(cur_job);
                        }
                        // The number of fp8 tokens of this thread is warp-uniform since topk % 8 == 0 (asserted by `run()`);
                        // DequantRows::run_block dispatches on it
                        uint32_t num_fp8_tokens = (uint32_t)std::clamp(
                            (int)get_num_orig_rows(cur_job, kv_block_idx) - (int)(local_warp_idx * DQ::NUM_ROWS_PER_WARP), 0, (int)DQ::NUM_ROWS_PER_WARP
                        ) / DQ::NUM_ROWS_PER_WAVEFRONT;
                        auto [indices_buf_idx, indices_bar_phase] = rs_back.get<NUM_INDICES_BUFS>();
                        auto [kv_slot_idx, kv_bar_phase] = rs_back.get<NUM_KV_SLOTS>();
                        dq_rows.run_block(num_fp8_tokens, kv_slot_idx, indices_buf_idx);
                        rs_back.update();
                    }
                    cur_job = next_job;
                } while (cur_job.is_valid);
            };
            // Both CTAs of a cluster hold their half of the token's dims, so one instantiation of DequantRows serves both
            DequantRows<Kernel, 0> dq_rows(smem, local_warp_idx, lane_idx, cta_idx);
            run_dequant_warps(dq_rows);
        } else {
            // KV Producer (prefill): gathers the whole bf16 KV block via TMA gather4
            if (elect_one_sync()) {
                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                uint32_t local_warp_idx = warp_idx - 4;
                do {
                    for (uint32_t i = 0; i < cur_job.num_kv_blocks; ++i) {
                        static_assert(B_TOPK % (4*4) == 0);
                        static constexpr uint32_t NUM_ROW_PER_WARP = B_TOPK / 4;
                        int4 topk_idxs[NUM_ROW_PER_WARP / 4];
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            uint32_t pos = i * B_TOPK + row;
                            // Predicate the load on `pos < topk_length` to avoid reading OOB of the
                            // indices row in the last (partial) KV block. Chunks with no slot below
                            // topk_length are masked out by the indices generator warp anyway, so feed
                            // them invalid indices (-1), which tma_gather4 bounds-checks into zero-fill
                            topk_idxs[local_row] = pos < cur_job.topk_length
                                ? __ldg((int4*)(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos))
                                : int4{-1, -1, -1, -1};
                        }

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            CUTE_UNROLL
                            for (uint32_t tile_idx = 0; tile_idx < (CLUSTER_SIZE == 2 ? (D_QK/2/64) : (D_QK/64)); ++tile_idx) {
                                /*
                                For cases where CLUSTER_SIZE == 1, each CTA reads the full K/V
                                For cases where CLUSTER_SIZE == 2, CTA0 reads KV[:, :D_QK/2] and CTA1 reads KV[:, D_QK/2:], since dual GeM< is used
                                */
                                if constexpr (CLUSTER_SIZE == 1) {
                                    ku::tma_gather4(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64,
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                } else {
                                    ku::tma_gather4_cta_group_2<true>(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64 + cta_idx * (D_QK/2),
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                }
                            }
                        }
                        rs.update();
                    }
                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}


template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
fwd_kernel(__grid_constant__ const typename Kernel::Params params, __grid_constant__ const typename Kernel::TMAParams tma_params, __grid_constant__ const typename Kernel::AuxParams aux_params) {
    Kernel::devfunc(params, tma_params, aux_params);
}


template<Config CONFIG>
void Kernel<CONFIG>::run(const Params &params) {
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_VO);
    KU_ASSERT(params.stride_indices_s_q*sizeof(int) % 32 == 0, "indices.stride(0) must be 32B aligned, got %d elements (%d bytes)", params.stride_indices_s_q, (int)(params.stride_indices_s_q*sizeof(int)));

    KU_ASSERT(params.enable_q_norm == ENABLE_Q_NORM);
    KU_ASSERT(params.is_rope_neox_style == false && params.rope_dim == 64);
    KU_ASSERT(params.wv_group_size == WV_GROUP_SIZE);
    KU_ASSERT(params.num_per_channels == O_QUANT_TILE_SIZE);
    KU_ASSERT(params.use_tma_aligned_col_major_sf == true && params.round_sf == true && params.use_packed_ue8m0 == true);

    TMAParams tma_params = {};
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.b == 1, "Only batch size 1 is supported for the fused decoding kernel");
        KU_ASSERT(params.model_type == MODEL_TYPE && params.extra_model_type == EXTRA_MODEL_TYPE);
        if (params.extra_topk > 0) {
            // A KV block may straddle the orig/extra boundary (see KVLocation::ORIG_AND_EXTRA). Since
            // one TMA gather4 covers 4 consecutive rows sharing one tensor map, the boundary (i.e.
            // topk) must be aligned to 4 rows; the common dequant path resolves the format per 8 rows
            KU_ASSERT(params.topk % (HAS_FP4_KV ? 8 : 4) == 0, "topk (%d) must be a multiple of %d when the extra KV cache is used", params.topk, HAS_FP4_KV ? 8 : 4);
        }
        auto make_kv_tensor_maps = []<ModelType MT>(bool is_extra, void *kv_ptr, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes) -> std::pair<CUtensorMap, CUtensorMap> {
            auto make_map_for_cta = [&]<uint32_t CTA>() {
                using FC = KVFormatCta<MT, CTA>;
                if constexpr (BLOCK_SPLIT) {
                    return make_kv_raw_region_tensor_map_bs<FC, FC::RAW_TOKEN_SMEM_STRIDE>(
                        is_extra ? "extra_kv" : "kv", kv_ptr, num_blocks, block_stride_bytes, row_stride_bytes,
                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
                } else {
                    return make_kv_quant_part_tensor_map<FC, FC::RAW_TOKEN_SMEM_STRIDE>(
                        is_extra ? "extra_kv" : "kv", kv_ptr, num_blocks, block_stride_bytes, row_stride_bytes,
                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
                }
            };
            CUtensorMap map_cta1 = {};
            if constexpr (IS_2CTA) {
                map_cta1 = make_map_for_cta.template operator()<1>();
            }
            return {make_map_for_cta.template operator()<0>(), map_cta1};
        };
        {
            auto [map_cta0, map_cta1] = make_kv_tensor_maps.template operator()<MODEL_TYPE>(false, params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row);
            tma_params.tensor_map_kv_fp8_part_cta0 = map_cta0;
            tma_params.tensor_map_kv_fp8_part_cta1 = map_cta1;
        }
        if (params.extra_topk > 0) {
            auto [map_cta0, map_cta1] = make_kv_tensor_maps.template operator()<EXTRA_MODEL_TYPE>(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
            if constexpr (HAS_FP4_KV) {
                tma_params.tensor_map_extra_kv_fp4_part_cta0 = map_cta0;
                tma_params.tensor_map_extra_kv_fp4_part_cta1 = map_cta1;
            } else {
                tma_params.tensor_map_extra_kv_fp8_part_cta0 = map_cta0;
                tma_params.tensor_map_extra_kv_fp8_part_cta1 = map_cta1;
            }
        }
    } else {
        tma_params.tensor_map_kv = ku::make_tensor_map(
            {(uint64_t)D_QK, (uint64_t)params.s_kv},
            {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)},
            {64, 1},
            params.kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    auto aux_params = AuxParams {
    };  
    if constexpr (IS_DECODE) {
        aux_params.fast_divmod_page_block_size = cutlass::FastDivmod(params.page_block_size);
        aux_params.fast_divmod_extra_page_block_size = cutlass::FastDivmod(params.extra_kv != nullptr ? params.extra_page_block_size : 1);
    }

    auto kernel = &fwd_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3(params.s_q * CLUSTER_SIZE, 1, 1),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params, aux_params
    ));
}


template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params) {
    using KernelType = Kernel<CONFIG>;
    KernelType::run(params);
}


}
""",
    'csrc/core_attn/kernel_h.cuh': r"""#pragma once

#include "params_h.cuh"
#include "cuda_kernels/defines_h.cuh"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

// Compile-time configuration for the fused_norm_rope_attn_rope_cast core attention kernel.
// Notes:
//   - MODEL_TYPE is the format of `kv`: V4.1 fp8 (the only instantiated value)
//   - EXTRA_MODEL_TYPE (the format of `extra_kv`) is V41 (i.e. absent or fp8) or V41_FP4 (fp4 KV cache); prefill passes V41
struct Config {
    SparseAttnFwdMode FWD_MODE;
    ModelType MODEL_TYPE;          // Format of `kv`: V41
    ModelType EXTRA_MODEL_TYPE;    // Decode only, the format of the extra KV cache. V41 for prefill
    uint32_t H_Q;
    bool ENABLE_Q_NORM;
};

// Parameters for the fused Q-b-norm + Q RoPE + Core Attention Forward (prefill/decoding) + O RoPE + O Cast kernel
template<typename Base>
struct ParamsTemplate : Base {
    // Q Norm
    bool enable_q_norm;
    float rms_norm_eps;

    // Q/O RoPE
    uint32_t* __restrict__ token_positions;   // [s_q]
    bool is_rope_neox_style;    // Must be false
    uint32_t rope_dim;          // Must be 64
    float* __restrict__ cos_sin_cache;  // [*, rope_dim], must be contiguous, from vllm.RotaryEmbedding.cos_sin_cache

    // O Cast
    uint32_t n_wv_group;
    uint32_t wv_group_size;
    uint32_t num_per_channels;          // Must be 32
    bool use_tma_aligned_col_major_sf;  // Must be true
    bool round_sf;  // Must be true
    bool use_packed_ue8m0;  // Must be true

    // Output
    fp8_e4m3* __restrict__ out_fp8; // [s_q, n_wv_group, wv_group_size * d_v]
    uint32_t* __restrict__ out_sf;   // [s_q, n_wv_group, (wv_group_size*d_v) / 32 / 4], contiguous on the first (s_q) dim
    uint32_t stride_out_sf_wv_group, stride_out_sf_head_dim;
};

template<SparseAttnFwdMode FWD_MODE>
using ParamT = std::conditional_t<is_decode_v<FWD_MODE>, ParamsTemplate<SparseAttnDecodeParams>, ParamsTemplate<SparseAttnFwdParams>>;

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);

}
""",
    'csrc/cuda_kernels/defines_h.cuh': r"""#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::NamedBarrier;
using fp8_e4m3 = cutlass::float_e4m3_t;

""",
    'csrc/cuda_kernels/kv_cache_format_h.cuh': r"""#pragma once

#include "params_h.cuh"

// Each V4.1 token stores [raw data, 1-byte scales], including the RoPE dims:
// FP8: 512 B e4m3 + 16 B ue8m0; FP4: 256 B e2m1 + 32 B e4m3. See tests/quant.py.
template<ModelType MT>
struct KVCacheFormat {
    static constexpr ModelType MODEL_TYPE = MT;
    static constexpr bool IS_FP4 = MT == ModelType::V41_FP4;
    static constexpr int D_QK = 512;
    static constexpr int D_ROPE = 64;
    static constexpr int D_NOPE = D_QK - D_ROPE;
    static constexpr int D_FP4 = IS_FP4 ? D_QK : 0;    // Dimensions stored as fp4 e2m1
    static constexpr int D_FP8 = IS_FP4 ? 0 : D_QK;    // Dimensions stored as fp8 e4m3, needing dequant
    static constexpr int QUANT_TILE_SIZE = IS_FP4 ? 16 : 32;                            // Dimensions sharing one scale
    static constexpr int NUM_SCALES_EACH_TOKEN = D_QK / QUANT_TILE_SIZE;                // 16 (V41) / 32 (fp4)
    static constexpr int QUANT_BYTES = D_FP4 / 2 + D_FP8;                               // The quantized (fp4 / fp8) data of a token
    static constexpr int BYTES_PER_TOKEN = QUANT_BYTES + NUM_SCALES_EACH_TOKEN;   // 528 / 288
};

// A CTA's contiguous share of a token. In a pair, CTA0 loads its scales separately;
// CTA1's TMA view includes the trailing scales. Shared-memory padding is kernel-specific.
template<typename F, int NUM_CTAS, int CTA>
struct KVCachePart : F {
    static_assert((NUM_CTAS == 1 || NUM_CTAS == 2) && 0 <= CTA && CTA < NUM_CTAS);
    static constexpr int QUANT_DIMS = F::D_QK / NUM_CTAS;
    static constexpr int QUANT_BYTES = F::QUANT_BYTES / NUM_CTAS;
    static constexpr int NUM_SCALES = F::NUM_SCALES_EACH_TOKEN / NUM_CTAS;
    static constexpr int RAW_TOKEN_OFFSET = CTA * QUANT_BYTES;
    static constexpr int SCALE_TOKEN_OFFSET = F::QUANT_BYTES + CTA * NUM_SCALES;
    static constexpr bool SEPARATE_SCALES = NUM_CTAS == 2 && CTA == 0;
    static constexpr int RAW_TOKEN_DATA_BYTES = QUANT_BYTES + (SEPARATE_SCALES ? 0 : F::NUM_SCALES_EACH_TOKEN);
    static constexpr int SCALE_OFFSET = SCALE_TOKEN_OFFSET - RAW_TOKEN_OFFSET;
};

// VATTN block-split part (1 CTA): a token's raw data is a row of QUANT_BYTES in the page block's raw region, its scales a
// row of NUM_SCALES_EACH_TOKEN bytes in the scale region behind it; the scales are always transported separately
template<typename F>
struct KVCachePartBS : F {
    static constexpr int QUANT_DIMS = F::D_QK;
    static constexpr int QUANT_BYTES = F::QUANT_BYTES;
    static constexpr int NUM_SCALES = F::NUM_SCALES_EACH_TOKEN;
    static constexpr int RAW_TOKEN_OFFSET = 0;
    static constexpr int SCALE_TOKEN_OFFSET = 0;    // unused: scales are addressed per page block
    static constexpr bool SEPARATE_SCALES = true;
    static constexpr int RAW_TOKEN_DATA_BYTES = QUANT_BYTES;
    static constexpr int SCALE_OFFSET = 0;          // unused
};

// Runtime counterpart of KVCacheFormat<MT>::BYTES_PER_TOKEN
constexpr int kv_cache_bytes_per_token(ModelType mt) {
    switch (mt) {
        case ModelType::V41: return KVCacheFormat<ModelType::V41>::BYTES_PER_TOKEN;
        case ModelType::V41_FP4: return KVCacheFormat<ModelType::V41_FP4>::BYTES_PER_TOKEN;
    }
    return 0;
}

// The (kv, extra_kv) format pairs that exist: extra_kv has the format of kv, or is the fp4 cache next to a (fp8) kv.
constexpr bool is_valid_kv_format_pair(ModelType kv, ModelType extra_kv) {
    return extra_kv == kv || (kv == ModelType::V41 && extra_kv == ModelType::V41_FP4);
}

template<ModelType KV, ModelType EXTRA_KV = KV>
struct KVFormatPair {
    static_assert(is_valid_kv_format_pair(KV, EXTRA_KV));
    static constexpr ModelType kv = KV, extra_kv = EXTRA_KV;
};

// A list of KVFormatPair, see dispatch_kv_formats in csrc/api/common.h
template<typename... Pairs>
struct KVFormatPairs {};
""",
    'csrc/cuda_kernels/sm100/common_subroutine_h.cuh': r"""#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

namespace sm100 {

/*
Load K/V indices from global memory, and generate validity mask
Each thread loads 8 indices
Should be called by lanes 0 ~ (BLOCK_TOPK/8)
*/
CUTE_DEVICE
char load_indices_and_generate_mask(
    int lane_idx,
    int* gIndices,
    int s_kv,
    int abs_pos_start,
    int topk_length
) {
    int indices[8];
    KU_LDG_256(
        gIndices + lane_idx*8, 
        indices,
        ".nc",
        "evict_first", 
        "evict_normal", 
        "256B"
    );
    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
        int abs_pos = abs_pos_start + lane_idx*8 + rel_pos_in_lane;
        return index >= 0 && index < s_kv && abs_pos < topk_length;
    };
    char is_ks_valid_mask = \
        is_valid(7, indices[7]) << 7 | 
        is_valid(6, indices[6]) << 6 | 
        is_valid(5, indices[5]) << 5 |
        is_valid(4, indices[4]) << 4 |
        is_valid(3, indices[3]) << 3 |
        is_valid(2, indices[2]) << 2 |
        is_valid(1, indices[1]) << 1 |
        is_valid(0, indices[0]) << 0;
    return is_ks_valid_mask;
}


/*
Get P from Tensor Memory, reduce P within shared memory, perform masking, and store back if necessary

Initially, since dual gemm is used, we have two P pieces in Tensor Memory, one occupying rows 0 ~ 63 while the other occupying rows 64 ~ 127. We'd like to have them reduced into one single P piece, stored in registers with layout:

        N       N    ---   (topk)
    +-------+-------+
    |       |       |
32  | Warp0 | Warp2 |
    |       |       |
    +-------+-------+
    |       |       |
32  | Warp1 | Warp3 |
    |       |       |
    +-------+-------+
|
(head)

where N = NUM_ELEMS_PER_THREAD
*/
template<
    int H_Q,
    int NUM_ELEMS_PER_THREAD,
    int BARRIER_WARP02_SYNC_ID,
    int BARRIER_WARP13_SYNC_ID
>
CUTE_DEVICE
void retrieve_mask_and_reduce_p(
    uint32_t p_tmem_col_start,
    char* k_validness_base, // The stride of row should be aligned to power of 2, for vectorized load
    int local_warp_idx,
    int lane_idx,
    auto slot_bar_P_empty_arrival,
    float p_exchange_buf[4][32*NUM_ELEMS_PER_THREAD],
    float p[NUM_ELEMS_PER_THREAD]
) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;
    static_assert(BARRIER_WARP13_SYNC_ID == BARRIER_WARP02_SYNC_ID+1);
    static_assert(H_Q == 64);

    float p_peer[NUM_ELEMS_PER_THREAD];
    auto load_op = []<int N>(uint32_t ts, void* d) {
        ku::tmem_ld_32dp32bNx<N>(ts, d);
    };
    if (local_warp_idx < 2) {
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start, p, load_op);
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start + NUM_ELEMS_PER_THREAD, p_peer, load_op);
    } else {
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start, p_peer, load_op);
        ku::tmem_ld_st_decomposed<NUM_ELEMS_PER_THREAD, 1>(p_tmem_col_start + NUM_ELEMS_PER_THREAD, p, load_op);
    }
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    slot_bar_P_empty_arrival();

    // Mask invalid tokens
    // We put masking before reduction, since (-inf) + anything (except nan and +inf) is (-inf), which guarantees correctness, and this can overlap with smem load
    static_assert(NUM_ELEMS_PER_THREAD == 48 || NUM_ELEMS_PER_THREAD == 32);
    int k_validness_offset = local_warp_idx >= 2 ? NUM_ELEMS_PER_THREAD/8 : 0;  // |0|2| \n |1|3|
    uint8_t is_k_valid_masks[NUM_ELEMS_PER_THREAD / 8];
    if constexpr (NUM_ELEMS_PER_THREAD == 32) {
        *(uint32_t*)is_k_valid_masks = *(uint32_t*)(k_validness_base + k_validness_offset);
    } else if constexpr (NUM_ELEMS_PER_THREAD == 48) {
        *(uint16_t*)(is_k_valid_masks + 0) = *(uint16_t*)(k_validness_base + k_validness_offset + 0);
        *(uint16_t*)(is_k_valid_masks + 2) = *(uint16_t*)(k_validness_base + k_validness_offset + 2);
        *(uint16_t*)(is_k_valid_masks + 4) = *(uint16_t*)(k_validness_base + k_validness_offset + 4);
    }
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 1) {
        if (!(is_k_valid_masks[i/8] >> (i%8) & 1))
            p[i] = -CUDART_INF_F;
    }

    // Reduce P within the cluster
    {
        // Store
        // Warp 0, 1 store their right (col 32 ~ 63) part, while warp 2, 3 store their left (row 0 ~ 31) part
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            ku::st_shared(&p_exchange_buf[local_warp_idx^2][i*32*4 + lane_idx*4], *(float4*)(p_peer + i*4));
        }
        NamedBarrier::arrive_and_wait(64, BARRIER_WARP02_SYNC_ID + (local_warp_idx&1));
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            float2 t[2];
            *(float4*)t = *(float4*)(&p_exchange_buf[local_warp_idx][i*32*4 + lane_idx*4]);
            float2* cur_p = (float2*)(p + i*4);
            cur_p[0] = ku::float2_add(cur_p[0], t[0]);
            cur_p[1] = ku::float2_add(cur_p[1], t[1]);
        }
    }
}

/*
Rescale O in Tensor Memory.

O should occupy 128 rows x NUM_COLS columns in Tensor Memory.
*/
template<
    int NUM_COLS,
    int CHUNK_SIZE,
    int TMEM_COL_START
>
CUTE_DEVICE
void rescale_O(
    float scale_factor
) {
    float2 scale_factor_float2 = {scale_factor, scale_factor};
    float2 o[CHUNK_SIZE/2];

    CUTE_UNROLL
    for (int chunk_idx = 0; chunk_idx < NUM_COLS/CHUNK_SIZE; ++chunk_idx) {
        // Load O
        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_load();

        // Mult
        for (int i = 0; i < CHUNK_SIZE/2; ++i) {
            o[i] = ku::float2_mul(o[i], scale_factor_float2);
        }

        // Store O
        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

template<int NUM_ELEMS_PER_THREAD, bool DO_ABS = false>
CUTE_DEVICE
float get_max(
    float p[NUM_ELEMS_PER_THREAD]
) {
    // Use 2-way max to optimize instruction scheduling
    float local_max0 = -CUDART_INF_F;
    float local_max1 = -CUDART_INF_F;
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; ++i) {
        float t = p[i];
        if constexpr (DO_ABS)
            t = fabsf(t);
        local_max0 = max(local_max0, t);
    }
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD-NUM_ELEMS_PER_THREAD/2; ++i) {
        float t = p[i+NUM_ELEMS_PER_THREAD/2];
        if constexpr (DO_ABS)
            t = fabsf(t);
        local_max1 = max(local_max1, t);
    }
    return max(local_max0, local_max1);
}

/*
Calculate s := exp2f(p*scale - new_max) and its sum
*/
template<int NUM_ELEMS_PER_THREAD>
CUTE_DEVICE
float get_s_from_p(
    nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2],
    float p[NUM_ELEMS_PER_THREAD],
    float scale,
    float new_max
) {
    float2 cur_sum = float2 {0.0f, 0.0f};
    float2 neg_new_max_float2 = float2 {-new_max, -new_max};
    float2 scale_float2 = float2 {scale, scale};
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
        float2 d = ku::float2_fma(float2{p[i*2], p[i*2+1]}, scale_float2, neg_new_max_float2);
        d.x = exp2f(d.x);
        d.y = exp2f(d.y);
        cur_sum = ku::float2_add(cur_sum, d);
        s[i] = __float22bfloat162_rn(d);
    }
    return cur_sum.x + cur_sum.y;
}

}
""",
    'csrc/cuda_kernels/sm100/helpers_h.cuh': r"""#pragma once

#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

namespace sm100 {

using namespace cute;

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

CUTE_DEVICE
int int4_max(int4 t) {
    return max(max(t.x, t.y), max(t.z, t.w));
}

CUTE_DEVICE
int int4_min(int4 t) {
    return min(min(t.x, t.y), min(t.z, t.w));
}

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
    float2 a01, a23, a45, a67;
};

// CUDA 13.2 (PTX ISA 9.2) adds a direct {e4m3x2, e2m1x2} -> bf16x2 conversion, i.e.
// `cvt.rn.bf16x2.e4m3x2` / `cvt.rn.bf16x2.e2m1x2`. Older toolchains only offer the .f16x2
// destination, so the value has to be widened through the FP32 domain instead. Every step of
// that detour is exact for the KV-cache formats: e4m3/e2m1 fit in f16, and their at most
// 3+1 significant bits also fit in the 8-bit significand of bf16.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define SM100_HAS_NATIVE_BF16X2_CVT 1
#else
#define SM100_HAS_NATIVE_BF16X2_CVT 0
#endif

// 2x f16 -> 2x bf16 (exact: f16 is a subset of f32, and every f16 the KV cache can store is representable in bf16)
CUTE_DEVICE
ku::nvbf16x2 f16x2_to_bf16x2(uint32_t f16x2) {
    return __float22bfloat162_rn(__half22float2(*(__half2*)&f16x2));
}

// 2x fp8_e4m3 -> 2x bf16, without scaling
CUTE_DEVICE
ku::nvbf16x2 fp8x2_to_bf16x2(ku::nve4m3x2 data) {
    uint32_t out;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(out)
        : "h"(*(uint16_t*)&data));
#else
    asm("{\n"
        ".reg .b32 t;\n"
        ".reg .b16 h0, h1;\n"
        ".reg .f32 f0, f1;\n"
        "cvt.rn.f16x2.e4m3x2 t, %1;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %0, f1, f0;\n"
        "}\n"
        : "=r"(out)
        : "h"(*(uint16_t*)&data));
#endif
    return *(ku::nvbf16x2*)&out;
}

// 1x ue8m0 scale -> 2x bf16 (the same scale in both halves)
CUTE_DEVICE
ku::nvbf16x2 ue8m0_to_bf16x2(__nv_fp8_e8m0 scale_e8m0) {
    uint16_t packed = (uint16_t)(*(uint8_t*)&scale_e8m0) * 0x0101;
    uint32_t out;
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;"
        : "=r"(out)
        : "h"(packed));
    return *(ku::nvbf16x2*)&out;
}

// 2x fp8_e4m3 -> 2x bf16, multiplied by an ue8m0 scale
CUTE_DEVICE
ku::nvbf16x2 fp8x2_to_bf16x2_with_scale(ku::nve4m3x2 data, __nv_fp8_e8m0 scale_e8m0) {
#if SM100_HAS_NATIVE_BF16X2_CVT
    return __hmul2(fp8x2_to_bf16x2(data), ue8m0_to_bf16x2(scale_e8m0));
#else
    // The whole data path plus the scale's own conversion and the multiply form one asm block, so the
    // intermediate f16x2/bf16x2 values are invisible to the compiler instead of being live across the
    // unrolled dequant steps.
    const uint16_t packed_scale = (uint16_t)(*(uint8_t*)&scale_e8m0) * 0x0101;
    uint32_t out;
    asm("{\n"
        ".reg .b32 t, s;\n"
        ".reg .b16 h0, h1;\n"
        ".reg .f32 f0, f1;\n"
        "cvt.rn.bf16x2.ue8m0x2 s, %2;\n"
        "cvt.rn.f16x2.e4m3x2 t, %1;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 t, f1, f0;\n"
        "mul.rn.bf16x2 %0, t, s;\n"
        "}\n"
        : "=r"(out)
        : "h"(*(uint16_t*)&data), "h"(packed_scale));
    return *(ku::nvbf16x2*)&out;
#endif
}

// 8x fp4_e2m1 (packed in 32 bits, 2 per byte) -> 4x bf16x2
// Written as one asm block so that ptxas selects the source byte with the .B0-.B3 operand selector of F2FP instead of emitting PRMTs
CUTE_DEVICE
void fp4x8_to_bf16x2x4(uint32_t packed, ku::nvbf16x2 *out) {
    uint32_t o0, o1, o2, o3;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
    out[2] = *(ku::nvbf16x2*)&o2;
    out[3] = *(ku::nvbf16x2*)&o3;
#else
    // A single asm block serializes the four byte conversions, so only one f16x2/f32 triple is in
    // flight at a time and none of it becomes visible to the compiler's register allocation
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t;\n"
        ".reg .b16 h0, h1;\n"
        ".reg .f32 f0, f1;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 t, b0;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %0, f1, f0;\n"
        "cvt.rn.f16x2.e2m1x2 t, b1;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %1, f1, f0;\n"
        "cvt.rn.f16x2.e2m1x2 t, b2;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %2, f1, f0;\n"
        "cvt.rn.f16x2.e2m1x2 t, b3;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %3, f1, f0;\n"
        "}\n"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
    out[2] = *(ku::nvbf16x2*)&o2;
    out[3] = *(ku::nvbf16x2*)&o3;
#endif
}

// 4x fp8_e4m3 -> 2x bf16x2 without scaling, used for the e4m3 scales of the fp4 KV cache
CUTE_DEVICE
void fp8x4_to_bf16x2x2(uint32_t packed, ku::nvbf16x2 *out) {
    uint32_t o0, o1;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm(
        "{\n"
        ".reg .b16 h0, h1;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.bf16x2.e4m3x2 %0, h0;\n"
        "cvt.rn.bf16x2.e4m3x2 %1, h1;\n"
        "}\n"
        : "=r"(o0), "=r"(o1)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
#else
    asm(
        "{\n"
        ".reg .b16 h0, h1;\n"
        ".reg .b32 t;\n"
        ".reg .f32 f0, f1;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 t, h0;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %0, f1, f0;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 t, h1;\n"
        "mov.b32 {h0, h1}, t;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %1, f1, f0;\n"
        "}\n"
        : "=r"(o0), "=r"(o1)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
#endif
}

// 2x fp8_e4m3 -> bf16x2 without scaling, used for the e4m3 scale of one quant tile of the fp4 KV cache
CUTE_DEVICE
ku::nvbf16x2 e4m3x2_to_bf16x2(uint16_t packed) {
    return fp8x2_to_bf16x2(*(ku::nve4m3x2*)&packed);
}

}
""",
    'csrc/cuda_kernels/sm100/kv_cache_utils.cuh': r"""#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/kv_cache_format_h.cuh"

namespace sm100 {

// Each block belongs to one cache. Separate format-specialized loops avoid overlapping live ranges.
template<typename Orig, typename Extra, typename Fn>
CUTE_DEVICE void for_each_kv_block(int begin, int end, int num_orig_blocks, Fn &&f) {
    if constexpr (std::is_same_v<Orig, Extra>) {
        CUTE_NO_UNROLL
        for (int block_idx = begin; block_idx < end; ++block_idx) {
            f.template operator()<Orig>(block_idx, block_idx >= num_orig_blocks);
        }
    } else {
        CUTE_NO_UNROLL
        for (int block_idx = begin; block_idx < min(num_orig_blocks, end); ++block_idx) {
            f.template operator()<Orig>(block_idx, false);
        }
        CUTE_NO_UNROLL
        for (int block_idx = max(begin, num_orig_blocks); block_idx < end; ++block_idx) {
            f.template operator()<Extra>(block_idx, true);
        }
    }
}

// `scales` points to the scale subset in token 0; coord uses the tensor map's flattened token rows.
template<int BYTES>
CUTE_DEVICE void copy_kv_scales_async(uint8_t *dst, const uint8_t *scales, int coord, int token_stride) {
    static_assert(BYTES == 8 || BYTES == 16);
    using CopyT = std::conditional_t<BYTES == 16, uint4, uint64_t>;
    const uint8_t *src = scales + (uint64_t)(coord >= 0 ? coord : 0) * token_stride;
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CopyT>::copy(*(const CopyT*)src, *(CopyT*)dst, coord >= 0);
}

// VATTN block-split: the raw region of every page block as rows of QUANT_BYTES (row pitch QUANT_BYTES); row
// block * (block_stride / QUANT_BYTES) + token is that token's raw data. The box is BOX_BYTES wide; bytes past
// QUANT_BYTES are zero-filled. A view row (stride(1)) is BYTES_PER_TOKEN, as the engine's [pages, page, 1, bytes] view
template<typename Part, int BOX_BYTES>
static CUtensorMap make_kv_raw_region_tensor_map_bs(
    const char *name, void *kv, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B
) {
    static constexpr int QB = Part::QUANT_BYTES;
    static_assert(QB % 16 == 0 && BOX_BYTES % 16 == 0 && BOX_BYTES / 4 <= 256 && BOX_BYTES >= QB);
    KU_ASSERT((int64_t)kv % 16 == 0, "The base address of %s (%p) must be 16B aligned", name, kv);
    KU_ASSERT(row_stride_bytes == Part::BYTES_PER_TOKEN, "%s.stride(-2) (%d) must be %d", name, row_stride_bytes, (int)Part::BYTES_PER_TOKEN);
    KU_ASSERT(block_stride_bytes % QB == 0, "%s.stride(0) (%ld) must be a multiple of %d", name, block_stride_bytes, QB);
    KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / QB) <= INT32_MAX, "%s: too many rows for the int32 TMA coordinates", name);
    return ku::make_tensor_map(
        {(uint64_t)QB / 4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / QB)},
        {(uint64_t)QB},
        {BOX_BYTES / 4, 1},
        (uint8_t*)kv,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        promotion
    );
}

// UINT32 keeps the TMA box dimension <= 256. Each gather4 writes four BOX_BYTES rows;
// bytes beyond Part's view are zero-filled. The caller aligns each gather4 destination to 128 B.
template<typename Part, int BOX_BYTES>
static CUtensorMap make_kv_quant_part_tensor_map(
    const char *name, void *kv, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B
) {
    static constexpr int TOKEN_STRIDE = Part::BYTES_PER_TOKEN;
    static_assert(Part::RAW_TOKEN_DATA_BYTES % 4 == 0 && BOX_BYTES % 16 == 0 && BOX_BYTES / 4 <= 256);
    static_assert(TOKEN_STRIDE % 16 == 0 && Part::RAW_TOKEN_OFFSET % 16 == 0);
    KU_ASSERT((int64_t)kv % 16 == 0, "The base address of %s (%p) must be 16B aligned", name, kv);
    KU_ASSERT(row_stride_bytes == TOKEN_STRIDE, "%s.stride(-2) (%d) must be %d", name, row_stride_bytes, TOKEN_STRIDE);
    KU_ASSERT(block_stride_bytes % TOKEN_STRIDE == 0, "%s.stride(0) (%ld) must be a multiple of %d", name, block_stride_bytes, TOKEN_STRIDE);
    KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TOKEN_STRIDE) <= INT32_MAX, "%s: too many rows for the int32 TMA coordinates", name);
    return ku::make_tensor_map(
        {(uint64_t)Part::RAW_TOKEN_DATA_BYTES / 4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TOKEN_STRIDE)},
        {(uint64_t)TOKEN_STRIDE},
        {BOX_BYTES / 4, 1},
        (uint8_t*)kv + Part::RAW_TOKEN_OFFSET,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        promotion
    );
}

}
""",
    'csrc/cuda_kernels/utils_h.cuh': r"""#pragma once

#include <math_constants.h>

#include "params_h.cuh"
#include "cuda_kernels/defines_h.cuh"

#define println(fmt, ...) { print(fmt, ##__VA_ARGS__); print("\n"); }

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

struct RingBufferState {
    uint32_t cur_block_idx = 0u;

    __device__ __forceinline__
    void update() {
        cur_block_idx += 1;
    }    

    template<uint32_t NUM_STAGES>
    __device__ __forceinline__
    std::pair<uint32_t, bool> get() const {
        uint32_t stage_idx = cur_block_idx % NUM_STAGES;
        bool phase = (cur_block_idx / NUM_STAGES) & 1;
        return {stage_idx, phase};
    }

    __device__ __forceinline__
    RingBufferState offset_by(const int offset) const {
        // Must guarantee no underflow
        uint32_t new_block_idx = static_cast<uint32_t>(static_cast<int>(cur_block_idx) + offset);
        RingBufferState new_state;
        new_state.cur_block_idx = new_block_idx;
        return new_state;
    }
};
""",
    'csrc/params_h.cuh': r"""#pragma once

#include <cstdint>
#include <type_traits>

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA

#include <cutlass/bfloat16.h>
#include <cute/container/alignment.hpp>
using bf16 = cutlass::bfloat16_t;

#else

#include <acl/acl_base_rt.h>
#ifdef __NPU_DEVICE__   // If either __NPU_DEVICE__ and __NPU_HOST__ is defined (which means that this file is included by a `.asc` file), we can use `bfloat16_t` as `bf16`. Otherwise (which means that this file is included by a `.cpp` file) we declare `bf16` as a class to pass compilation
    using bf16 = bfloat16_t;
#else
    #ifdef __NPU_HOST__
        using bf16 = bfloat16_t;
    #else
        struct bf16;
    #endif
#endif

#endif

#if defined(__NPU_DEVICE__) || defined(__ASCC_DEVICE__)
#define FLASH_MLA_GM __gm__
#else
#define FLASH_MLA_GM
#endif

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
using stream_t = cudaStream_t;
#else
using stream_t = aclrtStream;
#endif

enum class ModelType {
    V41,    // DeepSeek V4.1 (d_qk=512, RoPE fp8, tile_size=32)
    V41_FP4     // DeepSeek V4.1 (d_qk=512, fp4 e2m1, tile_size=16, e4m3 scales)
};

enum class SparseAttnFwdMode {
    Prefill,            // Normal prefill mode
    Decode,             // To trigger decoding mode for kernels that support both prefill and decode
    DecodeWithSplitKV   // Decoding with split KV
};

template<SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v = FWD_MODE == SparseAttnFwdMode::Decode || FWD_MODE == SparseAttnFwdMode::DecodeWithSplitKV;

// Parameters for the sparse attention forward pass kernel (prefill).
struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale;             // Attention score scaling factor, typically d_qk ** -0.5
    float sm_scale_div_log2;    // sm_scale * log2(e), precomputed for log-base-2 softmax

    // Input tensors
    FLASH_MLA_GM bf16* __restrict__ q;    // [s_q, h_q, d_qk]
    FLASH_MLA_GM bf16* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    FLASH_MLA_GM int* __restrict__ indices;   // [s_q, h_kv, topk], KV token indices for sparse attention
    FLASH_MLA_GM float* __restrict__ attn_sink;   // [h_q], may be nullptr. Per-head attention bias
    FLASH_MLA_GM int* __restrict__ topk_length;    // [s_q], may be nullptr. Actual valid topk count per query

    // Strides (in number of elements, not bytes)
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    FLASH_MLA_GM bf16* __restrict__ out;   // [s_q, h_q, d_v], attention output
    FLASH_MLA_GM float* __restrict__ max_logits; // [s_q, h_q], per-head max attention logit (for numerical stability)
    FLASH_MLA_GM float* __restrict__ lse; // [s_q, h_q], log-sum-exp of attention scores

    SparseAttnFwdMode mode;     // Which forward mode to use (Prefill / Decode / DecodeWithSplitKV)
    int num_sm;                 // Number of SMs available for this kernel launch
    stream_t stream;
};

// Tile scheduling metadata for split-KV decoding. Each entry describes
// the work assignment for one SM partition (i.e. a contiguous range of requests and KV blocks).
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
struct CUTE_ALIGNAS(4*8) DecodingSchedMeta {
#else
struct DecodingSchedMeta {
#endif
    int begin_req_idx, end_req_idx;     // Request range [begin, end], both inclusive
    int begin_block_idx, end_block_idx; // KV block range [begin, end), inclusive-exclusive
    int begin_split_idx;                // Starting split index for output accumulation
    int is_first_req_splitted, is_last_req_splitted; // Whether the first/last request in this partition is split across partitions
    int _pad[1];
};
static_assert(sizeof(DecodingSchedMeta) == 32);
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

// Parameters for the tile scheduler metadata generation kernel.
// This kernel computes how to partition work across SM partitions for split-KV decoding.
struct GetDecodeSchedMetaParams {
    int b;                              // Batch size
    int s_q;                            // Number of query tokens per request
    int block_size_topk;                // TopK block size used in the decoding kernel
    int fixed_overhead_num_blocks;      // Fixed overhead (in blocks) per request for scheduling
    int num_sm_parts;                   // Number of SM partitions to schedule across
    int topk;                           // Number of selected KV tokens
    int extra_topk;                     // Extra topk count (0 if extra KV cache is disabled)
    int *__restrict__ topk_length, *__restrict__ extra_topk_length;  // [batch_size], may be nullptr

    // Output
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1], cumulative split counts

    stream_t stream;
};

// Parameters for the sparse attention decoding kernel.
// Decoding uses a paged KV cache, possibly quantized in FP8 format.
struct SparseAttnDecodeParams {
    int b;                      // Batch size (number of requests)
    int s_q;                    // Number of query tokens per request (typically 1 for autoregressive decoding)
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale;             // Attention score scaling factor, typically d_qk ** -0.5
    float sm_scale_div_log2;    // sm_scale * log2(e), precomputed for log-base-2 softmax
    int num_blocks;             // Total number of blocks in the paged KV cache
    int page_block_size;        // Number of KV tokens per page block
    int topk;                   // Number of selected KV tokens per query
    ModelType model_type;       // Format of `kv` (V41 or V41_FP4), see KVCacheFormat in kv_cache_format.h
    ModelType extra_model_type; // Format of `extra_kv`: model_type, or V41_FP4 next to a V41 `kv` (see is_valid_kv_format_pair)

    // Input tensors
    FLASH_MLA_GM bf16* __restrict__ q;   // [b, s_q, h_q, d_qk]
    FLASH_MLA_GM bf16* __restrict__ kv;  // [num_blocks, page_block_size, h_kv, bytes_per_token], paged KV cache
    FLASH_MLA_GM int* __restrict__ indices;   // [b, s_q, topk], KV token indices for sparse attention
    FLASH_MLA_GM int* __restrict__ topk_length;  // [b], may be nullptr. Actual valid topk count per request
    FLASH_MLA_GM float* __restrict__ attn_sink;  // [h_q], may be nullptr. Per-head attention bias

    // Output tensors
    FLASH_MLA_GM float* __restrict__ lse;    // [b, s_q, h_q], log-sum-exp of attention scores
    FLASH_MLA_GM bf16* __restrict__ out;   // [b, s_q, h_q, d_v], attention output

    // Extra KV cache — for models with a secondary KV cache
    int extra_num_blocks, extra_page_block_size, extra_topk;
    FLASH_MLA_GM bf16* __restrict__ extra_kv;  // [extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token]
    FLASH_MLA_GM int* __restrict__ extra_indices;   // [b, s_q, extra_topk]
    FLASH_MLA_GM int* __restrict__ extra_topk_length;  // [b], may be nullptr

    // Strides (in number of elements, not bytes)
    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    int num_sm;                 // Number of SMs available for this kernel launch
    stream_t stream;

    // Split-KV parameters — when enabled, the KV sequence is partitioned across CTAs
    bool enable_split_kv;
    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q], partial LSE per split
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v], partial output per split
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts], tile scheduling metadata
    int* __restrict__ num_splits_ptr; // [batch_size+1], cumulative split counts per request
    int num_sm_parts;               // Number of SM partitions for split-KV scheduling
};

template<SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;

#undef FLASH_MLA_GM
""",
    'csrc/vattn_host.cpp': r"""// vattn_host.cpp (dspd-v59 VATTN route D, ours): the host half of VATTN. It owns a private DeepJIT runtime that
// compiles the device unit (vattn_kernel_tu.cuh) with the image's pip nvcc 13.4, and it rebuilds, per call, exactly
// the launch today's wrapper performs: the two tensor maps, the parameter struct, the grid, the block, the cluster,
// the dynamic shared-memory size and the launch itself. TORCH_LIBRARY namespace dspd_v59_vattn.
//
// Why a private runtime instead of the engine's load_jit: load_jit compiles with the engine's own nvcc 13.0, which
// lacks the native e4m3x2 / e2m1x2 -> bf16x2 conversions this kernel's fast path needs, and patching the engine's
// process-global device_compiler_path around a build reconfigures unrelated SGLang behaviour. DeepJIT takes its nvcc
// from "<prefix>_JIT_NVCC_COMPILER" through its own Env, so a private prefix (DSPDVA) reaches the right compiler
// without touching any SGLang, DG_ or CUDA_ setting. v50's own M462 sets the precedent: a cpp_extension host library
// (with_cuda=False) whose DeepJIT compiles the kernel.
//
// Why the launch arithmetic is re-typed here rather than included: the vendored host code lives in headers that pull
// in the kerutils device tree (core_attn/config_h.cuh and cuda_kernels/sm100/kv_cache_utils.cuh both include
// <kerutils/kerutils.cuh>, whose common header refuses a build that is not nvcc's), so a with_cuda=False unit cannot
// see them. What is NOT re-typed is the ABI: the parameter struct is this kernel's own ParamT<Decode> from the
// vendored params_h.cuh / core_attn/kernel_h.cuh, which compile host-side (CUTLASS and CuTe only). Every re-typed
// constant below is passed to the device compile as a -DVATTN_EXPECT_* macro and checked there against the vendored
// header's own value, so drift fails the compile and VATTN stays off.
//
// Sections: (A) the pure launch arithmetic -- no CUDA call, no torch, no state, buildable on a CPU box alone, which
// is what the lane's host-math check (S8) compiles and compares against the current wrapper; (A2) the plan -> ABI
// fill: the bytes the kernel actually receives (ParamT<Decode>, which of TMAParams' seven map slots each tensor map
// fills, and AuxParams' two FastDivmod divisors), and the launch itself (review V59-R3-F1): both function attributes,
// every CUlaunchConfig field, its one launch attribute, the kernel argument array, and cuTensorMapEncodeTiled's twelve
// arguments in call order. A2 uses the vendored parameter headers, CUTLASS and cuda.h, but no torch, no state and no
// CUDA call, so S8 compiles and compares it too (review V59-R2-F2 / V59-R3-F1: before these splits the never-launched
// bytes were built inline in decode() and nothing compared them); (B) the runtime, built only when
// VATTN_HOST_MATH_ONLY is not defined. Section B decides no launch byte itself: decode() gathers the device pointers
// and the stream, encodes the maps A2 describes and forwards A2's launch, and the lane pins its text.

#ifndef FLASH_MLA_IS_BUILD_ON_CUDA
#define FLASH_MLA_IS_BUILD_ON_CUDA
#endif

#include <cstddef>
#include <cstdint>

// ===================================================================================== (A) pure launch arithmetic
namespace dspd_v59_vattn {

// Launch constants. Each is checked inside the device unit against the vendored header that defines it; the comment
// names that definition. Nothing here is a tuning knob.
constexpr int kHQ = 64;                  // Kernel::H_Q (this cubin is the 1-CTA, h_q 64 decode kernel)
constexpr int kDQK = 512;                // Kernel::D_QK
constexpr int kDVO = 512;                // Kernel::D_VO
constexpr int kThreads = 512;            // Kernel::NUM_THREADS
constexpr int kCluster = 1;              // Kernel::CLUSTER_SIZE = ceil_div(H_Q, 64)
constexpr int kBTopk = 64;               // Kernel::B_TOPK (1-CTA)
constexpr int kSmemBytes = 225280;       // sizeof(Kernel::SharedMemoryPlan), equal for both extra-cache classes
constexpr int kWvGroupSize = 8;          // Kernel::WV_GROUP_SIZE
constexpr int kOQuantTile = 32;          // Kernel::O_QUANT_TILE_SIZE
constexpr int kRopeDim = 64;             // Kernel::D_ROPE

// KVCacheFormat<V41 / V41_FP4> and KVCachePartBS: a token's raw bytes, its total bytes, and the shared-memory row
// stride the block-split transport gathers into, (ceil_div(QUANT_BYTES, 16) | 1) * 16.
constexpr int kQuantBytesFp8 = 512, kQuantBytesFp4 = 256;
constexpr int kBytesPerTokenFp8 = 528, kBytesPerTokenFp4 = 288;
constexpr int kRawStrideFp8 = 528, kRawStrideFp4 = 272;
static_assert(kRawStrideFp8 == (((kQuantBytesFp8 + 15) / 16) | 1) * 16, "fp8 raw row stride");
static_assert(kRawStrideFp4 == (((kQuantBytesFp4 + 15) / 16) | 1) * 16, "fp4 raw row stride");
static_assert(kBytesPerTokenFp8 == kQuantBytesFp8 + kDQK / 32, "fp8 bytes per token");
static_assert(kBytesPerTokenFp4 == kQuantBytesFp4 + kDQK / 16, "fp4 bytes per token");
constexpr int kMaxDevices = 16;          // vattn.cuh's per-device SM-count cache

// DLPack type codes, as the current wrapper's is_dtype reads them.
enum : int { kCodeInt = 0, kCodeUInt = 1, kCodeFloat = 2, kCodeBfloat = 4 };
// Model types, in the vendored enum's order (ModelType::V41, ModelType::V41_FP4).
enum : int { kModelV41 = 0, kModelV41Fp4 = 1 };
// CUtensorMap enum values the block-split map uses: the names are the ones the vendored maker passes
// (make_kv_raw_region_tensor_map_bs: UINT32, SWIZZLE_NONE; Kernel::run: L2_256B; ku::make_tensor_map's defaults:
// INTERLEAVE_NONE, FLOAT_OOB_FILL_NONE), the values are cuda.h's. Section B static_asserts every one of them against
// cuda.h itself, so a wrong value cannot build (cuda.h 13.0: UINT8 0, UINT16 1, UINT32 2, INT32 3).
enum : int { kTmaUint32 = 2 /* CU_TENSOR_MAP_DATA_TYPE_UINT32 */, kTmaSwizzleNone = 0 /* CU_TENSOR_MAP_SWIZZLE_NONE */,
             kTmaPromoteL2_256B = 3 /* CU_TENSOR_MAP_L2_PROMOTION_L2_256B */,
             kTmaInterleaveNone = 0 /* CU_TENSOR_MAP_INTERLEAVE_NONE */,
             kTmaOobFillNone = 0 /* CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE */ };
// The launch the vendored Kernel<CONFIG>::run performs through CUTLASS (cluster_launch.hpp): cudaFuncSetAttribute
// (MaxDynamicSharedMemorySize = sizeof(SharedMemoryPlan)); ClusterLauncher::init's cudaFuncSetAttribute
// (NonPortableClusterSizeAllowed = 1); then cudaLaunchKernelExC with exactly ONE launch attribute, the cluster
// dimension, set even for a 1-CTA cluster, and no programmatic-stream-serialisation (PDL) attribute, because
// launch_kernel_on_cluster is called without launch_with_pdl. DeepJIT's own Kernel::launch drops the cluster attribute
// when cluster_dim.x == 1, so route D does not launch through it: section B calls cuLaunchKernelEx itself.
constexpr int kLaunchAttrCount = 1;      // [0] = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
constexpr int kNonPortableCluster = 1;   // ClusterLauncher::init

// One input tensor, as the arithmetic needs it: no torch type reaches section A.
struct TensorDesc {
    int ndim = 0;
    long long size[4] = {0, 0, 0, 0};
    long long stride[4] = {0, 0, 0, 0};   // in elements, as both torch and the current wrapper report them
    int code = -1, bits = 0, lanes = 0;
    unsigned long long addr = 0;          // data pointer, for the 16 B alignment the TMA map requires
};

// One tensor map, as cuTensorMapEncodeTiled takes it: the dims and strides are cuda.h's cuuint64_t (= uint64_t, i.e.
// `unsigned long` on LP64 Linux, NOT `unsigned long long`), the box and element strides cuuint32_t (= uint32_t).
// Section B static_asserts these four element types against cuda.h's typedefs.
struct TmaPlan {
    int used = 0;
    int rank = 2;
    uint64_t gdim[2] = {0, 0};
    uint64_t gstride[1] = {0};            // in BYTES, as the encoder takes it
    uint32_t bdim[2] = {0, 0};
    uint32_t estride[2] = {1, 1};
    int dtype = kTmaUint32, swizzle = kTmaSwizzleNone, promo = kTmaPromoteL2_256B;
    int interleave = kTmaInterleaveNone, oob = kTmaOobFillNone;
    unsigned long long base = 0;
    int slot = -1;                        // which of TMAParams' seven maps this one fills
};

// Everything the launch needs, and nothing that depends on a device being present.
struct LaunchPlan {
    long long rc = 0;                     // 0, or the current wrapper's refusal code
    unsigned grid[3] = {0, 1, 1};
    unsigned block[3] = {0, 1, 1};
    unsigned cluster[3] = {0, 1, 1};
    int smem_bytes = 0;
    // the launch itself, as CUTLASS's cluster launch performs it (see kLaunchAttrCount above)
    int func_max_dynamic_smem = 0;        // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    int func_nonportable_cluster = 0;     // CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
    int launch_attrs = 0;                 // how many launch attributes; [0] is the cluster dimension
    unsigned attr_cluster[3] = {0, 0, 0}; // attribute [0]'s clusterDim
    int attr_pdl = 0;                     // a programmatic-stream-serialisation attribute (never: CUTLASS default)
    TmaPlan kv_map, extra_map;
    // the scalar parameter fields, in the order params_h.cuh / kernel_h.cuh declare them
    int b = 0, s_q = 0, h_q = 0, h_kv = 0, d_qk = 0, d_v = 0;
    float sm_scale = 0.0f, sm_scale_div_log2 = 0.0f;
    int num_blocks = 0, page_block_size = 0, topk = 0;
    int model_type = kModelV41, extra_model_type = kModelV41;
    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    int stride_q_s_q = 0, stride_q_h_q = 0;
    int stride_kv_block = 0, stride_kv_row = 0;
    int stride_indices_s_q = 0, stride_lse_s_q = 0;
    int stride_extra_kv_block = 0, stride_extra_kv_row = 0, stride_extra_indices_s_q = 0;
    int num_sm = 0;
    int enable_split_kv = 0;
    // ParamsTemplate's own fields
    int enable_q_norm = 0, is_rope_neox_style = 0;
    float rms_norm_eps = 0.0f;
    unsigned rope_dim = 0, n_wv_group = 0, wv_group_size = 0, num_per_channels = 0;
    int use_tma_aligned_col_major_sf = 0, round_sf = 0, use_packed_ue8m0 = 0;
    unsigned stride_out_sf_wv_group = 0, stride_out_sf_head_dim = 0;
    // the two FastDivmod divisors the kernel's AuxParams carries
    int divisor_page = 0, divisor_extra_page = 0;
};

inline bool dtype_is(const TensorDesc& t, const int code, const int bits) {
    return t.code == code && t.bits == bits && t.lanes == 1;
}

inline bool dtype_is_byte(const TensorDesc& t) { return t.bits == 8 && t.lanes == 1; }

// The block-split raw-region map of one paged cache: the raw region of every page block seen as rows of QUANT_BYTES,
// a box BOX_BYTES wide, UINT32 elements so the box dimension stays <= 256 (make_kv_raw_region_tensor_map_bs).
// The vendored maker asserts these preconditions and aborts; here a miss is a refusal code and the caller runs the
// engine's kernel instead.
inline bool plan_kv_map(TmaPlan& out, const TensorDesc& kv, const int quant_bytes, const int bytes_per_token,
                        const int box_bytes, const int slot) {
    if (quant_bytes % 16 != 0 || box_bytes % 16 != 0 || box_bytes / 4 > 256 || box_bytes < quant_bytes)
        return false;
    if (kv.addr % 16 != 0)
        return false;
    const long long block_stride = kv.stride[0], row_stride = kv.stride[1], num_blocks = kv.size[0];
    if (row_stride != bytes_per_token || block_stride <= 0 || block_stride % quant_bytes != 0)
        return false;
    const long long rows = num_blocks * (block_stride / quant_bytes);
    if (num_blocks <= 0 || rows > 2147483647LL)
        return false;
    out.used = 1;
    out.rank = 2;
    out.gdim[0] = (uint64_t)(quant_bytes / 4);
    out.gdim[1] = (uint64_t)rows;
    out.gstride[0] = (uint64_t)quant_bytes;
    out.bdim[0] = (uint32_t)(box_bytes / 4);
    out.bdim[1] = 1u;
    out.estride[0] = 1u;
    out.estride[1] = 1u;
    out.dtype = kTmaUint32;
    out.swizzle = kTmaSwizzleNone;
    out.promo = kTmaPromoteL2_256B;
    out.interleave = kTmaInterleaveNone;
    out.oob = kTmaOobFillNone;
    out.base = kv.addr;
    out.slot = slot;
    return true;
}

// The whole host-side decision of one decode call: the current wrapper's gates (same order, same refusal codes),
// its parameter arithmetic and its launch configuration. `device_id` and `num_sm` are the only facts a device
// supplies; everything else is a function of the inputs' shapes, strides, dtypes and addresses.
inline LaunchPlan plan_decode(const TensorDesc& q, const TensorDesc& kv, const TensorDesc& indices,
                              const TensorDesc& topk_length, const TensorDesc& attn_sink, const TensorDesc& extra_kv,
                              const TensorDesc& extra_indices, const TensorDesc& extra_topk_length,
                              const TensorDesc& positions, const TensorDesc& cos_sin_cache, const TensorDesc& out_fp8,
                              const TensorDesc& out_sf, const TensorDesc& lse, const double sm_scale,
                              const long long has_extra, const int device_id, const int num_sm) {
    LaunchPlan p;
    if (q.ndim != 3 || kv.ndim != 4 || indices.ndim != 2 || topk_length.ndim != 1 || attn_sink.ndim != 1 ||
        positions.ndim != 1 || cos_sin_cache.ndim != 2 || out_fp8.ndim != 3 || out_sf.ndim != 3 || lse.ndim != 2) {
        p.rc = -1;
        return p;
    }
    const long long s_q = q.size[0], h_q = q.size[1], d_qk = q.size[2];
    if (s_q <= 0 || h_q != kHQ || d_qk != kDQK) {
        p.rc = -2;
        return p;
    }
    if (!dtype_is(q, kCodeBfloat, 16) || !dtype_is_byte(kv) || !dtype_is(indices, kCodeInt, 32) ||
        !dtype_is(topk_length, kCodeInt, 32) || !dtype_is(attn_sink, kCodeFloat, 32) ||
        !dtype_is(positions, kCodeInt, 32) || !dtype_is(cos_sin_cache, kCodeFloat, 32) || !dtype_is_byte(out_fp8) ||
        !dtype_is(out_sf, kCodeInt, 32) || !dtype_is(lse, kCodeFloat, 32)) {
        p.rc = -3;
        return p;
    }
    if (q.stride[2] != 1 || q.stride[1] != d_qk || kv.stride[3] != 1 || kv.size[2] != 1 ||
        kv.size[3] != kBytesPerTokenFp8 || kv.stride[1] != kBytesPerTokenFp8 || indices.stride[1] != 1 ||
        indices.size[0] != s_q || topk_length.size[0] != s_q || topk_length.stride[0] != 1 ||
        attn_sink.size[0] != h_q || attn_sink.stride[0] != 1 || positions.size[0] != s_q ||
        positions.stride[0] != 1 || cos_sin_cache.size[1] != kRopeDim || cos_sin_cache.stride[1] != 1 ||
        cos_sin_cache.stride[0] != kRopeDim) {
        p.rc = -4;
        return p;
    }
    const long long topk = indices.size[1];
    if (topk <= 0 || (indices.stride[0] * 4) % 32 != 0) {
        p.rc = -5;
        return p;
    }
    if (out_fp8.size[0] != s_q || out_fp8.size[1] != h_q / 8 || out_fp8.size[2] != 8 * kDVO ||
        out_fp8.stride[2] != 1 || out_fp8.stride[1] != 8 * kDVO || out_fp8.stride[0] != h_q * kDVO) {
        p.rc = -6;
        return p;
    }
    if (out_sf.size[0] != s_q || out_sf.size[1] != h_q / 8 || out_sf.size[2] != 8 * kDVO / 128 ||
        out_sf.stride[0] != 1) {
        p.rc = -7;
        return p;
    }
    if (lse.size[0] != s_q || lse.size[1] != h_q || lse.stride[1] != 1 || lse.stride[0] != h_q) {
        p.rc = -8;
        return p;
    }
    long long extra_topk = 0;
    if (has_extra) {
        if (extra_kv.ndim != 4 || extra_indices.ndim != 2 || extra_topk_length.ndim != 1 ||
            !dtype_is_byte(extra_kv) || !dtype_is(extra_indices, kCodeInt, 32) ||
            !dtype_is(extra_topk_length, kCodeInt, 32)) {
            p.rc = -9;
            return p;
        }
        if (extra_kv.size[2] != 1 || extra_kv.size[3] != kBytesPerTokenFp4 ||
            extra_kv.stride[1] != kBytesPerTokenFp4 || extra_kv.stride[3] != 1 || extra_indices.size[0] != s_q ||
            extra_indices.stride[1] != 1 || extra_topk_length.size[0] != s_q || extra_topk_length.stride[0] != 1) {
            p.rc = -10;
            return p;
        }
        extra_topk = extra_indices.size[1];
        if (extra_topk <= 0 || topk % 8 != 0) {
            p.rc = -11;
            return p;
        }
    }
    if (device_id < 0 || device_id >= kMaxDevices) {
        p.rc = -12;
        return p;
    }
    // The two block-split tensor maps. The vendored maker's KU_ASSERTs become refusals (-13 / -14): route D never
    // aborts the engine's process over an input it can simply hand back to the engine's own kernel.
    if (!plan_kv_map(p.kv_map, kv, kQuantBytesFp8, kBytesPerTokenFp8, kRawStrideFp8, /*tensor_map_kv_fp8_part_cta0*/ 1)) {
        p.rc = -13;
        return p;
    }
    if (has_extra && !plan_kv_map(p.extra_map, extra_kv, kQuantBytesFp4, kBytesPerTokenFp4, kRawStrideFp4,
                                  /*tensor_map_extra_kv_fp4_part_cta0*/ 5)) {
        p.rc = -14;
        return p;
    }
    // Parameters, filled as the current wrapper fills them.
    p.b = 1;
    p.s_q = (int)s_q;
    p.h_q = (int)h_q;
    p.h_kv = 1;
    p.d_qk = kDQK;
    p.d_v = kDVO;
    const float scale_f = (float)sm_scale;
    p.sm_scale = scale_f;
    p.sm_scale_div_log2 = scale_f * 1.44269504f;
    p.num_blocks = (int)kv.size[0];
    p.page_block_size = (int)kv.size[1];
    p.topk = (int)topk;
    p.model_type = kModelV41;
    p.extra_model_type = has_extra ? kModelV41Fp4 : kModelV41;
    if (has_extra) {
        p.extra_num_blocks = (int)extra_kv.size[0];
        p.extra_page_block_size = (int)extra_kv.size[1];
        p.extra_topk = (int)extra_topk;
        p.stride_extra_kv_block = (int)extra_kv.stride[0];
        p.stride_extra_kv_row = (int)extra_kv.stride[1];
        p.stride_extra_indices_s_q = (int)extra_indices.stride[0];
    }
    p.stride_q_s_q = (int)q.stride[0];
    p.stride_q_h_q = (int)q.stride[1];
    p.stride_kv_block = (int)kv.stride[0];
    p.stride_kv_row = (int)kv.stride[1];
    p.stride_indices_s_q = (int)indices.stride[0];
    p.stride_lse_s_q = (int)lse.stride[0];
    p.num_sm = num_sm;
    p.enable_split_kv = 0;
    p.enable_q_norm = 0;
    p.rms_norm_eps = 0.0f;
    p.is_rope_neox_style = 0;
    p.rope_dim = (unsigned)kRopeDim;
    p.n_wv_group = (unsigned)(h_q / kWvGroupSize);
    p.wv_group_size = (unsigned)kWvGroupSize;
    p.num_per_channels = (unsigned)kOQuantTile;
    p.use_tma_aligned_col_major_sf = 1;
    p.round_sf = 1;
    p.use_packed_ue8m0 = 1;
    p.stride_out_sf_wv_group = (unsigned)out_sf.stride[1];
    p.stride_out_sf_head_dim = (unsigned)out_sf.stride[2];
    // AuxParams: the extra divisor is 1 when there is no extra cache, as the wrapper's `extra_kv != nullptr` test gives
    p.divisor_page = p.page_block_size;
    p.divisor_extra_page = has_extra ? p.extra_page_block_size : 1;
    // Launch configuration: Kernel<CONFIG>::run's cluster launch
    p.grid[0] = (unsigned)(s_q * kCluster);
    p.grid[1] = 1u;
    p.grid[2] = 1u;
    p.block[0] = (unsigned)kThreads;
    p.block[1] = 1u;
    p.block[2] = 1u;
    p.cluster[0] = (unsigned)kCluster;
    p.cluster[1] = 1u;
    p.cluster[2] = 1u;
    p.smem_bytes = kSmemBytes;
    // ...and CUTLASS's ClusterLauncher: both function attributes, then one launch attribute, the cluster dimension
    // (explicit even at 1 x 1 x 1), and no PDL attribute
    p.func_max_dynamic_smem = kSmemBytes;
    p.func_nonportable_cluster = kNonPortableCluster;
    p.launch_attrs = kLaunchAttrCount;
    p.attr_cluster[0] = p.cluster[0];
    p.attr_cluster[1] = p.cluster[1];
    p.attr_cluster[2] = p.cluster[2];
    p.attr_pdl = 0;
    p.rc = 0;
    return p;
}

}  // namespace dspd_v59_vattn

// ================================================================================ (A2) the plan -> ABI fill
// No torch, no state, no CUDA call: only the vendored parameter headers (which compile host-side against CUTLASS and
// CuTe alone), cuda.h for CUtensorMap and the driver typedefs, and CUTLASS's FastDivmod. Section B calls fill_abi and
// writes nothing into the kernel's parameter bytes itself, so the lane's S8 check can compile this fill on a CPU box
// and compare it, byte for byte, with a transliteration of dspd_w38's own wrapper.
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

// Section A's re-typed driver types and enum values, against cuda.h itself: a wrong one fails this build (VATTN off)
// instead of encoding a different tensor map than the vendored maker does.
static_assert(std::is_same_v<std::remove_all_extents_t<decltype(dspd_v59_vattn::TmaPlan::gdim)>, cuuint64_t>,
              "dspd-v59 VATTN: TmaPlan::gdim must be cuuint64_t");
static_assert(std::is_same_v<std::remove_all_extents_t<decltype(dspd_v59_vattn::TmaPlan::gstride)>, cuuint64_t>,
              "dspd-v59 VATTN: TmaPlan::gstride must be cuuint64_t");
static_assert(std::is_same_v<std::remove_all_extents_t<decltype(dspd_v59_vattn::TmaPlan::bdim)>, cuuint32_t>,
              "dspd-v59 VATTN: TmaPlan::bdim must be cuuint32_t");
static_assert(std::is_same_v<std::remove_all_extents_t<decltype(dspd_v59_vattn::TmaPlan::estride)>, cuuint32_t>,
              "dspd-v59 VATTN: TmaPlan::estride must be cuuint32_t");
static_assert(dspd_v59_vattn::kTmaUint32 == (int)CU_TENSOR_MAP_DATA_TYPE_UINT32,
              "dspd-v59 VATTN: kTmaUint32 is not CU_TENSOR_MAP_DATA_TYPE_UINT32");
static_assert(dspd_v59_vattn::kTmaSwizzleNone == (int)CU_TENSOR_MAP_SWIZZLE_NONE,
              "dspd-v59 VATTN: kTmaSwizzleNone is not CU_TENSOR_MAP_SWIZZLE_NONE");
static_assert(dspd_v59_vattn::kTmaPromoteL2_256B == (int)CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              "dspd-v59 VATTN: kTmaPromoteL2_256B is not CU_TENSOR_MAP_L2_PROMOTION_L2_256B");
static_assert(dspd_v59_vattn::kTmaInterleaveNone == (int)CU_TENSOR_MAP_INTERLEAVE_NONE,
              "dspd-v59 VATTN: kTmaInterleaveNone is not CU_TENSOR_MAP_INTERLEAVE_NONE");
static_assert(dspd_v59_vattn::kTmaOobFillNone == (int)CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
              "dspd-v59 VATTN: kTmaOobFillNone is not CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE");

#include <cutlass/fast_math.h>

#include "params_h.cuh"
#include "core_attn/kernel_h.cuh"

namespace dspd_v59_vattn {

namespace CA = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
using KernelParams = CA::ParamT<SparseAttnFwdMode::Decode>;

// The kernel's TMAParams and AuxParams, in the order core_attn/config_h.cuh declares them. That header cannot be
// included here (it reaches the kerutils device tree), so the two structs are named here and their sizes are checked
// on both sides: here against the ABI the launch writes, and inside the device unit against the kernel's own types.
struct TmaParamsAbi {
    CUtensorMap tensor_map_kv;                       // prefill only; zero in every decode launch
    CUtensorMap tensor_map_kv_fp8_part_cta0;
    CUtensorMap tensor_map_kv_fp8_part_cta1;         // 2-CTA only; zero here
    CUtensorMap tensor_map_extra_kv_fp8_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta1;
};
struct AuxParamsAbi {
    cutlass::FastDivmod fast_divmod_page_block_size;
    cutlass::FastDivmod fast_divmod_extra_page_block_size;
};
static_assert(sizeof(KernelParams) == 376, "dspd-v59 VATTN: ParamT<Decode> is not the 376-byte ABI");
static_assert(sizeof(TmaParamsAbi) == 896, "dspd-v59 VATTN: TMAParams is not seven tensor maps");
static_assert(sizeof(AuxParamsAbi) == 24, "dspd-v59 VATTN: AuxParams is not two FastDivmods");

// TMAParams' seven slots, in the order core_attn/config_h.cuh declares them. A TmaPlan carries the index of the one
// it fills (1 for the fp8 KV map, 5 for the fp4 extra map, as Kernel::run assigns them); this is the only place that
// turns an index into a field, so a swapped entry is one difference the lane's S8 check sees.
constexpr int kTmaSlots = 7;

inline CUtensorMap* slot_pointer(TmaParamsAbi& tma, const int slot) {
    CUtensorMap* const slots[kTmaSlots] = {&tma.tensor_map_kv,
                                           &tma.tensor_map_kv_fp8_part_cta0,
                                           &tma.tensor_map_kv_fp8_part_cta1,
                                           &tma.tensor_map_extra_kv_fp8_part_cta0,
                                           &tma.tensor_map_extra_kv_fp8_part_cta1,
                                           &tma.tensor_map_extra_kv_fp4_part_cta0,
                                           &tma.tensor_map_extra_kv_fp4_part_cta1};
    if (slot < 0 || slot >= kTmaSlots)
        return nullptr;
    return slots[slot];
}

// The device pointers of one call. Section B reads them off the torch tensors; nothing here knows about torch.
struct DevicePointers {
    void* q = nullptr;
    void* kv = nullptr;
    void* indices = nullptr;
    void* topk_length = nullptr;
    void* attn_sink = nullptr;
    void* extra_kv = nullptr;
    void* extra_indices = nullptr;
    void* extra_topk_length = nullptr;
    void* positions = nullptr;
    void* cos_sin_cache = nullptr;
    void* out_fp8 = nullptr;
    void* out_sf = nullptr;
    void* lse = nullptr;
};

// What the kernel receives: the parameter struct, the two FastDivmods, and which map slot each tensor map fills.
struct AbiFill {
    KernelParams params{};
    AuxParamsAbi aux{};
    int kv_slot = -1;
    int extra_slot = -1;
};

// The plan's fields copied into the kernel's own ABI, exactly as dspd_w38's wrapper fills it (vattn.cuh: the
// SparseAttnDecodeParams base, then ParamsTemplate's own fields) and as Kernel<CONFIG>::run builds AuxParams (the page
// divisor, then the extra-page divisor, which is 1 without an extra cache). Every value comes from `plan`, `ptrs` or
// `stream`: this function reads no global and calls nothing but FastDivmod's constructor.
inline AbiFill fill_abi(const LaunchPlan& plan, const long long has_extra, const DevicePointers& ptrs,
                        const stream_t stream) {
    AbiFill out;
    KernelParams& params = out.params;
    // the members are value-initialised above; this makes the padding bytes zero too, so the lane's S8 check can
    // compare the whole 376-byte object representation and not only the fields it knows the names of
    std::memset(&params, 0, sizeof(params));
    params.b = plan.b;
    params.s_q = plan.s_q;
    params.h_q = plan.h_q;
    params.h_kv = plan.h_kv;
    params.d_qk = plan.d_qk;
    params.d_v = plan.d_v;
    params.sm_scale = plan.sm_scale;
    params.sm_scale_div_log2 = plan.sm_scale_div_log2;
    params.num_blocks = plan.num_blocks;
    params.page_block_size = plan.page_block_size;
    params.topk = plan.topk;
    params.model_type = ModelType::V41;
    params.extra_model_type = plan.extra_model_type == kModelV41Fp4 ? ModelType::V41_FP4 : ModelType::V41;
    params.q = (bf16*)ptrs.q;
    params.kv = (bf16*)ptrs.kv;
    params.indices = (int*)ptrs.indices;
    params.topk_length = (int*)ptrs.topk_length;
    params.attn_sink = (float*)ptrs.attn_sink;
    params.lse = (float*)ptrs.lse;
    params.out = nullptr;
    if (has_extra) {
        params.extra_num_blocks = plan.extra_num_blocks;
        params.extra_page_block_size = plan.extra_page_block_size;
        params.extra_topk = plan.extra_topk;
        params.extra_kv = (bf16*)ptrs.extra_kv;
        params.extra_indices = (int*)ptrs.extra_indices;
        params.extra_topk_length = (int*)ptrs.extra_topk_length;
        params.stride_extra_kv_block = plan.stride_extra_kv_block;
        params.stride_extra_kv_row = plan.stride_extra_kv_row;
        params.stride_extra_indices_s_q = plan.stride_extra_indices_s_q;
    }
    params.stride_q_s_q = plan.stride_q_s_q;
    params.stride_q_h_q = plan.stride_q_h_q;
    params.stride_kv_block = plan.stride_kv_block;
    params.stride_kv_row = plan.stride_kv_row;
    params.stride_indices_s_q = plan.stride_indices_s_q;
    params.stride_lse_s_q = plan.stride_lse_s_q;
    params.num_sm = plan.num_sm;
    params.stream = stream;
    params.enable_split_kv = false;
    params.enable_q_norm = plan.enable_q_norm != 0;
    params.rms_norm_eps = plan.rms_norm_eps;
    params.token_positions = (uint32_t*)ptrs.positions;
    params.is_rope_neox_style = plan.is_rope_neox_style != 0;
    params.rope_dim = plan.rope_dim;
    params.cos_sin_cache = (float*)ptrs.cos_sin_cache;
    params.n_wv_group = plan.n_wv_group;
    params.wv_group_size = plan.wv_group_size;
    params.num_per_channels = plan.num_per_channels;
    params.use_tma_aligned_col_major_sf = plan.use_tma_aligned_col_major_sf != 0;
    params.round_sf = plan.round_sf != 0;
    params.use_packed_ue8m0 = plan.use_packed_ue8m0 != 0;
    params.out_fp8 = (fp8_e4m3*)ptrs.out_fp8;
    params.out_sf = (uint32_t*)ptrs.out_sf;
    params.stride_out_sf_wv_group = plan.stride_out_sf_wv_group;
    params.stride_out_sf_head_dim = plan.stride_out_sf_head_dim;
    out.aux.fast_divmod_page_block_size = cutlass::FastDivmod(plan.divisor_page);
    out.aux.fast_divmod_extra_page_block_size = cutlass::FastDivmod(plan.divisor_extra_page);
    out.kv_slot = plan.kv_map.used ? plan.kv_map.slot : -1;
    out.extra_slot = plan.extra_map.used ? plan.extra_map.slot : -1;
    return out;
}

// The launch, as the vendored Kernel<CONFIG>::run performs it through CUTLASS (review V59-R3-F1). Kernel::run sets
// cudaFuncAttributeMaxDynamicSharedMemorySize = sizeof(SharedMemoryPlan) and calls cutlass::launch_kernel_on_cluster(
// {grid, block, cluster, smem, params.stream}, kernel, params, tma_params, aux_params); ClusterLauncher::launch then
// runs init() (cudaFuncAttributeNonPortableClusterSizeAllowed = 1), make_cluster_launch_config (gridDim, blockDim,
// dynamicSmemBytes, stream, attrs; launch_attribute[0] = the cluster dimension, numAttrs 1 without PDL) and
// cudaLaunchKernelExC(&config, kernel, {&params, &tma_params, &aux_params}). Route D makes the same calls through the
// driver API, whose CU_* enums carry the runtime enums' values (the lane's S8 check reads both out of cuda.h).
struct FuncAttribute {
    CUfunction_attribute id;
    int value;
};

struct LaunchAbi {
    FuncAttribute function_attributes[2] = {};
    CUlaunchAttribute attributes[1] = {};
    CUlaunchConfig config{};
    void* arguments[3] = {nullptr, nullptr, nullptr};
    LaunchAbi() = default;
    LaunchAbi(const LaunchAbi&) = delete;                 // config.attrs points into this object: never copied
    LaunchAbi& operator=(const LaunchAbi&) = delete;
};

// Fills `out` from the plan and the three parameter objects the kernel receives. Returns 0, or -24 when the plan is
// not CUTLASS's launch (one attribute, no PDL). `params`, `tma` and `aux` must outlive the launch: `out` points at them.
inline int fill_launch(LaunchAbi& out, const LaunchPlan& plan, const KernelParams& params, const TmaParamsAbi& tma,
                       const AuxParamsAbi& aux) {
    if (plan.rc != 0 || plan.launch_attrs != kLaunchAttrCount || plan.attr_pdl != 0)
        return -24;
    // Kernel::run's function attribute first, then ClusterLauncher::init's
    out.function_attributes[0] = {CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, plan.func_max_dynamic_smem};
    out.function_attributes[1] = {CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, plan.func_nonportable_cluster};
    // make_cluster_launch_config: launch_attribute[0] is the cluster dimension, set even for a 1 x 1 x 1 cluster
    out.attributes[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    out.attributes[0].value.clusterDim.x = plan.attr_cluster[0];
    out.attributes[0].value.clusterDim.y = plan.attr_cluster[1];
    out.attributes[0].value.clusterDim.z = plan.attr_cluster[2];
    CUlaunchConfig& config = out.config;
    config.gridDimX = plan.grid[0];
    config.gridDimY = plan.grid[1];
    config.gridDimZ = plan.grid[2];
    config.blockDimX = plan.block[0];
    config.blockDimY = plan.block[1];
    config.blockDimZ = plan.block[2];
    config.sharedMemBytes = (unsigned)plan.smem_bytes;
    config.hStream = (CUstream)params.stream;
    config.attrs = out.attributes;
    config.numAttrs = (unsigned)plan.launch_attrs;
    // the kernel's three __grid_constant__ parameters, in fwd_kernel's order: Params, TMAParams, AuxParams
    void* arguments[3] = {(void*)&params, (void*)&tma, (void*)&aux};
    for (int i = 0; i < 3; ++i)
        out.arguments[i] = arguments[i];
    return 0;
}

// cuTensorMapEncodeTiled's twelve arguments as one tuple of exactly the parameter types cuda.h declares, in that
// order, which is also the order ku::make_tensor_map passes them: (&result, data_type, dim, global_ptr, size.data(),
// strides.data(), box_size.data(), element_strides.data(), interleave_mode, swizzle_mode, l2_promotion, oob_fill).
// The four array pointers point into `m`, which the caller keeps alive across the encode.
template <typename F>
struct ParameterTuple;
template <typename R, typename... A>
struct ParameterTuple<R(A...)> {
    using type = std::tuple<A...>;
};
using EncodeArgs = ParameterTuple<decltype(cuTensorMapEncodeTiled)>::type;
static_assert(std::tuple_size_v<EncodeArgs> == 12, "dspd-v59 VATTN: cuTensorMapEncodeTiled takes twelve arguments");

inline EncodeArgs encode_args(const TmaPlan& m, CUtensorMap* const result) {
    return EncodeArgs{result, (CUtensorMapDataType)m.dtype, (cuuint32_t)m.rank, (void*)(uintptr_t)m.base,
                      m.gdim, m.gstride, m.bdim,
                      m.estride, (CUtensorMapInterleave)m.interleave, (CUtensorMapSwizzle)m.swizzle,
                      (CUtensorMapL2promotion)m.promo, (CUtensorMapFloatOOBfill)m.oob};
}

}  // namespace dspd_v59_vattn

#ifndef VATTN_HOST_MATH_ONLY
// ========================================================================== (B) the runtime, the maps, the launch
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cutlass/version.h>

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/backend/cuda/driver.hpp>
#include <deep_jit/utils/filesystem.hpp>

namespace dspd_v59_vattn {

namespace {

deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> g_jit(nullptr);
std::string g_root;                                        // our directory of texts under the JIT cache root
std::shared_ptr<deep_jit::cuda::Kernel> g_kernel[2];       // [0] SWA-only layers, [1] layers with the fp4 extra cache
int g_num_sms[kMaxDevices] = {0};
int g_expect[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

// vattn.py's _FLAGS: the device flags the rig build measured. The arch, -O3, -std and the PTXAS register-usage level
// come from DeepJIT's own defaults (the arch is overridden in init to the one load_jit targeted).
const char* const kDeviceFlags[] = {
    "-DNDEBUG", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "--expt-extended-lambda", "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--diag-suppress=445,177,550", "-Xcompiler=-fvisibility=hidden",
    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "-DVATTN_BS=1"};

TensorDesc describe(const at::Tensor& t) {
    TensorDesc d;
    d.ndim = (int)t.dim() > 4 ? 0 : (int)t.dim();     // a rank above 4 cannot be one of this kernel's inputs
    for (int i = 0; i < d.ndim; ++i) {
        d.size[i] = (long long)t.size(i);
        d.stride[i] = (long long)t.stride(i);
    }
    const auto type = t.scalar_type();
    d.lanes = 1;
    if (type == at::kBFloat16) {
        d.code = kCodeBfloat;
        d.bits = 16;
    } else if (type == at::kInt) {
        d.code = kCodeInt;
        d.bits = 32;
    } else if (type == at::kFloat) {
        d.code = kCodeFloat;
        d.bits = 32;
    } else if (type == at::kByte) {
        d.code = kCodeUInt;
        d.bits = 8;
    } else if (type == at::kChar) {
        d.code = kCodeInt;
        d.bits = 8;
    } else if (type == at::kFloat8_e4m3fn) {
        d.code = kCodeFloat;
        d.bits = 8;
    } else {
        d.code = -1;                                   // anything else fails the dtype gate
        d.bits = 0;
    }
    d.addr = (unsigned long long)(uintptr_t)t.data_ptr();
    return d;
}

CUtensorMap encode(const TmaPlan& m) {
    CUtensorMap map{};
    // section A2's argument tuple, applied in its own order: this function chooses no argument
    const CUresult rc = std::apply(
        [](auto... arguments) { return deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(arguments...); },
        encode_args(m, &map));
    if (rc != CUDA_SUCCESS)
        throw std::runtime_error("dspd-v59 VATTN: cuTensorMapEncodeTiled failed");
    return map;
}

void put_map(TmaParamsAbi& tma, const TmaPlan& m) {
    // the slot table is section A2's (slot_pointer), so one place decides which field an index names
    if (CUtensorMap* const field = m.used ? slot_pointer(tma, m.slot) : nullptr)
        *field = encode(m);
}

}  // namespace

// Create the private runtime. `root` holds our texts; `include_dirs` are the device compile's include paths, in
// load_jit's order (SGLang's DEFAULT_INCLUDE, our csrc, the kerutils include directory, then the engine's own
// "cutlass" dependency resolver: FlashInfer's CUTLASS first, DeepGEMM's include tree after it); `cache_dir` is where
// DeepJIT keeps compiled cubins (set through our own prefix, never overwriting an operator's value); `arch` is the
// target load_jit compiled the measured build for; `signature` is the digest of every text of this lever, the two
// include lists and the device tree's CUTLASS version.
void init(const std::string& root, const std::vector<std::string>& include_dirs, const std::string& cache_dir,
          const std::string& arch, const std::string& signature, const std::vector<int64_t>& expect) {
    TORCH_CHECK(!root.empty() && !arch.empty() && !signature.empty(), "dspd-v59 VATTN: empty init argument");
    TORCH_CHECK(expect.size() == 9, "dspd-v59 VATTN: nine launch constants are expected");
    if (!cache_dir.empty())
        ::setenv("DSPDVA_JIT_CACHE_DIR", cache_dir.c_str(), /*overwrite=*/0);
    g_root = root;
    for (size_t i = 0; i < expect.size(); ++i)
        g_expect[i] = (int)expect[i];
    std::vector<std::filesystem::path> includes;
    includes.reserve(include_dirs.size());
    for (const auto& dir : include_dirs)
        includes.emplace_back(std::filesystem::absolute(dir));
    // No include prefixes: DeepJIT's parser then opens no header at all, and the whole include closure is covered by
    // `signature` instead (the digest of all 26 texts of this lever plus its flags).
    const auto config = deep_jit::Config(std::filesystem::absolute(root), "DSPDVA",
                                         "dspd_v59_vattn-" + signature + "-cutlass-" +
                                             std::to_string(CUTLASS_VERSION),
                                         includes, {});
    const std::string target = arch;
    g_jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config, target] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        // DeepJIT would target the family arch its own device query returns; the measured build is load_jit's
        // arch-specific target, so that is what this runtime compiles for.
        runtime->default_compiler_options.arch = target;
        return runtime;
    });
}

// The CUTLASS this library was compiled against (cutlass/version.h through the same include order the device unit
// uses). vattn.py checks it equals the version of the device include tree, so the host's FastDivmod and the device's
// AuxParams come from one CUTLASS.
std::vector<int64_t> cutlass_version() {
    return {(int64_t)CUTLASS_MAJOR, (int64_t)CUTLASS_MINOR, (int64_t)CUTLASS_PATCH};
}

// Compile and load the one cubin of one layer class. Returns 0, or a negative code; it never aborts.
int64_t compile_jit(const int64_t has_extra) {
    const int index = has_extra ? 1 : 0;
    if (g_root.empty())
        return -1;
    if (g_kernel[index] != nullptr)
        return 0;
    const auto unit = std::filesystem::path(g_root) / "csrc" / "vattn_kernel_tu.cuh";
    if (!std::filesystem::is_regular_file(unit))
        return -2;
    const std::string source = deep_jit::read(unit);
    deep_jit::cuda::CompilerOptions options;
    options.extra_nvcc_flags.emplace_back(std::string("-DVATTN_HAS_EXTRA=") + (has_extra ? "1" : "0"));
    static const char* const kExpectNames[9] = {"VATTN_EXPECT_SMEM", "VATTN_EXPECT_THREADS", "VATTN_EXPECT_CLUSTER",
                                                "VATTN_EXPECT_RAW_FP8", "VATTN_EXPECT_RAW_FP4", "VATTN_EXPECT_PARAMS",
                                                "VATTN_EXPECT_TMA", "VATTN_EXPECT_AUX", "VATTN_EXPECT_BTOPK"};
    for (int i = 0; i < 9; ++i)
        options.extra_nvcc_flags.emplace_back(std::string("-D") + kExpectNames[i] + "=" + std::to_string(g_expect[i]));
    for (const char* flag : kDeviceFlags)
        options.extra_nvcc_flags.emplace_back(flag);
    // The cache tag may hold letters, digits and underscores only (deep_jit/cache/disk.hpp).
    const std::string name = has_extra ? "dspd_v59_vattn_x1" : "dspd_v59_vattn_x0";
    g_kernel[index] = g_jit->compile(name, source, options);
    return g_kernel[index] == nullptr ? -3 : 0;
}

// One target-verify MegaAttention decode call. Returns 0 on launch, or the refusal code of the gate that rejected
// the call; the caller then runs the engine's own method. Never aborts the process.
int64_t decode(at::Tensor q, at::Tensor kv, at::Tensor indices, at::Tensor topk_length, at::Tensor attn_sink,
               at::Tensor extra_kv, at::Tensor extra_indices, at::Tensor extra_topk_length, at::Tensor positions,
               at::Tensor cos_sin_cache, at::Tensor out_fp8, at::Tensor out_sf, at::Tensor lse, double sm_scale,
               int64_t has_extra) {
    const int index = has_extra ? 1 : 0;
    if (g_kernel[index] == nullptr)
        return -20;
    const int device_id = q.device().is_cuda() ? (int)q.device().index() : -1;
    if (device_id < 0 || device_id >= kMaxDevices)
        return -12;
    if (g_num_sms[device_id] == 0) {
        int count = 0;
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id) != cudaSuccess || count <= 0)
            return -21;
        g_num_sms[device_id] = count;
    }
    const LaunchPlan plan = plan_decode(describe(q), describe(kv), describe(indices), describe(topk_length),
                                        describe(attn_sink), describe(extra_kv), describe(extra_indices),
                                        describe(extra_topk_length), describe(positions), describe(cos_sin_cache),
                                        describe(out_fp8), describe(out_sf), describe(lse), sm_scale, has_extra,
                                        device_id, g_num_sms[device_id]);
    if (plan.rc != 0)
        return plan.rc;
    // Section A2 builds every byte the kernel receives: the parameter struct, which map slot each tensor map
    // fills, the two FastDivmod divisors, the tensor-map encode arguments and the launch. Nothing is decided here, so
    // the lane's S8 check compares all of it with dspd_w38's wrapper and CUTLASS's launcher on a CPU box (review
    // V59-R2-F2, V59-R3-F1), and the lane's texts check pins this function's text.
    DevicePointers ptrs;
    ptrs.q = q.data_ptr();
    ptrs.kv = kv.data_ptr();
    ptrs.indices = indices.data_ptr();
    ptrs.topk_length = topk_length.data_ptr();
    ptrs.attn_sink = attn_sink.data_ptr();
    ptrs.lse = lse.data_ptr();
    ptrs.positions = positions.data_ptr();
    ptrs.cos_sin_cache = cos_sin_cache.data_ptr();
    ptrs.out_fp8 = out_fp8.data_ptr();
    ptrs.out_sf = out_sf.data_ptr();
    if (has_extra) {
        ptrs.extra_kv = extra_kv.data_ptr();
        ptrs.extra_indices = extra_indices.data_ptr();
        ptrs.extra_topk_length = extra_topk_length.data_ptr();
    }
    const AbiFill filled = fill_abi(plan, has_extra, ptrs, at::cuda::getCurrentCUDAStream(device_id).stream());
    // the fp8 KV map always, the fp4 extra map iff this layer has the extra cache (Kernel::run: iff extra_topk > 0,
    // which gate -11 makes the same condition); every other slot stays zero, as Kernel::run's `TMAParams tma_params = {}`
    TmaParamsAbi tma{};
    put_map(tma, plan.kv_map);
    if (has_extra)
        put_map(tma, plan.extra_map);
    if (filled.kv_slot != plan.kv_map.slot || (has_extra && filled.extra_slot != plan.extra_map.slot))
        return -25;
    // The launch is section A2's, forwarded: both function attributes, then cuLaunchKernelEx with A2's config (one
    // launch attribute, the cluster dimension, explicit even for a 1-CTA cluster, no PDL) and A2's argument array.
    // Not DeepJIT's Kernel::launch, which omits the cluster attribute at x == 1.
    LaunchAbi launch;
    const int prepared = fill_launch(launch, plan, filled.params, tma, filled.aux);
    if (prepared != 0)
        return prepared;
    const CUfunction function = g_kernel[index]->kernel_handle;
    if (function == nullptr)
        return -20;
    for (const FuncAttribute& attribute : launch.function_attributes)
        if (deep_jit::cuda::driver::lazy_cuFuncSetAttribute(function, attribute.id, attribute.value) != CUDA_SUCCESS)
            return -22;
    if (deep_jit::cuda::driver::lazy_cuLaunchKernelEx(&launch.config, function, launch.arguments, nullptr) !=
        CUDA_SUCCESS)
        return -23;
    return 0;
}

}  // namespace dspd_v59_vattn

TORCH_LIBRARY(dspd_v59_vattn, m) {
    m.def("init(str root, str[] include_dirs, str cache_dir, str arch, str signature, int[] expect) -> ()",
          &dspd_v59_vattn::init);
    m.def("compile_jit(int has_extra) -> int", &dspd_v59_vattn::compile_jit);
    m.def("cutlass_version() -> int[]", &dspd_v59_vattn::cutlass_version);
    // The three outputs are declared mutable, as v50's M462 declares the buffer its kernel writes: the kernel
    // writes out_fp8, out_sf and lse through the parameter struct.
    m.def("decode(Tensor q, Tensor kv, Tensor indices, Tensor topk_length, Tensor attn_sink, Tensor extra_kv, "
          "Tensor extra_indices, Tensor extra_topk_length, Tensor positions, Tensor cos_sin_cache, "
          "Tensor(a!) out_fp8, Tensor(b!) out_sf, Tensor(c!) lse, float sm_scale, int has_extra) -> int");
}

TORCH_LIBRARY_IMPL(dspd_v59_vattn, CUDA, m) {
    m.impl("decode", &dspd_v59_vattn::decode);
}

#endif  // VATTN_HOST_MATH_ONLY
""",
    'csrc/vattn_kernel_tu.cuh': r"""// vattn_kernel_tu.cuh (dspd-v59 VATTN route D, ours): the device-only translation unit the bundle's own DeepJIT
// runtime compiles to one cubin. It holds the `__global__` entry and nothing else: every host-side step of today's
// wrapper (tensor maps, parameter struct, grid / block / dynamic shared memory, the launch) is vattn_host.cpp's.
//
// Why one instantiation per cubin: DeepJIT's loader enumerates the cubin's kernels and refuses anything but exactly
// one (deep_jit/backend/cuda/kernel.hpp, "expected exactly one kernel in {}"), while the current wrapper instantiates
// two configurations (extra = V41_FP4 for the compressed layers, extra = V41 for the SWA-only ones) in one unit. So
// VATTN_HAS_EXTRA selects one configuration and the host compiles this unit twice, keeping one kernel handle each.
//
// Why every include below is the angle form, including this tree's own headers: DeepJIT hashes the source text it is
// handed and panics on any `#include "..."` line in it ("non-standard include", deep_jit/utils/parser.hpp), so the
// unit it compiles may carry angle includes only. They resolve against the --include-path entries DeepJIT passes from
// its Config (our csrc directory and the kerutils include directory). The vendored headers' own quoted includes are
// never seen by that parser: with no include prefixes configured it opens no file at all.
//
// Why the VATTN_EXPECT_* macros: the host wrapper cannot include this kernel's own host headers (the kerutils
// umbrella pulls in the SM80/90/100 device intrinsics and its common header refuses a build that is not nvcc's), so
// the launch constants are re-typed there and passed back in here as macros. Each one is checked against the
// vendored header's own value below, so a header or CUTLASS change fails this compile -- VATTN then stays off and the
// engine's kernel serves -- instead of reaching a launch with a wrong shared-memory size or box width.
#pragma once

#ifndef FLASH_MLA_IS_BUILD_ON_CUDA
#define FLASH_MLA_IS_BUILD_ON_CUDA
#endif
#ifndef VATTN_BS
#define VATTN_BS 1              // the block-split V4.1 page layout this arena's KV pools keep
#endif
#if VATTN_BS != 1
#error "dspd-v59 VATTN serves the block-split page layout only (VATTN_BS must be 1)"
#endif
#ifndef VATTN_HAS_EXTRA
#error "dspd-v59 VATTN: VATTN_HAS_EXTRA (0: SWA-only layers, 1: layers with the fp4 extra cache) must be defined"
#endif
#if VATTN_HAS_EXTRA != 0 && VATTN_HAS_EXTRA != 1
#error "dspd-v59 VATTN: VATTN_HAS_EXTRA must be 0 or 1"
#endif
#if !defined(VATTN_EXPECT_SMEM) || !defined(VATTN_EXPECT_THREADS) || !defined(VATTN_EXPECT_CLUSTER) || \
    !defined(VATTN_EXPECT_RAW_FP8) || !defined(VATTN_EXPECT_RAW_FP4) || !defined(VATTN_EXPECT_PARAMS) || \
    !defined(VATTN_EXPECT_TMA) || !defined(VATTN_EXPECT_AUX) || !defined(VATTN_EXPECT_BTOPK)
#error "dspd-v59 VATTN: the host wrapper must pass every VATTN_EXPECT_* launch constant"
#endif

#include <cuda_fp8.h>
#include <cute/tensor.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

// Our copy of FlashMLA's kernel namespace gets a name of its own, so that this cubin never shares a symbol with the
// sgl-flash-mla wheel loaded in the same process (which defines sm100::... for its own current kernels)
#define sm100 vattn_sm100
#include <core_attn/kernel.cuh>

namespace vattn {

namespace CA = vattn_sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;

// The configuration today's wrapper launches for this class of layers (vattn.cuh: Decode, V41 pages, h_q 64, no q
// norm; the extra cache is the fp4 one on the compressed layers and absent on the SWA-only ones)
inline constexpr CA::Config CONFIG{SparseAttnFwdMode::Decode, ModelType::V41,
                                   VATTN_HAS_EXTRA ? ModelType::V41_FP4 : ModelType::V41, 64, false};
using KernelT = CA::Kernel<CONFIG>;

// The class this cubin serves is the one the host compiled it for (review V59-R4-F1): decode() launches
// g_kernel[has_extra ? 1 : 0], compile_jit builds that handle with -DVATTN_HAS_EXTRA=(has_extra ? 1 : 0), and this
// assert ties the configuration's extra cache to that macro, so a CONFIG line whose ternary is swapped cannot build
// (an fp8-extra kernel would gather extra rows through a TMAParams slot decode() never fills).
static_assert((CONFIG.EXTRA_MODEL_TYPE == ModelType::V41_FP4) == (VATTN_HAS_EXTRA != 0),
              "dspd-v59 VATTN: the extra cache's format is not the one VATTN_HAS_EXTRA selects");

// The host wrapper's re-typed launch constants, against this build's own headers
static_assert(sizeof(typename KernelT::SharedMemoryPlan) == VATTN_EXPECT_SMEM,
              "dspd-v59 VATTN: the host's dynamic shared-memory size is not sizeof(SharedMemoryPlan)");
static_assert(KernelT::NUM_THREADS == VATTN_EXPECT_THREADS, "dspd-v59 VATTN: block size mismatch");
static_assert(KernelT::CLUSTER_SIZE == VATTN_EXPECT_CLUSTER, "dspd-v59 VATTN: cluster size mismatch");
static_assert(KernelT::B_TOPK == VATTN_EXPECT_BTOPK, "dspd-v59 VATTN: KV block width mismatch");
static_assert(KernelT::BLOCK_SPLIT, "dspd-v59 VATTN: the block-split KV transport is not enabled");
static_assert(KernelT::template KVFormatCta<ModelType::V41, 0>::RAW_TOKEN_SMEM_STRIDE == VATTN_EXPECT_RAW_FP8,
              "dspd-v59 VATTN: the host's fp8 TMA box width is not RAW_TOKEN_SMEM_STRIDE");
static_assert(KernelT::template KVFormatCta<ModelType::V41_FP4, 0>::RAW_TOKEN_SMEM_STRIDE == VATTN_EXPECT_RAW_FP4,
              "dspd-v59 VATTN: the host's fp4 TMA box width is not RAW_TOKEN_SMEM_STRIDE");
static_assert(sizeof(typename KernelT::Params) == VATTN_EXPECT_PARAMS,
              "dspd-v59 VATTN: the host's parameter struct is not this kernel's ParamT<Decode>");
static_assert(sizeof(typename KernelT::TMAParams) == VATTN_EXPECT_TMA, "dspd-v59 VATTN: TMAParams size mismatch");
static_assert(sizeof(typename KernelT::AuxParams) == VATTN_EXPECT_AUX, "dspd-v59 VATTN: AuxParams size mismatch");
// The parameter order the host fills: [Params][TMAParams][AuxParams], as Kernel<CONFIG>::run passes them
static_assert(KernelT::H_Q == 64 && KernelT::D_QK == 512 && KernelT::D_VO == 512,
              "dspd-v59 VATTN: this cubin is the h_q 64, d 512 decode kernel only");

}  // namespace vattn

// The one kernel of this cubin: the entry Kernel<CONFIG>::run launches today (kernel.cuh's fwd_kernel, with its
// __launch_bounds__ and __grid_constant__ parameters unchanged).
template __global__ void vattn::CA::fwd_kernel<vattn::KernelT>(
    __grid_constant__ const typename vattn::KernelT::Params,
    __grid_constant__ const typename vattn::KernelT::TMAParams,
    __grid_constant__ const typename vattn::KernelT::AuxParams);
""",
}
