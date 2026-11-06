
NAME = "peer_read.cuh"
TAG = "df1d422a991e46fc"

SOURCE = r"""#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace v9_peer_read {

constexpr int kWorld = 4;
constexpr int kCols = 24;
constexpr int kColsPerRank = 6;
constexpr int kDim = 256;
constexpr int kBlk = 32;
constexpr int kGroups = kCols * (kDim / kBlk);
constexpr int kHandleBytes = 80;

struct Tables {
  const uint8_t* w[kWorld];
  const uint8_t* s[kWorld];
  int64_t lo[kWorld];
  int64_t hi[kWorld];
};

__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ void ue8m0_scale(float amax, uint32_t& sf, float& mult) {
  const float normalized = __fmul_rn(amax, __uint_as_float(0x3B124925u));
  const int bits = __float_as_int(normalized);
  const int exponent = (bits >> 23) & 255;
  const int mantissa = bits & 0x7FFFFF;
  const bool bump = (mantissa != 0) && !((exponent == 0) && (mantissa <= 0x400000));
  const int e = min(exponent + (bump ? 1 : 0), 254);
  const int s = (normalized <= 0.0f) ? 0 : e;
  sf = static_cast<uint32_t>(s);
  mult = (s == 0) ? 0.0f : __int_as_float((254 - s) << 23);
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}

__global__ __launch_bounds__(kGroups) void peer_place_kernel(const int64_t* __restrict__ ids, int64_t ids_stride,
                                                            const __grid_constant__ Tables t, uint4* __restrict__ q,
                                                            uint32_t* __restrict__ s, int mine, int n_local) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x;
  const int gg = threadIdx.x;
  const int c = gg >> 3, k = gg & 7;
  const int owner = c / kColsPerRank;
  uint32_t qw[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  uint32_t sf = 0;
  if (row < mine) {
    const int64_t idx = ids[static_cast<int64_t>(row) * ids_stride + c];
    const bool owned = idx >= t.lo[owner] && idx < t.hi[owner];
    const int64_t local = owned ? idx - t.lo[owner] : 0;
    uint4 w0 = make_uint4(0u, 0u, 0u, 0u), w1 = w0;
    uint32_t e = 0;
    if (owned) {
      const uint4* src = reinterpret_cast<const uint4*>(t.w[owner] + local * kDim + k * 32);
      w0 = __ldg(src);
      w1 = __ldg(src + 1);
      e = __ldg(t.s[owner] + local * (kDim / kBlk) + k);
    }
    const float sc = (e == 0) ? __uint_as_float(0x00400000u) : __int_as_float(static_cast<int>(e << 23));
    const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    float f[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float prod = __fmul_rn(e4m3_to_f32((ww[i >> 2] >> (8 * (i & 3))) & 0xFFu), sc);
      f[i] = owned ? __bfloat162float(__float2bfloat16_rn(prod)) : 0.0f;
    }
    float amax = fabsf(f[0]);
#pragma unroll
    for (int i = 1; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
    float mult;
    ue8m0_scale(amax, sf, mult);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v[4];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) v[kk] = fminf(__fmul_rn(__fadd_rn(f[4 * i + kk], 0.0f), mult), 448.0f);
      qw[i] = cvt_e4m3x2(v[0], v[1]) | (cvt_e4m3x2(v[2], v[3]) << 16);
    }
  }
  if (row < n_local) {
    uint4* dst = q + (static_cast<int64_t>(row) * kGroups + gg) * 2;
    dst[0] = make_uint4(qw[0], qw[1], qw[2], qw[3]);
    dst[1] = make_uint4(qw[4], qw[5], qw[6], qw[7]);
  }
  uint32_t word = sf;
  word |= __shfl_down_sync(0xffffffffu, word, 1) << 8;
  word |= __shfl_down_sync(0xffffffffu, word, 2) << 16;
  if ((lane & 3) == 0) {
    s[static_cast<int64_t>(row >> 7) * (kGroups * 32) + (gg >> 2) * 128 + (row & 31) * 4 + ((row >> 5) & 3)] = word;
  }
}

__device__ __forceinline__ uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__global__ __launch_bounds__(kGroups) void peer_lookup_bf16_kernel(const int64_t* __restrict__ ids, int64_t ids_stride,
                                                                  const __grid_constant__ Tables t,
                                                                  uint4* __restrict__ out, int mine) {
  const int row = blockIdx.x;
  const int gg = threadIdx.x;
  const int c = gg >> 3, k = gg & 7;
  const int owner = c / kColsPerRank;
  uint32_t o[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i] = 0u;
  if (row < mine) {
    const int64_t idx = ids[static_cast<int64_t>(row) * ids_stride + c];
    const bool owned = idx >= t.lo[owner] && idx < t.hi[owner];
    if (owned) {
      const int64_t local = idx - t.lo[owner];
      const uint4* src = reinterpret_cast<const uint4*>(t.w[owner] + local * kDim + k * 32);
      const uint4 w0 = __ldg(src), w1 = __ldg(src + 1);
      const uint32_t e = __ldg(t.s[owner] + local * (kDim / kBlk) + k);
      const float sc = (e == 0) ? __uint_as_float(0x00400000u) : __int_as_float(static_cast<int>(e << 23));
      const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float a = __bfloat162float(__float2bfloat16_rn(
            __fmul_rn(e4m3_to_f32((ww[(2 * i) >> 2] >> (8 * ((2 * i) & 3))) & 0xFFu), sc)));
        const float b = __bfloat162float(__float2bfloat16_rn(
            __fmul_rn(e4m3_to_f32((ww[(2 * i + 1) >> 2] >> (8 * ((2 * i + 1) & 3))) & 0xFFu), sc)));
        o[i] = cvt_bf16x2(__fadd_rn(a, 0.0f), __fadd_rn(b, 0.0f));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) o[i] = cvt_bf16x2(__fadd_rn(0.0f, 0.0f), __fadd_rn(0.0f, 0.0f));
    }
  }
  uint4* dst = out + (static_cast<int64_t>(row) * kGroups + gg) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

template <typename F>
inline F driver_fn(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult status;
  const cudaError_t err = cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &status);
  sglang::host::RuntimeCheck(err == cudaSuccess && status == cudaDriverEntryPointSuccess && fn != nullptr,
                             "bundle-v9 PEER_READ: driver entry point not found: ", name);
  return reinterpret_cast<F>(fn);
}

}

struct V9PeerRead {

  static void export_handle(int64_t ptr, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    using GetRange = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);
    using GetAttr = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 8 && out.numel() == v9_peer_read::kHandleBytes,
                 "bundle-v9 PEER_READ: export_handle writes a CPU u8 [80]");
    static const GetRange get_range = v9_peer_read::driver_fn<GetRange>("cuMemGetAddressRange");
    static const GetAttr get_attr = v9_peer_read::driver_fn<GetAttr>("cuPointerGetAttribute");
    CUdeviceptr base = 0;
    size_t size = 0;
    RuntimeCheck(get_range(&base, &size, static_cast<CUdeviceptr>(ptr)) == CUDA_SUCCESS,
                 "bundle-v9 PEER_READ: cuMemGetAddressRange failed");
    int capable = 0;
    RuntimeCheck(get_attr(&capable, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, base) == CUDA_SUCCESS,
                 "bundle-v9 PEER_READ: cuPointerGetAttribute failed");
    RuntimeCheck(capable != 0, "bundle-v9 PEER_READ: the Engram shard's allocation is not legacy-IPC capable "
                               "(VMM / expandable segments / stream-ordered pool)");
    cudaIpcMemHandle_t handle;
    RuntimeDeviceCheck(cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(base)));
    auto* bytes = static_cast<uint8_t*>(out.data_ptr());
    const int64_t offset = ptr - static_cast<int64_t>(base);
    const int64_t total = static_cast<int64_t>(size);
    std::memcpy(bytes, &handle, sizeof(handle));
    std::memcpy(bytes + 64, &offset, 8);
    std::memcpy(bytes + 72, &total, 8);
  }

  static void open_handle(tvm::ffi::TensorView in, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    RuntimeCheck(in.device().device_type == kDLCPU && in.dtype().bits == 8 && in.numel() == v9_peer_read::kHandleBytes,
                 "bundle-v9 PEER_READ: open_handle reads a CPU u8 [80]");
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 64 && out.numel() == 1,
                 "bundle-v9 PEER_READ: open_handle writes a CPU int64 [1]");
    const auto* bytes = static_cast<const uint8_t*>(in.data_ptr());
    cudaIpcMemHandle_t handle;
    int64_t offset = 0;
    std::memcpy(&handle, bytes, sizeof(handle));
    std::memcpy(&offset, bytes + 64, 8);
    void* base = nullptr;
    RuntimeDeviceCheck(cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess));
    *static_cast<int64_t*>(out.data_ptr()) = reinterpret_cast<int64_t>(base) + offset;
  }

  static void peer_place(tvm::ffi::TensorView ids, tvm::ffi::TensorView tables, tvm::ffi::TensorView q,
                         tvm::ffi::TensorView s, int64_t mine) {
    using namespace sglang::host;
    auto NL = SymbolicSize{"n_local"};
    auto IS = SymbolicSize{"ids_stride"};
    auto dev = SymbolicDevice{};
    TensorMatcher({NL, v9_peer_read::kCols}).with_strides({IS, 1}).with_device<kDLCUDA>(dev).with_dtype<int64_t>()
        .verify(ids);
    TensorMatcher({NL, v9_peer_read::kGroups * 32}).with_strides({v9_peer_read::kGroups * 32, 1}).with_device<kDLCUDA>(dev)
        .verify(q);
    RuntimeCheck(q.dtype().bits == 8, "bundle-v9 PEER_READ: q must be a 1-byte dtype");
    const int64_t n_local = NL.unwrap();
    const int64_t n_pad = (n_local + 127) / 128 * 128;
    TensorMatcher({n_pad * v9_peer_read::kGroups}).with_device<kDLCUDA>(dev).with_dtype<uint8_t>().verify(s);
    RuntimeCheck(tables.device().device_type == kDLCPU && tables.dtype().bits == 64 &&
                     tables.numel() == 4 * v9_peer_read::kWorld,
                 "bundle-v9 PEER_READ: tables is a CPU int64 [4, 4]");
    RuntimeCheck(0 <= mine && mine <= n_local, "bundle-v9 PEER_READ: 0 <= mine <= n_local");
    RuntimeCheck(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(s.data_ptr()) % 4 == 0,
                 "bundle-v9 PEER_READ: misaligned outputs");
    if (n_pad == 0) return;
    RuntimeCheck(n_pad * v9_peer_read::kGroups < (int64_t(1) << 31), "bundle-v9 PEER_READ: too many rows");
    const auto* tv = static_cast<const int64_t*>(tables.data_ptr());
    v9_peer_read::Tables t;
    for (int r = 0; r < v9_peer_read::kWorld; ++r) {
      t.w[r] = reinterpret_cast<const uint8_t*>(tv[4 * r + 0]);
      t.s[r] = reinterpret_cast<const uint8_t*>(tv[4 * r + 1]);
      t.lo[r] = tv[4 * r + 2];
      t.hi[r] = tv[4 * r + 3];
      RuntimeCheck(t.w[r] != nullptr && t.s[r] != nullptr && t.lo[r] <= t.hi[r] &&
                       reinterpret_cast<uintptr_t>(t.w[r]) % 16 == 0,
                   "bundle-v9 PEER_READ: table entry");
    }
    LaunchKernel(dim3(static_cast<unsigned>(n_pad)), v9_peer_read::kGroups, dev.unwrap())(
        v9_peer_read::peer_place_kernel, static_cast<const int64_t*>(ids.data_ptr()), IS.unwrap(), t,
        static_cast<uint4*>(q.data_ptr()), static_cast<uint32_t*>(s.data_ptr()), static_cast<int>(mine),
        static_cast<int>(n_local));
  }

  static void peer_lookup_bf16(tvm::ffi::TensorView ids, tvm::ffi::TensorView tables, tvm::ffi::TensorView out,
                               int64_t mine) {
    using namespace sglang::host;
    auto NL = SymbolicSize{"n_local"};
    auto IS = SymbolicSize{"ids_stride"};
    auto dev = SymbolicDevice{};
    TensorMatcher({NL, v9_peer_read::kCols}).with_strides({IS, 1}).with_device<kDLCUDA>(dev).with_dtype<int64_t>()
        .verify(ids);
    TensorMatcher({NL, v9_peer_read::kGroups * 32}).with_strides({v9_peer_read::kGroups * 32, 1}).with_device<kDLCUDA>(dev)
        .with_dtype<sglang::bf16_t>().verify(out);
    const int64_t n_local = NL.unwrap();
    RuntimeCheck(0 <= mine && mine <= n_local, "bundle-v9 PEER_READ: 0 <= mine <= n_local");
    RuntimeCheck(tables.device().device_type == kDLCPU && tables.dtype().bits == 64 &&
                     tables.numel() == 4 * v9_peer_read::kWorld,
                 "bundle-v9 PEER_READ: tables is a CPU int64 [4, 4]");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "bundle-v9 PEER_READ: misaligned output");
    if (n_local == 0) return;
    RuntimeCheck(n_local * v9_peer_read::kGroups < (int64_t(1) << 31), "bundle-v9 PEER_READ: too many rows");
    const auto* tv = static_cast<const int64_t*>(tables.data_ptr());
    v9_peer_read::Tables t;
    for (int r = 0; r < v9_peer_read::kWorld; ++r) {
      t.w[r] = reinterpret_cast<const uint8_t*>(tv[4 * r + 0]);
      t.s[r] = reinterpret_cast<const uint8_t*>(tv[4 * r + 1]);
      t.lo[r] = tv[4 * r + 2];
      t.hi[r] = tv[4 * r + 3];
      RuntimeCheck(t.w[r] != nullptr && t.s[r] != nullptr && t.lo[r] <= t.hi[r] &&
                       reinterpret_cast<uintptr_t>(t.w[r]) % 16 == 0,
                   "bundle-v9 PEER_READ: table entry");
    }
    LaunchKernel(dim3(static_cast<unsigned>(n_local)), v9_peer_read::kGroups, dev.unwrap())(
        v9_peer_read::peer_lookup_bf16_kernel, static_cast<const int64_t*>(ids.data_ptr()), IS.unwrap(), t,
        static_cast<uint4*>(out.data_ptr()), static_cast<int>(mine));
  }
};
"""
