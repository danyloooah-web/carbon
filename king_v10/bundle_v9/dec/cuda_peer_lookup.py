NAME = "peer_lookup.cuh"
TAG = "7f055ac9382d5097"

SOURCE = r"""
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace dec_peer_lookup {

constexpr int kHandleBytes = 80;

template <typename F>
inline F driver_fn(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult status;
  const cudaError_t err = cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &status);
  sglang::host::RuntimeCheck(err == cudaSuccess && status == cudaDriverEntryPointSuccess && fn != nullptr,
                             "bundle-v9 PEER_LOOKUP: driver entry point not found: ", name);
  return reinterpret_cast<F>(fn);
}

__global__ void mirror_kernel(const int64_t* __restrict__ src, volatile int64_t* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

}

struct DecPeerLookup {

  static void export_handle(int64_t ptr, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    using GetRange = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);
    using GetAttr = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 8 && out.numel() == dec_peer_lookup::kHandleBytes,
                 "bundle-v9 PEER_LOOKUP: export_handle writes a CPU u8 [80]");
    static const GetRange get_range = dec_peer_lookup::driver_fn<GetRange>("cuMemGetAddressRange");
    static const GetAttr get_attr = dec_peer_lookup::driver_fn<GetAttr>("cuPointerGetAttribute");
    CUdeviceptr base = 0;
    size_t size = 0;
    RuntimeCheck(get_range(&base, &size, static_cast<CUdeviceptr>(ptr)) == CUDA_SUCCESS,
                 "bundle-v9 PEER_LOOKUP: cuMemGetAddressRange failed");
    int capable = 0;
    RuntimeCheck(get_attr(&capable, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, base) == CUDA_SUCCESS,
                 "bundle-v9 PEER_LOOKUP: cuPointerGetAttribute failed");
    RuntimeCheck(capable != 0, "bundle-v9 PEER_LOOKUP: the Engram shard's allocation is not legacy-IPC capable "
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
    RuntimeCheck(in.device().device_type == kDLCPU && in.dtype().bits == 8 && in.numel() == dec_peer_lookup::kHandleBytes,
                 "bundle-v9 PEER_LOOKUP: open_handle reads a CPU u8 [80]");
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 64 && out.numel() == 1,
                 "bundle-v9 PEER_LOOKUP: open_handle writes a CPU int64 [1]");
    const auto* bytes = static_cast<const uint8_t*>(in.data_ptr());
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, bytes, sizeof(handle));
    void* base = nullptr;
    RuntimeDeviceCheck(cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess));
    *static_cast<int64_t*>(out.data_ptr()) = reinterpret_cast<int64_t>(base);
  }

  static void host_device_ptr(int64_t host_ptr, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 64 && out.numel() == 1,
                 "bundle-v9 PEER_LOOKUP: host_device_ptr writes a CPU int64 [1]");
    void* dev = nullptr;
    RuntimeDeviceCheck(cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(host_ptr), 0));
    *static_cast<int64_t*>(out.data_ptr()) = reinterpret_cast<int64_t>(dev);
  }

  static void mirror(tvm::ffi::TensorView src, int64_t dst) {
    using namespace sglang::host;
    RuntimeCheck(src.device().device_type == kDLCUDA && src.dtype().bits == 64,
                 "bundle-v9 PEER_LOOKUP: mirror reads a CUDA int64 tensor");
    const int n = static_cast<int>(src.numel());
    if (n == 0) return;
    LaunchKernel(dim3((n + 127) / 128), 128, src.device())
        .launch(dec_peer_lookup::mirror_kernel, static_cast<const int64_t*>(src.data_ptr()),
                reinterpret_cast<volatile int64_t*>(dst), n);
  }
};
"""
