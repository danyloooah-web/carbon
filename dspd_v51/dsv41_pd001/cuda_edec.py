"""dsv41-pd071 EDEC host helpers: generator/pd071/kernel/edec.cuh verbatim (written under SGLang's JIT cache root by edec._header). Regenerate with generator/pd071/gen_pd071.py, do not edit."""

NAME = "edec.cuh"
TAG = "c35b621b7e33dcd7"

SOURCE = r"""// dsv41-pd071 EDEC host helpers: legacy CUDA IPC export / open of the Engram table shards (the lookup itself is the
// fork's own Triton kernel sglang.kernels.ops.embeddings.engram_gather.engram_gather_peer), plus the twin's
// counter mirror (device counters copied into pinned host memory inside a captured graph; never launched by a
// production build).
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

namespace pd071_edec {

constexpr int kHandleBytes = 80;  // cudaIpcMemHandle_t (64) | offset (8) | allocation size (8)

template <typename F>
inline F driver_fn(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult status;
  const cudaError_t err = cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &status);
  sglang::host::RuntimeCheck(err == cudaSuccess && status == cudaDriverEntryPointSuccess && fn != nullptr,
                             "dsv41-pd071 EDEC: driver entry point not found: ", name);
  return reinterpret_cast<F>(fn);
}

__global__ void mirror_kernel(const int64_t* __restrict__ src, volatile int64_t* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

}  // namespace pd071_edec

struct Pd071Edec {
  // ptr: a device address inside a cudaMalloc allocation; out: CPU u8 [80] = the allocation's legacy IPC handle, ptr's
  // offset from its base, the allocation's size. Raises unless the allocation is legacy-IPC capable.
  static void export_handle(int64_t ptr, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    using GetRange = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);
    using GetAttr = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 8 && out.numel() == pd071_edec::kHandleBytes,
                 "dsv41-pd071 EDEC: export_handle writes a CPU u8 [80]");
    static const GetRange get_range = pd071_edec::driver_fn<GetRange>("cuMemGetAddressRange");
    static const GetAttr get_attr = pd071_edec::driver_fn<GetAttr>("cuPointerGetAttribute");
    CUdeviceptr base = 0;
    size_t size = 0;
    RuntimeCheck(get_range(&base, &size, static_cast<CUdeviceptr>(ptr)) == CUDA_SUCCESS,
                 "dsv41-pd071 EDEC: cuMemGetAddressRange failed");
    int capable = 0;
    RuntimeCheck(get_attr(&capable, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, base) == CUDA_SUCCESS,
                 "dsv41-pd071 EDEC: cuPointerGetAttribute failed");
    RuntimeCheck(capable != 0, "dsv41-pd071 EDEC: the Engram shard's allocation is not legacy-IPC capable "
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

  // in: CPU u8 [80] from a peer's export_handle; out: CPU int64 [1] = the peer allocation's base in this process
  // (the caller adds the offset and opens each handle once).
  static void open_handle(tvm::ffi::TensorView in, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    RuntimeCheck(in.device().device_type == kDLCPU && in.dtype().bits == 8 && in.numel() == pd071_edec::kHandleBytes,
                 "dsv41-pd071 EDEC: open_handle reads a CPU u8 [80]");
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 64 && out.numel() == 1,
                 "dsv41-pd071 EDEC: open_handle writes a CPU int64 [1]");
    const auto* bytes = static_cast<const uint8_t*>(in.data_ptr());
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, bytes, sizeof(handle));
    void* base = nullptr;
    RuntimeDeviceCheck(cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess));
    *static_cast<int64_t*>(out.data_ptr()) = reinterpret_cast<int64_t>(base);
  }

  // host_ptr: a pinned (cudaHostAlloc) host address; out: CPU int64 [1] = its device address.
  static void host_device_ptr(int64_t host_ptr, tvm::ffi::TensorView out) {
    using namespace sglang::host;
    RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 64 && out.numel() == 1,
                 "dsv41-pd071 EDEC: host_device_ptr writes a CPU int64 [1]");
    void* dev = nullptr;
    RuntimeDeviceCheck(cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(host_ptr), 0));
    *static_cast<int64_t*>(out.data_ptr()) = reinterpret_cast<int64_t>(dev);
  }

  // The twin only: src (CUDA int64, contiguous) copied to the pinned host buffer at device address dst, on the
  // current stream (a captured graph node when capturing).
  static void mirror(tvm::ffi::TensorView src, int64_t dst) {
    using namespace sglang::host;
    RuntimeCheck(src.device().device_type == kDLCUDA && src.dtype().bits == 64,
                 "dsv41-pd071 EDEC: mirror reads a CUDA int64 tensor");
    const int n = static_cast<int>(src.numel());
    if (n == 0) return;
    LaunchKernel(dim3((n + 127) / 128), 128, src.device())
        .launch(pd071_edec::mirror_kernel, static_cast<const int64_t*>(src.data_ptr()),
                reinterpret_cast<volatile int64_t*>(dst), n);
  }
};
"""
