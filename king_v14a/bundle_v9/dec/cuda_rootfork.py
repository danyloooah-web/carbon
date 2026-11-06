NAME = "rootfork.cuh"
TAG = "v13s-rootfork-1"

# king-v13s ROOT-FORK: three host-only driver calls (no kernel), resolved through cudaGetDriverEntryPointByVersion at
# the CUDA 12.0 ABI as peer_lookup.cuh resolves its own: the capture state of a stream (status, capture id, dependency
# nodes and their node types), emptying a capturing stream's dependency set (its next node is a graph root), and a
# non-blocking stream no torch pool hands out. Every call writes its CUresult instead of throwing.
SOURCE = r"""
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace dec_rootfork {

constexpr int kMaxDeps = 8;
constexpr int kInfoWords = 4 + 2 * kMaxDeps;

template <typename F>
inline F driver_fn(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult status;
  const cudaError_t err = cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &status);
  sglang::host::RuntimeCheck(err == cudaSuccess && status == cudaDriverEntryPointSuccess && fn != nullptr,
                             "king-v13s ROOT-FORK: driver entry point not found: ", name);
  return reinterpret_cast<F>(fn);
}

inline int64_t* cpu_words(tvm::ffi::TensorView out, int64_t words) {
  sglang::host::RuntimeCheck(out.device().device_type == kDLCPU && out.dtype().bits == 64 && out.numel() == words,
                             "king-v13s ROOT-FORK: the output is not a CPU int64 tensor of the expected length");
  return static_cast<int64_t*>(out.data_ptr());
}

}  // namespace dec_rootfork

struct DecRootFork {
  // out[0] CUresult, [1] capture status, [2] capture id, [3] dependency count, then (node, node type) pairs
  static void capture_info(int64_t stream, tvm::ffi::TensorView out) {
    using Info = CUresult (*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, size_t*);
    using NodeType = CUresult (*)(CUgraphNode, CUgraphNodeType*);
    static const Info info = dec_rootfork::driver_fn<Info>("cuStreamGetCaptureInfo");
    static const NodeType node_type = dec_rootfork::driver_fn<NodeType>("cuGraphNodeGetType");
    int64_t* w = dec_rootfork::cpu_words(out, dec_rootfork::kInfoWords);
    for (int i = 0; i < dec_rootfork::kInfoWords; ++i) w[i] = -1;
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    cuuint64_t id = 0;
    CUgraph graph = nullptr;
    const CUgraphNode* deps = nullptr;
    size_t n = 0;
    const CUresult rc = info(reinterpret_cast<CUstream>(stream), &status, &id, &graph, &deps, &n);
    w[0] = static_cast<int64_t>(rc);
    if (rc != CUDA_SUCCESS) return;
    const bool active = status == CU_STREAM_CAPTURE_STATUS_ACTIVE;
    w[1] = static_cast<int64_t>(status);
    w[2] = static_cast<int64_t>(id);
    w[3] = active ? static_cast<int64_t>(n) : 0;
    for (size_t i = 0; active && i < n && i < static_cast<size_t>(dec_rootfork::kMaxDeps); ++i) {
      CUgraphNodeType kind;
      w[4 + 2 * i] = reinterpret_cast<int64_t>(deps[i]);
      w[5 + 2 * i] = node_type(deps[i], &kind) == CUDA_SUCCESS ? static_cast<int64_t>(kind) : -1;
    }
  }

  // out[0]: CUresult of emptying a capturing stream's dependency set
  static void empty_deps(int64_t stream, tvm::ffi::TensorView out) {
    using Update = CUresult (*)(CUstream, CUgraphNode*, size_t, unsigned int);
    static const Update update = dec_rootfork::driver_fn<Update>("cuStreamUpdateCaptureDependencies");
    int64_t* w = dec_rootfork::cpu_words(out, 1);
    w[0] = static_cast<int64_t>(
        update(reinterpret_cast<CUstream>(stream), nullptr, 0, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
  }

  // out[0]: CUresult, out[1]: a new non-blocking stream of the calling thread's current context
  static void create_stream(tvm::ffi::TensorView out) {
    using Create = CUresult (*)(CUstream*, unsigned int);
    static const Create create = dec_rootfork::driver_fn<Create>("cuStreamCreate");
    int64_t* w = dec_rootfork::cpu_words(out, 2);
    CUstream s = nullptr;
    w[0] = static_cast<int64_t>(create(&s, CU_STREAM_NON_BLOCKING));
    w[1] = reinterpret_cast<int64_t>(s);
  }
};
"""
