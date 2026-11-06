"""king-v7 DGT sources: the bundle-owned DeepGEMM host path for wo_a (dgt.py builds and loads it).

Data only (no import-time work). The library root dgt.py writes under the JIT cache root holds:
- KING_TEXTS: the king's own DeepGEMM host texts, read from cuda_moe_prefill.SOURCES at prepare and pinned here by
  sha256 (runtime, utils, heuristics, runtime_utils, the libdwfl stub); a different text turns DGT off;
- SOURCES: this lever's texts: its JIT runtime (the king's jit.hpp with its own signature), the FP8-output epilogue
  input, the image's b7f3816c sm100_fp8_fp4_gemm_1d1d.hpp trimmed to sm100_fp8_bmm, and the torch library entry;
- include/IMAGE_HEADER: the image's sm100_fp8_fp4_gemm_1d1d.cuh (sha256 IMAGE_HEADER_SHA256, read from the installed
  deep_gemm at prepare) with HEADER_EDITS applied, each anchor matching exactly once: one
  cudaTriggerProgrammaticLaunchCompletion() after the TMA-load warp's persistent loop. Every other include resolves
  in the image's DeepGEMM include tree (the JIT include roots are this root's include/, then the image's).
"""
ENTRY = "csrc/dgt_api.cpp"
CXX_FLAGS = ('-std=c++20', '-O3', '-fvisibility=hidden', '-fvisibility-inlines-hidden', '-Wno-psabi',
             '-Wno-deprecated-declarations')
LD_FLAGS = ('-lnvrtc', '-lcublasLt', '-lcublas')
IMAGE_HEADER = "deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh"
IMAGE_HEADER_SHA256 = "774d060d75a45fd38251305e75ef21cb1f9b62a96a532c4ad15b72ac049c6baa"
HEADER_EDITS = (
    (# the TMA-load warp's persistent loop ends; the MMA warp's branch begins
     '            }\n'
     '        }\n'
     '    } else if (warp_idx == 1 and is_leader_cta) {\n',
     '            }\n'
     '        }\n'
     '\n'
     '        // king-v7 DGT: this CTA has issued its last TMA load; let the dependent grid launch (scheduling only:\n'
     "        // the dependent's griddepcontrol.wait still waits for this whole grid to complete and flush)\n"
     '        cudaTriggerProgrammaticLaunchCompletion();\n'
     '    } else if (warp_idx == 1 and is_leader_cta) {\n'),
)
KING_TEXTS = {
    "build_include/elfutils/libdwfl.h": "8a4890f4d71e7aeb8d2510f5e03475619aef186410639e27814d3fc7d0a11647",
    "csrc/runtime/launch.hpp": "4c38b3633216f4755a5d3173a96f78324c899621b27f5313c86cce6e113d6a5c",
    "csrc/runtime/runtime.hpp": "c3c597a8ee46eaab2370f60fd615e0e97345e485bea47c4f48edd9ca0804b0d2",
    "csrc/utils/compatibility.hpp": "677b06bf1bcac2334fb9892c0d2df5b4c16dec099497ef5565c67b6085f1dd1f",
    "csrc/utils/exception.hpp": "f4c5c857ab11d9221cb59dda41eeb0548336271fa2d8284dce8065a86b40da42",
    "csrc/utils/layout.hpp": "20192b3b0b7c15ea3e1ab46600328df8a58606b70a5be6153fd9f15a0c6b7c96",
    "csrc/utils/math.hpp": "8beea2858474e9513790b18b822f22c6a68a64d3e8b6293006d9702d7de7ab1f",
    "csrc/jit_kernels/heuristics/common.hpp": "73523d907e3fa4ca87b2807a2aad1363ed4694f5ee5bedb91d6bed414a54ab69",
    "csrc/jit_kernels/heuristics/config.hpp": "b1b93e1ca5e7ca8082347a00458c13905a6c2a3a4fc3b4a0923009f1f320cc9d",
    "csrc/jit_kernels/heuristics/runtime.hpp": "c37fec805b281afd656c27b0935fa42001dbe781bd867aa4f17bc48894f584d6",
    "csrc/jit_kernels/heuristics/sm100.hpp": "c695e615cf2a32f227636f7999e1be8f223f217062afef76cdd506896a10e98d",
    "csrc/jit_kernels/heuristics/sm90.hpp": "0e88f692d17e6f80fff96cb10c2192f4e244445aeb2ee0279ab9d2b51e6598b5",
    "csrc/jit_kernels/heuristics/utils.hpp": "cb11066de2cb28b968a7fde31f1737c95ebf878423bdc3c86d34620fda796560",
    "csrc/jit_kernels/impls/runtime_utils.hpp": "633f52d4d55fe007f63f001f246d23fe6b3a8413d111c5efa840245f8fa9c430",
}
SOURCES = {
    "csrc/runtime/jit.hpp": r"""#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

// king-v7 DGT: the king's MOE-PREFILL JIT runtime (cuda_moe_prefill.py "csrc/runtime/jit.hpp") with its own extra
// signature, so this library's kernels never share a cache key with the king's or stock's; include roots are this
// library's own (the edited sm100_fp8_fp4_gemm_1d1d.cuh) ahead of the image's DeepGEMM include tree.
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path, const std::string& image_include_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION) + "-dgt-king-v7-wo-a",
        {include_dir, std::filesystem::absolute(image_include_path)},
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");

        const auto major = runtime->device.get_arch_major();
        const auto minor = runtime->device.get_arch_minor();
        if (major == 10 and minor != 1)
            runtime->default_compiler_options.arch = "100f";
        if (major == 12) {
            runtime->default_compiler_options.arch = "120f";
            runtime->default_compiler_options.nvcc_flags->emplace_back(
                "-gencode=arch=compute_120f,code=sm_120f");
        }
        return runtime;
    });
}

}
""",
    "csrc/jit_kernels/impls/epilogue.hpp": r"""#pragma once

#include <optional>
#include <string>

#include <torch/torch.h>

#include <deep_gemm/common/types.cuh>

#include "../../utils/exception.hpp"

namespace deep_gemm {

// king-v7 DGT: the one epilogue input wo_a's einsum needs, sgl-deep-gemm 0.2.3.post2's FP8 (d, sfd) output pair:
// `epilogue::transform::EpilogueDynamicScaledFP8` (the image _C.so's own type string) over the kernel's EpilogueArgs
// {sfd, sfd_stride, shape_m, shape_n, alpha} (deep_gemm/common/types.cuh). The kernel stores SF word
// (batch_idx * shape_n + n) / 128 of row r at sfd[word * sfd_stride + r] (epilogue/transform.cuh, store_sf), so
// sfd_stride is the MN-major SF tensor's column stride and shape_n the per-batch N. Other epilogues are refused.
struct EpilogueInput {
    std::string type;
    EpilogueArgs args;
};

static EpilogueInput make_epilogue_input(const int& m, const int& n,
                                         const std::optional<std::string>& epilogue_type,
                                         const std::optional<float>& alpha,
                                         const std::optional<torch::Tensor>& sfd = std::nullopt) {
    DG_HOST_ASSERT(not epilogue_type.has_value() and not alpha.has_value() and sfd.has_value());
    DG_HOST_ASSERT(sfd->scalar_type() == torch::kInt and sfd->dim() == 2 and sfd->stride(0) == 1);
    EpilogueArgs args;
    args.sfd = static_cast<uint32_t*>(sfd->data_ptr());
    args.sfd_stride = static_cast<uint32_t>(sfd->stride(1));
    args.shape_m = static_cast<uint32_t>(m);
    args.shape_n = static_cast<uint32_t>(n);
    args.alpha = 1.0f;
    return {"epilogue::transform::EpilogueDynamicScaledFP8", args};
}

}
""",
    "csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp": r"""#pragma once

// king-v7 DGT: sgl-deep-gemm b7f3816c (the image's 0.2.3.post2) csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp,
// trimmed to SM100FP8FP4Gemm1D1DRuntime, get_sf_block_config and sm100_fp8_bmm (text otherwise unchanged; the JIT
// template it renders is the image _C.so's own string).

#include <format>
#include <torch/torch.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100FP8FP4Gemm1D1DRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;
        // TODO: move into descriptor
        EpilogueInput epilogue;

        // TODO: move into descriptor
        int gran_k_a, gran_k_b, k_alignment;
        bool use_padded_sf_layout = true;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        // Per-32 K-major FP4xFP4 uses native MXF4; coarser scales and other operands use MXF8F6F4.
        //        Both share the same kernel template and only differ in the operand dtype strings.
        const auto a_dtype = to_string(args.gemm_desc.a_dtype, not args.gemm_desc.is_mxf4_mma());
        const auto b_dtype = to_string(args.gemm_desc.b_dtype, not args.gemm_desc.is_mxf4_mma());
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        args.gran_k_a, args.gran_k_b, args.k_alignment,
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages, args.gemm_config.pipeline_config.num_tma_store_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.gemm_config.layout.swap_ab, args.gemm_desc.ensure_zero_padding,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        a_dtype, b_dtype, to_string(args.gemm_desc.cd_dtype),
        args.epilogue.type, args.use_padded_sf_layout));

        // Launch
        jit->launch(
            kernel, args.options,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.epilogue.args,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd
        );
    }
};

// Compute the SF block sizes for the SF TMA descriptors.
// Both MXF4 and MXF8F6F4 use UTCCP-aligned SF blocks, with a per-128 K sub-block count (`block_k / 128`).
static std::tuple<int, int, int> get_sf_block_config(const GemmConfig& config, const GemmDesc& desc) {
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(
        config.layout.block_m, config.layout.block_n, desc.get_mma_kind());
    return {sf_block_m, sf_block_n, config.layout.block_k / 128};
}

static void sm100_fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                          const torch::Tensor& b, const torch::Tensor& sfb,
                          const std::optional<torch::Tensor>& c,
                          const torch::Tensor& d,
                          const int& batch_size, const int& m, const int& n, const int& k,
                          const int& gran_k_a, const int& gran_k_b,
                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                          const std::string& compiled_dims,
                          const std::optional<torch::Tensor>& sfd = std::nullopt) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b)
    };
    DG_HOST_ASSERT(sfd.has_value() == (d.scalar_type() == torch::kFloat8_e4m3fn));
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const int load_block_m = config.storage_config.load_block_m;
    const auto [inner_dim_a, outer_dim_a] = get_inner_outer_dims(major_a, k, m);
    const auto [inner_block_a, outer_block_a] = get_inner_outer_dims(major_a, config.layout.block_k, load_block_m);
    const auto tensor_map_a = make_tma_3d_desc(a, inner_dim_a, outer_dim_a, batch_size,
                                               inner_block_a, outer_block_a, 1,
                                               a.stride(major_a == cute::UMMA::Major::K ? 1 : 2),
                                               a.stride(0),
                                               config.storage_config.swizzle_a_mode);

    const int load_block_n = config.storage_config.load_block_n;
    const auto [inner_dim_b, outer_dim_b] = get_inner_outer_dims(major_b, k, n);
    const auto [inner_block_b, outer_block_b] = get_inner_outer_dims(major_b, config.layout.block_k, load_block_n);
    const auto tensor_map_b = make_tma_3d_desc(b, inner_dim_b, outer_dim_b, batch_size,
                                               inner_block_b, outer_block_b, 1,
                                               b.stride(major_b == cute::UMMA::Major::K ? 1 : 2),
                                               b.stride(0),
                                               config.storage_config.swizzle_b_mode);

    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, batch_size,
                                                store_block_n, store_block_m, 1,
                                                d.stride(1), d.stride(0),
                                                config.storage_config.swizzle_cd_mode);

    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 sf_block_mn_a, gran_k_a, batch_size, 0, 0, false, sf_block_k);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 sf_block_mn_b, gran_k_b, batch_size, 0, 0, false, sf_block_k);

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_fp8_gemm_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue = make_epilogue_input(m, n, std::nullopt, std::nullopt, sfd),
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        // NOTES: `k_alignment` is only used by k-grouped psum, dummy here
        .k_alignment = gran_k_a,
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

} // namespace deep_gemm
""",
    "csrc/dgt_api.cpp": r"""#define DG_USE_TORCH_LIBRARY 1
#define DJ_DISABLE_GIL 1

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include <torch/library.h>

#include "utils/exception.hpp"
#include "utils/layout.hpp"
#include "runtime/launch.hpp"
#include "jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"

// king-v7 DGT: a bundle-owned host path for one DeepGEMM call, wo_a's
//   deep_gemm.fp8_einsum("bhr,hdr->bhd", (a, sfa), (b, sfb), (d, sfd), recipe=(1, 1, 32))
// with an FP8 (d, sfd) output on SM100, dispatched as sgl-deep-gemm 0.2.3.post2 (b7f3816c) does: permute to
// (batch, m, n, k) = (h, b, d, r), check, take the packed-UE8M0 (INT, 1, 32) SFs as they are (the image's
// transform_sf_into_required_layout only checks them), then sm100_fp8_bmm (its own text, unchanged) with
// compiled_dims "nk". The kernel it compiles is the image's sm100_fp8_fp4_gemm_1d1d.cuh plus one
// griddepcontrol.launch_dependents (this library's include root), so only the dependent's launch moves.

using namespace deep_gemm;
using torch::Tensor;

namespace {

void dgt_init(const std::string& library_root_path, const std::string& image_include_path) {
    init_jit(library_root_path, image_include_path);
}

int64_t dgt_get_num_sms() { return runtime->get_num_sms(); }
void dgt_set_num_sms(int64_t n) { runtime->set_num_sms(static_cast<int>(n)); }
int64_t dgt_get_tc_util() { return runtime->get_tc_util(); }
void dgt_set_tc_util(int64_t v) { runtime->set_tc_util(static_cast<int>(v)); }
bool dgt_get_pdl() { return *jit->default_launch_options.enable_pdl; }
void dgt_set_pdl(bool v) { jit->default_launch_options.enable_pdl = v; }
bool dgt_get_deterministic() { return heuristics_runtime->get_deterministic_algorithms(); }
void dgt_set_deterministic(bool v) { heuristics_runtime->use_deterministic_algorithms(v); }

GemmDesc wo_a_desc(const int& batch_size, const int& m, const int& n, const int& k) {
    return GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = torch::kFloat8_e4m3fn, .b_dtype = torch::kFloat8_e4m3fn,
        .cd_dtype = torch::kFloat8_e4m3fn,
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = "nk",
        .max_gran_k = 32
    };
}

// The config sm100_fp8_bmm will pick for this call (no compile, no launch): for the boot line and the served-set key.
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> dgt_config(
        int64_t batch_size, int64_t m, int64_t n, int64_t k) {
    const auto config = get_best_config<SM100ArchSpec>(
        wo_a_desc(static_cast<int>(batch_size), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)));
    return {config.layout.swap_ab, config.layout.block_m, config.layout.block_n, config.layout.cluster_m,
            config.layout.cluster_n, config.pipeline_config.num_stages, config.pipeline_config.smem_size};
}

void dgt_wo_a(const Tensor& a, const Tensor& sfa, const Tensor& b, const Tensor& sfb, const Tensor& d,
              const Tensor& sfd) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);

    // Permute dims to (batch_size, m, n, k) = (h, b, d, r), as the image's fp8_einsum does for "bhr,hdr->bhd"
    const auto perm_a = a.permute({1, 0, 2});
    const auto perm_sfa = sfa.permute({1, 0, 2});
    const auto perm_d = d.permute({1, 0, 2});

    // fp8_bmm: `[B, M, K] @ [B, N, K].T`, K-major operands only here
    DG_HOST_ASSERT(perm_a.stride(-1) == 1 and b.stride(-1) == 1 and perm_d.stride(-1) == 1);
    const auto [batch_size  , m , k ] = get_shape<3>(perm_a);
    const auto [batch_size_ , n , k_] = get_shape<3>(b);
    const auto [batch_size__, m_, n_] = get_shape<3>(perm_d);
    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(perm_a.scalar_type() == torch::kFloat8_e4m3fn and b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(perm_d.scalar_type() == torch::kFloat8_e4m3fn);

    // FP8 D: per-32 packed UE8M0 SFD over the `(m, batch_size * n)` view of D
    DG_HOST_ASSERT(perm_d.stride(0) == n and n % 128 == 0);
    check_sf_layout(sfd, m, batch_size * n, 1, 32, std::nullopt, true, false, torch::kInt);

    // (INT, 1, 32) SFA / SFB on SM100: checked, used as they are
    check_sf_layout(perm_sfa, m, k, 1, 32, batch_size, false, false, torch::kInt);
    check_sf_layout(sfb, n, k, 1, 32, batch_size, false, false, torch::kInt);
    check_sf_layout(perm_sfa, m, k, 1, 32, batch_size, true, false, torch::kInt);
    check_sf_layout(sfb, n, k, 1, 32, batch_size, true, false, torch::kInt);

    sm100_fp8_bmm(perm_a, perm_sfa, b, sfb, std::nullopt, perm_d, batch_size, m, n, k, 32, 32,
                  cute::UMMA::Major::K, cute::UMMA::Major::K, "nk", sfd);
}

}

TORCH_LIBRARY(dec_dgt, m) {
    m.def("init(str library_root_path, str image_include_path) -> ()", &dgt_init);
    m.def("get_num_sms() -> int", &dgt_get_num_sms);
    m.def("set_num_sms(int new_num_sms) -> ()", &dgt_set_num_sms);
    m.def("get_tc_util() -> int", &dgt_get_tc_util);
    m.def("set_tc_util(int new_tc_util) -> ()", &dgt_set_tc_util);
    m.def("get_pdl() -> bool", &dgt_get_pdl);
    m.def("set_pdl(bool enable) -> ()", &dgt_set_pdl);
    m.def("get_deterministic() -> bool", &dgt_get_deterministic);
    m.def("set_deterministic(bool enable) -> ()", &dgt_set_deterministic);
    m.def("config(int batch_size, int m, int n, int k) -> (int, int, int, int, int, int, int)", &dgt_config);
    m.def("wo_a(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(a!) d, Tensor(b!) sfd) -> ()");
}

TORCH_LIBRARY_IMPL(dec_dgt, CUDA, m) {
    m.impl("wo_a", &dgt_wo_a);
}
""",
}
