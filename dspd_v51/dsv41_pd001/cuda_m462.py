"""dsv41-pd065 M462 host and kernel sources: generator/pd060/kernel/dgms (sgl-deep-gemm 0.2.3.post2 host
sources and MegaMoE kernel headers with DeepGEMM #462's items as knobs; MIT, Copyright (c) 2025 DeepSeek;
kernel/ATTRIBUTION.txt) verbatim but for the op namespace and DeepJIT signature (written under SGLang's
JIT cache root by m462.source_root). Regenerate with generator/pd060/gen_pd060.py, do not edit."""

ENTRY = "csrc/dgms_api.cpp"
TAG = "f672110d41ccd404"
CXX_FLAGS = ('-std=c++20', '-O3', '-fvisibility=hidden', '-fvisibility-inlines-hidden', '-Wno-psabi', '-Wno-deprecated-declarations')
LD_FLAGS = ('-lnvrtc', '-lcublasLt', '-lcublas')

SOURCES = {
    "build_include/elfutils/libdwfl.h": r"""/* pd005 build stub: the sealed image ships DeepJIT's headers but not elfutils-dev. DeepJIT's exception.hpp only takes
   these declarations' types (decltype) and resolves the functions from libdw.so.1 with dlsym at run time, so the
   prototypes below (copied from elfutils 0.190 libdwfl.h) are all it needs. */
#pragma once
#include <elf.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t Dwarf_Addr;
typedef uint64_t Dwarf_Word;
typedef Elf64_Addr GElf_Addr;
typedef Elf64_Word GElf_Word;
typedef Elf64_Shdr GElf_Shdr;
typedef struct Elf Elf;
typedef struct Dwfl Dwfl;
typedef struct Dwfl_Module Dwfl_Module;
typedef struct Dwfl_Line Dwfl_Line;

typedef struct {
    int (*find_elf)(Dwfl_Module* mod, void** userdata, const char* modname, Dwarf_Addr base, char** file_name,
                    Elf** elfp);
    int (*find_debuginfo)(Dwfl_Module* mod, void** userdata, const char* modname, Dwarf_Addr base,
                          const char* file_name, const char* debuglink_file, GElf_Word debuglink_crc,
                          char** debuginfo_file_name);
    int (*section_address)(Dwfl_Module* mod, void** userdata, const char* modname, Dwarf_Addr base,
                           const char* secname, GElf_Word shndx, const GElf_Shdr* shdr, Dwarf_Addr* addr);
    char** debuginfo_path;
} Dwfl_Callbacks;

Dwfl* dwfl_begin(const Dwfl_Callbacks* callbacks);
int dwfl_linux_proc_report(Dwfl* dwfl, pid_t pid);
int dwfl_report_end(Dwfl* dwfl, int (*removed)(Dwfl_Module*, void*, const char*, Dwarf_Addr, void* arg), void* arg);
Dwfl_Module* dwfl_addrmodule(Dwfl* dwfl, Dwarf_Addr address);
const char* dwfl_module_addrname(Dwfl_Module* mod, GElf_Addr address);
Dwfl_Line* dwfl_module_getsrc(Dwfl_Module* mod, Dwarf_Addr addr);
const char* dwfl_lineinfo(Dwfl_Line* line, Dwarf_Addr* addr, int* linep, int* colp, Dwarf_Word* mtime,
                          Dwarf_Word* length);
int dwfl_linux_proc_find_elf(Dwfl_Module* mod, void** userdata, const char* module_name, Dwarf_Addr base,
                             char** file_name, Elf** elfp);

#ifdef __cplusplus
}
#endif
""",
    "csrc/apis/mega_moe.hpp": r"""#pragma once

#include <functional>
#include <string>
#include <torch/torch.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_mega_moe_pre_dispatch.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type,
    const std::optional<int64_t>& global_num_tokens = std::nullopt) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens,
                                      mma_kind, global_num_tokens);
    return block_m;
}

// pd060: the ring capacity and SF ring sizes of the symmetric buffer (moved here verbatim from
// get_symm_buffer_size_for_mega_moe, which calls it; the compile-only op computes the kernel's template values with it)
static std::pair<int, int> pd060_get_ring_tokens_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const MmaKind& mma_kind) {
    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = runtime->get_num_sms();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

    // Iterate all block candidates to get the maximum ring size
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
    if (num_max_tokens_per_rank >= 6144) {
        // Preserve the tuned non-wrapping prefill path from dev. The dynamic
        // scheduler still uses a smaller live-ring bound for decode-sized
        // buffers, while large prefill allocations cover the full token pool.
        num_ring_tokens = layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    }

    const auto with_sf = is_mma_with_sf(mma_kind);

    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    if (with_sf) {
        for (auto block_m: layout::kCandidateBlockM) {
            num_sf_ring_tokens = std::max(
                num_sf_ring_tokens,
                layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
        }
    }
    return {num_ring_tokens, num_sf_ring_tokens};
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_shared_experts >= 0);

    // SiTU is implemented only by the SM100 FP8xFP4 MegaMoE kernel.
    const auto mma_kind = parse_mma_kind(mma_type);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "swigluoai" or
                   (mma_kind == MmaKind::MXFP8FP4 and activation == "situ"));
    DG_HOST_ASSERT(num_shared_experts >= 0);

    // Shared
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // pd060: ring sizes from the helper the compile-only op shares
    int num_ring_tokens = 0, num_sf_ring_tokens = 0;
    std::tie(num_ring_tokens, num_sf_ring_tokens) = pd060_get_ring_tokens_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, mma_kind);

    // Parse MMA type
    const auto with_sf = is_mma_with_sf(mma_kind);

    const bool host_use_fp8_combine = with_sf and deep_jit::get_env<int>("DG_USE_FP8_COMBINE") != 0;

    // All buffers
    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, mma_kind,
        num_shared_experts, host_use_fp8_combine
    );

    // Activation layout: FP8 for MXFP8FP4, raw bytes holding 2 packed FP4 values for NVFP4/MXFP4
    const auto acts_dtype = with_sf ?
        (get_element_bits(mma_kind) == 4 ? torch::kUInt8 : torch::kFloat8_e4m3fn) : torch::kBFloat16;
    const int acts_storage_bits = static_cast<int>(c10::elementSize(acts_dtype)) * 8;
    const auto num_acts_cols = [=](const int& num_elems) {
        return num_elems * get_element_bits(mma_kind) / acts_storage_bits;
    };
    const auto num_sf_cols = [=](const int& num_elems) {
        return num_elems / (get_sf_gran_k(mma_kind) * 4);
    };

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(shared_intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates tensor views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, num_acts_cols(hidden)},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device()));
        auto x_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, num_sf_cols(hidden)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), num_sf_cols(hidden)},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, num_acts_cols(shared_intermediate_hidden)},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), num_sf_cols(shared_intermediate_hidden)},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, num_acts_cols(hidden)},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device()));
        auto l1_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, num_sf_cols(hidden)},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, num_acts_cols(intermediate_hidden)},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, num_sf_cols(intermediate_hidden)},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto x_scales = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_x_scales_buffer.base)),
            {num_max_tokens_per_rank},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, x_scales);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& mma_type,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const bool& use_x_scales,
    const std::optional<torch::Tensor>& l1_alphas,
    const std::optional<torch::Tensor>& l2_alphas,
    const std::optional<torch::Tensor>& l2_act_scales,
    const bool& post_expert_norm = false,
    const std::optional<torch::Tensor>& post_expert_norm_weight = std::nullopt,
    const float& post_expert_norm_eps = 1e-6f,
    const bool& post_expert_norm_weight_first = false,
    const bool& use_trtllm_weights = false,
    const std::optional<int64_t>& global_num_tokens = std::nullopt,
    const std::optional<torch::Tensor>& sm_locality_domains = std::nullopt
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    // pd005: one uint8 domain index per physical SM (DeepGEMM #462), read by each cluster leader's scheduler
    if (sm_locality_domains.has_value()) {
        DG_HOST_ASSERT(sm_locality_domains->scalar_type() == torch::kUInt8 and sm_locality_domains->is_contiguous());
        DG_HOST_ASSERT(sm_locality_domains->is_cuda() and sm_locality_domains->numel() == jit->device.get_num_sms());
    }
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and (rk == 32 or rk == 16));
    const auto mma_kind = parse_mma_kind(mma_type);
    if (mma_kind != MmaKind::MXFP8FP4 and mma_kind != MmaKind::MXFP4 and mma_kind != MmaKind::NVFP4)
        DG_HOST_UNREACHABLE("`" + mma_type + "` activations are not implemented by the SM100 "
                            "mega-MoE kernel (only `fp8xfp4`, `mxf4xmxf4` and `nvfp4xnvfp4` are); "
                            "allocate the symmetric buffer with one of those");
    if (rk != get_sf_gran_k(mma_kind))
        DG_HOST_UNREACHABLE("recipe K granularity " + std::to_string(rk) + " does not match `" +
                            mma_type + "` (expected " +
                            std::to_string(get_sf_gran_k(mma_kind)) + ")");
    DG_HOST_ASSERT(not use_x_scales or mma_kind == MmaKind::NVFP4);
    DG_HOST_ASSERT(not use_trtllm_weights or mma_kind == MmaKind::NVFP4);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "swigluoai" or
                   (mma_kind == MmaKind::MXFP8FP4 and activation == "situ"));
    DG_HOST_ASSERT(activation != "situ" or not activation_clamp_opt.has_value());
    const bool use_situ = activation == "situ";
    const float swiglu_alpha = activation == "swigluoai" ? 1.702f : 0.0f;
    // K3 SiTU config (activation_situ_{beta,linear_beta}). Passed to the kernel as template
    // parameters: it only takes its equal-transform shortcut for this pair, and computes the
    // literal SiTU for any other.
    const float situ_beta = 4.0f;
    const float situ_linear_beta = 25.0f;
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    const auto weight_dtype = l1_weights.scalar_type();
    DG_HOST_ASSERT(weight_dtype == torch::kFloat8_e4m3fn or weight_dtype == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == weight_dtype);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for byte-packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1;
    const int kGranK = rk;
    // TRT weights require matching dense TRT scales; descriptors encode their layout.
    if (use_trtllm_weights)
        DG_HOST_ASSERT(l1_weights_sf.is_contiguous() and l2_weights_sf.is_contiguous());
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, not use_trtllm_weights, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, not use_trtllm_weights, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) = shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) = shared_l2_weights_tuple_opt.value();
        const bool is_packed_fp4 = mma_kind == MmaKind::MXFP4 or mma_kind == MmaKind::NVFP4;
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1)) *
                                     (is_packed_fp4 ? 2 : 1);
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == (is_packed_fp4 ? hidden / 2 : hidden));
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        const auto shared_weight_dtype = is_packed_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn;
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == shared_weight_dtype);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == shared_weight_dtype);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        mma_type, activation, num_shared_experts
    );
    if (sym_buffer.nbytes() < static_cast<size_t>(num_required_bytes))
        DG_HOST_UNREACHABLE("symmetric buffer is " + std::to_string(sym_buffer.nbytes()) +
                            " bytes but `" + mma_type + "`/`" + activation + "` with " +
                            std::to_string(num_shared_experts) + " shared expert(s) needs " +
                            std::to_string(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    if (l1_alphas.has_value()) {
        DG_HOST_ASSERT(mma_kind == MmaKind::NVFP4);
        DG_HOST_ASSERT(l1_alphas->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(l1_alphas->is_contiguous());
        DG_HOST_ASSERT(l1_alphas->dim() == 2);
        DG_HOST_ASSERT(l1_alphas->size(0) == num_experts_per_rank);
        DG_HOST_ASSERT(l1_alphas->size(1) == 2);
    }

    if (l2_alphas.has_value()) {
        DG_HOST_ASSERT(mma_kind == MmaKind::NVFP4);
        DG_HOST_ASSERT(l2_alphas->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(l2_alphas->is_contiguous());
        DG_HOST_ASSERT(l2_alphas->dim() == 1);
        DG_HOST_ASSERT(l2_alphas->size(0) == num_experts_per_rank);
    }

    // Per-expert fc2 input global scales, `[num_experts_per_rank]` FP32
    if (l2_act_scales.has_value()) {
        DG_HOST_ASSERT(mma_kind == MmaKind::NVFP4);
        DG_HOST_ASSERT(l2_act_scales->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(l2_act_scales->is_contiguous());
        DG_HOST_ASSERT(l2_act_scales->dim() == 1);
        DG_HOST_ASSERT(l2_act_scales->size(0) == num_experts_per_rank);
    }

    // `[hidden]` FP32 gain, optional; `weight_first` picks `rms_norm(w * x)`.
    DG_HOST_ASSERT(post_expert_norm or not post_expert_norm_weight.has_value());
    DG_HOST_ASSERT(not post_expert_norm or num_shared_experts == 0);
    DG_HOST_ASSERT(not post_expert_norm or post_expert_norm_eps >= 0.0f);
    if (post_expert_norm_weight.has_value()) {
        DG_HOST_ASSERT(post_expert_norm_weight->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(post_expert_norm_weight->is_contiguous());
        DG_HOST_ASSERT(post_expert_norm_weight->dim() == 1);
        DG_HOST_ASSERT(post_expert_norm_weight->size(0) == hidden);
    }

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, x_scales] = slice(sym_buffer);
    const bool use_fp8_combine = deep_jit::get_env<int>("DG_USE_FP8_COMBINE") != 0;

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               shared_l1_acts, shared_l1_acts_sf,
                               shared_l2_acts, shared_l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               shared_l1_weights, shared_l2_weights,
                               shared_l1_weights_sf, shared_l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_shared_experts,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, swiglu_alpha, use_situ, fast_math,
                               use_x_scales,
                               l1_alphas.has_value()
                                   ? l1_alphas->const_data_ptr<float>() : nullptr,
                               l2_alphas.has_value()
                                   ? l2_alphas->const_data_ptr<float>() : nullptr,
                               l2_act_scales.has_value()
                                   ? l2_act_scales->const_data_ptr<float>() : nullptr,
                               post_expert_norm_weight.has_value()
                                   ? post_expert_norm_weight->const_data_ptr<float>() : nullptr,
                               post_expert_norm,
                               post_expert_norm_weight_first,
                               post_expert_norm_eps,
                               mma_kind,
                               use_fp8_combine, use_trtllm_weights, global_num_tokens,
                               situ_beta, situ_linear_beta,
                               sm_locality_domains.has_value() ? sm_locality_domains->data_ptr<uint8_t>() : nullptr);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const bool& post_expert_norm = false,
    const std::optional<torch::Tensor>& post_expert_norm_weight = std::nullopt,
    const float& post_expert_norm_eps = 1e-6f,
    const bool& post_expert_norm_weight_first = false,
    const bool& use_trtllm_weights = false,
    const std::optional<int64_t>& global_num_tokens = std::nullopt
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    DG_HOST_ASSERT(post_expert_norm or not post_expert_norm_weight.has_value());
    DG_HOST_ASSERT(not post_expert_norm or num_shared_experts == 0);
    DG_HOST_ASSERT(not post_expert_norm or post_expert_norm_eps >= 0.0f);
    if (post_expert_norm_weight.has_value()) {
        DG_HOST_ASSERT(post_expert_norm_weight->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(post_expert_norm_weight->is_contiguous());
        DG_HOST_ASSERT(post_expert_norm_weight->dim() == 1);
        DG_HOST_ASSERT(post_expert_norm_weight->size(0) == hidden);
        DG_HOST_ASSERT(post_expert_norm_weight->device() == y.device());
    }

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights,
                shared_l1_acts, _shared_l1_acts_sf, shared_l2_acts, _shared_l2_acts_sf,
                l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf, _x_scales] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts,
                            shared_l1_acts, shared_l2_acts,
                            l1_weights, l2_weights,
                            shared_l1_weights, shared_l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math,
                            post_expert_norm_weight.has_value()
                                ? post_expert_norm_weight->const_data_ptr<float>() : nullptr,
                            post_expert_norm, post_expert_norm_weight_first,
                            post_expert_norm_eps, use_trtllm_weights, global_num_tokens);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

#ifndef DG_USE_TORCH_LIBRARY
static void register_apis(pybind11::module_& m) {
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_block_m_for_mega_moe", &get_block_m_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("bf16_mega_moe", &bf16_mega_moe);
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch,
          pybind11::arg("x"),
          pybind11::arg("topk_idx"),
          pybind11::arg("topk_weights"),
          pybind11::arg("buf_x"),
          pybind11::arg("buf_x_sf"),
          pybind11::arg("buf_topk_idx"),
          pybind11::arg("buf_topk_weights"),
          pybind11::arg("num_tokens"),
          pybind11::arg("group_size") = 32,
          pybind11::arg("mma_type") = "fp8xfp4",
          pybind11::arg("buf_x_scales") = std::nullopt,
          pybind11::arg("expert_scales") = std::nullopt);
}

#endif

} // namespace deep_gemm::mega
""",
    "csrc/dgms_api.cpp": r"""// pd005 MEGASCHED: the sealed image's fp8xfp4 MegaMoE host path (sgl-deep-gemm 0.2.3.post2, csrc copied verbatim) with
// DeepGEMM #462's kernel items as per-call knobs, as a separate TORCH_LIBRARY namespace `dgms` so that it loads beside
// the image's `deep_gemm` `_C.so` in one process (hidden visibility: its DeepJIT / runtime singletons stay private).
#define DG_USE_TORCH_LIBRARY 1
#define DJ_DISABLE_GIL 1

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <torch/library.h>

// The image TU reaches these first through other APIs
#include "utils/exception.hpp"
#include "runtime/launch.hpp"
#include "apis/mega_moe.hpp"

using namespace deep_gemm;
using torch::Tensor;
using OptTensor = std::optional<Tensor>;

namespace {

void dgms_init(const std::string& library_root_path, const std::string& image_include_path) {
    init_jit(library_root_path, image_include_path);
}

int64_t dgms_get_num_sms() { return runtime->get_num_sms(); }
void dgms_set_num_sms(int64_t n) { runtime->set_num_sms(static_cast<int>(n)); }
int64_t dgms_get_device_num_sms() { return jit->device.get_num_sms(); }
bool dgms_get_pdl() { return *jit->default_launch_options.enable_pdl; }
void dgms_set_pdl(bool v) { jit->default_launch_options.enable_pdl = v; }

int64_t dgms_get_block_m_for_mega_moe(int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
                                      int64_t num_tokens, int64_t num_topk, const std::string& mma_type,
                                      std::optional<int64_t> global_num_tokens) {
    return mega::get_block_m_for_mega_moe(static_cast<int>(num_ranks), static_cast<int>(num_experts),
                                          static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_tokens),
                                          static_cast<int>(num_topk), mma_type, global_num_tokens);
}

// knobs = [store_split, store_l1, store_l2, prefetch, locality, epi462, regs462, shared_first, fence, num_sms]
void set_knobs(at::IntArrayRef knobs) {
    DG_HOST_ASSERT(knobs.size() == 10);
    pd005_knobs = Pd005Knobs{
        .store_split = static_cast<int>(knobs[0]),
        .store_l1 = static_cast<int>(knobs[1]), .store_l2 = static_cast<int>(knobs[2]),
        .prefetch = static_cast<int>(knobs[3]),
        .locality = static_cast<int>(knobs[4]),
        .epi462 = static_cast<int>(knobs[5]), .regs462 = static_cast<int>(knobs[6]),
        .shared_first = static_cast<int>(knobs[7]), .fence = static_cast<int>(knobs[8]),
        .num_sms = static_cast<int>(knobs[9]),
    };
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> dgms_get_config(
        at::IntArrayRef knobs, int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
        int64_t num_tokens, int64_t num_topk, int64_t hidden, int64_t intermediate_hidden,
        int64_t num_ring_tokens, int64_t num_sf_ring_tokens, std::optional<int64_t> global_num_tokens) {
    set_knobs(knobs);
    const auto c = get_mega_moe_config(static_cast<int>(num_ranks), static_cast<int>(num_experts),
                                       static_cast<int>(num_experts / num_ranks),
                                       static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_tokens),
                                       static_cast<int>(num_topk), static_cast<int>(hidden),
                                       static_cast<int>(intermediate_hidden), static_cast<int>(num_ring_tokens),
                                       static_cast<int>(num_sf_ring_tokens), MmaKind::MXFP8FP4, global_num_tokens);
    pd005_knobs = Pd005Knobs{};
    return {c.block_m, c.store_block_m, c.store_block_m_l2, c.num_epi_prefetch_stages, c.num_stages, c.smem_size,
            c.block_k};
}

// pd060: build (JIT) the fp8xfp4 MegaMoE kernel the launch path would build for these facts, without tensors or a
// launch, so the bundle's prepare leaves no compile for a served forward. The values mirror the launch path for
// activation "swiglu", recipe (1, 1, 32), no alphas / x scales / post-expert norm, megamoe weight layout; the ring
// sizes come from the buffer layout helper. Returns (block_m, store L1, store L2, prefetch stages, stages, smem).
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> pd060_compile_only_op(
        at::IntArrayRef knobs, int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
        int64_t num_topk, int64_t hidden, int64_t intermediate_hidden, int64_t num_shared_experts,
        const Tensor& weight_dtype_ref, std::optional<double> activation_clamp, bool fast_math,
        int64_t num_tokens, std::optional<int64_t> global_num_tokens) {
    struct Reset {
        ~Reset() { pd060_compile_only = false; pd005_knobs = Pd005Knobs{}; }
    } reset;
    set_knobs(knobs);
    pd060_compile_only = true;
    const auto mma_kind = MmaKind::MXFP8FP4;
    const int ranks = static_cast<int>(num_ranks), experts = static_cast<int>(num_experts);
    const int cap = static_cast<int>(num_max_tokens_per_rank), topk = static_cast<int>(num_topk);
    const int h = static_cast<int>(hidden), inter = static_cast<int>(intermediate_hidden);
    DG_HOST_ASSERT(experts % ranks == 0 and static_cast<int>(num_tokens) <= cap);
    int num_ring_tokens = 0, num_sf_ring_tokens = 0;
    std::tie(num_ring_tokens, num_sf_ring_tokens) =
        mega::pd060_get_ring_tokens_for_mega_moe(ranks, experts, cap, topk, h, inter, mma_kind);
    const auto config = get_mega_moe_config(ranks, experts, experts / ranks, cap, static_cast<int>(num_tokens), topk,
                                            h, inter, num_ring_tokens, num_sf_ring_tokens, mma_kind,
                                            global_num_tokens);
    const auto num_sms = pd005_knobs.num_sms > 0 ? pd005_knobs.num_sms : runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0 and num_sms <= jit->device.get_num_sms());
    const float clamp = activation_clamp.has_value() ? static_cast<float>(*activation_clamp)
                                                     : std::numeric_limits<float>::infinity();
    DG_HOST_ASSERT(clamp >= 0);
    const bool use_fp8_combine = deep_jit::get_env<int>("DG_USE_FP8_COMBINE") != 0;
    pd060_compile_fp8_fp4_mega_moe(cap, h, inter, experts, static_cast<int>(num_shared_experts), topk, config,
                                   mma_kind, num_sms, ranks, clamp, 0.0f, false, fast_math, false,
                                   false, false, false, use_fp8_combine, weight_dtype_ref.scalar_type(),
                                   4.0f, 25.0f, false, false, false, 1e-6f, false);
    return {config.block_m, config.store_block_m, config.store_block_m_l2, config.num_epi_prefetch_stages,
            config.num_stages, config.smem_size};
}

void dgms_fp8_fp4_mega_moe(const Tensor& y, const Tensor& l1_weights, const Tensor& l1_weights_sf,
                           const Tensor& l2_weights, const Tensor& l2_weights_sf,
                           const OptTensor& shared_l1_weights, const OptTensor& shared_l1_weights_sf,
                           const OptTensor& shared_l2_weights, const OptTensor& shared_l2_weights_sf,
                           const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
                           at::IntArrayRef sym_buffer_ptrs, int64_t rank_idx,
                           int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                           at::IntArrayRef recipe, const std::string& mma_type, const std::string& activation,
                           std::optional<double> activation_clamp, bool fast_math,
                           std::optional<int64_t> global_num_tokens, at::IntArrayRef knobs,
                           const OptTensor& sm_locality_domains) {
    DG_HOST_ASSERT(shared_l1_weights.has_value() == shared_l1_weights_sf.has_value());
    DG_HOST_ASSERT(shared_l2_weights.has_value() == shared_l2_weights_sf.has_value());
    DG_HOST_ASSERT(recipe.size() == 3);
    const auto shared_l1 = shared_l1_weights.has_value()
        ? std::make_optional(std::make_tuple(shared_l1_weights.value(), shared_l1_weights_sf.value())) : std::nullopt;
    const auto shared_l2 = shared_l2_weights.has_value()
        ? std::make_optional(std::make_tuple(shared_l2_weights.value(), shared_l2_weights_sf.value())) : std::nullopt;
    set_knobs(knobs);
    mega::fp8_fp4_mega_moe(
        y, std::make_tuple(l1_weights, l1_weights_sf), std::make_tuple(l2_weights, l2_weights_sf),
        shared_l1, shared_l2, cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs.vec(),
        static_cast<int>(rank_idx), static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        std::make_tuple(static_cast<int>(recipe[0]), static_cast<int>(recipe[1]), static_cast<int>(recipe[2])),
        mma_type, activation,
        activation_clamp.has_value() ? std::make_optional(static_cast<float>(*activation_clamp)) : std::nullopt,
        fast_math, false, std::nullopt, std::nullopt, std::nullopt,
        false, std::nullopt, 1e-6f, false, false, global_num_tokens, sm_locality_domains);
    pd005_knobs = Pd005Knobs{};
}

}  // namespace

TORCH_LIBRARY(pd065_m462, m) {
    m.def("init(str library_root_path, str image_include_path) -> ()", &dgms_init);
    m.def("get_num_sms() -> int", &dgms_get_num_sms);
    m.def("set_num_sms(int new_num_sms) -> ()", &dgms_set_num_sms);
    m.def("get_device_num_sms() -> int", &dgms_get_device_num_sms);
    m.def("get_pdl() -> bool", &dgms_get_pdl);
    m.def("set_pdl(bool enable) -> ()", &dgms_set_pdl);
    m.def("get_block_m_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_tokens, "
          "int num_topk, str mma_type, int? global_num_tokens) -> int", &dgms_get_block_m_for_mega_moe);
    m.def("get_config(int[] knobs, int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_tokens, "
          "int num_topk, int hidden, int intermediate_hidden, int num_ring_tokens, int num_sf_ring_tokens, "
          "int? global_num_tokens) -> (int, int, int, int, int, int, int)", &dgms_get_config);
    m.def("get_token_alignment_for_mega_moe() -> int", []() -> int64_t { return mega::get_token_alignment_for_mega_moe(); });
    m.def("compile_only(int[] knobs, int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, "
          "int hidden, int intermediate_hidden, int num_shared_experts, Tensor weight_dtype_ref, float? activation_clamp, "
          "bool fast_math, int num_tokens, int? global_num_tokens) -> (int, int, int, int, int, int)",
          &pd060_compile_only_op);
    m.def("fp8_fp4_mega_moe(Tensor(a!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, "
          "Tensor l2_weights_sf, Tensor? shared_l1_weights, Tensor? shared_l1_weights_sf, Tensor? shared_l2_weights, "
          "Tensor? shared_l2_weights_sf, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, "
          "int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, "
          "int[] recipe, str mma_type, str activation, float? activation_clamp, bool fast_math, "
          "int? global_num_tokens, int[] knobs, Tensor? sm_locality_domains) -> ()");
}

TORCH_LIBRARY_IMPL(pd065_m462, CUDA, m) {
    m.impl("fp8_fp4_mega_moe", &dgms_fp8_fp4_mega_moe);
}
""",
    "csrc/jit_kernels/heuristics/common.hpp": r"""#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "config.hpp"
#include "runtime.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // Choose the best layout
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
        const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
        if (ArchSpec::compare(candidate_info, layout_info))
            layout = layout_candidates[i], layout_info = candidate_info;
    }

    // Infer other configs
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        std::stringstream ss;
        ss << desc;
        const auto key = ss.str();

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << desc << ": " << gemm_config << ", " << layout_info << std::endl;
            printed.insert(key);
        }
    }
    return gemm_config;
}

} // namespace deep_gemm
""",
    "csrc/jit_kernels/heuristics/config.hpp": r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
#include <c10/core/ScalarType.h>
#include <deep_gemm/common/types.cuh>

#include "../../utils/math.hpp"

namespace deep_gemm {

/// GEMM descriptors
struct GemmDesc {
    GemmType gemm_type;
    KernelType kernel_type;
    int m, n, k, num_groups;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    cute::UMMA::Major major_a;
    cute::UMMA::Major major_b;
    bool with_accumulation;

    // Requirements from users
    int num_sms, tc_util;
    std::string compiled_dims;

    // SM100 m-grouped psum layout padding contract
    bool ensure_zero_padding = true;

    // SF granularity for MMA selection and split-K alignment: max(gran_k_a, gran_k_b).
    int max_gran_k = 128;

    // False for AB-swap (transposed, stride_cd_n != 1) output: the TMA-store epilogue
    // cannot express it, so the kernel falls back to the strided-store epilogue.
    bool cd_n_contiguous = true;

    // Shape for heuristic generation
    int expected_m = 0, expected_n = 0, expected_k = 0, expected_num_groups = 0;
    int get_expected_m() const { return expected_m > 0 ? expected_m : m; }
    int get_expected_n() const { return expected_n > 0 ? expected_n : n; }
    int get_expected_k() const { return expected_k > 0 ? expected_k : k; }
    int get_expected_num_groups() const { return expected_num_groups > 0 ? expected_num_groups : num_groups; }

    bool is_mxf4_mma() const {
        // Native MXF4 requires per-32 scales and K-major packed operands.
        // Coarser/mixed scales retain the unpacked MXF8F6F4 path.
        return a_dtype == kPackedFP4 and b_dtype == kPackedFP4 and max_gran_k == 32 and
               major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K;
    }

    MmaKind get_mma_kind() const {
        if (a_dtype == torch::kBFloat16)
            return MmaKind::BF16;
        return is_mxf4_mma() ? MmaKind::MXF4 : MmaKind::MXFP8FP4;
    }

    int get_smem_pack_factor() const {
        return is_mxf4_mma() ? 2 : 1;
    }

    void check_validity() const {
        if (get_mma_kind() == MmaKind::BF16) {
            DG_HOST_ASSERT(a_dtype == torch::kBFloat16 and b_dtype == torch::kBFloat16);
        } else {
            DG_HOST_ASSERT(a_dtype == torch::kFloat8_e4m3fn or a_dtype == kPackedFP4);
            DG_HOST_ASSERT(b_dtype == torch::kFloat8_e4m3fn or b_dtype == kPackedFP4);
        }
        // FP8 D implies casting with dynamic per-32 UE8M0 SFD output, only exposed for batched GEMMs
        DG_HOST_ASSERT(cd_dtype == torch::kBFloat16 or cd_dtype == torch::kFloat or
                       (cd_dtype == torch::kFloat8_e4m3fn and gemm_type == GemmType::Batched and
                        not with_accumulation));
        DG_HOST_ASSERT(num_sms % 2 == 0);
    }

    friend std::ostream& operator << (std::ostream& os, const GemmDesc& desc) {
        MmaKind mma_kind = desc.get_mma_kind();
        os << "GemmDesc(gemm_type=" << static_cast<int>(desc.gemm_type)
           << ", kernel_type=" << static_cast<int>(desc.kernel_type)
           << ", m=" << desc.m << ", n=" << desc.n << ", k=" << desc.k
           << ", num_groups=" << desc.num_groups
           << ", major_a=" << static_cast<int>(desc.major_a)
           << ", major_b=" << static_cast<int>(desc.major_b)
           << ", mma_kind=" << static_cast<int>(mma_kind)
           << ", a_dtype=" << c10::toString(desc.a_dtype)
           << ", b_dtype=" << c10::toString(desc.b_dtype)
           << ", cd_dtype=" << c10::toString(desc.cd_dtype)
           << ", with_accumulation=" << static_cast<int>(desc.with_accumulation)
           << ", num_sms=" << desc.num_sms
           << ", tc_util=" << desc.tc_util
           << ", compiled_dims=" << desc.compiled_dims
           << ", ensure_zero_padding=" << static_cast<int>(desc.ensure_zero_padding)
           << ", expected_m=" << desc.expected_m
           << ", expected_n=" << desc.expected_n
           << ", expected_k=" << desc.expected_k
           << ", expected_num_groups=" << desc.expected_num_groups << ")";
        return os;
    }
};

/// GEMM configs
struct Layout {
    int swap_ab;
    int block_m, block_n, block_k;
    int cluster_m, cluster_n;

    int get_cluster_size() const {
        return cluster_m * cluster_n;
    }

    friend std::ostream& operator << (std::ostream& os, const Layout& layout) {
        os << "Layout(swap_ab=" << layout.swap_ab
           << ", block_m=" << layout.block_m << ", block_n=" << layout.block_n << ", block_k=" << layout.block_k
           << ", cluster_m=" << layout.cluster_m << ", cluster_n=" << layout.cluster_n << ")";
        return os;
    }
};

struct StorageConfig {
    int load_block_m, load_block_n;
    int store_block_m, store_block_n;

    int swizzle_a_mode, swizzle_b_mode;
    int swizzle_cd_mode;

    friend std::ostream& operator << (std::ostream& os, const StorageConfig& config) {
        os << "StorageConfig("
           << "load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m << ", store_block_n=" << config.store_block_n
           << ", swizzle_a_mode=" << config.swizzle_a_mode << ", swizzle_b_mode=" << config.swizzle_b_mode
           << ", swizzle_cd_mode=" << config.swizzle_cd_mode << ")";
        return os;
    }
};

struct PipelineConfig {
    int smem_size;
    int num_stages;
    int num_tma_store_stages;

    friend std::ostream& operator << (std::ostream& os, const PipelineConfig& config) {
        os << "PipelineConfig("
           << "smem_size=" << config.smem_size
           << ", num_stages=" << config.num_stages
           << ", num_tma_store_stages=" << config.num_tma_store_stages << ")";
        return os;
    }
};

struct LaunchConfig {
    int num_sms;
    int num_sms_per_cluster;
    int num_threads;

    int num_tma_threads;
    int num_math_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const LaunchConfig& config) {
        os << "LaunchConfig("
           << "num_sms=" << config.num_sms << ", num_sms_per_cluster=" << config.num_sms_per_cluster
           << ", num_threads=" << config.num_threads
           << ", num_tma_threads=" << config.num_tma_threads << ", num_math_threads=" << config.num_math_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

struct GemmConfig {
    Layout layout;
    StorageConfig storage_config;
    PipelineConfig pipeline_config;
    LaunchConfig launch_config;
    int split_k_factor = 1;

    friend std::ostream& operator << (std::ostream& os, const GemmConfig& config) {
        os << "GemmConfig("
           << "layout=" << config.layout
           << ", storage_config=" << config.storage_config
           << ", pipeline_config=" << config.pipeline_config
           << ", launch_config=" << config.launch_config
           << ", split_k=" << config.split_k_factor << ")";
        return os;
    }
};

/// Config comparators
struct LayoutInfo {
    int num_waves;
    int last_wave_util;
    int64_t num_cycles;
    Layout layout;

    friend std::ostream& operator << (std::ostream& os, const LayoutInfo& config) {
        os << "LayoutInfo("
           << "num_waves=" << config.num_waves
           << ", last_wave_util=" << config.last_wave_util 
           << ", num_cycles=" << config.num_cycles << ")";
        return os;
    }
};

}  // namespace deep_gemm
""",
    "csrc/jit_kernels/heuristics/mega_moe.hpp": r"""#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"

namespace deep_gemm {

// pd005 MEGASCHED knobs, set by the `dgms` op for each call (defaults reproduce the image)
struct Pd005Knobs {
    int store_split = 0;            // 1: DeepGEMM #462 store blocks (BLOCK_M 240 fp8xfp4 -> L1 24 / L2 8, no tiles env)
    int store_l1 = 0, store_l2 = 0; // > 0: explicit store blocks (applied last)
    int prefetch = 0;               // epilogue TMEM prefetch stages; -1: #462 rule (BLOCK_M 240 -> 1)
    int locality = 0;               // 1: #462 per-domain task queues
    int epi462 = 0, regs462 = 0, shared_first = 0, fence = 0;
    int num_sms = 0;                // > 0: the kernel's SM count
};
inline Pd005Knobs pd005_knobs;

// pd060: a compile-only call (the bundle's prepare) builds the kernel and skips the launch; the code it compiled is
// recorded, so a served call that would still JIT (an unprepared config) says so on stderr instead of staying silent
inline bool pd060_compile_only = false;
inline std::unordered_set<std::string> pd060_prepared_codes;

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;      // L1 store block
    int store_block_m_l2;   // pd005: L2 store block
    int num_epi_prefetch_stages;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;
    // SF granularity along K
    int gran_k;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", store_block_m_l2=" << config.store_block_m_l2
           << ", num_epi_prefetch_stages=" << config.num_epi_prefetch_stages
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", gran_k=" << config.gran_k
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    if (mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8")
        return MmaKind::MXFP8FP4;
    if (mma_type_str == "mxf4xmxf4")
        return MmaKind::MXFP4;
    DG_HOST_ASSERT(mma_type_str == "nvfp4xnvfp4");
    return MmaKind::NVFP4;
}

static std::string to_mma_type_string(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return "bf16xbf16";
        case MmaKind::MXFP8FP4: return "fp8xfp4";
        case MmaKind::MXFP4:    return "mxf4xmxf4";
        case MmaKind::NVFP4:    return "nvfp4xnvfp4";
    }
    DG_HOST_UNREACHABLE("Unknown MMA kind");
}

static std::string to_mma_kind_name(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return "MmaKind::BF16";
        case MmaKind::MXFP8FP4: return "MmaKind::MXFP8FP4";
        case MmaKind::MXFP4:    return "MmaKind::MXFP4";
        case MmaKind::NVFP4:    return "MmaKind::NVFP4";
    }
    DG_HOST_UNREACHABLE("Unknown MMA kind");
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4 or mma_kind == MmaKind::MXFP4 or mma_kind == MmaKind::NVFP4;
}

static bool is_mxf4_mma_kind(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP4;
}

static bool is_nvfp4_mma_kind(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::NVFP4;
}

static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind,
    const std::optional<int64_t>& global_num_tokens) {
    // Experts receive tokens from every rank, so an idle or lightly loaded rank still computes remote load.
    // Without a caller hint, assume all ranks match the local count.
    DG_HOST_ASSERT(not global_num_tokens.has_value() or
                   (*global_num_tokens >= 0 and
                    *global_num_tokens <= static_cast<int64_t>(num_max_tokens_per_rank) * num_ranks));
    const float num_total_tokens = global_num_tokens.has_value() ?
        static_cast<float>(*global_num_tokens) : static_cast<float>(num_tokens) * num_ranks;

    // Expected tokens per expert, plus a one-sigma routing margin for the tile choice
    const float num_expected_tokens = num_total_tokens * num_topk / num_experts;
    const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);

    // Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
    // per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
    // A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
    int block_m = num_expected_tokens <= 10 ? 16 : 32;
    if (num_expected_tokens > 24) {
        int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
        if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
            num_blocks = 2;
        for (const int& candidate: {64, 128, 192, 240}) {
            block_m = candidate;
            if (num_blocks * candidate >= num_covered_tokens)
                break;
        }
    }
    // Must divide `WG_BLOCK_M` (= block_m / 2), which pins each arm: 240 -> 120 and
    // 128 -> 64 cannot take 48, 192 -> 96 can and halves its store iterations.
    int store_block_m = block_m <= 16 ? 8 : block_m <= 64 ? 16 :
                        block_m <= 128 ? 32 : block_m <= 192 ? 48 : 40;
    int block_k = 128 * 8 / get_element_bits(mma_kind);
    int cluster_size = 2, num_epilogue_warpgroups = 2;

    // Offline tuning hook: "cluster,block_m,store_block_m,block_k,epilogue_warpgroups",
    // `block_k` in the pre-scaling 8-bit units above.
    if (const auto override_str = deep_jit::get_env<std::string>("DG_MEGA_MOE_TILES"); not override_str.empty()) {
        int v[5];
        const int n = std::sscanf(override_str.c_str(), "%d,%d,%d,%d,%d",
                                  v + 0, v + 1, v + 2, v + 3, v + 4);
        DG_HOST_ASSERT(n == 5 and "DG_MEGA_MOE_TILES needs 5 comma-separated ints");
        cluster_size = v[0], block_m = v[1], store_block_m = v[2];
        block_k = v[3] * 8 / get_element_bits(mma_kind), num_epilogue_warpgroups = v[4];
    }

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    return {cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m, const int& store_block_m_l2,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int elem_bits = get_element_bits(mma_kind);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 output staging and L2 BF16 staging.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages * elem_bits / 8;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Schedule task payloads
    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    // + schedule task publish full/empty barriers.
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

    // Amax warp-pair reduction buffer for SwiGLU's cross-warp amax exchange.
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    const int smem_sf_inv_lut = mma_kind == MmaKind::NVFP4 ?
        256 * static_cast<int>(sizeof(float)) : 0;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;

    // Per-stage: A tile + B tile + optional SF tiles + full/empty barriers.
    // NOTES: for `MXFP8FP4` the FP4 weights are unpacked into 8-bit containers in smem,
    // so A and B always share the same per-element footprint
    const int smem_a_size_per_stage = load_block_m * block_k * elem_bits / 8;
    const int smem_b_size_per_stage = block_n * block_k * elem_bits / 8;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_tmem_ptr + smem_sf_inv_lut;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind,
    const std::optional<int64_t>& global_num_tokens) {

    // Block config
    const auto [cluster_size, block_m, store_block_m_default, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens,
                                      mma_kind, global_num_tokens);
    DG_HOST_ASSERT(num_ring_tokens % block_m == 0);

    // pd005: #462 asymmetric store blocks at FP8xFP4 prefill (saves shared memory for one more TMA stage)
    int store_block_m = store_block_m_default, store_block_m_l2 = store_block_m_default;
    const bool tiles_env = not deep_jit::get_env<std::string>("DG_MEGA_MOE_TILES").empty();
    if (pd005_knobs.store_split and not tiles_env and block_m == 240 and mma_kind == MmaKind::MXFP8FP4)
        store_block_m = 24, store_block_m_l2 = 8;
    if (pd005_knobs.store_l1 > 0)
        store_block_m = pd005_knobs.store_l1;
    if (pd005_knobs.store_l2 > 0)
        store_block_m_l2 = pd005_knobs.store_l2;
    const int num_epi_prefetch_stages = pd005_knobs.prefetch < 0 ? (block_m == 240 ? 1 : 0) : pd005_knobs.prefetch;
    const int block_n = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, mma_kind) : std::pair(0, 0);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle;
    // NVFP4 keeps FP4 packed, so a 128B swizzle atom covers twice as many elements
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = is_mma_with_sf(mma_kind) ? get_sf_gran_k(mma_kind) : 32;

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= num_max_pull_bytes
    const int num_max_pull_bytes = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    int num_bytes_per_pull = hidden * get_element_bits(mma_kind) / 8;
    while (num_bytes_per_pull > num_max_pull_bytes) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m, store_block_m_l2,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m, store_block_m_l2, num_epi_prefetch_stages,
        sf_block_m, sf_block_n, gran_k,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, global_num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens,
            global_num_tokens.value_or(-1), num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
""",
    "csrc/jit_kernels/heuristics/runtime.hpp": r"""#pragma once

#include <deep_jit/utils/lazy.hpp>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

class HeuristicsRuntime {
public:
    static constexpr int kLegacyMKAlignmentForContiguousLayout = 128;

    bool ignore_compile_dims = false;
    bool deterministic_algorithms = false;
    int block_m_multiple_of = 1;
    int block_n_multiple_of = 1;
    int mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;

    void use_deterministic_algorithms(const bool enabled) {
        deterministic_algorithms = enabled;
    }

    bool get_deterministic_algorithms() const {
        return deterministic_algorithms;
    }

    void set_ignore_compile_dims(const bool& new_value) {
        ignore_compile_dims = new_value;
    }

    bool get_ignore_compile_dims() const {
        return ignore_compile_dims;
    }

    void set_block_size_multiple_of(const int& new_block_m_multiple_of, const int& new_block_n_multiple_of) {
        block_m_multiple_of = new_block_m_multiple_of;
        block_n_multiple_of = new_block_n_multiple_of;
    }

    int get_block_m_multiple_of() const {
        return block_m_multiple_of;
    }

    int get_block_n_multiple_of() const {
        return block_n_multiple_of;
    }

    void set_mk_alignment_for_contiguous_layout(const int& new_value) {
        mk_alignment_for_contiguous_layout = new_value;
    }

    int get_mk_alignment_for_contiguous_layout() const {
        return mk_alignment_for_contiguous_layout;
    }

    // Per-arch BLOCK_M search: start at `max_block_m`, step down by `step` (never below
    // `min_block_m`) until the tile no longer over-covers M.
    struct ContiguousMKAlignment { int max_block_m, min_block_m, step; };

    static ContiguousMKAlignment get_contiguous_mk_alignment(const int& arch_major) {
        // SM120: BLOCK_M in {32, 64, 128}; 32 uses the kNWarps=4 warp layout (decode)
        if (arch_major == 12)
            return {128, 32, 32};
        // SM100: 224 down to 32 by 32 (sgl tuning)
        if (arch_major == 10)
            return {224, 32, 32};
        // SM90 and others: fixed legacy alignment, no shrinking
        return {kLegacyMKAlignmentForContiguousLayout, kLegacyMKAlignmentForContiguousLayout, 1};
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m,
                                                                    const std::optional<int>& num_groups = std::nullopt) {
        const auto spec = get_contiguous_mk_alignment(jit->device.get_arch_major());
        int block_m = spec.max_block_m;
        if (expected_m.has_value()) {
            // Grouped layouts must cover the per-group M, not the summed M
            int per_group_m = expected_m.value();
            if (num_groups.has_value() and num_groups.value() > 0)
                per_group_m = (per_group_m + num_groups.value() - 1) / num_groups.value();
            for (; block_m > spec.min_block_m and block_m - spec.step >= per_group_m; block_m -= spec.step);
        }
        // SM120 supports no 96-row tile (per-expert m > 64 measures best at 128)
        if (jit->device.get_arch_major() == 12 and block_m == 96)
            block_m = 128;
        return block_m;
    }
};

inline auto heuristics_runtime = deep_jit::LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
""",
    "csrc/jit_kernels/heuristics/sm100.hpp": r"""#pragma once

#include "../../runtime/runtime.hpp"

#include <algorithm>
#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "runtime.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
        const int& block_m, const int& block_n, const MmaKind& mma_kind) {
        constexpr int num_utccp_aligned_elems = 128;
        switch (mma_kind) {
            case MmaKind::BF16: return {0, 0};
            case MmaKind::MXFP8FP4: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            case MmaKind::MXFP4: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            case MmaKind::NVFP4: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            default: DG_HOST_UNREACHABLE("Unknown dtype");
        }
    }

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // Block K is always in a fixed manner
        const int block_k = 128 * 8 / get_num_element_bits(desc.get_mma_kind());

        // Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs
        if (desc.gemm_type == GemmType::MGroupedContiguous or
            desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout or
            desc.gemm_type == GemmType::MGroupedMasked) {
            const bool swap_ab = true;
            const auto block_n = 128;
            const auto block_m = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
            const auto cluster_m = 1;
            const auto cluster_n = ceil_div(desc.n, block_n) % 2 == 0 and desc.num_sms % 2 == 0 ? 2 : 1;
            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};
            std::vector<Layout> candidates = {layout};
            return candidates;
        }

        // Large K-major batched projections benefit from the deeper non-overlapped
        // pipeline. Keep overlapping tiles for smaller/shallow projections and MN-major B.
        const bool prefer_batched_nonoverlap =
            desc.gemm_type == GemmType::Batched and
            desc.a_dtype == torch::kFloat8_e4m3fn and desc.b_dtype == torch::kFloat8_e4m3fn and
            desc.cd_dtype == torch::kBFloat16 and not desc.with_accumulation and
            desc.major_a == cute::UMMA::Major::K and desc.major_b == cute::UMMA::Major::K and
            desc.m >= 8192 and desc.n <= 1024 and desc.k >= 2048 and desc.max_gran_k == 128;
        // This mixed-format FP32 accumulation family also favors the deeper pipeline.
        // BF16 output, other layouts, and deeper K retain the overlapping-tile gains.
        const bool prefer_mixed_nonoverlap =
            desc.gemm_type == GemmType::Normal and
            desc.a_dtype == torch::kFloat8_e4m3fn and desc.b_dtype == kPackedFP4 and
            desc.cd_dtype == torch::kFloat and desc.with_accumulation and desc.cd_n_contiguous and
            desc.major_a == cute::UMMA::Major::MN and desc.major_b == cute::UMMA::Major::MN and
            desc.m >= 8192 and desc.n == 4096 and desc.k == 1536;
        const bool prefer_nonoverlap = prefer_batched_nonoverlap or prefer_mixed_nonoverlap;

        // Enumerate all candidates
        std::vector<Layout> candidates;
        std::vector<Layout> preferred_candidates;
        for (int swap_ab = 0; swap_ab < 2; ++ swap_ab) {
            // Block M/N candidates
            std::vector<int> block_m_candidates;
            std::vector<int> block_n_candidates;
            if (swap_ab) {
                int step = std::lcm(16, heuristics_runtime->get_block_m_multiple_of());
                int end = 256;
                for (int i = step; i <= end; i += step)
                    block_m_candidates.push_back(i);

                // TODO: consider other block N
                block_n_candidates = {128};
            } else {
                // NOTES: smaller block M can avoid TMA L2 OOB bound
                // TODO: consider block M = 256
                if (desc.m <= 32) block_m_candidates = {32};
                else if (desc.m <= 64) block_m_candidates = {64};
                else block_m_candidates = {128};

                // Small block size for small shape
                if (16 % heuristics_runtime->get_block_n_multiple_of() == 0)
                    block_n_candidates.push_back(16);
                int step = std::lcm(32, heuristics_runtime->get_block_n_multiple_of());
                // For small K, fewer store blocks improve store/compute overlap and reduce epilogue bottleneck
                int end = desc.k <= 256 ? 128 : 256;
                for (int i = step; i <= end; i += step)
                    block_n_candidates.push_back(i);
            }

            for (int cluster_m = 1; cluster_m <= 2; ++ cluster_m) {
                // After swapping, layout A/D can only do on cluster N
                if (swap_ab == 1 and cluster_m > 1)
                    continue;

                for (int cluster_n = 1; cluster_n <= 2; ++ cluster_n) {
                    // We only support cluster 2
                    if (cluster_m * cluster_n > 2)
                        continue;

                    // Only support layout A/D
                    if (swap_ab == 0 and cluster_n > 1)
                        continue;

                    // SM count must be divisible
                    if (desc.num_sms % (cluster_m * cluster_n) != 0)
                        continue;

                    for (int block_m: block_m_candidates) {
                        // Ensure large swizzle sizes (32B swizzle yields poor performance)
                        const auto swizzle_a_requirement = desc.a_dtype == kPackedFP4 ? 128 : 64;
                        // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                        const auto load_block_m_requirement = desc.major_a == cute::UMMA::Major::MN ? swizzle_a_requirement : 8;
                        if ((block_m / cluster_n) % load_block_m_requirement != 0)
                            continue;

                        // Shape must be divisible for multicast
                        if (ceil_div(desc.m, block_m) % cluster_m != 0)
                            continue;

                        for (int block_n: block_n_candidates) {
                            // Ensure large swizzle sizes (32B swizzle yields poor performance)
                            const auto swizzle_b_requirement = desc.b_dtype == kPackedFP4 ? 128 : 64;
                            // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                            const auto load_block_n_requirement = desc.major_b == cute::UMMA::Major::MN ? swizzle_b_requirement : 8;
                            if ((block_n / cluster_m) % load_block_n_requirement != 0)
                                continue;

                            // Shape must be divisible for multicast
                            if (ceil_div(desc.n, block_n) % cluster_n != 0)
                                continue;

                            // SwapAB requires block N is layout A/D' UMMA M
                            constexpr int layout_ad_m = 128;
                            if (swap_ab and block_n != layout_ad_m)
                                continue;

                            // Check tensor memory capacity
                            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, desc.get_mma_kind());
                            const auto sf_granularity = get_sf_gran_k(desc.get_mma_kind());
                            const auto tmem_sf_cols = sf_granularity ?
                                (sf_block_m + sf_block_n) * (block_k / (sf_granularity * 4)) / 32 : sf_granularity;
                            const auto umma_n = swap_ab ? block_m : block_n;
                            if (umma_n + tmem_sf_cols > 512)
                                continue;

                            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};

                            // When neither A nor B is MN major, 128B swizzle is always feasible
                            if (desc.major_a == cute::UMMA::Major::K or desc.major_b == cute::UMMA::Major::K) {
                                const auto storage_config = get_storage_config(desc, layout);
                                if (storage_config.swizzle_a_mode != 128 or storage_config.swizzle_b_mode != 128)
                                    continue;
                            }

                            candidates.push_back(layout);
                            if (prefer_nonoverlap and 2 * umma_n + tmem_sf_cols <= 512)
                                preferred_candidates.push_back(layout);
                        }
                    }
                }
            }
        }

        // Explicit block-size constraints may leave only overlapping layouts available.
        if (not preferred_candidates.empty())
            candidates = std::move(preferred_candidates);

        // Dynamic FP8 output requires the physical store atom to own complete per-32 SF groups
        if (desc.cd_dtype == torch::kFloat8_e4m3fn) {
            const auto owns_partial_sf_groups = [&](const Layout& layout) {
                const auto store_block_n = layout.swap_ab ? layout.block_n :
                    get_storage_config(desc, layout).swizzle_cd_mode;
                return store_block_n % 32 != 0;
            };
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(), owns_partial_sf_groups),
                             candidates.end());
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int layout_ad_m = 128;
        constexpr int umma_step_n = 16;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        const auto load_block_m = layout.block_m / layout.cluster_n;
        const auto load_block_n = layout.block_n / layout.cluster_m;
        const auto store_block_m = layout.swap_ab ? umma_step_n : std::min(layout_ad_m, layout.block_m);
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        const int pack_factor = desc.get_smem_pack_factor();
        const int swizzle_mode_a = get_swizzle_mode(
            (desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m) / pack_factor,
            c10::elementSize(desc.a_dtype));
        const int swizzle_mode_b = get_swizzle_mode(
            (desc.major_b == cute::UMMA::Major::K ? layout.block_k : load_block_n) / pack_factor,
            c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_cd = get_swizzle_mode(
            store_block_n, c10::elementSize(desc.cd_dtype));

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static int get_num_tma_store_stages(const GemmDesc& desc, const Layout& layout) {
        // With single-stage TMA stores, each SM keeps at most one bulk store in
        // flight. This paces the store traffic and improves the achieved DRAM
        // throughput when C/D flushes dominate DRAM traffic; it also frees up
        // shared memory for the A/B mainloop and halves the C/D smem read/write
        // footprint. Only enabled for k-grouped GEMMs.
        if (desc.get_mma_kind() == MmaKind::BF16 or layout.swap_ab or
            not is_k_grouped_contiguous(desc.gemm_type))
            return 2;

        const int num_k_blocks_per_group = desc.k / std::max(desc.num_groups, 1) / layout.block_k;
        const int min_k_blocks = desc.cd_dtype != torch::kFloat ? 16 :
                                 desc.with_accumulation ? 24 : 32;
        return num_k_blocks_per_group >= min_k_blocks ? 1 : 2;
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 32;

        // C/D for TMA stores
        const int num_tma_store_stages = get_num_tma_store_stages(desc, layout);
        const int smem_cd = (layout.swap_ab ? storage_config.store_block_m * storage_config.store_block_n * c10::elementSize(desc.cd_dtype)
                                            : storage_config.store_block_m * storage_config.swizzle_cd_mode) * num_tma_store_stages;

        // TODO: remove SF barriers for BF16 GEMMs
        // A/B-and-SF-transpose full, SF-TMA full, empty, and tensor memory full/empty/overlap barriers
        // NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
        // NOTES: the last barrier is for tensor core utilization control
        const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 3 + 8;

        // Tensor memory pointer
        const int smem_tmem_ptr = 4;

        // Calculate A/B per stages
        const int pack_factor = desc.get_smem_pack_factor();
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype) / pack_factor;
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype) / pack_factor;

        // Calculate SF A/B per stages
        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(
                layout.block_m, layout.block_n, desc.get_mma_kind());
            smem_sfa_per_stage = sf_block_m * layout.block_k / 32;
            smem_sfb_per_stage = sf_block_n * layout.block_k / 32;
        }

        // Calculate stages
        int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
        int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages,
            num_tma_store_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            256,
            32, 128, 128, 128
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;
        // TODO: calculate expected cycles
        return {num_waves, last_wave_util, 0, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        // Single wave is always better
        if ((a.num_waves == 1 or b.num_waves == 1) and a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Doing multicast is better
        if (a.layout.get_cluster_size() != b.layout.get_cluster_size())
            return a.layout.get_cluster_size() > b.layout.get_cluster_size();

        // Smaller number of waves is better
        if (a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Larger last wave utilization is better
        if (a.last_wave_util != b.last_wave_util)
            return a.last_wave_util > b.last_wave_util;

        // More stages is better
        // Same block M, smaller block N is better
        // Same block N, smaller block M is better
        if (a.layout.block_m + a.layout.block_n != b.layout.block_m + b.layout.block_n)
            return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;

        // Less shared memory C/D, more stages is better
        return a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n;
    }
};

} // namespace deep_gemm
""",
    "csrc/jit_kernels/heuristics/sm90.hpp": r"""#pragma once

#include "../../runtime/runtime.hpp"

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM90ArchSpec {
    static constexpr int smem_capacity = 232448;
    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        constexpr int short_k_stages = 8;
        // Block K is fixed by the MMA operand width.
        const int block_k = 128 * 8 / get_num_element_bits(desc.get_mma_kind());
        const bool block32 = desc.kernel_type == KernelType::Kernel1D2D and desc.max_gran_k == 32;
        const int n_tile_multiple = block32 ? desc.max_gran_k : 16;
        const int max_1d2d_block_n = block32 ? block_k : 192;
        const int short_k = short_k_stages * block_k;
        const int max_block_m = desc.k <= short_k ? 2 * block_k : block_k;
        const bool allow_small_m = not block32 or desc.k <= 2 * short_k;

        // Block M candidates
        std::vector<int> block_m_candidates;
        if (desc.gemm_type == GemmType::Normal or
            desc.gemm_type == GemmType::Batched or
            desc.gemm_type == GemmType::KGroupedContiguous) {
            // TODO: check 256's performance
            block_m_candidates = {64, 128};
            // NOTES: smaller block M can avoid TMA L2 OOB bound
            if (desc.m <= 16 and allow_small_m) block_m_candidates.push_back(16);
            if (desc.m <= 32 and allow_small_m) block_m_candidates.push_back(32);

            // BF16 output GEMM supports 256
            if (desc.cd_dtype != torch::kFloat)
                block_m_candidates.push_back(256);
        } else if (desc.gemm_type == GemmType::MGroupedContiguous or
                   desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
            block_m_candidates = std::vector{heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
        } else if (desc.gemm_type == GemmType::MGroupedMasked) {
            block_m_candidates = {64, 128};
        }

        // Block N candidates
        std::vector<int> block_n_candidates;
        int step = std::lcm(n_tile_multiple, heuristics_runtime->get_block_n_multiple_of());
        int start = step;
        // Avoid bank conflicts for 1D1D kernel FP32 output
        if (desc.kernel_type == KernelType::Kernel1D1D and desc.cd_dtype == torch::kFloat) {
            DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K);
            DG_HOST_ASSERT(desc.major_b == cute::UMMA::Major::K);
            start = 24;
            block_n_candidates.push_back(16);
        }
        // Register spills
        int end = 256;
        if (desc.kernel_type == KernelType::Kernel1D2D)
            end = max_1d2d_block_n;
        if (desc.kernel_type == KernelType::Kernel1D1D)
            end = 160;
        // Enumerate
        for (int i = start; i <= end; i += step)
            block_n_candidates.push_back(i);

        // Disable multicast for performance
        const bool disable_multicast =
            // The number of k-groups is large (a heuristic)
            (desc.gemm_type == GemmType::KGroupedContiguous and desc.num_groups > 4) or
            // Not supported
            (desc.gemm_type == GemmType::Batched) or block32;

        // Enumerate all candidates
        std::vector<Layout> candidates;
        for (int cluster_m = 1; cluster_m <= (disable_multicast ? 1 : 2); ++ cluster_m) {
            for (int cluster_n = 1; cluster_n <= (disable_multicast ? 1 : 2); ++ cluster_n) {
                // We only support cluster 2
                if (cluster_m * cluster_n > 2)
                    continue;

                // SM count must be divisible
                if (desc.num_sms % (cluster_m * cluster_n) != 0)
                    continue;

                for (int block_m: block_m_candidates) {
                    for (int block_n: block_n_candidates) {
                        // Long K favors overlapping wide single-wave tiles; short K can
                        // amortize B loads across two M waves without spilling.
                        if (block32 and block_m > max_block_m)
                            continue;
                        // 1D2D kernel unroll requirement
                        if (desc.kernel_type == KernelType::Kernel1D2D and block_n > block_k and (block_n % (block_n - block_k) != 0 and block_k % (block_n - block_k) != 0))
                            continue;

                        // Multicast legality for masked layout
                        // TODO: add some comments about it
                        if ((desc.gemm_type == GemmType::MGroupedMasked or desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) and
                            ceil_div(desc.n, block_n) % (cluster_m * cluster_n) != 0)
                            continue;

                        // The block sizes cannot be too large (for enough registers), so at least one dim less than 128
                        if (block_m > 128 and block_n > 128)
                            continue;

                        // Calculate swizzling
                        const auto layout = Layout{0, block_m, block_n, block_k, cluster_m, cluster_n};
                        const auto storage_config = get_storage_config(desc, layout);

                        // Make sure swizzling is large enough (32B's performance is low)
                        if (storage_config.swizzle_a_mode % 64 != 0 or storage_config.swizzle_b_mode % 64 != 0)
                            continue;
                        
                        // To hide TMA latency, the stage count should be at least 3; for small matrices, at least 4
                        int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
                        if (num_stages < 3 or (block_m * block_n < 128 * 192 and num_stages < 4))
                            continue;

                        candidates.push_back(layout);
                    }
                }
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int wgmma_m = 64;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        // TODO: support swap AB
        DG_HOST_ASSERT(layout.swap_ab == 0);
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;
        // 1D1D kernel will do single warp-group stores
        const auto store_block_m = desc.kernel_type == KernelType::Kernel1D1D ? wgmma_m : layout.block_m;
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        const auto swizzle_mode_a = get_swizzle_mode(
            desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(
            desc.major_b == cute::UMMA::Major::K ? layout.block_k : load_block_n, c10::elementSize(desc.b_dtype));
        // We only enable swizzling for non-FP32 outputs
        const auto swizzle_mode_cd = desc.cd_dtype != torch::kFloat ?
            get_swizzle_mode(store_block_n, c10::elementSize(desc.cd_dtype)) : 0;

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;
        const bool block32 = desc.kernel_type == KernelType::Kernel1D2D and desc.max_gran_k == 32;
        const int scale_gran_k = block32 ? desc.max_gran_k : layout.block_k;
        const int scales_per_stage = layout.block_k / scale_gran_k;
        const int scale_b_rows = block32 ? layout.block_n / scale_gran_k :
            (layout.block_k % layout.block_n == 0 ? 1 : 2);

        // TODO: consider swap AB
        // C/D for TMA stores
        // NOTES: 1024 is for TMA swizzling alignment requirement
        const int smem_cd =
            align(layout.block_m * layout.block_n * static_cast<int>(c10::elementSize(desc.cd_dtype)), 1024);
        const int smem_barriers = kNumMaxStages * 8 * 2;

        // Calculate A/B per stages
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        // Calculate SF A/B per stages
        const int smem_sfa_per_stage = desc.kernel_type == KernelType::KernelNoSF ?
            0 : align(layout.block_m * scales_per_stage * static_cast<int>(sizeof(float)), 128);
        const int smem_sfb_per_stage = desc.kernel_type != KernelType::Kernel1D1D ?
            0 : align(layout.block_n * static_cast<int>(sizeof(float)), 128);

        // Extra SFB sizes for 1D2D kernels
        const int smem_extra_sfb = desc.kernel_type != KernelType::Kernel1D2D ?
            0 : align<int>(ceil_div(desc.k, scale_gran_k) * scale_b_rows * static_cast<int>(sizeof(float)), 8);

        // Extra tensormap for 1D1D kernels
        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 4 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        // Calculate stages
        const int smem_extra = smem_cd + smem_barriers + smem_extra_sfb + smem_tensormap;
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        const int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages,
            2
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_tma_threads = 128;
        const int num_math_threads = layout.block_m <= 64 ? 128 : 256;
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            num_tma_threads + num_math_threads,
            num_tma_threads, num_math_threads,
            0, 0 // Meaningless for SM90
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        // Utils
        const int l2_bandwidth_per_cycle = std::min(64. * desc.num_sms, 8e6 / (1.3e3)); // B/cycle
        const int l1_bandwidth_per_cycle = 128 * desc.num_sms; // B/cycle
        const int wgmma_m = 64;
        const int elem_size_ab = c10::elementSize(desc.a_dtype);
        const int elem_size_cd = c10::elementSize(desc.cd_dtype);
        DG_HOST_ASSERT(desc.a_dtype == desc.b_dtype);

        // Data movement per block
        int64_t expected_k = desc.get_expected_k();
        int64_t num_bytes_l2_ab = expected_k * (layout.block_m / layout.cluster_n + layout.block_n / layout.cluster_m) * elem_size_ab;
        int64_t num_bytes_l1_ab = expected_k * (layout.block_m + layout.block_n) * elem_size_ab;
        int64_t num_bytes_l1_tc = expected_k * (std::max(wgmma_m, layout.block_m) + layout.block_n) * elem_size_ab
                                  + layout.block_m * layout.block_n * elem_size_cd;
        int64_t num_bytes_l1_l2_cd = layout.block_m * layout.block_n * elem_size_cd * (desc.with_accumulation ? 2 : 1);

        // HBM bandwidth and total compute (Tensor/CUDA cores) are constant across configs
        // We only model L1/L2 cycles as they are the primary variables between configs
        int64_t num_l2_cycles = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks / l2_bandwidth_per_cycle;
        int64_t num_l1_cycles = (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) * num_blocks / l1_bandwidth_per_cycle;
        float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        int64_t num_cycles = std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency;

        // Disable multicasting if only one wave exists
        if (layout.cluster_n * layout.cluster_m > 1 and num_waves <= 1)
            num_cycles = std::numeric_limits<int64_t>::max();

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

} // namespace deep_gemm
""",
    "csrc/jit_kernels/heuristics/utils.hpp": r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

static int get_num_element_bits(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 16;
        case MmaKind::MXFP8FP4: return 8;
        case MmaKind::MXF4:     return 4;
        default: DG_HOST_UNREACHABLE("Unknown MMA kind");
    }
}

template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}

} // namespace deep_gemm
""",
    "csrc/jit_kernels/impls/runtime_utils.hpp": r"""#pragma once

#include "../../runtime/runtime.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <string>
#include <utility>

#include <cuda.h>
#include <torch/torch.h>

#include <deep_jit/backend/cuda/driver.hpp>
#include <deep_jit/utils/env.hpp>

#include "../heuristics/sm90.hpp"
#include "../../utils/math.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

static std::pair<int, int> get_inner_outer_dims(const cute::UMMA::Major& major, const int& k, const int& mn) {
    return major == cute::UMMA::Major::K ? std::make_pair(k, mn) : std::make_pair(mn, k);
}

static int get_non_contiguous_dim(const cute::UMMA::Major& major) {
    return major == cute::UMMA::Major::K ? -2 : -1;
}

static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
    if (heuristics_runtime->get_ignore_compile_dims())
        return 0;

    for (const char& c: compiled_dims) {
        if (name == c)
            return dim;
    }
    return 0;
}

static std::string to_string(const cute::UMMA::Major& major) {
    switch (major) {
        case cute::UMMA::Major::K:  return "cute::UMMA::Major::K";
        case cute::UMMA::Major::MN: return "cute::UMMA::Major::MN";
    }
    DG_HOST_UNREACHABLE("Unknown major");
}

static std::string to_string(const GemmType& type) {
    switch (type) {
        case GemmType::Normal:                              return "GemmType::Normal";
        case GemmType::MGroupedContiguous:                  return "GemmType::MGroupedContiguous";
        case GemmType::MGroupedMasked:                      return "GemmType::MGroupedMasked";
        case GemmType::MGroupedContiguousWithPsumLayout:    return "GemmType::MGroupedContiguousWithPsumLayout";
        case GemmType::KGroupedContiguous:                  return "GemmType::KGroupedContiguous";
        case GemmType::KGroupedContiguousWithPsumLayout:    return "GemmType::KGroupedContiguousWithPsumLayout";
        case GemmType::Batched:                             return "GemmType::Batched";
    }
    DG_HOST_UNREACHABLE("Unknown GEMM type");
}

static std::string to_string(const at::ScalarType& dtype, const bool& fp4_unpacked_smem = true) {
    switch (dtype) {
        case torch::kInt:           return "int";
        case torch::kFloat:         return "float";
        case torch::kBFloat16:      return "cutlass::bfloat16_t";
        case torch::kFloat8_e4m3fn: return "cutlass::float_e4m3_t";
        case kPackedFP4:            return fp4_unpacked_smem ? "cutlass::detail::float_e2m1_unpacksmem_t"
                                                             : "cutlass::float_e2m1_t";
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static std::string to_string(const float& v) {
    if (std::isfinite(v)) {
        return std::format("{}0x{:a}f", std::signbit(v) ? "-" : "", std::abs(v));
    } else if (std::isinf(v)) {
        return v > 0 ? "cute::numeric_limits<float>::infinity()"
                     : "-cute::numeric_limits<float>::infinity()";
    }
    DG_HOST_UNREACHABLE("NaN input is not supported");
}

static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype,
                                                          const bool& allow_tf32,
                                                          const bool& fp4_unpacked_smem) {
    if (allow_tf32 and dtype == torch::kFloat)
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;

    switch (dtype) {
        case torch::kInt:           return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16:      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case torch::kUInt8:         return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case kPackedFP4:            return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                                                             : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode, const int& base) {
    if (base != 0) {
        DG_HOST_ASSERT(base == 32 and mode == 128);
        return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    }

    DG_HOST_ASSERT(base == 0);
    switch (mode) {
        case   0:
        case  16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case  32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case  64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: DG_HOST_UNREACHABLE("Unsupported swizzling mode");
    }
}

static CUtensorMap make_tma_2d_desc(const torch::Tensor& t,
                                    int gmem_inner_dim, int gmem_outer_dim,
                                    int smem_inner_dim, int smem_outer_dim,
                                    const int& gmem_outer_stride,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    const auto num_gmem_outer_stride_bytes = static_cast<uint32_t>(gmem_outer_stride) * static_cast<uint32_t>(elem_size);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_outer_stride_bytes % 16u == 0u);
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_inner_dim % 128 == 0);

        // Fix FP4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_inner_dim = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, swizzle: %d (base: %d), elem size: %d, pointer: %llu\n",
               gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
               gmem_outer_stride, swizzle_mode, swizzle_base, elem_size,
               reinterpret_cast<unsigned long long>(t.data_ptr()));
    }
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap make_tma_3d_desc(const torch::Tensor& t,
                                    int gmem_dim_0, int gmem_dim_1, int gmem_dim_2,
                                    int smem_dim_0, int smem_dim_1, int smem_dim_2,
                                    const int& gmem_stride_0, const int& gmem_stride_1,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    const auto num_gmem_stride_0_bytes = static_cast<uint32_t>(gmem_stride_0) * static_cast<uint32_t>(elem_size);
    const auto num_gmem_stride_1_bytes = static_cast<uint32_t>(gmem_stride_1) * static_cast<uint32_t>(elem_size);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_stride_0_bytes % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_stride_1_bytes % 16u == 0u);
    if (swizzle_mode != 0)
        smem_dim_0 = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_dim_0 % 128 == 0);

        // Fix fp4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_dim_0 = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[3] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1), static_cast<cuuint64_t>(gmem_dim_2),};
    const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1), static_cast<cuuint32_t>(smem_dim_2)};
    const cuuint64_t gmem_strides[2] = {static_cast<cuuint64_t>(gmem_stride_0 * elem_size), static_cast<cuuint64_t>(gmem_stride_1 * elem_size)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Making 3D TMA desc: global memory: %d %d %d, shared memory: %d %d %d, outer stride: %d %d, swizzle: %d, elem size: %d\n",
               gmem_dim_0, gmem_dim_1, gmem_dim_2, smem_dim_0, smem_dim_1, smem_dim_2,
               gmem_stride_0, gmem_stride_1, swizzle_mode, elem_size);
    }
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        3, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap make_tma_a_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_m, const int& shape_k,
                                   const int& block_m, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false,
                                   const bool& fp4_unpacked_smem = true) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == cute::UMMA::Major::K);
    const auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m);
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32, fp4_unpacked_smem);
}

static CUtensorMap make_tma_b_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_n, const int& shape_k,
                                   const int& block_n, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false,
                                   const bool& fp4_unpacked_smem = true) {
    const auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_n);
    const auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_n);

    // `num_groups` is always applied into the outer dimensions
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim * num_groups,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32, fp4_unpacked_smem);
}

static CUtensorMap make_tma_cd_desc(const torch::Tensor& t,
                                    const int& shape_m, const int& shape_n,
                                    const int& block_m, const int& block_n,
                                    const int& outer_stride,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    // Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    // bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    return make_tma_2d_desc(t,
                            shape_n, shape_m * num_groups,
                            block_n, block_m,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_sf_desc(const cute::UMMA::Major& major,
                                    const torch::Tensor& t,
                                    int shape_mn, int shape_k,
                                    const int& block_mn, const int& gran_k,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const int& smem_outer_dim = 1) {
    DG_HOST_ASSERT(major == cute::UMMA::Major::MN);

    // TODO: maybe swizzle SF as well
    DG_HOST_ASSERT(swizzle_mode == 0);

    shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
    return make_tma_2d_desc(t,
                            shape_mn, ceil_div(shape_k, gran_k * (t.scalar_type() == torch::kFloat ? 1 : 4)) * num_groups,
                            block_mn, smem_outer_dim,
                            shape_mn,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

} // namespace deep_gemm
""",
    "csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp": r"""#pragma once

#include <format>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Canonical row `r` is stored at `s`: `s4=r0`, `s3=~r3`, `s2=r4`, `s[1:0]=r[2:1]`.
// These axes undo everything but `~r3`, which the SwiGLU epilogue absorbs.
static CUtensorMap make_trtllm_bf16_weight_tma_desc(
    const torch::Tensor& weights, const int rows, const bool is_l1) {
    DG_HOST_ASSERT(weights.is_contiguous() and rows % 128 == 0);
    CUtensorMap result;
    const cuuint64_t dims[5] = {64, is_l1 ? 2u : 4u, is_l1 ? 4u : 8u,
                                is_l1 ? 2u : static_cast<cuuint64_t>(rows / 32),
                                static_cast<cuuint64_t>(rows / 4)};
    const cuuint64_t strides[4] = {is_l1 ? 16u * 128u : 8u * 128u, 128,
                                   is_l1 ? 8u * 128u : 32u * 128u, 4u * 128u};
    const cuuint32_t box[5] = {64, is_l1 ? 2u : 4u, is_l1 ? 4u : 8u, is_l1 ? 2u : 4u, 2};
    const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, is_l1 ? 5 : 4, weights.data_ptr(), dims,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return result;
}

static void sm100_bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const float* post_expert_norm_weight,
    const bool& post_expert_norm,
    const bool& post_expert_norm_weight_first,
    const float& post_expert_norm_eps,
    const bool& use_trtllm_weights = false,
    const std::optional<int64_t>& global_num_tokens = std::nullopt
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16, global_num_tokens);

    if (use_trtllm_weights) {
        DG_HOST_ASSERT(num_shared_experts == 0);
        DG_HOST_ASSERT(hidden % 64 == 0 and intermediate_hidden % 64 == 0);
        DG_HOST_ASSERT(config.block_k % 64 == 0 and config.block_n == 128);
    }

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    // BlockMajorK storage: [expert, K/64, row, 64].
    const auto tensor_map_l1_weights = use_trtllm_weights ? make_trtllm_bf16_weight_tma_desc(
        l1_weights, num_experts_per_rank * (hidden / 64) * intermediate_hidden * 2, true) :
        make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_weights = use_trtllm_weights ? make_trtllm_bf16_weight_tma_desc(
        l2_weights, num_experts_per_rank * (intermediate_hidden / 64) * hidden, false) :
        make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = runtime->get_num_sms();

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}, {}, {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        post_expert_norm ? "true" : "false",
        post_expert_norm_weight_first ? "true" : "false",
        post_expert_norm_weight != nullptr ? "true" : "false",
        to_string(post_expert_norm_eps),
        use_trtllm_weights ? "true" : "false"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        post_expert_norm_weight,
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_weights,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_weights,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights);
}

} // namespace deep_gemm
""",
    "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp": r"""#pragma once

#include <format>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Undo TRT's L2 shuffle: GMEM row 32*i3 + 8*i1 + i2 -> SMEM row 32*i3 + 4*i2 + i1.
static CUtensorMap make_trtllm_fp4_l2_tma_desc(
    const torch::Tensor& weights, const int inner_bytes, const int rows) {
    DG_HOST_ASSERT(weights.is_contiguous() and rows % 128 == 0);
    DG_HOST_ASSERT(inner_bytes % 128 == 0);
    CUtensorMap result;
    const cuuint64_t dims[4] = {static_cast<cuuint64_t>(inner_bytes), 4, 8,
                                static_cast<cuuint64_t>(rows / 32)};
    const cuuint64_t strides[3] = {8ull * inner_bytes,
                                   static_cast<cuuint64_t>(inner_bytes), 32ull * inner_bytes};
    const cuuint32_t box[4] = {128, 4, 8, 4};
    const cuuint32_t element_strides[4] = {1, 1, 1, 1};
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, weights.data_ptr(), dims,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return result;
}

// TRT SFs: [row block][k chunk][128 words]. L2 also undoes the 32-row shuffle.
static CUtensorMap make_trtllm_fp4_sf_tma_desc(
    const torch::Tensor& sf, const bool& is_l1,
    const int& num_row_blocks, const int& num_k_chunks,
    const int& box_k_chunks, const int& box_row_blocks) {
    DG_HOST_ASSERT(num_row_blocks > 0 and num_k_chunks > 0);
    DG_HOST_ASSERT(box_k_chunks <= num_k_chunks and box_row_blocks <= num_row_blocks);
    CUtensorMap result;
    const auto row_block_stride = static_cast<cuuint64_t>(num_k_chunks) * 512ull;
    const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
    if (is_l1) {
        const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(num_k_chunks),
                                    static_cast<cuuint64_t>(num_row_blocks)};
        const cuuint64_t strides[2] = {512ull, row_block_stride};
        const cuuint32_t box[3] = {128, static_cast<cuuint32_t>(box_k_chunks),
                                   static_cast<cuuint32_t>(box_row_blocks)};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, sf.data_ptr(), dims,
            strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    } else {
        const cuuint64_t dims[5] = {4, 4, 8, static_cast<cuuint64_t>(num_k_chunks),
                                    static_cast<cuuint64_t>(num_row_blocks)};
        const cuuint64_t strides[4] = {128ull, 16ull, 512ull, row_block_stride};
        const cuuint32_t box[5] = {4, 4, 8, static_cast<cuuint32_t>(box_k_chunks),
                                   static_cast<cuuint32_t>(box_row_blocks)};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 5, sf.data_ptr(), dims,
            strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    return result;
}

// pd060: the kernel's code string and its JIT build, shared by the launch path below and the compile-only op
// (dgms_api.cpp), so both build the identical kernel. A compile-only call records the code; a served call whose code
// was not prepared reports it (it would JIT inside a served forward).
static auto pd060_compile_fp8_fp4_mega_moe(
    const int& num_max_tokens_per_rank, const int& hidden, const int& intermediate_hidden,
    const int& num_experts, const int& num_shared_experts, const int& num_topk,
    const MegaMoEConfig& config, const MmaKind& mma_kind, const int& num_sms, const int& num_ranks,
    const float& activation_clamp, const float& swiglu_alpha, const bool& use_situ, const bool& fast_math,
    const bool& use_x_scales, const bool& has_l1_alphas, const bool& has_l2_alphas, const bool& has_l2_act_scales,
    const bool& use_fp8_combine, const at::ScalarType& weight_dtype,
    const float& situ_beta, const float& situ_linear_beta,
    const bool& with_post_expert_norm, const bool& post_norm_weight_first, const bool& has_post_norm_weight,
    const float& post_norm_eps, const bool& use_trtllm_weights) {
    const auto code = std::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}, {}, {}, {},
        {}, {},
        {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", num_max_tokens_per_rank,
    hidden, intermediate_hidden,
    num_experts, num_shared_experts,
    num_topk,
    config.block_m, config.block_n, config.block_k,
    config.store_block_m,
    config.sf_block_m, config.sf_block_n,
    to_mma_kind_name(mma_kind),
    config.num_ring_tokens,
    config.num_sf_ring_tokens,
    config.num_stages,
    config.num_bytes_per_pull,
    config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
    num_sms, num_ranks,
    to_string(activation_clamp), to_string(swiglu_alpha),
    use_situ ? "true" : "false",
    fast_math ? "true" : "false",
    use_x_scales ? "true" : "false",
    has_l1_alphas ? "true" : "false",
    has_l2_alphas ? "true" : "false",
    has_l2_act_scales ? "true" : "false",
    use_fp8_combine ? "true" : "false",
    to_string(weight_dtype),
    to_string(situ_beta), to_string(situ_linear_beta),
    with_post_expert_norm ? "true" : "false",
    post_norm_weight_first ? "true" : "false",
    has_post_norm_weight ? "true" : "false",
    to_string(post_norm_eps),
    use_trtllm_weights ? "true" : "false",
    config.store_block_m_l2, config.num_epi_prefetch_stages,
    pd005_knobs.locality ? 2 : 1,
    pd005_knobs.epi462 ? "true" : "false",
    pd005_knobs.regs462 ? "true" : "false",
    pd005_knobs.shared_first ? "true" : "false",
    pd005_knobs.fence ? "true" : "false");
    if (pd060_compile_only)
        pd060_prepared_codes.insert(code);
    else if (not pd060_prepared_codes.empty() and not pd060_prepared_codes.contains(code))
        std::fprintf(stderr, "dsv41-pd065 M462: a served call compiles an unprepared MegaMoE kernel config\n");
    return jit->compile("sm100_fp8_fp4_mega_moe", code);
}

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const float& swiglu_alpha,
    const bool& use_situ,
    const bool& fast_math,
    const bool& use_x_scales,
    const float* l1_alphas,
    const float* l2_alphas,
    const float* l2_act_scales,
    const float* post_expert_norm_weight,
    const bool& with_post_expert_norm,
    const bool& post_norm_weight_first,
    const float& post_norm_eps,
    const MmaKind& mma_kind,
    const bool& use_fp8_combine,
    const bool& use_trtllm_weights = false,
    const std::optional<int64_t>& global_num_tokens = std::nullopt,
    const float& situ_beta = 4.0f,
    const float& situ_linear_beta = 25.0f,
    const uint8_t* sm_locality_domains = nullptr
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        mma_kind, global_num_tokens);

    if (use_trtllm_weights) {
        DG_HOST_ASSERT(mma_kind == MmaKind::NVFP4 and num_shared_experts == 0);
        DG_HOST_ASSERT(config.block_n % 128 == 0 and config.block_k % (config.gran_k * 4) == 0);
    }

    // Make tensormap
    const bool is_packed_fp4 = mma_kind == MmaKind::NVFP4 or mma_kind == MmaKind::MXFP4;
    const int kGranK = config.gran_k;
    const int elem_bits = get_element_bits(mma_kind);
    const auto to_inner = [=](const int& num_elems) { return num_elems * elem_bits / 8; };
    const auto weight_tensor = [=](const torch::Tensor& t) { return is_packed_fp4 ? t.view(torch::kUInt8) : t; };
    const auto weight_inner = [=](const int& num_elems) { return is_packed_fp4 ? num_elems / 2 : num_elems; };
    const int block_k_inner = to_inner(config.block_k);
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     to_inner(hidden), config.num_ring_tokens,
                                                     block_k_inner, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(weight_tensor(l1_weights),
                                                        weight_inner(hidden), num_experts_per_rank * intermediate_hidden * 2,
                                                        block_k_inner, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode, 0, false, not is_packed_fp4);
    const auto tensor_map_l1_weights_sf = use_trtllm_weights ? make_trtllm_fp4_sf_tma_desc(
        l1_weights_sf, true, num_experts_per_rank * intermediate_hidden * 2 / 128,
        hidden / (kGranK * 4), sf_smem_outer_dim, config.block_n / 128) :
        make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const int l1_out_block_n_bytes = to_inner(config.block_n / 2);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       to_inner(intermediate_hidden), config.num_ring_tokens,
                                                       l1_out_block_n_bytes, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       l1_out_block_n_bytes);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     to_inner(intermediate_hidden), config.num_ring_tokens,
                                                     block_k_inner, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = use_trtllm_weights ? make_trtllm_fp4_l2_tma_desc(
        l2_weights, weight_inner(intermediate_hidden), num_experts_per_rank * hidden) :
        make_tma_2d_desc(weight_tensor(l2_weights),
                                                        weight_inner(intermediate_hidden), num_experts_per_rank * hidden,
                                                        block_k_inner, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode, 0, false, not is_packed_fp4);
    const auto tensor_map_l2_weights_sf = use_trtllm_weights ? make_trtllm_fp4_sf_tma_desc(
        l2_weights_sf, false, num_experts_per_rank * hidden / 128,
        intermediate_hidden / (kGranK * 4), sf_smem_outer_dim, config.block_n / 128) :
        make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        to_inner(hidden), num_max_tokens_per_rank,
        block_k_inner, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        weight_tensor(shared_l1_weights),
        to_inner(hidden), shared_intermediate_hidden * 2,
        block_k_inner, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
        l1_out_block_n_bytes, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        l1_out_block_n_bytes) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
        block_k_inner, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        weight_tensor(shared_l2_weights),
        to_inner(shared_intermediate_hidden), hidden,
        block_k_inner, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = pd005_knobs.num_sms > 0 ? pd005_knobs.num_sms : runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0 and num_sms <= jit->device.get_num_sms());
    DG_HOST_ASSERT(not pd005_knobs.locality or sm_locality_domains != nullptr);

    // Compile (pd060: the shared helper, so the compile-only op builds the identical code)
    const auto kernel = pd060_compile_fp8_fp4_mega_moe(
        num_max_tokens_per_rank, hidden, intermediate_hidden, num_experts, num_shared_experts, num_topk, config,
        mma_kind, num_sms, num_ranks, activation_clamp, swiglu_alpha, use_situ, fast_math, use_x_scales,
        l1_alphas != nullptr, l2_alphas != nullptr, l2_act_scales != nullptr, use_fp8_combine,
        l1_weights.scalar_type(), situ_beta, situ_linear_beta, with_post_expert_norm, post_norm_weight_first,
        post_expert_norm_weight != nullptr, post_norm_eps, use_trtllm_weights);
    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        l1_alphas, l2_alphas, l2_act_scales,
        post_expert_norm_weight,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_acts_sf,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_weights_sf,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_acts_sf,
        tensor_map_shared_l2_weights,
        tensor_map_shared_l2_weights_sf,
        sm_locality_domains);
}

} // namespace deep_gemm
""",
    "csrc/jit_kernels/impls/sm100_mega_moe_pre_dispatch.hpp": r"""#pragma once

#include <torch/torch.h>

#include <optional>

#include "../heuristics/mega_moe.hpp"
#include "../../utils/exception.hpp"
#include <format>
#include "../../utils/math.hpp"

namespace deep_gemm {

// JIT runtime for `sm100_mega_moe_pre_dispatch` (see
// `deep_gemm/include/deep_gemm/impls/sm100_mega_moe_pre_dispatch.cuh`).
// Templated on (kGroupSize, kMmaKind, kUsePDL); host fn picks the
// instantiation from explicit args.
class SM100MegaMoEPreDispatchRuntime final {
public:
    struct Args {
        int group_size;
        MmaKind mma_kind;
        bool use_pdl;

        // Runtime args (passed to the kernel via the params struct).
        const void* x;
        const void* topk_idx;
        const void* topk_weights;
        const void* expert_scales;
        void*       buf_x;
        void*       buf_x_sf;
        void*       buf_x_scales;
        void*       buf_topk_idx;
        void*       buf_topk_weights;
        uint32_t    num_tokens;
        uint32_t    padded_max;
        uint32_t    hidden;
        uint32_t    num_groups;
        uint32_t    top_k;

        deep_jit::cuda::LaunchOptions launch_args;
    };

    static std::string generate(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm100_mega_moe_pre_dispatch.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&mega_moe_pre_dispatch_kernel<
        {}, {}, {}
    >);
}};
)", args.group_size,
    to_mma_kind_name(args.mma_kind),
    args.use_pdl ? "true" : "false");
    }

    static void launch(const std::shared_ptr<deep_jit::cuda::Kernel>& kernel, const Args& args) {
        jit->launch(kernel, args.launch_args,
            args.x, args.topk_idx, args.topk_weights, args.expert_scales,
            args.buf_x, args.buf_x_sf, args.buf_x_scales,
            args.buf_topk_idx, args.buf_topk_weights,
            args.num_tokens, args.padded_max, args.hidden, args.num_groups, args.top_k);
    }
};

// Host entry point. Layout contract (matches DeepGEMM's mega symm buffer):
//   - x:            (M, H) bf16, contiguous.
//   - topk_idx:     (M, K) int32, contiguous.
//   - topk_weights: (M, K) float, contiguous.
//   - buf_x:        (P, H) fp8_e4m3 for `fp8xfp4`, else (P, H/2) int8 (packed FP4).
//   - buf_x_sf:     (P, G/4) int32, contiguous; G = H / group_size; each int32
//                   stores 4 UE8M0 bytes row-major.
//   - buf_topk_idx: (P, K) int64.
//   - buf_topk_weights: (P, K) float.
//
// Pad-fill: rows in [num_tokens, padded_max) of buf_topk_idx / buf_topk_weights
// are filled with (-1, 0). buf_x and buf_x_sf rows in that range are NOT
// touched (the kernel only writes valid-token rows; pad rows must have been
// pre-zeroed by the caller if they need defined values).
static void mega_moe_pre_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& buf_x,
    const torch::Tensor& buf_x_sf,
    const torch::Tensor& buf_topk_idx,
    const torch::Tensor& buf_topk_weights,
    const int& num_tokens,
    const int& group_size,
    const std::string& mma_type,
    const std::optional<torch::Tensor>& buf_x_scales,
    const std::optional<torch::Tensor>& expert_scales) {
    const auto mma_kind = parse_mma_kind(mma_type);
    DG_HOST_ASSERT(mma_kind != MmaKind::BF16);
    const bool is_packed_fp4 = mma_kind == MmaKind::MXFP4 or mma_kind == MmaKind::NVFP4;
    if (mma_kind == MmaKind::NVFP4) {
        // NVFP4 per-token acts: 16-elem UE4M3 block SFs plus one FP32 outer
        // scale per token (written to `buf_x_scales`).
        DG_HOST_ASSERT(group_size == 16);
        DG_HOST_ASSERT(buf_x_scales.has_value());
        DG_HOST_ASSERT(buf_x_scales->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(buf_x_scales->is_contiguous());
        DG_HOST_ASSERT(buf_x_scales->numel() >= num_tokens);
    } else {
        DG_HOST_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
        DG_HOST_ASSERT(!buf_x_scales.has_value());
    }
    if (expert_scales.has_value()) {
        DG_HOST_ASSERT(expert_scales->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(expert_scales->is_contiguous());
        DG_HOST_ASSERT(expert_scales->dim() == 1);
    }
    DG_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(x.is_contiguous());
    DG_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(topk_idx.is_contiguous() && topk_weights.is_contiguous());
    DG_HOST_ASSERT(x.dim() == 2 && topk_idx.dim() == 2 && topk_weights.dim() == 2);
    DG_HOST_ASSERT(buf_x.dim() == 2 && buf_x_sf.dim() == 2);
    DG_HOST_ASSERT(buf_topk_idx.dim() == 2 && buf_topk_weights.dim() == 2);
    DG_HOST_ASSERT(buf_topk_idx.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(buf_topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(buf_x_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(buf_x_sf.is_contiguous());

    const auto m = static_cast<int>(x.size(0));
    const auto hidden = static_cast<int>(x.size(1));
    const auto top_k = static_cast<int>(topk_idx.size(1));
    const auto padded_max = static_cast<int>(buf_x.size(0));

    DG_HOST_ASSERT(num_tokens == m);
    DG_HOST_ASSERT(num_tokens <= padded_max);
    DG_HOST_ASSERT(static_cast<int>(topk_idx.size(0)) == m);
    DG_HOST_ASSERT(static_cast<int>(topk_weights.size(0)) == m);
    DG_HOST_ASSERT(static_cast<int>(topk_weights.size(1)) == top_k);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_idx.size(0)) == padded_max);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_idx.size(1)) == top_k);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_weights.size(0)) == padded_max);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_weights.size(1)) == top_k);

    DG_HOST_ASSERT(hidden % group_size == 0);
    const auto num_groups = hidden / group_size;
    DG_HOST_ASSERT(num_groups % 4 == 0);
    DG_HOST_ASSERT(static_cast<int>(buf_x_sf.size(0)) == padded_max);
    DG_HOST_ASSERT(static_cast<int>(buf_x_sf.size(1)) == num_groups / 4);

    if (is_packed_fp4) {
        // Packed FP4: (P, hidden/2) bytes. The symm-buffer slice views this
        // as kPackedFP4 (int8); accept either int8 / uint8 / float8_e4m3fn
        // re-views since callers may bind the slot differently.
        DG_HOST_ASSERT(static_cast<int>(buf_x.size(1)) == hidden / 2);
        DG_HOST_ASSERT(buf_x.element_size() == 1);
    } else {
        DG_HOST_ASSERT(buf_x.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(static_cast<int>(buf_x.size(1)) == hidden);
    }

    DG_HOST_ASSERT(hidden % 8 == 0);
    const auto num_threads = hidden / 8;
    DG_HOST_ASSERT(num_threads <= 1024);
    DG_HOST_ASSERT(num_threads >= top_k);

    const auto pad_slots = (padded_max - num_tokens) * top_k;
    const auto num_pad_blocks = pad_slots == 0 ? 0
        : math::ceil_div(pad_slots, num_threads);
    const auto num_total_blocks = num_tokens + num_pad_blocks;
    if (num_total_blocks == 0) return;

    const bool use_pdl = *jit->default_launch_options.enable_pdl;

    SM100MegaMoEPreDispatchRuntime::Args args = {
        .group_size = group_size,
        .mma_kind = mma_kind,
        .use_pdl = use_pdl,
        .x = x.const_data_ptr(),
        .topk_idx = topk_idx.const_data_ptr(),
        .topk_weights = topk_weights.const_data_ptr(),
        .expert_scales = expert_scales.has_value()
            ? expert_scales->const_data_ptr() : nullptr,
        .buf_x = buf_x.data_ptr(),
        .buf_x_sf = buf_x_sf.data_ptr(),
        .buf_x_scales = buf_x_scales.has_value()
            ? buf_x_scales->data_ptr() : nullptr,
        .buf_topk_idx = buf_topk_idx.data_ptr(),
        .buf_topk_weights = buf_topk_weights.data_ptr(),
        .num_tokens = static_cast<uint32_t>(num_tokens),
        .padded_max = static_cast<uint32_t>(padded_max),
        .hidden = static_cast<uint32_t>(hidden),
        .num_groups = static_cast<uint32_t>(num_groups),
        .top_k = static_cast<uint32_t>(top_k),
        .launch_args = make_launch_options(num_total_blocks, num_threads, /*smem_size=*/0,
                                  /*cluster_dim=*/1, /*enable_pdl=*/use_pdl)
    };

    const auto code = SM100MegaMoEPreDispatchRuntime::generate(args);
    const auto kernel = jit->compile("sm100_mega_moe_pre_dispatch", code);
    SM100MegaMoEPreDispatchRuntime::launch(kernel, args);
}

} // namespace deep_gemm
""",
    "csrc/runtime/jit.hpp": r"""#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

// pd005: the patched MegaMoE headers (`<library root>/include`) shadow the sealed image's DeepGEMM include tree
inline void init_jit(const std::string& library_root_path, const std::string& image_include_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION) + "-pd065-m462",
        {include_dir, std::filesystem::absolute(image_include_path)},
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        // Preserve the downstream family targets and explicit SM120 code generation.
        // DeepJIT requires CUDA 12.9+, where the family suffix is available.
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

}  // namespace deep_gemm
""",
    "csrc/runtime/launch.hpp": r"""#pragma once

#include <algorithm>
#include "runtime.hpp"

namespace deep_gemm {

// Construct native DeepJIT options for the downstream multidimensional launchers.
inline deep_jit::cuda::LaunchOptions make_launch_options(
    const dim3& grid, const int num_threads, const int smem_size = 0,
    const int cluster_size = 1, const std::optional<bool> enable_pdl = std::nullopt) {
    return {
        .num_smem_bytes = smem_size,
        .grid_dim = grid,
        .block_dim = dim3(num_threads, 1, 1),
        .cluster_dim = dim3(std::max(1, cluster_size), 1, 1),
        .enable_pdl = enable_pdl,
    };
}

}  // namespace deep_gemm
""",
    "csrc/runtime/runtime.hpp": r"""#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/lazy.hpp>

#include "../utils/exception.hpp"
#include "jit.hpp"

namespace deep_gemm {

class Runtime {
public:
    int num_sms = 0, tc_util = 0;

    // cuBLASLt utils
    // cuBLAS will select worse heuristics when workspace > 16 MiB with shape, e.g. m=128, n=7168, k=16384
    static constexpr size_t kCublasLtWorkspaceSize = 16 * 1024 * 1024;
    // cuBLASLt may use the workspace asynchronously, so concurrent streams cannot share it.
    std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> cublaslt_workspaces;

    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle;
    bool use_pytorch_managed_cublaslt_handle;
    bool use_temp_cublaslt_workspace;

    explicit Runtime() {
        // Whether to use PyTorch cuBLASLt
        // By default, we don't use it,
        // as `at::cuda::getCurrentCUDABlasLtHandle` has large CPU overhead with some PyTorch versions
        use_pytorch_managed_cublaslt_handle = deep_jit::get_env<int>("DG_USE_PYTORCH_CUBLASLT_HANDLE", 0) > 0;
        // Whether to create workspace tensor on each call instead of holding one.
        // Enabled by compute-sanitizer tests, which trigger `cudaErrorCudartUnloading`
        // when the workspace tensor is destructed after CUDA driver shutdown.
        use_temp_cublaslt_workspace = deep_jit::get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;

        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    }

    ~Runtime() noexcept(false) {
        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    cublasLtHandle_t get_cublaslt_handle() const {
        if (use_pytorch_managed_cublaslt_handle)
            return at::cuda::getCurrentCUDABlasLtHandle();

        // Self-managed handle
        return cublaslt_handle;
    }

    torch::Tensor get_cublaslt_workspace(const c10::cuda::CUDAStream& stream) {
        const auto options = dtype(torch::kByte).device(stream.device());
        if (use_temp_cublaslt_workspace)
            return torch::empty({kCublasLtWorkspaceSize}, options);

        auto& workspace = cublaslt_workspaces[stream];
        if (not workspace.defined())
            workspace = torch::empty({kCublasLtWorkspaceSize}, options);
        return workspace;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 < new_num_sms and new_num_sms <= jit->device.get_num_sms());
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = jit->device.get_num_sms();
        return num_sms;
    }

    bool is_cublaslt_available() {
        return get_num_sms() == jit->device.get_num_sms();
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

};

inline auto runtime = deep_jit::LazyInit<Runtime>([](){ return std::make_shared<Runtime>(); });

}  // namespace deep_gemm
""",
    "csrc/utils/compatibility.hpp": r"""#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <deep_gemm/common/exception.cuh>

DG_STATIC_ASSERT(TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3),
                 "DeepGEMM requires PyTorch 2.3 or newer");
DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer");
""",
    "csrc/utils/exception.hpp": r"""#pragma once

#include <exception>
#include <sstream>
#include <string>

#include <cublasLt.h>

#include "compatibility.hpp"

namespace deep_gemm {

class DGException final : public std::exception {
    std::string message = {};

public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw DGException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_HOST_UNREACHABLE
#define DG_HOST_UNREACHABLE(reason) (throw DGException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef DG_CUDA_RUNTIME_CHECK
#define DG_CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != cudaSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw DGException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef DG_CUBLASLT_CHECK
#define DG_CUBLASLT_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        std::ostringstream ss; \
        ss << static_cast<int>(e) << " (" << cublasGetStatusString(e) << ")"; \
        throw DGException("cuBLASLt", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

} // namespace deep_gemm
""",
    "csrc/utils/layout.hpp": r"""#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/torch.h>

#include "math.hpp"
#include "exception.hpp"
#include "../runtime/jit.hpp"

namespace deep_gemm {

// Major-ness stuffs
template <bool kRequireContiguousBatch = true>
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if constexpr (kRequireContiguousBatch) {
        if (dim == 3)
            DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    }
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

template <bool kRequireContiguousBatch = true>
static cute::UMMA::Major get_major_type_ab(const torch::Tensor& t) {
    major_check<kRequireContiguousBatch>(t);
    return t.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
}

template <bool kRequireContiguousBatch = true>
static void check_major_type_cd(const torch::Tensor& t) {
    // NOTES: the library only supports row-major output layouts
    major_check<kRequireContiguousBatch>(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static bool fp8_fp4_requires_k_major(const torch::Tensor& a, const torch::Tensor& b) {
    return jit->device.get_arch_major() == 9 or
           (a.scalar_type() == kPackedFP4 and b.scalar_type() == kPackedFP4);
}

// Tensor utils
template <int N>
static auto get_shape(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.is_cuda());
    DG_HOST_ASSERT(t.dim() == N);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

// Returns logical shape for packed FP4 by expanding the last dimension.
template <int N>
static auto get_logical_shape(const torch::Tensor& t) {
    auto shape = get_shape<N>(t);
    if (t.scalar_type() == kPackedFP4)
        std::get<N - 1>(shape) *= 2;
    return shape;
}

static std::tuple<int, int> check_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& arch_major) {
    auto [mn, k] = get_shape<2>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and (arch_major == 10 or arch_major == 12));
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(mn, k);
}

static std::tuple<int, int, int> check_grouped_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& arch_major) {
    auto [num_groups, mn, k] = get_shape<3>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and (arch_major == 10 or arch_major == 12));
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(num_groups, mn, k);
}

// Recipe
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(sfa_dtype == torch::kFloat and sfb_dtype == torch::kFloat);
        return {1, 128, 128};
    } else if (arch_major == 10 or arch_major == 12) {
        DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):   // Legacy format
            std::make_tuple(1,   1, 128);   // 1D1D kernels
    }
    DG_HOST_UNREACHABLE("Unknown recipe");
}

// SF layouts
static torch::Tensor check_sf_layout(const torch::Tensor& sf,
                                     const int& mn, const int& k,
                                     const int& gran_mn, const int& gran_k,
                                     const std::optional<int>& num_groups,
                                     const bool& tma_stride_check = false,
                                     const bool& sm90_sfb_check = false,
                                     const std::optional<torch::ScalarType>& type_check = std::nullopt) {
    // Type check
    if (type_check.has_value())
        DG_HOST_ASSERT(sf.scalar_type() == type_check.value());

    // Always do shape checks
    const auto sf_dtype = sf.scalar_type();
    DG_HOST_ASSERT(sf_dtype == torch::kFloat or sf_dtype == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);
    if (num_groups.has_value())
        DG_HOST_ASSERT(sf.size(-3) == num_groups.value());
    DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
    DG_HOST_ASSERT(sf.size(-1) == ceil_div(k, gran_k * (sf_dtype == torch::kFloat ? 1 : 4)));

    // TMA stride checks: TMA aligned and MN-major
    if (tma_stride_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));
        // Check contiguity in the MN direction
        DG_HOST_ASSERT(sf.stride(-2) == 1 or mn == 1);
        DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size()));
    }

    // SM90 SFB must be contiguous, or contiguous after transposing the last two dimensions
    if (sm90_sfb_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.size(-2) * sf.size(-1));
        DG_HOST_ASSERT((sf.stride(-1) == 1 and sf.stride(-2) == sf.size(-1)) or
                       (sf.stride(-1) == sf.size(-2) and sf.stride(-2) == 1));
    }
    return sf;
}

} // namespace deep_gemm
""",
    "csrc/utils/math.hpp": r"""// TODO: merge this file with `math.cuh` (the device part)
#pragma once

#include <torch/torch.h>

#include "exception.hpp"

namespace deep_gemm {

// TODO: use `torch::kFloat4_e2m1fn_x2`
constexpr auto kPackedFP4 = torch::kInt8;

template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

static torch::Tensor fp4_repack_to_k_major(const torch::Tensor& a, int logical_mn) {
    DG_HOST_ASSERT(a.scalar_type() == kPackedFP4);
    const int ndim = a.dim();
    DG_HOST_ASSERT(ndim == 2 or ndim == 3);
    const int mn_packed = a.size(-2);
    const int k = a.size(-1);
    DG_HOST_ASSERT(mn_packed * 2 == logical_mn and k % 2 == 0);

    auto lo = a.bitwise_and(0x0F);
    auto hi = a.to(torch::kByte).bitwise_right_shift(4).to(torch::kInt8).bitwise_and(0x0F);

    auto shape_full = a.sizes().vec();
    shape_full[ndim - 2] = logical_mn;
    auto codes = torch::empty(shape_full, a.options());
    using S = torch::indexing::Slice;
    codes.index_put_({torch::indexing::Ellipsis, S(0, torch::indexing::None, 2), S()}, lo);
    codes.index_put_({torch::indexing::Ellipsis, S(1, torch::indexing::None, 2), S()}, hi);

    auto shape_view = shape_full;
    shape_view[ndim - 1] = k / 2;
    shape_view.push_back(2);
    auto codes2 = codes.view(shape_view);
    auto result = codes2.select(-1, 0).bitwise_and(0x0F)
                  .bitwise_or(codes2.select(-1, 1).bitwise_and(0x0F).to(torch::kByte)
                              .bitwise_left_shift(4).to(torch::kInt8));
    return result.contiguous();
}

} // namespace deep_gemm
""",
    "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh": r"""#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// SiTU (Kimi-K3 / EB): gate beta * tanh(g / beta) * sigmoid(g); up' linear_beta * tanh(u / linear_beta).
//
// The equal-transform shortcut is exact only for the K3 config: the gate identity
//     sigmoid(g) = (1 + tanh(g/2)) / 2,   tanh(g/2) = 2T / (1 + T^2),   T = tanh(g/4)
//   => 4 * tanh(g/4) * sigmoid(g) = 2T(1+T)^2 / (1 + T^2)
// needs beta == 4, and the tanh-only up branch needs linear_beta == 25. Both helpers fall back to
// the literal definition for any other pair, so a different config stays correct -- just slower.
template <bool kFastMath, float kSituBeta, float kSituLinearBeta>
__forceinline__ __device__ float2 sm100_fp8_fp4_mega_moe_situ_gate(float2 gate) {
    if constexpr (kSituBeta == 4.0f and kSituLinearBeta == 25.0f) {
        const float2 one = {1.0f, 1.0f};
        const float2 q = __fmul2_rn(gate, make_float2(0.25f, 0.25f));
        float2 t;
        if constexpr (kFastMath) {
            t = {__tanhf(q.x), __tanhf(q.y)};
        } else {
            t = {tanhf(q.x), tanhf(q.y)};
        }
        const float2 s = __fadd2_rn(t, one);                              // 1 + T
        const float2 d = __ffma2_rn(t, t, one);                           // 1 + T^2
        const float2 n = __fmul2_rn(__fadd2_rn(t, t), __fmul2_rn(s, s));  // 2T (1+T)^2
        if constexpr (kFastMath) {
            return __ffma2_rn(n, make_float2(math::fast_rcp(d.x), math::fast_rcp(d.y)), make_float2(0.0f, 0.0f));
        } else {
            return {n.x / d.x, n.y / d.y};
        }
    } else {
        float2 t;
        if constexpr (kFastMath) {
            t = {__tanhf(gate.x / kSituBeta), __tanhf(gate.y / kSituBeta)};
        } else {
            t = {tanhf(gate.x / kSituBeta), tanhf(gate.y / kSituBeta)};
        }
        const auto neg_gate_exp = make_float2(
            kFastMath ? __expf(-gate.x) : expf(-gate.x),
            kFastMath ? __expf(-gate.y) : expf(-gate.y));
        const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
        float2 sig;
        if constexpr (kFastMath) {
            sig = {math::fast_rcp(denom.x), math::fast_rcp(denom.y)};
        } else {
            sig = {1.0f / denom.x, 1.0f / denom.y};
        }
        return __fmul2_rn(__fmul2_rn(t, make_float2(kSituBeta, kSituBeta)), sig);
    }
}

template <bool kFastMath, float kSituBeta, float kSituLinearBeta>
__forceinline__ __device__ float2 sm100_fp8_fp4_mega_moe_situ_up(float2 up) {
    if constexpr (kSituBeta == 4.0f and kSituLinearBeta == 25.0f) {
        if constexpr (kFastMath) {
            const float2 q = __fmul2_rn(up, make_float2(0.04f, 0.04f));
            const float2 t = {__tanhf(q.x), __tanhf(q.y)};
            return __ffma2_rn(t, make_float2(kSituLinearBeta, kSituLinearBeta), make_float2(0.0f, 0.0f));
        } else {
            return {kSituLinearBeta * tanhf(up.x / kSituLinearBeta),
                    kSituLinearBeta * tanhf(up.y / kSituLinearBeta)};
        }
    } else {
        float2 t;
        if constexpr (kFastMath) {
            t = {__tanhf(up.x / kSituLinearBeta), __tanhf(up.y / kSituLinearBeta)};
        } else {
            t = {tanhf(up.x / kSituLinearBeta), tanhf(up.y / kSituLinearBeta)};
        }
        return __fmul2_rn(t, make_float2(kSituLinearBeta, kSituLinearBeta));
    }
}

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    MmaKind kMmaKind,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    float kSwiGLUAlpha,
    bool kUseSitu,
    bool kFastMath,
    bool kUseXScales,
    bool kWithL1Alphas,
    bool kWithL2Alphas,
    bool kWithL2ActScales,
    // When true, the L2 epilogue ships FP8 E4M3 + a per-(token, N block) UE8M0 SF over
    // NVLink instead of BF16, and the combine reduce dequantizes on the fly. Halves the
    // second all-to-all's bytes per token. Independent of the MMA kind, which only
    // controls the dispatch a2a and the mainloops.
    bool kUseFp8Combine,
    typename weight_dtype_t,
    float kSituBeta = 4.0f,
    float kSituLinearBeta = 25.0f,
    bool kWithPostExpertNorm = false,
    bool kPostNormWeightFirst = false,
    bool kPostNormHasWeight = false,
    float kPostNormEps = 0.0f,
    bool kUseTrtllmWeights = false,
    // pd005 MEGASCHED (DeepGEMM #462) knobs; defaults reproduce the image kernel
    uint32_t STORE_BLOCK_M_L2 = STORE_BLOCK_M,      // `STORE_BLOCK_M` is the L1 store block
    uint32_t kNumEpiPrefetchStages = 0,             // #462: BLOCK_M == 240 ? 1 : 0
    uint32_t kNumLocalityDomainsArg = 1,            // #462: 2 (per-die task queues)
    bool kEpi462 = false,                           // #462 epilogue: TMEM prefetch ring, release after the last valid block
    bool kRegs462 = false,                          // #462 register split for > 64 experts a rank: 72 / 168
    bool kSharedFirstCombine = false,               // #462 combine order (numerics): shared slot first
    bool kFenceTaskRelease = false,                 // #441 task-slot fence
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const float* __restrict__ l1_alphas,
                            const float* __restrict__ l2_alphas,
                            const float* __restrict__ l2_act_scales,
                            const float* __restrict__ post_expert_norm_weight,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf,
                            const uint8_t* sm_locality_domains) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // pd005: the two store blocks, and #462's domain count (one domain when clusters cannot split evenly)
    constexpr uint32_t STORE_BLOCK_M_L1 = STORE_BLOCK_M;
    constexpr uint32_t kNumLocalityDomains =
        kNumLocalityDomainsArg > 1 and
        L1_SHAPE_N % (BLOCK_N * 2 * layout::kNumDeviceLocalityDomains) == 0 and
        L2_SHAPE_N % (BLOCK_N * 2 * layout::kNumDeviceLocalityDomains) == 0 ? layout::kNumDeviceLocalityDomains : 1;
    DG_STATIC_ASSERT(kEpi462 or kNumEpiPrefetchStages == 0, "The TMEM prefetch needs the #462 epilogue");

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(not kWithPostExpertNorm or kNumSharedExperts == 0,
                     "Post-expert norm does not support shared experts");
    DG_STATIC_ASSERT(kWithPostExpertNorm or not kPostNormHasWeight,
                     "Post-expert norm weight without the post-expert norm");
    DG_STATIC_ASSERT(not kUseTrtllmWeights or
                     (kMmaKind == MmaKind::NVFP4 and kNumSharedExperts == 0),
                     "TRT weight layout requires routed NVFP4 experts");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    DG_STATIC_ASSERT(kMmaKind == MmaKind::NVFP4 or kMmaKind == MmaKind::MXFP4 or kMmaKind == MmaKind::MXFP8FP4,
                     "Invalid MMA kind");
    constexpr bool kIsNVFP4 = kMmaKind == MmaKind::NVFP4;
    constexpr bool kIsMXFP4 = kMmaKind == MmaKind::MXFP4;
    constexpr bool kIsFP4Acts = kIsNVFP4 or kIsMXFP4;

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        kMmaKind,
        kNumSharedExperts,
        kUseFp8Combine
    );
    const auto workspace = buffer.workspace;
    constexpr uint32_t kNumMaxPoolTokens = layout::get_num_max_pool_tokens<uint32_t>(
        kNumRanks, kNumMaxTokensPerRank, kNumTopk, kNumExpertsPerRank);
    constexpr bool kRingCoversFullPool = kNumRingTokens >= kNumMaxPoolTokens;
    constexpr bool kUseFullPoolFP8FP4Path =
        not kHasShared and kRingCoversFullPool and BLOCK_K == BLOCK_N and
        (L2_SHAPE_K / BLOCK_K) <= 32;
    const auto get_ring_block_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kRingCoversFullPool)
            return pool_block_idx;
        else
            return pool_block_idx % kNumRingBlocks;
    };
    const auto get_ring_token_idx = [](const uint32_t& pool_token_idx) {
        if constexpr (kRingCoversFullPool)
            return pool_token_idx;
        else
            return pool_token_idx % kNumRingTokens;
    };
    const auto get_ring_wave_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kRingCoversFullPool)
            return 0u;
        else
            return pool_block_idx / kNumRingBlocks;
    };

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF and its buffer configs
    constexpr uint32_t kGranK = get_sf_gran_k(kMmaKind);
    constexpr uint32_t kSFChunkK = kGranK * 4;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kNumSFChunksPerBlock = BLOCK_K / kSFChunkK;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    constexpr uint32_t kCombineGranK = 128;
    DG_STATIC_ASSERT(kHidden % kCombineGranK == 0, "kHidden must be a multiple of 128 for FP8 combine SF");
    // Data types
    // NOTES: MXFP8FP4 uses FP8 (e4m3) activations with FP4 (e2m1) weights unpacked into
    // 8-bit smem containers; NVFP4/MXFP4 use packed FP4 for both. A and B tiles are therefore
    // always addressed in bytes, and only the MMA instruction descriptor sees the real formats.
    // Dense `float_e2m1_t` resolves to `MXF4Format::E2M1`, `float_e2m1_unpacksmem_t` to
    // `MXF8F6F4Format::E2M1` — mixing them up launches but hits `cudaErrorIllegalInstruction`
    constexpr uint32_t kNumElemBits = kIsFP4Acts ? 4 : 8;
    using a_dtype_t = uint8_t;
    using b_dtype_t = uint8_t;
    using shared_b_dtype_t = uint8_t;
    using mma_a_fmt_t = cute::conditional_t<kIsFP4Acts, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
    using mma_b_fmt_t = cute::conditional_t<kIsFP4Acts, cutlass::float_e2m1_t, weight_dtype_t>;
    constexpr bool kIsWeightFP8 = cute::is_same_v<weight_dtype_t, cutlass::float_e4m3_t>;
    using mma_shared_b_fmt_t = cute::conditional_t<kIsFP4Acts, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
    using sf_fmt_t = cute::conditional_t<kIsNVFP4, cutlass::float_ue4m3_t, cutlass::float_ue8m0_t>;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t BLOCK_K_BYTES = BLOCK_K * kNumElemBits / 8;
    constexpr uint32_t UMMA_BLOCK_K_BYTES = 128;
    constexpr uint32_t UMMA_K_BYTES = 32;
    constexpr uint32_t UMMA_K = UMMA_K_BYTES * 8 / kNumElemBits;
    constexpr uint32_t kNumMMAsPerSFChunk = kSFChunkK / UMMA_K;
    constexpr uint32_t kNumRoutedBTxTiles = (kIsFP4Acts or kIsWeightFP8) ? 2 : 1;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K_BYTES % UMMA_BLOCK_K_BYTES == 0, "Invalid block K");
    DG_STATIC_ASSERT(kNumMMAsPerSFChunk == (kIsNVFP4 ? 1u : kIsMXFP4 ? 2u : 4u), "Invalid SF chunking");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    constexpr uint64_t kActsCacheHint = static_cast<uint64_t>(BLOCK_M <= 128 ?
        cute::TMA::CacheHintSm100::EVICT_LAST : cute::TMA::CacheHintSm100::EVICT_NORMAL);

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t L1_OUT_BLOCK_N_BYTES = L1_OUT_BLOCK_N * kNumElemBits / 8;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M_L1 / 2; // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) uint8_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M_L1 * L1_OUT_BLOCK_N_BYTES];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M_L2 * BLOCK_N];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K_BYTES];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K_BYTES];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / kSFChunkK)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / kSFChunkK)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
        float nvfp4_sf_inv_lut[kIsNVFP4 ? math::kNumNVFP4SFInvLutEntries : 1];
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    } else if (warp_idx == 4) {
        if constexpr (kIsNVFP4)
            math::init_nvfp4_sf_inv_lut(shared_storage.nvfp4_sf_inv_lut, lane_idx);
    }
    cute::cluster_sync();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts,
        kNumLocalityDomains, kFenceTaskRelease>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Adjust registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : (kRegs462 ? 72 : 88);
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : (kRegs462 ? 168 : 160);
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Weight-first folds the producing rank's `post_expert_norm_weight` into the sum of
    // squares, so that weight must be identical on every rank.
    // ponytail: BF16 combine only; the FP8 combine would have to fold in its per-tile SF
    constexpr bool kFusedPostNormPartials = kWithPostExpertNorm and not kUseFp8Combine;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Ordered before any adder: locally by the grid sync below, across ranks by the
        // `kBeforeDispatchPullBarrierTag` barrier every rank enters after it.
        if constexpr (kFusedPostNormPartials) {
            const auto sum_sq_ptr = workspace.get_post_norm_sum_sq_ptr();
            for (uint32_t i = sm_idx * kNumDispatchThreads + thread_idx;
                 i < num_tokens * kNumTopk; i += kNumSMs * kNumDispatchThreads)
                sum_sq_ptr[i] = 0.0f;
        }

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            // Push this launch's grid index to every peer for tagging combine readiness; +1 differs from a zeroed workspace
            DG_STATIC_ASSERT(kNumRanks <= kNumDispatchThreads, "Insufficient threads for the grid index push");
            if (thread_idx < kNumRanks) {
                // Graph replays reuse grid IDs. Clear local readiness before the
                // dispatch NVLink barrier permits this launch's peer notifications.
                *workspace.get_combine_ready_grid_idx_ptr(thread_idx) = 0;
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            }
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumHiddenTokenBytes = kHidden * kNumElemBits / 8;
            constexpr uint32_t kNumChunks = kNumHiddenTokenBytes / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kNumHiddenTokenBytes, "kNumBytesPerPull must divide the token bytes");
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            if constexpr (not kRingCoversFullPool) {
                const auto l1_empty_count_target = get_ring_wave_idx(pool_block_idx) * kNumL1BlockNs;
                if (l1_empty_count_target > 0) {
                    const auto empty_ptr = workspace.get_l1_empty_count_ptr(get_ring_block_idx(pool_block_idx));
                    while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                }
            }

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(get_ring_token_idx(pool_token_idx)).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if constexpr (kUseFullPoolFP8FP4Path) {
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(), src_base_ptr, pull_mbarrier, kNumHiddenTokenBytes);
                }
            } else {
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumChunks; ++ i) {
                        ptx::tma_load_1d(
                            pull_buffer.get_base_ptr(),
                            math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                            pull_mbarrier, kNumBytesPerPull
                        );
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                        i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                    }
                }
            }
            __syncwarp();

            // Load the weight first, so that its remote latency overlaps with the SF copy below
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / kSFChunkK;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % kSFChunkK == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = get_ring_block_idx(pool_block_idx);
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(get_ring_token_idx(pool_token_idx)).template get_base_ptr<float>() = weight;

                if constexpr (kUseXScales) {
                    const auto x_scale = *sym_buffer.map(
                        buffer.input_x_scales_buffer.get_base_ptr<float>() + src_token_idx,
                        current_rank_in_expert_idx);
                    *buffer.l1_x_scales_buffer.get_data_buffer(get_ring_token_idx(pool_token_idx)).template get_base_ptr<float>() = x_scale;
                }

                if constexpr (kUseFullPoolFP8FP4Path) {
                    // Wait for the one-shot TMA token load to complete
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumHiddenTokenBytes);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    // Store token to local L1 buffer via TMA
                    ptx::tma_store_1d(
                        dst_base_ptr, pull_buffer.get_base_ptr(), kNumHiddenTokenBytes);

                    // Write source metadata for combine write-back (logical pool token)
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(get_ring_block_idx(pool_block_idx)),
                        (token_idx == expert_end_idx - 1) ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u);
                } else {
                    // Write source metadata for combine write-back (logical pool token)
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    // Complete last chunk's store
                    issue_and_wait_pull_store(kNumChunks - 1);
                    const bool is_last_token = (token_idx == expert_end_idx - 1);
                    uint32_t ready_ring_block_idx = get_ring_block_idx(pool_block_idx);
                    if constexpr (kIsNVFP4 and not kRingCoversFullPool and kIntermediateHidden > 4096 and not kUseXScales) {
                        // Keep the ready-counter address local to this release. Reusing the
                        // empty-counter base across remote pulls spills dispatch registers.
                        asm volatile("mov.b32 %0, %1;" : "=r"(ready_ring_block_idx) : "r"(ready_ring_block_idx));
                    }
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(ready_ring_block_idx),
                        is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                    );
                }
            }
            __syncwarp();
        }

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++ i) {
                    workspace.get_l1_task_count_ptr()[i] = workspace.get_l2_task_count_ptr()[i] = 0;
                    workspace.get_shared_l1_task_count_ptr()[i] = workspace.get_shared_l2_task_count_ptr()[i] = 0;
                }
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = get_ring_block_idx(pool_block_idx);
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == sched::BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);

                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offsets from block index
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K_BYTES;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / kSFChunkK);

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K_BYTES, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2, 0, kActsCacheHint);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_a[0]) * 2 + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kSFChunkK);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K_BYTES;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ? k_block_idx * (BLOCK_K / kSFChunkK) : task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / kSFChunkK);

                // TMA copy weights with SF
                if constexpr (kUseTrtllmWeights) {
                    // L1's row swap is handled in the epilogue; L2's shuffle in the descriptor.
                    if (task_info.block_phase == sched::BlockPhase::Linear1) {
                        if (cute::elect_one_sync()) {
                            tma::copy<BLOCK_K_BYTES, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t atom = 0; atom < BLOCK_K_BYTES / 128; ++ atom) {
                            auto* tile = shared_storage.smem_b[stage_idx] + atom * LOAD_BLOCK_N * 128;
                            auto* full = reinterpret_cast<uint64_t*>(&shared_storage.full_barriers[stage_idx]);
                            const auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
                            if (cute::elect_one_sync()) {
                                cute::SM100_TMA_2SM_LOAD_4D::copy(
                                    tensor_map_b_ptr, full, cache_hint, tile,
                                    k_idx + atom * 128, 0, 0, n_idx / 32);
                            }
                        }
                    }
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        const bool is_l1 = task_info.block_phase == sched::BlockPhase::Linear1;
                        auto* full = reinterpret_cast<uint64_t*>(&shared_storage.full_barriers[stage_idx]);
                        const auto hint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
                        const uint32_t sf_row_block = n_idx / 128;
                        const uint32_t sf_k_chunk = k_block_idx * kNumSFChunksPerBlock;
                        if (is_l1)
                            cute::SM100_TMA_2SM_LOAD_3D::copy(
                                tensor_map_sfb_ptr, full, hint, shared_storage.smem_sfb[stage_idx],
                                0, sf_k_chunk, sf_row_block);
                        else
                            cute::SM100_TMA_2SM_LOAD_5D::copy(
                                tensor_map_sfb_ptr, full, hint, shared_storage.smem_sfb[stage_idx],
                                0, 0, 0, sf_k_chunk, sf_row_block);
                        if (is_leader_cta)
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                                (sizeof(SharedStorage::smem_b[0]) + sizeof(SharedStorage::smem_sfb[0])) * 2);
                        else
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                } else if (cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        tma::copy<BLOCK_K_BYTES, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * 2 + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    } else {
                        tma::copy<BLOCK_K_BYTES, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * kNumRoutedBTxTiles + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    mma_b_fmt_t, mma_a_fmt_t, float, sf_fmt_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
                >();
            auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                mma_shared_b_fmt_t, mma_a_fmt_t, float, sf_fmt_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(sizeof(weight_dtype_t) == sizeof(shared_b_dtype_t), "Weight SMEM descriptors must use identical addressing");
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K_BYTES, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K_BYTES, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;

                // Dynamic update of UMMA N based on effective M
                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                    const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t sf_chunk_idx = 0; sf_chunk_idx < BLOCK_K / kSFChunkK; ++ sf_chunk_idx) {
                            // UTCCP copy SFA and SFB to TMEM
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + sf_chunk_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] +
                                    (kUseTrtllmWeights
                                        ? i * (kNumSFChunksPerBlock * kNumUTCCPAlignedElems) + sf_chunk_idx * kNumUTCCPAlignedElems
                                        : sf_chunk_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumMMAsPerSFChunk; ++ k) {
                                const uint32_t mma_idx = sf_chunk_idx * kNumMMAsPerSFChunk + k;
                                const uint32_t atom_byte_offset = (mma_idx * UMMA_K_BYTES / UMMA_BLOCK_K_BYTES) * UMMA_BLOCK_K_BYTES;
                                const uint32_t in_atom_byte_idx = mma_idx * UMMA_K_BYTES - atom_byte_offset;
                                // `kind::mxf4` uses `scale_vec::2X`
                                const uint32_t sf_id = kIsMXFP4 ? k * 2 : k;
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sf_id, sf_id);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, atom_byte_offset * LOAD_BLOCK_M, in_atom_byte_idx);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, atom_byte_offset * LOAD_BLOCK_N, in_atom_byte_idx);
                                if constexpr (kIsNVFP4) {
                                    ptx::SM100_MMA_MXF4NVF4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or mma_idx > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                } else if constexpr (kIsMXFP4) {
                                    ptx::SM100_MMA_MXF4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or mma_idx > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                } else {
                                    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or mma_idx > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                }
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens, sm_locality_domains);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M_L1 == 0 and WG_BLOCK_M % STORE_BLOCK_M_L2 == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M_L1 % ATOM_M == 0 and STORE_BLOCK_M_L2 % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = get_ring_block_idx(pool_block_idx);
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            // pd005 (#462): rows of this warpgroup that hold tokens, and the TMEM helpers of the prefetching epilogue
            const uint32_t wg_start_m = epilogue_wg_idx * WG_BLOCK_M;
            const uint32_t num_valid_wg_rows = valid_m <= wg_start_m ? 0u : cute::min(valid_m - wg_start_m, WG_BLOCK_M);
            const auto release_tmem = [&]() {
                ptx::tcgen05_before_thread_sync();
                shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
            };
            // Store blocks are loaded from TMEM into a ring of registers ahead of use
            const auto load_epi_block = [&]<uint32_t kNumBuffers, uint32_t kNumStoreAtoms>(
                    const uint32_t& s, uint32_t (&raw_ring)[kNumBuffers][kNumStoreAtoms][ATOM_M]) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumStoreAtoms; ++ i) {
                    auto& values = raw_ring[s % kNumBuffers][i];
                    const uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + (s * kNumStoreAtoms + i) * ATOM_M;
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                           values[0], values[1], values[2], values[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                           values[4], values[5], values[6], values[7]);
                }
            };
            // Issue the loads of store block `s` if it holds valid tokens; the last valid block is waited for right away
            // and the TMEM stage is released, as nothing else will be read from it
            const auto load_epi_block_if_valid = [&]<uint32_t kNumBuffers, uint32_t kNumStoreAtoms>(
                    const uint32_t& s, const uint32_t& num_valid_store_blocks,
                    uint32_t (&raw_ring)[kNumBuffers][kNumStoreAtoms][ATOM_M]) {
                if (s < num_valid_store_blocks) {
                    load_epi_block(s, raw_ring);
                    if (s + 1 == num_valid_store_blocks) {
                        cutlass::arch::fence_view_async_tmem_load();
                        release_tmem();
                    }
                }
            };
            // Prefetch the first `kNumBuffers - 1` store blocks (no wait); a warpgroup without valid tokens releases at once
            const auto prefetch_prologue = [&]<uint32_t kNumBuffers, uint32_t kNumStoreAtoms>(
                    const uint32_t& num_valid_store_blocks, uint32_t (&raw_ring)[kNumBuffers][kNumStoreAtoms][ATOM_M]) {
                #pragma unroll
                for (uint32_t s = 0; s + 1 < kNumBuffers; ++ s)
                    load_epi_block_if_valid(s, num_valid_store_blocks, raw_ring);
                if (num_valid_store_blocks == 0)
                    release_tmem();
            };
            // Before computing store block `s`: wait for its loads (issued a block earlier, or right here without
            // prefetch), then issue block `s + kNumBuffers - 1` so that it lands while block `s` is processed
            const auto wait_and_prefetch_next = [&]<uint32_t kNumBuffers, uint32_t kNumStoreAtoms>(
                    const uint32_t& s, const uint32_t& num_valid_store_blocks,
                    uint32_t (&raw_ring)[kNumBuffers][kNumStoreAtoms][ATOM_M]) {
                constexpr uint32_t kNumPrefetchStages = kNumBuffers - 1;
                if constexpr (kNumPrefetchStages == 0)
                    load_epi_block_if_valid(s, num_valid_store_blocks, raw_ring);
                cutlass::arch::fence_view_async_tmem_load();
                if constexpr (kNumPrefetchStages > 0)
                    load_epi_block_if_valid(s + kNumPrefetchStages, num_valid_store_blocks, raw_ring);
            };

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    // Wait L2 block empty
                    if constexpr (not kRingCoversFullPool) {
                        const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                        const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * get_ring_wave_idx(pool_block_idx);
                        while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                    }
                }

                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 1.0f;
                float stored_cached_x_scale = 1.0f;

                float2 l1_alpha = {1.0f, 1.0f};
                if constexpr (kWithL1Alphas) {
                    if (not task_info.is_shared())
                        l1_alpha = __ldg(reinterpret_cast<const float2*>(l1_alphas) +
                                         task_info.local_expert_idx);
                }

                // fc2 input global scale; the caller folds the inverse into `l2_alphas`
                float l2_act_gs = 1.0f;
                if constexpr (kWithL2ActScales) {
                    if (not task_info.is_shared())
                        l2_act_gs = __ldg(l2_act_scales + task_info.local_expert_idx);
                }

                constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M_L1 / ATOM_M;
                constexpr uint32_t kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M_L1;
                constexpr uint32_t kNumPrefetchStages = kEpi462 ? math::constexpr_min(kNumEpiPrefetchStages, kNumStoreBlocks - 1) : 0;
                uint32_t raw_ring[kNumPrefetchStages + 1][kNumAtomsPerStore][ATOM_M];
                const uint32_t num_valid_store_blocks = math::ceil_div(num_valid_wg_rows, STORE_BLOCK_M_L1);
                if constexpr (kEpi462)
                    prefetch_prologue(num_valid_store_blocks, raw_ring);

                #pragma unroll
                for (uint32_t s = 0; s < kNumStoreBlocks; ++ s) {
                    if constexpr (kEpi462) {
                        if (s >= num_valid_store_blocks)
                            break;
                        wait_and_prefetch_next(s, num_valid_store_blocks, raw_ring);
                    } else {
                        // Early break if the entire store block is beyond the valid token range
                        if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1 >= valid_m) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                            break;
                        }
                    }
                    auto& store_block_raw_values = raw_ring[s % (kNumPrefetchStages + 1)];

                    // Iterate all atoms in the store block
                    float2 activation_values[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        // NOTES: the post-expert norm applies the top-k weight in the combine
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if (not kWithPostExpertNorm and not task_info.is_shared() and
                            (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .template get_base_ptr<float>();
                        }

                        // Per-token L1 input outer scales, cached like the topk weights.
                        // Routed rows read the ring copy made at dispatch; shared rows
                        // read the local input scales directly.
                        if constexpr (kUseXScales) {
                            if ((j * ATOM_M) % 32 == 0 and
                                (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                                const uint32_t row = epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx;
                                stored_cached_x_scale = task_info.is_shared() ?
                                    *buffer.input_x_scales_buffer.get_data_buffer(m_idx + row).template get_base_ptr<float>() :
                                    *buffer.l1_x_scales_buffer.get_data_buffer(ring_m_idx + row).template get_base_ptr<float>();
                            }
                        }
                        const float2 x_scales = kUseXScales ? float2{
                            ptx::exchange(stored_cached_x_scale, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_x_scale, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        } : float2{1.0f, 1.0f};
                        const float2 gate_scales = kWithL1Alphas ?
                            __fmul2_rn(x_scales, {l1_alpha.x, l1_alpha.x}) : x_scales;
                        const float2 up_scales = kWithL1Alphas ?
                            __fmul2_rn(x_scales, {l1_alpha.y, l1_alpha.y}) : x_scales;

                        // Load weights from register cache
                        float2 weights = {1.0f, 1.0f};
                        if constexpr (not kWithPostExpertNorm) {
                            weights = {
                                ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                                ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                            };
                        }
                        if constexpr (kWithL2ActScales)
                            weights = __fmul2_rn(weights, {l2_act_gs, l2_act_gs});

                        // Load from TMEM (image: per atom; #462: the ring filled above)
                        auto& raw_values = store_block_raw_values[i];
                        if constexpr (not kEpi462) {
                            uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                   raw_values[0], raw_values[1], raw_values[2], raw_values[3]);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                   raw_values[4], raw_values[5], raw_values[6], raw_values[7]);
                            cutlass::arch::fence_view_async_tmem_load();

                            // Signal tensor memory consumed on the last atom
                            if (j == WG_BLOCK_M / ATOM_M - 1) {
                                ptx::tcgen05_before_thread_sync();
                                shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                            }
                        }

                        // Apply the selected activation: SwiGLU, OAI SwiGLU, or Kimi-K3 SiTU
                        // SiTU:
                        //   act = kSituBeta * tanh(gate/kSituBeta) * sigmoid(gate)
                        //   up' = kSituLinearBeta * tanh(up/kSituLinearBeta)
                        auto fp32_values = reinterpret_cast<float2*>(raw_values);
                        // TRT keeps `row ^ 8`, so up precedes gate
                        constexpr uint32_t kGateSlot = kUseTrtllmWeights ? 1u : 0u;
                        constexpr uint32_t kUpSlot = 1u - kGateSlot;
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            if constexpr (kUseXScales or kWithL1Alphas) {
                                fp32_values[k * 2 + kGateSlot] = __fmul2_rn(fp32_values[k * 2 + kGateSlot], gate_scales);
                                fp32_values[k * 2 + kUpSlot] = __fmul2_rn(fp32_values[k * 2 + kUpSlot], up_scales);
                            }
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + kGateSlot]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + kUpSlot]);

                            // Clamp (SwiGLU-with-limit only; SiTU soft-clips below)
                            if constexpr (not kUseSitu and kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            constexpr bool kIsOAISwiGLU = kSwiGLUAlpha != 0.0f;
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto up = __bfloat1622float2(bf16_up);
                            if constexpr (kUseSitu) {
                                // Tanh-bounded gate and soft-clipped up branch.
                                gate = sm100_fp8_fp4_mega_moe_situ_gate<kFastMath, kSituBeta, kSituLinearBeta>(gate);
                                up = sm100_fp8_fp4_mega_moe_situ_up<kFastMath, kSituBeta, kSituLinearBeta>(up);
                            } else {
                                // SwiGLU: silu(gate) * up
                                const auto sigmoid_in = kIsOAISwiGLU ?
                                    __fmul2_rn(gate, {kSwiGLUAlpha, kSwiGLUAlpha}) : gate;
                                const auto neg_gate_exp = make_float2(
                                    kFastMath ? __expf(-sigmoid_in.x) : expf(-sigmoid_in.x),
                                    kFastMath ? __expf(-sigmoid_in.y) : expf(-sigmoid_in.y));
                                const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                                float2 sig;
                                if constexpr (kFastMath) {
                                    sig = {math::fast_rcp(denom.x), math::fast_rcp(denom.y)};
                                } else {
                                    sig = {1.0f / denom.x, 1.0f / denom.y};
                                }
                                gate = __fmul2_rn(gate, sig);
                                if constexpr (kIsOAISwiGLU)
                                    up = __fadd2_rn(up, {1.0f, 1.0f});
                            }
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                            thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());

                        // Reduce amax (warp-pair-level)
                        if (not kIsNVFP4 and lane_idx < 4)
                            shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `shared_storage.amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast into the L1 output dtype and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Calculate SF
                        float2 sf, sf_inv;
                        uint2 sf_bits;
                        if constexpr (kIsNVFP4) {
                            math::get_nvfp4_sf_inv_from_lut(amax_values[i], shared_storage.nvfp4_sf_inv_lut,
                                                            sf_inv, sf_bits);
                        } else {
                            // Reduce amax (warp-pair-level)
                            const float2 wp_amax =
                                shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                            amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                            amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);
                            if constexpr (kIsMXFP4)
                                math::get_e2m1_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                            else
                                math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                            sf_bits = {*reinterpret_cast<const uint32_t*>(&sf.x) >> 23,
                                       *reinterpret_cast<const uint32_t*>(&sf.y) >> 23};
                        }

                        // Cast
                        const float2 first = __fmul2_rn(activation_values[i][0], sf_inv);
                        const float2 second = __fmul2_rn(activation_values[i][1], sf_inv);

                        const auto smem_base = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                            + i * ATOM_M * L1_OUT_BLOCK_N_BYTES;
                        if constexpr (kIsFP4Acts) {
                            const uint32_t packed = math::cast_into_e2m1x2_pairs(first, second);
                            const uint32_t byte_in_row = warp_idx_in_wg * (L1_OUT_BLOCK_N_BYTES / 4) + lane_idx / 4;
                            #pragma unroll
                            for (uint32_t t = 0; t < 2; ++ t) {
                                const uint32_t row = (lane_idx % 4) * 2 + t;
                                // 32B swizzle: 2 bank groups, XOR-ed by `row / 4`
                                const uint32_t swizzled_byte =
                                    ((byte_in_row / kNumBankGroupBytes) ^ ((row / 4) & 1)) * kNumBankGroupBytes +
                                    byte_in_row % kNumBankGroupBytes;
                                smem_base[row * L1_OUT_BLOCK_N_BYTES + swizzled_byte] =
                                    static_cast<uint8_t>(packed >> (t * 8));
                            }
                        } else {
                            const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(first.x, first.y, second.x, second.y));

                            // STSM
                            uint32_t row = lane_idx;
                            uint32_t col = warp_idx_in_wg;
                            const auto smem_ptr = smem_base
                                + row * L1_OUT_BLOCK_N_BYTES
                                // Use 64B swizzle for SwiGLU, so divided by 2
                                + (col ^ (row / 2)) * kNumBankGroupBytes;
                            ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
                        }

                        // Store SF to `buffer.l2_sf_buffer` (MN-major layout)
                        // For MXFP8FP4 only one warp per pair writes (both hold the same SF after the
                        // cross-warp reduce); for NVFP4 every warp owns its own 16-element group
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if ((kIsNVFP4 or warp_idx_in_wg % 2 == 0) and lane_idx < 4) {
                            const uint32_t k_idx = kIsNVFP4 ? n_block_idx * 4 + warp_idx_in_wg
                                                            : n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                            const auto sf_base_ptr = task_info.is_shared() ?
                                buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = task_info.m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2
                            //   - `task_info.pool_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `task_info.m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `task_info.m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1 + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_token_idx = block_idx * SF_BLOCK_M
                                + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] = static_cast<uint8_t>(sf_bits.x);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                static_cast<uint8_t>(sf_bits.y);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N_BYTES;
                        const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            tensor_map_l1_output_ptr,
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2 and increment L1 empty count
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);

                        // Increment L1 empty count for this physical slot (one per N block)
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if constexpr (not kRingCoversFullPool) {
                    if (not task_info.is_shared()) {
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            ptx::red_add(
                                workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                        }
                    }
                }
                __syncwarp();

                constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M_L2 / ATOM_M;
                constexpr uint32_t kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M_L2;
                constexpr uint32_t kNumPrefetchStages = kEpi462 ? math::constexpr_min(kNumEpiPrefetchStages, kNumStoreBlocks - 1) : 0;
                constexpr uint32_t kNumRowsPerWarp = kNumAtomsPerStore;
                uint32_t raw_ring[kNumPrefetchStages + 1][kNumAtomsPerStore][ATOM_M];
                const uint32_t num_valid_store_blocks = math::ceil_div(num_valid_wg_rows, STORE_BLOCK_M_L2);
                if constexpr (kEpi462)
                    prefetch_prologue(num_valid_store_blocks, raw_ring);

                float l2_alpha = 1.0f;
                if constexpr (kWithL2Alphas) {
                    if (not task_info.is_shared())
                        l2_alpha = __ldg(l2_alphas + task_info.local_expert_idx);
                }

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < kNumStoreBlocks; ++ s) {
                    if constexpr (kEpi462) {
                        if (s >= num_valid_store_blocks)
                            break;
                        wait_and_prefetch_next(s, num_valid_store_blocks, raw_ring);
                    } else {
                        // Early break if the entire store block is beyond the valid token range
                        // TODO: check performance
                        if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 >= valid_m) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                            break;
                        }
                    }
                    auto& store_block_raw_values = raw_ring[s % (kNumPrefetchStages + 1)];

                    // Read the source metadata of this warp's rows before the TMEM loads to overlap the latency
                    layout::TokenSrcMetadata cached_src_metadata[kNumRowsPerWarp];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        if (m_idx_in_block < valid_m)
                            cached_src_metadata[j] = task_info.is_shared() ?
                                layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk) :
                                *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                    }
                    __syncwarp();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements (image: per atom;
                        // #462: the ring filled above). Start from lane index 0 and 16
                        auto& values = store_block_raw_values[i];
                        if constexpr (not kEpi462) {
                            uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + i * ATOM_M;
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                                   values[0], values[1], values[2], values[3]);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                                   values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                        }

                        if constexpr (kWithL2Alphas) {
                            auto fp32_values = reinterpret_cast<float*>(values);
                            #pragma unroll
                            for (uint32_t v = 0; v < ATOM_M; ++ v)
                                fp32_values[v] *= l2_alpha;
                        }

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if constexpr (not kEpi462) {
                            if (s == kNumStoreBlocks - 1 and i == kNumAtomsPerStore - 1) {
                                ptx::tcgen05_before_thread_sync();
                                shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                            }
                        }

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto& [dst_rank_idx, dst_token_idx, dst_topk_idx] = cached_src_metadata[j];

                        // Read from shared memory
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        if constexpr (kUseFp8Combine) {
                            // BF16 (in `packed`) -> FP8 E4M3 + one UE8M0 SF per row tile.
                            // 16 lanes (lane_idx & ~15u) cover one row's BLOCK_N=128
                            // elements (8 BF16 each), so the per-row amax is a reduction
                            // over that 16-lane group.
                            const auto bf_pairs = reinterpret_cast<const nv_bfloat162*>(&packed);
                            float local_amax = 0.0f;
                            #pragma unroll
                            for (uint32_t q = 0; q < 4; ++q) {
                                const float2 vf = __bfloat1622float2(bf_pairs[q]);
                                local_amax = cute::max(local_amax, cute::abs(vf.x));
                                local_amax = cute::max(local_amax, cute::abs(vf.y));
                            }
                            // A 16-lane mask, not `0xffffffff`: the `m_idx_in_block >= valid_m`
                            // break above can retire the other half-warp on padding rows, and a
                            // full-warp shuffle would then wait on lanes that have already exited.
                            const uint32_t row_mask = 0x0000ffffu << (16u * (lane_idx / 16));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 1));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 2));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 4));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 8));

                            const int log2_ceil = math::fast_log2_ceil(local_amax * (1.0f / 448.0f));
                            const float sf_inv = math::fast_pow2(-log2_ceil);
                            const uint8_t sf_byte = static_cast<uint8_t>(log2_ceil + 127);

                            float4 lo, hi;
                            const auto lo_pair = __bfloat1622float2(bf_pairs[0]);
                            const auto lo_pair_b = __bfloat1622float2(bf_pairs[1]);
                            const auto hi_pair = __bfloat1622float2(bf_pairs[2]);
                            const auto hi_pair_b = __bfloat1622float2(bf_pairs[3]);
                            lo.x = lo_pair.x * sf_inv;
                            lo.y = lo_pair.y * sf_inv;
                            lo.z = lo_pair_b.x * sf_inv;
                            lo.w = lo_pair_b.y * sf_inv;
                            hi.x = hi_pair.x * sf_inv;
                            hi.y = hi_pair.y * sf_inv;
                            hi.z = hi_pair_b.x * sf_inv;
                            hi.w = hi_pair_b.y * sf_inv;
                            const __nv_fp8x4_e4m3 fp8_lo(lo);
                            const __nv_fp8x4_e4m3 fp8_hi(hi);
                            const uint64_t fp8_uint64 =
                                (uint64_t(fp8_lo.__x)) |
                                (uint64_t(fp8_hi.__x) << 32);

                            // Write 8 FP8 bytes (uint64) to remote, replacing
                            // the BF16 16-byte write.
                            const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto dst_ptr = math::advance_ptr<uint64_t>(
                                dst_token.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(uint8_t)) +
                                (lane_idx % 16) * static_cast<uint32_t>(sizeof(uint64_t)));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = fp8_uint64;

                            // 1 SF byte per row tile, written by lane 0 of the 16-lane group.
                            if ((lane_idx & 15u) == 0) {
                                const auto sf_token = buffer.combine_sf_buffer.get_rank_buffer(dst_topk_idx)
                                                     .get_data_buffer(dst_token_idx);
                                const auto sf_ptr = math::advance_ptr<uint8_t>(
                                    sf_token.get_base_ptr(),
                                    n_block_idx * static_cast<uint32_t>(sizeof(uint8_t)));
                                *sym_buffer.map(sf_ptr, dst_rank_idx) = sf_byte;
                            }
                        } else {
                            // Default BF16 path (16 bytes/lane = 8 BF16).
                            const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto dst_ptr = math::advance_ptr<float4>(
                                dst_token.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;

                            // Squares the values just stored, so the norm sees what the
                            // combine would have read back.
                            if constexpr (kFusedPostNormPartials) {
                                if (not task_info.is_shared()) {
                                    const uint32_t elem_base = n_idx + (lane_idx % 16) * 8;
                                    const auto bf_pairs = reinterpret_cast<const nv_bfloat162*>(&packed);
                                    float partial = 0.0f;
                                    #pragma unroll
                                    for (uint32_t q = 0; q < 4; ++ q) {
                                        const float2 vf = __bfloat1622float2(bf_pairs[q]);
                                        float a = vf.x, b = vf.y;
                                        // Weight-first normalizes `w * x`
                                        if constexpr (kPostNormHasWeight and kPostNormWeightFirst) {
                                            a *= __ldg(post_expert_norm_weight + elem_base + q * 2 + 0);
                                            b *= __ldg(post_expert_norm_weight + elem_base + q * 2 + 1);
                                        }
                                        partial = __fmaf_rn(a, a, partial);
                                        partial = __fmaf_rn(b, b, partial);
                                    }
                                    // 16-lane mask: the `valid_m` break can retire the other half-warp
                                    const uint32_t row_mask = 0x0000ffffu << (16u * (lane_idx / 16));
                                    #pragma unroll
                                    for (uint32_t d = 1; d < 16; d <<= 1)
                                        partial += __shfl_xor_sync(row_mask, partial, d);
                                    if ((lane_idx & 15u) == 0) {
                                        const auto sum_sq_ptr = workspace.get_post_norm_sum_sq_ptr(
                                            dst_token_idx, dst_topk_idx);
                                        if constexpr (kNumRanks == 1) {
                                            atomicAdd(sum_sq_ptr, partial);
                                        } else {
                                            // System scope: the accumulator is in a peer's workspace
                                            ptx::red_add_sys(
                                                sym_buffer.map(sum_sq_ptr, dst_rank_idx), partial);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        constexpr uint32_t kNumElemsPerLoadUint4 = kUseFp8Combine ? 16u : 8u;
        constexpr uint32_t kNumRowLoadBytes = kUseFp8Combine ? kNumHiddenBytes / 2 : kNumHiddenBytes;
        constexpr uint32_t kNumRowLoadUint4PerLane = kNumRowLoadBytes / sizeof(uint4) / 32;
        DG_STATIC_ASSERT(not kWithPostExpertNorm or
                         kNumRowLoadUint4PerLane * 32 * sizeof(uint4) == kNumRowLoadBytes,
                         "Hidden is not a whole number of 16-byte vectors per lane");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Peers' grid indices for tagging their combine readiness; the load overlaps with the grid sync
        DG_STATIC_ASSERT(kNumRanks <= kNumEpilogueThreads, "Insufficient threads for combine readiness");
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);

        // All local L2 writes are done after this grid sync
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx), peer_grid_idx);

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        const auto grid_idx = ptx::get_grid_idx() + 1;
        for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;

            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Wait for the ranks of the selected experts to finish their L2 writes
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });

            // The RMS spans the full hidden dim, of which the L2 epilogue holds only
            // `BLOCK_N`, so it lands here -- and so must the top-k weight, which the
            // scale-invariant norm would otherwise erase.
            float slot_norm_scale = 1.0f;
            if constexpr (kFusedPostNormPartials) {
                // Lane `s` reads slot `s`'s total, already accumulated by the L2 tiles.
                const float my_topk_weight = lane_idx < kNumTopk ?
                    __ldg(buffer.input_topk_weights_buffer.get_base_ptr<float>() +
                          token_idx * kNumTopk + lane_idx) : 0.0f;
                // Not `__ldg`: peers write this over NVLink, the read-only path is not coherent
                float sum_sq = 0.0f;
                if (lane_idx < kNumTopk and (total_mask >> lane_idx & 1))
                    sum_sq = *workspace.get_post_norm_sum_sq_ptr(token_idx, lane_idx);
                const float inv_rms = rsqrtf(sum_sq / static_cast<float>(kHidden) + kPostNormEps);
                slot_norm_scale = my_topk_weight * inv_rms;
                __syncwarp();
            } else if constexpr (kWithPostExpertNorm) {
                const float my_topk_weight = lane_idx < kNumTopk ?
                    __ldg(buffer.input_topk_weights_buffer.get_base_ptr<float>() +
                          token_idx * kNumTopk + lane_idx) : 0.0f;
                #pragma unroll 1
                for (uint32_t slot_idx = 0; slot_idx < kNumTopk; ++ slot_idx) {
                    // Warp-uniform: `total_mask` is a ballot result
                    if (not (total_mask >> slot_idx & 1))
                        continue;
                    const auto row_ptr = buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                         .get_data_buffer(token_idx).get_base_ptr<uint4>();
                    const uint8_t* sf_row_ptr = nullptr;
                    if constexpr (kUseFp8Combine)
                        sf_row_ptr = buffer.combine_sf_buffer.get_rank_buffer(slot_idx)
                                           .get_data_buffer(token_idx).get_base_ptr<uint8_t>();

                    float sum_sq = 0.0f;
                    #pragma unroll 4
                    for (uint32_t j = 0; j < kNumRowLoadUint4PerLane; ++ j) {
                        const uint32_t elem_idx = (j * 32 + lane_idx) * kNumElemsPerLoadUint4;
                        const uint4 raw = __ldg(row_ptr + j * 32 + lane_idx);
                        float values[kNumElemsPerLoadUint4];
                        if constexpr (kUseFp8Combine) {
                            const auto bytes = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumElemsPerLoadUint4; ++ k)
                                values[k] = static_cast<float>(bytes[k]);
                        } else {
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&raw);
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumElemsPerLoadUint4 / 2; ++ k) {
                                const float2 pair = __bfloat1622float2(bf16_values[k]);
                                values[k * 2 + 0] = pair.x;
                                values[k * 2 + 1] = pair.y;
                            }
                        }

                        float partial = 0.0f;
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumElemsPerLoadUint4; ++ k) {
                            float value = values[k];
                            // Weight-first normalizes `w * x`, so `w` is inside the RMS
                            if constexpr (kPostNormHasWeight and kPostNormWeightFirst)
                                value *= __ldg(post_expert_norm_weight + elem_idx + k);
                            partial = __fmaf_rn(value, value, partial);
                        }
                        if constexpr (kUseFp8Combine) {
                            // All 16 elements share one SF, so it factors out of the square
                            const float sf = math::fast_pow2(
                                static_cast<int>(__ldg(sf_row_ptr + elem_idx / kCombineGranK)) - 127);
                            partial *= sf * sf;
                        }
                        sum_sq += partial;
                    }

                    const float inv_rms = rsqrtf(
                        math::warp_reduce_sum(sum_sq) / static_cast<float>(kHidden) + kPostNormEps);
                    if (lane_idx == slot_idx)
                        slot_norm_scale = my_topk_weight * inv_rms;
                }
            }

            // Stream B: FP8 path loads kNumChunkBytes / 2 per slot (FP8 = 1 byte/elem)
            // and reads a per-(slot, token, n_block) UE8M0 SF byte to dequant
            // on the fly. Output is BF16 either way → store byte count is
            // kNumChunkBytes regardless.
            constexpr uint32_t kNumLoadBytesPerChunk =
                kUseFp8Combine ? (kNumChunkBytes / 2) : kNumChunkBytes;
            constexpr uint32_t kNumLoadUint4PerLane =
                kUseFp8Combine ? (kNumUint4PerLane / 2) : kNumUint4PerLane;
            // Per-uint4 load: BF16 → 8 BF16 = 4 float2 pairs.
            //                 FP8  → 16 FP8 = 8 float2 pairs (dequant'd).
            constexpr uint32_t kNumF32PairsPerLoadUint4 =
            kUseFp8Combine ? 8u : 4u;
            // Per-element offset in the chunk for SF lookup:
            //   sf_idx = (chunk * kNumLoadElemsPerChunk + elem_in_chunk) / 128
            constexpr uint32_t kNumLoadElemsPerChunk = kHidden / kNumChunks;

            // Process the chunk assigned by the upstream token-chunk scheduler.
            {
                const uint32_t chunk = chunk_idx;
                const uint32_t chunk_byte_offset = chunk * kNumLoadBytesPerChunk;

                // Move mask and load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) -> int {
                    if (mask) {
                        // Move (#462 with kSharedFirstCombine: shared slot first, then routed slots in top-k order)
                        const uint32_t slot_idx = (kSharedFirstCombine and kNumSharedExperts > 0 and (mask >> kNumTopk)) ?
                            kNumTopk : __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        // Load FP8 / BF16 chunk
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumLoadBytesPerChunk);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumLoadBytesPerChunk);
                        }
                        __syncwarp();
                        return static_cast<int>(slot_idx);
                    }
                    return -1;
                };

                // Load the first selection
                int active_slot = move_mask_and_load(load_stage_idx);

                // Accumulate all top-k contributions for this chunk in float registers
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (active_slot >= 0) {
                    // Prefetch next top-k into the buffer while current is being accumulated.
                    int next_slot = move_mask_and_load(load_stage_idx ^ 1);

                    // Wait for current slot's load.
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);

                    float slot_scale = 1.0f;
                    if constexpr (kWithPostExpertNorm)
                        slot_scale = __shfl_sync(0xffffffff, slot_norm_scale,
                                                 static_cast<uint32_t>(active_slot));

                    if constexpr (kUseFp8Combine) {
                        // Per-slot SF base for this token.
                        const uint8_t* sf_token_ptr =
                            buffer.combine_sf_buffer.get_rank_buffer(static_cast<uint32_t>(active_slot))
                                              .get_data_buffer(token_idx).get_base_ptr<uint8_t>();
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumLoadUint4PerLane; ++ j) {
                            const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                            // SF for the 16 elements at offset (chunk_elem_offset + (j*32 + lane)*16);
                            // since 16 < 128, all 16 elements share a single SF byte.
                            const uint32_t sf_idx =
                                (chunk * kNumLoadElemsPerChunk + (j * 32 + lane_idx) * 16) / kCombineGranK;
                            const uint8_t sf_byte = __ldg(sf_token_ptr + sf_idx);
                            float sf = math::fast_pow2(static_cast<int>(sf_byte) - 127);
                            if constexpr (kWithPostExpertNorm)
                                sf *= slot_scale;
                            const uint32_t* w = reinterpret_cast<const uint32_t*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < 4; ++ l) {
                                // Each uint32 = 4 FP8 = 2 FP8x2 — convert via FP16 intermediate.
                                uint32_t f16_lo, f16_hi;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                                             : "=r"(f16_lo) : "h"(uint16_t(w[l] & 0xFFFFu)));
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                                             : "=r"(f16_hi) : "h"(uint16_t(w[l] >> 16)));
                                float vlx, vly, vhx, vhy;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vlx) : "h"(uint16_t(f16_lo & 0xFFFFu)));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vly) : "h"(uint16_t(f16_lo >> 16)));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vhx) : "h"(uint16_t(f16_hi & 0xFFFFu)));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vhy) : "h"(uint16_t(f16_hi >> 16)));
                                auto& acc_lo = reduced[j * kNumF32PairsPerLoadUint4 + l * 2 + 0];
                                auto& acc_hi = reduced[j * kNumF32PairsPerLoadUint4 + l * 2 + 1];
                                acc_lo.x = __fmaf_rn(vlx, sf, acc_lo.x);
                                acc_lo.y = __fmaf_rn(vly, sf, acc_lo.y);
                                acc_hi.x = __fmaf_rn(vhx, sf, acc_hi.x);
                                acc_hi.y = __fmaf_rn(vhy, sf, acc_hi.y);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                            const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l) {
                                auto& acc = reduced[j * kNumElemsPerUint4 + l];
                                if constexpr (kWithPostExpertNorm) {
                                    const float2 pair = __bfloat1622float2(bf16_values[l]);
                                    acc.x = __fmaf_rn(pair.x, slot_scale, acc.x);
                                    acc.y = __fmaf_rn(pair.y, slot_scale, acc.y);
                                } else {
                                    ptx::accumulate(acc, bf16_values[l]);
                                }
                            }
                        }
                    }
                    cutlass::arch::fence_view_async_shared();
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                    active_slot = next_slot;
                }

                // Both norm forms reduce to `w * sum_k(inv_rms_k * topk_weight_k * x_k)`.
                if constexpr (kPostNormHasWeight) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumLoadUint4PerLane; ++ j) {
                        const uint32_t base_elem_idx = chunk * kNumLoadElemsPerChunk +
                                                       (j * 32 + lane_idx) * kNumElemsPerLoadUint4;
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumF32PairsPerLoadUint4; ++ l) {
                            auto& acc = reduced[j * kNumF32PairsPerLoadUint4 + l];
                            acc = __fmul2_rn(acc, __ldg(reinterpret_cast<const float2*>(
                                post_expert_norm_weight + base_elem_idx + l * 2)));
                        }
                    }
                }

                // Cast & write to smem store-buffer.
                // BF16 path: kNumUint4PerLane stores, mapping accumulator[j*4+l] → store-uint4 j.
                // FP8 path: kNumLoadUint4PerLane * 2 stores, mapping accumulator[j*8+l] → store-uint4 (j*2 + (l/4)).
                if constexpr (kUseFp8Combine) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumLoadUint4PerLane; ++ j) {
                        // Lower BF16 uint4 (8 elements: pairs 0..3 of this input uint4).
                        uint4 lo, hi;
                        auto lo_bf = reinterpret_cast<nv_bfloat162*>(&lo);
                        auto hi_bf = reinterpret_cast<nv_bfloat162*>(&hi);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l) {
                            lo_bf[l] = __float22bfloat162_rn(reduced[j * kNumF32PairsPerLoadUint4 + l]);
                            hi_bf[l] = __float22bfloat162_rn(reduced[j * kNumF32PairsPerLoadUint4 + 4 + l]);
                        }
                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        // Layout: each input uint4 j (16 elements) → 2 BF16 uint4 at
                        //   indices (j * 32 + lane_idx) * 2 + {0, 1}.
                        ptx::st_shared(combine_store_buffer + (j * 32 + lane_idx) * 2 + 0,
                                       lo.x, lo.y, lo.z, lo.w);
                        ptx::st_shared(combine_store_buffer + (j * 32 + lane_idx) * 2 + 1,
                                       hi.x, hi.y, hi.z, hi.w);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                       casted.x, casted.y, casted.z, casted.w);
                    }
                }
                __syncwarp();

                // TMA store the BF16 chunk to gmem y. Output byte offset still
                // tracks BF16 chunks (= kNumChunkBytes).
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk * kNumChunkBytes),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
""",
    "include/deep_gemm/layout/mega_moe.cuh": r"""#pragma once

#include <cstddef>
#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {

// pd005 MEGASCHED: the B300 has two dies; DeepGEMM #462 schedules MegaMoE tasks per die
static constexpr uint32_t kNumDeviceLocalityDomains = 2;

static constexpr int kNumCandidateBlockMs = 8;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192, 240};
static constexpr int kMaxCandidateBlockM = 240;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 1920;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;

    TokenSrcMetadata() = default;

    CUTLASS_DEVICE TokenSrcMetadata(const uint32_t& rank_idx, const uint32_t& token_idx, const uint32_t& topk_idx):
        rank_idx(rank_idx), token_idx(token_idx), topk_idx(topk_idx) {}
};

struct alignas(128) MegaMoESignals {
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;
    static constexpr uint32_t kNumMaxExperts = 2048;
    static constexpr uint32_t kNumMaxExpertsPerRank = 512;
    static constexpr uint32_t kNumMaxRingBlocks = 1u << 20;  // 20 MiB for the ring signals
    static constexpr uint32_t kNumMaxSharedL2Blocks = 1u << 15;  // 128 KiB

    // Grid and NVLink synchronization
    uint32_t grid_sync_count[kNumMaxGridSyncCounters];
    uint32_t nvl_barrier_counter;
    int nvl_barrier_signals[2];

    // Task scheduling: one counter per device locality domain (DeepGEMM #462); single-domain schedules use [0].
    // NOTES: the arrays end at byte 60 < 128, so every field from `combine_ready_grid_idx` on keeps the image's offset
    uint32_t l1_task_count[kNumDeviceLocalityDomains];
    uint32_t l2_task_count[kNumDeviceLocalityDomains];
    uint32_t shared_l1_task_count[kNumDeviceLocalityDomains];
    uint32_t shared_l2_task_count[kNumDeviceLocalityDomains];

    // Combine readiness: `combine_ready_grid_idx[peer] == own grid index` means the peer's L2 writes into this
    // rank are done. Dispatch clears readiness before its NVLink barrier because graph replays reuse grid IDs.
    alignas(128) uint64_t combine_ready_grid_idx[kNumMaxRanks];
    alignas(128) uint64_t peer_grid_idx[kNumMaxRanks];

    // Expert token counts
    alignas(128) uint64_t expert_send_count[kNumMaxExperts];
    uint64_t expert_recv_count[kNumMaxExperts];
    uint64_t expert_recv_count_sum[kNumMaxExpertsPerRank];

    // Routed-expert ring signals: `l2_full_mask` is one bit per L1 N block up to
    // 64 blocks, or a cumulative completion counter for wider intermediates.
    uint32_t l1_full_count[kNumMaxRingBlocks];
    uint32_t l1_empty_count[kNumMaxRingBlocks];
    uint64_t l2_full_mask[kNumMaxRingBlocks];
    uint32_t l2_empty_count[kNumMaxRingBlocks];

    // Shared-expert signals
    uint32_t shared_l2_full_count[kNumMaxSharedL2Blocks];
};
DG_STATIC_ASSERT(offsetof(MegaMoESignals, combine_ready_grid_idx) == 128, "Invalid control signal layout");

struct Workspace {
    MegaMoESignals* signals;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_topk;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        signals(static_cast<MegaMoESignals*>(base)),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank), num_topk(num_topk) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);

        DG_UNIFIED_ASSERT(num_ranks > 0);
        DG_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        DG_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        DG_UNIFIED_ASSERT(num_experts <= MegaMoESignals::kNumMaxExperts);
        DG_UNIFIED_ASSERT(num_experts_per_rank <= MegaMoESignals::kNumMaxExpertsPerRank);
        DG_UNIFIED_ASSERT(num_ring_tokens <= MegaMoESignals::kNumMaxRingBlocks * kMinCandidateBlockM);
        DG_UNIFIED_ASSERT(num_shared_l2_pool_blocks <= MegaMoESignals::kNumMaxSharedL2Blocks);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Fixed-position global and expert signals
        num_bytes += sizeof(MegaMoESignals);

        // Source token-topk: [local expert][source rank][token], with distinct experts per token
        num_bytes += num_experts * num_max_tokens_per_rank * sizeof(uint32_t);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Post-expert-norm partial sums of squares, one per (token, top-k slot). Sized
        // unconditionally so the layout does not depend on a kernel template flag.
        num_bytes += num_max_tokens_per_rank * num_topk * sizeof(float);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(signals, get_num_bytes());
    }

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < MegaMoESignals::kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return signals->grid_sync_count + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return &signals->nvl_barrier_counter;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return signals->nvl_barrier_signals + phase;
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_ready_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->combine_ready_grid_idx + peer_rank_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_peer_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->peer_grid_idx + peer_rank_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return signals->l1_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return signals->l2_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return signals->shared_l1_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return signals->shared_l2_task_count;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_send_count + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count_sum + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_full_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_empty_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_full_mask_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_full_mask + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_empty_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        return signals->shared_l2_full_count + block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        return get_l1_full_count_ptr(pool_block_idx);
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        return get_l2_full_mask_ptr(pool_block_idx);
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto offset = (expert_idx * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
        return math::advance_ptr<uint32_t>(signals, sizeof(MegaMoESignals)) + offset;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }

    // Each L2 N tile accumulates its partial sum of squares here for the combine.
    CUTLASS_DEVICE
    float* get_post_norm_sum_sq_ptr(const uint32_t& token_idx = 0, const uint32_t& topk_idx = 0) const {
        const auto base = reinterpret_cast<float*>(get_token_src_metadata_ptr(num_max_pool_tokens));
        return base + token_idx * num_topk + topk_idx;
    }
};

// SM90 MegaMoE predates the reusable ring workspace used by the SM100 kernels.
// Keep its compact layout explicit so Hopper codegen and buffer slicing stay
// identical to the tuned SM90 implementation while SM100 can use Workspace.
struct SM90Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    SM90Workspace(void* base,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += num_experts * sizeof(uint64_t) * 2;
        num_bytes += num_experts_per_rank * sizeof(uint64_t);
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);
        return math::align<uint64_t>(num_bytes, 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(
            base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer,
           combine_sf_buffer;

    // Optional per-token FP32 outer scales for the L1 (fc13) input, applied on
    // the L1 accumulator before activation.
    Buffer input_x_scales_buffer,
           l1_x_scales_buffer;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const MmaKind& mma_kind,
                  const uint32_t& num_shared_experts = 0,
                  const bool& use_fp8_combine = false) {
        const bool with_sf = get_sf_gran_k(mma_kind) != 0;
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t elem_bits = get_element_bits(mma_kind);
        // NOTE: multiply before dividing — `elem_bits / 8` truncates to 0 for the
        // 4-bit kinds (NVFP4/MXFP4), which would zero-size every activation slot.
        const auto num_token_bytes = [=](const uint32_t& num_elems) { return num_elems * elem_bits / 8; };
        const uint32_t gran_k = with_sf ? get_sf_gran_k(mma_kind) : 1;
        const auto input_token_layout = layout::Data(num_token_bytes(hidden));
        const auto combine_token_layout = layout::Data(
            use_fp8_combine ? hidden : hidden * 2);
        const auto combine_sf_layout = layout::Data(
            use_fp8_combine ? hidden / 128 : 0, false);
        const auto intermediate_token_layout = layout::Data(num_token_bytes(intermediate_hidden));
        const auto shared_intermediate_token_layout = layout::Data(num_token_bytes(shared_intermediate_hidden));
        const auto input_sf_layout = layout::Data(with_sf ? hidden / gran_k : 0, false);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / gran_k : 0, false);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / gran_k : 0, false);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);
        const auto x_scale_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            combine_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());
        combine_sf_buffer = Buffer(
            combine_sf_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            combine_token_buffer.get_end_ptr());

        input_x_scales_buffer = Buffer(
            x_scale_layout, 1, num_max_tokens_per_rank,
            combine_sf_buffer.get_end_ptr());
        l1_x_scales_buffer = Buffer(
            x_scale_layout, 1, num_ring_tokens,
            input_x_scales_buffer.get_end_ptr());
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(l1_x_scales_buffer.get_end_ptr())
               - reinterpret_cast<uint8_t*>(workspace.signals);
    }
};

} // namespace deep_gemm::layout
""",
    "include/deep_gemm/scheduler/mega_moe.cuh": r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// TODO: add MegaMoE namespace
namespace deep_gemm::sched {

// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    // When each M block has more L1 tasks than L2 tasks, the interleaved schedule
    // leaves `(num_l1_pair_n_blocks - num_l2_pair_n_blocks)` extra L1 tasks behind
    // per M block. To avoid deadlock, no L2 task for an M block should be scheduled
    // before that block's L1 tasks have been issued. The last M block is the
    // bottleneck: it needs its own `num_l1_pair_n_blocks` L1 tasks, and the preceding
    // `num_total_m_blocks - 1` M blocks accumulate
    // `(num_total_m_blocks - 1) * (num_l1_pair_n_blocks - num_l2_pair_n_blocks)`
    // pending L1 tasks, so we issue them during the warmup phase. Add one extra
    // CTA-pair wave to cover partial-wave rounding.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    // TODO: may delay combine NVLink
    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Host-side ring capacity helper for the MegaMoE schedule.
CUTLASS_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden) {
    constexpr int kMegaMoEBlockN = 128;
    constexpr int kNumCTAsPerCluster = 2;

    DG_UNIFIED_ASSERT((intermediate_hidden * 2) % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    DG_UNIFIED_ASSERT(hidden % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    const int num_clusters = num_sms / kNumCTAsPerCluster;
    const int num_l1_n_clusters = intermediate_hidden * 2 / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l2_n_clusters = hidden / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters);
    const int num_l1_warmup_waves = cute::min(num_min_l1_warmup_waves, num_l1_waves);
    const int num_l1_warmup_clusters = cute::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

    // Conservative closed-form bound for the L2/L1 alternating tail. If L1 advances faster in
    // M-block space (B1 < B2), live blocks can grow by about total_m_blocks * (B2 - B1) / B2.
    // Add one global-wave margin to cover partial waves / pipeline lag.
    // This is slightly larger than the tight bound, but cheaper and simpler.
    // TODO: refactor here
    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(
        num_clusters, cute::min(num_l1_n_clusters, num_l2_n_clusters));
    return cute::min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& pair_n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_cluster_idx(pair_n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_DEVICE
    uint32_t is_valid() const {
        return (block_phase != BlockPhase::None);
    }

    CUTLASS_DEVICE
    uint32_t get_umma_aligned_valid_m() const {
        return math::align(valid_m, 16u);
    }

    CUTLASS_DEVICE uint32_t is_shared() const {
        return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

#if defined(__CUDACC__) or defined(__CLION_IDE__)

// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K,
          bool kUseMask = (L1_SHAPE_N / BLOCK_N <= 64)>
struct L2KBlockDependency;

template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K, true> {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "Too many L1 N blocks for the L2 readiness mask");
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask = kFullMask;

    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* mask_ptr, const uint32_t& generation_idx):
        mask_ptr(mask_ptr), expected_mask(generation_idx & 1 ? 0ull : kFullMask) {}

    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

    // Re-read the mask only when the K block is not fed yet
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};

// Wider intermediates cannot fit one bit per L1 N block in the readiness word.
// Reuse that word as a cumulative completion counter and wait for the entire L1
// result, as in the original ring protocol. L2 empty counts prevent producers
// from overwriting this generation before all L2 consumers have finished.
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K, false> {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* count_ptr;
    uint64_t expected_count;
    bool ready = false;

    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* count_ptr, const uint32_t& generation_idx):
        count_ptr(count_ptr), expected_count((static_cast<uint64_t>(generation_idx) + 1) * kNumL1BlockNs) {}

    CUTLASS_DEVICE static void arrive(const uint64_t* count_ptr, const uint32_t&) {
        ptx::red_add_rel(count_ptr, uint64_t{1});
    }

    CUTLASS_DEVICE void wait(const uint32_t&) {
        if (not ready) {
            while (ptx::ld_acq_gpu(count_ptr) < expected_count) {}
            ready = true;
        }
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumLocalityDomains = 1,
          bool kFenceTaskRelease = false,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all CTA-pair schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    // pd005 (DeepGEMM #462): locality domain affinity. When a queue is drained, we try work stealing to ensure load
    // balancing. But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == layout::kNumDeviceLocalityDomains, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumL1Clusters % kNumLocalityDomains == 0 and kNumL2Clusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
    uint32_t sm_locality_domain_idx = 0;

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace):
        workspace(workspace) {}

    CUTLASS_DEVICE MegaMoEScheduler(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    CUTLASS_DEVICE void release_task_info() const {
        // DeepGEMM #441: complete metadata reads before the scheduler can overwrite this slot
        if constexpr (kFenceTaskRelease)
            asm volatile("fence.acq_rel.cta;" ::: "memory");
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array.
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs / 2, kNumL1Clusters, kNumL2Clusters);
        num_sched_l1_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                        const uint32_t& m_block_idx,
                                        const uint32_t& n_cluster_idx,
                                        const uint32_t& shape_n,
                                        const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();

        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_local_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr + locality_domain_idx, 1u);
        return ptx::exchange(result, 0);
    }

    // A domain's queue holds N clusters `[d * C / D, (d + 1) * C / D)` of every M block, M-block major
    template <uint32_t kNumClusters>
    static CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx, const uint32_t& num_m_blocks,
                                                 uint32_t& m_block_idx, uint32_t& n_cluster_idx) {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
        return true;
    }

    template <uint32_t kNumClusters>
    CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& num_m_blocks,
                                          uint32_t& m_block_idx, uint32_t& n_cluster_idx) const {
        if (get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx, num_m_blocks, m_block_idx, n_cluster_idx))
            return true;
        return kNumLocalityDomains > 1 and num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
               get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, n_cluster_idx);
    }

    CUTLASS_DEVICE task_info_t get_next_task() {
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one CTA-pair wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                if (not get_next_task_idx<kNumL1Clusters>(workspace.get_l1_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(BlockPhase::Linear1, m_block_idx, n_cluster_idx, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                if (not get_next_task_idx<kNumL2Clusters>(workspace.get_l2_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx))
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(BlockPhase::Linear2, m_block_idx, n_cluster_idx, L2_SHAPE_N, L2_SHAPE_K);

                // Wait until all required L1 tasks are fetched from all scheduling queues
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * (kNumL1Clusters / kNumLocalityDomains);
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++ i)
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr() + i) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        DG_STATIC_ASSERT(kNumNClusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            if (not get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m_block_idx, n_cluster_idx))
                break;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens, const uint8_t* sm_locality_domains = nullptr) {
        const auto lane_idx = ptx::get_lane_idx();
        sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[ptx::get_sm_idx()];

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap, else at the tail
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2;
        const bool is_shared_l2_early = math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2;
        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            if (not is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

// Hopper MegaMoE retains the original wave scheduler. SM100 uses the task
// producer/consumer scheduler above so shared experts can participate in the
// same pipeline, while SM90 keeps its tuned single-CTA scheduling contract.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K,
          typename WorkspaceT = layout::Workspace>
struct LegacyMegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerWave > 0 and kNumExpertsPerWave <= kNumExpertsPerRank, "Invalid wave config");
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even");

    const WorkspaceT& workspace;
    BlockPhase next_phase = BlockPhase::Linear1;
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit LegacyMegaMoEScheduler(const WorkspaceT& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        const auto aligned = math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
        return cute::min(aligned, kNumExpertsPerRank);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (current_local_expert_idx < kNumExpertsPerRank) {
            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                }
                next_phase = BlockPhase::Linear2;
                set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
            } else {
                if (fetch_next_l2_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                }
                next_phase = BlockPhase::Linear1;
            }
        }
        return {BlockPhase::None, 0, 0, 0};
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        fetch_expert_recv_count();
        set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, local_expert_idx, block_m_idx, block_n_idx);
            if (block_phase == BlockPhase::None)
                break;
            func(block_phase, local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 block_m_idx, block_n_idx);
        }
    }
};

#endif

} // namespace deep_gemm::sched
""",
}
