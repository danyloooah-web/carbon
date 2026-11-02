"""dsv41-pd065 M462: the cold prefill forward's MegaMoE on DeepGEMM #462's FP8xFP4 prefill pipeline items, ported onto
the image's own kernel (generator/pd060; the port is generator/pd002/megasched/dgms, cuda_m462.py holds its sources).

Inside the candidate's own extend forwards (`routed`, next to FMLAP's; never while a CUDA graph captures, and never in a
breakable-graph or replay-tail scope) `deep_gemm.fp8_fp4_mega_moe` -- the function
dsv41_pd001.mega_forward.run_mega_routed_experts looks up per call -- runs the image kernel's source (sgl-deep-gemm
0.2.3.post2 sm100_fp8_fp4_mega_moe) with #462's items for BLOCK_M 240: L1 / L2 store blocks 24 / 8 instead of 40 / 40,
which frees shared memory for a 6th TMA stage, #462's epilogue (TMEM loads of the next store block issued ahead, the
accumulator released after the last valid block) and one prefetch stage. They regroup epilogue rows and reorder waits
only: every output element keeps its K-ordered MMA chain, per-row epilogue and slot-order combine, so y is the image
kernel's bytes (generator/pd060 bench: bitwise on every rank at 8K-32K rows a rank, recorded and uniform routing).
Measured at the 1,100 W cap with call-level pairing: +2.0% a call at 12K rows a rank, +1.6% at 16K, +1.2% at 32K.

The host library (the image's DeepGEMM host sources plus the knobs, TORCH_LIBRARY namespace pd065_m462) is built at
prepare with torch's cpp_extension (ninja, the image's c++) under SGLang's JIT cache root, and its DeepJIT builds the
one kernel config the domain serves there too (`compile_only`), so no served forward compiles. A PD decode engine
builds nothing.

Domain, decided per call: prepared; an fp8xfp4 buffer with the prepared facts (EP ranks, experts, token cap, top-k,
hidden / intermediate sizes, one shared expert); swiglu with the prepared clamp, recipe (1, 1, 32), fast math, the
megamoe weight layout and no alphas, x scales, post-expert norm or receive stats; no DG_MEGA_MOE_TILES override (the
tile bands keep the image kernel); the prepared MegaMoE SM count; a DP-global token hint (#46093; every EP rank sees
the same one, so every rank takes the same decision for the collective kernel) of >= MIN_GLOBAL_TOKENS; and the image
heuristic's BLOCK_M 240 for the call. Every other call runs the image function unchanged; inside the domain the
kernel runs or raises. CHECK (the twin, never submitted) also runs the image kernel on the first calls of each token
bucket, on the same dispatched inputs (restored between the two launches), and compares every output byte under
"dsv41-pd065chk"; it prints no timings."""

from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache

import torch

CHECK = False  # gen_pd060 check: the twin's bitwise comparisons (measurement only, never submit)
CHECK_CALLS = 3  # per DP-global token bucket
PREFIX = "dsv41-pd065chk" if CHECK else "dsv41-pd065"
BLOCK_M = 240
# 4 x 12,288: the smallest measured DP-global size (+2.0% a call); the gain was measured from there to 4 x 32,768
MIN_GLOBAL_TOKENS = 49152
HIDDEN, INTER, EXPERTS = 5120, 2304, 384
# dgms knobs: store_split, store_l1, store_l2, prefetch, locality, epi462, regs462, shared_first, fence (+ num_sms):
# #462's store split (BLOCK_M 240 -> 24 / 8), its epilogue and its prefetch rule (BLOCK_M 240 -> 1 stage)
KNOBS = (1, 0, 0, -1, 0, 1, 0, 0, 0)

_STOCK: dict = {}
_FACTS: dict = {}
_ON = [False]
_PDL = [None]
_CHK: dict = {}


# ---- the host library ------------------------------------------------------------------------------------------------
def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial file (as fmlap._write)."""
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
def source_root() -> str:
    """Every host source under <JIT cache root>/cacheon-dsv41-pd065/<TAG>/, written on first use."""
    from sglang.kernels.jit.utils.compile.cache import cache_root

    from dsv41_pd001 import cuda_m462

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd065", cuda_m462.TAG)
    for rel, text in cuda_m462.SOURCES.items():
        path = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            _write(path, text.encode())
    os.makedirs(os.path.join(directory, "build"), exist_ok=True)
    return directory


@lru_cache(maxsize=None)
def ops():
    """Build and load the host library (once a process), and point its DeepJIT at the patched kernel headers (this
    tree's include/) ahead of the image's DeepGEMM include tree."""
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import load as build_library

    from dsv41_pd001 import cuda_m462

    root = source_root()
    cuda_home = CUDA_HOME or "/usr/local/cuda"
    image_include = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    includes = [os.path.join(root, "include"), image_include, os.path.join(root, "csrc"),
                os.path.join(root, "build_include"), os.path.join(cuda_home, "include")]
    cccl = os.path.join(cuda_home, "include", "cccl")
    if os.path.isdir(cccl):
        includes.append(cccl)
    # with_cuda=False: one C++ source, no nvcc (cpp_extension's CUDA mode would derive arch flags from a visible GPU);
    # the CUDA runtime and torch's CUDA libraries are linked explicitly
    build_library(name=f"pd065_m462_{cuda_m462.TAG}", sources=[os.path.join(root, cuda_m462.ENTRY)],
                  extra_cflags=list(cuda_m462.CXX_FLAGS),
                  extra_ldflags=[f"-L{os.path.join(cuda_home, 'lib64')}", "-lcudart", "-lc10_cuda", "-ltorch_cuda",
                                 *cuda_m462.LD_FLAGS],
                  extra_include_paths=includes, build_directory=os.path.join(root, "build"), with_cuda=False,
                  is_python_module=False, verbose=False)
    lib = torch.ops.pd065_m462
    lib.init(root, image_include)
    return lib


# ---- the replacement -------------------------------------------------------------------------------------------------
def _graph_scope() -> bool:
    """A CUDA graph captures, or a breakable-prefill / replay-tail graph scope is open: the image kernel runs."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.models.deepseek_v4_replay_tail import in_replay_tail_graph

    return torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph() or in_replay_tail_graph()


def _global_tokens(global_num_tokens) -> int:
    return int(global_num_tokens) if global_num_tokens is not None else 0


def serves(y, l1_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
           recipe, activation, activation_clamp, fast_math, use_x_scales, l1_alphas, l2_alphas, l2_act_scales,
           weight_layout, global_num_tokens, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps,
           post_expert_norm_weight_first) -> bool:
    """The domain (module docstring); cheap tests first, the image heuristic's BLOCK_M last."""
    if not (_ON[0] and _FACTS):
        return False
    f = _FACTS
    if not (
        activation == "swiglu"
        and tuple(recipe) == (1, 1, 32)
        and fast_math is True
        and not use_x_scales
        and l1_alphas is None
        and l2_alphas is None
        and l2_act_scales is None
        and weight_layout == "megamoe"
        and not post_expert_norm
        and post_expert_norm_weight is None
        and post_expert_norm_eps == 1e-6
        and not post_expert_norm_weight_first
        and cumulative_local_expert_recv_stats is None
        and shared_l1_weights is not None
        and shared_l2_weights is not None
        and activation_clamp is not None
        and float(activation_clamp) == f["clamp"]
        and getattr(sym_buffer, "mma_type", None) == "fp8xfp4"
        and getattr(sym_buffer, "activation", None) == "swiglu"
        and sym_buffer.num_experts == f["experts"]
        and sym_buffer.num_max_tokens_per_rank == f["cap"]
        and sym_buffer.num_topk == f["topk"]
        and sym_buffer.hidden == f["hidden"]
        and sym_buffer.intermediate_hidden == f["inter"]
        and sym_buffer.num_shared_experts == f["shared"]
        and sym_buffer.group.size() == f["ranks"]
        and l1_weights[0].dtype == f["dtype"]
        and y.dtype == torch.bfloat16
        and y.dim() == 2
        and y.shape[1] == f["hidden"]
        and global_num_tokens is not None
        and _global_tokens(global_num_tokens) >= MIN_GLOBAL_TOKENS
        and "DG_MEGA_MOE_TILES" not in os.environ
    ):
        return False
    import deep_gemm

    if deep_gemm.get_num_sms() != f["num_sms"] or _graph_scope():
        return False
    block_m = deep_gemm.get_block_m_for_mega_moe(
        num_ranks=f["ranks"], num_experts=f["experts"], num_max_tokens_per_rank=f["cap"], num_tokens=y.shape[0],
        num_topk=f["topk"], mma_type="fp8xfp4", global_num_tokens=global_num_tokens)
    return block_m == BLOCK_M


def _launch(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, activation_clamp,
            global_num_tokens) -> None:
    import deep_gemm

    lib = ops()
    pdl = bool(deep_gemm.get_pdl())
    if _PDL[0] != pdl:
        lib.set_pdl(pdl)
        _PDL[0] = pdl
    (l1_w, l1_sf), (l2_w, l2_sf) = l1_weights, l2_weights
    (s1_w, s1_sf), (s2_w, s2_sf) = shared_l1_weights, shared_l2_weights
    lib.fp8_fp4_mega_moe(y, l1_w, l1_sf, l2_w, l2_sf, s1_w, s1_sf, s2_w, s2_sf, None, sym_buffer.buffer,
                         sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
                         sym_buffer.num_experts, sym_buffer.num_topk, [1, 1, 32], "fp8xfp4", "swiglu",
                         float(activation_clamp), True, global_num_tokens, list(KNOBS) + [_FACTS["num_sms"]], None)


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    """Differing bytes of two equally shaped tensors (-1 when shapes or dtypes differ), in 16 MiB slices (a
    whole-tensor bool sum on ~1G elements ran a twin out of memory, chain_k55 k55t)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    x, y = a.contiguous().view(torch.uint8).reshape(-1), b.contiguous().view(torch.uint8).reshape(-1)
    step = 1 << 24
    return sum(int((x[i:i + step] != y[i:i + step]).sum()) for i in range(0, x.numel(), step))


def _checked(y, sym_buffer, global_num_tokens, run_stock, run_ours) -> bool:
    """Twin: on the first CHECK_CALLS calls of a DP-global token bucket (the same decision on every EP rank) run the
    image kernel into a reference, restore the dispatched inputs it read (x, x_sf, top-k ids / weights, the shared
    expert's SF pages), run ours and compare every output byte. False when the bucket is already checked."""
    g = _global_tokens(global_num_tokens)
    key = max(g, 1).bit_length()
    n = _CHK.get(key, 0)
    if n >= CHECK_CALLS:
        return False
    _CHK[key] = n + 1
    rows = y.shape[0]
    b = sym_buffer
    inputs = [b.x[:rows], b.x_sf[:rows], b.topk_idx[:rows], b.topk_weights[:rows]]
    if b.shared_l1_acts_sf.numel():
        inputs.append(b.shared_l1_acts_sf)
    saved = [t.clone() for t in inputs]
    ref = torch.empty_like(y)
    run_stock(ref)
    for t, s in zip(inputs, saved):
        t.copy_(s)
    run_ours(y)
    diff = _differ(y, ref)
    print(f"{PREFIX} M462 {y.device} rows {rows} global {g}: "
          f"{'OK' if diff == 0 else 'MISMATCH'} bytes-differ {diff}", file=sys.stderr, flush=True)
    return True


def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights=None, shared_l2_weights=None,
                     cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu",
                     activation_clamp=None, fast_math=True, use_x_scales=False, l1_alphas=None, l2_alphas=None,
                     l2_act_scales=None, weight_layout="megamoe", global_num_tokens=None, post_expert_norm=False,
                     post_expert_norm_weight=None, post_expert_norm_eps=1e-6, post_expert_norm_weight_first=False):
    """deep_gemm.fp8_fp4_mega_moe (writes y), M462 inside the domain."""
    stock = _STOCK["fp8_fp4_mega_moe"]

    def run_stock(out):
        stock(out, l1_weights, l2_weights, sym_buffer, shared_l1_weights=shared_l1_weights,
              shared_l2_weights=shared_l2_weights,
              cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats, recipe=recipe,
              activation=activation, activation_clamp=activation_clamp, fast_math=fast_math,
              use_x_scales=use_x_scales, l1_alphas=l1_alphas, l2_alphas=l2_alphas, l2_act_scales=l2_act_scales,
              weight_layout=weight_layout, global_num_tokens=global_num_tokens, post_expert_norm=post_expert_norm,
              post_expert_norm_weight=post_expert_norm_weight, post_expert_norm_eps=post_expert_norm_eps,
              post_expert_norm_weight_first=post_expert_norm_weight_first)

    if not serves(y, l1_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
                  recipe, activation, activation_clamp, fast_math, use_x_scales, l1_alphas, l2_alphas, l2_act_scales,
                  weight_layout, global_num_tokens, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps,
                  post_expert_norm_weight_first):
        return run_stock(y)

    def run_ours(out):
        _launch(out, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, activation_clamp,
                global_num_tokens)

    if CHECK and _checked(y, sym_buffer, global_num_tokens, run_stock, run_ours):
        return None
    return run_ours(y)


# ---- binding ---------------------------------------------------------------------------------------------------------
def prepare_facts(facts: dict) -> tuple:
    """Build the host library and DeepJIT-compile the one kernel config the domain serves, for these facts (keys as
    _FACTS); returns the compiled (block_m, store L1, store L2, prefetch stages, stages, smem). Raises unless that is
    #462's BLOCK_M 240 config (24 / 8, one prefetch stage, 6 stages)."""
    lib = ops()
    got = tuple(int(v) for v in lib.compile_only(
        list(KNOBS) + [facts["num_sms"]], facts["ranks"], facts["experts"], facts["cap"], facts["topk"],
        facts["hidden"], facts["inter"], facts["shared"], facts["weight_ref"], facts["clamp"], True,
        facts["cap"], facts["cap"] * facts["ranks"]))
    if got[:5] != (BLOCK_M, 24, 8, 1, 6):
        raise RuntimeError(f"dsv41-pd065 M462: compiled config {got} is not #462's BLOCK_M 240 one")
    _FACTS.clear()
    _FACTS.update({k: v for k, v in facts.items() if k != "weight_ref"}, dtype=facts["weight_ref"].dtype)
    return got


def prepare(model) -> None:
    """Capture the image's deep_gemm.fp8_fp4_mega_moe, read the facts every MoE layer shares (raising if they differ
    or fall outside the domain) and build + compile now (a build failure fails prepare). A PD decode engine runs no
    eligible forward and builds nothing."""
    import deep_gemm
    from sglang.srt.layers.moe import mega_moe as stock_mega
    from sglang.srt.runtime_context import get_disagg, get_parallel

    if get_disagg().disaggregation_mode == "decode":
        print(f"{PREFIX} M462: PD decode engine, not prepared", file=sys.stderr, flush=True)
        return
    if not _STOCK:
        fn = deep_gemm.fp8_fp4_mega_moe
        if fn.__module__ != "deep_gemm.mega" or fn.__qualname__ != "fp8_fp4_mega_moe":
            raise RuntimeError(f"dsv41-pd065 M462: deep_gemm.fp8_fp4_mega_moe is not the image's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK["fp8_fp4_mega_moe"] = fn
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(f"dsv41-pd065 M462 is built and proven for sm_103 (B300), got "
                           f"{torch.cuda.get_device_capability()}")
    seen = set()
    weight_ref = None
    for index, layer in enumerate(model.layers):
        moe = layer.mlp
        experts = moe.experts
        if stock_mega._mega_moe_mma_type(experts) != "fp8xfp4" or moe.mega_shared_l1_weights is None:
            raise RuntimeError(f"dsv41-pd065 M462: layer {index} is not an fp8xfp4 MegaMoE with a fused shared expert")
        weight_ref = experts.mega_l1_weights[0]
        seen.add((moe.config.num_experts_per_tok + moe.num_fused_shared_experts, moe.n_shared_experts,
                  moe.config.hidden_size, moe.config.moe_intermediate_size, experts.num_experts,
                  float(experts.moe_runner_config.swiglu_limit), weight_ref.dtype))
    if len(seen) != 1:
        raise RuntimeError(f"dsv41-pd065 M462: MoE layers differ: {sorted(map(str, seen))}")
    topk, shared, hidden, inter, experts_n, clamp, _dtype = seen.pop()
    if (hidden, inter, experts_n, shared) != (HIDDEN, INTER, EXPERTS, 1):
        raise RuntimeError(f"dsv41-pd065 M462: MoE facts {(hidden, inter, experts_n, shared)} are not DeepSeek-V4.1-Flash's")
    lib = ops()
    align = int(lib.get_token_alignment_for_mega_moe())
    cap = stock_mega.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    cap = (cap + align - 1) // align * align
    with stock_mega._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        num_sms = int(deep_gemm.get_num_sms())
    ranks = get_parallel().moe_ep_group.device_group.size()
    got = prepare_facts({"ranks": ranks, "experts": experts_n, "cap": cap, "topk": topk, "hidden": hidden,
                         "inter": inter, "shared": shared, "clamp": clamp, "num_sms": num_sms,
                         "weight_ref": weight_ref})
    # the scope test imports SGLang's breakable-graph context and replay-tail modules on first use (~230 ms in the
    # prodmod check, 10-30): do it now, not in the first served call
    _graph_scope()
    print(f"{PREFIX} M462 prepared: ranks {ranks} cap {cap} topk {topk} sms {num_sms} clamp {clamp} config {got}",
          file=sys.stderr, flush=True)


@contextlib.contextmanager
def routed(view, forward_batch):
    """One extend forward (no speculative modes, not capturing): the image's deep_gemm.fp8_fp4_mega_moe swapped for
    fp8_fp4_mega_moe and restored on exit; every other forward runs the image's untouched."""
    mode = getattr(forward_batch, "forward_mode", None)
    if (mode is None or not mode.is_extend_without_speculative() or torch.cuda.is_current_stream_capturing()
            or not _STOCK or not _FACTS):
        yield
        return
    import deep_gemm

    _ON[0] = True
    try:
        deep_gemm.fp8_fp4_mega_moe = fp8_fp4_mega_moe
        yield
    finally:
        _ON[0] = False
        deep_gemm.fp8_fp4_mega_moe = _STOCK["fp8_fp4_mega_moe"]
