from __future__ import annotations

import contextlib
import os
import sys

import torch

MAX_ROWS = 288
KNOBS = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0)
KNOBS_CDISP = KNOBS[:12] + (1,)
CDISP_MAX_ROWS = 96
# king-v5 DPF (dec_moe I1 as corrected by VERIFY_dec_moe, the DPF-K hook): the decode MegaMoE's TMA-B warp issues
# an L2 prefetch of the first routed L1 wave (local slots 0-3) at kernel entry, before the PDL wait, so those bytes
# stream from HBM in the dispatch window instead of the main phase. The FP8 shared expert is not prefetched. A
# prefetch is a cache hint: no value, barrier or count changes, so the output bytes cannot change. Rank-local: a
# rank with DPF off runs the king's kernel under the same protocol, so no vote is needed.
DPF = True                  # False: the king_v2b MegaMoE configs only, compiled and served exactly as before
DPF_BYTES = 50_200_000      # L2 prefetch bytes a layer a rank, in whole routed L1 tasks (one task = 2 x 128 rows x
                            # hidden/2 B of FP4 weights + 2 x 128 x hidden/128 x 4 B of UE8M0 scales = 696,320 B):
                            # 72 tasks = 50,135,040 B = local slots 0-3 (18 n-clusters each)
DPF_MIN_ROWS = 25           # rows a rank from which the DPF configs serve (below, local slots 0-3 are often empty)

_STOCK: dict = {}
_FACTS: dict = {}
_ON = [False]
_PDL = [None]
_DPF = [0]                  # routed L1 tasks the served DPF configs prefetch; 0 = DPF off (set once by prepare)


def serves(y, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats, recipe, activation,
           activation_clamp, fast_math, use_x_scales, l1_alphas, l2_alphas, l2_act_scales, weight_layout, post_expert_norm,
           post_expert_norm_weight, post_expert_norm_eps, post_expert_norm_weight_first) -> bool:
    if not (_ON[0] and _FACTS):
        return False
    f = _FACTS
    return (
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
        and y.dtype == torch.bfloat16
        and y.dim() == 2
        and y.shape[1] == f["hidden"]
        and y.shape[0] <= MAX_ROWS
    )


def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights=None, shared_l2_weights=None,
                     cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu",
                     activation_clamp=None, fast_math=True, use_x_scales=False, l1_alphas=None, l2_alphas=None,
                     l2_act_scales=None, weight_layout="megamoe", global_num_tokens=None, post_expert_norm=False,
                     post_expert_norm_weight=None, post_expert_norm_eps=1e-6, post_expert_norm_weight_first=False):
    if not serves(y, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats, recipe, activation,
                  activation_clamp, fast_math, use_x_scales, l1_alphas, l2_alphas, l2_act_scales, weight_layout,
                  post_expert_norm, post_expert_norm_weight, post_expert_norm_eps, post_expert_norm_weight_first):
        return _STOCK["fp8_fp4_mega_moe"](
            y, l1_weights, l2_weights, sym_buffer, shared_l1_weights=shared_l1_weights,
            shared_l2_weights=shared_l2_weights, cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            recipe=recipe, activation=activation, activation_clamp=activation_clamp, fast_math=fast_math,
            use_x_scales=use_x_scales, l1_alphas=l1_alphas, l2_alphas=l2_alphas, l2_act_scales=l2_act_scales,
            weight_layout=weight_layout, global_num_tokens=global_num_tokens, post_expert_norm=post_expert_norm,
            post_expert_norm_weight=post_expert_norm_weight, post_expert_norm_eps=post_expert_norm_eps,
            post_expert_norm_weight_first=post_expert_norm_weight_first)
    import deep_gemm

    from bundle_v9.dec import moe_prefill

    lib = moe_prefill.ops()
    pdl = bool(deep_gemm.get_pdl())
    if _PDL[0] != pdl:
        lib.set_pdl(pdl)
        _PDL[0] = pdl
    (l1_w, l1_sf), (l2_w, l2_sf) = l1_weights, l2_weights
    (s1_w, s1_sf), (s2_w, s2_sf) = shared_l1_weights, shared_l2_weights
    lib.fp8_fp4_mega_moe(y, l1_w, l1_sf, l2_w, l2_sf, s1_w, s1_sf, s2_w, s2_sf, None, sym_buffer.buffer,
                         sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
                         sym_buffer.num_experts, sym_buffer.num_topk, [1, 1, 32], "fp8xfp4", "swiglu",
                         float(activation_clamp), True, global_num_tokens, _knobs(y.shape[0]), None)
    return None


def _knobs(rows: int) -> list:
    """The launch knobs: the king's, plus DPF's task count as the 15th knob when DPF is on and rows >= DPF_MIN_ROWS."""
    knobs = list(KNOBS_CDISP if rows <= CDISP_MAX_ROWS else KNOBS) + [_FACTS["num_sms"]]
    if _DPF[0] > 0 and rows >= DPF_MIN_ROWS:
        knobs.append(_DPF[0])
    return knobs


def _bands() -> list:
    """(highest rows, tiles, knobs, lowest rows) of every band the king prebuilds, in the king's order."""
    from bundle_v9.dec.moe_route import _GRAPH_TILES

    bands = []
    low = 0
    prev = 0
    for rows, tiles in _GRAPH_TILES:
        first, prev = prev + 1, rows
        if tiles is None:
            continue
        if low < CDISP_MAX_ROWS:
            bands.append((min(rows, CDISP_MAX_ROWS), tiles, KNOBS_CDISP, first))
        if rows > CDISP_MAX_ROWS:
            bands.append((rows, tiles, KNOBS, max(first, CDISP_MAX_ROWS + 1)))
        low = rows
    return bands


def _compile_one(lib, facts, rows, tiles, knobs) -> tuple:
    os.environ["DG_MEGA_MOE_TILES"] = tiles
    try:
        return tuple(int(v) for v in lib.compile_only(
            list(knobs), facts["ranks"], facts["experts"], facts["cap"], facts["topk"],
            facts["hidden"], facts["inter"], facts["shared"], facts["weight_ref"], facts["clamp"], True,
            rows, rows * facts["ranks"]))
    finally:
        os.environ.pop("DG_MEGA_MOE_TILES", None)


def _compile_bands(lib, facts) -> list:
    return [_compile_one(lib, facts, rows, tiles, list(knobs) + [facts["num_sms"]])
            for rows, tiles, knobs, _ in _bands()]


def _dpf_tasks(facts) -> tuple:
    """(tasks, bytes, reason): DPF_BYTES in whole routed L1 tasks of this model, capped at a rank's L1 tasks."""
    if not DPF:
        return 0, 0, "DPF = False"
    hidden, inter = facts["hidden"], facts["inter"]
    if hidden % 256 or (2 * inter) % 256:
        return 0, 0, f"hidden {hidden} / intermediate {inter} are not whole 128-row / 128-K blocks"
    per_task = 2 * 128 * (hidden // 2) + 2 * 128 * (hidden // 128) * 4
    limit = facts["experts"] // facts["ranks"] * (2 * inter // 256)
    tasks = min(int(DPF_BYTES) // per_task, limit)
    if tasks <= 0:
        return 0, 0, f"DPF_BYTES {DPF_BYTES} is below one routed L1 task ({per_task} B)"
    return tasks, tasks * per_task, ""


def _dpf_layout(model, hidden) -> str:
    """'' when every layer's routed L1 weights and scales have the layout the DPF prefetch addresses (dense FP4
    rows of hidden / 2 bytes, 4-byte scale words, 16-byte aligned bases), else the first miss. The C++ launch
    checks the same on every call and hands the kernel no base (no prefetch) on a miss."""
    for index, layer in enumerate(model.layers):
        w, sf = layer.mlp.experts.mega_l1_weights[:2]
        if not (w.is_contiguous() and w.dim() >= 2 and w.stride(-2) * w.element_size() == hidden // 2
                and sf.element_size() == 4 and w.data_ptr() % 16 == 0 and sf.data_ptr() % 16 == 0):
            return (f"layer {index}'s routed L1 weights {tuple(w.shape)} {w.dtype} stride {tuple(w.stride())} / "
                    f"scales {sf.dtype} are not the dense layout the prefetch addresses")
    return ""


def _prepare_dpf(lib, facts, king, layout) -> str:
    """Compile the DPF variant of every band that serves rows >= DPF_MIN_ROWS; DPF serves only if the weight
    layout is the one it addresses (`layout` == '') and each variant compiled with the king's config (tile, stages,
    shared memory) for that band. The king's configs are already built, so any failure leaves DPF off and the
    king's kernel serving every call."""
    _DPF[0] = 0
    tasks, nbytes, why = _dpf_tasks(facts)
    if tasks and layout:
        why = layout
    if not tasks or layout:
        return f"bundle-v9 DPF off: {why}; the king's MegaMoE kernel serves every call"
    got = []
    for (rows, tiles, knobs, first), ref in zip(_bands(), king):
        if rows < DPF_MIN_ROWS:
            continue
        try:
            cfg = _compile_one(lib, facts, rows, tiles, list(knobs) + [facts["num_sms"], tasks])
        except Exception as e:  # noqa: BLE001  (a DPF build failure only turns DPF off)
            return (f"bundle-v9 DPF off: the DPF build of band {max(first, DPF_MIN_ROWS)}-{rows} ({tiles}) raised "
                    f"({type(e).__name__}: {str(e)[:200]}); the king's MegaMoE kernel serves every call")
        if cfg != ref:
            return (f"bundle-v9 DPF off: band {max(first, DPF_MIN_ROWS)}-{rows} DPF config {cfg} is not the king's "
                    f"{ref}; the king's MegaMoE kernel serves every call")
        got.append((max(first, DPF_MIN_ROWS), rows, tiles))
    if not got:
        return f"bundle-v9 DPF off: no band serves rows >= {DPF_MIN_ROWS}; the king's MegaMoE kernel serves every call"
    _DPF[0] = tasks
    slots = tasks / (2 * facts["inter"] // 256)
    return (f"bundle-v9 DPF ON: {tasks} routed L1 tasks a layer ({nbytes} B, local slots 0-{slots - 1:g}) "
            f"L2-prefetched at MegaMoE entry at >= {DPF_MIN_ROWS} rows; bands {got} (heuristic configs = the king's)")


def _compile_bands_evl(lib, facts) -> list:
    """king-v2e EVL: compile the bands from the EVL kernel headers (evl.py); on any failure, from the king's."""
    import deep_gemm

    from bundle_v9.dec import evl, moe_prefill

    if not evl.EVL_ON:
        evl.mark(False, "switched off by evl.EVL_ON; the king's kernel text")
        return _compile_bands(lib, facts)
    image_include = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    try:
        root = evl.jit_root()
        lib.init(root, image_include)
        got = _compile_bands(lib, facts)
    except Exception as error:
        evl.mark(False, f"FAIL, the king's kernel text instead: {type(error).__name__}: {str(error)[:600]}")
        lib.init(moe_prefill.source_root(), image_include)
        return _compile_bands(lib, facts)
    evl.mark(True, f"routed-expert weight + scale TMA loads evict-first, shared expert unchanged; JIT root {root}")
    return got


def prepare(model) -> None:
    import deep_gemm
    from sglang.srt.layers.moe import mega_moe as stock_mega
    from sglang.srt.runtime_context import get_disagg, get_parallel

    from bundle_v9.dec import moe_prefill

    if get_disagg().disaggregation_mode == "prefill":
        print("bundle-v9 MOE_DECODE: PD prefill engine, not prepared", file=sys.stderr, flush=True)
        return
    if not _STOCK:
        fn = deep_gemm.fp8_fp4_mega_moe
        if fn.__module__ != "deep_gemm.mega" or fn.__qualname__ != "fp8_fp4_mega_moe":
            raise RuntimeError(f"bundle-v9 MOE_DECODE: deep_gemm.fp8_fp4_mega_moe is not the image's ({fn.__module__}.{fn.__qualname__})")
        _STOCK["fp8_fp4_mega_moe"] = fn
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(f"bundle-v9 MOE_DECODE is built for sm_103 (B300), got {torch.cuda.get_device_capability()}")
    seen = set()
    weight_ref = None
    for index, layer in enumerate(model.layers):
        moe = layer.mlp
        experts = moe.experts
        if stock_mega._mega_moe_mma_type(experts) != "fp8xfp4" or moe.mega_shared_l1_weights is None:
            raise RuntimeError(f"bundle-v9 MOE_DECODE: layer {index} is not an fp8xfp4 MegaMoE with a fused shared expert")
        weight_ref = experts.mega_l1_weights[0]
        seen.add((moe.config.num_experts_per_tok + moe.num_fused_shared_experts, moe.n_shared_experts,
                  moe.config.hidden_size, moe.config.moe_intermediate_size, experts.num_experts,
                  float(experts.moe_runner_config.swiglu_limit)))
    if len(seen) != 1:
        raise RuntimeError(f"bundle-v9 MOE_DECODE: MoE layers differ: {sorted(map(str, seen))}")
    topk, shared, hidden, inter, experts_n, clamp = seen.pop()
    if (hidden, inter, experts_n, shared) != (moe_prefill.HIDDEN, moe_prefill.INTER, moe_prefill.EXPERTS, 1):
        raise RuntimeError(f"bundle-v9 MOE_DECODE: MoE facts {(hidden, inter, experts_n, shared)} are not DeepSeek-V4.1-Flash's")
    lib = moe_prefill.ops()
    align = int(lib.get_token_alignment_for_mega_moe())
    cap = stock_mega.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    cap = (cap + align - 1) // align * align
    with stock_mega._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        num_sms = int(deep_gemm.get_num_sms())
    ranks = get_parallel().moe_ep_group.device_group.size()
    facts = {"ranks": ranks, "experts": experts_n, "cap": cap, "topk": topk, "hidden": hidden, "inter": inter, "shared": shared,
             "clamp": clamp, "num_sms": num_sms, "weight_ref": weight_ref}
    # king-v7 merge: EVL picks the JIT root the bands compile from (the king's kernel text while evl.EVL_ON is False,
    # or on any EVL failure); DPF then compiles its variants on that same root and compares them with these configs.
    # Both kernel texts derive from cuda_moe_prefill.SOURCES (king-v5's, with the kDPFTasks code), so EVL's anchors
    # still match once and an EVL root carries DPF's prefetch too. Each lever keeps its own switch and boot line.
    got = _compile_bands_evl(lib, facts)  # king-v2e EVL; the king's _compile_bands on any failure or when off
    try:  # king-v5 DPF: never raises into prepare; off on any failure (the configs above are built and serve)
        dpf_line = _prepare_dpf(lib, facts, got, _dpf_layout(model, hidden) if DPF else "")
    except Exception as e:  # noqa: BLE001
        _DPF[0] = 0
        dpf_line = (f"bundle-v9 DPF off: prepare raised ({type(e).__name__}: {str(e)[:200]}); the king's MegaMoE "
                    f"kernel serves every call")
    _FACTS.clear()
    _FACTS.update({k: v for k, v in facts.items() if k != "weight_ref"})
    print(f"bundle-v9 MOE_DECODE prepared: ranks {ranks} cap {cap} topk {topk} sms {num_sms} clamp {clamp} configs {got}",
          file=sys.stderr, flush=True)
    from bundle_v9.dec import evl

    print(evl.line(), file=sys.stderr, flush=True)
    print(dpf_line, file=sys.stderr, flush=True)


@contextlib.contextmanager
def routed(view, forward_batch):
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not (mode.is_decode() or mode.is_target_verify()) or not _STOCK or not _FACTS:
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
