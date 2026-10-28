"""model.layers.*.mlp, lever N1: decode / target-verify MegaMoE routing through DeepGEMM's fused bf16 mega gate.

Stock (this arena leaves SGLANG_OPT_DEEPGEMM_MEGA_GATE off) routes a MegaMoE layer with the bf16 gate GEMM (fp32
logits), the Triton top-k router and a pre-dispatch kernel that copies the picks into the MegaMoE symmetric buffer. In
decode and target-verify forwards (1..2048 rows) this op takes the engine's own mega-gate branch of the same routed
path: one bf16_mega_gate launch scores the 384 experts (sqrtsoftplus, correction and image bias, top-6, renormalize,
routed scale) and writes the picks straight into the buffer, so pre-dispatch skips its top-k copy. The gate's valid-row
and image masks are built once per forward (cache reset at the first MoE layer), as the engine's forward scope does.

The bodies below are the engine's forward_mega_moe and _run_mega_routed with that branch taken; their source digests
are pinned and a mismatch, or any other mode or shape, runs the module class's stock forward. No routed expert is ever
dropped: the gate keeps all six picks. Precision class: tolerance (the fused gate's GEMM order differs from cuBLAS, so a
near-tied sixth pick can differ), graded by the eager audit's decode calls.

Lever N2t (prefill role): with decoder SWA bounded replay, layers 20-39 of a prefill forward run on each request's last
<= 128 tokens from row-bucket CUDA graphs (128-row steps up to 2048 rows per rank, forward mode EXTEND; every DP rank
replays the largest rank's bucket). #46593 raises DeepGEMM's MegaMoE BLOCK_M only for decode and target-verify graph
calls, so these tail graphs keep the raw heuristic, BLOCK_M 16 / 32 at <= ~384 rows per rank. While our node warms up or
captures a tail graph, N2t scopes the engine's own raise for exactly those two cases (DG_MEGA_MOE_TILES 2,64,16,128,2)
around the stock forward. Bit-identical: the 1-GPU sweep (real layer-20 experts, 18 configs, every tail bucket) found
every config equal to the default bit for bit. Speed: measured on the 8-GPU pair only for decode-graph calls (2,64,16
0.272 ms vs raw 0.377 ms at 288 rows per rank); a single GPU cannot show it (no cross-rank dispatch). The 64 -> 96 raise
is not applied (unmeasured on 8 GPUs, +3.8 % at 512 rows on one). Each config is a pure function of the call's rows and
the symmetric buffer's shape, so every rank of a bucket's collective capture (one bucket's rows on every rank) takes the
same one. The tail scope also covers eager tails inside a prefill-graph break, where ranks keep their own rows; there the
config can differ per rank exactly as stock's BLOCK_M already does (no global-token hint in capture mode), which the
kernel allows (BLOCK_M shapes each rank's own receive pool), and the output stays bit-identical. This arena turns
breakable prefill graphs off for DeepSeek-V4 without prefill context parallelism, so that path is not reached."""
import inspect
import os
from contextlib import nullcontext

import torch

from dspd.common import pins_match, say
from sglang.srt.layers.moe import mega_gate as MG
from sglang.srt.layers.moe import mega_moe as MM
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import get_forward

try:
    from sglang.srt.models.deepseek_v4_replay_tail import in_replay_tail_graph
except ImportError:                                   # an engine without decoder tail graphs: N2t stays off
    in_replay_tail_graph = None

ENABLED = True
MAX_ROWS = 2048
# N2t off in production: its 2,64,16 MegaMoE kernel JIT-compiles (~2 s, measured cold) at the first lazy tail-graph capture,
# i.e. inside a scored window, which costs about what N2t saves per window (0.1-0.5 s); the code stays for a precompiled run
TAIL_TILES = False
# N2t: DeepGEMM's raw MegaMoE BLOCK_M for a tail bucket -> the engine's decode-graph raise (mega_moe.py
# _MEGA_MOE_DECODE_BLOCK_M 16 / 32 -> (64, 16)) as a DG_MEGA_MOE_TILES config
TAIL_RAISE = {16: "2,64,16,128,2", 32: "2,64,16,128,2"}
# run_mega_routed_experts computes the pre-dispatch shared BLOCK_M and launches the kernel under the scoped env; the
# tiles scope steps aside while the env is set; the buffer whose shape the raw heuristic reads; the tail flag
TAIL_PINS = {"run_mega_routed_experts": "809828b1c3cc6de6", "_mega_moe_tiles_env": "db0d498b2cba2ec3",
             "_mega_moe_buffer": "9dabc8eb5d53f8d5", "_mega_moe_global_tokens_hint": "8636c40efda0d1b4",
             "in_replay_tail_graph": "c44962b8ff23f4cd"}
# N2d: decode-graph MegaMoE tiles by this forward's rows per rank (every DP rank replays the same bucket, so the
# collective agrees): (lower bound, DG_MEGA_MOE_TILES). Measured on real decode batches (dspd-tilerig, eager, all
# configs bit-equal): at 288 rows 2,128,32 = 0.243 ms vs the engine's 2,64,16 0.272 ms; below ~270 rows the engine's
# choice is the best measured, so it stays.
DECODE_TILES = ((270, "2,128,32,128,2"),)
# _run_mega_routed: 70c7016a = nightly 5884e4fe + main d3a157a2 + #46093/#46593 (decode tiles), d67fbac8 = the plain
# nightly; the body below is the same routing either way, and decode_tiles is passed only where the engine takes it
PINS = {"forward_mega_moe": "55b0fc1607126ded", "_run_mega_routed": frozenset({"70c7016ac2dd4ece", "d67fbac89061dfa4"}),
        "run_mega_gate": "af03d2ee4f77d802", "_gate_row_masks": "716b4fa04c4da0fb"}
_g = {"pins": None, "first": None, "tiles": None, "tail_pins": None, "tail_cfg": {}}


class _State:
    __slots__ = ("module", "stock")

    def __init__(self, module):
        self.module = module
        self.stock = type(module).forward


def prepare(module):
    lid = getattr(module, "layer_id", None)
    if isinstance(lid, int) and (_g["first"] is None or lid < _g["first"]):
        _g["first"] = lid
    return _State(module)


def _pins_ok():
    if _g["pins"] is None:
        _g["tiles"] = ("decode_tiles" in inspect.signature(MM.run_mega_routed_experts).parameters
                       and hasattr(MM, "_use_mega_moe_decode_tiles"))
        _g["pins"] = pins_match("mlp", PINS, {"forward_mega_moe": MM.forward_mega_moe,
                                              "_run_mega_routed": MM._run_mega_routed,
                                              "run_mega_gate": MG.run_mega_gate,
                                              "_gate_row_masks": MG._gate_row_masks})
    return _g["pins"]


def _gate_ok(moe, hidden_states, forward_batch):
    if not ENABLED or forward_batch is None:
        return False
    mode = forward_batch.forward_mode
    return ((mode.is_decode() or mode.is_target_verify())
            and 0 < hidden_states.shape[0] <= MAX_ROWS
            and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
            and not get_forward().sp_active
            and moe.mega_gate_supported
            and moe.mega_shared_l1_weights is not None
            and MM._mega_moe_mma_type(moe.experts) == "fp8xfp4"
            and MM.should_use_mega_moe(moe, hidden_states)
            and _pins_ok())


def _tail_pins_ok():
    if _g["tail_pins"] is None:
        # a name this engine lacks reads as a pin mismatch (stock forward), never an AttributeError in the forward
        fns = {"run_mega_routed_experts": getattr(MM, "run_mega_routed_experts", None),
               "_mega_moe_tiles_env": getattr(MM, "_mega_moe_tiles_env", None),
               "_mega_moe_buffer": getattr(MM, "_mega_moe_buffer", None),
               "_mega_moe_global_tokens_hint": getattr(MM, "_mega_moe_global_tokens_hint", None),
               "in_replay_tail_graph": in_replay_tail_graph}
        _g["tail_pins"] = (all(fn is not None for fn in fns.values()) and in_replay_tail_graph is not None
                           and pins_match("mlp-tail", TAIL_PINS, fns))
    return _g["tail_pins"]


def _tail_cfg(moe, hidden_states, forward_batch):
    """N2t's DG_MEGA_MOE_TILES for this call, or None (not a tail-graph EXTEND MegaMoE call, or no raise for its rows)."""
    if (not TAIL_TILES or forward_batch is None or in_replay_tail_graph is None or not in_replay_tail_graph()
            or "DG_MEGA_MOE_TILES" in os.environ):
        return None
    mode = forward_batch.forward_mode
    rows = hidden_states.shape[0]
    if mode.is_decode() or mode.is_target_verify() or not 0 < rows <= MAX_ROWS or not _tail_pins_ok():
        return None
    if not (moe.mega_shared_l1_weights is not None and MM._mega_moe_mma_type(moe.experts) == "fp8xfp4"
            and MM.should_use_mega_moe(moe, hidden_states)):
        return None
    buf = MM._mega_moe_buffer(moe)
    key = (rows, buf.group.size(), buf.num_experts, buf.num_max_tokens_per_rank, buf.num_topk)
    if key not in _g["tail_cfg"]:
        import deep_gemm

        # the raw heuristic as stock's capture computes it (no global-token hint under capture)
        raw = deep_gemm.get_block_m_for_mega_moe(num_ranks=buf.group.size(), num_experts=buf.num_experts,
                                                 num_max_tokens_per_rank=buf.num_max_tokens_per_rank,
                                                 num_tokens=rows, num_topk=buf.num_topk, mma_type=buf.mma_type)
        _g["tail_cfg"][key] = TAIL_RAISE.get(raw)
    return _g["tail_cfg"][key]


def forward(state, hidden_states, forward_batch=None, gemm_output_zero_allocator=None, input_ids=None,
            input_ids_global=None, skip_shared_experts=False, return_moe_output=False, leave_output_sum=False):
    moe = state.module
    if forward_batch is not None and moe.layer_id == _g["first"]:
        try:
            # a new forward: the gate masks of the last one are stale (the cache attribute _gate_row_masks reads)
            setattr(forward_batch, "_mega_gate_row_masks", {})
        except (AttributeError, TypeError):
            pass
    if _gate_ok(moe, hidden_states, forward_batch):
        say("mlp", "N1 mega gate on (decode / target-verify)")
        return _mega_moe(moe, hidden_states, forward_batch, input_ids_global)
    cfg = _tail_cfg(moe, hidden_states, forward_batch)
    if cfg is None:
        return state.stock(moe, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids, input_ids_global,
                           skip_shared_experts, return_moe_output, leave_output_sum)
    say("mlp", f"N2t tail-graph tiles {cfg} at {hidden_states.shape[0]} rows")
    os.environ["DG_MEGA_MOE_TILES"] = cfg      # stock's tiles scope steps aside while this is set (as for N2d)
    try:
        return state.stock(moe, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids, input_ids_global,
                           skip_shared_experts, return_moe_output, leave_output_sum)
    finally:
        os.environ.pop("DG_MEGA_MOE_TILES", None)


def _mega_moe(moe, hidden_states, forward_batch, input_ids_global):
    """forward_mega_moe (CUDA path) with _routed below."""
    num_tokens = hidden_states.shape[0]
    fused_shared = moe.mega_shared_l1_weights is not None
    fork = (moe.alt_stream is not None and (fused_shared or moe.num_fused_shared_experts == 0)
            and num_tokens > 0 and get_is_capture_mode())
    if fork:
        main = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(main)
        ctx = torch.cuda.stream(moe.alt_stream)
    else:
        ctx = nullcontext()
    shared = None if fused_shared else moe._forward_shared_experts(hidden_states)
    with ctx:
        y = _routed(moe, hidden_states, forward_batch, input_ids_global, num_tokens)
    if fork:
        main.wait_stream(moe.alt_stream)
    if shared is not None:
        y.add_(shared)
    return y


def _routed(moe, hidden_states, forward_batch, input_ids_global, num_tokens):
    """_run_mega_routed with the mega-gate branch taken (fp8xfp4, so the picks land in the buffer)."""
    import deep_gemm

    buf = MM._mega_moe_buffer(moe)
    with MM._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        routing = MG.run_mega_gate(moe, hidden_states, input_ids_global, buf, forward_batch)
    if routing is not None:
        topk_ids, topk_weights = routing.topk_ids, routing.topk_weights
        routed_scaling_factor = routing.routed_scaling_factor
    else:
        # The gate declines (EPLB lp / fake dispatch, a hash layer without ids): stock routing, as stock does.
        say("mlp", "mega gate declined on a call: stock gate + top-k for it")
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        if not moe.is_hash and getattr(moe.gate, "e_score_correction_bias_vl", None) is not None:
            topk_output = vision_topk(moe, router_logits, input_ids_global)
        else:
            topk_output = moe.topk(hidden_states, router_logits, **topk_kwargs)
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        routed_scaling_factor = 1.0 if moe.routed_scaling_factor_fused_in_topk else float(moe.routed_scaling_factor)
    tiles = {"decode_tiles": MM._use_mega_moe_decode_tiles(forward_batch)} if _g["tiles"] else {}
    cfg = None
    if get_is_capture_mode() and "DG_MEGA_MOE_TILES" not in os.environ:
        for lower, value in DECODE_TILES:
            if num_tokens >= lower:
                cfg = value
    if cfg is None:
        return _launch(moe, hidden_states, topk_ids, topk_weights, num_tokens, routed_scaling_factor, routing, tiles)
    say("mlp", f"N2d decode tiles {cfg} from {num_tokens} rows")
    os.environ["DG_MEGA_MOE_TILES"] = cfg      # the engine's own decode-tile scope steps aside while this is set
    try:
        return _launch(moe, hidden_states, topk_ids, topk_weights, num_tokens, routed_scaling_factor, routing, tiles)
    finally:
        os.environ.pop("DG_MEGA_MOE_TILES", None)


def _launch(moe, hidden_states, topk_ids, topk_weights, num_tokens, routed_scaling_factor, routing, tiles):
    return MM.run_mega_routed_experts(
        moe.experts,
        hidden_states,
        topk_ids,
        topk_weights,
        hidden_size=moe.config.hidden_size,
        intermediate_size=moe.config.moe_intermediate_size,
        top_k=moe.config.num_experts_per_tok + moe.num_fused_shared_experts,
        num_tokens=num_tokens,
        activation_clamp=moe.experts.moe_runner_config.swiglu_limit,
        routed_scaling_factor=routed_scaling_factor,
        num_shared_experts=0 if moe.mega_shared_l1_weights is None else moe.n_shared_experts,
        shared_l1_weights=moe.mega_shared_l1_weights,
        shared_l2_weights=moe.mega_shared_l2_weights,
        routing_in_buffer=routing is not None and routing.in_buffer,
        **tiles,
    )
