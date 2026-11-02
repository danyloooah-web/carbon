"""dsv41-pd001 MEGA_FRONT: forward_mega_moe, _run_mega_routed and run_mega_routed_experts copied from the
sealed sglang/srt/layers/moe/mega_moe.py (sha256 2251a72090d99353; forward_mega_moe lines 288-331, _run_mega_routed lines 420-496, run_mega_routed_experts lines 499-679) by generator/pd001/mega/
gen_mega_forward.py. The only edits (marked dsv41-pd001 MEGA_FRONT) replace vision_topk, the routed-scale
multiply and mega_moe_pre_dispatch with dsv41_pd001.mega_front.mega_routed_front, which writes the same
buffer bytes in one launch. Regenerate from the sealed source; never hand-edit."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

from sglang.srt.layers.moe.mega_moe import (  # noqa: F401  (the copied bodies' module names)
    _check_mega_moe_decode_block_m,
    _configure_mega_moe_deep_gemm_num_sms,
    _device_sm,
    _get_mega_moe_symm_buffer,
    _mega_moe_buffer,
    _mega_moe_global_tokens_hint,
    _mega_moe_input_in_buffer,
    _mega_moe_mma_type,
    _mega_moe_tiles_env,
    _use_amd_flydsl_mega_moe,
    _use_mega_moe_decode_tiles,
    envs,
    get_is_capture_mode,
    mega_moe_pre_dispatch,
    nullcontext,
    run_mega_gate,
    run_sm90_mega_routed,
    should_use_mega_gate,
    torch,
    vision_topk,
)

from dsv41_pd001 import mega_front as _pd001_mega


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _use_amd_flydsl_mega_moe():
        from sglang.srt.layers.moe.mega_moe_flydsl import (
            forward_mega_moe as forward_flydsl_mega_moe,
        )

        return forward_flydsl_mega_moe(
            moe, hidden_states, forward_batch, input_ids_global
        )

    num_tokens = hidden_states.shape[0]
    fused_shared = moe.mega_shared_l1_weights is not None
    fork_mega_moe = (
        moe.alt_stream is not None
        and (fused_shared or moe.num_fused_shared_experts == 0)
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if fork_mega_moe:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        mega_stream_ctx = nullcontext()

    shared_output = None if fused_shared else moe._forward_shared_experts(hidden_states)

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if fork_mega_moe:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    hidden_size = moe.config.hidden_size

    routing = None
    pd001_front = None  # dsv41-pd001 MEGA_FRONT: (moe, router_logits, input_ids) routed by the fused launch
    if should_use_mega_gate(moe, hidden_states, forward_batch):
        import deep_gemm

        # Only the fp8xfp4 pre-dispatch can skip its top-k copy (kCopyTopk).
        buf = (
            _mega_moe_buffer(moe)
            if _mega_moe_mma_type(moe.experts) == "fp8xfp4"
            else None
        )
        with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
            routing = run_mega_gate(
                moe, hidden_states, input_ids_global, buf, forward_batch
            )

    if routing is not None:
        topk_ids = routing.topk_ids
        topk_weights = routing.topk_weights
        routed_scaling_factor = routing.routed_scaling_factor
    elif num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        if _pd001_front_covered(moe, router_logits, hidden_states):  # dsv41-pd001 MEGA_FRONT
            pd001_front = (moe, router_logits, input_ids_global)
            topk_output = _PD001_DEFERRED_TOPK
        elif (
            not moe.is_hash
            and getattr(moe.gate, "e_score_correction_bias_vl", None) is not None
        ):
            topk_output = vision_topk(
                moe,
                router_logits,
                input_ids_global,
            )
        else:
            topk_output = moe.topk(
                hidden_states,
                router_logits,
                **topk_kwargs,
            )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None
    if routing is None:
        routed_scaling_factor = (
            1.0
            if moe.routed_scaling_factor_fused_in_topk
            else float(moe.routed_scaling_factor)
        )

    return run_mega_routed_experts(
        moe.experts,
        hidden_states,
        topk_ids,
        topk_weights,
        hidden_size=hidden_size,
        intermediate_size=moe.config.moe_intermediate_size,
        top_k=moe.config.num_experts_per_tok + moe.num_fused_shared_experts,
        num_tokens=num_tokens,
        activation_clamp=moe.experts.moe_runner_config.swiglu_limit,
        routed_scaling_factor=routed_scaling_factor,
        num_shared_experts=(
            0 if moe.mega_shared_l1_weights is None else moe.n_shared_experts
        ),
        shared_l1_weights=moe.mega_shared_l1_weights,
        shared_l2_weights=moe.mega_shared_l2_weights,
        routing_in_buffer=routing is not None and routing.in_buffer,
        pd001_front=pd001_front,  # dsv41-pd001 MEGA_FRONT
        decode_tiles=_use_mega_moe_decode_tiles(forward_batch),
    )


def run_mega_routed_experts(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    activation_clamp: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
    activation: str = "swiglu",
    num_shared_experts: int = 0,
    shared_l1_weights=None,
    shared_l2_weights=None,
    routing_in_buffer: bool = False,
    pd001_front=None,  # dsv41-pd001 MEGA_FRONT
    decode_tiles: bool = False,
) -> torch.Tensor:
    # Rows are this rank's tokens; the returned rows are fully combined.
    import deep_gemm

    from sglang.srt.runtime_context import get_parallel

    ep_group = get_parallel().moe_ep_group.device_group
    num_experts = experts.num_experts
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or lower "
        f"--cuda-graph-max-bs-decode / --chunked-prefill-size accordingly"
    )

    mma_type = _mega_moe_mma_type(experts)
    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
        num_shared_experts=num_shared_experts,
        mma_type=mma_type,
        activation=activation,
    )

    if routing_in_buffer:
        assert num_tokens > 0 and _device_sm != 90 and mma_type == "fp8xfp4"
        topk_ids_in = topk_weights_in = None
    elif pd001_front is not None:  # dsv41-pd001 MEGA_FRONT: routed by the fused launch below
        topk_ids_in = topk_weights_in = None
    elif num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    if _device_sm == 90:
        return run_sm90_mega_routed(
            experts,
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf,
            num_tokens,
            hidden_size=hidden_size,
            activation_clamp=activation_clamp,
            routed_scaling_factor=routed_scaling_factor,
        )

    if routed_scaling_factor != 1.0 and pd001_front is None:  # dsv41-pd001 MEGA_FRONT folds the scale
        assert not routing_in_buffer
        topk_weights_in = topk_weights_in * routed_scaling_factor

    mega_kwargs = {"recipe": (1, 1, 32)}
    if mma_type == "nvfp4xnvfp4":
        # Per-token outer scales go to buf.x_scales; the kernel folds them with
        # the per-expert alphas in the L1 / L2 epilogues.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=16,
            mma_type=mma_type,
            buf_x_scales=buf.x_scales,
        )
        mega_kwargs = {
            "recipe": (1, 1, 16),
            "use_x_scales": True,
            "l1_alphas": experts.mega_l1_alphas,
            "l2_alphas": experts.mega_l2_alphas,
            "l2_act_scales": experts.mega_l2_act_scales,
        }
        if experts._mega_moe_trtllm_layout:
            mega_kwargs["weight_layout"] = "trtllm"
    elif mma_type == "mxf4xmxf4":
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            mma_type=mma_type,
        )
    else:
        global_num_tokens = _mega_moe_global_tokens_hint(buf)
        if global_num_tokens is not None:
            mega_kwargs["global_num_tokens"] = global_num_tokens
        shared_block_m = 0
        if shared_l1_weights is not None:
            with _pd040_tiles_env(buf, num_tokens, decode_tiles) as decode_block_m:  # dsv41-pd040 TILES
                shared_block_m = deep_gemm.get_block_m_for_mega_moe(
                    num_ranks=buf.group.size(),
                    num_experts=buf.num_experts,
                    # DeepGEMM rounds the requested capacity up for its buffer.
                    num_max_tokens_per_rank=buf.num_max_tokens_per_rank,
                    # Match the non-null output allocation passed to DeepGEMM.
                    num_tokens=max(num_tokens, 1),
                    num_topk=buf.num_topk,
                    mma_type=buf.mma_type,
                    global_num_tokens=global_num_tokens,
                )
                _check_mega_moe_decode_block_m(shared_block_m, decode_block_m)
        if pd001_front is not None:  # dsv41-pd001 MEGA_FRONT: router + scale + pre_dispatch, one launch
            _pd001_front(
                pd001_front,
                hidden_states,
                buf,
                routed_scaling_factor,
                buf.shared_l1_acts_sf if shared_l1_weights is not None else None,
                shared_block_m,
            )
        elif not (
            routing_in_buffer
            and _mega_moe_input_in_buffer(hidden_states, buf, num_tokens)
        ):
            mega_moe_pre_dispatch(
                hidden_states,
                buf.topk_idx if routing_in_buffer else topk_ids_in,
                buf.topk_weights if routing_in_buffer else topk_weights_in,
                buf.x,
                buf.x_sf,
                buf.topk_idx,
                buf.topk_weights,
                quant_group_size=32,
                shared_x_sf=(
                    buf.shared_l1_acts_sf if shared_l1_weights is not None else None
                ),
                shared_block_m=shared_block_m,
                copy_topk=not routing_in_buffer,
            )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    with (
        _configure_mega_moe_deep_gemm_num_sms(deep_gemm),
        _pd040_tiles_env(buf, num_tokens, decode_tiles),  # dsv41-pd040 TILES
    ):
        deep_gemm.fp8_fp4_mega_moe(
            y,
            experts.mega_l1_weights,
            experts.mega_l2_weights,
            buf,
            shared_l1_weights=shared_l1_weights,
            shared_l2_weights=shared_l2_weights,
            activation=activation,
            activation_clamp=activation_clamp,
            fast_math=True,
            **mega_kwargs,
        )
    return y[:num_tokens]


# ---- dsv41-pd001 MEGA_FRONT helpers ---------------------------------------------------------------------------
# Bitwise == stock at 1..32768 rows (generator/pd001/mega/NOTES.md) and faster at every one: the bound is MegaMoE's
# per-rank prefill cap, not a speed crossover.
PD001_MEGA_MAX_ROWS = 32768


class _DeferredTopK(NamedTuple):
    topk_ids: None = None
    topk_weights: None = None


_PD001_DEFERRED_TOPK = _DeferredTopK()


def _pd001_front_covered(moe, router_logits, hidden_states) -> bool:
    return _pd001_mega.covered(moe, router_logits, hidden_states, PD001_MEGA_MAX_ROWS)


def _pd001_front(front, hidden_states, buf, routed_scaling_factor, shared_x_sf, shared_block_m) -> None:
    moe, router_logits, input_ids = front
    # No PDL: in the forked decode chain a plain launch was as fast or faster at every verify row count but
    # 1-4 and 24-48 (by <= 0.3 us), and up to 1.6 us faster at 8-16 rows (tiny-GEMM gate).
    _pd001_mega.mega_routed_front(
        moe, router_logits, input_ids, hidden_states, buf, routed_scaling_factor=routed_scaling_factor,
        shared_x_sf=shared_x_sf, shared_block_m=shared_block_m, use_pdl=False)


# ---- dsv41-pd040 TILES -------------------------------------------------------------------------------------------
# DeepGEMM MegaMoE tiles for decode / verify graph calls (generator/pd001/tiles/tables_v3.json, dec2): every listed
# config gave y bitwise equal to stock's selection on EP4 (one ordered K chain per element, per-row epilogue, fixed-order
# slot combine) and was faster at its rows. "cluster,BLOCK_M,STORE_BLOCK_M,BLOCK_K,epilogue_warpgroups"; rows outside
# the table, a None entry and every eager call keep stock's #46593 selection.
import os as _pd040_os  # noqa: E402
from contextlib import contextmanager as _pd040_contextmanager  # noqa: E402

_PD040_GRAPH_TILES = ((4, None), (48, '2,16,8,128,2'), (200, '2,32,8,128,2'))


@_pd040_contextmanager
def _pd040_tiles_env(buf, num_tokens, decode_tiles):
    """Stock's _mega_moe_tiles_env with pd040's decode table: yields the BLOCK_M it forces, or stock's own yield."""
    tiles = None
    if decode_tiles and "DG_MEGA_MOE_TILES" not in _pd040_os.environ:  # an operator override wins, as in stock
        tiles = next((t for max_rows, t in _PD040_GRAPH_TILES if num_tokens <= max_rows), None)
    if tiles is None:
        with _mega_moe_tiles_env(buf, num_tokens, decode_tiles) as block_m:
            yield block_m
        return
    _pd040_os.environ["DG_MEGA_MOE_TILES"] = tiles
    try:
        yield int(tiles.split(",")[1])
    finally:
        _pd040_os.environ.pop("DG_MEGA_MOE_TILES", None)
