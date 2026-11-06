
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

from sglang.srt.layers.moe.mega_moe import (
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
    should_use_mega_moe,
    torch,
    vision_topk,
)

from bundle_v9 import moe_front as _v4_mega


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
    v4_front = None
    if should_use_mega_gate(moe, hidden_states, forward_batch):
        import deep_gemm

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
        if _v4_front_covered(moe, router_logits, hidden_states):
            v4_front = (moe, router_logits, input_ids_global)
            topk_output = _V4_DEFERRED_TOPK
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
        v4_front=v4_front,
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
    v4_front=None,
    decode_tiles: bool = False,
) -> torch.Tensor:
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
    elif v4_front is not None:
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

    if routed_scaling_factor != 1.0 and v4_front is None:
        assert not routing_in_buffer
        topk_weights_in = topk_weights_in * routed_scaling_factor

    mega_kwargs = {"recipe": (1, 1, 32)}
    if mma_type == "nvfp4xnvfp4":
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
            with _mega_moe_tiles_env(buf, num_tokens, decode_tiles) as decode_block_m:
                shared_block_m = deep_gemm.get_block_m_for_mega_moe(
                    num_ranks=buf.group.size(),
                    num_experts=buf.num_experts,
                    num_max_tokens_per_rank=buf.num_max_tokens_per_rank,
                    num_tokens=max(num_tokens, 1),
                    num_topk=buf.num_topk,
                    mma_type=buf.mma_type,
                    global_num_tokens=global_num_tokens,
                )
                _check_mega_moe_decode_block_m(shared_block_m, decode_block_m)
        if v4_front is _REHOME_WRITTEN:
            pass
        elif v4_front is not None:
            _v4_front(
                v4_front,
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

    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    with (
        _configure_mega_moe_deep_gemm_num_sms(deep_gemm),
        _mega_moe_tiles_env(buf, num_tokens, decode_tiles),
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


V4_MEGA_MAX_ROWS = 32768


class _DeferredTopK(NamedTuple):
    topk_ids: None = None
    topk_weights: None = None


_V4_DEFERRED_TOPK = _DeferredTopK()


def _v4_front_covered(moe, router_logits, hidden_states) -> bool:
    return _v4_mega.covered(moe, router_logits, hidden_states, V4_MEGA_MAX_ROWS)


def _v4_front(front, hidden_states, buf, routed_scaling_factor, shared_x_sf, shared_block_m) -> None:
    moe, router_logits, input_ids = front
    _v4_mega.mega_routed_front(
        moe, router_logits, input_ids, hidden_states, buf, routed_scaling_factor=routed_scaling_factor,
        shared_x_sf=shared_x_sf, shared_block_m=shared_block_m, use_pdl=False)


_REHOME_WRITTEN = object()


def rehome_front(moe, hidden_states, forward_batch, input_ids_global, *, row_offset: int, total_rows: int, pad: bool,
             live_all: bool) -> int:
    import deep_gemm

    rows = hidden_states.shape[0]
    if not (
        not _use_amd_flydsl_mega_moe()
        and should_use_mega_moe(moe, hidden_states)
        and not get_is_capture_mode()
        and moe.mega_shared_l1_weights is not None
        and _device_sm != 90
        and _mega_moe_mma_type(moe.experts) == "fp8xfp4"
        and not should_use_mega_gate(moe, hidden_states, forward_batch)
        and 0 < rows
        and 0 <= row_offset
        and row_offset + rows <= total_rows
    ):
        raise RuntimeError(f"v9-rehome: MoE layer {moe.layer_id} block [{row_offset}, {row_offset + rows}) of "
                           f"{total_rows} rows is outside the fused-shared fp8xfp4 MOE_FRONT path")
    router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
    if not _v4_front_covered(moe, router_logits, hidden_states):
        raise RuntimeError(f"v9-rehome: MOE_FRONT does not cover MoE layer {moe.layer_id} at {rows} rows")
    buf = _mega_moe_buffer(moe)
    with _mega_moe_tiles_env(buf, total_rows, False):
        shared_block_m = deep_gemm.get_block_m_for_mega_moe(
            num_ranks=buf.group.size(),
            num_experts=buf.num_experts,
            num_max_tokens_per_rank=buf.num_max_tokens_per_rank,
            num_tokens=max(total_rows, 1),
            num_topk=buf.num_topk,
            mma_type=buf.mma_type,
            global_num_tokens=_mega_moe_global_tokens_hint(buf),
        )
    _v4_mega.mega_routed_front(
        moe, router_logits, input_ids_global, hidden_states, buf,
        routed_scaling_factor=1.0 if moe.routed_scaling_factor_fused_in_topk else float(moe.routed_scaling_factor),
        shared_x_sf=buf.shared_l1_acts_sf, shared_block_m=shared_block_m, use_pdl=False, row_offset=row_offset,
        pad=pad, live_all=live_all)
    return shared_block_m


def rehome_experts(moe, hidden_states, total_rows: int) -> torch.Tensor:
    return run_mega_routed_experts(
        moe.experts,
        hidden_states,
        None,
        None,
        hidden_size=moe.config.hidden_size,
        intermediate_size=moe.config.moe_intermediate_size,
        top_k=moe.config.num_experts_per_tok + moe.num_fused_shared_experts,
        num_tokens=total_rows,
        activation_clamp=moe.experts.moe_runner_config.swiglu_limit,
        routed_scaling_factor=1.0 if moe.routed_scaling_factor_fused_in_topk else float(moe.routed_scaling_factor),
        num_shared_experts=0 if moe.mega_shared_l1_weights is None else moe.n_shared_experts,
        shared_l1_weights=moe.mega_shared_l1_weights,
        shared_l2_weights=moe.mega_shared_l2_weights,
        routing_in_buffer=False,
        v4_front=_REHOME_WRITTEN,
        decode_tiles=False,
    )
