
from __future__ import annotations

from sglang.srt.models.deepseek_v2 import (
    BumpAllocator,
    ForwardBatch,
    MergedMoEOutput,
    MoEOutput,
    Optional,
    Union,
    _hip_moe,
    _is_hip,
    dsv2_flashinfer_moe_dual_stream_graph,
    dsv4_flashinfer_moe_dual_stream_outputs,
    get_forward,
    get_is_capture_mode,
    is_in_breakable_cuda_graph,
    torch,
)


def forward(
    self,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    gemm_output_zero_allocator: BumpAllocator = None,
    input_ids: Optional[torch.Tensor] = None,
    input_ids_global: Optional[torch.Tensor] = None,
    skip_shared_experts: bool = False,
    return_moe_output: bool = False,
    leave_output_sum: bool = False,
) -> Union[torch.Tensor, MoEOutput]:
    from sglang.srt.layers.moe.mega_moe import should_use_mega_moe

    from bundle_v9.moe_route import forward_mega_moe

    if should_use_mega_moe(self, hidden_states):
        return forward_mega_moe(
            self,
            hidden_states,
            forward_batch,
            input_ids_global=input_ids_global,
        )

    use_vision_topk = self.gate.e_score_correction_bias_vl is not None
    if use_vision_topk and _is_hip:
        use_vision_topk = _hip_moe.batch_has_images(forward_batch)
    if not self._enable_a2a_moe:
        if self._can_dual_stream_graph(hidden_states):
            fwd = get_forward()
            if leave_output_sum:
                routed, shared = dsv4_flashinfer_moe_dual_stream_outputs(
                    hidden_states,
                    self.layer_id,
                    fwd.mlp_reduce_scatter,
                    self._shared_expert_tp1,
                )
                return MergedMoEOutput(
                    routed,
                    shared if self._shared_expert_tp1 else None,
                    self.experts,
                    1.0,
                    self._shared_expert_tp1,
                )
            return dsv2_flashinfer_moe_dual_stream_graph(
                hidden_states,
                self.layer_id,
                fwd.mlp_reduce_scatter,
            )
        elif (
            self.alt_stream is not None
            and self.num_fused_shared_experts == 0
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
            and not is_in_breakable_cuda_graph()
        ):
            return self.forward_normal_dual_stream(
                hidden_states,
                gemm_output_zero_allocator,
                input_ids,
                input_ids_global=input_ids_global,
                use_vision_topk=use_vision_topk,
                return_moe_output=return_moe_output,
                leave_output_sum=leave_output_sum,
            )
        else:
            return self.forward_normal(
                hidden_states,
                gemm_output_zero_allocator,
                input_ids,
                input_ids_global=input_ids_global,
                skip_shared_experts=skip_shared_experts,
                use_vision_topk=use_vision_topk,
                return_moe_output=return_moe_output,
                leave_output_sum=leave_output_sum,
            )
    else:
        return self.forward_deepep(
            hidden_states, forward_batch, input_ids_global=input_ids_global
        )
