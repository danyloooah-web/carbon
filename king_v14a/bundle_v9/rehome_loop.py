
from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.models.deepseek_v4 import (
    AuxHiddenStatePacker,
    Backend,
    Optional,
    Phase,
    Tuple,
    _gather_late_layer_ids,
    _is_cuda,
    _is_hip,
    _low_ratio_sources_break,
    bcg_deepseek_v4_engram_hash_ids,
    bcg_deepseek_v4_enter_late_layer_tail,
    bcg_deepseek_v4_exit_late_layer_tail,
    bcg_deepseek_v4_write_tail_source_kv,
    check_cuda_graph_backend,
    enter_late_layer_dp,
    exit_late_layer_dp,
    forward_layer_fused_boundary,
    fused_boundary_attn_input,
    get_attn_backend,
    get_forward,
    get_global_expert_distribution_recorder,
    get_moe_a2a_backend,
    get_parallel,
    hc_boundary,
    is_cp_active,
    is_in_breakable_cuda_graph,
    late_layer_dp_counts,
    mhc,
    nullcontext,
    residual_batch,
    torch,
)

if TYPE_CHECKING:
    ForwardBatch = object
    LateLayerTail = object


from bundle_v9 import rehome_block


def forward_layers(
    self,
    positions,
    hidden_states,
    forward_batch,
    input_ids,
    input_ids_global,
    capture_dspark,
    dspark_aux_hidden_states,
):
    layer_input = [hidden_states]
    del hidden_states
    run = rehome_block.begin(self, forward_batch, positions, input_ids, input_ids_global, capture_dspark)
    if run is None:
        return rehome_block.stock_loop()(
            self, positions, layer_input.pop(), forward_batch, input_ids, input_ids_global, capture_dspark,
            dspark_aux_hidden_states)
    try:
        return _forward_layers_hc_pre_from_prev(
            self, positions, layer_input.pop(), forward_batch, input_ids, input_ids_global, capture_dspark,
            dspark_aux_hidden_states)
    finally:
        rehome_block.end(run)


def _forward_layers_hc_pre_from_prev(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_ids: torch.Tensor,
    input_ids_global: torch.Tensor,
    capture_dspark: bool,
    dspark_aux_hidden_states: Optional[AuxHiddenStatePacker],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[LateLayerTail]]:
    assert self.pp_group.world_size == 1, "pre-mix hand-off across PP is not wired"
    hash_ids = None
    cp_extend = (
        is_cp_active(forward_batch) and forward_batch.forward_mode.is_extend()
    )
    if self.engram_hasher is not None:
        if cp_extend:
            total = int(forward_batch.attn_cp_metadata.total_seq_lens)
            hash_ids = self.engram_hasher(
                forward_batch.input_ids[:total], forward_batch
            )
            parallel = get_parallel()
            hash_ids = hash_ids[parallel.attn_cp_rank :: parallel.attn_cp_size]
            pad_rows = hidden_states.shape[0] - hash_ids.shape[0]
            if pad_rows > 0:
                hash_ids = torch.cat(
                    [hash_ids, hash_ids.new_zeros(pad_rows, *hash_ids.shape[1:])]
                )
        elif (
            forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph()
        ):
            hash_ids = self.engram_hasher.new_hash_ids(input_ids)
            bcg_deepseek_v4_engram_hash_ids(self.engram_hasher, input_ids, hash_ids)
        else:
            hash_ids = self.engram_hasher(input_ids, forward_batch)
    tail = None
    if (
        self.late_layer_start is not None
        and forward_batch.forward_mode.is_extend_without_speculative()
    ):
        attn_backend = get_attn_backend()
        tail = attn_backend.tail_forward_metadata.late_layer_tail
        self._check_late_layer_tail_readers(forward_batch, tail)
    tail_in_graph = (
        tail is not None
        and tail.scatter_indices is not None
        and is_in_breakable_cuda_graph()
    )
    late_replay_counts = None
    saved_dp = None
    if (
        self.late_layer_start is not None
        and get_parallel().attn_dp_size > 1
        and (forward_batch.is_extend_in_batch or tail_in_graph or _is_hip)
    ):
        late_replay_counts = late_layer_dp_counts(
            forward_batch,
            (
                tail.token_indices.shape[0]
                if tail is not None
                else hidden_states.shape[0]
            ),
            in_graph=tail_in_graph,
        )
    use_tail_graph, tail_graph_dp_rows = False, None
    if self.decoder_replay_tail is not None:
        use_tail_graph, tail_graph_dp_rows = self.decoder_replay_tail.plan(
            tail,
            forward_batch=forward_batch,
            num_tokens=input_ids.shape[0],
            tail_in_graph=tail_in_graph,
            late_dp_counts=late_replay_counts,
        )
    tail_sentinel = positions.new_zeros((1,)) if tail_in_graph else None
    saved_full = None
    assert not get_forward().sp_active
    state = mhc.HcState(hidden_states)
    if _is_cuda:
        residual_batch.start(forward_batch)
        hidden_states = residual_batch.set_written(state, forward_batch)
    rehome = rehome_block.active()
    state, hidden_states = rehome.enter(self, state, hidden_states, tail)
    for i in range(self.start_layer, self.end_layer):
        if _is_cuda:
            layer = self.layers[i]
            if (
                (
                    (tail is not None or use_tail_graph)
                    and i == self.late_layer_start
                )
                or layer.engram is not None
                or (capture_dspark and i in self.dspark_layers_to_capture)
            ):
                call = hc_boundary.LayerCall(
                    input_ids,
                    input_ids_global,
                )
                next_boundary = (
                    hc_boundary.HcRead(layer, True)
                    .bind(forward_batch, call)
                    .next_boundary
                )
                hidden_states = state = hc_boundary.materialize(
                    hidden_states, forward_batch, next=next_boundary
                )
        rehome_tail = rehome.leave(self, i, state, tail) if i == self.late_layer_start else None
        if tail is not None and i == self.late_layer_start:
            if self.tail_writes_source_kv:
                source = self.layers[i]
                if _is_hip and source.hc_boundary_fused:
                    source_x = fused_boundary_attn_input(source, state)
                else:
                    source_x = mhc.combine(source.attn_hc, state) if rehome_tail is None else rehome_tail[0]
                if _low_ratio_sources_break(attn_backend, forward_batch):
                    bcg_deepseek_v4_write_tail_source_kv(
                        source.self_attn, source_x, positions
                    )
                else:
                    attn_backend.write_tail_source_kv(
                        layer=source.self_attn,
                        x=source_x,
                        positions=positions,
                        forward_batch=forward_batch,
                    )
            if tail_in_graph:
                tail_sentinel = bcg_deepseek_v4_enter_late_layer_tail(tail_sentinel)
                saved_full = attn_backend.prefill_graph_late_layer_saved
                if (
                    self.decoder_replay_tail is not None
                    and self.decoder_replay_tail.breaks_out_of_prefill_graph(
                        tail, num_tokens=input_ids.shape[0]
                    )
                ):
                    residual, pre, aux = (
                        self.decoder_replay_tail.run_in_prefill_graph(
                            state=state,
                            positions=positions,
                            input_ids=input_ids,
                            input_ids_global=input_ids_global,
                            capture_dspark=capture_dspark,
                            block_rows=tail.token_indices.shape[0],
                            source_kv_layer_id=(
                                self.late_layer_start
                                if self.tail_writes_source_kv
                                else None
                            ),
                        )
                    )
                    if capture_dspark:
                        dspark_aux_hidden_states.extend(aux)
                    residual_batch.clear(forward_batch)
                    bcg_deepseek_v4_exit_late_layer_tail(tail_sentinel)
                    return residual, pre, tail
            else:
                saved_full = attn_backend.enter_late_layer_tail(forward_batch)
            state = state.take_rows(tail.rows) if rehome_tail is None else rehome_tail[1]
            if _is_cuda:
                hidden_states = residual_batch.set_written(state, forward_batch)
            input_ids = tail.rows(input_ids)
            if not (
                get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none()
            ):
                input_ids_global = tail.rows(input_ids_global)
            positions = tail.positions
            if hash_ids is not None:
                hash_ids = tail.rows(hash_ids)
            if use_tail_graph:
                residual, pre, aux = self.decoder_replay_tail.run(
                    state=state,
                    positions=positions,
                    input_ids=input_ids,
                    input_ids_global=input_ids_global,
                    forward_batch=forward_batch,
                    capture_dspark=capture_dspark,
                    dp_rows=tail_graph_dp_rows,
                )
                if capture_dspark:
                    dspark_aux_hidden_states.extend(aux)
                residual_batch.clear(forward_batch)
                attn_backend.exit_late_layer_tail(saved_full, forward_batch)
                return residual, pre, tail
        if use_tail_graph and tail is None and i == self.late_layer_start:
            residual, pre, aux = self.decoder_replay_tail.run_idle(
                state=state,
                positions=positions,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                forward_batch=forward_batch,
                capture_dspark=capture_dspark,
                dp_rows=tail_graph_dp_rows,
            )
            if capture_dspark:
                dspark_aux_hidden_states.extend(aux)
            residual_batch.clear(forward_batch)
            return residual, pre, None
        if late_replay_counts is not None and i == self.late_layer_start:
            saved_dp = enter_late_layer_dp(forward_batch, late_replay_counts)
            if get_moe_a2a_backend().is_none():
                input_ids_global = (
                    _gather_late_layer_ids(input_ids, late_replay_counts, forward_batch)
                    if self.late_layer_vl_routing
                    else None
                )
        engram = self.layers[i].engram
        if engram is not None:
            hidden_states = rehome.engram(
                self.layers[i],
                state.residual,
                hash_ids[:, engram.layer_hash_index],
                input_ids=input_ids,
                cp_all_tokens=cp_extend,
            )
            state = state.with_residual(hidden_states)
            if _is_cuda:
                hidden_states = residual_batch.set_written(state, forward_batch)
        if capture_dspark and i in self.dspark_layers_to_capture:
            from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean

            aux = state.residual
            if tail is not None and i < self.late_layer_start:
                aux = tail.rows(aux)
            mhc_stream_mean(aux, out=dspark_aux_hidden_states.slot(aux[:, 0]))
            del aux
        ctx = (
            nullcontext()
            if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
            else get_global_expert_distribution_recorder().with_current_layer(i)
        )
        with ctx:
            if _is_cuda:
                hidden_states = rehome.layer(i, self.layers[i])(
                    positions=positions,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_ids_global=input_ids_global,
                )
            else:
                if _is_hip and self.layers[i].hc_boundary_fused:
                    state = forward_layer_fused_boundary(
                        self,
                        i,
                        state,
                        positions=positions,
                        input_ids=input_ids,
                        forward_batch=forward_batch,
                        input_ids_global=input_ids_global,
                        capture_dspark=capture_dspark,
                    )
                    continue
                state = self.layers[i].forward_hc_pre_from_prev(
                    positions=positions,
                    state=state,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_ids_global=input_ids_global,
                )
    if _is_cuda:
        state = residual_batch.read_output(
            hidden_states, forward_batch, hc_boundary.FinalRead(), None
        )
        residual_batch.clear(forward_batch)
    state = state.materialized(self.layers[self.end_layer - 1].hc_cfg)
    if saved_dp is not None:
        exit_late_layer_dp(forward_batch, saved_dp)
    if saved_full is not None:
        if tail_in_graph:
            tail_sentinel = bcg_deepseek_v4_exit_late_layer_tail(tail_sentinel)
        else:
            attn_backend.exit_late_layer_tail(saved_full, forward_batch)
        return state.residual, state.pre, tail
    return state.residual, state.pre, None
