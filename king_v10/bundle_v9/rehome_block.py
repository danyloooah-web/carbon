
from __future__ import annotations

from functools import partial
from types import MethodType

import torch

from bundle_v9 import rehome_attn, rehome_transport as T

_RUN: list = []
_STOCK: list = []


def prepare(view, module) -> bool:
    from sglang.srt.models.deepseek_v4 import DeepseekV4Model

    from bundle_v9 import rehome_loop

    if not T.active():
        return False
    loop = DeepseekV4Model._forward_layers_hc_pre_from_prev
    if (loop.__module__, loop.__qualname__) != (
        "sglang.srt.models.deepseek_v4",
        "DeepseekV4Model._forward_layers_hc_pre_from_prev",
    ):
        raise RuntimeError("v9-rehome: DeepseekV4Model._forward_layers_hc_pre_from_prev is not stock's")
    late = module.late_layer_start
    if late != 20 or module.decoder_replay_tail is None or not module.tail_writes_source_kv:
        raise RuntimeError(f"v9-rehome requires late_layer_start=20 with source-KV write, got late={late}")
    _STOCK[:] = [loop]
    for i in range(late):
        rehome_attn.check_layer(i, module.layers[i].self_attn)
    view._forward_layers_hc_pre_from_prev = MethodType(rehome_loop.forward_layers, view)
    return True


def stock_loop():
    return _STOCK[0]


def begin(model, forward_batch, positions, input_ids, input_ids_global, capture_dspark):
    from bundle_v9 import rehome_plan

    plan = rehome_plan.step_plan(forward_batch)
    if plan is None:
        return None
    if T.link().role != "prefill":
        import torch.distributed as dist
        from sglang.srt.runtime_context import get_parallel

        pure = torch.tensor(
            [int(input_ids.shape[0] == 0 or forward_batch.forward_mode.is_extend_without_speculative())],
            dtype=torch.int32,
        )
        dist.all_reduce(pure, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
        if not int(pure.item()):
            return None
    from sglang.srt.runtime_context import get_parallel

    from bundle_v9 import rehome_edges

    owner_kv, helper_kv, over = rehome_edges.l0_kv_exchange(
        plan, int(get_parallel().attn_dp_rank), forward_batch.seq_lens_cpu, forward_batch.extend_seq_lens_cpu
    )
    if over:
        return None
    _RUN[:] = [Run(model, plan, forward_batch, positions, input_ids, input_ids_global, capture_dspark)]
    _RUN[0].owner_kv, _RUN[0].helper_kv = owner_kv, helper_kv
    return _RUN[0]


def end(run) -> None:
    _RUN.clear()
    T.join()


def in_flight() -> bool:
    return bool(_RUN)


def active():
    if not _RUN:
        raise RuntimeError("v9-rehome: no re-home run in flight")
    return _RUN[0]


class Run:

    def __init__(self, model, plan, forward_batch, positions, input_ids, input_ids_global, capture_dspark):
        from sglang.srt.models.deepseek_v4 import is_cp_active, is_in_breakable_cuda_graph
        from sglang.srt.runtime_context import get_forward, get_parallel

        self.rank = int(get_parallel().attn_dp_rank)
        self.plan, self.fb, self.n = plan, forward_batch, plan.counts[self.rank]
        self.late = model.late_layer_start
        self.out = tuple(sorted(plan.outgoing(self.rank), key=lambda mv: mv.lo))
        incoming = plan.incoming(self.rank)
        self.move = incoming[0] if incoming else None
        self.kept = plan.kept(self.rank)
        self.input_ids = input_ids
        dspark = model.dspark_layers_to_capture if capture_dspark else None
        problems = [
            input_ids.shape[0] != self.n or positions.shape[0] != self.n,
            input_ids_global is not input_ids,
            self.n > 0 and not forward_batch.forward_mode.is_extend_without_speculative(),
            is_cp_active(forward_batch) or is_in_breakable_cuda_graph() or get_forward().sp_active,
            self.late is None or any(i < self.late for i in (dspark or ())),
            bool(self.out) and bool(incoming),
            len(incoming) > 1,
            bool(self.out)
            and [(mv.lo, mv.hi) for mv in self.out]
            != list(zip([self.kept] + [mv.hi for mv in self.out[:-1]], [mv.hi for mv in self.out])),
            bool(self.out) and (self.out[-1].hi != self.n or self.kept < 1),
            self.move is not None and (self.move.at != self.n or self.move.hi <= self.move.lo),
        ]
        if any(problems):
            raise RuntimeError(
                f"v9-rehome: rank {self.rank} ({self.n} rows) cannot run plan {plan} "
                f"(failed checks {[k for k, bad in enumerate(problems) if bad]})"
            )
        self.fused = [i > 0 and model.layers[i].engram is None for i in range(self.late)]
        self.ids_m = self.hs_m = self.stream_m = None
        self.owner_kv = self.helper_kv = None

    def enter(self, model, state, hidden_states, tail):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models import deepseek_v4_mhc as mhc

        from bundle_v9 import rehome_edges

        fb = self.fb
        if self.out:
            rehome_edges.l0_send(self.plan, self.rank, self.input_ids, fb.extend_seq_lens_cpu, tail.extend_seq_lens_cpu)
            state = mhc.HcState(state.residual[: self.kept])
            return state, residual_batch.set_written(state, fb)
        if self.move is not None:
            own = fb.residual_stream
            self.ids_m, residual = rehome_edges.l0_receive(self.plan, self.rank, model)
            self.hs_m = residual_batch.set_written(mhc.HcState(residual), fb)
            self.stream_m, fb.residual_stream = fb.residual_stream, own
        return state, hidden_states

    def layer(self, i, layer):
        if i >= self.late or not (self.out or self.move is not None):
            return layer
        return partial(self._owner if self.out else self._helper, i, layer)

    def engram(self, layer, x, indices, *, input_ids, cp_all_tokens):
        from sglang.srt.layers.engram import engram_gate
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models import deepseek_v4_boundary as hc_boundary

        from bundle_v9 import peer_lookup, row_exchange

        engram, fb = layer.engram, self.fb
        if cp_all_tokens or fb.engram_staged_rows is not None or x.shape[0] != self.kept:
            raise RuntimeError(
                f"bundle-v9 ENGRAM: rank {self.rank} rows {x.shape[0]} (kept {self.kept}), CP "
                f"{cp_all_tokens}, staged rows {fb.engram_staged_rows is not None}"
            )
        if hasattr(engram.embed, "_egp_table"):
            emb = peer_lookup.rehome_lookup(engram.embed, engram.embed._egp_table, indices, fb, self.plan, self.rank)
        else:
            emb = peer_lookup.rh_lookup(engram.embed, indices, fb, self.plan, self.rank)
            if emb is None:
                emb = row_exchange.rehome_lookup(engram.embed, indices, fb, self.plan, self.rank)
        if emb.shape[0] == 0:
            return x
        kv, _ = engram.wkv(emb.flatten(-2))
        image = engram.image_token_id is not None
        k = x.shape[0]
        gate = partial(
            engram_gate,
            q_weight=engram.q_weight,
            k_weight=engram.k_weight,
            eps=engram.eps,
            clamp_value=engram.clamp_value,
            image_token_id=engram.image_token_id,
        )
        out = gate(x, kv[:k], input_ids=input_ids[:k] if image else None) if k else x
        if self.move is not None:
            own = fb.residual_stream
            fb.residual_stream = self.stream_m
            call = hc_boundary.LayerCall(self.ids_m, self.ids_m)
            nxt = hc_boundary.HcRead(layer, True).bind(fb, call).next_boundary
            state = hc_boundary.materialize(self.hs_m, fb, next=nxt)
            rows = gate(state.residual, kv[k:], input_ids=self.ids_m if image else None)
            self.hs_m = residual_batch.set_written(state.with_residual(rows), fb)
            self.stream_m, fb.residual_stream = fb.residual_stream, own
        return out

    def leave(self, model, i, state, tail):
        from bundle_v9 import rehome_edges

        layer = model.layers[i]
        if self.move is not None:
            from sglang.srt.models import deepseek_v4_boundary as hc_boundary

            fb = self.fb
            own = fb.residual_stream
            fb.residual_stream = self.stream_m
            call = hc_boundary.LayerCall(self.ids_m, self.ids_m)
            nxt = hc_boundary.HcRead(layer, True).bind(fb, call).next_boundary
            moved = hc_boundary.materialize(self.hs_m, fb, next=nxt)
            fb.residual_stream = own
            self.hs_m = self.stream_m = None
            rehome_edges.l20_return(self.plan, self.rank, moved, layer.attn_hc)
            return None
        if self.out:
            return rehome_edges.l20_compose(self.plan, self.rank, state, layer.attn_hc, tail, self.fb.extend_seq_lens_cpu)
        return None

    def _owner(self, i, layer, *, positions, hidden_states, input_ids, forward_batch, input_ids_global):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models import deepseek_v4_boundary as hc_boundary

        from bundle_v9 import attn_boundary

        fb, k = forward_batch, self.kept
        call = hc_boundary.LayerCall(input_ids[:k], input_ids_global[:k])
        if i in rehome_attn.SOURCE_LAYERS:
            x_all = T.view("x", self.n)
            if self.fused[i]:
                x = _into(
                    attn_boundary._REHOME_Y_BF16,
                    x_all[:k],
                    layer.attn_boundary.prepare,
                    hidden_states,
                    fb,
                    read_context=call,
                )
            else:
                x = layer.attn_boundary.prepare(hidden_states, fb, read_context=call)
                x_all[:k].copy_(x)
        else:
            x = layer.attn_boundary.prepare(hidden_states, fb, read_context=call)
        pair = residual_batch.written_residual(fb).quantized
        if self.fused[i] and pair is None:
            raise RuntimeError(f"v9-rehome: layer {i} fused seam left no DeepGEMM pair for wqkv_a")
        with layer.self_attn.maybe_use_decode_attn_tp(fb):
            out = rehome_attn.owner_attention(
                i, layer.self_attn, x, pair, positions, fb, self.n, k, self.out, self.owner_kv
            )
            del x, pair
            hidden_states = layer.attn_boundary.finish(out, fb)
            del out
        hidden_states = layer.ffn_boundary.prepare(hidden_states, fb, read_context=call)
        transport = layer.ffn_boundary.transport_context(fb)
        with layer.ffn_boundary.exit(fb) as output:
            hidden_states = layer.mlp(
                hidden_states,
                fb,
                input_ids=transport.input_ids,
                input_ids_global=transport.input_ids_global,
                skip_shared_experts=transport.do_shared_local,
                return_moe_output=transport.return_moe_output,
                leave_output_sum=True,
            )
            return output.finish(hidden_states)

    def _helper(self, i, layer, *, positions, hidden_states, input_ids, forward_batch, input_ids_global):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models import deepseek_v4_boundary as hc_boundary
        from sglang.srt.runtime_context import get_forward

        from bundle_v9 import moe_route

        fb, mv, own = forward_batch, self.move, self.n
        total = own + mv.hi - mv.lo
        own_stream = fb.residual_stream
        fb.residual_stream = self.stream_m
        call_m = hc_boundary.LayerCall(self.ids_m, self.ids_m)
        x_m = layer.attn_boundary.prepare(self.hs_m, fb, read_context=call_m)
        self.stream_m = fb.residual_stream
        pair_m = residual_batch.written_residual(fb).quantized
        with layer.self_attn.maybe_use_decode_attn_tp(fb):
            q_m = rehome_attn.helper_pre(i, layer.self_attn, x_m, pair_m, mv)
        del x_m, pair_m
        fb.residual_stream = own_stream
        call_o = hc_boundary.LayerCall(input_ids, input_ids_global)
        x_o = layer.attn_boundary.prepare(hidden_states, fb, read_context=call_o)
        quant_o = residual_batch.written_residual(fb).quantized
        with layer.self_attn.maybe_use_decode_attn_tp(fb):
            hidden_states = layer._run_boundary_attention(x_o, positions, fb, quant_o)
            del x_o, quant_o
            hidden_states = layer.attn_boundary.finish(hidden_states, fb)
        moe_o = layer.ffn_boundary.prepare(hidden_states, fb, read_context=call_o)
        transport_o, exit_o = _exit(layer, fb)
        if get_forward().mlp_reduce_scatter or get_forward().defer_moe_finalize:
            raise RuntimeError("v9-rehome: MOE_FRONT runs outside the exit scope only while its flags are default")
        block_m = (
            [
                moe_route.rehome_front(
                    layer.mlp, moe_o, fb, transport_o.input_ids_global, row_offset=0, total_rows=total, pad=False, live_all=False
                )
            ]
            if own
            else []
        )
        with layer.self_attn.maybe_use_decode_attn_tp(fb):
            attn_m = rehome_attn.helper_post(i, layer.self_attn, q_m, mv, self.helper_kv)
        del q_m
        fb.residual_stream = self.stream_m
        hidden_m = layer.attn_boundary.finish(attn_m, fb)
        del attn_m
        moe_m = layer.ffn_boundary.prepare(hidden_m, fb, read_context=call_m)
        transport_m, exit_m = _exit(layer, fb)
        block_m.append(
            moe_route.rehome_front(
                layer.mlp, moe_m, fb, transport_m.input_ids_global, row_offset=own, total_rows=total, pad=True, live_all=True
            )
        )
        if len(set(block_m)) != 1:
            raise RuntimeError(f"v9-rehome: layer {i} blocks disagree on BLOCK_M {block_m}")
        with exit_o, exit_m:
            y = moe_route.rehome_experts(layer.mlp, moe_m, total)
            self.hs_m = exit_m.finish(y[own:])
            hidden_states = exit_o.finish(y[:own])
        self.stream_m, fb.residual_stream = fb.residual_stream, own_stream
        return hidden_states


def _exit(layer, fb):
    transport = layer.ffn_boundary.transport_context(fb)
    if transport.mlp_reduce_scatter or transport.do_shared_local or transport.return_moe_output:
        raise RuntimeError("v9-rehome: MoE transport is not the plain megamoe one")
    return transport, layer.ffn_boundary.exit(fb)


def _into(slot: list, out: torch.Tensor, fn, *args, **kwargs):
    slot[:] = [out]
    try:
        result = fn(*args, **kwargs)
    finally:
        left = len(slot)
        slot.clear()
    if left or result.data_ptr() != out.data_ptr() or result.shape != out.shape:
        raise RuntimeError(f"v9-rehome: in-place output not consumed ({left} left, {tuple(result.shape)})")
    return result
