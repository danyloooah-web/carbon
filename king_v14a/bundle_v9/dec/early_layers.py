from __future__ import annotations

import logging
from functools import partial
from types import MethodType
from typing import NamedTuple, Optional

import deep_gemm
import torch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.moe.mega_moe import (
    _configure_mega_moe_deep_gemm_num_sms,
    _mega_moe_buffer,
    _mega_moe_global_tokens_hint,
    should_use_mega_moe,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.runtime_context import get_forward
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

from bundle_v9.dec import moe_route

logger = logging.getLogger(__name__)

EARLY = 20
LO, RMAX = 129, 4095
GRID = (192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4095)
HIDDEN, MULT, MIX = 5120, 4, 24
SF_WORDS = HIDDEN // 128
MOE_REPLICA = True

_LEAN: list = []
_PLAN: list = [None]


def _ceil4(n: int) -> int:
    return -(-n // 4) * 4


class _Facts(NamedTuple):
    mix: bool
    pending: bool
    next_attn: Optional[object]


class _Plan:

    __slots__ = ("lean", "batch", "n", "rows", "qsf", "handoff", "check", "moe", "oy", "x", "xq", "res_a", "res_f",
                 "stats_a", "stats_f", "qsf_src", "pair")

    def __init__(self, lean, batch, n):
        rows = lean.bucket[n]
        self.lean, self.batch, self.n, self.rows = lean, batch, n, rows
        self.qsf = torch.empty_strided((n, SF_WORDS), (1, _ceil4(n)), dtype=torch.int32, device=lean.device)
        self.handoff = self.check = self.moe = None
        self.oy, self.x, self.xq, self.res_a, self.res_f = (lean.oy[:n], lean.x[:n], lean.xq[:n], lean.res_a[:n],
                                                            lean.res_f[:n])
        self.stats_a = (lean.st_a[0][:n].squeeze(-1), lean.st_a[1][:n].squeeze(-1), lean.st_a[2][:n])
        self.stats_f = (lean.st_f[0][:n].squeeze(-1), lean.st_f[1][:n].squeeze(-1), lean.st_f[2][:n])
        self.qsf_src = lean.qsf_rows[rows][:n]
        self.pair = (self.xq, self.qsf)


def _regime(n: int, sms: int) -> tuple:
    from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as mmhc

    return mmhc.num_splits(n, HIDDEN, sms), n >= 4096, n >= 32


def _buckets(sms: int) -> list:
    edges = set(GRID)
    edges.update(n for n in range(LO, RMAX) if _regime(n, sms) != _regime(n + 1, sms))
    return sorted(e for e in edges if LO <= e <= RMAX)


class _Lean:

    def __init__(self, view, module, sk_impl):
        from sglang.srt.runtime_context import get_platform

        from bundle_v9.dec import attn_boundary

        if module.late_layer_start != EARLY or len(module.layers) != 40 or not module.hc_pre_from_prev_sublayer:
            raise RuntimeError(f"bundle-v9 LEAN requires the replay tail at layer {EARLY} of 40 with lagged mHC, got "
                               f"late_layer_start={module.late_layer_start}")
        if not get_platform().is_sm100:
            raise RuntimeError("bundle-v9 LEAN requires an SM10x GPU (Mega-mHC prefill boundaries)")
        self.device = view.layers[0].self_attn.wq_b.weight.device
        mmhc_facts = attn_boundary._MMHC.get(self.device.index)
        if mmhc_facts is None:
            raise RuntimeError("bundle-v9 LEAN: attn_boundary.prepare has not set up the mega mHC on this GPU")
        self.barriers, self.epochs, self.sms = mmhc_facts
        self.sk = sk_impl
        self.views = list(view.layers)
        facts = []
        for i in range(EARLY):
            layer = self.views[i]
            nxt = self.views[i + 1]
            pending = i + 1 < EARLY and nxt.engram is not None
            self._check_layer(layer)
            if not pending:
                self._check_attention_entry(nxt)
            mix = i == 0 or layer.engram is not None
            facts.append(_Facts(mix, pending, None if pending else nxt))
        if self.views[EARLY].engram is not None:
            raise RuntimeError("bundle-v9 LEAN: the tail's first layer carries an Engram")
        self.facts = facts
        self.moe_rsf = [1.0 if v.mlp.routed_scaling_factor_fused_in_topk else float(v.mlp.routed_scaling_factor)
                        for v in self.views[:EARLY]]
        self.moe_buf = None
        self.moe_covered = set()
        self.edges = _buckets(self.sms)
        self.bucket = [0] * (RMAX + 1)
        lo = LO
        for edge in self.edges:
            regime = _regime(edge, self.sms)
            for n in range(lo, edge + 1):
                if _regime(n, self.sms) != regime:
                    raise RuntimeError(f"bundle-v9 LEAN: bucket {edge} serves {n} rows across a regime change")
                self.bucket[n] = edge
            lo = edge + 1
        rows = _ceil4(RMAX)
        dev = self.device
        self.res_a = torch.zeros((rows, MULT, HIDDEN), dtype=torch.bfloat16, device=dev)
        self.res_f = torch.zeros_like(self.res_a)
        self.oy = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev)
        self.x = torch.zeros_like(self.oy)
        self.xq = torch.zeros((rows, HIDDEN), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        self.qsf = torch.zeros((rows * SF_WORDS,), dtype=torch.int32, device=dev)
        self.st_a = (torch.zeros((rows, MULT, 1), dtype=torch.float32, device=dev),
                     torch.zeros((rows, MULT, 1), dtype=torch.float32, device=dev),
                     torch.zeros((rows, MULT, MULT), dtype=torch.float32, device=dev))
        self.st_f = tuple(torch.zeros_like(t) for t in self.st_a)
        self.qsf_rows = {r: torch.as_strided(self.qsf, (r, SF_WORDS), (1, _ceil4(r))) for r in self.edges}
        self.static_bytes = sum(t.numel() * t.element_size() for t in
                                (self.res_a, self.res_f, self.oy, self.x, self.xq, self.qsf, *self.st_a, *self.st_f))
        self.graph_a, self.graph_c = {}, {}
        self.pool = torch.cuda.graph_pool_handle()

    @staticmethod
    def _check_layer(layer):
        from sglang.srt.layers.moe.mega_moe import _mega_moe_mma_type, is_mega_moe_experts_ready

        for hc, norm in ((layer.attn_hc, layer.input_layernorm), (layer.ffn_hc, layer.post_attention_layernorm)):
            if hc is None or hc.norm is not norm or not _hc_shapes_ok(hc):
                raise RuntimeError(f"bundle-v9 LEAN: layer {layer.layer_id} mHC parameters are not DSv4.1-Flash's")
            if not mhc.make_boundary(norm, accepts_mxfp8=False, hc=hc).norm_fusable:
                raise RuntimeError(f"bundle-v9 LEAN: layer {layer.layer_id} norm does not fold into Mega-mHC")
        if not (layer.boundary_pre_from_prev and layer.boundary_mhc_fusions and mhc._can_use_mega_mhc(layer.hc_cfg)):
            raise RuntimeError(f"bundle-v9 LEAN: layer {layer.layer_id} does not take Mega-mHC boundaries")
        if layer.self_attn.attn_tp_size != 1:
            raise RuntimeError("bundle-v9 LEAN requires attention TP 1 (attention output owes no sum)")
        moe = layer.mlp
        if (moe.mega_shared_l1_weights is None or moe.num_fused_shared_experts != 0
                or _mega_moe_mma_type(moe.experts) != "fp8xfp4" or not is_mega_moe_experts_ready(moe.experts)):
            raise RuntimeError(f"bundle-v9 LEAN: layer {layer.layer_id} MoE is not the fused-shared fp8xfp4 MegaMoE")

    @staticmethod
    def _check_attention_entry(layer):
        if not (layer.self_attn.accepts_deepgemm_mxfp8_input() and layer.boundary_mhc_fusions
                and _hc_shapes_ok(layer.attn_hc) and layer.attn_hc.norm is layer.input_layernorm
                and mhc.make_boundary(layer.input_layernorm, accepts_mxfp8=False, hc=layer.attn_hc).norm_fusable):
            raise RuntimeError(f"bundle-v9 LEAN: layer {layer.layer_id}'s attention entry is not the MXFP8 mega mHC")

    def _body_a(self, i, rows):
        layer = self.views[i]
        if self.facts[i].mix:
            pre, post, comb = self.sk(layer.attn_hc, self.res_a[:rows])
            pre, post = pre.unsqueeze(-1), post.unsqueeze(-1)
        else:
            pre, post, comb = (t[:rows] for t in self.st_a)
        nxt = layer.ffn_hc
        deep_gemm.mega_mhc(
            x=self.oy[:rows], residual=self.res_a[:rows], shifted_prev_mix=pre, post_mix=post, comb_res_mix=comb,
            fn=nxt.fn, mix_scales=nxt.scale, mix_bases=nxt.base, hc_mult=MULT, hc_norm_eps=nxt.cfg.rms_eps,
            hc_pre_eps=nxt.cfg.eps, hc_post_scale=2.0, sinkhorn_eps=nxt.cfg.eps,
            num_sinkhorn_iters=nxt.cfg.sinkhorn_iters, rmsnorm_weight=nxt.norm.weight,
            rmsnorm_eps=nxt.norm.variance_epsilon, rmsnorm_scale=1.0, new_residual=self.res_f[:rows],
            new_prev_mix=self.st_f[0][:rows], new_post_mix=self.st_f[1][:rows], new_comb_res_mix=self.st_f[2][:rows],
            y_bf16=self.x[:rows], y_fp8=None, y_gemm_sf=None, y_routed_sf=None, y_shared_sf=None, shared_sf_block_m=0)

    def _body_c(self, i, rows):
        from flashinfer.mega_mhc import prepare_mega_mhc

        from bundle_v9.dec import attn_boundary

        nxt = self.facts[i].next_attn.attn_hc
        args = (self.oy[:rows], self.res_f[:rows], self.st_f[0][:rows].squeeze(-1), self.st_f[1][:rows].squeeze(-1),
                self.st_f[2][:rows], nxt.fn, nxt.scale, nxt.base, nxt.norm.weight, self.xq[:rows], self.qsf_rows[rows],
                None)
        if not attn_boundary._mmhc_served(*args):
            raise RuntimeError(f"bundle-v9 LEAN: the mega mHC does not serve layer {i + 1}'s entry at {rows} rows")
        out = dict(new_residual=self.res_a[:rows], new_prev_mix=self.st_a[0][:rows], new_post_mix=self.st_a[1][:rows],
                   new_comb_res_mix=self.st_a[2][:rows], y_bf16=self.x[:rows], y_fp8=self.xq[:rows],
                   y_gemm_sf=self.qsf_rows[rows])
        prepare_mega_mhc(
            x=self.oy[:rows], residual=self.res_f[:rows], post_mix=self.st_f[1][:rows], comb_res_mix=self.st_f[2][:rows],
            shifted_prev_mix=self.st_f[0][:rows], fn=nxt.fn, mix_scales=nxt.scale, mix_bases=nxt.base,
            rmsnorm_weight=nxt.norm.weight, hc_mult=MULT, hc_norm_eps=nxt.cfg.rms_eps, hc_pre_eps=nxt.cfg.eps,
            hc_post_scale=2.0, sinkhorn_eps=nxt.cfg.eps, num_sinkhorn_iters=nxt.cfg.sinkhorn_iters,
            rmsnorm_eps=nxt.norm.variance_epsilon, rmsnorm_scale=1.0, sf_layout="col", out=out,
            split_barriers=self.barriers, launch_epochs=self.epochs).run()

    def _capture(self, stream, body):
        with torch.cuda.stream(stream):
            body()
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                body()
            finally:
                graph.capture_end()
        return graph

    def capture(self):
        torch.cuda.synchronize(self.device)
        free = torch.cuda.mem_get_info(self.device)[0]
        reserved = torch.cuda.memory_reserved(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        for rows in reversed(self.edges):
            for i in range(EARLY):
                self.graph_a[(i, rows)] = self._capture(stream, partial(self._body_a, i, rows))
                if not self.facts[i].pending:
                    self.graph_c[(i, rows)] = self._capture(stream, partial(self._body_c, i, rows))
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.pool_bytes = torch.cuda.memory_reserved(self.device) - reserved
        self.graph_bytes = free - torch.cuda.mem_get_info(self.device)[0]
        logger.warning("bundle-v9 LEAN: %d graphs over %d row buckets %s; static %.1f MiB, allocator +%.1f MiB "
                       "during capture, device free -%.1f MiB", len(self.graph_a) + len(self.graph_c),
                       len(self.edges), self.edges, self.static_bytes / 2**20, self.pool_bytes / 2**20,
                       self.graph_bytes / 2**20)

    def layer(self, view, plan, positions, hidden_states, input_ids, input_ids_global, fusion):
        i, batch, n = view.layer_id, plan.batch, plan.n
        facts = self.facts[i]
        stream = residual_batch.stream_of(batch)
        if (not fusion or stream.pending is not None or stream.residual is not hidden_states
                or type(hidden_states) is not mhc.HcState):
            raise RuntimeError(f"bundle-v9 LEAN: layer {i} entered without a written mHC stream")
        call = hc_boundary.LayerCall(input_ids, input_ids_global, fusion)
        if facts.mix:
            x, read = hc_boundary.HcRead(view, True).bind(batch, call).read(hidden_states, view.input_layernorm)
            if read.stats is not None or read.stats_stream is not None:
                raise RuntimeError(f"bundle-v9 LEAN: layer {i}'s entry carries stats or a stats stream")
            x_quant = read.quantized
            plan.res_a.copy_(read.streams)
        else:
            if hidden_states is not plan.handoff:
                raise RuntimeError(f"bundle-v9 LEAN: layer {i} did not receive layer {i - 1}'s handoff")
            x, x_quant = plan.x, plan.pair
            hidden_states.release()
        with view.self_attn.maybe_use_decode_attn_tp(batch):
            o = view.self_attn(x=x, positions=positions, forward_batch=batch, x_quant=x_quant, defer_all_reduce=True)
        if type(o) is not torch.Tensor or o.shape[0] != n or o.shape[1] != HIDDEN:
            raise RuntimeError(f"bundle-v9 LEAN: layer {i}'s attention returned {type(o).__name__}, not [n, hidden]")
        plan.oy.copy_(o)
        del o, x, x_quant
        self.graph_a[(i, plan.rows)].replay()
        with get_forward().scoped(mlp_reduce_scatter=False, defer_moe_finalize=False):
            y = self._moe(view, plan, input_ids, input_ids_global)
        if facts.pending:
            stream = ResidualStream(hc_boundary.ReadState(plan.res_f, stats=plan.stats_f, call=call))
            batch.residual_stream = stream
            return stream.record(y, view.ffn_boundary.plan.path_for(batch).output.update)
        self.graph_c[(i, plan.rows)].replay()
        plan.qsf.copy_(plan.qsf_src)
        state = mhc.HcState(plan.res_a, plan.stats_f[0], mhc.HcQuantized(plan.x, plan.pair), plan.stats_a)
        plan.handoff = state
        if i == EARLY - 1:
            _PLAN[0] = None
        return residual_batch.set_written(state, batch)

    def _moe(self, view, plan, input_ids, input_ids_global):
        if not MOE_REPLICA:
            y = view.mlp(plan.x, plan.batch, input_ids=input_ids, input_ids_global=input_ids_global,
                         skip_shared_experts=False, return_moe_output=False, leave_output_sum=True)
            return plan.oy.copy_(y)
        return self._moe_replica(view, plan, input_ids_global)

    def _moe_replica(self, view, plan, input_ids_global):
        from bundle_v9.dec import prefill_gemm

        with prefill_gemm.moe_tiles(plan.batch):
            moe, x, n = view.mlp, plan.x, plan.n
            if plan.moe is None:
                if not should_use_mega_moe(moe, x):
                    raise RuntimeError("bundle-v9 LEAN: stock would not run MegaMoE on this forward")
                if self.moe_buf is None:
                    self.moe_buf = _mega_moe_buffer(moe)
                buf = self.moe_buf
                hint = _mega_moe_global_tokens_hint(buf)
                kwargs = {"recipe": (1, 1, 32)}
                if hint is not None:
                    kwargs["global_num_tokens"] = hint
                block_m = deep_gemm.get_block_m_for_mega_moe(
                    num_ranks=buf.group.size(), num_experts=buf.num_experts,
                    num_max_tokens_per_rank=buf.num_max_tokens_per_rank, num_tokens=max(n, 1), num_topk=buf.num_topk,
                    mma_type=buf.mma_type, global_num_tokens=hint)
                plan.moe = (kwargs, block_m)
            kwargs, block_m = plan.moe
            buf = self.moe_buf
            logits = moe.gate(x, forward_batch=plan.batch)
            if view.layer_id not in self.moe_covered:
                if not moe_route._front_covered(moe, logits, x):
                    raise RuntimeError(f"bundle-v9 LEAN: MOE_FRONT does not serve layer {view.layer_id} here")
                self.moe_covered.add(view.layer_id)
            elif logits.dtype is not torch.float32 or logits.shape[0] != n:
                raise RuntimeError(f"bundle-v9 LEAN: layer {view.layer_id}'s router logits are not fp32 [n, experts]")
            moe_route._front((moe, logits, input_ids_global), x, buf, self.moe_rsf[view.layer_id],
                                      buf.shared_l1_acts_sf, block_m)
            y = plan.oy
            with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
                deep_gemm.fp8_fp4_mega_moe(
                    y, moe.experts.mega_l1_weights, moe.experts.mega_l2_weights, buf,
                    shared_l1_weights=moe.mega_shared_l1_weights, shared_l2_weights=moe.mega_shared_l2_weights,
                    activation="swiglu", activation_clamp=moe.experts.moe_runner_config.swiglu_limit, fast_math=True,
                    **kwargs)
            return y


def _hc_shapes_ok(hc) -> bool:
    cfg = hc.cfg
    return (cfg.mult == MULT and cfg.hidden == HIDDEN and cfg.pre_from_prev and not cfg.cp_prefill
            and tuple(hc.fn.shape) == (MIX, MULT * HIDDEN) and hc.fn.dtype == torch.float32
            and hc.norm.weight.dtype == torch.bfloat16 and tuple(hc.norm.weight.shape) == (HIDDEN,))


def layer_forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global, *,
                  allow_cross_layer_fusion=True):
    if self.layer_id == 0:
        _PLAN[0] = _plan(forward_batch, input_ids)
    plan = _PLAN[0]
    if plan is None or plan.batch is not forward_batch:
        return type(self).forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global,
                                  allow_cross_layer_fusion=allow_cross_layer_fusion)
    return plan.lean.layer(self, plan, positions, hidden_states, input_ids, input_ids_global,
                           allow_cross_layer_fusion)


def _plan(batch, input_ids) -> Optional[_Plan]:
    if not _LEAN or batch is None or batch.forward_mode is not ForwardMode.EXTEND:
        return None
    n = int(input_ids.shape[0])
    if not LO <= n <= RMAX or torch.cuda.is_current_stream_capturing():
        return None
    fwd = get_forward()
    lean = _LEAN[0]
    if (fwd.sp_active or not fwd.is_extend_in_batch or is_cp_active(batch)
            or lean.views[0].attn_boundary.plan.variant_for(batch) is not BatchVariant.ORDINARY
            or get_global_expert_distribution_recorder().recording or get_global_experts_capturer() is not None):
        return None
    return _Plan(lean, batch, n)


def prepare(view, module, sk_impl) -> None:
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        logger.warning("bundle-v9 LEAN: PD decode engine, no early-layer graphs")
        return
    if _LEAN or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("bundle-v9 LEAN: prepare ran twice or inside a CUDA graph capture")
    lean = _Lean(view, module, sk_impl)
    lean.capture()
    _LEAN.append(lean)
    for layer_view in view.layers[:EARLY]:
        layer_view.forward = MethodType(layer_forward, layer_view)
