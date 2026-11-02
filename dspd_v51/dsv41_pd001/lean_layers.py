"""dsv41-pd032 LEAN: decoder layers 0-19 of small prefill forwards without the layer-boundary framework, their two mHC
boundaries a layer replayed from per-row-bucket CUDA graphs.

Why: on small prefill forwards (<= 8K rows on the busiest rank, ~30% of steady-state prefill time) the prefill engine is
host-bound, and pd029t (legs t36/t40) put 1.05 of an eager layer's 2.38 ms of host time outside the attention:
attention / FFN boundary preparation, their finish / exit, and the MoE call. Those launch five kernels a layer; the
p43 torch trace splits their host time about 60% framework Python, 40% launch wrappers (the FlashInfer mega-mHC plan
build, deep_gemm.mega_mhc, cuBLAS, MegaMoE).

Layer 0's view forward decides each forward's plan (`_plan`). On a rank whose forward is in the domain (an EXTEND
forward of LO..RMAX rows on this rank, no capture, CP, SP or routing record), layer i runs:
  1. stock's attention module, called as DeepseekV4CudaDecoderLayer.forward calls it;
  2. graph A(i, R): the attention's post and the FFN entry (deep_gemm.mega_mhc, stock's arguments, static buffers);
     layer 0 and the Engram layers, whose attention entry carries no precomputed stats, first run the SK mix stats;
  3. the MoE at the live rows: stock's eager MegaMoE call sequence (MOE_REPLICA) or the MoE view's forward;
  4. graph C(i, R): the MoE's post and layer i+1's attention entry (FlashInfer's mega mHC as pd022's v16 lever
     serves it, writing the next attention's MXFP8 pair);
and publishes the residual stream stock's next reader expects. FUSED: the written HcState stock's materialize returns
unchanged (DSpark capture, the layer-20 tail switch). PENDING, before an Engram layer: the MoE output with the FFN's
HcUpdate, so stock's materialize runs the plain post, as for pd034.

Bitwise: each graphed kernel computes a row from that row's inputs with a reduction structure set by its split count.
A bucket R serves only row counts n whose mega-mHC split count (FlashInfer's num_splits, which the cake shares with
deep_gemm.mega_mhc) and mix-stats projection (SK / DeepGEMM below 4096 rows) equal R's, so rows [0, n) of a replay
are stock's bytes for n rows. Rows [n, R) keep stale values that only row-local kernels read. dsv41_pd032t compares
every replay against stock's run_attn_post / run_moe_post at n rows, and the replica against the MoE view's forward.

DP: no collective is added, removed, reordered or graphed. The MoE runs at the live rows and calls MegaMoE once a
layer with stock's arguments (global token hint, shared block_m, output rows), so a rank-local lean / stock choice
leaves every rank's collective sequence unchanged; IDLE ranks and out-of-domain forwards run pd034's path.
"""

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

from dsv41_pd001 import mega_forward

logger = logging.getLogger(__name__)

EARLY = 20  # layers 0..19 run eagerly in stock; layer 20 starts the decoder replay tail
# A rank's rows. Below LO the cake skips its 1/64/65-row launches and the mix stats change kernel (< 32 rows); in t40
# 99.7% of small forwards have every busy rank at >= 129 rows. Above RMAX: 11.9% of the bounding rank's small-forward
# host time (t40), partly GPU-bound already; RMAX also keeps the mix stats on SK's DeepGEMM projection (< 4096 rows).
LO, RMAX = 129, 4095
GRID = (192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4095)
HIDDEN, MULT, MIX = 5120, 4, 24
SF_WORDS = HIDDEN // 128
MOE_REPLICA = True  # build-time: stock's eager MegaMoE sequence at the live rows; False = the MoE view's forward

_LEAN: list = []  # the process's _Lean (one CUDA device a process)
_PLAN: list = [None]  # this forward's _Plan: set by layer 0's view forward, dropped at layer EARLY - 1's exit


def _ceil4(n: int) -> int:
    return -(-n // 4) * 4


class _Facts(NamedTuple):
    mix: bool  # the attention entry carries no precomputed stats (layer 0, an Engram layer): A runs the mix stats
    pending: bool  # the next layer carries an Engram: leave the MoE output for stock's materialize, no graph C
    next_attn: Optional[object]  # layer i+1's attention view (graph C's target), None when pending


class _Plan:
    """One forward's lean state on this rank: its rows, bucket, the n-row views of the static buffers (built once a
    forward, not once a layer) and the MoE's per-forward arguments."""

    __slots__ = ("lean", "batch", "n", "rows", "qsf", "handoff", "check", "moe", "oy", "x", "xq", "res_a", "res_f",
                 "stats_a", "stats_f", "qsf_src", "pair")

    def __init__(self, lean, batch, n):
        rows = lean.bucket[n]
        self.lean, self.batch, self.n, self.rows = lean, batch, n, rows
        # the next attention's scale words in DeepGEMM's layout for n rows (fp8_utils checks stride (1, ceil4(n)))
        self.qsf = torch.empty_strided((n, SF_WORDS), (1, _ceil4(n)), dtype=torch.int32, device=lean.device)
        self.handoff = self.check = self.moe = None
        self.oy, self.x, self.xq, self.res_a, self.res_f = (lean.oy[:n], lean.x[:n], lean.xq[:n], lean.res_a[:n],
                                                            lean.res_f[:n])
        self.stats_a = (lean.st_a[0][:n].squeeze(-1), lean.st_a[1][:n].squeeze(-1), lean.st_a[2][:n])
        self.stats_f = (lean.st_f[0][:n].squeeze(-1), lean.st_f[1][:n].squeeze(-1), lean.st_f[2][:n])
        self.qsf_src = lean.qsf_rows[rows][:n]
        self.pair = (self.xq, self.qsf)


def _regime(n: int, sms: int) -> tuple:
    """Every row-count choice a graphed kernel makes: the mega-mHC split count and the mix-stats projection."""
    from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as cake

    return cake.num_splits(n, HIDDEN, sms), n >= 4096, n >= 32


def _buckets(sms: int) -> list:
    """The bucket row counts: GRID plus every last row count of a regime in LO..RMAX, so a bucket serves the row
    counts above the previous bucket, all in its own regime."""
    edges = set(GRID)
    edges.update(n for n in range(LO, RMAX) if _regime(n, sms) != _regime(n + 1, sms))
    return sorted(e for e in edges if LO <= e <= RMAX)


class _Lean:
    """Static buffers, row buckets and captured graphs of one prefill rank."""

    def __init__(self, view, module, sk_impl):
        from sglang.srt.runtime_context import get_platform

        from dsv41_pd001 import v16_levers

        if module.late_layer_start != EARLY or len(module.layers) != 40 or not module.hc_pre_from_prev_sublayer:
            raise RuntimeError(f"dsv41-pd032 LEAN requires the replay tail at layer {EARLY} of 40 with lagged mHC, got "
                               f"late_layer_start={module.late_layer_start}")
        if not get_platform().is_sm100:
            raise RuntimeError("dsv41-pd032 LEAN requires an SM10x GPU (Mega-mHC prefill boundaries)")
        self.device = view.layers[0].self_attn.wq_b.weight.device
        cake_facts = v16_levers._CAKE.get(self.device.index)
        if cake_facts is None:
            raise RuntimeError("dsv41-pd032 LEAN: v16_levers.prepare has not set up the mega mHC on this GPU")
        self.barriers, self.epochs, self.sms = cake_facts
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
            raise RuntimeError("dsv41-pd032 LEAN: the tail's first layer carries an Engram")
        self.facts = facts
        self.moe_rsf = [1.0 if v.mlp.routed_scaling_factor_fused_in_topk else float(v.mlp.routed_scaling_factor)
                        for v in self.views[:EARLY]]
        self.moe_buf = None
        self.moe_covered = set()  # layers whose MEGA_FRONT domain facts covered() confirmed (static after the first)
        self.edges = _buckets(self.sms)
        self.bucket = [0] * (RMAX + 1)
        lo = LO
        for edge in self.edges:
            regime = _regime(edge, self.sms)
            for n in range(lo, edge + 1):
                if _regime(n, self.sms) != regime:
                    raise RuntimeError(f"dsv41-pd032 LEAN: bucket {edge} serves {n} rows across a regime change")
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
        """The static facts graphs A / C and the stock-call order rely on; raise otherwise."""
        from sglang.srt.layers.moe.mega_moe import _mega_moe_mma_type, is_mega_moe_experts_ready

        for hc, norm in ((layer.attn_hc, layer.input_layernorm), (layer.ffn_hc, layer.post_attention_layernorm)):
            if hc is None or hc.norm is not norm or not _hc_shapes_ok(hc):
                raise RuntimeError(f"dsv41-pd032 LEAN: layer {layer.layer_id} mHC parameters are not DSv4.1-Flash's")
            if not mhc.make_boundary(norm, accepts_mxfp8=False, hc=hc).norm_fusable:
                raise RuntimeError(f"dsv41-pd032 LEAN: layer {layer.layer_id} norm does not fold into Mega-mHC")
        if not (layer.boundary_pre_from_prev and layer.boundary_mhc_fusions and mhc._can_use_mega_mhc(layer.hc_cfg)):
            raise RuntimeError(f"dsv41-pd032 LEAN: layer {layer.layer_id} does not take Mega-mHC boundaries")
        if layer.self_attn.attn_tp_size != 1:
            raise RuntimeError("dsv41-pd032 LEAN requires attention TP 1 (attention output owes no sum)")
        moe = layer.mlp
        if (moe.mega_shared_l1_weights is None or moe.num_fused_shared_experts != 0
                or _mega_moe_mma_type(moe.experts) != "fp8xfp4" or not is_mega_moe_experts_ready(moe.experts)):
            raise RuntimeError(f"dsv41-pd032 LEAN: layer {layer.layer_id} MoE is not the fused-shared fp8xfp4 MegaMoE")

    @staticmethod
    def _check_attention_entry(layer):
        if not (layer.self_attn.accepts_deepgemm_mxfp8_input() and layer.boundary_mhc_fusions
                and _hc_shapes_ok(layer.attn_hc) and layer.attn_hc.norm is layer.input_layernorm
                and mhc.make_boundary(layer.input_layernorm, accepts_mxfp8=False, hc=layer.attn_hc).norm_fusable):
            raise RuntimeError(f"dsv41-pd032 LEAN: layer {layer.layer_id}'s attention entry is not the MXFP8 mega mHC")

    # ---- graph bodies: stock's kernels with stock's arguments, into static buffers ------------------------------
    def _body_a(self, i, rows):
        """mhc_mega_boundary(y, residual, pre, post, comb, <ffn_hc>) as pd034's eager FFN entry calls it (no MXFP8 pair:
        the cake does not serve it and the MegaMoE gate is decode-only), its outputs written into RES_F / ST_F / X."""
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
        """v16_levers._cake (FlashInfer's mega mHC: the MoE post and layer i+1's attention entry with its MXFP8 pair)
        with its outputs written into RES_A / ST_A / X / XQ / QSF."""
        from flashinfer.mega_mhc import prepare_mega_mhc

        from dsv41_pd001 import v16_levers

        nxt = self.facts[i].next_attn.attn_hc
        args = (self.oy[:rows], self.res_f[:rows], self.st_f[0][:rows].squeeze(-1), self.st_f[1][:rows].squeeze(-1),
                self.st_f[2][:rows], nxt.fn, nxt.scale, nxt.base, nxt.norm.weight, self.xq[:rows], self.qsf_rows[rows],
                None)
        if not v16_levers._cake_served(*args):
            raise RuntimeError(f"dsv41-pd032 LEAN: the mega mHC does not serve layer {i + 1}'s entry at {rows} rows")
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
        """Warm `body` on the capture stream (JIT builds, deep_gemm's per-stream barriers), then capture it into the
        private pool. thread_local: the scheduler's other threads keep their CUDA calls during a capture."""
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
        """Every (layer, bucket) graph A and C, largest bucket first, at prepare (no collective: rank-local)."""
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
        logger.warning("dsv41-pd032 LEAN: %d graphs over %d row buckets %s; static %.1f MiB, allocator +%.1f MiB "
                       "during capture, device free -%.1f MiB", len(self.graph_a) + len(self.graph_c),
                       len(self.edges), self.edges, self.static_bytes / 2**20, self.pool_bytes / 2**20,
                       self.graph_bytes / 2**20)

    # ---- one layer -------------------------------------------------------------------------------------------
    def layer(self, view, plan, positions, hidden_states, input_ids, input_ids_global, fusion):
        i, batch, n = view.layer_id, plan.batch, plan.n
        facts = self.facts[i]
        stream = residual_batch.stream_of(batch)
        if (not fusion or stream.pending is not None or stream.residual is not hidden_states
                or type(hidden_states) is not mhc.HcState):
            raise RuntimeError(f"dsv41-pd032 LEAN: layer {i} entered without a written mHC stream")
        call = hc_boundary.LayerCall(input_ids, input_ids_global, fusion)
        if facts.mix:
            # stock's attention read of a written state (layer 0, after an Engram): its combine, then this layer's
            # residual into the static buffer graph A reads
            x, read = hc_boundary.HcRead(view, True).bind(batch, call).read(hidden_states, view.input_layernorm)
            if read.stats is not None or read.stats_stream is not None:
                raise RuntimeError(f"dsv41-pd032 LEAN: layer {i}'s entry carries stats or a stats stream")
            x_quant = read.quantized
            plan.res_a.copy_(read.streams)
        else:
            if hidden_states is not plan.handoff:
                raise RuntimeError(f"dsv41-pd032 LEAN: layer {i} did not receive layer {i - 1}'s handoff")
            x, x_quant = plan.x, plan.pair
            hidden_states.release()  # as stock's read does at the combine
        with view.self_attn.maybe_use_decode_attn_tp(batch):
            o = view.self_attn(x=x, positions=positions, forward_batch=batch, x_quant=x_quant, defer_all_reduce=True)
        if type(o) is not torch.Tensor or o.shape[0] != n or o.shape[1] != HIDDEN:
            raise RuntimeError(f"dsv41-pd032 LEAN: layer {i}'s attention returned {type(o).__name__}, not [n, hidden]")
        plan.oy.copy_(o)
        del o, x, x_quant
        self.graph_a[(i, plan.rows)].replay()
        with get_forward().scoped(mlp_reduce_scatter=False, defer_moe_finalize=False):  # MoeRowTransport.scope()
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
            _PLAN[0] = None  # the stock loop takes the handoff from here (layer 20's tail switch)
        return residual_batch.set_written(state, batch)

    def _moe(self, view, plan, input_ids, input_ids_global):
        """The MoE at the live rows: its output in OY[:n] (graph C's input)."""
        if not MOE_REPLICA:
            y = view.mlp(plan.x, plan.batch, input_ids=input_ids, input_ids_global=input_ids_global,
                         skip_shared_experts=False, return_moe_output=False, leave_output_sum=True)
            return plan.oy.copy_(y)
        return self._moe_replica(view, plan, input_ids_global)

    def _moe_replica(self, view, plan, input_ids_global):
        """moe_forward.forward -> mega_forward.forward_mega_moe / _run_mega_routed / run_mega_routed_experts on this
        domain (EXTEND, no mega gate, fused shared expert, fp8xfp4, eager, no decode tiles): the same router GEMM,
        MEGA_FRONT launch and MegaMoE call with the same global token hint and shared block_m, MegaMoE writing OY[:n]
        instead of a fresh tensor. What does not change between the layers of a forward (stock's MegaMoE predicate,
        the hint, block_m) is decided at the forward's first MoE, where stock's first MegaMoE call also creates the
        shared symmetric buffer; covered()'s static facts are checked once a layer, its dynamic ones by `_plan`."""
        from dsv41_pd001 import prefill_gemm

        with prefill_gemm.moe_tiles(plan.batch):  # dsv41-pd046 PGEMM v3: the tile band on lean ranks too
            moe, x, n = view.mlp, plan.x, plan.n
            if plan.moe is None:
                if not should_use_mega_moe(moe, x):
                    raise RuntimeError("dsv41-pd032 LEAN: stock would not run MegaMoE on this forward")
                if self.moe_buf is None:
                    self.moe_buf = _mega_moe_buffer(moe)  # the one symmetric buffer every MoE layer shares (stock's cache)
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
                if not mega_forward._pd001_front_covered(moe, logits, x):
                    raise RuntimeError(f"dsv41-pd032 LEAN: MEGA_FRONT does not serve layer {view.layer_id} here")
                self.moe_covered.add(view.layer_id)
            elif logits.dtype is not torch.float32 or logits.shape[0] != n:
                raise RuntimeError(f"dsv41-pd032 LEAN: layer {view.layer_id}'s router logits are not fp32 [n, experts]")
            mega_forward._pd001_front((moe, logits, input_ids_global), x, buf, self.moe_rsf[view.layer_id],
                                      buf.shared_l1_acts_sf, block_m)
            y = plan.oy  # stock allocates max(n, 1) rows; n >= LO here
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
    """An early layer view's forward: the lean path inside this forward's plan, stock's forward otherwise. Layer 0
    (called first on every rank in every forward, IDLE included) decides the plan."""
    if self.layer_id == 0:
        _PLAN[0] = _plan(forward_batch, input_ids)
    plan = _PLAN[0]
    if plan is None or plan.batch is not forward_batch:
        return type(self).forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global,
                                  allow_cross_layer_fusion=allow_cross_layer_fusion)
    return plan.lean.layer(self, plan, positions, hidden_states, input_ids, input_ids_global,
                           allow_cross_layer_fusion)


def _plan(batch, input_ids) -> Optional[_Plan]:
    """This rank's plan when the forward is in the domain, else None (pd034's path). Rank-local by design: both paths
    run the same collectives."""
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
    """Install the early layer views' forward and capture every graph, except in a PD decode engine (which runs no
    small EXTEND forwards and keeps pd034's path). Runs inside cacheon's first dispatched call, before serving."""
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        logger.warning("dsv41-pd032 LEAN: PD decode engine, no early-layer graphs")
        return
    if _LEAN or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("dsv41-pd032 LEAN: prepare ran twice or inside a CUDA graph capture")
    lean = _Lean(view, module, sk_impl)
    lean.capture()
    _LEAN.append(lean)
    for layer_view in view.layers[:EARLY]:
        layer_view.forward = MethodType(layer_forward, layer_view)
