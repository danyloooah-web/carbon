from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch

from bundle_v9.dec import cuda_moe_front

HIDDEN = 5120
EXPERTS = 384
TOP_K = 6
V2_MIN_ROWS = 4096
_paths: dict = {}
_modules: dict = {}


def _write(path: str, data: bytes) -> None:
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


def _header() -> str:
    path = _paths.get(cuda_moe_front.NAME)
    if path is None:
        from sglang.kernels.jit.utils.compile.cache import cache_root

        data = cuda_moe_front.SOURCE.encode()
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-moe-front",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, os.path.basename(cuda_moe_front.NAME))
        if not os.path.isfile(path):
            _write(path, data)
        _paths[cuda_moe_front.NAME] = path
    return path


def _module(experts: int, top_k: int, hidden: int, renorm: bool, scale: bool, shared_sf: bool, eq: bool = False):
    key = (experts, top_k, hidden, renorm, scale, shared_sf) + ((True,) if eq else ())
    module = _modules.get(key)
    if module is None and eq:
        # king-v7 XGF-EQ: the king's v1 front text with the x-quant warps ahead of the wait (xgf.eq_source), in its
        # own namespace and JIT module; only the v1 entry is exported (the v2 kernel keeps the king's text)
        from sglang.kernels.jit.utils import load_jit, make_cpp_args

        from bundle_v9.dec import xgf

        path, digest = xgf.eq_header()
        args = make_cpp_args(experts, top_k, hidden, renorm, scale, shared_sf)
        module = _modules[key] = load_jit(
            "dec_moe_front_eq",
            *args,
            digest,
            cuda_files=[path],
            cuda_wrappers=[("run", f"sglang::dec_front_eq::MegaFront<{args}>::run")],
        )
    if module is None:
        from sglang.kernels.jit.utils import load_jit, make_cpp_args

        args = make_cpp_args(experts, top_k, hidden, renorm, scale, shared_sf)
        module = _modules[key] = load_jit(
            "dec_moe_front",
            *args,
            cuda_files=[_header()],
            cuda_wrappers=[("run", f"sglang::dec_front::MegaFront<{args}>::run"),
                           ("run_v2", f"sglang::dec_front::MegaFrontV2<{args}>::run")],
        )
    return module


def moe_front(
    logits: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    *,
    bias_alt: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: int = 0,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize: bool = True,
    renormalize_epsilon: float = 0.0,
    routed_scaling_factor: float = 1.0,
    shared_x_sf: Optional[torch.Tensor] = None,
    shared_block_m: int = 0,
    use_pdl: bool = True,
    early_quant: bool = False,
) -> None:
    rows, experts = logits.shape
    scale = float(routed_scaling_factor if routed_scaling_factor is not None else 1.0)
    eq = bool(early_quant and use_pdl and rows < V2_MIN_ROWS)  # king-v7 XGF-EQ: v1 rows only, PDL launches only
    module = _module(experts, buf_topk_idx.shape[1], x.shape[-1], bool(renormalize), scale != 1.0,
                     shared_x_sf is not None, eq)
    (module.run_v2 if rows >= V2_MIN_ROWS else module.run)(
        logits,
        bias,
        bias_alt if input_ids is not None else None,
        input_ids,
        int(image_token_id),
        num_token_non_padded,
        x,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        shared_x_sf,
        int(shared_block_m),
        scale,
        float(renormalize_epsilon),
        bool(use_pdl),
    )


def covered(moe, router_logits: torch.Tensor, hidden_states: torch.Tensor, max_rows: int) -> bool:
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
    from sglang.srt.layers.moe.mega_moe import _mega_moe_mma_type
    from sglang.srt.layers.moe.topk import routing_simulation_enabled
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.models.deepseek_common.utils import _device_sm
    from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

    config = moe.topk.topk_config
    rows = hidden_states.shape[0]
    return (
        get_moe_a2a_backend().is_megamoe()
        and _device_sm // 10 == 10
        and _mega_moe_mma_type(moe.experts) == "fp8xfp4"
        and not moe.is_hash
        and moe.gate.e_score_correction_bias_vl is not None
        and moe.gate.e_score_correction_bias.dtype == torch.float32
        and config.scoring_func == "sqrtsoftplus"
        and config.num_shared_topk_slots == 0
        and config.top_k == TOP_K
        and not config.apply_routed_scaling_factor_on_output
        and moe.num_fused_shared_experts == 0
        and not routing_simulation_enabled()
        and get_global_experts_capturer() is None
        and not get_global_expert_distribution_recorder().recording
        and 0 < rows <= max_rows
        and router_logits.dtype == torch.float32
        and tuple(router_logits.shape) == (rows, EXPERTS)
        and hidden_states.dtype == torch.bfloat16
        and tuple(hidden_states.shape) == (rows, HIDDEN)
        and hidden_states.is_contiguous()
    )


def mega_routed_front(moe, router_logits: torch.Tensor, input_ids: Optional[torch.Tensor],
                      hidden_states: torch.Tensor, buf, *, routed_scaling_factor: float, shared_x_sf, shared_block_m: int,
                      use_pdl: bool = True, early_quant: bool = False) -> None:
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    bound, dispatch_info = moe.topk.routing_inputs()
    if dispatch_info is not None:
        raise RuntimeError(f"mega_routed_front: MoE layer {moe.layer_id} has an expert-location dispatch (EPLB); "
                           "its remap is outside this kernel")
    if input_ids is not None and tuple(input_ids.shape) != (router_logits.shape[0],):
        raise RuntimeError("mega_routed_front: input_ids must be [rows] (moe_fused_gate's contract)")
    config = moe.topk.topk_config
    moe_front(
        router_logits,
        moe.gate.e_score_correction_bias,
        hidden_states,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        bias_alt=moe.gate.e_score_correction_bias_vl,
        input_ids=input_ids,
        image_token_id=moe.config.image_token_id,
        num_token_non_padded=bound,
        renormalize=config.renormalize and config.top_k > 1,
        renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
        routed_scaling_factor=routed_scaling_factor,
        shared_x_sf=shared_x_sf,
        shared_block_m=shared_block_m,
        use_pdl=use_pdl,
        early_quant=early_quant,
    )
