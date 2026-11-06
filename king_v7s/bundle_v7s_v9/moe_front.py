
from __future__ import annotations

import dataclasses
import hashlib
import os
from typing import Optional

import torch
import torch.distributed as dist

from bundle_v9 import cuda_moe_front, tactic_pins

HIDDEN = 5120
EXPERTS = 384
TOP_K = 6
V2_MIN_ROWS = 4096
PROBE_CELLS = ((1, 64), (6, 64), (48, 64), (144, 64), (288, 64), (4096, 240), (4097, 192), (16383, 128),
               (32768, 240))
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
        directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9",
                                 hashlib.sha256(data).hexdigest()[:16])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, os.path.basename(cuda_moe_front.NAME))
        if not os.path.isfile(path):
            _write(path, data)
        _paths[cuda_moe_front.NAME] = path
    return path


def _module(experts: int, top_k: int, hidden: int, renorm: bool, scale: bool, shared_sf: bool, has_offset: bool):
    key = (experts, top_k, hidden, renorm, scale, shared_sf, has_offset)
    module = _modules.get(key)
    if module is None:
        from sglang.kernels.jit.utils import load_jit, make_cpp_args

        args = make_cpp_args(experts, top_k, hidden, renorm, scale, shared_sf, has_offset)
        wrappers = [("run", f"sglang::v4_mega::MegaFront<{args}>::run")]
        if not has_offset:
            v2_args = make_cpp_args(experts, top_k, hidden, renorm, scale, shared_sf)
            wrappers.append(("run_v2", f"sglang::v4_mega::MegaFrontV2<{v2_args}>::run"))
        module = _modules[key] = load_jit(
            "v4_moe_front",
            *args,
            cuda_files=[_header()],
            cuda_wrappers=wrappers,
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
    row_offset: int = 0,
    pad: bool = True,
) -> None:
    rows, experts = logits.shape
    scale = float(routed_scaling_factor if routed_scaling_factor is not None else 1.0)
    has_offset = not (row_offset == 0 and pad)
    module = _module(experts, buf_topk_idx.shape[1], x.shape[-1], bool(renormalize), scale != 1.0,
                     shared_x_sf is not None, has_offset)
    args = (
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
    if rows >= V2_MIN_ROWS and not has_offset:
        module.run_v2(*args)
    else:
        module.run(*args, int(row_offset), bool(pad))


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
                      use_pdl: bool = True, row_offset: int = 0, pad: bool = True, live_all: bool = False) -> None:
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
        num_token_non_padded=None if live_all else bound,
        renormalize=config.renormalize and config.top_k > 1,
        renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
        routed_scaling_factor=routed_scaling_factor,
        shared_x_sf=shared_x_sf,
        shared_block_m=shared_block_m,
        use_pdl=use_pdl,
        row_offset=row_offset,
        pad=pad,
    )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


@torch.no_grad()
def probe_frontfix(module, device: torch.device) -> int:
    from sglang.kernels.ops.moe import dsv4 as dsv4_moe
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON
    from sglang.srt.model_executor import forward_context as fc
    from sglang.srt.multimodal.dsv41 import vl_routing
    from sglang.srt.runtime_context import get_parallel

    tactic_pins.verify_frontfix()
    group = get_parallel().tp_group
    rank = int(group.rank_in_group)
    prev = fc.get_forward_context() if fc.has_forward_context() else None
    if prev is not None:
        fc.set_forward_context(dataclasses.replace(prev, moe_num_token_non_padded=None))

    bad: list[str] = []
    cells = 0
    try:
        sample = [module.layers[0].mlp, module.layers[-1].mlp] if len(module.layers) > 1 else [module.layers[0].mlp]
        for mlp in sample:
            lid = int(mlp.layer_id)
            cfg = mlp.topk.topk_config
            renorm = bool(cfg.renormalize and cfg.top_k > 1)
            rsf = 1.0 if mlp.routed_scaling_factor_fused_in_topk else float(mlp.routed_scaling_factor)
            has_ssf = mlp.mega_shared_l1_weights is not None
            img_id = int(mlp.config.image_token_id)
            for rows, sbm in PROBE_CELLS:
                cap = (rows + 23) // 4 * 4
                srows = ((rows + sbm - 1) // sbm) * ((sbm + 127) // 128 * 128) + 128
                for padded in (False, True):
                    g = torch.Generator().manual_seed(1_000_033 * (lid + 1) + 8191 * rows + 37 * int(padded) + rank)
                    x = (torch.randn(rows, HIDDEN, generator=g) * 0.75).to(torch.bfloat16).to(device)
                    logits = (torch.randn(rows, EXPERTS, generator=g, dtype=torch.float32) * 0.5).to(device)
                    ids = torch.randint(0, 1 << 15, (rows,), generator=g, dtype=torch.int64)
                    if img_id >= 0:
                        ids[::3] = img_id
                    ids = ids.to(device)
                    bound = torch.tensor([max(rows - rows // 3, 1)], dtype=torch.int32, device=device) if padded else None

                    def _alloc():
                        bx = torch.zeros((cap, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
                        xsf_full = torch.full((cap, 44), -1, dtype=torch.int32, device=device)
                        tidx = torch.full((cap, TOP_K), -99, dtype=torch.int64, device=device)
                        tw = torch.full((cap, TOP_K), -9.0, dtype=torch.float32, device=device)
                        ssf = torch.zeros((40, srows), dtype=torch.int32, device=device).t() if has_ssf else None
                        return bx, xsf_full[:, :40], xsf_full, tidx, tw, ssf

                    bx_s, xsf_s, xsf_s_full, tidx_s, tw_s, ssf_s = _alloc()
                    out = vl_routing.vision_topk(mlp, logits, ids, num_token_non_padded=bound)
                    w_in = out.topk_weights.to(torch.float32)
                    if rsf != 1.0:
                        w_in = w_in * rsf
                    dsv4_moe.mega_moe_pre_dispatch(
                        x, out.topk_ids.to(torch.int32), w_in, bx_s, xsf_s, tidx_s, tw_s,
                        quant_group_size=32, shared_x_sf=ssf_s, shared_block_m=(sbm if has_ssf else 0), copy_topk=True,
                    )

                    bx_m, xsf_m, xsf_m_full, tidx_m, tw_m, ssf_m = _alloc()
                    moe_front(
                        logits, mlp.gate.e_score_correction_bias, x, bx_m, xsf_m, tidx_m, tw_m,
                        bias_alt=mlp.gate.e_score_correction_bias_vl, input_ids=ids, image_token_id=img_id,
                        num_token_non_padded=bound, renormalize=renorm, renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
                        routed_scaling_factor=rsf, shared_x_sf=ssf_m, shared_block_m=(sbm if has_ssf else 0),
                        use_pdl=False, row_offset=0, pad=True,
                    )
                    tag = f"L{lid} r={rows} pad={int(padded)}"
                    if not _same_bits(bx_s[:rows], bx_m[:rows]):
                        bad.append(f"{tag}:x")
                    if not _same_bits(xsf_s_full[:rows], xsf_m_full[:rows]):
                        bad.append(f"{tag}:x_sf")
                    if not _same_bits(tidx_s, tidx_m):
                        bad.append(f"{tag}:topk_idx")
                    if not _same_bits(tw_s, tw_m):
                        bad.append(f"{tag}:topk_w")
                    if has_ssf and not _same_bits(ssf_s, ssf_m):
                        bad.append(f"{tag}:shared_sf")
                    cells += 1
                    if bad:
                        break
                if bad:
                    break
            if bad:
                break
    except Exception as exc:
        bad.append(f"{type(exc).__name__}: {exc}")
    finally:
        if prev is not None:
            fc.set_forward_context(prev)

    gathered = [None] * group.world_size
    dist.all_gather_object(gathered, bad[:3], group=group.cpu_group)
    failures = [f"rank {r}: {', '.join(errs)}" for r, errs in enumerate(gathered) if errs]
    if failures:
        raise RuntimeError(f"bundle-v9 FRONTFIX STRICT byte-probe failed ({' | '.join(failures)})")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return cells
