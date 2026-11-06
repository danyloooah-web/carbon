
from __future__ import annotations

import ast
import hashlib
import inspect

COMMIT = "15338a8e6455d661268553c1efc7241bcb2f982f"

FRONTFIX_SEGMENTS = (
    ("deepseek_v2", "DeepseekV2MoE.forward", "f3a64ad5dbad841b"),
    ("deepseek_v2", "MoEGate.forward", "7fc1c304c885345a"),
    ("mega_moe", "forward_mega_moe", "55b0fc1607126ded"),
    ("mega_moe", "_run_mega_routed", "70c7016ac2dd4ece"),
    ("mega_moe", "run_mega_routed_experts", "809828b1c3cc6de6"),
    ("mega_moe", "_mega_moe_buffer", "9dabc8eb5d53f8d5"),
    ("vl_routing", "vision_topk", "37e033a058db75d1"),
    ("vl_routing", "_finish_vision_topk", "444aa0b1756b9df0"),
    ("topk", "TopK.routing_inputs", "87e2d7efd9e0aafb"),
    ("topk", "resolve_routing_inputs", "7eee807c53b7e0a4"),
    ("moe_fused_gate", "moe_fused_gate", "96fe4e49add84457"),
    ("moe_fused_gate", "_router_triton_kernel", "b86b6e50d103be7f"),
    ("dsv4_moe", "mega_moe_pre_dispatch", "991bccd96cf33865"),
)

EGP_SEGMENTS = (
    ("engram", "EngramEmbedding.__init__", "2f6cde7545a375b7"),
    ("engram", "EngramEmbedding.forward", "68f791628981ea39"),
    ("engram", "EngramEmbedding._owned_rows", "47211cf1e657646c"),
    ("engram", "EngramEmbedding._dp_sharded_lookup", "af2f31382bda6ce8"),
    ("engram", "Engram.forward", "68cd442bb74b9e82"),
    ("engram", "Engram.embed_rows", "2b4088ed4ba1496c"),
)

EGP_CALLABLES = (
    ("engram_gather_peer", "c0df68dbac7865b8"),
    ("_engram_gather_peer_kernel", "b9038cf0b84cae9f"),
)

DENSE_SEGMENTS = (
    ("fp8", "Fp8LinearMethod.__init__", "0e940c3f3e87544f"),
    ("fp8", "Fp8LinearMethod.apply", "ece80e7de8e5b1dc"),
    ("fp8", "Fp8LinearMethod.process_weights_after_loading", "96a9880eb6305321"),
    ("fp8", "Fp8LinearMethod.prepare_deepgemm_mxfp8_input", "dd6e889e319c8250"),
    ("fp8", "Fp8LinearMethod._prepare_block_fp8_as_mxfp8", "41d031aa858c50e1"),
    ("fp8", "Fp8LinearMethod._process_mxfp8_linear_weight_scale", "2f1259bbada08375"),
    ("fp8_utils", "deepgemm_w8a8_mxfp8_linear_with_fallback", "4bbbde15b0da0650"),
    ("fp8_utils", "flashinfer_mxfp8_blockscaled_linear", "44d063202cb3947e"),
    ("fp8_utils", "dispatch_block_fp8_mxfp8_linear", "775393f38782a76f"),
    ("fp8_utils", "pack_mxfp8_weight_scale_for_deepgemm", "a5fbd319cef10a8d"),
    ("fp8_kernel", "sglang_per_token_group_quant_fp8", "98427d7ec7223970"),
    ("mxfp8_dispatch_cache", "maybe_cache_mxfp8_dispatch", "8dd784351f098dc0"),
    ("mxfp8_dispatch_cache", "Mxfp8DispatchCache.__call__", "e695fdc9b907c29e"),
)


def _definition(tree: ast.AST, qualname: str):
    node = tree
    for part in qualname.split("."):
        body = getattr(node, "body", ())
        found = [
            child
            for child in body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part
        ]
        if not found:
            return None
        node = found[-1]
    return node


def _segment_sha16(module, qualname: str, cache: dict) -> str:
    hit = cache.get(module.__name__)
    if hit is None:
        src = inspect.getsource(module)
        hit = cache[module.__name__] = (src, ast.parse(src))
    src, tree = hit
    node = _definition(tree, qualname)
    if node is None:
        return "missing"
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    seg = "".join(src.splitlines(keepends=True)[start - 1 : node.end_lineno])
    return hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]


def _verify_segments(group: str, modules: dict, specs: tuple) -> None:
    cache: dict = {}
    bad = []
    for mod_key, qualname, expected in specs:
        mod = modules[mod_key]
        try:
            got = _segment_sha16(mod, qualname, cache)
        except (OSError, TypeError, SyntaxError) as exc:
            got = f"err:{type(exc).__name__}"
        if got != expected:
            bad.append(f"{mod.__name__}:{qualname}={got}!={expected}")
    if bad:
        raise RuntimeError(f"bundle-v9 {group} source-pin mismatch against {COMMIT[:12]}: {', '.join(bad)}")


def verify_frontfix() -> None:
    import sglang.kernels.ops.moe.dsv4 as dsv4_moe
    import sglang.kernels.ops.moe.moe_fused_gate as moe_fused_gate
    import sglang.srt.layers.moe.mega_moe as mega_moe
    import sglang.srt.layers.moe.topk as topk
    import sglang.srt.models.deepseek_v2 as deepseek_v2
    import sglang.srt.multimodal.dsv41.vl_routing as vl_routing

    _verify_segments(
        "FRONTFIX",
        {
            "deepseek_v2": deepseek_v2,
            "mega_moe": mega_moe,
            "vl_routing": vl_routing,
            "topk": topk,
            "moe_fused_gate": moe_fused_gate,
            "dsv4_moe": dsv4_moe,
        },
        FRONTFIX_SEGMENTS,
    )


def verify_egp() -> None:
    import sglang.kernels.ops.embeddings.engram_gather as egk
    import sglang.srt.layers.engram as engram

    _verify_segments("PEER", {"engram": engram}, EGP_SEGMENTS)
    fns = {
        "engram_gather_peer": egk.engram_gather_peer,
        "_engram_gather_peer_kernel": egk._engram_gather_peer_kernel.fn,
    }
    bad = []
    for name, expected in EGP_CALLABLES:
        try:
            got = hashlib.sha256(inspect.getsource(fns[name]).encode("utf-8")).hexdigest()[:16]
        except (OSError, TypeError) as exc:
            got = f"err:{type(exc).__name__}"
        if got != expected:
            bad.append(f"{name}={got}!={expected}")
    if bad:
        raise RuntimeError(f"bundle-v9 PEER kernel-pin mismatch against {COMMIT[:12]}: {', '.join(bad)}")


def verify_mxt() -> None:
    import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel
    import sglang.srt.layers.quantization.fp8 as fp8
    import sglang.srt.layers.quantization.fp8_utils as fp8_utils
    import sglang.srt.layers.quantization.mxfp8_dispatch_cache as mxfp8_dispatch_cache

    _verify_segments(
        "DENSE",
        {
            "fp8": fp8,
            "fp8_utils": fp8_utils,
            "fp8_kernel": fp8_kernel,
            "mxfp8_dispatch_cache": mxfp8_dispatch_cache,
        },
        DENSE_SEGMENTS,
    )
