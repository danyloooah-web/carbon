from __future__ import annotations

import contextlib
import hashlib
import os
from functools import lru_cache

import torch

CUDA_FLAGS = ("--expt-extended-lambda", "--use_fast_math", "-DNDEBUG", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "--diag-suppress=445",
              "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")
WRAPPERS = [("decode_fp4", "dattn_entry::MegaDecodeFp4::run")]
NAMESPACE = "dattn"

_STOCK: dict = {}


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


def sources() -> dict:
    from bundle_v9.dec import cuda_prefill_attn, cuda_decode_attn

    out = {rel.replace("pattn", NAMESPACE): text.replace("pattn", NAMESPACE)
           for rel, text in cuda_prefill_attn.SOURCES.items() if rel != cuda_prefill_attn.ENTRY}
    out[cuda_decode_attn.ENTRY] = cuda_decode_attn.SOURCE.replace("pattn", NAMESPACE)
    return out


@lru_cache(maxsize=None)
def source_root() -> tuple:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    tree = sources()
    h = hashlib.sha256()
    for rel in sorted(tree):
        h.update(rel.encode())
        h.update(tree[rel].encode())
    digest = h.hexdigest()[:16]
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-decode-attn", digest)
    for rel, text in tree.items():
        path = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            _write(path, text.encode())
    return directory, digest


def wheel_nvcc() -> str:
    import nvidia

    for root in list(nvidia.__path__):
        path = os.path.join(root, "cu13", "bin", "nvcc")
        if os.path.isfile(path):
            versions = [name[len("nvidia_cuda_nvcc-"):-len(".dist-info")] for name in os.listdir(os.path.dirname(root))
                        if name.startswith("nvidia_cuda_nvcc-") and name.endswith(".dist-info")]
            if len(versions) != 1 or tuple(int(x) for x in versions[0].split(".")[:2]) < (13, 2):
                raise RuntimeError(f"bundle-v9 DECODE_ATTN: {path} belongs to nvidia-cuda-nvcc {versions}; 13.2+ is required")
            return path
    raise RuntimeError("bundle-v9 DECODE_ATTN: no nvidia/cu13/bin/nvcc (the nvidia-cuda-nvcc wheel) next to the nvidia package")


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile import toolchain

    from bundle_v9.dec import cuda_decode_attn

    nvcc = wheel_nvcc()
    directory, digest = source_root()
    stock = toolchain.device_compiler_path
    toolchain.device_compiler_path = lambda: nvcc
    try:
        return load_jit("dec_decode_attn", digest, cuda_files=[os.path.join(directory, cuda_decode_attn.ENTRY)], cuda_wrappers=WRAPPERS,
                        extra_cuda_cflags=list(CUDA_FLAGS), extra_ldflags=["-lcuda"],
                        extra_include_paths=[directory, os.path.join(directory, "kerutils_include")],
                        extra_dependencies=["cutlass"])
    finally:
        toolchain.device_compiler_path = stock


def decode_like() -> bool:
    from sglang.srt.model_executor.runner_utils.prefill_graph import is_in_prefill_graph_scope
    from sglang.srt.runtime_context import get_forward

    fwd = get_forward()
    return not (fwd.sp_active or fwd.is_extend_in_batch or is_in_prefill_graph_scope())


def serves(q, kv, extra_kv, attn_sink, topk_length, extra_indices, extra_topk_length, enable_q_norm, use_legacy) -> bool:
    return (
        use_legacy
        and not enable_q_norm
        and extra_kv is not None
        and extra_kv.shape[-1] == 288
        and kv.shape[-1] == 528
        and q.dim() == 3
        and q.shape[0] > 0
        and q.shape[1] == 64
        and q.shape[2] == 512
        and attn_sink is not None
        and topk_length is not None
        and extra_indices is not None
        and extra_topk_length is not None
        and decode_like()
    )


def mega_decode(q, kv, indices, positions, cos_sin_cache, softmax_scale, *, attn_sink=None, topk_length=None, extra_kv=None, extra_indices=None,
                extra_topk_length=None, enable_q_norm=False, rms_norm_eps=0.0, use_legacy=True):
    if not serves(q, kv, extra_kv, attn_sink, topk_length, extra_indices, extra_topk_length, enable_q_norm, use_legacy):
        return _STOCK["flash_mla_mega_decode"](q, kv, indices, positions, cos_sin_cache, softmax_scale, attn_sink=attn_sink, topk_length=topk_length,
                                               extra_kv=extra_kv, extra_indices=extra_indices, extra_topk_length=extra_topk_length,
                                               enable_q_norm=enable_q_norm, rms_norm_eps=rms_norm_eps, use_legacy=use_legacy)
    s_q = q.shape[0]
    aligned = (s_q + 3) // 4 * 4
    out_fp8 = torch.empty((s_q, 8, 4096), dtype=torch.float8_e4m3fn, device=q.device)
    out_sf = torch.empty((8, 32, aligned), dtype=torch.int32, device=q.device).transpose(1, 2)[:, :s_q].transpose(0, 1)
    lse = torch.empty((s_q, 64), dtype=torch.float32, device=q.device)
    module().decode_fp4(q, kv, indices, topk_length, extra_kv, extra_indices, extra_topk_length, attn_sink, positions, cos_sin_cache, out_fp8, out_sf, lse,
                        float(softmax_scale))
    return out_fp8, out_sf, lse


def prepare(model) -> None:
    from sglang.kernels.ops.attention import flash_mla_interface

    if not _STOCK:
        fn = flash_mla_interface.flash_mla_mega_decode
        if fn.__module__ != "sglang.kernels.ops.attention.flash_mla_interface" or fn.__qualname__ != "flash_mla_mega_decode":
            raise RuntimeError(f"bundle-v9 DECODE_ATTN: flash_mla_mega_decode is not stock's ({fn.__module__}.{fn.__qualname__})")
        _STOCK["flash_mla_mega_decode"] = fn
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(f"bundle-v9 DECODE_ATTN is built and proven for sm_103 (B300), got {torch.cuda.get_device_capability()}")
    module()


@contextlib.contextmanager
def routed():
    from sglang.kernels.ops.attention import flash_mla_interface

    flash_mla_interface.flash_mla_mega_decode = mega_decode
    try:
        yield
    finally:
        flash_mla_interface.flash_mla_mega_decode = _STOCK["flash_mla_mega_decode"]
