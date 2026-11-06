from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache

import torch

CHECK = False
CHECK_CALLS = 3
PREFIX = "bundle-v9-check" if CHECK else "bundle-v9"
H_Q, D_QK = 64, 512
TOPKS = (128, 640)
ROWS_MIN = 8192

_STOCK: dict = {}
_ON = [False]
_CHK: dict = {}


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


@lru_cache(maxsize=None)
def source_root() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9.dec import cuda_prefill_attn

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-prefill-attn", cuda_prefill_attn.TAG)
    for rel, text in cuda_prefill_attn.SOURCES.items():
        path = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            _write(path, text.encode())
    return directory


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from bundle_v9.dec import cuda_prefill_attn

    directory = source_root()
    return load_jit("dec_prefill_attn", cuda_prefill_attn.TAG, cuda_files=[os.path.join(directory, cuda_prefill_attn.ENTRY)],
                    cuda_wrappers=list(cuda_prefill_attn.WRAPPERS), extra_cuda_cflags=list(cuda_prefill_attn.CUDA_FLAGS),
                    extra_ldflags=["-lcuda"], extra_include_paths=[directory, os.path.join(directory, "kerutils_include")],
                    extra_dependencies=["cutlass"])


def _graph_scope() -> bool:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.models.deepseek_v4_replay_tail import in_replay_tail_graph

    return torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph() or in_replay_tail_graph()


def serves(q, kv, indices, positions, cos_sin_cache, attn_sink, topk_length, enable_q_norm, use_legacy) -> bool:
    return (
        _ON[0]
        and use_legacy is True
        and not enable_q_norm
        and q.dim() == 3
        and q.shape[0] >= ROWS_MIN
        and q.shape[1] == H_Q
        and q.shape[2] == D_QK
        and q.dtype == torch.bfloat16
        and q.stride(2) == 1
        and q.stride(1) == D_QK
        and kv.dim() == 3
        and kv.shape[0] > 0
        and kv.shape[1] == 1
        and kv.shape[2] == D_QK
        and kv.dtype == torch.bfloat16
        and kv.stride(2) == 1
        and indices.dim() == 3
        and indices.shape[0] == q.shape[0]
        and indices.shape[1] == 1
        and indices.shape[2] in TOPKS
        and indices.dtype == torch.int32
        and indices.stride(2) == 1
        and topk_length is not None
        and topk_length.dtype == torch.int32
        and topk_length.shape == (q.shape[0],)
        and topk_length.is_contiguous()
        and positions.dtype == torch.int32
        and positions.shape == (q.shape[0],)
        and positions.is_contiguous()
        and attn_sink is not None
        and attn_sink.dtype == torch.float32
        and attn_sink.shape == (H_Q,)
        and attn_sink.is_contiguous()
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.is_contiguous()
        and q.is_cuda
        and all(t.device == q.device for t in (kv, indices, topk_length, positions, attn_sink, cos_sin_cache))
        and not _graph_scope()
    )


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    x, y = a.contiguous().view(torch.uint8).reshape(-1), b.contiguous().view(torch.uint8).reshape(-1)
    step = 1 << 24
    return sum(int((x[i:i + step] != y[i:i + step]).sum()) for i in range(0, x.numel(), step))


def _checked(rows: int, ours, stock) -> None:
    key = max(int(rows), 1).bit_length()
    n = _CHK.get(key, 0)
    if n >= CHECK_CALLS:
        return
    _CHK[key] = n + 1
    ref = stock()
    diffs = [_differ(a, b) for a, b in zip(ours, ref)]
    ok = len(ours) == len(ref) and all(d == 0 for d in diffs)
    print(f"{PREFIX} PREFILL_ATTN gpu{torch.cuda.current_device()} rows {rows}: {'OK' if ok else 'MISMATCH'} bytes-differ {diffs}",
          file=sys.stderr, flush=True)


def mega_prefill(q, kv, indices, positions, cos_sin_cache, softmax_scale, *, attn_sink=None, topk_length=None,
                 enable_q_norm=False, rms_norm_eps=0.0, use_legacy=True):
    stock = _STOCK["flash_mla_mega_prefill"]

    def run_stock():
        return stock(q, kv, indices, positions, cos_sin_cache, softmax_scale, attn_sink=attn_sink,
                     topk_length=topk_length, enable_q_norm=enable_q_norm, rms_norm_eps=rms_norm_eps,
                     use_legacy=use_legacy)

    if not serves(q, kv, indices, positions, cos_sin_cache, attn_sink, topk_length, enable_q_norm, use_legacy):
        return run_stock()
    s_q = q.shape[0]
    aligned = (s_q + 3) // 4 * 4
    out_fp8 = torch.empty((s_q, 8, 4096), dtype=torch.float8_e4m3fn, device=q.device)
    out_sf = torch.empty((8, 32, aligned), dtype=torch.int32, device=q.device).transpose(1, 2)[:, :s_q].transpose(0, 1)
    max_logits = torch.empty((s_q, H_Q), dtype=torch.float32, device=q.device)
    lse = torch.empty((s_q, H_Q), dtype=torch.float32, device=q.device)
    module().prefill(q, kv, indices, topk_length, attn_sink, positions, cos_sin_cache, out_fp8, out_sf, max_logits, lse,
                     float(softmax_scale), float(rms_norm_eps))
    out = (out_fp8, out_sf, max_logits, lse)
    if CHECK:
        _checked(s_q, out, run_stock)
    return out


def prepare(model) -> None:
    from sglang.kernels.ops.attention import flash_mla_interface
    from sglang.srt.runtime_context import get_exec, get_parallel

    if not _STOCK:
        fn = flash_mla_interface.flash_mla_mega_prefill
        if fn.__module__ != "sglang.kernels.ops.attention.flash_mla_interface" or fn.__qualname__ != "flash_mla_mega_prefill":
            raise RuntimeError(f"bundle-v9 PREFILL_ATTN: flash_mla_mega_prefill is not stock's ({fn.__module__}.{fn.__qualname__})")
        _STOCK["flash_mla_mega_prefill"] = fn
    backend = get_exec().kernel.dsv4_attn_backend
    if backend != "flashmla_mega":
        raise RuntimeError(f"bundle-v9 PREFILL_ATTN requires the flashmla_mega attention backend, got {backend}")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(f"bundle-v9 PREFILL_ATTN is built and proven for sm_103 (B300), got {torch.cuda.get_device_capability()}")
    heads = model.config.num_attention_heads // get_parallel().attn_tp_size
    if heads != H_Q:
        raise RuntimeError(f"bundle-v9 PREFILL_ATTN requires {H_Q} attention heads a rank, got {heads}")
    module()
    if CHECK:
        print(f"{PREFIX} PREFILL_ATTN prepared: rows >= {ROWS_MIN}, topk {TOPKS}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def routed(view, forward_batch):
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend_without_speculative() or torch.cuda.is_current_stream_capturing() or not _STOCK:
        yield
        return
    from sglang.kernels.ops.attention import flash_mla_interface

    _ON[0] = True
    try:
        flash_mla_interface.flash_mla_mega_prefill = mega_prefill
        yield
    finally:
        _ON[0] = False
        flash_mla_interface.flash_mla_mega_prefill = _STOCK["flash_mla_mega_prefill"]
