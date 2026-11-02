"""dsv41-pd055 FMLAP: the cold prefill forward's MegaAttention core on pd007's FlashMLA "grp" kernel (cuda_fmlap.py;
generator/pd055, the kernel port and its first benches in generator/pd007/attn).

Inside the candidate's own extend forwards (`routed`, next to SKFUSE's; never while a CUDA graph captures, and never in
a breakable-graph or replay-tail scope) `flash_mla_interface.flash_mla_mega_prefill` -- the function
DeepseekV4MegaAttnBackend._forward_mega_prefill looks up per call for every eager sparse-prefill layer -- runs the image
kernel's own source (sgl-project/FlashMLA v1.0.0 csrc_legacy core_attn) with deepseek-ai/FlashMLA #227's H64 items and
shared PV completion (G + R) and sgl-project/FlashMLA #21's predicated index loads (P), built at prepare by SGLang's
load_jit (CUDA_HOME's nvcc, the image's CUTLASS). Those items reorder waits, loads and barrier arrivals only: out_fp8,
out_sf, max_logits and lse are the image kernel's bytes (generator/pd055/bench: cold-round shapes, ragged, partial and
adversarial inputs).

Domain, decided per call: use_legacy, no q norm, q [s_q >= ROWS_MIN, 64, 512] bf16 contiguous within a token, the
[s_kv, 1, 512] bf16 workspace, int32 indices [s_q, 1, topk] with topk in TOPKS, int32 topk_length and positions, fp32
attn_sink [64] and cos_sin_cache [*, 64], all on q's device. Every other call runs stock's function unchanged; inside
the domain the kernel runs or raises. CHECK (the twin, never submitted) also runs stock on the first calls of each row
bucket and compares every output byte under "dsv41-pd055chk"; its timings would not be evidence, so it prints none."""

from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache

import torch

CHECK = False  # gen_pd055 check: the twin's bitwise comparisons (measurement only, never submit)
CHECK_CALLS = 3  # per row bucket
PREFIX = "dsv41-pd055chk" if CHECK else "dsv41-pd055"
H_Q, D_QK = 64, 512
# combined_topk_width(0, 128) and (512, 128): the SWA-only layers' window rows and the compressed layers' [top-k | window]
TOPKS = (128, 640)
# q rows a call. Above LEAN's 4,095-row ceiling and the <= 8K-row band where prefill forwards were host-bound: a served
# call costs ~5 us more host time than stock's (bench host: 25.0 vs 20.1 us) and saves >= 18 us of kernel time from 8K
# rows (bench ladder, GEMM heat); smaller calls stay stock
ROWS_MIN = 8192

_STOCK: dict = {}
_ON = [False]
_CHK: dict = {}


# ---- the JIT module --------------------------------------------------------------------------------------------------
def _write(path: str, data: bytes) -> None:
    """Publish ``data`` at ``path`` by rename, so a reader never sees a partial file (as skfuse._write)."""
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
    """Every kernel source under <JIT cache root>/cacheon-dsv41-pd055/<TAG>/, written on first use."""
    from sglang.kernels.jit.utils.compile.cache import cache_root

    from dsv41_pd001 import cuda_fmlap

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "cacheon-dsv41-pd055", cuda_fmlap.TAG)
    for rel, text in cuda_fmlap.SOURCES.items():
        path = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            _write(path, text.encode())
    return directory


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from dsv41_pd001 import cuda_fmlap

    directory = source_root()
    return load_jit("pd055_fmlap", cuda_fmlap.TAG, cuda_files=[os.path.join(directory, cuda_fmlap.ENTRY)],
                    cuda_wrappers=list(cuda_fmlap.WRAPPERS), extra_cuda_cflags=list(cuda_fmlap.CUDA_FLAGS),
                    extra_ldflags=["-lcuda"], extra_include_paths=[directory, os.path.join(directory, "kerutils_include")],
                    extra_dependencies=["cutlass"])


# ---- the replacement -------------------------------------------------------------------------------------------------
def _graph_scope() -> bool:
    """A CUDA graph captures, or a breakable-prefill / replay-tail graph scope is open: stock's kernel is what runs."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.models.deepseek_v4_replay_tail import in_replay_tail_graph

    return torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph() or in_replay_tail_graph()


def serves(q, kv, indices, positions, cos_sin_cache, attn_sink, topk_length, enable_q_norm, use_legacy) -> bool:
    """The domain (module docstring); the cheap row test first, so small calls leave at once."""
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
    """Differing bytes of two equally shaped tensors (their logical elements; -1 when shapes or dtypes differ)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    # 16 MiB slices: a whole-tensor bool sum promotes to int64 (8 B an element) and ran the twin out of memory on a
    # 32K-row out_fp8 (chain_k55 k55t, 10-29)
    x, y = a.contiguous().view(torch.uint8).reshape(-1), b.contiguous().view(torch.uint8).reshape(-1)
    step = 1 << 24
    return sum(int((x[i:i + step] != y[i:i + step]).sum()) for i in range(0, x.numel(), step))


def _checked(rows: int, ours, stock) -> None:
    """Twin: on the first CHECK_CALLS calls of a row bucket run stock (its own outputs) and compare every byte."""
    key = max(int(rows), 1).bit_length()
    n = _CHK.get(key, 0)
    if n >= CHECK_CALLS:
        return
    _CHK[key] = n + 1
    ref = stock()
    diffs = [_differ(a, b) for a, b in zip(ours, ref)]
    ok = len(ours) == len(ref) and all(d == 0 for d in diffs)
    print(f"{PREFIX} FMLAP gpu{torch.cuda.current_device()} rows {rows}: {'OK' if ok else 'MISMATCH'} bytes-differ {diffs}",
          file=sys.stderr, flush=True)


def mega_prefill(q, kv, indices, positions, cos_sin_cache, softmax_scale, *, attn_sink=None, topk_length=None,
                 enable_q_norm=False, rms_norm_eps=0.0, use_legacy=True):
    """flash_mla_interface.flash_mla_mega_prefill: (out_fp8, out_sf, max_logits, lse), FMLAP inside the domain."""
    stock = _STOCK["flash_mla_mega_prefill"]

    def run_stock():
        return stock(q, kv, indices, positions, cos_sin_cache, softmax_scale, attn_sink=attn_sink,
                     topk_length=topk_length, enable_q_norm=enable_q_norm, rms_norm_eps=rms_norm_eps,
                     use_legacy=use_legacy)

    if not serves(q, kv, indices, positions, cos_sin_cache, attn_sink, topk_length, enable_q_norm, use_legacy):
        return run_stock()
    s_q = q.shape[0]
    aligned = (s_q + 3) // 4 * 4
    # The legacy API's allocations, in its order: out_sf is [s_q, 8, 32] over int32 storage [8, 32, aligned rows]
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


# ---- binding ---------------------------------------------------------------------------------------------------------
def prepare(model) -> None:
    """Capture stock's flash_mla_mega_prefill (before any candidate forward swaps it), check the static facts the
    kernel is built for and build it now (a build failure fails prepare)."""
    from sglang.kernels.ops.attention import flash_mla_interface
    from sglang.srt.runtime_context import get_exec, get_parallel

    if not _STOCK:
        fn = flash_mla_interface.flash_mla_mega_prefill
        if fn.__module__ != "sglang.kernels.ops.attention.flash_mla_interface" or fn.__qualname__ != "flash_mla_mega_prefill":
            raise RuntimeError(f"dsv41-pd055 FMLAP: flash_mla_mega_prefill is not stock's ({fn.__module__}.{fn.__qualname__})")
        _STOCK["flash_mla_mega_prefill"] = fn
    backend = get_exec().kernel.dsv4_attn_backend
    if backend != "flashmla_mega":
        raise RuntimeError(f"dsv41-pd055 FMLAP requires the flashmla_mega attention backend, got {backend}")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(f"dsv41-pd055 FMLAP is built and proven for sm_103 (B300), got {torch.cuda.get_device_capability()}")
    heads = model.config.num_attention_heads // get_parallel().attn_tp_size
    if heads != H_Q:
        raise RuntimeError(f"dsv41-pd055 FMLAP requires {H_Q} attention heads a rank, got {heads}")
    module()
    if CHECK:
        print(f"{PREFIX} FMLAP prepared: rows >= {ROWS_MIN}, topk {TOPKS}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def routed(view, forward_batch):
    """One extend forward (no speculative modes, not capturing): stock's flash_mla_mega_prefill swapped for
    mega_prefill and restored on exit; every other forward runs stock's untouched."""
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
