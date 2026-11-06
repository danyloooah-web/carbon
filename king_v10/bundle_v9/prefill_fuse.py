
from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache

import torch

CHECK = False
CHECK_CALLS = 3
PREFIX = "v4hk" if CHECK else "v4"
Q_LORA = 1280
IDX_HEADS, IDX_DIM = 32, 128
WORLD, ENGRAM_WIDTH = 4, 1536

_STOCK: dict = {}
_ON = [False]
_CHK: dict = {}
_MEMO: list = []


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
def _header() -> str:
    import hashlib

    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9 import cuda_prefill_fuse

    data = cuda_prefill_fuse.SOURCE.encode()
    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9",
                             hashlib.sha256(data).hexdigest()[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_prefill_fuse.NAME)
    if not os.path.isfile(path):
        _write(path, data)
    return path


@lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    from bundle_v9 import cuda_prefill_fuse

    return load_jit("v4_pfuse", cuda_prefill_fuse.TAG, cuda_files=[_header()], cuda_wrappers=list(cuda_prefill_fuse.WRAPPERS))


def _same(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return int((a != b).sum())


def _checked(name: str, rows: int, ours, stock) -> None:
    key = (name, max(int(rows), 1).bit_length())
    n = _CHK.get(key, 0)
    if n >= CHECK_CALLS:
        return
    _CHK[key] = n + 1
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    torch.cuda.synchronize()
    ev[0].record()
    got = ours()
    ev[1].record()
    ref = stock()
    ev[2].record()
    torch.cuda.synchronize()
    diffs = [_same(a, b) for a, b in zip(got, ref)]
    ok = len(got) == len(ref) and all(d == 0 for d in diffs)
    print(f"{PREFIX} {name} gpu{torch.cuda.current_device()} rows {rows}: {'OK' if ok else 'MISMATCH'} "
          f"bytes-differ {diffs} ours {ev[0].elapsed_time(ev[1]) * 1e3:.1f} us stock {ev[1].elapsed_time(ev[2]) * 1e3:.1f} us",
          file=sys.stderr, flush=True)


def rmsnorm_mxfp8(x: torch.Tensor, weight: torch.Tensor, eps: float):
    stock = _STOCK["rmsnorm_mxfp8"]
    if not (_ON[0] and x.dim() == 2 and x.shape[1] == Q_LORA and x.shape[0] > 0 and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16 and weight.shape == (Q_LORA,) and weight.is_contiguous()
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0
            and not torch.cuda.is_current_stream_capturing()):
        return stock(x, weight, eps)
    m = x.shape[0]

    def run():
        y = torch.empty((m, Q_LORA), dtype=torch.bfloat16, device=x.device)
        q = torch.empty((m, Q_LORA), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((Q_LORA // 32) * ((m + 127) // 128) * 128, dtype=torch.uint8, device=x.device)
        module().qnorm(x, weight, y, q, s, float(eps))
        return y, q, s

    out = run()
    if CHECK:
        _checked("QNORM", m, lambda: out, lambda: stock(x, weight, eps))
    return out


def wob_scale(input_scale: torch.Tensor) -> torch.Tensor:
    rows, words = input_scale.shape
    out = torch.empty(((rows + 127) // 128) * 128 * words * 4, dtype=torch.uint8, device=input_scale.device)
    module().wob_scale(input_scale, out, 0)
    if CHECK:
        from flashinfer.quantization.fp4_quantization import block_scale_interleave

        _checked("WOBSF", rows, lambda: (out,),
                 lambda: (block_scale_interleave(input_scale.contiguous().view(torch.uint8)).reshape(-1),))
    return out


def engram_place(received: torch.Tensor, local: torch.Tensor, mine: int) -> None:
    if mine:
        module().engram_place(received, local, WORLD, 0)
    if CHECK:
        def stock():
            ref = local.new_empty((mine, WORLD * ENGRAM_WIDTH))
            torch.add(received.view(WORLD, mine, ENGRAM_WIDTH).transpose(0, 1), 0.0,
                      out=ref.view(mine, WORLD, ENGRAM_WIDTH))
            return (ref,)

        _checked("XPLACE", mine, lambda: (local[:mine],), stock)


def engram_quant(received: torch.Tensor, mine: int, n_local: int):
    width = WORLD * ENGRAM_WIDTH
    q = torch.empty((n_local, width), dtype=torch.float8_e4m3fn, device=received.device)
    s = torch.empty(((n_local + 127) // 128) * 128 * (width // 32), dtype=torch.uint8, device=received.device)
    module().engram_quant(received, q, s, WORLD)
    if CHECK:
        def stock():
            from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

            local = received.new_empty((n_local, width))
            if mine:
                torch.add(received.view(WORLD, mine, ENGRAM_WIDTH).transpose(0, 1), 0.0,
                          out=local[:mine].view(mine, WORLD, ENGRAM_WIDTH))
            if n_local > mine:
                local[mine:].zero_()
            rq, rs = flashinfer_mxfp8_quantize(local, is_sf_swizzled_layout=True, alignment=32)
            return rq, rs.reshape(-1)

        _checked("XQUANT", n_local, lambda: (q, s), stock)
    return q, s


def dequantize_k_cache_paged_dsv4(quant_k_cache, page_table_1_flattened, page_size, out=None, layout=None):
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    stock = _STOCK["dequantize_k_cache_paged_dsv4"]
    lay = KVLayout.parse(KVLayout.V4 if layout is None else layout)
    ids = page_table_1_flattened
    n = ids.shape[0] if ids.dim() == 1 else -1
    if not (_ON[0] and lay.encoding is KVLayout.V41 and n > 0 and quant_k_cache.dim() == 2
            and quant_k_cache.is_contiguous() and quant_k_cache.element_size() * quant_k_cache.shape[1] % 16 == 0
            and ids.dtype in (torch.int32, torch.int64) and ids.is_contiguous()
            and (out is None or (out.dtype == torch.bfloat16 and tuple(out.shape) == (n, 1, 512)
                                 and out.stride(2) == 1 and out.stride(0) % 8 == 0 and out.data_ptr() % 16 == 0))
            and not torch.cuda.is_current_stream_capturing()):
        return stock(quant_k_cache, page_table_1_flattened, page_size, out=out, layout=lay)
    cache = quant_k_cache.view(torch.uint8)
    if cache.shape[1] < page_size * lay.bytes_per_token:
        raise RuntimeError(f"v9 DEQ: {cache.shape[1]} bytes a page cannot hold {page_size} tokens of {lay}")
    data_stride, scale_base, scale_stride = lay.row_addressing(page_size)
    dst = out if out is not None else torch.empty((n, 1, 512), dtype=torch.bfloat16, device=quant_k_cache.device)
    module().dequant_v41(cache, ids, dst, int(page_size), int(data_stride), int(scale_base), int(scale_stride))
    if CHECK:
        def ref():
            scratch = torch.empty_like(dst)
            return (stock(quant_k_cache, page_table_1_flattened, page_size, out=scratch, layout=lay),)

        _checked("DEQ", n, lambda: (dst,), ref)
    return dst


def build_swa_token_ids(seq_lens, extend_seq_lens, req_pool_indices, req_to_token, full_to_swa, swa_window, total_swa):
    stock = _STOCK["build_swa_token_ids"]
    if not (_ON[0] and seq_lens.dtype == extend_seq_lens.dtype == req_pool_indices.dtype == req_to_token.dtype
            == torch.int32 and full_to_swa.dtype == torch.int64 and req_to_token.dim() == 2 and req_to_token.stride(1) == 1
            and seq_lens.is_contiguous() and req_pool_indices.is_contiguous() and full_to_swa.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()):
        return stock(seq_lens=seq_lens, extend_seq_lens=extend_seq_lens, req_pool_indices=req_pool_indices,
                     req_to_token=req_to_token, full_to_swa=full_to_swa, swa_window=swa_window, total_swa=total_swa)
    num_reqs = seq_lens.shape[0]
    device = seq_lens.device
    swa_gather_lens = torch.minimum(seq_lens, extend_seq_lens + (swa_window - 1)).to(torch.int32)
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if total_swa == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets
    module().swa_ids(swa_token_ids, swa_first_pos, swa_offsets, req_pool_indices, req_to_token, full_to_swa)
    if CHECK:
        def ref():
            r = stock(seq_lens=seq_lens, extend_seq_lens=extend_seq_lens, req_pool_indices=req_pool_indices,
                      req_to_token=req_to_token, full_to_swa=full_to_swa, swa_window=swa_window, total_swa=total_swa)
            end = int(r[3][-1])
            return r[0][:end], r[1], r[2], r[3]

        end = int(swa_offsets[-1])
        _checked("SWAIDS", total_swa, lambda: (swa_token_ids[:end], swa_first_pos, swa_gather_lens, swa_offsets), ref)
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


def index_q_rope_pack_weights(input, freqs_cis, positions, head_weights, weight_scale):
    stock = _STOCK["index_q_rope_pack_weights"]
    if not (_ON[0] and input.dim() == 3 and input.shape[1:] == (IDX_HEADS, IDX_DIM) and input.shape[0] > 0
            and input.dtype == torch.bfloat16 and input.is_contiguous() and input.data_ptr() % 16 == 0
            and freqs_cis.dtype == torch.float32 and freqs_cis.dim() == 2 and freqs_cis.shape[1] == 64
            and freqs_cis.is_contiguous() and freqs_cis.data_ptr() % 16 == 0
            and positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
            and head_weights.dtype == torch.bfloat16 and head_weights.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()):
        return stock(input, freqs_cis, positions, head_weights, weight_scale)
    num_tokens, heads, head_dim = input.shape
    rows = num_tokens * heads
    payload = input.new_empty((rows, head_dim // 2), dtype=torch.int8)
    scale = input.new_empty((rows,), dtype=torch.int32)
    weights = input.new_empty((num_tokens, heads), dtype=torch.float32)
    module().index_q(input, freqs_cis, positions, payload, scale, head_weights, weights, float(weight_scale))
    if CHECK:
        _checked("IDXQ", num_tokens, lambda: (payload, scale, weights),
                 lambda: stock(input, freqs_cis, positions, head_weights, weight_scale))
    return payload, scale, weights


def _graph_scope() -> bool:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.models.deepseek_v4_replay_tail import in_replay_tail_graph

    return torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph() or in_replay_tail_graph()


def combine_compressed(self, compress_ratio, sparse_raw_indices):
    from bundle_v9 import rehome_block

    stock = _STOCK["combine_compressed"]
    if not _ON[0] or _graph_scope() or rehome_block.in_flight():
        return stock(self, compress_ratio, sparse_raw_indices)
    raw = sparse_raw_indices
    key = (int(compress_ratio), raw.data_ptr(), tuple(raw.shape), tuple(raw.stride()), raw.dtype, raw.device)
    if _MEMO and _MEMO[0] is self and _MEMO[1] == key:
        out = _MEMO[2]
        if CHECK:
            _checked(f"CMEMO-s{_MEMO[3]}", raw.shape[0], lambda: tuple(t.clone() for t in out),
                     lambda: stock(self, compress_ratio, raw))
        return out
    computed = _MEMO[3] + 1 if _MEMO and _MEMO[0] is self else 1
    out = stock(self, compress_ratio, raw)
    _MEMO[:] = [self, key, out, computed]
    return out


def forward_low_ratio_sources(self, *args, **kwargs):
    if _MEMO:
        _MEMO[1] = None
    return _STOCK["forward_low_ratio_sources"](self, *args, **kwargs)


def _owners():
    from sglang.kernels.ops.layernorm import rmsnorm_mxfp8 as norm_mod
    from sglang.srt.layers.attention import deepseek_v4_backend as backend_mod
    from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils as prefill_mod
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring as scoring_mod

    return norm_mod, backend_mod, prefill_mod, scoring_mod


def _classes():
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.deepseek_v4_mega_backend import DeepseekV4MegaAttnBackend
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import SparsePrefillChunkCache

    return SparsePrefillChunkCache, DeepseekV4AttnBackend, DeepseekV4MegaAttnBackend


def _stock_fn(fn, origin: str, qualname: str):
    if fn.__module__ != origin or fn.__qualname__ != qualname:
        raise RuntimeError(f"v9 PFUSE: {qualname} is not stock's ({fn.__module__}.{fn.__qualname__})")
    return fn


def prepare(model) -> None:
    norm_mod, backend_mod, prefill_mod, scoring_mod = _owners()
    cache_cls, backend_cls, mega_cls = _classes()
    if not _STOCK:
        _STOCK["combine_compressed"] = _stock_fn(cache_cls.combine_compressed,
                                                 "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
                                                 "SparsePrefillChunkCache.combine_compressed")
        _STOCK["forward_low_ratio_sources"] = _stock_fn(backend_cls.forward_low_ratio_sources,
                                                        "sglang.srt.layers.attention.deepseek_v4_backend",
                                                        "DeepseekV4AttnBackend.forward_low_ratio_sources")
        if mega_cls.forward_low_ratio_sources is not backend_cls.forward_low_ratio_sources:
            raise RuntimeError("v9 CMEMO: the flashmla_mega backend overrides forward_low_ratio_sources")
        _STOCK["rmsnorm_mxfp8"] = _stock_fn(norm_mod.rmsnorm_mxfp8, "sglang.kernels.ops.layernorm.rmsnorm_mxfp8",
                                            "rmsnorm_mxfp8")
        _STOCK["dequantize_k_cache_paged_dsv4"] = _stock_fn(backend_mod.dequantize_k_cache_paged_dsv4,
                                                            "sglang.kernels.ops.attention.dsv4.dequant_k_cache",
                                                            "dequantize_k_cache_paged_dsv4")
        _STOCK["build_swa_token_ids"] = _stock_fn(prefill_mod.build_swa_token_ids,
                                                  "sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels",
                                                  "build_swa_token_ids")
        _STOCK["index_q_rope_pack_weights"] = _stock_fn(scoring_mod.index_q_rope_pack_weights,
                                                        "sglang.kernels.ops.attention.dsv4.fp4_indexer_rope",
                                                        "index_q_rope_pack_weights")
    config = model.config
    facts = (config.q_lora_rank, config.index_n_heads, config.index_head_dim,
             (config.engram_max_ngram_size - 1) * config.engram_n_heads // WORLD * config.engram_head_dim)
    if facts != (Q_LORA, IDX_HEADS, IDX_DIM, ENGRAM_WIDTH):
        raise RuntimeError(f"v9 PFUSE requires (q_lora_rank, index heads, index head dim, Engram exchange "
                           f"width) = {(Q_LORA, IDX_HEADS, IDX_DIM, ENGRAM_WIDTH)}, got {facts}")
    module()
    if CHECK:
        print(f"{PREFIX} prepared: {sorted(_STOCK)}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def routed(view, forward_batch):
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend_without_speculative() or torch.cuda.is_current_stream_capturing() or not _STOCK:
        yield
        return
    norm_mod, backend_mod, prefill_mod, scoring_mod = _owners()
    cache_cls, backend_cls, _ = _classes()
    del _MEMO[:]
    _ON[0] = True
    try:
        norm_mod.rmsnorm_mxfp8 = rmsnorm_mxfp8
        backend_mod.dequantize_k_cache_paged_dsv4 = dequantize_k_cache_paged_dsv4
        prefill_mod.build_swa_token_ids = build_swa_token_ids
        scoring_mod.index_q_rope_pack_weights = index_q_rope_pack_weights
        cache_cls.combine_compressed = combine_compressed
        backend_cls.forward_low_ratio_sources = forward_low_ratio_sources
        yield
    finally:
        _ON[0] = False
        del _MEMO[:]
        norm_mod.rmsnorm_mxfp8 = _STOCK["rmsnorm_mxfp8"]
        backend_mod.dequantize_k_cache_paged_dsv4 = _STOCK["dequantize_k_cache_paged_dsv4"]
        prefill_mod.build_swa_token_ids = _STOCK["build_swa_token_ids"]
        scoring_mod.index_q_rope_pack_weights = _STOCK["index_q_rope_pack_weights"]
        cache_cls.combine_compressed = _STOCK["combine_compressed"]
        backend_cls.forward_low_ratio_sources = _STOCK["forward_low_ratio_sources"]
