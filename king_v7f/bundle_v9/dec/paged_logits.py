from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import torch

from bundle_v9.dec import cuda_paged_logits

NEXT_N = 6
HEADS = 32
_STOCK: list = []
_ON = [False]


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


def _digest() -> str:
    return hashlib.sha256(cuda_paged_logits.SOURCE.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def _header() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    directory = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-paged-logits", _digest())
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, cuda_paged_logits.NAME)
    if not os.path.isfile(path):
        _write(path, cuda_paged_logits.SOURCE.encode())
    return path


@lru_cache(maxsize=None)
def _module():
    import deep_gemm
    from sglang.kernels.jit.utils import load_jit, override_jit_cuda_arch

    include = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    with override_jit_cuda_arch(10, 0, "f"):
        return load_jit("dec_paged_logits", _digest(), cuda_files=[_header()], cuda_wrappers=[("run", "paged_logits::PagedLogits::run")],
                        extra_cuda_cflags=["--expt-extended-lambda", "--ptxas-options=--register-usage-level=10",
                                           "--compiler-options=-fconcepts"],
                        extra_include_paths=[include])


def _stock():
    if not _STOCK:
        from sglang.kernels.ops.attention.dsv4 import index_logits

        fn = index_logits.deep_gemm_fp4_paged_mqa_logits
        if fn.__module__ != index_logits.__name__ or fn.__qualname__ != "deep_gemm_fp4_paged_mqa_logits":
            raise RuntimeError(f"bundle-v9 PAGED_LOGITS: index_logits.deep_gemm_fp4_paged_mqa_logits is not stock's "
                               f"({fn.__module__}.{fn.__qualname__})")
        _STOCK.append(fn)
    return _STOCK[0]


def stock_deep_gemm_fp4_paged_mqa_logits():
    return _stock()


def _stride(max_seq_len: int) -> int:
    return ((max_seq_len + 255) // 256 * 256 + 511) // 512 * 512


def _serves(q, k_cache, weights, seq_lens, page_table, meta) -> bool:
    import deep_gemm

    if not (isinstance(q, tuple) and len(q) == 2 and isinstance(meta, torch.Tensor)):
        return False
    q_fp4, q_sf = q
    requests = q_fp4.shape[0] if q_fp4.dim() == 4 else 0
    return (requests > 0 and tuple(q_fp4.shape[1:]) == (NEXT_N, HEADS, 64) and q_fp4.element_size() == 1
            and q_fp4.is_contiguous() and q_sf.dtype == torch.int32 and tuple(q_sf.shape) == (requests, NEXT_N, HEADS)
            and q_sf.is_contiguous() and k_cache.dim() == 4 and tuple(k_cache.shape[1:]) == (128, 1, 68)
            and k_cache.dtype == torch.uint8 and k_cache.stride(3) == 1 and k_cache.stride(1) == 68
            and k_cache.stride(0) % 16 == 0 and k_cache.shape[0] > 0
            and tuple(weights.shape) == (requests * NEXT_N, HEADS) and weights.stride(1) == 1
            and tuple(seq_lens.shape) == (requests, NEXT_N) and page_table.dim() == 2 and page_table.shape[0] == requests
            and page_table.dtype == torch.int32 and page_table.stride(1) == 1 and meta.dim() == 2 and meta.shape[1] == 2
            and meta.dtype == torch.int32 and meta.is_contiguous() and meta.shape[0] == deep_gemm.get_num_sms() + 1)


def _run(q, k_cache, weights, seq_lens, page_table, meta, max_seq_len: int) -> torch.Tensor:
    import deep_gemm

    q_fp4, q_sf = q
    out = torch.empty((q_fp4.shape[0] * NEXT_N, _stride(max_seq_len)), dtype=torch.bfloat16, device=q_fp4.device)
    _module().run(q_fp4, q_sf, k_cache, weights.to(torch.bfloat16), seq_lens.to(torch.int32).contiguous(), page_table,
                  meta, out, int(bool(deep_gemm.get_pdl())))
    return out[:, :max_seq_len]


def deep_gemm_fp4_paged_mqa_logits(q_fp4, k_cache, weights, seq_lens, page_table, deep_gemm_metadata, max_seq_len):
    if _ON[0] and 0 < max_seq_len < 1 << 30 and _serves(q_fp4, k_cache, weights, seq_lens, page_table,
                                                        deep_gemm_metadata):
        return _run(q_fp4, k_cache, weights, seq_lens, page_table, deep_gemm_metadata, max_seq_len)
    return _stock()(q_fp4, k_cache, weights, seq_lens, page_table, deep_gemm_metadata, max_seq_len)


def _lens(requests: int, width: int, mode: str, gen) -> torch.Tensor:
    step = torch.arange(NEXT_N, dtype=torch.int32)
    if mode == "random":
        last = torch.randint(NEXT_N, width + 1, (requests, 1), generator=gen, dtype=torch.int32)
    elif mode == "short":
        last = torch.randint(NEXT_N, 3 * NEXT_N, (requests, 1), generator=gen, dtype=torch.int32)
    elif mode == "full":
        last = torch.full((requests, 1), width, dtype=torch.int32)
    elif mode == "padded":
        last = torch.randint(NEXT_N, width + 1, (requests, 1), generator=gen, dtype=torch.int32)
        last[requests // 2:] = 0
    else:
        edges = [NEXT_N, 127, 128, 129, 255, 256, 257, 383, 384, 512, 513, 4095, 4096, 4097, width - 257, width - 256,
                 width - 255, width - 129, width - 1, width]
        last = torch.tensor([[edges[i % len(edges)]] for i in range(requests)], dtype=torch.int32)
    return torch.where(last > 0, (last - NEXT_N + 1 + step).clamp(min=1), torch.zeros_like(last + step))


def _pool(pages: int, page_bytes: int, family: str, gen, device) -> torch.Tensor:
    if family == "moderate":
        raw = torch.randint(112, 144, (pages, page_bytes), generator=gen, dtype=torch.uint8)
    elif family == "zero":
        raw = torch.zeros((pages, page_bytes), dtype=torch.uint8)
    else:
        raw = torch.randint(0, 256, (pages, page_bytes), generator=gen, dtype=torch.uint8)
        sf = raw[:, 128 * 64:128 * 68]
        if family == "wide":
            sf.copy_(torch.randint(96, 160, sf.shape, generator=gen, dtype=torch.uint8))
        else:
            sf.copy_(torch.randint(0, 255, sf.shape, generator=gen, dtype=torch.uint8))
    return raw.to(device).as_strided((pages, 128, 1, 68), (page_bytes, 68, 68, 1))


def _prove(device) -> int:
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMDecodeData

    stock = _stock()
    gen = torch.Generator().manual_seed(20261105)
    pages = 1536
    cases = 0
    for requests, width in ((48, 131328), (48, 262656), (40, 131328), (8, 262656), (1, 131328), (48, 72064)):
        cols = width // 128
        for family in ("moderate", "wide", "extreme", "zero"):
            page_bytes = 8704 if (requests + cases) % 2 == 0 else 8960
            k = _pool(pages, page_bytes, family, gen, device)
            rows = requests * NEXT_N
            q_fp4 = torch.randint(-128, 128, (rows, 1, HEADS, 64), generator=gen, dtype=torch.int8).to(device)
            low, high = (122, 133) if family in ("moderate", "zero") else ((96, 160) if family == "wide" else (0, 255))
            q_sf = torch.randint(low, high, (rows, 1, HEADS, 4), generator=gen, dtype=torch.uint8).view(torch.int32)
            q_sf = q_sf.view(rows, 1, HEADS).to(device)
            w = (torch.randn(rows, HEADS, generator=gen) * 0.1).to(torch.bfloat16).to(device)
            for mode in ("random", "short", "edges", "full", "padded"):
                for pad in ("zero", "noise"):
                    lens = _lens(requests, width, mode, gen)
                    need = (lens[:, -1].long() + 127) // 128
                    table = torch.randint(0, pages, (requests, cols), generator=gen, dtype=torch.int32)
                    if pad == "zero":
                        table[torch.arange(cols)[None, :] >= need[:, None]] = 0
                    table = table.to(device)
                    meta = PagedIndexerMetadata(page_size=256, compressed_page_size=128,
                                                page_table=table.repeat_interleave(NEXT_N, 0),
                                                compressed_seq_lens=lens.reshape(-1).to(device), use_topk_v2=False,
                                                compress_ratio=1 if width > 131328 else 2, rows_per_request=NEXT_N,
                                                use_prefill_cuda_graph=True)
                    q, lv, tbl = DeepGEMMDecodeData(q_fp4=q_fp4, q_sf=q_sf, weights=w, k_cache=k).paged_logits_args(meta)
                    args = (q, k, w, lv, tbl, meta.deep_gemm_metadata, meta.max_compressed_seq_len)
                    if not _serves(*args[:6]):
                        raise RuntimeError(f"bundle-v9 PAGED_LOGITS: SGLang's decode call at {rows} rows x {width} is outside the "
                                           "domain")
                    want = stock(*args)
                    got = _run(*args)
                    if want.shape != got.shape or want.stride() != got.stride() or want.dtype != got.dtype:
                        raise RuntimeError(f"bundle-v9 PAGED_LOGITS: logits {tuple(got.shape)} / {got.stride()} vs stock's "
                                           f"{tuple(want.shape)} / {want.stride()}")
                    own = lens.reshape(-1).to(device).long()
                    reach = (own.view(requests, NEXT_N)[:, -1:] + 255) // 256 * 256
                    limit = torch.minimum(reach.expand(requests, NEXT_N).reshape(-1), own.new_tensor(width)) \
                        if pad == "zero" else own
                    column = torch.arange(want.shape[1], device=device)
                    differ = (want.view(torch.int16) != got.view(torch.int16)) & (column[None, :] < limit[:, None])
                    bad = int(differ.sum())
                    if bad:
                        raise RuntimeError(f"bundle-v9 PAGED_LOGITS: {bad} logits differ from stock's ({family}, {rows} rows x "
                                           f"{width}, lengths {mode}, table past the pages {pad})")
                    cases += 1
    torch.cuda.synchronize(device)
    return cases


def prepare(model) -> None:
    from sglang.kernels.ops.attention.dsv4 import index_logits
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, sparse_table

    stock = _stock()
    if full_topk.deep_gemm_fp4_paged_mqa_logits is not stock or sparse_table.deep_gemm_fp4_paged_mqa_logits is not stock:
        raise RuntimeError("bundle-v9 PAGED_LOGITS: a decode paged logits entry was swapped before prepare")
    if index_logits.index_logits_dtype() != torch.bfloat16 or index_logits.index_weights_dtype() != torch.bfloat16:
        raise TypeError("bundle-v9 PAGED_LOGITS serves bf16 index logits and head weights; this stack scores in "
                        f"{index_logits.index_logits_dtype()} / {index_logits.index_weights_dtype()}")
    device = next(model.parameters()).device
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise TypeError("bundle-v9 PAGED_LOGITS is built for SM100-family devices")
    cases = _prove(device)
    _ON[0] = True
    print(f"bundle-v9 PAGED_LOGITS: decode paged index logits, bytes proven on {cases} cases", flush=True)


def enabled() -> bool:
    return _ON[0]
