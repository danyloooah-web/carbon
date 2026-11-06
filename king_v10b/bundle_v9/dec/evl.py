"""king-v2e EVL: an evict-first L2 cache hint on the decode MegaMoE's routed-expert weight and weight-scale TMA loads.

MOE-DECODE (moe_decode.py) runs every 1..288-row decode / verify MegaMoE call on the MOE-PREFILL host library
(moe_prefill.ops()), whose DeepJIT runtime compiles `sm100_fp8_fp4_mega_moe_impl` from the kernel headers under
`<jit root>/include`. In the king the routed branch of the TMA-B warp issues its weight and scale copies with the
default EVICT_NORMAL policy, so each layer's ~1.8 GB per-rank weight stream competes in L2 with the dispatch ring, the
combine buffers, the evict-last activations and the residual stream. Here the routed (not the shared-expert) weight
and scale copies pass CacheHintSm100::EVICT_FIRST through tma::copy's cache-hint operand, the same operand the king
already passes for the activations (kActsCacheHint). The shared expert keeps EVICT_NORMAL (five m-blocks re-read it).

Only the policy operand of two TMA instructions changes: no barrier, transaction count, coordinate or data path does,
so the kernel's output bytes are unchanged by construction.

The edited kernel headers live in their own DeepJIT root (`<cache_root>/bundle-v9-dec-moe-evl/<sha256[:16]>/include`,
the three `include/` texts of cuda_moe_prefill.SOURCES with the edits below applied). The host library is the king's,
built from the king's source root under the king's TAG, unchanged. moe_decode.prepare re-points the library's JIT at
the EVL root before it compiles its bands; any failure (an anchor that does not match exactly once, a write error, a
compile error) re-points it at the king's root and compiles the king's kernel text, so the king's path is the fallback.
"""
from __future__ import annotations

import hashlib
import os

# The lead's switch. True: the decode MegaMoE compiles with the EVL edits. False: the king's kernel text, unchanged.
EVL_ON = False  # king-v6t: off (v2e profile: EVL broke the draft MoE JIT; R-L2 neutral)

HEADER = "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"

_ACTS = """    constexpr uint64_t kActsCacheHint = static_cast<uint64_t>(BLOCK_M <= 128 ?
        cute::TMA::CacheHintSm100::EVICT_LAST : cute::TMA::CacheHintSm100::EVICT_NORMAL);
"""
_ROUTED = """                        tma::copy<BLOCK_K_BYTES, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * kNumRoutedBTxTiles"""

_ACTS_EVL = """    constexpr uint64_t kActsCacheHint = static_cast<uint64_t>(BLOCK_M <= 128 ?
        cute::TMA::CacheHintSm100::EVICT_LAST : cute::TMA::CacheHintSm100::EVICT_NORMAL);
    // king-v2e EVL: the routed experts' weights and scales are read once a call; evict them first from L2
    constexpr uint64_t kRoutedWeightsCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST);
"""
_ROUTED_EVL = """                        tma::copy<BLOCK_K_BYTES, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2, 0, kRoutedWeightsCacheHint);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2, 0, kRoutedWeightsCacheHint);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * kNumRoutedBTxTiles"""

EDITS = ((_ACTS, _ACTS_EVL), (_ROUTED, _ROUTED_EVL))

_STATE = {"on": False, "why": "not prepared"}


def include_texts() -> dict:
    """The three kernel-header texts of the EVL root. Raises unless every edit anchor matches exactly once."""
    from bundle_v9.dec import cuda_moe_prefill

    texts = {rel: text for rel, text in cuda_moe_prefill.SOURCES.items() if rel.startswith("include/")}
    if HEADER not in texts:
        raise RuntimeError(f"{HEADER} is not in cuda_moe_prefill.SOURCES")
    text = texts[HEADER]
    for old, new in EDITS:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"an EVL anchor matches {count} times in {HEADER}, not once")
        text = text.replace(old, new)
    if text.count("kRoutedWeightsCacheHint") != 3:
        raise RuntimeError("the EVL edits did not land as one declaration and two uses")
    texts[HEADER] = text
    return texts


def jit_root() -> str:
    """Writes the EVL include tree (once per content digest) and returns its root."""
    from sglang.kernels.jit.utils.compile.cache import cache_root

    from bundle_v9.dec import moe_prefill

    texts = include_texts()
    digest = hashlib.sha256()
    for rel in sorted(texts):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(texts[rel].encode())
        digest.update(b"\0")
    root = os.path.join(os.path.abspath(os.fspath(cache_root())), "bundle-v9-dec-moe-evl", digest.hexdigest()[:16])
    for rel, text in texts.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            moe_prefill._write(path, text.encode())
    return root


def mark(on: bool, why: str) -> None:
    _STATE["on"] = bool(on)
    _STATE["why"] = why


def line() -> str:
    return f"bundle-v9 king-v2e EVL: {'ON' if _STATE['on'] else 'off'} ({_STATE['why']})"
