"""Per-forward packed index-K reuse with single-segment ownership.

The cache key includes the mapped compressed source and the slot tensor's
identity. A different source or slot tensor replaces the entry, and every model
forward clears it on entry and exit. No packed values survive a replay.
"""
from __future__ import annotations

import copy
from types import MethodType


class IndexReuse:
    """Hold at most one packed source segment for one model forward."""

    def __init__(self):
        self.entry = None

    def get(self, pool, layer_id: int, slots):
        """Return a packed gather, or the identical tuple already read this segment."""
        ratio, source_id, _ = pool.layer_mapping[layer_id]
        key = (ratio, source_id, slots.data_ptr(), slots.shape, slots.stride(), slots._version)
        if self.entry is not None and self.entry[0] == key:
            return self.entry[1]
        value = pool.get_low_ratio_index_k_fp4(layer_id, slots)
        self.entry = (key, value)
        return value

    def clear(self) -> None:
        """Release the preceding segment at a forward boundary."""
        self.entry = None


def _packed_gather(view, layer_id: int, slots):
    return view._index_reuse.get(view._stock_pool, layer_id, slots)


def pool_view(pool, reuse: IndexReuse):
    """Copy the pool and override only the scoped packed gather on that copy."""
    view = copy.copy(pool)
    view._stock_pool = pool
    view._index_reuse = reuse
    view.get_low_ratio_index_k_fp4 = MethodType(_packed_gather, view)
    return view


def owner_view(owner, reuse: IndexReuse):
    """Copy an owner so stock methods bind to it and read its scoped pool view."""
    view = copy.copy(owner)
    view.token_to_kv_pool = pool_view(owner.token_to_kv_pool, reuse)
    return view


def backend_view(backend, reuse: IndexReuse):
    """Copy a backend so its stock prepare methods use the scoped pool views."""
    view = copy.copy(backend)
    view.token_to_kv_pool = pool_view(backend.token_to_kv_pool, reuse)
    view.full_topk_indexer = owner_view(backend.full_topk_indexer, reuse)
    view.prefill_candidates = (
        None if backend.prefill_candidates is None else owner_view(backend.prefill_candidates, reuse)
    )
    return view
