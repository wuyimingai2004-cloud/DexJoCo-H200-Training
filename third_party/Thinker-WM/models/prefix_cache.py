"""Inference-only exact reuse of fixed conditions within ONE denoising block.

The sampler owns separate instances for CFG branches and creates new instances
at every AR block/call. No global or module-level state, no checkpoint keys.
A byte budget limits stored tensors; exceeding it recomputes that site without
dropping any visible history. This is not approximate attention/KV eviction.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from thinker_wm.models.temporal_layout import ROLE_TARGET


def fixed_rows(layout, name, length=None, device=None):
    if name == "und":
        return torch.ones(length, dtype=torch.bool, device=device)
    return layout.segment(name).role != ROLE_TARGET


@dataclass
class Partition:
    fixed: torch.Tensor
    moving: torch.Tensor
    length: int

    @classmethod
    def from_mask(cls, mask):
        # One nonzero/synchronisation per stream and block, not per operator/step.
        return cls(mask.nonzero().flatten(), (~mask).nonzero().flatten(), mask.numel())


def join_rows(partition, fixed_value, moving_value):
    """Restore token order using precomputed indices (no CUDA boolean indexing)."""
    shape = list(fixed_value.shape)
    shape[1] = partition.length
    out = fixed_value.new_empty(shape)
    out.index_copy_(1, partition.fixed, fixed_value)
    out.index_copy_(1, partition.moving, moving_value)
    return out


def _bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_bytes(v) for v in value)
    if isinstance(value, dict):
        return sum(_bytes(v) for v in value.values())
    return 0


@dataclass
class PrefixKVCache:
    max_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self):
        if self.max_bytes < 0:
            raise ValueError("prefix cache budget must be non-negative")
        self.entries = {}
        self.partitions = {}
        self.bytes = self.hits = self.misses = self.budget_skips = self.reused_rows = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key, value):
        size = _bytes(value)
        if self.bytes + size > self.max_bytes:
            self.budget_skips += 1
            return
        self.entries[key] = value
        self.bytes += size

    def partition(self, key, fixed):
        if key not in self.partitions:
            self.partitions[key] = Partition.from_mask(fixed)
        p = self.partitions[key]
        if p.length != fixed.numel():
            raise ValueError("prefix cache cannot be reused for a different layout")
        return p

    def rows(self, key, x, fixed, fn, partition=None):
        """Cache fixed rows of a token-local operation; recompute moving rows."""
        p = partition if partition is not None else self.partition(("op", key), fixed)
        if not p.fixed.numel():
            return fn(x)
        cached = self.get(key)
        if cached is None:
            y = fn(x)
            self.put(key, y.index_select(1, p.fixed).detach())
            return y
        self.reused_rows += p.fixed.numel()
        moving = fn(x.index_select(1, p.moving)) if p.moving.numel() else cached[:, :0]
        return join_rows(p, cached, moving)

    def stats(self):
        return dict(bytes=self.bytes, entries=len(self.entries), hits=self.hits,
                    misses=self.misses, budget_skips=self.budget_skips,
                    reused_rows=self.reused_rows,
                    index_bytes=sum(_bytes(p.fixed) + _bytes(p.moving) for p in self.partitions.values()))


def cached_rows(layout, key, name, x, fn):
    cache = getattr(layout, "prefix_cache", None)
    if cache is None:
        return fn(x)
    p = cache.partitions.get(("stream", name))
    if p is None:
        p = cache.partition(("stream", name), fixed_rows(layout, name, x.size(1), x.device))
    # Small action GEMMs can choose different BF16 kernels after row slicing.
    # Preserve their original shape until a separately verified kernel exists.
    if name == "action" and p.moving.numel():
        return fn(x)
    return cache.rows(key, x, None, fn, partition=p)
