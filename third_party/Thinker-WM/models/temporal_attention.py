"""Block-causal visibility over physical time, for every attention path.

``flash_attention`` only takes lengths and windows; ``causal=True`` would be a
token-order triangle, not physical-time block causality. This module builds
explicit boolean masks from ``temporal_layout.Segment`` roles/blocks and runs
SDPA with them. Masks are computed once per forward and cached on the layout.

First version favours exactness over speed: one masked SDPA call per reader,
keys concatenated over the sources that reader may see.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from thinker_wm.models.temporal_layout import (
    ROLE_CLEAN, ROLE_HISTORY, ROLE_OBS, ROLE_TARGET, ROLE_GOAL, Segment, TemporalLayout,
)


def visibility(q: Segment, k: Segment, *, target_reads_earlier_targets: bool = True) -> torch.Tensor:
    """Bool [Lq, Lk]: may query token i read key token j."""
    qb, qr = q.block[:, None], q.role[:, None]
    kb, kr = k.block[None, :], k.role[None, :]
    hist_k, obs_k = kr == ROLE_HISTORY, kr == ROLE_OBS
    clean_k, tgt_k = kr == ROLE_CLEAN, kr == ROLE_TARGET
    goal_k = kr == ROLE_GOAL
    q_goal = qr == ROLE_GOAL
    le, lt, eq = kb <= qb, kb < qb, kb == qb

    q_hist = qr == ROLE_HISTORY
    q_obs = qr == ROLE_OBS
    q_clean = qr == ROLE_CLEAN
    q_tgt = qr == ROLE_TARGET

    from_hist = hist_k & le
    allowed_hist_q = from_hist
    # Same-block targets read OBS, so OBS must not carry that block's clean copy.
    allowed_obs_q = hist_k | (obs_k & le) | (clean_k & lt)
    allowed_clean_q = hist_k | ((obs_k | clean_k | goal_k) & le)
    # A goal is a fixed visual condition, never a summary of its own clean
    # GT action copy or any noisy target. This also closes indirect D0 leaks.
    allowed_goal_q = hist_k | ((obs_k | goal_k) & le) | (clean_k & lt)
    # D0 is noisy at time 0, never a clean/GT condition. Only the first future
    # block may read it; D0 itself sees observed time <=0 and itself only.
    tgt_from_tgt = tgt_k & (((kb > 0) & (le if target_reads_earlier_targets else eq))
                            | ((kb == 0) & (qb <= 1)))
    allowed_tgt_q = hist_k | ((obs_k | goal_k) & le) | (clean_k & lt) | tgt_from_tgt
    return ((q_hist & allowed_hist_q) | (q_obs & allowed_obs_q)
            | (q_clean & allowed_clean_q) | (q_tgt & allowed_tgt_q)
            | (q_goal & allowed_goal_q))


def und_visibility(k: Segment) -> torch.Tensor:
    """Bool [Lk]: tokens a global und condition may summarize (observed, block<=0)."""
    return ((k.role == ROLE_HISTORY) | (k.role == ROLE_OBS)) & (k.block <= 0)


def _with_validity(mask: torch.Tensor, q: Optional[Segment], k: Segment, batch: int) -> torch.Tensor:
    """[Lq,Lk] -> [B or 1, Lq, Lk]; invalid keys hidden, invalid queries keep self."""
    if k.valid is None:
        return mask.unsqueeze(0)
    m = mask.unsqueeze(0) & k.valid[:, None, :].to(mask.device)
    if q is not None and q is k:
        eye = torch.eye(len(q), dtype=torch.bool, device=mask.device).unsqueeze(0)
        m = m | (eye & ~k.valid[:, :, None].to(mask.device))
    return m.expand(batch, -1, -1) if m.size(0) == 1 else m


def _bias_columns(hist_cols: Optional[torch.Tensor], bias: Optional[torch.Tensor]):
    # Rebuilt on every call (never cached): it depends on a learnable parameter,
    # and activation checkpointing must record the same ops on recomputation.
    if hist_cols is None or bias is None:
        return None
    return hist_cols.float() * bias.float().reshape(())


def self_mask(layout: TemporalLayout, name: str, batch: int):
    key = ("self", name)
    if key not in layout.cache:
        seg = layout.segment(name)
        m = visibility(seg, seg, target_reads_earlier_targets=layout.request.target_reads_earlier_targets)
        hist = seg.role == ROLE_HISTORY
        layout.cache[key] = (_with_validity(m, seg, seg, batch), hist if bool(hist.any()) else None)
    mask, hist = layout.cache[key]
    return mask, _bias_columns(hist, layout.history_logit_bias)


def reader_mask(layout: TemporalLayout, reader: str, sources: Sequence[str], batch: int):
    key = ("uni", reader, tuple(sources))
    if key not in layout.cache:
        q = layout.segment(reader)
        masks, hists = [], []
        for s in sources:
            k = layout.segment(s)
            m = visibility(q, k, target_reads_earlier_targets=layout.request.target_reads_earlier_targets)
            masks.append(_with_validity(m, q if s == reader else None, k, batch))
            hists.append(k.role == ROLE_HISTORY)
        hist = torch.cat(hists)
        width = max(x.size(0) for x in masks)
        masks = [x.expand(width, -1, -1) if x.size(0) == 1 else x for x in masks]
        layout.cache[key] = (torch.cat(masks, dim=-1), hist if bool(hist.any()) else None)
    mask, hist = layout.cache[key]
    return mask, _bias_columns(hist, layout.history_logit_bias)


def und_reader_mask(layout: TemporalLayout, sources: Sequence[str], und_len: int, batch: int):
    key = ("und", tuple(sources), und_len)
    if key not in layout.cache:
        cols = []
        for s in sources:
            k = layout.segment(s)
            allowed = und_visibility(k)
            if layout.request.und_reads == "none":
                allowed = torch.zeros_like(allowed)
            c = allowed.unsqueeze(0)
            if k.valid is not None:
                c = c & k.valid.to(c.device)
            cols.append(c.expand(batch, -1) if c.size(0) == 1 else c)
        m = torch.cat(cols, dim=-1)[:, None, :].expand(batch, und_len, -1)
        layout.cache[key] = m
    return layout.cache[key]


# "flex": block-sparse flex_attention built from the same bool mask (CUDA,
# skips fully hidden 128x128 tiles); "dense": one bool-masked SDPA.
ATTN_IMPL = os.environ.get("MOTUS_TEMPORAL_ATTN", "flex")
if ATTN_IMPL not in ("flex", "dense"):
    raise ValueError(f"MOTUS_TEMPORAL_ATTN must be flex or dense, got {ATTN_IMPL!r}")
FLEX_MIN_QUERIES = 256  # short readers (action) are faster on dense SDPA
_flex = None


def _flex_attention():
    global _flex
    if _flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        import torch._dynamo
        # One compile per (reader, mode, shape); the default limit of 8 would
        # silently fall back to the unfused (slow) path.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _flex = torch.compile(flex_attention, dynamic=False)
    return _flex


def _block_mask(mask: torch.Tensor):
    """Cached on the mask tensor: layout masks are built once per forward and
    reused by every layer and by checkpoint recomputation."""
    bm = getattr(mask, "_flex_block_mask", None)
    if bm is None:
        from torch.nn.attention.flex_attention import create_block_mask
        shared = mask.size(0) == 1
        padded = F.pad(mask, (0, _key_pad(mask.size(2)), 0, _key_pad(mask.size(1))))
        one = padded[0]

        def mask_mod(b, h, qi, ki):
            return one[qi, ki] if shared else padded[b, qi, ki]

        bm = create_block_mask(mask_mod, None if shared else mask.size(0), None,
                               padded.size(1), padded.size(2), device=mask.device)
        mask._flex_block_mask = bm
    return bm


def _key_pad(lk):
    # An unaligned key length makes the score_mod bias backward ~3x slower.
    return -lk % 128


def _flex_masked_attention(q, k, v, mask, bias_cols, work):
    # Pad BOTH axes to the sparse tile size. In torch 2.7.1 the backward
    # kernel for a partial query tile (Q=2600, KV=2688) triggered an illegal
    # access in production. Extra queries have no visible keys and are discarded;
    # extra keys are masked, preserving the original attention and gradients.
    query_len = q.size(1)
    q = F.pad(q, (0, 0, 0, 0, 0, _key_pad(query_len)))
    pad = _key_pad(k.size(1))
    k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (k, v))
    qh, kh, vh = (x.to(work).transpose(1, 2) for x in (q, k, v))
    score_mod = None
    if bias_cols is not None:
        bias = F.pad(bias_cols.to(work), (0, pad))

        def score_mod(score, b, h, qi, ki):
            return score + bias[ki]
    # Rows with no visible key come back as zeros, matching the dense path.
    return _flex_attention()(qh, kh, vh, score_mod=score_mod, block_mask=_block_mask(mask)).transpose(1, 2)[:, :query_len]


def masked_attention(q, k, v, mask: torch.Tensor, bias_cols: Optional[torch.Tensor] = None):
    """q [B,Lq,N,D], k/v [B,Lk,N,D], mask bool [B|1,Lq,Lk]. Rows with no
    visible key return zeros (not NaN). Output dtype follows q."""
    out_dtype = q.dtype
    work = v.dtype if v.dtype in (torch.float16, torch.bfloat16) and q.is_cuda else torch.float32
    if ATTN_IMPL == "flex" and q.is_cuda and q.size(1) >= FLEX_MIN_QUERIES:
        return _flex_masked_attention(q, k, v, mask, bias_cols, work).to(out_dtype)
    qh, kh, vh = (x.to(work).transpose(1, 2) for x in (q, k, v))
    if bias_cols is None:
        attn_mask = mask[:, None]
    else:
        attn_mask = torch.where(mask, bias_cols.to(work)[None, None, :],
                                torch.tensor(float("-inf"), dtype=work, device=mask.device))[:, None]
    empty = ~mask.any(dim=-1)                                   # [B|1, Lq]
    if bool(empty.any()):
        if bias_cols is None:
            attn_mask = attn_mask | empty[:, None, :, None]
        else:
            attn_mask = torch.where(empty[:, None, :, None], torch.zeros((), dtype=work, device=mask.device), attn_mask)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask).transpose(1, 2)
    if bool(empty.any()):
        out = out.masked_fill(empty[:, :, None, None], 0.0)
    return out.to(out_dtype)


def rope_apply_pos(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Wan 3D RoPE with explicit (f,h,w) per token; negative f = conjugate.

    Numerically identical to ``wan.modules.model.rope_apply`` when pos is the
    arange grid of an unpadded sequence. x [B,L,N,2C]; pos [L,3] or [B,L,3].
    """
    b, length, n, cc = x.shape
    c = cc // 2
    c_f = c - 2 * (c // 3)
    c_h = c // 3
    fpart, hpart, wpart = freqs.to(x.device).split([c_f, c_h, c // 3], dim=1)
    pos = pos.to(x.device)
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    if pos.size(-2) != length:
        raise ValueError(f"rope positions cover {pos.size(-2)} tokens, sequence has {length}")
    if pos.is_floating_point():
        # The first table step contains the angular frequencies (all <=1 rad).
        # Evaluate real time / pooled-cell centres directly, including old
        # events beyond the precomputed table, without rounding or slot shifts.
        def at(part, coordinate):
            angle = coordinate.double()[..., None] * torch.angle(part[1]).double()
            return torch.polar(torch.ones_like(angle), angle)
        fr = torch.cat([at(p, pos[..., i]) for i, p in enumerate((fpart, hpart, wpart))], dim=-1)
    else:
        f_idx = pos[..., 0]
        if int(f_idx.abs().max()) >= fpart.size(0) or int(pos[..., 1:].max()) >= hpart.size(0):
            raise ValueError("rope position exceeds the frequency table")
        ff = fpart[f_idx.abs()]
        ff = torch.where((f_idx < 0)[..., None], ff.conj(), ff)
        fr = torch.cat([ff, hpart[pos[..., 1]], wpart[pos[..., 2]]], dim=-1)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(b, length, n, -1, 2))
    y = x_c * fr[:, :, None, :]
    return torch.view_as_real(y).reshape(b, length, n, -1).float()


def masked_self_attention(attn, x, pos, freqs, mask, bias_cols=None, kv_state=None):
    """WanSelfAttention.forward with explicit positions and a visibility mask."""
    b, s, n, d = *x.shape[:2], attn.num_heads, attn.head_dim
    q = attn.norm_q(attn.q(x)).view(b, s, n, d)
    k = attn.norm_k(attn.k(x)).view(b, s, n, d)
    v = attn.v(x).view(b, s, n, d)
    q, k = rope_apply_pos(q, pos, freqs), rope_apply_pos(k, pos, freqs)
    if kv_state is not None:
        from thinker_wm.models.prefix_cache import join_rows
        partition = kv_state["partition"]
        if "k" in kv_state:
            k = join_rows(partition, kv_state["k"], k)
            v = join_rows(partition, kv_state["v"], v)
        else:
            kv_state["k"] = k.index_select(1, partition.fixed).detach()
            kv_state["v"] = v.index_select(1, partition.fixed).detach()
    y = masked_attention(q, k, v, mask, bias_cols)
    return attn.o(y.flatten(2))


def wan_block_forward_masked(block, x, e, pos, freqs, context, context_lens, mask, bias_cols=None,
                             prefix_cache=None, cache_key=None, fixed=None):
    """``WanAttentionBlock.forward`` with masked, position-explicit self-attention.

    Cross-attention (to und / robot tokens) and FFN are the block's own modules.
    """
    kv_state, cached, partition = None, None, None
    if prefix_cache is not None:
        partition = prefix_cache.partition(("stream", cache_key[1]), fixed)
        if cache_key[1] == "action" and partition.moving.numel():
            partition = None  # keep the original small-action BF16 GEMM shape
    if partition is not None and partition.fixed.numel():
        cached = prefix_cache.get(cache_key)
        if cached is not None:
            prefix_cache.reused_rows += partition.fixed.numel()
            if not partition.moving.numel():
                return cached["output"]
            kv_state = dict(partition=partition, k=cached["k"], v=cached["v"])
            x, e = x.index_select(1, partition.moving), e.index_select(1, partition.moving)
            pos = pos.index_select(-2, partition.moving)
            mask = mask.index_select(1, partition.moving)
        else:
            kv_state = dict(partition=partition)
    assert e.dtype == torch.float32
    with torch.amp.autocast("cuda", dtype=torch.float32):
        e = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    y = masked_self_attention(
        block.self_attn,
        block.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        pos, freqs, mask, bias_cols, kv_state=kv_state)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        x = x + y * e[2].squeeze(2)
    x = x + block.cross_attn(block.norm3(x), context, context_lens)
    y = block.ffn(block.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
    with torch.amp.autocast("cuda", dtype=torch.float32):
        x = x + y * e[5].squeeze(2)
    if kv_state is not None:
        from thinker_wm.models.prefix_cache import join_rows
        if cached is not None:
            x = join_rows(partition, cached["output"], x)
        else:
            prefix_cache.put(cache_key, dict(k=kv_state["k"], v=kv_state["v"],
                                            output=x.index_select(1, partition.fixed).detach()))
    return x
