"""Visual history slots for the block-causal WAM.

History frames are encoded by the frozen Wan VAE from history pixels only
(recent clip as one causal clip, sparse frames one by one), then patch-embedded
with the video branch's own patch embedding (same latent space). Optional
spatial pooling bounds the token budget. Real time offsets enter twice: as
negative RoPE frame positions (temporal_layout.history_segment) and as a
zero-initialised additive embedding. Slot kind (recent / sparse / episode
start / event-before / event-after) and the measured state at that time are zero-initialised adapters too.

``logit_bias`` is added to every attention logit that targets a history key.
It starts strongly negative so that a converted bidirectional checkpoint
initially attends mostly to the current window; it is learned afterwards.
This is an approximation of "start from the old behaviour", not an exact one.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinker_wm.wan.modules.model import sinusoidal_embedding_1d


class HistoryEncoder(nn.Module):
    def __init__(self, dim: int, state_dim: int, freq_dim: int = 256, pool: int = 1,
                 logit_bias_init: float = -6.0, num_kinds: int = 5):
        super().__init__()
        if pool < 1:
            raise ValueError("history pool must be >= 1")
        self.dim, self.freq_dim, self.pool = dim, freq_dim, int(pool)
        self.stream_embed = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.time_proj = nn.Linear(freq_dim, dim)
        self.kind_embed = nn.Embedding(num_kinds, dim)
        self.state_proj = nn.Linear(state_dim, dim)
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias_init)))
        for m in (self.time_proj, self.state_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.kind_embed.weight)

    def grid(self, h: int, w: int):
        return (h + self.pool - 1) // self.pool, (w + self.pool - 1) // self.pool

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "kind_embed.weight"
        old = state_dict.get(key)
        if old is not None and old.shape == (3, self.dim) and self.kind_embed.num_embeddings == 5:
            state_dict[key] = torch.cat([old, old.new_zeros(2, self.dim)], dim=0)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, patch_embedding: nn.Module, latents: torch.Tensor, times_s: torch.Tensor,
                valid: torch.Tensor, kind: Optional[torch.Tensor] = None,
                state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """latents [B,C,Th,H,W] -> tokens [B, Th*hp*wp, dim] (slot-major)."""
        b, _, th = latents.shape[:3]
        if times_s.shape != (b, th) or valid.shape != (b, th):
            raise ValueError("history times/valid must be [B, Th]")
        u = patch_embedding(latents)                              # [B,D,Th,h,w]
        if self.pool > 1:
            u = F.avg_pool3d(u.float(), kernel_size=(1, self.pool, self.pool),
                            ceil_mode=True, count_include_pad=False).to(u.dtype)
        d, hp, wp = u.shape[1], u.shape[3], u.shape[4]
        tok = u.permute(0, 2, 3, 4, 1).reshape(b, th, hp * wp, d)
        cent = (times_s.float() * 100.0).flatten()
        temb = sinusoidal_embedding_1d(self.freq_dim, cent).float().view(b, th, -1)
        add = self.time_proj(temb.to(self.time_proj.weight.dtype))
        if kind is not None:
            add = add + self.kind_embed(kind.long())
        if state is not None:
            add = add + self.state_proj(state.to(self.state_proj.weight.dtype))
        tok = tok + (self.stream_embed + add[:, :, None, :]).to(tok.dtype)
        tok = tok * valid[:, :, None, None].to(tok.dtype)
        return tok.reshape(b, th * hp * wp, d)
