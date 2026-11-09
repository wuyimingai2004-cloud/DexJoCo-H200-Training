"""Real ``UBT_WM`` training/inference methods on a tiny real world model (CPU).

Only the frozen, heavy pieces are replaced: the Wan VAE by a deterministic
causal-layout stand-in (latent 0 = frame 0, latent f = frames 4f-3..4f), the
VLM by a fixed text hash feature, and per-robot normalisation by identity.
"""
from __future__ import annotations

import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import _rsi_cpu_stubs as S
from thinker_wm.models.ubt_wm import MotusConfig, UBT_WM

H_PIX = W_PIX = 8          # latent 4x4 (stride 2), tiny Wan patch (1,1,1)
VLM_DIM = 2048


def config(**over) -> MotusConfig:
    base = dict(num_video_frames=8, video_action_freq_ratio=4, action_t_downsample=4,
                video_height=H_PIX, video_width=W_PIX, action_dim=8, action_state_dim=8,
                unified_action_dim=8, independent_modality_noise=True,
                temporal_mode="block_causal", control_hz=25.0)
    base.update(over)
    return MotusConfig(**base)


class Harness(UBT_WM):
    def __init__(self, cfg: MotusConfig, seed: int = 0):
        nn.Module.__init__(self)
        from thinker_wm.models import temporal_training as TT
        TT.validate_temporal_config(cfg)
        self.config = cfg
        self.dtype = torch.float32
        self._modality_dropout_sample_idx = 0
        self._temporal_sample_idx = 0
        self._action_flow_shift = 1.0
        self._repr_specs = None
        self._norm_policy = None
        self.fm_train_scheduler_action = None
        self.world_model = S.tiny_world(
            enable_depth=bool(cfg.enable_depth), enable_history=bool(cfg.history_enable),
            history_pool=int(cfg.history_spatial_pool),
            recovery_feature_dim=int(cfg.recovery_feature_dim) if cfg.recovery_condition_enable else 0,
            action_dim=int(cfg.unified_action_dim), t_downsample=int(cfg.action_t_downsample),
            ratio=int(cfg.video_action_freq_ratio), seed=seed)
        g = torch.Generator().manual_seed(7)
        self.register_buffer("_vae_proj", torch.randn(8, 3, generator=g) * 0.5)
        und = self.world_model.und_branch

        def vlm_feats(messages, pils=None, **_):
            rows = []
            for m in messages:
                gg = torch.Generator().manual_seed(zlib.crc32(str(m).encode()))
                rows.append(torch.randn(5, VLM_DIM, generator=gg) * 0.1)
            return torch.stack(rows), torch.full((len(messages),), 5, dtype=torch.long)

        und.precompute_vlm_features = vlm_feats
        und.encode = lambda messages, pils=None, **k: und.encode_from_vlm_hidden(*vlm_feats(messages))
        und.get_uncond_vlm_feature = lambda: (torch.zeros(1, 1, VLM_DIM), torch.ones(1, dtype=torch.long))
        und._last_rewrite_cache_stats = (0, 0)

    @property
    def device(self):
        return torch.device("cpu")

    def encode_video(self, x):
        x = F.avg_pool3d(x.float(), (1, 2, 2))
        first = x[:, :, :1]
        lat = [first] if x.shape[2] == 1 else [first, F.avg_pool3d(x[:, :, 1:], (4, 1, 1))]
        z = torch.cat(lat, dim=2)
        return torch.einsum("oc,bcthw->bothw", self._vae_proj, z)

    def decode_video(self, z):
        x = torch.einsum("oc,bothw->bcthw", self._vae_proj, z)
        x = torch.cat([x[:, :, :1], x[:, :, 1:].repeat_interleave(4, dim=2)], dim=2)
        return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

    def _normalize_state_actions(self, state, actions, names):
        return state, actions

    def _denormalize_actions(self, actions, names):
        return actions

    def _gather_native_actions(self, a, state_local, names, cols, offset, dim):
        return a[..., offset:offset + dim]

    def _log_depth_latent_stats(self, *a, **k):
        pass

    def _sample_flow_timestep(self, b, scheduler=None):
        g = torch.Generator().manual_seed(11 + (scheduler is not None))
        s = torch.rand(b, 1, generator=g) * 0.8 + 0.1
        return s * 1000.0, s.view(b, 1, 1, 1, 1), s.view(b, 1, 1)


def batch(b=2, seed=0, frames=8, ratio=4, depth=False, history_slots=0):
    g = torch.Generator().manual_seed(seed)
    out = dict(first_frame=torch.rand(b, 3, H_PIX, W_PIX, generator=g),
               video_frames=torch.rand(b, frames, 3, H_PIX, W_PIX, generator=g),
               state=torch.randn(b, 8, generator=g),
               actions=torch.randn(b, frames * ratio, 8, generator=g),
               vlm_input_messages=[f"task {i}" for i in range(b)])
    if depth:
        out["depth_first"] = torch.rand(b, 3, H_PIX, W_PIX, generator=g)
        out["depth_frames"] = torch.rand(b, frames, 3, H_PIX, W_PIX, generator=g)
        out["depth_valid_mask"] = torch.ones(b, frames + 1, H_PIX, W_PIX, dtype=torch.bool)
    if history_slots:
        out["history_inputs"] = history_inputs(b, history_slots, g)
    return out


def history_inputs(b, sparse_slots, g, recent_frames=5, frame_dt=0.16):
    """Sparse slots at -2^k s (oldest first) + one recent 1+4m clip ending one frame ago."""
    ts = -torch.tensor([2.0 ** k for k in range(sparse_slots, 0, -1)]).repeat(b, 1)
    sv = torch.ones(b, sparse_slots, dtype=torch.bool)
    sv[-1, 0] = False
    tr = -frame_dt * torch.arange(recent_frames, 0, -1, dtype=torch.float32).repeat(b, 1)
    rv = torch.ones(b, recent_frames, dtype=torch.bool)
    rv[-1, 0] = False
    return dict(
        history_sparse_rgb=torch.rand(b, sparse_slots, 3, H_PIX, W_PIX, generator=g),
        history_sparse_times=ts, history_sparse_valid=sv,
        history_sparse_kind=torch.ones(b, sparse_slots, dtype=torch.long),
        history_sparse_state=torch.randn(b, sparse_slots, 8, generator=g),
        history_recent_rgb=torch.rand(b, recent_frames, 3, H_PIX, W_PIX, generator=g),
        history_recent_times=tr, history_recent_valid=rv,
        history_recent_state=torch.randn(b, recent_frames, 8, generator=g),
    )
