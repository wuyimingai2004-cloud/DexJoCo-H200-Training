"""Pixel motion mask → Wan-VAE latent weight for depth loss.

Gates match ``periodic_eval.runner._depth_motion_mask`` (no pose, no CC):
  |Δz| > 0.10 m, blur residual > 0.05 m (ks=41), RGB luma > 0.05.

Latent pool is **mean** (time groups of 4 after the first frame, spatial 16×16)
so motion_frac is preserved. max-pool would inflate f and break λ.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_motion_mask(
    depth_m: torch.Tensor,
    valid: torch.Tensor,
    rgb: torch.Tensor | None = None,
    thresh_m: float = 0.10,
    rgb_thresh: float = 0.05,
    residual_m: float = 0.05,
    blur_ks: int = 41,
) -> torch.Tensor:
    """Batched object-motion mask. ``depth_m``/``valid`` are ``[B, T, H, W]``."""
    if depth_m.shape[1] < 2:
        return torch.zeros_like(valid)
    pair_ok = valid[:, 1:] & valid[:, :-1]
    dz = (depth_m[:, 1:] - depth_m[:, :-1]).abs()
    moving = pair_ok & (dz > thresh_m)
    if residual_m > 0 and blur_ks >= 3:
        ks = int(blur_ks) | 1
        pad = ks // 2
        b, t, h, w = dz.shape
        x = dz.reshape(b * t, 1, h, w)
        lf = F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode="replicate"),
            kernel_size=ks,
            stride=1,
        ).reshape(b, t, h, w)
        moving = moving & ((dz - lf) > residual_m)
    if rgb is not None:
        if rgb.dim() == 5 and rgb.shape[2] in (1, 3):
            luma = rgb.float().mean(dim=2)
        elif rgb.dim() == 5 and rgb.shape[1] in (1, 3):
            luma = rgb.float().mean(dim=1)
        else:
            luma = rgb.float()
        t = min(luma.shape[1], depth_m.shape[1])
        di = (luma[:, 1:t] - luma[:, : t - 1]).abs()
        if di.shape[-2:] != dz.shape[-2:]:
            b, tt, _, _ = di.shape
            di = F.interpolate(
                di.reshape(b * tt, 1, *di.shape[-2:]),
                size=dz.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(b, tt, *dz.shape[-2:])
        moving = moving[:, : di.shape[1]] & (di > rgb_thresh)
    out = torch.zeros_like(valid)
    n = moving.shape[1]
    out[:, 1 : n + 1] |= moving
    out[:, :n] |= moving
    return out


def pool_motion_to_latent(
    mask: torch.Tensor,
    latent_t: int,
    latent_hw: tuple[int, int],
) -> torch.Tensor:
    """Mean-pool pixel mask ``[B, T, H, W]`` → ``[B, 1, Lt, Lh, Lw]``.

    Wan layout: latent 0 = pixel 0; latent k≥1 = mean of the next 4 pixels.
    """
    b, t, h, w = mask.shape
    lh, lw = latent_hw
    x = F.adaptive_avg_pool2d(mask.reshape(b * t, 1, h, w).float(), (lh, lw))
    x = x.reshape(b, t, lh, lw)
    if t == latent_t:
        pooled = x
    elif t >= 2 and 1 + (t - 1) // 4 == latent_t and (t - 1) % 4 == 0:
        rest = x[:, 1:].reshape(b, latent_t - 1, 4, lh, lw).mean(dim=2)
        pooled = torch.cat([x[:, :1], rest], dim=1)
    else:
        pooled = F.adaptive_avg_pool1d(
            x.permute(0, 2, 3, 1).reshape(b * lh * lw, 1, t), latent_t
        ).reshape(b, lh, lw, latent_t).permute(0, 3, 1, 2)
    return pooled.unsqueeze(1)


def depth_motion_latent_weight(
    depth_01: torch.Tensor,
    rgb_01: torch.Tensor,
    latent_t: int,
    latent_hw: tuple[int, int],
    *,
    z_near: float = 0.15,
    z_far: float = 4.0,
    mode: str = "log",
    thresh_m: float = 0.10,
    rgb_thresh: float = 0.05,
    residual_m: float = 0.05,
    blur_ks: int = 41,
    frac_max: float = 0.12,
    require_pose: bool = False,
    ego_pose_used: torch.Tensor | None = None,
    ego_poses: torch.Tensor | None = None,
    ego_head_valid: torch.Tensor | None = None,
    ego_K: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``depth_01``/``rgb_01`` are ``[B, T, 3, H, W]`` in [0, 1].

    Agi (or Ego without pose) uses the 3-gate mask. Ego samples with
    ``ego_pose_used[b]=True`` replace that sample's mask with pose-aligned gates.

    ``require_pose`` zeroes the weight for every sample that did not get a
    pose-aligned mask. On Agi the raw 3-gate mask fires on DA3 temporal jitter
    (~92% of in-mask pixels are static per head-camera sensor depth), so
    upweighting it trains the model on label noise.

    Returns latent weight ``[B, 1, Lt, Lh, Lw]`` and per-sample pixel
    ``motion_frac`` ``[B]``. Samples with ``frac > frac_max`` get weight 0
    (uniform-only fallback).
    """
    from thinker_wm.models.depth_codec import decode_to_depth_m

    depth_m, valid = decode_to_depth_m(
        depth_01, z_near=z_near, z_far=z_far, mode=mode, in_range="01"
    )
    pix = depth_motion_mask(
        depth_m,
        valid,
        rgb=rgb_01,
        thresh_m=thresh_m,
        rgb_thresh=rgb_thresh,
        residual_m=residual_m,
        blur_ks=blur_ks,
    )
    posed = torch.zeros(pix.shape[0], dtype=torch.bool, device=pix.device)
    if (
        ego_pose_used is not None
        and bool(ego_pose_used.any())
        and ego_poses is not None
        and ego_head_valid is not None
        and ego_K is not None
    ):
        from thinker_wm.models.ego_pose import pose_motion_mask_torch

        device = depth_m.device
        idx = torch.where(ego_pose_used.bool().view(-1).to(device))[0]
        luma = rgb_01.float().mean(dim=2)
        aligned, ok = pose_motion_mask_torch(
            depth_m.index_select(0, idx),
            valid.index_select(0, idx),
            luma.index_select(0, idx),
            ego_poses.to(device=device).index_select(0, idx),
            ego_head_valid.to(device=device).index_select(0, idx),
            ego_K.to(device=device).index_select(0, idx),
            thresh_m=thresh_m,
            rgb_thresh=rgb_thresh,
            residual_m=residual_m,
            blur_ks=blur_ks,
        )
        if ok.any():
            pix[idx[ok]] = aligned[ok]
            posed[idx[ok]] = True
    den = valid.flatten(1).sum(dim=1).clamp_min(1)
    frac = pix.flatten(1).float().sum(dim=1) / den
    weight = pool_motion_to_latent(pix, latent_t, latent_hw)
    drop = frac > frac_max
    if require_pose:
        drop = drop | (~posed)
    if drop.any():
        weight = weight.clone()
        weight[drop] = 0
    return weight, frac
