"""Decoded depth output geometry. No GT depth, model noise, or latent edits."""
from __future__ import annotations

import torch


def depth_content_mask(camera_transform: str, hw, *, device=None):
    """Return an HW bool mask for a known padded camera transform, else None.

    The mask describes camera coverage, not measured-depth validity. Never infer
    it from black RGB pixels (real objects can be black) or predicted depth.
    """
    prefix = "robotwin_3d/motus_t/pad/"
    if not camera_transform.startswith(prefix):
        return None
    h, w = map(int, hw)
    if camera_transform[len(prefix):] != f"{h}x{w}":
        raise ValueError("depth shape does not match camera_transform")
    from thinker_wm.data_prepare.robotwin_3d import motus_t_view_slots
    r0, r1, c0, c1 = motus_t_view_slots((h, w))["content"]
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    mask[r0:r1, c0:c1] = True
    return mask


def mask_decoded_depth(depth, content_mask):
    """Zero known padding on [...,H,W] normalized [0,1] depth, out of place.

    Zero is a display sentinel, NOT a valid far-plane measurement. Geometry
    consumers must also use content_mask, including after metric decoding.
    """
    if depth is None or content_mask is None:
        return depth
    if content_mask.dtype != torch.bool or tuple(content_mask.shape) != tuple(depth.shape[-2:]):
        raise ValueError("depth content mask must be bool HW matching decoded depth")
    return torch.where(content_mask.to(depth.device), depth, torch.zeros_like(depth))
