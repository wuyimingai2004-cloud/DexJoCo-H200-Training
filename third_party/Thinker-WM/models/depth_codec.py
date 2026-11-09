"""Depth codec: uint16-mm metric depth <-> model-space 3-channel [-1, 1] grayscale.

Rynn-style "depth as video": metric depth is turned into an RGB-shaped grayscale
tensor that the frozen Wan VAE can encode with the same path as RGB. Depth is
NOT supervised in metric space, so this codec is a lossy *view* for the VAE, not
a lossless store. The lossless ground truth stays on disk as uint16 mm sidecars.

Convention (fixed across encode/decode/train/infer):
  - Input depth is uint16/int/float millimetres; ``0`` means invalid/pad.
  - Normalized value ``v`` in [0, 1]: **near = 1 (bright), far = 0 (dark)**.
  - Invalid pixels map to ``v = 0`` ("far"). Set ``invalid_to="near"`` to flip.
  - ``out_range="01"`` returns ``v`` in [0, 1] (mirrors the RGB ``video_frames``
    boundary, which the model shifts to [-1, 1] before the VAE); ``out_range=
    "pm1"`` returns ``v * 2 - 1`` in [-1, 1] directly. Either way replicated to
    3 identical channels.

Range mappings, all near-bright / far-dark and analytically reversible:
  - ``mode="log"`` (default): equal codes per distance-doubling. Best when the
    scene depth spans several octaves (measured body band ~0.9-4.1 m here).
  - ``mode="inverse"`` (disparity): dense codes very near, heavily compressed
    beyond; ideal only when content sits right at ``z_near``.
  - ``mode="linear"``: uniform in metres; simple but wastes codes on far range.

Only the last two dims are treated as (H, W); a channel axis of size 3 is
inserted just before them, so ``[B,T,H,W] -> [B,T,3,H,W]`` (same layout as the
RGB ``video_frames`` tensor ``[B,T,C,H,W]``).
"""
from __future__ import annotations

import math
from typing import Literal

import torch

Mode = Literal["inverse", "linear", "log"]
InvalidTo = Literal["far", "near"]
OutRange = Literal["pm1", "01"]


def _to_tensor(depth) -> torch.Tensor:
    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)
    return depth


def encode_depth_mm(
    depth_mm,
    z_near: float = 0.1,
    z_far: float = 10.0,
    mode: Mode = "log",
    invalid_to: InvalidTo = "far",
    out_range: OutRange = "pm1",
) -> torch.Tensor:
    """metric depth in mm -> 3-channel model tensor.

    Args:
        depth_mm: ``[..., H, W]`` millimetres; ``0`` = invalid/pad.
        z_near, z_far: clip / normalization range in metres.
        mode: ``"log"`` (default), ``"inverse"`` (disparity), or ``"linear"``.
        invalid_to: where invalid pixels land, ``"far"`` (v=0) or ``"near"``.
        out_range: ``"pm1"`` -> [-1, 1] (``v*2-1``); ``"01"`` -> [0, 1] (``v``,
            mirrors the RGB ``video_frames`` batch boundary).

    Returns:
        ``[..., 3, H, W]`` float32; the 3 channels are identical.
    """
    depth = _to_tensor(depth_mm).float()
    z = depth / 1000.0
    valid = depth > 0

    if mode == "inverse":
        inv_near = 1.0 / z_near
        inv_far = 1.0 / z_far
        z_c = z.clamp(z_near, z_far)
        v = (1.0 / z_c - inv_far) / (inv_near - inv_far)
    elif mode == "log":
        log_near, log_far = math.log(z_near), math.log(z_far)
        z_c = z.clamp(z_near, z_far)
        v = 1.0 - (torch.log(z_c) - log_near) / (log_far - log_near)
    elif mode == "linear":
        v = 1.0 - z.clamp(0.0, z_far) / z_far
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    invalid_v = 0.0 if invalid_to == "far" else 1.0
    v = torch.where(valid, v, torch.full_like(v, invalid_v))
    v = v.clamp(0.0, 1.0)

    x = v if out_range == "01" else v * 2.0 - 1.0
    x = x.unsqueeze(-3).expand(*x.shape[:-2], 3, *x.shape[-2:])
    return x.contiguous()


def decode_to_depth_m(
    x3,
    z_near: float = 0.1,
    z_far: float = 10.0,
    mode: Mode = "log",
    invalid_eps: float = 0.0,
    in_range: OutRange = "pm1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-channel model tensor -> metric depth in metres (for QC / vis).

    Note: both invalid pixels and exactly-``z_far`` depths encode to ``v = 0``
    and are indistinguishable; ``v <= invalid_eps`` (default 0.0, i.e. exactly
    the far endpoint) is flagged invalid. Depths just inside ``z_far`` carry
    tiny positive ``v`` and are kept valid.

    Args:
        x3: ``[..., 3, H, W]`` (or ``[..., 1, H, W]`` / ``[..., H, W]``).
        invalid_eps: ``v <= invalid_eps`` is flagged invalid.
        in_range: range of ``x3``: ``"pm1"`` for [-1, 1], ``"01"`` for [0, 1]
            (must match the ``out_range`` used at encode time).

    Returns:
        (depth_m ``[..., H, W]`` metres, valid_mask ``[..., H, W]`` bool).
        Invalid pixels carry ``z_far`` in ``depth_m`` but ``False`` in the mask.
    """
    x = _to_tensor(x3).float()
    if x.shape[-3] in (1, 3) and x.dim() >= 3:
        # Average the channels instead of taking channel 0. Encoding replicates
        # one value into 3 identical channels, so this is exact for anything that
        # round-trips the codec; for *generated* depth the channels disagree
        # slightly and averaging uses all three estimates instead of discarding
        # two.
        x = x.mean(dim=-3)

    v = (x if in_range == "01" else (x + 1.0) / 2.0).clamp(0.0, 1.0)
    valid = v > invalid_eps

    if mode == "inverse":
        inv_near = 1.0 / z_near
        inv_far = 1.0 / z_far
        inv = v * (inv_near - inv_far) + inv_far
        z = 1.0 / inv.clamp_min(1e-8)
    elif mode == "log":
        log_near, log_far = math.log(z_near), math.log(z_far)
        z = torch.exp((1.0 - v) * (log_far - log_near) + log_near)
    elif mode == "linear":
        z = (1.0 - v) * z_far
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return z, valid
