"""Depth target contracts shared by training and sampling (no backbone imports).

Validity is conservatively pooled within nominal VAE cells, including temporal
bins. This is only a latent-mask approximation: the VAE receptive field crosses
cell boundaries, so it cannot exactly remove all influence of invalid pixels.
Invalid pixels are neutralized before encoding; invalid cells never serve as GT.
"""
import torch
import torch.nn.functional as F


def depth_condition_mode(config):
    mode = getattr(config, "depth_condition_mode", "first_frame")
    if mode not in ("first_frame", "rgb_only"):
        raise ValueError(f"Unsupported depth_condition_mode: {mode!r}")
    return mode


def validate_depth_losses(config, masked=False):
    if depth_condition_mode(config) == "rgb_only" or masked:
        if (getattr(config, "depth_motion_loss_enable", False)
                or getattr(config, "depth_edge_loss_enable", False)):
            raise ValueError("Masked/rgb_only depth does not support motion/edge loss")


def sanitize_depth_targets(full_depth, valid_mask, depth_present=None):
    """Input normalized [B,C,T,H,W], mask bool [B,T,H,W]; fill invalid with 0.

    In rgb_only mode depth_first is a *training label*, never a condition.
    The caller must reject it at inference, where labels are forbidden.
    """
    expected = (full_depth.shape[0], *full_depth.shape[2:])
    if (not torch.is_tensor(valid_mask) or valid_mask.dtype != torch.bool
            or tuple(valid_mask.shape) != expected):
        raise ValueError(f"depth_valid_mask must be bool {expected}")
    mask = valid_mask.to(device=full_depth.device)
    if depth_present is not None:
        if tuple(depth_present.shape) != (full_depth.shape[0],):
            raise ValueError("depth_present must have shape [B]")
        mask = mask & depth_present.to(device=mask.device, dtype=torch.bool)[:, None, None, None]
    if not bool((torch.isfinite(full_depth) | ~mask[:, None]).all()):
        raise ValueError("Nonfinite depth at a pixel marked valid")
    return torch.where(mask[:, None], full_depth, 0.0), mask


def depth_latent_valid_mask(valid_mask, latent_shape):
    """Conservative any-invalid pooling, first frame separate, causal VAE stride 4.

    Assumes the existing Wan contract T=1+4*k -> latent T=1+k. Spatial pooling
    uses adaptive max over invalid pixels, never averages or nearest sampling.
    This is NOT an exact VAE receptive-field validity calculation.
    """
    b, _, t, h, w = latent_shape
    if valid_mask.dtype != torch.bool or valid_mask.ndim != 4 or valid_mask.shape[0] != b:
        raise ValueError("depth_valid_mask must be bool [B,T+1,H,W]")
    if valid_mask.shape[1] != 1 + 4 * (t - 1):
        raise ValueError("Depth validity does not match Wan temporal stride 4")
    invalid = (~valid_mask).float().unsqueeze(1)
    first = F.adaptive_max_pool3d(invalid[:, :, :1], (1, h, w))
    if t > 1:
        future = F.adaptive_max_pool3d(invalid[:, :, 1:], (t - 1, h, w))
        first = torch.cat([first, future], dim=2)
    return first == 0


def masked_depth_mse(prediction, target, valid_mask):
    """Per-sample mean over valid latent elements; empty samples have zero loss.

    Returns losses and [B] validity, so callers also exclude empty samples from
    the batch denominator. torch.where before MSE avoids invalid NaN targets.
    """
    expected = (prediction.shape[0], 1, *prediction.shape[2:])
    if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != expected:
        raise ValueError(f"latent depth mask must be bool {expected}")
    mask = valid_mask.to(device=prediction.device).expand_as(prediction)
    p = torch.where(mask, prediction.float(), 0.0)
    y = torch.where(mask, target.float(), 0.0)
    count = mask.sum(dim=(1, 2, 3, 4))
    loss = (p - y).square().sum(dim=(1, 2, 3, 4)) / count.clamp_min(1)
    return loss, count > 0
