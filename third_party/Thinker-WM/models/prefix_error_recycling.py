"""Bounded prefix error replay: Joint RGB/action pairs; A2V RGB only.

This is a small context-only adaptation of the error-recycling idea, not a
port of Long-WAM's full latent/noise augmentation. Errors come from previous
training forwards, never from the current target before its forward. Actual
history, current observation and target inputs/labels are untouched. Optional
Joint action replay only changes allowed continuous dimensions of past CLEAN
copies, never A2V's specified commands. Paired errors share sample/block/forward;
this is a local robustness approximation, not a new physical transition label.

The CPU buffer is rank-local transient training state (not model weights).
Restarting training clears it and repeats warmup; exact replay after restart
is not promised. No distributed collectives or parameter changes are needed.
"""
from collections import deque
from dataclasses import dataclass
import math

import torch

from thinker_wm.models.temporal_layout import ROLE_CLEAN, ROLE_TARGET


def validate_config(cfg):
    if getattr(cfg, "joint_action_prefix_error_enable", False):
        if not getattr(cfg, "rgb_prefix_error_enable", False):
            raise ValueError("joint_action_prefix_error requires rgb_prefix_error_enable")
        if (getattr(cfg, "action_target_repr", "absolute_native") != "absolute_native"
                or getattr(cfg, "action_norm_policy", "zscore") != "zscore"):
            raise ValueError("joint action prefix replay currently requires absolute_native/zscore")
        dims = getattr(cfg, "joint_action_prefix_error_dims", None)
        if (not dims or any(isinstance(i, bool) or int(i) != i or i < 0
                            or i >= cfg.unified_action_dim for i in dims)
                or len(set(dims)) != len(dims)):
            raise ValueError("joint_action_prefix_error_dims must explicitly list valid continuous columns")
        for name in ("scale", "max_abs"):
            value = float(getattr(cfg, "joint_action_prefix_error_" + name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid joint_action_prefix_error_" + name)
    if not getattr(cfg, "rgb_prefix_error_enable", False):
        return
    if (getattr(cfg, "temporal_mode", None) != "block_causal"
            or not getattr(cfg, "ar_packed_teacher_forcing", False)):
        raise ValueError("rgb_prefix_error requires packed block-causal training")
    for name, lo, hi in (("prob", 0., 1.), ("scale", 0., float("inf")),
                         ("max_rms_ratio", 0., 1.), ("max_mb", 0., float("inf"))):
        value = float(getattr(cfg, "rgb_prefix_error_" + name))
        if not math.isfinite(value) or not lo <= value <= hi or (name == "max_mb" and value == 0):
            raise ValueError("invalid rgb_prefix_error_" + name)
    for name, minimum in (("warmup_steps", 0), ("capacity", 1)):
        value = getattr(cfg, "rgb_prefix_error_" + name)
        if int(value) != value or int(value) < minimum:
            raise ValueError("invalid rgb_prefix_error_" + name)


def empty_metrics():
    return dict(rgb_prefix_error_active=0, rgb_prefix_error_applied_blocks=0,
                rgb_prefix_error_max_rms_ratio=0., rgb_prefix_error_mean_rms_ratio=0.,
                rgb_prefix_error_raw_rms_ratio=0., rgb_prefix_error_clipped_frac=0.,
                rgb_prefix_error_entries=0,
                rgb_prefix_error_bytes=0, joint_action_prefix_error_active=0,
                joint_action_prefix_error_applied_blocks=0,
                joint_action_prefix_error_max_abs=0.)


@dataclass
class PrefixError:
    rgb: torch.Tensor
    action: torch.Tensor = None
    action_valid: torch.Tensor = None

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.rgb, self.action, self.action_valid) if x is not None)


def _prefix_blocks(entries):
    blocks = {}
    for packed_index, (src, block, role) in enumerate(entries):
        if role == ROLE_CLEAN and block > 0:
            blocks.setdefault(block, []).append((packed_index, src))
    return blocks


class RGBPrefixErrorBuffer:
    def __init__(self, cfg):
        validate_config(cfg)
        self.prob = float(cfg.rgb_prefix_error_prob)
        self.scale = float(cfg.rgb_prefix_error_scale)
        self.max_ratio = float(cfg.rgb_prefix_error_max_rms_ratio)
        self.warmup = int(cfg.rgb_prefix_error_warmup_steps)
        self.capacity = int(cfg.rgb_prefix_error_capacity)
        self.max_bytes = int(float(cfg.rgb_prefix_error_max_mb) * 1024**2)
        self.items = deque()
        self.nbytes = 0
        self.seen = {}
        self.joint_action = bool(getattr(cfg, "joint_action_prefix_error_enable", False))
        self.action_dims = tuple(getattr(cfg, "joint_action_prefix_error_dims", None) or ())
        self.action_scale = float(getattr(cfg, "joint_action_prefix_error_scale", .25))
        self.action_max_abs = float(getattr(cfg, "joint_action_prefix_error_max_abs", .02))

    @staticmethod
    def _key(mode, block, shape, group):
        return mode, int(block), tuple(shape), group

    def metrics(self):
        return dict(rgb_prefix_error_entries=len(self.items), rgb_prefix_error_bytes=self.nbytes)

    @torch.no_grad()
    def perturb(self, packed, entries, mode, step, groups, valid):
        video, _, metrics = self.perturb_paired(packed, entries, mode, step, groups, valid)
        return video, metrics

    @torch.no_grad()
    def perturb_paired(self, packed, entries, mode, step, groups, valid,
                       action_packed=None, action_entries=(), action_valid=None):
        metrics = empty_metrics()
        metrics["rgb_prefix_error_active"] = 1
        joint = self.joint_action and mode == "joint_policy" and action_packed is not None
        metrics["joint_action_prefix_error_active"] = int(joint)
        action_blocks = _prefix_blocks(action_entries) if joint else {}
        metrics.update(self.metrics())
        seen = self.seen.get(mode, 0)
        if seen < self.warmup or not self.items or self.scale == 0 or self.max_ratio == 0:
            return packed, action_packed, metrics
        gen = torch.Generator(device="cpu").manual_seed(20261015 * 1_000_003 + int(step))
        # One gate per call preserves a fraction of completely clean prefixes.
        if torch.rand((), generator=gen).item() >= self.prob:
            return packed, action_packed, metrics
        result = None
        action_result = None
        ratios, raw_ratios, clipped = [], [], 0
        for block, slots in _prefix_blocks(entries).items():
            idx = [i for i, _ in slots]
            for b in range(packed.shape[0]):
                if not valid[b]:
                    continue
                ref = packed[b, :, idx].float()
                action_idx = [i for i, _ in action_blocks.get(block, [])]
                action_src = [s for _, s in action_blocks.get(block, [])]
                action_ref = action_packed[b, action_idx].float() if action_idx else None
                mask = None
                if action_ref is not None:
                    if action_valid is None:
                        raise ValueError("paired action replay requires an explicit validity mask")
                    mask = action_valid[b, action_src].clone()
                    allowed = torch.zeros(action_ref.shape[-1], dtype=torch.bool, device=mask.device)
                    allowed[list(self.action_dims)] = True
                    mask &= allowed.unsqueeze(0)
                need_pair = mask is not None and bool(mask.any())
                key = self._key(mode, block, ref.shape, groups[b])
                choices = [err for k, err in self.items if k == key]
                if need_pair:
                    choices = [e for e in choices if e.action is not None
                               and e.action.shape == action_ref.shape]
                if not choices or not torch.isfinite(ref).all():
                    continue
                i = int(torch.randint(len(choices), (), generator=gen))
                error = choices[i]
                delta = error.rgb.to(device=ref.device, dtype=torch.float32) * self.scale
                ref_rms = ref.square().mean().sqrt().clamp_min(1e-6)
                delta_rms = delta.square().mean().sqrt()
                factor = (self.max_ratio * ref_rms / delta_rms.clamp_min(1e-12)).clamp(max=1.)
                action_delta = None
                if need_pair:
                    mask &= error.action_valid.to(mask.device)
                    action_delta = error.action.to(action_ref.device) * self.action_scale
                    action_delta = torch.where(mask, action_delta, torch.zeros_like(action_delta))
                    # A shared attenuation factor keeps the sampled pair together.
                    factor = torch.minimum(factor, (self.action_max_abs /
                                           action_delta.abs().amax().clamp_min(1e-12)).clamp(max=1.))
                delta = delta * factor
                value = (ref + delta).to(packed.dtype)
                if not torch.isfinite(value).all():
                    continue
                action_value = None
                if action_delta is not None and bool(mask.any()):
                    action_value = (action_ref + action_delta * factor).to(action_packed.dtype)
                    # Preserve unselected/invalid columns bit-for-bit, including
                    # their original dtype. Do not alter gripper state semantics.
                    action_value = torch.where(mask, action_value, action_packed[b, action_idx])
                    if not torch.isfinite(action_value[mask]).all():
                        continue
                if result is None:
                    result = packed.clone()
                result[b, :, idx] = value
                if action_value is not None:
                    if action_result is None:
                        action_result = action_packed.clone()
                    action_result[b, action_idx] = action_value
                    metrics["joint_action_prefix_error_applied_blocks"] += 1
                    max_abs = float((action_value.float() - action_ref)[mask].abs().amax())
                    metrics["joint_action_prefix_error_max_abs"] = max(
                        metrics["joint_action_prefix_error_max_abs"], max_abs)
                metrics["rgb_prefix_error_applied_blocks"] += 1
                # Report the actual perturbation after dtype conversion.
                ratio = float((value.float() - ref).square().mean().sqrt() / ref_rms)
                metrics["rgb_prefix_error_max_rms_ratio"] = max(metrics["rgb_prefix_error_max_rms_ratio"], ratio)
                ratios.append(ratio)
                raw_ratios.append(float(delta_rms / ref_rms))
                clipped += int(float(factor) < 1.)
        if ratios:
            metrics["rgb_prefix_error_mean_rms_ratio"] = sum(ratios) / len(ratios)
            metrics["rgb_prefix_error_raw_rms_ratio"] = sum(raw_ratios) / len(raw_ratios)
            metrics["rgb_prefix_error_clipped_frac"] = clipped / len(ratios)
        return (packed if result is None else result,
                action_packed if action_result is None else action_result, metrics)

    @torch.no_grad()
    def observe(self, noisy, velocity, clean, sigma, entries, mode, groups, valid,
                action_noisy=None, action_velocity=None, action_clean=None,
                action_sigma=None, action_entries=(), action_valid=None):
        """Flow matching: x0_hat = x_sigma - sigma*v_hat; e = x0_hat-x0.

        Keep only target blocks that can become a prefix. Never store current
        frame, invalid/zero-weight samples, nonfinite errors or graph tensors.
        Partition by mode, source/robot, shape and temporal block position.
        """
        targets = {src for src, _, role in entries if role == ROLE_TARGET}
        joint = self.joint_action and mode == "joint_policy" and action_velocity is not None
        action_blocks = _prefix_blocks(action_entries) if joint else {}
        action_targets = {src for src, _, role in action_entries if role == ROLE_TARGET}
        for block, slots in _prefix_blocks(entries).items():
            src = [s for _, s in slots]
            if not all(s in targets for s in src):
                continue
            for b in range(clean.shape[0]):
                if not valid[b]:
                    continue
                err = (noisy[b, :, src].float()
                       - sigma[b].float() * velocity[b, :, src].float()
                       - clean[b, :, src].float())
                if not torch.isfinite(err).all():
                    continue
                # FP32 CPU entries avoid overflow on transfer; budget counts bytes.
                err = err.detach().to(device="cpu", dtype=torch.float32, copy=True)
                action_error = action_mask = None
                action_src = [s for _, s in action_blocks.get(block, [])]
                if action_src and all(s in action_targets for s in action_src):
                    if action_valid is None:
                        raise ValueError("paired action replay requires an explicit validity mask")
                    action_mask = action_valid[b, action_src].clone()
                    allowed = torch.zeros(action_clean.shape[-1], dtype=torch.bool, device=action_mask.device)
                    allowed[list(self.action_dims)] = True
                    action_mask &= allowed.unsqueeze(0)
                    action_error = (action_noisy[b, action_src].float()
                                    - action_sigma[b].float() * action_velocity[b, action_src].float()
                                    - action_clean[b, action_src].float())
                    if not bool(action_mask.any()) or not torch.isfinite(action_error[action_mask]).all():
                        action_error = action_mask = None
                    else:
                        action_error = torch.where(action_mask, action_error, torch.zeros_like(action_error))
                        action_error = action_error.detach().to("cpu", copy=True)
                        action_mask = action_mask.detach().to("cpu", copy=True)
                error = PrefixError(err, action_error, action_mask)
                size = error.nbytes
                if size > self.max_bytes:
                    continue
                key = self._key(mode, block, err.shape, groups[b])
                # Capacity is per key so frequent modes cannot evict rarer ones;
                # the byte budget is global and evicts the oldest entry.
                same = [i for i, (k, _) in enumerate(self.items) if k == key]
                if len(same) >= self.capacity:
                    _, old = self.items[same[0]]
                    del self.items[same[0]]
                    self.nbytes -= old.nbytes
                while self.items and self.nbytes + size > self.max_bytes:
                    _, old = self.items.popleft()
                    self.nbytes -= old.nbytes
                self.items.append((key, error))
                self.nbytes += size
        self.seen[mode] = self.seen.get(mode, 0) + 1
