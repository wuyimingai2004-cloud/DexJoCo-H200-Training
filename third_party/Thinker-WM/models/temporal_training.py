"""Block-causal training helpers (pure functions; used by ``UBT_WM.training_step``).

Per update step one ``TemporalRequest`` is drawn for the whole batch from a
generator seeded by the trainer's global step, so every rank builds the same
layout (identical token counts and masks; required by ZeRO bucket reduction).

Current recipe: Joint + A2V (RGB and depth), one mode per update/all ranks.
Legacy IDM/FDM configurations and interfaces remain compatible:
  inverse_dynamics          clean future RGB goals -> noisy action targets;
                            depth, including D0, stays predicted/supervised.
  joint_policy              predict RGB/depth and actions of the target blocks
  action_conditioned_world  every action is a given command (role OBS, t=0, no
                            action loss); only RGB/depth of target blocks are
                            supervised. Used for consequence prediction and
                            for scoring candidate action chunks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from thinker_wm.models.temporal_layout import (
    ROLE_TARGET, TemporalRequest, TimeGrid, action_step_roles, latent_roles,
    num_target_blocks_check, packed_action_entries, packed_latent_entries,
)

MODE_JOINT = "joint_policy"
MODE_WORLD = "action_conditioned_world"
MODE_IDM = "inverse_dynamics"
MODE_NAMES = {"joint": MODE_JOINT, "idm": MODE_IDM, "fdm": MODE_WORLD}
_SEED_BASE = 20261012


# YAML ``model.wm.<key>`` -> MotusConfig field (absent keys keep MotusConfig defaults).
_WM_KEYS = {
    "temporal_mode": str, "ar_latents_per_block": int, "ar_target_reads_earlier_targets": bool,
    "ar_packed_teacher_forcing": bool,
    "temporal_und_reads": str, "world_mode_prob": float, "history_enable": bool,
    "history_spatial_pool": int, "history_logit_bias_init": float,
    "history_encode_valid_only": bool,
    "recovery_condition_enable": bool, "recovery_feature_dim": int, "recovery_num_tokens": int,
    "rgb_prefix_error_enable": bool, "rgb_prefix_error_prob": float,
    "rgb_prefix_error_scale": float, "rgb_prefix_error_max_rms_ratio": float,
    "rgb_prefix_error_warmup_steps": int, "rgb_prefix_error_capacity": int,
    "rgb_prefix_error_max_mb": float,
    "joint_action_prefix_error_enable": bool,
    "joint_action_prefix_error_scale": float,
    "joint_action_prefix_error_max_abs": float,
}


def motus_temporal_kwargs(config) -> dict:
    """Temporal/history/recovery MotusConfig kwargs from a training/eval YAML config."""
    wm = getattr(getattr(config, "model", None), "wm", None)
    from thinker_wm.models.capability_training import CONFIG_FIELDS
    out = {}
    for key, cast in {**_WM_KEYS, **CONFIG_FIELDS}.items():
        v = getattr(wm, key, None) if wm is not None else None
        if v is not None:
            out[key] = cast(v)
    mixture = getattr(wm, "training_mode_probs", None) if wm is not None else None
    if mixture is not None:
        out["training_mode_probs"] = {str(k): float(v) for k, v in mixture.items()}
    probs = getattr(wm, "ar_cut_probs", None) if wm is not None else None
    if probs is not None:
        out["ar_cut_probs"] = [float(p) for p in probs]
    dims = getattr(wm, "joint_action_prefix_error_dims", None) if wm is not None else None
    if dims is not None:
        out["joint_action_prefix_error_dims"] = list(dims)
    hz = getattr(getattr(config, "common", None), "control_hz", None)
    if hz is not None:
        out["control_hz"] = float(hz)
    elif out.get("temporal_mode") == "block_causal":
        raise ValueError("block_causal needs an explicit common.control_hz (latent seconds, history times)")
    return out


def time_grid_from_config(cfg) -> TimeGrid:
    return TimeGrid(
        video_action_freq_ratio=int(cfg.video_action_freq_ratio),
        action_t_downsample=int(cfg.action_t_downsample),
        latents_per_block=int(getattr(cfg, "ar_latents_per_block", 1)),
        control_hz=float(getattr(cfg, "control_hz", 25.0)),
    )


def training_mode_probs(cfg):
    """Return probabilities in legacy internal order (Joint, IDM, world).

    The current two-mode recipe uses {joint, a2v}. A2V is the existing
    action_conditioned_world path, including depth, with exactly zero IDM
    probability. Keep the old three-key schema for checkpoint/config reuse;
    reject mixed aliases rather than silently adding or replacing a weight.
    """
    mixture = getattr(cfg, "training_mode_probs", None)
    if mixture is None:
        return None
    if set(mixture) == {"joint", "a2v"}:
        values = [float(mixture["joint"]), 0.0, float(mixture["a2v"])]
    elif set(mixture) == set(MODE_NAMES):
        values = [float(mixture[k]) for k in MODE_NAMES]
    else:
        raise ValueError("training_mode_probs requires exactly joint, a2v "
                         "(or legacy joint, idm, fdm); do not mix aliases")
    if any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("training_mode_probs must be finite, non-negative and sum to 1")
    if float(getattr(cfg, "world_mode_prob", 0.0)) != 0:
        raise ValueError("training_mode_probs replaces world_mode_prob; set the latter to 0")
    return values


def validate_temporal_config(cfg) -> None:
    from thinker_wm.models.capability_training import validate_config as validate_capability_config
    validate_capability_config(cfg)
    from thinker_wm.models.prefix_error_recycling import validate_config as validate_error_config
    validate_error_config(cfg)
    mixture = training_mode_probs(cfg)
    mode = str(getattr(cfg, "temporal_mode", "bidirectional"))
    if mode not in ("bidirectional", "block_causal"):
        raise ValueError(f"temporal_mode must be bidirectional|block_causal, got {mode!r}")
    if mode == "bidirectional":
        if mixture is not None:
            raise ValueError("training_mode_probs requires temporal_mode=block_causal")
        for name in ("history_enable", "recovery_condition_enable", "ar_packed_teacher_forcing"):
            if bool(getattr(cfg, name, False)):
                raise ValueError(f"{name} requires temporal_mode=block_causal")
        if float(getattr(cfg, "world_mode_prob", 0.0)) > 0:
            raise ValueError("world_mode_prob requires temporal_mode=block_causal")
        return
    grid = time_grid_from_config(cfg)
    n_lat = grid.num_latents(int(cfg.num_video_frames))
    n_blocks = grid.num_blocks(n_lat)
    if (n_lat - 1) % grid.latents_per_block:
        raise ValueError("num_video_frames/4 must be a multiple of ar_latents_per_block")
    grid.action_token_blocks(int(cfg.num_video_frames) * int(cfg.video_action_freq_ratio))
    probs = getattr(cfg, "ar_cut_probs", None)
    if bool(getattr(cfg, "ar_packed_teacher_forcing", False)):
        if probs:
            raise ValueError("ar_packed_teacher_forcing replaces ar_cut_probs; do not set both")
        if bool(getattr(cfg, "ar_target_reads_earlier_targets", True)):
            raise ValueError("ar_packed_teacher_forcing requires ar_target_reads_earlier_targets=false")
        if n_blocks < 2:
            raise ValueError("ar_packed_teacher_forcing needs at least one future block")
        if bool(getattr(cfg, "zero_action_loss_force_max_noise", False)):
            raise ValueError("zero_action_loss_force_max_noise cannot hide actions under "
                             "ar_packed_teacher_forcing: CLEAN prefix copies carry the real actions")
    if probs:
        if len(probs) > n_blocks - 1:
            raise ValueError(f"ar_cut_probs has {len(probs)} entries, window has only "
                             f"{n_blocks - 1} future blocks")
        if any(p < 0 for p in probs) or sum(probs) <= 0:
            raise ValueError("ar_cut_probs must be non-negative with positive sum")
    if not 0.0 <= float(getattr(cfg, "world_mode_prob", 0.0)) <= 1.0:
        raise ValueError("world_mode_prob must be in [0, 1]")
    for name in ("depth_motion_loss_enable", "depth_edge_loss_enable"):
        if bool(getattr(cfg, name, False)):
            raise ValueError(f"{name} assumes latent 0 is the only clean frame; "
                             "not supported with temporal_mode=block_causal")


@dataclass(frozen=True)
class StepDecision:
    request: TemporalRequest
    mode: str

    def __post_init__(self):
        expected = (MODE_WORLD if self.request.action_role == "world" else
                    MODE_IDM if self.request.video_role == "goal" else MODE_JOINT)
        if self.mode != expected:
            raise ValueError(f"mode {self.mode!r} disagrees with request roles ({expected})")


def sample_step_decision(cfg, seed_step: int, training: bool,
                         depth_first_frame_conditioned: bool) -> StepDecision:
    grid = time_grid_from_config(cfg)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SEED_BASE * 1_000_003 + int(seed_step))
    cut = 1
    probs = getattr(cfg, "ar_cut_probs", None)
    if training and probs:
        p = torch.tensor([float(x) for x in probs], dtype=torch.float64)
        cut = int(torch.multinomial(p / p.sum(), 1, generator=gen).item()) + 1
    mixture = training_mode_probs(cfg)
    selected = MODE_JOINT
    if training:
        if mixture is not None:
            i = int(torch.multinomial(torch.tensor(mixture, dtype=torch.float64), 1, generator=gen))
            selected = list(MODE_NAMES.values())[i]
        elif torch.rand((), generator=gen) < float(getattr(cfg, "world_mode_prob", 0.0)):
            selected = MODE_WORLD
    world = selected == MODE_WORLD
    packed = {}
    if bool(getattr(cfg, "ar_packed_teacher_forcing", False)):
        n_future = grid.num_blocks(grid.num_latents(int(cfg.num_video_frames))) - 1
        packed = dict(packed=True, clean_blocks=n_future - 1, first_target=1, last_target=n_future)
    request = TemporalRequest(
        grid, cut=cut, action_role="world" if world else "policy",
        video_role="goal" if selected == MODE_IDM else "predict",
        depth_first_frame_conditioned=depth_first_frame_conditioned,
        target_reads_earlier_targets=bool(getattr(cfg, "ar_target_reads_earlier_targets", True)),
        und_reads=str(getattr(cfg, "temporal_und_reads", "observed")),
        **packed,
    )
    return StepDecision(request, selected)


def block_request(request: TemporalRequest, block: int) -> TemporalRequest:
    """Packed inference request for generating future block ``block`` alone,
    with blocks 1..block-1 as already generated clean prefix."""
    import dataclasses
    if not request.packed:
        raise ValueError("block-by-block generation needs a packed request")
    return dataclasses.replace(request, clean_blocks=int(block) - 1,
                               first_target=int(block), last_target=int(block))


@dataclass(frozen=True)
class PackPlan:
    """Entries (source index, block, role) of each packed stream."""
    video: tuple
    depth: tuple
    action: tuple


def pack_plan(request: TemporalRequest) -> PackPlan:
    return PackPlan(tuple(packed_latent_entries(request)),
                    tuple(packed_latent_entries(request, depth=True)),
                    tuple(packed_action_entries(request)))


def pack(clean: torch.Tensor, noisy: torch.Tensor, entries, dim: int) -> torch.Tensor:
    """Gather packed stream: TARGET entries from ``noisy``, all others from ``clean``."""
    parts = []
    for src, _, role in entries:
        x = noisy if role == ROLE_TARGET else clean
        parts.append(x.narrow(dim, int(src), 1))
    return torch.cat(parts, dim=dim)


def unpack_targets(packed: torch.Tensor, entries, length: int, dim: int) -> torch.Tensor:
    """Scatter packed TARGET outputs back to window positions (zeros elsewhere)."""
    pos = [i for i, e in enumerate(entries) if e[2] == ROLE_TARGET]
    src = [int(entries[i][0]) for i in pos]
    shape = list(packed.shape)
    shape[dim] = int(length)
    out = packed.new_zeros(shape)
    if not pos:
        return out + packed.sum() * 0.0
    picked = packed.index_select(dim, torch.tensor(pos, dtype=torch.long, device=packed.device))
    return out.index_copy(dim, torch.tensor(src, dtype=torch.long, device=packed.device), picked)


def packed_supervision(plan: PackPlan, num_steps: int) -> "Supervision":
    tgt = lambda es: sorted(int(e[0]) for e in es if e[2] == ROLE_TARGET)  # noqa: E731
    steps = torch.zeros(int(num_steps), dtype=torch.bool)
    for k in tgt(plan.action):
        steps[k] = True
    return Supervision(torch.tensor(tgt(plan.video), dtype=torch.long),
                       torch.tensor(tgt(plan.depth), dtype=torch.long), steps)


@dataclass
class Supervision:
    video_frames: torch.Tensor          # long indices of TARGET latents
    depth_frames: torch.Tensor
    action_steps: torch.Tensor          # bool [A]; all False in world mode


def write_back_clean(request: TemporalRequest, noisy_video, clean_video, noisy_depth, clean_depth,
                     noisy_actions, actions) -> Supervision:
    """Replace non-target positions by GT (in place) and return loss supports."""
    f_n = clean_video.shape[2]
    v_roles = torch.tensor(latent_roles(request, f_n))
    v_clean = (v_roles != ROLE_TARGET).nonzero().flatten()
    noisy_video[:, :, v_clean] = clean_video[:, :, v_clean]
    d_idx = torch.zeros(0, dtype=torch.long)
    if clean_depth is not None:
        d_roles = torch.tensor(latent_roles(request, clean_depth.shape[2], depth=True))
        d_clean = (d_roles != ROLE_TARGET).nonzero().flatten()
        if noisy_depth is not None:
            noisy_depth[:, :, d_clean] = clean_depth[:, :, d_clean]
        d_idx = (d_roles == ROLE_TARGET).nonzero().flatten()
    a_sup = torch.zeros(0, dtype=torch.bool)
    if actions is not None:
        a_roles = torch.tensor(action_step_roles(request, actions.shape[1]))
        a_sup = a_roles == ROLE_TARGET
        if noisy_actions is not None and bool((~a_sup).any()):
            keep = (~a_sup).to(actions.device)
            noisy_actions[:, keep] = actions[:, keep].to(noisy_actions.dtype)
    return Supervision((v_roles == ROLE_TARGET).nonzero().flatten(), d_idx, a_sup)


def encode_history_frames(encode: Callable[[torch.Tensor], torch.Tensor], pixels: torch.Tensor,
                          valid: torch.Tensor, dtype, valid_only: bool = False) -> torch.Tensor:
    """[B,Th,3,H,W] in [0,1] -> [B,C,Th,h,w]; each slot is its own 1-frame VAE clip.

    Single-frame encoding keeps one latent per real timestamp (no mixing of
    unequally spaced frames inside a causal VAE clip). Invalid slots are
    encoded from a neutral frame and must stay masked by ``valid``.
    ``valid_only`` encodes only the valid slots and leaves invalid latents at
    zero (one neutral frame is still encoded if none is valid, for the shape).
    """
    b, th = pixels.shape[:2]
    x = pixels * 2.0 - 1.0
    v = valid.to(x.device)
    x = torch.where(v[:, :, None, None, None], x, torch.zeros_like(x))
    flat = x.reshape(b * th, *x.shape[2:]).unsqueeze(2).to(dtype)          # [B*Th,3,1,H,W]
    keep = v.reshape(-1).nonzero().flatten() if valid_only else None
    with torch.no_grad():
        if keep is None:
            lat = encode(flat)                                              # [B*Th,C,1,h,w]
        elif keep.numel():
            part = encode(flat[keep])
            lat = part.new_zeros((b * th, *part.shape[1:]))
            lat[keep] = part
        else:
            part = encode(torch.zeros_like(flat[:1]))
            lat = part.new_zeros((b * th, *part.shape[1:]))
    if lat.shape[2] != 1:
        raise ValueError(f"VAE returned {lat.shape[2]} latents for a single frame")
    lat = lat[:, :, 0].reshape(b, th, *lat.shape[1:2], *lat.shape[3:])
    return lat.permute(0, 2, 1, 3, 4).contiguous()


def recent_clip_latent_frames(num_frames: int, stride: int = 4):
    """Frame index represented by each latent of a causal VAE clip (its last frame)."""
    if num_frames < 1 or (num_frames - 1) % stride:
        raise ValueError(f"recent history clip needs 1+{stride}m frames, got {num_frames}")
    return [0] + [stride * f for f in range(1, (num_frames - 1) // stride + 1)]


def recent_clip_latent_valid(frame_valid: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """[B,Tr] frame validity -> [B,1+m] latent validity (all frames of its group valid).

    Data must pad missing leading frames by replicating the earliest real frame
    (never zeros) and mark them invalid; a latent made only of real frames is
    valid even if the causal context saw replicated padding.
    """
    b, tr = frame_valid.shape
    frames = recent_clip_latent_frames(tr, stride)
    cols = [frame_valid[:, :1].all(dim=1)]
    for f in range(1, len(frames)):
        cols.append(frame_valid[:, 4 * f - 3:4 * f + 1].all(dim=1))
    return torch.stack(cols, dim=1)


def encode_recent_clip(encode: Callable[[torch.Tensor], torch.Tensor], pixels: torch.Tensor,
                       dtype) -> torch.Tensor:
    """[B,Tr,3,H,W] in [0,1], Tr = 1+4m -> [B,C,1+m,h,w] (one causal VAE clip)."""
    recent_clip_latent_frames(pixels.shape[1])
    x = (pixels * 2.0 - 1.0).permute(0, 2, 1, 3, 4).to(dtype)
    with torch.no_grad():
        return encode(x)


def masked_step_mean(err: torch.Tensor, step_mask: torch.Tensor,
                     dim_mask: Optional[torch.Tensor] = None,
                     validity: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample mean of err [B,A,U] over supervised steps/valid dims."""
    m = step_mask.to(err.device)[None, :, None].expand_as(err)
    if dim_mask is not None:
        m = m & dim_mask.to(err.device)[:, None, :]
    if validity is not None:
        m = m & validity.to(err.device)
    m = m.to(err.dtype)
    return (err * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1)


def frame_mse(pred: torch.Tensor, target: torch.Tensor, frames: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(frames, dtype=torch.long, device=pred.device)
    if idx.numel() == 0:
        return pred.flatten(1).sum(dim=1) * 0.0
    return torch.nn.functional.mse_loss(
        pred.index_select(2, idx), target.index_select(2, idx), reduction="none"
    ).mean(dim=(1, 2, 3, 4))


_MODE_TAGS = {MODE_JOINT: "joint", MODE_WORLD: "a2v", MODE_IDM: "idm"}


@torch.no_grad()
def block_loss_metrics(mode: str, grid: TimeGrid, sup: Supervision, *,
                       video_target, video_pred=None,
                       action_err=None, action_dim_mask=None, action_validity=None,
                       depth_pred=None, depth_target=None, depth_mask=None) -> dict:
    """Detached per-block batch-mean losses ``block_<mode>/<stream>_b<k>`` (logging only).

    Only supervised (TARGET) latents/steps count; depth block 0 is D0.
    Unweighted over samples, so values can differ from the weighted totals.
    """
    tag = _MODE_TAGS.get(mode, str(mode))
    vals = {}
    n_lat = video_target.shape[2]
    n_blocks = grid.num_blocks(n_lat)
    lat_block = [grid.latent_block(f) for f in range(n_lat)]
    if video_pred is not None:
        sel = sup.video_frames.tolist()
        for b in range(1, n_blocks):
            idx = [f for f in sel if lat_block[f] == b]
            if idx:
                vals[f"video_b{b}"] = frame_mse(video_pred, video_target, idx).float().mean()
    if action_err is not None and bool(sup.action_steps.any()):
        steps = torch.tensor([grid.action_step_block(k) for k in range(action_err.shape[1])])
        for b in range(1, n_blocks):
            m = sup.action_steps & (steps == b)
            if bool(m.any()):
                vals[f"action_b{b}"] = masked_step_mean(
                    action_err.float(), m, action_dim_mask, action_validity).mean()
    if depth_pred is not None:
        sel = sup.depth_frames.tolist()
        for b in range(0, n_blocks):
            idx = [f for f in sel if lat_block[f] == b]
            if not idx:
                continue
            i = torch.as_tensor(idx, dtype=torch.long, device=depth_pred.device)
            p, y = depth_pred.index_select(2, i).float(), depth_target.index_select(2, i).float()
            if depth_mask is None:
                vals[f"depth_b{b}"] = (p - y).square().mean()
            else:
                from thinker_wm.models.depth_conditioning import masked_depth_mse
                loss, ok = masked_depth_mse(p, y, depth_mask.index_select(2, i.to(depth_mask.device)))
                if bool(ok.any()):
                    vals[f"depth_b{b}"] = loss[ok].mean()
    if not vals:
        return {}
    keys = list(vals)
    dev = vals[keys[0]].device
    flat = torch.stack([vals[k].float().to(dev) for k in keys]).tolist()
    return {f"block_{tag}/{k}": float(v) for k, v in zip(keys, flat)}
