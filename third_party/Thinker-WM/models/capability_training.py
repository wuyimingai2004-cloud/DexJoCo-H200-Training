"""Optional, evidence-backed objectives for Joint language/history and A2V.

All positive targets are recorded trajectories. A mismatched instruction or an
ablated history is a stop-gradient reference, never an imitation target.
"""
from __future__ import annotations

import math
import torch


CONFIG_FIELDS = {
    "capability_language_weight": float, "capability_history_weight": float,
    "capability_a2v_weight": float, "capability_margin": float,
    "capability_aux_interval": int, "capability_min_sigma": float,
}


def forward_world(forward, kwargs):
    """Keep the legacy five positional arguments for existing tracing hooks."""
    rest = dict(kwargs)
    positional = [rest.pop(k) for k in ("t", "vlm_input_message", "vlm_input_first_frame", "video", "action")]
    return forward(*positional, **rest)


def validate_config(cfg):
    for name in ("capability_language_weight", "capability_history_weight",
                 "capability_a2v_weight", "capability_margin"):
        value = float(getattr(cfg, name, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if int(getattr(cfg, "capability_aux_interval", 4)) < 1:
        raise ValueError("capability_aux_interval must be positive")
    sigma = float(getattr(cfg, "capability_min_sigma", .5))
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError("capability_min_sigma must lie in [0,1]")
    if enabled(cfg) and (getattr(cfg, "temporal_mode", "") != "block_causal"
                         or not getattr(cfg, "ar_packed_teacher_forcing", False)):
        raise ValueError("capability objectives require packed block-causal training")
    if getattr(cfg, "capability_history_weight", 0) and not getattr(cfg, "history_enable", False):
        raise ValueError("history objective requires history_enable")


def enabled(cfg):
    return any(float(getattr(cfg, f"capability_{k}_weight", 0)) > 0
               for k in ("language", "history", "a2v"))


def partners(records, kind, device):
    """Disjoint pairs only; same declared simulator+history context, no seed proxy."""
    n = len(records)
    indices, valid = list(range(n)), [False] * n
    for i, left in enumerate(records):
        if left is None or valid[i]:
            continue
        for j in range(i + 1, n):
            right = records[j]
            if right is None or valid[j] or left["context_id"] != right["context_id"]:
                continue
            if kind == "language":
                ok = (left["joint_valid"] and right["joint_valid"]
                      and left["instruction_id"] != right["instruction_id"])
            elif kind == "a2v":
                ok = (left["instruction_id"] == right["instruction_id"]
                      and left["branch_id"] != right["branch_id"])
            else:
                raise ValueError(kind)
            if ok:
                indices[i], indices[j], valid[i], valid[j] = j, i, True, True
                break
    return (torch.tensor(indices, device=device, dtype=torch.long),
            torch.tensor(valid, device=device, dtype=torch.bool))


def validate_pair_inputs(context, indices, valid, kind=None):
    """Metadata is necessary but not sufficient: reject mismatched observations."""
    if not bool(valid.any()):
        return
    tensors = {"current RGB": context["clean_full_latent"][:, :, :1],
               "state": context.get("state"), "action dimensions": context.get("action_dim_mask")}
    if context.get("clean_depth_latent") is not None:
        tensors["current depth label"] = context["clean_depth_latent"][:, :, :1]
    if kind == "a2v":
        for key in ("und_vlm_hidden", "und_real_lens"):
            if context.get(key) is not None:
                tensors[key] = context[key]
        messages = context["masked_vlm_messages"]
        for i, j in enumerate(indices.tolist()):
            if valid[i] and messages[i] != messages[j]:
                raise ValueError("A2V pairs must keep the literal instruction fixed")
    if kind == "language":
        messages = context["masked_vlm_messages"]
        for i, j in enumerate(indices.tolist()):
            if valid[i] and messages[i] == messages[j]:
                raise ValueError("different instruction IDs cannot contain identical instruction text")
    history = context.get("history")
    if history is not None:
        tensors.update({"history/" + k: v for k, v in history.items() if torch.is_tensor(v)})
    for name, value in tensors.items():
        if value is None:
            continue
        a, b = value[valid], value.index_select(0, indices)[valid]
        same = (torch.equal(a, b) if not a.is_floating_point() else
                torch.allclose(a, b, rtol=0, atol=1e-5))
        if not same:
            raise ValueError(f"capability pair has different {name}; use a verified shared context")


def masked_error(pred, target, mask):
    """Per-sample FP32 MSE, respecting all spatial/step/dimension masks."""
    mask = torch.broadcast_to(mask.bool(), pred.shape)
    pred = torch.where(mask, pred.float(), 0.)
    target = torch.where(mask, target.float(), 0.)
    err = (pred - target).square()
    dims = tuple(range(1, err.ndim))
    count = mask.sum(dim=dims)
    err = torch.where(mask, err, 0.).sum(dim=dims) / count.clamp_min(1)
    return err, count > 0


def ranking_loss(positive, reference, valid, margin):
    # Only improve the correct-condition prediction. Never deliberately train
    # worse predictions for empty history, a different goal, or a different action.
    value = (positive - reference.detach() + margin).clamp_min(0)
    return torch.where(valid, value, 0.).sum() / valid.sum().clamp_min(1)


def paired_effect_loss(pred, target, mask, indices, valid):
    """Match differences between two REAL outcomes under matched noise/timesteps."""
    pair_mask = mask & mask.index_select(0, indices)
    error, has_values = masked_error(
        pred - pred.index_select(0, indices), target - target.index_select(0, indices), pair_mask)
    # One copy per pair, even if padding repeated a row in a grouped batch.
    active = valid & has_values & (torch.arange(len(indices), device=indices.device) < indices)
    return torch.where(active, error, 0.).sum() / active.sum().clamp_min(1)


def share_pair_values(value, indices, valid):
    leaders = torch.minimum(torch.arange(len(indices), device=indices.device), indices)
    leaders = torch.where(valid, leaders, torch.arange(len(indices), device=indices.device))
    return value.index_select(0, leaders)


def choose_aux(cfg, step, joint, world=False):
    if step % cfg.capability_aux_interval:
        return None
    if world and cfg.capability_a2v_weight > 0:
        return "a2v"
    if not joint:
        return None
    choices = [k for k in ("language", "history") if getattr(cfg, f"capability_{k}_weight") > 0]
    return choices[(step // cfg.capability_aux_interval) % len(choices)] if choices else None


def reference_forward(forward, kwargs, rng, device):
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.set_rng_state(rng[0])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng[1], device)
        return forward_world(forward, kwargs)


def history_metrics(context, encoder):
    if encoder is None:
        return {}
    out = {"history_logit_bias": float(encoder.logit_bias.detach().float())}
    history = context.get("history")
    if history is not None:
        valid = history["valid"].bool()
        out["history_valid_latents_mean"] = float(valid.sum(1).float().mean())
        out["history_empty_latents_fraction"] = float((~valid.any(1)).float().mean())
    return out


def extra_losses(cfg, context, decision, plan, predictions, targets, forward_kwargs,
                 forward, aux_kind, aux_rng, pair_index, pair_valid, sigma, sigma_action,
                 action_weights, video_weights, depth_present):
    """Called inside the same autocast context as the main world-model forward."""
    from thinker_wm.models import temporal_training as TT
    vp, ap, dp = predictions
    vt, at, dt = targets
    zero = vt.new_zeros((), dtype=torch.float32)
    metrics = {f"capability_{k}_loss": 0. for k in ("language", "history", "a2v")}
    if decision is None or plan is None or decision.request.first_target > 1:
        return zero, metrics
    b = vt.shape[0]
    records = context.get("capability_records") or [None] * b
    metrics["capability_annotated_samples"] = sum(r is not None for r in records)
    grid = decision.request.grid
    n_steps, n_latents = grid.block_action_steps, grid.latents_per_block
    # Only the first future block: no target identity leaked by past clean actions.
    amask = torch.zeros_like(at, dtype=torch.bool)
    amask[:, :n_steps] = True
    if context.get("action_dim_mask") is not None:
        amask &= context["action_dim_mask"][:, None].bool()
    if context.get("action_validity_mask") is not None:
        amask &= context["action_validity_mask"].bool()
    if action_weights is not None:
        amask &= action_weights.reshape(b, 1, 1) > 0
    high_sigma = sigma.reshape(b) >= cfg.capability_min_sigma

    if decision.mode == TT.MODE_WORLD and cfg.capability_a2v_weight > 0:
        metrics["capability_a2v_paired_samples"] = int(pair_valid.sum())
        active = pair_valid & high_sigma & high_sigma.index_select(0, pair_index)
        # Commands really differ; an unchanged command is no counterfactual.
        commands = context["actions"][:, :n_steps]
        active &= (commands - commands.index_select(0, pair_index)).abs().flatten(1).amax(1) > 1e-6
        loss = zero
        references = {}
        if aux_kind == "a2v":
            kw = dict(forward_kwargs)
            kw["action"] = kw["action"].index_select(0, pair_index)
            rv, _, rd = reference_forward(forward, kw, aux_rng, at.device)
            if rv is not None:
                references["rgb"] = TT.unpack_targets(rv, plan.video, vt.shape[2], 2)
            if rd is not None:
                references["depth"] = TT.unpack_targets(rd, plan.depth, dt.shape[2], 2)
        for tag, pred, target in (("rgb", vp, vt), ("depth", dp, dt)):
            if pred is None or target is None:
                continue
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, :, 1:1+n_latents] = True  # D0 must NEVER depend on future commands.
            if video_weights is not None:
                mask &= video_weights.reshape(b, 1, 1, 1, 1) > 0
            if tag == "depth":
                depth_mask = context.get("depth_latent_valid_mask")
                if depth_mask is not None:
                    mask &= depth_mask.bool()
                if depth_present is not None:
                    mask &= depth_present.reshape(b, 1, 1, 1, 1).bool()
            value = paired_effect_loss(pred, target, mask, pair_index, active)
            metrics[f"capability_a2v_{tag}_loss"] = float(value.detach())
            loss = loss + value
            if tag in references:
                positive, valid_values = masked_error(pred, target, mask)
                reference, _ = masked_error(references[tag], target, mask)
                rank = ranking_loss(positive, reference, active & valid_values, cfg.capability_margin)
                metrics[f"capability_a2v_{tag}_condition_loss"] = float(rank.detach())
                metrics[f"capability_a2v_{tag}_error_gap"] = float(
                    torch.where(active & valid_values, reference - positive.detach(), 0.).sum()
                    / (active & valid_values).sum().clamp_min(1))
                loss = loss + rank
        metrics["capability_a2v_active_samples"] = int(active.sum())
        metrics["capability_a2v_loss"] = float(loss.detach())
        return cfg.capability_a2v_weight * loss, metrics

    if aux_kind is None or ap is None:
        return zero, metrics
    # Always execute the scheduled reference forward on every rank, even when
    # that rank has no valid pairs. Its outputs have no gradient or loss target.
    kw = dict(forward_kwargs)
    if aux_kind == "language":
        active = pair_valid.clone()
        commands = context["actions"][:, :n_steps]
        command_mask = amask[:, :n_steps] & amask.index_select(0, pair_index)[:, :n_steps]
        difference = torch.where(command_mask, commands - commands.index_select(0, pair_index), 0.)
        active &= difference.abs().flatten(1).amax(1) > 1e-6
        hidden = kw.get("und_precomputed")
        if hidden is None:
            raise ValueError("language objective requires precomputed VLM features")
        kw["und_precomputed"] = tuple(x.index_select(0, pair_index) for x in hidden)
        kw["vlm_input_message"] = [kw["vlm_input_message"][i] for i in pair_index.tolist()]
    else:
        active = torch.tensor([bool(r and r["history_required"] and r["joint_valid"])
                               for r in records], device=at.device)
        h = kw.get("history")
        if h is not None:
            active &= h["valid"].any(1)
            kw["history"] = {**h, "valid": torch.zeros_like(h["valid"])}
        else:
            active.zero_()
    active &= high_sigma & (sigma_action.reshape(b) >= cfg.capability_min_sigma)
    # Share stochastic layers' RNG state as well as input noise and timesteps.
    _, reference, _ = reference_forward(forward, kw, aux_rng, at.device)
    reference = TT.unpack_targets(reference, plan.action, at.shape[1], 1)
    pos_error, valid_steps = masked_error(ap, at, amask)
    ref_error, _ = masked_error(reference, at, amask)
    active &= valid_steps
    loss = ranking_loss(pos_error, ref_error, active, cfg.capability_margin)
    metrics[f"capability_{aux_kind}_active_samples"] = int(active.sum())
    metrics[f"capability_{aux_kind}_loss"] = float(loss.detach())
    metrics[f"capability_{aux_kind}_error_gap"] = float(
        torch.where(active, ref_error - pos_error.detach(), 0.).sum() / active.sum().clamp_min(1))
    return getattr(cfg, f"capability_{aux_kind}_weight") * loss, metrics
