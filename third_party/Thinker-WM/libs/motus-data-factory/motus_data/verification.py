"""Training-boundary verification without importing the Motus model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


class TinyVideoTrainingStep(nn.Module):
    """Small trainable consumer exercising the same video batch boundary."""

    def __init__(self, channels: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        first = batch["first_frame"].float().mean(dim=(-2, -1))
        target = batch["video_frames"].float().mean(dim=(1, 3, 4))
        prediction = self.network(first)
        per_sample = (prediction - target).square().mean(dim=-1)
        weights = batch.get("video_loss_weights")
        if weights is not None:
            per_sample = per_sample * weights.float()
        return per_sample.mean()


def run_optimizer_smoke(
    pipeline,
    steps: int,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    batch_validator: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    target_device = torch.device(device)
    model = TinyVideoTrainingStep().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    before = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    losses = []
    source_counts: Dict[str, int] = {}
    child_counts: Dict[str, int] = {}
    episode_counts: Dict[str, int] = {}
    queue_wait_total = 0.0

    with pipeline:
        for _ in range(int(steps)):
            batch, stats = pipeline.next()
            if batch_validator is not None:
                batch_validator(batch)
            sources = batch.get("source_datasets") or ["unknown"]
            source = str(sources[0])
            source_counts[source] = source_counts.get(source, 0) + 1
            children = batch.get("source_children") or [source]
            child = str(children[0])
            child_counts[child] = child_counts.get(child, 0) + 1
            episode_ids = batch.get("episode_ids") or ["unknown"]
            episode_id = str(episode_ids[0])
            episode_counts[episode_id] = episode_counts.get(episode_id, 0) + 1
            queue_wait_total += float(stats.get("ready_queue_get_wait_seconds", 0.0))
            moved = move_to_device(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(moved)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite mock training loss: {loss.item()}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"invalid gradient for {name}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if progress_callback is not None:
                progress_callback({
                    "step": len(losses),
                    "loss": losses[-1],
                    "source": source,
                    "child": child,
                    "episode_id": episode_id,
                    "manifest_id": str((batch.get("manifest_ids") or ["unknown"])[0]),
                    "queue_wait_seconds": float(
                        stats.get("ready_queue_get_wait_seconds", 0.0)
                    ),
                    "queue_size_after_get": stats.get("ready_queue_size_after_get"),
                    "queue_was_empty": stats.get("ready_queue_was_empty"),
                })

    changed = any(
        not torch.equal(old, parameter.detach().cpu())
        for old, parameter in zip(before, model.cpu().parameters())
    )
    if not changed:
        raise RuntimeError("optimizer smoke did not update model parameters")
    return {
        "steps": int(steps),
        "device": str(target_device),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "parameters_updated": changed,
        "source_counts": source_counts,
        "child_counts": child_counts,
        "unique_episode_count": len(episode_counts),
        "episode_counts": episode_counts,
        "queue_wait_total_seconds": queue_wait_total,
    }
