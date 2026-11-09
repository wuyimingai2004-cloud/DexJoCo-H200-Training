"""Deterministic signatures used to freeze and compare data-pipeline baselines."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


AGIBOT_BETA_BASELINE_V1 = "agibot_beta_direct_baseline_v1"


def _sample_start_frame(sample: Mapping[str, Any]) -> int:
    metadata_start = (sample.get("metadata") or {}).get("start_frame")
    if metadata_start is not None:
        return int(metadata_start)
    frame_indices = sample.get("frame_indices") or []
    if not frame_indices:
        raise ValueError("sample has neither metadata.start_frame nor frame_indices")
    return int(frame_indices[0])


def tensor_signature(value: torch.Tensor) -> dict[str, Any]:
    """Return a full-content fingerprint plus inexpensive numeric diagnostics."""

    if not torch.is_tensor(value):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    cpu = value.detach().cpu().contiguous()
    raw_bytes = cpu.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw_bytes))
    numeric = cpu.float()
    finite = bool(torch.isfinite(numeric).all())
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "all_finite": finite,
        "min": float(numeric.min()) if numeric.numel() else None,
        "max": float(numeric.max()) if numeric.numel() else None,
        "mean": float(numeric.mean()) if numeric.numel() else None,
        "std": float(numeric.std()) if numeric.numel() > 1 else 0.0,
        "sha256": digest.hexdigest(),
    }


def sample_signature(sample: Mapping[str, Any]) -> dict[str, Any]:
    tensor_keys = ("first_frame", "video_frames", "initial_state", "action_sequence")
    return {
        "episode_id": str(sample.get("episode_id")),
        "manifest_id": str(sample.get("manifest_id")),
        "source_dataset": str(sample.get("source_dataset")),
        "source_child": str(sample.get("source_child")),
        "split": str(sample.get("split")),
        "start_frame": _sample_start_frame(sample),
        "frame_indices": [int(value) for value in sample.get("frame_indices", [])],
        "action_video_fps_ratio": int(sample.get("action_video_fps_ratio", 1)),
        "robot_name": str(sample.get("robot_name")),
        "t5_input_message": str(sample.get("t5_input_message")),
        "vlm_input_message": str(sample.get("vlm_input_message")),
        "tensors": {
            key: tensor_signature(sample[key]) if sample.get(key) is not None else None
            for key in tensor_keys
        },
    }


def batch_signature(batch: Mapping[str, Any]) -> dict[str, Any]:
    tensor_keys = (
        "first_frame",
        "video_frames",
        "initial_state",
        "action_sequence",
        "action_loss_weights",
        "video_loss_weights",
    )
    list_keys = (
        "episode_ids",
        "manifest_ids",
        "frame_indices",
        "source_datasets",
        "source_children",
        "robot_names",
        "t5_input_messages",
        "vlm_input_messages",
        "state_action_policies",
        "action_video_fps_ratios",
    )
    return {
        "tensors": {
            key: tensor_signature(batch[key]) if batch.get(key) is not None else None
            for key in tensor_keys
        },
        "lists": {
            key: list(batch[key]) if batch.get(key) is not None else None
            for key in list_keys
        },
    }


def selection_records(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the minimal selections needed to replay baseline samples."""

    return [
        {
            "episode_id": str(sample["episode_id"]),
            "start_frame": _sample_start_frame(sample),
            "video_frame_indices": [int(value) for value in sample["frame_indices"]],
        }
        for sample in samples
    ]


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
