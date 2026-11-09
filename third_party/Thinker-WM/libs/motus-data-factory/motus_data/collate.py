"""Canonical collate function shared by all source-specific datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from motus_data.contracts import validate_sample


def _optional_stack(samples: List[Dict[str, Any]], key: str) -> Optional[torch.Tensor]:
    values = [sample.get(key) for sample in samples]
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"A homogeneous batch cannot mix present and missing {key}")
    return torch.stack(values)


def _optional_list(samples: List[Dict[str, Any]], key: str):
    values = [sample.get(key) for sample in samples]
    return values if all(value is not None for value in values) else None


def collate_samples(items: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    samples = [item for item in items if item is not None]
    if not samples:
        return None
    for sample in samples:
        validate_sample(sample)

    first_frames = torch.stack([sample["first_frame"] for sample in samples])
    video_frames = torch.stack([sample["video_frames"] for sample in samples])
    initial_states = _optional_stack(samples, "initial_state")
    action_sequences = _optional_stack(samples, "action_sequence")
    batch_size = len(samples)

    source_datasets = _optional_list(samples, "source_dataset")
    if source_datasets is not None and len(set(source_datasets)) != 1:
        raise ValueError(f"Mixed source datasets inside one batch are not allowed: {source_datasets}")

    action_weights = [sample.get("action_loss_weight") for sample in samples]
    video_weights = [sample.get("video_loss_weight") for sample in samples]
    sample_metadata = _optional_list(samples, "metadata")
    result = {
        "first_frame": first_frames,
        "video_frames": video_frames,
        "action_sequence": action_sequences,
        "vlm_inputs": None,
        "language_embedding": None,
        "first_frame_pils": _optional_list(samples, "first_frame_pil"),
        "t5_input_messages": _optional_list(samples, "t5_input_message"),
        "vlm_input_messages": _optional_list(samples, "vlm_input_message"),
        "robot_names": _optional_list(samples, "robot_name"),
        "source_datasets": source_datasets,
        "source_children": _optional_list(samples, "source_child"),
        "action_loss_weights": (
            torch.tensor(action_weights, dtype=torch.float32)
            if all(value is not None for value in action_weights) else None
        ),
        "video_loss_weights": (
            torch.tensor(video_weights, dtype=torch.float32)
            if all(value is not None for value in video_weights) else None
        ),
        "state_action_policies": _optional_list(samples, "state_action_policy"),
        "manifest_ids": _optional_list(samples, "manifest_id"),
        "manifest_tasks": _optional_list(samples, "manifest_task"),
        "episode_ids": _optional_list(samples, "episode_id"),
        "frame_indices": _optional_list(samples, "frame_indices"),
        "action_video_fps_ratios": _optional_list(
            samples, "action_video_fps_ratio"
        ),
        "initial_state_valid": torch.full((batch_size,), initial_states is not None, dtype=torch.bool),
        "action_sequence_valid": torch.full((batch_size,), action_sequences is not None, dtype=torch.bool),
    }
    if initial_states is not None:
        result["initial_state"] = initial_states
    if (
        sample_metadata is not None
        and any(
            metadata.get("synthetic_state_action") is True
            for metadata in sample_metadata
        )
    ):
        result["metadata"] = sample_metadata
    return result


# Compatibility name used by the current Motus training code.
collate_fn = collate_samples
