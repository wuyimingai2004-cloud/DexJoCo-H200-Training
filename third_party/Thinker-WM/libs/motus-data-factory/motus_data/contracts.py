"""Public sample and batch contracts for model-independent data loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


@dataclass
class MotusSample:
    """Canonical boundary object returned by a source-specific dataset."""

    first_frame: torch.Tensor
    video_frames: torch.Tensor
    t5_text: str
    vlm_text: str
    robot_name: str
    source_group: str
    source_child: str
    sample_id: str
    episode_id: str
    frame_indices: List[int] = field(default_factory=list)
    action_video_fps_ratio: Optional[int] = None
    split: str = "train"
    initial_state: Optional[torch.Tensor] = None
    action_sequence: Optional[torch.Tensor] = None
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_training_dict(self) -> Dict[str, Any]:
        return {
            "first_frame": self.first_frame,
            "video_frames": self.video_frames,
            "initial_state": self.initial_state,
            "action_sequence": self.action_sequence,
            "t5_input_message": self.t5_text,
            "vlm_input_message": self.vlm_text,
            "robot_name": self.robot_name,
            "source_dataset": self.source_group,
            "source_child": self.source_child,
            "manifest_id": self.sample_id,
            "episode_id": self.episode_id,
            "frame_indices": list(self.frame_indices),
            "action_video_fps_ratio": self.action_video_fps_ratio,
            "split": self.split,
            "video_loss_weight": self.video_loss_weight,
            "action_loss_weight": self.action_loss_weight,
            "metadata": dict(self.metadata),
        }


class ContractError(ValueError):
    pass


MOTUS_MODEL_BATCH_V1 = "motus_model_batch_v1"
MOTUS_NATIVE_MULTIMODAL_BATCH_V1 = "motus_native_multimodal_batch_v1"

# This profile freezes the batch boundary consumed by the current Motus trainer.
# It is intentionally stricter than the generic contract below: Stage1 ignores
# action/state in the model, but the current trainer still moves these tensors.
MOTUS_MODEL_BATCH_V1_REQUIRED_KEYS = (
    "first_frame",
    "video_frames",
    "initial_state",
    "action_sequence",
    "vlm_inputs",
    "language_embedding",
    "first_frame_pils",
    "t5_input_messages",
    "vlm_input_messages",
    "robot_names",
    "source_datasets",
    "action_loss_weights",
    "video_loss_weights",
    "state_action_policies",
)

MOTUS_NATIVE_MULTIMODAL_BATCH_V1_REQUIRED_KEYS = (
    "batch_adapter_profile",
    "first_frame",
    "video_frames",
    "initial_state",
    "action_sequence",
    "vlm_inputs",
    "language_embedding",
    "first_frame_pils",
    "t5_input_messages",
    "vlm_input_messages",
    "robot_names",
    "source_datasets",
    "source_children",
    "action_loss_weights",
    "video_loss_weights",
    "state_action_policies",
    "state_dimension",
    "action_dimension",
    "normalization_modes",
)


def validate_sample(sample: Mapping[str, Any]) -> None:
    """Validate the model-facing shape contract without knowing the model."""

    for key in ("first_frame", "video_frames"):
        if key not in sample or not torch.is_tensor(sample[key]):
            raise ContractError(f"sample.{key} must be a torch.Tensor")
    first = sample["first_frame"]
    video = sample["video_frames"]
    if first.ndim != 3:
        raise ContractError(f"first_frame must be [C,H,W], got {tuple(first.shape)}")
    if video.ndim != 4:
        raise ContractError(f"video_frames must be [T,C,H,W], got {tuple(video.shape)}")
    if tuple(first.shape) != tuple(video.shape[1:]):
        raise ContractError(
            f"first/video spatial shape mismatch: {tuple(first.shape)} vs {tuple(video.shape[1:])}"
        )
    state = sample.get("initial_state")
    action = sample.get("action_sequence")
    if state is not None and (not torch.is_tensor(state) or state.ndim != 1):
        raise ContractError("initial_state must be None or [D]")
    if action is not None and (not torch.is_tensor(action) or action.ndim != 2):
        raise ContractError("action_sequence must be None or [T_action,D]")


def validate_batch(batch: Mapping[str, Any], expected_batch_size: Optional[int] = None) -> None:
    first = batch.get("first_frame")
    video = batch.get("video_frames")
    if not torch.is_tensor(first) or first.ndim != 4:
        raise ContractError("batch.first_frame must be [B,C,H,W]")
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ContractError("batch.video_frames must be [B,T,C,H,W]")
    batch_size = int(first.shape[0])
    if video.shape[0] != batch_size:
        raise ContractError("batch dimension differs between first_frame and video_frames")
    if tuple(first.shape[1:]) != tuple(video.shape[2:]):
        raise ContractError("batch first/video spatial shapes differ")
    if expected_batch_size is not None and batch_size != int(expected_batch_size):
        raise ContractError(f"expected batch size {expected_batch_size}, got {batch_size}")
    for key in ("action_sequence", "initial_state"):
        value = batch.get(key)
        if value is not None and (not torch.is_tensor(value) or value.shape[0] != batch_size):
            raise ContractError(f"batch.{key} must be None or have leading dimension B")


def _require_batch_list(
    batch: Mapping[str, Any], key: str, batch_size: int
) -> Sequence[Any]:
    value = batch.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != batch_size:
        raise ContractError(f"batch.{key} must be a list/tuple with length B={batch_size}")
    return value


def _require_finite_tensor(
    batch: Mapping[str, Any], key: str, batch_size: int, ndim: int
) -> torch.Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value) or value.ndim != ndim or value.shape[0] != batch_size:
        raise ContractError(f"batch.{key} must be a rank-{ndim} tensor with leading dimension B")
    if not torch.isfinite(value).all():
        raise ContractError(f"batch.{key} contains non-finite values")
    return value


def validate_motus_model_batch_v1(
    batch: Mapping[str, Any],
    *,
    expected_batch_size: Optional[int] = None,
    num_video_frames: Optional[int] = None,
    video_action_freq_ratio: Optional[int] = None,
) -> None:
    """Validate the exact batch boundary used by the current Motus trainer.

    `vlm_inputs` and `language_embedding` are required keys but may be ``None``
    before the model-side text preprocessors populate them.
    """

    missing = [key for key in MOTUS_MODEL_BATCH_V1_REQUIRED_KEYS if key not in batch]
    if missing:
        raise ContractError(f"{MOTUS_MODEL_BATCH_V1} missing keys: {missing}")

    validate_batch(batch, expected_batch_size=expected_batch_size)
    first = batch["first_frame"]
    video = batch["video_frames"]
    batch_size = int(first.shape[0])
    if not first.is_floating_point() or not video.is_floating_point():
        raise ContractError("first_frame and video_frames must be floating-point tensors")
    if not torch.isfinite(first).all() or not torch.isfinite(video).all():
        raise ContractError("first_frame or video_frames contains non-finite values")
    if first.numel() and (float(first.min()) < -1e-6 or float(first.max()) > 1.0 + 1e-6):
        raise ContractError("first_frame values must be in [0, 1]")
    if video.numel() and (float(video.min()) < -1e-6 or float(video.max()) > 1.0 + 1e-6):
        raise ContractError("video_frames values must be in [0, 1]")
    if num_video_frames is not None and int(video.shape[1]) != int(num_video_frames):
        raise ContractError(
            f"expected {num_video_frames} future video frames, got {int(video.shape[1])}"
        )

    state = _require_finite_tensor(batch, "initial_state", batch_size, ndim=2)
    action = _require_finite_tensor(batch, "action_sequence", batch_size, ndim=3)
    if state.shape[-1] != action.shape[-1]:
        raise ContractError(
            f"state/action dim mismatch: {int(state.shape[-1])} vs {int(action.shape[-1])}"
        )
    batch_ratios = batch.get("action_video_fps_ratios")
    if batch_ratios is not None:
        if not isinstance(batch_ratios, (list, tuple)) or len(batch_ratios) != batch_size:
            raise ContractError(
                "batch.action_video_fps_ratios must be None or have length B"
            )
        normalized_ratios = [int(value) for value in batch_ratios]
        if any(value <= 0 for value in normalized_ratios):
            raise ContractError("batch.action_video_fps_ratios must be positive")
        if len(set(normalized_ratios)) != 1:
            raise ContractError(
                "one homogeneous batch cannot mix action/video FPS ratios"
            )
        expected_actions = int(video.shape[1]) * normalized_ratios[0]
        if int(action.shape[1]) != expected_actions:
            raise ContractError(
                f"expected action length {expected_actions}, got {int(action.shape[1])}"
            )
    elif video_action_freq_ratio is not None:
        expected_actions = int(video.shape[1]) * int(video_action_freq_ratio)
        if int(action.shape[1]) != expected_actions:
            raise ContractError(
                f"expected action length {expected_actions}, got {int(action.shape[1])}"
            )

    for key in (
        "first_frame_pils",
        "t5_input_messages",
        "vlm_input_messages",
        "robot_names",
        "source_datasets",
        "state_action_policies",
    ):
        _require_batch_list(batch, key, batch_size)
    sources = batch["source_datasets"]
    if len(set(str(source) for source in sources)) != 1:
        raise ContractError(f"{MOTUS_MODEL_BATCH_V1} batches must be source-homogeneous")

    for key in ("action_loss_weights", "video_loss_weights"):
        value = _require_finite_tensor(batch, key, batch_size, ndim=1)
        if value.shape != (batch_size,):
            raise ContractError(f"batch.{key} must have shape [B]")


def validate_motus_native_multimodal_batch_v1(
    batch: Mapping[str, Any],
    *,
    expected_batch_size: Optional[int] = None,
    num_video_frames: Optional[int] = None,
    video_action_freq_ratio: Optional[int] = None,
    expected_state_dimension: Optional[int] = None,
    expected_action_dimension: Optional[int] = None,
) -> None:
    """Validate native multimodal batches with independent state/action dims."""

    missing = [
        key
        for key in MOTUS_NATIVE_MULTIMODAL_BATCH_V1_REQUIRED_KEYS
        if key not in batch
    ]
    if missing:
        raise ContractError(
            f"{MOTUS_NATIVE_MULTIMODAL_BATCH_V1} missing keys: {missing}"
        )
    if batch["batch_adapter_profile"] != MOTUS_NATIVE_MULTIMODAL_BATCH_V1:
        raise ContractError("native multimodal batch profile name changed")

    validate_batch(batch, expected_batch_size=expected_batch_size)
    first = batch["first_frame"]
    video = batch["video_frames"]
    batch_size = int(first.shape[0])
    for name, value in (("first_frame", first), ("video_frames", video)):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ContractError(f"batch.{name} must be finite floating point")
        if value.numel() and (
            float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6
        ):
            raise ContractError(f"batch.{name} values must be in [0,1]")
    if (
        num_video_frames is not None
        and int(video.shape[1]) != int(num_video_frames)
    ):
        raise ContractError(
            f"expected {num_video_frames} future video frames, "
            f"got {int(video.shape[1])}"
        )

    state = _require_finite_tensor(
        batch, "initial_state", batch_size, ndim=2
    )
    action = _require_finite_tensor(
        batch, "action_sequence", batch_size, ndim=3
    )
    state_dimension = batch["state_dimension"]
    action_dimension = batch["action_dimension"]
    if isinstance(state_dimension, bool) or not isinstance(state_dimension, int):
        raise ContractError("batch.state_dimension must be an integer")
    if isinstance(action_dimension, bool) or not isinstance(action_dimension, int):
        raise ContractError("batch.action_dimension must be an integer")
    if state_dimension <= 0 or action_dimension <= 0:
        raise ContractError("native state/action dimensions must be positive")
    if int(state.shape[-1]) != state_dimension:
        raise ContractError("initial_state differs from state_dimension")
    if int(action.shape[-1]) != action_dimension:
        raise ContractError("action_sequence differs from action_dimension")
    if (
        expected_state_dimension is not None
        and state_dimension != int(expected_state_dimension)
    ):
        raise ContractError(
            f"expected state dimension {expected_state_dimension}, "
            f"got {state_dimension}"
        )
    if (
        expected_action_dimension is not None
        and action_dimension != int(expected_action_dimension)
    ):
        raise ContractError(
            f"expected action dimension {expected_action_dimension}, "
            f"got {action_dimension}"
        )

    batch_ratios = batch.get("action_video_fps_ratios")
    if batch_ratios is not None:
        if (
            not isinstance(batch_ratios, (list, tuple))
            or len(batch_ratios) != batch_size
        ):
            raise ContractError(
                "batch.action_video_fps_ratios must be None or length B"
            )
        ratios = [int(value) for value in batch_ratios]
        if any(value <= 0 for value in ratios) or len(set(ratios)) != 1:
            raise ContractError(
                "one native batch requires one positive action/video ratio"
            )
        expected_actions = int(video.shape[1]) * ratios[0]
        if int(action.shape[1]) != expected_actions:
            raise ContractError(
                f"expected action length {expected_actions}, "
                f"got {int(action.shape[1])}"
            )
    elif video_action_freq_ratio is not None:
        expected_actions = int(video.shape[1]) * int(video_action_freq_ratio)
        if int(action.shape[1]) != expected_actions:
            raise ContractError(
                f"expected action length {expected_actions}, "
                f"got {int(action.shape[1])}"
            )

    for key in (
        "first_frame_pils",
        "t5_input_messages",
        "vlm_input_messages",
        "robot_names",
        "source_datasets",
        "source_children",
        "state_action_policies",
    ):
        _require_batch_list(batch, key, batch_size)
    if len(set(str(value) for value in batch["source_datasets"])) != 1:
        raise ContractError("native multimodal batches must be source-homogeneous")
    if set(str(value) for value in batch["state_action_policies"]) != {
        "real_native_dimensions"
    }:
        raise ContractError("native multimodal state/action policy changed")
    for key in ("action_loss_weights", "video_loss_weights"):
        value = _require_finite_tensor(batch, key, batch_size, ndim=1)
        if tuple(value.shape) != (batch_size,):
            raise ContractError(f"batch.{key} must have shape [B]")
    modes = batch["normalization_modes"]
    if not isinstance(modes, Mapping) or set(modes) != {"state", "action"}:
        raise ContractError(
            "batch.normalization_modes must contain state and action"
        )
    if any(str(value) not in {"raw", "zscore"} for value in modes.values()):
        raise ContractError("unsupported native normalization mode")
