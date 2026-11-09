"""Replaceable adapters from indexed Episode data to model-specific batches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from motus_data.collate import collate_samples
from motus_data.contracts import (
    MOTUS_MODEL_BATCH_V1,
    MOTUS_NATIVE_MULTIMODAL_BATCH_V1,
    validate_motus_model_batch_v1,
    validate_motus_native_multimodal_batch_v1,
)
from motus_data.datasets.agibot_common import VLM_PROMPT, tensor_to_pil
from motus_data.reader_registry import resolve_text_composer
from motus_data.repository import EpisodeSelectionData
from motus_data.sample_pool import IndexedEpisodeBatch, SamplePoolDataset


class BatchAdapterError(ValueError):
    """Raised when indexed data cannot satisfy a requested model profile."""


class IndexedBatchAdapter(Protocol):
    """Interface implemented by model-specific indexed-batch adapters."""

    profile_name: str

    def adapt(self, batch: IndexedEpisodeBatch) -> Mapping[str, Any]:
        """Convert one source-independent batch into a model batch."""


BatchAdapterFactory = Callable[..., IndexedBatchAdapter]
_BATCH_ADAPTER_FACTORIES: dict[str, BatchAdapterFactory] = {}


def register_indexed_batch_adapter(
    profile_name: str,
    factory: BatchAdapterFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a model adapter without changing Dataset or Repository code."""

    name = str(profile_name).strip()
    if not name:
        raise ValueError("profile_name must be non-empty")
    if not callable(factory):
        raise TypeError("batch adapter factory must be callable")
    if name in _BATCH_ADAPTER_FACTORIES and not replace:
        raise KeyError(f"batch adapter profile is already registered: {name}")
    _BATCH_ADAPTER_FACTORIES[name] = factory


def build_indexed_batch_adapter(
    profile_name: str, **options: Any
) -> IndexedBatchAdapter:
    """Build a registered adapter for the model-owned output profile."""

    name = str(profile_name)
    try:
        factory = _BATCH_ADAPTER_FACTORIES[name]
    except KeyError as error:
        raise KeyError(
            f"unknown indexed batch adapter profile={name!r}; "
            f"available={sorted(_BATCH_ADAPTER_FACTORIES)}"
        ) from error
    adapter = factory(**options)
    if str(adapter.profile_name) != name:
        raise ValueError(
            f"adapter profile mismatch: registry={name!r} "
            f"adapter={adapter.profile_name!r}"
        )
    return adapter


def indexed_batch_adapter_profiles() -> tuple[str, ...]:
    return tuple(sorted(_BATCH_ADAPTER_FACTORIES))


@dataclass(frozen=True)
class IndexedBatchCollator:
    """DataLoader collator that keeps raw and model-specific layers separate."""

    adapter: IndexedBatchAdapter

    def __call__(
        self, items: Sequence[EpisodeSelectionData]
    ) -> Mapping[str, Any]:
        indexed = SamplePoolDataset.collate_fn(items)
        return self.adapter.adapt(indexed)


def _selection_id(sample: EpisodeSelectionData) -> str:
    payload = {
        "episode_id": sample.info.episode_id,
        "video_frame_indices": list(sample.video_frame_indices),
        "state_indices": list(sample.state_indices),
        "action_indices": list(sample.action_indices),
        "text_indices": list(sample.text_indices),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sample_pool_selection:" + hashlib.sha256(encoded).hexdigest()[:24]


def _compose_t5_text(sample: EpisodeSelectionData) -> str:
    """拼 t5 输入文本：注册了 composer 的源（egoverse/egoscale——catalog 文本
    是裸字段，按源自有方式拼装、片段所有字段都拼入）走 composer；其余源维持
    catalog 文本顺序直接 join（agibot 预格式化文本、libero 单条指令等）。

    选中 item 顺序由采样侧保证：episode context 在前、窗口重叠片段按时间序。
    """
    composer = resolve_text_composer(sample.info.source_dataset)
    if composer is not None:
        return str(composer(sample.text)).strip()
    return "".join(item.text for item in sample.text).strip()


@dataclass(frozen=True)
class MotusModelBatchV1Adapter:
    """Current Motus batch profile, implemented outside the indexed Dataset."""

    num_video_frames: int = 48
    action_video_fps_ratio: int = 1
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    state_action_policy: str = "real"
    allow_missing_state_action: bool = False
    placeholder_state_action_dim: int | None = None
    profile_name: str = MOTUS_MODEL_BATCH_V1

    def __post_init__(self) -> None:
        if int(self.num_video_frames) <= 0:
            raise ValueError("num_video_frames must be positive")
        if int(self.action_video_fps_ratio) <= 0:
            raise ValueError("action_video_fps_ratio must be positive")
        for name, value in (
            ("video_loss_weight", self.video_loss_weight),
            ("action_loss_weight", self.action_loss_weight),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")
        if not str(self.state_action_policy).strip():
            raise ValueError("state_action_policy must be non-empty")
        if self.allow_missing_state_action:
            if (
                self.placeholder_state_action_dim is None
                or int(self.placeholder_state_action_dim) <= 0
            ):
                raise ValueError(
                    "allow_missing_state_action requires a positive "
                    "placeholder_state_action_dim"
                )
            if float(self.action_loss_weight) != 0.0:
                raise ValueError(
                    "missing state/action placeholders require "
                    "action_loss_weight=0"
                )

    def _adapt_sample(self, sample: EpisodeSelectionData) -> dict[str, Any]:
        expected_video = int(self.num_video_frames) + 1
        expected_action = (
            int(self.num_video_frames) * int(self.action_video_fps_ratio)
        )
        if len(sample.video_frame_indices) != expected_video:
            raise BatchAdapterError(
                f"{self.profile_name} requires {expected_video} selected video "
                f"frames, got {len(sample.video_frame_indices)}"
            )
        missing_state_action = (
            sample.info.state_count == 0
            and sample.info.action_count == 0
        )
        if missing_state_action:
            if not self.allow_missing_state_action:
                raise BatchAdapterError(
                    f"{self.profile_name} does not allow missing state/action"
                )
            if sample.state_indices or sample.action_indices:
                raise BatchAdapterError(
                    "absent state/action require empty model-owned indices"
                )
            if tuple(sample.state.shape) != (0, 0):
                raise BatchAdapterError(
                    f"absent state must be [0,0], got {tuple(sample.state.shape)}"
                )
            if tuple(sample.action.shape) != (0, 0):
                raise BatchAdapterError(
                    f"absent action must be [0,0], got {tuple(sample.action.shape)}"
                )
            dimension = int(self.placeholder_state_action_dim)
            initial_state = torch.zeros(dimension, dtype=torch.float32)
            action_sequence = torch.zeros(
                (expected_action, dimension),
                dtype=torch.float32,
            )
        else:
            if sample.info.state_count == 0 or sample.info.action_count == 0:
                raise BatchAdapterError(
                    "state/action must either both exist or both be absent"
                )
            if len(sample.state_indices) != 1:
                raise BatchAdapterError(
                    f"{self.profile_name} requires exactly one explicit state "
                    f"index, got {len(sample.state_indices)}"
                )
            if len(sample.action_indices) != expected_action:
                raise BatchAdapterError(
                    f"{self.profile_name} requires {expected_action} explicit "
                    f"action indices, got {len(sample.action_indices)}"
                )
            initial_state = sample.state[0]
            action_sequence = sample.action
        if not sample.text_indices:
            raise BatchAdapterError(
                f"{self.profile_name} requires at least one explicit text index"
            )
        if (
            not torch.is_tensor(sample.video)
            or sample.video.ndim != 4
            or int(sample.video.shape[0]) != expected_video
            or not sample.video.is_floating_point()
        ):
            raise BatchAdapterError(
                f"{self.profile_name} requires float video [T,C,H,W], got "
                f"{type(sample.video).__name__} "
                f"{tuple(sample.video.shape) if torch.is_tensor(sample.video) else ''}"
            )
        if int(sample.video.shape[1]) != 3:
            raise BatchAdapterError(
                f"{self.profile_name} requires RGB video, got C={sample.video.shape[1]}"
            )
        if (
            not missing_state_action
            and (
                tuple(sample.state.shape[:1]) != (1,)
                or sample.state.ndim != 2
            )
        ):
            raise BatchAdapterError(
                f"{self.profile_name} requires state [1,D], got "
                f"{tuple(sample.state.shape)}"
            )
        if (
            not missing_state_action
            and (
                sample.action.ndim != 2
                or int(sample.action.shape[0]) != expected_action
            )
        ):
            raise BatchAdapterError(
                f"{self.profile_name} requires action [{expected_action},D], got "
                f"{tuple(sample.action.shape)}"
            )
        if (
            not missing_state_action
            and int(sample.state.shape[-1]) != int(sample.action.shape[-1])
        ):
            raise BatchAdapterError(
                f"state/action dim mismatch: {sample.state.shape[-1]} vs "
                f"{sample.action.shape[-1]}"
            )
        if len(sample.text) != len(sample.text_indices):
            raise BatchAdapterError("decoded text count differs from explicit text indices")

        text = _compose_t5_text(sample)
        if not text:
            raise BatchAdapterError(
                f"{self.profile_name} requires non-empty selected text"
            )
        first_frame = sample.video[0]
        return {
            "first_frame": first_frame,
            "video_frames": sample.video[1:],
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "first_frame_pil": tensor_to_pil(first_frame),
            "t5_input_message": text,
            "vlm_input_message": f"{VLM_PROMPT}{text}".strip(),
            "robot_name": sample.info.robot_name,
            "source_dataset": sample.info.source_dataset,
            "source_child": sample.info.source_child,
            "action_loss_weight": float(self.action_loss_weight),
            "video_loss_weight": float(self.video_loss_weight),
            "state_action_policy": str(self.state_action_policy),
            "manifest_id": _selection_id(sample),
            "episode_id": sample.info.episode_id,
            "frame_indices": list(sample.video_frame_indices),
            "action_video_fps_ratio": int(self.action_video_fps_ratio),
            "split": sample.info.split,
            "metadata": {
                "batch_adapter_profile": self.profile_name,
                "video_frame_indices": list(sample.video_frame_indices),
                "state_indices": list(sample.state_indices),
                "action_indices": list(sample.action_indices),
                "text_indices": list(sample.text_indices),
                **(
                    {"synthetic_state_action": True}
                    if missing_state_action
                    else {}
                ),
            },
        }

    def adapt(self, batch: IndexedEpisodeBatch) -> Mapping[str, Any]:
        if not isinstance(batch, IndexedEpisodeBatch):
            raise TypeError(
                f"{self.profile_name} expects IndexedEpisodeBatch, got "
                f"{type(batch).__name__}"
            )
        samples = [self._adapt_sample(sample) for sample in batch.samples]
        result = collate_samples(samples)
        if result is None:
            raise BatchAdapterError("cannot adapt an empty indexed batch")
        validate_motus_model_batch_v1(
            result,
            expected_batch_size=batch.batch_size,
            num_video_frames=int(self.num_video_frames),
            video_action_freq_ratio=int(self.action_video_fps_ratio),
        )
        return result


register_indexed_batch_adapter(
    MOTUS_MODEL_BATCH_V1, MotusModelBatchV1Adapter
)


@dataclass(frozen=True)
class NativeMultimodalBatchV1Adapter:
    """Default native batch profile with independent state/action dims."""

    num_video_frames: int = 48
    action_video_fps_ratio: int = 1
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    profile_name: str = MOTUS_NATIVE_MULTIMODAL_BATCH_V1

    def __post_init__(self) -> None:
        if int(self.num_video_frames) <= 0:
            raise ValueError("num_video_frames must be positive")
        if int(self.action_video_fps_ratio) <= 0:
            raise ValueError("action_video_fps_ratio must be positive")
        for name, value in (
            ("video_loss_weight", self.video_loss_weight),
            ("action_loss_weight", self.action_loss_weight),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")

    @staticmethod
    def _normalization_modes(
        sample: EpisodeSelectionData,
    ) -> dict[str, str]:
        result = {}
        for modality in ("state", "action"):
            value = str(
                sample.info.modality_schemas[modality].metadata.get(
                    "normalization", "raw"
                )
            ).lower()
            if value == "none":
                value = "raw"
            if value not in {"raw", "zscore"}:
                raise BatchAdapterError(
                    f"unsupported {modality} normalization={value!r}"
                )
            result[modality] = value
        return result

    def _adapt_sample(self, sample: EpisodeSelectionData) -> dict[str, Any]:
        expected_video = int(self.num_video_frames) + 1
        expected_action = (
            int(self.num_video_frames) * int(self.action_video_fps_ratio)
        )
        if len(sample.video_frame_indices) != expected_video:
            raise BatchAdapterError(
                f"{self.profile_name} requires {expected_video} selected "
                f"video frames, got {len(sample.video_frame_indices)}"
            )
        if sample.info.state_count <= 0 or sample.info.action_count <= 0:
            raise BatchAdapterError(
                f"{self.profile_name} requires real state and action"
            )
        if len(sample.state_indices) != 1:
            raise BatchAdapterError(
                f"{self.profile_name} requires one explicit state index, "
                f"got {len(sample.state_indices)}"
            )
        if len(sample.action_indices) != expected_action:
            raise BatchAdapterError(
                f"{self.profile_name} requires {expected_action} explicit "
                f"action indices, got {len(sample.action_indices)}"
            )
        if (
            not torch.is_tensor(sample.video)
            or sample.video.ndim != 4
            or tuple(sample.video.shape[:2]) != (expected_video, 3)
            or not sample.video.is_floating_point()
        ):
            raise BatchAdapterError(
                f"{self.profile_name} requires float RGB video [T,3,H,W]"
            )
        if sample.state.ndim != 2 or int(sample.state.shape[0]) != 1:
            raise BatchAdapterError(
                f"{self.profile_name} requires state [1,D_state], got "
                f"{tuple(sample.state.shape)}"
            )
        if (
            sample.action.ndim != 2
            or int(sample.action.shape[0]) != expected_action
        ):
            raise BatchAdapterError(
                f"{self.profile_name} requires action "
                f"[{expected_action},D_action], got "
                f"{tuple(sample.action.shape)}"
            )
        if int(sample.state.shape[1]) <= 0 or int(sample.action.shape[1]) <= 0:
            raise BatchAdapterError(
                "native state/action dimensions must be positive"
            )
        if (
            not sample.text_indices
            or len(sample.text) != len(sample.text_indices)
        ):
            raise BatchAdapterError(
                f"{self.profile_name} requires explicitly selected text"
            )
        text = _compose_t5_text(sample)
        if not text:
            raise BatchAdapterError(
                f"{self.profile_name} requires non-empty selected text"
            )
        first_frame = sample.video[0]
        normalization_modes = self._normalization_modes(sample)
        return {
            "first_frame": first_frame,
            "video_frames": sample.video[1:],
            "initial_state": sample.state[0],
            "action_sequence": sample.action,
            "first_frame_pil": tensor_to_pil(first_frame),
            "t5_input_message": text,
            "vlm_input_message": f"{VLM_PROMPT}{text}".strip(),
            "robot_name": sample.info.robot_name,
            "source_dataset": sample.info.source_dataset,
            "source_child": sample.info.source_child,
            "action_loss_weight": float(self.action_loss_weight),
            "video_loss_weight": float(self.video_loss_weight),
            "state_action_policy": "real_native_dimensions",
            "manifest_id": _selection_id(sample),
            "episode_id": sample.info.episode_id,
            "frame_indices": list(sample.video_frame_indices),
            "action_video_fps_ratio": int(self.action_video_fps_ratio),
            "split": sample.info.split,
            "metadata": {
                "batch_adapter_profile": self.profile_name,
                "video_frame_indices": list(sample.video_frame_indices),
                "state_indices": list(sample.state_indices),
                "action_indices": list(sample.action_indices),
                "text_indices": list(sample.text_indices),
                "state_dimension": int(sample.state.shape[1]),
                "action_dimension": int(sample.action.shape[1]),
                "normalization_modes": normalization_modes,
            },
        }

    def adapt(self, batch: IndexedEpisodeBatch) -> Mapping[str, Any]:
        if not isinstance(batch, IndexedEpisodeBatch):
            raise TypeError(
                f"{self.profile_name} expects IndexedEpisodeBatch, got "
                f"{type(batch).__name__}"
            )
        samples = [self._adapt_sample(sample) for sample in batch.samples]
        state_dimensions = {
            int(sample["metadata"]["state_dimension"]) for sample in samples
        }
        action_dimensions = {
            int(sample["metadata"]["action_dimension"]) for sample in samples
        }
        normalization_modes = {
            (
                sample["metadata"]["normalization_modes"]["state"],
                sample["metadata"]["normalization_modes"]["action"],
            )
            for sample in samples
        }
        if (
            len(state_dimensions) != 1
            or len(action_dimensions) != 1
            or len(normalization_modes) != 1
        ):
            raise BatchAdapterError(
                "one native batch requires homogeneous dimensions and "
                "normalization modes"
            )
        result = collate_samples(samples)
        if result is None:
            raise BatchAdapterError("cannot adapt an empty indexed batch")
        state_mode, action_mode = next(iter(normalization_modes))
        result.update(
            {
                "batch_adapter_profile": self.profile_name,
                "state_dimension": next(iter(state_dimensions)),
                "action_dimension": next(iter(action_dimensions)),
                "normalization_modes": {
                    "state": state_mode,
                    "action": action_mode,
                },
                "metadata": [sample["metadata"] for sample in samples],
            }
        )
        validate_motus_native_multimodal_batch_v1(
            result,
            expected_batch_size=batch.batch_size,
            num_video_frames=int(self.num_video_frames),
            video_action_freq_ratio=int(self.action_video_fps_ratio),
        )
        return result


register_indexed_batch_adapter(
    MOTUS_NATIVE_MULTIMODAL_BATCH_V1,
    NativeMultimodalBatchV1Adapter,
)
