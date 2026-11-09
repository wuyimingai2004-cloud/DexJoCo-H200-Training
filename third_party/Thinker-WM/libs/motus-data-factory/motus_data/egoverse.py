"""EgoVerse Zarr v3 source profiles and indexed Episode reader."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from motus_data.reader_registry import (
    register_episode_source_reader,
    register_text_composer,
)
from motus_data.repository import (
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
)


ZarrGroupOpener = Callable[[str], Any]


@dataclass(frozen=True)
class EgoVerseProfile:
    """Canonical four-modality view of one EgoVerse embodiment profile."""

    name: str
    robot_name: str
    raw_embodiments: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    state_dim: int
    action_dim: int
    state_semantics: str
    action_semantics: str
    action_source_signal: str
    action_is_control_command: bool
    video_key: str = "images.front_1"

    @property
    def source_child(self) -> str:
        return f"egoverse_{self.name}"


_HUMAN_BIMANUAL_KEYS = (
    "left.obs_ee_pose",
    "right.obs_ee_pose",
)
_EVA_BIMANUAL_STATE_KEYS = (
    "left.obs_ee_pose",
    "left.obs_gripper",
    "right.obs_ee_pose",
    "right.obs_gripper",
)
_EVA_BIMANUAL_ACTION_KEYS = (
    "left.cmd_ee_pose",
    "left.cmd_gripper",
    "right.cmd_ee_pose",
    "right.cmd_gripper",
)


EGOVERSE_PROFILES: dict[str, EgoVerseProfile] = {
    "aria_bimanual": EgoVerseProfile(
        name="aria_bimanual",
        robot_name="egoverse-aria-bimanual",
        raw_embodiments=("aria_bimanual",),
        state_keys=_HUMAN_BIMANUAL_KEYS,
        action_keys=_HUMAN_BIMANUAL_KEYS,
        state_dim=14,
        action_dim=14,
        state_semantics="raw_observed_ee_pose",
        action_semantics="raw_observed_ee_pose",
        action_source_signal="observed_ee_pose_timeline",
        action_is_control_command=False,
    ),
    "aria_left_arm": EgoVerseProfile(
        name="aria_left_arm",
        robot_name="egoverse-aria-left-arm",
        raw_embodiments=("aria_left_arm",),
        state_keys=("left.obs_ee_pose",),
        action_keys=("left.obs_ee_pose",),
        state_dim=7,
        action_dim=7,
        state_semantics="raw_observed_ee_pose",
        action_semantics="raw_observed_ee_pose",
        action_source_signal="observed_ee_pose_timeline",
        action_is_control_command=False,
    ),
    "aria_right_arm": EgoVerseProfile(
        name="aria_right_arm",
        robot_name="egoverse-aria-right-arm",
        raw_embodiments=("aria_right_arm",),
        state_keys=("right.obs_ee_pose",),
        action_keys=("right.obs_ee_pose",),
        state_dim=7,
        action_dim=7,
        state_semantics="raw_observed_ee_pose",
        action_semantics="raw_observed_ee_pose",
        action_source_signal="observed_ee_pose_timeline",
        action_is_control_command=False,
    ),
    "eva_bimanual": EgoVerseProfile(
        name="eva_bimanual",
        robot_name="egoverse-eva-bimanual",
        raw_embodiments=("eva_bimanual",),
        state_keys=_EVA_BIMANUAL_STATE_KEYS,
        action_keys=_EVA_BIMANUAL_ACTION_KEYS,
        state_dim=16,
        action_dim=16,
        state_semantics="raw_observed_ee_pose_gripper",
        action_semantics="raw_commanded_ee_pose_gripper",
        action_source_signal="commanded_ee_pose_gripper_timeline",
        action_is_control_command=True,
    ),
    "eva_left_arm": EgoVerseProfile(
        name="eva_left_arm",
        robot_name="egoverse-eva-left-arm",
        raw_embodiments=("eva_left_arm",),
        state_keys=("left.obs_ee_pose", "left.obs_gripper"),
        action_keys=("left.cmd_ee_pose", "left.cmd_gripper"),
        state_dim=8,
        action_dim=8,
        state_semantics="raw_observed_ee_pose_gripper",
        action_semantics="raw_commanded_ee_pose_gripper",
        action_source_signal="commanded_ee_pose_gripper_timeline",
        action_is_control_command=True,
    ),
    "eva_right_arm": EgoVerseProfile(
        name="eva_right_arm",
        robot_name="egoverse-eva-right-arm",
        raw_embodiments=("eva_right_arm",),
        state_keys=("right.obs_ee_pose", "right.obs_gripper"),
        action_keys=("right.cmd_ee_pose", "right.cmd_gripper"),
        state_dim=8,
        action_dim=8,
        state_semantics="raw_observed_ee_pose_gripper",
        action_semantics="raw_commanded_ee_pose_gripper",
        action_source_signal="commanded_ee_pose_gripper_timeline",
        action_is_control_command=True,
    ),
    "scale_bimanual": EgoVerseProfile(
        name="scale_bimanual",
        robot_name="egoverse-scale-bimanual",
        raw_embodiments=("scale", "scale_bimanual"),
        state_keys=_HUMAN_BIMANUAL_KEYS,
        action_keys=_HUMAN_BIMANUAL_KEYS,
        state_dim=14,
        action_dim=14,
        state_semantics="raw_observed_ee_pose",
        action_semantics="raw_observed_ee_pose",
        action_source_signal="observed_ee_pose_timeline",
        action_is_control_command=False,
    ),
    "mecka_freeform_bimanual": EgoVerseProfile(
        name="mecka_freeform_bimanual",
        robot_name="egoverse-mecka-freeform-bimanual",
        raw_embodiments=("MECKA_BIMANUAL", "mecka_bimanual"),
        state_keys=_HUMAN_BIMANUAL_KEYS,
        action_keys=_HUMAN_BIMANUAL_KEYS,
        state_dim=14,
        action_dim=14,
        state_semantics="raw_observed_ee_pose",
        action_semantics="raw_observed_ee_pose",
        action_source_signal="observed_ee_pose_timeline",
        action_is_control_command=False,
    ),
    "mecka_flagship_bimanual": EgoVerseProfile(
        name="mecka_flagship_bimanual",
        robot_name="egoverse-mecka-flagship-bimanual",
        raw_embodiments=("MECKA_BIMANUAL", "mecka_bimanual"),
        state_keys=_HUMAN_BIMANUAL_KEYS,
        action_keys=_HUMAN_BIMANUAL_KEYS,
        state_dim=14,
        action_dim=14,
        state_semantics="raw_observed_ee_pose",
        action_semantics="raw_observed_ee_pose",
        action_source_signal="observed_ee_pose_timeline",
        action_is_control_command=False,
    ),
}


def get_egoverse_profile(name: str) -> EgoVerseProfile:
    key = str(name).strip().lower().replace("-", "_")
    if key.startswith("egoverse_"):
        key = key[len("egoverse_") :]
    try:
        return EGOVERSE_PROFILES[key]
    except KeyError as error:
        raise KeyError(
            f"unknown EgoVerse profile={name!r}; "
            f"available={sorted(EGOVERSE_PROFILES)}"
        ) from error


def open_egoverse_zarr_group(path: str) -> Any:
    if sys.version_info < (3, 11):
        raise ImportError(
            "EgoVerse reading requires Python >=3.11 and zarr==3.1.5"
        )
    try:
        import zarr
    except ImportError as error:
        raise ImportError(
            "EgoVerse reading requires the optional 'egoverse' dependency "
            "with Python >=3.11 and zarr==3.1.5"
        ) from error
    if str(getattr(zarr, "__version__", "")) != "3.1.5":
        raise ImportError(
            "EgoVerse reading requires zarr==3.1.5; "
            f"found {getattr(zarr, '__version__', 'unknown')}"
        )
    return zarr.open_group(str(path), mode="r")


def decode_egoverse_annotation(value: Any) -> Mapping[str, Any]:
    """Decode one variable-length JSON annotation entry."""

    if isinstance(value, np.void):
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("EgoVerse annotation must decode to a mapping")
    return value


class EgoVerseEpisodeReader(EpisodeSourceReader):
    """Read model-selected rows from one complete EgoVerse Zarr v3 Episode."""

    source_dataset = "egoverse"

    def __init__(
        self,
        group_opener: ZarrGroupOpener | None = None,
        *,
        resize_mode: str = "pad",
    ):
        if resize_mode not in {"pad", "interpolate"}:
            raise ValueError(
                "EgoVerse resize_mode must be 'pad' or 'interpolate'"
            )
        self._group_opener = group_opener or open_egoverse_zarr_group
        self._resize_mode = resize_mode

    @staticmethod
    def _locator(
        record: Mapping[str, Any], modality: str
    ) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        if locator["storage_backend"] != "zarr_v3":
            raise EpisodeRepositoryError(
                f"EgoVerse {modality} requires zarr_v3 storage"
            )
        return locator

    @staticmethod
    def _meta(record: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
        return record[modality].get("metadata") or {}

    @classmethod
    def _dimension(cls, record: Mapping[str, Any], modality: str) -> int:
        dimension = int(cls._meta(record, modality).get("dimension", 0))
        if dimension <= 0:
            raise EpisodeRepositoryError(
                f"EgoVerse Catalog must declare {modality} dimension"
            )
        return dimension

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        state_meta = self._meta(record, "state")
        action_meta = self._meta(record, "action")
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_zarr_frame",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "key": self._meta(record, "video").get("key"),
                    "valid_length_source": "root.total_frames",
                    "default_resize_mode": self._resize_mode,
                },
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (self._dimension(record, "state"),),
                "zero_based_raw_aligned_zarr_timestep",
                {
                    "normalization": "none",
                    "keys": list(state_meta.get("keys") or []),
                    "semantics": state_meta.get("semantics", "raw_aligned"),
                },
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (self._dimension(record, "action"),),
                "zero_based_raw_aligned_zarr_timestep",
                {
                    "normalization": "none",
                    "keys": list(action_meta.get("keys") or []),
                    "semantics": action_meta.get("semantics", "raw_aligned"),
                    "source_signal": action_meta.get("source_signal"),
                    "is_control_command": bool(
                        action_meta.get("is_control_command", False)
                    ),
                    "indices": "model_provided",
                },
            ),
            "text": ModalitySchema(
                "text",
                len(record["text"]["items"]),
                "structured_text_item",
                (),
                "zero_based_catalog_text_item",
                {},
            ),
        }

    @staticmethod
    def _contiguous_runs(indices: Sequence[int]) -> list[tuple[int, int]]:
        unique = sorted(set(int(value) for value in indices))
        if not unique:
            return []
        runs = []
        start = previous = unique[0]
        for value in unique[1:]:
            if value != previous + 1:
                runs.append((start, previous + 1))
                start = value
            previous = value
        runs.append((start, previous + 1))
        return runs

    @classmethod
    def _rows(cls, array: Any, indices: Sequence[int]) -> np.ndarray:
        width = int(np.prod(array.shape[1:])) if len(array.shape) > 1 else 1
        if not indices:
            return np.empty((0, width), dtype=np.float32)
        unique = sorted(set(int(value) for value in indices))
        parts = []
        for start, end in cls._contiguous_runs(unique):
            values = np.asarray(array[start:end])
            parts.append(values.reshape(end - start, -1))
        stacked = np.concatenate(parts, axis=0)
        positions = {value: position for position, value in enumerate(unique)}
        return stacked[[positions[int(value)] for value in indices]]

    @classmethod
    def _encoded_rows(cls, array: Any, indices: Sequence[int]) -> list[bytes]:
        unique = sorted(set(int(value) for value in indices))
        decoded: dict[int, bytes] = {}
        for start, end in cls._contiguous_runs(unique):
            values = np.asarray(array[start:end], dtype=object).reshape(-1)
            for offset, value in enumerate(values):
                if isinstance(value, np.void):
                    value = value.item()
                if isinstance(value, memoryview):
                    value = value.tobytes()
                if isinstance(value, bytearray):
                    value = bytes(value)
                if not isinstance(value, bytes):
                    raise EpisodeRepositoryError(
                        "EgoVerse JPEG array returned a non-bytes value"
                    )
                decoded[start + offset] = value
        return [decoded[int(value)] for value in indices]

    def _resize_video(
        self,
        frames: torch.Tensor,
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(
                f"unsupported video output_format={output_format!r}"
            )
        if output_format == "rgb_uint8_hwc" and video_size is None:
            return frames
        chw = frames.permute(0, 3, 1, 2).float().div_(255.0)
        if video_size is not None:
            target_h, target_w = (int(value) for value in video_size)
            if target_h <= 0 or target_w <= 0:
                raise EpisodeRepositoryError(
                    "EgoVerse video_size values must be positive"
                )
            if self._resize_mode == "interpolate":
                chw = F.interpolate(
                    chw,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                height, width = (int(value) for value in chw.shape[-2:])
                scale = min(target_h / height, target_w / width)
                new_h = max(1, int(height * scale))
                new_w = max(1, int(width * scale))
                resized = F.interpolate(
                    chw,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                padded = torch.zeros(
                    (
                        int(chw.shape[0]),
                        int(chw.shape[1]),
                        target_h,
                        target_w,
                    ),
                    dtype=resized.dtype,
                    device=resized.device,
                )
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[
                    :,
                    :,
                    y_offset : y_offset + new_h,
                    x_offset : x_offset + new_w,
                ] = resized
                chw = padded
        if output_format == "rgb_float32_chw":
            return chw
        return (
            chw.mul(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        locator = self._locator(record, "video")
        key = str(self._meta(record, "video").get("key") or "")
        if not key:
            raise EpisodeRepositoryError("EgoVerse video Zarr key is missing")
        group = self._group_opener(str(locator["uri"]))
        if key not in group:
            raise EpisodeRepositoryError(
                f"EgoVerse Zarr is missing video key={key!r}"
            )
        arrays = []
        for value in self._encoded_rows(group[key], indices):
            try:
                with Image.open(io.BytesIO(value)) as image:
                    arrays.append(np.asarray(image.convert("RGB")).copy())
            except Exception as error:
                raise EpisodeRepositoryError(
                    f"EgoVerse JPEG cannot be decoded: {error}"
                ) from error
        if len({tuple(array.shape) for array in arrays}) != 1:
            raise EpisodeRepositoryError(
                "EgoVerse decoded video frames have inconsistent shapes"
            )
        frames = torch.from_numpy(np.stack(arrays)).to(torch.uint8)
        return self._resize_video(
            frames,
            output_format=output_format,
            video_size=video_size,
        )

    def _numeric_from_group(
        self,
        group: Any,
        record: Mapping[str, Any],
        modality: str,
        indices: Sequence[int],
    ) -> torch.Tensor:
        keys = tuple(self._meta(record, modality).get("keys") or ())
        if not keys:
            raise EpisodeRepositoryError(
                f"EgoVerse {modality} has no ordered Zarr keys"
            )
        missing = [key for key in keys if key not in group]
        if missing:
            raise EpisodeRepositoryError(
                f"EgoVerse Zarr is missing {modality} keys={missing}"
            )
        parts = [self._rows(group[key], indices) for key in keys]
        result = torch.from_numpy(np.concatenate(parts, axis=-1)).float()
        expected = self._dimension(record, modality)
        if tuple(result.shape) != (len(indices), expected):
            raise EpisodeRepositoryError(
                f"EgoVerse {modality} dimension mismatch"
            )
        if not torch.isfinite(result).all():
            raise EpisodeRepositoryError(
                f"EgoVerse {modality} contains NaN/Inf"
            )
        return result

    def _read_numeric(
        self,
        record: Mapping[str, Any],
        modality: str,
        indices: Sequence[int],
    ) -> torch.Tensor:
        locator = self._locator(record, modality)
        group = self._group_opener(str(locator["uri"]))
        return self._numeric_from_group(group, record, modality, indices)

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._read_numeric(record, "state", indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._read_numeric(record, "action", indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_locator = self._locator(record, "state")
        action_locator = self._locator(record, "action")
        if str(state_locator["uri"]) != str(action_locator["uri"]):
            return super().read_state_action(
                record,
                state_indices,
                action_indices,
            )
        group = self._group_opener(str(state_locator["uri"]))
        return (
            self._numeric_from_group(
                group, record, "state", state_indices
            ),
            self._numeric_from_group(
                group, record, "action", action_indices
            ),
        )


def _sentence(text: str) -> str:
    """补终止标点（agibot TASK_PROMPT 的 "Task: {task}." 同款句式）。"""
    return text if text.endswith((".", "!", "?", ";", ":")) else text + "."


def compose_egoverse_text(items: Sequence[Any]) -> str:
    """把选中的 EgoVerse 文本项拼成 t5 输入文本（batch adapter 经注册表调用）。

    items = 可选的 episode 级 task_description_en + 与采样窗口重叠的全部
    annotation_en 叙述片段（时间序）。空文本跳过，未识别 kind 原文照拼
    （不丢内容）。
    """
    lines: list[str] = []
    for item in items:
        kind = str(getattr(item, "kind", ""))
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if kind == "task_description_en":
            lines.append(f"Task: {_sentence(text)}")
        elif kind == "annotation_en":
            lines.append(f"Action: {_sentence(text)}")
        else:
            lines.append(_sentence(text))
    return "\n".join(lines).strip()


register_episode_source_reader("egoverse", EgoVerseEpisodeReader)
register_text_composer("egoverse", compose_egoverse_text)
