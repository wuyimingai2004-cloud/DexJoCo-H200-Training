"""EGOSCALE LeRobot-v3 complete-Episode reader."""

from __future__ import annotations

import bisect
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from motus_data.datasets.agibot_common import decode_video_frame_map_ffmpeg
from motus_data.reader_registry import (
    register_episode_source_reader,
    register_text_composer,
)
from motus_data.repository import (
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
)


EGOSCALE_CAMERA_MODES = (
    "base_0",
    "base_1",
    "stereo_horizontal",
)
EGOSCALE_POSE_PROFILES = (
    "camera_pose_with_validity",
    "world_pose_with_validity",
)

VideoDecoder = Callable[[str, Sequence[int]], Mapping[int, np.ndarray]]


RAW_MODALITY_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "episode_index": ("int64", ()),
    "frame_index": ("int64", ()),
    "timestamp": ("float32", ()),
    "task_index": ("int64", ()),
    "left_hand/tracks": ("float32", (21, 3)),
    "right_hand/tracks": ("float32", (21, 3)),
    "left_hand/timestamp": ("int64", ()),
    "right_hand/timestamp": ("int64", ()),
    "head/pose": ("float32", (7,)),
    "head/valid": ("bool", ()),
    "head/timestamp": ("int64", ()),
    "left_hand/eepose": ("float32", (7,)),
    "right_hand/eepose": ("float32", (7,)),
    "left_hand/world_tracks": ("float32", (21, 3)),
    "left_hand/world_eepose": ("float32", (7,)),
    "right_hand/world_tracks": ("float32", (21, 3)),
    "right_hand/world_eepose": ("float32", (7,)),
    "left_hand/mano/valid": ("bool", ()),
    "left_hand/mano/global_orient": ("float32", (1, 3, 3)),
    "left_hand/mano/hand_pose": ("float32", (15, 3, 3)),
    "left_hand/mano/betas": ("float32", (10,)),
    "left_hand/mano/transl": ("float32", (3,)),
    "left_hand/mano/scale": ("float32", ()),
    "left_hand/mano/extra_rot": ("float32", (3, 3)),
    "right_hand/mano/valid": ("bool", ()),
    "right_hand/mano/global_orient": ("float32", (1, 3, 3)),
    "right_hand/mano/hand_pose": ("float32", (15, 3, 3)),
    "right_hand/mano/betas": ("float32", (10,)),
    "right_hand/mano/transl": ("float32", (3,)),
    "right_hand/mano/scale": ("float32", ()),
    "right_hand/mano/extra_rot": ("float32", (3, 3)),
}


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "EGOSCALE reading requires the optional 'egoscale' dependency: "
            "install pyarrow"
        ) from error
    return pa, pq


class EgoScaleEpisodeReader(EpisodeSourceReader):
    """Read exact model-owned indices from one EGOSCALE Episode package."""

    source_dataset = "egoscale"

    def __init__(
        self,
        *,
        camera_mode: str = "base_0",
        pose_profile: str = "camera_pose_with_validity",
        resize_mode: str = "pad",
        video_decoder: VideoDecoder | None = None,
    ) -> None:
        camera_mode = str(camera_mode).strip().lower()
        pose_profile = str(pose_profile).strip().lower()
        resize_mode = str(resize_mode).strip().lower()
        if camera_mode not in EGOSCALE_CAMERA_MODES:
            raise ValueError(
                f"EGOSCALE camera_mode must be one of {EGOSCALE_CAMERA_MODES}"
            )
        if pose_profile not in EGOSCALE_POSE_PROFILES:
            raise ValueError(
                "EGOSCALE pose_profile must be one of " f"{EGOSCALE_POSE_PROFILES}"
            )
        if resize_mode not in {"pad", "interpolate"}:
            raise ValueError("EGOSCALE resize_mode must be 'pad' or 'interpolate'")
        self.camera_mode = camera_mode
        self.pose_profile = pose_profile
        self.resize_mode = resize_mode
        self._video_decoder = video_decoder or decode_video_frame_map_ffmpeg

    @staticmethod
    def _locator(record: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        expected = "mp4" if modality == "video" else "parquet"
        if locator["storage_backend"] != expected:
            raise EpisodeRepositoryError(
                f"EGOSCALE {modality} requires {expected} storage"
            )
        return locator

    @staticmethod
    def _meta(record: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
        return record[modality].get("metadata") or {}

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        video_meta = self._meta(record, "video")
        numeric_metadata = {
            "dimension": 24,
            "profile": self.pose_profile,
            "normalization": "none",
            "feature_order": [
                "head_pose[7]",
                "head_valid",
                "left_hand_eepose[7]",
                "left_hand_valid",
                "right_hand_eepose[7]",
                "right_hand_valid",
            ],
            "missing_policy": "zero_pose_with_explicit_validity_bit",
            "indices": "model_provided",
        }
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_decoded_video_frame",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "available_views": list(video_meta.get("views") or []),
                    "camera_mode": self.camera_mode,
                    "default_camera_mode": "base_0",
                    "native_size": list(video_meta.get("native_size") or []),
                    "default_resize_mode": self.resize_mode,
                },
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (24,),
                "zero_based_aligned_observed_pose_timestep",
                dict(numeric_metadata, semantics="observed_human_pose"),
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (24,),
                "zero_based_aligned_observed_pose_timestep",
                dict(
                    numeric_metadata,
                    semantics="observed_human_motion_proxy",
                    source_signal="same_aligned_observed_pose_as_state",
                    is_control_command=False,
                ),
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

    def source_modality_schemas(
        self, record: Mapping[str, Any]
    ) -> Mapping[str, ModalitySchema]:
        count = int(record["state"]["count"])
        return {
            name: ModalitySchema(
                name,
                count,
                dtype,
                shape,
                "zero_based_aligned_parquet_row",
                {
                    "source_column": name,
                    "null_policy": "zero_fill; request the matching validity field",
                },
            )
            for name, (dtype, shape) in RAW_MODALITY_SPECS.items()
        }

    @staticmethod
    def _row_group_bounds(parquet_file: Any) -> tuple[list[int], list[int]]:
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for row_group in range(parquet_file.metadata.num_row_groups):
            starts.append(cursor)
            cursor += int(parquet_file.metadata.row_group(row_group).num_rows)
            ends.append(cursor)
        return starts, ends

    @classmethod
    def _read_selected_columns(
        cls,
        path: str,
        requests: Mapping[str, Sequence[int]],
        *,
        expected_rows: int,
    ) -> dict[str, Any]:
        pa, pq = _require_pyarrow()
        active = {
            str(column): tuple(int(index) for index in indices)
            for column, indices in requests.items()
            if indices
        }
        if not active:
            return {}
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as error:
            raise EpisodeRepositoryError(
                f"failed to open EGOSCALE Parquet {path}: {error}"
            ) from error
        if int(parquet_file.metadata.num_rows) != int(expected_rows):
            raise EpisodeRepositoryError(
                "EGOSCALE Parquet row count differs from Catalog: " f"path={path}"
            )
        missing = sorted(set(active).difference(parquet_file.schema_arrow.names))
        if missing:
            raise EpisodeRepositoryError(
                f"EGOSCALE Parquet is missing columns={missing}"
            )
        starts, ends = cls._row_group_bounds(parquet_file)
        requested = sorted({index for indices in active.values() for index in indices})
        row_groups = sorted({bisect.bisect_right(ends, index) for index in requested})
        if row_groups and row_groups[-1] >= len(starts):
            raise EpisodeRepositoryError(
                "EGOSCALE requested row is outside Parquet metadata"
            )
        loaded_starts: dict[int, int] = {}
        loaded_cursor = 0
        for row_group in row_groups:
            loaded_starts[row_group] = loaded_cursor
            loaded_cursor += ends[row_group] - starts[row_group]
        try:
            table = parquet_file.read_row_groups(
                row_groups,
                columns=list(active),
                use_threads=False,
            )
        except Exception as error:
            raise EpisodeRepositoryError(
                f"failed to read EGOSCALE Parquet row groups: {error}"
            ) from error
        result = {}
        for column, indices in active.items():
            positions = []
            for index in indices:
                row_group = bisect.bisect_right(ends, index)
                positions.append(loaded_starts[row_group] + index - starts[row_group])
            result[column] = (
                table[column]
                .combine_chunks()
                .take(pa.array(positions, type=pa.int64()))
            )
        return result

    @staticmethod
    def _raw_tensor(
        array: Any | None,
        *,
        rows: int,
        dtype: str,
        shape: tuple[int, ...],
        column: str,
    ) -> torch.Tensor:
        torch_dtype = {
            "bool": torch.bool,
            "int64": torch.int64,
            "float32": torch.float32,
        }[dtype]
        numpy_dtype = {
            "bool": np.bool_,
            "int64": np.int64,
            "float32": np.float32,
        }[dtype]
        if rows == 0:
            return torch.empty((0, *shape), dtype=torch_dtype)
        if array is None:
            raise EpisodeRepositoryError(
                f"EGOSCALE source column {column!r} was not loaded"
            )
        values = []
        fill = False if dtype == "bool" else 0

        def replace_missing(value: Any) -> Any:
            if value is None:
                return fill
            if isinstance(value, (list, tuple)):
                return [replace_missing(item) for item in value]
            return value

        for position, value in enumerate(array.to_pylist()):
            if value is None:
                values.append(np.full(shape or (), fill, dtype=numpy_dtype))
                continue
            converted = np.asarray(replace_missing(value), dtype=numpy_dtype)
            if converted.shape != shape:
                raise EpisodeRepositoryError(
                    f"EGOSCALE {column}[{position}] shape={converted.shape} "
                    f"expected={shape}"
                )
            if dtype == "float32" and not np.isfinite(converted).all():
                converted = np.where(np.isfinite(converted), converted, 0.0)
            values.append(converted)
        result = np.asarray(values, dtype=numpy_dtype).reshape(rows, *shape)
        return torch.as_tensor(result, dtype=torch_dtype).clone()

    def read_source_modalities(
        self,
        record: Mapping[str, Any],
        requests: Mapping[str, Sequence[int]],
    ) -> Mapping[str, torch.Tensor]:
        unknown = sorted(set(requests).difference(RAW_MODALITY_SPECS))
        if unknown:
            raise EpisodeRepositoryError(
                f"unknown EGOSCALE source modalities={unknown}"
            )
        count = int(record["state"]["count"])
        path = str(self._locator(record, "state")["uri"])
        arrays = self._read_selected_columns(
            path,
            requests,
            expected_rows=count,
        )
        return {
            name: self._raw_tensor(
                arrays.get(name),
                rows=len(indices),
                dtype=RAW_MODALITY_SPECS[name][0],
                shape=RAW_MODALITY_SPECS[name][1],
                column=name,
            )
            for name, indices in requests.items()
        }

    @staticmethod
    def _pose_rows(
        pose_array: Any,
        valid_array: Any,
        *,
        rows: int,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poses = []
        validity = []
        pose_values = pose_array.to_pylist()
        valid_values = valid_array.to_pylist()
        for position, (raw_pose, raw_valid) in enumerate(
            zip(pose_values, valid_values)
        ):
            valid = bool(raw_valid) if raw_valid is not None else False
            if raw_pose is None:
                pose = np.zeros(7, dtype=np.float32)
                valid = False
            else:
                pose = np.asarray(raw_pose, dtype=np.float32)
                if pose.shape != (7,):
                    raise EpisodeRepositoryError(
                        f"EGOSCALE {name}[{position}] shape={pose.shape}"
                    )
                if not np.isfinite(pose).all():
                    pose = np.zeros(7, dtype=np.float32)
                    valid = False
            if not valid:
                pose = np.zeros(7, dtype=np.float32)
            poses.append(pose)
            validity.append(float(valid))
        if len(poses) != rows:
            raise EpisodeRepositoryError(f"EGOSCALE {name} row count changed")
        return (
            torch.from_numpy(np.stack(poses)).float(),
            torch.tensor(validity, dtype=torch.float32).reshape(rows, 1),
        )

    def _pose_tensor(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
    ) -> torch.Tensor:
        if not indices:
            return torch.empty((0, 24), dtype=torch.float32)
        eepose_suffix = (
            "world_eepose"
            if self.pose_profile == "world_pose_with_validity"
            else "eepose"
        )
        columns = (
            "head/pose",
            "head/valid",
            f"left_hand/{eepose_suffix}",
            "left_hand/mano/valid",
            f"right_hand/{eepose_suffix}",
            "right_hand/mano/valid",
        )
        requests = {column: indices for column in columns}
        arrays = self._read_selected_columns(
            str(self._locator(record, "state")["uri"]),
            requests,
            expected_rows=int(record["state"]["count"]),
        )
        head, head_valid = self._pose_rows(
            arrays["head/pose"],
            arrays["head/valid"],
            rows=len(indices),
            name="head/pose",
        )
        left, left_valid = self._pose_rows(
            arrays[f"left_hand/{eepose_suffix}"],
            arrays["left_hand/mano/valid"],
            rows=len(indices),
            name=f"left_hand/{eepose_suffix}",
        )
        right, right_valid = self._pose_rows(
            arrays[f"right_hand/{eepose_suffix}"],
            arrays["right_hand/mano/valid"],
            rows=len(indices),
            name=f"right_hand/{eepose_suffix}",
        )
        result = torch.cat(
            [head, head_valid, left, left_valid, right, right_valid], dim=-1
        )
        if tuple(result.shape) != (len(indices), 24):
            raise EpisodeRepositoryError("EGOSCALE pose profile dimension changed")
        return result

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._pose_tensor(record, indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._pose_tensor(record, indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        union = sorted(set(state_indices).union(action_indices))
        values = self._pose_tensor(record, union)
        positions = {index: position for position, index in enumerate(union)}

        def take(indices: Sequence[int]) -> torch.Tensor:
            if not indices:
                return values.new_empty((0, 24))
            return values[
                torch.tensor(
                    [positions[int(index)] for index in indices],
                    dtype=torch.long,
                )
            ].clone()

        return take(state_indices), take(action_indices)

    def _view_paths(self, record: Mapping[str, Any]) -> Mapping[str, str]:
        metadata = self._meta(record, "video")
        paths = metadata.get("view_paths") or {}
        if not isinstance(paths, Mapping):
            raise EpisodeRepositoryError(
                "EGOSCALE Catalog video.view_paths must be a mapping"
            )
        required = (
            ("base_0",)
            if self.camera_mode == "base_0"
            else ("base_1",) if self.camera_mode == "base_1" else ("base_0", "base_1")
        )
        missing = [name for name in required if not str(paths.get(name) or "")]
        if missing:
            raise EpisodeRepositoryError(
                f"EGOSCALE Catalog lacks video paths for views={missing}"
            )
        return {name: str(paths[name]) for name in required}

    def _decode_view(
        self,
        path: str,
        indices: Sequence[int],
    ) -> torch.Tensor:
        try:
            decoded = self._video_decoder(path, indices)
        except Exception as error:
            raise EpisodeRepositoryError(
                f"failed to decode EGOSCALE video {path}: {error}"
            ) from error
        missing = [index for index in set(indices) if index not in decoded]
        if missing:
            raise EpisodeRepositoryError(
                f"EGOSCALE video decoder missed frame indices={sorted(missing)}"
            )
        arrays = [np.asarray(decoded[int(index)]) for index in indices]
        if any(array.ndim != 3 or array.shape[-1] != 3 for array in arrays):
            raise EpisodeRepositoryError("EGOSCALE decoded frame shape changed")
        if len({array.shape for array in arrays}) != 1:
            raise EpisodeRepositoryError(
                "EGOSCALE decoded frames have inconsistent shapes"
            )
        return torch.from_numpy(np.stack(arrays)).to(torch.uint8)

    def _format_video(
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
                    "EGOSCALE video_size values must be positive"
                )
            if self.resize_mode == "interpolate":
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
                chw = torch.zeros(
                    (len(frames), 3, target_h, target_w),
                    dtype=resized.dtype,
                )
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                chw[
                    :,
                    :,
                    y_offset : y_offset + new_h,
                    x_offset : x_offset + new_w,
                ] = resized
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
        views = [
            self._decode_view(path, indices)
            for path in self._view_paths(record).values()
        ]
        if len(views) == 2:
            if views[0].shape[:2] != views[1].shape[:2] or (
                views[0].shape[3] != views[1].shape[3]
            ):
                raise EpisodeRepositoryError(
                    "EGOSCALE stereo views have incompatible shapes"
                )
            frames = torch.cat(views, dim=2)
        else:
            frames = views[0]
        return self._format_video(
            frames,
            output_format=output_format,
            video_size=video_size,
        )


def _sentence(text: str) -> str:
    """补终止标点（agibot TASK_PROMPT 的 "Task: {task}." 同款句式）。"""
    return text if text.endswith((".", "!", "?", ";", ":")) else text + "."


# episode 级任务字段 kind → 输出行标签（顺序即 _SEGMENTED_TEXT_FAMILIES 的
# context 顺序，拼接时按选中 item 顺序逐行输出）。
_EGOSCALE_CONTEXT_LABELS = {
    "task_name_en": "Task name",
    "task_description_en": "Task",
    "task_detail_en": "Task detail",
    "task_label": "Task label",
}
# 片段级 meta 字段 → 输出行标签（片段所有字段都拼入，空值跳过）。
_EGOSCALE_SEGMENT_META_LABELS = (
    ("action", "Action label"),
    ("hand", "Hand"),
    ("scene_semantic", "Scene semantic"),
    ("camera_motion", "Camera motion"),
)


def compose_egoscale_text(items: Sequence[Any]) -> str:
    """把选中的 EGOSCALE 文本项拼成 t5 输入文本（batch adapter 经注册表调用）。

    items = episode 级任务字段（task_name_en/task_description_en/
    task_detail_en/task_label，可缺）+ 与采样窗口重叠的全部 fine_action_en
    片段（时间序）。每个片段的所有字段（描述 text + action/hand/
    scene_semantic/camera_motion meta）都拼入；空字段跳过，未识别 kind 原文
    照拼（不丢内容）。
    """
    lines: list[str] = []
    for item in items:
        kind = str(getattr(item, "kind", ""))
        text = str(getattr(item, "text", "") or "").strip()
        if kind in _EGOSCALE_CONTEXT_LABELS:
            if text:
                lines.append(f"{_EGOSCALE_CONTEXT_LABELS[kind]}: {_sentence(text)}")
        elif kind == "fine_action_en":
            metadata = getattr(item, "metadata", None) or {}
            if text:
                lines.append(f"Action: {_sentence(text)}")
            for key, label in _EGOSCALE_SEGMENT_META_LABELS:
                value = str(metadata.get(key) or "").strip()
                if value:
                    lines.append(f"{label}: {_sentence(value)}")
        elif text:
            lines.append(_sentence(text))
    return "\n".join(lines).strip()


register_episode_source_reader("egoscale", EgoScaleEpisodeReader)
register_text_composer("egoscale", compose_egoscale_text)


__all__ = [
    "EGOSCALE_CAMERA_MODES",
    "EGOSCALE_POSE_PROFILES",
    "RAW_MODALITY_SPECS",
    "EgoScaleEpisodeReader",
    "compose_egoscale_text",
]
