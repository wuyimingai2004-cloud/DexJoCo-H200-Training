"""LIBERO LeRobot-v2 Parquet Episode reader."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from motus_data.reader_registry import register_episode_source_reader
from motus_data.repository import (
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
)


LIBERO_CAMERA_MODES = (
    "main",
    "wrist",
    "main_wrist_horizontal",
)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "LIBERO Parquet reading requires the optional 'libero' "
            "dependency: install pyarrow"
        ) from error
    return pa, pq


class LiberoParquetEpisodeReader(EpisodeSourceReader):
    """Read model-selected rows from one complete LIBERO Episode."""

    source_dataset = "libero"

    def __init__(
        self,
        *,
        camera_mode: str = "main",
        resize_mode: str = "pad",
        normalization_mode: str = "zscore",
        statistics_path: str | Path | None = None,
    ) -> None:
        camera_mode = str(camera_mode).strip().lower()
        if camera_mode not in LIBERO_CAMERA_MODES:
            raise ValueError(
                f"LIBERO camera_mode must be one of {LIBERO_CAMERA_MODES}"
            )
        if resize_mode not in {"pad", "interpolate"}:
            raise ValueError(
                "LIBERO resize_mode must be 'pad' or 'interpolate'"
            )
        normalization_mode = str(normalization_mode).strip().lower()
        if normalization_mode == "none":
            normalization_mode = "raw"
        if normalization_mode not in {"raw", "zscore"}:
            raise ValueError(
                "LIBERO normalization_mode must be 'raw' or 'zscore'"
            )
        self.camera_mode = camera_mode
        self.resize_mode = resize_mode
        self.normalization_mode = normalization_mode
        self.statistics_path = (
            None if statistics_path is None else Path(statistics_path)
        )
        self._expected_statistics_sha256: str | None = None
        self._statistics_cache: dict[
            str, dict[str, tuple[torch.Tensor, torch.Tensor]]
        ] = {}

    def bind_catalog(self, metadata: Mapping[str, Any]) -> None:
        if str(metadata.get("source")) != self.source_dataset:
            raise EpisodeRepositoryError("LIBERO Catalog source mismatch")
        source_digests = (
            (metadata.get("meta") or {}).get("source_metadata_sha256") or {}
        )
        expected = source_digests.get("stats.json")
        self._expected_statistics_sha256 = (
            None if expected is None else str(expected)
        )
        if (
            self.normalization_mode == "zscore"
            and self._expected_statistics_sha256 is None
        ):
            raise EpisodeRepositoryError(
                "LIBERO zscore requires Catalog-bound stats.json SHA256"
            )

    @staticmethod
    def _locator(
        record: Mapping[str, Any],
        modality: str,
    ) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        if locator["storage_backend"] != "parquet":
            raise EpisodeRepositoryError(
                f"LIBERO {modality} requires parquet storage"
            )
        return locator

    @staticmethod
    def _meta(
        record: Mapping[str, Any],
        modality: str,
    ) -> Mapping[str, Any]:
        return record[modality].get("metadata") or {}

    @classmethod
    def _dimension(
        cls,
        record: Mapping[str, Any],
        modality: str,
    ) -> int:
        dimension = int(cls._meta(record, modality).get("dimension", 0))
        if dimension <= 0:
            raise EpisodeRepositoryError(
                f"LIBERO Catalog must declare {modality} dimension"
            )
        return dimension

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        video_meta = self._meta(record, "video")
        state_meta = self._meta(record, "state")
        action_meta = self._meta(record, "action")
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_parquet_row",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "available_views": list(video_meta.get("views") or []),
                    "camera_mode": self.camera_mode,
                    "default_camera_mode": "main",
                    "native_size": list(
                        video_meta.get("native_size") or [256, 256]
                    ),
                    "default_resize_mode": self.resize_mode,
                },
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (self._dimension(record, "state"),),
                "zero_based_observation_parquet_row",
                {
                    "column": state_meta.get("column", "state"),
                    "semantics": state_meta.get(
                        "semantics", "absolute_eef_state"
                    ),
                    "normalization": self.normalization_mode,
                    "normalization_statistics": "meta/stats.json",
                    "indices": "model_provided",
                },
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (self._dimension(record, "action"),),
                "zero_based_command_parquet_row",
                {
                    "column": action_meta.get("column", "actions"),
                    "semantics": action_meta.get(
                        "semantics", "delta_eef_command"
                    ),
                    "normalization": self.normalization_mode,
                    "normalization_statistics": "meta/stats.json",
                    "zscore_includes_gripper": True,
                    "is_control_command": True,
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
                f"failed to open LIBERO Parquet {path}: {error}"
            ) from error
        actual_rows = int(parquet_file.metadata.num_rows)
        if actual_rows != int(expected_rows):
            raise EpisodeRepositoryError(
                "LIBERO Parquet row count differs from Catalog: "
                f"actual={actual_rows} expected={expected_rows} path={path}"
            )
        missing_columns = sorted(
            set(active).difference(parquet_file.schema_arrow.names)
        )
        if missing_columns:
            raise EpisodeRepositoryError(
                f"LIBERO Parquet is missing columns={missing_columns}"
            )

        starts, ends = cls._row_group_bounds(parquet_file)
        requested_indices = sorted(
            {index for indices in active.values() for index in indices}
        )
        row_groups = sorted(
            {
                bisect.bisect_right(ends, index)
                for index in requested_indices
            }
        )
        if row_groups and row_groups[-1] >= len(starts):
            raise EpisodeRepositoryError(
                "LIBERO requested row is outside Parquet metadata"
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
                f"failed to read LIBERO Parquet row groups: {error}"
            ) from error

        result = {}
        for column, indices in active.items():
            positions = []
            for index in indices:
                row_group = bisect.bisect_right(ends, index)
                positions.append(
                    loaded_starts[row_group] + index - starts[row_group]
                )
            values = table[column].combine_chunks()
            result[column] = values.take(pa.array(positions, type=pa.int64()))
        return result

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                payload = handle.read(1024 * 1024)
                if not payload:
                    return digest.hexdigest()
                digest.update(payload)

    def _resolved_statistics_path(
        self,
        record: Mapping[str, Any],
    ) -> Path:
        if self.statistics_path is not None:
            return self.statistics_path
        parquet_path = Path(str(self._locator(record, "state")["uri"]))
        try:
            dataset_root = parquet_path.parents[2]
        except IndexError as error:
            raise EpisodeRepositoryError(
                f"cannot derive LIBERO stats path from {parquet_path}"
            ) from error
        return dataset_root / "meta" / "stats.json"

    def _normalization_statistics(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        path = self._resolved_statistics_path(record)
        cache_key = str(path)
        cached = self._statistics_cache.get(cache_key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise EpisodeRepositoryError(
                f"LIBERO normalization statistics are missing: {path}"
            )
        actual_sha256 = self._sha256(path)
        if actual_sha256 != self._expected_statistics_sha256:
            raise EpisodeRepositoryError(
                "LIBERO stats.json SHA256 differs from Catalog binding: "
                f"actual={actual_sha256} "
                f"expected={self._expected_statistics_sha256}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EpisodeRepositoryError(
                f"failed to load LIBERO normalization statistics: {error}"
            ) from error
        result = {}
        for modality, key, dimension in (
            ("state", "state", 8),
            ("action", "actions", 7),
        ):
            item = value.get(key)
            if not isinstance(item, Mapping):
                raise EpisodeRepositoryError(
                    f"LIBERO stats.json lacks {key!r}"
                )
            mean = torch.as_tensor(item.get("mean"), dtype=torch.float32)
            std = torch.as_tensor(item.get("std"), dtype=torch.float32)
            if tuple(mean.shape) != (dimension,) or tuple(std.shape) != (
                dimension,
            ):
                raise EpisodeRepositoryError(
                    f"LIBERO {modality} statistics shape changed"
                )
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(std).all()
                or torch.any(std <= 0)
            ):
                raise EpisodeRepositoryError(
                    f"LIBERO {modality} statistics are invalid"
                )
            result[modality] = (mean, std)
        self._statistics_cache[cache_key] = result
        return result

    def _numeric_tensor(
        self,
        array: Any | None,
        *,
        rows: int,
        dimension: int,
        modality: str,
        record: Mapping[str, Any],
    ) -> torch.Tensor:
        if rows == 0:
            return torch.empty((0, dimension), dtype=torch.float32)
        if array is None:
            raise EpisodeRepositoryError(
                f"LIBERO {modality} column was not loaded"
            )
        try:
            values = np.asarray(array.to_pylist(), dtype=np.float32)
        except Exception as error:
            raise EpisodeRepositoryError(
                f"failed to decode LIBERO {modality}: {error}"
            ) from error
        if values.shape != (rows, dimension):
            raise EpisodeRepositoryError(
                f"LIBERO {modality} shape={values.shape} expected="
                f"({rows}, {dimension})"
            )
        result = torch.from_numpy(values.copy())
        if not torch.isfinite(result).all():
            raise EpisodeRepositoryError(
                f"LIBERO {modality} contains NaN/Inf"
            )
        if self.normalization_mode == "zscore":
            mean, std = self._normalization_statistics(record)[modality]
            result = (result - mean) / std
            if not torch.isfinite(result).all():
                raise EpisodeRepositoryError(
                    f"normalized LIBERO {modality} contains NaN/Inf"
                )
        return result

    @staticmethod
    def _decode_png_rows(array: Any, *, column: str) -> list[np.ndarray]:
        frames = []
        for position, value in enumerate(array.to_pylist()):
            if not isinstance(value, Mapping):
                raise EpisodeRepositoryError(
                    f"LIBERO {column}[{position}] is not an image struct"
                )
            payload = value.get("bytes")
            if isinstance(payload, memoryview):
                payload = payload.tobytes()
            if not isinstance(payload, bytes) or not payload:
                raise EpisodeRepositoryError(
                    f"LIBERO {column}[{position}] has no PNG bytes"
                )
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "PNG":
                        raise ValueError(f"encoding={image.format!r}")
                    frames.append(np.asarray(image.convert("RGB")).copy())
            except Exception as error:
                raise EpisodeRepositoryError(
                    f"failed to decode LIBERO {column}[{position}]: {error}"
                ) from error
        return frames

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
                    "LIBERO video_size values must be positive"
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
                new_h = max(1, int(round(height * scale)))
                new_w = max(1, int(round(width * scale)))
                resized = F.interpolate(
                    chw,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                padded = torch.zeros(
                    (len(frames), 3, target_h, target_w),
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
        if not indices:
            raise EpisodeRepositoryError("LIBERO video indices cannot be empty")
        columns = {
            "main": ("image",),
            "wrist": ("wrist_image",),
            "main_wrist_horizontal": ("image", "wrist_image"),
        }[self.camera_mode]
        arrays = self._read_selected_columns(
            str(locator["uri"]),
            {column: indices for column in columns},
            expected_rows=int(record["video"]["frame_count"]),
        )
        decoded = {
            column: self._decode_png_rows(arrays[column], column=column)
            for column in columns
        }
        if self.camera_mode == "main_wrist_horizontal":
            frame_arrays = [
                np.concatenate((main, wrist), axis=1)
                for main, wrist in zip(decoded["image"], decoded["wrist_image"])
            ]
        else:
            frame_arrays = decoded[columns[0]]
        if len({tuple(frame.shape) for frame in frame_arrays}) != 1:
            raise EpisodeRepositoryError(
                "LIBERO decoded video frames have inconsistent shapes"
            )
        frames = torch.from_numpy(np.stack(frame_arrays)).to(torch.uint8)
        return self._format_video(
            frames,
            output_format=output_format,
            video_size=video_size,
        )

    def _read_numeric(
        self,
        record: Mapping[str, Any],
        modality: str,
        indices: Sequence[int],
    ) -> torch.Tensor:
        dimension = self._dimension(record, modality)
        if not indices:
            return torch.empty((0, dimension), dtype=torch.float32)
        locator = self._locator(record, modality)
        column = str(
            self._meta(record, modality).get(
                "column", "state" if modality == "state" else "actions"
            )
        )
        arrays = self._read_selected_columns(
            str(locator["uri"]),
            {column: indices},
            expected_rows=int(record[modality]["count"]),
        )
        return self._numeric_tensor(
            arrays.get(column),
            rows=len(indices),
            dimension=dimension,
            modality=modality,
            record=record,
        )

    def read_state(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
    ) -> torch.Tensor:
        return self._read_numeric(record, "state", indices)

    def read_action(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
    ) -> torch.Tensor:
        return self._read_numeric(record, "action", indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_dimension = self._dimension(record, "state")
        action_dimension = self._dimension(record, "action")
        state_locator = self._locator(record, "state")
        action_locator = self._locator(record, "action")
        if str(state_locator["uri"]) != str(action_locator["uri"]):
            return super().read_state_action(
                record,
                state_indices,
                action_indices,
            )
        state_column = str(self._meta(record, "state").get("column", "state"))
        action_column = str(
            self._meta(record, "action").get("column", "actions")
        )
        arrays = self._read_selected_columns(
            str(state_locator["uri"]),
            {
                state_column: state_indices,
                action_column: action_indices,
            },
            expected_rows=int(record["state"]["count"]),
        )
        return (
            self._numeric_tensor(
                arrays.get(state_column),
                rows=len(state_indices),
                dimension=state_dimension,
                modality="state",
                record=record,
            ),
            self._numeric_tensor(
                arrays.get(action_column),
                rows=len(action_indices),
                dimension=action_dimension,
                modality="action",
                record=record,
            ),
        )


register_episode_source_reader("libero", LiberoParquetEpisodeReader)
