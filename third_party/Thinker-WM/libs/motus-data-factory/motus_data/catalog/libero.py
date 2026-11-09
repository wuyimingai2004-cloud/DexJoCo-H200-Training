"""Build split-neutral LIBERO complete-Episode Catalogs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_record_v2,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class LiberoChildSpec:
    name: str
    episode_start: int
    episode_stop: int
    task_start: int
    task_stop: int

    @property
    def episode_count(self) -> int:
        return self.episode_stop - self.episode_start


LIBERO_CHILDREN: dict[str, LiberoChildSpec] = {
    "libero_10": LiberoChildSpec("libero_10", 0, 379, 0, 10),
    "libero_goal": LiberoChildSpec("libero_goal", 379, 807, 10, 20),
    "libero_object": LiberoChildSpec("libero_object", 807, 1261, 20, 30),
    "libero_spatial": LiberoChildSpec("libero_spatial", 1261, 1693, 30, 40),
}


def get_libero_child(name: str) -> LiberoChildSpec:
    key = str(name).strip().lower().replace("-", "_")
    try:
        return LIBERO_CHILDREN[key]
    except KeyError as error:
        raise KeyError(
            f"unknown LIBERO child={name!r}; "
            f"available={sorted(LIBERO_CHILDREN)}"
        ) from error


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "LIBERO Catalog building requires the optional 'libero' "
            "dependency: install pyarrow"
        ) from error
    return pa, pq


def _metadata_path(catalog: Path) -> Path:
    return catalog.with_suffix(".meta.json")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            payload = handle.read(1024 * 1024)
            if not payload:
                return digest.hexdigest()
            digest.update(payload)


def _normalization_contract(stats_path: Path) -> dict[str, Any]:
    value = json.loads(stats_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "default_mode": "zscore",
        "supported_modes": ["raw", "zscore"],
        "statistics_file": "meta/stats.json",
        "zscore_includes_action_gripper": True,
        "modalities": {},
    }
    for modality, key, dimension in (
        ("state", "state", 8),
        ("action", "actions", 7),
    ):
        item = value.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"LIBERO stats.json lacks {key!r}")
        mean = np.asarray(item.get("mean"), dtype=np.float64)
        std = np.asarray(item.get("std"), dtype=np.float64)
        if mean.shape != (dimension,) or std.shape != (dimension,):
            raise ValueError(
                f"LIBERO {modality} normalization dimension changed"
            )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError(
                f"LIBERO {modality} normalization statistics are invalid"
            )
        result["modalities"][modality] = {
            "dimension": dimension,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parquet_relative_path(info: Mapping[str, Any], episode_index: int) -> Path:
    template = str(info.get("data_path") or "")
    expected = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    if template != expected:
        raise ValueError(
            f"LIBERO data_path template changed: {template!r}"
        )
    chunk_size = int(info.get("chunks_size", 0))
    if chunk_size <= 0:
        raise ValueError("LIBERO chunks_size must be positive")
    return Path(
        template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
    )


def _fixed_list_dimension(pa: Any, field: Any, *, name: str) -> int:
    if not pa.types.is_fixed_size_list(field.type):
        raise ValueError(f"LIBERO {name} must be a fixed-size list")
    if not pa.types.is_float32(field.type.value_type):
        raise ValueError(f"LIBERO {name} values must be float32")
    return int(field.type.list_size)


def _validate_image_type(pa: Any, field: Any, *, name: str) -> None:
    if not pa.types.is_struct(field.type):
        raise ValueError(f"LIBERO {name} must be a struct")
    children = {child.name: child.type for child in field.type}
    if set(children) != {"bytes", "path"}:
        raise ValueError(
            f"LIBERO {name} struct fields changed: {sorted(children)}"
        )
    if not pa.types.is_binary(children["bytes"]):
        raise ValueError(f"LIBERO {name}.bytes must be binary")
    if not pa.types.is_string(children["path"]):
        raise ValueError(f"LIBERO {name}.path must be string")


def _validate_null_statistics(parquet_file: Any, path: Path) -> None:
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            statistics = column.statistics
            if statistics is None or not statistics.has_null_count:
                raise ValueError(
                    "LIBERO Parquet lacks null statistics: "
                    f"path={path} column={column.path_in_schema}"
                )
            if int(statistics.null_count) != 0:
                raise ValueError(
                    "LIBERO Parquet contains null values: "
                    f"path={path} column={column.path_in_schema} "
                    f"null_count={statistics.null_count}"
                )


def _validate_parquet_episode(
    path: Path,
    *,
    expected_episode_index: int,
    expected_task_index: int,
    expected_rows: int,
    expected_global_start: int,
    fps: float,
    validate_numeric_finite: bool,
) -> dict[str, Any]:
    pa, pq = _require_pyarrow()
    parquet_file = pq.ParquetFile(path)
    if int(parquet_file.metadata.num_rows) != expected_rows:
        raise ValueError(
            f"LIBERO row count changed: path={path} "
            f"actual={parquet_file.metadata.num_rows} expected={expected_rows}"
        )
    schema = parquet_file.schema_arrow
    required = {
        "image",
        "wrist_image",
        "state",
        "actions",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if set(schema.names) != required:
        raise ValueError(
            f"LIBERO Parquet schema columns changed: {sorted(schema.names)}"
        )
    _validate_image_type(pa, schema.field("image"), name="image")
    _validate_image_type(
        pa,
        schema.field("wrist_image"),
        name="wrist_image",
    )
    state_dimension = _fixed_list_dimension(
        pa,
        schema.field("state"),
        name="state",
    )
    action_dimension = _fixed_list_dimension(
        pa,
        schema.field("actions"),
        name="actions",
    )
    if state_dimension != 8 or action_dimension != 7:
        raise ValueError(
            "LIBERO native dimensions changed: "
            f"state={state_dimension} action={action_dimension}"
        )
    _validate_null_statistics(parquet_file, path)

    columns = [
        "state",
        "actions",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = parquet_file.read(columns=columns, use_threads=False)
    state = np.asarray(table["state"].combine_chunks().to_pylist(), dtype=np.float32)
    action = np.asarray(
        table["actions"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    if state.shape != (expected_rows, 8) or action.shape != (expected_rows, 7):
        raise ValueError(f"LIBERO numeric shapes changed: path={path}")
    if validate_numeric_finite and (
        not np.isfinite(state).all() or not np.isfinite(action).all()
    ):
        raise ValueError(f"LIBERO state/action contains NaN/Inf: path={path}")

    def integers(column: str) -> np.ndarray:
        return np.asarray(table[column].combine_chunks().to_numpy())

    frame_index = integers("frame_index")
    episode_index = integers("episode_index")
    global_index = integers("index")
    task_index = integers("task_index")
    timestamp = np.asarray(
        table["timestamp"].combine_chunks().to_numpy(),
        dtype=np.float64,
    )
    if not np.array_equal(frame_index, np.arange(expected_rows)):
        raise ValueError(f"LIBERO frame_index is not contiguous: path={path}")
    if not np.all(episode_index == expected_episode_index):
        raise ValueError(f"LIBERO episode_index changed: path={path}")
    if not np.all(task_index == expected_task_index):
        raise ValueError(f"LIBERO task_index changed: path={path}")
    if not np.array_equal(
        global_index,
        np.arange(expected_global_start, expected_global_start + expected_rows),
    ):
        raise ValueError(f"LIBERO global index is not contiguous: path={path}")
    expected_timestamp = np.arange(expected_rows, dtype=np.float64) / fps
    if not np.allclose(timestamp, expected_timestamp, rtol=0.0, atol=2e-5):
        raise ValueError(f"LIBERO timestamp is not aligned to fps: path={path}")
    if not math.isfinite(float(state.min())) or not math.isfinite(float(action.min())):
        raise ValueError(f"LIBERO numeric extrema are not finite: path={path}")
    return {
        "state_min": state.min(axis=0).astype(float).tolist(),
        "state_max": state.max(axis=0).astype(float).tolist(),
        "action_min": action.min(axis=0).astype(float).tolist(),
        "action_max": action.max(axis=0).astype(float).tolist(),
        "row_groups": int(parquet_file.metadata.num_row_groups),
    }


def _combine_extrema(
    current: list[float] | None,
    values: list[float],
    *,
    minimum: bool,
) -> list[float]:
    if current is None:
        return list(values)
    function = min if minimum else max
    return [function(left, right) for left, right in zip(current, values)]


def build_libero_episode_catalog(
    data_root: str | Path,
    child: str,
    output_path: str | Path,
    *,
    catalog_id: str,
    data_version: str,
    storage_root: str | Path | None = None,
    validate_numeric_finite: bool = True,
    max_episodes: int | None = None,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build one immutable, split-neutral LIBERO suite Catalog."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = Path(output_path)
    metadata_output = _metadata_path(output)
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True, exist_ok=True)
    if not force and (output.exists() or metadata_output.exists()):
        raise FileExistsError(
            f"LIBERO Catalog output already exists: {output}"
        )
    spec = get_libero_child(child)
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "stats.json"
    for path in (info_path, tasks_path, episodes_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    normalization_contract = _normalization_contract(stats_path)
    tasks = _read_jsonl(tasks_path)
    episodes = _read_jsonl(episodes_path)
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("LIBERO info total_episodes differs from episodes.jsonl")
    if int(info.get("total_tasks", -1)) != len(tasks):
        raise ValueError("LIBERO info total_tasks differs from tasks.jsonl")
    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError("LIBERO fps must be positive")
    task_by_text: dict[str, int] = {}
    for position, row in enumerate(tasks):
        task_index = int(row.get("task_index", -1))
        task_text = str(row.get("task") or "").strip()
        if task_index != position or not task_text or task_text in task_by_text:
            raise ValueError("LIBERO tasks.jsonl is not unique and contiguous")
        task_by_text[task_text] = task_index

    global_starts = []
    cursor = 0
    for position, row in enumerate(episodes):
        if int(row.get("episode_index", -1)) != position:
            raise ValueError("LIBERO episodes.jsonl is not contiguous")
        global_starts.append(cursor)
        cursor += int(row.get("length", 0))
    if cursor != int(info.get("total_frames", -1)):
        raise ValueError("LIBERO total frame count differs from info.json")

    selected = episodes[spec.episode_start : spec.episode_stop]
    if len(selected) != spec.episode_count:
        raise ValueError(f"LIBERO {spec.name} Episode range is incomplete")
    if max_episodes is not None:
        if int(max_episodes) <= 0:
            raise ValueError("max_episodes must be positive")
        selected = selected[: int(max_episodes)]
    published_root = Path(storage_root) if storage_root is not None else root

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    started = time.monotonic()
    digest = hashlib.sha256()
    state_min = state_max = action_min = action_max = None
    total_frames = 0
    row_groups = 0
    task_counts: dict[str, int] = {}
    try:
        with temporary.open("xb") as handle:
            for position, episode in enumerate(selected, start=1):
                episode_index = int(episode["episode_index"])
                length = int(episode.get("length", 0))
                episode_tasks = episode.get("tasks")
                if (
                    length <= 0
                    or not isinstance(episode_tasks, list)
                    or len(episode_tasks) != 1
                ):
                    raise ValueError(
                        f"LIBERO Episode metadata invalid: index={episode_index}"
                    )
                task_text = str(episode_tasks[0]).strip()
                if task_text not in task_by_text:
                    raise ValueError(
                        f"LIBERO Episode task is unknown: index={episode_index}"
                    )
                task_index = task_by_text[task_text]
                if not spec.task_start <= task_index < spec.task_stop:
                    raise ValueError(
                        f"LIBERO task {task_index} is outside child={spec.name}"
                    )
                relative_path = _parquet_relative_path(info, episode_index)
                scan_path = root / relative_path
                if not scan_path.is_file():
                    raise FileNotFoundError(scan_path)
                quality = _validate_parquet_episode(
                    scan_path,
                    expected_episode_index=episode_index,
                    expected_task_index=task_index,
                    expected_rows=length,
                    expected_global_start=global_starts[episode_index],
                    fps=fps,
                    validate_numeric_finite=validate_numeric_finite,
                )
                state_min = _combine_extrema(
                    state_min, quality["state_min"], minimum=True
                )
                state_max = _combine_extrema(
                    state_max, quality["state_max"], minimum=False
                )
                action_min = _combine_extrema(
                    action_min, quality["action_min"], minimum=True
                )
                action_max = _combine_extrema(
                    action_max, quality["action_max"], minimum=False
                )
                row_groups += int(quality["row_groups"])
                total_frames += length
                task_counts[str(task_index)] = task_counts.get(str(task_index), 0) + 1

                storage = {
                    "type": "parquet",
                    "path": str(published_root / relative_path),
                }
                record = {
                    "episode_id": f"libero:{spec.name}:{episode_index:06d}",
                    "robot": "libero-panda",
                    "video": {
                        "count": length,
                        "storage": storage,
                        "meta": {
                            "views": ["image", "wrist_image"],
                            "default_view": "image",
                            "fps": fps,
                            "encoding": "png",
                            "native_size": [256, 256],
                            "index_column": "frame_index",
                        },
                    },
                    "state": {
                        "count": length,
                        "storage": storage,
                        "meta": {
                            "column": "state",
                            "dimension": 8,
                            "semantics": "absolute_eef_state",
                            "feature_names": [
                                "eef_position_x",
                                "eef_position_y",
                                "eef_position_z",
                                "eef_axis_angle_x",
                                "eef_axis_angle_y",
                                "eef_axis_angle_z",
                                "gripper_left_qpos",
                                "gripper_right_qpos",
                            ],
                            "normalization": "none",
                        },
                    },
                    "action": {
                        "count": length,
                        "storage": storage,
                        "meta": {
                            "column": "actions",
                            "dimension": 7,
                            "semantics": "delta_eef_command",
                            "feature_names": [
                                "delta_x",
                                "delta_y",
                                "delta_z",
                                "delta_axis_angle_x",
                                "delta_axis_angle_y",
                                "delta_axis_angle_z",
                                "gripper_command",
                            ],
                            "gripper_values": {
                                "open": -1.0,
                                "close": 1.0,
                            },
                            "is_control_command": True,
                            "normalization": "none",
                            "indices": "model_provided",
                        },
                    },
                    "texts": [
                        {
                            "index": 0,
                            "kind": "task_en",
                            "text": task_text,
                            "video_range": [0, length],
                            "meta": {"task_index": task_index},
                        }
                    ],
                    "meta": {
                        "suite": spec.name,
                        "original_episode_index": episode_index,
                        "task_index": task_index,
                        "fps": fps,
                        "alignment": "observation_t_then_action_t",
                        "source_format": "lerobot_v2_parquet",
                    },
                }
                validate_episode_record_v2(record)
                payload = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.write(payload)
                digest.update(payload)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "catalog_build",
                            "child": spec.name,
                            "processed": position,
                            "total": len(selected),
                            "percent": 100.0 * position / len(selected),
                            "frames": total_frames,
                            "elapsed_seconds": time.monotonic() - started,
                            "done": position == len(selected),
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())

        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": "libero",
            "child": spec.name,
            "episodes": len(selected),
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "split_neutral": True,
                "storage_root": str(published_root),
                "source_metadata_sha256": {
                    "info.json": _file_sha256(info_path),
                    "tasks.jsonl": _file_sha256(tasks_path),
                    "episodes.jsonl": _file_sha256(episodes_path),
                    "stats.json": _file_sha256(stats_path),
                },
                "normalization": normalization_contract,
                "build": {
                    "frames": total_frames,
                    "fps": fps,
                    "row_groups": row_groups,
                    "task_episode_counts": task_counts,
                    "state_min": state_min,
                    "state_max": state_max,
                    "action_min": action_min,
                    "action_max": action_max,
                    "validate_numeric_finite": bool(validate_numeric_finite),
                    "elapsed_seconds": time.monotonic() - started,
                },
            },
        }
        validate_catalog_metadata_v2(metadata)
        os.replace(temporary, output)
        _atomic_json(metadata_output, metadata)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "LIBERO_CHILDREN",
    "LiberoChildSpec",
    "build_libero_episode_catalog",
    "get_libero_child",
]
