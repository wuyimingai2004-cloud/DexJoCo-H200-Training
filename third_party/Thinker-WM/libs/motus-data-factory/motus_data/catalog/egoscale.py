"""Build deterministic, split-neutral EGOSCALE Episode Catalog shards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from motus_data.egoscale import RAW_MODALITY_SPECS
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_record_v2,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]

EGOSCALE_EPISODE_PATTERN = re.compile(
    r"^(?P<group>session_[A-Za-z0-9]+_\d{5}_task_\d{4})" r"(?:_\d+_\d+)?_lerobot_v3$"
)
EGOSCALE_REQUIRED_PATHS = (
    "source_manifest.json",
    "calibration/undistorted_intrinsics.json",
    "calibration/stereo_calibration.json",
    "meta/tasks.parquet",
    "meta/episodes.parquet",
    "meta/custom_metadata.csv",
    "meta/custom_annotation.json",
    "meta/info.json",
    "data/chunk-000/episode_000000.parquet",
    "videos/base_0_camera/rgb/image/chunk-000/episode_000000.mp4",
    "videos/base_1_camera/rgb/image/chunk-000/episode_000000.mp4",
)
EGOSCALE_PARQUET_COLUMNS = tuple(RAW_MODALITY_SPECS)
EGOSCALE_CATALOG_CHECKPOINT_FORMAT = "motus_egoscale_catalog_checkpoint_v1"


def _column_order_fingerprint(columns: Sequence[str]) -> str:
    payload = json.dumps(list(columns), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "EGOSCALE Catalog building requires the optional 'egoscale' "
            "dependency: install pyarrow"
        ) from error
    return pa, pq


def egoscale_source_group_key(episode_name: str) -> str:
    match = EGOSCALE_EPISODE_PATTERN.fullmatch(str(episode_name))
    if match is None:
        raise ValueError(f"unexpected EGOSCALE Episode directory name={episode_name!r}")
    return str(match.group("group"))


def egoscale_shard_index(source_group_key: str, shard_count: int) -> int:
    if isinstance(shard_count, bool) or int(shard_count) <= 0:
        raise ValueError("shard_count must be a positive integer")
    digest = hashlib.sha256(str(source_group_key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % int(shard_count)


def _normalize_shard_indices(
    *,
    shard_count: int,
    shard_index: int | None,
    shard_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if isinstance(shard_count, bool) or int(shard_count) <= 0:
        raise ValueError("shard_count must be a positive integer")
    if shard_indices is None:
        if shard_index is None:
            raise ValueError("exactly one shard selection must be provided")
        values = (shard_index,)
    else:
        if shard_index is not None:
            raise ValueError("shard_index and shard_indices are mutually exclusive")
        if isinstance(shard_indices, (str, bytes)):
            raise TypeError("shard_indices must be a sequence of integers")
        values = tuple(shard_indices)
        if not values:
            raise ValueError("shard_indices must be non-empty")

    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("shard indices must be integers")
        index = int(value)
        if not 0 <= index < int(shard_count):
            raise ValueError("shard indices must satisfy 0 <= index < shard_count")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("shard_indices must not contain duplicates")
    return tuple(sorted(normalized))


def select_egoscale_episode_dirs(
    root: str | Path,
    *,
    shard_count: int,
    shard_index: int | None = None,
    shard_indices: Sequence[int] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    data_root = Path(root)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    selected_indices = _normalize_shard_indices(
        shard_count=shard_count,
        shard_index=shard_index,
        shard_indices=shard_indices,
    )
    selected_index_set = set(selected_indices)
    selected = []
    all_groups = set()
    selected_groups = set()
    bucket_episodes = Counter()
    bucket_groups: dict[int, set[str]] = {
        index: set() for index in range(int(shard_count))
    }
    invalid_names = []
    episode_dirs = 0
    for entry in os.scandir(data_root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        episode_dirs += 1
        try:
            group = egoscale_source_group_key(entry.name)
        except ValueError:
            invalid_names.append(entry.name)
            continue
        bucket = egoscale_shard_index(group, shard_count)
        all_groups.add(group)
        bucket_groups[bucket].add(group)
        bucket_episodes[bucket] += 1
        if bucket in selected_index_set:
            selected.append(Path(entry.path))
            selected_groups.add(group)
    if invalid_names:
        raise ValueError(
            f"EGOSCALE root contains unexpected Episode names={invalid_names[:5]}"
        )
    selected.sort(key=lambda path: path.name)
    selection = {
        "algorithm": ("uint64_be(sha256(source_group_key)[:8]) % shard_count"),
        "shard_count": int(shard_count),
        "selection_mode": (
            "single_hash_bucket" if len(selected_indices) == 1 else "hash_bucket_union"
        ),
        "selected_shard_indices": list(selected_indices),
        "all_episode_dirs": episode_dirs,
        "all_source_groups": len(all_groups),
        "selected_episode_dirs": len(selected),
        "selected_source_groups": len(selected_groups),
        "bucket_episode_counts": {
            str(index): int(bucket_episodes[index]) for index in range(int(shard_count))
        },
        "bucket_group_counts": {
            str(index): len(bucket_groups[index]) for index in range(int(shard_count))
        },
    }
    if len(selected_indices) == 1:
        selection["shard_index"] = selected_indices[0]
    return selected, selection


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read EGOSCALE JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"EGOSCALE JSON must contain an object: {path}")
    return value


def _metadata_path(catalog: Path) -> Path:
    return catalog.with_suffix(".meta.json")


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


def _catalog_resume_paths(output: Path) -> tuple[Path, Path]:
    partial = output.with_name(f".{output.name}.partial")
    checkpoint = output.with_name(f".{output.name}.checkpoint.json")
    return partial, checkpoint


def _episode_selection_fingerprint(selected: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for episode in selected:
        digest.update(episode.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _restore_partial_digest(
    partial: Path,
    *,
    committed_bytes: int,
    committed_records: int,
    expected_sha256: str,
) -> Any:
    if int(committed_bytes) < 0 or int(committed_records) < 0:
        raise RuntimeError("EGOSCALE resume checkpoint has negative progress")
    if len(str(expected_sha256)) != 64:
        raise RuntimeError("EGOSCALE resume checkpoint SHA256 is invalid")
    if not partial.is_file():
        raise RuntimeError(f"EGOSCALE resume partial is missing: {partial}")
    actual_size = partial.stat().st_size
    if actual_size < int(committed_bytes):
        raise RuntimeError(
            "EGOSCALE resume partial is shorter than checkpoint: "
            f"actual={actual_size} checkpoint={committed_bytes}"
        )
    if actual_size > int(committed_bytes):
        with partial.open("r+b") as handle:
            handle.truncate(int(committed_bytes))
            handle.flush()
            os.fsync(handle.fileno())

    digest = hashlib.sha256()
    records = 0
    last_byte = b""
    with partial.open("rb") as handle:
        while True:
            payload = handle.read(8 * 1024 * 1024)
            if not payload:
                break
            digest.update(payload)
            records += payload.count(b"\n")
            last_byte = payload[-1:]
    if committed_bytes and last_byte != b"\n":
        raise RuntimeError("EGOSCALE resume checkpoint ends inside a JSONL record")
    if records != int(committed_records):
        raise RuntimeError(
            "EGOSCALE resume record count mismatch: "
            f"partial={records} checkpoint={committed_records}"
        )
    if digest.hexdigest() != str(expected_sha256):
        raise RuntimeError("EGOSCALE resume partial SHA256 differs from checkpoint")
    return digest


def _task_texts(
    case: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    frame_count: int,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = case.get("tasks") or []
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("EGOSCALE source manifest must contain exactly one task")
    task = tasks[0]
    if not isinstance(task, Mapping):
        raise ValueError("EGOSCALE task must be an object")
    source = task.get("source") or {}
    if not isinstance(source, Mapping):
        raise ValueError("EGOSCALE task source must be an object")

    texts: list[dict[str, Any]] = []

    def append_text(kind: str, text: Any, metadata: Mapping[str, Any]) -> None:
        value = str(text or "").strip()
        if not value:
            return
        texts.append(
            {
                "index": len(texts),
                "kind": kind,
                "text": value,
                "video_range": [0, frame_count],
                "meta": dict(metadata),
            }
        )

    common_meta = {
        "task_index": int(task.get("task_index", 0)),
        "task_id": str(task.get("task_id") or ""),
        "task_usable": bool(task.get("task_usable", True)),
    }
    append_text("task_name_en", task.get("task_name"), common_meta)
    append_text(
        "task_description_en",
        task.get("task_description"),
        common_meta,
    )
    detail = str(source.get("detail") or "").strip()
    description = str(task.get("task_description") or "").strip()
    if detail and detail != description:
        append_text("task_detail_en", detail, common_meta)
    append_text(
        "task_label",
        source.get("task_label"),
        {"task_index": common_meta["task_index"]},
    )

    annotation_episodes = annotation.get("episodes") or []
    if not isinstance(annotation_episodes, list) or len(annotation_episodes) != 1:
        raise ValueError("EGOSCALE custom annotation must contain exactly one Episode")
    annotation_episode = annotation_episodes[0]
    extras = annotation_episode.get("extras") or {}
    actions = extras.get("actions") or []
    if not isinstance(actions, list):
        raise ValueError("EGOSCALE annotation extras.actions must be a list")
    covered = set()
    for action_position, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise ValueError("EGOSCALE fine action must be an object")
        start = int(action["start_frame"])
        end = int(action["end_frame"])
        if not 0 <= start < end <= frame_count:
            raise ValueError(f"EGOSCALE fine action range [{start}, {end}) is invalid")
        if not math.isclose(
            float(action.get("start_seconds", start / fps)),
            start / fps,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(action.get("end_seconds", end / fps)),
            end / fps,
            abs_tol=1e-6,
        ):
            raise ValueError("EGOSCALE fine action seconds differ from frames")
        covered.update(range(start, end))
        description = str(action.get("description") or "").strip()
        label = str(action.get("action") or "").strip()
        text = description or label
        if not text:
            raise ValueError("EGOSCALE fine action text is empty")
        texts.append(
            {
                "index": len(texts),
                "kind": "fine_action_en",
                "text": text,
                "video_range": [start, end],
                "meta": {
                    "action_index": int(action.get("action_index", action_position)),
                    "action": label,
                    "hand": str(action.get("hand") or ""),
                    "scene_semantic": str(action.get("scene_semantic") or ""),
                    "camera_motion": str(action.get("camera_motion") or ""),
                },
            }
        )
    if not texts:
        raise ValueError("EGOSCALE Episode has no usable text")
    coverage = len(covered) / frame_count if frame_count else 0.0
    return texts, {
        "fine_action_count": len(actions),
        "fine_annotation_coverage": coverage,
        "annotation_episode_id": str(annotation_episode.get("episode_id") or ""),
    }


def _arrow_type_contains_null(pa: Any, value_type: Any) -> bool:
    if pa.types.is_null(value_type):
        return True
    if (
        pa.types.is_list(value_type)
        or pa.types.is_large_list(value_type)
        or pa.types.is_fixed_size_list(value_type)
    ):
        return _arrow_type_contains_null(pa, value_type.value_type)
    if pa.types.is_struct(value_type):
        return any(
            _arrow_type_contains_null(pa, field.type) for field in value_type
        )
    return False


def _episode_record(
    episode: Path,
    *,
    published_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pa, pq = _require_pyarrow()
    for relative in EGOSCALE_REQUIRED_PATHS:
        path = episode / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    source_manifest = _read_json(episode / "source_manifest.json")
    info = _read_json(episode / "meta/info.json")
    annotation = _read_json(episode / "meta/custom_annotation.json")
    cases = source_manifest.get("cases") or []
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("EGOSCALE source manifest must contain one case")
    case = cases[0]
    if not isinstance(case, Mapping):
        raise ValueError("EGOSCALE source case must be an object")
    frame_count = int(case.get("frame_count", 0))
    fps = float(case.get("fps", 0))
    duration = float(case.get("duration", 0))
    if frame_count <= 0 or fps <= 0:
        raise ValueError("EGOSCALE frame_count and fps must be positive")
    if not math.isclose(duration, frame_count / fps, abs_tol=1e-6):
        raise ValueError("EGOSCALE duration differs from frame_count/fps")
    if (
        str(info.get("codebase_version")) != "v3.0"
        or str(info.get("robot_type")) != "orb_stereo"
        or float(info.get("fps", 0)) != fps
    ):
        raise ValueError("EGOSCALE info.json identity fields changed")

    parquet_relative = Path("data/chunk-000/episode_000000.parquet")
    parquet_path = episode / parquet_relative
    parquet_file = pq.ParquetFile(parquet_path)
    if int(parquet_file.metadata.num_rows) != frame_count:
        raise ValueError("EGOSCALE Parquet row count differs from frame count")
    schema = parquet_file.schema_arrow
    parquet_columns = tuple(schema.names)
    if len(parquet_columns) != len(EGOSCALE_PARQUET_COLUMNS) or set(
        parquet_columns
    ) != set(EGOSCALE_PARQUET_COLUMNS):
        missing = sorted(set(EGOSCALE_PARQUET_COLUMNS).difference(parquet_columns))
        unexpected = sorted(set(parquet_columns).difference(EGOSCALE_PARQUET_COLUMNS))
        raise ValueError(
            "EGOSCALE Parquet columns changed: "
            f"missing={missing} unexpected={unexpected} "
            f"actual_count={len(parquet_columns)}"
        )
    column_order_matches = parquet_columns == EGOSCALE_PARQUET_COLUMNS
    column_order_fingerprint = _column_order_fingerprint(parquet_columns)
    null_columns = [
        field.name
        for field in schema
        if _arrow_type_contains_null(pa, field.type)
    ]

    tasks_table = pq.read_table(episode / "meta/tasks.parquet")
    episodes_table = pq.read_table(episode / "meta/episodes.parquet")
    if tasks_table.num_rows != 1 or episodes_table.num_rows != 1:
        raise ValueError("EGOSCALE task/Episode metadata must contain one row")
    task_row = tasks_table.to_pylist()[0]
    episode_row = episodes_table.to_pylist()[0]
    if int(task_row.get("task_index", -1)) != 0:
        raise ValueError("EGOSCALE task_index must be zero inside each package")
    if int(episode_row.get("episode_index", -1)) != 0:
        raise ValueError("EGOSCALE episode_index must be zero inside each package")

    texts, annotation_stats = _task_texts(
        case,
        annotation,
        frame_count=frame_count,
        fps=fps,
    )
    source_group = egoscale_source_group_key(episode.name)
    allowed_annotation_ids = {episode.name, f"{source_group}_lerobot_v3"}
    annotation_id = annotation_stats["annotation_episode_id"]
    if annotation_id not in allowed_annotation_ids:
        raise ValueError("EGOSCALE annotation Episode id does not match package")

    published_episode = published_root / episode.name
    parquet_storage = {
        "type": "parquet",
        "path": str(published_episode / parquet_relative),
    }
    view_paths = {
        "base_0": str(
            published_episode
            / "videos/base_0_camera/rgb/image/chunk-000/episode_000000.mp4"
        ),
        "base_1": str(
            published_episode
            / "videos/base_1_camera/rgb/image/chunk-000/episode_000000.mp4"
        ),
    }
    task = (case.get("tasks") or [{}])[0]
    source_task = task.get("source") or {}
    source_start = source_task.get("source_start_frame")
    source_end = source_task.get("source_end_frame")
    record = {
        "episode_id": f"egoscale:{episode.name}",
        "robot": "egoscale-orb-stereo-human",
        "video": {
            "count": frame_count,
            "storage": {"type": "mp4", "path": view_paths["base_0"]},
            "meta": {
                "views": ["base_0", "base_1"],
                "default_view": "base_0",
                "view_paths": view_paths,
                "fps": fps,
                "native_size": [1080, 1920],
                "encoding": "mp4",
            },
        },
        "state": {
            "count": frame_count,
            "storage": parquet_storage,
            "meta": {
                "dimension": 24,
                "default_profile": "camera_pose_with_validity",
                "supported_profiles": [
                    "camera_pose_with_validity",
                    "world_pose_with_validity",
                ],
                "normalization": "none",
                "semantics": "observed_human_pose",
            },
        },
        "action": {
            "count": frame_count,
            "storage": parquet_storage,
            "meta": {
                "dimension": 24,
                "default_profile": "camera_pose_with_validity",
                "supported_profiles": [
                    "camera_pose_with_validity",
                    "world_pose_with_validity",
                ],
                "normalization": "none",
                "semantics": "observed_human_motion_proxy",
                "source_signal": "same_aligned_observed_pose_as_state",
                "is_control_command": False,
                "indices": "model_provided",
            },
        },
        "texts": texts,
        "meta": {
            "source_format": "lerobot_v3_parquet_mp4",
            "source_episode_dir": episode.name,
            "source_group_key": source_group,
            "source_data_type": str(source_manifest.get("data_type") or ""),
            "source_frame_range": (
                [int(source_start), int(source_end)]
                if source_start is not None and source_end is not None
                else None
            ),
            "alignment": "all_frame_modalities_share_parquet_frame_index",
            "fps": fps,
            "duration_seconds": duration,
            "task_usable": bool(task.get("task_usable", True)),
            "calibration": {
                "intrinsics": str(
                    published_episode / "calibration/undistorted_intrinsics.json"
                ),
                "stereo": str(
                    published_episode / "calibration/stereo_calibration.json"
                ),
            },
            "raw_parquet_columns": list(parquet_columns),
            "quality": {
                "null_value_columns": null_columns,
                "parquet_column_order_matches_canonical": column_order_matches,
                "parquet_column_order_fingerprint": column_order_fingerprint,
                "fine_action_count": annotation_stats["fine_action_count"],
                "fine_annotation_coverage": annotation_stats[
                    "fine_annotation_coverage"
                ],
                "annotation_uses_source_group_id": annotation_id != episode.name,
            },
        },
    }
    validate_episode_record_v2(record)
    return record, {
        "frames": frame_count,
        "texts": len(texts),
        "fine_actions": annotation_stats["fine_action_count"],
        "annotation_coverage": annotation_stats["fine_annotation_coverage"],
        "null_value_columns": null_columns,
        "parquet_column_order_matches_canonical": column_order_matches,
        "parquet_column_order_fingerprint": column_order_fingerprint,
        "annotation_uses_source_group_id": annotation_id != episode.name,
        "task_usable": bool(task.get("task_usable", True)),
    }


def build_egoscale_episode_catalog(
    data_root: str | Path,
    output_path: str | Path,
    *,
    catalog_id: str,
    data_version: str,
    child: str = "egoscale_first500h",
    shard_count: int = 10,
    shard_index: int | None = 0,
    shard_indices: Sequence[int] | None = None,
    storage_root: str | Path | None = None,
    max_workers: int = 1,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
    checkpoint_interval_episodes: int = 500,
    force: bool = False,
) -> dict[str, Any]:
    """Build one immutable EGOSCALE source-group hash-bucket selection."""

    root = Path(data_root)
    output = Path(output_path)
    metadata_output = _metadata_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not force and (output.exists() or metadata_output.exists()):
        raise FileExistsError(f"EGOSCALE Catalog output exists: {output}")
    published_root = Path(storage_root) if storage_root is not None else root
    selected, selection = select_egoscale_episode_dirs(
        root,
        shard_count=shard_count,
        shard_index=shard_index,
        shard_indices=shard_indices,
    )
    if not selected:
        raise ValueError("EGOSCALE shard selected no Episodes")
    if isinstance(max_workers, bool) or int(max_workers) <= 0:
        raise ValueError("max_workers must be a positive integer")
    if (
        isinstance(checkpoint_interval_episodes, bool)
        or int(checkpoint_interval_episodes) <= 0
    ):
        raise ValueError("checkpoint_interval_episodes must be a positive integer")

    partial_path, checkpoint_path = _catalog_resume_paths(output)
    temporary = (
        partial_path
        if resume
        else output.with_name(f".{output.name}.tmp.{os.getpid()}")
    )
    if force:
        output.unlink(missing_ok=True)
        metadata_output.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    elif not resume:
        if partial_path.exists() or checkpoint_path.exists():
            raise RuntimeError(
                "EGOSCALE resumable build artifacts exist; rerun with "
                "resume=True or explicitly discard them with force=True"
            )
        temporary.unlink(missing_ok=True)

    selection_identity = {
        "format": EGOSCALE_CATALOG_CHECKPOINT_FORMAT,
        "catalog_id": str(catalog_id),
        "data_version": str(data_version),
        "child": str(child),
        "data_root": str(root.resolve()),
        "storage_root": str(published_root.resolve()),
        "selected_episode_count": len(selected),
        "selected_episode_fingerprint": _episode_selection_fingerprint(selected),
        "canonical_parquet_columns_fingerprint": _column_order_fingerprint(
            EGOSCALE_PARQUET_COLUMNS
        ),
    }
    started = time.monotonic()
    digest = hashlib.sha256()
    frames = texts = fine_actions = 0
    annotation_coverage_sum = 0.0
    null_column_counts = Counter()
    column_order_counts = Counter()
    warning_counts = Counter()
    unusable_tasks = 0
    completed = 0
    elapsed_before_resume = 0.0
    resume_count = 0
    resumed_from = 0
    resumed = False
    published = False
    executor = None

    if resume and checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path)
        actual_identity = checkpoint.get("identity")
        if actual_identity != selection_identity:
            raise RuntimeError(
                "EGOSCALE resume checkpoint does not match this Catalog/input selection"
            )
        completed = int(checkpoint.get("completed_episodes", -1))
        if not 0 <= completed <= len(selected):
            raise RuntimeError("EGOSCALE resume checkpoint completed count is invalid")
        digest = _restore_partial_digest(
            partial_path,
            committed_bytes=int(checkpoint.get("partial_bytes", -1)),
            committed_records=completed,
            expected_sha256=str(checkpoint.get("partial_sha256", "")),
        )
        frames = int(checkpoint.get("frames", 0))
        texts = int(checkpoint.get("texts", 0))
        fine_actions = int(checkpoint.get("fine_actions", 0))
        annotation_coverage_sum = float(
            checkpoint.get("annotation_coverage_sum", 0.0)
        )
        null_column_counts.update(checkpoint.get("null_column_counts") or {})
        column_order_counts.update(checkpoint.get("column_order_counts") or {})
        warning_counts.update(checkpoint.get("warning_counts") or {})
        unusable_tasks = int(checkpoint.get("unusable_tasks", 0))
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        resume_count = int(checkpoint.get("resume_count", 0)) + 1
        resumed_from = completed
        resumed = True
    elif resume and partial_path.exists():
        # A crash before the first durable checkpoint cannot be resumed safely.
        partial_path.unlink()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.monotonic() - started

    def write_checkpoint(handle: Any, position: int) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        _atomic_json(
            checkpoint_path,
            {
                "identity": selection_identity,
                "completed_episodes": int(position),
                "partial_bytes": int(handle.tell()),
                "partial_sha256": digest.hexdigest(),
                "frames": frames,
                "texts": texts,
                "fine_actions": fine_actions,
                "annotation_coverage_sum": annotation_coverage_sum,
                "null_column_counts": dict(sorted(null_column_counts.items())),
                "column_order_counts": dict(sorted(column_order_counts.items())),
                "warning_counts": dict(sorted(warning_counts.items())),
                "unusable_tasks": unusable_tasks,
                "elapsed_seconds": elapsed_seconds(),
                "resume_count": resume_count,
            },
        )

    try:
        build_record = partial(_episode_record, published_root=published_root)
        if int(max_workers) == 1:
            records = map(build_record, selected[completed:])
        else:
            executor = ThreadPoolExecutor(
                max_workers=int(max_workers),
                thread_name_prefix="egoscale_catalog",
            )
            records = executor.map(build_record, selected[completed:])
        mode = "ab" if resumed else "xb"
        with temporary.open(mode) as handle:
            if resume and not resumed:
                write_checkpoint(handle, 0)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "catalog_build",
                        "processed": completed,
                        "total": len(selected),
                        "percent": 100.0 * completed / len(selected),
                        "frames": frames,
                        "elapsed_seconds": elapsed_seconds(),
                        "resumed": resumed,
                        "resumed_from": resumed_from,
                        "done": completed == len(selected),
                    }
                )
            for position, (record, statistics) in enumerate(
                records, start=completed + 1
            ):
                frames += int(statistics["frames"])
                texts += int(statistics["texts"])
                fine_actions += int(statistics["fine_actions"])
                annotation_coverage_sum += float(statistics["annotation_coverage"])
                for column in statistics["null_value_columns"]:
                    null_column_counts[str(column)] += 1
                column_order_counts[
                    str(statistics["parquet_column_order_fingerprint"])
                ] += 1
                if not statistics["parquet_column_order_matches_canonical"]:
                    warning_counts["parquet_column_order_changed"] += 1
                if statistics["annotation_uses_source_group_id"]:
                    warning_counts["annotation_uses_source_group_id"] += 1
                if not statistics["task_usable"]:
                    unusable_tasks += 1
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
                if resume and (
                    position % int(checkpoint_interval_episodes) == 0
                    or position == len(selected)
                ):
                    write_checkpoint(handle, position)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "catalog_build",
                            "processed": position,
                            "total": len(selected),
                            "percent": 100.0 * position / len(selected),
                            "frames": frames,
                            "elapsed_seconds": elapsed_seconds(),
                            "resumed": resumed,
                            "resumed_from": resumed_from,
                            "done": position == len(selected),
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())

        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": "egoscale",
            "child": str(child),
            "episodes": len(selected),
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "split_neutral": True,
                "storage_root": str(published_root),
                "sharding": selection,
                "normalization": {
                    "default_mode": "raw",
                    "supported_modes": ["raw"],
                },
                "default_reader_options": {
                    "camera_mode": "base_0",
                    "pose_profile": "camera_pose_with_validity",
                    "resize_mode": "pad",
                },
                "raw_modalities": {
                    name: {"dtype": dtype, "shape": list(shape)}
                    for name, (dtype, shape) in RAW_MODALITY_SPECS.items()
                },
                "build": {
                    "frames": frames,
                    "texts": texts,
                    "fine_actions": fine_actions,
                    "mean_fine_annotation_coverage": (
                        annotation_coverage_sum / len(selected)
                    ),
                    "null_value_column_episode_counts": dict(
                        sorted(null_column_counts.items())
                    ),
                    "canonical_parquet_column_order_fingerprint": (
                        _column_order_fingerprint(EGOSCALE_PARQUET_COLUMNS)
                    ),
                    "parquet_column_order_fingerprint_counts": dict(
                        sorted(column_order_counts.items())
                    ),
                    "warning_counts": dict(sorted(warning_counts.items())),
                    "task_usable_false_episodes": unusable_tasks,
                    "max_workers": int(max_workers),
                    "elapsed_seconds": elapsed_seconds(),
                    "resumable_build": bool(resume),
                    "checkpoint_interval_episodes": (
                        int(checkpoint_interval_episodes) if resume else None
                    ),
                    "resumed": resumed,
                    "resumed_from_episode": resumed_from,
                    "resume_count": resume_count,
                },
            },
        }
        validate_catalog_metadata_v2(metadata)
        if force:
            output.unlink(missing_ok=True)
            metadata_output.unlink(missing_ok=True)
        os.replace(temporary, output)
        _atomic_json(metadata_output, metadata)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published = True
        checkpoint_path.unlink(missing_ok=True)
        return metadata
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if not resume or published:
            temporary.unlink(missing_ok=True)


__all__ = [
    "EGOSCALE_EPISODE_PATTERN",
    "EGOSCALE_PARQUET_COLUMNS",
    "EGOSCALE_REQUIRED_PATHS",
    "build_egoscale_episode_catalog",
    "egoscale_shard_index",
    "egoscale_source_group_key",
    "select_egoscale_episode_dirs",
]
