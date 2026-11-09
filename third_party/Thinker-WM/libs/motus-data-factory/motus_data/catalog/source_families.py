"""Offline Episode Catalog builders for RoboTwin and RobotMind families."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None

from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)
from motus_data.iso_bmff import read_video_sample_count


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _metadata_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".meta.json")
    return Path(str(catalog_path) + ".meta.json")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _source_path(row: Mapping[str, Any], modality: str) -> str:
    node = _require_mapping(row.get(modality), modality)
    path = str(node.get("uri") or "").strip()
    if not path:
        raise ValueError(f"{modality}.uri is missing")
    return path


def _episode_key(row: Mapping[str, Any], family: str) -> tuple[str, str]:
    if family == "robotwin":
        video_path = _source_path(row, "video")
        state_path = _source_path(row, "state")
        return video_path, state_path
    hdf5_path = _source_path(row, "video")
    if hdf5_path != _source_path(row, "state"):
        raise ValueError("RobotMind video/state must share one HDF5 path")
    if hdf5_path != _source_path(row, "action"):
        raise ValueError("RobotMind video/action must share one HDF5 path")
    return hdf5_path, hdf5_path


def _safe_component(value: Any, fallback: str) -> str:
    text = str(value or "").strip().replace(":", "_")
    return text or fallback


def _episode_id(row: Mapping[str, Any], family: str) -> str:
    metadata = _require_mapping(row.get("metadata") or {}, "metadata")
    robot = _require_mapping(row.get("robot"), "robot")
    robot_name = _safe_component(robot.get("name"), "unknown_robot")
    task = _safe_component(row.get("task"), "unknown_task")
    if family == "robotwin":
        split_name = _safe_component(metadata.get("data_split"), "unknown_split")
        episode = _safe_component(metadata.get("episode_name"), "unknown_episode")
        return f"robotwin:{robot_name}:{split_name}:{task}:{episode}"
    episode = _safe_component(
        metadata.get("episode_name")
        or metadata.get("episode_folder")
        or metadata.get("episode_id"),
        "unknown_episode",
    )
    return f"robotmind:{robot_name}:{task}:{episode}"


def _robotwin_texts(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = _require_mapping(row.get("metadata") or {}, "metadata")
    instructions = []
    meta_path = Path(str(metadata.get("meta_path") or ""))
    if meta_path.is_file():
        instructions = [
            line.strip()
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    text = _require_mapping(row.get("text"), "text")
    if not instructions:
        fallback = str(text.get("target") or text.get("t5") or "").strip()
        if fallback:
            instructions = [fallback]
    if not instructions:
        raise ValueError("RoboTwin Episode has no usable instruction")
    return [
        {
            "index": index,
            "kind": "instruction_en",
            "text": instruction,
            "meta": {"source_instruction_index": index},
        }
        for index, instruction in enumerate(instructions)
    ]


def _robotmind_texts(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = _require_mapping(row.get("text"), "text")
    en = str(text.get("en") or text.get("target") or "").strip()
    zh = str(text.get("zh") or "").strip()
    items = []
    if en:
        items.append(
            {
                "index": len(items),
                "kind": "instruction_en",
                "text": en,
            }
        )
    if zh and zh != en:
        items.append(
            {
                "index": len(items),
                "kind": "instruction_zh",
                "text": zh,
            }
        )
    if not items:
        fallback = str(text.get("t5") or "").strip()
        if fallback:
            items.append(
                {
                    "index": 0,
                    "kind": "instruction",
                    "text": fallback,
                }
            )
    if not items:
        raise ValueError("RobotMind Episode has no usable instruction")
    return items


def _load_qpos(path: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, Mapping):
        value = value.get("qpos")
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError("RoboTwin qpos must be a rank-2 tensor")
    return value.detach().to(dtype=torch.float32, device="cpu")


def _mp4_count(path: str) -> int:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        return int(read_video_sample_count(handle, file_path.stat().st_size))


def _robotwin_record(
    row: Mapping[str, Any],
    *,
    verify_source: bool,
) -> dict[str, Any]:
    video_path = _source_path(row, "video")
    qpos_path = _source_path(row, "state")
    if qpos_path != _source_path(row, "action"):
        raise ValueError("RoboTwin state/action paths differ")
    robot = _require_mapping(row.get("robot"), "robot")
    metadata = _require_mapping(row.get("metadata") or {}, "metadata")
    declared_count = int(metadata.get("effective_total_frames", 0))
    declared_dim = int(robot.get("state_dim", 0))
    if declared_count <= 0 or declared_dim <= 0:
        raise ValueError("RoboTwin Manifest lacks count or dimension")
    if int(robot.get("action_dim", -1)) != declared_dim:
        raise ValueError("RoboTwin state/action dimensions differ")
    if verify_source:
        qpos = _load_qpos(qpos_path)
        video_count = _mp4_count(video_path)
        if int(qpos.shape[1]) != declared_dim:
            raise ValueError(
                f"RoboTwin qpos dim={qpos.shape[1]} expected={declared_dim}"
            )
        count = min(video_count, int(qpos.shape[0]))
        if count != declared_count:
            raise ValueError(
                "RoboTwin effective count changed: "
                f"raw={count} manifest={declared_count}"
            )
    else:
        count = declared_count
    record = {
        "episode_id": _episode_id(row, "robotwin"),
        "robot": str(robot["name"]),
        "video": {
            "count": count,
            "storage": {"type": "file", "path": video_path},
            "meta": {"count_source": "verified_mp4_sample_table"},
        },
        "state": {
            "count": count,
            "storage": {"type": "file", "path": qpos_path},
            "meta": {"dimension": declared_dim, "key": "qpos"},
        },
        "action": {
            "count": count,
            "storage": {"type": "file", "path": qpos_path},
            "meta": {"dimension": declared_dim, "key": "qpos"},
        },
        "texts": _robotwin_texts(row),
        "meta": {
            "task_name": str(row.get("task") or ""),
            "data_split": str(metadata.get("data_split") or ""),
            "episode_name": str(metadata.get("episode_name") or ""),
            "dataset_root": str(metadata.get("dataset_root") or ""),
            "state_action_dim": declared_dim,
        },
    }
    validate_episode_record_v2(record)
    return record


def _ordered_robotmind_keys(row: Mapping[str, Any]) -> list[str]:
    state = _require_mapping(row.get("state"), "state")
    keys = _require_mapping(state.get("keys"), "state.keys")
    metadata = _require_mapping(row.get("metadata") or {}, "metadata")
    order = metadata.get("state_action_keys_order")
    if order is None:
        order = list(keys)
    ordered_names = [str(value) for value in order]
    missing = [name for name in ordered_names if name not in keys]
    if missing:
        raise ValueError(
            f"RobotMind state_action_keys_order references missing keys={missing}"
        )
    if len(ordered_names) != len(keys):
        extra = sorted(set(keys) - set(ordered_names))
        raise ValueError(f"RobotMind unordered state/action keys={extra}")
    return [str(keys[name]) for name in ordered_names]


def _robotmind_raw_layout(
    path: str,
    video_key: str,
    numeric_keys: list[str],
) -> tuple[int, int]:
    if h5py is None:
        raise ImportError("RobotMind Catalog construction requires h5py")
    required = [video_key, *numeric_keys]
    with h5py.File(path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"RobotMind HDF5 missing keys={missing}")
        counts = [int(handle[key].shape[0]) for key in required]
        if "metadata" in handle:
            trajectory_length = handle["metadata"].attrs.get(
                "trajectory_length"
            )
            if trajectory_length is not None:
                counts.append(int(trajectory_length))
        if len(set(counts)) != 1:
            raise ValueError(
                f"RobotMind modalities are not exactly aligned: counts={counts}"
            )
        dimensions = [
            int(np.prod(handle[key].shape[1:]))
            if len(handle[key].shape) > 1
            else 1
            for key in numeric_keys
        ]
    return counts[0], sum(dimensions)


def _robotmind_record(
    row: Mapping[str, Any],
    *,
    verify_source: bool,
) -> dict[str, Any]:
    path = _source_path(row, "video")
    video = _require_mapping(row.get("video"), "video")
    video_key = str(video.get("key") or "").strip()
    if not video_key:
        raise ValueError("RobotMind video key is missing")
    numeric_keys = _ordered_robotmind_keys(row)
    robot = _require_mapping(row.get("robot"), "robot")
    metadata = _require_mapping(row.get("metadata") or {}, "metadata")
    declared_count = int(metadata.get("total_frames", 0))
    declared_dim = int(robot.get("state_dim", 0))
    if declared_count <= 0 or declared_dim <= 0:
        raise ValueError("RobotMind Manifest lacks count or dimension")
    if int(robot.get("action_dim", -1)) != declared_dim:
        raise ValueError("RobotMind state/action dimensions differ")
    if verify_source:
        count, dimension = _robotmind_raw_layout(
            path, video_key, numeric_keys
        )
        if count != declared_count or dimension != declared_dim:
            raise ValueError(
                "RobotMind raw layout changed: "
                f"count={count}/{declared_count} dim={dimension}/{declared_dim}"
            )
    else:
        count = declared_count
    numeric_meta = {
        "dimension": declared_dim,
        "keys": numeric_keys,
        "key_order_names": list(
            metadata.get("state_action_keys_order") or []
        ),
    }
    record = {
        "episode_id": _episode_id(row, "robotmind"),
        "robot": str(robot["name"]),
        "video": {
            "count": count,
            "storage": {"type": "hdf5", "path": path},
            "meta": {"key": video_key, "encoding": "jpeg"},
        },
        "state": {
            "count": count,
            "storage": {"type": "hdf5", "path": path},
            "meta": numeric_meta,
        },
        "action": {
            "count": count,
            "storage": {"type": "hdf5", "path": path},
            "meta": numeric_meta,
        },
        "texts": _robotmind_texts(row),
        "meta": {
            "task_name": str(row.get("task") or ""),
            "episode_name": str(
                metadata.get("episode_name")
                or metadata.get("episode_folder")
                or metadata.get("episode_id")
                or ""
            ),
            "state_action_dim": declared_dim,
            "state_action_keys_order": list(
                metadata.get("state_action_keys_order") or []
            ),
        },
    }
    validate_episode_record_v2(record)
    return record


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_source_family_episode_catalog(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    family: str,
    source_child: str,
    catalog_id: str,
    data_version: str,
    verify_source: bool = True,
    max_episodes: int | None = None,
    fail_on_episode_error: bool = True,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Bootstrap a split-neutral Catalog, then remove Manifest from runtime."""

    family = str(family).strip().lower()
    if family not in {"robotwin", "robotmind"}:
        raise ValueError("family must be robotwin or robotmind")
    source_child = str(source_child).strip()
    if not source_child:
        raise ValueError("source_child must be non-empty")
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")

    output = Path(output_path)
    metadata_output = _metadata_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not force and (output.exists() or metadata_output.exists()):
        raise FileExistsError(
            f"Catalog output already exists; use force to replace: {output}"
        )
    temporary_catalog = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.name}.tmp.{os.getpid()}"
    )
    started = time.monotonic()
    total_bytes = manifest.stat().st_size
    processed_bytes = 0
    input_rows = 0
    written = 0
    skipped = 0
    errors: Counter[str] = Counter()
    error_examples = []
    seen_episode_keys: set[tuple[str, str]] = set()
    current_key = None
    current_row = None
    digest = hashlib.sha256()

    def publish_row(handle, row):
        nonlocal written, skipped
        try:
            record = (
                _robotwin_record(row, verify_source=verify_source)
                if family == "robotwin"
                else _robotmind_record(row, verify_source=verify_source)
            )
            payload = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            written += 1
        except Exception as error:
            skipped += 1
            errors[type(error).__name__] += 1
            if len(error_examples) < 20:
                error_examples.append(
                    {
                        "episode_id": str(row.get("id")),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if fail_on_episode_error:
                raise

    try:
        with manifest.open("rb") as source, temporary_catalog.open("xb") as target:
            for line_number, payload in enumerate(source, start=1):
                processed_bytes += len(payload)
                if not payload.strip():
                    raise ValueError(f"blank Manifest line at {line_number}")
                row = json.loads(payload.decode("utf-8"))
                input_rows += 1
                if str(row.get("source_dataset")) != family:
                    raise ValueError(
                        f"Manifest line {line_number} source_dataset="
                        f"{row.get('source_dataset')!r}, expected {family!r}"
                    )
                key = _episode_key(row, family)
                if current_key is None:
                    current_key, current_row = key, row
                elif key != current_key:
                    if current_key in seen_episode_keys:
                        raise ValueError(
                            f"Manifest Episode is non-contiguous: {current_key}"
                        )
                    seen_episode_keys.add(current_key)
                    publish_row(target, current_row)
                    if max_episodes is not None and written >= int(max_episodes):
                        current_row = None
                        break
                    current_key, current_row = key, row
                if progress_callback is not None and (
                    input_rows == 1 or input_rows % 100_000 == 0
                ):
                    progress_callback(
                        {
                            "input_rows": input_rows,
                            "processed_bytes": processed_bytes,
                            "total_bytes": total_bytes,
                            "percent": 100.0 * processed_bytes / total_bytes,
                            "written_episodes": written,
                            "skipped_episodes": skipped,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
            if current_row is not None and (
                max_episodes is None or written < int(max_episodes)
            ):
                if current_key in seen_episode_keys:
                    raise ValueError(
                        f"Manifest Episode is non-contiguous: {current_key}"
                    )
                publish_row(target, current_row)
            target.flush()
            os.fsync(target.fileno())

        if written <= 0:
            raise RuntimeError("Catalog builder produced zero valid Episodes")
        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": family,
            "child": source_child,
            "episodes": written,
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "build": {
                    "bootstrap_type": "window_manifest_to_episode_catalog",
                    "bootstrap_manifest": str(manifest),
                    "runtime_depends_on_bootstrap_manifest": False,
                    "verify_source": bool(verify_source),
                    "input_rows": input_rows,
                    "written_episodes": written,
                    "skipped_episodes": skipped,
                    "error_counts": dict(sorted(errors.items())),
                    "error_examples": error_examples,
                    "catalog_size_bytes": temporary_catalog.stat().st_size,
                    "elapsed_seconds": time.monotonic() - started,
                },
            },
        }
        validate_catalog_metadata_v2(metadata)
        count = 0
        with temporary_catalog.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                validate_episode_catalog_binding(record, metadata)
                count += 1
        if count != written:
            raise RuntimeError(
                f"post-write Catalog count={count} expected={written}"
            )
        _write_json(temporary_metadata, metadata)
        if force:
            output.unlink(missing_ok=True)
            metadata_output.unlink(missing_ok=True)
        os.replace(temporary_catalog, output)
        os.replace(temporary_metadata, metadata_output)
        if progress_callback is not None:
            progress_callback(
                {
                    "input_rows": input_rows,
                    "processed_bytes": processed_bytes,
                    "total_bytes": total_bytes,
                    "percent": 100.0,
                    "written_episodes": written,
                    "skipped_episodes": skipped,
                    "elapsed_seconds": time.monotonic() - started,
                    "done": True,
                }
            )
        return metadata
    finally:
        temporary_catalog.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
