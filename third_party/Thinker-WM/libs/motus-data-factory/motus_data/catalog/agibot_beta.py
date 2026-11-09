"""Offline Agibot Beta Episode Catalog construction."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import h5py
except ImportError:
    h5py = None

from motus_data.datasets.agibot_beta import (
    TarMemberRef,
    scan_agibot_beta_episodes,
)
from motus_data.datasets.agibot_common import TASK_PROMPT
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)
from motus_data.iso_bmff import read_tar_member_video_sample_count
from motus_data.storage import BoundedFileReader


ProgressCallback = Callable[[Mapping[str, Any]], None]
PROPRIO_KEYS = (
    "state/end/position",
    "state/end/orientation",
    "state/effector/position",
)


def _locator(reference: TarMemberRef) -> dict[str, Any]:
    return {
        "type": "tar",
        "path": str(reference.uri),
        "member": str(reference.member),
        "offset": int(reference.offset_data),
        "size": int(reference.size),
    }


def _read_proprio_count(reference: TarMemberRef) -> int:
    if h5py is None:
        raise ImportError("Agibot Beta Catalog construction requires h5py")
    with BoundedFileReader(reference.uri, reference.offset_data, reference.size) as member:
        with h5py.File(member, "r") as handle:
            missing = [key for key in PROPRIO_KEYS if key not in handle]
            if missing:
                raise ValueError(f"proprio HDF5 missing keys: {missing}")
            shapes = {key: tuple(int(value) for value in handle[key].shape) for key in PROPRIO_KEYS}
    counts = {shape[0] for shape in shapes.values() if shape}
    if len(counts) != 1:
        raise ValueError(f"proprio HDF5 timeline lengths differ: {shapes}")
    count = counts.pop()
    if count <= 0:
        raise ValueError(f"proprio HDF5 has empty timeline: {shapes}")
    if len(shapes["state/end/position"]) != 3 or shapes["state/end/position"][-2:] != (2, 3):
        raise ValueError(f"unexpected end position shape: {shapes['state/end/position']}")
    if len(shapes["state/end/orientation"]) != 3 or shapes["state/end/orientation"][-2:] != (2, 4):
        raise ValueError(
            f"unexpected end orientation shape: {shapes['state/end/orientation']}"
        )
    if shapes["state/effector/position"][0] != count:
        raise ValueError(f"unexpected effector shape: {shapes['state/effector/position']}")
    return int(count)


def _text_items(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    task = str(episode.get("task_name") or "")
    scene = str(episode.get("init_scene_text") or "")
    items = [
        {
            "index": 0,
            "kind": "episode_context",
            "text": TASK_PROMPT.format(scene=scene.strip(), task=task.strip()),
            "meta": {
                "task_name": task,
                "init_scene_text": scene,
            },
        }
    ]
    for source_index, raw_segment in enumerate(episode["action_config"]):
        segment = dict(raw_segment)
        try:
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid action_config frame range at source index {source_index}"
            ) from error
        if start < 0 or end <= start:
            raise ValueError(
                f"invalid action_config range [{start}, {end}) at source index {source_index}"
            )
        action_text = str(segment.get("action_text") or "")
        skill = str(segment.get("skill") or "")
        selected_text = action_text.strip() or skill.strip()
        items.append(
            {
                "index": len(items),
                "kind": "action_segment",
                "text": f"Action: {selected_text}\n",
                "video_range": [start, end],
                "meta": {
                    "source_segment_index": int(source_index),
                    "action_text": action_text,
                    "skill": skill,
                    "source_annotation": segment,
                },
            }
        )
    return items


def _episode_record(
    episode: Mapping[str, Any],
    *,
    robot_name: str,
) -> dict[str, Any]:
    video_ref = episode["video_ref"]
    proprio_ref = episode["proprio_ref"]
    video_count = read_tar_member_video_sample_count(
        video_ref.uri, video_ref.offset_data, video_ref.size
    )
    proprio_count = _read_proprio_count(proprio_ref)
    record = {
        "episode_id": str(episode["episode_uid"]),
        "robot": str(robot_name),
        "video": {
            "count": int(video_count),
            "storage": _locator(video_ref),
            "meta": {"count_source": "iso_bmff_video_sample_table"},
        },
        "state": {
            "count": int(proprio_count),
            "storage": _locator(proprio_ref),
            "meta": {
                "count_source": "hdf5_dataset_shapes",
                "source_keys": list(PROPRIO_KEYS),
            },
        },
        "action": {
            "count": int(proprio_count),
            "storage": _locator(proprio_ref),
            "meta": {
                "count_source": "hdf5_dataset_shapes",
                "source_keys": list(PROPRIO_KEYS),
            },
        },
        "texts": _text_items(episode),
        "meta": {
            "task_id": int(episode["task_id"]),
            "raw_episode_id": int(episode["episode_id"]),
            "annotation_path": str(episode["annotation_path"]),
            "annotation_item_index": int(episode["annotation_item_index"]),
            "annotation_occurrence_count": len(episode["annotation_locations"]),
            "annotation_locations": list(episode["annotation_locations"]),
        },
    }
    validate_episode_record_v2(record)
    return record


def _metadata_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".meta.json")
    return Path(str(catalog_path) + ".meta.json")


def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_agibot_beta_episode_catalog(
    *,
    data_root: str | Path,
    tar_catalog_dir: str | Path,
    output_path: str | Path,
    catalog_id: str,
    data_version: str,
    source_dataset: str = "agibot_beta",
    source_child: str = "agibot_beta",
    robot_name: str = "agibot-a2-d",
    max_episodes: int | None = None,
    fail_on_episode_error: bool = False,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one validated Agibot Beta Catalog."""

    started = time.monotonic()
    output = Path(output_path)
    metadata_output = _metadata_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not force and (output.exists() or metadata_output.exists()):
        raise FileExistsError(
            f"Catalog output already exists; use force to replace: {output}, {metadata_output}"
        )

    episodes, source_scan_stats = scan_agibot_beta_episodes(
        data_root, tar_catalog_dir, ()
    )
    episodes = sorted(
        episodes, key=lambda item: (int(item["task_id"]), int(item["episode_id"]))
    )
    episode_ids = [str(item["episode_uid"]) for item in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        duplicates = [item for item, count in Counter(episode_ids).items() if count > 1]
        raise ValueError(f"duplicate Agibot Beta episode IDs: {duplicates[:10]}")

    candidates = episodes
    if max_episodes is not None:
        if int(max_episodes) <= 0:
            raise ValueError("max_episodes must be positive")
        candidates = candidates[: int(max_episodes)]

    temporary_catalog = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.name}.tmp.{os.getpid()}"
    )
    errors: Counter[str] = Counter()
    error_examples = []
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary_catalog.open("x", encoding="utf-8") as handle:
            total = len(candidates)
            for position, episode in enumerate(candidates, start=1):
                try:
                    record = _episode_record(
                        episode,
                        robot_name=robot_name,
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
                    handle.write(payload.decode("utf-8"))
                    digest.update(payload)
                    written += 1
                except Exception as error:
                    errors[type(error).__name__] += 1
                    if len(error_examples) < 20:
                        error_examples.append(
                            {
                                "episode_id": str(episode.get("episode_uid")),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                    if fail_on_episode_error:
                        raise RuntimeError(
                            f"failed to build Catalog record for {episode.get('episode_uid')}"
                        ) from error
                if progress_callback is not None:
                    progress_callback(
                        {
                            "processed": position,
                            "total": total,
                            "written": written,
                            "skipped": position - written,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())

        if written <= 0:
            raise RuntimeError("Catalog builder produced zero valid Episode records")
        statistics = {
            "source_scan": source_scan_stats,
            "usable_source_episodes": len(episodes),
            "candidate_episodes": len(candidates),
            "written_episodes": written,
            "skipped_episodes": len(candidates) - written,
            "error_counts": dict(sorted(errors.items())),
            "error_examples": error_examples,
            "catalog_sha256": digest.hexdigest(),
            "catalog_size_bytes": temporary_catalog.stat().st_size,
            "elapsed_seconds": time.monotonic() - started,
        }
        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": str(source_dataset),
            "child": str(source_child),
            "episodes": int(written),
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "build": statistics,
            },
        }
        validate_catalog_metadata_v2(metadata)
        validated_records = validate_catalog_file_bindings(temporary_catalog, metadata)
        metadata["meta"]["build"]["post_write_validated_records"] = int(
            validated_records
        )
        metadata["meta"]["build"]["elapsed_seconds"] = (
            time.monotonic() - started
        )
        validate_catalog_metadata_v2(metadata)
        _write_json_file(temporary_metadata, metadata)

        if force:
            output.unlink(missing_ok=True)
            metadata_output.unlink(missing_ok=True)
        os.replace(temporary_catalog, output)
        os.replace(temporary_metadata, metadata_output)
        return metadata
    finally:
        temporary_catalog.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def validate_catalog_file_bindings(
    catalog_path: str | Path, metadata: Mapping[str, Any]
) -> int:
    """Stream a completed Catalog and validate every row against file metadata."""

    count = 0
    with Path(catalog_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank Catalog line at {line_number}")
            record = json.loads(line)
            validate_episode_catalog_binding(record, metadata)
            count += 1
    expected_count = (
        metadata.get("record_count")
        if "record_count" in metadata
        else metadata.get("episodes")
    )
    if count != int(expected_count if expected_count is not None else -1):
        raise ValueError(
            f"Catalog row count={count} differs from metadata={expected_count}"
        )
    return count
