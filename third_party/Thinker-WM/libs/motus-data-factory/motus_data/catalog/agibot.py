"""Offline split-neutral Episode Catalog construction for native Agibot."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from motus_data.datasets.agibot_common import (
    TASK_PROMPT,
    read_agibot_eef_actions,
)
from motus_data.datasets.agibot_native import scan_agibot_episodes
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)
from motus_data.iso_bmff import read_video_sample_count


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _video_count(path: str | Path) -> int:
    source = Path(path)
    with source.open("rb") as handle:
        return int(read_video_sample_count(handle, source.stat().st_size))


def _metadata_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".meta.json")
    return Path(str(catalog_path) + ".meta.json")


def _action_text(action: Mapping[str, Any]) -> str:
    text = str(action.get("action_text") or "").strip()
    if not text:
        text = str(action.get("skill") or "").strip()
    return f"Action: {text}\n"


def _texts(
    episode: Mapping[str, Any],
    *,
    video_count: int,
) -> list[dict[str, Any]]:
    task = str(episode.get("task_name") or "")
    scene = str(episode.get("init_scene_text") or "")
    items = [
        {
            "index": 0,
            "kind": "episode_context",
            "text": TASK_PROMPT.format(scene=scene.strip(), task=task.strip()),
            "meta": {"task_name": task, "init_scene_text": scene},
        }
    ]
    labels = episode.get("action_labels") or {}
    parsed = []
    for raw_start, raw_action in labels.items():
        try:
            start = int(raw_start)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Agibot action label key is not an integer: {raw_start!r}"
            ) from error
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"Agibot action label at {start} is not a mapping")
        if not 0 <= start < video_count:
            raise ValueError(
                f"Agibot action label start={start} outside video count={video_count}"
            )
        parsed.append((start, raw_action))
    parsed.sort(key=lambda item: item[0])
    if not parsed:
        raise ValueError("Agibot Episode has no action labels")
    for position, (start, action) in enumerate(parsed):
        end = (
            parsed[position + 1][0]
            if position + 1 < len(parsed)
            else video_count
        )
        if end <= start:
            raise ValueError(
                f"Agibot action label range is empty: [{start},{end})"
            )
        items.append(
            {
                "index": len(items),
                "kind": "action_segment",
                "text": _action_text(action),
                "video_range": [start, end],
                "meta": {
                    "source_start_frame": start,
                    "source_annotation": dict(action),
                },
            }
        )
    return items


def _episode_record(
    episode: Mapping[str, Any],
    *,
    robot_name: str,
) -> dict[str, Any]:
    ratio = int(episode["action_video_fps_ratio"])
    if ratio <= 0:
        raise ValueError("Agibot action_video_fps_ratio must be positive")
    video_count = _video_count(episode["video_path"])
    proprio_count = int(
        read_agibot_eef_actions(episode["proprio_path"]).shape[0]
    )
    complete_video_frames = proprio_count // ratio
    partial_tail_steps = proprio_count % ratio
    expected_video_count = complete_video_frames + int(partial_tail_steps > 0)
    if video_count != expected_video_count:
        raise ValueError(
            "Agibot video/proprio timeline mismatch: "
            f"video={video_count} proprio={proprio_count} ratio={ratio} "
            f"expected_video={expected_video_count}"
        )
    effective = [int(value) for value in episode["effective_frames"]]
    if (
        len(effective) != 2
        or effective[0] < 0
        or effective[1] < effective[0]
        or effective[1] >= video_count
    ):
        raise ValueError(
            "Agibot invalid effective_frames="
            f"{effective} video_count={video_count} "
            f"complete_video_frames={complete_video_frames}"
        )
    storage_video = {"type": "file", "path": str(episode["video_path"])}
    storage_proprio = {
        "type": "file",
        "path": str(episode["proprio_path"]),
    }
    record = {
        "episode_id": str(episode["episode_uid"]),
        "robot": str(robot_name),
        "video": {
            "count": video_count,
            "storage": storage_video,
            "meta": {
                "count_source": "iso_bmff_video_sample_table",
                "timeline": "video_frame",
            },
        },
        "state": {
            "count": proprio_count,
            "storage": storage_proprio,
            "meta": {
                "dimension": 26,
                "normalization": "agibot_eef_v1",
                "timeline": "proprio_timestep",
            },
        },
        "action": {
            "count": proprio_count,
            "storage": storage_proprio,
            "meta": {
                "dimension": 26,
                "normalization": "agibot_eef_v1",
                "timeline": "action_timestep",
            },
        },
        "texts": _texts(episode, video_count=video_count),
        "meta": {
            "task_id": int(episode["task_id"]),
            "raw_episode_id": int(episode["episode_id"]),
            "task_name": str(episode.get("task_name") or ""),
            "init_scene_text": str(episode.get("init_scene_text") or ""),
            "annotation_path": str(episode["annotation_path"]),
            "effective_video_frames": effective,
            "action_video_fps_ratio": ratio,
            "timeline_mapping": {
                "video_frame_to_proprio_start": "video_index * ratio",
                "ratio": ratio,
                "complete_video_frames": complete_video_frames,
                "partial_tail_steps": partial_tail_steps,
            },
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


def build_agibot_episode_catalog(
    *,
    annotation_folders: Sequence[str],
    data_root_folders: Sequence[str],
    dataset_names: Sequence[str],
    output_path: str | Path,
    catalog_id: str,
    data_version: str,
    source_child: str = "agibot",
    robot_name: str = "agibot-a2-d",
    max_episodes: int | None = None,
    fail_on_episode_error: bool = True,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the native Agibot Episode Catalog."""

    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")
    episodes = scan_agibot_episodes(
        annotation_folders, data_root_folders, dataset_names
    )
    episodes = sorted(
        episodes,
        key=lambda item: (int(item["task_id"]), int(item["episode_id"])),
    )
    episode_ids = [str(item["episode_uid"]) for item in episodes]
    duplicates = [
        value
        for value, count in Counter(episode_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate Agibot Episode IDs: {duplicates[:10]}")
    candidates = (
        episodes
        if max_episodes is None
        else episodes[: int(max_episodes)]
    )

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
    digest = hashlib.sha256()
    written = 0
    errors: Counter[str] = Counter()
    error_examples = []
    try:
        with temporary_catalog.open("xb") as handle:
            total = len(candidates)
            for position, episode in enumerate(candidates, start=1):
                try:
                    record = _episode_record(
                        episode, robot_name=robot_name
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
                    errors[type(error).__name__] += 1
                    if len(error_examples) < 20:
                        error_examples.append(
                            {
                                "episode_id": str(episode["episode_uid"]),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                    if fail_on_episode_error:
                        raise
                if progress_callback is not None:
                    progress_callback(
                        {
                            "processed": position,
                            "total": total,
                            "percent": 100.0 * position / total,
                            "written": written,
                            "skipped": position - written,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())
        if written <= 0:
            raise RuntimeError("Agibot Catalog builder produced zero Episodes")
        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": "agibot",
            "child": str(source_child),
            "episodes": written,
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "build": {
                    "source_annotations": list(annotation_folders),
                    "candidate_episodes": len(candidates),
                    "written_episodes": written,
                    "skipped_episodes": len(candidates) - written,
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
                validate_episode_catalog_binding(json.loads(line), metadata)
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
        return metadata
    finally:
        temporary_catalog.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
