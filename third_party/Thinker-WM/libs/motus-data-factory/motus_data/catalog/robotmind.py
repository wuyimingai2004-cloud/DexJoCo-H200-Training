"""Build a split-neutral Episode Catalog directly from native RobotMind HDF5."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError:
    h5py = None

from motus_data.datasets.robotmind import (
    RobotMindEmbodimentSchema,
    RobotMindEpisode,
    get_robotmind_schema,
    scan_robotmind_episodes,
)
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _metadata_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".meta.json")
    return Path(str(catalog_path) + ".meta.json")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def _validate_numeric_dataset(
    dataset: Any,
    *,
    key: str,
    chunk_rows: int = 4096,
) -> None:
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(
            f"RobotMind numeric key={key!r} has non-numeric dtype={dataset.dtype}"
        )
    for start in range(0, int(dataset.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(dataset.shape[0]))
        values = np.asarray(dataset[start:end])
        if not np.isfinite(values).all():
            raise ValueError(
                f"RobotMind numeric key={key!r} contains NaN/Inf in rows [{start},{end})"
            )


def _decode_image_shape(value: Any, *, key: str, index: int) -> tuple[int, int, int]:
    try:
        with Image.open(io.BytesIO(bytes(value))) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
    except Exception as error:
        raise ValueError(
            f"RobotMind JPEG key={key!r} index={index} cannot be decoded: {error}"
        ) from error
    return height, width, 3


def _inspect_episode(
    episode: RobotMindEpisode,
    schema: RobotMindEmbodimentSchema,
    *,
    validate_numeric_finite: bool,
    validate_jpeg: bool,
) -> tuple[int, tuple[int, ...], str, tuple[int, int, int] | None]:
    if h5py is None:
        raise ImportError("RobotMind Catalog construction requires h5py")
    numeric_keys = [path for _, path in schema.state_action_keys]
    required = [schema.rgb_key, *numeric_keys]
    with h5py.File(episode.hdf5_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"RobotMind HDF5 missing keys={missing}: {episode.hdf5_path}"
            )
        counts = [int(handle[key].shape[0]) for key in required]
        en_instruction = ""
        if "metadata" in handle:
            trajectory_length = handle["metadata"].attrs.get("trajectory_length")
            if trajectory_length is not None:
                counts.append(int(trajectory_length))
            en_instruction = _clean_text(
                handle["metadata"].attrs.get("language_instruction", "")
            )
        if len(set(counts)) != 1:
            raise ValueError(
                "RobotMind modalities are not exactly aligned: "
                f"episode={episode.episode_uid} counts={counts}"
            )
        count = counts[0]
        if count <= 0:
            raise ValueError(
                f"RobotMind Episode has no frames: {episode.episode_uid}"
            )
        field_dims = tuple(
            int(np.prod(handle[key].shape[1:]))
            if len(handle[key].shape) > 1
            else 1
            for key in numeric_keys
        )
        if sum(field_dims) != schema.state_action_dim:
            raise ValueError(
                f"RobotMind {schema.name} dimension mismatch: "
                f"episode={episode.episode_uid} fields={field_dims} "
                f"total={sum(field_dims)} expected={schema.state_action_dim}"
            )
        if validate_numeric_finite:
            for key in numeric_keys:
                _validate_numeric_dataset(handle[key], key=key)
        image_shape = None
        if validate_jpeg:
            first_shape = _decode_image_shape(
                handle[schema.rgb_key][0], key=schema.rgb_key, index=0
            )
            last_index = count - 1
            last_shape = _decode_image_shape(
                handle[schema.rgb_key][last_index],
                key=schema.rgb_key,
                index=last_index,
            )
            if first_shape != last_shape:
                raise ValueError(
                    "RobotMind first/last JPEG shapes differ: "
                    f"episode={episode.episode_uid} first={first_shape} last={last_shape}"
                )
            image_shape = first_shape
    return count, field_dims, en_instruction, image_shape


def _texts(episode: RobotMindEpisode, en_instruction: str) -> list[dict[str, Any]]:
    texts = []
    if en_instruction:
        texts.append(
            {
                "index": len(texts),
                "kind": "instruction_en",
                "text": en_instruction,
            }
        )
    zh_instruction = _clean_text(episode.zh_instruction)
    if zh_instruction and zh_instruction != en_instruction:
        texts.append(
            {
                "index": len(texts),
                "kind": "instruction_zh",
                "text": zh_instruction,
            }
        )
    if not texts:
        raise ValueError(
            f"RobotMind Episode has no usable instruction: {episode.episode_uid}"
        )
    return texts


def _episode_record(
    episode: RobotMindEpisode,
    schema: RobotMindEmbodimentSchema,
    *,
    robot_name: str,
    validate_numeric_finite: bool,
    validate_jpeg: bool,
) -> dict[str, Any]:
    count, field_dims, en_instruction, image_shape = _inspect_episode(
        episode,
        schema,
        validate_numeric_finite=validate_numeric_finite,
        validate_jpeg=validate_jpeg,
    )
    key_names = [name for name, _ in schema.state_action_keys]
    numeric_keys = [path for _, path in schema.state_action_keys]
    storage = {"type": "hdf5", "path": episode.hdf5_path}
    numeric_meta = {
        "dimension": schema.state_action_dim,
        "keys": numeric_keys,
        "key_order_names": key_names,
        "field_dimensions": list(field_dims),
    }
    video_meta: dict[str, Any] = {
        "key": schema.rgb_key,
        "encoding": "jpeg",
    }
    if image_shape is not None:
        video_meta["native_shape_hwc"] = list(image_shape)
    record = {
        "episode_id": episode.episode_uid,
        "robot": robot_name,
        "video": {
            "count": count,
            "storage": storage,
            "meta": video_meta,
        },
        "state": {
            "count": count,
            "storage": storage,
            "meta": dict(numeric_meta),
        },
        "action": {
            "count": count,
            "storage": storage,
            "meta": dict(numeric_meta),
        },
        "texts": _texts(episode, en_instruction),
        "meta": {
            "task_name": episode.task_name,
            "episode_name": episode.episode_name,
            "state_action_dim": schema.state_action_dim,
            "state_action_keys_order": key_names,
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


def build_robotmind_episode_catalog(
    *,
    data_root: str | Path,
    embodiment: str,
    output_path: str | Path,
    catalog_id: str,
    data_version: str,
    source_child: str | None = None,
    robot_name: str | None = None,
    max_episodes: int | None = None,
    validate_numeric_finite: bool = True,
    validate_jpeg: bool = True,
    fail_on_episode_error: bool = True,
    fail_on_missing_hdf5: bool = True,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a Catalog from complete raw Episodes."""

    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")
    schema = get_robotmind_schema(embodiment)
    child = str(source_child or f"robotmind_{schema.name}").strip()
    if not child:
        raise ValueError("source_child must be non-empty")
    resolved_robot_name = str(robot_name or schema.robot_name).strip()
    if not resolved_robot_name:
        raise ValueError("robot_name must be non-empty")
    root = Path(data_root)
    episodes = scan_robotmind_episodes(
        root,
        schema.name,
        fail_on_missing_hdf5=fail_on_missing_hdf5,
        max_episodes=max_episodes,
        progress_callback=progress_callback,
    )
    episode_ids = [episode.episode_uid for episode in episodes]
    duplicates = [
        value for value, count in Counter(episode_ids).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate RobotMind Episode IDs: {duplicates[:10]}")
    candidates = (
        episodes if max_episodes is None else episodes[: int(max_episodes)]
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
                        episode,
                        schema,
                        robot_name=resolved_robot_name,
                        validate_numeric_finite=validate_numeric_finite,
                        validate_jpeg=validate_jpeg,
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
                                "episode_id": episode.episode_uid,
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
            raise RuntimeError("RobotMind Catalog builder produced zero Episodes")
        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": "robotmind",
            "child": child,
            "episodes": written,
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "build": {
                    "source_type": "raw_episode_hdf5",
                    "data_root": str(root),
                    "embodiment": schema.name,
                    "candidate_episodes": len(candidates),
                    "written_episodes": written,
                    "skipped_episodes": len(candidates) - written,
                    "validate_numeric_finite": bool(validate_numeric_finite),
                    "validate_jpeg": bool(validate_jpeg),
                    "fail_on_missing_hdf5": bool(fail_on_missing_hdf5),
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
        if progress_callback is not None:
            progress_callback(
                {
                    "processed": len(candidates),
                    "total": len(candidates),
                    "percent": 100.0,
                    "written": written,
                    "skipped": len(candidates) - written,
                    "elapsed_seconds": time.monotonic() - started,
                    "done": True,
                }
            )
        return metadata
    finally:
        temporary_catalog.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
