"""Build split-neutral EgoVerse Catalogs from complete Zarr v3 Episodes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from motus_data.egoverse import (
    EgoVerseProfile,
    ZarrGroupOpener,
    decode_egoverse_annotation,
    get_egoverse_profile,
    open_egoverse_zarr_group,
)
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class EgoVerseEpisode:
    episode_id: str
    episode_name: str
    zarr_path: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class EgoVerseScanIssue:
    zarr_path: str
    error_type: str
    error: str


@dataclass(frozen=True)
class EgoVerseScanResult:
    episodes: tuple[EgoVerseEpisode, ...]
    candidate_count: int
    scanned_count: int
    skipped_profile_count: int
    issues: tuple[EgoVerseScanIssue, ...]


def _metadata_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".meta.json")
    return Path(str(catalog_path) + ".meta.json")


def _quarantine_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".quarantine.jsonl")
    return Path(str(catalog_path) + ".quarantine.jsonl")


def _normalized_embodiment(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _root_attributes(path: Path) -> Mapping[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"EgoVerse store lacks root zarr.json: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "group":
        raise ValueError(f"EgoVerse store is not a Zarr v3 group: {path}")
    attributes = metadata.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError(f"EgoVerse root attributes are missing: {path}")
    return attributes


def scan_egoverse_episode_inventory(
    data_root: str | Path,
    profile: str | EgoVerseProfile,
    *,
    max_episodes: int | None = None,
    fail_on_store_error: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> EgoVerseScanResult:
    """Enumerate complete Zarr Episodes and report invalid stores."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"EgoVerse data_root is not a directory: {root}")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")
    selected_profile = (
        profile if isinstance(profile, EgoVerseProfile) else get_egoverse_profile(profile)
    )
    accepted = {
        _normalized_embodiment(value)
        for value in selected_profile.raw_embodiments
    }
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.endswith(".zarr")
    )
    total = len(candidates)
    episodes: list[EgoVerseEpisode] = []
    issues: list[EgoVerseScanIssue] = []
    skipped_profile = 0
    scanned_count = 0
    for position, path in enumerate(candidates, start=1):
        scanned_count = position
        try:
            attributes = _root_attributes(path)
        except Exception as error:
            if fail_on_store_error:
                raise
            issues.append(
                EgoVerseScanIssue(
                    zarr_path=str(path),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
        else:
            raw_embodiment = _normalized_embodiment(
                attributes.get("embodiment")
            )
            if raw_embodiment not in accepted:
                skipped_profile += 1
            else:
                episode_name = path.name[: -len(".zarr")]
                episodes.append(
                    EgoVerseEpisode(
                        episode_id=(
                            f"egoverse:{selected_profile.name}:{episode_name}"
                        ),
                        episode_name=episode_name,
                        zarr_path=str(path),
                        attributes=dict(attributes),
                    )
                )
        done = max_episodes is not None and len(episodes) >= int(max_episodes)
        if progress_callback is not None and not done:
            progress_callback(
                {
                    "phase": "scan",
                    "processed": position,
                    "total": total,
                    "percent": 100.0 if done else 100.0 * position / max(1, total),
                    "discovered_episodes": len(episodes),
                    "skipped_profile": skipped_profile,
                    "scan_errors": len(issues),
                    "done": done,
                }
            )
        if done:
            break
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "scan",
                "processed": scanned_count,
                "total": total,
                "percent": 100.0,
                "discovered_episodes": len(episodes),
                "skipped_profile": skipped_profile,
                "scan_errors": len(issues),
                "done": True,
            }
        )
    return EgoVerseScanResult(
        episodes=tuple(episodes),
        candidate_count=total,
        scanned_count=scanned_count,
        skipped_profile_count=skipped_profile,
        issues=tuple(issues),
    )


def scan_egoverse_episodes(
    data_root: str | Path,
    profile: str | EgoVerseProfile,
    *,
    max_episodes: int | None = None,
    fail_on_store_error: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[EgoVerseEpisode]:
    """Compatibility wrapper returning only matching complete Episodes."""

    result = scan_egoverse_episode_inventory(
        data_root,
        profile,
        max_episodes=max_episodes,
        fail_on_store_error=fail_on_store_error,
        progress_callback=progress_callback,
    )
    return list(result.episodes)


def _field_dimension(array: Any) -> int:
    return int(np.prod(array.shape[1:])) if len(array.shape) > 1 else 1


def _validate_numeric_array(
    array: Any,
    *,
    key: str,
    valid_count: int,
    chunk_rows: int = 4096,
) -> None:
    dtype = np.dtype(array.dtype)
    if not np.issubdtype(dtype, np.number):
        raise ValueError(
            f"EgoVerse numeric key={key!r} has non-numeric dtype={dtype}"
        )
    for start in range(0, valid_count, chunk_rows):
        end = min(start + chunk_rows, valid_count)
        try:
            values = np.asarray(array[start:end])
        except Exception as error:
            raise ValueError(
                f"EgoVerse numeric key={key!r} cannot decode rows "
                f"[{start},{end}): {error}"
            ) from error
        if not np.isfinite(values).all():
            raise ValueError(
                f"EgoVerse key={key!r} contains NaN/Inf in rows [{start},{end})"
            )


def _bytes_value(value: Any, *, key: str, index: int) -> bytes:
    if isinstance(value, np.void):
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise ValueError(
            f"EgoVerse key={key!r} index={index} is not encoded bytes"
        )
    return value


def _jpeg_shape(array: Any, *, key: str, index: int) -> tuple[int, int, int]:
    raw = np.asarray(array[index : index + 1], dtype=object).reshape(-1)
    if len(raw) != 1:
        raise ValueError(
            f"EgoVerse key={key!r} index={index} returned {len(raw)} values"
        )
    value = _bytes_value(raw[0], key=key, index=index)
    try:
        with Image.open(io.BytesIO(value)) as image:
            width, height = image.convert("RGB").size
    except Exception as error:
        raise ValueError(
            f"EgoVerse JPEG key={key!r} index={index} cannot decode: {error}"
        ) from error
    return height, width, 3


def _text_items(
    group: Any,
    attributes: Mapping[str, Any],
    *,
    valid_count: int,
) -> list[dict[str, Any]]:
    items = []
    task_description = str(
        attributes.get("task_description")
        or attributes.get("task")
        or ""
    ).strip()
    if task_description:
        items.append(
            {
                "index": len(items),
                "kind": "task_description_en",
                "text": task_description,
                "meta": {"scope": "episode"},
            }
        )

    if "annotations" in group:
        annotations = group["annotations"]
        raw_entries = np.asarray(annotations[:], dtype=object).reshape(-1)
        for source_index, raw in enumerate(raw_entries):
            decoded = decode_egoverse_annotation(raw)
            text = str(decoded.get("text") or "").strip()
            start = int(decoded.get("start_idx", -1))
            end = int(decoded.get("end_idx", -1))
            if not text:
                raise ValueError(
                    f"EgoVerse annotation index={source_index} has empty text"
                )
            if start < 0 or end <= start or end > valid_count:
                raise ValueError(
                    "EgoVerse annotation range is invalid: "
                    f"index={source_index} range=[{start},{end}) "
                    f"valid_count={valid_count}"
                )
            items.append(
                {
                    "index": len(items),
                    "kind": "annotation_en",
                    "text": text,
                    "video_range": [start, end],
                    "meta": {"source_annotation_index": source_index},
                }
            )

    if not items:
        items.append(
            {
                "index": 0,
                "kind": "missing_instruction",
                "text": "",
                "meta": {"missing": True},
            }
        )
    return items


def _episode_record(
    episode: EgoVerseEpisode,
    profile: EgoVerseProfile,
    *,
    group_opener: ZarrGroupOpener,
    validate_numeric_finite: bool,
    validate_jpeg: bool,
) -> dict[str, Any]:
    group = group_opener(episode.zarr_path)
    attributes = dict(group.attrs)
    valid_count = int(attributes.get("total_frames", 0))
    if valid_count <= 0:
        raise ValueError(f"EgoVerse Episode has invalid total_frames: {episode.episode_id}")
    if _normalized_embodiment(attributes.get("embodiment")) not in {
        _normalized_embodiment(value) for value in profile.raw_embodiments
    }:
        raise ValueError(
            f"EgoVerse Episode embodiment changed: {episode.episode_id}"
        )

    state_keys = list(profile.state_keys)
    action_keys = list(profile.action_keys)
    required_keys = [profile.video_key, *state_keys, *action_keys]
    missing = sorted(set(key for key in required_keys if key not in group))
    if missing:
        raise ValueError(
            f"EgoVerse Episode is missing keys={missing}: {episode.episode_id}"
        )
    for key in sorted(set(required_keys)):
        if int(group[key].shape[0]) < valid_count:
            raise ValueError(
                "EgoVerse modality is shorter than total_frames: "
                f"episode={episode.episode_id} key={key!r} "
                f"shape0={group[key].shape[0]} total_frames={valid_count}"
            )

    state_field_dims = [_field_dimension(group[key]) for key in state_keys]
    action_field_dims = [_field_dimension(group[key]) for key in action_keys]
    if sum(state_field_dims) != profile.state_dim:
        raise ValueError(
            f"EgoVerse state dimension mismatch: episode={episode.episode_id} "
            f"fields={state_field_dims} expected={profile.state_dim}"
        )
    if sum(action_field_dims) != profile.action_dim:
        raise ValueError(
            f"EgoVerse action dimension mismatch: episode={episode.episode_id} "
            f"fields={action_field_dims} expected={profile.action_dim}"
        )
    if validate_numeric_finite:
        for key in sorted(set(state_keys + action_keys)):
            _validate_numeric_array(
                group[key],
                key=key,
                valid_count=valid_count,
            )

    image_shape = None
    if validate_jpeg:
        first_shape = _jpeg_shape(
            group[profile.video_key],
            key=profile.video_key,
            index=0,
        )
        last_shape = _jpeg_shape(
            group[profile.video_key],
            key=profile.video_key,
            index=valid_count - 1,
        )
        if first_shape != last_shape:
            raise ValueError(
                "EgoVerse first/last JPEG shapes differ: "
                f"episode={episode.episode_id} "
                f"first={first_shape} last={last_shape}"
            )
        image_shape = first_shape

    storage = {"type": "zarr_v3", "path": episode.zarr_path}
    video_meta: dict[str, Any] = {
        "key": profile.video_key,
        "encoding": "jpeg",
        "valid_length_source": "root.total_frames",
    }
    if image_shape is not None:
        video_meta["native_shape_hwc"] = list(image_shape)
    features = attributes.get("features")
    available_feature_keys = (
        sorted(str(key) for key in features)
        if isinstance(features, Mapping)
        else []
    )
    record = {
        "episode_id": episode.episode_id,
        "robot": profile.robot_name,
        "video": {
            "count": valid_count,
            "storage": storage,
            "meta": video_meta,
        },
        "state": {
            "count": valid_count,
            "storage": storage,
            "meta": {
                "dimension": profile.state_dim,
                "keys": state_keys,
                "field_dimensions": state_field_dims,
                "semantics": profile.state_semantics,
                "normalization": "none",
            },
        },
        "action": {
            "count": valid_count,
            "storage": storage,
            "meta": {
                "dimension": profile.action_dim,
                "keys": action_keys,
                "field_dimensions": action_field_dims,
                "semantics": profile.action_semantics,
                "source_signal": profile.action_source_signal,
                "is_control_command": profile.action_is_control_command,
                "indices": "model_provided",
                "normalization": "none",
            },
        },
        "texts": _text_items(
            group,
            attributes,
            valid_count=valid_count,
        ),
        "meta": {
            "profile": profile.name,
            "raw_embodiment": str(attributes.get("embodiment") or ""),
            "episode_name": episode.episode_name,
            "task_name": str(attributes.get("task_name") or ""),
            "fps": float(attributes.get("fps", 0.0)),
            "total_frames": valid_count,
            "available_feature_keys": available_feature_keys,
            "coordinate_semantics": "raw_source_values",
        },
    }
    if record["meta"]["fps"] <= 0:
        raise ValueError(f"EgoVerse Episode has invalid fps: {episode.episode_id}")
    validate_episode_record_v2(record)
    return record


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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


def _increment(mapping: dict[str, int], key: Any, amount: int = 1) -> None:
    normalized = str(key)
    mapping[normalized] = int(mapping.get(normalized, 0)) + int(amount)


def _new_quality_stats() -> dict[str, Any]:
    return {
        "episodes": 0,
        "total_frames": 0,
        "min_frames": None,
        "max_frames": 0,
        "text_items": 0,
        "annotation_items": 0,
        "text_source_counts": {
            "annotation_and_episode_text": 0,
            "annotation_only": 0,
            "episode_text_only": 0,
            "no_text": 0,
        },
        "text_kind_counts": {},
        "fps_counts": {},
        "raw_embodiment_counts": {},
        "native_video_shape_counts": {},
    }


def _update_quality_stats(
    quality: dict[str, Any],
    record: Mapping[str, Any],
) -> None:
    frame_count = int(record["video"]["count"])
    quality["episodes"] = int(quality["episodes"]) + 1
    quality["total_frames"] = int(quality["total_frames"]) + frame_count
    minimum = quality["min_frames"]
    quality["min_frames"] = (
        frame_count if minimum is None else min(int(minimum), frame_count)
    )
    quality["max_frames"] = max(int(quality["max_frames"]), frame_count)

    kinds = []
    for item in record["texts"]:
        kind = str(item["kind"])
        kinds.append(kind)
        quality["text_items"] = int(quality["text_items"]) + 1
        _increment(quality["text_kind_counts"], kind)
        if kind == "annotation_en":
            quality["annotation_items"] = int(
                quality["annotation_items"]
            ) + 1
    has_annotation = "annotation_en" in kinds
    has_episode_text = "task_description_en" in kinds
    if has_annotation and has_episode_text:
        category = "annotation_and_episode_text"
    elif has_annotation:
        category = "annotation_only"
    elif has_episode_text:
        category = "episode_text_only"
    else:
        category = "no_text"
    _increment(quality["text_source_counts"], category)

    metadata = record.get("meta") or {}
    _increment(quality["fps_counts"], metadata.get("fps", "unknown"))
    _increment(
        quality["raw_embodiment_counts"],
        metadata.get("raw_embodiment", "unknown"),
    )
    native_shape = (record["video"].get("meta") or {}).get(
        "native_shape_hwc"
    )
    if native_shape is not None:
        _increment(
            quality["native_video_shape_counts"],
            "x".join(str(int(value)) for value in native_shape),
        )


def _finalize_quality_stats(
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(quality))
    episodes = int(result["episodes"])
    result["mean_frames"] = (
        float(result["total_frames"]) / episodes if episodes else 0.0
    )
    return result


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inventory_digest(scan: EgoVerseScanResult) -> str:
    return _stable_digest(
        {
            "episodes": [
                [episode.episode_id, episode.zarr_path]
                for episode in scan.episodes
            ],
            "issues": [
                [
                    issue.zarr_path,
                    issue.error_type,
                    issue.error,
                ]
                for issue in scan.issues
            ],
            "candidate_count": scan.candidate_count,
            "scanned_count": scan.scanned_count,
            "skipped_profile_count": scan.skipped_profile_count,
        }
    )


def _resume_paths(output: Path) -> tuple[Path, Path]:
    partial = output.with_name(f".{output.name}.build.partial")
    state = output.with_name(f".{output.name}.build.state.json")
    return partial, state


def _digest_file(path: Path) -> tuple[Any, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for payload in handle:
            digest.update(payload)
            rows += 1
    return digest, rows


def _checkpoint_build(
    *,
    handle: Any,
    state_path: Path,
    spec_digest: str,
    inventory_digest: str,
    processed: int,
    written: int,
    errors: Mapping[str, int],
    error_examples: list[dict[str, Any]],
    quarantined_episodes: list[dict[str, Any]],
    quality: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    _write_json_atomic(
        state_path,
        {
            "format": "motus_egoverse_catalog_build_state_v1",
            "spec_sha256": spec_digest,
            "inventory_sha256": inventory_digest,
            "processed": int(processed),
            "written": int(written),
            "catalog_byte_offset": int(handle.tell()),
            "error_counts": dict(sorted(errors.items())),
            "error_examples": list(error_examples),
            "quarantined_episodes": list(quarantined_episodes),
            "quality": quality,
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def build_egoverse_episode_catalog(
    *,
    data_root: str | Path,
    profile: str,
    output_path: str | Path,
    catalog_id: str,
    data_version: str,
    source_child: str | None = None,
    max_episodes: int | None = None,
    validate_numeric_finite: bool = True,
    validate_jpeg: bool = True,
    fail_on_episode_error: bool = True,
    fail_on_store_error: bool = True,
    resume: bool = False,
    checkpoint_interval: int = 100,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
    group_opener: ZarrGroupOpener | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one profile-specific EgoVerse Catalog."""

    if resume and force:
        raise ValueError("EgoVerse --resume and --force are mutually exclusive")
    if int(checkpoint_interval) <= 0:
        raise ValueError("checkpoint_interval must be positive")
    selected_profile = get_egoverse_profile(profile)
    child = str(source_child or selected_profile.source_child).strip()
    if not child:
        raise ValueError("source_child must be non-empty")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")
    opener = group_opener or open_egoverse_zarr_group
    scan = scan_egoverse_episode_inventory(
        data_root,
        selected_profile,
        max_episodes=max_episodes,
        fail_on_store_error=fail_on_store_error,
        progress_callback=progress_callback,
    )
    episodes = list(scan.episodes)
    episode_ids = [episode.episode_id for episode in episodes]
    duplicates = [
        value for value, count in Counter(episode_ids).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate EgoVerse Episode IDs: {duplicates[:10]}")

    output = Path(output_path)
    metadata_output = _metadata_path(output)
    quarantine_output = _quarantine_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        output.exists()
        or metadata_output.exists()
        or quarantine_output.exists()
    ):
        if resume:
            raise FileExistsError(
                f"completed Catalog output already exists: {output}"
            )
    if not force and (
        output.exists()
        or metadata_output.exists()
        or quarantine_output.exists()
    ):
        raise FileExistsError(
            f"Catalog output already exists; use force to replace: {output}"
        )
    resume_catalog, resume_state = _resume_paths(output)
    if not resume and (resume_catalog.exists() or resume_state.exists()):
        if not force:
            raise FileExistsError(
                "unfinished resumable EgoVerse build exists; use --resume "
                f"or --force: {resume_state}"
            )
        resume_catalog.unlink(missing_ok=True)
        resume_state.unlink(missing_ok=True)
    temporary_catalog = (
        resume_catalog
        if resume
        else output.with_name(f".{output.name}.tmp.{os.getpid()}")
    )
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.name}.tmp.{os.getpid()}"
    )
    temporary_quarantine = quarantine_output.with_name(
        f".{quarantine_output.name}.tmp.{os.getpid()}"
    )
    build_spec = {
        "catalog_id": str(catalog_id),
        "data_version": str(data_version),
        "source_child": child,
        "data_root": str(Path(data_root)),
        "profile": selected_profile.name,
        "max_episodes": max_episodes,
        "validate_numeric_finite": bool(validate_numeric_finite),
        "validate_jpeg": bool(validate_jpeg),
        "fail_on_episode_error": bool(fail_on_episode_error),
        "fail_on_store_error": bool(fail_on_store_error),
    }
    spec_digest = _stable_digest(build_spec)
    inventory_digest = _inventory_digest(scan)
    errors: Counter[str]
    error_examples: list[dict[str, Any]]
    quarantined_episodes: list[dict[str, Any]]
    quality: dict[str, Any]
    processed = 0
    written = 0
    elapsed_before = 0.0
    digest = hashlib.sha256()
    resume_existing = resume and (
        temporary_catalog.exists() or resume_state.exists()
    )
    if resume_existing:
        if not (temporary_catalog.is_file() and resume_state.is_file()):
            raise RuntimeError(
                "incomplete EgoVerse resume artifacts; expected both "
                f"{temporary_catalog} and {resume_state}"
            )
        state = json.loads(resume_state.read_text(encoding="utf-8"))
        if state.get("format") != "motus_egoverse_catalog_build_state_v1":
            raise RuntimeError("unsupported EgoVerse resume state format")
        if state.get("spec_sha256") != spec_digest:
            raise RuntimeError(
                "EgoVerse resume build parameters changed; choose a new "
                "output path or restart with --force"
            )
        if state.get("inventory_sha256") != inventory_digest:
            raise RuntimeError(
                "EgoVerse source inventory changed since the checkpoint"
            )
        processed = int(state["processed"])
        written = int(state["written"])
        if not 0 <= processed <= len(episodes):
            raise RuntimeError(
                f"invalid resume processed={processed} total={len(episodes)}"
            )
        catalog_offset = int(state["catalog_byte_offset"])
        with temporary_catalog.open("r+b") as handle:
            handle.truncate(catalog_offset)
            handle.flush()
            os.fsync(handle.fileno())
        digest, rows = _digest_file(temporary_catalog)
        if rows != written:
            raise RuntimeError(
                f"resume Catalog rows={rows} expected written={written}"
            )
        errors = Counter(
            {
                str(key): int(value)
                for key, value in (state.get("error_counts") or {}).items()
            }
        )
        error_examples = list(state.get("error_examples") or [])
        quarantined_episodes = list(
            state.get("quarantined_episodes") or []
        )
        quality = dict(state.get("quality") or _new_quality_stats())
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "resume",
                    "processed": processed,
                    "total": len(episodes),
                    "percent": 100.0 * processed / max(1, len(episodes)),
                    "written": written,
                    "skipped": processed - written,
                    "done": False,
                }
            )
    else:
        errors = Counter()
        error_examples = []
        quarantined_episodes = []
        quality = _new_quality_stats()
        temporary_catalog.unlink(missing_ok=True)
        resume_state.unlink(missing_ok=True)

    started = time.monotonic()
    try:
        mode = "r+b" if resume_existing else "xb"
        with temporary_catalog.open(mode) as handle:
            if resume_existing:
                handle.seek(0, os.SEEK_END)
            elif resume:
                _checkpoint_build(
                    handle=handle,
                    state_path=resume_state,
                    spec_digest=spec_digest,
                    inventory_digest=inventory_digest,
                    processed=0,
                    written=0,
                    errors=errors,
                    error_examples=error_examples,
                    quarantined_episodes=quarantined_episodes,
                    quality=quality,
                    elapsed_seconds=0.0,
                )
            total = len(episodes)
            for position in range(processed + 1, total + 1):
                episode = episodes[position - 1]
                try:
                    record = _episode_record(
                        episode,
                        selected_profile,
                        group_opener=opener,
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
                    _update_quality_stats(quality, record)
                except Exception as error:
                    errors[type(error).__name__] += 1
                    quarantine_item = {
                        "episode_id": episode.episode_id,
                        "zarr_path": episode.zarr_path,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    quarantined_episodes.append(quarantine_item)
                    if len(error_examples) < 20:
                        error_examples.append(dict(quarantine_item))
                    if fail_on_episode_error:
                        raise
                processed = position
                elapsed = (
                    elapsed_before + time.monotonic() - started
                )
                if resume and (
                    processed % int(checkpoint_interval) == 0
                    or processed == total
                ):
                    _checkpoint_build(
                        handle=handle,
                        state_path=resume_state,
                        spec_digest=spec_digest,
                        inventory_digest=inventory_digest,
                        processed=processed,
                        written=written,
                        errors=errors,
                        error_examples=error_examples,
                        quarantined_episodes=quarantined_episodes,
                        quality=quality,
                        elapsed_seconds=elapsed,
                    )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "validate",
                            "processed": position,
                            "total": total,
                            "percent": 100.0 * position / max(1, total),
                            "written": written,
                            "skipped": position - written,
                            "elapsed_seconds": elapsed,
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())
        if written <= 0:
            raise RuntimeError("EgoVerse Catalog builder produced zero Episodes")

        quarantine_digest = hashlib.sha256()
        with temporary_quarantine.open("xb") as handle:
            for item in quarantined_episodes:
                payload = (
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(payload)
                quarantine_digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())

        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": "egoverse",
            "child": child,
            "episodes": written,
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "build": {
                    "source_type": "raw_episode_zarr_v3",
                    "data_root": str(Path(data_root)),
                    "profile": selected_profile.name,
                    "zarr_candidate_count": scan.candidate_count,
                    "scanned_candidate_count": scan.scanned_count,
                    "matching_episodes": len(episodes),
                    "skipped_profile_count": scan.skipped_profile_count,
                    "scan_error_count": len(scan.issues),
                    "scan_error_examples": [
                        {
                            "zarr_path": issue.zarr_path,
                            "error_type": issue.error_type,
                            "error": issue.error,
                        }
                        for issue in scan.issues[:20]
                    ],
                    "candidate_episodes": len(episodes),
                    "written_episodes": written,
                    "skipped_episodes": len(episodes) - written,
                    "validate_numeric_finite": bool(validate_numeric_finite),
                    "validate_jpeg": bool(validate_jpeg),
                    "fail_on_episode_error": bool(fail_on_episode_error),
                    "fail_on_store_error": bool(fail_on_store_error),
                    "resumable_build": bool(resume),
                    "checkpoint_interval": int(checkpoint_interval),
                    "error_counts": dict(sorted(errors.items())),
                    "error_examples": error_examples,
                    "quarantine": {
                        "filename": quarantine_output.name,
                        "episodes": len(quarantined_episodes),
                        "sha256": quarantine_digest.hexdigest(),
                        "size_bytes": temporary_quarantine.stat().st_size,
                    },
                    "profile_contract": {
                        "state_dimension": selected_profile.state_dim,
                        "action_dimension": selected_profile.action_dim,
                        "state_semantics": selected_profile.state_semantics,
                        "action_semantics": selected_profile.action_semantics,
                        "action_source_signal": (
                            selected_profile.action_source_signal
                        ),
                        "action_is_control_command": (
                            selected_profile.action_is_control_command
                        ),
                        "indices": "model_provided_per_modality",
                    },
                    "quality": _finalize_quality_stats(quality),
                    "catalog_size_bytes": temporary_catalog.stat().st_size,
                    "elapsed_seconds": (
                        elapsed_before + time.monotonic() - started
                    ),
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
            quarantine_output.unlink(missing_ok=True)
        os.replace(temporary_catalog, output)
        os.replace(temporary_quarantine, quarantine_output)
        os.replace(temporary_metadata, metadata_output)
        resume_state.unlink(missing_ok=True)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "validate",
                    "processed": len(episodes),
                    "total": len(episodes),
                    "percent": 100.0,
                    "written": written,
                    "skipped": len(episodes) - written,
                    "elapsed_seconds": (
                        elapsed_before + time.monotonic() - started
                    ),
                    "done": True,
                }
            )
        return metadata
    finally:
        if not resume:
            temporary_catalog.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        temporary_quarantine.unlink(missing_ok=True)
