"""Build split-neutral OpenVid complete-video Catalogs from CSV and ZIPs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_record_v2,
)
from motus_data.openvid import open_openvid_zip


ProgressCallback = Callable[[Mapping[str, Any]], None]
REQUIRED_CSV_COLUMNS = {
    "video",
    "caption",
    "frame",
    "fps",
    "seconds",
}
COMPLETE_ARCHIVE_PATTERN = re.compile(r"^OpenVid_part(\d+)\.zip$")
SPLIT_ARCHIVE_PATTERN = re.compile(
    r"^OpenVid_part(\d+)_part([a-z]+)$"
)


@dataclass(frozen=True)
class OpenVidArchive:
    number: int
    logical_name: str
    logical_path: str
    storage_type: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class OpenVidMember:
    archive: OpenVidArchive
    member: str
    file_size: int
    compress_size: int
    compress_type: int
    crc32: int
    header_offset: int


def _metadata_path(catalog: Path) -> Path:
    return catalog.with_suffix(".meta.json")


def _quarantine_path(catalog: Path) -> Path:
    return catalog.with_suffix(".quarantine.jsonl")


def _resume_paths(catalog: Path) -> tuple[Path, Path]:
    return (
        catalog.with_name(f".{catalog.name}.build.partial"),
        catalog.with_name(f".{catalog.name}.build.state.json"),
    )


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _parse_float(value: Any, *, field: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"OpenVid CSV {field} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"OpenVid CSV {field} is not finite")
    return result


def _parse_frame_count(value: Any) -> int:
    number = _parse_float(value, field="frame")
    result = int(number)
    if number != result or result <= 0:
        raise ValueError("OpenVid CSV frame must be a positive integer")
    return result


def discover_openvid_archives(
    data_root: str | Path,
    *,
    archive_numbers: Sequence[int] | None = None,
) -> tuple[OpenVidArchive, ...]:
    """Discover main OpenVid-1M archives, excluding duplicate OpenVidHD."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected = (
        None
        if archive_numbers is None
        else {int(value) for value in archive_numbers}
    )
    complete: dict[int, Path] = {}
    split: dict[int, list[tuple[str, Path]]] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = COMPLETE_ARCHIVE_PATTERN.match(path.name)
        if match:
            number = int(match.group(1))
            if selected is None or number in selected:
                complete[number] = path
            continue
        match = SPLIT_ARCHIVE_PATTERN.match(path.name)
        if match:
            number = int(match.group(1))
            if selected is None or number in selected:
                split.setdefault(number, []).append(
                    (match.group(2), path)
                )

    overlap = sorted(set(complete) & set(split))
    if overlap:
        raise ValueError(
            f"OpenVid archive has complete and split copies: {overlap}"
        )
    discovered = set(complete) | set(split)
    if selected is not None:
        missing = sorted(selected - discovered)
        if missing:
            raise FileNotFoundError(
                f"OpenVid archive numbers are missing: {missing}"
            )

    archives = []
    for number in sorted(discovered):
        logical_name = f"OpenVid_part{number}.zip"
        logical_path = str(root / logical_name)
        if number in complete:
            archives.append(
                OpenVidArchive(
                    number=number,
                    logical_name=logical_name,
                    logical_path=str(complete[number]),
                    storage_type="zip",
                    parts=(),
                )
            )
            continue
        ordered_parts = tuple(
            str(path)
            for _, path in sorted(split[number], key=lambda item: item[0])
        )
        if len(ordered_parts) < 2:
            raise ValueError(
                f"OpenVid split archive is incomplete: {logical_name}"
            )
        archives.append(
            OpenVidArchive(
                number=number,
                logical_name=logical_name,
                logical_path=logical_path,
                storage_type="zip_split",
                parts=ordered_parts,
            )
        )
    if not archives:
        raise RuntimeError("no OpenVid main archives were discovered")
    return tuple(archives)


def scan_openvid_zip_members(
    archives: Sequence[OpenVidArchive],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, OpenVidMember], dict[str, Any]]:
    """Index MP4 basenames across normal and split logical ZIPs."""

    members: dict[str, OpenVidMember] = {}
    duplicate_names = []
    non_video_members = 0
    started = time.monotonic()
    for position, archive in enumerate(archives, start=1):
        with open_openvid_zip(
            archive.logical_path,
            parts=archive.parts,
        ) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith(".mp4"):
                    non_video_members += 1
                    continue
                video_name = Path(info.filename).name
                if video_name in members:
                    duplicate_names.append(video_name)
                    continue
                members[video_name] = OpenVidMember(
                    archive=archive,
                    member=info.filename,
                    file_size=int(info.file_size),
                    compress_size=int(info.compress_size),
                    compress_type=int(info.compress_type),
                    crc32=int(info.CRC),
                    header_offset=int(info.header_offset),
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "zip_scan",
                    "processed": position,
                    "total": len(archives),
                    "percent": 100.0 * position / len(archives),
                    "discovered_episodes": len(members),
                    "duplicates": len(duplicate_names),
                    "elapsed_seconds": time.monotonic() - started,
                    "done": position == len(archives),
                }
            )
    if duplicate_names:
        raise ValueError(
            "OpenVid MP4 basenames are not globally unique: "
            f"{duplicate_names[:10]}"
        )
    return members, {
        "archives": len(archives),
        "complete_archives": sum(
            archive.storage_type == "zip" for archive in archives
        ),
        "split_archives": sum(
            archive.storage_type == "zip_split" for archive in archives
        ),
        "mp4_members": len(members),
        "non_video_members": non_video_members,
    }


def _storage(member: OpenVidMember) -> dict[str, Any]:
    result = {
        "type": member.archive.storage_type,
        "path": member.archive.logical_path,
        "member": member.member,
    }
    if member.archive.parts:
        result["parts"] = list(member.archive.parts)
    return result


def _optional_score(row: Mapping[str, Any], field: str) -> float | None:
    value = str(row.get(field) or "").strip()
    return None if not value else _parse_float(value, field=field)


def _episode_record(
    row: Mapping[str, Any],
    member: OpenVidMember,
) -> dict[str, Any]:
    video_name = str(row.get("video") or "").strip()
    caption = str(row.get("caption") or "").strip()
    if not video_name:
        raise ValueError("OpenVid CSV video is empty")
    frame_count = _parse_frame_count(row.get("frame"))
    fps = _parse_float(row.get("fps"), field="fps")
    seconds = _parse_float(row.get("seconds"), field="seconds")
    if fps <= 0 or seconds <= 0:
        raise ValueError("OpenVid fps and seconds must be positive")
    text_item = (
        {
            "index": 0,
            "kind": "caption_en",
            "text": caption,
            "meta": {"scope": "episode"},
        }
        if caption
        else {
            "index": 0,
            "kind": "missing_instruction",
            "text": "",
            "meta": {"missing": True},
        }
    )
    storage = _storage(member)
    record = {
        "episode_id": f"openvid:{video_name}",
        "robot": "openvid-null",
        "video": {
            "count": frame_count,
            "storage": storage,
            "meta": {
                "encoding": "mp4",
                "member_size": member.file_size,
                "compressed_size": member.compress_size,
                "compression_method": member.compress_type,
                "crc32": member.crc32,
                "header_offset": member.header_offset,
                "frame_count_source": "OpenVid-1M.csv",
            },
        },
        "state": {
            "count": 0,
            "storage": {"type": "none", "path": "none"},
            "meta": {"available": False, "dimension": 0},
        },
        "action": {
            "count": 0,
            "storage": {"type": "none", "path": "none"},
            "meta": {"available": False, "dimension": 0},
        },
        "texts": [text_item],
        "meta": {
            "video_filename": video_name,
            "zip_name": member.archive.logical_name,
            "fps": fps,
            "seconds": seconds,
            "aesthetic_score": _optional_score(row, "aesthetic score"),
            "motion_score": _optional_score(row, "motion score"),
            "temporal_consistency_score": _optional_score(
                row,
                "temporal consistency score",
            ),
            "camera_motion": str(row.get("camera motion") or "").strip(),
            "state_action_policy": "absent",
        },
    }
    validate_episode_record_v2(record)
    return record


def _new_quality() -> dict[str, Any]:
    return {
        "episodes": 0,
        "total_frames": 0,
        "min_frames": None,
        "max_frames": 0,
        "empty_captions": 0,
        "split_zip_episodes": 0,
        "camera_motion_counts": {},
    }


def _update_quality(
    quality: dict[str, Any],
    record: Mapping[str, Any],
) -> None:
    frames = int(record["video"]["count"])
    quality["episodes"] = int(quality["episodes"]) + 1
    quality["total_frames"] = int(quality["total_frames"]) + frames
    minimum = quality["min_frames"]
    quality["min_frames"] = (
        frames if minimum is None else min(int(minimum), frames)
    )
    quality["max_frames"] = max(int(quality["max_frames"]), frames)
    if record["texts"][0]["kind"] == "missing_instruction":
        quality["empty_captions"] = int(quality["empty_captions"]) + 1
    if record["video"]["storage"]["type"] == "zip_split":
        quality["split_zip_episodes"] = (
            int(quality["split_zip_episodes"]) + 1
        )
    camera_motion = str(
        (record.get("meta") or {}).get("camera_motion") or "missing"
    )
    counts = quality["camera_motion_counts"]
    counts[camera_motion] = int(counts.get(camera_motion, 0)) + 1


def _finalize_quality(quality: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(quality))
    episodes = int(result["episodes"])
    result["mean_frames"] = (
        float(result["total_frames"]) / episodes if episodes else 0.0
    )
    return result


def _digest_file(path: Path) -> tuple[Any, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for payload in handle:
            digest.update(payload)
            rows += 1
    return digest, rows


def _checkpoint(
    handle: Any,
    state_path: Path,
    *,
    spec_sha256: str,
    inventory_sha256: str,
    processed_rows: int,
    written: int,
    filtered_rows: int,
    errors: Mapping[str, int],
    quarantined: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    _atomic_json(
        state_path,
        {
            "format": "motus_openvid_catalog_build_state_v1",
            "spec_sha256": spec_sha256,
            "inventory_sha256": inventory_sha256,
            "processed_rows": int(processed_rows),
            "written": int(written),
            "filtered_rows": int(filtered_rows),
            "catalog_byte_offset": int(handle.tell()),
            "error_counts": dict(sorted(errors.items())),
            "quarantined": list(quarantined),
            "quality": quality,
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def build_openvid_episode_catalog(
    *,
    data_root: str | Path,
    csv_path: str | Path,
    output_path: str | Path,
    catalog_id: str,
    data_version: str,
    source_child: str = "openvid",
    archive_numbers: Sequence[int] | None = None,
    max_episodes: int | None = None,
    fail_on_row_error: bool = True,
    resume: bool = False,
    checkpoint_interval: int = 10000,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one complete-video OpenVid Catalog."""

    if resume and force:
        raise ValueError("OpenVid resume and force are mutually exclusive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)
    archives = discover_openvid_archives(
        data_root,
        archive_numbers=archive_numbers,
    )
    members, zip_stats = scan_openvid_zip_members(
        archives,
        progress_callback=progress_callback,
    )
    inventory = {
        "csv": {
            "path": str(csv_file),
            "size": csv_file.stat().st_size,
            "mtime_ns": csv_file.stat().st_mtime_ns,
        },
        "archives": [
            {
                "number": archive.number,
                "logical_path": archive.logical_path,
                "storage_type": archive.storage_type,
                "parts": [
                    {
                        "path": path,
                        "size": Path(path).stat().st_size,
                        "mtime_ns": Path(path).stat().st_mtime_ns,
                    }
                    for path in archive.parts
                ],
                "size": (
                    Path(archive.logical_path).stat().st_size
                    if not archive.parts
                    else sum(Path(path).stat().st_size for path in archive.parts)
                ),
            }
            for archive in archives
        ],
        "mp4_members": len(members),
    }
    inventory_sha256 = _stable_digest(inventory)
    spec = {
        "catalog_id": str(catalog_id),
        "data_version": str(data_version),
        "source_child": str(source_child),
        "data_root": str(Path(data_root)),
        "csv_path": str(csv_file),
        "archive_numbers": (
            None
            if archive_numbers is None
            else [int(value) for value in archive_numbers]
        ),
        "max_episodes": max_episodes,
        "fail_on_row_error": bool(fail_on_row_error),
    }
    spec_sha256 = _stable_digest(spec)

    output = Path(output_path)
    metadata_output = _metadata_path(output)
    quarantine_output = _quarantine_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    finals = (output, metadata_output, quarantine_output)
    if any(path.exists() for path in finals):
        if resume:
            raise FileExistsError(f"completed OpenVid Catalog exists: {output}")
        if not force:
            raise FileExistsError(
                f"OpenVid Catalog output exists; use force: {output}"
            )
    partial, state_path = _resume_paths(output)
    if not resume and (partial.exists() or state_path.exists()):
        if not force:
            raise FileExistsError(
                f"unfinished OpenVid build exists; use resume: {state_path}"
            )
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    temporary_catalog = (
        partial
        if resume
        else output.with_name(f".{output.name}.tmp.{os.getpid()}")
    )
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.name}.tmp.{os.getpid()}"
    )
    temporary_quarantine = quarantine_output.with_name(
        f".{quarantine_output.name}.tmp.{os.getpid()}"
    )

    processed_rows = 0
    written = 0
    filtered_rows = 0
    elapsed_before = 0.0
    errors: Counter[str] = Counter()
    quarantined: list[dict[str, Any]] = []
    quality = _new_quality()
    digest = hashlib.sha256()
    resume_existing = resume and (
        partial.exists() or state_path.exists()
    )
    if resume_existing:
        if not (partial.is_file() and state_path.is_file()):
            raise RuntimeError(
                "OpenVid resume requires both partial Catalog and state"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("format") != "motus_openvid_catalog_build_state_v1":
            raise RuntimeError("unsupported OpenVid resume state")
        if state.get("spec_sha256") != spec_sha256:
            raise RuntimeError("OpenVid resume parameters changed")
        if state.get("inventory_sha256") != inventory_sha256:
            raise RuntimeError("OpenVid source inventory changed")
        processed_rows = int(state["processed_rows"])
        written = int(state["written"])
        filtered_rows = int(state.get("filtered_rows", 0))
        with partial.open("r+b") as handle:
            handle.truncate(int(state["catalog_byte_offset"]))
            handle.flush()
            os.fsync(handle.fileno())
        digest, actual_rows = _digest_file(partial)
        if actual_rows != written:
            raise RuntimeError(
                f"OpenVid partial rows={actual_rows} expected={written}"
            )
        errors = Counter(
            {
                str(key): int(value)
                for key, value in (state.get("error_counts") or {}).items()
            }
        )
        quarantined = list(state.get("quarantined") or [])
        quality = dict(state.get("quality") or _new_quality())
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
    else:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    started = time.monotonic()
    seen: set[str] = set()
    matched_member_names: set[str] = set()
    mode = "r+b" if resume_existing else "xb"
    try:
        with temporary_catalog.open(mode) as destination:
            if resume_existing:
                destination.seek(0, os.SEEK_END)
            elif resume:
                _checkpoint(
                    destination,
                    state_path,
                    spec_sha256=spec_sha256,
                    inventory_sha256=inventory_sha256,
                    processed_rows=0,
                    written=0,
                    filtered_rows=0,
                    errors=errors,
                    quarantined=quarantined,
                    quality=quality,
                    elapsed_seconds=0.0,
                )
            with csv_file.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                missing_columns = sorted(
                    REQUIRED_CSV_COLUMNS - set(reader.fieldnames or ())
                )
                if missing_columns:
                    raise ValueError(
                        f"OpenVid CSV missing columns: {missing_columns}"
                    )
                for row_number, row in enumerate(reader, start=1):
                    video_name = str(row.get("video") or "").strip()
                    if row_number <= processed_rows:
                        if video_name:
                            seen.add(video_name)
                            if video_name in members:
                                matched_member_names.add(video_name)
                        if (
                            progress_callback is not None
                            and row_number % 10000 == 0
                        ):
                            progress_callback(
                                {
                                    "phase": "catalog_resume_seek",
                                    "processed": row_number,
                                    "written": written,
                                    "filtered": filtered_rows,
                                    "percent": min(
                                        99.99,
                                        100.0
                                        * source.buffer.tell()
                                        / csv_file.stat().st_size,
                                    ),
                                    "elapsed_seconds": (
                                        elapsed_before
                                        + time.monotonic()
                                        - started
                                    ),
                                }
                            )
                        continue
                    try:
                        if not video_name:
                            raise ValueError("OpenVid CSV video is empty")
                        if video_name in seen:
                            raise ValueError(
                                f"duplicate OpenVid CSV video={video_name!r}"
                            )
                        seen.add(video_name)
                        member = members.get(video_name)
                        if member is None and archive_numbers is not None:
                            filtered_rows += 1
                        else:
                            if member is None:
                                raise FileNotFoundError(
                                    "OpenVid video is absent from ZIPs: "
                                    f"{video_name}"
                                )
                            record = _episode_record(row, member)
                            payload = (
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                )
                                + "\n"
                            ).encode("utf-8")
                            destination.write(payload)
                            digest.update(payload)
                            written += 1
                            matched_member_names.add(video_name)
                            _update_quality(quality, record)
                    except Exception as error:
                        errors[type(error).__name__] += 1
                        quarantined.append(
                            {
                                "csv_row": row_number,
                                "video": video_name,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        if fail_on_row_error:
                            raise
                    processed_rows = row_number
                    elapsed = (
                        elapsed_before + time.monotonic() - started
                    )
                    if resume and (
                        processed_rows % int(checkpoint_interval) == 0
                    ):
                        _checkpoint(
                            destination,
                            state_path,
                            spec_sha256=spec_sha256,
                            inventory_sha256=inventory_sha256,
                            processed_rows=processed_rows,
                            written=written,
                            filtered_rows=filtered_rows,
                            errors=errors,
                            quarantined=quarantined,
                            quality=quality,
                            elapsed_seconds=elapsed,
                        )
                    if (
                        progress_callback is not None
                        and processed_rows % 1000 == 0
                    ):
                        progress_callback(
                            {
                                "phase": "catalog",
                                "processed": processed_rows,
                                "written": written,
                                "skipped": processed_rows - written,
                                "filtered": filtered_rows,
                                "percent": min(
                                    99.99,
                                    100.0
                                    * source.buffer.tell()
                                    / csv_file.stat().st_size,
                                ),
                                "elapsed_seconds": elapsed,
                            }
                        )
                    if (
                        max_episodes is not None
                        and written >= int(max_episodes)
                    ):
                        break
            destination.flush()
            os.fsync(destination.fileno())
            if resume:
                _checkpoint(
                    destination,
                    state_path,
                    spec_sha256=spec_sha256,
                    inventory_sha256=inventory_sha256,
                    processed_rows=processed_rows,
                    written=written,
                    filtered_rows=filtered_rows,
                    errors=errors,
                    quarantined=quarantined,
                    quality=quality,
                    elapsed_seconds=(
                        elapsed_before + time.monotonic() - started
                    ),
                )
        if written <= 0:
            raise RuntimeError("OpenVid Catalog produced zero Episodes")

        quarantine_digest = hashlib.sha256()
        with temporary_quarantine.open("xb") as handle:
            for item in quarantined:
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
        unmatched_members = len(set(members) - matched_member_names)
        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": "openvid",
            "child": str(source_child),
            "episodes": written,
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "build": {
                    "source_type": "openvid_csv_and_zip_inventory",
                    "data_root": str(Path(data_root)),
                    "csv_path": str(csv_file),
                    "processed_csv_rows": processed_rows,
                    "written_episodes": written,
                    "skipped_rows": processed_rows - written,
                    "archive_filtered_rows": filtered_rows,
                    "fail_on_row_error": bool(fail_on_row_error),
                    "archive_filter": (
                        None
                        if archive_numbers is None
                        else [int(value) for value in archive_numbers]
                    ),
                    "zip_inventory": zip_stats,
                    "matched_zip_members": len(matched_member_names),
                    "unmatched_zip_members": unmatched_members,
                    "error_counts": dict(sorted(errors.items())),
                    "error_examples": quarantined[:20],
                    "quarantine": {
                        "filename": quarantine_output.name,
                        "rows": len(quarantined),
                        "sha256": quarantine_digest.hexdigest(),
                        "size_bytes": temporary_quarantine.stat().st_size,
                    },
                    "quality": _finalize_quality(quality),
                    "catalog_size_bytes": temporary_catalog.stat().st_size,
                    "elapsed_seconds": (
                        elapsed_before + time.monotonic() - started
                    ),
                    "source_inventory_sha256": inventory_sha256,
                    "resumable_build": bool(resume),
                    "checkpoint_interval": int(checkpoint_interval),
                },
            },
        }
        validate_catalog_metadata_v2(metadata)
        with temporary_metadata.open("x", encoding="utf-8") as handle:
            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            for path in finals:
                path.unlink(missing_ok=True)
        os.replace(temporary_catalog, output)
        os.replace(temporary_quarantine, quarantine_output)
        os.replace(temporary_metadata, metadata_output)
        state_path.unlink(missing_ok=True)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "catalog",
                    "processed": processed_rows,
                    "written": written,
                    "skipped": processed_rows - written,
                    "filtered": filtered_rows,
                    "percent": 100.0,
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
