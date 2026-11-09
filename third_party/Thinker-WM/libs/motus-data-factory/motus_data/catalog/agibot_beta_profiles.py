"""Atomically split an immutable Agibot Beta Catalog by effector profile."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping

from motus_data.agibot_beta_profiles import (
    AGIBOT_BETA_EFFECTOR_PROFILES,
    get_agibot_beta_effector_profile,
)
from motus_data.catalog.sqlite_index import build_catalog_offset_index
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _metadata_path(catalog: Path) -> Path:
    if catalog.suffix == ".jsonl":
        return catalog.with_suffix(".meta.json")
    return Path(str(catalog) + ".meta.json")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _normalization_profile_id(child: str) -> str:
    return f"{child}_native_zscore_v1"


def _profile_record(
    parent: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    profile = get_agibot_beta_effector_profile(str(audit["profile"]))
    record = copy.deepcopy(dict(parent))
    record["robot"] = profile.robot_name
    exact_shape = [
        int(value)
        for value in audit["shapes"]["state/effector/position"]
    ]
    if exact_shape != [int(audit["timesteps"]), profile.raw_effector_width]:
        raise ValueError(
            f"audit effector shape/profile mismatch for {parent['episode_id']}: "
            f"shape={exact_shape} profile={profile.name}"
        )
    if int(parent["state"]["count"]) != int(audit["timesteps"]):
        raise ValueError(
            f"parent state count differs from audit for {parent['episode_id']}"
        )
    if int(parent["action"]["count"]) != int(audit["timesteps"]):
        raise ValueError(
            f"parent action count differs from audit for {parent['episode_id']}"
        )

    schema_id = f"agibot_beta_{profile.name}_native_v1"
    normalization_id = _normalization_profile_id(profile.source_child)
    modality_meta = {
        "effector_profile": profile.name,
        "raw_effector_shape": exact_shape,
        "raw_effector_item_shape": [profile.raw_effector_width],
        "per_side_effector_dim": profile.per_side_effector_dim,
        "state_action_dim": profile.state_action_dim,
        "feature_schema": schema_id,
        "normalization_profile": normalization_id,
    }
    for modality in ("state", "action"):
        existing = copy.deepcopy(record[modality].get("meta") or {})
        existing.update(modality_meta)
        record[modality]["meta"] = existing
    record_meta = copy.deepcopy(record.get("meta") or {})
    record_meta.update(
        {
            "effector_profile": profile.name,
            "raw_effector_shape": exact_shape,
            "state_action_dim": profile.state_action_dim,
            "feature_schema": schema_id,
            "normalization_profile": normalization_id,
        }
    )
    record["meta"] = record_meta
    validate_episode_record_v2(record)
    return record


def _validate_audit_summary(
    summary: Mapping[str, Any],
    parent_metadata: Mapping[str, Any],
) -> None:
    if summary.get("format") != "agibot_beta_effector_profile_audit_v1":
        raise ValueError("unsupported Agibot Beta effector audit format")
    if str(summary.get("input_catalog_id")) != str(parent_metadata["catalog_id"]):
        raise ValueError("audit summary is bound to another parent Catalog ID")
    if str(summary.get("input_catalog_sha256")) != str(parent_metadata["sha256"]):
        raise ValueError("audit summary is bound to another parent Catalog SHA256")
    expected = int(parent_metadata["episodes"])
    if int(summary.get("audited_episodes", -1)) != expected:
        raise ValueError(
            f"audit rows={summary.get('audited_episodes')} expected={expected}"
        )
    statuses = summary.get("status_counts") or {}
    if int(statuses.get("ok", 0)) != expected or int(statuses.get("error", 0)):
        raise ValueError(f"audit summary is not all-success: {statuses}")


def build_agibot_beta_profile_catalogs(
    *,
    parent_catalog_path: str | Path,
    audit_jsonl_path: str | Path,
    audit_summary_path: str | Path,
    output_directory: str | Path,
    release: str,
    data_version: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Route every parent row into one of two disjoint child Catalogs."""

    started = time.monotonic()
    parent_catalog = Path(parent_catalog_path).resolve()
    parent_metadata_path = _metadata_path(parent_catalog)
    audit_path = Path(audit_jsonl_path).resolve()
    audit_summary_file = Path(audit_summary_path).resolve()
    output = Path(output_directory).resolve()
    for path in (
        parent_catalog,
        parent_metadata_path,
        audit_path,
        audit_summary_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(
            f"profile Catalog release directory already exists: {output}"
        )
    if not str(release).strip() or not str(data_version).strip():
        raise ValueError("release and data_version must be non-empty")

    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata(parent_metadata)
    if parent_metadata.get("format") != MOTUS_EPISODE_CATALOG_METADATA_V2:
        raise ValueError("profile split requires a concise v2 parent Catalog")
    if str(parent_metadata["source"]) != "agibot_beta":
        raise ValueError("profile split requires source=agibot_beta")
    audit_summary = json.loads(audit_summary_file.read_text(encoding="utf-8"))
    _validate_audit_summary(audit_summary, parent_metadata)

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary release directory exists: {temporary}")
    temporary.mkdir(parents=True)
    handles: dict[str, Any] = {}
    digests = {
        name: hashlib.sha256() for name in AGIBOT_BETA_EFFECTOR_PROFILES
    }
    counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[int]] = {
        name: Counter() for name in AGIBOT_BETA_EFFECTOR_PROFILES
    }
    parent_digest = hashlib.sha256()
    audit_digest = hashlib.sha256()
    parent_bytes = parent_catalog.stat().st_size
    paths: dict[str, Path] = {}
    try:
        for name, profile in AGIBOT_BETA_EFFECTOR_PROFILES.items():
            path = temporary / (
                f"{profile.source_child}.{release}.episode_catalog.jsonl"
            )
            paths[name] = path
            handles[name] = path.open("xb")

        with parent_catalog.open("rb") as parent_handle, audit_path.open(
            "rb"
        ) as audit_handle:
            for ordinal, pair in enumerate(
                zip_longest(parent_handle, audit_handle), start=1
            ):
                parent_payload, audit_payload = pair
                if parent_payload is None or audit_payload is None:
                    raise ValueError(
                        "parent Catalog and audit JSONL have different row counts"
                    )
                parent_digest.update(parent_payload)
                audit_digest.update(audit_payload)
                parent_record = json.loads(parent_payload.decode("utf-8"))
                audit_record = json.loads(audit_payload.decode("utf-8"))
                validate_episode_catalog_binding(parent_record, parent_metadata)
                if str(parent_record["episode_id"]) != str(
                    audit_record.get("episode_id")
                ):
                    raise ValueError(
                        f"parent/audit episode order differs at row {ordinal}: "
                        f"{parent_record['episode_id']} vs "
                        f"{audit_record.get('episode_id')}"
                    )
                if audit_record.get("status") != "ok":
                    raise ValueError(
                        f"audit row is not successful: {parent_record['episode_id']}"
                    )
                profile_name = str(audit_record["profile"])
                profile = get_agibot_beta_effector_profile(profile_name)
                record = _profile_record(parent_record, audit_record)
                payload = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                handles[profile_name].write(payload)
                digests[profile_name].update(payload)
                counts[profile_name] += 1
                task_counts[profile_name][int(record["meta"]["task_id"])] += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "processed": ordinal,
                            "total": int(parent_metadata["episodes"]),
                            "parent_bytes": parent_handle.tell(),
                            "parent_total_bytes": parent_bytes,
                            "profile_counts": dict(counts),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        expected_parent_sha = str(parent_metadata["sha256"])
        if parent_digest.hexdigest() != expected_parent_sha:
            raise ValueError("parent Catalog SHA256 changed during profile split")
        expected_total = int(parent_metadata["episodes"])
        if sum(counts.values()) != expected_total:
            raise ValueError(
                f"child union={sum(counts.values())} expected={expected_total}"
            )
        if set(counts) != set(AGIBOT_BETA_EFFECTOR_PROFILES):
            raise ValueError(f"one or more profiles are empty: {dict(counts)}")

        child_results: dict[str, Any] = {}
        for name, profile in AGIBOT_BETA_EFFECTOR_PROFILES.items():
            catalog_path = paths[name]
            catalog_id = f"{profile.source_child}_{release}"
            metadata = {
                "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
                "catalog_id": catalog_id,
                "source": "agibot_beta",
                "child": profile.source_child,
                "episodes": int(counts[name]),
                "sha256": digests[name].hexdigest(),
                "meta": {
                    "data_version": str(data_version),
                    "effector_profile": profile.name,
                    "robot_name": profile.robot_name,
                    "state_action_dim": profile.state_action_dim,
                    "raw_effector_width": profile.raw_effector_width,
                    "per_side_effector_dim": profile.per_side_effector_dim,
                    "feature_names": list(profile.feature_names),
                    "feature_schema": (
                        f"agibot_beta_{profile.name}_native_v1"
                    ),
                    "normalization_profile": _normalization_profile_id(
                        profile.source_child
                    ),
                    "parent_catalog": {
                        "catalog_id": str(parent_metadata["catalog_id"]),
                        "sha256": expected_parent_sha,
                        "episodes": expected_total,
                        "filename": parent_catalog.name,
                    },
                    "shape_audit": {
                        "format": str(audit_summary["format"]),
                        "sha256": audit_digest.hexdigest(),
                        "filename": audit_path.name,
                        "audited_episodes": int(
                            audit_summary["audited_episodes"]
                        ),
                    },
                    "task_counts": {
                        str(task): int(value)
                        for task, value in sorted(task_counts[name].items())
                    },
                },
            }
            validate_catalog_metadata_v2(metadata)
            _write_json(_metadata_path(catalog_path), metadata)
            index_path = catalog_path.with_suffix(".index.sqlite")
            index_result = build_catalog_offset_index(
                catalog_path, output_path=index_path
            )
            child_results[name] = {
                "catalog_id": catalog_id,
                "source_child": profile.source_child,
                "robot_name": profile.robot_name,
                "episodes": int(counts[name]),
                "state_action_dim": profile.state_action_dim,
                "catalog": catalog_path.name,
                "metadata": _metadata_path(catalog_path).name,
                "index": index_path.name,
                "sha256": digests[name].hexdigest(),
                "indexed_records": int(index_result["catalog_records"]),
            }

        report = {
            "format": "agibot_beta_profile_catalog_release_v1",
            "release": str(release),
            "data_version": str(data_version),
            "parent_catalog_id": str(parent_metadata["catalog_id"]),
            "parent_catalog_sha256": expected_parent_sha,
            "parent_episodes": expected_total,
            "audit_sha256": audit_digest.hexdigest(),
            "children": child_results,
            "union_episodes": sum(counts.values()),
            "disjoint_by_construction": True,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(temporary / "release_report.json", report)
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return report
    except BaseException:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
