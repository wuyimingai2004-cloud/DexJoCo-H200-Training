"""Catalog-bound normalization statistics for native Agibot Beta profiles."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from motus_data.agibot_beta_profiles import (
    AgibotBetaEffectorProfile,
    get_agibot_beta_effector_profile,
    read_agibot_beta_native_timeline,
)
from motus_data.catalog.sqlite_index import catalog_metadata_path
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata,
    validate_episode_catalog_binding,
)
from motus_data.storage import BoundedFileReader


AGIBOT_BETA_NORMALIZATION_FORMAT = "motus_agibot_beta_normalization_v1"
AGIBOT_BETA_NORMALIZATION_STATE_FORMAT = (
    "motus_agibot_beta_normalization_scan_state_v1"
)
DEFAULT_CONSTANT_STD_THRESHOLD = 1.0e-3
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class AgibotBetaNormalizationArtifact:
    path: Path
    sha256: str
    profile: AgibotBetaEffectorProfile
    normalization_profile: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    applied_scale: np.ndarray

    def normalize(self, timeline: np.ndarray) -> np.ndarray:
        matrix = np.asarray(timeline, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.profile.state_action_dim:
            raise ValueError(
                "Agibot Beta timeline dimension differs from normalization "
                f"artifact: {matrix.shape}"
            )
        normalized = (matrix - self.mean) / self.applied_scale
        if not np.isfinite(normalized).all():
            raise ValueError("normalized Agibot Beta timeline is non-finite")
        return normalized.astype(np.float32)


class RunningVectorStatistics:
    """Merge complete Episode matrices without retaining individual rows."""

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("statistics dimension must be positive")
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)
        self.minimum = np.full(self.dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dimension, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(
                f"statistics input must have shape (N,{self.dimension}), "
                f"got={matrix.shape}"
            )
        if matrix.shape[0] <= 0:
            raise ValueError("statistics input cannot be empty")
        if not np.isfinite(matrix).all():
            raise ValueError("statistics input contains non-finite values")
        batch_count = int(matrix.shape[0])
        batch_mean = matrix.mean(axis=0, dtype=np.float64)
        centered = matrix - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            combined = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (batch_count / combined)
            self.m2 = (
                self.m2
                + batch_m2
                + delta * delta * self.count * batch_count / combined
            )
            self.count = combined
        self.minimum = np.minimum(self.minimum, matrix.min(axis=0))
        self.maximum = np.maximum(self.maximum, matrix.max(axis=0))

    def to_state(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }

    @classmethod
    def from_state(
        cls, dimension: int, state: Mapping[str, Any]
    ) -> "RunningVectorStatistics":
        result = cls(dimension)
        result.count = int(state["count"])
        for name in ("mean", "m2", "minimum", "maximum"):
            array = np.asarray(state[name], dtype=np.float64)
            if array.shape != (result.dimension,) or not np.isfinite(array).all():
                raise ValueError(f"invalid normalization checkpoint field={name}")
            setattr(result, name, array)
        if result.count < 0 or np.any(result.m2 < -1.0e-9):
            raise ValueError("invalid normalization checkpoint statistics")
        return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _feature_schema(profile: AgibotBetaEffectorProfile) -> str:
    return f"agibot_beta_{profile.name}_native_v1"


def _normalization_profile(profile: AgibotBetaEffectorProfile) -> str:
    return f"{profile.source_child}_native_zscore_v1"


def _state_storage(record: Mapping[str, Any]) -> Mapping[str, Any]:
    storage = record["state"]["storage"]
    if storage.get("type") != "tar":
        raise ValueError("Agibot Beta normalization requires tar state storage")
    missing = [
        key for key in ("path", "member", "offset", "size") if key not in storage
    ]
    if missing:
        raise ValueError(f"Agibot Beta state storage missing fields: {missing}")
    return storage


def _validate_profile_catalog(
    metadata: Mapping[str, Any],
) -> AgibotBetaEffectorProfile:
    validate_catalog_metadata(metadata)
    if metadata.get("format") != MOTUS_EPISODE_CATALOG_METADATA_V2:
        raise ValueError("normalization scan requires a concise v2 Catalog")
    if str(metadata.get("source")) != "agibot_beta":
        raise ValueError("normalization scan requires source=agibot_beta")
    profile = get_agibot_beta_effector_profile(
        str((metadata.get("meta") or {}).get("effector_profile"))
    )
    if str(metadata["child"]) != profile.source_child:
        raise ValueError("Catalog child does not match effector profile")
    meta = metadata.get("meta") or {}
    if int(meta.get("state_action_dim", -1)) != profile.state_action_dim:
        raise ValueError("Catalog state/action dimension does not match profile")
    if str(meta.get("feature_schema")) != _feature_schema(profile):
        raise ValueError("Catalog feature schema does not match profile")
    if str(meta.get("normalization_profile")) != _normalization_profile(profile):
        raise ValueError("Catalog normalization profile does not match profile")
    return profile


def _checkpoint_binding(
    metadata: Mapping[str, Any], profile: AgibotBetaEffectorProfile
) -> dict[str, Any]:
    return {
        "catalog_id": str(metadata["catalog_id"]),
        "catalog_sha256": str(metadata["sha256"]),
        "catalog_episodes": int(metadata["episodes"]),
        "source_child": profile.source_child,
        "effector_profile": profile.name,
        "feature_schema": _feature_schema(profile),
        "dimension": profile.state_action_dim,
    }


def load_agibot_beta_normalization_artifact(
    path: str | Path,
    *,
    catalog_metadata: Mapping[str, Any],
    expected_profile: str,
) -> AgibotBetaNormalizationArtifact:
    """Load and strictly bind one artifact to one profile Catalog."""

    artifact_path = Path(path).resolve()
    digest_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
    if not artifact_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError(
            f"normalization artifact requires {artifact_path} and {digest_path}"
        )
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    published_digest = digest_path.read_text(encoding="ascii").split()
    if not published_digest or published_digest[0] != digest:
        raise ValueError("normalization artifact SHA256 sidecar mismatch")
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    if value.get("format") != AGIBOT_BETA_NORMALIZATION_FORMAT:
        raise ValueError("unsupported Agibot Beta normalization format")
    profile = get_agibot_beta_effector_profile(expected_profile)
    if str(value.get("effector_profile")) != profile.name:
        raise ValueError("normalization artifact effector profile mismatch")
    if str(value.get("robot_name")) != profile.robot_name:
        raise ValueError("normalization artifact robot name mismatch")
    if str(value.get("feature_schema")) != _feature_schema(profile):
        raise ValueError("normalization artifact feature schema mismatch")
    if str(value.get("normalization_profile")) != _normalization_profile(profile):
        raise ValueError("normalization profile ID mismatch")
    expected_catalog = {
        "catalog_id": str(catalog_metadata["catalog_id"]),
        "sha256": str(catalog_metadata["sha256"]),
        "source": str(catalog_metadata["source"]),
        "child": str(catalog_metadata["child"]),
        "episodes": int(catalog_metadata["episodes"]),
    }
    if value.get("catalog") != expected_catalog:
        raise ValueError("normalization artifact is bound to another Catalog")
    dimension = profile.state_action_dim
    if int(value.get("dimension", -1)) != dimension:
        raise ValueError("normalization artifact dimension mismatch")
    feature_names = tuple(str(item) for item in value.get("feature_names") or ())
    if feature_names != profile.feature_names:
        raise ValueError("normalization artifact feature names mismatch")
    mean = np.asarray(value.get("mean"), dtype=np.float64)
    scale = np.asarray(value.get("applied_scale"), dtype=np.float64)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or int(value.get("nonfinite_values", -1)) != 0
    ):
        raise ValueError("normalization artifact vectors are invalid")
    return AgibotBetaNormalizationArtifact(
        path=artifact_path,
        sha256=digest,
        profile=profile,
        normalization_profile=str(value["normalization_profile"]),
        feature_names=feature_names,
        mean=mean,
        applied_scale=scale,
    )


def build_agibot_beta_normalization_artifact(
    *,
    catalog_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
    constant_std_threshold: float = DEFAULT_CONSTANT_STD_THRESHOLD,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Scan a complete child Catalog and publish one bound z-score artifact."""

    catalog = Path(catalog_path).resolve()
    metadata_path = catalog_metadata_path(catalog)
    output = Path(output_path).resolve()
    checkpoint = (
        output.with_suffix(output.suffix + ".scan_state.json")
        if checkpoint_path is None
        else Path(checkpoint_path).resolve()
    )
    if not catalog.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"normalization scan requires {catalog} and {metadata_path}"
        )
    if output.exists():
        raise FileExistsError(f"normalization artifact exists: {output}")
    if int(checkpoint_every) <= 0:
        raise ValueError("checkpoint_every must be positive")
    threshold = float(constant_std_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("constant_std_threshold must be finite and positive")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile = _validate_profile_catalog(metadata)
    binding = _checkpoint_binding(metadata, profile)
    stats = RunningVectorStatistics(profile.state_action_dim)
    processed = 0
    last_episode_id: str | None = None
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(
                f"normalization checkpoint exists; use resume: {checkpoint}"
            )
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("format") != AGIBOT_BETA_NORMALIZATION_STATE_FORMAT:
            raise ValueError("unsupported normalization checkpoint format")
        if state.get("binding") != binding:
            raise ValueError("normalization checkpoint is bound to another Catalog")
        processed = int(state["processed_episodes"])
        last_episode_id = state.get("last_episode_id")
        stats = RunningVectorStatistics.from_state(
            profile.state_action_dim, state["statistics"]
        )
    elif resume:
        raise FileNotFoundError(f"normalization checkpoint not found: {checkpoint}")

    expected_episodes = int(metadata["episodes"])
    if not 0 <= processed <= expected_episodes:
        raise ValueError("normalization checkpoint episode count is out of range")
    started = time.monotonic()
    catalog_size = catalog.stat().st_size

    def write_checkpoint(*, completed: bool) -> None:
        _atomic_json(
            checkpoint,
            {
                "format": AGIBOT_BETA_NORMALIZATION_STATE_FORMAT,
                "binding": binding,
                "processed_episodes": processed,
                "last_episode_id": last_episode_id,
                "completed": bool(completed),
                "statistics": stats.to_state(),
            },
        )

    with catalog.open("rb") as handle:
        for ordinal, payload in enumerate(handle):
            record = json.loads(payload.decode("utf-8"))
            validate_episode_catalog_binding(record, metadata)
            if ordinal < processed:
                if ordinal == processed - 1 and str(record["episode_id"]) != str(
                    last_episode_id
                ):
                    raise ValueError(
                        "normalization checkpoint last Episode differs from Catalog"
                    )
                continue
            record_meta = record.get("meta") or {}
            if str(record_meta.get("effector_profile")) != profile.name:
                raise ValueError(
                    f"record profile mismatch: {record['episode_id']}"
                )
            if int(record_meta.get("state_action_dim", -1)) != profile.state_action_dim:
                raise ValueError(
                    f"record dimension mismatch: {record['episode_id']}"
                )
            storage = _state_storage(record)
            with BoundedFileReader(
                storage["path"], storage["offset"], storage["size"]
            ) as member:
                timeline, decoded_profile = read_agibot_beta_native_timeline(
                    member, expected_profile=profile.name
                )
            expected_count = int(record["state"]["count"])
            if (
                timeline.shape != (expected_count, profile.state_action_dim)
                or int(record["action"]["count"]) != expected_count
            ):
                raise ValueError(
                    f"native timeline differs from Catalog: {record['episode_id']}"
                )
            if decoded_profile != profile:
                raise ValueError(
                    f"decoded profile mismatch: {record['episode_id']}"
                )
            stats.update(timeline)
            processed += 1
            last_episode_id = str(record["episode_id"])
            if processed % int(checkpoint_every) == 0:
                write_checkpoint(completed=False)
            if progress_callback is not None:
                progress_callback(
                    {
                        "processed_episodes": processed,
                        "total_episodes": expected_episodes,
                        "timesteps": stats.count,
                        "catalog_bytes": handle.tell(),
                        "catalog_total_bytes": catalog_size,
                        "elapsed_seconds": time.monotonic() - started,
                        "profile": profile.name,
                    }
                )

    if processed != expected_episodes or stats.count <= 0:
        raise ValueError(
            f"normalization scan incomplete: processed={processed} "
            f"expected={expected_episodes} timesteps={stats.count}"
        )
    variance = np.maximum(stats.m2 / stats.count, 0.0)
    empirical_std = np.sqrt(variance)
    constant_mask = empirical_std < threshold
    applied_scale = np.where(constant_mask, 1.0, empirical_std)
    artifact = {
        "format": AGIBOT_BETA_NORMALIZATION_FORMAT,
        "normalization_profile": _normalization_profile(profile),
        "catalog": {
            "catalog_id": str(metadata["catalog_id"]),
            "sha256": str(metadata["sha256"]),
            "source": str(metadata["source"]),
            "child": str(metadata["child"]),
            "episodes": expected_episodes,
        },
        "effector_profile": profile.name,
        "robot_name": profile.robot_name,
        "feature_schema": _feature_schema(profile),
        "feature_names": list(profile.feature_names),
        "dimension": profile.state_action_dim,
        "statistics_scope": "complete_child_catalog",
        "episodes": processed,
        "timesteps": int(stats.count),
        "nonfinite_values": 0,
        "mean": stats.mean.tolist(),
        "empirical_std": empirical_std.tolist(),
        "minimum": stats.minimum.tolist(),
        "maximum": stats.maximum.tolist(),
        "constant_std_threshold": threshold,
        "constant_mask": constant_mask.tolist(),
        "applied_scale": applied_scale.tolist(),
        "transform": "normalized = (raw - mean) / applied_scale",
        "constant_dimension_policy": (
            "center only: applied_scale=1 when empirical_std is below threshold"
        ),
        "state_action_share_timeline_and_statistics": True,
    }
    _atomic_json(output, artifact)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_temporary = digest_path.with_name(
        f".{digest_path.name}.tmp.{os.getpid()}"
    )
    with digest_temporary.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(digest_temporary, digest_path)
    write_checkpoint(completed=True)
    return {
        "artifact": str(output),
        "artifact_sha256": digest,
        "profile": profile.name,
        "episodes": processed,
        "timesteps": int(stats.count),
        "dimension": profile.state_action_dim,
        "constant_dimensions": int(constant_mask.sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
