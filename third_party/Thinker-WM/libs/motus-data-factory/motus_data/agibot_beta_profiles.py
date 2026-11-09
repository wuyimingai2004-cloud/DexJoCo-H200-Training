"""Explicit native effector profiles for Agibot Beta proprio data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


END_POSITION_KEY = "state/end/position"
END_ORIENTATION_KEY = "state/end/orientation"
EFFECTOR_POSITION_KEY = "state/effector/position"
AGIBOT_BETA_PROPRIO_KEYS = (
    END_POSITION_KEY,
    END_ORIENTATION_KEY,
    EFFECTOR_POSITION_KEY,
)

PARALLEL_GRIPPER = "parallel_gripper"
DEXTEROUS_HAND = "dexterous_hand"

# Official task-family information is a cross-check only. Runtime routing is
# always decided from the physical HDF5 schema of each Episode.
KNOWN_DEXTEROUS_TASK_IDS = frozenset(
    {
        475,
        536,
        549,
        554,
        577,
        578,
        595,
        608,
        620,
        622,
        660,
        679,
        705,
        710,
        727,
        730,
        731,
        749,
        753,
    }
)


@dataclass(frozen=True)
class AgibotBetaEffectorProfile:
    name: str
    source_child: str
    robot_name: str
    raw_effector_width: int
    per_side_effector_dim: int
    state_action_dim: int
    feature_names: tuple[str, ...]


def _feature_names(effector_names: tuple[str, ...]) -> tuple[str, ...]:
    result = []
    for side in ("left", "right"):
        result.extend(
            (
                f"{side}.position.x",
                f"{side}.position.y",
                f"{side}.position.z",
                f"{side}.rotation.rotvec_x",
                f"{side}.rotation.rotvec_y",
                f"{side}.rotation.rotvec_z",
            )
        )
        result.extend(f"{side}.{name}" for name in effector_names)
    return tuple(result)


AGIBOT_BETA_EFFECTOR_PROFILES: Mapping[str, AgibotBetaEffectorProfile] = {
    PARALLEL_GRIPPER: AgibotBetaEffectorProfile(
        name=PARALLEL_GRIPPER,
        source_child="agibot_beta_parallel_gripper",
        robot_name="agibot-beta-parallel-gripper",
        raw_effector_width=2,
        per_side_effector_dim=1,
        state_action_dim=14,
        feature_names=_feature_names(("gripper.position",)),
    ),
    DEXTEROUS_HAND: AgibotBetaEffectorProfile(
        name=DEXTEROUS_HAND,
        source_child="agibot_beta_dexterous_hand",
        robot_name="agibot-beta-dexterous-hand",
        raw_effector_width=12,
        per_side_effector_dim=6,
        state_action_dim=24,
        feature_names=_feature_names(
            tuple(f"hand.joint_{index}" for index in range(6))
        ),
    ),
}


def get_agibot_beta_effector_profile(
    name: str,
) -> AgibotBetaEffectorProfile:
    key = str(name).strip().lower()
    try:
        return AGIBOT_BETA_EFFECTOR_PROFILES[key]
    except KeyError as error:
        raise KeyError(
            f"unknown Agibot Beta effector profile={name!r}; "
            f"available={sorted(AGIBOT_BETA_EFFECTOR_PROFILES)}"
        ) from error


def expected_agibot_beta_profile_for_task(task_id: int) -> str:
    return (
        DEXTEROUS_HAND
        if int(task_id) in KNOWN_DEXTEROUS_TASK_IDS
        else PARALLEL_GRIPPER
    )


def classify_agibot_beta_effector_shape(
    shape: tuple[int, ...],
    *,
    timesteps: int,
) -> AgibotBetaEffectorProfile:
    normalized = tuple(int(value) for value in shape)
    if normalized == (int(timesteps), 2):
        return AGIBOT_BETA_EFFECTOR_PROFILES[PARALLEL_GRIPPER]
    if normalized == (int(timesteps), 12):
        return AGIBOT_BETA_EFFECTOR_PROFILES[DEXTEROUS_HAND]
    raise ValueError(
        "unsupported Agibot Beta state/effector/position shape: "
        f"got={normalized}, expected=({timesteps},2) or ({timesteps},12)"
    )


def _finite_statistics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    finite_values = array[finite].astype(np.float64, copy=False)
    result: dict[str, Any] = {
        "elements": int(array.size),
        "finite": int(finite.sum()),
        "nonfinite": int(array.size - int(finite.sum())),
    }
    if finite_values.size:
        result.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean(dtype=np.float64)),
                "std": float(finite_values.std(dtype=np.float64)),
            }
        )
    return result


def _per_column_statistics(values: np.ndarray) -> list[dict[str, Any]]:
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("per-column statistics require a timeline tensor")
    flattened = array.reshape(int(array.shape[0]), -1)
    return [
        {"column": int(index), **_finite_statistics(flattened[:, index])}
        for index in range(int(flattened.shape[1]))
    ]


def _inspect_agibot_beta_handle(
    handle: Any,
    *,
    numeric_mode: str,
) -> dict[str, Any]:
    mode = str(numeric_mode).strip().lower()
    if mode not in {"none", "effector", "all"}:
        raise ValueError("numeric_mode must be one of: none, effector, all")
    missing = [key for key in AGIBOT_BETA_PROPRIO_KEYS if key not in handle]
    if missing:
        raise ValueError(f"Agibot Beta proprio HDF5 missing keys: {missing}")
    shapes = {
        key: tuple(int(value) for value in handle[key].shape)
        for key in AGIBOT_BETA_PROPRIO_KEYS
    }
    dtypes = {key: str(handle[key].dtype) for key in AGIBOT_BETA_PROPRIO_KEYS}
    position_shape = shapes[END_POSITION_KEY]
    orientation_shape = shapes[END_ORIENTATION_KEY]
    if len(position_shape) != 3 or position_shape[1:] != (2, 3):
        raise ValueError(f"unexpected end position shape: {position_shape}")
    timesteps = int(position_shape[0])
    if timesteps <= 0:
        raise ValueError("Agibot Beta proprio timeline is empty")
    if orientation_shape != (timesteps, 2, 4):
        raise ValueError(
            f"unexpected end orientation shape: {orientation_shape}"
        )
    profile = classify_agibot_beta_effector_shape(
        shapes[EFFECTOR_POSITION_KEY], timesteps=timesteps
    )

    result: dict[str, Any] = {
        "timesteps": timesteps,
        "profile": profile.name,
        "source_child": profile.source_child,
        "robot_name": profile.robot_name,
        "state_action_dim": profile.state_action_dim,
        "raw_effector_width": profile.raw_effector_width,
        "per_side_effector_dim": profile.per_side_effector_dim,
        "shapes": {key: list(value) for key, value in shapes.items()},
        "dtypes": dtypes,
    }
    if mode in {"effector", "all"}:
        effector = np.asarray(handle[EFFECTOR_POSITION_KEY])
        result["effector_statistics"] = _finite_statistics(effector)
        result["effector_per_column_statistics"] = _per_column_statistics(
            effector
        )
    if mode == "all":
        positions = np.asarray(handle[END_POSITION_KEY])
        orientations = np.asarray(handle[END_ORIENTATION_KEY])
        result["position_statistics"] = _finite_statistics(positions)
        result["position_per_column_statistics"] = _per_column_statistics(
            positions
        )
        result["orientation_statistics"] = _finite_statistics(orientations)
        result["orientation_per_column_statistics"] = _per_column_statistics(
            orientations
        )
        norms = np.linalg.norm(
            orientations.astype(np.float64, copy=False), axis=-1
        )
        result["orientation_norm_statistics"] = _finite_statistics(norms)
    return result


def inspect_agibot_beta_proprio(
    source: str | Path | BinaryIO,
    *,
    numeric_mode: str = "none",
) -> dict[str, Any]:
    """Inspect one HDF5 payload without applying reshape or normalization."""

    if h5py is None:
        raise ImportError("Agibot Beta proprio inspection requires h5py")
    h5_source = str(source) if isinstance(source, (str, Path)) else source
    with h5py.File(h5_source, "r") as handle:
        return _inspect_agibot_beta_handle(handle, numeric_mode=numeric_mode)


def read_agibot_beta_native_timeline(
    source: str | Path | BinaryIO,
    *,
    expected_profile: str | None = None,
) -> tuple[np.ndarray, AgibotBetaEffectorProfile]:
    """Return native 14-D or 24-D rows with explicit left/right semantics."""

    if h5py is None or Rotation is None:
        raise ImportError("Agibot Beta native decoding requires h5py and scipy")
    h5_source = str(source) if isinstance(source, (str, Path)) else source
    with h5py.File(h5_source, "r") as handle:
        inspection = _inspect_agibot_beta_handle(handle, numeric_mode="none")
        profile = get_agibot_beta_effector_profile(inspection["profile"])
        if expected_profile is not None and profile.name != str(expected_profile):
            raise ValueError(
                "Agibot Beta effector profile differs from Catalog: "
                f"raw={profile.name} catalog={expected_profile}"
            )
        position = np.asarray(handle[END_POSITION_KEY], dtype=np.float64)
        quaternion = np.asarray(handle[END_ORIENTATION_KEY], dtype=np.float64)
        effector = np.asarray(handle[EFFECTOR_POSITION_KEY], dtype=np.float64)

    if not np.isfinite(position).all():
        raise ValueError("Agibot Beta end position contains non-finite values")
    if not np.isfinite(quaternion).all():
        raise ValueError("Agibot Beta end orientation contains non-finite values")
    if not np.isfinite(effector).all():
        raise ValueError("Agibot Beta effector position contains non-finite values")
    quaternion_norm = np.linalg.norm(quaternion, axis=-1)
    if np.any(quaternion_norm <= np.finfo(np.float64).eps):
        raise ValueError("Agibot Beta end orientation contains zero quaternion")

    timesteps = int(position.shape[0])
    rotation = Rotation.from_quat(quaternion.reshape(-1, 4)).as_rotvec()
    rotation = rotation.reshape(timesteps, 2, 3)
    if profile.name == PARALLEL_GRIPPER:
        per_side_effector = effector.reshape(timesteps, 2, 1)
    else:
        # The public dataset represents two 6-DoF hands as left-six followed
        # by right-six. The full-data audit must confirm this profile before
        # production release.
        per_side_effector = effector.reshape(timesteps, 2, 6)
    per_side = np.concatenate(
        (position, rotation, per_side_effector), axis=-1
    )
    timeline = per_side.reshape(timesteps, profile.state_action_dim)
    if timeline.shape != (timesteps, profile.state_action_dim):
        raise ValueError(
            "Agibot Beta native timeline shape mismatch: "
            f"got={timeline.shape} expected={(timesteps, profile.state_action_dim)}"
        )
    return timeline.astype(np.float32), profile
