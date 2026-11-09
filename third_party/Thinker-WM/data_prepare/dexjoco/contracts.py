"""Official DexJoCo action/state boundary; no simulator or torch imports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = "dexjoco"
CONTRACT = "dexjoco-absolute-eef-wxyz-to-rotvec-right-first/v1"
UPSTREAM_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"


@dataclass(frozen=True)
class Task:
    label: str
    name: str
    arms: int
    seed: int

    @property
    def dimension(self) -> int:
        return self.arms * 22

    @property
    def robot(self) -> str:
        return f"dexjoco_{self.name}"

    @property
    def cameras(self) -> tuple[str, ...]:
        return (("random_camera", "wrist") if self.arms == 1 else
                ("random_camera", "wrist_right", "wrist_left"))


TASKS = tuple(Task(*row) for row in (
    ("PickBucket", "pick_bucket", 1, 2026110901),
    ("Microwave", "bimanual_microwave_cook", 2, 2026110902),
    ("ClickMouse", "click_mouse", 1, 2026110903),
    ("FoldGlasses", "fold_glasses", 1, 2026110904),
    ("Hanoi", "bimanual_hanoi", 2, 2026110905),
    ("UnlockiPad", "bimanual_unlock_ipad", 2, 2026110906),
))


def task_for(name: str) -> Task:
    for task in TASKS:
        if name in (task.name, task.label):
            return task
    raise ValueError(f"unknown DexJoCo task: {name!r}")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def quat_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """wxyz quaternions to principal rotation vectors, including q == -q."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("expected finite wxyz quaternions")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or np.any(np.abs(norm - 1) > 0.02):
        raise ValueError("invalid quaternion norm")
    q = q / norm
    # At pi, use a deterministic sign for the largest vector component.
    pivot = np.take_along_axis(q[..., 1:], np.argmax(np.abs(q[..., 1:]), axis=-1)[..., None], -1)
    flip = (q[..., :1] < 0) | ((q[..., :1] == 0) & (pivot < 0))
    q = np.where(flip, -q, q)
    sine = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sine, q[..., :1])
    scale = np.divide(angle, sine, out=np.full_like(angle, 2.), where=sine > 1e-8)
    return (q[..., 1:] * scale).astype(np.float32)


def state_to_model(state: np.ndarray, arms: int) -> np.ndarray:
    """Take proprioception only; reorder both arms to the policy action layout.

    Native dual: [r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand, privileged...].
    Model dual: [r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand].
    """
    state = np.asarray(state)
    width = 23 if arms == 1 else 46
    if arms not in (1, 2) or state.ndim < 1 or state.shape[-1] < width:
        raise ValueError("native state must contain 23/46 proprioceptive dimensions")
    s = state[..., :width]
    if not np.isfinite(s).all():
        raise ValueError("nonfinite proprioceptive state")
    if arms == 1:
        pieces = (s[..., :3], quat_to_rotvec(s[..., 3:7]), s[..., 7:23])
    else:
        pieces = (s[..., :3], quat_to_rotvec(s[..., 3:7]), s[..., 14:30],
                  s[..., 7:10], quat_to_rotvec(s[..., 10:14]), s[..., 30:46])
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def action_to_environment(action: np.ndarray, arms: int) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64)
    if arms not in (1, 2) or a.shape[-1] != 22 * arms or not np.isfinite(a).all():
        raise ValueError("expected finite 22/44-dimensional policy action")

    def quat(v):
        angle = np.linalg.norm(v, axis=-1, keepdims=True)
        scale = np.divide(np.sin(angle / 2), angle,
                          out=np.full_like(angle, .5), where=angle > 1e-8)
        return np.concatenate((np.cos(angle / 2), v * scale), -1)

    right = (a[..., :3], quat(a[..., 3:6]))
    if arms == 1:
        pieces = (*right, a[..., 6:22])
    else:
        pieces = (*right, a[..., 22:25], quat(a[..., 25:28]), a[..., 6:22], a[..., 28:44])
    return np.concatenate(pieces, -1).astype(np.float32)


def validate_manifest(document: dict, *, strict_layout: bool = True) -> list[dict]:
    if document.get("schema") != "dexjoco-training-manifest/v1":
        raise ValueError("expected dexjoco-training-manifest/v1")
    if document.get("alignment") != "obs_t_action_t":
        raise ValueError("manifest must attest obs_t_action_t alignment")
    rows = document.get("episodes", [])
    if not rows:
        raise ValueError("manifest has no episodes")
    paths, groups, layouts = set(), {}, {}
    for row in rows:
        task = task_for(row["task"])
        if row["task"] != task.name:
            raise ValueError("manifest task names must use official identifiers")
        split = row["split"]
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev, or test")
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("episode_dir", "source_group", "layout_id", "instruction")):
            raise ValueError("every episode needs path, source_group, layout_id, and instruction")
        path = str(Path(row["episode_dir"]).resolve())
        if path in paths:
            raise ValueError(f"duplicate episode path: {path}")
        paths.add(path)
        # Camera and augmentation variants retain the original source_group.
        key = (task.name, row["source_group"])
        if key in groups and groups[key] != split:
            raise ValueError(f"original trajectory crosses splits: {key}")
        groups[key] = split
        key = (task.name, row["layout_id"])
        if strict_layout and key in layouts and layouts[key] != split:
            raise ValueError(f"layout crosses splits: {key}")
        layouts[key] = split
    return rows
