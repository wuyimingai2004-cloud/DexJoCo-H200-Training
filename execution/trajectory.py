"""JSONL trajectory log with versioned header and npz attachments.

Every record has a ``role`` from SEGMENT_ROLES so that measured data, final
commands and model predictions can never be confused when building training
manifests (predicted frames are never written as observations).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SCHEMA = "thinkerwam.trajectory/v1"
SEGMENT_ROLES = ("header", "measured", "history", "command", "forecast", "error", "decision", "progress",
                 "outcome", "segment_verification")


def action_hash(actions: np.ndarray) -> str:
    """Canonical command hash shared by inference and simulator-only logging."""
    values = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    return hashlib.sha256(values.tobytes() + str(values.shape).encode()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class TrajectoryLogger:
    def __init__(self, root: str | Path, episode_id: str, versions: Dict[str, Any]):
        for key in ("checkpoint", "action_stats_version", "camera_transform", "runtime"):
            if key not in versions:
                raise ValueError(f"trajectory versions lack {key!r}")
        self.dir = Path(root) / episode_id
        if (self.dir / "trajectory.jsonl").exists():
            raise FileExistsError(f"trajectory already exists: {self.dir}")
        (self.dir / "attachments").mkdir(parents=True, exist_ok=True)
        self._fh = open(self.dir / "trajectory.jsonl", "w", encoding="utf-8")
        self._n = 0
        self.write("header", schema=SCHEMA, episode_id=episode_id, versions=versions)

    def attach(self, name: str, **arrays) -> str:
        rel = f"attachments/{self._n:06d}_{name}.npz"
        np.savez_compressed(self.dir / rel, **{k: np.asarray(v) for k, v in arrays.items()})
        return rel

    def write(self, role: str, *, attachments: Optional[Dict[str, np.ndarray]] = None, **fields) -> None:
        if role not in SEGMENT_ROLES:
            raise ValueError(f"unknown segment role {role!r}")
        rec = dict(seq=self._n, role=role, **_jsonable(fields))
        if attachments:
            rec["attachment"] = self.attach(role, **attachments)
        self._fh.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
        self._fh.flush()
        self._n += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()


def read_trajectory(path: str | Path):
    with open(Path(path) / "trajectory.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
