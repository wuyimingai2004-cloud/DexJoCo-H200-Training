"""Read local DexJoCo LeRobot v2 episodes without installing the LeRobot stack.

Videos remain in place. Only selected state/action rows are materialized for Motus.
The caller must explicitly record rand_obj/rand_full and audited split membership.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contracts import Task, state_to_model


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("LeRobot path escapes dataset root")
    return path


class LeRobotV2:
    def __init__(self, root: Path, task: Task, variant: str):
        self.root, self.task = root, task
        self.info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        if self.info.get("codebase_version") not in ("v2.0", "v2.1"):
            raise ValueError("expected LeRobot v2.0/v2.1; v3 sharded videos are not supported")
        if self.info.get("fps") != 30:
            raise ValueError("DexJoCo LeRobot data must have 30 fps")
        if variant not in ("rand_obj", "rand_full"):
            raise ValueError("explicit dataset_variant rand_obj/rand_full is required")
        features = self.info["features"]
        for key, width in (("action", task.dimension), ("observation.state", 23 * task.arms)):
            if features[key]["shape"] != [width]:
                raise ValueError(f"{key} dimension disagrees with {task.name}")
        base = ("random_camera" if variant == "rand_full" else
                ("ego" if task.arms == 2 else "ego_right" if task.name == "click_mouse" else "front"))
        self.cameras = [base, "wrist"] if task.arms == 1 else [base, "wrist_right", "wrist_left"]
        for camera in self.cameras:
            key = "observation.images." + camera
            if features.get(key, {}).get("dtype") != "video":
                raise ValueError(f"missing {key}; check dataset_variant and task")
        self.episodes = {}
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            i = row["episode_index"]
            if type(i) is not int or i < 0 or i in self.episodes:
                raise ValueError("invalid/duplicate LeRobot episode_index")
            self.episodes[i] = row
        if len(self.episodes) != self.info["total_episodes"]:
            raise ValueError("LeRobot metadata episode count mismatch")

    def paths(self, index: int):
        if type(index) is not int or index not in self.episodes:
            raise ValueError("unknown LeRobot episode_index")
        values = dict(episode_index=index, episode_chunk=index // int(self.info["chunks_size"]))
        data = safe_path(self.root, self.info["data_path"].format(**values))
        videos = [safe_path(self.root, self.info["video_path"].format(
            **values, video_key="observation.images." + camera)) for camera in self.cameras]
        return data, videos

    def read(self, index: int, instruction: str):
        import pyarrow.parquet as pq
        meta = self.episodes[index]
        if instruction not in meta["tasks"]:
            raise ValueError("manifest instruction disagrees with LeRobot episode metadata")
        data_path, videos = self.paths(index)
        table = pq.read_table(data_path, columns=["observation.state", "action", "timestamp",
                                                 "episode_index", "frame_index"])
        n = len(table)
        if n != meta["length"] or n < 25:
            raise ValueError("LeRobot episode length mismatch or fewer than 25 frames")
        def values(key):
            return np.asarray(table[key].to_pylist())
        if not np.all(values("episode_index").reshape(-1) == index):
            raise ValueError("parquet contains a different episode")
        if not np.array_equal(values("frame_index").reshape(-1), np.arange(n)):
            raise ValueError("LeRobot frames must be contiguous and ordered from zero")
        times = values("timestamp").astype(np.float64).reshape(-1)
        if (not np.isfinite(times).all() or abs(times[0]) > 1e-4 or
                not np.allclose(times, np.arange(n) / 30, atol=1e-4, rtol=1e-6)):
            raise ValueError("LeRobot timestamps must be aligned at 30 Hz")
        state = state_to_model(values("observation.state"), self.task.arms)
        action = values("action").astype(np.float32)
        if state.shape != (n, self.task.dimension) or action.shape != state.shape or not np.isfinite(action).all():
            raise ValueError("invalid LeRobot state/action arrays")
        for video in videos:
            if not video.is_file():
                raise FileNotFoundError(video)
        return state, action, times, videos
