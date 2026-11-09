"""Verified RoboDojo training-window annotations; never infer pairing from seed."""
from __future__ import annotations

import json
from pathlib import Path


def load_annotations(path):
    rows = {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("episode_id", "context_id", "instruction_id", "branch_id"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{number}: {key} must be a nonempty string")
            if type(row.get("start_frame")) is not int or row["start_frame"] < 0:
                raise ValueError(f"{path}:{number}: start_frame must be a nonnegative integer")
            for key in ("joint_valid", "history_required"):
                if type(row.get(key)) is not bool:
                    raise ValueError(f"{path}:{number}: {key} must be boolean")
            if row.get("outcome_source") != "simulator_recording":
                raise ValueError(f"{path}:{number}: outcome_source must be simulator_recording")
            key = (row["episode_id"], row["start_frame"])
            if key in rows:
                raise ValueError(f"{path}:{number}: duplicate window {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"empty capability annotations: {path}")
    return rows


class CapabilityAnnotations:
    def __init__(self, paths):
        self.rows = {}
        for path in dict.fromkeys(str(p) for p in paths):
            for key, row in load_annotations(path).items():
                if key in self.rows and self.rows[key] != row:
                    raise ValueError(f"conflicting capability annotation: {key}")
                self.rows[key] = row

    def lookup_batch(self, batch):
        episodes, frames = batch.get("episode_ids"), batch.get("frame_indices")
        if episodes is None or frames is None or len(episodes) != len(frames):
            raise ValueError("capability annotations need episode_ids and frame_indices")
        return [self.rows.get((str(ep), int(frame[0]))) for ep, frame in zip(episodes, frames)]


def annotation_paths(config):
    explicit = getattr(config.training, "capability_annotations_path", None)
    paths = [str(explicit)] if explicit else []
    for source in config.dataset.sources:
        pool = source.get("pool", {})
        if pool.get("type") == "capability_ranges":
            paths.append(str(pool["annotations_path"]))
    return list(dict.fromkeys(paths))
