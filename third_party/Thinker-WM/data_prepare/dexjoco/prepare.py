"""Build separate train/dev catalogs, temporal pools and train-only statistics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contracts import CONTRACT, SOURCE, UPSTREAM_COMMIT, sha256, task_for, validate_manifest, write_json
from .raw import inspect_video, read_lowdim, video_path
from .lerobot import LeRobotV2


def _catalog(records: list[dict], directory: Path, split: str) -> None:
    from .reader import DexJoCoReader
    from motus_data.catalog.sqlite_index import build_catalog_offset_index
    from motus_data.index_contracts import validate_episode_record_v2, validate_catalog_metadata_v2
    from motus_data.repository import EpisodeRepository
    from ..sampling_ranges import build_sampling_ranges

    path = directory / f"{split}.episode_catalog.jsonl"
    with path.open("x") as stream:
        for row in records:
            validate_episode_record_v2(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    meta = dict(format="motus_episode_catalog_v2", catalog_id=f"{directory.name}_{split}",
                source=SOURCE, child=SOURCE, episodes=len(records), sha256=sha256(path),
                meta=dict(contract=CONTRACT, control_hz=30, split=split))
    validate_catalog_metadata_v2(meta)
    write_json(path.with_suffix(".meta.json"), meta)
    build_catalog_offset_index(path)
    repository = EpisodeRepository(catalog_path=path, readers={SOURCE: DexJoCoReader()})
    try:
        build_sampling_ranges(repository, directory / f"{split}.ranges.sqlite", dataset=SOURCE,
                              num_video_frames=12, action_fps_ratio=2, layout="video_stride",
                              action_start_offset=0, max_stride=1, min_episode_frames=25,
                              tail_pad_steps=0, workers=1)
    finally:
        repository.close()


def prepare_release(manifest: Path, output: Path, names: list[str], *, target_episodes: int = 100,
                    strict_layout: bool = True) -> dict:
    """No random split and no guessing missing metadata; never read test trajectories."""
    if target_episodes <= 0:
        raise ValueError("target_episodes must be positive")
    document = json.loads(manifest.read_text())
    # Resolve paths relative to the manifest, never relative to the launch cwd.
    for row in document.get("episodes", []):
        row["episode_dir"] = str((manifest.parent / row["episode_dir"]).resolve())
    rows = validate_manifest(document, strict_layout=strict_layout)
    tasks = [task_for(n) for n in names]
    if len(tasks) != len(set(tasks)):
        raise ValueError("duplicate task selection")
    for task in tasks:
        train = [r for r in rows if r["task"] == task.name and r["split"] == "train"]
        groups = {r["source_group"] for r in train}
        if len(groups) != target_episodes:
            raise ValueError(f"{task.name}: expected {target_episodes} original training trajectories, got {len(groups)}")
        if not any(r["task"] == task.name and r["split"] == "dev" for r in rows):
            raise ValueError(f"{task.name}: provide separate development demonstrations for 5k validation")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", document)
    reports = {}
    for task in tasks:
        directory = output / task.name
        directory.mkdir()
        records = {"train": [], "dev": []}
        total, mean, m2 = 0, np.zeros(task.dimension), np.zeros(task.dimension)
        audit = []
        datasets = {}
        for row in rows:
            if row["task"] != task.name or row["split"] == "test":
                continue
            root = Path(row["episode_dir"])
            data_format = row.get("format", "raw")
            if data_format == "lerobot_v2":
                key = (str(root), row["dataset_variant"])
                if key not in datasets:
                    datasets[key] = LeRobotV2(root, task, row["dataset_variant"])
                dataset = datasets[key]
                state, action, _, paths = dataset.read(row["episode_index"], row["instruction"])
                cameras = dataset.cameras
                if "cameras" in row and row["cameras"] != cameras:
                    raise ValueError("LeRobot camera order must match the declared dataset variant")
            else:
                state, action, _ = read_lowdim(root, task)
                cameras = row.get("cameras", list(task.cameras))
                paths = [video_path(root, camera) for camera in cameras]
            if (not isinstance(cameras, list) or len(cameras) != task.arms + 1
                    or len(set(cameras)) != len(cameras)
                    or any(Path(k).name != k for k in cameras)):
                raise ValueError("cameras must be base, right wrist, [left wrist], without duplicates")
            for path in paths:
                inspect_video(path, len(action))
            episode_id = f"dexjoco:{task.name}:{len(audit):05d}"
            episode = directory / f"episode_{len(audit):05d}"
            episode.mkdir()
            np.save(episode / "state.npy", state, allow_pickle=False)
            np.save(episode / "action.npy", action, allow_pickle=False)
            n = len(action)
            record = dict(
                episode_id=episode_id, robot=task.robot,
                video=dict(count=n, storage=dict(type="file", path=str(paths[0])),
                           meta=dict(camera_paths=list(map(str, paths)), cameras=cameras, control_hz=30)),
                texts=[dict(index=0, kind="instruction_en", text=row["instruction"],
                            video_range=[0, n], meta=dict(task_name=task.name))],
                meta=dict(task_name=task.name, scene_id=row["layout_id"],
                          source_group=row["source_group"], split=row["split"], contract=CONTRACT,
                          input_format=data_format, dataset_variant=row.get("dataset_variant")))
            for kind in ("state", "action"):
                record[kind] = dict(count=n, storage=dict(type="file", path=str(episode / f"{kind}.npy")),
                                    meta=dict(dimension=task.dimension, normalization="none",
                                              semantics="absolute_eef_rotvec_hand_joint"))
            records[row["split"]].append(record)
            # Stable parallel variance; statistics only see training actions.
            if row["split"] == "train":
                values = action.astype(np.float64)
                delta = values.mean(0) - mean
                count = len(values)
                m2 += ((values - values.mean(0)) ** 2).sum(0) + delta ** 2 * total * count / (total + count)
                mean += delta * count / (total + count)
                total += count
            audit.append(dict(episode_id=episode_id, split=row["split"], source_group=row["source_group"],
                              raw_path=str(root), rows=n, input_format=data_format,
                              dataset_variant=row.get("dataset_variant"), episode_index=row.get("episode_index"),
                              camera_sha256={str(p): sha256(p) for p in paths},
                              state_sha256=sha256(episode / "state.npy"), action_sha256=sha256(episode / "action.npy")))
        stats = dict(schema="dexjoco-action-stats/v1", task=task.name, robot=task.robot, contract=CONTRACT,
                     count=total, mean=mean.tolist(), std=np.maximum(np.sqrt(m2 / total), 1e-3).tolist(),
                     std_floor=1e-3, source_split="train", manifest_sha256=sha256(output / "manifest.json"))
        write_json(directory / "action_stats.json", stats)
        for split in records:
            _catalog(records[split], directory, split)
        write_json(directory / "episodes.json", audit)
        reports[task.name] = dict(train=len(records["train"]), dev=len(records["dev"]), train_steps=total)
    files = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    receipt = dict(schema="dexjoco-release/v1", contract=CONTRACT, upstream_commit=UPSTREAM_COMMIT,
                   strict_layout=strict_layout, target_episodes=target_episodes, tasks=reports, files=files)
    write_json(output / "release.json", receipt)
    return receipt


def verify_release(root: Path, task: str) -> dict:
    receipt = json.loads((root / "release.json").read_text())
    if receipt.get("schema") != "dexjoco-release/v1" or receipt.get("contract") != CONTRACT:
        raise ValueError("invalid or incomplete DexJoCo release")
    if task not in receipt["tasks"]:
        raise ValueError(f"task missing from prepared release: {task}")
    for relative, expected in receipt["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"prepared release changed: {relative}")
    return receipt
