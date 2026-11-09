"""Prepare one local LeRobot task and a new config; never launch training."""
from __future__ import annotations

import logging
from pathlib import Path
import re

import yaml

from .contracts import sha256, task_for, validate_manifest, write_json
from .lerobot import LeRobotV2
from .prepare import prepare_release
from .recipe import BASE, write_configs


def setup_task(*, task: str, data: Path, work_dir: Path, name_prefix: str,
               seed: int, no_dev: bool, variant: str = "rand_obj",
               assets_from: Path = BASE, target_episodes: int = 100,
               batch_size: int = 8) -> dict:
    """All supplied episodes are train data; no held-out layout claim is made.

    Asset paths are read from a caller-selected config and checked before writes.
    A fresh work directory keeps failed/previous preparations intact.
    """
    from .launch import preflight_assets, resolve_weights, validate_recipe

    selected = task_for(task)
    if not no_dev:
        raise ValueError("setup-task requires explicit --no-dev; use prepare for audited train/dev splits")
    if type(seed) is not int or seed <= 0:
        raise ValueError("supply a new positive training seed")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name_prefix):
        raise ValueError("name-prefix must be a simple filename component")
    if batch_size not in (4, 8):
        raise ValueError("batch_size must be 4 or 8")
    work_dir, data, assets_from = work_dir.resolve(), data.resolve(), assets_from.resolve()
    if work_dir.exists():
        raise FileExistsError(f"choose a new work directory; preserving {work_dir}")
    reference = yaml.safe_load(assets_from.read_text())
    # These are paths, not models to load; no CUDA allocation or pickle execution here.
    def asset(value):
        return (assets_from.parent / value).resolve()
    wan = asset(reference["model"]["wan"]["checkpoint_path"])
    vlm = asset(reference["model"]["vlm"]["checkpoint_path"])
    checkpoint = asset(reference["finetune"]["checkpoint_path"])
    weights = resolve_weights(checkpoint)
    for path in (wan / "config.json", wan / "Wan2.2_VAE.pth", vlm / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = LeRobotV2(data, selected, variant)
    if target_episodes <= 0 or len(dataset.episodes) != target_episodes:
        raise ValueError(f"expected exactly {target_episodes} episodes, got {len(dataset.episodes)}")
    rows = []
    for index, episode in sorted(dataset.episodes.items()):
        instructions = episode["tasks"]
        if len(instructions) != 1 or not isinstance(instructions[0], str) or not instructions[0].strip():
            raise ValueError(f"episode {index}: expected one explicit task instruction")
        rows.append(dict(task=selected.name, format="lerobot_v2", dataset_variant=variant,
                         episode_dir=str(data), episode_index=index, split="train",
                         source_group=f"lerobot:{selected.name}:{index}",
                         layout_id="unverified", instruction=instructions[0]))
    document = dict(schema="dexjoco-training-manifest/v1", alignment="obs_t_action_t", episodes=rows,
                    layout_audit="not_provided", evaluation="separate_simulator_rollouts_not_run",
                    source_group_basis="LeRobot episode identity; original augmentation relation unverified",
                    source_metadata_sha256={name: sha256(data / "meta" / name)
                                            for name in ("info.json", "episodes.jsonl")})
    validate_manifest(document, strict_layout=False)
    logging.getLogger(__name__).info("Found model file: %s; preparing %d training episodes", weights, len(rows))
    work_dir.mkdir(parents=True, exist_ok=False)
    manifest = work_dir / "manifest.json"
    write_json(manifest, document)
    release = work_dir / "prepared_release"
    prepare_release(manifest, release, [selected.name], target_episodes=target_episodes,
                    strict_layout=False, no_dev=True)
    configs = write_configs(work_dir / "configs", name_prefix, tasks=[selected.name], seed=seed,
                            release=release, wan_dir=wan, vlm_dir=vlm, init_checkpoint=checkpoint,
                            output_root=work_dir / "runs", batch_size=batch_size)
    config = yaml.safe_load(configs[0].read_text())
    validate_recipe(config)
    preflight_assets(config, gpu=False)
    return dict(config=str(configs[0]), manifest=str(manifest), release=str(release),
                training_episodes=len(rows), validation_policy="none", layout_audit="not_provided",
                seed=seed, global_batch=4 * batch_size, max_steps=20000,
                initialization_weights=str(weights), model_compatibility="not_yet_loaded", executed=False)
