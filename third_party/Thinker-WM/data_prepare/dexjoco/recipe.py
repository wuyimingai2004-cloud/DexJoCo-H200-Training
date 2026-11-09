"""Build the six independent recipes from the existing Joint24 architecture."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import yaml

from .contracts import CONTRACT, TASKS, task_for, sha256
from .prepare import verify_release

MODEL_ROOT = Path(__file__).resolve().parents[2]
BASE = MODEL_ROOT / "configs/robodojo_joint24_v2.yaml"


def build_config(name: str, *, release: Path, wan_dir: Path, vlm_dir: Path,
                 init_checkpoint: Path, output_root: Path, batch_size: int = 8) -> dict:
    task = task_for(name)
    if batch_size not in (4, 8):
        raise ValueError("plan supports per-GPU batch 8 or fallback 4; accumulation remains 1")
    receipt = verify_release(release, task.name)
    root = release.resolve() / task.name
    c = deepcopy(yaml.safe_load(BASE.read_text()))
    c["common"].update(control_hz=30, action_dim=task.dimension, state_dim=task.dimension,
                       unified_action_dim=task.dimension)
    c["dataset"] = dict(type="indexed_mix", batches_per_epoch=None, drop_last=True,
        cross_rank_source_policy="synchronized", robot_names={task.robot: dict(
            action_dim=task.dimension, unified_dim_map=list(range(task.dimension)), robot_token_index=0)},
        sources=[dict(name="dexjoco", dataset="dexjoco", weight=1., prompt_mode="vlm_image_text",
            catalog_path=str(root / "train.episode_catalog.jsonl"),
            index_path=str(root / "train.episode_catalog.index.sqlite"),
            eval_catalog_path=str(root / "dev.episode_catalog.jsonl"),
            eval_index_path=str(root / "dev.episode_catalog.index.sqlite"), reader=dict(type="dexjoco"),
            pool=dict(type="sampling_ranges", path=str(root / "train.ranges.sqlite"),
                      action_start_offset=0, without_replacement=True),
            eval_pool=dict(type="sampling_ranges", path=str(root / "dev.ranges.sqlite"),
                           action_start_offset=0, windows_per_episode=8))])
    c.pop("depth", None)
    c["history"] = dict(enable=False)
    wm = c["model"]["wm"]
    for key in list(wm):
        if key.startswith("depth_") or key.startswith("history_"):
            del wm[key]
    wm.update(enable_depth=False, history_enable=False, training_mode_probs=dict(joint=.6, a2v=.4))
    c["model"]["wan"].update(checkpoint_path=str(wan_dir.resolve()), config_path=str(wan_dir.resolve()),
                               vae_path=str(wan_dir.resolve() / "Wan2.2_VAE.pth"))
    c["model"]["vlm"].update(checkpoint_path=str(vlm_dir.resolve()), config_path=str(vlm_dir.resolve()))
    # The inherited T-layout prompt says left/right wrists in the opposite order.
    c["model"]["vlm_prompt_style"] = "dexjoco_tlayout"
    c["model"]["load_pretrained_backbones"] = False
    c["training"].update(batch_size=batch_size, max_steps=20000, gradient_accumulation_steps=1, seed=task.seed)
    for key in ("history_bias_learning_rate", "depth_learning_rate", "depth_clone_learning_rate"):
        c["training"].pop(key, None)
    c["system"].update(checkpoint_dir=str(output_root.resolve() / task.name / "checkpoints"),
        save_interval=5000, val_interval=5000, val_a2v=True, val_depth=False,
        val_tasks=[task.name], val_num_batches=receipt["tasks"][task.name]["dev"] * 8 // batch_size,
        val_full_set=True, val_distributed=True, val_seed=20261109,
        num_workers=0, ready_batch_queue_size=2, ready_batch_queue_clone_tensors=False)
    for key in ("val_episode_order_path", "val_windows_per_task"):
        c["system"].pop(key, None)
    c["logging"].update(run_name=None, tensorboard_log_dir="tensorboard", wandb_project="dexjoco")
    c["resume"] = dict(checkpoint_path=None, auto_resume_latest=False, reset_scheduler=False)
    c["finetune"] = dict(checkpoint_path=str(init_checkpoint.resolve()), action_interface_policy="dexjoco_rgb")
    c["monitor"] = dict(enabled=False)
    c["dexjoco"] = dict(task=task.name, label=task.label, contract=CONTRACT,
        native_state_dim=23 * task.arms, release=str(release.resolve()),
        release_sha256=sha256(release / "release.json"), action_stats_path=str(root / "action_stats.json"),
        action_stats_sha256=sha256(root / "action_stats.json"), target_gpus=4,
        execute_prefix=16, base_config_sha256=sha256(BASE))
    return c


def write_configs(directory: Path, prefix: str, **kwargs) -> list[Path]:
    """Caller chooses new names explicitly; no existing configuration is overwritten."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("name-prefix must be a simple filename component")
    names = [directory / f"{prefix}_{t.name}.yaml" for t in TASKS]
    if any(p.exists() for p in names):
        raise FileExistsError("refusing to replace an existing task configuration")
    configs = [build_config(t.name, **kwargs) for t in TASKS]
    directory.mkdir(parents=True, exist_ok=True)
    for path, config in zip(names, configs):
        with path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    return names


def install_statistics(config) -> None:
    """Install this task's train-only statistics at the model normalization boundary."""
    import numpy as np
    from thinker_wm.models.action_stats import ACTION_STATS
    spec = config.dexjoco
    task = task_for(str(spec.task))
    path = Path(spec.action_stats_path)
    if sha256(path) != spec.action_stats_sha256:
        raise ValueError("action statistics changed since config creation")
    stats = json.loads(path.read_text())
    if (stats.get("task") != task.name or stats.get("robot") != task.robot
            or stats.get("contract") != CONTRACT or stats.get("source_split") != "train"):
        raise ValueError("statistics task/contract/split mismatch")
    for key in ("mean", "std"):
        x = np.asarray(stats[key])
        if x.shape != (task.dimension,) or not np.isfinite(x).all():
            raise ValueError(f"invalid {key} statistics")
    if np.any(np.asarray(stats["std"]) <= 0):
        raise ValueError("nonpositive normalization std")
    ACTION_STATS[task.robot] = {k: stats[k] for k in ("mean", "std")}
