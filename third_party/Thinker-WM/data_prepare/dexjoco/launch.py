"""Single-node four-GPU launcher with explicit fresh/resume identities."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from .contracts import CONTRACT, sha256, task_for, write_json
from .prepare import verify_release
from .recipe import MODEL_ROOT

REPO = MODEL_ROOT.parents[1]


def resolve_weights(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    for name in ("pytorch_model/mp_rank_00_model_states.pt", "mp_rank_00_model_states.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no supported Motus model checkpoint at {path}")


def validate_recipe(c: dict) -> None:
    spec = c["dexjoco"]
    task = task_for(spec["task"])
    common = c["common"]
    expected = dict(control_hz=30, action_dim=task.dimension, state_dim=task.dimension,
                    unified_action_dim=task.dimension, num_video_frames=12, action_chunk_size=24,
                    video_action_freq_ratio=2, video_height=480, video_width=640)
    if any(common.get(k) != v for k, v in expected.items()) or spec.get("contract") != CONTRACT:
        raise ValueError("DexJoCo model/temporal contract differs from the recipe")
    if c["training"].get("seed", 0) <= 0 or c["training"].get("gradient_accumulation_steps") != 1:
        raise ValueError("explicit positive seed and gradient_accumulation_steps=1 required")
    if c["training"]["batch_size"] not in (4, 8):
        raise ValueError("batch_size must be 8 or the plan's fallback 4")
    wm = c["model"]["wm"]
    if (wm.get("enable_depth") or wm.get("history_enable") or c.get("history", {}).get("enable")
            or wm.get("action_target_repr") != "absolute_native" or wm.get("action_norm_policy") != "zscore"
            or wm.get("training_mode_probs") != {"joint": .6, "a2v": .4}):
        raise ValueError("expected RGB-only absolute/zscore joint:A2V=60:40")
    if c["resume"].get("auto_resume_latest") or c["resume"].get("checkpoint_path"):
        raise ValueError("use explicit --resume; automatic/YAML resume is disabled")
    if c["finetune"].get("action_interface_policy") != "dexjoco_rgb":
        raise ValueError("explicit DexJoCo interface migration is required")
    root = Path(spec["release"]).resolve()
    if sha256(root / "release.json") != spec["release_sha256"]:
        raise ValueError("prepared release identity changed")
    verify_release(root, task.name)
    sources = c["dataset"]["sources"]
    if len(sources) != 1 or sources[0].get("dataset") != "dexjoco":
        raise ValueError("each run must have exactly one DexJoCo source")
    source = sources[0]
    for key, filename in (("catalog_path", "train.episode_catalog.jsonl"),
                          ("eval_catalog_path", "dev.episode_catalog.jsonl")):
        if Path(source[key]).resolve() != root / task.name / filename:
            raise ValueError("cross-task or split catalog mismatch")
    for key, split in (("pool", "train"), ("eval_pool", "dev")):
        if Path(source[key]["path"]).resolve() != root / task.name / f"{split}.ranges.sqlite":
            raise ValueError("cross-task or split pool mismatch")
        if source[key].get("action_start_offset") != 0:
            raise ValueError("DexJoCo actions must start at observation t")
    if Path(spec["action_stats_path"]).resolve() != root / task.name / "action_stats.json":
        raise ValueError("cross-task action statistics")
    if sha256(spec["action_stats_path"]) != spec["action_stats_sha256"]:
        raise ValueError("statistics hash mismatch")
    if set(c["dataset"]["robot_names"]) != {task.robot}:
        raise ValueError("wrong robot identity")


def preflight_assets(c: dict, *, gpu: bool = False) -> None:
    resolve_weights(Path(c["finetune"]["checkpoint_path"]))
    for model in ("wan", "vlm"):
        for key in ("checkpoint_path", "config_path"):
            if not Path(c["model"][model][key]).is_dir():
                raise FileNotFoundError(c["model"][model][key])
    if not Path(c["model"]["wan"]["vae_path"]).is_file():
        raise FileNotFoundError(c["model"]["wan"]["vae_path"])
    # Raw RGB files remain outside the prepared release; verify those too.
    task_root = Path(c["dexjoco"]["release"]) / c["dexjoco"]["task"]
    for row in json.loads((task_root / "episodes.json").read_text()):
        for path, expected in row["camera_sha256"].items():
            if sha256(path) != expected:
                raise ValueError(f"RGB source changed since preparation: {path}")
    if gpu:
        missing = [m for m in ("torch", "accelerate", "deepspeed", "transformers", "diffusers",
                              "omegaconf", "cv2", "safetensors", "tensorboard", "qwen_vl_utils")
                   if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError(f"training environment is missing: {missing}")
        import torch
        if torch.cuda.device_count() != 4:
            raise RuntimeError("expose exactly four GPUs using CUDA_VISIBLE_DEVICES or the job allocation")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on the selected CUDA device")


@contextmanager
def exclusive_lock(path: Path):
    import fcntl
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another DexJoCo launch holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def plan_launch(config: Path, run_name: str, *, resume: Path | None = None,
                smoke_steps: int | None = None, max_steps: int | None = None) -> tuple[dict, Path, list[str]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run-name must be a simple directory component")
    if smoke_steps is not None and smoke_steps not in (100, 200):
        raise ValueError("smoke-steps must be 100 or 200")
    if max_steps is not None and (resume is None or max_steps not in (30000, 60000)):
        raise ValueError("extended budgets 30k/60k require an explicit same-run resume")
    if smoke_steps is not None and max_steps is not None:
        raise ValueError("smoke and extended budgets are mutually exclusive")
    c = yaml.safe_load(config.read_text())
    validate_recipe(c)
    if c["training"]["max_steps"] != 20000 or c["system"]["save_interval"] != 5000 or c["system"]["val_interval"] != 5000:
        raise ValueError("base recipe must retain 20k updates and 5k save/evaluation intervals")
    if smoke_steps is not None:
        c["training"]["max_steps"] = smoke_steps
        c["system"].update(save_interval=smoke_steps, val_interval=smoke_steps)
    if max_steps is not None:
        c["training"]["max_steps"] = max_steps
    c["logging"]["run_name"] = run_name
    root = Path(c["system"]["checkpoint_dir"]).resolve()
    run = root / config.stem / run_name
    c["logging"]["tensorboard_log_dir"] = str(run / "tensorboard")
    effective = run / "inputs" / config.name
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=4", "--max_restarts=0", "--module", "thinker_wm.train.train",
               "--config", str(effective), "--deepspeed", str(MODEL_ROOT / "configs/zero1.json")]
    if resume is not None:
        resume = resume.resolve()
        if resume.parent != run or not resume.name.startswith("checkpoint_step_"):
            raise ValueError("resume checkpoint must belong to this exact task/config/run")
        receipt = json.loads((resume / ".motus_checkpoint_complete.json").read_text())
        if receipt.get("status") != "complete" or not isinstance(receipt.get("step"), int):
            raise ValueError("resume checkpoint is incomplete")
        if receipt["step"] >= c["training"]["max_steps"]:
            raise ValueError("resume target must exceed the saved optimizer step")
        command += ["--resume_checkpoint", str(resume)]
    return c, run, command


def launch(config: Path, run_name: str, *, execute: bool = False, check: bool = False, **options) -> dict:
    c, run, command = plan_launch(config, run_name, **options)
    result = dict(task=c["dexjoco"]["task"], run_directory=str(run), command=command,
                  global_batch=4 * c["training"]["batch_size"], max_steps=c["training"]["max_steps"],
                  executed=False)
    if check or execute:
        preflight_assets(c, gpu=True)
    if not execute:
        return result
    for name in ("MOTUS_SKIP_BACKWARD", "MOTUS_SKIP_OPTIMIZER_WHEN_LR_ZERO", "MOTUS_FP8_LINEAR"):
        if os.environ.get(name, "0") not in ("", "0"):
            raise ValueError(f"unset {name}: it changes the training recipe")
    root = Path(c["system"]["checkpoint_dir"]).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    # A serial six-task queue and individual invocations share the same lock.
    with exclusive_lock(root / ".dexjoco-training.lock"):
        resume = options.get("resume")
        identity = dict(config_sha256=sha256(config), release_sha256=c["dexjoco"]["release_sha256"],
                        task=c["dexjoco"]["task"], smoke_steps=options.get("smoke_steps"))
        if resume is None:
            run.mkdir(parents=True, exist_ok=False)
            write_json(run / "run_identity.json", identity)
            (run / "inputs").mkdir()
        elif json.loads((run / "run_identity.json").read_text()) != identity:
            raise ValueError("resume configuration/release/smoke identity mismatch")
        effective = run / "inputs" / config.name
        effective.write_text(yaml.safe_dump(c, sort_keys=False))
        shutil.copyfile(c["dexjoco"]["action_stats_path"], run / "action_stats.json")
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join((str(REPO), str(MODEL_ROOT / "libs/motus-data-factory"),
                                             str(MODEL_ROOT / "libs/motus-monitor"), env.get("PYTHONPATH", ""))),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MOTUS_NONFINITE_FAIL_FAST="1",
                   MOTUS_TEMPORAL_ATTN="flex", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with (run / "train.log").open("a") as log:
            code = subprocess.call(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"training exited {code}; inspect {run / 'train.log'}; queue stopped")
        # A clean process exit may also mean an intentional early stop.
        final = run / f"checkpoint_step_{c['training']['max_steps']}" / ".motus_checkpoint_complete.json"
        final_receipt = json.loads(final.read_text()) if final.is_file() else {}
        if (final_receipt.get("status") != "complete"
                or final_receipt.get("step") != c["training"]["max_steps"]):
            raise RuntimeError("training did not reach its target checkpoint; queue stopped")
        result["executed"] = True
    return result


def launch_queue(configs: list[Path], run_name: str, *, execute: bool = False,
                 check: bool = False, smoke_steps: int | None = None) -> list[dict]:
    """One four-GPU run at a time; fail immediately without advancing on error."""
    from .contracts import TASKS
    if len(configs) != len(TASKS):
        raise ValueError("queue requires exactly six independent task configs")
    parsed = [yaml.safe_load(p.read_text()) for p in configs]
    mapping = {c["dexjoco"]["task"]: p for c, p in zip(parsed, configs)}
    if set(mapping) != {t.name for t in TASKS}:
        raise ValueError("queue must cover each selected task exactly once")
    if len({c["finetune"]["checkpoint_path"] for c in parsed}) != 1:
        raise ValueError("all six tasks must initialize from the same pretrained source")
    if len({c["training"]["seed"] for c in parsed}) != 6:
        raise ValueError("each task requires its own training seed")
    # Validate the entire queue before executing its first run.
    ordered = [mapping[t.name] for t in TASKS]
    for path in ordered:
        c, _, _ = plan_launch(path, run_name, smoke_steps=smoke_steps)
        if check or execute:
            preflight_assets(c, gpu=True)
    return [launch(path, run_name, execute=execute, smoke_steps=smoke_steps) for path in ordered]
