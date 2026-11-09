"""Publish full-holdout validation without mutating prior rounds or media.

Every rank renders the windows it scored with ``stage_window_media`` into a
shared staging directory right after inference; only lightweight entries
(identity + staged path) travel back to rank zero. ``publish_validation_artifacts``
then runs on rank zero after evaluate_distributed completes on every rank: it
checks that every scored window has complete staged media, moves them into a
fresh monitor revision and appends the completion record last. Pixels never
enter JSON. This publisher performs no inference.
"""
from __future__ import annotations

import fcntl
import json
import math
import numbers
import os
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from thinker_wm.periodic_eval import render

TASKS = ("insert_tubes", "pour_liquid_into_cup", "stack_bowls")
KNOWN_TASKS = TASKS + ("plug_in_charger", "insert_key", "fold_clothes",
                       "fill_pen_holder", "organize_table", "pick_from_conveyor_by_image",
                       "match_and_pick_from_conveyor", "sweep_blocks")
SOURCE = "robodojo_holdout"
FPS = 12.5
MEDIA_FILES = {"first_frame": "first_frame.png", "compare_video": "compare.mp4",
               "depth_compare_video": "depth_compare.mp4", "a2v_compare_video": "a2v_compare.mp4",
               "meta": "meta.json"}


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _json_value(value: Any) -> Any:
    """Metadata only: reject tensor/image payloads instead of stringifying them."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, numbers.Real):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite validation metadata")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    raise TypeError(f"Unsupported metadata payload: {type(value).__name__}")


def _scalar_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): _json_value(v) for k, v in values.items()
            if not str(k).startswith("_") and isinstance(v, numbers.Real)}


def _video(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 3 or value.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty T3HW tensor")
    value = value.detach().float().cpu()
    if not bool(torch.isfinite(value).all()) or float(value.min()) < 0 or float(value.max()) > 1:
        raise ValueError(f"{name} must be finite in [0,1]")
    return value


def _frames(value: torch.Tensor) -> Any:
    # The legacy renderer guesses CTHW first; make the layout explicit even
    # for one- or three-frame clips (where T can look like a channel axis).
    return render.tensor_to_uint8_frames(value.permute(1, 0, 2, 3))


def _actions(value: Any, name: str) -> list:
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != 14 or value.shape[0] == 0:
        raise ValueError(f"{name} must be raw [T,14] actions")
    return _json_value(value.detach().float().cpu().tolist())


def _summary(metrics: Mapping[str, Any], saved: list, expected_windows: int, expected_tasks=TASKS) -> tuple:
    windows = _json_value(metrics.get("_windows"))
    by_task = _json_value(metrics.get("_by_task"))
    if not isinstance(windows, list) or len(windows) != expected_windows:
        raise ValueError(f"expected all {expected_windows} validation windows")
    ids = [row["id"] for row in windows]
    if sorted(ids) != list(range(expected_windows)):
        raise ValueError("validation window IDs overlap or omit windows")
    counts = Counter(row.get("task_name") or row.get("task") for row in windows)
    if set(counts) != set(expected_tasks) or not isinstance(by_task, dict) or set(by_task) != set(expected_tasks):
        raise ValueError("full holdout must cover exactly the configured RoboDojo tasks")
    if metrics.get("num_samples") != expected_windows:
        raise ValueError("aggregate num_samples disagrees with full holdout coverage")
    if sorted(int(entry["window_id"]) for entry in saved) != list(range(expected_windows)):
        raise ValueError("staged media must cover every scored window exactly once")
    envelopes = {}
    for task, stats in by_task.items():
        if stats.get("num_samples") != counts[task]:
            raise ValueError(f"aggregate window count disagrees for {task}")
        task_metrics = {key: value for key, value in _scalar_metrics(stats).items() if key != "num_samples"}
        envelopes[task] = {"metrics": task_metrics,
                           "metric_counts": {key: counts[task] for key in task_metrics},
                           "num_windows": counts[task]}
    return windows, by_task, envelopes


def _sample_metadata(sample: Mapping[str, Any]) -> dict:
    task = sample.get("task_name") or sample.get("task")
    if task not in KNOWN_TASKS:
        raise ValueError(f"unknown RoboDojo validation task: {task!r}")
    prompt = sample.get("vlm_input_messages")
    if not prompt:
        raise ValueError("actual model vlm_input_messages is required; never reconstruct prompts")
    frame_indices = sample.get("frame_indices")
    if torch.is_tensor(frame_indices):
        if frame_indices.ndim != 1:
            raise ValueError("per-row frame_indices must be one-dimensional")
        frame_indices = frame_indices.detach().cpu().tolist()
    if not isinstance(frame_indices, (list, tuple)) or not frame_indices:
        raise ValueError("frame_indices is required")
    actions = _actions(sample.get("gt_actions"), "gt_actions")
    predicted_actions = _actions(sample.get("predicted_actions"), "predicted_actions")
    if len(actions) != len(predicted_actions):
        raise ValueError("GT and predicted actions have different lengths")
    return _json_value({
        "task_name": task, "task": task, "task_text": sample["task_text"],
        "episode_id": str(sample["episode_id"]), "window_id": int(sample["window_id"]),
        "frame_indices": frame_indices, "robot_name": sample.get("robot_name", "robodojo_arx_x5"),
        "metrics": _scalar_metrics(sample.get("metrics") or {}), "metrics_scope": "validation_window",
        "prompt_source": "model_argument", "vlm_prompt": prompt, "vlm_input_messages": prompt,
        "original_vlm_input_messages": sample.get("original_vlm_input_messages"),
        "vlm_output": None, "gt_actions": actions, "predicted_actions": predicted_actions,
        "action_dim_labels": [f"raw_joint14[{i}]" for i in range(14)],
        "action_space": "raw_joint14", "fps": FPS,
        "depth_metric_space": "codec_[0,1]", "depth_metric_units": "codec; not meters",
        "a2v_conditioning": "ground_truth_actions", "a2v_action_score_available": False,
        "media_labels": {
            "compare_video": "RGB: condition | ground truth | predicted",
            "depth_compare_video": "Depth codec [0,1]: RGB condition | GT depth | predicted depth (not meters)",
            "a2v_compare_video": "A2V: RGB condition + GT actions | GT RGB | predicted RGB; no predicted action score",
        },
    })


def _sample_videos(sample: Mapping[str, Any]) -> dict:
    values = {key: _video(sample.get(key), key) for key in (
        "gt_frames", "predicted_frames", "gt_depth_frames", "predicted_depth", "a2v_predicted_frames")}
    if len({tuple(value.shape) for value in values.values()}) != 1:
        raise ValueError("all saved RGB/depth/A2V clips must have identical T3HW shapes")
    first = sample.get("gt_first_frame")
    if not torch.is_tensor(first) or tuple(first.shape) != tuple(values["gt_frames"].shape[1:]):
        raise ValueError("gt_first_frame must be CHW matching RGB")
    values["first"] = _video(first.unsqueeze(0), "gt_first_frame")[0]
    mask = sample.get("depth_valid_mask")
    shape = values["predicted_depth"].shape
    if not torch.is_tensor(mask) or tuple(mask.shape) != (shape[0], shape[2], shape[3]):
        raise ValueError("depth_valid_mask must be THW matching the depth clip")
    mask = mask.detach().cpu()
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("depth_valid_mask must be boolean or binary")
    values["mask"] = mask.bool()
    return values


def stage_window_media(sample: Mapping[str, Any], staging_root: Any, rank: int) -> dict:
    """Render one scored window into ``staging_root/window_<id>`` on the scoring rank.

    Metrics were computed beforehand on the full-precision tensors; encoding
    only affects display. Returns a tensor-free entry for rank zero.
    """
    started = time.perf_counter()
    meta = _sample_metadata(sample)
    value = _sample_videos(sample)
    window_dir = Path(staging_root) / f"window_{meta['window_id']:04d}"
    window_dir.mkdir(parents=False, exist_ok=False)
    first = render.image_to_uint8(value["first"])
    render.save_png(window_dir / MEDIA_FILES["first_frame"], first)
    for file_key, gt_key, pred_key in (
        ("compare_video", "gt_frames", "predicted_frames"),
        ("depth_compare_video", "gt_depth_frames", "predicted_depth"),
        ("a2v_compare_video", "gt_frames", "a2v_predicted_frames"),
    ):
        gt, pred = value[gt_key], value[pred_key]
        if file_key == "depth_compare_video":
            # Invalid depth pixels are black on BOTH sides; do not turn
            # an unavailable GT pixel into an apparent prediction error.
            gt = gt * value["mask"][:, None]
            pred = pred * value["mask"][:, None]
        frames = render.make_compare_frames(first, _frames(gt), _frames(pred))
        label = {"compare_video": "RGB", "depth_compare_video": "Depth codec [0,1]; not meters",
                 "a2v_compare_video": "A2V: GT actions; RGB prediction"}[file_key]
        frames = np.stack([render._label_frame(frame, label) for frame in frames])
        render.write_video(window_dir / MEDIA_FILES[file_key], frames, fps=FPS)
    meta.update(eval_rank=int(rank), video_only=False,
                depth_valid_pixels=int(value["mask"].sum()), depth_total_pixels=value["mask"].numel(),
                depth_invalid_display="masked black in GT and prediction")
    render.write_meta_json(window_dir / MEDIA_FILES["meta"], meta)
    return {"window_id": meta["window_id"], "task": meta["task"], "episode_id": meta["episode_id"],
            "task_text": meta["task_text"], "vlm_input_messages": meta["vlm_input_messages"],
            "original_vlm_input_messages": meta["original_vlm_input_messages"],
            "t5_input_messages": sample.get("t5_input_messages"),
            "eval_rank": int(rank), "media_dir": str(window_dir),
            "media_seconds": time.perf_counter() - started}


def discard_staged_media(saved: list) -> None:
    """Remove a round's staging directory (after publication or when unpublished)."""
    roots = {Path(entry["media_dir"]).parent for entry in saved or [] if entry.get("media_dir")}
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)


def _staged_meta(entry: Mapping[str, Any], row: Mapping[str, Any]) -> dict:
    window_dir = Path(entry["media_dir"])
    missing = [name for name in MEDIA_FILES.values() if not (window_dir / name).is_file()]
    if missing:
        raise ValueError(f"window {entry['window_id']} staged media incomplete: {missing}")
    meta = json.loads((window_dir / MEDIA_FILES["meta"]).read_text())
    task = row.get("task_name") or row.get("task")
    if (meta.get("window_id") != row["id"] or meta.get("task") != task
            or str(meta.get("episode_id")) != str(row["episode_id"])
            or meta.get("frame_indices") != row.get("frame_indices")):
        raise ValueError("staged sample identity does not match scored validation window")
    return meta


def publish_validation_artifacts(
    config: Any, step: int, metrics: Mapping[str, Any], saved: list,
    monitor_run_dir: Any, seconds: float = 0.0, *, output_dir: Any = None,
    world_size: int | None = None, checkpoint: Any = None, expected_windows: int | None = None,
) -> dict:
    """Append a completed round with media for every window; return the round record.

    ``saved`` holds the entries returned by ``stage_window_media``. Media always
    live inside the existing monitor run. Optional output_dir receives a unique
    summary copy for a standalone invocation. A fresh UUID revision and
    mkdir(exist_ok=False) prohibit overwriting older artifacts. Call only on
    rank zero. Failures propagate before any stream record is written.
    """
    import torch.distributed as dist
    if (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0) or int(os.environ.get("RANK", "0")) != 0:
        raise RuntimeError("validation artifacts must be published by rank zero only")
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    run_dir = Path(monitor_run_dir).resolve()
    if not (run_dir / "run.json").is_file():
        raise ValueError(f"not an existing monitor run: {run_dir}")
    if int(step) < 0 or int(world_size) < 1 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid step, world_size, or elapsed seconds")
    system = _get(config, "system", {})
    expected_tasks = tuple(_get(system, "val_tasks", TASKS))
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks) or not set(expected_tasks) <= set(KNOWN_TASKS):
        raise ValueError("invalid configured validation tasks")
    if expected_windows is None:
        expected_windows = int(_get(system, "val_num_batches", 15)) * int(
            _get(_get(config, "training", {}), "batch_size", 8))
    windows, by_task, envelopes = _summary(metrics, saved, expected_windows, expected_tasks)
    task_quota = _get(system, "val_windows_per_task", None)
    if task_quota is not None and any(stats["num_samples"] != int(task_quota) for stats in by_task.values()):
        raise ValueError("validation window counts disagree with configured per-task quota")
    rows = {row["id"]: row for row in windows}
    ordered = sorted(saved, key=lambda entry: int(entry["window_id"]))
    # Validate every staged window before creating or publishing any records.
    metas = [_staged_meta(entry, rows[int(entry["window_id"])]) for entry in ordered]
    revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex
    relative = Path("eval") / f"step_{int(step):07d}" / f"revision_{revision}"
    revision_dir = run_dir / relative
    revision_dir.mkdir(parents=True, exist_ok=False)
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    base = {"step": int(step), "revision": revision, "round_id": f"{int(step)}:{revision}",
            "ts": ts, "source": SOURCE, "dataset": SOURCE, "split": "holdout",
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "schema_version": 2, "protocol": "distributed_full_holdout_all_windows_v2"}
    records = []
    started = time.perf_counter()
    for entry, meta in zip(ordered, metas):
        window_id = int(entry["window_id"])
        sample_rel = relative / meta["task"] / f"window_{window_id:04d}"
        (run_dir / sample_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(entry["media_dir"], run_dir / sample_rel)
        files = {key: str(sample_rel / name) for key, name in MEDIA_FILES.items()}
        # rank 0 publishes every record; sample_index = window ID keeps the
        # monitor's (step, rank, sample_index) key unique and window-ordered.
        meta.update(base, rank=0, sample_index=window_id, files=files)
        render.write_meta_json(run_dir / files["meta"], meta)
        # Action arrays are compact but belong only in meta, not every stream record.
        records.append({key: value for key, value in dict(meta, type="eval_sample").items()
                        if key not in ("gt_actions", "predicted_actions", "action_dim_labels")})
    depth_config = _get(config, "depth", {})
    round_record = dict(base, type="eval_round", num_samples=len(records),
                        num_saved_samples=len(records), viz_samples=len(records),
                        num_eval_samples=len(windows), num_windows=len(windows),
                        ready_ranks=min(world_size, len(windows)), world_size=world_size,
                        seconds=float(seconds), publish_seconds=time.perf_counter() - started,
                        media_seconds_total=math.fsum(float(e.get("media_seconds", 0.0)) for e in ordered),
                        inference_seconds_total=math.fsum(float(e.get("inference_seconds", 0.0)) for e in ordered),
                        sources={SOURCE: len(records)}, skipped=[],
                        metrics_scope="full_validation", metrics=_scalar_metrics(metrics),
                        metrics_by_task=envelopes, _by_task=by_task, _windows=windows,
                        prompt_source="model_argument", depth_metric_space="codec_[0,1]",
                        depth_codec={key: _get(depth_config, key) for key in ("mode", "z_near", "z_far", "invalid_to")},
                        a2v_conditioning="ground_truth_actions", a2v_action_score_available=False,
                        files={"summary": str(relative / "summary.json")})
    round_record = _json_value(round_record)
    render.write_meta_json(revision_dir / "summary.json", round_record)
    if output_dir is not None:
        export_dir = Path(output_dir).resolve() / f"step_{int(step):07d}_{revision}"
        export_dir.mkdir(parents=True, exist_ok=False)
        render.write_meta_json(export_dir / "summary.json", round_record)
    # The lock serializes future hook/standalone publishers. The existing
    # chunk writer opens with 'a' and resumes its last chunk; no stream rewrite.
    from motus_monitor.store import ChunkedJsonlWriter
    with (run_dir / "eval" / ".validation_publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        writer = ChunkedJsonlWriter(run_dir / "eval")
        try:
            for record in records:
                writer.write(record)
            writer.write(round_record)  # completion marker comes after all media/samples
        finally:
            writer.close()
    return round_record
