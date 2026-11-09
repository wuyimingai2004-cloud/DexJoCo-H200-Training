#!/usr/bin/env python3

import os
import re
import sys
import argparse
import json
import logging
import shutil
import subprocess
import time
import gc
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import warnings

#from thinker_wm.configs.debug_flag import DEBUG_FLAG as DEBUF_FLAG

# Set CUDA memory management environment variables to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, InitProcessGroupKwargs
import yaml
from omegaconf import OmegaConf
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "libs" / "motus-monitor"))  # motus_monitor 包

from thinker_wm.models.ubt_wm import UBT_WM, MotusConfig
from thinker_wm.models.temporal_training import motus_temporal_kwargs
from thinker_wm.data_prepare import create_eval_data_pipeline, create_training_data_pipeline
from thinker_wm.periodic_eval import build_periodic_evaluator
from thinker_wm.utils.output_paths import resolve_output_path
from thinker_wm.utils.scheduler import create_scheduler
from thinker_wm.utils.cuda_failure import exit_on_fatal_cuda_error
from thinker_wm.utils.training_control import TrainingStopRequest, step_timing_breakdown, format_step_timing
from thinker_wm.utils.vlm_utils import (
    VLM_PROMPT_STYLE_ROBOTWIN_TLAYOUT,
    build_vlm_rewrite_message,
    build_vlm_task_message,
    resolve_vlm_prompt_style,
    sample_vlm_field_dropout,
)
from thinker_wm.utils.vlm_utils import resolve_prompt_mode, PROMPT_MODE_TEXT_ONLY
from thinker_wm.train.sample import evaluate_model, log_evaluation_metrics, share_validation_error

try:
    from motus_monitor import NullMonitor, create_monitor
    from motus_monitor.timing import TimingStack
except Exception:  # monitor must never break training
    NullMonitor = None
    create_monitor = None
    # Minimal fallback: a dict-like object that supports span() as no-op
    class _FallbackTiming(dict):
        def span(self, name, sync_cuda=None):
            from contextlib import contextmanager
            @contextmanager
            def _noop():
                yield None
            return _noop()
        def to_flat_dict(self):
            return dict(self)
        def to_tree(self):
            return []
        def clear(self):
            super().clear()
    TimingStack = _FallbackTiming


def _monitor_call(monitor: Any, method: str, *args: Any, **kwargs: Any) -> None:
    """Invoke a monitor hook defensively: any failure is logged, never raised."""
    if monitor is None:
        return
    try:
        getattr(monitor, method)(*args, **kwargs)
    except Exception as exc:
        logger.warning("thinker_wm.monitor.%s failed (ignored): %s", method, exc)


# Set by main() once the early monitor exists; add_timing_event forwards
# initialization phases to it so the monitor UI shows init progress live.
_EARLY_MONITOR = None

logger = logging.getLogger(__name__)

CHECKPOINT_COMPLETE_MARKER = ".motus_checkpoint_complete.json"

# On a ZeRO world-size mismatch we convert the legacy checkpoint to universal
# format once via deepspeed.checkpoint.ds_to_universal (CPU-only, runs on the
# main process; other ranks poll for completion). Worker counts default to the
# tool's defaults and can be tuned down on memory-constrained nodes.
UNIVERSAL_CONVERSION_TIMEOUT = int(os.environ.get("MOTUS_UNIVERSAL_CONV_TIMEOUT", "21600"))  # 6h
UNIVERSAL_CONVERSION_POLL_INTERVAL = 10
UNIVERSAL_CONVERSION_WORKERS = int(os.environ.get("MOTUS_UNIVERSAL_CONV_WORKERS", "4"))
UNIVERSAL_CONVERSION_MERGE_WORKERS = int(os.environ.get("MOTUS_UNIVERSAL_CONV_MERGE_WORKERS", "2"))


class StepTimingWriter:
    """Append per-step timing records as JSONL with minimal training-loop overhead."""

    def __init__(self, checkpoint_dir: str, rank: int, enabled: bool = True):
        self.rank = rank
        self.enabled = enabled
        self.path = Path(checkpoint_dir) / "step_timing" / f"step_timing_rank{rank}.jsonl"
        self._fp = None

        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", buffering=1)

    def write(self, record: Dict[str, Any]):
        if not self.enabled or self._fp is None:
            return
        payload = {
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "rank": self.rank,
            **record,
        }
        self._fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None


class JsonlDiagnosticsWriter:
    """Append filtered diagnostic records for loss/data anomalies."""

    def __init__(self, checkpoint_dir: str, rank: int, name: str, enabled: bool = True):
        self.rank = rank
        self.enabled = enabled
        self.path = Path(checkpoint_dir) / name / f"{name}_rank{rank}.jsonl"
        self._fp = None
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", buffering=1)

    def write(self, record: Dict[str, Any]):
        if not self.enabled or self._fp is None:
            return
        payload = {
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "rank": self.rank,
            **record,
        }
        self._fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def _cuda_sync_if_needed(device: str):
    """Synchronize CUDA so timing reflects actual GPU work, not async launch time."""
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in ("1", "true", "yes", "on")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _read_proc_meminfo() -> Dict[str, int]:
    """Read selected host memory counters from /proc/meminfo in bytes."""
    wanted = {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "Shmem",
        "SReclaimable",
    }
    values: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as fp:
            for line in fp:
                key, _, rest = line.partition(":")
                if key not in wanted:
                    continue
                parts = rest.strip().split()
                if not parts:
                    continue
                values[key] = int(parts[0]) * 1024
    except Exception:
        pass
    return values


def _process_rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/statm", "r") as fp:
            parts = fp.read().strip().split()
        if len(parts) < 2:
            return None
        return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _statvfs_usage(path: str) -> Dict[str, int]:
    try:
        stat = os.statvfs(path)
    except Exception:
        return {}
    total = int(stat.f_blocks * stat.f_frsize)
    available = int(stat.f_bavail * stat.f_frsize)
    free = int(stat.f_bfree * stat.f_frsize)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "free_bytes": free,
        "used_bytes": max(0, total - free),
    }


def _memory_snapshot(prefix: str = "memory") -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    rss = _process_rss_bytes()
    if rss is not None:
        record[f"{prefix}_process_rss_bytes"] = int(rss)
    for key, value in _read_proc_meminfo().items():
        record[f"{prefix}_host_{key.lower()}_bytes"] = int(value)
    for key, value in _statvfs_usage("/dev/shm").items():
        record[f"{prefix}_dev_shm_{key}"] = int(value)
    if torch.cuda.is_available():
        try:
            device_idx = torch.cuda.current_device()
            record[f"{prefix}_cuda_device"] = int(device_idx)
            record[f"{prefix}_cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device_idx))
            record[f"{prefix}_cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device_idx))
            record[f"{prefix}_cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device_idx))
            record[f"{prefix}_cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device_idx))
        except Exception:
            pass
    return record


def _tensor_debug_stats(tensor: Optional[torch.Tensor], prefix: str) -> Dict[str, Any]:
    """Return small scalar stats for step timing without changing training semantics."""
    if tensor is None or not torch.is_tensor(tensor):
        return {}
    with torch.no_grad():
        data = tensor.detach().float()
        if data.numel() == 0:
            return {}
        flat = data.reshape(-1)
        stats = {
            f"{prefix}_mean": float(flat.mean().item()),
            f"{prefix}_std": float(flat.std(unbiased=False).item()),
            f"{prefix}_abs_mean": float(flat.abs().mean().item()),
            f"{prefix}_abs_max": float(flat.abs().max().item()),
            f"{prefix}_numel": int(flat.numel()),
        }
        if data.ndim >= 2:
            per_sample = data.reshape(data.shape[0], -1).abs().max(dim=1).values
            stats[f"{prefix}_sample_abs_max"] = [float(x) for x in per_sample.cpu().tolist()]
            stats[f"{prefix}_sample_abs_max_argmax"] = int(per_sample.argmax().item())
        return stats


def _distributed_mean(values: list, device: Optional[Any] = None) -> list:
    """Mean of numeric values across all ranks; no-op passthrough without distributed.

    Uses the same device-selection logic as _distributed_any (NCCL cannot
    all_reduce CPU tensors).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [float(v) for v in values]
    backend = str(dist.get_backend()).lower()
    if torch.cuda.is_available() and "nccl" in backend:
        tensor_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, torch.device):
        tensor_device = device
    elif isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        tensor_device = torch.device(device)
    else:
        tensor_device = torch.device("cpu")
    tensor = torch.tensor([float(v) for v in values], device=tensor_device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return [float(x) for x in tensor.tolist()]


def _distributed_any(value: bool, device: Optional[Any] = None) -> bool:
    """Return whether any rank reports True.

    NCCL process groups cannot all_reduce CPU tensors. Some launch paths keep
    Trainer.device as "cpu" even though the model is placed by DeepSpeed on the
    current CUDA device, so choose the reduction tensor device from the backend.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    backend = str(dist.get_backend()).lower()
    if torch.cuda.is_available() and "nccl" in backend:
        tensor_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, torch.device):
        tensor_device = device
    elif isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        tensor_device = torch.device(device)
    else:
        tensor_device = torch.device("cpu")
    tensor = torch.tensor([1 if value else 0], device=tensor_device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def _find_first_nonfinite_grad(model: torch.nn.Module) -> Dict[str, Any]:
    """Inspect available local gradients and report the first non-finite tensor."""
    checked = 0
    missing = 0
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        grad = None
        if hasattr(param, "grad") and param.grad is not None:
            grad = param.grad
        elif hasattr(param, "ds_grad") and param.ds_grad is not None:
            grad = param.ds_grad
        if grad is None:
            missing += 1
            continue
        checked += 1
        with torch.no_grad():
            finite = torch.isfinite(grad)
            if not bool(finite.all().item()):
                return {
                    "has_nonfinite_grad": True,
                    "first_nonfinite_grad_param": name,
                    "first_nonfinite_grad_shape": list(grad.shape),
                    "first_nonfinite_grad_nan_count": int(torch.isnan(grad).sum().item()),
                    "first_nonfinite_grad_inf_count": int(torch.isinf(grad).sum().item()),
                    "grad_tensors_checked": checked,
                    "grad_tensors_missing": missing,
                }
    return {
        "has_nonfinite_grad": False,
        "first_nonfinite_grad_param": None,
        "first_nonfinite_grad_shape": None,
        "first_nonfinite_grad_nan_count": 0,
        "first_nonfinite_grad_inf_count": 0,
        "grad_tensors_checked": checked,
        "grad_tensors_missing": missing,
    }


def _find_first_nonfinite_param(model: torch.nn.Module) -> Dict[str, Any]:
    """Inspect local parameter shards and report the first non-finite value."""
    checked = 0
    for name, param in model.named_parameters():
        data = getattr(param, "data", None)
        if data is None:
            continue
        checked += 1
        with torch.no_grad():
            if not bool(torch.isfinite(data).all().item()):
                return {
                    "has_nonfinite_param": True,
                    "first_nonfinite_param": name,
                    "first_nonfinite_param_shape": list(data.shape),
                    "first_nonfinite_param_nan_count": int(torch.isnan(data).sum().item()),
                    "first_nonfinite_param_inf_count": int(torch.isinf(data).sum().item()),
                    "param_tensors_checked": checked,
                }
    return {
        "has_nonfinite_param": False,
        "first_nonfinite_param": None,
        "first_nonfinite_param_shape": None,
        "first_nonfinite_param_nan_count": 0,
        "first_nonfinite_param_inf_count": 0,
        "param_tensors_checked": checked,
    }


def _optimizer_all_lrs_zero(optimizer: torch.optim.Optimizer) -> bool:
    """Return True when every optimizer param group has zero LR."""
    lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    return bool(lrs) and all(lr == 0.0 for lr in lrs)


def _nonfinite_fail_fast_enabled() -> bool:
    return os.environ.get("MOTUS_NONFINITE_FAIL_FAST", "1").lower() not in ("0", "false", "no")


def _outlier_loss_zero_threshold() -> float:
    """训练侧离群样本 loss 置零阈值（统一 delta-EEF 空间样本级 abs max）；<=0 关闭。

    检查对象是 ``prepare_training_context`` 产出的统一 action 张量 [B,T,U]
    （``action_target_repr=relative_eef_rot6`` 下 EEF pos=相对条件帧 state 的
    delta 物理量、rot6/夹爪∈[-1,1]）；delta EEF 目标表示下 raw |action|/|state|
    绝对值语义已过时，不再检查。库内 raise 经
    MOTUS_ACTION_MAGNITUDE_GUARD_THRESHOLD=0 关闭后，离群样本随 batch 进入
    训练，由本阈值检测并把其 loss 权重置零（默认 50：合法 delta 远低于 1，
    触发即数据损坏级离群）。
    """
    raw = os.environ.get("MOTUS_TRAIN_OUTLIER_LOSS_ZERO_THRESHOLD", "50")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 50.0


def _sanitize_nonfinite_batch(
    actions: Optional[torch.Tensor],
    state: Optional[torch.Tensor],
) -> Optional[Dict[str, Any]]:
    """``prepare_training_context`` 之前的 raw actions/state 非有限防御扫描。

    raw 非有限值会经归一化/scatter/validity-mask 提取污染整个 training
    context（且落在摘除维的 NaN 传播不到统一 delta-EEF 张量、context 后的
    阈值检查兜不住），必须前置拦截并 nan_to_num。命中样本由调用侧并入离群
    mask（loss 权重置零）。actions 为 None 返回 None；否则返回 dict（CPU
    tensor）：``mask``（逐样本 bool，True=含非有限，可能全 False）、
    ``state_abs_max``（逐样本 float，仅作信息性记录）。
    """
    if actions is None:
        return None
    with torch.no_grad():
        nonfinite = ~torch.isfinite(actions.detach()).all(
            dim=tuple(range(1, actions.ndim))
        )
        if state is not None:
            state_detached = state.detach()
            state_abs_max = state_detached.abs().amax(
                dim=tuple(range(1, state.ndim))
            )
            nonfinite |= ~torch.isfinite(state_detached).all(
                dim=tuple(range(1, state.ndim))
            )
        else:
            state_abs_max = torch.zeros(
                actions.shape[0], device=actions.device, dtype=torch.float32
            )
        hit = bool(nonfinite.any().item())
    if hit:
        actions.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if state is not None:
            state.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mask": nonfinite.cpu(),
        "state_abs_max": state_abs_max.float().cpu(),
    }


def _zero_outlier_sample_loss_weights(
    *,
    actions: Optional[torch.Tensor],
    action_loss_weights: Optional[torch.Tensor],
    video_loss_weights: Optional[torch.Tensor],
    threshold: float,
) -> tuple:
    """把统一 delta-EEF 空间 abs max 超阈值（或含非有限值）样本的 loss 权重置零。

    ``actions`` 是 ``prepare_training_context`` 产出的统一 action 张量
    [B,T,U]（delta EEF 目标表示：pos=相对锚点 delta 物理量、rot6/夹爪
    ∈[-1,1]）；raw |action|/|state| 绝对值不再检查（state 的非有限防御在
    ``_sanitize_nonfinite_batch`` 前置完成）。非有限样本同步 nan_to_num
    （防御 nan*0=nan 污染总 loss；此处张量即 loss 目标本身）。返回
    (action_loss_weights, video_loss_weights, outlier_count, outlier_details)；
    outlier_details 为 None（无离群样本）或 dict（CPU tensor）：
    ``mask``/``nonfinite``（逐样本 bool）、``action_abs_max``（逐样本
    float），供调用侧按样本索引回查 episode 元数据做记录。
    """
    if (
        threshold is None
        or float(threshold) <= 0
        or actions is None
        or action_loss_weights is None
    ):
        return action_loss_weights, video_loss_weights, 0, None
    with torch.no_grad():
        action_abs_max = actions.detach().abs().amax(
            dim=tuple(range(1, actions.ndim))
        )
        nonfinite = ~torch.isfinite(action_abs_max)
        outlier = nonfinite | (action_abs_max > float(threshold))
        count = int(outlier.sum().item())
    if count <= 0:
        return action_loss_weights, video_loss_weights, 0, None
    logger.warning(
        "zeroing loss for %d/%d outlier samples (unified delta-EEF |action| > %.4g or non-finite)",
        count,
        int(actions.shape[0]),
        float(threshold),
    )
    mask = outlier.cpu()
    details = {
        "mask": mask,
        "nonfinite": nonfinite.cpu(),
        "action_abs_max": action_abs_max.float().cpu(),
    }
    action_loss_weights = action_loss_weights.clone()
    action_loss_weights[mask] = 0.0
    if video_loss_weights is not None:
        video_loss_weights = video_loss_weights.clone()
        video_loss_weights[mask] = 0.0
    if bool(nonfinite.any().item()):
        actions.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return action_loss_weights, video_loss_weights, count, details


def _raise_nonfinite(reason: str, debug: Dict[str, Any]) -> None:
    fields = []
    for key in (
        "first_nonfinite_param",
        "first_nonfinite_param_shape",
        "first_nonfinite_param_nan_count",
        "first_nonfinite_param_inf_count",
        "pre_forward_first_nonfinite_param",
        "pre_forward_first_nonfinite_param_shape",
        "pre_forward_first_nonfinite_param_nan_count",
        "pre_forward_first_nonfinite_param_inf_count",
        "pre_optimizer_first_nonfinite_param",
        "pre_optimizer_first_nonfinite_param_shape",
        "pre_optimizer_first_nonfinite_param_nan_count",
        "pre_optimizer_first_nonfinite_param_inf_count",
        "post_optimizer_first_nonfinite_param",
        "post_optimizer_first_nonfinite_param_shape",
        "post_optimizer_first_nonfinite_param_nan_count",
        "post_optimizer_first_nonfinite_param_inf_count",
        "first_nonfinite_grad_param",
        "first_nonfinite_grad_shape",
        "first_nonfinite_grad_nan_count",
        "first_nonfinite_grad_inf_count",
    ):
        if key in debug:
            fields.append(f"{key}={debug.get(key)}")
    detail = ", ".join(fields)
    if detail:
        raise RuntimeError(f"Non-finite training state detected: {reason}; {detail}")
    raise RuntimeError(f"Non-finite training state detected: {reason}")


def _list_head(values: Any, n: int = 8) -> Optional[list]:
    if values is None:
        return None
    try:
        return [str(value) for value in list(values)[:n]]
    except Exception:
        return None


def _checkpoint_step(path: Path) -> Optional[int]:
    match = re.fullmatch(r"checkpoint_step_(\d+).*", path.name)
    if not match:
        return None
    return int(match.group(1))


def is_checkpoint_complete(path: Path) -> bool:
    """Return True only when a checkpoint has Motus' completion marker."""
    marker = path / CHECKPOINT_COMPLETE_MARKER
    if not marker.exists():
        return False
    try:
        with marker.open("r") as f:
            payload = json.load(f)
    except Exception:
        return False

    step = _checkpoint_step(path)
    try:
        marker_step = int(payload.get("step", -1))
    except (TypeError, ValueError):
        return False
    return payload.get("status") == "complete" and step is not None and marker_step == step


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the latest complete checkpoint_step_* directory under checkpoint_dir."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return None

    candidates = []
    for path in root.glob("checkpoint_step_*"):
        if not path.is_dir():
            continue
        step = _checkpoint_step(path)
        if step is None:
            continue
        if not is_checkpoint_complete(path):
            logger.warning(
                "Skipping incomplete checkpoint during auto resume: %s "
                "(missing or invalid %s)",
                path,
                CHECKPOINT_COMPLETE_MARKER,
            )
            continue
        candidates.append((step, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return str(candidates[-1][1])


def add_timing_event(events: list, name: str, start_time: float, **extra):
    """Append a startup timing event; also forwarded to the monitor (init phase)."""
    if events is None:
        return
    event = {
        "name": name,
        "seconds": time.perf_counter() - start_time,
    }
    event.update(extra)
    events.append(event)
    # startup.total_* 是全部阶段的总和，不作为独立初始化阶段上报（避免重复计数）
    if _EARLY_MONITOR is not None and not name.startswith("startup.total"):
        _monitor_call(_EARLY_MONITOR, "log_init_phase", name, event["seconds"])


def write_startup_timing(config: OmegaConf, events: list, rank: int, filename: str = "startup_timing.json"):
    """Write startup timing records on rank 0."""
    if rank != 0:
        return

    try:
        total_seconds = sum(
            event.get("seconds", 0.0)
            for event in events
            if event.get("name") != "startup.total_before_train_loop"
        )
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_dir": str(config.system.checkpoint_dir),
            "run_name": config.logging.get("run_name", None) if hasattr(config, "logging") else None,
            "dataset_type": str(config.dataset.get("type", "unknown")) if hasattr(config, "dataset") else "unknown",
            "total_recorded_seconds": total_seconds,
            "events": events,
        }
        output_path = Path(config.system.checkpoint_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Startup timing written to {output_path}")
        logger.info("Startup timing summary:")
        for event in events:
            logger.info(f"  {event['name']}: {event['seconds']:.3f}s")
    except Exception as e:
        logger.warning(f"Failed to write startup timing: {e}")


def get_robot_configs(config: OmegaConf) -> Optional[Dict[str, int]]:
    """Extract robot action dimensions from dataset.robot_names."""
    if not hasattr(config, "dataset") or not hasattr(config.dataset, "robot_names"):
        return None

    robot_names = OmegaConf.to_object(config.dataset.robot_names)
    if isinstance(robot_names, dict):
        robot_configs = {}
        for name, value in robot_names.items():
            if isinstance(value, dict):
                action_dim = value.get("action_dim", config.common.action_dim)
            else:
                action_dim = value
            robot_configs[str(name)] = int(action_dim)
        return robot_configs

    if isinstance(robot_names, list):
        return {str(name): int(config.common.action_dim) for name in robot_names}

    return None


def get_robot_layout(config: OmegaConf) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract unified-action-space layout from dataset.robot_names.

    Returns {name: {"offset": o, "token_index": i, "dim_map": m | None,
    "rot_groups": g | None}}:
      - unified_dim_map: 逐维语义槽位映射（本地第 k 维 → 统一空间列索引的显式
        列表，长度必须 == action_dim；**-1 = 摘除该本地维**（修复 #7，如
        agibot 灵巧手占位维：reader 输出恒 0 的平凡常数目标——不进 scatter/
        loss mask，推理输出端重建为 raw 0））。提供时优先生效，列语义由
        dataset.unified_bands 注册表约束（每列必须落在某个已注册 band 内），
        跨机器人共享同一列 = 显式语义对齐。
      - unified_rot_groups（B1/B2，20260811）：本地旋转维组 → 6D 列组
        [{"local": [本地维..], "repr": rotvec|quat|rot3, "cols": [6 列],
        "quat_order": 可选}]；local 维在 unified_dim_map 对应位置必须为 -1
        （由 rot group 接管，模型边界转 6D 后写 cols），cols 恰 6 列且完整
        落在同一已注册 *_rot6 band 内（跨机器人共享列允许 = 语义对齐）。
      - eef_relative（可选，默认 true；20260812）：B1 EEF 相对化开关。
        绝对位姿动作（agibot/egoverse）保持默认 true——pos 减条件帧
        state、旋转 R_sᵀ·R_i；增量命令动作（libero delta EEF，本身已是
        相对量）置 false——pos 不再减 state、rot group 的 delta 旋转直接
        转 6D（不做 R_s 复合），推理输出端对称跳过还原。
      - unified_validity_gates（可选，20260816）：逐帧有效性门控
        [{"local": 本地 0/1 validity 维, "cols": [start, end) 统一列区间}]——
        validity=0 的帧（及条件帧 state 该位=0 时整窗该手列）从 action
        loss 屏蔽（egoscale 无效手零位姿不计入 loss，见
        configs/stage2_20260815_v1.yaml 头注 6）；local 维在
        unified_dim_map 中必须为 -1（validity 维本身不进统一空间）。
      - unified_offset: 前填起点（缺省 0；仅供未写 dim_map 的旧配置回退，
        与 unified_dim_map 互斥）
      - robot_token_index: 每机器人 8 个可学习 token 的槽位（缺省按 robot_names
        注册顺序 0..N-1 自动分配；YAML 显式指定时禁止复用）
    校验：dim_map 长度 == action_dim、索引 ∈ {-1} ∪ [0, U)、非 -1 列无重复、
    落在已注册 band 内；offset 路径校验 offset+action_dim <=
    common.unified_action_dim；token_index 全局唯一（新增机器人只加 YAML
    条目即可用，模型零改动）。
    """
    if not hasattr(config, "dataset") or not hasattr(config.dataset, "robot_names"):
        return None
    entries = OmegaConf.to_object(config.dataset.robot_names)
    if not isinstance(entries, dict):
        return None  # list 形式无 per-robot 维度，无法构造 layout
    U = int(getattr(config.common, "unified_action_dim", 64))
    default_dim = int(config.common.action_dim)
    # 语义槽位注册表（可选；缺省时 dim_map 不做 band 归属校验，兼容旧配置）
    bands = None
    if hasattr(config.dataset, "unified_bands"):
        raw_bands = OmegaConf.to_object(config.dataset.unified_bands)
        if isinstance(raw_bands, dict) and raw_bands:
            bands = {
                str(band): (int(span[0]), int(span[1]))
                for band, span in raw_bands.items()
            }
    used_indices = set()
    layout = {}
    for order, (name, value) in enumerate(entries.items()):
        if isinstance(value, dict):
            dim = int(value.get("action_dim", default_dim))
            offset = int(value.get("unified_offset", 0))
            token_index = int(value.get("robot_token_index", order))
            raw_map = value.get("unified_dim_map", None)
            raw_groups = value.get("unified_rot_groups", None)
            raw_gates = value.get("unified_validity_gates", None)
            eef_relative = bool(value.get("eef_relative", True))
        else:
            dim = default_dim
            offset = 0
            token_index = order
            raw_map = None
            raw_groups = None
            raw_gates = None
            eef_relative = True
        dim_map = None
        active_cols: list = []
        if raw_map is not None:
            dim_map = [int(v) for v in raw_map]
            if "unified_offset" in value:
                raise ValueError(
                    f"robot {name}: unified_dim_map 与 unified_offset 互斥，只保留 dim_map"
                )
            if len(dim_map) != dim:
                raise ValueError(
                    f"robot {name}: unified_dim_map 长度 {len(dim_map)} != action_dim {dim}"
                )
            # -1 = 摘除维（修复 #7）：不参与重复/越界/band 校验
            active_cols = [c for c in dim_map if c >= 0]
            if len(set(active_cols)) != len(active_cols):
                raise ValueError(f"robot {name}: unified_dim_map 存在重复列索引")
            if any(c < -1 for c in dim_map) or (active_cols and max(active_cols) >= U):
                raise ValueError(
                    f"robot {name}: unified_dim_map 索引越界（允许 -1 摘除维或 [0, {U})）"
                )
            if bands is not None:
                for col in active_cols:
                    if not any(start <= col < end for start, end in bands.values()):
                        raise ValueError(
                            f"robot {name}: unified_dim_map 列 {col} 不在任何已注册 "
                            f"unified_bands 槽位内（先在 YAML 注册语义 band）"
                        )
        elif offset + dim > U:
            raise ValueError(
                f"robot {name}: unified_offset {offset} + action_dim {dim} "
                f"超过 common.unified_action_dim {U}"
            )
        # unified_rot_groups（B1/B2）：本地旋转维组 → 6D 列组
        rot_groups = None
        if raw_groups is not None:
            if dim_map is None:
                raise ValueError(
                    f"robot {name}: unified_rot_groups 需配合 unified_dim_map 使用"
                )
            rot_groups = []
            for gi, group in enumerate(raw_groups):
                local = [int(d) for d in group["local"]]
                cols = [int(c) for c in group["cols"]]
                repr_name = str(group.get("repr", "rotvec"))
                quat_order = str(group.get("quat_order", "xyzw"))
                for d in local:
                    if d < 0 or d >= dim:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] local 维 {d} 超出 "
                            f"action_dim {dim}"
                        )
                    if dim_map[d] != -1:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] local 维 {d} 在 "
                            f"unified_dim_map 中必须为 -1（由 rot group 接管），"
                            f"got {dim_map[d]}"
                        )
                if len(cols) != 6 or len(set(cols)) != 6:
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols 必须恰 6 个不重复列"
                        f"（6D 旋转），got {cols}"
                    )
                if any(c < 0 or c >= U for c in cols):
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols 越界（[0, {U})），got {cols}"
                    )
                if set(cols) & set(active_cols):
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols 与 unified_dim_map "
                        "有效列冲突"
                    )
                if bands is not None:
                    hit = [
                        band_name
                        for band_name, (start, end) in bands.items()
                        if all(start <= c < end for c in cols)
                        and band_name.endswith("_rot6")
                    ]
                    if len(hit) != 1:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] cols {cols} 未完整落在"
                            f"同一已注册 *_rot6 band 内（先在 YAML 注册语义 band）"
                        )
                rot_groups.append({
                    "local": local,
                    "repr": repr_name,
                    "cols": cols,
                    "quat_order": quat_order,
                })
        # unified_validity_gates（20260816）：本地 0/1 validity 维 → 该位为 0
        # 的帧（及条件帧该位为 0 时整窗）从 action loss 屏蔽 cols 统一列
        validity_gates = None
        if raw_gates is not None:
            if dim_map is None:
                raise ValueError(
                    f"robot {name}: unified_validity_gates 需配合 unified_dim_map 使用"
                )
            validity_gates = []
            used_gate_cols: set = set()
            for gi, gate in enumerate(raw_gates):
                local = int(gate["local"])
                cols = [int(c) for c in gate["cols"]]
                if local < 0 or local >= dim:
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] local 维 {local} 超出 "
                        f"action_dim {dim}"
                    )
                if dim_map[local] != -1:
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] local 维 {local} 在 "
                        f"unified_dim_map 中必须为 -1（validity 维本身不进统一空间），"
                        f"got {dim_map[local]}"
                    )
                if len(cols) != 2 or not (0 <= cols[0] < cols[1] <= U):
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] cols 须为 [start, end) 且 "
                        f"0 <= start < end <= {U}，got {cols}"
                    )
                gate_cols = set(range(cols[0], cols[1]))
                if used_gate_cols & gate_cols:
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] cols 与其他 gate 列区间重叠"
                    )
                if bands is not None:
                    for col in gate_cols:
                        if not any(start <= col < end for start, end in bands.values()):
                            raise ValueError(
                                f"robot {name}: validity_gates[{gi}] 列 {col} 不在任何"
                                "已注册 unified_bands 槽位内"
                            )
                used_gate_cols |= gate_cols
                validity_gates.append({"local": local, "cols": (cols[0], cols[1])})
        if token_index in used_indices:
            raise ValueError(
                f"robot {name}: robot_token_index {token_index} 已被其他机器人"
                "占用（禁止复用/重排）"
            )
        used_indices.add(token_index)
        layout[str(name)] = {
            "offset": offset,
            "token_index": token_index,
            "dim_map": dim_map,
            "rot_groups": rot_groups,
            "eef_relative": eef_relative,
            "validity_gates": validity_gates,
        }
    return layout

def setup_logging(rank: int = 0, log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress specific distributed warnings that are noisy but harmless for our flow
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )

def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Calculate derived parameters
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    
    # Validate dataset configuration
    dataset_config = {
        'dataset_type': config.dataset.type,
        'dataset_dir': config.dataset.dataset_dir if hasattr(config.dataset, 'dataset_dir') else None,
        'global_downsample_rate': config.common.global_downsample_rate,
        'video_action_freq_ratio': config.common.video_action_freq_ratio,
        'num_video_frames': config.common.num_video_frames
    }
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Dataset type: {config.dataset.type}")
    if hasattr(config, 'training_mode'):
        logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Action chunk size: {config.common.action_chunk_size}")
    logger.info(f"Video frames: {config.common.num_video_frames}")
    
    return config

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

class UniDiffuserTrainer:
    """Trainer class for Motus."""
    
    def __init__(
        self,
        model: UBT_WM,
        train_dataloader: Any,  # data_prepare 的 IndexedMixedTrainingDataPipeline
        val_dataloader: Optional[Any] = None,  # 同上（queue_enabled=False）
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: str = "./checkpoints_stage4",
        log_interval: int = 100,
        save_interval: int = 1000,
        val_interval: int = 1000,
        report_to: str = "wandb",
        tb_writer: Optional[SummaryWriter] = None,
        accelerator: Optional[Any] = None,
        config: Optional[Any] = None,
        startup_timing_events: Optional[list] = None,
        startup_start_time: Optional[float] = None,
        monitor: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        if monitor is not None:
            self.monitor = monitor
        elif NullMonitor is not None:
            self.monitor = NullMonitor()
        else:
            self.monitor = None
        
        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.report_to = report_to
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config
        self.startup_timing_events = startup_timing_events
        self.startup_start_time = startup_start_time

        # ready queue 开启时，depth 已由 producer 在后台 attach；复用同一 attacher
        # 作为同步 fallback（queue 关闭或外部 batch 未带 depth 时才真正读取）。
        self._depth_attacher = getattr(
            train_dataloader, "_repo_depth_attacher", None
        )
        if self._depth_attacher is None:
            self._depth_attacher = self._maybe_build_depth_attacher(config)
        else:
            logger.info(
                "DepthAttacher enabled in ready-queue producer: "
                "mode=%s z=[%.3f,%.3f] attach_workers=%d",
                self._depth_attacher.mode,
                self._depth_attacher.z_near,
                self._depth_attacher.z_far,
                self._depth_attacher.attach_workers,
            )
        self._ego_pose_cache = self._maybe_build_ego_pose_cache(config)
        # 历史帧（model.wm.history_enable）：与 depth 同为 batch 后附加，同步读取。
        from thinker_wm.data_prepare.history_attach import build_history_attacher
        self._history_attacher = build_history_attacher(config)
        if self._history_attacher is not None:
            logger.info("HistoryAttacher enabled: spec=%s readers=%s",
                        self._history_attacher.spec, sorted(self._history_attacher.readers))
        
        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        self.source_dataset_counts = Counter()
        timing_cfg = getattr(self.config, "timing", None) if self.config is not None else None
        self.step_timing_enabled = bool(timing_cfg.get("enabled", True)) if timing_cfg is not None else True
        self.step_timing_sync_cuda = bool(timing_cfg.get("sync_cuda", True)) if timing_cfg is not None else True
        self.step_timing_interval = int(timing_cfg.get("interval", 100)) if timing_cfg is not None else 100
        self.step_timing_writer = StepTimingWriter(
            checkpoint_dir=str(self.checkpoint_dir),
            rank=self.rank,
            enabled=self.step_timing_enabled,
        )
        diagnostics_cfg = getattr(self.config, "diagnostics", None) if self.config is not None else None
        self.action_diagnostics_enabled = (
            bool(diagnostics_cfg.get("action_diagnostics_enabled", True))
            if diagnostics_cfg is not None
            else True
        )
        self.action_diagnostics_interval = int(
            diagnostics_cfg.get("action_diagnostics_interval", 50)
            if diagnostics_cfg is not None
            else 50
        )
        self.action_loss_anomaly_threshold = float(
            diagnostics_cfg.get("action_loss_anomaly_threshold", 100.0)
            if diagnostics_cfg is not None
            else 100.0
        )
        self.action_tensor_abs_anomaly_threshold = float(
            diagnostics_cfg.get("action_tensor_abs_anomaly_threshold", 50.0)
            if diagnostics_cfg is not None
            else 50.0
        )
        self.grad_norm_anomaly_threshold = float(
            diagnostics_cfg.get("grad_norm_anomaly_threshold", 1000.0)
            if diagnostics_cfg is not None
            else 1000.0
        )
        self.action_diagnostics_writer = JsonlDiagnosticsWriter(
            checkpoint_dir=str(self.checkpoint_dir),
            rank=self.rank,
            name="action_diagnostics",
            enabled=self.action_diagnostics_enabled,
        )
        self.heartbeat_enabled = (
            _as_bool(diagnostics_cfg.get("heartbeat_enabled", _env_flag("MOTUS_HEARTBEAT_ENABLED", False)))
            if diagnostics_cfg is not None
            else _env_flag("MOTUS_HEARTBEAT_ENABLED", False)
        )
        self.heartbeat_memory_interval = int(
            diagnostics_cfg.get("heartbeat_memory_interval", os.environ.get("MOTUS_HEARTBEAT_MEMORY_INTERVAL", 10))
            if diagnostics_cfg is not None
            else os.environ.get("MOTUS_HEARTBEAT_MEMORY_INTERVAL", 10)
        )
        # 非有限扫描（全参数/全梯度 isfinite 逐 tensor 扫描 + 跨 rank 集合通信）
        # 的执行间隔（评审 #12）：1 = 每步（旧行为）；>1 时每 N 步扫描一次，
        # 间隔步不拦截非有限梯度/参数（loss 标量检查仍每步执行）。与
        # MOTUS_NONFINITE_FAIL_FAST 绑定：fail-fast 关闭时扫描整体关闭
        # （旧行为是扫描照跑只是不 raise，白白付出每步上千次 CUDA 同步）。
        self.nonfinite_scan_interval = max(
            1,
            int(
                diagnostics_cfg.get("nonfinite_scan_interval", 1)
                if diagnostics_cfg is not None
                else 1
            ),
        )
        if self.nonfinite_scan_interval > 1 or not _nonfinite_fail_fast_enabled():
            logger.info(
                "Nonfinite scans: interval=%d fail_fast_enabled=%s (loss check "
                "still runs every step)",
                self.nonfinite_scan_interval,
                _nonfinite_fail_fast_enabled(),
            )
        self.heartbeat_writer = JsonlDiagnosticsWriter(
            checkpoint_dir=str(self.checkpoint_dir),
            rank=self.rank,
            name="heartbeat",
            enabled=self.heartbeat_enabled,
        )
        system_cfg = getattr(self.config, "system", None) if self.config is not None else None
        self.train_loop_gc_interval = int(
            system_cfg.get("train_loop_gc_interval", 100)
            if system_cfg is not None
            else 100
        )
        monitor_cfg = getattr(self.config, "monitor", None) if self.config is not None else None
        # monitor 指标记录间隔（默认 1 = 每步；与 console/TB 的 system.log_interval 独立）
        self.monitor_metrics_interval = int(
            monitor_cfg.get("metrics_interval", 1)
            if monitor_cfg is not None
            else 1
        )
        # 周期性定集评测（periodic_eval 包；未启用/初始化失败时为 no-op）。
        # 构造时若评测素材缓存缺失，会拉起 CPU-only 后台准备进程预解码
        # （indexed 源 holdout 固定 batch / EWMBenchmark 固定 shard，
        # 带 manifest + sha256，跨训练任务复用），训练前台永远不解码视频。
        self.periodic_evaluator = build_periodic_evaluator(
            config,
            rank=rank,
            world_size=world_size,
            monitor=self.monitor,
            checkpoint_dir=str(self.checkpoint_dir),
        )

        logger.info(f"UBT_WM Trainer initialized on rank {rank}/{world_size}")
        logger.info(f"Logging backends: {report_to}")
        if self.step_timing_enabled:
            logger.info(
                "Step timing JSONL enabled: path=%s sync_cuda=%s",
                self.step_timing_writer.path,
                self.step_timing_sync_cuda,
            )
        if self.action_diagnostics_enabled:
            logger.info(
                "Action diagnostics JSONL enabled: path=%s interval=%d "
                "action_loss_threshold=%.3g tensor_abs_threshold=%.3g grad_norm_threshold=%.3g",
                self.action_diagnostics_writer.path,
                self.action_diagnostics_interval,
                self.action_loss_anomaly_threshold,
                self.action_tensor_abs_anomaly_threshold,
                self.grad_norm_anomaly_threshold,
            )
        if self.heartbeat_enabled:
            logger.info(
                "Heartbeat diagnostics JSONL enabled: path=%s memory_interval=%d gc_interval=%d",
                self.heartbeat_writer.path,
                self.heartbeat_memory_interval,
                self.train_loop_gc_interval,
            )

    def save_checkpoint(self, suffix: str = ""):
        """Save complete training state using accelerator."""
        timing = {}
        total_start = time.perf_counter()
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        marker_path = checkpoint_dir / CHECKPOINT_COMPLETE_MARKER

        marker_start = time.perf_counter()
        if self.accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if marker_path.exists():
                marker_path.unlink()
                logger.info(f"Removed stale checkpoint completion marker: {marker_path}")
        self.accelerator.wait_for_everyone()
        timing["checkpoint_prepare_seconds"] = time.perf_counter() - marker_start
        
        # Use accelerator to save complete training state
        # This saves model, optimizer, scheduler, dataloader, and RNG states
        save_state_start = time.perf_counter()
        self.accelerator.save_state(str(checkpoint_dir))
        timing["accelerator_save_state_seconds"] = time.perf_counter() - save_state_start
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        # Also save a config.json alongside weights for reproducibility
        config_start = time.perf_counter()
        try:
            from omegaconf import OmegaConf as _OmegaConf
            cfg_dict = _OmegaConf.to_container(self.config, resolve=True) if self.config is not None else {}
            # Filter only requested sections
            common = cfg_dict.get("common", {})
            model = cfg_dict.get("model", {})
            filtered = {
                "common": common,
                "action_expert": model.get("action_expert", {}),
                "und_expert": model.get("und_expert", {}),
                "time_distribution": model.get("time_distribution", {}),
                "ema": model.get("ema", {}),
            }
            import json as _json
            with open(checkpoint_dir / "config.json", "w") as f:
                _json.dump(filtered, f, indent=2)
            logger.info(f"Wrote config.json to {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")
        timing["checkpoint_config_seconds"] = time.perf_counter() - config_start

        marker_start = time.perf_counter()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            marker_payload = {
                "status": "complete",
                "step": int(self.global_step),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "world_size": int(self.world_size),
                "rank": int(self.rank),
                "checkpoint_dir": str(checkpoint_dir),
            }
            with marker_path.open("w") as f:
                json.dump(marker_payload, f, indent=2)
            logger.info(f"Checkpoint complete marker written to {marker_path}")
        self.accelerator.wait_for_everyone()
        timing["checkpoint_marker_seconds"] = time.perf_counter() - marker_start
        self._last_checkpoint_step = int(self.global_step)
        timing["checkpoint_total_seconds"] = time.perf_counter() - total_start
        _monitor_call(self.monitor, "log_checkpoint", self.global_step, str(checkpoint_dir), timing)
        return timing
    
    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = False):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            reset_scheduler: If True, reset scheduler to new config instead of
                loading from checkpoint. Default False: 弹性续训接着原 LR 计划
                走、不重新 warmup（评审 #13）。
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
            return
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Extract step number from checkpoint path (e.g., checkpoint_step_125000)
        step_match = re.search(r'step_(\d+)', checkpoint_path)
        if step_match:
            self.global_step = int(step_match.group(1))
            logger.info(f"Resuming from step {self.global_step}")
        else:
            logger.warning(f"Could not extract step number from {checkpoint_path}, starting from step 0")

        # Load using accelerator (includes model, optimizer, scheduler states).
        # First attempt uses the legacy ZeRO sharded format. A ZeRO world-size
        # mismatch means the checkpoint was saved under a different DP world
        # size; legacy loading cannot repartition optimizer state, so we convert
        # the checkpoint to universal format once and retry full state restore.
        ds_engine = self._deepspeed_engine()
        if ds_engine is not None:
            ds_engine._config.load_universal_checkpoint = False
        try:
            self.accelerator.load_state(checkpoint_path)
        except Exception as e:
            # ZeRORuntimeException inherits from Exception (NOT RuntimeError),
            # so we must catch broadly and filter by message below.
            if not self._is_zero_world_size_mismatch(e):
                raise
            logger.warning(
                "ZeRO checkpoint world-size mismatch: %s. Converting %s to "
                "universal format and retrying full state restore.",
                e,
                checkpoint_path,
            )
            converted = self._ensure_universal_checkpoint(checkpoint_path)
            if ds_engine is not None:
                ds_engine._config.load_universal_checkpoint = True
            try:
                self.accelerator.load_state(checkpoint_path)
            except Exception as e2:
                logger.error(
                    "Universal-format load also failed for %s (conversion was "
                    "performed on the fly: %s). Aborting instead of degrading "
                    "to weight-only loading.",
                    checkpoint_path,
                    converted,
                )
                raise
            load_event = "resume_universal_converted" if converted else "resume_universal"
            logger.info(
                "Checkpoint loaded via universal format from %s (%s)",
                checkpoint_path,
                "converted on the fly" if converted else "reused existing universal",
            )
        else:
            load_event = "resume"
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
        self._last_checkpoint_step = int(self.global_step)
        _monitor_call(self.monitor, "log_event", load_event, path=checkpoint_path, step=self.global_step)
        
        # Reset scheduler with new config if requested
        if reset_scheduler and self.config is not None and self.scheduler is not None:
            logger.info("Resetting scheduler to new configuration (not using checkpoint scheduler state)...")
            
            # Unwrap scheduler if it's wrapped by accelerator
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module
            
            # Check if it's our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'warm_up_steps'):
                # Update scheduler parameters with new config
                unwrapped_scheduler.warm_up_steps = self.config.training.warmup_steps
                unwrapped_scheduler.cycle_length = self.config.training.cycle_length
                unwrapped_scheduler.f_max = self.config.training.f_max
                unwrapped_scheduler.f_min = self.config.training.f_min
                # Keep the configured base rates. Optimizer groups may currently
                # hold a scheduled rate loaded from the checkpoint.
                unwrapped_scheduler.step_count = 0
                if hasattr(unwrapped_scheduler, "apply_scheduled_lrs"):
                    unwrapped_scheduler.apply_scheduled_lrs()
                
                logger.info(f"Updated scheduler config: warmup={unwrapped_scheduler.warm_up_steps}, "
                          f"cycle_length={unwrapped_scheduler.cycle_length}, "
                          f"f_max={unwrapped_scheduler.f_max}, f_min={unwrapped_scheduler.f_min}")
                logger.info(f"Base learning rates: {[f'{lr:.2e}' for lr in unwrapped_scheduler.base_lrs]}")
                logger.info(
                    "Reset scheduler to warmup start: %s",
                    [f"{lr:.2e}" for lr in unwrapped_scheduler.get_last_lr()],
                )
            
            # Log current learning rate (from checkpoint)
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (will be overridden by scheduler): {current_lr:.2e}")
        elif self.scheduler is not None:
            # If not resetting scheduler, sync scheduler progress with global_step
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module

            # Case 1: our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'step_count'):
                old_step_count = unwrapped_scheduler.step_count
                unwrapped_scheduler.step_count = self.global_step
                if hasattr(unwrapped_scheduler, "apply_scheduled_lrs"):
                    unwrapped_scheduler.apply_scheduled_lrs()
                logger.info(f"Synchronized scheduler step_count: {old_step_count} -> {self.global_step}")

            # Case 2: diffusers_cosine wrapper with inner scheduler
            if hasattr(unwrapped_scheduler, 'inner') and hasattr(unwrapped_scheduler.inner, 'last_epoch'):
                try:
                    old_epoch = int(getattr(unwrapped_scheduler.inner, 'last_epoch', -1))
                except Exception:
                    old_epoch = -1
                # Align inner scheduler epoch with current global_step so schedule continues
                unwrapped_scheduler.inner.last_epoch = int(self.global_step)
                logger.info(f"Aligned diffusers scheduler last_epoch: {old_epoch} -> {self.global_step}")

            # Log current optimizer LR (authoritative)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (optimizer): {current_lr:.2e}")

    @staticmethod
    def _is_zero_world_size_mismatch(error: Exception) -> bool:
        """Detect DeepSpeed ZeRO checkpoint world-size mismatch errors.

        Matches errors like: "The checkpoint being loaded used a DP world size
        of 24 but the current world size is 8. Automatic adjustment of ZeRO's
        optimizer state partitioning with a new world size is not currently
        supported." (ZeRORuntimeException inherits from Exception, not
        RuntimeError, hence the broad except at the call site.)
        """
        msg = str(error).lower()
        return "world size" in msg and (
            "zero" in msg or "optimizer state partitioning" in msg
        )

    def _fallback_load_weights_only(self, checkpoint_path: str, original_error: Exception):
        """Fallback for ZeRO world-size mismatch: load model weights only.

        DEPRECATED: superseded by automatic universal-format conversion
        (_ensure_universal_checkpoint). Kept only for reference; no callers.

        ZeRO partitions optimizer state by DP world size, so a checkpoint saved
        with a different world size cannot be fully resumed. Model weights are
        replicated across ranks under ZeRO-1, so we restore them from
        mp_rank_00_model_states.pt and restart with a fresh optimizer,
        scheduler, and iteration counter (equivalent to FINETUNE_CHECKPOINT).
        """
        logger.warning(
            "ZeRO checkpoint world-size mismatch: %s. Falling back to weight-only "
            "loading from %s — model weights will be restored, but optimizer "
            "state, scheduler progress, RNG and iteration counter are reset; "
            "training restarts from step 0.",
            original_error,
            checkpoint_path,
        )
        self.global_step = 0

        unwrapped_model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )
        if not hasattr(unwrapped_model, "load_pretrain_weights"):
            raise RuntimeError(
                "ZeRO world-size mismatch fallback requires load_pretrain_weights() "
                "on the model, but it is not available."
            ) from original_error
        if getattr(unwrapped_model.config, "training_mode", "finetune") != "finetune":
            raise RuntimeError(
                "ZeRO world-size mismatch fallback uses load_pretrain_weights(), "
                "which requires training_mode='finetune', got "
                f"{getattr(unwrapped_model.config, 'training_mode', None)!r}."
            ) from original_error
        unwrapped_model.load_pretrain_weights(checkpoint_path)

        # DeepSpeed ZeRO with bf16 (BF16_Optimizer) keeps fp32 master weights
        # inside the optimizer and copies them back into the module on every
        # optimizer step (update_lp_params). Weights loaded directly into the
        # module after accelerator.prepare() would be silently reverted by the
        # stale fp32 master at the first step, so push the freshly loaded
        # weights into the fp32 master copies as well.
        ds_optimizer = getattr(self.model, "optimizer", None)
        if hasattr(ds_optimizer, "refresh_fp32_params"):
            ds_optimizer.refresh_fp32_params()
            logger.info(
                "Synced fallback-loaded weights into DeepSpeed fp32 master "
                "params (refresh_fp32_params)."
            )
        logger.warning(
            "Weight-only fallback load complete from %s; continuing from step 0 "
            "with a fresh optimizer/scheduler.",
            checkpoint_path,
        )


    def _deepspeed_engine(self):
        """Return the DeepSpeed engine wrapped by accelerate, or None."""
        if self.accelerator is None:
            return None
        wrapped = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
        if wrapped is None:
            return None
        return getattr(wrapped, "engine", None)

    def _ensure_universal_checkpoint(self, checkpoint_path: str) -> bool:
        """Ensure a universal-format `zero/` directory exists for the checkpoint.

        DeepSpeed's universal loader expects
        {checkpoint_path}/pytorch_model/zero/optimizer_state.pt, while
        save_checkpoint always writes legacy bf16_zero_pp_rank_* shards. On a
        ZeRO world-size mismatch we convert the legacy checkpoint to universal
        format once (main process only; other ranks poll for completion) and
        move the resulting zero/ directory under pytorch_model/. Returns True if
        this process performed the conversion, False if the universal artifacts
        were already present (reused).
        """
        zero_dir = os.path.join(checkpoint_path, "pytorch_model", "zero")
        optimizer_state_path = os.path.join(zero_dir, "optimizer_state.pt")
        ready_marker = os.path.join(zero_dir, ".universal_ready")
        if os.path.isfile(optimizer_state_path) and os.path.isfile(ready_marker):
            return False

        if getattr(self.accelerator, "is_main_process", True):
            pytorch_model = os.path.join(checkpoint_path, "pytorch_model")
            if not os.path.isdir(pytorch_model):
                raise FileNotFoundError(
                    f"DeepSpeed checkpoint directory not found: {pytorch_model}"
                )
            tmp_out = os.path.join(checkpoint_path, ".universal_conversion")
            if os.path.isdir(tmp_out):
                shutil.rmtree(tmp_out, ignore_errors=True)
            os.makedirs(tmp_out, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "deepspeed.checkpoint.ds_to_universal",
                "--input_folder", pytorch_model,
                "--output_folder", tmp_out,
                "--num_extract_workers", str(UNIVERSAL_CONVERSION_WORKERS),
                "--num_merge_workers", str(UNIVERSAL_CONVERSION_MERGE_WORKERS),
                # deepspeed 0.18.x 的保存路径不写 universal_checkpoint_info，
                # 不传本 flag 时 ds_to_universal 的 _check_for_required_state
                # 必然断言失败；该 flag 只补内存中的 global_state，不影响产物。
                "--inject_missing_state",
            ]
            log_file = os.path.join(tmp_out, "conversion.log")
            t0 = time.perf_counter()
            logger.info(
                "Converting %s to universal format (this may take 30min-2h): %s",
                checkpoint_path,
                " ".join(cmd),
            )
            try:
                with open(log_file, "w") as lf:
                    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to launch ds_to_universal conversion for {checkpoint_path}: {exc}"
                ) from exc
            if proc.returncode != 0:
                tail = ""
                try:
                    with open(log_file) as lf:
                        tail = lf.read()[-8000:]
                except OSError:
                    pass
                logger.error("ds_to_universal conversion log tail:\n%s", tail)
                raise RuntimeError(
                    f"ds_to_universal conversion failed for {checkpoint_path} "
                    f"(exit code {proc.returncode}); see {log_file}"
                )
            src_zero = os.path.join(tmp_out, "zero")
            if not os.path.isdir(src_zero):
                raise RuntimeError(
                    f"ds_to_universal produced no zero/ directory at {src_zero}"
                )
            os.makedirs(zero_dir, exist_ok=True)
            for name in sorted(os.listdir(src_zero)):
                src = os.path.join(src_zero, name)
                dst = os.path.join(zero_dir, name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(tmp_out, ignore_errors=True)
            if not os.path.isfile(optimizer_state_path):
                raise RuntimeError(
                    f"Universal conversion did not produce {optimizer_state_path}"
                )
            with open(ready_marker, "w") as f:
                f.write(datetime.now().isoformat(timespec="seconds"))
            logger.info(
                "Converted %s to universal format in %.1fs",
                checkpoint_path,
                time.perf_counter() - t0,
            )
        else:
            # Non-main ranks: wait for the main process to finish converting.
            # No collectives are issued while waiting, to avoid NCCL timeouts.
            deadline = time.monotonic() + UNIVERSAL_CONVERSION_TIMEOUT
            while not (os.path.isfile(optimizer_state_path) and os.path.isfile(ready_marker)):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out after {UNIVERSAL_CONVERSION_TIMEOUT}s waiting for "
                        f"universal conversion of {checkpoint_path}"
                    )
                time.sleep(UNIVERSAL_CONVERSION_POLL_INTERVAL)
        return True

    def _write_heartbeat(
        self,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
        include_memory: bool = False,
        ready_queue: Optional[Any] = None,
    ) -> None:
        if not self.heartbeat_enabled:
            return
        record: Dict[str, Any] = {
            "phase": str(phase),
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "world_size": int(self.world_size),
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "time_unix": time.time(),
            "time_monotonic": time.perf_counter(),
        }
        if extra:
            for key, value in extra.items():
                if key in {"batch", "metrics"}:
                    continue
                record[key] = value
        if ready_queue is not None:
            try:
                record.update(ready_queue.snapshot())
            except Exception as exc:
                record["ready_queue_snapshot_error"] = str(exc)
        if include_memory:
            record.update(_memory_snapshot("memory"))
        self.heartbeat_writer.write(record)

    def _collect_outlier_episode_records(
        self,
        batch: Dict[str, Any],
        details: Optional[Dict[str, Any]],
    ) -> list:
        """把离群样本（统一 delta-EEF 空间 abs max 超阈值或 NaN/Inf）映射回 episode id 并跨 rank 汇总。

        所有 rank 在 train_step 同一位置调用（内部集合通信保持一致），返回合并
        后的记录列表（每条含 rank/episode_id/source_dataset/reason/abs_max），
        供 step_record 落盘与 monitor 事件使用；无离群时返回空列表。
        """
        local: list = []
        if details is not None:
            episode_ids = batch.get("episode_ids") or []
            source_datasets = batch.get("source_datasets") or []
            robot_names = batch.get("robot_names") or []
            mask = details["mask"]
            nonfinite = details["nonfinite"]
            action_abs_max = details["action_abs_max"]
            state_abs_max = details["state_abs_max"]
            for idx in mask.nonzero().flatten().tolist():
                record: Dict[str, Any] = {
                    "rank": int(self.rank),
                    "sample_index": int(idx),
                    "reason": "nonfinite" if bool(nonfinite[idx]) else "magnitude",
                    "action_abs_max": float(action_abs_max[idx]),
                    "state_abs_max": float(state_abs_max[idx]),
                }
                if idx < len(episode_ids):
                    record["episode_id"] = str(episode_ids[idx])
                if idx < len(source_datasets):
                    record["source_dataset"] = str(source_datasets[idx])
                if idx < len(robot_names):
                    record["robot_name"] = str(robot_names[idx])
                local.append(record)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            # 先 all_reduce 计数，全集群无离群时省去 all_gather_object 开销。
            flag = torch.tensor([len(local)], device=self.device)
            dist.all_reduce(flag)
            if int(flag.item()) <= 0:
                return []
            gathered: list = [None] * self.world_size
            dist.all_gather_object(gathered, local)
            merged: list = []
            for rank_records in gathered:
                merged.extend(rank_records or [])
            return merged
        return local

    @staticmethod
    def _maybe_build_depth_attacher(config: Optional[Any]):
        """enable_depth 且配置了 depth sidecar roots 时构建 DepthAttacher，否则
        None。roots 缺失但 enable_depth=True 时报错（避免静默无 depth 训练）。"""
        from thinker_wm.data_prepare.depth_attach import build_depth_attacher

        attacher = build_depth_attacher(config)
        if attacher is None:
            return None
        logger.info(
            "DepthAttacher enabled (synchronous fallback): "
            "mode=%s z=[%.3f,%.3f] attach_workers=%d",
            attacher.mode,
            attacher.z_near,
            attacher.z_far,
            attacher.attach_workers,
        )
        return attacher

    @staticmethod
    def _maybe_build_ego_pose_cache(config: Optional[Any]):
        """Ego motion mask 用 head/pose 对齐；Agi 无相机 pose，cache 对 Agi 样本空过。"""
        if config is None:
            return None
        wm = getattr(getattr(config, "model", None), "wm", None)
        if wm is None:
            return None
        if not bool(getattr(wm, "depth_motion_loss_enable", False)):
            return None
        if not bool(getattr(wm, "depth_motion_ego_pose", False)):
            return None
        from thinker_wm.models.ego_pose import EgoPoseCache
        logger.info("EgoPoseCache enabled: Ego pose-aligns before 3-gate; Agi stays 3-gate")
        return EgoPoseCache(config)

    def prepare_train_batch(
        self,
        batch: Dict[str, Any],
        timing: Any = None,
        heartbeat: Optional[Callable[[str, Optional[Dict[str, Any]], bool], None]] = None,
        model_timing: Optional[Dict[str, Any]] = None,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """训练步共享的 batch 准备：host→device、元数据提取、离群样本 loss 置零，
        以及模型侧噪声无关上下文（action/state 归一化 + VAE clean latent，
        ``UBT_WM.prepare_training_context``）。

        ``training.train_steps_per_vae_encode > 1`` 时 train 循环每个 batch 调
        一次并把产物跨 N 个 ``train_step`` 复用（VAE encode 只做一次，各步
        重新采样噪声）；默认每步一次，行为与拆分前一致。返回 dict 同时带有
        ``outlier_zeroed``/``outlier_episodes`` 供 train_step 写入 timing。
        """
        if timing is None:
            timing = TimingStack(enabled=False)
        _is_stack = isinstance(timing, TimingStack) if TimingStack is not None else False
        if not _is_stack:
            timing = TimingStack(enabled=False)

        if heartbeat is None:
            def heartbeat(_phase: str, _extra: Optional[Dict[str, Any]] = None, include_memory: bool = False) -> None:
                return None

        # depth 第三生成流：读与采样 clip 对齐的米制深度并附到 batch（depth_first
        # /depth_frames/depth_present）。未启用 depth 时 self._depth_attacher=None，
        # 跳过（batch 无 depth_* 键 → 下游 depth 全链路 None）。
        if getattr(self, "_depth_attacher", None) is not None:
            with timing.span("depth_attach"):
                batch = self._depth_attacher.attach(batch)
        history_inputs = None
        if getattr(self, "_history_attacher", None) is not None:
            from thinker_wm.data_prepare.history_attach import history_inputs_from_batch
            with timing.span("history_attach"):
                batch = self._history_attacher.attach(batch, step=int(self.global_step))
            history_inputs = history_inputs_from_batch(batch, self.device)

        # --- host_to_device ---
        heartbeat("host_to_device_start")
        with timing.span("host_to_device"):
            first_frame = batch['first_frame'].to(self.device, dtype=self.dtype)
            video_frames = batch['video_frames'].to(self.device, dtype=self.dtype)
            language_embeddings = batch['language_embedding']
            if language_embeddings is not None:
                language_embeddings = language_embeddings.to(self.device, dtype=self.dtype)
            state = batch.get('initial_state', None)
            if state is not None:
                # qpos 数值和归一化保留 FP32；进入网络线性层时 autocast 会自动
                # 转成 BF16。避免在 z-score 前量化原始关节/夹爪值。
                state = state.to(self.device, dtype=torch.float32)
            actions = batch['action_sequence'].to(self.device, dtype=torch.float32)
            vlm_inputs = batch['vlm_inputs']
            if vlm_inputs is not None:
                vlm_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in vlm_inputs.items()}
            # depth 流（可选）：[0,1] 3 通道，与 RGB 同 device/dtype；缺失则 None。
            depth_first = batch.get("depth_first")
            depth_frames = batch.get("depth_frames")
            depth_present = batch.get("depth_present")
            # Legacy configurations keep their original depth reduction; the
            # RGB-only recipe explicitly opts into valid-pixel supervision.
            depth_valid_mask = (
                batch.get("depth_valid_mask")
                if str(getattr(self.config.model.wm, "depth_condition_mode", "first_frame")) == "rgb_only"
                else None
            )
            if depth_first is not None:
                depth_first = depth_first.to(self.device, dtype=self.dtype)
            if depth_frames is not None:
                depth_frames = depth_frames.to(self.device, dtype=self.dtype)
            if depth_present is not None:
                depth_present = depth_present.to(self.device)
            if depth_valid_mask is not None:
                depth_valid_mask = depth_valid_mask.to(self.device, dtype=torch.bool)
        heartbeat("host_to_device_done", {"host_to_device_seconds": timing.get("host_to_device_seconds", 0.0)})

        ego_head_pose = ego_head_valid = ego_K = ego_pose_used = None
        if getattr(self, "_ego_pose_cache", None) is not None:
            with timing.span("ego_pose_attach"):
                T = int(video_frames.shape[1]) + 1
                # source_name 是 mix 子源名（queue_stats.mix_source_name）；
                # batch["source_datasets"] 是 catalog 家族名（egoscale），
                # 拿不到 release/index，开不了 repo。
                ego_head_pose, ego_head_valid, ego_K, ego_pose_used = (
                    self._ego_pose_cache.attach_tensors(
                        source_name,
                        batch.get("episode_ids") or [],
                        batch.get("frame_indices") or [],
                        T,
                    )
                )

        # --- extract batch metadata ---
        first_frame_pils = batch.get("first_frame_pils", None)
        t5_input_messages = batch.get("t5_input_messages", None)
        vlm_input_messages = batch.get("vlm_input_messages", None)
        if (
            bool(getattr(self.config.model, "unified_vlm_task_prompt", False))
            and t5_input_messages
        ):
            # 统一 VLM task prompt（见 utils.vlm_utils）：用原始任务文本
            # （t5 消息）重建 vlm 消息，替代数据管线拼的 VLM_PROMPT 前缀。
            # 训练态且配置了字段 dropout（vlm_field_dropout_prob>0）时逐样本
            # 对 Scene/Task/Action/Outcome 四字段随机 drop（被 drop 字段从格式
            # 指令中整字段移除、保底留 1）；drop mask 在本函数内每 batch 采一次，
            # train_steps_per_vae_encode>1 时同一 mask 随 prepared 被复用 latent
            # 的 N 个 update step 共用（不同于 condition dropout——后者在
            # training_step 内每个 update step 重采）。
            # prompt_mode=text_only 的源（dataset.sources[*].prompt_mode，如
            # openvid/egoverse）：不套四字段机器人模板、不带首帧图，只用 VLM
            # tokenizer + 语言模型编码原始文本（batch 内单源，collate 保证）。
            prompt_mode = resolve_prompt_mode(self.config, batch.get("source_datasets"))
            if prompt_mode == PROMPT_MODE_TEXT_ONLY:
                vlm_input_messages = [str(text).strip() for text in t5_input_messages]
                first_frame_pils = None
            elif bool(getattr(self.config.model, "vlm_text_rewrite", False)):
                # VLM 文本改写（20260819 v2，见 utils.vlm_utils）：消息改为压缩
                # 改写指令（纯文本、不带首帧图；输出 Task/Action + Scene 若有；
                # 按数据集家族分模板，egoscale 禁复述标注元数据），由
                # UndBranch._rewrite_prompts 生成短文本后再编码；字段 dropout
                # 20260818_v2 起改在改写输出侧作用（und_branch，缓存之后、
                # 不产生新 cache key），此处的输入侧路径不再适用。
                source_key = (batch.get("source_datasets") or [None])[0]
                vlm_input_messages = [
                    build_vlm_rewrite_message(text, source_key)
                    for text in t5_input_messages
                ]
            else:
                field_drop_prob = float(getattr(self.config.model, "vlm_field_dropout_prob", 0.0))
                prompt_style = resolve_vlm_prompt_style(self.config)
                # robotwin_tlayout 只有 T 形说明 + Task 一句，没有四字段可 drop。
                use_field_drop = (
                    self.model.training
                    and field_drop_prob > 0
                    and prompt_style != VLM_PROMPT_STYLE_ROBOTWIN_TLAYOUT
                )
                vlm_input_messages = [
                    build_vlm_task_message(
                        text,
                        drop_fields=(
                            sample_vlm_field_dropout(field_drop_prob)
                            if use_field_drop
                            else frozenset()
                        ),
                        style=prompt_style,
                    )
                    for text in t5_input_messages
                ]
        robot_names = batch.get("robot_names", None)
        source_datasets = batch.get("source_datasets", None)
        action_loss_weights = batch.get("action_loss_weights", None)
        video_loss_weights = batch.get("video_loss_weights", None)
        state_action_policies = batch.get("state_action_policies", None)

        # --- 离群样本 loss 置零（库内 guard raise 已关闭，训练侧兜底）---
        # 样本相关、噪声无关：每 batch 只做一次，复用 latent 的 N 个更新步共用。
        # 分两段（delta EEF 目标表示下 raw 绝对值语义已过时，state 不再单独检查）：
        #   pre（context 前）：raw actions/state 非有限防御扫描 + nan_to_num，
        #     命中样本 loss 权重置零（摘除维 NaN 传播不到统一张量，必须前置拦）；
        #   post（context 后）：统一 delta-EEF 张量 [B,T,U] 样本级 abs max 阈值
        #     检查（_zero_outlier_sample_loss_weights）。
        outlier_threshold = _outlier_loss_zero_threshold()
        pre_outlier = None
        if outlier_threshold > 0 and action_loss_weights is not None:
            with timing.span("outlier_pre_check"):
                pre_outlier = _sanitize_nonfinite_batch(actions, state)
            if pre_outlier is not None and bool(pre_outlier["mask"].any().item()):
                pre_mask = pre_outlier["mask"]
                action_loss_weights = action_loss_weights.clone()
                action_loss_weights[pre_mask] = 0.0
                if video_loss_weights is not None:
                    video_loss_weights = video_loss_weights.clone()
                    video_loss_weights[pre_mask] = 0.0

        # Load verified window annotations once; no sampler/label guessing.
        from thinker_wm.data_prepare.capability_annotations import CapabilityAnnotations, annotation_paths
        if not hasattr(self, "_capability_annotations"):
            paths = annotation_paths(self.config)
            self._capability_annotations = CapabilityAnnotations(paths) if paths else None
        with timing.span("capability_lookup"):
            capability_records = (self._capability_annotations.lookup_batch(batch)
                                  if self._capability_annotations is not None else None)

        # --- 模型侧噪声无关上下文（VAE encode 与 und 的 VLM 冻结段在这里，每 batch 一次）---
        model = self.model.module if hasattr(self.model, 'module') else self.model
        with timing.span("prepare_training_context"):
            training_context = model.prepare_training_context(
                first_frame=first_frame,
                video_frames=video_frames,
                state=state,
                actions=actions,
                vlm_input_messages=vlm_input_messages,
                robot_names=robot_names,
                source_datasets=source_datasets,
                state_action_policies=state_action_policies,
                first_frame_pils=first_frame_pils,
                depth_first=depth_first,
                depth_frames=depth_frames,
                depth_present=depth_present,
                depth_valid_mask=depth_valid_mask,
                ego_head_pose=ego_head_pose,
                ego_head_valid=ego_head_valid,
                ego_K=ego_K,
                ego_pose_used=ego_pose_used,
                history_inputs=history_inputs,
                timing=model_timing,
                timing_sync_cuda=self.step_timing_sync_cuda,
            )

        if capability_records is not None:
            training_context["capability_records"] = capability_records

        # --- post：统一 delta-EEF 空间离群检查（检查对象 = loss 目标本身）---
        with timing.span("outlier_post_check"):
            (
                action_loss_weights,
                video_loss_weights,
                outlier_zeroed,
                outlier_details,
            ) = _zero_outlier_sample_loss_weights(
                actions=training_context["actions"],
                action_loss_weights=action_loss_weights,
                video_loss_weights=video_loss_weights,
                threshold=outlier_threshold,
            )
        # 合并 pre 段命中样本（并集）；details 保持 mask/nonfinite/
        # action_abs_max/state_abs_max 四键契约（state_abs_max 为 raw 信息性值）。
        if pre_outlier is not None and bool(pre_outlier["mask"].any().item()):
            pre_mask = pre_outlier["mask"]
            if outlier_details is None:
                outlier_details = {
                    "mask": pre_mask,
                    "nonfinite": pre_mask,
                    "action_abs_max": torch.zeros(
                        pre_mask.shape, dtype=torch.float32
                    ),
                    "state_abs_max": pre_outlier["state_abs_max"],
                }
            else:
                outlier_details["mask"] |= pre_mask
                outlier_details["nonfinite"] |= pre_mask
                outlier_details["state_abs_max"] = pre_outlier["state_abs_max"]
            outlier_zeroed = int(outlier_details["mask"].sum().item())
        elif outlier_details is not None:
            outlier_details["state_abs_max"] = (
                pre_outlier["state_abs_max"]
                if pre_outlier is not None
                else torch.zeros_like(outlier_details["action_abs_max"])
            )
        # 记录离群样本对应的 episode id（跨 rank 汇总，供 step_record/monitor）。
        with timing.span("outlier_rank_sync"):
            outlier_episodes = self._collect_outlier_episode_records(batch, outlier_details)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "language_embeddings": language_embeddings,
            "state": training_context["state"],
            "actions": training_context["actions"],
            "vlm_inputs": vlm_inputs,
            "first_frame_pils": first_frame_pils,
            "t5_input_messages": t5_input_messages,
            "vlm_input_messages": vlm_input_messages,
            "robot_names": robot_names,
            "source_datasets": source_datasets,
            "action_loss_weights": action_loss_weights,
            "video_loss_weights": video_loss_weights,
            "state_action_policies": state_action_policies,
            # 逐样本 video 帧索引（list[list[int]]）batch 透传：监控 meta /
            # 训练切片记录用（模型自 20260815 固定 5fps 轮起不再消费）
            "frame_indices": batch.get("frame_indices"),
            "outlier_zeroed": outlier_zeroed,
            "outlier_episodes": outlier_episodes,
            "training_context": training_context,
        }

    def train_step(
        self,
        batch: Dict[str, Any],
        timing: Any = None,
        heartbeat: Optional[Callable[[str, Optional[Dict[str, Any]], bool], None]] = None,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Single training step for UniDiffuser (one optimizer update).

        ``timing`` may be a :class:`TimingStack` (preferred) or a plain
        ``dict`` (legacy path).  When using ``TimingStack`` the training-loop
        sections are recorded as a proper containment tree.

        ``prepared`` 是 :meth:`prepare_train_batch` 的产物：train 循环在
        ``training.train_steps_per_vae_encode > 1`` 时每 batch 准备一次并跨
        N 个 train_step 复用（VAE encode 只做一次，每步各自重采样噪声）；
        缺省 None 时本步自行准备，行为与拆分前一致。
        """
        if timing is None:
            timing = TimingStack(enabled=False)
        # Normalise: wrap plain dicts so span() works
        _is_stack = isinstance(timing, TimingStack) if TimingStack is not None else False
        if not _is_stack:
            timing = TimingStack(enabled=False)  # legacy dict path: no tree

        if heartbeat is None:
            def heartbeat(_phase: str, _extra: Optional[Dict[str, Any]] = None, include_memory: bool = False) -> None:
                return None

        self.model.train()
        heartbeat("train_step_start", include_memory=True)

        # 非有限全量扫描（参数/梯度逐 tensor isfinite + 跨 rank allreduce）按
        # diagnostics.nonfinite_scan_interval 降频，并与 fail-fast 开关绑定
        # （评审 #12）；loss 标量检查不受影响，仍每步执行。
        run_nonfinite_scans = (
            _nonfinite_fail_fast_enabled()
            and int(self.global_step) % self.nonfinite_scan_interval == 0
        )

        with timing.span("train_step_total"):
            # --- zero_grad ---
            heartbeat("optimizer_zero_grad_start")
            with timing.span("optimizer_zero_grad"):
                self.optimizer.zero_grad()

            # --- batch 准备（host→device + 离群置零 + VAE encode 等噪声无关工作）---
            # prepared 由 train 循环在复用 latent 的多个更新步间共享；缺省本步
            # 自行准备（train_steps_per_vae_encode = 1，行为与拆分前一致）。
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model_timing: Any = {} if self.step_timing_enabled else None
            if prepared is None:
                prepared = self.prepare_train_batch(
                    batch, timing=timing, heartbeat=heartbeat, model_timing=model_timing,
                )

            # --- pre-forward nonfinite param check ---
            if run_nonfinite_scans:
                pre_forward_param_debug = _find_first_nonfinite_param(self.model)
                for key, value in pre_forward_param_debug.items():
                    timing[f"pre_forward_{key}"] = value
                heartbeat("pre_forward_nonfinite_param_allreduce_start", pre_forward_param_debug)
                global_pre_forward_nonfinite_param = _distributed_any(
                    bool(pre_forward_param_debug.get("has_nonfinite_param", False)),
                    self.device,
                )
                heartbeat("pre_forward_nonfinite_param_allreduce_done",
                          {"pre_forward_nonfinite_param_global": bool(global_pre_forward_nonfinite_param)})
                timing["pre_forward_nonfinite_param_global"] = bool(global_pre_forward_nonfinite_param)
                if global_pre_forward_nonfinite_param:
                    timing["grad_total_norm_before_clip"] = float("nan")
                    timing["grad_clip_norm"] = float(
                        self.config.training.grad_clip_norm
                        if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                    )
                    timing["optimizer_step_skipped"] = True
                    timing["optimizer_step_skip_reason"] = "pre_forward_nonfinite_param"
                    self.optimizer.zero_grad()
                    return {
                        "total_loss": float("nan"),
                        "video_loss": float("nan"),
                        "action_loss": float("nan"),
                        "nonfinite_param_local": bool(pre_forward_param_debug.get("has_nonfinite_param", False)),
                        "nonfinite_param_global": True,
                        "optimizer_step_skipped": True,
                    }

            # --- 离群样本信息（prepare 阶段按 batch 算好；复用 latent 的多个
            # 更新步会写入同样的记录，step_record 以 vae_encode_repeat_index 区分）---
            timing["outlier_loss_zeroed_samples"] = prepared["outlier_zeroed"]
            timing["outlier_episodes"] = prepared["outlier_episodes"]

            # --- forward ---
            heartbeat("forward_start")
            with timing.span("model_training_step"):
                loss_dict = model.training_step(
                    first_frame=prepared["first_frame"],
                    video_frames=prepared["video_frames"],
                    state=prepared["state"],
                    actions=prepared["actions"],
                    language_embeddings=prepared["language_embeddings"],
                    vlm_inputs=prepared["vlm_inputs"],
                    first_frame_pils=prepared["first_frame_pils"],
                    t5_input_messages=prepared["t5_input_messages"],
                    vlm_input_messages=prepared["vlm_input_messages"],
                    robot_names=prepared["robot_names"],
                    source_datasets=prepared["source_datasets"],
                    action_loss_weights=prepared["action_loss_weights"],
                    video_loss_weights=prepared["video_loss_weights"],
                    state_action_policies=prepared["state_action_policies"],
                    return_dict=True,
                    timing=model_timing,
                    timing_sync_cuda=self.step_timing_sync_cuda,
                    training_context=prepared["training_context"],
                    # 模态 dropout 按 global_step 定种（修复 #9b）：决策跨 rank/
                    # 弹性重启一致（本地计数器不进 checkpoint，曾致分叉死锁风险）。
                    dropout_seed_step=int(self.global_step),
                )
            heartbeat("forward_done",
                      {"model_training_step_seconds": timing.get("model_training_step_seconds", 0.0)},
                      include_memory=True)
            if model_timing:
                for key, value in model_timing.items():
                    timing[f"model_{key}"] = value

            total_loss = loss_dict['total_loss']

            # --- nonfinite loss check ---
            local_nonfinite_loss = not bool(torch.isfinite(total_loss.detach()).all().item())
            heartbeat("nonfinite_loss_allreduce_start", {"nonfinite_loss_local": bool(local_nonfinite_loss)})
            global_nonfinite_loss = _distributed_any(local_nonfinite_loss, self.device)
            heartbeat("nonfinite_loss_allreduce_done",
                      {"nonfinite_loss_global": bool(global_nonfinite_loss)})
            timing["nonfinite_loss_local"] = bool(local_nonfinite_loss)
            timing["nonfinite_loss_global"] = bool(global_nonfinite_loss)
            if global_nonfinite_loss:
                timing["grad_total_norm_before_clip"] = float("nan")
                timing["grad_clip_norm"] = float(
                    self.config.training.grad_clip_norm
                    if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                )
                timing["optimizer_step_skipped"] = True
                timing["optimizer_step_skip_reason"] = "nonfinite_loss"
                metrics: Dict[str, Any] = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                metrics["nonfinite_loss_local"] = bool(local_nonfinite_loss)
                metrics["nonfinite_loss_global"] = bool(global_nonfinite_loss)
                metrics["optimizer_step_skipped"] = True
                return metrics

            # --- debug skip backward ---
            skip_backward_debug = os.environ.get("MOTUS_SKIP_BACKWARD", "0").lower() in ("1", "true", "yes")
            if skip_backward_debug:
                timing["grad_total_norm_before_clip"] = 0.0
                timing["grad_clip_norm"] = float(
                    self.config.training.grad_clip_norm
                    if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                )
                timing["optimizer_step_skipped"] = True
                timing["optimizer_step_skip_reason"] = "debug_skip_backward"
                self.optimizer.zero_grad()
                metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                metrics["nonfinite_loss_local"] = False
                metrics["nonfinite_loss_global"] = False
                metrics["nonfinite_grad_local"] = False
                metrics["nonfinite_grad_global"] = False
                metrics["optimizer_step_skipped"] = True
                return metrics

            # --- backward ---
            heartbeat("backward_start", include_memory=True)
            with timing.span("backward"):
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    self.accelerator.backward(total_loss)
                else:
                    total_loss.backward()
            heartbeat("backward_done",
                      {"backward_seconds": timing.get("backward_seconds", 0.0)},
                      include_memory=True)

            # --- nonfinite grad check ---
            if run_nonfinite_scans:
                grad_debug = _find_first_nonfinite_grad(self.model)
                timing.update(grad_debug)
                heartbeat("nonfinite_grad_allreduce_start", grad_debug)
                global_nonfinite_grad = _distributed_any(
                    bool(grad_debug.get("has_nonfinite_grad", False)),
                    self.device,
                )
                heartbeat("nonfinite_grad_allreduce_done",
                          {"nonfinite_grad_global": bool(global_nonfinite_grad)})
                timing["nonfinite_grad_global"] = bool(global_nonfinite_grad)
                if global_nonfinite_grad:
                    grad_clip_norm = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                    timing["grad_total_norm_before_clip"] = float("nan")
                    timing["grad_clip_norm"] = float(grad_clip_norm)
                    timing["optimizer_step_skipped"] = True
                    timing["optimizer_step_skip_reason"] = "nonfinite_grad"
                    self.optimizer.zero_grad()
                    metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                    metrics["nonfinite_loss_local"] = False
                    metrics["nonfinite_loss_global"] = False
                    metrics["nonfinite_grad_local"] = bool(grad_debug.get("has_nonfinite_grad", False))
                    metrics["nonfinite_grad_global"] = True
                    metrics["optimizer_step_skipped"] = True
                    return metrics

            # --- pre-optimizer nonfinite param check ---
            if run_nonfinite_scans:
                pre_step_param_debug = _find_first_nonfinite_param(self.model)
                for key, value in pre_step_param_debug.items():
                    timing[f"pre_optimizer_{key}"] = value
                heartbeat("pre_optimizer_nonfinite_param_allreduce_start", pre_step_param_debug)
                global_pre_step_nonfinite_param = _distributed_any(
                    bool(pre_step_param_debug.get("has_nonfinite_param", False)),
                    self.device,
                )
                heartbeat("pre_optimizer_nonfinite_param_allreduce_done",
                          {"pre_optimizer_nonfinite_param_global": bool(global_pre_step_nonfinite_param)})
                timing["pre_optimizer_nonfinite_param_global"] = bool(global_pre_step_nonfinite_param)
                if global_pre_step_nonfinite_param:
                    grad_clip_norm = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                    timing["grad_total_norm_before_clip"] = float("nan")
                    timing["grad_clip_norm"] = float(grad_clip_norm)
                    timing["optimizer_step_skipped"] = True
                    timing["optimizer_step_skip_reason"] = "pre_optimizer_nonfinite_param"
                    self.optimizer.zero_grad()
                    metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                    metrics["nonfinite_param_local"] = bool(pre_step_param_debug.get("has_nonfinite_param", False))
                    metrics["nonfinite_param_global"] = True
                    metrics["optimizer_step_skipped"] = True
                    return metrics

            # --- grad clip ---
            grad_clip_norm = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
            heartbeat("grad_clip_start")
            ds_engine = self._deepspeed_engine()
            if ds_engine is not None:
                # ZeRO 下梯度不在 param.grad（修复 #2：此前 clip_grad_norm_
                # 全程空转、grad_total_norm_before_clip 恒 0）：实际裁剪由
                # engine 侧在 accelerator.backward() 内的 engine.step() 按
                # DeepSpeed config 的 gradient_clipping 执行（configs/zero1.json，
                # 取值须与 training.grad_clip_norm 一致）；此处只读取 engine
                # 已算好的全局范数做记录（optimizer.step() 包装器是 no-op，
                # 到本行时裁剪与更新均已发生）。
                try:
                    grad_total_norm = ds_engine.get_global_grad_norm()
                    grad_total_norm = (
                        float(grad_total_norm) if grad_total_norm is not None else float("nan")
                    )
                except Exception:
                    grad_total_norm = float("nan")
                timing["grad_total_norm_before_clip"] = grad_total_norm
            else:
                # 非 DeepSpeed 路径（本机 micro/单卡调试）：梯度在 param.grad，
                # 本地裁剪生效并记录真实范数。
                with timing.span("grad_clip"):
                    grad_total_norm_t = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip_norm
                    )
                try:
                    timing["grad_total_norm_before_clip"] = float(grad_total_norm_t.item())
                except Exception:
                    timing["grad_total_norm_before_clip"] = float(grad_total_norm_t)
            timing["grad_clip_norm"] = float(grad_clip_norm)
            heartbeat(
                "grad_clip_done",
                {
                    "grad_clip_seconds": timing.get("grad_clip_seconds", 0.0),
                    "grad_total_norm_before_clip": timing["grad_total_norm_before_clip"],
                },
            )

            # --- zero-LR check ---
            skip_zero_lr_optimizer = os.environ.get("MOTUS_SKIP_OPTIMIZER_WHEN_LR_ZERO", "1").lower() not in ("0", "false", "no")
            if skip_zero_lr_optimizer and _optimizer_all_lrs_zero(self.optimizer):
                timing["optimizer_step_skipped"] = True
                timing["optimizer_step_skip_reason"] = "zero_lr"
                self.optimizer.zero_grad()
                metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                metrics["nonfinite_loss_local"] = False
                metrics["nonfinite_loss_global"] = False
                metrics["nonfinite_grad_local"] = False
                metrics["nonfinite_grad_global"] = False
                metrics["optimizer_step_skipped"] = True
                return metrics

            # --- optimizer step ---
            heartbeat("optimizer_step_start", include_memory=True)
            with timing.span("optimizer_step"):
                self.optimizer.step()
            heartbeat("optimizer_step_done",
                      {"optimizer_step_seconds": timing.get("optimizer_step_seconds", 0.0)},
                      include_memory=True)

            # --- post-optimizer nonfinite param check ---
            if run_nonfinite_scans:
                post_step_param_debug = _find_first_nonfinite_param(self.model)
                for key, value in post_step_param_debug.items():
                    timing[f"post_optimizer_{key}"] = value
                heartbeat("post_optimizer_nonfinite_param_allreduce_start", post_step_param_debug)
                global_post_step_nonfinite_param = _distributed_any(
                    bool(post_step_param_debug.get("has_nonfinite_param", False)),
                    self.device,
                )
                heartbeat("post_optimizer_nonfinite_param_allreduce_done",
                          {"post_optimizer_nonfinite_param_global": bool(global_post_step_nonfinite_param)})
                timing["post_optimizer_nonfinite_param_global"] = bool(global_post_step_nonfinite_param)
                if global_post_step_nonfinite_param:
                    timing["optimizer_step_skipped"] = False
                    timing["optimizer_step_skip_reason"] = "post_optimizer_nonfinite_param"
                    self.optimizer.zero_grad()
                    metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                    metrics["nonfinite_param_local"] = bool(post_step_param_debug.get("has_nonfinite_param", False))
                    metrics["nonfinite_param_global"] = True
                    metrics["optimizer_step_skipped"] = False
                    return metrics

            # --- scheduler step ---
            with timing.span("scheduler_step"):
                if self.scheduler:
                    heartbeat("scheduler_step_start")
                    self.scheduler.step()
                    heartbeat("scheduler_step_done",
                              {"scheduler_step_seconds": timing.get("scheduler_step_seconds", 0.0)})

            timing["optimizer_step_skipped"] = False

            # --- assemble metrics ---
            metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
            metrics["nonfinite_loss_local"] = False
            metrics["nonfinite_loss_global"] = False
            metrics["nonfinite_grad_local"] = False
            metrics["nonfinite_grad_global"] = False
            metrics["optimizer_step_skipped"] = False

        heartbeat("train_step_done",
                  {"train_step_total_seconds": timing.get("train_step_total_seconds", 0.0)},
                  include_memory=True)
        return metrics

    def _maybe_write_action_diagnostics(self, step_record: Dict[str, Any], metrics: Dict[str, Any]):
        if not self.action_diagnostics_enabled:
            return
        step = int(step_record.get("step", 0) or 0)
        if step <= 0:
            return

        action_loss = float(metrics.get("action_loss", 0.0) or 0.0)
        action_loss_max = float(metrics.get("action_loss_per_sample_max", 0.0) or 0.0)
        action_abs_max = float(step_record.get("batch_action_abs_max", 0.0) or 0.0)
        state_abs_max = float(step_record.get("batch_state_abs_max", 0.0) or 0.0)
        grad_norm = float(step_record.get("grad_total_norm_before_clip", 0.0) or 0.0)
        total_loss = float(metrics.get("total_loss", 0.0) or 0.0)
        video_loss = float(metrics.get("video_loss", 0.0) or 0.0)

        reasons = []
        if not all(torch.isfinite(torch.tensor(v)) for v in (total_loss, video_loss, action_loss)):
            reasons.append("nonfinite_loss")
        if bool(step_record.get("optimizer_step_skipped", False)):
            reasons.append(str(step_record.get("optimizer_step_skip_reason", "optimizer_step_skipped")))
        if bool(step_record.get("has_nonfinite_grad", False)) or bool(step_record.get("nonfinite_grad_global", False)):
            reasons.append("nonfinite_grad")
        if bool(step_record.get("pre_forward_has_nonfinite_param", False)) or bool(step_record.get("pre_forward_nonfinite_param_global", False)):
            reasons.append("pre_forward_nonfinite_param")
        if bool(step_record.get("pre_optimizer_has_nonfinite_param", False)) or bool(step_record.get("pre_optimizer_nonfinite_param_global", False)):
            reasons.append("pre_optimizer_nonfinite_param")
        if bool(step_record.get("post_optimizer_has_nonfinite_param", False)) or bool(step_record.get("post_optimizer_nonfinite_param_global", False)):
            reasons.append("post_optimizer_nonfinite_param")
        if action_loss >= self.action_loss_anomaly_threshold:
            reasons.append("action_loss")
        if action_loss_max >= self.action_loss_anomaly_threshold:
            reasons.append("action_loss_per_sample")
        if max(action_abs_max, state_abs_max) >= self.action_tensor_abs_anomaly_threshold:
            reasons.append("batch_action_state_abs")
        if grad_norm >= self.grad_norm_anomaly_threshold:
            reasons.append("grad_norm")

        periodic = self.action_diagnostics_interval > 0 and step % self.action_diagnostics_interval == 0
        if not reasons and not periodic:
            return

        lrs = [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]
        record = {
            "step": step,
            "step_before": step_record.get("step_before"),
            "reason": reasons or ["periodic"],
            "source_dataset": step_record.get("source_dataset"),
            "source_seen": step_record.get("source_seen"),
            "batch_size": step_record.get("batch_size"),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            "learning_rates": lrs,
            "batch_action_abs_max": action_abs_max,
            "batch_action_abs_mean": step_record.get("batch_action_abs_mean"),
            "batch_action_mean": step_record.get("batch_action_mean"),
            "batch_action_std": step_record.get("batch_action_std"),
            "batch_action_sample_abs_max": step_record.get("batch_action_sample_abs_max"),
            "batch_action_abs_max_manifest_id": step_record.get("batch_action_abs_max_manifest_id"),
            "batch_action_abs_max_robot_name": step_record.get("batch_action_abs_max_robot_name"),
            "batch_action_abs_max_manifest_task": step_record.get("batch_action_abs_max_manifest_task"),
            "batch_state_abs_max": state_abs_max,
            "batch_state_abs_mean": step_record.get("batch_state_abs_mean"),
            "batch_state_mean": step_record.get("batch_state_mean"),
            "batch_state_std": step_record.get("batch_state_std"),
            "action_loss_max_manifest_id": step_record.get("action_loss_max_manifest_id"),
            "action_loss_max_robot_name": step_record.get("action_loss_max_robot_name"),
            "action_loss_max_manifest_task": step_record.get("action_loss_max_manifest_task"),
            "robot_names": step_record.get("robot_names"),
            "manifest_ids": step_record.get("manifest_ids"),
            "manifest_tasks": step_record.get("manifest_tasks"),
            "grad_total_norm_before_clip": grad_norm,
            "grad_clip_norm": step_record.get("grad_clip_norm"),
            "ready_queue_enabled": step_record.get("ready_queue_enabled"),
            "ready_queue_get_wait_seconds": step_record.get("ready_queue_get_wait_seconds"),
            "ready_queue_size_before_get": step_record.get("ready_queue_size_before_get"),
            "ready_queue_size_after_get": step_record.get("ready_queue_size_after_get"),
            "ready_queue_was_empty": step_record.get("ready_queue_was_empty"),
            "dataloader_next_seconds": step_record.get("dataloader_next_seconds"),
            "step_time_seconds": step_record.get("step_time_seconds"),
        }
        for key, value in step_record.items():
            if (
                key.startswith("model_")
                and (
                    key.endswith("_all_finite")
                    or key.endswith("_nan_count")
                    or key.endswith("_inf_count")
                    or key.endswith("_finite_abs_max")
                    or key.endswith("_finite_mean")
                    or key.endswith("_finite_std")
                    or key.endswith("_numel")
                )
            ):
                record[key] = value
        for key in (
            "nonfinite_loss_local",
            "nonfinite_loss_global",
            "has_nonfinite_grad",
            "nonfinite_grad_global",
            "first_nonfinite_grad_param",
            "first_nonfinite_grad_shape",
            "first_nonfinite_grad_nan_count",
            "first_nonfinite_grad_inf_count",
            "grad_tensors_checked",
            "grad_tensors_missing",
            "pre_forward_has_nonfinite_param",
            "pre_forward_nonfinite_param_global",
            "pre_forward_first_nonfinite_param",
            "pre_forward_first_nonfinite_param_shape",
            "pre_forward_first_nonfinite_param_nan_count",
            "pre_forward_first_nonfinite_param_inf_count",
            "pre_forward_param_tensors_checked",
            "pre_optimizer_has_nonfinite_param",
            "pre_optimizer_nonfinite_param_global",
            "pre_optimizer_first_nonfinite_param",
            "pre_optimizer_first_nonfinite_param_shape",
            "pre_optimizer_first_nonfinite_param_nan_count",
            "pre_optimizer_first_nonfinite_param_inf_count",
            "pre_optimizer_param_tensors_checked",
            "post_optimizer_has_nonfinite_param",
            "post_optimizer_nonfinite_param_global",
            "post_optimizer_first_nonfinite_param",
            "post_optimizer_first_nonfinite_param_shape",
            "post_optimizer_first_nonfinite_param_nan_count",
            "post_optimizer_first_nonfinite_param_inf_count",
            "post_optimizer_param_tensors_checked",
            "optimizer_step_skipped",
            "optimizer_step_skip_reason",
        ):
            if key in step_record:
                record[key] = step_record.get(key)
        self.action_diagnostics_writer.write(record)
    
    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500, reset_scheduler: Optional[bool] = None):
        """
        Main training loop.
        
        Args:
            max_steps: Maximum number of training steps
            resume_from: Path to checkpoint to resume from
            val_interval: Validation interval in steps
            reset_scheduler: If True, reset scheduler to new config. If None, use
                config.resume.reset_scheduler (default False: 弹性续训接着原
                LR 计划走、不重新 warmup，见评审 #13；只有想从旧 ckpt 热启动
                新 LR 计划时才显式置 True)
        """
        # Load checkpoint if specified
        if resume_from:
            # Determine whether to reset scheduler
            if reset_scheduler is None:
                # Use config value if available, otherwise default to False
                # (elastic resume continues the LR schedule instead of
                # re-running warmup on every restart — 评审 #13)
                if self.config is not None and hasattr(self.config, 'resume') and hasattr(self.config.resume, 'reset_scheduler'):
                    reset_scheduler = bool(self.config.resume.reset_scheduler)
                else:
                    reset_scheduler = False  # Default behavior
            if not reset_scheduler:
                logger.info(
                    "Resuming with reset_scheduler=False: scheduler continues "
                    "the original LR schedule (step_count synced to global_step)."
                )
            
            resume_start = time.perf_counter()
            self.load_checkpoint(resume_from, reset_scheduler=reset_scheduler)
            add_timing_event(
                self.startup_timing_events,
                "checkpoint.resume_full_training_state",
                resume_start,
                path=str(resume_from),
                reset_scheduler=bool(reset_scheduler),
            )

        if self.startup_start_time is not None:
            add_timing_event(self.startup_timing_events, "startup.total_before_train_loop", self.startup_start_time)
        write_startup_timing(self.config, self.startup_timing_events, self.rank)
        
        logger.info(f"Starting UniDiffuser training for {max_steps} steps")
        _monitor_call(self.monitor, "set_max_steps", max_steps)
        _monitor_call(self.monitor, "set_state", "running")
        _monitor_call(self.monitor, "log_event", "train_loop_start", max_steps=max_steps)
        
        start_time = time.time()
        
        # Step-based training loop
        # train_dataloader 是 data_prepare 的 IndexedMixedTrainingDataPipeline：
        # ready queue 内建（queue_enabled 由 system.ready_batch_queue_enabled 决定），
        # pipeline.next() -> (batch, stats)，stats 键与旧 ReadyBatchQueue 一致（另含 mix_*）。
        data_pipeline = self.train_dataloader
        # 续训不恢复数据游标：pipeline 会从排列头部重走，于是最后一段训练只是
        # 把开头见过的窗口重放一遍（V11_A 7000→8000 实测 step 7001 与 step 1
        # 同一条 episode、同一 mix_step）。按 global_step 推进 epoch，让续训段
        # 拿到一份没训过的排列。set_epoch 必须在 queue 起来之前调。
        start_epoch = 0
        if resume_from:
            start_epoch = int(self.global_step)
            data_pipeline.set_epoch(start_epoch)
            logger.info(
                "Resumed at step %d: data epoch advanced to %d "
                "(stream is not replayed from the start).",
                int(self.global_step),
                start_epoch,
            )
        logger.info(
            "Indexed data pipeline: queue_enabled=%s snapshot=%s",
            data_pipeline.queue_enabled,
            data_pipeline.snapshot(),
        )
        epoch = start_epoch

        # 每个 batch VAE encode 一次后、按不同噪声独立参数更新的步数
        # （training.train_steps_per_vae_encode，默认 1 = 不复用）。>1 时取数与
        # encode 摊薄到 N 个 optimizer step 上；global_step / LR schedule /
        # checkpoint / monitor 仍按 optimizer step 计，数据消耗相应慢 N 倍。
        train_steps_per_vae_encode = 1
        if self.config is not None:
            train_steps_per_vae_encode = int(
                getattr(self.config.training, "train_steps_per_vae_encode", 1) or 1
            )
        train_steps_per_vae_encode = max(1, train_steps_per_vae_encode)
        logger.info(
            "train_steps_per_vae_encode=%d（每 batch VAE encode 一次后按不同噪声独立训练的步数）",
            train_steps_per_vae_encode,
        )
        repeats_left = 0
        batch_cache: Optional[Dict[str, Any]] = None
        stop_control = TrainingStopRequest(self.checkpoint_dir)
        stopped_for_request = False

        while self.global_step < max_steps:
            step_record = {
                "step_before": int(self.global_step),
                "epoch": int(epoch),
                "world_size": int(self.world_size),
                "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            }
            loop_start = time.perf_counter()
            include_memory = (
                self.heartbeat_memory_interval > 0
                and int(self.global_step) % self.heartbeat_memory_interval == 0
            )
            self._write_heartbeat(
                "loop_start",
                step_record,
                include_memory=include_memory,
                ready_queue=data_pipeline,
            )

            # Training step（提前创建 timing stack：取 batch 分支里
            # prepare_train_batch 的 host_to_device / VAE encode span 也记进来）
            timing_enabled = (
                self.step_timing_enabled
                and self.step_timing_interval > 0
                and (int(self.global_step) % self.step_timing_interval == 0)
            )
            # CUDA event 计时模式：零逐段同步，可每步采集真实 GPU 时长
            timing_cuda_events = str(os.environ.get("MOTUS_TIMING_CUDA_EVENTS", "0")).strip().lower() in ("1", "true")
            _timing_kwargs: Dict[str, Any] = {
                "sync_cuda": self.step_timing_sync_cuda,
                "enabled": timing_enabled or timing_cuda_events,
            }
            if timing_cuda_events:
                _timing_kwargs["use_cuda_events"] = True
            train_step_timing = TimingStack(**_timing_kwargs) if TimingStack is not None else {}

            # --- 取 batch + 噪声无关准备（含 VAE encode）：每个 batch 只做一次，
            # 产物缓存供 train_steps_per_vae_encode 个 optimizer step 复用 ---
            if repeats_left == 0:
                dataloader_start = time.perf_counter()
                self._write_heartbeat(
                    "before_ready_queue_get",
                    step_record,
                    include_memory=include_memory,
                    ready_queue=data_pipeline,
                )
                batch, queue_stats = data_pipeline.next()
                self._write_heartbeat(
                    "after_ready_queue_get",
                    step_record,
                    include_memory=include_memory,
                    ready_queue=data_pipeline,
                )
                if batch is None:  # Handle None batches
                    step_record.update(queue_stats)
                    step_record["batch_is_none"] = True
                    step_record["loop_total_seconds"] = time.perf_counter() - loop_start
                    self.step_timing_writer.write(step_record)
                    del batch, step_record
                    continue
                epoch_for_record = epoch
                epoch = int(queue_stats.get(
                    "ready_queue_epoch",
                    queue_stats.get("mix_epoch", queue_stats.get("data_epoch", epoch)),
                ))
                prepare_model_timing: Any = {} if timing_enabled else None
                prepare_start = time.perf_counter()
                prepared = self.prepare_train_batch(
                    batch,
                    timing=train_step_timing,
                    heartbeat=None,
                    model_timing=prepare_model_timing,
                    source_name=queue_stats.get("mix_source_name"),
                )
                prepare_seconds = time.perf_counter() - prepare_start

                # batch 元数据（N 个更新步共用，逐 step 从缓存写回 step_record）
                meta: Dict[str, Any] = {}
                source_datasets = batch.get("source_datasets")
                batch_source_dataset = None
                if source_datasets:
                    batch_source_dataset = str(source_datasets[0])
                    if any(str(source_dataset) != batch_source_dataset for source_dataset in source_datasets):
                        raise ValueError(f"Mixed source datasets inside one training batch: {source_datasets}")
                    self.source_dataset_counts[batch_source_dataset] += 1
                meta["source_dataset"] = batch_source_dataset or "unknown"
                meta["batch_size"] = len(source_datasets) if source_datasets else None
                meta["source_seen"] = dict(self.source_dataset_counts)
                try:
                    meta["first_frame_shape"] = list(batch["first_frame"].shape)
                    meta["video_frames_shape"] = list(batch["video_frames"].shape)
                    meta["action_sequence_shape"] = list(batch["action_sequence"].shape)
                    if "initial_state" in batch and batch["initial_state"] is not None:
                        meta["initial_state_shape"] = list(batch["initial_state"].shape)
                    meta.update(_tensor_debug_stats(batch.get("action_sequence"), "batch_action"))
                    meta.update(_tensor_debug_stats(batch.get("initial_state"), "batch_state"))
                except Exception:
                    pass
                robot_names = batch.get("robot_names")
                manifest_ids = batch.get("manifest_ids")
                manifest_tasks = batch.get("manifest_tasks")
                meta["robot_names"] = _list_head(robot_names)
                meta["manifest_ids"] = _list_head(manifest_ids)
                meta["manifest_tasks"] = _list_head(manifest_tasks)
                # 训练切片元数据：monitor 训练集 tab 按 (episode_id, 窗口起点, ratio)
                # 重建 selection 并远端解码可视化（batch 内逐样本，全部保留不截断）。
                episode_ids = batch.get("episode_ids")
                frame_indices = batch.get("frame_indices")
                action_ratios = batch.get("action_video_fps_ratios")
                meta["episode_ids"] = [str(v) for v in episode_ids] if episode_ids else []
                meta["frame_indices"] = (
                    [[int(f) for f in frames] for frames in frame_indices] if frame_indices else []
                )
                meta["action_video_fps_ratios"] = (
                    [int(r) for r in action_ratios] if action_ratios else []
                )
                try:
                    action_top_idx = int(meta.get("batch_action_sample_abs_max_argmax", -1))
                    if manifest_ids is not None and action_top_idx >= 0:
                        meta["batch_action_abs_max_manifest_id"] = str(manifest_ids[action_top_idx])
                    if robot_names is not None and action_top_idx >= 0:
                        meta["batch_action_abs_max_robot_name"] = str(robot_names[action_top_idx])
                    if manifest_tasks is not None and action_top_idx >= 0:
                        meta["batch_action_abs_max_manifest_task"] = str(manifest_tasks[action_top_idx])
                except Exception:
                    pass
                batch_cache = {
                    "batch": batch,
                    "queue_stats": queue_stats,
                    "prepared": prepared,
                    "prepare_model_timing": prepare_model_timing,
                    "prepare_seconds": prepare_seconds,
                    "dataloader_next_seconds": time.perf_counter() - dataloader_start,
                    "dataloader_restarted": bool(queue_stats.get("ready_queue_producer_restart_count", 0) > 0),
                    "epoch_for_record": epoch_for_record,
                    "meta": meta,
                    "batch_source_dataset": batch_source_dataset,
                    "robot_names": robot_names,
                    "manifest_ids": manifest_ids,
                    "manifest_tasks": manifest_tasks,
                }
                repeats_left = train_steps_per_vae_encode

            batch = batch_cache["batch"]
            prepared = batch_cache["prepared"]
            batch_source_dataset = batch_cache["batch_source_dataset"]
            robot_names = batch_cache["robot_names"]
            manifest_ids = batch_cache["manifest_ids"]
            manifest_tasks = batch_cache["manifest_tasks"]
            vae_encode_repeat_index = train_steps_per_vae_encode - repeats_left
            step_record["epoch"] = batch_cache["epoch_for_record"]
            step_record.update(batch_cache["queue_stats"])
            step_record.update(batch_cache["meta"])
            step_record["train_steps_per_vae_encode"] = int(train_steps_per_vae_encode)
            step_record["vae_encode_repeat_index"] = int(vae_encode_repeat_index)
            # 取数/准备（含 VAE encode）只发生在每个 batch 的首个更新步
            step_record["prepare_train_batch_seconds"] = (
                float(batch_cache["prepare_seconds"]) if vae_encode_repeat_index == 0 else 0.0
            )
            step_record["dataloader_iter_restart_seconds"] = 0.0
            step_record["dataloader_next_seconds"] = (
                batch_cache["dataloader_next_seconds"] if vae_encode_repeat_index == 0 else 0.0
            )
            step_record["dataloader_restarted"] = batch_cache["dataloader_restarted"]
            if include_memory:
                step_record.update(_memory_snapshot("after_batch_get_memory"))

            self._maybe_simulate_elastic_failure()

            step_start_time = time.perf_counter()

            heartbeat_context = {
                "step_before": int(self.global_step),
                "source_dataset": step_record.get("source_dataset"),
                "batch_size": step_record.get("batch_size"),
                "robot_names": step_record.get("robot_names"),
                "manifest_ids": step_record.get("manifest_ids"),
                "manifest_tasks": step_record.get("manifest_tasks"),
            }

            def train_step_heartbeat(
                phase: str,
                extra: Optional[Dict[str, Any]] = None,
                include_memory: bool = False,
            ) -> None:
                record = dict(heartbeat_context)
                if extra:
                    record.update(extra)
                self._write_heartbeat(
                    phase,
                    record,
                    include_memory=include_memory,
                    ready_queue=data_pipeline,
                )

            self._write_heartbeat(
                "before_train_step",
                heartbeat_context,
                include_memory=include_memory,
                ready_queue=data_pipeline,
            )
            metrics = self.train_step(batch, timing=train_step_timing, heartbeat=train_step_heartbeat, prepared=prepared)
            if timing_cuda_events:
                _resolve_pending = getattr(train_step_timing, "resolve_pending", None)
                if callable(_resolve_pending):
                    _resolve_seconds = float(_resolve_pending(sync=True))
                    _record_overhead = float(getattr(train_step_timing, "_event_record_overhead", 0.0))
                    train_step_timing._event_record_overhead = 0.0
                    step_record["timing_resolve_seconds"] = _resolve_seconds
                    # 计时机制自身开销 = 解析耗时 + 事件创建/record 的 CPU 耗时
                    step_record["timing_overhead_seconds"] = _resolve_seconds + _record_overhead
            if getattr(self, "_history_attacher", None) is not None:
                metrics.update(self._history_attacher.last_metrics)
                step_record.update(self._history_attacher.last_metrics)
            if TimingStack is not None and isinstance(train_step_timing, TimingStack):
                step_record.update(train_step_timing.to_flat_dict())
            else:
                step_record.update(train_step_timing)
            # prepare 阶段（含 VAE encode）的模型侧计时只并入该 batch 的首个更新步
            if vae_encode_repeat_index == 0 and batch_cache["prepare_model_timing"]:
                for _pk, _pv in batch_cache["prepare_model_timing"].items():
                    if isinstance(_pv, (int, float)):
                        step_record[f"model_{_pk}"] = _pv

            # Preserve rank-local losses BEFORE all_reduce: each rank's grouped
            # batch belongs to its own task, so global means cannot be attributed
            # to the episode/task appearing on rank 0.
            local_task_metrics = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float)) and (
                    k in ("total_loss", "video_loss", "action_loss", "depth_loss")
                    or k.startswith("modality_dropped_")
                    or k.startswith("temporal_")
                )
            }
            local_optimizer_skipped = bool(metrics.get("optimizer_step_skipped", False))
            # 全局平均 loss：所有 node/GPU 的均值（替代 rank0 单卡值）。
            # 所有 rank 都执行到此处，all_reduce 不会错位。
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                _loss_keys = ("total_loss", "video_loss", "action_loss", "depth_loss")
                # Keep the collective shape fixed even when depth is disabled.
                _means = _distributed_mean(
                    [metrics.get(k, 0.0 if k == "depth_loss" else float("nan"))
                     for k in _loss_keys], self.device
                )
                for _k, _v in zip(_loss_keys, _means):
                    if _k != "depth_loss" or _k in metrics:
                        metrics[_k] = _v

            # Attribute losses to the sampled task; logging only, no extra forward.
            # Sampling is global-step deterministic, so every rank has this mode.
            if metrics.get("temporal_cut", 0):
                _mode = ("joint", "idm", "fdm")[int(metrics.get("temporal_mode_id", 0))]
                for _key in ("total_loss", "video_loss", "action_loss"):
                    metrics[f"mode_{_mode}/{_key}"] = float(metrics[_key])
                if "depth_loss" in metrics:
                    metrics[f"mode_{_mode}/depth_loss"] = float(metrics["depth_loss"])
                    # Preserve the existing explicitly local diagnostic series.
                    metrics[f"mode_{_mode}/depth_loss_local"] = local_task_metrics["depth_loss"]

            step_time = time.perf_counter() - step_start_time
            self.global_step += 1
            repeats_left -= 1
            step_record["step"] = int(self.global_step)
            step_record["step_time_seconds"] = float(step_time)
            step_record["metrics"] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            try:
                loss_top_idx = int(metrics.get("action_loss_per_sample_argmax", -1))
                if manifest_ids is not None and loss_top_idx >= 0:
                    step_record["action_loss_max_manifest_id"] = str(manifest_ids[loss_top_idx])
                if robot_names is not None and loss_top_idx >= 0:
                    step_record["action_loss_max_robot_name"] = str(robot_names[loss_top_idx])
                if manifest_tasks is not None and loss_top_idx >= 0:
                    step_record["action_loss_max_manifest_task"] = str(manifest_tasks[loss_top_idx])
            except Exception:
                pass
            # 离群样本（|action/state| 超阈值或 NaN/Inf）的 episode id：
            # 记入日志 + monitor 事件流（step_record 已带 outlier_episodes 字段）。
            outlier_episodes = step_record.get("outlier_episodes") or []
            if outlier_episodes:
                logger.warning(
                    "Step %d: %d outlier sample(s) loss-zeroed, episodes: %s",
                    int(self.global_step),
                    len(outlier_episodes),
                    [r.get("episode_id") for r in outlier_episodes],
                )
                _monitor_call(
                    self.monitor,
                    "log_event",
                    "outlier_episodes",
                    step=int(self.global_step),
                    count=len(outlier_episodes),
                    episodes=outlier_episodes,
                )
            self._maybe_write_action_diagnostics(step_record, metrics)
            
            nonfinite_skip_reasons = {
                "pre_forward_nonfinite_param",
                "nonfinite_loss",
                "nonfinite_grad",
                "pre_optimizer_nonfinite_param",
                "post_optimizer_nonfinite_param",
            }
            skip_reason = str(step_record.get("optimizer_step_skip_reason", ""))
            should_fail_fast = (
                _nonfinite_fail_fast_enabled()
                and (
                    skip_reason in nonfinite_skip_reasons
                    or bool(step_record.get("nonfinite_loss_global", False))
                    or bool(step_record.get("nonfinite_grad_global", False))
                    or bool(step_record.get("pre_forward_nonfinite_param_global", False))
                    or bool(step_record.get("pre_optimizer_nonfinite_param_global", False))
                    or bool(step_record.get("post_optimizer_nonfinite_param_global", False))
                )
            )
            if should_fail_fast:
                step_record["validation_seconds"] = 0.0
                step_record["checkpoint_total_seconds"] = 0.0
                step_record["loop_total_seconds"] = time.perf_counter() - loop_start
                self.step_timing_writer.write(step_record)
                self.action_diagnostics_writer.close()
                self.heartbeat_writer.close()
                logger.info("Data pipeline snapshot: %s", data_pipeline.snapshot())
                data_pipeline.close()
                logger.error("Fail-fast on non-finite training state at step %s: %s", self.global_step, skip_reason)
                _raise_nonfinite(skip_reason or "nonfinite", step_record)

            # Checkpoint first. Validation of this same step can fail without
            # erasing the weights needed to resume. Include the final save in
            # this step's wall time even when it is not an interval boundary.
            if self.global_step % self.save_interval == 0:
                checkpoint_timing = self.save_checkpoint()
                if checkpoint_timing:
                    step_record.update(checkpoint_timing)
            elif self.global_step == max_steps:
                checkpoint_timing = self.save_checkpoint()
                if checkpoint_timing:
                    step_record.update(checkpoint_timing)
            else:
                step_record["checkpoint_total_seconds"] = 0.0

            # Full holdout: either exact unpadded distributed windows or the
            # legacy rank-0 evaluator. Both preserve epoch 0 and report errors
            # collectively before the next training step.
            if self.val_dataloader is not None and val_interval > 0 and self.global_step % val_interval == 0:
                validation_start = time.perf_counter()
                val_error = None
                saved_val = []
                distributed_val = bool(getattr(self.config.system, "val_distributed", False))
                if distributed_val:
                    from thinker_wm.train.distributed_validation import evaluate_distributed
                    try:
                        val_metrics, saved_val = evaluate_distributed(
                            self.model, self.val_dataloader, self.accelerator,
                            self.config, getattr(self, "_history_attacher", None))
                    except Exception as exc:
                        # The evaluator shares each rank's failure before raising.
                        val_error = f"{type(exc).__name__}: {exc}"
                if self.rank == 0:
                    try:
                        if val_error:
                            raise RuntimeError(val_error)
                        if not distributed_val:
                            if hasattr(self.val_dataloader, "set_epoch"):
                                self.val_dataloader.set_epoch(0)
                            val_batches = int(getattr(self.config.system, "val_num_batches", 128) or 128)
                            val_metrics, saved_val = evaluate_model(
                                self.model, _pipeline_batch_iter(
                                    self.val_dataloader, getattr(self, "_history_attacher", None)),
                                self.accelerator, self.config,
                                num_eval_batches=val_batches,
                            )
                        logger.info(f"Validation - Step {self.global_step}")
                        log_evaluation_metrics(val_metrics, self.tb_writer, self.accelerator, self.global_step)
                        record_path = _write_validation_record(
                            self.config, self.global_step, ok=True, metrics=val_metrics)
                        logger.info("Validation record: %s", record_path)

                        pt_folder = os.path.join(self.config.system.checkpoint_dir, "saved_val")
                        os.makedirs(pt_folder, exist_ok=True)
                        torch.save(
                            saved_val,
                            os.path.join(pt_folder, f"{self.global_step}.pt")
                        )
                        try:
                            messages = []
                            for item in saved_val:
                                messages.append({
                                    "t5_input_messages": item.get("t5_input_messages"),
                                    "vlm_input_messages": item.get("vlm_input_messages"),
                                    "original_vlm_input_messages": item.get("original_vlm_input_messages"),
                                    "task": item.get("task"),
                                    "episode_id": item.get("episode_id"),
                                    "window_id": item.get("window_id"),
                                })
                            json_path = os.path.join(pt_folder, f"{self.global_step}_messages.json")
                            with open(json_path, 'w') as f:
                                json.dump(messages, f, indent=2)
                            logger.info(f"Saved input messages to {json_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save input messages: {e}")
                        monitor_dir = getattr(self.monitor, "run_dir", None)
                        if monitor_dir is not None and (bool(getattr(self.config.system, "val_a2v", False))
                                                        and bool(getattr(self.config.system, "val_depth", False))):
                            from thinker_wm.periodic_eval.validation_artifacts import publish_validation_artifacts
                            publish_validation_artifacts(
                                self.config, self.global_step, val_metrics, saved_val, monitor_dir,
                                seconds=time.perf_counter() - validation_start,
                                world_size=int(self.accelerator.num_processes),
                                checkpoint=os.path.join(self.config.system.checkpoint_dir,
                                                        f"checkpoint_step_{self.global_step}"))

                    except Exception as exc:
                        val_error = f"{type(exc).__name__}: {exc}"
                        logger.exception("Validation failed at step %s", self.global_step)
                        try:
                            record_path = _write_validation_record(
                                self.config, self.global_step, ok=False, error=val_error)
                            logger.error("Validation failure record: %s", record_path)
                        except Exception:
                            logger.exception("Failed to write the validation failure record")
                    finally:
                        if distributed_val:
                            from thinker_wm.periodic_eval.validation_artifacts import discard_staged_media
                            discard_staged_media(saved_val)
                val_error = share_validation_error(val_error)
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier(device_ids=[torch.cuda.current_device()])
                    except TypeError:
                        dist.barrier()
                step_record["validation_error"] = val_error or ""
                if val_error:
                    step_record["validation_seconds"] = time.perf_counter() - validation_start
                    step_record["loop_total_seconds"] = time.perf_counter() - loop_start
                    failed_timing = step_timing_breakdown(step_record)
                    step_record.update({f"wall_{key}": value for key, value in failed_timing.items()})
                    self.step_timing_writer.write(step_record)
                    if self.rank == 0:
                        logger.error("Step %d failed validation; %s", self.global_step,
                                     format_step_timing(failed_timing))
                    raise RuntimeError(
                        f"validation failed at step {self.global_step}: {val_error}"
                    )
                step_record["validation_seconds"] = time.perf_counter() - validation_start
            else:
                step_record["validation_seconds"] = 0.0

            # Periodic eval（periodic_eval 包）：每 interval_steps 步、每个 rank
            # 各评一个固定 batch（素材启动时后台预解码，前台零解码；全流程
            # try/except 不影响训练）；产物写 monitor run_dir 的 eval 流，
            # Web UI「eval 评估」tab 展示。所有 rank 都调用（内部有集合通信）。
            periodic_eval_start = time.perf_counter()
            periodic_eval_scalars = self.periodic_evaluator.maybe_run(
                self.model, int(self.global_step)
            )
            step_record["periodic_eval_seconds"] = time.perf_counter() - periodic_eval_start
            if periodic_eval_scalars:
                step_record.update(periodic_eval_scalars)

            # All ranks reach this boundary after update/evaluation/checkpoint.
            # Do not send SIGTERM to torchrun: it terminates workers before a
            # collective checkpoint can complete. Use request_training_stop.py.
            stop_requested = stop_control.requested(self.device)
            if stop_requested:
                if getattr(self, "_last_checkpoint_step", None) != self.global_step:
                    checkpoint_timing = self.save_checkpoint()
                    if checkpoint_timing:
                        step_record.update(checkpoint_timing)
                stop_control.complete(
                    self.global_step, self.checkpoint_dir / f"checkpoint_step_{self.global_step}")
                stopped_for_request = True
                step_record["stop_requested"] = True
                _monitor_call(self.monitor, "log_event", "graceful_stop_saved", step=self.global_step)
                logger.info("Save-and-stop complete at step %d; all ranks will exit.", self.global_step)

            if self.train_loop_gc_interval > 0 and self.global_step % self.train_loop_gc_interval == 0:
                gc_start = time.perf_counter()
                collected = gc.collect()
                self._write_heartbeat(
                    "gc_collect",
                    {
                        "step": int(self.global_step),
                        "gc_collected": int(collected),
                        "gc_seconds": time.perf_counter() - gc_start,
                    },
                    include_memory=True,
                    ready_queue=data_pipeline,
                )
            if include_memory:
                step_record.update(_memory_snapshot("step_end_memory"))
            step_record["loop_total_seconds"] = time.perf_counter() - loop_start
            wall_timing = step_timing_breakdown(step_record)
            step_record.update({f"wall_{key}": value for key, value in wall_timing.items()})
            # Log only after evaluation/save/GC so Total covers the whole step.
            if self.rank == 0 and (self.global_step % self.log_interval == 0
                                   or stop_requested or self.global_step == max_steps
                                   or wall_timing["checkpoint_seconds"] > 0
                                   or wall_timing["validation_seconds"] > 0):
                # Log optimizer LR as authoritative (may differ from inner scheduler get_last_lr)
                groups = list(self.optimizer.param_groups)
                lrs = [g['lr'] for g in groups]
                lr_main = lrs[0] if len(lrs) > 0 else 0.0
                # depth I/O 组（名字标记，见 create_model_and_optimizer）单列，
                # 否则它会被当成 wan 组显示成误导性的 "wan lr"。
                lr_depth = next(
                    (float(g['lr']) for g in groups if g.get('name') == 'depth'), None
                )
                lr_clone = next(
                    (float(g['lr']) for g in groups if g.get('name') == 'depth_clone'), None
                )
                lr_wan = lrs[1] if (len(lrs) > 1 and lr_depth is None) else lr_main
                step_record["lr_main"] = float(lr_main)
                step_record["lr_wan"] = float(lr_wan)
                if lr_depth is not None:
                    step_record["lr_depth"] = float(lr_depth)
                if lr_clone is not None:
                    step_record["lr_depth_clone"] = float(lr_clone)

                log_str = (
                    f"Step {self.global_step}/{max_steps}, "
                    f"Loss: {metrics['total_loss']:.4f} "
                    f"(Video: {metrics['video_loss']:.4f}, Action: {metrics['action_loss']:.4f}), "
                    f"LR(main/wan): {lr_main:.2e}/{lr_wan:.2e}, "
                    + (f"LR(depth): {lr_depth:.2e}, " if lr_depth is not None else "")
                    + (f"LR(clone): {lr_clone:.2e}, " if lr_clone is not None else "")
                    + f"Source: {batch_source_dataset or 'unknown'} "
                    + f"Seen: {dict(self.source_dataset_counts)}, "
                    + format_step_timing(wall_timing)
                )
                if step_record.get("ready_queue_enabled"):
                    log_str += (
                        f", QueueWait: {step_record.get('ready_queue_get_wait_seconds', 0.0):.3f}s"
                        f", QueueSize: {step_record.get('ready_queue_size_after_get')}"
                        f", QueueEmpty: {int(bool(step_record.get('ready_queue_was_empty', False)))}"
                    )
                if step_record.get("optimizer_step_skipped"):
                    log_str += (
                        f", OptimizerSkipped: {step_record.get('optimizer_step_skip_reason')}"
                    )
                logger.info(log_str)

                # Log to WandB
                if "wandb" in self.report_to:
                    wandb.log({
                        **metrics,
                        'learning_rate_main': lr_main,
                        'learning_rate_wan': lr_wan,
                        'step_time': wall_timing['total_seconds'],
                        **{f'timing/{key}': value for key, value in wall_timing.items()},
                        'epoch': epoch,
                        'global_step': self.global_step,
                        })

                # Log to TensorBoard
                if self.tb_writer is not None:
                    for key, value in metrics.items():
                        self.tb_writer.add_scalar(f'train/{key}', value, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_main', lr_main, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_wan', lr_wan, self.global_step)
                    if lr_depth is not None:
                        self.tb_writer.add_scalar('train/learning_rate_depth', lr_depth, self.global_step)
                    self.tb_writer.add_scalar('train/step_time', wall_timing['total_seconds'], self.global_step)
                    for key, value in wall_timing.items():
                        self.tb_writer.add_scalar(f'timing/{key}', value, self.global_step)
                    self.tb_writer.add_scalar('train/epoch', epoch, self.global_step)
                    if step_record.get("ready_queue_enabled"):
                        self.tb_writer.add_scalar(
                            'data/ready_queue_get_wait_seconds',
                            step_record.get("ready_queue_get_wait_seconds", 0.0),
                            self.global_step,
                        )
                        self.tb_writer.add_scalar(
                            'data/ready_queue_size_after_get',
                            step_record.get("ready_queue_size_after_get", 0),
                            self.global_step,
                        )
                        self.tb_writer.add_scalar(
                            'data/ready_queue_was_empty',
                            int(bool(step_record.get("ready_queue_was_empty", False))),
                            self.global_step,
                        )
                    for source_name, count in self.source_dataset_counts.items():
                        self.tb_writer.add_scalar(f"data/source_seen/{source_name}", count, self.global_step)

            self.step_timing_writer.write(step_record)
            # monitor 按独立间隔记录（默认每步；console/TB 仍用 system.log_interval）
            if (
                self.rank == 0
                and self.monitor_metrics_interval > 0
                and self.global_step % self.monitor_metrics_interval == 0
            ):
                if "lr_main" not in step_record:
                    lrs = [g["lr"] for g in self.optimizer.param_groups]
                    step_record["lr_main"] = float(lrs[0]) if lrs else 0.0
                    step_record["lr_wan"] = float(lrs[1]) if len(lrs) > 1 else step_record["lr_main"]
                _monitor_call(self.monitor, "log_step_record", step_record)
            # 训练切片全 rank 记录：每 rank 把自己的 batch 切片（episode/窗口）
            # 写进 runs/<run_id>/slices/rank%04d.jsonl（server 端按 node/gpu
            # 聚合；与 log_step_record 同间隔，_monitor_call 容错不影响训练）。
            if (
                self.monitor_metrics_interval > 0
                and self.global_step % self.monitor_metrics_interval == 0
            ):
                slice_episode_ids = step_record.get("episode_ids") or []
                if slice_episode_ids:
                    slice_source = str(
                        step_record.get("mix_source_name")
                        or step_record.get("source_dataset")
                        or "unknown"
                    )
                    slice_frames = step_record.get("frame_indices") or []
                    slice_ratios = step_record.get("action_video_fps_ratios") or []
                    samples = []
                    for i, episode_id in enumerate(slice_episode_ids):
                        frames = slice_frames[i] if i < len(slice_frames) else []
                        ratio = slice_ratios[i] if i < len(slice_ratios) else None
                        samples.append(
                            {
                                "source": slice_source,
                                "episode_id": str(episode_id),
                                "window_start": int(frames[0]) if frames else None,
                                "frame_indices": [int(f) for f in frames],
                                "ratio": int(ratio) if ratio else None,
                            }
                        )
                    _monitor_call(
                        self.monitor,
                        "log_slice_record",
                        {"step": int(self.global_step), "samples": samples,
                         "local_metrics": local_task_metrics,
                         "optimizer_step_skipped": local_optimizer_skipped},
                    )
            self._write_heartbeat(
                "loop_done",
                {"step": int(self.global_step), "loop_total_seconds": step_record["loop_total_seconds"]},
                include_memory=include_memory,
                ready_queue=data_pipeline,
            )
            del batch, metrics, train_step_timing, step_record
            if stop_requested:
                break
        logger.info("Data pipeline final snapshot: %s", data_pipeline.snapshot())
        data_pipeline.close()
        if self.val_dataloader is not None:
            self.val_dataloader.close()
        self.periodic_evaluator.close()
        self.action_diagnostics_writer.close()
        self.heartbeat_writer.close()

        total_time = time.time() - start_time
        if self.rank == 0:
            outcome = "stopped after saving" if stopped_for_request else "completed"
            logger.info(f"UniDiffuser training {outcome} in {total_time:.2f}s ({self.global_step} steps)")

        # save_checkpoint() is collective because accelerator.save_state() and the
        # surrounding barriers involve every rank. Do not call it from rank 0 only.
        # If the final step was already saved by the interval checkpoint above,
        # avoid re-entering the same checkpoint path and removing its completion
        # marker.
        if getattr(self, "_last_checkpoint_step", None) != self.global_step:
            self.save_checkpoint()
        self.step_timing_writer.close()
        _monitor_call(self.monitor, "close", "completed")

    def _maybe_simulate_elastic_failure(self):
        """Optionally fail one rank once to validate torch elastic restart."""
        if self.config is None or not hasattr(self.config, "elastic_test"):
            return

        test_cfg = self.config.elastic_test
        if not bool(test_cfg.get("enabled", False)):
            return

        target_rank = int(test_cfg.get("rank", 0))
        fail_after_step = int(test_cfg.get("fail_after_step", 0))
        if self.rank != target_rank or self.global_step < fail_after_step:
            return

        marker_name = str(test_cfg.get("marker_file", ".elastic_failure_injected"))
        marker_path = self.checkpoint_dir / marker_name
        if marker_path.exists():
            return

        if self.rank == target_rank:
            marker_path.write_text(
                f"Injected elastic failure at global_step={self.global_step}, rank={self.rank}\n"
            )
            logger.error(
                "Injecting one-time elastic failure for validation: "
                f"rank={self.rank}, global_step={self.global_step}, marker={marker_path}"
            )
            raise RuntimeError("Intentional one-time elastic failure for validation")

def create_model_and_optimizer(config: OmegaConf, timing_events: Optional[list] = None) -> tuple:
    """Create UniDiffuser model and optimizer from config."""
    model_config = MotusConfig(
        # pretrained
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        # resume/finetune 热启动时 main() 置 False：跳过预训练 backbone 加载，
        # 权重全部由 checkpoint 恢复（见 main() 开关设置处注释）
        load_pretrained_backbones=getattr(config.model, "load_pretrained_backbones", None),

        # data setting
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,

        # network layer
        video_encoder_layer_num=config.model.wm.video_encoder_layer_num,
        video_decoder_layer_num=config.model.wm.video_decoder_layer_num,
        action_encoder_layer_num=config.model.wm.action_encoder_layer_num,
        action_decoder_layer_num=config.model.wm.action_decoder_layer_num,
        und_layer_num=config.model.wm.und_layer_num,
        action_t_downsample = config.model.wm.action_t_downsample,
        flow_shift=float(getattr(config.model.wm, "flow_shift", 5.0)),

        # Video dimensions from config
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        # loss weight
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        freeze_action_branch=bool(getattr(config.model, "freeze_action_branch", False)),
        gradient_checkpointing=bool(getattr(config.model, "gradient_checkpointing", False)),
        checkpoint_video_branch=bool(getattr(config.model, "checkpoint_video_branch", True)),
        checkpoint_uni_branch=bool(getattr(config.model, "checkpoint_uni_branch", True)),
        checkpoint_action_branch=bool(getattr(config.model, "checkpoint_action_branch", False)),
        timing_level=int(getattr(config.model, "timing_level", 1)),
        vlm_rewrite_cache_dir=str(getattr(config.model, "vlm_rewrite_cache_dir", "")),
        vlm_field_dropout_prob=float(getattr(config.model, "vlm_field_dropout_prob", 0.0)),
        vlm_text_rewrite=bool(getattr(config.model, "vlm_text_rewrite", False)),
        vlm_rewrite_max_new_tokens=int(getattr(config.model, "vlm_rewrite_max_new_tokens", 128)),
        training_mode=getattr(config, 'training_mode', 'finetune'),
        robot_configs=get_robot_configs(config),
        unified_action_dim=int(getattr(config.common, "unified_action_dim", 64)),
        robot_layout=get_robot_layout(config),
        # 语义槽位注册表透传（B3 归一化维分类 / B1 eef pos 相对化用）
        unified_bands=(
            OmegaConf.to_object(config.dataset.unified_bands)
            if hasattr(config, "dataset") and hasattr(config.dataset, "unified_bands")
            else None
        ),
        # 20260811 MotusBrain 对齐（B1/B2/B3/D1，见 configs/stage2_20260807_v2.yaml）：
        # D1 per-modality timeshift（action 独立 σ shift；None = 跟随 flow_shift）
        action_flow_shift=(
            float(config.model.wm.action_flow_shift)
            if getattr(config.model.wm, "action_flow_shift", None) is not None
            else None
        ),
        # B3 action/state 归一化策略（zscore | gripper_hand_minmax）
        action_norm_policy=str(getattr(config.model.wm, "action_norm_policy", "zscore")),
        # B1/B2 动作目标表示（absolute_native | relative_eef_rot6）
        action_target_repr=str(getattr(config.model.wm, "action_target_repr", "absolute_native")),
        # 模态 dropout（评审 #4）：video/action/prompt 各自独立概率 + 独立噪声
        modality_dropout_video=float(getattr(config.model.wm, "modality_dropout_video", 0.0)),
        modality_dropout_action=float(getattr(config.model.wm, "modality_dropout_action", 0.0)),
        modality_dropout_prompt=float(getattr(config.model.wm, "modality_dropout_prompt", 0.0)),
        independent_modality_noise=bool(getattr(config.model.wm, "independent_modality_noise", False)),
        # depth 第三生成流（Rynn 式 depth-as-video）
        enable_depth=bool(getattr(config.model.wm, "enable_depth", False)),
        depth_condition_mode=str(getattr(config.model.wm, "depth_condition_mode", "first_frame")),
        depth_loss_weight=float(getattr(config.model.wm, "depth_loss_weight", 1.0)),
        depth_motion_loss_enable=bool(
            getattr(config.model.wm, "depth_motion_loss_enable", False)),
        depth_motion_lambda=float(
            getattr(config.model.wm, "depth_motion_lambda", 0.3)),
        depth_motion_frac_max=float(
            getattr(config.model.wm, "depth_motion_frac_max", 0.12)),
        depth_motion_ego_pose=bool(
            getattr(config.model.wm, "depth_motion_ego_pose", False)),
        depth_motion_require_pose=bool(
            getattr(config.model.wm, "depth_motion_require_pose", False)),
        depth_edge_loss_enable=bool(
            getattr(config.model.wm, "depth_edge_loss_enable", False)),
        depth_edge_lambda=float(
            getattr(config.model.wm, "depth_edge_lambda", 0.5)),
        depth_edge_persist=bool(
            getattr(config.model.wm, "depth_edge_persist", True)),
        depth_z_near=float(getattr(config.depth, "z_near", 0.15))
        if getattr(config, "depth", None) is not None else 0.15,
        depth_z_far=float(getattr(config.depth, "z_far", 4.0))
        if getattr(config, "depth", None) is not None else 4.0,
        depth_mode=str(getattr(config.depth, "mode", "log"))
        if getattr(config, "depth", None) is not None else "log",
        depth_read_by_action=bool(getattr(config.model.wm, "depth_read_by_action", False)),
        depth_read_by_video=bool(getattr(config.model.wm, "depth_read_by_video", False)),
        depth_read_by_und=bool(getattr(config.model.wm, "depth_read_by_und", False)),
        depth_detach_video_kv=bool(getattr(config.model.wm, "depth_detach_video_kv", False)),
        depth_share_video_noise=bool(
            getattr(config.model.wm, "depth_share_video_noise", False)),
        depth_uni_clone_layers=int(
            getattr(config.model.wm, "depth_uni_clone_layers", 0) or 0),
        modality_dropout_depth=float(getattr(config.model.wm, "modality_dropout_depth", 0.0)),
        freeze_depth_branch=bool(getattr(config.model, "freeze_depth_branch", False)),
        # 20260816：零 action loss 样本 σ 强制最大噪声端点（默认关，本轮暂缓
        # 使用、机制保留备用）
        zero_action_loss_force_max_noise=bool(
            getattr(config.model.wm, "zero_action_loss_force_max_noise", False)),
        # 变 FPS 条件注入已移除（20260815 固定 5fps 轮，models/fps_condition.py
        # 删除；MotusConfig 不再有 fps_condition 字段）
        # block-causal 时间布局 / 历史 / 恢复条件（未配置的键保持 MotusConfig 默认 = 旧行为）
        **motus_temporal_kwargs(config),
    )
    
    from thinker_wm.models.capability_training import enabled as capabilities_enabled, validate_config as validate_capabilities
    from thinker_wm.data_prepare.capability_annotations import annotation_paths, CapabilityAnnotations
    validate_capabilities(model_config)
    capability_paths = annotation_paths(config)
    if capabilities_enabled(model_config) and not capability_paths:
        raise ValueError("capability objectives require verified capability_annotations_path or capability_ranges sources")
    if capability_paths:
        CapabilityAnnotations(capability_paths)  # Fail on malformed labels before loading the 5B model.
    if model_config.capability_language_weight > 0 and (model_config.vlm_text_rewrite or model_config.vlm_field_dropout_prob > 0):
        raise ValueError("language pairs require verbatim instructions: disable VLM rewrite/field dropout")

    # Create model (Accelerator will handle device placement and DDP)
    model_start = time.perf_counter()
    model = UBT_WM(model_config)
    add_timing_event(timing_events, "model.init_ubt_wm", model_start)
    
    # Optimizer - parameter groups for separate WAN (video model) learning rate
    optimizer_start = time.perf_counter()
    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))

    # Collect WAN params explicitly (exclude VAE, we only train diffusion WAN)
    all_trainable = [p for p in model.parameters() if p.requires_grad]

    # depth I/O 独立参数组：depth_learning_rate > 0 时把 depth 专属 patch
    # embed / head / stream_embed 拆出来给更高 LR，共享骨架仍走 base_lr。
    # 参数组数量变化与 checkpoint 里的优化器状态不兼容，只能配合仅加载权重的
    # finetune 热启动使用（完整 resume 会因参数组数不匹配失败）。
    depth_lr = float(getattr(config.training, 'depth_learning_rate', 0.0) or 0.0)
    depth_params = model.depth_io_parameters() if depth_lr > 0 else []
    depth_ids = {id(p) for p in depth_params if p.requires_grad}

    # depth uni 克隆层独立参数组：从已训好的 RGB uni 层 deepcopy 而来，既不该
    # 用 depth I/O 那档新参数的高 LR 冲掉预训练结构，也不该跟共享骨架一样保守。
    # 未配 depth_clone_learning_rate 时并入 depth 组（保持两组行为）。
    clone_lr = float(getattr(config.training, 'depth_clone_learning_rate', 0.0) or 0.0)
    clone_params = model.depth_uni_clone_parameters() if clone_lr > 0 else []
    clone_ids = {id(p) for p in clone_params if p.requires_grad} - depth_ids

    param_groups = []
    if depth_ids or clone_ids:
        backbone = [p for p in all_trainable
                    if id(p) not in depth_ids and id(p) not in clone_ids]
        depth_trainable = [p for p in all_trainable if id(p) in depth_ids]
        clone_trainable = [p for p in all_trainable if id(p) in clone_ids]
        param_groups.append({'params': backbone, 'lr': base_lr, 'name': 'backbone'})
        if depth_trainable:
            param_groups.append({'params': depth_trainable, 'lr': depth_lr, 'name': 'depth'})
        if clone_trainable:
            param_groups.append({'params': clone_trainable, 'lr': clone_lr, 'name': 'depth_clone'})
        logger.info(
            "Depth I/O parameter group: tensors=%d params=%.3fM lr=%.2e "
            "(backbone tensors=%d lr=%.2e)",
            len(depth_trainable),
            sum(p.numel() for p in depth_trainable) / 1e6,
            depth_lr,
            len(backbone),
            base_lr,
        )
        if clone_trainable:
            logger.info(
                "Depth uni clone parameter group: tensors=%d params=%.3fM lr=%.2e",
                len(clone_trainable),
                sum(p.numel() for p in clone_trainable) / 1e6,
                clone_lr,
            )
    else:
        param_groups.append({'params': all_trainable, 'lr': base_lr})

    from thinker_wm.models.history_optimization import split_history_groups
    param_groups = split_history_groups(
        param_groups, getattr(model.world_model, "history_encoder", None),
        adapter_lr=float(getattr(config.training, "history_learning_rate", 0.0)),
        bias_lr=float(getattr(config.training, "history_bias_learning_rate", 0.0)))
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95)
    )
    add_timing_event(timing_events, "optimizer.init_adamw", optimizer_start)
    
    # Scheduler
    scheduler_start = time.perf_counter()
    scheduler = create_scheduler(optimizer, config)
    add_timing_event(timing_events, "scheduler.init", scheduler_start)
    
    return model, optimizer, scheduler

def _write_validation_record(config, step, *, ok, error=None, metrics=None) -> str:
    """Persist one validation round next to checkpoints, including a failure."""
    folder = os.path.join(config.system.checkpoint_dir, "validation")
    os.makedirs(folder, exist_ok=True)
    payload = {"step": int(step), "ok": bool(ok), "error": None if error is None else str(error)}
    if metrics:
        by_task = metrics.get("_by_task")
        windows = metrics.get("_windows")
        payload["metrics"] = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value == value
        }
        if by_task:
            payload["by_task"] = by_task
        if windows:
            payload["windows"] = windows
    path = os.path.join(folder, f"step_{int(step)}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _validation_enabled(config: OmegaConf) -> bool:
    val_interval = int(getattr(config.system, "val_interval", 0))
    if val_interval <= 0:
        return False
    # The existing configs use a very large sentinel to effectively disable val.
    if val_interval >= 1_000_000_000:
        return False
    return True


def _pipeline_batch_iter(pipeline, history_attacher=None):
    """Iterate batches from a data_prepare pipeline (eval use; caller breaks)."""
    while True:
        batch, _stats = pipeline.next()
        if history_attacher is not None and "history_recent_rgb" not in batch:
            batch = history_attacher.attach(batch, step=0)
        yield batch


def create_dataloaders(config: OmegaConf, rank: int, world_size: int, timing_events: Optional[list] = None) -> tuple:
    """Create train/val indexed-mix data pipelines (motus_data Catalog/SamplePool).

    取代旧的 create_dataset + DataLoader + ReadyBatchQueue 组合：
    返回的 train pipeline 内建 ready batch queue（stats 键与旧队列一致），
    ``pipeline.next()`` 直接产出 (batch, stats)；val pipeline 无 queue，
    支持单卡完整集合或所有 rank 的不重复有限分片。
    """
    # 数据采样种子：决定整条采样链（跨源调度/组内洗牌/窗口起点）的确定性。
    # 缺省不再用 0——0 会让所有运行看到完全相同的训练数据；新建配置须在
    # training.seed 显式写新的非零值（见 .AGENTS_CONTENT/03_config.md）。
    seed = int(getattr(config.training, "seed", 20260818) or 20260818)
    train_pipeline = create_training_data_pipeline(
        config,
        rank=rank,
        world_size=world_size,
        seed=seed,
        timing_events=timing_events,
    )

    val_pipeline = None
    if _validation_enabled(config):
        val_rank, val_world = rank, world_size
        # Legacy rank-0 evaluation needs the complete set. Distributed full-set
        # evaluation below uses an unpadded, finite partition instead.
        if bool(getattr(config.system, "val_full_set", False)):
            val_rank, val_world = 0, 1
        if bool(getattr(config.system, "val_distributed", False)):
            from thinker_wm.data_prepare.finite_validation import FiniteValidationPipeline
            val_pipeline = FiniteValidationPipeline(
                config, rank=rank, world_size=world_size, seed=seed)
        else:
            val_pipeline = create_eval_data_pipeline(
                config, rank=val_rank, world_size=val_world,
                seed=seed, timing_events=timing_events)
    else:
        add_timing_event(
            timing_events,
            "dataset.create_val_skipped",
            time.perf_counter(),
            reason="val_interval_disabled",
            val_interval=int(getattr(config.system, "val_interval", 0)),
        )

    snapshot = train_pipeline.snapshot()
    print(f"[Rank {rank}] Training sources: {snapshot.get('mix_sources')}")
    if val_pipeline is not None:
        print(f"[Rank {rank}] Eval Set enabled")
    else:
        print(f"[Rank {rank}] Eval Set disabled")
    return train_pipeline, val_pipeline

def main():
    startup_events = []
    startup_total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train Three-Modal UniDiffuser Model")
    
    # Configuration file
    parser.add_argument("--config", type=str, 
                       default="configs/aloha_agilex_2.yaml",
                       help="Path to configuration file")
    
    # System settings
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--save_interval", type=int, default=None, help="Override system.save_interval")
    parser.add_argument("--log_interval", type=int, default=None, help="Override system.log_interval")
    
    # Logging settings
    parser.add_argument("--report_to", type=str, default=None, 
                       choices=["wandb", "tensorboard", "all", "none"],
                       help="Logging backends to use")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")
    
    # DeepSpeed settings
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument(
        "--auto_resume_latest",
        action="store_true",
        help="Resume from the latest checkpoint_step_* under the run checkpoint directory when resume.checkpoint_path is empty.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint directory to resume from. Overrides resume.checkpoint_path in the config.",
    )
    parser.add_argument(
        "--finetune_checkpoint",
        type=str,
        default=None,
        help="Load model weights only from this checkpoint. Optimizer/scheduler are initialized from scratch.",
    )
    parser.add_argument(
        "--elastic_simulate_failure_step",
        type=int,
        default=None,
        help="For elastic validation only: fail one target rank once when global_step reaches this value.",
    )
    parser.add_argument(
        "--elastic_simulate_failure_rank",
        type=int,
        default=0,
        help="Rank used with --elastic_simulate_failure_step.",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_start = time.perf_counter()
    config = load_config(args.config)
    if config.get("dexjoco") is not None:
        from accelerate.utils import set_seed
        from thinker_wm.data_prepare.dexjoco.recipe import install_statistics
        set_seed(int(config.training.seed))
        install_statistics(config)
    add_timing_event(startup_events, "config.load", config_start, config_path=args.config)
    if args.checkpoint_dir is not None:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.report_to is not None:
        config.logging.report_to = args.report_to
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.run_name is not None:
        config.logging.run_name = args.run_name
    if args.max_steps is not None:
        config.training.max_steps = int(args.max_steps)
    if args.save_interval is not None:
        config.system.save_interval = int(args.save_interval)
    if args.log_interval is not None:
        config.system.log_interval = int(args.log_interval)
    if args.auto_resume_latest:
        if not hasattr(config, "resume"):
            config.resume = {}
        config.resume.auto_resume_latest = True
    if args.resume_checkpoint is not None:
        if not hasattr(config, "resume"):
            config.resume = {}
        config.resume.checkpoint_path = args.resume_checkpoint
    if args.finetune_checkpoint is not None:
        if not hasattr(config, "finetune"):
            config.finetune = {}
        config.finetune.checkpoint_path = args.finetune_checkpoint
    if args.elastic_simulate_failure_step is not None:
        config.elastic_test = {
            "enabled": True,
            "rank": int(args.elastic_simulate_failure_rank),
            "fail_after_step": int(args.elastic_simulate_failure_step),
            "marker_file": ".elastic_failure_injected",
        }
    # Extract dataset name from config file path for checkpoint organization
    config_filename = os.path.basename(args.config)  # e.g., "ac_one.yaml"
    dataset_name = os.path.splitext(config_filename)[0]  # e.g., "ac_one"
    
    # Update checkpoint directory to include dataset name
    base_checkpoint_dir = resolve_output_path(config.system.checkpoint_dir)
    logger.info(f"Checkpoint root resolved to: {base_checkpoint_dir}")
    config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, dataset_name)
    
    # Create the dataset directory if it doesn't exist
    mkdir_start = time.perf_counter()
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    add_timing_event(startup_events, "checkpoint_dir.create_dataset_dir", mkdir_start, path=config.system.checkpoint_dir)
    
    # Initialize Accelerator with DeepSpeed (if provided)
    accelerator_start = time.perf_counter()
    accelerator_project_config = ProjectConfiguration(total_limit=20)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=config.training.get('gradient_accumulation_steps', 1),
        mixed_precision="bf16",
        log_with=config.logging.get('report_to', 'tensorboard'),
        project_dir=config.system.checkpoint_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=([InitProcessGroupKwargs(timeout=timedelta(
            seconds=int(config.system.distributed_timeout_seconds)))]
            if config.system.get("distributed_timeout_seconds") is not None else []),
    )
    add_timing_event(startup_events, "accelerator.init", accelerator_start, deepspeed=bool(args.deepspeed))
    
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level)

    # MixDataset uses a batch_sampler, so PyTorch reports dataloader.batch_size=None.
    # DeepSpeed/Accelerate cannot infer "auto" micro batch size from that; set it
    # explicitly from the training config before accelerator.prepare().
    if args.deepspeed is not None and accelerator.state.deepspeed_plugin is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        micro_batch_size = int(config.training.batch_size)
        grad_accum = int(config.training.get('gradient_accumulation_steps', 1))
        ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
        ds_config["gradient_accumulation_steps"] = grad_accum
        ds_config["train_batch_size"] = micro_batch_size * grad_accum * world_size
        logger.info(
            "DeepSpeed batch sizes set explicitly: "
            f"micro={micro_batch_size}, grad_accum={grad_accum}, "
            f"train_batch={ds_config['train_batch_size']}"
        )
    
    # Handle report_to settings - expand "all" to individual backends
    report_to = config.logging.report_to
    if report_to == "all":
        report_to = ["wandb", "tensorboard"]
    elif report_to == "none":
        report_to = []
    elif isinstance(report_to, str):
        report_to = [report_to]
    
    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.logging.get('run_name', None)
    if not run_name:
        run_name = f"unidiffuser_{config.dataset.type}_bs{config.training.batch_size}_lr{config.training.learning_rate}"
    
    # Update checkpoint directory to include run name
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    run_dir_start = time.perf_counter()
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    add_timing_event(startup_events, "checkpoint_dir.create_run_dir", run_dir_start, path=config.system.checkpoint_dir)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Checkpoints will be saved to: {config.system.checkpoint_dir}")

    resume_path = getattr(config.resume, "checkpoint_path", None) if hasattr(config, "resume") else None
    auto_resume_latest = bool(
        getattr(config.resume, "auto_resume_latest", False)
    ) if hasattr(config, "resume") else False
    if not resume_path and auto_resume_latest:
        latest_checkpoint = find_latest_checkpoint(config.system.checkpoint_dir)
        if latest_checkpoint is not None:
            config.resume.checkpoint_path = latest_checkpoint
            resume_path = latest_checkpoint
            logger.info(f"Auto resume latest checkpoint: {latest_checkpoint}")
        else:
            logger.info(
                "Auto resume latest checkpoint enabled, but no complete "
                f"checkpoint_step_* found under {config.system.checkpoint_dir}. "
                "Starting from scratch. Use --resume_checkpoint to load a legacy "
                "checkpoint without a completion marker."
            )
    elif resume_path:
        logger.info(f"Explicit resume checkpoint configured: {resume_path}")

    # Decide backbone loading policy:
    # If resuming or finetuning from a pretrain checkpoint, skip loading WAN/VLM pretrained weights.
    # 依据：ZeRO-1 ckpt 保存完整 module.state_dict()（exclude_frozen_parameters
    # 未设置，冻结的 VLM/VAE 权重也在其中），resume 为 strict 加载、finetune 走
    # load_pretrain_weights 全量覆盖——预训练 backbone 加载在此会被完全覆盖，
    # 纯属冗余（每 rank 白读 VLM ~4GB + Wan DiT ~10GB+），故置 False 让
    # UBT_WM 只建架构。仅在 checkpoint 路径真实存在时跳过：路径写错时维持
    # "回退预训练权重继续"的现状容错语义，避免静默用随机 backbone 训练。
    try:
        resume_ckpt = getattr(config.resume, 'checkpoint_path', None)
        finetune_ckpt = (
            getattr(config.finetune, 'checkpoint_path', None)
            if hasattr(config, 'finetune') else None
        )
        # 与权重实际恢复条件严格对齐：resume 在 train() 开头无条件全量恢复；
        # finetune 仅 training_mode=='finetune' 时由 load_pretrain_weights 覆盖
        # （见下方模型创建后的加载分支）——防止 pretrain 模式误挂
        # finetune.checkpoint_path 时跳过预训练加载却无人恢复权重。
        finetune_active = getattr(config, 'training_mode', 'finetune') == 'finetune'
        backbone_ckpt = resume_ckpt or (finetune_ckpt if finetune_active else None)
        if backbone_ckpt:
            if os.path.exists(str(backbone_ckpt)):
                config.model.load_pretrained_backbones = False
                logger.info(
                    f"Checkpoint {backbone_ckpt} will restore all model weights; "
                    "skipping pretrained WAN/VLM backbone loading"
                )
            else:
                logger.warning(
                    f"Checkpoint path {backbone_ckpt} does not exist; keeping "
                    "pretrained WAN/VLM backbone loading enabled"
                )
    except Exception:
        pass

    # Training monitor — created EARLY (right after checkpoint dir / resume
    # resolution) so it also covers the initialization phase (model/dataloader
    # build, accelerate prepare), not just the training loop. torch.distributed
    # is already up (accelerator init), so all ranks share rank 0's run id via
    # broadcast inside create_monitor. No-op unless config.monitor.enabled or
    # MOTUS_MONITOR_ENABLED=1.
    monitor = None
    if create_monitor is not None:
        try:
            monitor = create_monitor(
                config,
                run_name=run_name,
                checkpoint_dir=config.system.checkpoint_dir,
                max_steps=int(config.training.max_steps),
                rank=rank,
                local_rank=int(os.environ.get("LOCAL_RANK", 0)),
                world_size=world_size,
                resume_from=getattr(config.resume, "checkpoint_path", None) if hasattr(config, "resume") else None,
                finetune_from=getattr(config.finetune, "checkpoint_path", None) if hasattr(config, "finetune") else None,
            )
            # 回填 monitor 创建之前已发生的初始化阶段
            global _EARLY_MONITOR
            _EARLY_MONITOR = monitor
            for _ev in startup_events:
                _monitor_call(monitor, "log_init_phase", _ev.get("name"), _ev.get("seconds"))
        except Exception as exc:
            logger.warning("Monitor init failed, continuing without it: %s", exc)
            monitor = None
    
    # Initialize TensorBoard writer
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        tb_start = time.perf_counter()
        tb_log_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        config_dict = OmegaConf.to_container(config, resolve=True)
        tb_writer.add_text('config', yaml.dump(config_dict))
        add_timing_event(startup_events, "logging.tensorboard_init", tb_start, path=tb_log_dir)
    
    # Initialize WandB
    if rank == 0 and "wandb" in report_to:
        wandb_start = time.perf_counter()
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )
        add_timing_event(startup_events, "logging.wandb_init", wandb_start)
    
    try:
        # Create model and optimizer
        logger.info("Creating UniDiffuser model and optimizer...")
        model, optimizer, scheduler = create_model_and_optimizer(config, timing_events=startup_events)

        # Optional: load finetune weights for partial init
        resume_ckpt = (
            getattr(config.resume, 'checkpoint_path', None)
            if hasattr(config, 'resume') else None
        )
        configured_finetune_ckpt = (
            getattr(config.finetune, 'checkpoint_path', None)
            if hasattr(config, 'finetune') else None
        )
        # Full resume is authoritative. In particular, auto-resume must not first
        # read the YAML's legacy finetune checkpoint only to overwrite it later.
        finetune_ckpt = None if resume_ckpt else configured_finetune_ckpt
        if resume_ckpt and configured_finetune_ckpt:
            logger.info(
                "Resume checkpoint %s takes precedence; skipping finetune "
                "checkpoint %s.",
                resume_ckpt,
                configured_finetune_ckpt,
            )
        if getattr(config, 'training_mode', 'finetune') == 'finetune' and finetune_ckpt:
            logger.info(f"Loading finetune weights from {finetune_ckpt} (partial)...")
            finetune_start = time.perf_counter()
            try:
                transfer_options = {}
                if getattr(config.finetune, "action_interface_policy", None) is not None:
                    transfer_options = dict(
                        action_interface_policy=config.finetune.action_interface_policy,
                        audit_path=(os.path.join(config.system.checkpoint_dir, "initialization_audit.json")
                                    if rank == 0 else None),
                    )
                (model.module if hasattr(model, 'module') else model).load_pretrain_weights(
                    finetune_ckpt, **transfer_options)
                logger.info("Finetune weights loaded (partial).")
                copy_rows = getattr(config.finetune, "robot_token_copy_rows", None)
                if copy_rows:
                    from thinker_wm.models.robot_tokens import RobotTokenEmbedding
                    unwrapped = model.module if hasattr(model, 'module') else model
                    tables = [m for m in unwrapped.modules() if isinstance(m, RobotTokenEmbedding)]
                    if not tables:
                        raise RuntimeError("finetune.robot_token_copy_rows set but model has no robot token table")
                    for m in tables:
                        m.copy_rows(copy_rows)
                    logger.info("Robot token rows copied (src->dst) %s on %d table(s).",
                                [list(p) for p in copy_rows], len(tables))
            except Exception as e:
                # 加载失败不能继续：否则会用随机初始化的权重跑完整训练。
                logger.error(f"Failed to load finetune weights: {e}")
                raise
            finally:
                add_timing_event(startup_events, "checkpoint.load_finetune_weights", finetune_start, path=str(finetune_ckpt))
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(config, rank, world_size, timing_events=startup_events)
        
        # Create custom saving hook to avoid NCCL timeout issues
        def save_model_hook(models, weights, output_dir):
            """Custom save hook to save model safely and avoid NCCL timeouts."""
            if accelerator.is_main_process:
                logger.info(f"Saving model to {output_dir}")
                for i, model_to_save in enumerate(models):
                    # Unwrap the model if it's wrapped by DDP/DeepSpeed
                    unwrapped_model = accelerator.unwrap_model(model_to_save)
                    
                    # Save using torch.save instead of accelerator's default method
                    model_save_path = os.path.join(output_dir, f"pytorch_model_{i}.bin")
                    torch.save(unwrapped_model.state_dict(), model_save_path)
                    logger.info(f"Model {i} saved to {model_save_path}")
        
        # Register the custom save hook
        accelerator.register_save_state_pre_hook(save_model_hook)
        
        # Prepare model/optimizer/scheduler with accelerator. 数据管线不参与 prepare：
        # motus_data indexed 管线按 rank/world_size 在构造时自行分片（见 data_prepare.factory）。
        logger.info("Preparing model, optimizer, and scheduler with Accelerator...")
        prepare_start = time.perf_counter()
        model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )
        add_timing_event(startup_events, "accelerator.prepare_train_objects", prepare_start)
        
        # Training monitor was created earlier (right after checkpoint dir /
        # resume resolution, so initialization phases are covered too).

        # Create trainer
        trainer = UniDiffuserTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=accelerator.device,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval,
            save_interval=config.system.save_interval,
            val_interval=config.system.val_interval,
            report_to=report_to,
            tb_writer=tb_writer,
            accelerator=accelerator,
            config=config,
            startup_timing_events=startup_events,
            startup_start_time=startup_total_start,
            monitor=monitor,
        )
        
        # Start training
        trainer.train(
            max_steps=config.training.max_steps, 
            resume_from=getattr(config.resume, "checkpoint_path", None),
            val_interval=config.system.val_interval
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        print(f"[CRITICAL ERROR] Training failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        _monitor_call(monitor, "log_event", "error", message=str(e))
        _monitor_call(monitor, "close", "failed")
        exit_on_fatal_cuda_error(e)
        raise
    finally:
        # Clean up resources
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()

if __name__ == "__main__":
    main()
