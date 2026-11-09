"""Runtime guards for obviously invalid action/state tensors."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import torch


class ActionMagnitudeOutlier(ValueError):
    """Raised when a sample has action/state magnitudes too large to train on."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ActionStateNonFinite(ValueError):
    """Raised when selected action/state values contain NaN or infinity."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


def _parse_source_set(raw: Optional[str]) -> set[str]:
    if raw is None:
        return {"agibot_beta"}
    raw = str(raw).strip()
    if not raw:
        return set()
    if raw.lower() in {"*", "all"}:
        return {"*"}
    return {part.strip() for part in raw.split(",") if part.strip()}


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    value = float(raw)
    if value <= 0:
        return None
    return value


def should_guard_source(source_dataset: Optional[str], sources: Optional[Iterable[str]] = None) -> bool:
    source = str(source_dataset or "")
    source_set = set(sources) if sources is not None else _parse_source_set(
        os.getenv("MOTUS_ACTION_MAGNITUDE_GUARD_SOURCES")
    )
    return "*" in source_set or source in source_set


def tensor_abs_max(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None
    with torch.no_grad():
        return float(tensor.detach().float().abs().max().item())


def validate_action_state_finite(
    *,
    source_dataset: Optional[str],
    action_sequence: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    selection_id: Optional[str] = None,
) -> None:
    """Reject non-finite selected values for every source before video I/O.

    环境开关（与 MOTUS_ACTION_MAGNITUDE_GUARD_THRESHOLD 同风格的反向语义）：
    ``MOTUS_ACTION_FINITE_GUARD_ENABLED`` 缺省开启；设为 0/false/no 时本检查
    整体跳过（非有限样本交由下游处理，如训练侧 loss 置零）。
    """

    raw_enabled = os.getenv("MOTUS_ACTION_FINITE_GUARD_ENABLED")
    if raw_enabled is not None and str(raw_enabled).strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return

    nonfinite_counts = {}
    for name, tensor in (
        ("action", action_sequence),
        ("state", initial_state),
    ):
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        count = int((~torch.isfinite(tensor.detach())).sum().item())
        if count:
            nonfinite_counts[name] = count
    if not nonfinite_counts:
        return

    details = {
        "source_dataset": source_dataset,
        "selection_id": selection_id,
        "nonfinite_counts": nonfinite_counts,
    }
    raise ActionStateNonFinite(
        "non-finite action/state values "
        f"source={source_dataset} selection_id={selection_id} "
        f"counts={nonfinite_counts}",
        details=details,
    )


def validate_action_state_magnitude(
    *,
    source_dataset: Optional[str],
    action_sequence: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    manifest_id: Optional[str] = None,
    manifest_task: Optional[str] = None,
    threshold: Optional[float] = None,
    sources: Optional[Iterable[str]] = None,
) -> None:
    """Reject catastrophic action/state outliers before they can poison training."""

    if not should_guard_source(source_dataset, sources=sources):
        return
    limit = _env_float("MOTUS_ACTION_MAGNITUDE_GUARD_THRESHOLD", threshold if threshold is not None else 100.0)
    if limit is None:
        return

    action_abs_max = tensor_abs_max(action_sequence)
    state_abs_max = tensor_abs_max(initial_state)
    observed_values = [value for value in (action_abs_max, state_abs_max) if value is not None]
    if not observed_values:
        return
    observed = max(observed_values)
    if observed <= limit:
        return

    details = {
        "source_dataset": source_dataset,
        "manifest_id": manifest_id,
        "manifest_task": manifest_task,
        "threshold": float(limit),
        "action_abs_max": action_abs_max,
        "state_abs_max": state_abs_max,
        "observed_abs_max": float(observed),
    }
    raise ActionMagnitudeOutlier(
        "action/state magnitude outlier "
        f"source={source_dataset} manifest_id={manifest_id} "
        f"observed_abs_max={observed:.6g} threshold={limit:.6g}",
        details=details,
    )
