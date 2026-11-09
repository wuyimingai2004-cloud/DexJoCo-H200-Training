"""Training-side monitor client.

:class:`MonitorLogger` is the single entry point the training loop talks to.
It writes run metadata, per-step metrics, sparse events and system samples
into the monitor data root (see :mod:`monitor.paths`) using append-only
chunked JSONL so the local web server can sync increments over SSH.

Design rules:

* **Zero training impact** -- every public method swallows its own errors
  (logged once) and, when ``enabled=False``, is a pure no-op.
* **Rank aware** -- rank 0 writes step metrics / events; the process with
  ``local_rank == 0`` on each node additionally samples system stats, so an
  N-node run produces one system stream per node.
* **Resume aware** -- the run id is stored in the checkpoint directory, so
  elastic restarts and auto-resume keep appending to the same run; a run that
  starts from another run's checkpoint records that parent link for the
  checkpoint tree in the web UI.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .paths import (
    checkpoint_step_from_path,
    default_data_dir,
    find_parent_run_id,
    new_run_id,
    read_run_id,
    sanitize_name,
    write_run_id,
)
from .store import ChunkedJsonlWriter, atomic_write_json, read_json
from .system_stats import SystemStatsSampler

logger = logging.getLogger(__name__)

ENABLED_ENV = "MOTUS_MONITOR_ENABLED"


def sanitize(value: Any, depth: int = 0) -> Any:
    """Convert arbitrary training values into strict-JSON-safe data.

    Non-finite floats become ``None`` (NaN/Inf are not valid JSON and would
    break ``JSON.parse`` in the browser); tensors/numpy scalars are unwrapped;
    unknown objects fall back to ``str()``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        if depth >= 8:
            return str(value)
        return {str(k): sanitize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if depth >= 8:
            return str(value)
        return [sanitize(v, depth + 1) for v in list(value)[:128]]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return sanitize(value.item(), depth + 1)
        except Exception:
            pass
    return str(value)


def _detect_node_index() -> Optional[int]:
    """Node index from the launcher env (torchrun elastic / K8s Indexed Job).

    Matches the per-node log naming (train_node<N>_*.log), so monitor node
    labels line up with the rest of the tooling. Falls back to hostname-based
    labels when no launcher index is present.
    """
    for var in ("NODE_RANK", "GROUP_RANK", "JOB_COMPLETION_INDEX"):
        value = os.environ.get(var)
        if value is not None and str(value).strip().lstrip("-").isdigit():
            return int(value)
    return None


class MonitorLogger:
    """Append-only training monitor. No-op when ``enabled=False``."""

    def __init__(
        self,
        run_name: str,
        checkpoint_dir: Optional[object] = None,
        config: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        rank: int = 0,
        local_rank: Optional[int] = None,
        world_size: int = 1,
        resume_from: Optional[str] = None,
        finetune_from: Optional[str] = None,
        data_dir: Optional[object] = None,
        enabled: bool = True,
        system_interval: float = 2.0,
        status_interval: float = 2.0,
        collect_system: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if local_rank is None:
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            except ValueError:
                local_rank = 0
        self.local_rank = int(local_rank)
        self.is_metrics_writer = self.rank == 0
        self.is_system_writer = self.local_rank == 0 and collect_system

        self.run_name = str(run_name)
        self.max_steps = max_steps
        self._status_interval = max(0.5, float(status_interval))
        self._last_status_write = 0.0
        self._closed = False
        self._failed_once = set()
        self._stage_acc: Dict[str, float] = {}
        self._stage_lock = threading.Lock()
        self._sampler: Optional[SystemStatsSampler] = None

        if not self.enabled:
            self.run_id = None
            self.run_dir = None
            return

        self.data_root = Path(str(data_dir)) if data_dir else default_data_dir()
        # Run id resolution (all ranks must agree):
        #   1. explicit run_id (e.g. broadcast from rank 0 by create_monitor)
        #   2. the id recorded in the checkpoint directory (resume continues
        #      the same run)
        #   3. a fresh id (only safe single-process; create_monitor serializes
        #      the distributed case)
        existing = read_run_id(checkpoint_dir) if checkpoint_dir else None
        self.run_id = run_id or existing or new_run_id(self.run_name)
        if checkpoint_dir and not existing:
            write_run_id(checkpoint_dir, self.run_id)
        self.run_dir = self.data_root / "runs" / self.run_id
        self._metrics: Optional[ChunkedJsonlWriter] = None
        self._events: Optional[ChunkedJsonlWriter] = None
        self._system: Optional[ChunkedJsonlWriter] = None
        self._slices_fp: Any = None  # 懒创建：runs/<run_id>/slices/rank%04d.jsonl
        # 节点标签所有 rank 都算一份（切片记录按 node/gpu 维度聚合要用；
        # 规则与 system 流一致：launcher 索引优先，hostname 兜底）
        hostname = os.environ.get("HOSTNAME") or socket.gethostname() or "node"
        node_idx = _detect_node_index()
        if node_idx is not None:
            self._node_label = sanitize_name(f"node{node_idx}")
        else:
            self._node_label = sanitize_name(f"{hostname}-local{self.local_rank}")
        try:
            if self.is_metrics_writer:
                self._metrics = ChunkedJsonlWriter(self.run_dir / "metrics")
                self._events = ChunkedJsonlWriter(self.run_dir / "events")
                self._init_run_document(
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    resume_from=resume_from,
                    finetune_from=finetune_from,
                    extra=extra,
                )
                try:
                    from .sampling_catalog import export_sampling_catalog
                    export_sampling_catalog(self.run_dir, config)
                except Exception as exc:
                    # Diagnostics must not prevent training or regular logging.
                    print(f"[monitor] task catalog export failed: {exc}", flush=True)
            if self.is_system_writer:
                # Per-node subdirectory: on shared filesystems two nodes must
                # never append to the same chunk file concurrently. Node label
                # prefers the launcher index (NODE_RANK etc., matches the
                # train_node<N>_*.log naming); hostname is only a fallback.
                node_label = self._node_label
                self._system = ChunkedJsonlWriter(self.run_dir / "system" / node_label)
                # 节点注册：让 UI/诊断能看到哪些节点在写系统流（排查节点缺失）
                atomic_write_json(
                    self.run_dir / "system" / node_label / "node_info.json",
                    {
                        "node": node_label,
                        "node_rank": _detect_node_index(),
                        "hostname": os.environ.get("HOSTNAME") or socket.gethostname() or "node",
                        "local_rank": self.local_rank,
                        "world_size": self.world_size,
                        "data_root": str(self.data_root),
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                self._sampler = SystemStatsSampler(
                    self._system,
                    interval=system_interval,
                    data_dir=self.data_root,
                    node_label=node_label,
                )
                self._sampler.start()
        except Exception as exc:
            logger.warning("monitor: init failed, disabling (%s)", exc)
            self.enabled = False
            return

        logger.info(
            "monitor: run_id=%s data_root=%s (metrics=%s system=%s)",
            self.run_id,
            self.data_root,
            self.is_metrics_writer,
            self.is_system_writer,
        )

    # ------------------------------------------------------------------ init
    def _init_run_document(
        self,
        config: Optional[Dict[str, Any]],
        checkpoint_dir: Optional[object],
        resume_from: Optional[str],
        finetune_from: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> None:
        run_json_path = self.run_dir / "run.json"
        doc = read_json(run_json_path, default=None)
        is_restart = isinstance(doc, dict)
        parent = self._resolve_parent(resume_from, finetune_from)
        now = datetime.now().isoformat(timespec="seconds")
        if not isinstance(doc, dict):
            doc = {
                "schema": 1,
                "run_id": self.run_id,
                "run_name": self.run_name,
                "created_at": now,
                "config": sanitize(config) if config else {},
                "parent": parent,
                "checkpoints": [],
                "restarts": 0,
            }
        else:
            doc["restarts"] = int(doc.get("restarts", 0)) + 1
            if parent and not doc.get("parent"):
                doc["parent"] = parent
        doc.update(
            {
                "run_name": self.run_name,
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "max_steps": self.max_steps,
                "world_size": self.world_size,
                "last_start_at": now,
                "state": "initializing",
            }
        )
        if extra:
            doc["extra"] = sanitize(extra)
        atomic_write_json(run_json_path, doc)
        # 谱系基点：本次从 resume ckpt（step N）续跑时，N 之后来自旧一次执行的
        # 记录属于"被放弃的 future"，前端据此隐藏（step<=N 的旧记录保留）。
        # 重启（run.json 已存在）但非 resume（从 0 重跑）时基点为 0：旧 attempt
        # 的记录（含 system/GPU 流）全部作废。必须总是显式写——status.json 是
        # update 合并，不写会残留上一次 resume 的陈旧基点。
        status_extra: Dict[str, Any] = {"attempt_started_ts": time.time()}
        if (
            parent
            and parent.get("kind") == "resume"
            and isinstance(parent.get("step"), int)
        ):
            status_extra["lineage_base_step"] = int(parent["step"])
        elif is_restart:
            status_extra["lineage_base_step"] = 0
        else:
            status_extra["lineage_base_step"] = None
        self._write_status(state="initializing", force=True, **status_extra)
        self._write_event("run_start", max_steps=self.max_steps, restart_count=doc["restarts"])

    @staticmethod
    def _resolve_parent(
        resume_from: Optional[str], finetune_from: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        for kind, path in (("resume", resume_from), ("finetune", finetune_from)):
            if not path:
                continue
            return {
                "kind": kind,
                "path": str(path),
                "run_id": find_parent_run_id(path),
                "step": checkpoint_step_from_path(path),
            }
        return None

    # ----------------------------------------------------------------- write
    def _write_event(self, kind: str, **fields: Any) -> None:
        if not (self.enabled and self.is_metrics_writer and self._events is not None):
            return
        record = {"type": "event", "kind": str(kind), "ts": time.time()}
        record.update(sanitize(fields))
        self._events.write(record)

    def _write_status(self, state: Optional[str] = None, force: bool = False, **fields: Any) -> None:
        if not (self.enabled and self.is_metrics_writer and self.run_dir is not None):
            return
        now = time.time()
        if not force and now - self._last_status_write < self._status_interval:
            return
        self._last_status_write = now
        status = read_json(self.run_dir / "status.json", default={}) or {}
        status.update(sanitize(fields))
        status["ts"] = now
        status["state"] = state or status.get("state", "running")
        status["run_id"] = self.run_id
        try:
            atomic_write_json(self.run_dir / "status.json", status)
        except Exception as exc:
            self._warn_once("status", exc)

    def _warn_once(self, key: str, exc: Exception) -> None:
        if key not in self._failed_once:
            logger.warning("monitor: %s failed (%s)", key, exc)
            self._failed_once.add(key)

    def _guard(self, key: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            self._warn_once(key, exc)

    # -------------------------------------------------------------- public
    def log_step_record(self, record: Dict[str, Any]) -> None:
        """Append one per-step record (losses, LR, timings, source stats...)."""
        if not (self.enabled and self.is_metrics_writer and self._metrics is not None):
            return
        step = record.get("step", record.get("global_step"))
        payload = sanitize(record)
        payload["type"] = "step"
        payload["ts"] = time.time()
        with self._stage_lock:
            if self._stage_acc:
                payload["stage_seconds"] = dict(self._stage_acc)
                self._stage_acc.clear()
        self._guard("step", self._metrics.write, payload)
        self._last_step = step
        status_fields = {"last_step": step, "epoch": record.get("epoch")}
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            status_fields["last_total_loss"] = metrics.get("total_loss")
        self._write_status(**status_fields)

    def log_metrics(self, step: int, metrics: Dict[str, Any], **extra: Any) -> None:
        """Convenience wrapper for simple loops (micro training, eval)."""
        record = {"step": int(step), "metrics": dict(metrics)}
        record.update(extra)
        self.log_step_record(record)

    def log_slice_record(self, record: Dict[str, Any]) -> None:
        """Append one training-slice record; callable on **every** rank.

        Each rank appends to its own append-only file
        ``runs/<run_id>/slices/rank<global_rank:04d>.jsonl`` (one JSON object
        per line), so the server can aggregate slices per node/GPU. The caller
        supplies ``{step, samples:[{source, episode_id, window_start,
        frame_indices, ratio}]}``; this method stamps ``ts``/``node``/``gpu``/
        ``rank``. All errors are swallowed (logged once) — training must never
        be affected by monitor I/O.
        """
        if not self.enabled or self.run_dir is None:
            return
        payload = sanitize(record)
        payload["type"] = "slices"
        payload["ts"] = time.time()
        payload["node"] = self._node_label
        payload["gpu"] = self.local_rank
        payload["rank"] = self.rank

        def _write() -> None:
            if self._slices_fp is None:
                slices_dir = self.run_dir / "slices"
                slices_dir.mkdir(parents=True, exist_ok=True)
                path = slices_dir / f"rank{self.rank:04d}.jsonl"
                self._slices_fp = path.open("a", encoding="utf-8", buffering=1)
            self._slices_fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

        self._guard("slices", _write)

    def log_event(self, kind: str, **fields: Any) -> None:
        self._write_event(kind, **fields)

    def log_init_phase(self, name: str, seconds: Optional[float] = None, **extra: Any) -> None:
        """Record one completed initialization phase.

        Written into run.json's ``init_phases`` (always available to the UI,
        unlike the chunked events stream whose tail may drop old entries),
        mirrored into status.json (state stays "initializing" until the
        training loop starts) and into the events stream.
        """
        if not (self.enabled and self.is_metrics_writer and self.run_dir is not None):
            return
        entry: Dict[str, Any] = {"name": str(name), "ts": time.time()}
        if seconds is not None:
            entry["seconds"] = float(seconds)
        if extra:
            entry.update(sanitize(extra))

        def _update() -> None:
            run_json_path = self.run_dir / "run.json"
            doc = read_json(run_json_path, default={}) or {}
            phases = doc.get("init_phases") or []
            phases.append(entry)
            doc["init_phases"] = phases
            atomic_write_json(run_json_path, doc)

        self._guard("init_phase", _update)
        elapsed = None
        try:
            doc = read_json(self.run_dir / "run.json", default={}) or {}
            elapsed = sum(float(p.get("seconds", 0.0) or 0.0) for p in doc.get("init_phases") or [])
        except Exception:
            pass
        status_fields: Dict[str, Any] = {"init_phase": str(name)}
        if elapsed is not None:
            status_fields["init_elapsed_seconds"] = round(elapsed, 3)
        self._write_status(state="initializing", force=True, **status_fields)
        self._write_event("init_phase", name=str(name), seconds=seconds, **sanitize(extra))

    def log_checkpoint(self, step: int, path: object, timing: Optional[Dict[str, Any]] = None) -> None:
        """Record a finished checkpoint: event + entry in run.json's list."""
        if not (self.enabled and self.is_metrics_writer and self.run_dir is not None):
            return
        entry = {
            "step": int(step),
            "path": str(path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        def _update() -> None:
            run_json_path = self.run_dir / "run.json"
            doc = read_json(run_json_path, default={}) or {}
            ckpts = doc.get("checkpoints") or []
            if not any(int(c.get("step", -1)) == int(step) for c in ckpts if isinstance(c, dict)):
                ckpts.append(entry)
                ckpts.sort(key=lambda c: int(c.get("step", 0)))
            doc["checkpoints"] = ckpts
            atomic_write_json(run_json_path, doc)

        self._guard("checkpoint", _update)
        self._write_event("checkpoint", step=int(step), path=str(path), timing=timing or {})
        self._write_status(last_checkpoint_step=int(step), force=True)

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        self._write_status(max_steps=self.max_steps, force=True)

    def set_state(self, state: str, **fields: Any) -> None:
        if not (self.enabled and self.is_metrics_writer and self.run_dir is not None):
            return

        def _update() -> None:
            run_json_path = self.run_dir / "run.json"
            doc = read_json(run_json_path, default={}) or {}
            doc["state"] = str(state)
            if state in ("completed", "failed"):
                doc["ended_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(run_json_path, doc)

        self._guard("state", _update)
        self._write_status(state=state, force=True, **fields)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a code section; totals are folded into the next step record.

        Usage::

            with monitor.stage("action_branch"):
                ...
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._stage_lock:
                self._stage_acc[str(name)] = self._stage_acc.get(str(name), 0.0) + elapsed

    def close(self, state: Optional[str] = None) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        # Flush the final status so a short/finished run never loses its last
        # step to status-write throttling.
        if self.is_metrics_writer and getattr(self, "_last_step", None) is not None:
            self._write_status(last_step=self._last_step, force=True)
        self._write_event("run_closed")
        if state:
            self.set_state(state)
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=3.0)
        if self._slices_fp is not None:
            def _close_slices() -> None:
                self._slices_fp.close()
                self._slices_fp = None

            self._guard("close", _close_slices)
        for writer in (self._metrics, self._events, self._system):
            if writer is not None:
                self._guard("close", writer.close)


class NullMonitor(MonitorLogger):
    """Do-nothing stand-in with the full MonitorLogger API."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Bypass the real __init__ entirely: no directories, no threads.
        self.enabled = False
        self.run_id = None
        self.run_dir = None
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_metrics_writer = False
        self.is_system_writer = False
        self.max_steps = None
        self._closed = True


def _config_to_plain_dict(config: Any) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, dict):
        return config
    try:  # OmegaConf
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            converted = OmegaConf.to_container(config, resolve=False)
            return converted if isinstance(converted, dict) else None
    except Exception:
        pass
    try:
        return dict(config)
    except Exception:
        return None


def create_monitor(
    config: Any,
    run_name: str,
    checkpoint_dir: Optional[object] = None,
    max_steps: Optional[int] = None,
    rank: int = 0,
    local_rank: Optional[int] = None,
    world_size: int = 1,
    resume_from: Optional[str] = None,
    finetune_from: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> MonitorLogger:
    """Build a MonitorLogger from a training config's ``monitor:`` section.

    Resolution of the on/off switch (highest priority first):
    ``$MOTUS_MONITOR_ENABLED`` > ``config.monitor.enabled`` > False.
    """
    plain = _config_to_plain_dict(config) or {}
    section = plain.get("monitor") or {}
    if not isinstance(section, dict):
        section = {}

    env = os.environ.get(ENABLED_ENV, "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        enabled = True
    elif env in ("0", "false", "no", "off"):
        enabled = False
    else:
        enabled = bool(section.get("enabled", False))

    if not enabled:
        return NullMonitor()

    # All ranks must share one run id. Rank 0 resolves it (checkpoint-dir
    # marker for resume continuity, otherwise a fresh id) and broadcasts it
    # when torch.distributed is up; single-process runs just resolve locally.
    run_id: Optional[str] = None
    if int(rank) == 0:
        existing = read_run_id(checkpoint_dir) if checkpoint_dir else None
        run_id = existing or new_run_id(run_name)
    run_id = _broadcast_run_id(run_id)

    data_dir = section.get("data_dir") or None
    return MonitorLogger(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        config=plain,
        max_steps=max_steps,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        resume_from=resume_from,
        finetune_from=finetune_from,
        data_dir=data_dir,
        enabled=True,
        system_interval=float(section.get("system_interval", 2.0)),
        status_interval=float(section.get("status_interval", 2.0)),
        collect_system=bool(section.get("collect_system", True)),
        extra=extra,
        run_id=run_id,
    )


def _broadcast_run_id(run_id: Optional[str]) -> Optional[str]:
    """Share rank 0's run id with every rank when torch.distributed is live."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            payload = [run_id]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
    except Exception:
        pass
    return run_id
