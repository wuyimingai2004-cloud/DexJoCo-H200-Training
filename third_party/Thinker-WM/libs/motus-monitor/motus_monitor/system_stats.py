"""System load sampler for the Motus monitor (stdlib-only, graceful degradation).

Samples host CPU / memory / /dev/shm / disk / per-GPU utilization on a
background thread and appends one JSON record per interval to a
:class:`~monitor.store.ChunkedJsonlWriter`. Any individual probe may fail
(headless container, no nvidia-smi, non-Linux) -- the corresponding fields are
simply omitted and sampling continues.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _read_cpu_times() -> Optional[List[int]]:
    try:
        with open("/proc/stat", "r") as fp:
            for line in fp:
                if line.startswith("cpu "):
                    return [int(x) for x in line.split()[1:]]
    except (OSError, ValueError):
        pass
    return None


def _read_meminfo() -> Dict[str, int]:
    wanted = {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "Shmem"}
    values: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as fp:
            for line in fp:
                key, _, rest = line.partition(":")
                if key in wanted:
                    parts = rest.split()
                    if parts:
                        values[key] = int(parts[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _process_rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/statm", "r") as fp:
            parts = fp.read().split()
        if len(parts) >= 2:
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        pass
    return None


def _vfs_usage(path: object) -> Dict[str, int]:
    try:
        stat = os.statvfs(str(path))
    except OSError:
        return {}
    total = int(stat.f_blocks * stat.f_frsize)
    free = int(stat.f_bfree * stat.f_frsize)
    avail = int(stat.f_bavail * stat.f_frsize)
    used = max(0, total - free)
    return {
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": avail,
        "used_percent": round(100.0 * used / total, 2) if total else 0.0,
    }


def _query_gpus() -> List[Dict[str, Any]]:
    """Per-GPU stats via nvidia-smi; empty list when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    gpus: List[Dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "util_percent": float(parts[1]),
                    "mem_used_bytes": int(float(parts[2]) * 1024 * 1024),
                    "mem_total_bytes": int(float(parts[3]) * 1024 * 1024),
                    "temperature_c": float(parts[4]),
                    "power_watts": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return gpus


class SystemStatsSampler(threading.Thread):
    """Daemon thread appending ``{"type": "system", ...}`` records."""

    def __init__(
        self,
        writer: Any,
        interval: float = 2.0,
        data_dir: Optional[object] = None,
        node_label: Optional[str] = None,
    ):
        super().__init__(name="motus-monitor-system-stats", daemon=True)
        self.writer = writer
        self.interval = max(0.5, float(interval))
        self.data_dir = str(data_dir) if data_dir else None
        self.node_label = node_label or os.environ.get("HOSTNAME", "node")
        self._stop_event = threading.Event()
        self._prev_cpu: Optional[List[int]] = None
        self._failed_probes = set()

    def stop(self) -> None:
        self._stop_event.set()

    def _cpu_percent(self) -> Optional[float]:
        times = _read_cpu_times()
        if times is None:
            return None
        prev, self._prev_cpu = self._prev_cpu, times
        if prev is None or len(prev) != len(times):
            return None
        deltas = [b - a for a, b in zip(prev, times)]
        total = sum(deltas)
        if total <= 0:
            return None
        idle = deltas[3] + (deltas[4] if len(deltas) > 4 else 0)  # idle + iowait
        return round(100.0 * (total - idle) / total, 2)

    def sample(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "type": "system",
            "ts": time.time(),
            "node": self.node_label,
        }
        cpu = self._cpu_percent()
        if cpu is not None:
            record["cpu_percent"] = cpu
        try:
            load1, load5, load15 = os.getloadavg()
            record["loadavg"] = [round(load1, 2), round(load5, 2), round(load15, 2)]
            record["cpu_count"] = os.cpu_count()
        except OSError:
            pass
        mem = _read_meminfo()
        if mem:
            total = mem.get("MemTotal", 0)
            avail = mem.get("MemAvailable", 0)
            record["memory"] = {
                "total_bytes": total,
                "available_bytes": avail,
                "used_bytes": max(0, total - avail),
                "used_percent": round(100.0 * (total - avail) / total, 2) if total else 0.0,
                "shmem_bytes": mem.get("Shmem", 0),
            }
        rss = _process_rss_bytes()
        if rss is not None:
            record["process_rss_bytes"] = rss
        shm = _vfs_usage("/dev/shm")
        if shm:
            record["dev_shm"] = shm
        if self.data_dir:
            disk = _vfs_usage(self.data_dir)
            if disk:
                record["data_dir_disk"] = disk
        try:
            gpus = _query_gpus()
            if gpus:
                record["gpus"] = gpus
        except Exception as exc:  # never let probing kill the sampler
            if "gpu" not in self._failed_probes:
                logger.debug("monitor gpu probe failed: %s", exc)
                self._failed_probes.add("gpu")
        return record

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.writer.write(self.sample())
            except Exception as exc:
                if "write" not in self._failed_probes:
                    logger.warning("monitor system-stats write failed: %s", exc)
                    self._failed_probes.add("write")
            self._stop_event.wait(self.interval)
