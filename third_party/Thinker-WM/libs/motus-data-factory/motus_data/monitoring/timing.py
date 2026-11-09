import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def get_dataset_timing_threshold() -> float:
    return float(os.getenv("MOTUS_DATASET_TIMING_THRESHOLD", "5.0"))


def dataset_timing_enabled() -> bool:
    return os.getenv("MOTUS_DATASET_TIMING_ENABLED", "1").lower() not in {"0", "false", "no"}


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def write_dataset_timing(source: str, record: Dict[str, Any], threshold_seconds: float = None):
    """Write one slow dataset sample timing record as JSONL.

    The output directory defaults to ``./dataset_timing`` relative to the current
    training working directory. The file is append-only and includes PID/rank so
    dataloader workers can write independently.
    """
    if not dataset_timing_enabled():
        return

    total_seconds = float(record.get("total_seconds", 0.0))
    if threshold_seconds is None:
        threshold_seconds = get_dataset_timing_threshold()
    if total_seconds < float(threshold_seconds):
        return

    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    worker = "main"
    try:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is not None:
            worker = str(info.id)
    except Exception:
        pass

    out_dir = Path(os.getenv("MOTUS_DATASET_TIMING_DIR", "./dataset_timing"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}_rank{rank}_worker{worker}_pid{os.getpid()}.jsonl"
    payload = {
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "source_dataset": source,
        "rank": rank,
        "local_rank": local_rank,
        "worker": worker,
        "pid": os.getpid(),
        **record,
    }
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
