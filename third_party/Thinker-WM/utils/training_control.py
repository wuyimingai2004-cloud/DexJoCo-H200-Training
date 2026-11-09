"""Coordinated stop requests and wall-clock training step summaries."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist


STOP_REQUEST = ".stop_requested"
STOP_RECEIPT = "stop_completed.json"
CHECKPOINT_MARKER = ".motus_checkpoint_complete.json"


class TrainingStopRequest:
    """Poll only at a boundary reached by every rank, never in a signal handler.

    The request lives in the shared run checkpoint directory. Rank 0 checks
    it and broadcasts the decision so filesystem visibility cannot split ranks.
    A failed save leaves the request in place for diagnosis/retry.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def requested(self, device: Any) -> bool:
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        # A filesystem failure must reach every rank, rather than stranding
        # peers in a collective while rank 0 raises locally.
        value = 0
        if rank == 0:
            try:
                (self.run_dir / STOP_REQUEST).stat()
                value = 1
            except FileNotFoundError:
                pass
            except OSError:
                value = -1
        if distributed:
            collective_device = device if dist.get_backend() == "nccl" else "cpu"
            flag = torch.tensor([value], dtype=torch.int32, device=collective_device)
            dist.broadcast(flag, src=0)
            value = int(flag.item())
        if value < 0:
            raise RuntimeError(f"Cannot inspect training stop request in {self.run_dir}")
        return value == 1

    def complete(self, step: int, checkpoint_dir: Path) -> None:
        """Acknowledge only a complete checkpoint; leave the request on error."""
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        error = None
        if rank == 0:
            try:
                checkpoint_dir = Path(checkpoint_dir)
                marker = json.loads((checkpoint_dir / CHECKPOINT_MARKER).read_text())
                if marker.get("status") != "complete" or int(marker["step"]) != int(step):
                    raise RuntimeError("checkpoint completion marker does not match stop step")
                receipt = self.run_dir / STOP_RECEIPT
                temporary = receipt.with_name(receipt.name + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps({
                    "status": "checkpoint_saved", "step": int(step),
                    "checkpoint_dir": str(checkpoint_dir),
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }, indent=2) + "\n")
                os.replace(temporary, receipt)
                (self.run_dir / STOP_REQUEST).unlink(missing_ok=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if distributed:
            payload = [error]
            dist.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(f"Failed to acknowledge saved training stop: {error}")


def step_timing_breakdown(record: Dict[str, Any]) -> Dict[str, float]:
    """Non-overlapping host wall times, measured before emitting this summary.

    dataloader_next_seconds already INCLUDES prepare_train_batch_seconds.
    Update includes forward/backward/optimizer and metric synchronization, not
    just optimizer.step() (a no-op wrapper with DeepSpeed). Other includes GC,
    control synchronization and unclassified bookkeeping. These are wall times,
    not individually CUDA-synchronized kernel timings.
    """
    prepare = float(record.get("prepare_train_batch_seconds", 0.0))
    result = {
        "data_seconds": max(0.0, float(record.get("dataloader_next_seconds", 0.0)) - prepare),
        "prepare_seconds": prepare,
        "update_seconds": float(record.get("step_time_seconds", 0.0)),
        "validation_seconds": float(record.get("validation_seconds", 0.0)),
        "periodic_eval_seconds": float(record.get("periodic_eval_seconds", 0.0)),
        "checkpoint_seconds": float(record.get("checkpoint_total_seconds", 0.0)),
    }
    total = float(record["loop_total_seconds"])
    result["other_seconds"] = max(0.0, total - sum(result.values()))
    result["total_seconds"] = total
    return result


def format_step_timing(timing: Dict[str, float]) -> str:
    return ", ".join(
        f"{label}: {timing[key]:.3f}s"
        for label, key in (
            ("Total", "total_seconds"), ("Data", "data_seconds"),
            ("Prepare", "prepare_seconds"), ("Update", "update_seconds"),
            ("Validation", "validation_seconds"), ("Checkpoint", "checkpoint_seconds"),
            ("PeriodicEval", "periodic_eval_seconds"), ("Other", "other_seconds"),
        )
    )
