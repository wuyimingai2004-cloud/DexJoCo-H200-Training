"""Data-directory resolution and run-id helpers for the Motus monitor.

The monitor data root lives OUTSIDE the repository (by default in the
repository's parent directory) so that training artifacts never pollute the
git tree and so the sync daemon can treat it as a self-contained unit:

    <repo_parent>/motus_monitor_data/

Resolution order:
    1. $MOTUS_MONITOR_DIR (explicit override, always wins)
    2. <repo_parent>/motus_monitor_data (creatable by the current user)
    3. ~/.local/share/motus_monitor_data (fallback, e.g. repo mounted at /)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR_ENV = "MOTUS_MONITOR_DIR"
DATA_DIR_NAME = "motus_monitor_data"
RUN_ID_MARKER = ".motus_monitor_run_id"


def repo_root() -> Path:
    """Repository root (the checkout that contains this ``motus_monitor`` package).

    The package lives at ``libs/motus-monitor/motus_monitor/`` inside the
    thinkerwm-pretrain repo, so a fixed ``parents[N]`` is fragile (it breaks
    under a non-editable pip install). Walk upwards looking for a repo marker
    instead (``pixi.toml`` — synced to the cluster — or ``.git``), falling
    back to the in-repo layout depth when no marker is found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pixi.toml").exists() or (parent / ".git").exists():
            return parent
    # In-repo layout: libs/motus-monitor/motus_monitor/paths.py -> repo root.
    return here.parents[3]


def _is_creatable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        # NOTE: the probe file MUST be unique per process. A fixed probe name
        # races on shared filesystems when several nodes probe at the same
        # time (one unlinks while others are mid-probe -> FileNotFoundError),
        # which used to make all-but-one node fall back to a pod-local dir.
        fd, probe = tempfile.mkstemp(prefix=".write_probe.", dir=str(path))
        os.close(fd)
        os.unlink(probe)
        return True
    except OSError as exc:
        logger.warning("monitor: data dir probe failed for %s: %s", path, exc)
        return False


def default_data_dir(create: bool = True) -> Path:
    """Resolve the monitor data root; see module docstring for the order."""
    env = os.environ.get(DATA_DIR_ENV, "").strip()
    if env:
        path = Path(env).expanduser()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    primary = repo_root().parent / DATA_DIR_NAME
    if not create:
        return primary
    if _is_creatable(primary):
        return primary

    fallback = Path.home() / ".local" / "share" / DATA_DIR_NAME
    if create:
        fallback.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "monitor: primary data dir %s not writable; falling back to pod-local %s "
        "(this node's data will NOT be synced!)",
        primary,
        fallback,
    )
    return fallback


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_name(name: object) -> str:
    """Make an arbitrary string safe as a single path component."""
    cleaned = _SAFE_NAME.sub("-", str(name)).strip("-.")
    return cleaned[:120] or "run"


def new_run_id(run_name: str) -> str:
    """Fresh run id: sanitized run name plus a start timestamp."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_name(run_name)}__{stamp}"


def read_run_id(checkpoint_dir: object) -> Optional[str]:
    """Return the run id previously recorded inside a checkpoint directory."""
    try:
        marker = Path(str(checkpoint_dir)) / RUN_ID_MARKER
        if marker.is_file():
            value = marker.read_text().strip()
            return value or None
    except OSError:
        pass
    return None


def write_run_id(checkpoint_dir: object, run_id: str) -> None:
    """Persist the run id inside the checkpoint directory.

    Elastic restarts / auto-resume reuse the same checkpoint directory, so the
    marker lets the next process continue the SAME monitor run instead of
    opening a new one.
    """
    try:
        path = Path(str(checkpoint_dir))
        path.mkdir(parents=True, exist_ok=True)
        (path / RUN_ID_MARKER).write_text(str(run_id) + "\n")
    except OSError:
        pass


def find_parent_run_id(checkpoint_path: object, max_depth: int = 4) -> Optional[str]:
    """Walk upwards from a resume/finetune checkpoint path looking for the
    run-id marker of the run that produced it.

    Layout convention: <ckpt_root>/<dataset>/<run_name>/checkpoint_step_N, so
    the marker sits one or two levels above the checkpoint directory itself.
    """
    try:
        current = Path(str(checkpoint_path)).resolve()
        if current.is_file():
            current = current.parent
        for _ in range(max_depth + 1):
            found = read_run_id(current)
            if found:
                return found
            parent = current.parent
            if parent == current:
                break
            current = parent
    except OSError:
        pass
    return None


def checkpoint_step_from_path(checkpoint_path: object) -> Optional[int]:
    """Extract the step number N from a ``checkpoint_step_N`` path."""
    match = re.search(r"checkpoint_step_(\d+)", str(checkpoint_path))
    return int(match.group(1)) if match else None


def annotation_key(run_id: str, step: object) -> str:
    """File-safe key identifying one checkpoint of one run."""
    return f"{sanitize_name(run_id)}__step_{int(step)}"
