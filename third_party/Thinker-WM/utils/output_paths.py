"""Environment-aware output root resolution for Motus training.

Outputs and checkpoints are written outside the repository checkout:

- Default candidate is the repository's parent directory (H cluster layout,
  where the checkout lives on shared storage and siblings like
  ``periodic_eval_cache`` already live one level up).
- If the parent directory is not readable/writable (e.g. a local checkout at
  the filesystem root), the fallback is ``/tmp/<repo_dir_name>`` — but this
  fallback **raises by default**; it is only taken when
  ``MOTUS_ALLOW_TMP_OUTPUT=1`` (smoke tests / local debug; ``train_micro.sh``
  sets it automatically).
- ``MOTUS_OUTPUT_ROOT`` overrides everything (highest priority).

Absolute paths are always passed through unchanged; only relative paths are
resolved against :func:`output_root`.
"""

import functools
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_OUTPUT_ROOT = "MOTUS_OUTPUT_ROOT"
_ENV_ALLOW_TMP = "MOTUS_ALLOW_TMP_OUTPUT"


def allow_tmp_output() -> bool:
    """Whether the /tmp fallback is permitted (smoke tests / local debug)."""
    return os.environ.get(_ENV_ALLOW_TMP, "0").lower() in ("1", "true", "yes")


def repo_root() -> Path:
    """Absolute path of the repository checkout (parent of ``utils/``)."""
    source_root = Path(__file__).resolve().parent.parent
    # Preserve the enclosing project output root after source vendoring.
    if source_root.parent.name == "third_party":
        return source_root.parent.parent
    return source_root


def _parent_writable(parent: Path) -> bool:
    """Probe read/write access to *parent* with a uniquely-named temp file.

    The probe uses ``tempfile.mkstemp`` (unique names) because fixed probe
    filenames race when multiple cluster nodes resolve concurrently.
    """
    try:
        if not os.access(parent, os.R_OK | os.W_OK):
            return False
        fd, probe = tempfile.mkstemp(dir=str(parent), prefix=".motus_output_probe_")
        os.close(fd)
        os.unlink(probe)
        return True
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def output_root() -> Path:
    """Root directory under which relative output/checkpoint paths resolve.

    Cached per process so the fallback warning/error is emitted only once.
    """
    override = os.environ.get(_ENV_OUTPUT_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    parent = repo_root().parent
    if _parent_writable(parent):
        return parent
    fallback = Path(tempfile.gettempdir()) / repo_root().name
    if not allow_tmp_output():
        raise RuntimeError(
            f"Output root: parent directory {parent} is not writable and the "
            f"/tmp fallback ({fallback}) is not allowed. Set {_ENV_ALLOW_TMP}=1 "
            f"to allow it (smoke tests / local debug only), or set "
            f"{_ENV_OUTPUT_ROOT} to a writable directory."
        )
    logger.warning(
        "Output root: parent directory %s is not writable; falling back to %s "
        "(%s=1, set %s to override)",
        parent,
        fallback,
        _ENV_ALLOW_TMP,
        _ENV_OUTPUT_ROOT,
    )
    return fallback


def resolve_output_path(path) -> str:
    """Resolve *path* against :func:`output_root` when it is relative.

    Empty and absolute paths are returned unchanged (as strings).
    """
    if not path:
        return path
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return str(p)
    return str(output_root() / p)
