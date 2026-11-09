"""Let torchrun terminate peers promptly when a CUDA context is unusable."""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys


def exit_on_fatal_cuda_error(error: BaseException) -> None:
    """Never attempt NCCL teardown after a fatal CUDA execution error.

    The caller must log the traceback and mark its monitor failed first.
    torchrun observes this nonzero exit and terminates the other workers.
    OOM, data errors, and ordinary Python exceptions keep normal cleanup.
    """
    message = str(error).lower()
    fatal = any(token in message for token in (
        "an illegal memory access was encountered",
        "device-side assert triggered",
        "cuda error: misaligned address",
        "cuda error: unspecified launch failure",
    ))
    if not fatal:
        return
    logging.getLogger(__name__).critical(
        "Fatal CUDA execution error: exiting worker without NCCL teardown; "
        "torchrun must terminate peers. The failed CUDA context cannot save a checkpoint."
    )
    # Forked DataLoader workers can inherit CUDA driver file descriptors.
    # Leaving them orphaned keeps the dead parent's GPU allocations alive.
    # Terminate only this worker's own multiprocessing children, without CUDA.
    children = multiprocessing.active_children()
    for child in children:
        try:
            child.terminate()
        except Exception:
            pass
    for child in children:
        try:
            child.join(timeout=0.25)
            if child.is_alive():
                child.kill()
                child.join(timeout=0.25)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(1)
