"""Motus training-side logging and timing API.

Public training subset: remote synchronization, servers and administration are omitted.
"""

from .client import MonitorLogger, NullMonitor, create_monitor
from .paths import default_data_dir

__version__ = "0.1.0"

__all__ = [
    "MonitorLogger",
    "NullMonitor",
    "create_monitor",
    "default_data_dir",
    "__version__",
]
