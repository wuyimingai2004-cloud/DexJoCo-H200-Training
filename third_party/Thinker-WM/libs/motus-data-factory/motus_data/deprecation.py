"""Deprecation helpers for the legacy Episode-native (direct) pipeline."""

from __future__ import annotations

import warnings

_MIGRATION_HINT = (
    "请迁移到 Indexed（SamplePool）管线："
    "motus_data.open_dataset / open_training_data / open_mixed_training_data。"
)


def warn_episode_native_deprecated(api: str, *, stacklevel: int = 2) -> None:
    """Warn that an Episode-native (direct) pipeline API is deprecated.

    Uses ``FutureWarning`` so the warning is shown to end users by default.
    """
    warnings.warn(
        f"{api} 属于 Episode-native（直接）管线，该管线已弃用并将在后续版本中删除。"
        + _MIGRATION_HINT,
        FutureWarning,
        stacklevel=stacklevel,
    )
