"""Versioned Episode Catalog releases used by the public data APIs.

Built-in defaults (Catalog root and registered releases) are declared in the
``releases.yaml`` file shipped next to this module, not hardcoded in Python.
The YAML travels with the package for both pip installs and direct source
copies. Runtime configuration is passed explicitly via API arguments; this
module never reads environment variables.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


RELEASES_CONFIG_PATH = Path(__file__).with_name("releases.yaml")


def _load_releases_config() -> Mapping[str, Any]:
    if not RELEASES_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"releases config is missing: {RELEASES_CONFIG_PATH}"
        )
    config = OmegaConf.to_container(OmegaConf.load(RELEASES_CONFIG_PATH), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"releases config must be a mapping: {RELEASES_CONFIG_PATH}")
    root = str(config.get("default_catalog_root") or "").strip()
    if not root:
        raise ValueError("releases config requires default_catalog_root")
    releases = config.get("releases")
    if not isinstance(releases, list) or not releases:
        raise ValueError("releases config requires a non-empty releases list")
    return config


_RELEASES_CONFIG = _load_releases_config()

DEFAULT_EPISODE_CATALOG_ROOT = Path(
    str(_RELEASES_CONFIG["default_catalog_root"]).strip()
)


def _name(value: str, field: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


@dataclass(frozen=True)
class EpisodeDatasetRelease:
    """Immutable paths and accepted runtime defaults for one dataset release."""

    dataset: str
    release: str
    catalog_path: str
    index_path: str | None = None
    default_video_size: tuple[int, int] | None = None
    default_video_output_format: str = "rgb_float32_chw"
    default_record_cache_size: int = 256
    default_num_workers: int = 0
    default_pin_memory: bool = False
    default_queue_size: int = 2
    default_queue_timeout_seconds: float = 600.0
    action_magnitude_guard_threshold: float | None = 100.0
    action_magnitude_guard_sources: tuple[str, ...] | None = None
    reader_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _name(self.dataset, "dataset"))
        object.__setattr__(self, "release", _name(self.release, "release"))
        if not str(self.catalog_path).strip():
            raise ValueError("catalog_path must be non-empty")
        if self.index_path is not None and not str(self.index_path).strip():
            raise ValueError("index_path must be None or non-empty")
        if self.default_video_size is not None:
            size = tuple(int(value) for value in self.default_video_size)
            if len(size) != 2 or any(value <= 0 for value in size):
                raise ValueError("default_video_size must contain positive H,W")
            object.__setattr__(self, "default_video_size", size)
        if self.default_video_output_format not in {
            "rgb_uint8_hwc",
            "rgb_float32_chw",
        }:
            raise ValueError("unsupported default_video_output_format")
        if int(self.default_record_cache_size) < 0:
            raise ValueError("default_record_cache_size must be >= 0")
        if int(self.default_num_workers) < 0:
            raise ValueError("default_num_workers must be >= 0")
        if int(self.default_queue_size) <= 0:
            raise ValueError("default_queue_size must be positive")
        if float(self.default_queue_timeout_seconds) <= 0:
            raise ValueError("default_queue_timeout_seconds must be positive")
        if (
            self.action_magnitude_guard_threshold is not None
            and float(self.action_magnitude_guard_threshold) <= 0
        ):
            raise ValueError(
                "action_magnitude_guard_threshold must be positive or None"
            )
        sources = self.action_magnitude_guard_sources
        if sources is not None:
            parsed = tuple(str(value).strip() for value in sources)
            if not parsed or any(not value for value in parsed):
                raise ValueError(
                    "action_magnitude_guard_sources must be None or non-empty"
                )
            object.__setattr__(self, "action_magnitude_guard_sources", parsed)
        reader_options = self.reader_options
        if reader_options is not None:
            if not isinstance(reader_options, Mapping):
                raise ValueError("reader_options must be a mapping or None")
            normalized_options = {}
            for source, raw_options in reader_options.items():
                source_name = _name(str(source), "reader_options source")
                if not isinstance(raw_options, Mapping):
                    raise ValueError(
                        f"reader_options.{source_name} must be a mapping"
                    )
                normalized_options[source_name] = dict(raw_options)
            object.__setattr__(self, "reader_options", normalized_options)

    @property
    def release_id(self) -> str:
        return f"{self.dataset}:{self.release}"

    @staticmethod
    def _resolve_path(value: str, root: Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    def resolve_paths(
        self,
        catalog_root: str | Path | None = None,
    ) -> tuple[Path, Path | None]:
        root = (
            Path(catalog_root)
            if catalog_root is not None
            else DEFAULT_EPISODE_CATALOG_ROOT
        )
        return (
            self._resolve_path(self.catalog_path, root),
            (
                None
                if self.index_path is None
                else self._resolve_path(self.index_path, root)
            ),
        )


class EpisodeDatasetReleaseRegistry:
    """Thread-safe registry of immutable dataset releases."""

    def __init__(self) -> None:
        self._releases: dict[tuple[str, str], EpisodeDatasetRelease] = {}
        self._defaults: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(
        self,
        release: EpisodeDatasetRelease,
        *,
        default: bool = False,
        replace: bool = False,
    ) -> None:
        if not isinstance(release, EpisodeDatasetRelease):
            raise TypeError("release must be an EpisodeDatasetRelease")
        key = (release.dataset, release.release)
        with self._lock:
            if key in self._releases and not replace:
                raise KeyError(
                    f"dataset release is already registered: {release.release_id}"
                )
            self._releases[key] = release
            if default or release.dataset not in self._defaults:
                self._defaults[release.dataset] = release.release

    def get(
        self,
        dataset: str,
        release: str | EpisodeDatasetRelease | None = None,
    ) -> EpisodeDatasetRelease:
        dataset_name = _name(dataset, "dataset")
        if isinstance(release, EpisodeDatasetRelease):
            if release.dataset != dataset_name:
                raise ValueError(
                    f"release dataset={release.dataset!r} does not match "
                    f"requested dataset={dataset_name!r}"
                )
            return release
        with self._lock:
            release_name = (
                self._defaults.get(dataset_name)
                if release is None
                else _name(release, "release")
            )
            if release_name is None:
                raise KeyError(
                    f"dataset has no default release: {dataset_name!r}"
                )
            try:
                return self._releases[(dataset_name, release_name)]
            except KeyError as error:
                available = sorted(
                    name
                    for registered_dataset, name in self._releases
                    if registered_dataset == dataset_name
                )
                raise KeyError(
                    f"unknown release={release_name!r} for "
                    f"dataset={dataset_name!r}; available={available}"
                ) from error

    def names(self, dataset: str | None = None) -> tuple[str, ...]:
        with self._lock:
            values: Sequence[EpisodeDatasetRelease] = tuple(
                self._releases.values()
            )
        if dataset is not None:
            dataset_name = _name(dataset, "dataset")
            values = tuple(
                value for value in values if value.dataset == dataset_name
            )
        return tuple(sorted(value.release_id for value in values))


EPISODE_DATASET_RELEASES = EpisodeDatasetReleaseRegistry()


def register_episode_dataset_release(
    release: EpisodeDatasetRelease,
    *,
    default: bool = False,
    replace: bool = False,
) -> None:
    EPISODE_DATASET_RELEASES.register(
        release,
        default=default,
        replace=replace,
    )


def get_episode_dataset_release(
    dataset: str,
    release: str | EpisodeDatasetRelease | None = None,
) -> EpisodeDatasetRelease:
    return EPISODE_DATASET_RELEASES.get(dataset, release)


def episode_dataset_releases(dataset: str | None = None) -> tuple[str, ...]:
    return EPISODE_DATASET_RELEASES.names(dataset)


def _release_from_config(entry: Mapping[str, Any]) -> EpisodeDatasetRelease:
    known = {
        "dataset",
        "release",
        "catalog_path",
        "index_path",
        "default_video_size",
        "default_video_output_format",
        "default_record_cache_size",
        "default_num_workers",
        "default_pin_memory",
        "default_queue_size",
        "default_queue_timeout_seconds",
        "action_magnitude_guard_threshold",
        "action_magnitude_guard_sources",
        "reader_options",
        "default",
    }
    unknown = set(entry) - known
    if unknown:
        raise ValueError(
            f"unknown release config fields {sorted(unknown)} for "
            f"dataset={entry.get('dataset')!r}"
        )
    kwargs = {key: value for key, value in entry.items() if key != "default"}
    return EpisodeDatasetRelease(**kwargs)


for _entry in _RELEASES_CONFIG["releases"]:
    register_episode_dataset_release(
        _release_from_config(_entry),
        default=bool(_entry.get("default", False)),
    )

del _entry
