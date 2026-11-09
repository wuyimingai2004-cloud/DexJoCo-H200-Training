"""Episode source reader plugin registry.

Third-party datasets register an ``EpisodeSourceReader`` subclass (or instance)
under the ``source`` name declared by their Catalog metadata, after which
``open_dataset`` / ``open_training_data`` / ``open_mixed_training_data`` resolve
the reader automatically without a per-call ``readers=`` mapping.

This module deliberately avoids importing ``motus_data.repository`` so that
``repository`` can depend on the registry without an import cycle; readers are
validated structurally (duck-typed) instead of by ``isinstance``.

Also hosts the per-source text composer registry (20260819): sources whose
Catalog text items are raw fields (egoverse/egoscale) register a composer that
joins the selected ``TextItem`` sequence (episode context items first, then the
window-overlapping segment items in temporal order) into the t5 input text;
batch adapters fall back to plain concatenation for unregistered sources.
Composers are duck-typed callables for the same no-cycle reason.
"""

from __future__ import annotations

import threading
from typing import Any

_READER_METHODS = ("schemas", "read_video", "read_state", "read_action")


class EpisodeSourceReaderRegistry:
    """Thread-safe registry of Episode source reader plugins."""

    def __init__(self) -> None:
        self._readers: dict[str, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(name: str) -> str:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("source reader name must be non-empty")
        return key

    @staticmethod
    def _validate(reader: Any) -> None:
        missing = [
            method
            for method in _READER_METHODS
            if not callable(getattr(reader, method, None))
        ]
        if missing:
            raise TypeError(
                "source reader must be an EpisodeSourceReader subclass or an "
                f"instance providing {missing[0]}()"
            )

    def register(self, name: str, reader: Any, *, replace: bool = False) -> None:
        key = self._key(name)
        with self._lock:
            if key in self._readers and not replace:
                raise KeyError(f"source reader is already registered: {key!r}")
            self._validate(reader)
            self._readers[key] = reader

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._readers))

    def snapshot(
        self,
        options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return reader instances, applying per-source constructor options."""

        with self._lock:
            registered = dict(self._readers)
        configured = dict(options or {})
        unknown = sorted(set(configured) - set(registered))
        if unknown:
            raise KeyError(
                f"reader options reference unregistered sources: {unknown}"
            )
        result = {}
        for name, reader in registered.items():
            kwargs = dict(configured.get(name) or {})
            if isinstance(reader, type):
                result[name] = reader(**kwargs)
            else:
                if kwargs:
                    raise TypeError(
                        f"reader options cannot configure registered instance {name!r}"
                    )
                result[name] = reader
        return result


EPISODE_SOURCE_READERS = EpisodeSourceReaderRegistry()


def register_episode_source_reader(
    name: str,
    reader: Any,
    *,
    replace: bool = False,
) -> None:
    """Register a source reader under the Catalog ``source`` name.

    ``reader`` may be an ``EpisodeSourceReader`` instance or subclass; classes
    are instantiated (zero-argument) per ``EpisodeRepository`` construction.
    """

    EPISODE_SOURCE_READERS.register(name, reader, replace=replace)


def episode_source_readers() -> tuple[str, ...]:
    """Return the sorted names of all registered source readers."""

    return EPISODE_SOURCE_READERS.names()


class TextComposerRegistry:
    """Thread-safe registry of per-source text composers.

    A composer receives the selected ``TextItem`` sequence of one sample and
    returns the composed t5 input text. ``None`` from :meth:`resolve` means the
    source keeps the adapter default (plain concatenation of item texts).
    """

    def __init__(self) -> None:
        self._composers: dict[str, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(name: str) -> str:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("text composer name must be non-empty")
        return key

    def register(self, name: str, composer: Any, *, replace: bool = False) -> None:
        key = self._key(name)
        if not callable(composer):
            raise TypeError("text composer must be callable")
        with self._lock:
            if key in self._composers and not replace:
                raise KeyError(f"text composer is already registered: {key!r}")
            self._composers[key] = composer

    def resolve(self, name: str) -> Any | None:
        key = self._key(name)
        with self._lock:
            return self._composers.get(key)


TEXT_COMPOSERS = TextComposerRegistry()


def register_text_composer(
    name: str,
    composer: Any,
    *,
    replace: bool = False,
) -> None:
    """Register a text composer under the Catalog ``source`` name.

    The composer is called with the selected ``TextItem`` sequence (episode
    context items first, then the window-overlapping segment items in temporal
    order) and must return the composed t5 input text for that sample.
    """

    TEXT_COMPOSERS.register(name, composer, replace=replace)


def resolve_text_composer(name: str) -> Any | None:
    """Return the composer registered under ``name`` (None when absent)."""

    return TEXT_COMPOSERS.resolve(name)
