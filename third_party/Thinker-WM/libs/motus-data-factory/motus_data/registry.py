"""Dataset plugin registry."""

from __future__ import annotations

from typing import Callable, Dict, Iterable


DatasetBuilder = Callable[..., object]


class DatasetRegistry:
    def __init__(self) -> None:
        self._builders: Dict[str, DatasetBuilder] = {}

    def register(self, name: str, builder: DatasetBuilder, *, replace: bool = False) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("dataset type cannot be empty")
        if key in self._builders and not replace:
            raise KeyError(f"dataset type already registered: {key}")
        self._builders[key] = builder

    def build(self, name: str, **kwargs):
        key = str(name).strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown dataset type {key!r}; registered={sorted(self._builders)}"
            ) from exc
        return builder(**kwargs)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._builders))


DATASETS = DatasetRegistry()


def register_dataset(name: str, *, replace: bool = False):
    def decorator(builder: DatasetBuilder):
        DATASETS.register(name, builder, replace=replace)
        return builder
    return decorator

