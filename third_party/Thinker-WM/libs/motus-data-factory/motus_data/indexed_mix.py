"""Weighted mixing for independent Catalog/SamplePool training sources."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from motus_data.loading import ReadyBatchQueue
from motus_data.releases import EpisodeDatasetRelease
from motus_data.sample_pool import SupportsSamplePool


_BATCH_KEY = "__motus_indexed_mix_batch__"
_STATS_KEY = "__motus_indexed_mix_stats__"

CROSS_RANK_SOURCE_POLICY_SYNCHRONIZED = "synchronized"
CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP = "weighted_per_step"
CROSS_RANK_SOURCE_POLICIES = (
    CROSS_RANK_SOURCE_POLICY_SYNCHRONIZED,
    CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP,
)


def _transform_envelope(
    envelope: Any,
    batch_transform: Callable[[Any], Any],
) -> dict[str, Any]:
    """Apply a repo-provided CPU batch transform without changing mix metadata."""

    if (
        not isinstance(envelope, Mapping)
        or _BATCH_KEY not in envelope
        or _STATS_KEY not in envelope
    ):
        raise RuntimeError("indexed mix stream produced an invalid envelope")
    started = time.perf_counter()
    transformed = dict(envelope)
    transformed[_BATCH_KEY] = batch_transform(envelope[_BATCH_KEY])
    stats = dict(envelope[_STATS_KEY])
    stats["mix_batch_transform_seconds"] = time.perf_counter() - started
    transformed[_STATS_KEY] = stats
    return transformed


def _non_empty_name(value: Any, field_name: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{field_name} must be non-empty")
    return name


@dataclass(frozen=True)
class IndexedTrainingSource:
    """One independently versioned child in an indexed training mixture."""

    name: str
    dataset: str
    sample_pool: SupportsSamplePool | str | Path | Callable[..., SupportsSamplePool]
    weight: float | None = None
    family: str | None = None
    release: str | EpisodeDatasetRelease | None = None
    catalog_root: str | Path | None = None
    catalog_path: str | Path | None = None
    index_path: str | Path | None = None
    use_index: bool = True
    verify_sha256: bool | None = None
    record_cache_size: int | None = None
    readers: Mapping[str, Any] | None = None
    reader_options: Mapping[str, Mapping[str, Any]] | None = None
    output_adapter: Any = None
    adapter_options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_name(self.name, "name"))
        object.__setattr__(
            self, "dataset", _non_empty_name(self.dataset, "dataset").lower()
        )
        if isinstance(self.sample_pool, (str, Path)):
            if not str(self.sample_pool).strip():
                raise ValueError("sample_pool must be non-empty")
        elif not (
            isinstance(self.sample_pool, SupportsSamplePool)
            or callable(self.sample_pool)
        ):
            raise TypeError(
                "sample_pool must be a path, a SupportsSamplePool instance, "
                "or a pool factory callable"
            )
        if self.weight is not None:
            weight = float(self.weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("weight must be finite and positive")
            object.__setattr__(self, "weight", weight)
        family = self.dataset if self.family is None else self.family
        object.__setattr__(
            self, "family", _non_empty_name(family, "family").lower()
        )

    @classmethod
    def from_value(
        cls, value: "IndexedTrainingSource | Mapping[str, Any]"
    ) -> "IndexedTrainingSource":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "indexed mix sources must be IndexedTrainingSource or mappings"
            )
        return cls(**dict(value))


class WeightedIndexedBatchStream:
    """Finite weighted stream with an explicit cross-rank source policy."""

    def __init__(
        self,
        children: Sequence[Any],
        sources: Sequence[IndexedTrainingSource],
        *,
        batches_per_epoch: int,
        seed: int,
        shuffle: bool,
        rank: int = 0,
        world_size: int = 1,
        cross_rank_source_policy: str = CROSS_RANK_SOURCE_POLICY_SYNCHRONIZED,
    ) -> None:
        if len(children) != len(sources) or not children:
            raise ValueError("children and sources must have the same non-zero length")
        if isinstance(batches_per_epoch, bool) or int(batches_per_epoch) <= 0:
            raise ValueError("batches_per_epoch must be positive")
        names = [source.name for source in sources]
        if len(names) != len(set(names)):
            raise ValueError(f"indexed mix source names must be unique: {names}")
        if isinstance(world_size, bool) or int(world_size) <= 0:
            raise ValueError("world_size must be positive")
        if isinstance(rank, bool) or not 0 <= int(rank) < int(world_size):
            raise ValueError(
                f"rank must be within [0,{int(world_size)}), got {rank}"
            )
        policy = str(cross_rank_source_policy).strip().lower()
        if policy not in CROSS_RANK_SOURCE_POLICIES:
            raise ValueError(
                f"unknown cross_rank_source_policy={policy!r}; "
                f"expected one of {CROSS_RANK_SOURCE_POLICIES}"
            )
        self.children = tuple(children)
        self.sources = tuple(sources)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cross_rank_source_policy = policy
        self.epoch = 0
        # ReadyBatchQueue discovers set_epoch through dataloader.batch_sampler.
        self.batch_sampler = self

    def __len__(self) -> int:
        return self.batches_per_epoch

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        weights = tuple(float(source.weight) for source in self.sources)
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def source_quotas(self) -> tuple[int, ...]:
        """Largest-remainder quotas keep each finite epoch close to exact weights."""

        expected = [
            weight * self.batches_per_epoch
            for weight in self.normalized_weights
        ]
        quotas = [math.floor(value) for value in expected]
        remaining = self.batches_per_epoch - sum(quotas)
        order = sorted(
            range(len(expected)),
            key=lambda index: (
                -(expected[index] - quotas[index]),
                self.sources[index].name,
            ),
        )
        for index in order[:remaining]:
            quotas[index] += 1
        return tuple(quotas)

    def _synchronized_source_schedule(self) -> tuple[int, ...]:
        schedule = [
            source_index
            for source_index, count in enumerate(self.source_quotas())
            for _ in range(count)
        ]
        if self.shuffle:
            random.Random(self.seed + self.epoch * 1_000_003).shuffle(schedule)
        return tuple(schedule)

    def _iter_weighted_per_step_source_schedule(self) -> Iterator[int]:
        """Assign this rank from a deterministic weighted global rank row.

        Every data step contains ``world_size`` integer rank slots. A source gets
        either ``floor(weight * world_size)`` or ``ceil(...)`` slots. Fractional
        slots are carried across steps, so low-weight sources appear at the
        correct long-run frequency without rank-to-rank communication.
        """

        weights = self.normalized_weights
        exact_per_step = [weight * self.world_size for weight in weights]
        base_counts = [math.floor(value) for value in exact_per_step]
        extra_slots = self.world_size - sum(base_counts)
        fractions = [
            value - base for value, base in zip(exact_per_step, base_counts)
        ]
        credits = [0.0 for _ in self.sources]

        rng = random.Random(self.seed + self.epoch * 1_000_003)
        tie_order = list(range(len(self.sources)))
        if self.shuffle:
            rng.shuffle(tie_order)
        tie_priority = {
            source_index: priority
            for priority, source_index in enumerate(tie_order)
        }

        for step in range(self.batches_per_epoch):
            for source_index, fraction in enumerate(fractions):
                credits[source_index] += fraction
            selected = sorted(
                range(len(self.sources)),
                key=lambda source_index: (
                    -credits[source_index],
                    tie_priority[source_index],
                    self.sources[source_index].name,
                ),
            )[:extra_slots]
            counts = list(base_counts)
            for source_index in selected:
                counts[source_index] += 1
                credits[source_index] -= 1.0

            rank_sources = [
                source_index
                for source_index, count in enumerate(counts)
                for _ in range(count)
            ]
            if len(rank_sources) != self.world_size:
                raise RuntimeError(
                    "weighted_per_step produced an invalid global rank row: "
                    f"step={step} slots={len(rank_sources)} "
                    f"world_size={self.world_size}"
                )
            if self.shuffle:
                rng.shuffle(rank_sources)
            elif self.world_size > 1:
                offset = (step + self.epoch) % self.world_size
                rank_sources = rank_sources[offset:] + rank_sources[:offset]
            yield rank_sources[self.rank]

    def _weighted_per_step_source_schedule(self) -> tuple[int, ...]:
        return tuple(self._iter_weighted_per_step_source_schedule())

    def iter_source_schedule(self) -> Iterator[int]:
        if (
            self.cross_rank_source_policy
            == CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP
        ):
            yield from self._iter_weighted_per_step_source_schedule()
            return
        yield from self._synchronized_source_schedule()

    def source_schedule(self) -> tuple[int, ...]:
        if (
            self.cross_rank_source_policy
            == CROSS_RANK_SOURCE_POLICY_WEIGHTED_PER_STEP
        ):
            return self._weighted_per_step_source_schedule()
        return self._synchronized_source_schedule()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for child in self.children:
            child.set_epoch(self.epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for step, source_index in enumerate(self.iter_source_schedule()):
            source = self.sources[source_index]
            batch, child_stats = self.children[source_index].next()
            yield {
                _BATCH_KEY: batch,
                _STATS_KEY: {
                    "mix_epoch": int(self.epoch),
                    "mix_step_in_epoch": int(step),
                    "mix_source_index": int(source_index),
                    "mix_source_name": source.name,
                    "mix_source_dataset": source.dataset,
                    "mix_source_family": source.family,
                    "mix_source_weight": float(source.weight),
                    "mix_source_normalized_weight": float(
                        self.normalized_weights[source_index]
                    ),
                    "mix_cross_rank_source_policy": (
                        self.cross_rank_source_policy
                    ),
                    "mix_rank": int(self.rank),
                    "mix_world_size": int(self.world_size),
                    "mix_child_stats": child_stats,
                },
            }


@dataclass
class IndexedMixedTrainingDataPipeline:
    """Owned lifecycle for weighted, child-homogeneous indexed batches."""

    sources: tuple[IndexedTrainingSource, ...]
    children: tuple[Any, ...]
    stream: WeightedIndexedBatchStream
    queue: ReadyBatchQueue | None
    rank: int
    world_size: int
    queue_close_timeout_seconds: float
    batch_transform: Callable[[Any], Any] | None = None
    _iterator: Any = None
    _epoch: int = 0
    _closed: bool = False
    _consumed_by_source: Counter[str] = field(default_factory=Counter)

    @property
    def dataloader(self) -> WeightedIndexedBatchStream:
        """Trainer compatibility: the stream owns child DataLoaders."""

        return self.stream

    @property
    def batch_sampler(self) -> WeightedIndexedBatchStream:
        return self.stream

    @property
    def queue_enabled(self) -> bool:
        return self.queue is not None

    def start(self) -> "IndexedMixedTrainingDataPipeline":
        if self._closed:
            raise RuntimeError("mixed training data pipeline is closed")
        if self.queue is not None:
            self.queue.start()
        elif self._iterator is None:
            self._iterator = iter(self.stream)
        return self

    @staticmethod
    def _unwrap(envelope: Any) -> tuple[Any, dict[str, Any]]:
        if (
            not isinstance(envelope, Mapping)
            or _BATCH_KEY not in envelope
            or _STATS_KEY not in envelope
        ):
            raise RuntimeError("indexed mix stream produced an invalid envelope")
        return envelope[_BATCH_KEY], dict(envelope[_STATS_KEY])

    def next(self):
        self.start()
        if self.queue is not None:
            envelope, stats = self.queue.next()
        else:
            try:
                envelope = next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self.stream.set_epoch(self._epoch)
                self._iterator = iter(self.stream)
                envelope = next(self._iterator)
            if self.batch_transform is not None:
                envelope = _transform_envelope(envelope, self.batch_transform)
            stats = {
                "ready_queue_enabled": False,
                "data_epoch": int(self._epoch),
            }
        batch, mix_stats = self._unwrap(envelope)
        source_name = str(mix_stats["mix_source_name"])
        self._consumed_by_source[source_name] += 1
        stats.update(mix_stats)
        stats["mix_consumed_by_source"] = dict(self._consumed_by_source)
        return batch, stats

    def set_epoch(self, epoch: int) -> None:
        if self.queue is not None and self.queue.started:
            raise RuntimeError("cannot set epoch after the mixed queue has started")
        self._epoch = int(epoch)
        self.stream.set_epoch(self._epoch)
        self._iterator = None

    def snapshot(self) -> dict[str, Any]:
        stats = (
            {
                "ready_queue_enabled": False,
                "data_epoch": int(self._epoch),
            }
            if self.queue is None
            else self.queue.snapshot()
        )
        stats.update(
            {
                "data_backend": "motus_indexed_mix",
                "mix_sources": [source.name for source in self.sources],
                "mix_normalized_weights": {
                    source.name: weight
                    for source, weight in zip(
                        self.sources, self.stream.normalized_weights
                    )
                },
                "mix_epoch_quotas": {
                    source.name: quota
                    for source, quota in zip(
                        self.sources, self.stream.source_quotas()
                    )
                },
                "mix_consumed_by_source": dict(self._consumed_by_source),
                "mix_cross_rank_source_policy": (
                    self.stream.cross_rank_source_policy
                ),
                "mix_rank": int(self.rank),
                "mix_world_size": int(self.world_size),
                "mix_batch_transform_enabled": self.batch_transform is not None,
            }
        )
        return stats

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        if self.queue is not None:
            try:
                self.queue.close(timeout=self.queue_close_timeout_seconds)
            except BaseException as error:
                errors.append(error)
        self._iterator = None
        for child in self.children:
            try:
                child.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                f"failed to close {len(errors)} indexed mix resources"
            ) from errors[0]

    def __enter__(self) -> "IndexedMixedTrainingDataPipeline":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def open_mixed_training_data(
    sources: Sequence[IndexedTrainingSource | Mapping[str, Any]],
    *,
    batch_size: int,
    batches_per_epoch: int | None = None,
    output_adapter: Any = None,
    adapter_options: Mapping[str, Any] | None = None,
    video_size: tuple[int, int] | None = None,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    queue_enabled: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    queue_size: int = 2,
    queue_timeout_seconds: float = 600.0,
    queue_clone_tensors: bool = False,
    batch_transform: Callable[[Any], Any] | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    cross_rank_source_policy: str = CROSS_RANK_SOURCE_POLICY_SYNCHRONIZED,
) -> IndexedMixedTrainingDataPipeline:
    """Open a weighted mix of independently bound indexed training sources."""

    from motus_data.indexed_pipeline import (
        _distributed_context,
        open_training_data,
    )

    parsed_sources = tuple(
        IndexedTrainingSource.from_value(source) for source in sources
    )
    if not parsed_sources:
        raise ValueError("open_mixed_training_data requires at least one source")
    rank, world_size = _distributed_context(rank, world_size)
    if int(num_workers) < 0:
        raise ValueError("num_workers must be >= 0")
    if int(queue_size) <= 0:
        raise ValueError("queue_size must be positive")
    if float(queue_timeout_seconds) <= 0:
        raise ValueError("queue_timeout_seconds must be positive")

    children = []
    try:
        for source in parsed_sources:
            child_adapter = (
                output_adapter
                if source.output_adapter is None
                else source.output_adapter
            )
            child_adapter_options = dict(adapter_options or {})
            child_adapter_options.update(dict(source.adapter_options or {}))
            children.append(
                open_training_data(
                    source.dataset,
                    sample_pool=source.sample_pool,
                    batch_size=batch_size,
                    output_adapter=child_adapter,
                    video_size=video_size,
                    release=source.release,
                    adapter_options=(
                        child_adapter_options
                        if child_adapter_options
                        else None
                    ),
                    seed=seed,
                    rank=rank,
                    world_size=world_size,
                    queue_enabled=False,
                    catalog_root=source.catalog_root,
                    catalog_path=source.catalog_path,
                    index_path=source.index_path,
                    use_index=source.use_index,
                    verify_sha256=source.verify_sha256,
                    record_cache_size=source.record_cache_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    readers=source.readers,
                    reader_options=source.reader_options,
                )
            )
        resolved_sources = tuple(
            source
            if source.weight is not None
            else replace(source, weight=float(len(child.sample_pool)))
            for source, child in zip(parsed_sources, children)
        )
        if batches_per_epoch is None:
            batches_per_epoch = sum(
                len(child.batch_sampler) for child in children
            )
        stream = WeightedIndexedBatchStream(
            children,
            resolved_sources,
            batches_per_epoch=int(batches_per_epoch),
            seed=int(seed),
            shuffle=bool(shuffle),
            rank=rank,
            world_size=world_size,
            cross_rank_source_policy=cross_rank_source_policy,
        )
        queue = None
        if queue_enabled:
            queue = ReadyBatchQueue(
                stream,
                max_size=int(queue_size),
                get_timeout_seconds=float(queue_timeout_seconds),
                low_watermark=0,
                name=f"indexed_mix.rank{rank}",
                clone_tensors=bool(queue_clone_tensors),
                producer_transform=(
                    (lambda envelope: _transform_envelope(envelope, batch_transform))
                    if batch_transform is not None
                    else None
                ),
            )
        return IndexedMixedTrainingDataPipeline(
            sources=resolved_sources,
            children=tuple(children),
            stream=stream,
            queue=queue,
            rank=rank,
            world_size=world_size,
            queue_close_timeout_seconds=float(queue_timeout_seconds),
            batch_transform=batch_transform,
        )
    except BaseException:
        for child in reversed(children):
            child.close()
        raise
