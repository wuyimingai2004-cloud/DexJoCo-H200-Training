"""Model-owned SamplePool loading and same-Episode grouped indexed Dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch.utils.data import Dataset, Sampler

from motus_data.datasets.action_safety import (
    validate_action_state_finite,
    validate_action_state_magnitude,
)
from motus_data.index_contracts import (
    IndexContractError,
    validate_pool_catalog_binding,
    validate_sample_pool_header,
    validate_sample_selection_v1,
    validate_selection_pool_binding,
)
from motus_data.repository import (
    EpisodeInfo,
    EpisodeRepository,
    EpisodeRepositoryError,
    EpisodeSelectionData,
    TextItem,
)


class SamplePoolError(ValueError):
    """Raised when a SamplePool cannot be bound or consumed exactly."""


@runtime_checkable
class SupportsSamplePool(Protocol):
    """Formal interface every SamplePool implementation must satisfy.

    The indexed training pipeline (``SamplePoolDataset`` /
    ``EpisodeGroupedBatchSampler`` / ``open_training_data`` /
    ``open_mixed_training_data``) only relies on this surface:

    - ``selections``: sequence of selection mappings, indexable by integer
      position; each mapping follows the ``motus_sample_selection_v1``
      contract (``episode_id`` / ``video_frame_indices`` / ``state_indices``
      / ``action_indices`` / ``text_indices``).
    - ``episode_groups``: ``episode_id -> tuple of selection positions``;
      every group feeds same-Episode batches and grouped union reads.
    - ``pool_id``: diagnostics identifier.
    - ``__len__``: total selection count (also the default source weight).

    Selection *strategy* (static file, runtime-generated windows, custom
    per-content rules) is owned by user code; this library only standardizes
    formats, loading, packing, and transport. The static file-backed
    ``SamplePool`` satisfies this protocol.

    Pools may optionally define ``set_epoch(epoch: int)``. When present, the
    sampler chain (``WeightedIndexedBatchStream`` ->
    ``IndexedTrainingDataPipeline`` -> ``EpisodeGroupedBatchSampler`` ->
    ``SamplePoolDataset``) forwards every epoch rollover, so runtime-generated
    pools can refresh their selections per epoch. All ranks must generate
    identical selections for the same position, e.g. by deriving randomness
    from ``motus_data.sampling.compose_seed`` without rank/worker inputs.
    """

    @property
    def selections(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def episode_groups(self) -> Mapping[str, tuple[int, ...]]: ...

    @property
    def pool_id(self) -> str: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class IndexedEpisodeBatch:
    """One same-Episode batch before model-specific shape/collate adaptation."""

    info: EpisodeInfo
    samples: tuple[EpisodeSelectionData, ...]

    @property
    def episode_id(self) -> str:
        return self.info.episode_id

    @property
    def batch_size(self) -> int:
        return len(self.samples)


class SamplePool:
    """Validated v1/v2 SamplePool loaded without source decoding."""

    def __init__(self, path: str | Path, repository: EpisodeRepository):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        header = None
        selections: list[Mapping[str, Any]] = []
        episode_groups: dict[str, list[int]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SamplePoolError(
                        f"invalid SamplePool JSON at line {line_number}: {error}"
                    ) from error
                if header is None:
                    try:
                        validate_sample_pool_header(row)
                        validate_pool_catalog_binding(row, repository.metadata)
                    except IndexContractError as error:
                        raise SamplePoolError(
                            f"invalid SamplePool header at line {line_number}: {error}"
                        ) from error
                    header = copy.deepcopy(row)
                    continue
                if isinstance(row, Mapping) and (
                    row.get("record_type") == "sample_pool_header"
                    or str(row.get("format", "")).startswith("motus_sample_pool_")
                ):
                    raise SamplePoolError(
                        f"SamplePool contains a second header at line {line_number}"
                    )
                try:
                    validate_sample_selection_v1(row)
                except IndexContractError as error:
                    raise SamplePoolError(
                        f"invalid SamplePool selection at line {line_number}: {error}"
                    ) from error
                try:
                    episode = repository.get_episode(str(row["episode_id"]))
                except KeyError as error:
                    raise SamplePoolError(
                        f"SamplePool line {line_number} references missing "
                        f"episode_id={row['episode_id']!r}"
                    ) from error
                try:
                    validate_selection_pool_binding(
                        row, episode._catalog_record, header
                    )
                    episode._validated_selection_indices(row)
                except (IndexContractError, EpisodeRepositoryError) as error:
                    raise SamplePoolError(
                        f"invalid SamplePool selection at line {line_number}: {error}"
                    ) from error
                selection_index = len(selections)
                selections.append(copy.deepcopy(row))
                episode_groups.setdefault(str(row["episode_id"]), []).append(
                    selection_index
                )
        if header is None:
            raise SamplePoolError("SamplePool has no header")
        if not selections:
            raise SamplePoolError("SamplePool has no sample selections")
        expected_count = header.get("record_count", header.get("samples"))
        if expected_count is not None and len(selections) != int(expected_count):
            raise SamplePoolError(
                f"SamplePool record count mismatch: selections={len(selections)} "
                f"header={int(expected_count)}"
            )
        self.header = header
        self.selections = tuple(selections)
        self.episode_groups = {
            episode_id: tuple(indices)
            for episode_id, indices in episode_groups.items()
        }

    def __len__(self) -> int:
        return len(self.selections)

    @property
    def pool_id(self) -> str:
        return str(self.header.get("sample_pool_id", self.header.get("pool_id")))


def resolve_sample_pool(
    sample_pool: "SupportsSamplePool | str | Path | Callable[..., SupportsSamplePool]",
    repository: EpisodeRepository,
    *,
    seed: int = 0,
) -> SupportsSamplePool:
    """Resolve a pool specification into a ``SupportsSamplePool`` instance.

    Accepted forms:

    - a ``SupportsSamplePool`` instance: used as-is (user-owned strategy);
    - a factory callable: invoked as ``factory(repository=repository,
      seed=seed)`` and must return a ``SupportsSamplePool`` (for strategies
      that need catalog access or pipeline-seeded randomness);
    - a ``str``/``Path``: loaded as a static file-backed ``SamplePool``.
    """
    if isinstance(sample_pool, SupportsSamplePool):
        return sample_pool
    if callable(sample_pool):
        pool = sample_pool(repository=repository, seed=int(seed))
        if not isinstance(pool, SupportsSamplePool):
            raise SamplePoolError(
                "sample_pool factory must return a SupportsSamplePool, got "
                f"{type(pool).__name__}"
            )
        return pool
    return SamplePool(sample_pool, repository)


class SamplePoolDataset(Dataset):
    """Decode model-owned selections, with grouped union reads when batched."""

    def __init__(
        self,
        repository: EpisodeRepository,
        sample_pool: SupportsSamplePool | str | Path,
        *,
        video_output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
        action_magnitude_guard_threshold: float | None = 100.0,
        action_magnitude_guard_sources: Sequence[str] | None = None,
    ):
        self.repository = repository
        self.sample_pool = (
            sample_pool
            if isinstance(sample_pool, SupportsSamplePool)
            else SamplePool(sample_pool, repository)
        )
        self.video_output_format = str(video_output_format)
        self.video_size = (
            None
            if video_size is None
            else tuple(int(value) for value in video_size)
        )
        if (
            action_magnitude_guard_threshold is not None
            and (
                not math.isfinite(float(action_magnitude_guard_threshold))
                or float(action_magnitude_guard_threshold) <= 0
            )
        ):
            raise SamplePoolError(
                "action_magnitude_guard_threshold must be positive or None"
            )
        self.action_magnitude_guard_threshold = (
            None
            if action_magnitude_guard_threshold is None
            else float(action_magnitude_guard_threshold)
        )
        if isinstance(action_magnitude_guard_sources, str):
            parsed_guard_sources = tuple(
                value.strip()
                for value in action_magnitude_guard_sources.split(",")
                if value.strip()
            )
        else:
            parsed_guard_sources = action_magnitude_guard_sources
        self.action_magnitude_guard_sources = (
            None
            if parsed_guard_sources is None
            else tuple(str(value) for value in parsed_guard_sources)
        )
        self.episode_groups = self.sample_pool.episode_groups

    def _selection_id(self, selection: Mapping[str, Any]) -> str:
        payload = json.dumps(
            selection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"{self.sample_pool.pool_id}:{digest}"

    def _preflight_validate(
        self,
        info: EpisodeInfo,
        selection: Mapping[str, Any],
        state: torch.Tensor,
        action: torch.Tensor,
        text: tuple[TextItem, ...],
    ) -> None:
        selection_id = self._selection_id(selection)
        validate_action_state_finite(
            source_dataset=info.source_dataset,
            action_sequence=action,
            initial_state=state,
            selection_id=selection_id,
        )
        if self.action_magnitude_guard_threshold is None:
            return
        task_name = info.metadata.get("task_name")
        if task_name is None:
            task_name = next(
                (item.text for item in text if item.kind == "episode_context"),
                None,
            )
        validate_action_state_magnitude(
            source_dataset=info.source_dataset,
            action_sequence=action,
            initial_state=state,
            manifest_id=selection_id,
            manifest_task=None if task_name is None else str(task_name),
            threshold=self.action_magnitude_guard_threshold,
            sources=self.action_magnitude_guard_sources,
        )

    def __len__(self) -> int:
        return len(self.sample_pool)

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to pools that regenerate selections per epoch."""
        set_epoch = getattr(self.sample_pool, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def selection(self, index: int) -> Mapping[str, Any]:
        return copy.deepcopy(self.sample_pool.selections[int(index)])

    def __getitem__(self, index: int) -> EpisodeSelectionData:
        selection = self.sample_pool.selections[int(index)]
        return self.repository.get_episode(
            str(selection["episode_id"])
        ).read_selection(
            selection,
            video_output_format=self.video_output_format,
            video_size=self.video_size,
            preflight_validator=self._preflight_validate,
        )

    def __getitems__(self, indices: Sequence[int]) -> list[EpisodeSelectionData]:
        return list(self.get_grouped_samples(indices))

    def get_grouped_samples(
        self, indices: Sequence[int]
    ) -> tuple[EpisodeSelectionData, ...]:
        if not indices:
            raise SamplePoolError("grouped Dataset read requires at least one index")
        selections = [
            self.sample_pool.selections[int(index)] for index in indices
        ]
        episode_ids = {str(selection["episode_id"]) for selection in selections}
        if len(episode_ids) != 1:
            raise SamplePoolError(
                "one grouped Dataset read must contain exactly one Episode"
            )
        episode_id = next(iter(episode_ids))
        return self.repository.get_episode(episode_id).read_selections(
            selections,
            video_output_format=self.video_output_format,
            video_size=self.video_size,
            preflight_validator=self._preflight_validate,
        )

    def get_grouped_batch(self, indices: Sequence[int]) -> IndexedEpisodeBatch:
        samples = self.get_grouped_samples(indices)
        return IndexedEpisodeBatch(info=samples[0].info, samples=samples)

    @staticmethod
    def collate_fn(items: Sequence[EpisodeSelectionData]) -> IndexedEpisodeBatch:
        if not items:
            raise SamplePoolError("cannot collate an empty indexed batch")
        episode_ids = {item.info.episode_id for item in items}
        if len(episode_ids) != 1:
            raise SamplePoolError(
                "IndexedEpisodeBatch must contain exactly one Episode"
            )
        return IndexedEpisodeBatch(info=items[0].info, samples=tuple(items))

    def create_batch_sampler(
        self,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> "EpisodeGroupedBatchSampler":
        return EpisodeGroupedBatchSampler(
            self,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )


class EpisodeGroupedBatchSampler(Sampler[list[int]]):
    """Batch SamplePool rows by Episode, then shard complete batches by rank."""

    def __init__(
        self,
        dataset: SamplePoolDataset,
        *,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(num_replicas, bool) or int(num_replicas) <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError(
                f"rank must be within [0,{int(num_replicas)}), got {rank}"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        global_batch_count = self._global_batch_count()
        if global_batch_count <= 0:
            raise ValueError(
                "SamplePool produces no grouped batches with the requested "
                f"batch_size={self.batch_size} drop_last={self.drop_last}"
            )
        if self.drop_last and global_batch_count < self.num_replicas:
            raise ValueError(
                "grouped batch count is smaller than num_replicas with "
                f"drop_last=True: batches={global_batch_count} "
                f"num_replicas={self.num_replicas}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        set_epoch = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.epoch)

    def _global_batch_count(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self.dataset.episode_groups.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.dataset.episode_groups.values()
        )

    def _global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        batches = []
        for episode_id in sorted(self.dataset.episode_groups):
            indices = list(self.dataset.episode_groups[episode_id])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _rank_batches(self) -> list[list[int]]:
        batches = self._global_batches()
        if self.drop_last:
            total = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[:total]
        else:
            total = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
            if total > len(batches):
                batches.extend(
                    [list(batches[index % len(batches)])
                     for index in range(total - len(batches))]
                )
        return batches[self.rank : len(batches) : self.num_replicas]

    def __len__(self) -> int:
        if self.drop_last:
            return self._global_batch_count() // self.num_replicas
        return math.ceil(self._global_batch_count() / self.num_replicas)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._rank_batches()
