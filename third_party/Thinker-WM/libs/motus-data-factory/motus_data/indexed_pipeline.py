"""Minimal public APIs for Catalog-backed access and training consumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch.distributed as dist
from torch.utils.data import DataLoader

from motus_data.batch_adapters import (
    IndexedBatchAdapter,
    IndexedBatchCollator,
    build_indexed_batch_adapter,
)
from motus_data.loading import ReadyBatchQueue
from motus_data.releases import (
    EpisodeDatasetRelease,
    get_episode_dataset_release,
)
from motus_data.repository import EpisodeRepository, EpisodeSourceReader
from motus_data.sample_pool import (
    EpisodeGroupedBatchSampler,
    IndexedEpisodeBatch,
    SamplePoolDataset,
    SupportsSamplePool,
    resolve_sample_pool,
)


IndexedBatchTransform = Callable[[IndexedEpisodeBatch], Any]


@dataclass(frozen=True)
class CallableIndexedBatchCollator:
    """Adapt an IndexedEpisodeBatch with a model-owned callable."""

    transform: IndexedBatchTransform

    def __call__(self, items):
        return self.transform(SamplePoolDataset.collate_fn(items))


@dataclass
class IndexedTrainingDataPipeline:
    """Owned Catalog/SamplePool/DataLoader/queue lifecycle for one training rank."""

    release: EpisodeDatasetRelease
    repository: EpisodeRepository
    sample_pool: SupportsSamplePool
    dataset: SamplePoolDataset
    batch_sampler: EpisodeGroupedBatchSampler
    dataloader: DataLoader
    queue: ReadyBatchQueue | None
    rank: int
    world_size: int
    queue_close_timeout_seconds: float
    _iterator: Any = None
    _epoch: int = 0
    _closed: bool = False

    @property
    def queue_enabled(self) -> bool:
        return self.queue is not None

    def start(self) -> "IndexedTrainingDataPipeline":
        if self._closed:
            raise RuntimeError("training data pipeline is closed")
        if self.queue is not None:
            self.queue.start()
        elif self._iterator is None:
            self._iterator = iter(self.dataloader)
        return self

    def next(self):
        self.start()
        if self.queue is not None:
            batch, stats = self.queue.next()
        else:
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self.batch_sampler.set_epoch(self._epoch)
                self._iterator = iter(self.dataloader)
                batch = next(self._iterator)
            stats = {
                "ready_queue_enabled": False,
                "data_epoch": int(self._epoch),
            }
        if batch is None:
            raise RuntimeError("DataLoader produced an empty batch")
        return batch, stats

    def snapshot(self) -> dict[str, Any]:
        if self.queue is None:
            return {
                "ready_queue_enabled": False,
                "data_epoch": int(self._epoch),
            }
        return self.queue.snapshot()

    def set_epoch(self, epoch: int) -> None:
        if self.queue is not None and self.queue.started:
            raise RuntimeError("cannot set epoch after the training queue has started")
        self._epoch = int(epoch)
        self.batch_sampler.set_epoch(self._epoch)
        self._iterator = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.queue is not None:
            self.queue.close(timeout=self.queue_close_timeout_seconds)
        self._iterator = None
        self.repository.close()

    def __enter__(self) -> "IndexedTrainingDataPipeline":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _distributed_context(
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int]:
    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size must be provided together")
    if rank is None:
        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
        else:
            rank, world_size = 0, 1
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within [0, world_size)")
    return rank, world_size


def _collate_for_adapter(
    output_adapter: str | IndexedBatchAdapter | IndexedBatchTransform | None,
    adapter_options: Mapping[str, Any] | None,
):
    options = dict(adapter_options or {})
    if output_adapter is None:
        if options:
            raise ValueError(
                "adapter_options require output_adapter to be specified"
            )
        return SamplePoolDataset.collate_fn
    if isinstance(output_adapter, str):
        adapter = build_indexed_batch_adapter(output_adapter, **options)
        return IndexedBatchCollator(adapter)
    if options:
        raise ValueError(
            "adapter_options are only valid when output_adapter is a profile name"
        )
    if hasattr(output_adapter, "adapt") and callable(output_adapter.adapt):
        return IndexedBatchCollator(output_adapter)
    if callable(output_adapter):
        return CallableIndexedBatchCollator(output_adapter)
    raise TypeError(
        "output_adapter must be None, a registered profile name, "
        "an adapter with adapt(), or a callable"
    )


def open_dataset(
    dataset: str,
    *,
    release: str | EpisodeDatasetRelease | None = None,
    catalog_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    index_path: str | Path | None = None,
    use_index: bool = True,
    verify_sha256: bool | None = None,
    record_cache_size: int | None = None,
    readers: Mapping[str, EpisodeSourceReader] | None = None,
    reader_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> EpisodeRepository:
    """Open one registered Episode dataset for general-purpose access."""

    selected_release = get_episode_dataset_release(dataset, release)
    release_catalog, release_index = selected_release.resolve_paths(catalog_root)
    resolved_catalog = (
        release_catalog if catalog_path is None else Path(catalog_path)
    )
    resolved_index = None
    if use_index:
        resolved_index = (
            release_index if index_path is None else Path(index_path)
        )
        if resolved_index is None:
            raise ValueError(
                f"dataset release has no offset index: "
                f"{selected_release.release_id}"
            )
    elif index_path is not None:
        raise ValueError("index_path cannot be used with use_index=False")
    if verify_sha256 is None:
        verify_sha256 = resolved_index is None
    cache_size = (
        selected_release.default_record_cache_size
        if record_cache_size is None
        else int(record_cache_size)
    )
    return EpisodeRepository(
        resolved_catalog,
        readers=readers,
        reader_options=(
            selected_release.reader_options
            if reader_options is None
            else reader_options
        ),
        index_path=resolved_index,
        verify_sha256=bool(verify_sha256),
        record_cache_size=cache_size,
    )


def open_training_data(
    dataset: str,
    *,
    sample_pool: (
        SupportsSamplePool | str | Path | Callable[..., SupportsSamplePool]
    ),
    batch_size: int,
    output_adapter: (
        str | IndexedBatchAdapter | IndexedBatchTransform | None
    ) = None,
    video_size: tuple[int, int] | None = None,
    release: str | EpisodeDatasetRelease | None = None,
    adapter_options: Mapping[str, Any] | None = None,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    queue_enabled: bool = True,
    catalog_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    index_path: str | Path | None = None,
    use_index: bool = True,
    verify_sha256: bool | None = None,
    record_cache_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    queue_size: int | None = None,
    queue_timeout_seconds: float | None = None,
    queue_clone_tensors: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    readers: Mapping[str, EpisodeSourceReader] | None = None,
    reader_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> IndexedTrainingDataPipeline:
    """Open the complete indexed CPU data path for one training rank."""

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    selected_release = get_episode_dataset_release(dataset, release)
    rank, world_size = _distributed_context(rank, world_size)
    resolved_video_size = (
        selected_release.default_video_size
        if video_size is None
        else tuple(int(value) for value in video_size)
    )
    if (
        resolved_video_size is not None
        and (
            len(resolved_video_size) != 2
            or any(value <= 0 for value in resolved_video_size)
        )
    ):
        raise ValueError("video_size must contain positive H,W")
    workers = (
        selected_release.default_num_workers
        if num_workers is None
        else int(num_workers)
    )
    if workers < 0:
        raise ValueError("num_workers must be >= 0")
    pin = (
        selected_release.default_pin_memory
        if pin_memory is None
        else bool(pin_memory)
    )
    ready_size = (
        selected_release.default_queue_size
        if queue_size is None
        else int(queue_size)
    )
    timeout = (
        selected_release.default_queue_timeout_seconds
        if queue_timeout_seconds is None
        else float(queue_timeout_seconds)
    )
    if ready_size <= 0:
        raise ValueError("queue_size must be positive")
    if timeout <= 0:
        raise ValueError("queue_timeout_seconds must be positive")

    repository = open_dataset(
        dataset,
        release=selected_release,
        catalog_root=catalog_root,
        catalog_path=catalog_path,
        index_path=index_path,
        use_index=use_index,
        verify_sha256=verify_sha256,
        record_cache_size=record_cache_size,
        readers=readers,
        reader_options=reader_options,
    )
    try:
        pool = resolve_sample_pool(sample_pool, repository, seed=int(seed))
        indexed_dataset = SamplePoolDataset(
            repository,
            pool,
            video_output_format=selected_release.default_video_output_format,
            video_size=resolved_video_size,
            action_magnitude_guard_threshold=(
                selected_release.action_magnitude_guard_threshold
            ),
            action_magnitude_guard_sources=(
                selected_release.action_magnitude_guard_sources
            ),
        )
        batch_sampler = indexed_dataset.create_batch_sampler(
            batch_size=int(batch_size),
            drop_last=bool(drop_last),
            shuffle=bool(shuffle),
            num_replicas=world_size,
            rank=rank,
            seed=int(seed),
        )
        collate_fn = _collate_for_adapter(output_adapter, adapter_options)
        loader_kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": pin,
        }
        if workers > 0:
            loader_kwargs["persistent_workers"] = False
        dataloader = DataLoader(
            indexed_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        queue = None
        if queue_enabled:
            queue = ReadyBatchQueue(
                dataloader,
                max_size=ready_size,
                get_timeout_seconds=timeout,
                low_watermark=0,
                name=(
                    f"{selected_release.dataset}."
                    f"{selected_release.release}.rank{rank}"
                ),
                clone_tensors=bool(queue_clone_tensors),
            )
        return IndexedTrainingDataPipeline(
            release=selected_release,
            repository=repository,
            sample_pool=pool,
            dataset=indexed_dataset,
            batch_sampler=batch_sampler,
            dataloader=dataloader,
            queue=queue,
            rank=rank,
            world_size=world_size,
            queue_close_timeout_seconds=timeout,
        )
    except BaseException:
        repository.close()
        raise
