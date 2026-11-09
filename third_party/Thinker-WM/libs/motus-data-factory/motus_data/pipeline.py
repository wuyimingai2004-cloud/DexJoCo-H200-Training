"""Model-independent Dataset -> DataLoader -> ready queue pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from motus_data.collate import collate_samples
from motus_data.contracts import validate_batch
from motus_data.factory import build_dataset
from motus_data.loading.ready_queue import ReadyBatchQueue


def _close_dataset(dataset) -> None:
    for child in getattr(dataset, "datasets", []):
        _close_dataset(child)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def load_config(config: Any) -> DictConfig:
    if isinstance(config, (str, Path)):
        return OmegaConf.load(str(config))
    if isinstance(config, DictConfig):
        return config
    return OmegaConf.create(config)


def _loader_kwargs(config: DictConfig) -> Dict[str, Any]:
    system = config.get("system", {})
    num_workers = int(system.get("num_workers", 0))
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(system.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(system.get("persistent_workers", False))
        prefetch = system.get("prefetch_factor")
        if prefetch is not None:
            kwargs["prefetch_factor"] = int(prefetch)
    return kwargs


def build_dataloader(
    config: Any,
    *,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
):
    cfg = load_config(config)
    dataset = build_dataset(cfg, split=split, rank=rank, seed=seed)
    batch_size = int(cfg.training.batch_size)
    kwargs = _loader_kwargs(cfg)
    dataset_collate = getattr(dataset, "collate_fn", None)
    collate = dataset_collate if callable(dataset_collate) else collate_samples
    if hasattr(dataset, "create_batch_sampler"):
        sampler = dataset.create_batch_sampler(
            batch_size=batch_size,
            drop_last=True,
            shuffle=split == "train",
            num_replicas=int(world_size),
            rank=int(rank),
            seed=int(seed),
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=True,
        collate_fn=collate,
        **kwargs,
    )


@dataclass
class DataPipeline:
    config: DictConfig
    dataloader: DataLoader
    queue: Optional[ReadyBatchQueue]
    expected_batch_size: int
    _iterator: Any = None

    @property
    def dataset(self):
        return self.dataloader.dataset

    def start(self) -> "DataPipeline":
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
                self._iterator = iter(self.dataloader)
                batch = next(self._iterator)
            stats = {"ready_queue_enabled": False}
        if batch is None:
            raise RuntimeError("DataLoader produced an empty batch")
        validate_batch(batch, expected_batch_size=self.expected_batch_size)
        return batch, stats

    def close(self) -> None:
        if self.queue is not None:
            self.queue.close()
        _close_dataset(self.dataset)
        self._iterator = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()


def build_data_pipeline(
    config: Any,
    *,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    queue_enabled: Optional[bool] = None,
) -> DataPipeline:
    cfg = load_config(config)
    loader = build_dataloader(cfg, split=split, rank=rank, world_size=world_size, seed=seed)
    system = cfg.get("system", {})
    enabled = bool(system.get("ready_batch_queue_enabled", True))
    if queue_enabled is not None:
        enabled = bool(queue_enabled)
    queue = None
    if enabled:
        queue = ReadyBatchQueue(
            loader,
            max_size=int(system.get("ready_batch_queue_size", 1)),
            get_timeout_seconds=float(system.get("ready_batch_queue_timeout_seconds", 600.0)),
            low_watermark=int(system.get("ready_batch_queue_low_watermark", 0)),
            name=f"{split}.rank{rank}",
            clone_tensors=bool(system.get("ready_batch_queue_clone_tensors", False)),
        )
    return DataPipeline(
        config=cfg,
        dataloader=loader,
        queue=queue,
        expected_batch_size=int(cfg.training.batch_size),
    )
