import bisect
import math
import random
from typing import List, Optional, Sequence

from torch.utils.data import Dataset, Sampler

from motus_data.deprecation import warn_episode_native_deprecated


class MixDataset(Dataset):
    """A map-style wrapper for multiple Motus datasets."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        names: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        batches_per_epoch: Optional[int] = None,
    ):
        warn_episode_native_deprecated("MixDataset")
        if len(datasets) == 0:
            raise ValueError("MixDataset requires at least one child dataset")
        if len(datasets) != len(names):
            raise ValueError("datasets and names must have the same length")

        self.datasets = list(datasets)
        self.names = [str(name) for name in names]
        self.lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"All child datasets must be non-empty, got lengths={self.lengths}")

        if weights is None:
            weights = self.lengths
        if len(weights) != len(self.datasets):
            raise ValueError("weights must have the same length as datasets")

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [float(weight) / total_weight for weight in weights]
        self.batches_per_epoch = batches_per_epoch

        self._offsets = []
        running = 0
        for length in self.lengths:
            self._offsets.append(running)
            running += length
        self._total_len = running

    def __len__(self):
        return self._total_len

    @staticmethod
    def _is_prebatched_dataset(dataset: Dataset) -> bool:
        return bool(getattr(dataset, "produces_prebatched_batches", False))

    @staticmethod
    def collate_fn(items):
        if (
            len(items) == 1
            and isinstance(items[0], dict)
            and items[0].pop("_mix_prebatched", False)
        ):
            return items[0]
        from motus_data.collate import collate_samples

        return collate_samples(items)

    def __getitem__(self, index):
        dataset_idx = bisect.bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[dataset_idx]
        sample = self.datasets[dataset_idx][local_idx]
        if isinstance(sample, dict):
            sample.setdefault("source_dataset", self.names[dataset_idx])
            sample.setdefault("source_child", self.names[dataset_idx])
        return sample

    def __getitems__(self, indices):
        if not indices:
            return []

        mapped = []
        for index in indices:
            dataset_idx = bisect.bisect_right(self._offsets, int(index)) - 1
            local_idx = int(index) - self._offsets[dataset_idx]
            mapped.append((dataset_idx, local_idx))

        dataset_indices = {dataset_idx for dataset_idx, _ in mapped}
        if len(dataset_indices) != 1:
            return [self[int(index)] for index in indices]

        dataset_idx = mapped[0][0]
        child = self.datasets[dataset_idx]
        local_indices = [local_idx for _, local_idx in mapped]
        if self._is_prebatched_dataset(child) and hasattr(child, "get_grouped_batch"):
            batch = child.get_grouped_batch(local_indices)
            if isinstance(batch, dict):
                batch.setdefault("source_datasets", [self.names[dataset_idx]] * len(local_indices))
                if not batch.get("source_children"):
                    batch["source_children"] = [self.names[dataset_idx]] * len(local_indices)
                batch["_mix_prebatched"] = True
            return [batch]

        return [self[int(index)] for index in indices]

    def global_index(self, dataset_idx: int, local_idx: int) -> int:
        return self._offsets[dataset_idx] + local_idx

    def create_batch_sampler(
        self,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        return MixDatasetBatchSampler(
            self,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )


class MixDatasetBatchSampler(Sampler[List[int]]):
    """Batch sampler that keeps every batch within one child dataset."""

    def __init__(
        self,
        dataset: MixDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self._child_batch_samplers = []
        self._child_batch_iterators = []
        self._child_batch_epochs = []

        for child in self.dataset.datasets:
            if (
                self.dataset._is_prebatched_dataset(child)
                and hasattr(child, "create_batch_sampler")
            ):
                sampler = child.create_batch_sampler(
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                    shuffle=self.shuffle,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    seed=self.seed,
                )
            else:
                sampler = None
            self._child_batch_samplers.append(sampler)
            self._child_batch_iterators.append(None)
            self._child_batch_epochs.append(0)

        if dataset.batches_per_epoch is not None:
            total_batches = int(dataset.batches_per_epoch)
        elif drop_last:
            total_batches = sum(length // batch_size for length in dataset.lengths)
        else:
            total_batches = sum(math.ceil(length / batch_size) for length in dataset.lengths)
        self.num_batches = max(1, math.ceil(total_batches / num_replicas))

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for child in self.dataset.datasets:
            if hasattr(child, "set_epoch"):
                child.set_epoch(self.epoch)
        for sampler in self._child_batch_samplers:
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
        self._child_batch_iterators = [None for _ in self._child_batch_iterators]
        self._child_batch_epochs = [self.epoch for _ in self._child_batch_epochs]

    def _next_child_batch(self, dataset_idx: int) -> List[int]:
        sampler = self._child_batch_samplers[dataset_idx]
        if sampler is None:
            raise ValueError(f"Dataset {dataset_idx} does not have a child batch sampler")

        iterator = self._child_batch_iterators[dataset_idx]
        if iterator is None:
            iterator = iter(sampler)
            self._child_batch_iterators[dataset_idx] = iterator

        try:
            return [int(local_idx) for local_idx in next(iterator)]
        except StopIteration:
            self._child_batch_epochs[dataset_idx] += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._child_batch_epochs[dataset_idx])
            iterator = iter(sampler)
            self._child_batch_iterators[dataset_idx] = iterator
            return [int(local_idx) for local_idx in next(iterator)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 100003 + self.rank)
        dataset_indices = list(range(len(self.dataset.datasets)))

        for _ in range(self.num_batches):
            if self.shuffle:
                dataset_idx = rng.choices(dataset_indices, weights=self.dataset.weights, k=1)[0]
                if self._child_batch_samplers[dataset_idx] is not None:
                    local_indices = self._next_child_batch(dataset_idx)
                else:
                    local_indices = [
                        rng.randrange(self.dataset.lengths[dataset_idx])
                        for _ in range(self.batch_size)
                    ]
            else:
                dataset_idx = dataset_indices[_ % len(dataset_indices)]
                if self._child_batch_samplers[dataset_idx] is not None:
                    local_indices = self._next_child_batch(dataset_idx)
                else:
                    length = self.dataset.lengths[dataset_idx]
                    start = ((_ * self.num_replicas + self.rank) * self.batch_size) % length
                    local_indices = [(start + i) % length for i in range(self.batch_size)]

            yield [
                self.dataset.global_index(dataset_idx, local_idx)
                for local_idx in local_indices
            ]
