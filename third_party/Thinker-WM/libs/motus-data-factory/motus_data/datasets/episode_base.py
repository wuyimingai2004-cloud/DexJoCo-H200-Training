"""Base class for datasets that align modalities at runtime per episode."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset, get_worker_info

from motus_data.contracts import validate_sample
from motus_data.sampling import ContiguousWindowSampler, EpisodeRange, WindowSelection, compose_seed


class EpisodeDataset(Dataset, ABC):
    """Source adapter boundary for future dynamic sampling implementations."""

    def __init__(
        self,
        episodes: Sequence[Any],
        *,
        dataset_name: str,
        sampler: ContiguousWindowSampler,
        split: str = "train",
        seed: int = 0,
        rank: int = 0,
        virtual_samples_per_episode: int = 1,
        max_load_retries: int = 1,
    ):
        if not episodes:
            raise ValueError(f"{dataset_name} has no episodes for split={split}")
        self.episodes = list(episodes)
        self.dataset_name = str(dataset_name)
        self.sampler = sampler
        self.split = str(split)
        self.seed = int(seed)
        self.rank = int(rank)
        self.virtual_samples_per_episode = int(virtual_samples_per_episode)
        self.max_load_retries = int(max_load_retries)
        if self.virtual_samples_per_episode <= 0:
            raise ValueError("virtual_samples_per_episode must be positive")
        if self.max_load_retries <= 0:
            raise ValueError("max_load_retries must be positive")
        self.epoch = 0

    def __len__(self):
        return len(self.episodes) * self.virtual_samples_per_episode

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @abstractmethod
    def episode_range(self, episode: Any) -> EpisodeRange:
        raise NotImplementedError

    @abstractmethod
    def load_aligned_sample(
        self,
        episode: Any,
        selection: WindowSelection,
    ):
        """Decode and align video, text, state, and action for one selection."""
        raise NotImplementedError

    def _load_logical_index(self, index: int):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        generator = torch.Generator().manual_seed(
            compose_seed(
                self.seed,
                self.epoch,
                self.rank,
                worker_id,
                self.dataset_name,
                int(index),
            )
        )
        episode_index = int(index) // self.virtual_samples_per_episode
        episode = self.episodes[episode_index]
        selection = self.sampler.sample(
            self.episode_range(episode),
            generator=generator,
            training=self.split == "train",
        )
        sample = self.load_aligned_sample(episode, selection)
        validate_sample(sample)
        return sample

    def __getitem__(self, index: int):
        last_error = None
        for retry in range(self.max_load_retries):
            candidate = (int(index) + retry * self.virtual_samples_per_episode) % len(self)
            try:
                return self._load_logical_index(candidate)
            except Exception as error:
                last_error = error
                if retry + 1 < self.max_load_retries:
                    logging.getLogger(__name__).warning(
                        "%s failed logical index %d; retrying another episode: %s",
                        self.dataset_name,
                        candidate,
                        error,
                    )
        raise RuntimeError(
            f"{self.dataset_name} failed to load index={index} after "
            f"{self.max_load_retries} attempts"
        ) from last_error
