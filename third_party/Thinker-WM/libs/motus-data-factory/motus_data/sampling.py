"""Runtime window sampling primitives for episode-native datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class EpisodeRange:
    episode_id: str
    first_frame: int
    last_frame: int


@dataclass(frozen=True)
class WindowSelection:
    episode_id: str
    start_frame: int
    frame_indices: tuple[int, ...]


def compose_seed(
    global_seed: int,
    epoch: int,
    rank: int,
    worker_id: int,
    dataset_name: str,
    sample_index: int,
) -> int:
    payload = (
        f"{int(global_seed)}:{int(epoch)}:{int(rank)}:{int(worker_id)}:"
        f"{dataset_name}:{int(sample_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


class ContiguousWindowSampler:
    def __init__(self, num_frames: int, frame_step: int = 1, random_train: bool = True):
        self.num_frames = int(num_frames)
        self.frame_step = int(frame_step)
        self.random_train = bool(random_train)
        if self.num_frames <= 0 or self.frame_step <= 0:
            raise ValueError("num_frames and frame_step must be positive")

    @property
    def span(self) -> int:
        return (self.num_frames - 1) * self.frame_step + 1

    def sample(
        self,
        episode: EpisodeRange,
        *,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> WindowSelection:
        min_start = int(episode.first_frame)
        max_start = int(episode.last_frame) - self.span + 1
        if max_start < min_start:
            raise ValueError(
                f"episode {episode.episode_id!r} has {episode.last_frame - episode.first_frame + 1} "
                f"frames but sampling requires span={self.span}"
            )
        if training and self.random_train and max_start > min_start:
            start = int(torch.randint(min_start, max_start + 1, (1,), generator=generator).item())
        else:
            start = min_start + (max_start - min_start) // 2
        indices = tuple(start + offset * self.frame_step for offset in range(self.num_frames))
        return WindowSelection(episode.episode_id, start, indices)

