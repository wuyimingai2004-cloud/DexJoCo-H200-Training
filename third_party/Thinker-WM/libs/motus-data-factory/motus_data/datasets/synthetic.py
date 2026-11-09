"""Deterministic dataset used for contract and pipeline verification."""

from __future__ import annotations

import time
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from motus_data.deprecation import warn_episode_native_deprecated


class SyntheticMotusDataset(Dataset):
    def __init__(
        self,
        length: int = 128,
        num_video_frames: int = 4,
        video_size=(16, 16),
        action_dim: int = 6,
        state_dim: int = 6,
        source_dataset: str = "synthetic",
        source_child: Optional[str] = None,
        include_action: bool = True,
        sleep_seconds: float = 0.0,
        seed: int = 0,
        **_,
    ):
        warn_episode_native_deprecated("SyntheticMotusDataset")
        self.length = int(length)
        self.num_video_frames = int(num_video_frames)
        self.video_size = tuple(int(value) for value in video_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.source_dataset = str(source_dataset)
        self.source_child = str(source_child or source_dataset)
        self.include_action = bool(include_action)
        self.sleep_seconds = float(sleep_seconds)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003 + int(index)
        )
        height, width = self.video_size
        frames = torch.rand(
            self.num_video_frames + 1, 3, height, width, generator=generator
        )
        state = torch.rand(self.state_dim, generator=generator) if self.include_action else None
        action = (
            torch.rand(self.num_video_frames, self.action_dim, generator=generator)
            if self.include_action else None
        )
        first_frame_pil = Image.fromarray(
            frames[0].mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        )
        return {
            "first_frame": frames[0],
            "video_frames": frames[1:],
            "initial_state": state,
            "action_sequence": action,
            "first_frame_pil": first_frame_pil,
            "t5_input_message": f"Task: synthetic item {index}",
            "vlm_input_message": f"Task: synthetic item {index}",
            "robot_name": f"{self.source_child}-robot",
            "source_dataset": self.source_dataset,
            "source_child": self.source_child,
            "action_loss_weight": 1.0 if self.include_action else 0.0,
            "video_loss_weight": 1.0,
            "state_action_policy": "real" if self.include_action else "none",
            "manifest_id": f"{self.source_child}:{index}",
            "manifest_task": "synthetic",
            "episode_id": f"episode:{index // 4}",
            "frame_indices": list(range(self.num_video_frames + 1)),
        }
