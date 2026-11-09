"""Direct source-specific Agibot Dataset adapted from the pre-Manifest loader."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch.utils.data import Dataset, get_worker_info

from motus_data.contracts import validate_sample
from motus_data.datasets.agibot_common import (
    TASK_PROMPT,
    VLM_PROMPT,
    action_text_for_start,
    build_agibot_state_action,
    decode_video_frames,
    padded_frame_indices,
    read_agibot_eef_actions,
    tensor_to_pil,
)
from motus_data.deprecation import warn_episode_native_deprecated
from motus_data.sampling import compose_seed


logger = logging.getLogger(__name__)


def scan_agibot_episodes(
    annotation_folders: Sequence[str],
    data_root_folders: Sequence[str],
    dataset_names: Sequence[str],
) -> list[Dict[str, Any]]:
    if not (
        len(annotation_folders) == len(data_root_folders) == len(dataset_names)
    ):
        raise ValueError(
            "annotation_folders, data_root_folders and dataset_names must have equal lengths"
        )
    episodes = []
    for annotation_value, data_root_value, dataset_name in zip(
        annotation_folders, data_root_folders, dataset_names
    ):
        if "agibot-world" not in str(dataset_name).lower():
            raise ValueError(f"unsupported Agibot dataset name: {dataset_name}")
        annotation_root = Path(annotation_value)
        data_root = Path(data_root_value)
        for annotation_path in sorted(annotation_root.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as handle:
                items = json.load(handle)
            for item in items:
                task_id = item["task_id"]
                episode_id = item["episode_id"]
                episodes.append(
                    {
                        "episode_uid": f"agibot:{task_id}:{episode_id}",
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "task_name": item.get("task_name", ""),
                        "init_scene_text": item.get("init_scene_text", ""),
                        "video_path": str(
                            data_root
                            / "observations"
                            / str(task_id)
                            / str(episode_id)
                            / "videos_5fps"
                            / "head_color.mp4"
                        ),
                        "proprio_path": str(
                            data_root
                            / "proprio_stats"
                            / str(task_id)
                            / str(episode_id)
                            / "proprio_stats.h5"
                        ),
                        "action_video_fps_ratio": int(item["action_video_fps_ratio"]),
                        "action_labels": (item.get("label_info") or {}).get(
                            "flatten_action_config"
                        )
                        or {},
                        "effective_frames": [int(value) for value in item["effective_frames"]],
                        "annotation_path": str(annotation_path),
                    }
                )
    if not episodes:
        raise ValueError(f"no Agibot episodes found in {list(annotation_folders)}")
    return episodes


def _episode_split_score(episode_uid: str, seed: int) -> int:
    payload = f"{int(seed)}:{episode_uid}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def split_agibot_episodes(
    episodes: Sequence[Dict[str, Any]], split: str, test_fraction: float, split_seed: int
) -> list[Dict[str, Any]]:
    if not 0.0 <= float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    ordered_for_holdout = sorted(
        episodes,
        key=lambda item: (_episode_split_score(item["episode_uid"], split_seed), item["episode_uid"]),
    )
    test_count = int(round(len(ordered_for_holdout) * float(test_fraction)))
    if test_fraction > 0 and episodes:
        test_count = max(1, test_count)
    test_uids = {
        item["episode_uid"] for item in ordered_for_holdout[:test_count]
    }
    requested = "test" if str(split).lower() in {"test", "val", "validation"} else "train"
    selected = [
        item
        for item in episodes
        if (item["episode_uid"] in test_uids) == (requested == "test")
    ]
    return sorted(selected, key=lambda item: (int(item["task_id"]), int(item["episode_id"])))


class AgibotDataset(Dataset):
    def __init__(
        self,
        annotation_folders: Sequence[str],
        data_root_folders: Sequence[str],
        dataset_names: Sequence[str] = ("agibot-world",),
        *,
        split: str = "train",
        video_size: Tuple[int, int] = (320, 384),
        num_video_frames: int = 48,
        action_dim: int = 26,
        state_dim: int = 26,
        source_dataset: str = "agibot",
        source_child: str = "agibot",
        robot_name: str = "agibot-a2-d",
        test_fraction: float = 0.01,
        split_seed: int = 20260720,
        virtual_samples_per_episode: int = 1,
        max_decode_retries: int = 16,
        seed: int = 0,
        rank: int = 0,
    ):
        warn_episode_native_deprecated("AgibotDataset")
        self.video_size = tuple(int(value) for value in video_size)
        self.num_video_frames = int(num_video_frames)
        self.frames_clip_num = self.num_video_frames + 1
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.source_dataset = str(source_dataset)
        self.source_child = str(source_child)
        self.robot_name = str(robot_name)
        self.split = "test" if str(split).lower() in {"test", "val", "validation"} else "train"
        self.seed = int(seed)
        self.rank = int(rank)
        self.epoch = 0
        self.virtual_samples_per_episode = int(virtual_samples_per_episode)
        self.max_decode_retries = int(max_decode_retries)
        if self.virtual_samples_per_episode <= 0 or self.max_decode_retries <= 0:
            raise ValueError("virtual_samples_per_episode and max_decode_retries must be positive")

        all_episodes = scan_agibot_episodes(
            annotation_folders, data_root_folders, dataset_names
        )
        self.total_episode_count = len(all_episodes)
        self.episodes = split_agibot_episodes(
            all_episodes, self.split, test_fraction, split_seed
        )
        if not self.episodes:
            raise ValueError(f"Agibot has no episodes for split={self.split}")
        logger.info(
            "AgibotDataset initialized directly from annotations: split=%s "
            "episodes=%d total=%d virtual_samples_per_episode=%d",
            self.split,
            len(self.episodes),
            self.total_episode_count,
            self.virtual_samples_per_episode,
        )

    def __len__(self) -> int:
        return len(self.episodes) * self.virtual_samples_per_episode

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_start(
        self, episode: Dict[str, Any], frame_numbers: int, generator: torch.Generator
    ) -> int:
        effective_start, effective_end = episode["effective_frames"]
        max_start = max(
            min(int(effective_end), int(frame_numbers) - self.frames_clip_num),
            int(effective_start),
        )
        if self.split == "train" and max_start > int(effective_start):
            return int(
                torch.randint(
                    int(effective_start), max_start + 1, (1,), generator=generator
                ).item()
            )
        return int(effective_start) + (max_start - int(effective_start)) // 2

    def _load_episode(
        self,
        episode: Dict[str, Any],
        generator: torch.Generator,
        explicit_start: int | None = None,
    ) -> Dict[str, Any]:
        ratio = int(episode["action_video_fps_ratio"])
        if ratio <= 0:
            raise ValueError(f"invalid action/video ratio for {episode['episode_uid']}")
        eef_actions = read_agibot_eef_actions(episode["proprio_path"])
        frame_numbers = int(eef_actions.shape[0] // ratio)
        start_frame = (
            self._sample_start(episode, frame_numbers, generator)
            if explicit_start is None
            else int(explicit_start)
        )
        effective_start, effective_end = episode["effective_frames"]
        max_start = max(
            min(int(effective_end), frame_numbers - self.frames_clip_num),
            int(effective_start),
        )
        if not int(effective_start) <= start_frame <= max_start:
            raise ValueError(
                f"start_frame={start_frame} outside [{effective_start}, {max_start}] "
                f"for {episode['episode_uid']}"
            )
        frame_indices, frame_mask = padded_frame_indices(
            start_frame, frame_numbers, self.frames_clip_num
        )
        frames = decode_video_frames(episode["video_path"], frame_indices, self.video_size)
        initial_state, action_sequence = build_agibot_state_action(
            eef_actions,
            frame_indices,
            frame_mask,
            action_video_fps_ratio=ratio,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        task_text = TASK_PROMPT.format(
            scene=str(episode.get("init_scene_text") or "").strip(),
            task=str(episode.get("task_name") or "").strip(),
        )
        action_text = action_text_for_start(
            episode["action_labels"], start_frame, require_key=True
        )
        t5_text = f"{task_text}{action_text}".strip()
        first_frame = frames[0]
        sample = {
            "first_frame": first_frame,
            "video_frames": frames[1:],
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "first_frame_pil": tensor_to_pil(first_frame),
            "t5_input_message": t5_text,
            "vlm_input_message": f"{VLM_PROMPT}{t5_text}".strip(),
            "robot_name": self.robot_name,
            "source_dataset": self.source_dataset,
            "source_child": self.source_child,
            "action_loss_weight": 1.0,
            "video_loss_weight": 1.0,
            "state_action_policy": "real",
            "manifest_id": (
                f"{self.source_dataset}:{self.split}:{episode['task_id']}:"
                f"{episode['episode_id']}:window:{start_frame:06d}"
            ),
            "manifest_task": str(episode.get("task_name") or ""),
            "episode_id": episode["episode_uid"],
            "frame_indices": frame_indices,
            "frame_mask": frame_mask,
            "action_video_fps_ratio": ratio,
            "split": self.split,
            "metadata": {
                "annotation_path": episode["annotation_path"],
                "start_frame": start_frame,
                "action_video_fps_ratio": ratio,
            },
        }
        validate_sample(sample)
        return sample

    def load_window(self, episode_index: int, start_frame: int) -> Dict[str, Any]:
        generator = torch.Generator().manual_seed(0)
        return self._load_episode(
            self.episodes[int(episode_index)], generator, explicit_start=start_frame
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        last_error = None
        for retry in range(self.max_decode_retries):
            candidate = (
                int(index) + retry * self.virtual_samples_per_episode
            ) % len(self)
            episode_index = candidate // self.virtual_samples_per_episode
            generator = torch.Generator().manual_seed(
                compose_seed(
                    self.seed,
                    self.epoch,
                    self.rank,
                    worker_id,
                    self.source_child,
                    candidate,
                )
            )
            try:
                return self._load_episode(self.episodes[episode_index], generator)
            except Exception as error:
                last_error = error
                if retry + 1 < self.max_decode_retries:
                    logger.warning(
                        "Agibot failed logical index %d; retrying another episode: %s",
                        candidate,
                        error,
                    )
        raise RuntimeError(
            f"Agibot failed index={index} after {self.max_decode_retries} attempts"
        ) from last_error


# Compatibility name for the first experimental data-factory revision.
AgibotEpisodeDataset = AgibotDataset
