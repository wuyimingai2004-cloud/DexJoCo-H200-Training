"""Direct Agibot Beta dataset over task annotations and tar member catalogs.

The catalog contains only physical tar-member locations. Training windows are
sampled at runtime and are never stored in a Manifest or index file.
"""

from __future__ import annotations

import io
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, get_worker_info

from motus_data.collate import collate_samples
from motus_data.contracts import validate_sample
from motus_data.datasets.action_safety import validate_action_state_magnitude
from motus_data.datasets.agibot_common import (
    TASK_PROMPT,
    VLM_PROMPT,
    action_text_for_span_start,
    build_agibot_state_action,
    decode_video_frames,
    read_agibot_eef_actions,
    tensor_to_pil,
)
from motus_data.datasets.agibot_native import split_agibot_episodes
from motus_data.deprecation import warn_episode_native_deprecated
from motus_data.sampling import compose_seed


logger = logging.getLogger(__name__)
DEFAULT_EXCLUDED_TASK_NAMES = (
    "Water the flowers",
    "Print documents with a printer.",
    "Pack the medicine",
    "Clap hands",
    "Wave goodbye",
)


@dataclass(frozen=True)
class TarMemberRef:
    uri: str
    member: str
    offset_data: int
    size: int


def read_tar_member_bytes(reference: TarMemberRef) -> bytes:
    with open(reference.uri, "rb") as handle:
        handle.seek(int(reference.offset_data))
        data = handle.read(int(reference.size))
    if len(data) != int(reference.size):
        raise ValueError(
            f"short tar member read for {reference.uri}!{reference.member}: "
            f"got={len(data)} expected={reference.size}"
        )
    return data


def _load_catalog_file(path: Path) -> Dict[tuple[int, int], TarMemberRef]:
    done_path = path.with_suffix(path.suffix + ".done")
    if not path.is_file() or not done_path.is_file():
        raise FileNotFoundError(
            f"incomplete Agibot Beta tar catalog: expected {path} and {done_path}"
        )
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["task_id"]), int(row["episode_id"]))
            if key in result:
                raise ValueError(f"duplicate catalog key={key} in {path}:{line_number}")
            result[key] = TarMemberRef(
                uri=str(row["uri"]),
                member=str(row["member"]),
                offset_data=int(row["tar_offset_data"]),
                size=int(row["tar_size"]),
            )
    return result


def load_agibot_beta_tar_catalog(
    catalog_dir: str | Path, task_ids: Iterable[int]
) -> tuple[Dict[tuple[int, int], TarMemberRef], Dict[tuple[int, int], TarMemberRef]]:
    """Load the reusable physical member map produced by the old tar scan."""

    root = Path(catalog_dir)
    proprio = _load_catalog_file(root / "proprio.jsonl")
    videos = {}
    for task_id in sorted({int(value) for value in task_ids}):
        videos.update(_load_catalog_file(root / f"video_task_{task_id}.jsonl"))
    return videos, proprio


def _task_id_from_path(path: Path) -> int:
    match = re.fullmatch(r"task_(\d+)\.json", path.name)
    if not match:
        raise ValueError(f"cannot infer task_id from {path}")
    return int(match.group(1))


def _action_span(action_config: Sequence[Dict[str, Any]]) -> tuple[int, int] | None:
    starts = []
    ends = []
    for action in action_config:
        try:
            start = int(action.get("start_frame"))
            end = int(action.get("end_frame"))
        except (TypeError, ValueError):
            continue
        if end > start:
            starts.append(start)
            ends.append(end)
    if not starts:
        return None
    return min(starts), max(ends)


def scan_agibot_beta_episodes(
    data_root: str | Path,
    catalog_dir: str | Path,
    excluded_task_names: Iterable[str] = DEFAULT_EXCLUDED_TASK_NAMES,
) -> tuple[list[Dict[str, Any]], Dict[str, int]]:
    """Join raw task annotations to physical tar members, one row per episode."""

    data_root = Path(data_root)
    annotation_paths = sorted((data_root / "task_info").glob("task_*.json"))
    if not annotation_paths:
        raise ValueError(f"no Agibot Beta task_info found under {data_root / 'task_info'}")
    task_ids = [_task_id_from_path(path) for path in annotation_paths]
    video_catalog, proprio_catalog = load_agibot_beta_tar_catalog(catalog_dir, task_ids)
    stats = {
        "annotation_episodes": 0,
        "unique_annotation_episodes": 0,
        "duplicate_annotation_exact": 0,
        "missing_video": 0,
        "missing_proprio": 0,
        "missing_action_config": 0,
        "bad_action_span": 0,
        "excluded_bad_task": 0,
    }
    excluded_tasks = {str(value).strip() for value in excluded_task_names}
    annotations: Dict[tuple[int, int], Dict[str, Any]] = {}
    for annotation_path, inferred_task_id in zip(annotation_paths, task_ids):
        with annotation_path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        if not isinstance(items, list):
            raise ValueError(f"task_info must contain a list: {annotation_path}")
        for annotation_item_index, raw_item in enumerate(items):
            stats["annotation_episodes"] += 1
            item = dict(raw_item)
            task_id = int(item.get("task_id") or inferred_task_id)
            episode_id = int(item["episode_id"])
            key = (task_id, episode_id)
            location = {
                "path": str(annotation_path),
                "item_index": int(annotation_item_index),
            }
            previous = annotations.get(key)
            if previous is not None:
                if previous["item"] != item:
                    first_location = previous["locations"][0]
                    raise ValueError(
                        "conflicting duplicate Agibot Beta annotation for "
                        f"task_id={task_id} episode_id={episode_id}: "
                        f"first={first_location['path']}:{first_location['item_index']} "
                        f"duplicate={location['path']}:{location['item_index']}"
                    )
                previous["locations"].append(location)
                stats["duplicate_annotation_exact"] += 1
                continue
            annotations[key] = {"item": item, "locations": [location]}

    stats["unique_annotation_episodes"] = len(annotations)
    episodes = []
    for (task_id, episode_id), annotation in annotations.items():
        item = annotation["item"]
        locations = annotation["locations"]
        if str(item.get("task_name") or "").strip() in excluded_tasks:
            stats["excluded_bad_task"] += 1
            continue
        key = (task_id, episode_id)
        video_ref = video_catalog.get(key)
        proprio_ref = proprio_catalog.get(key)
        action_config = (item.get("label_info") or {}).get("action_config") or []
        span = _action_span(action_config)
        if video_ref is None:
            stats["missing_video"] += 1
            continue
        if proprio_ref is None:
            stats["missing_proprio"] += 1
            continue
        if not action_config:
            stats["missing_action_config"] += 1
            continue
        if span is None or span[1] <= span[0]:
            stats["bad_action_span"] += 1
            continue
        episodes.append(
            {
                "episode_uid": f"agibot_beta:{task_id}:{episode_id}",
                "task_id": task_id,
                "episode_id": episode_id,
                "task_name": item.get("task_name", ""),
                "init_scene_text": item.get("init_scene_text", ""),
                "action_config": action_config,
                "action_span": span,
                "video_ref": video_ref,
                "proprio_ref": proprio_ref,
                "annotation_path": locations[0]["path"],
                "annotation_item_index": int(locations[0]["item_index"]),
                "annotation_locations": locations,
            }
        )
    if not episodes:
        raise ValueError(f"no usable Agibot Beta episodes under {data_root}")
    return episodes, stats


def _merged_valid_start_ranges(
    action_config: Sequence[Dict[str, Any]], max_start: int
) -> list[tuple[int, int]]:
    ranges = []
    for action in action_config:
        try:
            start = max(0, int(action.get("start_frame")))
            end = min(int(max_start), int(action.get("end_frame")) - 1)
        except (TypeError, ValueError):
            continue
        if start <= end:
            ranges.append((start, end))
    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _range_value_at(ranges: Sequence[tuple[int, int]], position: int) -> int:
    cursor = int(position)
    for start, end in ranges:
        length = end - start + 1
        if cursor < length:
            return start + cursor
        cursor -= length
    raise IndexError(position)


class AgibotBetaDataset(Dataset):
    """Dynamic Agibot Beta windows with one tar read/decode per homogeneous batch."""

    produces_prebatched_batches = True

    def __init__(
        self,
        data_root: str,
        tar_catalog_dir: str,
        *,
        split: str = "train",
        video_size: Tuple[int, int] = (320, 384),
        num_video_frames: int = 48,
        action_video_fps_ratio: int = 1,
        action_dim: int = 26,
        state_dim: int = 26,
        source_dataset: str = "agibot_beta",
        source_child: str = "agibot_beta",
        robot_name: str = "agibot-a2-d",
        test_fraction: float = 0.01,
        split_seed: int = 20260720,
        virtual_samples_per_episode: int = 64,
        max_decode_retries: int = 16,
        video_decode_backend: str = "ffmpeg",
        action_magnitude_guard_threshold: float | None = 100.0,
        excluded_task_names: Sequence[str] = DEFAULT_EXCLUDED_TASK_NAMES,
        seed: int = 0,
        rank: int = 0,
    ):
        warn_episode_native_deprecated("AgibotBetaDataset")
        self.video_size = tuple(int(value) for value in video_size)
        self.num_video_frames = int(num_video_frames)
        self.frames_clip_num = self.num_video_frames + 1
        self.action_video_fps_ratio = int(action_video_fps_ratio)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.source_dataset = str(source_dataset)
        self.source_child = str(source_child)
        self.robot_name = str(robot_name)
        self.split = "test" if str(split).lower() in {"test", "val", "validation"} else "train"
        self.virtual_samples_per_episode = int(virtual_samples_per_episode)
        self.max_decode_retries = int(max_decode_retries)
        self.video_decode_backend = str(video_decode_backend)
        self.action_magnitude_guard_threshold = action_magnitude_guard_threshold
        self.seed = int(seed)
        self.rank = int(rank)
        self.epoch = 0
        if self.action_video_fps_ratio <= 0:
            raise ValueError("action_video_fps_ratio must be positive")
        if self.virtual_samples_per_episode <= 0 or self.max_decode_retries <= 0:
            raise ValueError("virtual_samples_per_episode and max_decode_retries must be positive")

        all_episodes, scan_stats = scan_agibot_beta_episodes(
            data_root, tar_catalog_dir, excluded_task_names
        )
        self.total_episode_count = len(all_episodes)
        self.scan_stats = scan_stats
        self.episodes = split_agibot_episodes(
            all_episodes, self.split, test_fraction, split_seed
        )
        if not self.episodes:
            raise ValueError(f"Agibot Beta has no episodes for split={self.split}")
        logger.info(
            "AgibotBetaDataset initialized: split=%s episodes=%d total=%d stats=%s",
            self.split,
            len(self.episodes),
            self.total_episode_count,
            self.scan_stats,
        )

    def __len__(self) -> int:
        return len(self.episodes) * self.virtual_samples_per_episode

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _logical_parts(self, logical_index: int) -> tuple[int, int]:
        return divmod(int(logical_index), self.virtual_samples_per_episode)

    def _generator(self, logical_index: int) -> torch.Generator:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        return torch.Generator().manual_seed(
            compose_seed(
                self.seed,
                self.epoch,
                self.rank,
                worker_id,
                self.source_child,
                logical_index,
            )
        )

    def _sample_start(
        self,
        episode: Dict[str, Any],
        frame_numbers: int,
        logical_slot: int,
        generator: torch.Generator,
    ) -> int:
        annotated_end = int(episode["action_span"][1])
        max_start = min(annotated_end, int(frame_numbers)) - self.frames_clip_num
        ranges = _merged_valid_start_ranges(episode["action_config"], max_start)
        total = sum(end - start + 1 for start, end in ranges)
        if total <= 0:
            raise ValueError(
                f"episode {episode['episode_uid']} has no full aligned window: "
                f"frames={frame_numbers} annotated_end={annotated_end} required={self.frames_clip_num}"
            )
        if self.split == "train":
            position = int(torch.randint(0, total, (1,), generator=generator).item())
        elif self.virtual_samples_per_episode == 1:
            position = (total - 1) // 2
        else:
            position = round(
                (total - 1) * int(logical_slot) / (self.virtual_samples_per_episode - 1)
            )
        return _range_value_at(ranges, position)

    def _make_sample(
        self,
        episode: Dict[str, Any],
        start_frame: int,
        frame_indices: list[int],
        frames_by_index: Dict[int, torch.Tensor],
        eef_actions: np.ndarray,
    ) -> Dict[str, Any]:
        frame_mask = [True] * len(frame_indices)
        initial_state, action_sequence = build_agibot_state_action(
            eef_actions,
            frame_indices,
            frame_mask,
            action_video_fps_ratio=self.action_video_fps_ratio,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        task_text = TASK_PROMPT.format(
            scene=str(episode.get("init_scene_text") or "").strip(),
            task=str(episode.get("task_name") or "").strip(),
        )
        action_text = action_text_for_span_start(
            episode["action_config"], start_frame, require_match=True
        )
        t5_text = f"{task_text}{action_text}".strip()
        frames = torch.stack([frames_by_index[index] for index in frame_indices])
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
            "action_video_fps_ratio": self.action_video_fps_ratio,
            "split": self.split,
            "metadata": {
                "annotation_path": episode["annotation_path"],
                "start_frame": start_frame,
                "action_video_fps_ratio": self.action_video_fps_ratio,
                "video_tar_uri": episode["video_ref"].uri,
                "video_tar_member": episode["video_ref"].member,
                "proprio_tar_uri": episode["proprio_ref"].uri,
                "proprio_tar_member": episode["proprio_ref"].member,
            },
        }
        validate_action_state_magnitude(
            source_dataset=self.source_dataset,
            action_sequence=action_sequence,
            initial_state=initial_state,
            manifest_id=sample["manifest_id"],
            manifest_task=sample["manifest_task"],
            threshold=self.action_magnitude_guard_threshold,
            sources={self.source_dataset},
        )
        validate_sample(sample)
        return sample

    def _load_episode_windows(
        self,
        episode_index: int,
        logical_indices: Sequence[int],
        explicit_starts: Sequence[int] | None = None,
    ) -> list[Dict[str, Any]]:
        episode = self.episodes[int(episode_index)]
        proprio_data = read_tar_member_bytes(episode["proprio_ref"])
        eef_actions = read_agibot_eef_actions(io.BytesIO(proprio_data))
        frame_numbers = int(eef_actions.shape[0] // self.action_video_fps_ratio)

        starts = []
        for position, logical_index in enumerate(logical_indices):
            _, slot = self._logical_parts(logical_index)
            start = (
                int(explicit_starts[position])
                if explicit_starts is not None
                else self._sample_start(
                    episode, frame_numbers, slot, self._generator(logical_index)
                )
            )
            starts.append(start)
        frame_lists = [
            list(range(start, start + self.frames_clip_num)) for start in starts
        ]
        unique_indices = sorted({index for values in frame_lists for index in values})

        video_data = read_tar_member_bytes(episode["video_ref"])
        suffix = Path(episode["video_ref"].member).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(
            prefix="motus_agibot_beta_", suffix=suffix
        ) as temp_video:
            temp_video.write(video_data)
            temp_video.flush()
            decoded = decode_video_frames(
                temp_video.name,
                unique_indices,
                self.video_size,
                backend=self.video_decode_backend,
            )
        frames_by_index = {
            frame_index: decoded[position]
            for position, frame_index in enumerate(unique_indices)
        }
        return [
            self._make_sample(episode, start, frame_list, frames_by_index, eef_actions)
            for start, frame_list in zip(starts, frame_lists)
        ]

    def load_window(self, episode_index: int, start_frame: int) -> Dict[str, Any]:
        logical_index = int(episode_index) * self.virtual_samples_per_episode
        return self._load_episode_windows(
            int(episode_index), [logical_index], [int(start_frame)]
        )[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get_grouped_samples([int(index)])[0]

    def __getitems__(self, indices: Sequence[int]):
        return [self.get_grouped_batch([int(index) for index in indices])]

    @staticmethod
    def collate_fn(items):
        if len(items) != 1:
            raise ValueError(f"expected one pre-batched item, got {len(items)}")
        return items[0]

    def get_grouped_samples(self, indices: Sequence[int]) -> list[Dict[str, Any]]:
        if not indices:
            raise ValueError("Agibot Beta grouped load requires indices")
        original_episode, _ = self._logical_parts(indices[0])
        if any(self._logical_parts(index)[0] != original_episode for index in indices):
            raise ValueError("Agibot Beta grouped indices must belong to one episode")
        last_error = None
        slots = [self._logical_parts(index)[1] for index in indices]
        for retry in range(self.max_decode_retries):
            episode_index = (original_episode + retry) % len(self.episodes)
            candidates = [
                episode_index * self.virtual_samples_per_episode + slot for slot in slots
            ]
            try:
                return self._load_episode_windows(episode_index, candidates)
            except Exception as error:
                last_error = error
                logger.warning(
                    "Agibot Beta failed episode=%d retry=%d/%d: %s",
                    episode_index,
                    retry + 1,
                    self.max_decode_retries,
                    error,
                )
        raise RuntimeError(
            f"Agibot Beta failed after {self.max_decode_retries} episode attempts"
        ) from last_error

    def get_grouped_batch(self, indices: Sequence[int]) -> Dict[str, Any]:
        batch = collate_samples(self.get_grouped_samples(indices))
        if batch is None:
            raise RuntimeError("Agibot Beta produced an empty grouped batch")
        return batch

    def create_batch_sampler(
        self,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        return AgibotBetaBatchSampler(
            self,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )


class AgibotBetaBatchSampler(Sampler[list[int]]):
    """Keep every batch on one episode so its tar members are read once."""

    def __init__(
        self,
        dataset: AgibotBetaDataset,
        *,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        num_replicas: int,
        rank: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.dataset.virtual_samples_per_episode < self.batch_size:
            raise ValueError(
                "Agibot Beta virtual_samples_per_episode must be >= batch_size "
                f"({self.dataset.virtual_samples_per_episode} < {self.batch_size})"
            )
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be within num_replicas")
        virtual = self.dataset.virtual_samples_per_episode
        self.batches_per_episode = (
            virtual // self.batch_size
            if self.drop_last
            else (virtual + self.batch_size - 1) // self.batch_size
        )
        total = len(self.dataset.episodes) * self.batches_per_episode
        self.num_batches = (total + self.num_replicas - 1 - self.rank) // self.num_replicas

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(
            compose_seed(self.seed, self.epoch, 0, 0, self.dataset.source_child, 0)
        )
        if self.shuffle:
            episode_order = torch.randperm(
                len(self.dataset.episodes), generator=generator
            ).tolist()
        else:
            episode_order = list(range(len(self.dataset.episodes)))
        global_position = 0
        virtual = self.dataset.virtual_samples_per_episode
        usable = virtual - virtual % self.batch_size if self.drop_last else virtual
        for batch_round in range(self.batches_per_episode):
            start = batch_round * self.batch_size
            for episode_index in episode_order:
                selected = list(range(start, min(start + self.batch_size, usable)))
                if len(selected) < self.batch_size and self.drop_last:
                    continue
                if global_position % self.num_replicas == self.rank:
                    yield [
                        episode_index * virtual + slot for slot in selected
                    ]
                global_position += 1
