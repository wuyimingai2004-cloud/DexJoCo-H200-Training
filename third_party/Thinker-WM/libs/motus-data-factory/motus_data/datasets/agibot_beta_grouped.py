"""Agibot_beta video-key grouped Manifest dataset.

This prototype keeps the public training batch format compatible with the
existing collate output, but it builds a whole batch from records sharing the
same physical mp4. The video is read/decompressed once per batch using the
union of requested frame indices.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from motus_data.collate import collate_samples
from motus_data.datasets.action_safety import ActionMagnitudeOutlier, validate_action_state_magnitude
from motus_data.datasets.manifest import MotusManifestDataset
from motus_data.deprecation import warn_episode_native_deprecated
from motus_data.monitoring.timing import elapsed, now, write_dataset_timing

collate_fn = collate_samples

_GROUPED_INDEX_CACHE_FORMAT = "agibot_beta_grouped_index_cache_v1"


class AgibotBetaGroupedManifestDataset(Dataset):
    """Batch-producing grouped dataset for Agibot_beta tar Manifest records."""

    produces_prebatched_batches = True

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        val: bool = False,
        video_size: Tuple[int, int] = (320, 384),
        num_video_frames: int = 48,
        video_action_freq_ratio: int = 1,
        action_dim: int = 26,
        state_dim: int = 26,
        source_dataset: str = "agibot_beta",
        robot_name: str = "agibot-a2-d",
        instruction_lang: str = "default",
        max_samples: Optional[int] = None,
        max_decode_retries: int = 256,
        action_magnitude_guard_threshold: Optional[float] = None,
        action_magnitude_guard_sources: Optional[Iterable[str]] = None,
        grouped_index_cache_path: Optional[str] = None,
    ):
        warn_episode_native_deprecated("AgibotBetaGroupedManifestDataset")
        cache = self._load_grouped_index_cache(
            grouped_index_cache_path=grouped_index_cache_path,
            manifest_path=manifest_path,
            split=split,
            val=val,
            max_samples=max_samples,
        )
        self.base = MotusManifestDataset(
            manifest_path=manifest_path,
            split=split,
            val=val,
            video_size=video_size,
            num_video_frames=num_video_frames,
            video_action_freq_ratio=video_action_freq_ratio,
            action_dim=action_dim,
            state_dim=state_dim,
            source_dataset=source_dataset,
            robot_name=robot_name,
            instruction_lang=instruction_lang,
            max_samples=max_samples,
            max_decode_retries=max_decode_retries,
            action_magnitude_guard_threshold=action_magnitude_guard_threshold,
            action_magnitude_guard_sources=action_magnitude_guard_sources,
            prebuilt_offsets=None if cache is None else cache["offsets"],
            prebuilt_stats=None if cache is None else cache.get("stats"),
            prebuilt_drop_reasons=None if cache is None else cache.get("drop_reasons"),
        )
        self._groups_by_video_key: Dict[str, List[int]] = defaultdict(list)
        self._video_keys: List[str] = []
        self._video_key_pos: Dict[str, int] = {}
        self.grouped_index_cache_path = grouped_index_cache_path
        self.action_magnitude_guard_threshold = action_magnitude_guard_threshold
        self.action_magnitude_guard_sources = (
            None if action_magnitude_guard_sources is None else set(action_magnitude_guard_sources)
        )
        if cache is None:
            self._build_video_groups()
        else:
            self._groups_by_video_key = defaultdict(list, cache["groups_by_video_key"])
            self._video_keys = list(cache.get("video_keys") or sorted(self._groups_by_video_key))
            self._video_key_pos = {key: pos for pos, key in enumerate(self._video_keys)}
        self._close_base_manifest_file()

    @staticmethod
    def _manifest_signature(manifest_path: str) -> Dict[str, Any]:
        path = Path(manifest_path).resolve()
        stat = path.stat()
        return {
            "manifest_path": str(path),
            "manifest_size": int(stat.st_size),
            "manifest_mtime_ns": int(stat.st_mtime_ns),
        }

    @classmethod
    def _load_grouped_index_cache(
        cls,
        grouped_index_cache_path: Optional[str],
        manifest_path: str,
        split: str,
        val: bool,
        max_samples: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not grouped_index_cache_path:
            return None
        cache_path = Path(grouped_index_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Agibot_beta grouped index cache not found: {cache_path}")

        with cache_path.open("rb") as f:
            cache = pickle.load(f)

        if cache.get("format") != _GROUPED_INDEX_CACHE_FORMAT:
            raise ValueError(
                f"Unsupported Agibot_beta grouped index cache format: {cache.get('format')}"
            )

        expected = cls._manifest_signature(manifest_path)
        mismatches = []
        for key in ("manifest_path", "manifest_size", "manifest_mtime_ns"):
            if cache.get(key) != expected[key]:
                mismatches.append(f"{key}: cache={cache.get(key)!r} current={expected[key]!r}")
        cache_max_samples = cache.get("max_samples")
        expected_max_samples = None if max_samples is None else int(max_samples)
        if cache.get("split") != str(split):
            mismatches.append(f"split: cache={cache.get('split')!r} current={str(split)!r}")
        if bool(cache.get("val", False)) != bool(val):
            mismatches.append(f"val: cache={cache.get('val')!r} current={bool(val)!r}")
        if cache_max_samples != expected_max_samples:
            mismatches.append(
                f"max_samples: cache={cache_max_samples!r} current={expected_max_samples!r}"
            )
        if mismatches:
            raise ValueError(
                "Agibot_beta grouped index cache does not match current manifest/config: "
                + "; ".join(mismatches)
            )

        required_keys = ("offsets", "groups_by_video_key")
        missing = [key for key in required_keys if key not in cache]
        if missing:
            raise ValueError(f"Agibot_beta grouped index cache missing keys: {missing}")

        return cache

    def __getstate__(self):
        state = self.__dict__.copy()
        base = state.get("base")
        if base is not None:
            try:
                base_state = base.__getstate__()
                base.__dict__.update(base_state)
            except AttributeError:
                pass
        return state

    def _close_base_manifest_file(self) -> None:
        fp = getattr(self.base, "_fp", None)
        if fp is not None:
            try:
                fp.close()
            except OSError:
                pass
            self.base._fp = None

    def close(self) -> None:
        self.base.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_video_groups(self) -> None:
        for index, offset in enumerate(self.base._offsets):
            record = self.base._read_record_at(offset)
            video = record.get("video") or {}
            video_uri = self.base._video_uri(video)
            video_backend = self.base._storage_backend(video, default="file")
            key = self.base._video_key(video, video_uri=video_uri, video_backend=video_backend)
            self._groups_by_video_key[key].append(index)
        self._video_keys = sorted(self._groups_by_video_key)
        self._video_key_pos = {key: pos for pos, key in enumerate(self._video_keys)}

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def collate_fn(items):
        if len(items) != 1:
            raise ValueError(f"Expected one pre-batched grouped item, got {len(items)}")
        return items[0]

    def create_batch_sampler(
        self,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        return AgibotBetaGroupedBatchSampler(
            self,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )

    def __getitem__(self, indices):
        if isinstance(indices, (list, tuple)):
            return self.get_grouped_batch([int(index) for index in indices])
        return self.base[int(indices)]

    def __getitems__(self, indices):
        return [self.get_grouped_batch([int(index) for index in indices])]

    def get_grouped_batch(self, indices: List[int]) -> Dict[str, Any]:
        last_error = None
        attempted_keys = set()
        bad_tasks = set()
        for attempt in range(self.base.max_decode_retries):
            candidate_indices = (
                indices
                if attempt == 0
                else self._replacement_grouped_indices(indices, attempt, attempted_keys, bad_tasks)
            )
            if not candidate_indices:
                break
            try:
                return self._make_grouped_batch_once(candidate_indices)
            except ActionMagnitudeOutlier as exc:
                last_error = exc
                manifest_task = getattr(exc, "details", {}).get("manifest_task")
                if manifest_task:
                    bad_tasks.add(str(manifest_task))
                write_dataset_timing(
                    "agibot_beta_action_magnitude_outlier",
                    {
                        **getattr(exc, "details", {}),
                        "requested_indices": [int(index) for index in indices[:8]],
                        "candidate_indices": [int(index) for index in candidate_indices[:8]],
                        "attempt": int(attempt + 1),
                        "max_decode_retries": int(self.base.max_decode_retries),
                        "outlier_error": str(exc),
                    },
                    threshold_seconds=0.0,
                )
                continue
            except Exception as exc:
                last_error = exc
                bad_video_keys, candidate_manifest_ids, candidate_tasks = self._mark_grouped_video_failure(
                    candidate_indices
                )
                attempted_keys.update(bad_video_keys)
                write_dataset_timing(
                    "agibot_beta_grouped_sample_failed",
                    {
                        "requested_indices": [int(index) for index in indices[:8]],
                        "candidate_indices": [int(index) for index in candidate_indices[:8]],
                        "candidate_manifest_ids": candidate_manifest_ids,
                        "candidate_tasks": candidate_tasks,
                        "bad_video_keys": bad_video_keys[:8],
                        "attempt": int(attempt + 1),
                        "max_decode_retries": int(self.base.max_decode_retries),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                    threshold_seconds=0.0,
                )
                continue
        raise RuntimeError(
            f"Failed to build Agibot_beta grouped batch after {self.base.max_decode_retries} attempts: {last_error}"
        )

    def _mark_grouped_video_failure(self, indices: List[int]) -> Tuple[List[str], List[Any], List[Any]]:
        bad_video_keys = []
        candidate_manifest_ids = []
        candidate_tasks = []
        known_bad_video_uris = getattr(self.base, "_bad_video_uris", None)
        for index in indices:
            try:
                record = self.base._read_record_at(self.base._offsets[int(index)])
                candidate_manifest_ids.append(record.get("id"))
                candidate_tasks.append(record.get("task"))
                video = record.get("video") or {}
                video_uri = self.base._video_uri(video)
                video_backend = self.base._storage_backend(video, default="file")
                video_key = self.base._video_key(
                    video,
                    video_uri=video_uri,
                    video_backend=video_backend,
                )
                if video_key and video_key not in bad_video_keys:
                    bad_video_keys.append(video_key)
                    if known_bad_video_uris is not None:
                        known_bad_video_uris.add(video_key)
            except Exception:
                continue
        return bad_video_keys, candidate_manifest_ids, candidate_tasks

    def _replacement_grouped_indices(
        self,
        original_indices: List[int],
        attempt: int,
        attempted_keys: set,
        bad_tasks: Optional[set] = None,
    ) -> Optional[List[int]]:
        if not original_indices or not self._video_keys:
            return None
        bad_tasks = bad_tasks or set()
        batch_size = len(original_indices)
        try:
            record = self.base._read_record_at(self.base._offsets[int(original_indices[0])])
            video = record.get("video") or {}
            key = self.base._video_key(
                video,
                video_uri=self.base._video_uri(video),
                video_backend=self.base._storage_backend(video, default="file"),
            )
            start_pos = self._video_key_pos.get(key, 0)
        except Exception:
            start_pos = 0

        for delta in range(attempt, attempt + len(self._video_keys)):
            key = self._video_keys[(start_pos + delta) % len(self._video_keys)]
            if key in attempted_keys:
                continue
            if key in getattr(self.base, "_bad_video_uris", set()):
                attempted_keys.add(key)
                continue
            group_indices = self._groups_by_video_key.get(key) or []
            if len(group_indices) < batch_size:
                continue
            if bad_tasks:
                try:
                    candidate_record = self.base._read_record_at(self.base._offsets[int(group_indices[0])])
                    candidate_task = str(candidate_record.get("task") or "")
                    if candidate_task in bad_tasks:
                        attempted_keys.add(key)
                        continue
                except Exception:
                    pass
            attempted_keys.add(key)
            max_start = len(group_indices) - batch_size
            offset = (attempt * batch_size) % (max_start + 1)
            return [int(index) for index in group_indices[offset : offset + batch_size]]
        return None

    def _make_grouped_batch_once(self, indices: List[int]) -> Dict[str, Any]:
        if not indices:
            raise ValueError("Grouped batch requires at least one index")

        total_start = now()
        records = [self.base._read_record_at(self.base._offsets[index]) for index in indices]
        video_keys = [
            self.base._video_key(
                record.get("video") or {},
                video_uri=self.base._video_uri(record.get("video") or {}),
                video_backend=self.base._storage_backend(record.get("video") or {}, default="file"),
            )
            for record in records
        ]
        if len(set(video_keys)) != 1:
            raise ValueError(f"Grouped batch received multiple video_keys: {video_keys}")

        video = records[0]["video"]
        video_backend = self.base._storage_backend(video, default="file")
        if not self.base._is_tar_backend(video_backend):
            samples = [self.base._make_sample(index) for index in indices]
            batch = collate_fn(samples)
            if batch is None:
                raise ValueError("Grouped fallback collate returned None")
            return batch

        frame_lists = [self.base._video_frame_indices(record["video"]) for record in records]
        if any(len(frame_list) != self.base.frames_clip_num for frame_list in frame_lists):
            raise ValueError("All grouped records must have fixed frame_indices")
        unique_indices = sorted({frame for frame_list in frame_lists for frame in frame_list})
        resize_mode = str(video.get("resize_mode") or "interpolate").lower()

        preloaded_samples = []
        for record in records:
            initial_state, action_sequence = self.base._load_state_action(record)
            t5_text, vlm_text = self.base._format_text(record)
            robot = record.get("robot") or {}
            loss = record.get("loss") or {}
            source_dataset = self.base._source_dataset(record)
            validate_action_state_magnitude(
                source_dataset=source_dataset,
                action_sequence=action_sequence,
                initial_state=initial_state,
                manifest_id=record.get("id"),
                manifest_task=record.get("task"),
                threshold=self.action_magnitude_guard_threshold,
                sources=self.action_magnitude_guard_sources,
            )
            preloaded_samples.append(
                {
                    "initial_state": initial_state,
                    "action_sequence": action_sequence,
                    "t5_text": t5_text,
                    "vlm_text": vlm_text,
                    "robot": robot,
                    "loss": loss,
                    "source_dataset": source_dataset,
                    "robot_name": self.base._robot_name(record),
                }
            )

        timing = {
            "source_dataset": "agibot_beta",
            "video_key": video_keys[0],
            "batch_size": len(indices),
            "requested_frames": sum(len(frame_list) for frame_list in frame_lists),
            "unique_frames": len(unique_indices),
            "frame_reduction": (
                1.0 - len(unique_indices) / max(sum(len(frame_list) for frame_list in frame_lists), 1)
            ),
        }

        data = self.base._read_tar_member_bytes(video, timing=timing, timing_prefix="grouped_video_tar")
        suffix = Path(str(video.get("member") or "video.mp4")).suffix or ".mp4"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="motus_grouped_tar_video_", suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            frame_map = self.base._load_video_frame_map_ffmpeg(tmp_path, unique_indices, timing=timing)
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        samples = []
        for record, frame_list, preloaded in zip(records, frame_lists, preloaded_samples):
            frames_np = [frame_map[int(frame)].copy() for frame in frame_list]
            frames = torch.from_numpy(np.stack(frames_np, axis=0)).permute(0, 3, 1, 2).float() / 255.0
            frames = self.base._resize_frames(frames, resize_mode=resize_mode)

            initial_state = preloaded["initial_state"]
            action_sequence = preloaded["action_sequence"]
            t5_text = preloaded["t5_text"]
            vlm_text = preloaded["vlm_text"]
            robot = preloaded["robot"]
            loss = preloaded["loss"]
            source_dataset = preloaded["source_dataset"]
            first_frame = frames[0]
            samples.append(
                {
                    "first_frame": first_frame,
                    "video_frames": frames[1:],
                    "initial_state": initial_state,
                    "action_sequence": action_sequence,
                    "first_frame_pil": self.base._tensor_to_pil(first_frame),
                    "t5_input_message": t5_text,
                    "vlm_input_message": vlm_text,
                    "robot_name": preloaded["robot_name"],
                    "source_dataset": source_dataset,
                    "action_loss_weight": float(loss.get("action_weight", 1.0)),
                    "video_loss_weight": float(loss.get("video_weight", 1.0)),
                    "state_action_policy": str(robot.get("action_policy") or "real"),
                    "manifest_id": record.get("id"),
                    "manifest_task": record.get("task"),
                }
            )

        batch = collate_fn(samples)
        if batch is None:
            raise ValueError("Grouped collate returned None")
        timing["total_seconds"] = elapsed(total_start)
        timing["manifest_ids"] = [sample.get("manifest_id") for sample in samples[:8]]
        write_dataset_timing("agibot_beta_grouped", timing)
        return batch


class AgibotBetaGroupedBatchSampler(Sampler[List[int]]):
    """Yield batches whose indices all share the same video_key."""

    def __init__(
        self,
        dataset: AgibotBetaGroupedManifestDataset,
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
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        total_batches = 0
        for indices in self.dataset._groups_by_video_key.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        self.num_batches = max(1, (total_batches + self.num_replicas - 1) // self.num_replicas)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + self.rank)
        batches: List[List[int]] = []
        video_keys = list(self.dataset._video_keys)
        if self.shuffle:
            rng.shuffle(video_keys)

        for key in video_keys:
            indices = list(self.dataset._groups_by_video_key[key])
            # Keep per-video records in manifest order so adjacent sliding
            # windows land in the same batch and share decoded frames.
            end = len(indices) - (len(indices) % self.batch_size if self.drop_last else 0)
            for start in range(0, end, self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        for pos, batch in enumerate(batches):
            if pos % self.num_replicas == self.rank:
                yield batch
