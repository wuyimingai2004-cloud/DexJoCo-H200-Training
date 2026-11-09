"""Direct RobotMind HDF5 datasets with runtime multimodal alignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import h5py
except ImportError:
    h5py = None

from motus_data.datasets.agibot_common import tensor_to_pil
from motus_data.datasets.episode_base import EpisodeDataset
from motus_data.deprecation import warn_episode_native_deprecated
from motus_data.sampling import ContiguousWindowSampler, EpisodeRange, WindowSelection


logger = logging.getLogger(__name__)

DEFAULT_RGB_KEY = "camera_observations/color_images/camera_top"


@dataclass(frozen=True)
class RobotMindEmbodimentSchema:
    """Physical HDF5 schema for one RobotMind embodiment."""

    name: str
    robot_name: str
    state_action_keys: Tuple[Tuple[str, str], ...]
    state_action_dim: int
    rgb_key: str = DEFAULT_RGB_KEY


ROBOTMIND_SCHEMAS: Dict[str, RobotMindEmbodimentSchema] = {
    "tianyi": RobotMindEmbodimentSchema(
        name="tianyi",
        robot_name="robotmind-tianyi",
        state_action_keys=(
            ("left_arm", "puppet/arm_left_position_align/data"),
            ("right_arm", "puppet/arm_right_position_align/data"),
            ("left_gripper", "puppet/end_effector_left_position_align/data"),
            ("right_gripper", "puppet/end_effector_right_position_align/data"),
        ),
        state_action_dim=16,
    ),
    "ark": RobotMindEmbodimentSchema(
        name="ark",
        robot_name="robotmind-ark",
        state_action_keys=(
            ("left_arm", "puppet/arm_left_position_align/data"),
            ("left_gripper", "puppet/end_effector_left_position_align/data"),
            ("right_arm", "puppet/arm_right_position_align/data"),
            ("right_gripper", "puppet/end_effector_right_position_align/data"),
        ),
        state_action_dim=14,
    ),
    "agilex": RobotMindEmbodimentSchema(
        name="agilex",
        robot_name="robotmind-agilex",
        state_action_keys=(
            ("left_arm", "puppet/arm_left_position_align/data"),
            ("left_gripper", "puppet/end_effector_left_position_align/data"),
            ("right_arm", "puppet/arm_right_position_align/data"),
            ("right_gripper", "puppet/end_effector_right_position_align/data"),
        ),
        state_action_dim=14,
        rgb_key="camera_observations/color_images/camera_front",
    ),
    "arkmobile": RobotMindEmbodimentSchema(
        name="arkmobile",
        robot_name="robotmind-arkmobile",
        state_action_keys=(
            ("left_arm", "puppet/arm_left_position_align/data"),
            ("right_arm", "puppet/arm_right_position_align/data"),
            ("left_gripper", "puppet/end_effector_left_position_align/data"),
            ("right_gripper", "puppet/end_effector_right_position_align/data"),
            ("chassis_twist", "puppet/chassis_twist_align/data"),
        ),
        state_action_dim=20,
    ),
    "tienkung": RobotMindEmbodimentSchema(
        name="tienkung",
        robot_name="robotmind-tienkung",
        state_action_keys=(
            ("left_arm", "puppet/arm_left_position_align/data"),
            ("right_arm", "puppet/arm_right_position_align/data"),
            ("left_gripper", "puppet/end_effector_left_position_align/data"),
            ("right_gripper", "puppet/end_effector_right_position_align/data"),
        ),
        state_action_dim=16,
    ),
}


def register_robotmind_schema(schema: RobotMindEmbodimentSchema, *, replace: bool = False) -> None:
    key = str(schema.name).strip().lower()
    if not key:
        raise ValueError("RobotMind schema name cannot be empty")
    if key in ROBOTMIND_SCHEMAS and not replace:
        raise KeyError(f"RobotMind schema already registered: {key}")
    ROBOTMIND_SCHEMAS[key] = schema


def get_robotmind_schema(name: str) -> RobotMindEmbodimentSchema:
    key = str(name).strip().lower().replace("robotmind-", "")
    try:
        return ROBOTMIND_SCHEMAS[key]
    except KeyError as error:
        raise KeyError(
            f"unknown RobotMind embodiment={name!r}; available={sorted(ROBOTMIND_SCHEMAS)}"
        ) from error


@dataclass(frozen=True)
class RobotMindEpisode:
    episode_uid: str
    task_name: str
    episode_name: str
    hdf5_path: str
    zh_instruction: str


@dataclass(frozen=True)
class RobotMindEpisodeInfo:
    total_frames: int
    en_instruction: str
    field_dims: Tuple[int, ...]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def scan_robotmind_episodes(
    data_root: str | Path,
    embodiment: str,
    *,
    fail_on_missing_hdf5: bool = False,
    max_episodes: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[RobotMindEpisode]:
    """Scan source-native episode paths without constructing windows or indices."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"RobotMind data_root is not a directory: {root}")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive")
    schema = get_robotmind_schema(embodiment)
    episodes = []
    task_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    total_tasks = len(task_dirs)
    for task_position, task_dir in enumerate(task_dirs, start=1):
        success_dir = task_dir / "success_episodes"
        if not success_dir.is_dir():
            continue
        zh_path = task_dir / "zh_description.txt"
        zh_instruction = (
            zh_path.read_text(encoding="utf-8").strip() if zh_path.is_file() else ""
        )
        for episode_dir in sorted(path for path in success_dir.iterdir() if path.is_dir()):
            hdf5_path = episode_dir / "data" / "trajectory.hdf5"
            if not hdf5_path.is_file():
                if fail_on_missing_hdf5:
                    raise FileNotFoundError(
                        f"RobotMind Episode is missing trajectory.hdf5: {episode_dir}"
                    )
                continue
            uid = f"robotmind:{schema.name}:{task_dir.name}:{episode_dir.name}"
            episodes.append(
                RobotMindEpisode(
                    episode_uid=uid,
                    task_name=task_dir.name,
                    episode_name=episode_dir.name,
                    hdf5_path=str(hdf5_path),
                    zh_instruction=zh_instruction,
                )
            )
            if max_episodes is not None and len(episodes) >= int(max_episodes):
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "scan",
                            "processed": task_position,
                            "total": total_tasks,
                            "percent": 100.0,
                            "discovered_episodes": len(episodes),
                            "done": True,
                        }
                    )
                return episodes
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "scan",
                    "processed": task_position,
                    "total": total_tasks,
                    "percent": 100.0 * task_position / max(1, total_tasks),
                    "discovered_episodes": len(episodes),
                }
            )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "scan",
                "processed": total_tasks,
                "total": total_tasks,
                "percent": 100.0,
                "discovered_episodes": len(episodes),
                "done": True,
            }
        )
    return episodes


def _split_score(episode_uid: str, seed: int) -> int:
    payload = f"{int(seed)}:{episode_uid}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def split_robotmind_episodes(
    episodes: Sequence[RobotMindEpisode], split: str, test_fraction: float, seed: int
) -> list[RobotMindEpisode]:
    if not 0.0 <= float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    ordered = sorted(
        episodes,
        key=lambda item: (_split_score(item.episode_uid, seed), item.episode_uid),
    )
    count = int(round(len(ordered) * float(test_fraction)))
    if test_fraction > 0 and ordered:
        count = max(1, count)
    test_ids = {episode.episode_uid for episode in ordered[:count]}
    want_test = str(split).lower() in {"test", "val", "validation"}
    return sorted(
        [episode for episode in episodes if (episode.episode_uid in test_ids) == want_test],
        key=lambda item: item.episode_uid,
    )


class RobotMindDataset(EpisodeDataset):
    """Generic RobotMind family Dataset over source HDF5 episodes."""

    def __init__(
        self,
        data_root: str,
        embodiment: str,
        *,
        split: str = "train",
        video_size: Tuple[int, int] = (480, 640),
        num_video_frames: int = 48,
        video_frame_step: int = 1,
        action_video_fps_ratio: int = 1,
        instruction_lang: str = "en",
        source_dataset: str = "robotmind",
        source_child: str | None = None,
        robot_name: str | None = None,
        test_fraction: float = 0.01,
        split_seed: int = 20260720,
        virtual_samples_per_episode: int = 8,
        max_decode_retries: int = 16,
        resize_mode: str = "interpolate",
        video_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        seed: int = 0,
        rank: int = 0,
    ):
        warn_episode_native_deprecated("RobotMindDataset")
        if h5py is None:
            raise ImportError("RobotMindDataset requires h5py")
        self.schema = get_robotmind_schema(embodiment)
        self.data_root = str(data_root)
        self.video_size = tuple(int(value) for value in video_size)
        self.num_video_frames = int(num_video_frames)
        self.video_frame_step = int(video_frame_step)
        self.action_video_fps_ratio = int(action_video_fps_ratio)
        self.instruction_lang = str(instruction_lang).lower()
        self.source_dataset = str(source_dataset)
        self.source_child = str(source_child or f"robotmind_{self.schema.name}")
        self.robot_name = str(robot_name or self.schema.robot_name)
        self.resize_mode = str(resize_mode).lower()
        self.video_loss_weight = float(video_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        self._episode_info: Dict[str, RobotMindEpisodeInfo] = {}
        if self.num_video_frames <= 0:
            raise ValueError("num_video_frames must be positive")
        if self.video_frame_step <= 0 or self.action_video_fps_ratio <= 0:
            raise ValueError("video_frame_step and action_video_fps_ratio must be positive")
        if self.video_frame_step % self.action_video_fps_ratio != 0:
            raise ValueError(
                "video_frame_step must be divisible by action_video_fps_ratio so video and action "
                "share exact HDF5 frame indices"
            )
        self.action_frame_step = self.video_frame_step // self.action_video_fps_ratio
        if self.instruction_lang not in {"en", "zh", "both"}:
            raise ValueError("instruction_lang must be en, zh, or both")
        if self.resize_mode not in {"interpolate", "bilinear"}:
            raise ValueError("RobotMind direct Dataset currently supports interpolate/bilinear resize")

        all_episodes = scan_robotmind_episodes(data_root, self.schema.name)
        self.total_episode_count = len(all_episodes)
        normalized_split = (
            "test" if str(split).lower() in {"test", "val", "validation"} else "train"
        )
        selected = split_robotmind_episodes(
            all_episodes, normalized_split, test_fraction, split_seed
        )
        super().__init__(
            selected,
            dataset_name=self.source_child,
            sampler=ContiguousWindowSampler(
                num_frames=self.num_video_frames + 1,
                frame_step=self.video_frame_step,
                random_train=True,
            ),
            split=normalized_split,
            seed=seed,
            rank=rank,
            virtual_samples_per_episode=virtual_samples_per_episode,
            max_load_retries=max_decode_retries,
        )
        self._path_to_index = {
            str(Path(episode.hdf5_path)): index for index, episode in enumerate(self.episodes)
        }
        logger.info(
            "RobotMindDataset initialized: embodiment=%s split=%s episodes=%d total=%d "
            "virtual_samples_per_episode=%d video_frame_step=%d ratio=%d language=%s",
            self.schema.name,
            self.split,
            len(self.episodes),
            self.total_episode_count,
            self.virtual_samples_per_episode,
            self.video_frame_step,
            self.action_video_fps_ratio,
            self.instruction_lang,
        )

    def _inspect_episode(self, episode: RobotMindEpisode) -> RobotMindEpisodeInfo:
        cached = self._episode_info.get(episode.episode_uid)
        if cached is not None:
            return cached
        required = [self.schema.rgb_key, *[path for _, path in self.schema.state_action_keys]]
        with h5py.File(episode.hdf5_path, "r") as handle:
            missing = [key for key in required if key not in handle]
            if missing:
                raise KeyError(f"missing RobotMind keys={missing} in {episode.hdf5_path}")
            lengths = [int(handle[key].shape[0]) for key in required]
            metadata_length = lengths[0]
            en_instruction = ""
            if "metadata" in handle:
                metadata_length = int(
                    handle["metadata"].attrs.get("trajectory_length", metadata_length)
                )
                en_instruction = _clean_text(
                    handle["metadata"].attrs.get("language_instruction", "")
                )
            total_frames = min([*lengths, metadata_length])
            field_dims = tuple(
                int(np.prod(handle[key].shape[1:])) if len(handle[key].shape) > 1 else 1
                for _, key in self.schema.state_action_keys
            )
        if total_frames < self.sampler.span:
            raise ValueError(
                f"episode {episode.episode_uid} has {total_frames} aligned frames, "
                f"requires {self.sampler.span}"
            )
        if sum(field_dims) != self.schema.state_action_dim:
            raise ValueError(
                f"RobotMind {self.schema.name} dimension mismatch in {episode.hdf5_path}: "
                f"fields={field_dims} total={sum(field_dims)} expected={self.schema.state_action_dim}"
            )
        info = RobotMindEpisodeInfo(total_frames, en_instruction, field_dims)
        self._episode_info[episode.episode_uid] = info
        return info

    def episode_range(self, episode: RobotMindEpisode) -> EpisodeRange:
        info = self._inspect_episode(episode)
        return EpisodeRange(episode.episode_uid, 0, info.total_frames - 1)

    def _instruction(self, episode: RobotMindEpisode) -> str:
        info = self._inspect_episode(episode)
        zh = _clean_text(episode.zh_instruction)
        en = _clean_text(info.en_instruction)
        if self.instruction_lang == "zh":
            result = zh or en
        elif self.instruction_lang == "both":
            result = f"{zh} / {en}" if zh and en else zh or en
        else:
            result = en or zh
        if not result:
            raise ValueError(f"RobotMind episode has no usable instruction: {episode.episode_uid}")
        return result

    @staticmethod
    def _decode_jpeg_frames(dataset: Any, indices: Iterable[int]) -> torch.Tensor:
        frames = []
        for index in indices:
            raw = bytes(dataset[int(index)])
            with Image.open(io.BytesIO(raw)) as image:
                frames.append(np.asarray(image.convert("RGB")))
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

    @staticmethod
    def _read_rows(dataset: Any, indices: Sequence[int]) -> np.ndarray:
        rows = np.asarray(dataset[list(indices)])
        if rows.ndim == 1:
            rows = rows[:, None]
        return rows.reshape(len(indices), -1)

    def load_aligned_sample(
        self, episode: RobotMindEpisode, selection: WindowSelection
    ) -> Dict[str, Any]:
        info = self._inspect_episode(episode)
        video_indices = tuple(int(index) for index in selection.frame_indices)
        first_index = video_indices[0]
        last_index = video_indices[-1]
        action_indices = tuple(
            range(first_index + self.action_frame_step, last_index + 1, self.action_frame_step)
        )
        expected_actions = self.num_video_frames * self.action_video_fps_ratio
        if len(action_indices) != expected_actions:
            raise ValueError(
                f"aligned action length mismatch: got={len(action_indices)} expected={expected_actions}"
            )
        if last_index >= info.total_frames:
            raise IndexError(
                f"selection exceeds aligned timeline: last={last_index} total={info.total_frames}"
            )

        with h5py.File(episode.hdf5_path, "r") as handle:
            frames = self._decode_jpeg_frames(handle[self.schema.rgb_key], video_indices)
            state_parts = []
            action_parts = []
            for _, key in self.schema.state_action_keys:
                dataset = handle[key]
                state_parts.append(self._read_rows(dataset, [first_index])[0])
                action_parts.append(self._read_rows(dataset, action_indices))

        frames = F.interpolate(
            frames, size=self.video_size, mode="bilinear", align_corners=False
        )
        initial_state = torch.from_numpy(np.concatenate(state_parts)).float()
        action_sequence = torch.from_numpy(np.concatenate(action_parts, axis=-1)).float()
        if initial_state.numel() != self.schema.state_action_dim:
            raise ValueError("RobotMind initial_state dimension changed after HDF5 read")
        if tuple(action_sequence.shape) != (expected_actions, self.schema.state_action_dim):
            raise ValueError(
                f"RobotMind action shape={tuple(action_sequence.shape)} expected="
                f"{(expected_actions, self.schema.state_action_dim)}"
            )

        target = self._instruction(episode)
        sample_id = f"{episode.episode_uid}:start:{first_index:06d}"
        return {
            "first_frame": frames[0],
            "video_frames": frames[1:],
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "first_frame_pil": tensor_to_pil(frames[0]),
            "t5_input_message": f"Task: {target}.\n",
            "vlm_input_message": (
                "Based on the following situation, please predict what the robot will do: \n"
                + target
            ),
            "robot_name": self.robot_name,
            "source_dataset": self.source_dataset,
            "source_child": self.source_child,
            "action_loss_weight": self.action_loss_weight,
            "video_loss_weight": self.video_loss_weight,
            "state_action_policy": "real",
            "manifest_id": sample_id,
            "manifest_task": episode.task_name,
            "episode_id": episode.episode_uid,
            "frame_indices": list(video_indices),
            "action_video_fps_ratio": self.action_video_fps_ratio,
        }

    def load_window(self, episode_index: int, start_frame: int) -> Dict[str, Any]:
        """Load an explicit window for parity/debugging; training remains dynamic."""

        episode = self.episodes[int(episode_index)]
        episode_range = self.episode_range(episode)
        indices = tuple(
            int(start_frame) + offset * self.video_frame_step
            for offset in range(self.num_video_frames + 1)
        )
        if indices[-1] > episode_range.last_frame:
            raise ValueError(
                f"explicit window exceeds episode: last={indices[-1]} "
                f"available={episode_range.last_frame}"
            )
        return self.load_aligned_sample(
            episode,
            WindowSelection(episode.episode_uid, int(start_frame), indices),
        )

    def episode_index_for_path(self, hdf5_path: str | Path) -> int:
        normalized = str(Path(hdf5_path))
        if normalized not in self._path_to_index:
            raise KeyError(f"episode path is not in split={self.split}: {normalized}")
        return self._path_to_index[normalized]
