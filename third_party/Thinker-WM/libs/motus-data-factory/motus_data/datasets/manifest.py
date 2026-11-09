import json
import logging
import io
import os
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from motus_data.datasets.action_safety import ActionMagnitudeOutlier, validate_action_state_magnitude
from motus_data.deprecation import warn_episode_native_deprecated
from motus_data.monitoring.timing import elapsed, now, write_dataset_timing

try:
    import decord
except ImportError:
    decord = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

_DEFAULT_FFMPEG_BIN = "/code1/7TB/v2/bin/ffmpeg-static/bin/ffmpeg"

if decord is not None:
    decord.bridge.set_bridge("torch")

_TRAIN_SPLITS = {"train", "trainval", "all"}
_VAL_SPLITS = {"val"}
_TEST_SPLITS = {"test"}
_ROBOTMIND_STATE_ACTION_BACKENDS = {
    "robotmind_hdf5",
    "robotmind_tianyi_hdf5",
    "robotmind_ark_hdf5",
    "robotmind_arkmobile_hdf5",
    "robotmind_tienkung_hdf5",
}
_ROBOTMIND_DEFAULT_STATE_ACTION_KEYS = {
    "left_arm": "puppet/arm_left_position_align/data",
    "right_arm": "puppet/arm_right_position_align/data",
    "left_gripper": "puppet/end_effector_left_position_align/data",
    "right_gripper": "puppet/end_effector_right_position_align/data",
}
_ROBOTMIND_DEFAULT_STATE_ACTION_ORDER = (
    "left_arm",
    "right_arm",
    "left_gripper",
    "right_gripper",
)
_AGIBOT_ACTION_MEAN = torch.tensor([
    0.6623259, -0.02415569, 0.65893349, -0.91274923, 0.93820975, -0.62816762,
    64.11719045790569,
    -0.01470473, 0.27863706, 0.4925647, 0.47402732, 0.51696799, -1.0560014
], dtype=torch.float32)
_AGIBOT_ACTION_STD = torch.tensor([
    0.08038633, 0.3268345, 0.19179442, 1.23554225, 1.35022756, 0.92738767,
    37.00465005224069,
    2.12378374e-01, 3.09977879e-01, 4.75664546e-01, 4.69941935e-01,
    4.90734943e-01, 1.17136790e-04
], dtype=torch.float32)


class MotusManifestDataset(Dataset):
    """Generic Motus Manifest v1 dataset.

    The manifest is a JSONL file where each line describes one logical
    training window. This dataset converts a manifest record into the same
    sample dictionary used by the existing Robotwin/Agibot/Ego4D datasets.
    """

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
        source_dataset: Optional[str] = None,
        robot_name: Optional[str] = None,
        instruction_lang: str = "default",
        max_samples: Optional[int] = None,
        max_decode_retries: int = 256,
        action_magnitude_guard_threshold: Optional[float] = None,
        action_magnitude_guard_sources: Optional[Iterable[str]] = None,
        prebuilt_offsets: Optional[Sequence[int]] = None,
        prebuilt_stats: Optional[Dict[str, int]] = None,
        prebuilt_drop_reasons: Optional[Dict[str, int]] = None,
    ):
        warn_episode_native_deprecated("MotusManifestDataset")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Motus manifest file not found: {self.manifest_path}")

        self.split = "val" if val else str(split).lower()
        if val or self.split == "val":
            self.allowed_splits = _VAL_SPLITS
        elif self.split == "test":
            self.allowed_splits = _TEST_SPLITS
        elif self.split == "train":
            self.allowed_splits = _TRAIN_SPLITS
        else:
            self.allowed_splits = {self.split}
        self.video_size = tuple(video_size)
        self.num_video_frames = int(num_video_frames)
        self.frames_clip_num = self.num_video_frames + 1
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.default_action_dim = int(action_dim)
        self.default_state_dim = int(state_dim)
        self.default_source_dataset = None if source_dataset is None else str(source_dataset)
        self.default_robot_name = None if robot_name is None else str(robot_name)
        self.instruction_lang = str(instruction_lang or "default").lower()
        if self.instruction_lang not in {"default", "zh", "en", "both"}:
            raise ValueError(
                f"instruction_lang must be one of default/zh/en/both, got {instruction_lang!r}"
            )
        self.max_samples = None if max_samples is None else int(max_samples)
        self.max_decode_retries = max(1, int(max_decode_retries))
        self.action_magnitude_guard_threshold = action_magnitude_guard_threshold
        self.action_magnitude_guard_sources = (
            None if action_magnitude_guard_sources is None else set(action_magnitude_guard_sources)
        )
        self.action_chunk_size = self.frames_clip_num * self.video_action_freq_ratio - 1
        self._bad_video_uris = set()

        self._offsets: List[int] = []
        self._fp = None
        self.stats = Counter(prebuilt_stats or {})
        self.drop_reasons = Counter(prebuilt_drop_reasons or {})
        if prebuilt_offsets is None:
            self._build_offsets()
        else:
            self._offsets = prebuilt_offsets

        if len(self._offsets) == 0:
            raise ValueError(
                f"No Motus manifest samples left after filtering manifest_path={self.manifest_path}, "
                f"allowed_splits={sorted(self.allowed_splits)}"
            )

        logger.info(
            "MotusManifestDataset initialized: manifest=%s split=%s samples=%d "
            "video_frames=%d frame_size=%s action_chunk=%d",
            self.manifest_path,
            self.split,
            len(self._offsets),
            self.num_video_frames,
            self.video_size,
            self.action_chunk_size,
        )
        if prebuilt_offsets is not None:
            logger.info("Motus manifest offsets loaded from prebuilt cache")
        logger.info("Motus manifest kept counts: %s", dict(self.stats))
        logger.info("Motus manifest dropped reason counts: %s", dict(self.drop_reasons))

    def __len__(self):
        return len(self._offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        state["_bad_video_uris"] = set()
        return state

    def close(self) -> None:
        fp = self._fp
        self._fp = None
        if fp is not None:
            fp.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _file(self):
        if self._fp is None:
            self._fp = self.manifest_path.open("rb")
        return self._fp

    def _read_record_at(self, offset: int) -> Dict[str, Any]:
        fp = self._file()
        fp.seek(offset)
        return json.loads(fp.readline())

    def _build_offsets(self):
        with self.manifest_path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.drop_reasons["bad_json"] += 1
                    continue

                keep, reason = self._keep_record(record)
                if not keep:
                    self.drop_reasons[reason] += 1
                    continue

                self._offsets.append(offset)
                self.stats[f"source:{self._source_dataset(record)}"] += 1
                self.stats[f"split:{record.get('split')}"] += 1
                self.stats[f"task:{record.get('task')}"] += 1

                if self.max_samples is not None and len(self._offsets) >= self.max_samples:
                    break

    def _keep_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        if record.get("schema_version") != "motus_manifest_v1":
            return False, "schema_version"
        if record.get("split") not in self.allowed_splits:
            return False, "split"
        video = record.get("video") or {}
        if not self._video_uri(video):
            return False, "missing_video_uri"
        has_frame_indices = len(self._video_frame_indices(video)) > 0
        start_sec, end_sec = self._video_time_window(video)
        if not has_frame_indices and (start_sec is None or end_sec is None or end_sec <= start_sec):
            return False, "bad_time"
        text = record.get("text") or {}
        if not self._clean_text(text.get("t5")) and not self._clean_text(text.get("vlm")):
            return False, "empty_text"
        return True, ""

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _source_dataset(self, record: Dict[str, Any]) -> str:
        return str(record.get("source_dataset") or self.default_source_dataset or "unknown")

    def _robot_name(self, record: Dict[str, Any]) -> str:
        robot = record.get("robot") or {}
        return str(robot.get("name") or self.default_robot_name or "default")

    @staticmethod
    def _video_uri(video: Dict[str, Any]) -> Optional[str]:
        return video.get("uri") or video.get("path")

    @staticmethod
    def _storage_backend(item: Dict[str, Any], default: str = "none") -> str:
        return str(item.get("storage_backend") or item.get("storage") or default).lower()

    @staticmethod
    def _is_tar_backend(backend: str) -> bool:
        return backend in {"tar", "tar_file", "file_tar"}

    @staticmethod
    def _is_robotmind_hdf5_jpeg_backend(backend: str) -> bool:
        return backend in {"robotmind_hdf5_jpeg", "robotmind_tianyi_hdf5_jpeg", "hdf5_jpeg"}

    @staticmethod
    def _video_time_window(video: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        try:
            return float(video.get("start_sec", 0.0)), float(video.get("end_sec"))
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _video_frame_indices(video: Dict[str, Any]) -> List[int]:
        indices = video.get("frame_indices") or video.get("indices") or []
        if isinstance(indices, str):
            return []
        try:
            return [int(index) for index in indices]
        except (TypeError, ValueError):
            return []

    def _format_text(self, record: Dict[str, Any]) -> Tuple[str, str]:
        text = record.get("text") or {}
        override = self._instruction_override_text(text)
        if override:
            return self._make_instruction_text(override)
        t5 = self._clean_text(text.get("t5"))
        vlm = self._clean_text(text.get("vlm"))
        target = self._clean_text(text.get("target"))
        fallback = target or self._clean_text(record.get("task")) or ""
        return t5 or fallback, vlm or t5 or fallback

    def _instruction_override_text(self, text: Dict[str, Any]) -> str:
        if self.instruction_lang == "default":
            return ""
        zh = self._clean_text(text.get("zh"))
        en = self._clean_text(text.get("en"))
        if self.instruction_lang == "zh":
            return zh or en
        if self.instruction_lang == "en":
            return en or zh
        if zh and en:
            return f"{zh} / {en}"
        return zh or en

    @staticmethod
    def _make_instruction_text(instruction: str) -> Tuple[str, str]:
        target = str(instruction).strip()
        t5 = f"Task: {target}.\n"
        vlm = "Based on the following situation, please predict what the robot will do: \n" + target
        return t5, vlm

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        image_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def _sample_frame_indices(self, vr: Any, start_sec: float, end_sec: float) -> List[int]:
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError("empty video")

        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0

        start_idx = max(0, min(total_frames - 1, int(round(start_sec * fps))))
        end_idx = max(start_idx, min(total_frames - 1, int(round(end_sec * fps))))
        if end_idx == start_idx:
            return [start_idx] * self.frames_clip_num

        indices = np.linspace(start_idx, end_idx, num=self.frames_clip_num)
        indices = np.rint(indices).astype(np.int64)
        indices = np.clip(indices, 0, total_frames - 1)
        return indices.tolist()

    def _load_video_clip(
        self,
        video_uri: str,
        start_sec: float,
        end_sec: float,
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if decord is None:
            return self._load_video_clip_cv2(video_uri, start_sec, end_sec, timing=timing)

        open_start = now()
        vr = decord.VideoReader(video_uri, num_threads=1)
        if timing is not None:
            timing["video_open_seconds"] = elapsed(open_start)
            timing["video_num_frames"] = len(vr)
            try:
                timing["video_fps"] = float(vr.get_avg_fps())
            except Exception:
                pass

        sample_start = now()
        frame_indices = self._sample_frame_indices(vr, start_sec, end_sec)
        if timing is not None:
            timing["sample_indices_seconds"] = elapsed(sample_start)
            timing["frame_index_first"] = int(frame_indices[0]) if frame_indices else None
            timing["frame_index_last"] = int(frame_indices[-1]) if frame_indices else None

        decode_start = now()
        frames = self._decord_batch_to_tensor(vr.get_batch(frame_indices))
        if timing is not None:
            timing["video_get_batch_seconds"] = elapsed(decode_start)

        resize_start = now()
        frames = F.interpolate(frames, size=self.video_size, mode="bilinear", align_corners=False)
        if timing is not None:
            timing["video_resize_seconds"] = elapsed(resize_start)
        return frames

    @staticmethod
    def _resize_frames_with_padding(frames: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if cv2 is None:
            raise ImportError("resize_mode=pad requires cv2")
        target_h, target_w = size
        frames_np = (frames.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        resized = []
        for frame in frames_np:
            h, w = frame.shape[:2]
            scale = min(target_h / h, target_w / w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            resized_frame = cv2.resize(frame, (new_w, new_h))
            padded = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_frame
            resized.append(padded)
        return torch.from_numpy(np.stack(resized, axis=0)).permute(0, 3, 1, 2).float() / 255.0

    def _resize_frames(
        self,
        frames: torch.Tensor,
        resize_mode: str = "interpolate",
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        resize_start = now()
        if resize_mode == "pad":
            frames = self._resize_frames_with_padding(frames, self.video_size)
        elif resize_mode in {"interpolate", "bilinear", ""}:
            frames = F.interpolate(frames, size=self.video_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unsupported video.resize_mode: {resize_mode}")
        if timing is not None:
            timing["video_resize_seconds"] = elapsed(resize_start)
            timing["video_resize_mode"] = resize_mode or "interpolate"
        return frames

    def _load_video_frames(
        self,
        video_uri: str,
        frame_indices: List[int],
        resize_mode: str = "interpolate",
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if not frame_indices:
            raise ValueError(f"Empty frame_indices for video: {video_uri}")
        if decord is None:
            return self._load_video_frames_cv2(video_uri, frame_indices, resize_mode=resize_mode, timing=timing)

        open_start = now()
        vr = decord.VideoReader(video_uri, num_threads=1)
        if timing is not None:
            timing["video_open_seconds"] = elapsed(open_start)
            timing["video_num_frames"] = len(vr)
            try:
                timing["video_fps"] = float(vr.get_avg_fps())
            except Exception:
                pass

        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Empty video: {video_uri}")
        safe_indices = [max(0, min(total_frames - 1, int(index))) for index in frame_indices]
        if len(safe_indices) != self.frames_clip_num:
            raise ValueError(
                f"Manifest frame_indices length must equal {self.frames_clip_num}, "
                f"got {len(safe_indices)} for video={video_uri}"
            )
        if timing is not None:
            timing["frame_index_first"] = int(safe_indices[0])
            timing["frame_index_last"] = int(safe_indices[-1])

        decode_start = now()
        frames = self._decord_batch_to_tensor(vr.get_batch(safe_indices))
        if timing is not None:
            timing["video_get_batch_seconds"] = elapsed(decode_start)

        return self._resize_frames(frames, resize_mode=resize_mode, timing=timing)

    def _load_video_frames_from_robotmind_hdf5(
        self,
        video: Dict[str, Any],
        frame_indices: List[int],
        resize_mode: str = "interpolate",
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if h5py is None:
            raise ImportError("RobotMind HDF5 JPEG manifest records require h5py")
        if not frame_indices:
            raise ValueError(f"Empty frame_indices for RobotMind HDF5 video: {self._video_uri(video)}")
        if len(frame_indices) != self.frames_clip_num:
            raise ValueError(
                f"Manifest frame_indices length must equal {self.frames_clip_num}, "
                f"got {len(frame_indices)} for video={self._video_uri(video)}"
            )

        video_uri = self._video_uri(video)
        if not video_uri:
            raise ValueError("RobotMind HDF5 video record missing uri/path")
        key = str(video.get("key") or "camera_observations/color_images/camera_top")

        open_start = now()
        frames = []
        with h5py.File(video_uri, "r") as f:
            if key not in f:
                raise KeyError(f"RobotMind HDF5 video key not found: {key} in {video_uri}")
            dataset = f[key]
            total_frames = int(dataset.shape[0])
            if total_frames <= 0:
                raise ValueError(f"RobotMind HDF5 video has no frames: {video_uri}")
            safe_indices = [max(0, min(total_frames - 1, int(index))) for index in frame_indices]
            if timing is not None:
                timing["video_open_seconds"] = elapsed(open_start)
                timing["video_num_frames"] = total_frames
                timing["frame_index_first"] = int(safe_indices[0])
                timing["frame_index_last"] = int(safe_indices[-1])
                timing["robotmind_hdf5_video_key"] = key

            decode_start = now()
            for index in safe_indices:
                raw = bytes(dataset[index])
                with Image.open(io.BytesIO(raw)) as image:
                    frames.append(np.array(image.convert("RGB")))

        if timing is not None:
            timing["video_get_batch_seconds"] = elapsed(decode_start)
            timing["video_decode_backend"] = "robotmind_hdf5_jpeg"

        tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        return self._resize_frames(tensor, resize_mode=resize_mode, timing=timing)

    @staticmethod
    def _decord_batch_to_tensor(batch) -> torch.Tensor:
        """Convert a decord batch in NHWC uint8 layout to NCHW float tensor.

        ``decord.bridge.set_bridge("torch")`` is process-local and can be
        inconsistent across DataLoader workers. Accept both torch Tensor and
        decord NDArray to keep manifest loading stable.
        """
        if isinstance(batch, torch.Tensor):
            tensor = batch
        elif hasattr(batch, "asnumpy"):
            tensor = torch.from_numpy(batch.asnumpy())
        else:
            tensor = torch.as_tensor(batch)
        return tensor.permute(0, 3, 1, 2).float() / 255.0

    def _load_video_frames_cv2(
        self,
        video_uri: str,
        frame_indices: List[int],
        resize_mode: str = "interpolate",
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if cv2 is None:
            raise ImportError("MotusManifestDataset requires decord or cv2 to load videos")

        open_start = now()
        cap = cv2.VideoCapture(video_uri)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_uri}")
        if timing is not None:
            timing["video_open_seconds"] = elapsed(open_start)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if timing is not None:
            timing["video_num_frames"] = total_frames
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Empty video: {video_uri}")

        safe_indices = [max(0, min(total_frames - 1, int(index))) for index in frame_indices]
        if len(safe_indices) != self.frames_clip_num:
            cap.release()
            raise ValueError(
                f"Manifest frame_indices length must equal {self.frames_clip_num}, "
                f"got {len(safe_indices)} for video={video_uri}"
            )
        if timing is not None:
            timing["frame_index_first"] = int(safe_indices[0])
            timing["frame_index_last"] = int(safe_indices[-1])

        decode_start = now()
        try:
            frames = self._read_cv2_frames_by_seek(cap, safe_indices, video_uri)
        except ValueError:
            cap.release()
            fallback_start = now()
            cap = cv2.VideoCapture(video_uri)
            if not cap.isOpened():
                raise ValueError(f"Failed to reopen video for sequential decode: {video_uri}")
            frames = self._read_cv2_frames_sequential(cap, safe_indices, video_uri)
            if timing is not None:
                timing["video_cv2_seek_failed"] = 1
                timing["video_cv2_sequential_seconds"] = elapsed(fallback_start)
        finally:
            cap.release()
        if timing is not None:
            timing["video_get_batch_seconds"] = elapsed(decode_start)

        tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        return self._resize_frames(tensor, resize_mode=resize_mode, timing=timing)

    def _read_cv2_frames_by_seek(
        self,
        cap: Any,
        safe_indices: List[int],
        video_uri: str,
    ) -> List[np.ndarray]:
        frames = []
        for idx in safe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise ValueError(f"Failed to read frame {idx} from video: {video_uri}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames

    def _read_cv2_frames_sequential(
        self,
        cap: Any,
        safe_indices: List[int],
        video_uri: str,
    ) -> List[np.ndarray]:
        targets = [(int(index), pos) for pos, index in enumerate(safe_indices)]
        targets.sort(key=lambda item: item[0])
        frames_by_pos: Dict[int, np.ndarray] = {}
        target_cursor = 0
        current_index = 0
        last_frame = None

        while target_cursor < len(targets):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = frame
            while target_cursor < len(targets) and targets[target_cursor][0] <= current_index:
                _, original_pos = targets[target_cursor]
                frames_by_pos[original_pos] = frame.copy()
                target_cursor += 1
            current_index += 1

        if target_cursor < len(targets):
            if last_frame is None:
                raise ValueError(f"Failed to read frame {targets[target_cursor][0]} from video: {video_uri}")
            while target_cursor < len(targets):
                _, original_pos = targets[target_cursor]
                frames_by_pos[original_pos] = last_frame.copy()
                target_cursor += 1

        return [frames_by_pos[pos] for pos in range(len(safe_indices))]

    def _read_tar_member_bytes(
        self,
        item: Dict[str, Any],
        timing: Optional[Dict[str, float]] = None,
        timing_prefix: str = "tar",
    ) -> bytes:
        tar_uri = item.get("uri") or item.get("path")
        member = item.get("member") or item.get("tar_member")
        if not tar_uri or not member:
            raise ValueError(f"Tar manifest item requires uri and member: uri={tar_uri} member={member}")

        read_start = now()
        offset = item.get("tar_offset_data", item.get("offset_data"))
        size = item.get("tar_size", item.get("size"))
        if offset is not None and size is not None:
            with open(tar_uri, "rb") as f:
                f.seek(int(offset))
                data = f.read(int(size))
            if len(data) != int(size):
                raise ValueError(
                    f"Short tar member read: uri={tar_uri} member={member} "
                    f"got={len(data)} expected={int(size)}"
                )
        else:
            with tarfile.open(tar_uri, "r") as tf:
                extracted = tf.extractfile(str(member))
                if extracted is None:
                    raise FileNotFoundError(f"Tar member not found: {tar_uri}!{member}")
                data = extracted.read()

        if timing is not None:
            timing[f"{timing_prefix}_read_seconds"] = elapsed(read_start)
            timing[f"{timing_prefix}_member_size"] = int(len(data))
            timing[f"{timing_prefix}_member"] = member
        return data

    def _load_video_frames_from_tar(
        self,
        video: Dict[str, Any],
        frame_indices: List[int],
        resize_mode: str = "interpolate",
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        data = self._read_tar_member_bytes(video, timing=timing, timing_prefix="video_tar")
        suffix = Path(str(video.get("member") or "video.mp4")).suffix or ".mp4"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="motus_tar_video_", suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            if timing is not None:
                timing["video_tar_temp_path"] = tmp_path
            decode_backend = str(
                video.get("decode_backend") or os.environ.get("MOTUS_TAR_VIDEO_BACKEND") or ""
            ).lower()
            if decode_backend == "ffmpeg":
                if timing is not None:
                    timing["video_decode_backend"] = "ffmpeg"
                return self._load_video_frames_ffmpeg(
                    tmp_path,
                    frame_indices,
                    resize_mode=resize_mode,
                    timing=timing,
                )
            if cv2 is None:
                raise ImportError("Tar video backend requires cv2 unless decode_backend=ffmpeg")
            try:
                if timing is not None:
                    timing["video_decode_backend"] = "cv2"
                return self._load_video_frames_cv2(tmp_path, frame_indices, resize_mode=resize_mode, timing=timing)
            except Exception as cv2_error:
                if timing is not None:
                    timing["video_cv2_failed"] = 1
                    timing["video_cv2_error"] = str(cv2_error)[:300]
                    timing["video_decode_backend"] = "ffmpeg_after_cv2"
                return self._load_video_frames_ffmpeg(
                    tmp_path,
                    frame_indices,
                    resize_mode=resize_mode,
                    timing=timing,
                )
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _ffmpeg_bin(self) -> str:
        configured = os.environ.get("MOTUS_FFMPEG_BIN")
        if configured:
            return configured
        if os.path.exists(_DEFAULT_FFMPEG_BIN):
            return _DEFAULT_FFMPEG_BIN
        return "ffmpeg"

    @staticmethod
    def _optional_positive_env_int(name: str) -> Optional[int]:
        raw_value = os.environ.get(name)
        if raw_value is None or str(raw_value).strip() == "":
            return None
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return None
        return max(1, value)

    def _load_video_frames_ffmpeg(
        self,
        video_uri: str,
        frame_indices: List[int],
        resize_mode: str = "interpolate",
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if not frame_indices:
            raise ValueError(f"Empty frame_indices for video: {video_uri}")
        safe_indices = [max(0, int(index)) for index in frame_indices]
        if len(safe_indices) != self.frames_clip_num:
            raise ValueError(
                f"Manifest frame_indices length must equal {self.frames_clip_num}, "
                f"got {len(safe_indices)} for video={video_uri}"
            )

        unique_indices = sorted(set(safe_indices))
        frames_by_index = self._load_video_frame_map_ffmpeg(
            video_uri,
            unique_indices,
            timing=timing,
        )
        frames = [frames_by_index[index].copy() for index in safe_indices]

        tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        return self._resize_frames(tensor, resize_mode=resize_mode, timing=timing)

    def _load_video_frame_map_ffmpeg(
        self,
        video_uri: str,
        unique_indices: List[int],
        timing: Optional[Dict[str, float]] = None,
    ) -> Dict[int, np.ndarray]:
        if not unique_indices:
            raise ValueError(f"Empty unique frame indices for video: {video_uri}")
        unique_indices = sorted({max(0, int(index)) for index in unique_indices})

        ranges = []
        range_start = unique_indices[0]
        previous = unique_indices[0]
        for index in unique_indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            ranges.append((range_start, previous))
            range_start = previous = index
        ranges.append((range_start, previous))

        select_parts = []
        for start, end in ranges:
            if start == end:
                select_parts.append(f"eq(n\\,{start})")
            else:
                select_parts.append(f"between(n\\,{start}\\,{end})")
        select_filter = "select=" + "+".join(select_parts)

        decode_start = now()
        with tempfile.TemporaryDirectory(prefix="motus_ffmpeg_frames_") as frame_dir:
            frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
            ffmpeg_threads = self._optional_positive_env_int("MOTUS_FFMPEG_THREADS")
            ffmpeg_filter_threads = self._optional_positive_env_int("MOTUS_FFMPEG_FILTER_THREADS")
            cmd = [
                self._ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
            ]
            if ffmpeg_threads is not None:
                cmd.extend(["-threads", str(ffmpeg_threads)])
            if ffmpeg_filter_threads is not None:
                cmd.extend([
                    "-filter_threads",
                    str(ffmpeg_filter_threads),
                    "-filter_complex_threads",
                    str(ffmpeg_filter_threads),
                ])
            cmd.extend([
                "-i",
                video_uri,
                "-vf",
                select_filter,
                "-vsync",
                "0",
            ])
            if ffmpeg_threads is not None:
                cmd.extend(["-threads", str(ffmpeg_threads)])
            cmd.append(frame_pattern)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise ValueError(
                    f"ffmpeg failed to decode frames from video={video_uri}: "
                    f"{result.stderr.strip()[:1000]}"
                )

            frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
            if len(frame_files) != len(unique_indices):
                raise ValueError(
                    f"ffmpeg decoded unexpected frame count for video={video_uri}: "
                    f"got={len(frame_files)} expected={len(unique_indices)}"
                )

            frames_by_index = {}
            for index, frame_path in zip(unique_indices, frame_files):
                with Image.open(frame_path) as image:
                    frames_by_index[index] = np.array(image.convert("RGB"))

        if timing is not None:
            timing["video_ffmpeg_decode_seconds"] = elapsed(decode_start)
            timing["video_ffmpeg_frame_count"] = len(unique_indices)
            timing["video_ffmpeg_select_ranges"] = len(ranges)
            timing["video_ffmpeg_bin"] = self._ffmpeg_bin()
            timing["video_ffmpeg_threads"] = ffmpeg_threads if ffmpeg_threads is not None else "auto"
            timing["video_ffmpeg_filter_threads"] = (
                ffmpeg_filter_threads if ffmpeg_filter_threads is not None else "auto"
            )
        return frames_by_index

    def _load_video_clip_cv2(
        self,
        video_uri: str,
        start_sec: float,
        end_sec: float,
        timing: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if cv2 is None:
            raise ImportError("MotusManifestDataset requires decord or cv2 to load videos")

        open_start = now()
        cap = cv2.VideoCapture(video_uri)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_uri}")
        if timing is not None:
            timing["video_open_seconds"] = elapsed(open_start)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if timing is not None:
            timing["video_num_frames"] = total_frames
            timing["video_fps"] = fps
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Empty video: {video_uri}")

        start_idx = max(0, min(total_frames - 1, int(round(start_sec * fps))))
        end_idx = max(start_idx, min(total_frames - 1, int(round(end_sec * fps))))
        if end_idx == start_idx:
            frame_indices = [start_idx] * self.frames_clip_num
        else:
            frame_indices = np.rint(np.linspace(start_idx, end_idx, num=self.frames_clip_num)).astype(np.int64)
            frame_indices = np.clip(frame_indices, 0, total_frames - 1).tolist()
        if timing is not None:
            timing["frame_index_first"] = int(frame_indices[0]) if frame_indices else None
            timing["frame_index_last"] = int(frame_indices[-1]) if frame_indices else None

        frames = []
        decode_start = now()
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                cap.release()
                raise ValueError(f"Failed to read frame {idx} from video: {video_uri}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if timing is not None:
            timing["video_get_batch_seconds"] = elapsed(decode_start)

        resize_start = now()
        frames_np = np.stack(frames, axis=0)
        tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        tensor = F.interpolate(tensor, size=self.video_size, mode="bilinear", align_corners=False)
        if timing is not None:
            timing["video_resize_seconds"] = elapsed(resize_start)
        return tensor

    def _load_agibot_state_action(
        self,
        record: Dict[str, Any],
        timing: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h5py is None or R is None:
            raise ImportError("Agibot manifest records require h5py and scipy")

        robot = record.get("robot") or {}
        state_dim = int(robot.get("state_dim") or self.default_state_dim)
        action_dim = int(robot.get("action_dim") or self.default_action_dim)
        action = record.get("action") or {}
        video = record.get("video") or {}
        state = record.get("state") or {}
        action_storage = self._storage_backend(action)
        state_storage = self._storage_backend(state)
        proprio_uri = action.get("uri") or state.get("uri")
        if not proprio_uri:
            raise ValueError(f"Agibot manifest record missing action/state uri: {record.get('id')}")

        frame_indices = action.get("frame_indices") or video.get("frame_indices") or []
        frame_mask = action.get("frame_mask")
        if not frame_indices:
            raise ValueError(f"Agibot manifest record missing frame_indices: {record.get('id')}")
        frame_indices = [int(index) for index in frame_indices]
        if len(frame_indices) != self.frames_clip_num:
            raise ValueError(
                f"Agibot frame_indices length must equal {self.frames_clip_num}, "
                f"got {len(frame_indices)} for id={record.get('id')}"
            )
        if frame_mask is None:
            frame_mask_tensor = torch.ones(len(frame_indices), dtype=torch.bool)
        else:
            frame_mask_tensor = torch.tensor([bool(item) for item in frame_mask], dtype=torch.bool)
        if frame_mask_tensor.numel() != len(frame_indices):
            raise ValueError(f"Agibot frame_mask length mismatch for id={record.get('id')}")

        ratio = int(action.get("action_video_fps_ratio") or self.video_action_freq_ratio)
        if ratio <= 0:
            raise ValueError(f"Agibot action_video_fps_ratio must be positive for id={record.get('id')}")

        h5_start = now()
        if action_storage == "agibot_proprio_hdf5_tar" or state_storage == "agibot_proprio_hdf5_tar":
            h5_data = self._read_tar_member_bytes(action or state, timing=timing, timing_prefix="h5_tar")
            h5_source = io.BytesIO(h5_data)
        else:
            h5_source = proprio_uri

        with h5py.File(h5_source, "r") as f:
            timesteps = np.array(f["state/end/position"]).shape[0]
            eef_actions = np.concatenate(
                [
                    np.array(f["state/end/position"]),
                    R.from_quat(np.array(f["state/end/orientation"]).reshape(-1, 4))
                    .as_rotvec()
                    .reshape(-1, 2, 3),
                    np.array(f["state/effector/position"])[:, :, None].reshape(timesteps, 2, -1),
                ],
                axis=-1,
            )
        if timing is not None:
            timing["h5_proprio_seconds"] = elapsed(h5_start)
            timing["timesteps"] = int(timesteps)

        action_start = now()
        start = int(frame_indices[0] * ratio)
        end = int((frame_indices[-1] + 1) * ratio)
        action_chunk = torch.tensor(eef_actions[start:end], dtype=torch.float32)
        expected_real_actions = int(frame_mask_tensor.sum().item() * ratio)
        if action_chunk.shape[0] < expected_real_actions:
            raise ValueError(
                f"Agibot action chunk too short for id={record.get('id')}: "
                f"got={action_chunk.shape[0]} expected_real={expected_real_actions}"
            )
        pad_num = int((~frame_mask_tensor).sum().item() * ratio)
        if pad_num > 0:
            action_chunk = torch.cat([action_chunk, action_chunk[-1:, :, :].expand(pad_num, -1, -1)], dim=0)

        is_dexterous_hand = action_chunk.shape[-1] > 7
        if is_dexterous_hand:
            action_chunk = torch.cat(
                [
                    action_chunk[:, :, :6],
                    torch.zeros(action_chunk.shape[0], action_chunk.shape[1], 1),
                    action_chunk[:, :, 6:],
                ],
                dim=-1,
            )
            action_chunk[:, :, 6] = _AGIBOT_ACTION_MEAN[6]
        else:
            action_chunk = torch.cat(
                [action_chunk, torch.zeros(action_chunk.shape[0], action_chunk.shape[1], 6)],
                dim=-1,
            )
            action_chunk[:, :, 7:] = _AGIBOT_ACTION_MEAN[7:]

        action_chunk = (action_chunk - _AGIBOT_ACTION_MEAN[None, None, :]) / _AGIBOT_ACTION_STD[None, None, :]
        video_frame_nums = len(frame_indices)
        last_frame_action_num = action_chunk.size(0) // video_frame_nums
        initial_state = action_chunk[0].reshape(-1)
        action_end = -last_frame_action_num + 1
        action_sequence = action_chunk[1 : (None if action_end == 0 else action_end)].reshape(-1, action_dim)
        if initial_state.numel() != state_dim:
            raise ValueError(
                f"Agibot initial_state dim mismatch for id={record.get('id')}: "
                f"got={initial_state.numel()} expected={state_dim}"
            )
        if action_sequence.size(-1) != action_dim:
            raise ValueError(
                f"Agibot action_sequence dim mismatch for id={record.get('id')}: "
                f"got={tuple(action_sequence.shape)} expected last dim={action_dim}"
            )
        if timing is not None:
            timing["action_build_seconds"] = elapsed(action_start)
            timing["action_sequence_length"] = int(action_sequence.shape[0])
        return initial_state.float(), action_sequence.float()

    def _load_robotwin_state_action(
        self,
        record: Dict[str, Any],
        timing: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        robot = record.get("robot") or {}
        state_dim = int(robot.get("state_dim") or self.default_state_dim)
        action_dim = int(robot.get("action_dim") or self.default_action_dim)
        action = record.get("action") or {}
        state = record.get("state") or {}
        qpos_uri = action.get("uri") or state.get("uri")
        if not qpos_uri:
            raise ValueError(f"Robotwin manifest record missing qpos uri: {record.get('id')}")

        frame_indices = action.get("frame_indices") or action.get("indices") or []
        if not frame_indices:
            raise ValueError(f"Robotwin manifest record missing action.frame_indices: {record.get('id')}")
        frame_indices = [int(index) for index in frame_indices]
        ratio = int(action.get("video_action_freq_ratio") or self.video_action_freq_ratio)
        if ratio <= 0:
            raise ValueError(f"Robotwin video_action_freq_ratio must be positive for id={record.get('id')}")
        expected_action_indices = self.frames_clip_num * ratio
        if len(frame_indices) != expected_action_indices:
            raise ValueError(
                f"Robotwin action frame_indices length must equal {expected_action_indices}, "
                f"got {len(frame_indices)} for id={record.get('id')}"
            )

        qpos_start = now()
        qpos_data = torch.load(qpos_uri, map_location="cpu")
        if timing is not None:
            timing["qpos_load_seconds"] = elapsed(qpos_start)
            timing["qpos_frames"] = int(qpos_data.shape[0]) if qpos_data.ndim >= 1 else None
        if qpos_data.ndim != 2:
            raise ValueError(f"Robotwin qpos tensor must be rank-2 for id={record.get('id')}")
        if qpos_data.shape[-1] != action_dim:
            raise ValueError(
                f"Robotwin action_dim mismatch for id={record.get('id')}: "
                f"qpos_dim={qpos_data.shape[-1]} expected={action_dim}"
            )
        if state_dim != action_dim:
            raise ValueError(
                f"Robotwin state_dim/action_dim mismatch for id={record.get('id')}: "
                f"state_dim={state_dim} action_dim={action_dim}"
            )
        max_index = int(qpos_data.shape[0]) - 1
        if max_index < 0:
            raise ValueError(f"Robotwin empty qpos tensor for id={record.get('id')}")
        safe_indices = [max(0, min(max_index, index)) for index in frame_indices]

        build_start = now()
        action_chunk = qpos_data[safe_indices].float()
        initial_state = action_chunk[0]
        action_sequence = action_chunk[1:]
        if initial_state.numel() != state_dim:
            raise ValueError(
                f"Robotwin initial_state dim mismatch for id={record.get('id')}: "
                f"got={initial_state.numel()} expected={state_dim}"
            )
        if action_sequence.size(-1) != action_dim:
            raise ValueError(
                f"Robotwin action_sequence dim mismatch for id={record.get('id')}: "
                f"got={tuple(action_sequence.shape)} expected last dim={action_dim}"
            )
        expected_action_length = expected_action_indices - 1
        if action_sequence.size(0) != expected_action_length:
            raise ValueError(
                f"Robotwin action_sequence length mismatch for id={record.get('id')}: "
                f"got={action_sequence.size(0)} expected={expected_action_length}"
            )
        if timing is not None:
            timing["action_build_seconds"] = elapsed(build_start)
            timing["action_sequence_length"] = int(action_sequence.shape[0])
        return initial_state.float(), action_sequence.float()

    def _robotmind_state_action_key_order(
        self,
        record: Dict[str, Any],
        state_keys: Dict[str, Any],
        action_keys: Dict[str, Any],
    ) -> List[str]:
        metadata = record.get("metadata") or {}
        order = metadata.get("state_action_keys_order")
        if isinstance(order, (list, tuple)) and order:
            return [str(key) for key in order]

        keys = action_keys or state_keys
        if isinstance(keys, dict) and keys:
            return [str(key) for key in keys.keys()]

        return list(_ROBOTMIND_DEFAULT_STATE_ACTION_ORDER)

    @staticmethod
    def _robotmind_resolve_hdf5_key(
        logical_key: str,
        keys: Dict[str, Any],
    ) -> str:
        if isinstance(keys, dict) and logical_key in keys:
            return str(keys[logical_key])
        if logical_key in _ROBOTMIND_DEFAULT_STATE_ACTION_KEYS:
            return _ROBOTMIND_DEFAULT_STATE_ACTION_KEYS[logical_key]
        raise KeyError(
            f"RobotMind state/action logical key {logical_key!r} is not present in manifest keys"
        )

    def _load_robotmind_hdf5_state_action(
        self,
        record: Dict[str, Any],
        timing: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h5py is None:
            raise ImportError("RobotMind manifest records require h5py")

        robot = record.get("robot") or {}
        robot_name = self._robot_name(record)
        state_dim = int(robot.get("state_dim") or self.default_state_dim)
        action_dim = int(robot.get("action_dim") or self.default_action_dim)
        if state_dim != action_dim:
            raise ValueError(
                f"RobotMind state_dim/action_dim mismatch for id={record.get('id')}: "
                f"state_dim={state_dim} action_dim={action_dim}"
            )

        action = record.get("action") or {}
        state = record.get("state") or {}
        hdf5_uri = action.get("uri") or state.get("uri") or self._video_uri(record.get("video") or {})
        if not hdf5_uri:
            raise ValueError(f"RobotMind record missing hdf5 uri: {record.get('id')}")

        frame_indices = action.get("frame_indices") or action.get("indices") or []
        if not frame_indices:
            raise ValueError(f"RobotMind record missing action.frame_indices: {record.get('id')}")
        frame_indices = [int(index) for index in frame_indices]
        expected_action_indices = self.frames_clip_num * int(action.get("video_action_freq_ratio") or self.video_action_freq_ratio)
        if len(frame_indices) != expected_action_indices:
            raise ValueError(
                f"RobotMind action frame_indices length must equal {expected_action_indices}, "
                f"got {len(frame_indices)} for id={record.get('id')}"
            )

        state_keys = state.get("keys") or {}
        action_keys = action.get("keys") or {}
        key_order = self._robotmind_state_action_key_order(record, state_keys, action_keys)
        state_index = int(state.get("index", frame_indices[0]))

        h5_start = now()
        with h5py.File(hdf5_uri, "r") as f:
            state_arrays = []
            action_arrays = []
            for logical_key in key_order:
                state_key = self._robotmind_resolve_hdf5_key(logical_key, state_keys or action_keys)
                action_key = self._robotmind_resolve_hdf5_key(logical_key, action_keys or state_keys)
                if state_key not in f:
                    raise KeyError(f"RobotMind state key not found: {state_key} in {hdf5_uri}")
                if action_key not in f:
                    raise KeyError(f"RobotMind action key not found: {action_key} in {hdf5_uri}")
                state_arr = np.asarray(f[state_key])
                action_arr = np.asarray(f[action_key])
                if state_arr.ndim == 1:
                    state_arr = state_arr[:, None]
                if action_arr.ndim == 1:
                    action_arr = action_arr[:, None]
                state_arrays.append(state_arr)
                action_arrays.append(action_arr)

            total_frames = min(
                int(arr.shape[0]) for arr in [*state_arrays, *action_arrays]
            )
            if total_frames <= 0:
                raise ValueError(f"RobotMind empty state/action arrays for id={record.get('id')}")
            safe_state_index = max(0, min(total_frames - 1, state_index))
            safe_indices = [max(0, min(total_frames - 1, index)) for index in frame_indices]
            initial_state_np = np.concatenate(
                [arr[safe_state_index] for arr in state_arrays],
                axis=-1,
            )
            action_chunk_np = np.concatenate(
                [arr[safe_indices] for arr in action_arrays],
                axis=-1,
            )
        if timing is not None:
            timing["h5_proprio_seconds"] = elapsed(h5_start)
            timing["timesteps"] = int(total_frames)
            timing["robotmind_robot_name"] = robot_name
            timing["robotmind_state_action_keys"] = ",".join(key_order)

        build_start = now()
        initial_state = torch.tensor(initial_state_np, dtype=torch.float32)
        action_chunk = torch.tensor(action_chunk_np, dtype=torch.float32)
        if initial_state.numel() != state_dim:
            raise ValueError(
                f"RobotMind state_dim mismatch for id={record.get('id')}: "
                f"got={initial_state.numel()} expected={state_dim} keys={key_order}"
            )
        if action_chunk.shape[-1] != action_dim:
            raise ValueError(
                f"RobotMind action_dim mismatch for id={record.get('id')}: "
                f"got={action_chunk.shape[-1]} expected={action_dim} keys={key_order}"
            )
        action_sequence = action_chunk[1:]
        if action_sequence.size(0) != expected_action_indices - 1:
            raise ValueError(
                f"RobotMind action_sequence length mismatch for id={record.get('id')}: "
                f"got={action_sequence.size(0)} expected={expected_action_indices - 1}"
            )
        if timing is not None:
            timing["action_build_seconds"] = elapsed(build_start)
            timing["action_sequence_length"] = int(action_sequence.shape[0])
        return initial_state.float(), action_sequence.float()

    def _load_robotmind_tianyi_state_action(
        self,
        record: Dict[str, Any],
        timing: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._load_robotmind_hdf5_state_action(record, timing=timing)

    def _load_state_action(
        self,
        record: Dict[str, Any],
        timing: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        robot = record.get("robot") or {}
        state_dim = int(robot.get("state_dim") or self.default_state_dim)
        action_dim = int(robot.get("action_dim") or self.default_action_dim)

        state = record.get("state") or {}
        action = record.get("action") or {}
        state_storage = self._storage_backend(state)
        action_storage = self._storage_backend(action)

        if state_storage in {"agibot_proprio_hdf5", "agibot_proprio_hdf5_tar"} or action_storage in {
            "agibot_proprio_hdf5",
            "agibot_proprio_hdf5_tar",
        }:
            if state_storage != action_storage:
                raise ValueError(
                    "Agibot manifest state/action storage backends must match"
                )
            return self._load_agibot_state_action(record, timing=timing)

        if state_storage == "robotwin_qpos_pt" or action_storage == "robotwin_qpos_pt":
            if state_storage != action_storage:
                raise ValueError(
                    "Robotwin manifest state/action storage backends must both be robotwin_qpos_pt"
                )
            return self._load_robotwin_state_action(record, timing=timing)

        if state_storage in _ROBOTMIND_STATE_ACTION_BACKENDS or action_storage in _ROBOTMIND_STATE_ACTION_BACKENDS:
            if state_storage != action_storage:
                raise ValueError(
                    "RobotMind manifest state/action storage backends must match"
                )
            return self._load_robotmind_hdf5_state_action(record, timing=timing)

        if state_storage not in {"none", "null"} or action_storage not in {"none", "null"}:
            raise NotImplementedError(
                f"Unsupported Motus manifest state/action storage_backend: "
                f"state={state_storage}, action={action_storage}"
            )

        initial_state = torch.zeros(state_dim, dtype=torch.float32)
        action_sequence = torch.zeros(self.action_chunk_size, action_dim, dtype=torch.float32)
        return initial_state, action_sequence

    def _make_sample(self, index: int, timing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        read_start = now()
        record = self._read_record_at(self._offsets[index])
        if timing is not None:
            timing["read_record_seconds"] = elapsed(read_start)

        video = record["video"]
        video_uri = self._video_uri(video)
        video_backend = self._storage_backend(video, default="file")
        video_key = self._video_key(video, video_uri=video_uri, video_backend=video_backend)
        if video_key in self._bad_video_uris:
            raise ValueError(f"Skipping known bad video uri: {video_key}")
        frame_indices = self._video_frame_indices(video)
        start_sec, end_sec = self._video_time_window(video)
        if not frame_indices and (start_sec is None or end_sec is None):
            raise ValueError(f"Invalid manifest video window for sample {record.get('id')}")
        if timing is not None:
            timing["start_sec"] = start_sec
            timing["end_sec"] = end_sec
            timing["manifest_id"] = record.get("id")
            timing["manifest_task"] = record.get("task")
            timing["manifest_subtask"] = record.get("subtask")
            timing["video_uri"] = video_uri
            timing["video_storage_backend"] = video.get("storage_backend")
            timing["video_resize_mode"] = video.get("resize_mode")
            timing["robot_name"] = self._robot_name(record)
            metadata = record.get("metadata") or {}
            window_metadata = metadata.get("window_metadata") or metadata.get("window_precut") or {}
            if isinstance(window_metadata, dict):
                timing["window_index"] = window_metadata.get("window_index")
                timing["window_count"] = window_metadata.get("window_count")
                timing["original_segment_start_sec"] = window_metadata.get("original_segment_start_sec")
                timing["original_segment_end_sec"] = window_metadata.get("original_segment_end_sec")
                timing["chunk_duration_sec"] = window_metadata.get("chunk_duration_sec")

        resize_mode = str(video.get("resize_mode") or "interpolate").lower()
        if frame_indices:
            if self._is_robotmind_hdf5_jpeg_backend(video_backend):
                frames = self._load_video_frames_from_robotmind_hdf5(
                    video,
                    frame_indices,
                    resize_mode=resize_mode,
                    timing=timing,
                )
            elif self._is_tar_backend(video_backend):
                frames = self._load_video_frames_from_tar(
                    video,
                    frame_indices,
                    resize_mode=resize_mode,
                    timing=timing,
                )
            else:
                frames = self._load_video_frames(video_uri, frame_indices, resize_mode=resize_mode, timing=timing)
        else:
            frames = self._load_video_clip(video_uri, start_sec, end_sec, timing=timing)
        first_frame = frames[0]
        video_frames = frames[1:]

        sample_build_start = now()
        initial_state, action_sequence = self._load_state_action(record, timing=timing)
        t5_text, vlm_text = self._format_text(record)
        robot = record.get("robot") or {}
        loss = record.get("loss") or {}
        action_policy = str(robot.get("action_policy") or "real")
        source_dataset = self._source_dataset(record)
        validate_action_state_magnitude(
            source_dataset=source_dataset,
            action_sequence=action_sequence,
            initial_state=initial_state,
            manifest_id=record.get("id"),
            manifest_task=record.get("task"),
            threshold=self.action_magnitude_guard_threshold,
            sources=self.action_magnitude_guard_sources,
        )
        if timing is not None:
            timing["sample_build_seconds"] = elapsed(sample_build_start)
            timing["source_dataset"] = source_dataset
            timing["manifest_id"] = record.get("id")
            timing["manifest_task"] = record.get("task")
            timing["robot_name"] = self._robot_name(record)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "first_frame_pil": self._tensor_to_pil(first_frame),
            "t5_input_message": t5_text,
            "vlm_input_message": vlm_text,
            "robot_name": self._robot_name(record),
            "source_dataset": source_dataset,
            "action_loss_weight": float(loss.get("action_weight", 1.0)),
            "video_loss_weight": float(loss.get("video_weight", 1.0)),
            "state_action_policy": action_policy,
            "manifest_id": record.get("id"),
            "manifest_task": record.get("task"),
        }

    def _video_key(
        self,
        video: Dict[str, Any],
        video_uri: Optional[str] = None,
        video_backend: Optional[str] = None,
    ) -> str:
        uri = self._video_uri(video) if video_uri is None else video_uri
        backend = self._storage_backend(video, default="file") if video_backend is None else video_backend
        if self._is_tar_backend(backend):
            return f"{uri}!{video.get('member')}"
        return str(uri)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        total_start = now()
        last_error = None
        decode_attempts = 0
        skipped_bad_videos = 0
        for scan_offset in range(len(self._offsets)):
            candidate_index = (index + scan_offset) % len(self._offsets)
            try:
                precheck_record = self._read_record_at(self._offsets[candidate_index])
                precheck_video = precheck_record.get("video") or {}
                precheck_key = self._video_key(precheck_video)
                if precheck_key in self._bad_video_uris:
                    skipped_bad_videos += 1
                    continue
            except Exception:
                precheck_key = None

            decode_attempts += 1
            if decode_attempts > self.max_decode_retries:
                break
            timing = {
                "requested_index": int(index),
                "candidate_index": int(candidate_index),
                "attempt": int(decode_attempts),
                "max_decode_retries": int(self.max_decode_retries),
                "skipped_bad_videos": int(skipped_bad_videos),
            }
            try:
                sample = self._make_sample(candidate_index, timing=timing)
                timing["total_seconds"] = elapsed(total_start)
                write_dataset_timing("motus_manifest", timing)
                return sample
            except Exception as exc:
                last_error = exc
                if isinstance(exc, ActionMagnitudeOutlier):
                    outlier_timing = {
                        **timing,
                        **getattr(exc, "details", {}),
                        "total_seconds": elapsed(total_start),
                        "outlier_error": str(exc),
                    }
                    write_dataset_timing("action_magnitude_outlier", outlier_timing, threshold_seconds=0.0)
                    logger.warning(
                        "Motus manifest action/state outlier skipped index=%s candidate=%s attempt=%s/%s: %s",
                        index,
                        candidate_index,
                        decode_attempts,
                        self.max_decode_retries,
                        exc,
                    )
                    continue
                try:
                    record = self._read_record_at(self._offsets[candidate_index])
                    video = record.get("video") or {}
                    video_uri = self._video_uri(video)
                    video_backend = self._storage_backend(video, default="file")
                    if video_uri:
                        video_key = self._video_key(
                            video,
                            video_uri=video_uri,
                            video_backend=video_backend,
                        )
                        self._bad_video_uris.add(video_key)
                except Exception:
                    pass
                logger.warning(
                    "Motus manifest sample failed index=%s candidate=%s attempt=%s/%s: %s",
                    index,
                    candidate_index,
                    decode_attempts,
                    self.max_decode_retries,
                    exc,
                )
        raise RuntimeError(
            f"Failed to decode Motus manifest sample after {self.max_decode_retries} attempts "
            f"from index={index} skipped_bad_videos={skipped_bad_videos}: {last_error}"
        )
