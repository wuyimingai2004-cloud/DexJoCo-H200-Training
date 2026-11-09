"""Shared Agibot decoding and alignment primitives."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import decord
except ImportError:
    decord = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


AGIBOT_ACTION_MEAN = torch.tensor(
    [
        0.6623259, -0.02415569, 0.65893349, -0.91274923, 0.93820975,
        -0.62816762, 64.11719045790569, -0.01470473, 0.27863706,
        0.4925647, 0.47402732, 0.51696799, -1.0560014,
    ],
    dtype=torch.float32,
)
AGIBOT_ACTION_STD = torch.tensor(
    [
        0.08038633, 0.3268345, 0.19179442, 1.23554225, 1.35022756,
        0.92738767, 37.00465005224069, 2.12378374e-01, 3.09977879e-01,
        4.75664546e-01, 4.69941935e-01, 4.90734943e-01, 1.17136790e-04,
    ],
    dtype=torch.float32,
)

TASK_PROMPT = "Scene Description: {scene}\nTask: {task}.\n"
VLM_PROMPT = "Based on the following situation, please predict what the robot will do: \n"
_DEFAULT_FFMPEG_BIN = "/code1/7TB/v2/bin/ffmpeg-static/bin/ffmpeg"

# 单次 ffmpeg select 表达式的最大帧数（分块解码粒度）：集群静态 ffmpeg
# 构建的表达式解析缓冲约 1KB（实测 ~100 个 eq/between 项是临界点），
# 64 帧/块对应表达式约 700 字符，留有充足余量。
_FFMPEG_SELECT_CHUNK_SIZE = 64


def padded_frame_indices(start: int, total_frames: int, count: int) -> Tuple[list[int], list[bool]]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    real_end = min(int(total_frames), int(start) + int(count))
    indices = list(range(int(start), real_end))
    mask = [True] * len(indices)
    if len(indices) < int(count):
        pad = int(count) - len(indices)
        indices.extend([int(total_frames) - 1] * pad)
        mask.extend([False] * pad)
    return indices, mask


def _as_frame_tensor(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        tensor = batch
    elif hasattr(batch, "asnumpy"):
        tensor = torch.from_numpy(batch.asnumpy())
    else:
        tensor = torch.as_tensor(batch)
    return tensor.permute(0, 3, 1, 2).float() / 255.0


def decode_video_frames(
    video_path: str,
    frame_indices: Iterable[int],
    video_size: Tuple[int, int],
    *,
    backend: str = "auto",
) -> torch.Tensor:
    indices = [int(index) for index in frame_indices]
    if not indices:
        raise ValueError("frame_indices cannot be empty")
    backend = str(backend).lower()
    if backend == "ffmpeg":
        frame_map = decode_video_frame_map_ffmpeg(video_path, indices)
        frames = _as_frame_tensor(
            np.stack([frame_map[index].copy() for index in indices], axis=0)
        )
    elif backend in {"auto", "decord"} and decord is not None:
        reader = decord.VideoReader(str(video_path), num_threads=1)
        if len(reader) <= 0:
            raise ValueError(f"empty video: {video_path}")
        safe = [max(0, min(len(reader) - 1, index)) for index in indices]
        frames = _as_frame_tensor(reader.get_batch(safe))
    elif backend in {"auto", "opencv", "cv2"} and cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"failed to open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = []
        try:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, index)))
                ok, frame = cap.read()
                if not ok:
                    if not decoded:
                        raise ValueError(f"failed to decode frame {index}: {video_path}")
                    frame = decoded[-1][:, :, ::-1].copy()
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        frames = _as_frame_tensor(np.stack(decoded, axis=0))
    else:
        raise ImportError(
            f"Agibot video decoding backend={backend!r} is unavailable; "
            "install decord/opencv-python or use ffmpeg"
        )
    return F.interpolate(frames, size=tuple(video_size), mode="bilinear", align_corners=False)


def _ffmpeg_bin() -> str:
    configured = os.environ.get("MOTUS_FFMPEG_BIN")
    if configured:
        return configured
    if os.path.exists(_DEFAULT_FFMPEG_BIN):
        return _DEFAULT_FFMPEG_BIN
    return "ffmpeg"


def _optional_positive_env_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or not str(raw_value).strip():
        return None
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return None


def decode_video_frame_map_ffmpeg(
    video_path: str, frame_indices: Iterable[int]
) -> Dict[int, np.ndarray]:
    """Decode unique source-frame indices exactly once with ffmpeg."""

    unique_indices = sorted({max(0, int(index)) for index in frame_indices})
    if not unique_indices:
        raise ValueError("frame_indices cannot be empty")

    result_frames: Dict[int, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="motus_agibot_frames_") as frame_dir:
        # 分块解码：集群静态 ffmpeg 构建的 select 表达式解析缓冲约 1KB
        # （实测 ~100 个 eq/between 项），超出即 "Error while parsing
        # expression"。连续布局下并集索引高度重叠、表达式很短；稀疏跨步
        # 采样（变 FPS 变倍率窗口）下并集几乎不重叠，表达式会膨胀到数百
        # 项，故按帧数分块、每块单独跑一遍 ffmpeg 再合并。
        for chunk_no, chunk_start in enumerate(
            range(0, len(unique_indices), _FFMPEG_SELECT_CHUNK_SIZE)
        ):
            chunk = unique_indices[chunk_start:chunk_start + _FFMPEG_SELECT_CHUNK_SIZE]
            ranges = []
            range_start = previous = chunk[0]
            for index in chunk[1:]:
                if index == previous + 1:
                    previous = index
                    continue
                ranges.append((range_start, previous))
                range_start = previous = index
            ranges.append((range_start, previous))
            select_parts = [
                f"eq(n\\,{start})" if start == end else f"between(n\\,{start}\\,{end})"
                for start, end in ranges
            ]

            chunk_dir = os.path.join(frame_dir, f"chunk_{chunk_no:04d}")
            os.makedirs(chunk_dir, exist_ok=True)
            pattern = os.path.join(chunk_dir, "frame_%06d.png")
            ffmpeg_threads = _optional_positive_env_int("MOTUS_FFMPEG_THREADS")
            filter_threads = _optional_positive_env_int("MOTUS_FFMPEG_FILTER_THREADS")
            command = [
                _ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
            ]
            if ffmpeg_threads is not None:
                command.extend(["-threads", str(ffmpeg_threads)])
            if filter_threads is not None:
                command.extend(
                    [
                        "-filter_threads",
                        str(filter_threads),
                        "-filter_complex_threads",
                        str(filter_threads),
                    ]
                )
            command.extend([
                "-i",
                str(video_path),
                "-vf",
                "select=" + "+".join(select_parts),
                "-vsync",
                "0",
            ])
            if ffmpeg_threads is not None:
                command.extend(["-threads", str(ffmpeg_threads)])
            command.append(pattern)
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode != 0:
                raise ValueError(
                    f"ffmpeg failed to decode {video_path}: {result.stderr.strip()[:1000]}"
                )
            frame_paths = sorted(Path(chunk_dir).glob("frame_*.png"))
            if len(frame_paths) != len(chunk):
                raise ValueError(
                    f"ffmpeg decoded {len(frame_paths)} frames, expected {len(chunk)} "
                    f"from {video_path}"
                )
            for index, frame_path in zip(chunk, frame_paths):
                with Image.open(frame_path) as image:
                    result_frames[index] = np.asarray(image.convert("RGB")).copy()
    return result_frames


def load_agibot_state_action(
    proprio_path: str,
    frame_indices: Iterable[int],
    frame_mask: Iterable[bool],
    *,
    action_video_fps_ratio: int,
    state_dim: int,
    action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if h5py is None or Rotation is None:
        raise ImportError("Agibot state/action decoding requires h5py and scipy")
    indices = [int(index) for index in frame_indices]
    mask = torch.tensor([bool(value) for value in frame_mask], dtype=torch.bool)
    ratio = int(action_video_fps_ratio)
    if not indices or len(indices) != mask.numel() or ratio <= 0:
        raise ValueError("invalid Agibot frame indices, mask, or action/video ratio")

    eef_actions = read_agibot_eef_actions(proprio_path)
    return build_agibot_state_action(
        eef_actions,
        frame_indices,
        frame_mask,
        action_video_fps_ratio=action_video_fps_ratio,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def read_agibot_eef_actions(proprio_source: str | Path | BinaryIO) -> np.ndarray:
    if h5py is None or Rotation is None:
        raise ImportError("Agibot state/action decoding requires h5py and scipy")
    h5_source = (
        str(proprio_source)
        if isinstance(proprio_source, (str, Path))
        else proprio_source
    )
    with h5py.File(h5_source, "r") as handle:
        end_position = np.asarray(handle["state/end/position"])
        timesteps = int(end_position.shape[0])
        end_rotation = Rotation.from_quat(
            np.asarray(handle["state/end/orientation"]).reshape(-1, 4)
        ).as_rotvec().reshape(-1, 2, 3)
        effector = np.asarray(handle["state/effector/position"])[:, :, None].reshape(
            timesteps, 2, -1
        )
        return np.concatenate([end_position, end_rotation, effector], axis=-1)


def build_agibot_state_action(
    eef_actions: np.ndarray,
    frame_indices: Iterable[int],
    frame_mask: Iterable[bool],
    *,
    action_video_fps_ratio: int,
    state_dim: int,
    action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = [int(index) for index in frame_indices]
    mask = torch.tensor([bool(value) for value in frame_mask], dtype=torch.bool)
    ratio = int(action_video_fps_ratio)
    if not indices or len(indices) != mask.numel() or ratio <= 0:
        raise ValueError("invalid Agibot frame indices, mask, or action/video ratio")

    start = int(indices[0] * ratio)
    end = int((indices[-1] + 1) * ratio)
    canonical_timeline = normalize_agibot_eef_actions(eef_actions)
    action_chunk = canonical_timeline[start:end]
    expected_real = int(mask.sum().item() * ratio)
    if action_chunk.shape[0] < expected_real or action_chunk.shape[0] == 0:
        raise ValueError(
            f"Agibot action chunk too short: got={action_chunk.shape[0]} expected={expected_real}"
        )
    pad_count = int((~mask).sum().item() * ratio)
    if pad_count:
        action_chunk = torch.cat(
            [action_chunk, action_chunk[-1:].expand(pad_count, -1, -1)], dim=0
        )

    actions_per_video_frame = action_chunk.shape[0] // len(indices)
    initial_state = action_chunk[0]
    action_end = -actions_per_video_frame + 1
    action_sequence = action_chunk[1 : (None if action_end == 0 else action_end)]
    if initial_state.numel() != int(state_dim):
        raise ValueError(
            f"Agibot state dim mismatch: got={initial_state.numel()} expected={state_dim}"
        )
    expected_actions = (len(indices) - 1) * ratio
    if tuple(action_sequence.shape) != (expected_actions, int(action_dim)):
        raise ValueError(
            f"Agibot action shape mismatch: got={tuple(action_sequence.shape)} "
            f"expected={(expected_actions, int(action_dim))}"
        )
    return initial_state.float(), action_sequence.float()


def normalize_agibot_eef_actions(eef_actions: np.ndarray) -> torch.Tensor:
    """Convert a complete Agibot proprio timeline to canonical 26-D rows."""

    action_timeline = torch.as_tensor(eef_actions, dtype=torch.float32)
    if action_timeline.ndim != 3 or action_timeline.shape[1] != 2:
        raise ValueError(
            "Agibot EEF timeline must have shape [T,2,D], got "
            f"{tuple(action_timeline.shape)}"
        )
    if action_timeline.shape[-1] > 7:
        action_timeline = torch.cat(
            [
                action_timeline[:, :, :6],
                torch.zeros(
                    action_timeline.shape[0],
                    action_timeline.shape[1],
                    1,
                    dtype=action_timeline.dtype,
                ),
                action_timeline[:, :, 6:],
            ],
            dim=-1,
        )
        action_timeline[:, :, 6] = AGIBOT_ACTION_MEAN[6]
    else:
        action_timeline = torch.cat(
            [
                action_timeline,
                torch.zeros(
                    action_timeline.shape[0],
                    action_timeline.shape[1],
                    6,
                    dtype=action_timeline.dtype,
                ),
            ],
            dim=-1,
        )
        action_timeline[:, :, 7:] = AGIBOT_ACTION_MEAN[7:]

    if action_timeline.shape[-1] != AGIBOT_ACTION_MEAN.numel():
        raise ValueError(
            "Agibot canonical arm width mismatch: got "
            f"{action_timeline.shape[-1]} expected {AGIBOT_ACTION_MEAN.numel()}"
        )
    normalized = (
        action_timeline - AGIBOT_ACTION_MEAN[None, None, :]
    ) / AGIBOT_ACTION_STD[None, None, :]
    return normalized.reshape(normalized.shape[0], -1).float()


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    array = (
        frame.detach().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array)


def action_text_for_start(
    labels: Dict[str, Any], start_frame: int, *, require_key: bool = False
) -> str:
    key = str(int(start_frame))
    if require_key and key not in labels:
        raise KeyError(f"missing Agibot action label for start_frame={start_frame}")
    action_info = labels.get(key)
    text = ""
    if isinstance(action_info, dict):
        if str(action_info.get("action_text") or "").strip():
            text = str(action_info["action_text"]).strip()
        elif str(action_info.get("skill") or "").strip():
            text = str(action_info["skill"]).strip() + "."
    return f"Action: {text}\n"


def action_text_for_span_start(
    action_config: Sequence[Dict[str, Any]], start_frame: int, *, require_match: bool = False
) -> str:
    """Resolve Beta's interval annotation at the same frame used by all modalities."""

    for action in action_config:
        try:
            start = int(action.get("start_frame"))
            end = int(action.get("end_frame"))
        except (TypeError, ValueError):
            continue
        if start <= int(start_frame) < end:
            text = str(action.get("action_text") or "").strip()
            if not text:
                text = str(action.get("skill") or "").strip()
            return f"Action: {text}\n"
    if require_match:
        raise KeyError(f"missing Agibot action span for start_frame={start_frame}")
    return ""
