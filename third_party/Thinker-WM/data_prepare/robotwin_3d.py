"""Flex-π robotwin_3d LeRobot loader: 3 RGB + 3 FFV1 depth + 14D qpos.

Three cameras are stitched into the Motus T canvas (360x320), then aspect-pad
to the ThinkerWAM 480x640 slot. Depth stays metric millimetres on the canvas;
``encode_depth_mm`` applies one shared log curve afterwards.

Do not min-max each view. Head ~0.7 m and a wrist at 50 mm are the same world
scale; per-view normalize would make 50 mm look like 700 mm after encode.
Wrist values below the global ``z_near`` clip. Default 0.01–2.0 m matches
Flex-π pointmap validity ``(z > 0.01) & (z < 2.0)``. Not the body-cam 0.15 m.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ._bootstrap import ensure_motus_data_importable

ensure_motus_data_importable()

from motus_data.catalog.sqlite_index import build_catalog_offset_index  # noqa: E402
from motus_data.datasets.agibot_common import (  # noqa: E402
    decode_video_frame_map_ffmpeg,
)
from motus_data.index_contracts import (  # noqa: E402
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata_v2,
    validate_episode_catalog_binding,
    validate_episode_record_v2,
)
from motus_data.reader_registry import (  # noqa: E402
    EPISODE_SOURCE_READERS,
    register_episode_source_reader,
)
from motus_data.repository import (  # noqa: E402
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


SOURCE_DATASET = "robotwin_3d"
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
NATIVE_HW = (240, 320)  # H, W per camera
COMPOSE_HW = (360, 320)  # Motus T: head 320x240 top, wrists 160x120
HEAD_HW = (240, 320)
WRIST_HW = (120, 160)
QPOS_DIM = 14
QPOS_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
# Flex-π pointmap validity. 0.15 m (v9 body-cam) clips ~30-100 mm wrist GT.
DEFAULT_Z_NEAR_M = 0.01
DEFAULT_Z_FAR_M = 2.0


def episode_chunk(episode_index: int) -> str:
    return f"chunk-{int(episode_index) // 1000:03d}"


def episode_stem(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}"


def parse_episode_index(episode_id: str) -> int:
    prefix = f"{SOURCE_DATASET}:"
    if not str(episode_id).startswith(prefix):
        raise ValueError(f"not a {SOURCE_DATASET} episode_id: {episode_id!r}")
    return int(str(episode_id)[len(prefix) :])


def robotwin_3d_paths(root: str | Path, episode_index: int) -> dict[str, Path]:
    root = Path(root)
    chunk = episode_chunk(episode_index)
    stem = episode_stem(episode_index)
    out: dict[str, Path] = {
        "parquet": root / "data" / chunk / f"{stem}.parquet",
    }
    for cam in CAMS:
        out[f"rgb_{cam}"] = (
            root / "videos" / chunk / f"observation.images.{cam}" / f"{stem}.mp4"
        )
        out[f"depth_{cam}"] = (
            root
            / "videos"
            / chunk
            / f"observation.depth_ffv1.{cam}"
            / f"{stem}.mkv"
        )
    return out


def compose_motus_t_rgb(
    head: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Stitch 320x240 views into Motus T 360x320. RGB bilinear into wrist slots."""
    canvas = np.zeros((COMPOSE_HW[0], COMPOSE_HW[1], 3), dtype=np.uint8)
    head_u8 = _as_rgb_hwc(head, HEAD_HW, nearest=False)
    canvas[0:240, 0:320] = head_u8
    canvas[240:360, 0:160] = _as_rgb_hwc(left, WRIST_HW, nearest=False)
    canvas[240:360, 160:320] = _as_rgb_hwc(right, WRIST_HW, nearest=False)
    return canvas


def compose_motus_t_depth_mm(
    head: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Stitch uint16-mm views. Wrist slots nearest; empty canvas is 0=invalid."""
    canvas = np.zeros(COMPOSE_HW, dtype=np.uint16)
    canvas[0:240, 0:320] = _as_depth_hw(head, HEAD_HW)
    canvas[240:360, 0:160] = _as_depth_hw(left, WRIST_HW)
    canvas[240:360, 160:320] = _as_depth_hw(right, WRIST_HW)
    return canvas


def pad_rgb_to_size(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Aspect-preserving bilinear resize + zero pad. ``size`` is (H, W)."""
    return _pad_stack(frames, size, nearest=False)


def pad_depth_mm_to_size(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Same geometry as RGB pad, nearest so 0=invalid and discontinuities hold."""
    return _pad_stack(frames, size, nearest=True)


def motus_t_view_slots(
    target_hw: tuple[int, int],
    compose_hw: tuple[int, int] = COMPOSE_HW,
) -> dict[str, tuple[int, int, int, int]]:
    """Pixel boxes on the padded canvas after ``pad_*_to_size(compose, target)``.

    Values are ``(r0, r1, c0, c1)`` half-open. Uses the same integer rounding as
    ``_pad_stack``, then splits the T by native head / wrist fractions.
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    ch, cw = int(compose_hw[0]), int(compose_hw[1])
    scale = min(target_h / ch, target_w / cw)
    new_h = max(1, int(ch * scale))
    new_w = max(1, int(cw * scale))
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    split = y0 + int(round(new_h * (HEAD_HW[0] / float(ch))))
    mid = x0 + new_w // 2
    y1 = y0 + new_h
    x1 = x0 + new_w
    return {
        "content": (y0, y1, x0, x1),
        "head": (y0, split, x0, x1),
        "left_wrist": (split, y1, x0, mid),
        "right_wrist": (split, y1, mid, x1),
    }


def _as_rgb_hwc(frame: np.ndarray, hw: tuple[int, int], *, nearest: bool) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"RGB frame must be HWC3, got {frame.shape}")
    h, w = hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return np.ascontiguousarray(frame)
    if cv2 is None:
        raise ImportError("cv2 is required to resize robotwin_3d views")
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


def _as_depth_hw(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError(f"depth frame must be HW, got {frame.shape}")
    h, w = hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return np.ascontiguousarray(frame)
    if cv2 is None:
        raise ImportError("cv2 is required to resize robotwin_3d depth")
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)


def _pad_stack(frames: np.ndarray, size: tuple[int, int], *, nearest: bool) -> np.ndarray:
    target_h, target_w = (int(size[0]), int(size[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"invalid pad size {size}")
    out = []
    for frame in frames:
        height, width = frame.shape[:2]
        scale = min(target_h / height, target_w / width)
        new_h = max(1, int(height * scale))
        new_w = max(1, int(width * scale))
        if cv2 is None:
            raise ImportError("cv2 is required for robotwin_3d pad")
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        padded = np.zeros((target_h, target_w) + frame.shape[2:], dtype=frame.dtype)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        padded[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        out.append(padded)
    return np.stack(out, axis=0)


def _ffmpeg_bin() -> str:
    configured = os.environ.get("MOTUS_FFMPEG_BIN")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    default = "/code1/7TB/v2/bin/ffmpeg-static/bin/ffmpeg"
    if os.path.exists(default):
        return default
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found for robotwin_3d decode") from exc


_LOGGER = logging.getLogger(__name__)
# AVERROR(EAGAIN) = -11; ffmpeg surfaces it as exit 245 (256-11).
_FFMPEG_TRANSIENT_EXITS = {11, 245}
_FFMPEG_TRANSIENT_ERRNOS = {errno.EAGAIN}
if hasattr(errno, "EWOULDBLOCK"):
    _FFMPEG_TRANSIENT_ERRNOS.add(errno.EWOULDBLOCK)
_FFMPEG_TRANSIENT_RETRIES = 3


def _ffmpeg_input_path(cmd: list[str]) -> str:
    try:
        return cmd[cmd.index("-i") + 1]
    except (ValueError, IndexError):
        return "?"


def _is_transient_ffmpeg(exc: BaseException) -> bool:
    if isinstance(exc, subprocess.CalledProcessError):
        return exc.returncode in _FFMPEG_TRANSIENT_EXITS
    if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
        return exc.errno in _FFMPEG_TRANSIENT_ERRNOS
    return False


def _ffmpeg_retry_delay(attempt: int) -> float:
    """0.5 / 1.0 / 2.0s base, jittered to 0.7x–1.3x so ranks do not retry together."""
    return (0.5 * (2 ** attempt)) * (0.7 + 0.6 * random.random())


def _ffmpeg_check_output(cmd: list[str]) -> bytes:
    """Run ffmpeg; retry transient EAGAIN / fork failures."""
    last: BaseException | None = None
    for attempt in range(_FFMPEG_TRANSIENT_RETRIES):
        try:
            return subprocess.check_output(cmd)
        except (subprocess.CalledProcessError, OSError) as exc:
            last = exc
            if not _is_transient_ffmpeg(exc) or attempt + 1 >= _FFMPEG_TRANSIENT_RETRIES:
                raise
            delay = _ffmpeg_retry_delay(attempt)
            detail = (
                f"exit {exc.returncode}"
                if isinstance(exc, subprocess.CalledProcessError)
                else f"OSError errno={exc.errno}"
            )
            _LOGGER.warning(
                "ffmpeg %s on %s, retry %d/%d in %.2fs",
                detail,
                _ffmpeg_input_path(cmd),
                attempt + 1,
                _FFMPEG_TRANSIENT_RETRIES - 1,
                delay,
            )
            time.sleep(delay)
    raise last  # pragma: no cover


def decode_depth_mm_frame_map(
    path: str | Path,
    frame_indices: Sequence[int],
    *,
    height: int = NATIVE_HW[0],
    width: int = NATIVE_HW[1],
) -> dict[int, np.ndarray]:
    """Decode selected FFV1 gray16le frames as uint16 millimetres."""
    unique = sorted({max(0, int(i)) for i in frame_indices})
    if not unique:
        raise ValueError("frame_indices cannot be empty")
    bpf = int(height) * int(width) * 2
    result: dict[int, np.ndarray] = {}
    # Same select-chunk limit as motus_data ffmpeg RGB decode.
    chunk_size = 64
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start : start + chunk_size]
        select = "+".join(f"eq(n\\,{idx})" for idx in chunk)
        raw = _ffmpeg_check_output(
            [
                _ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                # decoder / filter / encoder all single-thread: ffmpeg sizes
                # thread pools from nproc (192 here) and 48–64 procs overflow
                # the pod pids quota (exit 245 / EAGAIN).
                "-threads",
                "1",
                "-i",
                str(path),
                "-filter_threads",
                "1",
                "-vf",
                f"select={select}",
                "-vsync",
                "0",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "-",
            ]
        )
        if len(raw) != bpf * len(chunk):
            raise RuntimeError(
                f"depth decode {path} frames {chunk[0]}..{chunk[-1]}: "
                f"got {len(raw)} bytes, want {bpf * len(chunk)}"
            )
        frames = np.frombuffer(raw, dtype=np.uint16).reshape(
            len(chunk), int(height), int(width)
        )
        for idx, frame in zip(chunk, frames):
            result[idx] = np.ascontiguousarray(frame)
    return result


def _require_pyarrow():
    if pq is None:
        raise ImportError("robotwin_3d parquet reading requires pyarrow")
    return pq


class Robotwin3dEpisodeReader(EpisodeSourceReader):
    """Read 3 RGB mp4 + parquet 14D qpos; compose Motus T then pad."""

    source_dataset = SOURCE_DATASET

    def __init__(
        self,
        *,
        resize_mode: str = "pad",
        video_decoder=None,
    ) -> None:
        if resize_mode not in {"pad", "interpolate"}:
            raise ValueError("robotwin_3d resize_mode must be 'pad' or 'interpolate'")
        self._resize_mode = resize_mode
        self._video_decoder = video_decoder or decode_video_frame_map_ffmpeg
        os.environ.setdefault("MOTUS_FFMPEG_BIN", _ffmpeg_bin())

    def bind_catalog(self, metadata: Mapping[str, Any]) -> None:
        if str(metadata.get("source")) != self.source_dataset:
            raise EpisodeRepositoryError(
                f"robotwin_3d catalog source mismatch: {metadata.get('source')!r}"
            )

    @staticmethod
    def _meta(record: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
        return record[modality].get("metadata") or {}

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        dim = int(self._meta(record, "state").get("dimension", QPOS_DIM))
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_decoded_video_frame",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "layout": "motus_t_compose",
                    "native_compose_size": list(COMPOSE_HW),
                    "default_resize_mode": self._resize_mode,
                },
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (dim,),
                "zero_based_qpos_timestep",
                {"normalization": "none", "column": "observation.state"},
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (dim,),
                "zero_based_qpos_timestep",
                {"normalization": "none", "column": "action"},
            ),
            "text": ModalitySchema(
                "text",
                len(record["text"]["items"]),
                "structured_text_item",
                (),
                "zero_based_catalog_text_item",
                {},
            ),
        }

    def _view_paths(self, record: Mapping[str, Any]) -> dict[str, str]:
        meta = self._meta(record, "video")
        views = meta.get("views") or {}
        missing = [cam for cam in CAMS if cam not in views]
        if missing:
            raise EpisodeRepositoryError(
                f"robotwin_3d catalog missing RGB views {missing} "
                f"for {record.get('episode_id')}"
            )
        return {cam: str(views[cam]) for cam in CAMS}

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(
                f"unsupported video output_format={output_format!r}"
            )
        if not indices:
            empty = torch.empty((0, COMPOSE_HW[0], COMPOSE_HW[1], 3), dtype=torch.uint8)
            if output_format == "rgb_uint8_hwc":
                return empty
            return empty.permute(0, 3, 1, 2).float()
        paths = self._view_paths(record)
        decoded = {
            cam: self._video_decoder(paths[cam], indices) for cam in CAMS
        }
        composed = []
        for idx in indices:
            composed.append(
                compose_motus_t_rgb(
                    decoded["cam_high"][int(idx)],
                    decoded["cam_left_wrist"][int(idx)],
                    decoded["cam_right_wrist"][int(idx)],
                )
            )
        native = np.stack(composed, axis=0)
        if video_size is None:
            frames = native
        elif self._resize_mode == "pad":
            frames = pad_rgb_to_size(native, (int(video_size[0]), int(video_size[1])))
        else:
            chw = torch.from_numpy(native).permute(0, 3, 1, 2).float().div_(255.0)
            chw = F.interpolate(
                chw,
                size=(int(video_size[0]), int(video_size[1])),
                mode="bilinear",
                align_corners=False,
            )
            if output_format == "rgb_float32_chw":
                return chw
            return (
                chw.mul(255.0)
                .round_()
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        tensor = torch.from_numpy(np.ascontiguousarray(frames))
        if output_format == "rgb_uint8_hwc":
            return tensor
        return tensor.permute(0, 3, 1, 2).float().div_(255.0)

    def _read_qpos(
        self, record: Mapping[str, Any], column: str, indices: Sequence[int]
    ) -> torch.Tensor:
        locator = record["state" if column == "observation.state" else "action"][
            "locator"
        ]
        path = str(locator["uri"])
        expected = int(record["state"]["count"])
        parquet = _require_pyarrow()
        table = parquet.read_table(path, columns=[column])
        if table.num_rows != expected:
            raise EpisodeRepositoryError(
                f"parquet rows {table.num_rows} != catalog count {expected}: {path}"
            )
        values = np.stack(
            [np.asarray(row, dtype=np.float32) for row in table.column(column).to_pylist()],
            axis=0,
        )
        if values.ndim != 2 or int(values.shape[1]) != QPOS_DIM:
            raise EpisodeRepositoryError(
                f"{column} shape {values.shape} is not (T, {QPOS_DIM}): {path}"
            )
        if not indices:
            return torch.empty((0, QPOS_DIM), dtype=torch.float32)
        take = torch.from_numpy(values[list(indices)].copy())
        return take.to(dtype=torch.float32)

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._read_qpos(record, "observation.state", indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._read_qpos(record, "action", indices)


class Robotwin3dInlineDepthReader:
    """Compose 3 FFV1 mkv streams to Motus T millimetres, then pad like RGB."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        os.environ.setdefault("MOTUS_FFMPEG_BIN", _ffmpeg_bin())

    def read_mm(
        self,
        episode_id: str,
        frame_indices: Sequence[int],
        *,
        target_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor | None, bool]:
        try:
            ep_i = parse_episode_index(episode_id)
        except ValueError:
            return None, False
        paths = robotwin_3d_paths(self.root, ep_i)
        if not all(paths[f"depth_{cam}"].is_file() for cam in CAMS):
            return None, False
        if not frame_indices:
            empty = torch.empty((0, COMPOSE_HW[0], COMPOSE_HW[1]), dtype=torch.int32)
            return empty, True
        decoded = {
            cam: decode_depth_mm_frame_map(paths[f"depth_{cam}"], frame_indices)
            for cam in CAMS
        }
        composed = []
        for idx in frame_indices:
            composed.append(
                compose_motus_t_depth_mm(
                    decoded["cam_high"][int(idx)],
                    decoded["cam_left_wrist"][int(idx)],
                    decoded["cam_right_wrist"][int(idx)],
                )
            )
        mm = np.stack(composed, axis=0)
        if target_hw is not None and tuple(target_hw) != COMPOSE_HW:
            mm = pad_depth_mm_to_size(mm, (int(target_hw[0]), int(target_hw[1])))
        return torch.from_numpy(np.ascontiguousarray(mm)).to(torch.int32), True


def _task_of(task_map: Mapping[str, Any], episode_index: int) -> str:
    tasks = task_map.get("tasks") or {}
    for name, spec in tasks.items():
        if int(spec["episode_start"]) <= episode_index <= int(spec["episode_end"]):
            return str(name)
    return "unknown"


# V13 正在读的 catalog。all_seen 构建禁止写进这个目录。
LIVE_V13_CATALOG = Path(
    "/code1/zxw_data/pretrain/ThinkerWAM-Depth-Action-RGB-RoboTwin/"
    "tmp/catalogs/robotwin_3d/robotwin_3d.episode_catalog.jsonl"
)


def _instructions_of(
    rec: Mapping[str, Any], task: str, *, all_seen: bool = False
) -> list[str]:
    """LeRobot ``episodes.jsonl`` 的 ``tasks`` = 该 episode 的官方 seen 列表。

    默认只留第一条（V13 行为）。``all_seen=True`` 保留全部非空句，去重保序。
    """
    texts: list[str] = []
    seen = set()
    for item in rec.get("tasks") or []:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
        if not all_seen:
            break
    if not texts:
        texts = [task.replace("_", " ")]
    return texts


def _instruction_of(rec: Mapping[str, Any], task: str) -> str:
    return _instructions_of(rec, task, all_seen=False)[0]


def build_robotwin_3d_episode_catalog(
    *,
    data_root: str | Path,
    output_path: str | Path,
    catalog_id: str = "robotwin_3d_20260908_flexpi_v1",
    data_version: str = "flexpi_robotwin_3d_v1",
    max_episodes: int | None = None,
    force: bool = False,
    all_seen: bool = False,
) -> dict[str, Any]:
    """Write a v2 Episode Catalog + sidecar meta from the LeRobot tree."""
    root = Path(data_root)
    output = Path(output_path)
    meta_output = (
        output.with_suffix(".meta.json")
        if output.suffix == ".jsonl"
        else Path(str(output) + ".meta.json")
    )
    if output.resolve() == LIVE_V13_CATALOG.resolve() and all_seen:
        raise RuntimeError(
            "refuse to write all_seen catalog over the live V13 catalog: "
            f"{LIVE_V13_CATALOG}"
        )
    if not force and (output.exists() or meta_output.exists()):
        raise FileExistsError(f"catalog exists; pass force=True: {output}")
    episodes_path = root / "meta" / "episodes.jsonl"
    task_map = json.loads((root / "task_episode_map.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    tmp_meta = meta_output.with_name(f".{meta_output.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    written = 0
    started = time.monotonic()
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for line in open(episodes_path, encoding="utf-8"):
                rec = json.loads(line)
                ep_i = int(rec["episode_index"])
                length = int(rec["length"])
                paths = robotwin_3d_paths(root, ep_i)
                task = _task_of(task_map, ep_i)
                instructions = _instructions_of(rec, task, all_seen=all_seen)
                record = {
                    "episode_id": f"{SOURCE_DATASET}:{ep_i:06d}",
                    "robot": "aloha",
                    "video": {
                        "count": length,
                        "storage": {
                            "type": "file",
                            "path": str(paths["rgb_cam_high"]),
                        },
                        "meta": {
                            "layout": "motus_t_compose",
                            "native_size": [NATIVE_HW[0], NATIVE_HW[1]],
                            "compose_size": [COMPOSE_HW[0], COMPOSE_HW[1]],
                            "fps": 50,
                            "views": {
                                cam: str(paths[f"rgb_{cam}"]) for cam in CAMS
                            },
                            "depth_views": {
                                cam: str(paths[f"depth_{cam}"]) for cam in CAMS
                            },
                            "depth_unit": "mm",
                            "depth_codec": "ffv1_gray16le",
                        },
                    },
                    "state": {
                        "count": length,
                        "storage": {
                            "type": "file",
                            "path": str(paths["parquet"]),
                        },
                        "meta": {
                            "column": "observation.state",
                            "dimension": QPOS_DIM,
                            "feature_names": list(QPOS_NAMES),
                            "normalization": "none",
                        },
                    },
                    "action": {
                        "count": length,
                        "storage": {
                            "type": "file",
                            "path": str(paths["parquet"]),
                        },
                        "meta": {
                            "column": "action",
                            "dimension": QPOS_DIM,
                            "feature_names": list(QPOS_NAMES),
                            "normalization": "none",
                            "semantics": "absolute_qpos",
                        },
                    },
                    "texts": [
                        {
                            "index": i,
                            "kind": "instruction_en",
                            "text": text,
                            "video_range": [0, length],
                            "meta": {
                                "task_name": task,
                                "seen_rank": i,
                                "n_seen": len(instructions),
                            },
                        }
                        for i, text in enumerate(instructions)
                    ],
                    "meta": {
                        "episode_index": ep_i,
                        "task_name": task,
                        "chunk": episode_chunk(ep_i),
                        "data_root": str(root),
                        "state_action_dim": QPOS_DIM,
                    },
                }
                validate_episode_record_v2(record)
                payload = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.write(payload.decode("utf-8"))
                digest.update(payload)
                written += 1
                if max_episodes is not None and written >= int(max_episodes):
                    break
            handle.flush()
            os.fsync(handle.fileno())
        if written <= 0:
            raise RuntimeError("robotwin_3d catalog wrote 0 episodes")
        metadata = {
            "format": MOTUS_EPISODE_CATALOG_METADATA_V2,
            "catalog_id": str(catalog_id),
            "source": SOURCE_DATASET,
            "child": SOURCE_DATASET,
            "episodes": written,
            "sha256": digest.hexdigest(),
            "meta": {
                "data_version": str(data_version),
                "data_root": str(root),
                "layout": "motus_t_compose",
                "all_seen": bool(all_seen),
                "z_near_m": DEFAULT_Z_NEAR_M,
                "z_far_m": DEFAULT_Z_FAR_M,
                "elapsed_seconds": time.monotonic() - started,
            },
        }
        validate_catalog_metadata_v2(metadata)
        with tmp.open("r", encoding="utf-8") as handle:
            for line in handle:
                validate_episode_catalog_binding(json.loads(line), metadata)
        tmp_meta.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if force:
            output.unlink(missing_ok=True)
            meta_output.unlink(missing_ok=True)
        os.replace(tmp, output)
        os.replace(tmp_meta, meta_output)
        return metadata
    finally:
        tmp.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)


def build_robotwin_3d_catalog_and_index(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    max_episodes: int | None = None,
    force: bool = False,
    all_seen: bool = False,
    catalog_id: str | None = None,
    data_version: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    catalog = output_dir / "robotwin_3d.episode_catalog.jsonl"
    if all_seen:
        catalog_id = catalog_id or "robotwin_3d_20260916_seen100"
        data_version = data_version or "flexpi_robotwin_3d_seen100_v1"
    else:
        catalog_id = catalog_id or "robotwin_3d_20260908_flexpi_v1"
        data_version = data_version or "flexpi_robotwin_3d_v1"
    metadata = build_robotwin_3d_episode_catalog(
        data_root=data_root,
        output_path=catalog,
        catalog_id=catalog_id,
        data_version=data_version,
        max_episodes=max_episodes,
        force=force,
        all_seen=all_seen,
    )
    index = build_catalog_offset_index(catalog, force=force)
    metadata["catalog_path"] = str(catalog)
    metadata["index_path"] = str(index["index_path"])
    return metadata


if SOURCE_DATASET not in EPISODE_SOURCE_READERS.names():
    register_episode_source_reader(SOURCE_DATASET, Robotwin3dEpisodeReader)
