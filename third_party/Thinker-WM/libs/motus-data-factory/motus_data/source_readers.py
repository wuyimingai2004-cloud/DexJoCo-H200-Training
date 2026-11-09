"""Episode readers shared by non-tar source families."""

from __future__ import annotations

import io
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import h5py
except ImportError:
    h5py = None

from motus_data.datasets.agibot_common import (
    normalize_agibot_eef_actions,
    read_agibot_eef_actions,
)
from motus_data.reader_registry import register_episode_source_reader
from motus_data.repository import (
    AgibotBetaEpisodeReader,
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
)


def _text_schema(record: Mapping[str, Any]) -> ModalitySchema:
    return ModalitySchema(
        "text",
        len(record["text"]["items"]),
        "structured_text_item",
        (),
        "zero_based_catalog_text_item",
        {},
    )


class _FileVideoEpisodeReader(AgibotBetaEpisodeReader):
    """Shared exact-frame MP4 reader for non-tar Episode sources."""

    @staticmethod
    def _file_locator(
        record: Mapping[str, Any], modality: str
    ) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        if locator["storage_backend"] != "file":
            raise EpisodeRepositoryError(
                f"{record['source_dataset']} {modality} requires file storage"
            )
        return locator

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        locator = self._file_locator(record, "video")
        return self._decode_video_path(
            str(locator["uri"]),
            indices,
            output_format=output_format,
            video_size=video_size,
        )


class AgibotEpisodeReader(_FileVideoEpisodeReader):
    """Read native Agibot MP4 and normalized EEF timeline files."""

    source_dataset = "agibot"

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_decoded_video_frame",
                {"formats": ["rgb_uint8_hwc", "rgb_float32_chw"]},
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (26,),
                "zero_based_canonical_proprio_timestep",
                {"normalization": "agibot_eef_v1"},
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (26,),
                "zero_based_canonical_action_timestep",
                {"normalization": "agibot_eef_v1"},
            ),
            "text": _text_schema(record),
        }

    def _canonical_timeline(self, record: Mapping[str, Any]) -> torch.Tensor:
        locator = self._file_locator(record, "state")
        timeline = normalize_agibot_eef_actions(
            read_agibot_eef_actions(str(locator["uri"]))
        )
        expected_state = int(record["state"]["count"])
        expected_action = int(record["action"]["count"])
        if (
            tuple(timeline.shape) != (expected_state, 26)
            or expected_state != expected_action
        ):
            raise EpisodeRepositoryError(
                "Agibot canonical timeline does not match Catalog counts"
            )
        return timeline

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._take(self._canonical_timeline(record), indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._take(self._canonical_timeline(record), indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timeline = self._canonical_timeline(record)
        return self._take(timeline, state_indices), self._take(
            timeline, action_indices
        )


class RobotwinEpisodeReader(_FileVideoEpisodeReader):
    """Read one RoboTwin MP4 and its aligned qpos tensor."""

    source_dataset = "robotwin"

    def __init__(
        self,
        video_decoder: Any | None = None,
        event_callback: Any | None = None,
        *,
        resize_mode: str = "pad",
    ):
        super().__init__(
            video_decoder=video_decoder,
            event_callback=event_callback,
        )
        if resize_mode not in {"pad", "interpolate"}:
            raise ValueError(
                "RoboTwin resize_mode must be 'pad' or 'interpolate'"
            )
        self._resize_mode = resize_mode

    @staticmethod
    def _resize_frames_with_padding(
        frames: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        """Apply the legacy RoboTwin aspect-preserving resize exactly."""
        if cv2 is None:
            raise ImportError("RoboTwin resize_mode=pad requires cv2")
        target_h, target_w = (int(value) for value in size)
        if target_h <= 0 or target_w <= 0:
            raise EpisodeRepositoryError(
                "RoboTwin video_size values must be positive"
            )

        resized = []
        for frame in frames.detach().cpu().numpy():
            height, width = frame.shape[:2]
            scale = min(target_h / height, target_w / width)
            new_h = max(1, int(height * scale))
            new_w = max(1, int(width * scale))
            resized_frame = cv2.resize(frame, (new_w, new_h))
            padded = np.zeros(
                (target_h, target_w, frame.shape[2]),
                dtype=frame.dtype,
            )
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[
                y_offset : y_offset + new_h,
                x_offset : x_offset + new_w,
            ] = resized_frame
            resized.append(padded)
        return torch.from_numpy(np.stack(resized, axis=0)).to(torch.uint8)

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
        if video_size is None or self._resize_mode == "interpolate":
            return super().read_video(
                record,
                indices,
                output_format=output_format,
                video_size=video_size,
            )

        native = super().read_video(
            record,
            indices,
            output_format="rgb_uint8_hwc",
            video_size=None,
        )
        padded = self._resize_frames_with_padding(native, video_size)
        if output_format == "rgb_uint8_hwc":
            return padded
        return padded.permute(0, 3, 1, 2).float().div_(255.0)

    @staticmethod
    def _dimension(record: Mapping[str, Any]) -> int:
        state_meta = record["state"].get("metadata") or {}
        dimension = int(
            state_meta.get(
                "dimension",
                (record.get("metadata") or {}).get("state_action_dim", 0),
            )
        )
        if dimension <= 0:
            raise EpisodeRepositoryError(
                "RoboTwin Catalog must declare state/action dimension"
            )
        return dimension

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        dimension = self._dimension(record)
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_decoded_video_frame",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "native_frame_layout": "upstream_composite",
                    "default_resize_mode": self._resize_mode,
                    "native_resolution_preserved_when_video_size_is_none": True,
                },
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (dimension,),
                "zero_based_qpos_timestep",
                {"normalization": "none"},
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (dimension,),
                "zero_based_qpos_timestep",
                {"normalization": "none"},
            ),
            "text": _text_schema(record),
        }

    def _timeline(self, record: Mapping[str, Any]) -> torch.Tensor:
        locator = self._file_locator(record, "state")
        try:
            value = torch.load(
                str(locator["uri"]),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            value = torch.load(str(locator["uri"]), map_location="cpu")
        if isinstance(value, Mapping):
            value = value.get("qpos")
        if not torch.is_tensor(value) or value.ndim != 2:
            raise EpisodeRepositoryError("RoboTwin qpos must be a rank-2 tensor")
        timeline = value.detach().to(dtype=torch.float32, device="cpu")
        expected_count = int(record["state"]["count"])
        expected_dim = self._dimension(record)
        if (
            tuple(timeline.shape) != (expected_count, expected_dim)
            or expected_count != int(record["action"]["count"])
        ):
            raise EpisodeRepositoryError(
                "RoboTwin qpos does not match Catalog count/dimension"
            )
        return timeline

    @staticmethod
    def _take(timeline: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        if not indices:
            return timeline.new_empty((0, int(timeline.shape[1])))
        return timeline[torch.tensor(indices, dtype=torch.long)].clone()

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._take(self._timeline(record), indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._take(self._timeline(record), indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timeline = self._timeline(record)
        return self._take(timeline, state_indices), self._take(
            timeline, action_indices
        )


class RobotmindEpisodeReader(EpisodeSourceReader):
    """Read schema-described RobotMind HDF5 video and numeric timelines."""

    source_dataset = "robotmind"

    @staticmethod
    def _locator(
        record: Mapping[str, Any], modality: str
    ) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        if locator["storage_backend"] != "hdf5":
            raise EpisodeRepositoryError(
                f"RobotMind {modality} requires hdf5 storage"
            )
        return locator

    @staticmethod
    def _meta(record: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
        return record[modality].get("metadata") or {}

    @classmethod
    def _dimension(cls, record: Mapping[str, Any]) -> int:
        dimension = int(
            cls._meta(record, "state").get(
                "dimension",
                (record.get("metadata") or {}).get("state_action_dim", 0),
            )
        )
        if dimension <= 0:
            raise EpisodeRepositoryError(
                "RobotMind Catalog must declare state/action dimension"
            )
        return dimension

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        dimension = self._dimension(record)
        state_meta = self._meta(record, "state")
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_hdf5_jpeg_timestep",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "key": self._meta(record, "video").get("key"),
                },
            ),
            "state": ModalitySchema(
                "state",
                int(record["state"]["count"]),
                "float32",
                (dimension,),
                "zero_based_aligned_hdf5_timestep",
                {
                    "normalization": "none",
                    "keys": list(state_meta.get("keys") or []),
                },
            ),
            "action": ModalitySchema(
                "action",
                int(record["action"]["count"]),
                "float32",
                (dimension,),
                "zero_based_aligned_hdf5_timestep",
                {
                    "normalization": "none",
                    "keys": list(
                        self._meta(record, "action").get("keys") or []
                    ),
                },
            ),
            "text": _text_schema(record),
        }

    @staticmethod
    def _require_h5py() -> None:
        if h5py is None:
            raise ImportError("RobotMind Episode reading requires h5py")

    @staticmethod
    def _rows(dataset: Any, indices: Sequence[int]) -> np.ndarray:
        if not indices:
            width = int(np.prod(dataset.shape[1:])) if dataset.ndim > 1 else 1
            return np.empty((0, width), dtype=np.float32)
        unique = sorted(set(int(index) for index in indices))
        values = np.asarray(dataset[unique])
        if values.ndim == 1:
            values = values[:, None]
        values = values.reshape(len(unique), -1)
        positions = {index: position for position, index in enumerate(unique)}
        return values[[positions[int(index)] for index in indices]]

    @classmethod
    def _numeric(
        cls,
        handle: Any,
        keys: Sequence[str],
        indices: Sequence[int],
    ) -> torch.Tensor:
        if not keys:
            raise EpisodeRepositoryError(
                "RobotMind Catalog numeric modality has no ordered keys"
            )
        missing = [key for key in keys if key not in handle]
        if missing:
            raise EpisodeRepositoryError(
                f"RobotMind HDF5 is missing keys={missing}"
            )
        parts = [cls._rows(handle[key], indices) for key in keys]
        return torch.from_numpy(np.concatenate(parts, axis=-1)).float()

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        self._require_h5py()
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(
                f"unsupported video output_format={output_format!r}"
            )
        locator = self._locator(record, "video")
        key = str(self._meta(record, "video").get("key") or "")
        if not key:
            raise EpisodeRepositoryError("RobotMind video HDF5 key is missing")
        arrays = []
        with h5py.File(str(locator["uri"]), "r") as handle:
            if key not in handle:
                raise EpisodeRepositoryError(
                    f"RobotMind HDF5 is missing video key={key!r}"
                )
            encoded = handle[key]
            for index in indices:
                with Image.open(io.BytesIO(bytes(encoded[int(index)]))) as image:
                    arrays.append(np.asarray(image.convert("RGB")).copy())
        if len({tuple(array.shape) for array in arrays}) != 1:
            raise EpisodeRepositoryError(
                "RobotMind decoded video frames have inconsistent shapes"
            )
        frames = torch.from_numpy(np.stack(arrays)).to(torch.uint8)
        if output_format == "rgb_uint8_hwc" and video_size is None:
            return frames
        chw = frames.permute(0, 3, 1, 2).float().div_(255.0)
        if video_size is not None:
            chw = F.interpolate(
                chw,
                size=tuple(int(value) for value in video_size),
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

    def _read_numeric(
        self,
        record: Mapping[str, Any],
        modality: str,
        indices: Sequence[int],
    ) -> torch.Tensor:
        self._require_h5py()
        locator = self._locator(record, modality)
        keys = tuple(self._meta(record, modality).get("keys") or ())
        with h5py.File(str(locator["uri"]), "r") as handle:
            result = self._numeric(handle, keys, indices)
        if int(result.shape[-1]) != self._dimension(record):
            raise EpisodeRepositoryError(
                f"RobotMind {modality} dimension mismatch"
            )
        return result

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._read_numeric(record, "state", indices)

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        return self._read_numeric(record, "action", indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_h5py()
        state_locator = self._locator(record, "state")
        action_locator = self._locator(record, "action")
        if str(state_locator["uri"]) != str(action_locator["uri"]):
            return super().read_state_action(
                record, state_indices, action_indices
            )
        state_keys = tuple(self._meta(record, "state").get("keys") or ())
        action_keys = tuple(self._meta(record, "action").get("keys") or ())
        with h5py.File(str(state_locator["uri"]), "r") as handle:
            state = self._numeric(handle, state_keys, state_indices)
            action = self._numeric(handle, action_keys, action_indices)
        dimension = self._dimension(record)
        if int(state.shape[-1]) != dimension or int(action.shape[-1]) != dimension:
            raise EpisodeRepositoryError(
                "RobotMind state/action dimension mismatch"
            )
        return state, action


# Built-in readers self-register exactly like third-party plugins: one reader
# class bound to the Catalog ``source`` name. EpisodeRepository imports this
# module lazily via _ensure_builtin_source_readers().
register_episode_source_reader("agibot", AgibotEpisodeReader)
register_episode_source_reader("robotwin", RobotwinEpisodeReader)
register_episode_source_reader("robotmind", RobotmindEpisodeReader)
