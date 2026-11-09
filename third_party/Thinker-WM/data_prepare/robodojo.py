"""RoboDojo EpisodeSourceReader (motus_data protocol) over ``robodojo_core``.

Reader spec (per indexed source in the data YAML)::

    reader: {type: robodojo, field_map: <path to audited field map>, robot: <name>}

Depth goes through ``RoboDojoInlineDepthReader.read_mm`` (same composition as
RGB), wired into ``data_prepare.depth_attach.DepthAttacher`` by source name;
it is never routed through the ``robotwin_3d`` reader.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ._bootstrap import ensure_motus_data_importable
from .robodojo_core import SOURCE_DATASET, FieldMap, RoboDojoHDF5, pad_or_resize

ensure_motus_data_importable()

from motus_data.reader_registry import (  # noqa: E402
    EPISODE_SOURCE_READERS,
    register_episode_source_reader,
)
from motus_data.releases import (  # noqa: E402
    EpisodeDatasetRelease,
    episode_dataset_releases,
    register_episode_dataset_release,
)
from motus_data.repository import (  # noqa: E402
    EpisodeRepositoryError,
    EpisodeSourceReader,
    ModalitySchema,
)


def _pad_or_resize(chw: torch.Tensor, size, mode: str, nearest: bool) -> torch.Tensor:
    try:
        return pad_or_resize(chw, size, mode, nearest)
    except ValueError as e:
        raise EpisodeRepositoryError(str(e)) from e


class RoboDojoEpisodeReader(EpisodeSourceReader):
    source_dataset = SOURCE_DATASET

    def __init__(self, *, field_map: str | None = None, robot: str = "robodojo",
                 resize_mode: str = "pad") -> None:
        # The registry instantiates every registered reader without options;
        # a reader without a field map only fails once bound to a catalog.
        if resize_mode not in {"pad", "interpolate"}:
            raise ValueError("robodojo resize_mode must be 'pad' or 'interpolate'")
        self.fm = None if field_map is None else FieldMap.load(field_map)
        self.robot = robot
        self._io = None if self.fm is None else RoboDojoHDF5(self.fm)
        self._resize_mode = resize_mode

    def bind_catalog(self, metadata: Mapping[str, Any]) -> None:
        if self.fm is None:
            raise EpisodeRepositoryError(
                "robodojo reader has no field map: set reader.field_map on the source "
                "(or --robodojo-field-map for build_sampling_ranges)")
        if str(metadata.get("source")) != self.source_dataset:
            raise EpisodeRepositoryError(f"robodojo catalog source mismatch: {metadata.get('source')!r}")
        digest = (metadata.get("meta") or {}).get("field_map_digest")
        if digest and self.fm.digest and digest != self.fm.digest:
            raise EpisodeRepositoryError(
                f"catalog built with field map {digest}, reader uses {self.fm.digest}")

    @staticmethod
    def _path(record: Mapping[str, Any], modality: str) -> str:
        return str(record[modality]["locator"]["uri"])

    def _rows(self, record, indices: Sequence[int]):
        rows = self._io.frame_rows(self._path(record, "state"))
        return list(indices) if rows is None else [int(rows[i]) for i in indices]

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        meta_s = record["state"].get("metadata") or {}
        meta_a = record["action"].get("metadata") or {}
        return {
            "video": ModalitySchema("video", int(record["video"]["frame_count"]), "uint8|float32",
                                    (None, None, 3), "zero_based_decoded_video_frame",
                                    {"formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                                     "layout": "robodojo_field_map",
                                     "native_compose_size": list(self.fm.layout.size),
                                     "default_resize_mode": self._resize_mode}),
            "state": ModalitySchema("state", int(record["state"]["count"]), "float32",
                                    (meta_s.get("dimension"),), "zero_based_state_row",
                                    {"normalization": "none"}),
            "action": ModalitySchema("action", int(record["action"]["count"]), "float32",
                                     (meta_a.get("dimension"),), "zero_based_action_row",
                                     {"normalization": "none",
                                      "semantics": self.fm.action_semantics}),
            "text": ModalitySchema("text", len(record["text"]["items"]), "structured_text_item",
                                   (), "zero_based_catalog_text_item", {}),
        }

    def read_video(self, record, indices, *, output_format: str, video_size=None) -> torch.Tensor:
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(f"unsupported video output_format={output_format!r}")
        h, w = self.fm.layout.size
        if not indices:
            empty = torch.empty((0, h, w, 3), dtype=torch.uint8)
            return empty if output_format == "rgb_uint8_hwc" else empty.permute(0, 3, 1, 2).float()
        x = torch.from_numpy(np.ascontiguousarray(self._io.read_rgb(self._path(record, "video"), indices)))
        chw = x.permute(0, 3, 1, 2)
        if video_size is not None:
            chw = _pad_or_resize(chw if self._resize_mode == "pad" else chw.float() / 255.0,
                                 video_size, self._resize_mode, nearest=False)
            if self._resize_mode == "interpolate":
                if output_format == "rgb_float32_chw":
                    return chw
                return chw.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        if output_format == "rgb_uint8_hwc":
            return chw.permute(0, 2, 3, 1).contiguous()
        return chw.float().div_(255.0)

    def read_state(self, record, indices) -> torch.Tensor:
        return torch.from_numpy(self._io.read_rows(self._path(record, "state"), "state",
                                                   self._rows(record, indices)))

    def read_action(self, record, indices) -> torch.Tensor:
        return torch.from_numpy(self._io.read_rows(self._path(record, "action"), "action",
                                                   self._rows(record, indices)))

    def depth_reader(self) -> "RoboDojoInlineDepthReader":
        return RoboDojoInlineDepthReader(self.fm, resize_mode=self._resize_mode)


class RoboDojoInlineDepthReader:
    """``read_mm(episode_id, frame_indices, target_hw)`` for DepthAttacher.

    The episode file is resolved through ``register_episode`` (filled by the
    dataset from catalog records), so episode ids stay opaque.
    """

    def __init__(self, field_map: FieldMap, resize_mode: str = "pad"):
        if not field_map.depth_views:
            raise ValueError("field map has no depth views")
        self._io = RoboDojoHDF5(field_map)
        self._paths: dict = {}
        self._resize_mode = resize_mode

    def register_episode(self, episode_id: str, path: str) -> None:
        self._paths[str(episode_id)] = str(path)

    def read_mm(self, episode_id: str, frame_indices, *, target_hw=None):
        path = self._paths.get(str(episode_id))
        if path is None:
            return None, False
        mm, _ = self._io.read_depth_mm(path, frame_indices)
        t = torch.from_numpy(np.ascontiguousarray(mm)).unsqueeze(1)
        if target_hw is not None:
            t = _pad_or_resize(t, target_hw, self._resize_mode, nearest=True)
        return t[:, 0], True


if SOURCE_DATASET not in EPISODE_SOURCE_READERS.names():
    register_episode_source_reader(SOURCE_DATASET, RoboDojoEpisodeReader)

# Runtime defaults only: training sources must set catalog_path / index_path and
# a reader spec with the field map, which override these placeholders.
if not episode_dataset_releases(SOURCE_DATASET):
    register_episode_dataset_release(
        EpisodeDatasetRelease(
            dataset=SOURCE_DATASET,
            release="field_map_v1",
            catalog_path="robodojo/robodojo.episode_catalog.jsonl",
            index_path="robodojo/robodojo.episode_catalog.index.sqlite",
            default_video_size=(480, 640),
            action_magnitude_guard_sources=(SOURCE_DATASET,),
        ),
        default=True,
    )
