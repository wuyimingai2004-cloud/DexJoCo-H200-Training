"""Motus EpisodeSourceReader for audited DexJoCo releases."""
from __future__ import annotations

import numpy as np
import torch

from .._bootstrap import ensure_motus_data_importable
from .contracts import SOURCE, CONTRACT
from .raw import compose_rgb, read_frames

ensure_motus_data_importable()
from motus_data.repository import EpisodeSourceReader, ModalitySchema
from motus_data.reader_registry import EPISODE_SOURCE_READERS, register_episode_source_reader
from motus_data.releases import EpisodeDatasetRelease, episode_dataset_releases, register_episode_dataset_release


class DexJoCoReader(EpisodeSourceReader):
    source_dataset = SOURCE

    def bind_catalog(self, metadata):
        if metadata.get("source") != SOURCE or metadata.get("meta", {}).get("contract") != CONTRACT:
            raise ValueError("DexJoCo catalog contract mismatch")

    def schemas(self, record):
        return {
            "video": ModalitySchema("video", record["video"]["frame_count"], "uint8", (480, 640, 3),
                                    "zero_based_decoded_video_frame", {}),
            **{k: ModalitySchema(k, record[k]["count"], "float32",
                                (record[k]["metadata"]["dimension"],), "zero_based_row", {})
               for k in ("state", "action")},
            "text": ModalitySchema("text", len(record["text"]["items"]), "structured_text_item",
                                   (), "zero_based_catalog_text_item", {}),
        }

    def read_video(self, record, indices, *, output_format, video_size=None):
        if video_size is not None and tuple(video_size) != (480, 640):
            raise ValueError("DexJoCo recipe uses a fixed 480x640 canvas")
        if output_format not in ("rgb_uint8_hwc", "rgb_float32_chw"):
            raise ValueError(output_format)
        paths = record["video"]["metadata"]["camera_paths"]
        frames = (compose_rgb([read_frames(p, indices) for p in paths]) if indices
                  else np.empty((0, 480, 640, 3), dtype=np.uint8))
        x = torch.from_numpy(frames)
        return x if output_format == "rgb_uint8_hwc" else x.permute(0, 3, 1, 2).float().div_(255.)

    def _rows(self, record, kind, indices):
        a = np.load(record[kind]["locator"]["uri"], mmap_mode="r", allow_pickle=False)
        return torch.from_numpy(np.array(a[list(indices)], dtype=np.float32, copy=True))

    def read_state(self, record, indices):
        return self._rows(record, "state", indices)

    def read_action(self, record, indices):
        return self._rows(record, "action", indices)


if SOURCE not in EPISODE_SOURCE_READERS.names():
    register_episode_source_reader(SOURCE, DexJoCoReader)
if not episode_dataset_releases(SOURCE):
    register_episode_dataset_release(EpisodeDatasetRelease(
        dataset=SOURCE, release="absolute_eef_v1",
        catalog_path="dexjoco/train.episode_catalog.jsonl",
        index_path="dexjoco/train.episode_catalog.index.sqlite",
        default_video_size=(480, 640), action_magnitude_guard_sources=(SOURCE,),
    ), default=True)
