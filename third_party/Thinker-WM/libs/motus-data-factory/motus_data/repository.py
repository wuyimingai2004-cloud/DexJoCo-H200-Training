"""Catalog-backed complete Episode access with source-specific readers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from motus_data.agibot_beta_normalization import (
    AgibotBetaNormalizationArtifact,
    load_agibot_beta_normalization_artifact,
)
from motus_data.agibot_beta_profiles import (
    AgibotBetaEffectorProfile,
    get_agibot_beta_effector_profile,
    read_agibot_beta_native_timeline,
)
from motus_data.datasets.agibot_common import (
    decode_video_frame_map_ffmpeg,
    normalize_agibot_eef_actions,
    read_agibot_eef_actions,
)
from motus_data.catalog.sqlite_index import CatalogOffsetIndex
from motus_data.reader_registry import EPISODE_SOURCE_READERS
from motus_data.index_contracts import (
    MOTUS_EPISODE_CATALOG_METADATA_V1,
    MOTUS_EPISODE_CATALOG_METADATA_V2,
    validate_catalog_metadata,
    validate_episode_catalog_binding,
    validate_sample_selection_v1,
)
from motus_data.storage import BoundedFileReader


class EpisodeRepositoryError(ValueError):
    """Raised when a Catalog or Episode cannot be accessed safely."""


class EpisodeMemoryLimitError(EpisodeRepositoryError):
    """Raised before a complete video would exceed the caller's memory limit."""


@dataclass(frozen=True)
class ModalitySchema:
    name: str
    count: int
    dtype: str
    item_shape: tuple[int | None, ...]
    index_semantics: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TextItem:
    index: int
    kind: str
    text: str
    frame_range: tuple[int, int] | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: str
    catalog_id: str
    source_dataset: str
    source_child: str
    robot_name: str
    split: str | None
    video_count: int
    state_count: int
    action_count: int
    text_count: int
    modality_schemas: Mapping[str, ModalitySchema]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VideoChunk:
    indices: tuple[int, ...]
    frames: torch.Tensor


@dataclass(frozen=True)
class EpisodeData:
    info: EpisodeInfo
    video: torch.Tensor | None
    state: torch.Tensor | None
    action: torch.Tensor | None
    text: tuple[TextItem, ...] | None


@dataclass(frozen=True)
class EpisodeSelectionData:
    info: EpisodeInfo
    video_frame_indices: tuple[int, ...]
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    text_indices: tuple[int, ...]
    video: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    text: tuple[TextItem, ...]


VideoDecoder = Callable[[str, Iterable[int]], Mapping[int, np.ndarray]]
ReaderEventCallback = Callable[[str, Mapping[str, Any]], None]
SelectionPreflightValidator = Callable[
    [
        EpisodeInfo,
        Mapping[str, Any],
        torch.Tensor,
        torch.Tensor,
        tuple[TextItem, ...],
    ],
    None,
]


def _metadata_path(catalog_path: Path) -> Path:
    if catalog_path.suffix == ".jsonl":
        return catalog_path.with_suffix(".meta.json")
    return Path(str(catalog_path) + ".meta.json")


def _normalize_v2_storage(storage: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "storage_backend": (
            "tar_member" if storage["type"] == "tar" else str(storage["type"])
        ),
        "uri": str(storage["path"]),
    }
    if "member" in storage:
        result["member"] = str(storage["member"])
    if "offset" in storage:
        result["offset_data"] = int(storage["offset"])
    if "size" in storage:
        result["size"] = int(storage["size"])
    if "parts" in storage:
        result["parts"] = [str(path) for path in storage["parts"]]
    return result


def _normalize_catalog_record(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Map concise v2 storage fields into the source-reader internal shape."""

    if metadata.get("format") != MOTUS_EPISODE_CATALOG_METADATA_V2:
        return record

    def modality(name: str, *, video: bool = False) -> dict[str, Any]:
        raw = record[name]
        result = {
            ("frame_count" if video else "count"): int(raw["count"]),
            "locator": _normalize_v2_storage(raw["storage"]),
        }
        if "meta" in raw:
            result["metadata"] = copy.deepcopy(raw["meta"])
        return result

    text_items = []
    for raw in record["texts"]:
        item = {
            "index": int(raw["index"]),
            "kind": str(raw["kind"]),
            "text": str(raw["text"]),
        }
        if "video_range" in raw:
            item["frame_range"] = list(raw["video_range"])
        if "meta" in raw:
            item["metadata"] = copy.deepcopy(raw["meta"])
        text_items.append(item)

    return {
        "episode_id": str(record["episode_id"]),
        "source_dataset": str(metadata["source"]),
        "source_child": str(metadata["child"]),
        "robot_name": str(record["robot"]),
        "video": modality("video", video=True),
        "state": modality("state"),
        "action": modality("action"),
        "text": {"items": text_items},
        "metadata": copy.deepcopy(record.get("meta") or {}),
    }


def _indices(
    values: Iterable[int],
    *,
    count: int,
    modality: str,
    allow_empty: bool,
) -> list[int]:
    result = list(values)
    if not result and not allow_empty:
        raise EpisodeRepositoryError(f"{modality} indices cannot be empty")
    for position, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, int):
            raise EpisodeRepositoryError(
                f"{modality} index at position {position} must be an integer"
            )
        if value < 0 or value >= int(count):
            raise EpisodeRepositoryError(
                f"{modality} index {value} is out of bounds for count {count}"
            )
    return [int(value) for value in result]


def _text_items(record: Mapping[str, Any]) -> tuple[TextItem, ...]:
    result = []
    for raw in record["text"]["items"]:
        raw_range = raw.get("frame_range")
        frame_range = (
            None
            if raw_range is None
            else (int(raw_range[0]), int(raw_range[1]))
        )
        result.append(
            TextItem(
                index=int(raw["index"]),
                kind=str(raw["kind"]),
                text=str(raw["text"]),
                frame_range=frame_range,
                metadata=copy.deepcopy(raw.get("metadata") or {}),
            )
        )
    return tuple(result)


class EpisodeSourceReader:
    """Source adapter behind the uniform Episode API."""

    source_dataset: str

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        raise NotImplementedError

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def read_state(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        raise NotImplementedError

    def read_action(
        self, record: Mapping[str, Any], indices: Sequence[int]
    ) -> torch.Tensor:
        raise NotImplementedError

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.read_state(record, state_indices),
            self.read_action(record, action_indices),
        )

    def source_modality_schemas(
        self, record: Mapping[str, Any]
    ) -> Mapping[str, ModalitySchema]:
        """Describe optional source-native frame-aligned fields."""

        return {}

    def read_source_modalities(
        self,
        record: Mapping[str, Any],
        requests: Mapping[str, Sequence[int]],
    ) -> Mapping[str, torch.Tensor]:
        """Read optional source-native fields without changing batch contracts."""

        if requests:
            raise EpisodeRepositoryError(
                f"source={self.source_dataset!r} has no native field reader"
            )
        return {}

    def iter_video_chunks(
        self,
        record: Mapping[str, Any],
        index_chunks: Iterable[Sequence[int]],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> Iterator[tuple[tuple[int, ...], torch.Tensor]]:
        for raw_indices in index_chunks:
            indices = tuple(int(value) for value in raw_indices)
            yield (
                indices,
                self.read_video(
                    record,
                    indices,
                    output_format=output_format,
                    video_size=video_size,
                ),
            )


class AgibotBetaEpisodeReader(EpisodeSourceReader):
    """Decode Agibot Beta tar members into canonical Episode modalities."""

    source_dataset = "agibot_beta"

    def __init__(
        self,
        video_decoder: VideoDecoder | None = None,
        event_callback: ReaderEventCallback | None = None,
        normalization_artifacts: Mapping[str, str | Path] | None = None,
        excluded_episode_ids: Iterable[str] | None = None,
    ):
        self._video_decoder = video_decoder or decode_video_frame_map_ffmpeg
        self._event_callback = event_callback
        self._normalization_paths = {
            str(name).strip().lower(): Path(path)
            for name, path in dict(normalization_artifacts or {}).items()
        }
        self._excluded_episode_ids = frozenset(
            str(value) for value in (excluded_episode_ids or ())
        )
        self._catalog_metadata: Mapping[str, Any] | None = None
        self._normalization_artifact: (
            AgibotBetaNormalizationArtifact | None
        ) = None

    def bind_catalog(self, metadata: Mapping[str, Any]) -> None:
        """Bind profile readers to the immutable Catalog and stats artifact."""

        self._catalog_metadata = copy.deepcopy(dict(metadata))
        catalog_meta = metadata.get("meta") or {}
        profile_name = catalog_meta.get("effector_profile")
        if profile_name is None:
            self._normalization_artifact = None
            return
        profile = get_agibot_beta_effector_profile(str(profile_name))
        if str(metadata.get("source")) != self.source_dataset:
            raise EpisodeRepositoryError("Agibot Beta profile Catalog source mismatch")
        if str(metadata.get("child")) != profile.source_child:
            raise EpisodeRepositoryError("Agibot Beta profile Catalog child mismatch")
        configured_path = self._normalization_paths.get(profile.name)
        if configured_path is None:
            configured_path = self._normalization_paths.get(profile.source_child)
        if configured_path is None:
            raise EpisodeRepositoryError(
                "Agibot Beta profile Catalog requires a normalization artifact: "
                f"profile={profile.name}"
            )
        try:
            self._normalization_artifact = (
                load_agibot_beta_normalization_artifact(
                    configured_path,
                    catalog_metadata=metadata,
                    expected_profile=profile.name,
                )
            )
        except Exception as error:
            raise EpisodeRepositoryError(
                "failed to bind Agibot Beta normalization artifact: "
                f"{error}"
            ) from error

    @staticmethod
    def _profile(
        record: Mapping[str, Any],
    ) -> AgibotBetaEffectorProfile | None:
        profile_name = (record.get("metadata") or {}).get("effector_profile")
        if profile_name is None:
            return None
        return get_agibot_beta_effector_profile(str(profile_name))

    def _emit(self, event: str, **details: Any) -> None:
        if self._event_callback is not None:
            self._event_callback(str(event), dict(details))

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        profile = self._profile(record)
        dimension = 26 if profile is None else profile.state_action_dim
        normalization = (
            "agibot_eef_v1"
            if profile is None
            else str(record["metadata"]["normalization_profile"])
        )
        state_index_semantics = (
            "zero_based_canonical_proprio_timestep"
            if profile is None
            else "zero_based_native_proprio_timestep"
        )
        action_index_semantics = (
            "zero_based_canonical_action_timestep"
            if profile is None
            else "zero_based_native_action_timestep"
        )
        numeric_metadata = {"normalization": normalization}
        if profile is not None:
            numeric_metadata.update(
                {
                    "effector_profile": profile.name,
                    "feature_schema": str(record["metadata"]["feature_schema"]),
                    "feature_names": profile.feature_names,
                    "native_dimension": profile.state_action_dim,
                }
            )
        return {
            "video": ModalitySchema(
                name="video",
                count=int(record["video"]["frame_count"]),
                dtype="uint8|float32",
                item_shape=(None, None, 3),
                index_semantics="zero_based_decoded_video_frame",
                metadata={
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "native_resolution_preserved_when_video_size_is_none": True,
                },
            ),
            "state": ModalitySchema(
                name="state",
                count=int(record["state"]["count"]),
                dtype="float32",
                item_shape=(dimension,),
                index_semantics=state_index_semantics,
                metadata=numeric_metadata,
            ),
            "action": ModalitySchema(
                name="action",
                count=int(record["action"]["count"]),
                dtype="float32",
                item_shape=(dimension,),
                index_semantics=action_index_semantics,
                metadata=numeric_metadata,
            ),
            "text": ModalitySchema(
                name="text",
                count=len(record["text"]["items"]),
                dtype="structured_text_item",
                item_shape=(),
                index_semantics="zero_based_catalog_text_item",
                metadata={},
            ),
        }

    @staticmethod
    def _locator(record: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        if locator["storage_backend"] != "tar_member":
            raise EpisodeRepositoryError(
                f"Agibot Beta {modality} requires tar_member storage"
            )
        return locator

    @contextmanager
    def _temporary_video(self, record: Mapping[str, Any]):
        locator = self._locator(record, "video")
        suffix = Path(str(locator["member"])).suffix or ".mp4"
        self._emit(
            "video_member_extract",
            episode_id=str(record["episode_id"]),
            member=str(locator["member"]),
            size=int(locator["size"]),
        )
        with BoundedFileReader(
            locator["uri"], locator["offset_data"], locator["size"]
        ) as source:
            with tempfile.NamedTemporaryFile(
                prefix="motus_episode_video_", suffix=suffix
            ) as temporary:
                shutil.copyfileobj(source, temporary, length=8 * 1024 * 1024)
                temporary.flush()
                yield temporary.name

    def _decode_video_path(
        self,
        video_path: str,
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        unique_indices = sorted(set(int(index) for index in indices))
        self._emit(
            "video_decode",
            requested_count=len(indices),
            unique_count=len(unique_indices),
        )
        decoded = self._video_decoder(video_path, unique_indices)
        missing = [index for index in unique_indices if index not in decoded]
        if missing:
            raise EpisodeRepositoryError(
                f"video decoder did not return requested frame {missing[0]}"
            )
        arrays = [np.asarray(decoded[index]) for index in indices]
        if any(array.ndim != 3 or array.shape[-1] != 3 for array in arrays):
            raise EpisodeRepositoryError("video decoder must return RGB HWC frames")
        if len({tuple(array.shape) for array in arrays}) != 1:
            raise EpisodeRepositoryError("decoded video frames have inconsistent shapes")
        frames = torch.from_numpy(np.stack(arrays, axis=0).copy()).to(torch.uint8)

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
        with self._temporary_video(record) as video_path:
            return self._decode_video_path(
                video_path,
                indices,
                output_format=output_format,
                video_size=video_size,
            )

    def iter_video_chunks(
        self,
        record: Mapping[str, Any],
        index_chunks: Iterable[Sequence[int]],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> Iterator[tuple[tuple[int, ...], torch.Tensor]]:
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(
                f"unsupported video output_format={output_format!r}"
            )
        with self._temporary_video(record) as video_path:
            for raw_indices in index_chunks:
                indices = tuple(int(value) for value in raw_indices)
                yield (
                    indices,
                    self._decode_video_path(
                        video_path,
                        indices,
                        output_format=output_format,
                        video_size=video_size,
                    ),
                )

    def _canonical_timeline(self, record: Mapping[str, Any]) -> torch.Tensor:
        profile = self._profile(record)
        if profile is not None:
            return self._native_profile_timeline(record, profile)
        locator = self._locator(record, "state")
        self._emit(
            "canonical_timeline_load",
            episode_id=str(record["episode_id"]),
            member=str(locator["member"]),
            size=int(locator["size"]),
        )
        with BoundedFileReader(
            locator["uri"], locator["offset_data"], locator["size"]
        ) as member:
            eef_actions = read_agibot_eef_actions(member)
        timeline = normalize_agibot_eef_actions(eef_actions)
        state_count = int(record["state"]["count"])
        action_count = int(record["action"]["count"])
        if timeline.shape != (state_count, 26) or state_count != action_count:
            raise EpisodeRepositoryError(
                "Agibot Beta canonical timeline does not match Catalog counts: "
                f"timeline={tuple(timeline.shape)} state={state_count} action={action_count}"
            )
        return timeline

    def _native_profile_timeline(
        self,
        record: Mapping[str, Any],
        profile: AgibotBetaEffectorProfile,
    ) -> torch.Tensor:
        episode_id = str(record["episode_id"])
        if episode_id in self._excluded_episode_ids:
            raise EpisodeRepositoryError(
                "Agibot Beta Episode is excluded by numeric quality policy: "
                f"{episode_id}"
            )
        artifact = self._normalization_artifact
        if artifact is None or artifact.profile != profile:
            raise EpisodeRepositoryError(
                "Agibot Beta profile normalization is not bound: "
                f"profile={profile.name}"
            )
        record_meta = record.get("metadata") or {}
        if str(record_meta.get("feature_schema")) != (
            f"agibot_beta_{profile.name}_native_v1"
        ):
            raise EpisodeRepositoryError(
                f"Agibot Beta feature schema mismatch: {episode_id}"
            )
        if str(record_meta.get("normalization_profile")) != (
            artifact.normalization_profile
        ):
            raise EpisodeRepositoryError(
                f"Agibot Beta normalization profile mismatch: {episode_id}"
            )
        locator = self._locator(record, "state")
        self._emit(
            "native_profile_timeline_load",
            episode_id=episode_id,
            profile=profile.name,
            member=str(locator["member"]),
            size=int(locator["size"]),
        )
        with BoundedFileReader(
            locator["uri"], locator["offset_data"], locator["size"]
        ) as member:
            raw, decoded_profile = read_agibot_beta_native_timeline(
                member, expected_profile=profile.name
            )
        if decoded_profile != profile:
            raise EpisodeRepositoryError(
                f"Agibot Beta decoded profile mismatch: {episode_id}"
            )
        try:
            normalized = artifact.normalize(raw)
        except ValueError as error:
            raise EpisodeRepositoryError(str(error)) from error
        state_count = int(record["state"]["count"])
        action_count = int(record["action"]["count"])
        if (
            normalized.shape != (state_count, profile.state_action_dim)
            or state_count != action_count
        ):
            raise EpisodeRepositoryError(
                "Agibot Beta native timeline does not match Catalog counts: "
                f"timeline={tuple(normalized.shape)} state={state_count} "
                f"action={action_count}"
            )
        return torch.from_numpy(normalized.copy())

    @staticmethod
    def _take(timeline: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        if not indices:
            return torch.empty((0, int(timeline.shape[1])), dtype=torch.float32)
        return timeline[torch.tensor(indices, dtype=torch.long)].clone()

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


class _BaseModalityView:
    def __init__(self, episode: "EpisodeView", name: str):
        self._episode = episode
        self.name = name

    @property
    def schema(self) -> ModalitySchema:
        return self._episode.info.modality_schemas[self.name]

    @property
    def count(self) -> int:
        return self.schema.count


class VideoModalityView(_BaseModalityView):
    def read(
        self,
        indices: Iterable[int],
        *,
        output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        return self._episode.read_video(
            indices, output_format=output_format, video_size=video_size
        )

    def iter_chunks(
        self,
        *,
        chunk_size: int = 64,
        start: int = 0,
        stop: int | None = None,
        output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> Iterator[VideoChunk]:
        return self._episode.iter_video_chunks(
            chunk_size=chunk_size,
            start=start,
            stop=stop,
            output_format=output_format,
            video_size=video_size,
        )


class NumericModalityView(_BaseModalityView):
    def read(self, indices: Iterable[int]) -> torch.Tensor:
        if self.name == "state":
            return self._episode.read_state(indices)
        return self._episode.read_action(indices)


class TextModalityView(_BaseModalityView):
    @property
    def items(self) -> tuple[TextItem, ...]:
        return self._episode.read_text(range(self.count))

    def read(self, indices: Iterable[int]) -> tuple[TextItem, ...]:
        return self._episode.read_text(indices)


class EpisodeView:
    """Complete logical Episode with lazy, exact modality access."""

    def __init__(
        self,
        repository: "EpisodeRepository",
        record: Mapping[str, Any],
        reader: EpisodeSourceReader,
        *,
        catalog_record: Mapping[str, Any] | None = None,
    ):
        self._repository = repository
        self._record = record
        self._catalog_record = catalog_record or record
        self._reader = reader
        schemas = dict(reader.schemas(record))
        self.info = EpisodeInfo(
            episode_id=str(record["episode_id"]),
            catalog_id=str(repository.catalog_id),
            source_dataset=str(record["source_dataset"]),
            source_child=str(record["source_child"]),
            robot_name=str(record["robot_name"]),
            split=(
                str(record["split"]) if record.get("split") is not None else None
            ),
            video_count=int(record["video"]["frame_count"]),
            state_count=int(record["state"]["count"]),
            action_count=int(record["action"]["count"]),
            text_count=len(record["text"]["items"]),
            modality_schemas=schemas,
            metadata=copy.deepcopy(record["metadata"]),
        )
        self.video = VideoModalityView(self, "video")
        self.state = NumericModalityView(self, "state")
        self.action = NumericModalityView(self, "action")
        self.text = TextModalityView(self, "text")

    @property
    def episode_id(self) -> str:
        return self.info.episode_id

    @property
    def metadata(self) -> Mapping[str, Any]:
        return copy.deepcopy(self.info.metadata)

    def read_video(
        self,
        indices: Iterable[int],
        *,
        output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        selected = _indices(
            indices,
            count=self.info.video_count,
            modality="video",
            allow_empty=False,
        )
        return self._reader.read_video(
            self._record,
            selected,
            output_format=output_format,
            video_size=video_size,
        )

    def read_state(self, indices: Iterable[int]) -> torch.Tensor:
        selected = _indices(
            indices,
            count=self.info.state_count,
            modality="state",
            allow_empty=True,
        )
        return self._reader.read_state(self._record, selected)

    def read_action(self, indices: Iterable[int]) -> torch.Tensor:
        selected = _indices(
            indices,
            count=self.info.action_count,
            modality="action",
            allow_empty=True,
        )
        return self._reader.read_action(self._record, selected)

    def read_text(self, indices: Iterable[int]) -> tuple[TextItem, ...]:
        items = _text_items(self._record)
        selected = _indices(
            indices,
            count=len(items),
            modality="text",
            allow_empty=True,
        )
        return tuple(items[index] for index in selected)

    def source_modality_schemas(self) -> Mapping[str, ModalitySchema]:
        """Return source-native fields available outside standard batches."""

        return copy.deepcopy(self._reader.source_modality_schemas(self._record))

    def read_source_modalities(
        self,
        requests: Mapping[str, Iterable[int]],
    ) -> Mapping[str, torch.Tensor]:
        """Read explicitly indexed source-native, frame-aligned fields."""

        if not isinstance(requests, Mapping):
            raise EpisodeRepositoryError("source modality requests must be a mapping")
        schemas = self._reader.source_modality_schemas(self._record)
        unknown = sorted(set(requests).difference(schemas))
        if unknown:
            raise EpisodeRepositoryError(
                f"unknown source modalities={unknown}; available={sorted(schemas)}"
            )
        selected = {
            str(name): tuple(
                _indices(
                    values,
                    count=schemas[name].count,
                    modality=f"source modality {name}",
                    allow_empty=True,
                )
            )
            for name, values in requests.items()
        }
        return self._reader.read_source_modalities(self._record, selected)

    def iter_video_chunks(
        self,
        *,
        chunk_size: int = 64,
        start: int = 0,
        stop: int | None = None,
        output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> Iterator[VideoChunk]:
        if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
            raise EpisodeRepositoryError("chunk_size must be a positive integer")
        start = int(start)
        stop = self.info.video_count if stop is None else int(stop)
        if start < 0 or stop < start or stop > self.info.video_count:
            raise EpisodeRepositoryError(
                f"invalid video chunk range [{start}, {stop}) for count "
                f"{self.info.video_count}"
            )
        index_chunks = (
            tuple(range(chunk_start, min(stop, chunk_start + int(chunk_size))))
            for chunk_start in range(start, stop, int(chunk_size))
        )
        for chunk_indices, frames in self._reader.iter_video_chunks(
            self._record,
            index_chunks,
            output_format=output_format,
            video_size=video_size,
        ):
            yield VideoChunk(
                indices=chunk_indices,
                frames=frames,
            )

    def materialize_video(
        self,
        *,
        max_bytes: int,
        output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if isinstance(max_bytes, bool) or int(max_bytes) <= 0:
            raise EpisodeRepositoryError("max_bytes must be a positive integer")
        first = self.read_video(
            [0], output_format=output_format, video_size=video_size
        )
        estimated = int(first[0].numel() * first.element_size() * self.info.video_count)
        if estimated > int(max_bytes):
            raise EpisodeMemoryLimitError(
                f"complete video requires approximately {estimated} bytes, "
                f"exceeding max_bytes={int(max_bytes)}"
            )
        if self.info.video_count == 1:
            return first
        return self.read_video(
            range(self.info.video_count),
            output_format=output_format,
            video_size=video_size,
        )

    def materialize(
        self,
        *,
        modalities: Sequence[str] = ("state", "action", "text"),
        max_video_bytes: int | None = None,
        video_output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> EpisodeData:
        requested = tuple(str(value) for value in modalities)
        unknown = sorted(set(requested) - {"video", "state", "action", "text"})
        if unknown:
            raise EpisodeRepositoryError(f"unknown modalities: {unknown}")
        if len(set(requested)) != len(requested):
            raise EpisodeRepositoryError("modalities cannot contain duplicates")
        if "video" in requested and max_video_bytes is None:
            raise EpisodeRepositoryError(
                "max_video_bytes is required when materializing complete video"
            )

        state_indices = (
            list(range(self.info.state_count)) if "state" in requested else []
        )
        action_indices = (
            list(range(self.info.action_count)) if "action" in requested else []
        )
        state = action = None
        if "state" in requested and "action" in requested:
            state, action = self._reader.read_state_action(
                self._record, state_indices, action_indices
            )
        elif "state" in requested:
            state = self._reader.read_state(self._record, state_indices)
        elif "action" in requested:
            action = self._reader.read_action(self._record, action_indices)

        return EpisodeData(
            info=self.info,
            video=(
                self.materialize_video(
                    max_bytes=int(max_video_bytes),
                    output_format=video_output_format,
                    video_size=video_size,
                )
                if "video" in requested
                else None
            ),
            state=state,
            action=action,
            text=(
                self.read_text(range(self.info.text_count))
                if "text" in requested
                else None
            ),
        )

    def read_selection(
        self,
        selection: Mapping[str, Any],
        *,
        video_output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
        preflight_validator: SelectionPreflightValidator | None = None,
    ) -> EpisodeSelectionData:
        return self.read_selections(
            [selection],
            video_output_format=video_output_format,
            video_size=video_size,
            preflight_validator=preflight_validator,
        )[0]

    def _validated_selection_indices(
        self, selection: Mapping[str, Any]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        validate_sample_selection_v1(selection)
        if selection["episode_id"] != self.episode_id:
            raise EpisodeRepositoryError(
                "sample selection episode_id does not match EpisodeView"
            )
        video_indices = tuple(
            _indices(
                selection["video_frame_indices"],
                count=self.info.video_count,
                modality="video",
                allow_empty=False,
            )
        )
        state_indices = tuple(
            _indices(
                selection["state_indices"],
                count=self.info.state_count,
                modality="state",
                allow_empty=True,
            )
        )
        action_indices = tuple(
            _indices(
                selection["action_indices"],
                count=self.info.action_count,
                modality="action",
                allow_empty=True,
            )
        )
        text_indices = tuple(
            _indices(
                selection["text_indices"],
                count=self.info.text_count,
                modality="text",
                allow_empty=True,
            )
        )
        return video_indices, state_indices, action_indices, text_indices

    @staticmethod
    def _take_union(
        values: torch.Tensor,
        union_indices: Sequence[int],
        requested_indices: Sequence[int],
    ) -> torch.Tensor:
        if not requested_indices:
            return values.new_empty((0, *values.shape[1:]))
        positions = {int(index): position for position, index in enumerate(union_indices)}
        return values[
            torch.tensor(
                [positions[int(index)] for index in requested_indices],
                dtype=torch.long,
            )
        ].clone()

    def read_selections(
        self,
        selections: Sequence[Mapping[str, Any]],
        *,
        video_output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
        preflight_validator: SelectionPreflightValidator | None = None,
    ) -> tuple[EpisodeSelectionData, ...]:
        if not selections:
            raise EpisodeRepositoryError("read_selections requires at least one selection")
        validated = [
            self._validated_selection_indices(selection)
            for selection in selections
        ]
        video_union = sorted(
            {index for indices, _, _, _ in validated for index in indices}
        )
        state_union = sorted(
            {index for _, indices, _, _ in validated for index in indices}
        )
        action_union = sorted(
            {index for _, _, indices, _ in validated for index in indices}
        )

        if state_union or action_union:
            union_state, union_action = self._reader.read_state_action(
                self._record, state_union, action_union
            )
        else:
            state_width = int(self.state.schema.item_shape[0])
            action_width = int(self.action.schema.item_shape[0])
            union_state = torch.empty((0, state_width), dtype=torch.float32)
            union_action = torch.empty((0, action_width), dtype=torch.float32)
        all_text = _text_items(self._record)

        prepared = []
        for selection, (
            video_indices,
            state_indices,
            action_indices,
            text_indices,
        ) in zip(selections, validated):
            state = self._take_union(
                union_state, state_union, state_indices
            )
            action = self._take_union(
                union_action, action_union, action_indices
            )
            text = tuple(all_text[index] for index in text_indices)
            if preflight_validator is not None:
                preflight_validator(
                    self.info,
                    selection,
                    state,
                    action,
                    text,
                )
            prepared.append(
                (
                    video_indices,
                    state_indices,
                    action_indices,
                    text_indices,
                    state,
                    action,
                    text,
                )
            )

        # Video extraction and decode are intentionally last. Numeric safety
        # validators can reject a selection without paying the MP4 I/O cost.
        union_video = self.read_video(
            video_union,
            output_format=video_output_format,
            video_size=video_size,
        )

        result = []
        for (
            video_indices,
            state_indices,
            action_indices,
            text_indices,
            state,
            action,
            text,
        ) in prepared:
            result.append(
                EpisodeSelectionData(
                    info=self.info,
                    video_frame_indices=video_indices,
                    state_indices=state_indices,
                    action_indices=action_indices,
                    text_indices=text_indices,
                    video=self._take_union(
                        union_video, video_union, video_indices
                    ),
                    state=state,
                    action=action,
                    text=text,
                )
            )
        return tuple(result)


_BUILTIN_SOURCE_READERS_LOCK = threading.Lock()


def _ensure_builtin_source_readers() -> None:
    """Register the built-in source readers into the global registry once.

    Third-party readers registered via ``register_episode_source_reader`` live
    in the same registry and follow the exact same shape: one reader class (or
    instance) bound to the Catalog ``source`` name.
    """

    with _BUILTIN_SOURCE_READERS_LOCK:
        if "agibot_beta" not in EPISODE_SOURCE_READERS.names():
            EPISODE_SOURCE_READERS.register("agibot_beta", AgibotBetaEpisodeReader)
        # Importing motus_data.source_readers self-registers the agibot,
        # robotwin, and robotmind readers at module bottom.
        import motus_data.source_readers  # noqa: F401
        # EgoVerse is isolated so its optional Zarr dependency is imported
        # only when a real EgoVerse Episode is opened.
        import motus_data.egoverse  # noqa: F401
        import motus_data.egoscale  # noqa: F401
        import motus_data.libero  # noqa: F401
        import motus_data.openvid  # noqa: F401


class EpisodeRepository:
    """Validated versioned Catalog repository with a stable public API."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        readers: Mapping[str, EpisodeSourceReader] | None = None,
        reader_options: Mapping[str, Mapping[str, Any]] | None = None,
        verify_sha256: bool = True,
        index_path: str | Path | None = None,
        record_cache_size: int = 256,
    ):
        self.catalog_path = Path(catalog_path)
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else _metadata_path(self.catalog_path)
        )
        if not self.catalog_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Catalog requires {self.catalog_path} and {self.metadata_path}"
        )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        validate_catalog_metadata(self.metadata)
        self.protocol_version = str(
            self.metadata.get("schema_version", self.metadata.get("format"))
        )
        self.catalog_id = str(self.metadata["catalog_id"])
        # Resolution order: registered readers (built-in + third-party
        # plugins) provide the defaults; a per-call readers= mapping wins.
        _ensure_builtin_source_readers()
        self._readers = EPISODE_SOURCE_READERS.snapshot(
            None
            if reader_options is None
            else {
                str(source).strip().lower(): dict(options)
                for source, options in reader_options.items()
            }
        )
        self._readers.update(readers or {})
        if isinstance(record_cache_size, bool) or int(record_cache_size) < 0:
            raise ValueError("record_cache_size must be an integer >= 0")
        self.record_cache_size = int(record_cache_size)
        self.index_path = None if index_path is None else Path(index_path)
        self._offset_index: CatalogOffsetIndex | None = None
        self._record_cache: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
        self._catalog_fd: int | None = None
        self._catalog_fd_pid: int | None = None
        self._catalog_fd_lock = threading.RLock()

        expected_count = self.metadata.get(
            "record_count", self.metadata.get("episodes")
        )
        expected_sha = self.metadata.get("sha256")
        if expected_sha is None:
            expected_sha = (
                (self.metadata.get("statistics") or {}).get("catalog_sha256")
            )
        if self.index_path is None:
            digest = hashlib.sha256()
            records: dict[str, Mapping[str, Any]] = {}
            with self.catalog_path.open("rb") as handle:
                for line_number, payload in enumerate(handle, start=1):
                    digest.update(payload)
                    if not payload.strip():
                        continue
                    try:
                        record = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise EpisodeRepositoryError(
                            f"invalid Catalog JSON at line {line_number}: {error}"
                        ) from error
                    validate_episode_catalog_binding(record, self.metadata)
                    episode_id = str(record["episode_id"])
                    if episode_id in records:
                        raise EpisodeRepositoryError(
                            f"duplicate episode_id={episode_id!r} at line {line_number}"
                        )
                    records[episode_id] = record
            if expected_count is not None and len(records) != int(expected_count):
                raise EpisodeRepositoryError(
                    f"Catalog record count mismatch: rows={len(records)} "
                    f"metadata={int(expected_count)}"
                )
            actual_sha = digest.hexdigest()
            if (
                verify_sha256
                and expected_sha is not None
                and actual_sha != expected_sha
            ):
                raise EpisodeRepositoryError(
                    f"Catalog SHA256 mismatch: actual={actual_sha} "
                    f"expected={expected_sha}"
                )
            self.catalog_sha256 = actual_sha
            self._records: dict[str, Mapping[str, Any]] | None = records
        else:
            self._offset_index = CatalogOffsetIndex(
                self.index_path,
                catalog_path=self.catalog_path,
                catalog_metadata=self.metadata,
            )
            if (
                expected_count is not None
                and len(self._offset_index) != int(expected_count)
            ):
                raise EpisodeRepositoryError(
                    f"Catalog offset index rows={len(self._offset_index)} "
                    f"metadata={int(expected_count)}"
                )
            if verify_sha256:
                digest = hashlib.sha256()
                with self.catalog_path.open("rb") as handle:
                    for payload in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(payload)
                actual_sha = digest.hexdigest()
                if expected_sha is not None and actual_sha != expected_sha:
                    raise EpisodeRepositoryError(
                        f"Catalog SHA256 mismatch: actual={actual_sha} "
                        f"expected={expected_sha}"
                    )
                self.catalog_sha256 = actual_sha
            else:
                if expected_sha is None:
                    raise EpisodeRepositoryError(
                        "indexed Repository requires Catalog metadata SHA256"
                    )
                self.catalog_sha256 = str(expected_sha)
            self._records = None

        source = str(
            self.metadata.get("source_dataset", self.metadata.get("source"))
        )
        if source not in self._readers:
            raise EpisodeRepositoryError(
                f"no Episode source reader registered for {source!r}; "
                f"registered={sorted(self._readers)}"
            )
        bind_catalog = getattr(self._readers[source], "bind_catalog", None)
        if callable(bind_catalog):
            bind_catalog(self.metadata)
    def __len__(self) -> int:
        if self._records is not None:
            return len(self._records)
        if self._offset_index is None:
            raise EpisodeRepositoryError("Repository storage is not initialized")
        return len(self._offset_index)

    @property
    def indexed(self) -> bool:
        return self._offset_index is not None

    def _catalog_file_descriptor(self) -> int:
        pid = os.getpid()
        with self._catalog_fd_lock:
            if self._catalog_fd is None or self._catalog_fd_pid != pid:
                if self._catalog_fd is not None:
                    os.close(self._catalog_fd)
                self._catalog_fd = os.open(self.catalog_path, os.O_RDONLY)
                self._catalog_fd_pid = pid
            return self._catalog_fd

    def _indexed_record(self, episode_id: str) -> Mapping[str, Any]:
        key = str(episode_id)
        with self._catalog_fd_lock:
            cached = self._record_cache.get(key)
            if cached is not None:
                self._record_cache.move_to_end(key)
                return cached
            if self._offset_index is None:
                raise EpisodeRepositoryError("Repository has no Catalog offset index")
            try:
                byte_offset, byte_length = self._offset_index.locate(key)
            except KeyError as error:
                raise KeyError(key) from error
            payload = os.pread(
                self._catalog_file_descriptor(),
                byte_length,
                byte_offset,
            )
            if len(payload) != byte_length:
                raise EpisodeRepositoryError(
                    f"short Catalog record read for episode_id={key!r}: "
                    f"got={len(payload)} expected={byte_length}"
                )
            try:
                record = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EpisodeRepositoryError(
                    f"invalid indexed Catalog JSON for episode_id={key!r}: {error}"
                ) from error
            validate_episode_catalog_binding(record, self.metadata)
            if str(record["episode_id"]) != key:
                raise EpisodeRepositoryError(
                    f"Catalog offset index points episode_id={key!r} to "
                    f"record={record['episode_id']!r}"
                )
            if self.record_cache_size > 0:
                self._record_cache[key] = record
                self._record_cache.move_to_end(key)
                while len(self._record_cache) > self.record_cache_size:
                    self._record_cache.popitem(last=False)
            return record

    def _record(self, episode_id: str) -> Mapping[str, Any]:
        key = str(episode_id)
        if self._records is not None:
            try:
                return self._records[key]
            except KeyError as error:
                raise KeyError(key) from error
        return self._indexed_record(key)

    def iter_episode_ids(self) -> Iterator[str]:
        if self._records is not None:
            yield from self._records
            return
        if self._offset_index is None:
            raise EpisodeRepositoryError("Repository storage is not initialized")
        yield from self._offset_index.iter_episode_ids()

    def _reader(self, record: Mapping[str, Any]) -> EpisodeSourceReader:
        source = str(
            record.get(
                "source_dataset",
                self.metadata.get("source_dataset", self.metadata.get("source")),
            )
        )
        reader = self._readers.get(source)
        if reader is None:
            raise EpisodeRepositoryError(
                f"no Episode source reader registered for {source!r}; "
                f"registered={sorted(self._readers)}"
            )
        return reader

    def get_episode(self, episode_id: str) -> EpisodeView:
        try:
            record = self._record(str(episode_id))
        except KeyError as error:
            raise KeyError(
                f"episode_id={episode_id!r} is not present in Catalog "
                f"{self.catalog_id!r}"
            ) from error
        normalized = _normalize_catalog_record(record, self.metadata)
        return EpisodeView(
            self,
            normalized,
            self._reader(normalized),
            catalog_record=record,
        )

    def iter_episode_infos(
        self,
        *,
        split: str | None = None,
        source_child: str | None = None,
        robot_name: str | None = None,
    ) -> Iterator[EpisodeInfo]:
        if split is not None and str(split) not in {"train", "test"}:
            raise EpisodeRepositoryError("split must be 'train', 'test', or None")
        if (
            split is not None
            and self.protocol_version != MOTUS_EPISODE_CATALOG_METADATA_V1
        ):
            raise EpisodeRepositoryError(
                "split filtering is unavailable for a split-neutral Catalog"
            )
        for episode_id in self.iter_episode_ids():
            record = self._record(episode_id)
            if split is not None and record.get("split") != str(split):
                continue
            normalized = _normalize_catalog_record(record, self.metadata)
            if (
                source_child is not None
                and normalized["source_child"] != str(source_child)
            ):
                continue
            if (
                robot_name is not None
                and normalized["robot_name"] != str(robot_name)
            ):
                continue
            yield EpisodeView(
                self,
                normalized,
                self._reader(normalized),
                catalog_record=record,
            ).info

    def close(self) -> None:
        with self._catalog_fd_lock:
            if self._catalog_fd is not None:
                os.close(self._catalog_fd)
                self._catalog_fd = None
                self._catalog_fd_pid = None
            if self._offset_index is not None:
                self._offset_index.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_catalog_fd"] = None
        state["_catalog_fd_pid"] = None
        state["_catalog_fd_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._catalog_fd_lock = threading.RLock()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def load_full_episode(
        self,
        episode_id: str,
        *,
        modalities: Sequence[str] = ("state", "action", "text"),
        max_video_bytes: int | None = None,
        video_output_format: str = "rgb_uint8_hwc",
        video_size: tuple[int, int] | None = None,
    ) -> EpisodeData:
        return self.get_episode(episode_id).materialize(
            modalities=modalities,
            max_video_bytes=max_video_bytes,
            video_output_format=video_output_format,
            video_size=video_size,
        )
