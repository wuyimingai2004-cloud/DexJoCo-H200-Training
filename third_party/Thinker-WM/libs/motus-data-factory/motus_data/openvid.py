"""OpenVid ZIP-backed complete-video Episode reader."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from motus_data.reader_registry import register_episode_source_reader
from motus_data.repository import (
    AgibotBetaEpisodeReader,
    EpisodeRepositoryError,
    ModalitySchema,
)
from motus_data.storage import ConcatenatedFileReader


@contextmanager
def open_openvid_zip(
    path: str | Path,
    *,
    parts: Sequence[str | Path] = (),
) -> Iterator[zipfile.ZipFile]:
    """Open one normal or physically split ZIP without reconstructing it."""

    if parts:
        raw = ConcatenatedFileReader(parts)
        buffered = io.BufferedReader(raw)
        try:
            with zipfile.ZipFile(buffered, "r") as archive:
                yield archive
        finally:
            buffered.close()
    else:
        with zipfile.ZipFile(str(path), "r") as archive:
            yield archive


class OpenVidEpisodeReader(AgibotBetaEpisodeReader):
    """Decode model-selected frames from an MP4 member in an OpenVid ZIP."""

    source_dataset = "openvid"

    @staticmethod
    def _locator(
        record: Mapping[str, Any],
        modality: str = "video",
    ) -> Mapping[str, Any]:
        locator = record[modality]["locator"]
        if locator["storage_backend"] not in {"zip", "zip_split"}:
            raise EpisodeRepositoryError(
                f"OpenVid {modality} requires zip or zip_split storage"
            )
        if not locator.get("member"):
            raise EpisodeRepositoryError("OpenVid ZIP member is missing")
        return locator

    def schemas(self, record: Mapping[str, Any]) -> Mapping[str, ModalitySchema]:
        return {
            "video": ModalitySchema(
                "video",
                int(record["video"]["frame_count"]),
                "uint8|float32",
                (None, None, 3),
                "zero_based_decoded_video_frame",
                {
                    "formats": ["rgb_uint8_hwc", "rgb_float32_chw"],
                    "container": "mp4_in_zip",
                },
            ),
            "state": ModalitySchema(
                "state",
                0,
                "float32",
                (0,),
                "absent",
                {"available": False},
            ),
            "action": ModalitySchema(
                "action",
                0,
                "float32",
                (0,),
                "absent",
                {"available": False},
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

    @contextmanager
    def _temporary_video(
        self,
        record: Mapping[str, Any],
    ) -> Iterator[str]:
        locator = self._locator(record)
        member = str(locator["member"])
        parts = tuple(locator.get("parts") or ())
        suffix = Path(member).suffix or ".mp4"
        with open_openvid_zip(locator["uri"], parts=parts) as archive:
            try:
                info = archive.getinfo(member)
            except KeyError as error:
                raise EpisodeRepositoryError(
                    f"OpenVid ZIP member is missing: {member}"
                ) from error
            metadata = record["video"].get("metadata") or {}
            expected_crc = metadata.get("crc32")
            if expected_crc is not None and int(expected_crc) != int(info.CRC):
                raise EpisodeRepositoryError(
                    f"OpenVid member CRC changed: {member}"
                )
            expected_size = metadata.get("member_size")
            if (
                expected_size is not None
                and int(expected_size) != int(info.file_size)
            ):
                raise EpisodeRepositoryError(
                    f"OpenVid member size changed: {member}"
                )
            with archive.open(info, "r") as source:
                with tempfile.NamedTemporaryFile(
                    prefix="motus_openvid_",
                    suffix=suffix,
                ) as temporary:
                    shutil.copyfileobj(
                        source,
                        temporary,
                        length=8 * 1024 * 1024,
                    )
                    temporary.flush()
                    yield temporary.name

    @staticmethod
    def _empty_numeric(indices: Sequence[int]) -> torch.Tensor:
        if indices:
            raise EpisodeRepositoryError(
                "OpenVid has no state/action indices"
            )
        return torch.empty((0, 0), dtype=torch.float32)

    def read_state(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
    ) -> torch.Tensor:
        return self._empty_numeric(indices)

    def read_action(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
    ) -> torch.Tensor:
        return self._empty_numeric(indices)

    def read_state_action(
        self,
        record: Mapping[str, Any],
        state_indices: Sequence[int],
        action_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._empty_numeric(state_indices),
            self._empty_numeric(action_indices),
        )


register_episode_source_reader("openvid", OpenVidEpisodeReader)
