"""Minimal ISO-BMFF metadata reader for exact video sample counts.

The reader walks box headers through a bounded tar-member view. It skips media
payloads and reads the video track's `stsz`/`stz2` sample count without decoding
pixels or copying the complete MP4 member.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from motus_data.storage import BoundedFileReader


class IsoBmffError(ValueError):
    pass


@dataclass(frozen=True)
class Box:
    kind: bytes
    start: int
    end: int
    payload_start: int


def _read_exact(handle: BinaryIO, offset: int, size: int) -> bytes:
    handle.seek(offset)
    data = handle.read(size)
    if len(data) != size:
        raise IsoBmffError(
            f"short ISO-BMFF read at offset={offset}: got={len(data)} expected={size}"
        )
    return data


def _boxes(handle: BinaryIO, start: int, end: int) -> Iterator[Box]:
    cursor = int(start)
    while cursor < end:
        if end - cursor < 8:
            raise IsoBmffError(f"trailing {end - cursor} bytes cannot form a box header")
        header = _read_exact(handle, cursor, 8)
        box_size = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        header_size = 8
        if box_size == 1:
            box_size = int.from_bytes(_read_exact(handle, cursor + 8, 8), "big")
            header_size = 16
        elif box_size == 0:
            box_size = end - cursor
        if kind == b"uuid":
            header_size += 16
        if box_size < header_size:
            raise IsoBmffError(
                f"invalid box size={box_size} header={header_size} kind={kind!r}"
            )
        box_end = cursor + box_size
        if box_end > end:
            raise IsoBmffError(
                f"box {kind!r} exceeds container: end={box_end} container_end={end}"
            )
        yield Box(kind=kind, start=cursor, end=box_end, payload_start=cursor + header_size)
        cursor = box_end


def _children(handle: BinaryIO, parent: Box, kind: bytes) -> list[Box]:
    return [box for box in _boxes(handle, parent.payload_start, parent.end) if box.kind == kind]


def _one_child(handle: BinaryIO, parent: Box, kind: bytes) -> Box | None:
    matches = _children(handle, parent, kind)
    if len(matches) > 1:
        raise IsoBmffError(f"multiple {kind!r} children in {parent.kind!r}")
    return matches[0] if matches else None


def _handler_type(handle: BinaryIO, mdia: Box) -> bytes | None:
    hdlr = _one_child(handle, mdia, b"hdlr")
    if hdlr is None or hdlr.end - hdlr.payload_start < 12:
        return None
    return _read_exact(handle, hdlr.payload_start + 8, 4)


def _sample_count(handle: BinaryIO, mdia: Box) -> int:
    minf = _one_child(handle, mdia, b"minf")
    if minf is None:
        raise IsoBmffError("video track has no minf box")
    stbl = _one_child(handle, minf, b"stbl")
    if stbl is None:
        raise IsoBmffError("video track has no stbl box")
    tables = _children(handle, stbl, b"stsz") + _children(handle, stbl, b"stz2")
    if len(tables) != 1:
        raise IsoBmffError(
            f"video sample table requires exactly one stsz/stz2 box, got {len(tables)}"
        )
    table = tables[0]
    if table.end - table.payload_start < 12:
        raise IsoBmffError("video sample-size box is too short")
    count = int.from_bytes(_read_exact(handle, table.payload_start + 8, 4), "big")
    if count <= 0:
        raise IsoBmffError(f"invalid video sample count={count}")
    return count


def read_video_sample_count(handle: BinaryIO, size: int) -> int:
    top_level = list(_boxes(handle, 0, int(size)))
    moov_boxes = [box for box in top_level if box.kind == b"moov"]
    if len(moov_boxes) != 1:
        raise IsoBmffError(f"expected one moov box, got {len(moov_boxes)}")
    counts = []
    for trak in _children(handle, moov_boxes[0], b"trak"):
        mdia = _one_child(handle, trak, b"mdia")
        if mdia is not None and _handler_type(handle, mdia) == b"vide":
            counts.append(_sample_count(handle, mdia))
    if len(counts) != 1:
        raise IsoBmffError(f"expected one video track, got counts={counts}")
    return counts[0]


def read_tar_member_video_sample_count(
    path: str | Path, offset_data: int, size: int
) -> int:
    with BoundedFileReader(path, offset_data, size) as member:
        return read_video_sample_count(member, member.size)
