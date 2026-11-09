"""Storage primitives for the Motus monitor.

Two building blocks:

* :class:`ChunkedJsonlWriter` -- append-only JSONL files rotated by size.
  Append-only files are what makes byte-level incremental sync over SSH cheap:
  the reader side remembers ``(chunk_index, byte_pos)`` and only ever fetches
  the tail that was appended since the last poll.
* :func:`atomic_write_json` -- tmp-file + ``os.replace`` for the small JSON
  documents (run.json / status.json / annotations) that are rewritten whole.

Everything here is stdlib-only and safe to call from training processes.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_CHUNK_BYTES = 256 * 1024


def _json_default(value: Any) -> str:
    return str(value)


def atomic_write_json(path: object, payload: Any) -> None:
    """Write JSON atomically (tmp file in the same directory + rename)."""
    target = Path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=_json_default)
        fp.write("\n")
    os.replace(tmp, target)


def read_json(path: object, default: Any = None) -> Any:
    try:
        with Path(str(path)).open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return default


class ChunkedJsonlWriter:
    """Append-only JSONL writer that rotates files once they grow past a size.

    Files are named ``000000.jsonl``, ``000001.jsonl``, ... inside ``directory``.
    Each ``write()`` is flushed immediately (line buffered) so a concurrent
    reader/syncer sees records almost in real time. Thread-safe.
    """

    def __init__(self, directory: object, max_bytes: int = DEFAULT_CHUNK_BYTES):
        self.directory = Path(str(directory))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._index, self._fp = self._open_current()

    # -- internals ---------------------------------------------------------
    def _chunk_path(self, index: int) -> Path:
        return self.directory / f"{index:06d}.jsonl"

    def _existing_indexes(self) -> List[int]:
        indexes = []
        for path in self.directory.glob("*.jsonl"):
            try:
                indexes.append(int(path.stem))
            except ValueError:
                continue
        return sorted(indexes)

    def _open_current(self) -> Tuple[int, Any]:
        indexes = self._existing_indexes()
        index = indexes[-1] if indexes else 0
        if indexes and self._chunk_path(index).stat().st_size >= self.max_bytes:
            index += 1
        fp = self._chunk_path(index).open("a", encoding="utf-8", buffering=1)
        return index, fp

    def _rotate_locked(self) -> None:
        self._fp.close()
        self._index += 1
        self._fp = self._chunk_path(self._index).open("a", encoding="utf-8", buffering=1)

    # -- public API --------------------------------------------------------
    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        with self._lock:
            if self._fp is None:
                return
            self._fp.write(line)
            try:
                if self._fp.tell() >= self.max_bytes:
                    self._rotate_locked()
            except (OSError, ValueError):
                pass

    def position(self) -> str:
        """Current write position in reader offset format (``index:pos``)."""
        with self._lock:
            try:
                return f"{self._index}:{self._fp.tell()}"
            except (OSError, ValueError):
                return f"{self._index}:0"

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                finally:
                    self._fp = None

    @property
    def closed(self) -> bool:
        return self._fp is None


def _parse_offset(offset: Optional[str]) -> Tuple[int, int]:
    if not offset:
        return 0, 0
    try:
        index_str, pos_str = str(offset).split(":", 1)
        return max(0, int(index_str)), max(0, int(pos_str))
    except (ValueError, AttributeError):
        return 0, 0


class ChunkedJsonlReader:
    """Byte-incremental reader over a directory of rotated JSONL chunks.

    Offsets are opaque strings ``"<chunk_index>:<byte_pos>"``. A caller keeps
    the last offset it saw and passes it to :meth:`read_incremental` to get
    exactly the bytes appended since then.
    """

    def __init__(self, directory: object):
        self.directory = Path(str(directory))

    def _chunk_path(self, index: int) -> Path:
        return self.directory / f"{index:06d}.jsonl"

    def _indexes(self) -> List[int]:
        if not self.directory.is_dir():
            return []
        indexes = []
        for path in self.directory.glob("*.jsonl"):
            try:
                indexes.append(int(path.stem))
            except ValueError:
                continue
        return sorted(indexes)

    def end_offset(self) -> str:
        indexes = self._indexes()
        if not indexes:
            return "0:0"
        last = indexes[-1]
        try:
            return f"{last}:{self._chunk_path(last).stat().st_size}"
        except OSError:
            return f"{last}:0"

    def read_incremental(self, offset: Optional[str]) -> Dict[str, Any]:
        """Return ``{"offset": <new>, "data": <text>}`` with bytes after ``offset``.

        If the requested position points past the current end (e.g. the files
        were rotated away or truncated) the reader clamps to the current end
        and returns empty data instead of failing.
        """
        index, pos = _parse_offset(offset)
        indexes = self._indexes()
        if not indexes:
            return {"offset": offset or "0:0", "data": ""}

        chunks: List[bytes] = []
        current_index, current_pos = index, pos
        for idx in indexes:
            if idx < index:
                continue
            path = self._chunk_path(idx)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            start = pos if idx == index else 0
            if start > size:
                start = 0  # truncated/rotated underneath us: resend chunk
            if start < size:
                try:
                    with path.open("rb") as fp:
                        fp.seek(start)
                        chunks.append(fp.read())
                except OSError:
                    continue
            current_index, current_pos = idx, size

        data = b"".join(chunks)
        return {
            "offset": f"{current_index}:{current_pos}",
            "data": data.decode("utf-8", errors="replace"),
        }

    def read_tail(self, max_bytes: int = 64 * 1024) -> Dict[str, Any]:
        """Return the last ``max_bytes`` (whole lines only) plus the offset of
        the position right after them -- the entry point for a new client that
        wants recent history without downloading everything."""
        indexes = self._indexes()
        if not indexes:
            return {"offset": "0:0", "data": ""}
        index = indexes[-1]
        path = self._chunk_path(index)
        try:
            size = path.stat().st_size
        except OSError:
            return {"offset": "0:0", "data": ""}
        start = max(0, size - int(max_bytes))
        with path.open("rb") as fp:
            fp.seek(start)
            raw = fp.read()
        if start > 0:
            # Drop the first (probably partial) line so the client only ever
            # parses complete JSON records.
            newline = raw.find(b"\n")
            if newline >= 0:
                raw = raw[newline + 1 :]
        return {
            "offset": f"{index}:{size}",
            "data": raw.decode("utf-8", errors="replace"),
        }

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """Yield every parsed record in chunk order (used by tools/tests)."""
        for idx in self._indexes():
            try:
                with self._chunk_path(idx).open("r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except ValueError:
                            continue
            except OSError:
                continue
