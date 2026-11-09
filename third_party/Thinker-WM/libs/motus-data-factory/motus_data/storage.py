"""Read-only bounded views over byte ranges in large container files."""

from __future__ import annotations

import io
import os
from bisect import bisect_right
from pathlib import Path
from typing import Sequence


class BoundedFileReader(io.RawIOBase):
    """Expose one `[offset, offset + size)` region as an independent file."""

    def __init__(self, path: str | Path, offset: int, size: int):
        super().__init__()
        self.path = str(path)
        self.offset = int(offset)
        self.size = int(size)
        if self.offset < 0 or self.size <= 0:
            raise ValueError("bounded file offset must be >= 0 and size must be > 0")
        self._handle = open(self.path, "rb", buffering=0)
        file_size = os.fstat(self._handle.fileno()).st_size
        if self.offset + self.size > file_size:
            self._handle.close()
            raise ValueError(
                f"bounded range exceeds file: offset={self.offset} size={self.size} "
                f"file_size={file_size} path={self.path}"
            )
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_SET:
            target = int(offset)
        elif whence == io.SEEK_CUR:
            target = self._position + int(offset)
        elif whence == io.SEEK_END:
            target = self.size + int(offset)
        else:
            raise ValueError(f"unsupported whence={whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self._position = target
        return self._position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        remaining = max(0, self.size - self._position)
        requested = remaining if size is None or int(size) < 0 else min(int(size), remaining)
        if requested <= 0:
            return b""
        self._handle.seek(self.offset + self._position)
        data = self._handle.read(requested)
        self._position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        count = len(data)
        buffer[:count] = data
        return count

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


class ConcatenatedFileReader(io.RawIOBase):
    """Expose ordered read-only files as one seekable logical byte stream."""

    def __init__(self, paths: Sequence[str | Path]):
        super().__init__()
        if not paths:
            raise ValueError("concatenated file requires at least one path")
        self.paths = tuple(str(Path(path)) for path in paths)
        self._handles = []
        self._offsets = [0]
        try:
            for path in self.paths:
                handle = open(path, "rb", buffering=0)
                self._handles.append(handle)
                size = os.fstat(handle.fileno()).st_size
                if size <= 0:
                    raise ValueError(
                        f"concatenated file part is empty: {path}"
                    )
                self._offsets.append(self._offsets[-1] + size)
        except Exception:
            for handle in self._handles:
                handle.close()
            raise
        self._position = 0

    @property
    def size(self) -> int:
        return self._offsets[-1]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_SET:
            target = int(offset)
        elif whence == io.SEEK_CUR:
            target = self._position + int(offset)
        elif whence == io.SEEK_END:
            target = self.size + int(offset)
        else:
            raise ValueError(f"unsupported whence={whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self._position = target
        return self._position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        remaining = max(0, self.size - self._position)
        requested = (
            remaining
            if size is None or int(size) < 0
            else min(int(size), remaining)
        )
        if requested <= 0:
            return b""

        chunks = []
        bytes_read = 0
        while bytes_read < requested and self._position < self.size:
            part_index = bisect_right(
                self._offsets,
                self._position,
            ) - 1
            part_end = self._offsets[part_index + 1]
            part_remaining = part_end - self._position
            count = min(requested - bytes_read, part_remaining)
            handle = self._handles[part_index]
            handle.seek(self._position - self._offsets[part_index])
            chunk = handle.read(count)
            if not chunk:
                raise OSError(
                    "unexpected EOF in concatenated file part "
                    f"{self.paths[part_index]}"
                )
            chunks.append(chunk)
            self._position += len(chunk)
            bytes_read += len(chunk)
        return b"".join(chunks)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        count = len(data)
        buffer[:count] = data
        return count

    def close(self) -> None:
        if not self.closed:
            for handle in self._handles:
                handle.close()
        super().close()
