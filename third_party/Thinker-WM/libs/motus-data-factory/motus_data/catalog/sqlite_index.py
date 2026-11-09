"""Immutable SQLite byte-offset index for an Episode Catalog JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping

from motus_data.index_contracts import (
    validate_catalog_metadata,
    validate_episode_catalog_binding,
)


CATALOG_OFFSET_INDEX_FORMAT = "motus_catalog_offset_index_v1"


class CatalogOffsetIndexError(ValueError):
    """Raised when an offset index is invalid or bound to another Catalog."""


def catalog_metadata_path(catalog_path: str | Path) -> Path:
    catalog = Path(catalog_path)
    if catalog.suffix == ".jsonl":
        return catalog.with_suffix(".meta.json")
    return Path(str(catalog) + ".meta.json")


def default_catalog_index_path(catalog_path: str | Path) -> Path:
    catalog = Path(catalog_path)
    if catalog.suffix == ".jsonl":
        return catalog.with_suffix(".index.sqlite")
    return Path(str(catalog) + ".index.sqlite")


def _catalog_binding(
    catalog: Path,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    expected_records = metadata.get("record_count", metadata.get("episodes"))
    expected_sha256 = metadata.get("sha256")
    if expected_sha256 is None:
        expected_sha256 = (metadata.get("statistics") or {}).get("catalog_sha256")
    if expected_records is None or expected_sha256 is None:
        raise CatalogOffsetIndexError(
            "Catalog metadata must publish record count and SHA256"
        )
    return {
        "format": CATALOG_OFFSET_INDEX_FORMAT,
        "catalog_id": str(metadata["catalog_id"]),
        "catalog_format": str(
            metadata.get("schema_version", metadata.get("format"))
        ),
        "catalog_sha256": str(expected_sha256),
        "catalog_size_bytes": str(catalog.stat().st_size),
        "catalog_records": str(int(expected_records)),
        "catalog_filename": catalog.name,
    }


def _read_index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.DatabaseError as error:
        raise CatalogOffsetIndexError(
            f"invalid Catalog offset index metadata: {error}"
        ) from error
    result = {str(key): str(value) for key, value in rows}
    required = {
        "format",
        "catalog_id",
        "catalog_format",
        "catalog_sha256",
        "catalog_size_bytes",
        "catalog_records",
        "catalog_filename",
    }
    missing = sorted(required - set(result))
    if missing:
        raise CatalogOffsetIndexError(
            f"Catalog offset index metadata missing fields: {missing}"
        )
    if result["format"] != CATALOG_OFFSET_INDEX_FORMAT:
        raise CatalogOffsetIndexError(
            f"unsupported Catalog offset index format={result['format']!r}"
        )
    return result


def build_catalog_offset_index(
    catalog_path: str | Path,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    """Validate a Catalog and atomically publish its Episode byte offsets."""

    catalog = Path(catalog_path).resolve()
    metadata_path = catalog_metadata_path(catalog)
    if not catalog.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Catalog index requires {catalog} and {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_catalog_metadata(metadata)
    catalog_size = catalog.stat().st_size

    output = (
        default_catalog_index_path(catalog)
        if output_path is None
        else Path(output_path).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"Catalog offset index already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)

    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    rows = 0
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE episodes ("
            "ordinal INTEGER PRIMARY KEY, "
            "episode_id TEXT NOT NULL UNIQUE, "
            "byte_offset INTEGER NOT NULL, "
            "byte_length INTEGER NOT NULL"
            ")"
        )

        batch = []
        with catalog.open("rb") as handle:
            while True:
                byte_offset = handle.tell()
                payload = handle.readline()
                if not payload:
                    break
                digest.update(payload)
                if not payload.strip():
                    raise CatalogOffsetIndexError(
                        f"blank Catalog row at byte offset {byte_offset}"
                    )
                try:
                    record = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CatalogOffsetIndexError(
                        f"invalid Catalog JSON at byte offset {byte_offset}: {error}"
                    ) from error
                validate_episode_catalog_binding(record, metadata)
                episode_id = str(record["episode_id"])
                if episode_id in seen_ids:
                    raise CatalogOffsetIndexError(
                        f"duplicate episode_id={episode_id!r}"
                    )
                seen_ids.add(episode_id)
                batch.append((rows, episode_id, byte_offset, len(payload)))
                rows += 1
                if len(batch) >= 4096:
                    connection.executemany(
                        "INSERT INTO episodes VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
                if progress_callback is not None:
                    progress_callback(
                        {
                            "records": rows,
                            "bytes": handle.tell(),
                            "total_bytes": catalog_size,
                        }
                    )
        if batch:
            connection.executemany(
                "INSERT INTO episodes VALUES (?, ?, ?, ?)",
                batch,
            )

        binding = _catalog_binding(catalog, metadata)
        actual_sha256 = digest.hexdigest()
        if rows != int(binding["catalog_records"]):
            raise CatalogOffsetIndexError(
                f"Catalog rows={rows} differ from metadata={binding['catalog_records']}"
            )
        if actual_sha256 != binding["catalog_sha256"]:
            raise CatalogOffsetIndexError(
                "Catalog SHA256 differs from metadata: "
                f"actual={actual_sha256} expected={binding['catalog_sha256']}"
            )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(binding.items()),
        )
        connection.commit()
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise CatalogOffsetIndexError("SQLite quick_check did not return ok")
        indexed_rows = int(
            connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        )
        if indexed_rows != rows:
            raise CatalogOffsetIndexError(
                f"SQLite rows={indexed_rows} differ from Catalog rows={rows}"
            )
        connection.close()
        connection = None

        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if force:
            output.unlink(missing_ok=True)
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            **binding,
            "index_path": str(output),
            "index_size_bytes": output.stat().st_size,
        }
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


class CatalogOffsetIndex:
    """Read-only process-safe access to one immutable Catalog offset index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        catalog_path: str | Path,
        catalog_metadata: Mapping[str, Any],
    ):
        self.path = Path(index_path).resolve()
        self.catalog_path = Path(catalog_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        expected = _catalog_binding(self.catalog_path, catalog_metadata)
        connection = self._open_connection()
        try:
            actual = _read_index_metadata(connection)
            for key, expected_value in expected.items():
                if actual.get(key) != expected_value:
                    raise CatalogOffsetIndexError(
                        f"Catalog offset index binding mismatch for {key}: "
                        f"index={actual.get(key)!r} catalog={expected_value!r}"
                    )
            count = int(
                connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            )
            if count != int(expected["catalog_records"]):
                raise CatalogOffsetIndexError(
                    f"Catalog offset index rows={count} "
                    f"differ from metadata={expected['catalog_records']}"
                )
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise CatalogOffsetIndexError(
                    "Catalog offset index quick_check did not return ok"
                )
        finally:
            connection.close()
        self.metadata = expected
        self._count = int(expected["catalog_records"])
        self._connection: sqlite3.Connection | None = None
        self._pid: int | None = None
        self._lock = threading.RLock()

    def _open_connection(self) -> sqlite3.Connection:
        uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)

    def _get_connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        with self._lock:
            if self._connection is None or self._pid != pid:
                if self._connection is not None:
                    self._connection.close()
                self._connection = self._open_connection()
                self._pid = pid
            return self._connection

    def __len__(self) -> int:
        return self._count

    def locate(self, episode_id: str) -> tuple[int, int]:
        with self._lock:
            row = self._get_connection().execute(
                "SELECT byte_offset, byte_length FROM episodes WHERE episode_id=?",
                (str(episode_id),),
            ).fetchone()
        if row is None:
            raise KeyError(str(episode_id))
        return int(row[0]), int(row[1])

    def iter_episode_ids(self) -> Iterator[str]:
        last_ordinal = -1
        while True:
            with self._lock:
                rows = self._get_connection().execute(
                    "SELECT ordinal, episode_id FROM episodes "
                    "WHERE ordinal > ? ORDER BY ordinal LIMIT 4096",
                    (last_ordinal,),
                ).fetchall()
            if not rows:
                return
            for ordinal, episode_id in rows:
                last_ordinal = int(ordinal)
                yield str(episode_id)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_pid"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
