"""Depth sidecar reader: (episode_id, native frame indices) -> uint16-mm frames.

Reads the metric-depth sidecars produced by ``scripts/dump_*_depth.py`` and
returns exactly the requested native video-frame indices as ``uint16 mm``. This
is the disk-facing half of the depth branch; ``models.depth_codec`` turns the mm
frames into the model tensor.

Two on-disk layouts are supported transparently (per ``metadata.json``):
  - ``clips_native_index`` (EgoScale FFS): only a subset of frames is stored;
    ``metadata.frame_index`` is the sorted union of native indices, and array
    row ``r`` holds native frame ``frame_index[r]``. A requested native index
    absent from that list means the frame was never dumped -> not present.
  - full timeline (AgiBot DA3 legacy npz): every native frame ``i`` is stored at
    row ``i``; the array's first dim equals the episode length.

Storage is ``depth.zarr`` (zstd) when available, else ``depth.npz`` (key
``depth``). ``depth_selections.jsonl`` at each root maps ``episode_id -> source``
(hence the on-disk directory) and lists the dumped clips; only ``prod_keep=true``
episodes are indexed (the depth QC gate).

Missing episode / missing file / any requested frame not resolvable -> the read
returns ``(None, present=False)`` so callers can drop the depth stream for that
sample instead of failing the batch.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# Only the live index files. Backups like depth_selections.shard10.pre_qc_v4.jsonl
# must not be globbed — later-root/later-file wins would resurrect DROP rows.
_LIVE_SHARD_NAME = re.compile(r"^depth_selections\.shard\d+\.jsonl$")

import numpy as np
import torch


@dataclass(frozen=True)
class _Entry:
    root: Path
    source: str
    clips: tuple[tuple[int, ...], ...]


def _episode_dirname(episode_id: str) -> str:
    # Matches ep_dir() in the dump scripts: ':' -> '_' (episode_id is globally
    # unique across sources, so this is a safe filesystem-friendly key).
    return episode_id.replace(":", "_")


def load_depth_selections(
    roots: Sequence[str | Path],
    *,
    prod_keep_only: bool = True,
) -> dict[str, _Entry]:
    """Parse ``depth_selections.jsonl`` under each root into ``episode_id -> _Entry``.

    Later roots win on episode_id collision (kept deterministic by input order).
    Rows missing required fields or with ``prod_keep=false`` (when
    ``prod_keep_only``) are skipped.
    """
    index: dict[str, _Entry] = {}
    for root in roots:
        root = Path(root)
        paths = [root / "depth_selections.jsonl"]
        if root.is_dir():
            paths.extend(
                sorted(
                    p for p in root.iterdir()
                    if p.is_file() and _LIVE_SHARD_NAME.match(p.name)
                )
            )
        for path in paths:
            if not path.is_file():
                continue
            _load_selection_file(path, root, index, prod_keep_only)
    return index


def _load_selection_file(
    path: Path,
    root: Path,
    index: dict[str, _Entry],
    prod_keep_only: bool,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_id = row.get("episode_id")
            source = row.get("source")
            clips = row.get("clips")
            if not episode_id or not source or not isinstance(clips, list):
                continue
            if prod_keep_only and not bool(row.get("prod_keep", False)):
                continue
            index[str(episode_id)] = _Entry(
                root=root,
                source=str(source),
                clips=tuple(tuple(int(i) for i in clip) for clip in clips),
            )


class DepthSidecarReader:
    """Resolve depth frames for a sampled clip, keyed by episode + native indices.

    Per-process caches keep the parsed selections index, per-episode metadata,
    and open **zarr** handles (chunked, not the full volume). ``depth.npz`` is
    never cached: ``np.load(...)["depth"]`` materializes the whole
    ``[T,H,W]`` uint16 array (EgoScale ~90 MB, AgiBot fulltimeline up to ~3 GB)
    and an unbounded per-episode cache of those arrays was the V4/V5 host-RSS
    leak. Instances are cheap to re-create per DataLoader worker.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        prod_keep_only: bool = True,
    ):
        self._index = load_depth_selections(roots, prod_keep_only=prod_keep_only)
        self._meta_cache: dict[str, tuple[dict, dict[int, int] | None, int]] = {}
        # zarr store handles only. npz arrays must not go here.
        self._array_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self._index)

    def has(self, episode_id: str) -> bool:
        return str(episode_id) in self._index

    @property
    def episodes(self) -> Mapping[str, _Entry]:
        return self._index

    def _ep_dir(self, episode_id: str) -> Path | None:
        entry = self._index.get(str(episode_id))
        if entry is None:
            return None
        return entry.root / entry.source / _episode_dirname(episode_id)

    def _load_meta(
        self, episode_id: str, ep_dir: Path
    ) -> tuple[dict, dict[int, int] | None, int] | None:
        cached = self._meta_cache.get(str(episode_id))
        if cached is not None:
            return cached
        meta_path = ep_dir / "metadata.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        frame_index = meta.get("frame_index")
        native_to_row: dict[int, int] | None = None
        if isinstance(frame_index, list) and frame_index:
            native_to_row = {int(n): r for r, n in enumerate(frame_index)}
        shape = meta.get("shape") or []
        rows_total = int(shape[0]) if shape else 0
        result = (meta, native_to_row, rows_total)
        self._meta_cache[str(episode_id)] = result
        return result

    def _load_rows(self, episode_id: str, ep_dir: Path, rows: np.ndarray):
        zarr_path = ep_dir / "depth.zarr"
        npz_path = ep_dir / "depth.npz"
        if zarr_path.exists():
            cached = self._array_cache.get(str(episode_id))
            if cached is None:
                import zarr  # local import: optional dependency

                cached = ("zarr", zarr.open(str(zarr_path), mode="r"))
                self._array_cache[str(episode_id)] = cached
            kind, arr = cached
            if kind != "zarr":
                return None
            return np.asarray(arr.oindex[rows])
        if npz_path.is_file():
            # Compressed npz cannot mmap. Load, copy the requested rows, drop
            # the full array immediately so RSS cannot grow with unique episodes.
            with np.load(npz_path) as npz:
                arr = npz["depth"]
                return np.asarray(arr[rows])
        return None

    def read_mm(
        self, episode_id: str, frame_indices: Sequence[int]
    ) -> tuple[torch.Tensor | None, bool]:
        """Return ``([T, H, W] int32 mm, present)`` for the given native indices.

        ``present=False`` (with ``None``) when the episode is not indexed, the
        sidecar files are missing, or any requested native index cannot be
        resolved to a stored row (strict: a partially-covered clip is dropped
        whole rather than silently padded).
        """
        ep_dir = self._ep_dir(episode_id)
        if ep_dir is None:
            return None, False
        loaded = self._load_meta(episode_id, ep_dir)
        if loaded is None:
            return None, False
        _meta, native_to_row, rows_total = loaded

        wanted = [int(i) for i in frame_indices]
        if native_to_row is not None:
            if any(i not in native_to_row for i in wanted):
                return None, False
            rows = np.asarray([native_to_row[i] for i in wanted], dtype=np.int64)
        else:
            if rows_total <= 0 or any(i < 0 or i >= rows_total for i in wanted):
                return None, False
            rows = np.asarray(wanted, dtype=np.int64)

        arr = self._load_rows(episode_id, ep_dir, rows)
        if arr is None:
            return None, False
        return torch.from_numpy(arr.astype(np.int32)), True
