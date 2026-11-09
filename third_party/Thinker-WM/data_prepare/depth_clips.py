"""``depth_clips`` SamplePool: sample exactly the clips that have depth on disk.

The depth sidecars store only a few deterministic windows per episode (see
``scripts/dump_*_depth.py``); random ``sampling_ranges`` windows would almost
never coincide, leaving the depth stream empty. This pool restricts a source to
exactly the dumped clips (from ``depth_selections.jsonl``), so the sampled RGB
``video_frame_indices`` equal the stored depth clip and the two streams align
frame-for-frame.

Each stored clip is turned back into a full ``motus_sample_selection_v1`` via the
existing, tested ``build_selection`` (reusing its text/state/action index rules),
keyed off the clip's start (``clip[0]``) and native stride (``clip[1]-clip[0]``).
Clips whose episode is absent from this source's catalog, or that no longer pass
selection validation, are skipped.

Implements ``motus_data.SupportsSamplePool`` (static selections; no ``set_epoch``
because the clip set is fixed on disk).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Sequence

from ._bootstrap import ensure_motus_data_importable

ensure_motus_data_importable()

from .sampling_ranges import (  # noqa: E402
    LAYOUT_CONSECUTIVE,
    LAYOUT_VIDEO_STRIDE,
    build_selection,
)

logger = logging.getLogger(__name__)


class DepthClipsError(ValueError):
    """Raised when a depth-clips pool cannot be built."""


_VALID_SPLITS = ("train", "holdout", "all")


def _episode_in_split(
    episode_id: str, *, holdout_fraction: float, split: str, holdout_seed: int
) -> bool:
    """Deterministic episode-level split (no clip-level leakage).

    Buckets each ``episode_id`` into ``[0, 1)`` via a stable hash keyed by
    ``holdout_seed``; episodes with bucket < ``holdout_fraction`` are the
    holdout. ``split='all'`` (or ``holdout_fraction<=0``) keeps everything.
    The same episode always lands in the same split across processes/workers.
    """
    if split == "all" or holdout_fraction <= 0.0:
        return True
    digest = hashlib.sha1(f"{holdout_seed}:{episode_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    is_holdout = bucket < holdout_fraction
    return is_holdout if split == "holdout" else not is_holdout


class DepthClipsPool:
    pool_kind = "depth_clips"

    def __init__(
        self,
        repository: Any,
        *,
        roots: Sequence[str],
        num_video_frames: int,
        prod_keep_only: bool = True,
        holdout_fraction: float = 0.0,
        split: str = "train",
        holdout_seed: int = 20260824,
    ):
        from thinker_wm.models.depth_reader import load_depth_selections

        if not roots:
            raise DepthClipsError("depth_clips pool requires at least one root")
        if split not in _VALID_SPLITS:
            raise DepthClipsError(
                f"depth_clips split must be one of {_VALID_SPLITS}, got {split!r}"
            )
        self._num_video_frames = int(num_video_frames)
        self._holdout_fraction = float(holdout_fraction)
        self._split = str(split)
        self._holdout_seed = int(holdout_seed)
        index = load_depth_selections(roots, prod_keep_only=prod_keep_only)
        repo_ids = set(str(eid) for eid in repository.iter_episode_ids())

        selections: list[Mapping[str, Any]] = []
        groups: dict[str, list[int]] = {}
        skipped_missing_episode = 0
        skipped_bad_clip = 0
        skipped_other_split = 0

        for episode_id, entry in index.items():
            if episode_id not in repo_ids:
                skipped_missing_episode += 1
                continue
            if not _episode_in_split(
                episode_id,
                holdout_fraction=self._holdout_fraction,
                split=self._split,
                holdout_seed=self._holdout_seed,
            ):
                skipped_other_split += 1
                continue
            episode = repository.get_episode(episode_id)
            for clip in entry.clips:
                selection = self._selection_from_clip(episode, clip)
                if selection is None:
                    skipped_bad_clip += 1
                    continue
                position = len(selections)
                selections.append(selection)
                groups.setdefault(episode_id, []).append(position)

        if not selections:
            raise DepthClipsError(
                f"depth_clips pool built no selections for catalog "
                f"{getattr(repository, 'catalog_id', '?')} split={self._split} "
                f"(missing_episode={skipped_missing_episode}, bad_clip="
                f"{skipped_bad_clip}, other_split={skipped_other_split})"
            )
        logger.info(
            "depth_clips pool: %d selections over %d episodes "
            "(split=%s holdout_fraction=%.3f; skipped missing_episode=%d "
            "bad_clip=%d other_split=%d)",
            len(selections),
            len(groups),
            self._split,
            self._holdout_fraction,
            skipped_missing_episode,
            skipped_bad_clip,
            skipped_other_split,
        )
        self._selections = tuple(selections)
        self._episode_groups = {eid: tuple(pos) for eid, pos in groups.items()}
        dataset = getattr(repository, "catalog_id", "depth")
        # split 进 pool_id：train/holdout 池 id 不同，避免缓存/日志混淆。
        split_tag = "" if self._split == "all" else f"_{self._split}"
        self._pool_id = f"{dataset}_depth_clips{split_tag}_v1"

    def _selection_from_clip(
        self, episode: Any, clip: Sequence[int]
    ) -> Mapping[str, Any] | None:
        if len(clip) != self._num_video_frames + 1:
            return None
        start = int(clip[0])
        stride = int(clip[1]) - int(clip[0]) if len(clip) >= 2 else 1
        if stride <= 0:
            return None
        layout = LAYOUT_VIDEO_STRIDE if stride > 1 else LAYOUT_CONSECUTIVE
        try:
            selection = build_selection(
                episode,
                start=start,
                num_video_frames=self._num_video_frames,
                ratio=stride,
                layout=layout,
            )
        except Exception:
            return None
        if selection is None:
            return None
        # Guard: the reconstructed video frames must equal the dumped clip, or
        # depth would silently misalign with RGB.
        if [int(i) for i in selection["video_frame_indices"]] != [int(i) for i in clip]:
            return None
        return selection

    @property
    def selections(self) -> Sequence[Mapping[str, Any]]:
        return self._selections

    @property
    def episode_groups(self) -> Mapping[str, tuple[int, ...]]:
        return self._episode_groups

    @property
    def pool_id(self) -> str:
        return self._pool_id

    def __len__(self) -> int:
        return len(self._selections)


def _build_pool_from_options(
    repository: Any,
    *,
    seed: int,
    roots: Sequence[str],
    num_video_frames: int,
    prod_keep_only: bool = True,
    holdout_fraction: float = 0.0,
    split: str = "train",
    holdout_seed: int = 20260824,
) -> DepthClipsPool:
    return DepthClipsPool(
        repository,
        roots=roots,
        num_video_frames=int(num_video_frames),
        prod_keep_only=bool(prod_keep_only),
        holdout_fraction=float(holdout_fraction),
        split=str(split),
        holdout_seed=int(holdout_seed),
    )


def register_provider() -> None:
    from .pool_providers import register_pool_builder

    register_pool_builder(
        "depth_clips",
        _build_pool_from_options,
        known_options=(
            "roots", "num_video_frames", "prod_keep_only",
            "holdout_fraction", "split", "holdout_seed",
        ),
    )


register_provider()
