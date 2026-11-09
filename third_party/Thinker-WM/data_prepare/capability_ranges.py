"""Keep verified same-context branches together in a local training batch."""
from __future__ import annotations

import hashlib
from collections import defaultdict

from .capability_annotations import load_annotations
from .pool_providers import register_pool_builder
from .sampling_ranges import SamplingRangesPool, SamplingRangesError, _EpisodeRanges, LAYOUT_VIDEO_STRIDE


class CapabilityRangesPool(SamplingRangesPool):
    def __init__(self, path, repository, *, annotations_path, group_size, **options):
        if isinstance(group_size, bool) or int(group_size) < 2 or int(group_size) % 2:
            raise SamplingRangesError("capability_ranges requires an even local batch size >= 2")
        super().__init__(path, repository, windows_per_episode=1, **options)
        if self._layout != LAYOUT_VIDEO_STRIDE or self._stride_max != 1 or self._video_only:
            raise SamplingRangesError("capability_ranges requires fixed-stride robot action/video data")
        records = load_annotations(annotations_path)
        episodes = {ep.episode_id: ep for ep in self._episodes}
        groups = defaultdict(list)
        for (episode_id, start), row in sorted(records.items()):
            ep = episodes.get(episode_id)
            if ep is None or not any(lo <= start <= hi for lo, hi in ep.intervals):
                raise SamplingRangesError(f"annotated window absent from training ranges: {episode_id}@{start}")
            # One fixed legal start; reuse all standard action/text indexing.
            fixed = _EpisodeRanges(episode_id, ep.ratio, [(start, start)], ep.text_payload,
                                   episode_len=ep.episode_len, max_stride=1)
            groups[row["context_id"]].append(fixed)
        self._fixed = []
        self._episode_groups = {}
        size = int(group_size)
        for context, entries in sorted(groups.items()):
            if len(entries) < 2 or len(entries) > size:
                raise SamplingRangesError(f"context {context!r} needs 2..{size} branches in one batch")
            # Repeats are explicit padding; annotations and pairing diagnostics
            # retain their identity. No cross-context padding or random start.
            begin = len(self._fixed)
            self._fixed.extend(entries[i % len(entries)] for i in range(size))
            self._episode_groups[f"capability:{context}"] = tuple(range(begin, begin + size))
        self._length = len(self._fixed)
        digest = hashlib.sha256(repr(sorted(records.items())).encode()).hexdigest()[:16]
        self._pool_id += f"_capability_{digest}"

    def _locate(self, index):
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        return self._fixed[int(index)], int(index)


def _build(repository, **options):
    return CapabilityRangesPool(repository=repository, **options)


register_pool_builder("capability_ranges", _build, known_options=(
    "path", "annotations_path", "group_size", "num_video_frames", "text_choice", "action_start_offset"))
