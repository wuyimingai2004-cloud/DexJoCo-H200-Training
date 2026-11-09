"""Task/episode-uniform training draws, retaining same-episode batch decoding.

Each virtual group draws one task and one eligible episode with replacement.
Its slots optionally share an episode-wide Feistel deck. A virtual epoch
is a scheduling budget, NOT a traversal or a unique-window coverage guarantee.
"""
from __future__ import annotations

import logging
import random
from array import array
from collections import defaultdict

from motus_data.sampling import compose_seed

from .pool_providers import register_pool_builder
from .sampling_ranges import SamplingRangesError, SamplingRangesPool


class TaskBalancedRangesPool(SamplingRangesPool):
    def __init__(self, ranges_path, repository, *, group_size,
                 without_replacement=False, **options):
        if isinstance(group_size, bool) or int(group_size) <= 0:
            raise SamplingRangesError("group_size must be a positive batch size")
        self.group_size = int(group_size)
        # Avoid allocating the old length-weighted slots; only reuse valid ranges.
        super().__init__(ranges_path, repository, windows_per_episode=1,
                         without_replacement=without_replacement, **options)
        if self._without_replacement and self._stride_max != 1:
            raise SamplingRangesError("balanced no-replacement requires stride_max=1")
        eligible = {ep.episode_id: ep for ep in self._episodes}
        task_episodes = defaultdict(list)
        for info in repository.iter_episode_infos():
            if info.episode_id not in eligible:
                continue
            task = info.metadata.get("task_name")
            if not isinstance(task, str) or not task.strip():
                raise SamplingRangesError(
                    f"missing task_name for eligible episode {info.episode_id}"
                )
            task_episodes[task].append(eligible.pop(info.episode_id))
        if eligible:
            raise SamplingRangesError("ranges contain episodes absent from catalog")
        self._tasks = tuple(sorted(task_episodes))
        self._task_episodes = tuple(
            tuple(sorted(task_episodes[task], key=lambda ep: ep.episode_id))
            for task in self._tasks
        )
        # Preserve the previous grouped sampler's nominal scheduling budget.
        group_count = sum(ep.total_starts // self.group_size
                          for ep in self._episodes)
        if group_count <= 0:
            raise SamplingRangesError("ranges produce no full virtual batches")
        self._pool_id += "_task_episode_uniform_v1"
        self._episode_groups = {
            f"draw:{group:09d}": tuple(range(group * self.group_size,
                                           (group + 1) * self.group_size))
            for group in range(group_count)
        }
        self._length = group_count * self.group_size
        self._draw_offsets = None
        if self._without_replacement:
            self._build_draw_offsets()
        logging.getLogger(__name__).info(
            "Task-balanced ranges: %d tasks, %d eligible episodes, %d virtual "
            "batches; task/episode sampled per batch, window_without_replacement=%s; "
            "no full-coverage guarantee", len(self._tasks), len(self._episodes),
            group_count, self._without_replacement,
        )

    def _build_draw_offsets(self):
        # Allocate per-episode draw ordinals BEFORE rank sharding/queue prefetch.
        # No mutable consumption counters: restart and arbitrary access agree.
        counts = defaultdict(int)
        offsets = array("Q")
        for group in range(self._length // self.group_size):
            episode, _ = self._locate(group * self.group_size)
            offsets.append(counts[episode.episode_id])
            counts[episode.episode_id] += self.group_size
        self._draw_offsets = offsets

    def set_epoch(self, epoch):
        changed = int(epoch) != self._epoch
        super().set_epoch(epoch)
        if changed and self._without_replacement:
            self._build_draw_offsets()

    def _window_draw(self, index, offset):
        if self._draw_offsets is None:
            return super()._window_draw(index, offset)
        group, slot = divmod(int(index), self.group_size)
        return int(self._draw_offsets[group]) + slot

    def _locate(self, index):
        if not 0 <= index < self._length:
            raise IndexError(index)
        group = index // self.group_size
        rng = random.Random(compose_seed(
            self._seed, self._epoch, 0, 0, self._pool_id + ":group", group,
        ))
        episodes = self._task_episodes[rng.randrange(len(self._tasks))]
        return episodes[rng.randrange(len(episodes))], group * self.group_size


def _build(repository, *, path, **options):
    return TaskBalancedRangesPool(path, repository, **options)


register_pool_builder(
    "task_balanced_ranges", _build,
    known_options=("path", "num_video_frames", "stride_max", "text_choice",
                   "action_start_offset", "group_size", "without_replacement"),
)
