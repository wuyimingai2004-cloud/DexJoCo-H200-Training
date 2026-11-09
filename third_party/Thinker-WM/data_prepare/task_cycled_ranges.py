"""Task-balanced draws with episode cycling and time-stratified windows.

Same task weights as ``task_balanced_ranges`` (uniform over tasks), but:

- tasks are dealt in shuffled rounds of all tasks;
- each task deals its episodes from a shuffled deck, reshuffled after every
  episode has been used once;
- the ``group_size`` windows of one batch come from ``group_size`` equal
  time strata of the episode's valid starts, one start per stratum; every
  stratum keeps its own no-replacement deck across that episode's uses.

Decks are ordered by *consumption order*, not by group id: the downstream
``EpisodeGroupedBatchSampler`` shuffles groups with ``Random(seed + epoch *
1_000_003)`` and deals ``batches[rank::world_size]``, so global batch position
``p`` is consumed by rank ``p % world_size`` as its ``p // world_size``-th
batch of this source. The pool replays that shuffle to map group -> ``p``.

Each later cycle is a new shuffle of the same deck. ``install_exact_resume``
maps ``global_step`` to ``cycle = (step - base) // batches_per_epoch`` and an
offset inside that cycle, then stores that cycle on both the stream and
``ReadyBatchQueue._epoch``. The queue increments its own counter when an
iterator ends. Leaving the counter at 0 replays cycle 1 after the resumed
cycle, so the samples diverge from an uninterrupted run.
"""
from __future__ import annotations

import logging
import random
from array import array
from collections import Counter

from motus_data.sampling import compose_seed

from .pool_providers import register_pool_builder
from .sampling_ranges import SamplingRangesError, _feistel_permute
from .task_balanced_ranges import TaskBalancedRangesPool

_SAMPLER_EPOCH_STRIDE = 1_000_003


class TaskCycledRangesPool(TaskBalancedRangesPool):
    def __init__(self, ranges_path, repository, *, group_size, cycle_base_step, **options):
        if options.pop("without_replacement", None) is not None:
            raise SamplingRangesError(
                "task_cycled_ranges always deals without replacement; drop the option")
        super().__init__(ranges_path, repository, group_size=group_size,
                         without_replacement=False, **options)
        if self._stride_max != 1:
            raise SamplingRangesError("task_cycled_ranges requires stride_max=1")
        if int(cycle_base_step) < 0:
            raise SamplingRangesError("cycle_base_step must be >= 0")
        self._pool_id += "_cycled_stratified_v3"
        self._cycle_base_step = int(cycle_base_step)
        self._group_count = self._length // self.group_size
        self._position_of_group = None
        self._build_positions()
        logging.getLogger(__name__).info(
            "Task-cycled ranges: %d tasks, %d episodes, %d groups; episode decks "
            "per task, %d time strata per batch, base epoch %d", len(self._tasks),
            len(self._episodes), self._group_count, self.group_size,
            self._cycle_base_step)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise SamplingRangesError(f"task_cycled_ranges epoch must be >= 0, got {epoch}")
        changed = epoch != self._epoch
        super().set_epoch(epoch)
        if changed:
            self._build_positions()

    def _build_positions(self):
        # Must replay EpisodeGroupedBatchSampler._global_batches (shuffle=True)
        # exactly: groups in sorted key order, one 8-slot shuffle each, then
        # one shuffle of the batch list.
        rng = random.Random(self._seed + self._epoch * _SAMPLER_EPOCH_STRIDE)
        slots = list(range(self.group_size))
        for _ in range(self._group_count):
            rng.shuffle(slots)
        order = list(range(self._group_count))
        rng.shuffle(order)
        positions = array("Q", bytes(8 * self._group_count))
        for position, group in enumerate(order):
            positions[group] = position
        self._position_of_group = positions

    def _key(self, *parts):
        *labels, index = parts
        name = "|".join([self._pool_id, *map(str, labels)])
        # Epoch 0 keeps the previous seed. Later epochs reshuffle the deck.
        return compose_seed(self._seed, self._cycle_base_step, self._epoch, 0, name, int(index))

    def _deal(self, group):
        """Return (task_index, episode, use_count) for one group."""
        return self.deal_at(int(self._position_of_group[group]))

    def deal_at(self, position):
        """Return (task_index, episode, use_count) for one deck position."""
        task_count = len(self._tasks)
        round_index, seat = divmod(position, task_count)
        task = _feistel_permute(seat, task_count, self._key("task", round_index))
        episodes = self._task_episodes[task]
        use, slot = divmod(round_index, len(episodes))
        episode = episodes[_feistel_permute(
            slot, len(episodes), self._key("episode", task, use))]
        return task, episode, use

    def _locate(self, index):
        if not 0 <= index < self._length:
            raise IndexError(index)
        group = index // self.group_size
        _task, episode, _use = self._deal(group)
        return episode, group * self.group_size

    def _window_draw(self, index, offset):
        _task, _episode, use = self._deal(int(index) // self.group_size)
        return use * self.group_size + (int(index) - offset)

    def _pick_start(self, episode, draw, rng):
        use, stratum = divmod(int(draw), self.group_size)
        total = episode.total_starts
        lo = stratum * total // self.group_size
        hi = (stratum + 1) * total // self.group_size
        if hi <= lo:
            return episode.start_at(min(lo, total - 1))
        size = hi - lo
        rotation, position = divmod(use, size)
        key = self._key("window", episode.episode_id, stratum, rotation)
        return episode.start_at(lo + _feistel_permute(position, size, key))


def install_exact_resume(base_step):
    """Map train.py's ``set_epoch(global_step)`` onto one data cycle.

    The call happens before the ready queue starts. It selects cycle
    ``(step - base) // batches_per_epoch``, skips the offset inside that
    cycle, and stores the same cycle in ``ReadyBatchQueue._epoch``. The queue
    increments that counter when the iterator ends. Left at 0, the next cycle
    would replay epoch 1.
    """
    from motus_data.indexed_mix import (IndexedMixedTrainingDataPipeline,
                                        WeightedIndexedBatchStream)
    from motus_data.sample_pool import EpisodeGroupedBatchSampler

    base_step = int(base_step)
    if getattr(IndexedMixedTrainingDataPipeline, "_exact_resume_base", None) is not None:
        if IndexedMixedTrainingDataPipeline._exact_resume_base != base_step:
            raise RuntimeError("install_exact_resume called with a different base step")
        return
    original_set_epoch = IndexedMixedTrainingDataPipeline.set_epoch
    original_stream_iter = WeightedIndexedBatchStream.__iter__
    original_sampler_iter = EpisodeGroupedBatchSampler.__iter__

    def set_epoch(self, epoch):
        step = int(epoch) - base_step
        if step < 0:
            raise RuntimeError(f"resume step {epoch} precedes cycle base {base_step}")
        per_epoch = int(self.stream.batches_per_epoch)
        cycle, offset = divmod(step, per_epoch)
        cycle_epoch = base_step + cycle
        queue = getattr(self, "queue", None)
        if queue is not None and getattr(queue, "started", False):
            raise RuntimeError(
                "cannot sync the ready-queue epoch after it has started")
        original_set_epoch(self, cycle_epoch)
        self.stream._resume_skip_steps = offset
        if queue is not None:
            # StopIteration does _epoch += 1 and then set_epoch on the stream.
            # The counter has to already be this cycle, or the next pass is epoch 1.
            queue._epoch = cycle_epoch
        logging.getLogger(__name__).info(
            "Exact resume: data cycle %d, skipping %d batches into it "
            "(consumed step %d, %d batches/epoch, queue epoch %d)",
            cycle_epoch, offset, int(epoch), per_epoch, cycle_epoch)

    def stream_iter(self):
        skip = int(getattr(self, "_resume_skip_steps", 0) or 0)
        self._resume_skip_steps = 0
        if not skip:
            yield from original_stream_iter(self)
            return
        schedule = type(self).iter_source_schedule(self)
        consumed = Counter()
        for _ in range(skip):
            consumed[next(schedule)] += 1
        for index, child in enumerate(self.children):
            child.batch_sampler._resume_skip_batches = consumed[index]
            child._iterator = None
        # mix_step_in_epoch restarts at 0 after a resume; the batches are exact.
        self.iter_source_schedule = lambda: schedule
        try:
            yield from original_stream_iter(self)
        finally:
            del self.iter_source_schedule

    def sampler_iter(self):
        skip = int(getattr(self, "_resume_skip_batches", 0) or 0)
        self._resume_skip_batches = 0
        batches = self._rank_batches()
        if skip > len(batches):
            raise RuntimeError("resume skip exceeds this rank's batches in the epoch")
        yield from batches[skip:]

    IndexedMixedTrainingDataPipeline.set_epoch = set_epoch
    WeightedIndexedBatchStream.__iter__ = stream_iter
    EpisodeGroupedBatchSampler.__iter__ = sampler_iter
    IndexedMixedTrainingDataPipeline._exact_resume_base = base_step
    IndexedMixedTrainingDataPipeline._exact_resume_originals = (
        original_set_epoch, original_stream_iter, original_sampler_iter)


def _build(repository, *, path, **options):
    return TaskCycledRangesPool(path, repository, **options)


register_pool_builder(
    "task_cycled_ranges", _build,
    known_options=("path", "num_video_frames", "stride_max", "text_choice",
                   "action_start_offset", "group_size", "cycle_base_step"),
)
