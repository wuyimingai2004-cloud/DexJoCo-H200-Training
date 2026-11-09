"""V16d: chronological, globally dealt episodes and two-episode batches.

The mixer still chooses one source per rank/update. A source child reads two
four-window episode groups, then merges their adapted tensors. No vendor or
data-factory files are modified. A shared deterministic plan maps each rank's
source-local draw count to its actual global consumption ordinal.
"""
from __future__ import annotations

import copy
import logging
import random
from array import array
from functools import lru_cache
from types import SimpleNamespace

from motus_data.indexed_mix import WeightedIndexedBatchStream
from motus_data.sampling import compose_seed
from thinker_wm.data_prepare.pool_providers import register_pool_builder
from thinker_wm.data_prepare.sampling_ranges import SamplingRangesError, _feistel_permute
from thinker_wm.data_prepare.task_balanced_ranges import TaskBalancedRangesPool
from thinker_wm.data_prepare.task_cycled_ranges import install_exact_resume


@lru_cache(maxsize=8)
def consumption_plan(names, weights, seed, base, world, steps, episodes_per_batch):
    sources = [SimpleNamespace(name=n, weight=w) for n, w in zip(names, weights)]
    streams = [WeightedIndexedBatchStream(
        [None] * len(sources), sources, batches_per_epoch=steps, seed=seed,
        shuffle=True, rank=r, world_size=world,
        cross_rank_source_policy="weighted_per_step") for r in range(world)]
    for stream in streams:
        stream.epoch = base
    schedules = [iter(s.iter_source_schedule()) for s in streams]
    local_counts = [[0] * len(sources) for _ in range(world)]
    ordinals = [0] * len(sources)
    maps = [{} for _ in sources]
    rows = []
    for _ in range(steps):
        row = tuple(next(s) for s in schedules)
        rows.append(row)
        for rank, source in enumerate(row):
            for _ in range(episodes_per_batch):
                position = rank + world * local_counts[rank][source]
                maps[source][position] = ordinals[source]
                local_counts[rank][source] += 1
                ordinals[source] += 1
    return tuple(maps), tuple(rows)


class GlobalCycledRangesPool(TaskBalancedRangesPool):
    def __init__(self, path, repository, *, group_size, cycle_base_step,
                 source_name, source_names, source_weights, world_size,
                 schedule_steps, episodes_per_batch=2, **options):
        super().__init__(path, repository, group_size=group_size,
                         without_replacement=False, **options)
        self._cycle_base_step = int(cycle_base_step)
        self.world_size = int(world_size)
        self.schedule_steps = int(schedule_steps)
        self.episodes_per_batch = int(episodes_per_batch)
        if self.group_size != 4 or self.episodes_per_batch != 2 or self._stride_max != 1:
            raise SamplingRangesError("V16d requires 2 episodes x 4 windows, stride 1")
        if min(len(eps) for eps in self._task_episodes) < 2:
            raise SamplingRangesError("at least two episodes per task required")
        names, weights = tuple(source_names), tuple(float(x) for x in source_weights)
        maps, _ = consumption_plan(names, weights, self._seed,
                                    self._cycle_base_step, self.world_size,
                                    self.schedule_steps, self.episodes_per_batch)
        self._source_positions = maps[names.index(source_name)]
        # Enough groups even if one source were chosen by every rank every step.
        self._group_count = self.world_size * self.schedule_steps * self.episodes_per_batch
        self._length = self._group_count * self.group_size
        self._episode_groups = {
            f"draw:{g:09d}": tuple(range(g * self.group_size, (g + 1) * self.group_size))
            for g in range(self._group_count)}
        self._pool_id += "_global_chronological_2x4_v1"
        self._build_positions()

    def _build_positions(self):
        rng = random.Random(self._seed + self._epoch * 1_000_003)
        slots = list(range(self.group_size))
        for _ in range(self._group_count):
            rng.shuffle(slots)
        order = list(range(self._group_count))
        rng.shuffle(order)
        self._position_of_group = array("Q", [0]) * self._group_count
        for position, group in enumerate(order):
            self._position_of_group[group] = position

    def set_epoch(self, epoch):
        if int(epoch) not in (0, self._cycle_base_step):
            raise SamplingRangesError("global pool must stay in its branch base epoch")
        changed = int(epoch) != self._epoch
        super().set_epoch(epoch)
        if changed:
            self._build_positions()

    def _key(self, *parts):
        return compose_seed(self._seed, self._cycle_base_step, 0, 0,
                            self._pool_id + "|" + "|".join(map(str, parts[:-1])),
                            int(parts[-1]))

    def deal_at(self, ordinal):
        round_index, seat = divmod(int(ordinal), len(self._tasks))
        task = _feistel_permute(seat, len(self._tasks), self._key("task", round_index))
        episodes = self._task_episodes[task]
        use, slot = divmod(round_index, len(episodes))
        # A fixed shuffled episode deck per task also prevents repeats at the
        # boundary between two passes. Window locations change on each visit.
        episode = episodes[_feistel_permute(slot, len(episodes), self._key("episode", task))]
        return task, episode, use

    def _deal(self, group):
        position = int(self._position_of_group[group])
        try:
            ordinal = self._source_positions[position]
        except KeyError as exc:
            raise SamplingRangesError("draw is outside the configured global schedule") from exc
        return self.deal_at(ordinal)

    def _locate(self, index):
        if not 0 <= index < self._length:
            raise IndexError(index)
        group = index // self.group_size
        _, episode, _ = self._deal(group)
        return episode, group * self.group_size

    def _window_draw(self, index, offset):
        _, _, use = self._deal(int(index) // self.group_size)
        return use * self.group_size + int(index) - offset

    def _pick_start(self, episode, draw, rng):
        use, slot = divmod(int(draw), self.group_size)
        strata = 2 * self.group_size
        stratum = 2 * slot + use % 2
        lo = stratum * episode.total_starts // strata
        hi = (stratum + 1) * episode.total_starts // strata
        if hi <= lo:
            return episode.start_at(min(lo, episode.total_starts - 1))
        rotation, position = divmod(use // 2, hi - lo)
        key = self._key("window", episode.episode_id, stratum, rotation)
        return episode.start_at(lo + _feistel_permute(position, hi - lo, key))


class PairedSampler:
    """Expose optimizer-batch skip units to install_exact_resume."""
    def __init__(self, sampler):
        self.sampler = sampler

    @property
    def _resume_skip_batches(self):
        return int(getattr(self.sampler, "_resume_skip_batches", 0)) // 2

    @_resume_skip_batches.setter
    def _resume_skip_batches(self, count):
        self.sampler._resume_skip_batches = 2 * int(count)

    def __len__(self):
        return len(self.sampler) // 2


class PairedChild:
    def __init__(self, child):
        self.child = child
        self.batch_sampler = PairedSampler(child.batch_sampler)

    def __getattr__(self, name):
        return getattr(self.child, name)

    @property
    def _iterator(self):
        return self.child._iterator

    @_iterator.setter
    def _iterator(self, value):
        self.child._iterator = value

    def set_epoch(self, epoch):
        self.child.set_epoch(epoch)

    def next(self):
        from thinker_wm.data_prepare.factory import _merge_single_batches
        a, sa = self.child.next()
        b, sb = self.child.next()
        if set(a["episode_ids"]) & set(b["episode_ids"]):
            raise RuntimeError("paired batch repeated the same episode")
        batch = _merge_single_batches([a, b])
        if len(batch["episode_ids"]) != 8 or len(set(batch["episode_ids"])) != 2:
            raise RuntimeError("V16d batch is not two episodes x four windows")
        return batch, {"episode_batches": 2, "first": sa, "second": sb}

    def close(self):
        self.child.close()


def create_training_data_pipeline(config, **kwargs):
    from thinker_wm.data_prepare.factory import create_training_data_pipeline as original
    if int(config.training.batch_size) != 8:
        raise RuntimeError("V16d optimizer batch size must be 8 per rank")
    if int(kwargs.get("world_size", 1)) != int(config.dataset.sources[0].pool.world_size):
        raise RuntimeError("V16d rank count differs from its sampling plan")
    if config.dataset.cross_rank_source_policy != "weighted_per_step":
        raise RuntimeError("V16d requires weighted_per_step")
    c = copy.deepcopy(config)
    c.training.batch_size = 4
    pipeline = original(c, **kwargs)
    # Queue construction is lazy; replace children before it starts producing.
    paired = tuple(PairedChild(child) for child in pipeline.children)
    pipeline.children = paired
    pipeline.stream.children = paired
    return pipeline


def _build(repository, *, path, **options):
    return GlobalCycledRangesPool(path, repository, **options)


register_pool_builder("global_cycled_ranges", _build, known_options=(
    "path", "num_video_frames", "stride_max", "text_choice", "action_start_offset",
    "group_size", "cycle_base_step", "source_name", "source_names", "source_weights",
    "world_size", "schedule_steps", "episodes_per_batch"))
