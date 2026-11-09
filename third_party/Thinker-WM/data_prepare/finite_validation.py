"""Finite, unpadded holdout partition; each window is decoded by one rank."""
import json
from pathlib import Path

from .factory import (
    _indexed_sources, _mix_adapter_options, get_episode_dataset_release,
    open_dataset, resolve_sample_pool, SamplePoolDataset,
    build_indexed_batch_adapter, MOTUS_MODEL_BATCH_V1, IndexedEpisodeBatch,
)


def _task_balanced_indices(pool, episode_order: list[str], episode_tasks: dict[str, str],
                           tasks: list[str], quota: int) -> list[int]:
    """Fixed per-task quota, balanced across every holdout episode.

    Select from the existing deterministic candidate pool. Deduplicate actual
    windows, then spread picks over the candidates ordered by frame index.
    This does not change the training pool or the train/holdout split.
    """
    if quota <= 0 or not tasks or len(tasks) != len(set(tasks)):
        raise ValueError('validation task quota and task list must be positive/unique')
    if {episode_tasks[eid] for eid in episode_order} != set(tasks):
        raise ValueError('validation task list must match the complete holdout')
    selected = set()
    for task in tasks:
        episodes = [eid for eid in episode_order if episode_tasks[eid] == task]
        base_count, remainder = divmod(quota, len(episodes))
        if base_count == 0:
            raise ValueError('validation quota must cover every holdout episode')
        for position, eid in enumerate(episodes):
            count = base_count + int(position < remainder)
            candidates = {}
            for index in pool.episode_groups[eid]:
                selection = pool.generate(int(index))
                key = (tuple(selection['video_frame_indices']), tuple(selection['action_indices']))
                candidates.setdefault(key, int(index))
            ordered = [candidates[key] for key in sorted(candidates)]
            if count > len(ordered):
                raise ValueError(f'{eid}: need {count} distinct validation windows, have {len(ordered)}')
            # Integer bin midpoints select distinct candidates without RNG state.
            selected.update(ordered[((2 * j + 1) * len(ordered)) // (2 * count)]
                            for j in range(count))
    return [int(i) for eid in episode_order for i in pool.episode_groups[eid] if int(i) in selected]


class FiniteValidationPipeline:
    """Single-source holdout with optional fixed, balanced per-task window quotas.

    Do not use the training sampler here: it pads shards and its pipeline wraps
    into another epoch. Inference uses one window at a time, independently of
    the training batch size, without changing the selected windows.
    """

    def __init__(self, config, *, rank, world_size, seed):
        sources = _indexed_sources(config, eval=True)
        if len(sources) != 1 or not bool(config.system.val_full_set):
            raise ValueError("distributed validation requires one source and val_full_set=true")
        if not 0 <= rank < world_size:
            raise ValueError("invalid validation rank/world_size")
        source = sources[0]
        release = get_episode_dataset_release(source['dataset'], source.get('release'))
        self.repository = open_dataset(
            source['dataset'], release=release, catalog_root=source.get('catalog_root'),
            catalog_path=source.get('catalog_path'), index_path=source.get('index_path'),
            readers=source.get('readers'))
        try:
            pool = resolve_sample_pool(source['sample_pool'], self.repository, seed=seed)
            self.dataset = SamplePoolDataset(
                self.repository, pool, video_output_format=release.default_video_output_format,
                video_size=(int(config.common.video_height), int(config.common.video_width)),
                action_magnitude_guard_threshold=release.action_magnitude_guard_threshold,
                action_magnitude_guard_sources=release.action_magnitude_guard_sources)
            options = _mix_adapter_options(config)
            options.update(source.get('adapter_options') or {})
            self.adapter = build_indexed_batch_adapter(
                source.get('output_adapter') or MOTUS_MODEL_BATCH_V1, **options)
            episode_order = sorted(pool.episode_groups)
            order_path = getattr(config.system, 'val_episode_order_path', None)
            if order_path:
                episode_order = json.loads(Path(order_path).read_text())
                if (not isinstance(episode_order, list)
                        or not all(isinstance(eid, str) for eid in episode_order)
                        or len(episode_order) != len(set(episode_order))
                        or set(episode_order) != set(pool.episode_groups)):
                    raise ValueError('validation episode order must cover the holdout exactly once')
            # Without quotas, retain the legacy ordering and all candidate slots.
            indices = [int(i) for eid in episode_order
                       for i in pool.episode_groups[eid]]
            quota = getattr(config.system, 'val_windows_per_task', None)
            if quota is not None:
                catalog_path = source.get('catalog_path')
                if not catalog_path:
                    raise ValueError('task-balanced validation requires an explicit catalog_path')
                records = [json.loads(line) for line in Path(catalog_path).read_text().splitlines() if line.strip()]
                episode_tasks = {r['episode_id']: r['meta']['task_name'] for r in records}
                indices = _task_balanced_indices(
                    pool, episode_order, episode_tasks, list(config.system.val_tasks), int(quota))
            self.total_samples = len(indices)
            expected = int(config.system.val_num_batches) * int(config.training.batch_size)
            if self.total_samples != expected:
                raise ValueError(f"holdout has {self.total_samples} windows, expected {expected}")
            self.global_ids = list(range(rank, self.total_samples, world_size))
            self.indices = [indices[i] for i in self.global_ids]
            self.set_epoch(0)
        except Exception:
            self.repository.close()
            raise

    def set_epoch(self, epoch):
        if epoch != 0:
            raise ValueError("validation windows must stay at epoch 0")
        self.dataset.set_epoch(0)
        self.cursor = 0

    def next(self):
        if self.cursor >= len(self.indices):
            raise StopIteration
        samples = self.dataset.get_grouped_samples([self.indices[self.cursor]])
        batch = dict(self.adapter.adapt(IndexedEpisodeBatch(info=samples[0].info, samples=samples)))
        self.cursor += 1
        return batch, {}

    def close(self):
        self.repository.close()
