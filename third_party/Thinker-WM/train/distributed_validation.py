"""Distributed full-window validation with finite, non-overlapping shards."""
import logging
import math
import json
import os
import tempfile
import shutil
import time
import torch
import torch.distributed as dist
from thinker_wm.train.sample import (evaluate_model, _validation_seed, _capture_rng_state,
                    _restore_rng_state, create_video_grid)

logger = logging.getLogger(__name__)

# RoboDojo v1 task text is the bare instruction. The data-pipeline VLM prefix
# is not what the model encodes, so identity uses the T5 task sentence.
_TEXT_TO_TASK = {
    "Insert the three tubes into the rack one by one.": "insert_tubes",
    "Pour the liquid from the bottle into the cup.": "pour_liquid_into_cup",
    "Stack the three bowls together.": "stack_bowls",
    "Plug the charger into the power strip.": "plug_in_charger",
    "Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.": "insert_key",
    "Fold the clothes neatly.": "fold_clothes",
}


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _catalog_tasks(config):
    result = {}
    dataset = getattr(config, "dataset", None)
    for source in getattr(dataset, "sources", []) or []:
        path = source.get("catalog_path")
        if not path:
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task = (row.get("meta") or {}).get("task_name")
                if task:
                    result[str(row["episode_id"])] = str(task)
    return result


def _window_identity(batch, catalog_tasks=None):
    episode = _first(batch.get("episode_ids"))
    if episode is None:
        episode = _first(batch.get("episode_id"))
    text = _first(batch.get("t5_input_messages"))
    if text is None:
        text = _first(batch.get("t5_input_message"))
    text = str(text or "").strip()
    episode = str(episode or "")
    if catalog_tasks is not None and episode.startswith("robodojo:") and episode not in catalog_tasks:
        raise ValueError(f"validation episode has no catalog task: {episode}")
    task = (catalog_tasks or {}).get(episode) or _first(batch.get("task_names"))
    return episode, text, str(task or _TEXT_TO_TASK.get(text, "") or "unknown")


def _gather(value):
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def _check_errors(payloads):
    errors = [f"rank {rank}: {item['error']}" for rank, item in enumerate(payloads)
              if item.get('error')]
    if errors:
        raise RuntimeError('distributed validation failed: ' + '; '.join(errors))


def _mean_rows(rows):
    keys = set(rows[0]['metrics'])
    if any(set(row['metrics']) != keys for row in rows):
        raise RuntimeError('inconsistent validation metric keys')
    result = {}
    count = len(rows)
    for key in sorted(keys):
        values = [row['metrics'][key] for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f'non-finite validation metric: {key}')
        mean = math.fsum(values) / count
        result[key] = mean
        result[key + '_std'] = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / count)
    result['num_samples'] = count
    return result


def aggregate_window_metrics(rows, total_samples):
    """Every row represents exactly one window; no mean-of-rank-means bias."""
    ids = [row['id'] for row in rows]
    if sorted(ids) != list(range(total_samples)):
        raise RuntimeError(f"validation coverage mismatch: {ids}")
    return _mean_rows(sorted(rows, key=lambda row: row['id']))


def _by_task(rows):
    groups = {}
    for row in rows:
        task = row.get('task') or ''
        if not task:
            continue
        groups.setdefault(task, []).append(row)
    return {task: _mean_rows(items) for task, items in sorted(groups.items())}


def _evaluate_distributed(model, pipeline, accelerator, config, history_attacher=None, staging_roots=None):
    """All ranks call this, including ranks with no remaining window.

    A collective after each single-window inference bounds normal idle waits
    to one inference, not the whole holdout. This is not a background heartbeat:
    a single hung/slow forward is still bounded by the explicit process-group
    timeout. Python failures are propagated before anybody starts the next round.
    """
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    system = getattr(config, 'system', None)
    # Media for every window needs the GT-action A2V clip and predicted depth.
    stage_media = bool(getattr(system, 'val_a2v', False)) and bool(getattr(system, 'val_depth', False))
    error = None
    sample_root = None
    try:
        plugin = getattr(getattr(accelerator, 'state', None), 'deepspeed_plugin', None)
        if plugin is not None and int(plugin.deepspeed_config.get(
                'zero_optimization', {}).get('stage', 0)) == 3:
            raise ValueError('distributed validation requires replicated weights (ZeRO-0/1/2)')
        pipeline.set_epoch(0)
        catalog_tasks = _catalog_tasks(config)
        if rank == 0 and stage_media:
            # Shared checkpoint filesystem: each rank writes its own windows'
            # media here; no video tensor crosses NCCL or reaches rank 0.
            base = getattr(system, 'checkpoint_dir', None)
            if base:
                base = os.path.join(base, 'validation_samples_tmp')
                os.makedirs(base, exist_ok=True)
            sample_root = tempfile.mkdtemp(prefix='holdout-', dir=base)
            if staging_roots is not None:
                staging_roots.append(sample_root)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    setup = _gather({'error': error, 'total': pipeline.total_samples,
                     'ids': pipeline.global_ids, 'sample_root': sample_root})
    _check_errors(setup)
    sample_root = setup[0]['sample_root']
    total = pipeline.total_samples
    if any(item['total'] != total for item in setup) or sorted(
            i for item in setup for i in item['ids']) != list(range(total)):
        raise RuntimeError('validation partitions overlap or omit windows')

    rows, saved_val = [], []
    preview = None  # one rank-0 window kept in memory for the TensorBoard grid
    for round_id in range((total + world_size - 1) // world_size):
        payload = {'error': None, 'row': None, 'entry': None}
        if round_id < len(pipeline.global_ids):
            sample_id = pipeline.global_ids[round_id]
            try:
                batch, _ = pipeline.next()
                if history_attacher is not None and 'history_recent_rgb' not in batch:
                    batch = history_attacher.attach(batch, step=0)
                episode, text, task = _window_identity(batch, catalog_tasks)
                keep = stage_media or (rank == 0 and preview is None)
                inference_model = accelerator.unwrap_model(model)
                started = time.perf_counter()
                metrics, samples = evaluate_model(
                    inference_model, [batch], accelerator, config, num_eval_batches=1,
                    seed=_validation_seed(config) + sample_id, expected_batch_size=1,
                    visualize=False, max_saved_samples=1 if keep else 0)
                inference_seconds = time.perf_counter() - started
                scalar_metrics = {key: float(value) for key, value in metrics.items()
                                  if isinstance(value, (int, float)) and not key.endswith('_std')}
                frames = _first(batch.get('frame_indices'))
                if hasattr(frames, 'tolist'):
                    frames = frames.tolist()
                payload['row'] = {'id': sample_id, 'episode_id': episode, 'task': task,
                                  'task_text': text, 'frame_indices': frames,
                                  'metrics': scalar_metrics}
                if keep:
                    if len(samples) != 1:
                        raise ValueError('validation window returned no saved sample')
                    sample = samples[0]
                    sample.update(task=task, task_name=task, episode_id=episode,
                                  task_text=text, window_id=sample_id, frame_indices=frames,
                                  metrics=scalar_metrics)
                    if rank == 0 and preview is None:
                        preview = sample
                    if stage_media:
                        from thinker_wm.periodic_eval.validation_artifacts import stage_window_media
                        entry = stage_window_media(sample, sample_root, rank)
                        entry['inference_seconds'] = inference_seconds
                        payload['entry'] = entry
                    del sample, samples
            except Exception as exc:
                payload['error'] = f'window {sample_id}: {type(exc).__name__}: {exc}'
                logger.exception('Validation window %s failed on rank %s', sample_id, rank)
        gathered = _gather(payload)
        _check_errors(gathered)
        rows.extend(item['row'] for item in gathered if item['row'] is not None)
        saved_val.extend(item['entry'] for item in gathered if item['entry'] is not None)
        if rank == 0:
            logger.info('Validation progress: %d/%d windows (%d ranks)', len(rows), total, world_size)
    if stage_media:
        if sorted(entry['window_id'] for entry in saved_val) != sorted(row['id'] for row in rows):
            raise RuntimeError('validation media does not cover every scored window')
        saved_val.sort(key=lambda entry: entry['window_id'])
        if rank == 0:
            logger.info('Validation media staged for %d windows: inference %.1fs, media %.1fs (sum over ranks)',
                        len(saved_val), math.fsum(e['inference_seconds'] for e in saved_val),
                        math.fsum(e['media_seconds'] for e in saved_val))
    if rank != 0:
        saved_val = []
    metrics = aggregate_window_metrics(rows, total)
    by_task = _by_task(rows)
    for task, stats in by_task.items():
        for key, value in stats.items():
            metrics[f'{key}__{task}'] = float(value)
    metrics['_by_task'] = by_task
    metrics['_windows'] = [
        {'id': row['id'], 'episode_id': row.get('episode_id', ''),
         'task': row.get('task', ''), 'frame_indices': row.get('frame_indices'), 'metrics': row['metrics']}
        for row in sorted(rows, key=lambda row: row['id'])
    ]
    if rank == 0 and preview is not None:
        metrics['visualization'] = create_video_grid(
            preview['predicted_frames'].unsqueeze(0), preview['gt_frames'].unsqueeze(0), num_samples=1)
    return metrics, saved_val


def evaluate_distributed(model, pipeline, accelerator, config, history_attacher=None):
    rng = _capture_rng_state()
    was_training = bool(model.training)
    staging_roots = []
    completed = False
    try:
        result = _evaluate_distributed(
            model, pipeline, accelerator, config, history_attacher, staging_roots)
        completed = True
        return result
    finally:
        # Rank 0 owns the staging root even if no entry has been returned yet.
        # Window failures are shared after every rank finishes its current write,
        # so deleting here cannot race another rank's normal media writer.
        # Successful runs transfer ownership to the publisher/caller instead.
        if not completed:
            for root in staging_roots:
                try:
                    shutil.rmtree(root)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception('Failed to remove validation staging directory %s', root)
        _restore_rng_state(rng)
        model.train(was_training)
