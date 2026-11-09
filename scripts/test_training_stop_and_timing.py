"""CPU/Gloo integration of the actual trainer loop with a tiny optimizer.

Extract only train()/save_checkpoint() to avoid importing the 5B model stack.
Data/model work is tiny; loop control, collective saving and timing are real.
"""
import ast
from collections import Counter
from datetime import datetime, timedelta
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thinker_wm.utils.training_control import (
    CHECKPOINT_MARKER, STOP_REQUEST, STOP_RECEIPT, TrainingStopRequest,
    step_timing_breakdown, format_step_timing,
)


def distributed_mean(values, device):
    tensor = torch.tensor(values, device=device)
    dist.all_reduce(tensor)
    return (tensor / dist.get_world_size()).tolist()


def validation_error(error):
    payload = [error]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def trainer_class():
    tree = ast.parse((ROOT / 'third_party/Thinker-WM/train/train.py').read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UniDiffuserTrainer')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ('train', 'save_checkpoint')]
    node = ast.ClassDef(name='Trainer', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(globals())
    namespace.update(
        logger=logging.getLogger('loop-test'), CHECKPOINT_COMPLETE_MARKER=CHECKPOINT_MARKER,
        TimingStack=None, write_startup_timing=lambda *a: None,
        _monitor_call=lambda *a, **k: None, _tensor_debug_stats=lambda *a: {},
        _memory_snapshot=lambda *a: {}, _list_head=lambda x: x,
        _distributed_mean=distributed_mean, _nonfinite_fail_fast_enabled=lambda: True,
        _pipeline_batch_iter=lambda *a: iter(()), log_evaluation_metrics=lambda *a: None,
        share_validation_error=validation_error,
        _write_validation_record=lambda *a, **k: 'test-validation-record',
    )
    exec(compile(module, str(ROOT / 'third_party/Thinker-WM/train/train.py'), 'exec'), namespace)
    return namespace['Trainer'], namespace


class Writer:
    def __init__(self):
        self.records = []
        self.scalars = {}
    def write(self, record):
        self.records.append(dict(record))
    def add_scalar(self, key, value, step):
        self.scalars[(step, key)] = value
    def close(self):
        pass


class Pipeline:
    queue_enabled = False
    def next(self):
        time.sleep(.003)
        return {'source_datasets': ['test'], 'first_frame': torch.ones(1, 3, 2, 2),
                'video_frames': torch.ones(1, 4, 3, 2, 2),
                'action_sequence': torch.ones(1, 8, 14)}, {}
    def snapshot(self):
        return {}
    def set_epoch(self, epoch):
        pass
    def close(self):
        pass


def run_case(rank, base, case):
    run = base / case
    if rank == 0:
        (run / 'step_timing').mkdir(parents=True)
    dist.barrier()
    Trainer, namespace = trainer_class()
    trainer = Trainer.__new__(Trainer)
    trainer.rank, trainer.world_size, trainer.device = rank, 2, torch.device('cpu')
    trainer.global_step, trainer.checkpoint_dir = 0, run
    trainer.log_interval, trainer.save_interval = 10, 100
    if case in ('interval_stop', 'checkpoint_stop', 'validation_failure', 'distributed_validation_failure'):
        trainer.save_interval = 2
    trainer.startup_start_time, trainer.startup_timing_events = None, []
    trainer.config = OmegaConf.create({'training': {'train_steps_per_vae_encode': 1},
                                      'system': {'checkpoint_dir': str(run), 'val_num_batches': 1}})
    trainer.monitor, trainer.monitor_metrics_interval = None, 0
    trainer.train_dataloader = Pipeline()
    trainer.val_dataloader = Pipeline() if case in ('eval_stop', 'validation_failure') else None
    trainer.periodic_evaluator = SimpleNamespace(maybe_run=lambda *a: {}, close=lambda: None)
    trainer.heartbeat_memory_interval, trainer.train_loop_gc_interval = 0, 1
    trainer.step_timing_enabled, trainer.step_timing_interval, trainer.step_timing_sync_cuda = False, 10, False
    trainer.source_dataset_counts = Counter()
    trainer.report_to = []
    trainer.tb_writer = Writer()
    trainer.step_timing_writer, trainer.action_diagnostics_writer, trainer.heartbeat_writer = Writer(), Writer(), Writer()
    trainer._write_heartbeat = lambda *a, **k: None
    trainer._maybe_simulate_elastic_failure = lambda: None
    trainer._maybe_write_action_diagnostics = lambda *a: None
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.01)
    if case in ('distributed_eval_stop', 'distributed_validation_failure'):
        sys.path.insert(0, str(ROOT / 'third_party/Thinker-WM/train'))
        from test_distributed_validation import Pipeline as EvalPipeline, config as eval_config
        import distributed_validation
        distributed_validation.create_video_grid = lambda *a, **kw: None
        trainer.val_dataloader = EvalPipeline(rank, 2, 3)
        trainer.config.model = eval_config().model
        trainer.config.common = eval_config().common
        trainer.config.training.batch_size = 8
        trainer.config.system.val_distributed = True
        trainer.model.config = SimpleNamespace(enable_depth=False)
        def inference_step(**kwargs):
            time.sleep(.02)
            if rank == 1 and case == 'distributed_validation_failure':
                raise RuntimeError('injected validation failure on rank 1')
            if rank == 1:
                (run / STOP_REQUEST).write_text('stop during distributed validation')
            return torch.zeros(1, 3, 12, 2, 2), torch.zeros(1, 24, 14)
        trainer.model.inference_step = inference_step

    saves = []

    def prepare(*args, **kwargs):
        time.sleep(.006)
        return {}

    def update(*args, **kwargs):
        trainer.optimizer.zero_grad()
        loss = trainer.model(torch.ones(1, 1)).square().sum()
        loss.backward()
        trainer.optimizer.step()
        # A request originates on rank 1, while only rank 0 polls shared storage.
        if rank == 1 and trainer.global_step == 1 and case in ('stop', 'interval_stop', 'save_failure'):
            (run / STOP_REQUEST).write_text('request')
        return {'total_loss': float(loss), 'video_loss': 0., 'action_loss': float(loss)}

    def save_state(directory):
        saves.append(trainer.global_step)
        if case == 'save_failure':
            raise OSError('injected save failure on all ranks')
        time.sleep(.015)
        torch.save({'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict()},
                   Path(directory) / f'rank{rank}.pt')
        if rank == 1 and case == 'checkpoint_stop':
            (run / STOP_REQUEST).write_text('request during checkpoint')

    def evaluate(*args, **kwargs):
        time.sleep(.02)
        if case == 'validation_failure':
            raise RuntimeError('injected validation failure')
        (run / STOP_REQUEST).write_text('request during validation')
        return {}, []

    namespace['evaluate_model'] = evaluate
    trainer.prepare_train_batch, trainer.train_step = prepare, update
    trainer.accelerator = SimpleNamespace(is_main_process=rank == 0,
                                         wait_for_everyone=dist.barrier, save_state=save_state,
                                         process_index=rank, num_processes=2, unwrap_model=lambda m: m)
    try:
        trainer.train(max_steps=3, val_interval=2)
    except RuntimeError as exc:
        assert case in ('validation_failure', 'distributed_validation_failure'), str(exc)
        assert 'injected validation failure' in str(exc)
        assert saves == [2]
        assert (run / 'checkpoint_step_2' / CHECKPOINT_MARKER).exists()
        record = trainer.step_timing_writer.records[-1]
        assert record['wall_validation_seconds'] >= .02
        assert record['wall_checkpoint_seconds'] >= .015
        dist.barrier()
        if rank == 0:
            print('PASS validation failure keeps checkpoint and reports full failed-step timing', flush=True)
        return
    except OSError:
        assert case == 'save_failure'
        assert (run / STOP_REQUEST).exists()
        assert not (run / STOP_RECEIPT).exists()
        assert not (run / 'checkpoint_step_2' / CHECKPOINT_MARKER).exists()
        dist.barrier()
        if rank == 0:
            print('PASS failed checkpoint leaves stop request and no success receipt', flush=True)
        return
    assert case != 'save_failure'
    expected_step = 3 if case == 'normal' else 2
    assert trainer.global_step == expected_step, (case, trainer.global_step)
    assert saves == [expected_step], (case, saves)
    checkpoint = run / f'checkpoint_step_{expected_step}'
    assert (checkpoint / CHECKPOINT_MARKER).exists()
    state = torch.load(checkpoint / f'rank{rank}.pt', weights_only=True)
    clone = torch.nn.Linear(1, 1)
    clone.load_state_dict(state['model'])
    restored_optimizer = torch.optim.AdamW(clone.parameters(), lr=.01)
    restored_optimizer.load_state_dict(state['optimizer'])
    assert len(restored_optimizer.state) == len(trainer.optimizer.state)
    for x, y in zip(clone.parameters(), trainer.model.parameters()):
        assert torch.equal(x, y)
    if case != 'normal':
        assert not (run / STOP_REQUEST).exists()
        assert json.loads((run / STOP_RECEIPT).read_text())['step'] == expected_step
    record = trainer.step_timing_writer.records[-1]
    assert record['checkpoint_total_seconds'] >= .015
    assert record['wall_prepare_seconds'] >= .006
    assert record['wall_data_seconds'] >= .003
    assert record['wall_total_seconds'] >= record['wall_checkpoint_seconds'] + record['wall_prepare_seconds']
    if case in ('eval_stop', 'distributed_eval_stop'):
        assert record['wall_validation_seconds'] >= .02
    if rank == 0:
        assert trainer.tb_writer.scalars[(expected_step, 'train/step_time')] == record['wall_total_seconds']
        assert trainer.tb_writer.scalars[(expected_step, 'timing/checkpoint_seconds')] >= .015
        print(f'PASS actual loop {case}: both ranks save step {expected_step} exactly once; reload model+optimizer; full timing', flush=True)
    dist.barrier()


def worker(rank, base):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{base}/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    # Existing production validation uses current_device for a barrier; Gloo
    # ignores device_ids and this test never initializes a CUDA context.
    torch.cuda.current_device = lambda: 0
    try:
        for case in ('normal', 'stop', 'interval_stop', 'eval_stop', 'checkpoint_stop', 'save_failure', 'validation_failure', 'distributed_eval_stop', 'distributed_validation_failure'):
            run_case(rank, Path(base), case)
    finally:
        dist.destroy_process_group()


def main():
    timing = step_timing_breakdown({'dataloader_next_seconds': 5., 'prepare_train_batch_seconds': 4.,
                                   'step_time_seconds': 10., 'validation_seconds': 20.,
                                   'checkpoint_total_seconds': 30., 'loop_total_seconds': 66.})
    assert timing['data_seconds'] == 1. and timing['other_seconds'] == 1.
    assert sum(v for k, v in timing.items() if k != 'total_seconds') == 66.
    assert 'Total: 66.000s' in format_step_timing(timing)
    with tempfile.TemporaryDirectory() as temp:
        run = Path(temp) / 'cli'
        (run / 'step_timing').mkdir(parents=True)
        cmd = [sys.executable, str(ROOT / 'scripts/request_training_stop.py'), '--run-dir', str(run)]
        subprocess.run(cmd, check=True, capture_output=True)
        first = (run / STOP_REQUEST).read_bytes()
        subprocess.run(cmd, check=True, capture_output=True)
        assert (run / STOP_REQUEST).read_bytes() == first
        try:
            TrainingStopRequest(run).complete(2, run / 'missing_checkpoint')
        except RuntimeError:
            pass
        else:
            raise AssertionError('incomplete checkpoint accepted')
        assert (run / STOP_REQUEST).exists()
        print('PASS timing excludes duplicate preparation; CLI idempotent; incomplete checkpoint rejected', flush=True)
        mp.spawn(worker, args=(temp,), nprocs=2, join=True)


if __name__ == '__main__':
    main()
