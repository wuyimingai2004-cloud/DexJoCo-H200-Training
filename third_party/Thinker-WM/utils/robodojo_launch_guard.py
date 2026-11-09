"""Preflight for a named Joint24 run; called while holding the node launch lock."""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path


def _config(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def guard_launch(config, output, checkpoints, node_rank, resume=None):
    config, output, checkpoints = Path(config), Path(output), Path(checkpoints).resolve()
    snapshot = output / 'config_at_launch.yaml'
    current = _config(config)
    if snapshot.exists() and _config(snapshot) != current:
        raise ValueError('configuration differs from config_at_launch.yaml; use a new RUN_NAME')
    candidates = list(checkpoints.glob('checkpoint_step_*'))
    if resume:
        selected = Path(resume).resolve(strict=True)
        match = re.fullmatch(r'checkpoint_step_(\d+)(?:_[A-Za-z0-9_-]+)?', selected.name)
        if selected.parent != checkpoints or not match:
            raise ValueError('RESUME_CHECKPOINT must be a checkpoint directly inside this run')
        receipt = json.loads((selected / '.motus_checkpoint_complete.json').read_text())
        step = int(match.group(1))
        if receipt.get('status') != 'complete' or type(receipt.get('step')) is not int or receipt['step'] != step or step <= 0:
            raise ValueError('invalid checkpoint completion receipt')
        if not snapshot.is_file():
            raise ValueError('cannot verify resume configuration: config_at_launch.yaml is missing')
        for candidate in candidates:
            later = re.fullmatch(r'checkpoint_step_(\d+)(?:_[A-Za-z0-9_-]+)?', candidate.name)
            if later and int(later.group(1)) > step and (candidate / '.motus_checkpoint_complete.json').exists():
                raise ValueError('a newer completed checkpoint exists; refusing to rewind and overwrite it')
        if current.get('resume', {}).get('reset_scheduler', False):
            raise ValueError('resume must preserve the scheduler')
    elif candidates or (output / f'train_node{node_rank}.log').exists() or (output / f'launch_node{node_rank}.json').exists():
        raise ValueError('run already has training state; set RESUME_CHECKPOINT or use a new RUN_NAME')

    output.mkdir(parents=True, exist_ok=True)
    # Publish complete bytes atomically, without overwriting another node's
    # snapshot. A second node may join this fresh run after rank 0 has started.
    if not snapshot.exists():
        fd, temporary = tempfile.mkstemp(prefix='.config-', dir=output)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(config.read_bytes())
            try:
                os.link(temporary, snapshot)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        if _config(snapshot) != current:
            raise ValueError('nodes launched with different configurations')
    (output / f'launch_node{node_rank}.json').write_text(json.dumps(
        {'resume_checkpoint': str(Path(resume).resolve()) if resume else None,
         'config': str(config.resolve())}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--node-rank', type=int, required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    try:
        guard_launch(args.config, args.output, args.checkpoints, args.node_rank, args.resume)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
