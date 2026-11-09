"""CLI: python -m thinker_wm.data_prepare.dexjoco --help."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .contracts import TASKS


def main():
    parser = argparse.ArgumentParser(description="DexJoCo six independent tasks on four H200 GPUs")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="validate manifest, prepare catalogs/ranges/statistics")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--tasks", nargs="+", default=[t.name for t in TASKS])
    prep.add_argument("--target-episodes", type=int, default=100)
    prep.add_argument("--no-dev", action="store_true", help="explicitly disable offline development validation")
    prep.add_argument("--allow-layout-overlap", action="store_true",
                      help="explicitly report that this is NOT a strictly unseen-layout split")
    configs = sub.add_parser("configs", help="write NEW selected-task configs with a user-chosen name prefix")
    for flag in ("release", "wan-dir", "vlm-dir", "init-checkpoint", "output-root", "config-dir"):
        configs.add_argument("--" + flag, type=Path, required=True)
    configs.add_argument("--name-prefix", required=True, help="user-approved configuration name")
    configs.add_argument("--batch-size", type=int, choices=(4, 8), default=8)
    configs.add_argument("--tasks", nargs="+", help="default: all six tasks")
    configs.add_argument("--seed", type=int, help="new seed for first selected task, incremented for subsequent tasks")
    setup = sub.add_parser("setup-task", help="prepare one LeRobot task and config, all episodes train; never train")
    setup.add_argument("--task", choices=[t.name for t in TASKS], required=True)
    setup.add_argument("--data", type=Path, required=True)
    setup.add_argument("--work-dir", type=Path, required=True)
    setup.add_argument("--name-prefix", required=True)
    setup.add_argument("--seed", type=int, required=True)
    setup.add_argument("--no-dev", action="store_true", required=True,
                       help="acknowledge no offline validation and no verified layout holdout")
    setup.add_argument("--variant", choices=("rand_obj", "rand_full"), default="rand_obj")
    setup.add_argument("--assets-from", type=Path, help="read only weight paths from this YAML; default: project base config")
    setup.add_argument("--target-episodes", type=int, default=100)
    setup.add_argument("--batch-size", type=int, choices=(4, 8), default=8)
    train = sub.add_parser("train", help="preview/check/execute ONE task, always four GPUs")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-name", required=True)
    train.add_argument("--smoke-steps", type=int, choices=(100, 200))
    train.add_argument("--resume", type=Path)
    train.add_argument("--max-steps", type=int, choices=(30000, 60000))
    train.add_argument("--check", action="store_true", help="validate source assets, dependencies and GPUs without training")
    train.add_argument("--execute", action="store_true", help="actually start torchrun; default only prints the plan")
    queue = sub.add_parser("queue", help="preview or execute all six tasks sequentially, stopping on failure")
    queue.add_argument("--configs", type=Path, nargs="+", required=True)
    queue.add_argument("--run-name", required=True)
    queue.add_argument("--smoke-steps", type=int, choices=(100, 200))
    queue.add_argument("--check", action="store_true")
    queue.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if command == "prepare":
        from .prepare import prepare_release
        args["names"] = args.pop("tasks")
        args["strict_layout"] = not args.pop("allow_layout_overlap")
        result = prepare_release(**args)
    elif command == "configs":
        from .recipe import write_configs
        directory, prefix = args.pop("config_dir"), args.pop("name_prefix")
        result = list(map(str, write_configs(directory, prefix, **args)))
    elif command == "setup-task":
        from .task_setup import setup_task
        if args["assets_from"] is None:
            args.pop("assets_from")
        result = setup_task(**args)
    elif command == "queue":
        from .launch import launch_queue
        result = launch_queue(**args)
    else:
        from .launch import launch
        result = launch(**args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
