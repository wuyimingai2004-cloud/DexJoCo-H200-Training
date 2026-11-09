#!/usr/bin/env python3
"""Request save-and-stop for one run, without terminating torchrun workers."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Exact run checkpoint directory, not the dataset/output root")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or not (run_dir / "step_timing").is_dir():
        parser.error("run directory must exist and contain step_timing/; wait for trainer initialization")
    request = run_dir / ".stop_requested"
    try:
        with request.open("x") as f:
            json.dump({"requested_at": datetime.now().isoformat(timespec="seconds")}, f)
            f.write("\n")
    except FileExistsError:
        print(f"Stop already requested: {request}")
    else:
        print(f"Stop requested: {request}")
    print("The trainer will finish the current step (including any evaluation already running),")
    print("save a complete checkpoint on all ranks, then exit. This command does not wait.")
    print(f"Confirm the new receipt and process exit: {run_dir / 'stop_completed.json'}")


if __name__ == "__main__":
    main()
