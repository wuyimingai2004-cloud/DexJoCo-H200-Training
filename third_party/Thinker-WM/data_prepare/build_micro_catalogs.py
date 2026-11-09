#!/usr/bin/env python3
"""从 micro_data 的 manifest 批量构建本地 Episode Catalog（robotwin×4 + robotmind×4）。

agibot / agibot_beta 不在本地构建（micro_data 缺 new_task_info 标注 / 48GB tar），
这两源由 H 集群的冒烟测试覆盖。

用法：
    python data_prepare/build_micro_catalogs.py \
        --micro-data-dir micro_data --output-dir micro_data/catalogs [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thinker_wm.data_prepare import ensure_motus_data_importable  # noqa: E402

ensure_motus_data_importable()

from motus_data.catalog.source_families import (  # noqa: E402
    build_source_family_episode_catalog,
)
from motus_data.catalog.sqlite_index import build_catalog_offset_index  # noqa: E402


MICRO_SOURCES = (
    ("robotwin_aloha", "robotwin"),
    ("robotwin_ur5", "robotwin"),
    ("robotwin_arx", "robotwin"),
    ("robotwin_piper", "robotwin"),
    ("robotmind_tianyi", "robotmind"),
    ("robotmind_arkmobile", "robotmind"),
    ("robotmind_ark", "robotmind"),
    ("robotmind_tienkung", "robotmind"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-data-dir", default="micro_data")
    parser.add_argument("--output-dir", default="micro_data/catalogs")
    parser.add_argument(
        "--no-verify-source",
        action="store_true",
        help="跳过源文件校验（默认校验 mp4 帧数 / qpos / hdf5 键）",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    micro_data_dir = Path(args.micro_data_dir)
    output_dir = Path(args.output_dir)
    for name, family in MICRO_SOURCES:
        # RobotMind 源优先用窗口对齐后的 manifest（align_robotmind_micro_windows.py 产物）
        windowed = micro_data_dir / f"{name}_manifest.micro_windowed.jsonl"
        manifest = micro_data_dir / f"{name}_manifest.micro.jsonl"
        if family == "robotmind" and windowed.is_file():
            manifest = windowed
        if not manifest.is_file():
            print(f"SKIP {name}: manifest not found: {manifest}", flush=True)
            continue
        output = output_dir / f"{name}.episode_catalog.jsonl"
        result = build_source_family_episode_catalog(
            manifest_path=manifest,
            output_path=output,
            family=family,
            source_child=name,
            catalog_id=f"{name}_micro_v1",
            data_version="micro",
            verify_source=not args.no_verify_source,
            max_episodes=args.max_episodes,
            force=args.force,
        )
        index_result = build_catalog_offset_index(output, force=args.force)
        print(
            f"CATALOG_OK {name} family={family} episodes={result['episodes']} "
            f"output={output} index={index_result['index_path']}",
            flush=True,
        )
    print("MICRO_CATALOGS_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
