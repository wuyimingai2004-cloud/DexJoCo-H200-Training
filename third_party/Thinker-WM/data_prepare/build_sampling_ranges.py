#!/usr/bin/env python3
"""从 Episode Catalog 构建每 Episode 采样范围 sidecar（ranges sqlite）。

窗口域规则与历史枚举器一致（agibot ratio 取 metadata、下界取
effective_video_frames；文本 agibot/agibot_beta 需 episode_context +
覆盖 start 的 action_segment，其余优先 instruction_en）；无有效窗口/文本的
episode 被剔除。num_video_frames 与 ratio、layout 烘焙进区间，改动需重建。

布局（--video-stride 开启）：缺省 consecutive（视频连续 N+1 帧、动作按 ratio
放大）；video_stride 为旧时代 1:6 语义（动作连续 N*ratio 帧、视频每 ratio 帧
抽 1 帧；短 episode 保留 s=0 单起点并末帧填充，clamp 上限以 episode_len 列
烘焙进库）。

用法（micro ranges，48 帧、ratio 默认）：
    python data_prepare/build_sampling_ranges.py \
        --catalog micro_data/catalogs/robotwin_aloha.episode_catalog.jsonl \
        --output micro_data/catalogs/robotwin_aloha.ranges.sqlite \
        --num-video-frames 48 [--force]

旧时代 1:6（48 帧、ratio 6、stride + 末帧填充）：
    python data_prepare/build_sampling_ranges.py \
        --catalog ... --dataset robotwin_aloha --num-video-frames 48 \
        --action-fps-ratio 6 --video-stride --output ... [--force]

train/holdout（同一 catalog 两次调用，split 划分确定性一致）：
    python data_prepare/build_sampling_ranges.py ... --split train \
        --holdout-fraction 0.01 --output .../xx.train.ranges.sqlite
    python data_prepare/build_sampling_ranges.py ... --split holdout \
        --holdout-fraction 0.01 --output .../xx.holdout.ranges.sqlite

窗口级过滤（静止区段剔除 + 离群窗口剔除；consecutive 与 video_stride 布局
均支持）：
    python data_prepare/build_sampling_ranges.py ... --split train \
        --holdout-fraction 0.01 \
        --window-static-eps 0.05 --window-max-static-frames 12 \
        --window-outlier-threshold 50 \
        --workers 32 --output .../xx.train.ranges.sqlite

纯视频源（无 state/action，如 openvid；自动识别，metadata 记
state_action_policy=absent，不支持窗口过滤；20260813 起支持
--video-stride/--max-stride 变倍率抽帧）：
    python data_prepare/build_sampling_ranges.py ... --output .../xx.ranges.sqlite

建库期最小时长过滤（20260818 起；有效长度低于阈值的 episode 整条剔除，
不再保留 video_stride 短 episode 的 s=0 末帧填充退化窗口）：
    python data_prepare/build_sampling_ranges.py ... --min-episode-frames 289

冒烟抽样（只扫描 catalog 前 N 条 episode）：
    python data_prepare/build_sampling_ranges.py ... --max-episodes 200
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

from motus_data.releases import get_episode_dataset_release  # noqa: E402
from motus_data.repository import EpisodeRepository  # noqa: E402

from thinker_wm.data_prepare.sampling_ranges import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    LAYOUT_VIDEO_STRIDE,
    build_sampling_ranges,
)
from thinker_wm.data_prepare import robotwin_3d as _robotwin_3d  # noqa: E402,F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Episode Catalog JSONL 路径")
    parser.add_argument(
        "--index",
        default=None,
        help="SQLite 偏移索引路径（默认取 <catalog 去掉 .jsonl>.index.sqlite）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="ranges sqlite 输出路径（默认取 <catalog 去掉 .jsonl>.ranges.sqlite）",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="源数据集名（仅用于 agibot 特例与 metadata；默认取 catalog 文件名 stem）",
    )
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument(
        "--action-fps-ratio",
        type=int,
        default=None,
        help="动作/视频频率比（默认 agibot 取 episode metadata，其余为 1）",
    )
    parser.add_argument(
        "--video-stride",
        action="store_true",
        help="旧时代 1:6 布局：动作连续 N*ratio 帧、视频每 ratio 帧抽 1 帧"
        "（短 episode 末帧填充）；缺省为 consecutive 布局",
    )
    parser.add_argument(
        "--action-start-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="动作窗口相对条件 state 的起点：0=action[t]（RoboTwin/LeRobot），"
        "1=旧行为 action[t+1]；写入 ranges metadata，缺省 1",
    )
    parser.add_argument(
        "--action-guard-threshold",
        type=float,
        default=None,
        help="episode 级动作/状态离群值过滤阈值（与 releases.yaml 运行时 guard"
        " 同语义）：abs_max 超阈值或含非有限值的 episode 整条剔除；缺省不过滤",
    )
    parser.add_argument(
        "--window-static-eps",
        type=float,
        default=None,
        help="窗口级静止过滤（两道规则叠加）：静止帧 = 帧内动作步逐维最大变化"
        " < eps（归一化单位）——1) 连续静止 >= --window-max-static-frames 视频帧的"
        "区段从窗口起点域剔除（孤立静止帧保留）；2) 窗口末帧 s+N 位于长度 >= K 的"
        "静止 run 内（窗口末尾连续静止 >= K 帧）的起点剔除。须与"
        " --window-max-static-frames 成对设置，缺省不过滤",
    )
    parser.add_argument(
        "--window-max-static-frames",
        type=int,
        default=None,
        help="连续静止帧数下限（视频帧，经 ratio 换算动作步）；与 --window-static-eps 成对",
    )
    parser.add_argument(
        "--window-outlier-threshold",
        type=float,
        default=None,
        help="窗口级离群过滤：窗口动作/状态幅值超阈值则剔除该窗口起点；缺省不过滤",
    )
    parser.add_argument(
        "--max-stride",
        type=int,
        default=None,
        help="每 episode 时长倍率上限的封顶值（video_stride 布局变 FPS 训练用）："
        "episode 级 max_stride 由过滤后区间预计算写入 max_stride 列，本参数仅封顶；"
        "缺省不封顶",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=None,
        help="建库期最小时长过滤（原生帧）：episode 有效长度（纯视频源取 "
        "video_count，否则取 min(video, state, action)）低于阈值的 episode 整条"
        "剔除（short_rejected 计数），不再保留 video_stride 布局的 s=0 末帧填充"
        "退化窗口；缺省不过滤",
    )
    parser.add_argument(
        "--tail-pad-steps",
        type=int,
        default=0,
        help="video_stride 布局：起点上限再往后放 N 个原生步，越界部分由读取侧"
        "clamp 到末帧/末动作（学“完成后保持不动”）；缺省 0 = 旧行为",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="行计算并行进程数（>1 时按 episode 多进程扫描，map 保序、"
        "sqlite 单线程写；大 catalog + guard 过滤时显著加速）",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "holdout"),
        default="all",
        help="episode 划分（train/holdout 需配合 --holdout-fraction）",
    )
    parser.add_argument("--holdout-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="只扫描 catalog 前 N 条 episode（split 划分在其后应用）；"
        "冒烟/小规模验证用，缺省全量",
    )
    parser.add_argument(
        "--all-text-indices",
        action="store_true",
        help="非分段文本族 episode 记录全部有效英文指令候选（text_payload "
        "kind=choice_one），运行时由 pool 的 text_choice 选恰好一条；缺省只记"
        "第一条（kind=fixed，旧行为，逐位可复现）",
    )
    parser.add_argument(
        "--robodojo-field-map",
        default=None,
        help="robodojo 源 catalog：注册 RoboDojo reader 并按该 field map 读取"
        "（须与建 catalog 时同一份）；其余源不需要",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    catalog = Path(args.catalog)
    name = catalog.name
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else name
    if args.index:
        index = Path(args.index)
    else:
        # 约定：<catalog 去掉末尾 .jsonl>.index.sqlite
        index = catalog.with_name(f"{stem}.index.sqlite")
    if not index.is_file():
        raise FileNotFoundError(f"catalog offset index not found: {index}")
    if args.output:
        output = Path(args.output)
    else:
        # 约定：<catalog 去掉末尾 .jsonl>.ranges.sqlite
        output = catalog.with_name(f"{stem}.ranges.sqlite")
    dataset = args.dataset or catalog.name.split(".")[0]

    # reader_options 解析（20260812）：profile/归一化绑定的 catalog（agibot_beta
    # effector profile、libero 等）的 reader 在 bind_catalog 时强制要求 release
    # reader_options（归一化 artifacts / 统计文件）；窗口过滤读 state/action
    # 也必须拿到归一化后的值，eps 阈值才与运行时可以比较。按 catalog meta 的
    # child → --dataset → 文件名 stem 顺序查 release 注册表；查不到保持 None
    # （旧行为，无绑定要求的源不受影响）。
    reader_options = None
    meta_path = catalog.with_name(f"{stem}.meta.json")
    candidates: list[str] = []
    if meta_path.is_file():
        try:
            import json as _json

            child = (_json.loads(meta_path.read_text(encoding="utf-8")) or {}).get(
                "child"
            )
            if child:
                candidates.append(str(child))
        except Exception:
            pass
    candidates.extend([str(dataset), catalog.name.split(".")[0]])
    for candidate in candidates:
        try:
            reader_options = get_episode_dataset_release(candidate).reader_options
            break
        except Exception:
            continue

    if args.robodojo_field_map:
        from thinker_wm.data_prepare import robodojo as _robodojo  # noqa: F401  注册 reader

        reader_options = {"robodojo": {"field_map": str(args.robodojo_field_map)}}

    repository = EpisodeRepository(
        catalog,
        index_path=index,
        verify_sha256=False,
        reader_options=reader_options,
    )
    try:
        stats = build_sampling_ranges(
            repository,
            output,
            dataset=dataset,
            num_video_frames=int(args.num_video_frames),
            action_fps_ratio=args.action_fps_ratio,
            layout=(
                LAYOUT_VIDEO_STRIDE if args.video_stride else "consecutive"
            ),
            action_guard_threshold=args.action_guard_threshold,
            window_static_eps=args.window_static_eps,
            window_max_static_frames=args.window_max_static_frames,
            window_outlier_threshold=args.window_outlier_threshold,
            max_stride=args.max_stride,
            min_episode_frames=args.min_episode_frames,
            all_text_candidates=bool(args.all_text_indices),
            action_start_offset=int(args.action_start_offset),
            tail_pad_steps=int(args.tail_pad_steps),
            workers=int(args.workers),
            split=str(args.split),
            holdout_fraction=args.holdout_fraction,
            split_seed=int(args.split_seed),
            max_episodes=args.max_episodes,
            force=bool(args.force),
            reader_options=reader_options,
        )
        filter_stats = (
            f"static_rejected_starts={stats['static_rejected_starts']} "
            f"tail_rejected_starts={stats['tail_rejected_starts']} "
            f"outlier_rejected_starts={stats['outlier_rejected_starts']} "
            f"filtered_episodes={stats['filtered_episodes']} "
            f"short_rejected={stats['short_rejected']} "
        )
        print(
            f"RANGES_OK dataset={stats['dataset']} catalog_id={stats['catalog_id']} "
            f"episodes={stats['episodes']}/{len(repository)} "
            f"intervals={stats['intervals']} valid_starts={stats['valid_starts']} "
            f"ratio={stats['ratio']} layout={stats['layout']} "
            f"padded_episodes={stats['padded_episodes']} "
            f"guard_rejected={stats['guard_rejected']} "
            f"{filter_stats} split={stats['split']} output={stats['output']}",
            flush=True,
        )
    finally:
        repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
