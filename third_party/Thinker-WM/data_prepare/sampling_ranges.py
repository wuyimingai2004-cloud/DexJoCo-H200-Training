"""按 Episode 采样范围（SQLite sidecar）+ 内存动态随机索引的 SamplePool 策略。

这是 repo 自有的索引选择策略（经 motus_data 的 SupportsSamplePool 注入点进入
管线；motus_data 库只负责统一格式/加载/打包/传输，不定义任何选择策略）。

Ranges sidecar 格式（约定路径 <catalog 去掉 .jsonl>.ranges.sqlite）：

    metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)
        format=motus_sampling_ranges_v1, catalog_id, dataset, num_video_frames,
        action_video_fps_ratio(统一值或 mixed), action_start_offset(0|1),
        episodes, split 信息, created_at,
        state_action_policy(real|absent；absent=纯视频源，state/action 索引为空)
    episode_ranges(
        ordinal INTEGER PRIMARY KEY,
        episode_id TEXT NOT NULL UNIQUE,
        ratio INTEGER NOT NULL,
        intervals TEXT NOT NULL,    -- JSON [[lo,hi],...] 闭区间：有效窗口起点域
        text_payload TEXT NOT NULL, -- JSON {"kind":"fixed","indices":[i]} |
                                    --      {"kind":"choice_one","indices":[i,...]}（V14 多候选，运行时选一）|
                                    --      {"kind":"agibot","context":i,"segments":[[idx,lo,hi],...]}（旧库） |
                                    --      {"kind":"segmented","context":[i,...],"segments":[[idx,lo,hi],...]}
                                    -- 注意 fixed 的多条 indices 语义是「一起用」
                                    -- （会被 _compose_t5_text 拼成一条 prompt）；
                                    -- 「多候选选一」必须用 choice_one，两者不可混。
        episode_len INTEGER         -- stride 布局的末帧填充上限（仅新建库；
                                    --   旧库无此列，加载时按 None 处理不 clamp）
        max_stride INTEGER          -- 该 episode 支持的最大时长倍率 k（变 FPS
                                    --   训练用，建库时由过滤后区间预计算；旧库
                                    --   无此列，加载时按 1 处理）
    )

- 窗口域规则与原 build_sample_pool_enumerate.py 一致（agibot ratio 取 episode
  metadata、下界取 effective_video_frames）。文本：分段文本族
  （_SEGMENTED_TEXT_FAMILIES：agibot/agibot_beta/egoverse/egoscale）随窗口
  选 context（episode 级文本，可为空）+ 与窗口 [start, start+span] 有正重叠
  的全部片段（按 lo 升序；span=video_stride? N*ratio : N），intervals 取片段
  覆盖并集——不被任何片段覆盖的起点剔除；其余源优先 instruction_en、否则首条
  可用文本（fixed，与窗口无关）。20260819 前构建的 agibot 旧库（"agibot"
  payload：恰好一个覆盖 start 的 action_segment）解析路径保留不变。
- 布局（metadata layout 键，缺省 consecutive）：
  - consecutive：视频连续 N+1 帧，state=s*ratio；动作起点由
    action_start_offset 控制（0 从 state 同索引开始，1 为旧的下一索引）。
  - video_stride（旧时代 1:6 语义）：窗口 = N*ratio 个连续动作帧
    offset=1 时为 [s+1, s+N*ratio]，offset=0 时为
    [s, s+N*ratio-1]；视频每 ratio 帧抽 1 帧 [s, s+ratio, ..., s+N*ratio]，
    state=s；视频与动作覆盖同一时间跨度。短 episode（L < N*ratio+2，L≥2）
    保留 s=0 单起点、索引 clamp 到 L-1（末帧填充）；L<2 剔除。纯视频源
    （absent）20260813 起支持本布局：窗口域只受视频帧数约束、state/action
    索引为空，变倍率 k 同样作用于视频抽帧步长（ratio*k）。
- num_video_frames(N) 与 ratio、layout 在建库时烘焙进区间；改 N 需重建 ranges。
- train/holdout 由 split 表达（--split train|holdout 两个 ranges 文件，按
  blake2b(split_seed, episode_id) 确定性划分），替代原来的两个 pool 文件。
- SamplingRangesPool 在 open 期把 ranges 全量读入内存（不持 sqlite 句柄，
  DataLoader worker fork/pickle 安全），每个虚拟 selection 由
  compose_seed(seed, epoch, 0, 0, pool_id, index) 无状态生成——同
  (seed, epoch, index) 跨 rank/worker 完全一致；set_epoch 推进即换窗。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import random
import sqlite3
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from ._bootstrap import ensure_motus_data_importable

ensure_motus_data_importable()

from motus_data.index_contracts import validate_sample_selection_v1  # noqa: E402
from motus_data.sampling import compose_seed  # noqa: E402


logger = logging.getLogger(__name__)


MOTUS_SAMPLING_RANGES_V1 = "motus_sampling_ranges_v1"
DEFAULT_SPLIT_SEED = 20260713

# 窗口布局：consecutive（默认，视频连续 + 动作按 ratio 放大）| video_stride
# （旧时代 1:6 语义：动作连续 N*ratio 帧 + 视频每 ratio 帧抽 1 帧，短 episode
# 末帧填充）。
LAYOUT_CONSECUTIVE = "consecutive"
LAYOUT_VIDEO_STRIDE = "video_stride"
_KNOWN_LAYOUTS = (LAYOUT_CONSECUTIVE, LAYOUT_VIDEO_STRIDE)


class SamplingRangesError(ValueError):
    """Raised when a sampling-ranges sidecar cannot be built or bound."""


# ---------------------------------------------------------------------------
# 窗口域规则（与原 build_sample_pool_enumerate.py 相同，单一事实源）
# ---------------------------------------------------------------------------


def window_ratio(dataset: str, episode: Any, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    if dataset == "agibot":
        return int(episode.info.metadata["action_video_fps_ratio"])
    return 1


def episode_is_video_only(episode: Any) -> bool:
    """纯视频 episode（state/action 计数均为 0，如 openvid；state_action_policy=absent）。"""
    return int(episode.info.state_count) <= 0 and int(episode.info.action_count) <= 0


def is_usable_text_item(item: Any) -> bool:
    """可用于文本训练的文本项：剔除 missing_instruction 占位与空白文本。"""
    if item.kind == "missing_instruction":
        return False
    return bool(str(getattr(item, "text", "") or "").strip())


def window_start_bounds(
    dataset: str,
    episode: Any,
    *,
    num_video_frames: int,
    ratio: int,
    layout: str = LAYOUT_CONSECUTIVE,
    tail_pad_steps: int = 0,
) -> tuple[int, int]:
    if tail_pad_steps and layout != LAYOUT_VIDEO_STRIDE:
        raise SamplingRangesError("tail_pad_steps requires the video_stride layout")
    if layout == LAYOUT_VIDEO_STRIDE:
        # 起点 s 在 1:1 原生时间线上：需要 s + N*ratio <= L-1；
        # 不足整条窗口的短 episode（L>=2）退化为 s=0 单起点 + 末帧填充。
        # 纯视频源（无 state/action 时间线）只受视频帧数约束。
        if episode_is_video_only(episode):
            length = int(episode.info.video_count)
        else:
            length = min(
                int(episode.info.video_count),
                int(episode.info.state_count),
                int(episode.info.action_count),
            )
        maximum = length - num_video_frames * ratio - 1
        if tail_pad_steps:
            # 末尾最多 tail_pad_steps 步越界，读取侧 clamp 到末帧/末动作（保持不动）。
            maximum = max(maximum, min(maximum + int(tail_pad_steps), length - 2))
        if maximum < 0:
            maximum = 0 if length >= 2 else -1
        return 0, maximum

    minimum = 0
    if episode_is_video_only(episode):
        # 纯视频源：窗口域只受视频帧数约束（无 state/action 时间线）。
        return 0, int(episode.info.video_count) - num_video_frames - 1
    if dataset == "agibot":
        effective = episode.info.metadata["effective_video_frames"]
        minimum = int(effective[0])
        maximum = int(effective[1]) - num_video_frames
    else:
        maximum = episode.info.video_count - num_video_frames - 1

    maximum = min(
        maximum,
        (episode.info.action_count - 1) // ratio - num_video_frames,
        (episode.info.state_count - 1) // ratio,
    )
    return minimum, maximum


# 分段文本族（20260819 起）：episode 级 context kinds + 片段 kind。这些源的
# 文本选择随窗口走（窗口区间内的所有片段文本都拼进 prompt，统一 VLM 输出格式
# 由 utils/vlm_utils.build_vlm_task_message 负责；输入拼接按数据集各自处理——
# agibot catalog 文本已预格式化直接 join，egoverse/egoscale 由 motus_data 侧
# 注册的 composer 拼装，见 batch_adapters._compose_t5_text）。
# 零片段回退（20260819 用户拍板）：episode 没有任何片段但有可用 context 时
# （egoverse aria×3/eva 全 catalog 无 annotation 标注、仅 task_description），
# 不裁剪起点域、按 fixed 处理（prompt = 统一四字段格式要求 + task 文本）；
# 片段与 context 都没有（missing_instruction/空白）的 episode 才整条剔除。
_SEGMENTED_TEXT_FAMILIES: dict[str, tuple[tuple[str, ...], str]] = {
    "agibot": (("episode_context",), "action_segment"),
    "agibot_beta": (("episode_context",), "action_segment"),
    "egoverse": (("task_description_en",), "annotation_en"),
    "egoscale": (
        ("task_name_en", "task_description_en", "task_detail_en", "task_label"),
        "fine_action_en",
    ),
}


def _segmented_context_segments(
    episode: Any,
    context_kinds: tuple[str, ...],
    segment_kind: str,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """分段文本族的 (context 索引列表, 片段 (index, lo, hi) 列表)（item 顺序）。"""
    context: list[int] = []
    segments: list[tuple[int, int, int]] = []
    for item in episode.text.items:
        if not is_usable_text_item(item):
            continue
        if item.kind in context_kinds:
            context.append(int(item.index))
        elif item.kind == segment_kind and item.frame_range is not None:
            segments.append(
                (
                    int(item.index),
                    int(item.frame_range[0]),
                    int(item.frame_range[1]),
                )
            )
    return context, segments


def _overlap_segment_indices(
    segments: Sequence[Sequence[int]], start: int, end: int
) -> list[int]:
    """与窗口 [start, end] 有正重叠的片段 index（按 lo 升序）。"""
    overlapping = sorted(
        (int(lo), int(index))
        for index, lo, hi in segments
        if int(lo) <= int(end) and int(hi) > int(start)
    )
    return [index for _lo, index in overlapping]


def selection_text_indices(
    episode: Any,
    start: int,
    end: int,
    *,
    all_text_candidates: bool = False,
) -> list[int]:
    """窗口 [start, end]（与 start 同单位）的文本索引选择。

    分段文本族（_SEGMENTED_TEXT_FAMILIES）：context + 与窗口正重叠的全部片段
    （按 lo 升序）；episode 没有任何片段但有 context 时回退 fixed（仅 context，
    与窗口无关）；有片段但窗口无重叠返回 []（该起点不进 ranges）。其余源：
    优先 instruction_en，否则首条可用文本（fixed，与窗口无关）。

    all_text_candidates（V14，默认关）：非分段族返回**全部**有效英文候选，与
    kind="choice_one" 的 payload 对齐（建库抽检用）。注意这是「候选集」，运行时
    由 pool 从其中选恰好一条——不要把返回的多条直接当成一个窗口的最终文本。
    """
    family = _SEGMENTED_TEXT_FAMILIES.get(episode.info.source_dataset)
    if family is not None:
        context, segments = _segmented_context_segments(episode, *family)
        if not segments:
            # 零片段回退（20260819 拍板）：仅 task 的 episode 不裁剪，文本=context。
            return list(context)
        overlapping = _overlap_segment_indices(segments, start, end)
        if not overlapping:
            return []
        return context + overlapping

    items = episode.text.items
    usable = [item for item in items if is_usable_text_item(item)]
    english = [item.index for item in usable if item.kind == "instruction_en"]
    if english:
        return [int(i) for i in english] if all_text_candidates else [english[0]]
    return [usable[0].index] if usable else []


def build_selection(
    episode: Any,
    *,
    start: int,
    num_video_frames: int,
    ratio: int,
    layout: str = LAYOUT_CONSECUTIVE,
    all_text_candidates: bool = False,
    action_start_offset: int = 1,
) -> Optional[dict[str, Any]]:
    if int(action_start_offset) not in (0, 1):
        raise SamplingRangesError(
            "action_start_offset must be 0 (action[t]) or 1 (legacy action[t+1])"
        )
    action_start_offset = int(action_start_offset)
    # 文本窗口末端（与 start 同单位）：video_stride 下起点单位是原生步，窗口
    # 跨 N*ratio 步；consecutive 下是视频帧，窗口跨 N 帧（变倍率 k 仅运行时
    # 存在，建库校验按基准 k=1）。
    if layout == LAYOUT_VIDEO_STRIDE:
        text_window_end = start + num_video_frames * ratio
    else:
        text_window_end = start + num_video_frames
    text_indices = selection_text_indices(
        episode, start, text_window_end,
        all_text_candidates=all_text_candidates,
    )
    if not text_indices:
        return None
    if layout == LAYOUT_VIDEO_STRIDE:
        video_only = episode_is_video_only(episode)
        limit = (
            int(episode.info.video_count)
            if video_only
            else min(
                int(episode.info.video_count),
                int(episode.info.state_count),
                int(episode.info.action_count),
            )
        ) - 1
        video_indices = [
            min(start + ratio * k, limit) for k in range(num_video_frames + 1)
        ]
        if video_only:
            # 纯视频源：state/action 索引为空（adapter 侧按 absent 补占位）。
            selection = {
                "episode_id": episode.episode_id,
                "video_frame_indices": video_indices,
                "state_indices": [],
                "action_indices": [],
                "text_indices": text_indices,
            }
            validate_sample_selection_v1(selection)
            episode._validated_selection_indices(selection)
            return selection
        selection = {
            "episode_id": episode.episode_id,
            "video_frame_indices": video_indices,
            "state_indices": [start],
            "action_indices": [
                min(i, limit)
                for i in range(
                    start + action_start_offset,
                    start + action_start_offset + num_video_frames * ratio,
                )
            ],
            "text_indices": text_indices,
        }
    else:
        video_indices = list(range(start, start + num_video_frames + 1))
        if episode_is_video_only(episode):
            # 纯视频源：state/action 索引为空（adapter 侧按 absent 补占位）。
            selection = {
                "episode_id": episode.episode_id,
                "video_frame_indices": video_indices,
                "state_indices": [],
                "action_indices": [],
                "text_indices": text_indices,
            }
            validate_sample_selection_v1(selection)
            episode._validated_selection_indices(selection)
            return selection
        state_start = start * ratio
        action_start = state_start + action_start_offset
        action_stop = action_start + num_video_frames * ratio
        selection = {
            "episode_id": episode.episode_id,
            "video_frame_indices": video_indices,
            "state_indices": [state_start],
            "action_indices": list(range(action_start, action_stop)),
            "text_indices": text_indices,
        }
    validate_sample_selection_v1(selection)
    episode._validated_selection_indices(selection)
    return selection


# ---------------------------------------------------------------------------
# split 划分（确定性）
# ---------------------------------------------------------------------------


def split_assignment(episode_id: str, split_seed: int) -> float:
    """把 episode 映射到 [0,1) 的确定性分数（与目录顺序无关）。"""
    payload = f"{int(split_seed)}:{episode_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value / float(1 << 64)


def _split_keeps(
    split: str,
    episode_id: str,
    *,
    holdout_fraction: Optional[float],
    split_seed: int,
) -> bool:
    if split == "all":
        return True
    is_holdout = split_assignment(episode_id, split_seed) < float(holdout_fraction)
    return is_holdout if split == "holdout" else not is_holdout


# ---------------------------------------------------------------------------
# ranges 构建
# ---------------------------------------------------------------------------


def default_ranges_path(catalog_path: str | Path) -> Path:
    catalog = Path(catalog_path)
    name = catalog.name
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else name
    return catalog.with_name(f"{stem}.ranges.sqlite")


def _episode_ranges_row(
    dataset: str,
    episode: Any,
    *,
    num_video_frames: int,
    ratio_override: Optional[int],
    layout: str = LAYOUT_CONSECUTIVE,
    window_static_eps: Optional[float] = None,
    window_max_static_frames: Optional[int] = None,
    window_outlier_threshold: Optional[float] = None,
    max_stride: Optional[int] = None,
    min_episode_frames: Optional[int] = None,
    all_text_candidates: bool = False,
    action_start_offset: int = 1,
    tail_pad_steps: int = 0,
) -> tuple[Optional[dict[str, Any]], dict[str, int]]:
    """计算单个 episode 的有效起点域 + 文本载荷。

    all_text_candidates（V14，默认关）：非分段文本族的 episode 记录**全部**
    有效英文指令候选（kind="choice_one"，运行时选一），而不是只记第一条。
    默认关 = 旧库与旧采样行为逐位可复现。分段文本族不受影响。

    返回 (row | None, filter_stats)。无有效窗口返回 (None, {})；开启窗口级
    过滤后整条被滤掉返回 (None, {"filtered_out": 1, ...})；有效长度低于
    min_episode_frames 整条剔除返回 (None, {"short_rejected": 1})。
    """
    if min_episode_frames is not None:
        # 建库期最小时长过滤（20260818，openvid 加回用）：有效长度定义与
        # episode_len 列一致——纯视频源取 video_count，否则取
        # min(video, state, action)；短于阈值整条剔除（不再保留 s=0 单起点
        # 末帧填充的退化窗口）。
        if episode_is_video_only(episode):
            effective_length = int(episode.info.video_count)
        else:
            effective_length = min(
                int(episode.info.video_count),
                int(episode.info.state_count),
                int(episode.info.action_count),
            )
        if effective_length < int(min_episode_frames):
            return None, {"short_rejected": 1}
    ratio = window_ratio(dataset, episode, ratio_override)
    if ratio <= 0:
        raise SamplingRangesError(
            f"{episode.episode_id}: non-positive action/video ratio={ratio}"
        )
    minimum, maximum = window_start_bounds(
        dataset,
        episode,
        num_video_frames=num_video_frames,
        ratio=ratio,
        layout=layout,
        tail_pad_steps=tail_pad_steps,
    )
    if maximum < minimum:
        return None, {}

    items = episode.text.items
    family = _SEGMENTED_TEXT_FAMILIES.get(episode.info.source_dataset)
    if family is not None:
        # 分段文本族（agibot/agibot_beta/egoverse/egoscale）：payload 记录
        # context + 全部片段，运行时按窗口重叠选文本；起点域 = 片段覆盖并集
        # （不被任何片段覆盖的起点剔除）。零片段回退（20260819 拍板）：episode
        # 没有任何片段但有可用 context（egoverse aria×3/eva 仅 task）时按
        # fixed 处理、起点域不裁剪；context 也没有才整条剔除。
        context, segments = _segmented_context_segments(episode, *family)
        if not segments:
            if not context:
                return None, {}
            intervals = [(minimum, maximum)]
            text_payload = {"kind": "fixed", "indices": [int(i) for i in context]}
        else:
            intervals = _coverage_union_intervals(segments, minimum, maximum)
            if not intervals:
                return None, {}
            text_payload = {
                "kind": "segmented",
                "context": context,
                "segments": [[int(idx), lo, hi] for idx, lo, hi in segments],
            }
    else:
        usable = [item for item in items if is_usable_text_item(item)]
        english = [item.index for item in usable if item.kind == "instruction_en"]
        if english:
            if all_text_candidates:
                # V14 多候选库：payload 记录**全部**有效英文指令候选，运行时
                # 由 pool 按 text_choice 选恰好一条（choice_one）。不能沿用
                # kind="fixed"——fixed 的语义是「这些文本一起用」（分段文本族
                # 的 context 就是多条 fixed），运行时会被拼成一条长 prompt。
                text_payload = {
                    "kind": "choice_one",
                    "indices": [int(i) for i in english],
                }
            else:
                text_payload = {"kind": "fixed", "indices": [int(english[0])]}
        elif usable:
            # 无英文指令：保持旧行为取首条可用文本，不进多候选策略。
            text_payload = {"kind": "fixed", "indices": [int(usable[0].index)]}
        else:
            # 无可用文本（空 caption / missing_instruction）：整条 episode 剔除。
            return None, {}
        intervals = [(minimum, maximum)]

    stats: dict[str, int] = {}
    video_only = episode_is_video_only(episode)
    if video_only and (
        window_static_eps is not None or window_outlier_threshold is not None
    ):
        raise SamplingRangesError(
            f"{episode.episode_id}: window-level filters require action data; "
            "video-only episodes (no state/action) do not support them"
        )
    if not video_only and (
        window_static_eps is not None or window_outlier_threshold is not None
    ):
        _state, action = episode._reader.read_state_action(
            episode._record,
            range(int(episode.info.state_count)),
            range(int(episode.info.action_count)),
        )
        if action.numel() == 0:
            return None, {"filtered_out": 1}
        if window_static_eps is not None:
            excluded = _long_static_ranges(
                action, ratio, window_static_eps, window_max_static_frames
            )
            if layout == LAYOUT_VIDEO_STRIDE:
                # video_stride 起点域单位 = 原生步（consecutive 为视频帧）：
                # 静止区段（motion 帧单位）换算成步单位再挖除。
                excluded = [
                    (lo * ratio, (hi + 1) * ratio - 1) for lo, hi in excluded
                ]
            if excluded:
                stats["static_rejected_starts"] = _ranges_overlap_count(
                    intervals, excluded
                )
                intervals = _subtract_ranges(intervals, excluded)
                if not intervals:
                    stats["filtered_out"] = 1
                    return None, stats
            # 尾部规则：窗口末尾连续静止 >= max_static_frames 帧的起点剔除
            intervals, tail_rejected = _window_tail_rejected(
                intervals,
                action,
                ratio,
                num_video_frames,
                window_static_eps,
                window_max_static_frames,
                layout=layout,
            )
            stats["tail_rejected_starts"] = tail_rejected
            if not intervals:
                stats["filtered_out"] = 1
                return None, stats
        if window_outlier_threshold is not None:
            intervals, rejected = _refine_intervals_by_outliers(
                intervals,
                action,
                ratio,
                num_video_frames,
                window_outlier_threshold,
                layout=layout,
                action_start_offset=action_start_offset,
            )
            stats["outlier_rejected_starts"] = rejected
            if not intervals:
                stats["filtered_out"] = 1
                return None, stats

    # 建库期抽检：每个区间的首尾起点必须能通过完整 selection 校验
    for lo, hi in intervals:
        for start in {lo, hi}:
            selection = build_selection(
                episode,
                start=start,
                num_video_frames=num_video_frames,
                ratio=ratio,
                layout=layout,
                action_start_offset=action_start_offset,
            )
            if selection is None:
                raise SamplingRangesError(
                    f"{episode.episode_id}: interval [{lo},{hi}] boundary "
                    f"start={start} failed selection build"
                )

    # 该 episode 支持的最大时长倍率 k（变 FPS 训练）：窗口需再跨
    # (k-1)*N*ratio 个原生步仍落在同一区间内；再按 max_stride 封顶（至少 1）。
    span = int(num_video_frames) * int(ratio)
    episode_max_stride = max(1 + (int(hi) - int(lo)) // span for lo, hi in intervals)
    if max_stride is not None:
        episode_max_stride = min(episode_max_stride, int(max_stride))
    episode_max_stride = max(episode_max_stride, 1)

    return (
        {
            "episode_id": str(episode.episode_id),
            "ratio": int(ratio),
            "intervals": [[int(lo), int(hi)] for lo, hi in intervals],
            "text_payload": text_payload,
            "max_stride": int(episode_max_stride),
            "episode_len": (
                int(episode.info.video_count)
                if video_only
                else min(
                    int(episode.info.video_count),
                    int(episode.info.state_count),
                    int(episode.info.action_count),
                )
            ),
            "video_only": bool(video_only),
        },
        stats,
    )


def _coverage_union_intervals(
    segments: Sequence[tuple[int, int, int]],
    minimum: int,
    maximum: int,
) -> list[tuple[int, int]]:
    """起点域 = [minimum,maximum] 中被至少一个 segment 覆盖的部分（合并重叠/相邻）。

    分段文本族的起点域规则（20260819 起，用户拍板）：起点不被任何片段覆盖的
    区域剔除；同一位置被多个片段覆盖不剔除（运行时窗口重叠选择会全部拼入）。
    """
    clipped = sorted(
        (max(int(lo), minimum), min(int(hi) - 1, maximum))
        for _idx, lo, hi in segments
        if max(int(lo), minimum) <= min(int(hi) - 1, maximum)
    )
    merged: list[tuple[int, int]] = []
    for lo, hi in clipped:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# ---------------------------------------------------------------------------
# 窗口级过滤（build 期）：静止帧区段剔除 + 离群窗口起点剔除
# ---------------------------------------------------------------------------


def _frame_motion(action: torch.Tensor, ratio: int) -> torch.Tensor:
    """每视频帧的运动量 = 该帧覆盖动作步内的相邻变化量最大值。

    action 时间线 A（T×D）；帧 f 覆盖动作步 [f*r, (f+1)*r)，帧内 motion 取
    这些步之间（含进入下一帧的边界步）逐维最大变化量的最大值。r=1 时即
    相邻 qpos/状态帧的逐维最大差。T<2 或末尾不足的帧按 0（静止）计——
    末尾帧不在有效窗口起点域内，不影响过滤结果。
    """
    ratio = max(int(ratio), 1)
    t_count = int(action.shape[0])
    n_frames = (t_count + ratio - 1) // ratio
    if t_count < 2:
        return torch.zeros(n_frames, dtype=torch.float32)
    diff = (action[1:] - action[:-1]).abs().amax(dim=-1)  # (T-1,)
    pad = n_frames * ratio - int(diff.numel())
    if pad > 0:
        diff = torch.cat(
            [diff, torch.zeros(pad, dtype=diff.dtype)], dim=0
        )
    return diff.view(n_frames, ratio).amax(dim=-1)


def _long_static_ranges(
    action: torch.Tensor,
    ratio: int,
    eps: float,
    min_run: int,
) -> list[tuple[int, int]]:
    """返回长度 >= min_run 的连续静止视频帧区段 [lo, hi]（闭区间）。

    静止帧 = 帧运动量 < eps；只剔除"静止时间过长"的连续区段，孤立静止帧
    （瞬时停顿）保留。
    """
    motion = _frame_motion(action, ratio)
    static = motion < float(eps)
    n = int(static.numel())
    if n == 0 or int(min_run) <= 0:
        return []
    is_start = static & ~torch.cat(
        [torch.tensor([False], dtype=torch.bool), static[:-1]], dim=0
    )
    is_end = static & ~torch.cat(
        [static[1:], torch.tensor([False], dtype=torch.bool)], dim=0
    )
    starts = torch.nonzero(is_start).flatten().tolist()
    ends = torch.nonzero(is_end).flatten().tolist()
    return [
        (int(lo), int(hi))
        for lo, hi in zip(starts, ends)
        if hi - lo + 1 >= int(min_run)
    ]


def _subtract_ranges(
    intervals: Sequence[tuple[int, int]],
    excluded: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """从 intervals（闭区间列表）中挖掉 excluded，合并相邻结果。"""
    result: list[tuple[int, int]] = []
    for lo, hi in intervals:
        cursor = int(lo)
        for elo, ehi in sorted((int(a), int(b)) for a, b in excluded):
            if ehi < cursor:
                continue
            if elo > hi:
                break
            if elo > cursor:
                result.append((cursor, elo - 1))
            cursor = max(cursor, ehi + 1)
            if cursor > hi:
                break
        if cursor <= hi:
            result.append((cursor, hi))
    merged: list[tuple[int, int]] = []
    for lo, hi in result:
        if merged and merged[-1][1] + 1 >= lo:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _ranges_overlap_count(
    intervals: Sequence[tuple[int, int]],
    ranges: Sequence[tuple[int, int]],
) -> int:
    """intervals 与 ranges 的交叠起点数（统计静止剔除量用）。"""
    total = 0
    for lo, hi in intervals:
        for rlo, rhi in sorted((int(a), int(b)) for a, b in ranges):
            if rhi < lo:
                continue
            if rlo > hi:
                break
            total += min(hi, rhi) - max(lo, rlo) + 1
    return total


def _refine_intervals_by_mask(
    intervals: Sequence[tuple[int, int]],
    bad_by_interval: Sequence[torch.Tensor],
) -> tuple[list[tuple[int, int]], int]:
    """按逐区间坏起点掩码拆分区间；bad_by_interval[i] 与 intervals[i] 的起点一一对应。"""
    rejected = 0
    refined: list[tuple[int, int]] = []
    for (lo, hi), bad in zip(intervals, bad_by_interval):
        rejected += int(bad.sum().item())
        segment_start: Optional[int] = None
        for i in range(int(bad.numel())):
            if not bool(bad[i].item()):
                if segment_start is None:
                    segment_start = int(lo) + i
            elif segment_start is not None:
                refined.append((segment_start, int(lo) + i - 1))
                segment_start = None
        if segment_start is not None:
            refined.append((segment_start, int(hi)))
    return refined, rejected


def _refine_intervals_by_outliers(
    intervals: Sequence[tuple[int, int]],
    action: torch.Tensor,
    ratio: int,
    num_video_frames: int,
    threshold: float,
    layout: str = LAYOUT_CONSECUTIVE,
    action_start_offset: int = 1,
) -> tuple[list[tuple[int, int]], int]:
    """剔除动作/状态幅值超阈值的窗口起点。

    consecutive 布局下窗口 start s 的动作起点 = s*r+offset；
    video_stride 布局下动作起点 = s+offset；state 分别为 A[s*r]/A[s]。
    两种布局的动作窗口都是 N*r 个连续动作步。
    窗口内任一 action/state |值| > 阈值 → 该起点剔除，
    其余区间按坏起点拆开、合并。返回 (新 intervals, 剔除起点数)。
    """
    a = action.abs().amax(dim=-1)  # (T,)
    t_count = int(a.numel())
    span = int(num_video_frames) * int(ratio)
    window_max = (
        a.unfold(0, span, 1).amax(dim=-1) if t_count >= span else None
    )  # window_max[j] = max(|A[j:j+span]|)
    frame_head = a[::ratio]  # |A[f*r]|，f=0,1,...
    bad_by_interval: list[torch.Tensor] = []
    for lo, hi in intervals:
        starts = torch.arange(int(lo), int(hi) + 1)
        bad = torch.zeros(int(hi - lo) + 1, dtype=torch.bool)
        if layout == LAYOUT_VIDEO_STRIDE:
            if window_max is not None:
                idx = starts + int(action_start_offset)
                ok = idx + span <= t_count
                bad[ok] = bad[ok] | (window_max[idx[ok]] > float(threshold))
            ok = starts < t_count
            bad[ok] = bad[ok] | (a[starts[ok]] > float(threshold))
        else:
            if window_max is not None:
                idx = starts * ratio + int(action_start_offset)
                ok = idx + span <= t_count
                bad[ok] = bad[ok] | (window_max[idx[ok]] > float(threshold))
            sidx = starts * ratio
            ok = sidx < t_count
            bad[ok] = bad[ok] | (frame_head[starts[ok]] > float(threshold))
        bad_by_interval.append(bad)
    return _refine_intervals_by_mask(intervals, bad_by_interval)


def _window_tail_rejected(
    intervals: Sequence[tuple[int, int]],
    action: torch.Tensor,
    ratio: int,
    num_video_frames: int,
    eps: float,
    min_run: int,
    layout: str = LAYOUT_CONSECUTIVE,
) -> tuple[list[tuple[int, int]], int]:
    """尾部规则：窗口末尾连续静止 >= min_run 帧的起点剔除。

    consecutive：起点 s 的窗口末帧 = s + N（视频帧域）；video_stride：起点域
    单位 = 原生步，窗口末采样帧在步 s + N*r，对应 motion 帧 (s + N*r) // r。
    若窗口末帧位于长度 >= min_run 的静止 run 内，说明窗口末尾
    K 帧画面静止，剔除该起点。中段/前段静止不受影响。
    """
    runs = _long_static_ranges(action, ratio, eps, min_run)
    n_frames = int(_frame_motion(action, ratio).numel())
    if not runs or n_frames == 0:
        return list(intervals), 0
    run_len = torch.zeros(n_frames, dtype=torch.long)
    for run_lo, run_hi in runs:
        run_len[run_lo : run_hi + 1] = torch.arange(
            1, int(run_hi) - int(run_lo) + 2
        )
    bad_by_interval: list[torch.Tensor] = []
    for lo, hi in intervals:
        starts = torch.arange(int(lo), int(hi) + 1)
        if layout == LAYOUT_VIDEO_STRIDE:
            tail_idx = (starts + int(num_video_frames) * int(ratio)) // int(ratio)
        else:
            tail_idx = starts + int(num_video_frames)
        ok = tail_idx < n_frames
        bad = torch.zeros(int(hi - lo) + 1, dtype=torch.bool)
        bad[ok] = run_len[tail_idx[ok]] >= int(min_run)
        bad_by_interval.append(bad)
    return _refine_intervals_by_mask(intervals, bad_by_interval)


def _episode_guard_rejected(episode: Any, threshold: float) -> bool:
    """episode 级离群/非有限值检查（reader 归一化后的完整 action/state 时间线，
    经 reader.read_state_action 单次读取）。"""
    action_count = int(episode.info.action_count)
    state_count = int(episode.info.state_count)
    if action_count <= 0 and state_count <= 0:
        return False
    # read_state_action 是 EpisodeSourceReader 的公开 API；agibot_beta 等
    # reader 覆写为单次 canonical timeline 读取（比 read_state+read_action
    # 各读一遍省一半 I/O）。
    state, action = episode._reader.read_state_action(
        episode._record, range(state_count), range(action_count)
    )
    worst = 0.0
    for values in (state, action):
        if values.numel() == 0:
            continue
        if not bool(torch.isfinite(values).all().item()):
            return True
        worst = max(worst, float(values.abs().max().item()))
    if worst > threshold:
        logger.info(
            "action guard: reject episode %s (abs_max=%.4g > threshold=%.4g)",
            episode.episode_id,
            worst,
            threshold,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# ranges 构建（多进程 worker：每 worker 进程自持一个 EpisodeRepository）
# ---------------------------------------------------------------------------

_SCAN_CONTEXT: dict[str, Any] = {}


def _scan_worker_init(
    catalog_path: str,
    index_path: Optional[str],
    dataset: str,
    num_video_frames: int,
    ratio_override: Optional[int],
    layout: str,
    action_guard_threshold: Optional[float],
    window_static_eps: Optional[float],
    window_max_static_frames: Optional[int],
    window_outlier_threshold: Optional[float],
    max_stride: Optional[int],
    min_episode_frames: Optional[int],
    split: str,
    holdout_fraction: Optional[float],
    split_seed: int,
    reader_options: Optional[dict] = None,
    all_text_candidates: bool = False,
    action_start_offset: int = 1,
    tail_pad_steps: int = 0,
) -> None:
    from motus_data.repository import EpisodeRepository

    _SCAN_CONTEXT.update(
        repository=EpisodeRepository(
            catalog_path,
            index_path=index_path,
            verify_sha256=False,
            # 与主进程同一 reader_options：profile/归一化绑定的 catalog
            # （agibot_beta profile、libero）缺它 bind_catalog 直接报错
            reader_options=reader_options,
        ),
        dataset=dataset,
        num_video_frames=int(num_video_frames),
        ratio_override=ratio_override,
        layout=layout,
        action_guard_threshold=action_guard_threshold,
        window_static_eps=window_static_eps,
        window_max_static_frames=window_max_static_frames,
        window_outlier_threshold=window_outlier_threshold,
        max_stride=max_stride,
        min_episode_frames=min_episode_frames,
        all_text_candidates=all_text_candidates,
        action_start_offset=int(action_start_offset),
        tail_pad_steps=int(tail_pad_steps),
        split=split,
        holdout_fraction=holdout_fraction,
        split_seed=int(split_seed),
    )


def _is_numeric_quality_exclusion(exc: BaseException) -> bool:
    """agibot_beta 数值质量策略剔除的 episode（reader 读取时 raise）。"""
    return "excluded by numeric quality policy" in str(exc)


def _scan_episode_row(
    episode_id: str,
) -> tuple[str, Optional[dict[str, Any]], dict[str, int]]:
    ctx = _SCAN_CONTEXT
    if not _split_keeps(
        ctx["split"],
        str(episode_id),
        holdout_fraction=ctx["holdout_fraction"],
        split_seed=ctx["split_seed"],
    ):
        return ("skip", None, {})
    try:
        episode = ctx["repository"].get_episode(str(episode_id))
        if ctx["action_guard_threshold"] is not None and _episode_guard_rejected(
            episode, float(ctx["action_guard_threshold"])
        ):
            return ("guard", None, {})
        row, stats = _episode_ranges_row(
            ctx["dataset"],
            episode,
            num_video_frames=ctx["num_video_frames"],
            ratio_override=ctx["ratio_override"],
            layout=ctx["layout"],
            window_static_eps=ctx["window_static_eps"],
            window_max_static_frames=ctx["window_max_static_frames"],
            window_outlier_threshold=ctx["window_outlier_threshold"],
            max_stride=ctx["max_stride"],
            min_episode_frames=ctx["min_episode_frames"],
            all_text_candidates=ctx["all_text_candidates"],
            action_start_offset=ctx["action_start_offset"],
            tail_pad_steps=ctx["tail_pad_steps"],
        )
    except Exception as exc:
        # 质量策略剔除的 episode 按 guard 剔除计数，不拖垮整个构建
        if _is_numeric_quality_exclusion(exc):
            return ("guard", None, {})
        raise
    if row is None:
        return ("filtered" if stats.get("filtered_out") else "row", None, stats)
    return ("row", row, stats)


def build_sampling_ranges(
    repository: Any,
    output: str | Path,
    *,
    dataset: str,
    num_video_frames: int,
    action_fps_ratio: Optional[int] = None,
    layout: str = LAYOUT_CONSECUTIVE,
    action_guard_threshold: Optional[float] = None,
    window_static_eps: Optional[float] = None,
    window_max_static_frames: Optional[int] = None,
    window_outlier_threshold: Optional[float] = None,
    max_stride: Optional[int] = None,
    min_episode_frames: Optional[int] = None,
    all_text_candidates: bool = False,
    action_start_offset: int = 1,
    tail_pad_steps: int = 0,
    workers: int = 1,
    split: str = "all",
    holdout_fraction: Optional[float] = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    max_episodes: Optional[int] = None,
    force: bool = False,
    reader_options: Optional[dict] = None,
) -> dict[str, Any]:
    """从 EpisodeRepository 构建 ranges sidecar（原子写）。返回统计信息。

    ``action_guard_threshold``：episode 级动作/状态离群值过滤（与 releases.yaml
    的运行时 guard 同语义）：对每条 episode 读 reader 归一化后的完整
    action/state 时间线，abs_max 超过阈值或含非有限值的 episode 整条剔除，
    避免运行时 guard 异常杀死 ready-queue producer（进而拖挂整个分布式训练）。

    ``window_static_eps`` + ``window_max_static_frames``（必须成对）：窗口级
    静止过滤（两道规则叠加）——静止帧 = 帧内动作步逐维最大变化 < eps：
    1) 起点域规则：连续静止 >= max_static_frames 视频帧的区段从窗口起点域中
    剔除（孤立静止帧保留）；
    2) 尾部规则：窗口末帧 s+N 位于长度 >= max_static_frames 的静止 run 内
    （即窗口末尾连续静止 >= K 帧）的起点剔除，中段/前段静止不受影响。
    ``window_outlier_threshold``：窗口级离群过滤——窗口动作/状态幅值超阈值
    的窗口起点剔除。三者默认 None（关闭），consecutive 与 video_stride 布局
    均支持（video_stride 下起点域单位为原生步，区段/窗口按 ratio 换算）。

    ``max_stride``：每 episode 支持的最大时长倍率 k（变 FPS 训练用）的封顶
    值——episode 级 max_stride 由最终过滤后区间预计算
    （max(1 + (hi-lo)//(N*ratio))，至少 1）写入 episode_ranges.max_stride
    列，本参数仅封顶；None = 不封顶。

    ``min_episode_frames``：建库期最小时长过滤（20260818 起，openvid 加回
    用）——episode 有效长度（与 episode_len 列同定义：纯视频源取
    video_count，否则取 min(video, state, action)）低于阈值的 episode 整条
    剔除（short_rejected 计数），不再保留 video_stride 布局的 s=0 单起点
    末帧填充退化窗口；None = 不过滤（旧行为）。

    ``max_episodes``：只扫描 catalog 前 N 条 episode（split 划分在其后应用），
    供冒烟/小规模验证用；None = 全量。

    纯视频源（state/action 计数为 0，如 openvid）：自动识别，窗口域只受视频
    帧数约束，selection 的 state/action 索引为空，metadata 记录
    state_action_policy=absent；窗口级过滤与 video_stride 布局不支持。
    """
    # 每 episode 的计算量极小；多核节点上 torch 默认按核数开 intra-op 线程池，
    # 微小张量操作会被线程池同步开销拖慢数百倍（192 核节点实测 normalize 0.46s
    # -> 1ms）。构建侧固定单线程，避免无谓的线程抖动。
    torch.set_num_threads(1)
    if split not in {"all", "train", "holdout"}:
        raise SamplingRangesError(f"unknown split={split!r}")
    if layout not in _KNOWN_LAYOUTS:
        raise SamplingRangesError(
            f"unknown layout={layout!r} (known: {sorted(_KNOWN_LAYOUTS)})"
        )
    if int(num_video_frames) <= 0:
        raise SamplingRangesError("--num-video-frames must be positive")
    if int(action_start_offset) not in (0, 1):
        raise SamplingRangesError("--action-start-offset must be 0 or 1")
    if action_guard_threshold is not None and not float(action_guard_threshold) > 0:
        raise SamplingRangesError("--action-guard-threshold must be positive")
    if (window_static_eps is not None) != (window_max_static_frames is not None):
        raise SamplingRangesError(
            "--window-static-eps and --window-max-static-frames must be set together"
        )
    if window_static_eps is not None and not float(window_static_eps) > 0:
        raise SamplingRangesError("--window-static-eps must be positive")
    if window_max_static_frames is not None and not int(window_max_static_frames) > 0:
        raise SamplingRangesError("--window-max-static-frames must be positive")
    if (
        window_outlier_threshold is not None
        and not float(window_outlier_threshold) > 0
    ):
        raise SamplingRangesError("--window-outlier-threshold must be positive")
    if max_stride is not None and not int(max_stride) > 0:
        raise SamplingRangesError("--max-stride must be positive")
    if min_episode_frames is not None and not int(min_episode_frames) > 0:
        raise SamplingRangesError("--min-episode-frames must be positive")
    if int(tail_pad_steps) < 0:
        raise SamplingRangesError("--tail-pad-steps must be >= 0")
    if int(tail_pad_steps) and layout != LAYOUT_VIDEO_STRIDE:
        raise SamplingRangesError("--tail-pad-steps requires --video-stride")
    if split != "all":
        if holdout_fraction is None:
            raise SamplingRangesError(f"split={split} requires holdout_fraction")
        if not 0.0 < float(holdout_fraction) < 1.0:
            raise SamplingRangesError("holdout_fraction must be within (0,1)")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise SamplingRangesError("--max-episodes must be positive")
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"output exists; use --force: {output}")

    episode_ids: Any = repository.iter_episode_ids()
    if max_episodes is not None:
        episode_ids = islice(episode_ids, int(max_episodes))

    rows: list[dict[str, Any]] = []
    ratios: set[int] = set()
    guard_rejected = 0
    static_rejected_starts = 0
    tail_rejected_starts = 0
    outlier_rejected_starts = 0
    filtered_episodes = 0
    short_rejected = 0
    if int(workers) > 1:
        # 多进程并行：行计算按 episode 相互独立，executor.map 保持 catalog
        # 顺序（结果与串行路径逐字节一致）；sqlite 写入仍在主进程单线程。
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_scan_worker_init,
            initargs=(
                str(repository.catalog_path),
                (
                    None
                    if repository.index_path is None
                    else str(repository.index_path)
                ),
                str(dataset),
                int(num_video_frames),
                action_fps_ratio,
                str(layout),
                action_guard_threshold,
                window_static_eps,
                window_max_static_frames,
                window_outlier_threshold,
                max_stride,
                min_episode_frames,
                str(split),
                holdout_fraction,
                int(split_seed),
                reader_options,
                bool(all_text_candidates),
                int(action_start_offset),
                int(tail_pad_steps),
            ),
        ) as pool:
            for tag, row, stats in pool.map(
                _scan_episode_row, episode_ids, chunksize=64
            ):
                if tag == "guard":
                    guard_rejected += 1
                    continue
                if row is None:
                    if stats.get("filtered_out"):
                        filtered_episodes += 1
                    short_rejected += int(stats.get("short_rejected", 0))
                    continue
                ratios.add(int(row["ratio"]))
                rows.append(row)
                static_rejected_starts += int(stats.get("static_rejected_starts", 0))
                tail_rejected_starts += int(stats.get("tail_rejected_starts", 0))
                outlier_rejected_starts += int(stats.get("outlier_rejected_starts", 0))
    else:
        for episode_id in episode_ids:
            if not _split_keeps(
                split,
                str(episode_id),
                holdout_fraction=holdout_fraction,
                split_seed=split_seed,
            ):
                continue
            episode = None
            try:
                episode = repository.get_episode(str(episode_id))
                if action_guard_threshold is not None and _episode_guard_rejected(
                    episode, float(action_guard_threshold)
                ):
                    guard_rejected += 1
                    continue
                row, stats = _episode_ranges_row(
                    dataset,
                    episode,
                    num_video_frames=int(num_video_frames),
                    ratio_override=action_fps_ratio,
                    layout=layout,
                    window_static_eps=window_static_eps,
                    window_max_static_frames=window_max_static_frames,
                    window_outlier_threshold=window_outlier_threshold,
                    max_stride=max_stride,
                    min_episode_frames=min_episode_frames,
                    all_text_candidates=all_text_candidates,
                    action_start_offset=action_start_offset,
                    tail_pad_steps=int(tail_pad_steps),
                )
            except Exception as exc:
                # 质量策略剔除的 episode 按 guard 剔除计数（与多进程路径一致）
                if _is_numeric_quality_exclusion(exc):
                    guard_rejected += 1
                    continue
                raise
            if row is None:
                if stats.get("filtered_out"):
                    filtered_episodes += 1
                short_rejected += int(stats.get("short_rejected", 0))
                continue
            ratios.add(int(row["ratio"]))
            rows.append(row)
            static_rejected_starts += int(stats.get("static_rejected_starts", 0))
            tail_rejected_starts += int(stats.get("tail_rejected_starts", 0))
            outlier_rejected_starts += int(stats.get("outlier_rejected_starts", 0))
    if not rows:
        raise SamplingRangesError(
            f"no valid episode ranges built from catalog (split={split})"
        )
    video_only_rows = sum(1 for row in rows if row.pop("video_only"))
    if 0 < video_only_rows < len(rows):
        raise SamplingRangesError(
            f"mixed video-only/action episodes in one catalog are not supported "
            f"({video_only_rows}/{len(rows)} video-only)"
        )
    video_only = video_only_rows > 0

    metadata = {
        "format": MOTUS_SAMPLING_RANGES_V1,
        "catalog_id": str(repository.catalog_id),
        "dataset": str(dataset),
        "num_video_frames": str(int(num_video_frames)),
        "action_video_fps_ratio": (
            str(next(iter(ratios))) if len(ratios) == 1 else "mixed"
        ),
        "layout": str(layout),
        "state_action_policy": "absent" if video_only else "real",
        "action_start_offset": str(int(action_start_offset)),
        # V14：非分段族的文本载荷口径。choice_one = 记录全部英文候选、运行时
        # 选一（pool 的 text_choice 决定随机还是固定首条）；first_only = 旧行为。
        "text_payload_policy": "choice_one" if all_text_candidates else "first_only",
        "episodes": str(len(rows)),
        "split": str(split),
        "holdout_fraction": (
            "" if holdout_fraction is None else str(float(holdout_fraction))
        ),
        "split_seed": str(int(split_seed)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if window_static_eps is not None:
        metadata["window_static_eps"] = str(float(window_static_eps))
        metadata["window_max_static_frames"] = str(int(window_max_static_frames))
        metadata["window_static_tail"] = "1"
    if window_outlier_threshold is not None:
        metadata["window_outlier_threshold"] = str(float(window_outlier_threshold))
    if max_stride is not None:
        metadata["max_stride"] = str(int(max_stride))
    if min_episode_frames is not None:
        metadata["min_episode_frames"] = str(int(min_episode_frames))
    if int(tail_pad_steps):
        metadata["tail_pad_steps"] = str(int(tail_pad_steps))

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", dir=str(output.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        connection = sqlite3.connect(str(temporary))
        try:
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE episode_ranges ("
                "ordinal INTEGER PRIMARY KEY, "
                "episode_id TEXT NOT NULL UNIQUE, "
                "ratio INTEGER NOT NULL, "
                "intervals TEXT NOT NULL, "
                "text_payload TEXT NOT NULL, "
                "episode_len INTEGER NOT NULL, "
                "max_stride INTEGER NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            connection.executemany(
                "INSERT INTO episode_ranges "
                "(ordinal, episode_id, ratio, intervals, text_payload, "
                "episode_len, max_stride) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        ordinal,
                        row["episode_id"],
                        row["ratio"],
                        json.dumps(row["intervals"], separators=(",", ":")),
                        json.dumps(
                            row["text_payload"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        int(row["episode_len"]),
                        int(row["max_stride"]),
                    )
                    for ordinal, row in enumerate(rows)
                ],
            )
            connection.commit()
            connection.execute("PRAGMA quick_check")
        finally:
            connection.close()
        if force:
            output.unlink(missing_ok=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    padded_episodes = (
        sum(
            1
            for row in rows
            if int(row["episode_len"]) < int(num_video_frames) * int(row["ratio"]) + 1
        )
        if layout == LAYOUT_VIDEO_STRIDE
        else 0
    )
    return {
        "dataset": str(dataset),
        "catalog_id": str(repository.catalog_id),
        "episodes": len(rows),
        "intervals": sum(len(row["intervals"]) for row in rows),
        "valid_starts": sum(
            hi - lo + 1 for row in rows for lo, hi in row["intervals"]
        ),
        "ratio": metadata["action_video_fps_ratio"],
        "layout": str(layout),
        "max_stride": (None if max_stride is None else int(max_stride)),
        "padded_episodes": int(padded_episodes),
        "guard_rejected": int(guard_rejected),
        "static_rejected_starts": int(static_rejected_starts),
        "tail_rejected_starts": int(tail_rejected_starts),
        "outlier_rejected_starts": int(outlier_rejected_starts),
        "filtered_episodes": int(filtered_episodes),
        "short_rejected": int(short_rejected),
        "split": split,
        "output": str(output),
    }


# ---------------------------------------------------------------------------
# 运行时 pool（SupportsSamplePool 实现）
# ---------------------------------------------------------------------------


def _feistel_permute(position: int, size: int, key: int) -> int:
    """把 [0,size) 映射为自身上的确定性置换（without_replacement 用）。

    4 轮非平衡 Feistel：把 2^bits 空间按 (narrow, wide) 两半切分，每轮
    (hi, lo) -> (lo, hi ^ F(lo))、两半宽度逐轮互换（任意轮函数 F 下该
    变换都是 2^bits 空间上的双射），轮函数 F = blake2b(key, round, lo)
    取整后按对半宽掩码；cycle-walking（结果 >= size 时以它为输入重跑，
    直至落入 [0,size)）把 2^bits 上的双射收缩为子集 [0,size) 上的双射。
    O(1) 时间/内存，无需像 Fisher-Yates 那样按 episode 物化整张置换表。
    size=2（bits=1，切分退化）按 key 奇偶交换特判。
    """
    if size <= 1:
        return 0
    if size == 2:
        return int(position) ^ (int(key) & 1)
    bits = (size - 1).bit_length()  # >= 2
    left_bits = bits // 2
    right_bits = bits - left_bits  # 两半均 >= 1 bit
    x = int(position)
    while True:
        value = x
        narrow, wide = left_bits, right_bits
        for round_index in range(4):
            narrow_mask = (1 << narrow) - 1
            hi = (value >> wide) & narrow_mask
            lo = value & ((1 << wide) - 1)
            digest = hashlib.blake2b(
                f"{int(key)}:{round_index}:{lo}".encode("utf-8"),
                digest_size=8,
            ).digest()
            round_value = int.from_bytes(digest, "little") & narrow_mask
            value = (lo << narrow) | ((hi ^ round_value) & narrow_mask)
            narrow, wide = wide, narrow
        if value < size:
            return value
        x = value  # cycle walking


class _EpisodeRanges:
    __slots__ = ("episode_id", "ratio", "intervals", "starts_prefix", "total_starts", "text_payload", "episode_len", "max_stride")

    def __init__(
        self,
        episode_id: str,
        ratio: int,
        intervals: Sequence[Sequence[int]],
        text_payload: Mapping[str, Any],
        episode_len: Optional[int] = None,
        max_stride: int = 1,
    ):
        self.episode_id = episode_id
        self.ratio = int(ratio)
        self.intervals = tuple((int(lo), int(hi)) for lo, hi in intervals)
        prefix = [0]
        for lo, hi in self.intervals:
            prefix.append(prefix[-1] + (hi - lo + 1))
        self.starts_prefix = tuple(prefix)
        self.total_starts = prefix[-1]
        self.text_payload = dict(text_payload)
        self.episode_len = None if episode_len is None else int(episode_len)
        self.max_stride = int(max_stride)

    def start_at(self, pick: int) -> int:
        """把 [0,total_starts) 的全局序号映射为区间内的窗口起点。"""
        interval_index = bisect.bisect_right(self.starts_prefix, pick) - 1
        lo, _hi = self.intervals[interval_index]
        return lo + (pick - self.starts_prefix[interval_index])

    @staticmethod
    def start_at_in(
        intervals: Sequence[tuple[int, int]], pick: int
    ) -> int:
        """同 start_at，但作用于临时收缩后的区间列表（变倍率起点域）。"""
        prefix = [0]
        for lo, hi in intervals:
            prefix.append(prefix[-1] + (int(hi) - int(lo) + 1))
        interval_index = bisect.bisect_right(prefix, pick) - 1
        lo, _hi = intervals[interval_index]
        return int(lo) + (pick - prefix[interval_index])

    def text_indices(
        self,
        start: int,
        end: Optional[int] = None,
        text_rng: Optional[random.Random] = None,
    ) -> list[int]:
        payload = self.text_payload
        kind = payload["kind"]
        if kind == "fixed":
            return [int(index) for index in payload["indices"]]
        if kind == "choice_one":
            # V14 多候选：恰好选一条（选中的才会被 _compose_t5_text 拼成
            # prompt；返回多条会被拼成一条长句）。text_rng=None = 评测口径，
            # 固定取第一条候选。
            indices = [int(index) for index in payload["indices"]]
            if not indices:
                raise SamplingRangesError(
                    f"{self.episode_id}: choice_one text_payload is empty "
                    "(ranges file corrupted?)"
                )
            if text_rng is None:
                return [indices[0]]
            return [text_rng.choice(indices)]
        if kind == "agibot":
            # 旧库（20260819 前构建）：恰好一个覆盖 start 的 action_segment。
            context = int(payload["context"])
            covering = [
                int(index)
                for index, lo, hi in payload["segments"]
                if int(lo) <= start < int(hi)
            ]
            if len(covering) != 1:
                raise SamplingRangesError(
                    f"{self.episode_id}: start={start} is not covered by exactly "
                    "one action_segment (ranges file corrupted?)"
                )
            return [context, covering[0]]
        if kind == "segmented":
            # context + 与窗口 [start, end] 有正重叠的全部片段（按 lo 升序）。
            if end is None:
                raise SamplingRangesError(
                    "segmented text_payload requires the window end"
                )
            overlapping = _overlap_segment_indices(
                payload["segments"], start, end
            )
            if not overlapping:
                raise SamplingRangesError(
                    f"{self.episode_id}: start={start} end={end} overlaps no "
                    "text segment (ranges file corrupted?)"
                )
            return [int(index) for index in payload["context"]] + overlapping
        raise SamplingRangesError(f"unknown text_payload kind={kind!r}")


class _VirtualSelections(Sequence):
    """惰性虚拟 selection 序列：按下标无状态生成。"""

    def __init__(self, pool: "SamplingRangesPool"):
        self._pool = pool

    def __len__(self) -> int:
        return len(self._pool)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return self._pool.generate(int(index))


class SamplingRangesPool:
    """读 ranges sidecar、在内存中按 (seed, epoch, index) 动态生成随机窗口的 pool。

    实现 motus_data.SupportsSamplePool 协议：selections / episode_groups /
    pool_id / __len__ / set_epoch。virtual 长度 = 每 episode
    windows_per_episode（缺省 = 该 episode 的有效起点总数）。
    """

    pool_kind = "sampling_ranges"

    def __init__(
        self,
        ranges_path: str | Path,
        repository: Any,
        *,
        seed: int = 0,
        num_video_frames: Optional[int] = None,
        windows_per_episode: Optional[int] = None,
        stride_max: int = 1,
        text_choice: str = "first",
        without_replacement: bool = False,
        action_start_offset: int = 1,
    ):
        """text_choice（V14）：kind="choice_one" 库的候选选择口径。

        ``"first"``（缺省，旧行为）= 恒取第一条候选，评测/回归用；
        ``"random"`` = 每个采样身份用独立文本 RNG 选一条，训练用。
        对 fixed/segmented/agibot 载荷无影响。

        without_replacement（V14）：episode 内起点抽取口径。``False``（缺省，
        旧行为）= 每个虚拟槽独立均匀有放回；``True`` = 槽序号经 Feistel 置换
        无放回发牌，同一轮扫描内每个有效起点至多一次。只影响"槽被取到后抽
        哪个起点"，不改虚拟长度、episode_groups、组批与 episode 间配比。
        eval_pool 不要开（periodic_eval 定集跨轮可比性）。
        """
        path = Path(ranges_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if windows_per_episode is not None and int(windows_per_episode) <= 0:
            raise SamplingRangesError("windows_per_episode must be positive")
        if int(stride_max) < 1:
            raise SamplingRangesError("stride_max must be >= 1")
        if str(text_choice) not in ("first", "random"):
            raise SamplingRangesError(
                f"unknown text_choice={text_choice!r}; known: first, random"
            )
        self._text_choice = str(text_choice)
        if int(action_start_offset) not in (0, 1):
            raise SamplingRangesError(
                "action_start_offset must be 0 (action[t]) or 1 (legacy action[t+1])"
            )
        # LeRobot/RoboTwin 的 action[t] 是 observation.state[t] 后发出的命令，
        # 数据中可验证 action[t] == observation.state[t+1]。旧 video_stride
        # 路径从 t+1 开始会额外跳过一个控制拍。默认 1 保持其他既有配置行为；
        # RoboTwin V14 在 YAML 中显式设 0。
        self._action_start_offset = int(action_start_offset)

        connection = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=1", uri=True
        )
        try:
            metadata = dict(
                connection.execute("SELECT key, value FROM metadata").fetchall()
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(episode_ranges)"
                ).fetchall()
            }
            has_episode_len = "episode_len" in columns
            has_max_stride = "max_stride" in columns
            rows = connection.execute(
                "SELECT episode_id, ratio, intervals, text_payload"
                + (", episode_len" if has_episode_len else "")
                + (", max_stride" if has_max_stride else "")
                + " FROM episode_ranges ORDER BY ordinal"
            ).fetchall()
        finally:
            connection.close()

        if metadata.get("format") != MOTUS_SAMPLING_RANGES_V1:
            raise SamplingRangesError(
                f"{path}: format must be {MOTUS_SAMPLING_RANGES_V1}, got "
                f"{metadata.get('format')!r}"
            )
        baked_action_start_offset = metadata.get("action_start_offset")
        if (
            baked_action_start_offset is not None
            and int(baked_action_start_offset) != self._action_start_offset
        ):
            raise SamplingRangesError(
                f"{path}: ranges were built with action_start_offset="
                f"{baked_action_start_offset} but config requires "
                f"{self._action_start_offset}; rebuild or use a matching pool option"
            )
        catalog_id = str(metadata.get("catalog_id", ""))
        if catalog_id != str(repository.catalog_id):
            raise SamplingRangesError(
                f"{path}: catalog_id={catalog_id!r} does not match repository "
                f"catalog_id={str(repository.catalog_id)!r}"
            )
        baked_frames = int(metadata["num_video_frames"])
        if num_video_frames is not None and int(num_video_frames) != baked_frames:
            raise SamplingRangesError(
                f"{path}: ranges were built with num_video_frames="
                f"{baked_frames} but config requires {int(num_video_frames)}; "
                "rebuild the ranges file"
            )
        if not rows:
            raise SamplingRangesError(f"{path}: no episode ranges")

        self.path = path
        self.metadata = metadata
        self.num_video_frames = baked_frames
        self._seed = int(seed)
        self._epoch = 0
        layout = str(metadata.get("layout") or LAYOUT_CONSECUTIVE)
        if layout not in _KNOWN_LAYOUTS:
            raise SamplingRangesError(
                f"{path}: unknown layout={layout!r} (known: {sorted(_KNOWN_LAYOUTS)})"
            )
        if layout == LAYOUT_VIDEO_STRIDE and not has_episode_len:
            raise SamplingRangesError(
                f"{path}: layout=video_stride requires the episode_len column; "
                "rebuild the ranges file"
            )
        self._layout = layout
        self._stride_max = int(stride_max)
        if self._stride_max > 1:
            # 变 FPS 训练：时长倍率 k ∈ [1, stride_max] 整数均匀；仅 video_stride
            # 布局，且 ranges 必须含建库预计算的 max_stride 列（防呆：旧库缺列时
            # 直接报错，避免静默退化成 k=1）。
            if layout != LAYOUT_VIDEO_STRIDE:
                raise SamplingRangesError(
                    f"{path}: stride_max>1 requires the video_stride layout, "
                    f"got layout={layout!r}"
                )
            if not has_max_stride:
                raise SamplingRangesError(
                    f"{path}: stride_max>1 requires the max_stride column; "
                    "rebuild the ranges file with --max-stride"
                )
        state_action_policy = str(metadata.get("state_action_policy") or "real")
        if state_action_policy not in ("real", "absent"):
            raise SamplingRangesError(
                f"{path}: unknown state_action_policy={state_action_policy!r}"
            )
        self._video_only = state_action_policy == "absent"
        self._pool_id = f"{metadata.get('dataset', path.stem)}_sampling_ranges_v1"
        self._windows_per_episode = (
            None if windows_per_episode is None else int(windows_per_episode)
        )
        # 无放回起点抽取（V14）：episode 内第 j 个槽映射到
        # (seed, epoch, episode, pass) 定种的 Feistel 置换第 j 位——同一轮扫描
        # 内每个有效起点恰好被抽一次（旧有放回路径约 37% 抽取撞已抽起点、约
        # 37% 起点整轮落空）；仍是 (seed, epoch, index) 的纯函数，跨 rank/worker
        # 与弹性重启逐位一致。
        self._without_replacement = bool(without_replacement)

        episodes: list[_EpisodeRanges] = []
        offsets: list[int] = []
        groups: dict[str, tuple[int, ...]] = {}
        total = 0
        max_stride_column = 4 + (1 if has_episode_len else 0)
        for row in rows:
            episode_id, ratio, intervals_json, payload_json = row[:4]
            episode = _EpisodeRanges(
                str(episode_id),
                int(ratio),
                json.loads(intervals_json),
                json.loads(payload_json),
                episode_len=(int(row[4]) if has_episode_len else None),
                max_stride=(int(row[max_stride_column]) if has_max_stride else 1),
            )
            width = (
                episode.total_starts
                if self._windows_per_episode is None
                else self._windows_per_episode
            )
            episodes.append(episode)
            offsets.append(total)
            groups[episode.episode_id] = tuple(range(total, total + width))
            total += width
        self._episodes = tuple(episodes)
        self._offsets = tuple(offsets)
        self._episode_groups = groups
        self._length = total
        self.selections = _VirtualSelections(self)

    @property
    def pool_id(self) -> str:
        return self._pool_id

    @property
    def episode_groups(self) -> Mapping[str, tuple[int, ...]]:
        return self._episode_groups

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _pick_start(
        self, episode: _EpisodeRanges, draw: int, rng: random.Random
    ) -> int:
        """把 episode 内第 draw 个槽映射为窗口起点。

        默认（without_replacement 关闭）：独立均匀随机起点（旧行为，rng
        消耗序列逐位不变）。开启：draw 经 (seed, epoch, episode, pass)
        定种的 Feistel 置换无放回发牌——同一轮扫描内每个有效起点恰好
        被抽一次；draw >= total_starts 时按 pass = draw // total_starts
        换新置换（置换 key 不含 rank/worker/index，跨 rank 一致）。
        """
        if not self._without_replacement:
            return episode.start_at(rng.randrange(episode.total_starts))
        pass_index, position = divmod(int(draw), episode.total_starts)
        key = compose_seed(
            self._seed,
            self._epoch,
            0,
            0,
            f"{self._pool_id}|wor|{episode.episode_id}",
            pass_index,
        )
        return episode.start_at(
            _feistel_permute(position, episode.total_starts, key)
        )

    def _locate(self, index: int) -> tuple[_EpisodeRanges, int]:
        if not 0 <= index < self._length:
            raise IndexError(
                f"selection index {index} out of range [0,{self._length})"
            )
        episode_index = bisect.bisect_right(self._offsets, index) - 1
        return self._episodes[episode_index], self._offsets[episode_index]

    def _window_draw(self, index: int, offset: int) -> int:
        return int(index) - offset

    def generate(self, index: int) -> dict[str, Any]:
        episode, offset = self._locate(int(index))
        rng = random.Random(
            compose_seed(
                self._seed,
                self._epoch,
                0,
                0,
                self._pool_id,
                int(index),
            )
        )
        # 文本候选选择用**独立**的 RNG（V14）：同一采样身份 + 固定的文本用途
        # 标识派生，不从上面那个 rng 抽——窗口起点与变倍率 k 都吃它的流，多抽
        # 一次会整体挪动窗口采样。text_choice="first" 时不建 RNG（评测口径）。
        text_rng = (
            random.Random(
                compose_seed(
                    self._seed,
                    self._epoch,
                    0,
                    0,
                    f"{self._pool_id}:text_choice",
                    int(index),
                )
            )
            if self._text_choice == "random"
            else None
        )
        frames = self.num_video_frames
        ratio = episode.ratio
        if self._layout == LAYOUT_VIDEO_STRIDE:
            # 旧时代 1:6 语义：动作连续 N*ratio 帧 + 视频每 ratio 帧抽 1 帧；
            # 短 episode 索引 clamp 到 episode_len-1（末帧填充，建库期烘焙上限）。
            limit = int(episode.episode_len) - 1
            k = 1
            if self._stride_max > 1:
                # 变 FPS 训练：时长倍率 k ∈ [1, stride_max] 整数均匀（同一 rng
                # 定种；rng 消耗只发生在本分支，stride_max==1 的旧路径逐位不变）。
                k = min(rng.randint(1, self._stride_max), episode.max_stride)
                span = frames * ratio
                # 起点域按 k 精确收缩（hi_k = hi - span*(k-1)），保证窗口末端
                # 不越出原有效区间；收缩后无有效区间时退 k=1（恒可用）。
                adjusted = [
                    (lo, hi - span * (k - 1))
                    for lo, hi in episode.intervals
                    if hi - span * (k - 1) >= lo
                ]
                if adjusted:
                    start = _EpisodeRanges.start_at_in(
                        adjusted,
                        rng.randrange(
                            sum(hi - lo + 1 for lo, hi in adjusted)
                        ),
                    )
                else:
                    k = 1
                    start = episode.start_at(rng.randrange(episode.total_starts))
            else:
                start = self._pick_start(episode, self._window_draw(index, offset), rng)
            # 文本窗口末端 = 窗口实际覆盖的末帧（clamp 到 limit；变倍率 k 放大跨度）。
            text_end = min(start + frames * ratio * k, limit)
            if self._video_only:
                # 纯视频源（absent）：state/action 索引为空，adapter 侧补占位；
                # video_stride 下同样按倍率 k 抽帧（limit 已烘焙为 video_count-1）。
                selection = {
                    "episode_id": episode.episode_id,
                    "video_frame_indices": [
                        min(start + ratio * k * j, limit)
                        for j in range(frames + 1)
                    ],
                    "state_indices": [],
                    "action_indices": [],
                    "text_indices": episode.text_indices(start, text_end, text_rng=text_rng),
                }
                validate_sample_selection_v1(selection)
                return selection
            selection = {
                "episode_id": episode.episode_id,
                "video_frame_indices": [
                    min(start + ratio * k * j, limit) for j in range(frames + 1)
                ],
                "state_indices": [start],
                "action_indices": [
                    min(start + k * (j + self._action_start_offset), limit)
                    for j in range(frames * ratio)
                ],
                "text_indices": episode.text_indices(start, text_end, text_rng=text_rng),
            }
        else:
            start = self._pick_start(episode, self._window_draw(index, offset), rng)
            video_indices = list(range(start, start + frames + 1))
            # consecutive 布局文本窗口末端 = 末视频帧（start 与片段范围同单位）。
            text_end = video_indices[-1]
            if self._video_only:
                # 纯视频源（absent）：state/action 索引为空，adapter 侧补占位。
                selection = {
                    "episode_id": episode.episode_id,
                    "video_frame_indices": video_indices,
                    "state_indices": [],
                    "action_indices": [],
                    "text_indices": episode.text_indices(start, text_end, text_rng=text_rng),
                }
                validate_sample_selection_v1(selection)
                return selection
            state_start = start * ratio
            selection = {
                "episode_id": episode.episode_id,
                "video_frame_indices": video_indices,
                "state_indices": [state_start],
                "action_indices": list(
                    range(
                        state_start + self._action_start_offset,
                        state_start + self._action_start_offset + frames * ratio,
                    )
                ),
                "text_indices": episode.text_indices(start, text_end, text_rng=text_rng),
            }
        validate_sample_selection_v1(selection)
        return selection


# ---------------------------------------------------------------------------
# provider 注册（data_prepare.pool_providers）
# ---------------------------------------------------------------------------


def _build_pool_from_options(
    repository: Any,
    *,
    seed: int,
    path: str,
    num_video_frames: Optional[int] = None,
    windows_per_episode: Optional[int] = None,
    stride_max: int = 1,
    text_choice: str = "first",
    without_replacement: bool = False,
    action_start_offset: int = 1,
) -> SamplingRangesPool:
    return SamplingRangesPool(
        path,
        repository,
        seed=seed,
        num_video_frames=num_video_frames,
        windows_per_episode=windows_per_episode,
        stride_max=stride_max,
        text_choice=text_choice,
        without_replacement=without_replacement,
        action_start_offset=action_start_offset,
    )


def register_provider() -> None:
    from .pool_providers import register_pool_builder

    register_pool_builder(
        "sampling_ranges",
        _build_pool_from_options,
        known_options=(
            "path",
            "num_video_frames",
            "windows_per_episode",
            "stride_max",
            "text_choice",
            "without_replacement",
            "action_start_offset",
        ),
    )


register_provider()
