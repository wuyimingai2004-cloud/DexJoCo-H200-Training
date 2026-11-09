"""periodic_eval 配置解析：YAML ``periodic_eval`` 节 → 类型化配置对象。

评测集构成：训练 YAML ``dataset.sources`` 中各 indexed 源的 holdout（eval_pool）
+ 可选 EWMBenchmark（视频-only 基准，集群 /data/chen.liao/EWMBenchmark）。
各源权重决定 world_size 个 rank 的 batch 来源配比（largest-remainder，
纯函数、跨 rank/跨轮次确定一致——满足"每次 eval 内容一样、配比一样"）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from thinker_wm.utils.output_paths import resolve_output_path

#: EWMBenchmark 在权重表/源名中使用的保留名字（不会与 indexed 源重名）。
EWMBENCHMARK_SOURCE_NAME = "ewmbenchmark"

#: EWMBenchmark 样本走模型时借用的已注册机器人名（EWM 无 action/state 语义；
#: 未注册名会被 action 分支拒绝，state=None 也不被 action encoder 接受）。
EWMBENCHMARK_ROBOT_NAME = "agibot-a2-d"


@dataclass
class SourceAssignment:
    """一个 rank 在一轮评测中负责的固定 batch。"""

    source_name: str  # indexed 源名或 EWMBENCHMARK_SOURCE_NAME
    kind: str  # "indexed" | "ewmbenchmark"
    shard_ordinal: int  # 同一源内第几个 batch（源内确定顺序）
    # 取自哪个池："holdout"（eval_pool，默认）| "train"（pool）。train 用于在
    # 训练集自身的 clip 上做同口径评测：与 holdout 并排看才能分辨"没学会"
    # （两边都差）与"没泛化"（train 好 holdout 差）。ewmbenchmark 无 train 池。
    split: str = "holdout"
    # None = 用 pe_cfg.batch_size / 训练 batch 的缺省。holdout_family_counts
    # 把一个家族的多条 clip 打进同一个 rank 时会写上明确的条数。
    batch_size: Optional[int] = None
    # True：shard_ordinal 是洗牌后 episode 的起始下标（取 [start, start+bs)），
    # 用来把 N 条 holdout 拆到多卡且不重叠。False：旧口径 start=shard*bs。
    shard_as_start: bool = False


@dataclass
class PeriodicEvalConfig:
    enabled: bool = False
    interval_steps: int = 1000
    num_inference_steps: Optional[int] = None  # None = model.inference.num_inference_timesteps
    batch_size: Optional[int] = None  # None = 训练 batch_size // 4（最小 1）
    seed: int = 20260805
    vlm_generate: bool = True
    vlm_max_new_tokens: int = 128
    fps: int = 6
    keep_rounds: Optional[int] = None  # None = 全保留
    save_separate_videos: bool = False
    # GT-depth action 对照臂：额外跑一次 teacher-force GT 深度的推理，只取
    # action，指标以 ``*_gtdepth`` 后缀并入。用来给"好深度对 action 的收益"
    # 定上界——生成深度自身有误差时，action 读它没提升说不清是通路不通还是
    # 深度不够好，这一臂把两者分开。代价是评测耗时约翻倍，默认关。
    gt_depth_action_arm: bool = False
    # 无历史对照臂（仅 history_enable 时生效）：主评估带历史，这一臂同种子
    # 去掉历史再推理一次，指标以 ``*_nohist`` 后缀并入，用来量化历史记忆的
    # 收益。代价是评测耗时约翻倍，默认关。
    no_history_arm: bool = False
    # A2V 臂（仅逐块 AR 模型）：给定真实执行动作只预测 RGB 后果，指标后缀
    # ``*_a2v``（有真实后果标签）；同种子再用"整段保持首步动作"推理一次，
    # 与 A2V 预测之差记为 ``*_a2v_sensitivity``（动作敏感性，无标签，不是
    # 准确率）。代价是多两次推理，默认关。
    a2v_arm: bool = False
    # 分给训练池（pool）的 rank 比例，其余 rank 走 holdout（eval_pool）。
    # 0.0（默认）= 全部 holdout，保持旧行为。8 卡 + 0.5 → 4 个 train rank + 4 个 holdout rank。
    train_pool_fraction: float = 0.0
    # 按家族指定 holdout clip 条数（agibot / egoscale）。空 = 每个 holdout
    # rank 一条（旧行为）。非空时在家族内按 source_weights 做 largest-remainder，
    # 再把各源的条数打进 holdout rank 的 batch_size；train rank 不受影响。
    holdout_family_counts: Dict[str, int] = field(default_factory=dict)
    cache_dir: Optional[str] = None
    source_weights: Dict[str, float] = field(default_factory=dict)  # 显式覆盖；缺省见下
    ewmbenchmark_enabled: bool = False
    ewmbenchmark_root: Optional[str] = None
    ewmbenchmark_robot_name: str = EWMBENCHMARK_ROBOT_NAME
    # EWM GT 抽帧：原生帧率 → 目标帧率（stride = max(1, round(native/gt))，
    # 30→6 抽 1、10→2 抽 1、5→不抽），GT 与 pred 同为等效 5fps 逐帧对齐。
    ewmbenchmark_native_fps: int = 30
    ewmbenchmark_gt_fps: int = 5


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def parse_periodic_eval_config(config) -> PeriodicEvalConfig:
    """从训练 YAML 的 OmegaConf 解析 ``periodic_eval`` 节（缺失 → enabled=False）。"""
    section = config.get("periodic_eval") if config is not None else None
    if section is None:
        return PeriodicEvalConfig(enabled=False)
    section = OmegaConf.to_container(section, resolve=True) or {}

    weights_raw = section.get("source_weights") or {}
    weights = {str(k): float(v) for k, v in dict(weights_raw).items() if v is not None}

    counts_raw = section.get("holdout_family_counts") or {}
    holdout_family_counts = {
        str(k).lower(): max(0, int(v))
        for k, v in dict(counts_raw).items()
        if v is not None and str(k).strip()
    }

    ewm = section.get("ewmbenchmark") or {}

    return PeriodicEvalConfig(
        enabled=_as_bool(section.get("enabled"), False),
        interval_steps=max(1, int(section.get("interval_steps", 1000) or 1000)),
        num_inference_steps=(
            int(section["num_inference_steps"])
            if section.get("num_inference_steps") is not None
            else None
        ),
        batch_size=(
            max(1, int(section["batch_size"]))
            if section.get("batch_size") is not None
            else None
        ),
        seed=int(section.get("seed", 20260805)),
        vlm_generate=_as_bool(section.get("vlm_generate"), True),
        vlm_max_new_tokens=int(section.get("vlm_max_new_tokens", 128) or 128),
        fps=int(section.get("fps", 6) or 6),
        keep_rounds=(
            max(1, int(section["keep_rounds"]))
            if section.get("keep_rounds") is not None
            else None
        ),
        save_separate_videos=_as_bool(section.get("save_separate_videos"), False),
        gt_depth_action_arm=_as_bool(section.get("gt_depth_action_arm"), False),
        no_history_arm=_as_bool(section.get("no_history_arm"), False),
        a2v_arm=_as_bool(section.get("a2v_arm"), False),
        train_pool_fraction=min(
            1.0, max(0.0, float(section.get("train_pool_fraction", 0.0) or 0.0))
        ),
        holdout_family_counts=holdout_family_counts,
        cache_dir=resolve_output_path(section.get("cache_dir")) or None,
        source_weights=weights,
        ewmbenchmark_enabled=_as_bool(ewm.get("enabled"), False),
        ewmbenchmark_root=ewm.get("root") or None,
        ewmbenchmark_robot_name=str(ewm.get("robot_name") or EWMBENCHMARK_ROBOT_NAME),
        ewmbenchmark_native_fps=max(1, int(ewm.get("native_fps", 30) or 30)),
        ewmbenchmark_gt_fps=max(1, int(ewm.get("gt_fps", 5) or 5)),
    )


def resolve_num_inference_steps(pe_cfg: PeriodicEvalConfig, config) -> int:
    """eval 去噪步数：periodic_eval 覆盖 → model.inference.num_inference_timesteps → 50。"""
    if pe_cfg.num_inference_steps is not None:
        return int(pe_cfg.num_inference_steps)
    try:
        value = config.model.inference.num_inference_timesteps
        if value is not None:
            return int(value)
    except Exception:
        pass
    return 50


def resolve_guidance_scale(config) -> float:
    """CFG 引导强度（评审 #6）：model.inference.guidance_scale → 1.0（关闭）。"""
    try:
        value = config.model.inference.guidance_scale
        if value is not None:
            return float(value)
    except Exception:
        pass
    return 1.0


def resolve_action_inference_params(config) -> Dict[str, Any]:
    """action / depth 侧推理参数（修复 #1/#3b/#9c）：model.inference 下的
    action_guidance_scale（默认 1.0 = action 不做 CFG）、
    action_num_inference_timesteps（默认 None = 与 video 同步数）、
    action_temporal_smooth_alpha（默认 0.0 = 关闭时序平滑）、
    depth_temporal_smooth_alpha（默认 0.0 = 关闭深度时序平滑）。
    键缺失时全部回退默认（旧配置行为不变）。
    """
    params: Dict[str, Any] = {
        "action_guidance_scale": 1.0,
        "action_num_inference_steps": None,
        "action_smooth_alpha": 0.0,
        "depth_smooth_alpha": 0.0,
    }
    try:
        inference = config.model.inference
    except Exception:
        return params
    try:
        value = getattr(inference, "action_guidance_scale", None)
        if value is not None:
            params["action_guidance_scale"] = float(value)
        value = getattr(inference, "action_num_inference_timesteps", None)
        if value is not None:
            params["action_num_inference_steps"] = int(value)
        value = getattr(inference, "action_temporal_smooth_alpha", None)
        if value is not None:
            params["action_smooth_alpha"] = float(value)
        value = getattr(inference, "depth_temporal_smooth_alpha", None)
        if value is not None:
            params["depth_smooth_alpha"] = float(value)
    except Exception:
        pass
    return params


def resolve_source_weights(pe_cfg: PeriodicEvalConfig, config) -> List[Tuple[str, str, float]]:
    """评测源权重表：``[(source_name, kind, weight), ...]``，顺序固定。

    - indexed 源按 YAML ``dataset.sources`` 声明顺序，weight 缺省取训练权重
      （再缺省 1.0）；``source_weights`` 按名覆盖；权重 <= 0 的源被排除。
    - EWMBenchmark 排在最后；缺省权重 0.25。
    """
    entries: List[Tuple[str, str, float]] = []
    sources = []
    try:
        sources = list(config.dataset.sources)
    except Exception:
        sources = []
    for child in sources:
        name = str(child.get("name") or child.get("dataset"))
        default_weight = float(child.get("weight", 1.0) or 1.0)
        weight = float(pe_cfg.source_weights.get(name, default_weight))
        if weight <= 0:
            continue
        entries.append((name, "indexed", weight))
    if pe_cfg.ewmbenchmark_enabled and pe_cfg.ewmbenchmark_root:
        weight = float(pe_cfg.source_weights.get(EWMBENCHMARK_SOURCE_NAME, 0.25))
        if weight > 0:
            entries.append((EWMBENCHMARK_SOURCE_NAME, "ewmbenchmark", weight))
    return entries


def _source_family(source_name: str) -> str:
    """源名 → 家族（agibot_beta_dexterous_hand → agibot）。与 runner 同口径。"""
    name = str(source_name or "unknown")
    for family in ("agibot", "egoscale", "openvid", "ewmbenchmark", "libero", "egoverse"):
        if name.startswith(family):
            return family
    return name


def _largest_remainder_counts(
    entries: List[Tuple[str, str, float]], slots: int
) -> List[int]:
    """按权重把 slots 分到 entries，返回与 entries 等长的整数列表。"""
    if slots <= 0 or not entries:
        return [0] * len(entries)
    total_weight = sum(w for _, _, w in entries)
    if total_weight <= 0:
        return [0] * len(entries)
    quotas = [w / total_weight * slots for _, _, w in entries]
    counts = [int(q) for q in quotas]
    remainder = slots - sum(counts)
    order = sorted(range(len(entries)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _allocate(
    entries: List[Tuple[str, str, float]], slots: int, split: str
) -> List[SourceAssignment]:
    """把 slots 个 rank 按权重分到 entries（largest-remainder），纯函数。"""
    if slots <= 0 or not entries:
        return []
    counts = _largest_remainder_counts(entries, slots)
    assignments: List[SourceAssignment] = []
    for (name, kind, _), count in zip(entries, counts):
        for ordinal in range(count):
            assignments.append(
                SourceAssignment(
                    source_name=name, kind=kind, shard_ordinal=ordinal, split=split
                )
            )
    return assignments[:slots]


def _family_holdout_source_plan(
    entries: List[Tuple[str, str, float]],
    family_counts: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """holdout_family_counts → 每源 clip 数。未点名的源各留 1 条，避免默默丢掉。"""
    counts_by_source: Dict[str, int] = {}
    kinds: Dict[str, str] = {}
    for family, n_clips in family_counts.items():
        n_clips = int(n_clips)
        if n_clips <= 0:
            continue
        family_entries = [e for e in entries if _source_family(e[0]) == family]
        if not family_entries:
            raise ValueError(
                f"periodic_eval: holdout_family_counts[{family!r}]={n_clips} "
                f"but no positive-weight source in that family"
            )
        for (name, kind, _), count in zip(
            family_entries, _largest_remainder_counts(family_entries, n_clips)
        ):
            if count <= 0:
                continue
            counts_by_source[name] = counts_by_source.get(name, 0) + count
            kinds[name] = kind

    counted_families = {str(k).lower() for k, v in family_counts.items() if int(v) > 0}
    leftover = [e for e in entries if _source_family(e[0]) not in counted_families]
    for name, kind, _ in leftover:
        counts_by_source[name] = counts_by_source.get(name, 0) + 1
        kinds[name] = kind
    return counts_by_source, kinds


def _allocate_holdout_by_family(
    entries: List[Tuple[str, str, float]],
    slots: int,
    family_counts: Dict[str, int],
) -> List[SourceAssignment]:
    """按家族条数分配 holdout，再一对一打进 ``slots`` 个 rank。

    每个有正条数的源占一个 rank，``batch_size`` = 该源条数。源数必须等于
    ``slots``。``assign_sources`` 会先按源数扣住 holdout rank，再调用这里。
    """
    if slots <= 0:
        return []
    counts_by_source, kinds = _family_holdout_source_plan(entries, family_counts)
    ordered = [name for name, _, _ in entries if name in counts_by_source]
    if len(ordered) != slots:
        detail = ", ".join(f"{n}×{counts_by_source[n]}" for n in ordered)
        raise ValueError(
            f"periodic_eval: holdout_family_counts needs {len(ordered)} holdout "
            f"ranks (one per source: {detail}) but {slots} holdout ranks are "
            f"available; adjust train_pool_fraction or source_weights"
        )
    return [
        SourceAssignment(
            source_name=name,
            kind=kinds[name],
            shard_ordinal=0,
            split="holdout",
            batch_size=counts_by_source[name],
        )
        for name in ordered
    ]


def _spread_holdout_across_ranks(
    entries: List[Tuple[str, str, float]],
    counts_by_source: Dict[str, int],
    kinds: Dict[str, str],
    world_size: int,
) -> List[SourceAssignment]:
    """把各源 holdout 条数拆到全部 rank，全程 holdout、不重叠。"""
    ordered = [name for name, _, _ in entries if name in counts_by_source]
    if not ordered:
        raise ValueError("periodic_eval: holdout_family_counts produced no sources")
    rank_budget = _largest_remainder_counts(
        [(name, kinds[name], float(counts_by_source[name])) for name in ordered],
        world_size,
    )
    assignments: List[SourceAssignment] = []
    for name, n_ranks in zip(ordered, rank_budget):
        n_clips = int(counts_by_source[name])
        n_ranks = max(0, int(n_ranks))
        if n_ranks <= 0 or n_clips <= 0:
            continue
        extra = n_clips % n_ranks
        base = n_clips // n_ranks
        start = 0
        for r in range(n_ranks):
            count = base + (1 if r < extra else 0)
            if count <= 0:
                continue
            assignments.append(
                SourceAssignment(
                    source_name=name,
                    kind=kinds[name],
                    shard_ordinal=start,
                    split="holdout",
                    batch_size=count,
                    shard_as_start=True,
                )
            )
            start += count
    if not assignments:
        raise ValueError("periodic_eval: holdout spread produced no rank assignments")
    while len(assignments) < world_size:
        assignments.append(assignments[-1])
    return assignments[:world_size]


def assign_sources(
    weights: List[Tuple[str, str, float]],
    world_size: int,
    train_pool_fraction: float = 0.0,
    holdout_family_counts: Optional[Dict[str, int]] = None,
) -> List[SourceAssignment]:
    """把 world_size 个 rank 按权重确定性分配到评测源（largest-remainder）。

    返回长度 world_size 的列表，第 r 项是 rank r 的 (源, shard_ordinal, split)。
    同一源被分给 c 个 rank 时，按 rank 顺序依次取该源确定顺序的第
    0..c-1 个 batch。纯函数：所有 rank 各自计算结果一致，逐轮一致。

    ``train_pool_fraction > 0`` 时先按该比例切出末尾若干 rank 走训练池
    （``pool``），其余走 holdout（``eval_pool``）；两组各自独立做权重分配，
    因此组内配比仍然守权重。ewmbenchmark 只有 holdout 语义，不进 train 组。

    ``holdout_family_counts`` 非空时：
    - ``train_pool_fraction<=0``：全部 rank 走 holdout，把各源条数拆到
      多卡（``shard_as_start``），不再把多余 rank 塞进 train。
    - ``train_pool_fraction>0``：holdout 侧仍按家族条数、每源一个 rank，
      多出来的 rank 给 train（旧 Ego/Agi 混评行为）。
    """
    world_size = max(1, int(world_size))
    entries = [(name, kind, float(w)) for name, kind, w in weights if float(w) > 0]
    if not entries:
        # 兜底：无可用源时全部 rank 分到第一个声明的源（调用方保证 weights 非空）。
        raise ValueError("periodic_eval: no eval sources with positive weight")

    fraction = min(1.0, max(0.0, float(train_pool_fraction)))
    indexed = [e for e in entries if e[1] == "indexed"]
    train_slots = int(round(world_size * fraction)) if indexed else 0
    train_slots = min(train_slots, world_size)
    holdout_slots = world_size - train_slots

    counts = {
        str(k).lower(): int(v)
        for k, v in dict(holdout_family_counts or {}).items()
        if int(v) > 0
    }
    if counts:
        needed, kinds = _family_holdout_source_plan(entries, counts)
        n_holdout = len(needed)
        if n_holdout > world_size:
            detail = ", ".join(f"{n}×{needed[n]}" for n in needed)
            raise ValueError(
                f"periodic_eval: holdout_family_counts needs {n_holdout} holdout "
                f"ranks (one per source: {detail}) but world_size={world_size}"
            )
        if fraction <= 0:
            return _spread_holdout_across_ranks(entries, needed, kinds, world_size)
        holdout_slots = n_holdout
        train_slots = world_size - n_holdout
        assignments = _allocate_holdout_by_family(entries, holdout_slots, counts)
    else:
        assignments = _allocate(entries, holdout_slots, "holdout")
    assignments += _allocate(indexed, train_slots, "train")
    return assignments[:world_size]


def _file_stamp(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """spec 用的廉价文件指纹（size+mtime）；完整内容 sha256 由准备进程另存。"""
    if not path:
        return None
    try:
        stat = Path(path).stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def build_source_spec(
    config,
    pe_cfg: PeriodicEvalConfig,
    assignment: SourceAssignment,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    """构造一个 rank 评测 batch 的完整 spec（决定缓存目录名 spec_key）。

    覆盖所有影响 batch 内容的输入：源定义（catalog/index/pool 路径 + 文件
    指纹、reader/adapter options）、采样种子、batch 形状参数、shard 序号。
    """
    common = config.common
    spec: Dict[str, Any] = {
        "kind": assignment.kind,
        "source_name": assignment.source_name,
        "shard_ordinal": int(assignment.shard_ordinal),
        # split 进 spec：train / holdout 取的是不同池，缓存必须分开。
        "split": str(assignment.split),
        "seed": int(pe_cfg.seed),
        "batch_size": int(batch_size),
        "num_video_frames": int(common.num_video_frames),
        "video_action_freq_ratio": int(common.get("video_action_freq_ratio", 1)),
        "video_height": int(common.video_height),
        "video_width": int(common.video_width),
        "spec_version": 2,
        "shard_as_start": bool(assignment.shard_as_start),
    }
    if assignment.kind == "ewmbenchmark":
        manifest_path = (
            str(Path(pe_cfg.ewmbenchmark_root) / "manifest.jsonl")
            if pe_cfg.ewmbenchmark_root
            else None
        )
        spec["ewmbenchmark"] = {
            "root": pe_cfg.ewmbenchmark_root,
            "manifest": _file_stamp(manifest_path),
            "robot_name": pe_cfg.ewmbenchmark_robot_name,
            "state_dim": int(common.state_dim),
            # 借位占位声明（20260819 起写入缓存 meta，见 sources.py）。进 spec
            # 让旧缓存（meta 无此声明）spec_key 失效重备——仅 ewmbenchmark
            # 缓存重建，indexed 源不受影响。
            "state_action_policy": "placeholder",
            # GT 抽帧口径（20260820 起）：原生帧率 → 目标 5fps（stride =
            # max(1, round(native/gt))，30→6 抽 1）。进 spec 让旧缓存
            # （连续 30fps GT）spec_key 失效重备。
            "native_fps": int(pe_cfg.ewmbenchmark_native_fps),
            "gt_fps": int(pe_cfg.ewmbenchmark_gt_fps),
        }
        return spec

    child = None
    for candidate in config.dataset.sources:
        if str(candidate.get("name") or candidate.get("dataset")) == assignment.source_name:
            child = candidate
            break
    if child is None:
        raise ValueError(
            f"periodic_eval: source {assignment.source_name!r} not found in dataset.sources"
        )
    if assignment.split == "train":
        pool_spec = child.get("pool")
    else:
        pool_spec = child.get("eval_pool") or child.get("pool")
    pool_container = OmegaConf.to_container(pool_spec, resolve=True) if pool_spec else {}
    pool_path = pool_container.get("path") if isinstance(pool_container, dict) else None
    spec["indexed"] = {
        "dataset": str(child.get("dataset") or child.get("name")),
        "catalog_path": child.get("catalog_path"),
        "index_path": child.get("index_path"),
        "catalog": _file_stamp(child.get("catalog_path")),
        "index": _file_stamp(child.get("index_path")),
        "pool": pool_container,
        "pool_file": _file_stamp(pool_path),
        "reader": (
            OmegaConf.to_container(child.get("reader"), resolve=True)
            if child.get("reader") is not None
            else None
        ),
        "adapter_options": (
            OmegaConf.to_container(child.get("adapter_options"), resolve=True)
            if child.get("adapter_options") is not None
            else None
        ),
        "mix_adapter_options": {
            "num_video_frames": int(common.num_video_frames),
            "action_video_fps_ratio": int(common.get("video_action_freq_ratio", 1)),
        },
    }
    return spec


def spec_key(spec: Dict[str, Any]) -> str:
    """spec → 短哈希（缓存目录名）。canonical JSON 保证跨进程一致。"""
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_action_dim_labels(config) -> Dict[str, List[str]]:
    """从训练 YAML 构造每机器人**原生** action 逐维语义标签（eval 曲线标题用）。

    返回 ``{robot_name 或 alias: [label, ...]}``（长度 == 该机器人 action_dim）：

    - ``unified_dim_map`` 有效列 → 所属 ``unified_bands`` band 名 + band 内偏移
      （如 ``eef_left_pos[0]``；band 宽 1 时不带偏移，如 ``eef_left_gripper``）；
    - ``unified_rot_groups`` 接管的本地旋转维 → 目标 rot6 band 基名 + 本地表示
      （如 cols 落 ``eef_left_rot6``、repr=quat → ``eef_left_quat[0]``）；
    - dim_map 中的 -1 摘除维 → ``dim{k}(dropped)``；无 band 命中/无配置 →
      ``dim{k}`` 回退（调用方也可整体缺失时回退 ``dim {k}``）。

    纯函数、全防御：配置缺任何一节都返回 {} 或逐维回退，绝不抛异常。
    标签保持 ASCII（matplotlib 默认字体无 CJK，中文标题会渲染成方框）。
    """
    try:
        entries = OmegaConf.to_object(config.dataset.robot_names)
    except Exception:
        return {}
    if not isinstance(entries, dict):
        return {}
    try:
        raw_bands = OmegaConf.to_object(config.dataset.unified_bands) or {}
    except Exception:
        raw_bands = {}
    bands: List[Tuple[str, int, int]] = []
    if isinstance(raw_bands, dict):
        for band_name, span in raw_bands.items():
            try:
                bands.append((str(band_name), int(span[0]), int(span[1])))
            except Exception:
                continue

    def band_of(col: int) -> Optional[Tuple[str, int, int]]:
        for name, start, end in bands:
            if start <= col < end:
                return name, start, end
        return None

    def band_label(col: int) -> Optional[str]:
        hit = band_of(col)
        if hit is None:
            return None
        name, start, end = hit
        return name if end - start == 1 else f"{name}[{col - start}]"

    labels: Dict[str, List[str]] = {}
    for name, value in entries.items():
        if not isinstance(value, dict):
            continue
        try:
            dim = int(value.get("action_dim", 0) or 0)
        except Exception:
            continue
        if dim <= 0:
            continue
        dim_map = value.get("unified_dim_map")
        if not isinstance(dim_map, (list, tuple)):
            dim_map = None
        # rot group 本地维 → 目标 cols 所属 rot6 band 基名 + 本地 repr
        rot_labels: Dict[int, str] = {}
        for group in value.get("unified_rot_groups") or []:
            if not isinstance(group, dict):
                continue
            local = [int(v) for v in (group.get("local") or [])]
            cols = [int(c) for c in (group.get("cols") or [])]
            repr_name = str(group.get("repr") or "rot")
            base = None
            for col in cols:
                hit = band_of(col)
                if hit is not None:
                    base = hit[0]
                    break
            if base and base.endswith("_rot6"):
                base = f"{base[:-len('_rot6')]}_{repr_name}"
            elif base:
                base = f"{base}_{repr_name}"
            for j, ld in enumerate(local):
                rot_labels[ld] = f"{base}[{j}]" if base else f"{repr_name}[{j}]"
        try:
            offset = int(value.get("unified_offset", 0) or 0)
        except Exception:
            offset = 0
        dims: List[str] = []
        for k in range(dim):
            if k in rot_labels:
                dims.append(rot_labels[k])
                continue
            col: Optional[int] = None
            if dim_map is not None and k < len(dim_map):
                mapped = int(dim_map[k])
                if mapped < 0:
                    dims.append(f"dim{k}(dropped)")
                    continue
                col = mapped
            elif dim_map is None:
                col = offset + k
            label = band_label(col) if col is not None else None
            dims.append(label or f"dim{k}")
        keys = {str(name)}
        for alias in value.get("aliases") or []:
            keys.add(str(alias))
        for key in keys:
            labels[key] = dims
    return labels


def build_eef_display_layouts(config) -> Dict[str, Dict[str, Any]]:
    """从训练 YAML 构造每机器人 EEF 规范显示布局（eval 曲线/指标/meta.json 用）。

    返回 ``{robot_name 或 alias: {"slots": [...], "labels": [...]}}``。槽位按
    ``dataset.unified_bands`` 声明顺序逐臂排列，每臂固定 pos3 + rot6 6 +
    gripper1（``periodic_eval.render.actions_to_eef_display`` 消费）：

    - pos 槽：``{"kind": "pos", "local": k, "label": ...}``——dim_map 命中该
      pos band 列的本地维（按列排序、恰好填满 band 才采用），直通；
    - rot6 槽：``{"kind": "rot6", "local": [...], "repr": ..., "quat_order":
      ..., "component": j, "label": ...}``——cols 完整落在该 rot6 band 的
      第一个 rot_group，转 6D 后取第 j 分量；
    - gripper 槽：``{"kind": "gripper", "local": k, "available": True, ...}``；
      该臂 pos/rot6 在但 gripper band 无命中（无夹爪本体）→
      ``available=False`` 槽（显示为空白、指标跳过，标签加 ``(n/a)``）。

    ``eef_relative: false``（delta 动作源，如 libero）的 pos/rot6 标签加
    ``(delta)`` 后缀。无 ``unified_dim_map`` 或整机器人零 band 命中 →
    不进返回 dict（调用方回退 raw 逐维显示）。

    纯函数、全防御：配置缺任何一节都返回 {} 或逐机器人跳过，绝不抛异常。
    标签保持 ASCII（matplotlib 默认字体无 CJK）。
    """
    try:
        entries = OmegaConf.to_object(config.dataset.robot_names)
    except Exception:
        return {}
    if not isinstance(entries, dict):
        return {}
    try:
        raw_bands = OmegaConf.to_object(config.dataset.unified_bands) or {}
    except Exception:
        raw_bands = {}
    if not isinstance(raw_bands, dict):
        raw_bands = {}
    bands: List[Tuple[str, int, int]] = []
    for band_name, span in raw_bands.items():
        try:
            bands.append((str(band_name), int(span[0]), int(span[1])))
        except Exception:
            continue

    layouts: Dict[str, Dict[str, Any]] = {}
    for name, value in entries.items():
        if not isinstance(value, dict):
            continue
        dim_map = value.get("unified_dim_map")
        if not isinstance(dim_map, (list, tuple)):
            continue  # 无 dim_map 无法定位语义槽位（旧 offset 配置回退 raw 显示）
        try:
            cols_of_local = [int(c) for c in dim_map]
        except Exception:
            continue
        delta_suffix = "" if bool(value.get("eef_relative", True)) else "(delta)"
        rot_groups: List[Dict[str, Any]] = []
        for group in value.get("unified_rot_groups") or []:
            if not isinstance(group, dict):
                continue
            try:
                g_local = [int(v) for v in (group.get("local") or [])]
                g_cols = [int(c) for c in (group.get("cols") or [])]
            except Exception:
                continue
            if not g_local or len(g_cols) != 6:
                continue
            rot_groups.append(
                {
                    "local": g_local,
                    "cols": g_cols,
                    "repr": str(group.get("repr") or "rot6"),
                    "quat_order": str(group.get("quat_order") or "xyzw"),
                }
            )
        slots: List[Dict[str, Any]] = []
        arms_seen: set = set()
        for band_name, start, end in bands:
            if not band_name.startswith("eef_") or "_" not in band_name[len("eef_"):]:
                continue
            arm_part, kind = band_name[len("eef_"):].rsplit("_", 1)
            arm_key = f"eef_{arm_part}"
            if kind == "pos":
                hits = sorted(
                    (c, k)
                    for k, c in enumerate(cols_of_local)
                    if start <= c < end
                )
                if len(hits) != end - start:
                    continue  # pos band 缺维不凑假槽位
                for j, (_, k) in enumerate(hits):
                    slots.append(
                        {
                            "kind": "pos",
                            "local": k,
                            "available": True,
                            "label": f"{band_name}[{j}]{delta_suffix}",
                        }
                    )
                arms_seen.add(arm_key)
            elif kind == "rot6":
                group = next(
                    (
                        g
                        for g in rot_groups
                        if all(start <= c < end for c in g["cols"])
                    ),
                    None,
                )
                if group is None:
                    continue
                for j in range(6):
                    slots.append(
                        {
                            "kind": "rot6",
                            "local": list(group["local"]),
                            "repr": group["repr"],
                            "quat_order": group["quat_order"],
                            "component": j,
                            "available": True,
                            "label": f"{band_name}[{j}]{delta_suffix}",
                        }
                    )
                arms_seen.add(arm_key)
            elif kind == "gripper":
                hits = [
                    k for k, c in enumerate(cols_of_local) if start <= c < end
                ]
                if hits:
                    slots.append(
                        {
                            "kind": "gripper",
                            "local": hits[0],
                            "available": True,
                            "label": band_name,
                        }
                    )
                elif arm_key in arms_seen:
                    slots.append(
                        {
                            "kind": "gripper",
                            "local": None,
                            "available": False,
                            "label": f"{band_name}(n/a)",
                        }
                    )
        if not slots:
            continue
        layout = {
            "slots": slots,
            "labels": [str(slot["label"]) for slot in slots],
        }
        keys = {str(name)}
        for alias in value.get("aliases") or []:
            keys.add(str(alias))
        for key in keys:
            layouts[key] = layout
    return layouts
