#!/usr/bin/env python3
"""计算数据源的 action/state 逐维统计量（均值/方差，剔除离群值）。

对 Episode Catalog 逐 episode 读取 reader 归一化后的完整 state/action 时间线
（与训练侧所见完全一致），做两件事：

1. 精确的 raw 统计：逐维 count / mean / population std / min / max（Welford
   合并，多进程 worker 各自累计后归并）。
2. 剔除离群值后的 robust 统计：每维 reservoir 采样（有界内存、无偏），在
   合并后的采样上取 [q_low, q_high] 分位数作为边界，边界内的样本重算
   trimmed mean/std（同时记录中位数与分位数边界，便于人工核查）。

产物为 JSON，供模型侧归一化/反归一化烘焙（models/action_stats.py）使用。
统计键（state/action 分开；本仓库 robotwin / agibot_beta 的 state 与 action
来自同一时间线，二者统计量应一致）。

用法（与 build_sampling_ranges.py 同一套 split 语义，train 划分一致，避免
holdout 泄漏）：
    python data_prepare/compute_action_stats.py \
        --catalog <cat>.episode_catalog.jsonl \
        --dataset robotwin_aloha --robot aloha-agilex \
        --split train --holdout-fraction 0.01 \
        --workers 16 --output stats_robotwin_aloha.json [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thinker_wm.data_prepare import ensure_motus_data_importable  # noqa: E402

ensure_motus_data_importable()

import torch  # noqa: E402

from motus_data.releases import get_episode_dataset_release  # noqa: E402
from motus_data.repository import EpisodeRepository  # noqa: E402

from thinker_wm.data_prepare.sampling_ranges import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    _split_keeps,
)
from thinker_wm.data_prepare import robotwin_3d as _robotwin_3d  # noqa: E402,F401

DEFAULT_Q_LOW = 0.005
DEFAULT_Q_HIGH = 0.995
DEFAULT_MAX_SAMPLES_PER_DIM = 100_000

# ---------------------------------------------------------------------------
# 单进程统计累加器
# ---------------------------------------------------------------------------


class _StatsAccumulator:
    """对一条 modality 时间线的逐维统计累加（Welford + min/max + reservoir）。

    reservoir 形状 [D, capacity]（float32），n_seen 为已见行数（各维一致，
    因为一行即所有维的一次取值）。
    """

    def __init__(self, dim: int, capacity: int, seed: int):
        self.dim = dim
        self.capacity = capacity
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)
        self.min = torch.full((dim,), float("inf"), dtype=torch.float64)
        self.max = torch.full((dim,), float("-inf"), dtype=torch.float64)
        self.reservoir = torch.zeros(dim, capacity, dtype=torch.float32)
        self.n_seen = 0
        self._rng = torch.Generator().manual_seed(int(seed))

    def update(self, values: torch.Tensor) -> None:
        """values: [T, D] float32，逐行更新（非有限值整行丢弃）。"""
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(
                f"expected [T, {self.dim}] values, got {tuple(values.shape)}"
            )
        values = values.to(torch.float64)
        valid = torch.isfinite(values).all(dim=1)
        if not bool(valid.any()):
            return
        values = values[valid]
        t = values.shape[0]

        # Welford 在线更新（逐维）
        n = self.count + t
        delta = values.mean(dim=0) - self.mean
        self.mean = self.mean + delta * (t / n)
        m2_add = ((values - values.mean(dim=0)) ** 2).sum(dim=0)
        self.m2 = self.m2 + m2_add + delta**2 * (self.count * t / n)
        self.count = int(n)

        self.min = torch.minimum(self.min, values.min(dim=0).values)
        self.max = torch.maximum(self.max, values.max(dim=0).values)

        # reservoir（按行抽样的 Algorithm R，slot 选择对整行共享）
        k = self.n_seen + torch.arange(t, dtype=torch.long)
        cap = self.capacity
        fill = k < cap
        slots = torch.where(fill, k, torch.randint(0, cap, (t,), generator=self._rng))
        replace = torch.rand(t, generator=self._rng) * (k + 1) < cap
        use = fill | replace
        if bool(use.any()):
            self.reservoir[:, slots[use]] = values[use].t().to(torch.float32)
        self.n_seen += t

    def partial(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "min": self.min,
            "max": self.max,
            "reservoir": self.reservoir,
            "n_seen": self.n_seen,
        }


def _scan_episode_into(
    episode: Any,
    state_acc: _StatsAccumulator,
    action_acc: _StatsAccumulator,
    dim: int,
) -> None:
    """把单条 episode 的 state/action 时间线累加进两个 accumulator。"""
    state_count = int(episode.info.state_count)
    action_count = int(episode.info.action_count)
    if state_count <= 0 and action_count <= 0:
        return
    state, action = episode._reader.read_state_action(
        episode._record,
        range(state_count),
        range(action_count),
    )
    if state is not None and state.numel() > 0:
        state_acc.update(state.reshape(-1, dim))
    if action is not None and action.numel() > 0:
        action_acc.update(action.reshape(-1, dim))


# ---------------------------------------------------------------------------
# 多进程 worker（与 build_sampling_ranges.py 同款：每 worker 自持 repository；
# 每 worker 只返回一个累计 partial，避免逐 episode 传输/驻留 reservoir）
# ---------------------------------------------------------------------------

_SCAN_CONTEXT: dict = {}


def _scan_worker_init(
    catalog_path: str,
    index_path: str,
    dim: int,
    capacity: int,
    seed: int,
    reader_options: Optional[dict] = None,
) -> None:
    from motus_data.repository import EpisodeRepository
    from thinker_wm.data_prepare import robotwin_3d as _robotwin_3d  # noqa: F401

    _SCAN_CONTEXT.update(
        repository=EpisodeRepository(
            catalog_path,
            index_path=index_path,
            verify_sha256=False,
            reader_options=reader_options,
        ),
        dim=dim,
        capacity=capacity,
        seed=seed,
    )


def _scan_worker_chunk(episode_ids: list[str]) -> dict:
    """处理一批 episode，返回该 worker 的累计 state/action partial。"""
    ctx = _SCAN_CONTEXT
    dim = ctx["dim"]
    state_acc = _StatsAccumulator(dim, ctx["capacity"], ctx["seed"])
    action_acc = _StatsAccumulator(dim, ctx["capacity"], int(ctx["seed"]) + 1)
    skipped = 0
    for episode_id in episode_ids:
        episode = ctx["repository"].get_episode(str(episode_id))
        try:
            _scan_episode_into(episode, state_acc, action_acc, dim)
        except Exception:
            # catalog 可能含被 reader 数值质量策略排除的 episode
            # （如 agibot_beta:554:913085），统计按"可读即计入"跳过
            skipped += 1
    return {
        "state": state_acc.partial(),
        "action": action_acc.partial(),
        "skipped": skipped,
    }


def _scan_all(repository: Any, episode_ids: list[str], dim: int, capacity: int, seed: int) -> dict:
    """单进程扫描全部 episode（累计到一个 partial）。"""
    state_acc = _StatsAccumulator(dim, capacity, seed)
    action_acc = _StatsAccumulator(dim, capacity, int(seed) + 1)
    skipped = 0
    for episode_id in episode_ids:
        episode = repository.get_episode(str(episode_id))
        try:
            _scan_episode_into(episode, state_acc, action_acc, dim)
        except Exception:
            skipped += 1
    return {
        "state": state_acc.partial(),
        "action": action_acc.partial(),
        "skipped": skipped,
    }


def _split_chunks(items: list[str], num_chunks: int) -> list[list[str]]:
    """把 episode id 列表切分为 num_chunks 块（近似等长、保序）。"""
    if num_chunks <= 1:
        return [items]
    size = (len(items) + num_chunks - 1) // num_chunks
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# 统计归并 + 输出
# ---------------------------------------------------------------------------


def _merge_welford(
    acc_count: int,
    acc_mean: torch.Tensor,
    acc_m2: torch.Tensor,
    part_count: int,
    part_mean: torch.Tensor,
    part_m2: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    n = acc_count + part_count
    delta = part_mean - acc_mean
    mean = acc_mean + delta * (part_count / n)
    m2 = acc_m2 + part_m2 + delta**2 * (acc_count * part_count / n)
    return int(n), mean, m2


def _finalize_stats(partials: list[dict], q_low: float, q_high: float) -> dict:
    """归并多进程 partial 并输出每维统计（raw + trimmed + 分位数边界）。"""
    count = 0
    mean = torch.zeros_like(partials[0]["mean"])
    m2 = torch.zeros_like(partials[0]["m2"])
    minimum = torch.full_like(partials[0]["min"], float("inf"))
    maximum = torch.full_like(partials[0]["max"], float("-inf"))
    reservoirs = []
    n_seen = 0
    for part in partials:
        count, mean, m2 = _merge_welford(
            count, mean, m2, part["count"], part["mean"], part["m2"]
        )
        minimum = torch.minimum(minimum, part["min"])
        maximum = torch.maximum(maximum, part["max"])
        # 只取已填充的 reservoir 槽位（未填充槽为 0，会污染分位数/trimmed 统计）
        filled = min(int(part["n_seen"]), int(part["reservoir"].shape[1]))
        reservoirs.append(part["reservoir"][:, :filled])
        n_seen += part["n_seen"]

    raw_std = torch.sqrt(m2 / count) if count > 0 else torch.zeros_like(mean)
    merged = torch.cat(reservoirs, dim=1)  # [D, workers*cap]

    per_dim = []
    for d in range(merged.shape[0]):
        values = merged[d]
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            per_dim.append(
                {
                    "count": 0,
                    "mean": 0.0,
                    "std": 1.0,
                    "mean_raw": float(mean[d].item()),
                    "std_raw": float(raw_std[d].item()),
                    "q_low": 0.0,
                    "q_high": 0.0,
                    "median": 0.0,
                    "min": float(minimum[d].item()),
                    "max": float(maximum[d].item()),
                }
            )
            continue
        sorted_v, _ = torch.sort(values)
        q_low_v = float(
            torch.quantile(sorted_v, q_low, interpolation="linear").item()
        )
        q_high_v = float(
            torch.quantile(sorted_v, q_high, interpolation="linear").item()
        )
        median = float(torch.quantile(sorted_v, 0.5, interpolation="linear").item())
        inliers = values[(values >= q_low_v) & (values <= q_high_v)]
        if inliers.numel() == 0:
            trimmed_mean = 0.0
            trimmed_std = 1.0
        else:
            trimmed_mean = float(inliers.mean().item())
            trimmed_var = float(inliers.var(unbiased=False).item())
            trimmed_std = (
                float(math.sqrt(trimmed_var)) if trimmed_var > 0 else 1.0
            )
        per_dim.append(
            {
                "count": int(inliers.numel()),
                "mean": trimmed_mean,
                "std": trimmed_std,
                "mean_raw": float(mean[d].item()),
                "std_raw": float(raw_std[d].item()),
                "q_low": q_low_v,
                "q_high": q_high_v,
                "median": median,
                "min": float(minimum[d].item()),
                "max": float(maximum[d].item()),
            }
        )
    return {
        "count": count,
        "n_seen": n_seen,
        "dim": merged.shape[0],
        "per_dim": per_dim,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


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
        help="统计 JSON 输出路径（默认取 <catalog 去掉 .jsonl>.action_stats.json）",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="源数据集名（默认取 catalog 文件名 stem）",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="规范机器人名（默认取 catalog 首条记录的 robot 字段）",
    )
    parser.add_argument(
        "--q-low",
        type=float,
        default=DEFAULT_Q_LOW,
        help="离群值下分位数（缺省 0.005）",
    )
    parser.add_argument(
        "--q-high",
        type=float,
        default=DEFAULT_Q_HIGH,
        help="离群值上分位数（缺省 0.995）",
    )
    parser.add_argument(
        "--max-samples-per-dim",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_DIM,
        help="每维 reservoir 容量（缺省 100000）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行进程数（每 worker 自持 repository，map 保序）",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "holdout"),
        default="all",
        help="episode 划分（train/holdout 需配合 --holdout-fraction；"
        "默认 all 扫全部）",
    )
    parser.add_argument("--holdout-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多扫描的 episode 数（调试用）",
    )
    parser.add_argument("--seed", type=int, default=0, help="reservoir 随机种子")
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
        index = catalog.with_name(f"{stem}.index.sqlite")
    if not index.is_file():
        raise FileNotFoundError(f"catalog offset index not found: {index}")
    if args.output:
        output = Path(args.output)
    else:
        output = catalog.with_name(f"{stem}.action_stats.json")
    if output.is_file() and not args.force:
        raise FileExistsError(
            f"output exists: {output} (use --force to overwrite)"
        )
    dataset = args.dataset or stem.split(".")[0]

    # reader_options 解析（20260815，与 build_sampling_ranges.py 同款逻辑）：
    # profile/归一化绑定的 catalog（agibot_beta effector profile 等）的 reader
    # 在 bind_catalog 时强制要求 release reader_options（归一化 artifacts）；
    # 统计也必须读到归一化后的值才与运行时一致。按 catalog meta 的 child →
    # --dataset → 文件名 stem 顺序查 release 注册表；查不到保持 None（旧行为，
    # 无绑定要求的源不受影响）。
    reader_options = None
    meta_path = catalog.with_name(f"{stem}.meta.json")
    candidates: list[str] = []
    if meta_path.is_file():
        try:
            child = (json.loads(meta_path.read_text(encoding="utf-8")) or {}).get(
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

    repository = EpisodeRepository(
        catalog,
        index_path=index,
        verify_sha256=False,
        reader_options=reader_options,
    )
    try:
        episode_ids = list(repository.iter_episode_ids())
        if args.split != "all":
            episode_ids = [
                eid
                for eid in episode_ids
                if _split_keeps(
                    args.split,
                    str(eid),
                    holdout_fraction=args.holdout_fraction,
                    split_seed=int(args.split_seed),
                )
            ]
        if args.max_episodes is not None:
            episode_ids = episode_ids[: int(args.max_episodes)]

        if not episode_ids:
            raise RuntimeError("no episodes to scan (empty catalog or split)")

        robot = args.robot
        probe = repository.get_episode(str(episode_ids[0]))
        if robot is None:
            robot = str(probe.info.robot_name or "")
        dim = int(probe.info.modality_schemas["action"].item_shape[0])
        if dim <= 0:
            raise RuntimeError(f"cannot infer action dim from catalog: {dim}")

        capacity = int(args.max_samples_per_dim)
        seed = int(args.seed)
        if args.workers > 1:
            from multiprocessing import Pool

            chunks = _split_chunks(episode_ids, int(args.workers))
            pool = Pool(
                min(int(args.workers), len(chunks)),
                initializer=_scan_worker_init,
                initargs=(
                    str(catalog),
                    str(index),
                    dim,
                    capacity,
                    seed,
                    reader_options,
                ),
            )
            try:
                partials = pool.map(_scan_worker_chunk, chunks)
            finally:
                pool.close()
                pool.join()
        else:
            partials = [_scan_all(repository, episode_ids, dim, capacity, seed)]

        skipped = sum(int(p.get("skipped", 0)) for p in partials)
        if skipped >= len(episode_ids):
            raise RuntimeError(
                f"all {len(episode_ids)} episodes failed to read; aborting"
            )

        state_stats = _finalize_stats(
            [p["state"] for p in partials], args.q_low, args.q_high
        )
        action_stats = _finalize_stats(
            [p["action"] for p in partials], args.q_low, args.q_high
        )

        def _shape(stats: dict) -> dict:
            return {
                "count": stats["count"],
                "n_seen": stats["n_seen"],
                "dim": stats["dim"],
                "mean": [d["mean"] for d in stats["per_dim"]],
                "std": [d["std"] for d in stats["per_dim"]],
                "mean_raw": [d["mean_raw"] for d in stats["per_dim"]],
                "std_raw": [d["std_raw"] for d in stats["per_dim"]],
                "q_low": [d["q_low"] for d in stats["per_dim"]],
                "q_high": [d["q_high"] for d in stats["per_dim"]],
                "median": [d["median"] for d in stats["per_dim"]],
                "min": [d["min"] for d in stats["per_dim"]],
                "max": [d["max"] for d in stats["per_dim"]],
                "inlier_count": [d["count"] for d in stats["per_dim"]],
            }

        payload = {
            "dataset": dataset,
            "robot": robot,
            "catalog_id": str(repository.catalog_id),
            "episodes": len(episode_ids),
            "episodes_skipped": skipped,
            "split": args.split,
            "holdout_fraction": args.holdout_fraction,
            "q_low": args.q_low,
            "q_high": args.q_high,
            "max_samples_per_dim": capacity,
            "dim": dim,
            "state": _shape(state_stats),
            "action": _shape(action_stats),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(output)
        print(
            f"STATS_OK dataset={dataset} robot={robot} catalog_id={payload['catalog_id']} "
            f"episodes={payload['episodes']} skipped={skipped} dim={dim} "
            f"state_rows={state_stats['count']} action_rows={action_stats['count']} "
            f"split={args.split} output={output}",
            flush=True,
        )
        return 0
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
