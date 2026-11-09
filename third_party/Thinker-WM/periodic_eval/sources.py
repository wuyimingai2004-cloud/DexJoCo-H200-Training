"""评测 batch 的数据来源：indexed 源（holdout eval_pool）与 EWMBenchmark。

本模块只在**后台准备进程**（periodic_eval.prepare）里执行重活（视频解码/
PNG 读取），产物经 periodic_eval.cache 落盘；训练前台永远不调用本模块的
取数函数。

确定性保证：
- indexed 源：跨 episode 多样化取批（data_prepare.create_diverse_eval_batch）
  ——seed 固定洗牌 episode 顺序、按 shard_ordinal 步进取 batch_size 条不同
  episode 各一窗口，跨次运行逐字节一致；
- EWMBenchmark：manifest 按 (task, episode, trial) 排序后按 shard 连续切片，
  固定不变。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def fetch_indexed_eval_batch(
    config,
    source_name: str,
    shard_ordinal: int,
    *,
    seed: int,
    batch_size: int,
    use_eval_pool: bool = True,
    start_index: Optional[int] = None,
) -> Dict[str, Any]:
    """从指定 indexed 源的 eval（holdout）池取跨 episode 多样化的固定 batch。

    ``use_eval_pool=False`` 时取训练池（``pool``），用于训练集同口径评测。

    委托 data_prepare.create_diverse_eval_batch（索引选择策略归 data_prepare
    所有）：seed 固定洗牌 episode 顺序，按 shard_ordinal 步进取 batch_size
    条不同 episode 各一个窗口，batch 内任务互不相同（episode 数不足时回绕并
    换窗口）。无 DataLoader，准备进程内直接解码，纯函数确定性可复现。
    """
    from thinker_wm.data_prepare import create_diverse_eval_batch

    return create_diverse_eval_batch(
        config,
        source_name,
        batch_size=int(batch_size),
        shard_ordinal=int(shard_ordinal),
        seed=int(seed),
        use_eval_pool=bool(use_eval_pool),
        start_index=start_index,
    )


def _load_frame(path: Path, width: int, height: int) -> "torch.Tensor":
    """读一张 PNG → [C,H,W] float [0,1]，必要时 resize 到 (height, width)。"""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        array = np.asarray(img).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def load_ewmbenchmark_batch(
    root: str,
    shard_ordinal: int,
    *,
    batch_size: int,
    num_video_frames: int,
    video_height: int,
    video_width: int,
    state_dim: int,
    robot_name: str,
    native_fps: int = 30,
    gt_fps: int = 5,
) -> Dict[str, Any]:
    """EWMBenchmark（/data/chen.liao/EWMBenchmark）固定 shard → batch dict。

    数据集：63 条 = 7 task × 3 episode × 3 trial；每条为 PNG 帧序列
    （640×480，155–297 帧，原生 30fps）+ official_instruction 文本；无
    action/state GT。排序 (task, episode, trial) 后第 shard_ordinal 个连续
    切片（回绕取满 batch_size），逐轮/跨任务固定。

    对齐训练语义：first_frame = 第 0 帧，GT video = 之后每 ``stride`` 帧抽
    1 帧共 N 个（N = num_video_frames，与预测帧一一对应）——``stride =
    max(1, round(native_fps / gt_fps))``（默认 30→5 即 6 抽 1，GT 与 pred
    同为等效 5fps 逐帧对齐，同 indexed 源 video_stride 布局的
    ``[s, s+r, …, s+N·r]`` 语义）；帧不足时重复末帧补齐。
    """
    root_path = Path(root)
    manifest_path = root_path / "manifest.jsonl"
    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"EWMBenchmark manifest empty: {manifest_path}")

    def _sort_key(rec: Dict[str, Any]):
        def _num(value):
            try:
                return (0, int(value))
            except (TypeError, ValueError):
                return (1, str(value))

        return (_num(rec.get("task_id")), _num(rec.get("episode_id")), _num(rec.get("trial_id")))

    records.sort(key=_sort_key)

    start = (int(shard_ordinal) * int(batch_size)) % len(records)
    chosen = [records[(start + i) % len(records)] for i in range(int(batch_size))]

    first_frames: List[torch.Tensor] = []
    gt_videos: List[torch.Tensor] = []
    pils: List[Any] = []
    texts: List[str] = []
    episode_ids: List[str] = []
    frame_indices: List[List[int]] = []
    from PIL import Image

    # GT 抽帧步长：原生帧率 → 目标帧率（30fps→6 抽 1 得等效 5fps，与 pred
    # 逐帧对齐；10fps→2、5fps→1 不抽）。
    stride = max(1, int(round(int(native_fps) / int(gt_fps))))

    for rec in chosen:
        frames_dir = root_path / str(rec["gt_frames_dir"])
        # first_frame = frame 0；GT = 之后每 stride 帧抽 1（与预测的 N 帧对齐）。
        wanted = [stride * k for k in range(0, int(num_video_frames) + 1)]
        frames: List[torch.Tensor] = []
        available = int(rec.get("num_frames") or 0)
        for idx in wanted:
            frame_idx = min(idx, max(0, available - 1))
            frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
            if not frame_path.exists():
                # 兜底：目录内排序后的第 frame_idx 张（命名不标准时）。
                candidates = sorted(frames_dir.glob("*.png"))
                if not candidates:
                    raise FileNotFoundError(f"no PNG frames under {frames_dir}")
                frame_path = candidates[min(frame_idx, len(candidates) - 1)]
            frames.append(_load_frame(frame_path, video_width, video_height))
        first_frames.append(frames[0])
        gt_videos.append(torch.stack(frames[1:], dim=0))  # [N,C,H,W]
        with Image.open(frames_dir / "frame_000.png") as img:
            pil = img.convert("RGB")
            if pil.size != (video_width, video_height):
                pil = pil.resize((video_width, video_height), Image.BILINEAR)
            pils.append(pil.copy())
        text = str(rec.get("official_instruction") or "").strip()
        texts.append(text)
        episode_ids.append(
            f"task{rec.get('task_id')}_episode{rec.get('episode_id')}_trial{rec.get('trial_id')}"
        )
        frame_indices.append(wanted)

    batch_size_actual = len(chosen)
    batch: Dict[str, Any] = {
        "first_frame": torch.stack(first_frames, dim=0),  # [B,C,H,W] float [0,1]
        "video_frames": torch.stack(gt_videos, dim=0),  # [B,N,C,H,W] float [0,1]
        "initial_state": torch.zeros(batch_size_actual, int(state_dim)),
        "action_sequence": None,  # 视频-only 基准：无 action GT
        "vlm_inputs": None,
        "language_embedding": None,
        "first_frame_pils": pils,
        "t5_input_messages": texts,
        "vlm_input_messages": list(texts),  # runner 按 unified_vlm_task_prompt 重建
        "robot_names": [str(robot_name)] * batch_size_actual,
        "source_datasets": ["ewmbenchmark"] * batch_size_actual,
        "episode_ids": episode_ids,
        "manifest_ids": episode_ids,
        "manifest_tasks": [f"task_{rec.get('task_id')}" for rec in chosen],
        "frame_indices": frame_indices,
        "action_video_fps_ratios": [1] * batch_size_actual,
        # 借位占位声明（batch 契约 state_action_policies）：state/action 无 GT
        # 语义、零 state 仅作锚点，但 action 照常预测——runner 据此向
        # inference_step 传 predict_action=True，覆盖"全 0 state=action 缺失"
        # 的零值启发式。
        "state_action_policies": ["placeholder"] * batch_size_actual,
        "video_only": True,
    }
    return batch
