"""评测产物渲染：compare 视频、meta.json（action 逐维曲线 20260820 起不再
落盘 actions.png，改由 monitor 前端 canvas 按 meta.json 的
``eef_display``/``gt_actions``/``predicted_actions`` 渲染，y 范围按 dim band
固定 + 网页端自定义）。

视频/图像 helpers 移植自 scripts/eval_video_ckpt.py（scripts/ 非包不可
import，按仓库惯例小量复制）。写盘约定：媒体先写 ``<name>.writing.<ext>``
再 ``os.replace`` 到最终名（monitor sync 只看最终名的 mtime/size，写一次
拉一次）；meta.json 走 monitor.store.atomic_write_json（.tmp 后缀，sync
不拉半成品）。
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def tensor_to_uint8_frames(frames: torch.Tensor) -> np.ndarray:
    """[T,C,H,W] 或 [C,T,H,W]（[0,1]）→ uint8 [T,H,W,C]。"""
    frames = frames.detach().float().cpu()
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape={tuple(frames.shape)}")
    if frames.shape[0] in (1, 3):
        frames = frames.permute(1, 2, 3, 0)
    elif frames.shape[1] in (1, 3):
        frames = frames.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Cannot infer channel dimension for shape={tuple(frames.shape)}")
    frames = torch.clamp(frames, 0.0, 1.0)
    return (frames.numpy() * 255.0).round().astype(np.uint8)


def image_to_uint8(frame: torch.Tensor) -> np.ndarray:
    """[C,H,W]（[0,1]）→ uint8 [H,W,C]。"""
    frame = frame.detach().float().cpu()
    if frame.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image tensor, got shape={tuple(frame.shape)}")
    if frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    frame = torch.clamp(frame, 0.0, 1.0)
    return (frame.numpy() * 255.0).round().astype(np.uint8)


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    """写视频（先写 ``<name>.writing.<ext>`` 再原子改名；imageio 优先，cv2 兜底）。

    编码按扩展名：``.mp4`` = 默认 h264 有损（compare 展示用）；``.mkv`` =
    FFV1 + bgr0 全色度**无损**（gt/pred 供像素级指标离线提取——h264 即使
    高质量也有损、yuv420p 还丢色度，深度面板的高频细节尤其经不起）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    lossless = ext == ".mkv"
    tmp = path.with_name(path.name + ".writing" + ext)
    try:
        import imageio.v2 as imageio

        kwargs: Dict[str, Any] = {"fps": float(fps), "macro_block_size": 1}
        if lossless:
            # bgr0 = 24bit RGB 无损（gbrp 在部分 ffmpeg 构建不被 ffv1 接受，
            # 会自动回退 bgr0；显式指定避免告警）。
            kwargs.update(codec="ffv1", pixelformat="bgr0")
        imageio.mimsave(str(tmp), list(frames), **kwargs)
    except Exception as exc:
        logger.warning("imageio video write failed for %s: %s; trying cv2", path, exc)
        try:
            import cv2

            height, width = frames.shape[1:3]
            writer = cv2.VideoWriter(
                str(tmp),
                cv2.VideoWriter_fourcc(*("FFV1" if lossless else "mp4v")),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception as cv_exc:
            raise RuntimeError(f"Failed to write video {path}") from cv_exc
    os.replace(tmp, path)


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, min(img.width, 280), 28), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(img)


def make_compare_frames(first: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """condition|gt|pred 三联屏（首帧即 condition，逐帧对齐 gt/pred）。"""
    rows = []
    for t in range(pred.shape[0]):
        cond = first.copy()
        gt_frame = gt[min(t, gt.shape[0] - 1)]
        pred_frame = pred[t]
        panels = [
            _label_frame(cond, "condition"),
            _label_frame(gt_frame, f"gt t+{t + 1}"),
            _label_frame(pred_frame, f"pred t+{t + 1}"),
        ]
        rows.append(np.concatenate(panels, axis=1))
    return np.stack(rows, axis=0)


def save_png(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing.png")
    Image.fromarray(array).save(str(tmp), format="PNG")
    os.replace(tmp, path)


def actions_to_eef_display(
    actions: Optional[Any],
    layout: Dict[str, Any],
) -> Optional[np.ndarray]:
    """raw local action [T, D] → EEF 规范显示数组 [T, N]（float32 numpy）。

    按 ``periodic_eval.config.build_eef_display_layouts`` 的槽位转换：
    pos/gripper 直通；rot6 槽经 ``models.action_repr`` 转 6D（quat 符号
    翻转在此天然消除——q 与 −q 是同一旋转矩阵）；``available=False`` 槽
    （如无夹爪本体）与任何转换失败槽填 NaN（matplotlib 断成空白、
    ``action_metrics`` 跳过该维）。actions 为 None 时返回 None。
    """
    if actions is None:
        return None
    from thinker_wm.models import action_repr  # 延迟 import：render 其余路径不依赖 models

    if isinstance(actions, np.ndarray):
        tensor = torch.from_numpy(np.asarray(actions, dtype=np.float32))
    else:
        tensor = actions.detach().float().cpu()
    if tensor.ndim != 2:
        raise ValueError(
            f"actions_to_eef_display expects [T, D], got shape={tuple(tensor.shape)}"
        )
    T, D = int(tensor.shape[0]), int(tensor.shape[1])
    slots = list(layout.get("slots") or [])
    out = torch.full((T, len(slots)), float("nan"), dtype=torch.float32)
    rot6_cache: Dict[tuple, torch.Tensor] = {}
    for idx, slot in enumerate(slots):
        if not isinstance(slot, dict) or not slot.get("available", True):
            continue
        kind = slot.get("kind")
        local = slot.get("local")
        if kind in ("pos", "gripper"):
            if local is None:
                continue
            k = int(local)
            if 0 <= k < D:
                out[:, idx] = tensor[:, k]
        elif kind == "rot6":
            try:
                locals_ = tuple(int(v) for v in (local or ()))
                if not locals_ or max(locals_) >= D:
                    continue
                key = (
                    locals_,
                    str(slot.get("repr") or "rot6"),
                    str(slot.get("quat_order") or "xyzw"),
                )
                rot6 = rot6_cache.get(key)
                if rot6 is None:
                    matrix = action_repr.repr_to_matrix(
                        tensor[:, list(locals_)], key[1], key[2]
                    )
                    rot6 = action_repr.matrix_to_6d(matrix)
                    rot6_cache[key] = rot6
                component = int(slot.get("component", 0))
                if 0 <= component < int(rot6.shape[-1]):
                    out[:, idx] = rot6[:, component]
            except Exception as exc:
                logger.warning(
                    "actions_to_eef_display: rot6 slot %r failed: %s",
                    slot.get("label"),
                    exc,
                )
    return out.numpy()


def write_meta_json(path: Path, payload: Dict[str, Any]) -> None:
    """meta.json：.tmp + rename 原子写（同步不拉半成品）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)
        fp.write("\n")
    os.replace(tmp, path)


def video_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """像素域 [0,1] 的 MSE/PSNR（截齐帧数）。pred/gt: [T,C,H,W]。"""
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    min_t = min(pred.shape[0], gt.shape[0])
    if min_t <= 0:
        return {}
    pred = pred[:min_t]
    gt = gt[:min_t]
    mse = float(((pred - gt) ** 2).mean().item())
    return {"video_mse": mse, "video_psnr": float(-10.0 * math.log10(max(mse, 1e-12)))}


def action_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    layout: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """action MSE（截齐长度/维度）。

    ``layout``（``build_eef_display_layouts`` 条目）提供时，pred/gt 视为
    EEF 显示空间数组：只在有效槽位上统计（``available=False`` 及含非有限
    值的维跳过；rot6 对 quat 符号翻转免疫），并附加分组指标
    ``action_mse_pos`` / ``action_mse_rot6`` / ``action_mse_gripper``
    （该组无有效维则不产出对应键）。缺省保持旧行为：raw 空间整体 MSE。
    """
    pred = torch.as_tensor(pred).detach().float().cpu()
    gt = torch.as_tensor(gt).detach().float().cpu()
    min_t = min(pred.shape[0], gt.shape[0])
    min_d = min(pred.shape[-1], gt.shape[-1])
    if min_t <= 0 or min_d <= 0:
        return {}
    pred = pred[:min_t, :min_d]
    gt = gt[:min_t, :min_d]
    if layout is None:
        diff = pred - gt
        return {"action_mse": float((diff**2).mean().item())}
    slots = list(layout.get("slots") or [])
    valid_dims: List[int] = []
    group_dims: Dict[str, List[int]] = {}
    for idx in range(min_d):
        slot = slots[idx] if idx < len(slots) else None
        if not isinstance(slot, dict) or not slot.get("available", True):
            continue
        if not torch.isfinite(torch.stack([pred[:, idx], gt[:, idx]])).all():
            continue
        valid_dims.append(idx)
        group_dims.setdefault(str(slot.get("kind") or "other"), []).append(idx)
    if not valid_dims:
        return {}
    cols = torch.tensor(valid_dims, dtype=torch.long)
    metrics = {
        "action_mse": float(((pred[:, cols] - gt[:, cols]) ** 2).mean().item())
    }
    for kind, dims in group_dims.items():
        group_cols = torch.tensor(dims, dtype=torch.long)
        metrics[f"action_mse_{kind}"] = float(
            ((pred[:, group_cols] - gt[:, group_cols]) ** 2).mean().item()
        )
    return metrics
