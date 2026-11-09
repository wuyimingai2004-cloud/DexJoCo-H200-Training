"""PeriodicEvaluator：训练中周期性定集评测的训练侧接口。

工作方式（设计要点见 AGENTS.md「周期性评测」节）：

- ``__init__``：按权重 largest-remainder 算出本 rank 的固定评测 batch
  （源 + shard 序号，跨轮次/跨任务一致）；缓存缺失则拉起 CPU-only 后台
  子进程（``python -m periodic_eval.prepare``）预解码素材到
  ``<cache_dir>/<spec_key>/``（带 manifest + sha256，跨训练任务复用）。
- ``maybe_run(model, global_step)``：每 ``interval_steps`` 步执行一轮——
  仅当缓存就绪：load 缓存（秒级，前台零解码）→ ``model.inference_step``
  → 可选 VLM generate 出文本 → 指标 → 媒体落盘（monitor run_dir 下
  ``eval/step_xxx/rankR/sampleK/``）→ 跨 rank 汇聚索引记录 → rank0 追加
  eval JSONL 索引流。缓存未就绪则跳过本轮（集合通信仍参与，不会死锁）。
- 全流程防御性 try/except：任何失败只记日志 + monitor 事件，绝不影响训练。
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from . import cache as cache_mod
from .config import (
    PeriodicEvalConfig,
    SourceAssignment,
    assign_sources,
    build_action_dim_labels,
    build_eef_display_layouts,
    build_source_spec,
    parse_periodic_eval_config,
    resolve_action_inference_params,
    resolve_guidance_scale,
    resolve_num_inference_steps,
    resolve_source_weights,
    spec_key,
)

logger = logging.getLogger(__name__)


def _robotwin_depth_required(config: Any, meta: Optional[Dict[str, Any]] = None, source_name: Optional[str] = None) -> bool:
    """V10 这类任务：enable_depth + robotwin_3d_root，且 batch 是 robotwin_3d。

    为真时评测必须挂上深度，不能再静默降级成只有 RGB/action。
    """
    wm = getattr(getattr(config, "model", None), "wm", None)
    if not bool(getattr(wm, "enable_depth", False)):
        return False
    depth_cfg = config.get("depth") if hasattr(config, "get") else getattr(config, "depth", None)
    inline = None
    if depth_cfg is not None:
        inline = depth_cfg.get("robotwin_3d_root") if hasattr(depth_cfg, "get") else getattr(depth_cfg, "robotwin_3d_root", None)
    if not inline or not str(inline).strip():
        return False
    sources = list((meta or {}).get("source_datasets") or [])
    eids = list((meta or {}).get("episode_ids") or [])
    if source_name:
        sources.append(source_name)
    return any(str(s) == "robotwin_3d" for s in sources) or any(
        str(e).startswith("robotwin_3d:") for e in eids
    )


def _slice_eval_item(value: Any, start: int, end: int) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[start:end]
    if isinstance(value, (list, tuple)):
        return value[start:end]
    if isinstance(value, dict):
        return {k: _slice_eval_item(v, start, end) for k, v in value.items()}
    return value


def _with_blocks(score_fn, pred, gt, blocks: int, width: int, suffix: str = "") -> Dict[str, Any]:
    """整段 + 逐块（``_b1.._bN``）指标；suffix 区分对照臂。"""
    out = {f"{k}{suffix}": v for k, v in score_fn(pred, gt).items()}
    for b in range(blocks):
        sl = slice(b * width, (b + 1) * width)
        out.update({f"{k}_b{b + 1}{suffix}": v for k, v in score_fn(pred[sl], gt[sl]).items()})
    return out


def _cat_eval_output(chunks: List[Any]) -> Any:
    present = [item for item in chunks if item is not None]
    if not present:
        return None
    if torch.is_tensor(present[0]):
        return torch.cat(present, dim=0)
    return present


def _monitor_event(monitor: Any, kind: str, **fields: Any) -> None:
    """与 train.py 的 _monitor_call 同款容错事件上报。"""
    if monitor is None:
        return
    try:
        monitor.log_event(kind, **fields)
    except Exception as exc:
        logger.warning("thinker_wm.monitor.log_event(%s) failed (ignored): %s", kind, exc)


class NullPeriodicEvaluator:
    """periodic_eval 未启用/初始化失败时的 no-op 替身。"""

    enabled = False

    def maybe_run(self, model: Any, global_step: int) -> Optional[Dict[str, Any]]:
        return None

    def close(self) -> None:
        return None


class PeriodicEvaluator:
    def __init__(
        self,
        config: Any,
        pe_cfg: PeriodicEvalConfig,
        *,
        rank: int,
        world_size: int,
        monitor: Any = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.config = config
        self.cfg = pe_cfg
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.monitor = monitor
        self.enabled = bool(pe_cfg.enabled)
        self.num_inference_steps = resolve_num_inference_steps(pe_cfg, config)
        self.guidance_scale = resolve_guidance_scale(config)
        # action 侧推理参数（修复 #1/#3b/#9c）：action CFG（默认关）、独立
        # 去噪步数、时序平滑；旧配置缺键时全部回退默认（行为不变）。
        self.action_inference_params = resolve_action_inference_params(config)
        self._prepare_proc: Optional[subprocess.Popen] = None
        self._prepare_attempts = 0
        self._max_prepare_attempts = 3
        self._cache_verified = False
        self._eval_writer = None  # rank0 懒创建的 ChunkedJsonlWriter
        # 防静默回归哨兵：未声明 state_action_policies 且 state 全 0 的源，
        # 按源名去重只 warning 一次（见 _run_local_eval）
        self._warned_zero_state: set = set()
        # depth attacher 懒建（enable_depth 关或 sidecar 不可用时恒为 None）
        self._depth_attacher_built = False
        self._depth_attacher_obj = None
        self._history_attacher_built = False
        self._history_attacher_obj = None
        # 每机器人原生 action 逐维语义标签（dataset.robot_names+unified_bands；
        # eval 曲线标题/meta.json 用，缺失时回退 dim{k}）
        try:
            self._action_dim_labels: Dict[str, List[str]] = build_action_dim_labels(config)
        except Exception as exc:
            logger.warning("build_action_dim_labels failed (ignored): %s", exc)
            self._action_dim_labels = {}
        # 每机器人 EEF 规范显示布局（每臂 3pos+6rot+1gripper；命中时 eval
        # 曲线/指标/meta.json 的 eef_display 均在显示空间——rot6 对 quat
        # 符号翻转免疫；未命中的机器人回退 raw local 逐维显示）
        try:
            self._eef_display_layouts: Dict[str, Dict[str, Any]] = (
                build_eef_display_layouts(config)
            )
        except Exception as exc:
            logger.warning("build_eef_display_layouts failed (ignored): %s", exc)
            self._eef_display_layouts = {}

        if not self.enabled:
            return

        weights = resolve_source_weights(pe_cfg, config)
        if not weights:
            raise ValueError("periodic_eval enabled but no eval sources resolved")
        assignments = assign_sources(
            weights,
            self.world_size,
            pe_cfg.train_pool_fraction,
            holdout_family_counts=pe_cfg.holdout_family_counts,
        )
        self.assignment: SourceAssignment = assignments[self.rank]
        # 评测缓存 / 本 rank 条数：assignment 上的家族条数优先，否则 YAML
        # 显式覆盖，再否则 = 训练 batch_size 的 1/4（最小 1）。
        default_bs = (
            int(pe_cfg.batch_size)
            if pe_cfg.batch_size is not None
            else max(1, int(config.training.batch_size) // 4)
        )
        self.batch_size = (
            int(self.assignment.batch_size)
            if self.assignment.batch_size is not None
            else default_bs
        )
        # 推理切块：holdout_family_counts 会把本 rank 条数写成 16～17，
        # 整批推进去显存会高于训练 bs。YAML 的 batch_size 在这里只当
        # 单次 inference_step 的上限，不改缓存、不丢 clip。
        self.infer_chunk_size = (
            int(pe_cfg.batch_size)
            if pe_cfg.batch_size is not None
            else min(self.batch_size, max(1, int(config.training.batch_size)))
        )
        self.spec = build_source_spec(
            config, pe_cfg, self.assignment, batch_size=self.batch_size
        )
        self.spec_key = spec_key(self.spec)

        cache_root = pe_cfg.cache_dir
        if not cache_root:
            cache_root = str(Path(checkpoint_dir or ".") / "periodic_eval_cache")
            logger.warning(
                "thinker_wm.periodic_eval.cache_dir 未配置，回退到 %s（跨任务不可复用）", cache_root
            )
        self.cache_root = Path(cache_root)
        self.cache_dir = self.cache_root / self.spec_key

        # 媒体/索引根：monitor run_dir（sync 可见）优先；monitor 关闭时回退
        # checkpoint_dir 下的 periodic_eval/（仅留档，Web UI 看不到）。
        run_dir = getattr(monitor, "run_dir", None) if monitor is not None else None
        if run_dir:
            self.media_root = Path(run_dir)
        else:
            self.media_root = Path(checkpoint_dir or ".") / "periodic_eval"

        logger.info(
            "periodic_eval: rank %d/%d -> source=%s kind=%s split=%s shard=%d "
            "interval=%d batch_size=%d infer_chunk=%d inference_steps=%d "
            "guidance_scale=%.2f action_inference=%s cache=%s",
            self.rank,
            self.world_size,
            self.assignment.source_name,
            self.assignment.kind,
            self.assignment.split,
            self.assignment.shard_ordinal,
            self.cfg.interval_steps,
            self.batch_size,
            self.infer_chunk_size,
            self.num_inference_steps,
            self.guidance_scale,
            self.action_inference_params,
            self.cache_dir,
        )
        _monitor_event(
            self.monitor,
            "periodic_eval_init",
            source=self.assignment.source_name,
            split=self.assignment.split,
            shard=self.assignment.shard_ordinal,
            interval_steps=self.cfg.interval_steps,
            cache_dir=str(self.cache_dir),
        )
        if not cache_mod.cache_is_complete(self.cache_dir):
            self._spawn_prepare()

    # ------------------------------------------------------------------ 准备进程

    def _spawn_prepare(self) -> None:
        """拉起 CPU-only 后台准备进程（幂等；缓存已就绪则不起）。"""
        if self._prepare_attempts >= self._max_prepare_attempts:
            return
        try:
            support_dir = self.cache_root / "_prepare"
            support_dir.mkdir(parents=True, exist_ok=True)
            config_dump = support_dir / f"{self.spec_key}.yaml"
            spec_json = support_dir / f"{self.spec_key}.spec.json"
            log_path = support_dir / f"{self.spec_key}.log"
            if not config_dump.exists():
                tmp = config_dump.with_suffix(".yaml.tmp")
                with tmp.open("w", encoding="utf-8") as fp:
                    fp.write(OmegaConf.to_yaml(self.config))
                os.replace(tmp, config_dump)
            if not spec_json.exists():
                import json

                tmp = spec_json.with_suffix(".json.tmp")
                with tmp.open("w", encoding="utf-8") as fp:
                    json.dump(self.spec, fp, ensure_ascii=False, indent=2, default=str)
                os.replace(tmp, spec_json)

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""  # 数据解码纯 CPU，不占训练 GPU
            env["MOTUS_MONITOR_ENABLED"] = "0"
            log_fp = log_path.open("a", buffering=1)
            self._prepare_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "thinker_wm.periodic_eval.prepare",
                    "--config-dump",
                    str(config_dump),
                    "--spec-json",
                    str(spec_json),
                    "--out-dir",
                    str(self.cache_dir),
                ],
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            )
            self._prepare_attempts += 1
            logger.info(
                "periodic_eval: prepare process spawned (pid=%d, attempt %d/%d, log=%s)",
                self._prepare_proc.pid,
                self._prepare_attempts,
                self._max_prepare_attempts,
                log_path,
            )
        except Exception as exc:
            logger.warning("periodic_eval: failed to spawn prepare process: %s", exc)
            self._prepare_proc = None

    # ------------------------------------------------------------------ depth
    def _depth_attacher(self):
        """懒建 DepthAttacher（仅 enable_depth + 配置了 depth.roots 时）。

        评测缓存的 meta 已含 ``episode_ids`` 与 ``frame_indices``（见
        cache.META_LIST_KEYS），所以深度可以在训练前台直接按 sidecar 补齐，
        不需要重新解码视频；代价是每轮几秒的 zarr 读取。任何失败都降级为
        "本轮无 depth"，不影响 RGB/action 评测。
        """
        if self._depth_attacher_built:
            return self._depth_attacher_obj
        self._depth_attacher_built = True
        try:
            if not bool(getattr(self.config.model.wm, "enable_depth", False)):
                return None
            from omegaconf import OmegaConf

            from thinker_wm.data_prepare.depth_attach import DepthAttacher

            depth_cfg = self.config.get("depth")
            raw = (
                OmegaConf.to_container(depth_cfg, resolve=True)
                if depth_cfg is not None
                else {}
            ) or {}
            roots = list(raw.get("roots") or [])
            inline = raw.get("robotwin_3d_root")
            inline = str(inline) if inline else None
            if not roots and not inline:
                return None

            self._depth_attacher_obj = DepthAttacher(
                roots,
                z_near=float(raw.get("z_near", 0.1)),
                z_far=float(raw.get("z_far", 10.0)),
                mode=str(raw.get("mode", "log")),
                invalid_to=str(raw.get("invalid_to", "far")),
                prod_keep_only=bool(raw.get("prod_keep_only", True)),
                robotwin_3d_root=inline,
            )
            logger.info(
                "periodic_eval: depth attacher ready (roots=%d robotwin_3d_root=%s)",
                len(roots),
                inline,
            )
        except Exception as exc:
            logger.warning("periodic_eval: depth attacher unavailable (%s)", exc)
            self._depth_attacher_obj = None
        return self._depth_attacher_obj

    def _history_attacher(self):
        """懒建 HistoryAttacher（model.wm.history_enable 时），与训练同一构造器；
        失败直接抛出：主评估必须与部署一样带历史，不能静默退化成无历史。"""
        if not self._history_attacher_built:
            self._history_attacher_built = True
            from thinker_wm.data_prepare.history_attach import build_history_attacher

            self._history_attacher_obj = build_history_attacher(self.config)
        return self._history_attacher_obj

    def _depth_panels_and_metrics(self, render, predicted_depth, depth_gt, sample_idx, extra_rgb=None,
                                  depth_valid_mask=None, block_frames=0):
        """一个样本的 depth 三联屏帧 + 指标；depth 缺失/异常时返回 None。"""
        if predicted_depth is None or depth_gt is None:
            return None

        def luma(x: torch.Tensor) -> torch.Tensor:
            """深度是灰度语义：压成亮度供渲染，避免三通道分歧被渲成彩噪。"""
            return x.mean(dim=-3, keepdim=True).expand_as(x)

        try:
            pred_d = predicted_depth[sample_idx].permute(1, 0, 2, 3)  # [T,3,H,W]
            gt_d = depth_gt[sample_idx]  # [T,3,H,W]
            n = min(pred_d.shape[0], gt_d.shape[0])
            if n <= 0:
                return None
            pred_d, gt_d = pred_d[:n], gt_d[:n]
            from thinker_wm.models.depth_output import depth_content_mask, mask_decoded_depth
            content_mask = None
            # Explicit reader geometry only: never guess padding from RGB/GT values.
            if self.assignment.source_name == "robotwin_3d":
                sources = self.config.get("dataset", {}).get("sources", [])
                for source in sources:
                    if source.get("name") == "robotwin_3d" and source.get("reader", {}).get("resize_mode") == "pad":
                        h, w = pred_d.shape[-2:]
                        content_mask = depth_content_mask(f"robotwin_3d/motus_t/pad/{h}x{w}", (h, w), device=pred_d.device)
                        break
            pred_d = mask_decoded_depth(pred_d, content_mask)
            gt_d = mask_decoded_depth(gt_d, content_mask)
            panel = render.make_compare_frames(
                render.image_to_uint8(luma(gt_d[0])),
                render.tensor_to_uint8_frames(luma(gt_d)),
                render.tensor_to_uint8_frames(luma(pred_d)),
            )

            from thinker_wm.models.depth_codec import decode_to_depth_m

            depth_cfg = self.config.get("depth")
            codec = {
                "z_near": float(getattr(depth_cfg, "z_near", 0.1)),
                "z_far": float(getattr(depth_cfg, "z_far", 10.0)),
                "mode": str(getattr(depth_cfg, "mode", "log")),
                "in_range": "01",
            }
            pred_m, _ = decode_to_depth_m(pred_d, **codec)
            gt_m, valid = decode_to_depth_m(gt_d, **codec)
            if depth_valid_mask is not None:
                valid = depth_valid_mask[sample_idx, :n].to(device=gt_m.device, dtype=torch.bool)
                if valid.shape != gt_m.shape:
                    raise ValueError("depth metric validity mask is not aligned with GT depth")
            if content_mask is not None:
                valid = valid & content_mask.to(valid.device)
            scores = {}
            rel = (pred_m - gt_m).abs() / gt_m.clamp_min(1e-6)
            if bool(valid.any()):
                scores["depth_gray_mse"] = float(((pred_d - gt_d) ** 2).mean(dim=1)[valid].mean())
                scores["depth_absrel"] = float(rel[valid].mean().item())
                scores["depth_mse_m2"] = float(((pred_m - gt_m)[valid] ** 2).mean().item())
            for b in range(n // block_frames if block_frames else 0):
                sl = slice(b * block_frames, (b + 1) * block_frames)
                if bool(valid[sl].any()):
                    scores[f"depth_absrel_b{b + 1}"] = float(rel[sl][valid[sl]].mean().item())
            rgb = None
            if extra_rgb is not None:
                rgb = extra_rgb[:n]
            motion = _depth_motion_mask(gt_m, valid, thresh_m=0.10, rgb=rgb)
            mot = valid & motion
            bg = valid & ~motion
            scores["depth_motion_frac"] = float(mot.sum() / valid.sum().clamp_min(1))
            if bool(mot.any()):
                scores["depth_absrel_motion"] = float(rel[mot].mean().item())
            if bool(bg.any()):
                scores["depth_absrel_background"] = float(rel[bg].mean().item())
            structure_valid = valid
            if content_mask is not None:
                # Sobel/Laplacian use 3x3 neighborhoods: exclude the one-pixel
                # band that would read padding, not just padding centers.
                near_padding = torch.nn.functional.max_pool2d(
                    (~content_mask).float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
                structure_valid = valid & ~near_padding.to(valid.device)
            scores.update(
                _depth_structure_metrics(
                    pred_m,
                    gt_m,
                    structure_valid,
                    bg & structure_valid,
                    z_near=codec["z_near"],
                    z_far=codec["z_far"],
                )
            )
            return panel, scores
        except Exception as exc:
            logger.warning("periodic_eval: depth panel/metrics failed (ignored): %s", exc)
            return None

    def _cache_ready(self) -> bool:
        """就绪判定：首次做完整 sha256 校验，之后记忆（内容落盘后不可变）。"""
        if self._cache_verified:
            return True
        if not cache_mod.cache_is_complete(self.cache_dir):
            return False
        if cache_mod.verify_cache(self.cache_dir):
            self._cache_verified = True
            return True
        logger.warning("periodic_eval: cache verification failed: %s", self.cache_dir)
        return False

    def _poll_prepare(self) -> None:
        """准备进程退出但缓存未就绪 → 有限次重拉。"""
        if self._prepare_proc is None:
            if self._prepare_attempts < self._max_prepare_attempts:
                self._spawn_prepare()
            return
        retcode = self._prepare_proc.poll()
        if retcode is None:
            return
        logger.warning(
            "periodic_eval: prepare process exited with code %s but cache not ready: %s",
            retcode,
            self.cache_dir,
        )
        self._prepare_proc = None
        if self._prepare_attempts < self._max_prepare_attempts:
            self._spawn_prepare()
        else:
            _monitor_event(
                self.monitor,
                "periodic_eval_prepare_failed",
                cache_dir=str(self.cache_dir),
                attempts=self._prepare_attempts,
            )

    # ------------------------------------------------------------------ 评测主流程

    def maybe_run(self, model: Any, global_step: int) -> Optional[Dict[str, Any]]:
        """训练循环 hook：到间隔步执行一轮评测；返回并入 step_record 的标量。"""
        if not self.enabled:
            return None
        if global_step <= 0 or global_step % self.cfg.interval_steps != 0:
            return None

        records: List[Dict[str, Any]] = []
        local_summary: Dict[str, Any] = {
            "source": self.assignment.source_name,
            "kind": self.assignment.kind,
            "ready": False,
        }
        if not self._cache_ready():
            self._poll_prepare()
            logger.info(
                "periodic_eval: step %d skipped (cache not ready: %s)",
                global_step,
                self.cache_dir,
            )
            local_summary["skip_reason"] = "cache_pending"
        else:
            try:
                records, local_summary = self._run_local_eval(model, global_step)
                local_summary["ready"] = True
            except Exception as exc:
                logger.exception("periodic_eval: eval round failed at step %d", global_step)
                local_summary["skip_reason"] = f"error: {exc!r}"
                _monitor_event(
                    self.monitor,
                    "periodic_eval_error",
                    step=int(global_step),
                    error=repr(exc),
                )

        # 跨 rank 汇聚（所有 rank 都到这一步；未就绪的 rank 贡献空记录，不会死锁）。
        gathered: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            gathered = [None] * self.world_size  # type: ignore[list-item]
            dist.all_gather_object(gathered, (records, local_summary))
        else:
            gathered = [(records, local_summary)]

        scalars: Dict[str, Any] = {}
        if self.rank == 0:
            scalars = self._write_round(global_step, gathered)
        return scalars

    def _run_local_eval(
        self, model: Any, global_step: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """本 rank 的一轮评测：load 缓存 → 推理 → VLM 文本 → 指标 → 媒体。"""
        from . import render

        round_start = time.perf_counter()
        data = cache_mod.load_cache(self.cache_dir)
        tensors = data["tensors"]
        pils = data["pils"]
        meta = data["meta"]
        num_samples = int(tensors["first_frame"].shape[0])

        task_texts: List[str] = [str(t) for t in meta.get("t5_input_messages") or []]
        vlm_messages: List[str] = [str(t) for t in meta.get("vlm_input_messages") or []]
        # prompt_mode=text_only 的源（openvid/egoverse 等）：原始文本纯文本
        # 编码（不带首帧图），与训练侧 train.py prepare_train_batch 一致。
        from thinker_wm.utils.vlm_utils import resolve_prompt_mode, PROMPT_MODE_TEXT_ONLY

        text_only_prompt = (
            resolve_prompt_mode(self.config, [self.assignment.source_name])
            == PROMPT_MODE_TEXT_ONLY
        )
        if getattr(self.config.model, "unified_vlm_task_prompt", False) and task_texts:
            # 与训练侧一致（train.py prepare_train_batch）：统一 task prompt 重建
            from thinker_wm.utils.vlm_utils import build_vlm_rewrite_message, build_vlm_task_message

            if text_only_prompt:
                vlm_messages = [str(text).strip() for text in task_texts]
            elif bool(getattr(self.config.model, "vlm_text_rewrite", False)):
                # vlm_text_rewrite 开启：与训练侧一致改用压缩改写指令消息
                # （纯文本、不带首帧图；按数据集家族分模板）。
                vlm_messages = [
                    build_vlm_rewrite_message(text, self.assignment.source_name)
                    for text in task_texts
                ]
            else:
                from thinker_wm.utils.vlm_utils import resolve_vlm_prompt_style

                vlm_messages = [
                    build_vlm_task_message(
                        text, style=resolve_vlm_prompt_style(self.config)
                    )
                    for text in task_texts
                ]
        robot_names = meta.get("robot_names") or None
        video_only = bool(meta.get("video_only", False))

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.bfloat16
        first_frame = (tensors["first_frame"].float() / 255.0).to(device=device, dtype=dtype)
        state = tensors.get("initial_state")
        if state is not None:
            # state 保持 fp32：inference_step 内的归一化统计量（action_stats）
            # 是 fp32，bf16 输入会在 _apply_per_name 的 dtype 检查上报错。
            state = state.to(device=device, dtype=torch.float32)
        # action 预测的显式声明（batch 契约 state_action_policies；batch 逐源
        # 齐次，取首项）：placeholder（EWMBenchmark 借位，零 state 仅锚点）→
        # 强制预测；absent/none（纯视频源）→ 显式跳过；real/未声明（含旧缓存）
        # → None 走模型"全 0 state=action 缺失"启发式（旧行为）。
        policies = meta.get("state_action_policies") or []
        policy = str(policies[0]) if policies else ""
        predict_action: Optional[bool] = {"placeholder": True, "absent": False, "none": False}.get(policy)
        if (
            predict_action is None
            and torch.is_tensor(state)
            and state.numel() > 0
            and not bool(state.abs().amax().item())
            and self.assignment.source_name not in self._warned_zero_state
        ):
            self._warned_zero_state.add(self.assignment.source_name)
            logger.warning(
                "periodic_eval: source %s 的 eval batch state 全 0 且未声明 "
                "state_action_policies，action 预测被零值启发式跳过；若该源应"
                "预测 action（借位机器人），请在 batch 中声明 placeholder",
                self.assignment.source_name,
            )
        gt_video = tensors["video_frames"].float() / 255.0  # [B,T,C,H,W] CPU [0,1]
        gt_actions = tensors.get("action_sequence")  # [B,A,D] fp32 CPU or None

        # depth 第三生成流：按 sidecar / robotwin inline 给缓存 batch 补深度。
        # robotwin_3d + enable_depth 时必须全员 present，禁止再静默降级成 RGB-only。
        depth_first = None
        depth_gt = None
        depth_valid_mask = None
        attacher = self._depth_attacher()
        require_robotwin_depth = _robotwin_depth_required(
            self.config, meta, source_name=self.assignment.source_name
        )
        if require_robotwin_depth and attacher is None:
            raise RuntimeError(
                "periodic_eval: robotwin_3d depth required but attacher is None "
                "(enable_depth / depth.robotwin_3d_root missing or failed to build)"
            )
        if attacher is not None:
            try:
                attached = attacher.attach(
                    {
                        "video_frames": tensors["video_frames"],
                        "episode_ids": meta.get("episode_ids"),
                        "frame_indices": meta.get("frame_indices"),
                        # DepthAttacher 靠 source_datasets==robotwin_3d 才走
                        # inline reader；漏传时会落到空的 sidecar roots，整批
                        # depth_present=False，评测就完全没有深度。
                        "source_datasets": meta.get("source_datasets"),
                    }
                )
                present = attached["depth_present"]
                n = int(present.numel())
                n_ok = int(present.sum().item())
                if require_robotwin_depth and n_ok < n:
                    missing = [
                        str((meta.get("episode_ids") or [None])[i])
                        for i in range(n)
                        if not bool(present[i])
                    ]
                    raise RuntimeError(
                        f"periodic_eval: robotwin_3d depth missing for {n - n_ok}/{n} "
                        f"samples: {missing}"
                    )
                if n_ok > 0:
                    depth_first = attached["depth_first"].to(device=device, dtype=dtype)
                    depth_gt = attached["depth_frames"].float()  # CPU [B,T,3,H,W]
                    depth_valid_mask = attached["depth_valid_mask"][:, 1:].bool()
                else:
                    logger.warning(
                        "periodic_eval: depth attach produced 0/%d present samples "
                        "(source_datasets=%s episode0=%s)",
                        n,
                        meta.get("source_datasets"),
                        (meta.get("episode_ids") or [None])[0],
                    )
            except Exception as exc:
                if require_robotwin_depth:
                    raise
                logger.warning("periodic_eval: depth attach failed (ignored): %s", exc)

        unwrapped = getattr(model, "module", model)
        # 逐块 AR 模型：生成训练窗口内的全部块，指标按块拆分。
        block_kwargs: Dict[str, Any] = {}
        block_frames = block_steps = 0
        eval_blocks = 0
        mcfg = unwrapped.config
        if (str(getattr(mcfg, "temporal_mode", "")) == "block_causal"
                and bool(getattr(mcfg, "ar_packed_teacher_forcing", False))):
            block_frames = 4 * int(getattr(mcfg, "ar_latents_per_block", 1))
            block_steps = block_frames * int(mcfg.video_action_freq_ratio)
            eval_blocks = int(mcfg.num_video_frames) // block_frames
            block_kwargs["num_blocks"] = eval_blocks
        # 历史输入：与训练/部署同一 HistoryAttacher；step 固定为 0，槽位抖动每轮一致。
        history_inputs = None
        history_stats: Dict[str, float] = {}
        history_attacher = self._history_attacher()
        if history_attacher is not None:
            from thinker_wm.data_prepare.history_attach import history_inputs_from_batch

            attached_history = history_attacher.attach(
                {
                    "video_frames": tensors["video_frames"],
                    "episode_ids": meta.get("episode_ids"),
                    "frame_indices": meta.get("frame_indices"),
                    "source_datasets": meta.get("source_datasets"),
                },
                step=0,
            )
            history_inputs = history_inputs_from_batch(attached_history, device)
            history_stats = dict(history_attacher.last_metrics)
        rgb_only_depth = str(getattr(unwrapped.config, "depth_condition_mode", "first_frame")) == "rgb_only"
        return_predicted_depth = bool(getattr(unwrapped.config, "enable_depth", False)) and (
            rgb_only_depth or depth_first is not None)
        was_training = bool(model.training)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        # 逐 (源, shard, eval_seed) 定种子：跨轮次生成一致，且与训练 RNG 隔离。
        seed_material = (
            f"{self.assignment.source_name}|{self.assignment.shard_ordinal}|{self.cfg.seed}"
        )
        eval_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)

        predicted_frames = None
        predicted_actions = None
        predicted_depth = None
        predicted_actions_gtdepth = None
        nohist_frames = None
        nohist_actions = None
        a2v_frames = None
        a2v_hold_frames = None
        vlm_outputs: Optional[List[Optional[str]]] = None
        chunk = max(1, int(self.infer_chunk_size))

        def generate(hist, commands=None):
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)
            frame_chunks: List[Any] = []
            action_chunks: List[Any] = []
            depth_chunks: List[Any] = []
            for start in range(0, num_samples, chunk):
                end = min(start + chunk, num_samples)
                out = unwrapped.inference_step(
                    first_frame=first_frame[start:end],
                    state=_slice_eval_item(state, start, end),
                    num_inference_steps=int(self.num_inference_steps),
                    first_frame_pils=_slice_eval_item(
                        None if text_only_prompt else pils, start, end
                    ),
                    vlm_input_messages=_slice_eval_item(
                        vlm_messages, start, end
                    ),
                    robot_names=_slice_eval_item(robot_names, start, end),
                    guidance_scale=float(self.guidance_scale),
                    predict_action=predict_action,
                    depth_first=None if rgb_only_depth else _slice_eval_item(depth_first, start, end),
                    return_depth=return_predicted_depth,
                    history_inputs=_slice_eval_item(hist, start, end),
                    condition_actions=_slice_eval_item(commands, start, end),
                    **block_kwargs,
                    **self.action_inference_params,
                )
                if return_predicted_depth:
                    pred_f, pred_a, pred_d = out
                    depth_chunks.append(pred_d)
                else:
                    pred_f, pred_a = out
                frame_chunks.append(pred_f)
                action_chunks.append(pred_a)
            return (_cat_eval_output(frame_chunks), _cat_eval_output(action_chunks),
                    _cat_eval_output(depth_chunks) if return_predicted_depth else None)

        try:
            model.eval()
            with torch.no_grad():
                predicted_frames, predicted_actions, predicted_depth = generate(history_inputs)
                if history_inputs is not None and self.cfg.no_history_arm:
                    # 无历史对照臂：同种子同输入只去掉历史，指标以 *_nohist 后缀并入。
                    nohist_frames, nohist_actions, _ = generate(None)
                if self.cfg.a2v_arm and block_kwargs and gt_actions is not None and not video_only:
                    # A2V：给定真实执行动作，只预测 RGB 后果，对照真实视频（*_a2v）。
                    # 敏感性：同种子改为整段保持首步动作（绝对动作 = 原地不动），
                    # 量预测随动作的变化（*_a2v_sensitivity）；无真实后果标签，不是准确率。
                    try:
                        commands = gt_actions[:, :eval_blocks * block_steps].to(device=device, dtype=torch.float32)
                        a2v_frames = generate(history_inputs, commands)[0]
                        a2v_hold_frames = generate(history_inputs,
                                                   commands[:, :1].expand_as(commands).contiguous())[0]
                    except Exception as exc:
                        logger.warning("periodic_eval: a2v arm failed (ignored): %s", exc)
                        a2v_frames = a2v_hold_frames = None
                if (
                    self.cfg.gt_depth_action_arm
                    and not rgb_only_depth
                    and depth_first is not None
                    and depth_gt is not None
                ):
                    # GT-depth 对照臂：整段深度钉死为 GT（t_depth=0），action 读到
                    # 的是训练时见过的干净几何。与主臂同种子同噪声，配对可比。
                    try:
                        torch.manual_seed(eval_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(eval_seed)
                        arm_chunks: List[Any] = []
                        depth_gt_dev = depth_gt.to(device=device, dtype=dtype)
                        for start in range(0, num_samples, chunk):
                            end = min(start + chunk, num_samples)
                            arm = unwrapped.inference_step(
                                first_frame=first_frame[start:end],
                                state=_slice_eval_item(state, start, end),
                                num_inference_steps=int(self.num_inference_steps),
                                first_frame_pils=_slice_eval_item(
                                    None if text_only_prompt else pils, start, end
                                ),
                                vlm_input_messages=_slice_eval_item(
                                    vlm_messages, start, end
                                ),
                                robot_names=_slice_eval_item(
                                    robot_names, start, end
                                ),
                                guidance_scale=float(self.guidance_scale),
                                predict_action=predict_action,
                                depth_first=_slice_eval_item(
                                    depth_first, start, end
                                ),
                                return_depth=True,
                                condition_depth_frames=_slice_eval_item(
                                    depth_gt_dev, start, end
                                ),
                                history_inputs=_slice_eval_item(history_inputs, start, end),
                                **self.action_inference_params,
                            )
                            arm_chunks.append(arm[1])
                        predicted_actions_gtdepth = _cat_eval_output(arm_chunks)
                    except Exception as exc:
                        logger.warning(
                            "periodic_eval: gt-depth action arm failed (ignored): %s",
                            exc,
                        )
                if self.cfg.vlm_generate:
                    vlm_outputs = self._generate_vlm_texts(
                        unwrapped, vlm_messages, None if text_only_prompt else pils, device
                    )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if was_training:
                model.train()

        # predicted_frames: [B,C,T,H,W] [0,1] fp32（CPU 转换在下方逐样本做）；
        # 视频模态缺失（评审 #4）时 predicted_frames 为 None，仅动作侧继续。
        if predicted_frames is not None:
            predicted_frames = predicted_frames.detach().float().cpu()
        if predicted_actions is not None:
            predicted_actions = predicted_actions.detach().float().cpu()
        if predicted_depth is not None:
            predicted_depth = predicted_depth.detach().float().cpu()
        if predicted_actions_gtdepth is not None:
            predicted_actions_gtdepth = predicted_actions_gtdepth.detach().float().cpu()
        if nohist_frames is not None:
            nohist_frames = nohist_frames.detach().float().cpu()
        if nohist_actions is not None:
            nohist_actions = nohist_actions.detach().float().cpu()
        if a2v_frames is not None:
            a2v_frames = a2v_frames.detach().float().cpu()
        if a2v_hold_frames is not None:
            a2v_hold_frames = a2v_hold_frames.detach().float().cpu()

        step_dir = (
            self.media_root
            / "eval"
            / f"step_{int(global_step):07d}"
            / f"rank{self.rank}"
        )
        records: List[Dict[str, Any]] = []
        ts = datetime.now().isoformat(timespec="milliseconds")
        for sample_idx in range(num_samples):
            sample_rel = Path("eval") / f"step_{int(global_step):07d}" / f"rank{self.rank}" / f"sample_{sample_idx}"
            sample_dir = self.media_root / sample_rel
            pred_video = (
                predicted_frames[sample_idx].permute(1, 0, 2, 3)
                if predicted_frames is not None else None
            )  # [T,C,H,W] 或 None（video 模态缺失，评审 #4）
            gt_video_sample = gt_video[sample_idx]
            first = tensors["first_frame"][sample_idx].float() / 255.0

            metrics = dict(history_stats)
            if pred_video is not None:
                metrics.update(_with_blocks(render.video_metrics, pred_video, gt_video_sample,
                                            eval_blocks, block_frames))
            if nohist_frames is not None:
                metrics.update(_with_blocks(render.video_metrics, nohist_frames[sample_idx].permute(1, 0, 2, 3),
                                            gt_video_sample, eval_blocks, block_frames, "_nohist"))
            if a2v_frames is not None:
                a2v_video = a2v_frames[sample_idx].permute(1, 0, 2, 3)
                metrics.update(_with_blocks(render.video_metrics, a2v_video, gt_video_sample,
                                            eval_blocks, block_frames, "_a2v"))
                if a2v_hold_frames is not None:
                    metrics.update(_with_blocks(render.video_metrics, a2v_hold_frames[sample_idx].permute(1, 0, 2, 3),
                                                a2v_video, eval_blocks, block_frames, "_a2v_sensitivity"))
            pred_act = predicted_actions[sample_idx] if predicted_actions is not None else None
            gt_act = gt_actions[sample_idx] if gt_actions is not None else None
            robot_name = robot_names[sample_idx] if robot_names else None
            dim_labels = (
                self._action_dim_labels.get(str(robot_name)) if robot_name else None
            )
            # EEF 规范显示空间（每臂 3pos+6rot+1gripper）：命中布局时曲线/
            # 指标/eef_display 统一在显示空间（rot6 免疫 quat 符号翻转）；
            # 未命中布局的机器人回退 raw local 逐维（旧行为）。
            eef_layout = (
                self._eef_display_layouts.get(str(robot_name)) if robot_name else None
            )
            if eef_layout is not None:
                disp_labels = list(eef_layout.get("labels") or [])
                pred_disp = render.actions_to_eef_display(pred_act, eef_layout)
                gt_disp = (
                    None
                    if video_only
                    else render.actions_to_eef_display(gt_act, eef_layout)
                )
            else:
                disp_labels = dim_labels
                pred_disp = pred_act
                gt_disp = None if video_only else gt_act
            if gt_disp is not None and pred_disp is not None:
                act_score = functools.partial(render.action_metrics, layout=eef_layout)
                metrics.update(_with_blocks(act_score, pred_disp, gt_disp, eval_blocks, block_steps))
                if nohist_actions is not None:
                    nh = nohist_actions[sample_idx]
                    nh = render.actions_to_eef_display(nh, eef_layout) if eef_layout is not None else nh
                    metrics.update(_with_blocks(act_score, nh, gt_disp, eval_blocks, block_steps, "_nohist"))
                if predicted_actions_gtdepth is not None:
                    arm_act = predicted_actions_gtdepth[sample_idx]
                    arm_disp = (
                        render.actions_to_eef_display(arm_act, eef_layout)
                        if eef_layout is not None
                        else arm_act
                    )
                    if arm_disp is not None:
                        metrics.update(
                            {
                                f"{k}_gtdepth": v
                                for k, v in render.action_metrics(
                                    arm_disp, gt_disp, layout=eef_layout
                                ).items()
                            }
                        )

            first_uint8 = render.image_to_uint8(first)
            gt_uint8 = render.tensor_to_uint8_frames(gt_video_sample)
            render.save_png(sample_dir / "first_frame.png", first_uint8)
            gt_video_rel = None
            pred_video_rel = None
            compare_video_rel = None
            if pred_video is not None:
                pred_uint8 = render.tensor_to_uint8_frames(pred_video)
                compare_frames = render.make_compare_frames(
                    first_uint8, gt_uint8, pred_uint8
                )
                # depth 三联屏拼在 RGB 之下（同一个 compare.mp4，Web UI 不用改）：
                # 深度是灰度语义，压成亮度再渲染——生成输出的三通道会有几个百分点
                # 的分歧，直接当 RGB 播会渲染成刺眼彩噪，把误差看得远比实际严重。
                depth_metrics = self._depth_panels_and_metrics(
                    render, predicted_depth, depth_gt, sample_idx, extra_rgb=gt_video_sample,
                    depth_valid_mask=depth_valid_mask, block_frames=block_frames,
                )
                if depth_metrics is not None:
                    depth_frames_panel, depth_scores = depth_metrics
                    metrics.update(depth_scores)
                    n = min(len(compare_frames), len(depth_frames_panel))
                    compare_frames = np.concatenate(
                        [compare_frames[:n], depth_frames_panel[:n]], axis=1
                    )
                render.write_video(
                    sample_dir / "compare.mp4",
                    compare_frames,
                    fps=self.cfg.fps,
                )
                compare_video_rel = str(sample_rel / "compare.mp4")
                if self.cfg.save_separate_videos:
                    # gt/pred 走无损 mkv（FFV1）：这两路是给离线像素级指标用的，
                    # h264 的有损 + yuv420p 丢色度会污染 PSNR / 深度结构指标。
                    # compare.mp4 仍是有损 h264，只作展示。
                    render.write_video(sample_dir / "gt.mkv", gt_uint8, fps=self.cfg.fps)
                    render.write_video(sample_dir / "pred.mkv", pred_uint8, fps=self.cfg.fps)
                    gt_video_rel = str(sample_rel / "gt.mkv")
                    pred_video_rel = str(sample_rel / "pred.mkv")
            # actions.png 不再落盘（20260820 起）：逐维曲线改由 monitor 前端
            # canvas 渲染（固定 y 范围按 dim band + 网页端自定义），数据即下方
            # meta.json 的 eef_display（命中布局）/gt_actions/predicted_actions。

            vlm_output = vlm_outputs[sample_idx] if vlm_outputs else None
            meta_payload = {
                "step": int(global_step),
                "rank": self.rank,
                "sample_index": sample_idx,
                "source": self.assignment.source_name,
                "kind": self.assignment.kind,
                "split": self.assignment.split,
                "shard_ordinal": self.assignment.shard_ordinal,
                "video_only": video_only,
                "robot_name": robot_name,
                "action_dim_labels": dim_labels,
                # 显示空间标记 + EEF 规范显示数组（命中布局时与前端曲线/
                # 指标同空间；raw local 数组保留在 gt_actions/predicted_actions）
                "action_display_space": (
                    "eef_v1" if eef_layout is not None else "raw_local"
                ),
                "eef_display": (
                    {
                        "labels": disp_labels,
                        "gt": _display_list(gt_disp),
                        "pred": _display_list(pred_disp),
                    }
                    if eef_layout is not None
                    else None
                ),
                "episode_id": _meta_value(meta, "episode_ids", sample_idx),
                "manifest_id": _meta_value(meta, "manifest_ids", sample_idx),
                "task_text": _meta_value(meta, "t5_input_messages", sample_idx),
                "vlm_prompt": (vlm_messages[sample_idx] if vlm_messages else None),
                "vlm_output": vlm_output,
                "initial_state": _tensor_list(tensors.get("initial_state"), sample_idx),
                "gt_actions": _tensor_list(gt_actions, sample_idx),
                "predicted_actions": _tensor_list(predicted_actions, sample_idx),
                "num_inference_steps": int(self.num_inference_steps),
                "guidance_scale": float(self.guidance_scale),
                "metrics": metrics,
            }
            render.write_meta_json(sample_dir / "meta.json", meta_payload)
            records.append(
                {
                    "type": "eval_sample",
                    "step": int(global_step),
                    "ts": ts,
                    "rank": self.rank,
                    "sample_index": sample_idx,
                    "source": self.assignment.source_name,
                    "dataset": self.assignment.source_name,
                    # split=train 的样本来自训练池，只用于"学没学会"，不能当泛化指标
                    "split": self.assignment.split,
                    "robot_name": meta_payload["robot_name"],
                    "episode_id": meta_payload["episode_id"],
                    "manifest_id": meta_payload["manifest_id"],
                    "video_only": video_only,
                    "task_text": meta_payload["task_text"],
                    "vlm_prompt": meta_payload["vlm_prompt"],
                    "vlm_output": vlm_output,
                    "initial_state": meta_payload["initial_state"],
                    "metrics": metrics,
                    "files": {
                        "first_frame": str(sample_rel / "first_frame.png"),
                        "compare_video": compare_video_rel,
                        "gt_video": gt_video_rel,
                        "pred_video": pred_video_rel,
                        # action 曲线不再落盘 PNG：前端 canvas 渲染，数据在
                        # meta.json（eef_display / gt_actions / predicted_actions）
                        "actions_plot": None,
                        "meta": str(sample_rel / "meta.json"),
                    },
                }
            )

        summary = {
            "source": self.assignment.source_name,
            "kind": self.assignment.kind,
            "split": self.assignment.split,
            "ready": True,
            "num_samples": num_samples,
            "seconds": time.perf_counter() - round_start,
            "metrics": _mean_metrics([r["metrics"] for r in records]),
        }
        logger.info(
            "periodic_eval: step %d rank %d evaluated %d samples from %s in %.1fs (%s)",
            global_step,
            self.rank,
            num_samples,
            self.assignment.source_name,
            summary["seconds"],
            summary["metrics"],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return records, summary

    def _generate_vlm_texts(
        self,
        unwrapped: Any,
        vlm_messages: List[str],
        pils: Optional[List[Any]],
        device: torch.device,
    ) -> List[Optional[str]]:
        """冻结 VLM 对同一输入跑一次 greedy generate，decode 新 token 为文字。

        模型内 VLM 纯编码（und 分支只取 last_hidden_state），这是评测专用的
        旁路生成（模式同 scripts/inspect_vlm_per_dataset.py）。任何失败
        （如 micro 替身无 generate）降级为 None 列表。pils=None 时走纯文本
        生成（prompt_mode=text_only 的源）。
        """
        try:
            from thinker_wm.utils.vlm_utils import (
                preprocess_vlm_messages,
                preprocess_vlm_text_only,
            )

            und_branch = unwrapped.world_model.und_branch
            vlm_model = und_branch.vlm_model
            processor = und_branch.vlm_preprocess
        except Exception as exc:
            logger.warning("periodic_eval: VLM handle unavailable (%s); skip generate", exc)
            return [None] * len(vlm_messages)

        outputs: List[Optional[str]] = []
        for index, message in enumerate(vlm_messages):
            try:
                if pils is None:
                    inputs = preprocess_vlm_text_only(message, processor)
                else:
                    inputs = preprocess_vlm_messages(message, pils[index], processor)
                gen_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                gen_ids = vlm_model.generate(
                    **gen_inputs,
                    max_new_tokens=int(self.cfg.vlm_max_new_tokens),
                    do_sample=False,
                )
                text = processor.tokenizer.decode(
                    gen_ids[0, gen_inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                outputs.append(str(text))
            except Exception as exc:
                logger.warning("periodic_eval: VLM generate failed (ignored): %s", exc)
                outputs.append(None)
        return outputs

    # ------------------------------------------------------------------ 索引写出

    def _write_round(
        self,
        global_step: int,
        gathered: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """rank0：聚合一轮的记录，追加 eval 索引流 + 事件 + 清理旧轮媒体。"""
        all_records: List[Dict[str, Any]] = []
        summaries: List[Dict[str, Any]] = []
        for records, summary in gathered:
            all_records.extend(records or [])
            summaries.append(summary or {})

        ready = [s for s in summaries if s.get("ready")]
        skipped = [s for s in summaries if not s.get("ready")]
        round_metrics = _mean_metrics([r["metrics"] for r in all_records])
        source_counts: Dict[str, int] = {}
        for record in all_records:
            source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
        grouped_metrics = _group_metrics_by_family_split(all_records)
        round_seconds = max((float(s.get("seconds", 0.0)) for s in ready), default=0.0)

        writer = self._get_eval_writer()
        if writer is not None:
            ts = datetime.now().isoformat(timespec="milliseconds")
            writer.write(
                {
                    "type": "eval_round",
                    "step": int(global_step),
                    "ts": ts,
                    "seconds": round_seconds,
                    "num_samples": len(all_records),
                    "ready_ranks": len(ready),
                    "world_size": self.world_size,
                    "sources": source_counts,
                    "metrics": round_metrics,
                    "metrics_by_group": grouped_metrics,
                    "skipped": [s.get("source") for s in skipped],
                }
            )
            for record in all_records:
                writer.write(record)

        self._prune_old_rounds()
        _monitor_event(
            self.monitor,
            "periodic_eval_done",
            step=int(global_step),
            seconds=round_seconds,
            num_samples=len(all_records),
            ready_ranks=len(ready),
            world_size=self.world_size,
            sources=source_counts,
            metrics=round_metrics,
            metrics_by_group=grouped_metrics,
        )
        logger.info(
            "periodic_eval: step %d round done: %d samples (%s), ready %d/%d ranks, metrics=%s",
            global_step,
            len(all_records),
            source_counts,
            len(ready),
            self.world_size,
            round_metrics,
        )
        if grouped_metrics:
            logger.info(
                "periodic_eval: step %d depth by family/split: %s",
                global_step,
                {
                    group: {k: round(v, 4) for k, v in vals.items() if k.startswith("depth_")}
                    for group, vals in grouped_metrics.items()
                },
            )
            logger.info(
                "periodic_eval: step %d action by family/split: %s",
                global_step,
                {
                    group: {k: round(v, 4) for k, v in vals.items() if k.startswith("action_")}
                    for group, vals in grouped_metrics.items()
                },
            )
        scalars: Dict[str, Any] = {
            "periodic_eval_samples": float(len(all_records)),
            "periodic_eval_ready_ranks": float(len(ready)),
        }
        for key, value in round_metrics.items():
            scalars[f"periodic_eval_{key}"] = float(value)
        for summary in ready:
            prefix = f"periodic_eval_{summary['source']}"
            for key, value in (summary.get("metrics") or {}).items():
                scalars[f"{prefix}_{key}"] = float(value)
        for group, vals in grouped_metrics.items():
            for key, value in vals.items():
                scalars[f"periodic_eval_{group}_{key}"] = float(value)
        return scalars

    def _get_eval_writer(self):
        """rank0 懒创建 eval 索引流 writer（monitor run_dir/eval/*.jsonl）。"""
        if self._eval_writer is not None:
            return self._eval_writer
        run_dir = getattr(self.monitor, "run_dir", None) if self.monitor is not None else None
        if not run_dir:
            return None
        try:
            from motus_monitor.store import ChunkedJsonlWriter

            self._eval_writer = ChunkedJsonlWriter(Path(run_dir) / "eval")
        except Exception as exc:
            logger.warning("periodic_eval: eval stream writer init failed: %s", exc)
            self._eval_writer = None
        return self._eval_writer

    def _prune_old_rounds(self) -> None:
        """keep_rounds：只保留最近 N 轮的媒体目录（rank0 调用，gather 之后）。"""
        if not self.cfg.keep_rounds:
            return
        try:
            import re
            import shutil

            eval_root = self.media_root / "eval"
            if not eval_root.is_dir():
                return
            steps = []
            for path in eval_root.iterdir():
                match = re.fullmatch(r"step_(\d+)", path.name)
                if match and path.is_dir():
                    steps.append((int(match.group(1)), path))
            steps.sort()
            excess = len(steps) - int(self.cfg.keep_rounds)
            for _, path in steps[: max(0, excess)]:
                shutil.rmtree(path, ignore_errors=True)
                logger.info("periodic_eval: pruned old round media %s", path)
        except Exception as exc:
            logger.warning("periodic_eval: prune failed (ignored): %s", exc)

    def close(self) -> None:
        if self._eval_writer is not None:
            try:
                self._eval_writer.close()
            except Exception:
                pass
            self._eval_writer = None


def _meta_value(meta: Dict[str, Any], key: str, index: int) -> Any:
    value = meta.get(key)
    if isinstance(value, list) and index < len(value):
        return value[index]
    return None


def _tensor_list(tensor: Optional[torch.Tensor], index: int) -> Any:
    if tensor is None:
        return None
    return tensor[index].detach().float().cpu().tolist()


def _display_list(array: Any) -> Any:
    """EEF 显示数组（numpy [T, N]）→ JSON 安全嵌套列表（NaN/Inf → None）。

    n/a 槽位（无夹爪本体等）在显示数组中为 NaN；meta.json 走标准 JSON
    （前端 JSON.parse 不接受 NaN 字面量），故映射为 null。
    """
    if array is None:
        return None
    return [
        [None if not math.isfinite(float(value)) else float(value) for value in row]
        for row in array
    ]


def _depth_motion_mask_raw(
    depth_m: torch.Tensor,
    valid: torch.Tensor,
    thresh_m: float = 0.10,
) -> torch.Tensor:
    """原始帧差：|Δz|>thresh。相机转动 / DA3 背景闪烁都会进 mask。"""
    if depth_m.shape[0] < 2:
        return torch.zeros_like(valid)
    pair_ok = valid[1:] & valid[:-1]
    moving = pair_ok & ((depth_m[1:] - depth_m[:-1]).abs() > thresh_m)
    out = torch.zeros_like(valid)
    out[1:] |= moving
    out[:-1] |= moving
    return out


def _depth_motion_mask(
    depth_m: torch.Tensor,
    valid: torch.Tensor,
    thresh_m: float = 0.10,
    rgb: torch.Tensor | None = None,
    rgb_thresh: float = 0.05,
    residual_m: float = 0.05,
    blur_ks: int = 41,
) -> torch.Tensor:
    """物体运动 mask，压掉相机转动和深度闪烁。

    在原始 |Δz|>thresh 之上加两刀：
    1. 减去大核模糊后的 |Δz| 场——相机转动造成的是空间平滑的深度变化，
       臂/被抓物是局部凸起。
    2. 若有 RGB：还要求亮度帧差过线。DA3 背景闪烁是深度跳、RGB 不动。
    """
    if depth_m.shape[0] < 2:
        return torch.zeros_like(valid)
    pair_ok = valid[1:] & valid[:-1]
    dz = (depth_m[1:] - depth_m[:-1]).abs()
    moving = pair_ok & (dz > thresh_m)
    if residual_m > 0 and blur_ks >= 3:
        ks = int(blur_ks) | 1
        pad = ks // 2
        x = dz.unsqueeze(1)
        lf = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate"),
            kernel_size=ks,
            stride=1,
        ).squeeze(1)
        moving = moving & ((dz - lf) > residual_m)
    if rgb is not None:
        if rgb.dim() == 4 and rgb.shape[1] in (1, 3):
            luma = rgb.float().mean(dim=1)
        elif rgb.dim() == 4 and rgb.shape[-3] in (1, 3):
            luma = rgb.float().mean(dim=-3)
        else:
            luma = rgb.float()
        t = min(luma.shape[0], depth_m.shape[0])
        di = (luma[1:t] - luma[: t - 1]).abs()
        if di.shape[-2:] != dz.shape[-2:]:
            di = torch.nn.functional.interpolate(
                di.unsqueeze(1), size=dz.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        moving = moving[: di.shape[0]] & (di > rgb_thresh)
    out = torch.zeros_like(valid)
    n = moving.shape[0]
    out[1 : n + 1] |= moving
    out[:n] |= moving
    return out


_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
) / 8.0
_LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _conv3(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """[T,H,W] 上的 3x3 卷积（replicate padding，保持尺寸）。"""
    weight = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    padded = torch.nn.functional.pad(
        x.unsqueeze(1), (1, 1, 1, 1), mode="replicate"
    )
    return torch.nn.functional.conv2d(padded, weight).squeeze(1)


def _masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """mask 内的 Pearson 相关（去均值后的余弦）。"""
    x = a[mask].float()
    y = b[mask].float()
    x = x - x.mean()
    y = y - y.mean()
    den = float(x.norm() * y.norm())
    if den < 1e-8:
        return float("nan")
    return float(torch.dot(x, y) / den)


def _depth_structure_metrics(
    pred_m: torch.Tensor,
    gt_m: torch.Tensor,
    valid: torch.Tensor,
    bg: torch.Tensor,
    z_near: float,
    z_far: float,
    edge_pct: float = 75.0,
) -> Dict[str, float]:
    """log-depth 梯度结构指标（补 AbsRel 看不见的「背景糊成一片」）。

    AbsRel 是逐像素幅值误差：背景被抹成平滑的一坨、只要平均深度对，
    AbsRel 几乎不动。这里三个指标直接测结构：

    - ``depth_grad_corr`` / ``_bg``：GT 与 pred 的 log-depth Sobel 梯度场
      相关（↑）。背景版本是「背景清晰度」的主判据。
    - ``depth_edge_absrel``：只在 GT 梯度幅值前 25% 的边缘像素上算
      AbsRel（↓），边界糊掉会被直接惩罚。
    - ``depth_lap_bg``：pred 背景 Laplacian 方差（↑），细节量的绝对水平；
      与 grad_corr_bg 一起看可区分「没细节」和「细节但对不上」。

    口径与离线版 ``tmp/edge_metrics_from_compare.py`` 一致（逐 clip 的
    75 分位边缘阈值），便于和已有 V5/V6.1/V7 台账对齐。
    """
    out: Dict[str, float] = {}
    if not bool(valid.any()):
        return out
    log_gt = gt_m.clamp(z_near, z_far).log()
    log_pred = pred_m.clamp(z_near, z_far).log()
    sobel_y = _SOBEL_X.t()
    ggx, ggy = _conv3(log_gt, _SOBEL_X), _conv3(log_gt, sobel_y)
    pgx, pgy = _conv3(log_pred, _SOBEL_X), _conv3(log_pred, sobel_y)

    grad_pred = torch.cat([pgx, pgy], dim=0)
    grad_gt = torch.cat([ggx, ggy], dim=0)
    valid2 = torch.cat([valid, valid], dim=0)
    corr = _masked_corr(grad_gt, grad_pred, valid2)
    if corr == corr:
        out["depth_grad_corr"] = corr
    if bool(bg.any()):
        bg2 = torch.cat([bg, bg], dim=0)
        corr_bg = _masked_corr(grad_gt, grad_pred, bg2)
        if corr_bg == corr_bg:
            out["depth_grad_corr_bg"] = corr_bg
        lap = _conv3(log_pred, _LAPLACIAN)
        out["depth_lap_bg"] = float(lap[bg].float().var())

    mag = torch.hypot(ggx, ggy)
    flat = mag[valid].float().flatten()
    # torch.quantile 有 ~16M 元素上限；大 clip 上等间隔抽样后取分位。
    stride = max(1, flat.numel() // 4_000_000)
    thr = torch.quantile(flat[::stride], float(edge_pct) / 100.0)
    edge = valid & (mag >= thr)
    if int(edge.sum()) > 50:
        rel = (pred_m - gt_m).abs() / gt_m.clamp_min(1e-3)
        out["depth_edge_absrel"] = float(rel[edge].mean())
    return out


def _mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for metrics in metrics_list:
        for key, value in (metrics or {}).items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key] > 0}


def _source_family(source_name: str) -> str:
    """把源名折成家族名（agibot_beta_dexterous_hand -> agibot）。"""
    name = str(source_name or "unknown")
    for family in ("agibot", "egoscale", "openvid", "ewmbenchmark", "libero", "egoverse"):
        if name.startswith(family):
            return family
    return name


def _group_metrics_by_family_split(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """按 家族×split 分组求指标均值。

    深度标签质量在家族间不可比：agibot 的 DA3 深度有传感器对标，egoscale 的 FFS
    是野外自我中心视频、无绝对真值，两家混在一个 AbsRel 平均里看不出哪边在动。
    split 再分一层用于区分"没学会"（train/holdout 都差）与"没泛化"。
    """
    buckets: Dict[str, List[Dict[str, float]]] = {}
    for record in records:
        group = f"{_source_family(record.get('source'))}_{record.get('split') or 'unknown'}"
        buckets.setdefault(group, []).append(record.get("metrics") or {})
    return {group: _mean_metrics(items) for group, items in buckets.items()}


def build_periodic_evaluator(
    config: Any,
    *,
    rank: int,
    world_size: int,
    monitor: Any = None,
    checkpoint_dir: Optional[str] = None,
) -> Any:
    """构建 PeriodicEvaluator；未启用/任何失败 → NullPeriodicEvaluator（绝不让训练起不来）。"""
    try:
        pe_cfg = parse_periodic_eval_config(config)
    except Exception as exc:
        logger.warning("periodic_eval: config parse failed (%s); disabled", exc)
        return NullPeriodicEvaluator()
    if not pe_cfg.enabled:
        return NullPeriodicEvaluator()
    try:
        return PeriodicEvaluator(
            config,
            pe_cfg,
            rank=rank,
            world_size=world_size,
            monitor=monitor,
            checkpoint_dir=checkpoint_dir,
        )
    except Exception as exc:
        logger.exception("periodic_eval: init failed (%s); disabled", exc)
        _monitor_event(monitor, "periodic_eval_init_failed", error=repr(exc))
        return NullPeriodicEvaluator()
