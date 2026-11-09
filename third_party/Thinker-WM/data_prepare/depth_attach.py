"""Attach a depth visual stream to an already-collated model batch.

Repo-side depth semantics (same philosophy as source_readers / pool_providers):
given a collated ``motus_model_batch_v1`` dict, this reads the metric-depth
sidecars for each sample's exact sampled clip and adds a depth stream that
*mirrors the RGB one*. Training invokes it in the ready-queue producer so the
CPU work overlaps the previous GPU step; queue-disabled/eval paths may invoke
it synchronously:

    depth_first   [B, 3, H, W]      float32 in [0, 1]   (mirrors first_frame)
    depth_frames  [B, T, 3, H, W]   float32 in [0, 1]   (mirrors video_frames)
    depth_present [B]               bool
    depth_valid_mask [B, T+1, H, W] bool (finite, positive, within codec range)

Alignment is exact because the depth-aligned sampling pool (``depth_clips``)
makes the sampled RGB ``frame_indices`` equal the dumped depth clip, so depth
frame ``k`` is the same timestamp as RGB frame ``k``. Samples whose sidecar is
missing (non-depth sources, dropped-QC episodes) get zero depth and
``present=False``; the depth branch is skipped for them downstream.

Depth is emitted in the RGB batch-boundary range [0, 1] (the model shifts to
[-1, 1] before the shared frozen VAE, identical to RGB), so the depth branch
reuses the exact RGB encode path.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from thinker_wm.models.depth_codec import InvalidTo, Mode, encode_depth_mm
from thinker_wm.models.depth_reader import DepthSidecarReader

from .robotwin_3d import Robotwin3dInlineDepthReader, SOURCE_DATASET as ROBOTWIN_3D

# Per-sample depth read is one ffmpeg subprocess per camera, so the batch loop is
# IO-bound and thread-parallel. <=1 restores the serial loop.
DEPTH_ATTACH_WORKERS_ENV = "MOTUS_DEPTH_ATTACH_WORKERS"
_DEFAULT_ATTACH_WORKERS = 8
_DEPTH_BATCH_KEYS = ("depth_first", "depth_frames", "depth_present")


def _resolve_attach_workers(workers: int | None) -> int:
    if workers is None:
        raw = os.environ.get(DEPTH_ATTACH_WORKERS_ENV)
        try:
            workers = int(raw) if raw is not None else _DEFAULT_ATTACH_WORKERS
        except ValueError:
            workers = _DEFAULT_ATTACH_WORKERS
    return max(1, int(workers))


class DepthAttacher:
    """Add ``depth_first`` / ``depth_frames`` / ``depth_present`` to a batch."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        z_near: float = 0.1,
        z_far: float = 10.0,
        mode: Mode = "log",
        invalid_to: InvalidTo = "far",
        prod_keep_only: bool = True,
        robotwin_3d_root: str | Path | None = None,
        attach_workers: int | None = None,
        require_valid_mask: bool = False,
        inline_readers: Mapping[str, Any] | None = None,
    ):
        # source_dataset -> object with read_mm(episode_id, indices, target_hw=)
        # (e.g. robodojo.RoboDojoInlineDepthReader); resolved by source name only.
        self.inline_readers = dict(inline_readers or {})
        self.reader = DepthSidecarReader(list(roots) if roots else [], prod_keep_only=prod_keep_only)
        self.attach_workers = _resolve_attach_workers(attach_workers)
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.mode: Mode = mode
        self.invalid_to: InvalidTo = invalid_to
        if not 0 < self.z_near < self.z_far:
            raise ValueError("depth range must satisfy 0 < z_near < z_far (metres)")
        self.require_valid_mask = bool(require_valid_mask)
        self.robotwin_3d = (
            None
            if robotwin_3d_root is None or not str(robotwin_3d_root).strip()
            else Robotwin3dInlineDepthReader(robotwin_3d_root)
        )

    def _resize_mm(self, mm: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Nearest-resize ``[T, H0, W0]`` mm depth to ``hw`` (never interpolate
        across depth discontinuities; keeps 0=invalid intact)."""
        if tuple(mm.shape[-2:]) == hw:
            return mm
        resized = F.interpolate(
            mm.unsqueeze(1).float(), size=hw, mode="nearest"
        ).squeeze(1)
        return resized

    def attach(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        attached_keys = tuple(key for key in _DEPTH_BATCH_KEYS if key in batch)
        if len(attached_keys) == len(_DEPTH_BATCH_KEYS):
            if self.require_valid_mask and "depth_valid_mask" not in batch:
                raise ValueError("rgb_only depth training requires depth_valid_mask from metric labels")
            # Training ready-queue producer may already have attached depth.
            # Returning the same object makes the trainer-side synchronous path
            # a safe fallback without reading/encoding the clip twice.
            return batch
        if attached_keys:
            raise RuntimeError(
                "partially attached depth batch: expected all of "
                f"{_DEPTH_BATCH_KEYS}, got {attached_keys}"
            )

        video = batch["video_frames"]  # [B, T, 3, H, W] in [0, 1]
        B, T = int(video.shape[0]), int(video.shape[1])
        H, W = int(video.shape[-2]), int(video.shape[-1])
        episode_ids = batch.get("episode_ids")
        frame_indices = batch.get("frame_indices")

        depth_full = torch.zeros((B, T + 1, 3, H, W), dtype=torch.float32)
        present = torch.zeros((B,), dtype=torch.bool)
        valid_full = torch.zeros((B, T + 1, H, W), dtype=torch.bool)

        sources = batch.get("source_datasets")

        def read_one(b: int) -> None:
            indices = frame_indices[b]
            if indices is None or len(indices) != T + 1:
                return
            source = None
            if sources is not None:
                source = str(sources[b])
            ep = str(episode_ids[b])
            # episode_id 前缀也可识别 robotwin_3d，避免调用方漏传
            # source_datasets 时整批静默变成无深度。
            use_inline = self.robotwin_3d is not None and (
                source == ROBOTWIN_3D or ep.startswith(f"{ROBOTWIN_3D}:")
            )
            if source is not None and source in self.inline_readers:
                mm, ok = self.inline_readers[source].read_mm(ep, indices, target_hw=(H, W))
            elif use_inline:
                mm, ok = self.robotwin_3d.read_mm(
                    ep,
                    indices,
                    target_hw=(H, W),
                )
            else:
                mm, ok = self.reader.read_mm(str(episode_ids[b]), indices)
                if ok and mm is not None:
                    mm = self._resize_mm(mm, (H, W))  # [T+1, H, W]
            if not ok or mm is None:
                return
            mm = torch.as_tensor(mm).float()
            if tuple(mm.shape) != (T + 1, H, W):
                raise ValueError(f"depth clip shape {tuple(mm.shape)} != {(T + 1, H, W)} for {ep}")
            valid = (torch.isfinite(mm) & (mm > 0)
                     & (mm >= self.z_near * 1000) & (mm <= self.z_far * 1000))
            # Codec sentinel is only a finite VAE input; it is not a supervised
            # far-plane measurement. Keep the independent metric validity mask.
            mm = torch.where(torch.isfinite(mm) & (mm > 0), mm, torch.zeros_like(mm))
            x = encode_depth_mm(
                mm,
                z_near=self.z_near,
                z_far=self.z_far,
                mode=self.mode,
                invalid_to=self.invalid_to,
                out_range="01",
            )  # [T+1, 3, H, W] in [0, 1]
            # 每个 b 写各自的切片，preallocated 张量无重分配，线程间不重叠。
            depth_full[b] = x
            valid_full[b] = valid
            present[b] = bool(valid.any())

        if episode_ids is not None and frame_indices is not None:
            workers = min(self.attach_workers, B)
            if workers > 1 and B > 1:
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="depth-attach"
                ) as pool:
                    # list() 让 worker 里的异常在此抛出，与串行循环行为一致。
                    list(pool.map(read_one, range(B)))
            else:
                for b in range(B):
                    read_one(b)

        result = dict(batch)
        result["depth_first"] = depth_full[:, 0]  # [B, 3, H, W]
        result["depth_frames"] = depth_full[:, 1:]  # [B, T, 3, H, W]
        result["depth_present"] = present
        result["depth_valid_mask"] = valid_full
        return result


def _config_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def build_depth_attacher(config: Any) -> DepthAttacher | None:
    """Build the configured depth attacher for training or synchronous fallback."""

    wm_cfg = _config_get(_config_get(config, "model"), "wm")
    if not bool(_config_get(wm_cfg, "enable_depth", False)):
        return None

    depth_cfg = _config_get(config, "depth")
    raw_roots = _config_get(depth_cfg, "roots")
    if raw_roots is None:
        roots: list[str | Path] = []
    elif isinstance(raw_roots, (str, Path)):
        roots = [raw_roots]
    else:
        roots = list(raw_roots)
    inline_root = _config_get(depth_cfg, "robotwin_3d_root")
    if inline_root is not None and not str(inline_root).strip():
        inline_root = None
    inline_readers = {}
    rd_map = _config_get(depth_cfg, "robodojo_field_map")
    if rd_map is not None and str(rd_map).strip():
        import json
        from .robodojo_core import FieldMap
        from .robodojo import RoboDojoInlineDepthReader
        rd_catalog = _config_get(depth_cfg, "robodojo_catalog")
        if rd_catalog is None or not str(rd_catalog).strip():
            raise ValueError("depth.robodojo_field_map requires depth.robodojo_catalog")
        rd = RoboDojoInlineDepthReader(FieldMap.load(str(rd_map)))
        with open(str(rd_catalog), encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                rd.register_episode(rec["episode_id"], rec["video"]["storage"]["path"])
        inline_readers["robodojo"] = rd
    if not roots and inline_root is None and not inline_readers:
        raise ValueError(
            "enable_depth=True 但未配置 depth.roots 或 depth.robotwin_3d_root。"
            "sidecar 源填 depth.roots；robotwin_3d 填 depth.robotwin_3d_root。"
        )

    return DepthAttacher(
        roots,
        z_near=float(_config_get(depth_cfg, "z_near", 0.1)),
        z_far=float(_config_get(depth_cfg, "z_far", 10.0)),
        mode=str(_config_get(depth_cfg, "mode", "log")),
        invalid_to=str(_config_get(depth_cfg, "invalid_to", "far")),
        prod_keep_only=bool(_config_get(depth_cfg, "prod_keep_only", True)),
        robotwin_3d_root=inline_root,
        attach_workers=_config_get(depth_cfg, "attach_workers"),
        require_valid_mask=str(_config_get(wm_cfg, "depth_condition_mode", "first_frame")) == "rgb_only",
        inline_readers=inline_readers,
    )
