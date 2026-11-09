"""评测 batch 缓存：后台准备进程一次性解码落盘，训练前台只 load。

目录布局（``<cache_dir>/<spec_key>/``）：

- ``tensors.pt``：``first_frame``/``video_frames``（uint8，[0,255]）与
  ``initial_state``/``action_sequence``（fp32，可 None）；
- ``pils/sample_<k>.png``：首帧 PIL（VLM 输入用）；
- ``meta.json``：文本与元数据（t5/vlm messages、robot_names、episode_ids 等）；
- ``manifest.json``：完成标记（最后落盘）——spec 明细、每个产物文件与输入
  数据文件（ranges/catalog/EWM manifest）的 sha256、size。

所有文件先写 ``<name>.tmp`` 再 ``os.replace`` 原子改名；读者只认
``manifest.json`` 存在且 ``status == "complete"`` 的目录（校验失败/半截
写入一律视为未就绪）。并发任务准备同一 spec 目录时内容幂等（相同输入产生
相同字节），后者覆盖前者无害。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

TENSORS_NAME = "tensors.pt"
META_NAME = "meta.json"
MANIFEST_NAME = "manifest.json"
PILS_DIR = "pils"

#: meta.json 中按 list 保存的逐样本键（从 batch dict 提取/回填）。
META_LIST_KEYS = (
    "t5_input_messages",
    "vlm_input_messages",
    "robot_names",
    "source_datasets",
    "episode_ids",
    "manifest_ids",
    "manifest_tasks",
    "frame_indices",
    "action_video_fps_ratios",
    "state_action_policies",
)
META_SCALAR_KEYS = ("video_only",)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        while True:
            chunk = fp.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as fp:
        fp.write(data)
    os.replace(tmp, target)


def _atomic_write_json(target: Path, payload: Any) -> None:
    _atomic_write_bytes(
        target, (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8")
    )


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def write_cache(
    cache_dir: Path,
    batch: Dict[str, Any],
    spec: Dict[str, Any],
    input_files: Optional[List[str]] = None,
) -> Path:
    """把一个评测 batch 落盘到 ``cache_dir``（原子；manifest 最后写）。

    batch 取 batch dict 的子集：4 个张量键（first_frame/video_frames 转
    uint8，initial_state/action_sequence fp32 原样、None 跳过）、
    first_frame_pils（存 PNG）、META_LIST_KEYS 的逐样本元数据。
    input_files：输入数据文件（ranges/catalog/EWM manifest），记录完整
    内容 sha256 供审计与跨任务复用校验。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, torch.Tensor] = {}
    for key in ("first_frame", "video_frames"):
        value = batch.get(key)
        if value is None:
            raise ValueError(f"batch missing required tensor {key!r}")
        tensor = value.detach().cpu().float().clamp(0.0, 1.0)
        tensors[key] = (tensor * 255.0).round().to(torch.uint8)
    for key in ("initial_state", "action_sequence"):
        value = batch.get(key)
        if value is not None:
            tensors[key] = value.detach().cpu().float()

    torch.save(tensors, str(cache_dir / (TENSORS_NAME + ".tmp")))
    os.replace(cache_dir / (TENSORS_NAME + ".tmp"), cache_dir / TENSORS_NAME)

    pils = batch.get("first_frame_pils") or []
    pils_dir = cache_dir / PILS_DIR
    pils_dir.mkdir(parents=True, exist_ok=True)
    for index, pil in enumerate(pils):
        target = pils_dir / f"sample_{index:04d}.png"
        tmp = pils_dir / f"sample_{index:04d}.png.tmp"
        pil.save(str(tmp), format="PNG")
        os.replace(tmp, target)

    meta: Dict[str, Any] = {}
    for key in META_LIST_KEYS:
        if batch.get(key) is not None:
            meta[key] = _to_jsonable(batch[key])
    for key in META_SCALAR_KEYS:
        if batch.get(key) is not None:
            meta[key] = _to_jsonable(batch[key])
    meta["num_samples"] = len(pils) if pils else int(tensors["first_frame"].shape[0])
    _atomic_write_json(cache_dir / META_NAME, meta)

    files: Dict[str, Dict[str, Any]] = {}
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file() and path.suffix != ".tmp" and path.name != MANIFEST_NAME:
            rel = path.relative_to(cache_dir).as_posix()
            files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}

    inputs: Dict[str, Any] = {}
    for raw in input_files or []:
        path = Path(str(raw))
        try:
            inputs[str(path)] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        except OSError as exc:
            inputs[str(path)] = {"error": str(exc)}

    manifest = {
        "status": "complete",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "spec": spec,
        "files": files,
        "input_files": inputs,
    }
    _atomic_write_json(cache_dir / MANIFEST_NAME, manifest)
    logger.info("periodic_eval cache written: %s (%d files)", cache_dir, len(files))
    return cache_dir


def read_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(cache_dir) / MANIFEST_NAME
    try:
        with path.open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return None
    return manifest


def cache_is_complete(cache_dir: Path) -> bool:
    """快速就绪判定：manifest 存在且标记 complete（不做全量哈希校验）。"""
    return read_manifest(cache_dir) is not None


def verify_cache(cache_dir: Path) -> bool:
    """严格校验：manifest 记录的每个产物文件 sha256 全对才算就绪。"""
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return False
    for rel, info in (manifest.get("files") or {}).items():
        path = cache_dir / rel
        try:
            if path.stat().st_size != int(info.get("size", -1)):
                return False
            if sha256_file(path) != info.get("sha256"):
                return False
        except OSError:
            return False
    return True


def load_cache(cache_dir: Path) -> Dict[str, Any]:
    """读回缓存 batch：张量（uint8 原样，调用方转 float）、pils、meta、manifest。"""
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"periodic_eval cache not complete: {cache_dir}")
    try:
        tensors = torch.load(
            str(cache_dir / TENSORS_NAME), map_location="cpu", weights_only=True
        )
    except TypeError:
        tensors = torch.load(str(cache_dir / TENSORS_NAME), map_location="cpu")
    with (cache_dir / META_NAME).open("r", encoding="utf-8") as fp:
        meta = json.load(fp)

    from PIL import Image

    pils: List[Any] = []
    pils_dir = cache_dir / PILS_DIR

    def _pil_index(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return 0

    for path in sorted(pils_dir.glob("sample_*.png"), key=_pil_index):
        with Image.open(path) as img:
            pils.append(img.convert("RGB").copy())
    return {"tensors": tensors, "pils": pils, "meta": meta, "manifest": manifest}
