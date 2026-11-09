"""后台准备进程入口：``python -m periodic_eval.prepare``。

由训练进程在每个 rank 的评测缓存缺失时以后台子进程拉起（CPU-only，父进程
已把 ``CUDA_VISIBLE_DEVICES`` 置空）。职责：把一个 rank 固定的评测 batch
（indexed 源 holdout 第 shard_ordinal 个 batch / EWMBenchmark 固定 shard）
解码出来，经 periodic_eval.cache 原子落盘（含 manifest + sha256）。新训练
任务命中已有缓存时本进程直接退出（零解码）。

参数：
  --config-dump  训练 YAML 的 dump（runner 原子写出，含 dataset/model 等节）
  --spec-json    本 rank 评测 batch 的 spec（JSON 文件；决定缓存目录名）
  --out-dir      缓存输出目录（<cache_dir>/<spec_key>）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from thinker_wm.periodic_eval import cache as cache_mod  # noqa: E402

logger = logging.getLogger("thinker_wm.periodic_eval.prepare")


def _indexed_input_files(config, source_name: str) -> List[str]:
    """收集 indexed 源的输入数据文件（供 manifest 记录内容 sha256）。"""
    files: List[str] = []
    for child in config.dataset.sources:
        if str(child.get("name") or child.get("dataset")) != source_name:
            continue
        for key in ("catalog_path", "index_path"):
            value = child.get(key)
            if value:
                files.append(str(value))
        pool_spec = child.get("eval_pool") or child.get("pool")
        if pool_spec:
            pool = OmegaConf.to_container(pool_spec, resolve=True)
            if isinstance(pool, dict) and pool.get("path"):
                files.append(str(pool["path"]))
        break
    return files


def run(config_dump: Path, spec_json: Path, out_dir: Path) -> int:
    with spec_json.open("r", encoding="utf-8") as fp:
        spec: Dict[str, Any] = json.load(fp)

    if cache_mod.cache_is_complete(out_dir):
        logger.info("cache already complete, nothing to do: %s", out_dir)
        return 0

    config = OmegaConf.load(config_dump)
    kind = str(spec.get("kind"))
    source_name = str(spec.get("source_name"))
    shard_ordinal = int(spec.get("shard_ordinal", 0))
    seed = int(spec.get("seed", 0))

    if kind == "indexed":
        from thinker_wm.periodic_eval.sources import fetch_indexed_eval_batch

        batch = fetch_indexed_eval_batch(
            config,
            source_name,
            shard_ordinal,
            seed=seed,
            batch_size=int(spec["batch_size"]),
            use_eval_pool=(str(spec.get("split", "holdout")) != "train"),
            start_index=(
                int(shard_ordinal) if spec.get("shard_as_start") else None
            ),
        )
        input_files = _indexed_input_files(config, source_name)
    elif kind == "ewmbenchmark":
        from thinker_wm.periodic_eval.sources import load_ewmbenchmark_batch

        ewm = spec.get("ewmbenchmark") or {}
        batch = load_ewmbenchmark_batch(
            str(ewm["root"]),
            shard_ordinal,
            batch_size=int(spec["batch_size"]),
            num_video_frames=int(spec["num_video_frames"]),
            video_height=int(spec["video_height"]),
            video_width=int(spec["video_width"]),
            state_dim=int(ewm["state_dim"]),
            robot_name=str(ewm.get("robot_name") or "agibot-a2-d"),
            native_fps=int(ewm.get("native_fps", 30)),
            gt_fps=int(ewm.get("gt_fps", 5)),
        )
        input_files = [str(Path(str(ewm["root"])) / "manifest.jsonl")]
    else:
        raise ValueError(f"unknown periodic_eval source kind: {kind!r}")

    cache_mod.write_cache(out_dir, batch, spec, input_files=input_files)
    if not cache_mod.verify_cache(out_dir):
        raise RuntimeError(f"cache verification failed right after write: {out_dir}")
    logger.info(
        "prepared eval cache: source=%s shard=%d samples=%d -> %s",
        source_name,
        shard_ordinal,
        int(batch["first_frame"].shape[0]),
        out_dir,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dump", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(Path(args.config_dump), Path(args.spec_json), Path(args.out_dir))
    except Exception:
        logger.exception("periodic_eval prepare failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
