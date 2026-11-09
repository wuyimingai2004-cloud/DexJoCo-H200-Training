#!/usr/bin/env python3
"""把 micro_data 的 RobotMind 逐记录 slim hdf5 改写为窗口对齐布局。

`fetch_h200_micro_dataset.py --hdf5-slim` 生成的 slim hdf5：video 键只含本窗口
49 帧（索引重写为 0..N），但 state/action 数值键保留完整轨迹（如 528 步）。
motus_data 的 Catalog 要求 video/state/action 在 hdf5 内严格等长，因此本地把
数值键切片到本窗口范围 [s, s+n)（s = action.frame_indices[0] - 1），并把每个
窗口记录改写成独立的 49 帧"窗口 episode"：

- state.index → 0；action.frame_indices → 1..n-1
- metadata.episode_name 追加窗口后缀（保证 catalog episode_id 唯一）
- metadata.total_frames/trajectory_length/raw_lengths → n

输出新 manifest `<name>_manifest.micro_windowed.jsonl`（原文件保留），
slim hdf5 原地重写。之后即可用 build_micro_catalogs.py 构建 catalog。

用法：
    python data_prepare/align_robotmind_micro_windows.py [--micro-data-dir micro_data]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROBOTMIND_SOURCES = (
    "robotmind_tianyi",
    "robotmind_arkmobile",
    "robotmind_ark",
    "robotmind_tienkung",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-data-dir", default="micro_data")
    return parser.parse_args()


def _align_one(record: dict) -> dict:
    """改写一条记录对应的 slim hdf5 并返回更新后的记录。"""
    out = json.loads(json.dumps(record))
    video = out["video"]
    state = out["state"]
    action = out["action"]
    hdf5_path = video.get("uri") or video.get("path")
    video_key = str(video["key"])
    n = len(video["frame_indices"])
    # 两种原始约定：state=k+1/actions=[k+1..k+49]（如 tianyi）或
    # state=k/actions=[k..k+48]（arkmobile/ark/tienkung）。统一切 [start, start+n)，
    # 使新索引 0 = 首帧状态、1..n-1 = 各目标帧动作。
    state_index = int(state.get("index") or 0)
    action_first = int(action["frame_indices"][0])
    if abs(state_index - action_first) > 1:
        raise ValueError(
            f"state.index={state_index} vs action start={action_first}: {hdf5_path}"
        )
    start = min(state_index, action_first)
    if start < 0:
        raise ValueError(f"invalid window start={start}: {hdf5_path}")

    numeric_keys = sorted(
        {key for keys in (state.get("keys") or {}, action.get("keys") or {}) for key in keys.values()}
    )

    # 幂等：已对齐（标记 attr 或所有数值键已 == n）的文件跳过重写。
    with h5py.File(hdf5_path, "r") as probe:
        lengths = {int(probe[key].shape[0]) for key in numeric_keys}
        already_aligned = (
            int(probe.attrs.get("motus_window_aligned", 0)) == 1 or lengths == {n}
        )
    if already_aligned:
        with h5py.File(hdf5_path, "r+") as mark:
            mark.attrs["motus_window_aligned"] = 1
    else:
        tmp_path = hdf5_path + f".align_tmp.{os.getpid()}"
        try:
            with h5py.File(hdf5_path, "r") as fsrc, h5py.File(tmp_path, "w") as fdst:
                src_video = fsrc[video_key]
                if int(src_video.shape[0]) != n:
                    raise ValueError(
                        f"video key {video_key} has {src_video.shape[0]} frames, "
                        f"manifest references {n}: {hdf5_path}"
                    )
                parent = video_key.rsplit("/", 1)[0]
                if parent:
                    fdst.require_group(parent)
                base = h5py.check_vlen_dtype(src_video.dtype)
                if base is not None:
                    out_ds = fdst.create_dataset(
                        video_key, (n,), dtype=h5py.vlen_dtype(np.dtype(base))
                    )
                    for j in range(n):
                        out_ds[j] = np.asarray(src_video[j], dtype=base)
                else:
                    fdst.create_dataset(video_key, data=np.asarray(src_video))
                for key in numeric_keys:
                    src_ds = fsrc[key]
                    if int(src_ds.shape[0]) < start + n:
                        raise ValueError(
                            f"key {key} has {src_ds.shape[0]} steps, "
                            f"need [{start},{start + n}): {hdf5_path}"
                        )
                    parent = key.rsplit("/", 1)[0]
                    grp = fdst.require_group(parent) if parent else fdst
                    grp.create_dataset(
                        key.rsplit("/", 1)[-1],
                        data=np.asarray(src_ds[start : start + n]),
                    )
                fdst.attrs["motus_window_aligned"] = 1
            os.replace(tmp_path, hdf5_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    video["frame_indices"] = list(range(n))
    state["index"] = 0
    action["frame_indices"] = list(range(1, n))
    for item in (state, action):
        if "uri" in item:
            item["uri"] = hdf5_path
        elif "path" in item:
            item["path"] = hdf5_path
    metadata = out.setdefault("metadata", {})
    episode_name = str(metadata.get("episode_name") or "episode")
    metadata["episode_name"] = f"{episode_name}_w{start:05d}"
    metadata["total_frames"] = n
    metadata["trajectory_length"] = n
    if "raw_lengths" in metadata:
        metadata["raw_lengths"] = [n] * len(metadata["raw_lengths"])
    return out


def main() -> int:
    args = _parse_args()
    micro_data_dir = Path(args.micro_data_dir)
    for name in ROBOTMIND_SOURCES:
        manifest = micro_data_dir / f"{name}_manifest.micro.jsonl"
        if not manifest.is_file():
            print(f"SKIP {name}: manifest not found: {manifest}", flush=True)
            continue
        output = micro_data_dir / f"{name}_manifest.micro_windowed.jsonl"
        rewritten = 0
        with manifest.open("r", encoding="utf-8") as src, output.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                dst.write(
                    json.dumps(_align_one(record), ensure_ascii=False) + "\n"
                )
                rewritten += 1
        print(f"ALIGN_OK {name} records={rewritten} output={output}", flush=True)
    print("ROBOTMIND_MICRO_ALIGN_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
