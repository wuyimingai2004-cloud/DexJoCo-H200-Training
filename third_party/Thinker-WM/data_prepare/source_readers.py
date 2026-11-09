"""repo 侧自定义 Episode source reader（经 IndexedTrainingSource.readers 注入）。

motus_data 库只负责统一格式/加载/打包/传输；本模块的 reader 以
``readers={"<source_dataset>": 实例}`` 覆盖库注册表里的默认 reader
（EpisodeRepository(readers=...) 语义），库零改动。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ._bootstrap import ensure_motus_data_importable

ensure_motus_data_importable()

from motus_data.repository import EpisodeRepositoryError  # noqa: E402
from motus_data.source_readers import RobotwinEpisodeReader  # noqa: E402


class RobotwinMainViewEpisodeReader(RobotwinEpisodeReader):
    """RoboTwin 拼接帧裁剪 reader：只保留主视角（上部 320x240 后视相机）。

    生产 RoboTwin MP4 为 360x320 拼接帧：y=240 硬分界，上部 320x240 主视角
    （固定后视相机），底部 2x160x120 左右腕相机。本 reader 解原生帧后先按
    ``crop=(x, y, w, h)`` 裁剪，再走父类的 pad/interpolate resize 与格式转换。
    帧尺寸装不下裁剪框时直接报错（宁可训练失败也不错训错误画面）。
    """

    def __init__(self, *args: Any, crop=(0, 0, 320, 240), **kwargs: Any):
        super().__init__(*args, **kwargs)
        x, y, w, h = (int(value) for value in crop)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"invalid main-view crop box: {crop!r}")
        self._main_view_crop = (x, y, w, h)

    def read_video(
        self,
        record: Mapping[str, Any],
        indices: Sequence[int],
        *,
        output_format: str,
        video_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        if output_format not in {"rgb_uint8_hwc", "rgb_float32_chw"}:
            raise EpisodeRepositoryError(
                f"unsupported video output_format={output_format!r}"
            )
        # 跳过 RobotwinEpisodeReader.read_video（它会直接 resize 整帧），
        # 调其上一级 _FileVideoEpisodeReader 解原生帧，裁剪后再 resize。
        native = super(RobotwinEpisodeReader, self).read_video(
            record,
            indices,
            output_format="rgb_uint8_hwc",
            video_size=None,
        )
        x, y, w, h = self._main_view_crop
        if int(native.shape[1]) < y + h or int(native.shape[2]) < x + w:
            raise EpisodeRepositoryError(
                f"RoboTwin main-view crop {self._main_view_crop} does not fit "
                f"decoded frame (H={int(native.shape[1])}, W={int(native.shape[2])}) "
                f"for episode {record.get('episode_id')!r}"
            )
        cropped = native[:, y : y + h, x : x + w, :]

        if video_size is not None and self._resize_mode == "pad":
            resized = self._resize_frames_with_padding(cropped, video_size)
            if output_format == "rgb_uint8_hwc":
                return resized
            return resized.permute(0, 3, 1, 2).float().div_(255.0)

        # video_size 为空或 interpolate 模式：与基类一致的双线性缩放/直通
        if output_format == "rgb_uint8_hwc" and video_size is None:
            return cropped
        chw = cropped.permute(0, 3, 1, 2).float().div_(255.0)
        if video_size is not None:
            chw = F.interpolate(
                chw,
                size=tuple(int(value) for value in video_size),
                mode="bilinear",
                align_corners=False,
            )
        if output_format == "rgb_float32_chw":
            return chw
        return (
            chw.mul(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
