"""motus_data（libs/motus-data-factory）的 sys.path 引导。

仓库以 git submodule 方式就地使用 motus-data-factory（不 pip 安装），
与入口脚本把仓库根目录加入 sys.path 的约定一致。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def ensure_motus_data_importable() -> None:
    """把 libs/motus-data-factory 加入 sys.path（幂等）。

    另外在本机无 ffmpeg 且未显式配置 MOTUS_FFMPEG_BIN 时，回退到
    imageio_ffmpeg 自带的 ffmpeg 二进制（motus_data 的视频解码子进程需要；
    集群上其内置默认路径 /code1/7TB/v2/bin/ffmpeg-static/bin/ffmpeg 已可用）。
    """
    try:
        import motus_data  # noqa: F401
    except ImportError:
        lib_dir = (
            Path(__file__).resolve().parent.parent / "libs" / "motus-data-factory"
        )
        if not (lib_dir / "motus_data").is_dir():
            raise ImportError(f"motus_data package not found at {lib_dir}")
        if str(lib_dir) not in sys.path:
            sys.path.insert(0, str(lib_dir))
        import motus_data  # noqa: F401

    if (
        not os.environ.get("MOTUS_FFMPEG_BIN")
        and shutil.which("ffmpeg") is None
        and not os.path.exists("/code1/7TB/v2/bin/ffmpeg-static/bin/ffmpeg")
    ):
        try:
            import imageio_ffmpeg

            os.environ["MOTUS_FFMPEG_BIN"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
