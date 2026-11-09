"""数据准备包：motus-data-factory（motus_data）的仓库接入层与索引构建工具。

- factory:               训练/评估 YAML 配置 → motus_data indexed（Catalog/SamplePool）管线
- sampling_ranges:       每 Episode 采样范围 sidecar + 内存动态随机索引 Pool（SupportsSamplePool）
- pool_providers:        pool spec（YAML `pool:`）→ pool 工厂注册表
- build_micro_catalogs:  从 micro_data manifest 构建本地 Episode Catalog（CLI）
- build_sampling_ranges: 从 Episode Catalog 构建采样范围 sidecar（CLI）
"""

from __future__ import annotations

from ._bootstrap import ensure_motus_data_importable

# 纯模块（temporal_windows / robodojo_core 等）在部署/CPU 测试环境里不应依赖
# motus_data 的第三方依赖（omegaconf 等）：引导失败时推迟到真正使用 factory 时再抛。
try:
    ensure_motus_data_importable()
    _BOOTSTRAP_ERROR = None
except ImportError as _e:  # noqa: F841
    _BOOTSTRAP_ERROR = _e

if _BOOTSTRAP_ERROR is None:
    from .factory import (  # noqa: E402
        create_diverse_eval_batch,
        create_eval_data_pipeline,
        create_training_data_pipeline,
    )


def __getattr__(name):
    if name in ("create_training_data_pipeline", "create_eval_data_pipeline",
                "create_diverse_eval_batch") and _BOOTSTRAP_ERROR is not None:
        raise ImportError(f"thinker_wm.data_prepare.{name} needs motus_data: {_BOOTSTRAP_ERROR}") from _BOOTSTRAP_ERROR
    raise AttributeError(name)

__all__ = [
    "create_training_data_pipeline",
    "create_eval_data_pipeline",
    "create_diverse_eval_batch",
    "ensure_motus_data_importable",
]
