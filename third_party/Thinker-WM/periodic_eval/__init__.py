"""periodic_eval：训练中周期性定集评测（每 N 步、每 GPU 一个固定 batch）。

对外接口只有 ``build_periodic_evaluator`` 与返回实例的
``maybe_run(model, global_step)`` / ``close()``：

- 训练启动时（Trainer 构造期）按权重给每个 rank 分配固定评测 batch
  （indexed 源 holdout 第 k 个 batch / EWMBenchmark 固定 shard），缓存缺失
  则拉起 CPU-only 后台子进程预解码到 ``<cache_dir>/<spec_key>/``（manifest
  + sha256，跨训练任务复用）；
- 训练循环每到 ``interval_steps`` 的倍数步，``maybe_run`` 只做缓存 load +
  一次 batch 推理 + 可选 VLM 文本生成 + 媒体/索引落盘（monitor run_dir 下
  ``eval/``），跨 rank 汇聚后 rank0 写 eval 索引流，Web UI「eval 评估」tab
  展示；
- 未启用或任何失败均为 no-op / 跳过，绝不影响训练。

配置见 ``configs/stage2_20260805_v1.yaml`` 的 ``periodic_eval`` 节与
``periodic_eval/config.py`` 的字段注释。
"""

from .config import (  # noqa: F401
    EWMBENCHMARK_SOURCE_NAME,
    PeriodicEvalConfig,
    SourceAssignment,
    assign_sources,
    parse_periodic_eval_config,
)
from .runner import (  # noqa: F401
    NullPeriodicEvaluator,
    PeriodicEvaluator,
    build_periodic_evaluator,
)
