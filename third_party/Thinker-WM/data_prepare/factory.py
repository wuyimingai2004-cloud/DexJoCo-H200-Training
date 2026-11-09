"""训练/评估数据管线工厂：YAML 配置 → motus_data indexed（Catalog/SamplePool）管线。

取代旧的 data/dataset.py（create_dataset/collate_fn）+ data/ready_batch_queue.py。
配置 schema（configs/*.yaml 的 dataset 节）：

dataset:
  type: indexed_mix
  batches_per_epoch: null                # 可选；null = 各源 batch_sampler 长度之和
  cross_rank_source_policy: synchronized # 可选；synchronized(缺省，全 rank 同源)
                                         #   | weighted_per_step(按权重在 rank 间分源)
  sources:
  - name: robotwin_aloha                 # 源名（mix 统计/日志用，必须唯一）
    dataset: robotwin_aloha              # motus_data release 注册名（releases.yaml）
    weight: 0.025                        # 可选；缺省 = pool 长度
    pool:                                # 必需；索引选择策略 spec（repo 自有，见
      type: sampling_ranges              #   data_prepare/pool_providers.py）
      path: /path/catalog.ranges.sqlite  # type 之外的键作 provider options
      windows_per_episode: null          #   （sampling_ranges: path/windows_per_episode）
    eval_pool: {...}                     # 可选；评估管线专用 spec（缺省回退 pool）
    catalog_path: /path/catalog.jsonl    # 可选；覆盖 release 解析（本地 micro 用）
    index_path: /path/catalog.index.sqlite  # 可选；与 catalog_path 配套
    adapter_options:                     # 可选；覆盖 mix 级 adapter 常量
      action_loss_weight: 0.0
    output_adapter: motus_model_batch_v1 # 可选；batch profile 覆盖（libero 等
                                         #   state/action 原生异维源用
                                         #   motus_native_multimodal_batch_v1）
    prompt_mode: vlm_image_text          # 可选；vlm_image_text(缺省，首帧图+
                                         #   四字段模板) | text_only(纯文本编码，
                                         #   不带首帧图；openvid/egoverse 用)
    reader:                              # 可选；repo 自定义 reader（覆盖库默认）
      type: robotwin_main_view           #   robotwin 拼接帧裁剪主视角
      crop: [0, 0, 320, 240]             #   (x, y, w, h)，缺省即 RoboTwin 标准布局
      # type: robotwin_3d                #   Flex-π 三路 RGB+depth 现场拼 Motus T

pool spec 经 data_prepare.pool_providers 解析为 pool 工厂 callable，由 motus_data
的 SupportsSamplePool 注入点（open_training_data sample_pool 参数）进入管线；
selection 的动态生成（如 sampling_ranges 的内存随机窗口）在训练进程内完成，
不落盘枚举 pool 文件。评估管线 seed 固定、epoch=0、不 shuffle，窗口跨次运行
可复现。

batch 契约：motus_data.collate.collate_samples 输出的 14 个训练键与旧
data.dataset.collate_fn 完全一致（vlm_inputs/language_embedding 恒为 None，
由模型侧从 first_frame_pils/t5_input_messages/vlm_input_messages 现算）。
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as torch_mp
from omegaconf import OmegaConf

from ._bootstrap import ensure_motus_data_importable

ensure_motus_data_importable()

from motus_data.batch_adapters import build_indexed_batch_adapter  # noqa: E402
from motus_data.contracts import MOTUS_MODEL_BATCH_V1  # noqa: E402
from motus_data.indexed_mix import open_mixed_training_data  # noqa: E402
from motus_data.indexed_pipeline import open_dataset  # noqa: E402
from motus_data.releases import get_episode_dataset_release  # noqa: E402
from motus_data.sample_pool import (  # noqa: E402
    IndexedEpisodeBatch,
    SamplePoolDataset,
    resolve_sample_pool,
)

from . import sampling_ranges as _sampling_ranges  # noqa: E402,F401  # 注册 provider
from . import task_balanced_ranges as _task_balanced_ranges  # noqa: E402,F401
from . import task_cycled_ranges as _task_cycled_ranges  # noqa: E402,F401
from . import capability_ranges as _capability_ranges  # noqa: E402,F401
from . import depth_clips as _depth_clips  # noqa: E402,F401  # 注册 depth_clips provider
from .depth_attach import build_depth_attacher  # noqa: E402
from .history_attach import build_history_attacher  # noqa: E402
from .pool_providers import pool_factory_from_spec  # noqa: E402
from .robotwin_3d import Robotwin3dEpisodeReader  # noqa: E402
from .source_readers import RobotwinMainViewEpisodeReader  # noqa: E402


def _append_timing_event(timing_events, name: str, start_time: float, **extra):
    if timing_events is None:
        return
    event = {
        "name": name,
        "seconds": time.perf_counter() - start_time,
    }
    event.update(extra)
    timing_events.append(event)


def _readers_from_spec(child, *, source_name: str) -> Optional[Dict[str, Any]]:
    """解析 per-source ``reader:`` spec 为 motus_data 的 readers 注入映射。

    spec 形式：{type: robotwin_main_view, crop: [x, y, w, h](可缺省)}；
    键为 catalog 的 source_dataset 名（robotwin 四个源共用 "robotwin"）。
    """
    spec = child.get("reader")
    if spec is None:
        return None
    spec = OmegaConf.to_container(spec, resolve=True)
    kind = str(spec.get("type") or "").strip()
    options = {str(k): v for k, v in spec.items() if k != "type"}
    if kind == "robotwin_main_view":
        kwargs: Dict[str, Any] = {}
        if options.get("crop") is not None:
            kwargs["crop"] = tuple(int(v) for v in options.pop("crop"))
        kwargs.update(options)  # 其余键透传构造函数（如 resize_mode）
        return {"robotwin": RobotwinMainViewEpisodeReader(**kwargs)}
    if kind == "robotwin_3d":
        return {"robotwin_3d": Robotwin3dEpisodeReader(**options)}
    if kind == "dexjoco":
        from .dexjoco.reader import DexJoCoReader
        return {"dexjoco": DexJoCoReader(**options)}
    if kind == "robodojo":
        from .robodojo import RoboDojoEpisodeReader
        if not options.get("field_map"):
            raise ValueError(
                f"indexed source {source_name!r}: reader type robodojo requires an audited "
                "field_map (see scripts/audit_robodojo_hdf5.py)")
        return {"robodojo": RoboDojoEpisodeReader(**options)}
    raise ValueError(
        f"indexed source {source_name!r} got unknown reader type {kind!r}"
    )


def _mix_adapter_options(config) -> Dict[str, Any]:
    """mix 级 adapter 常量；per-source adapter_options 优先（库内保证）。

    不含 state_action_policy：motus_model_batch_v1 默认即 "real"，而
    motus_native_multimodal_batch_v1（libero 等 state/action 原生维度不同
    的源，见 per-source output_adapter）不接受该字段。
    """
    common = config.common
    options: Dict[str, Any] = {
        "num_video_frames": int(common.num_video_frames),
        "action_video_fps_ratio": int(common.get("video_action_freq_ratio", 1)),
        "video_loss_weight": 1.0,
        "action_loss_weight": 1.0,
    }
    override = config.dataset.get("adapter_options")
    if override:
        options.update(OmegaConf.to_container(override, resolve=True))
    return options


def _indexed_sources(config, *, eval: bool = False, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for child in config.dataset.sources:
        # A single-source evaluation must not validate unrelated training-only
        # pools. Keep full validation for normal multi-source construction.
        if source_name is not None and str(child.get("name") or child.get("dataset")) != source_name:
            continue
        pool_spec = None
        if eval:
            pool_spec = child.get("eval_pool")
        if pool_spec is None:
            pool_spec = child.get("pool")
        if not pool_spec:
            raise ValueError(
                f"indexed source {child.get('name')!r} missing pool"
                + ("/eval_pool" if eval else "")
            )
        pool_spec = OmegaConf.to_container(pool_spec, resolve=True)
        if pool_spec.get("type") in ("task_balanced_ranges", "capability_ranges", "task_cycled_ranges"):
            if eval:
                raise ValueError("task_balanced_ranges is training-only; set eval_pool")
            batch_size = int(config.training.batch_size)
            if int(pool_spec.get("group_size", batch_size)) != batch_size:
                raise ValueError("task_balanced_ranges group_size must equal batch_size")
            pool_spec["group_size"] = batch_size
        # 保留键：供 provider 做构建期 N 与配置 N 的交叉校验（sampling_ranges）
        pool_spec.setdefault(
            "num_video_frames", int(config.common.num_video_frames)
        )
        # prod_keep_only 是 depth_clips 专属选项（sidecar clip 过滤）；其余 pool
        # 类型（sampling_ranges 等）不接受该键，注入会直接构建失败。
        depth_cfg = config.get("depth")
        if depth_cfg is not None and pool_spec.get("type") == "depth_clips":
            pool_spec.setdefault(
                "prod_keep_only",
                bool(depth_cfg.get("prod_keep_only", True)),
            )
        dataset_name = child.get("dataset") or child.get("name")
        source: Dict[str, Any] = {
            "name": str(child.get("name") or dataset_name),
            "dataset": str(dataset_name),
            "sample_pool": pool_factory_from_spec(pool_spec),
        }
        if child.get("weight") is not None:
            source["weight"] = float(child.weight)
        readers = _readers_from_spec(child, source_name=source["name"])
        if readers is not None:
            source["readers"] = readers
        for key in ("family", "release", "catalog_root", "catalog_path", "index_path"):
            if child.get(key) is not None:
                source[key] = child.get(key)
        if eval:
            for key in ("catalog_path", "index_path"):
                if child.get("eval_" + key) is not None:
                    source[key] = child.get("eval_" + key)
        if child.get("output_adapter") is not None:
            # per-source batch profile 覆盖（如 libero 用
            # motus_native_multimodal_batch_v1 支持 state 8D/action 7D 原生异维）
            source["output_adapter"] = str(child.output_adapter)
        if child.get("adapter_options"):
            source["adapter_options"] = OmegaConf.to_container(
                child.adapter_options, resolve=True
            )
        sources.append(source)
    if not sources:
        raise ValueError("dataset.sources must be non-empty for type=indexed_mix")
    return sources


def _install_cycled_resume(config) -> None:
    """task_cycled_ranges 整个 run 只在 cycle_base_step 这一个数据 epoch 内：
    续训时 train.py 的 set_epoch(global_step) 须改成跳过已消费的 step。"""
    bases = {int(child.pool.cycle_base_step) for child in config.dataset.sources
             if child.get("pool") is not None and child.pool.get("type") == "task_cycled_ranges"}
    if len(bases) > 1:
        raise ValueError(f"task_cycled_ranges sources disagree on cycle_base_step: {sorted(bases)}")
    if bases:
        _task_cycled_ranges.install_exact_resume(bases.pop())


def _system_option(config, key: str, default):
    system = config.get("system")
    if system is None:
        return default
    value = system.get(key, default)
    return default if value is None else value


def _background_history_attacher(config, queue_enabled: bool):
    """Producer-side HistoryAttacher, or None to keep the trainer's foreground read.

    Only when the window ignores the training step (no sparse jitter). This
    instance is never shared with the trainer: the trainer keeps its own, whose
    idempotent call recomputes ``last_metrics`` for the batch it trains on.
    """
    if not queue_enabled:
        return None
    attacher = build_history_attacher(config)
    if attacher is None or float(attacher.spec.sparse_jitter) != 0.0:
        return None
    return attacher


def _producer_transform(depth_attacher, history_attacher):
    if depth_attacher is None and history_attacher is None:
        return None

    def transform(batch):
        if depth_attacher is not None:
            batch = depth_attacher.attach(batch)
        if history_attacher is not None:
            # step only seeds sparse jitter, which is 0 on this path.
            batch = history_attacher.attach(batch, step=0)
        return batch

    return transform


_WORKER_TRANSFORM = None


def _init_transform_worker(config) -> None:
    global _WORKER_TRANSFORM
    # The parent's OpenMP pool does not survive fork; a parallel op would hang.
    torch.set_num_threads(1)
    _WORKER_TRANSFORM = _producer_transform(
        build_depth_attacher(config), _background_history_attacher(config, True)
    )


def _run_transform_worker(batch):
    return _WORKER_TRANSFORM(batch)


class _SubprocessTransform:
    """Producer transform in one forked CPU process, so attach work does not
    hold the trainer's GIL. Tensors cross the boundary via shared memory.

    The worker is forked here, before the producer thread starts; it must not
    touch CUDA (forked children cannot re-initialise it) and runs torch ops
    single-threaded.
    """

    def __init__(self, config):
        self._pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=torch_mp.get_context("fork"),
            initializer=_init_transform_worker,
            initargs=(config,),
        )
        self._pool.submit(int).result()

    def __call__(self, batch):
        return self._pool.submit(_run_transform_worker, batch).result()


def create_training_data_pipeline(
    config,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    num_workers: Optional[int] = None,
    timing_events=None,
):
    """构建训练用 indexed mix 管线（batch 内单源、按权重调度）。

    跨 rank 数据源分配由 ``dataset.cross_rank_source_policy`` 选择：
    ``synchronized``（缺省，全 rank 同步同源）或 ``weighted_per_step``
    （同一 data step 的 rank 槽位按源权重分配，见库文档
    docs/CROSS_RANK_SOURCE_SAMPLING_ZH.md）。

    返回 IndexedMixedTrainingDataPipeline：``with pipeline: batch, stats = pipeline.next()``。
    stats 键与旧 ReadyBatchQueue 完全一致（另含 mix_* 字段）。
    """
    start = time.perf_counter()
    sources = _indexed_sources(config)
    _install_cycled_resume(config)
    if num_workers is None:
        num_workers = int(_system_option(config, "num_workers", 0))
    queue_enabled = bool(_system_option(config, "ready_batch_queue_enabled", True))
    # Only queued training benefits from producer-side attachment. When the
    # queue is disabled, UniDiffuserTrainer keeps the synchronous fallback.
    depth_attacher = build_depth_attacher(config) if queue_enabled else None
    history_attacher = _background_history_attacher(config, queue_enabled)
    batch_transform = _producer_transform(depth_attacher, history_attacher)
    transform_subprocess = batch_transform is not None and bool(
        _system_option(config, "ready_batch_transform_subprocess", False)
    )
    if transform_subprocess:
        batch_transform = _SubprocessTransform(config)
    pipeline = open_mixed_training_data(
        sources,
        batch_size=int(config.training.batch_size),
        batches_per_epoch=config.dataset.get("batches_per_epoch"),
        output_adapter=MOTUS_MODEL_BATCH_V1,
        adapter_options=_mix_adapter_options(config),
        video_size=(int(config.common.video_height), int(config.common.video_width)),
        seed=int(seed),
        rank=int(rank),
        world_size=int(world_size),
        queue_enabled=queue_enabled,
        num_workers=num_workers,
        pin_memory=bool(_system_option(config, "pin_memory", False)),
        queue_size=int(_system_option(config, "ready_batch_queue_size", 2)),
        queue_timeout_seconds=float(
            _system_option(config, "ready_batch_queue_timeout_seconds", 600)
        ),
        queue_clone_tensors=bool(
            _system_option(config, "ready_batch_queue_clone_tensors", False)
        ),
        batch_transform=batch_transform,
        shuffle=bool(config.dataset.get("shuffle", True)),
        drop_last=bool(config.dataset.get("drop_last", True)),
        cross_rank_source_policy=str(
            config.dataset.get("cross_rank_source_policy", "synchronized")
        ),
    )
    _append_timing_event(
        timing_events,
        "dataset.create_train_total",
        start,
        dataset_type="indexed_mix",
        source_names=[source["name"] for source in sources],
        depth_attach_background=depth_attacher is not None,
        history_attach_background=history_attacher is not None,
        batch_transform_subprocess=transform_subprocess,
    )
    # Reuse the producer's reader/caches in the trainer-side idempotent fallback.
    # This is repo-owned metadata; motus_data remains generic.
    pipeline._repo_depth_attacher = depth_attacher
    return pipeline


def create_eval_data_pipeline(
    config,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    timing_events=None,
):
    """构建评估用 indexed mix 管线（无 ready queue、不 shuffle、不丢尾批）。"""
    start = time.perf_counter()
    sources = _indexed_sources(config, eval=True)
    pipeline = open_mixed_training_data(
        sources,
        batch_size=int(config.training.batch_size),
        batches_per_epoch=config.dataset.get("batches_per_epoch"),
        output_adapter=MOTUS_MODEL_BATCH_V1,
        adapter_options=_mix_adapter_options(config),
        video_size=(int(config.common.video_height), int(config.common.video_width)),
        seed=int(seed),
        rank=int(rank),
        world_size=int(world_size),
        queue_enabled=False,
        num_workers=int(_system_option(config, "num_workers", 0)),
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    _append_timing_event(
        timing_events,
        "dataset.create_val_total",
        start,
        dataset_type="indexed_mix",
        source_names=[source["name"] for source in sources],
    )
    return pipeline


def _merge_single_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把 N 个单样本 batch dict 沿 batch 维合并（张量 cat / list 拼接 / None 透传）。

    motus_data 的 collate/adapt 按单 episode 约束，评测多样化需要跨 episode
    拼批，这里只使用公开 API 逐样本 adapt 后本地合并，键集与训练 batch 契约
    一致。标量/字典常量键（native adapter 的 batch_adapter_profile、
    state/action_dimension、normalization_modes 等——同 source 逐样本相同）
    取共有值。出现混合类型或不一致的常量即视为实现错误。
    """
    if not batches:
        raise ValueError("cannot merge an empty batch list")
    keys: List[str] = []
    seen = set()
    for batch in batches:
        for key in batch:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    merged: Dict[str, Any] = {}
    for key in keys:
        values = [batch[key] for batch in batches]
        if all(value is None for value in values):
            merged[key] = None
        elif all(isinstance(value, torch.Tensor) for value in values):
            merged[key] = torch.cat([value for value in values], dim=0)
        elif all(isinstance(value, (list, tuple)) for value in values):
            merged[key] = [item for value in values for item in value]
        elif all(value == values[0] for value in values):
            merged[key] = values[0]
        else:
            raise TypeError(
                f"cannot merge eval batch key {key!r}: mixed types "
                f"{[type(value).__name__ for value in values]}"
            )
    return merged


def create_diverse_eval_batch(
    config,
    source_name: str,
    *,
    batch_size: int,
    shard_ordinal: int = 0,
    seed: int = 0,
    use_eval_pool: bool = True,
    episode_id: Optional[str] = None,
    start_index: Optional[int] = None,
) -> Dict[str, Any]:
    """取 indexed 源 holdout（eval_pool，缺省回退 pool）的一条**跨 episode 多样** batch。

    ``use_eval_pool=False`` 时改取训练池（``pool``）——用于在训练集自身的 clip
    上做同口径评测（对比 holdout 可分辨"没学会"与"没泛化"）。

    背景：`EpisodeGroupedBatchSampler` 按 episode 组批——一条 batch 只含同一
    episode 的窗口，长 episode（如 agibot_beta）会独占大量 shard，导致评测
    大量样本是同一任务。本函数改为确定性选出 batch_size 条**不同 episode**
    各取一个窗口（仅当 episode 数不足时回绕，重复 episode 也换不同窗口）：

    - episode 顺序：`random.Random(f"{seed}|{source_name}")` 洗牌（固定 seed
      跨进程/跨轮次一致）；
    - shard k 的第 i 个样本 = order[(k*batch_size + i) % E]；``start_index``
      非空时改为 order[(start_index + i) % E]（多卡拆 holdout 不重叠）。
    - 窗口索引由 `random.Random(f"{seed}|{source_name}|{shard}|{i}")` 在
      episode 组内选；``start_index`` 时用 `|start+i|0`，与 offline bs=1
      逐条拉取对齐。
    - 逐样本 decode + adapter.adapt 后按 batch 维合并（见 _merge_single_batches）。

    纯函数、无 DataLoader（准备进程内 CPU 解码），输出键与训练 batch 契约一致。
    """
    batch_size = max(1, int(batch_size))
    shard_ordinal = max(0, int(shard_ordinal))
    seed = int(seed)

    sources = _indexed_sources(config, eval=bool(use_eval_pool), source_name=source_name)
    source = next((s for s in sources if s["name"] == source_name), None)
    if source is None:
        raise ValueError(f"source {source_name!r} not found in dataset.sources")

    dataset_name = str(source["dataset"])
    release = get_episode_dataset_release(dataset_name, source.get("release"))
    repository = open_dataset(
        dataset_name,
        release=release,
        catalog_root=source.get("catalog_root"),
        catalog_path=source.get("catalog_path"),
        index_path=source.get("index_path"),
        readers=source.get("readers"),
    )
    try:
        pool = resolve_sample_pool(source["sample_pool"], repository, seed=seed)
        dataset = SamplePoolDataset(
            repository,
            pool,
            video_output_format=release.default_video_output_format,
            video_size=(
                int(config.common.video_height),
                int(config.common.video_width),
            ),
            action_magnitude_guard_threshold=(
                release.action_magnitude_guard_threshold
            ),
            action_magnitude_guard_sources=(
                release.action_magnitude_guard_sources
            ),
        )
        adapter_options = _mix_adapter_options(config)
        if source.get("adapter_options"):
            adapter_options.update(source["adapter_options"])
        adapter = build_indexed_batch_adapter(
            source.get("output_adapter") or MOTUS_MODEL_BATCH_V1,
            **adapter_options,
        )

        groups = pool.episode_groups
        episode_ids = sorted(groups)
        if not episode_ids:
            raise ValueError(f"source {source_name!r} has no episodes in eval pool")
        order = list(episode_ids)
        random.Random(f"{seed}|{source_name}").shuffle(order)

        indices: List[int] = []
        if episode_id is not None:
            eid = str(episode_id)
            if eid not in groups:
                raise ValueError(
                    f"episode {eid!r} not in {source_name} pool ({len(groups)} episodes)"
                )
            group = groups[eid]
            pick = random.Random(
                f"{seed}|{source_name}|{shard_ordinal}|0"
            ).randrange(len(group))
            indices.append(int(group[pick]))
        else:
            for i in range(batch_size):
                if start_index is not None:
                    ep_index = int(start_index) + i
                    shard_for_rng = int(start_index) + i
                    i_for_rng = 0
                else:
                    ep_index = shard_ordinal * batch_size + i
                    shard_for_rng = shard_ordinal
                    i_for_rng = i
                eid = order[ep_index % len(order)]
                group = groups[eid]
                pick = random.Random(
                    f"{seed}|{source_name}|{shard_for_rng}|{i_for_rng}"
                ).randrange(len(group))
                indices.append(int(group[pick]))

        single_batches: List[Dict[str, Any]] = []
        for index in indices:
            samples = dataset.get_grouped_samples([index])
            adapted = adapter.adapt(
                IndexedEpisodeBatch(info=samples[0].info, samples=samples)
            )
            single_batches.append(dict(adapted))
        return _merge_single_batches(single_batches)
    finally:
        repository.close()
