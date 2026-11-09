"""Config-driven construction of source datasets and weighted mixtures."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from motus_data.datasets.agibot_beta_grouped import AgibotBetaGroupedManifestDataset
from motus_data.datasets.agibot_beta import AgibotBetaDataset
from motus_data.datasets.agibot_native import AgibotDataset
from motus_data.datasets.manifest import MotusManifestDataset
from motus_data.datasets.robotmind import RobotMindDataset
from motus_data.datasets.synthetic import SyntheticMotusDataset
from motus_data.mixing.mix_dataset import MixDataset
from motus_data.registry import DATASETS, DatasetRegistry


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return value


def _register_builtins() -> None:
    if "motus_manifest" not in DATASETS.names():
        DATASETS.register("motus_manifest", _build_manifest)
    if "synthetic" not in DATASETS.names():
        DATASETS.register("synthetic", _build_synthetic)
    if "agibot_native" not in DATASETS.names():
        DATASETS.register("agibot_native", _build_agibot_native)
    if "agibot" not in DATASETS.names():
        DATASETS.register("agibot", _build_agibot_native)
    if "agibot_beta" not in DATASETS.names():
        DATASETS.register("agibot_beta", _build_agibot_beta)
    if "robotmind" not in DATASETS.names():
        DATASETS.register("robotmind", _build_robotmind)


def _build_manifest(*, node: Dict[str, Any], common: Dict[str, Any], split: str, **_):
    params = {
        "manifest_path": node["manifest_path"],
        "split": split,
        "val": split in {"val", "validation"},
        "video_size": (
            int(common.get("video_height", 320)),
            int(common.get("video_width", 384)),
        ),
        "num_video_frames": int(common.get("num_video_frames", 48)),
        "video_action_freq_ratio": int(common.get("video_action_freq_ratio", 1)),
        "action_dim": int(common.get("action_dim", 26)),
        "state_dim": int(common.get("state_dim", 26)),
        "source_dataset": node.get("source_dataset", node.get("name")),
        "robot_name": node.get("robot_name"),
        "instruction_lang": node.get("instruction_lang", "default"),
        "max_samples": node.get("max_samples"),
        "max_decode_retries": int(node.get("max_decode_retries", 256)),
    }
    for key in (
        "action_magnitude_guard_threshold",
        "action_magnitude_guard_sources",
        "grouped_index_cache_path",
    ):
        if key in node:
            params[key] = node[key]

    if bool(node.get("grouped_video_loader", False)):
        source = str(params.get("source_dataset") or "")
        if source != "agibot_beta":
            raise ValueError(
                "The standalone v1 library currently supports grouped_video_loader "
                f"only for agibot_beta, got {source!r}"
            )
        return AgibotBetaGroupedManifestDataset(**params)
    params.pop("grouped_index_cache_path", None)
    return MotusManifestDataset(**params)


def _build_synthetic(*, node: Dict[str, Any], common: Dict[str, Any], **_):
    return SyntheticMotusDataset(
        length=int(node.get("length", 128)),
        num_video_frames=int(common.get("num_video_frames", node.get("num_video_frames", 4))),
        video_size=(
            int(common.get("video_height", node.get("video_height", 16))),
            int(common.get("video_width", node.get("video_width", 16))),
        ),
        action_dim=int(common.get("action_dim", node.get("action_dim", 6))),
        state_dim=int(common.get("state_dim", node.get("state_dim", 6))),
        source_dataset=node.get("source_dataset", node.get("name", "synthetic")),
        source_child=node.get("name"),
        include_action=bool(node.get("include_action", True)),
        sleep_seconds=float(node.get("sleep_seconds", 0.0)),
        seed=int(node.get("seed", 0)),
    )


def _build_agibot_native(
    *, node: Dict[str, Any], common: Dict[str, Any], split: str, rank: int, seed: int, **_
):
    return AgibotDataset(
        annotation_folders=node["annotation_folders"],
        data_root_folders=node["data_root_folders"],
        dataset_names=node.get("dataset_names", ["agibot-world"]),
        split=split,
        video_size=(
            int(common.get("video_height", 320)),
            int(common.get("video_width", 384)),
        ),
        num_video_frames=int(common.get("num_video_frames", 48)),
        action_dim=int(common.get("action_dim", 26)),
        state_dim=int(common.get("state_dim", 26)),
        source_dataset=node.get("source_dataset", "agibot"),
        source_child=node.get("name", "agibot_native"),
        robot_name=node.get("robot_name", "agibot-a2-d"),
        test_fraction=float(node.get("test_fraction", 0.01)),
        split_seed=int(node.get("split_seed", 20260720)),
        virtual_samples_per_episode=int(node.get("virtual_samples_per_episode", 1)),
        max_decode_retries=int(node.get("max_decode_retries", 16)),
        seed=int(node.get("seed", seed)),
        rank=rank,
    )


def _build_agibot_beta(
    *, node: Dict[str, Any], common: Dict[str, Any], split: str, rank: int, seed: int, **_
):
    return AgibotBetaDataset(
        data_root=node["data_root"],
        tar_catalog_dir=node["tar_catalog_dir"],
        split=split,
        video_size=(
            int(common.get("video_height", 320)),
            int(common.get("video_width", 384)),
        ),
        num_video_frames=int(common.get("num_video_frames", 48)),
        action_video_fps_ratio=int(
            node.get(
                "action_video_fps_ratio",
                common.get("video_action_freq_ratio", 1),
            )
        ),
        action_dim=int(common.get("action_dim", 26)),
        state_dim=int(common.get("state_dim", 26)),
        source_dataset=node.get("source_dataset", "agibot_beta"),
        source_child=node.get("name", "agibot_beta"),
        robot_name=node.get("robot_name", "agibot-a2-d"),
        test_fraction=float(node.get("test_fraction", 0.01)),
        split_seed=int(node.get("split_seed", 20260720)),
        virtual_samples_per_episode=int(node.get("virtual_samples_per_episode", 64)),
        max_decode_retries=int(node.get("max_decode_retries", 16)),
        video_decode_backend=node.get("video_decode_backend", "ffmpeg"),
        action_magnitude_guard_threshold=node.get(
            "action_magnitude_guard_threshold", 100.0
        ),
        excluded_task_names=node.get(
            "excluded_task_names",
            [
                "Water the flowers",
                "Print documents with a printer.",
                "Pack the medicine",
                "Clap hands",
                "Wave goodbye",
            ],
        ),
        seed=int(node.get("seed", seed)),
        rank=rank,
    )


def _build_robotmind(
    *, node: Dict[str, Any], common: Dict[str, Any], split: str, rank: int, seed: int, **_
):
    ratio = int(
        node.get("action_video_fps_ratio", common.get("video_action_freq_ratio", 1))
    )
    return RobotMindDataset(
        data_root=node["data_root"],
        embodiment=node["embodiment"],
        split=split,
        video_size=(
            int(common.get("video_height", 320)),
            int(common.get("video_width", 384)),
        ),
        num_video_frames=int(common.get("num_video_frames", 48)),
        video_frame_step=int(node.get("video_frame_step", ratio)),
        action_video_fps_ratio=ratio,
        instruction_lang=node.get("instruction_lang", "en"),
        source_dataset=node.get("source_dataset", "robotmind"),
        source_child=node.get("name"),
        robot_name=node.get("robot_name"),
        test_fraction=float(node.get("test_fraction", 0.01)),
        split_seed=int(node.get("split_seed", 20260720)),
        virtual_samples_per_episode=int(node.get("virtual_samples_per_episode", 8)),
        max_decode_retries=int(node.get("max_decode_retries", 16)),
        resize_mode=node.get("resize_mode", "interpolate"),
        video_loss_weight=float(node.get("video_loss_weight", 1.0)),
        action_loss_weight=float(node.get("action_loss_weight", 1.0)),
        seed=int(node.get("seed", seed)),
        rank=rank,
    )


def _build_node(
    node: Dict[str, Any],
    common: Dict[str, Any],
    requested_split: Optional[str],
    registry: DatasetRegistry,
    context: Dict[str, int],
):
    dataset_type = str(node.get("type", "")).lower()
    if not dataset_type:
        raise ValueError("dataset.type is required")
    if dataset_type == "mix":
        datasets = []
        names = []
        weights = []
        children = node.get("datasets") or []
        if not children:
            raise ValueError("mix dataset requires dataset.datasets")
        for position, raw_child in enumerate(children):
            child = _plain(raw_child)
            child_common = copy.deepcopy(common)
            child_common.update(_plain(child.pop("common", {})))
            name = str(child.get("name", f"{child.get('type', 'dataset')}_{position}"))
            weight = float(child.pop("weight", 1.0))
            child_split = str(requested_split or child.get("split", "train")).lower()
            datasets.append(_build_node(child, child_common, child_split, registry, context))
            names.append(name)
            weights.append(weight)
        return MixDataset(
            datasets=datasets,
            names=names,
            weights=weights,
            batches_per_epoch=node.get("batches_per_epoch"),
        )

    split = str(requested_split or node.get("split", "train")).lower()
    return registry.build(dataset_type, node=node, common=common, split=split, **context)


def build_dataset(
    config: Any,
    split: Optional[str] = None,
    registry: DatasetRegistry = DATASETS,
    *,
    rank: int = 0,
    seed: int = 0,
):
    """Build one dataset or a weighted mixture from a full or dataset-only config."""

    _register_builtins()
    plain = _plain(config)
    if "dataset" in plain:
        node = _plain(plain["dataset"])
        common = _plain(plain.get("common", {}))
    else:
        node = plain
        common = _plain(node.pop("common", {}))
    return _build_node(node, common, split, registry, {"rank": int(rank), "seed": int(seed)})
