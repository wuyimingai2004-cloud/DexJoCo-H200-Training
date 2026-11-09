"""Contracts for versioned Episode Catalogs and model-owned SamplePools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


MOTUS_EPISODE_CATALOG_METADATA_V1 = "motus_episode_catalog_metadata_v1"
MOTUS_EPISODE_RECORD_V1 = "motus_episode_record_v1"
MOTUS_SAMPLE_POOL_HEADER_V1 = "motus_sample_pool_header_v1"
MOTUS_EPISODE_CATALOG_METADATA_V2 = "motus_episode_catalog_v2"
# V2 Episode rows deliberately carry no per-record format field. This constant
# remains public only as a semantic API label.
MOTUS_EPISODE_RECORD_V2 = "motus_episode_v2"
MOTUS_SAMPLE_POOL_HEADER_V2 = "motus_sample_pool_v2"
MOTUS_SAMPLE_SELECTION_V1 = "motus_sample_selection_v1"


class IndexContractError(ValueError):
    """Raised when a Catalog or SamplePool record violates its frozen contract."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IndexContractError(f"{context} must be an object")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None,
    context: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise IndexContractError(f"{context} missing fields: {missing}")
    allowed = required | (optional or set())
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise IndexContractError(f"{context} has unknown fields: {unknown}")


def _string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise IndexContractError(f"{context} must be {requirement}")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IndexContractError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, context: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexContractError(f"{context} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise IndexContractError(
            f"{context} must be within [{minimum}, {maximum}], got {result}"
        )
    return result


def _json_serializable(value: Any, context: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise IndexContractError(f"{context} must be finite JSON data: {error}") from error


def _validate_split(value: Any, context: str) -> str:
    split = _string(value, context).lower()
    if split not in {"train", "test"}:
        raise IndexContractError(f"{context} must be 'train' or 'test', got {split!r}")
    return split


def validate_catalog_metadata_v1(value: Mapping[str, Any]) -> None:
    metadata = _mapping(value, "catalog metadata")
    _keys(
        metadata,
        required={
            "schema_version",
            "record_schema_version",
            "catalog_id",
            "source_dataset",
            "source_child",
            "data_version",
            "split_policy",
        },
        optional={"record_count", "statistics"},
        context="catalog metadata",
    )
    if metadata["schema_version"] != MOTUS_EPISODE_CATALOG_METADATA_V1:
        raise IndexContractError("unsupported catalog metadata schema_version")
    if metadata["record_schema_version"] != MOTUS_EPISODE_RECORD_V1:
        raise IndexContractError("unsupported catalog record_schema_version")
    for key in ("catalog_id", "source_dataset", "source_child", "data_version"):
        _string(metadata[key], f"catalog metadata.{key}")
    split_policy = _mapping(metadata["split_policy"], "catalog metadata.split_policy")
    _keys(
        split_policy,
        required={"type", "seed", "test_fraction"},
        optional=None,
        context="catalog metadata.split_policy",
    )
    if split_policy["type"] != "stable_ranked_episode_hash":
        raise IndexContractError(
            "catalog metadata.split_policy.type must be 'stable_ranked_episode_hash'"
        )
    _integer(split_policy["seed"], "catalog metadata.split_policy.seed")
    test_fraction = _number(
        split_policy["test_fraction"],
        "catalog metadata.split_policy.test_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if test_fraction >= 1.0:
        raise IndexContractError(
            "catalog metadata.split_policy.test_fraction must be within [0, 1)"
        )
    if "record_count" in metadata:
        _integer(metadata["record_count"], "catalog metadata.record_count")
    if "statistics" in metadata:
        _mapping(metadata["statistics"], "catalog metadata.statistics")
    _json_serializable(metadata, "catalog metadata")


def validate_catalog_metadata_v2(value: Mapping[str, Any]) -> None:
    """Validate concise split-neutral complete-Episode inventory metadata."""

    metadata = _mapping(value, "catalog metadata")
    _keys(
        metadata,
        required={
            "format",
            "catalog_id",
            "source",
            "child",
            "episodes",
            "sha256",
        },
        optional={"meta"},
        context="catalog metadata",
    )
    if metadata["format"] != MOTUS_EPISODE_CATALOG_METADATA_V2:
        raise IndexContractError("unsupported catalog metadata format")
    for key in ("catalog_id", "source", "child", "sha256"):
        _string(metadata[key], f"catalog metadata.{key}")
    _integer(metadata["episodes"], "catalog metadata.episodes", minimum=1)
    if len(metadata["sha256"]) != 64:
        raise IndexContractError("catalog metadata.sha256 must be 64 hex characters")
    try:
        int(metadata["sha256"], 16)
    except ValueError as error:
        raise IndexContractError(
            "catalog metadata.sha256 must be 64 hex characters"
        ) from error
    if "meta" in metadata:
        _mapping(metadata["meta"], "catalog metadata.meta")
    _json_serializable(metadata, "catalog metadata")


def validate_catalog_metadata(value: Mapping[str, Any]) -> None:
    metadata = _mapping(value, "catalog metadata")
    version = metadata.get("schema_version", metadata.get("format"))
    if version == MOTUS_EPISODE_CATALOG_METADATA_V1:
        validate_catalog_metadata_v1(value)
        return
    if version == MOTUS_EPISODE_CATALOG_METADATA_V2:
        validate_catalog_metadata_v2(value)
        return
    raise IndexContractError(
        f"unsupported catalog metadata format={version!r}"
    )


def _validate_locator(value: Any, context: str) -> None:
    locator = _mapping(value, context)
    _keys(
        locator,
        required={"storage_backend", "uri"},
        optional={"member", "offset_data", "size"},
        context=context,
    )
    backend = _string(locator["storage_backend"], f"{context}.storage_backend")
    _string(locator["uri"], f"{context}.uri")
    if backend == "tar_member":
        for key in ("member", "offset_data", "size"):
            if key not in locator:
                raise IndexContractError(f"{context} tar_member requires {key}")
        _string(locator["member"], f"{context}.member")
        _integer(locator["offset_data"], f"{context}.offset_data")
        _integer(locator["size"], f"{context}.size", minimum=1)
    else:
        if "member" in locator:
            _string(locator["member"], f"{context}.member")
        if "offset_data" in locator:
            _integer(locator["offset_data"], f"{context}.offset_data")
        if "size" in locator:
            _integer(locator["size"], f"{context}.size", minimum=1)


def _validate_episode_record(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    require_split: bool,
) -> None:
    record = _mapping(value, "episode record")
    required = {
        "schema_version",
        "episode_id",
        "source_dataset",
        "source_child",
        "robot_name",
        "video",
        "state",
        "action",
        "text",
        "metadata",
    }
    if require_split:
        required.add("split")
    _keys(
        record,
        required=required,
        optional=None,
        context="episode record",
    )
    if record["schema_version"] != schema_version:
        raise IndexContractError("unsupported episode record schema_version")
    episode_id = _string(record["episode_id"], "episode record.episode_id")
    source_dataset = _string(
        record["source_dataset"], "episode record.source_dataset"
    )
    _string(record["source_child"], "episode record.source_child")
    _string(record["robot_name"], "episode record.robot_name")
    if require_split:
        _validate_split(record["split"], "episode record.split")
    if not episode_id.startswith(f"{source_dataset}:"):
        raise IndexContractError(
            "episode record.episode_id must begin with '<source_dataset>:'"
        )

    video = _mapping(record["video"], "episode record.video")
    _keys(
        video,
        required={"frame_count", "locator"},
        optional={"metadata"},
        context="episode record.video",
    )
    video_frame_count = _integer(
        video["frame_count"], "episode record.video.frame_count", minimum=1
    )
    _validate_locator(video["locator"], "episode record.video.locator")
    if "metadata" in video:
        _mapping(video["metadata"], "episode record.video.metadata")

    for modality_name in ("state", "action"):
        modality = _mapping(record[modality_name], f"episode record.{modality_name}")
        _keys(
            modality,
            required={"count", "locator"},
            optional={"metadata"},
            context=f"episode record.{modality_name}",
        )
        _integer(
            modality["count"], f"episode record.{modality_name}.count", minimum=1
        )
        _validate_locator(
            modality["locator"], f"episode record.{modality_name}.locator"
        )
        if "metadata" in modality:
            _mapping(modality["metadata"], f"episode record.{modality_name}.metadata")

    text = _mapping(record["text"], "episode record.text")
    _keys(
        text,
        required={"items"},
        optional=None,
        context="episode record.text",
    )
    items = text["items"]
    if not isinstance(items, list) or not items:
        raise IndexContractError("episode record.text.items must be a non-empty list")
    for position, raw_item in enumerate(items):
        context = f"episode record.text.items[{position}]"
        item = _mapping(raw_item, context)
        _keys(
            item,
            required={"index", "kind", "text"},
            optional={"frame_range", "metadata"},
            context=context,
        )
        index = _integer(item["index"], f"{context}.index")
        if index != position:
            raise IndexContractError(
                "episode record.text item indices must equal their list positions"
            )
        _string(item["kind"], f"{context}.kind")
        _string(item["text"], f"{context}.text", allow_empty=True)
        if "frame_range" in item:
            frame_range = item["frame_range"]
            if not isinstance(frame_range, list) or len(frame_range) != 2:
                raise IndexContractError(f"{context}.frame_range must be [start, end)")
            start = _integer(frame_range[0], f"{context}.frame_range[0]")
            end = _integer(frame_range[1], f"{context}.frame_range[1]", minimum=1)
            if end <= start or end > video_frame_count:
                raise IndexContractError(
                    f"{context}.frame_range must satisfy 0 <= start < end <= video count"
                )
        if "metadata" in item:
            _mapping(item["metadata"], f"{context}.metadata")

    _mapping(record["metadata"], "episode record.metadata")
    _json_serializable(record, "episode record")


def validate_episode_record_v1(value: Mapping[str, Any]) -> None:
    _validate_episode_record(
        value,
        schema_version=MOTUS_EPISODE_RECORD_V1,
        require_split=True,
    )


def _validate_storage_v2(value: Any, context: str) -> None:
    storage = _mapping(value, context)
    _keys(
        storage,
        required={"type", "path"},
        optional={"member", "offset", "size", "parts"},
        context=context,
    )
    storage_type = _string(storage["type"], f"{context}.type")
    _string(storage["path"], f"{context}.path")
    if storage_type == "tar":
        for key in ("member", "offset", "size"):
            if key not in storage:
                raise IndexContractError(f"{context} type='tar' requires {key}")
        _string(storage["member"], f"{context}.member")
        _integer(storage["offset"], f"{context}.offset")
        _integer(storage["size"], f"{context}.size", minimum=1)
    else:
        if "member" in storage:
            _string(storage["member"], f"{context}.member")
        if "offset" in storage:
            _integer(storage["offset"], f"{context}.offset")
        if "size" in storage:
            _integer(storage["size"], f"{context}.size", minimum=1)
    if "parts" in storage:
        parts = storage["parts"]
        if (
            not isinstance(parts, list)
            or len(parts) < 2
            or any(not isinstance(path, str) or not path for path in parts)
        ):
            raise IndexContractError(
                f"{context}.parts must contain at least two non-empty paths"
            )
        if storage_type != "zip_split":
            raise IndexContractError(
                f"{context}.parts is supported only for type='zip_split'"
            )
    if storage_type == "zip_split" and "parts" not in storage:
        raise IndexContractError(
            f"{context} type='zip_split' requires parts"
        )


def validate_episode_record_v2(value: Mapping[str, Any]) -> None:
    """Validate one concise v2 Episode row with no repeated format/source."""

    record = _mapping(value, "episode record")
    _keys(
        record,
        required={
            "episode_id",
            "robot",
            "video",
            "state",
            "action",
            "texts",
        },
        optional={"meta"},
        context="episode record",
    )
    _string(record["episode_id"], "episode record.episode_id")
    _string(record["robot"], "episode record.robot")

    video_count = None
    for modality_name in ("video", "state", "action"):
        modality = _mapping(
            record[modality_name], f"episode record.{modality_name}"
        )
        _keys(
            modality,
            required={"count", "storage"},
            optional={"meta"},
            context=f"episode record.{modality_name}",
        )
        count = _integer(
            modality["count"],
            f"episode record.{modality_name}.count",
            minimum=1 if modality_name == "video" else 0,
        )
        if modality_name == "video":
            video_count = count
        _validate_storage_v2(
            modality["storage"],
            f"episode record.{modality_name}.storage",
        )
        storage_type = str(modality["storage"]["type"])
        if modality_name != "video":
            if count == 0 and storage_type != "none":
                raise IndexContractError(
                    f"episode record.{modality_name} count=0 requires "
                    "storage.type='none'"
                )
            if count > 0 and storage_type == "none":
                raise IndexContractError(
                    f"episode record.{modality_name} non-empty count cannot "
                    "use storage.type='none'"
                )
        if "meta" in modality:
            _mapping(
                modality["meta"],
                f"episode record.{modality_name}.meta",
            )

    texts = record["texts"]
    if not isinstance(texts, list) or not texts:
        raise IndexContractError("episode record.texts must be a non-empty list")
    for position, raw_item in enumerate(texts):
        context = f"episode record.texts[{position}]"
        item = _mapping(raw_item, context)
        _keys(
            item,
            required={"index", "kind", "text"},
            optional={"video_range", "meta"},
            context=context,
        )
        index = _integer(item["index"], f"{context}.index")
        if index != position:
            raise IndexContractError(
                "episode record text indices must equal their list positions"
            )
        _string(item["kind"], f"{context}.kind")
        _string(item["text"], f"{context}.text", allow_empty=True)
        if "video_range" in item:
            video_range = item["video_range"]
            if not isinstance(video_range, list) or len(video_range) != 2:
                raise IndexContractError(
                    f"{context}.video_range must be [start, end)"
                )
            start = _integer(video_range[0], f"{context}.video_range[0]")
            end = _integer(
                video_range[1],
                f"{context}.video_range[1]",
                minimum=1,
            )
            if end <= start or end > int(video_count):
                raise IndexContractError(
                    f"{context}.video_range must satisfy "
                    "0 <= start < end <= video count"
                )
        if "meta" in item:
            _mapping(item["meta"], f"{context}.meta")

    if "meta" in record:
        _mapping(record["meta"], "episode record.meta")
    _json_serializable(record, "episode record")


def validate_episode_record(value: Mapping[str, Any]) -> None:
    record = _mapping(value, "episode record")
    version = record.get("schema_version")
    if version == MOTUS_EPISODE_RECORD_V1:
        validate_episode_record_v1(value)
        return
    if version is None and "robot" in record and "texts" in record:
        validate_episode_record_v2(value)
        return
    raise IndexContractError(
        f"unsupported episode record format={version!r}"
    )


def validate_sample_pool_header_v1(value: Mapping[str, Any]) -> None:
    metadata = _mapping(value, "sample pool header")
    _keys(
        metadata,
        required={
            "schema_version",
            "record_type",
            "record_schema_version",
            "sample_pool_id",
            "catalog_id",
            "source_dataset",
            "source_child",
            "split",
        },
        optional={"record_count", "generation_metadata"},
        context="sample pool header",
    )
    if metadata["schema_version"] != MOTUS_SAMPLE_POOL_HEADER_V1:
        raise IndexContractError("unsupported sample pool header schema_version")
    if metadata["record_type"] != "sample_pool_header":
        raise IndexContractError("sample pool header.record_type must be 'sample_pool_header'")
    if metadata["record_schema_version"] != MOTUS_SAMPLE_SELECTION_V1:
        raise IndexContractError("unsupported sample pool record_schema_version")
    for key in ("sample_pool_id", "catalog_id", "source_dataset", "source_child"):
        _string(metadata[key], f"sample pool header.{key}")
    _validate_split(metadata["split"], "sample pool header.split")
    if "record_count" in metadata:
        _integer(metadata["record_count"], "sample pool header.record_count")
    if "generation_metadata" in metadata:
        _mapping(metadata["generation_metadata"], "sample pool header.generation_metadata")
    _json_serializable(metadata, "sample pool header")


def validate_sample_pool_header_v2(value: Mapping[str, Any]) -> None:
    """Validate a concise split-neutral Pool bound by catalog_id."""

    metadata = _mapping(value, "sample pool header")
    _keys(
        metadata,
        required={
            "format",
            "pool_id",
            "catalog_id",
        },
        optional={"samples", "meta"},
        context="sample pool header",
    )
    if metadata["format"] != MOTUS_SAMPLE_POOL_HEADER_V2:
        raise IndexContractError("unsupported sample pool header format")
    for key in ("pool_id", "catalog_id"):
        _string(metadata[key], f"sample pool header.{key}")
    if "samples" in metadata:
        _integer(metadata["samples"], "sample pool header.samples", minimum=1)
    if "meta" in metadata:
        _mapping(metadata["meta"], "sample pool header.meta")
    _json_serializable(metadata, "sample pool header")


def validate_sample_pool_header(value: Mapping[str, Any]) -> None:
    metadata = _mapping(value, "sample pool header")
    version = metadata.get("schema_version", metadata.get("format"))
    if version == MOTUS_SAMPLE_POOL_HEADER_V1:
        validate_sample_pool_header_v1(value)
        return
    if version == MOTUS_SAMPLE_POOL_HEADER_V2:
        validate_sample_pool_header_v2(value)
        return
    raise IndexContractError(
        f"unsupported sample pool header format={version!r}"
    )


def validate_sample_selection_v1(value: Mapping[str, Any]) -> None:
    selection = _mapping(value, "sample selection")
    _keys(
        selection,
        required={
            "episode_id",
            "video_frame_indices",
            "state_indices",
            "action_indices",
            "text_indices",
        },
        optional=None,
        context="sample selection",
    )
    _string(selection["episode_id"], "sample selection.episode_id")
    for key, minimum_length in (
        ("video_frame_indices", 1),
        ("state_indices", 0),
        ("action_indices", 0),
        ("text_indices", 0),
    ):
        indices = selection[key]
        if not isinstance(indices, list) or len(indices) < minimum_length:
            raise IndexContractError(
                f"sample selection.{key} must be a list with at least {minimum_length} items"
            )
        for position, index in enumerate(indices):
            _integer(index, f"sample selection.{key}[{position}]")
    _json_serializable(selection, "sample selection")


def validate_agibot_beta_selection_v1(
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> None:
    """Validate exact model-owned indices against one Beta Episode."""

    validate_sample_selection_v1(selection)
    validate_episode_record_v1(episode)
    if episode["source_dataset"] != "agibot_beta":
        raise IndexContractError("Agibot Beta policy requires source_dataset=agibot_beta")
    if selection["episode_id"] != episode["episode_id"]:
        raise IndexContractError("sample selection episode_id does not match episode record")
    modality_counts = {
        "video_frame_indices": int(episode["video"]["frame_count"]),
        "state_indices": int(episode["state"]["count"]),
        "action_indices": int(episode["action"]["count"]),
        "text_indices": len(episode["text"]["items"]),
    }
    for key, count in modality_counts.items():
        invalid = [int(index) for index in selection[key] if int(index) >= count]
        if invalid:
            raise IndexContractError(
                f"sample selection.{key} contains out-of-bounds index {invalid[0]} "
                f"for count {count}"
            )


def validate_agibot_beta_selection_v2(
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> None:
    """Validate exact model-owned indices against one split-neutral Episode."""

    validate_sample_selection_v1(selection)
    validate_episode_record_v2(episode)
    if not str(episode["episode_id"]).startswith("agibot_beta:"):
        raise IndexContractError(
            "Agibot Beta policy requires episode_id prefix 'agibot_beta:'"
        )
    if selection["episode_id"] != episode["episode_id"]:
        raise IndexContractError(
            "sample selection episode_id does not match episode record"
        )
    modality_counts = {
        "video_frame_indices": int(episode["video"]["count"]),
        "state_indices": int(episode["state"]["count"]),
        "action_indices": int(episode["action"]["count"]),
        "text_indices": len(episode["texts"]),
    }
    for key, count in modality_counts.items():
        invalid = [
            int(index) for index in selection[key] if int(index) >= count
        ]
        if invalid:
            raise IndexContractError(
                f"sample selection.{key} contains out-of-bounds index "
                f"{invalid[0]} for count {count}"
            )


def validate_agibot_beta_selection(
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> None:
    record = _mapping(episode, "episode record")
    version = record.get("schema_version")
    if version == MOTUS_EPISODE_RECORD_V1:
        validate_agibot_beta_selection_v1(selection, episode)
        return
    if version is None and "robot" in record and "texts" in record:
        validate_agibot_beta_selection_v2(selection, episode)
        return
    raise IndexContractError(
        f"unsupported episode record format={version!r}"
    )


def validate_pool_catalog_binding_v1(
    pool_metadata: Mapping[str, Any], catalog_metadata: Mapping[str, Any]
) -> None:
    validate_sample_pool_header_v1(pool_metadata)
    validate_catalog_metadata_v1(catalog_metadata)
    for key in ("catalog_id", "source_dataset", "source_child"):
        if pool_metadata[key] != catalog_metadata[key]:
            raise IndexContractError(
                f"sample pool {key}={pool_metadata[key]!r} does not match "
                f"catalog {key}={catalog_metadata[key]!r}"
            )


def validate_pool_catalog_binding_v2(
    pool_metadata: Mapping[str, Any], catalog_metadata: Mapping[str, Any]
) -> None:
    validate_sample_pool_header_v2(pool_metadata)
    validate_catalog_metadata_v2(catalog_metadata)
    if pool_metadata["catalog_id"] != catalog_metadata["catalog_id"]:
        raise IndexContractError(
            f"sample pool catalog_id={pool_metadata['catalog_id']!r} does not match "
            f"catalog catalog_id={catalog_metadata['catalog_id']!r}"
        )


def validate_pool_catalog_binding(
    pool_metadata: Mapping[str, Any], catalog_metadata: Mapping[str, Any]
) -> None:
    pool = _mapping(pool_metadata, "sample pool header")
    catalog = _mapping(catalog_metadata, "catalog metadata")
    pool_version = pool.get("schema_version", pool.get("format"))
    catalog_version = catalog.get("schema_version", catalog.get("format"))
    if (
        pool_version == MOTUS_SAMPLE_POOL_HEADER_V1
        and catalog_version == MOTUS_EPISODE_CATALOG_METADATA_V1
    ):
        validate_pool_catalog_binding_v1(pool_metadata, catalog_metadata)
        return
    if (
        pool_version == MOTUS_SAMPLE_POOL_HEADER_V2
        and catalog_version == MOTUS_EPISODE_CATALOG_METADATA_V2
    ):
        validate_pool_catalog_binding_v2(pool_metadata, catalog_metadata)
        return
    raise IndexContractError(
        "SamplePool and Catalog protocol versions are incompatible: "
        f"pool={pool_version!r} catalog={catalog_version!r}"
    )


def validate_episode_catalog_binding_v1(
    episode: Mapping[str, Any], catalog_metadata: Mapping[str, Any]
) -> None:
    validate_episode_record_v1(episode)
    validate_catalog_metadata_v1(catalog_metadata)
    for key in ("source_dataset", "source_child"):
        if episode[key] != catalog_metadata[key]:
            raise IndexContractError(
                f"episode {key}={episode[key]!r} does not match "
                f"catalog {key}={catalog_metadata[key]!r}"
            )


def validate_episode_catalog_binding_v2(
    episode: Mapping[str, Any], catalog_metadata: Mapping[str, Any]
) -> None:
    validate_episode_record_v2(episode)
    validate_catalog_metadata_v2(catalog_metadata)
    expected_prefix = f"{catalog_metadata['source']}:"
    if not str(episode["episode_id"]).startswith(expected_prefix):
        raise IndexContractError(
            f"episode_id={episode['episode_id']!r} must begin with "
            f"catalog source prefix {expected_prefix!r}"
        )


def validate_episode_catalog_binding(
    episode: Mapping[str, Any], catalog_metadata: Mapping[str, Any]
) -> None:
    record = _mapping(episode, "episode record")
    catalog = _mapping(catalog_metadata, "catalog metadata")
    record_version = record.get("schema_version")
    catalog_version = catalog.get("schema_version", catalog.get("format"))
    if (
        record_version == MOTUS_EPISODE_RECORD_V1
        and catalog_version == MOTUS_EPISODE_CATALOG_METADATA_V1
    ):
        validate_episode_catalog_binding_v1(episode, catalog_metadata)
        return
    if (
        record_version is None
        and "robot" in record
        and "texts" in record
        and catalog_version == MOTUS_EPISODE_CATALOG_METADATA_V2
    ):
        validate_episode_catalog_binding_v2(episode, catalog_metadata)
        return
    raise IndexContractError(
        "Episode and Catalog protocol versions are incompatible: "
        f"episode={record_version!r} catalog={catalog_version!r}"
    )


def validate_selection_pool_binding_v1(
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
    pool_metadata: Mapping[str, Any],
) -> None:
    validate_sample_selection_v1(selection)
    validate_episode_record_v1(episode)
    validate_sample_pool_header_v1(pool_metadata)
    if selection["episode_id"] != episode["episode_id"]:
        raise IndexContractError("sample selection episode_id does not match episode record")
    for key in ("source_dataset", "source_child", "split"):
        if episode[key] != pool_metadata[key]:
            raise IndexContractError(
                f"episode {key}={episode[key]!r} does not match "
                f"sample pool {key}={pool_metadata[key]!r}"
            )
    modality_counts = {
        "video_frame_indices": int(episode["video"]["frame_count"]),
        "state_indices": int(episode["state"]["count"]),
        "action_indices": int(episode["action"]["count"]),
        "text_indices": len(episode["text"]["items"]),
    }
    for key, count in modality_counts.items():
        for index in selection[key]:
            if int(index) >= count:
                raise IndexContractError(
                    f"sample selection.{key} index {index} is out of bounds for count {count}"
                )


def _validate_selection_bounds(
    selection: Mapping[str, Any], episode: Mapping[str, Any]
) -> None:
    is_v2 = "robot" in episode and "texts" in episode
    modality_counts = {
        "video_frame_indices": int(
            episode["video"]["count" if is_v2 else "frame_count"]
        ),
        "state_indices": int(episode["state"]["count"]),
        "action_indices": int(episode["action"]["count"]),
        "text_indices": len(
            episode["texts"] if is_v2 else episode["text"]["items"]
        ),
    }
    for key, count in modality_counts.items():
        for index in selection[key]:
            if int(index) >= count:
                raise IndexContractError(
                    f"sample selection.{key} index {index} is out of bounds for count {count}"
                )


def validate_selection_pool_binding_v2(
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
    pool_metadata: Mapping[str, Any],
) -> None:
    validate_sample_selection_v1(selection)
    validate_episode_record_v2(episode)
    validate_sample_pool_header_v2(pool_metadata)
    if selection["episode_id"] != episode["episode_id"]:
        raise IndexContractError(
            "sample selection episode_id does not match episode record"
        )
    _validate_selection_bounds(selection, episode)


def validate_selection_pool_binding(
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
    pool_metadata: Mapping[str, Any],
) -> None:
    record = _mapping(episode, "episode record")
    pool = _mapping(pool_metadata, "sample pool header")
    record_version = record.get("schema_version")
    pool_version = pool.get("schema_version", pool.get("format"))
    if (
        record_version == MOTUS_EPISODE_RECORD_V1
        and pool_version == MOTUS_SAMPLE_POOL_HEADER_V1
    ):
        validate_selection_pool_binding_v1(selection, episode, pool_metadata)
        return
    if (
        record_version is None
        and "robot" in record
        and "texts" in record
        and pool_version == MOTUS_SAMPLE_POOL_HEADER_V2
    ):
        validate_selection_pool_binding_v2(selection, episode, pool_metadata)
        return
    raise IndexContractError(
        "selection Episode and SamplePool protocol versions are incompatible: "
        f"episode={record_version!r} pool={pool_version!r}"
    )
