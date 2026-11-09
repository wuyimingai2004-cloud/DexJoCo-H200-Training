"""Explicit action-interface migration, separate from same-robot resume."""
from __future__ import annotations

import json
from pathlib import Path


INTERFACE_PREFIXES = (
    "world_model.action_branch.encoder.action_embedding.",
    "world_model.action_branch.encoder.state_embedding.",
    "world_model.action_branch.decoder.action_unembedding.",
)
OPTIONAL_PREFIXES = (
    "world_model.history_encoder.",
    "world_model.video_branch.depth_patch_embedding.",
    "world_model.video_branch.depth_head.",
    "world_model.video_branch.depth_stream_embed",
    "world_model.uni_branch.depth_uni_blocks.",
)


def is_interface(key: str) -> bool:
    return key == "ego4d_state_prompt" or key.startswith(INTERFACE_PREFIXES) or "robot_token_embedding." in key


def select_transfer_weights(source: dict, target: dict, legacy_prefixes=()) -> tuple[dict, dict]:
    """Reinitialize the action boundary even when shapes accidentally match.

    All remaining target tensors must be present with exactly matching shapes.
    Unknown checkpoint keys are rejected instead of accepting partial backbones.
    """
    filtered, dropped, errors = {}, {}, []
    for key, tensor in source.items():
        if is_interface(key):
            dropped[key] = "new EEF/hand semantics: reinitialize"
        elif key.startswith(tuple(legacy_prefixes)):
            dropped[key] = "removed legacy module"
        elif key not in target and key.startswith(OPTIONAL_PREFIXES):
            dropped[key] = "RGB first round: disabled depth/history"
        elif key not in target:
            errors.append(f"unexpected key {key}")
        elif tuple(tensor.shape) != tuple(target[key].shape):
            errors.append(f"shape mismatch {key}: {tuple(tensor.shape)} vs {tuple(target[key].shape)}")
        else:
            filtered[key] = tensor
    missing = sorted(k for k in target if k not in filtered and not is_interface(k))
    if missing or errors:
        raise ValueError(f"incompatible shared backbone; missing={missing}; errors={errors}")
    if not filtered:
        raise ValueError("checkpoint contains no shared backbone weights")
    audit = dict(policy="dexjoco_rgb", loaded=sorted(filtered), dropped=dropped,
                 reinitialized=sorted(k for k in target if is_interface(k)))
    return filtered, audit


def load_dexjoco_transfer(model, state_dict: dict, source_path: str, audit_path: str | None) -> None:
    if model.config.enable_depth or model.config.history_enable:
        raise ValueError("dexjoco_rgb transfer requires depth/history disabled")
    filtered, audit = select_transfer_weights(state_dict, model.state_dict(), model._LEGACY_STATE_DICT_PREFIXES)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected or any(not is_interface(k) for k in missing):
        raise RuntimeError(f"transfer load mismatch: missing={missing}, unexpected={unexpected}")
    if audit_path is not None:
        from thinker_wm.data_prepare.dexjoco.contracts import sha256
        audit.update(source=str(Path(source_path).resolve()), source_sha256=sha256(source_path))
        Path(audit_path).write_text(json.dumps(audit, indent=2) + "\n")
