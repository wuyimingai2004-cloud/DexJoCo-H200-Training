"""Optional history-specific learning rates for a new fine-tuning run."""
from __future__ import annotations

import math


def split_history_groups(groups, encoder, *, adapter_lr=0., bias_lr=0.):
    for value in (adapter_lr, bias_lr):
        if not math.isfinite(value) or value < 0:
            raise ValueError("history learning rates must be finite and nonnegative")
    if not adapter_lr and not bias_lr:
        return groups  # Preserve legacy group ordering/optimizer resume exactly.
    if encoder is None:
        raise ValueError("history learning rates require a history encoder")
    bias = encoder.logit_bias
    adapters = [p for p in encoder.parameters() if p is not bias and p.requires_grad]
    selected = ({id(p) for p in adapters} if adapter_lr else set())
    if bias_lr and bias.requires_grad:
        selected.add(id(bias))
    original = {id(p) for group in groups for p in group["params"]}
    if not selected <= original:
        raise ValueError("history parameter missing from original optimizer groups")
    result = []
    for group in groups:
        kept = [p for p in group["params"] if id(p) not in selected]
        if kept:
            result.append({**group, "params": kept})
    if adapter_lr and adapters:
        result.append({"params": adapters, "lr": adapter_lr, "name": "history"})
    if bias_lr and bias.requires_grad:
        result.append({"params": [bias], "lr": bias_lr, "weight_decay": 0., "name": "history_bias"})
    return result
