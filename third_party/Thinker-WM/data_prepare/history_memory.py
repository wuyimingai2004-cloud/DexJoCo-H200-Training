"""Shared causal event selection, recent padding and history diagnostics.

Events describe gripper transitions, never successful grasps. Only already
arrived observations may finish a before/after pair. No visual model is run.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import numpy as np

KIND_EVENT_BEFORE, KIND_EVENT_AFTER = 3, 4


@dataclass(frozen=True)
class EventSpec:
    enabled: bool = False
    gripper_indices: tuple = (6, 13)
    closed_threshold: float = 0.3
    open_threshold: float = 0.7
    after_s: float = 0.08
    cooldown_s: float = 0.16
    max_frames: int = 8
    capacity: int = 32
    # >0: a reading <= closed_threshold counts as closed only once the gripper
    # has stopped, |g(t) - g(t - closed_settle_s)| <= closed_settle_delta, so
    # grippers that stall at the object width (any value) still close. The
    # before frame is then the last row still at the previous level.
    closed_settle_s: float = 0.0
    closed_settle_delta: float = 0.01

    def __post_init__(self):
        if not self.closed_threshold < self.open_threshold:
            raise ValueError("gripper hysteresis requires closed_threshold < open_threshold")
        if self.closed_settle_s < 0 or self.closed_settle_delta <= 0:
            raise ValueError("invalid closed-settle window")
        if self.after_s <= 0 or self.cooldown_s <= 0 or self.max_frames < 0 or self.capacity < 1:
            raise ValueError("invalid event time/budget")
        if self.max_frames % 2 or self.max_frames > self.capacity * 2:
            raise ValueError("event budget must contain complete before/after pairs")
        if self.enabled and not self.gripper_indices:
            raise ValueError("events require explicit gripper state dimensions")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            return cls()
        values = {k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg}
        if "gripper_indices" in values:
            values["gripper_indices"] = tuple(values["gripper_indices"])
        return cls(**values)


class EventDetector:
    """Bounded streaming detector. State thresholds are in native state units."""
    def __init__(self, spec: EventSpec, hz: float):
        self.spec, self.hz = spec, float(hz)
        self.reset()

    def reset(self):
        self.records = []  # (before_row, after_row), sorted by completion
        self.pending = None  # (before_row, trigger_row)
        self.levels = None
        self.level_rows = None  # last row whose reading matched the held level
        self._recent = deque()  # (time, readings) spanning closed_settle_s
        self.last_row = None
        self.last_trigger = np.full(len(self.spec.gripper_indices), -math.inf)
        self.last_time = None
        self.pending_time = None

    def update(self, row: int, state, time_s=None):
        if not self.spec.enabled:
            return
        if self.last_row is not None and row <= self.last_row:
            raise ValueError("event rows must strictly increase")
        now = row / self.hz if time_s is None else float(time_s)
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("event timestamps must be finite and strictly increase")
        v = np.asarray(state)[list(self.spec.gripper_indices)]
        if not np.isfinite(v).all():
            raise ValueError("non-finite gripper state")
        levels = np.where(v <= self.spec.closed_threshold, 0,
                          np.where(v >= self.spec.open_threshold, 1, -1))
        if self.spec.closed_settle_s > 0:
            levels = np.where((levels == 0) & ~self._settled(now, v), -1, levels)
        if self.pending and now + 1e-9 >= self.pending_time + self.spec.after_s:
            self.records.append((self.pending[0], int(row)))
            self.records = self.records[-self.spec.capacity:]
            self.pending = None
        if self.levels is not None:
            changed = (levels >= 0) & (self.levels >= 0) & (levels != self.levels)
            changed &= now - self.last_trigger + 1e-9 >= self.spec.cooldown_s
            if changed.any():
                # Cooldown is per gripper; a change while a pair is pending is
                # covered by that pair's later after-frame.
                self.last_trigger[changed] = now
                if self.pending is None:
                    before = self.last_row
                    if self.spec.closed_settle_s > 0:
                        before = int(self.level_rows[changed].min())
                    self.pending = (int(before), int(row))
                    self.pending_time = now
            self.levels = np.where(levels >= 0, levels, self.levels)
        else:
            self.levels = levels
            self.level_rows = np.full(len(levels), int(row))
        self.level_rows = np.where((levels >= 0) & (levels == self.levels), int(row), self.level_rows)
        if self.spec.closed_settle_s > 0:
            self._recent.append((now, v))
            while len(self._recent) > 1 and self._recent[1][0] <= now - self.spec.closed_settle_s + 1e-9:
                self._recent.popleft()
        self.last_row = int(row)
        self.last_time = now

    def _settled(self, now, v):
        ref = None
        for t, past in self._recent:
            if t > now - self.spec.closed_settle_s + 1e-9:
                break
            ref = past
        if ref is None:
            return np.zeros(len(v), dtype=bool)
        return np.abs(v - ref) <= self.spec.closed_settle_delta

    def retained_rows(self):
        rows = {r for pair in self.records for r in pair}
        if self.pending:
            rows.add(self.pending[0])
        return rows


def select_event_history(window, spec, events: EventSpec, records):
    """Use older complete event pairs first, then uniform sparse fallback.

    The episode-start slot is additional to the sparse/event budget. All
    selected frames precede the entire recent window, so the two parts do not
    duplicate one another. Selection depends only on records already observed.
    """
    if not events.enabled:
        return window
    cutoff = window.decision - spec.recent_frames * spec.video_action_freq_ratio
    chosen = {}
    budget = min(events.max_frames, spec.sparse_slots)
    budget -= budget % 2
    for before, after in reversed(records):
        if not (0 <= before < after < cutoff) or before in chosen or after in chosen:
            continue
        if len(chosen) + 2 > budget:
            break
        chosen[before], chosen[after] = KIND_EVENT_BEFORE, KIND_EVENT_AFTER
    # Ordinary sparse fallback consumes only unused slots, never replaces events.
    for row, valid, kind in reversed(list(zip(window.sparse_rows, window.sparse_valid, window.sparse_kind))):
        if len(chosen) >= spec.sparse_slots:
            break
        if valid and kind == 1 and int(row) not in chosen:
            chosen[int(row)] = 1
    if spec.include_episode_start and cutoff > 0 and 0 not in chosen:
        chosen[0] = 2
    n = spec.history_slots
    selected = sorted(chosen.items())
    rows = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    kinds = np.ones(n, dtype=np.int64)
    times = np.full(n, -1.0, dtype=np.float64)
    for i, (row, kind) in enumerate(selected):
        rows[i], valid[i], kinds[i] = row, True, kind
        times[i] = (row - window.decision) / spec.control_hz
    return replace(window, sparse_rows=rows, sparse_valid=valid, sparse_kind=kinds, sparse_times=times)


def pad_recent_pixels(pixels, valid):
    """Replicate leading padding; causal hold for internal holes; keep mask.

    pixels [T,C,H,W], valid [T]. All-invalid clips remain zero and fully masked.
    This is the single implementation used by training and measured runtime.
    """
    import torch
    v = torch.as_tensor(valid, device=pixels.device, dtype=torch.bool)
    out = pixels.clone()
    real = v.nonzero().flatten()
    if not real.numel():
        return out.zero_()
    first = int(real[0])
    out[:first] = out[first]
    for i in range(first + 1, len(v)):
        if not bool(v[i]):
            out[i] = out[i - 1]
    return out


def history_metrics(inputs, batch_size=1):
    """Rank-local batch means; valid pixels are not necessarily usable VAE latents."""
    import torch
    h = inputs or {}
    recent = h.get("history_recent_valid", torch.zeros(batch_size, 0, dtype=torch.bool))
    sparse = h.get("history_sparse_valid", torch.zeros(batch_size, 0, dtype=torch.bool))
    kinds = h.get("history_sparse_kind", torch.ones_like(sparse, dtype=torch.long))
    rc, sc = recent.sum(-1).float().cpu(), sparse.sum(-1).float().cpu()
    ec = (sparse & ((kinds == 3) | (kinds == 4))).sum(-1).float().cpu()
    usable_recent = torch.zeros_like(rc)
    if recent.shape[-1]:
        usable_recent += recent[:, 0].float().cpu()
        for end in range(4, recent.shape[-1], 4):
            usable_recent += recent[:, end-3:end+1].all(-1).float().cpu()
    return dict(history_recent_valid_frames=float(rc.mean()), history_event_valid_frames=float(ec.mean()),
                history_sparse_valid_frames=float(sc.mean()),
                history_empty_fraction=float(((rc + sc) == 0).float().mean()),
                history_usable_empty_fraction=float(((usable_recent + sc) == 0).float().mean()),
                history_frames_without_latents_fraction=float((((rc + sc) > 0) &
                    ((usable_recent + sc) == 0)).float().mean()))


def history_window_spec(config):
    from .temporal_windows import WindowSpec
    def get(c, key, default=None):
        return c.get(key, default) if hasattr(c, "get") else getattr(c, key, default)
    h, common = get(config, "history", {}), get(config, "common", {})
    if get(common, "control_hz") is None:
        raise ValueError("history needs an explicit common.control_hz")
    return WindowSpec(video_action_freq_ratio=int(get(common, "video_action_freq_ratio", 4)),
                      num_video_frames=int(get(common, "num_video_frames", 16)),
                      control_hz=float(get(common, "control_hz")),
                      recent_frames=int(get(h, "recent_frames", 5)),
                      sparse_slots=int(get(h, "sparse_slots", 8)),
                      sparse_stride_s=float(get(h, "sparse_stride_s", 1.0)),
                      sparse_jitter=float(get(h, "sparse_jitter", .25)),
                      include_episode_start=bool(get(h, "include_episode_start", True)),
                      require_full_future=False)
