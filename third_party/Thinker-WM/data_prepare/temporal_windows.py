"""Decision-point windows: which native rows feed one block-causal sample.

All indices are native control rows (``control_hz``). With ratio r and F
future frames (``video_stride`` layout, same as the existing pipeline):

  video frame n  -> row d + n*r            (n = 0..F, frame 0 = current)
  action step k  -> row d + k              (k = 0..F*r-1, drives frame k//r + 1)
  recent clip    -> rows d - j*r, j = Tr..1   (1+4m frames, one causal VAE clip)
  sparse slot j  -> about d - (recent span + j*stride) s, jittered, oldest first
  episode start  -> row 0 (kind EPISODE_START), only if older than every other slot

History rows are always < d: no future observation, state or depth enters a
history slot. Missing slots are marked invalid and point at the earliest real
row (replicate padding; never zero images). Times are real offsets in seconds.
If the video stream has its own frame rate, convert rows with
``rows_to_frames`` before reading frames.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

KIND_RECENT, KIND_SPARSE, KIND_EPISODE_START = 0, 1, 2


@dataclass(frozen=True)
class WindowSpec:
    video_action_freq_ratio: int = 4
    num_video_frames: int = 8
    control_hz: float = 25.0
    recent_frames: int = 5
    sparse_slots: int = 8
    sparse_stride_s: float = 1.0
    sparse_jitter: float = 0.25
    include_episode_start: bool = True
    require_full_future: bool = True

    def __post_init__(self):
        if self.num_video_frames % 4:
            raise ValueError("num_video_frames must be a multiple of 4 (VAE temporal stride)")
        if self.recent_frames and (self.recent_frames - 1) % 4:
            raise ValueError("recent_frames must be 0 or 1+4m")
        if not 0.0 <= self.sparse_jitter < 0.5:
            raise ValueError("sparse_jitter must be in [0, 0.5) of the stride")

    @property
    def frame_dt(self) -> float:
        return self.video_action_freq_ratio / self.control_hz

    @property
    def action_steps(self) -> int:
        return self.num_video_frames * self.video_action_freq_ratio

    @property
    def future_rows(self) -> int:
        return self.action_steps

    @property
    def history_slots(self) -> int:
        return self.sparse_slots + int(self.include_episode_start)


@dataclass
class TemporalWindow:
    decision: int
    video_rows: np.ndarray
    video_times: np.ndarray
    action_rows: np.ndarray
    action_valid: np.ndarray
    recent_rows: np.ndarray
    recent_times: np.ndarray
    recent_valid: np.ndarray
    sparse_rows: np.ndarray
    sparse_times: np.ndarray
    sparse_valid: np.ndarray
    sparse_kind: np.ndarray

    def to_dict(self) -> Dict[str, list]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}


def _rng(seed: int, episode_id: str, decision: int) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}:{episode_id}:{decision}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def decision_points(spec: WindowSpec, n_rows: int, stride_rows: int = 1,
                    min_decision: int = 0) -> List[int]:
    last = n_rows - 1 - (spec.future_rows if spec.require_full_future else 0)
    return list(range(max(0, int(min_decision)), max(last + 1, 0), int(stride_rows)))


def select_window(spec: WindowSpec, n_rows: int, decision: int, *, episode_id: str = "",
                  seed: int = 0, jitter: bool = True) -> TemporalWindow:
    d, r, hz = int(decision), spec.video_action_freq_ratio, spec.control_hz
    if not 0 <= d < n_rows:
        raise ValueError(f"decision {d} outside episode of {n_rows} rows")
    if spec.require_full_future and d + spec.future_rows > n_rows - 1:
        raise ValueError(f"decision {d}: future window needs rows up to {d + spec.future_rows}, "
                         f"episode has {n_rows}")
    n = np.arange(spec.num_video_frames + 1)
    video_rows = d + n * r
    k = np.arange(spec.action_steps)
    action_rows = d + k
    action_valid = action_rows <= n_rows - 1
    video_rows = np.minimum(video_rows, n_rows - 1)
    action_rows = np.minimum(action_rows, n_rows - 1)

    tr = spec.recent_frames
    j = np.arange(tr, 0, -1)
    recent_rows_raw = d - j * r
    recent_valid = recent_rows_raw >= 0
    recent_rows = np.maximum(recent_rows_raw, 0)
    recent_times = -(j * r) / hz

    rng = _rng(seed, episode_id, d)
    base = tr * spec.frame_dt
    js = np.arange(spec.sparse_slots, 0, -1)
    u = rng.uniform(-spec.sparse_jitter, spec.sparse_jitter, size=js.shape) if jitter else 0.0
    sparse_t = -(base + (js + u) * spec.sparse_stride_s)
    rows = d + np.round(sparse_t * hz).astype(np.int64)
    oldest_recent = d - tr * r if tr else d
    valid = (rows >= 0) & (rows < oldest_recent)
    for i in range(1, len(rows)):                      # strictly increasing among valid slots
        prev = rows[:i][valid[:i]]
        if valid[i] and prev.size and rows[i] <= prev.max():
            valid[i] = False
    rows = np.where(valid, rows, 0)
    times = np.where(valid, (rows - d) / hz, sparse_t)
    kinds = np.full(rows.shape, KIND_SPARSE)
    if spec.include_episode_start:
        older = rows[valid]
        first_valid = (d > 0) and (older.size == 0 or older.min() > 0) and oldest_recent > 0
        rows = np.concatenate([[0], rows])
        times = np.concatenate([[-d / hz if d > 0 else -1.0 / hz], times])
        valid = np.concatenate([[first_valid], valid])
        kinds = np.concatenate([[KIND_EPISODE_START], kinds])
    return TemporalWindow(
        decision=d, video_rows=video_rows, video_times=n * r / hz,
        action_rows=action_rows, action_valid=action_valid,
        recent_rows=recent_rows, recent_times=recent_times, recent_valid=recent_valid,
        sparse_rows=rows.astype(np.int64), sparse_times=times.astype(np.float64),
        sparse_valid=valid.astype(bool), sparse_kind=kinds.astype(np.int64))


def rows_to_frames(frame_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Map native rows to the last video frame recorded at or before each row."""
    fr = np.asarray(frame_rows, dtype=np.int64)
    idx = np.searchsorted(fr, np.asarray(rows, dtype=np.int64), side="right") - 1
    if (idx < 0).any():
        raise ValueError("row precedes the first video frame")
    return idx


def check_window(spec: WindowSpec, w: TemporalWindow) -> None:
    """Invariants used by tests and by the dataset adapter before reading data."""
    d = w.decision
    if (w.recent_rows[w.recent_valid] >= d).any() or (w.sparse_rows[w.sparse_valid] >= d).any():
        raise AssertionError("history slot at or after the decision row")
    if not np.all(np.diff(w.recent_times) > 0):
        raise AssertionError("recent times not increasing")
    vt = w.sparse_times[w.sparse_valid]
    if vt.size > 1 and not np.all(np.diff(vt) > 0):
        raise AssertionError("valid sparse slots not strictly increasing in time")
    if w.recent_valid.any() and vt.size and vt.max() >= w.recent_times[w.recent_valid].min():
        raise AssertionError("sparse slot overlaps the recent clip")
    if len(w.action_rows) != spec.action_steps or len(w.video_rows) != spec.num_video_frames + 1:
        raise AssertionError("future window length mismatch")


def batch_history_inputs(windows: List[TemporalWindow], rgb_of, state_of=None) -> Dict[str, object]:
    """Assemble the ``UBT_WM._prepare_history`` contract from windows.

    rgb_of(i, rows) -> float tensor [len(rows),3,H,W] in [0,1] for sample i;
    state_of(i, rows) -> tensor [len(rows), Ds] (optional).
    """
    import torch
    from .history_memory import pad_recent_pixels
    out: Dict[str, object] = {}
    if windows[0].sparse_rows.size:
        out["history_sparse_rgb"] = torch.stack([rgb_of(i, w.sparse_rows) for i, w in enumerate(windows)])
        out["history_sparse_times"] = torch.tensor(np.stack([w.sparse_times for w in windows]), dtype=torch.float32)
        out["history_sparse_valid"] = torch.tensor(np.stack([w.sparse_valid for w in windows]))
        out["history_sparse_kind"] = torch.tensor(np.stack([w.sparse_kind for w in windows]))
        if state_of is not None:
            out["history_sparse_state"] = torch.stack([state_of(i, w.sparse_rows) for i, w in enumerate(windows)])
    if windows[0].recent_rows.size:
        out["history_recent_rgb"] = torch.stack([pad_recent_pixels(rgb_of(i, w.recent_rows), w.recent_valid) for i, w in enumerate(windows)])
        out["history_recent_times"] = torch.tensor(np.stack([w.recent_times for w in windows]), dtype=torch.float32)
        out["history_recent_valid"] = torch.tensor(np.stack([w.recent_valid for w in windows]))
        if state_of is not None:
            out["history_recent_state"] = torch.stack([state_of(i, w.recent_rows) for i, w in enumerate(windows)])
    return out
