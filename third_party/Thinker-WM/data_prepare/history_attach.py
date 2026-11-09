"""Attach episode history (RGB + state) to a collated batch, like DepthAttacher.

For each sample the decision frame is ``frame_indices[b][0]`` (the clip's
first frame). History slots are chosen with
``temporal_windows.select_window`` in that episode's native frame index space:
the recent clip uses the sampled clip's frame spacing (so it is one uniformly
spaced 1+4m VAE clip, like the current window), sparse slots are spaced in
seconds via the source fps, plus the episode-start slot. Training jitter is
seeded by (seed, global step, episode, decision): deterministic and identical
on every rank for the same sample.

Unregistered sources fail by default (strict_sources=True in the config
builder). Explicit permissive callers receive all-invalid slots. A declared
source without a reader always fails. Output keys follow ``UBT_WM._prepare_history``.

Config::

    history:
      enable: true
      sources: [robodojo]
      recent_frames: 5
      sparse_slots: 8
      sparse_stride_s: 1.0
      sparse_jitter: 0.25
      include_episode_start: true
      seed: <int>
      robodojo_field_map: <path>       # same field map as the dataset reader
      robodojo_catalog: <path>         # catalog jsonl (episode_id -> file)
      robodojo_resize_mode: pad        # must equal the dataset reader's resize_mode
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .robodojo_core import FieldMap, RoboDojoHDF5, pad_or_resize
from .temporal_windows import WindowSpec, select_window
from .history_memory import (EventSpec, EventDetector, select_event_history, pad_recent_pixels,
                             history_metrics, history_window_spec)

HISTORY_KEYS = ("history_sparse_rgb", "history_sparse_times", "history_sparse_valid", "history_sparse_kind",
                "history_sparse_state", "history_recent_rgb", "history_recent_times", "history_recent_valid",
                "history_recent_state")


class RoboDojoHistoryReader:
    """RGB/state rows of registered RoboDojo episodes through the field map."""

    def __init__(self, field_map: FieldMap, resize_mode: str = "pad"):
        self.fm, self._io, self.resize_mode = field_map, RoboDojoHDF5(field_map), resize_mode
        self._paths: Dict[str, str] = {}
        self._rows: Dict[str, Optional[np.ndarray]] = {}

    def register_episode(self, episode_id: str, path: str) -> None:
        self._paths[str(episode_id)] = str(path)

    def _frame_rows(self, episode_id: str) -> Optional[np.ndarray]:
        if episode_id not in self._rows:
            self._rows[episode_id] = self._io.frame_rows(self._paths[episode_id])
        return self._rows[episode_id]

    def fps(self, episode_id: str) -> float:
        """Video frame rate in the frame-index space used by ``frame_indices``."""
        fr = self._frame_rows(str(episode_id)) if str(episode_id) in self._paths else None
        if fr is None or len(fr) < 2:
            return float(self.fm.control_hz)
        return float(self.fm.control_hz) * (len(fr) - 1) / float(fr[-1] - fr[0])

    def rgb(self, episode_id: str, frames: Sequence[int], target_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        p = self._paths.get(str(episode_id))
        if p is None:
            return None
        chw = torch.from_numpy(np.ascontiguousarray(self._io.read_rgb(p, frames))).permute(0, 3, 1, 2)
        if self.resize_mode == "pad":
            return pad_or_resize(chw, target_hw, "pad", False).float() / 255.0
        return pad_or_resize(chw.float() / 255.0, target_hw, "interpolate", False).clamp(0, 1)

    def state(self, episode_id: str, frames: Sequence[int]) -> Optional[torch.Tensor]:
        p = self._paths.get(str(episode_id))
        if p is None:
            return None
        fr = self._frame_rows(str(episode_id))
        rows = list(frames) if fr is None else [int(fr[i]) for i in frames]
        return torch.from_numpy(self._io.read_rows(p, "state", rows))


class RobotwinHistoryReader:
    """Reuse the production three-camera RGB/state reader, with an LRU state cache."""
    def __init__(self, catalog, resize_mode="pad"):
        from .robotwin_3d import Robotwin3dEpisodeReader
        from functools import lru_cache
        self.reader = Robotwin3dEpisodeReader(resize_mode=resize_mode)
        self.records = {}
        with open(catalog, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for name in ("video", "state", "action"):
                    m = rec[name]
                    m.setdefault("metadata", m.get("meta", {}))
                    if "locator" not in m:
                        m["locator"] = {"uri": m["storage"]["path"]}
                self.records[str(rec["episode_id"])] = rec
        self._states = lru_cache(maxsize=8)(self._read_states)

    def _read_states(self, ep):
        rec = self.records[ep]
        return self.reader.read_state(rec, list(range(rec["state"]["count"])))

    def fps(self, ep):
        return float(self.records[ep]["video"]["metadata"]["fps"])

    def rgb(self, ep, frames, target_hw):
        if not len(frames):
            return torch.zeros(0, 3, *target_hw)
        return self.reader.read_video(self.records[ep], list(frames),
                                      output_format="rgb_float32_chw", video_size=target_hw)

    def state(self, ep, frames):
        return self._states(ep)[list(frames)].clone()


def _valid_rows(w) -> set:
    return ({int(r) for r, v in zip(w.sparse_rows, w.sparse_valid) if v} |
            {int(r) for r, v in zip(w.recent_rows, w.recent_valid) if v})


def _slots(px, at, slot_rows, valid):
    """Full slot layout from decoded rows; invalid slots are zeros."""
    out = px.new_zeros((len(slot_rows), *px.shape[1:]))
    for j, (r, v) in enumerate(zip(slot_rows, valid)):
        if v:
            out[j] = px[at[int(r)]]
    return out


class HistoryAttacher:
    def __init__(self, spec: WindowSpec, readers: Mapping[str, Any], *, seed: int,
                 sources: Sequence[str] = (), with_state: bool = True,
                 events: Optional[EventSpec] = None, strict_sources: bool = False):
        missing = [s for s in sources if s not in readers]
        if missing:
            raise ValueError(f"history.sources without a reader: {missing}")
        self.spec, self.readers, self.seed, self.with_state = spec, dict(readers), int(seed), with_state
        self.events, self.strict_sources = events or EventSpec(), strict_sources
        self.last_metrics = {}

    def _window(self, ep: str, frames: Sequence[int], fps: float, step: int):
        f = [int(x) for x in frames]
        stride = f[1] - f[0] if len(f) > 1 else 1
        spec = replace(self.spec, video_action_freq_ratio=max(int(stride), 1), control_hz=float(fps),
                       require_full_future=False)
        return select_window(spec, f[0] + 1, f[0], episode_id=ep, seed=self.seed * 1_000_003 + int(step),
                             jitter=True)

    def _read_rgb(self, plans, hw):
        """{i: (reader, ep, window)} -> {i: (sparse px, recent px) or None if unreadable}.

        One read per episode over the union of its samples' valid rows; each
        sample gets back its full slot layout, invalid slots zero, and callers
        apply the sparse mask and ``pad_recent_pixels`` as before. An episode
        whose samples have no valid slot still reads row 0, so a missing or
        broken episode fails instead of passing as an empty history.
        """
        groups: Dict[Tuple[int, str], list] = {}
        for i, (reader, ep, w) in plans.items():
            g = groups.setdefault((id(reader), ep), [reader, set(), []])
            g[1] |= _valid_rows(w)
            g[2].append(i)
        res = {}
        for (_, ep), (reader, rows, members) in groups.items():
            rows = sorted(rows) or [0]
            px = reader.rgb(ep, rows, hw)
            at = {r: k for k, r in enumerate(rows)}
            for i in members:
                w = plans[i][2]
                res[i] = None if px is None else (_slots(px, at, w.sparse_rows, w.sparse_valid),
                                                  _slots(px, at, w.recent_rows, w.recent_valid))
        return res

    def attach(self, batch: Mapping[str, Any], step: int) -> Dict[str, Any]:
        if all(k in batch for k in ("history_recent_rgb", "history_sparse_rgb")):
            self.last_metrics = history_metrics(batch, int(batch["video_frames"].shape[0]))
            return dict(batch)
        video = batch["video_frames"]
        b, hw = int(video.shape[0]), (int(video.shape[-2]), int(video.shape[-1]))
        eps, idx = batch.get("episode_ids"), batch.get("frame_indices")
        srcs = batch.get("source_datasets") or [None] * b
        s_slots, r_slots = self.spec.history_slots, self.spec.recent_frames
        out = {
            "history_sparse_rgb": torch.zeros(b, s_slots, 3, *hw), "history_sparse_times": torch.full((b, s_slots), -1.0),
            "history_sparse_valid": torch.zeros(b, s_slots, dtype=torch.bool),
            "history_sparse_kind": torch.ones(b, s_slots, dtype=torch.long),
            "history_recent_rgb": torch.zeros(b, r_slots, 3, *hw),
            "history_recent_times": torch.full((b, r_slots), -1.0),
            "history_recent_valid": torch.zeros(b, r_slots, dtype=torch.bool),
        }
        states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        plans = {}
        for i in range(b):
            reader = self.readers.get(None if srcs[i] is None else str(srcs[i]))
            if reader is None or eps is None or idx is None or idx[i] is None:
                if self.strict_sources:
                    raise ValueError(f"missing history reader/metadata for source {srcs[i]!r}")
                continue
            ep = str(eps[i])
            f = [int(x) for x in idx[i]]
            if len(f) > 1:
                data_dt = (f[1] - f[0]) / reader.fps(ep)
                cfg_dt = self.spec.video_action_freq_ratio / self.spec.control_hz
                if abs(data_dt - cfg_dt) > 1e-6 * cfg_dt:
                    raise ValueError(f"{ep}: video frame spacing {data_dt:.6f}s != configured "
                                     f"ratio/control_hz {cfg_dt:.6f}s")
            w = self._window(ep, idx[i], reader.fps(ep), step)
            if self.events.enabled:
                # Runtime scans every control row; scanning a subsampled video
                # clock would shift before/after frames and hysteresis timing.
                if abs(reader.fps(ep) - self.spec.control_hz) > 1e-6 * self.spec.control_hz:
                    raise ValueError(f"{ep}: history events need video rows at control_hz "
                                     f"({reader.fps(ep)} != {self.spec.control_hz})")
                detector = EventDetector(self.events, self.spec.control_hz)
                # Scan only observations available strictly before this decision.
                rows = list(range(w.decision))
                past = reader.state(ep, rows) if rows else []
                for row, state in zip(rows, past):
                    detector.update(row, np.asarray(state))
                w = select_event_history(w, self.spec, self.events, detector.records)
            plans[i] = (reader, ep, w)
        pixels = self._read_rgb(plans, hw) if plans else {}
        for i, (reader, ep, w) in plans.items():
            if pixels[i] is None:
                if self.strict_sources:
                    raise ValueError(f"missing history episode {ep}")
                continue
            px_s, px_r = pixels[i]
            out["history_sparse_rgb"][i] = px_s * torch.from_numpy(w.sparse_valid)[:, None, None, None]
            out["history_sparse_times"][i] = torch.from_numpy(w.sparse_times).float()
            out["history_sparse_valid"][i] = torch.from_numpy(w.sparse_valid)
            out["history_sparse_kind"][i] = torch.from_numpy(w.sparse_kind)
            out["history_recent_rgb"][i] = pad_recent_pixels(px_r, w.recent_valid)
            out["history_recent_times"][i] = torch.from_numpy(w.recent_times).float()
            out["history_recent_valid"][i] = torch.from_numpy(w.recent_valid)
            if self.with_state:
                ss, sr = reader.state(ep, w.sparse_rows.tolist()), reader.state(ep, w.recent_rows.tolist())
                if ss is not None and sr is not None:
                    states[i] = (ss * torch.from_numpy(w.sparse_valid)[:, None],
                                 sr * torch.from_numpy(w.recent_valid)[:, None])
        if self.with_state and states:
            dim = next(iter(states.values()))[0].shape[-1]
            out["history_sparse_state"] = torch.zeros(b, s_slots, dim)
            out["history_recent_state"] = torch.zeros(b, r_slots, dim)
            for i, (ss, sr) in states.items():
                out["history_sparse_state"][i], out["history_recent_state"][i] = ss, sr
        self.last_metrics = history_metrics(out, b)
        merged = dict(batch)
        merged.update(out)
        return merged


def history_inputs_from_batch(batch: Mapping[str, Any], device=None) -> Optional[Dict[str, torch.Tensor]]:
    got = {k: batch[k] for k in HISTORY_KEYS if k in batch}
    if not got:
        return None
    return {k: (v.to(device) if device is not None and torch.is_tensor(v) else v) for k, v in got.items()}


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_history_attacher(config: Any) -> Optional[HistoryAttacher]:
    wm = _get(_get(config, "model"), "wm")
    if not bool(_get(wm, "history_enable", False)):
        return None
    h = _get(config, "history")
    if h is None or not bool(_get(h, "enable", False)):
        raise ValueError("model.wm.history_enable=true requires a history: {enable: true, ...} section")
    spec = history_window_spec(config)
    readers: Dict[str, Any] = {}
    fm_path = _get(h, "robodojo_field_map")
    if fm_path:
        cat = _get(h, "robodojo_catalog")
        if not cat:
            raise ValueError("history.robodojo_field_map requires history.robodojo_catalog")
        rd = RoboDojoHistoryReader(FieldMap.load(str(fm_path)), str(_get(h, "robodojo_resize_mode", "pad")))
        with open(str(cat), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    rd.register_episode(rec["episode_id"], rec["video"]["storage"]["path"])
        readers["robodojo"] = rd
    twin_catalog = _get(h, "robotwin_catalog")
    if twin_catalog:
        readers["robotwin_3d"] = RobotwinHistoryReader(str(twin_catalog),
                                                     str(_get(h, "robotwin_resize_mode", "pad")))
    seed = _get(h, "seed")
    if seed is None:
        raise ValueError("history.seed must be set explicitly")
    return HistoryAttacher(spec, readers, seed=int(seed), sources=list(_get(h, "sources", []) or []),
                           events=EventSpec.from_config(_get(h, "events")),
                           strict_sources=bool(_get(h, "strict_sources", True)))
