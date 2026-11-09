"""RoboDojo HDF5 access without motus_data (field map, views, catalog records).

Field names are never guessed. Every HDF5 path comes from an explicit field
map (JSON/YAML) written after ``scripts/audit_robodojo_hdf5.py`` has been run
on real episodes; a missing required key is an error, not a default.

Field map schema (version 1)::

    schema_version: 1
    control_hz: <float>                 # native rate of state/action rows
    rgb:   {views: {<view>: <h5 path>}, encoding: uint8_thwc | uint8_tchw | encoded_bytes}
    depth: {views: {<view>: <h5 path>}, unit_to_mm: <float>, invalid_values: [..]}   # optional
    state:  {path: <h5 path> | paths: [<h5 path>, ...], columns: [start, end] | null,
             feature_names: [...] | null}    # paths: [T, d_i] arrays joined on the last axis
    action: {path: <h5 path> | paths: [...], columns: [start, end] | null, semantics: <str>,
             feature_names: [...] | null}
    frame_rows: {path: <h5 path>} | null   # video frame -> state/action row (if rates differ)
    instruction: {attr: <h5 attr>} | {path: <h5 path>} | null
    scene: {attr: <h5 attr>} | {parent_dir: true} | {path_tail: <n>}   # last n path parts
    layout: {rows: [[<view>, ...], ...], row_hw: [[h, w], ...], canvas_hw: [H, W] | null}

``layout`` is applied identically to RGB (bilinear) and depth (nearest), so
the two streams stay pixel-aligned after composition. With ``canvas_hw`` the
tiles are centred on an H x W zero canvas, so the composed frame can already
match another source's padded placement (e.g. the RoboTwin Motus T).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SOURCE_DATASET = "robodojo"
FIELD_MAP_VERSION = 1
_ENCODINGS = ("uint8_thwc", "uint8_tchw", "encoded_bytes")


class FieldMapError(ValueError):
    pass


def _req(d: Mapping[str, Any], key: str, where: str):
    if key not in d or d[key] is None:
        raise FieldMapError(f"field map {where}: missing required key {key!r}")
    return d[key]


@dataclass(frozen=True)
class Layout:
    rows: Tuple[Tuple[str, ...], ...]
    row_hw: Tuple[Tuple[int, int], ...]
    canvas_hw: Optional[Tuple[int, int]] = None

    @property
    def content_size(self) -> Tuple[int, int]:
        h = sum(r[0] for r in self.row_hw)
        w = max(r[1] * len(v) for r, v in zip(self.row_hw, self.rows))
        return h, w

    @property
    def size(self) -> Tuple[int, int]:
        return self.canvas_hw or self.content_size

    @property
    def views(self) -> Tuple[str, ...]:
        return tuple(v for row in self.rows for v in row)


@dataclass(frozen=True)
class FieldMap:
    control_hz: float
    rgb_views: Dict[str, str]
    rgb_encoding: str
    state_path: str
    action_path: str
    action_semantics: str
    layout: Layout
    state_columns: Optional[Tuple[int, int]] = None
    action_columns: Optional[Tuple[int, int]] = None
    state_names: Optional[Tuple[str, ...]] = None
    action_names: Optional[Tuple[str, ...]] = None
    state_paths: Tuple[str, ...] = ()
    action_paths: Tuple[str, ...] = ()
    depth_views: Optional[Dict[str, str]] = None
    depth_unit_to_mm: Optional[float] = None
    depth_invalid_values: Tuple[float, ...] = ()
    frame_rows_path: Optional[str] = None
    instruction: Optional[Dict[str, str]] = None
    scene: Dict[str, Any] = field(default_factory=lambda: {"parent_dir": True})
    digest: str = ""

    @staticmethod
    def load(path: str | Path) -> "FieldMap":
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yaml", ".yml")):
            import yaml
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return FieldMap.from_dict(raw, digest=hashlib.sha256(text.encode()).hexdigest()[:16])

    @staticmethod
    def from_dict(raw: Mapping[str, Any], digest: str = "") -> "FieldMap":
        if int(_req(raw, "schema_version", "root")) != FIELD_MAP_VERSION:
            raise FieldMapError(f"unsupported field map schema_version {raw.get('schema_version')}")
        rgb = _req(raw, "rgb", "root")
        enc = str(_req(rgb, "encoding", "rgb"))
        if enc not in _ENCODINGS:
            raise FieldMapError(f"rgb.encoding must be one of {_ENCODINGS}")
        views = dict(_req(rgb, "views", "rgb"))
        st, ac = _req(raw, "state", "root"), _req(raw, "action", "root")
        lay = _req(raw, "layout", "root")
        canvas = lay.get("canvas_hw")
        layout = Layout(tuple(tuple(r) for r in _req(lay, "rows", "layout")),
                        tuple((int(h), int(w)) for h, w in _req(lay, "row_hw", "layout")),
                        None if canvas is None else (int(canvas[0]), int(canvas[1])))
        if len(layout.rows) != len(layout.row_hw):
            raise FieldMapError("layout.rows and layout.row_hw differ in length")
        if layout.canvas_hw and any(c < n for c, n in zip(layout.canvas_hw, layout.content_size)):
            raise FieldMapError(f"layout.canvas_hw {layout.canvas_hw} is smaller than the tiles "
                                f"{layout.content_size}")
        unknown = [v for v in layout.views if v not in views]
        if unknown:
            raise FieldMapError(f"layout uses views without an rgb path: {unknown}")
        depth = raw.get("depth")
        dviews = None
        if depth:
            dviews = dict(_req(depth, "views", "depth"))
            missing = [v for v in layout.views if v not in dviews]
            if missing:
                raise FieldMapError(f"depth.views lacks layout views {missing}")
        cols = lambda c: None if c is None else (int(c[0]), int(c[1]))  # noqa: E731
        names = lambda n: None if n is None else tuple(str(x) for x in n)  # noqa: E731
        sp, ap = _paths(st, "state"), _paths(ac, "action")
        return FieldMap(
            control_hz=float(_req(raw, "control_hz", "root")),
            rgb_views=views, rgb_encoding=enc,
            state_path=",".join(sp), action_path=",".join(ap), state_paths=sp, action_paths=ap,
            action_semantics=str(_req(ac, "semantics", "action")),
            layout=layout,
            state_columns=cols(st.get("columns")), action_columns=cols(ac.get("columns")),
            state_names=names(st.get("feature_names")), action_names=names(ac.get("feature_names")),
            depth_views=dviews,
            depth_unit_to_mm=None if not depth else float(_req(depth, "unit_to_mm", "depth")),
            depth_invalid_values=tuple(float(v) for v in (depth or {}).get("invalid_values", [])),
            frame_rows_path=(raw.get("frame_rows") or {}).get("path"),
            instruction=raw.get("instruction"),
            scene=dict(raw.get("scene") or {"parent_dir": True}),
            digest=digest,
        )


def _paths(spec: Mapping[str, Any], where: str) -> Tuple[str, ...]:
    if ("path" in spec) == ("paths" in spec):
        raise FieldMapError(f"field map {where}: give exactly one of 'path' or 'paths'")
    paths = (spec["path"],) if "path" in spec else tuple(spec["paths"] or ())
    if not paths or not all(isinstance(p, str) and p for p in paths):
        raise FieldMapError(f"field map {where}: empty or non-string path in {paths!r}")
    return paths


def _h5():
    import h5py
    return h5py


def _rows_of(f, paths: Sequence[str], which: str) -> int:
    n = {int(f[p].shape[0]) for p in paths}
    if len(n) != 1:
        raise ValueError(f"{which} paths {list(paths)} have different row counts {sorted(n)}")
    return n.pop()


def _decode_bytes(buf) -> np.ndarray:
    import cv2
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("failed to decode encoded image bytes")
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# JPEG decode releases the GIL, so frames decode in parallel threads; HDF5
# reads stay serial (h5py holds a global lock). <=1 restores the serial loop.
DECODE_THREADS_ENV = "MOTUS_ROBODOJO_DECODE_THREADS"
_DECODE_POOL: Tuple[Optional[int], Any] = (None, None)


def _decode_pool():
    global _DECODE_POOL
    if int(os.environ.get(DECODE_THREADS_ENV, "8")) <= 1:
        return None
    pid, pool = _DECODE_POOL
    if pid != os.getpid():  # never reuse a pool inherited through fork
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=int(os.environ.get(DECODE_THREADS_ENV, "8")),
                                  thread_name_prefix="robodojo-decode")
        _DECODE_POOL = (os.getpid(), pool)
    return pool


def _read_frames(ds, indices: Sequence[int], encoding: str) -> np.ndarray:
    idx = [int(i) for i in indices]
    if encoding == "encoded_bytes":
        bufs = [bytes(ds[i]) for i in idx]
        pool = _decode_pool() if len(bufs) > 1 else None
        imgs = list(pool.map(_decode_bytes, bufs)) if pool is not None else [_decode_bytes(b) for b in bufs]
        return np.stack(imgs, axis=0)
    order = np.argsort(idx)
    sorted_idx = np.asarray(idx)[order]
    uniq, inv = np.unique(sorted_idx, return_inverse=True)
    data = ds[uniq.tolist()]
    data = data[inv][np.argsort(order)]
    if encoding == "uint8_tchw":
        data = np.transpose(data, (0, 2, 3, 1))
    return np.asarray(data)


def _resize(frames: np.ndarray, hw: Tuple[int, int], nearest: bool) -> np.ndarray:
    import cv2
    h, w = hw
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in frames], axis=0)


def compose(frames_by_view: Mapping[str, np.ndarray], layout: Layout, nearest: bool) -> np.ndarray:
    """Tile views row by row; rows are left-aligned and right-padded with zeros.

    With ``layout.canvas_hw`` the tiled block is centred on the canvas.
    """
    h_total, w_total = layout.size
    ch, cw = layout.content_size
    first = next(iter(frames_by_view.values()))
    t = first.shape[0]
    extra = first.shape[3:] if first.ndim == 4 else ()
    out = np.zeros((t, h_total, w_total, *extra), dtype=first.dtype)
    y = (h_total - ch) // 2
    for row, (rh, rw) in zip(layout.rows, layout.row_hw):
        x = (w_total - cw) // 2
        for v in row:
            tile = _resize(frames_by_view[v], (rh, rw), nearest)
            out[:, y:y + rh, x:x + rw] = tile
            x += rw
        y += rh
    return out


def pad_or_resize(chw, size, mode: str, nearest: bool):
    """Composed [T,C,h,w] -> model size; shared by the dataset reader and the runtime."""
    import torch.nn.functional as F
    h, w = int(size[0]), int(size[1])
    if tuple(chw.shape[-2:]) == (h, w):
        return chw
    if mode == "pad":
        ph, pw = h - chw.shape[-2], w - chw.shape[-1]
        if ph < 0 or pw < 0:
            raise ValueError(f"compose size {tuple(chw.shape[-2:])} exceeds {(h, w)}")
        return F.pad(chw, (0, pw, 0, ph))
    return F.interpolate(chw.float(), size=(h, w), mode="nearest" if nearest else "bilinear",
                         align_corners=None if nearest else False)


class RoboDojoHDF5:
    """Read one episode file through a FieldMap. Opened lazily per call."""

    def __init__(self, field_map: FieldMap):
        self.fm = field_map

    def _cols(self, x: np.ndarray, cols):
        return x if cols is None else x[:, cols[0]:cols[1]]

    def lengths(self, path: str) -> Dict[str, int]:
        with _h5().File(path, "r") as f:
            out = {"state": _rows_of(f, self.fm.state_paths, "state"),
                   "action": _rows_of(f, self.fm.action_paths, "action")}
            out["video"] = min(int(f[p].shape[0]) for p in self.fm.rgb_views.values())
            if self.fm.depth_views:
                out["depth"] = min(int(f[p].shape[0]) for p in self.fm.depth_views.values())
        return out

    def frame_rows(self, path: str) -> Optional[np.ndarray]:
        if not self.fm.frame_rows_path:
            return None
        with _h5().File(path, "r") as f:
            return np.asarray(f[self.fm.frame_rows_path][()], dtype=np.int64)

    def read_rgb(self, path: str, indices: Sequence[int]) -> np.ndarray:
        with _h5().File(path, "r") as f:
            views = {v: _read_frames(f[self.fm.rgb_views[v]], indices, self.fm.rgb_encoding)
                     for v in self.fm.layout.views}
        for v, x in views.items():
            if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
                raise ValueError(f"rgb view {v!r} decoded to {x.dtype}{x.shape}, expected uint8 THW3")
        return compose(views, self.fm.layout, nearest=False)

    def read_depth_mm(self, path: str, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Composed depth in millimetres (0 = invalid) and its validity mask."""
        if not self.fm.depth_views:
            raise ValueError("field map has no depth views")
        enc = "encoded_bytes" if self.fm.rgb_encoding == "encoded_bytes" else "uint8_thwc"
        with _h5().File(path, "r") as f:
            views = {}
            for v in self.fm.layout.views:
                ds = f[self.fm.depth_views[v]]
                raw = (_read_frames(ds, indices, enc) if ds.dtype.kind in ("O", "S", "V")
                       else _read_frames(ds, indices, "uint8_thwc"))
                if raw.ndim == 4:
                    if raw.shape[-1] != 1:
                        raise ValueError(f"depth view {v!r} has {raw.shape[-1]} channels")
                    raw = raw[..., 0]
                d = raw.astype(np.float32)
                bad = ~np.isfinite(d)
                for iv in self.fm.depth_invalid_values:
                    bad |= d == iv
                d = np.where(bad, 0.0, d * float(self.fm.depth_unit_to_mm))
                views[v] = d
        mm = compose(views, self.fm.layout, nearest=True)
        return mm, mm > 0

    def read_rows(self, path: str, which: str, indices: Sequence[int]) -> np.ndarray:
        paths, cols = ((self.fm.state_paths, self.fm.state_columns) if which == "state"
                       else (self.fm.action_paths, self.fm.action_columns))
        with _h5().File(path, "r") as f:
            parts = [np.asarray(f[p][()], dtype=np.float32) for p in paths]
        for p, x in zip(paths, parts):
            if x.ndim != 2:
                raise ValueError(f"{which} {p!r} has shape {x.shape}, expected [T, D]")
        x = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        x = self._cols(x, cols)
        return x[[int(i) for i in indices]]

    def instruction(self, path: str) -> Optional[str]:
        spec = self.fm.instruction
        if not spec:
            return None
        with _h5().File(path, "r") as f:
            if "attr" in spec:
                v = f.attrs.get(spec["attr"])
            else:
                v = f[spec["path"]][()]
        if v is None:
            return None
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        if isinstance(v, np.ndarray):
            v = v.reshape(-1)[0]
            v = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        return str(v).strip() or None

    def scene_id(self, path: str) -> str:
        spec = self.fm.scene
        if spec.get("attr"):
            with _h5().File(path, "r") as f:
                v = f.attrs.get(spec["attr"])
            if v is None:
                raise ValueError(f"scene attr {spec['attr']!r} missing in {path}")
            return v.decode() if isinstance(v, bytes) else str(v)
        if spec.get("path_tail"):
            return "/".join(Path(path).with_suffix("").parts[-int(spec["path_tail"]):])
        return Path(path).parent.name

    def width(self, path: str, which: str) -> int:
        """Row width after joining ``paths`` and selecting ``columns``."""
        paths, cols = ((self.fm.state_paths, self.fm.state_columns) if which == "state"
                       else (self.fm.action_paths, self.fm.action_columns))
        if cols is not None:
            return cols[1] - cols[0]
        with _h5().File(path, "r") as f:
            return sum(int(f[p].shape[1]) for p in paths)


def episode_record(path: str | Path, fm: FieldMap, reader: Optional[RoboDojoHDF5] = None,
                   *, robot: str, task_name: Optional[str] = None) -> Dict[str, Any]:
    """Catalog v2 record (same shape as data_prepare/robotwin_3d.py builds)."""
    reader = reader or RoboDojoHDF5(fm)
    path = str(Path(path).resolve())
    n = reader.lengths(path)
    if n["state"] != n["action"]:
        raise ValueError(f"{path}: state rows {n['state']} != action rows {n['action']}")
    rows = reader.frame_rows(path)
    if rows is None and n["video"] != n["state"]:
        raise ValueError(f"{path}: {n['video']} video frames vs {n['state']} rows and no "
                         "frame_rows mapping in the field map")
    if fm.depth_views and n.get("depth") != n["video"]:
        raise ValueError(f"{path}: depth frames {n.get('depth')} != rgb frames {n['video']}")
    text = reader.instruction(path) or (task_name or "").replace("_", " ")
    stem = hashlib.sha1(path.encode()).hexdigest()[:12]
    meta_video = {"layout": "robodojo_field_map", "compose_size": list(fm.layout.size),
                  "views": dict(fm.rgb_views), "field_map_digest": fm.digest,
                  "control_hz": fm.control_hz}
    if fm.depth_views:
        meta_video.update(depth_views=dict(fm.depth_views), depth_unit="mm")
    def dim(which, cols, paths):
        if cols is not None:
            return cols[1] - cols[0]
        return reader.width(path, which) if len(paths) > 1 else None
    return {
        "episode_id": f"{SOURCE_DATASET}:{stem}",
        "robot": robot,
        "video": {"count": n["video"], "storage": {"type": "file", "path": path}, "meta": meta_video},
        "state": {"count": n["state"], "storage": {"type": "file", "path": path},
                  "meta": {"path": fm.state_path, "columns": fm.state_columns,
                           "dimension": dim("state", fm.state_columns, fm.state_paths),
                           "feature_names": list(fm.state_names or []), "normalization": "none"}},
        "action": {"count": n["action"], "storage": {"type": "file", "path": path},
                   "meta": {"path": fm.action_path, "columns": fm.action_columns,
                            "dimension": dim("action", fm.action_columns, fm.action_paths),
                            "feature_names": list(fm.action_names or []), "normalization": "none",
                            "semantics": fm.action_semantics}},
        "texts": [{"index": 0, "kind": "instruction_en", "text": text,
                   "video_range": [0, n["video"]], "meta": {"task_name": task_name}}],
        "meta": {"scene_id": reader.scene_id(path), "task_name": task_name,
                 "field_map_digest": fm.digest, "has_frame_rows": rows is not None},
    }


def split_by_scene(records: Sequence[Mapping[str, Any]], *, val_fraction: float, seed: int,
                   exclude_scenes: Sequence[str] = ()) -> Dict[str, List[str]]:
    """Deterministic scene-level split: no scene appears in two splits."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    excl = set(map(str, exclude_scenes))
    out: Dict[str, List[str]] = {"train": [], "val": [], "excluded": []}
    for r in records:
        scene = str(r["meta"]["scene_id"])
        if scene in excl:
            out["excluded"].append(r["episode_id"])
            continue
        h = int(hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out["val" if h < val_fraction else "train"].append(r["episode_id"])
    return out
