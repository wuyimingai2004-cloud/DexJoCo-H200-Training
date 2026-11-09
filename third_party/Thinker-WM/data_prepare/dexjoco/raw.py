"""Read official replay.zarr and RGB videos; validate without the simulator."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .contracts import Task, state_to_model


def read_lowdim(root: Path, task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (root / "replay.zarr").is_dir():
        import zarr
        data = zarr.open_group(str(root / "replay.zarr"), mode="r")
        ends = np.asarray(data["meta/episode_ends"][:])
        data = data["data"]
        arrays = {k: np.asarray(data[k][:]) for k in ("state", "timestamp")}
        action_key = "action_rotvec" if "action_rotvec" in data else "action"
        arrays[action_key] = np.asarray(data[action_key][:])
        if len(ends) != 1 or int(ends[0]) != len(arrays["state"]):
            raise ValueError("one original episode per replay.zarr is required")
    elif (root / "replay.npz").is_file():
        # Explicit uncompressed interchange/fixture equivalent of replay.zarr/data.
        with np.load(root / "replay.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    else:
        raise FileNotFoundError(f"no replay.zarr (or replay.npz) in {root}")
    state = state_to_model(arrays["state"], task.arms)
    if "action_rotvec" in arrays:
        action = np.asarray(arrays["action_rotvec"], dtype=np.float32)
    else:
        native = arrays["action"]
        if native.shape[-1] != 23 * task.arms:
            raise ValueError("action without action_rotvec must use native quaternion layout")
        action = state_to_model(native, task.arms)
    times = np.asarray(arrays["timestamp"], dtype=np.float64).reshape(-1)
    n = len(state)
    if state.shape != (n, task.dimension) or action.shape != (n, task.dimension) or len(times) != n:
        raise ValueError("state/action/timestamp lengths or dimensions disagree")
    if n < 25 or not np.isfinite(action).all() or not np.isfinite(times).all():
        raise ValueError("episode needs at least 25 finite steps")
    if not np.allclose(np.diff(times), 1 / 30, atol=1e-4, rtol=1e-3):
        raise ValueError("recording is not aligned at 30 Hz; resample explicitly before preparation")
    return state, action, times


def video_path(root: Path, camera: str) -> Path:
    for suffix in (".mp4", ".npy"):
        path = root / "videos" / (camera + suffix)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"missing camera {camera!r} in {root / 'videos'}")


def inspect_video(path: Path, count: int) -> None:
    if path.suffix == ".npy":
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if frames.ndim != 4 or frames.shape[0] != count or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"invalid RGB video array: {path}")
        return
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 30) > .05:
            raise ValueError(f"RGB video must have 30 fps: {path}")
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"invalid RGB frame: {path}")
            decoded += 1
        if decoded != count:
            raise ValueError(f"decoded {decoded} video frames vs {count} lowdim rows: {path}")
    finally:
        cap.release()


def read_frames(path: str, indices) -> np.ndarray:
    indices = [int(i) for i in indices]
    if Path(path).suffix == ".npy":
        return np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)[indices])
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        frames = {}
        # Decode one contiguous short span, avoiding a seek for every frame.
        start, end = min(indices), max(indices)
        if (not cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - start) > .5):
            raise ValueError(f"cannot seek video: {path}")
        wanted = set(indices)
        for i in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"cannot decode frame {i}: {path}")
            if i in wanted:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.stack([frames[i] for i in indices])
    finally:
        cap.release()


def compose_rgb(views: list[np.ndarray]) -> np.ndarray:
    """480x640 T canvas: base at top; right wrist then left wrist at bottom."""
    from PIL import Image, ImageOps
    if len(views) not in (2, 3) or len({len(v) for v in views}) != 1:
        raise ValueError("expected aligned base + one/two wrist views")
    output = np.zeros((len(views[0]), 480, 640, 3), dtype=np.uint8)
    for t in range(len(output)):
        for v, frames in enumerate(views):
            size = (640, 320) if v == 0 else (320, 160)
            x = 0 if v == 0 else (160 if len(views) == 2 else (v - 1) * 320)
            y = 0 if v == 0 else 320
            frame = ImageOps.pad(Image.fromarray(frames[t]), size, method=Image.Resampling.BILINEAR)
            output[t, y:y + size[1], x:x + size[0]] = np.asarray(frame)
    return output
