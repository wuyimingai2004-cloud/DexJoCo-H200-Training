"""Read-only RoboColiseum LeRobot v2.1 indexing and aligned offline windows.

This is recorded data, not an env.step implementation or a native task verifier.
Source splits are preserved; an explicit complete split map can reserve holdouts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "thinkerwam.robocoliseum-episodes/v1"
CONTRACT = "geniesim-g2-arm-base-absolute-38d-xyzw-30hz/v1"
VIEWS = ("top_head", "hand_left", "hand_right")
SPLITS = {"train", "development", "validation", "val", "test", "holdout", "excluded"}
# All 38 mapped state fields have the same semantics as the action fields.
STATE_COLUMNS = [0, 1, *range(24, 30), *range(16, 24), *range(30, 44),
                 *range(58, 61), *range(61, 66)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes source root: {relative}")
    return path


def _split(info, index):
    matches = []
    for name, span in info.get("splits", {}).items():
        lo, hi = map(int, span.split(":"))
        if lo <= index < hi:
            matches.append(name)
    if len(matches) != 1 or matches[0] not in SPLITS:
        raise ValueError(f"missing/overlapping/unsupported source split for episode {index}")
    return matches[0]


def _contract(info):
    if (info.get("codebase_version") != "v2.1" or info.get("robot_type") != "g2a_sim"
            or info.get("fps") != 30):
        raise ValueError("expected LeRobot v2.1 g2a_sim at 30 Hz")
    features = info["features"]
    for key, size in (("observation.state", 109), ("action", 38)):
        if features[key]["shape"] != [size]:
            raise ValueError(f"unexpected {key} dimension")
    checks = (("observation.state", "state/end/arm_position", list(range(24, 30))),
              ("observation.state", "state/end/arm_orientation", list(range(16, 24))),
              ("action", "action/end/position", list(range(2, 8))),
              ("action", "action/end/orientation", list(range(8, 16))))
    for feature, field, indices in checks:
        description = features[feature]["field_descriptions"][field]
        if description["indices"] != indices or "arm_base_link" not in description["description"]:
            raise ValueError(f"unrecognized arm-base field: {field}")
        if "orientation" in field and "x/y/z/w" not in description["description"]:
            raise ValueError("quaternion order must be explicitly xyzw")


def _segments(info, annotation, index, length):
    source = "info.instruction_segments"
    raw = info.get("instruction_segments", {}).get(str(index))
    if not raw:
        source = "annotations.action_steps"
        raw = [dict(instruction=s.get("action_text"), start_frame_index=s["start_frame"],
                    end_frame_index=s["end_frame"], track=s.get("track"))
               for s in annotation.get("action_steps", [])]
    segments = []
    for item in raw or []:
        lo, hi = int(item["start_frame_index"]), int(item["end_frame_index"]) + 1
        text = str(item.get("instruction") or "").strip()
        if not text or not 0 <= lo < hi <= length:
            raise ValueError(f"invalid instruction interval {lo}:{hi}, length={length}")
        overlaps = [s for s in annotation.get("action_steps", [])
                    if int(s["start_frame"]) < hi and int(s["end_frame"]) + 1 > lo]
        # Require full annotation coverage, not just an overlapping good label.
        covered = set()
        for s in overlaps:
            if s.get("is_mistake") is False:
                covered.update(range(max(lo, int(s["start_frame"])),
                                     min(hi, int(s["end_frame"]) + 1)))
        reviewed = (annotation.get("episode_status") == "approved"
                    and len(covered) == hi - lo
                    and all(s.get("is_mistake") is False for s in overlaps))
        segments.append(dict(instruction=text, start=lo, stop=hi, text_source=source,
                             reviewed_demonstration=reviewed))
    if not segments:
        raise ValueError("no usable instruction segments; task names are not instructions")
    return segments


def read_arrays(record, *, verify_hash=True):
    """Validate all rows, timestamps and identity; pyarrow/numpy are lazy dependencies."""
    import numpy as np
    import pyarrow.parquet as pq

    path = Path(record["parquet"])
    if verify_hash and file_hash(path) != record["parquet_sha256"]:
        raise ValueError("parquet differs from indexed source")
    table = pq.read_table(path)
    n = record["length"]
    if table.num_rows != n:
        raise ValueError("parquet length differs from metadata")
    arrays = {k: np.asarray(table[k].to_pylist()) for k in
              ("observation.state", "action", "frame_index", "episode_index", "timestamp")}
    for key, dim in (("observation.state", 109), ("action", 38)):
        if arrays[key].shape != (n, dim) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid shape/nonfinite values in {key}")
    if not np.array_equal(arrays["frame_index"], np.arange(n)):
        raise ValueError("frame indices must be contiguous and episode-local")
    if not np.all(arrays["episode_index"] == record["episode_index"]):
        raise ValueError("cross-episode parquet rows")
    if not np.allclose(arrays["timestamp"], np.arange(n) / record["fps"], atol=1e-4, rtol=0):
        raise ValueError("timestamp/video 30 Hz alignment mismatch")
    return arrays


def inspect_videos(record, *, hash_media=False):
    try:
        import av
    except ImportError:
        av = None

    for view, entry in record["videos"].items():
        if av is not None:
            # Container metadata does not require initializing a video decoder.
            with av.open(entry["path"]) as container:
                if len(container.streams.video) != 1:
                    raise ValueError(f"expected one video stream: {view}")
                stream = container.streams.video[0]
                count, fps = stream.frames, float(stream.average_rate or 0)
                shape = [stream.height, stream.width]
        else:
            import cv2

            cap = cv2.VideoCapture(entry["path"])
            try:
                if not cap.isOpened():
                    raise ValueError(f"cannot open video: {view}")
                count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                shape = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
            finally:
                cap.release()
        if count != record["length"] or abs(fps - record["fps"]) > 0.05 or shape != entry["size_hw"]:
            raise ValueError(f"video count/fps/shape mismatch: {view}")
        if hash_media:
            entry["sha256"] = file_hash(entry["path"])


def scan(root, *, split_map=None, validate_rows=False, validate_videos=False, hash_media=False,
         workers=1, _task_dirs=None):
    """Index all tasks; reject bad episodes explicitly, never silently relabel outcomes.

    split_map maps every episode_id to an allowed split. It overrides source splits
    and must cover the entire source exactly (prevents accidental holdout leakage).
    Video validation checks stream headers; it is not a full-frame decode test.
    """
    root = Path(root).resolve(strict=True)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    task_dirs = _task_dirs if _task_dirs is not None else [
        p for p in sorted(root.iterdir()) if p.is_dir() and (p / "meta/info.json").exists()]
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        def one(task_dir):
            prefix = f"robocoliseum_if:{task_dir.name}:"
            subset = None if split_map is None else {k: v for k, v in split_map.items() if k.startswith(prefix)}
            return scan(root, split_map=subset, validate_rows=validate_rows,
                        validate_videos=validate_videos, hash_media=hash_media,
                        _task_dirs=[task_dir])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(one, task_dirs))
        records = [r for rows, _ in batches for r in rows]
        errors = [e for _, rejected in batches for e in rejected]
        seen = {r["episode_id"] for r in records} | {e["episode_id"] for e in errors}
        if not seen or (split_map is not None and set(split_map) != seen):
            raise ValueError("empty source or split map contains unknown episode IDs")
        return records, errors
    records, errors, seen = [], [], set()
    for task_dir in task_dirs:
        if not task_dir.is_dir() or not (task_dir / "meta/info.json").exists():
            continue
        info = read_json(task_dir / "meta/info.json")
        _contract(info)
        annotations = read_json(task_dir / "meta/annotations.json")
        episodes = jsonl(task_dir / "meta/episodes.jsonl")
        if len(episodes) != info["total_episodes"]:
            raise ValueError(f"episode inventory mismatch: {task_dir.name}")
        meta_hashes = {name: file_hash(task_dir / "meta" / name)
                       for name in ("info.json", "annotations.json", "episodes.jsonl", "tasks.jsonl")}
        for ep in episodes:
            index, n = int(ep["episode_index"]), int(ep["length"])
            eid = f"robocoliseum_if:{task_dir.name}:{index:06d}"
            if eid in seen:
                raise ValueError(f"duplicate episode identity: {eid}")
            seen.add(eid)
            if split_map is not None and (eid not in split_map or split_map[eid] not in SPLITS):
                raise ValueError(f"invalid/missing split assignment: {eid}")
            try:
                if n < 2:
                    raise ValueError("episode has no complete transition")
                source_split = _split(info, index)
                fields = dict(episode_index=index, episode_chunk=index // int(info["chunks_size"]))
                parquet = inside(task_dir, info["data_path"].format(**fields))
                annotation = annotations.get(str(index), {})
                record = dict(episode_id=eid, task=task_dir.name, episode_index=index,
                              robot="geniesim-g2", fps=30, length=n, action_contract=CONTRACT,
                              source_root=str(root), source_split=source_split,
                              split=split_map[eid] if split_map is not None else source_split,
                              outcome="unverified", source_review=annotation.get("episode_status"),
                              source_episode_id=annotation.get("episode_id"),
                              parquet=str(parquet), parquet_sha256=file_hash(parquet),
                              metadata_sha256=meta_hashes, videos={},
                              instruction_segments=_segments(info, annotation, index, n),
                              validation=dict(rows=validate_rows, video_headers=validate_videos,
                                              video_hashes=hash_media))
                for view in VIEWS:
                    key = "observation.images." + view
                    feature = info["features"][key]
                    path = inside(task_dir, info["video_path"].format(**fields, video_key=key))
                    if not path.is_file():
                        raise ValueError(f"missing video: {path}")
                    record["videos"][view] = dict(path=str(path), size_hw=feature["shape"][:2],
                                                 bytes=path.stat().st_size)
                if validate_rows:
                    read_arrays(record, verify_hash=False)
                if validate_videos or hash_media:
                    inspect_videos(record, hash_media=hash_media)
                    record["validation"]["video_headers"] = True
                record["record_sha256"] = digest(record)
                records.append(record)
            except (ValueError, KeyError, OSError) as exc:
                errors.append(dict(episode_id=eid, error=str(exc)))
    if not seen:
        raise ValueError("no LeRobot task datasets found")
    if split_map is not None and set(split_map) != seen:
        raise ValueError("split map has unknown episode IDs")
    return records, errors


def load_catalog(path):
    records = jsonl(path)
    seen = set()
    for record in records:
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        if digest(body) != record["record_sha256"] or record["episode_id"] in seen:
            raise ValueError("catalog digest mismatch or duplicate episode")
        if record["action_contract"] != CONTRACT or record["split"] not in SPLITS:
            raise ValueError("unsupported catalog contract/split")
        seen.add(record["episode_id"])
    return records


def read_window(record, start, stop, *, action_offset):
    """Return N actions, N+1 observation indices, and native/raw + aligned state.

    [start, stop) is the action-transition interval; offset must be explicitly
    chosen as 0 or 1 after verifying the collection/control convention. No
    terminal-frame padding or guessed action shift is allowed.
    """
    import numpy as np

    if action_offset not in (0, 1) or type(action_offset) is not int:
        raise ValueError("explicit action_offset 0 or 1 required")
    if not 0 <= start < stop < record["length"]:
        raise ValueError("window needs a real observation after its last action")
    arrays = read_arrays(record)
    raw = arrays["observation.state"][start:stop + 1].astype(np.float32)
    return dict(raw_states=raw, states=raw[:, STATE_COLUMNS],
                actions=arrays["action"][start + action_offset:stop + action_offset].astype(np.float32),
                observation_indices=np.arange(start, stop + 1),
                timestamps=arrays["timestamp"][start:stop + 1],
                action_indices=np.arange(start + action_offset, stop + action_offset))


def read_rgb(record, view, indices):
    import cv2
    import numpy as np

    if view not in VIEWS or not indices:
        raise ValueError("a known camera and nonempty frame indices are required")
    entry = record["videos"][view]
    if "sha256" in entry and file_hash(entry["path"]) != entry["sha256"]:
        raise ValueError("video differs from indexed source")
    cap = cv2.VideoCapture(entry["path"])
    frames = []
    try:
        for index in indices:
            if not 0 <= index < record["length"]:
                raise ValueError("video index outside episode")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or list(frame.shape[:2]) != entry["size_hw"]:
                raise ValueError(f"cannot decode {view} frame {index}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames)
