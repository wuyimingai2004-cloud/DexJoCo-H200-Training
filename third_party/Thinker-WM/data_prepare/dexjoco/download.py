"""Pinned third-party ZIP downloads; standard library + curl, also on Windows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def verified(path: Path, size: int, expected: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest() == expected


def download_all(output: Path, *, only: list[str] | None = None, verify_only: bool = False):
    manifest = json.loads(Path(__file__).with_name("archive_manifest.json").read_text())
    selected = manifest["tasks"]
    if only:
        unknown = set(only) - {Path(k).name for k in selected}
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        selected = {k: v for k, v in selected.items() if Path(k).name in only}
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not verify_only and not curl:
        raise RuntimeError("curl is required (Windows includes curl.exe)")
    output.mkdir(parents=True, exist_ok=True)
    print(f"{len(selected)} ZIPs, {sum(v['zip_bytes'] for v in selected.values()) / 1e9:.2f} GB", flush=True)
    for task, entry in selected.items():
        path = output / (Path(task).name + ".zip")
        if verified(path, entry["zip_bytes"], entry["sha256"]):
            print("Already verified:", path.name, flush=True)
            continue
        if path.exists() or verify_only:
            raise RuntimeError(f"missing/corrupt ZIP: {path}; preserve or rename it before retrying")
        partial = path.with_suffix(".zip.part")
        if not verified(partial, entry["zip_bytes"], entry["sha256"]):
            url = (f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/"
                   f"{manifest['revision']}/{task}.zip")
            print("Downloading:", path.name, flush=True)
            subprocess.run([curl, "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                            "--continue-at", "-", "--output", str(partial), url], check=True)
        if not verified(partial, entry["zip_bytes"], entry["sha256"]):
            raise RuntimeError(f"SHA256/size mismatch: {partial}; do not extract")
        partial.replace(path)
        print("Verified:", path.name, flush=True)
    receipt = {**manifest, "tasks": selected}
    (output / "download_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("All selected ZIPs verified. Extract each ZIP into this directory.", flush=True)


def main():
    p = argparse.ArgumentParser(description="Download six rand_obj LeRobot v2 ZIPs, with resume and SHA256")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    download_all(a.output, only=a.tasks, verify_only=a.verify_only)
