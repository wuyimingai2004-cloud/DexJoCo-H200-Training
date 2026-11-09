"""Load EgoScale head/pose + left-camera K for motion-mask compensation.

AgiBot has no camera pose. Only ``egoscale:*`` episodes are filled; others stay
empty and keep the uncompensated 3-gate mask.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

TRAIN_H, TRAIN_W = 480, 640
NATIVE_H, NATIVE_W = 1080, 1920


def k_train_from_native(k_native: np.ndarray) -> np.ndarray:
    new_h, new_w = NATIVE_H // 3, NATIVE_W // 3
    y0, x0 = (TRAIN_H - new_h) // 2, (TRAIN_W - new_w) // 2
    K = np.asarray(k_native, dtype=np.float64).copy()
    K[0, :] *= new_w / NATIVE_W
    K[1, :] *= new_h / NATIVE_H
    K[0, 2] += x0
    K[1, 2] += y0
    return K


def _pose7_to_T(p: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in p[3:7]]
    n = (x * x + y * y + z * z + w * w) ** 0.5
    T = np.eye(4, dtype=np.float64)
    if n >= 1e-8:
        x, y, z, w = x / n, y / n, z / n, w / n
        T[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    T[:3, 3] = np.asarray(p[:3], dtype=np.float64)
    return T


def _inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _warp_by_T(
    z: np.ndarray, valid: np.ndarray, luma: np.ndarray, T_rel: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = z.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    Z = z.astype(np.float64)
    P = np.stack([(us - cx) * Z / fx, (vs - cy) * Z / fy, Z], axis=-1)
    P2 = P @ T_rel[:3, :3].T + T_rel[:3, 3]
    Z2 = P2[..., 2]
    u2 = fx * P2[..., 0] / np.clip(Z2, 1e-6, None) + cx
    v2 = fy * P2[..., 1] / np.clip(Z2, 1e-6, None) + cy
    m = valid & (Z > 1e-4) & (Z2 > 1e-4)
    uu = np.rint(u2[m]).astype(np.int32)
    vv = np.rint(v2[m]).astype(np.int32)
    zz = Z2[m].astype(np.float32)
    lu = luma[m].astype(np.float32)
    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    uu, vv, zz, lu = uu[ok], vv[ok], zz[ok], lu[ok]
    out_z = np.full((h, w), np.inf, np.float32)
    out_l = np.zeros((h, w), np.float32)
    out_v = np.zeros((h, w), dtype=bool)
    if uu.size:
        order = np.argsort(-zz)
        uu, vv, zz, lu = uu[order], vv[order], zz[order], lu[order]
        out_z[vv, uu] = zz
        out_l[vv, uu] = lu
        out_v[vv, uu] = True
    out_z[~out_v] = 0.0
    return out_z, out_v, out_l


def pose_motion_mask(
    z: np.ndarray,
    valid: np.ndarray,
    luma: np.ndarray,
    poses: np.ndarray,
    head_ok: np.ndarray,
    K: np.ndarray,
    thresh_m: float = 0.10,
    rgb_thresh: float = 0.05,
    residual_m: float = 0.05,
    blur_ks: int = 41,
) -> tuple[np.ndarray, bool]:
    """Same 3 gates as the base mask, after pose-aligning frame t onto t+1.

    Second value is True if at least one consecutive pair had valid poses.
    Callers should keep the uncompensated 3-gate mask when this is False.
    """
    from thinker_wm.models.depth_motion import depth_motion_mask

    n, h, w = z.shape
    out = np.zeros((n, h, w), dtype=bool)
    Ts = np.stack([_pose7_to_T(p) for p in poses], 0)
    warped = 0
    for t in range(n - 1):
        if not (bool(head_ok[t]) and bool(head_ok[t + 1])):
            continue
        T_rel = _inv_T(Ts[t + 1]) @ Ts[t]
        za, va, la = _warp_by_T(z[t], valid[t], luma[t], T_rel, K)
        z_pair = np.stack([za, z[t + 1]], 0)
        v_pair = np.stack([va, valid[t + 1]], 0)
        rgb = np.stack(
            [np.stack([la] * 3, 0), np.stack([luma[t + 1]] * 3, 0)], 0
        )
        m = depth_motion_mask(
            torch.from_numpy(z_pair).unsqueeze(0),
            torch.from_numpy(v_pair).unsqueeze(0),
            rgb=torch.from_numpy(rgb).unsqueeze(0),
            thresh_m=thresh_m,
            rgb_thresh=rgb_thresh,
            residual_m=residual_m,
            blur_ks=blur_ks,
        )[0, 1].numpy()
        out[t + 1] |= m
        out[t] |= m
        warped += 1
    return out, warped > 0


def _poses7_to_T(poses: torch.Tensor) -> torch.Tensor:
    xyz = poses[..., :3]
    x, y, z, w = poses[..., 3], poses[..., 4], poses[..., 5], poses[..., 6]
    n = (x * x + y * y + z * z + w * w).clamp_min(1e-8).sqrt()
    x, y, z, w = x / n, y / n, z / n, w / n
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    T = poses.new_zeros(*poses.shape[:-1], 4, 4)
    T[..., 3, 3] = 1
    T[..., :3, :3] = R
    T[..., :3, 3] = xyz
    return T


def _invert_T(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    Ti = torch.zeros_like(T)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    Ti[..., 3, 3] = 1
    return Ti


def _warp_prev_to_next(
    z: torch.Tensor,
    valid: torch.Tensor,
    luma: torch.Tensor,
    T_rel: torch.Tensor,
    K: torch.Tensor,
    pair_ok: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp frames ``[:, :-1]`` into the next camera. All args already sliced."""
    b, c, h, w = z.shape
    dtype = z.dtype
    fx = K[:, 0, 0].to(dtype=dtype).view(b, 1, 1, 1)
    fy = K[:, 1, 1].to(dtype=dtype).view(b, 1, 1, 1)
    cx = K[:, 0, 2].to(dtype=dtype).view(b, 1, 1, 1)
    cy = K[:, 1, 2].to(dtype=dtype).view(b, 1, 1, 1)
    vs = torch.arange(h, device=z.device, dtype=dtype).view(1, 1, h, 1)
    us = torch.arange(w, device=z.device, dtype=dtype).view(1, 1, 1, w)
    X = (us - cx) * z / fx
    Y = (vs - cy) * z / fy
    P = torch.stack([X, Y, z], dim=-1)
    R = T_rel[..., :3, :3]
    t = T_rel[..., :3, 3]
    P2 = torch.einsum("bchwi,bcji->bchwj", P, R) + t[:, :, None, None, :]
    z2 = P2[..., 2]
    u2 = fx * P2[..., 0] / z2.clamp_min(1e-6) + cx
    v2 = fy * P2[..., 1] / z2.clamp_min(1e-6) + cy
    inb = (
        valid
        & pair_ok.view(b, c, 1, 1)
        & (z > 1e-4)
        & (z2 > 1e-4)
        & (u2 >= 0)
        & (u2 < w)
        & (v2 >= 0)
        & (v2 < h)
    )
    uu = u2.round().long().clamp(0, w - 1)
    vv = v2.round().long().clamp(0, h - 1)
    n = h * w
    idx = (vv * w + uu).reshape(b, c, n)
    hit = inb.reshape(b, c, n)
    z_src = torch.where(hit, z2.reshape(b, c, n), z.new_full((), float("inf")))
    out_z = torch.full((b, c, n), float("inf"), device=z.device, dtype=dtype)
    out_z.scatter_reduce_(2, idx, z_src, reduce="amin", include_self=True)
    won = hit & (z_src == out_z.gather(2, idx))
    idx_l = torch.where(won, idx, idx.new_full((), n))
    out_l = luma.new_zeros((b, c, n + 1))
    out_l.scatter_(2, idx_l, luma.reshape(b, c, n))
    out_z = out_z.reshape(b, c, h, w)
    out_v = torch.isfinite(out_z)
    out_z = torch.where(out_v, out_z, torch.zeros_like(out_z))
    return out_z, out_v, out_l[..., :n].reshape(b, c, h, w)


def pose_motion_mask_torch(
    depth_m: torch.Tensor,
    valid: torch.Tensor,
    luma: torch.Tensor,
    poses: torch.Tensor,
    head_ok: torch.Tensor,
    K: torch.Tensor,
    *,
    thresh_m: float = 0.10,
    rgb_thresh: float = 0.05,
    residual_m: float = 0.05,
    blur_ks: int = 41,
    chunk_t: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GPU pose-align + 3-gate. Inputs ``[B, T, H, W]`` / poses ``[B, T, 7]``.

    Returns mask ``[B, T, H, W]`` and per-sample ``ok`` ``[B]``.
    """
    from thinker_wm.models.depth_motion import depth_motion_mask

    b, t, h, w = depth_m.shape
    device = depth_m.device
    dtype = depth_m.dtype
    poses = poses.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    head_ok = head_ok.to(device=device).bool()
    pair_ok = head_ok[:, :-1] & head_ok[:, 1:]
    Ts = _poses7_to_T(poses)
    T_rel = _invert_T(Ts[:, 1:]) @ Ts[:, :-1]
    z_w = depth_m.new_zeros(b, t - 1, h, w)
    v_w = torch.zeros(b, t - 1, h, w, dtype=torch.bool, device=device)
    l_w = luma.new_zeros(b, t - 1, h, w)
    for s in range(0, t - 1, chunk_t):
        e = min(s + chunk_t, t - 1)
        za, va, la = _warp_prev_to_next(
            depth_m[:, s:e],
            valid[:, s:e],
            luma[:, s:e],
            T_rel[:, s:e],
            K,
            pair_ok[:, s:e],
        )
        z_w[:, s:e] = za
        v_w[:, s:e] = va
        l_w[:, s:e] = la
    tp = t - 1
    z2 = torch.stack([z_w, depth_m[:, 1:]], dim=2).reshape(b * tp, 2, h, w)
    v2 = torch.stack([v_w, valid[:, 1:]], dim=2).reshape(b * tp, 2, h, w)
    l2 = torch.stack([l_w, luma[:, 1:]], dim=2).reshape(b * tp, 2, h, w)
    rgb = l2.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
    moving = depth_motion_mask(
        z2,
        v2,
        rgb=rgb,
        thresh_m=thresh_m,
        rgb_thresh=rgb_thresh,
        residual_m=residual_m,
        blur_ks=blur_ks,
    )[:, 1].reshape(b, tp, h, w)
    moving = moving & pair_ok.view(b, tp, 1, 1)
    out = torch.zeros(b, t, h, w, dtype=torch.bool, device=device)
    out[:, 1:] |= moving
    out[:, :-1] |= moving
    return out, pair_ok.any(dim=1)


class EgoPoseCache:
    """Process-local repo cache. One open_dataset per Ego source."""

    def __init__(self, config: Any):
        self.config = config
        self._repos: dict[str, Any] = {}
        self._k: dict[str, np.ndarray] = {}
        self._failed: set[str] = set()

    def close(self) -> None:
        for repo in self._repos.values():
            try:
                repo.close()
            except Exception:
                pass
        self._repos.clear()

    def _repo(self, source_name: str):
        if source_name in self._repos:
            return self._repos[source_name]
        from thinker_wm.data_prepare.factory import _indexed_sources
        from motus_data.indexed_pipeline import open_dataset
        from motus_data.releases import get_episode_dataset_release

        sources = _indexed_sources(self.config, eval=False)
        source = next((s for s in sources if s["name"] == source_name), None)
        if source is None:
            sources = _indexed_sources(self.config, eval=True)
            source = next((s for s in sources if s["name"] == source_name), None)
        if source is None:
            raise KeyError(source_name)
        release = get_episode_dataset_release(source["dataset"], source.get("release"))
        repo = open_dataset(
            source["dataset"],
            release=release,
            catalog_root=source.get("catalog_root"),
            catalog_path=source.get("catalog_path"),
            index_path=source.get("index_path"),
            readers=source.get("readers"),
        )
        self._repos[source_name] = repo
        return repo

    def _calib_path(self, episode) -> Optional[Path]:
        rec = episode._catalog_record
        for blob in (rec.get("meta"), rec.get("metadata"), episode.info.metadata):
            if not isinstance(blob, dict):
                continue
            cal = blob.get("calibration")
            if isinstance(cal, dict):
                p = cal.get("intrinsics") or cal.get("undistorted_intrinsics")
                if p and Path(p).is_file():
                    return Path(p)
        return None

    def read(
        self, source_name: str, episode_id: str, indices: list[int]
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        key = f"{source_name}|{episode_id}"
        if key in self._failed:
            return None
        try:
            repo = self._repo(source_name)
            ep = repo.get_episode(episode_id)
            raw = ep.read_source_modalities(
                {"head/pose": indices, "head/valid": indices}
            )
            pose = raw["head/pose"].numpy()
            valid = raw["head/valid"].numpy().reshape(-1).astype(bool)
            if episode_id not in self._k:
                cal = self._calib_path(ep)
                if cal is None:
                    raise FileNotFoundError(f"no K for {episode_id}")
                intr = json.loads(cal.read_text())
                self._k[episode_id] = k_train_from_native(
                    np.asarray(intr["cameras"]["left"]["K"], dtype=np.float64)
                )
            return pose, valid, self._k[episode_id]
        except Exception as error:
            logger.warning("ego pose skip %s: %s", episode_id, error)
            self._failed.add(key)
            return None

    def attach_tensors(
        self,
        source_name: Optional[str],
        episode_ids: list,
        frame_indices: list,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns pose [B,T,7], valid [B,T], K [B,3,3], used [B] bool.

        ``source_name`` must be the mix child name (``egoscale_first500h_*``),
        not the catalog family (``egoscale``): only the child carries the
        release/index needed to open the repo.
        """
        b = len(episode_ids)
        poses = np.zeros((b, T, 7), dtype=np.float32)
        valid = np.zeros((b, T), dtype=bool)
        Ks = np.zeros((b, 3, 3), dtype=np.float32)
        used = np.zeros((b,), dtype=bool)
        for i in range(b):
            eid = str(episode_ids[i])
            if not source_name or not eid.startswith("egoscale"):
                continue
            fi = [int(x) for x in list(frame_indices[i])]
            # depth_01 = first+video has T frames; adapter stores the same T indices
            if len(fi) != T:
                continue
            got = self.read(str(source_name), eid, fi)
            if got is None:
                continue
            p, v, k = got
            poses[i] = p
            valid[i] = v
            Ks[i] = k
            used[i] = True
        return (
            torch.from_numpy(poses),
            torch.from_numpy(valid),
            torch.from_numpy(Ks),
            torch.from_numpy(used),
        )
