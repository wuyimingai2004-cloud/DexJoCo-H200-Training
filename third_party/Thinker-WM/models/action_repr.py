"""旋转表示转换工具（MotusBrain 对齐改造 B1/B2，20260811）。

统一动作表示的训练目标用 6D 旋转（论文 §2.2：每臂 pos3+rot6+grip1，
输入 quat/rotvec、训练目标统一 6D）；相对表示为相对条件帧 state 的
R_rel = R_sᵀ·R_i（推理输出端 R_abs = R_s·R_rel 还原）。

本模块只做纯张量转换（batch 维透明，内部一律 fp32 计算后 cast 回输入
dtype，bf16 安全），不依赖任何配置/机器人语义。已知表示：

- ``rotvec``（3 维，agibot 本地 EEF 姿态）
- ``quat``（4 维，robotmind tianyi/tienkung；分量序由 quat_order 指定，
  默认 xyzw——robotmind 的分量序仓库无文档，见
  configs/micro_train_debug.yaml 风险注释）
- ``rot3``（3 维，robotmind ark/arkmobile；定义仓库无文档，按 rotvec
  处理——仅 micro 路径涉及，见 configs/micro_train_debug.yaml 风险注释）
- ``rot6``（6 维，Zhou et al. 连续 6D：旋转矩阵前两列拼接）
"""

from __future__ import annotations

from typing import Tuple

import torch

REPR_WIDTHS = {"rotvec": 3, "quat": 4, "rot3": 3, "rot6": 6}

_EPS = 1e-8


def repr_width(repr_name: str) -> int:
    if repr_name not in REPR_WIDTHS:
        raise ValueError(
            f"未知旋转表示 {repr_name!r}（已注册：{sorted(REPR_WIDTHS)}）"
        )
    return REPR_WIDTHS[repr_name]


def _eye(ref: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=ref.dtype, device=ref.device).expand(
        *ref.shape[:-2], 3, 3
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """rotvec [..., 3] → 旋转矩阵 [..., 3, 3]（Rodrigues，小角度稳定）。

    用未归一化形式 R = I + A·K + B·K²（K = skew(rv)，A = sinθ/θ，
    B = (1−cosθ)/θ²），θ→0 时 A→1、B→1/2，避免 0/0。
    """
    dtype_in = rotvec.dtype
    rv = rotvec.float()
    theta = rv.norm(dim=-1, keepdim=True)  # [..., 1]
    small = theta < 1e-4
    theta_safe = theta.clamp(min=1e-8)
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(theta_safe) / theta_safe)
    b = torch.where(
        small,
        0.5 - theta**2 / 24.0,
        (1.0 - torch.cos(theta_safe)) / theta_safe**2,
    )
    x, y, z = rv.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*rv.shape[:-1], 3, 3)
    k2 = k @ k
    eye = torch.eye(3, dtype=k.dtype, device=k.device).expand_as(k)
    r = eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2
    return r.to(dtype_in)


def matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 [..., 3, 3] → rotvec [..., 3]（经 quaternion 中转）。

    直取反对称分量的公式在 θ→π 时 1/sinθ 病态；改经 Shepperd quat 中转：
    θ = 2·arccos(w)（w=cos(θ/2)，θ≈π 处 w≈0 条件良好）、axis = xyz/sin(θ/2)
    （θ≈π 处 sin(θ/2)≈1，全角度良态）；θ→0 用一阶展开 rv≈2·xyz。
    """
    dtype_in = matrix.dtype
    q = matrix_to_quat(matrix, "wxyz").float()
    w, xyz = q[..., :1], q[..., 1:4]
    s = (1.0 - w * w).clamp(min=0.0).sqrt()  # sin(θ/2) ≥ 0（w 已规范化到 ≥0 半球）
    theta = 2.0 * torch.arccos(w.clamp(-1.0, 1.0))
    small = s < 1e-4
    rv = torch.where(small, 2.0 * xyz, theta * xyz / s.clamp(min=1e-8))
    return rv.to(dtype_in)


def quat_to_matrix(quat: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    """quat [..., 4] → 旋转矩阵 [..., 3, 3]（零范数回退单位旋转）。"""
    dtype_in = quat.dtype
    q = quat.float()
    if order == "xyzw":
        x, y, z, w = q.unbind(-1)
    elif order == "wxyz":
        w, x, y, z = q.unbind(-1)
    else:
        raise ValueError(f"未知 quat 分量序 {order!r}（支持 xyzw/wxyz）")
    norm = (x**2 + y**2 + z**2 + w**2).sqrt().clamp(min=1e-8)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    two = 2.0
    r = torch.stack(
        [
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    zero_input = (quat.float().norm(dim=-1) < 1e-8)[..., None, None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(r)
    r = torch.where(zero_input, eye, r)
    return r.to(dtype_in)


def matrix_to_quat(matrix: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    """旋转矩阵 [..., 3, 3] → quat [..., 4]（Shepperd 法，全角度良态）。

    各分量分别开方后按反对称分量定号，归一化到单位四元数并规范到
    w ≥ 0 半球（消 q/−q 二义性，下游 rotvec 主值一致）。
    """
    dtype_in = matrix.dtype
    r = matrix.float()
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    w = torch.sqrt((1.0 + m00 + m11 + m22).clamp(min=0.0)) / 2.0
    x = torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=0.0)) / 2.0
    y = torch.sqrt((1.0 - m00 + m11 - m22).clamp(min=0.0)) / 2.0
    z = torch.sqrt((1.0 - m00 - m11 + m22).clamp(min=0.0)) / 2.0
    x = torch.copysign(x, r[..., 2, 1] - r[..., 1, 2])
    y = torch.copysign(y, r[..., 0, 2] - r[..., 2, 0])
    z = torch.copysign(z, r[..., 1, 0] - r[..., 0, 1])
    q = torch.stack([x, y, z, w], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    neg = q[..., 3:4] < 0.0
    q = torch.where(neg.expand_as(q), -q, q)
    if order == "xyzw":
        out = q
    elif order == "wxyz":
        out = torch.cat([q[..., 3:4], q[..., :3]], dim=-1)
    else:
        raise ValueError(f"未知 quat 分量序 {order!r}（支持 xyzw/wxyz）")
    return out.to(dtype_in)


def matrix_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 [..., 3, 3] → 6D [..., 6]（前两列拼接：x 列 ++ y 列）。"""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def d6_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D [..., 6] → 旋转矩阵 [..., 3, 3]（Gram-Schmidt 正交化）。

    近零/退化防护：a1 或正交化后 a2 范数过小（如 U 扩容新列零初始化权重
    在训练初期的近零输出）回退单位旋转，避免 0/0 产出 NaN。
    """
    dtype_in = d6.dtype
    x = d6.float()
    a1, a2 = x[..., :3], x[..., 3:6]
    n1 = a1.norm(dim=-1, keepdim=True)
    b1 = a1 / n1.clamp(min=1e-8)
    e2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    n2 = e2.norm(dim=-1, keepdim=True)
    b2 = e2 / n2.clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    r = torch.stack([b1, b2, b3], dim=-1)  # 列为基
    degenerate = ((n1 < 1e-6) | (n2 < 1e-6))[..., None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(r)
    r = torch.where(degenerate, eye, r)
    return r.to(dtype_in)


def repr_to_matrix(x: torch.Tensor, repr_name: str, quat_order: str = "xyzw") -> torch.Tensor:
    """任意已注册表示 [..., W] → 旋转矩阵 [..., 3, 3]。"""
    if repr_name in ("rotvec", "rot3"):
        return rotvec_to_matrix(x)
    if repr_name == "quat":
        return quat_to_matrix(x, order=quat_order)
    if repr_name == "rot6":
        return d6_to_matrix(x)
    raise ValueError(f"未知旋转表示 {repr_name!r}")


def matrix_to_repr(matrix: torch.Tensor, repr_name: str, quat_order: str = "xyzw") -> torch.Tensor:
    """旋转矩阵 [..., 3, 3] → 任意已注册表示 [..., W]。"""
    if repr_name in ("rotvec", "rot3"):
        return matrix_to_rotvec(matrix)
    if repr_name == "quat":
        return matrix_to_quat(matrix, order=quat_order)
    if repr_name == "rot6":
        return matrix_to_6d(matrix)
    raise ValueError(f"未知旋转表示 {repr_name!r}")


def relative_rotation_6d(
    state_rot: torch.Tensor,
    action_rot: torch.Tensor,
    repr_name: str,
    quat_order: str = "xyzw",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """B1/B2 训练目标变换：state 绝对 6D + action 相对 6D。

    Args:
        state_rot: [..., W] 条件帧姿态（本地原生表示）。
        action_rot: [..., T, W] 动作序列姿态（本地原生表示；state 的
            batch 维对 T 维广播）。
        repr_name: 本地原生旋转表示（rotvec/quat/rot3）。
        quat_order: quat 分量序。

    Returns:
        (state_6d [..., 6], action_rel_6d [..., T, 6])：state 为绝对姿态的
        6D（state token 保持绝对参考点）；action 为 R_rel = R_sᵀ·R_i 的 6D。
    """
    r_s = repr_to_matrix(state_rot, repr_name, quat_order)  # [..., 3, 3]
    r_i = repr_to_matrix(action_rot, repr_name, quat_order)  # [..., T, 3, 3]
    r_rel = r_s.transpose(-1, -2).unsqueeze(-3) @ r_i
    return matrix_to_6d(r_s), matrix_to_6d(r_rel)


def absolute_rotation_from_6d(
    state_rot: torch.Tensor,
    rel_6d: torch.Tensor,
    repr_name: str,
    quat_order: str = "xyzw",
) -> torch.Tensor:
    """B1/B2 推理还原：R_abs = R_s·R_rel，转回本地原生表示。

    Args:
        state_rot: [..., W] 条件帧姿态（本地原生表示，物理量纲）。
        rel_6d: [..., T, 6] 模型输出的相对旋转 6D（经正交化）。

    Returns:
        [..., T, W] 绝对姿态的本地原生表示。
    """
    r_s = repr_to_matrix(state_rot, repr_name, quat_order)  # [..., 3, 3]
    r_rel = d6_to_matrix(rel_6d)  # [..., T, 3, 3]
    r_abs = r_s.unsqueeze(-3) @ r_rel
    return matrix_to_repr(r_abs, repr_name, quat_order)
