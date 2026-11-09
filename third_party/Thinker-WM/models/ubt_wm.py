import sys
import json
import PIL
import torch
import logging
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import pickle
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import PIL.Image
import numpy as np

from thinker_wm.utils.vlm_utils import preprocess_vlm_messages

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from thinker_wm.wan.modules.model import sinusoidal_embedding_1d
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model import WanVideoModel
from .world_model import UniWorldModel
# Add Flow-Matching schedulers
from thinker_wm.wan.utils.fm import FlowMatchScheduler

from thinker_wm.utils.vlm_utils import _process_vlm_inputs_batch
from thinker_wm.models.action_stats import (
    normalize_batch, denormalize_batch, build_gripper_hand_policy)
from thinker_wm.models import action_repr
from thinker_wm.models.depth_conditioning import (
    depth_condition_mode, validate_depth_losses, sanitize_depth_targets,
    depth_latent_valid_mask, masked_depth_mse,
)
from thinker_wm.models import temporal_training as TT
from thinker_wm.models.prefix_error_recycling import RGBPrefixErrorBuffer, empty_metrics as prefix_error_metrics
from thinker_wm.models import capability_training as CT

logger = logging.getLogger(__name__)


def _record_tensor_finite_stats(timing: Dict[str, Any], name: str, tensor: Optional[torch.Tensor]) -> None:
    """Record lightweight finite/value stats for NaN source localization."""
    if timing is None or tensor is None or not torch.is_tensor(tensor):
        return
    with torch.no_grad():
        data = tensor.detach()
        finite = torch.isfinite(data)
        timing[f"{name}_all_finite"] = bool(finite.all().item())
        timing[f"{name}_nan_count"] = int(torch.isnan(data).sum().item())
        timing[f"{name}_inf_count"] = int(torch.isinf(data).sum().item())
        timing[f"{name}_numel"] = int(data.numel())
        if bool(finite.any().item()):
            finite_data = data[finite].float()
            timing[f"{name}_finite_abs_max"] = float(finite_data.abs().max().item())
            timing[f"{name}_finite_mean"] = float(finite_data.mean().item())
            timing[f"{name}_finite_std"] = float(finite_data.std(unbiased=False).item())
        else:
            timing[f"{name}_finite_abs_max"] = float("nan")
            timing[f"{name}_finite_mean"] = float("nan")
            timing[f"{name}_finite_std"] = float("nan")


# 无条件分支的固定文本条件（评审 #6）：训练期条件 dropout 把被 drop 样本的
# VLM 文本替换为该常量，推理期 CFG 的无条件前向也用同一常量，保证训练/推理
# 两侧的无条件输入一致。
UNCONDITIONAL_VLM_MESSAGE = ""


def _robot_action_columns(
    config: "MotusConfig",
    robot_names: Optional[List[str]],
) -> Tuple[Optional[List[int]], int, int]:
    """统一空间 → per-robot 原始维的目标列（推理输出端切片用）。

    返回 (dim_map 列索引列表 | None, offset, dim)：robot 注册了
    unified_dim_map 时返回逐维列索引（语义槽位 gather），否则回退
    offset 前填切片。batch 源齐次，取 robot_names[0]。
    """
    action_dim = int(config.action_dim)
    robot_offset = 0
    cols: Optional[List[int]] = None
    if robot_names:
        name = robot_names[0] if isinstance(robot_names, (list, tuple)) else robot_names
        layout = getattr(config, "robot_layout", None) or {}
        entry = layout.get(str(name))
        if entry is not None:
            dim_map = entry.get("dim_map")
            if dim_map is not None:
                cols = [int(c) for c in dim_map]
            else:
                robot_offset = int(entry["offset"])
        robot_configs = config.robot_configs or {}
        action_dim = int(robot_configs.get(str(name), action_dim))
    return cols, robot_offset, action_dim


def _slice_unified_actions(
    actions: torch.Tensor,
    cols: Optional[List[int]],
    offset: int,
    dim: int,
) -> torch.Tensor:
    """把统一空间 [B, T, U] 切回 per-robot 原始维 [B, T, dim]（dim_map gather
    或 offset 前填切片）。

    dim_map 含 -1（摘除维，如 agibot 灵巧手占位维，修复 #7）时，该本地维
    gather 列 0 占位并在归一化空间置 0；调用方在反归一化后必须再次置 0
    （这些维在 reader 输出中恒为 raw 0，置 0 即精确重建，见
    inference_step / inference_action_step 的 _dropped_local_dims 处理）。
    """
    if cols is not None:
        idx = torch.tensor(
            [max(int(c), 0) for c in cols], dtype=torch.long, device=actions.device
        )
        out = actions.index_select(-1, idx)
        if any(int(c) < 0 for c in cols):
            dropped = torch.tensor(
                [int(c) < 0 for c in cols], dtype=torch.bool, device=actions.device
            )
            out = out.masked_fill(dropped.view(*([1] * (out.ndim - 1)), -1), 0.0)
        return out
    return actions[..., offset:offset + dim]


def _dropped_local_dims(cols: Optional[List[int]]) -> List[int]:
    """dim_map 中被摘除（-1）的本地维下标列表（修复 #7；无摘除/无 dim_map 返回空）。"""
    if cols is None:
        return []
    return [i for i, c in enumerate(cols) if int(c) < 0]


def _temporal_ema(x: torch.Tensor, alpha: float, dim: int = 1) -> torch.Tensor:
    """沿时间维的指数滑动平均（修复 #3b，Diffusion Policy 式推理时序平滑）：
    y[0]=x[0]；y[t]=α·x[t]+(1−α)·y[t−1]。α∈(0,1]，α>=1 原样返回。

    flow matching 的速度场误差逐帧近似白噪声，50 步 Euler 把它累积进轨迹
    （近静止数据下抖动可达 GT 的 30~80 倍）；EMA 以 O(T) 零训练成本压低
    逐帧高频分量（白噪声方差缩 α/(2−α)），代价是约 (1−α)/α 帧滞后。

    dim 默认 1（action [B,T,U]）。depth 解码后是 [B,C,T,H,W]，传 dim=2。
    """
    alpha = float(alpha)
    if dim != 1:
        return _temporal_ema(x.transpose(1, dim), alpha, dim=1).transpose(1, dim)
    if alpha >= 1.0 or x.shape[1] <= 1:
        return x
    ys = [x[:, 0]]
    prev = ys[0]
    for t in range(1, x.shape[1]):
        prev = alpha * x[:, t] + (1.0 - alpha) * prev
        ys.append(prev)
    return torch.stack(ys, dim=1)


@dataclass 
class MotusConfig:
    """Configuration for Motus."""
    # Video model settings
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"

    # VLM settings
    vlm_checkpoint_path: str = ""
    # VLM 文本改写（20260819 v2）：训练/推理的文本条件先由 VLM 按压缩改写
    # 指令（build_vlm_rewrite_message，纯文本、不输入首帧图；输出
    # Task/Action + Scene 若有；按数据集家族分模板，egoscale 禁复述标注元
    # 数据）贪心生成短文本，再走纯文本编码为 und token——und token 数由
    # vlm_rewrite_max_new_tokens 硬上限保证（分段文本改造后原始消息可超
    # uni 分支 rope freqs 的 1024 行上限，且改写是模型主动压缩、非机械
    # 截断）；False = 原始消息直接编码（旧行为）。需 unified_vlm_task_prompt
    # 开启（消息重建分支内构建改写指令）；开启时 vlm_field_dropout_prob
    # 不适用（编码文本是生成结果，指令侧 dropout 不影响内容）。
    vlm_text_rewrite: bool = False
    vlm_rewrite_max_new_tokens: int = 128
    # VLM 改写结果磁盘缓存目录（20260818_v2）：非空时 UndBranch._rewrite_prompts
    # 按 sha256(VLM 模型名 + 消息文本) 查/写分片文件缓存（utils/
    # vlm_rewrite_cache.py；多 rank 并发安全、原子写），命中则跳过 generate；
    # 空 = 禁用（逐次 generate 旧行为）。模板/模型变更自动隔离命名空间，
    # 旧条目需人工清理目录。
    vlm_rewrite_cache_dir: str = ""
    
    # video settings
    video_height: int = 480             # Input video height
    video_width: int = 640              # Input video width

    action_state_dim: int = 26
    action_dim: int = 26
    video_action_freq_ratio: int = 6    # Video:Action frequency ratio
    action_t_downsample: int = 24        # Action 时序下采样因子（20260814 起生产配置 ratio=6/d=6：每 token 6 步、无裁剪、state_unembedding 移除）
    num_video_frames: int = 32           # Number of video frames to predict

    # Flow matching σ 调度的 shift（Wan2.2 TI2V-5B 官方为 5.0）。
    # 训练的时间标签（shift 后 σ×1000）与推理的 σ 网格共用同一 shift，
    # 保证训练/推理两侧喂给 time_embedding 的标签分布一致。
    flow_shift: float = 5.0
    # D1（20260811 对齐 MotusBrain 论文，per-modality timeshift：video=6 /
    # action=1）：action 侧训练 σ 采样（independent_modality_noise=True 路径）
    # 与推理 action σ 网格的独立 timeshift；None = 跟随 flow_shift（旧行为）。
    # 注意：independent_modality_noise=False 的共享噪声路径训练不使用它
    # （与推理网格分布不一致，__init__ 会 warning）。
    action_flow_shift: Optional[float] = None

    video_encoder_layer_num:int = 5
    video_decoder_layer_num:int = 5
    action_encoder_layer_num:int = 5
    action_decoder_layer_num:int = 5
    und_layer_num:int = 6

    # depth 第三生成流（Rynn 式 depth-as-video）：True 时 VideoBranch 建 depth
    # 专属 patch_embed/head + 零初始化 stream_embed（共享骨架），uni 走单向
    # 可见性（video/und 不读 depth，depth 读 video/action）。False = 与原模型
    # 逐字节一致。
    enable_depth: bool = False
    # depth flow-matching loss 权重（与 video 同 MSE/FM 目标，只在非首帧 latent
    # 上统计，按 depth_present 逐样本 mask）。
    depth_loss_weight: float = 1.0
    # L = mean(err) + λ · mean(err on motion). Off by default (V6.1).
    # V7 开过：运动区 GT 最脏，越加权越学破洞。V8 不要开。
    depth_motion_loss_enable: bool = False
    depth_motion_lambda: float = 0.3
    depth_motion_frac_max: float = 0.12
    # 边/梯度项（V8）：在 flow-matching 反推的 x0 上对齐空间梯度，
    # 权重是 GT 空间边（可要求跨帧持续）。不是运动区加权。
    depth_edge_loss_enable: bool = False
    depth_edge_lambda: float = 0.5
    depth_edge_persist: bool = True
    # Ego only: align frames with head/pose before the 3 gates. Agi ignores this.
    depth_motion_ego_pose: bool = False
    # Only upweight samples that actually got a pose-aligned mask (i.e. Ego).
    depth_motion_require_pose: bool = False
    depth_z_near: float = 0.15
    depth_z_far: float = 4.0
    depth_mode: str = "log"
    # 阶段开关：True 时 uni 里 action 读 depth（阶段 B）；False（默认，阶段 A）
    # 时 action 不读 depth，depth 只单向读 video/action，RGB 先验零污染。
    depth_read_by_action: bool = False
    # True（默认）时 uni 里 action 读 video。False 仅用于推理消融：action 不读 RGB。
    video_read_by_action: bool = True
    # True 时 uni 里 video 读 depth（RGB↔depth 双向）。False（默认）护 RGB 先验。
    # 打开后务必保留 modality_dropout_depth>0，否则 RGB 会绑死在 depth 上。
    depth_read_by_video: bool = False
    # True 时 uni cross-attn 里 und 读 depth（und↔depth 双向）。depth 读 und
    # 本来就开着。False（默认）堵住 depth→und→video 间接泄漏。
    depth_read_by_und: bool = False
    # True：depth 读原 uni 的 video K/V 时 detach（V6.1，护 RGB）。
    # action 读 depth、depth 读 action 仍不 detach。False = V6 旧行为。
    depth_detach_video_kv: bool = False
    # depth 与 video 共用同一份 flow-matching 噪声（RynnWorld-4D 做法：
    # noise_depth = noise_video.clone()）。两条流同 VAE/同 shape/同 σ，共用噪声
    # 使去噪轨迹对齐；False（默认）为独立采样，保持旧行为逐字节一致。
    depth_condition_mode: str = "first_frame"  # or rgb_only: all depth frames are targets
    depth_share_video_noise: bool = False
    # depth 专属 uni 容量（X-WAM 式克隆）：uni 最后 N 层 deepcopy 一份只给 depth
    # 用，video/action/und 仍走原层；depth 在克隆层读 video/action/und 的 K/V 复用
    # 原层投影并 detach，故 depth loss 在克隆区完全碰不到共享 RGB 权重。副本是
    # 精确拷贝 ⇒ 副本未训练时前向与 N=0 逐位等价，热启动无回归。每层 163.7M 参数
    # （Wan2.2-TI2V-5B，dim 3072 / ffn 14336）。0 = 全共享（旧行为）。
    depth_uni_clone_layers: int = 0
    
    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    # 文本条件 dropout 概率（评审 #6）：**20260818_v2 起弃用、代码不再使用**
    # ——逐样本 condition dropout 已并入 modality_dropout_prompt（batch 级、
    # 跨 rank 一致、空文本无条件分布），无条件分支训练由模态 dropout 单一
    # 机制承担。字段保留仅为旧 yaml 兼容（读到非 0 值会在 __init__ warning）。
    condition_dropout_prob: float = 0.0
    # VLM 字段 dropout 概率：rewrite 开启（vlm_text_rewrite=true）时在改写
    # **输出**文本上按字段行（Task/Action/Scene）逐样本伯努利 drop（整行
    # 移除、全中保底留 1，仅训练态；und_branch.precompute_vlm_features，
    # 20260818_v2 起——缓存之后作用、不产生新 cache key）；rewrite 关闭时
    # 为输入侧旧语义（消息重建时从四字段格式指令整字段移除，见
    # utils.vlm_utils.build_vlm_task_message）。0 = 关闭，标准取值 ~0.1。
    vlm_field_dropout_prob: float = 0.0
    robot_configs: Optional[Dict[str, int]] = None
    # 统一 action 空间预留容量（band 注册表见 YAML dataset.unified_bands）
    unified_action_dim: int = 64
    # {name: {"offset": int, "token_index": int, "dim_map": list[int] | None,
    #          "rot_groups": list[dict] | None}}
    # rot_groups（B1/B2，relative_eef_rot6 时用）：[{"local": [本地维..],
    # "repr": rotvec|quat|rot3, "cols": [6 个统一列], "quat_order": 可选}]——
    # 本地旋转维组转 6D 写入 cols；这些本地维在 dim_map 中必须为 -1。
    robot_layout: Optional[Dict[str, Dict[str, Any]]] = None
    # 语义槽位注册表（YAML dataset.unified_bands 透传）：{band: [start, end)}，
    # 供模型边界 band 语义推导（B3 归一化维分类 / B1 eef pos 相对化）。
    unified_bands: Optional[Dict[str, Any]] = None
    # B3（20260811 对齐 MotusBrain 论文）：action/state 归一化策略——
    # "zscore"（旧：全维 per-robot z-score，models/action_stats.py 的
    # normalize_batch）| "gripper_hand_minmax"（仅夹爪/灵巧手维归一化
    # [-1,1]，其余维保持物理量纲，见 action_stats.GripperHandNormPolicy）。
    action_norm_policy: str = "zscore"
    # B1/B2（20260811 对齐 MotusBrain 论文）：动作目标表示——
    # "absolute_native"（旧：绝对坐标 + 各本体原生旋转表示）|
    # "relative_eef_rot6"（EEF 位姿维相对条件帧 state：位置直减、旋转
    # R_s⁻¹R_i、统一 6D 旋转目标；仅作用 eef band，关节空间本体不受影响；
    # 需 robot_layout 条目配 rot_groups，旋转维组转到 *_rot6 band）。
    action_target_repr: str = "absolute_native"
    freeze_action_branch: bool = False
    # depth 分支冻结：True 时冻结 depth 专属 I/O 参数（depth_patch_embedding /
    # depth_head / depth_stream_embed），用于"只训基座、depth 不动"的场景，与
    # freeze_action_branch 对称。depth warmup（只训 depth I/O、冻结共享基座）
    # 由 freeze_for_depth_warmup() 单独提供。
    freeze_depth_branch: bool = False
    # 模态 dropout（评审 #4）：video/action/prompt 三模态各自独立概率（batch
    # 级、训练态、每 update step 重采、跨 rank 一致）；video 与 action 同时被
    # drop 时随机保底保留一个可训练模态。推理侧对缺失模态（全 0 输入）不输入
    # 网络、不生成该模态。20260818_v2 起逐样本 condition dropout 并入
    # modality_dropout_prompt（无条件分支训练的单一机制）。
    modality_dropout_video: float = 0.0
    modality_dropout_action: float = 0.0
    modality_dropout_prompt: float = 0.0
    # depth 第三生成流的模态 dropout（独立伯努利，不参与 video/action 的
    # "至少保留一个"保底——depth 是辅助流，被 drop 只是不喂 depth、其 loss
    # 置零）。enable_depth=False / 无 depth 数据时该开关无效。
    modality_dropout_depth: float = 0.0
    independent_modality_noise: bool = False   # True 时 video/action 各自独立采样 timestep/σ
    # 零 action loss 样本的 σ 强制最大噪声端点（20260816 引入；用户拍板本轮
    # 暂缓使用、机制保留备用）：True 时 action_loss_weights<=0 的样本（如
    # action_loss_weight=0 的源、训练侧 action-guard 置零样本）action σ 强制
    # 为 1.0（t 标签 1000），action encoder 输入纯噪声，真实动作不经 uni
    # cross-attn 泄漏为 video/und 条件；False（默认）= 不干预（仅 loss 置零
    # 的旧行为）。
    zero_action_loss_force_max_noise: bool = False
    gradient_checkpointing: bool = False
    checkpoint_video_branch: bool = True
    checkpoint_uni_branch: bool = True
    checkpoint_action_branch: bool = False

    # 块级因果 / AR（RSI，models/temporal_*.py）。默认 bidirectional = V14 原行为。
    # block_causal：每块 ar_latents_per_block 个 latent（=4·ratio 个原生动作步/latent），
    # 后块不影响前块；ar_cut_probs[i] = cut=i+1 的概率（cut>1 时块 1..cut-1 为
    # 干净 GT，AR-1 teacher forcing；None = 恒 cut=1）；world_mode_prob = 本步
    # 走动作条件世界模型（动作全为给定命令、不算 action loss）的概率。
    # 决策按 global_step 定种，跨 rank 一致。
    temporal_mode: str = "bidirectional"
    ar_latents_per_block: int = 1
    ar_cut_probs: Optional[List[float]] = None
    ar_target_reads_earlier_targets: bool = True
    # 四块 clean/noisy 打包 teacher forcing（替代 ar_cut_probs）：每步全部未来块
    # 计损失，第 k 块只读前 k-1 块的干净 RGB/动作；推理逐块生成，块数由
    # inference_step(num_blocks=...) 显式给（默认 1）。需 target_reads_earlier_targets=False。
    ar_packed_teacher_forcing: bool = False
    temporal_und_reads: str = "observed"
    world_mode_prob: float = 0.0
    # Current recipe: {joint: 0.6, a2v: 0.4}; legacy joint/idm/fdm also accepted.
    training_mode_probs: Optional[Dict[str, float]] = None
    # Context-only RGB error replay in BOTH Joint and A2V packed training.
    # The transient CPU buffer restarts empty on resume; inference is untouched.
    rgb_prefix_error_enable: bool = False
    rgb_prefix_error_prob: float = 0.25
    rgb_prefix_error_scale: float = 0.5
    rgb_prefix_error_max_rms_ratio: float = 0.1
    rgb_prefix_error_warmup_steps: int = 100
    rgb_prefix_error_capacity: int = 32
    rgb_prefix_error_max_mb: float = 64.0
    # Joint-only paired action context corruption. Explicit continuous unified
    # columns are required; grippers/discrete columns must not be listed.
    joint_action_prefix_error_enable: bool = False
    joint_action_prefix_error_dims: Optional[List[int]] = None
    joint_action_prefix_error_scale: float = 0.25
    joint_action_prefix_error_max_abs: float = 0.02  # normalized zscore units
    control_hz: float = 25.0
    # 视觉历史槽（单帧 VAE latent + 真实负时间偏移），仅 block_causal。
    history_enable: bool = False
    history_spatial_pool: int = 1
    history_logit_bias_init: float = -6.0
    # 只对有效稀疏槽位做 VAE 编码（无效槽位 latent 置零，HistoryEncoder 本来就遮掉）。
    history_encode_valid_only: bool = False
    # Opt-in RoboDojo objectives. No new inference parameters or sampling steps.
    capability_language_weight: float = 0.0
    capability_history_weight: float = 0.0
    capability_a2v_weight: float = 0.0
    capability_margin: float = 0.01
    capability_aux_interval: int = 4
    capability_min_sigma: float = 0.5
    # 预测误差条件的恢复（RecoveryEncoder → und 前缀 token），仅 block_causal。
    recovery_condition_enable: bool = False
    recovery_feature_dim: int = 0
    recovery_num_tokens: int = 4

    # Timing checkpoint detail for model internals (flows into the monitor's
    # step records as model_* keys): 1 = module level (branch encode/decode),
    # 2 = + per-block checkpoints (adds CUDA syncs per block; debug use).
    timing_level: int = 1

    # Control whether to load pretrained WAN/VLM backbones.
    # None = default behavior (load), False = skip loading (init from config only,
    # weights are then restored from the training checkpoint — train.py 在
    # resume/finetune 热启动时自动置 False，见 main() 开关设置处注释)
    load_pretrained_backbones: Optional[bool] = None

    # Debug flag: build tiny randomly-initialized backbone stand-ins
    # (train_debug/micro_backbones.py) instead of loading the real pretrained
    # VLM/Wan/VAE. Used by train_debug/micro_train.py only.
    micro_train: bool = False
    # Optional overrides for the tiny Wan config (only used when micro_train=True).
    micro_wan_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size = num_video_frames × video_action_freq_ratio
        # （video_stride 布局契约：动作连续 N·ratio 步，0807 起）。旧公式
        # (N+1)*ratio-1=293 是 consecutive 布局残留，且 293 % t_downsample(6)
        # != 0 会触发推理路径 T % t_downsample == 0 断言（20260815 修复）。
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio


def _spatial_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B,C,T,H,W] → (dy [B,C,T,H-1,W], dx [B,C,T,H,W-1])."""
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def depth_edge_loss_per_sample(
    x0_hat: torch.Tensor,
    x0_gt: torch.Tensor,
    persist: bool = True,
) -> torch.Tensor:
    """空间梯度对齐，逐样本 [B]。和评测 ``grad_corr`` / ``edge_absrel`` 同方向。

    在 flow-matching 反推的 x0 上算 ∇hat vs ∇gt。权是 GT 空间边强度；
    ``persist=True`` 时还要求和邻帧同时有边——DA3 闪一下的破洞过不了，
    墙/桌/手的稳定轮廓留下。不是 V7 的运动区 mask。
    """
    hat_y, hat_x = _spatial_grad(x0_hat)
    gt_y, gt_x = _spatial_grad(x0_gt)
    err = (hat_y[..., :, :-1] - gt_y[..., :, :-1]).pow(2) + (
        hat_x[..., :-1, :] - gt_x[..., :-1, :]
    ).pow(2)
    mag = (gt_y[..., :, :-1].detach().pow(2) + gt_x[..., :-1, :].detach().pow(2)).sqrt()
    w = mag.mean(dim=1, keepdim=True)
    if persist and w.size(2) >= 2:
        prev = torch.cat([w[:, :, :1], w[:, :, :-1]], dim=2)
        w = torch.minimum(w, prev)
    denom = w.reshape(w.size(0), -1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1, 1)
    w = (w / denom).clamp(max=4.0)
    return (err * w).mean(dim=(1, 2, 3, 4))


class UBT_WM(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM, VLM, and Action modules.
    """

    @property
    def device(self) -> torch.device:
        """当前计算设备（运行时按 world_model 实际参数设备求值）。

        不能在 __init__ 里一次性捕获：Accelerator 在模型创建之后才把参数
        搬到 GPU，且 ``next(world_model.parameters())`` 依赖子模块注册顺序
        —— 690dd8e 起 world_model 首个子模块是 RobotTokenEmbedding（CPU
        参数），初始化时捕获会得到 cpu，导致 sigma/t 落在 CPU 与 cuda
        latent 混算（stage2_20260807 首训练步即崩）。每次调用取当前参数
        设备，prepare/加载 checkpoint 后自动正确。
        """
        return next(self.world_model.parameters()).device

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config
        depth_condition_mode(config)
        validate_depth_losses(config)
        TT.validate_temporal_config(config)
        self._temporal_sample_idx = 0

        # 模态 dropout 采样序号（跨 rank 一致性计数器）：各 rank 训练循环步数
        # 严格一致，第 N 次采样用 (固定基种子 + N) 定种子的专用 generator，
        # 保证所有 rank 同一 update step 做出相同的 drop 决策。不能用全局
        # RNG——各 rank 数据/噪声消耗不同早已分叉，独立决策会让被 drop 模态
        # 的参数在部分 rank 无梯度，ZeRO-1 梯度桶 allreduce 跨 rank 尺寸不
        # 一致而静默死锁（stage2-20260807 step0 NCCL watchdog 根因）。
        self._modality_dropout_sample_idx = 0

        # condition_dropout_prob 已于 20260818_v2 弃用（并入 modality_dropout_
        # prompt，见 MotusConfig 字段注释）；旧 yaml 的非 0 值不再生效。
        if float(getattr(config, "condition_dropout_prob", 0.0)) > 0.0:
            logger.warning(
                "condition_dropout_prob=%s 已弃用且不再生效（20260818_v2 起并入 "
                "modality_dropout_prompt）；请从配置中移除该键。",
                getattr(config, "condition_dropout_prob", None),
            )

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        if config.micro_train:
            # Debug mode: tiny randomly-initialized stand-ins for VLM/Wan/VAE
            # with identical interfaces; no pretrained weights are loaded.
            logger.info(
                "micro_train=True: building tiny randomly-initialized backbones "
                "(debug mode); no pretrained weights loaded"
            )
            from thinker_wm.train_debug.micro_backbones import build_micro_backbones

            vlm_preprocess, vlm_model, wan_video_model = build_micro_backbones(
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=self.dtype,
                wan_config=config.micro_wan_config,
            )
        elif config.load_pretrained_backbones is False:
            # resume/finetune 热启动：只建架构（随机初始化），不读预训练权重——
            # 全部权重（含冻结的 VLM/VAE）随后由训练 checkpoint 全量恢复
            # （ZeRO-1 ckpt 保存完整 module.state_dict()，exclude_frozen_parameters
            # 未设置；resume 为 strict 加载），预训练 backbone 加载在此纯属冗余。
            # 注意：本分支依赖"ckpt 含全部冻结权重"这一前提——若未来切换
            # ZeRO-3 或设置 exclude_frozen_parameters=True，需重新评估。
            logger.info(
                "load_pretrained_backbones=False: building VLM/Wan architecture "
                "only (random init); all weights will be restored from the "
                "training checkpoint"
            )
            logger.info("Loading VLM Preprocessor...")
            vlm_preprocess = AutoProcessor.from_pretrained(config.vlm_checkpoint_path)

            logger.info("Building Wan Model from config (no pretrained weights)...")
            wan_video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                precision=config.video_precision
            )

            logger.info("Building VLM from config (no pretrained weights)...")
            vlm_config = AutoConfig.from_pretrained(
                config.vlm_checkpoint_path, trust_remote_code=True
            )
            # 直接在 GPU 上构建：随机初始化走 GPU kernel（2B 参数 CPU 初始化需
            # 数十秒），并省掉一次 ~4GB H2D 拷贝（默认分支 device_map="cuda" 同
            # 为 GPU 常驻）；权重随后由训练 checkpoint 全量覆盖。
            with torch.device("cuda"):
                try:
                    vlm_model = Qwen3VLForConditionalGeneration._from_config(
                        vlm_config, dtype=self.dtype
                    )
                except TypeError:  # 兼容无 dtype kwarg 的 transformers 版本
                    vlm_model = Qwen3VLForConditionalGeneration(vlm_config).to(dtype=self.dtype)
        else:
            logger.info("Loading VLM Preprocessor...")
            vlm_preprocess = AutoProcessor.from_pretrained(config.vlm_checkpoint_path)

            logger.info("Loading Wan Model...")
            wan_video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )

            logger.info("Loading VLM...")
            vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )

        wan_model = wan_video_model.wan_model

        logger.info("Freezing VLM...")
        for param in vlm_model.parameters():
            param.requires_grad = False


        # Move models to device
        self.vae = wan_video_model.vae

        self.world_model = UniWorldModel(
            vlm_preprocess, vlm_model, wan_model,
            action_dim = config.action_dim,
            video_encoder_layer_num = config.video_encoder_layer_num,
            video_decoder_layer_num = config.video_decoder_layer_num,
            action_encoder_layer_num = config.action_encoder_layer_num,
            action_decoder_layer_num = config.action_decoder_layer_num,
            und_layer_num = config.und_layer_num,
            action_t_downsample = config.action_t_downsample,
            video_action_freq_ratio = config.video_action_freq_ratio,
            robot_configs = config.robot_configs,
            unified_action_dim = config.unified_action_dim,
            robot_token_indices = {k: v["token_index"] for k, v in (config.robot_layout or {}).items()},
            gradient_checkpointing = config.gradient_checkpointing,
            checkpoint_video_branch = config.checkpoint_video_branch,
            checkpoint_uni_branch = config.checkpoint_uni_branch,
            vlm_text_rewrite = config.vlm_text_rewrite,
            vlm_rewrite_max_new_tokens = config.vlm_rewrite_max_new_tokens,
            vlm_rewrite_cache_dir = getattr(config, "vlm_rewrite_cache_dir", ""),
            vlm_field_dropout_prob = config.vlm_field_dropout_prob,
            vlm_model_tag = os.path.basename(str(config.vlm_checkpoint_path).rstrip("/")),
            enable_depth = bool(getattr(config, "enable_depth", False)),
            depth_uni_clone_layers = int(getattr(config, "depth_uni_clone_layers", 0) or 0),
            enable_history = bool(getattr(config, "history_enable", False)),
            history_spatial_pool = int(getattr(config, "history_spatial_pool", 1)),
            history_logit_bias_init = float(getattr(config, "history_logit_bias_init", -6.0)),
            enable_recovery_condition = bool(getattr(config, "recovery_condition_enable", False)),
            recovery_feature_dim = int(getattr(config, "recovery_feature_dim", 0)),
            recovery_num_tokens = int(getattr(config, "recovery_num_tokens", 4)),
        ).to(dtype=self.dtype)
        logger.info(
            "Gradient checkpointing config: enabled=%s video_branch=%s uni_branch=%s action_branch=%s",
            config.gradient_checkpointing,
            config.checkpoint_video_branch,
            config.checkpoint_uni_branch,
            config.checkpoint_action_branch,
        )

        self.ego4d_state_prompt = nn.Parameter(
            torch.empty(config.action_state_dim, device=self.device, dtype=self.dtype)
        )
        nn.init.normal_(self.ego4d_state_prompt, mean=0.0, std=0.02)
        logger.info(
            "Initialized Ego4D state prompt: "
            f"state={tuple(self.ego4d_state_prompt.shape)}"
        )
        # VAE is only ever used under torch.no_grad() (video encode/decode);
        # freeze it in every training mode so it stays out of the optimizer
        # (weight decay would otherwise erode grad-less weights).
        frozen_no_grad = {}
        if isinstance(self.vae, nn.Module):
            frozen_no_grad["vae"] = self._freeze_module(self.vae)
        logger.info(
            "Frozen no-grad modules (all modes): %s",
            {
                name: {"tensors": int(tensors), "params_m": float(params) / 1e6}
                for name, (tensors, params) in frozen_no_grad.items()
            },
        )
        if config.freeze_action_branch:
            self.freeze_action_branch()
        if getattr(config, "enable_depth", False) and getattr(config, "freeze_depth_branch", False):
            self.freeze_depth_branch()

        self.fm_train_scheduler = FlowMatchScheduler(
            shift=config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000
        )
        self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        # D1（20260811 对齐 MotusBrain，per-modality timeshift video=6/action=1）：
        # action_flow_shift 与 flow_shift 不同时为 action 建第二个 scheduler
        # （训练侧 independent_modality_noise=True 路径用它采样；推理 action
        # σ 网格恒按 action_flow_shift）。共享噪声路径（independent=False）
        # 训练不用它——若同时设置 action_flow_shift，训练/推理 action σ 分布
        # 将不一致，warning 提示。
        action_flow_shift = getattr(config, "action_flow_shift", None)
        self.fm_train_scheduler_action = None
        if action_flow_shift is not None:
            if float(action_flow_shift) != float(config.flow_shift):
                self.fm_train_scheduler_action = FlowMatchScheduler(
                    shift=float(action_flow_shift),
                    sigma_min=0.0,
                    extra_one_step=True,
                    num_train_timesteps=1000
                )
                self.fm_train_scheduler_action.set_timesteps(
                    num_inference_steps=1000, training=True)
            if not bool(getattr(config, "independent_modality_noise", False)):
                logger.warning(
                    "action_flow_shift=%s 已设置但 independent_modality_noise=False："
                    "训练共享噪声路径 video/action 同一 σ（不用 action scheduler），"
                    "而推理 action σ 网格仍按 action_flow_shift——训练/推理 action "
                    "σ 分布将不一致，请确认配置意图",
                    action_flow_shift,
                )
        # 推理 action σ 网格的 shift（无独立 action scheduler 时回退
        # flow_shift，保持训练/推理一致）。
        self._action_flow_shift = (
            float(action_flow_shift)
            if self.fm_train_scheduler_action is not None
            else float(config.flow_shift)
        )

        # B3（20260811 对齐 MotusBrain）：归一化策略——gripper_hand_minmax
        # 用 robot_layout + unified_bands 推导每机器人夹爪/灵巧手维（见
        # action_stats.build_gripper_hand_policy）；zscore 保持旧
        # normalize_batch 全维 z-score 路径。
        self._norm_policy = None
        if str(getattr(config, "action_norm_policy", "zscore")) == "gripper_hand_minmax":
            self._norm_policy = build_gripper_hand_policy(
                config.robot_layout or {}, config.unified_bands or {})
        # B1/B2（20260811 对齐 MotusBrain）：relative_eef_rot6 时构建
        # per-robot eef pos 维 / rot group 索引结构（_build_repr_specs）。
        self._repr_specs = None
        if str(getattr(config, "action_target_repr", "absolute_native")) == "relative_eef_rot6":
            self._repr_specs = self._build_repr_specs()

        # Log parameter counts
        self.log_parameter_counts()

        # fixme:
        self._video_action_freq_ratio = config.video_action_freq_ratio

    def freeze_action_branch(self) -> None:
        """Freeze action-only modules for video/text-only finetuning."""

        frozen_params = 0
        frozen_tensors = 0
        for param in self.world_model.action_branch.parameters():
            if param.requires_grad:
                frozen_params += param.numel()
                frozen_tensors += 1
            param.requires_grad = False

        if self.ego4d_state_prompt.requires_grad:
            frozen_params += self.ego4d_state_prompt.numel()
            frozen_tensors += 1
        self.ego4d_state_prompt.requires_grad = False

        logger.info(
            "Frozen action branch for video/text-only training: tensors=%d params=%.3fM",
            frozen_tensors,
            frozen_params / 1e6,
        )

    def _depth_io_modules(self) -> List[nn.Module]:
        """depth 专属 I/O 模块（enable_depth 时才存在）：depth_patch_embedding /
        depth_head 两个模块 + depth_stream_embed 一个 Parameter。共享骨架
        （patch/head 外的 transformer blocks、und、action）不在此列。"""
        vb = self.world_model.video_branch
        mods: List[nn.Module] = []
        if getattr(vb, "depth_patch_embedding", None) is not None:
            mods.append(vb.depth_patch_embedding)
        if getattr(vb, "depth_head", None) is not None:
            mods.append(vb.depth_head)
        return mods

    def depth_io_parameters(self) -> List[nn.Parameter]:
        """depth 专属 I/O 的全部参数（depth_patch_embedding / depth_head 的参数
        + depth_stream_embed）。供优化器把它们分到独立参数组给更高 LR：depth
        适配器是新参数（虽从 RGB 权重 warm-start），需要的步长比已训好的共享
        骨架大一到两个数量级。enable_depth=False 时返回空列表。"""
        params = [p for m in self._depth_io_modules() for p in m.parameters()]
        stream_embed = getattr(self.world_model.video_branch, "depth_stream_embed", None)
        if stream_embed is not None:
            params.append(stream_embed)
        return params

    def depth_uni_clone_parameters(self) -> List[nn.Parameter]:
        """depth 专属 uni 克隆层（depth_uni_clone_layers > 0 时存在）的全部参数。
        供优化器分到独立参数组：它们从已训好的 RGB uni 层 deepcopy 而来，既不该用
        depth I/O 那档新参数的高 LR 冲掉预训练结构，也不该跟共享骨架一样保守。
        未启用克隆时返回空列表。"""
        blocks = getattr(self.world_model.uni_branch, "depth_uni_blocks", None)
        if blocks is None:
            return []
        return [p for b in blocks for p in b.parameters()]

    def freeze_depth_branch(self) -> None:
        """冻结 depth 专属 I/O 参数（与 freeze_action_branch 对称）：只训基座、
        depth 不动的场景。enable_depth=False 时无 depth 参数，空操作。"""
        frozen_params = 0
        frozen_tensors = 0
        vb = self.world_model.video_branch
        params = list(
            p for m in self._depth_io_modules() for p in m.parameters()
        )
        if getattr(vb, "depth_stream_embed", None) is not None:
            params.append(vb.depth_stream_embed)
        for param in params:
            if param.requires_grad:
                frozen_params += param.numel()
                frozen_tensors += 1
            param.requires_grad = False
        logger.info(
            "Frozen depth branch (depth I/O only): tensors=%d params=%.3fM",
            frozen_tensors,
            frozen_params / 1e6,
        )

    def freeze_for_depth_warmup(self) -> None:
        """depth warmup：冻结除 depth 专属 I/O 外的一切（共享 transformer 骨架、
        video/action/und 分支），只训 depth_patch_embedding / depth_head /
        depth_stream_embed。用于把 depth 适配器"焐热"而不动已训好的 RGB/action
        先验；阶段 A（depth_read_by_action=False）下本就零污染，冻结基座是额外
        的安全 + 省算力。warmup 结束后调用方按需重新打开基座 requires_grad。"""
        depth_param_ids = {
            id(p) for m in self._depth_io_modules() for p in m.parameters()
        }
        vb = self.world_model.video_branch
        if getattr(vb, "depth_stream_embed", None) is not None:
            depth_param_ids.add(id(vb.depth_stream_embed))
        if not depth_param_ids:
            logger.warning(
                "freeze_for_depth_warmup: 无 depth 专属参数（enable_depth=False?），"
                "跳过——不冻结任何参数以免误伤基座训练。"
            )
            return
        frozen_params = 0
        frozen_tensors = 0
        trainable_params = 0
        for param in self.parameters():
            if id(param) in depth_param_ids:
                param.requires_grad = True
                trainable_params += param.numel()
                continue
            if param.requires_grad:
                frozen_params += param.numel()
                frozen_tensors += 1
            param.requires_grad = False
        logger.info(
            "Depth warmup: 冻结基座 tensors=%d params=%.3fM，仅训 depth I/O params=%.3fM",
            frozen_tensors,
            frozen_params / 1e6,
            trainable_params / 1e6,
        )

    @staticmethod
    def _freeze_module(module: nn.Module) -> Tuple[int, int]:
        frozen_params = 0
        frozen_tensors = 0
        for param in module.parameters():
            if param.requires_grad:
                frozen_params += param.numel()
                frozen_tensors += 1
            param.requires_grad = False
        return frozen_tensors, frozen_params

    def _apply_ego4d_action_state_prompt(
        self,
        state: Optional[torch.Tensor],
        actions: torch.Tensor,
        source_datasets: Optional[List[str]] = None,
        state_action_policies: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        if source_datasets is None and state_action_policies is None:
            return state, actions

        if state_action_policies is not None:
            prompt_mask_values = [
                str(policy).lower() == "learned_prompt"
                for policy in state_action_policies
            ]
        else:
            prompt_mask_values = [
                str(source_dataset).lower() == "ego4d"
                for source_dataset in source_datasets
            ]

        prompt_mask = torch.tensor(prompt_mask_values, device=actions.device, dtype=torch.bool)
        if not bool(prompt_mask.any()):
            return state, actions

        if state is not None:
            if state.shape[1:] != self.ego4d_state_prompt.shape:
                raise ValueError(
                    "Ego4D state prompt shape does not match batch state shape: "
                    f"prompt={tuple(self.ego4d_state_prompt.shape)}, "
                    f"state={tuple(state.shape[1:])}"
                )
            prompt_state = self.ego4d_state_prompt.to(device=state.device, dtype=state.dtype)
            state = torch.where(
                prompt_mask.view(-1, 1),
                prompt_state.unsqueeze(0).expand(state.shape[0], -1),
                state,
            )

        return state, actions

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")

    # 评审 #10 已删除的模块（T5 与 action 分支的 T5 text_embedding 兼容路径）。
    # 旧 checkpoint 的 state_dict 仍带这些键；加载前统一丢弃，使 DeepSpeed
    # engine 的 strict 加载、评估 strict=True 与 finetune 部分加载都能直接
    # 读取旧 ckpt 的模型权重。
    _LEGACY_STATE_DICT_PREFIXES = (
        "world_model.t5_emb.",
        "world_model.action_branch.encoder.text_embedding.",
        "world_model.action_branch.decoder.text_embedding.",
        # 统一 action 空间（多机器人）：per-robot projector 分支已移除，旧 ckpt
        # 的 per-robot 权重不迁移（键过滤后缺失键属预期，新参数随机初始化起步）。
        "world_model.action_branch.encoder.state_projectors.",
        "world_model.action_branch.encoder.action_projectors.",
        "world_model.action_branch.decoder.action_unprojectors.",
        # 共享 projector 已整体移除（20260811，统一维 U 直通，无 Linear/MLP
        # 投影）；旧 ckpt 的共享 projector 权重（Linear 键 *.weight 与 MLP
        # 键 *.0.weight 等）不迁移，embedding/unembedding 权重经
        # _remap_unified_capacity_growth 沿 U 维零填充扩容后加载。
        "world_model.action_branch.encoder.state_projector.",
        "world_model.action_branch.encoder.action_projector.",
        "world_model.action_branch.decoder.action_unprojector.",
        # FPS 条件已整体移除（20260815 固定 5fps 轮，models/fps_condition.py
        # 删除）：旧 ckpt 的 fps_conditioner.* 键（0812_v2 初版逐层 heads.*
        # 与 20260814 全局单头 input_proj.*/head.*）全部不迁移。
        "world_model.fps_conditioner.",
        # action 时序下采样（20260814）：state_unembedding 已移除（state
        # token 仅作序列条件、不再补步），旧 ckpt 该键不迁移。
        "world_model.action_branch.decoder.state_unembedding.",
        # Ego4D 动作 prompt 从未使用，且长度跟 action_chunk_size 走（旧 ckpt
        # 是 [64, 14]）。整键删除，加载时丢弃，不再按时间维迁移。
        "ego4d_action_prompt",
        # 注意（V14，20260918）：encoder.state_embedding 不能列在旧键过滤项中；
        # 否则 V14 从自己的 checkpoint 续训时会静默丢掉已学好的
        # state_embedding 权重，且不报错。decoder.state_unembedding 仍是废弃
        # 键（V14 的 state token 在输出投影前切掉，不需要 unembedding），保留过滤。
    )

    def load_state_dict(self, state_dict, *args, **kwargs):
        filtered = {
            k: v for k, v in state_dict.items()
            if not k.startswith(self._LEGACY_STATE_DICT_PREFIXES)
        }
        dropped = len(state_dict) - len(filtered)
        if dropped:
            logger.info(
                "Dropped %d legacy checkpoint keys under removed-module "
                "prefixes %s", dropped, self._LEGACY_STATE_DICT_PREFIXES,
            )
        # U 维 remap 链（顺序敏感）：先 64→80 超容量零填充扩容（0807 前
        # ckpt），再 80→20 列切片（20260815 U=20 delta-EEF 轮），最后
        # t_downsample 1→d 时间核/输出维迁移（0807 系 ckpt）。
        filtered = self._remap_unified_capacity_growth(filtered)
        filtered = self._remap_unified_eef20(filtered)
        filtered = self._remap_action_t_downsample(filtered)
        return super().load_state_dict(filtered, *args, **kwargs)

    def _remap_action_t_downsample(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """action 时序下采样 remap（20260814，t_downsample 1→d）。

        新划分（无裁剪、每 token d 步、state_unembedding 移除）下两个键
        形状变化，旧 ckpt 按下述规则迁移：
        - encoder.action_embedding.weight [dim,U,1,1,1] → [dim,U,d,1,1]：
          d 个 tap 各填旧权重/d（= d 步均值池化，中性初始化）；
        - decoder.action_unembedding.{weight,bias} [U,*] → [d·U,*]：沿输出
          维平铺 d 份（开局每 token 解出 d 步常值轨迹，可学习分化）。
        仅 ckpt 与模型形状不匹配时触发；state_unembedding 键由 legacy
        前缀过滤丢弃。
        """
        model_state = self.state_dict()
        emb_key = "world_model.action_branch.encoder.action_embedding.weight"
        unw_key = "world_model.action_branch.decoder.action_unembedding.weight"
        unb_key = "world_model.action_branch.decoder.action_unembedding.bias"
        out = dict(state_dict)
        remapped = []
        for key in (emb_key, unw_key, unb_key):
            value = out.get(key)
            target = model_state.get(key)
            if value is None or target is None or value.shape == target.shape:
                continue
            if key == emb_key:
                # [dim,U,1,1,1] → [dim,U,d,1,1]
                if (value.dim() == 5 and target.dim() == 5
                        and value.shape[:2] == target.shape[:2]
                        and value.shape[3:] == target.shape[3:]
                        and value.shape[2] == 1):
                    d = target.shape[2]
                    out[key] = value.repeat(1, 1, d, 1, 1) / d
                    remapped.append(f"{key} (mean-pool x{d})")
            elif key == unw_key:
                # [U,dim] → [d·U,dim]
                if (value.dim() == 2 and target.dim() == 2
                        and target.shape[0] % value.shape[0] == 0
                        and target.shape[1] == value.shape[1]):
                    d = target.shape[0] // value.shape[0]
                    out[key] = value.repeat(d, 1)
                    remapped.append(f"{key} (tile x{d})")
            else:
                # [U] → [d·U]
                if (value.dim() == 1 and target.dim() == 1
                        and target.shape[0] % value.shape[0] == 0):
                    d = target.shape[0] // value.shape[0]
                    out[key] = value.repeat(d)
                    remapped.append(f"{key} (tile x{d})")
        if remapped:
            logger.info(
                "Remapped action t_downsample checkpoint keys: %s", remapped,
            )
        return out

    def _remap_unified_capacity_growth(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """统一 action 空间超容量增长 remap（03_config.md 协议，20260811）。

        旧 ckpt 的统一空间容量 U_old 小于当前 ``config.unified_action_dim``
        时，把依赖 U 的张量沿 U 维零填充扩容：action/state embedding 的
        Conv3d 输入通道（旧通道原样拷贝、新通道零初始化）、unembedding
        输出维权重与 bias（旧行拷贝、新行零初始化）。
        通用触发规则（不硬编码版本/键名）：键存在于当前模型、恰好一维
        不匹配、其余维全等、ckpt 该维 < 模型该维；不满足的不动（交由
        调用方按 shape-mismatch skip 或 strict 报错处理）。
        """
        model_state = self.state_dict()
        # 20260815 U=20 轮：action 边界三键（embedding/unembedding）的 U 维
        # 迁移由 _remap_unified_eef20 全权处理（含 64→80 补零 + 80→20 切片），
        # 此处跳过——否则通用"恰好一维更小"规则会把 unembedding [80,dim] →
        # [d·20,dim] 误判为扩容而零填充到最终形状，抢走切片 remap 的输入。
        skip_for_eef20 = (
            int(getattr(self.config, "unified_action_dim", 0))
            == len(self._EEF20_REMAP_COLUMNS)
        )
        grown: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if skip_for_eef20 and key in self._EEF20_REMAP_KEYS:
                continue
            target = model_state.get(key)
            if (
                target is None
                or tuple(target.shape) == tuple(value.shape)
                or value.dim() == 0
                or target.dim() != value.dim()
            ):
                continue
            diff = [d for d in range(value.dim()) if value.shape[d] != target.shape[d]]
            if len(diff) != 1 or value.shape[diff[0]] > target.shape[diff[0]]:
                continue
            d = diff[0]
            pad_shape = list(value.shape)
            pad_shape[d] = int(target.shape[d]) - int(value.shape[d])
            grown[key] = torch.cat([value, value.new_zeros(pad_shape)], dim=d)
        if grown:
            logger.info(
                "U 扩容 remap：%d 个张量沿 U 维零填充扩容（旧列原样拷贝、新列"
                "零初始化）: %s",
                len(grown),
                {k: f"{tuple(state_dict[k].shape)}->{tuple(v.shape)}" for k, v in grown.items()},
            )
            state_dict = dict(state_dict)
            state_dict.update(grown)
        return state_dict

    # 统一 action 空间 80→20 紧凑布局列切片（20260815，U=20 delta-EEF 轮）：
    # 旧 80 列中保留双臂 pos/rot6/gripper 共 20 列——左 pos [14,17) → 新
    # [0,3)、左 rot6 [64,70) → 新 [3,9)、左 gripper 20 → 新 9、右 pos
    # [21,24) → 新 [10,13)、右 rot6 [70,76) → 新 [13,19)、右 gripper 27 →
    # 新 19；joint_*/hand_*/quat/rot3/base_twist/reserve 列全部弃用。
    # 语义一致性：新旧 rot6/pos 同为相对首帧 delta、gripper 同为 minmax
    # [-1,1]，切片权重直接可用；中间 block 权重形状不变直接加载。
    _EEF20_REMAP_OLD_U = 80
    # 20260811 前的更老 ckpt 容量 U=64：先零填充补到 80（等价
    # _remap_unified_capacity_growth 的扩容语义）再切片。
    _EEF20_REMAP_OLDER_U = 64
    _EEF20_REMAP_COLUMNS = (
        14, 15, 16, 64, 65, 66, 67, 68, 69, 20,
        21, 22, 23, 70, 71, 72, 73, 74, 75, 27,
    )
    _EEF20_REMAP_KEYS = (
        "world_model.action_branch.encoder.action_embedding.weight",
        "world_model.action_branch.decoder.action_unembedding.weight",
        "world_model.action_branch.decoder.action_unembedding.bias",
    )

    def _remap_unified_eef20(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """U=80（或更老的 U=64）→ U=20 紧凑布局列切片 remap（列映射见类
        常量注释）。

        只对恰好 U=80/64 的旧 ckpt 键生效（新 U=20 ckpt 原样通过）：
        - encoder.action_embedding.weight [dim,U,kt,kh,kw] → 输入通道维
          （dim 1）按列映射 gather（bias [dim] 不含 U 维，不动）；
        - decoder.action_unembedding.weight [d·U,dim] / bias [d·U] → 按
          d 个连续 U 块 reshape 后沿 U 维 gather，再 reshape 回 [d·20,*]。
        U=64 的键先沿 U 维零填充到 80 再切片（与 capacity growth 同语义；
        本函数接管这些键后，growth 在 U=20 布局下跳过它们，见
        _remap_unified_capacity_growth）。时间核维 kt / 输出维 d 允许与
        模型不同：0807 系 ckpt（kt=1、d_old=1）切片后由
        _remap_action_t_downsample 接力（1→d 均值池化/平铺），故串联顺序
        必须 growth → 本切片 → t_downsample（load_state_dict 注释）。
        """
        old_u = self._EEF20_REMAP_OLD_U
        older_u = self._EEF20_REMAP_OLDER_U
        new_u = len(self._EEF20_REMAP_COLUMNS)
        emb_key, unw_key, unb_key = self._EEF20_REMAP_KEYS
        model_state = self.state_dict()
        out = dict(state_dict)
        remapped = []
        for key in self._EEF20_REMAP_KEYS:
            value = out.get(key)
            target = model_state.get(key)
            if value is None or target is None or value.shape == target.shape:
                continue
            cols = torch.tensor(self._EEF20_REMAP_COLUMNS, dtype=torch.long,
                                device=value.device)
            if key == emb_key:
                # [dim,U,kt,kh,kw] → [dim,20,kt,kh,kw]（仅切 U 维；时间核维
                # kt 允许与模型不同——0807 系 ckpt kt=1，切片后由
                # _remap_action_t_downsample 接力 1→d）
                if (value.dim() == 5 and target.dim() == 5
                        and value.shape[1] in (old_u, older_u)
                        and target.shape[1] == new_u
                        and value.shape[0] == target.shape[0]
                        and value.shape[3:] == target.shape[3:]):
                    v = value
                    if v.shape[1] < old_u:
                        pad = v.new_zeros(
                            v.shape[0], old_u - v.shape[1], *v.shape[2:])
                        v = torch.cat([v, pad], dim=1)
                    out[key] = v.index_select(1, cols)
                    remapped.append(
                        f"{key} (U {value.shape[1]}->{new_u})")
            elif key == unw_key:
                # [d_old·U,dim] → [d_old·20,dim]：d_old 个连续 U 块 reshape
                # 后 gather。d_old 允许与模型 t_downsample 不同（0807 系
                # d_old=1 → 切出 [20,dim]，再由 _remap_action_t_downsample
                # 平铺到 [d·20,dim]）。
                if (value.dim() == 2 and target.dim() == 2
                        and value.shape[1] == target.shape[1]
                        and target.shape[0] % new_u == 0
                        and (value.shape[0] % old_u == 0
                             or value.shape[0] % older_u == 0)):
                    u_old = (
                        old_u if value.shape[0] % old_u == 0 else older_u)
                    d = value.shape[0] // u_old
                    v = value.view(d, u_old, value.shape[1])
                    if u_old < old_u:
                        v = torch.cat(
                            [v, v.new_zeros(d, old_u - u_old, v.shape[2])],
                            dim=1)
                    out[key] = (
                        v.index_select(1, cols)
                        .reshape(d * new_u, value.shape[1])
                    )
                    remapped.append(
                        f"{key} (d·U {value.shape[0]}->{d * new_u})")
            else:
                # [d_old·U] → [d_old·20]（同 unw 的 U/d_old 规则）
                if (value.dim() == 1 and target.dim() == 1
                        and target.shape[0] % new_u == 0
                        and (value.shape[0] % old_u == 0
                             or value.shape[0] % older_u == 0)):
                    u_old = (
                        old_u if value.shape[0] % old_u == 0 else older_u)
                    d = value.shape[0] // u_old
                    v = value.view(d, u_old)
                    if u_old < old_u:
                        v = torch.cat(
                            [v, v.new_zeros(d, old_u - u_old)], dim=1)
                    out[key] = (
                        v.index_select(1, cols)
                        .reshape(d * new_u)
                    )
                    remapped.append(
                        f"{key} (d·U {value.shape[0]}->{d * new_u})")
        if remapped:
            logger.info(
                "Remapped unified eef20 checkpoint keys (U=80/64 -> U=20 列切片): %s",
                remapped,
            )
        return out

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)
    
        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']  
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        
        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items() 
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str, *, action_interface_policy: str | None = None,
                              audit_path: str | None = None) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")
        
        # Handle directory path - try two possible locations
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            # Try two possible paths
            possible_paths = [
                checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
                checkpoint_path / "mp_rank_00_model_states.pt",                   
            ]
            
            checkpoint_file = None
            for p in possible_paths:
                if p.exists():
                    checkpoint_file = p
                    logger.info(f"Found checkpoint: {checkpoint_file}")
                    break
            
            if checkpoint_file is None:
                raise FileNotFoundError(
                    f"Checkpoint not found. Tried:\n"
                    f"  - {possible_paths[0]}\n"
                    f"  - {possible_paths[1]}"
                )
            path = str(checkpoint_file)
        else:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading pretrain weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)

        if action_interface_policy is not None:
            if action_interface_policy != "dexjoco_rgb":
                raise ValueError(f"unknown action interface migration: {action_interface_policy}")
            from thinker_wm.models.transfer import load_dexjoco_transfer
            load_dexjoco_transfer(self, state_dict, str(path), audit_path)
            return

        # U 扩容 remap（20260811）：旧 ckpt 统一空间容量小于当前
        # unified_action_dim 时，先把依赖 U 的张量（action/state embedding、
        # unembedding）零填充扩到当前形状，再走下方形状过滤（否则会被当
        # shape mismatch skip，新列虽也零初始化但旧列权重会丢失）。
        state_dict = self._remap_unified_capacity_growth(state_dict)
        state_dict = self._remap_depth_uni_clone_growth(state_dict)

        # Filter out parameters with shape mismatches
        model_state_dict = self.state_dict()
        filtered = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                model_shape = tuple(model_state_dict[k].shape)
                checkpoint_shape = tuple(v.shape)
                
                if model_shape != checkpoint_shape:
                    logger.warning(f"Shape mismatch for {k}: checkpoint {checkpoint_shape} vs model {model_shape}, skipping...")
                    skipped_keys.append(k)
                    continue
            
            filtered[k] = v
        
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights. Skipped (shape mismatch): {len(skipped_keys)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if skipped_keys:
            logger.info(f"Skipped keys due to shape mismatch: {skipped_keys}")
        if missing:
            logger.info(f"Missing keys: {sorted(missing)}")
        if unexpected:
            logger.info(f"Unexpected keys: {sorted(unexpected)}")
        # 热启动必须可验收（20260918）：只允许缺失本轮新增的参数，其余
        # missing/unexpected/shape mismatch 都说明 ckpt 与当前模型结构不符。
        # 旧行为只打日志，会让整支动作分支静默停在随机初始化上继续训练。
        self._assert_pretrain_load_acceptable(missing, unexpected, skipped_keys)
        self._reclone_depth_uni_blocks_if_absent(filtered)

    # 允许在从旧 ckpt 热启动时缺失的参数前缀。V14 新增 state token 的输入
    # 投影，V13 及更早的 ckpt 里没有这两个张量，构造期已零初始化。
    # RSI 的视觉历史编码器（零初始化）在 V15 及更早的 ckpt 里不存在。
    _PRETRAIN_ALLOWED_MISSING_PREFIXES = (
        "world_model.action_branch.encoder.state_embedding.",
        "world_model.history_encoder.",
    )

    def _assert_pretrain_load_acceptable(
        self,
        missing: List[str],
        unexpected: List[str],
        skipped_keys: List[str],
    ) -> None:
        unexplained_missing = [
            k for k in missing
            if not k.startswith(self._PRETRAIN_ALLOWED_MISSING_PREFIXES)
        ]
        if not (unexplained_missing or unexpected or skipped_keys):
            return
        raise RuntimeError(
            "Pretrain/finetune weight load rejected: checkpoint does not match "
            "the current model structure.\n"
            f"  unexplained missing ({len(unexplained_missing)}): {sorted(unexplained_missing)}\n"
            f"  unexpected ({len(unexpected)}): {sorted(unexpected)}\n"
            f"  shape mismatch skipped ({len(skipped_keys)}): {sorted(skipped_keys)}\n"
            "Only newly added parameters may be missing: "
            f"{list(self._PRETRAIN_ALLOWED_MISSING_PREFIXES)}"
        )

    _DEPTH_CLONE_PREFIX = "world_model.uni_branch.depth_uni_blocks."

    def _remap_depth_uni_clone_growth(
        self, state_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """depth_uni_clone_layers 扩容 remap（20260831）：克隆副本按 uni 的**末尾
        N 层**取，副本下标 j 对应 uni_blocks[first_layer + j]，而
        first_layer = len(uni_blocks) - N 随 N 变化。所以 N 从 6 扩到 14 时同名的
        depth_uni_blocks.0 在两边指的不是同一层（旧=uni_blocks[8]，新=uni_blocks[0]），
        而每层形状完全一致 ⇒ 下方的形状过滤查不出来，strict=False 会静默错位加载。
        这里按两边的 first_layer 差把副本下标平移回同一 uni 层。"""
        blocks = getattr(self.world_model.uni_branch, "depth_uni_blocks", None)
        if blocks is None:
            return state_dict
        prefix = self._DEPTH_CLONE_PREFIX
        ckpt_idx = set()
        for k in state_dict:
            if not k.startswith(prefix):
                continue
            head = k[len(prefix):].split(".", 1)[0]
            if head.isdigit():
                ckpt_idx.add(int(head))
        if not ckpt_idx:
            return state_dict
        n_ckpt = max(ckpt_idx) + 1
        n_cur = len(blocks)
        if n_ckpt == n_cur:
            return state_dict
        # first_layer 之差即副本下标偏移（uni 总层数两边相同）。
        offset = n_cur - n_ckpt
        remapped: Dict[str, Any] = {}
        moved = dropped = 0
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                remapped[k] = v
                continue
            rest = k[len(prefix):]
            head, _, tail = rest.partition(".")
            if not head.isdigit():
                remapped[k] = v
                continue
            new_i = int(head) + offset
            if not 0 <= new_i < n_cur:
                dropped += 1
                continue
            remapped[f"{prefix}{new_i}.{tail}"] = v
            moved += 1
        logger.info(
            "Depth uni clone growth remap: checkpoint had %d clone layers, model has "
            "%d; shifted clone index by %+d (tensors moved=%d dropped=%d). "
            "Checkpoint clone j now lands on the same uni layer it was copied from.",
            n_ckpt, n_cur, offset, moved, dropped,
        )
        return remapped

    def _reclone_depth_uni_blocks_if_absent(self, loaded_state: Dict[str, Any]) -> None:
        """depth uni 克隆层在 __init__ 里就 deepcopy 好了，此时持有的是**预训练
        Wan** 的权重。从一个不含 depth_uni_blocks.* 的 checkpoint（如开启克隆前的
        run）热启动时，strict=False 会静默放过这些键，副本就会与刚加载的 uni_blocks
        不一致——等价性和"从已训好的层出发"两个前提同时失效。这里逐层检查：
        checkpoint 覆盖到的副本层保持不动，没覆盖到的（如 6→14 扩容新增的那 8 层）
        按加载后的 uni_blocks 重新克隆。"""
        uni = self.world_model.uni_branch
        blocks = getattr(uni, "depth_uni_blocks", None)
        if blocks is None:
            return
        prefix = self._DEPTH_CLONE_PREFIX
        covered = {
            int(head)
            for k in loaded_state
            if k.startswith(prefix)
            and (head := k[len(prefix):].split(".", 1)[0]).isdigit()
        }
        missing = [i for i in range(len(blocks)) if i not in covered]
        if not missing:
            logger.info(
                "Depth uni clone blocks restored from checkpoint (%d layers).", len(blocks))
            return
        with torch.no_grad():
            for i in missing:
                blocks[i].load_state_dict(
                    uni.uni_blocks[uni.depth_uni_first_layer + i].state_dict())
        logger.info(
            "Re-cloned %d/%d depth uni clone layers from uni_blocks after load "
            "(clone idx %s -> uni_blocks idx %s); restored-from-checkpoint=%d. "
            "重克隆层前向与 depth_uni_clone_layers=0 逐位等价。",
            len(missing), len(blocks), missing,
            [uni.depth_uni_first_layer + i for i in missing], len(covered),
        )

    
    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode video pixels to latent space.
        
        Args:
            video_pixels: Video in pixel space [B, C, T, H, W], range [-1, 1]
            
        Returns:
            Video latents [B, C', T', H', W']
        """
        with torch.no_grad():
            return self.vae.encode(video_pixels)
    
    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode video latents to pixel space.
        
        Args:
            video_latents: Video latents [B, C, T, H, W]
            
        Returns:
            Video pixels [B, C', T', H', W'], range [-1, 1]
        """
        with torch.no_grad():
            video_pixels = []
            for i in range(video_latents.shape[0]):
                pixels = self.vae.decode([video_latents[i]])[0]
                video_pixels.append(pixels)
            result = torch.stack(video_pixels, dim=0)
            return result

    @staticmethod
    def _log_depth_latent_stats(
        depth_latent: torch.Tensor,
        depth_present: Optional[torch.Tensor],
    ) -> None:
        """Log compact clean-depth-latent stats (step 3: sanity, no loss yet)."""
        with torch.no_grad():
            z = depth_latent.float()
            if depth_present is not None:
                present = depth_present.to(z.device).bool()
                n_present = int(present.sum())
                if n_present > 0:
                    zp = z[present]
                    stats = (
                        f"present={n_present}/{int(present.numel())} "
                        f"shape={tuple(depth_latent.shape)} "
                        f"mean={zp.mean().item():.3f} std={zp.std().item():.3f} "
                        f"min={zp.min().item():.3f} max={zp.max().item():.3f}"
                    )
                else:
                    stats = (
                        f"present=0/{int(present.numel())} "
                        f"shape={tuple(depth_latent.shape)} (all-absent batch)"
                    )
            else:
                stats = (
                    f"present=unknown shape={tuple(depth_latent.shape)} "
                    f"mean={z.mean().item():.3f} std={z.std().item():.3f} "
                    f"min={z.min().item():.3f} max={z.max().item():.3f}"
                )
        logger.info("clean_depth_latent %s", stats)

    def save_debug_dict(self,
            debug_dict: List[Dict[str, Any]],
            save_dir: str = "./debug_files",
            prefix: str = "unidiffuser_debug"
    ) -> str:
        """
        保存 debug_dict 到文件，支持张量、PIL图片、numpy数组、字符串等类型

        Args:
            debug_dict: 要保存的调试字典列表
            save_dir: 保存目录（不存在则自动创建）
            prefix: 文件名前缀

        Returns:
            保存的文件路径
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 生成带时间戳的文件名（避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        save_path = os.path.join(save_dir, f"{prefix}_{timestamp}.pth")

        # 保存（torch.save 兼容所有PyTorch张量，且比pickle更适合PyTorch数据）
        torch.save(debug_dict, save_path)

        print(f"Debug dict saved to: {save_path}")
        return save_path

    def load_debug_dict(self,load_path: str, device: str = "cpu") -> List[Dict[str, Any]]:
        """
        读取保存的 debug_dict 文件，并可指定张量加载的设备

        Args:
            load_path: 保存的文件路径
            device: 张量加载到的设备（"cpu"/"cuda"）

        Returns:
            恢复的 debug_dict
        """
        # 加载数据
        debug_dict = torch.load(load_path, map_location=device)

        # 验证数据完整性（可选）
        if isinstance(debug_dict, list) and len(debug_dict) > 0:
            first_item = debug_dict[0]
            print(f"Loaded debug dict keys: {list(first_item.keys())}")
        else:
            print("Warning: Loaded debug dict is empty or not a list")

        return debug_dict


    def _normalize_state_actions(
        self,
        state: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        robot_names: Optional[List[str]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """模型边界归一化入口（B3 策略分流）：gripper_hand_minmax 走
        GripperHandNormPolicy（仅夹爪/灵巧手维 [-1,1]，其余物理量纲），
        否则旧全维 zscore（normalize_batch）。"""
        if self._norm_policy is not None:
            return self._norm_policy.normalize(state, actions, robot_names)
        return normalize_batch(state, actions, robot_names)

    def _denormalize_actions(
        self,
        actions: Optional[torch.Tensor],
        robot_names: Optional[List[str]],
    ) -> Optional[torch.Tensor]:
        """模型边界反归一化入口（与 _normalize_state_actions 策略对称）。"""
        if self._norm_policy is not None:
            return self._norm_policy.denormalize(actions, robot_names)
        return denormalize_batch(actions, robot_names)

    _ROT6_BAND_SUFFIX = "_rot6"

    def _build_repr_specs(self) -> Dict[str, Dict[str, Any]]:
        """relative_eef_rot6 的 per-robot 索引结构（B1/B2，模型边界用）。

        返回 {name: {"pos_dims": LongTensor, "groups": [dict]}}：
        - pos_dims：dim_map 落入 ``eef_*_pos`` band 的本地维（B1 相对化：
          动作减条件帧 state；关节/底盘/夹爪/灵巧手维不参与）；
          robot_layout 条目 ``eef_relative: false``（增量命令动作，如
          libero delta EEF——本身已是相对量）时置空，跳过相对化；
        - groups：robot_layout 条目的 ``rot_groups``（local/repr/cols/
          quat_order），cols 恰 6 列且落在同一 ``*_rot6`` band 内
          （train/train.py get_robot_layout 已校验，此处防御性复核）；
          每组带 ``relative`` 标记（= 条目 eef_relative），false 时动作
          旋转直接转 6D（delta 旋转，不做 R_sᵀ·R_i 复合）。
        """
        bands = {
            str(band): (int(span[0]), int(span[1]))
            for band, span in (self.config.unified_bands or {}).items()
        }
        specs: Dict[str, Dict[str, Any]] = {}
        for name, entry in (self.config.robot_layout or {}).items():
            dim_map = entry.get("dim_map")
            relative = bool(entry.get("eef_relative", True))
            pos_dims: List[int] = []
            if relative and dim_map and bands:
                for k, c in enumerate(dim_map):
                    c = int(c)
                    if c < 0:
                        continue
                    for band_name, (start, end) in bands.items():
                        if (
                            start <= c < end
                            and band_name.startswith("eef_")
                            and band_name.endswith("_pos")
                        ):
                            pos_dims.append(k)
                            break
            groups: List[Dict[str, Any]] = []
            for group in entry.get("rot_groups") or []:
                cols = [int(c) for c in group["cols"]]
                local = [int(d) for d in group["local"]]
                repr_name = str(group.get("repr", "rotvec"))
                if len(cols) != 6:
                    raise ValueError(
                        f"robot {name}: rot_groups 的 cols 必须恰 6 列（6D 旋转），"
                        f"got {cols}"
                    )
                if len(local) != action_repr.repr_width(repr_name):
                    raise ValueError(
                        f"robot {name}: rot_groups local 维数 {len(local)} 与 repr "
                        f"{repr_name!r} 宽度 {action_repr.repr_width(repr_name)} 不符"
                    )
                if bands:
                    hit = [
                        band_name
                        for band_name, (start, end) in bands.items()
                        if all(start <= c < end for c in cols)
                        and band_name.endswith(self._ROT6_BAND_SUFFIX)
                    ]
                    if len(hit) != 1:
                        raise ValueError(
                            f"robot {name}: rot_groups cols {cols} 未完整落在同一 "
                            f"*{self._ROT6_BAND_SUFFIX} band 内（命中 {hit}）"
                        )
                groups.append({
                    "local": torch.tensor(local, dtype=torch.long),
                    "cols": torch.tensor(cols, dtype=torch.long),
                    "repr": repr_name,
                    "quat_order": str(group.get("quat_order", "xyzw")),
                    "relative": relative,
                })
            specs[str(name)] = {
                "pos_dims": torch.tensor(pos_dims, dtype=torch.long),
                "groups": groups,
            }
        return specs

    def _repr_rot_local_dims(self, robot_name: Any) -> set:
        """该机器人被 rot group 接管的本地维（推理输出端 dropped 置零需排除）。"""
        if not self._repr_specs:
            return set()
        spec = self._repr_specs.get(str(robot_name))
        if spec is None:
            return set()
        dims = set()
        for group in spec["groups"]:
            dims.update(int(d) for d in group["local"].tolist())
        return dims

    def _gather_native_actions(
        self,
        actions: torch.Tensor,
        state_local: Optional[torch.Tensor],
        robot_names: Optional[List[str]],
        cols: Optional[List[int]],
        offset: int,
        dim: int,
    ) -> torch.Tensor:
        """统一空间 [B, T, U] → per-robot 原始维 [B, T, dim]（推理输出端）。

        dim_map gather（_slice_unified_actions）后做 B1/B2 还原：eef pos 维
        加回条件帧 state；rot6 列 → R_rel（正交化）→ R_abs = R_s·R_rel →
        本地原生旋转表示写入 rot group 本地维（这些维在 dim_map 中为 -1，
        gather 出来是占位 0，此处覆写为还原值）。absolute_native 下等价于
        纯 _slice_unified_actions。

        state_local：策略归一化后的 local 空间条件帧 state [B, dim]（
        relative_eef_rot6 的相对参考点；缺省时按单位旋转/零位移还原）。
        """
        out = _slice_unified_actions(actions, cols, offset, dim)
        if self._repr_specs is None:
            return out
        B = out.shape[0]
        names = robot_names if robot_names is not None else [None] * B
        if isinstance(names, str):
            names = [names] * B
        for i in range(B):
            spec = self._repr_specs.get(str(names[i])) if names[i] is not None else None
            if spec is None:
                continue
            pos_dims = spec["pos_dims"].to(out.device)
            if pos_dims.numel() and state_local is not None:
                out[i, :, pos_dims] += state_local[i, pos_dims].to(out.dtype)
            for group in spec["groups"]:
                local = group["local"].to(out.device)
                cols_t = group["cols"].to(out.device)
                if int(local.max()) >= out.shape[-1]:
                    continue
                rel_6d = actions[i, :, cols_t]
                if not group.get("relative", True):
                    # eef_relative=false（delta 旋转目标）：直接 6D → 本地表示，
                    # 不做 R_s 复合
                    out[i, :, local] = action_repr.matrix_to_repr(
                        action_repr.d6_to_matrix(rel_6d),
                        group["repr"], group["quat_order"],
                    ).to(out.dtype)
                    continue
                if state_local is not None:
                    s_rot = state_local[i, local]
                else:
                    # 防御：缺 state 参考时零姿态（各 repr 的零值均映射单位旋转）
                    s_rot = torch.zeros(
                        int(local.numel()), dtype=actions.dtype, device=actions.device)
                abs_rot = action_repr.absolute_rotation_from_6d(
                    s_rot, rel_6d, group["repr"], group["quat_order"])
                out[i, :, local] = abs_rot.to(out.dtype)
        return out

    def _scatter_to_unified(
        self,
        state: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        robot_names: Optional[List[str]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """把逐机器人 action/state 散开到统一维 [B,..,U]（模型边界，归一化之后）。

        优先按 ``self.config.robot_layout[name]["dim_map"]``（逐维语义槽位映射：
        本地第 k 维 → 统一空间列索引，见 train/train.py get_robot_layout 与
        YAML dataset.unified_bands 注册表）把 [B,..,dim_r] 写进 [B,..,U] 零张量；
        未提供 dim_map 的旧配置回退 ``["offset"]`` 前填（语义混合，warning）。
        同时产出 [B, U] 有效维 bool mask（供 action loss 只对有效维加权）。
        未注册 layout 的机器人默认 offset=0、dim=config.action_dim 兜底并
        warning（机器人维度取自 ``config.robot_configs[name]``，与
        robot_configs 同源）。数据管线零改动（见 .AGENTS_CONTENT/04_data_model.md）。

        ``action_target_repr == "relative_eef_rot6"``（B1/B2）时额外：eef
        pos 维动作减条件帧 state（local 空间直减），robot_layout 条目的
        rot_groups 旋转维组转 6D 写入 ``*_rot6`` band 列（state 写绝对
        姿态、actions 写 R_sᵀ·R_i 相对姿态）；这些本地维在 dim_map 中须为
        -1（train/train.py 校验），scatter 后由 group_writes 覆写。
        robot_layout 条目 ``eef_relative: false``（20260812，增量命令动作
        如 libero delta EEF）时：pos 不减 state、动作旋转直接转 6D（delta
        旋转本身已是相对量，不做 R_sᵀ·R_i 复合）。
        """
        U = int(self.config.unified_action_dim)
        B = state.shape[0] if state is not None else actions.shape[0]
        device = (state if state is not None else actions).device
        dtype = (state if state is not None else actions).dtype
        layout = self.config.robot_layout or {}
        robot_configs = self.config.robot_configs or {}
        names = robot_names if robot_names is not None else [None] * B
        if isinstance(names, str):
            names = [names] * B
        # batch 源齐次：实际张量最后一维（robot_names 缺省/未注册时以实际维
        # 兜底，避免切片越界）
        actual_dim = (state if state is not None else actions).shape[-1]
        # 每样本目标列配对（dim_map 语义槽位 或 offset 前填展开）。dim_map
        # 可含 -1 = 摘除该本地维（修复 #7，如 agibot 灵巧手占位维）：不进
        # scatter、不进有效维 mask；推理输出端由 _slice_unified_actions +
        # _dropped_local_dims 重建为 raw 0。
        pair_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for name in names:
            key = str(name) if name is not None else ""
            entry = layout.get(key)
            if entry is None:
                if name is not None:
                    logger.warning(
                        "robot %s 不在 robot_layout 中，按 offset=0、dim=%d 兜底",
                        name, self.config.action_dim,
                    )
                dim = int(robot_configs.get(key, self.config.action_dim))
                cols = list(range(0, dim))
            else:
                dim_map = entry.get("dim_map")
                if dim_map is not None:
                    cols = [int(c) for c in dim_map]
                else:
                    logger.warning(
                        "robot %s 未注册 unified_dim_map，回退 offset=%d 前填"
                        "（统一空间列语义混合，建议尽快注册 dim_map）",
                        name, int(entry["offset"]),
                    )
                    dim = int(robot_configs.get(key, self.config.action_dim))
                    start = int(entry["offset"])
                    cols = list(range(start, start + dim))
            # (本地维 k, 统一列 c) 显式配对：k 不越过实际张量维，c 必须在
            # 统一空间容量内（-1 摘除维在此被过滤）。
            pairs = [
                (k, c) for k, c in enumerate(cols[:actual_dim]) if 0 <= c < U
            ]
            pair_indices.append(
                (
                    torch.tensor([k for k, _ in pairs], dtype=torch.long, device=device),
                    torch.tensor([c for _, c in pairs], dtype=torch.long, device=device),
                )
            )
        # B1/B2（relative_eef_rot6，20260811 对齐 MotusBrain）：local 空间做
        # EEF 相对化（pos 直减条件帧 state；rot group R_rel = R_sᵀ·R_i 转 6D），
        # rot group 的 6D 值在下方 scatter 后按 cols 覆写（这些本地维在
        # dim_map 中为 -1，不在 pair_indices 内）。state 侧 rot 写绝对姿态 6D；
        # scatter 后的 state 在 V14 中既作为相对化锚点，也作为 state token 输入。
        # absolute_native 下
        # group_writes 为 None，行为与旧版一致。
        group_writes = None
        if self._repr_specs is not None:
            group_writes = [[] for _ in range(B)]
            if actions is not None:
                actions = actions.clone()
            for i in range(B):
                spec = (
                    self._repr_specs.get(str(names[i]))
                    if names[i] is not None else None
                )
                if spec is None:
                    continue
                pos_dims = spec["pos_dims"].to(device)
                if (
                    pos_dims.numel()
                    and state is not None
                    and actions is not None
                ):
                    actions[i, ..., pos_dims] -= state[i, pos_dims]
                for group in spec["groups"]:
                    local = group["local"].to(device)
                    if int(local.max()) >= actual_dim:
                        continue
                    cols_t = group["cols"].to(device)
                    s_rot = state[i, local] if state is not None else None
                    s6 = None
                    if s_rot is not None:
                        s6 = action_repr.matrix_to_6d(action_repr.repr_to_matrix(
                            s_rot, group["repr"], group["quat_order"]))
                    a6 = None
                    if actions is not None:
                        a_rot = actions[i, ..., local]
                        if s_rot is not None and group.get("relative", True):
                            _, a6 = action_repr.relative_rotation_6d(
                                s_rot, a_rot, group["repr"], group["quat_order"])
                        else:
                            # 无 state 参考（防御路径）或 eef_relative=false
                            # （delta 旋转本身已是相对量）：直接转 6D
                            a6 = action_repr.matrix_to_6d(
                                action_repr.repr_to_matrix(
                                    a_rot, group["repr"], group["quat_order"]))
                    group_writes[i].append((cols_t, s6, a6))
        mask = torch.zeros(B, U, dtype=torch.bool, device=device)
        for i in range(B):
            _, col_idx = pair_indices[i]
            if col_idx.numel():
                mask[i, col_idx] = True
        if state is not None:
            scattered_state = torch.zeros(B, U, dtype=dtype, device=device)
            for i in range(B):
                local_idx, col_idx = pair_indices[i]
                if col_idx.numel():
                    scattered_state[i, col_idx] = state[i, local_idx]
        else:
            scattered_state = None
        if actions is not None:
            scattered_actions = torch.zeros(*actions.shape[:-1], U, dtype=dtype, device=device)
            for i in range(B):
                local_idx, col_idx = pair_indices[i]
                if col_idx.numel():
                    scattered_actions[i, ..., col_idx] = actions[i, ..., local_idx]
        else:
            scattered_actions = None
        # B1/B2：rot group 的 6D 值覆写（state 绝对 6D / actions 相对 6D），
        # 并标记有效维 mask（loss 参与）。
        if group_writes is not None:
            for i in range(B):
                for cols_t, s6, a6 in group_writes[i]:
                    mask[i, cols_t] = True
                    if s6 is not None and scattered_state is not None:
                        scattered_state[i, cols_t] = s6.to(dtype)
                    if a6 is not None and scattered_actions is not None:
                        scattered_actions[i, ..., cols_t] = a6.to(dtype)
        return scattered_state, scattered_actions, mask


    def _build_action_validity_mask(
        self,
        state: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        robot_names: Optional[List[str]],
    ) -> Optional[torch.Tensor]:
        """逐帧有效性 mask（[B, T, U] bool；batch 内无门控机器人时返回 None）。

        robot_layout 条目 ``validity_gates``（20260816，YAML
        ``dataset.robot_names[*].unified_validity_gates``，校验见
        train/train.py get_robot_layout）：[{local: 本地 0/1 validity 维,
        cols: (start, end) 统一空间列区间}]。用于 egoscale 人体观测代理
        动作：检测缺失的手由数据工厂零位姿补全 + validity=0（缺失协议），
        这些帧的手部列不计入 action loss（用户拍板"无效手的部分 loss 取 0
        忽略"——零位姿不能被当成真实坐标目标）；条件帧（state）该手无效时
        delta 相对化锚点同样无意义，整窗该手列屏蔽。必须在归一化前从原始
        action/state 提取（0/1 位经不起归一化/相对化变换）。
        """
        layout = self.config.robot_layout or {}
        if not layout or actions is None:
            return None
        B, T = actions.shape[0], actions.shape[1]
        names = robot_names if robot_names is not None else [None] * B
        if isinstance(names, str):
            names = [names] * B
        mask: Optional[torch.Tensor] = None
        for i in range(B):
            entry = layout.get(str(names[i])) if names[i] is not None else None
            gates = (entry or {}).get("validity_gates") or []
            if not gates:
                continue
            if mask is None:
                mask = torch.ones(
                    B, T, int(self.config.unified_action_dim),
                    dtype=torch.bool, device=actions.device,
                )
            for gate in gates:
                local = int(gate["local"])
                if local >= actions.shape[-1]:
                    continue
                valid = actions[i, :, local] > 0.5
                if state is not None and local < state.shape[-1]:
                    valid = valid & bool(state[i, local] > 0.5)
                start, end = int(gate["cols"][0]), int(gate["cols"][1])
                mask[i, ~valid, start:end] = False
        return mask

    def prepare_training_context(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        vlm_input_messages: Optional[List[str]] = None,
        robot_names: Optional[List[str]] = None,
        source_datasets: Optional[List[str]] = None,
        state_action_policies: Optional[List[str]] = None,
        first_frame_pils: Optional[List[PIL.Image.Image]] = None,
        depth_first: Optional[torch.Tensor] = None,    # [B, 3, H, W] in [0,1]
        depth_frames: Optional[torch.Tensor] = None,   # [B, num_frames, 3, H, W] in [0,1]
        depth_present: Optional[torch.Tensor] = None,  # [B] bool
        ego_head_pose: Optional[torch.Tensor] = None,  # [B, T, 7]
        ego_head_valid: Optional[torch.Tensor] = None,  # [B, T]
        ego_K: Optional[torch.Tensor] = None,  # [B, 3, 3]
        ego_pose_used: Optional[torch.Tensor] = None,  # [B]
        timing: Optional[Dict[str, float]] = None,
        timing_sync_cuda: bool = True,
        depth_valid_mask: Optional[torch.Tensor] = None,  # bool [B,T+1,H,W]
        history_inputs: Optional[Dict[str, torch.Tensor]] = None,  # see _prepare_history
    ) -> Dict[str, Any]:
        """构建噪声无关的训练上下文（每个 batch 一次）。

        包含 ego4d prompt 策略 + 逐机器人 action/state 归一化、视频像素预处理、
        冻结 VAE encode 得到的 clean latent、masked VLM 消息，以及（20260818_v2
        起）und 的 VLM 冻结段特征（rewrite[磁盘缓存] → 输出侧字段 dropout[训练
        态] → extract，见 und_branch.precompute_vlm_features）。训练侧在
        ``training.train_steps_per_vae_encode > 1`` 时构建一次并跨多次
        ``training_step`` 调用复用（每次调用各自重采样噪声/timestep），省去
        重复 VAE encode 与逐步 VLM 编码；可训练的 bridge+und_encoder 不在此
        （每次参数更新后在 forward 内经 und_precomputed 重算，梯度正常回传）。
        """
        depth_mode = depth_condition_mode(self.config)
        validate_depth_losses(self.config, masked=depth_valid_mask is not None)
        if depth_mode == "rgb_only":
            if depth_valid_mask is None:
                raise ValueError("rgb_only training requires depth_valid_mask [B,T+1,H,W]")
            if depth_first is None or depth_frames is None:
                raise ValueError("rgb_only training requires current+future depth labels")
        if depth_frames is not None and depth_first is None:
            raise ValueError("depth_frames requires depth_first as its current-frame label")
        if depth_valid_mask is not None and depth_frames is None:
            raise ValueError("depth_valid_mask supplied without depth targets")
        collect_timing = timing is not None
        if timing is None:
            timing = {}

        B = video_frames.shape[0]

        prompt_start = time.perf_counter()
        state, actions = self._apply_ego4d_action_state_prompt(
            state=state,
            actions=actions,
            source_datasets=source_datasets,
            state_action_policies=state_action_policies,
        )
        # 逐帧有效性 mask（20260816，egoscale 无效手 loss 屏蔽）：必须在
        # 归一化/scatter 之前从原始 0/1 validity 位提取。
        action_validity_mask = self._build_action_validity_mask(
            state, actions, robot_names)
        # 逐机器人 action/state 归一化（B3 策略分流：gripper_hand_minmax 仅
        # 夹爪/灵巧手维 [-1,1]、其余物理量纲；zscore 旧全维归一化）：flow
        # matching / 损失全程在模型内部空间进行（解码输出同属该空间，训练无
        # 需反归一化）。
        state, actions = self._normalize_state_actions(state, actions, robot_names)
        # 统一 action 空间：归一化之后、模型边界完成 scatter（把每机器人
        # [B,..,dim_r] 按 robot_layout 的 unified_dim_map 语义槽位写入
        # [B,..,U] 零张量，未注册 dim_map 的旧配置回退 offset 前填；产出
        # [B, U] 有效维 mask 供 action loss 加权；数据管线零改动）。
        state, actions, action_dim_mask = self._scatter_to_unified(
            state, actions, robot_names)
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["ego4d_prompt_seconds"] = time.perf_counter() - prompt_start

        # Video pipeline
        video_prepare_start = time.perf_counter()
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, frames+1, H, W]
        if collect_timing:
            _record_tensor_finite_stats(timing, "input_first_frame", first_frame)
            _record_tensor_finite_stats(timing, "input_video_frames", video_frames)
            _record_tensor_finite_stats(timing, "full_video_norm", full_video)
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["video_prepare_seconds"] = time.perf_counter() - video_prepare_start

        encode_start = time.perf_counter()
        with torch.no_grad():
            clean_full_latent = self.encode_video(full_video.to(self.dtype))  # [B, 48, latent_frames, H', W']
        if collect_timing:
            _record_tensor_finite_stats(timing, "clean_full_latent", clean_full_latent)
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["vae_encode_video_seconds"] = time.perf_counter() - encode_start

        # Current+future depth LABELS use the frozen RGB VAE. In rgb_only mode
        # none of these labels is a clean condition. Neutralize invalid pixels
        # before VAE encoding, then conservatively mask nominal latent cells.
        # The mask approximates, rather than exactly traces, the VAE receptive field.
        clean_depth_latent = None
        latent_depth_mask = None
        if depth_frames is not None:
            depth_encode_start = time.perf_counter()
            depth_first_norm = (depth_first * 2.0 - 1.0).unsqueeze(2)  # [B, 3, 1, H, W]
            depth_normalized = (depth_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]
            full_depth = torch.cat([depth_first_norm, depth_normalized], dim=2)  # [B, 3, T+1, H, W]
            if depth_valid_mask is not None:
                full_depth, depth_valid_mask = sanitize_depth_targets(
                    full_depth, depth_valid_mask, depth_present)
            with torch.no_grad():
                clean_depth_latent = self.encode_video(full_depth.to(self.dtype))
            if depth_valid_mask is not None:
                latent_depth_mask = depth_latent_valid_mask(
                    depth_valid_mask, clean_depth_latent.shape).to(clean_depth_latent.device)
                # Invalid latent cells are not GT or conditioning values.
                clean_depth_latent = torch.where(latent_depth_mask, clean_depth_latent, 0.0)
            if collect_timing:
                _record_tensor_finite_stats(timing, "clean_depth_latent", clean_depth_latent)
                if timing_sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing["vae_encode_depth_seconds"] = (
                    time.perf_counter() - depth_encode_start
                )
            self._log_depth_latent_stats(clean_depth_latent, depth_present)

        depth_motion_weight = None
        depth_motion_frac = None
        depth_motion_pose_used = None
        if (
            clean_depth_latent is not None
            and bool(getattr(self.config, "depth_motion_loss_enable", False))
            and depth_first is not None
            and depth_frames is not None
        ):
            from thinker_wm.models.depth_motion import depth_motion_latent_weight

            depth_01 = torch.cat(
                [depth_first.unsqueeze(1), depth_frames], dim=1
            )
            rgb_01 = torch.cat(
                [first_frame.unsqueeze(1), video_frames], dim=1
            )
            use_pose = bool(getattr(self.config, "depth_motion_ego_pose", False))
            depth_motion_weight, depth_motion_frac = depth_motion_latent_weight(
                depth_01,
                rgb_01,
                latent_t=int(clean_depth_latent.shape[2]),
                latent_hw=(
                    int(clean_depth_latent.shape[3]),
                    int(clean_depth_latent.shape[4]),
                ),
                z_near=float(getattr(self.config, "depth_z_near", 0.15)),
                z_far=float(getattr(self.config, "depth_z_far", 4.0)),
                mode=str(getattr(self.config, "depth_mode", "log")),
                frac_max=float(getattr(self.config, "depth_motion_frac_max", 0.12)),
                require_pose=bool(
                    getattr(self.config, "depth_motion_require_pose", False)
                ),
                ego_pose_used=ego_pose_used if use_pose else None,
                ego_poses=ego_head_pose if use_pose else None,
                ego_head_valid=ego_head_valid if use_pose else None,
                ego_K=ego_K if use_pose else None,
            )
            depth_motion_weight = depth_motion_weight.to(
                device=clean_depth_latent.device, dtype=clean_depth_latent.dtype
            )
            if use_pose and ego_pose_used is not None:
                depth_motion_pose_used = ego_pose_used

        masked_vlm_messages = list(vlm_input_messages) if vlm_input_messages else ["Please output empty string"] * B

        # prepare 级 VLM 半缓存（20260818_v2）：VLM 冻结段（rewrite[磁盘缓存]
        # → 输出侧字段 dropout[训练态] → 特征提取）每 batch 一次，随本 context
        # 跨 train_steps_per_vae_encode 个 update step 复用；training_step 内
        # 按本步模态 dropout 在「本特征 / uncond token 嵌入常量」间选择后，
        # 只跑可训练的 bridge+und_encoder。autocast 语境与原 forward 内编码
        # 一致（数值路径对齐）；extract/generate 内部均有 no_grad。
        und_start = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=self.dtype,
                enabled=torch.cuda.is_available(),
            ):
                und_vlm_hidden, und_real_lens = (
                    self.world_model.und_branch.precompute_vlm_features(
                        masked_vlm_messages, first_frame_pils,
                        timing=timing if collect_timing else None,
                        timing_sync_cuda=timing_sync_cuda,
                        timing_level=int(getattr(self.config, "timing_level", 1)),
                    )
                )
        if collect_timing:
            _record_tensor_finite_stats(timing, "und_vlm_hidden", und_vlm_hidden)
            _hits, _misses = self.world_model.und_branch._last_rewrite_cache_stats
            timing["vlm_rewrite_cache_hits"] = int(_hits)
            timing["vlm_rewrite_cache_misses"] = int(_misses)
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["und_precompute_seconds"] = time.perf_counter() - und_start

        history_start = time.perf_counter()
        history = self._prepare_history(
            history_inputs, robot_names, state_action_policies, source_datasets)
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["history_encode_seconds"] = time.perf_counter() - history_start

        return {
            "history": history,
            "state": state,
            "actions": actions,
            "clean_full_latent": clean_full_latent,
            "masked_vlm_messages": masked_vlm_messages,
            "und_vlm_hidden": und_vlm_hidden,
            "und_real_lens": und_real_lens,
            "action_dim_mask": action_dim_mask,
            "action_validity_mask": action_validity_mask,
            "clean_depth_latent": clean_depth_latent,
            "depth_latent_valid_mask": latent_depth_mask,
            "depth_condition_mode": depth_mode,
            "depth_present": depth_present,
            "depth_motion_weight": depth_motion_weight,
            "depth_motion_frac": depth_motion_frac,
            "depth_motion_pose_used": depth_motion_pose_used,
        }

    def _history_state_unified(self, state, robot_names):
        rows = []
        for j in range(state.shape[1]):
            s_j, _ = self._normalize_state_actions(state[:, j], None, robot_names)
            s_j, _, _ = self._scatter_to_unified(s_j, None, robot_names)
            rows.append(s_j)
        return torch.stack(rows, dim=1).to(self.device)

    def _prepare_history(self, history_inputs, robot_names,
                         state_action_policies=None, source_datasets=None, *, sparse_latent=None):
        """Batch history (data_prepare.temporal_windows contract) -> world_model history dict.

        history_inputs keys (each group optional, both may be present):
          history_sparse_rgb [B,Ts,3,H,W] [0,1], history_sparse_times [B,Ts] s (<0),
          history_sparse_valid [B,Ts] bool, history_sparse_kind [B,Ts] long,
          history_sparse_state [B,Ts,Ds] raw;
          history_recent_rgb [B,1+4m,3,H,W], history_recent_times [B,1+4m],
          history_recent_valid [B,1+4m], history_recent_state [B,1+4m,Ds].
        Sparse frames are encoded one by one; the recent clip is one causal VAE
        clip (keeps motion), each of its latents timed at its last frame.
        Slots are ordered oldest first (sparse, then recent). RGB only: depth
        history is never encoded. Samples whose state is replaced by the
        learned prompt (ego4d-style policy) get no history state.
        """
        if not history_inputs:
            return None
        if not bool(getattr(self.config, "history_enable", False)):
            raise ValueError("history inputs given but history_enable=False")
        device = self.device
        latents, times, valids, kinds, states = [], [], [], [], []
        has_state = []
        if (history_inputs.get("history_sparse_rgb") is not None
                and history_inputs["history_sparse_rgb"].shape[1] > 0):
            px = history_inputs["history_sparse_rgb"].to(device)
            v = history_inputs["history_sparse_valid"].to(device=device, dtype=torch.bool)
            if sparse_latent is not None:
                # Runtime-only VAE reuse. Time/kind/state and all attention
                # representations are rebuilt below for this observation.
                if self.training:
                    raise ValueError("cached history latents are inference-only")
                if (sparse_latent.ndim != 5 or sparse_latent.shape[0] != px.shape[0]
                        or sparse_latent.shape[2] != px.shape[1]):
                    raise ValueError("cached sparse latent does not match history slots")
                latents.append(sparse_latent.to(device))
            else:
                latents.append(TT.encode_history_frames(
                    self.encode_video, px, v, self.dtype,
                    valid_only=bool(getattr(self.config, "history_encode_valid_only", False))))
            times.append(history_inputs["history_sparse_times"].to(device, torch.float32))
            valids.append(v)
            k = history_inputs.get("history_sparse_kind")
            kinds.append(torch.full_like(v, 1, dtype=torch.long) if k is None else k.to(device).long())
            states.append(history_inputs.get("history_sparse_state"))
        if (history_inputs.get("history_recent_rgb") is not None
                and history_inputs["history_recent_rgb"].shape[1] > 0):
            px = history_inputs["history_recent_rgb"].to(device)
            fv = history_inputs["history_recent_valid"].to(device=device, dtype=torch.bool)
            idx = TT.recent_clip_latent_frames(px.shape[1])
            latents.append(TT.encode_recent_clip(self.encode_video, px, self.dtype))
            times.append(history_inputs["history_recent_times"].to(device, torch.float32)[:, idx])
            v = TT.recent_clip_latent_valid(fv)
            valids.append(v)
            kinds.append(torch.zeros_like(v, dtype=torch.long))
            st = history_inputs.get("history_recent_state")
            states.append(None if st is None else st[:, idx])
        if not latents:
            return None
        latent = torch.cat(latents, dim=2)
        times = torch.cat(times, dim=1)
        valid = torch.cat(valids, dim=1)
        kind = torch.cat(kinds, dim=1)
        if bool((times[valid] >= 0).any()):
            raise ValueError("history times must be strictly before the current frame")
        hstate = None
        if any(s is not None for s in states):
            if any(s is None for s in states):
                raise ValueError("history state must be given for all history groups or none")
            hstate = torch.cat([self._history_state_unified(s, robot_names) for s in states], dim=1)
            prompt = None
            if state_action_policies is not None:
                prompt = [str(p).lower() == "learned_prompt" for p in state_action_policies]
            elif source_datasets is not None:
                prompt = [str(s).lower() == "ego4d" for s in source_datasets]
            if prompt is not None and any(prompt):
                keep = ~torch.tensor(prompt, dtype=torch.bool, device=device)
                hstate = hstate * keep[:, None, None].to(hstate.dtype)
        return {
            "latent": latent,
            "times": times.to(device=device, dtype=torch.float32),
            "valid": valid,
            "kind": None if kind is None else kind.to(device=device, dtype=torch.long),
            "state": hstate,
        }

    def _temporal_decision(self, seed_step, depth_first_frame_conditioned: bool):
        idx = int(seed_step) if seed_step is not None else self._temporal_sample_idx
        if self.training:
            self._temporal_sample_idx += 1
        return TT.sample_step_decision(self.config, idx, self.training, depth_first_frame_conditioned)

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        first_frame_pils: Optional[List[PIL.Image.Image]] = None,
        t5_input_messages: Optional[List[str]] = None,
        vlm_input_messages: Optional[List[str]] = None,
        robot_names: Optional[List[str]] = None,
        source_datasets: Optional[List[str]] = None,
        action_loss_weights: Optional[torch.Tensor] = None,
        video_loss_weights: Optional[torch.Tensor] = None,
        state_action_policies: Optional[List[str]] = None,
        depth_first: Optional[torch.Tensor] = None,    # [B, 3, H, W] in [0,1]
        depth_frames: Optional[torch.Tensor] = None,   # [B, num_frames, 3, H, W] in [0,1]
        depth_present: Optional[torch.Tensor] = None,  # [B] bool
        return_dict: bool = True,
        timing: Optional[Dict[str, float]] = None,
        timing_sync_cuda: bool = True,
        training_context: Optional[Dict[str, Any]] = None,
        dropout_seed_step: Optional[int] = None,
        depth_valid_mask: Optional[torch.Tensor] = None,  # bool [B,T+1,H,W]
        history_inputs: Optional[Dict[str, torch.Tensor]] = None,
        recovery_features: Optional[torch.Tensor] = None,  # [B, recovery_feature_dim]
        recovery_valid: Optional[torch.Tensor] = None,     # [B] bool
        temporal_decision: Optional["TT.StepDecision"] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities.

        temporal_mode=block_causal 时：每步按 global_step 定种抽一个
        TemporalRequest（cut / world 模式，跨 rank 一致；temporal_decision
        可显式覆盖，测试与离线诊断用），非 TARGET 角色位置写回 GT，loss 只统计
        TARGET latent / 动作步；world 模式动作全为给定命令（t=0、不 drop、
        不算 action loss、不受 zero_action_loss_force_max_noise 影响）。
        
        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            return_dict: Whether to return detailed outputs
            training_context: 可选，prepare_training_context 的产物；传入后跳过
                本步的归一化/视频预处理/VAE encode（跨多次参数更新复用同一
                latent，每次调用仍各自重采样噪声/timestep）。
            dropout_seed_step: 可选，模态 dropout 采样的定种步号（修复 #9b，
                通常传 trainer 的 global_step）：跨 rank/弹性重启一致——决策
                变为 global_step 的纯函数，不再依赖本地采样计数器（计数器不
                进 checkpoint，重启后各 rank 可能分叉死锁）。缺省 None 回退
                内部计数器（旧行为）。

        Returns:
            Dictionary containing losses and metrics
        """


        # assert len(first_frame_pils) == 1
        # first_frame_pils[0].save('./show.jpg', "JPEG", quality=95)
        # print("t5_input_messages:", t5_input_messages)
        # print("vlm_input_messages:", vlm_input_messages)
        # print('state:', state)
        # raise ValueError

        collect_timing = timing is not None
        if timing is None:
            timing = {}

        # 噪声无关准备（归一化 + 视频像素预处理 + VAE encode + masked 消息）：
        # training.train_steps_per_vae_encode > 1 时由调用方经
        # prepare_training_context 构建一次并传入，跨多次参数更新复用，
        # 省去重复 VAE encode；缺省在内部构建，行为与拆分前一致。
        if training_context is None:
            training_context = self.prepare_training_context(
                first_frame=first_frame,
                video_frames=video_frames,
                state=state,
                actions=actions,
                vlm_input_messages=vlm_input_messages,
                robot_names=robot_names,
                source_datasets=source_datasets,
                state_action_policies=state_action_policies,
                first_frame_pils=first_frame_pils,
                depth_first=depth_first,
                depth_frames=depth_frames,
                depth_present=depth_present,
                depth_valid_mask=depth_valid_mask,
                timing=timing if collect_timing else None,
                timing_sync_cuda=timing_sync_cuda,
                history_inputs=history_inputs,
            )
        actions = training_context["actions"]
        # V14 state token：prepare_training_context 里的 "state" 已经过归一化
        # 与 _scatter_to_unified，是统一维 [B, U]，不能再 scatter 一次；也不能
        # 用 training_step 的同名入参（那是未变换的 per-robot [B, state_dim]）。
        scattered_state = training_context.get("state")
        clean_full_latent = training_context["clean_full_latent"]
        masked_vlm_messages = training_context["masked_vlm_messages"]
        action_dim_mask = training_context.get("action_dim_mask")
        action_validity_mask = training_context.get("action_validity_mask")
        # depth 第三生成流（step 3 起 prepare_training_context 产出）：非 None 时
        # 走 depth flow-matching loss；None（未接 depth 数据/enable_depth=False）
        # 时整条 depth 路径跳过，与原模型一致。
        depth_mode = depth_condition_mode(self.config)
        depth_first_frame_conditioned = depth_mode == "first_frame"
        latent_depth_mask = training_context.get("depth_latent_valid_mask")
        if depth_valid_mask is not None and latent_depth_mask is None:
            raise ValueError("Rebuild training_context with depth_valid_mask before encoding targets")
        if latent_depth_mask is not None and training_context.get("depth_motion_weight") is not None:
            raise ValueError("Masked depth context cannot contain unmasked motion weights")
        validate_depth_losses(self.config, masked=latent_depth_mask is not None)
        if training_context.get("depth_condition_mode", "first_frame") != depth_mode:
            raise ValueError("Training context depth mode differs from model config")
        if depth_mode == "rgb_only" and latent_depth_mask is None:
            raise ValueError("rgb_only training context requires depth_latent_valid_mask")
        clean_depth_latent = training_context.get("clean_depth_latent")
        if depth_mode == "rgb_only" and clean_depth_latent is None:
            raise ValueError("rgb_only training context requires depth targets")
        depth_present = training_context.get("depth_present")
        depth_motion_weight = training_context.get("depth_motion_weight")
        depth_motion_frac = training_context.get("depth_motion_frac")
        depth_motion_pose_used = training_context.get("depth_motion_pose_used")
        B = clean_full_latent.shape[0]

        # 模态 dropout（评审 #4）：video/action/prompt 三模态各自独立概率、batch
        # 级、训练态、每 update step 重采、跨 rank 全局一致（否则 ZeRO-1 梯度
        # 归约跨 rank 分歧死锁，见 _sample_modality_dropout；同 step 全 rank
        # drop 相同模态，兼负跨 rank 负载均衡）。被 drop 模态不进入网络
        # （video/action 传 None → world_model 跳过 encode/decode），其 loss
        # 置零；video 与 action 同时被 drop 时随机保底保留一个可训练模态
        #（prompt 缺失可由无条件文本表示替代，无需保底）。
        # 20260818_v2 起逐样本 condition dropout 并入本机制
        #（modality_dropout_prompt 单一承担无条件分支训练）。
        drop_video, drop_action, drop_prompt, drop_depth = self._sample_modality_dropout(
            float(getattr(self.config, "modality_dropout_video", 0.0)),
            float(getattr(self.config, "modality_dropout_action", 0.0)),
            float(getattr(self.config, "modality_dropout_prompt", 0.0)),
            float(getattr(self.config, "modality_dropout_depth", 0.0)),
            training=self.training,
            seed_step=dropout_seed_step,
        )
        modality_dropped_prompt = drop_prompt

        decision = None
        if str(getattr(self.config, "temporal_mode", "bidirectional")) == "block_causal":
            decision = temporal_decision or self._temporal_decision(
                dropout_seed_step, depth_first_frame_conditioned)
            if decision.request.depth_first_frame_conditioned != depth_first_frame_conditioned:
                raise ValueError("training request depth conditioning differs from depth_condition_mode")
            TT.num_target_blocks_check(decision.request, int(clean_full_latent.shape[2]))
            # Explicit mode mixtures define their own conditioning task; do not
            # silently turn Joint/IDM/FDM into a missing-RGB/action task.
            if self.config.training_mode_probs is not None or decision.mode == TT.MODE_IDM:
                drop_video = drop_action = 0
            if decision.mode == TT.MODE_WORLD:
                # 世界模式：动作是给定条件，不能被 drop；至少保留一个预测目标。
                drop_action = 0
                if drop_video and (clean_depth_latent is None or drop_depth):
                    drop_video = 0
        elif temporal_decision is not None:
            raise ValueError("temporal_decision requires temporal_mode=block_causal")
        world_mode = decision is not None and decision.mode == TT.MODE_WORLD
        capability_enabled = CT.enabled(self.config)
        capability_records = training_context.get("capability_records") or [None] * B
        if len(capability_records) != B:
            raise ValueError("capability_records must have one row per sample")
        pair_index = torch.arange(B, device=actions.device)
        pair_valid = torch.zeros(B, device=actions.device, dtype=torch.bool)
        if capability_enabled:
            CT.validate_config(self.config)
            pair_index, pair_valid = CT.partners(
                capability_records, "a2v" if world_mode else "language", actions.device)
            CT.validate_pair_inputs(training_context, pair_index, pair_valid, "a2v" if world_mode else "language")
        if any(r is not None for r in capability_records):
            if not world_mode:
                # A failed recording is valid world-model data, not a positive
                # Joint policy/video/depth target. Unannotated ordinary demos retain old weights.
                joint_valid = torch.tensor([r is None or r["joint_valid"] for r in capability_records],
                                           device=actions.device, dtype=torch.bool)
                def _joint_weights(weights):
                    return joint_valid.float() if weights is None else weights.to(actions.device) * joint_valid
                action_loss_weights = _joint_weights(action_loss_weights)
                video_loss_weights = _joint_weights(video_loss_weights)
                depth_present = joint_valid if depth_present is None else depth_present.to(actions.device).bool() & joint_valid
        capability_step = int(dropout_seed_step or 0)
        aux_kind = (CT.choose_aux(self.config, capability_step, decision is not None and decision.mode == TT.MODE_JOINT, world=world_mode)
                    if capability_enabled and not modality_dropped_prompt and not drop_video and not drop_action else None)


        # und 条件选择（prepare 级 VLM 半缓存，20260818_v2）：VLM 冻结段特征
        # 已随 training_context 预算；prompt 模态 dropout 步改用 uncond
        # token 嵌入常量（tokenizer-only、不过语言模型，und_branch
        # .get_uncond_vlm_feature；与推理 CFG uncond 前向同表示，首帧图像
        # 条件保留）。可学习 null token 仅保留给推理期"prompt 模态缺失"
        # 语义（world_model 的 drop_prompt/null_und_cache 路径不变）。
        und_precomputed = None
        if "und_vlm_hidden" in training_context:
            if drop_prompt:
                uncond_hidden, uncond_len = (
                    self.world_model.und_branch.get_uncond_vlm_feature()
                )
                und_precomputed = (
                    uncond_hidden.expand(B, -1, -1),
                    uncond_len.repeat(B),
                )
            else:
                und_precomputed = (
                    training_context["und_vlm_hidden"],
                    training_context["und_real_lens"],
                )
            drop_prompt = 0
        elif drop_prompt:
            # 兼容路径（training_context 非本类 prepare 产物）：退回空文本
            # 消息，由 encode 处理（encode 遇空消息同样路由到 uncond 常量）。
            masked_vlm_messages = [UNCONDITIONAL_VLM_MESSAGE] * B
            drop_prompt = 0

        # Random timesteps and sigma（每次调用重新采样：复用同一 latent 的多次
        # 独立参数更新各自使用不同的噪声强度）
        scheduler_start = time.perf_counter()
        if bool(getattr(self.config, "independent_modality_noise", False)):
            # 独立噪声（评审 #4 决策 4）：video/action 各自独立采样 timestep/σ，
            # 同批两模态处于不同噪声水平。t 取 video 侧（world_model 主 t 参数，
            # uni 实际按 t_video/t_action 分别构建时间标签）。
            # D1：action_flow_shift 生效时 action 从独立 scheduler 采样
            # （per-modality timeshift，video=6/action=1）。
            t_video, sigma, _ = self._sample_flow_timestep(B)
            t_action, _, sigma_action = self._sample_flow_timestep(
                B, scheduler=self.fm_train_scheduler_action)
            t = t_video
        else:
            # 共享噪声（旧行为）：video/action 同一 timestep/σ。
            t_video, sigma, sigma_action = self._sample_flow_timestep(B)
            t_action = t_video.clone()
            t = t_video
        # 零 action loss 样本的 σ 强制最大噪声端点（20260816 引入，默认关；
        # 用户拍板本轮暂缓使用、机制保留备用——开关
        # model.wm.zero_action_loss_force_max_noise）：开启时
        # action_loss_weights<=0 的样本（action_loss_weight=0 的源、训练侧
        # action-guard 置零的离群样本）action σ 强制为 1.0（t 标签 1000），
        # action encoder 输入为纯噪声，无 loss 样本的真实动作不再经 uni
        # cross-attn 泄漏为 video/und 条件。注意共享噪声分支 sigma_action 是
        # sigma 的视图，必须先 clone 再改写，避免污染 video σ。
        if (
            bool(getattr(self.config, "zero_action_loss_force_max_noise", False))
            and action_loss_weights is not None
            and not world_mode
        ):
            no_action_loss = (
                action_loss_weights.reshape(B).to(device=sigma_action.device) <= 0
            )
            if bool(no_action_loss.any()):
                sigma_action = sigma_action.clone()
                sigma_action[no_action_loss] = 1.0
                t_action = t_action.clone()
                t_action[no_action_loss] = 1000.0
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["scheduler_sample_seconds"] = time.perf_counter() - scheduler_start

        noise_start = time.perf_counter()
        # sigma/sigma_action 已由 _sample_flow_timestep 采样（独立噪声时两模态
        # 各自独立）：sigma [B,1,1,1,1] 用于 video，sigma_action [B,1,1] 用于 action。

        # fp32 采样噪声再 cast 到 bf16（评审 16e：bf16 尾数仅 8 bit，直接
        # 以 bf16 采样会损失噪声精度；惯例为 fp32 采样后 cast）。
        video_noise = torch.randn_like(clean_full_latent, dtype=torch.float32).to(self.dtype)
        if capability_enabled and world_mode and self.config.capability_a2v_weight > 0:
            # Matched flow noise makes the predicted difference attributable to
            # the recorded actions/outcomes, rather than two unrelated noise draws.
            t_video = CT.share_pair_values(t_video, pair_index, pair_valid)
            t = t_video
            sigma = CT.share_pair_values(sigma, pair_index, pair_valid)
            video_noise = CT.share_pair_values(video_noise, pair_index, pair_valid)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # 首帧是干净条件帧（teacher forcing）：不加噪，与推理侧
        # （video_latent[:, :, 0:1] = condition_frame_latent）及 Wan2.2 TI2V
        # 官方实现一致；其 token 时间标签为 0（见 video_branch/uni_branch），
        # loss 只统计非首帧 latent（见下方 video_loss）。
        noisy_video_latent[:, :, 0:1] = clean_full_latent[:, :, 0:1]
        video_target = video_noise - clean_full_latent

        # Depth follows video sigma, but rgb_only also noises/supervises frame 0.
        # Only the legacy first_frame mode clamps a clean depth condition.
        noisy_depth_latent = None
        depth_target = None
        if clean_depth_latent is not None:
            # depth_share_video_noise：与 RynnWorld-4D 一致复用 video 的噪声
            # （noise_depth = noise_video.clone()）。depth 与 video 同 VAE、同
            # shape、同 σ，共用噪声让两条流走同一条去噪轨迹——depth 读 video 时
            # 它看到的带噪 RGB 就是自己目标的直接提示；独立采样则等于让 depth
            # 去拟合一个与所见 RGB 无关的噪声实现。
            if bool(getattr(self.config, "depth_share_video_noise", False)):
                depth_noise = video_noise
            else:
                depth_noise = torch.randn_like(clean_depth_latent, dtype=torch.float32).to(self.dtype)
            if capability_enabled and world_mode and self.config.capability_a2v_weight > 0:
                depth_noise = CT.share_pair_values(depth_noise, pair_index, pair_valid)
            noisy_depth_latent = clean_depth_latent * (1 - sigma) + depth_noise * sigma
            if depth_first_frame_conditioned:
                noisy_depth_latent[:, :, 0:1] = clean_depth_latent[:, :, 0:1]
            depth_target = depth_noise - clean_depth_latent

        # Action pipeline（fp32 采样噪声再 cast 到 bf16，与 video 噪声同惯例，
        # 评审 16e）
        # 动作目标、噪声及混合保持 FP32；主干前向仍由 autocast 使用 BF16。
        action_noise = torch.randn_like(actions, dtype=torch.float32)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        if action_dim_mask is not None:
            # 无效列（统一空间中该机器人未注册/被摘除的补 0 列）不进噪声输入
            # （修复 #9a）：与 scatter 零初始化、loss 有效维 mask 一致，避免
            # robotwin 50 列/agibot 38 列纯噪声作为 encoder 输入。
            noisy_actions = noisy_actions * action_dim_mask.unsqueeze(1).to(noisy_actions.dtype)
        action_target = action_noise - actions
        supervision = None
        plan = None
        video_in, depth_in, action_in = noisy_video_latent, noisy_depth_latent, noisy_actions
        if decision is not None and decision.request.packed:
            # clean 副本取 GT、TARGET 取带噪；输出再按 TARGET 位置还原到窗口长度。
            plan = TT.pack_plan(decision.request)
            supervision = TT.packed_supervision(plan, actions.shape[1])
            video_in = TT.pack(clean_full_latent, noisy_video_latent, plan.video, dim=2)
            if noisy_depth_latent is not None:
                depth_in = TT.pack(clean_depth_latent, noisy_depth_latent, plan.depth, dim=2)
            action_in = TT.pack(actions.to(noisy_actions.dtype), noisy_actions, plan.action, dim=1)
        elif decision is not None:
            supervision = TT.write_back_clean(
                decision.request, noisy_video_latent, clean_full_latent,
                noisy_depth_latent, clean_depth_latent, noisy_actions, actions)
        # Replay only PREVIOUS forwards' errors into the clean RGB copies.
        # Joint may replay paired action errors into CLEAN copies only. Never
        # mutate training_context, noisy targets, real history/current RGB,
        # A2V commands, depth or supervision.
        er = None
        er_action_valid = None
        er_metrics = prefix_error_metrics()
        if (self.training and self.config.rgb_prefix_error_enable and plan is not None
                and decision.mode in (TT.MODE_JOINT, TT.MODE_WORLD) and not drop_video
                and self.config.video_loss_weight > 0):
            er = getattr(self, "_rgb_prefix_error_buffer", None)
            if er is None:
                er = self._rgb_prefix_error_buffer = RGBPrefixErrorBuffer(self.config)
            er_groups = [((str(source_datasets[b]) if source_datasets is not None else "unknown"),
                          (str(robot_names[b]) if robot_names is not None else "unknown"))
                         for b in range(B)]
            er_valid = ([True] * B if video_loss_weights is None else
                        (torch.isfinite(video_loss_weights) & (video_loss_weights > 0))
                        .reshape(B).detach().cpu().tolist())
            er_step = (int(dropout_seed_step) if dropout_seed_step is not None
                       else sum(er.seen.values()))
            if self.config.joint_action_prefix_error_enable and decision.mode == TT.MODE_JOINT:
                er_action_valid = torch.ones_like(actions, dtype=torch.bool)
                if action_dim_mask is not None:
                    er_action_valid &= action_dim_mask.to(actions.device).bool().unsqueeze(1)
                if action_validity_mask is not None:
                    er_action_valid &= action_validity_mask.to(actions.device).bool()
                if self.config.action_loss_weight <= 0:
                    er_action_valid.zero_()
                if action_loss_weights is not None:
                    weights = action_loss_weights.to(actions.device).reshape(B,1,1)
                    er_action_valid &= torch.isfinite(weights) & (weights > 0)
                elif source_datasets is not None:
                    for b, source in enumerate(source_datasets):
                        if str(source).lower() == "ego4d":
                            er_action_valid[b] = False
            video_in, action_in, er_metrics = er.perturb_paired(
                video_in, plan.video, decision.mode, er_step, er_groups, er_valid,
                action_packed=action_in, action_entries=plan.action, action_valid=er_action_valid)
        if collect_timing:
            _record_tensor_finite_stats(timing, "sigma", sigma)
            _record_tensor_finite_stats(timing, "video_noise", video_noise)
            _record_tensor_finite_stats(timing, "noisy_video_latent", noisy_video_latent)
            _record_tensor_finite_stats(timing, "video_target", video_target)
            _record_tensor_finite_stats(timing, "input_actions", actions)
            _record_tensor_finite_stats(timing, "noisy_actions", noisy_actions)
            _record_tensor_finite_stats(timing, "action_target", action_target)
        if collect_timing and timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if collect_timing:
            timing["noise_and_condition_seconds"] = time.perf_counter() - noise_start

        # 3. MoT forward
        world_model_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # depth 第三生成流：clean_depth_latent 在场时喂带噪 depth（跟随 video
            # σ/t），depth_read_by_action / depth_read_by_video 由 config 控制；
            # 缺 depth 数据时 noisy_depth_latent=None，world_model 跳过 depth 路径。
            forward_kwargs = dict(
                t=t, vlm_input_message=masked_vlm_messages, vlm_input_first_frame=first_frame_pils,
                video=None if drop_video else video_in,
                action=None if drop_action else action_in,
                t_video=t_video, t_action=t_action,
                robot_names=robot_names,
                drop_prompt=bool(drop_prompt),
                und_precomputed=und_precomputed,
                depth=None if drop_depth else depth_in, t_depth=t_video,
                depth_first_frame_conditioned=depth_first_frame_conditioned,
                depth_read_by_action=bool(getattr(self.config, "depth_read_by_action", False)),
                depth_detach_video_kv=bool(getattr(self.config, "depth_detach_video_kv", False)),
                depth_read_by_video=bool(getattr(self.config, "depth_read_by_video", False)),
                depth_read_by_und=bool(getattr(self.config, "depth_read_by_und", False)),
                video_read_by_action=bool(getattr(self.config, "video_read_by_action", True)),
                # action 被模态 dropout 掉时 world_model 内部会一并忽略 state
                # （has_state_token 要求 action 在场），不会留下孤立 state token。
                state=scattered_state,
                timing=timing if collect_timing else None,
                timing_sync_cuda=timing_sync_cuda,
                timing_level=int(getattr(self.config, "timing_level", 1)),
                temporal=None if decision is None else decision.request,
                history=(training_context.get("history")
                         if decision is not None and not drop_video else None),
                recovery=(None if recovery_features is None
                          else {"features": recovery_features, "valid": recovery_valid}),
            )
            aux_rng = None
            if aux_kind is not None:
                aux_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(actions.device) if actions.is_cuda else None)
            video_pred, action_pred, depth_pred = CT.forward_world(self.world_model, forward_kwargs)
            if plan is not None:
                if video_pred is not None:
                    video_pred = TT.unpack_targets(video_pred, plan.video, clean_full_latent.shape[2], 2)
                if depth_pred is not None:
                    depth_pred = TT.unpack_targets(depth_pred, plan.depth, clean_depth_latent.shape[2], 2)
                # 世界模式无动作目标：保留打包输出，供下方 ×0 维持计算图。
                if action_pred is not None and not world_mode:
                    action_pred = TT.unpack_targets(action_pred, plan.action, actions.shape[1], 1)
            if collect_timing:
                _record_tensor_finite_stats(timing, "video_pred", video_pred)
                _record_tensor_finite_stats(timing, "action_pred", action_pred)
            if collect_timing and timing_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            if collect_timing:
                timing["world_model_forward_seconds"] = time.perf_counter() - world_model_start


            loss_prepare_start = time.perf_counter()
            # Prepare predictions and targets
            action_pred_masked = action_pred.clone() if action_pred is not None else None
            action_target_masked = action_target.clone()

            # Calculate losses
            # fixme:
            #k = self._video_action_freq_ratio - 1
            k = 0
            # video loss 只在非首帧 latent 上统计（评审 16f）：首帧是干净条件帧
            # （teacher forcing），此前把首帧 pred/target 置零再求全序列 mean，
            # 分母混入 1/(F) 个零项，损失被系统性缩到 (F-1)/F；直接在切片上算。
            if video_pred is None:
                # 模态 dropout：video 被 drop，无预测，loss 置零（保持 [B] 形状，
                # 下游 per-sample 统计与总损失公式不变；无梯度路径，符合意图）。
                video_loss_per_sample = video_target[:, :, 1:].new_zeros(B)
                video_loss = video_loss_per_sample.sum()
            else:
                if supervision is not None:
                    video_loss_per_sample = TT.frame_mse(
                        video_pred, video_target, supervision.video_frames)
                else:
                    video_loss_per_sample = torch.nn.functional.mse_loss(
                        video_pred[:, :, 1:],
                        video_target[:, :, 1:],
                        reduction='none',
                    ).mean(dim=(1, 2, 3, 4))
                if collect_timing:
                    _record_tensor_finite_stats(timing, "video_pred_masked", video_pred[:, :, 1:])
                    _record_tensor_finite_stats(timing, "video_target_masked", video_target[:, :, 1:])
                    _record_tensor_finite_stats(timing, "video_loss_per_sample", video_loss_per_sample)
                if video_loss_weights is not None:
                    video_loss_weights = video_loss_weights.to(
                        device=video_loss_per_sample.device,
                        dtype=video_loss_per_sample.dtype,
                    )
                    video_weight_sum = video_loss_weights.sum()
                    if video_weight_sum > 0:
                        video_loss = (video_loss_per_sample * video_loss_weights).sum() / video_weight_sum
                    else:
                        video_loss = video_loss_per_sample.sum() * 0.0
                else:
                    video_loss = video_loss_per_sample.mean()
            if action_pred is None:
                # 模态 dropout：action 被 drop，loss 置零（保持 [B] 形状）。
                action_loss_per_sample = action_target_masked.new_zeros(B)
                action_loss = action_loss_per_sample.sum()
            elif world_mode:
                # 动作是给定命令：无 action 监督；×0 保持计算图（各 rank 参数集一致）。
                action_loss_per_sample = action_pred_masked.float().sum(dim=(1, 2)) * 0.0
                action_loss = action_loss_per_sample.sum()
            else:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_pred_masked[:, k:, :],
                    action_target_masked[:, k:, :],
                    reduction='none',
                )
                if supervision is not None:
                    action_loss_per_sample = TT.masked_step_mean(
                        action_loss_per_sample, supervision.action_steps,
                        action_dim_mask, action_validity_mask)
                elif action_dim_mask is not None:
                    # 统一 action 空间：只对每样本有效维（robot 注册 dim 所在列）
                    # 统计损失，其余补 0 列不计入；逐样本保持 [B] 形状
                    # （下游权重逻辑不变）。
                    _t_mask = action_loss_per_sample.shape[1]
                    if action_validity_mask is not None:
                        # 逐帧有效性门控（20260816，egoscale 无效手）：与静态
                        # 有效维 mask 合并为 [B,T,U]，被屏蔽帧/列不计入 loss，
                        # 分母同步缩小（全屏蔽样本退化为 0 loss）。
                        frame_mask = action_dim_mask.unsqueeze(1) & action_validity_mask
                        action_loss_per_sample = (
                            (action_loss_per_sample * frame_mask).sum(dim=(1, 2))
                            / frame_mask.sum(dim=(1, 2)).clamp(min=1)
                        )
                    else:
                        action_loss_per_sample = (
                            (action_loss_per_sample * action_dim_mask.unsqueeze(1)).sum(dim=(1, 2))
                            / (action_dim_mask.sum(dim=1) * _t_mask).clamp(min=1)
                        )
                else:
                    action_loss_per_sample = action_loss_per_sample.mean(dim=(1, 2))
                if collect_timing:
                    _record_tensor_finite_stats(timing, "action_pred_masked", action_pred_masked)
                    _record_tensor_finite_stats(timing, "action_target_masked", action_target_masked)
                    _record_tensor_finite_stats(timing, "action_loss_per_sample", action_loss_per_sample)
                if action_loss_weights is not None:
                    action_loss_weights = action_loss_weights.to(
                        device=action_loss_per_sample.device,
                        dtype=action_loss_per_sample.dtype,
                    )
                    weight_sum = action_loss_weights.sum()
                    if weight_sum > 0:
                        action_loss = (action_loss_per_sample * action_loss_weights).sum() / weight_sum
                    else:
                        action_loss = action_loss_per_sample.sum() * 0.0
                elif source_datasets is not None:
                    fallback_action_loss_weights = torch.tensor(
                        [0.0 if str(source_dataset).lower() == "ego4d" else 1.0 for source_dataset in source_datasets],
                        device=action_loss_per_sample.device,
                        dtype=action_loss_per_sample.dtype,
                    )
                    weight_sum = fallback_action_loss_weights.sum()
                    if weight_sum > 0:
                        action_loss = (action_loss_per_sample * fallback_action_loss_weights).sum() / weight_sum
                    else:
                        action_loss = action_loss_per_sample.sum() * 0.0
                else:
                    action_loss = action_loss_per_sample.mean()
            action_loss_per_sample_detached = action_loss_per_sample.detach()

            # rgb_only supervises current+future valid latent cells; legacy mode
            # excludes its conditioned frame 0. Empty samples do not dilute loss.
            depth_edge_per_sample = video_target[:, :, 1:].new_zeros(B)
            if depth_pred is None:
                depth_loss_per_sample = video_target[:, :, 1:].new_zeros(B)
                depth_loss = depth_loss_per_sample.sum()
            else:
                depth_start = 1 if depth_first_frame_conditioned else 0
                d_idx = None if supervision is None else supervision.depth_frames.to(depth_pred.device)
                contiguous = d_idx is None or (
                    d_idx.numel() > 0 and bool((d_idx == torch.arange(
                        int(d_idx[0]), int(d_idx[0]) + d_idx.numel(), device=d_idx.device)).all()))
                if contiguous:
                    # Slicing (not index_select) keeps the legacy reduction bit-exact.
                    lo = depth_start if d_idx is None else int(d_idx[0])
                    hi = None if d_idx is None else lo + d_idx.numel()
                    pick = lambda x: x[:, :, lo:hi]  # noqa: E731
                else:
                    pick = lambda x: x.index_select(2, d_idx.to(x.device))  # noqa: E731
                pred_slice = pick(depth_pred)
                target_slice = pick(depth_target)
                depth_has_valid = None
                if latent_depth_mask is not None:
                    depth_loss_per_sample, depth_has_valid = masked_depth_mse(
                        pred_slice, target_slice, pick(latent_depth_mask))
                else:
                    # Legacy unmasked first_frame path: preserve its exact reduction.
                    err = torch.nn.functional.mse_loss(
                        pred_slice, target_slice, reduction='none')
                    depth_loss_per_sample = err.mean(dim=(1, 2, 3, 4))
                if depth_motion_weight is not None:
                    # 按 /N 归一（不是 /Σw）：每个 latent cell 的有效放大是
                    # 1 + λ·w，与 mask 大小无关。除以 Σw 会让 mask 越小放大
                    # 越猛（Ego pose mask frac 0.013 → 中位 13×），λ 失去意义。
                    lam = float(getattr(self.config, "depth_motion_lambda", 0.3))
                    w = depth_motion_weight[:, :, 1:].to(
                        device=err.device, dtype=err.dtype
                    )
                    mot = (err * w).mean(dim=(1, 2, 3, 4))
                    depth_loss_per_sample = depth_loss_per_sample + lam * mot
                if bool(getattr(self.config, "depth_edge_loss_enable", False)):
                    # x_t = (1-σ)x0 + σ ε，v = ε - x0 ⇒ x0 = x_t - σ v
                    x0_hat = noisy_depth_latent - sigma * depth_pred
                    depth_edge_per_sample = depth_edge_loss_per_sample(
                        x0_hat[:, :, 1:],
                        clean_depth_latent[:, :, 1:],
                        persist=bool(getattr(self.config, "depth_edge_persist", True)),
                    )
                    lam_e = float(getattr(self.config, "depth_edge_lambda", 0.5))
                    depth_loss_per_sample = (
                        depth_loss_per_sample + lam_e * depth_edge_per_sample
                    )
                if collect_timing:
                    _record_tensor_finite_stats(timing, "depth_pred_masked", depth_pred[:, :, 1:])
                    _record_tensor_finite_stats(timing, "depth_target_masked", depth_target[:, :, 1:])
                    _record_tensor_finite_stats(timing, "depth_loss_per_sample", depth_loss_per_sample)
                if depth_present is not None or depth_has_valid is not None:
                    depth_weight = (
                        depth_present.to(device=depth_loss_per_sample.device,
                                         dtype=depth_loss_per_sample.dtype).reshape(B)
                        if depth_present is not None else torch.ones_like(depth_loss_per_sample)
                    )
                    if depth_has_valid is not None:
                        depth_weight = depth_weight * depth_has_valid.to(depth_weight.dtype)
                    depth_weight_sum = depth_weight.sum()
                    if depth_weight_sum > 0:
                        depth_loss = (depth_loss_per_sample * depth_weight).sum() / depth_weight_sum
                    else:
                        depth_loss = depth_loss_per_sample.sum() * 0.0
                else:
                    depth_loss = depth_loss_per_sample.mean()

            block_metrics = {}
            if supervision is not None and return_dict:
                block_metrics = TT.block_loss_metrics(
                    decision.mode, decision.request.grid, supervision,
                    video_target=video_target, video_pred=video_pred,
                    action_err=(None if action_pred is None or world_mode else
                                (action_pred_masked.float() - action_target_masked.float()).square()),
                    action_dim_mask=action_dim_mask, action_validity=action_validity_mask,
                    depth_pred=depth_pred,
                    depth_target=depth_target if depth_pred is not None else None,
                    depth_mask=latent_depth_mask if depth_pred is not None else None)

            capability_loss, capability_metrics = (video_loss.new_zeros(()), {})
            if capability_enabled:
                capability_loss, capability_metrics = CT.extra_losses(
                    self.config, training_context, decision, plan,
                    (video_pred, action_pred, depth_pred), (video_target, action_target, depth_target),
                    forward_kwargs, self.world_model, aux_kind, aux_rng,
                    pair_index, pair_valid, sigma, sigma_action,
                    action_loss_weights, video_loss_weights, depth_present)
            capability_metrics.update(CT.history_metrics(
                training_context, getattr(self.world_model, "history_encoder", None)))
            total_loss = (
                capability_loss + self.config.video_loss_weight * video_loss +
                self.config.action_loss_weight * action_loss +
                self.config.depth_loss_weight * depth_loss
            )
            if collect_timing and timing_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            if collect_timing:
                timing["loss_compute_seconds"] = time.perf_counter() - loss_prepare_start

        # Only after the forward: same-step prediction errors cannot leak into
        # their own inputs. No extra model rollout or backward through replay.
        if er is not None and video_pred is not None:
            er.observe(noisy_video_latent, video_pred, clean_full_latent, sigma,
                       plan.video, decision.mode, er_groups, er_valid,
                       action_noisy=noisy_actions, action_velocity=action_pred,
                       action_clean=actions, action_sigma=sigma_action,
                       action_entries=plan.action, action_valid=er_action_valid)
            er_metrics.update(er.metrics())
            if er_metrics["rgb_prefix_error_applied_blocks"]:
                tag = TT._MODE_TAGS.get(decision.mode, str(decision.mode))
                for name in ("mean_rms_ratio", "raw_rms_ratio", "clipped_frac"):
                    er_metrics[f"prefix_error_{tag}/{name}"] = er_metrics["rgb_prefix_error_" + name]
        
        if return_dict:
            action_loss_per_sample_argmax = int(action_loss_per_sample_detached.argmax().item())
            video_loss_per_sample_detached = video_loss_per_sample.detach()
            depth_loss_per_sample_detached = depth_loss_per_sample.detach()
            return {
                **er_metrics,
                **block_metrics,
                **capability_metrics,
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'depth_loss': depth_loss,
                'depth_loss_per_sample_mean': depth_loss_per_sample_detached.mean().item(),
                'depth_loss_per_sample_max': depth_loss_per_sample_detached.max().item(),
                'depth_edge_loss': float(depth_edge_per_sample.detach().float().mean().item()),
                'depth_motion_frac': (
                    float(depth_motion_frac.detach().float().mean().item())
                    if depth_motion_frac is not None
                    else 0.0
                ),
                'depth_motion_pose_used': (
                    float(depth_motion_pose_used.detach().float().mean().item())
                    if depth_motion_pose_used is not None
                    else 0.0
                ),
                'action_loss_per_sample_mean': action_loss_per_sample_detached.mean().item(),
                'action_loss_per_sample_max': action_loss_per_sample_detached.max().item(),
                'action_loss_per_sample_argmax': action_loss_per_sample_argmax,
                'video_loss_per_sample_mean': video_loss_per_sample_detached.mean().item(),
                'video_loss_per_sample_max': video_loss_per_sample_detached.max().item(),
                'video_loss_per_sample_argmax': int(video_loss_per_sample_detached.argmax().item()),
                'video_timestep_mean': t_video.float().mean().item(),
                'action_timestep_mean': t_action.float().mean().item(),
                'modality_dropped_video': drop_video,
                'modality_dropped_action': drop_action,
                'modality_dropped_depth': drop_depth,
                # batch 级 prompt drop 记录原始决策（修复 #9d 后消息被替换为
                # 空文本、drop_prompt 变量已清零，不能直接用）。
                'modality_dropped_prompt': modality_dropped_prompt,
                'temporal_cut': 0 if decision is None else int(decision.request.cut),
                'temporal_world_mode': int(world_mode),
                'temporal_a2v_mode': int(world_mode),
                'temporal_joint_mode': int(decision is not None and decision.mode == TT.MODE_JOINT),
                'temporal_idm_mode': int(decision is not None and decision.mode == TT.MODE_IDM),
                'temporal_mode_id': (0 if decision is None or decision.mode == TT.MODE_JOINT
                                     else 1 if decision.mode == TT.MODE_IDM else 2),
            }

    def _sample_flow_timestep(
        self, B: int,
        scheduler: Optional[FlowMatchScheduler] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样一个 flow-matching timestep（CPU/GPU 均可，评审 #4 独立噪声用）。

        Args:
            B: batch 大小。
            scheduler: 采样用的 FlowMatchScheduler（D1，20260811）；None =
                self.fm_train_scheduler（video 侧，shift=flow_shift），
                action 侧传 self.fm_train_scheduler_action（shift=
                action_flow_shift；未构建时同为 None 回退）。

        Returns:
            t [B, 1]：time_embedding 标签（scheduler.timesteps 数组，
                σ×1000，与推理 σ 网格共用同一 shift）。
            sigma [B, 1, 1, 1, 1]：video latent 加噪强度。
            sigma_action [B, 1, 1]：action 加噪强度（同 sigma 的视图）。
        """
        if scheduler is None:
            scheduler = self.fm_train_scheduler
        timestep_id = torch.randint(0, scheduler.num_train_timesteps, (B, 1))
        # 保留 scheduler 时间标签精度；sinusoidal embedding 内升精度无法
        # 恢复已被 BF16 舍入的信息。训练/推理标签统一使用 FP32。
        t = scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=self.device)  # [B, 1]
        sigma_fp32 = scheduler.sigmas[timestep_id].to(
            dtype=torch.float32, device=self.device)
        sigma = sigma_fp32.to(self.dtype).view(B, 1, 1, 1, 1)
        # action 对数值精度更敏感，避免 BF16 σ 在小动作区间的量化误差。
        sigma_action = sigma_fp32.view(B, 1, 1)
        return t, sigma, sigma_action

    # 模态 dropout 采样基种子：固定常数即可，作用仅是配合采样序号让各 rank
    # 逐次决策一致（见 __init__ 计数器注释）。
    _MODALITY_DROPOUT_SEED_BASE = 20260807

    def _sample_modality_dropout(
        self,
        p_video: float,
        p_action: float,
        p_prompt: float,
        p_depth: float = 0.0,
        training: bool = True,
        seed_step: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """采样模态 dropout mask（batch 级、每 update step 重采、跨 rank 一致；CPU 可测）。

        video/action/prompt 三模态各自独立伯努利；video 与 action 同时命中时
        随机保底保留一个可训练模态（prompt 缺失可由 null token 替代，无需
        保底）。depth 为辅助生成流，独立伯努利、不参与保底。
        返回 (drop_video, drop_action, drop_prompt, drop_depth) 四个 0/1。

        跨 rank 一致性：用 ``(基种子 + 采样序号)`` 定种子的专用 generator，
        不消耗全局 RNG；各 rank 采样次数相同 → 决策逐次相同。序号只在训练
        态递增，非训练态恒返回全 0 且不影响序号。

        seed_step（修复 #9b）：传入 trainer 的 global_step 时以其为采样序号
        ——决策变为 global_step 的纯函数，弹性重启/续训后各 rank 不会分叉
        （内部计数器不进 checkpoint，重启后曾存在跨 rank 决策分叉导致
        ZeRO-1 梯度桶尺寸不一致死锁的风险）。缺省 None 回退内部计数器。
        """
        if not training:
            return 0, 0, 0, 0
        sample_idx = (
            int(seed_step) if seed_step is not None else self._modality_dropout_sample_idx
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(
            self._MODALITY_DROPOUT_SEED_BASE * 1_000_003 + sample_idx
        )
        self._modality_dropout_sample_idx += 1
        drop_video = int(torch.rand((), generator=gen) < float(p_video))
        drop_action = int(torch.rand((), generator=gen) < float(p_action))
        drop_prompt = int(torch.rand((), generator=gen) < float(p_prompt))
        if drop_video and drop_action:  # 至少保留一个可训练模态
            if torch.rand((), generator=gen) < 0.5:
                drop_video = 0
            else:
                drop_action = 0
        # depth 在既有三模态决策之后再抽一次（p_depth=0 时该抽样不影响
        # video/action/prompt 的决策，保持 enable_depth 前的逐字节行为）。
        drop_depth = int(torch.rand((), generator=gen) < float(p_depth))
        return drop_video, drop_action, drop_prompt, drop_depth

    @staticmethod
    def _modality_missing(x: Any) -> bool:
        """推理模态缺失判定：x 为 None 或张量全 0（评审 #4：首帧/state 全 0
        视为该模态缺失，不输入网络、不生成该模态）。非张量一律视为在场
        （prompt 文本缺失由调用处另行判定）。"""
        if x is None:
            return True
        if not torch.is_tensor(x):
            return False
        return bool(x.numel() == 0 or x.abs().amax() == 0)

    def _inference_sigmas(self, num_inference_steps: int, shift: Optional[float] = None) -> torch.Tensor:
        """推理 σ 网格：与训练一致的 shift 变换。

        s 取 [1, 0] 均匀网格，σ = shift·s / (1 + (shift-1)·s)（末点为 0），
        与 FlowMatchScheduler.set_timesteps 的变换一致；时间标签 = σ×1000，
        与训练时喂给 time_embedding 的标签同分布。shift 缺省 =
        config.flow_shift（video 侧）；action 侧传 self._action_flow_shift
        （D1，per-modality timeshift）。
        """
        shift = float(self.config.flow_shift) if shift is None else float(shift)
        s = torch.linspace(1.0, 0.0, num_inference_steps + 1,
                           device=self.device, dtype=torch.float32)
        return shift * s / (1.0 + (shift - 1.0) * s)

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        first_frame_pils: Optional[List[PIL.Image.Image]] = None,
        t5_input_messages: Optional[List[str]] = None,
        vlm_input_messages: Optional[List[str]] = None,
        robot_names: Optional[List[str]] = None,
        guidance_scale: float = 1.0,
        action_guidance_scale: float = 1.0,
        action_num_inference_steps: Optional[int] = None,
        action_smooth_alpha: float = 0.0,
        depth_smooth_alpha: float = 0.0,
        predict_action: Optional[bool] = None,
        depth_first: Optional[torch.Tensor] = None,   # [B, 3, H, W] in [0,1]，首帧深度条件
        return_depth: bool = False,
        decode_outputs: bool = True,
        condition_video_frames: Optional[torch.Tensor] = None,  # [B, T, C, H, W] [0,1] GT 未来帧
        condition_actions: Optional[torch.Tensor] = None,  # [B, chunk, dim] GT action（原始维）
        condition_depth_frames: Optional[torch.Tensor] = None,  # [B, T, C, H, W] [0,1] GT 未来深度
        init_video_noise: Optional[torch.Tensor] = None,
        init_action_noise: Optional[torch.Tensor] = None,
        init_depth_noise: Optional[torch.Tensor] = None,
        noise_capture: Optional[dict] = None,
        return_depth_first_frame: bool = False,
        temporal_request=None,
        history_context: Optional[Dict[str, Any]] = None,
        history_inputs: Optional[Dict[str, torch.Tensor]] = None,
        recovery_features: Optional[torch.Tensor] = None,
        recovery_valid: Optional[torch.Tensor] = None,
        latent_capture: Optional[dict] = None,
        num_blocks: Optional[int] = None,
        prefix_cache_mode: str = "off",
        prefix_cache_max_mb: float = 1024.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.

        块级因果（temporal_mode=block_causal）：temporal_request 缺省为 cut=1
        的部署窗口；action_role="world" 需同时给 condition_actions（动作作为
        给定命令，只预测 RGB/depth 后果）。history_context 为 _prepare_history
        的产物（runtime 可缓存复用），或传 history_* 原始输入由本函数编码。
        ar_packed_teacher_forcing 时逐块生成：块 k 以块 1..k-1 的生成结果为干净
        前缀、各跑完整去噪调度；num_blocks（缺省 1）为生成块数，输出只含这些块
        （动作 num_blocks×块长步、视频 num_blocks×4·latents_per_block 帧）；
        condition_actions 给 num_blocks×块长步。
        latent_capture 非 None 时写入最终 video/depth latent 与统一空间
        action latent（供预测存储与误差对齐，decode_outputs=False 时也可得）。

        模态缺失语义（评审 #4 决策 3）：
        - video 缺失 = first_frame 为 None 或全 0 → 不输入视频分支、不生成视频
          （predicted_frames=None；und 的首帧图用全黑 PIL 占位）；
        - action 缺失 = state 为 None 或全 0 → predicted_actions=None；
          （缺省启发式；调用方可传 predict_action 显式覆盖，见 Args）
        - prompt 缺失 = vlm_input_messages 为 None 或全部为空字符串 → und 用
          训练用空文本表示，此时两种 guidance_scale 必须
          =1.0（CFG 无意义，直接 raise）。
        逐样本缺失状态必须 batch 内一致（不一致 raise，评测请按模态齐次分批）。
        现有评估路径传真实输入，行为不变。

        Args:
            first_frame: Initial frame [B, C, H, W]（可 None/全 0 = video 缺失）
            state: Initial robot state [B, state_dim]（可 None/全 0 = action 缺失）
            num_inference_steps: Number of denoising steps（video 侧）
            vlm_input_messages: VLM 文本消息（可 None/全空 = prompt 缺失）
            guidance_scale: classifier-free guidance 强度（评审 #6，仅作用于
                video 速度场）；1.0 = 关闭（单次条件前向，旧行为），>1 时每个
                去噪步额外跑一次无条件（空文本）前向并按
                v = v_uncond + scale·(v_cond - v_uncond) 合并。前提是训练开启了
                prompt 模态 dropout（无条件分支被训练过；20260818_v2 起
                condition dropout 已并入 modality_dropout_prompt，且无条件
                表示为 tokenizer-only 的 token 嵌入常量——本前向经
                und_branch.encode 自动命中同一常量，训练/推理一致）。
            action_guidance_scale: action 速度场的 CFG 强度（修复 #1）——
                默认 1.0 = 不对 action 施加 CFG（π0/Diffusion Policy 均不对
                action 做 CFG；弱条件下 cond−uncond 方向近似噪声，放大后是
                推理动作剧烈振荡的头号放大器）。!=1.0 仅为消融对照保留。
            action_num_inference_steps: action 独立去噪步数（修复 #9c，UWM 式
                per-modality σ 网格）；None/<1 = 与 video 共用步数（旧行为）。
                较少步数减少逐帧速度场误差的 Euler 累积（π0 action 用 10 步）。
            action_smooth_alpha: 推理时序平滑强度（修复 #3b，Diffusion Policy
                式指数平滑）：>0 时对去噪完的 action 轨迹沿时间维做
                y[t]=α·x[t]+(1−α)·y[t−1]；0 = 关闭（旧行为）。
            depth_smooth_alpha: 深度推理时序平滑，公式同 action。>0 时对 VAE
                解码后的深度图（含首帧条件）沿时间做 EMA，再丢掉条件帧；
                0 = 关闭（旧行为）。只改推理观感/指标，不影响训练。
            predict_action: action 模态的显式声明（默认 None = 上述零值启发式，
                训练与现有评估路径行为不变）；True = 即使 state 全 0 也强制
                预测 action（periodic_eval 的 EWMBenchmark 借位场景：零 state
                是占位锚点而非"无 action"）；False = 即使 state 非零也跳过
                action。state 非张量时无法预测，仍按缺失处理。
            decode_outputs: 去噪结束后是否把 RGB/depth latent 经 VAE 解成像素。
                True = 旧行为。False = 跳过 decode_video，predicted_frames /
                predicted_depth 为 None。不改变 depth latent 是否创建或联合去噪。
            return_depth: 仅控制返回第三个输出与 depth 像素解码；两个 mode 均不
                用它关闭 depth 流。first_frame 仍需 depth_first；rgb_only 禁止 GT
                depth 输入，由 RGB 条件生成所有深度帧。
            return_depth_first_frame: False 返回未来 T 帧（兼容旧长度）；True
                返回当前+未来 T+1 帧，[B,3,T+1,H,W]。需 return_depth=True 且
                decode_outputs=True 才有像素输出。训练 depth_first 仅是标签。

        Returns:
            Tuple of (predicted_frames, predicted_actions)；缺失模态返回 None。
        """
        depth_mode = depth_condition_mode(self.config)
        depth_first_frame_conditioned = depth_mode == "first_frame"
        if depth_mode == "rgb_only" and (
            depth_first is not None or condition_depth_frames is not None
        ):
            raise ValueError("rgb_only inference forbids depth_first and GT depth teacher forcing")
        # ---- 模态缺失检测（逐样本判定后要求 batch 内一致：评测按模态齐次分批）
        if first_frame is not None and torch.is_tensor(first_frame):
            B = first_frame.shape[0]
        elif state is not None and torch.is_tensor(state):
            B = state.shape[0]
        elif vlm_input_messages is not None and isinstance(vlm_input_messages, (list, tuple)):
            B = len(vlm_input_messages)
        else:
            raise ValueError(
                "inference_step: first_frame/state/vlm_input_messages 全缺失，"
                "无法确定 batch 大小"
            )
        video_missing = self._modality_missing(first_frame)
        state_missing = self._modality_missing(state)
        if predict_action is not None:
            # 显式模态声明（periodic_eval 按 batch state_action_policies 传入）
            # 优先于零值启发式：True = 零 state 占位也强制预测（EWMBenchmark
            # 借位）；False = 即使 state 非零也跳过 action。
            # state 非张量时无法预测，保持缺失。
            state_missing = (not bool(predict_action)) or (not torch.is_tensor(state))
        if first_frame is not None and torch.is_tensor(first_frame) and not video_missing:
            per_sample = (first_frame.abs().amax(dim=(1, 2, 3)) == 0)
            if bool(per_sample.any()) and not bool(per_sample.all()):
                raise ValueError(
                    "inference_step: first_frame 缺失状态 batch 内不一致"
                    "（部分样本全 0），评测请按模态齐次分批"
                )
        if (predict_action is None and state is not None
                and torch.is_tensor(state) and not state_missing):
            per_sample = (state.abs().amax(dim=1) == 0)
            if bool(per_sample.any()) and not bool(per_sample.all()):
                raise ValueError(
                    "inference_step: state 缺失状态 batch 内不一致"
                    "（部分样本全 0），评测请按模态齐次分批"
                )
        prompt_missing = (
            vlm_input_messages is None
            or (isinstance(vlm_input_messages, (list, tuple))
                and all(str(m).strip() == "" for m in vlm_input_messages))
        )
        if video_missing and state_missing and prompt_missing:
            raise ValueError(
                "inference_step: video/action/prompt 三模态全部缺失，无可用输入"
            )

    #    assert language_embeddings is None
    #    assert vlm_inputs is None

        # assert len(first_frame_pils) == 1
        # first_frame_pils[0].save('./show.jpg', "JPEG", quality=95)
        # print("t5_input_messages:",t5_input_messages)
        # print("vlm_input_messages:",vlm_input_messages)
        # print('state:',state)
        # raise ValueError
        action_cols, robot_offset, action_dim = _robot_action_columns(
            self.config, robot_names)

        block_causal = str(getattr(self.config, "temporal_mode", "bidirectional")) == "block_causal"
        if not block_causal and (temporal_request is not None or history_context is not None
                                 or history_inputs or recovery_features is not None):
            raise ValueError("temporal/history/recovery inference requires temporal_mode=block_causal")
        packed_ar = block_causal and bool(getattr(self.config, "ar_packed_teacher_forcing", False))
        from thinker_wm.models.prefix_cache import PrefixKVCache
        import math
        if prefix_cache_mode not in ("off", "fixed"):
            raise ValueError("prefix_cache_mode must be off or fixed")
        if not math.isfinite(float(prefix_cache_max_mb)) or prefix_cache_max_mb < 0:
            raise ValueError("prefix_cache_max_mb must be finite and non-negative")
        if prefix_cache_mode == "fixed" and (not packed_ar or self.training):
            raise ValueError("fixed prefix cache requires an eval-mode packed AR model")

        if num_blocks is not None and not packed_ar:
            raise ValueError("num_blocks requires ar_packed_teacher_forcing (block-by-block generation)")
        if block_causal:
            if condition_video_frames is not None and state_missing:
                raise ValueError("IDM requires state and an enabled action stream")
            if condition_video_frames is not None and condition_actions is not None:
                raise ValueError("choose IDM goal RGB or FDM fixed actions, not both")
            if temporal_request is None:
                temporal_request = TT.sample_step_decision(
                    self.config, 0, False, depth_first_frame_conditioned).request
                from dataclasses import replace
                temporal_request = replace(
                    temporal_request, action_role="world" if condition_actions is not None else "policy",
                    video_role="goal" if condition_video_frames is not None else "predict")
            if (temporal_request.video_role == "goal") != (condition_video_frames is not None):
                raise ValueError("IDM video_role='goal' requires condition_video_frames and vice versa")
            if (temporal_request.action_role == "world") != (condition_actions is not None):
                raise ValueError("FDM action_role='world' requires condition_actions and vice versa")
            if int(temporal_request.cut) != 1:
                raise ValueError("inference uses the deployment window (cut=1); past blocks enter as history")
            if bool(temporal_request.packed) != packed_ar:
                raise ValueError("temporal_request.packed differs from ar_packed_teacher_forcing")
            if packed_ar:
                n_future = temporal_request.grid.num_blocks(1 + self.config.num_video_frames // 4) - 1
                num_blocks = 1 if num_blocks is None else int(num_blocks)
                if not 1 <= num_blocks <= n_future:
                    raise ValueError(f"num_blocks must be in [1, {n_future}], got {num_blocks}")
            if temporal_request.depth_first_frame_conditioned != depth_first_frame_conditioned:
                raise ValueError("temporal_request depth conditioning differs from depth_condition_mode")
            if temporal_request.action_role == "world" and condition_actions is None:
                raise ValueError("action_role='world' requires condition_actions (the command to evaluate)")
            TT.num_target_blocks_check(temporal_request, 1 + self.config.num_video_frames // 4)
            if history_context is None and history_inputs:
                history_context = self._prepare_history(history_inputs, robot_names)
            if history_context is not None and video_missing:
                raise ValueError("history requires the RGB stream")
        recovery = (None if recovery_features is None
                    else {"features": recovery_features.to(self.device), "valid": recovery_valid})

        # und 仍需首帧图（VLM 图像输入）：video 缺失时用全黑 PIL 占位（与真实
        # 首帧同尺寸），避免改动 und_branch 的图像分支。
        und_pils = first_frame_pils
        if video_missing and und_pils is None:
            black = PIL.Image.new(
                "RGB", (self.config.video_width, self.config.video_height), (0, 0, 0))
            und_pils = [black] * B

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # 1. Video latents init
            condition_frame_latent = None
            if not video_missing:
                # Condition frame encode
                first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
                with torch.no_grad():
                    condition_frame_latent = self.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

            # 条件（首帧图 + 文本）全程不变：und token 在去噪循环外算一次并
            # 缓存，避免每个去噪步重跑冻结 VLM + und encoder（评审 #11）。
            with torch.no_grad():
                if prompt_missing:
                    # Match the trained prompt-dropout representation; do not use
                    # the separate legacy null token, which this path never trains.
                    if guidance_scale != 1.0 or action_guidance_scale != 1.0:
                        raise ValueError(
                            "inference_step: prompt 模态缺失（None/全空）时 "
                            "guidance_scale != 1.0 的 CFG 无意义，请用 guidance_scale=1.0"
                        )
                    und_cache = self.world_model.und_branch.encode(
                        [UNCONDITIONAL_VLM_MESSAGE] * B, und_pils)
                    und_cache_uncond = None
                else:
                    und_cache = self.world_model.und_branch.encode(
                        vlm_input_messages, und_pils)
                    # CFG（评审 #6）：guidance_scale != 1.0 时再编码一份无条件
                    # （空文本）und token，去噪步内做 cond/uncond 双前向合并。
                    und_cache_uncond = None
                    if guidance_scale != 1.0 or action_guidance_scale != 1.0:
                        und_cache_uncond = self.world_model.und_branch.encode(
                            [UNCONDITIONAL_VLM_MESSAGE] * B, und_pils)

            # Branch presence is independent of output formatting. rgb_only
            # creates depth from noise without any depth observation.
            tf_depth = (
                condition_depth_frames is not None
                and torch.is_tensor(condition_depth_frames)
            )
            predict_depth = bool(
                getattr(self.config, "enable_depth", False)
                and not video_missing
                and (depth_mode == "rgb_only" or torch.is_tensor(depth_first))
            )
            tf_depth = bool(tf_depth and predict_depth)
            depth_condition_latent = None
            if predict_depth and depth_first_frame_conditioned:
                depth_first_norm = (depth_first * 2.0 - 1.0).unsqueeze(2).to(self.dtype)
                with torch.no_grad():
                    depth_condition_latent = self.encode_video(depth_first_norm)

            # Init video/action latents（fp32 采样噪声再 cast，评审 16e）
            video_latent = None
            if not video_missing:
                B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
                num_total_latent_frames = 1 + self.config.num_video_frames // 4
                if init_video_noise is not None:
                    video_latent = init_video_noise.to(device=self.device, dtype=self.dtype).clone()
                else:
                    video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=torch.float32).to(self.dtype)
                if noise_capture is not None:
                    # 首帧条件写回之前的噪声。重放时走同一段写回，不能存写回之后的张量。
                    noise_capture["video_noise"] = video_latent.detach().clone()
            depth_latent = None
            if predict_depth:
                # 必须与训练一致：depth_share_video_noise 时复用同一份噪声再
                # 各自盖上首帧条件。训练是 depth_noise = video_noise；旧推理
                # 用 randn_like 另抽一份，捷径在推理失效，越训定集越差。
                if bool(getattr(self.config, "depth_share_video_noise", False)):
                    depth_latent = video_latent.clone()
                elif init_depth_noise is not None:
                    depth_latent = init_depth_noise.to(device=self.device, dtype=self.dtype).clone()
                else:
                    depth_latent = torch.randn_like(video_latent, dtype=torch.float32).to(self.dtype)
                if (
                    noise_capture is not None
                    and not bool(getattr(self.config, "depth_share_video_noise", False))
                ):
                    noise_capture["depth_noise"] = depth_latent.detach().clone()
            if video_latent is not None:
                video_latent[:, :, 0:1] = condition_frame_latent
            if depth_latent is not None and depth_first_frame_conditioned:
                depth_latent[:, :, 0:1] = depth_condition_latent
            # Teacher-force 未来 RGB：整段 VAE latent 钉死为 GT（t_video=0），
            # 只让 depth 读真视频。默认 None = 旧的联合去噪。
            tf_video = (
                condition_video_frames is not None
                and torch.is_tensor(condition_video_frames)
                and not video_missing
                and condition_frame_latent is not None
            )
            if block_causal and condition_video_frames is not None:
                expected_frames = (num_blocks * temporal_request.grid.latents_per_block * 4
                                   if packed_ar else self.config.num_video_frames)
                expected = (B, expected_frames, 3, self.config.video_height, self.config.video_width)
                if not torch.is_tensor(condition_video_frames) or tuple(condition_video_frames.shape) != expected:
                    raise ValueError(f"IDM goal video must be {expected} future RGB frames")
                if not torch.isfinite(condition_video_frames).all() or not ((condition_video_frames >= 0) & (condition_video_frames <= 1)).all():
                    raise ValueError("IDM goal RGB must be finite and in [0, 1]")
                if not tf_video:
                    raise ValueError("IDM requires current RGB and the video stream")
            if tf_video:
                vf = condition_video_frames.to(device=self.device, dtype=self.dtype)
                video_normalized = (vf * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
                full_video = torch.cat([first_frame_norm, video_normalized], dim=2)
                with torch.no_grad():
                    video_latent = self.encode_video(full_video.to(self.dtype))
                video_latent[:, :, 0:1] = condition_frame_latent
            # Teacher-force 未来 depth：整段 VAE latent 钉死为 GT（t_depth=0），
            # 让 action 读训练时见过的几何，而不是正在去噪的噪声。
            if tf_depth:
                df = condition_depth_frames.to(device=self.device, dtype=self.dtype)
                depth_normalized = (df * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
                full_depth = torch.cat([depth_first_norm, depth_normalized], dim=2)
                with torch.no_grad():
                    depth_latent = self.encode_video(full_depth.to(self.dtype))
                depth_latent[:, :, 0:1] = depth_condition_latent
            action_latent = None
            if not state_missing:
                # 统一 action 空间：latent 用统一维 U 初始化（模型边界 scatter
                # 空间），输出端再按 robot 切片回原始维。fp32 采样噪声再 cast
                # （与训练侧同惯例，评审 16e）。
                if init_action_noise is not None:
                    action_latent = init_action_noise.to(device=self.device, dtype=torch.float32).clone()
                else:
                    action_latent = torch.randn(
                        (B, self.config.action_chunk_size, self.config.unified_action_dim),
                        device=self.device, dtype=torch.float32)
                if noise_capture is not None:
                    noise_capture["action_noise"] = action_latent.detach().clone()

            # 归一化条件输入（state）到模型内部空间（B3 策略或旧 zscore）；
            # action_latent 从 randn 初始化即处于该空间。state_local 保留
            # local 空间副本，供输出端 B1 还原（eef pos/rot 相对化参考点），
            # 随后另行 scatter 为 V14 的 state token 输入。
            state_local = None
            tf_action = False
            if not state_missing:
                state = self._normalize_state_actions(state, None, robot_names)[0]
                state_local = state

            tf_action = (
                condition_actions is not None
                and torch.is_tensor(condition_actions)
                and not state_missing
                and state_local is not None
            )
            if block_causal and condition_actions is not None:
                need_steps = (num_blocks * temporal_request.grid.block_action_steps
                              if packed_ar else self.config.action_chunk_size)
                if (not torch.is_tensor(condition_actions) or condition_actions.ndim != 3
                        or condition_actions.shape[0] != B or condition_actions.shape[1] < need_steps
                        or condition_actions.shape[2] != action_dim):
                    raise ValueError(f"FDM commands must be [B={B}, A>={need_steps}, Da={action_dim}]")
                if not torch.isfinite(condition_actions).all():
                    raise ValueError("FDM commands must be finite")
                if not tf_action:
                    raise ValueError("FDM requires state and an enabled action stream")
            if tf_action:
                cond_act = condition_actions.to(device=self.device, dtype=torch.float32)
                cond_act = self._normalize_state_actions(None, cond_act, robot_names)[1]
                _, cond_act, _ = self._scatter_to_unified(state_local, cond_act, robot_names)
                if packed_ar and cond_act.shape[1] < num_blocks * temporal_request.grid.block_action_steps:
                    raise ValueError(f"condition_actions must cover {num_blocks} block(s) = "
                                     f"{num_blocks * temporal_request.grid.block_action_steps} steps")
                action_latent = cond_act

            # V14 state token：把归一化后的 local 空间 state 再 scatter 到统一维
            # 作为网络输入（state_local 仍保留 local 副本，供输出端相对化还原）。
            # 循环外算一次：每个去噪步喂的都是同一个当前观测 qpos，全程干净、
            # 不进 Euler 更新。state 缺失时 action 分支整体不在场，不做 scatter
            # （_scatter_to_unified 需要 state/actions 至少一个非 None 取 batch）。
            state_uni = None
            action_dim_mask = None
            if not state_missing and state_local is not None:
                state_uni, _, action_dim_mask = self._scatter_to_unified(state_local, None, robot_names)
                # Training zeroes unregistered unified columns in the noisy input;
                # keep them zero here too (noise + every velocity update).
                if action_dim_mask is not None:
                    action_dim_mask = action_dim_mask.to(device=self.device, dtype=torch.float32).unsqueeze(1)
                    if action_latent is not None and not tf_action:
                        action_latent = action_latent * action_dim_mask

            # 3. Denoising loop: from noise (sigma=1) to clean (sigma=0)
            # per-modality σ 网格（修复 #9c，UWM 式）：action 可用独立（通常
            # 更少）的去噪步数——训练侧 independent_modality_noise 下
            # (σ_v, σ_a) 独立分布，推理共用同一 σ 只查询对角线，存在轻微
            # train/inference 错位。循环按两网格的较细者推进；较粗模态在
            # 部分循环步 dt=0 保持不动（其时间标签仍按当前 σ 供给网络）。
            sigmas = self._inference_sigmas(num_inference_steps)
            n_action_steps = num_inference_steps
            if action_num_inference_steps is not None and int(action_num_inference_steps) >= 1:
                n_action_steps = int(action_num_inference_steps)
            sigmas_action = (
                self._inference_sigmas(n_action_steps, shift=self._action_flow_shift)
                if not state_missing else None
            )
            n_loop = max(num_inference_steps, n_action_steps)
            # 逐块生成（packed）：每块重跑完整 σ 网格；块 k 的打包输入取已生成的
            # 块 1..k-1 为干净前缀，速度场只写回块 k 的位置。非 packed 为单趟全窗。
            gen_blocks = list(range(1, num_blocks + 1)) if packed_ar else [None]
            n_v = None if video_latent is None else video_latent.shape[2]
            n_d = None if depth_latent is None else depth_latent.shape[2]
            n_a = None if action_latent is None else action_latent.shape[1]

            def _pack_in(plan, v, a, d):
                if plan is None:
                    return v, a, d
                return (None if v is None else TT.pack(v, v, plan.video, 2),
                        None if a is None else TT.pack(a, a, plan.action, 1),
                        None if d is None else TT.pack(d, d, plan.depth, 2))

            def _unpack_out(plan, v, a, d):
                if plan is None:
                    return v, a, d
                return (None if v is None else TT.unpack_targets(v, plan.video, n_v, 2),
                        None if a is None else TT.unpack_targets(a, plan.action, n_a, 1),
                        None if d is None else TT.unpack_targets(d, plan.depth, n_d, 2))

            req_i, plan = temporal_request, None
            prefix_cond = prefix_uncond = None
            prefix_cache_stats = []
            for gen_block, i in [(blk, i) for blk in gen_blocks for i in range(n_loop)]:
                if gen_block is not None and i == 0:
                    req_i = TT.block_request(temporal_request, gen_block)
                    plan = TT.pack_plan(req_i)
                    if prefix_cache_mode == "fixed":
                        # Never reuse across blocks, observations, or CFG branches.
                        branches = 2 if und_cache_uncond is not None else 1
                        budget = int(prefix_cache_max_mb * 1024 * 1024) // branches
                        prefix_cond = PrefixKVCache(budget)
                        prefix_uncond = PrefixKVCache(budget) if branches == 2 else None
                # Timesteps（各模态当前网格位置；action 网格更粗时同一格点
                # 覆盖多个 video 循环步）
                i_v = min(i * num_inference_steps // n_loop, num_inference_steps - 1)
                i_a = min(i * n_action_steps // n_loop, n_action_steps - 1)
                sigma = sigmas[i_v]
                sigma_action = sigmas_action[i_a] if sigmas_action is not None else sigma
                # 本循环步是否推进该模态到下一网格点（不推进则 dt=0 保持不动）
                adv_video = (i + 1) * num_inference_steps // n_loop > i_v
                adv_action = (i + 1) * n_action_steps // n_loop > i_a
                dt_video = float(sigmas[i_v + 1] - sigma) if adv_video else 0.0
                dt_action = (
                    float(sigmas_action[i_a + 1] - sigma_action)
                    if (adv_action and sigmas_action is not None) else 0.0
                )
                # 时间标签按各模态当前 σ 给（t = σ×1000，与训练标签一致）
                t_video = (sigma.float() * 1000).expand(B).unsqueeze(1)
                t_action = (sigma_action.float() * 1000).expand(B).unsqueeze(1)
                # depth 跟 video 的 σ 网格。不钉 depth 时 t_depth 必须留在当前
                # σ：若跟着 t_video 一起置 0，depth 仍是噪声却按干净样本查速度。
                t_depth = t_video.clone()
                if tf_video:
                    t_video = torch.zeros_like(t_video)
                if tf_action:
                    t_action = torch.zeros_like(t_action)
                if tf_depth:
                    t_depth = torch.zeros_like(t_depth)

                # Trimodal MoT forward - joint denoising for WAN, Action, Understanding
                # depth 与 video 共用 σ 网格（t_depth=t_video），随循环同步去噪；
                # depth 读当前 video/action；depth_read_by_video 时 video 也读 depth。
                # 不对 video/und 施加 depth CFG（depth 非文本条件），uncond 前向
                # 不算 depth 省算力。
                v_in, a_in, d_in = _pack_in(plan, video_latent, action_latent, depth_latent)
                video_velocity, action_velocity, depth_velocity = _unpack_out(plan, *self.world_model(
                    t_video, vlm_input_messages, und_pils,
                    v_in, a_in,
                    t_video=t_video, t_action=t_action,
                    robot_names=robot_names,
                    und_cache=und_cache,
                    depth=d_in, t_depth=t_depth,
                    depth_first_frame_conditioned=depth_first_frame_conditioned,
                    depth_read_by_action=bool(getattr(self.config, "depth_read_by_action", False)),
                    depth_detach_video_kv=bool(getattr(self.config, "depth_detach_video_kv", False)),
                    depth_read_by_video=bool(getattr(self.config, "depth_read_by_video", False)),
                    depth_read_by_und=bool(getattr(self.config, "depth_read_by_und", False)),
                    video_read_by_action=bool(getattr(self.config, "video_read_by_action", True)),
                    state=state_uni,
                    temporal=req_i if block_causal else None,
                    history=history_context,
                    recovery=recovery,
                    _prefix_cache=prefix_cond,
                ))
                if und_cache_uncond is not None:
                    # classifier-free guidance（评审 #6）：
                    # v = v_uncond + guidance_scale·(v_cond - v_uncond)
                    # RGB 读 depth 时 uncond 也必须带 depth，否则 CFG 会把
                    # 「有没有 depth」和文本条件混在一起放大。action 读 depth
                    # 时 RGB 可经 action token 间接读到 depth，同样要带。
                    _uncond_needs_depth = bool(
                        getattr(self.config, "depth_read_by_video", False)
                        or getattr(self.config, "depth_read_by_und", False)
                        or getattr(self.config, "depth_read_by_action", False)
                    )
                    video_velocity_uncond, action_velocity_uncond, _ = _unpack_out(plan, *self.world_model(
                        t_video, [UNCONDITIONAL_VLM_MESSAGE] * B, und_pils,
                        v_in, a_in,
                        t_video=t_video, t_action=t_action,
                        robot_names=robot_names,
                        und_cache=und_cache_uncond,
                        depth=d_in if _uncond_needs_depth else None,
                        t_depth=t_depth if _uncond_needs_depth else None,
                        depth_first_frame_conditioned=depth_first_frame_conditioned,
                        depth_read_by_action=bool(getattr(self.config, "depth_read_by_action", False)),
                        depth_detach_video_kv=bool(getattr(self.config, "depth_detach_video_kv", False)),
                        depth_read_by_video=bool(getattr(self.config, "depth_read_by_video", False)),
                        depth_read_by_und=bool(getattr(self.config, "depth_read_by_und", False)),
                        video_read_by_action=bool(getattr(self.config, "video_read_by_action", True)),
                        # CFG 的 uncond 支只去掉文本条件，state 与 cond 支保持
                        # 同一个：否则会把「有没有 state」混进文本 guidance 放大。
                        state=state_uni,
                        # 历史/恢复条件同理与 cond 支一致，CFG 只放大文本条件。
                        temporal=req_i if block_causal else None,
                        history=history_context,
                        recovery=recovery,
                        _prefix_cache=prefix_uncond,
                    ))
                    if video_velocity is not None and video_velocity_uncond is not None:
                        video_velocity = video_velocity_uncond + guidance_scale * (
                            video_velocity - video_velocity_uncond)
                    # action 默认不做 CFG（修复 #1；π0 / Diffusion Policy 均不
                    # 对 action 施加 CFG）：弱条件下 cond−uncond 方向近似噪声，
                    # ×5 放大是推理动作剧烈振荡的头号放大器（micro 对照 g=5
                    # 比 g=1 的 action MSE 一致差 2.1~2.8×）。
                    # action_guidance_scale != 1.0 仅为消融对照保留。
                    if (
                        float(action_guidance_scale) != 1.0
                        and action_velocity is not None and action_velocity_uncond is not None
                    ):
                        action_velocity = action_velocity_uncond + float(action_guidance_scale) * (
                            action_velocity - action_velocity_uncond)

                # Euler integration（缺失模态 / 该模态不推进的循环步跳过）
                # teacher-force 的 video/action/depth 不推进。
                if (not tf_video) and video_velocity is not None and dt_video != 0.0:
                    video_latent = video_latent + video_velocity * dt_video
                if (not tf_action) and action_velocity is not None and dt_action != 0.0:
                    if action_dim_mask is not None:
                        action_velocity = action_velocity * action_dim_mask
                    action_latent = action_latent + action_velocity * dt_action
                if (
                    (not tf_depth)
                    and depth_latent is not None
                    and depth_velocity is not None
                    and dt_video != 0.0
                ):
                    depth_latent = depth_latent + depth_velocity * dt_video

                # Teacher Forcing
                if video_latent is not None:
                    video_latent[:, :, 0:1] = condition_frame_latent
                if depth_latent is not None and depth_first_frame_conditioned:
                    depth_latent[:, :, 0:1] = depth_condition_latent
                if prefix_cond is not None and i == n_loop - 1:
                    prefix_cache_stats.append(dict(block=gen_block, cond=prefix_cond.stats(),
                        uncond=None if prefix_uncond is None else prefix_uncond.stats()))
                    prefix_cond = prefix_uncond = None  # release tensors before the next block

            if packed_ar:
                # 未生成的块仍是初始噪声，不返回。
                keep_lat = 1 + num_blocks * temporal_request.grid.latents_per_block
                keep_steps = num_blocks * temporal_request.grid.block_action_steps
                if video_latent is not None:
                    video_latent = video_latent[:, :, :keep_lat]
                if depth_latent is not None:
                    depth_latent = depth_latent[:, :, :keep_lat]
                if action_latent is not None:
                    action_latent = action_latent[:, :keep_steps]

            if latent_capture is not None:
                latent_capture["video_latent"] = None if video_latent is None else video_latent.detach().clone()
                latent_capture["depth_latent"] = None if depth_latent is None else depth_latent.detach().clone()
                latent_capture["action_latent"] = None if action_latent is None else action_latent.detach().clone()
                latent_capture["temporal_request"] = temporal_request if block_causal else None
                latent_capture["prefix_cache"] = dict(mode=prefix_cache_mode, blocks=prefix_cache_stats)

            # 4. Decode outputs。闭环部署 decode_outputs=False：不去像素，
            # 联合去噪（含 depth latent）已经做完，action 从 latent 直接还原。
            predicted_frames = None
            if decode_outputs and video_latent is not None:
                with torch.no_grad():
                    decoded_frames = self.decode_video(video_latent)
                    predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
                    predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
                    predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
            
            predicted_actions = None
            if action_latent is not None:
                # 推理时序平滑（修复 #3b）：EMA 压低逐帧白噪声分量（抖动），
                # 零训练成本；alpha<=0 关闭（旧行为）。
                # teacher-force 的动作是给定命令，不平滑（否则返回的不是输入动作）。
                if float(action_smooth_alpha) > 0.0 and not tf_action:
                    action_latent = _temporal_ema(action_latent, float(action_smooth_alpha))
                predicted_actions = action_latent.float()  # [B, action_chunk_size, U]（统一空间）
                # 切回该机器人原始维（batch 源齐次，dim_map gather 或 offset
                # 前填切片，取 robot_names[0]）+ B1/B2 还原（eef pos 加回
                # state、rot6 → 绝对姿态本地表示；absolute_native 为纯切片）
                predicted_actions = self._gather_native_actions(
                    predicted_actions, state_local, robot_names,
                    action_cols, robot_offset, action_dim)
                predicted_actions = self._denormalize_actions(predicted_actions, robot_names)
                # 摘除维（dim_map -1，修复 #7，如 agibot 灵巧手占位维）重建为
                # raw 0：这些维在 reader 输出中恒 0，模型不对其预测，反归一化
                # 后显式置 0 即精确重建（否则输出未训练垃圾值污染指标/下游）。
                # rot group 接管的本地维（也是 -1）已由 _gather_native_actions
                # 还原真实值，从置零清单排除。
                dropped_dims = _dropped_local_dims(action_cols)
                if self._repr_specs is not None and robot_names:
                    rot_dims = self._repr_rot_local_dims(robot_names[0])
                    if rot_dims:
                        dropped_dims = [d for d in dropped_dims if d not in rot_dims]
                if dropped_dims:
                    predicted_actions[..., dropped_dims] = 0.0

            # depth 解码（depth-as-video 表示，[0,1] 3 通道）：depth_latent 与
            # video 同处 VAE latent 空间（depth 专属 head/patch 只在 transformer
            # 分支内、已在去噪前向用过），故最终 latent→像素用共享 VAE decode
            # （self.decode_video）。转回米制深度由调用方用
            # depth_codec.decode_to_depth_m 完成（需 mode/z_near/z_far，随数据集
            # 配置，不在本函数内假定）。
            predicted_depth = None
            if return_depth and decode_outputs and depth_latent is not None:
                with torch.no_grad():
                    decoded_depth = self.decode_video(depth_latent)
                    # 先归一化再 EMA：和 action 一样作用在模型输出空间
                    # （log-depth 的 [0,1] 编码）。含首帧条件，让干净条件
                    # 钉住 y[0]，再切掉条件帧。
                    decoded_depth = (decoded_depth + 1.0) / 2.0
                    decoded_depth = torch.clamp(decoded_depth, 0, 1).float()
                    if float(depth_smooth_alpha) > 0.0:
                        decoded_depth = _temporal_ema(
                            decoded_depth, float(depth_smooth_alpha), dim=2,
                        )
                    predicted_depth = (decoded_depth if return_depth_first_frame
                                       else decoded_depth[:, :, 1:])

            if return_depth:
                return predicted_frames, predicted_actions, predicted_depth
            return predicted_frames, predicted_actions

    def inference_action_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        conditioning_actions: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        first_frame_pils: Optional[List[PIL.Image.Image]] = None,
        t5_input_messages: Optional[List[str]] = None,
        vlm_input_messages: Optional[List[str]] = None,
        robot_names: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction with action conditioning.
        At each denoising step, the action and state inputs are replaced with
        conditioning_actions (set to t=0), and text is set to empty strings.
        
        Args:
            first_frame: Initial frame [B, C, H, W]
            state: Initial robot state [B, state_dim] (used for conditioning)
            conditioning_actions: Actions to condition on [B, action_chunk_size, action_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings (should be None, text will be empty)
            vlm_inputs: VLM inputs (should be None, text will be empty)
            first_frame_pils: PIL images for preprocessing
            t5_input_messages: Text messages (will be empty strings)
            vlm_input_messages: VLM messages (will be empty strings)
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        if conditioning_actions is None:
            raise ValueError("inference_action_step requires conditioning_actions; use inference_step for joint sampling")
        if depth_condition_mode(self.config) == "rgb_only":
            raise ValueError("rgb_only action-conditioned rollout uses inference_step(condition_actions=...), not the legacy helper")
        B = first_frame.shape[0]

        assert language_embeddings is None
        assert vlm_inputs is None
        action_cols, robot_offset, action_dim = _robot_action_columns(
            self.config, robot_names)
        if conditioning_actions is not None:
            action_dim = int(conditioning_actions.shape[-1])

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # 1. Video/Action latents init
            # Condition frame encode
            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
            with torch.no_grad():
                condition_frame_latent = self.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

            # 条件全程不变：und token 循环外算一次并缓存（评审 #11）。
            with torch.no_grad():
                und_cache = self.world_model.und_branch.encode(
                    vlm_input_messages, first_frame_pils)

            # Init video/action latents（fp32 采样噪声再 cast，评审 16e）
            B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
            num_total_latent_frames = 1 + self.config.num_video_frames // 4
            video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=torch.float32).to(self.dtype)
            video_latent[:, :, 0:1] = condition_frame_latent
            # 统一 action 空间：latent 用统一维 U 初始化，输出端再切回原始维。
            action_shape = (B, self.config.action_chunk_size, self.config.unified_action_dim)
            action_latent = torch.randn(action_shape, device=self.device, dtype=torch.float32)

            # 归一化条件输入（state / 条件动作）到模型内部空间（B3 策略或旧
            # zscore）；action_latent 从 randn 初始化即处于该空间。
            # state_local 保留 local 空间副本供输出端 B1 还原与条件动作
            # 相对化锚点，随后另行 scatter 为 V14 的 state token 输入。
            state = self._normalize_state_actions(state, None, robot_names)[0]
            state_local = state
            if conditioning_actions is not None:
                conditioning_actions = self._normalize_state_actions(
                    None, conditioning_actions, robot_names
                )[1]
                # 统一 action 空间：条件动作同样 scatter 到 [B, T, U]
                # （B1：以同一条件帧 state 为参考做相对化）
                _, conditioning_actions, _ = self._scatter_to_unified(
                    state_local, conditioning_actions, robot_names)

            # V14 state token：统一维 state 作为网络输入（循环外算一次，全程干净）。
            state_uni = None
            if state_local is not None:
                state_uni = self._scatter_to_unified(state_local, None, robot_names)[0]

            # 3. Denoising loop: from noise (sigma=1) to clean (sigma=0)
            sigmas = self._inference_sigmas(num_inference_steps)
            for i in range(num_inference_steps):
                # Timesteps
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                dt = sigma_next - sigma
                t_scaled = (sigma.float() * 1000).expand(B).unsqueeze(1)

                # Prepare conditioned inputs:
                # - Use conditioning_actions with t_action=0 if provided
                # - Text is always empty strings
                t_action = torch.zeros_like(t_scaled)  # Set action timestep to 0
                
                if conditioning_actions is not None:
                    # Use conditioning actions instead of current noisy actions
                    input_actions = conditioning_actions
                else:
                    input_actions = action_latent
                
                # Trimodal MoT forward - joint denoising with action conditioning
                video_velocity, action_velocity, _ = self.world_model(
                    t_scaled, vlm_input_messages, first_frame_pils,
                    video_latent, input_actions,
                    t_video=t_scaled, t_action=t_action,  # Action at t=0 for conditioning
                    robot_names=robot_names,
                    und_cache=und_cache,
                    state=state_uni,
                )

                # Euler integration
                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent
    
            # 4. Decode outputs
            with torch.no_grad():
                decoded_frames = self.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
                predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
            
            predicted_actions = (
                conditioning_actions.float() if conditioning_actions is not None else action_latent.float()
            )
            # 切回该机器人原始维（batch 源齐次，dim_map gather 或 offset 前填
            # 切片，取 robot_names[0]）+ B1/B2 还原（同 inference_step）
            predicted_actions = self._gather_native_actions(
                predicted_actions, state_local, robot_names,
                action_cols, robot_offset, action_dim)
            predicted_actions = self._denormalize_actions(predicted_actions, robot_names)
            # 摘除维（dim_map -1，修复 #7）重建为 raw 0（同 inference_step；
            # rot group 接管的本地维已还原真实值，排除在置零清单外）。
            dropped_dims = _dropped_local_dims(action_cols)
            if self._repr_specs is not None and robot_names:
                rot_dims = self._repr_rot_local_dims(robot_names[0])
                if rot_dims:
                    dropped_dims = [d for d in dropped_dims if d not in rot_dims]
            if dropped_dims:
                predicted_actions[..., dropped_dims] = 0.0

            return predicted_frames, predicted_actions
