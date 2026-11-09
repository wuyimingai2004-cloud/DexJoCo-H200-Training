import logging
import os

import torch
import torch.nn as nn

from typing import Dict, Optional, Tuple

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from thinker_wm.wan.modules.model import WanModel
import PIL.Image as Image

from thinker_wm.models.video_branch import VideoBranch
from thinker_wm.models.action_branch import ActionBranch
from thinker_wm.models.und_branch import UndBranch
from thinker_wm.models.uni_branch import UniBranch
from thinker_wm.models.robot_tokens import RobotTokenEmbedding
from motus_monitor.timing import timed_section

_FP8_LINEAR_INCLUDE = ("uni_branch.uni_blocks.", "video_branch.decode_block.")
_FP8_LINEAR_EXCLUDE = ("cross_attn", "modulation", "norm", "head", "embedding", "projection")


def _fp8_exclude(name, module) -> bool:
    return not name.startswith(_FP8_LINEAR_INCLUDE) or any(tag in name for tag in _FP8_LINEAR_EXCLUDE)


class UniWorldModel(nn.Module):
    def __init__(self, 
                 vlm_preprocess, vlm_model, wan_model:WanModel,
                 action_dim:int,
                 video_encoder_layer_num:int, video_decoder_layer_num:int,
                 action_encoder_layer_num:int, action_decoder_layer_num:int,
                 und_layer_num:int,
                 action_t_downsample:int = 24,
                 video_action_freq_ratio:int = 6,
                 robot_configs=None,
                 unified_action_dim: int = 64,
                 robot_token_indices: Optional[Dict[str, int]] = None,
                 gradient_checkpointing: bool = False,
                 checkpoint_video_branch: bool = True,
                 checkpoint_uni_branch: bool = True,
                 vlm_text_rewrite: bool = False,
                 vlm_rewrite_max_new_tokens: int = 128,
                 vlm_rewrite_cache_dir: str = "",
                 vlm_field_dropout_prob: float = 0.0,
                 vlm_model_tag: str = "",
                 enable_depth: bool = False,
                 depth_uni_clone_layers: int = 0,
                 enable_history: bool = False,
                 history_spatial_pool: int = 1,
                 history_logit_bias_init: float = -6.0,
                 enable_recovery_condition: bool = False,
                 recovery_feature_dim: int = 0,
                 recovery_num_tokens: int = 4,
    ):
        super().__init__()

        wan_dim = wan_model.dim
        wan_ffn_dim = wan_model.ffn_dim
        wan_num_heads = wan_model.num_heads

        # wan_model.text_embedding（4096→wan_dim MLP）只服务已移除的 T5
        # embedding 兼容路径（评审 #10），在 Motus 前向中永不使用。保留模块
        # 本身（vendor Wan DiT 预训练权重的一部分，保证权重加载键兼容），
        # 但冻结以退出 optimizer / weight-decay。
        for p in wan_model.text_embedding.parameters():
            p.requires_grad = False

        # 每机器人 8 个可学习 token（统一 action 空间配套，models/robot_tokens.py）：
        # 槽位由 YAML robot_token_index 显式指定，append-only 预留 64 槽；超容量
        # 走增长协议（见 models/robot_tokens.py 头注）。
        self.robot_token_embedding = RobotTokenEmbedding(
            robot_token_indices or {}, max_robots=64, num_tokens=8, dim=wan_dim,
        )

        # depth 第三生成流（Rynn 式 depth-as-video）：VideoBranch 内新增 depth
        # 专属 patch_embed/head + 零初始化 stream_embed，共享 encode/decode blocks。
        self.enable_depth = bool(enable_depth)
        self.video_branch = VideoBranch(
            wan_model,
            video_encoder_layer_num,
            video_decoder_layer_num,
            gradient_checkpointing=bool(gradient_checkpointing and checkpoint_video_branch),
            robot_token_embedding=self.robot_token_embedding,
            enable_depth=self.enable_depth,
        )

        self.action_branch = ActionBranch(
            action_dim,
            wan_dim, wan_ffn_dim, wan_num_heads,
            action_encoder_layer_num,
            action_decoder_layer_num,
            t_downsample=action_t_downsample,
            video_action_freq_ratio=video_action_freq_ratio,
            # 统一 action 空间：无投影层，统一维 U 直接作为模型内部 action
            # 维；robot_configs 保留作签名兼容。
            unified_action_dim=unified_action_dim,
            robot_token_embedding=self.robot_token_embedding,
            robot_configs=robot_configs,
        )

        self.und_branch = UndBranch(
            vlm_preprocess, vlm_model,
            wan_dim, wan_ffn_dim,
            und_layer_num,
            vlm_text_rewrite=vlm_text_rewrite,
            vlm_rewrite_max_new_tokens=vlm_rewrite_max_new_tokens,
            vlm_rewrite_cache_dir=vlm_rewrite_cache_dir,
            vlm_field_dropout_prob=vlm_field_dropout_prob,
            vlm_model_tag=vlm_model_tag,
        )

        # prompt 模态缺失/dropout 的 und 占位（评审 #4）：可学习 null token，
        # 输出三元组与 und_branch.encode 一致（token/grid_sizes/seq_lens），
        # uni 分支无感知（und 恒在场）。
        self.null_und_token = nn.Parameter(torch.zeros(1, 1, wan_dim).normal_(std=0.02))
        
        self.uni_branch = UniBranch(
            wan_model=wan_model,
            encode_layer_num=video_encoder_layer_num,
            decode_layer_num=video_decoder_layer_num,
            gradient_checkpointing=bool(gradient_checkpointing and checkpoint_uni_branch),
            # RoPE 时间间距 = t_downsample/(ratio*4)，4 为 VAE 时间 stride。
            # 无 state 时动作 j 位于 j；V14 有 state 前缀时 state@0、动作
            # j@(j+1)。此处只缩放间距，不补偿前缀或数据 action_start_offset。
            action_rope_time_scale=action_t_downsample / (video_action_freq_ratio * 4.0),
            # depth 专属容量：uni 最后 N 层克隆一份给 depth（0 = 全共享，旧行为）。
            depth_uni_clone_layers=int(depth_uni_clone_layers or 0) if self.enable_depth else 0,
        )

        # 视觉历史（块级 AR 路径专用）与偏差恢复条件。均零初始化，关闭时不建，
        # 旧 checkpoint 键与前向不受影响。
        self.history_encoder = None
        if enable_history:
            from thinker_wm.models.history_encoder import HistoryEncoder
            self.history_encoder = HistoryEncoder(
                wan_dim, unified_action_dim, freq_dim=wan_model.freq_dim,
                pool=int(history_spatial_pool), logit_bias_init=float(history_logit_bias_init))
        self.recovery_encoder = None
        if enable_recovery_condition:
            from thinker_wm.models.recovery_encoder import RecoveryEncoder
            if int(recovery_feature_dim) < 1:
                raise ValueError("enable_recovery_condition requires recovery_feature_dim >= 1")
            self.recovery_encoder = RecoveryEncoder(
                int(recovery_feature_dim), wan_dim, num_tokens=int(recovery_num_tokens))

        # FP8 rowwise Linear (ported from thinkerwm-pretrain, default off):
        # self-attn q/k/v/o + ffn of the shared uni trunk and video decoder
        # blocks only. Depth clones, history, action/und and cross-attn stay bf16.
        if os.environ.get("MOTUS_FP8_LINEAR", "").lower() in ("1", "true"):
            from thinker_wm.models.fp8_linear import convert_to_fp8_linear
            count = convert_to_fp8_linear(self, weight_only=os.environ.get("MOTUS_W8A16", "").lower() in ("1", "true"),
                                          exclude=_fp8_exclude)
            logging.getLogger(__name__).info("FP8 Linear conversion: %d modules replaced", count)
    
    def null_und_cache(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 und 三元组 (token, grid_sizes, seq_lens) 的 null 版本：
        每个样本 1 个可学习 null token（grid [1,1,1]、seq_len 1 无 padding）。
        用于 prompt 模态缺失/dropout 时替代 und_branch.encode 的输出，
        uni 分支无感知（und 恒在场）。
        """
        token = self.null_und_token.expand(B, 1, -1)
        grid_sizes = torch.ones(B, 3, dtype=torch.long, device=token.device)
        seq_lens = torch.ones(B, dtype=torch.long, device=token.device)
        return token, grid_sizes, seq_lens

    def forward(self, t:torch.Tensor,
                vlm_input_message:list[str], vlm_input_first_frame:list[Image.Image],
                video:Optional[torch.Tensor], action:Optional[torch.Tensor],
                t_video:torch.Tensor=None, t_action:torch.Tensor=None,
                robot_names=None,
                return_list = False,
                und_cache=None,
                drop_prompt: bool = False,
                und_precomputed=None,
                depth:Optional[torch.Tensor]=None, t_depth:torch.Tensor=None,
                depth_read_by_action: bool = False,
                depth_detach_video_kv: bool = False,
                depth_read_by_video: bool = False,
                depth_read_by_und: bool = False,
                video_read_by_action: bool = True,
                state:Optional[torch.Tensor]=None,
                timing=None, timing_sync_cuda: bool = False, timing_level: int = 1,
                depth_first_frame_conditioned: bool = True,
                temporal=None,
                history: Optional[Dict[str, torch.Tensor]] = None,
                recovery: Optional[Dict[str, torch.Tensor]] = None,
                _prefix_cache=None,
    ):
        """
        temporal: 可选 models.temporal_layout.TemporalRequest。非 None 时走块级
            因果路径：所有 encoder/uni/decoder 自注意力按物理时间块限制可见性，
            und 只汇总已观测 token，t_video/t_action/t_depth 按 token 角色展开
            （干净 token 标签为 0）。调用方负责把干净角色位置的 latent/动作写回
            真值（与 temporal_layout.latent_roles/action_step_roles 一致）。
        history: 可选 {latent [B,C,Th,H,W], times [B,Th] 秒(负), valid [B,Th] bool,
            kind [B,Th] long, state [B,Th,U]}，只在 temporal 路径使用。
        recovery: 可选 {features [B,Dr], valid [B] bool}，经 RecoveryEncoder 变成
            und 前缀 token（无效样本不加），供偏差条件的恢复动作生成。

        video/action 可为 None（模态 dropout / 推理缺失）：对应分支的
        encode/uni/decode 跳过，返回 None。

        state: 条件帧 qpos [B, U]（已归一化并 scatter 到统一维），V14 起作为
            动作序列第 0 个 token 的输入，扩散时间恒 0、不加噪、不进 loss。
            None 或 action 缺失时完全退回无 state token 的旧路径——action 被
            模态 dropout 掉时不会留下孤立的 state token。新增关键字参数排在
            已有位置参数之后，旧调用不受影响。

        drop_prompt: True 时忽略 vlm_input_message，und 三元组改用
            null_und_cache（可学习 null token），用于 prompt 模态 dropout。

        und_cache: 可选 (und_token, und_grid_sizes, und_seq_lens) 三元组。
            推理路径在去噪循环外预先算好并传入，避免每个去噪步重跑冻结
            VLM + und encoder（评审 #11）；训练路径不传（und 分支可训练，
            梯度需经 encode 回流）。

        und_precomputed: 可选 (vlm_hidden, real_lens)——训练侧 prepare 级
            半缓存（20260818_v2）：VLM 冻结段（rewrite/字段 dropout/特征
            提取）在 prepare_training_context 每 batch 算一次，本参数传入
            后 und 只跑可训练的 bridge+und_encoder（梯度正常回传）；
            prompt 模态 dropout 步由调用方改传 uncond token 嵌入常量
            （tokenizer-only，und_branch.get_uncond_vlm_feature）。
        """

        # Allow separate timesteps for video and action (for masking during training)
        if t_video is None:
            t_video = t
        if t_action is None:
            t_action = t
        # depth 默认与 video 同噪声调度。
        if t_depth is None:
            t_depth = t_video

        # stage2_20260802 起：und（VLM 对「首帧图 + 任务文本」的理解）是全局条件。
        # 它先于 encoder 计算，并作为 video/action encoder 的 cross-attn K/V；
        # T5 已移除（评审 #10），文本信息全部经 VLM 进入 und。
        # und_seq_lens 是逐样本真实（未 padding）长度，所有把 und 当 K/V 的
        # cross-attn 都据此 mask 掉 VLM padding 位（评审 #1）。
        if und_cache is not None:
            und_token, und_grid_sizes, und_seq_lens = und_cache
        elif und_precomputed is not None:
            # prepare 级半缓存（20260818_v2）：跳过 VLM 冻结段，只跑可训练
            # bridge+und_encoder。
            with timed_section(timing, "wm_und_encode_seconds", timing_sync_cuda):
                und_token, und_grid_sizes, und_seq_lens = self.und_branch.encode_from_vlm_hidden(
                    und_precomputed[0], und_precomputed[1],
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)
        elif drop_prompt:
            # prompt 模态 dropout：und 退化为可学习 null token（评审 #4）。
            # 调用方保证 video/action 至少一个在场（两者全缺时 B 取第一个
            # 非 None 模态；全缺属上游错误，由 UBT_WM 入口拦截）。
            if video is not None:
                b = video.shape[0]
            elif action is not None:
                b = action.shape[0]
            else:
                b = len(vlm_input_message)
            und_token, und_grid_sizes, und_seq_lens = self.null_und_cache(b)
        else:
            with timed_section(timing, "wm_und_encode_seconds", timing_sync_cuda):
                und_token, und_grid_sizes, und_seq_lens = self.und_branch.encode(
                    vlm_input_message, vlm_input_first_frame,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        if recovery is not None:
            und_token, und_grid_sizes, und_seq_lens = self._prepend_recovery(
                und_token, und_grid_sizes, und_seq_lens, recovery)

        if _prefix_cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("prefix KV cache is inference-only (eval + no_grad)")
            if temporal is None or not temporal.packed or temporal.first_target != temporal.last_target:
                raise ValueError("prefix KV cache requires one packed AR target block")
        if temporal is not None:
            return self._forward_temporal(
                temporal, history, und_token, und_grid_sizes, und_seq_lens,
                video, action, depth, t_video, t_action, t_depth, state, robot_names,
                return_list, depth_read_by_action, depth_detach_video_kv,
                depth_read_by_video, depth_read_by_und, video_read_by_action,
                timing, timing_sync_cuda, timing_level, _prefix_cache)
        if history is not None:
            raise ValueError("history requires the temporal (block-causal) path")

        # branch encode - use separate timesteps
        video_token = video_grid_sizes = video_seq_lens = video_e = video_e0 = None
        if video is not None:
            with timed_section(timing, "wm_video_encode_seconds", timing_sync_cuda):
                video_token, video_grid_sizes, video_seq_lens, video_e, video_e0 = self.video_branch.encode(
                    video, t_video, und_token, context_lens=und_seq_lens,
                    robot_names=robot_names,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        # has_state_token 在此统一判定一次，encode/uni/decode 三段共用同一个
        # 布尔量，避免任何一段按序列长度反推（V14）。
        has_state_token = action is not None and state is not None
        action_token = action_grid_sizes = action_seq_lens = action_e0 = None
        if action is not None:
            with timed_section(timing, "wm_action_encode_seconds", timing_sync_cuda):
                action_token, action_grid_sizes, action_seq_lens, action_e0 = self.action_branch.encode(
                    action, t_action, und_token, robot_names=robot_names,
                    context_lens=und_seq_lens,
                    state=state if has_state_token else None,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        # depth encode（第三生成流）：走 depth 专属 patch_embedding + stream_embed
        # + 共享 encode blocks（默认与 video 同噪声调度 t_depth）。
        depth_token = depth_grid_sizes = depth_seq_lens = depth_e = depth_e0 = None
        if depth is not None:
            assert self.enable_depth, "forward(depth=...) 需 enable_depth=True"
            with timed_section(timing, "wm_depth_encode_seconds", timing_sync_cuda):
                depth_token, depth_grid_sizes, depth_seq_lens, depth_e, depth_e0 = self.video_branch.encode_depth(
                    depth, t_depth, und_token, context_lens=und_seq_lens,
                    first_frame_conditioned=depth_first_frame_conditioned,
                    robot_names=robot_names,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        # uni forward - video/action/depth 分别用各自的 timestep 构建时间标签。
        # depth 默认单向：video/und 不读 depth；depth 读 video/action。
        # depth_read_by_action=True 时 action 也读 depth；
        # depth_read_by_video=True 时 video 也读 depth（RGB↔depth 双向）；
        # depth_read_by_und=True 时 und 也读 depth（und↔depth 双向）。
        with timed_section(timing, "wm_uni_process_seconds", timing_sync_cuda):
            video_token, action_token, depth_token, und_token = self.uni_branch.uni_process(
                video_token, video_grid_sizes,
                action_token, action_grid_sizes,
                und_token, und_grid_sizes,
                t_video, t_action, video_seq_lens, action_seq_lens, und_seq_lens,
                depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
                depth_seq_lens=depth_seq_lens, t_depth=t_depth,
                depth_read_by_action=depth_read_by_action,
                depth_detach_video_kv=depth_detach_video_kv,
                depth_read_by_video=depth_read_by_video,
                depth_read_by_und=depth_read_by_und,
                video_read_by_action=video_read_by_action,
                has_state_token=has_state_token,
                depth_first_frame_conditioned=depth_first_frame_conditioned,
                timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        # branch decode - decoder 的 cross-attn 条件为 uni 最后一层输出的 und_token
        video_out = None
        if video_token is not None:
            with timed_section(timing, "wm_video_decode_seconds", timing_sync_cuda):
                video_out = self.video_branch.decode(
                    video_token, und_token, video_grid_sizes, video_seq_lens, video_e, video_e0,
                    context_lens=und_seq_lens,
                    robot_names=robot_names,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)
        action_out = None
        if action_token is not None:
            with timed_section(timing, "wm_action_decode_seconds", timing_sync_cuda):
                action_out = self.action_branch.decode(
                    action_token, und_token, action_grid_sizes, action_seq_lens, action_e0,
                    robot_names=robot_names, context_lens=und_seq_lens,
                    has_state_token=has_state_token,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        depth_out = None
        if depth_token is not None:
            with timed_section(timing, "wm_depth_decode_seconds", timing_sync_cuda):
                depth_out = self.video_branch.decode_depth(
                    depth_token, und_token, depth_grid_sizes, depth_seq_lens, depth_e, depth_e0,
                    context_lens=und_seq_lens,
                    robot_names=robot_names,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

        if not return_list:
            video_out = torch.stack(video_out, dim=0) if video_out is not None else None
            action_out = torch.stack(action_out, dim=0) if action_out is not None else None
            depth_out = torch.stack(depth_out, dim=0) if depth_out is not None else None

        # return（depth_out 仅在 depth 非 None 时非 None）
        return video_out, action_out, depth_out

    def _prepend_recovery(self, und_token, und_grid_sizes, und_seq_lens, recovery):
        """Insert recovery tokens right after each sample's real und tokens.

        Samples without a valid recovery condition keep their length, so the
        inserted rows fall into masked padding: the condition is absent, not a
        zero-valued pseudo condition.
        """
        if self.recovery_encoder is None:
            raise ValueError("recovery condition given but enable_recovery_condition=False")
        feats = recovery["features"]
        b = und_token.size(0)
        valid = recovery.get("valid")
        if valid is None:
            valid = torch.ones(b, dtype=torch.bool, device=und_token.device)
        rec = self.recovery_encoder(feats.to(und_token.device)).to(und_token.dtype)   # [B,R,D]
        r = rec.size(1)
        lens = und_seq_lens.to(und_token.device).long()
        out = und_token.new_zeros(b, und_token.size(1) + r, und_token.size(2))
        new_lens = lens.clone()
        for i in range(b):
            n = int(lens[i])
            out[i, :n] = und_token[i, :n]
            if bool(valid[i]):
                out[i, n:n + r] = rec[i]
                new_lens[i] = n + r
        grid = und_grid_sizes.clone()
        grid[:, 0] = out.size(1)
        return out, grid, new_lens.to(und_seq_lens.device)

    def _forward_temporal(self, request, history, und_token, und_grid_sizes, und_seq_lens,
                          video, action, depth, t_video, t_action, t_depth, state, robot_names,
                          return_list, depth_read_by_action, depth_detach_video_kv,
                          depth_read_by_video, depth_read_by_und, video_read_by_action,
                          timing, timing_sync_cuda, timing_level, prefix_cache=None):
        from thinker_wm.models.temporal_layout import (
            Segment, TemporalLayout, action_segment, action_step_roles, entries_action_segment,
            entries_video_segment, history_segment, latent_roles, packed_action_entries,
            packed_latent_entries, token_timesteps, video_segment,
        )
        grid = request.grid
        b = und_token.size(0)
        ps = self.video_branch.patch_size
        layout = TemporalLayout(request)
        layout.prefix_cache = prefix_cache
        hist_tokens = None
        t_vid = t_dep = t_act = None

        def latent_seg(x, depth_stream):
            # packed: x is already the packed stream (temporal_training.pack).
            f_n, hh, ww = x.shape[2:]
            h, w = hh // ps[1], ww // ps[2]
            if not request.packed:
                return video_segment(latent_roles(request, f_n, depth=depth_stream), grid, h, w,
                                     device=x.device)
            entries = packed_latent_entries(request, depth=depth_stream)
            if len(entries) != f_n:
                raise ValueError(f"packed {'depth' if depth_stream else 'video'} has {f_n} latents, "
                                 f"request expects {len(entries)}")
            return entries_video_segment(entries, h, w, device=x.device)

        if video is not None:
            cur = latent_seg(video, False)
            if history is not None:
                if self.history_encoder is None:
                    raise ValueError("history given but enable_history=False")
                hp, wp = self.history_encoder.grid(video.shape[3] // ps[1], video.shape[4] // ps[2])
                hseg = history_segment(history["times"].to(video.device),
                                       history["valid"].to(video.device), grid, hp, wp,
                                       pool=self.history_encoder.pool,
                                       source_hw=(video.shape[3] // ps[1], video.shape[4] // ps[2]))
                hist_tokens = self.history_encoder(
                    self.video_branch.patch_embedding, history["latent"].to(video.device, video.dtype),
                    history["times"].to(video.device), history["valid"].to(video.device),
                    history.get("kind"), history.get("state"))
                layout.history_logit_bias = self.history_encoder.logit_bias
                layout.video = Segment.cat([hseg, cur], b)
            else:
                layout.video = cur
            t_vid = token_timesteps(t_video.to(video.device), layout.video)
        elif history is not None:
            raise ValueError("history needs the video stream present")
        if depth is not None:
            layout.depth = latent_seg(depth, True)
            t_dep = token_timesteps((t_depth if t_depth is not None else t_video).to(depth.device),
                                    layout.depth)
        has_state_token = action is not None and state is not None
        if action is not None:
            if request.packed:
                entries = packed_action_entries(request)
                if len(entries) != action.size(1):
                    raise ValueError(f"packed action has {action.size(1)} steps, "
                                     f"request expects {len(entries)}")
                layout.action = entries_action_segment(entries, grid, has_state_token,
                                                       device=action.device)
            else:
                layout.action = action_segment(action_step_roles(request, action.size(1)), grid,
                                               has_state_token, device=action.device)
            ta = t_action.to(action.device)
            if has_state_token and ta.dim() == 2 and ta.size(1) == len(layout.action) - 1:
                ta = torch.cat([ta.new_zeros(ta.size(0), 1), ta], dim=1)
            t_full = token_timesteps(ta, layout.action)
            t_act = t_full[:, 1:] if has_state_token else t_full

        video_token = video_grid_sizes = video_seq_lens = video_e = video_e0 = None
        if video is not None:
            with timed_section(timing, "wm_video_encode_seconds", timing_sync_cuda):
                video_token, video_grid_sizes, video_seq_lens, video_e, video_e0 = (
                    self.video_branch.encode_temporal(
                        video, t_vid, und_token, layout, "video", context_lens=und_seq_lens,
                        robot_names=robot_names, prefix_tokens=hist_tokens,
                        timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level))
        action_token = action_grid_sizes = action_seq_lens = action_e0 = None
        if action is not None:
            with timed_section(timing, "wm_action_encode_seconds", timing_sync_cuda):
                action_token, action_grid_sizes, action_seq_lens, action_e0 = self.action_branch.encode(
                    action, t_act, und_token, robot_names=robot_names, context_lens=und_seq_lens,
                    state=state if has_state_token else None,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level,
                    layout=layout)
        depth_token = depth_grid_sizes = depth_seq_lens = depth_e = depth_e0 = None
        if depth is not None:
            assert self.enable_depth, "forward(depth=...) 需 enable_depth=True"
            with timed_section(timing, "wm_depth_encode_seconds", timing_sync_cuda):
                depth_token, depth_grid_sizes, depth_seq_lens, depth_e, depth_e0 = (
                    self.video_branch.encode_temporal(
                        depth, t_dep, und_token, layout, "depth", context_lens=und_seq_lens,
                        robot_names=robot_names,
                        timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level))

        with timed_section(timing, "wm_uni_process_seconds", timing_sync_cuda):
            video_token, action_token, depth_token, und_token = self.uni_branch.uni_process(
                video_token, video_grid_sizes,
                action_token, action_grid_sizes,
                und_token, und_grid_sizes,
                t_vid, t_act, video_seq_lens, action_seq_lens, und_seq_lens,
                depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
                depth_seq_lens=depth_seq_lens, t_depth=t_dep,
                depth_read_by_action=depth_read_by_action,
                depth_detach_video_kv=depth_detach_video_kv,
                depth_read_by_video=depth_read_by_video,
                depth_read_by_und=depth_read_by_und,
                video_read_by_action=video_read_by_action,
                has_state_token=has_state_token,
                timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level,
                layout=layout)

        video_out = action_out = depth_out = None
        if video_token is not None:
            with timed_section(timing, "wm_video_decode_seconds", timing_sync_cuda):
                video_out = self.video_branch.decode_temporal(
                    video_token, und_token, video_grid_sizes, video_e, video_e0, layout, "video",
                    context_lens=und_seq_lens, robot_names=robot_names,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)
        if action_token is not None:
            with timed_section(timing, "wm_action_decode_seconds", timing_sync_cuda):
                action_out = self.action_branch.decode(
                    action_token, und_token, action_grid_sizes, action_seq_lens, action_e0,
                    robot_names=robot_names, context_lens=und_seq_lens,
                    has_state_token=has_state_token,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level,
                    layout=layout)
        if depth_token is not None:
            with timed_section(timing, "wm_depth_decode_seconds", timing_sync_cuda):
                depth_out = self.video_branch.decode_temporal(
                    depth_token, und_token, depth_grid_sizes, depth_e, depth_e0, layout, "depth",
                    context_lens=und_seq_lens, robot_names=robot_names,
                    timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)
        if not return_list:
            video_out = torch.stack(video_out, dim=0) if video_out is not None else None
            action_out = torch.stack(action_out, dim=0) if action_out is not None else None
            depth_out = torch.stack(depth_out, dim=0) if depth_out is not None else None
        return video_out, action_out, depth_out
