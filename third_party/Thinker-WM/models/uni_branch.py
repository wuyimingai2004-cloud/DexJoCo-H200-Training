import copy
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from thinker_wm.wan.modules.model import (
    WanModel,
    WanSelfAttention,
    WanCrossAttention,
    WanAttentionBlock,
    rope_apply,
    flash_attention,
    sinusoidal_embedding_1d
)
from motus_monitor.timing import block_timing, timed_section
from thinker_wm.models.prefix_cache import cached_rows, fixed_rows
from thinker_wm.models.temporal_attention import (
    masked_attention, reader_mask, rope_apply_pos, und_reader_mask,
)


class UniBranch(nn.Module):
    def __init__(self, wan_model:WanModel,
                 encode_layer_num, decode_layer_num,
                 gradient_checkpointing: bool = False,
                 action_rope_time_scale: float = 0.25,
                 depth_uni_clone_layers: int = 0):
        """
        load pre-trained WanModel
        """
        super().__init__()
        assert encode_layer_num + decode_layer_num < len(wan_model.blocks)

        self.patch_size = wan_model.patch_size
        self.in_dim = wan_model.in_dim
        self.dim = wan_model.dim
        self.ffn_dim = wan_model.ffn_dim
        self.freq_dim = wan_model.freq_dim
        self.out_dim = wan_model.out_dim
        self.num_heads = wan_model.num_heads
        self.num_layers = wan_model.num_layers
        self.window_size = wan_model.window_size
        self.qk_norm = wan_model.qk_norm
        self.cross_attn_norm = wan_model.cross_attn_norm
        self.eps = wan_model.eps

        self.patch_embedding = wan_model.patch_embedding
        # 不引用 wan_model.text_embedding：它只服务已移除的 T5 embedding
        # 兼容路径（评审 #10），由 UniWorldModel 冻结以退出 optimizer。
        self.time_embedding = wan_model.time_embedding
        self.time_projection = wan_model.time_projection

        self.uni_blocks = wan_model.blocks[encode_layer_num:-decode_layer_num]

        # depth 专属 uni 层（X-WAM 式克隆）：uni 的最后 N 层给 depth 一份独立
        # 副本，video/action/und 仍走原层。副本是精确 deepcopy；depth 读的
        # video/action K/V 复用原层投影。video K/V 默认 detach（护 RGB）；
        # action K/V 不 detach（让 depth 用运动学）。action 读 depth 也不 detach。
        n_clone = int(depth_uni_clone_layers or 0)
        assert 0 <= n_clone <= len(self.uni_blocks), (
            f"depth_uni_clone_layers={n_clone} 超出 uni 层数 {len(self.uni_blocks)}")
        self.depth_uni_clone_layers = n_clone
        self.depth_uni_first_layer = len(self.uni_blocks) - n_clone
        self.depth_uni_blocks = nn.ModuleList(
            copy.deepcopy(b)
            for b in self.uni_blocks[self.depth_uni_first_layer:]
        ) if n_clone > 0 else None

        self.head = wan_model.head

        self.freqs = wan_model.freqs
        # action 的 RoPE 时间间距由 world_model.py 缩放。V14 有 state
        # 前缀时 grid=[T'+1,1,1]，state@0、动作 j@(j+1)*scale；无前缀
        # 时动作 j@j*scale。未补偿 action_start_offset，h/w 位置恒 0。
        ang = torch.angle(wan_model.freqs)
        self.freqs_action = torch.polar(
            torch.ones_like(ang), ang * action_rope_time_scale)
        self.gradient_checkpointing = gradient_checkpointing
        # MOTUS_UNI_CHECKPOINT_EVERY_K=k>=2: every k-th uni block keeps its
        # activations (no recompute); trades memory for speed.
        self.checkpoint_every_k = int(os.environ.get("MOTUS_UNI_CHECKPOINT_EVERY_K", "1"))
        if self.checkpoint_every_k < 1:
            raise ValueError("MOTUS_UNI_CHECKPOINT_EVERY_K must be >= 1")
    
    @staticmethod
    def self_attn_proc(
            model: WanSelfAttention,
            video_token, video_grid_sizes,
            action_token, action_grid_sizes,
            und_token, und_grid_sizes,
            freqs, freqs_action,
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=None, depth_grid_sizes=None, depth_seq_lens=None,
            depth_read_by_action=False, depth_model=None,
            depth_detach_video_kv=False, depth_read_by_video=False,
            video_read_by_action=True, layout=None):
        r"""
        layout: 可选 TemporalLayout。非 None 时 RoPE 用显式 (f,h,w) 位置，每个
            reader 在模态有向可见性之上再叠加物理时间块可见性（显式 mask）。
        Directed self-attention over A 组 = (video, action, depth) + und（自读）。
        可见性是**有向**的（每个 reader 用自己的 K/V 集合各调一次 flash，数学
        上与稠密 mask 等价）：
          - video reads {video, action}          —— 默认不读 depth（护 RGB 先验）；
            depth_read_by_video=True 时读 {video, action, depth}（RGB↔depth 双向）
          - action reads {video, action}          —— 阶段 A；depth_read_by_action=True
            时读 {video, action, depth}（阶段 B）；video_read_by_action=False
            时从 action 的 K/V 里去掉 video（推理消融 D）；depth K/V 不 detach
          - depth reads {video, action, depth}    —— depth 读 RGB/action + 自读
            depth_detach_video_kv=True 时 video K/V detach（depth loss 改不了
            原 uni 的 RGB 投影）；action K/V 始终不 detach
          - und  reads {und}                        —— 与 A 组的互读在 cross_attn_proc

        video/action/depth 任一可为 None（模态 dropout / 推理缺失）：缺失模态
        跳过、返回 None；und 恒在场。

        RoPE：video/depth 用完整 3D RoPE（depth latent 与 video 同 [F,H,W]）；
        action 用时间缩放的 freqs_action；und 用 1D text-like RoPE。

        depth_model: 克隆层（UniBranch.depth_uni_blocks[j]）的 self_attn。非 None
            时 depth 的 q/k/v/o 全部走它；depth 读的 video/action K/V 复用
            `model` 已算好的投影。video K/V 是否 detach 见 depth_detach_video_kv。
        """
        b, n, d = und_token.shape[0], model.num_heads, model.head_dim
        L_und = und_token.size(1)

        uq = model.norm_q(model.q(und_token)).view(b, L_und, n, d)
        uk = model.norm_k(model.k(und_token)).view(b, L_und, n, d)
        uv = model.v(und_token).view(b, L_und, n, d)
        # und: 1D text-like rope (grid [L,1,1])
        uq = rope_apply(uq, und_grid_sizes, freqs)
        uk = rope_apply(uk, und_grid_sizes, freqs)

        # 在场 A 组模态各自 q/k/v + RoPE（video/depth: 3D；action: freqs_action）。
        qkv = {}
        a_L = {}
        slens = {}
        for name, tok, gs, rfreqs, sl in (
            ("video", video_token, video_grid_sizes, freqs, video_seq_lens),
            ("action", action_token, action_grid_sizes, freqs_action, action_seq_lens),
            ("depth", depth_token, depth_grid_sizes, freqs, depth_seq_lens),
        ):
            if tok is None:
                continue
            # 克隆层里 depth 用副本的投影，其余模态仍用原层。
            proj = depth_model if (name == "depth" and depth_model is not None) else model
            Ln = tok.size(1)
            q = proj.norm_q(proj.q(tok)).view(b, Ln, n, d)
            if getattr(layout, "prefix_cache", None) is None:
                k = proj.norm_k(proj.k(tok)).view(b, Ln, n, d)
                v = proj.v(tok).view(b, Ln, n, d)
            else:
                kv = cached_rows(layout, ("uni-self-kv", id(model), name), name, tok,
                                 lambda u: torch.cat([proj.norm_k(proj.k(u)), proj.v(u)], dim=-1))
                k, v = (u.view(b, Ln, n, d) for u in kv.chunk(2, dim=-1))
            if layout is not None:
                pos = layout.segment(name).pos
                q = rope_apply_pos(q, pos, rfreqs)
                k = rope_apply_pos(k, pos, rfreqs)
            else:
                q = rope_apply(q, gs, rfreqs)
                k = rope_apply(k, gs, rfreqs)
            qkv[name] = (q, k, v)
            a_L[name] = Ln
            slens[name] = sl

        def kv_sources(reader):
            if reader == "video":
                srcs = (["video", "action", "depth"] if depth_read_by_video
                        else ["video", "action"])
            elif reader == "depth":
                srcs = ["video", "action", "depth"]
            else:  # action
                srcs = ["action"]
                if video_read_by_action:
                    srcs.insert(0, "video")
                if depth_read_by_action:
                    srcs.append("depth")
            return [s for s in srcs if s in qkv]

        # 每个 reader 用自己的 K/V 集合各调一次 flash（有向可见性）。
        a_out = {}
        for reader in ("video", "action", "depth"):
            if reader not in qkv:
                continue
            srcs = kv_sources(reader)
            ks, vs = [], []
            for s in srcs:
                k_s, v_s = qkv[s][1], qkv[s][2]
                # 只切 depth←RGB：前向仍读图，depth loss 不回原 uni 的 video 投影。
                if depth_detach_video_kv and reader == "depth" and s == "video":
                    k_s, v_s = k_s.detach(), v_s.detach()
                ks.append(k_s)
                vs.append(v_s)
            k = torch.cat(ks, dim=1)
            v = torch.cat(vs, dim=1)
            if layout is not None:
                mask, bias = reader_mask(layout, reader, srcs, b)
                a_out[reader] = masked_attention(qkv[reader][0], k, v, mask, bias)
                continue
            k_lens = None
            for s in srcs:
                k_lens = slens[s] if k_lens is None else k_lens + slens[s]
            a_out[reader] = flash_attention(
                qkv[reader][0], k, v,
                q_lens=slens[reader], k_lens=k_lens,
                window_size=model.window_size,
            )

        # und <-> und。q 侧全长（varlen 输出要求）；k 侧用真实长度 mask VLM padding。
        und_token = flash_attention(
            uq, uk, uv,
            q_lens=None, k_lens=und_seq_lens,
            window_size=model.window_size,
        )

        order = [m for m in ("video", "action", "depth") if m in qkv]
        # 输出投影：克隆层里 depth 段走副本的 o，其余段仍一次算完。
        split_depth_o = depth_model is not None and "depth" in a_out
        main = [m for m in order if not (split_depth_o and m == "depth")]
        x = torch.cat([a_out[m] for m in main] + [und_token], dim=1).flatten(2)
        x = model.o(x)
        parts = dict(zip(main + ["und"], x.split([a_L[m] for m in main] + [L_und], dim=1)))
        if split_depth_o:
            parts["depth"] = depth_model.o(a_out["depth"].flatten(2))
        return (
            parts.get("video"),
            parts.get("action"),
            parts.get("depth"),
            parts["und"],
        )
    
    @staticmethod
    def cross_attn_proc(
            model: WanCrossAttention,
            video_token, action_token, und_token,
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=None, depth_seq_lens=None, depth_model=None,
            depth_read_by_und=False, layout=None):
        r"""
        layout: 非 None 时 B -> A（und 读 A 组）只汇总已观测 token（历史与
            block<=0 的 OBS），und 不再把带噪/未来块广播回所有分支。

        有向 cross-attention，A 组 = (video, action, depth)，B = und：
          - A -> B: video+action+**depth** query und（depth 读文本条件）；
          - B -> A: und 默认只 query video+action；depth_read_by_und=True 时
            也读 depth（und↔depth 双向）。旧默认是堵住 depth→und→video
            间接泄漏；RGB 已直接读 depth 后这条保护不再必要。
        Self-terms 在 self_attn_proc。Cross-attention 无 RoPE（同原 Wan）。

        video/action/depth 任一可为 None（模态 dropout / 推理缺失）：跳过、
        返回 None；und 恒在场。A 组全缺失（不应发生）时 B->A 退化为恒等。

        depth_model: 克隆层的 cross_attn。非 None 时 depth 的 q/o 走它，depth
            读的 und K/V 复用 `model` 的投影并 detach（与 self_attn_proc 同
            策略）；und 读 depth 时用克隆层已算好的 K/V，不 detach。
        """
        b, n, d = und_token.shape[0], model.num_heads, model.head_dim
        L_und = und_token.size(1)

        uq = model.norm_q(model.q(und_token)).view(b, L_und, n, d)
        uk = model.norm_k(model.k(und_token)).view(b, L_und, n, d)
        uv = model.v(und_token).view(b, L_und, n, d)
        # A->B 的 K 侧用 und 真实长度（mask VLM padding 位）。
        ab_k_lens = und_seq_lens

        qkv = {}
        a_L = {}
        slens = {}
        for name, tok, sl in (
            ("video", video_token, video_seq_lens),
            ("action", action_token, action_seq_lens),
            ("depth", depth_token, depth_seq_lens),
        ):
            if tok is None:
                continue
            # 克隆层里 depth 用副本的投影，其余模态仍用原层。
            proj = depth_model if (name == "depth" and depth_model is not None) else model
            Ln = tok.size(1)
            q = proj.norm_q(proj.q(tok)).view(b, Ln, n, d)
            if getattr(layout, "prefix_cache", None) is None:
                k = proj.norm_k(proj.k(tok)).view(b, Ln, n, d)
                v = proj.v(tok).view(b, Ln, n, d)
            else:
                kv = cached_rows(layout, ("uni-cross-kv", id(model), name), name, tok,
                                 lambda u: torch.cat([proj.norm_k(proj.k(u)), proj.v(u)], dim=-1))
                k, v = (u.view(b, Ln, n, d) for u in kv.chunk(2, dim=-1))
            qkv[name] = (q, k, v)
            a_L[name] = Ln
            slens[name] = sl

        # A -> B : video+action+depth query und（q_lens 用在场组逐样本长度和，
        # k_lens 用 und 真实长度 mask VLM padding 位）。克隆层里 depth 单独一次
        # flash（K/V 用 detach 的 und 投影），其余模态仍合成一次。
        order = [m for m in ("video", "action", "depth") if m in qkv]
        split_depth = depth_model is not None and "depth" in qkv
        a_out = {}
        main = [m for m in order if not (split_depth and m == "depth")]
        if main:
            a_q = torch.cat([qkv[m][0] for m in main], dim=1)
            group_seq_lens = None
            for m in main:
                group_seq_lens = slens[m] if group_seq_lens is None else group_seq_lens + slens[m]
            out_ab = flash_attention(
                a_q, uk, uv,
                q_lens=group_seq_lens,
                k_lens=ab_k_lens,
                window_size=model.window_size,
            )
            a_out = dict(zip(main, out_ab.split([a_L[m] for m in main], dim=1)))
        if split_depth:
            a_out["depth"] = flash_attention(
                qkv["depth"][0], uk.detach(), uv.detach(),
                q_lens=slens["depth"],
                k_lens=ab_k_lens,
                window_size=depth_model.window_size,
            )

        # B -> A : und query video+action，可选再读 depth。q 侧全长，k 侧全长。
        b_srcs = ["video", "action", "depth"] if depth_read_by_und else ["video", "action"]
        b_order = [m for m in b_srcs if m in qkv]
        if b_order:
            b_k = torch.cat([qkv[m][1] for m in b_order], dim=1)
            b_v = torch.cat([qkv[m][2] for m in b_order], dim=1)
            if layout is not None:
                und_from_ab = masked_attention(
                    uq, b_k, b_v, und_reader_mask(layout, b_order, L_und, b))
            else:
                b_seq_lens = None
                for m in b_order:
                    b_seq_lens = slens[m] if b_seq_lens is None else b_seq_lens + slens[m]
                und_from_ab = flash_attention(
                    uq, b_k, b_v,
                    q_lens=None,
                    k_lens=b_seq_lens,
                    window_size=model.window_size,
                )
        else:
            # A 组全缺失（不应发生）：und 恒等。
            und_from_ab = uq

        # 输出投影：克隆层里 depth 段走副本的 o，其余段仍一次算完。
        x = torch.cat([a_out[m] for m in main] + [und_from_ab], dim=1).flatten(2)
        x = model.o(x)
        parts = dict(zip(main + ["und"], x.split([a_L[m] for m in main] + [L_und], dim=1)))
        if split_depth:
            parts["depth"] = depth_model.o(a_out["depth"].flatten(2))
        return (
            parts.get("video"),
            parts.get("action"),
            parts.get("depth"),
            parts["und"],
        )

    @staticmethod
    def attn_block_proc(
            model: WanAttentionBlock,
            video_token, video_grid_sizes,
            action_token, action_grid_sizes,
            und_token, und_grid_sizes,
            freqs, freqs_action, e,
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=None, depth_grid_sizes=None, depth_seq_lens=None,
            depth_read_by_action=False, depth_model=None,
            depth_detach_video_kv=False, depth_read_by_video=False,
            depth_read_by_und=False, video_read_by_action=True, layout=None):
        r"""
        Args:
            layout: 可选 TemporalLayout（块级 AR），透传给 self/cross attention。
            video_token/action_token/depth_token/und_token(Tensor): [B, L, C]。
                video/action/depth 任一可为 None（模态 dropout / 推理缺失）；
                und 恒在场。可见性有向（见 self_attn_proc/cross_attn_proc）。
            e(Tensor): [B, L_total, 6, C]，段顺序 (video, action, depth, und)。
            depth_read_by_action(bool): 阶段 B 开关，True 时 action 读 depth。
            video_read_by_action(bool): False 时 action 不读 video（推理消融 D）。
            depth_read_by_video(bool): True 时 video 读 depth（RGB↔depth 双向）。
            depth_read_by_und(bool): True 时 und 读 depth（und↔depth 双向）。
            depth_detach_video_kv(bool): True 时 depth 读 video 的 K/V detach。
            depth_model(WanAttentionBlock): depth 专属克隆层（见 UniBranch
                .depth_uni_blocks）。非 None 时 depth 段的 modulation / self_attn
                / norm3 / cross_attn / ffn 全部走副本；norm1、norm2 是
                elementwise_affine=False 的无参 LayerNorm，两份权重等价，故仍在
                全段上一次算完。
        """
        assert e.dtype == torch.float32

        tokens = {"video": video_token, "action": action_token,
                  "depth": depth_token, "und": und_token}
        # 段顺序固定 (video, action, depth, und)，与 uni_process 构建 e 的顺序一致。
        present = [m for m in ("video", "action", "depth", "und") if tokens[m] is not None]
        split_depth = depth_model is not None and "depth" in present

        def cat_present(dct):
            return torch.cat([dct[m] for m in present], dim=1)

        def split_present(x, ref):
            return dict(zip(present, x.split([ref[m].size(1) for m in present], dim=1)))

        def seg_apply(fn, depth_fn, dct):
            """按段应用有参模块：depth 段走 depth_fn，其余段合并一次算完。"""
            if not split_depth:
                return split_present(fn(cat_present(dct)), dct)
            main = [m for m in present if m != "depth"]
            out = fn(torch.cat([dct[m] for m in main], dim=1))
            res = dict(zip(main, out.split([dct[m].size(1) for m in main], dim=1)))
            res["depth"] = depth_fn(dct["depth"])
            return res

        with torch.amp.autocast('cuda', dtype=torch.float32):
            if split_depth:
                # modulation 是逐段的可学习偏置：depth 段用副本的那份。
                dim = model.modulation.shape[-1]
                mod = torch.cat([
                    (depth_model if m == "depth" else model).modulation
                    .unsqueeze(0).expand(1, tokens[m].size(1), 6, dim)
                    for m in present
                ], dim=1)
            else:
                mod = model.modulation.unsqueeze(0)
            e = (mod + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention（有向可见性在 self_attn_proc 内部处理）
        normed = (model.norm1(cat_present(tokens)).float()
                  * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        normed = split_present(normed, tokens)

        sa = UniBranch.self_attn_proc(
            model.self_attn,
            normed.get("video"), video_grid_sizes,
            normed.get("action"), action_grid_sizes,
            normed["und"], und_grid_sizes,
            freqs, freqs_action,
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=normed.get("depth"), depth_grid_sizes=depth_grid_sizes,
            depth_seq_lens=depth_seq_lens, depth_read_by_action=depth_read_by_action,
            depth_model=depth_model.self_attn if split_depth else None,
            depth_detach_video_kv=depth_detach_video_kv,
            depth_read_by_video=depth_read_by_video,
            video_read_by_action=video_read_by_action,
            layout=layout,
        )
        sa = {"video": sa[0], "action": sa[1], "depth": sa[2], "und": sa[3]}

        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = cat_present(tokens) + torch.cat([sa[m] for m in present], dim=1) * e[2].squeeze(2)
            cur = split_present(x, tokens)

        # cross attention: A(video,action,depth query und) <-> und(query A 组)
        pre_cross = cur
        cur_norm = seg_apply(model.norm3, depth_model.norm3 if split_depth else None, pre_cross)
        ca = UniBranch.cross_attn_proc(
            model.cross_attn,
            cur_norm.get("video"), cur_norm.get("action"), cur_norm["und"],
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=cur_norm.get("depth"), depth_seq_lens=depth_seq_lens,
            depth_model=depth_model.cross_attn if split_depth else None,
            depth_read_by_und=depth_read_by_und,
            layout=layout,
        )
        ca = {"video": ca[0], "action": ca[1], "depth": ca[2], "und": ca[3]}
        for m in present:
            cur[m] = pre_cross[m] + ca[m]

        # FFN（norm2 无参，调制在全段上算完；克隆层里 depth 段单独过副本的 ffn）
        ffn_in = (model.norm2(cat_present(cur)) *
                  (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        cache = getattr(layout, "prefix_cache", None)
        # In Joint/IDM, retaining the full FFN GEMM shape avoids BF16 drift
        # for the small action stream. FDM has no moving action rows.
        action_partition = None if cache is None else cache.partitions.get(("stream", "action"))
        if cache is not None and (action_partition is None or not action_partition.moving.numel()):
            ffn_parts = split_present(ffn_in, cur)
            main = [m for m in present if not (split_depth and m == "depth")]
            main_in = torch.cat([ffn_parts[m] for m in main], dim=1)
            partition_key = ("uni-main", tuple(main))
            partition = cache.partitions.get(partition_key)
            if partition is None:
                fixed = torch.cat([fixed_rows(layout, m, cur[m].size(1), cur[m].device) for m in main])
                partition = cache.partition(partition_key, fixed)
            main_y = cache.rows(("uni-ffn", id(model), "main"), main_in, None,
                                model.ffn, partition=partition)
            ffn_y = dict(zip(main, main_y.split([cur[m].size(1) for m in main], dim=1)))
            if split_depth:
                ffn_y["depth"] = cached_rows(layout, ("uni-ffn", id(model), "depth"),
                                               "depth", ffn_parts["depth"], depth_model.ffn)
            y = torch.cat([ffn_y[m] for m in present], dim=1)
        elif split_depth:
            ffn_y = seg_apply(model.ffn, depth_model.ffn, split_present(ffn_in, cur))
            y = torch.cat([ffn_y[m] for m in present], dim=1)
        else:
            y = model.ffn(ffn_in)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            y = y * e[5].squeeze(2)
            ffn_out = split_present(y, cur)
            for m in present:
                cur[m] = cur[m] + ffn_out[m]

        # return（缺失模态返回 None）
        return (
            cur.get("video"),
            cur.get("action"),
            cur.get("depth"),
            cur["und"],
        )

    def uni_process(self,
                    video_token, video_grid_sizes,
                    action_token, action_grid_sizes,
                    und_token, und_grid_sizes,
                    t_video, t_action,
                    video_seq_lens, action_seq_lens, und_seq_lens,
                    depth_token=None, depth_grid_sizes=None, depth_seq_lens=None,
                    t_depth=None, depth_read_by_action=False,
                    depth_detach_video_kv=False, depth_read_by_video=False,
                    depth_read_by_und=False, video_read_by_action=True,
                    has_state_token=False,
                    timing=None, timing_sync_cuda=False, timing_level=1,
                    depth_first_frame_conditioned=True, layout=None):
        # has_state_token（V14）：action 段第 0 位是否为 state token。由
        # world_model 统一判定后传入，不在此按段长反推。
        # 在场模态（video/action/depth 任一可为 None——模态 dropout / 推理缺失）；
        # und 恒在场（prompt 缺失时上游给 null token，uni 无感知）。
        video_present = video_token is not None
        action_present = action_token is not None
        depth_present = depth_token is not None
        video_seq_len = video_token.size(1) if video_present else 0
        action_seq_len = action_token.size(1) if action_present else 0
        depth_seq_len = depth_token.size(1) if depth_present else 0
        und_seq_len = und_token.size(1)
        seq_len = video_seq_len + action_seq_len + depth_seq_len + und_seq_len

        if t_video is None:
            t_video = t_action
        if t_action is None:
            t_action = t_video
        # depth 默认与 video 同噪声调度（t_depth=None → 走 t_video）。
        if t_depth is None:
            t_depth = t_video
        base_t = t_video if t_video is not None else t_action
        if base_t is None:
            raise ValueError("uni_process: t_video 与 t_action 至少一个非 None")
        if base_t.dim() == 1:
            base_t = base_t.unsqueeze(1)
        if t_video is not None and t_video.dim() == 1:
            t_video = t_video.unsqueeze(1)
        if t_action is not None and t_action.dim() == 1:
            t_action = t_action.unsqueeze(1)
        if t_depth is not None and t_depth.dim() == 1:
            t_depth = t_depth.unsqueeze(1)

        # time embeddings（按在场模态分段构建，段顺序 video/action/depth/und，
        # 与 attn_block_proc 的 present 顺序一致）。
        segs = []
        if video_present and layout is not None:
            # 块级 AR：调用方已按 TemporalLayout 给出逐 token 标签（历史/观测/
            # 干净前缀为 0），不再按“首帧”推断。
            assert t_video.shape == (video_token.size(0), video_seq_len), tuple(t_video.shape)
            segs.append(t_video.float().clone())
        elif video_present:
            # 首帧是干净条件帧：其全部 h×w 个 token 置 0（对齐 Wan2.2 TI2V
            # 官方 expand_timesteps 的整帧处理，而非只置第一个 token）。
            tv = t_video.expand(t_video.size(0), video_seq_len).clone()
            tokens_per_frame = int(video_grid_sizes[0, 1] * video_grid_sizes[0, 2])
            tv[:, :tokens_per_frame] = 0.0
            segs.append(tv)
        if action_present:
            # 20260815—V14 之前：action 段全部为带噪动作 token，无置 0 的特殊位。
            # V14 起 has_state_token=True 时段首是 state token（已观测的干净
            # 条件），与 action_branch.encoder 侧一致地把它的噪声时刻钉 0；
            # 其余 48 个动作位保持各自采样到的时刻。只改 encoder 不改这里，
            # state 会在 uni 里被按动作噪声水平调制，就不再是干净条件。
            # 时间输入契约与 action_branch.encoder 一致：t_action 按“动作步数”
            # 给（标量或逐 token），state 位的 0 在这里补，调用方不必自己拼。
            num_action_steps = action_seq_len - int(has_state_token)
            assert num_action_steps >= 1, (
                f"action 段长度 {action_seq_len} 放不下动作 token"
                f"（has_state_token={has_state_token}）"
            )
            if t_action.size(1) == 1:
                ta = t_action.expand(t_action.size(0), num_action_steps).clone()
            else:
                assert t_action.size(1) == num_action_steps, (
                    f"逐 token 动作时间长度 {tuple(t_action.shape)} 与动作 token 数 "
                    f"{num_action_steps} 不一致"
                )
                ta = t_action.clone()
            if has_state_token:
                ta = torch.cat([ta.new_zeros(ta.size(0), 1), ta], dim=1)
            segs.append(ta)
        if depth_present and layout is not None:
            assert t_depth.shape == (depth_token.size(0), depth_seq_len), tuple(t_depth.shape)
            segs.append(t_depth.float().clone())
        elif depth_present:
            # Only first_frame mode clamps depth time; rgb_only predicts frame 0 too.
            td = t_depth.expand(t_depth.size(0), depth_seq_len).clone()
            depth_tokens_per_frame = int(depth_grid_sizes[0, 1] * depth_grid_sizes[0, 2])
            if depth_first_frame_conditioned:
                td[:, :depth_tokens_per_frame] = 0.0
            segs.append(td)
        # und token 是干净 VLM 条件：整段置 0，不被噪声水平 e(t) 调制。
        tu = base_t.new_zeros(base_t.size(0), und_seq_len)
        segs.append(tu)
        t = torch.cat(segs, dim=1)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        self.freqs = self.freqs.to(und_token.device)
        self.freqs_action = self.freqs_action.to(und_token.device)
        bt = block_timing(timing, timing_level)
        for i, block in enumerate(self.uni_blocks):
            # 克隆区（uni 最后 N 层）：depth 走自己的副本，video/action/und 走原层。
            depth_block = (
                self.depth_uni_blocks[i - self.depth_uni_first_layer]
                if (self.depth_uni_blocks is not None and i >= self.depth_uni_first_layer)
                else None
            )
            with timed_section(bt, f"wm_uni_block_{i:02d}_seconds", timing_sync_cuda):
                # video/action/depth 可为 None（模态 dropout / 推理缺失）：
                # use_reentrant=False 的 checkpoint 支持非张量输入，缺失模态照样
                # 走 checkpoint —— 否则 dropout 触发时直接前向会把全部 uni 层激活
                # 留在显存，显存暴涨 OOM。
                k = self.checkpoint_every_k
                if self.gradient_checkpointing and self.training and (k == 1 or i % k != k - 1):
                    def block_forward(video_token, action_token, depth_token, und_token, e0,
                                      block=block, depth_block=depth_block):
                        return self.attn_block_proc(
                            model=block,
                            video_token=video_token, video_grid_sizes=video_grid_sizes,
                            action_token=action_token, action_grid_sizes=action_grid_sizes,
                            und_token=und_token, und_grid_sizes=und_grid_sizes,
                            freqs=self.freqs, freqs_action=self.freqs_action, e=e0,
                            video_seq_lens=video_seq_lens,
                            action_seq_lens=action_seq_lens,
                            und_seq_lens=und_seq_lens,
                            depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
                            depth_seq_lens=depth_seq_lens,
                            depth_read_by_action=depth_read_by_action,
                            depth_model=depth_block,
                            depth_detach_video_kv=depth_detach_video_kv,
                            depth_read_by_video=depth_read_by_video,
                            depth_read_by_und=depth_read_by_und,
                            video_read_by_action=video_read_by_action,
                            layout=layout)

                    video_token, action_token, depth_token, und_token = checkpoint(
                        block_forward,
                        video_token,
                        action_token,
                        depth_token,
                        und_token,
                        e0,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    (video_token, action_token, depth_token, und_token) = self.attn_block_proc(
                        model=block,
                        video_token=video_token, video_grid_sizes=video_grid_sizes,
                        action_token=action_token, action_grid_sizes=action_grid_sizes,
                        und_token=und_token, und_grid_sizes=und_grid_sizes,
                        freqs=self.freqs, freqs_action=self.freqs_action, e=e0,
                        video_seq_lens=video_seq_lens,
                        action_seq_lens=action_seq_lens,
                        und_seq_lens=und_seq_lens,
                        depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
                        depth_seq_lens=depth_seq_lens,
                        depth_read_by_action=depth_read_by_action,
                        depth_model=depth_block,
                        depth_detach_video_kv=depth_detach_video_kv,
                        depth_read_by_video=depth_read_by_video,
                        depth_read_by_und=depth_read_by_und,
                        video_read_by_action=video_read_by_action,
                        layout=layout)

        return video_token, action_token, depth_token, und_token

if __name__ == "__main__":
    from math import sqrt
    from thinker_wm.wan.modules.model import rope_params

    device = torch.device("cuda:2")
    frames = 1

    dim = 256
    ffn_dim = 1024
    num_heads = 8
    sub_dim = dim // num_heads
    bs = 1

    L_video = 16
    L_action = 1
    L_und = 4
    L_e = L_video + L_action + L_und

    HW_video = int(sqrt(L_video))

    model = WanAttentionBlock(
        dim = dim,
        ffn_dim = ffn_dim,
        num_heads = num_heads
    ).to(device)

    freqs = torch.cat([
                rope_params(1024, sub_dim - 4 * (sub_dim // 6)),
                rope_params(1024, 2 * (sub_dim // 6)),
                rope_params(1024, 2 * (sub_dim // 6))
            ], dim=1).to(device)

    video_token = torch.randn(bs, L_video, dim).to(device)
    action_token = torch.randn(bs, L_action, dim).to(device)
    und_token = torch.randn(bs, L_und, dim).to(device)

    video_grid_sizes = torch.tensor([[frames, HW_video, HW_video] for _ in range(bs)], dtype=torch.long, device=device)
    action_grid_sizes = torch.tensor([[L_action, 1, 1] for _ in range(bs)], dtype=torch.long, device=device)
    und_grid_sizes = torch.tensor([[L_und, 1, 1] for _ in range(bs)], dtype=torch.long, device=device)

    e = torch.randn(bs, L_e, 6, dim, device=device)

    video_seq_lens = torch.tensor([L_video for _ in range(bs)], device=device)
    action_seq_lens = torch.tensor([L_action for _ in range(bs)], device=device)
    und_seq_lens = torch.tensor([L_und for _ in range(bs)], device=device)

    video_token, action_token, depth_token, und_token = UniBranch.attn_block_proc(
        model,
        video_token, video_grid_sizes,
        action_token, action_grid_sizes,
        und_token, und_grid_sizes,
        freqs, freqs, e,
        video_seq_lens, action_seq_lens, und_seq_lens)

    print(video_token.shape)
    print(action_token.shape)
    print(depth_token)
    print(und_token.shape)

    # 模态 dropout / 推理缺失：action 缺失（None）时 A 组退化为单模态，
    # video 与 und 正常、action 返回 None。
    e_video_only = torch.randn(bs, L_video + L_und, 6, dim, device=device)
    video_only, action_only, depth_only, und_only = UniBranch.attn_block_proc(
        model,
        video_token, video_grid_sizes,
        None, None,
        und_token, und_grid_sizes,
        freqs, freqs, e_video_only,
        video_seq_lens, None, und_seq_lens)
    print(video_only.shape)
    print(action_only)
    print(depth_only)
    print(und_only.shape)
