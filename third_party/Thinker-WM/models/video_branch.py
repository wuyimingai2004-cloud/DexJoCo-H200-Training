import copy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from thinker_wm.wan.modules.model import WanModel, sinusoidal_embedding_1d
from motus_monitor.timing import block_timing, timed_section

class VideoBranch(nn.Module):
    def __init__(self, wan_model:WanModel,
                 encode_layer_num, decode_layer_num,
                 gradient_checkpointing: bool = False,
                 robot_token_embedding=None,
                 enable_depth: bool = False):
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
        # 注意：不引用 wan_model.text_embedding——它只服务已移除的 T5
        # embedding 兼容路径（评审 #10），由 UniWorldModel 冻结以退出 optimizer。
        self.time_embedding = wan_model.time_embedding
        self.time_projection = wan_model.time_projection

        self.encode_blocks = wan_model.blocks[:encode_layer_num]
        self.decode_block = wan_model.blocks[-decode_layer_num:]

        self.head = wan_model.head

        self.freqs = wan_model.freqs
        self.gradient_checkpointing = gradient_checkpointing
        # 每机器人 8 个可学习 token（models/robot_tokens.py）：robot_names 非
        # None 时拼到 cross-attn context（und token）之前，context_lens 相应 +8。
        self.robot_token_embedding = robot_token_embedding

        # Depth 第三流（Rynn 式 depth-as-video）：复用同一冻结 Wan VAE + 共享
        # encode/decode transformer blocks，只新增 depth 专属 patch embedding /
        # output head（warm-start 复制 video 权重——depth latent 与 video latent
        # 同 VAE、同 shape、统计相近）+ 零初始化 stream_embed（加到每个 depth
        # token，让共享块区分 depth / RGB）。enable_depth=False 时不建，与原模型
        # 逐字节一致。
        self.enable_depth = bool(enable_depth)
        if self.enable_depth:
            self.depth_patch_embedding = copy.deepcopy(wan_model.patch_embedding)
            self.depth_head = copy.deepcopy(wan_model.head)
            self.depth_stream_embed = nn.Parameter(torch.zeros(1, 1, self.dim))

    def _run_block(self, block, x, e0, context, kwargs):
        if not (self.gradient_checkpointing and self.training):
            return block(x, **kwargs)

        def block_forward(x, e0, context):
            block_kwargs = dict(kwargs)
            block_kwargs["e"] = e0
            block_kwargs["context"] = context
            return block(x, **block_kwargs)

        return checkpoint(
            block_forward,
            x,
            e0,
            context,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    
    def encode(self, x, t, context, context_lens=None, robot_names=None, timing=None, timing_sync_cuda=False, timing_level=1):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (Tensor):
                已投影到 wan_dim 的 und token，shape [B, L_und, dim]，直接作为
                cross-attn 的 K/V
            context_lens (Tensor, optional):
                context（und token）逐样本真实长度 [B]；为 None 时按全长处理
                （旧行为）。训练路径应传入以 mask 掉 VLM padding 位（评审 #1）。
            robot_names (list[str], optional):
                非 None 时每机器人 8 个可学习 token 拼到 context 之前
                （context_lens 相应 +8，与 VLM padding mask 语义一致）
            seq_len (`int`):
                Maximum sequence length for positional encoding

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens[0]
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        
        #print('x:',x.shape)
        #print('t:',t.shape)
        #print('context:',context.shape)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(x.size(0), seq_len)
        elif t.dim() == 2 and t.size(1) == 1:
            #t = t.expand(1, seq_len)
            t = t.expand(t.size(0),seq_len)

        # 首帧是干净条件帧（teacher forcing）：其全部 h×w 个 token 的时间
        # 标签置 0（对齐 Wan2.2 TI2V 官方 expand_timesteps 的整帧处理，
        # 见 wan/textimage2video.py 的 temp_ts），而非只置第一个 token。
        t = t.clone()  # Make a copy to avoid in-place modification
        tokens_per_frame = int(grid_sizes[0, 1] * grid_sizes[0, 2])
        t[:, :tokens_per_frame] = 0.0
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context: und token 已投影到 wan_dim ([B, L_und, dim])，直接作为
        # cross-attn 的 K/V。context_lens 为逐样本真实长度，供 cross-attn
        # mask 掉 VLM padding 位（None = 全长，旧行为）。T5 文本 embedding
        # 列表的兼容路径已随 T5 一并移除（评审 #10）。
        if not torch.is_tensor(context):
            raise TypeError(
                "VideoBranch.encode context must be a projected und token "
                "tensor ([B, L_und, dim]); T5 embedding list input is no "
                "longer supported."
            )

        # robot token（每机器人 8 个可学习 token）拼到 und context 之前，
        # 作为额外的全局条件（context_lens 相应 +8，与 VLM padding mask 一致）。
        rt = None
        if robot_names is not None and self.robot_token_embedding is not None:
            rt = self.robot_token_embedding(robot_names)

        # cross-attn context 循环外一次性构建：[rt, und]（robot token 最前；
        # context_lens 为 None 时保持 None = 全长旧行为）。
        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        bt = block_timing(timing, timing_level)
        for i, block in enumerate(self.encode_blocks):
            with timed_section(bt, f"wm_video_encode_block_{i:02d}_seconds", timing_sync_cuda):
                x = self._run_block(block, x, e0, context, kwargs)
        return x, grid_sizes, seq_lens, e, e0
    
    def decode(self, x, context, grid_sizes, seq_lens, e, e0, context_lens=None, robot_names=None, timing=None, timing_sync_cuda=False, timing_level=1):
        device = self.time_embedding[0].weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # context_lens：uni 输出 und token 的逐样本真实长度（None = 全长，
        # 旧行为）；训练路径传入以 mask 掉 padding 位（评审 #1）。

        # robot token（每机器人 8 个可学习 token）拼到 und context 之前，
        # 与 encode 侧保持一致（context_lens 相应 +8）。
        rt = None
        if robot_names is not None and self.robot_token_embedding is not None:
            rt = self.robot_token_embedding(robot_names)

        # cross-attn context 循环外一次性构建：[rt, und]（规则同 encode）。
        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
        )
        bt = block_timing(timing, timing_level)
        for i, block in enumerate(self.decode_block):
            with timed_section(bt, f"wm_video_decode_block_{i:02d}_seconds", timing_sync_cuda):
                x = self._run_block(block, x, e0, context, kwargs)
        
        x = self.head(x, e)

        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def encode_depth(self, x, t, context, context_lens=None, robot_names=None, timing=None, timing_sync_cuda=False, timing_level=1, first_frame_conditioned=True):
        r"""Depth-stream encoder: mirror of ``encode`` but with the depth-specific
        patch embedding + a learnable stream_embed added to every depth token.
        Shares self.encode_blocks weights with the video path. x is a list of
        depth latents [C_in, F, H, W] (same shape/VAE as video latents)."""
        assert self.enable_depth
        device = self.depth_patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.depth_patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens[0]
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # stream_embed（零初始化）加到每个 depth token：让共享块区分 depth / RGB。
        x = x + self.depth_stream_embed.to(x.dtype)

        if t.dim() == 1:
            t = t.unsqueeze(1).expand(x.size(0), seq_len)
        elif t.dim() == 2 and t.size(1) == 1:
            t = t.expand(t.size(0), seq_len)
        # 首帧是干净条件帧：其全部 h×w token 时间标签置 0（与 video 一致）。
        t = t.clone()
        tokens_per_frame = int(grid_sizes[0, 1] * grid_sizes[0, 2])
        if first_frame_conditioned:
            t[:, :tokens_per_frame] = 0.0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        rt = None
        if robot_names is not None and self.robot_token_embedding is not None:
            rt = self.robot_token_embedding(robot_names)
        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        bt = block_timing(timing, timing_level)
        for i, block in enumerate(self.encode_blocks):
            with timed_section(bt, f"wm_depth_encode_block_{i:02d}_seconds", timing_sync_cuda):
                x = self._run_block(block, x, e0, context, kwargs)
        return x, grid_sizes, seq_lens, e, e0

    def decode_depth(self, x, context, grid_sizes, seq_lens, e, e0, context_lens=None, robot_names=None, timing=None, timing_sync_cuda=False, timing_level=1):
        r"""Depth-stream decoder: mirror of ``decode`` but with the depth-specific
        output head. Shares self.decode_block weights with the video path."""
        assert self.enable_depth
        device = self.time_embedding[0].weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        rt = None
        if robot_names is not None and self.robot_token_embedding is not None:
            rt = self.robot_token_embedding(robot_names)
        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
        )
        bt = block_timing(timing, timing_level)
        for i, block in enumerate(self.decode_block):
            with timed_section(bt, f"wm_depth_decode_block_{i:02d}_seconds", timing_sync_cuda):
                x = self._run_block(block, x, e0, context, kwargs)

        x = self.depth_head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def _context_with_robot(self, context, context_lens, robot_names):
        if robot_names is not None and self.robot_token_embedding is not None:
            rt = self.robot_token_embedding(robot_names)
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        return context, context_lens

    def _time_e(self, t_tok):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt, seq_len = t_tok.shape
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_tok.flatten())
                .unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        return e, e0

    def _run_masked(self, blocks, x, e0, context, context_lens, layout, name, timing,
                    timing_sync_cuda, timing_level, tag):
        from thinker_wm.models.temporal_attention import self_mask, wan_block_forward_masked
        from thinker_wm.models.temporal_layout import ROLE_TARGET
        seg = layout.segment(name)
        mask, bias = self_mask(layout, name, x.size(0))
        bt = block_timing(timing, timing_level)
        for i, block in enumerate(blocks):
            with timed_section(bt, f"wm_{tag}_block_{i:02d}_seconds", timing_sync_cuda):
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(
                        lambda x, e0, context, block=block: wan_block_forward_masked(
                            block, x, e0, seg.pos, self.freqs, context, context_lens, mask, bias,
                        prefix_cache=getattr(layout, "prefix_cache", None),
                        cache_key=("branch", name, tag, i), fixed=seg.role != ROLE_TARGET),
                        x, e0, context, use_reentrant=False, preserve_rng_state=False)
                else:
                    x = wan_block_forward_masked(
                        block, x, e0, seg.pos, self.freqs, context, context_lens, mask, bias,
                        prefix_cache=getattr(layout, "prefix_cache", None),
                        cache_key=("branch", name, tag, i), fixed=seg.role != ROLE_TARGET)
        return x

    def encode_temporal(self, x, t_tok, context, layout, name="video", context_lens=None,
                        robot_names=None, prefix_tokens=None,
                        timing=None, timing_sync_cuda=False, timing_level=1):
        r"""Block-causal encoder (temporal_layout). Legacy ``encode`` is unchanged.

        x: current-window latents [B,C,F,H,W] (RGB or depth per ``name``).
        t_tok: [B, L_total] per-token noise labels incl. prefix (clean = 0).
        prefix_tokens: [B, Lp, dim] already-embedded history tokens (video only).
        Returns tokens over [prefix | current], current grid, seq_lens, e, e0.
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        depth = name == "depth"
        if depth:
            assert self.enable_depth
        patch = self.depth_patch_embedding if depth else self.patch_embedding
        u = patch(x)
        grid_sizes = torch.tensor(u.shape[2:], dtype=torch.long).unsqueeze(0).expand(x.size(0), 3).contiguous()
        u = u.flatten(2).transpose(1, 2)
        if depth:
            u = u + self.depth_stream_embed.to(u.dtype)
        if prefix_tokens is not None:
            u = torch.cat([prefix_tokens.to(u.dtype), u], dim=1)
        seg = layout.segment(name)
        if u.size(1) != len(seg) or t_tok.shape != (u.size(0), u.size(1)):
            raise ValueError(f"{name}: {u.size(1)} tokens, layout {len(seg)}, t {tuple(t_tok.shape)}")
        e, e0 = self._time_e(t_tok)
        context, context_lens = self._context_with_robot(context, context_lens, robot_names)
        seq_lens = torch.full((u.size(0),), u.size(1), dtype=torch.long)
        u = self._run_masked(self.encode_blocks, u, e0, context, context_lens, layout, name,
                             timing, timing_sync_cuda, timing_level, f"{name}_encode")
        return u, grid_sizes, seq_lens, e, e0

    def decode_temporal(self, x, context, grid_sizes, e, e0, layout, name="video",
                        context_lens=None, robot_names=None,
                        timing=None, timing_sync_cuda=False, timing_level=1):
        """Masked decoder; drops the history prefix before unpatchify."""
        device = self.time_embedding[0].weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        context, context_lens = self._context_with_robot(context, context_lens, robot_names)
        x = self._run_masked(self.decode_block, x, e0, context, context_lens, layout, name,
                             timing, timing_sync_cuda, timing_level, f"{name}_decode")
        head = self.depth_head if name == "depth" else self.head
        x = head(x, e)
        n_prefix = layout.segment(name).n_prefix
        x = self.unpatchify(x[:, n_prefix:], grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
            List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

if __name__ == "__main__":
    import torch
    
    device = torch.device("cuda:2")
    
    wan_model = WanModel(
        model_type='t2v',
        patch_size=(1, 1, 1),
        text_len=512,
        in_dim=16,
        dim=256,
        ffn_dim=1024,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=8,
        num_layers=6,
    ).to(device)

    from thinker_wm.models.robot_tokens import RobotTokenEmbedding
    rt_emb = RobotTokenEmbedding(
        {"smoke-robot": 0}, max_robots=64, num_tokens=8, dim=256,
    ).to(device)
    
    video_branch = VideoBranch(
        wan_model,
        encode_layer_num=2,
        decode_layer_num=2,
        robot_token_embedding=rt_emb,
    ).to(device)
    
    frames = 1
    
    c_in = 16
    dim = 256
    ffn_dim = 1024
    num_heads = 8
    sub_dim = dim // num_heads
    bs = 1
    
    L_video = 16
    L_action = 1
    L_und = 4
    L_context = 512
    
    video = [torch.randn(c_in, frames, 4, 4, device=device)]
    t = torch.tensor([10], dtype=torch.long, device=device)
    # context 为已投影到 wan_dim 的 und token（[B, L_und, dim]）
    context = torch.randn(bs, L_und, dim, device=device)
    # robot_names 非 None：robot token 拼入 context（L_und -> L_und+8），
    # context_lens 相应 +8（训练路径语义）。
    ctx_lens = torch.tensor([L_und], dtype=torch.long, device=device)
    robot_names = ["smoke-robot"]
    
    with torch.no_grad():
        print(video[0].size()) # [16, 1, 4, 4]
        x, grid_sizes, seq_lens, e, e0 = video_branch.encode(video, t, context,
                                                             context_lens=ctx_lens,
                                                             robot_names=robot_names)
        print(x.size()) # [1, 16, 256]
        x = video_branch.decode(x, context, grid_sizes, seq_lens, e, e0,
                                context_lens=ctx_lens,
                                robot_names=robot_names)
        print(x[0].size()) # [16, 1, 4, 4]
    print("video_branch smoke OK")
