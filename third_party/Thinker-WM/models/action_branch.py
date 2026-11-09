"""
action(shape[B, T, U]) -> ActionEncoder -> WanModel_Middle_Layer -> ActionDecoder -> action(shape[B, T, U])

统一 action 空间数据流（多机器人，见 .AGENTS_CONTENT/03_config.md）：
  - 模型边界（UBT_WM._scatter_to_unified）把每机器人 [B,..,dim_r] 按
    robot_layout 的 unified_dim_map 语义槽位映射逐维散开到统一维
    [B,..,U]（common.unified_action_dim，其余补 0；未注册 dim_map 的旧配置
    回退 offset 前填）；本模块不再有 per-robot projector 分支。
  - 无投影层：统一维 U 直接作为模型内部 action 维——encoder 的
    action 以 U 维直接进 Conv3d embedding，decoder 的 unembedding
    直接输出 U 维（20260811 起移除 U↔common_action_dim 共享 projector，
    MLP/Linear 均不再存在）。
  - robot token（每机器人 8 个可学习 token，models/robot_tokens.py）在
    encoder/decoder 的 cross-attn context（und token）前拼入（context_lens+8，
    与 VLM padding mask 语义一致）。

超容量增长协议：U 用满时增大 common.unified_action_dim → action
embedding 的 Conv3d 输入通道与 unembedding 输出维沿 U 维零填充扩容
（旧通道原样拷贝、新通道零初始化），load_state_dict 加对应 remap；robot
token 表超 64 槽同理按行扩展（见 models/robot_tokens.py）。YAML
dataset.unified_bands 注册表原为 append-only；20260815 起新分支周期
重定义为 U=20 六 band 紧凑布局，旧 80/64 列 ckpt 经 _remap_unified_eef20
列切片迁移（见 models/ubt_wm.py）。

时序下采样（20260814 起）：动作全部 T 步经 stride=t_downsample 的 Conv3d
分组成 token（每 token d 步，要求 T % d == 0；ratio=6/d=6 时 288 步 →
48 token）。旧"裁 ratio-1 步 + state_unembedding 补步"划分废弃；旧 ckpt
的 embedding/unembedding 权重经 models/ubt_wm.py load_state_dict 的
_remap_action_t_downsample 迁移。

V14（20260918）重新引入 state token：条件帧 qpos 经独立的 state_embedding
（kernel=1 Conv3d，零初始化）编成 1 个 token，**前插**到动作 token 之前，
于是 state@0、动作@1…T''（相对 V13 整体 +1，动作相对 video latent 的时间
对齐也偏一格，属选定方案已接受的代价）。该 token 扩散时间恒 0（encoder 与
uni_branch 各钉一次）、不加噪、不进 loss，decoder 在输出投影前切掉。
state=None 时完整退回上述无 state token 的旧路径。旧 ckpt 的
state_embedding 键不再被 _LEGACY_STATE_DICT_PREFIXES 丢弃。
"""
import torch
import torch.nn as nn

from thinker_wm.wan.modules.model import (
    WanAttentionBlock,
    sinusoidal_embedding_1d,
    rope_params,
)
from motus_monitor.timing import block_timing, timed_section


def _run_action_blocks(blocks, x, kwargs, layout, freqs, timing, timing_sync_cuda,
                       timing_level, tag):
    bt = block_timing(timing, timing_level)
    if layout is None:
        for i, block in enumerate(blocks):
            with timed_section(bt, f"wm_action_{tag}_block_{i:02d}_seconds", timing_sync_cuda):
                x = block(x, **kwargs)
        return x
    from thinker_wm.models.temporal_attention import self_mask, wan_block_forward_masked
    from thinker_wm.models.temporal_layout import ROLE_TARGET
    seg = layout.segment("action")
    if len(seg) != x.size(1):
        raise ValueError(f"action layout has {len(seg)} tokens, sequence {x.size(1)}")
    mask, bias = self_mask(layout, "action", x.size(0))
    for i, block in enumerate(blocks):
        with timed_section(bt, f"wm_action_{tag}_block_{i:02d}_seconds", timing_sync_cuda):
            x = wan_block_forward_masked(block, x, kwargs["e"], seg.pos, freqs,
                                         kwargs["context"], kwargs["context_lens"], mask, bias,
                                         prefix_cache=getattr(layout, "prefix_cache", None),
                                         cache_key=("branch", "action", tag, i), fixed=seg.role != ROLE_TARGET)
    return x


class ActionEncoder(nn.Module):
    def __init__(self,
                 action_dim, layer_num,
                 wan_dim, wan_ffn_dim, wan_num_heads,
                 freq_dim=256, text_len=512, text_dim=4096,
                 t_downsample=24, video_action_freq_ratio=6,
                 unified_action_dim=64,
                 robot_token_embedding=None,
                 robot_configs=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.dim = wan_dim
        self.t_downsample = t_downsample
        self.video_action_freq_ratio = video_action_freq_ratio

        # 统一 action 空间（多机器人）：无投影层，统一维 U 直接作为模型内部
        # action 维（输入为模型边界 scatter 后的 [B,..,U]）。robot_configs
        # 参数保留作调用方签名兼容（不再参与任何逻辑）。
        self.unified_action_dim = unified_action_dim
        self.robot_token_embedding = robot_token_embedding

        # self.action_embedding = nn.Conv3d(
        #     action_dim, wan_dim,
        #     kernel_size=(self.t_downsample, 1, 1),
        #     stride=(self.t_downsample, 1, 1)
        # )
        self.action_embedding = nn.Conv3d(
            unified_action_dim, wan_dim,
            kernel_size=(self.t_downsample, 1, 1),
            stride=(self.t_downsample, 1, 1)
        )

        # state token（V14 起重新引入，见文件头）：条件帧 qpos 单独成 token，
        # 前插到动作 token 之前占 RoPE 时间位 0。输入是模型边界 scatter 后的
        # 统一维 [B, U]，与 action_embedding 同一输入空间，不得硬编码 14。
        self.state_embedding = nn.Conv3d(
            unified_action_dim, wan_dim,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, wan_dim),
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim)
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim * 6)
        )

        self.blocks = nn.ModuleList([
            WanAttentionBlock(wan_dim, wan_ffn_dim, wan_num_heads)
            for _ in range(layer_num)
        ])


        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (wan_dim % wan_num_heads) == 0 and (wan_dim // wan_num_heads) % 2 == 0
        d = wan_dim // wan_num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # initialize weights
        self.init_weights()

    def _check_action_dims(self, action):
        """统一 action 空间：输入已是模型边界 scatter 后的统一维张量
        （action [B, T, U]），无投影层，只校验统一维。

        Args:
            action: Tensor [B, T, unified_action_dim]
        """
        assert action.size(-1) == self.unified_action_dim, (
            f"action dim mismatch: expected unified_action_dim={self.unified_action_dim}, "
            f"got {action.size(-1)}"
        )

    def forward(
        self,
        action,
        t,
        context,
        robot_names=None,
        context_lens=None,
        state=None,
        timing=None, timing_sync_cuda=False, timing_level=1,
        layout=None,
    ):
        """
        layout: 可选 TemporalLayout（块级 AR）；None = 旧的全双向路径。
        action: (B, T, U)
        state: 条件帧 qpos (B, U)，已归一化并 scatter 到统一维；None = 退回
            无 state token 的旧路径（48 token）。非 None 时前插为序列第 0 个
            token，扩散时间钉 0（干净条件，不进 loss）。
        robot_names: List[str] 长度为 B；非 None 时每机器人 8 个可学习 token
            拼到 cross-attn context 之前（context_lens 相应 +8）
        context_lens: context（und token）逐样本真实长度 [B]；None = 全长
            （旧行为）。训练路径应传入以 mask 掉 VLM padding 位（评审 #1）。
        """
        # 统一 action 空间：输入已是模型边界 scatter 后的 [B, T, U]，
        # 无投影层，统一维直接进 Conv3d embedding。
        self._check_action_dims(action)

        # 时序下采样（20260814 起）：无裁剪，动作全部 T 步经 stride=t_downsample
        # 的 Conv3d 分组——每 token 覆盖 d 步（d=ratio 时 288 步 → 48 token）。
        # 旧"裁 ratio-1 步 + state 补步"划分已废弃。V14 的 state 是独立前缀
        # token（见下方拼接处），动作步数本身不受影响，本断言只约束 48 步动作。
        T = action.size(1)
        assert T % self.t_downsample == 0, (
            f"action 步数 T={T} 不能被 t_downsample={self.t_downsample} 整除"
        )

        # (B, T, D) -> (B, D, T, 1, 1) 进 Conv3d（统一 action 空间下 batch 内
        # T 齐次，无需 padding）
        x = action.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        device = self.action_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings（动作先加噪再 embed，state 全程干净）
        x = self.action_embedding(x)
        x = x.flatten(2).transpose(1, 2)    # (B, T'', wan_dim)
        num_action_tokens = x.size(1)

        # state token 前插（V14）：state@0、动作@1…T''。动作位置相对 V13 整体
        # +1，因此动作相对 video latent 的时间对齐也整体偏一格——这是选定
        # 方案已接受的代价（换 state 在时间轴上严格早于全部动作），不要为了
        # 对齐 V13 而把动作位置减回去。
        has_state_token = state is not None
        if has_state_token:
            assert state.dim() == 2 and state.size(-1) == self.unified_action_dim, (
                f"state dim mismatch: expected [B, {self.unified_action_dim}], "
                f"got {tuple(state.shape)}"
            )
            assert state.size(0) == x.size(0), (
                f"state batch {state.size(0)} != action batch {x.size(0)}"
            )
            # (B, U) -> (B, U, 1, 1, 1) 进 kernel=1 的 Conv3d -> (B, 1, wan_dim)
            s = state.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            s = self.state_embedding(s).flatten(2).transpose(1, 2)
            x = torch.cat([s, x], dim=1)    # (B, 1 + T'', wan_dim)

        # grid/seq_lens 在拼接之后构建，长度 = T'' + int(has_state_token)：
        # 序列位 i 直接对应 RoPE 时间位 i（rope_apply 只按 grid 的时间轴取
        # arange，没有逐 token 位置表，所以位置只能由序列顺序表达）。
        seq_len = x.size(1)
        grid_sizes = torch.stack(
            [torch.tensor([seq_len, 1, 1], dtype=torch.long) for _ in range(x.size(0))])

        seq_lens = torch.full((x.size(0),), seq_len, dtype=torch.long)

        # time embeddings：先按动作 token 数展开，再给 state 位前插 0。
        # 逐 token 时间输入也走同一条路径，不能只覆盖标量时间。
        if t.dim() == 1:
            # [B] 是每样本一个标量时间：先补出通道维再展开，直接 expand 到
            # (B, num_action_tokens) 只在 B == num_action_tokens 时偶然成立。
            t = t.unsqueeze(1).expand(x.size(0), num_action_tokens)
        elif t.dim() == 2 and t.size(1) == 1:
            t = t.expand(t.size(0), num_action_tokens)
        else:
            assert t.dim() == 2 and t.size(1) == num_action_tokens, (
                f"逐 token 扩散时间长度 {tuple(t.shape)} 与动作 token 数 "
                f"{num_action_tokens} 不一致"
            )
        if has_state_token:
            # state 是已观测的干净条件：扩散时间恒 0（uni_branch 侧同样钉一次）
            t = torch.cat([t.new_zeros(t.size(0), 1), t], dim=1)
        assert t.size(1) == seq_len, (
            f"扩散时间长度 {t.size(1)} 与 token 数 {seq_len} 不一致"
        )

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
                "ActionEncoder context must be a projected und token tensor "
                "([B, L_und, dim]); T5 embedding list input is no longer supported."
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
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        x = _run_action_blocks(self.blocks, x, kwargs, layout, self.freqs, timing,
                               timing_sync_cuda, timing_level, "encode")
        return x, grid_sizes, seq_lens, e0
    
    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.action_embedding.weight.flatten(1))
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # state_embedding 零初始化：V14 起点上 state token 不注入任何信息，
        # 便于把动作变化归因到 state 本身（注意这只让 embedding 输出为零，
        # 多出的 token 仍改变 softmax 分母与动作 RoPE 位置，起点不等价于 V13）。
        # 只在构造期执行；checkpoint 回载之后不得再清零。
        nn.init.zeros_(self.state_embedding.weight)
        nn.init.zeros_(self.state_embedding.bias)


class ActionDecoder(nn.Module):
    def __init__(self,
                 action_dim, layer_num,
                 wan_dim, wan_ffn_dim, wan_num_heads,
                 freq_dim=256, text_len=512, text_dim=4096,
                 t_downsample=24, video_action_freq_ratio=6,
                 unified_action_dim=64,
                 robot_token_embedding=None,
                 robot_configs=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.dim = wan_dim
        self.t_downsample = t_downsample
        self.video_action_freq_ratio = video_action_freq_ratio

        # 统一 action 空间：无投影层，unembedding 直接输出统一维 U；
        # robot_configs 保留作签名兼容。
        self.unified_action_dim = unified_action_dim
        self.robot_token_embedding = robot_token_embedding

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, wan_dim),
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim)
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim * 6)
        )

        self.blocks = nn.ModuleList([
            WanAttentionBlock(wan_dim, wan_ffn_dim, wan_num_heads)
            for _ in range(layer_num)
        ])

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (wan_dim % wan_num_heads) == 0 and (wan_dim // wan_num_heads) % 2 == 0
        d = wan_dim // wan_num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # self.action_unembedding = nn.Linear(wan_dim, t_downsample * action_dim)
        self.action_unembedding = nn.Linear(wan_dim, t_downsample * unified_action_dim)
        # state_unembedding 已移除（20260814 时序下采样划分）；V14 的 state
        # token 只作条件，在输出投影前切掉。动作步全部由 action token 重建。

        # initialize weights
        self.init_weights()
    
    def forward(
        self,
        x,
        context,
        grid_sizes,
        seq_lens,
        e0,
        robot_names=None,
        context_lens=None,
        has_state_token=False,
        timing=None, timing_sync_cuda=False, timing_level=1,
        layout=None,
    ):
        """has_state_token: 序列第 0 位是否为 state token（V14）。显式传入，
        不按序列长度猜测。为 True 时 state 继续参与 decoder blocks 的注意力，
        只在输出投影前切掉，不进 unembedding、不进 loss。
        """
        device = self.time_embedding[0].weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # context_lens：uni 输出 und token 的逐样本真实长度（None = 全长，
        # 旧行为）；训练路径传入以 mask 掉 padding 位（评审 #1）。

        # robot token（每机器人 8 个可学习 token）拼到 und context 之前，
        # 与 encoder 侧保持一致（context_lens 相应 +8）。
        rt = None
        if robot_names is not None and self.robot_token_embedding is not None:
            rt = self.robot_token_embedding(robot_names)

        # cross-attn context 循环外一次性构建：[rt, und]（规则同 encoder 侧）。
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
            context_lens=context_lens)
        x = _run_action_blocks(self.blocks, x, kwargs, layout, self.freqs, timing,
                               timing_sync_cuda, timing_level, "decode")

        # (B, L, D)：L = T // t_downsample + int(has_state_token)。state token
        # 已参与上面的注意力，这里切掉不进 unembedding——动作步全部由 action
        # token 重建（每 token d 步，合计 T 步）。
        if has_state_token:
            assert x.size(1) >= 2, f"has_state_token 但序列长度只有 {x.size(1)}"
            x_action = x[:, 1:]
        else:
            x_action = x
        bs = x_action.size(0)

        # action unemb
        x_action = self.action_unembedding(x_action)  # (B, T // t_downsample, t_downsample * unified_action_dim)
        ts = x_action.size(1)
        x = x_action.view(bs, ts * self.t_downsample, -1)  # (B, T, unified_action_dim)

        # 统一 action 空间：无投影层，直接返回 List[(T, U)]；切回原始
        # per-robot 维由 ubt_wm 边界负责（按 robot_layout 的 dim_map gather
        # 或 offset 切片 + denormalize_batch）。
        return [xb for xb in x]   ## List[(T, unified_action_dim)]

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

class ActionBranch(nn.Module):
    def __init__(self,
                 action_dim, wan_dim, wan_ffn_dim, wan_num_heads,
                 encoder_layer_num, decoder_layer_num,
                 t_downsample=24, video_action_freq_ratio=6,
                 unified_action_dim=64, robot_token_embedding=None,
                 robot_configs=None):
        super().__init__()
        self.encoder = ActionEncoder(
            action_dim=action_dim,
            layer_num=encoder_layer_num,
            wan_dim=wan_dim,
            wan_ffn_dim=wan_ffn_dim,
            wan_num_heads=wan_num_heads,
            t_downsample=t_downsample,
            video_action_freq_ratio=video_action_freq_ratio,
            unified_action_dim=unified_action_dim,
            robot_token_embedding=robot_token_embedding,
            robot_configs=robot_configs,
        )
        self.decoder = ActionDecoder(
            action_dim=action_dim,
            layer_num=decoder_layer_num,
            wan_dim=wan_dim,
            wan_ffn_dim=wan_ffn_dim,
            wan_num_heads=wan_num_heads,
            t_downsample=t_downsample,
            video_action_freq_ratio=video_action_freq_ratio,
            unified_action_dim=unified_action_dim,
            robot_token_embedding=robot_token_embedding,
            robot_configs=robot_configs,
        )
    
    def encode(self, action, t, context, robot_names=None, context_lens=None,
               state=None,
               timing=None, timing_sync_cuda=False, timing_level=1, layout=None):
        return self.encoder(action, t, context, robot_names=robot_names,
                            context_lens=context_lens, state=state,
                            timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level,
                            layout=layout)

    def decode(self, action,
               context, grid_sizes, seq_lens, e0, robot_names=None, context_lens=None,
               has_state_token=False,
               timing=None, timing_sync_cuda=False, timing_level=1, layout=None):
        return self.decoder(action, context, grid_sizes, seq_lens, e0, robot_names=robot_names,
                            context_lens=context_lens, has_state_token=has_state_token,
                            timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level,
                            layout=layout)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from thinker_wm.models.robot_tokens import RobotTokenEmbedding

    action_frame = 48
    action_dim = 14
    unified_action_dim = 14  # 冒烟用统一空间小值（生产见 YAML common.unified_action_dim）
    rt_emb = RobotTokenEmbedding(
        {"aloha-agilex": 0}, max_robots=64, num_tokens=8, dim=128,
    ).to(device)
    encoder = ActionEncoder(
        action_dim=action_dim,
        layer_num=1,
        wan_dim=128,
        wan_ffn_dim=256,
        wan_num_heads=8,
        t_downsample=24,
        video_action_freq_ratio=6,
        unified_action_dim=unified_action_dim,
        robot_token_embedding=rt_emb,
    ).to(device)
    decoder = ActionDecoder(
        action_dim=action_dim,
        layer_num=1,
        wan_dim=128,
        wan_ffn_dim=256,
        wan_num_heads=8,
        t_downsample=24,
        video_action_freq_ratio=6,
        unified_action_dim=unified_action_dim,
        robot_token_embedding=rt_emb,
    ).to(device)
    action = torch.randn(1, action_frame, unified_action_dim, device=device)
    t = torch.ones(1, 1, device=device)
    # context 为已投影到 wan_dim 的 und token（[B, L_und, dim]）
    context = torch.randn(1, 8, 128, device=device)

    x, grid_sizes, seq_lens, e0 = encoder(action, t, context)
    x = decoder(x, torch.randn(1, 1, 128, device=device), grid_sizes, seq_lens, e0)
    # x = List[Tensor], Tensor Size = [T, D]；48 步/d=24 → 2 token → 48 步
    assert x[0].shape == (action_frame, unified_action_dim), x[0].shape

    # 1:6 训练配置 round-trip：T=288、ratio=6、t_downsample=6（无裁剪，
    # 288 步 → 48 个动作 token；无 state 路径输出仍为 48×6=288 步）。
    num_video_frames = 48
    ratio = 6
    action_len = num_video_frames * ratio
    encoder_q6 = ActionEncoder(
        action_dim=action_dim,
        layer_num=1,
        wan_dim=128,
        wan_ffn_dim=256,
        wan_num_heads=8,
        t_downsample=ratio,
        video_action_freq_ratio=ratio,
        unified_action_dim=unified_action_dim,
        robot_token_embedding=rt_emb,
    ).to(device)
    decoder_q6 = ActionDecoder(
        action_dim=action_dim,
        layer_num=1,
        wan_dim=128,
        wan_ffn_dim=256,
        wan_num_heads=8,
        t_downsample=ratio,
        video_action_freq_ratio=ratio,
        unified_action_dim=unified_action_dim,
        robot_token_embedding=rt_emb,
    ).to(device)
    action_q6 = torch.randn(2, action_len, unified_action_dim, device=device)
    t_q6 = torch.ones(2, 1, device=device)
    context_q6 = torch.randn(2, 8, 128, device=device)
    ctx_lens_q6 = torch.tensor([8, 8], dtype=torch.long, device=device)
    x, grid_sizes, seq_lens, e0 = encoder_q6(
        action_q6, t_q6, context_q6,
        robot_names=["aloha-agilex", "aloha-agilex"],
        context_lens=ctx_lens_q6,
    )
    # 288 步 / d=6 → 48 token（state=None 的旧路径，无前缀 token）
    assert x.size(1) == action_len // ratio, x.size(1)
    out = decoder_q6(
        x, torch.randn(2, 1, 128, device=device), grid_sizes, seq_lens, e0,
        robot_names=["aloha-agilex", "aloha-agilex"],
        context_lens=ctx_lens_q6,
    )
    assert all(o.shape == (action_len, unified_action_dim) for o in out), (
        f"q6 round-trip shape mismatch: {[tuple(o.shape) for o in out]}"
    )

    # V14 有 state token 的 round-trip：真的传 state，不是在旧断言上加 1。
    # encoder 序列 = 动作 token 数 + 1，decoder 输出仍是原动作长度。
    state_q6 = torch.randn(2, unified_action_dim, device=device)
    xs, grid_s, seq_s, e0_s = encoder_q6(
        action_q6, t_q6, context_q6,
        robot_names=["aloha-agilex", "aloha-agilex"],
        context_lens=ctx_lens_q6,
        state=state_q6,
    )
    assert xs.size(1) == action_len // ratio + 1, xs.size(1)
    assert int(grid_s[0, 0]) == action_len // ratio + 1, grid_s[0].tolist()
    assert int(seq_s[0]) == action_len // ratio + 1, seq_s.tolist()
    out_s = decoder_q6(
        xs, torch.randn(2, 1, 128, device=device), grid_s, seq_s, e0_s,
        robot_names=["aloha-agilex", "aloha-agilex"],
        context_lens=ctx_lens_q6,
        has_state_token=True,
    )
    assert all(o.shape == (action_len, unified_action_dim) for o in out_s), (
        f"q6+state round-trip shape mismatch: {[tuple(o.shape) for o in out_s]}"
    )
    # 零初始化：state token 不注入信息，真 state 与打乱 state 的 encoder 输出
    # 必须逐位相同（robotwin 无 EEF 维，state 也不参与相对化锚点）。
    xs_shuf, _, _, _ = encoder_q6(
        action_q6, t_q6, context_q6,
        robot_names=["aloha-agilex", "aloha-agilex"],
        context_lens=ctx_lens_q6,
        state=state_q6.flip(0),
    )
    assert torch.equal(xs, xs_shuf), "零初始化下 state 内容不应影响 encoder 输出"

    # projector 已整体移除：统一维 U 直接作为模型内部 action 维
    assert not hasattr(encoder, "state_projector")
    assert not hasattr(encoder, "action_projector")
    assert not hasattr(decoder, "action_unprojector")
    # V14：state token 重新引入，encoder 有零初始化的 state_embedding
    assert hasattr(encoder, "state_embedding")
    assert torch.count_nonzero(encoder.state_embedding.weight) == 0
    assert torch.count_nonzero(encoder.state_embedding.bias) == 0
    print("action_branch smoke OK")
