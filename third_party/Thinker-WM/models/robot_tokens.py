"""每机器人 8 个可学习 token（robot token embedding）。

多机器人统一 action 空间的配套机制：每个机器人有 8 个可学习 token，在
video/action 两分支的 encoder/decoder cross-attn 之前拼入 context（context
为 und token，shape [B, L_und, dim]；拼入后每样本序列长 +8，context_lens
相应 +8，与 VLM padding mask 语义一致）。

参数表 shape [max_robots, num_tokens, dim]，预留 64 槽 append-only：机器人→槽位
由 YAML ``dataset.robot_names[*].robot_token_index`` 显式指定（禁止复用/重排，
保证 ckpt 行稳定）；未注册 robot_token_index 的机器人由
train/train.py 的 get_robot_layout 按注册顺序自动分配（缺省 0..N-1）。

超容量增长协议（详见 .AGENTS_CONTENT/03_config.md）：
  - robot 槽位超 max_robots（64）：增大 max_robots，参数表按行扩展（旧行原样
    拷贝、新行零初始化），load_state_dict 加对应 remap（新行随机初始化起步
    亦可，属新参数）；
  - 统一 action 空间超容量（common.unified_action_dim 用满）：增大维度，共享
    projector 输入列权重扩展（旧列原样拷贝、新列零初始化），load_state_dict
    加对应 remap；YAML dataset.unified_bands 注册表只增不改不挪。
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class RobotTokenEmbedding(nn.Module):
    """每机器人 8 个可学习 token，拼入 video/action 分支 cross-attn context。

    参数表 [max_robots, 8, dim] 预留 64 槽 append-only；机器人→槽位由
    YAML robot_token_index 显式指定（禁止复用/重排，保证 ckpt 行稳定）。
    """

    def __init__(
        self,
        robot_token_indices: Dict[str, int],
        max_robots: int = 64,
        num_tokens: int = 8,
        dim: int = 0,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"RobotTokenEmbedding dim must be > 0, got {dim}")
        if max_robots <= 0:
            raise ValueError(f"RobotTokenEmbedding max_robots must be > 0, got {max_robots}")
        if num_tokens <= 0:
            raise ValueError(f"RobotTokenEmbedding num_tokens must be > 0, got {num_tokens}")
        self.max_robots = max_robots
        self.num_tokens = num_tokens
        self.dim = dim
        # append-only 槽位表：机器人 → 行号（YAML 显式指定，禁止复用/重排）
        self._indices: Dict[str, int] = {}
        for name, index in (robot_token_indices or {}).items():
            key = str(name)
            index = int(index)
            if not (0 <= index < max_robots):
                raise ValueError(
                    f"robot {key}: robot_token_index {index} 超出容量 [0, {max_robots})"
                )
            if index in self._indices.values():
                raise ValueError(
                    f"robot {key}: robot_token_index {index} 已被其他机器人占用（禁止复用）"
                )
            self._indices[key] = index
        # 0.02 std 初始化，与 Wan DiT 的 embedding 初始化量级一致
        self.table = nn.Parameter(
            torch.empty(max_robots, num_tokens, dim).normal_(std=0.02)
        )

    def forward(
        self, robot_names: Optional[Union[str, List[str]]]
    ) -> Optional[torch.Tensor]:
        """返回 [B, num_tokens, dim]；robot_names 为 None/空时返回 None（不拼 token）。"""
        if robot_names is None or len(robot_names) == 0:
            return None
        if isinstance(robot_names, str):
            robot_names = [robot_names]
        slots: List[int] = []
        for name in robot_names:
            key = str(name)
            if key not in self._indices:
                raise KeyError(
                    f"Unknown robot name {key!r}: 未在 YAML robot_names 注册 "
                    f"robot_token_index（已注册: {sorted(self._indices)}）"
                )
            slots.append(self._indices[key])
        idx = torch.tensor(slots, dtype=torch.long, device=self.table.device)
        return self.table[idx]  # [B, num_tokens, dim]

    @torch.no_grad()
    def copy_rows(self, pairs) -> None:
        """finetune 热启动：新槽位 dst 从已训练槽位 src 拷贝起步（[[src, dst], ...]）。"""
        for src, dst in pairs:
            src, dst = int(src), int(dst)
            if src == dst or not (0 <= src < self.max_robots and 0 <= dst < self.max_robots):
                raise ValueError(f"robot token copy {src}->{dst} 非法（容量 [0, {self.max_robots})）")
            self.table[dst].copy_(self.table[src])
