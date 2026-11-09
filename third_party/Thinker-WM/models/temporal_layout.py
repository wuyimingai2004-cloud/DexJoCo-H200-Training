"""Physical-time token layout shared by data, training, sampling and runtime.

One function family decides, for every RGB/depth/action/history token, its
fixed time block, its role and its rotary position. Branches never infer time
from token order.

Time units (``video_stride`` layout of the data pipeline):
  - native control step k (``control_hz``), action step k drives the transition
    into video frame ``k // ratio + 1``;
  - video frame n is native step ``n * ratio``; frame 0 is the clean current
    observation;
  - Wan VAE latent 0 encodes frame 0 alone, latent f>=1 encodes frames
    ``4f-3 .. 4f`` (``vae_temporal_stride`` = 4);
  - a block is ``latents_per_block`` consecutive latents. Block 0 is the current
    observation; blocks 1.. are future; history slots use negative blocks.

Roles and visibility (see ``temporal_attention.visibility``):
  HISTORY  real past observation, clean;            reads earlier/equal history
  OBS      observed/commanded condition, clean;     readable by same-block targets
  CLEAN    teacher-forced GT copy of a future block; NOT readable by same-block targets
  TARGET   noisy token being denoised;              reads same block (and earlier
                                                    targets when configured)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

ROLE_HISTORY = 0
ROLE_OBS = 1
ROLE_CLEAN = 2
ROLE_TARGET = 3
ROLE_GOAL = 4  # clean future RGB supplied to inverse dynamics; never an observation
ROLE_NAMES = {ROLE_HISTORY: "history", ROLE_OBS: "obs", ROLE_CLEAN: "clean", ROLE_TARGET: "target", ROLE_GOAL: "goal"}

HISTORY_KIND_RECENT = 0
HISTORY_KIND_SPARSE = 1
HISTORY_KIND_EPISODE_START = 2


@dataclass(frozen=True)
class TimeGrid:
    """Frequencies and block size; validated so blocks never split a token."""

    video_action_freq_ratio: int
    action_t_downsample: int
    latents_per_block: int = 1
    vae_temporal_stride: int = 4
    control_hz: float = 25.0

    def __post_init__(self):
        for name in ("video_action_freq_ratio", "action_t_downsample",
                     "latents_per_block", "vae_temporal_stride"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"TimeGrid.{name} must be >= 1")
        if not self.control_hz > 0:
            raise ValueError("TimeGrid.control_hz must be positive")
        if self.block_action_steps % self.action_t_downsample != 0:
            raise ValueError(
                f"block of {self.block_action_steps} action steps "
                f"(latents_per_block*{self.vae_temporal_stride}*ratio) is not a multiple of "
                f"action_t_downsample={self.action_t_downsample}; an action token would "
                "straddle two blocks")
        if (self.vae_temporal_stride * self.video_action_freq_ratio) % self.action_t_downsample != 0:
            raise ValueError("one VAE latent must cover a whole number of action tokens")

    @property
    def video_frame_seconds(self) -> float:
        return self.video_action_freq_ratio / self.control_hz

    @property
    def latent_seconds(self) -> float:
        return self.vae_temporal_stride * self.video_frame_seconds

    @property
    def block_action_steps(self) -> int:
        return self.latents_per_block * self.vae_temporal_stride * self.video_action_freq_ratio

    @property
    def block_seconds(self) -> float:
        return self.block_action_steps / self.control_hz

    def num_latents(self, num_future_video_frames: int) -> int:
        if num_future_video_frames % self.vae_temporal_stride != 0:
            raise ValueError(
                f"num_video_frames={num_future_video_frames} must be a multiple of the VAE "
                f"temporal stride {self.vae_temporal_stride}")
        return 1 + num_future_video_frames // self.vae_temporal_stride

    def latent_block(self, latent_index: int) -> int:
        if latent_index <= 0:
            return 0
        return (latent_index - 1) // self.latents_per_block + 1

    def num_blocks(self, num_latents: int) -> int:
        return self.latent_block(num_latents - 1) + 1

    def action_step_block(self, step: int) -> int:
        frame = step // self.video_action_freq_ratio + 1
        latent = (frame - 1) // self.vae_temporal_stride + 1
        return self.latent_block(latent)

    def action_token_blocks(self, num_steps: int) -> List[int]:
        d = self.action_t_downsample
        if num_steps % d != 0:
            raise ValueError(f"{num_steps} action steps not divisible by t_downsample={d}")
        blocks = []
        for g in range(num_steps // d):
            first, last = self.action_step_block(g * d), self.action_step_block(g * d + d - 1)
            if first != last:
                raise ValueError(f"action token {g} straddles blocks {first}/{last}")
            blocks.append(first)
        return blocks

    def execution_steps_for_blocks(self, n_blocks: int) -> int:
        """Native steps covered by the first ``n_blocks`` future blocks."""
        return int(n_blocks) * self.block_action_steps

    def identity(self) -> dict:
        return dict(ratio=self.video_action_freq_ratio, t_downsample=self.action_t_downsample,
                    latents_per_block=self.latents_per_block,
                    vae_temporal_stride=self.vae_temporal_stride, control_hz=self.control_hz)


@dataclass(frozen=True)
class TemporalRequest:
    """How one forward assigns roles. ``cut`` = first block that is a target.

    cut=1: current observation + every future block noisy (deployment window).
    cut=k>1: future blocks 1..k-1 are teacher-forced clean GT (AR-1 training).
    video_role="goal": future RGB is a clean visual goal for IDM (ROLE_GOAL),
    not an observation. It cannot affect D0 or read its own GT action copy.
    action_role="policy": actions in target blocks are generated;
    action_role="world": every action is a given command (role OBS) and only
    RGB/depth are predicted (action-conditioned consequence prediction).

    packed=True (clean/noisy copies, see ``packed_latent_entries``): RGB/action
    of blocks 1..clean_blocks enter as CLEAN copies, blocks
    first_target..last_target as noisy TARGET copies; a target reads only
    clean blocks before its own. Training: clean 1..N-1, targets 1..N.
    Inference of block k: clean 1..k-1, target k. Depth has no clean prefix.
    """

    grid: TimeGrid
    cut: int = 1
    action_role: str = "policy"
    video_role: str = "predict"  # predict (Joint/FDM) | goal (IDM)
    depth_first_frame_conditioned: bool = True
    target_reads_earlier_targets: bool = True
    und_reads: str = "observed"
    packed: bool = False
    clean_blocks: int = 0
    first_target: int = 1
    last_target: int = 1

    def __post_init__(self):
        if int(self.cut) < 1:
            raise ValueError("TemporalRequest.cut must be >= 1")
        if self.action_role not in ("policy", "world"):
            raise ValueError("action_role must be 'policy' or 'world'")
        if self.video_role not in ("predict", "goal"):
            raise ValueError("video_role must be 'predict' or 'goal'")
        if self.video_role == "goal" and self.action_role != "policy":
            raise ValueError("goal RGB requires generated actions (IDM)")
        if self.und_reads not in ("observed", "none"):
            raise ValueError("und_reads must be 'observed' or 'none'")
        if self.packed:
            if int(self.cut) != 1:
                raise ValueError("packed requests express the prefix by clean_blocks, not cut")
            if self.target_reads_earlier_targets:
                raise ValueError("packed targets must not read other blocks' noisy targets")
            if not 1 <= int(self.first_target) <= int(self.last_target):
                raise ValueError("packed request needs 1 <= first_target <= last_target")
            if not int(self.first_target) - 1 <= int(self.clean_blocks) <= int(self.last_target) - 1:
                raise ValueError("clean_blocks must cover every block before first_target "
                                 "and no target block's own clean copy beyond last_target-1")


def block_role(block: int, cut: int) -> int:
    if block <= 0:
        return ROLE_OBS
    return ROLE_CLEAN if block < cut else ROLE_TARGET


def latent_roles(request: TemporalRequest, num_latents: int, *, depth: bool = False) -> List[int]:
    """Role of every RGB/depth latent frame of the current window."""
    roles = [block_role(request.grid.latent_block(f), request.cut) for f in range(num_latents)]
    if not depth and request.video_role == "goal":
        roles = [ROLE_OBS] + [ROLE_GOAL] * (num_latents - 1)
    if depth and not request.depth_first_frame_conditioned:
        roles[0] = ROLE_TARGET
    return roles


def action_step_roles(request: TemporalRequest, num_steps: int) -> List[int]:
    if request.action_role == "world":
        return [ROLE_OBS] * num_steps
    return [block_role(request.grid.action_step_block(k), request.cut) for k in range(num_steps)]


def num_target_blocks_check(request: TemporalRequest, num_latents: int) -> None:
    if request.packed:
        if request.last_target >= request.grid.num_blocks(num_latents):
            raise ValueError(f"packed last_target={request.last_target} exceeds the "
                             f"{request.grid.num_blocks(num_latents) - 1} future blocks")
        return
    if request.cut >= request.grid.num_blocks(num_latents):
        raise ValueError(
            f"cut={request.cut} leaves no target block in a window of "
            f"{request.grid.num_blocks(num_latents)} blocks")


def block_latents(grid: TimeGrid, block: int) -> range:
    if block < 1:
        raise ValueError("future blocks start at 1")
    return range((block - 1) * grid.latents_per_block + 1, block * grid.latents_per_block + 1)


def block_steps(grid: TimeGrid, block: int) -> range:
    if block < 1:
        raise ValueError("future blocks start at 1")
    return range((block - 1) * grid.block_action_steps, block * grid.block_action_steps)


def packed_latent_entries(request: TemporalRequest, *, depth: bool = False) -> List[tuple]:
    """(source latent, block, role) per packed RGB/depth latent, in packed order.

    RGB: current frame (OBS), clean copies of blocks 1..clean_blocks, then noisy
    copies of the target blocks. Depth carries no clean prefix; in rgb_only the
    current depth latent is a target at observed time 0. It is denoised with
    block 1 but must not read that block's future observations or commands.
    """
    if not request.packed:
        raise ValueError("packed_latent_entries needs a packed request")
    g = request.grid
    if not depth and request.video_role == "goal":
        return [(0, 0, ROLE_OBS)] + [
            (f, b, ROLE_GOAL) for b in range(1, request.last_target + 1)
            for f in block_latents(g, b)]
    out = []
    if not depth or request.depth_first_frame_conditioned:
        out.append((0, 0, ROLE_OBS))
    if not depth:
        for b in range(1, request.clean_blocks + 1):
            out.extend((f, b, ROLE_CLEAN) for f in block_latents(g, b))
    for b in range(request.first_target, request.last_target + 1):
        if depth and not request.depth_first_frame_conditioned and b == 1:
            out.append((0, 0, ROLE_TARGET))
        out.extend((f, b, ROLE_TARGET) for f in block_latents(g, b))
    return out


def packed_action_entries(request: TemporalRequest) -> List[tuple]:
    """(source step, block, role) per packed action step. World mode: every
    step of blocks 1..last_target is one given command (OBS), no copies."""
    if not request.packed:
        raise ValueError("packed_action_entries needs a packed request")
    g = request.grid
    if request.action_role == "world":
        return [(k, b, ROLE_OBS) for b in range(1, request.last_target + 1) for k in block_steps(g, b)]
    out = [(k, b, ROLE_CLEAN) for b in range(1, request.clean_blocks + 1) for k in block_steps(g, b)]
    for b in range(request.first_target, request.last_target + 1):
        out.extend((k, b, ROLE_TARGET) for k in block_steps(g, b))
    return out


@dataclass
class Segment:
    """Token layout of one modality stream. pos is (f, h, w) per token."""

    block: torch.Tensor                 # [L] long
    role: torch.Tensor                  # [L] long
    pos: torch.Tensor                   # [L,3] or [B,L,3] long; f may be negative
    valid: Optional[torch.Tensor] = None  # [B,L] bool; None = all valid
    n_prefix: int = 0                   # leading history tokens (video stream only)

    def __len__(self):
        return int(self.block.numel())

    def to(self, device):
        return Segment(self.block.to(device), self.role.to(device), self.pos.to(device),
                       None if self.valid is None else self.valid.to(device), self.n_prefix)

    @property
    def clean(self) -> torch.Tensor:
        return self.role != ROLE_TARGET

    @staticmethod
    def cat(parts: Sequence["Segment"], batch: int) -> "Segment":
        parts = [p for p in parts if p is not None and len(p) > 0]
        batched = any(p.pos.dim() == 3 for p in parts)
        pos = [p.pos.unsqueeze(0).expand(batch, -1, -1) if batched and p.pos.dim() == 2 else p.pos
               for p in parts]
        valid = None
        if any(p.valid is not None for p in parts):
            valid = torch.cat([
                p.valid if p.valid is not None
                else torch.ones(batch, len(p), dtype=torch.bool, device=p.block.device)
                for p in parts], dim=1)
        return Segment(torch.cat([p.block for p in parts]), torch.cat([p.role for p in parts]),
                       torch.cat(pos, dim=-2), valid, n_prefix=parts[0].n_prefix if parts else 0)


def video_segment(roles: Sequence[int], grid: TimeGrid, h: int, w: int,
                  device=None) -> Segment:
    """Current-window latent tokens, frame-major (matches patchify flatten)."""
    f_n = len(roles)
    hw = h * w
    blocks = torch.tensor([grid.latent_block(f) for f in range(f_n)], dtype=torch.long)
    block = blocks.repeat_interleave(hw)
    role = torch.tensor(list(roles), dtype=torch.long).repeat_interleave(hw)
    ff, hh, ww = torch.meshgrid(torch.arange(f_n), torch.arange(h), torch.arange(w), indexing="ij")
    pos = torch.stack([ff.reshape(-1), hh.reshape(-1), ww.reshape(-1)], dim=-1)
    seg = Segment(block, role, pos)
    return seg.to(device) if device is not None else seg


def action_segment(roles_per_step: Sequence[int], grid: TimeGrid, has_state: bool,
                   device=None) -> Segment:
    """Action tokens (one per t_downsample steps), optional OBS state token first."""
    n_steps = len(roles_per_step)
    d = grid.action_t_downsample
    token_blocks = grid.action_token_blocks(n_steps)
    token_roles = []
    for g in range(n_steps // d):
        rs = set(roles_per_step[g * d:(g + 1) * d])
        if len(rs) != 1:
            raise ValueError(f"action token {g} mixes roles {sorted(rs)}")
        token_roles.append(rs.pop())
    if has_state:
        token_blocks = [0] + token_blocks
        token_roles = [ROLE_OBS] + token_roles
    n = len(token_blocks)
    pos = torch.zeros(n, 3, dtype=torch.long)
    pos[:, 0] = torch.arange(n)
    seg = Segment(torch.tensor(token_blocks, dtype=torch.long),
                  torch.tensor(token_roles, dtype=torch.long), pos)
    return seg.to(device) if device is not None else seg


def entries_video_segment(entries: Sequence[tuple], h: int, w: int, device=None) -> Segment:
    """Packed latent tokens, frame-major; RoPE frame = source latent index, so a
    clean copy and its noisy target share positions."""
    hw = h * w
    src = torch.tensor([e[0] for e in entries], dtype=torch.long)
    block = torch.tensor([e[1] for e in entries], dtype=torch.long).repeat_interleave(hw)
    role = torch.tensor([e[2] for e in entries], dtype=torch.long).repeat_interleave(hw)
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pos = torch.stack([src.repeat_interleave(hw), hh.reshape(-1).repeat(len(entries)),
                       ww.reshape(-1).repeat(len(entries))], dim=-1)
    seg = Segment(block, role, pos)
    return seg.to(device) if device is not None else seg


def entries_action_segment(entries: Sequence[tuple], grid: TimeGrid, has_state: bool,
                           device=None) -> Segment:
    """Packed action tokens (one per t_downsample consecutive source steps),
    optional OBS state token first; RoPE = source token index (+1 after state)."""
    d = grid.action_t_downsample
    if len(entries) % d:
        raise ValueError(f"{len(entries)} packed action steps not divisible by t_downsample={d}")
    blocks, roles, pos = [], [], []
    for g in range(len(entries) // d):
        grp = entries[g * d:(g + 1) * d]
        first = grp[0][0]
        if first % d or [e[0] for e in grp] != list(range(first, first + d)) \
                or len({(e[1], e[2]) for e in grp}) != 1:
            raise ValueError(f"packed action token {g} is not one aligned single-role group")
        blocks.append(grp[0][1])
        roles.append(grp[0][2])
        pos.append(first // d + int(has_state))
    if has_state:
        blocks, roles, pos = [0] + blocks, [ROLE_OBS] + roles, [0] + pos
    p = torch.zeros(len(pos), 3, dtype=torch.long)
    p[:, 0] = torch.tensor(pos, dtype=torch.long)
    seg = Segment(torch.tensor(blocks, dtype=torch.long), torch.tensor(roles, dtype=torch.long), p)
    return seg.to(device) if device is not None else seg


def history_frame_positions(times_s: torch.Tensor, grid: TimeGrid) -> torch.Tensor:
    """Continuous physical time in latent units; never renumber existing frames
    when an event is inserted. Equal timestamps intentionally share a position.
    """
    if times_s.dim() != 2:
        raise ValueError("history times must be [B, Th]")
    return times_s.double() / grid.latent_seconds


def history_segment(times_s: torch.Tensor, valid: torch.Tensor, grid: TimeGrid,
                    hp: int, wp: int, pool: int = 1, source_hw=None) -> Segment:
    """History slots as clean tokens with negative blocks and real-time RoPE.

    Each slot is one latent frame pooled to ``hp x wp`` tokens. Spatial positions
    use the pooled cell centre on the full-resolution grid so they stay aligned
    with the current window's (h, w) positions.
    """
    b, th = times_s.shape
    if valid.shape != (b, th) or valid.dtype != torch.bool:
        raise ValueError("history_valid must be bool [B, Th]")
    if th > 1 and bool((times_s[:, 1:] < times_s[:, :-1])[valid[:, 1:] & valid[:, :-1]].any()):
        raise ValueError("history must be ordered oldest first")
    hw = hp * wp
    device = times_s.device
    block = torch.arange(-th, 0, dtype=torch.long, device=device).repeat_interleave(hw)
    role = torch.full((th * hw,), ROLE_HISTORY, dtype=torch.long, device=device)
    fpos = history_frame_positions(times_s, grid).to(device)            # [B, Th]
    hh, ww = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device),
                            indexing="ij")
    sh, sw = source_hw or (hp * pool, wp * pool)
    def centres(x, extent):
        first = x.reshape(-1) * pool
        last = (first + pool - 1).clamp(max=extent - 1)
        return (first + last).double() / 2
    hpos, wpos = centres(hh, sh), centres(ww, sw)
    pos = torch.stack([
        fpos.repeat_interleave(hw, dim=1),
        hpos.repeat(th).unsqueeze(0).expand(b, -1),
        wpos.repeat(th).unsqueeze(0).expand(b, -1),
    ], dim=-1)
    tok_valid = valid.repeat_interleave(hw, dim=1)
    return Segment(block, role, pos, tok_valid, n_prefix=th * hw)


@dataclass
class TemporalLayout:
    """Per-forward layouts for every present stream (current window + history)."""

    request: TemporalRequest
    video: Optional[Segment] = None      # history prefix + current window
    action: Optional[Segment] = None
    depth: Optional[Segment] = None
    history_logit_bias: Optional[torch.Tensor] = None
    cache: dict = field(default_factory=dict)

    def segment(self, name: str) -> Optional[Segment]:
        return getattr(self, name)


def token_timesteps(t: torch.Tensor, seg: Segment) -> torch.Tensor:
    """[B] / [B,1] / [B,L] noise level -> [B, L] per-token labels, 0 on clean tokens."""
    if t.dim() == 1:
        t = t.unsqueeze(1)
    if t.dim() != 2 or t.size(1) not in (1, len(seg)):
        raise ValueError(f"timesteps {tuple(t.shape)} incompatible with {len(seg)} tokens")
    out = t.float().expand(t.size(0), len(seg)).clone()
    out[:, seg.clean.to(out.device)] = 0.0
    return out
