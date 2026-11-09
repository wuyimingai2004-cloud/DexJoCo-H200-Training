"""Structured inference over ``UBT_WM.inference_step`` (no third implementation).

    generate_action(observation, history, instruction, recovery=None) -> PolicyOutput
    predict_outcome(observation, history, instruction, executed_plan)  -> Forecast
    generate_action_for_goal(observation, history, instruction, goal_video_frames) -> GoalActionOutput

``PolicyOutput.forecast`` is the RGB/depth prediction denoised jointly with the
generated actions. It is reusable only for the exact final command it was
generated with (``runtime.prediction_store`` enforces this after
post-processing). ``predict_outcome`` fixes the given actions (role OBS, t=0,
never re-generated or smoothed) and predicts their consequence.

Final latents are always returned; ``decode_outputs=False`` means "not decoded
to pixels", not "no video inference".
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

KIND_JOINT = "joint"
KIND_CONDITIONED = "action_conditioned"


# Compatibility export: hashing is also needed by the simulator-only executor.
from execution.trajectory import action_hash


@dataclass
class Observation:
    rgb: torch.Tensor                 # [3,H,W] in [0,1], composed with the training layout
    state: torch.Tensor               # [Ds] raw measured state (model normalises)
    step: int                         # native control row of this observation
    time_s: float
    robot_name: Optional[str]         # None only for models without robot tokens
    source: str = "measured"          # never "predicted" for policy conditioning
    pil: Any = None


@dataclass(frozen=True)
class GenerationSettings:
    num_inference_steps: int = 10
    action_num_inference_steps: Optional[int] = None
    guidance_scale: float = 1.0
    action_guidance_scale: float = 1.0
    seed: int = 0
    decode_outputs: bool = False
    return_depth: bool = False
    # 逐块生成的块数（仅 ar_packed_teacher_forcing 模型）；None = 1。
    # 长时程规划由调用方显式提升，见方案第 6 节。
    num_blocks: Optional[int] = None
    prefix_cache_mode: str = "off"    # fixed: exact intra-block reuse, no context eviction
    prefix_cache_max_mb: float = 1024.0  # total budget including both CFG branches


@dataclass
class Forecast:
    kind: str
    decision_step: int
    decision_time_s: float
    actions: np.ndarray               # native [A, Da]: generated (joint) or given (conditioned)
    action_hash: str
    action_stats_version: str
    camera_transform: str
    checkpoint: str
    settings: Dict[str, Any]
    frame_steps: List[int]            # native row of predicted video frames 1..F
    latent_steps: List[int]           # native row represented by latent 1..L-1 (its last frame)
    video_latent: Optional[torch.Tensor] = None
    depth_latent: Optional[torch.Tensor] = None
    frames: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    created_monotonic: float = field(default_factory=time.monotonic)
    cache_stats: Optional[Dict[str, Any]] = None
    depth_content_mask: Optional[torch.Tensor] = None  # HW camera coverage, not GT validity

    def meta(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("actions", "video_latent", "depth_latent", "frames", "depth", "depth_content_mask")}
        d["has_video_latent"] = self.video_latent is not None
        d["has_frames"] = self.frames is not None
        d["has_depth_content_mask"] = self.depth_content_mask is not None
        return d


@dataclass
class PolicyOutput:
    actions: np.ndarray               # native [A, Da], before runtime post-processing
    forecast: Forecast
    settings: Dict[str, Any]


@dataclass
class GoalActionOutput:
    """IDM commands for a supplied goal clip. No forecast: goals are not predictions."""
    actions: np.ndarray
    goal_frames: torch.Tensor
    settings: Dict[str, Any]
    cache_stats: Optional[Dict[str, Any]] = None


def make_message_builder(config, source_name: str) -> Callable[[str], str]:
    """Same VLM message construction as training / periodic_eval."""
    from thinker_wm.utils.vlm_utils import (PROMPT_MODE_TEXT_ONLY, build_vlm_rewrite_message,
                                 build_vlm_task_message, resolve_prompt_mode,
                                 resolve_vlm_prompt_style)
    if not bool(getattr(config.model, "unified_vlm_task_prompt", False)):
        return lambda text: str(text)
    if resolve_prompt_mode(config, [source_name]) == PROMPT_MODE_TEXT_ONLY:
        return lambda text: str(text).strip()
    if bool(getattr(config.model, "vlm_text_rewrite", False)):
        return lambda text: build_vlm_rewrite_message(text, source_name)
    style = resolve_vlm_prompt_style(config)
    return lambda text: build_vlm_task_message(text, style=style)


class WAMInference:
    def __init__(self, model, *, checkpoint: str, action_stats_version: str,
                 camera_transform: str, settings: GenerationSettings = GenerationSettings(),
                 message_builder: Callable[[str], str] = str):
        self.model = model
        self.checkpoint = checkpoint
        self.action_stats_version = action_stats_version
        self.camera_transform = camera_transform
        self.settings = settings
        self.message_builder = message_builder
        cfg = model.config
        self.ratio = int(cfg.video_action_freq_ratio)
        self.block_causal = str(getattr(cfg, "temporal_mode", "bidirectional")) == "block_causal"
        self.packed = self.block_causal and bool(getattr(cfg, "ar_packed_teacher_forcing", False))
        if settings.num_blocks is not None and not self.packed:
            raise ValueError("GenerationSettings.num_blocks needs an ar_packed_teacher_forcing model")
        # 预测/动作长度 = 实际生成的块；packed 模型每块 4·latents_per_block 帧。
        self.num_blocks = (settings.num_blocks or 1) if self.packed else None
        self.num_frames = (self.num_blocks * 4 * int(getattr(cfg, "ar_latents_per_block", 1))
                           if self.packed else int(cfg.num_video_frames))

    def identity(self) -> Dict[str, Any]:
        return dict(checkpoint=self.checkpoint, action_stats_version=self.action_stats_version,
                    camera_transform=self.camera_transform, settings=asdict(self.settings),
                    temporal_mode="block_causal" if self.block_causal else "bidirectional")

    def _steps(self, d: int):
        frames = [d + n * self.ratio for n in range(1, self.num_frames + 1)]
        latents = [d + 4 * f * self.ratio for f in range(1, self.num_frames // 4 + 1)]
        return frames, latents

    def _history(self, history, observation: Observation):
        if history is None:
            return None, None
        if hasattr(history, "context_for"):
            return history.context_for(self.model, observation.step, self.checkpoint), None
        return None, history

    def _run(self, observation: Observation, history, instruction: str, *, condition=None,
             recovery=None, goal=None):
        if observation.source != "measured":
            raise ValueError("policy/outcome inference must start from a measured observation")
        if goal is not None and (not self.block_causal or condition is not None):
            raise ValueError("IDM needs block_causal and cannot also fix actions")
        s = self.settings
        rgb = observation.rgb.float().clamp(0, 1).unsqueeze(0)
        pil = observation.pil
        if pil is None:
            import PIL.Image
            arr = (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            pil = PIL.Image.fromarray(arr)
        ctx, raw = self._history(history, observation)
        kwargs = {}
        if self.block_causal:
            import dataclasses

            from thinker_wm.models import temporal_training as TT
            from thinker_wm.models.depth_conditioning import depth_condition_mode
            dffc = depth_condition_mode(self.model.config) == "first_frame"
            req = TT.sample_step_decision(self.model.config, 0, False, dffc).request
            if condition is not None:
                req = dataclasses.replace(req, action_role="world")
            if goal is not None:
                req = dataclasses.replace(req, video_role="goal")
                kwargs["condition_video_frames"] = goal.unsqueeze(0)
            kwargs.update(temporal_request=req, history_context=ctx, history_inputs=raw)
            if self.packed:
                kwargs.update(num_blocks=self.num_blocks)
            if recovery is not None:
                kwargs.update(recovery_features=recovery["features"],
                              recovery_valid=recovery.get("valid"))
        elif history is not None or recovery is not None:
            raise ValueError("history/recovery need a block_causal model")
        cap: Dict[str, Any] = {}
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(s.seed) + int(observation.step))
            with torch.no_grad():
                out = self.model.inference_step(
                    first_frame=rgb.to(self.model.device),
                    state=observation.state.to(device=self.model.device, dtype=torch.float32).unsqueeze(0),
                    num_inference_steps=s.num_inference_steps,
                    action_num_inference_steps=s.action_num_inference_steps,
                    first_frame_pils=[pil], vlm_input_messages=[self.message_builder(instruction)],
                    robot_names=None if observation.robot_name is None else [observation.robot_name],
                    guidance_scale=s.guidance_scale,
                    action_guidance_scale=s.action_guidance_scale, action_smooth_alpha=0.0,
                    predict_action=True, decode_outputs=s.decode_outputs,
                    return_depth=s.return_depth,
                    condition_actions=None if condition is None else torch.as_tensor(
                        condition, dtype=torch.float32).unsqueeze(0),
                    latent_capture=cap, prefix_cache_mode=s.prefix_cache_mode,
                    prefix_cache_max_mb=s.prefix_cache_max_mb, **kwargs)
        frames, actions = out[0], out[1]
        depth = out[2] if s.return_depth else None
        return frames, actions, depth, cap

    def _forecast(self, kind, observation, actions_np, frames, depth, cap) -> Forecast:
        from thinker_wm.models.depth_output import depth_content_mask, mask_decoded_depth
        content_mask = depth_content_mask(self.camera_transform, observation.rgb.shape[-2:])
        depth = mask_decoded_depth(depth, content_mask)
        fsteps, lsteps = self._steps(observation.step)
        cpu = lambda x: None if x is None else x.detach().float().cpu()  # noqa: E731
        return Forecast(kind=kind, decision_step=int(observation.step),
                        decision_time_s=float(observation.time_s), actions=actions_np,
                        action_hash=action_hash(actions_np),
                        action_stats_version=self.action_stats_version,
                        camera_transform=self.camera_transform, checkpoint=self.checkpoint,
                        settings=asdict(self.settings), frame_steps=fsteps, latent_steps=lsteps,
                        video_latent=cpu(cap.get("video_latent")), depth_latent=cpu(cap.get("depth_latent")),
                        frames=cpu(frames), depth=cpu(depth), cache_stats=cap.get("prefix_cache"),
                        depth_content_mask=content_mask)

    def generate_action(self, observation: Observation, history, instruction: str,
                        recovery: Optional[Dict[str, torch.Tensor]] = None) -> PolicyOutput:
        frames, actions, depth, cap = self._run(observation, history, instruction, recovery=recovery)
        if actions is None:
            raise RuntimeError("model returned no actions")
        a = actions[0].detach().float().cpu().numpy()
        fc = self._forecast(KIND_JOINT, observation, a, frames, depth, cap)
        return PolicyOutput(actions=a, forecast=fc, settings=asdict(self.settings))

    def predict_outcome(self, observation: Observation, history, instruction: str,
                        executed_plan: np.ndarray) -> Forecast:
        plan = np.asarray(executed_plan, dtype=np.float32)
        need = self.num_frames * self.ratio
        if plan.ndim != 2 or plan.shape[0] != need:
            raise ValueError(f"executed_plan must be [{need}, Da] native actions, got {plan.shape}")
        if not np.isfinite(plan).all():
            raise ValueError("executed_plan must be finite")
        frames, actions, depth, cap = self._run(observation, history, instruction, condition=plan)
        a = actions[0].detach().float().cpu().numpy() if actions is not None else plan
        if not np.allclose(a, plan, atol=1e-4):
            raise RuntimeError("conditioned actions changed during inference")
        return self._forecast(KIND_CONDITIONED, observation, plan, frames, depth, cap)


    def generate_action_for_goal(self, observation: Observation, history, instruction: str,
                                 goal_video_frames: torch.Tensor) -> GoalActionOutput:
        """IDM: supply the complete future RGB clip, excluding the current frame.

        Shape [F,3,H,W], [0,1], same camera layout and sampling rate as training.
        A single end-goal image is NOT equivalent to this clip-conditioned IDM.
        Use predict_outcome separately if these commands need consequence checking.
        """
        expected = (self.num_frames, *observation.rgb.shape)
        if not torch.is_tensor(goal_video_frames) or tuple(goal_video_frames.shape) != expected:
            raise ValueError(f"goal_video_frames must have shape {expected}")
        if not torch.isfinite(goal_video_frames).all() or not ((goal_video_frames >= 0) & (goal_video_frames <= 1)).all():
            raise ValueError("goal RGB must be finite and in [0,1]")
        _, actions, _, cap = self._run(observation, history, instruction, goal=goal_video_frames)
        if actions is None:
            raise RuntimeError("IDM returned no actions")
        return GoalActionOutput(actions[0].detach().float().cpu().numpy(),
                                goal_video_frames.detach().float().cpu().clone(), asdict(self.settings),
                                cache_stats=cap.get("prefix_cache"))
