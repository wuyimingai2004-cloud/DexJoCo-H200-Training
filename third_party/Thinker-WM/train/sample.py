#!/usr/bin/env python3
"""
Evaluation utilities for Motus.
Implements inference sampling and metrics computation for validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Suppress matplotlib font manager debug messages
matplotlib.set_loglevel("WARNING")
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
import random
import os

from thinker_wm.utils.vlm_utils import (
    build_vlm_rewrite_message,
    build_vlm_task_message,
    resolve_vlm_prompt_style,
)
from thinker_wm.utils.vlm_utils import resolve_prompt_mode, PROMPT_MODE_TEXT_ONLY

logger = logging.getLogger(__name__)


def create_video_tensor_with_action_plot(*args, **kwargs):
    """Optional visualization helper.

    The training loop imports this module even when validation is disabled. Keep
    the main training path independent from the old top-level gen.py script.
    """
    try:
        from gen import create_video_tensor_with_action_plot as _create_video_tensor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Video visualization helper gen.py is not available. "
            "Training is unaffected unless validation visualization is enabled."
        ) from exc
    return _create_video_tensor(*args, **kwargs)


def create_video_grid(predicted_frames: torch.Tensor, ground_truth_frames: torch.Tensor, 
                     num_samples: int = 4) -> Image.Image:
    """
    Create a grid visualization comparing predicted and ground truth video frames.
    
    Args:
        predicted_frames: (B, T, C, H, W) predicted video frames
        ground_truth_frames: (B, T, C, H, W) ground truth video frames  
        num_samples: number of samples to visualize
        
    Returns:
        PIL Image of the comparison grid
    """
    batch_size = min(predicted_frames.shape[0], num_samples)
    num_frames = predicted_frames.shape[1]
    
    # Convert to numpy (B, T, H, W, C)
    pred_np = predicted_frames[:batch_size].detach().cpu().permute(0, 1, 3, 4, 2).numpy()
    gt_np = ground_truth_frames[:batch_size].detach().cpu().permute(0, 1, 3, 4, 2).numpy()

    # Clip values to [0, 1] (safety)
    pred_np = np.clip(pred_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    # Create grid: rows are samples, columns are [GT_frame1, GT_frame2, ..., GT_frameN, Pred_frame1, Pred_frame2, ..., Pred_frameN]
    fig, axes = plt.subplots(batch_size, num_frames * 2, figsize=(4 * num_frames * 2, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    elif num_frames * 2 == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(batch_size):
        for t in range(num_frames):
            # Ground truth frame
            axes[i, t].imshow(gt_np[i, t])
            axes[i, t].set_title(f'GT Frame {t+1}')
            axes[i, t].axis('off')
            
            # Predicted frame  
            axes[i, t + num_frames].imshow(pred_np[i, t])
            axes[i, t + num_frames].set_title(f'Pred Frame {t+1}')
            axes[i, t + num_frames].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    img_array = img_array[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return Image.fromarray(img_array)


@torch.no_grad()
def _history_inputs(batch, device):
    """Pass history through when the batch already carries it."""
    if batch.get("history_recent_rgb") is None and batch.get("history_sparse_rgb") is None:
        return None
    from thinker_wm.data_prepare.history_attach import history_inputs_from_batch
    return history_inputs_from_batch(batch, device)


def _validation_window(config):
    """Packed training window used by validation, or None for other callers.

    A RoboDojo batch is 3 blocks: 12 video frames and 24 action steps. Passing
    num_blocks explicitly keeps inference from defaulting to a single block.
    """
    common = getattr(config, "common", None)
    model_cfg = getattr(config, "model", None)
    wm = getattr(model_cfg, "wm", None) if model_cfg is not None else None
    if common is None or wm is None:
        return None
    if not bool(getattr(wm, "ar_packed_teacher_forcing", False)):
        return None
    frames = int(common.num_video_frames)
    actions = int(common.action_chunk_size)
    ratio = int(common.video_action_freq_ratio)
    latents = int(getattr(wm, "ar_latents_per_block", 1) or 1)
    frames_per_block = latents * 4
    if frames_per_block <= 0 or frames % frames_per_block != 0:
        raise ValueError(
            f"num_video_frames={frames} is not a whole number of "
            f"{frames_per_block}-frame blocks"
        )
    blocks = frames // frames_per_block
    steps_per_block = latents * 4 * ratio
    if blocks < 1 or blocks * steps_per_block != actions:
        raise ValueError(
            f"action_chunk_size={actions} does not match {blocks} blocks "
            f"of {steps_per_block} steps"
        )
    return {
        "blocks": blocks,
        "frames": frames,
        "actions": actions,
        "first_block_actions": steps_per_block,
    }


def inference_sample(model, batch: Dict, config, *, a2v=False, details=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference to predict future video frames and actions using UniDiffuser's native inference method.
    
    Args:
        model: UniDiffuser model (LatentActionWorldModel)
        batch: Input batch containing observations, states, language embeddings, text instructions
        config: Configuration object containing inference parameters
        
    Returns:
        Tuple of (predicted_frames, predicted_actions)
        - predicted_frames: (B, num_pred_frames, C, H, W) in pixel space [0, 255]
        - predicted_actions: (B, action_chunk_size, action_dim)
    """
    was_training = bool(model.training)
    model.eval()
    try:
        dtype = torch.bfloat16
    
        # Extract inference parameters from config
        num_inference_steps = config.model.inference.num_inference_timesteps
        # classifier-free guidance 强度（评审 #6，仅 video）；缺省 1.0 = 关闭（旧行为）
        guidance_scale = float(getattr(config.model.inference, "guidance_scale", 1.0))
        # action 侧推理参数（修复 #1/#3b/#9c）：action CFG（默认 1.0=不做）、
        # 独立去噪步数（默认 None=与 video 同）、时序平滑（默认 0=关闭）
        inference_cfg = config.model.inference
        action_guidance_scale = float(getattr(inference_cfg, "action_guidance_scale", 1.0))
        action_num_inference_steps = getattr(inference_cfg, "action_num_inference_timesteps", None)
        if action_num_inference_steps is not None:
            action_num_inference_steps = int(action_num_inference_steps)
        action_smooth_alpha = float(getattr(inference_cfg, "action_temporal_smooth_alpha", 0.0))
        depth_smooth_alpha = float(getattr(inference_cfg, "depth_temporal_smooth_alpha", 0.0))
    
        # Move batch data to device
        device = next(model.parameters()).device
        first_frame = batch['first_frame'].to(device=device, dtype=dtype)  # [B, C, H, W] - conditioning frame
    
        state = batch['initial_state'].to(device=device, dtype=torch.float32) if 'initial_state' in batch and batch['initial_state'] is not None else None
    
        language_embeddings = batch['language_embedding']
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(device=device, dtype=dtype)
    
        vlm_inputs = batch['vlm_inputs']
        if vlm_inputs is not None:
            # Move all tensors in the VLM inputs dict to device；只对浮点 tensor
            # cast dtype（评审 16b：input_ids 等整型 tensor 被 cast 成 bf16 会
            # 直接损坏）。
            vlm_inputs = {k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
                          if isinstance(v, torch.Tensor) else v
                          for k, v in vlm_inputs.items()}
    
        # data for pre_embedding
        first_frame_pils = batch.get("first_frame_pils", None)
        t5_input_messages = batch.get("t5_input_messages", None)
        vlm_input_messages = batch.get("vlm_input_messages", None)
        if (
            bool(getattr(config.model, "unified_vlm_task_prompt", False))
            and t5_input_messages
        ):
            # 与训练侧一致：用统一 task prompt 重建 vlm 消息（见 utils.vlm_utils）；
            # prompt_mode=text_only 的源用原始文本纯文本编码（不带首帧图）。
            if resolve_prompt_mode(config, batch.get("source_datasets")) == PROMPT_MODE_TEXT_ONLY:
                vlm_input_messages = [str(text).strip() for text in t5_input_messages]
                first_frame_pils = None
            elif bool(getattr(config.model, "vlm_text_rewrite", False)):
                # vlm_text_rewrite 开启：与训练侧一致改用压缩改写指令消息（纯文本、
                # 不带首帧图；按数据集家族分模板），由 UndBranch._rewrite_prompts
                # 生成短文本后再编码。
                source_key = (batch.get("source_datasets") or [None])[0]
                vlm_input_messages = [
                    build_vlm_rewrite_message(text, source_key)
                    for text in t5_input_messages
                ]
            else:
                vlm_input_messages = [
                    build_vlm_task_message(text, style=resolve_vlm_prompt_style(config))
                    for text in t5_input_messages
                ]
        robot_names = batch.get("robot_names", None)

        # Match deployment: RGB-only depth never receives GT depth as a condition.
        depth_first = None
        inference_model = getattr(model, "module", model)
        model_config = inference_model.config
        if bool(getattr(model_config, "enable_depth", False)) and str(
            getattr(model_config, "depth_condition_mode", "first_frame")
        ) == "first_frame":
            if "depth_first" not in batch:
                from thinker_wm.data_prepare.depth_attach import build_depth_attacher
                attacher = build_depth_attacher(config)
                if attacher is not None:
                    batch = attacher.attach(batch)
            depth_first = batch.get("depth_first")
            present = batch.get("depth_present")
            if depth_first is None or present is None or not bool(present.all()):
                raise ValueError("first_frame depth evaluation requires valid depth conditions for the entire batch")
            depth_first = depth_first.to(device=first_frame.device, dtype=first_frame.dtype)
    
        window = _validation_window(config)
        num_blocks = None if window is None else int(window["blocks"])
        condition_kwargs = {}
        if a2v:
            if window is None or str(getattr(model_config, "temporal_mode", "")) != "block_causal":
                raise ValueError("A2V validation requires packed block-causal inference")
            commands = batch.get("action_sequence")
            expected = (first_frame.shape[0], window["actions"],
                        state.shape[-1] if state is not None else -1)
            if state is None or commands is None or tuple(commands.shape) != expected:
                raise ValueError(f"A2V requires raw action_sequence with shape {expected}")
            if not torch.isfinite(commands).all():
                raise ValueError("A2V condition actions contain non-finite values")
            # Native joint commands: inference_step selects action_role=world and
            # applies the robot's normalization. Never pre-normalize here.
            condition_kwargs = {"condition_actions": commands.to(device=device, dtype=torch.float32),
                                "predict_action": True}
        want_depth = (details is not None and not a2v and
                      bool(getattr(getattr(config, "system", None), "val_depth", False)))
        if want_depth and not bool(getattr(model_config, "enable_depth", False)):
            raise ValueError("val_depth requires an enabled depth branch")
        if want_depth:
            condition_kwargs["return_depth"] = True
        with torch.no_grad():
            output = inference_model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
                first_frame_pils=first_frame_pils,
                t5_input_messages=t5_input_messages,
                vlm_input_messages=vlm_input_messages,
                robot_names=robot_names,
                guidance_scale=guidance_scale,
                action_guidance_scale=action_guidance_scale,
                action_num_inference_steps=action_num_inference_steps,
                action_smooth_alpha=action_smooth_alpha,
                depth_smooth_alpha=depth_smooth_alpha,
                depth_first=depth_first,
                history_inputs=_history_inputs(batch, device),
                num_blocks=num_blocks,
                **condition_kwargs,
            )
    
        if want_depth:
            predicted_frames, predicted_actions, predicted_depth = output
        else:
            predicted_frames, predicted_actions = output
            predicted_depth = None
        if details is not None:
            details.update(predicted_depth=predicted_depth,
                           vlm_input_messages=vlm_input_messages,
                           original_vlm_input_messages=batch.get("vlm_input_messages"))
        return predicted_frames, predicted_actions
    finally:
        model.train(was_training)



_VAL_VIZ_SAMPLES = 4


def _capture_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _validation_seed(config) -> int:
    system = getattr(config, "system", None)
    value = getattr(system, "val_seed", None) if system is not None else None
    return 20261021 if value is None else int(value)


def _configured_batch_size(config):
    training = getattr(config, "training", None)
    if training is None:
        return None
    value = getattr(training, "batch_size", None)
    return None if value is None else int(value)


def _seed_validation_noise(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def share_validation_error(error: Optional[str]) -> Optional[str]:
    """Broadcast rank 0's validation error before the training barrier.

    Every rank calls this. Rank 0 passes the error text, or None on success.
    Other ranks pass None and receive rank 0's text. Raising on rank 0 before
    this broadcast leaves the other ranks blocked in the barrier.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return error
    payload = [None if error is None else str(error)]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _reject_prediction(predicted_frames, predicted_actions, batch, window):
    """Return a reason this batch cannot count as a successful validation sample."""
    if predicted_frames is None:
        return "no predicted frames"
    if predicted_actions is None:
        return "no predicted actions"
    if predicted_frames.ndim != 5 or predicted_actions.ndim != 3:
        return (
            f"unexpected prediction ranks frames={tuple(predicted_frames.shape)} "
            f"actions={tuple(predicted_actions.shape)}"
        )
    pred_frames = int(predicted_frames.shape[2])
    label_frames = int(batch["video_frames"].shape[1])
    if pred_frames != label_frames:
        return f"predicted video length {pred_frames} != label {label_frames}"
    if "action_sequence" not in batch or batch["action_sequence"] is None:
        return "batch has no action labels"
    pred_actions = int(predicted_actions.shape[1])
    label_actions = int(batch["action_sequence"].shape[1])
    if pred_actions != label_actions:
        return f"predicted action length {pred_actions} != label {label_actions}"
    if predicted_frames.shape[0] != predicted_actions.shape[0]:
        return "predicted video and action batch sizes differ"
    if window is not None:
        if pred_frames != int(window["frames"]) or label_frames != int(window["frames"]):
            return (
                f"video length {pred_frames} != training window {int(window['frames'])}"
            )
        if pred_actions != int(window["actions"]) or label_actions != int(window["actions"]):
            return (
                f"action length {pred_actions} != training window {int(window['actions'])}"
            )
    if not torch.isfinite(predicted_frames).all() or not torch.isfinite(predicted_actions).all():
        return "prediction has non-finite values"
    return None



def compute_a2v_video_metrics(predicted_frames, ground_truth_frames, blocks):
    """RGB in [0,1], BTCHW; score the full canvas, matching joint validation."""
    if predicted_frames.shape != ground_truth_frames.shape or predicted_frames.ndim != 5:
        raise ValueError("A2V video prediction and target shapes must match exactly")
    if blocks < 1 or predicted_frames.shape[1] % blocks:
        raise ValueError("A2V video length must contain complete blocks")
    if not torch.isfinite(predicted_frames).all() or not torch.isfinite(ground_truth_frames).all():
        raise ValueError("A2V video contains non-finite values")
    square = (predicted_frames.float() - ground_truth_frames.float()).square()
    result = {"a2v_video_mse": square.mean().item()}
    for i, part in enumerate(square.chunk(blocks, dim=1), 1):
        result[f"a2v_video_mse_block_{i}"] = part.mean().item()
    return result



def compute_depth_metrics(predicted, target, valid_mask):
    """MSE in the training depth codec's [0,1] range, on valid pixels only."""
    if predicted is None or predicted.shape != target.shape or predicted.ndim != 5:
        raise ValueError("depth prediction and label shapes must match exactly")
    expected = (target.shape[0], target.shape[1], *target.shape[-2:])
    if tuple(valid_mask.shape) != expected:
        raise ValueError(f"future depth mask must have shape {expected}")
    valid = valid_mask.bool().unsqueeze(2).expand_as(target)
    if not bool(valid.any()):
        raise ValueError("validation window has no valid depth pixels")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("non-finite validation depth")
    mse = (predicted.float()[valid] - target.float()[valid]).square().mean().item()
    return {"depth_mse_valid": mse}


def a2v_comparison_video(saved_val):
    """At most two CPU videos: GT left, A2V right; half-size for TensorBoard."""
    clips = []
    for row in saved_val[:2]:
        if "a2v_predicted_frames" not in row:
            continue
        pair = torch.cat((row["gt_frames"].float(), row["a2v_predicted_frames"].float()), dim=-1)
        clips.append(F.interpolate(pair, scale_factor=0.5, mode="area").clamp(0, 1))
    return torch.stack(clips) if clips else None


def _cpu_row(tensor, index):
    """Copy one sample. A batch view would keep the whole batch storage alive."""
    if tensor is None:
        return None
    return tensor[index].detach().to("cpu").clone()


def compute_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> Dict[str, float]:
    """Per-sample action MSE and RMSE.

    ``l2_error`` is the root mean square error, sqrt(mean(square error)) over
    each sample's action vector. It is not mean absolute error.
    """
    square = (predicted_actions - ground_truth_actions).float().square()
    mse_per_sample = square.reshape(predicted_actions.shape[0], -1).mean(1)
    rmse_per_sample = mse_per_sample.sqrt()
    return {
        "mse_loss": mse_per_sample.mean().item(),
        "l2_error": rmse_per_sample.mean().item(),
        "mse_std": mse_per_sample.std().item() if len(mse_per_sample) > 1 else 0.0,
        "l2_std": rmse_per_sample.std().item() if len(rmse_per_sample) > 1 else 0.0,
    }


@torch.no_grad()
def evaluate_training_model(model, dataloader, accelerator, config, num_eval_batches: int = 2) -> Dict[str, float]:
    """
    Local-only evaluation: no distributed aggregation; safe for rank0-only evaluation.
    """
    logger.info(f"Running UniDiffuser evaluation for {num_eval_batches} batches...")
    model.eval()
    
    from collections import defaultdict
    metrics = defaultdict(list)
    visual_samples = []

    saved_val = []

    for step, batch in enumerate(dataloader):
        if step > num_eval_batches:
            break
        if batch is None:
            continue
        
        # Inference
        gt_video = batch["video_frames"]
        logger.info(f"Running step {step}... (gt video: {gt_video.size()})")
        try:
            inference_details = {}
            predicted_frames, predicted_actions = inference_sample(model, batch, config, details=inference_details)
        except Exception as e:
            logger.error(f"step {step}, Error occurred during inference_sample:")
            logger.exception(e)  # This will print full traceback
            continue
        if predicted_frames is None:
            # 模态缺失（评审 #4）：video 被 drop/缺失时不生成视频，跳过视频指标与可视化
            logger.warning(
                f"step {step}, inference returned predicted_frames=None "
                "(video modality missing); video metrics/visualization skipped"
            )
        else:
            logger.info(f"(pred video: {predicted_frames.size()})")
            gt_frames = batch['video_frames'].to(predicted_frames.device)  # [B, T, C, H, W]
            predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)     # [B, T, C, H, W]

            t5_input_messages = batch.get("t5_input_messages")
            vlm_input_messages = inference_details["vlm_input_messages"]
            for i in range(predicted_frames.size(0)):
                saved_val.append(
                    {
                        "gt_first_frame": batch["first_frame"].cpu()[i],
                        "gt_frames": batch["video_frames"].cpu()[i],
                        "gt_initial_state": batch["initial_state"].cpu()[i],
                        "gt_actions": batch["action_sequence"].cpu()[i],
                        "predicted_frames": predicted_frames.cpu()[i],
                        "predicted_actions": predicted_actions.cpu()[i] if predicted_actions is not None else None,
                        "t5_input_messages": t5_input_messages[i] if t5_input_messages is not None else None,
                        "vlm_input_messages": vlm_input_messages[i] if vlm_input_messages is not None else None
                    }
                )

            # Video metrics (local)
            video_mse = F.mse_loss(predicted_frames, gt_frames, reduction='mean').item()
            metrics['video_mse'].append(video_mse)

            # Visualization sample
            if step == 0:
                visual_samples.append({
                    'predicted_frames': predicted_frames[:4],
                    'ground_truth_frames': gt_frames[:4],
                    'predicted_actions': predicted_actions[:4] if predicted_actions is not None else None,
                    'ground_truth_actions': batch.get('action_sequence', None)[:4] if batch.get('action_sequence', None) is not None else None
                })

        # Action metrics (local) - evaluate_training_model（与视频模态独立）
        if 'action_sequence' in batch and predicted_actions is not None:
            gt_actions = batch['action_sequence'].to(predicted_actions.device)
            # Handle length mismatch: trim both to match minimum length
            min_len = min(gt_actions.shape[1], predicted_actions.shape[1])
            gt_actions = gt_actions[:, :min_len]
            pred_actions = predicted_actions[:, :min_len]
            action_metrics = compute_action_metrics(pred_actions, gt_actions)
            for key, value in action_metrics.items():
                metrics[f'action_{key}'].append(value)
    
    # Aggregate metrics
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values))
    
    if visual_samples:
        sample = visual_samples[0]
        grid_visualization = create_video_grid(
            sample['predicted_frames'],
            sample['ground_truth_frames'],
            num_samples=4
        )
        final_metrics['visualization'] = grid_visualization
    
    # Generate video tensor for TensorBoard (first sample only)
    if saved_val:
        try:
            video_tensor = create_video_tensor_with_action_plot(saved_val[0], fps=5)
            final_metrics['video_tensor'] = video_tensor  # shape (T, C, H, W)
        except Exception as e:
            logger.error(f"Failed to generate video tensor:")
            logger.exception(e)  # This will print full traceback
    
    model.train()
    return final_metrics, saved_val

@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = 2,
                   *, seed=None, expected_batch_size=None, visualize=True,
                   max_saved_samples=_VAL_VIZ_SAMPLES):
    """Score exactly ``num_eval_batches`` batches locally, without collectives.

    The iterator is pulled that many times and no further, so a 15-batch
    round cannot consume the first batch of the next epoch. Inference noise
    is reseeded from ``system.val_seed`` and the training RNG is restored
    afterwards. A short round raises instead of returning empty metrics.
    """
    num_eval_batches = int(num_eval_batches)
    logger.info(f"Running UniDiffuser evaluation for {num_eval_batches} batches...")
    was_training = bool(model.training)
    model.eval()
    rng_state = _capture_rng_state()
    _seed_validation_noise(_validation_seed(config) if seed is None else seed)
    expected_batch = (_configured_batch_size(config) if expected_batch_size is None
                      else int(expected_batch_size))
    window = _validation_window(config)
    run_a2v = bool(getattr(getattr(config, "system", None), "val_a2v", False))
    run_depth = bool(getattr(getattr(config, "system", None), "val_depth", False))
    depth_attacher = None
    if window is not None:
        logger.info(
            "validation window: %d blocks, %d video frames, %d action steps",
            window["blocks"], window["frames"], window["actions"],
        )

    from collections import defaultdict
    metrics = defaultdict(list)
    visual_samples = []
    saved_val = []
    successes = 0
    failures = 0
    sample_count = 0
    last_error = None
    iterator = iter(dataloader)
    try:
        for step in range(num_eval_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                failures += 1
                last_error = f"dataloader exhausted at batch {step}"
                logger.error(last_error)
                break
            if batch is None:
                failures += 1
                last_error = f"batch {step} is empty"
                logger.error(last_error)
                continue

            gt_video = batch["video_frames"]
            logger.info(f"Running step {step}... (gt video: {gt_video.size()})")
            try:
                if run_depth and "depth_frames" not in batch:
                    if depth_attacher is None:
                        from thinker_wm.data_prepare.depth_attach import build_depth_attacher
                        depth_attacher = build_depth_attacher(config)
                    if depth_attacher is None:
                        raise ValueError("val_depth requires depth labels")
                    batch = depth_attacher.attach(batch)
                inference_rng = _capture_rng_state() if run_a2v else None
                inference_details = {}
                predicted_frames, predicted_actions = inference_sample(model, batch, config, details=inference_details)
            except Exception as exc:
                failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                logger.error("step %s, inference failed: %s", step, last_error)
                logger.exception(exc)
                continue
            rejection = _reject_prediction(predicted_frames, predicted_actions, batch, window)
            if rejection is not None:
                failures += 1
                last_error = f"batch {step}: {rejection}"
                logger.error(last_error)
                continue

            bsz = int(batch["video_frames"].shape[0])
            if expected_batch is not None and bsz != expected_batch:
                failures += 1
                last_error = (
                    f"batch {step} has {bsz} samples, expected {expected_batch}"
                )
                logger.error(last_error)
                break
            successes += 1
            sample_count += bsz

            logger.info(f"(pred video: {predicted_frames.size()})")
            gt_frames = batch["video_frames"].to(predicted_frames.device)
            predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)

            predicted_depth = None
            if run_depth:
                predicted_depth = inference_details["predicted_depth"]
                if predicted_depth is None:
                    raise ValueError("depth validation returned no prediction")
                predicted_depth = predicted_depth.permute(0, 2, 1, 3, 4)
                gt_depth = batch["depth_frames"].to(predicted_depth.device)
                future_depth_mask = batch["depth_valid_mask"][:, 1:].to(predicted_depth.device)
                for key, value in compute_depth_metrics(predicted_depth, gt_depth, future_depth_mask).items():
                    metrics[key].append(value)

            a2v_frames = None
            if run_a2v:
                # Pair the inference noise, but leave the joint evaluator's RNG
                # stream unchanged for subsequent batches. Any A2V failure raises
                # into the existing distributed error-sharing protocol.
                after_joint_rng = _capture_rng_state()
                try:
                    _restore_rng_state(inference_rng)
                    a2v_frames, _ = inference_sample(model, batch, config, a2v=True)
                    if a2v_frames is None:
                        raise ValueError("A2V returned no video")
                    a2v_frames = a2v_frames.permute(0, 2, 1, 3, 4)
                    for key, value in compute_a2v_video_metrics(
                            a2v_frames, gt_frames, window["blocks"]).items():
                        metrics[key].append(value)
                finally:
                    _restore_rng_state(after_joint_rng)

            room = max_saved_samples - len(saved_val)
            if room > 0:
                t5_input_messages = batch.get("t5_input_messages")
                vlm_input_messages = inference_details["vlm_input_messages"]
                for i in range(min(room, predicted_frames.shape[0])):
                    saved_val.append(
                        {
                            "gt_first_frame": _cpu_row(batch["first_frame"], i),
                            "gt_frames": _cpu_row(batch["video_frames"], i),
                            "gt_initial_state": _cpu_row(batch["initial_state"], i),
                            "gt_actions": _cpu_row(batch["action_sequence"], i),
                            "predicted_frames": _cpu_row(predicted_frames, i),
                            "predicted_actions": _cpu_row(predicted_actions, i)
                            if predicted_actions is not None else None,
                            "t5_input_messages": t5_input_messages[i]
                            if t5_input_messages is not None else None,
                            "vlm_input_messages": vlm_input_messages[i]
                            if vlm_input_messages is not None else None,
                        }
                    )

            if room > 0:
                take = min(room, predicted_frames.shape[0])
                for i, row in enumerate(saved_val[-take:]):
                    original = inference_details["original_vlm_input_messages"]
                    row["original_vlm_input_messages"] = original[i] if original is not None else None
                    if predicted_depth is not None:
                        row["predicted_depth"] = _cpu_row(predicted_depth, i)
                        row["gt_depth_frames"] = _cpu_row(batch["depth_frames"], i)
                        row["depth_valid_mask"] = _cpu_row(future_depth_mask, i)

            if a2v_frames is not None and room > 0:
                take = min(room, predicted_frames.shape[0])
                for i, row in enumerate(saved_val[-take:]):
                    row["a2v_predicted_frames"] = _cpu_row(a2v_frames, i)

            video_mse = F.mse_loss(predicted_frames, gt_frames, reduction="mean").item()
            metrics["video_mse"].append(video_mse)
            if visualize and step == 0:
                take = min(4, predicted_frames.shape[0])
                action_sequence = batch.get("action_sequence", None)
                visual_samples.append({
                    "predicted_frames": predicted_frames[:take].detach().clone(),
                    "ground_truth_frames": gt_frames[:take].detach().clone(),
                    "predicted_actions": predicted_actions[:take].detach().clone()
                    if predicted_actions is not None else None,
                    "ground_truth_actions": action_sequence[:take].detach().clone()
                    if action_sequence is not None else None,
                })

            gt_actions = batch["action_sequence"].to(predicted_actions.device)
            action_metrics = compute_action_metrics(predicted_actions, gt_actions)
            for key, value in action_metrics.items():
                metrics[f"action_{key}"].append(value)
            if window is not None:
                first = int(window["first_block_actions"])
                # Deployment executes the first block only. This is an extra
                # report; the full-window metrics above already required 24 steps.
                first_metrics = compute_action_metrics(
                    predicted_actions[:, :first], gt_actions[:, :first])
                for key, value in first_metrics.items():
                    metrics[f"action_first_block_{key}"].append(value)

        required_samples = (
            num_eval_batches * expected_batch if expected_batch is not None else sample_count
        )
        logger.info(
            "validation round successes=%d failures=%d samples=%d "
            "required_batches=%d required_samples=%d",
            successes, failures, sample_count, num_eval_batches, required_samples,
        )
        if successes != num_eval_batches or failures or sample_count != required_samples:
            raise RuntimeError(
                "validation incomplete: "
                f"successes={successes} failures={failures} samples={sample_count} "
                f"required_batches={num_eval_batches} required_samples={required_samples} "
                f"last_error={last_error}"
            )

        final_metrics = {}
        for key, values in metrics.items():
            if values:
                final_metrics[key] = float(np.mean(values))
                final_metrics[f"{key}_std"] = float(np.std(values))
        if visual_samples:
            sample = visual_samples[0]
            final_metrics["visualization"] = create_video_grid(
                sample["predicted_frames"],
                sample["ground_truth_frames"],
                num_samples=min(4, sample["predicted_frames"].shape[0]),
            )
        if visualize and run_a2v and saved_val:
            final_metrics["a2v_video_tensor"] = a2v_comparison_video(saved_val)
        if visualize and saved_val:
            try:
                final_metrics["video_tensor"] = create_video_tensor_with_action_plot(
                    saved_val[0], fps=5)
            except Exception as exc:
                logger.error("Failed to generate video tensor:")
                logger.exception(exc)
        return final_metrics, saved_val
    finally:
        _restore_rng_state(rng_state)
        model.train(was_training)

def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """
    Log evaluation metrics to tensorboard and wandb.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        writer: TensorBoard writer (can be None)
        accelerator: HuggingFace accelerator  
        global_step: Current training step
    """
    if accelerator.is_main_process:
        # Log scalar metrics
        log_dict = {}
        for key, value in metrics.items():
            if key not in ['visualization', 'visual_samples'] and isinstance(value, (int, float)):
                log_dict[f'eval/{key}'] = value
        
        # Log to accelerator (wandb)
        if log_dict:
            accelerator.log(log_dict, step=global_step)
        
        # Log to TensorBoard
        if writer is not None:
            # Log scalar metrics to TensorBoard
            for key, value in log_dict.items():
                writer.add_scalar(key, value, global_step)
            
            # Log grid visualization
            if 'visualization' in metrics:
                img_array = np.array(metrics['visualization']).transpose(2, 0, 1)
                writer.add_image('eval/video_grid', img_array, global_step)
            
            if metrics.get("a2v_video_tensor") is not None:
                # RoboDojo: 25 Hz commands / stride 2 = 12.5 Hz RGB.
                writer.add_video("eval/a2v_gt_left_pred_right",
                                 metrics["a2v_video_tensor"], global_step, fps=12.5)

            # Log video tensor
            if 'video_tensor' in metrics:
                video_tensor = metrics['video_tensor']  # shape (T, C, H, W)
                # Add batch dimension
                video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
                writer.add_video('eval/video_with_action', video_tensor, global_step, fps=5)

        # Print summary
        logger.info("=== UniDiffuser Evaluation Results ===")
        for key, value in metrics.items():
            if key != 'visualization' and isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
