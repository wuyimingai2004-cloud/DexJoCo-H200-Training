"""Joint24 (robodojo_joint24_v1): one jointly denoised future block, no in-window AR (CPU).

Run: python scripts/test_joint24_layout.py
The YAML window is 12 frames / 24 steps = latents 0..3, future block 1 = latents
1..3 and steps 0..23. Model-level checks use the tiny harness with the same
structure (8 frames, ratio 4, ar_latents_per_block 2 -> one 32-step block).
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_harness as Hn  # noqa: E402

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from thinker_wm.models import temporal_training as TT  # noqa: E402
from thinker_wm.models.temporal_attention import visibility  # noqa: E402
from thinker_wm.models.temporal_layout import (  # noqa: E402
    ROLE_CLEAN, ROLE_OBS, ROLE_TARGET, entries_action_segment, entries_video_segment,
)

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "third_party/Thinker-WM/configs/robodojo_joint24_v1.yaml"
B, C, H, W, A = 2, 8, 4, 4, 32


def yaml_config():
    cfg = OmegaConf.load(YAML)
    return Hn.config(num_video_frames=int(cfg.common.num_video_frames),
                     video_action_freq_ratio=int(cfg.common.video_action_freq_ratio),
                     action_t_downsample=int(cfg.model.wm.action_t_downsample),
                     **TT.motus_temporal_kwargs(cfg))


def single_block_config(**over):
    return Hn.config(ar_packed_teacher_forcing=True, ar_target_reads_earlier_targets=False,
                     ar_latents_per_block=2, **over)


def capture_calls(model):
    calls = []
    real = model.world_model.forward

    def spy(*a, **k):
        calls.append(dict(video=a[3], action=a[4], **k))
        return real(*a, **k)

    model.world_model.forward = spy
    return calls


class YamlWindow(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml_config()
        TT.validate_temporal_config(self.cfg)
        self.grid = TT.time_grid_from_config(self.cfg)

    def test_one_future_block_of_24_steps(self):
        self.assertEqual(self.grid.block_action_steps, 24)
        self.assertAlmostEqual(self.grid.block_seconds, 0.96)
        self.assertEqual(self.grid.num_blocks(self.grid.num_latents(12)), 2)  # current + one future
        self.assertEqual(set(self.grid.action_token_blocks(24)), {1})
        self.assertFalse(self.cfg.rgb_prefix_error_enable or self.cfg.joint_action_prefix_error_enable)

    def test_joint_plan_has_no_clean_future_copy(self):
        req = TT.sample_step_decision(self.cfg, 0, True, False).request
        req = dataclasses.replace(req, action_role="policy")
        self.assertEqual((req.packed, req.clean_blocks, req.first_target, req.last_target), (True, 0, 1, 1))
        plan = TT.pack_plan(req)
        self.assertEqual(list(plan.video), [(0, 0, ROLE_OBS)] + [(f, 1, ROLE_TARGET) for f in (1, 2, 3)])
        self.assertEqual(list(plan.action), [(k, 1, ROLE_TARGET) for k in range(24)])
        self.assertEqual(list(plan.depth), [(0, 0, ROLE_TARGET)] + [(f, 1, ROLE_TARGET) for f in (1, 2, 3)])
        for entries in (plan.video, plan.action, plan.depth):
            self.assertNotIn(ROLE_CLEAN, [e[2] for e in entries])
        sup = TT.packed_supervision(plan, 24)
        self.assertEqual(sup.video_frames.tolist(), [1, 2, 3])
        self.assertTrue(bool(sup.action_steps.all()))

    def test_a2v_actions_are_given_commands(self):
        req = dataclasses.replace(TT.sample_step_decision(self.cfg, 0, True, False).request, action_role="world")
        plan = TT.pack_plan(req)
        self.assertEqual(list(plan.action), [(k, 1, ROLE_OBS) for k in range(24)])
        self.assertFalse(bool(TT.packed_supervision(plan, 24).action_steps.any()))

    def test_mode_mixture_is_joint_and_a2v_only(self):
        modes = [TT.sample_step_decision(self.cfg, s, True, False).mode for s in range(2000)]
        self.assertEqual(set(modes), {TT.MODE_JOINT, TT.MODE_WORLD})
        self.assertAlmostEqual(modes.count(TT.MODE_JOINT) / len(modes), 0.6, delta=0.04)

    def test_future_tokens_are_bidirectional_within_the_block(self):
        req = dataclasses.replace(TT.sample_step_decision(self.cfg, 0, False, False).request, action_role="policy")
        plan = TT.pack_plan(req)
        video = entries_video_segment(plan.video, 1, 1)
        action = entries_action_segment(plan.action, self.grid, has_state=True)
        vv = visibility(video, video, target_reads_earlier_targets=False)
        va = visibility(video, action, target_reads_earlier_targets=False)
        av = visibility(action, video, target_reads_earlier_targets=False)
        tgt_v, tgt_a = video.role == ROLE_TARGET, action.role == ROLE_TARGET
        self.assertTrue(bool(vv[tgt_v][:, tgt_v].all()))      # every future frame reads every other
        self.assertTrue(bool(va[tgt_v][:, tgt_a].all()))      # ... and every future action
        self.assertTrue(bool(av[tgt_a][:, tgt_v].all()))
        self.assertFalse(bool(vv[~tgt_v][:, tgt_v].any()))    # the observation never reads targets


class SingleBlockModel(unittest.TestCase):
    def test_joint_step_feeds_only_noisy_future(self):
        m = Hn.Harness(single_block_config(enable_depth=True, depth_condition_mode="rgb_only")).train()
        b = Hn.batch(depth=True)
        ctx = m.prepare_training_context(**b)
        dec = TT.StepDecision(dataclasses.replace(
            TT.sample_step_decision(m.config, 0, True, False).request, action_role="policy"), TT.MODE_JOINT)
        calls = capture_calls(m)
        out = m.training_step(**b, training_context=ctx, temporal_decision=dec, dropout_seed_step=0)
        call, clean = calls[-1], ctx["clean_full_latent"]
        self.assertEqual(call["video"].shape[2], 3)          # obs + 2 noisy future latents, no clean copy
        torch.testing.assert_close(call["video"][:, :, 0], clean[:, :, 0])
        for f in (1, 2):
            self.assertFalse(torch.allclose(call["video"][:, :, f], clean[:, :, f]))
        self.assertEqual(call["action"].shape[1], A)          # 32 noisy steps, no clean copy
        self.assertFalse(torch.allclose(call["action"], ctx["actions"].to(call["action"].dtype)))
        for k in ("video_loss", "action_loss", "depth_loss"):
            self.assertGreater(float(out[k]), 0.0, k)
        out["total_loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.world_model.action_branch.parameters()))

    def test_a2v_step_conditions_on_gt_actions_without_action_loss(self):
        m = Hn.Harness(single_block_config()).train()
        b = Hn.batch()
        ctx = m.prepare_training_context(**b)
        dec = TT.StepDecision(dataclasses.replace(
            TT.sample_step_decision(m.config, 0, True, True).request, action_role="world"), TT.MODE_WORLD)
        calls = capture_calls(m)
        out = m.training_step(**b, training_context=ctx, temporal_decision=dec, dropout_seed_step=0)
        torch.testing.assert_close(calls[-1]["action"], ctx["actions"].to(calls[-1]["action"].dtype))
        self.assertEqual(float(out["action_loss"]), 0.0)
        self.assertGreater(float(out["video_loss"]), 0.0)
        out["total_loss"].backward()

    def test_inference_runs_one_denoising_loop(self):
        steps = 3
        b = Hn.batch()
        inp = dict(first_frame=b["first_frame"], state=b["state"],
                   vlm_input_messages=b["vlm_input_messages"], num_inference_steps=steps,
                   decode_outputs=False)
        single = Hn.Harness(single_block_config()).eval()
        calls = capture_calls(single)
        cap = {}
        with torch.no_grad():
            _, actions = single.inference_step(**inp, latent_capture=cap)
        self.assertEqual(tuple(actions.shape), (B, A, 8))
        self.assertEqual(cap["video_latent"].shape[2], 3)
        self.assertEqual(len(calls), steps)                  # not one loop per 8-step block
        with self.assertRaises(ValueError):
            single.inference_step(**inp, num_blocks=2)
        ar = Hn.Harness(Hn.config(ar_packed_teacher_forcing=True,
                                  ar_target_reads_earlier_targets=False)).eval()
        ar_calls = capture_calls(ar)
        with torch.no_grad():
            ar.inference_step(**inp, num_blocks=2)
        self.assertEqual(len(ar_calls), 2 * steps)

    def test_inference_api_returns_the_whole_window(self):
        from thinker_wm.models.inference_api import GenerationSettings, WAMInference
        w = WAMInference(Hn.Harness(single_block_config()).eval(), checkpoint="t",
                         action_stats_version="t", camera_transform="t",
                         settings=GenerationSettings(num_inference_steps=2, num_blocks=1))
        self.assertEqual((w.num_blocks, w.num_frames, w.num_frames * w.ratio), (1, 8, A))


if __name__ == "__main__":
    unittest.main(verbosity=2)
