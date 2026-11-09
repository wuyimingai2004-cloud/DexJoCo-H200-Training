"""Tiny CPU WAM: 22/44-D Joint24 forward/backward and audited weight migration.

Uses existing CPU harness replacements for VAE/VLM and attention kernels.
This does not validate the 5B backbone, CUDA, BF16, or DeepSpeed.
"""
from pathlib import Path
import sys
import tempfile
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_harness as H
import torch
from thinker_wm.models import temporal_training as TT
from thinker_wm.models.transfer import is_interface
from thinker_wm.utils.vlm_utils import build_vlm_task_message


def config(dimension):
    return H.config(num_video_frames=12, video_action_freq_ratio=2, action_t_downsample=1,
                    action_dim=dimension, action_state_dim=dimension, unified_action_dim=dimension,
                    control_hz=30., ar_latents_per_block=3, ar_packed_teacher_forcing=True,
                    ar_target_reads_earlier_targets=False, temporal_und_reads="observed",
                    enable_depth=False, history_enable=False, training_mode_probs=dict(joint=.6, a2v=.4))


class DexJoCoModel(unittest.TestCase):
    def test_pure_a2v_trains_action_conditioning_from_video_loss(self):
        c = config(22)
        c.training_mode_probs = dict(joint=0., a2v=1.)
        model = H.Harness(c).train()
        modes = {TT.sample_step_decision(c, step, True, False).mode for step in range(200)}
        self.assertEqual(modes, {TT.MODE_WORLD})
        batch = H.batch(b=1, frames=12, ratio=2)
        batch.update(state=torch.randn(1, 22), actions=torch.randn(1, 24, 22))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        output = model.training_step(**batch, dropout_seed_step=0)
        self.assertEqual(float(output["action_loss"]), 0.)
        self.assertGreater(float(output["video_loss"]), 0.)
        self.assertTrue(torch.isfinite(output["total_loss"]))
        output["total_loss"].backward()
        encoder = dict(model.named_parameters())["world_model.action_branch.encoder.action_embedding.weight"]
        self.assertIsNotNone(encoder.grad)
        self.assertTrue(torch.isfinite(encoder.grad).all())
        self.assertGreater(float(encoder.grad.abs().sum()), 0.)
        before = encoder.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, encoder))

    def test_22_and_44_dim_joint_a2v_updates(self):
        for dimension in (22, 44):
            model = H.Harness(config(dimension)).train()
            batch = H.batch(b=1, frames=12, ratio=2)
            batch.update(state=torch.zeros(1, dimension), actions=torch.randn(1, 24, dimension))
            decisions = {}
            for step in range(100):
                decision = TT.sample_step_decision(model.config, step, True, False)
                decisions.setdefault(decision.mode, step)
            self.assertEqual(set(decisions), {TT.MODE_JOINT, TT.MODE_WORLD})
            for mode, step in decisions.items():
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95))
                optimizer.zero_grad()
                output = model.training_step(**batch, dropout_seed_step=step)
                self.assertTrue(torch.isfinite(output["total_loss"]))
                self.assertEqual(float(output["action_loss"]) == 0, mode == TT.MODE_WORLD)
                output["total_loss"].backward()
                grads = [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in grads))
                before = grads[0].detach().clone()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
                self.assertFalse(torch.equal(before, grads[0]))

    def test_actual_loader_migrates_joint14_to_22_and_44(self):
        source = H.Harness(config(14))
        # The tiny harness bypasses UBT_WM.__init__; include its state-sized prompt
        # so the regression covers the parameter present in production checkpoints.
        source.ego4d_state_prompt = torch.nn.Parameter(torch.full((14,), 5.0))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initial.pt"
            torch.save({"module": source.state_dict()}, checkpoint)
            for dimension in (14, 22, 44):
                target = H.Harness(config(dimension), seed=123)
                target.ego4d_state_prompt = torch.nn.Parameter(torch.full((dimension,), .02))
                initial_prompt = target.ego4d_state_prompt.detach().clone()
                initial = {k: v.clone() for k, v in target.state_dict().items() if is_interface(k)}
                audit = Path(directory) / f"audit_{dimension}.json"
                target.load_pretrain_weights(str(checkpoint), action_interface_policy="dexjoco_rgb", audit_path=str(audit))
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, initial[key] if is_interface(key) else source.state_dict()[key])
                torch.testing.assert_close(target.ego4d_state_prompt, initial_prompt)
                record = json.loads(audit.read_text())
                self.assertIn("ego4d_state_prompt", record["reinitialized"])
                self.assertIn("ego4d_state_prompt", record["dropped"])
                self.assertEqual(len(record["source_sha256"]), 64)

                state, actions = torch.randn(1, dimension), torch.randn(1, 24, dimension)
                observed, returned_actions = target._apply_ego4d_action_state_prompt(
                    state, actions, source_datasets=["dexjoco"])
                self.assertIs(observed, state)
                self.assertIs(returned_actions, actions)

    def test_camera_prompt_matches_right_first_canvas(self):
        text = build_vlm_task_message("open microwave", style="dexjoco_tlayout")
        self.assertIn("right wrist is bottom-left", text)
        self.assertIn("Task: open microwave", text)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
