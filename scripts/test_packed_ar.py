"""Packed (clean/noisy copy) multi-block AR: training/inference contracts (CPU).

Run: python scripts/test_packed_ar.py
Window: ratio 4, t_downsample 4, 1 latent per block, 8 future frames ->
latents 0/1/2 (blocks 0/1/2), 32 actions (block 1 = 0..15, block 2 = 16..31).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rsi_cpu_stubs as S  # noqa: E402
import _rsi_harness as Hn  # noqa: E402

import torch  # noqa: E402

from thinker_wm.models import temporal_training as TT  # noqa: E402
from thinker_wm.models.temporal_attention import visibility  # noqa: E402
from thinker_wm.models.temporal_layout import (  # noqa: E402
    ROLE_CLEAN, ROLE_OBS, ROLE_TARGET, Segment, TemporalRequest, TimeGrid,
    packed_action_entries, packed_latent_entries,
)

ATOL = 1e-5
GRID = TimeGrid(video_action_freq_ratio=4, action_t_downsample=4)
B, C, H, W, A = 2, 8, 4, 4, 32
RGB_ONLY = dict(depth_first_frame_conditioned=False)


def train_req(**kw):
    return TemporalRequest(GRID, target_reads_earlier_targets=False, packed=True,
                           clean_blocks=1, first_target=1, last_target=2, **kw)


def window(seed):
    g = torch.Generator().manual_seed(seed)
    return dict(video=torch.randn(B, C, 3, H, W, generator=g),
                depth=torch.randn(B, C, 3, H, W, generator=g),
                action=torch.randn(B, A, 8, generator=g))


def hist(seed=5):
    g = torch.Generator().manual_seed(seed)
    return dict(latent=torch.randn(B, C, 2, H, W, generator=g),
                times=torch.tensor([[-2.0, -1.0]] * B),
                valid=torch.ones(B, 2, dtype=torch.bool),
                kind=torch.tensor([[1, 0]] * B),
                state=torch.randn(B, 2, 8, generator=g))


class Model:
    model = None

    @classmethod
    def get(cls):
        if cls.model is None:
            cls.model = S.tiny_world(enable_depth=True, enable_history=True)
        return cls.model


def run_packed(req, clean, noisy, state, history=None):
    """Packed forward; returns window-length outputs at TARGET positions."""
    plan = TT.pack_plan(req)
    v = TT.pack(clean["video"], noisy["video"], plan.video, 2)
    d = TT.pack(clean["depth"], noisy["depth"], plan.depth, 2)
    a = TT.pack(clean["action"], noisy["action"], plan.action, 1)
    with torch.no_grad():
        vo, ao, do = Model.get()(torch.tensor([[600.0], [800.0]]), None, None, v, a,
                                 und_cache=S.und_cache(B), depth=d, state=state,
                                 temporal=req, history=history)
    return (TT.unpack_targets(vo, plan.video, 3, 2), TT.unpack_targets(ao, plan.action, A, 1),
            TT.unpack_targets(do, plan.depth, 3, 2))


def close(a, b):
    return torch.allclose(a, b, atol=ATOL, rtol=0)


class LayoutRules(unittest.TestCase):
    def test_entries(self):
        r = train_req(**RGB_ONLY)
        self.assertEqual(packed_latent_entries(r), [
            (0, 0, ROLE_OBS), (1, 1, ROLE_CLEAN), (1, 1, ROLE_TARGET), (2, 2, ROLE_TARGET)])
        self.assertEqual(packed_latent_entries(r, depth=True), [
            (0, 0, ROLE_TARGET), (1, 1, ROLE_TARGET), (2, 2, ROLE_TARGET)])
        act = packed_action_entries(r)
        self.assertEqual(len(act), 48)
        self.assertEqual({e[2] for e in act[:16]}, {ROLE_CLEAN})
        world = packed_action_entries(train_req(action_role="world"))
        self.assertEqual([e[0] for e in world], list(range(A)))
        self.assertEqual({e[2] for e in world}, {ROLE_OBS})
        b2 = TT.block_request(r, 2)
        self.assertEqual(packed_latent_entries(b2), [
            (0, 0, ROLE_OBS), (1, 1, ROLE_CLEAN), (2, 2, ROLE_TARGET)])
        self.assertEqual(packed_latent_entries(b2, depth=True), [(2, 2, ROLE_TARGET)])

    def test_request_validation(self):
        with self.assertRaises(ValueError):
            TemporalRequest(GRID, packed=True, clean_blocks=1, last_target=2)  # reads targets
        with self.assertRaises(ValueError):
            TemporalRequest(GRID, target_reads_earlier_targets=False, packed=True,
                            clean_blocks=0, first_target=2, last_target=2)       # prefix gap

    def test_obs_does_not_read_same_block_clean_copy(self):
        q = Segment(torch.tensor([1, 2]), torch.tensor([ROLE_OBS, ROLE_OBS]),
                    torch.zeros(2, 3, dtype=torch.long))
        k = Segment(torch.tensor([1]), torch.tensor([ROLE_CLEAN]), torch.zeros(1, 3, dtype=torch.long))
        self.assertEqual(visibility(q, k).tolist(), [[False], [True]])


class PackedKernel(unittest.TestCase):
    def setUp(self):
        self.clean, self.noisy = window(0), window(1)
        self.state = torch.randn(B, 8, generator=torch.Generator().manual_seed(2))

    def test_training_layout_equals_block_by_block(self):
        """Every target of the 2-block training forward equals the forward that
        generates that block alone from a clean prefix (= inference)."""
        h = hist()
        for extra in (RGB_ONLY, {}):
            req = train_req(**extra)
            v, a, d = run_packed(req, self.clean, self.noisy, self.state, h)
            v1, a1, d1 = run_packed(TT.block_request(req, 1), self.clean, self.noisy, self.state, h)
            v2, a2, d2 = run_packed(TT.block_request(req, 2), self.clean, self.noisy, self.state, h)
            self.assertTrue(close(v[:, :, 1], v1[:, :, 1]) and close(v[:, :, 2], v2[:, :, 2]))
            self.assertTrue(close(a[:, :16], a1[:, :16]) and close(a[:, 16:], a2[:, 16:]))
            self.assertTrue(close(d[:, :, 1], d1[:, :, 1]) and close(d[:, :, 2], d2[:, :, 2]))
            if extra:
                self.assertTrue(close(d[:, :, 0], d1[:, :, 0]))

    def test_target_reads_neither_own_clean_copy_nor_other_targets(self):
        req = train_req(**RGB_ONLY)
        v0, a0, d0 = run_packed(req, self.clean, self.noisy, self.state)
        c = {k: x.clone() for k, x in self.clean.items()}
        c["video"][:, :, 1] += 3.0
        c["action"][:, :16] += 3.0
        v1, a1, d1 = run_packed(req, c, self.noisy, self.state)
        self.assertTrue(close(v1[:, :, 1], v0[:, :, 1]) and close(a1[:, :16], a0[:, :16]))
        self.assertFalse(close(v1[:, :, 2], v0[:, :, 2]))
        self.assertTrue(close(d1[:, :, :2], d0[:, :, :2]))
        n = {k: x.clone() for k, x in self.noisy.items()}
        n["video"][:, :, 1] += 3.0
        n["action"][:, :16] += 3.0
        n["depth"][:, :, :2] += 3.0
        v2, a2, d2 = run_packed(req, self.clean, n, self.state)
        self.assertTrue(close(v2[:, :, 2], v0[:, :, 2]) and close(a2[:, 16:], a0[:, 16:]))
        self.assertTrue(close(d2[:, :, 2], d0[:, :, 2]))

    def test_world_mode_target_ignores_same_block_clean_copy(self):
        req = train_req(action_role="world", **RGB_ONLY)
        v0, _, _ = run_packed(req, self.clean, self.noisy, self.state)
        c = {k: x.clone() for k, x in self.clean.items()}
        c["video"][:, :, 1] += 3.0
        v1, _, _ = run_packed(req, c, self.noisy, self.state)
        self.assertTrue(close(v1[:, :, 1], v0[:, :, 1]))
        self.assertFalse(close(v1[:, :, 2], v0[:, :, 2]))


def packed_config(**over):
    return Hn.config(ar_packed_teacher_forcing=True, ar_target_reads_earlier_targets=False, **over)


def capture_calls(model):
    calls = []
    real = model.world_model.forward

    def spy(*a, **k):
        calls.append(dict(video=a[3], action=a[4], **k))
        return real(*a, **k)

    model.world_model.forward = spy
    return calls


class PackedTraining(unittest.TestCase):
    def test_config_validation(self):
        TT.validate_temporal_config(packed_config())
        with self.assertRaises(ValueError):
            TT.validate_temporal_config(Hn.config(ar_packed_teacher_forcing=True))
        with self.assertRaises(ValueError):
            TT.validate_temporal_config(packed_config(ar_cut_probs=[1.0]))
        d = TT.sample_step_decision(packed_config(), 0, True, True).request
        self.assertEqual((d.packed, d.clean_blocks, d.first_target, d.last_target), (True, 1, 1, 2))

    def test_step_packs_inputs_and_supervises_all_blocks(self):
        m = Hn.Harness(packed_config(enable_depth=True, depth_condition_mode="rgb_only")).train()
        b = Hn.batch(depth=True)
        ctx = m.prepare_training_context(**b)
        calls = capture_calls(m)
        out = m.training_step(**b, training_context=ctx, dropout_seed_step=0)
        call, clean = calls[-1], ctx["clean_full_latent"]
        self.assertEqual(call["video"].shape[2], 4)                      # obs, clean 1, noisy 1, noisy 2
        torch.testing.assert_close(call["video"][:, :, :2], clean[:, :, :2])
        self.assertFalse(torch.allclose(call["video"][:, :, 2], clean[:, :, 1]))
        self.assertEqual(call["action"].shape[1], 48)
        torch.testing.assert_close(call["action"][:, :16], ctx["actions"][:, :16].to(call["action"].dtype))
        self.assertEqual(call["depth"].shape[2], 3)
        for k in ("video_loss", "action_loss", "depth_loss"):
            self.assertGreater(float(out[k]), 0.0, k)
        out["total_loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.world_model.action_branch.parameters()))

    def test_formal_four_block_window(self):
        """16 frames / 64 steps / t_downsample 1: 4 blocks, all supervised."""
        cfg = packed_config(num_video_frames=16, action_t_downsample=1)
        req = TT.sample_step_decision(cfg, 0, True, True).request
        plan = TT.pack_plan(req)
        self.assertEqual((req.clean_blocks, req.last_target), (3, 4))
        self.assertEqual((len(plan.video), len(plan.action)), (1 + 3 + 4, 48 + 64))
        sup = TT.packed_supervision(plan, 64)
        self.assertEqual(sup.video_frames.tolist(), [1, 2, 3, 4])
        self.assertTrue(bool(sup.action_steps.all()))
        m = Hn.Harness(cfg).train()
        out = m.training_step(**Hn.batch(frames=16), dropout_seed_step=0)
        self.assertGreater(float(out["action_loss"]), 0.0)
        out["total_loss"].backward()

    def test_world_mode_step(self):
        m = Hn.Harness(packed_config()).train()
        b = Hn.batch()
        req = TT.sample_step_decision(m.config, 0, True, True).request
        import dataclasses
        dec = TT.StepDecision(dataclasses.replace(req, action_role="world"), TT.MODE_WORLD)
        calls = capture_calls(m)
        out = m.training_step(**b, temporal_decision=dec, dropout_seed_step=0)
        self.assertEqual(calls[-1]["action"].shape[1], A)
        self.assertEqual(float(out["action_loss"]), 0.0)
        self.assertGreater(float(out["video_loss"]), 0.0)
        out["total_loss"].backward()


class PackedA2VWithHistory(unittest.TestCase):
    """Periodic-eval A2V arm: given commands + history + all blocks in one call."""

    def test_a2v_history_all_blocks(self):
        m = Hn.Harness(packed_config(history_enable=True)).eval()
        b = Hn.batch(history_slots=4)
        inp = dict(first_frame=b["first_frame"], state=b["state"], vlm_input_messages=b["vlm_input_messages"],
                   num_inference_steps=2, history_inputs=b["history_inputs"], num_blocks=2)

        def run(cmd):
            torch.manual_seed(0)
            with torch.no_grad():
                return m.inference_step(**inp, condition_actions=cmd)[0]

        cmd = b["actions"][:, :32]
        v = run(cmd)
        self.assertEqual(v.shape[2], 8)
        torch.testing.assert_close(v, run(cmd))
        held = run(cmd[:, :1].expand_as(cmd).contiguous())
        self.assertGreater(float((v - held).abs().max()), 1e-4)


class PackedInference(unittest.TestCase):
    def setUp(self):
        self.m = Hn.Harness(packed_config(enable_depth=True, depth_condition_mode="rgb_only")).eval()
        b = Hn.batch()
        g = torch.Generator().manual_seed(3)
        self.inp = dict(first_frame=b["first_frame"], state=b["state"],
                        vlm_input_messages=b["vlm_input_messages"], num_inference_steps=3,
                        decode_outputs=False,
                        init_video_noise=torch.randn(B, C, 3, H, W, generator=g),
                        init_depth_noise=torch.randn(B, C, 3, H, W, generator=g),
                        init_action_noise=torch.randn(B, A, 8, generator=g))
        self.actions = b["actions"]

    def test_default_one_block_and_prefix_consistency(self):
        cap1, cap2 = {}, {}
        with torch.no_grad():
            _, a1 = self.m.inference_step(**self.inp, latent_capture=cap1)
            _, a2 = self.m.inference_step(**self.inp, num_blocks=2, latent_capture=cap2)
        self.assertEqual(tuple(a1.shape), (B, 16, 8))
        self.assertEqual(tuple(a2.shape), (B, 32, 8))
        self.assertEqual(cap1["video_latent"].shape[2], 2)
        self.assertEqual(cap2["depth_latent"].shape[2], 3)
        torch.testing.assert_close(a2[:, :16], a1)
        torch.testing.assert_close(cap2["video_latent"][:, :, :2], cap1["video_latent"])
        torch.testing.assert_close(cap2["depth_latent"][:, :, :2], cap1["depth_latent"])

    def test_four_block_prefix_consistency(self):
        m = Hn.Harness(packed_config(num_video_frames=16, enable_depth=True,
                                     depth_condition_mode="rgb_only")).eval()
        g = torch.Generator().manual_seed(4)
        inp = dict(self.inp, init_video_noise=torch.randn(B, C, 5, H, W, generator=g),
                   init_depth_noise=torch.randn(B, C, 5, H, W, generator=g),
                   init_action_noise=torch.randn(B, 64, 8, generator=g))
        cap1, cap4 = {}, {}
        with torch.no_grad():
            _, a1 = m.inference_step(**inp, latent_capture=cap1)
            _, a4 = m.inference_step(**inp, num_blocks=4, latent_capture=cap4)
        self.assertEqual(tuple(a4.shape), (B, 64, 8))
        torch.testing.assert_close(a4[:, :16], a1)
        torch.testing.assert_close(cap4["video_latent"][:, :, :2], cap1["video_latent"])
        torch.testing.assert_close(cap4["depth_latent"][:, :, :2], cap1["depth_latent"])

    def test_four_block_d0_and_block1_ignore_later_blocks(self):
        req = TemporalRequest(GRID, target_reads_earlier_targets=False, packed=True, clean_blocks=1,
                              first_target=1, last_target=4, **RGB_ONLY)
        plan = TT.pack_plan(req)
        g = torch.Generator().manual_seed(11)
        mk = lambda: dict(video=torch.randn(B, C, 5, H, W, generator=g),  # noqa: E731
                          depth=torch.randn(B, C, 5, H, W, generator=g),
                          action=torch.randn(B, 64, 8, generator=g))
        clean, noisy, st = mk(), mk(), torch.randn(B, 8, generator=g)

        def run(c, n):
            v = TT.pack(c["video"], n["video"], plan.video, 2)
            d = TT.pack(c["depth"], n["depth"], plan.depth, 2)
            a = TT.pack(c["action"], n["action"], plan.action, 1)
            with torch.no_grad():
                vo, ao, do = Model.get()(torch.tensor([[600.0], [800.0]]), None, None, v, a,
                                         und_cache=S.und_cache(B), depth=d, state=st, temporal=req)
            return (TT.unpack_targets(vo, plan.video, 5, 2), TT.unpack_targets(ao, plan.action, 64, 1),
                    TT.unpack_targets(do, plan.depth, 5, 2))

        base = run(clean, noisy)
        c = {k: v.clone() for k, v in clean.items()}
        n = {k: v.clone() for k, v in noisy.items()}
        for data in (c, n):
            data["video"][:, :, 2:] += 10
            data["depth"][:, :, 2:] -= 10
            data["action"][:, 16:] += 10
        changed = run(c, n)
        self.assertTrue(close(base[0][:, :, 1], changed[0][:, :, 1]))
        self.assertTrue(close(base[1][:, :16], changed[1][:, :16]))
        self.assertTrue(close(base[2][:, :, :2], changed[2][:, :, :2]))
        self.assertFalse(close(base[0][:, :, 4], changed[0][:, :, 4]))

    def test_unregistered_unified_columns_stay_zero(self):
        # 8 real dims in a 10-column unified space: training feeds zeros there.
        m = Hn.Harness(packed_config(unified_action_dim=10)).eval()
        calls = capture_calls(m)
        inp = {k: v for k, v in self.inp.items() if not k.startswith("init_")}
        with torch.no_grad():
            _, a = m.inference_step(**inp, num_blocks=2)
        self.assertEqual(tuple(a.shape), (B, 32, 8))
        self.assertTrue(calls)
        for c in calls:
            self.assertEqual(c["action"].shape[-1], 10)
            self.assertEqual(float(c["action"][..., 8:].abs().max()), 0.0)

    def test_num_blocks_bounds_and_world_mode(self):
        with self.assertRaises(ValueError):
            self.m.inference_step(**self.inp, num_blocks=3)
        import dataclasses
        req = dataclasses.replace(TT.sample_step_decision(self.m.config, 0, False, False).request,
                                  action_role="world")
        with torch.no_grad():
            _, a = self.m.inference_step(**self.inp, temporal_request=req,
                                         condition_actions=self.actions[:, :16])
        torch.testing.assert_close(a, self.actions[:, :16])

    def test_non_packed_rejects_num_blocks(self):
        m = Hn.Harness(Hn.config()).eval()
        with self.assertRaises(ValueError):
            m.inference_step(**self.inp, num_blocks=1)

    def test_inference_api_window(self):
        from thinker_wm.models.inference_api import GenerationSettings, WAMInference
        w = WAMInference(self.m, checkpoint="t", action_stats_version="t", camera_transform="t",
                         settings=GenerationSettings(num_inference_steps=2))
        self.assertEqual((w.num_blocks, w.num_frames), (1, 4))
        w2 = WAMInference(self.m, checkpoint="t", action_stats_version="t", camera_transform="t",
                          settings=GenerationSettings(num_blocks=2))
        self.assertEqual(w2.num_frames, 8)
        with self.assertRaises(ValueError):
            WAMInference(Hn.Harness(Hn.config()).eval(), checkpoint="t", action_stats_version="t",
                         camera_transform="t", settings=GenerationSettings(num_blocks=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
