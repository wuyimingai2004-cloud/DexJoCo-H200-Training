"""CPU contract and pipeline tests. No simulator, model weights or CUDA required."""
from __future__ import annotations

from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from thinker_wm.data_prepare.dexjoco.contracts import (
    TASKS, action_to_environment, state_to_model, validate_manifest, write_json,
)
from thinker_wm.data_prepare.dexjoco.prepare import prepare_release
from thinker_wm.data_prepare.dexjoco.recipe import build_config, install_statistics, write_configs
from thinker_wm.data_prepare.dexjoco.launch import launch, launch_queue, plan_launch
from thinker_wm.models.transfer import select_transfer_weights


def fixture_episode(path, task, *, value=1., count=65):
    path.mkdir(parents=True)
    state = np.zeros((count, 23 * task.arms + 7), dtype=np.float32)
    state[:, :3] = value
    state[:, 3] = 1
    if task.arms == 1:
        state[:, 7:23] = .2
    else:
        state[:, 7:10] = value + .1
        state[:, 10] = 1
        state[:, 14:30] = .2
        state[:, 30:46] = .4
    state[:, 23 * task.arms:] = 9999  # privileged state must never enter the model
    action = state_to_model(state, task.arms)
    action[:, 0] += np.arange(count) * .001
    np.savez(path / "replay.npz", state=state, action_rotvec=action, timestamp=np.arange(count) / 30)
    (path / "videos").mkdir()
    for i, camera in enumerate(task.cameras):
        frames = np.full((count, 8, 8, 3), 40 + 40 * i, dtype=np.uint8)
        np.save(path / "videos" / f"{camera}.npy", frames)


class ContractsTest(unittest.TestCase):
    def test_dual_arm_order_and_quaternion_roundtrip(self):
        a = np.arange(44, dtype=np.float32) * .01
        env = action_to_environment(a, 2)
        self.assertEqual(env.shape, (46,))
        np.testing.assert_allclose(env[14:30], a[6:22])
        np.testing.assert_allclose(env[30:46], a[28:44])
        np.testing.assert_allclose(state_to_model(np.r_[env, 9999], 2), a, atol=1e-6)
        env[3:7] *= -1
        np.testing.assert_allclose(state_to_model(env, 2), a, atol=1e-6)

    def test_quaternion_rejects_corrupt_values(self):
        with self.assertRaisesRegex(ValueError, "quaternion"):
            state_to_model(np.zeros(23), 1)

    def test_split_leakage(self):
        row = dict(task="pick_bucket", episode_dir="/train", source_group="original1",
                   layout_id="layout1", instruction="pick", split="train")
        d = dict(schema="dexjoco-training-manifest/v1", alignment="obs_t_action_t",
                 episodes=[row, dict(row, episode_dir="/augmented", split="test")])
        with self.assertRaisesRegex(ValueError, "trajectory crosses"):
            validate_manifest(d)
        d["episodes"][1]["source_group"] = "original2"
        with self.assertRaisesRegex(ValueError, "layout crosses"):
            validate_manifest(d)
        validate_manifest(d, strict_layout=False)

    def test_transfer_reinitializes_even_equal_shapes(self):
        io = "world_model.action_branch.encoder.action_embedding.weight"
        src = {"backbone.weight": torch.ones(2, 2), io: torch.ones(2, 2),
               "world_model.history_encoder.old": torch.ones(1)}
        dst = {"backbone.weight": torch.zeros(2, 2), io: torch.zeros(2, 2)}
        selected, audit = select_transfer_weights(src, dst)
        self.assertEqual(list(selected), ["backbone.weight"])
        self.assertEqual(audit["reinitialized"], [io])
        src["backbone.weight"] = torch.ones(3, 3)
        with self.assertRaisesRegex(ValueError, "incompatible shared backbone"):
            select_transfer_weights(src, dst)
        with self.assertRaisesRegex(ValueError, "missing"):
            select_transfer_weights({}, dst)

    def test_transfer_reinitializes_robot_state_prompt_without_relaxing_backbone(self):
        source = {"backbone.weight": torch.ones(2, 2), "ego4d_state_prompt": torch.ones(14)}
        for dimension in (14, 22, 44):
            with self.subTest(dimension=dimension):
                target = {"backbone.weight": torch.zeros(2, 2),
                          "ego4d_state_prompt": torch.zeros(dimension)}
                selected, audit = select_transfer_weights(source, target)
                self.assertEqual(list(selected), ["backbone.weight"])
                self.assertEqual(audit["reinitialized"], ["ego4d_state_prompt"])
                self.assertIn("ego4d_state_prompt", audit["dropped"])
                for invalid in (
                    dict(source, **{"backbone.weight": torch.ones(3, 3)}),
                    {"ego4d_state_prompt": source["ego4d_state_prompt"]},
                    dict(source, ego4d_state_prompt_other=torch.ones(14)),
                ):
                    with self.assertRaisesRegex(ValueError, "incompatible shared backbone"):
                        select_transfer_weights(invalid, target)


class RawFormatsTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("zarr"), "Zarr is an optional raw-data test dependency")
    def test_official_zarr_native_quaternion_action(self):
        import zarr
        from thinker_wm.data_prepare.dexjoco.raw import read_lowdim
        for task in TASKS[:2]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture_episode(root / "fixture", task)
                with np.load(root / "fixture/replay.npz") as arrays:
                    state, timestamp = arrays["state"], arrays["timestamp"]
                    action = action_to_environment(arrays["action_rotvec"], task.arms)
                group = zarr.open_group(str(root / "replay.zarr"), mode="w")
                for key, value in dict(state=state, timestamp=timestamp, action=action).items():
                    group.create_dataset("data/" + key, data=value)
                group.create_dataset("meta/episode_ends", data=np.array([65]))
                s, a, t = read_lowdim(root, task)
                self.assertEqual(a.shape, (65, task.dimension))
                self.assertLess(s.max(), 2)
                np.testing.assert_allclose(t, np.arange(65) / 30)
                group["data/timestamp"][1] = .9
                with self.assertRaisesRegex(ValueError, "30 Hz"):
                    read_lowdim(root, task)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is an optional raw-data test dependency")
    def test_real_mp4_decode_and_frame_count_guard(self):
        import cv2
        from thinker_wm.data_prepare.dexjoco.raw import inspect_video, read_frames
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30., (16, 16))
            self.assertTrue(writer.isOpened())
            for i in range(30):
                writer.write(np.full((16, 16, 3), 20 + i * 5, dtype=np.uint8))
            writer.release()
            inspect_video(path, 30)
            decoded = read_frames(str(path), [20, 22, 24, 20])
            self.assertEqual(decoded.shape, (4, 16, 16, 3))
            self.assertGreater(decoded[2].mean(), decoded[0].mean() + 15)
            np.testing.assert_array_equal(decoded[0], decoded[3])
            with self.assertRaisesRegex(ValueError, "video frames"):
                inspect_video(path, 31)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="dexjoco-training-test-")
        cls.root = Path(cls.tmp.name)
        rows = []
        for task in TASKS:
            for split, value in (("train", 1.), ("dev", 10.)):
                path = cls.root / "raw" / task.name / split
                fixture_episode(path, task, value=value)
                rows.append(dict(task=task.name, episode_dir=str(path), split=split,
                                 source_group=f"{split}_original", layout_id=split, instruction="test task"))
            # Test-only path deliberately does not exist; preparation must not read it.
            rows.append(dict(task=task.name, episode_dir=str(cls.root / "absent" / task.name), split="test",
                             source_group="test_original", layout_id="test", instruction="test task"))
        manifest = cls.root / "manifest.json"
        write_json(manifest, dict(schema="dexjoco-training-manifest/v1", alignment="obs_t_action_t", episodes=rows))
        cls.release = cls.root / "release"
        prepare_release(manifest, cls.release, [t.name for t in TASKS], target_episodes=1)
        cls.kwargs = dict(release=cls.release, wan_dir=cls.root / "wan", vlm_dir=cls.root / "vlm",
                          init_checkpoint=cls.root / "initial.pt", output_root=cls.root / "runs", batch_size=4)
        cls.configs = write_configs(cls.root / "configs", "test_recipe", **cls.kwargs)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_six_configs_match_report_and_are_isolated(self):
        configs = [yaml.safe_load(p.read_text()) for p in self.configs]
        self.assertEqual(len({c["training"]["seed"] for c in configs}), 6)
        self.assertEqual(len({c["system"]["checkpoint_dir"] for c in configs}), 6)
        self.assertEqual(len({c["finetune"]["checkpoint_path"] for c in configs}), 1)
        for task, c in zip(TASKS, configs):
            self.assertEqual(c["common"]["action_dim"], task.dimension)
            self.assertEqual(c["training"]["max_steps"], 20000)
            self.assertEqual(c["system"]["save_interval"], 5000)
            self.assertFalse(c["model"]["wm"]["enable_depth"])
            self.assertEqual(c["training"]["cycle_length"], 1000000)
        with self.assertRaises(FileExistsError):
            write_configs(self.root / "configs", "test_recipe", **self.kwargs)

    def test_statistics_never_see_development_or_privileged_state(self):
        from thinker_wm.models.action_stats import normalize, denormalize
        c = OmegaConf.load(self.configs[0])
        install_statistics(c)
        stats = json.loads(Path(c.dexjoco.action_stats_path).read_text())
        self.assertAlmostEqual(stats["mean"][0], 1.032, places=5)
        self.assertLess(max(stats["mean"]), 2)
        x = torch.arange(22).float().view(1, 22) * .01
        torch.testing.assert_close(denormalize(normalize(x, TASKS[0].robot), TASKS[0].robot), x)

    def test_window_selection_is_unique_and_temporally_aligned(self):
        from thinker_wm.data_prepare.factory import _indexed_sources
        from motus_data.indexed_pipeline import open_dataset
        from motus_data.sample_pool import resolve_sample_pool
        from motus_data.releases import get_episode_dataset_release
        c = OmegaConf.load(self.configs[0])
        source = _indexed_sources(c)[0]
        repo = open_dataset("dexjoco", release=get_episode_dataset_release("dexjoco"),
                            catalog_path=source["catalog_path"], index_path=source["index_path"],
                            readers=source["readers"])
        try:
            pool = resolve_sample_pool(source["sample_pool"], repo, seed=c.training.seed)
            pool.set_epoch(0)
            selections = [pool.generate(i) for i in range(32)]
            starts = [s["action_indices"][0] for s in selections]
            self.assertEqual(len(set(starts)), 32)
            for s in selections:
                start = s["state_indices"][0]
                self.assertEqual(s["action_indices"], list(range(start, start + 24)))
                self.assertEqual(s["video_frame_indices"], list(range(start, start + 25, 2)))
        finally:
            repo.close()

    def test_single_and_dual_real_cpu_batches_and_distributed_validation(self):
        from thinker_wm.data_prepare.factory import create_training_data_pipeline
        from thinker_wm.data_prepare.finite_validation import FiniteValidationPipeline
        for config in self.configs[:2]:
            c = OmegaConf.load(config)
            c.system.ready_batch_queue_enabled = False
            starts = []
            for rank in range(4):
                pipeline = create_training_data_pipeline(c, rank=rank, world_size=4, seed=c.training.seed)
                try:
                    batch, _ = pipeline.next()
                    self.assertEqual(tuple(batch["action_sequence"].shape), (4, 24, c.common.action_dim))
                    self.assertEqual(tuple(batch["initial_state"].shape), (4, c.common.state_dim))
                    self.assertEqual(tuple(batch["first_frame"].shape), (4, 3, 480, 640))
                    self.assertEqual(tuple(batch["video_frames"].shape), (4, 12, 3, 480, 640))
                    starts.extend(int(indices[0]) for indices in batch["frame_indices"])
                    del batch
                finally:
                    pipeline.close()
            self.assertEqual(len(set(starts)), 16)
            ids = []
            for rank in range(4):
                val = FiniteValidationPipeline(c, rank=rank, world_size=4, seed=20261109)
                try:
                    ids.extend(val.global_ids)
                    if rank == 0:
                        b, _ = val.next()
                        self.assertGreater(float(b["initial_state"][0, 0]), 9)
                finally:
                    val.close()
            self.assertEqual(sorted(ids), list(range(8)))

    def test_preview_never_launches_or_creates_run(self):
        result = launch(self.configs[0], "smoke", smoke_steps=100)
        self.assertFalse(result["executed"])
        self.assertFalse(Path(result["run_directory"]).exists())
        self.assertIn("--nproc_per_node=4", result["command"])
        self.assertEqual(result["max_steps"], 100)

    def test_cross_task_resume_and_source_mix_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact task"):
            plan_launch(self.configs[0], "run", resume=self.root / "other_task" / "checkpoint_step_5000")
        c = yaml.safe_load(self.configs[0].read_text())
        c["dataset"]["sources"][0]["pool"]["path"] = str(self.release / TASKS[1].name / "train.ranges.sqlite")
        path = self.root / "bad.yaml"
        path.write_text(yaml.safe_dump(c))
        with self.assertRaisesRegex(ValueError, "pool mismatch"):
            plan_launch(path, "run")

    def test_explicit_same_run_resume_retains_optimizer_restore_command(self):
        _, run, _ = plan_launch(self.configs[0], "resume_test")
        checkpoint = run / "checkpoint_step_20000"
        checkpoint.mkdir(parents=True)
        write_json(checkpoint / ".motus_checkpoint_complete.json", dict(status="complete", step=20000))
        c, same_run, command = plan_launch(self.configs[0], "resume_test", resume=checkpoint, max_steps=30000)
        self.assertEqual(run, same_run)
        self.assertEqual(c["training"]["max_steps"], 30000)
        self.assertIn("--resume_checkpoint", command)
        self.assertNotIn("--finetune_checkpoint", command)

    def test_queue_is_ordered_independent_and_preview_only(self):
        results = launch_queue(list(reversed(self.configs)), "queue_preview")
        self.assertEqual([r["task"] for r in results], [t.name for t in TASKS])
        self.assertTrue(all(not r["executed"] and not Path(r["run_directory"]).exists() for r in results))
        with self.assertRaisesRegex(ValueError, "exactly six"):
            launch_queue(self.configs[:1], "queue_preview")


if __name__ == "__main__":
    unittest.main()
