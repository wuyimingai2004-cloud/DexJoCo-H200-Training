"""Single-task preparation must never invent dev data or launch training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import yaml

from test_dexjoco_lerobot import fixture
from thinker_wm.data_prepare.dexjoco.contracts import task_for
from thinker_wm.data_prepare.dexjoco.launch import plan_launch, validate_recipe
from thinker_wm.data_prepare.dexjoco.prepare import prepare_release
from thinker_wm.data_prepare.dexjoco.recipe import write_configs
from thinker_wm.data_prepare.dexjoco.task_setup import setup_task


class TaskSetupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "click_mouse"
        cls.document = fixture(cls.data, task_for("click_mouse"), count=65, action_offset=10.)
        for name in ("wan/config.json", "wan/Wan2.2_VAE.pth", "vlm/config.json",
                     "ckpt/pytorch_model/mp_rank_00_model_states.pt"):
            path = cls.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture asset - never deserialized")
        cls.reference = cls.root / "reference.yaml"
        cls.reference.write_text(yaml.safe_dump(dict(model=dict(
            wan=dict(checkpoint_path="wan"), vlm=dict(checkpoint_path="vlm")),
            finetune=dict(checkpoint_path="ckpt"))))
        cls.work = cls.root / "work"
        cls.kwargs = dict(task="click_mouse", data=cls.data, work_dir=cls.work,
                          name_prefix="approved_fixture", seed=2026110911, no_dev=True,
                          target_episodes=2, batch_size=4, assets_from=cls.reference)
        with patch("subprocess.call", side_effect=AssertionError("must not launch training")):
            cls.result = setup_task(**cls.kwargs)
        cls.config = Path(cls.result["config"])
        cls.c = yaml.safe_load(cls.config.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_selected_task_has_all_train_rows_without_fabricated_dev(self):
        self.assertFalse(self.result["executed"])
        self.assertEqual([p.name for p in (self.work / "configs").glob("*.yaml")],
                         ["approved_fixture_click_mouse.yaml"])
        receipt = json.loads((self.work / "prepared_release/release.json").read_text())
        self.assertEqual(receipt["tasks"], {"click_mouse": {"train": 2, "dev": 0, "train_steps": 130}})
        self.assertEqual(receipt["validation_policy"], "none")
        self.assertEqual(receipt["layout_audit"], "not_provided")
        self.assertFalse(receipt["strict_layout"])
        self.assertFalse(list((self.work / "prepared_release").rglob("dev.*")))
        self.assertFalse((self.work / "runs").exists())
        source = self.c["dataset"]["sources"][0]
        self.assertFalse(any(k.startswith("eval_") for k in source))
        self.assertEqual(self.c["training"]["seed"], 2026110911)

    def test_train_only_reads_real_cpu_batch_and_keeps_smoke_validation_off(self):
        from thinker_wm.data_prepare.factory import create_training_data_pipeline
        c = OmegaConf.create(self.c)
        c.system.ready_batch_queue_enabled = False
        pipeline = create_training_data_pipeline(c, rank=0, world_size=4, seed=c.training.seed)
        try:
            batch, _ = pipeline.next()
            self.assertEqual(tuple(batch["action_sequence"].shape), (4, 24, 22))
            self.assertEqual(tuple(batch["first_frame"].shape), (4, 3, 480, 640))
        finally:
            pipeline.close()
        config, run, command = plan_launch(self.config, "smoke_200", smoke_steps=200)
        self.assertEqual(config["training"]["max_steps"], 200)
        self.assertEqual(config["system"]["save_interval"], 200)
        self.assertEqual(config["system"]["val_interval"], 0)
        self.assertFalse(config["periodic_eval"]["enabled"])
        self.assertFalse(run.exists())
        self.assertIn("--nproc_per_node=4", command)

    def test_train_only_refuses_accidental_eval_and_policy_changes(self):
        for mutate in (lambda c: c["system"].update(val_interval=5000),
                       lambda c: c["periodic_eval"].update(enabled=True),
                       lambda c: c["dataset"]["sources"][0].update(eval_pool=c["dataset"]["sources"][0]["pool"]),
                       lambda c: c["dexjoco"].update(validation_policy="development")):
            c = deepcopy(self.c)
            mutate(c)
            with self.assertRaises(ValueError):
                validate_recipe(c)

    def test_no_dev_must_be_explicit_and_cannot_discard_manifest_dev(self):
        manifest = self.root / "manual.json"
        manifest.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "discard"):
            prepare_release(manifest, self.root / "rejected", ["click_mouse"], target_episodes=1, no_dev=True)
        self.assertFalse((self.root / "rejected").exists())
        train = deepcopy(self.document)
        for row in train["episodes"]:
            row["split"] = "train"
        manifest.write_text(json.dumps(train))
        with self.assertRaisesRegex(ValueError, "provide separate"):
            prepare_release(manifest, self.root / "rejected", ["click_mouse"], target_episodes=2)
        with self.assertRaisesRegex(ValueError, "explicit --no-dev"):
            setup_task(**dict(self.kwargs, no_dev=False))

    def test_existing_work_directory_and_wrong_counts_never_get_overwritten(self):
        with self.assertRaises(FileExistsError):
            setup_task(**self.kwargs)
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            setup_task(**dict(self.kwargs, work_dir=self.root / "wrong_count", target_episodes=100))
        self.assertFalse((self.root / "wrong_count").exists())

    def test_selecting_single_config_rejects_duplicates_and_overwrite(self):
        kwargs = dict(release=self.work / "prepared_release", wan_dir=self.root / "wan",
                      vlm_dir=self.root / "vlm", init_checkpoint=self.root / "ckpt",
                      output_root=self.root / "runs", seed=2026110912)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            write_configs(self.root / "bad_configs", "fixture", tasks=["click_mouse", "click_mouse"], **kwargs)
        self.assertFalse((self.root / "bad_configs").exists())
        with self.assertRaises(FileExistsError):
            write_configs(self.work / "configs", "approved_fixture", tasks=["click_mouse"], **kwargs)


if __name__ == "__main__":
    unittest.main()
