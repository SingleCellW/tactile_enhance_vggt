"""CPU-only launcher regressions: environment entrypoint and v2 data contract."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ablation_launcher_under_test", ROOT / "scripts/run_ablation_suite.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class AblationLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.output = self.root / "suite"
        self.python = self.root / ".venv/bin/python"
        self.python.parent.mkdir(parents=True)
        # A real executable behind a venv-style symlink reproduces the failure:
        # resolving this path discards the directory that identifies the venv.
        self.python.symlink_to(Path(sys.executable).resolve())
        self.manifest = {
            "complete": True, "historical_source_tactile_superseded": True,
            "native_tactile_report": {"complete": True, "velocity_dt": .01},
            "configuration": {
                "rgb_alignment": {"verified": True, "rgb_lag_frames": 1},
                "action_label_repair": {"action_target_offset_frames": 0},
            },
            "split": {"train": [0], "val": [1]},
        }
        self.write_manifest()
        for name, array in {
            "episode_index": np.repeat([0, 1], 8),
            "frame_index": np.tile(np.arange(8), 2),
            "obs_frame_indices": np.array([3, 11]),
            "rgb_source_frame_indices": np.array([4, 12]),
            "completed": np.ones(2, dtype=bool),
            "state": np.zeros((16, 22), dtype=np.float32),
            "action": np.zeros((16, 22), dtype=np.float32),
            "tactile": np.zeros((16, 44), dtype=np.float32),
            "points": np.zeros((2, 556, 8), dtype=np.float32),
        }.items():
            np.save(self.cache / f"{name}.npy", array)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "experiment": "v2_physics_aligned_current_command", "seed": 43,
            "batch_size": 1, "reference_steps": 160000, "reference_frames": 20212,
            "reference_batch_size": 64, "amp": "bfloat16", "learning_rate": 1e-5,
            "eval_every": 500, "save_every": 500, "eval_batches": 16,
            "policy": {"mode": "anchored", "chunk_size": 16, "n_action_steps": 8},
        }))
        self.args = argparse.Namespace(
            cache=str(self.cache), output_root=str(self.output), config=str(self.config),
            python=str(self.python), seeds=None, steps=6000, batch_size=None, amp=None,
            cpu_threads=1, alignment_report=None,
        )

    def write_manifest(self):
        (self.cache / "manifest.json").write_text(json.dumps(self.manifest))

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            launcher.prepare(self.args)

    def test_plan_preserves_virtualenv_python_symlink_for_every_mode(self):
        self.assertTrue(self.python.is_symlink())
        self.assertNotEqual(self.python.absolute(), self.python.resolve())
        with patch.object(launcher.subprocess, "run") as process:
            self.prepare()
            process.assert_not_called()
        plan = json.loads((self.output / "suite_plan.json").read_text())
        self.assertEqual(plan["modes"], ["anchored", "unanchored", "no_tactile"])
        self.assertEqual(plan["steps_per_run"], 6000)
        self.assertEqual(plan["native_velocity_dt"], .01)
        self.assertEqual(plan["action_label_repair"]["action_target_offset_frames"], 0)
        for job in plan["jobs"]:
            self.assertEqual(job["command"][0], str(self.python.absolute()))
            self.assertNotEqual(job["command"][0], str(self.python.resolve()))
            config = json.loads(Path(job["config"]).read_text())
            self.assertEqual(config["policy"]["mode"], job["mode"])
            self.assertEqual(config["eval_every"], config["save_every"])
            self.assertEqual(config["save_every"], 500)

    def test_v2_rejects_missing_or_historical_next_frame_action_labels(self):
        for action_repair in (None, {"action_target_offset_frames": 1}):
            with self.subTest(action_repair=action_repair):
                self.manifest["configuration"].pop("action_label_repair", None)
                if action_repair is not None:
                    self.manifest["configuration"]["action_label_repair"] = action_repair
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "V2 requires"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_v2_rejects_wall_clock_velocity_even_with_correct_actions(self):
        for velocity_dt in (None, .05):
            with self.subTest(velocity_dt=velocity_dt):
                self.manifest["native_tactile_report"]["velocity_dt"] = velocity_dt
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "V2 requires"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_v2_rejects_old_action_and_wall_clock_cache_together(self):
        self.manifest["configuration"]["action_label_repair"] = {"action_target_offset_frames": 1}
        self.manifest["native_tactile_report"]["velocity_dt"] = None
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "V2 requires"):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
