from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "run_registered_gradient_audit.py"
spec = importlib.util.spec_from_file_location("paper_registered_gradient_audit", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class RegisteredGradientScheduleTest(unittest.TestCase):
    def load(self, name: str) -> dict:
        path = REPO_ROOT / "paper" / "strong_mechanism" / "configs" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_equal_dose_late_checkpoint_weights(self) -> None:
        config = self.load("qwen3_4b_equal_dose_temporal_schedules.json")
        expected = {
            "front_loaded_equal_dose": 0.1265625,
            "uniform_equal_dose": 0.16328125,
            "back_loaded_equal_dose": 0.2,
        }
        for arm_name, expected_weight in expected.items():
            arm = config["intervention_arms"][arm_name]
            weight48, step48 = module.checkpoint_schedule_weight(arm, 48, 64)
            weight64, step64 = module.checkpoint_schedule_weight(arm, 64, 64)
            self.assertEqual(step48, 48)
            self.assertEqual(step64, 63)
            self.assertAlmostEqual(weight48, expected_weight, places=12)
            self.assertAlmostEqual(weight64, expected_weight, places=12)

    def test_decay_final_weight_matches_last_applied_discrete_step(self) -> None:
        config = self.load("qwen3_4b_context_replication.json")
        arm = config["intervention_arms"]["natural_diverse_10"]
        weight, schedule_step = module.checkpoint_schedule_weight(arm, 64, 64)
        self.assertEqual(schedule_step, 63)
        self.assertAlmostEqual(weight, 0.10208333333333333, places=12)
        self.assertNotAlmostEqual(weight, 0.1, places=12)

    def test_schedule_hash_input_is_deterministic(self) -> None:
        config = self.load("qwen3_4b_equal_dose_temporal_schedules.json")
        arm = config["intervention_arms"]["uniform_equal_dose"]
        first = module.schedule_vector(arm, 64)
        second = module.schedule_vector(arm, 64)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first), 10.45, places=12)


if __name__ == "__main__":
    unittest.main()
