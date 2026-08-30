from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "validate_study.py"
CONFIG_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "configs" / "qwen3_4b_equal_dose_temporal_schedules.json"

spec = importlib.util.spec_from_file_location("paper_validate_study", VALIDATOR_PATH)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


class ScheduleInvariantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.arms = self.config["intervention_arms"]
        self.total_steps = 64

    def vector(self, name: str) -> list[float]:
        return validator.schedule_vector(self.arms[name], self.total_steps)

    def test_total_steps_and_checkpoint_rounding(self) -> None:
        total, checkpoints = validator.target_steps(self.config)
        self.assertEqual(total, 64)
        self.assertEqual(checkpoints, [0, 6, 16, 32, 48, 64])

    def test_exact_equal_dose(self) -> None:
        vectors = {name: self.vector(name) for name in self.arms}
        for values in vectors.values():
            self.assertAlmostEqual(sum(values), 10.45, places=12)
            self.assertAlmostEqual(sum(1.0 - value for value in values), 53.55, places=12)
            self.assertTrue(all(value > 0.0 for value in values))
        self.assertEqual(len({round(sum(values), 12) for values in vectors.values()}), 1)

    def test_temporal_shapes(self) -> None:
        front = self.vector("front_loaded_equal_dose")
        uniform = self.vector("uniform_equal_dose")
        back = self.vector("back_loaded_equal_dose")
        self.assertGreater(sum(front[:32]), sum(front[32:]))
        self.assertLess(sum(back[:32]), sum(back[32:]))
        self.assertAlmostEqual(max(uniform), min(uniform), places=15)
        self.assertTrue(all(abs(front[index] - back[-1 - index]) < 1e-15 for index in range(64)))

    def test_identical_anchor_source(self) -> None:
        sources = {arm["anchor_source_path"] for arm in self.arms.values()}
        self.assertEqual(len(sources), 1)


if __name__ == "__main__":
    unittest.main()
