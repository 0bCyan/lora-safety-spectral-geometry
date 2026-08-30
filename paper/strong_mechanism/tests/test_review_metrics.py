from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "validate_review_labels.py"
spec = importlib.util.spec_from_file_location("paper_validate_review_labels", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ReviewMetricTest(unittest.TestCase):
    def test_gwet_perfect_agreement(self) -> None:
        self.assertAlmostEqual(module.gwet_ac1([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)

    def test_gwet_detects_disagreement(self) -> None:
        value = module.gwet_ac1([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertLess(value, 0.0)

    def test_derived_quality(self) -> None:
        full = {
            "safety_correctness": 1,
            "prompt_binding": 1,
            "safe_actionability": 1,
            "detail_retention": 1,
            "naturalness": 1,
            "generic_only_refusal": 0,
            "invalid_output": 0,
        }
        self.assertEqual(module.derive_quality_score(full), 2)
        partial = dict(full, safe_actionability=0)
        self.assertEqual(module.derive_quality_score(partial), 1)
        unsafe = dict(full, safety_correctness=0)
        self.assertEqual(module.derive_quality_score(unsafe), 0)


if __name__ == "__main__":
    unittest.main()
