from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "aggregate_results.py"
spec = importlib.util.spec_from_file_location("paper_aggregate_results", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class AggregateHelperTest(unittest.TestCase):
    def test_exact_mcnemar(self) -> None:
        a = np.array([0, 0, 1, 1, 1])
        b = np.array([1, 0, 0, 1, 1])
        result = module.exact_mcnemar(a, b)
        self.assertEqual(result["a_only"], 1)
        self.assertEqual(result["b_only"], 1)
        self.assertEqual(result["discordant"], 2)
        self.assertAlmostEqual(result["p_value"], 1.0)

    def test_paired_bootstrap_sign(self) -> None:
        frame = pd.DataFrame({"a": [0, 0, 0, 0], "b": [1, 1, 0, 1]})
        estimate, lower, upper = module.paired_bootstrap_effect(
            frame,
            "a",
            "b",
            iterations=500,
            rng=np.random.default_rng(7),
            confidence=0.95,
        )
        self.assertAlmostEqual(estimate, 0.75)
        self.assertLessEqual(lower, estimate)
        self.assertGreaterEqual(upper, estimate)


if __name__ == "__main__":
    unittest.main()
