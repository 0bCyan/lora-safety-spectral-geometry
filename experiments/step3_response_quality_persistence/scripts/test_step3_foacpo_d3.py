from __future__ import annotations

import unittest

import step3_foacpo_common as common


class FactorizedObjectiveTests(unittest.TestCase):
    def test_independent_pool_identity(self) -> None:
        self.assertAlmostEqual(
            common.factorized_objective([1.0, 3.0], [2.0, 4.0, 6.0], 0.25),
            0.25 * 2.0 + 0.75 * 4.0,
        )

    def test_factorized_objective_rejects_empty_pool(self) -> None:
        with self.assertRaises(common.FoacpoMathError):
            common.factorized_objective([], [1.0], 0.5)


class ProtectedBlendTests(unittest.TestCase):
    def test_full_behavior_when_all_certificates_pass(self) -> None:
        result = common.maximal_protected_blend(
            {"a": 2.0, "b": 1.0}, {"a": 3.0, "b": 0.95}, 0.9
        )
        self.assertEqual(result["lambda"], 1.0)
        self.assertGreaterEqual(result["minimum_retention"], 0.9)

    def test_closed_form_limiter_is_exact(self) -> None:
        result = common.maximal_protected_blend(
            {"risk": 2.0, "tarc": 4.0}, {"risk": -2.0, "tarc": 3.8}, 0.95
        )
        self.assertAlmostEqual(result["lambda"], 0.025)
        self.assertEqual(result["limiting_certificates"], ["risk"])
        self.assertAlmostEqual(result["retentions"]["risk"], 0.95)

    def test_zero_lambda_remains_feasible(self) -> None:
        result = common.maximal_protected_blend(
            {"risk": 1.0}, {"risk": -1e30}, 1.0
        )
        self.assertEqual(result["lambda"], 0.0)
        self.assertEqual(result["minimum_retention"], 1.0)

    def test_requires_positive_anchor(self) -> None:
        with self.assertRaises(common.FoacpoMathError):
            common.maximal_protected_blend({"risk": 0.0}, {"risk": 1.0}, 0.95)


if __name__ == "__main__":
    unittest.main()
