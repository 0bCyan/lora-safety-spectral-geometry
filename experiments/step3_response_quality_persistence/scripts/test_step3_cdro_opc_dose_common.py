from __future__ import annotations

import math
import unittest

import step3_cdro_opc_dose_common as dose


class UnitDoseTests(unittest.TestCase):
    def test_exact_limit_handles_negative_fallback(self) -> None:
        result = dose.unit_safe_dose(
            {"risk": -0.1, "tarc": 0.01},
            {"risk": -0.12, "tarc": 0.02},
            ["risk", "tarc"],
            allowed_relative_drop=0.04,
            score_scale_floor=0.02,
        )
        self.assertAlmostEqual(result["safe_dose"], 0.2)
        self.assertEqual(result["limiting_certificates"], ["risk"])
        self.assertEqual(result["per_certificate_safe_doses"]["tarc"], 1.0)

    def test_all_improving_certificates_allow_full_dose(self) -> None:
        result = dose.unit_safe_dose(
            {"a": -0.1, "b": 0.1},
            {"a": -0.05, "b": 0.2},
            ["a", "b"],
            0.04,
            0.02,
        )
        self.assertEqual(result["safe_dose"], 1.0)


class ConformalDoseTests(unittest.TestCase):
    def test_twenty_units_use_minimum_for_20_over_21_bound(self) -> None:
        fallback = [{"risk": 1.0} for _ in range(20)]
        endpoint = [{"risk": 1.1} for _ in range(19)] + [{"risk": 0.2}]
        result = dose.conformal_safe_dose(
            fallback,
            endpoint,
            ["risk"],
            allowed_relative_drop=0.04,
            score_scale_floor=0.02,
            alpha=0.05,
        )
        self.assertEqual(result["lower_order_statistic_rank"], 1)
        self.assertAlmostEqual(result["selected_safe_dose"], 0.05)
        self.assertAlmostEqual(
            result["finite_sample_marginal_protection_lower_bound"], 20 / 21
        )
        self.assertFalse(result["conditional_on_selected_dose_coverage_claimed"])

    def test_too_few_units_reject_requested_alpha(self) -> None:
        with self.assertRaises(dose.ConformalDoseError):
            dose.conformal_safe_dose(
                [{"a": 1.0}] * 10,
                [{"a": 1.0}] * 10,
                ["a"],
                0.04,
                0.02,
                0.05,
            )

    def test_second_order_statistic_at_thirty_nine_units(self) -> None:
        fallback = [{"a": 1.0} for _ in range(39)]
        endpoint = [{"a": 0.0}, {"a": 0.5}] + [{"a": 2.0}] * 37
        result = dose.conformal_safe_dose(
            fallback, endpoint, ["a"], 0.04, 0.02, 0.05
        )
        self.assertEqual(result["lower_order_statistic_rank"], 2)
        self.assertAlmostEqual(result["selected_safe_dose"], 0.08)
        self.assertAlmostEqual(
            result["finite_sample_marginal_protection_lower_bound"], 0.95
        )


class CosineInterpolationTests(unittest.TestCase):
    def test_matches_direct_two_dimensional_geometry(self) -> None:
        fallback = [1.0, 0.0]
        endpoint = [0.0, 2.0]
        gradient = [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
        fallback_cosine = gradient[0]
        endpoint_cosine = gradient[1]
        actual = dose.interpolate_cosine(
            fallback_cosine, endpoint_cosine, fallback, endpoint, 0.25
        )
        mixed = [0.75, 0.5]
        expected = sum(a * b for a, b in zip(gradient, mixed)) / math.sqrt(
            sum(value * value for value in mixed)
        )
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
