from __future__ import annotations

import unittest
from unittest import mock

import torch

import step3_medscmcd_common as medsc


class GradientMeanTests(unittest.TestCase):
    def test_coordinatewise_mean_supports_arbitrary_unit_count(self) -> None:
        units = [
            {
                "a": {"p": torch.tensor([float(index), 2.0 * index])},
                "b": {"p": torch.tensor([3.0 * index, 4.0 * index])},
            }
            for index in range(1, 4)
        ]
        mean = medsc.mean_certificate_gradients(units)
        self.assertTrue(torch.equal(mean["a"]["p"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(mean["b"]["p"], torch.tensor([6.0, 8.0])))


class MemoryEfficientPathTests(unittest.TestCase):
    def test_shared_path_uses_one_map_call_per_fraction(self) -> None:
        baseline = {"p": torch.tensor([0.6, -0.2], dtype=torch.float64)}
        raw = {"p": torch.tensor([0.2, 0.0], dtype=torch.float64)}
        target = {"p": torch.tensor([0.0, 0.2], dtype=torch.float64)}
        panels = {
            "one": {
                "a": {"p": torch.tensor([1.0, 0.0], dtype=torch.float64)},
                "b": {"p": torch.tensor([0.0, 1.0], dtype=torch.float64)},
            },
            "two": {
                "a": {"p": torch.tensor([1.0, 1.0], dtype=torch.float64)},
                "b": {"p": torch.tensor([1.0, -1.0], dtype=torch.float64)},
            },
        }
        with mock.patch.object(
            medsc.frsc,
            "realized_displacement",
            wraps=medsc.frsc.realized_displacement,
        ) as realized:
            scored = medsc.score_path_panels_memory_efficient(
                baseline,
                raw,
                target,
                panels,
                [0.0, 0.5, 1.0],
                0.2,
                0.2,
                1.0,
                1e-2,
                1.0,
            )
        self.assertEqual(realized.call_count, 3)
        for index in range(3):
            self.assertEqual(
                scored["one"][index]["displacement_norm_fraction"],
                scored["two"][index]["displacement_norm_fraction"],
            )


class MinimumEffectiveDoseTests(unittest.TestCase):
    @staticmethod
    def record(fraction: float, protected: float, target: float) -> dict:
        scores = {"protected": protected, "target": target}
        return {
            "fraction": fraction,
            "scores": scores,
            "minimum_score": min(scores.values()),
            "mean_score": sum(scores.values()) / 2,
        }

    def test_selects_smallest_candidate_reaching_effect_floor(self) -> None:
        panel = [
            self.record(0.0, 1.0, 0.5),
            self.record(0.25, 0.98, 0.501),
            self.record(0.50, 0.96, 0.503),
            self.record(1.0, 0.90, 0.51),
        ]
        selected, audit = medsc.select_minimum_effective_candidate(
            panel,
            [panel, panel],
            ["protected"],
            ["target"],
            retention_floor=0.95,
            target_gain_floor=0.002,
            target_score_floor=0.1,
        )
        self.assertEqual(selected, 2)
        self.assertEqual(audit["eligible_candidates"], 1)
        self.assertEqual(audit["selected"]["fraction"], 0.5)

    def test_falls_back_to_raw_without_claiming_eligibility(self) -> None:
        panel = [
            self.record(0.0, 1.0, 0.5),
            self.record(1.0, 0.90, 0.51),
        ]
        selected, audit = medsc.select_minimum_effective_candidate(
            panel,
            [panel, panel],
            ["protected"],
            ["target"],
            0.95,
            0.002,
            0.1,
        )
        self.assertEqual(selected, 0)
        self.assertFalse(audit["selected"]["eligible"])


class SplitConformalTests(unittest.TestCase):
    def test_joint_nonconformity_is_worst_relative_retention_shortfall(self) -> None:
        value, residuals = medsc.joint_protected_nonconformity(
            {"a": 2.0, "b": 4.0},
            {"a": 1.96, "b": 3.76},
            ["a", "b"],
            0.95,
        )
        self.assertAlmostEqual(residuals["a"], -0.03)
        self.assertAlmostEqual(residuals["b"], 0.01)
        self.assertAlmostEqual(value, 0.01)

    def test_twenty_calibration_units_give_registered_coverage(self) -> None:
        result = medsc.split_conformal_upper_bound(
            [-0.03 + index * 0.001 for index in range(20)], alpha=0.05
        )
        self.assertEqual(result["order_statistic_rank"], 20)
        self.assertAlmostEqual(
            result["finite_sample_marginal_coverage_lower_bound"], 20 / 21
        )
        self.assertAlmostEqual(result["threshold"], -0.011)
        self.assertTrue(result["all_calibration_units_protected"])

    def test_too_few_units_reject_requested_coverage(self) -> None:
        with self.assertRaises(ValueError):
            medsc.split_conformal_upper_bound([-0.1] * 10, alpha=0.05)

    def test_accept_or_fallback_policy_reports_only_marginal_guarantee(self) -> None:
        accepted = medsc.split_conformal_accept_or_fallback(
            [-0.03 + index * 0.001 for index in range(20)],
            alpha=0.05,
            acceptance_threshold=0.0,
            fallback_nonconformity=-0.05,
        )
        self.assertTrue(accepted["candidate_accepted"])
        self.assertEqual(accepted["policy_action"], "candidate")
        self.assertAlmostEqual(
            accepted["finite_sample_marginal_policy_failure_upper_bound"], 1 / 21
        )
        self.assertFalse(
            accepted["conditional_on_candidate_acceptance_coverage_claimed"]
        )

        rejected = medsc.split_conformal_accept_or_fallback(
            [-0.03] * 19 + [0.01],
            alpha=0.05,
            acceptance_threshold=0.0,
            fallback_nonconformity=-0.05,
        )
        self.assertFalse(rejected["candidate_accepted"])
        self.assertEqual(rejected["policy_action"], "fallback")

    def test_accept_or_fallback_rejects_unsafe_fallback(self) -> None:
        with self.assertRaises(ValueError):
            medsc.split_conformal_accept_or_fallback(
                [-0.1] * 20,
                alpha=0.05,
                acceptance_threshold=0.0,
                fallback_nonconformity=0.01,
            )


if __name__ == "__main__":
    unittest.main()
