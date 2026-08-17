from __future__ import annotations

import unittest

import torch

import step3_frsc_common as frsc
import step3_ramp_common as gradient_math


class ExactSubsetSearchTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[dict, dict]:
        baseline = {"p": torch.tensor([0.6, -0.4, 0.2], dtype=torch.float64)}
        certificates = {
            "risk": {"p": torch.tensor([1.0, 0.1, 0.0], dtype=torch.float64)},
            "route": {"p": torch.tensor([0.1, 1.0, 0.1], dtype=torch.float64)},
            "branch": {"p": torch.tensor([0.2, 0.1, 1.0], dtype=torch.float64)},
        }
        return baseline, certificates

    def test_exhaustive_family_contains_all_nonempty_subsets(self) -> None:
        baseline, certificates = self.fixture()
        selected, full, audit = frsc.enumerate_finite_radius_subsets(
            baseline,
            certificates,
            budget_norm=0.2,
            auxiliary_scale=0.2,
            max_norm=1.0,
            adam_epsilon=1e-2,
        )
        self.assertEqual(audit["attempted_subsets"], 7)
        self.assertEqual(
            audit["feasible_subsets"] + audit["infeasible_subsets"], 7
        )
        self.assertEqual(
            set(audit["full_certificate_control"]["subset"]), set(certificates)
        )
        self.assertGreaterEqual(
            audit["selected"]["minimum_exact_score"],
            audit["full_certificate_control"]["minimum_exact_score"],
        )
        self.assertAlmostEqual(gradient_math.gradient_norm(selected), 0.2, places=9)
        self.assertAlmostEqual(gradient_math.gradient_norm(full), 0.2, places=9)

    def test_search_is_deterministic(self) -> None:
        baseline, certificates = self.fixture()
        first = frsc.enumerate_finite_radius_subsets(
            baseline, certificates, 0.2, 0.2, 1.0, 1e-2
        )[2]
        second = frsc.enumerate_finite_radius_subsets(
            baseline, certificates, 0.2, 0.2, 1.0, 1e-2
        )[2]
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["top_candidates"], second["top_candidates"])

    def test_displacement_scores_use_fixed_baseline_norm(self) -> None:
        _, certificates = self.fixture()
        displacement = {
            "p": torch.tensor([0.4, 0.3, 0.2], dtype=torch.float64)
        }
        result = frsc.displacement_scores(
            certificates, displacement, reference_norm=2.0
        )
        risk_unit = certificates["risk"]["p"] / certificates["risk"]["p"].norm()
        expected = float(torch.dot(risk_unit, displacement["p"]) / 2.0)
        self.assertAlmostEqual(result["scores"]["risk"], expected, places=12)


class FiniteStepMetricsTests(unittest.TestCase):
    def test_first_order_and_actual_reductions_are_kept_separate(self) -> None:
        initial = {"risk": 2.0, "route": 1.0}
        final = {"risk": 1.8, "route": 0.9}
        gradients = {
            "risk": {"p": torch.tensor([2.0, 0.0], dtype=torch.float64)},
            "route": {"p": torch.tensor([0.0, 1.0], dtype=torch.float64)},
        }
        displacement = {
            "p": torch.tensor([1.0, 1.0], dtype=torch.float64)
        }
        result = frsc.finite_step_loss_metrics(
            initial, final, gradients, displacement, learning_rate=0.1
        )
        self.assertAlmostEqual(
            result["objectives"]["risk"]["predicted_first_order_reduction"],
            0.2,
        )
        self.assertAlmostEqual(
            result["objectives"]["risk"]["actual_loss_reduction"], 0.2
        )
        self.assertAlmostEqual(
            result["objectives"]["route"]["actual_to_predicted_ratio"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
