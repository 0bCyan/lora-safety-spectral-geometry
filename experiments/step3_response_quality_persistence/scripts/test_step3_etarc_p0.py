from __future__ import annotations

import unittest

import torch

import step3_etarc_p0 as p0


class SplitTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, str]]:
        rows = []
        for prompt in range(40):
            category = f"c{prompt % 8}"
            sample_id = f"p{prompt:02d}"
            for target in range(2):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target_id": f"{sample_id}t{target}",
                        "category": category,
                    }
                )
        return rows

    def test_split_is_deterministic_disjoint_and_exhaustive(self) -> None:
        first = p0.select_calibration_and_remainder(self.rows(), 17)
        second = p0.select_calibration_and_remainder(self.rows(), 17)
        self.assertEqual(first, second)
        self.assertEqual(len(first["calibration"]), 8)
        self.assertEqual(len(first["validation"]), 32)
        self.assertFalse(set(first["calibration"]).intersection(first["validation"]))
        self.assertEqual(len(set(first["calibration"] + first["validation"])), 40)


class DirectionTests(unittest.TestCase):
    def test_complete_path_objective_uses_exact_route_competitors(self) -> None:
        response_logits = [
            torch.tensor([[0.0, 5.0, 1.0], [0.0, 1.0, 5.0]]),
            torch.tensor([[0.0, 4.0, 1.0], [0.0, 4.0, 1.0]]),
        ]
        response_targets = [torch.tensor([1, 2]), torch.tensor([1, 1])]
        pair = {
            "paths": [[1, 2], [1, 1]],
            "features": [{"route_depth": 2}, {"route_depth": 2}],
        }
        loss, metrics = p0._etarc_path_certificate(
            response_logits, response_targets, pair, 0.1
        )
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(metrics["complete_route_certified"])
        self.assertTrue(metrics["winning_path_all_token_margins_positive"])
        self.assertEqual(metrics["margin_terms"], 4)

    def test_scaled_directions_have_equal_norm(self) -> None:
        calibration = {
            "uniform": {"p": torch.tensor([3.0, 4.0])},
            "ramp": {"p": torch.tensor([1.0, 0.0])},
            "tarc": {"p": torch.tensor([0.0, 2.0])},
            "etarc": {"p": torch.tensor([1.0, -1.0])},
            "continuation": {"p": torch.tensor([0.0, 1.0])},
        }
        directions, coefficients, projection = p0.scaled_directions(
            calibration, 0.25
        )
        self.assertTrue(projection["projection_active"])
        for direction in directions.values():
            self.assertAlmostEqual(p0.gradient_math.gradient_norm(direction), 1.25)
        self.assertLess(coefficients["maximum_relative_budget_error"], 1e-6)


class StatisticalTests(unittest.TestCase):
    def test_exact_sign_test_ignores_ties(self) -> None:
        result = p0.exact_sign_test([1.0, 2.0, -1.0, 0.0])
        self.assertEqual(result["positive"], 2)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["exact_two_sided_p"], 1.0)

    def test_bootstrap_is_deterministic(self) -> None:
        first = p0.paired_bootstrap([1.0, 2.0, 3.0], 100, 9)
        second = p0.paired_bootstrap([1.0, 2.0, 3.0], 100, 9)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
