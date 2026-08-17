from __future__ import annotations

import unittest

import torch

import step3_qetarc_p0 as p0


class DirectionTests(unittest.TestCase):
    def test_all_directions_share_the_registered_budget(self) -> None:
        calibration = {
            "uniform": {"p": torch.tensor([3.0, 4.0])},
            "ramp": {"p": torch.tensor([1.0, 0.0])},
            "tarc": {"p": torch.tensor([0.0, 2.0])},
            "etarc": {"p": torch.tensor([1.0, 0.0])},
            "continuation": {"p": torch.tensor([0.0, 1.0])},
        }
        directions, coefficients, geometry = p0.scaled_directions(
            calibration, 0.25, 0.5
        )
        for direction in directions.values():
            self.assertLess(
                abs(p0.gradient_math.gradient_norm(direction) - 1.25), 1e-6
            )
        self.assertAlmostEqual(geometry["achieved_continuation_cosine"], 0.5)
        self.assertLess(coefficients["maximum_relative_budget_error"], 1e-6)


class PairSummaryTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, object]]:
        rows = []
        for objective in ("etarc", "continuation"):
            for index in range(4):
                rows.append(
                    {
                        "split": "validation",
                        "objective": objective,
                        "directional": {
                            "tarc": {"dot": 1.0, "cosine": 0.1},
                            "qetarc": {
                                "dot": 2.0 if objective == "continuation" else 3.0,
                                "cosine": 0.5,
                            },
                        },
                    }
                )
        return rows

    def test_pair_summary_preserves_paired_advantage(self) -> None:
        result = p0.summarize_pair_directions(
            self.rows(), ("tarc", "qetarc"), 100, 5
        )
        bootstrap = result["qetarc_continuation_advantage_over_tarc"]["bootstrap"]
        self.assertEqual(bootstrap["mean"], 1.0)
        self.assertEqual(bootstrap["ci95_lower"], 1.0)
        self.assertEqual(result["candidates"]["qetarc"]["route_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
