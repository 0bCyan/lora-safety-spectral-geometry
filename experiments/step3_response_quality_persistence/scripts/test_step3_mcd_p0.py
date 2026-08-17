from __future__ import annotations

import unittest

import torch

import step3_mcd_p0 as p0


class DirectionBudgetTests(unittest.TestCase):
    @staticmethod
    def gradients() -> dict[str, dict[str, torch.Tensor]]:
        return {
            "uniform": {"p": torch.tensor([3.0, 4.0, 0.0])},
            "ramp": {"p": torch.tensor([1.0, 1.0, 0.0])},
            "tarc": {"p": torch.tensor([1.0, 0.0, 0.0])},
            "etarc": {"p": torch.tensor([1.0, 0.2, 0.0])},
            "continuation": {"p": torch.tensor([0.2, 1.0, 0.0])},
            "branch_entropy": {"p": torch.tensor([0.5, 0.5, 0.5])},
            "branch_negative_gap": {"p": torch.tensor([0.6, 0.4, 0.5])},
        }

    def test_all_controls_and_mcd_share_equal_norm_budget(self) -> None:
        directions, budget, quota, dual = p0.scaled_directions(
            self.gradients(), 1.0, 0.5
        )
        self.assertEqual(set(directions), set(p0.CANDIDATES))
        for direction in directions.values():
            self.assertAlmostEqual(
                p0.gradient_math.gradient_norm(direction), 5.0, places=5
            )
        self.assertLess(budget["maximum_relative_budget_error"], 1e-6)
        self.assertGreaterEqual(quota["achieved_continuation_cosine"], 0.5 - 1e-6)
        self.assertGreater(dual["dual_margin"], 0.0)


class PairSummaryTests(unittest.TestCase):
    def test_mcd_advantage_is_computed_on_paired_rows(self) -> None:
        rows = []
        for objective in p0.VALIDATION_OBJECTIVES:
            for index in range(4):
                rows.append(
                    {
                        "objective": objective,
                        "sample_id": str(index),
                        "directional": {
                            candidate: {
                                "dot": 3.0
                                if candidate == "mcd_etarc"
                                else (2.0 if candidate == "qetarc" else 1.0),
                                "cosine": 0.5,
                            }
                            for candidate in p0.CANDIDATES
                        },
                    }
                )
        protocol = {
            "statistics": {"bootstrap_samples": 100, "bootstrap_seed": 7}
        }
        result = p0.paired_summary(rows, protocol)
        advantage = result["mcd_continuation_advantage_over_qetarc"]
        self.assertEqual(advantage["bootstrap"]["mean"], 1.0)
        self.assertEqual(advantage["positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
