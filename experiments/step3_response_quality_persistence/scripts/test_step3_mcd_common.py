from __future__ import annotations

import math
import unittest

import torch

import step3_mcd_common as mcd


class SimplexDualTests(unittest.TestCase):
    def test_two_orthogonal_certificates_have_equal_dual_weights(self) -> None:
        weights, audit = mcd.minimum_norm_simplex_weights(
            [[1.0, 0.0], [0.0, 1.0]]
        )
        self.assertAlmostEqual(weights[0], 0.5, places=8)
        self.assertAlmostEqual(weights[1], 0.5, places=8)
        self.assertAlmostEqual(audit["dual_squared_margin"], 0.5, places=8)
        self.assertLess(audit["kkt_residual"], 1e-8)

    def test_redundant_easy_certificate_is_inactive(self) -> None:
        root = 1.0 / math.sqrt(2.0)
        gram = [
            [1.0, 0.0, root],
            [0.0, 1.0, root],
            [root, root, 1.0],
        ]
        weights, _ = mcd.minimum_norm_simplex_weights(gram)
        self.assertAlmostEqual(weights[0], 0.5, places=7)
        self.assertAlmostEqual(weights[1], 0.5, places=7)
        self.assertAlmostEqual(weights[2], 0.0, places=7)


class CommonDirectionTests(unittest.TestCase):
    def test_primal_direction_matches_bisector_and_budget(self) -> None:
        gradients = {
            "route": {"p": torch.tensor([2.0, 0.0])},
            "continuation": {"p": torch.tensor([0.0, 3.0])},
        }
        direction, audit = mcd.minimax_common_direction(gradients, 2.0)
        expected = torch.tensor([math.sqrt(2.0), math.sqrt(2.0)])
        self.assertTrue(torch.allclose(direction["p"], expected, atol=1e-6))
        self.assertAlmostEqual(audit["dual_margin"], 1.0 / math.sqrt(2.0), places=6)
        self.assertLess(audit["primal_dual_gap"], 1e-6)
        self.assertAlmostEqual(audit["final_direction_norm"], 2.0, places=6)

    def test_direction_is_invariant_to_positive_objective_scaling(self) -> None:
        first, _ = mcd.minimax_common_direction(
            {
                "a": {"p": torch.tensor([1.0, 0.0])},
                "b": {"p": torch.tensor([0.0, 1.0])},
            },
            1.0,
        )
        second, _ = mcd.minimax_common_direction(
            {
                "a": {"p": torch.tensor([100.0, 0.0])},
                "b": {"p": torch.tensor([0.0, 0.01])},
            },
            1.0,
        )
        self.assertTrue(torch.allclose(first["p"], second["p"], atol=1e-6))

    def test_opposing_certificates_report_no_strict_common_descent(self) -> None:
        with self.assertRaisesRegex(ValueError, "no strict common descent"):
            mcd.minimax_common_direction(
                {
                    "a": {"p": torch.tensor([1.0, 0.0])},
                    "b": {"p": torch.tensor([-1.0, 0.0])},
                },
                1.0,
            )


class RouteTests(unittest.TestCase):
    def test_weak_effect_has_distinct_stop(self) -> None:
        gates = {
            "integrity": True,
            "gradient_valid": True,
            "budget_matched": True,
            "vram": True,
            "state_restored": True,
            "dual_certificate": True,
            "common_transfer": True,
            "route_effect": True,
            "continuation_effect": True,
            "branch_effect": True,
            "effect_not_weak": False,
        }
        self.assertEqual(
            mcd.select_route(gates), "STOP_MCD_ETARC_P0_EFFECT_TOO_WEAK"
        )


if __name__ == "__main__":
    unittest.main()
