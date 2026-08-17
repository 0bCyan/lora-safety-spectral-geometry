from __future__ import annotations

import math
import unittest

import torch

import step3_qetarc_common as qetarc


class QuotaGeometryTests(unittest.TestCase):
    def test_active_solution_has_exact_budget_and_quota(self) -> None:
        route = {"p": torch.tensor([1.0, 0.0])}
        continuation = {"p": torch.tensor([0.0, 1.0])}
        direction, audit = qetarc.quota_constrained_direction(
            route, continuation, 0.5, 2.0
        )
        expected = torch.tensor([math.sqrt(3.0), 1.0])
        self.assertTrue(torch.allclose(direction["p"], expected, atol=1e-6))
        self.assertAlmostEqual(audit["final_direction_norm"], 2.0, places=6)
        self.assertAlmostEqual(audit["achieved_continuation_cosine"], 0.5, places=6)
        self.assertAlmostEqual(
            audit["achieved_route_cosine"], math.sqrt(3.0) / 2.0, places=6
        )

    def test_feasible_route_direction_is_only_normalized(self) -> None:
        route = {"p": torch.tensor([1.0, 1.0])}
        continuation = {"p": torch.tensor([0.0, 1.0])}
        direction, audit = qetarc.quota_constrained_direction(
            route, continuation, 0.5, 3.0
        )
        self.assertFalse(audit["quota_active"])
        self.assertGreater(audit["achieved_continuation_cosine"], 0.7)
        self.assertGreater(audit["achieved_route_cosine"], 0.999999)
        self.assertLess(
            abs(qetarc.etarc_geometry.gradient_norm(direction) - 3.0), 1e-6
        )

    def test_closed_form_dominates_all_feasible_two_dimensional_angles(self) -> None:
        route = {"p": torch.tensor([1.0, 0.0])}
        continuation = {"p": torch.tensor([0.0, 1.0])}
        _, audit = qetarc.quota_constrained_direction(
            route, continuation, 0.5, 1.0
        )
        feasible_route_cosines = []
        for degree in range(31, 150):
            angle = math.radians(degree)
            candidate = torch.tensor([math.cos(angle), math.sin(angle)])
            if float(candidate[1]) >= 0.5:
                feasible_route_cosines.append(float(candidate[0]))
        self.assertGreaterEqual(
            audit["achieved_route_cosine"], max(feasible_route_cosines) - 1e-6
        )

    def test_first_order_continuation_lower_bound_is_realized(self) -> None:
        route = {"p": torch.tensor([1.0, 0.0])}
        continuation = {"p": torch.tensor([0.0, 4.0])}
        _, audit = qetarc.quota_constrained_direction(
            route, continuation, 0.5, 2.0
        )
        self.assertGreaterEqual(
            audit["continuation_first_order_dot"] + 1e-6,
            audit["continuation_dot_lower_bound"],
        )


class DecisionTests(unittest.TestCase):
    def test_weak_effect_stops(self) -> None:
        gates = {
            "integrity": True,
            "gradient_valid": True,
            "budget_matched": True,
            "vram": True,
            "state_restored": True,
            "certificate_structure": True,
            "quota_geometry": True,
            "route_transfer": True,
            "continuation_transfer": True,
            "branch_mechanism": True,
            "effect_not_weak": False,
        }
        self.assertEqual(
            qetarc.select_route(gates), "STOP_QETARC_P0_EFFECT_TOO_WEAK"
        )


if __name__ == "__main__":
    unittest.main()
