from __future__ import annotations

import math
import unittest

import torch

import step3_cspr_common as cspr


class CsprProjectionTests(unittest.TestCase):
    def test_inactive_projection_is_preconditioned_preference_descent(self) -> None:
        g_p = torch.tensor([1.0, -2.0], dtype=torch.float64)
        g_s = torch.tensor([1.0, 1.0], dtype=torch.float64)
        diagonal = torch.tensor([2.0, 0.5], dtype=torch.float64)
        result = cspr.project_preconditioned_preference_direction(g_p, g_s, diagonal)
        self.assertFalse(result.active)
        torch.testing.assert_close(result.direction, -diagonal * g_p)
        self.assertLess(result.preference_directional_derivative, 0.0)
        self.assertLessEqual(result.safety_directional_derivative, 0.0)

    def test_active_projection_satisfies_kkt_and_both_first_order_conditions(self) -> None:
        g_p = torch.tensor([1.0, 0.0], dtype=torch.float64)
        g_s = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        diagonal = torch.tensor([2.0, 0.5], dtype=torch.float64)
        result = cspr.project_preconditioned_preference_direction(g_p, g_s, diagonal)
        self.assertTrue(result.active)
        self.assertAlmostEqual(result.safety_directional_derivative, 0.0, places=12)
        self.assertLess(result.preference_directional_derivative, 0.0)
        self.assertLess(result.kkt_stationarity_norm, 1e-12)
        self.assertLess(result.kkt_complementarity_abs, 1e-12)

    def test_active_projection_is_minimum_metric_correction(self) -> None:
        g_p = torch.tensor([1.0, 0.25, -0.5], dtype=torch.float64)
        g_s = torch.tensor([-1.0, 0.5, 0.25], dtype=torch.float64)
        diagonal = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        result = cspr.project_preconditioned_preference_direction(g_p, g_s, diagonal)
        self.assertTrue(result.active)
        correction = result.direction - result.unconstrained_direction
        optimum_distance = float(torch.dot(correction / diagonal, correction))
        tangent = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float64)
        self.assertAlmostEqual(float(torch.dot(g_s, tangent)), 0.0, places=12)
        for scale in (-2.0, -0.5, 0.5, 2.0):
            competitor = result.direction + scale * tangent
            competitor_correction = competitor - result.unconstrained_direction
            competitor_distance = float(
                torch.dot(competitor_correction / diagonal, competitor_correction)
            )
            self.assertGreaterEqual(competitor_distance + 1e-12, optimum_distance)

    def test_zero_safety_gradient_needs_no_projection(self) -> None:
        result = cspr.project_preconditioned_preference_direction(
            torch.tensor([1.0, -1.0]), torch.zeros(2), torch.ones(2)
        )
        self.assertFalse(result.active)
        self.assertEqual(result.denominator, 0.0)
        self.assertEqual(result.safety_directional_derivative, 0.0)

    def test_nonpositive_preconditioner_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cspr.project_preconditioned_preference_direction(
                torch.ones(2), torch.ones(2), torch.tensor([1.0, 0.0])
            )


class CsprFiniteStepTests(unittest.TestCase):
    def test_finite_difference_matches_projection_certificate(self) -> None:
        point = torch.tensor([0.2, -0.3], dtype=torch.float64)
        target_p = torch.tensor([1.0, 0.5], dtype=torch.float64)
        target_s = torch.tensor([-0.5, 0.75], dtype=torch.float64)
        diagonal = torch.tensor([1.25, 0.75], dtype=torch.float64)

        def preference(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((x - target_p) ** 2).sum()

        def safety(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((x - target_s) ** 2).sum()

        g_p = point - target_p
        g_s = point - target_s
        result = cspr.project_preconditioned_preference_direction(g_p, g_s, diagonal)
        preference_fd = cspr.finite_difference_directional_derivative(
            preference, point, result.direction, 1e-6
        )
        safety_fd = cspr.finite_difference_directional_derivative(
            safety, point, result.direction, 1e-6
        )
        self.assertAlmostEqual(
            preference_fd, result.preference_directional_derivative, places=8
        )
        self.assertAlmostEqual(safety_fd, result.safety_directional_derivative, places=8)

    def test_dual_armijo_accepts_a_joint_descent_step(self) -> None:
        point = torch.zeros(2, dtype=torch.float64)
        preference_target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        safety_target = torch.tensor([1.0, 1.0], dtype=torch.float64)

        def preference(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((x - preference_target) ** 2).sum()

        def safety(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((x - safety_target) ** 2).sum()

        g_p = point - preference_target
        g_s = point - safety_target
        projection = cspr.project_preconditioned_preference_direction(
            g_p, g_s, torch.ones(2, dtype=torch.float64)
        )
        result = cspr.dual_objective_backtracking(
            point,
            projection.direction,
            preference,
            safety,
            projection.preference_directional_derivative,
            projection.safety_directional_derivative,
        )
        self.assertTrue(result.accepted)
        self.assertLess(result.final_preference_loss, result.initial_preference_loss)
        self.assertLess(result.final_safety_loss, result.initial_safety_loss)

    def test_tangent_first_order_direction_can_fail_every_finite_step(self) -> None:
        point = torch.zeros(2, dtype=torch.float64)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)

        def preference(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * (x[0] - 1.0) ** 2

        def safety(x: torch.Tensor) -> torch.Tensor:
            return x[0] ** 2 + x[1] ** 2

        result = cspr.dual_objective_backtracking(
            point,
            direction,
            preference,
            safety,
            preference_directional_derivative=-1.0,
            safety_directional_derivative=0.0,
            maximum_trials=12,
        )
        self.assertFalse(result.accepted)
        torch.testing.assert_close(result.point, point)


class CsprPreferenceLossTests(unittest.TestCase):
    def test_dpo_loss_is_strictly_decreasing_in_preference_gap(self) -> None:
        beta = 0.2
        lower = cspr.dpo_pair_loss(
            torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), beta
        )
        higher = cspr.dpo_pair_loss(
            torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), beta
        )
        self.assertLess(float(higher), float(lower))

    def test_dpo_gap_derivative_matches_closed_form(self) -> None:
        beta = 0.3
        gap = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        loss = cspr.dpo_pair_loss(gap, 0.0, 0.0, 0.0, beta)
        loss.backward()
        expected = -beta / (1.0 + math.exp(beta * float(gap.detach())))
        self.assertAlmostEqual(float(gap.grad), expected, places=12)


if __name__ == "__main__":
    unittest.main()
