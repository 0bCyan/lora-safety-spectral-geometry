from __future__ import annotations

import math
import unittest

import torch

import step3_ojmcd_common as ojmcd
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


class RealizedMapTests(unittest.TestCase):
    def test_first_adam_displacement_matches_closed_form(self) -> None:
        raw = {"p": torch.tensor([3.0, 4.0], dtype=torch.float64)}
        displacement, audit = ojmcd.clipped_adam_first_displacement(
            raw, max_norm=1.0, adam_epsilon=1e-2, clip_epsilon=0.0
        )
        clipped = torch.tensor([0.6, 0.8], dtype=torch.float64)
        expected = clipped / (clipped.abs() + 1e-2)
        self.assertTrue(torch.allclose(displacement["p"], expected, atol=1e-12))
        self.assertTrue(audit["clipping_active"])
        self.assertAlmostEqual(float(audit["clip_coefficient"]), 0.2, places=12)

    def test_first_displacement_matches_torch_adamw_after_clipping(self) -> None:
        parameter = torch.nn.Parameter(
            torch.tensor([3.0, -4.0, 0.2], dtype=torch.float64)
        )
        raw = {"p": torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)}
        expected, _ = ojmcd.clipped_adam_first_displacement(
            raw, max_norm=0.5, adam_epsilon=1e-4
        )
        before = parameter.detach().clone()
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1e-4,
            weight_decay=0.0,
        )
        parameter.grad = raw["p"].clone()
        torch.nn.utils.clip_grad_norm_([parameter], 0.5)
        optimizer.step()
        realized = before - parameter.detach()
        self.assertTrue(torch.allclose(realized, expected["p"], atol=1e-12))

    def test_jvp_matches_centered_finite_difference(self) -> None:
        raw = {"p": torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)}
        tangent = {"p": torch.tensor([0.3, 0.1, -0.2], dtype=torch.float64)}
        analytic, _ = ojmcd.clipped_adam_first_jvp(
            raw, tangent, max_norm=0.5, adam_epsilon=1e-2, clip_epsilon=1e-6
        )
        step = 1e-6
        plus = gradient_ops.add_gradients(raw, tangent, right_scale=step)
        minus = gradient_ops.add_gradients(raw, tangent, right_scale=-step)
        plus_value, _ = ojmcd.clipped_adam_first_displacement(
            plus, 0.5, 1e-2, 1e-6
        )
        minus_value, _ = ojmcd.clipped_adam_first_displacement(
            minus, 0.5, 1e-2, 1e-6
        )
        finite = (plus_value["p"] - minus_value["p"]) / (2.0 * step)
        self.assertTrue(torch.allclose(analytic["p"], finite, rtol=1e-6, atol=1e-7))

    def test_vjp_is_the_adjoint_of_jvp(self) -> None:
        raw = {"p": torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)}
        tangent = {"p": torch.tensor([0.3, 0.1, -0.2], dtype=torch.float64)}
        cotangent = {"p": torch.tensor([-0.2, 0.5, 0.4], dtype=torch.float64)}
        jvp, _ = ojmcd.clipped_adam_first_jvp(
            raw, tangent, max_norm=0.5, adam_epsilon=1e-2
        )
        vjp, _ = ojmcd.clipped_adam_first_vjp(
            raw, cotangent, max_norm=0.5, adam_epsilon=1e-2
        )
        self.assertAlmostEqual(
            gradient_math.gradient_dot(jvp, cotangent),
            gradient_math.gradient_dot(tangent, vjp),
            places=10,
        )

    def test_active_clipping_nearly_annihilates_radial_changes(self) -> None:
        raw = {"p": torch.tensor([3.0, 4.0], dtype=torch.float64)}
        jvp, _ = ojmcd.clipped_adam_first_jvp(
            raw, raw, max_norm=1.0, adam_epsilon=1e-2, clip_epsilon=0.0
        )
        self.assertLess(gradient_math.gradient_norm(jvp), 1e-10)


class OptimizerJacobianDirectionTests(unittest.TestCase):
    def test_transformed_direction_has_common_jacobian_descent(self) -> None:
        raw = {"p": torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)}
        certificates = {
            "a": {"p": torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64)},
            "b": {"p": torch.tensor([0.1, 1.0, 0.2], dtype=torch.float64)},
            "c": {"p": torch.tensor([0.2, 0.1, 1.0], dtype=torch.float64)},
        }
        direction, audit = ojmcd.optimizer_jacobian_common_direction(
            raw,
            certificates,
            budget_norm=0.1,
            max_norm=1.0,
            adam_epsilon=1e-2,
        )
        self.assertGreater(audit["dual_margin"], 0.0)
        self.assertAlmostEqual(gradient_math.gradient_norm(direction), 0.1, places=8)
        realized, _ = ojmcd.clipped_adam_first_jvp(
            raw, direction, max_norm=1.0, adam_epsilon=1e-2
        )
        for certificate in certificates.values():
            self.assertGreater(gradient_math.gradient_dot(certificate, realized), 0.0)

    def test_exact_increment_converges_to_linear_prediction(self) -> None:
        raw = {"p": torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)}
        certificates = {
            "a": {"p": torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64)},
            "b": {"p": torch.tensor([0.1, 1.0, 0.2], dtype=torch.float64)},
        }
        auxiliary = {"p": torch.tensor([0.02, 0.01, 0.03], dtype=torch.float64)}
        metrics = ojmcd.realized_increment_metrics(
            raw,
            auxiliary,
            certificates,
            auxiliary_scale=1e-4,
            max_norm=1.0,
            adam_epsilon=1e-2,
        )
        self.assertGreater(metrics["linear_exact_cosine"], 1.0 - 1e-8)
        self.assertTrue(
            math.isclose(metrics["linear_exact_norm_ratio"], 1.0, rel_tol=2e-4)
        )


class AffineMinimaxTests(unittest.TestCase):
    def test_equal_offsets_recover_common_descent_direction(self) -> None:
        gradients = {
            "a": {"p": torch.tensor([1.0, 0.0], dtype=torch.float64)},
            "b": {"p": torch.tensor([0.0, 1.0], dtype=torch.float64)},
        }
        affine, certificate = ojmcd.affine_minimax_direction(
            gradients, {"a": 0.3, "b": 0.3}, budget_norm=0.5
        )
        common, _ = ojmcd.mcd.minimax_common_direction(
            gradients, budget_norm=0.5
        )
        self.assertGreater(
            gradient_math.gradient_cosine(affine, common), 1.0 - 1e-9
        )
        self.assertLess(certificate["kkt_residual"], 1e-8)
        self.assertLess(certificate["primal_dual_gap"], 1e-8)

    def test_offsets_shift_the_balanced_solution(self) -> None:
        gradients = {
            "weak": {"p": torch.tensor([1.0, 0.0], dtype=torch.float64)},
            "strong": {"p": torch.tensor([0.0, 1.0], dtype=torch.float64)},
        }
        direction, certificate = ojmcd.affine_minimax_direction(
            gradients, {"weak": 0.0, "strong": 0.5}, budget_norm=1.0
        )
        expected_second = (-1.0 + math.sqrt(7.0)) / 4.0
        expected_first = expected_second + 0.5
        self.assertTrue(
            torch.allclose(
                direction["p"],
                torch.tensor(
                    [expected_first, expected_second], dtype=torch.float64
                ),
                atol=1e-7,
            )
        )
        self.assertAlmostEqual(
            certificate["certificate_scores"]["weak"],
            certificate["certificate_scores"]["strong"],
            places=7,
        )

    def test_baseline_aware_direction_improves_predicted_worst_score(self) -> None:
        raw = {"p": torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)}
        certificates = {
            "a": {"p": torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64)},
            "b": {"p": torch.tensor([0.1, 1.0, 0.2], dtype=torch.float64)},
            "c": {"p": torch.tensor([0.2, 0.1, 1.0], dtype=torch.float64)},
        }
        direction, audit = ojmcd.baseline_aware_optimizer_jacobian_direction(
            raw,
            certificates,
            budget_norm=0.1,
            auxiliary_scale=0.2,
            max_norm=1.0,
            adam_epsilon=1e-2,
        )
        self.assertAlmostEqual(gradient_math.gradient_norm(direction), 0.1, places=8)
        self.assertGreaterEqual(
            audit["predicted_minimum_score"],
            audit["baseline_minimum_score"] - 1e-9,
        )
        self.assertLess(audit["kkt_residual"], 1e-7)
        self.assertLess(audit["primal_dual_gap"], 1e-7)


if __name__ == "__main__":
    unittest.main()
