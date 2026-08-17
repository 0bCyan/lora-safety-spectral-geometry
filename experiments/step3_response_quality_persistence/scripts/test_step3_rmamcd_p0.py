from __future__ import annotations

import math
import unittest

import torch

import step3_rmamcd_p0 as p0


class RmaProtocolTests(unittest.TestCase):
    def test_fixed_angle_is_inside_registered_cap(self) -> None:
        self.assertGreater(p0.FIXED_THETA, 0.0)
        self.assertLess(p0.FIXED_THETA, p0.base.THETA_MAX)
        self.assertGreaterEqual(math.cos(p0.FIXED_THETA), p0.base.DESIGN_COSINE_FLOOR)

    def test_route_namespace_preserves_margin_gate_order(self) -> None:
        gates = {
            "integrity": True,
            "protected_nullspace_geometry": True,
            "fit_minimum_effective_angle": True,
            "split_conformal_accept_or_raw_policy": False,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates), "STOP_RMAMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED"
        )

    def test_global_geometry_projects_only_raw_displacement(self) -> None:
        torch.manual_seed(7)
        fit_gradients = []
        fit_coverages = []
        for _ in range(p0.base.FIT_UNITS):
            common = torch.ones(32)
            fit_gradients.append(
                {
                    name: {"x": common + 0.05 * torch.randn(32)}
                    for name in p0.base.CERTIFICATES
                }
            )
            fit_coverages.append(
                {
                    "mode": "eval",
                    "risk_microbatches": 1,
                    "safety_microbatches": 1,
                    "all_gradients_finite_nonzero": True,
                }
            )
        config = {
            "dual_numeric_tolerance": 1e-9,
            "safety_mix": 0.2,
            "auxiliary_gradient_norm_ratio": 1.0,
            "max_grad_norm": 1.0,
            "adam_epsilon": 1e-8,
            "clip_epsilon": 1e-6,
        }
        protocol = {
            "design": {
                "projection_rcond": 1e-10,
                "projection_refinement_steps": 2,
                "affine_solver_tolerance": 1e-9,
                "affine_kkt_residual_max": 1e-7,
                "minimum_retained_global_covector_fraction": 0.01,
                "minimum_retained_target_proposal_fraction": 0.01,
            },
            "gates": {
                "minimum_nonzero_angle": 1e-8,
                "numeric_tolerance": 1e-12,
                "fit_retention_identity_error_max": 1e-5,
                "projection_residual_relative_max": 1e-6,
                "candidate_norm_relative_tolerance": 1e-6,
            },
        }
        geometry, _ = p0.build_fit_geometry(
            fit_gradients, fit_coverages, config, protocol
        )
        self.assertTrue(geometry["projection_valid"])
        self.assertEqual(
            geometry["tangent_audit"]["target_covector_projection"]["constraints"],
            1,
        )
        self.assertEqual(geometry["tangent_audit"]["target_covector_count"], 7)
        self.assertEqual(
            geometry["tangent_audit"]["enforcement"],
            "global_pooled_margin_not_unit_level_nullspace",
        )
        self.assertLess(
            abs(geometry["tangent_audit"]["raw_tangent_cosine"]), 1e-6
        )
        self.assertTrue(p0._global_margin_geometry_gate(geometry, protocol))

    def test_unit_retention_identity_is_diagnostic_not_a_gate(self) -> None:
        raw = {
            name: 1.0
            for name in (*p0.base.PROTECTED_CERTIFICATES, *p0.base.TARGET_CERTIFICATES)
        }
        raw.update({name: 0.05 for name in p0.base.TARGET_CERTIFICATES})
        candidate = {
            name: 0.96 for name in p0.base.PROTECTED_CERTIFICATES
        }
        candidate.update({name: 0.053 for name in p0.base.TARGET_CERTIFICATES})
        protocol = {
            "gates": {
                "minimum_nonzero_angle": 1e-8,
                "numeric_tolerance": 1e-12,
                "fit_retention_identity_error_max": 1e-5,
            }
        }
        audit = {
            "solver_valid": True,
            "solver_margin": 1e-3,
            "solver": {"primal_value": 1e-3},
            "objective_names": [
                *(f"protected:{name}" for name in p0.base.PROTECTED_CERTIFICATES),
                *(f"target:{name}" for name in p0.base.TARGET_CERTIFICATES),
            ],
        }
        search = p0._margin_search([raw] * p0.base.FIT_UNITS,
                                   [candidate] * p0.base.FIT_UNITS,
                                   audit, True, protocol)
        self.assertTrue(search["eligible"])
        self.assertGreater(search["maximum_protected_retention_identity_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
