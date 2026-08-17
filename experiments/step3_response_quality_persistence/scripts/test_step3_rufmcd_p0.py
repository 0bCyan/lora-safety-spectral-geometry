from __future__ import annotations

import math
import unittest

import torch

import step3_rufmcd_p0 as p0


class RufmcdProtocolTests(unittest.TestCase):
    def test_fixed_angle_is_inside_registered_cap(self) -> None:
        self.assertGreater(p0.FIXED_THETA, 0.0)
        self.assertLess(p0.FIXED_THETA, p0.base.THETA_MAX)
        self.assertGreaterEqual(
            math.cos(p0.FIXED_THETA), p0.base.DESIGN_COSINE_FLOOR
        )

    def test_facet_registry_contains_all_eighty_four_unit_facets(self) -> None:
        torch.manual_seed(19)
        gradients = [
            {
                name: {"x": torch.randn(8)}
                for name in p0.base.CERTIFICATES
            }
            for _ in range(p0.base.FIT_UNITS)
        ]
        specs = p0._facet_specs(gradients)
        self.assertEqual(len(specs), 84)
        self.assertEqual(
            sum(spec["kind"] == "protected" for spec in specs), 48
        )
        self.assertEqual(sum(spec["kind"] == "target" for spec in specs), 36)
        self.assertTrue(
            all(
                abs(p0.gradient_math.gradient_norm(spec["gradient"]) - 1.0)
                < 1e-6
                for spec in specs
            )
        )

    def test_joint_margin_search_uses_all_fit_thresholds(self) -> None:
        raw = {name: 1.0 for name in p0.base.PROTECTED_CERTIFICATES}
        raw.update({name: 0.05 for name in p0.base.TARGET_CERTIFICATES})
        candidate = {
            name: 0.96 for name in p0.base.PROTECTED_CERTIFICATES
        }
        candidate.update({name: 0.053 for name in p0.base.TARGET_CERTIFICATES})
        protocol = {
            "design": {"affine_actual_minus_predicted_max": 1e-6},
            "gates": {"minimum_nonzero_angle": 1e-8, "numeric_tolerance": 1e-12},
        }
        audit = {
            "solver_valid": True,
            "solver_margin": 0.001,
            "actual_minimum_margin": 0.001,
            "actual_minus_predicted": 0.0,
            "solver": {"primal_value": 0.001},
            "objective_names": [f"facet_{index:02d}" for index in range(84)],
        }
        result = p0._margin_search(
            [raw] * p0.base.FIT_UNITS,
            [candidate] * p0.base.FIT_UNITS,
            audit,
            True,
            protocol,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["fit_protected_facet_count"], 48)
        self.assertEqual(result["fit_target_facet_count"], 36)
        self.assertAlmostEqual(result["candidate_actual_minimum_margin"], 0.001)

    def test_geometry_gate_requires_all_projected_facets(self) -> None:
        protocol = {
            "design": {
                "minimum_retained_unit_facet_fraction": 0.01,
                "affine_kkt_residual_max": 1e-7,
                "affine_actual_minus_predicted_max": 1e-6,
                "constraint_generation_max_rounds": 12,
                "constraint_generation_full_gap_max": 1e-7,
            },
            "gates": {
                "projection_residual_relative_max": 1e-6,
                "candidate_norm_relative_tolerance": 1e-6,
            },
        }
        fit_geometry = {
            "projection_valid": True,
            "search": {"solver_candidate_actual_minus_predicted": 2e-8},
            "tangent_audit": {
                "enforcement": "unit_facet_margin_not_unit_level_nullspace",
                "target_covector_count": 84,
                "minimum_unit_facet_projection_fraction": 0.9,
                "raw_tangent_cosine": 1e-9,
                "tangent_to_raw_norm_ratio": 1.0,
                "target_covector_projection": {
                    "constraints": 1,
                    "vectors": [{} for _ in range(84)],
                    "maximum_relative_final_constraint_residual": 1e-9,
                },
                "affine_solver": {
                    "solver_success": True,
                    "kkt_residual": 3e-8,
                    "primal_dual_gap": 2e-8,
                    "constraint_generation_certified": True,
                    "constraint_generation_total_facet_count": 84,
                    "constraint_generation_round_count": 2,
                    "full_certificate_scores": {
                        f"facet_{index:02d}": 0.001 for index in range(84)
                    },
                    "full_primal_value": 0.001,
                    "full_dual_upper_bound": 0.00100000002,
                    "full_primal_dual_gap": 2e-8,
                },
            },
        }
        self.assertTrue(p0._unit_facet_geometry_gate(fit_geometry, protocol))
        fit_geometry["tangent_audit"]["target_covector_projection"]["vectors"].pop()
        self.assertFalse(p0._unit_facet_geometry_gate(fit_geometry, protocol))

    def test_constraint_generation_matches_direct_full_simplex_solution(self) -> None:
        gradients = {
            "protected:a0": {"x": torch.tensor([1.0, 0.0, 1.0])},
            "protected:a1": {"x": torch.tensor([-1.0, 0.0, 1.0])},
            "target:b0": {"x": torch.tensor([0.0, 1.0, 1.0])},
            "target:b1": {"x": torch.tensor([0.0, -1.0, 1.0])},
        }
        offsets = {
            "protected:a0": 0.0,
            "protected:a1": 0.1,
            "target:b0": 0.0,
            "target:b1": 0.1,
        }
        groups = {
            "protected:a0": "protected",
            "protected:a1": "protected",
            "target:b0": "target",
            "target:b1": "target",
        }
        direct_direction, direct = p0.optimizer_geometry.affine_minimax_direction(
            gradients, offsets, 1.0, 1e-9
        )
        generated_direction, generated = p0._constraint_generation_affine_minimax(
            gradients, offsets, groups, 1.0, 1e-9, 1e-7, 1e-7, 2
        )
        self.assertTrue(generated["constraint_generation_certified"])
        self.assertEqual(generated["constraint_generation_total_facet_count"], 4)
        self.assertEqual(generated["constraint_generation_active_facet_count"], 4)
        self.assertLessEqual(generated["full_primal_dual_gap"], 1e-7)
        self.assertAlmostEqual(
            generated["full_primal_value"], direct["primal_value"], places=6
        )
        self.assertAlmostEqual(
            p0.gradient_math.gradient_norm(generated_direction),
            p0.gradient_math.gradient_norm(direct_direction),
            places=6,
        )

    def test_synthetic_full_geometry_closes_all_eighty_four_facets(self) -> None:
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
                "affine_actual_minus_predicted_max": 1e-6,
                "minimum_retained_unit_facet_fraction": 0.01,
                "constraint_generation_max_rounds": 12,
                "constraint_generation_full_gap_max": 1e-7,
            },
            "gates": {
                "minimum_nonzero_angle": 1e-8,
                "numeric_tolerance": 1e-12,
                "projection_residual_relative_max": 1e-6,
                "candidate_norm_relative_tolerance": 1e-6,
            },
        }
        geometry, _ = p0.build_fit_geometry(
            fit_gradients, fit_coverages, config, protocol
        )
        solver = geometry["tangent_audit"]["affine_solver"]
        self.assertTrue(geometry["projection_valid"])
        self.assertEqual(geometry["facet_count"], 84)
        self.assertTrue(solver["constraint_generation_certified"])
        self.assertEqual(len(solver["full_certificate_scores"]), 84)
        self.assertLessEqual(solver["full_primal_dual_gap"], 1e-7)
        self.assertLessEqual(
            abs(geometry["search"]["solver_candidate_actual_minus_predicted"]),
            1e-6,
        )
        self.assertTrue(p0._unit_facet_geometry_gate(geometry, protocol))

    def test_route_namespace_uses_unit_facet_geometry_name(self) -> None:
        gates = {
            "integrity": True,
            "protected_nullspace_geometry": False,
            "fit_minimum_effective_angle": True,
            "split_conformal_accept_or_raw_policy": True,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "STOP_RUFMCD_P0_UNIT_FACET_GEOMETRY_INVALID",
        )


if __name__ == "__main__":
    unittest.main()
