from __future__ import annotations

import math
import unittest
from fractions import Fraction

import torch

import step3_rufmcd_jrp_p0 as jrp


class RufmcdJrpTests(unittest.TestCase):
    def _fit(self) -> list[dict[str, dict[str, torch.Tensor]]]:
        torch.manual_seed(41)
        return [
            {name: {"x": torch.randn(16)} for name in jrp.base.CERTIFICATES}
            for _ in range(jrp.base.FIT_UNITS)
        ]

    def test_reflection_count_and_rule(self) -> None:
        fit = self._fit()
        panels = [{name: 0.2 for name in jrp.base.CERTIFICATES} for _ in fit]
        specs, audit = jrp._jackknife_reflected_tarc_specs(fit, panels)
        self.assertEqual(len(specs), jrp.ROBUST_EXTRA_FACET_COUNT)
        self.assertEqual(audit["observed_vertex_count"], 12)
        self.assertEqual(audit["reflected_vertex_count"], 12)
        self.assertTrue(all(spec["vertex_type"] == "leave_one_out_reflection" for spec in specs))

    def test_reflection_is_two_loo_mean_minus_observation(self) -> None:
        fit = [{"tarc": {"x": torch.tensor([float(i), 1.0])}} for i in range(12)]
        for item in fit:
            for name in jrp.base.CERTIFICATES:
                item.setdefault(name, {"x": torch.tensor([1.0, 0.5])})
        panels = [
            {name: float(index + 1) for name in jrp.base.CERTIFICATES}
            for index in range(len(fit))
        ]
        specs, _ = jrp._jackknife_reflected_tarc_specs(fit, panels)
        response_x = [
            float(jrp.v2._normalized_gradient(item["tarc"], "test")[0]["x"][0])
            / float(index + 1)
            for index, item in enumerate(fit)
        ]
        expected = 2.0 * sum(response_x[1:]) / 11.0 - response_x[0]
        self.assertAlmostEqual(float(specs[0]["gradient"]["x"][0]), expected, places=6)

    def test_total_facet_count_is_original_plus_reflections(self) -> None:
        self.assertEqual(jrp.FACET_COUNT, 84 + 12)
        self.assertEqual(jrp.LEGACY_LARGEST_GROUP_ROUND_BOUND, 24)
        self.assertEqual(jrp.CONSTRAINT_GENERATION_COMPLETE_ROUND_BOUND, 90)
        self.assertGreater(jrp.FIXED_THETA, 0.0)
        self.assertLess(jrp.FIXED_THETA, math.pi / 2.0)

    def test_reflected_margin_uses_response_covector_once(self) -> None:
        projected = {"x": torch.tensor([2.0])}
        tangent = {"x": torch.tensor([1.0])}
        expected = (
            math.cos(jrp.FIXED_THETA)
            - jrp.base.PROTECTED_RETENTION_FLOOR
            + math.sin(jrp.FIXED_THETA) * 2.0
        )
        self.assertAlmostEqual(
            jrp._robust_margin(projected, tangent, 1.0), expected, places=6
        )

    def test_realized_dot_bound_contains_exact_float32_dot(self) -> None:
        left = {"x": torch.tensor([0.1, -3.25, 7.0, 1.0 / 3.0])}
        right = {"x": torch.tensor([-2.0, 0.125, 0.75, -5.5])}
        value, bound, audit = jrp._dot_with_roundoff_bound(left, right, 2.0)
        exact = sum(
            Fraction.from_float(float(a)) * Fraction.from_float(float(b))
            for a, b in zip(left["x"], right["x"], strict=True)
        )
        self.assertLessEqual(abs(value - float(exact)), bound)
        self.assertEqual(audit["elements"], 4)
        self.assertEqual(audit["safety_factor"], 2.0)
        self.assertGreater(bound, 0.0)

    def test_direct_realized_certificate_accepts_positive_margin(self) -> None:
        gradients = {
            "a": {"x": torch.tensor([1.0, 0.0])},
            "b": {"x": torch.tensor([0.0, 1.0])},
        }
        direction, audit = jrp._direct_full_set_realized_affine_minimax(
            gradients, {"a": 1.0, "b": 1.0}, 1.0, 1e-9, 1e-7,
            1e-7, 0.0, 2.0,
        )
        self.assertTrue(audit["continuous_solver_success"])
        self.assertLessEqual(audit["continuous_kkt_residual"], 1e-7)
        self.assertLessEqual(audit["continuous_primal_dual_gap"], 1e-7)
        self.assertTrue(audit["realized_certificate_valid"])
        self.assertTrue(audit["realized_candidate_feasible"])
        self.assertGreater(audit["full_primal_lower_bound"], 0.0)
        self.assertAlmostEqual(jrp.gradient_math.gradient_norm(direction), 1.0, places=6)

    def test_direct_realized_certificate_separates_negative_feasibility(self) -> None:
        gradients = {
            "a": {"x": torch.tensor([1.0, 0.0])},
            "b": {"x": torch.tensor([0.0, 1.0])},
        }
        _, audit = jrp._direct_full_set_realized_affine_minimax(
            gradients, {"a": -2.0, "b": -2.0}, 1.0, 1e-9, 1e-7,
            1e-7, 0.0, 2.0,
        )
        self.assertTrue(audit["continuous_solver_success"])
        self.assertTrue(audit["realized_certificate_valid"])
        self.assertFalse(audit["realized_candidate_feasible"])
        self.assertLess(audit["full_primal_lower_bound"], 0.0)

    def test_gate_evaluator_keeps_pre_patch_base_reference(self) -> None:
        self.assertIsNot(jrp._BASE_EVALUATE_GATES, jrp.evaluate_gates)
        original = jrp.base.evaluate_gates
        try:
            jrp.base.evaluate_gates = jrp.evaluate_gates
            self.assertIsNot(jrp._BASE_EVALUATE_GATES, jrp.base.evaluate_gates)
        finally:
            jrp.base.evaluate_gates = original

    def test_original_facet_raw_score_comes_from_registered_panel(self) -> None:
        spec = {"unit_index": 1, "certificate": "tarc"}
        panels = [{}, {"tarc": 0.1}, {"tarc": 0.375}]
        self.assertEqual(jrp._registered_raw_score(spec, panels), 0.375)

    def test_synthetic_full_geometry_closes_all_ninety_six_facets(self) -> None:
        torch.manual_seed(43)
        fit_gradients = []
        fit_coverages = []
        for _ in range(jrp.base.FIT_UNITS):
            common = torch.ones(32)
            fit_gradients.append(
                {
                    name: {"x": common + 0.05 * torch.randn(32)}
                    for name in jrp.base.CERTIFICATES
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
        geometry, _ = jrp.build_fit_geometry(
            fit_gradients, fit_coverages, config, protocol
        )
        solver = geometry["tangent_audit"]["affine_solver"]
        self.assertTrue(geometry["projection_valid"], geometry["projection_error"])
        self.assertEqual(geometry["facet_count"], 96)
        self.assertEqual(geometry["robust_protected_constraint_count"], 12)
        self.assertTrue(solver["constraint_generation_certified"])
        self.assertEqual(len(solver["full_certificate_scores"]), 96)
        self.assertLessEqual(solver["full_primal_dual_gap"], 1e-7)
        self.assertTrue(jrp._geometry_gate(geometry, protocol))

    def test_synthetic_direct_realized_geometry_certifies_all_facets(self) -> None:
        torch.manual_seed(43)
        fit_gradients = []
        fit_coverages = []
        for _ in range(jrp.base.FIT_UNITS):
            common = torch.ones(32)
            fit_gradients.append(
                {
                    name: {"x": common + 0.05 * torch.randn(32)}
                    for name in jrp.base.CERTIFICATES
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
            "status": jrp.REALIZED_CERTIFICATE_RECOVERY_STATUS,
            "design": {
                "projection_rcond": 1e-10,
                "projection_refinement_steps": 2,
                "affine_solver_tolerance": 1e-9,
                "affine_kkt_residual_max": 1e-7,
                "continuous_kkt_residual_max": 1e-7,
                "continuous_primal_dual_gap_max": 1e-7,
                "affine_actual_minus_predicted_max": 1e-6,
                "minimum_retained_unit_facet_fraction": 0.01,
                "constraint_generation_max_rounds": 90,
                "constraint_generation_full_gap_max": 1e-7,
                "realized_direction_norm_relative_max": 1e-7,
                "realized_margin_lower_bound_min": 0.0,
                "dot_roundoff_safety_factor": 2.0,
            },
            "gates": {
                "minimum_nonzero_angle": 1e-8,
                "numeric_tolerance": 1e-12,
                "projection_residual_relative_max": 1e-6,
                "candidate_norm_relative_tolerance": 1e-6,
            },
        }
        geometry, _ = jrp.build_fit_geometry(
            fit_gradients, fit_coverages, config, protocol
        )
        solver = geometry["tangent_audit"]["affine_solver"]
        self.assertTrue(
            geometry["projection_valid"],
            {
                "error": geometry["projection_error"],
                "continuous_kkt": solver.get("continuous_kkt_residual"),
                "continuous_gap": solver.get("continuous_primal_dual_gap"),
                "realized_norm_error": solver.get(
                    "realized_direction_norm_relative_error"
                ),
                "realized_lower": solver.get("full_primal_lower_bound"),
                "realized_valid": solver.get("realized_certificate_valid"),
            },
        )
        self.assertEqual(solver["direct_full_set_solve_count"], 1)
        self.assertEqual(len(solver["full_certificate_scores"]), 96)
        self.assertEqual(len(solver["full_certificate_score_lower_bounds"]), 96)
        self.assertTrue(solver["realized_certificate_valid"])
        self.assertTrue(jrp._geometry_gate(geometry, protocol))

    def test_v3r5_frozen_artifact_uses_complete_bound(self) -> None:
        path = jrp.REPO / (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3r5/protocol/"
            "rufmcd_p0_v3r5_protocol.json"
        )
        protocol = jrp._read_json(path)
        config = jrp._read_json(jrp.base.resolve_repo_path(protocol["config_path"]))
        repair = protocol["execution_repair"]
        self.assertEqual(
            protocol["status"], jrp.GLOBAL_ROUND_BOUND_RECOVERY_STATUS
        )
        self.assertEqual(
            protocol["design"]["constraint_generation_max_rounds"],
            jrp.CONSTRAINT_GENERATION_COMPLETE_ROUND_BOUND,
        )
        self.assertEqual(
            config["rufmcd_constraint_generation_max_rounds"],
            jrp.CONSTRAINT_GENERATION_COMPLETE_ROUND_BOUND,
        )
        self.assertFalse(repair["method_changed"])
        self.assertFalse(repair["thresholds_changed"])
        self.assertFalse(repair["selection_changed"])


if __name__ == "__main__":
    unittest.main()
