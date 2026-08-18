from __future__ import annotations

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
