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


if __name__ == "__main__":
    unittest.main()
