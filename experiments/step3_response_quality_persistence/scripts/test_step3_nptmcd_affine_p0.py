from __future__ import annotations

import unittest

import step3_nptmcd_affine_p0 as p0


class AffineProtocolTests(unittest.TestCase):
    def test_fixed_theta_has_registered_protection_margin(self) -> None:
        self.assertGreater(p0.FIXED_THETA, 0.0)
        self.assertLess(p0.FIXED_THETA, p0.base.THETA_MAX)
        self.assertGreaterEqual(
            __import__("math").cos(p0.FIXED_THETA), p0.base.DESIGN_COSINE_FLOOR
        )

    def test_route_namespace_preserves_base_gate_order(self) -> None:
        gates = {
            "integrity": False,
            "protected_nullspace_geometry": True,
            "fit_minimum_effective_angle": True,
            "split_conformal_accept_or_raw_policy": True,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(p0.select_route(gates), "STOP_NPTAMCD_P0_INVALID")


if __name__ == "__main__":
    unittest.main()
