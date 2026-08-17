from __future__ import annotations

import math
import unittest

import step3_nptlrmcd_p0 as p0


class LowRankProtocolTests(unittest.TestCase):
    def test_fixed_theta_retains_design_margin(self) -> None:
        self.assertGreater(p0.FIXED_THETA, 0.0)
        self.assertLess(p0.FIXED_THETA, p0.base.THETA_MAX)
        self.assertGreaterEqual(
            math.cos(p0.FIXED_THETA), p0.base.DESIGN_COSINE_FLOOR
        )

    def test_route_namespace_keeps_gate_order(self) -> None:
        gates = {
            "integrity": True,
            "protected_nullspace_geometry": True,
            "fit_minimum_effective_angle": False,
            "split_conformal_accept_or_raw_policy": True,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates), "STOP_NPTLRMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE"
        )


if __name__ == "__main__":
    unittest.main()
