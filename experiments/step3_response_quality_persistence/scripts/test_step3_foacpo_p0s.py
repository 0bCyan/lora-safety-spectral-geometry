from __future__ import annotations

import unittest

import step3_foacpo_p0s as p0s


class SegmentSolverTests(unittest.TestCase):
    def test_exact_interior_maximin(self) -> None:
        result = p0s.maximin_segment_blend(
            {"left": 1.0, "right": 0.0}, {"left": 0.0, "right": 1.0}
        )
        self.assertAlmostEqual(result["weight"], 0.5)
        self.assertAlmostEqual(result["minimum_score"], 0.5)
        self.assertEqual(result["active_certificates"], ["left", "right"])

    def test_endpoint_optimum(self) -> None:
        result = p0s.maximin_segment_blend(
            {"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 3.0}
        )
        self.assertEqual(result["weight"], 1.0)
        self.assertEqual(result["minimum_score"], 2.0)


class RouteTests(unittest.TestCase):
    def test_robust_anchor_failure_has_priority(self) -> None:
        gates = {
            "structural": True,
            "robust_anchor": False,
            "candidate_selected": False,
            "calibration": False,
            "mechanism": False,
            "state_restored": True,
            "zero_optimizer": True,
        }
        self.assertEqual(p0s.route_from_gates(gates), "STOP_FOACPO_P0S_NO_ROBUST_ANCHOR")


if __name__ == "__main__":
    unittest.main()
