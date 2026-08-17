from __future__ import annotations

import math
import unittest

import analyze_step3_rufmcd_v2_angle as audit


class RufmcdV2AngleAuditTests(unittest.TestCase):
    def test_trig_roots_recover_known_crossings(self) -> None:
        roots = audit.trig_roots(1.0, 0.0, 0.95, 0.0, 1.0)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], math.acos(0.95), places=12)

    def test_tangent_reconstruction_is_exact_at_selected_angle(self) -> None:
        theta = 0.2
        raw = 0.17
        tangent = -0.08
        candidate = audit.score_at(raw, tangent, theta)
        recovered = (candidate - raw * math.cos(theta)) / math.sin(theta)
        self.assertAlmostEqual(recovered, tangent, places=14)

    def test_interval_classifier_preserves_disjoint_components(self) -> None:
        intervals = audit._feasible_intervals(
            [0.0, 0.2, 0.4, 0.6],
            lambda theta: theta < 0.2 or theta > 0.4,
        )
        self.assertEqual(
            intervals,
            [{"lower": 0.0, "upper": 0.2}, {"lower": 0.4, "upper": 0.6}],
        )


if __name__ == "__main__":
    unittest.main()
