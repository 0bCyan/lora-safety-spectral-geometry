from __future__ import annotations

import unittest

import step3_cdro_opc_m2_development as development


class DevelopmentHelpersTests(unittest.TestCase):
    def test_behavior_interpolation_preserves_registry_shape(self) -> None:
        panel = {
            "calibration": {
                "quality": {
                    "fallback_aggregate_cosine": 0.0,
                    "candidate_aggregate_cosine": 1.0,
                    "fallback": {"values": [0.0, 0.0]},
                    "candidate": {"values": [1.0, -1.0]},
                }
            }
        }
        result = development.behavior_at_dose(
            panel, [1.0, 0.0], [0.0, 1.0], 0.5
        )
        self.assertEqual(result["calibration"]["quality"]["count"], 2)
        self.assertEqual(result["calibration"]["quality"]["positive"], 1)

    def test_direct_residual_is_zero_at_exact_limit(self) -> None:
        residuals = development.direct_residuals(
            [{"risk": 1.0}],
            [{"risk": 0.2}],
            ["risk"],
            selected_dose=0.05,
            allowed_relative_drop=0.04,
            score_scale_floor=0.02,
        )
        self.assertAlmostEqual(residuals[0]["risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
