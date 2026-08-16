"""Route tests for the prospectively frozen CSPR P2T A/B analysis."""

from __future__ import annotations

import unittest

import step3_cspr_p2t_analysis as analysis


class RouteTests(unittest.TestCase):
    def test_all_gates_pass_authorizes_only_c_protocol(self) -> None:
        self.assertEqual(
            analysis.select_route(True, True, True),
            "AUTHORIZE_CSPR_P2T_REVIEWER_C_PROTOCOL_ONLY",
        )

    def test_calibration_failure_has_priority(self) -> None:
        self.assertEqual(
            analysis.select_route(False, False, False),
            "STOP_CSPR_P2T_CALIBRATION_UNRELIABLE",
        )

    def test_duplicate_failure_is_distinct(self) -> None:
        self.assertEqual(
            analysis.select_route(True, True, False),
            "STOP_CSPR_P2T_DUPLICATE_UNRELIABLE",
        )

    def test_target_failure_follows_calibration_and_duplicates(self) -> None:
        self.assertEqual(
            analysis.select_route(True, False, True),
            "STOP_CSPR_P2T_TARGET_UNRELIABLE",
        )


if __name__ == "__main__":
    unittest.main()
