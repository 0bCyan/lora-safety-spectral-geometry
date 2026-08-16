from __future__ import annotations

import unittest

import step3_ramp_development as development


class RouteTests(unittest.TestCase):
    @staticmethod
    def passing() -> dict[str, bool]:
        return {
            "all_arms_present": True,
            "steps_exact": True,
            "no_skipped_updates": True,
            "finite": True,
            "vram": True,
            "initial_state_equal": True,
            "risk_order_equal": True,
            "target_order_equal": True,
            "target_exposure_equal": True,
            "safety_draws_equal": True,
            "safety_response_tokens_equal": True,
            "available_prefix_terms_equal": True,
            "available_suffix_terms_equal": True,
            "first_step_risk_loss_equal": True,
            "first_step_uniform_safety_loss_equal": True,
            "initial_uniform_diagnostic_equal": True,
            "initial_prefix_diagnostic_equal": True,
            "initial_suffix_diagnostic_equal": True,
            "assignment_counts_correct": True,
            "all_registered_assignments_exercised": True,
            "sealed_data_untouched": True,
            "direction_realized": False,
        }

    def test_development_ignores_teacher_forced_direction(self) -> None:
        self.assertEqual(
            development.select_development_route(self.passing()),
            "PASS_RAMP_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE",
        )

    def test_numerical_failure_has_distinct_route(self) -> None:
        checks = self.passing()
        checks["finite"] = False
        self.assertEqual(
            development.select_development_route(checks),
            "STOP_RAMP_DEVELOPMENT16_NUMERICAL_FAILED",
        )

    def test_matching_failure_precedes_numerical_failure(self) -> None:
        checks = self.passing()
        checks["initial_state_equal"] = False
        checks["finite"] = False
        self.assertEqual(
            development.select_development_route(checks),
            "STOP_RAMP_DEVELOPMENT16_MATCHING_FAILED",
        )

    def test_assignment_failure_has_distinct_route(self) -> None:
        checks = self.passing()
        checks["assignment_counts_correct"] = False
        self.assertEqual(
            development.select_development_route(checks),
            "STOP_RAMP_DEVELOPMENT16_ASSIGNMENT_NOT_EXERCISED",
        )


if __name__ == "__main__":
    unittest.main()
