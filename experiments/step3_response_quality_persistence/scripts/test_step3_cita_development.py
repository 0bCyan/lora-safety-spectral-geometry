from __future__ import annotations

import unittest

import step3_cita_development as development


class DevelopmentRouteTests(unittest.TestCase):
    @staticmethod
    def checks() -> dict[str, bool]:
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
            "safety_tokens_equal": True,
            "safety_weight_budget_equal": True,
            "first_step_risk_loss_equal": True,
            "first_step_uniform_safety_loss_equal": True,
            "all_registered_assignments_exercised": True,
            "sealed_data_untouched": True,
        }

    def test_pass_route_is_development_specific(self) -> None:
        self.assertEqual(
            development.development_route(self.checks()),
            "PASS_CITA_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE",
        )

    def test_matching_failure_is_development_specific(self) -> None:
        checks = self.checks()
        checks["risk_order_equal"] = False
        self.assertEqual(
            development.development_route(checks),
            "STOP_CITA_DEVELOPMENT16_MATCHING_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
