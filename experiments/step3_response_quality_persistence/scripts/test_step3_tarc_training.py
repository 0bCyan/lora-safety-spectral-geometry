from __future__ import annotations

import unittest

import torch

import step3_tarc_training as training


class LossTests(unittest.TestCase):
    def test_each_arm_uses_only_registered_auxiliary(self) -> None:
        uniform = torch.tensor(2.0)
        ramp = torch.tensor(4.0)
        tarc = torch.tensor(8.0)
        self.assertEqual(
            float(training.compose_safety_loss(uniform, ramp, tarc, "uniform", 0.25, 0.5)),
            2.0,
        )
        self.assertEqual(
            float(training.compose_safety_loss(uniform, ramp, tarc, "ramp", 0.25, 0.5)),
            3.0,
        )
        self.assertEqual(
            float(training.compose_safety_loss(uniform, ramp, tarc, "tarc", 0.25, 0.5)),
            6.0,
        )


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
            "safety_pair_draws_equal": True,
            "safety_target_rows_equal": True,
            "safety_response_tokens_equal": True,
            "available_ramp_terms_equal": True,
            "available_tarc_terms_equal": True,
            "first_step_risk_loss_equal": True,
            "first_step_uniform_safety_loss_equal": True,
            "initial_uniform_diagnostic_equal": True,
            "initial_ramp_diagnostic_equal": True,
            "initial_tarc_diagnostic_equal": True,
            "assignment_counts_correct": True,
            "all_registered_assignments_exercised": True,
            "sealed_data_untouched": True,
            "direction_realized": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            training.select_smoke_route(self.passing()),
            "PASS_TARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL",
        )

    def test_matching_failure_precedes_direction(self) -> None:
        checks = self.passing()
        checks["initial_state_equal"] = False
        checks["direction_realized"] = False
        self.assertEqual(
            training.select_smoke_route(checks), "STOP_TARC_SMOKE_MATCHING_FAILED"
        )

    def test_direction_failure_has_distinct_stop(self) -> None:
        checks = self.passing()
        checks["direction_realized"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_TARC_SMOKE_DIRECTION_NOT_REALIZED",
        )


if __name__ == "__main__":
    unittest.main()
