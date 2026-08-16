from __future__ import annotations

import unittest

import torch

import step3_ramp_training as training


class LossTests(unittest.TestCase):
    def test_uniform_uses_no_auxiliary_loss(self) -> None:
        value = training.compose_safety_loss(
            torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0),
            "uniform", 0.25, 0.5,
        )
        self.assertEqual(float(value), 2.0)

    def test_prefix_and_suffix_use_their_frozen_coefficients(self) -> None:
        prefix = training.compose_safety_loss(
            torch.tensor(2.0), torch.tensor(4.0), torch.tensor(8.0),
            "prefix", 0.25, 0.5,
        )
        suffix = training.compose_safety_loss(
            torch.tensor(2.0), torch.tensor(4.0), torch.tensor(8.0),
            "suffix", 0.25, 0.5,
        )
        self.assertEqual(float(prefix), 3.0)
        self.assertEqual(float(suffix), 6.0)


class RouteTests(unittest.TestCase):
    @staticmethod
    def passing() -> dict[str, bool | float]:
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
            "direction_realized": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            training.select_smoke_route(self.passing()),
            "PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL",
        )

    def test_matching_failure_precedes_direction(self) -> None:
        checks = self.passing()
        checks["initial_state_equal"] = False
        checks["direction_realized"] = False
        self.assertEqual(
            training.select_smoke_route(checks), "STOP_RAMP_SMOKE_MATCHING_FAILED"
        )

    def test_direction_has_distinct_stop(self) -> None:
        checks = self.passing()
        checks["direction_realized"] = False
        self.assertEqual(
            training.select_smoke_route(checks), "STOP_RAMP_SMOKE_DIRECTION_NOT_REALIZED"
        )


if __name__ == "__main__":
    unittest.main()
