from __future__ import annotations

import math
import unittest

import torch

import step3_ramp_common as ramp


class PositionTests(unittest.TestCase):
    def test_prefix_and_suffix_have_equal_registered_width(self) -> None:
        prefix = ramp.selected_positions(22, 16, "prefix")
        suffix = ramp.selected_positions(22, 16, "suffix")
        self.assertEqual(prefix, list(range(16)))
        self.assertEqual(suffix, list(range(6, 22)))
        self.assertEqual(len(prefix), len(suffix))

    def test_short_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "registered width"):
            ramp.selected_positions(15, 16, "prefix")


class MarginTests(unittest.TestCase):
    def test_strongest_non_target_competitor_is_used(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 3.0], [5.0, 2.0, 1.0]])
        targets = torch.tensor([1, 2])
        margins, competitors = ramp.strongest_competitor_margin(logits, targets)
        torch.testing.assert_close(margins, torch.tensor([1.0, -4.0]))
        torch.testing.assert_close(competitors, torch.tensor([2, 0]))

    def test_barrier_decreases_as_margin_increases(self) -> None:
        low = ramp.margin_barrier(torch.tensor([-1.0]))
        high = ramp.margin_barrier(torch.tensor([1.0]))
        self.assertGreater(float(low), float(high))


class GradientTests(unittest.TestCase):
    @staticmethod
    def vector(*values: float) -> dict[str, torch.Tensor]:
        return {"p": torch.tensor(values)}

    def test_calibration_matches_requested_norm(self) -> None:
        coefficient = ramp.calibrated_coefficient(4.0, 2.0, 0.25)
        self.assertEqual(coefficient, 0.5)
        scaled = ramp.scaled_gradient(self.vector(2.0, 0.0), coefficient)
        self.assertAlmostEqual(ramp.gradient_norm(scaled), 1.0)

    def test_directional_metrics_reward_cross_prompt_prefix_alignment(self) -> None:
        result = ramp.directional_metrics(
            self.vector(0.0, 1.0),
            self.vector(1.0, 0.0),
            self.vector(0.0, 1.0),
            self.vector(1.0, 0.0),
            0.25,
            0.25,
        )
        self.assertEqual(result["calibration_validation_prefix_cosine"], 1.0)
        self.assertEqual(result["suffix_calibration_to_prefix_validation_cosine"], 0.0)
        self.assertGreater(result["ramp_increment_over_uniform_dot"], 0.0)

    def test_zero_norm_has_no_defined_cosine(self) -> None:
        value = ramp.gradient_cosine(self.vector(0.0), self.vector(1.0))
        self.assertTrue(math.isnan(value))


class RouteTests(unittest.TestCase):
    @staticmethod
    def passing() -> dict[str, bool]:
        return {
            "integrity": True,
            "gradient_valid": True,
            "budget_matched": True,
            "vram": True,
            "state_restored": True,
            "prefix_behavioral": True,
            "placebo_distinct": True,
            "cross_prompt_directional": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            ramp.select_route(self.passing()),
            "PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        )

    def test_invalid_precedes_mechanism_failures(self) -> None:
        gates = self.passing()
        gates["integrity"] = False
        gates["cross_prompt_directional"] = False
        self.assertEqual(ramp.select_route(gates), "STOP_RAMP_P0_INVALID")


if __name__ == "__main__":
    unittest.main()
