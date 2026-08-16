"""Synthetic tests for the CSPR M0 zero-update mechanism runner."""

from __future__ import annotations

import unittest

import step3_cspr_m0_mechanism as m0


class SelectionTests(unittest.TestCase):
    def test_category_selection_is_deterministic_and_diverse(self) -> None:
        rows = [
            {"category": category, "id": f"{category}-{index}"}
            for category in "abcd"
            for index in range(3)
        ]
        first = m0.select_category_diverse(rows, 3, 7, "category", "id", "test")
        second = m0.select_category_diverse(list(reversed(rows)), 3, 7, "category", "id", "test")
        self.assertEqual(first, second)
        self.assertEqual(len({row["category"] for row in first}), 3)

    def test_insufficient_categories_are_rejected(self) -> None:
        rows = [{"category": "a", "id": "1"}, {"category": "a", "id": "2"}]
        with self.assertRaisesRegex(m0.M0ValidationError, "fewer"):
            m0.select_category_diverse(rows, 2, 7, "category", "id", "test")

    def test_candidate_choice_is_one_per_pair(self) -> None:
        rows = [
            {"pair_id": "p1", "opaque_authoring_id": "a"},
            {"pair_id": "p1", "opaque_authoring_id": "b"},
            {"pair_id": "p2", "opaque_authoring_id": "c"},
        ]
        selected = m0.choose_candidate_per_pair(rows, 9)
        self.assertEqual({row["pair_id"] for row in selected}, {"p1", "p2"})
        self.assertEqual(len(selected), 2)


class ProjectionSummaryTests(unittest.TestCase):
    def test_inactive_pair_retains_full_preference(self) -> None:
        import torch

        result = m0.pair_projection_metrics(
            torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
        )
        self.assertFalse(result["active"])
        self.assertEqual(result["retained_preference_fraction"], 1.0)

    def test_active_pair_retention_matches_sine_squared(self) -> None:
        import torch

        result = m0.pair_projection_metrics(
            torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 0.0])
        )
        self.assertTrue(result["active"])
        self.assertAlmostEqual(result["retained_preference_fraction"], 0.5)


class RouteTests(unittest.TestCase):
    @staticmethod
    def gates() -> dict[str, bool]:
        return {
            "structural_and_gradient_valid": True,
            "state_restored": True,
            "label_swap_control": True,
            "pair_activation": True,
            "aggregate_activation": True,
            "retained_preference": True,
            "directional_and_kkt": True,
            "finite_joint_step": True,
            "zero_optimizer": True,
        }

    def test_all_gates_pass(self) -> None:
        self.assertEqual(
            m0.select_route(self.gates()),
            "PASS_CSPR_M0_REAL_MODEL_MECHANISM_FEASIBLE",
        )

    def test_control_failure_has_priority(self) -> None:
        gates = self.gates()
        gates["label_swap_control"] = False
        gates["aggregate_activation"] = False
        self.assertEqual(m0.select_route(gates), "STOP_CSPR_M0_DPO_CONTROL_FAILED")

    def test_inactive_projection_is_distinct(self) -> None:
        gates = self.gates()
        gates["aggregate_activation"] = False
        self.assertEqual(
            m0.select_route(gates), "STOP_CSPR_M0_PROJECTION_NOT_EXERCISED"
        )

    def test_degenerate_direction_is_distinct(self) -> None:
        gates = self.gates()
        gates["retained_preference"] = False
        self.assertEqual(
            m0.select_route(gates), "STOP_CSPR_M0_PROJECTED_DIRECTION_DEGENERATE"
        )

    def test_line_search_failure_is_distinct(self) -> None:
        gates = self.gates()
        gates["finite_joint_step"] = False
        self.assertEqual(m0.select_route(gates), "STOP_CSPR_M0_NO_FINITE_JOINT_STEP")

    def test_optimizer_violation_is_invalid(self) -> None:
        gates = self.gates()
        gates["zero_optimizer"] = False
        self.assertEqual(m0.select_route(gates), "STOP_CSPR_M0_INVALID")


if __name__ == "__main__":
    unittest.main()
