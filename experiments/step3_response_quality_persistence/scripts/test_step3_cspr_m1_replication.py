"""Synthetic tests for the CSPR M1 disjoint-batch replication runner."""

from __future__ import annotations

import unittest

import step3_cspr_m1_replication as m1


class PreferenceAssignmentTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, str]]:
        repeated = [
            {"category": category, "pair_id": f"{category}-{index}"}
            for category in "abcdef"
            for index in range(2)
        ]
        singletons = [
            {"category": category, "pair_id": f"{category}-0"}
            for category in "ghijklmn"
        ]
        return repeated + singletons

    def test_assignment_is_balanced_diverse_and_disjoint(self) -> None:
        batches = m1.assign_preference_batches(self.rows(), 17, 3, 6, 6)
        self.assertEqual([len(batch) for batch in batches], [6, 6, 6])
        self.assertTrue(all(len({row["category"] for row in batch}) == 6 for batch in batches))
        ids = [row["pair_id"] for batch in batches for row in batch]
        self.assertEqual(len(ids), len(set(ids)))

    def test_assignment_is_order_invariant(self) -> None:
        forward = m1.assign_preference_batches(self.rows(), 17, 3, 6, 6)
        reverse = m1.assign_preference_batches(list(reversed(self.rows())), 17, 3, 6, 6)
        self.assertEqual(forward, reverse)

    def test_insufficient_repeated_categories_are_rejected(self) -> None:
        with self.assertRaisesRegex(m1.M1ValidationError, "repeated"):
            m1.assign_preference_batches(self.rows()[:10], 17, 3, 6, 6)

    def test_insufficient_singleton_categories_are_rejected(self) -> None:
        with self.assertRaisesRegex(m1.M1ValidationError, "singleton"):
            m1.assign_preference_batches(self.rows()[:12], 17, 3, 6, 6)


class SafetyAssignmentTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, str]]:
        common = [
            {"category": category, "target_id": f"{category}-{index}"}
            for category in "abcde"
            for index in range(4)
        ]
        rare = [
            {"category": category, "target_id": f"{category}-0"}
            for category in "fgh"
        ]
        return common + rare

    def test_assignment_is_balanced_diverse_and_disjoint(self) -> None:
        batches = m1.assign_safety_batches(self.rows(), 19, 3, 5, 3)
        self.assertEqual([len(batch) for batch in batches], [6, 6, 6])
        self.assertTrue(all(len({row["category"] for row in batch}) == 6 for batch in batches))
        ids = [row["target_id"] for batch in batches for row in batch]
        self.assertEqual(len(ids), len(set(ids)))

    def test_assignment_is_order_invariant(self) -> None:
        forward = m1.assign_safety_batches(self.rows(), 19, 3, 5, 3)
        reverse = m1.assign_safety_batches(list(reversed(self.rows())), 19, 3, 5, 3)
        self.assertEqual(forward, reverse)

    def test_insufficient_shared_categories_are_rejected(self) -> None:
        with self.assertRaisesRegex(m1.M1ValidationError, "shared"):
            m1.assign_safety_batches(self.rows()[:11], 19, 3, 5, 3)

    def test_insufficient_rare_categories_are_rejected(self) -> None:
        with self.assertRaisesRegex(m1.M1ValidationError, "rare"):
            m1.assign_safety_batches(self.rows()[:20], 19, 3, 5, 3)


class CoverageTests(unittest.TestCase):
    def test_legacy_coverage_field_is_renamed_by_meaning(self) -> None:
        result = m1.normalize_coverage(
            {
                "parameter_tensors": 4,
                "parameter_elements": 20,
                "missing_gradient_tensors": [],
                "finite": True,
                "nonzero_elements": 4,
            }
        )
        self.assertNotIn("nonzero_elements", result)
        self.assertEqual(result["nonzero_gradient_tensors"], 4)
        self.assertTrue(m1.coverage_valid(result))

    def test_zero_gradient_tensor_fails_coverage(self) -> None:
        result = {
            "parameter_tensors": 4,
            "parameter_elements": 20,
            "missing_gradient_tensors": [],
            "finite": True,
            "nonzero_gradient_tensors": 3,
        }
        self.assertFalse(m1.coverage_valid(result))


class RescueTests(unittest.TestCase):
    def test_safety_rescue_requires_active_accepted_and_tolerance_exceedance(self) -> None:
        search = {"accepted": True}
        unconstrained = {"safety_loss": 1.000002}
        self.assertTrue(m1.same_step_safety_rescue(True, search, unconstrained, 1.0, 1e-6))
        self.assertFalse(m1.same_step_safety_rescue(False, search, unconstrained, 1.0, 1e-6))
        self.assertFalse(
            m1.same_step_safety_rescue(
                True, search, {"safety_loss": 1.0000005}, 1.0, 1e-6
            )
        )


class RouteTests(unittest.TestCase):
    @staticmethod
    def gates() -> dict[str, bool]:
        return {
            "structural_and_gradient_valid": True,
            "directional_and_kkt": True,
            "state_restored": True,
            "zero_optimizer": True,
            "label_swap_control": True,
            "pair_activation": True,
            "aggregate_activation": True,
            "retained_preference": True,
            "finite_joint_step": True,
            "replicated_safety_rescue": True,
        }

    def test_all_gates_pass(self) -> None:
        self.assertEqual(
            m1.select_route(self.gates()), "PASS_CSPR_M1_BATCH_ROBUST_MECHANISM"
        )

    def test_invalid_has_priority(self) -> None:
        gates = self.gates()
        gates["state_restored"] = False
        gates["label_swap_control"] = False
        self.assertEqual(m1.select_route(gates), "STOP_CSPR_M1_INVALID")

    def test_label_swap_failure_is_distinct(self) -> None:
        gates = self.gates()
        gates["label_swap_control"] = False
        self.assertEqual(m1.select_route(gates), "STOP_CSPR_M1_DPO_CONTROL_FAILED")

    def test_activation_failure_is_distinct(self) -> None:
        gates = self.gates()
        gates["aggregate_activation"] = False
        self.assertEqual(m1.select_route(gates), "STOP_CSPR_M1_PROJECTION_NOT_REPLICATED")

    def test_retention_failure_is_distinct(self) -> None:
        gates = self.gates()
        gates["retained_preference"] = False
        self.assertEqual(
            m1.select_route(gates), "STOP_CSPR_M1_PROJECTED_DIRECTION_DEGENERATE"
        )

    def test_joint_step_failure_is_distinct(self) -> None:
        gates = self.gates()
        gates["finite_joint_step"] = False
        self.assertEqual(
            m1.select_route(gates), "STOP_CSPR_M1_FINITE_STEP_NOT_REPLICATED"
        )

    def test_rescue_failure_is_distinct(self) -> None:
        gates = self.gates()
        gates["replicated_safety_rescue"] = False
        self.assertEqual(
            m1.select_route(gates), "STOP_CSPR_M1_NO_REPLICATED_SAFETY_RESCUE"
        )


if __name__ == "__main__":
    unittest.main()
