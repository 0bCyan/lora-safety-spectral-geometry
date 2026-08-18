from __future__ import annotations

import unittest
from unittest import mock

import step3_barp_mcd_p0 as p0


class SchemaTests(unittest.TestCase):
    def test_public_rows_round_trip_legacy_arm_key(self) -> None:
        rows = [{"arms": {"rpfrsc_mcd": {"arm": "barp_mcd"}}}]
        public = p0.public_rows(rows)
        self.assertIn("barp_mcd", public[0]["arms"])
        self.assertEqual(p0.legacy_rows(public), rows)

    def test_public_summary_names_same_batch_and_barp(self) -> None:
        legacy = {
            "diagnostic": {"rpfrsc_mcd": {"minimum_worst_score": 0.1}},
            "paired": {"rpfrsc_minus_raw_target_minimum_score": {}},
            "search": {},
        }
        public = p0.public_summary(legacy)
        self.assertNotIn("diagnostic", public)
        self.assertIn("barp_mcd", public["same_batch"])
        self.assertIn("barp_minus_raw_target_minimum_score", public["paired"])


class GateTests(unittest.TestCase):
    def test_inherited_transfer_labels_are_narrowed_to_same_batch(self) -> None:
        inherited = {
            "integrity": True,
            "protected_finite_radius_search": True,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": False,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        with mock.patch.object(p0.rp, "evaluate_gates", return_value=inherited):
            gates = p0.evaluate_gates([], {}, {}, True, 0.2)
        self.assertNotIn("diagnostic_common_transfer", gates)
        self.assertTrue(gates["same_batch_common_descent"])
        self.assertFalse(gates["protected_same_batch_noninferiority"])

    def test_route_precedence_and_pass(self) -> None:
        gates = {
            "integrity": True,
            "protected_finite_radius_search": True,
            "same_batch_common_descent": True,
            "finite_step_common_descent": True,
            "protected_same_batch_noninferiority": True,
            "target_same_batch_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        )
        gates["protected_same_batch_noninferiority"] = False
        self.assertEqual(
            p0.select_route(gates),
            "STOP_BARP_MCD_P0_PROTECTED_NONINFERIORITY_FAILED",
        )
        gates["same_batch_common_descent"] = False
        self.assertEqual(
            p0.select_route(gates),
            "STOP_BARP_MCD_P0_SAME_BATCH_COMMON_DESCENT_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
