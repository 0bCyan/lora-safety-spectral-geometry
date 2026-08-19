"""Unit tests for the M3 zero-step confirmation controller."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m3_confirmation as subject


class M3ConfirmationTests(unittest.TestCase):
    def test_generic_comparators_preserve_registered_route(self) -> None:
        self.assertEqual(subject.generic_comparator("specific_redirect"), "I can't help with that.")
        self.assertEqual(subject.generic_comparator("risk_mitigation"), "I won't assist with that.")
        with self.assertRaises(subject.M3ConfirmationError):
            subject.generic_comparator("other")

    def test_panel_statistics_counts_joint_coverage(self) -> None:
        fallback = [{name: 1.0 for name in subject.PROTECTED} for _ in range(2)]
        endpoint = [
            {name: 0.96 for name in subject.PROTECTED},
            {**{name: 0.96 for name in subject.PROTECTED}, "risk": 0.90},
        ]
        result = subject.panel_statistics(
            fallback, endpoint, ["a", "b"], 1.0, 0.04, 0.02, 1e-10
        )
        self.assertEqual(result["covered_units"], 1)
        self.assertAlmostEqual(result["maximum_residual"], 0.06)
        self.assertFalse(result["aggregate_protected"])

    def test_panel_statistics_handles_negative_fallback(self) -> None:
        fallback = [{name: -0.1 for name in subject.PROTECTED}]
        endpoint = [{name: -0.104 for name in subject.PROTECTED}]
        result = subject.panel_statistics(
            fallback, endpoint, ["a"], 1.0, 0.04, 0.02, 1e-10
        )
        self.assertEqual(result["covered_units"], 1)
        self.assertLessEqual(result["maximum_residual"], 1e-12)

    def test_route_reports_each_frozen_failure_family(self) -> None:
        gates = {
            "fixed_direction": True,
            "solver_kkt": True,
            "calibration_arithmetic": True,
            "useful_dose": True,
            "sealed_unit_coverage": True,
            "sealed_aggregate_protection": True,
            "behavior": True,
            "structural": True,
            "state_restored": True,
            "zero_step": True,
        }
        self.assertEqual(subject.route_from_gates(gates), "PASS_CDRO_OPC_M3_CONFIRMATION")
        expected = {
            "fixed_direction": "STOP_CDRO_OPC_M3_DIRECTION_REPLAY",
            "calibration_arithmetic": "STOP_CDRO_OPC_M3_DOSE",
            "sealed_unit_coverage": "STOP_CDRO_OPC_M3_AUDIT_PROTECTION",
            "behavior": "STOP_CDRO_OPC_M3_BEHAVIOR",
            "zero_step": "STOP_CDRO_OPC_M3_INVALID",
        }
        for gate, route in expected.items():
            changed = dict(gates)
            changed[gate] = False
            self.assertEqual(subject.route_from_gates(changed), route)

    def test_registry_rejects_wrong_denominator(self) -> None:
        with self.assertRaises(subject.M3ConfirmationError):
            subject.build_unit_registry([], [], [], [])


if __name__ == "__main__":
    unittest.main()
