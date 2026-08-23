"""Unit tests for the M5 fixed-dose zero-step confirmation."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m5_confirmation as subject


class M5ConfirmationTests(unittest.TestCase):
    def test_audit_requires_every_calibration_gate(self) -> None:
        names = {
            "fixed_direction",
            "solver_kkt",
            "protected_coverage",
            "aggregate_protection",
            "behavior",
            "structural",
            "state_restored",
            "zero_step",
        }
        gates = {name: True for name in names}
        self.assertTrue(subject.calibration_authorizes_audit(gates))
        for name in names:
            changed = dict(gates)
            changed[name] = False
            self.assertFalse(subject.calibration_authorizes_audit(changed))

    def test_extra_or_missing_gate_fails_closed(self) -> None:
        gates = {
            "fixed_direction": True,
            "solver_kkt": True,
            "protected_coverage": True,
            "aggregate_protection": True,
            "behavior": True,
            "structural": True,
            "state_restored": True,
            "zero_step": True,
        }
        missing = dict(gates)
        missing.pop("behavior")
        extra = dict(gates, unexpected=True)
        self.assertFalse(subject.calibration_authorizes_audit(missing))
        self.assertFalse(subject.calibration_authorizes_audit(extra))


if __name__ == "__main__":
    unittest.main()
