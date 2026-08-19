"""Static tests for the M3 V3 direction-audit wrapper."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m3_confirmation_v2 as v2


class M3ConfirmationV3Tests(unittest.TestCase):
    def test_v3_tolerance_is_strict_relative_to_m1_basis_gate(self) -> None:
        audit = {
            "fallback_max_abs_error": 9e-7,
            "candidate_max_abs_error": 9e-7,
            "solver_audit": {"optimization_valid": True},
            "basis_audit": {"orthonormality_max_error": 5.7e-8},
        }
        self.assertTrue(v2.numerical_direction_valid(audit, {"coordinate_tolerance": 1e-6}))
        audit["candidate_max_abs_error"] = 1.1e-6
        self.assertFalse(v2.numerical_direction_valid(audit, {"coordinate_tolerance": 1e-6}))
        self.assertLess(1e-6, 2e-4 / 100)


if __name__ == "__main__":
    unittest.main()
