"""Tests for the M3 direction-identity transport repair."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m3_confirmation_v2 as subject


class M3ConfirmationV2Tests(unittest.TestCase):
    def test_numeric_identity_does_not_require_json_sha_identity(self) -> None:
        audit = {
            "fallback_max_abs_error": 5e-11,
            "candidate_max_abs_error": 8e-11,
            "solver_audit": {"optimization_valid": True},
            "basis_audit": {"orthonormality_max_error": 1e-8},
        }
        self.assertTrue(
            subject.numerical_direction_valid(audit, {"coordinate_tolerance": 1e-10})
        )

    def test_numeric_identity_still_rejects_real_drift(self) -> None:
        audit = {
            "fallback_max_abs_error": 0.0,
            "candidate_max_abs_error": 2e-10,
            "solver_audit": {"optimization_valid": True},
            "basis_audit": {"orthonormality_max_error": 1e-8},
        }
        self.assertFalse(
            subject.numerical_direction_valid(audit, {"coordinate_tolerance": 1e-10})
        )


if __name__ == "__main__":
    unittest.main()
