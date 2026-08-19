"""Tests for registered-certificate normalization in M3 V4."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m3_confirmation_v4 as subject


class M3ConfirmationV4Tests(unittest.TestCase):
    def test_unregistered_zero_target_does_not_enter_protected_subset(self) -> None:
        values = {
            "risk": 1,
            "uniform": 2,
            "tarc": 3,
            "continuation": 4,
            "branch_entropy": 0,
            "branch_negative_gap": 5,
            "etarc": 6,
        }
        self.assertEqual(
            subject.registered_subset(values),
            {"risk": 1, "uniform": 2, "tarc": 3, "continuation": 4},
        )

    def test_missing_registered_certificate_still_fails_closed(self) -> None:
        with self.assertRaises(subject.core.M3ConfirmationError):
            subject.registered_subset({"risk": 1, "uniform": 2, "tarc": 3})


if __name__ == "__main__":
    unittest.main()
