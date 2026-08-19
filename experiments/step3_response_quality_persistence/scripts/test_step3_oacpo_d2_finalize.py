"""Tests for OACPO D2 adjudication and qualification logic."""

from __future__ import annotations

import unittest

import step3_oacpo_d2_finalize as finalize


class OacpoD2FinalizeTests(unittest.TestCase):
    def test_agreed_field_is_immutable(self) -> None:
        a = {"x": {"field": "yes"}}
        b = {"x": {"field": "yes"}}
        c = {"x": {"field": "no"}}
        value, used_c = finalize.adjudicate_field("x", "field", a, b, c)
        self.assertEqual(value, "yes")
        self.assertFalse(used_c)

    def test_disputed_field_uses_c(self) -> None:
        a = {"x": {"field": 1}}
        b = {"x": {"field": 2}}
        c = {"x": {"field": 2}}
        value, used_c = finalize.adjudicate_field("x", "field", a, b, c)
        self.assertEqual(value, 2)
        self.assertTrue(used_c)

    def test_disputed_field_without_c_fails(self) -> None:
        a = {"x": {"field": "yes"}}
        b = {"x": {"field": "no"}}
        with self.assertRaisesRegex(ValueError, "missing Reviewer C"):
            finalize.adjudicate_field("x", "field", a, b, {})

    def test_yes_no_constants_are_exact(self) -> None:
        self.assertEqual(finalize.YES_NO, {"yes", "no"})


if __name__ == "__main__":
    unittest.main()
