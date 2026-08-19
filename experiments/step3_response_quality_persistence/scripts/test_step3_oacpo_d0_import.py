"""Tests for mechanical OACPO D0 author-output validation."""

from __future__ import annotations

import copy
import unittest

import step3_oacpo_d0_import as author_import


class OacpoD0ImportTests(unittest.TestCase):
    def gates(self) -> dict:
        return {
            "required_rows": 2,
            "schema_exact": ["author_id", "authored_response"],
            "minimum_words": 4,
            "maximum_words": 8,
            "forbidden_substrings": ["<think>", "</think>"],
        }

    def packet(self) -> list[dict]:
        return [
            {"author_id": "a", "prompt": "prompt a"},
            {"author_id": "b", "prompt": "prompt b"},
        ]

    def output(self) -> list[dict]:
        return [
            {"author_id": "a", "authored_response": "one safe useful response here"},
            {"author_id": "b", "authored_response": "another distinct safe response here"},
        ]

    def test_valid_output_passes_and_reports_counts(self) -> None:
        canonical, diagnostics = author_import.validate_author_output(
            self.packet(), self.output(), self.gates()
        )
        self.assertEqual(canonical, self.output())
        self.assertEqual(diagnostics["row_count"], 2)
        self.assertEqual(diagnostics["distinct_normalized_response_count"], 2)

    def test_order_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID set or order"):
            author_import.validate_author_output(
                self.packet(), list(reversed(self.output())), self.gates()
            )

    def test_extra_field_fails(self) -> None:
        output = copy.deepcopy(self.output())
        output[0]["label"] = "good"
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            author_import.validate_author_output(self.packet(), output, self.gates())

    def test_word_limit_fails(self) -> None:
        output = copy.deepcopy(self.output())
        output[0]["authored_response"] = "too short"
        with self.assertRaisesRegex(ValueError, "word count"):
            author_import.validate_author_output(self.packet(), output, self.gates())

    def test_forbidden_tag_fails_case_insensitively(self) -> None:
        output = copy.deepcopy(self.output())
        output[0]["authored_response"] = "one safe <THINK> response here"
        with self.assertRaisesRegex(ValueError, "forbidden substring"):
            author_import.validate_author_output(self.packet(), output, self.gates())

    def test_duplicate_normalized_response_fails(self) -> None:
        output = copy.deepcopy(self.output())
        output[1]["authored_response"] = output[0]["authored_response"].upper()
        with self.assertRaisesRegex(ValueError, "not all distinct"):
            author_import.validate_author_output(self.packet(), output, self.gates())


if __name__ == "__main__":
    unittest.main()
