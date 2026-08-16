"""Tests for the conservative CSPR P1R empty-response repair."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import step3_cspr_p1r_data_audit as repair


class EmptyResponseRepairTests(unittest.TestCase):
    def write_source(self, rows: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "source.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def row(self, prompt: str, response: str, safe: bool) -> dict:
        return {
            "prompt": prompt,
            "response": response,
            "is_safe": safe,
            "category": {"a": not safe},
        }

    def test_empty_response_is_counted_but_cannot_form_pair(self) -> None:
        rows = [
            self.row("eligible prompt", "safe answer", True),
            self.row("eligible prompt", "", False),
            self.row("eligible prompt", "unsafe answer", False),
            self.row("ineligible prompt", "safe answer", True),
            self.row("ineligible prompt", "   ", False),
        ]
        candidates, inventory = repair.load_beaver_groups(
            self.write_source(rows),
            {"source": {"expected_rows": 5, "expected_categories": ["a"]}},
        )
        self.assertEqual(inventory["rows"], 5)
        self.assertEqual(inventory["empty_response_rows_excluded"], 2)
        self.assertEqual(inventory["empty_unsafe_response_rows_excluded"], 2)
        self.assertEqual(inventory["mixed_safe_unsafe_prompts"], 1)
        self.assertEqual(len(candidates), 1)

    def test_empty_safe_response_does_not_create_safe_side(self) -> None:
        rows = [
            self.row("prompt", "", True),
            self.row("prompt", "unsafe answer", False),
        ]
        candidates, inventory = repair.load_beaver_groups(
            self.write_source(rows),
            {"source": {"expected_rows": 2, "expected_categories": ["a"]}},
        )
        self.assertEqual(inventory["empty_safe_response_rows_excluded"], 1)
        self.assertEqual(inventory["mixed_safe_unsafe_prompts"], 0)
        self.assertEqual(candidates, [])

    def test_empty_prompt_remains_a_schema_failure(self) -> None:
        rows = [self.row(" ", "response", True)]
        with self.assertRaisesRegex(ValueError, "empty prompt"):
            repair.load_beaver_groups(
                self.write_source(rows),
                {"source": {"expected_rows": 1, "expected_categories": ["a"]}},
            )


if __name__ == "__main__":
    unittest.main()
