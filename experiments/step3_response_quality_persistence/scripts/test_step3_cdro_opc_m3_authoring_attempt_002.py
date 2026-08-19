"""Tests for the CDR-OPC M3 attempt 002 serialization repair."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import step3_cdro_opc_m3_authoring_attempt_002 as transport
import step3_cdro_opc_m3_finalize_worker as finalizer


def safe_response(index: int) -> str:
    body = ["I", "can't", "help", "with", 'the', 'quoted', '"request"', "directly."]
    body.extend(f"word{index}_{offset}" for offset in range(52))
    return " ".join(body)


class Attempt002Tests(unittest.TestCase):
    def test_frozen_transport_denominators(self) -> None:
        reg = transport.load_registry()
        _, requests = transport.source_bundle(reg)
        assignments = transport.assignment_rows(reg, requests)
        self.assertEqual(len(assignments), 12)
        self.assertEqual(sum(map(len, assignments.values())), 120)
        self.assertEqual({len(rows) for rows in assignments.values()}, {10})

    def test_finalizer_escapes_prose_and_is_immutable(self) -> None:
        assignment = [
            {
                "opaque_request_id": f"request-{index:02d}",
                "author_mode": "specific_redirect",
                "prompt": f"prompt {index}",
            }
            for index in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignment_path = root / "assignment.jsonl"
            assignment_path.write_text(
                "".join(json.dumps(row) + "\n" for row in assignment), encoding="utf-8"
            )
            drafts = root / "drafts"
            drafts.mkdir()
            for index in range(1, 11):
                (drafts / f"{index:02d}.txt").write_text(safe_response(index), encoding="utf-8")
            output = root / "submission.jsonl"
            result = finalizer.finalize(assignment_path, drafts, output)
            self.assertEqual(result["rows"], 10)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertIn('"request"', rows[0]["response"])
            with self.assertRaisesRegex(finalizer.FinalizeError, "already exists"):
                finalizer.finalize(assignment_path, drafts, output)

    def test_finalizer_rejects_extra_draft(self) -> None:
        assignment = [
            {
                "opaque_request_id": f"request-{index:02d}",
                "author_mode": "specific_redirect",
            }
            for index in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drafts = root / "drafts"
            drafts.mkdir()
            for index in range(1, 12):
                (drafts / f"{index:02d}.txt").write_text(safe_response(index), encoding="utf-8")
            with self.assertRaisesRegex(finalizer.FinalizeError, "file set"):
                finalizer.build_submission(assignment, drafts)


if __name__ == "__main__":
    unittest.main()
