from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import record_step3ct_t5h_waiting as waiting


class Step3CTT5HWaitingTests(unittest.TestCase):
    def registry(self) -> dict:
        return {
            "external_review": {"required_primary_reviewers": 2, "first_domain": "safety"},
        }

    def test_entry_requires_p6a1(self) -> None:
        with self.assertRaises(RuntimeError):
            waiting.validate_entry({"route": "P6A0_non_eol_content_or_git_blob_mismatch"}, self.registry())

    def test_entry_requires_two_safety_reviewers(self) -> None:
        self.assertEqual(waiting.validate_entry({"route": waiting.SUCCESS_P6A_ROUTE}, self.registry()), 2)

    def test_waiting_record_is_zero_access_and_not_method_result(self) -> None:
        record = waiting.build_qualification("a" * 40, 2, "b" * 64, "c" * 64, {})
        self.assertEqual(record["route"], waiting.WAITING_ROUTE)
        self.assertFalse(record["method_result"])
        self.assertEqual(record["qualified_reviewers"], 0)
        self.assertTrue(all(value is False or value == 0 for value in record["study_access"].values()))
        self.assertTrue(all(value == 0 for value in record["outputs"].values()))

    def test_writer_emits_lf_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            waiting.write_text_lf(path, "left\nright\n")
            self.assertEqual(path.read_bytes(), b"left\nright\n")


if __name__ == "__main__":
    unittest.main()
