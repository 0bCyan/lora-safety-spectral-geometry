from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import record_step3ct_t5h_safety_freeze as safety_freeze


class Step3CTT5HSafetyFreezeTests(unittest.TestCase):
    def records(self) -> dict:
        return {
            "a": {"anonymized_reviewer_id": "reviewer-a", "session_id": "session-a", "rows": 630},
            "b": {"anonymized_reviewer_id": "reviewer-b", "session_id": "session-b", "rows": 630},
        }

    def test_complete_distinct_pair_passes(self) -> None:
        safety_freeze.validate_pair_records(self.records())

    def test_missing_reviewer_is_rejected(self) -> None:
        records = self.records()
        del records["b"]
        with self.assertRaises(ValueError):
            safety_freeze.validate_pair_records(records)

    def test_duplicate_session_is_rejected(self) -> None:
        records = self.records()
        records["b"]["session_id"] = "session-a"
        with self.assertRaises(ValueError):
            safety_freeze.validate_pair_records(records)

    def test_partial_row_count_is_rejected(self) -> None:
        records = self.records()
        records["b"]["rows"] = 629
        with self.assertRaises(ValueError):
            safety_freeze.validate_pair_records(records)


if __name__ == "__main__":
    unittest.main()
