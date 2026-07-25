from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import record_step3ct_t5h_quality_utility_freeze as h3_freeze


class Step3CTT5HQualityUtilityFreezeTests(unittest.TestCase):
    def domains(self) -> dict:
        result = {}
        for domain, rows in h3_freeze.DOMAIN_ROWS.items():
            result[domain] = {
                "reviewers": {
                    role: {
                        "anonymized_reviewer_id": f"{domain}-{role}",
                        "session_id": f"session-{domain}-{role}",
                        "rows": rows,
                    }
                    for role in ("a", "b")
                }
            }
        return result

    def test_complete_all_domain_records_pass(self) -> None:
        h3_freeze.validate_all_records(self.domains())

    def test_missing_domain_is_rejected(self) -> None:
        domains = self.domains()
        del domains["utility"]
        with self.assertRaises(ValueError):
            h3_freeze.validate_all_records(domains)

    def test_partial_quality_rows_are_rejected(self) -> None:
        domains = self.domains()
        domains["quality"]["reviewers"]["b"]["rows"] = 329
        with self.assertRaises(ValueError):
            h3_freeze.validate_all_records(domains)

    def test_cross_domain_session_reuse_is_rejected(self) -> None:
        domains = self.domains()
        domains["utility"]["reviewers"]["a"]["session_id"] = "session-quality-a"
        with self.assertRaises(ValueError):
            h3_freeze.validate_all_records(domains)


if __name__ == "__main__":
    unittest.main()
