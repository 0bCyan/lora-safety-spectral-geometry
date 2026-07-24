from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import record_step3ct_t5h_model_qualification as qualification


class Step3CTT5HModelQualificationTests(unittest.TestCase):
    def reviewer(self, role: str, session: str) -> dict:
        return {
            "reviewer_role": f"safety_{role}",
            "anonymized_reviewer_id": f"reviewer_{role}",
            "session_id": session,
            "reviewer_type": "policy_compliant_model_session",
            "model_provider": "OpenAI",
            "runtime_requested_model_id": "model-v1",
            "self_reported_model_id": "model-v1",
            "model_id_version_for_final_provenance": "model-v1",
            "fork_context": False,
            "qualification_confirmations": {
                "prior_study_involvement": False,
                "repository_access_so_far": False,
                "study_rows_accessed": 0,
                "other_reviewer_contact": False,
                "agrees_independent_complete_630_review": True,
                "agrees_no_partial_output": True,
                "agrees_no_external_retrieval": True,
            },
        }

    def test_valid_reviewer_passes(self) -> None:
        qualification.validate_reviewer("a", self.reviewer("a", "fresh-session"), set())

    def test_failed_session_reuse_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("a", self.reviewer("a", "failed-session"), {"failed-session"})

    def test_context_inheritance_is_rejected(self) -> None:
        reviewer = self.reviewer("a", "fresh-session")
        reviewer["fork_context"] = True
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("a", reviewer, set())

    def test_prequalification_study_access_is_rejected(self) -> None:
        reviewer = self.reviewer("a", "fresh-session")
        reviewer["qualification_confirmations"]["study_rows_accessed"] = 1
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("a", reviewer, set())

    def test_partial_output_contract_is_required(self) -> None:
        reviewer = self.reviewer("a", "fresh-session")
        reviewer["qualification_confirmations"]["agrees_no_partial_output"] = False
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("a", reviewer, set())


if __name__ == "__main__":
    unittest.main()
