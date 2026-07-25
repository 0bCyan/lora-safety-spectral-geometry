from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import record_step3ct_t5h_quality_utility_qualification as qualification


class Step3CTT5HQualityUtilityQualificationTests(unittest.TestCase):
    def reviewer(self, role: str, session: str) -> dict:
        return {
            "reviewer_role": role,
            "anonymized_reviewer_id": f"reviewer-{role}",
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
                "agrees_independent_complete_330_review": True,
                "agrees_no_partial_output": True,
                "agrees_no_external_retrieval": True,
            },
        }

    def test_valid_reviewer_passes(self) -> None:
        qualification.validate_reviewer("quality_a", self.reviewer("quality_a", "fresh"), set())

    def test_safety_session_reuse_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("quality_a", self.reviewer("quality_a", "safety-session"), {"safety-session"})

    def test_wrong_domain_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("utility_a", self.reviewer("quality_a", "fresh"), set())

    def test_context_inheritance_is_rejected(self) -> None:
        reviewer = self.reviewer("quality_a", "fresh")
        reviewer["fork_context"] = True
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("quality_a", reviewer, set())

    def test_prequalification_study_access_is_rejected(self) -> None:
        reviewer = self.reviewer("utility_b", "fresh")
        reviewer["qualification_confirmations"]["study_rows_accessed"] = 1
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("utility_b", reviewer, set())

    def test_complete_or_abort_contract_is_required(self) -> None:
        reviewer = self.reviewer("quality_b", "fresh")
        reviewer["qualification_confirmations"]["agrees_no_partial_output"] = False
        with self.assertRaises(ValueError):
            qualification.validate_reviewer("quality_b", reviewer, set())


if __name__ == "__main__":
    unittest.main()
