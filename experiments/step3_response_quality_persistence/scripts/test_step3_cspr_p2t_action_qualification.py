"""Synthetic tests for P2T Actionability zero-access qualification."""

from __future__ import annotations

import unittest

import step3_cspr_p2t_action_qualification as qualification


def protocol() -> dict:
    return {
        "prospective_reviewers": {
            "actionability_a": {"runtime_requested_model_id": "gpt-5.5"},
            "actionability_b": {"runtime_requested_model_id": "gpt-5.6-sol"},
        },
        "forbidden_prior_agent_ids": ["old-agent"],
    }


def attestation(role: str, agent: str, session: str, model: str) -> dict:
    return {
        "name": "step3_cspr_p2t_zero_access_reviewer_attestation_v1",
        "reviewer_role": role,
        "agent_id": agent,
        "session_id": session,
        "model_provider": "OpenAI",
        "runtime_requested_model_id": model,
        "self_reported_model_id": "GPT-5",
        "fork_context": False,
        "prior_study_involvement": False,
        "repository_material_opened": False,
        "study_rows_accessed": 0,
        "other_reviewer_contact": False,
        "agrees_independent_complete_review": True,
        "agrees_no_partial_output": True,
        "agrees_no_external_retrieval": True,
        "agrees_forbidden_material_not_opened": True,
        "committed_packet_rows": 312,
        "completion_status": "qualified_zero_access",
    }


class QualificationTests(unittest.TestCase):
    def test_valid_pair_passes(self) -> None:
        cfg = protocol()
        rows = [
            qualification.validate_attestation(
                attestation("actionability_a", "a", "sa", "gpt-5.5"),
                "actionability_a",
                cfg,
            ),
            qualification.validate_attestation(
                attestation("actionability_b", "b", "sb", "gpt-5.6-sol"),
                "actionability_b",
                cfg,
            ),
        ]
        qualification.validate_pair(rows)

    def test_wrong_row_commitment_is_rejected(self) -> None:
        row = attestation("actionability_a", "a", "sa", "gpt-5.5")
        row["committed_packet_rows"] = 424
        with self.assertRaisesRegex(qualification.ActionQualificationError, "packet"):
            qualification.validate_attestation(row, "actionability_a", protocol())

    def test_study_access_is_rejected(self) -> None:
        row = attestation("actionability_a", "a", "sa", "gpt-5.5")
        row["study_rows_accessed"] = 1
        with self.assertRaisesRegex(qualification.ActionQualificationError, "study rows"):
            qualification.validate_attestation(row, "actionability_a", protocol())

    def test_old_agent_is_rejected(self) -> None:
        row = attestation("actionability_a", "old-agent", "sa", "gpt-5.5")
        with self.assertRaisesRegex(qualification.ActionQualificationError, "prior"):
            qualification.validate_attestation(row, "actionability_a", protocol())

    def test_reuse_is_rejected(self) -> None:
        rows = [
            attestation("actionability_a", "same", "sa", "gpt-5.5"),
            attestation("actionability_b", "same", "sb", "gpt-5.6-sol"),
        ]
        with self.assertRaisesRegex(qualification.ActionQualificationError, "agent reuse"):
            qualification.validate_pair(rows)


if __name__ == "__main__":
    unittest.main()
