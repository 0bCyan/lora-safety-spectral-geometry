"""Synthetic tests for CSPR P2T zero-access reviewer qualification."""

from __future__ import annotations

import copy
import unittest

import step3_cspr_p2t_recovery as recovery


def protocol() -> dict:
    return {
        "prospective_reviewers": {
            "safety_a": {"runtime_requested_model_id": "gpt-5.5"},
            "safety_b": {"runtime_requested_model_id": "gpt-5.6-sol"},
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
        "self_reported_model_id": model,
        "fork_context": False,
        "prior_study_involvement": False,
        "repository_material_opened": False,
        "study_rows_accessed": 0,
        "other_reviewer_contact": False,
        "agrees_independent_complete_review": True,
        "agrees_no_partial_output": True,
        "agrees_no_external_retrieval": True,
        "agrees_forbidden_material_not_opened": True,
        "committed_packet_rows": 424,
        "completion_status": "qualified_zero_access",
    }


class AttestationTests(unittest.TestCase):
    def test_valid_cross_model_pair_passes(self) -> None:
        cfg = protocol()
        rows = [
            recovery.validate_attestation(
                attestation("safety_a", "agent-a", "session-a", "gpt-5.5"),
                "safety_a",
                cfg,
            ),
            recovery.validate_attestation(
                attestation("safety_b", "agent-b", "session-b", "gpt-5.6-sol"),
                "safety_b",
                cfg,
            ),
        ]
        recovery.validate_pair(rows, cfg)

    def test_study_access_before_freeze_is_rejected(self) -> None:
        value = attestation("safety_a", "agent-a", "session-a", "gpt-5.5")
        value["study_rows_accessed"] = 1
        with self.assertRaisesRegex(recovery.RecoveryValidationError, "study rows"):
            recovery.validate_attestation(value, "safety_a", protocol())

    def test_requested_model_mismatch_is_rejected(self) -> None:
        value = attestation("safety_a", "agent-a", "session-a", "gpt-5.6-sol")
        with self.assertRaisesRegex(recovery.RecoveryValidationError, "model mismatch"):
            recovery.validate_attestation(value, "safety_a", protocol())

    def test_prior_agent_is_rejected(self) -> None:
        value = attestation("safety_a", "old-agent", "session-a", "gpt-5.5")
        with self.assertRaisesRegex(recovery.RecoveryValidationError, "forbidden prior"):
            recovery.validate_attestation(value, "safety_a", protocol())

    def test_agent_or_session_reuse_is_rejected(self) -> None:
        cfg = protocol()
        base = attestation("safety_a", "agent-a", "session-a", "gpt-5.5")
        cases = [
            [base, attestation("safety_b", "agent-a", "session-b", "gpt-5.6-sol")],
            [base, attestation("safety_b", "agent-b", "session-a", "gpt-5.6-sol")],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(recovery.RecoveryValidationError, "reuse"):
                    recovery.validate_pair(rows, cfg)

    def test_same_model_pair_is_rejected(self) -> None:
        cfg = protocol()
        rows = [
            attestation("safety_a", "agent-a", "session-a", "gpt-5.5"),
            attestation("safety_b", "agent-b", "session-b", "gpt-5.5"),
        ]
        with self.assertRaisesRegex(recovery.RecoveryValidationError, "cross-model"):
            recovery.validate_pair(rows, cfg)

    def test_schema_expansion_is_rejected(self) -> None:
        value = attestation("safety_a", "agent-a", "session-a", "gpt-5.5")
        changed = copy.deepcopy(value)
        changed["unfrozen_note"] = "extra"
        with self.assertRaisesRegex(recovery.RecoveryValidationError, "schema"):
            recovery.validate_attestation(changed, "safety_a", protocol())


if __name__ == "__main__":
    unittest.main()
