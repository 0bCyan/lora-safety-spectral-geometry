"""Synthetic tests for CSPR P2T qualified Safety reviewer identity gates."""

from __future__ import annotations

import copy
import unittest

import step3_cspr_p2t_safety_review as review


def execution() -> dict:
    return {
        "qualified_reviewers": {
            "a": {
                "agent_id": "agent-a",
                "session_id": "session-a",
                "runtime_requested_model_id": "gpt-5.5",
                "self_reported_model_id": "GPT-5",
                "qualification_attestation_sha256": "hash-a",
            },
            "b": {
                "agent_id": "agent-b",
                "session_id": "session-b",
                "runtime_requested_model_id": "gpt-5.6-sol",
                "self_reported_model_id": "GPT-5",
                "qualification_attestation_sha256": "hash-b",
            },
        },
        "forbidden_prior_agent_ids": ["old-agent"],
    }


def provenance(role: str) -> dict:
    expected = execution()["qualified_reviewers"][role]
    return {
        "reviewer_role": role,
        "agent_id": expected["agent_id"],
        "session_id": expected["session_id"],
        "model_provider": "OpenAI",
        "runtime_requested_model_id": expected["runtime_requested_model_id"],
        "self_reported_model_id": expected["self_reported_model_id"],
        "qualification_attestation_sha256": expected[
            "qualification_attestation_sha256"
        ],
        "complete_attempt_index": 1,
        "packet_access_after_qualification_push": True,
    }


class IdentityTests(unittest.TestCase):
    def test_qualified_cross_model_pair_passes(self) -> None:
        review.validate_identity([provenance("a"), provenance("b")], execution())

    def test_agent_mismatch_is_rejected(self) -> None:
        changed = provenance("a")
        changed["agent_id"] = "new-agent"
        with self.assertRaisesRegex(review.SafetyReviewValidationError, "agent_id"):
            review.validate_identity([changed, provenance("b")], execution())

    def test_model_mismatch_is_rejected(self) -> None:
        changed = provenance("b")
        changed["runtime_requested_model_id"] = "gpt-5.5"
        with self.assertRaisesRegex(review.SafetyReviewValidationError, "model"):
            review.validate_identity([provenance("a"), changed], execution())

    def test_attestation_hash_mismatch_is_rejected(self) -> None:
        changed = provenance("a")
        changed["qualification_attestation_sha256"] = "wrong"
        with self.assertRaisesRegex(review.SafetyReviewValidationError, "attestation"):
            review.validate_identity([changed, provenance("b")], execution())

    def test_prior_agent_is_rejected(self) -> None:
        cfg = execution()
        cfg["qualified_reviewers"]["a"]["agent_id"] = "old-agent"
        changed = provenance("a")
        changed["agent_id"] = "old-agent"
        with self.assertRaisesRegex(review.SafetyReviewValidationError, "old P2"):
            review.validate_identity([changed, provenance("b")], cfg)

    def test_missing_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(review.SafetyReviewValidationError, "roles"):
            review.validate_identity([provenance("a"), copy.deepcopy(provenance("a"))], execution())


if __name__ == "__main__":
    unittest.main()
