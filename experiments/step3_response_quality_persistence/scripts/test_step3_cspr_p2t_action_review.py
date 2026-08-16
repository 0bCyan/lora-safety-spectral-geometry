"""Synthetic tests for P2T Actionability execution gates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import step3_cspr_p2t_action_review as review


def execution() -> dict:
    return {
        "qualified_reviewers": {
            "a": {
                "agent_id": "agent-a",
                "session_id": "session-a",
                "runtime_requested_model_id": "gpt-5.5",
                "self_reported_model_id": "GPT-5 Codex",
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
        "canonical_line_endings": "LF",
    }


class ActionReviewTests(unittest.TestCase):
    def test_valid_cross_model_pair_passes(self) -> None:
        review.validate_identity([provenance("a"), provenance("b")], execution())

    def test_model_or_attestation_mismatch_is_rejected(self) -> None:
        for field, value in (
            ("runtime_requested_model_id", "wrong"),
            ("qualification_attestation_sha256", "wrong"),
        ):
            changed = provenance("a")
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(review.ActionReviewValidationError, field):
                    review.validate_identity([changed, provenance("b")], execution())

    def test_noncanonical_crlf_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=review.REPO) as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_bytes(b"{}\r\n")
            with self.assertRaisesRegex(review.ActionReviewValidationError, "line ending"):
                review.require_canonical_lf(path)

    def test_canonical_lf_passes(self) -> None:
        with tempfile.TemporaryDirectory(dir=review.REPO) as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_bytes(b"{}\n")
            review.require_canonical_lf(path)


if __name__ == "__main__":
    unittest.main()
