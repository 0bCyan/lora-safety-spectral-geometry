"""Tests for the M8-A attempt-002 transport and finalizer."""

from __future__ import annotations

import unittest

import finalize_step3_cdro_opc_m8_authoring_attempt_002 as finalizer
import prepare_step3_cdro_opc_m8_authoring_attempt_002 as subject


class M8AuthoringAttempt002Tests(unittest.TestCase):
    def authoring(self) -> dict[str, object]:
        return {
            "responses_schema": [
                "request_id",
                "opaque_prompt_id",
                "sample_id",
                "author_mode",
                "response",
            ],
            "single_line_response": True,
            "minimum_response_whitespace_tokens": 40,
            "maximum_response_whitespace_tokens": 160,
            "required_openings": {
                "specific_redirect": "I can't help",
                "risk_mitigation": "I won't assist",
            },
        }

    def expected(self) -> dict[str, str]:
        return {
            "request_id": "m8_req_000",
            "opaque_prompt_id": "m8_prompt_001_hash",
            "sample_id": "risk_pkuqa_000001",
            "author_mode": "specific_redirect",
        }

    def actual(self) -> dict[str, str]:
        response = (
            "I can't help carry out that harmful request. Preserve the relevant "
            "records, stop the risky action, notify the responsible service owner, "
            "and use an authorized review process to address the underlying concern "
            "safely and transparently without targeting another person or system."
        )
        return {**self.expected(), "response": response}

    def test_valid_response_is_sealed(self) -> None:
        result = finalizer.validate_response_row(
            self.expected(), self.actual(), self.authoring(), set()
        )
        self.assertGreaterEqual(result["response_whitespace_tokens"], 40)
        self.assertEqual(len(result["response_sha256"]), 64)

    def test_metadata_change_is_rejected(self) -> None:
        actual = self.actual()
        actual["sample_id"] = "changed"
        with self.assertRaises(finalizer.M8AuthoringFinalizationError):
            finalizer.validate_response_row(
                self.expected(), actual, self.authoring(), set()
            )

    def test_duplicate_and_newline_are_rejected(self) -> None:
        hashes: set[str] = set()
        finalizer.validate_response_row(
            self.expected(), self.actual(), self.authoring(), hashes
        )
        with self.assertRaises(finalizer.M8AuthoringFinalizationError):
            finalizer.validate_response_row(
                self.expected(), self.actual(), self.authoring(), hashes
            )
        actual = self.actual()
        actual["response"] += "\n"
        with self.assertRaises(finalizer.M8AuthoringFinalizationError):
            finalizer.validate_response_row(
                self.expected(), actual, self.authoring(), set()
            )

    def test_permission_probe_hash_is_frozen(self) -> None:
        self.assertEqual(
            subject.sha256_file(
                subject.Path(
                    "C:/Users/1/Documents/Codex/2026-08-23/"
                    "apiidparam-input-192-id-invalid-id/work/"
                    "m8a2-permission-probe/probe.txt"
                )
            ),
            "a217158ef80bdfcfa68c5bf1a5b96cc011e4be4d82db8fb80adc7baa9139a49e",
        )


if __name__ == "__main__":
    unittest.main()
