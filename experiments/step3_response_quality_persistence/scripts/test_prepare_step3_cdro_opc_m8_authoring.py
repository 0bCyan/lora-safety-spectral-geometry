"""Tests for M8 independent-cohort packet preparation."""

from __future__ import annotations

import unittest

import prepare_step3_cdro_opc_m8_authoring as subject


class M8AuthoringPreparationTests(unittest.TestCase):
    def protocol(self) -> dict[str, object]:
        return {
            "schema": subject.SCHEMA,
            "status": subject.STATUS,
            "prompt_registry_sha256": "0" * 64,
            "selection": {
                "prompts": 60,
                "historical_unique_samples_expected": 120,
                "risk_rows_expected": 1000,
            },
            "authoring": {
                "requests": 120,
                "shards": 12,
                "rows_per_shard": 10,
                "modes": ["specific_redirect", "risk_mitigation"],
                "minimum_response_whitespace_tokens": 40,
                "maximum_response_whitespace_tokens": 160,
                "required_openings": {
                    "specific_redirect": "I can't help",
                    "risk_mitigation": "I won't assist",
                },
                "immutable_submission": True,
                "retry_or_reassignment_allowed": False,
                "prompt_replacement_or_top_up_allowed": False,
            },
            "authorization_boundary": {
                "candidate_generation_authorized": False,
                "checkpoint_writes_authorized": False,
                "core100_opened": False,
                "evaluated_model_generation_authorized": False,
                "heldout_opened": False,
                "optimizer_steps": 0,
                "result_dependent_rerun_authorized": False,
                "training_authorized": False,
                "transient_virtual_parameter_evaluations_authorized": False,
            },
        }

    def test_protocol_denominators_are_strict(self) -> None:
        protocol = self.protocol()
        subject.validate_protocol(protocol, allow_pending=False)
        protocol["authoring"]["requests"] = 119  # type: ignore[index]
        with self.assertRaises(subject.M8AuthoringError):
            subject.validate_protocol(protocol, allow_pending=False)

    def test_pending_hash_only_allowed_for_derivation(self) -> None:
        protocol = self.protocol()
        protocol["prompt_registry_sha256"] = subject.PENDING_HASH
        subject.validate_protocol(protocol, allow_pending=True)
        with self.assertRaises(subject.M8AuthoringError):
            subject.validate_protocol(protocol, allow_pending=False)

    def test_prompt_normalization_is_nfkc_casefolded_and_collapsed(self) -> None:
        self.assertEqual(
            subject.normalize_prompt("  Ｔest\tPROMPT  "),
            "test prompt",
        )

    def test_packet_assignment_is_complete_and_disjoint(self) -> None:
        protocol = self.protocol()
        registry = {
            "prompts": [
                {
                    "opaque_prompt_id": f"m8_prompt_{rank:03d}_hash",
                    "sample_id": f"sample_{rank:03d}",
                    "prompt": f"prompt {rank}",
                    "rank": rank,
                }
                for rank in range(1, 61)
            ]
        }
        shards = [subject.packet_rows(protocol, registry, index) for index in range(12)]
        self.assertTrue(all(len(rows) == 10 for rows in shards))
        requests = [row["request_id"] for rows in shards for row in rows]
        self.assertEqual(len(requests), 120)
        self.assertEqual(len(set(requests)), 120)
        self.assertEqual(set(requests), {f"m8_req_{index:03d}" for index in range(120)})


if __name__ == "__main__":
    unittest.main()

