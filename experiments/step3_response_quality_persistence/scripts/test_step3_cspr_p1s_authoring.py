"""Synthetic tests for CSPR P1S authoring packet and mechanical validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import step3_cspr_p1s_authoring as authoring


def words(prefix: str, count: int = 45) -> str:
    return " ".join([prefix, *[f"word{index}" for index in range(count - 1)]])


class AuthoringPacketTests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "design": {"seed": 41, "prompt_rows": 2},
            "mechanical_validation": {"minimum_words": 40, "maximum_words": 160},
            "review_supply_gates": {
                "minimum_train_candidate_prompts": 1,
                "minimum_calibration_prompts": 0,
                "minimum_mechanism_dev_prompts": 0,
                "minimum_total_prompts": 1,
                "minimum_categories_with_at_least_two_prompts": 0,
            },
        }

    def source(self) -> list[dict]:
        return [
            {
                "pair_id": "pair-1",
                "source_action_id": "source-1",
                "prompt": "prompt one",
                "prompt_normalized_sha256": authoring.sha256_text("prompt one"),
                "prospective_split": "train_candidate",
                "assigned_category": "cat-a",
            },
            {
                "pair_id": "pair-2",
                "source_action_id": "source-2",
                "prompt": "prompt two",
                "prompt_normalized_sha256": authoring.sha256_text("prompt two"),
                "prospective_split": "calibration",
                "assigned_category": "cat-b",
            },
        ]

    def test_two_authors_receive_complementary_modes_without_private_fields(self) -> None:
        built = authoring.build_author_packets(self.source(), self.protocol())
        authoring.validate_packet_structure(built, self.protocol())
        self.assertEqual(len(built["author_a_packet"]), 2)
        self.assertEqual(len(built["author_b_packet"]), 2)
        self.assertTrue(
            all(
                set(row) == {"opaque_authoring_id", "prompt", "authoring_mode"}
                for row in built["author_a_packet"] + built["author_b_packet"]
            )
        )
        modes_by_pair = {}
        for row in built["hidden_key"]:
            modes_by_pair.setdefault(row["pair_id"], set()).add(row["authoring_mode"])
        self.assertTrue(all(modes == set(authoring.MODES) for modes in modes_by_pair.values()))

    def test_packet_construction_is_deterministic(self) -> None:
        forward = authoring.build_author_packets(self.source(), self.protocol())
        reverse = authoring.build_author_packets(list(reversed(self.source())), self.protocol())
        self.assertEqual(forward, reverse)

    def test_private_field_leak_is_rejected(self) -> None:
        built = authoring.build_author_packets(self.source(), self.protocol())
        built["author_a_packet"][0]["pair_id"] = "leak"
        with self.assertRaisesRegex(authoring.AuthoringValidationError, "private field leaked"):
            authoring.validate_packet_structure(built, self.protocol())

    def test_candidate_ledger_excludes_thinking_prior_and_duplicate_rows(self) -> None:
        built = authoring.build_author_packets(self.source(), self.protocol())
        with tempfile.TemporaryDirectory(dir=authoring.REPO) as directory:
            root = Path(directory)
            key_path = root / "key.jsonl"
            key_path.write_text(authoring.jsonl_text(built["hidden_key"]), encoding="utf-8")
            protocol = self.protocol()
            protocol["packet_outputs"] = {"hidden_key": authoring.relative(key_path)}
            ids_a = [row["opaque_authoring_id"] for row in built["author_a_packet"]]
            ids_b = [row["opaque_authoring_id"] for row in built["author_b_packet"]]
            duplicate_text = words("duplicate")
            prior_text = words("prior")
            authors = {
                "a": [
                    {"opaque_authoring_id": ids_a[0], "response": duplicate_text},
                    {"opaque_authoring_id": ids_a[1], "response": words("<think>bad</think>")},
                ],
                "b": [
                    {"opaque_authoring_id": ids_b[0], "response": duplicate_text},
                    {"opaque_authoring_id": ids_b[1], "response": prior_text},
                ],
            }
            prior_hash = authoring.sha256_text(authoring.normalize_text(prior_text))
            with patch.object(authoring, "prior_response_hashes", return_value={prior_hash}):
                ledger, analysis = authoring.candidate_ledger(authors, protocol)
        reason_counts = analysis["mechanical_exclusion_reason_counts"]
        self.assertEqual(reason_counts["thinking_tag_present"], 1)
        self.assertEqual(reason_counts["normalized_exact_prior_response_match"], 1)
        self.assertEqual(reason_counts["normalized_exact_authored_duplicate"], 1)
        self.assertEqual(analysis["mechanically_valid_candidates"], 1)
        self.assertEqual(sum(row["mechanically_valid"] for row in ledger), 1)

    def test_word_count_boundary_is_inclusive(self) -> None:
        built = authoring.build_author_packets(self.source(), self.protocol())
        with tempfile.TemporaryDirectory(dir=authoring.REPO) as directory:
            root = Path(directory)
            key_path = root / "key.jsonl"
            key_path.write_text(authoring.jsonl_text(built["hidden_key"]), encoding="utf-8")
            protocol = self.protocol()
            protocol["packet_outputs"] = {"hidden_key": authoring.relative(key_path)}
            all_rows = built["author_a_packet"] + built["author_b_packet"]
            authors = {
                "a": [
                    {"opaque_authoring_id": row["opaque_authoring_id"], "response": words(f"a{index}", 40)}
                    for index, row in enumerate(built["author_a_packet"])
                ],
                "b": [
                    {"opaque_authoring_id": row["opaque_authoring_id"], "response": words(f"b{index}", 160)}
                    for index, row in enumerate(built["author_b_packet"])
                ],
            }
            self.assertEqual(len(all_rows), 4)
            with patch.object(authoring, "prior_response_hashes", return_value=set()):
                _, analysis = authoring.candidate_ledger(authors, protocol)
        self.assertEqual(analysis["mechanically_valid_candidates"], 4)

    def test_source_upper_supply_uses_unique_prompts_not_candidate_rows(self) -> None:
        built = authoring.build_author_packets(self.source(), self.protocol())
        with tempfile.TemporaryDirectory(dir=authoring.REPO) as directory:
            root = Path(directory)
            key_path = root / "key.jsonl"
            key_path.write_text(authoring.jsonl_text(built["hidden_key"]), encoding="utf-8")
            protocol = self.protocol()
            protocol["packet_outputs"] = {"hidden_key": authoring.relative(key_path)}
            authors = {
                role: [
                    {
                        "opaque_authoring_id": row["opaque_authoring_id"],
                        "response": words(f"{role}{index}"),
                    }
                    for index, row in enumerate(built[f"author_{role}_packet"])
                ]
                for role in ("a", "b")
            }
            with patch.object(authoring, "prior_response_hashes", return_value=set()):
                _, analysis = authoring.candidate_ledger(authors, protocol)
        self.assertEqual(analysis["mechanically_valid_candidates"], 4)
        self.assertEqual(analysis["prompts_with_at_least_one_valid_candidate"], 2)


if __name__ == "__main__":
    unittest.main()
