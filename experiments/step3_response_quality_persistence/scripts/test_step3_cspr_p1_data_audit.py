"""Synthetic tests for the CSPR P1 public-source audit."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import step3_cspr_p1_data_audit as audit


class TextAuditTests(unittest.TestCase):
    def test_normalization_and_stable_rank_are_deterministic(self) -> None:
        self.assertEqual(audit.normalize_text("  A\n B  "), "a b")
        self.assertEqual(audit.stable_rank(7, "x"), audit.stable_rank(7, "x"))
        self.assertNotEqual(audit.stable_rank(7, "x"), audit.stable_rank(8, "x"))

    def test_lexical_similarity_detects_containment(self) -> None:
        short = audit.token_shingles("alpha beta gamma", 5)
        long = audit.token_shingles("zero alpha beta gamma four", 5)
        self.assertEqual(audit.lexical_similarity(short, long), 1.0)

    def test_indexed_similarity_matches_direct_similarity(self) -> None:
        texts = ["alpha beta gamma", "red green blue"]
        features, index, longest = audit.build_feature_index(texts, 5)
        observed = audit.maximum_indexed_similarity(
            "alpha beta gamma four", features, index, longest, 5
        )
        expected = max(
            audit.lexical_similarity(audit.token_shingles("alpha beta gamma four", 5), row)
            for row in features
        )
        self.assertEqual(observed, expected)


class CandidateSelectionTests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "source": {"expected_categories": ["a", "b"]},
            "selection": {
                "seed": 17,
                "prompt_count": 4,
                "calibration_per_category": 1,
                "development_per_category": 1,
            },
        }

    def candidates(self) -> list[dict]:
        rows = []
        for index in range(6):
            prompt_hash = f"{index + 1:064x}"
            safe_hash = f"{index + 101:064x}"
            unsafe_hash = f"{index + 201:064x}"
            rows.append(
                {
                    "prompt_normalized_sha256": prompt_hash,
                    "prompt_exact_sha256": prompt_hash,
                    "unsafe_categories": {"a" if index < 3 else "b"},
                    "safe": {
                        safe_hash: {
                            "source_index": index * 2 + 1,
                            "response_sha256": safe_hash,
                            "response_normalized_sha256": safe_hash,
                        }
                    },
                    "unsafe": {
                        unsafe_hash: {
                            "source_index": index * 2 + 2,
                            "response_sha256": unsafe_hash,
                            "response_normalized_sha256": unsafe_hash,
                        }
                    },
                }
            )
        return rows

    def test_selection_is_unique_balanced_and_prospective(self) -> None:
        selected = audit.select_pairs(self.candidates(), self.protocol())
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["prompt_normalized_sha256"] for row in selected}), 4)
        self.assertEqual({row["assigned_category"] for row in selected}, {"a", "b"})
        self.assertEqual(
            {row["prospective_split"] for row in selected},
            {"calibration", "mechanism_dev"},
        )
        self.assertTrue(
            all(
                row["chosen_response_normalized_sha256"]
                != row["rejected_response_normalized_sha256"]
                for row in selected
            )
        )

    def test_selection_is_byte_order_deterministic(self) -> None:
        forward = audit.select_pairs(self.candidates(), self.protocol())
        reverse = audit.select_pairs(list(reversed(self.candidates())), self.protocol())
        self.assertEqual(forward, reverse)

    def test_insufficient_category_supply_fails(self) -> None:
        candidates = [row for row in self.candidates() if "a" in row["unsafe_categories"]]
        with self.assertRaisesRegex(ValueError, "insufficient prompt supply"):
            audit.select_pairs(candidates, self.protocol())


class SourceIntegrityTests(unittest.TestCase):
    def test_cross_label_identical_response_cannot_form_a_pair(self) -> None:
        rows = [
            {
                "prompt": "same prompt",
                "response": "same response",
                "is_safe": True,
                "category": {"a": False},
            },
            {
                "prompt": "same prompt",
                "response": "same response",
                "is_safe": False,
                "category": {"a": True},
            },
        ]
        protocol = {
            "source": {
                "expected_rows": 2,
                "expected_categories": ["a"],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            candidates, inventory = audit.load_beaver_groups(path, protocol)
        self.assertEqual(inventory["mixed_safe_unsafe_prompts"], 1)
        self.assertEqual(inventory["cross_label_identical_responses_removed"], 1)
        self.assertEqual(inventory["pair_eligible_prompts_after_response_conflict_removal"], 0)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
