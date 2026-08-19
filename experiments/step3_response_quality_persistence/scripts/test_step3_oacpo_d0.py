"""Tests for OACPO D0 source selection and author packet construction."""

from __future__ import annotations

import copy
import unittest

import step3_oacpo_d0 as d0


class OacpoD0Tests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "selection_design": {
                "seed": 17,
                "category_count": 2,
                "per_split_per_category": {
                    "train_candidate": 2,
                    "calibration": 1,
                    "mechanism_dev": 1,
                },
                "selected_pair_count": 8,
                "expected_split_counts": {
                    "train_candidate": 4,
                    "calibration": 2,
                    "mechanism_dev": 2,
                },
                "per_category_total": 4,
            },
            "author_packet_design": {"seed": 19},
        }

    def pairs(self) -> list[dict]:
        rows = []
        for split, supply in (("train_candidate", 3), ("calibration", 2), ("mechanism_dev", 2)):
            for category in ("a", "b"):
                for index in range(supply):
                    prompt = f"{split} {category} prompt {index}"
                    rows.append(
                        {
                            "pair_id": f"{split}_{category}_{index}",
                            "prospective_split": split,
                            "assigned_category": category,
                            "prompt": prompt,
                            "prompt_normalized_sha256": d0.sha256_text(d0.normalize_text(prompt)),
                            "chosen_source_index": index + 1,
                            "rejected_source_index": index + 101,
                            "chosen_response_normalized_sha256": f"chosen_{split}_{category}_{index}",
                            "rejected_response_normalized_sha256": f"rejected_{split}_{category}_{index}",
                        }
                    )
        return rows

    def test_selection_is_balanced_and_deterministic(self) -> None:
        selected = d0.select_pairs(self.pairs(), self.protocol())
        reversed_selected = d0.select_pairs(list(reversed(self.pairs())), self.protocol())
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(len(selected), 8)

    def test_author_packet_hides_source_roles_and_splits(self) -> None:
        selected = d0.select_pairs(self.pairs(), self.protocol())
        artifacts = d0.build_author_artifacts(selected, self.protocol())
        d0.validate_author_artifacts(artifacts, self.protocol())
        self.assertTrue(all(set(row) == {"author_id", "prompt"} for row in artifacts["author_packet"]))

    def test_author_packet_order_matches_blank_template(self) -> None:
        selected = d0.select_pairs(self.pairs(), self.protocol())
        artifacts = d0.build_author_artifacts(selected, self.protocol())
        self.assertEqual(
            [row["author_id"] for row in artifacts["author_packet"]],
            [row["author_id"] for row in artifacts["author_template"]],
        )
        self.assertTrue(all(row["authored_response"] == "" for row in artifacts["author_template"]))

    def test_private_leak_is_rejected(self) -> None:
        selected = d0.select_pairs(self.pairs(), self.protocol())
        artifacts = d0.build_author_artifacts(selected, self.protocol())
        tampered = copy.deepcopy(artifacts)
        tampered["author_packet"][0]["prospective_split"] = "train_candidate"
        with self.assertRaisesRegex(ValueError, "private field leaked"):
            d0.validate_author_artifacts(tampered, self.protocol())


if __name__ == "__main__":
    unittest.main()
