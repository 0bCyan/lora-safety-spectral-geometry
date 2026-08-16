"""Synthetic tests for CSPR P2S calibration-interleaved packets."""

from __future__ import annotations

import copy
import unittest

import step3_cspr_p2s_packets as packets


class P2SPacketTests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "packet_design": {
                "seed": 53,
                "target_candidate_rows": 4,
                "target_rejected_rows": 2,
                "action_calibration_primary_rows": 6,
                "safety_calibration_primary_rows": 4,
                "action_calibration_duplicates_per_score": 1,
                "action_target_duplicates": 2,
                "safety_calibration_duplicates_per_label": 1,
                "safety_target_duplicates_per_type": 1,
                "action_hidden_duplicates": 5,
                "safety_hidden_duplicates": 4,
                "action_packet_rows_per_reviewer": 15,
                "safety_packet_rows_per_reviewer": 14,
            }
        }

    def inputs(self) -> tuple[list[dict], dict[str, str], list[dict], list[dict], list[dict]]:
        candidates = []
        prompts = {}
        for pair_index in range(2):
            for role in ("a", "b"):
                authoring_id = f"author-{pair_index}-{role}"
                prompts[authoring_id] = f"prompt {pair_index}"
                candidates.append(
                    {
                        "opaque_authoring_id": authoring_id,
                        "pair_id": f"pair-{pair_index}",
                        "author_role": role,
                        "prospective_split": "train_candidate",
                        "assigned_category": f"cat-{pair_index}",
                        "response": f"candidate {pair_index} {role}",
                    }
                )
        rejected = [
            {
                "pair_id": f"pair-{index}",
                "prompt": f"prompt {index}",
                "response": f"rejected {index}",
                "prospective_split": "train_candidate",
                "assigned_category": f"cat-{index}",
                "source_safety_id": f"source-{index}",
            }
            for index in range(2)
        ]
        action_calibration = []
        for score in (0, 1, 2):
            for index in range(2):
                action_calibration.append(
                    {
                        "source_id": f"action-anchor-{score}-{index}",
                        "prompt": f"anchor prompt {score} {index}",
                        "response": f"anchor response {score} {index}",
                        "assigned_category": f"anchor-cat-{index}",
                        "reference_safe_route": "no" if score == 0 else "yes",
                        "reference_executable_step": "yes" if score == 2 else "no",
                        "reference_prompt_binding": "yes" if score == 2 else "no",
                        "reference_score": score,
                    }
                )
        safety_calibration = []
        for label in ("yes", "no"):
            for index in range(2):
                safety_calibration.append(
                    {
                        "source_id": f"safety-anchor-{label}-{index}",
                        "prompt": f"safety prompt {label} {index}",
                        "response": f"safety response {label} {index}",
                        "assigned_category": f"anchor-cat-{index}",
                        "reference_clean_safe": label,
                    }
                )
        return candidates, prompts, rejected, action_calibration, safety_calibration

    def build(self) -> dict[str, list[dict]]:
        return packets.build_packets(*self.inputs(), self.protocol())

    def test_primary_and_duplicate_composition_is_exact(self) -> None:
        built = self.build()
        packets.validate_packet_structure(built, self.protocol())
        action_primary = [row for row in built["action_key"] if not row["is_hidden_duplicate"]]
        safety_primary = [row for row in built["safety_key"] if not row["is_hidden_duplicate"]]
        self.assertEqual(len(action_primary), 10)
        self.assertEqual(len(built["action_key"]), 15)
        self.assertEqual(len(safety_primary), 10)
        self.assertEqual(len(built["safety_key"]), 14)

    def test_visible_packets_hide_identity_reference_and_duplicate_fields(self) -> None:
        built = self.build()
        self.assertTrue(
            all(
                set(row) == {"opaque_action_id", "prompt", "response"}
                for row in built["action_a_packet"]
            )
        )
        self.assertTrue(
            all(
                set(row) == {"opaque_safety_id", "prompt", "response"}
                for row in built["safety_a_packet"]
            )
        )

    def test_reviewer_orders_differ_but_row_sets_match(self) -> None:
        built = self.build()
        action_a = [row["opaque_action_id"] for row in built["action_a_packet"]]
        action_b = [row["opaque_action_id"] for row in built["action_b_packet"]]
        safety_a = [row["opaque_safety_id"] for row in built["safety_a_packet"]]
        safety_b = [row["opaque_safety_id"] for row in built["safety_b_packet"]]
        self.assertEqual(set(action_a), set(action_b))
        self.assertEqual(set(safety_a), set(safety_b))
        self.assertNotEqual(action_a, action_b)
        self.assertNotEqual(safety_a, safety_b)

    def test_packet_construction_is_deterministic(self) -> None:
        first = self.build()
        candidates, prompts, rejected, action, safety = self.inputs()
        second = packets.build_packets(
            list(reversed(candidates)),
            prompts,
            list(reversed(rejected)),
            list(reversed(action)),
            list(reversed(safety)),
            self.protocol(),
        )
        self.assertEqual(first, second)

    def test_duplicate_rows_copy_visible_content_exactly(self) -> None:
        built = self.build()
        packet = {row["opaque_action_id"]: row for row in built["action_a_packet"]}
        for key in built["action_key"]:
            if not key["is_hidden_duplicate"]:
                continue
            duplicate = packet[key["opaque_action_id"]]
            primary = packet[key["duplicate_of"]]
            self.assertEqual(duplicate["prompt"], primary["prompt"])
            self.assertEqual(duplicate["response"], primary["response"])

    def test_category_diverse_selection_is_group_balanced(self) -> None:
        rows = [
            {"id": f"{group}-{index}", "group": group, "category": f"cat-{index % 3}"}
            for group in (0, 1, 2)
            for index in range(6)
        ]
        selected = packets.select_category_diverse(
            rows,
            group_field="group",
            groups=[0, 1, 2],
            per_group=3,
            id_field="id",
            category_field="category",
            seed=7,
            namespace="test",
        )
        self.assertEqual(len(selected), 9)
        self.assertEqual({group: sum(row["group"] == group for row in selected) for group in (0, 1, 2)}, {0: 3, 1: 3, 2: 3})
        for group in (0, 1, 2):
            self.assertEqual(len({row["category"] for row in selected if row["group"] == group}), 3)

    def test_private_field_tampering_is_detectable(self) -> None:
        built = self.build()
        tampered = copy.deepcopy(built)
        tampered["action_a_packet"][0]["item_type"] = "target_candidate"
        with self.assertRaisesRegex(packets.PacketValidationError, "private field leaked"):
            packets.validate_packet_structure(tampered, self.protocol())


if __name__ == "__main__":
    unittest.main()
