"""Synthetic tests for CSPR P2 packet construction."""

from __future__ import annotations

import copy
import unittest

import step3_cspr_p2_packets as packets


class PacketConstructionTests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "packet_design": {
                "seed": 31,
                "primary_pairs": 2,
                "safety_duplicates_per_role": 1,
                "actionability_duplicates": 1,
            }
        }

    def pairs(self) -> list[dict]:
        rows = []
        for index in range(2):
            prompt = f"prompt {index}"
            chosen = f"chosen {index}"
            rejected = f"rejected {index}"
            rows.append(
                {
                    "pair_id": f"pair_{index}",
                    "assigned_category": f"category_{index}",
                    "prospective_split": "train_candidate" if index == 0 else "calibration",
                    "prompt": prompt,
                    "prompt_normalized_sha256": packets.sha256_text(packets.normalize_text(prompt)),
                    "chosen_source_index": 10 + index,
                    "chosen_response": chosen,
                    "chosen_response_normalized_sha256": packets.sha256_text(
                        packets.normalize_text(chosen)
                    ),
                    "rejected_source_index": 20 + index,
                    "rejected_response": rejected,
                    "rejected_response_normalized_sha256": packets.sha256_text(
                        packets.normalize_text(rejected)
                    ),
                }
            )
        return rows

    def test_packets_hide_private_fields_and_balance_duplicates(self) -> None:
        built = packets.build_packets(self.pairs(), self.protocol())
        packets.validate_packet_structure(built, self.protocol())
        self.assertEqual(len(built["safety_a_packet"]), 6)
        self.assertEqual(len(built["action_a_packet"]), 3)
        self.assertTrue(all(set(row) == {"opaque_safety_id", "prompt", "response"} for row in built["safety_a_packet"]))
        self.assertTrue(all(set(row) == {"opaque_action_id", "prompt", "response"} for row in built["action_a_packet"]))

    def test_reviewer_orders_differ_but_row_sets_match(self) -> None:
        built = packets.build_packets(self.pairs(), self.protocol())
        safety_a = [row["opaque_safety_id"] for row in built["safety_a_packet"]]
        safety_b = [row["opaque_safety_id"] for row in built["safety_b_packet"]]
        action_a = [row["opaque_action_id"] for row in built["action_a_packet"]]
        action_b = [row["opaque_action_id"] for row in built["action_b_packet"]]
        self.assertEqual(set(safety_a), set(safety_b))
        self.assertEqual(set(action_a), set(action_b))
        self.assertNotEqual(safety_a, safety_b)
        self.assertNotEqual(action_a, action_b)

    def test_packet_construction_is_deterministic(self) -> None:
        self.assertEqual(
            packets.build_packets(self.pairs(), self.protocol()),
            packets.build_packets(list(reversed(self.pairs())), self.protocol()),
        )

    def test_private_field_leak_is_rejected(self) -> None:
        built = packets.build_packets(self.pairs(), self.protocol())
        tampered = copy.deepcopy(built)
        tampered["safety_a_packet"][0]["response_role"] = "chosen"
        with self.assertRaisesRegex(ValueError, "private field leaked"):
            packets.validate_packet_structure(tampered, self.protocol())


if __name__ == "__main__":
    unittest.main()
