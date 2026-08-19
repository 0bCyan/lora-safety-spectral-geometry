"""Tests for OACPO D1 decomposed blinded packet construction."""

from __future__ import annotations

import copy
import unittest

import step3_oacpo_d1_packets as d1


class OacpoD1PacketTests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "packet_design": {
                "seed": 31,
                "primary_triplets": 2,
                "safety_duplicates_per_role": 1,
                "quality_duplicates_per_role": 1,
                "structure_duplicates_per_role": 1,
            }
        }

    def triplets(self) -> list[dict]:
        return [
            {
                "pair_id": f"pair_{index}",
                "prompt": f"prompt {index}",
                "prospective_split": "train_candidate" if index == 0 else "calibration",
                "assigned_category": f"category_{index}",
                "authored": f"authored response {index}",
                "source_safe": f"source safe response {index}",
                "source_unsafe": f"source unsafe response {index}",
            }
            for index in range(2)
        ]

    def test_construct_row_counts_and_blinding(self) -> None:
        packets = d1.build_packets(self.triplets(), self.protocol())
        d1.validate_packets(packets, self.protocol())
        self.assertEqual(len(packets["safety_a_packet"]), 9)
        self.assertEqual(len(packets["quality_a_packet"]), 6)
        self.assertEqual(len(packets["structure_a_packet"]), 6)
        self.assertTrue(all(set(row) == {"opaque_structure_id", "response"} for row in packets["structure_a_packet"]))

    def test_reviewer_orders_differ_with_equal_row_sets(self) -> None:
        packets = d1.build_packets(self.triplets(), self.protocol())
        for construct, id_field in (
            ("safety", "opaque_safety_id"),
            ("quality", "opaque_quality_id"),
            ("structure", "opaque_structure_id"),
        ):
            a = [row[id_field] for row in packets[f"{construct}_a_packet"]]
            b = [row[id_field] for row in packets[f"{construct}_b_packet"]]
            self.assertEqual(set(a), set(b))
            self.assertNotEqual(a, b)

    def test_construction_is_deterministic(self) -> None:
        self.assertEqual(
            d1.build_packets(self.triplets(), self.protocol()),
            d1.build_packets(list(reversed(self.triplets())), self.protocol()),
        )

    def test_private_leak_is_rejected(self) -> None:
        packets = d1.build_packets(self.triplets(), self.protocol())
        tampered = copy.deepcopy(packets)
        tampered["quality_a_packet"][0]["response_role"] = "authored"
        with self.assertRaisesRegex(ValueError, "private field leaked"):
            d1.validate_packets(tampered, self.protocol())


if __name__ == "__main__":
    unittest.main()
