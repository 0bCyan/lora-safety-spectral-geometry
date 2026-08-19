"""Tests for OACPO D1 Reviewer C packet validation."""

from __future__ import annotations

import copy
import unittest

import step3_oacpo_d1_c_packets as c_packets


class OacpoD1CPacketTests(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "packet_design": {"expected_rows": {"safety": 1, "quality": 1, "structure": 1}},
            "allowed_fields": {
                "safety": ["clean_safe"],
                "quality": ["detail_retention", "safe_actionability", "naturalness"],
                "structure": ["mechanical_template"],
            },
        }

    def packets(self) -> dict:
        return {
            "safety_c_packet": [
                {"opaque_safety_id": "s", "prompt": "p", "response": "r", "disputed_fields": ["clean_safe"]}
            ],
            "safety_c_template": [{"opaque_safety_id": "s", "clean_safe": "", "short_reason": ""}],
            "quality_c_packet": [
                {
                    "opaque_quality_id": "q",
                    "prompt": "p",
                    "response": "r",
                    "disputed_fields": ["safe_actionability"],
                }
            ],
            "quality_c_template": [
                {
                    "opaque_quality_id": "q",
                    "detail_retention": None,
                    "safe_actionability": "",
                    "naturalness": None,
                    "short_reason": "",
                }
            ],
            "structure_c_packet": [
                {"opaque_structure_id": "t", "response": "r", "disputed_fields": ["mechanical_template"]}
            ],
            "structure_c_template": [
                {"opaque_structure_id": "t", "mechanical_template": "", "short_reason": ""}
            ],
        }

    def test_valid_packets_pass(self) -> None:
        c_packets.validate_packets(self.packets(), self.protocol())

    def test_private_field_leak_fails(self) -> None:
        packets = copy.deepcopy(self.packets())
        packets["safety_c_packet"][0]["response_role"] = "authored"
        with self.assertRaisesRegex(ValueError, "private field leaked"):
            c_packets.validate_packets(packets, self.protocol())

    def test_agreed_quality_field_must_be_immutable(self) -> None:
        packets = copy.deepcopy(self.packets())
        packets["quality_c_template"][0]["detail_retention"] = ""
        with self.assertRaisesRegex(ValueError, "mutability mismatch"):
            c_packets.validate_packets(packets, self.protocol())

    def test_invalid_disputed_field_fails(self) -> None:
        packets = copy.deepcopy(self.packets())
        packets["structure_c_packet"][0]["disputed_fields"] = ["source_role"]
        with self.assertRaisesRegex(ValueError, "invalid disputed field"):
            c_packets.validate_packets(packets, self.protocol())


if __name__ == "__main__":
    unittest.main()
