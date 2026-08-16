"""Synthetic tests for CSPR P2U calibration-view preparation."""

from __future__ import annotations

import copy
import json
import unittest

import step3_cspr_p2u_external_measurement as p2u


def raw(row: dict) -> tuple[bytes, dict]:
    return (json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n", row)


def fixture() -> tuple[list, list, list]:
    packet = [
        raw({"opaque_id": "target", "prompt": "p0", "response": "r0"}),
        raw({"opaque_id": "cal-dup", "prompt": "p1", "response": "r1"}),
        raw({"opaque_id": "cal-primary", "prompt": "p2", "response": "r2"}),
    ]
    template = [
        raw({"opaque_id": row[1]["opaque_id"], "label": "", "short_reason": ""})
        for row in packet
    ]
    key = [
        raw(
            {
                "opaque_id": "cal-primary",
                "item_type": "calibration_anchor",
                "is_hidden_duplicate": False,
            }
        ),
        raw(
            {
                "opaque_id": "target",
                "item_type": "target_candidate",
                "is_hidden_duplicate": False,
            }
        ),
        raw(
            {
                "opaque_id": "cal-dup",
                "item_type": "calibration_anchor",
                "is_hidden_duplicate": True,
                "duplicate_of": "cal-primary",
            }
        ),
    ]
    return packet, template, key


class CalibrationSubsequenceTests(unittest.TestCase):
    def test_exact_ordered_subsequence_is_preserved(self) -> None:
        packet, template, key = fixture()
        packet_bytes, template_bytes, counts = p2u.calibration_subsequence(
            packet, template, key, "opaque_id", ("label",), 1, 1
        )
        self.assertEqual(packet_bytes, packet[1][0] + packet[2][0])
        self.assertEqual(template_bytes, template[1][0] + template[2][0])
        self.assertEqual(
            counts,
            {"primary_rows": 1, "hidden_duplicate_rows": 1, "total_rows": 2},
        )

    def test_target_row_is_not_exported(self) -> None:
        packet, template, key = fixture()
        packet_bytes, _, _ = p2u.calibration_subsequence(
            packet, template, key, "opaque_id", ("label",), 1, 1
        )
        self.assertNotIn(b'"opaque_id":"target"', packet_bytes)

    def test_id_set_mismatch_is_rejected(self) -> None:
        packet, template, key = fixture()
        key.pop()
        with self.assertRaisesRegex(p2u.P2UValidationError, "ID sets differ"):
            p2u.calibration_subsequence(
                packet, template, key, "opaque_id", ("label",), 1, 1
            )

    def test_packet_template_order_mismatch_is_rejected(self) -> None:
        packet, template, key = fixture()
        template[0], template[1] = template[1], template[0]
        with self.assertRaisesRegex(p2u.P2UValidationError, "order mismatch"):
            p2u.calibration_subsequence(
                packet, template, key, "opaque_id", ("label",), 1, 1
            )

    def test_duplicate_id_is_rejected(self) -> None:
        packet, template, key = fixture()
        packet[2] = copy.deepcopy(packet[1])
        with self.assertRaisesRegex(p2u.P2UValidationError, "duplicate"):
            p2u.calibration_subsequence(
                packet, template, key, "opaque_id", ("label",), 1, 1
            )

    def test_count_mismatch_is_rejected(self) -> None:
        packet, template, key = fixture()
        with self.assertRaisesRegex(p2u.P2UValidationError, "count mismatch"):
            p2u.calibration_subsequence(
                packet, template, key, "opaque_id", ("label",), 2, 0
            )

    def test_nonblank_template_is_rejected(self) -> None:
        packet, template, key = fixture()
        template[1] = raw(
            {"opaque_id": "cal-dup", "label": "yes", "short_reason": ""}
        )
        with self.assertRaisesRegex(p2u.P2UValidationError, "not blank"):
            p2u.calibration_subsequence(
                packet, template, key, "opaque_id", ("label",), 1, 1
            )

    def test_packet_schema_expansion_is_rejected(self) -> None:
        packet, template, key = fixture()
        packet[0] = raw(
            {
                "opaque_id": "target",
                "prompt": "p0",
                "response": "r0",
                "hidden": True,
            }
        )
        with self.assertRaisesRegex(p2u.P2UValidationError, "schema mismatch"):
            p2u.calibration_subsequence(
                packet, template, key, "opaque_id", ("label",), 1, 1
            )


if __name__ == "__main__":
    unittest.main()
