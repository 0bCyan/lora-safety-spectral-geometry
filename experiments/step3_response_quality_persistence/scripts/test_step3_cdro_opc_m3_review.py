"""Tests for CDR-OPC M3 blinded review packet and transport."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import step3_cdro_opc_m3_finalize_review as finalizer
import step3_cdro_opc_m3_review as review


class M3ReviewTests(unittest.TestCase):
    def test_frozen_packet_denominators_and_blinding(self) -> None:
        reg = review.load_registry()
        bundle = review.packet_bundle(reg, "a" * 40)
        self.assertEqual(len(bundle["primary"]), 120)
        for role in review.ROLES:
            packet = bundle["packages"][role]["packet"]
            hidden = bundle["packages"][role]["hidden_map"]
            self.assertEqual(len(packet), 156)
            self.assertEqual(len(hidden), 156)
            self.assertTrue(
                all(set(row) == {"opaque_candidate_id", "prompt", "response"} for row in packet)
            )

    def test_review_finalizer_encodes_ordered_rows(self) -> None:
        packet = [
            {"opaque_candidate_id": f"candidate-{index}", "prompt": "p", "response": "r"}
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            packet_path.write_text(
                "".join(json.dumps(row) + "\n" for row in packet), encoding="utf-8"
            )
            draft = root / "review.tsv"
            draft.write_text(
                "2\tyes\tyes\tyes\tyes\tyes\tSpecific and safe.\n" * 3,
                encoding="utf-8",
            )
            output = root / "review.jsonl"
            result = finalizer.finalize(packet_path, draft, output)
            self.assertEqual(result["rows"], 3)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["opaque_candidate_id"] for row in rows], [row["opaque_candidate_id"] for row in packet])
            with self.assertRaisesRegex(finalizer.ReviewFinalizeError, "already exists"):
                finalizer.finalize(packet_path, draft, output)

    def test_review_finalizer_rejects_wrong_row_count(self) -> None:
        packet = [{"opaque_candidate_id": "candidate", "prompt": "p", "response": "r"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            packet_path.write_text(json.dumps(packet[0]) + "\n", encoding="utf-8")
            draft = root / "review.tsv"
            draft.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(finalizer.ReviewFinalizeError, "expected 1"):
                finalizer.finalize(packet_path, draft, root / "output.jsonl")


if __name__ == "__main__":
    unittest.main()
