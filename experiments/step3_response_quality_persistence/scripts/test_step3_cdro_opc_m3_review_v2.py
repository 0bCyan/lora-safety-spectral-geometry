"""Tests for label-independent CDR-OPC M3 review repair V2."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import step3_cdro_opc_m3_finalize_review_v2 as finalizer
import step3_cdro_opc_m3_review_v2 as v2


def positive_row(candidate_id: str) -> dict[str, object]:
    return {
        "opaque_candidate_id": candidate_id,
        "detail_retention": 2,
        "safe_route": "yes",
        "executable_step": "yes",
        "prompt_binding": "yes",
        "naturalness": "yes",
        "clean_safe": "yes",
        "short_reason": "Specific, actionable, natural, and safe.",
    }


class M3ReviewV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = v2.load_registry()
        cls.p0, cls.prompts, cls.primary = v2.load_primary(cls.reg)

    def test_anchor_and_packet_denominators(self) -> None:
        bundle = v2.build_packets(self.reg, "a" * 40)
        self.assertEqual(len(bundle["primary"]), 120)
        self.assertEqual(len(bundle["anchors"]), 48)
        self.assertEqual(
            {family: sum(row["item_role"] == family for row in bundle["anchors"]) for family in v2.ANCHOR_FAMILIES},
            {family: 12 for family in v2.ANCHOR_FAMILIES},
        )
        for role in v2.REVIEWERS_AB:
            packet = bundle["packages"][role]["packet"]
            self.assertEqual(len(packet), 180)
            self.assertTrue(
                all(set(row) == {"opaque_candidate_id", "prompt", "response"} for row in packet)
            )

    def test_all_positive_primary_can_pass_reliability(self) -> None:
        primary = []
        a = {}
        b = {}
        for index in range(120):
            candidate_id = f"candidate-{index:03d}"
            row = {
                "opaque_candidate_id": candidate_id,
                "opaque_prompt_id": f"prompt-{index // 2:02d}",
            }
            primary.append(row)
            a[candidate_id] = positive_row(candidate_id)
            b[candidate_id] = positive_row(candidate_id)
        result = v2.primary_reliability(self.reg, primary, a, b, self.p0, "b" * 64)
        self.assertTrue(result["overall_pass"])
        self.assertTrue(result["negative_class_reported_not_gated"])
        self.assertEqual(result["candidate_qualified"]["no_ratings"], 0)

    def test_review_finalizer_accepts_c_reason_fields(self) -> None:
        packet = [
            {
                "opaque_candidate_id": "candidate",
                "prompt": "prompt",
                "response": "response",
                "reviewer_d_short_reason": "reason d",
                "reviewer_e_short_reason": "reason e",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            packet_path.write_text(json.dumps(packet[0]) + "\n", encoding="utf-8")
            draft = root / "draft.tsv"
            draft.write_text("2\tyes\tyes\tyes\tyes\tyes\tQualified.\n", encoding="utf-8")
            output = root / "review.jsonl"
            result = finalizer.finalize(packet_path, draft, output)
            self.assertEqual(result["rows"], 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["opaque_candidate_id"], "candidate")

    def test_attempt_001_row_level_inputs_are_forbidden(self) -> None:
        boundary = self.reg["attempt_001_boundary"]
        self.assertFalse(boundary["row_level_reviews_readable"])
        self.assertFalse(boundary["short_reasons_readable"])
        self.assertFalse(boundary["hidden_maps_readable"])
        self.assertFalse(boundary["labels_reusable"])
        self.assertTrue(boundary["aggregate_stop_report_only"])


if __name__ == "__main__":
    unittest.main()
