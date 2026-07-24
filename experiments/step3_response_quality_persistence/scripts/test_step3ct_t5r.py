from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import step3ct_t5r_common as common


def quality_row(opaque_id: str, detail: int, action: int, naturalness: str = "yes") -> dict:
    return {
        "opaque_quality_id": opaque_id,
        "detail_retention": detail,
        "safe_actionability": action,
        "naturalness": naturalness,
        "short_reason": "Synthetic fixture reason.",
    }


class Step3CTT5RTests(unittest.TestCase):
    def test_zero_variance_chance_metrics_are_undefined(self) -> None:
        nominal = common.nominal_metrics([("yes", "yes")] * 8, ["yes", "no"])
        ordinal = common.ordinal_metrics([(2, 2)] * 8)
        self.assertTrue(nominal["zero_variance"])
        self.assertIsNone(nominal["gwet_ac1"])
        self.assertTrue(ordinal["zero_variance"])
        self.assertIsNone(ordinal["gwet_ac2"])
        self.assertTrue(common.nominal_gate(nominal, 0.85, 0.70, 0.95))

    def test_agreement_metric_fixtures(self) -> None:
        perfect = common.nominal_metrics([("yes", "yes"), ("no", "no")], ["yes", "no"])
        one_step = common.ordinal_metrics([(0, 1), (1, 2), (2, 1), (1, 0)])
        self.assertEqual(perfect["raw_agreement"], 1.0)
        self.assertAlmostEqual(perfect["gwet_ac1"], 1.0)
        self.assertEqual(one_step["linear_weighted_agreement"], 0.5)
        self.assertEqual(one_step["two_level_disagreement_rate"], 0.0)

    def test_review_schema_and_id_order(self) -> None:
        template = [{
            "opaque_quality_id": "q1",
            "detail_retention": "",
            "safe_actionability": "",
            "naturalness": "",
            "short_reason": "",
        }]
        common.validate_review_rows("quality", [quality_row("q1", 2, 1)], template)
        with self.assertRaises(ValueError):
            common.validate_review_rows("quality", [quality_row("wrong", 2, 1)], template)
        with self.assertRaises(ValueError):
            common.validate_review_rows("quality", [quality_row("q1", 3, 1)], template)

    def test_duplicate_full_vector_consistency(self) -> None:
        rows = [
            quality_row("p1", 2, 2),
            quality_row("d1", 2, 2),
            quality_row("p2", 1, 1),
            quality_row("d2", 0, 1),
        ]
        duplicates = [
            {"primary_opaque_id": "p1", "duplicate_opaque_id": "d1"},
            {"primary_opaque_id": "p2", "duplicate_opaque_id": "d2"},
        ]
        result = common.duplicate_consistency("quality", rows, duplicates)
        self.assertEqual(result["full_vector_consistent_count"], 1)
        self.assertEqual(result["full_vector_consistency"], 0.5)

    def test_disagreement_packet_hides_reviews_and_consensus_rows(self) -> None:
        packet = [
            {"opaque_quality_id": "q1", "prompt": "synthetic prompt 1", "response": "synthetic response 1"},
            {"opaque_quality_id": "q2", "prompt": "synthetic prompt 2", "response": "synthetic response 2"},
        ]
        reviewer_a = [quality_row("q1", 2, 1), quality_row("q2", 1, 1)]
        reviewer_b = [quality_row("q1", 1, 1), quality_row("q2", 1, 1)]
        c_packet, c_template = common.build_disagreement_packet("quality", packet, reviewer_a, reviewer_b)
        self.assertEqual(len(c_packet), 1)
        self.assertEqual(c_packet[0]["disagreement_fields"], ["detail_retention"])
        self.assertEqual(set(c_packet[0]), {"opaque_quality_id", "prompt", "response", "disagreement_fields"})
        self.assertTrue(all(value == "" for key, value in c_template[0].items() if key != "opaque_quality_id"))

    def test_disagreement_packet_rejects_hidden_identity(self) -> None:
        packet = [{"opaque_quality_id": "q1", "prompt": "p", "response": "r", "arm": "hidden"}]
        with self.assertRaises(ValueError):
            common.build_disagreement_packet("quality", packet, [quality_row("q1", 2, 2)], [quality_row("q1", 1, 2)])

    def test_paired_bootstrap_preserves_pairing(self) -> None:
        left = {f"p{i}": float(i + 1) for i in range(6)}
        right = {f"p{i}": float(i) for i in range(6)}
        result = common.paired_bootstrap_difference(left, right, 100, 7)
        self.assertEqual(result["difference"], 1.0)
        self.assertEqual(result["ci95"], [1.0, 1.0])

    def test_exact_mcnemar_and_holm(self) -> None:
        left = {"a": True, "b": True, "c": False, "d": False}
        right = {"a": False, "b": False, "c": True, "d": False}
        result = common.exact_mcnemar(left, right)
        self.assertEqual(result["left_only"], 2)
        self.assertEqual(result["right_only"], 1)
        adjusted = common.holm_adjust({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.04)

    def test_quality_eligibility(self) -> None:
        self.assertEqual(common.quality_eligible(quality_row("q", 1, 1, "yes")), "yes")
        self.assertEqual(common.quality_eligible(quality_row("q", 0, 2, "yes")), "no")

    def test_provenance_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=common.REPO) as temp:
            root = Path(temp)
            packet = root / "packet.jsonl"
            template = root / "template.jsonl"
            rubric = root / "rubric.md"
            output = root / "output.jsonl"
            packet.write_text("{}\n", encoding="utf-8")
            template.write_text("{}\n", encoding="utf-8")
            rubric.write_text("synthetic rubric\n", encoding="utf-8")
            output.write_text("{}\n", encoding="utf-8")
            provenance = {
                "domain": "quality",
                "reviewer_role": "quality_a",
                "anonymized_reviewer_id": "reviewer-new",
                "reviewer_type": "external_human",
                "session_id": "session-new",
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T01:00:00Z",
                "packet_path": packet.relative_to(common.REPO).as_posix(),
                "packet_sha256": common.sha256_file(packet),
                "template_path": template.relative_to(common.REPO).as_posix(),
                "template_sha256": common.sha256_file(template),
                "rubric_path": rubric.relative_to(common.REPO).as_posix(),
                "rubric_sha256": common.sha256_file(rubric),
                "output_path": output.relative_to(common.REPO).as_posix(),
                "output_sha256": common.sha256_file(output),
                "output_rows": 1,
                "completion_status": "complete",
                "confirmations": {key: True for key in (
                    "fresh_context",
                    "visible_packet_and_rubric_only",
                    "no_repository_history",
                    "no_arm_or_checkpoint_identity",
                    "no_training_or_mechanism_results",
                    "no_prior_or_other_reviewer_labels",
                    "no_hidden_key_or_duplicate_links",
                    "no_external_retrieval",
                    "no_cross_reviewer_communication",
                    "unchanged_complete_packet",
                )},
            }
            common.validate_provenance(
                provenance,
                domain="quality",
                reviewer_role="quality_a",
                packet_path=packet,
                template_path=template,
                rubric_path=rubric,
                output_path=output,
                output_rows=1,
                prohibited_session_ids={"old-session"},
            )


if __name__ == "__main__":
    unittest.main()
