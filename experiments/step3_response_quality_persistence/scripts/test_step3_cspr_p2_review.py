"""Synthetic tests for frozen CSPR P2 review analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import step3_cspr_p2_review as review


def action_row(row_id: str, safe: str, step: str, binding: str) -> dict:
    return {
        "opaque_action_id": row_id,
        "safe_route": safe,
        "executable_step": step,
        "prompt_binding": binding,
        "short_reason": "synthetic reason",
    }


class MetricTests(unittest.TestCase):
    def test_nominal_metrics_include_positive_and_negative_agreement(self) -> None:
        pairs = [("yes", "yes"), ("no", "no"), ("yes", "no"), ("no", "yes")]
        metrics = review.nominal_metrics(pairs)
        self.assertEqual(metrics["raw_agreement"], 0.5)
        self.assertEqual(metrics["positive_agreement"], 0.5)
        self.assertEqual(metrics["negative_agreement"], 0.5)

    def test_gwet_coefficients_handle_perfect_and_zero_variance_reviews(self) -> None:
        self.assertEqual(review.gwet_ac1([("yes", "yes")] * 8, ["yes", "no"]), 1.0)
        self.assertEqual(review.gwet_ac2([(2, 2)] * 8, [0, 1, 2]), 1.0)

    def test_derived_score_matches_frozen_three_level_rule(self) -> None:
        self.assertEqual(review.derived_score(action_row("a", "no", "yes", "yes")), 0)
        self.assertEqual(review.derived_score(action_row("b", "yes", "no", "yes")), 1)
        self.assertEqual(review.derived_score(action_row("c", "yes", "yes", "yes")), 2)

    def test_bootstrap_is_deterministic(self) -> None:
        pairs = [(0, 0), (1, 1), (2, 2), (1, 2)] * 5
        first = review.bootstrap_interval(
            pairs, lambda rows: review.gwet_ac2(rows, [0, 1, 2]), replicates=200, seed=17
        )
        second = review.bootstrap_interval(
            pairs, lambda rows: review.gwet_ac2(rows, [0, 1, 2]), replicates=200, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["finite_replicates"], 200)


class StructureTests(unittest.TestCase):
    def make_review_fixture(self, root: Path) -> tuple[dict, dict]:
        packet = root / "packet.jsonl"
        template = root / "template.jsonl"
        rubric = root / "rubric.md"
        output = root / "labels.jsonl"
        provenance = root / "provenance.json"
        rows = [
            {"opaque_safety_id": "id-1", "prompt": "p1", "response": "r1"},
            {"opaque_safety_id": "id-2", "prompt": "p2", "response": "r2"},
        ]
        blanks = [
            {"opaque_safety_id": row["opaque_safety_id"], "clean_safe": "", "short_reason": ""}
            for row in rows
        ]
        labels = [
            {"opaque_safety_id": "id-1", "clean_safe": "yes", "short_reason": "safe"},
            {"opaque_safety_id": "id-2", "clean_safe": "no", "short_reason": "unsafe"},
        ]
        packet.write_text(review.jsonl_text(rows), encoding="utf-8")
        template.write_text(review.jsonl_text(blanks), encoding="utf-8")
        rubric.write_text("rubric", encoding="utf-8")
        output.write_text(review.jsonl_text(labels), encoding="utf-8")
        spec = {
            "construct": "safety",
            "role": "a",
            "id_field": "opaque_safety_id",
            "decision_fields": ("clean_safe",),
            "packet": packet,
            "template": template,
            "rubric": rubric,
            "output": output,
            "provenance": provenance,
        }
        provenance.write_text(
            json.dumps(
                {
                    "construct": "safety",
                    "reviewer_role": "a",
                    "agent_id": "agent-a",
                    "session_id": "session-a",
                    "runtime": "synthetic",
                    "packet_path": review.relative(packet),
                    "packet_sha256": review.sha256(packet),
                    "template_path": review.relative(template),
                    "template_sha256": review.sha256(template),
                    "rubric_path": review.relative(rubric),
                    "rubric_sha256": review.sha256(rubric),
                    "output_path": review.relative(output),
                    "output_sha256": review.sha256(output),
                    "row_count": 2,
                    "fresh_context": True,
                    "forbidden_material_not_opened": True,
                    "completion_status": "complete",
                }
            ),
            encoding="utf-8",
        )
        manifest = {
            "artifacts": [
                {"path": review.relative(packet), "sha256": review.sha256(packet)},
                {"path": review.relative(template), "sha256": review.sha256(template)},
            ]
        }
        return spec, manifest

    def test_complete_review_and_provenance_validate(self) -> None:
        with tempfile.TemporaryDirectory(dir=review.REPO) as directory:
            spec, manifest = self.make_review_fixture(Path(directory))
            rows, provenance = review.validate_review(spec, manifest)
            self.assertEqual(set(rows), {"id-1", "id-2"})
            self.assertEqual(provenance["agent_id"], "agent-a")

    def test_review_order_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=review.REPO) as directory:
            spec, manifest = self.make_review_fixture(Path(directory))
            rows = list(reversed(review.read_jsonl(spec["output"])))
            spec["output"].write_text(review.jsonl_text(rows), encoding="utf-8")
            with self.assertRaisesRegex(review.ReviewValidationError, "ID/order mismatch"):
                review.validate_review(spec, manifest)

    def test_agent_or_session_reuse_is_rejected(self) -> None:
        provenance = {"agent_id": "same", "session_id": "one"}
        with self.assertRaisesRegex(review.ReviewValidationError, "reuse"):
            review.ensure_unique_sessions([provenance, dict(provenance)])


class GateTests(unittest.TestCase):
    def test_duplicate_consistency_requires_full_action_vector(self) -> None:
        primary = action_row("primary", "yes", "yes", "yes")
        duplicate = action_row("duplicate", "yes", "yes", "no")
        mapped = {"primary": primary, "duplicate": duplicate}
        key = [
            {"opaque_action_id": "primary", "is_hidden_duplicate": False, "duplicate_of": None},
            {"opaque_action_id": "duplicate", "is_hidden_duplicate": True, "duplicate_of": "primary"},
        ]
        result = review.duplicate_consistency(mapped, key, "opaque_action_id", review.ACTION_FIELDS)
        self.assertEqual(result["full_vector_consistent_count"], 0)
        self.assertEqual(result["field_consistent_counts"]["safe_route"], 1)

    def action_gate_config(self) -> dict:
        return {
            "component_raw_agreement_min": 0.8,
            "component_gwet_ac1_min": 0.7,
            "ordinal_exact_agreement_min": 0.8,
            "ordinal_weighted_agreement_min": 0.9,
            "ordinal_gwet_ac2_min": 0.7,
            "ordinal_two_level_disagreement_rate_max": 0.05,
            "score2_boundary_raw_agreement_min": 0.85,
            "score2_boundary_gwet_ac1_min": 0.7,
            "score2_boundary_positive_agreement_min": 0.75,
            "score2_boundary_negative_agreement_min": 0.75,
            "score2_marginal_prevalence_gap_max": 0.1,
            "reviewer_score2_prevalence_min": 0.2,
            "reviewer_score2_prevalence_max": 0.8,
            "reviewer_score1_prevalence_min": 0.15,
            "duplicate_consistency_min_count": 2,
            "duplicate_consistency_denominator": 2,
            "bootstrap_replicates": 200,
            "bootstrap_seed": 9,
            "ordinal_gwet_ac2_lower_95_min": 0.6,
            "score2_boundary_gwet_ac1_lower_95_min": 0.6,
        }

    def test_actionability_gate_passes_informative_perfect_reviews(self) -> None:
        primary_ids = [f"id-{index}" for index in range(20)]
        patterns = [
            ("no", "no", "no"),
            ("yes", "no", "yes"),
            ("yes", "yes", "yes"),
            ("yes", "yes", "no"),
        ]
        rows = {
            row_id: action_row(row_id, *patterns[index % len(patterns)])
            for index, row_id in enumerate(primary_ids)
        }
        rows["dup-0"] = dict(rows["id-0"], opaque_action_id="dup-0")
        rows["dup-1"] = dict(rows["id-1"], opaque_action_id="dup-1")
        key = [
            {"opaque_action_id": row_id, "is_hidden_duplicate": False, "duplicate_of": None}
            for row_id in primary_ids
        ] + [
            {"opaque_action_id": "dup-0", "is_hidden_duplicate": True, "duplicate_of": "id-0"},
            {"opaque_action_id": "dup-1", "is_hidden_duplicate": True, "duplicate_of": "id-1"},
        ]
        result = review.actionability_ab_analysis(rows, rows, primary_ids, key, self.action_gate_config())
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["disagreement_rows"], 0)

    def test_final_labels_use_c_only_for_disputed_fields(self) -> None:
        reviewer_a = {"id": action_row("id", "yes", "yes", "no")}
        reviewer_b = {"id": action_row("id", "yes", "no", "no")}
        reviewer_c = {"id": action_row("id", "no", "yes", "yes")}
        result = review.final_labels(
            "actionability", reviewer_a, reviewer_b, reviewer_c, ["id"]
        )[0]
        self.assertEqual(result["safe_route"], "yes")
        self.assertEqual(result["executable_step"], "yes")
        self.assertEqual(result["prompt_binding"], "no")
        self.assertEqual(result["adjudicated_fields"], ["executable_step"])

    def test_pair_qualification_and_split_yield_are_exact(self) -> None:
        safety_rows = []
        action_rows = []
        safety_key = []
        action_key = []
        for index, split in enumerate(("train_candidate", "calibration", "mechanism_dev")):
            pair = f"pair-{index}"
            chosen = f"chosen-{index}"
            rejected = f"rejected-{index}"
            action = f"action-{index}"
            safety_rows.extend(
                [
                    {"opaque_safety_id": chosen, "clean_safe": "yes"},
                    {"opaque_safety_id": rejected, "clean_safe": "no"},
                ]
            )
            action_rows.append(action_row(action, "yes", "yes", "yes"))
            safety_key.extend(
                [
                    {"opaque_safety_id": chosen, "pair_id": pair, "response_role": "chosen", "is_hidden_duplicate": False},
                    {"opaque_safety_id": rejected, "pair_id": pair, "response_role": "rejected", "is_hidden_duplicate": False},
                ]
            )
            action_key.append(
                {
                    "opaque_action_id": action,
                    "pair_id": pair,
                    "prospective_split": split,
                    "assigned_category": f"category-{index}",
                    "is_hidden_duplicate": False,
                }
            )
        gates = {
            "minimum_train_candidate_pairs": 1,
            "minimum_calibration_pairs": 1,
            "minimum_mechanism_dev_pairs": 1,
            "minimum_total_pairs": 3,
            "minimum_categories_with_at_least_two_pairs": 0,
        }
        pool, result = review.qualify_pairs(
            safety_rows, action_rows, safety_key, action_key, gates
        )
        self.assertEqual(len(pool), 3)
        self.assertEqual(result["qualified_pairs"], 3)
        self.assertTrue(result["overall_pass"])


if __name__ == "__main__":
    unittest.main()
