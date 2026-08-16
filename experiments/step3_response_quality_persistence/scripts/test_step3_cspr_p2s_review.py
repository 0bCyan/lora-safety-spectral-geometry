"""Synthetic tests for calibration-separated CSPR P2S review analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import step3_cspr_p2s_packets as packets
import step3_cspr_p2s_review as review


def safety_row(row_id: str, label: str) -> dict:
    return {
        "opaque_safety_id": row_id,
        "clean_safe": label,
        "short_reason": "synthetic reason",
    }


def action_row(row_id: str, values: tuple[str, str, str]) -> dict:
    return {
        "opaque_action_id": row_id,
        "safe_route": values[0],
        "executable_step": values[1],
        "prompt_binding": values[2],
        "short_reason": "synthetic reason",
    }


def safety_protocol() -> dict:
    return {
        "safety_calibration_gates": {
            "ab_raw_agreement_min": 0.9,
            "ab_gwet_ac1_min": 0.8,
            "ab_positive_agreement_min": 0.75,
            "ab_negative_agreement_min": 0.75,
            "per_reviewer_reference_raw_min": 0.9,
            "per_reviewer_reference_gwet_ac1_min": 0.8,
            "per_reviewer_reference_positive_min": 0.75,
            "per_reviewer_reference_negative_min": 0.75,
        },
        "target_reliability_gates": {
            "safety_raw_agreement_min": 0.9,
            "safety_gwet_ac1_min": 0.8,
            "safety_positive_agreement_min": 0.75,
            "safety_negative_agreement_min": 0.75,
        },
        "duplicate_gates": {
            "safety_denominator": 0,
            "safety_full_vector_consistency_min_count": 0,
        },
    }


def action_protocol() -> dict:
    return {
        "action_calibration_gates": {
            "component_ab_raw_min": 0.9,
            "component_ab_gwet_ac1_min": 0.8,
            "ordinal_ab_exact_min": 0.9,
            "ordinal_ab_weighted_min": 0.9,
            "ordinal_ab_gwet_ac2_min": 0.8,
            "ordinal_two_level_disagreement_rate_max": 0.0,
            "score2_boundary_ab_raw_min": 0.9,
            "score2_boundary_ab_gwet_ac1_min": 0.8,
            "score2_boundary_ab_positive_min": 0.75,
            "score2_boundary_ab_negative_min": 0.75,
            "score2_marginal_prevalence_gap_max": 0.0,
            "reviewer_score2_prevalence_min": 0.2,
            "reviewer_score2_prevalence_max": 0.8,
            "reviewer_score1_prevalence_min": 0.15,
            "per_reviewer_reference_component_raw_min": 0.9,
            "per_reviewer_reference_component_gwet_ac1_min": 0.8,
            "per_reviewer_reference_ordinal_exact_min": 0.9,
            "per_reviewer_reference_ordinal_weighted_min": 0.9,
            "per_reviewer_reference_score2_boundary_raw_min": 0.9,
            "per_reviewer_reference_score2_boundary_gwet_ac1_min": 0.8,
            "per_reviewer_reference_score2_boundary_positive_min": 0.75,
            "per_reviewer_reference_score2_boundary_negative_min": 0.75,
            "bootstrap_replicates": 100,
            "bootstrap_seed": 31,
            "ordinal_gwet_ac2_lower_95_min": 0.8,
            "score2_boundary_gwet_ac1_lower_95_min": 0.8,
        },
        "target_reliability_gates": {
            "action_component_raw_min": 0.9,
            "action_component_gwet_ac1_min": 0.8,
            "action_ordinal_exact_min": 0.9,
            "action_ordinal_weighted_min": 0.9,
            "action_ordinal_gwet_ac2_min": 0.8,
            "action_two_level_disagreement_rate_max": 0.0,
            "action_score2_boundary_raw_min": 0.9,
            "action_score2_boundary_gwet_ac1_min": 0.8,
            "action_score2_boundary_positive_min": 0.75,
            "action_score2_boundary_negative_min": 0.75,
        },
        "duplicate_gates": {
            "action_denominator": 0,
            "action_full_vector_consistency_min_count": 0,
        },
    }


def action_fixture() -> tuple[dict[str, dict], list[dict]]:
    patterns = [
        ("no", "no", "no"),
        ("no", "no", "no"),
        ("yes", "no", "yes"),
        ("yes", "no", "yes"),
        ("yes", "yes", "yes"),
        ("yes", "yes", "yes"),
    ]
    rows = {}
    key = []
    for index, values in enumerate(patterns):
        row_id = f"cal-{index}"
        rows[row_id] = action_row(row_id, values)
        key.append(
            {
                "opaque_action_id": row_id,
                "item_type": "calibration_anchor",
                "is_hidden_duplicate": False,
                "duplicate_of": None,
                "reference_safe_route": values[0],
                "reference_executable_step": values[1],
                "reference_prompt_binding": values[2],
            }
        )
    target_patterns = [("no", "no", "no")] + [("yes", "yes", "yes")] * 9
    for index, values in enumerate(target_patterns):
        row_id = f"target-{index}"
        rows[row_id] = action_row(row_id, values)
        key.append(
            {
                "opaque_action_id": row_id,
                "item_type": "target_candidate",
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )
    return rows, key


class SeparationAndGateTests(unittest.TestCase):
    def safety_fixture(self) -> tuple[dict[str, dict], list[dict]]:
        rows = {}
        key = []
        for index, label in enumerate(("yes", "yes", "no", "no")):
            row_id = f"cal-{index}"
            rows[row_id] = safety_row(row_id, label)
            key.append(
                {
                    "opaque_safety_id": row_id,
                    "item_type": "calibration_anchor",
                    "is_hidden_duplicate": False,
                    "duplicate_of": None,
                    "reference_clean_safe": label,
                }
            )
        for index, label in enumerate(("yes", "yes", "no", "no")):
            row_id = f"target-{index}"
            rows[row_id] = safety_row(row_id, label)
            key.append(
                {
                    "opaque_safety_id": row_id,
                    "item_type": "target_candidate",
                    "is_hidden_duplicate": False,
                    "duplicate_of": None,
                    "reference_clean_safe": None,
                }
            )
        return rows, key

    def test_calibration_and_target_metrics_are_separate(self) -> None:
        rows, key = self.safety_fixture()
        result = review.safety_analysis(rows, rows, key, safety_protocol())
        self.assertEqual(result["calibration_rows"], 4)
        self.assertEqual(result["target_rows"], 4)
        self.assertTrue(result["calibration_pass"])
        self.assertTrue(result["target_pass"])

    def test_reference_concordance_is_a_hard_gate(self) -> None:
        rows, key = self.safety_fixture()
        wrong = {
            row_id: safety_row(row_id, "no" if row["clean_safe"] == "yes" else "yes")
            if row_id.startswith("cal-")
            else row
            for row_id, row in rows.items()
        }
        result = review.safety_analysis(wrong, wrong, key, safety_protocol())
        self.assertEqual(result["calibration_ab"]["raw_agreement"], 1.0)
        self.assertFalse(result["calibration_pass"])
        self.assertFalse(result["calibration_gates"]["a_reference_raw"])

    def test_target_score_prevalence_is_reported_but_not_a_gate(self) -> None:
        rows, key = action_fixture()
        result = review.action_analysis(rows, rows, key, action_protocol())
        self.assertTrue(result["calibration_pass"])
        self.assertTrue(result["target_pass"])
        self.assertEqual(result["target_ab"]["score_prevalence"]["left_score2"], 0.9)
        self.assertEqual(result["target_ab"]["score_prevalence"]["left_score1"], 0.0)
        self.assertFalse(result["target_score_prevalence_used_as_gate"])

    def test_action_reference_concordance_is_a_hard_gate(self) -> None:
        rows, key = action_fixture()
        for key_row in key:
            if key_row["item_type"] == "calibration_anchor":
                key_row["reference_safe_route"] = (
                    "no" if key_row["reference_safe_route"] == "yes" else "yes"
                )
        result = review.action_analysis(rows, rows, key, action_protocol())
        self.assertFalse(result["calibration_pass"])
        self.assertFalse(result["calibration_gates"]["a_reference_safe_route_raw"])

    def test_c_packet_contains_only_target_disagreements(self) -> None:
        rows, key = self.safety_fixture()
        reviewer_b = {row_id: dict(row) for row_id, row in rows.items()}
        reviewer_b["cal-0"] = safety_row("cal-0", "no")
        reviewer_b["target-0"] = safety_row("target-0", "no")
        result = review.safety_analysis(rows, reviewer_b, key, safety_protocol())
        self.assertEqual(result["target_disagreement_ids"], ["target-0"])

        with tempfile.TemporaryDirectory(dir=review.REPO) as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            template_path = root / "template.jsonl"
            packet_path.write_text(
                review.jsonl_text(
                    [
                        {"opaque_safety_id": row_id, "prompt": "p", "response": "r"}
                        for row_id in rows
                    ]
                ),
                encoding="utf-8",
            )
            template_path.write_text("", encoding="utf-8")
            execution = {
                "review_files": {
                    "safety": {
                        "reviewer_a_labels": review.relative(root / "labels.jsonl"),
                        "reviewer_a_provenance": review.relative(root / "provenance.json"),
                    }
                }
            }
            protocol = {
                "packet_outputs": {
                    "safety_a_packet": review.relative(packet_path),
                    "safety_a_template": review.relative(template_path),
                },
                "frozen_inputs": {"safety_rubric": {"path": review.relative(root / "rubric.md")}},
            }
            packet, template = review.build_c_packet(
                "safety", result["target_disagreement_ids"], execution, protocol
            )
            self.assertEqual([row["opaque_safety_id"] for row in packet], ["target-0"])
            self.assertEqual([row["opaque_safety_id"] for row in template], ["target-0"])


class AdjudicationAndPoolTests(unittest.TestCase):
    def test_duplicate_consistency_requires_full_action_vector(self) -> None:
        mapped = {
            "primary": action_row("primary", ("yes", "yes", "yes")),
            "duplicate": action_row("duplicate", ("yes", "yes", "no")),
        }
        key = [
            {
                "opaque_action_id": "duplicate",
                "is_hidden_duplicate": True,
                "duplicate_of": "primary",
            }
        ]
        result = review.duplicate_consistency(
            mapped, key, "opaque_action_id", review.ACTION_FIELDS
        )
        self.assertEqual(result["full_vector_consistent_count"], 0)
        self.assertEqual(result["field_consistent_counts"]["safe_route"], 1)

    def test_c_cannot_rewrite_agreed_fields(self) -> None:
        reviewer_a = {"id": action_row("id", ("yes", "yes", "no"))}
        reviewer_b = {"id": action_row("id", ("yes", "no", "no"))}
        reviewer_c = {"id": action_row("id", ("no", "yes", "yes"))}
        result = review.final_target_labels(
            "actionability", reviewer_a, reviewer_b, reviewer_c, ["id"]
        )[0]
        self.assertEqual(result["safe_route"], "yes")
        self.assertEqual(result["executable_step"], "yes")
        self.assertEqual(result["prompt_binding"], "no")
        self.assertEqual(result["adjudicated_fields"], ["executable_step"])

    def test_stable_hash_selection_and_fixed_pool_yield(self) -> None:
        safety_final = []
        action_final = []
        safety_key = []
        action_key = []
        splits = ("train_candidate", "calibration", "mechanism_dev")
        for index, split in enumerate(splits):
            pair_id = f"pair-{index}"
            rejected_id = f"safe-rejected-{index}"
            safety_final.append(safety_row(rejected_id, "no"))
            safety_key.append(
                {
                    "opaque_safety_id": rejected_id,
                    "source_authoring_id": None,
                    "pair_id": pair_id,
                    "item_type": "target_rejected",
                    "is_hidden_duplicate": False,
                }
            )
            for role in ("a", "b"):
                authoring_id = f"author-{index}-{role}"
                safety_id = f"safe-{index}-{role}"
                action_id = f"action-{index}-{role}"
                safety_final.append(safety_row(safety_id, "yes"))
                action_final.append(action_row(action_id, ("yes", "yes", "yes")))
                safety_key.append(
                    {
                        "opaque_safety_id": safety_id,
                        "source_authoring_id": authoring_id,
                        "pair_id": pair_id,
                        "item_type": "target_candidate",
                        "is_hidden_duplicate": False,
                    }
                )
                action_key.append(
                    {
                        "opaque_action_id": action_id,
                        "source_authoring_id": authoring_id,
                        "pair_id": pair_id,
                        "item_type": "target_candidate",
                        "is_hidden_duplicate": False,
                        "author_role": role,
                        "prospective_split": split,
                        "assigned_category": f"category-{index}",
                    }
                )
        protocol = {
            "packet_design": {"seed": 53},
            "fixed_pool_yield_gates": {
                "minimum_train_candidate_pairs": 1,
                "minimum_calibration_pairs": 1,
                "minimum_mechanism_dev_pairs": 1,
                "minimum_total_pairs": 3,
                "minimum_categories_with_at_least_two_pairs": 0,
            },
        }
        pool, result = review.qualify_pairs(
            safety_final, action_final, safety_key, action_key, protocol
        )
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["qualified_pairs"], 3)
        first_authors = ["author-0-a", "author-0-b"]
        expected = min(
            first_authors,
            key=lambda author_id: packets.stable_rank(
                53, "qualified-candidate-selection", author_id
            ),
        )
        self.assertEqual(pool[0]["selected_authoring_id"], expected)


class IsolationTests(unittest.TestCase):
    def test_agent_session_and_prior_reuse_are_rejected(self) -> None:
        valid = [
            {"agent_id": "agent-a", "session_id": "session-a"},
            {"agent_id": "agent-b", "session_id": "session-b"},
        ]
        review.ensure_reviewer_isolation(valid, [])
        cases = [
            ([valid[0], {"agent_id": "agent-a", "session_id": "session-c"}], [], "agent"),
            ([valid[0], {"agent_id": "agent-c", "session_id": "session-a"}], [], "session"),
            (valid, ["agent-b"], "old P2"),
        ]
        for provenances, forbidden, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(review.ReviewValidationError, message):
                    review.ensure_reviewer_isolation(provenances, forbidden)


if __name__ == "__main__":
    unittest.main()
