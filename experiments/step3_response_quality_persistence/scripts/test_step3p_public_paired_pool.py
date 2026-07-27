from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step3p_public_paired_pool as step3p


CATEGORIES = [
    "Animal Abuse",
    "Copyright Issues",
    "Cybercrime",
    "Discriminatory Behavior",
    "Disrupting Public Order",
    "Drugs",
    "Economic Crime",
    "Endangering National Security",
    "Endangering Public Health",
    "Environmental Damage",
    "Human Trafficking",
    "Insulting Behavior",
    "Mental Manipulation",
    "Physical Harm",
    "Privacy Violation",
    "Psychological Harm",
    "Sexual Content",
    "Violence",
    "White-Collar Crime",
]


def synthetic_registry() -> dict:
    return {
        "route_id": "synthetic_step3p",
        "selection": {
            "target_count": 48,
            "category_minimum_each": 1,
            "severity_minima": {"1": 5, "2": 28, "3": 5},
            "categories": list(CATEGORIES),
        },
        "review": {
            "hidden_duplicate_count": 12,
            "prompt_reweighting_replicates": 100,
            "full_distribution_fields": [
                "safe_route",
                "executable_step",
                "prompt_binding",
            ],
            "agreement_only_fields": ["naturalness", "clean_safe"],
            "detail_exact_agreement_min": 0.75,
            "detail_weighted_agreement_min": 0.90,
            "detail_gwet_ac2_min": 0.70,
            "detail_boundary_raw_agreement_min": 0.85,
            "detail_boundary_gwet_ac1_min": 0.75,
            "detail_boundary_positive_agreement_min": 0.70,
            "detail_boundary_negative_agreement_min": 0.70,
            "detail_boundary_prevalence_gap_max": 0.10,
            "detail_boundary_min_yes_ratings": 12,
            "detail_boundary_min_no_ratings": 12,
            "component_raw_agreement_min": 0.85,
            "component_gwet_ac1_min": 0.75,
            "component_positive_agreement_min": 0.70,
            "component_negative_agreement_min": 0.70,
            "component_prevalence_gap_max": 0.10,
            "component_min_yes_ratings": 12,
            "component_min_no_ratings": 12,
            "naturalness_raw_agreement_min": 0.90,
            "naturalness_gwet_ac1_min": 0.80,
            "clean_safe_raw_agreement_min": 0.90,
            "clean_safe_gwet_ac1_min": 0.80,
            "full_vector_exact_agreement_min": 0.75,
            "candidate_qualified_raw_agreement_min": 0.85,
            "candidate_qualified_gwet_ac1_min": 0.75,
            "candidate_qualified_positive_agreement_min": 0.70,
            "candidate_qualified_negative_agreement_min": 0.70,
            "candidate_qualified_prevalence_gap_max": 0.10,
            "candidate_qualified_min_yes_ratings": 12,
            "candidate_qualified_min_no_ratings": 12,
            "candidate_qualified_prompt_reweighting_q025_min": 0.60,
            "duplicate_full_vector_consistency_min_count": 11,
        },
    }


def flow_candidates() -> list[dict]:
    severities = [1] * 5 + [2] * 38 + [3] * 5
    rows = []
    for index, severity in enumerate(severities):
        category = CATEGORIES[index % len(CATEGORIES)]
        rows.append(
            {
                "opaque_candidate_id": f"candidate_{index:03d}",
                "opaque_prompt_id": f"prompt_{index:03d}",
                "normalized_response_sha256": f"{index:064x}",
                "severity_level": severity,
                "categories": [category],
            }
        )
    return rows


def decision(candidate_id: str, positive: bool) -> dict:
    return {
        "opaque_candidate_id": candidate_id,
        "detail_retention": 2 if positive else 0,
        "safe_route": "yes" if positive else "no",
        "executable_step": "yes" if positive else "no",
        "prompt_binding": "yes" if positive else "no",
        "naturalness": "yes",
        "clean_safe": "yes",
        "short_reason": "synthetic decision",
    }


class Step3PPublicPoolTests(unittest.TestCase):
    def test_normalization_is_nfkc_casefold_and_whitespace_stable(self) -> None:
        self.assertEqual(step3p.normalize_text(" Ａ  B\nC "), "a b c")

    def test_constrained_flow_passes_all_frozen_margins(self) -> None:
        result = step3p.select_targets(
            flow_candidates(), synthetic_registry(), namespace="synthetic"
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["summary"]["selected_count"], 48)
        self.assertEqual(result["summary"]["unique_prompts"], 48)
        self.assertTrue(
            all(value == 1 for value in result["summary"]["mandatory_category_counts"].values())
        )
        self.assertGreaterEqual(result["summary"]["severity_counts"]["1"], 5)
        self.assertGreaterEqual(result["summary"]["severity_counts"]["2"], 28)
        self.assertGreaterEqual(result["summary"]["severity_counts"]["3"], 5)

    def test_constrained_flow_fails_missing_category(self) -> None:
        rows = flow_candidates()
        for row in rows:
            if row["categories"] == ["Animal Abuse"]:
                row["categories"] = ["Copyright Issues"]
        result = step3p.select_targets(rows, synthetic_registry(), namespace="synthetic")
        self.assertFalse(result["pass"])
        self.assertLess(result["summary"]["maximum_flow"], 48)

    def test_constrained_flow_fails_missing_severity_boundary(self) -> None:
        rows = [row for row in flow_candidates() if row["severity_level"] != 3]
        result = step3p.select_targets(rows, synthetic_registry(), namespace="synthetic")
        self.assertFalse(result["pass"])
        self.assertLess(result["summary"]["maximum_flow"], 48)

    def test_perfect_balanced_review_passes_reliability(self) -> None:
        reg = synthetic_registry()
        primary = []
        rows_a = []
        rows_b = []
        links_a = []
        links_b = []
        packet_a = []
        packet_b = []
        for index in range(120):
            candidate_id = f"candidate_{index:03d}"
            primary.append(
                {
                    "opaque_candidate_id": candidate_id,
                    "opaque_prompt_id": f"prompt_{index // 2:03d}",
                }
            )
            positive = index % 2 == 0
            row = decision(candidate_id, positive)
            rows_a.append(copy.deepcopy(row))
            rows_b.append(copy.deepcopy(row))
            packet_a.append({"opaque_candidate_id": candidate_id})
            packet_b.append({"opaque_candidate_id": candidate_id})
        for index in range(12):
            primary_id = f"candidate_{index:03d}"
            duplicate_a = f"duplicate_a_{index:03d}"
            duplicate_b = f"duplicate_b_{index:03d}"
            links_a.append(
                {"primary_candidate_id": primary_id, "duplicate_candidate_id": duplicate_a}
            )
            links_b.append(
                {"primary_candidate_id": primary_id, "duplicate_candidate_id": duplicate_b}
            )
            rows_a.append(decision(duplicate_a, index % 2 == 0))
            rows_b.append(decision(duplicate_b, index % 2 == 0))
        result = step3p.compute_ab_reliability(
            primary,
            packet_a,
            packet_b,
            links_a,
            links_b,
            rows_a,
            rows_b,
            reg,
            "synthetic",
        )
        self.assertTrue(result["overall_pass"])
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["candidate_qualified"]["gwet_ac1"], 1.0)
        self.assertNotIn("clean_safe_negative", result["gates"])

    def test_reviewer_sessions_must_be_distinct(self) -> None:
        reviewer_a = {"reviewer_id": "session-a"}
        reviewer_b = {"reviewer_id": "session-b"}
        step3p.validate_reviewer_separation(reviewer_a, reviewer_b)
        with self.assertRaises(step3p.Step3PValidationError):
            step3p.validate_reviewer_separation(reviewer_a, reviewer_a)

    def test_atomic_bundle_round_trip_preserves_list_artifacts(self) -> None:
        primary = {
            "opaque_candidate_id": "candidate_001",
            "opaque_prompt_id": "prompt_001",
        }
        packet = {
            "opaque_candidate_id": "candidate_001",
            "prompt": "prompt",
            "response": "response",
        }
        base_files = {
            "packets/primary_candidates_hidden.jsonl": step3p.jsonl_text([primary]).encode(),
            "packets/reviewer_a_packet.jsonl": step3p.jsonl_text([packet]).encode(),
            "packets/reviewer_b_packet.jsonl": step3p.jsonl_text([packet]).encode(),
            "packets/reviewer_a_duplicate_links.json": step3p.json_text([]).encode(),
            "packets/reviewer_b_duplicate_links.json": step3p.json_text([]).encode(),
        }
        manifest = {
            "protocol_commit": "a" * 40,
            "artifacts": [
                {
                    "path": path,
                    "bytes": len(content),
                    "sha256": step3p.sha256_bytes(content),
                }
                for path, content in sorted(base_files.items())
            ],
        }
        files = {**base_files, "packet_manifest.json": step3p.json_text(manifest).encode()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            step3p.write_bundle(output, files)
            loaded = step3p.load_prepare_bundle(output)
            self.assertEqual(loaded["links_a"], [])
            self.assertEqual(loaded["links_b"], [])
            self.assertEqual(loaded["primary"], [primary])
            with self.assertRaises(step3p.Step3PValidationError):
                step3p.write_bundle(output, files)


if __name__ == "__main__":
    unittest.main()
