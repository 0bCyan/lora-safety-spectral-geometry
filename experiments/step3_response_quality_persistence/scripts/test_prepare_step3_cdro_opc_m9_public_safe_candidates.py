"""Tests for M9 public-safe single-target candidate selection."""

from __future__ import annotations

import copy
import unittest

import prepare_step3_cdro_opc_m9_public_safe_candidates as subject


class M9CandidateTests(unittest.TestCase):
    def protocol(self) -> dict[str, object]:
        return {
            "schema": subject.SCHEMA,
            "status": subject.STATUS,
            "entry_commit": "a" * 40,
            "candidate_registry_sha256": "0" * 64,
            "selection": {
                "candidate_ordering": "SHA256(candidate_ordering_salt|source_candidate_id)",
                "candidate_ordering_salt": "cdro-opc-m9-public-candidate-v1",
                "exclude_by": [
                    "sample_id",
                    "NFKC_casefold_whitespace_normalized_prompt_sha256",
                ],
                "forbidden_selection_inputs": [
                    "prior_review_labels",
                    "M6_M7_row_metrics",
                    "category",
                    "severity",
                    "response_content_score",
                ],
                "fresh_public_safe_prompts_expected": 77,
                "prompt_ordering": "SHA256(prompt_ordering_salt|sample_id)",
                "prompt_ordering_salt": "cdro-opc-m9-public-prompt-v1",
                "prompts": 60,
                "responses_per_prompt": 1,
            },
            "review": {
                "primary_candidates": 60,
                "positive_anchors": 12,
                "generic_anchors": 12,
                "mismatch_anchors": 12,
                "clean_safety_negative_anchors": 12,
                "hidden_primary_duplicates": 12,
                "rows_per_reviewer": 120,
                "reliability_and_qualification_gates": "identical_to_M5",
                "eligible_rule": "single_candidate_qualified",
                "eligible_prompts_required": 40,
                "cohort_ordering": "SHA256(590031|cohort|opaque_prompt_id)",
                "calibration_units": 20,
                "sealed_audit_units": 20,
                "split_rule": "alternating_zero_based_selected_rank_even_calibration_odd_sealed",
            },
            "gpu_validation": {
                "targets_per_unit": 1,
                "fixed_dose": 0.0337860296353059,
                "virtual_scale_eta": 0.0002,
                "protected_certificates": [
                    "risk",
                    "uniform",
                    "tarc",
                    "continuation",
                ],
                "allowed_relative_drop": 0.04,
                "score_scale_floor": 0.02,
                "required_covered_units_per_split": 19,
                "required_total_units_per_split": 20,
                "certificate_mean_deterioration_max": 0.04,
                "minimum_behavior_prediction": 0.0,
                "minimum_behavior_actual": 0.0,
                "behavior_ratio_lower": 0.25,
                "behavior_ratio_upper": 1.75,
                "maximum_map_relative_l2_error": 0.000002,
                "maximum_map_absolute_error": 0.00001,
                "sealed_opening_rule": "all_common_and_calibration_gates_pass",
                "two_route_branching_claim_authorized": False,
                "positive_mirror_authorized": False,
                "radius_grid_authorized": False,
                "dose_tuning_authorized": False,
                "result_dependent_rerun_authorized": False,
            },
            "authorization_boundary": {
                "candidate_generation_authorized": False,
                "core100_opened": False,
                "heldout_opened": False,
                "optimizer_steps": 0,
                "training_authorized": False,
            },
        }

    def test_protocol_is_strict(self) -> None:
        protocol = self.protocol()
        subject.validate_protocol(protocol, allow_pending=False)
        changed = copy.deepcopy(protocol)
        changed["selection"]["prompts"] = 59  # type: ignore[index]
        with self.assertRaises(subject.M9CandidateError):
            subject.validate_protocol(changed, allow_pending=False)

    def test_pending_hash_only_allowed_before_lock(self) -> None:
        protocol = self.protocol()
        protocol["candidate_registry_sha256"] = subject.PENDING_HASH
        subject.validate_protocol(protocol, allow_pending=True)
        with self.assertRaises(subject.M9CandidateError):
            subject.validate_protocol(protocol, allow_pending=False)

    def test_review_and_gpu_contracts_are_frozen(self) -> None:
        changed_review = copy.deepcopy(self.protocol())
        changed_review["review"]["eligible_prompts_required"] = 39  # type: ignore[index]
        with self.assertRaises(subject.M9CandidateError):
            subject.validate_protocol(changed_review, allow_pending=False)
        changed_gpu = copy.deepcopy(self.protocol())
        changed_gpu["gpu_validation"]["fixed_dose"] = 0.03  # type: ignore[index]
        with self.assertRaises(subject.M9CandidateError):
            subject.validate_protocol(changed_gpu, allow_pending=False)

    def test_full_entry_commit_and_lowercase_hash_are_required(self) -> None:
        short_entry = copy.deepcopy(self.protocol())
        short_entry["entry_commit"] = "a5bfdbd"
        with self.assertRaises(subject.M9CandidateError):
            subject.validate_protocol(short_entry, allow_pending=False)
        uppercase_hash = copy.deepcopy(self.protocol())
        uppercase_hash["candidate_registry_sha256"] = "A" * 64
        with self.assertRaises(subject.M9CandidateError):
            subject.validate_protocol(uppercase_hash, allow_pending=False)

    def test_normalization_matches_m8_exclusion_contract(self) -> None:
        self.assertEqual(subject.m8.normalize_prompt("  Ｔest\tPROMPT  "), "test prompt")


if __name__ == "__main__":
    unittest.main()
