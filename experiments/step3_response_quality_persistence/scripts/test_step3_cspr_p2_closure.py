"""Synthetic tests for the deterministic CSPR P2 resource closure bound."""

from __future__ import annotations

import unittest

import step3_cspr_p2_closure as closure


def review_row(row_id: str, safe: str, step: str, binding: str) -> dict:
    return {
        "opaque_action_id": row_id,
        "safe_route": safe,
        "executable_step": step,
        "prompt_binding": binding,
    }


def key_row(row_id: str, split: str = "train_candidate", category: str = "cat") -> dict:
    return {
        "opaque_action_id": row_id,
        "is_hidden_duplicate": False,
        "prospective_split": split,
        "assigned_category": category,
    }


class ClosureBoundTests(unittest.TestCase):
    def gates(self) -> dict:
        return {
            "minimum_train_candidate_pairs": 2,
            "train_candidate_denominator": 3,
            "minimum_calibration_pairs": 1,
            "calibration_denominator": 1,
            "minimum_mechanism_dev_pairs": 1,
            "mechanism_dev_denominator": 1,
            "minimum_total_pairs": 4,
            "total_denominator": 5,
            "minimum_categories_with_at_least_two_pairs": 1,
            "category_denominator": 2,
        }

    def fixture(self) -> tuple[dict, dict, list[dict]]:
        left = {
            "consensus": review_row("consensus", "yes", "yes", "yes"),
            "disputed": review_row("disputed", "yes", "yes", "yes"),
            "locked": review_row("locked", "yes", "no", "yes"),
            "cal": review_row("cal", "yes", "yes", "yes"),
            "dev": review_row("dev", "yes", "yes", "yes"),
        }
        right = {
            "consensus": review_row("consensus", "yes", "yes", "yes"),
            "disputed": review_row("disputed", "yes", "no", "yes"),
            "locked": review_row("locked", "no", "no", "yes"),
            "cal": review_row("cal", "yes", "yes", "yes"),
            "dev": review_row("dev", "yes", "yes", "yes"),
        }
        key = [
            key_row("consensus", "train_candidate", "cat-a"),
            key_row("disputed", "train_candidate", "cat-a"),
            key_row("locked", "train_candidate", "cat-b"),
            key_row("cal", "calibration", "cat-b"),
            key_row("dev", "mechanism_dev", "cat-b"),
        ]
        return left, right, key

    def test_agreed_no_is_immutable_under_disagreement_only_c(self) -> None:
        left, right, key = self.fixture()
        result = closure.compute_bounds(left, right, key, self.gates())
        self.assertEqual(result["consensus_score2_lower_bound"], 3)
        self.assertEqual(result["disputed_rows_that_could_reach_score2"], 1)
        self.assertEqual(result["adjudication_best_case_score2_upper_bound"], 4)
        self.assertEqual(result["rows_locked_below_score2_by_at_least_one_ab_agreed_no"], 1)

    def test_qualification_upper_bound_equals_potential_score2_count(self) -> None:
        left, right, key = self.fixture()
        result = closure.compute_bounds(left, right, key, self.gates())
        self.assertEqual(result["total_bound"]["qualification_upper_bound"], 4)
        self.assertTrue(result["total_bound"]["total_yield_mathematically_reachable"])

    def test_strict_shortfall_closes_the_pool(self) -> None:
        left, right, key = self.fixture()
        gates = self.gates()
        gates["minimum_total_pairs"] = 5
        result = closure.compute_bounds(left, right, key, gates)
        self.assertTrue(result["fixed_pool_mathematically_unreachable"])
        self.assertEqual(result["total_bound"]["shortfall_even_under_best_case_adjudication"], 1)

    def test_split_bounds_do_not_borrow_across_splits(self) -> None:
        left, right, key = self.fixture()
        result = closure.compute_bounds(left, right, key, self.gates())
        self.assertEqual(
            result["split_bounds"]["train_candidate"]["adjudication_best_case_score2_upper_bound"],
            2,
        )
        self.assertEqual(
            result["split_bounds"]["calibration"]["adjudication_best_case_score2_upper_bound"],
            1,
        )

    def test_hidden_duplicates_are_excluded_before_bounds(self) -> None:
        left, right, key = self.fixture()
        left["duplicate"] = review_row("duplicate", "yes", "yes", "yes")
        right["duplicate"] = review_row("duplicate", "yes", "yes", "yes")
        key.append(
            {
                "opaque_action_id": "duplicate",
                "is_hidden_duplicate": True,
                "duplicate_of": "consensus",
                "prospective_split": "train_candidate",
                "assigned_category": "cat-a",
            }
        )
        mapped_a, mapped_b, primary = closure.validate_action_reviews(
            list(left.values()), list(right.values()), key
        )
        result = closure.compute_bounds(mapped_a, mapped_b, primary, self.gates())
        self.assertEqual(result["primary_candidate_rows"], 5)
        self.assertEqual(result["adjudication_best_case_score2_upper_bound"], 4)

    def test_invalid_label_is_rejected(self) -> None:
        left, right, key = self.fixture()
        left["consensus"]["safe_route"] = "maybe"
        with self.assertRaisesRegex(closure.ClosureValidationError, "invalid safe_route"):
            closure.validate_action_reviews(list(left.values()), list(right.values()), key)


if __name__ == "__main__":
    unittest.main()
