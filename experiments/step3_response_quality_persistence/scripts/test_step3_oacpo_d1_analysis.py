"""Tests for OACPO D1 review validation and route selection."""

from __future__ import annotations

import unittest

import step3_oacpo_d1_analysis as analysis


class OacpoD1AnalysisTests(unittest.TestCase):
    def test_route_stops_when_any_construct_fails(self) -> None:
        constructs = {"safety": {"pass": True}, "quality": {"pass": False}, "structure": {"pass": True}}
        self.assertEqual(analysis.route_for(constructs, 3), "STOP_OACPO_D1_REVIEW_UNRELIABLE")

    def test_route_authorizes_c_only_when_reliable_with_disagreements(self) -> None:
        constructs = {name: {"pass": True} for name in ("safety", "quality", "structure")}
        self.assertEqual(
            analysis.route_for(constructs, 1), "AUTHORIZE_OACPO_D1_REVIEWER_C_PROTOCOL_ONLY"
        )

    def test_route_skips_c_when_reliable_without_disagreements(self) -> None:
        constructs = {name: {"pass": True} for name in ("safety", "quality", "structure")}
        self.assertEqual(
            analysis.route_for(constructs, 0),
            "AUTHORIZE_OACPO_D2_TRIPLET_QUALIFICATION_PROTOCOL_ONLY",
        )

    def test_duplicate_consistency_counts_full_vectors(self) -> None:
        review = {
            "p": {"x": 1, "y": "yes"},
            "d": {"x": 1, "y": "no"},
        }
        key = [
            {"opaque_id": "p", "is_hidden_duplicate": False, "duplicate_of": None},
            {"opaque_id": "d", "is_hidden_duplicate": True, "duplicate_of": "p"},
        ]
        result = analysis.duplicate_consistency(review, key, "opaque_id", ("x", "y"))
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["full_vector_consistent_count"], 0)

    def test_at_least_rejects_undefined_metric(self) -> None:
        self.assertFalse(analysis.at_least(None, 0.8))
        self.assertTrue(analysis.at_least(0.8, 0.8))


if __name__ == "__main__":
    unittest.main()
