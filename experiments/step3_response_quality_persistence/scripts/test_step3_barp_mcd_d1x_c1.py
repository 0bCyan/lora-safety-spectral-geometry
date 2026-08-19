"""Tests for D1X-C1 conditional route qualification."""

from __future__ import annotations

import unittest

import step3_barp_mcd_d1x_c1 as c1


def row(value: float, active: bool = True, retention: float = 0.97) -> dict:
    return {
        "route_exercised": active,
        "minimum_local_protected_retention": retention,
        "optimizer_completed_steps": 0,
        "target_macro_deltas": [value, value, value],
        "support_rows": [
            {
                "exact_delta_scores": {
                    "risk": 0.001,
                    "uniform": 0.001,
                    "tarc": 0.001,
                    "etarc": value,
                    "continuation": 0.001,
                    "branch_entropy": value,
                    "branch_negative_gap": value,
                }
            }
            for _ in range(3)
        ],
    }


class C1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = {
            "route_id": "mask072_lambda0.500",
            "source_mask": 72,
            "source_subset": ["etarc", "branch_negative_gap"],
            "homotopy_fraction": 0.5,
        }
        self.gates = {
            "proposal_activity_min": 3,
            "applied_local_protected_retention_min": 0.95,
            "target_macro_mean_min": 0.002,
            "target_stability_lower_min": 0.0,
            "positive_proposal_clusters_min": 3,
            "negative_proposal_clusters_max": 0,
            "target_component_mean_min": 0.0,
            "protected_mean_disadvantage_max": 0.005,
            "protected_worst_disadvantage_max": 0.02,
        }

    def test_anchor_fallback_has_applied_retention_one(self) -> None:
        rows = [row(0.01) for _ in range(3)] + [row(0.0, False, 0.2)]
        summary = c1.summarize_conditional(self.descriptor, rows)
        self.assertEqual(summary["proposal_activity"], 3)
        self.assertEqual(summary["minimum_candidate_local_protected_retention"], 0.2)
        self.assertEqual(summary["minimum_applied_local_protected_retention"], 0.97)

    def test_conditional_policy_passes_registered_pattern(self) -> None:
        rows = [row(0.01) for _ in range(3)] + [row(0.0, False, 0.2)]
        summary = c1.summarize_conditional(self.descriptor, rows)
        checks = c1.qualification_checks(summary, self.gates)
        self.assertTrue(all(checks.values()))
        self.assertEqual(
            c1.select_route(checks),
            "PASS_BARP_MCD_D1X_C1_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        )

    def test_two_active_proposals_fail_activity(self) -> None:
        rows = [row(0.01) for _ in range(2)] + [row(0.0, False), row(0.0, False)]
        checks = c1.qualification_checks(
            c1.summarize_conditional(self.descriptor, rows), self.gates
        )
        self.assertFalse(checks["route_activity"])
        self.assertEqual(
            c1.select_route(checks),
            "STOP_BARP_MCD_D1X_C1_ROUTE_ACTIVITY_FAILED",
        )

    def test_negative_cluster_fails_sign(self) -> None:
        rows = [row(0.02), row(0.02), row(0.02), row(-0.001)]
        summary = c1.summarize_conditional(self.descriptor, rows)
        checks = c1.qualification_checks(summary, self.gates)
        self.assertFalse(checks["target_cluster_sign"])

    def test_nonpositive_component_fails(self) -> None:
        rows = [row(0.01) for _ in range(4)]
        for item in rows:
            for support in item["support_rows"]:
                support["exact_delta_scores"]["branch_entropy"] = -0.001
        checks = c1.qualification_checks(
            c1.summarize_conditional(self.descriptor, rows), self.gates
        )
        self.assertFalse(checks["target_components"])

    def test_optimizer_state_drift_precedes_effect(self) -> None:
        rows = [row(0.01) for _ in range(4)]
        rows[0]["optimizer_completed_steps"] = 1
        checks = c1.qualification_checks(
            c1.summarize_conditional(self.descriptor, rows), self.gates
        )
        self.assertEqual(
            c1.select_route(checks),
            "STOP_BARP_MCD_D1X_C1_OPTIMIZER_STATE_DRIFT",
        )


if __name__ == "__main__":
    unittest.main()
