"""Tests for BARP-MCD D1X cross-fitted route qualification."""

from __future__ import annotations

import math
import unittest

import step3_barp_mcd_d1x as d1x


def proposal(value: float, retention: float = 0.97, active: bool = True) -> dict:
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
                    "continuation": 0.001,
                }
            }
            for _ in range(3)
        ],
    }


class D1XTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates = {
            "required_proposal_activity": 4,
            "local_protected_retention_min": 0.95,
            "target_cluster_lower_min": 0.0,
            "discovery_positive_clusters_min": 3,
            "qualification_positive_clusters_min": 4,
            "qualification_target_mean_min": 0.002,
            "protected_mean_disadvantage_max": 0.005,
            "protected_worst_disadvantage_max": 0.02,
        }
        self.descriptor = d1x.route_registry()[0]

    def test_route_registry_is_fixed_target_only_family(self) -> None:
        rows = d1x.route_registry()
        self.assertEqual(len(rows), 56)
        self.assertEqual(len({row["route_id"] for row in rows}), 56)
        targets = set(d1x.rp.TARGET_CERTIFICATES)
        self.assertTrue(all(set(row["source_subset"]).issubset(targets) for row in rows))
        self.assertTrue(all(row["source_subset"] for row in rows))

    def test_cluster_lower_bound_uses_four_clusters(self) -> None:
        summary = d1x.cluster_summary([0.01, 0.01, 0.01, 0.01])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 0.01)
        self.assertAlmostEqual(summary["studentized_lower_score"], 0.01)
        self.assertFalse(summary["population_confidence_interval_claimed"])
        self.assertEqual(summary["positive"], 4)

    def test_cluster_summary_rejects_wrong_count(self) -> None:
        with self.assertRaises(d1x.D1XValidationError):
            d1x.cluster_summary([0.01, 0.02, 0.03])

    def test_stable_route_is_eligible(self) -> None:
        summary = d1x.summarize_route(
            self.descriptor, [proposal(0.01) for _ in range(4)], self.gates
        )
        self.assertTrue(summary["eligible"])
        self.assertTrue(all(summary["checks"].values()))

    def test_local_protection_failure_rejects_route(self) -> None:
        rows = [proposal(0.01) for _ in range(4)]
        rows[2]["minimum_local_protected_retention"] = 0.94
        summary = d1x.summarize_route(self.descriptor, rows, self.gates)
        self.assertFalse(summary["eligible"])
        self.assertFalse(summary["checks"]["local_protection"])

    def test_discovery_selection_is_deterministic(self) -> None:
        weak = d1x.summarize_route(
            d1x.route_registry()[0], [proposal(0.01) for _ in range(4)], self.gates
        )
        strong = d1x.summarize_route(
            d1x.route_registry()[1], [proposal(0.02) for _ in range(4)], self.gates
        )
        selected = d1x.select_discovery_route([weak, strong])
        self.assertEqual(selected["route_id"], strong["route_id"])

    def test_qualification_pass_requires_effect_and_sign(self) -> None:
        summary = d1x.summarize_route(
            self.descriptor, [proposal(0.01) for _ in range(4)], self.gates
        )
        checks = d1x.qualification_checks(summary, self.gates)
        self.assertTrue(all(checks.values()))
        self.assertEqual(
            d1x.select_route(summary, summary, checks),
            "PASS_BARP_MCD_D1X_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        )

    def test_weak_qualification_stops(self) -> None:
        summary = d1x.summarize_route(
            self.descriptor, [proposal(0.001) for _ in range(4)], self.gates
        )
        checks = d1x.qualification_checks(summary, self.gates)
        self.assertFalse(checks["target_effect"])
        self.assertEqual(
            d1x.select_route(summary, summary, checks),
            "STOP_BARP_MCD_D1X_TARGET_EFFECT_TOO_WEAK",
        )

    def test_no_discovery_route_does_not_open_qualification(self) -> None:
        self.assertEqual(
            d1x.select_route(None, None, None),
            "STOP_BARP_MCD_D1X_DISCOVERY_NO_STABLE_ROUTE",
        )


if __name__ == "__main__":
    unittest.main()
