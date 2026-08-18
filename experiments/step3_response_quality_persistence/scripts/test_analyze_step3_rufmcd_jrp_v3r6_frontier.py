from __future__ import annotations

import copy
import unittest

import analyze_step3_rufmcd_jrp_v3r6_frontier as audit


def fixture(*, joint_exists: bool, gap: float | None) -> dict:
    return {
        "name": "base",
        "analysis_status": "base",
        "facet_counts": {"fit_protected": 48, "fit_target": 36},
        "joint_feasible_angle_exists": joint_exists,
        "angular_incompatibility_gap": gap,
    }


class RufmcdJrpV3r6FrontierTests(unittest.TestCase):
    def test_range_rejects_empty_or_nonfinite_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and nonempty"):
            audit._range([])
        with self.assertRaisesRegex(ValueError, "finite and nonempty"):
            audit._range([float("nan")])

    def test_relaxed_disjoint_intervals_prove_full_set_impossible(self) -> None:
        result = audit.decorate(fixture(joint_exists=False, gap=0.04))
        proof = result["relaxed_fit_proof"]
        self.assertTrue(proof["strict_interval_separation"])
        self.assertEqual(proof["full_registered_fit_facets"], 96)
        self.assertTrue(result["interpretation"]["direction_change_required"])

    def test_relaxed_joint_interval_cannot_prove_full_set_impossible(self) -> None:
        result = audit.decorate(fixture(joint_exists=True, gap=-0.01))
        self.assertFalse(
            result["relaxed_fit_proof"]["strict_interval_separation"]
        )

    def test_wrong_relaxed_facet_count_is_rejected(self) -> None:
        value = fixture(joint_exists=False, gap=0.04)
        value = copy.deepcopy(value)
        value["facet_counts"]["fit_target"] = 35
        with self.assertRaisesRegex(ValueError, "fit-facet count"):
            audit.decorate(value)


if __name__ == "__main__":
    unittest.main()
