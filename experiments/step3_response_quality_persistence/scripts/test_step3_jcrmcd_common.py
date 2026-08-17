from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

import step3_jcrmcd_common as jcrmcd


class ConvexHullGeometryTests(unittest.TestCase):
    def test_distance_is_zero_for_point_inside_segment(self) -> None:
        distance, audit = jcrmcd.convex_hull_distance_from_gram(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.5, 0.5],
            target_squared_norm=0.5,
        )
        self.assertAlmostEqual(distance, 0.0, places=12)
        self.assertAlmostEqual(audit["weights"][0], 0.5, places=12)
        self.assertAlmostEqual(audit["weights"][1], 0.5, places=12)

    def test_jackknife_radius_is_maximum_leave_one_out_residual(self) -> None:
        points = [1.0, 2.0, 3.0]
        gram = [[left * right for right in points] for left in points]
        radius, audit = jcrmcd.jackknife_convex_hull_radius(gram)
        self.assertAlmostEqual(radius, 1.0, places=10)
        self.assertEqual(len(audit["residuals"]), 3)
        self.assertAlmostEqual(
            max(item["distance"] for item in audit["residuals"]), radius
        )


class RobustPathSelectionTests(unittest.TestCase):
    NAMES = ("protected", "target")

    @classmethod
    def record(
        cls,
        fraction: float,
        protected: float,
        target: float,
        norm: float,
        distance: float,
    ) -> dict:
        scores = {"protected": protected, "target": target}
        return {
            "fraction": fraction,
            "scores": scores,
            "minimum_score": min(scores.values()),
            "mean_score": sum(scores.values()) / len(scores),
            "displacement_norm_fraction": norm,
            "distance_from_raw_fraction": distance,
        }

    def test_raw_anchor_is_feasible_when_robust_lower_bound_is_positive(self) -> None:
        center = [self.record(0.0, 1.0, 0.6, 1.0, 0.0)]
        supports = [
            [self.record(0.0, 1.0, 0.6, 1.0, 0.0)],
            [self.record(0.0, 0.98, 0.58, 1.0, 0.0)],
            [self.record(0.0, 1.02, 0.62, 1.0, 0.0)],
        ]
        selected, audit = jcrmcd.select_robust_path_candidate(
            center,
            supports,
            {"protected": 0.05, "target": 0.05},
            ["protected"],
            ["target"],
            retention_floor=0.95,
        )
        self.assertEqual(selected, 0)
        self.assertTrue(audit["uncertainty_set_valid"])
        self.assertTrue(audit["raw_anchor"]["feasible"])
        self.assertGreaterEqual(
            audit["raw_anchor"]["minimum_protected_robust_margin"], 0.0
        )

    def test_protection_uses_expanded_hull_residual_inequality(self) -> None:
        rho = 0.9
        raw = torch.tensor([1.0, 0.0], dtype=torch.float64)
        candidate = torch.tensor([1.0, 0.1], dtype=torch.float64)
        units = [
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([0.8, 0.6], dtype=torch.float64),
            torch.tensor([0.8, -0.6], dtype=torch.float64),
        ]
        center = [
            self.record(0.0, 1.0, 1.0, float(raw.norm()), 0.0),
            self.record(
                1.0,
                1.0,
                1.3,
                float(candidate.norm()),
                float((candidate - raw).norm()),
            ),
        ]
        supports = []
        for unit in units:
            supports.append(
                [
                    self.record(
                        0.0,
                        float(torch.dot(unit, raw)),
                        float(torch.dot(unit, raw)),
                        float(raw.norm()),
                        0.0,
                    ),
                    self.record(
                        1.0,
                        float(torch.dot(unit, candidate)),
                        float(torch.dot(unit, candidate)) + 0.3,
                        float(candidate.norm()),
                        float((candidate - raw).norm()),
                    ),
                ]
            )
        epsilon = 0.1
        selected, audit = jcrmcd.select_robust_path_candidate(
            center,
            supports,
            {"protected": epsilon, "target": epsilon},
            ["protected"],
            ["target"],
            retention_floor=rho,
        )
        residual = candidate - rho * raw
        expected = min(float(torch.dot(unit, residual)) for unit in units)
        expected -= epsilon * float(residual.norm())
        self.assertEqual(selected, 1)
        actual = audit["selected"][
            "protected_robust_margins"
        ]["protected"]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_selects_best_feasible_target_candidate(self) -> None:
        center = [
            self.record(0.0, 1.0, 0.5, 1.0, 0.0),
            self.record(0.5, 0.96, 0.7, 0.96, 0.04),
            self.record(1.0, 0.80, 0.9, 0.80, 0.20),
        ]
        supports = [center, center, center]
        selected, audit = jcrmcd.select_robust_path_candidate(
            center,
            supports,
            {"protected": 0.0, "target": 0.0},
            ["protected"],
            ["target"],
            retention_floor=0.95,
        )
        self.assertEqual(selected, 1)
        self.assertEqual(audit["feasible_candidates"], 2)
        self.assertAlmostEqual(audit["selected"]["fraction"], 0.5)
        self.assertGreater(
            audit["selected_robust_target_gain_over_raw"], 0.0
        )


class SharedPathScoringTests(unittest.TestCase):
    def test_panels_share_identical_displacement_geometry(self) -> None:
        baseline = {"p": torch.tensor([0.6, -0.2], dtype=torch.float64)}
        raw = {"p": torch.tensor([0.2, 0.0], dtype=torch.float64)}
        target = {"p": torch.tensor([0.0, 0.2], dtype=torch.float64)}
        panels = {
            "left": {
                "a": {"p": torch.tensor([1.0, 0.0], dtype=torch.float64)},
                "b": {"p": torch.tensor([0.0, 1.0], dtype=torch.float64)},
            },
            "right": {
                "a": {"p": torch.tensor([1.0, 1.0], dtype=torch.float64)},
                "b": {"p": torch.tensor([1.0, -1.0], dtype=torch.float64)},
            },
        }
        with mock.patch.object(
            jcrmcd.frsc,
            "realized_displacement",
            wraps=jcrmcd.frsc.realized_displacement,
        ) as realized:
            scored = jcrmcd.score_path_panels(
                baseline,
                raw,
                target,
                panels,
                [0.0, 0.5, 1.0],
                budget_norm=0.2,
                auxiliary_scale=0.2,
                max_norm=1.0,
                adam_epsilon=1e-2,
                reference_norm=1.0,
            )
        self.assertEqual(realized.call_count, 3)
        for index in range(3):
            self.assertEqual(
                scored["left"][index]["displacement_norm_fraction"],
                scored["right"][index]["displacement_norm_fraction"],
            )
            self.assertEqual(
                scored["left"][index]["distance_from_raw_fraction"],
                scored["right"][index]["distance_from_raw_fraction"],
            )


if __name__ == "__main__":
    unittest.main()
