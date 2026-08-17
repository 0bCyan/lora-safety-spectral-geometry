from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step3_nptmcd_common as npt
import step3_ramp_common as gradient_math


def vector(values: list[float]) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(values, dtype=torch.float64)}


class NullspaceProjectionTests(unittest.TestCase):
    def test_tangent_is_protected_null_raw_orthogonal_and_norm_matched(self) -> None:
        raw = vector([0.0, 1.0, 0.0])
        proposal = vector([1.0, 1.0, 2.0])
        protected = [vector([1.0, 0.0, 0.0])]
        tangent, audit = npt.protected_angular_tangent(
            raw,
            proposal,
            protected,
            rcond=1e-12,
            refinement_steps=1,
            minimum_retained_fraction=0.1,
        )
        self.assertAlmostEqual(gradient_math.gradient_dot(tangent, protected[0]), 0.0)
        self.assertAlmostEqual(gradient_math.gradient_dot(tangent, raw), 0.0)
        self.assertAlmostEqual(
            gradient_math.gradient_norm(tangent), gradient_math.gradient_norm(raw)
        )
        self.assertAlmostEqual(audit["retained_target_proposal_fraction"], 2 / math.sqrt(6))

    def test_projection_rejects_vanished_target_component(self) -> None:
        with self.assertRaises(ValueError):
            npt.protected_angular_tangent(
                vector([1.0, 0.0]),
                vector([0.0, 1.0]),
                [vector([0.0, 1.0])],
                rcond=1e-12,
                refinement_steps=1,
                minimum_retained_fraction=0.01,
            )


class AngularGeometryTests(unittest.TestCase):
    def test_protected_score_retention_is_cosine(self) -> None:
        raw = vector([1.0, 0.0])
        tangent = vector([0.0, 1.0])
        theta = 0.2
        candidate = npt.angular_displacement(raw, tangent, theta)
        protected = vector([2.0, 0.0])
        retention = gradient_math.gradient_dot(protected, candidate) / gradient_math.gradient_dot(
            protected, raw
        )
        self.assertAlmostEqual(retention, math.cos(theta))
        self.assertAlmostEqual(gradient_math.gradient_norm(candidate), 1.0)

    def test_selects_leftmost_effective_angle(self) -> None:
        result = npt.select_minimum_effective_angle(
            raw_panels=[{"p": 1.0, "t": 0.04}, {"p": 0.5, "t": 0.045}],
            tangent_panels=[{"p": 0.0, "t": 0.10}, {"p": 0.0, "t": 0.08}],
            protected_names=["p"],
            target_names=["t"],
            protected_retention_floor=0.95,
            design_cosine_floor=0.951,
            target_gain_floor=0.002,
            target_score_floor=0.05,
            numeric_tolerance=1e-10,
            angle_guard=1e-8,
        )
        self.assertTrue(result["eligible"])
        self.assertGreater(result["selected_theta"], 0.0)
        self.assertGreaterEqual(result["selected_target_score_floor"], 0.05)
        self.assertGreaterEqual(result["minimum_protected_retention"], 0.95)
        self.assertLess(result["maximum_protected_retention_identity_error"], 1e-12)

    def test_infeasible_target_returns_raw_anchor(self) -> None:
        result = npt.select_minimum_effective_angle(
            raw_panels=[{"p": 1.0, "t": 0.01}, {"p": 0.5, "t": 0.02}],
            tangent_panels=[{"p": 0.0, "t": 0.001}, {"p": 0.0, "t": 0.001}],
            protected_names=["p"],
            target_names=["t"],
            protected_retention_floor=0.95,
            design_cosine_floor=0.951,
            target_gain_floor=0.002,
            target_score_floor=0.05,
            numeric_tolerance=1e-10,
            angle_guard=1e-8,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["selected_theta"], 0.0)
        self.assertEqual(result["minimum_protected_retention"], 1.0)


if __name__ == "__main__":
    unittest.main()
