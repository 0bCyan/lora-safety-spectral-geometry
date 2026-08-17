from __future__ import annotations

import math
import unittest

import torch

import step3_nptmcd_affine_common as affine
import step3_ramp_common as gradient_math


def vector(values: list[float]) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(values, dtype=torch.float64)}


class AffineNullspaceTests(unittest.TestCase):
    def test_shared_projection_and_affine_tangent_are_protected(self) -> None:
        raw = vector([0.0, 2.0, 0.0, 0.0])
        protected = [vector([1.0, 0.0, 0.0, 0.0])]
        targets = [
            ("a", vector([0.0, 0.0, 1.0, 0.0])),
            ("b", vector([0.0, 0.0, 1.0, 1.0])),
        ]
        tangent, candidate, audit = affine.affine_maxmin_target_tangent(
            raw,
            targets,
            {"a": 0.1, "b": 0.1},
            protected,
            reference_norm=2.0,
            theta=0.2,
            rcond=1e-12,
            refinement_steps=1,
            solver_tolerance=1e-9,
        )
        self.assertAlmostEqual(gradient_math.gradient_dot(tangent, raw), 0.0)
        self.assertAlmostEqual(
            gradient_math.gradient_dot(tangent, protected[0]), 0.0
        )
        self.assertAlmostEqual(gradient_math.gradient_norm(tangent), 2.0)
        self.assertAlmostEqual(gradient_math.gradient_norm(candidate), 2.0)
        self.assertGreater(audit["candidate_actual_minimum_score"], 0.0)
        self.assertLess(audit["solver"]["kkt_residual"], 1e-6)
        self.assertLess(abs(audit["candidate_actual_minus_predicted"]), 1e-6)

    def test_target_without_nullspace_component_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            affine.affine_maxmin_target_tangent(
                vector([0.0, 1.0]),
                [("a", vector([0.0, 2.0])), ("b", vector([0.0, 3.0]))],
                {"a": 0.1, "b": 0.1},
                [vector([1.0, 0.0])],
                reference_norm=1.0,
                theta=math.pi / 6.0,
                rcond=1e-12,
                refinement_steps=1,
                solver_tolerance=1e-9,
            )


if __name__ == "__main__":
    unittest.main()
