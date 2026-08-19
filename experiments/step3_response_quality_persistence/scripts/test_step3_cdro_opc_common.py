from __future__ import annotations

import unittest

import numpy as np

import step3_cdro_opc_common as cdro


class EllipsoidTests(unittest.TestCase):
    def test_factor_reconstructs_sample_covariance(self) -> None:
        fitted = cdro.fit_ellipsoidal_moments(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]], ridge=1e-4
        )
        factor = np.asarray(fitted["factor"])
        covariance = np.asarray(fitted["covariance"])
        self.assertTrue(np.allclose(factor.T @ factor, covariance, atol=1e-12))
        self.assertLess(fitted["factorization_relative_error"], 1e-12)

    def test_support_function_matches_constructed_worst_case(self) -> None:
        mean = np.asarray([0.8, 0.4])
        factor = np.asarray([[0.3, 0.0], [0.1, 0.2]])
        point = np.asarray([0.5, -0.25])
        radius = 1.7
        transformed = factor @ point
        worst_u = -radius * transformed / np.linalg.norm(transformed)
        worst_score = float((mean + factor.T @ worst_u) @ point)
        self.assertAlmostEqual(
            cdro.ellipsoidal_lower_bound(mean, factor, point, radius),
            worst_score,
            places=12,
        )


class ConvexControllerTests(unittest.TestCase):
    @staticmethod
    def constraints(uncertainty_radius: float = 0.5) -> list[dict]:
        return [
            {
                "name": "retention",
                "mean": [1.0, 1.0],
                "factor": [[0.2, 0.0], [0.0, 0.2]],
                "uncertainty_radius": uncertainty_radius,
                "floor": 0.3,
            }
        ]

    def test_candidate_passes_full_kkt_audit(self) -> None:
        point, audit = cdro.solve_cdro_opc(
            behavior_target=[0.8, 0.7],
            metric=[[1.0, 0.0], [0.0, 1.0]],
            quality_score=[1.0, 0.2],
            safety_score=[0.1, 1.0],
            protected_constraints=self.constraints(),
            fallback=[0.4, 0.4],
            trust_radius=1.2,
        )
        self.assertEqual(audit["route"], "candidate")
        self.assertTrue(audit["optimization_valid"])
        self.assertLessEqual(audit["primal_violation"], 1e-7)
        self.assertLessEqual(audit["normalized_stationarity_residual"], 1e-5)
        self.assertLessEqual(audit["complementarity_residual"], 1e-5)
        self.assertGreater(float(np.linalg.norm(point)), 0.0)
        self.assertTrue(audit["objective_dominates_fallback"])

    def test_larger_uncertainty_radius_is_more_conservative(self) -> None:
        arguments = {
            "behavior_target": [0.95, 0.25],
            "metric": [[1.0, 0.0], [0.0, 1.0]],
            "quality_score": [1.0, 0.0],
            "safety_score": [1.0, 0.1],
            "fallback": [0.4, 0.6],
            "trust_radius": 1.2,
        }
        low_constraint = [
            {
                "name": "anisotropic_retention",
                "mean": [0.0, 1.0],
                "factor": [[0.2, 0.0]],
                "uncertainty_radius": 0.0,
                "floor": 0.4,
            }
        ]
        high_constraint = [
            {
                **low_constraint[0],
                "uncertainty_radius": 2.0,
            }
        ]
        low, low_audit = cdro.solve_cdro_opc(
            protected_constraints=low_constraint, **arguments
        )
        high, high_audit = cdro.solve_cdro_opc(
            protected_constraints=high_constraint, **arguments
        )
        self.assertTrue(low_audit["optimization_valid"])
        self.assertTrue(high_audit["optimization_valid"])
        low_bound = cdro.ellipsoidal_lower_bound(
            [0.0, 1.0], [[0.2, 0.0]], low, 2.0
        )
        high_bound = cdro.ellipsoidal_lower_bound(
            [0.0, 1.0], [[0.2, 0.0]], high, 2.0
        )
        self.assertGreaterEqual(high_bound + 1e-7, low_bound)
        self.assertGreaterEqual(high_bound, 0.4 - 1e-7)
        self.assertGreater(high[1], low[1])

    def test_unsafe_fallback_is_rejected_before_optimization(self) -> None:
        with self.assertRaises(cdro.CdroOpcMathError):
            cdro.solve_cdro_opc(
                behavior_target=[0.8, 0.7],
                metric=[[1.0, 0.0], [0.0, 1.0]],
                quality_score=[1.0, 0.0],
                safety_score=[0.0, 1.0],
                protected_constraints=[
                    {
                        "name": "impossible",
                        "mean": [1.0, 1.0],
                        "factor": [[0.0, 0.0]],
                        "uncertainty_radius": 1.0,
                        "floor": 1.0,
                    }
                ],
                fallback=[0.1, 0.1],
                trust_radius=1.0,
            )

    def test_indefinite_metric_is_rejected(self) -> None:
        with self.assertRaises(cdro.CdroOpcMathError):
            cdro.solve_cdro_opc(
                behavior_target=[0.8, 0.7],
                metric=[[1.0, 2.0], [2.0, 1.0]],
                quality_score=[1.0, 0.0],
                safety_score=[0.0, 1.0],
                protected_constraints=self.constraints(),
                fallback=[0.4, 0.4],
                trust_radius=1.2,
            )

    def test_centered_robust_change_keeps_fallback_feasible(self) -> None:
        fallback = [0.4, -0.2]
        point, audit = cdro.solve_cdro_opc(
            behavior_target=[0.8, 0.4],
            metric=[[1.0, 0.0], [0.0, 1.0]],
            quality_score=[1.0, 0.0],
            safety_score=[0.0, 1.0],
            protected_constraints=[
                {
                    "name": "signed_change",
                    "mean": [0.0, 1.0],
                    "factor": [[0.2, 0.0]],
                    "center": fallback,
                    "uncertainty_radius": 1.0,
                    "floor": -0.05,
                }
            ],
            fallback=fallback,
            trust_radius=1.2,
        )
        self.assertTrue(audit["fallback_feasible"])
        self.assertGreaterEqual(
            audit["fallback_constraint_values"]["protected:signed_change"], 0.05
        )
        self.assertTrue(np.isfinite(point).all())


class ConformalPolicyTests(unittest.TestCase):
    def test_twenty_units_certify_complete_policy(self) -> None:
        raw = [{"a": 1.0, "b": 2.0} for _ in range(20)]
        candidate = [
            {"a": 0.98 - 0.0005 * index, "b": 1.96 - 0.001 * index}
            for index in range(20)
        ]
        policy = cdro.conformal_transport_policy(
            raw,
            candidate,
            protected_names=["a", "b"],
            retention_floor=0.95,
            alpha=0.05,
            fallback_nonconformity=-0.05,
        )
        self.assertEqual(policy["order_statistic_rank"], 20)
        self.assertEqual(policy["policy_action"], "candidate")
        self.assertAlmostEqual(
            policy["finite_sample_marginal_policy_protection_lower_bound"],
            20 / 21,
        )
        self.assertFalse(
            policy["conditional_on_candidate_acceptance_coverage_claimed"]
        )

    def test_one_bad_tail_unit_routes_to_fallback(self) -> None:
        raw = [{"a": 1.0} for _ in range(20)]
        candidate = [{"a": 0.97} for _ in range(19)] + [{"a": 0.90}]
        policy = cdro.conformal_transport_policy(
            raw,
            candidate,
            protected_names=["a"],
            retention_floor=0.95,
            alpha=0.05,
            fallback_nonconformity=-0.05,
        )
        self.assertEqual(policy["policy_action"], "fallback")
        self.assertFalse(policy["candidate_accepted"])

    def test_standardized_change_handles_negative_and_near_zero_fallbacks(self) -> None:
        fallback = [{"risk": -0.1, "tarc": 0.001} for _ in range(20)]
        candidate = [{"risk": -0.102, "tarc": 0.0005} for _ in range(20)]
        policy = cdro.conformal_standardized_change_policy(
            fallback,
            candidate,
            protected_names=["risk", "tarc"],
            allowed_relative_drop=0.05,
            score_scale_floor=0.02,
            alpha=0.05,
        )
        self.assertEqual(policy["policy_action"], "candidate")
        self.assertTrue(policy["negative_fallback_scores_supported"])
        self.assertAlmostEqual(policy["per_unit_scales"][0]["risk"], 0.1)
        self.assertAlmostEqual(policy["per_unit_scales"][0]["tarc"], 0.02)


if __name__ == "__main__":
    unittest.main()
