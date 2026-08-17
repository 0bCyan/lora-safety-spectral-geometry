from __future__ import annotations

import unittest

import torch

import step3_ojmcd_p0 as p0


class OptimizerBlockTests(unittest.TestCase):
    def test_baseline_uses_registered_risk_and_safety_mix(self) -> None:
        gradients = {
            "risk": {"p": torch.tensor([2.0, 0.0])},
            "uniform": {"p": torch.tensor([0.0, 3.0])},
        }
        baseline = p0._baseline_gradient(gradients, 0.2)
        self.assertTrue(torch.equal(baseline["p"], torch.tensor([1.6, 0.6])))


class SummaryTests(unittest.TestCase):
    @staticmethod
    def metric(worst: float, baseline: float = 0.1) -> dict:
        return {
            "minimum_exact_candidate_normalized_score": worst,
            "minimum_exact_increment_normalized_score": worst - baseline,
            "minimum_baseline_normalized_score": baseline,
            "increment_norm_fraction_of_baseline": 0.2,
            "linear_exact_cosine": 0.98,
            "linear_exact_norm_ratio": 1.02,
            "certificates": {
                name: {"exact_candidate_normalized_score": worst + 0.01}
                for name in p0.CERTIFICATES
            },
        }

    def test_summary_keeps_optimizer_unit_pairing(self) -> None:
        rows = [
            {
                "block_id": "calibration",
                "split": "calibration",
                "candidates": {
                    "raw_mcd": self.metric(0.2),
                    "jacobian_mcd": self.metric(0.25),
                    "ojmcd": self.metric(0.3),
                },
            }
        ]
        for index in range(7):
            rows.append(
                {
                    "block_id": f"validation_{index + 1:02d}",
                    "split": "validation",
                    "candidates": {
                        "raw_mcd": self.metric(0.20 + index * 0.001),
                        "jacobian_mcd": self.metric(0.25 + index * 0.001),
                        "ojmcd": self.metric(0.30 + index * 0.001),
                    },
                }
            )
        protocol = {
            "statistics": {"bootstrap_samples": 200, "bootstrap_seed": 7}
        }
        result = p0.summarize_rows(rows, protocol)
        comparison = result["paired_validation"]["ojmcd_minus_raw_mcd"]
        self.assertAlmostEqual(comparison["bootstrap"]["mean"], 0.1)
        self.assertEqual(comparison["positive_rate"], 1.0)
        self.assertEqual(
            comparison["sign_test"]["exact_two_sided_p"], 0.015625
        )


class RouteTests(unittest.TestCase):
    def test_route_passes_only_when_every_gate_passes(self) -> None:
        gates = {
            "integrity": True,
            "clipping_exercised": True,
            "dual_certificate": True,
            "secant_fidelity": True,
            "exact_common_descent": True,
            "optimizer_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "PASS_OJMCD_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        )
        gates["secant_fidelity"] = False
        self.assertEqual(
            p0.select_route(gates), "STOP_OJMCD_P0_SECANT_FIDELITY_FAILED"
        )


if __name__ == "__main__":
    unittest.main()
