from __future__ import annotations

import unittest

import torch

import step3_pfrsc_common as pfrsc
import step3_ramp_common as gradient_math


class ProtectedHomotopyTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[dict, dict]:
        baseline = {"p": torch.tensor([0.6, 0.4, 0.2], dtype=torch.float64)}
        certificates = {
            "risk": {"p": torch.tensor([1.0, 0.1, 0.0], dtype=torch.float64)},
            "route": {"p": torch.tensor([0.1, 1.0, 0.1], dtype=torch.float64)},
            "branch": {"p": torch.tensor([0.2, 0.1, 1.0], dtype=torch.float64)},
        }
        return baseline, certificates

    def run_search(self) -> tuple[dict, dict, dict]:
        baseline, certificates = self.fixture()
        return pfrsc.enumerate_protected_homotopy(
            baseline,
            certificates,
            budget_norm=0.2,
            auxiliary_scale=0.2,
            max_norm=1.0,
            adam_epsilon=1e-2,
            protected_names=["risk"],
            target_names=["route", "branch"],
            protected_retention=0.9,
            homotopy_fractions=[0.0, 0.5, 1.0],
            source_top_k=2,
        )

    def test_finite_family_counts_exact_map_evaluations(self) -> None:
        selected, raw, audit = self.run_search()
        self.assertEqual(audit["attempted_source_subsets"], 7)
        self.assertEqual(
            audit["feasible_source_subsets"] + audit["infeasible_source_subsets"],
            7,
        )
        self.assertEqual(audit["screened_source_count"], 2)
        self.assertEqual(audit["attempted_homotopy_candidates"], 5)
        self.assertEqual(audit["exact_map_evaluations"], 9)
        self.assertAlmostEqual(gradient_math.gradient_norm(selected), 0.2, places=9)
        self.assertAlmostEqual(gradient_math.gradient_norm(raw), 0.2, places=9)

    def test_raw_anchor_makes_both_certificates_hold(self) -> None:
        _, _, audit = self.run_search()
        selected = audit["selected"]
        raw = audit["raw_anchor"]
        self.assertGreaterEqual(selected["minimum_protected_retention"], 0.9)
        self.assertGreaterEqual(
            selected["minimum_target_exact_score"],
            raw["minimum_target_exact_score"],
        )
        self.assertGreaterEqual(audit["protected_feasible_candidates"], 1)

    def test_search_is_deterministic(self) -> None:
        first = self.run_search()[2]
        second = self.run_search()[2]
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(
            first["top_protected_candidates"], second["top_protected_candidates"]
        )

    def test_partition_and_fraction_registry_are_validated(self) -> None:
        baseline, certificates = self.fixture()
        with self.assertRaises(ValueError):
            pfrsc.enumerate_protected_homotopy(
                baseline,
                certificates,
                0.2,
                0.2,
                1.0,
                1e-2,
                ["risk"],
                ["route"],
                0.9,
                [0.0, 1.0],
                2,
            )
        with self.assertRaises(ValueError):
            pfrsc.enumerate_protected_homotopy(
                baseline,
                certificates,
                0.2,
                0.2,
                1.0,
                1e-2,
                ["risk"],
                ["route", "branch"],
                0.9,
                [0.0, 0.5, 0.5, 1.0],
                2,
            )


if __name__ == "__main__":
    unittest.main()
