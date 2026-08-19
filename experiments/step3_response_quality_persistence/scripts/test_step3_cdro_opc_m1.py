from __future__ import annotations

import unittest

import numpy as np
import torch

import step3_cdro_opc_m1 as m1


class ProtectedDesignTests(unittest.TestCase):
    def test_calibration_is_disjoint_and_fit_is_complete(self) -> None:
        targets = [
            {"sample_id": f"s{index:02d}", "category": f"c{index % 8}"}
            for index in range(40)
        ]
        historical = {
            name: {
                "safety_sample_ids": [f"s{offset + index:02d}" for index in range(4)],
                "risk_row_indices": list(range(offset, offset + 4)),
            }
            for name, offset in zip(
                ("train", "calibration", "mechanism_dev"), (0, 4, 8), strict=True
            )
        }
        design = m1.protected_design(targets, 64, 17, historical)
        self.assertEqual(len(design["fit_blocks"]), 6)
        self.assertEqual(len(design["calibration_units"]), 20)
        self.assertTrue(design["calibration_safety_disjoint_from_historical"])
        self.assertTrue(design["calibration_risk_disjoint_from_historical"])
        self.assertEqual(
            len(
                {
                    row["safety_sample_ids"][0]
                    for row in design["calibration_units"]
                }
            ),
            20,
        )


class BasisTests(unittest.TestCase):
    def test_basis_is_orthonormal_and_reconstructs_span(self) -> None:
        proposals = {
            "a": {"p": torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)},
            "b": {"p": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)},
            "c": {"p": torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)},
        }
        basis, audit = m1.orthonormal_update_basis(proposals, 1e-10, 3)
        self.assertEqual(audit["retained_rank"], 3)
        self.assertLess(audit["orthonormality_max_error"], 1e-12)
        reference = m1.gradient_math.gradient_norm(proposals["b"])
        value = m1.coordinates(basis, proposals["b"], reference)
        reconstructed = m1.map_from_coordinates(basis, value, reference)
        self.assertTrue(torch.allclose(reconstructed["p"], proposals["b"]["p"], atol=1e-12))
        self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0)


class RouteTests(unittest.TestCase):
    def test_only_full_conjunction_passes(self) -> None:
        gates = {
            "structural": True,
            "basis": True,
            "solver": True,
            "nontrivial": True,
            "conformal_transport": True,
            "behavior": True,
            "state_restored": True,
            "zero_step": True,
        }
        self.assertEqual(
            m1.route_from_gates(gates), "PASS_CDRO_OPC_M1_EXPLORATORY_TRANSPORT"
        )
        gates["conformal_transport"] = False
        self.assertEqual(m1.route_from_gates(gates), "STOP_CDRO_OPC_M1_TRANSPORT")


if __name__ == "__main__":
    unittest.main()
