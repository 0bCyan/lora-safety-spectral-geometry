from __future__ import annotations

import unittest
from pathlib import Path

import torch

import step3ct_t2r_common as recovery
import step3ct_training_common as parent


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1.json"


class Step3CTT2RCommonTests(unittest.TestCase):
    def test_metadata_hard_coverage(self) -> None:
        audit = recovery.metadata_hard_coverage(REPO, parent.read_json(CONFIG))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["hard_active_positions_all"], 58)
        self.assertEqual(audit["hard_active_positions_eta_fit_excluded"], 4)
        self.assertEqual(audit["hard_active_positions_calibration"], 54)
        self.assertEqual(audit["hard_active_calibration_windows"], 17)

    def test_fit_positions_and_windows_are_exact(self) -> None:
        rows = [{"position": index} for index in range(80)]
        windows = recovery.calibration_windows(rows)
        self.assertEqual(len(windows), 18)
        self.assertEqual([row["position"] for window in windows for row in window], list(range(8, 80)))
        self.assertTrue(all(len(window) == 4 for window in windows))

    def test_fit_excluded_contamination_rejected(self) -> None:
        rows = [{"position": index} for index in range(7, 80)]
        with self.assertRaises(ValueError):
            recovery.calibration_windows(rows)

    def test_float64_sum_hash_and_serialization_order_invariance(self) -> None:
        first = {"b": torch.tensor([1.0, 2.0]), "a": torch.tensor([3.0])}
        second = {"a": torch.tensor([4.0]), "b": torch.tensor([5.0, 6.0])}
        total = recovery.sum_gradients([first, second])
        reordered = recovery.sum_gradients([
            {"a": first["a"], "b": first["b"]},
            {"b": second["b"], "a": second["a"]},
        ])
        self.assertEqual(total["a"].dtype, torch.float64)
        self.assertEqual(recovery.gradient_hash(total), recovery.gradient_hash(reordered))
        self.assertEqual(recovery.gradient_norm(total), recovery.gradient_norm(reordered))

    def test_fp16_scaled_coefficient_removal(self) -> None:
        true_gradient = {"x": torch.tensor([2.0, -3.0], dtype=torch.float32)}
        coefficient = 0.05 * 65536.0
        scaled = {"x": true_gradient["x"] * coefficient}
        total = recovery.zeros_like_float64(scaled)
        recovery.add_in_place(total, scaled, divisor=coefficient)
        self.assertTrue(torch.allclose(total["x"], true_gradient["x"].double()))

    def test_direction_and_residual(self) -> None:
        a = {"x": torch.tensor([1.0, 0.0], dtype=torch.float64)}
        b = {"x": torch.tensor([1.0, 1.0], dtype=torch.float64)}
        self.assertAlmostEqual(recovery.gradient_cosine(a, b), 2 ** -0.5)
        self.assertAlmostEqual(recovery.orthogonal_residual_ratio(a, b), 2 ** -0.5)
        self.assertTrue(recovery.gradients_finite_nonzero(a))


if __name__ == "__main__":
    unittest.main()
