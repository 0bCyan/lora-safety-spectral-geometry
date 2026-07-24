from __future__ import annotations

import math
import unittest

import torch

import train_step3ct_tournament as trainer


class Step3CTT3DiagnosticTests(unittest.TestCase):
    def test_main_and_smoke_diagnostic_labels(self) -> None:
        registered = [0, 16, 32, 64]
        main = [
            trainer.diagnostic_label_for_step("main", step, 64, 64, registered)
            for step in range(64)
        ]
        self.assertEqual([(step, label) for step, label in enumerate(main) if label is not None], [
            (0, 0),
            (16, 16),
            (32, 32),
            (63, 64),
        ])
        self.assertEqual(
            [trainer.diagnostic_label_for_step("smoke", step, 2, 64, registered) for step in range(2)],
            [1, 2],
        )

    def test_optimizer_unit_uses_exact_gradient_sum(self) -> None:
        safety_parts = [
            {"lora_A": torch.tensor([float(index), 1.0])}
            for index in range(1, 5)
        ]
        binding_parts = [
            {"lora_A": torch.tensor([1.0, float(index)])}
            for index in range(1, 5)
        ]
        safety = None
        binding = None
        for safety_part, binding_part in zip(safety_parts, binding_parts):
            safety = trainer.accumulate_gradient_unit(safety, safety_part)
            binding = trainer.accumulate_gradient_unit(binding, binding_part)
        assert safety is not None
        assert binding is not None
        self.assertTrue(torch.equal(safety["lora_A"], torch.tensor([10.0, 4.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(binding["lora_A"], torch.tensor([4.0, 10.0], dtype=torch.float64)))
        row = trainer.optimizer_unit_diagnostic(16, 17, [64, 65, 66, 67], safety, binding, 0.25, 2)
        self.assertEqual(row["microstep_count"], 4)
        self.assertEqual(row["hard_active_positions"], 2)
        self.assertTrue(math.isclose(row["eta_weighted_binding_safety_ratio"], 0.25))
        self.assertTrue(row["finite_nonzero"])

    def test_control_has_zero_binding_ratio(self) -> None:
        safety = {"lora_A": torch.tensor([3.0, 4.0], dtype=torch.float64)}
        row = trainer.optimizer_unit_diagnostic(1, 1, [0, 1, 2, 3], safety, None, 0.0, 0)
        self.assertEqual(row["binding_gradient_norm"], 0.0)
        self.assertEqual(row["eta_weighted_binding_safety_ratio"], 0.0)
        self.assertEqual(row["binding_gradient_hash"], "")
        self.assertTrue(row["finite_nonzero"])


if __name__ == "__main__":
    unittest.main()
