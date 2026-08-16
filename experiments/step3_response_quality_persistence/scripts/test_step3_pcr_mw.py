from __future__ import annotations

import unittest

import torch

import step3_pcr_mw_common as pcr


class PcrMwTests(unittest.TestCase):
    def test_mismatch_builder_is_deterministic_and_preserves_response(self) -> None:
        rows = []
        for prompt_index in range(3):
            for target_index in range(2):
                rows.append(
                    {
                        "sample_id": f"s{prompt_index}",
                        "target_id": f"t{prompt_index}_{target_index}",
                        "category": "shared",
                        "prompt": f"prompt {prompt_index}",
                        "response": f"response {prompt_index} {target_index}",
                    }
                )
        first = pcr.build_mismatch_rows(rows, 3)
        second = pcr.build_mismatch_rows(rows, 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(len(set(row["wrong_sample_ids"])) == 2 for row in first))
        self.assertTrue(all(row["sample_id"] not in row["wrong_sample_ids"] for row in first))

    def test_builder_excludes_categories_without_three_prompts(self) -> None:
        rows = [
            {"sample_id": f"s{i}", "target_id": f"t{i}", "category": "small", "prompt": str(i), "response": str(i)}
            for i in range(2)
        ]
        self.assertEqual(pcr.build_mismatch_rows(rows, 3), [])

    def test_reference_centering_starts_at_zero_margin_change(self) -> None:
        correct = torch.tensor(2.0, requires_grad=True)
        wrong = torch.tensor([3.0, 4.0], requires_grad=True)
        reference = (wrong - correct).detach()
        detail = pcr.ranking_losses(correct, wrong, reference, 0.1)
        self.assertTrue(torch.equal(detail["centered_margins"], torch.zeros(2)))
        self.assertAlmostEqual(float(detail["mean_loss"]), 0.693147, places=5)

    def test_mean_worst_emphasizes_larger_violation(self) -> None:
        correct = torch.tensor(0.0)
        wrong = torch.tensor([2.0, -1.0])
        reference = torch.zeros(2)
        detail = pcr.ranking_losses(correct, wrong, reference, 0.1)
        self.assertGreater(float(detail["worst_loss"]), float(detail["mean_loss"]))
        self.assertGreater(float(detail["mean_worst_loss"]), float(detail["mean_loss"]))

    def test_linear_quantile(self) -> None:
        self.assertEqual(pcr.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)


if __name__ == "__main__":
    unittest.main()
