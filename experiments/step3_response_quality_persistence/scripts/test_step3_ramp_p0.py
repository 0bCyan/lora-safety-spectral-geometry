from __future__ import annotations

import hashlib
import unittest

import torch

import step3_ramp_p0 as p0


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SourceTests(unittest.TestCase):
    @staticmethod
    def row(prompt: int, target: int) -> dict[str, object]:
        prompt_text = f"prompt {prompt}"
        response = f"response {prompt} {target}"
        return {
            "sample_id": f"s{prompt}",
            "target_id": f"t{prompt}-{target}",
            "category": f"c{prompt % 2}",
            "prompt": prompt_text,
            "response": response,
            "prompt_hash": digest(prompt_text),
            "target_hash": digest(response),
            "step2g_split": "safety_train",
            "reviewer_decision": "approve",
        }

    def test_two_approved_targets_per_prompt_are_required(self) -> None:
        rows = [self.row(prompt, target) for prompt in range(2) for target in range(2)]
        result = p0.validate_source_rows(rows, 4, 2, 2)
        self.assertEqual(result["targets_per_prompt"], 2)

    def test_hash_drift_is_rejected(self) -> None:
        rows = [self.row(prompt, target) for prompt in range(2) for target in range(2)]
        rows[0]["response"] = "changed"
        with self.assertRaisesRegex(p0.RampP0ValidationError, "target hash"):
            p0.validate_source_rows(rows, 4, 2, 2)


class ObjectiveTests(unittest.TestCase):
    def test_prefix_margin_uses_first_semantic_positions(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 2.0],
                [3.0, 0.0, 1.0],
                [0.0, 2.0, 4.0],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([0, 1, 2, 2])
        loss, metrics = p0.response_objectives(
            logits, targets, [0, 1, 2, 3], 2, "prefix"
        )
        self.assertEqual(metrics["margin_terms"], 2)
        self.assertEqual(metrics["certified_token_rate"], 1.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_suffix_margin_uses_last_semantic_positions(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 2.0],
                [3.0, 0.0, 1.0],
                [0.0, 2.0, 4.0],
            ]
        )
        targets = torch.tensor([0, 1, 2, 2])
        _, metrics = p0.response_objectives(
            logits, targets, [0, 1, 2, 3], 2, "suffix"
        )
        self.assertEqual(metrics["margin_terms"], 2)
        self.assertEqual(metrics["certified_token_rate"], 0.5)

    def test_uniform_objective_scores_all_response_tokens(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        loss, metrics = p0.response_objectives(
            logits, targets, [0], 1, "uniform"
        )
        self.assertAlmostEqual(float(loss), metrics["mean_nll"])


if __name__ == "__main__":
    unittest.main()
