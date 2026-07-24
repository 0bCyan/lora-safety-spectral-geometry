from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import torch

import step3cm_c3_common as common


class Step3CMC3CommonTests(unittest.TestCase):
    def test_masked_mean_padding_and_truncation(self) -> None:
        core = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        right = torch.cat([core, torch.full((1, 2, 2), 99.0)], dim=1)
        left = torch.cat([torch.full((1, 2, 2), 99.0), core], dim=1)
        self.assertTrue(torch.allclose(common.masked_mean(right, torch.tensor([[1, 1, 0, 0]])), torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.allclose(common.masked_mean(left, torch.tensor([[0, 0, 1, 1]])), torch.tensor([[2.0, 3.0]])))
        truncated = common.masked_mean(core[:, :1], torch.tensor([[1]]))
        self.assertTrue(torch.allclose(truncated, torch.tensor([[1.0, 2.0]])))

    def test_positive_and_negative_masks(self) -> None:
        positives = common.positive_mask(["p1", "p2"], ["p1", "p1", "p2", "p3"])
        self.assertEqual(positives.tolist(), [[True, True, False, False], [False, False, True, False]])
        hard = common.hard_negative_mask(["p1"], ["a"], ["p1", "p2", "p3"], ["a", "a", "b"])
        self.assertEqual(hard.tolist(), [[False, True, False]])

    def test_multi_positive_finite_gradient_and_hard_disable_equivalence(self) -> None:
        torch.manual_seed(7)
        base = torch.nn.Linear(4, 4)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        lora = torch.nn.Linear(4, 4, bias=False)
        queries = common.l2_normalize(base(torch.randn(2, 4)) + lora(torch.randn(2, 4)))
        keys = common.l2_normalize(torch.randn(4, 4))
        positives = common.positive_mask(["p1", "p2"], ["p1", "p1", "p2", "p3"])
        hard = common.hard_negative_mask(["p1", "p2"], ["a", "b"], ["p1", "p1", "p2", "p3"], ["a", "a", "b", "a"])
        loss_a, _ = common.multi_positive_loss(queries, keys, positives, 0.1)
        loss_b, _ = common.multi_positive_loss(queries, keys, positives, 0.1, hard, 0.0)
        self.assertTrue(torch.allclose(loss_a, loss_b))
        loss_a.backward()
        self.assertTrue(torch.isfinite(lora.weight.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(lora.weight.grad)), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

    def test_singleton_category_has_no_hard_row(self) -> None:
        queries = common.l2_normalize(torch.tensor([[1.0, 0.0]]))
        keys = common.l2_normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        positives = common.positive_mask(["p1"], ["p1", "p2"])
        hard = common.hard_negative_mask(["p1"], ["solo"], ["p1", "p2"], ["solo", "other"])
        _, detail = common.multi_positive_loss(queries, keys, positives, 0.1, hard, 1.0)
        self.assertEqual(detail["hard_rows"], 0)

    def test_queue_wraparound_and_determinism(self) -> None:
        first = common.DeterministicQueue(3, 2)
        second = common.DeterministicQueue(3, 2)
        values = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        ids = [f"p{i}" for i in range(5)]
        cats = ["a"] * 5
        first.enqueue(values, ids, cats)
        second.enqueue(values, ids, cats)
        out_a = first.ordered()
        out_b = second.ordered()
        self.assertTrue(torch.equal(out_a[0], values[-3:]))
        self.assertEqual(out_a[1], ids[-3:])
        self.assertTrue(torch.equal(out_a[0], out_b[0]))

    def test_temperature_and_eta_rules(self) -> None:
        rows = [
            {"temperature": 0.05, "median_positive_probability": 0.1},
            {"temperature": 0.1, "median_positive_probability": 0.19},
            {"temperature": 0.2, "median_positive_probability": 0.21},
        ]
        selected = common.choose_temperature(rows, 0.05, 0.5, 0.2)
        self.assertEqual(selected["temperature"], 0.1)
        eta, ratio = common.calibrate_eta(2.0, 4.0, 0.25)
        self.assertTrue(math.isclose(eta, 0.5))
        self.assertTrue(math.isclose(ratio, 0.25))

    def test_projection_and_retrieval_determinism(self) -> None:
        head_a = common.projection_head(4, 3, 11)
        head_b = common.projection_head(4, 3, 11)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(head_a.state_dict().values(), head_b.state_dict().values())))
        similarities = torch.tensor([[0.9, 0.8, 0.1], [0.2, 0.1, 0.7]])
        metrics = common.retrieval_metrics(similarities, ["p1", "p2"], ["p1", "p1", "p2"])
        self.assertEqual(metrics["top1"], 1.0)
        self.assertEqual(metrics["multi_positive_recall_at5"], 1.0)

    def test_no_optimizer_or_scheduler_construction(self) -> None:
        root = Path(__file__).resolve().parent
        for filename in ["step3cm_c3_common.py", "run_step3cm_c3_no_update.py"]:
            source = (root / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            forbidden = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    text = ast.unparse(node.func).lower()
                    if "optim" in text or "scheduler" in text:
                        forbidden.append(text)
            self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
