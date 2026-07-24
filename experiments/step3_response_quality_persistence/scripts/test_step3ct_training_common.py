from __future__ import annotations

import math
import random
import unittest
import copy

import torch

import step3ct_training_common as common


class Step3CTTrainingCommonTests(unittest.TestCase):
    def test_masked_mean_left_right_padding_and_truncation(self) -> None:
        core = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        right = torch.cat([core, torch.full((1, 2, 2), 99.0)], dim=1)
        left = torch.cat([torch.full((1, 2, 2), 99.0), core], dim=1)
        self.assertTrue(torch.allclose(common.masked_mean(right, torch.tensor([[1, 1, 0, 0]])), torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.allclose(common.masked_mean(left, torch.tensor([[0, 0, 1, 1]])), torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.allclose(common.masked_mean(core[:, :1], torch.tensor([[1]])), torch.tensor([[1.0, 2.0]])))

    def test_two_positive_indexing_and_same_prompt_exclusion(self) -> None:
        positives = common.positive_mask(["p1"], ["p1", "p1", "p2", "p3"], torch.device("cpu"))
        hard = common.hard_negative_mask(
            ["p1"], ["a"], ["p1", "p1", "p2", "p3"], ["a", "a", "a", "b"], torch.device("cpu")
        )
        self.assertEqual(positives.tolist(), [[True, True, False, False]])
        self.assertEqual(hard.tolist(), [[False, False, True, False]])

    def test_hard_disabled_equals_global_and_singleton_falls_back(self) -> None:
        query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.5]]), dim=-1)
        keys = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1)
        positives = common.positive_mask(["p1"], ["p1", "p1", "p2"], torch.device("cpu"))
        no_hard = common.hard_negative_mask(
            ["p1"], ["solo"], ["p1", "p1", "p2"], ["solo", "solo", "other"], torch.device("cpu")
        )
        global_loss, _ = common.multi_positive_loss(query, keys, positives, 0.05)
        disabled_loss, _ = common.multi_positive_loss(query, keys, positives, 0.05, no_hard, 0.0)
        singleton_loss, detail = common.multi_positive_loss(query, keys, positives, 0.05, no_hard, 1.0)
        self.assertTrue(torch.equal(global_loss, disabled_loss))
        self.assertTrue(torch.equal(global_loss, singleton_loss))
        self.assertEqual(detail["hard_rows"], 0)

    def test_objective_parentheses(self) -> None:
        risk = torch.tensor(2.0)
        safety = torch.tensor(3.0)
        binding = torch.tensor(5.0)
        result = common.weighted_total_loss(risk, safety, binding, 0.2, 0.1)
        self.assertTrue(torch.allclose(result, torch.tensor(2.3)))
        outside_parentheses = 0.8 * risk + 0.2 * safety + 0.1 * binding
        self.assertFalse(torch.allclose(result, outside_parentheses))

    def test_queue_prefill_wraparound_exclusion_resume_and_hash(self) -> None:
        first = common.DeterministicQueue(3, 2)
        values = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        first.enqueue(values[:3], ["p1", "p2", "p3"], ["a", "a", "b"], ["t1", "t2", "t3"])
        filtered = first.ordered("p2")
        self.assertEqual(filtered[1], ["p1", "p3"])
        before = first.content_hash()
        restored = common.DeterministicQueue.from_state_dict(first.state_dict())
        self.assertEqual(before, restored.content_hash())
        evicted = restored.enqueue(values[3:], ["p4", "p5"], ["b", "c"], ["t4", "t5"])
        self.assertEqual(evicted, ["t1", "t2"])
        self.assertEqual(restored.ordered()[3], ["t3", "t4", "t5"])

    def test_deterministic_prefill_order(self) -> None:
        rows = [{"target_id": f"t{index}"} for index in range(10)]
        a = common.deterministic_prefill_rows(rows, 4, 7)
        b = common.deterministic_prefill_rows(list(reversed(rows)), 4, 7)
        self.assertEqual(a, b)

    def test_projection_frozen_and_rng_isolation(self) -> None:
        torch.manual_seed(9)
        random.seed(9)
        expected_torch = torch.rand(3)
        expected_python = random.random()
        torch.manual_seed(9)
        random.seed(9)
        with common.isolated_rng():
            _ = torch.rand(100)
            _ = [random.random() for _ in range(10)]
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertEqual(random.random(), expected_python)
        head_a = common.projection_head(4, 3, 11)
        head_b = common.projection_head(4, 3, 11)
        self.assertEqual(common.module_state_hash(head_a), common.module_state_hash(head_b))
        self.assertFalse(any(parameter.requires_grad for parameter in head_a.parameters()))

    def test_embedding_gradients_finite_and_hard_changes_direction(self) -> None:
        query = torch.nn.functional.normalize(torch.tensor([1.0, 0.2, 0.1]), dim=0)
        positives = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.1, 0.0], [0.9, 0.2, 0.1]]), dim=1
        )
        negatives = torch.nn.functional.normalize(
            torch.tensor([[0.2, 1.0, 0.0], [0.7, 0.6, 0.1], [0.0, 0.1, 1.0]]), dim=1
        )
        args = (query, positives, negatives, "p1", "a", ["p2", "p3", "p4"], ["a", "b", "c"], 0.05)
        global_grads, global_detail = common.binding_embedding_gradients(*args, 0.0)
        hard_grads, hard_detail = common.binding_embedding_gradients(*args, 1.0)
        self.assertTrue(all(torch.isfinite(value).all() for value in global_grads + hard_grads))
        self.assertGreater(global_detail["binding_loss"], 0.0)
        self.assertGreater(hard_detail["hard_binding_loss"], 0.0)
        cosine = torch.nn.functional.cosine_similarity(global_grads[0], hard_grads[0], dim=0)
        self.assertTrue(math.isfinite(float(cosine)))

    def test_toy_checkpoint_resume_is_deterministic(self) -> None:
        def build() -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
            torch.manual_seed(17)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=4
            )
            return model, optimizer, scheduler

        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[0.5, -0.5]])
        model, optimizer, scheduler = build()

        def step() -> float:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            return float(loss)

        step()
        checkpoint = {
            "model": {name: value.clone() for name, value in model.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": torch.random.get_rng_state(),
        }
        expected_loss = step()
        expected_state = {name: value.clone() for name, value in model.state_dict().items()}

        resumed_model, resumed_optimizer, resumed_scheduler = build()
        resumed_model.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        resumed_scheduler.load_state_dict(checkpoint["scheduler"])
        torch.random.set_rng_state(checkpoint["rng"])
        model, optimizer, scheduler = resumed_model, resumed_optimizer, resumed_scheduler
        resumed_loss = step()
        self.assertEqual(expected_loss, resumed_loss)
        self.assertTrue(
            all(torch.equal(expected_state[name], value) for name, value in model.state_dict().items())
        )


if __name__ == "__main__":
    unittest.main()
