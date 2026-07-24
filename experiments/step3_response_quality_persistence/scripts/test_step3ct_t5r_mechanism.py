from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

import torch

import step3ct_t5r_mechanism_common as common


class Step3CTT5RMechanismTests(unittest.TestCase):
    def synthetic_targets(self) -> list[dict[str, str]]:
        return [
            {
                "sample_id": f"p{prompt:02d}",
                "target_id": f"t{prompt:02d}_{variant}",
                "prompt": f"prompt {prompt}",
                "response": f"response {prompt} {variant}",
                "category": f"c{prompt % 4}",
            }
            for prompt in range(40)
            for variant in range(2)
        ]

    def test_exact_40_prompt_80_target_coverage(self) -> None:
        prompts, targets, order = common.canonical_diagnostic_rows(self.synthetic_targets())
        self.assertEqual((len(prompts), len(targets)), (40, 80))
        self.assertEqual((order["prompt_count"], order["target_count"]), (40, 80))
        with self.assertRaises(ValueError):
            common.canonical_diagnostic_rows(self.synthetic_targets()[:-1])

    def test_cross_arm_order_equality(self) -> None:
        _, _, order = common.canonical_diagnostic_rows(self.synthetic_targets())
        passed = common.validate_cross_arm_order({"C0R": order, "C3a": dict(order), "C3b": dict(order)})
        self.assertTrue(passed["passed"])
        changed = dict(order)
        changed["target_order_hash"] = "changed"
        self.assertFalse(common.validate_cross_arm_order({"C0R": order, "C3a": changed})["passed"])

    def test_rng_restoration(self) -> None:
        random.seed(91)
        torch.manual_seed(91)
        expected_python = random.random()
        expected_torch = torch.rand(3)
        random.seed(91)
        torch.manual_seed(91)
        with common.isolated_rng():
            random.seed(8)
            torch.manual_seed(8)
            _ = random.random()
            _ = torch.rand(20)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_diagnostic_eval_disables_dropout_and_restores_modes(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.9))
        model.train()
        value = torch.ones(1, 3)
        with common.diagnostic_eval(model):
            self.assertFalse(model.training)
            self.assertFalse(model[1].training)
            first = model(value)
            second = model(value)
            self.assertTrue(torch.equal(first, second))
        self.assertTrue(model.training)
        self.assertTrue(model[1].training)

    def test_retrieval_and_effective_rank_fixtures(self) -> None:
        prompt_rows = [
            {"sample_id": "p1", "category": "a"},
            {"sample_id": "p2", "category": "a"},
        ]
        target_rows = [
            {"sample_id": "p1", "target_id": "t1", "category": "a"},
            {"sample_id": "p1", "target_id": "t2", "category": "a"},
            {"sample_id": "p2", "target_id": "t3", "category": "a"},
            {"sample_id": "p2", "target_id": "t4", "category": "a"},
        ]
        prompts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        result = common.representation_metrics(prompts, targets, prompt_rows, target_rows, 100, 7, 10, 8)
        self.assertEqual(result["retrieval"]["top1"], 1.0)
        self.assertEqual(result["retrieval"]["multi_positive_recall_at5"], 1.0)
        self.assertTrue(math.isclose(result["prompt_geometry"]["effective_rank"], 1.0))

    def test_deterministic_representation_rerun_equality(self) -> None:
        prompt_rows = [{"sample_id": "p1", "category": "a"}, {"sample_id": "p2", "category": "b"}]
        target_rows = [
            {"sample_id": "p1", "target_id": "t1", "category": "a"},
            {"sample_id": "p1", "target_id": "t2", "category": "a"},
            {"sample_id": "p2", "target_id": "t3", "category": "b"},
            {"sample_id": "p2", "target_id": "t4", "category": "b"},
        ]
        prompts = torch.nn.functional.normalize(torch.tensor([[2.0, 1.0], [1.0, 2.0]]), dim=1)
        targets = torch.nn.functional.normalize(torch.tensor([[2.0, 1.0], [3.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), dim=1)
        first = common.representation_metrics(prompts, targets, prompt_rows, target_rows, 100, 11, 10, 12)
        second = common.representation_metrics(prompts, targets, prompt_rows, target_rows, 100, 11, 10, 12)
        self.assertEqual(first, second)

    def test_attention_mlp_grouping_and_grouped_cosine_fixture(self) -> None:
        attention = "base.layers.0.self_attn.q_proj.lora_A.default.weight"
        mlp = "base.layers.0.mlp.up_proj.lora_A.default.weight"
        safety = {attention: torch.tensor([1.0, 0.0]), mlp: torch.tensor([0.0, 1.0])}
        binding = {attention: torch.tensor([1.0, 0.0]), mlp: torch.tensor([0.0, -1.0])}
        result = common.grouped_gradient_metrics(safety, binding)
        self.assertEqual(result["group_cosine"]["attention"], 1.0)
        self.assertEqual(result["group_cosine"]["mlp"], -1.0)
        self.assertEqual(result["global_binding_safety_cosine"], 0.0)
        self.assertEqual({row["group"] for row in result["per_layer_group"]}, {"attention", "mlp"})

    def test_cross_arm_cosine_and_orthogonal_residual_fixture(self) -> None:
        name = "base.layers.1.self_attn.v_proj.lora_B.default.weight"
        c3a = {name: torch.tensor([1.0, 0.0])}
        c3b = {name: torch.tensor([1.0, 1.0])}
        result = common.cross_arm_gradient_metrics(c3a, c3b)
        self.assertTrue(math.isclose(result["global_cosine"], 1 / math.sqrt(2)))
        self.assertTrue(math.isclose(result["global_orthogonal_residual_ratio"], 1 / math.sqrt(2)))

    def test_source_artifact_immutability_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.bin"
            second = root / "b.json"
            first.write_bytes(b"adapter")
            second.write_text("{}", encoding="utf-8")
            before = common.hash_file_set([first, second], root)
            self.assertTrue(common.compare_file_sets(before, common.hash_file_set([first, second], root))["passed"])
            first.write_bytes(b"changed")
            self.assertFalse(common.compare_file_sets(before, common.hash_file_set([first, second], root))["passed"])

    def test_disposable_prefix_is_bounded_for_windows(self) -> None:
        prefixes = [common.disposable_prefix(label, arm) for label in (0, 16, 32, 64) for arm in range(3)]
        self.assertTrue(all(len(prefix) <= 12 for prefix in prefixes))
        self.assertTrue(all("C3" not in prefix for prefix in prefixes))
        with self.assertRaises(ValueError):
            common.disposable_prefix(-1, 0)

    def test_no_full_gradient_serialization(self) -> None:
        common.assert_no_full_gradient_serialization({"gradient_summary": {"norm": 2.0}, "rows": [1, 2]})
        with self.assertRaises(TypeError):
            common.assert_no_full_gradient_serialization({"payload": torch.ones(2)})
        with self.assertRaises(ValueError):
            common.assert_no_full_gradient_serialization({"gradients": {"x": [1.0, 2.0]}})

    def test_failed_grouped_metrics_are_withheld(self) -> None:
        result = {
            "passed": False,
            "arms": {
                "C3a": {"reproduction_passed": False, "grouped_metrics": {"attention_cosine": 0.5}},
                "C3b": {"reproduction_passed": True, "grouped_metrics": {"attention_cosine": 0.6}},
            },
            "c3a_c3b_direction": {"global_cosine": 0.9},
        }
        redacted = common.redact_failed_grouped_metrics(result)
        self.assertFalse(redacted["arms"]["C3a"]["grouped_metrics"]["registered"])
        self.assertEqual(redacted["arms"]["C3b"]["grouped_metrics"]["attention_cosine"], 0.6)
        self.assertFalse(redacted["c3a_c3b_direction"]["registered"])
        self.assertFalse(redacted["approximate_grouped_metrics_reported"])

    def test_registered_diagnostic_comparison(self) -> None:
        row = {
            "checkpoint_step": 0,
            "optimizer_step": 1,
            "gradient_unit": "unit",
            "positions": [0, 1, 2, 3],
            "microstep_count": 4,
            "safety_gradient_hash": "s",
            "binding_gradient_hash": "b",
            "hard_active_positions": 0,
            "finite_nonzero": True,
            "safety_gradient_norm": 1.0,
            "binding_gradient_norm": 2.0,
            "unweighted_binding_safety_ratio": 2.0,
            "eta_weighted_binding_safety_ratio": 0.5,
            "binding_safety_cosine": 0.1,
            "safety_energy": {"attention": 0.5, "mlp": 0.5},
            "binding_energy": {"attention": 0.4, "mlp": 0.6},
        }
        observed = dict(row)
        observed["safety_gradient_norm"] += 1e-7
        self.assertTrue(common.compare_registered_diagnostic(observed, row, 1e-5)["passed"])
        observed["safety_gradient_hash"] = "different"
        self.assertFalse(common.compare_registered_diagnostic(observed, row, 1e-5)["passed"])


if __name__ == "__main__":
    unittest.main()
