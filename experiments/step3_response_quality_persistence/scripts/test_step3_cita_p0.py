from __future__ import annotations

import hashlib
import math
import unittest

import torch

import step3_cita_p0 as p0


class FakeTokenizer:
    all_special_ids = [99]

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> str:
        del tokenize
        if not add_generation_prompt:
            return "full"
        return "direct" if enable_thinking is None else "forced"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        values = {
            "direct": [1],
            "forced": [1, 2],
            "full": [1, 2, 10, 11, 99, 12],
        }
        return {"input_ids": values[text]}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SerializationTests(unittest.TestCase):
    def test_no_think_prefix_and_response_boundary(self) -> None:
        feature = p0.serialized_response(FakeTokenizer(), "prompt", "response", 32)
        self.assertEqual(feature["direct_prefix_tokens"], 1)
        self.assertEqual(feature["prefix_tokens"], 2)
        self.assertEqual(feature["response_token_ids"], [10, 11, 99, 12])
        self.assertEqual(feature["semantic_mask"], [True, True, False, False])


class PacketTests(unittest.TestCase):
    @staticmethod
    def row(index: int) -> dict[str, object]:
        correct = f"correct {index}"
        wrong = [f"wrong {index} a", f"wrong {index} b"]
        response = f"response {index}"
        return {
            "ranking_id": f"r{index}",
            "sample_id": f"s{index}",
            "target_id": f"t{index}",
            "category": f"c{index % 2}",
            "correct_prompt": correct,
            "correct_prompt_sha256": digest(correct),
            "wrong_prompts": wrong,
            "wrong_prompt_sha256": [digest(value) for value in wrong],
            "response": response,
            "response_sha256": digest(response),
        }

    def test_packet_hashes_and_counts(self) -> None:
        result = p0.validate_packet_rows([self.row(0), self.row(1)], 2, 2, 2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["prompt_clusters"], 2)

    def test_packet_rejects_text_hash_drift(self) -> None:
        rows = [self.row(0), self.row(1)]
        rows[0]["response"] = "changed"
        with self.assertRaisesRegex(p0.CitaP0ValidationError, "response hash"):
            p0.validate_packet_rows(rows, 2, 2, 2)


class CacheTests(unittest.TestCase):
    def test_exact_budget_cache_is_valid(self) -> None:
        source = [{"target_id": "t1", "response_sha256": "abc"}]
        cache = [
            {
                "target_id": "t1",
                "response_sha256": "abc",
                "response_token_ids": [10, 11, 99],
                "semantic_mask": [True, True, False],
                "correct_nll": [1.0, 2.0, 3.0],
                "wrong_nlls": [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                "advantages": [1.5, 1.5],
                "sham_scores": [1.0, 1.0],
                "uniform_weights": [1.0, 1.0, 1.0],
                "cita_weights": [0.25, 1.75, 1.0],
                "sham_weights": [1.75, 0.25, 1.0],
                "token_alignment_exact": True,
                "weight_multiset_equal": True,
            }
        ]
        observed = p0.validate_weight_cache(cache, source, {})
        self.assertTrue(observed["all_weight_multisets_equal"])

    def test_termination_weight_cannot_absorb_budget(self) -> None:
        source = [{"target_id": "t1", "response_sha256": "abc"}]
        cache = [
            {
                "target_id": "t1",
                "response_sha256": "abc",
                "response_token_ids": [10, 11, 99],
                "semantic_mask": [True, True, False],
                "correct_nll": [1.0, 2.0, 3.0],
                "wrong_nlls": [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                "advantages": [1.5, 1.5],
                "sham_scores": [1.0, 1.0],
                "uniform_weights": [1.0, 1.0, 1.0],
                "cita_weights": [0.25, 1.75, 2.0],
                "sham_weights": [1.75, 0.25, 1.0],
                "token_alignment_exact": True,
                "weight_multiset_equal": True,
            }
        ]
        with self.assertRaisesRegex(p0.CitaP0ValidationError, "budget"):
            p0.validate_weight_cache(cache, source, {})


class GradientTests(unittest.TestCase):
    def test_structural_zero_is_recorded_but_not_missing(self) -> None:
        a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        a.grad = torch.zeros_like(a)
        b.grad = torch.tensor([0.0, 2.0])
        gradients, coverage = p0.capture_gradients([("lora_A", a), ("lora_B", b)])
        self.assertEqual(coverage["missing_gradient_tensors"], [])
        self.assertEqual(coverage["nonzero_gradient_tensors"], 1)
        self.assertEqual(set(gradients), {"lora_A", "lora_B"})

    def test_registered_gradient_comparisons(self) -> None:
        def vector(*values: float) -> dict[str, torch.Tensor]:
            return {"p": torch.tensor(values)}

        gradients = {
            "calibration": {
                "uniform": vector(1.0, 0.0, 0.0),
                "cita": vector(1.0, 1.0, 0.0),
                "sham": vector(1.0, 0.0, 1.0),
            },
            "validation": {
                "uniform": vector(2.0, 0.0, 0.0),
                "cita": vector(2.0, 1.0, 0.0),
                "sham": vector(2.0, 0.0, 1.0),
            },
        }
        result = p0.gradient_comparisons(gradients)
        self.assertAlmostEqual(result["calibration_cita_delta_to_uniform_ratio"], 1.0)
        self.assertAlmostEqual(result["calibration_cita_uniform_cosine"], 2 ** -0.5)
        self.assertAlmostEqual(result["calibration_cita_sham_delta_cosine"], 0.0)
        self.assertAlmostEqual(result["calibration_validation_cita_delta_cosine"], 1.0)


class OutputTests(unittest.TestCase):
    def test_nonfinite_json_values_are_explicit_nulls(self) -> None:
        result = p0.json_safe({"a": math.nan, "b": [math.inf, 1.0]})
        self.assertEqual(result, {"a": None, "b": [None, 1.0]})


if __name__ == "__main__":
    unittest.main()
