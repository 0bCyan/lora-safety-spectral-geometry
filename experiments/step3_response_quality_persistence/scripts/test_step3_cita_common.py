from __future__ import annotations

import unittest

import torch

import step3_cita_common as cita


class WeightTests(unittest.TestCase):
    def test_rank_weights_have_fixed_bounds_and_mean(self) -> None:
        weights = cita.linear_rank_weights([3.0, 1.0, 2.0, 4.0])
        self.assertEqual(min(weights), 0.25)
        self.assertEqual(max(weights), 1.75)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)

    def test_stable_ties_use_position(self) -> None:
        weights = cita.linear_rank_weights([1.0, 1.0, 1.0])
        self.assertEqual(weights, [0.25, 1.0, 1.75])

    def test_cita_and_sham_preserve_exact_weight_multiset(self) -> None:
        result = cita.token_weight_rows(
            [2.0, 3.0, 4.0, 1.0],
            [[3.0, 2.0, 5.0, 1.5], [4.0, 4.0, 4.5, 2.0]],
            [True, True, True, False],
        )
        self.assertEqual(
            sorted(result["semantic_cita_weights"]),
            sorted(result["semantic_sham_weights"]),
        )
        self.assertEqual(result["cita_weights"][-1], 1.0)
        self.assertEqual(result["sham_weights"][-1], 1.0)

    def test_malformed_token_vectors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal"):
            cita.token_weight_rows([1.0, 2.0], [[1.0], [2.0, 3.0]], [True, True])

    def test_semantic_mask_stops_at_serialized_termination(self) -> None:
        self.assertEqual(
            cita.semantic_response_mask([10, 11, 99, 12], {99}),
            [True, True, False, False],
        )

    def test_semantic_mask_requires_termination_and_two_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "termination"):
            cita.semantic_response_mask([10, 11], {99})
        with self.assertRaisesRegex(ValueError, "two semantic"):
            cita.semantic_response_mask([10, 99], {99})


class LossTests(unittest.TestCase):
    def test_uniform_weights_equal_response_cross_entropy(self) -> None:
        logits = torch.tensor(
            [[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]]
        )
        labels = torch.tensor([[-100, 1, 0, 1]])
        weights = torch.ones_like(labels, dtype=torch.float32)
        observed = cita.weighted_token_loss(logits, labels, weights)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1)
        )
        self.assertTrue(torch.allclose(observed, expected))

    def test_masked_tokens_do_not_change_loss(self) -> None:
        logits = torch.tensor([[[100.0, -100.0], [0.0, 1.0], [1.0, 0.0]]])
        labels = torch.tensor([[-100, -100, 0]])
        weights = torch.tensor([[99.0, 99.0, 1.0]])
        observed = cita.weighted_token_loss(logits, labels, weights)
        expected = torch.nn.functional.cross_entropy(logits[:, 1], torch.tensor([0]))
        self.assertTrue(torch.allclose(observed, expected))


class GradientTests(unittest.TestCase):
    def test_delta_norm_cosine_and_hash(self) -> None:
        uniform = {"b": torch.tensor([1.0, 0.0]), "a": torch.tensor([0.0])}
        cita_gradient = {"a": torch.tensor([0.0]), "b": torch.tensor([1.0, 1.0])}
        delta = cita.gradient_delta(cita_gradient, uniform)
        self.assertAlmostEqual(cita.gradient_norm(delta), 1.0)
        self.assertAlmostEqual(
            cita.gradient_cosine(cita_gradient, uniform), 2 ** -0.5
        )
        self.assertEqual(
            cita.gradient_hash(cita_gradient),
            cita.gradient_hash(dict(reversed(list(cita_gradient.items())))),
        )


class SelectionTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, str]]:
        return [
            {
                "sample_id": f"p{index}",
                "target_id": f"t{index}_{target}",
                "category": f"c{index % 4}",
            }
            for index in range(12)
            for target in range(2)
        ]

    def test_split_is_deterministic_prompt_disjoint_and_one_target_per_prompt(self) -> None:
        first = cita.select_prompt_disjoint_rows(self.rows(), 5, 5, 7)
        second = cita.select_prompt_disjoint_rows(list(reversed(self.rows())), 5, 5, 7)
        self.assertEqual(first, second)
        calibration = {row["sample_id"] for row in first["calibration"]}
        validation = {row["sample_id"] for row in first["validation"]}
        self.assertEqual(len(calibration), 5)
        self.assertEqual(len(validation), 5)
        self.assertFalse(calibration.intersection(validation))

    def test_insufficient_prompts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            cita.select_prompt_disjoint_rows(self.rows(), 7, 7, 7)


class StatisticsTests(unittest.TestCase):
    def test_rank_correlation_extremes(self) -> None:
        self.assertAlmostEqual(cita.pearson_ranks([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(cita.pearson_ranks([1, 2, 3], [6, 4, 2]), -1.0)

    def test_bootstrap_is_deterministic(self) -> None:
        first = cita.bootstrap_mean([1.0, 2.0, 3.0], 1000, 9)
        second = cita.bootstrap_mean([1.0, 2.0, 3.0], 1000, 9)
        self.assertEqual(first, second)


class RouteTests(unittest.TestCase):
    @staticmethod
    def gates() -> dict[str, bool]:
        return {
            "integrity": True,
            "gradient_valid": True,
            "state_restored": True,
            "token_signal": True,
            "counterfactual_specificity": True,
            "gradient_effect_size": True,
            "sham_distinct": True,
            "cross_split_replication": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            cita.select_route(self.gates()),
            "PASS_CITA_P0_AUTHORIZE_MATCHED_SMOKE",
        )

    def test_invalid_precedes_signal_failure(self) -> None:
        gates = self.gates()
        gates["integrity"] = False
        gates["token_signal"] = False
        self.assertEqual(cita.select_route(gates), "STOP_CITA_P0_INVALID")

    def test_weak_gradient_and_replication_routes_are_distinct(self) -> None:
        gates = self.gates()
        gates["gradient_effect_size"] = False
        self.assertEqual(cita.select_route(gates), "STOP_CITA_P0_GRADIENT_EFFECT_WEAK")
        gates = self.gates()
        gates["cross_split_replication"] = False
        self.assertEqual(cita.select_route(gates), "STOP_CITA_P0_ALIGNMENT_NOT_REPLICATED")


if __name__ == "__main__":
    unittest.main()
