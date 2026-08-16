from __future__ import annotations

import unittest

import torch

import step3_tarc_common as tarc


class RouteTests(unittest.TestCase):
    def test_registered_route_must_start_response(self) -> None:
        self.assertEqual(tarc.refusal_route_end("I can't help with that."), 12)
        with self.assertRaises(ValueError):
            tarc.refusal_route_end("Perhaps I can't help with that.")

    def test_trie_unions_children_at_shared_history(self) -> None:
        nodes = tarc.trie_nodes([[1, 2, 3], [1, 4]])
        by_history = {tuple(row["history"]): row for row in nodes}
        self.assertEqual(by_history[()]["allowed_token_ids"], [1])
        self.assertEqual(by_history[(1,)]["allowed_token_ids"], [2, 4])
        self.assertEqual(len(nodes), 3)


class MarginTests(unittest.TestCase):
    def test_positive_conservative_margin_certifies_all_allowed_tokens(self) -> None:
        logits = torch.tensor([0.0, 4.0, 4.5, 1.0])
        margin = tarc.conservative_allowed_margin(logits, [1, 2])
        self.assertGreater(float(margin), 0.0)
        self.assertGreater(float(logits[[1, 2]].min()), float(logits[[0, 3]].max()))

    def test_conservative_score_penalizes_multiple_branches(self) -> None:
        single = tarc.conservative_allowed_margin(
            torch.tensor([1.0, 2.0, -100.0]), [1]
        )
        branch = tarc.conservative_allowed_margin(
            torch.tensor([1.0, 2.0, 2.0]), [1, 2]
        )
        self.assertLess(float(branch), float(single))

    def test_barrier_decreases_with_margin(self) -> None:
        low = tarc.certificate_barrier(torch.tensor([-1.0, 0.0]))
        high = tarc.certificate_barrier(torch.tensor([1.0, 2.0]))
        self.assertLess(float(high), float(low))


class DecisionTests(unittest.TestCase):
    def test_direction_failure_is_not_a_pass(self) -> None:
        gates = {
            "integrity": True,
            "gradient_valid": True,
            "budget_matched": True,
            "vram": True,
            "state_restored": True,
            "route_structure": True,
            "objective_distinct": True,
            "cross_prompt_directional": False,
        }
        self.assertEqual(
            tarc.select_route(gates), "STOP_TARC_P0_DIRECTION_NOT_REPLICATED"
        )


if __name__ == "__main__":
    unittest.main()
