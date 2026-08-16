from __future__ import annotations

import unittest

import torch

import step3_trim_cs0_common as trim


class TrimCs0Tests(unittest.TestCase):
    def test_transition_split_preserves_exact_text(self) -> None:
        response = "I can't assist with that. I can help with a safe alternative."
        prefix, suffix = trim.transition_split(response, "I can help")
        self.assertEqual(prefix + suffix, response)
        self.assertTrue(suffix.lstrip().startswith("I can help"))

    def test_pair_builder_is_deterministic_and_bidirectional(self) -> None:
        rows = []
        for index in range(3):
            rows.append(
                {
                    "sample_id": f"a{index}",
                    "target_id": f"ta{index}",
                    "category": "a",
                    "prompt": f"prompt a {index}",
                    "response": "I can't assist. I can help safely.",
                }
            )
            rows.append(
                {
                    "sample_id": f"b{index}",
                    "target_id": f"tb{index}",
                    "category": "b",
                    "prompt": f"prompt b {index}",
                    "response": "I can't assist. I can help safely.",
                }
            )
        first = trim.build_pair_rows(rows, {"a": 2, "b": 1}, "I can help")
        second = trim.build_pair_rows(list(reversed(rows)), {"a": 2, "b": 1}, "I can help")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({row["direction"] for row in first}, {"a_to_b", "b_to_a"})

    def test_patch_changes_only_selected_positions(self) -> None:
        hidden = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
        replacement = torch.full((2, 6), -1.0)
        patched, metadata = trim.patch_layer_output((hidden, "keep"), [1, 3], replacement)
        self.assertEqual(metadata, "keep")
        self.assertTrue(torch.equal(patched[0, [1, 3], :], replacement))
        self.assertTrue(torch.equal(patched[0, [0, 2, 4], :], hidden[0, [0, 2, 4], :]))

    def test_bootstrap_uses_pair_as_unit(self) -> None:
        result = trim.paired_bootstrap_mean_ci(
            {"p1": [1.0, 3.0], "p2": [5.0, 7.0]}, seed=7, samples=1000
        )
        self.assertEqual(result["pairs"], 2)
        self.assertAlmostEqual(result["mean"], 4.0)
        self.assertLessEqual(result["ci95_lower"], result["mean"])
        self.assertGreaterEqual(result["ci95_upper"], result["mean"])


if __name__ == "__main__":
    unittest.main()
