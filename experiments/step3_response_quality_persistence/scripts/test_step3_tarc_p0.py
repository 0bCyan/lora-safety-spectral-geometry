from __future__ import annotations

import unittest

import torch

import step3_tarc_p0 as p0


class FakeTokenizer:
    pieces = {1: "I", 2: " can't", 3: " help", 4: " with"}

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        self.assert_skip = skip_special_tokens
        return "".join(self.pieces[value] for value in token_ids)


class RouteDepthTests(unittest.TestCase):
    def test_route_depth_uses_first_completed_marker(self) -> None:
        depth, marker, text = p0.route_depth(FakeTokenizer(), [1, 2, 3, 4], 4, 16)
        self.assertEqual(depth, 3)
        self.assertEqual(marker, "i can't help")
        self.assertEqual(text, "I can't help")


class SelectionTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, str]]:
        rows = []
        for category in ("a", "b"):
            for prompt in range(3):
                sample_id = f"{category}{prompt}"
                for target in range(2):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "target_id": f"{sample_id}t{target}",
                            "category": category,
                        }
                    )
        return rows

    def test_diverse_selection_is_deterministic_and_disjoint(self) -> None:
        rows = self.rows()
        for category in ("c", "d", "e", "f", "g", "h"):
            for prompt in range(2):
                sample_id = f"{category}{prompt}"
                for target in range(2):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "target_id": f"{sample_id}t{target}",
                            "category": category,
                        }
                    )
        first = p0.select_diverse_prompt_pairs(rows, 7)
        second = p0.select_diverse_prompt_pairs(rows, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["calibration"]), 8)
        self.assertEqual(len(first["validation"]), 8)
        self.assertFalse(set(first["calibration"]).intersection(first["validation"]))


class DirectionTests(unittest.TestCase):
    def test_tarc_alignment_advantage_is_explicit(self) -> None:
        calibration = {
            "uniform": {"p": torch.tensor([0.5, 0.5])},
            "ramp": {"p": torch.tensor([0.0, 1.0])},
            "tarc": {"p": torch.tensor([1.0, 0.0])},
        }
        validation = {"p": torch.tensor([1.0, 0.0])}
        result = p0.directional_metrics(calibration, validation, 0.25, 0.25)
        self.assertGreater(result["calibration_validation_tarc_cosine"], 0.99)
        self.assertGreater(result["tarc_over_ramp_validation_cosine_margin"], 0.99)
        self.assertGreater(
            result["tarc_tarc_validation_dot"], result["ramp_tarc_validation_dot"]
        )


if __name__ == "__main__":
    unittest.main()
