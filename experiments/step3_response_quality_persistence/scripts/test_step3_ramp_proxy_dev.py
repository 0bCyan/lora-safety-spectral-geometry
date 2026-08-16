from __future__ import annotations

import unittest

import generate_step3_ramp_proxy_dev as generation
import step3_ramp_proxy_dev as proxy


class GenerationRuleTests(unittest.TestCase):
    def test_thinking_content_is_removed(self) -> None:
        self.assertEqual(
            generation.clean_response("<think>hidden</think> I can't help with that."),
            "I can't help with that.",
        )

    def test_early_refusal_marker_is_literal_and_case_insensitive(self) -> None:
        self.assertTrue(
            generation.contains_any("I WON'T HELP with that.", generation.REFUSAL_MARKERS)
        )
        self.assertFalse(
            generation.contains_any("Here are the requested steps.", generation.REFUSAL_MARKERS)
        )


class SplitBoundaryTests(unittest.TestCase):
    def test_sample_and_prompt_overlap_are_both_detected(self) -> None:
        split = [
            {"sample_id": "a", "prompt": "one"},
            {"sample_id": "b", "prompt": "two"},
        ]
        targets = [
            {"sample_id": "a", "prompt": "different"},
            {"sample_id": "c", "prompt": "two"},
        ]
        result = proxy.prompt_overlap_counts(split, targets)
        self.assertEqual(result["sample_id_overlap"], 1)
        self.assertEqual(result["exact_prompt_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
