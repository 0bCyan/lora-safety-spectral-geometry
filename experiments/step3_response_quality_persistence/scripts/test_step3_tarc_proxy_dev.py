from __future__ import annotations

import unittest

import step3_tarc_proxy_dev as proxy


class OverlapTests(unittest.TestCase):
    def test_overlap_checks_id_and_exact_prompt(self) -> None:
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
