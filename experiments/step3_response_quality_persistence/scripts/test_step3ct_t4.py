from __future__ import annotations

import unittest

import prepare_step3ct_t4_packets as packets
import step3ct_t4_common as common


class Step3CTT4Tests(unittest.TestCase):
    def test_thinking_cleanup_drops_content_and_tags(self) -> None:
        cleaned, detail = common.clean_thinking_output("<think>private reasoning</think>\nVisible answer")
        self.assertEqual(cleaned, "Visible answer")
        self.assertEqual(detail["complete_thinking_blocks_removed"], 1)
        self.assertFalse(detail["thinking_content_persisted"])
        unmatched, detail = common.clean_thinking_output("Answer first<think>unfinished")
        self.assertEqual(unmatched, "Answer first")
        self.assertTrue(detail["unmatched_open_thinking_removed_to_end"])

    def test_duplicate_count_registered_bounds(self) -> None:
        self.assertEqual(common.duplicate_count(600, 0.1, 8, 30), 30)
        self.assertEqual(common.duplicate_count(300, 0.1, 8, 30), 30)
        self.assertEqual(common.duplicate_count(16, 0.1, 8, 30), 8)

    def test_balanced_duplicate_selection(self) -> None:
        rows = []
        for arm in ["a", "b", "c"]:
            for index in range(100):
                rows.append({"arm": arm, "source_key": f"{arm}-{index}"})
        selected = common.balanced_duplicate_sources(rows, 30, 7, ["arm"])
        self.assertEqual(len(selected), 30)
        self.assertEqual({arm: sum(row["arm"] == arm for row in selected) for arm in "abc"}, {"a": 10, "b": 10, "c": 10})

    def test_packet_visibility_and_exact_duplicates(self) -> None:
        registry = {
            "packets": {
                "duplicate_fraction": 0.1,
                "duplicate_min": 2,
                "duplicate_max": 2,
                "packet_seed": 11,
                "duplicate_seed": 12,
                "utility": {"primary_rows": 6, "hidden_duplicates": 2},
            }
        }
        sources = []
        for arm in ["a", "b"]:
            for index in range(3):
                response = f"response-{arm}-{index}"
                sources.append({
                    "arm": arm,
                    "checkpoint_step": 64,
                    "sample_id": f"s{index}",
                    "generation_id": f"g-{arm}-{index}",
                    "source_key": f"{arm}|64|s{index}|utility",
                    "prompt": f"prompt-{index}",
                    "reference_response": f"reference-{index}",
                    "response": response,
                    "prompt_sha256": common.sha256_text(f"prompt-{index}"),
                    "response_sha256": common.sha256_text(response),
                })
        built = packets.build_packet("utility", sources, registry)
        self.assertEqual(len(built["packet_rows"]), 8)
        self.assertTrue(built["duplicate_exact_match"])
        self.assertTrue(all(set(row) == {"opaque_response_id", "prompt", "reference_response", "response"} for row in built["packet_rows"]))
        self.assertFalse(any("arm" in row or "checkpoint_step" in row for row in built["packet_rows"]))


if __name__ == "__main__":
    unittest.main()
