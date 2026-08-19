"""Synthetic and frozen-input tests for CDR-OPC M3 authoring transport."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m3_data as m3


def response(opening: str, count: int = 60) -> str:
    words = opening.split() + [f"word{index}" for index in range(count - len(opening.split()))]
    return " ".join(words)


class M3AuthoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = m3.load_registry()
        _, cls.requests = m3.source_bundle(cls.reg)
        cls.assignments = m3.assignment_rows(cls.reg, cls.requests)

    def test_frozen_denominators(self) -> None:
        self.assertEqual(len(self.requests), 120)
        self.assertEqual(len({row["opaque_prompt_id"] for row in self.requests}), 60)
        self.assertEqual(len(self.assignments), 12)
        self.assertEqual({len(rows) for rows in self.assignments.values()}, {10})

    def test_assignments_cover_requests_once(self) -> None:
        assigned = [row["opaque_request_id"] for rows in self.assignments.values() for row in rows]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {row["opaque_request_id"] for row in self.requests})

    def test_shards_do_not_mix_modes(self) -> None:
        for role, rows in self.assignments.items():
            expected = "specific_redirect" if role.startswith("specific_") else "risk_mitigation"
            self.assertEqual({row["author_mode"] for row in rows}, {expected})

    def test_valid_worker_output(self) -> None:
        assignment = self.assignments["specific_01"]
        rows = [
            {
                "opaque_request_id": row["opaque_request_id"],
                "response": response("I can't help", 60 + index),
            }
            for index, row in enumerate(assignment)
        ]
        observed = m3.validate_worker_output(self.reg, assignment, rows)
        self.assertEqual(len(observed), 10)

    def test_wrong_opening_is_rejected(self) -> None:
        assignment = self.assignments["mitigation_01"]
        rows = [
            {
                "opaque_request_id": row["opaque_request_id"],
                "response": response("I can't help", 60 + index),
            }
            for index, row in enumerate(assignment)
        ]
        with self.assertRaisesRegex(m3.Step3QSubagentError, "route-opening"):
            m3.validate_worker_output(self.reg, assignment, rows)

    def test_thinking_tag_is_rejected(self) -> None:
        assignment = self.assignments["specific_01"]
        rows = [
            {
                "opaque_request_id": row["opaque_request_id"],
                "response": response("I can't help", 60 + index),
            }
            for index, row in enumerate(assignment)
        ]
        rows[0]["response"] += " <think>hidden</think>"
        with self.assertRaisesRegex(m3.Step3QSubagentError, "thinking tags"):
            m3.validate_worker_output(self.reg, assignment, rows)


if __name__ == "__main__":
    unittest.main()
