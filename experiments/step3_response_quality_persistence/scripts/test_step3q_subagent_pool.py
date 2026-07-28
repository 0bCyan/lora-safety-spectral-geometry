from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step3q_subagent_pool as pool


class Step3QSubagentPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = pool.load_registry()
        _, cls.p1, cls.requests = pool.source_bundle(cls.reg)
        cls.assignments = pool.assignment_rows(cls.reg, cls.requests)

    def response(self, index: int) -> str:
        return " ".join([f"safe-response-{index}"] + [f"detail-{index}-{position}" for position in range(1, 40)])

    def candidates(self) -> list[dict]:
        candidates = []
        for index, request in enumerate(self.requests):
            response = self.response(index)
            candidates.append(
                {
                    "opaque_request_id": request["opaque_request_id"],
                    "opaque_candidate_id": request["opaque_candidate_id"],
                    "opaque_prompt_id": request["opaque_prompt_id"],
                    "author_mode": request["author_mode"],
                    "prompt": request["prompt"],
                    "response": response,
                    "response_whitespace_tokens": len(response.split()),
                    "response_sha256": pool.sha256_bytes(response.encode()),
                    "normalized_response_sha256": pool.sha256_bytes(pool.normalize_text(response).encode()),
                    "authoring_source": "codex_multi_agent_v1",
                    "authoring_worker": f"worker-{index % 4}",
                    "authoring_agent_id": f"agent-{index % 4}",
                    "model_provenance": "inherited_parent_model_unversioned",
                }
            )
        return candidates

    def test_preflight_preserves_attempt_boundaries(self) -> None:
        result = pool.preflight(self.reg)
        self.assertEqual(result["fixed_requests"], 120)
        self.assertEqual(result["worker_rows"], {role: 30 for role in self.assignments})
        self.assertEqual(result["subagent_sessions_launched"], 0)
        self.assertFalse(result["external_api_called"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_assignments_are_mode_isolated_and_complete(self) -> None:
        request_ids = []
        for role, rows in self.assignments.items():
            expected = self.reg["authoring"]["worker_roles"][role]
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row["author_mode"] == expected["author_mode"] for row in rows))
            request_ids.extend(row["opaque_request_id"] for row in rows)
        self.assertEqual(len(request_ids), 120)
        self.assertEqual(len(set(request_ids)), 120)

    def test_worker_output_is_ordered_and_token_bounded(self) -> None:
        assignment = self.assignments["specific_1"]
        rows = [
            {"opaque_request_id": row["opaque_request_id"], "response": self.response(index)}
            for index, row in enumerate(assignment)
        ]
        self.assertEqual(len(pool.validate_worker_output(self.reg, assignment, rows)), 30)
        reversed_rows = list(reversed(rows))
        with self.assertRaises(pool.Step3QSubagentError):
            pool.validate_worker_output(self.reg, assignment, reversed_rows)
        rows[0]["response"] = "too short"
        with self.assertRaises(pool.Step3QSubagentError):
            pool.validate_worker_output(self.reg, assignment, rows)

    def test_global_deduplication_is_frozen(self) -> None:
        candidates = self.candidates()
        _, summary = pool.sealed_files(self.reg, self.p1, candidates)
        self.assertEqual(summary["route"], "PASS_SUBAGENT_AUTHORING")
        self.assertEqual(summary["valid_unique_candidates"], 120)
        self.assertEqual(summary["valid_prompts"], 60)
        self.assertFalse(summary["exact_provider_snapshot_available"])
        self.assertFalse(summary["external_api_called"])

    def test_import_builder_reuses_blind_packet_contract_without_api_claim(self) -> None:
        candidates = self.candidates()
        sealed, _ = pool.sealed_files(self.reg, self.p1, candidates)
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            (spool / "sealed").mkdir()
            for name, data in sealed.items():
                (spool / "sealed" / name).write_bytes(data)
            with mock.patch.object(
                pool,
                "collect_candidates",
                return_value=(spool, self.p1, candidates),
            ):
                files, summary = pool.build_import_files(self.reg, spool, "a" * 40)
        self.assertEqual(summary["primary_candidates"], 120)
        self.assertEqual(summary["primary_prompts"], 60)
        self.assertEqual(summary["review_rows_per_reviewer"], 156)
        self.assertFalse(summary["external_api_called"])
        self.assertIn("packets/reviewer_a_packet.jsonl", files)
        self.assertIn("packets/reviewer_b_packet.jsonl", files)
        manifest = json.loads(files["packet_manifest.json"])
        self.assertFalse(manifest["external_api_called"])
        self.assertFalse(manifest["training_authorized"])
        self.assertEqual(manifest["optimizer_steps"], 0)

    def test_external_spool_rejects_repository_path(self) -> None:
        with self.assertRaises(pool.Step3QSubagentError):
            pool.external_spool(pool.REPO_ROOT / "temporary-spool")

    def test_jsonl_rejects_crlf_and_blank_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_bytes(b'{"value":1}\r\n')
            with self.assertRaises(pool.Step3QSubagentError):
                pool.read_jsonl(path)
            path.write_bytes(b'{"value":1}\n\n')
            with self.assertRaises(pool.Step3QSubagentError):
                pool.read_jsonl(path)

    def test_registry_records_non_api_unversioned_provenance(self) -> None:
        self.assertEqual(self.reg["authoring"]["source"], "codex_multi_agent_v1")
        self.assertEqual(self.reg["authoring"]["model_provenance"], "inherited_parent_model_unversioned")
        self.assertFalse(self.reg["authoring"]["exact_provider_snapshot_available"])
        self.assertFalse(self.reg["external_api_called"])
        self.assertFalse(self.reg["training_authorized"])
        self.assertEqual(self.reg["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
