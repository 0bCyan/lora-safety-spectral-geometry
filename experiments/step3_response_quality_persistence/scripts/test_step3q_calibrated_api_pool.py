from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step3q_calibrated_api_pool as step3q


class FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def request_row(index: int = 0, mode: str = "specific_redirect") -> dict:
    return {
        "opaque_request_id": f"step3q_req_{index:024x}",
        "opaque_candidate_id": f"step3q_candidate_{index:024x}",
        "opaque_prompt_id": f"step3q_prompt_{index // 2:020x}",
        "author_mode": mode,
        "prompt": f"Synthetic risky prompt number {index} with enough identifying context.",
    }


def response_text(index: int) -> str:
    return " ".join([f"response{index}"] + [f"safe{position}" for position in range(1, 40)])


def api_response(request: dict, text: str, model: str = "gpt-5.6-sol") -> dict:
    output = json.dumps(
        {"opaque_request_id": request["opaque_request_id"], "response": text},
        separators=(",", ":"),
    )
    return {
        "id": f"resp_{request['opaque_request_id']}",
        "status": "completed",
        "error": None,
        "model": model,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": output}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 40, "total_tokens": 50},
    }


def receipt_for(
    request: dict,
    registry: dict,
    instructions: str,
    text: str,
    model: str = "gpt-5.6-sol",
) -> dict:
    candidate = step3q.validate_api_response(api_response(request, text, model), request, registry)
    _, request_sha, payload_sha = step3q.expected_receipt_hashes(
        request, registry, instructions
    )
    return {
        "schema": "step3q_api_receipt_v1",
        "route_id": registry["route_id"],
        "opaque_request_id": request["opaque_request_id"],
        "request_sha256": request_sha,
        "payload_sha256": payload_sha,
        "received_at_utc": "2026-07-27T00:00:00+00:00",
        "attempts": [
            {
                "attempt": 1,
                "http_status": 200,
                "elapsed_seconds": 0.1,
                "raw_response_sha256": "0" * 64,
                "api_response_id": candidate["api_response_id"],
                "effective_model": model,
            }
        ],
        "status": "valid",
        "candidate": candidate,
    }


class Step3QCalibratedApiPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = step3q.load_registry()

    def test_real_preflight_preserves_denominators_without_releasing_prompts(self) -> None:
        result = step3q.preflight(self.registry)
        self.assertEqual(result["source_rows"], 1000)
        self.assertEqual(result["eligible_fresh_prompts"], 780)
        self.assertEqual(result["slot_count"], 60)
        self.assertTrue(result["slot_matching_feasible"])
        self.assertGreaterEqual(result["minimum_slot_candidate_supply"], 1)
        self.assertFalse(result["selected_prompt_ids_released"])
        self.assertFalse(result["prompt_text_released"])
        calibration = result["calibration"]
        self.assertEqual(calibration["rows"], 96)
        self.assertEqual(
            calibration["role_counts"],
            {"component_anchor": 48, "prompt_mismatch": 24, "clean_safety_anchor": 24},
        )
        for counts in calibration["component_reference_counts"].values():
            self.assertGreaterEqual(counts["yes"], 12)
            self.assertGreaterEqual(counts["no"], 12)
        self.assertEqual(calibration["clean_safety_reference_counts"], {"yes": 12, "no": 12})
        self.assertTrue(calibration["mismatch_category_distinct"])
        for reviewer in ("reviewer_a", "reviewer_b"):
            sentinel = result["sentinel_allocations"][reviewer]
            self.assertEqual(sentinel["rows"], 24)
            self.assertTrue(sentinel["mismatch_pair_closure"])
            for counts in sentinel["component_reference_counts"].values():
                self.assertGreaterEqual(counts["yes"], 3)
                self.assertGreaterEqual(counts["no"], 3)
            self.assertEqual(sentinel["clean_safety_reference_counts"], {"yes": 4, "no": 4})

    def test_each_mismatch_sentinel_carries_its_paired_anchor(self) -> None:
        paths = step3q.validate_frozen_inputs(self.registry)
        namespace = step3q.selection_namespace(self.registry, "a" * 40)
        _, references, _ = step3q.build_calibration_bank(paths, self.registry, namespace)
        allocations = step3q.build_sentinel_allocations(references, self.registry, namespace)
        reference_by_id = {row["opaque_candidate_id"]: row for row in references}
        self.assertNotEqual(allocations["reviewer_a"], allocations["reviewer_b"])
        for reviewer, allocation in allocations.items():
            self.assertEqual(len(allocation), 24, reviewer)
            roles = Counter(reference_by_id[row_id]["calibration_role"] for row_id in allocation)
            self.assertEqual(roles, Counter(self.registry["calibration"]["sentinel_role_counts"]))
            component_rows = [
                reference_by_id[row_id]
                for row_id in allocation
                if reference_by_id[row_id]["calibration_role"] == "component_anchor"
            ]
            for field in step3q.COMPONENT_FIELDS:
                counts = Counter(row["expected"][field] for row in component_rows)
                self.assertGreaterEqual(counts["yes"], 3)
                self.assertGreaterEqual(counts["no"], 3)
            safety_rows = [
                reference_by_id[row_id]
                for row_id in allocation
                if reference_by_id[row_id]["calibration_role"] == "clean_safety_anchor"
            ]
            self.assertEqual(
                Counter(row["expected"]["clean_safe"] for row in safety_rows),
                Counter({"yes": 4, "no": 4}),
            )
            for row_id in allocation:
                row = reference_by_id[row_id]
                if row["calibration_role"] == "prompt_mismatch":
                    self.assertIn(row["matched_anchor_candidate_id"], allocation)

            selected_references = [reference_by_id[row_id] for row_id in allocation]
            reviews = []
            for reference in selected_references:
                review = {
                    "opaque_candidate_id": reference["opaque_candidate_id"],
                    "detail_retention": 2,
                    "safe_route": "yes",
                    "executable_step": "yes",
                    "prompt_binding": "yes",
                    "naturalness": "yes",
                    "clean_safe": "yes",
                }
                if reference["calibration_role"] == "component_anchor":
                    review.update(reference["expected"])
                elif reference["calibration_role"] == "prompt_mismatch":
                    review["detail_retention"] = 0
                    review["prompt_binding"] = "no"
                else:
                    review["clean_safe"] = reference["expected"]["clean_safe"]
                reviews.append(review)
            score = step3q.score_calibration_review(
                reviews, selected_references, self.registry
            )
            self.assertTrue(score["overall_pass"])
            for field in step3q.COMPONENT_FIELDS:
                self.assertEqual(
                    score["component_class_reference_accuracy"][field],
                    {"yes": 1.0, "no": 1.0},
                )

    def test_api_payload_is_stateless_strict_and_prompt_only(self) -> None:
        request = request_row()
        payload = step3q.build_api_payload(request, self.registry, "Frozen instructions")
        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["tools"], [])
        self.assertEqual(payload["truncation"], "disabled")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertFalse(payload["text"]["format"]["schema"]["additionalProperties"])
        user_input = json.loads(payload["input"])
        self.assertEqual(
            set(user_input), {"opaque_request_id", "author_mode", "prompt"}
        )
        self.assertNotIn("opaque_candidate_id", user_input)

    def test_api_response_accepts_frozen_alias_or_dated_snapshot_only(self) -> None:
        request = request_row()
        exact = step3q.validate_api_response(
            api_response(request, response_text(1)), request, self.registry
        )
        self.assertEqual(exact["effective_model"], "gpt-5.6-sol")
        dated = step3q.validate_api_response(
            api_response(request, response_text(2), "gpt-5.6-sol-2026-07-27"),
            request,
            self.registry,
        )
        self.assertEqual(dated["effective_model"], "gpt-5.6-sol-2026-07-27")
        with self.assertRaises(step3q.Step3QProvenanceError):
            step3q.validate_api_response(
                api_response(request, response_text(3), "gpt-5.6"),
                request,
                self.registry,
            )
        missing_usage = api_response(request, response_text(4))
        missing_usage["usage"] = None
        with self.assertRaises(step3q.Step3QProvenanceError):
            step3q.validate_api_response(missing_usage, request, self.registry)

    def test_existing_receipt_is_resumed_only_when_complete_and_consistent(self) -> None:
        request = request_row()
        instructions = "Frozen instructions"
        receipt = receipt_for(request, self.registry, instructions, response_text(4))
        with tempfile.TemporaryDirectory() as temporary:
            receipt_dir = Path(temporary)
            receipt_path = receipt_dir / f"{request['opaque_request_id']}.json"
            receipt_path.write_text(step3q.json_text(receipt), encoding="utf-8")
            with mock.patch.object(step3q.urllib.request, "urlopen") as urlopen:
                resumed = step3q._dispatch_one(
                    request, self.registry, instructions, "unused-key", receipt_dir
                )
            urlopen.assert_not_called()
            self.assertEqual(resumed, receipt)

            corrupted = copy.deepcopy(receipt)
            corrupted["candidate"]["response_sha256"] = "f" * 64
            receipt_path.write_text(step3q.json_text(corrupted), encoding="utf-8")
            with mock.patch.object(step3q.urllib.request, "urlopen") as urlopen:
                with self.assertRaises(step3q.Step3QValidationError):
                    step3q._dispatch_one(
                        request, self.registry, instructions, "unused-key", receipt_dir
                    )
            urlopen.assert_not_called()

    def test_malformed_http_200_is_one_terminal_semantic_failure(self) -> None:
        request = request_row()
        instructions = "Frozen instructions"
        with tempfile.TemporaryDirectory() as temporary:
            receipt_dir = Path(temporary)
            fake = FakeHTTPResponse(b"not-json")
            with mock.patch.object(step3q.urllib.request, "urlopen", return_value=fake) as urlopen:
                receipt = step3q._dispatch_one(
                    request, self.registry, instructions, "test-key", receipt_dir
                )
            self.assertEqual(receipt["status"], "semantic_failure")
            self.assertEqual(receipt["attempts"][0]["http_status"], 200)
            self.assertEqual(len(receipt["attempts"]), 1)
            urlopen.assert_called_once()

            with mock.patch.object(step3q.urllib.request, "urlopen") as resumed_urlopen:
                resumed = step3q._dispatch_one(
                    request, self.registry, instructions, "test-key", receipt_dir
                )
            resumed_urlopen.assert_not_called()
            self.assertEqual(resumed, receipt)

    def test_transport_failure_retries_then_freezes_valid_receipt(self) -> None:
        request = request_row()
        instructions = "Frozen instructions"
        body = step3q.canonical_json(api_response(request, response_text(5))).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                step3q.urllib.request,
                "urlopen",
                side_effect=[urllib.error.URLError("temporary"), FakeHTTPResponse(body)],
            ) as urlopen, mock.patch.object(step3q.time, "sleep") as sleep:
                receipt = step3q._dispatch_one(
                    request, self.registry, instructions, "test-key", Path(temporary)
                )
            self.assertEqual(receipt["status"], "valid")
            self.assertEqual(len(receipt["attempts"]), 2)
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(2.0)

    def test_dispatch_seal_deduplicates_and_cannot_be_overwritten(self) -> None:
        instructions = "Frozen instructions"
        requests = []
        receipts = []
        for index in range(120):
            mode = "specific_redirect" if index % 2 == 0 else "risk_mitigation"
            request = request_row(index, mode)
            requests.append(request)
            text = response_text(0 if index < 2 else index)
            receipts.append(receipt_for(request, self.registry, instructions, text))
        manifest = {"namespace_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_dir = root / "receipts"
            receipt_dir.mkdir()
            for receipt in receipts:
                path = receipt_dir / f"{receipt['opaque_request_id']}.json"
                path.write_text(step3q.json_text(receipt), encoding="utf-8")
            sealed = root / "sealed"
            result = step3q.seal_dispatch_receipts(
                self.registry,
                manifest,
                requests,
                receipts,
                receipt_dir,
                sealed,
                instructions,
            )
            self.assertEqual(result["route"], "PASS_API_AUTHORING")
            self.assertEqual(result["valid_before_deduplication"], 120)
            self.assertEqual(result["valid_unique_candidates"], 119)
            self.assertEqual(result["normalized_duplicate_failures"], 1)
            self.assertEqual(result["valid_prompts"], 60)
            self.assertEqual(result["effective_models"], ["gpt-5.6-sol"])
            self.assertTrue(result["model_provenance_pass"])
            self.assertEqual(len(step3q.read_jsonl(sealed / "candidates.jsonl")), 119)
            with self.assertRaises(step3q.Step3QValidationError):
                step3q.seal_dispatch_receipts(
                    self.registry,
                    manifest,
                    requests,
                    receipts,
                    receipt_dir,
                    sealed,
                    instructions,
                )

    def test_prepared_bundle_reconstructs_exactly_and_detects_tampering(self) -> None:
        protocol_commit = step3q._git_output("rev-parse", "HEAD")
        files, diagnostics = step3q.build_prepare_bundle(self.registry, protocol_commit)
        self.assertEqual(diagnostics["selected_prompts"], 60)
        self.assertEqual(diagnostics["api_requests"], 120)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            step3q.write_bundle(output, files)
            verified = step3q.verify_prepared_bundle(self.registry, output)
            self.assertEqual(verified["artifact_files"], 6)
            self.assertEqual(verified["calibration_rows"], 96)
            diagnostics_path = output / "diagnostics" / "step3q_prepare_diagnostics.json"
            diagnostics_path.write_bytes(diagnostics_path.read_bytes() + b" ")
            with self.assertRaises(step3q.Step3QValidationError):
                step3q.verify_prepared_bundle(self.registry, output)

    def test_dispatch_without_api_key_stops_before_network_or_disk_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}, clear=False
        ), mock.patch.object(step3q.urllib.request, "urlopen") as urlopen:
            with self.assertRaisesRegex(step3q.Step3QValidationError, "OPENAI_API_KEY"):
                step3q.dispatch_api(
                    self.registry, Path(temporary) / "missing", Path(temporary) / "spool"
                )
        urlopen.assert_not_called()

    def test_receipt_spool_must_remain_outside_repository(self) -> None:
        with self.assertRaises(step3q.Step3QValidationError):
            step3q.require_external_spool_root(step3q.REPO_ROOT / "api-spool")
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                step3q.require_external_spool_root(Path(temporary)),
                Path(temporary).resolve(),
            )


if __name__ == "__main__":
    unittest.main()
