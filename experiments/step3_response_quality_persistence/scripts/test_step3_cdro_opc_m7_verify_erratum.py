from __future__ import annotations

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

import step3_cdro_opc_m7_mirror_curvature as m7
import step3_cdro_opc_m7_verify_erratum as erratum


class M7VerifierErratumUnitTests(unittest.TestCase):
    def test_sorted_json_round_trip_is_order_insensitive_schema(self) -> None:
        source = {field: index for index, field in enumerate(m7.TABLE_FIELDS)}
        reloaded = json.loads(json.dumps(source, sort_keys=True))
        self.assertNotEqual(tuple(reloaded), m7.TABLE_FIELDS)
        normalized = erratum.normalize_table_row(reloaded)
        self.assertEqual(tuple(normalized), m7.TABLE_FIELDS)
        self.assertEqual(normalized, reloaded)
        self.assertEqual(
            erratum.canonical_sha256(normalized),
            erratum.canonical_sha256(reloaded),
        )

    def test_missing_and_extra_members_remain_errors(self) -> None:
        source = {field: index for index, field in enumerate(m7.TABLE_FIELDS)}
        missing = dict(source)
        missing.pop(m7.TABLE_FIELDS[-1])
        extra = {**source, "unexpected": 1}
        with self.assertRaisesRegex(
            erratum.M7VerifierErratumError, "exact key set differs"
        ):
            erratum.normalize_table_row(missing)
        with self.assertRaisesRegex(
            erratum.M7VerifierErratumError, "exact key set differs"
        ):
            erratum.normalize_table_row(extra)

    def test_official_attempt_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "claim.json"
            erratum.write_json_exclusive(path, {"attempt": 1})
            with self.assertRaisesRegex(
                erratum.M7VerifierErratumError, "already claimed"
            ):
                erratum.write_json_exclusive(path, {"attempt": 2})
            self.assertEqual(erratum.read_json(path), {"attempt": 1})

    def test_output_registry_and_commit_provenance_are_fail_closed(self) -> None:
        protocol_path = erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        protocol = erratum.read_json(protocol_path)
        cases = []
        escaped = copy.deepcopy(protocol)
        escaped["outputs"]["receipt"] = (
            erratum.ORIGINAL_M7_ROOT + "/unregistered/run/receipt.json"
        )
        cases.append(escaped)
        wrong_commit = copy.deepcopy(protocol)
        wrong_commit["entry_commit"] = "0" * 40
        cases.append(wrong_commit)
        for case in cases:
            with self.subTest(case=case["entry_commit"]):
                with tempfile.TemporaryDirectory(dir=erratum.REPO) as root:
                    path = Path(root) / "protocol.json"
                    erratum.write_json(path, case)
                    with self.assertRaises(erratum.M7VerifierErratumError):
                        erratum.load_context(path)

    def test_stop_routes_are_stage_and_drift_specific(self) -> None:
        self.assertEqual(
            erratum.stop_route(
                "scope_and_target_preflight",
                erratum.M7VerifierErratumError("VE1 frozen input differs: x"),
            ),
            erratum.TARGET_DRIFT_ROUTE,
        )
        self.assertEqual(
            erratum.stop_route(
                "scope_and_target_preflight",
                erratum.M7VerifierErratumError("scope differs"),
            ),
            erratum.SCOPE_INTEGRITY_ROUTE,
        )
        self.assertEqual(
            erratum.stop_route(
                "full_verify", erratum.M7VerifierErratumError("value differs")
            ),
            erratum.FULL_VERIFY_ROUTE,
        )

    def test_source_bytes_are_bound_to_registered_git_blobs(self) -> None:
        protocol = erratum.load_context(
            erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        )
        for name in erratum.EXPECTED_SOURCE_KEYS:
            with self.subTest(name=name):
                self.assertTrue(
                    erratum.git_file_matches_commit(
                        erratum.SOURCE_RESULT_COMMIT,
                        erratum.EXPECTED_FROZEN_PATHS[name],
                    )
                )
        drifted = copy.deepcopy(protocol)
        drifted["frozen_inputs"]["m7_runner"]["path"] = drifted[
            "frozen_inputs"
        ]["m7_runner_tests"]["path"]
        with self.assertRaisesRegex(
            erratum.M7VerifierErratumError, "frozen input path differs"
        ):
            erratum.validate_source_git_bindings(drifted)

    def test_scope_stop_failure_replay_preserves_input_boundary(self) -> None:
        protocol = erratum.read_json(
            erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        )
        protocol["entry_commit"] = "0" * 40
        with tempfile.TemporaryDirectory(dir=erratum.REPO) as root:
            path = Path(root) / "protocol.json"
            erratum.write_json(path, protocol)
            try:
                erratum.load_context(path)
            except Exception as exc:
                failure = {
                    "stage": "scope_and_target_preflight",
                    "error_class": type(exc).__name__,
                    "error_message": str(exc),
                }
            else:
                self.fail("mutated protocol unexpectedly passed")
            recorded = {
                "failure": failure,
                "input_validation_complete": False,
            }
            claim = {"runtime_commit": erratum.ENTRY_COMMIT}
            observed_protocol, input_complete = (
                erratum.replay_recorded_stop_failure(path, claim, recorded)
            )
            self.assertIsNone(observed_protocol)
            self.assertFalse(input_complete)

    def test_normalization_does_not_mutate_frozen_diagnostics(self) -> None:
        protocol_path = erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        protocol = erratum.load_context(protocol_path)
        m7_protocol = erratum.read_json(
            erratum.repo_path(protocol["frozen_inputs"]["m7_protocol"]["path"])
        )
        diagnostics = erratum.read_json(
            erratum.repo_path(
                protocol["frozen_inputs"]["m7_diagnostics"]["path"]
            )
        )
        before = json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":"))
        normalized, audit = erratum.normalize_diagnostics(
            diagnostics, m7_protocol
        )
        self.assertEqual(
            json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":")),
            before,
        )
        self.assertEqual(normalized, diagnostics)
        self.assertNotEqual(
            tuple(diagnostics["attribution"]["table_rows"][0]), m7.TABLE_FIELDS
        )
        self.assertEqual(
            tuple(normalized["attribution"]["table_rows"][0]), m7.TABLE_FIELDS
        )
        self.assertTrue(audit["mapping_equal_after_normalization"])
        self.assertTrue(audit["canonical_sha256_equal_after_normalization"])

    def test_value_tamper_is_not_hidden_by_order_adapter(self) -> None:
        protocol_path = erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        protocol = erratum.load_context(protocol_path)
        m7_protocol = erratum.read_json(
            erratum.repo_path(protocol["frozen_inputs"]["m7_protocol"]["path"])
        )
        m6_diagnostics = erratum.read_json(
            erratum.repo_path(
                protocol["frozen_inputs"]["m6_diagnostics"]["path"]
            )
        )
        decision = erratum.read_json(
            erratum.repo_path(protocol["frozen_inputs"]["m7_decision"]["path"])
        )
        diagnostics = erratum.read_json(
            erratum.repo_path(
                protocol["frozen_inputs"]["m7_diagnostics"]["path"]
            )
        )
        normalized, _ = erratum.normalize_diagnostics(diagnostics, m7_protocol)
        normalized["attribution"]["table_rows"][0][
            "central_deterioration"
        ] += 1e-3
        with self.assertRaises(m7.M7MirrorCurvatureError):
            m7.verify_artifacts(
                m7_protocol, m6_diagnostics, decision, normalized
            )

    def test_csv_retains_exact_registered_header_order(self) -> None:
        protocol = erratum.load_context(
            erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        )
        csv_path = erratum.repo_path(
            protocol["frozen_inputs"]["m7_attribution_csv"]["path"]
        )
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(tuple(reader.fieldnames or ()), m7.TABLE_FIELDS)
        self.assertEqual(len(rows), 80)

    def test_frozen_original_error_and_corrected_full_audit(self) -> None:
        protocol_path = erratum.repo_path(erratum.DEFAULT_PROTOCOL)
        protocol = erratum.load_context(protocol_path)
        receipt = erratum.perform_read_only_audit(protocol_path, protocol)
        self.assertEqual(
            receipt["original_verify"]["status"],
            erratum.ORIGINAL_VERIFY_STATUS,
        )
        self.assertEqual(receipt["erratum_route"], erratum.ERRATUM_ROUTE)
        self.assertEqual(
            receipt["source_sha256_before"], receipt["source_sha256_after"]
        )
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertFalse(receipt["training_authorized"])


if __name__ == "__main__":
    unittest.main()
