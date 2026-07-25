#!/usr/bin/env python3
"""Tests for the prospective Step3O-O4 append-only stage runner."""

from __future__ import annotations

import copy
from contextlib import ExitStack
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import step3o_o4_stage as stage


PROTOCOL_COMMIT = "1" * 40


def reliability(*, passed: bool) -> dict:
    return {
        "primary_rows": 120,
        "detail_retention": {},
        "binary_components": {},
        "full_vector_exact_agreement": 1.0 if passed else 0.5,
        "candidate_qualified": {
            "raw_agreement": 1.0 if passed else 0.5,
            "gwet_ac1": 1.0 if passed else 0.4,
        },
        "candidate_qualified_by_reviewer": {"A": {}, "B": {}},
        "stratified_prompt_reweighting": {
            "replicates": 10000,
            "finite_replicates": 10000,
            "q025": 1.0 if passed else 0.3,
            "failure": None,
        },
        "reviewer_a_duplicate_consistency": {},
        "reviewer_b_duplicate_consistency": {},
        "gates": {"synthetic_gate": passed},
        "overall_pass": passed,
    }


class StageSchemaTests(unittest.TestCase):
    def test_registry_freezes_stage_commands_and_key_artifact_paths(self) -> None:
        execution = stage.pool.registry()["execution"]
        self.assertEqual(
            execution["commands"],
            [
                "freeze-authors",
                "verify-authors",
                "freeze-candidates",
                "verify-candidates",
                "freeze-ab-reviews",
                "verify-ab-reviews",
                "freeze-c-gate",
                "verify-c-gate",
                "freeze-final",
                "verify-final",
                "verify-terminal-stop",
            ],
        )
        expected = {
            "author_receipt": stage.AUTHOR_RECEIPT,
            "author_a_result": stage.AUTHOR_A_RESULT,
            "author_b_result": stage.AUTHOR_B_RESULT,
            "author_a_provenance": stage.AUTHOR_A_PROVENANCE,
            "author_b_provenance": stage.AUTHOR_B_PROVENANCE,
            "author_manifest": stage.AUTHOR_MANIFEST,
            "author_report": stage.AUTHOR_REPORT,
            "candidate_key": stage.CANDIDATE_KEY,
            "reviewer_a_packet": stage.REVIEWER_A_PACKET,
            "reviewer_b_packet": stage.REVIEWER_B_PACKET,
            "reviewer_a_template": stage.REVIEWER_A_TEMPLATE,
            "reviewer_b_template": stage.REVIEWER_B_TEMPLATE,
            "reviewer_a_duplicates": stage.REVIEWER_A_DUPLICATES,
            "reviewer_b_duplicates": stage.REVIEWER_B_DUPLICATES,
            "candidate_manifest": stage.CANDIDATE_MANIFEST,
            "candidate_report": stage.CANDIDATE_REPORT,
            "ab_receipt": stage.AB_RECEIPT,
            "reviewer_a_result": stage.REVIEWER_A_RESULT,
            "reviewer_b_result": stage.REVIEWER_B_RESULT,
            "reviewer_a_provenance": stage.REVIEWER_A_PROVENANCE,
            "reviewer_b_provenance": stage.REVIEWER_B_PROVENANCE,
            "ab_manifest": stage.AB_MANIFEST,
            "ab_reliability": stage.AB_RELIABILITY,
            "reviewer_c_packet": stage.REVIEWER_C_PACKET,
            "reviewer_c_template": stage.REVIEWER_C_TEMPLATE,
            "c_gate_manifest": stage.C_GATE_MANIFEST,
            "c_gate_report": stage.C_GATE_REPORT,
            "c_receipt": stage.C_RECEIPT,
            "reviewer_c_result": stage.REVIEWER_C_RESULT,
            "reviewer_c_provenance": stage.REVIEWER_C_PROVENANCE,
            "final_labels": stage.FINAL_LABELS,
            "qualified_pool": stage.QUALIFIED_POOL,
            "qualified_pool_key": stage.QUALIFIED_POOL_KEY,
            "final_agreement": stage.FINAL_AGREEMENT,
            "final_yield": stage.FINAL_YIELD,
            "route_decision": stage.ROUTE_DECISION,
            "final_manifest": stage.FINAL_MANIFEST,
            "final_report": stage.FINAL_REPORT,
            "terminal_stop": stage.ATTEMPT_STOP,
        }
        self.assertEqual(set(execution["artifacts"]), set(expected))
        for key, path in expected.items():
            self.assertEqual(execution["artifacts"][key], stage.pool.repo_path(path))

    def test_registered_normal_stage_paths_are_unique(self) -> None:
        paths = [
            *stage.AUTHOR_STAGE_PATHS,
            *stage.CANDIDATE_STAGE_PATHS,
            *stage.AB_STAGE_PATHS,
            *stage.C_GATE_PASS_PATHS,
            *stage.FINAL_STAGE_PATHS,
        ]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            path.resolve().relative_to(stage.BASE.resolve())

    def test_receipt_is_role_ordered_and_hash_bound(self) -> None:
        value = stage.receipt(
            "synthetic_receipt",
            "SYNTHETIC",
            {"role_b": b"b\n", "role_a": b"a\n"},
        )
        self.assertEqual(
            [row["role"] for row in value["inputs"]], ["role_b", "role_a"]
        )
        self.assertEqual(value["inputs"][0]["sha256"], stage.sha256_bytes(b"b\n"))
        self.assertFalse(value["replacement_or_repair_allowed"])
        self.assertFalse(value["training_authorized"])

    def test_review_template_contains_no_label_or_identity(self) -> None:
        template = stage.review_template(
            [
                {
                    "opaque_candidate_id": "candidate_1",
                    "prompt": "prompt",
                    "response": "response",
                }
            ]
        )
        self.assertEqual(set(template[0]), set(stage.qualification.REVIEW_FIELDS))
        self.assertNotIn("author_role", template[0])
        self.assertNotIn("assigned_category", template[0])
        self.assertIsNone(template[0]["detail_retention"])

    def test_stop_and_route_artifacts_never_authorize_training(self) -> None:
        stop = stage.stop_artifact(
            stage="P3_AB_REVIEW",
            route="STOP_REVIEW_INVALID",
            parent=PROTOCOL_COMMIT,
            error=ValueError("synthetic"),
            evidence_artifacts=[],
        )
        self.assertFalse(stop["replacement_or_retry_allowed"])
        self.assertFalse(stop["training_authorized"])
        decision = stage.route_decision(
            route="PASS_MACHINE_RUBRIC_POOL",
            stage="P3_FINAL",
            protocol_commit=PROTOCOL_COMMIT,
            parent_commit="2" * 40,
            reliability_pass=True,
            target_yield_pass=True,
        )
        self.assertFalse(decision["training_authorized"])
        self.assertFalse(decision["training_suitability_validated"])

    def test_input_must_be_outside_frozen_result_tree(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "incoming.jsonl"
            path.write_bytes(b"{}\n")
            self.assertEqual(stage.read_exact_input(path), b"{}\n")


class CommitBoundaryTests(unittest.TestCase):
    def test_stage_commit_requires_exact_changed_path_set(self) -> None:
        paths = stage.AUTHOR_STAGE_PATHS
        with patch.object(
            stage.pool, "verify_exact_creation_commit", return_value="3" * 40
        ) as verifier:
            self.assertEqual(
                stage.verify_stage_commit(
                    paths, stage.AUTHOR_MANIFEST, "2" * 40
                ),
                "3" * 40,
            )
            verifier.assert_called_once_with(
                paths,
                stage.AUTHOR_MANIFEST,
                "2" * 40,
                reg=stage.pool.registry(),
                require_pushed=True,
            )

        with patch.object(
            stage.pool,
            "verify_exact_creation_commit",
            side_effect=ValueError("unexpected path set"),
        ):
            with self.assertRaises(stage.StageValidationError):
                stage.verify_stage_commit(paths, stage.AUTHOR_MANIFEST, "2" * 40)


class TerminalStopVerificationTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[dict, Path, Path, Path, str]:
        receipt = root / "receipt.json"
        evidence = root / "partial.jsonl"
        stop = root / "stop.json"
        receipt.write_bytes(b"receipt\n")
        evidence.write_bytes(b"partial\n")
        parent = "2" * 40
        spec = {
            "route": "STOP_REVIEW_INVALID",
            "allowed_paths": (receipt, evidence),
            "required_receipt": receipt,
        }
        with patch.object(stage, "STOP_STAGE_SPECS", {"SYNTHETIC": spec}):
            value = stage.stop_artifact(
                stage="SYNTHETIC",
                route="STOP_REVIEW_INVALID",
                parent=parent,
                error=ValueError("synthetic"),
                evidence_artifacts=(
                    stage.artifact(receipt, receipt.read_bytes()),
                    stage.artifact(evidence, evidence.read_bytes()),
                ),
            )
        stop.write_bytes(stage.json_bytes(value))
        return spec, receipt, evidence, stop, parent

    def test_terminal_stop_accepts_only_receipted_hash_bound_exact_paths(self) -> None:
        temp_root = stage.pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            spec, receipt, evidence, stop, parent = self.build_fixture(root)
            with (
                patch.object(stage, "ATTEMPT_STOP", stop),
                patch.object(stage, "STOP_STAGE_SPECS", {"SYNTHETIC": spec}),
                patch.object(stage, "expected_stop_parent", return_value=parent),
                patch.object(
                    stage, "verify_stage_commit", return_value="3" * 40
                ) as verifier,
            ):
                result = stage.verify_terminal_stop()
            self.assertTrue(result["checks_pass"])
            verifier.assert_called_once_with(
                (receipt, evidence, stop), stop, parent
            )

    def test_terminal_stop_rejects_missing_receipt(self) -> None:
        temp_root = stage.pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            spec, _receipt, _evidence, stop, parent = self.build_fixture(root)
            value = stage.parse_json_bytes(stop.read_bytes(), "synthetic stop")
            value["evidence_artifacts"] = value["evidence_artifacts"][1:]
            stop.write_bytes(stage.json_bytes(value))
            with (
                patch.object(stage, "ATTEMPT_STOP", stop),
                patch.object(stage, "STOP_STAGE_SPECS", {"SYNTHETIC": spec}),
                patch.object(stage, "expected_stop_parent", return_value=parent),
            ):
                with self.assertRaisesRegex(
                    stage.StageValidationError, "missing its required input receipt"
                ):
                    stage.verify_terminal_stop()

    def test_terminal_stop_rejects_tampered_hash_and_unregistered_path(self) -> None:
        temp_root = stage.pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            spec, _receipt, _evidence, stop, parent = self.build_fixture(root)
            original = stage.parse_json_bytes(stop.read_bytes(), "synthetic stop")
            tampered = copy.deepcopy(original)
            tampered["evidence_artifacts"][0]["sha256"] = "f" * 64
            stop.write_bytes(stage.json_bytes(tampered))
            with (
                patch.object(stage, "ATTEMPT_STOP", stop),
                patch.object(stage, "STOP_STAGE_SPECS", {"SYNTHETIC": spec}),
                patch.object(stage, "expected_stop_parent", return_value=parent),
            ):
                with self.assertRaisesRegex(
                    stage.StageValidationError, "hash or length mismatch"
                ):
                    stage.verify_terminal_stop()

            extra = root / "extra.json"
            extra.write_bytes(b"extra\n")
            with patch.object(stage.pool, "REPO_ROOT", root):
                unregistered = copy.deepcopy(original)
                unregistered["evidence_artifacts"].append(
                    stage.artifact(extra, extra.read_bytes())
                )
            stop.write_bytes(stage.json_bytes(unregistered))
            with (
                patch.object(stage, "ATTEMPT_STOP", stop),
                patch.object(stage, "STOP_STAGE_SPECS", {"SYNTHETIC": spec}),
                patch.object(stage, "expected_stop_parent", return_value=parent),
            ):
                with self.assertRaisesRegex(
                    stage.StageValidationError, "duplicate or unregistered"
                ):
                    stage.verify_terminal_stop()


class ReliabilityRouteTests(unittest.TestCase):
    def state(self) -> dict:
        return {
            "protocol_commit": PROTOCOL_COMMIT,
            "beacon_round": 1,
            "beacon_randomness": "2" * 64,
        }

    def test_reliability_failure_stops_before_c_packet(self) -> None:
        with (
            patch.object(stage, "load_candidate_package", return_value={}),
            patch.object(stage, "reviewer_rows", return_value=[]),
            patch.object(
                stage.qualification,
                "compute_ab_reliability",
                return_value=reliability(passed=False),
            ),
            patch.object(stage.pool, "sha256", return_value="3" * 64),
        ):
            contents, manifest = stage.build_c_gate_contents(
                self.state(), "4" * 40
            )
        self.assertEqual(manifest["status"], "STOP_REVIEW_UNRELIABLE")
        self.assertEqual(set(contents), set(stage.C_GATE_STOP_PATHS))
        self.assertNotIn(stage.REVIEWER_C_PACKET, contents)
        self.assertFalse(manifest["reviewer_c_authorized"])

    def test_reliability_pass_builds_blinded_c_packet_without_route(self) -> None:
        c_row = {
            "opaque_candidate_id": "candidate_1",
            "prompt": "prompt",
            "response": "response",
        }
        with (
            patch.object(stage, "load_candidate_package", return_value={}),
            patch.object(stage, "reviewer_rows", return_value=[]),
            patch.object(
                stage.qualification,
                "compute_ab_reliability",
                return_value=reliability(passed=True),
            ),
            patch.object(stage.qualification, "build_c_packet", return_value=[c_row]),
            patch.object(stage.pool, "sha256", return_value="3" * 64),
        ):
            contents, manifest = stage.build_c_gate_contents(
                self.state(), "4" * 40
            )
        self.assertEqual(
            manifest["status"], "PASS_AB_RELIABILITY_C_PACKET_FROZEN"
        )
        self.assertEqual(set(contents), set(stage.C_GATE_PASS_PATHS))
        self.assertNotIn(stage.ROUTE_DECISION, contents)
        self.assertFalse(manifest["ab_labels_or_reasons_visible_to_c"])
        self.assertFalse(manifest["disagreement_field_markers_visible_to_c"])


class SyntheticPipelineTests(unittest.TestCase):
    def test_complete_pure_pipeline_reaches_nontraining_pass(self) -> None:
        repo_temp = stage.pool.REPO_ROOT / "temp"
        repo_temp.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=repo_temp) as directory, ExitStack() as stack:
            root = Path(directory)
            pool_paths = {
                "AUTHOR_A_PACKET": root / "p1_author_a_packet.jsonl",
                "AUTHOR_B_PACKET": root / "p1_author_b_packet.jsonl",
                "AUTHORING_KEY": root / "p1_authoring_key.csv",
                "AUTHORING_INSTRUCTIONS": root / "authoring_instructions.md",
                "QUALIFICATION_RUBRIC": root / "qualification_rubric.md",
                "PROTOCOL_MANIFEST": root / "p1_manifest.json",
            }
            stage_paths = {
                name: root / f"{name.casefold()}.artifact"
                for name in (
                    "AUTHOR_RECEIPT",
                    "AUTHOR_A_RESULT",
                    "AUTHOR_B_RESULT",
                    "AUTHOR_A_PROVENANCE",
                    "AUTHOR_B_PROVENANCE",
                    "AUTHOR_MANIFEST",
                    "AUTHOR_REPORT",
                    "CANDIDATE_KEY",
                    "REVIEWER_A_PACKET",
                    "REVIEWER_B_PACKET",
                    "REVIEWER_A_TEMPLATE",
                    "REVIEWER_B_TEMPLATE",
                    "REVIEWER_A_DUPLICATES",
                    "REVIEWER_B_DUPLICATES",
                    "CANDIDATE_MANIFEST",
                    "CANDIDATE_REPORT",
                    "AB_RECEIPT",
                    "REVIEWER_A_RESULT",
                    "REVIEWER_B_RESULT",
                    "REVIEWER_A_PROVENANCE",
                    "REVIEWER_B_PROVENANCE",
                    "AB_MANIFEST",
                    "AB_RELIABILITY",
                    "REVIEWER_C_PACKET",
                    "REVIEWER_C_TEMPLATE",
                    "C_GATE_MANIFEST",
                    "C_GATE_REPORT",
                    "C_RECEIPT",
                    "REVIEWER_C_RESULT",
                    "REVIEWER_C_PROVENANCE",
                    "FINAL_LABELS",
                    "QUALIFIED_POOL",
                    "QUALIFIED_POOL_KEY",
                    "FINAL_AGREEMENT",
                    "FINAL_YIELD",
                    "ROUTE_DECISION",
                    "FINAL_MANIFEST",
                    "FINAL_REPORT",
                    "ATTEMPT_STOP",
                )
            }
            for name, path in pool_paths.items():
                stack.enter_context(patch.object(stage.pool, name, path))
            for name, path in stage_paths.items():
                stack.enter_context(patch.object(stage, name, path))

            prompts: list[dict] = []
            slots = stage.pool.registry()["selection"]["category_severity_slots"]
            for category in sorted(slots):
                for severity in (1, 2, 3):
                    for _ in range(int(slots[category][str(severity)])):
                        index = len(prompts)
                        prompts.append(
                            {
                                "opaque_prompt_id": f"prompt_{index:03d}",
                                "prompt": f"Synthetic safety prompt {index}",
                                "assigned_category": category,
                                "severity_level": severity,
                                "sample_id": f"synthetic_{index:03d}",
                                "source_index": index + 1,
                            }
                        )
            self.assertEqual(len(prompts), 60)
            order_a = list(prompts)
            order_b = list(reversed(prompts))
            pool_paths["AUTHOR_A_PACKET"].write_bytes(
                stage.jsonl_bytes(
                    [
                        {"opaque_prompt_id": row["opaque_prompt_id"], "prompt": row["prompt"]}
                        for row in order_a
                    ]
                )
            )
            pool_paths["AUTHOR_B_PACKET"].write_bytes(
                stage.jsonl_bytes(
                    [
                        {"opaque_prompt_id": row["opaque_prompt_id"], "prompt": row["prompt"]}
                        for row in order_b
                    ]
                )
            )
            pool_paths["AUTHORING_KEY"].write_bytes(
                stage.csv_bytes(
                    prompts,
                    [
                        "opaque_prompt_id",
                        "sample_id",
                        "source_index",
                        "assigned_category",
                        "severity_level",
                    ],
                )
            )
            pool_paths["AUTHORING_INSTRUCTIONS"].write_text(
                "synthetic authoring rubric\n", encoding="utf-8", newline="\n"
            )
            pool_paths["QUALIFICATION_RUBRIC"].write_text(
                "synthetic qualification rubric\n", encoding="utf-8", newline="\n"
            )
            pool_paths["PROTOCOL_MANIFEST"].write_bytes(b"{}\n")

            state = {
                "protocol_commit": "1" * 40,
                "beacon_round": 999,
                "beacon_randomness": "2" * 64,
                "p1_commit": "3" * 40,
            }

            def author_output(role: str, rows: list[dict]) -> bytes:
                output = []
                for row in rows:
                    index = int(row["opaque_prompt_id"].rsplit("_", 1)[1])
                    words = [f"author_{role}_{index:03d}"]
                    words.extend(f"safe_{role}_{index:03d}_{word:03d}" for word in range(39))
                    output.append(
                        {
                            "opaque_prompt_id": row["opaque_prompt_id"],
                            "response": " ".join(words),
                        }
                    )
                return stage.jsonl_bytes(output)

            author_inputs = {
                "author_a": author_output("A", order_a),
                "author_b": author_output("B", order_b),
            }
            for role, packet in (
                ("author_a", pool_paths["AUTHOR_A_PACKET"]),
                ("author_b", pool_paths["AUTHOR_B_PACKET"]),
            ):
                provenance = stage.qualification.build_provenance_record(
                    role=role,
                    stage="authoring",
                    agent_id=f"synthetic-{role}",
                    allowed_input_hashes=stage.expected_input_hashes(
                        packet, pool_paths["AUTHORING_INSTRUCTIONS"]
                    ),
                    row_count=60,
                )
                author_inputs[f"{role}_provenance"] = stage.json_bytes(provenance)

            author_contents = stage.build_author_contents(state, author_inputs)
            for path, data in author_contents.items():
                path.write_bytes(data)
            candidate_contents, package = stage.build_candidate_contents(
                state, "4" * 40
            )
            for path, data in candidate_contents.items():
                path.write_bytes(data)

            primary = {
                row["opaque_candidate_id"]: row
                for row in package["primary_candidates"]
            }
            disputed_id = next(
                candidate_id
                for candidate_id, row in primary.items()
                if row["author_role"] == "A"
            )

            def reviews(role: str) -> bytes:
                packet = package[
                    "reviewer_a_packet" if role == "A" else "reviewer_b_packet"
                ]
                links = package[
                    "reviewer_a_duplicates" if role == "A" else "reviewer_b_duplicates"
                ]
                duplicate_map = {
                    row["duplicate_candidate_id"]: row["primary_candidate_id"]
                    for row in links
                }
                output = []
                for visible in packet:
                    visible_id = visible["opaque_candidate_id"]
                    source_id = duplicate_map.get(visible_id, visible_id)
                    values = {
                        "detail_retention": 2,
                        "safe_route": "yes",
                        "executable_step": "yes",
                        "prompt_binding": "yes",
                        "naturalness": "yes",
                        "clean_safe": "yes",
                    }
                    if primary[source_id]["author_role"] == "B":
                        prompt_index = int(
                            primary[source_id]["opaque_prompt_id"].rsplit("_", 1)[1]
                        )
                        failed_field = stage.qualification.DECISION_FIELDS[
                            prompt_index % len(stage.qualification.DECISION_FIELDS)
                        ]
                        values[failed_field] = (
                            1 if failed_field == "detail_retention" else "no"
                        )
                    if role == "B" and source_id == disputed_id:
                        values["safe_route"] = "no"
                    output.append(
                        {
                            "opaque_candidate_id": visible_id,
                            **values,
                            "short_reason": "Synthetic visible-text reason.",
                        }
                    )
                return stage.jsonl_bytes(output)

            ab_inputs = {
                "reviewer_a": reviews("A"),
                "reviewer_b": reviews("B"),
            }
            for role, packet in (
                ("reviewer_a", stage_paths["REVIEWER_A_PACKET"]),
                ("reviewer_b", stage_paths["REVIEWER_B_PACKET"]),
            ):
                provenance = stage.qualification.build_provenance_record(
                    role=role,
                    stage="qualification",
                    agent_id=f"synthetic-{role}",
                    allowed_input_hashes=stage.expected_input_hashes(
                        packet, pool_paths["QUALIFICATION_RUBRIC"]
                    ),
                    row_count=132,
                )
                ab_inputs[f"{role}_provenance"] = stage.json_bytes(provenance)

            ab_contents = stage.build_ab_contents(state, "5" * 40, ab_inputs)
            for path, data in ab_contents.items():
                path.write_bytes(data)
            c_gate_contents, c_manifest = stage.build_c_gate_contents(
                state, "6" * 40
            )
            self.assertEqual(
                c_manifest["status"], "PASS_AB_RELIABILITY_C_PACKET_FROZEN"
            )
            self.assertEqual(c_manifest["reviewer_c_rows"], 1)
            for path, data in c_gate_contents.items():
                path.write_bytes(data)

            c_packet = stage.c_packet_rows()
            c_output = stage.jsonl_bytes(
                [
                    {
                        "opaque_candidate_id": c_packet[0]["opaque_candidate_id"],
                        "detail_retention": 2,
                        "safe_route": "yes",
                        "executable_step": "yes",
                        "prompt_binding": "yes",
                        "naturalness": "yes",
                        "clean_safe": "yes",
                        "short_reason": "Synthetic independent C reason.",
                    }
                ]
            )
            provenance_c = stage.qualification.build_provenance_record(
                role="reviewer_c",
                stage="qualification",
                agent_id="synthetic-reviewer_c",
                allowed_input_hashes=stage.expected_input_hashes(
                    stage_paths["REVIEWER_C_PACKET"],
                    pool_paths["QUALIFICATION_RUBRIC"],
                ),
                row_count=1,
            )
            final_contents = stage.build_final_contents(
                state,
                "7" * 40,
                {
                    "reviewer_c": c_output,
                    "reviewer_c_provenance": stage.json_bytes(provenance_c),
                },
            )
            route = stage.parse_json_bytes(
                final_contents[stage_paths["ROUTE_DECISION"]], "synthetic route"
            )
            final_manifest = stage.parse_json_bytes(
                final_contents[stage_paths["FINAL_MANIFEST"]],
                "synthetic final manifest",
            )
            self.assertEqual(route["route"], "PASS_MACHINE_RUBRIC_POOL")
            self.assertFalse(route["training_authorized"])
            self.assertEqual(final_manifest["qualified_prompt_rows"], 60)


if __name__ == "__main__":
    unittest.main()
