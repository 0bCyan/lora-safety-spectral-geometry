#!/usr/bin/env python3
"""Tests for the prospective Step3O-O4R sharded recovery protocol."""

from __future__ import annotations

import copy
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step3o_o4_qualification as qualification
import step3o_o4r_sharded_recovery as recovery


class RecoveryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if any(
            not recovery.author_shard_packet(lane, shard).exists()
            for lane in recovery.AUTHOR_LANES
            for shard in range(1, recovery.AUTHOR_SHARDS + 1)
        ):
            recovery.materialize_p0()

    def synthetic_author_rows(self, lane: str, shard: int) -> list[dict[str, str]]:
        packet = recovery.read_jsonl(recovery.author_shard_packet(lane, shard))
        rows = []
        for index, item in enumerate(packet):
            words = [
                "safe",
                lane,
                f"shard{shard}",
                f"row{index}",
                "response",
                "sets",
                "a",
                "clear",
                "boundary",
                "and",
                "offers",
                "a",
                "specific",
                "benign",
                "next",
                "action",
                "for",
                "the",
                "identified",
                "actor",
                "while",
                "preserving",
                "the",
                "prompt",
                "constraint",
                "relationship",
                "goal",
                "risk",
                "context",
                "reporting",
                "route",
                "authorized",
                "workflow",
                "support",
                "option",
                "prevention",
                "step",
                "relevant",
                "detail",
                "without",
                "unsafe",
                "procedural",
                "help",
            ]
            rows.append(
                {
                    "opaque_prompt_id": item["opaque_prompt_id"],
                    "response": " ".join(words),
                }
            )
        return rows

    def author_inputs(self) -> dict[str, tuple[bool, bytes]]:
        inputs: dict[str, tuple[bool, bytes]] = {}
        agent_index = 0
        for lane in recovery.AUTHOR_LANES:
            for shard in range(1, recovery.AUTHOR_SHARDS + 1):
                agent_index += 1
                key = f"author_{lane}_{shard:02d}"
                rows = self.synthetic_author_rows(lane, shard)
                inputs[key] = (True, recovery.jsonl_bytes(rows))
                provenance = qualification.build_provenance_record(
                    role=f"author_{lane}",
                    stage=f"authoring_shard_{shard:02d}",
                    agent_id=f"synthetic-o4r-author-{agent_index}",
                    allowed_input_hashes=recovery.expected_input_hashes(
                        recovery.author_shard_packet(lane, shard),
                        recovery.AUTHORING_RUBRIC,
                    ),
                    row_count=recovery.AUTHOR_ROWS_PER_SHARD,
                    attempt_id=recovery.ATTEMPT_ID,
                )
                inputs[f"{key}_provenance"] = (
                    True,
                    recovery.json_bytes(provenance),
                )
        return inputs

    def test_source_boundary_is_terminal_before_author_results(self) -> None:
        state = recovery.validate_source()
        self.assertEqual(state["terminal"]["route"], "STOP_AUTHORING")
        self.assertEqual(state["terminal"]["stage"], "P2_AUTHORING")
        self.assertFalse(state["terminal"]["training_authorized"])

    def test_author_shards_reconstruct_source_order(self) -> None:
        state = recovery.validate_source()
        for lane in recovery.AUTHOR_LANES:
            rows = []
            for shard in range(1, recovery.AUTHOR_SHARDS + 1):
                shard_rows = recovery.read_jsonl(
                    recovery.author_shard_packet(lane, shard)
                )
                self.assertEqual(len(shard_rows), 20)
                rows.extend(shard_rows)
            self.assertEqual(rows, state[f"packet_{lane}"])

    def test_missing_input_is_receipted_without_placeholder_file(self) -> None:
        value = recovery.receipt(
            "receipt", "P1_AUTHOR_INPUT", {"author_a_01": (False, b"")}
        )
        self.assertFalse(value["inputs"][0]["present"])
        self.assertEqual(value["inputs"][0]["bytes"], 0)
        self.assertFalse(value["replacement_or_repair_allowed"])

    def test_six_author_sessions_validate_and_reconstruct(self) -> None:
        p0 = {"state": recovery.validate_source()}
        outputs, agent_ids = recovery.build_author_stage(p0, self.author_inputs())
        self.assertEqual(len(agent_ids), 6)
        self.assertEqual(len(set(agent_ids)), 6)
        self.assertEqual(
            len(qualification.parse_jsonl(outputs[recovery.AUTHOR_A_COMBINED].decode("utf-8"), context="A")),
            60,
        )
        self.assertEqual(
            len(qualification.parse_jsonl(outputs[recovery.AUTHOR_B_COMBINED].decode("utf-8"), context="B")),
            60,
        )

    def synthetic_candidate_package(self) -> dict:
        p0 = {"state": recovery.validate_source()}
        outputs, _ = recovery.build_author_stage(p0, self.author_inputs())
        author_a = qualification.parse_jsonl(
            outputs[recovery.AUTHOR_A_COMBINED].decode("utf-8"), context="A"
        )
        author_b = qualification.parse_jsonl(
            outputs[recovery.AUTHOR_B_COMBINED].decode("utf-8"), context="B"
        )
        state = recovery.validate_source()
        source = state["registry"]["source"]
        return qualification.build_candidate_package(
            recovery.prompt_rows(),
            author_a,
            author_b,
            [row["opaque_prompt_id"] for row in state["packet_a"]],
            [row["opaque_prompt_id"] for row in state["packet_b"]],
            protocol_commit=source["original_protocol_commit"],
            beacon_round=source["beacon_round"],
            beacon_randomness=source["beacon_randomness"],
        )

    def test_review_shards_are_balanced_and_keep_repeat_pairs_together(self) -> None:
        package = self.synthetic_candidate_package()
        for lane in recovery.REVIEW_LANES:
            shards, links = recovery.build_review_shards(package, lane)
            self.assertEqual(len(shards), 6)
            self.assertTrue(all(len(shard) == 22 for shard in shards))
            self.assertTrue(all(len(value) == 2 for value in links))
            all_primary = set()
            primary_ids = {
                row["opaque_candidate_id"] for row in package["primary_candidates"]
            }
            for shard_rows, shard_links in zip(shards, links):
                ids = {row["opaque_candidate_id"] for row in shard_rows}
                self.assertEqual(len(ids & primary_ids), 20)
                for link in shard_links:
                    self.assertIn(link["primary_candidate_id"], ids)
                    self.assertIn(link["duplicate_candidate_id"], ids)
                all_primary.update(ids & primary_ids)
            self.assertEqual(all_primary, primary_ids)

    def test_review_sharding_is_deterministic_and_lane_specific(self) -> None:
        package = self.synthetic_candidate_package()
        first_a = recovery.build_review_shards(package, "a")
        second_a = recovery.build_review_shards(package, "a")
        first_b = recovery.build_review_shards(package, "b")
        self.assertEqual(first_a, second_a)
        self.assertNotEqual(first_a[0], first_b[0])

    def test_twenty_two_row_review_shard_uses_original_field_contract(self) -> None:
        package = self.synthetic_candidate_package()
        shards, _ = recovery.build_review_shards(package, "a")
        rows = [
            {
                "opaque_candidate_id": item["opaque_candidate_id"],
                "detail_retention": 2,
                "safe_route": "yes",
                "executable_step": "yes",
                "prompt_binding": "yes",
                "naturalness": "yes",
                "clean_safe": "yes",
                "short_reason": "The response is specific, safe, prompt-bound, and usable.",
            }
            for item in shards[0]
        ]
        validated = recovery.validate_reviewer_shard_bytes(
            recovery.jsonl_bytes(rows),
            [row["opaque_candidate_id"] for row in shards[0]],
            role="A",
        )
        self.assertEqual(len(validated), 22)

    def test_registry_freezes_no_retry_and_training_boundary(self) -> None:
        value = recovery.registry()
        self.assertFalse(value["authoring"]["replacement_or_retry_allowed"])
        self.assertFalse(value["review"]["replacement_or_retry_allowed"])
        self.assertEqual(value["review"]["c_max_rows"], 18)
        self.assertFalse(value["threat_model"]["malicious_operator_resistant"])
        self.assertFalse(
            value["threat_model"]["prior_output_semantic_reuse_cryptographically_excluded"]
        )
        self.assertTrue(
            value["threat_model"]["postcommit_artifact_substitution_detected"]
        )
        self.assertEqual(tuple(value["stage_tags"]), recovery.STAGES)
        self.assertEqual(len(set(value["stage_tags"].values())), 9)


class ProtocolHardeningTests(unittest.TestCase):
    def test_stage_publication_rolls_back_all_files_on_mid_write_failure(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            original_link = recovery.os.link
            calls = 0

            def fail_second(source: str | Path, target: str | Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic publication failure")
                original_link(source, target)

            with patch.object(recovery.os, "link", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "synthetic publication failure"):
                    recovery.write_stage(
                        {first: b"first\n", second: b"second\n"},
                        (first, second),
                    )
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_stage_preparation_failure_removes_temporary_file(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            root = Path(directory)
            target = root / "target.json"
            with patch.object(
                recovery.os, "fsync", side_effect=OSError("synthetic fsync failure")
            ):
                with self.assertRaisesRegex(OSError, "synthetic fsync failure"):
                    recovery.write_stage({target: b"value\n"}, (target,))
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_concurrent_single_writers_have_exactly_one_winner(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            root = Path(directory)
            target = root / "terminal.json"
            barrier = threading.Barrier(2)
            original_link = recovery.os.link

            def synchronized_link(source: str | Path, destination: str | Path) -> None:
                barrier.wait(timeout=5)
                original_link(source, destination)

            def writer(data: bytes) -> str:
                try:
                    recovery.write_once(target, data)
                except recovery.RecoveryValidationError:
                    return "rejected"
                return "published"

            with patch.object(
                recovery.os, "link", side_effect=synchronized_link
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    outcomes = list(executor.map(writer, (b"first\n", b"second\n")))
            self.assertEqual(sorted(outcomes), ["published", "rejected"])
            self.assertIn(target.read_bytes(), {b"first\n", b"second\n"})
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_concurrent_stage_writers_cannot_mix_or_overwrite_evidence(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            barrier = threading.Barrier(2)
            original_link = recovery.os.link

            def synchronized_first_link(
                source: str | Path, destination: str | Path
            ) -> None:
                if Path(destination) == first:
                    barrier.wait(timeout=5)
                original_link(source, destination)

            variants = (
                {first: b"a1\n", second: b"a2\n"},
                {first: b"b1\n", second: b"b2\n"},
            )

            def writer(contents: dict[Path, bytes]) -> str:
                try:
                    recovery.write_stage(contents, (first, second))
                except recovery.RecoveryValidationError:
                    return "rejected"
                return "published"

            with patch.object(
                recovery.os, "link", side_effect=synchronized_first_link
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    outcomes = list(executor.map(writer, variants))
            self.assertEqual(sorted(outcomes), ["published", "rejected"])
            self.assertIn(
                (first.read_bytes(), second.read_bytes()),
                {(b"a1\n", b"a2\n"), (b"b1\n", b"b2\n")},
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_single_terminal_style_write_is_atomic_on_failure(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            root = Path(directory)
            target = root / "terminal.json"
            with patch.object(
                recovery.os, "fsync", side_effect=OSError("synthetic stop failure")
            ):
                with self.assertRaisesRegex(OSError, "synthetic stop failure"):
                    recovery.write_once(target, b'{"terminal":true}\n')
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_c_gate_precondition_failure_does_not_create_terminal_stop(self) -> None:
        with (
            patch.object(
                recovery,
                "verify_ab_local",
                return_value={"terminal": False, "commit": "2" * 40},
            ),
            patch.object(
                recovery,
                "require_head",
                side_effect=recovery.RecoveryValidationError("wrong HEAD"),
            ),
            patch.object(recovery, "record_stop") as stop,
        ):
            with self.assertRaisesRegex(recovery.RecoveryValidationError, "wrong HEAD"):
                recovery.freeze_c_gate()
        stop.assert_not_called()

    def test_publication_failure_is_not_recorded_as_scientific_stop(self) -> None:
        with (
            patch.object(
                recovery,
                "verify_author_input_local",
                return_value={"commit": "2" * 40, "inputs": {}},
            ),
            patch.object(recovery, "require_head"),
            patch.object(recovery, "require_clean"),
            patch.object(recovery, "verify_p0_local", return_value={}),
            patch.object(
                recovery,
                "build_author_outcome_contents",
                return_value=({}, {"status": "PASS_AUTHORING_FROZEN"}),
            ),
            patch.object(
                recovery,
                "write_stage",
                side_effect=OSError("synthetic disk error"),
            ),
            patch.object(recovery, "record_stop") as stop,
        ):
            with self.assertRaisesRegex(OSError, "synthetic disk error"):
                recovery.freeze_authors()
        stop.assert_not_called()

    def test_stage_verifier_rejects_marker_only_commit_shape(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            root = Path(directory)
            marker = root / "manifest.json"
            missing = root / "required.jsonl"
            marker.write_bytes(b"{}\n")
            with self.assertRaisesRegex(recovery.RecoveryValidationError, "absent"):
                recovery.verify_stage_commit(
                    (missing, marker), marker, "1" * 40
                )

    def test_receipt_rejects_missing_or_tampered_frozen_input(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            target = Path(directory) / "author.jsonl"
            data = b'{"row":1}\n'
            targets = {"author_a_01": target}
            value = recovery.receipt(
                "synthetic", "P1_AUTHOR_INPUT", {"author_a_01": (True, data)}
            )
            with self.assertRaisesRegex(recovery.RecoveryValidationError, "absent"):
                recovery.validate_receipt(
                    value,
                    name="synthetic",
                    stage="P1_AUTHOR_INPUT",
                    targets=targets,
                )
            target.write_bytes(data)
            tampered = copy.deepcopy(value)
            tampered["inputs"][0]["sha256"] = "f" * 64
            with self.assertRaisesRegex(recovery.RecoveryValidationError, "hash"):
                recovery.validate_receipt(
                    tampered,
                    name="synthetic",
                    stage="P1_AUTHOR_INPUT",
                    targets=targets,
                )

    def _review_fixture(
        self, root: Path, *, reused_agent_id: str | None = None
    ) -> tuple[dict, dict[str, tuple[bool, bytes]], dict[tuple[str, int], Path], dict[tuple[str, int], Path]]:
        base = RecoveryProtocolTests()
        package = base.synthetic_candidate_package()
        inputs: dict[str, tuple[bool, bytes]] = {}
        packet_paths: dict[tuple[str, int], Path] = {}
        duplicate_paths: dict[tuple[str, int], Path] = {}
        agent_index = 0
        for lane in recovery.REVIEW_LANES:
            shards, links = recovery.build_review_shards(package, lane)
            for shard, (packet_rows, duplicate_links) in enumerate(zip(shards, links), 1):
                agent_index += 1
                packet = root / f"reviewer_{lane}_{shard:02d}_packet.jsonl"
                duplicate_map = root / f"reviewer_{lane}_{shard:02d}_duplicates.json"
                packet.write_bytes(recovery.jsonl_bytes(packet_rows))
                duplicate_map.write_bytes(
                    recovery.json_bytes({"links": duplicate_links})
                )
                packet_paths[(lane, shard)] = packet
                duplicate_paths[(lane, shard)] = duplicate_map
                rows = [
                    {
                        "opaque_candidate_id": row["opaque_candidate_id"],
                        "detail_retention": 2,
                        "safe_route": "yes",
                        "executable_step": "yes",
                        "prompt_binding": "yes",
                        "naturalness": "yes",
                        "clean_safe": "yes",
                        "short_reason": "Specific, safe, prompt-bound, and usable.",
                    }
                    for row in packet_rows
                ]
                key = f"reviewer_{lane}_{shard:02d}"
                inputs[key] = (True, recovery.jsonl_bytes(rows))
                agent_id = (
                    reused_agent_id
                    if agent_index == 1 and reused_agent_id is not None
                    else f"synthetic-o4r-review-{agent_index}"
                )
                provenance = qualification.build_provenance_record(
                    role=f"reviewer_{lane}",
                    stage=f"qualification_shard_{shard:02d}",
                    agent_id=agent_id,
                    allowed_input_hashes=recovery.expected_input_hashes(
                        packet, recovery.QUALIFICATION_RUBRIC
                    ),
                    row_count=22,
                    attempt_id=recovery.ATTEMPT_ID,
                )
                inputs[f"{key}_provenance"] = (
                    True,
                    recovery.json_bytes(provenance),
                )
        return package, inputs, packet_paths, duplicate_paths

    def test_ab_validation_rejects_author_reused_as_reviewer(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            package, inputs, packets, duplicates = self._review_fixture(
                Path(directory), reused_agent_id="author-session-1"
            )
            with (
                patch.object(recovery, "load_candidate_package", return_value=package),
                patch.object(
                    recovery,
                    "review_packet",
                    side_effect=lambda lane, shard: packets[(lane, shard)],
                ),
                patch.object(
                    recovery,
                    "review_duplicate_map",
                    side_effect=lambda lane, shard: duplicates[(lane, shard)],
                ),
            ):
                with self.assertRaisesRegex(
                    recovery.RecoveryValidationError, "reused as reviewer"
                ):
                    recovery.build_ab_stage(
                        inputs, author_agent_ids=("author-session-1",)
                    )

    def test_fixed_panel_diagnostics_cover_all_session_pairs(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            package, inputs, packets, _duplicates = self._review_fixture(Path(directory))
            rows: dict[str, list[dict]] = {"a": [], "b": []}
            for lane in recovery.REVIEW_LANES:
                packet_key = (
                    "reviewer_a_packet" if lane == "a" else "reviewer_b_packet"
                )
                by_id: dict[str, dict] = {}
                for shard in range(1, recovery.REVIEW_SHARDS + 1):
                    key = f"reviewer_{lane}_{shard:02d}"
                    for row in qualification.parse_jsonl(
                        inputs[key][1].decode("utf-8"), context=key
                    ):
                        by_id[row["opaque_candidate_id"]] = row
                rows[lane] = [
                    by_id[item["opaque_candidate_id"]]
                    for item in package[packet_key]
                ]
            with patch.object(
                recovery,
                "review_packet",
                side_effect=lambda lane, shard: packets[(lane, shard)],
            ):
                result = recovery.fixed_panel_diagnostics(
                    package,
                    rows["a"],
                    rows["b"],
                    {"a": [2] * 6, "b": [2] * 6},
                )
            self.assertEqual(len(result["by_reviewer_a_shard"]), 6)
            self.assertEqual(len(result["by_reviewer_b_shard"]), 6)
            self.assertEqual(
                sum(row["rows"] for row in result["nonempty_session_pair_overlaps"]),
                120,
            )
            self.assertTrue(result["per_shard_repeat_pass"])

    def _terminal_fixture(self, root: Path) -> tuple[Path, Path, dict, str, str]:
        receipt = root / "receipt.json"
        stop = root / "stop.json"
        receipt.write_bytes(b"receipt\n")
        parent = "2" * 40
        commit = "3" * 40
        value = recovery.stop_record(
            stage="P1_AUTHOR_OUTCOME",
            route="STOP_AUTHORING",
            parent=parent,
            error=ValueError("synthetic"),
            evidence=(receipt,),
        )
        stop.write_bytes(recovery.json_bytes(value))
        spec = {
            "routes": {"STOP_AUTHORING"},
            "evidence": {recovery.repo_path(receipt): receipt},
            "required_by_route": {
                "STOP_AUTHORING": {recovery.repo_path(receipt)}
            },
        }
        return receipt, stop, spec, parent, commit

    def test_terminal_validator_checks_full_boundary_and_evidence(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            receipt, stop, spec, parent, commit = self._terminal_fixture(Path(directory))

            def creation(path: Path) -> str:
                return commit if path == stop else parent

            with (
                patch.object(recovery, "TERMINAL_STOP", stop),
                patch.object(
                    recovery,
                    "STOP_STAGE_SPECS",
                    {"P1_AUTHOR_OUTCOME": spec},
                ),
                patch.object(recovery, "expected_stop_parent", return_value=parent),
                patch.object(recovery, "creation_commit", side_effect=creation),
                patch.object(recovery, "commit_parent", return_value=parent),
                patch.object(
                    recovery, "verify_stage_commit", return_value=commit
                ) as verifier,
                patch.object(recovery, "replay_terminal_stop"),
            ):
                result = recovery.verify_terminal_stop("P1_AUTHOR_OUTCOME")
            self.assertTrue(result["checks_pass"])
            verifier.assert_called_once_with((stop,), stop, parent)

            tampered = recovery.parse_json_bytes(stop.read_bytes(), "stop")
            tampered["training_authorized"] = True
            stop.write_bytes(recovery.json_bytes(tampered))
            with (
                patch.object(recovery, "TERMINAL_STOP", stop),
                patch.object(
                    recovery,
                    "STOP_STAGE_SPECS",
                    {"P1_AUTHOR_OUTCOME": spec},
                ),
            ):
                with self.assertRaisesRegex(
                    recovery.RecoveryValidationError, "boundary"
                ):
                    recovery.verify_terminal_stop("P1_AUTHOR_OUTCOME")

    def test_terminal_replay_rejects_fake_stop_and_error_mismatch(self) -> None:
        parent = "2" * 40
        value = recovery.stop_record(
            stage="P1_AUTHOR_OUTCOME",
            route="STOP_AUTHORING",
            parent=parent,
            error=recovery.RecoveryValidationError("registered failure"),
            evidence=(),
        )
        with (
            patch.object(recovery, "verify_p0_local", return_value={}),
            patch.object(recovery, "verify_author_input_local", return_value={}),
            patch.object(
                recovery,
                "build_author_outcome_contents",
                return_value=({}, {}),
            ),
        ):
            with self.assertRaisesRegex(
                recovery.RecoveryValidationError, "replay passed"
            ):
                recovery.replay_terminal_stop(value)

        with (
            patch.object(recovery, "verify_p0_local", return_value={}),
            patch.object(recovery, "verify_author_input_local", return_value={}),
            patch.object(
                recovery,
                "build_author_outcome_contents",
                side_effect=recovery.RecoveryValidationError("different failure"),
            ),
        ):
            with self.assertRaisesRegex(
                recovery.RecoveryValidationError, "does not match"
            ):
                recovery.replay_terminal_stop(value)

        with (
            patch.object(recovery, "verify_p0_local", return_value={}),
            patch.object(recovery, "verify_author_input_local", return_value={}),
            patch.object(
                recovery,
                "build_author_outcome_contents",
                side_effect=recovery.RecoveryValidationError("registered failure"),
            ),
        ):
            recovery.replay_terminal_stop(value)

    def test_historical_lock_accepts_ancestor_but_rejects_diverged_commit(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch.object(recovery, "canonical_remote_tip", return_value="9" * 40),
            patch.object(recovery, "run_git", return_value=completed) as git,
        ):
            recovery.require_remote_ancestor("1" * 40)
        git.assert_called_once_with(
            "merge-base", "--is-ancestor", "1" * 40, "9" * 40, check=False
        )

        diverged = subprocess.CompletedProcess([], 1, "", "")
        with (
            patch.object(recovery, "canonical_remote_tip", return_value="9" * 40),
            patch.object(recovery, "run_git", return_value=diverged),
        ):
            with self.assertRaisesRegex(
                recovery.RecoveryValidationError, "not on the canonical branch"
            ):
                recovery.require_remote_ancestor("1" * 40)

    def test_lock_creation_cannot_skip_predecessor(self) -> None:
        states = {
            "P3_AB_INPUT": {"commit": "2" * 40},
            "P3_AB_OUTCOME": {"commit": "3" * 40},
        }
        with (
            patch.object(recovery, "local_stage_state", side_effect=lambda stage: states[stage]),
            patch.object(recovery, "require_head"),
            patch.object(recovery, "require_clean"),
            patch.object(recovery, "commit_parent", return_value="2" * 40),
            patch.object(
                recovery,
                "require_stage_lock",
                side_effect=recovery.RecoveryValidationError("missing predecessor lock"),
            ),
            patch.object(recovery, "require_remote_tip") as remote_tip,
        ):
            with self.assertRaisesRegex(
                recovery.RecoveryValidationError, "missing predecessor lock"
            ):
                recovery.create_stage_lock("P3_AB_OUTCOME")
        remote_tip.assert_not_called()

    def test_reconstructed_final_route_rejects_fake_manifest_bytes(self) -> None:
        with TemporaryDirectory(dir=recovery.REPO_ROOT) as directory:
            path = Path(directory) / "route.json"
            expected = recovery.json_bytes(
                {"route": "PASS_MACHINE_RUBRIC_POOL", "training_authorized": False}
            )
            path.write_bytes(
                recovery.json_bytes(
                    {"route": "STOP_TARGET_YIELD", "training_authorized": False}
                )
            )
            with self.assertRaisesRegex(
                recovery.RecoveryValidationError, "reconstruction mismatch"
            ):
                recovery.compare_contents({path: expected})


if __name__ == "__main__":
    unittest.main()
