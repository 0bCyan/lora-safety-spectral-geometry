#!/usr/bin/env python3
"""Tests for the prospective Step3O-O4 prompt-pool freeze."""

from __future__ import annotations

import copy
import json
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import step3o_o4_training_pool as o4


class Step3OO4TrainingPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = o4.registry()
        # This is a non-materializing test preview. P1 uses only the later P0 SHA.
        cls.preview_commit = o4.git_head()
        cls.preview_randomness = {
            "beacon_round": 1,
            "beacon_randomness": "0" * 64,
        }
        cls.package = o4.build_prompt_package(
            cls.preview_commit, cls.preview_randomness
        )

    def test_registry_slots_are_exact(self) -> None:
        slots = o4.expand_slots(self.reg)
        self.assertEqual(len(slots), 60)
        severity = {str(level): 0 for level in (1, 2, 3)}
        for slot in slots:
            severity[str(slot["severity"])] += 1
        self.assertEqual(severity, {"1": 8, "2": 44, "3": 8})

    def test_author_packets_are_blinded_and_independently_ordered(self) -> None:
        packet_a = self.package["author_a_packet"]
        packet_b = self.package["author_b_packet"]
        ids_a = [row["opaque_prompt_id"] for row in packet_a]
        ids_b = [row["opaque_prompt_id"] for row in packet_b]
        self.assertEqual(len(packet_a), 60)
        self.assertEqual(set(ids_a), set(ids_b))
        self.assertEqual(len(set(ids_a)), 60)
        self.assertNotEqual(ids_a, ids_b)
        self.assertTrue(
            all(set(row) == {"opaque_prompt_id", "prompt"} for row in packet_a + packet_b)
        )

    def test_selection_counts_and_gates(self) -> None:
        audit = self.package["selection_audit"]
        self.assertEqual(audit["selected_prompts"], 60)
        self.assertEqual(audit["severity_counts"], {"1": 8, "2": 44, "3": 8})
        self.assertTrue(all(audit["gates"].values()))
        self.assertFalse(audit["source_harmful_responses_released"])
        self.assertFalse(audit["training_authorized"])

    def test_templates_follow_their_own_packet_orders(self) -> None:
        for role in ("a", "b"):
            packet_ids = [
                row["opaque_prompt_id"] for row in self.package[f"author_{role}_packet"]
            ]
            template = self.package[f"author_{role}_template"]
            self.assertEqual(packet_ids, [row["opaque_prompt_id"] for row in template])
            self.assertTrue(all(row["response"] == "" for row in template))

    def test_no_source_response_is_present_in_released_package(self) -> None:
        released = {
            "author_a_packet": self.package["author_a_packet"],
            "author_b_packet": self.package["author_b_packet"],
            "key": self.package["key"],
            "source_audit": self.package["source_audit"],
            "selection_audit": self.package["selection_audit"],
        }
        serialized = json.dumps(released, ensure_ascii=False)
        self.assertNotIn('"response":', serialized)
        self.assertNotIn('"reference_response":', serialized)

    def test_tracked_structured_prompt_scan_runs_at_commit(self) -> None:
        scan = self.package["source_audit"]["tracked_prompt_fingerprint_scan"]
        self.assertGreater(scan["structured_files_scanned"], 0)
        self.assertEqual(scan["source_ids"], sorted(set(scan["source_ids"])))
        self.assertTrue(
            set(scan["source_ids"])
            <= set(self.package["source_audit"]["excluded_source_ids"])
        )

    def test_short_prompt_containment_is_detected(self) -> None:
        short = o4.token_shingles("alpha beta gamma delta", 5)
        containing = o4.token_shingles("alpha beta gamma delta epsilon", 5)
        _, overlap, maximum = o4.lexical_similarity(short, containing)
        self.assertEqual(overlap, 1.0)
        self.assertGreaterEqual(maximum, 0.9)

    def test_character_sequence_check_catches_one_word_change(self) -> None:
        left = "Explain the safest reporting process for this specific workplace incident"
        right = "Explain the safest reporting process for that specific workplace incident"
        self.assertGreaterEqual(o4.sequence_similarity(left, right), 0.85)

    def test_nfkc_casefold_normalization(self) -> None:
        self.assertEqual(o4.normalize_text("Ａlpha  STRASSE"), "alpha strasse")

    def test_normalized_exact_overlap_is_excluded(self) -> None:
        rows = [{"sample_id": "synthetic", "prompt": "Ａlpha   Beta"}]
        references = [
            {"reference_id": "reference", "prompt": "alpha beta", "kind": "test"}
        ]
        kept, excluded = o4.lexical_filter(rows, references, self.reg)
        self.assertEqual(kept, [])
        self.assertEqual(excluded[0]["reason"], "exact_reference_overlap")

    def test_frozen_writes_refuse_even_byte_identical_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            o4.write_frozen(path, "{}\n")
            self.assertEqual(path.read_bytes(), b"{}\n")
            with self.assertRaises(ValueError):
                o4.write_frozen(path, "{}\n")

    def test_frozen_write_has_one_exclusive_concurrent_winner(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"

            def attempt(value: str) -> bool:
                try:
                    o4.write_frozen(path, value)
                    return True
                except ValueError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(attempt, ("a\n", "b\n")))
            self.assertEqual(sum(results), 1)
            self.assertIn(path.read_text(encoding="utf-8"), {"a\n", "b\n"})

    def test_partial_attempt_file_blocks_retry(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(b"{")
            with self.assertRaises(ValueError):
                o4.write_frozen(path, "{}\n")

    def test_crlf_worktree_is_not_equal_to_lf_git_blob(self) -> None:
        with self.assertRaises(ValueError):
            o4.require_exact_blob_bytes(b"a\r\nb\r\n", b"a\nb\n", "synthetic")
        o4.require_exact_blob_bytes(b"a\nb\n", b"a\nb\n", "synthetic")

    def test_matching_is_stable_under_input_shuffle(self) -> None:
        rows = [
            {
                "sample_id": f"synthetic_{index}",
                "true_categories": ["A", "B"] if index < 2 else ["B"],
                "severity_level": 2,
            }
            for index in range(3)
        ]
        slots = [
            {"slot_id": "A|2|1", "category": "A", "severity": 2},
            {"slot_id": "B|2|1", "category": "B", "severity": 2},
            {"slot_id": "B|2|2", "category": "B", "severity": 2},
        ]
        first = o4.deterministic_match(rows, slots, "synthetic-salt")
        second = o4.deterministic_match(list(reversed(rows)), list(reversed(slots)), "synthetic-salt")
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 3)

    def test_matching_fails_when_hall_condition_is_violated(self) -> None:
        rows = [
            {"sample_id": "only", "true_categories": ["A", "B"], "severity_level": 2}
        ]
        slots = [
            {"slot_id": "A|2|1", "category": "A", "severity": 2},
            {"slot_id": "B|2|1", "category": "B", "severity": 2},
        ]
        with self.assertRaises(ValueError):
            o4.deterministic_match(rows, slots, "synthetic-salt")

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            o4.parse_json_text('{"prompt":"a","prompt":"b"}', "synthetic")

    def test_nonstandard_json_constants_are_rejected(self) -> None:
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(ValueError):
                o4.parse_json_text('{"value":' + value + "}", "synthetic")

    def test_source_schema_rejects_bad_severity_types_and_extra_fields(self) -> None:
        categories = {
            category: index == 0
            for index, category in enumerate(
                sorted(self.reg["selection"]["category_severity_slots"])
            )
        }
        row = {
            "sample_id": "risk_pkuqa_" + "999999",
            "family": self.reg["source"]["expected_family"],
            "source": self.reg["source"]["expected_source"],
            "prompt": "A synthetic prompt with enough content for strict schema validation.",
            "response": "A non-empty synthetic source response.",
            "is_safe_source": False,
            "severity_level": 2,
            "harm_category": categories,
            "source_index": 999999,
            "content_hash": "",
        }
        row["content_hash"] = o4.sha256_text(row["prompt"] + "\n" + row["response"])
        o4.validate_source_row(row, self.reg["source"], set(categories))
        for invalid in (True, "2", 2.0):
            bad = copy.deepcopy(row)
            bad["severity_level"] = invalid
            with self.assertRaises(ValueError):
                o4.validate_source_row(bad, self.reg["source"], set(categories))
        bad = copy.deepcopy(row)
        bad["unexpected"] = "field"
        with self.assertRaises(ValueError):
            o4.validate_source_row(bad, self.reg["source"], set(categories))

    def test_prompt_only_structured_exposure_is_extracted(self) -> None:
        data = b'{"record":{"prompt":"Prompt text without a source ID"}}\n'
        values = list(o4.structured_prompt_values("synthetic.jsonl", data, {"prompt"}))
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][1], "Prompt text without a source ID")

    def test_casefolded_and_unicode_escaped_prompt_keys_are_extracted(self) -> None:
        for data in (
            b'{"Prompt":"casefolded key"}\n',
            b'{"pr\\u006fmpt":"escaped key"}\n',
        ):
            values = list(
                o4.structured_prompt_values("synthetic.jsonl", data, {"prompt"})
            )
            self.assertEqual(len(values), 1)

    def test_legacy_nan_does_not_bypass_tracked_prompt_scan(self) -> None:
        data = b'{"diagnostic":NaN,"Prompt":"still extracted"}\n'
        values = list(
            o4.structured_prompt_values("synthetic.jsonl", data, {"prompt"})
        )
        self.assertEqual(values[0][1], "still extracted")

    def test_p0_sources_contain_no_unregistered_production_id_fixture(self) -> None:
        for path in o4.P0_ADDED_PATHS:
            if path == o4.REGISTRY or not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertEqual(o4.RISK_ID_PATTERN.findall(text), [])

    def test_gitattributes_pins_its_own_lf_checkout(self) -> None:
        lines = o4.GIT_ATTRIBUTES.read_text(encoding="utf-8").splitlines()
        self.assertIn(".gitattributes text eol=lf", lines)
        self.assertIn("*.mjs text eol=lf", lines)

    def test_manifest_delete_and_tamper_fail_exact_validation(self) -> None:
        expected = o4.json_text({"name": "manifest", "outputs": [{"path": "a"}]})
        o4.verify_manifest_exact(expected, expected)
        with self.assertRaises(ValueError):
            o4.verify_manifest_exact("", expected)
        with self.assertRaises(ValueError):
            o4.verify_manifest_exact(
                o4.json_text({"name": "manifest", "outputs": []}), expected
            )

    def test_git_attributes_is_a_protocol_input(self) -> None:
        self.assertIn(o4.GIT_ATTRIBUTES, o4.protocol_inputs(self.reg))
        self.assertIn(o4.AGENT_DOC, o4.protocol_inputs(self.reg))
        self.assertIn(o4.MEASUREMENT_SCRIPT, o4.protocol_inputs(self.reg))
        self.assertIn(o4.STAGE_SCRIPT, o4.protocol_inputs(self.reg))
        self.assertIn(o4.STAGE_TEST, o4.protocol_inputs(self.reg))
        self.assertIn(o4.DRAND_VERIFIER_PACKAGE, o4.protocol_inputs(self.reg))
        self.assertIn(o4.DRAND_VERIFIER_LOCK, o4.protocol_inputs(self.reg))
        self.assertIn(o4.DRAND_VERIFIER_SCRIPT, o4.protocol_inputs(self.reg))

    def test_salts_are_commit_derived_and_purpose_separated(self) -> None:
        first = o4.derive_salt(
            self.reg,
            self.preview_commit,
            self.preview_randomness,
            "prompt_matching",
        )
        second = o4.derive_salt(
            self.reg,
            self.preview_commit,
            self.preview_randomness,
            "opaque_prompt_id",
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            first,
            o4.sha256_text(
                f"step3o_o4_v2|{self.preview_commit}|1|{'0' * 64}|prompt_matching"
            ),
        )

    def test_p1_creation_commit_rejects_an_extra_path(self) -> None:
        temp_root = o4.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                result = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
                return result.stdout.strip()

            git("init")
            git("config", "user.email", "o4-test@example.invalid")
            git("config", "user.name", "O4 Test")
            git("config", "core.autocrlf", "false")
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            git("add", "base.txt")
            git("commit", "-m", "base")
            parent = git("rev-parse", "HEAD")
            ledger = root / "ledger.json"
            manifest = root / "manifest.json"
            extra = root / "unregistered.txt"
            ledger.write_text("{}\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            extra.write_text("smuggled\n", encoding="utf-8")
            git("add", "ledger.json", "manifest.json", "unregistered.txt")
            git("commit", "-m", "invalid P1")
            with patch.object(o4, "REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "unexpected path set"):
                    o4.verify_exact_creation_commit(
                        [ledger, manifest],
                        manifest,
                        parent,
                        reg=self.reg,
                        require_pushed=False,
                    )

    def test_p1_terminal_stop_binds_evidence_hash_and_exact_commit_paths(self) -> None:
        temp_root = o4.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            request = root / "request.json"
            pulse = root / "pulse.json"
            evidence = root / "ledger.json"
            stop = root / "stop.json"
            for path, data in (
                (request, b"request\n"),
                (pulse, b"pulse\n"),
                (evidence, b"evidence\n"),
            ):
                path.write_bytes(data)
            protocol_commit = "1" * 40
            pulse_commit = "2" * 40
            stop_commit = "3" * 40
            randomness = {
                "pulse_commit": pulse_commit,
                "beacon_round": 123,
            }
            with patch.object(o4, "REPO_ROOT", root):
                value = o4.build_attempt_stop_event(
                    self.reg,
                    protocol_commit,
                    randomness,
                    "synthetic_phase",
                    ValueError("synthetic failure"),
                    [o4.artifact_from_bytes(evidence, evidence.read_bytes())],
                )
                stop.write_text(o4.json_text(value), encoding="utf-8", newline="\n")
                expected_tree = "\n".join(
                    o4.repo_path(path) for path in (request, pulse, evidence, stop)
                )
                with (
                    patch.object(o4, "BASE", root),
                    patch.object(o4, "BEACON_REQUEST", request),
                    patch.object(o4, "BEACON_PULSE", pulse),
                    patch.object(o4, "ATTEMPT_STOP", stop),
                    patch.object(o4, "registry", return_value=self.reg),
                    patch.object(o4, "validate_boundaries"),
                    patch.object(
                        o4, "resolve_protocol_commit", return_value=protocol_commit
                    ),
                    patch.object(o4, "require_protocol_inputs_frozen"),
                    patch.object(
                        o4,
                        "verify_randomness_chain",
                        return_value=randomness,
                    ),
                    patch.object(
                        o4, "expected_output_paths", return_value=[evidence]
                    ),
                    patch.object(
                        o4,
                        "verify_exact_creation_commit",
                        return_value=stop_commit,
                    ) as verifier,
                    patch.object(o4, "_git", return_value=expected_tree),
                ):
                    result = o4.verify_prompt_stop()
                self.assertEqual(result["commit"], stop_commit)
                verifier.assert_called_once_with(
                    [evidence, stop],
                    stop,
                    pulse_commit,
                    reg=self.reg,
                    require_pushed=True,
                )

                tampered = copy.deepcopy(value)
                tampered["evidence_artifacts"][0]["sha256"] = "f" * 64
                stop.write_text(
                    o4.json_text(tampered), encoding="utf-8", newline="\n"
                )
                with (
                    patch.object(o4, "BASE", root),
                    patch.object(o4, "BEACON_REQUEST", request),
                    patch.object(o4, "BEACON_PULSE", pulse),
                    patch.object(o4, "ATTEMPT_STOP", stop),
                    patch.object(o4, "registry", return_value=self.reg),
                    patch.object(o4, "validate_boundaries"),
                    patch.object(
                        o4, "resolve_protocol_commit", return_value=protocol_commit
                    ),
                    patch.object(o4, "require_protocol_inputs_frozen"),
                    patch.object(
                        o4,
                        "verify_randomness_chain",
                        return_value=randomness,
                    ),
                    patch.object(
                        o4, "expected_output_paths", return_value=[evidence]
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "hash or length"):
                        o4.verify_prompt_stop()


if __name__ == "__main__":
    unittest.main()
