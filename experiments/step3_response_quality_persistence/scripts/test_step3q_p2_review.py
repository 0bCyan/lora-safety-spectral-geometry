from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step3q_calibrated_api_pool as step3q
import step3q_p2_review as p2


def response_text(index: int) -> str:
    return " ".join(
        [f"candidate{index}"] + [f"safe_action_{index}_{position}" for position in range(1, 40)]
    )


def api_response(request: dict, text: str) -> dict:
    output = json.dumps(
        {"opaque_request_id": request["opaque_request_id"], "response": text},
        separators=(",", ":"),
    )
    return {
        "id": f"resp_{request['opaque_request_id']}",
        "status": "completed",
        "error": None,
        "model": "gpt-5.6-sol",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": output}],
            }
        ],
        "usage": {"input_tokens": 20, "output_tokens": 40, "total_tokens": 60},
    }


def receipt_for(request: dict, registry: dict, instructions: str, index: int) -> dict:
    candidate = step3q.validate_api_response(
        api_response(request, response_text(index)), request, registry
    )
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
                "raw_response_sha256": f"{index:064x}",
                "api_response_id": candidate["api_response_id"],
                "effective_model": "gpt-5.6-sol",
            }
        ],
        "status": "valid",
        "candidate": candidate,
    }


class Step3QP2ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="step3q-p2-tests-")
        cls.root = Path(cls.temporary.name)
        cls.registry_path = p2.DEFAULT_REGISTRY
        cls.registry, cls.p0 = p2.load_registry(cls.registry_path)
        cls.prepared_root = p2.prepared_root(cls.registry)
        cls.p1_manifest = p2.read_json(cls.prepared_root / "protocol_manifest.json")
        cls.requests = p2.read_jsonl(
            cls.prepared_root / "authoring" / "api_requests.jsonl"
        )
        instructions_path = p2.resolve_repo_path(
            cls.p0["inputs"]["authoring_instructions_path"]
        )
        cls.instructions = instructions_path.read_text(encoding="utf-8")

        cls.spool = cls.root / "external-spool"
        receipt_dir = cls.spool / "receipts"
        receipt_dir.mkdir(parents=True)
        receipts = []
        for index, request in enumerate(cls.requests):
            receipt = receipt_for(request, cls.p0, cls.instructions, index)
            receipts.append(receipt)
            (receipt_dir / f"{request['opaque_request_id']}.json").write_text(
                step3q.json_text(receipt), encoding="utf-8", newline="\n"
            )
        step3q.seal_dispatch_receipts(
            cls.p0,
            cls.p1_manifest,
            cls.requests,
            receipts,
            receipt_dir,
            cls.spool / "sealed",
            cls.instructions,
        )
        cls.dispatch = p2.verify_external_dispatch(cls.registry, cls.p0, cls.spool)
        files, cls.import_summary = p2.build_import_files(
            cls.registry,
            cls.p0,
            cls.dispatch,
            "a" * 40,
            cls.registry_path,
        )
        cls.output_root = cls.root / "imported"
        p2.write_bundle(cls.output_root, files)
        cls.bundle = p2.load_import_bundle(cls.output_root, cls.registry)
        cls.sources = p2._load_p1_review_sources(cls.registry)

        by_prompt: dict[str, list[str]] = {}
        for row in cls.bundle["primary"]:
            by_prompt.setdefault(row["opaque_prompt_id"], []).append(
                row["opaque_candidate_id"]
            )
        cls.qualified_candidate_ids = {
            sorted(candidate_ids)[0] for candidate_ids in by_prompt.values()
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def review_rows(
        cls,
        reviewer: str,
        *,
        primary_overrides: dict[str, str] | None = None,
        inconsistent_duplicate_ids: set[str] | None = None,
    ) -> list[dict]:
        packet = cls.bundle[f"packet_{reviewer}"]
        hidden = cls.bundle[f"hidden_{reviewer}"]
        mapping = {row["opaque_candidate_id"]: row for row in hidden}
        references = cls.sources["calibration_reference_by_id"]
        primary_overrides = primary_overrides or {}
        inconsistent_duplicate_ids = inconsistent_duplicate_ids or set()
        rows = []
        for packet_row in packet:
            visible_id = packet_row["opaque_candidate_id"]
            hidden_row = mapping[visible_id]
            canonical_id = hidden_row["canonical_candidate_id"]
            row = {
                "opaque_candidate_id": visible_id,
                "detail_retention": 2,
                "safe_route": "yes",
                "executable_step": "yes",
                "prompt_binding": "yes",
                "naturalness": "yes",
                "clean_safe": "yes",
                "short_reason": "Synthetic protocol test judgment.",
            }
            if hidden_row["item_role"] == "sentinel":
                reference = references[canonical_id]
                row.update(reference["expected"])
            elif hidden_row["item_role"] in {"primary", "duplicate"}:
                label = primary_overrides.get(
                    canonical_id,
                    "yes" if canonical_id in cls.qualified_candidate_ids else "no",
                )
                if label == "no":
                    row["detail_retention"] = 1
            if visible_id in inconsistent_duplicate_ids:
                row["naturalness"] = "no" if row["naturalness"] == "yes" else "yes"
            rows.append(row)
        return rows

    def test_external_120_receipt_spool_is_recomputed_byte_for_byte(self) -> None:
        self.assertEqual(self.dispatch["summary"]["route"], "PASS_API_AUTHORING")
        self.assertEqual(self.dispatch["summary"]["terminal_receipts"], 120)
        self.assertEqual(self.dispatch["summary"]["valid_unique_candidates"], 120)
        self.assertEqual(self.dispatch["summary"]["valid_prompts"], 60)
        self.assertEqual(len(self.dispatch["candidates"]), 120)

        tampered = self.root / "tampered-spool"
        shutil.copytree(self.spool, tampered)
        summary_path = tampered / "sealed" / "dispatch_summary.json"
        summary_path.write_bytes(summary_path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            p2.Step3QP2ValidationError, "differs from recomputation"
        ):
            p2.verify_external_dispatch(self.registry, self.p0, tampered)

    def test_role_blind_remapping_and_sentinel_pair_closure(self) -> None:
        primary_maps = {}
        for reviewer in p2.REVIEWERS_AB:
            packet = self.bundle[f"packet_{reviewer}"]
            hidden = self.bundle[f"hidden_{reviewer}"]
            self.assertEqual(len(packet), 120 + 24 + 12)
            self.assertTrue(
                all(p2.REVIEW_ID.fullmatch(row["opaque_candidate_id"]) for row in packet)
            )
            packet_bytes = p2.jsonl_text(packet).encode("utf-8")
            for forbidden in (
                b"step3q_candidate_",
                b"step3q_cal_",
                b"specific_redirect",
                b"risk_mitigation",
            ):
                self.assertNotIn(forbidden, packet_bytes)
            roles = [row["item_role"] for row in hidden]
            self.assertEqual(roles.count("primary"), 120)
            self.assertEqual(roles.count("sentinel"), 24)
            self.assertEqual(roles.count("duplicate"), 12)
            selected = {
                row["canonical_candidate_id"]
                for row in hidden
                if row["item_role"] == "sentinel"
            }
            for sentinel_id in selected:
                reference = self.sources["calibration_reference_by_id"][sentinel_id]
                if reference["calibration_role"] == "prompt_mismatch":
                    self.assertIn(reference["matched_anchor_candidate_id"], selected)
            primary_maps[reviewer] = {
                row["canonical_candidate_id"]: row["opaque_candidate_id"]
                for row in hidden
                if row["item_role"] == "primary"
            }
        self.assertEqual(set(primary_maps["reviewer_a"]), set(primary_maps["reviewer_b"]))
        self.assertTrue(
            all(
                primary_maps["reviewer_a"][candidate_id]
                != primary_maps["reviewer_b"][candidate_id]
                for candidate_id in primary_maps["reviewer_a"]
            )
        )

    def test_calibration_failure_never_creates_primary_metrics(self) -> None:
        rows_a = self.review_rows("reviewer_a")
        duplicate_ids = [
            row["opaque_candidate_id"]
            for row in self.bundle["hidden_reviewer_b"]
            if row["item_role"] == "duplicate"
        ][:2]
        rows_b = self.review_rows(
            "reviewer_b", inconsistent_duplicate_ids=set(duplicate_ids)
        )
        files, report = p2.build_ab_outcome_files(
            self.bundle,
            self.registry,
            self.p0,
            rows_a,
            rows_b,
            {"reviewer_id": "reviewer-a-session"},
            {"reviewer_id": "reviewer-b-session"},
            p2.jsonl_text(rows_a).encode("utf-8"),
            p2.jsonl_text(rows_b).encode("utf-8"),
            b"{}\n",
            b"{}\n",
        )
        self.assertEqual(report["route"], "STOP_REVIEW_CALIBRATION")
        self.assertFalse(report["primary_metrics_created"])
        self.assertNotIn("primary/ab_reliability.json", files)
        self.assertFalse(any(path.startswith("reviewer_c/") for path in files))

    def test_calibration_and_primary_reliability_pass_on_balanced_exact_reviews(self) -> None:
        rows_a = self.review_rows("reviewer_a")
        rows_b = self.review_rows("reviewer_b")
        score_a = p2.score_reviewer_calibration(
            rows_a,
            self.bundle["hidden_reviewer_a"],
            self.sources,
            self.p0,
            "reviewer_a",
        )
        score_b = p2.score_reviewer_calibration(
            rows_b,
            self.bundle["hidden_reviewer_b"],
            self.sources,
            self.p0,
            "reviewer_b",
        )
        self.assertTrue(score_a["overall_pass"])
        self.assertTrue(score_b["overall_pass"])
        primary_ids = {row["opaque_candidate_id"] for row in self.bundle["primary"]}
        canonical_a = p2.canonical_primary_reviews(
            rows_a, self.bundle["hidden_reviewer_a"], primary_ids, "reviewer_a"
        )
        canonical_b = p2.canonical_primary_reviews(
            rows_b, self.bundle["hidden_reviewer_b"], primary_ids, "reviewer_b"
        )
        reliability = p2.compute_primary_reliability(
            self.bundle["primary"],
            canonical_a,
            canonical_b,
            self.p0,
            self.bundle["manifest"]["namespace_sha256"],
        )
        self.assertTrue(reliability["overall_pass"])
        self.assertEqual(reliability["candidate_qualified"]["yes_ratings"], 120)
        self.assertEqual(reliability["candidate_qualified"]["no_ratings"], 120)
        self.assertTrue(
            reliability["component_class_specific"]["safe_route"]["negative"][
                "supported_on_primary"
            ]
            is False
        )

    def test_reviewer_c_packet_contains_only_derived_disagreement(self) -> None:
        rows_a = self.review_rows("reviewer_a")
        disagreement = sorted(self.qualified_candidate_ids)[0]
        rows_b = self.review_rows(
            "reviewer_b", primary_overrides={disagreement: "no"}
        )
        files, report = p2.build_ab_outcome_files(
            self.bundle,
            self.registry,
            self.p0,
            rows_a,
            rows_b,
            {"reviewer_id": "reviewer-a-session"},
            {"reviewer_id": "reviewer-b-session"},
            p2.jsonl_text(rows_a).encode("utf-8"),
            p2.jsonl_text(rows_b).encode("utf-8"),
            b"{}\n",
            b"{}\n",
        )
        self.assertEqual(report["route"], "PASS_AB_REVIEW")
        c_packet = p2.parse_jsonl_bytes(
            files["reviewer_c/reviewer_c_packet.jsonl"], "Reviewer C packet"
        )
        c_hidden = p2.parse_jsonl_bytes(
            files["private/reviewer_c_hidden_map.jsonl"], "Reviewer C hidden map"
        )
        self.assertEqual(len(c_packet), 1)
        self.assertEqual(c_hidden[0]["canonical_candidate_id"], disagreement)
        self.assertEqual(c_hidden[0]["item_role"], "primary_disagreement")

    def test_fixed_flow_selects_48_without_api_mode_quota(self) -> None:
        rows_a = self.review_rows("reviewer_a")
        rows_b = self.review_rows("reviewer_b")
        files, report = p2.build_final_files(
            self.bundle,
            self.registry,
            self.p0,
            rows_a,
            rows_b,
            [],
            [],
            None,
            None,
        )
        self.assertEqual(report["route"], "PASS_CALIBRATED_API_POOL")
        self.assertFalse(report["training_authorized"])
        self.assertTrue(report["no_api_mode_quota_or_preference"])
        selected = p2.parse_jsonl_bytes(
            files["machine_rubric_candidate_pool.jsonl"], "selected targets"
        )
        self.assertEqual(len(selected), 48)
        self.assertEqual(len({row["opaque_prompt_id"] for row in selected}), 48)
        self.assertEqual(
            len({row["normalized_response_sha256"] for row in selected}), 48
        )
        self.assertEqual(set(row["author_mode"] for row in selected), set(step3q.AUTHOR_MODES))

    def test_immutable_bundle_refuses_overwrite_and_detects_extra_file(self) -> None:
        root = self.root / "immutable-test"
        p2.write_bundle(root, {"artifact.json": b"{}\n"})
        with self.assertRaisesRegex(p2.Step3QP2ValidationError, "already exists"):
            p2.write_bundle(root, {"artifact.json": b"{}\n"})

        copied = self.root / "import-with-extra"
        shutil.copytree(self.output_root, copied)
        (copied / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(
            p2.Step3QP2ValidationError, "file set differs"
        ):
            p2.load_import_bundle(copied, self.registry)

    def test_self_signed_import_manifest_cannot_bypass_receipt_reconstruction(self) -> None:
        copied = self.root / "self-signed-import"
        shutil.copytree(self.output_root, copied)
        summary_path = copied / "import" / "import_summary.json"
        summary = p2.read_json(summary_path)
        summary["raw_receipts_committed"] = True
        summary_path.write_text(p2.json_text(summary), encoding="utf-8", newline="\n")
        manifest_path = copied / "packet_manifest.json"
        manifest = p2.read_json(manifest_path)
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "import/import_summary.json":
                content = summary_path.read_bytes()
                artifact["bytes"] = len(content)
                artifact["sha256"] = p2.sha256_bytes(content)
        manifest_path.write_text(p2.json_text(manifest), encoding="utf-8", newline="\n")
        self_signed = p2.load_import_bundle(copied, self.registry)
        with self.assertRaisesRegex(
            p2.Step3QP2ValidationError, "receipt reconstruction"
        ):
            p2.verify_import_reconstruction(
                copied,
                self_signed,
                self.registry,
                self.p0,
                self.dispatch,
                self.registry_path,
            )

    def test_self_signed_ab_pass_cannot_bypass_outcome_reconstruction(self) -> None:
        copied = self.root / "self-signed-ab"
        shutil.copytree(self.output_root, copied)
        bundle = p2.load_import_bundle(copied, self.registry)
        rows_a = self.review_rows("reviewer_a")
        rows_b = self.review_rows("reviewer_b")
        review_bytes_a = p2.jsonl_text(rows_a).encode("utf-8")
        review_bytes_b = p2.jsonl_text(rows_b).encode("utf-8")
        rubric_bytes = p2.resolve_repo_path(
            self.registry["frozen_inputs"]["review_rubric"]["path"]
        ).read_bytes()
        provenance = {}
        for reviewer, review_bytes in (
            ("reviewer_a", review_bytes_a),
            ("reviewer_b", review_bytes_b),
        ):
            packet_bytes = (copied / "packets" / f"{reviewer}_packet.jsonl").read_bytes()
            value = p2._provenance_template(
                self.registry,
                reviewer,
                packet_bytes,
                rubric_bytes,
                len(bundle[f"packet_{reviewer}"]),
            )
            value["reviewer_id"] = f"{reviewer}-session"
            value["model"] = "synthetic-reviewer"
            value["review_sha256"] = p2.sha256_bytes(review_bytes)
            provenance[reviewer] = value
        files, _ = p2.build_ab_outcome_files(
            bundle,
            self.registry,
            self.p0,
            rows_a,
            rows_b,
            provenance["reviewer_a"],
            provenance["reviewer_b"],
            review_bytes_a,
            review_bytes_b,
            p2.json_text(provenance["reviewer_a"]).encode("utf-8"),
            p2.json_text(provenance["reviewer_b"]).encode("utf-8"),
        )
        p2.write_bundle(copied / "ab_outcome", files)
        route_path = copied / "ab_outcome" / "ab_route.json"
        route = p2.read_json(route_path)
        route["calibration_pass"] = False
        route_path.write_text(p2.json_text(route), encoding="utf-8", newline="\n")
        manifest_path = copied / "ab_outcome" / "outcome_manifest.json"
        manifest = p2.read_json(manifest_path)
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "ab_route.json":
                content = route_path.read_bytes()
                artifact["bytes"] = len(content)
                artifact["sha256"] = p2.sha256_bytes(content)
        manifest_path.write_text(p2.json_text(manifest), encoding="utf-8", newline="\n")
        ab = p2.load_ab_outcome(copied, self.registry, bundle)
        with self.assertRaisesRegex(
            p2.Step3QP2ValidationError, "deterministic reconstruction"
        ):
            p2.verify_ab_outcome_reconstruction(
                copied, bundle, ab, self.registry, self.p0
            )


if __name__ == "__main__":
    unittest.main()
