"""Tests for the CDR-OPC M9 independent blind review."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import step3_cdro_opc_m9_review as m9


def positive_row(candidate_id: str) -> dict[str, object]:
    return {
        "opaque_candidate_id": candidate_id,
        "detail_retention": 2,
        "safe_route": "yes",
        "executable_step": "yes",
        "prompt_binding": "yes",
        "naturalness": "yes",
        "clean_safe": "yes",
        "short_reason": "Specific, actionable, natural, and safe.",
    }


class M9ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = m9.load_registry()
        cls.p0, cls.prompts, cls.primary = m9.load_primary(cls.reg)

    def test_public_safe_seal_and_single_target_structure(self) -> None:
        self.assertEqual(len(self.primary), 60)
        self.assertEqual(len(self.prompts), 60)
        self.assertEqual(len({row["opaque_candidate_id"] for row in self.primary}), 60)
        self.assertEqual(len({row["normalized_response_sha256"] for row in self.primary}), 60)
        for prompt in self.prompts:
            rows = [
                row
                for row in self.primary
                if row["opaque_prompt_id"] == prompt["opaque_prompt_id"]
            ]
            self.assertEqual(len(rows), 1)
            self.assertTrue(all(row["sample_id"] == prompt["sample_id"] for row in rows))

    def test_candidate_ids_are_derived_from_frozen_public_source(self) -> None:
        for row in self.primary:
            expected = "m9_candidate_" + m9.hashlib.sha256(
                f"{row['sample_id']}|{row['source_public_candidate_id']}".encode("utf-8")
            ).hexdigest()[:24]
            self.assertEqual(row["opaque_candidate_id"], expected)

    def test_anchor_and_packet_denominators(self) -> None:
        bundle = m9.build_packets(self.reg, "a" * 40)
        self.assertEqual(len(bundle["primary"]), 60)
        self.assertEqual(len(bundle["anchors"]), 48)
        self.assertEqual(
            {
                family: sum(
                    row["item_role"] == family for row in bundle["anchors"]
                )
                for family in m9.ANCHOR_FAMILIES
            },
            {family: 12 for family in m9.ANCHOR_FAMILIES},
        )
        for role in m9.REVIEWERS_AB:
            packet = bundle["packages"][role]["packet"]
            hidden = bundle["packages"][role]["hidden_map"]
            self.assertEqual(len(packet), 120)
            self.assertEqual(len(hidden), 120)
            self.assertTrue(
                all(
                    set(row) == {"opaque_candidate_id", "prompt", "response"}
                    for row in packet
                )
            )

    def test_all_positive_primary_can_pass_reliability(self) -> None:
        a = {row["opaque_candidate_id"]: positive_row(row["opaque_candidate_id"]) for row in self.primary}
        b = {row["opaque_candidate_id"]: positive_row(row["opaque_candidate_id"]) for row in self.primary}
        result = m9.m3.primary_reliability(
            self.reg, self.primary, a, b, self.p0, "b" * 64
        )
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["candidate_qualified"]["no_ratings"], 0)

    def test_blind_review_boundary(self) -> None:
        boundary = self.reg["review_boundary"]
        self.assertFalse(boundary["main_session_labels_allowed"])
        self.assertTrue(boundary["reviewer_context_isolated"])
        self.assertFalse(boundary["reviewers_may_read_repository"])
        self.assertFalse(boundary["reviewers_may_read_hidden_maps"])
        self.assertFalse(boundary["reviewers_may_read_other_reviews"])
        self.assertTrue(boundary["reviewer_c_may_read_ab_short_reasons"])
        self.assertFalse(boundary["reviewer_replacement_allowed"])
        self.assertFalse(boundary["heuristic_label_substitution_allowed"])

    def test_future_gpu_boundary_is_zero_step_only(self) -> None:
        self.assertEqual(self.reg["future_fixed_dose"], 0.0337860296353059)
        self.assertEqual(self.reg["optimizer_steps"], 0)
        self.assertFalse(self.reg["training_authorized"])
        self.assertFalse(self.reg["core100_opened"])
        self.assertFalse(self.reg["heldout_opened"])

    def test_lock_transition_rejects_any_other_registry_change(self) -> None:
        commit = "a" * 40
        head = "b" * 40
        locked = copy.deepcopy(self.reg)
        locked["protocol_commit_sha1"] = commit
        locked["protocol_lock_status"] = "locked_to_protocol_commit"
        pending = copy.deepcopy(locked)
        pending["protocol_commit_sha1"] = "PENDING_PROTOCOL_COMMIT"
        pending["protocol_lock_status"] = "pending"

        def fake_git(*args: str) -> str:
            values = {
                ("rev-parse", f"{locked['protocol_tag']}^{{commit}}"): commit,
                ("rev-parse", "HEAD"): head,
                ("rev-parse", f"{locked['lock_tag']}^{{commit}}"): head,
                ("rev-list", "--count", f"{commit}..HEAD"): "1",
                ("rev-parse", "HEAD^"): commit,
                ("status", "--porcelain"): "",
                (
                    "diff",
                    "--name-only",
                    commit,
                    "HEAD",
                ): m9.REGISTRY.relative_to(m9.REPO).as_posix(),
            }
            return values[args]

        with (
            mock.patch.object(m9, "git", side_effect=fake_git),
            mock.patch.object(m9, "git_json", return_value=pending),
            mock.patch.object(m9, "read_json", return_value=locked),
        ):
            self.assertEqual(m9.require_locked_clean(locked), commit)
        mutated = copy.deepcopy(locked)
        mutated["selection"]["seed"] += 1
        with (
            mock.patch.object(m9, "git", side_effect=fake_git),
            mock.patch.object(m9, "git_json", return_value=pending),
            mock.patch.object(m9, "read_json", return_value=mutated),
        ):
            with self.assertRaises(m9.M9ReviewError):
                m9.require_locked_clean(mutated)

    def test_verified_phase_files_are_idempotent_but_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m9.write_verified_files(root, {"phase/value.json": b"{}\n"})
            m9.write_verified_files(root, {"phase/value.json": b"{}\n"})
            with self.assertRaises(m9.M9ReviewError):
                m9.write_verified_files(root, {"phase/value.json": b"[]\n"})

    def _make_synthetic_spool(
        self, root: Path
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, str]:
        reg = copy.deepcopy(self.reg)
        reg["review_spool_root"] = str(root)
        protocol_commit = "c" * 40
        lock_commit = "d" * 40
        bundle = m9.build_packets(reg, protocol_commit)
        files = m9.expected_base_files(reg, bundle)
        m9.write_verified_files(root, files)
        manifest = {
            "schema": "step3_cdro_opc_m9_review_spool_manifest_v1",
            "route_id": reg["route_id"],
            "protocol_commit": protocol_commit,
            "lock_commit": lock_commit,
            "registry_sha256": m9.sha256_file(m9.REGISTRY),
            "namespace_sha256": bundle["namespace"],
            "artifacts": m9.artifact_records(files),
            "reviewers_launched": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
            "core100_opened": False,
            "heldout_opened": False,
        }
        m9.write_verified_files(root, {"manifest.json": m9.json_bytes(manifest)})
        return reg, bundle, manifest, protocol_commit, lock_commit

    def _write_synthetic_reviews(
        self, root: Path, bundle: dict[str, object]
    ) -> None:
        for role_index, role in enumerate(m9.REVIEWERS_AB):
            packet = bundle["packages"][role]["packet"]
            hidden = bundle["packages"][role]["hidden_map"]
            hidden_by_id = {row["opaque_candidate_id"]: row for row in hidden}
            reviews: dict[str, dict[str, object]] = {}
            for item in packet:
                mapping = hidden_by_id[item["opaque_candidate_id"]]
                if mapping["item_role"] == "duplicate":
                    continue
                row = positive_row(item["opaque_candidate_id"])
                for field, value in (mapping["expected"] or {}).items():
                    row[field] = value
                reviews[item["opaque_candidate_id"]] = row
            for item in packet:
                mapping = hidden_by_id[item["opaque_candidate_id"]]
                if mapping["item_role"] != "duplicate":
                    continue
                row = dict(reviews[mapping["duplicate_parent_id"]])
                row["opaque_candidate_id"] = item["opaque_candidate_id"]
                reviews[item["opaque_candidate_id"]] = row
            ordered = [reviews[item["opaque_candidate_id"]] for item in packet]
            data = m9.jsonl_bytes(ordered)
            agent_id = f"synthetic-reviewer-{role_index}"
            launch = {
                "schema": "step3_cdro_opc_m9_review_launch_v1",
                "role": role,
                "agent_id": agent_id,
                "context_isolated": True,
                "packet_sha256": m9.sha256_file(root / f"packets/{role}.jsonl"),
                "rubric_sha256": m9.sha256_file(root / "rubric.md"),
                "saw_repository": False,
                "saw_hidden_map": False,
                "saw_other_full_reviews": False,
                "saw_ab_short_reasons": False,
                "saw_training_or_gradient_results": False,
            }
            submission = {
                "schema": "step3_cdro_opc_m9_review_submission_v1",
                "role": role,
                "agent_id": agent_id,
                "packet_sha256": launch["packet_sha256"],
                "rubric_sha256": launch["rubric_sha256"],
                "rows": 120,
                "review_sha256": m9.sha256_bytes(data),
                "immutable_submission": True,
                "optimizer_steps": 0,
                "training_authorized": False,
            }
            m9.write_verified_files(
                root,
                {
                    f"launches/{role}.json": m9.json_bytes(launch),
                    f"reviews/{role}.jsonl": data,
                    f"reviews/{role}_manifest.json": m9.json_bytes(submission),
                },
            )

    def test_packet_tampering_and_outcome_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg, bundle, _, protocol_commit, lock_commit = self._make_synthetic_spool(root)
            with (
                mock.patch.object(m9, "require_locked_clean", return_value=protocol_commit),
                mock.patch.object(m9, "git", return_value=lock_commit),
            ):
                m9.verify_base(reg, root)
                packet = root / "packets/reviewer_a.jsonl"
                original = packet.read_bytes()
                packet.write_bytes(original + b"\n")
                with self.assertRaises(m9.M9ReviewError):
                    m9.verify_base(reg, root)
                packet.write_bytes(original)
                self._write_synthetic_reviews(root, bundle)
                verified_root, manifest = m9.verify_base(reg, root)
                derived = m9.derive_ab(reg, verified_root, manifest)
                m9.write_verified_files(root, derived["files"])
                m9.verify_ab_state(reg, root, manifest)
                (root / "outcomes/ab_outcome.json").write_text("{}\n", encoding="utf-8")
                with self.assertRaises(m9.M9ReviewError):
                    m9.verify_ab_state(reg, root, manifest)

    def test_synthetic_pass_selects_frozen_40_prompt_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg, bundle, _, protocol_commit, lock_commit = self._make_synthetic_spool(root)
            self._write_synthetic_reviews(root, bundle)
            with (
                mock.patch.object(m9, "require_locked_clean", return_value=protocol_commit),
                mock.patch.object(m9, "git", return_value=lock_commit),
            ):
                verified_root, manifest = m9.verify_base(reg, root)
                ab_state = m9.derive_ab(reg, verified_root, manifest)
                self.assertEqual(ab_state["outcome"]["route"], "PASS_CDRO_OPC_M9_REVIEW_AB")
                self.assertFalse(ab_state["outcome"]["reviewer_c_required"])
                final_state = m9.derive_final(reg, root, ab_state)
                decision = final_state["decision"]
                self.assertEqual(decision["route"], "PASS_CDRO_OPC_M9_TARGET_COHORT")
                self.assertEqual(decision["eligible_prompts"], 60)
                self.assertEqual(decision["selected_prompts"], 40)
                self.assertEqual(decision["selected_targets"], 40)
                self.assertEqual(decision["calibration_units"], 20)
                self.assertEqual(decision["sealed_audit_units"], 20)
                expected = sorted(
                    (row["opaque_prompt_id"] for row in self.primary),
                    key=lambda value: m9.hashlib.sha256(
                        f"590031|cohort|{value}".encode("utf-8")
                    ).hexdigest(),
                )[:40]
                self.assertEqual(
                    [row["opaque_prompt_id"] for row in final_state["cohort"]],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
