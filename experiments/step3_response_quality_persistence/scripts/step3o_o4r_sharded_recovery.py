#!/usr/bin/env python3
"""Append-only Step3O-O4R sharded recovery protocol and stage runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import step3o_o4_qualification as qualification
import step3o_o4_stage as original_stage
import step3o_o4_training_pool as original_pool


ROOT = original_pool.ROOT
REPO_ROOT = original_pool.REPO_ROOT
PLAN = ROOT / "Step3O_O4R_sharded_recovery_plan.md"
REGISTRY = ROOT / "configs/step3o_o4r_sharded_recovery_registry.json"
SCRIPT = ROOT / "scripts/step3o_o4r_sharded_recovery.py"
TEST = ROOT / "scripts/test_step3o_o4r_sharded_recovery.py"

SOURCE_BASE = ROOT / "results/step3o_o4_training_pool_attempt_005"
SOURCE_AUTHOR_A_PACKET = SOURCE_BASE / "packets/authoring/step3o_o4_author_a_packet.jsonl"
SOURCE_AUTHOR_B_PACKET = SOURCE_BASE / "packets/authoring/step3o_o4_author_b_packet.jsonl"
SOURCE_AUTHOR_KEY = SOURCE_BASE / "packets/authoring/step3o_o4_authoring_hidden_key.csv"
SOURCE_P1_MANIFEST = SOURCE_BASE / "reports/step3o_o4_p1_prompt_freeze_manifest.json"
SOURCE_AUTHOR_RECEIPT = SOURCE_BASE / "reports/step3o_o4_p2_author_input_receipt.json"
SOURCE_TERMINAL = SOURCE_BASE / "reports/step3o_o4_attempt_005_terminal_stop.json"
AUTHORING_RUBRIC = original_pool.AUTHORING_INSTRUCTIONS
QUALIFICATION_RUBRIC = original_pool.QUALIFICATION_RUBRIC

BASE = ROOT / "results/step3o_o4r_sharded_recovery_attempt_001"
P0_AUTHOR_PACKETS = BASE / "packets/authoring"
AUTHORS = BASE / "authors"
PROVENANCE = BASE / "provenance"
QUALIFICATION_PACKETS = BASE / "packets/qualification"
REVIEWS = BASE / "reviews"
DIAGNOSTICS = BASE / "diagnostics"
REPORTS = BASE / "reports"

P0_MANIFEST = REPORTS / "step3o_o4r_p0_manifest.json"
P0_REPORT = REPORTS / "Step3O_O4R_P0_report.md"
AUTHOR_RECEIPT = REPORTS / "step3o_o4r_p1_author_input_receipt.json"
AUTHOR_MANIFEST = REPORTS / "step3o_o4r_p1_author_manifest.json"
AUTHOR_REPORT = REPORTS / "Step3O_O4R_P1_author_report.md"
AUTHOR_A_COMBINED = AUTHORS / "step3o_o4r_author_a_combined.jsonl"
AUTHOR_B_COMBINED = AUTHORS / "step3o_o4r_author_b_combined.jsonl"

CANDIDATE_PACKAGE = QUALIFICATION_PACKETS / "step3o_o4r_candidate_package.json"
CANDIDATE_MANIFEST = REPORTS / "step3o_o4r_p2_candidate_manifest.json"
CANDIDATE_REPORT = REPORTS / "Step3O_O4R_P2_candidate_report.md"

AB_RECEIPT = REPORTS / "step3o_o4r_p3_ab_input_receipt.json"
AB_MANIFEST = REPORTS / "step3o_o4r_p3_ab_manifest.json"
AB_REPORT = REPORTS / "Step3O_O4R_P3_ab_report.md"
REVIEW_A_COMBINED = REVIEWS / "step3o_o4r_reviewer_a_combined.jsonl"
REVIEW_B_COMBINED = REVIEWS / "step3o_o4r_reviewer_b_combined.jsonl"

AB_RELIABILITY = DIAGNOSTICS / "step3o_o4r_ab_reliability.json"
C_PACKET_COMBINED = QUALIFICATION_PACKETS / "step3o_o4r_reviewer_c_packet.jsonl"
C_GATE_MANIFEST = REPORTS / "step3o_o4r_p4_c_gate_manifest.json"
C_GATE_REPORT = REPORTS / "Step3O_O4R_P4_c_gate_report.md"

C_RECEIPT = REPORTS / "step3o_o4r_p5_c_input_receipt.json"
C_REVIEW_COMBINED = REVIEWS / "step3o_o4r_reviewer_c_combined.jsonl"
FINAL_LABELS = REVIEWS / "step3o_o4r_final_labels.jsonl"
QUALIFIED_POOL = BASE / "machine_rubric_qualified_candidate_pool.jsonl"
QUALIFIED_POOL_KEY = BASE / "machine_rubric_qualified_candidate_pool_key.csv"
FINAL_YIELD = DIAGNOSTICS / "step3o_o4r_final_yield.json"
FINAL_AGREEMENT = DIAGNOSTICS / "step3o_o4r_final_agreement.json"
ROUTE_DECISION = REPORTS / "step3o_o4r_route_decision.json"
FINAL_MANIFEST = REPORTS / "step3o_o4r_p5_final_manifest.json"
FINAL_REPORT = REPORTS / "Step3O_O4R_final_report.md"
TERMINAL_STOP = REPORTS / "step3o_o4r_terminal_stop.json"

ATTEMPT_ID = "step3o_o4r_attempt_001"
AUTHOR_LANES = ("a", "b")
AUTHOR_SHARDS = 3
AUTHOR_ROWS_PER_SHARD = 20
REVIEW_LANES = ("a", "b")
REVIEW_SHARDS = 6
PRIMARY_PER_REVIEW_SHARD = 20
DUPLICATES_PER_REVIEW_SHARD = 2
C_MAX_ROWS = 18
FULL_SHA = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class RecoveryValidationError(ValueError):
    """A fail-closed O4R protocol or stage violation."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def json_bytes(value: Any) -> bytes:
    return original_pool.json_text(value).encode("utf-8")


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return original_pool.jsonl_text(rows).encode("utf-8")


def csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def parse_json_bytes(data: bytes, context: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryValidationError(f"invalid JSON for {context}") from exc
    if not isinstance(value, dict):
        raise RecoveryValidationError(f"{context} must be a JSON object")
    return value


def registry() -> dict[str, Any]:
    value = parse_json_bytes(REGISTRY.read_bytes(), "O4R registry")
    if value.get("schema") != "step3o_o4r_sharded_recovery_registry_v1":
        raise RecoveryValidationError("O4R registry schema mismatch")
    if value.get("attempt_id") != ATTEMPT_ID:
        raise RecoveryValidationError("O4R attempt ID mismatch")
    if value.get("threat_model") != {
        "malicious_operator_resistant": False,
        "session_freshness_evidence": (
            "operator and platform orchestration attestation without a model-service signature"
        ),
        "precommit_first_receipt_cryptographically_proven": False,
        "prior_output_semantic_reuse_cryptographically_excluded": False,
        "postcommit_artifact_substitution_detected": True,
    }:
        raise RecoveryValidationError("O4R threat model differs from implementation")
    source = value.get("source")
    if not isinstance(source, dict):
        raise RecoveryValidationError("O4R source registry is invalid")
    for field in (
        "original_protocol_commit",
        "prompt_freeze_commit",
        "terminal_commit",
    ):
        if not isinstance(source.get(field), str) or not FULL_SHA.fullmatch(source[field]):
            raise RecoveryValidationError(f"invalid source commit: {field}")
    if not isinstance(source.get("beacon_randomness"), str) or not HEX64.fullmatch(
        source["beacon_randomness"]
    ):
        raise RecoveryValidationError("invalid source beacon randomness")
    authoring = value.get("authoring")
    review = value.get("review")
    if (
        not isinstance(authoring, dict)
        or authoring.get("lanes") != len(AUTHOR_LANES)
        or authoring.get("shards_per_lane") != AUTHOR_SHARDS
        or authoring.get("rows_per_shard") != AUTHOR_ROWS_PER_SHARD
        or authoring.get("requested_model") != qualification.REQUESTED_MODEL
        or authoring.get("requested_reasoning_effort")
        != qualification.REQUESTED_REASONING_EFFORT
        or authoring.get("sampling_controls") != qualification.SAMPLING_CONTROLS
        or authoring.get("replacement_or_retry_allowed") is not False
    ):
        raise RecoveryValidationError("authoring registry differs from implementation")
    if (
        not isinstance(review, dict)
        or review.get("lanes") != len(REVIEW_LANES)
        or review.get("shards_per_lane") != REVIEW_SHARDS
        or review.get("primary_rows_per_shard") != PRIMARY_PER_REVIEW_SHARD
        or review.get("duplicate_pairs_per_shard") != DUPLICATES_PER_REVIEW_SHARD
        or review.get("visible_rows_per_shard")
        != PRIMARY_PER_REVIEW_SHARD + DUPLICATES_PER_REVIEW_SHARD
        or review.get("per_shard_duplicate_matches_required")
        != DUPLICATES_PER_REVIEW_SHARD
        or review.get("c_sessions") != 1
        or review.get("c_max_rows") != C_MAX_ROWS
        or review.get("requested_model") != qualification.REQUESTED_MODEL
        or review.get("requested_reasoning_effort")
        != qualification.REQUESTED_REASONING_EFFORT
        or review.get("sampling_controls") != qualification.SAMPLING_CONTROLS
        or review.get("replacement_or_retry_allowed") is not False
    ):
        raise RecoveryValidationError("review registry differs from implementation")
    expected_stages = {
        "P0",
        "P1_AUTHOR_INPUT",
        "P1_AUTHOR_OUTCOME",
        "P2_CANDIDATE_OUTCOME",
        "P3_AB_INPUT",
        "P3_AB_OUTCOME",
        "P4_C_GATE_OUTCOME",
        "P5_C_INPUT",
        "P5_FINAL_OUTCOME",
    }
    stage_tags = value.get("stage_tags")
    if (
        not isinstance(stage_tags, dict)
        or set(stage_tags) != expected_stages
        or len(set(stage_tags.values())) != len(stage_tags)
    ):
        raise RecoveryValidationError("stage-tag registry mismatch")
    return value


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        raise RecoveryValidationError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result


def git_head() -> str:
    value = run_git("rev-parse", "HEAD").stdout.strip()
    if not FULL_SHA.fullmatch(value):
        raise RecoveryValidationError("HEAD is not a full Git SHA")
    return value


def require_head(expected: str) -> None:
    if git_head() != expected:
        raise RecoveryValidationError(f"HEAD must equal registered parent {expected}")


def require_clean() -> None:
    if run_git("status", "--porcelain").stdout:
        raise RecoveryValidationError("tracked worktree must be clean before opening a stage")


def creation_commit(path: Path) -> str:
    values = [
        line.strip()
        for line in run_git(
            "log", "--diff-filter=A", "--format=%H", "--", repo_path(path)
        ).stdout.splitlines()
        if line.strip()
    ]
    if len(values) != 1 or not FULL_SHA.fullmatch(values[0]):
        raise RecoveryValidationError(f"path has no unique creation commit: {repo_path(path)}")
    return values[0]


def changed_paths(commit: str) -> set[str]:
    return {
        line.strip()
        for line in run_git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit
        ).stdout.splitlines()
        if line.strip()
    }


def git_blob_bytes(commit: str, path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{repo_path(path)}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode != 0:
        raise RecoveryValidationError(f"cannot read Git blob: {repo_path(path)}")
    return result.stdout


def commit_parent(commit: str) -> str:
    fields = run_git("rev-list", "--parents", "-n", "1", commit).stdout.split()
    if len(fields) != 2:
        raise RecoveryValidationError("stage commit must have exactly one parent")
    return fields[1]


def artifact(path: Path, data: bytes | None = None) -> dict[str, Any]:
    payload = path.read_bytes() if data is None else data
    return {"path": repo_path(path), "bytes": len(payload), "sha256": sha256_bytes(payload)}


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise RecoveryValidationError(f"append-only path already exists: {repo_path(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RecoveryValidationError(
                f"append-only path already exists: {repo_path(path)}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_or_match(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RecoveryValidationError(f"existing P0 artifact differs: {repo_path(path)}")
        return
    write_once(path, data)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return original_pool.read_jsonl(path)


def receipt(name: str, stage: str, inputs: Mapping[str, tuple[bool, bytes]]) -> dict[str, Any]:
    return {
        "name": name,
        "attempt_id": ATTEMPT_ID,
        "stage": stage,
        "inputs": [
            {
                "role": role,
                "present": present,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
            for role, (present, data) in inputs.items()
        ],
        "received_once": True,
        "replacement_or_repair_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def read_incoming(path: Path) -> tuple[bool, bytes]:
    if not path.exists():
        return False, b""
    if not path.is_file() or path.is_symlink():
        raise RecoveryValidationError(f"incoming path is not a regular file: {path}")
    try:
        path.resolve().relative_to(BASE.resolve())
    except ValueError:
        pass
    else:
        raise RecoveryValidationError("incoming files must be outside the frozen O4R tree")
    return True, path.read_bytes()


def ensure_pristine(paths: Sequence[Path]) -> None:
    existing = [repo_path(path) for path in paths if path.exists()]
    if existing:
        raise RecoveryValidationError(f"registered stage path already exists: {existing}")
    if TERMINAL_STOP.exists():
        raise RecoveryValidationError("O4R attempt is already terminal")


def write_stage(contents: Mapping[Path, bytes], ordered_paths: Sequence[Path]) -> None:
    if set(contents) != set(ordered_paths) or len(contents) != len(ordered_paths):
        raise RecoveryValidationError("stage content path set differs from registered paths")
    ensure_pristine(ordered_paths)
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for path in ordered_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                staged.append((temporary, path))
                handle.write(contents[path])
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in staged:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise RecoveryValidationError(
                    f"stage target already exists during publication: {repo_path(path)}"
                ) from exc
            published.append((temporary, path))
    except Exception as exc:
        rollback_errors: list[str] = []
        for temporary, path in reversed(published):
            try:
                if path.exists() and os.path.samefile(temporary, path):
                    path.unlink()
            except OSError:
                rollback_errors.append(repo_path(path))
        if rollback_errors:
            raise RecoveryValidationError(
                f"stage publication failed and rollback was incomplete: {rollback_errors}"
            ) from exc
        raise
    finally:
        for temporary, _path in staged:
            temporary.unlink(missing_ok=True)


def compare_contents(contents: Mapping[Path, bytes]) -> None:
    for path, expected in contents.items():
        if not path.is_file() or path.is_symlink():
            raise RecoveryValidationError(f"required stage artifact is absent: {repo_path(path)}")
        if path.read_bytes() != expected:
            raise RecoveryValidationError(f"stage artifact reconstruction mismatch: {repo_path(path)}")


def validate_receipt(
    value: Mapping[str, Any],
    *,
    name: str,
    stage: str,
    targets: Mapping[str, Path],
) -> dict[str, tuple[bool, bytes]]:
    expected_fields = {
        "name",
        "attempt_id",
        "stage",
        "inputs",
        "received_once",
        "replacement_or_repair_allowed",
        "optimizer_steps",
        "training_authorized",
    }
    if set(value) != expected_fields:
        raise RecoveryValidationError("input receipt field set mismatch")
    if (
        value.get("name") != name
        or value.get("attempt_id") != ATTEMPT_ID
        or value.get("stage") != stage
        or value.get("received_once") is not True
        or value.get("replacement_or_repair_allowed") is not False
        or value.get("optimizer_steps") != 0
        or value.get("training_authorized") is not False
    ):
        raise RecoveryValidationError("input receipt identity or boundary mismatch")
    rows = value.get("inputs")
    if not isinstance(rows, list) or len(rows) != len(targets):
        raise RecoveryValidationError("input receipt role count mismatch")
    if [row.get("role") if isinstance(row, dict) else None for row in rows] != list(targets):
        raise RecoveryValidationError("input receipt role order mismatch")
    result: dict[str, tuple[bool, bytes]] = {}
    for row, (role, path) in zip(rows, targets.items(), strict=True):
        if not isinstance(row, dict) or set(row) != {
            "role",
            "present",
            "bytes",
            "sha256",
        }:
            raise RecoveryValidationError("input receipt row schema mismatch")
        present = row["present"]
        if type(present) is not bool or type(row["bytes"]) is not int or row["bytes"] < 0:
            raise RecoveryValidationError("input receipt presence or byte count is invalid")
        if not isinstance(row["sha256"], str) or not HEX64.fullmatch(row["sha256"]):
            raise RecoveryValidationError("input receipt SHA-256 is invalid")
        if present:
            if not path.is_file() or path.is_symlink():
                raise RecoveryValidationError(f"receipted input is absent: {repo_path(path)}")
            data = path.read_bytes()
        else:
            if path.exists():
                raise RecoveryValidationError(f"absent input unexpectedly exists: {repo_path(path)}")
            data = b""
        if row["bytes"] != len(data) or row["sha256"] != sha256_bytes(data):
            raise RecoveryValidationError(f"input receipt hash or length mismatch: {role}")
        result[role] = (present, data)
    return result


def ingest_inputs(
    incoming_paths: Mapping[str, Path],
    targets: Mapping[str, Path],
    receipt_path: Path,
    *,
    name: str,
    stage: str,
) -> dict[str, Any]:
    if list(incoming_paths) != list(targets):
        raise RecoveryValidationError("incoming and frozen input roles differ")
    ensure_pristine((*targets.values(), receipt_path))
    inputs = {role: read_incoming(incoming_paths[role]) for role in targets}
    receipt_value = receipt(name, stage, inputs)
    contents = {
        path: inputs[role][1]
        for role, path in targets.items()
        if inputs[role][0]
    }
    contents[receipt_path] = json_bytes(receipt_value)
    write_stage(contents, tuple(contents))
    return receipt_value


def verify_input_commit(
    *,
    receipt_path: Path,
    targets: Mapping[str, Path],
    name: str,
    stage: str,
    parent: str,
) -> dict[str, Any]:
    value = parse_json_bytes(receipt_path.read_bytes(), f"{stage} input receipt")
    inputs = validate_receipt(value, name=name, stage=stage, targets=targets)
    paths = tuple(path for role, path in targets.items() if inputs[role][0]) + (
        receipt_path,
    )
    commit = verify_stage_commit(paths, receipt_path, parent)
    return {
        "status": f"{stage}_FROZEN",
        "commit": commit,
        "terminal": False,
        "inputs": inputs,
        "receipt": value,
    }


def stop_record(
    *, stage: str, route: str, parent: str, error: Exception, evidence: Sequence[Path]
) -> dict[str, Any]:
    return {
        "name": "step3o_o4r_terminal_stop_v1",
        "attempt_id": ATTEMPT_ID,
        "stage": stage,
        "route": route,
        "parent_commit": parent,
        "error_type": type(error).__name__,
        "error_message_sha256": sha256_bytes(str(error).encode("utf-8")),
        "evidence_artifacts": [artifact(path) for path in evidence if path.exists()],
        "replacement_or_retry_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def record_stop(
    *, stage: str, route: str, parent: str, error: Exception, evidence: Sequence[Path]
) -> None:
    if TERMINAL_STOP.exists():
        return
    write_once(
        TERMINAL_STOP,
        json_bytes(
            stop_record(
                stage=stage, route=route, parent=parent, error=error, evidence=evidence
            )
        ),
    )


def author_packet(lane: str) -> Path:
    return SOURCE_AUTHOR_A_PACKET if lane == "a" else SOURCE_AUTHOR_B_PACKET


def author_shard_packet(lane: str, shard: int) -> Path:
    return P0_AUTHOR_PACKETS / f"step3o_o4r_author_{lane}_shard_{shard:02d}.jsonl"


def author_output(lane: str, shard: int) -> Path:
    return AUTHORS / f"step3o_o4r_author_{lane}_shard_{shard:02d}.jsonl"


def author_provenance(lane: str, shard: int) -> Path:
    return PROVENANCE / f"step3o_o4r_author_{lane}_shard_{shard:02d}_provenance.json"


def review_packet(lane: str, shard: int) -> Path:
    return QUALIFICATION_PACKETS / f"step3o_o4r_reviewer_{lane}_shard_{shard:02d}.jsonl"


def review_duplicate_map(lane: str, shard: int) -> Path:
    return QUALIFICATION_PACKETS / f"step3o_o4r_reviewer_{lane}_shard_{shard:02d}_duplicates.json"


def review_output(lane: str, shard: int) -> Path:
    return REVIEWS / f"step3o_o4r_reviewer_{lane}_shard_{shard:02d}.jsonl"


def review_provenance(lane: str, shard: int) -> Path:
    return PROVENANCE / f"step3o_o4r_reviewer_{lane}_shard_{shard:02d}_provenance.json"


def c_packet(shard: int) -> Path:
    return QUALIFICATION_PACKETS / f"step3o_o4r_reviewer_c_shard_{shard:02d}.jsonl"


def c_output(shard: int) -> Path:
    return REVIEWS / f"step3o_o4r_reviewer_c_shard_{shard:02d}.jsonl"


def c_provenance(shard: int) -> Path:
    return PROVENANCE / f"step3o_o4r_reviewer_c_shard_{shard:02d}_provenance.json"


def validate_source() -> dict[str, Any]:
    reg = registry()
    source = reg["source"]
    terminal = parse_json_bytes(SOURCE_TERMINAL.read_bytes(), "attempt-005 terminal stop")
    if (
        terminal.get("route") != "STOP_AUTHORING"
        or terminal.get("stage") != "P2_AUTHORING"
        or terminal.get("parent_commit") != source["prompt_freeze_commit"]
    ):
        raise RecoveryValidationError("attempt-005 terminal boundary mismatch")
    if creation_commit(SOURCE_TERMINAL) != source["terminal_commit"]:
        raise RecoveryValidationError("attempt-005 terminal creation commit mismatch")
    if creation_commit(SOURCE_P1_MANIFEST) != source["prompt_freeze_commit"]:
        raise RecoveryValidationError("source P1 creation commit mismatch")
    if commit_parent(source["terminal_commit"]) != source["prompt_freeze_commit"]:
        raise RecoveryValidationError("source terminal commit is not the sole child of P1")
    if run_git(
        "merge-base",
        "--is-ancestor",
        source["original_protocol_commit"],
        source["prompt_freeze_commit"],
        check=False,
    ).returncode != 0:
        raise RecoveryValidationError("source P0 is not an ancestor of source P1")
    for path, commit in (
        (SOURCE_AUTHOR_A_PACKET, source["prompt_freeze_commit"]),
        (SOURCE_AUTHOR_B_PACKET, source["prompt_freeze_commit"]),
        (SOURCE_AUTHOR_KEY, source["prompt_freeze_commit"]),
        (SOURCE_P1_MANIFEST, source["prompt_freeze_commit"]),
        (SOURCE_AUTHOR_RECEIPT, source["terminal_commit"]),
        (SOURCE_TERMINAL, source["terminal_commit"]),
    ):
        if git_blob_bytes(commit, path) != path.read_bytes():
            raise RecoveryValidationError(f"source blob differs from registered commit: {repo_path(path)}")
    p1_manifest = parse_json_bytes(SOURCE_P1_MANIFEST.read_bytes(), "source P1 manifest")
    if (
        p1_manifest.get("protocol_commit") != source["original_protocol_commit"]
        or p1_manifest.get("beacon_round") != source["beacon_round"]
        or p1_manifest.get("beacon_randomness") != source["beacon_randomness"]
        or p1_manifest.get("author_a_prompt_rows") != 60
        or p1_manifest.get("author_b_prompt_rows") != 60
        or p1_manifest.get("training_authorized") is not False
    ):
        raise RecoveryValidationError("source P1 manifest context mismatch")
    receipt_value = parse_json_bytes(SOURCE_AUTHOR_RECEIPT.read_bytes(), "source author receipt")
    receipt_by_role = {
        row.get("role"): row for row in receipt_value.get("inputs", []) if isinstance(row, dict)
    }
    prior_b = receipt_by_role.get("author_b")
    if (
        receipt_value.get("attempt_id") != "step3o_o4_attempt_005"
        or receipt_value.get("stage") != "P2_AUTHORING"
        or receipt_value.get("replacement_or_repair_allowed") is not False
        or not isinstance(prior_b, dict)
        or prior_b.get("bytes") != source["prior_author_b_bytes"]
        or prior_b.get("sha256") != source["prior_author_b_sha256"]
    ):
        raise RecoveryValidationError("source author receipt does not bind prior B bytes")
    evidence = terminal.get("evidence_artifacts")
    if (
        not isinstance(evidence, list)
        or evidence != [artifact(SOURCE_AUTHOR_RECEIPT)]
        or terminal.get("replacement_or_retry_allowed") is not False
        or terminal.get("training_authorized") is not False
        or terminal.get("optimizer_steps") != 0
    ):
        raise RecoveryValidationError("source terminal evidence mismatch")
    packet_a = read_jsonl(SOURCE_AUTHOR_A_PACKET)
    packet_b = read_jsonl(SOURCE_AUTHOR_B_PACKET)
    if len(packet_a) != 60 or len(packet_b) != 60:
        raise RecoveryValidationError("source author packets must each contain 60 rows")
    for packet in (packet_a, packet_b):
        if any(set(row) != {"opaque_prompt_id", "prompt"} for row in packet):
            raise RecoveryValidationError("source author packet schema mismatch")
        ids = [row["opaque_prompt_id"] for row in packet]
        if len(ids) != len(set(ids)):
            raise RecoveryValidationError("source author packet has duplicate IDs")
    if {row["opaque_prompt_id"] for row in packet_a} != {
        row["opaque_prompt_id"] for row in packet_b
    }:
        raise RecoveryValidationError("source author packet ID sets differ")
    return {
        "registry": reg,
        "packet_a": packet_a,
        "packet_b": packet_b,
        "terminal": terminal,
    }


def p0_paths() -> tuple[Path, ...]:
    return (
        PLAN,
        REGISTRY,
        SCRIPT,
        TEST,
        *(author_shard_packet(lane, shard) for lane in AUTHOR_LANES for shard in range(1, 4)),
        P0_MANIFEST,
        P0_REPORT,
    )


def build_p0_contents(state: Mapping[str, Any]) -> tuple[dict[Path, bytes], dict[str, Any]]:
    outputs: dict[Path, bytes] = {}
    for lane in AUTHOR_LANES:
        packet = state[f"packet_{lane}"]
        for shard in range(1, AUTHOR_SHARDS + 1):
            start = (shard - 1) * AUTHOR_ROWS_PER_SHARD
            rows = packet[start : start + AUTHOR_ROWS_PER_SHARD]
            if len(rows) != AUTHOR_ROWS_PER_SHARD:
                raise RecoveryValidationError("author shard size mismatch")
            outputs[author_shard_packet(lane, shard)] = jsonl_bytes(rows)
    manifest = {
        "name": "step3o_o4r_p0_manifest_v1",
        "status": "P0_READY_FOR_COMMIT",
        "attempt_id": ATTEMPT_ID,
        "entry_commit": state["registry"]["entry_commit"],
        "source_protocol_commit": state["registry"]["source"]["original_protocol_commit"],
        "source_prompt_freeze_commit": state["registry"]["source"]["prompt_freeze_commit"],
        "source_terminal_commit": state["registry"]["source"]["terminal_commit"],
        "source_artifacts": [
            artifact(SOURCE_AUTHOR_A_PACKET),
            artifact(SOURCE_AUTHOR_B_PACKET),
            artifact(SOURCE_AUTHOR_KEY),
            artifact(SOURCE_P1_MANIFEST),
            artifact(SOURCE_AUTHOR_RECEIPT),
            artifact(SOURCE_TERMINAL),
        ],
        "author_shards": [artifact(path, data) for path, data in outputs.items()],
        "old_author_output_imported": False,
        "author_sessions_opened": 0,
        "review_sessions_opened": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = f"""# Step3O-O4R P0 Freeze\n\nStatus: READY FOR COMMIT.\n\n- reused prompt rows: `60`;\n- author lanes: `2`;\n- shards per lane: `3`;\n- rows per shard: `20`;\n- prior author output imported: `false`;\n- author or reviewer sessions opened: `0`;\n- optimizer steps: `0`;\n- training authorized: `false`.\n"""
    outputs[P0_MANIFEST] = json_bytes(manifest)
    outputs[P0_REPORT] = report.encode("utf-8")
    return outputs, manifest


def materialize_p0() -> dict[str, Any]:
    state = validate_source()
    require_head(state["registry"]["entry_commit"])
    outputs, manifest = build_p0_contents(state)
    for path, data in outputs.items():
        write_or_match(path, data)
    return manifest


def verify_p0_local() -> dict[str, Any]:
    state = validate_source()
    commit = verify_stage_commit(
        p0_paths(), P0_MANIFEST, state["registry"]["entry_commit"]
    )
    expected, manifest = build_p0_contents(state)
    compare_contents(expected)
    for lane in AUTHOR_LANES:
        packet = state[f"packet_{lane}"]
        rebuilt: list[dict[str, Any]] = []
        for shard in range(1, AUTHOR_SHARDS + 1):
            rows = read_jsonl(author_shard_packet(lane, shard))
            if len(rows) != AUTHOR_ROWS_PER_SHARD:
                raise RecoveryValidationError("P0 author shard row count mismatch")
            rebuilt.extend(rows)
        if rebuilt != packet:
            raise RecoveryValidationError("P0 shards do not reconstruct source packet order")
    return {"commit": commit, "manifest": manifest, "state": state}


def expected_input_hashes(packet: Path, rubric: Path) -> dict[str, str]:
    return {
        repo_path(packet): sha256_bytes(packet.read_bytes()),
        repo_path(rubric): sha256_bytes(rubric.read_bytes()),
    }


def validate_provenance(
    data: bytes,
    *,
    role: str,
    stage: str,
    packet: Path,
    rubric: Path,
    row_count: int,
) -> dict[str, Any]:
    value = parse_json_bytes(data, f"{role} {stage} provenance")
    return qualification.validate_provenance_record(
        value,
        expected_role=role,
        expected_stage=stage,
        expected_attempt_id=ATTEMPT_ID,
        expected_input_hashes=expected_input_hashes(packet, rubric),
        expected_row_count=row_count,
    )


def validate_author_shard_bytes(
    data: bytes, expected_prompt_ids: Sequence[str]
) -> list[dict[str, Any]]:
    if len(expected_prompt_ids) != AUTHOR_ROWS_PER_SHARD:
        raise RecoveryValidationError("author shard expected-ID count mismatch")
    if len(set(expected_prompt_ids)) != len(expected_prompt_ids):
        raise RecoveryValidationError("author shard expected IDs are duplicated")
    text = qualification.decode_utf8_lf(data, context="author shard JSONL")
    rows = qualification.parse_jsonl(text, context="author shard JSONL")
    if len(rows) != AUTHOR_ROWS_PER_SHARD:
        raise RecoveryValidationError("author shard must contain exactly 20 rows")
    validated: list[dict[str, Any]] = []
    for index, (row, expected_id) in enumerate(
        zip(rows, expected_prompt_ids, strict=True), 1
    ):
        if set(row) != {"opaque_prompt_id", "response"}:
            raise RecoveryValidationError(f"author shard row {index} schema mismatch")
        if row["opaque_prompt_id"] != expected_id:
            raise RecoveryValidationError(f"author shard row {index} order mismatch")
        response = row["response"]
        if not isinstance(response, str) or not response.strip():
            raise RecoveryValidationError(f"author shard row {index} response is empty")
        tokens = qualification.whitespace_token_count(response)
        if not qualification.MIN_RESPONSE_TOKENS <= tokens <= qualification.MAX_RESPONSE_TOKENS:
            raise RecoveryValidationError(
                f"author shard row {index} has {tokens} whitespace tokens"
            )
        validated.append(dict(row))
    return validated


def author_input_targets() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for lane in AUTHOR_LANES:
        for shard in range(1, AUTHOR_SHARDS + 1):
            key = f"author_{lane}_{shard:02d}"
            result[key] = author_output(lane, shard)
            result[f"{key}_provenance"] = author_provenance(lane, shard)
    return result


def author_outcome_paths() -> tuple[Path, ...]:
    return (
        AUTHOR_A_COMBINED,
        AUTHOR_B_COMBINED,
        AUTHOR_MANIFEST,
        AUTHOR_REPORT,
    )


def author_incoming_paths(incoming: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for lane in AUTHOR_LANES:
        for shard in range(1, AUTHOR_SHARDS + 1):
            key = f"author_{lane}_{shard:02d}"
            result[key] = incoming / f"{key}.jsonl"
            result[f"{key}_provenance"] = incoming / f"{key}_provenance.json"
    return result


def build_author_stage(
    p0: Mapping[str, Any], inputs: Mapping[str, tuple[bool, bytes]]
) -> tuple[dict[Path, bytes], list[str]]:
    reg = p0["state"]["registry"]
    banned = set(reg["source"]["ineligible_agent_ids"])
    combined: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    outputs: dict[Path, bytes] = {}
    provenance_records: list[dict[str, Any]] = []
    for lane in AUTHOR_LANES:
        for shard in range(1, AUTHOR_SHARDS + 1):
            key = f"author_{lane}_{shard:02d}"
            present, data = inputs[key]
            prov_present, prov_data = inputs[f"{key}_provenance"]
            if not present or not prov_present:
                raise RecoveryValidationError(f"missing incoming author shard: {key}")
            expected_ids = [
                row["opaque_prompt_id"] for row in read_jsonl(author_shard_packet(lane, shard))
            ]
            rows = validate_author_shard_bytes(data, expected_ids)
            provenance = validate_provenance(
                prov_data,
                role=f"author_{lane}",
                stage=f"authoring_shard_{shard:02d}",
                packet=author_shard_packet(lane, shard),
                rubric=AUTHORING_RUBRIC,
                row_count=AUTHOR_ROWS_PER_SHARD,
            )
            if provenance["agent_id"] in banned:
                raise RecoveryValidationError("attempt-005 author agent ID was reused")
            combined[lane].extend(rows)
            provenance_records.append(provenance)
    agent_ids = [record["agent_id"] for record in provenance_records]
    if len(agent_ids) != len(set(agent_ids)):
        raise RecoveryValidationError("author shard agent IDs are not all distinct")
    expected_a = [row["opaque_prompt_id"] for row in p0["state"]["packet_a"]]
    expected_b = [row["opaque_prompt_id"] for row in p0["state"]["packet_b"]]
    qualification.validate_author_jsonl_bytes(jsonl_bytes(combined["a"]), expected_a)
    qualification.validate_author_jsonl_bytes(jsonl_bytes(combined["b"]), expected_b)
    qualification.validate_author_pair(combined["a"], combined["b"])
    combined_a = jsonl_bytes(combined["a"])
    combined_b = jsonl_bytes(combined["b"])
    prior_hash = reg["source"]["prior_author_b_sha256"]
    if sha256_bytes(combined_a) == prior_hash or sha256_bytes(combined_b) == prior_hash:
        raise RecoveryValidationError("a reconstructed lane reused prior Author B bytes")
    outputs[AUTHOR_A_COMBINED] = combined_a
    outputs[AUTHOR_B_COMBINED] = combined_b
    return outputs, agent_ids


def ingest_author_inputs(incoming: Path) -> dict[str, Any]:
    p0 = verify_p0_local()
    require_head(p0["commit"])
    require_clean()
    return ingest_inputs(
        author_incoming_paths(incoming),
        author_input_targets(),
        AUTHOR_RECEIPT,
        name="step3o_o4r_p1_author_input_receipt_v1",
        stage="P1_AUTHOR_INPUT",
    )


def verify_author_input_local() -> dict[str, Any]:
    p0 = verify_p0_local()
    return verify_input_commit(
        receipt_path=AUTHOR_RECEIPT,
        targets=author_input_targets(),
        name="step3o_o4r_p1_author_input_receipt_v1",
        stage="P1_AUTHOR_INPUT",
        parent=p0["commit"],
    )


def build_author_outcome_contents(
    p0: Mapping[str, Any], input_state: Mapping[str, Any]
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    outputs, agent_ids = build_author_stage(p0, input_state["inputs"])
    manifest = {
        "name": "step3o_o4r_p1_author_manifest_v1",
        "status": "PASS_AUTHORING_FROZEN",
        "attempt_id": ATTEMPT_ID,
        "parent_commit": input_state["commit"],
        "input_receipt_sha256": sha256_bytes(AUTHOR_RECEIPT.read_bytes()),
        "author_agent_ids": agent_ids,
        "author_shards": 6,
        "author_rows_per_shard": AUTHOR_ROWS_PER_SHARD,
        "artifacts": [artifact(path, data) for path, data in outputs.items()],
        "replacement_or_repair_performed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = f"""# Step3O-O4R P1 Author Freeze\n\nStatus: PASS.\n\n- author sessions: `6`;\n- rows per session: `20`;\n- distinct recorded session IDs: `true`;\n- exact prior Author B lane bytes matched: `false`;\n- semantic session freshness: `operator/platform attestation`;\n- malicious-operator-proof freshness: `false`;\n- optimizer steps: `0`;\n- training authorized: `false`.\n"""
    outputs[AUTHOR_MANIFEST] = json_bytes(manifest)
    outputs[AUTHOR_REPORT] = report.encode("utf-8")
    return outputs, manifest


def freeze_authors() -> dict[str, Any]:
    input_state = verify_author_input_local()
    require_head(input_state["commit"])
    require_clean()
    p0 = verify_p0_local()
    try:
        contents, manifest = build_author_outcome_contents(p0, input_state)
    except Exception as exc:
        record_stop(
            stage="P1_AUTHOR_OUTCOME",
            route="STOP_AUTHORING",
            parent=input_state["commit"],
            error=exc,
            evidence=(AUTHOR_RECEIPT,),
        )
        raise
    write_stage(contents, author_outcome_paths())
    return manifest


def verify_stage_commit(paths: Sequence[Path], marker: Path, parent: str) -> str:
    if not paths or marker not in paths or len(paths) != len(set(paths)):
        raise RecoveryValidationError("stage path registration is invalid")
    missing = [repo_path(path) for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise RecoveryValidationError(f"stage artifact is absent: {missing}")
    commit = creation_commit(marker)
    if commit_parent(commit) != parent:
        raise RecoveryValidationError(f"stage parent mismatch for {repo_path(marker)}")
    expected = {repo_path(path) for path in paths}
    if changed_paths(commit) != expected:
        raise RecoveryValidationError(f"stage changed-path mismatch for {repo_path(marker)}")
    for path in paths:
        blob = git_blob_bytes(commit, path)
        if blob != path.read_bytes():
            raise RecoveryValidationError(f"stage blob mismatch: {repo_path(path)}")
    return commit


def verify_author_outcome_success_local() -> dict[str, Any]:
    p0 = verify_p0_local()
    input_state = verify_author_input_local()
    commit = verify_stage_commit(
        author_outcome_paths(), AUTHOR_MANIFEST, input_state["commit"]
    )
    expected, manifest = build_author_outcome_contents(p0, input_state)
    compare_contents(expected)
    return {"status": manifest["status"], "commit": commit, "terminal": False}


def verify_authors_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        return verify_terminal_stop("P1_AUTHOR_OUTCOME")
    return verify_author_outcome_success_local()


def prompt_rows() -> list[dict[str, Any]]:
    packet_a = {row["opaque_prompt_id"]: row["prompt"] for row in read_jsonl(SOURCE_AUTHOR_A_PACKET)}
    packet_b = {row["opaque_prompt_id"]: row["prompt"] for row in read_jsonl(SOURCE_AUTHOR_B_PACKET)}
    if packet_a != packet_b:
        raise RecoveryValidationError("source prompt mappings differ across lanes")
    keys = original_pool.read_csv(SOURCE_AUTHOR_KEY)
    if len(keys) != 60:
        raise RecoveryValidationError("source hidden key must contain 60 rows")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in keys:
        prompt_id = item.get("opaque_prompt_id", "")
        if prompt_id in seen or prompt_id not in packet_a:
            raise RecoveryValidationError("source hidden key prompt ID mismatch")
        seen.add(prompt_id)
        rows.append(
            {
                "opaque_prompt_id": prompt_id,
                "prompt": packet_a[prompt_id],
                "assigned_category": item["assigned_category"],
                "severity_level": int(item["severity_level"]),
            }
        )
    return rows


def load_candidate_package() -> dict[str, Any]:
    return parse_json_bytes(CANDIDATE_PACKAGE.read_bytes(), "O4R candidate package")


def recovery_rank(*parts: str) -> str:
    reg = registry()
    source = reg["source"]
    material = "|".join(
        (
            "step3o_o4r_sharding_v1",
            source["original_protocol_commit"],
            str(source["beacon_round"]),
            source["beacon_randomness"],
            *parts,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_review_shards(
    package: Mapping[str, Any], lane: str
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, str]]]]:
    key = "reviewer_a_packet" if lane == "a" else "reviewer_b_packet"
    duplicate_key = "reviewer_a_duplicates" if lane == "a" else "reviewer_b_duplicates"
    packet = package[key]
    links = package[duplicate_key]
    primary_ids = {
        row["opaque_candidate_id"] for row in package["primary_candidates"]
    }
    by_id = {row["opaque_candidate_id"]: dict(row) for row in packet}
    if len(primary_ids) != 120 or len(links) != 12:
        raise RecoveryValidationError("candidate package size mismatch before review sharding")
    shards_primary: list[list[str]] = [[] for _ in range(REVIEW_SHARDS)]
    shard_links: list[list[dict[str, str]]] = [[] for _ in range(REVIEW_SHARDS)]
    selected_primary: set[str] = set()
    for index, link in enumerate(links):
        shard_index = index // DUPLICATES_PER_REVIEW_SHARD
        primary = link["primary_candidate_id"]
        duplicate = link["duplicate_candidate_id"]
        if primary not in primary_ids or duplicate not in by_id or primary in selected_primary:
            raise RecoveryValidationError("invalid or duplicate hidden-repeat link")
        selected_primary.add(primary)
        shards_primary[shard_index].append(primary)
        shard_links[shard_index].append(dict(link))
    remaining = sorted(
        primary_ids - selected_primary,
        key=lambda candidate_id: recovery_rank("review_assign", lane, candidate_id),
    )
    cursor = 0
    for shard_index in range(REVIEW_SHARDS):
        need = PRIMARY_PER_REVIEW_SHARD - len(shards_primary[shard_index])
        shards_primary[shard_index].extend(remaining[cursor : cursor + need])
        cursor += need
    if cursor != len(remaining):
        raise RecoveryValidationError("review shard assignment did not consume all primaries")
    visible_shards: list[list[dict[str, Any]]] = []
    for shard_index in range(REVIEW_SHARDS):
        ids = list(shards_primary[shard_index]) + [
            link["duplicate_candidate_id"] for link in shard_links[shard_index]
        ]
        if len(shards_primary[shard_index]) != PRIMARY_PER_REVIEW_SHARD or len(ids) != 22:
            raise RecoveryValidationError("review shard row count mismatch")
        ordered = sorted(
            ids,
            key=lambda candidate_id: recovery_rank(
                "review_order", lane, f"{shard_index + 1:02d}", candidate_id
            ),
        )
        visible_shards.append([by_id[candidate_id] for candidate_id in ordered])
    flattened_primary = {
        candidate_id for shard in shards_primary for candidate_id in shard
    }
    if flattened_primary != primary_ids or sum(len(shard) for shard in shards_primary) != 120:
        raise RecoveryValidationError("review shards do not partition all primaries exactly once")
    return visible_shards, shard_links


def candidate_stage_paths() -> tuple[Path, ...]:
    return (
        CANDIDATE_PACKAGE,
        *(review_packet(lane, shard) for lane in REVIEW_LANES for shard in range(1, 7)),
        *(review_duplicate_map(lane, shard) for lane in REVIEW_LANES for shard in range(1, 7)),
        CANDIDATE_MANIFEST,
        CANDIDATE_REPORT,
    )


def build_candidate_contents(
    author_state: Mapping[str, Any]
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    reg = registry()
    source = reg["source"]
    author_a = read_jsonl(AUTHOR_A_COMBINED)
    author_b = read_jsonl(AUTHOR_B_COMBINED)
    expected_a = [row["opaque_prompt_id"] for row in read_jsonl(SOURCE_AUTHOR_A_PACKET)]
    expected_b = [row["opaque_prompt_id"] for row in read_jsonl(SOURCE_AUTHOR_B_PACKET)]
    package = qualification.build_candidate_package(
        prompt_rows(),
        author_a,
        author_b,
        expected_a,
        expected_b,
        protocol_commit=source["original_protocol_commit"],
        beacon_round=source["beacon_round"],
        beacon_randomness=source["beacon_randomness"],
    )
    outputs: dict[Path, bytes] = {CANDIDATE_PACKAGE: json_bytes(package)}
    for lane in REVIEW_LANES:
        packets, links = build_review_shards(package, lane)
        for shard, (packet, duplicate_links) in enumerate(zip(packets, links), 1):
            outputs[review_packet(lane, shard)] = jsonl_bytes(packet)
            outputs[review_duplicate_map(lane, shard)] = json_bytes(
                {
                    "name": "step3o_o4r_review_shard_duplicates_v1",
                    "attempt_id": ATTEMPT_ID,
                    "reviewer_lane": lane,
                    "shard": shard,
                    "links": duplicate_links,
                }
            )
    manifest = {
        "name": "step3o_o4r_p2_candidate_manifest_v1",
        "status": "PASS_CANDIDATE_PACKETS_FROZEN",
        "attempt_id": ATTEMPT_ID,
        "parent_commit": author_state["commit"],
        "primary_candidates": 120,
        "review_lanes": 2,
        "review_shards_per_lane": 6,
        "primary_rows_per_shard": 20,
        "duplicate_rows_per_shard": 2,
        "artifacts": [artifact(path, data) for path, data in outputs.items()],
        "review_sessions_opened": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = """# Step3O-O4R P2 Candidate Packets\n\nStatus: PASS.\n\n- primary candidates: `120`;\n- A/B review shards: `6 + 6`;\n- visible rows per shard: `22`;\n- hidden repeats per shard: `2`;\n- review sessions opened: `0`;\n- optimizer steps: `0`;\n- training authorized: `false`.\n"""
    outputs[CANDIDATE_MANIFEST] = json_bytes(manifest)
    outputs[CANDIDATE_REPORT] = report.encode("utf-8")
    return outputs, manifest


def freeze_candidates() -> dict[str, Any]:
    author_state = verify_authors_local()
    if author_state["terminal"]:
        raise RecoveryValidationError("author stop does not authorize candidates")
    require_head(author_state["commit"])
    require_clean()
    try:
        contents, manifest = build_candidate_contents(author_state)
    except Exception as exc:
        record_stop(
            stage="P2_CANDIDATE_OUTCOME",
            route="STOP_PROTOCOL_INTEGRITY",
            parent=author_state["commit"],
            error=exc,
            evidence=(),
        )
        raise
    write_stage(contents, candidate_stage_paths())
    return manifest


def verify_candidate_success_local() -> dict[str, Any]:
    authors = verify_author_outcome_success_local()
    commit = verify_stage_commit(
        candidate_stage_paths(), CANDIDATE_MANIFEST, authors["commit"]
    )
    expected, manifest = build_candidate_contents(authors)
    compare_contents(expected)
    return {"status": manifest["status"], "commit": commit, "terminal": False}


def verify_candidates_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        return verify_terminal_stop("P2_CANDIDATE_OUTCOME")
    return verify_candidate_success_local()


def review_incoming_paths(incoming: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for lane in REVIEW_LANES:
        for shard in range(1, REVIEW_SHARDS + 1):
            key = f"reviewer_{lane}_{shard:02d}"
            result[key] = incoming / f"{key}.jsonl"
            result[f"{key}_provenance"] = incoming / f"{key}_provenance.json"
    return result


def review_input_targets() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for lane in REVIEW_LANES:
        for shard in range(1, REVIEW_SHARDS + 1):
            key = f"reviewer_{lane}_{shard:02d}"
            result[key] = review_output(lane, shard)
            result[f"{key}_provenance"] = review_provenance(lane, shard)
    return result


def review_outcome_paths() -> tuple[Path, ...]:
    return (REVIEW_A_COMBINED, REVIEW_B_COMBINED, AB_MANIFEST, AB_REPORT)


def validate_shard_duplicates(
    rows: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, str]]
) -> int:
    by_id = {row["opaque_candidate_id"]: row for row in rows}
    matches = 0
    for link in links:
        primary = by_id.get(link["primary_candidate_id"])
        duplicate = by_id.get(link["duplicate_candidate_id"])
        if primary is None or duplicate is None:
            raise RecoveryValidationError("review shard is missing a repeat pair")
        if all(
            primary[field] == duplicate[field]
            for field in qualification.DECISION_FIELDS
        ):
            matches += 1
    return matches


def validate_reviewer_shard_bytes(
    data: bytes, expected_candidate_ids: Sequence[str], *, role: str
) -> list[dict[str, Any]]:
    if role not in {"A", "B"}:
        raise RecoveryValidationError("sharded reviewer role must be A or B")
    if len(expected_candidate_ids) != 22 or len(set(expected_candidate_ids)) != 22:
        raise RecoveryValidationError("review shard expected IDs must contain 22 unique rows")
    text = qualification.decode_utf8_lf(data, context=f"Reviewer {role} shard JSONL")
    rows = qualification.parse_jsonl(text, context=f"Reviewer {role} shard JSONL")
    if len(rows) != 22:
        raise RecoveryValidationError("review shard output must contain exactly 22 rows")
    validated: list[dict[str, Any]] = []
    for index, (row, expected_id) in enumerate(
        zip(rows, expected_candidate_ids, strict=True), 1
    ):
        if set(row) != set(qualification.REVIEW_FIELDS):
            raise RecoveryValidationError(f"review shard row {index} schema mismatch")
        if row["opaque_candidate_id"] != expected_id:
            raise RecoveryValidationError(f"review shard row {index} order mismatch")
        if type(row["detail_retention"]) is not int or row["detail_retention"] not in {
            0,
            1,
            2,
        }:
            raise RecoveryValidationError("review shard detail_retention is invalid")
        for field in qualification.BINARY_FIELDS:
            if row[field] not in {"yes", "no"}:
                raise RecoveryValidationError(f"review shard {field} is invalid")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise RecoveryValidationError("review shard short_reason is empty")
        validated.append(dict(row))
    return validated


def build_ab_stage(
    inputs: Mapping[str, tuple[bool, bytes]],
    *,
    author_agent_ids: Sequence[str] = (),
) -> tuple[dict[Path, bytes], list[str], dict[str, list[int]]]:
    package = load_candidate_package()
    banned = set(registry()["source"]["ineligible_agent_ids"]) | set(author_agent_ids)
    outputs: dict[Path, bytes] = {}
    agents: list[str] = []
    lane_maps: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    repeat_matches: dict[str, list[int]] = {"a": [], "b": []}
    for lane in REVIEW_LANES:
        for shard in range(1, REVIEW_SHARDS + 1):
            key = f"reviewer_{lane}_{shard:02d}"
            present, data = inputs[key]
            prov_present, prov_data = inputs[f"{key}_provenance"]
            if not present or not prov_present:
                raise RecoveryValidationError(f"missing incoming review shard: {key}")
            packet_rows = read_jsonl(review_packet(lane, shard))
            expected_ids = [row["opaque_candidate_id"] for row in packet_rows]
            role_letter = lane.upper()
            rows = validate_reviewer_shard_bytes(data, expected_ids, role=role_letter)
            provenance = validate_provenance(
                prov_data,
                role=f"reviewer_{lane}",
                stage=f"qualification_shard_{shard:02d}",
                packet=review_packet(lane, shard),
                rubric=QUALIFICATION_RUBRIC,
                row_count=22,
            )
            if provenance["agent_id"] in banned:
                raise RecoveryValidationError("ineligible or author agent ID reused as reviewer")
            agents.append(provenance["agent_id"])
            duplicate_info = parse_json_bytes(
                review_duplicate_map(lane, shard).read_bytes(), "review duplicate map"
            )
            matches = validate_shard_duplicates(rows, duplicate_info["links"])
            repeat_matches[lane].append(matches)
            for row in rows:
                candidate_id = row["opaque_candidate_id"]
                if candidate_id in lane_maps[lane]:
                    raise RecoveryValidationError("candidate appears in two review shards")
                lane_maps[lane][candidate_id] = row
    if len(agents) != len(set(agents)):
        raise RecoveryValidationError("A/B review agent IDs are not all distinct")
    full_rows: dict[str, list[dict[str, Any]]] = {}
    for lane in REVIEW_LANES:
        packet_key = "reviewer_a_packet" if lane == "a" else "reviewer_b_packet"
        order = [row["opaque_candidate_id"] for row in package[packet_key]]
        if set(order) != set(lane_maps[lane]) or len(order) != len(lane_maps[lane]):
            raise RecoveryValidationError("review shards do not reconstruct full lane packet")
        full_rows[lane] = [lane_maps[lane][candidate_id] for candidate_id in order]
    outputs[REVIEW_A_COMBINED] = jsonl_bytes(full_rows["a"])
    outputs[REVIEW_B_COMBINED] = jsonl_bytes(full_rows["b"])
    return outputs, agents, repeat_matches


def ingest_ab_inputs(incoming: Path) -> dict[str, Any]:
    candidate_state = verify_candidates_local()
    if candidate_state["terminal"]:
        raise RecoveryValidationError("candidate stop does not authorize reviews")
    require_head(candidate_state["commit"])
    require_clean()
    return ingest_inputs(
        review_incoming_paths(incoming),
        review_input_targets(),
        AB_RECEIPT,
        name="step3o_o4r_p3_ab_input_receipt_v1",
        stage="P3_AB_INPUT",
    )


def verify_ab_input_success_local() -> dict[str, Any]:
    candidate_state = verify_candidate_success_local()
    return verify_input_commit(
        receipt_path=AB_RECEIPT,
        targets=review_input_targets(),
        name="step3o_o4r_p3_ab_input_receipt_v1",
        stage="P3_AB_INPUT",
        parent=candidate_state["commit"],
    )


def verify_ab_input_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        raise RecoveryValidationError("terminal attempt has no A/B input stage to verify")
    return verify_ab_input_success_local()


def author_session_ids() -> list[str]:
    manifest = parse_json_bytes(AUTHOR_MANIFEST.read_bytes(), "O4R author manifest")
    values = manifest.get("author_agent_ids")
    if not isinstance(values, list) or len(values) != 6 or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise RecoveryValidationError("author session ID manifest is invalid")
    if len(values) != len(set(values)):
        raise RecoveryValidationError("author session IDs are not distinct")
    return values


def build_ab_outcome_contents(
    input_state: Mapping[str, Any]
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    outputs, agent_ids, repeat_matches = build_ab_stage(
        input_state["inputs"], author_agent_ids=author_session_ids()
    )
    manifest = {
        "name": "step3o_o4r_p3_ab_manifest_v1",
        "status": "PASS_AB_REVIEWS_FROZEN",
        "attempt_id": ATTEMPT_ID,
        "parent_commit": input_state["commit"],
        "input_receipt_sha256": sha256_bytes(AB_RECEIPT.read_bytes()),
        "review_agent_ids": agent_ids,
        "repeat_matches_by_shard": repeat_matches,
        "artifacts": [artifact(path, data) for path, data in outputs.items()],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = """# Step3O-O4R P3 A/B Review Freeze\n\nStatus: PASS INPUT VALIDATION.\n\nTwelve context-isolated review shards were frozen before any A/B reliability comparison. Reliability and the per-shard repeat gate have not yet been opened.\n"""
    outputs[AB_MANIFEST] = json_bytes(manifest)
    outputs[AB_REPORT] = report.encode("utf-8")
    return outputs, manifest


def freeze_ab_reviews() -> dict[str, Any]:
    input_state = verify_ab_input_local()
    if input_state["terminal"]:
        raise RecoveryValidationError("candidate stop does not authorize reviews")
    require_head(input_state["commit"])
    require_clean()
    try:
        contents, manifest = build_ab_outcome_contents(input_state)
    except Exception as exc:
        record_stop(
            stage="P3_AB_OUTCOME",
            route="STOP_REVIEW_INVALID",
            parent=input_state["commit"],
            error=exc,
            evidence=(AB_RECEIPT,),
        )
        raise
    write_stage(contents, review_outcome_paths())
    return manifest


def verify_ab_outcome_success_local() -> dict[str, Any]:
    input_state = verify_ab_input_success_local()
    commit = verify_stage_commit(
        review_outcome_paths(), AB_MANIFEST, input_state["commit"]
    )
    expected, manifest = build_ab_outcome_contents(input_state)
    compare_contents(expected)
    return {"status": manifest["status"], "commit": commit, "terminal": False}


def verify_ab_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        return verify_terminal_stop("P3_AB_OUTCOME")
    return verify_ab_outcome_success_local()


def combined_review_rows(lane: str) -> list[dict[str, Any]]:
    return read_jsonl(REVIEW_A_COMBINED if lane == "a" else REVIEW_B_COMBINED)


def c_gate_pass_paths(shard_count: int) -> tuple[Path, ...]:
    return (
        AB_RELIABILITY,
        C_PACKET_COMBINED,
        *(c_packet(shard) for shard in range(1, shard_count + 1)),
        C_GATE_MANIFEST,
        C_GATE_REPORT,
    )


def _panel_agreement(
    candidate_ids: Sequence[str],
    map_a: Mapping[str, Mapping[str, Any]],
    map_b: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = list(candidate_ids)
    if not ids:
        raise RecoveryValidationError("panel diagnostic requires a nonempty overlap")
    exact = sum(
        all(map_a[candidate_id][field] == map_b[candidate_id][field] for field in qualification.DECISION_FIELDS)
        for candidate_id in ids
    )
    derived = sum(
        qualification.candidate_qualified(map_a[candidate_id])
        == qualification.candidate_qualified(map_b[candidate_id])
        for candidate_id in ids
    )
    return {
        "rows": len(ids),
        "full_vector_exact_count": exact,
        "full_vector_exact_agreement": exact / len(ids),
        "candidate_qualified_match_count": derived,
        "candidate_qualified_raw_agreement": derived / len(ids),
    }


def fixed_panel_diagnostics(
    package: Mapping[str, Any],
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    repeat_matches: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    primary_ids = {
        row["opaque_candidate_id"] for row in package["primary_candidates"]
    }
    map_a = {row["opaque_candidate_id"]: row for row in rows_a}
    map_b = {row["opaque_candidate_id"]: row for row in rows_b}
    shard_ids: dict[str, list[set[str]]] = {"a": [], "b": []}
    for lane in REVIEW_LANES:
        for shard in range(1, REVIEW_SHARDS + 1):
            ids = {
                row["opaque_candidate_id"]
                for row in read_jsonl(review_packet(lane, shard))
                if row["opaque_candidate_id"] in primary_ids
            }
            if len(ids) != PRIMARY_PER_REVIEW_SHARD:
                raise RecoveryValidationError("review shard primary diagnostic size mismatch")
            shard_ids[lane].append(ids)
    by_a = [
        {"reviewer_a_shard": index, **_panel_agreement(sorted(ids), map_a, map_b)}
        for index, ids in enumerate(shard_ids["a"], 1)
    ]
    by_b = [
        {"reviewer_b_shard": index, **_panel_agreement(sorted(ids), map_a, map_b)}
        for index, ids in enumerate(shard_ids["b"], 1)
    ]
    pair_overlaps: list[dict[str, Any]] = []
    for index_a, ids_a in enumerate(shard_ids["a"], 1):
        for index_b, ids_b in enumerate(shard_ids["b"], 1):
            overlap = sorted(ids_a & ids_b)
            if overlap:
                pair_overlaps.append(
                    {
                        "reviewer_a_shard": index_a,
                        "reviewer_b_shard": index_b,
                        **_panel_agreement(overlap, map_a, map_b),
                    }
                )
    if set().union(*(shard_ids["a"])) != primary_ids or set().union(*(shard_ids["b"])) != primary_ids:
        raise RecoveryValidationError("panel diagnostics do not cover all primary candidates")
    return {
        "estimand": "fixed-shard panel pipeline",
        "sessions_per_lane": REVIEW_SHARDS,
        "primary_rows_per_session": PRIMARY_PER_REVIEW_SHARD,
        "hidden_repeat_pairs_per_session": DUPLICATES_PER_REVIEW_SHARD,
        "repeat_matches_by_shard": {
            lane: list(repeat_matches[lane]) for lane in REVIEW_LANES
        },
        "per_shard_repeat_pass": all(
            value == DUPLICATES_PER_REVIEW_SHARD
            for lane in REVIEW_LANES
            for value in repeat_matches[lane]
        ),
        "by_reviewer_a_shard": by_a,
        "by_reviewer_b_shard": by_b,
        "nonempty_session_pair_overlaps": pair_overlaps,
        "interpretation": (
            "descriptive fixed-panel shard diagnostics only; not an individual-reviewer "
            "or reviewer-population reliability claim"
        ),
    }


def build_c_gate_contents(
    ab_state: Mapping[str, Any]
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    reg = registry()["source"]
    package = load_candidate_package()
    rows_a = combined_review_rows("a")
    rows_b = combined_review_rows("b")
    reliability = qualification.compute_ab_reliability(
        package,
        rows_a,
        rows_b,
        protocol_commit=reg["original_protocol_commit"],
        beacon_round=reg["beacon_round"],
        beacon_randomness=reg["beacon_randomness"],
    )
    ab_manifest = parse_json_bytes(AB_MANIFEST.read_bytes(), "O4R A/B manifest")
    repeat_matches = ab_manifest["repeat_matches_by_shard"]
    diagnostics = fixed_panel_diagnostics(package, rows_a, rows_b, repeat_matches)
    reliability["o4r_fixed_panel"] = diagnostics
    reliability["original_o4_gates_pass"] = reliability["overall_pass"]
    reliability["overall_pass"] = bool(
        reliability["overall_pass"] and diagnostics["per_shard_repeat_pass"]
    )
    reliability_data = json_bytes(reliability)
    if not reliability["overall_pass"]:
        error = RecoveryValidationError("registered A/B fixed-panel reliability gate failed")
        stop = stop_record(
            stage="P4_C_GATE_OUTCOME",
            route="STOP_REVIEW_UNRELIABLE",
            parent=ab_state["commit"],
            error=error,
            evidence=(),
        )
        stop["evidence_artifacts"] = [artifact(AB_RELIABILITY, reliability_data)]
        contents = {
            AB_RELIABILITY: reliability_data,
            TERMINAL_STOP: json_bytes(stop),
        }
        return contents, {
            "status": "STOP_REVIEW_UNRELIABLE",
            "overall_pass": False,
            "parent_commit": ab_state["commit"],
            "terminal": True,
        }
    c_rows = qualification.build_c_packet(
        package,
        rows_a,
        rows_b,
        protocol_commit=reg["original_protocol_commit"],
        beacon_round=reg["beacon_round"],
        beacon_randomness=reg["beacon_randomness"],
    )
    if len(c_rows) > C_MAX_ROWS:
        raise RecoveryValidationError("C disagreement packet exceeds the registered one-session cap")
    outputs: dict[Path, bytes] = {
        AB_RELIABILITY: reliability_data,
        C_PACKET_COMBINED: jsonl_bytes(c_rows),
    }
    shard_count = (len(c_rows) + C_MAX_ROWS - 1) // C_MAX_ROWS
    for shard in range(1, shard_count + 1):
        start = (shard - 1) * C_MAX_ROWS
        outputs[c_packet(shard)] = jsonl_bytes(c_rows[start : start + C_MAX_ROWS])
    manifest = {
        "name": "step3o_o4r_p4_c_gate_manifest_v1",
        "status": "PASS_AB_RELIABILITY_C_PACKETS_FROZEN",
        "attempt_id": ATTEMPT_ID,
        "parent_commit": ab_state["commit"],
        "ab_reliability_sha256": sha256_bytes(reliability_data),
        "c_rows": len(c_rows),
        "c_shards": shard_count,
        "c_max_rows": C_MAX_ROWS,
        "artifacts": [
            artifact(path, data) for path, data in outputs.items()
        ],
        "c_sessions_opened": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = f"""# Step3O-O4R P4 C Gate\n\nStatus: PASS.\n\n- original O4 A/B gates pass: `true`;\n- all per-shard 2/2 repeat gates pass: `true`;\n- C disagreement rows: `{len(c_rows)}`;\n- C shards: `{shard_count}`;\n- C sessions opened: `0`;\n- optimizer steps: `0`;\n- training authorized: `false`.\n"""
    outputs[C_GATE_MANIFEST] = json_bytes(manifest)
    outputs[C_GATE_REPORT] = report.encode("utf-8")
    return outputs, {**manifest, "terminal": False}


def freeze_c_gate() -> dict[str, Any]:
    ab_state = verify_ab_local()
    if ab_state["terminal"]:
        raise RecoveryValidationError("A/B input stop does not authorize reliability")
    parent = ab_state["commit"]
    require_head(parent)
    require_clean()
    ensure_pristine(
        (
            AB_RELIABILITY,
            C_PACKET_COMBINED,
            c_packet(1),
            C_GATE_MANIFEST,
            C_GATE_REPORT,
            TERMINAL_STOP,
        )
    )
    try:
        contents, result = build_c_gate_contents(ab_state)
    except Exception as exc:
        if not TERMINAL_STOP.exists():
            record_stop(
                stage="P4_C_GATE_OUTCOME",
                route="STOP_PROTOCOL_INTEGRITY",
                parent=parent,
                error=exc,
                evidence=(AB_RELIABILITY,),
            )
        raise
    write_stage(contents, tuple(contents))
    return result


def verify_c_gate_success_local() -> dict[str, Any]:
    ab = verify_ab_outcome_success_local()
    expected, manifest = build_c_gate_contents(ab)
    if manifest["terminal"]:
        raise RecoveryValidationError("passing C gate does not reproduce registered gates")
    commit = verify_stage_commit(tuple(expected), C_GATE_MANIFEST, ab["commit"])
    compare_contents(expected)
    return {
        "status": manifest["status"],
        "commit": commit,
        "terminal": False,
        "manifest": manifest,
    }


def verify_c_gate_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        stop = parse_json_bytes(TERMINAL_STOP.read_bytes(), "O4R terminal stop")
        result = verify_terminal_stop("P4_C_GATE_OUTCOME")
        if stop.get("route") == "STOP_REVIEW_UNRELIABLE":
            ab = verify_ab_outcome_success_local()
            expected, expected_result = build_c_gate_contents(ab)
            if not expected_result["terminal"]:
                raise RecoveryValidationError("unreliable stop does not reproduce")
            compare_contents(expected)
        return result
    return verify_c_gate_success_local()


def c_incoming_paths(incoming: Path, shard_count: int) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for shard in range(1, shard_count + 1):
        key = f"reviewer_c_{shard:02d}"
        result[key] = incoming / f"{key}.jsonl"
        result[f"{key}_provenance"] = incoming / f"{key}_provenance.json"
    return result


def c_input_targets(shard_count: int) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for shard in range(1, shard_count + 1):
        key = f"reviewer_c_{shard:02d}"
        result[key] = c_output(shard)
        result[f"{key}_provenance"] = c_provenance(shard)
    return result


def final_outcome_paths() -> tuple[Path, ...]:
    return (
        C_REVIEW_COMBINED,
        FINAL_LABELS,
        QUALIFIED_POOL,
        QUALIFIED_POOL_KEY,
        FINAL_YIELD,
        FINAL_AGREEMENT,
        ROUTE_DECISION,
        FINAL_MANIFEST,
        FINAL_REPORT,
    )


def prior_session_ids() -> set[str]:
    author_manifest = parse_json_bytes(AUTHOR_MANIFEST.read_bytes(), "author manifest")
    ab_manifest = parse_json_bytes(AB_MANIFEST.read_bytes(), "A/B manifest")
    values = list(author_manifest["author_agent_ids"]) + list(ab_manifest["review_agent_ids"])
    if len(values) != len(set(values)):
        raise RecoveryValidationError("prior O4R session IDs are not distinct")
    return set(values)


def build_final_stage(
    gate: Mapping[str, Any], inputs: Mapping[str, tuple[bool, bytes]]
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    reg = registry()["source"]
    package = load_candidate_package()
    rows_a = combined_review_rows("a")
    rows_b = combined_review_rows("b")
    c_packet_rows = read_jsonl(C_PACKET_COMBINED)
    combined_c: list[dict[str, Any]] = []
    outputs: dict[Path, bytes] = {}
    agents = prior_session_ids()
    c_agent_ids: list[str] = []
    shard_count = gate["manifest"]["c_shards"]
    for shard in range(1, shard_count + 1):
        key = f"reviewer_c_{shard:02d}"
        present, data = inputs[key]
        prov_present, prov_data = inputs[f"{key}_provenance"]
        if not present or not prov_present:
            raise RecoveryValidationError(f"missing incoming C shard: {key}")
        packet_rows = read_jsonl(c_packet(shard))
        expected_ids = [row["opaque_candidate_id"] for row in packet_rows]
        rows = qualification.validate_reviewer_jsonl_bytes(
            data, expected_ids, role="C"
        )
        provenance = validate_provenance(
            prov_data,
            role="reviewer_c",
            stage=f"qualification_c_shard_{shard:02d}",
            packet=c_packet(shard),
            rubric=QUALIFICATION_RUBRIC,
            row_count=len(expected_ids),
        )
        agent_id = provenance["agent_id"]
        if agent_id in agents or agent_id in c_agent_ids:
            raise RecoveryValidationError("C agent ID is not distinct")
        c_agent_ids.append(agent_id)
        combined_c.extend(rows)
    if shard_count == 0 and inputs:
        raise RecoveryValidationError("zero-row C gate received unexpected session inputs")
    validated_c = qualification.validate_c_packet_and_reviews(
        c_packet_rows,
        combined_c,
        package,
        rows_a,
        rows_b,
        protocol_commit=reg["original_protocol_commit"],
        beacon_round=reg["beacon_round"],
        beacon_randomness=reg["beacon_randomness"],
    )
    final_labels = qualification.adjudicate_reviews(
        package, rows_a, rows_b, validated_c
    )
    yield_summary = qualification.summarize_yield(
        package,
        final_labels,
        protocol_commit=reg["original_protocol_commit"],
        beacon_round=reg["beacon_round"],
        beacon_randomness=reg["beacon_randomness"],
    )
    reliability = parse_json_bytes(AB_RELIABILITY.read_bytes(), "A/B reliability")
    if reliability.get("overall_pass") is not True:
        raise RecoveryValidationError("final stage requires passing A/B reliability")
    route = (
        "PASS_MACHINE_RUBRIC_POOL"
        if yield_summary["overall_pass"]
        else "STOP_TARGET_YIELD"
    )
    visible_pool, hidden_pool = original_stage.selected_pool_rows(package, yield_summary)
    hidden_fields = [
        "opaque_prompt_id",
        "opaque_candidate_id",
        "sample_id",
        "source_index",
        "assigned_category",
        "severity_level",
        "author_role",
        "prompt_sha256",
        "response_sha256",
    ]
    outputs[C_REVIEW_COMBINED] = jsonl_bytes(combined_c)
    outputs[FINAL_LABELS] = jsonl_bytes(final_labels)
    outputs[QUALIFIED_POOL] = jsonl_bytes(visible_pool)
    outputs[QUALIFIED_POOL_KEY] = csv_bytes(hidden_pool, hidden_fields)
    outputs[FINAL_YIELD] = json_bytes(yield_summary)
    outputs[FINAL_AGREEMENT] = json_bytes(reliability)
    decision = {
        "name": "step3o_o4r_route_decision_v1",
        "attempt_id": ATTEMPT_ID,
        "route": route,
        "ab_reliability_pass": True,
        "target_yield_pass": yield_summary["overall_pass"],
        "human_validated": False,
        "heterogeneous_judge_validated": False,
        "training_suitability_validated": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    outputs[ROUTE_DECISION] = json_bytes(decision)
    result = {
        "route": route,
        "c_agent_ids": c_agent_ids,
        "c_rows": len(combined_c),
        "qualified_prompts": yield_summary["best_of_two"]["qualified_prompts"],
        "visible_pool_rows": len(visible_pool),
    }
    return outputs, result


def ingest_c_inputs(incoming: Path) -> dict[str, Any]:
    gate = verify_c_gate_local()
    if gate["terminal"]:
        raise RecoveryValidationError("C-gate stop does not authorize final review")
    require_head(gate["commit"])
    require_clean()
    shard_count = gate["manifest"]["c_shards"]
    return ingest_inputs(
        c_incoming_paths(incoming, shard_count),
        c_input_targets(shard_count),
        C_RECEIPT,
        name="step3o_o4r_p5_c_input_receipt_v1",
        stage="P5_C_INPUT",
    )


def verify_c_input_success_local() -> dict[str, Any]:
    gate = verify_c_gate_success_local()
    shard_count = gate["manifest"]["c_shards"]
    state = verify_input_commit(
        receipt_path=C_RECEIPT,
        targets=c_input_targets(shard_count),
        name="step3o_o4r_p5_c_input_receipt_v1",
        stage="P5_C_INPUT",
        parent=gate["commit"],
    )
    state["gate"] = gate
    return state


def verify_c_input_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        raise RecoveryValidationError("terminal attempt has no C input stage to verify")
    return verify_c_input_success_local()


def build_final_outcome_contents(
    input_state: Mapping[str, Any]
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    gate = input_state["gate"]
    outputs, result = build_final_stage(gate, input_state["inputs"])
    manifest = {
        "name": "step3o_o4r_p5_final_manifest_v1",
        "status": result["route"],
        "attempt_id": ATTEMPT_ID,
        "parent_commit": input_state["commit"],
        "input_receipt_sha256": sha256_bytes(C_RECEIPT.read_bytes()),
        "c_rows": result["c_rows"],
        "c_agent_ids": result["c_agent_ids"],
        "qualified_prompts": result["qualified_prompts"],
        "artifacts": [artifact(path, data) for path, data in outputs.items()],
        "human_validated": False,
        "heterogeneous_judge_validated": False,
        "training_suitability_validated": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = f"""# Step3O-O4R Final Route\n\nOfficial route: `{result['route']}`.\n\n- machine-rubric-qualified prompts: `{result['qualified_prompts']}/60`;\n- human validated: `false`;\n- heterogeneous-judge validated: `false`;\n- training suitability validated: `false`;\n- optimizer steps: `0`;\n- training authorized: `false`.\n\nThis is a fixed-panel machine-rubric pool result, not a behavioral method or mechanism result.\n"""
    outputs[FINAL_MANIFEST] = json_bytes(manifest)
    outputs[FINAL_REPORT] = report.encode("utf-8")
    return outputs, manifest


def freeze_final() -> dict[str, Any]:
    input_state = verify_c_input_local()
    if input_state["terminal"]:
        raise RecoveryValidationError("C-gate stop does not authorize final review")
    require_head(input_state["commit"])
    require_clean()
    try:
        contents, manifest = build_final_outcome_contents(input_state)
    except Exception as exc:
        record_stop(
            stage="P5_FINAL_OUTCOME",
            route="STOP_REVIEW_INVALID",
            parent=input_state["commit"],
            error=exc,
            evidence=(C_RECEIPT,),
        )
        raise
    write_stage(contents, final_outcome_paths())
    return manifest


def verify_final_success_local() -> dict[str, Any]:
    input_state = verify_c_input_success_local()
    commit = verify_stage_commit(
        final_outcome_paths(), FINAL_MANIFEST, input_state["commit"]
    )
    expected, manifest = build_final_outcome_contents(input_state)
    compare_contents(expected)
    return {"status": manifest["status"], "commit": commit, "terminal": False}


def verify_final_local() -> dict[str, Any]:
    if TERMINAL_STOP.exists():
        return verify_terminal_stop("P5_FINAL_OUTCOME")
    return verify_final_success_local()


STOP_STAGE_SPECS: dict[str, dict[str, Any]] = {
    "P1_AUTHOR_OUTCOME": {
        "routes": {"STOP_AUTHORING"},
        "evidence": {repo_path(AUTHOR_RECEIPT): AUTHOR_RECEIPT},
        "required_by_route": {"STOP_AUTHORING": {repo_path(AUTHOR_RECEIPT)}},
    },
    "P2_CANDIDATE_OUTCOME": {
        "routes": {"STOP_PROTOCOL_INTEGRITY"},
        "evidence": {},
        "required_by_route": {"STOP_PROTOCOL_INTEGRITY": set()},
    },
    "P3_AB_OUTCOME": {
        "routes": {"STOP_REVIEW_INVALID"},
        "evidence": {repo_path(AB_RECEIPT): AB_RECEIPT},
        "required_by_route": {"STOP_REVIEW_INVALID": {repo_path(AB_RECEIPT)}},
    },
    "P4_C_GATE_OUTCOME": {
        "routes": {"STOP_REVIEW_UNRELIABLE", "STOP_PROTOCOL_INTEGRITY"},
        "evidence": {repo_path(AB_RELIABILITY): AB_RELIABILITY},
        "required_by_route": {
            "STOP_REVIEW_UNRELIABLE": {repo_path(AB_RELIABILITY)},
            "STOP_PROTOCOL_INTEGRITY": set(),
        },
    },
    "P5_FINAL_OUTCOME": {
        "routes": {"STOP_REVIEW_INVALID"},
        "evidence": {repo_path(C_RECEIPT): C_RECEIPT},
        "required_by_route": {"STOP_REVIEW_INVALID": {repo_path(C_RECEIPT)}},
    },
}


def expected_stop_parent(stage: str) -> str:
    if stage == "P1_AUTHOR_OUTCOME":
        return verify_author_input_local()["commit"]
    if stage == "P2_CANDIDATE_OUTCOME":
        return verify_author_outcome_success_local()["commit"]
    if stage == "P3_AB_OUTCOME":
        return verify_ab_input_success_local()["commit"]
    if stage == "P4_C_GATE_OUTCOME":
        return verify_ab_outcome_success_local()["commit"]
    if stage == "P5_FINAL_OUTCOME":
        return verify_c_input_success_local()["commit"]
    raise RecoveryValidationError(f"stage has no terminal route: {stage}")


def _require_matching_replayed_error(value: Mapping[str, Any], error: Exception) -> None:
    if (
        value.get("error_type") != type(error).__name__
        or value.get("error_message_sha256")
        != sha256_bytes(str(error).encode("utf-8"))
    ):
        raise RecoveryValidationError("terminal error does not match deterministic replay")


def replay_terminal_stop(value: Mapping[str, Any]) -> None:
    stage = value["stage"]
    route = value["route"]
    if stage == "P4_C_GATE_OUTCOME":
        ab = verify_ab_outcome_success_local()
        try:
            contents, result = build_c_gate_contents(ab)
        except Exception as exc:
            if route != "STOP_PROTOCOL_INTEGRITY":
                raise RecoveryValidationError("terminal route differs from C-gate replay") from exc
            _require_matching_replayed_error(value, exc)
            return
        if (
            route != "STOP_REVIEW_UNRELIABLE"
            or result.get("terminal") is not True
            or result.get("status") != route
            or contents.get(TERMINAL_STOP) != json_bytes(value)
        ):
            raise RecoveryValidationError("terminal C-gate route does not reproduce")
        compare_contents(contents)
        return

    builders: dict[str, Any] = {
        "P1_AUTHOR_OUTCOME": lambda: build_author_outcome_contents(
            verify_p0_local(), verify_author_input_local()
        ),
        "P2_CANDIDATE_OUTCOME": lambda: build_candidate_contents(
            verify_author_outcome_success_local()
        ),
        "P3_AB_OUTCOME": lambda: build_ab_outcome_contents(
            verify_ab_input_success_local()
        ),
        "P5_FINAL_OUTCOME": lambda: build_final_outcome_contents(
            verify_c_input_success_local()
        ),
    }
    builder = builders.get(stage)
    if builder is None:
        raise RecoveryValidationError("terminal stage has no deterministic replay")
    try:
        builder()
    except Exception as exc:
        _require_matching_replayed_error(value, exc)
        return
    raise RecoveryValidationError("terminal route is fake because deterministic replay passed")


def verify_terminal_stop(expected_stage: str | None = None) -> dict[str, Any]:
    value = parse_json_bytes(TERMINAL_STOP.read_bytes(), "O4R terminal stop")
    expected_fields = {
        "name",
        "attempt_id",
        "stage",
        "route",
        "parent_commit",
        "error_type",
        "error_message_sha256",
        "evidence_artifacts",
        "replacement_or_retry_allowed",
        "optimizer_steps",
        "training_authorized",
    }
    if set(value) != expected_fields:
        raise RecoveryValidationError("terminal stop field set mismatch")
    stage = value.get("stage")
    if expected_stage is not None and stage != expected_stage:
        raise RecoveryValidationError("terminal stop belongs to another stage")
    spec = STOP_STAGE_SPECS.get(stage)
    route = value.get("route")
    if (
        spec is None
        or route not in spec["routes"]
        or value.get("name") != "step3o_o4r_terminal_stop_v1"
        or value.get("attempt_id") != ATTEMPT_ID
        or not isinstance(value.get("error_type"), str)
        or not value["error_type"]
        or not isinstance(value.get("error_message_sha256"), str)
        or not HEX64.fullmatch(value["error_message_sha256"])
        or value.get("replacement_or_retry_allowed") is not False
        or value.get("optimizer_steps") != 0
        or value.get("training_authorized") is not False
    ):
        raise RecoveryValidationError("terminal stop identity, route, or boundary mismatch")
    parent = expected_stop_parent(stage)
    if value.get("parent_commit") != parent:
        raise RecoveryValidationError("terminal stop parent mismatch")
    evidence_rows = value.get("evidence_artifacts")
    if not isinstance(evidence_rows, list):
        raise RecoveryValidationError("terminal evidence must be a list")
    evidence_paths: set[str] = set()
    same_commit_paths: list[Path] = []
    commit = creation_commit(TERMINAL_STOP)
    if commit_parent(commit) != parent:
        raise RecoveryValidationError("terminal commit parent mismatch")
    for row in evidence_rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise RecoveryValidationError("terminal evidence row schema mismatch")
        path_text = row["path"]
        path = spec["evidence"].get(path_text)
        if path is None or path_text in evidence_paths:
            raise RecoveryValidationError("terminal evidence is duplicate or unregistered")
        evidence_paths.add(path_text)
        if (
            type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or not HEX64.fullmatch(row["sha256"])
        ):
            raise RecoveryValidationError("terminal evidence hash or length is invalid")
        if not path.is_file() or path.is_symlink():
            raise RecoveryValidationError("terminal evidence artifact is absent")
        data = path.read_bytes()
        if row["bytes"] != len(data) or row["sha256"] != sha256_bytes(data):
            raise RecoveryValidationError("terminal evidence hash or length mismatch")
        if creation_commit(path) == commit:
            same_commit_paths.append(path)
    if not spec["required_by_route"][route] <= evidence_paths:
        raise RecoveryValidationError("terminal stop is missing required evidence")
    paths = (*same_commit_paths, TERMINAL_STOP)
    verified_commit = verify_stage_commit(paths, TERMINAL_STOP, parent)
    replay_terminal_stop(value)
    return {
        "status": route,
        "commit": verified_commit,
        "terminal": True,
        "stage": stage,
        "checks_pass": True,
    }


STAGES = (
    "P0",
    "P1_AUTHOR_INPUT",
    "P1_AUTHOR_OUTCOME",
    "P2_CANDIDATE_OUTCOME",
    "P3_AB_INPUT",
    "P3_AB_OUTCOME",
    "P4_C_GATE_OUTCOME",
    "P5_C_INPUT",
    "P5_FINAL_OUTCOME",
)

STAGE_PREDECESSOR = {
    stage: STAGES[index - 1] if index else None for index, stage in enumerate(STAGES)
}


def local_stage_state(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise RecoveryValidationError(f"unknown O4R stage: {stage}")
    terminal_stage: str | None = None
    if TERMINAL_STOP.exists():
        stop = parse_json_bytes(TERMINAL_STOP.read_bytes(), "O4R terminal stop")
        terminal_stage = stop.get("stage")
        if terminal_stage not in STAGES:
            raise RecoveryValidationError("terminal stop stage is not registered")
        if STAGES.index(stage) > STAGES.index(terminal_stage):
            raise RecoveryValidationError("requested stage occurs after the terminal boundary")
        if stage == terminal_stage:
            return verify_terminal_stop(stage)
    if stage == "P0":
        state = verify_p0_local()
        return {"status": "P0_FROZEN", "commit": state["commit"], "terminal": False}
    if stage == "P1_AUTHOR_INPUT":
        return verify_author_input_local()
    if stage == "P1_AUTHOR_OUTCOME":
        return verify_author_outcome_success_local()
    if stage == "P2_CANDIDATE_OUTCOME":
        return verify_candidate_success_local()
    if stage == "P3_AB_INPUT":
        return verify_ab_input_success_local()
    if stage == "P3_AB_OUTCOME":
        return verify_ab_outcome_success_local()
    if stage == "P4_C_GATE_OUTCOME":
        return verify_c_gate_success_local()
    if stage == "P5_C_INPUT":
        return verify_c_input_success_local()
    if stage == "P5_FINAL_OUTCOME":
        return verify_final_success_local()
    raise AssertionError("unreachable stage")


def gh_env() -> dict[str, str]:
    cfg = registry()["remote"]
    token = os.environ.get(cfg["github_token_env"], "").strip()
    if not token:
        raise RecoveryValidationError(
            f"missing GitHub token environment variable {cfg['github_token_env']}"
        )
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    return env


def gh_json(
    args: Sequence[str],
    *,
    input_value: Mapping[str, Any] | None = None,
    allow_not_found: bool = False,
) -> dict[str, Any] | None:
    command = ["gh", *args]
    payload = None if input_value is None else json.dumps(input_value)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=gh_env(),
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        if allow_not_found and ("HTTP 404" in result.stderr or "Not Found" in result.stderr):
            return None
        raise RecoveryValidationError(
            f"GitHub command failed; stderr_sha256={sha256_bytes(result.stderr.encode('utf-8'))}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecoveryValidationError("GitHub command returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RecoveryValidationError("GitHub command did not return a JSON object")
    return value


def verify_gh_version() -> None:
    expected = registry()["remote"]["gh_cli_version"]
    result = subprocess.run(
        ["gh", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.startswith(f"gh version {expected} "):
        raise RecoveryValidationError("GitHub CLI version differs from O4R registry")


def stage_tag(stage: str) -> str:
    value = registry()["stage_tags"].get(stage)
    if not isinstance(value, str) or not value:
        raise RecoveryValidationError(f"missing fixed tag for stage {stage}")
    return value


def canonical_remote_tip() -> str:
    remote = registry()["remote"]
    url = run_git("remote", "get-url", remote["name"]).stdout.strip()
    if url != remote["url"]:
        raise RecoveryValidationError("canonical remote URL mismatch")
    run_git("fetch", remote["name"], remote["branch"])
    value = run_git("rev-parse", remote["upstream"]).stdout.strip()
    if not FULL_SHA.fullmatch(value):
        raise RecoveryValidationError("canonical remote tip is not a full SHA")
    return value


def require_remote_tip(commit: str) -> None:
    if canonical_remote_tip() != commit:
        raise RecoveryValidationError("local stage commit differs from canonical remote tip")


def require_remote_ancestor(commit: str) -> None:
    tip = canonical_remote_tip()
    if run_git("merge-base", "--is-ancestor", commit, tip, check=False).returncode != 0:
        raise RecoveryValidationError("historical stage commit is not on the canonical branch")


def verify_attestation(tag: str, commit: str) -> dict[str, Any]:
    verify_gh_version()
    repo = registry()["remote"]["github_repository"]
    result = gh_json(
        ["release", "verify", tag, "--repo", repo, "--format", "json"]
    )
    assert result is not None
    verification = result.get("verificationResult")
    if not isinstance(verification, dict):
        raise RecoveryValidationError("release attestation lacks verification result")
    statement = verification.get("statement")
    if not isinstance(statement, dict):
        raise RecoveryValidationError("release attestation lacks statement")
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or len(subjects) != 1:
        raise RecoveryValidationError("release attestation subject mismatch")
    subject = subjects[0]
    digest = subject.get("digest") if isinstance(subject, dict) else None
    predicate = statement.get("predicate")
    if (
        not isinstance(digest, dict)
        or digest.get("sha1") != commit
        or not isinstance(predicate, dict)
        or predicate.get("repository") != repo
        or predicate.get("repositoryId") != str(registry()["remote"]["github_repository_id"])
        or predicate.get("tag") != tag
    ):
        raise RecoveryValidationError("release attestation identity or digest mismatch")
    return {
        "tag": tag,
        "commit": commit,
        "verification_output_sha256": sha256_bytes(json_bytes(result)),
    }


def verify_stage_lock(
    stage: str,
    commit: str | None = None,
    *,
    _seen: set[str] | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise RecoveryValidationError(f"unknown O4R stage: {stage}")
    seen = set() if _seen is None else _seen
    if stage in seen:
        raise RecoveryValidationError("stage lock predecessor cycle detected")
    seen.add(stage)
    state = local_stage_state(stage)
    expected_commit = state["commit"]
    if commit is not None and commit != expected_commit:
        raise RecoveryValidationError("requested lock commit differs from local stage")
    require_remote_ancestor(expected_commit)
    predecessor = STAGE_PREDECESSOR[stage]
    predecessor_lock = None
    if predecessor is not None:
        parent = commit_parent(expected_commit)
        predecessor_state = local_stage_state(predecessor)
        if predecessor_state["commit"] != parent:
            raise RecoveryValidationError("stage commit skips its registered predecessor")
        predecessor_lock = verify_stage_lock(
            predecessor, parent, _seen=seen
        )
    remote = registry()["remote"]
    repo = remote["github_repository"]
    tag = stage_tag(stage)
    release = gh_json(
        ["api", f"repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}"],
    )
    assert release is not None
    if (
        release.get("tag_name") != tag
        or release.get("target_commitish") != expected_commit
        or release.get("immutable") is not True
        or release.get("draft") is not False
        or release.get("prerelease") is not True
    ):
        raise RecoveryValidationError("immutable release metadata mismatch")
    ref = gh_json(
        ["api", f"repos/{repo}/git/ref/tags/{urllib.parse.quote(tag, safe='')}"],
    )
    assert ref is not None
    target = ref.get("object")
    if (
        ref.get("ref") != f"refs/tags/{tag}"
        or not isinstance(target, dict)
        or target.get("type") != "commit"
        or target.get("sha") != expected_commit
    ):
        raise RecoveryValidationError("immutable release tag target mismatch")
    attestation = verify_attestation(tag, expected_commit)
    return {
        "stage": stage,
        "status": state["status"],
        "tag": tag,
        "commit": expected_commit,
        "release_id": release.get("id"),
        "published_at": release.get("published_at"),
        "immutable": True,
        "attestation": attestation,
        "predecessor_lock": predecessor_lock,
    }


def create_stage_lock(stage: str) -> dict[str, Any]:
    state = local_stage_state(stage)
    commit = state["commit"]
    require_head(commit)
    require_clean()
    predecessor = STAGE_PREDECESSOR[stage]
    if predecessor is not None:
        parent = commit_parent(commit)
        predecessor_state = local_stage_state(predecessor)
        if predecessor_state["commit"] != parent:
            raise RecoveryValidationError("stage commit skips its registered predecessor")
        require_stage_lock(predecessor, parent)
    require_remote_tip(commit)
    remote = registry()["remote"]
    repo = remote["github_repository"]
    enabled = gh_json(["api", f"repos/{repo}/immutable-releases"])
    if enabled is None or enabled.get("enabled") is not True:
        raise RecoveryValidationError("repository immutable releases are not enabled")
    tag = stage_tag(stage)
    existing = gh_json(
        ["api", f"repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}"],
        allow_not_found=True,
    )
    if existing is None:
        payload = {
            "tag_name": tag,
            "target_commitish": commit,
            "name": f"Step3O-O4R immutable lock: {stage}",
            "body": (
                f"attempt: {ATTEMPT_ID}\n"
                f"stage: {stage}\n"
                f"commit: {commit}\n"
                "training_authorized: false\n"
            ),
            "draft": False,
            "prerelease": True,
            "make_latest": "false",
        }
        created = gh_json(
            ["api", "--method", "POST", f"repos/{repo}/releases", "--input", "-"],
            input_value=payload,
        )
        if created is None:
            raise RecoveryValidationError("release creation returned no result")
    return verify_stage_lock(stage, commit)


def require_stage_lock(stage: str, commit: str) -> None:
    result = verify_stage_lock(stage, commit)
    if result["commit"] != commit:
        raise RecoveryValidationError("preceding stage lock verification mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize-p0")
    sub.add_parser("verify-p0")
    author_input = sub.add_parser("ingest-author-input")
    author_input.add_argument("--incoming-dir", type=Path, required=True)
    sub.add_parser("verify-author-input")
    sub.add_parser("freeze-author-outcome")
    sub.add_parser("verify-author-outcome")
    sub.add_parser("freeze-candidates")
    sub.add_parser("verify-candidates")
    ab_input = sub.add_parser("ingest-ab-input")
    ab_input.add_argument("--incoming-dir", type=Path, required=True)
    sub.add_parser("verify-ab-input")
    sub.add_parser("freeze-ab-outcome")
    sub.add_parser("verify-ab-outcome")
    sub.add_parser("freeze-c-gate")
    sub.add_parser("verify-c-gate")
    c_input = sub.add_parser("ingest-c-input")
    c_input.add_argument("--incoming-dir", type=Path, required=True)
    sub.add_parser("verify-c-input")
    sub.add_parser("freeze-final-outcome")
    sub.add_parser("verify-final")
    sub.add_parser("verify-terminal-stop")
    lock = sub.add_parser("lock-head")
    lock.add_argument("--stage", choices=STAGES, required=True)
    verify_lock = sub.add_parser("verify-lock")
    verify_lock.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()

    if args.command == "materialize-p0":
        result = materialize_p0()
    elif args.command == "verify-p0":
        result = verify_p0_local()
        result = {"status": "P0_FROZEN", "commit": result["commit"]}
    elif args.command == "ingest-author-input":
        p0 = verify_p0_local()
        require_stage_lock("P0", p0["commit"])
        result = ingest_author_inputs(args.incoming_dir.resolve())
    elif args.command == "verify-author-input":
        result = verify_author_input_local()
    elif args.command == "freeze-author-outcome":
        input_state = verify_author_input_local()
        require_stage_lock("P1_AUTHOR_INPUT", input_state["commit"])
        result = freeze_authors()
    elif args.command == "verify-author-outcome":
        result = verify_authors_local()
    elif args.command == "freeze-candidates":
        authors_state = verify_authors_local()
        require_stage_lock("P1_AUTHOR_OUTCOME", authors_state["commit"])
        result = freeze_candidates()
    elif args.command == "verify-candidates":
        result = verify_candidates_local()
    elif args.command == "ingest-ab-input":
        candidates_state = verify_candidates_local()
        require_stage_lock("P2_CANDIDATE_OUTCOME", candidates_state["commit"])
        result = ingest_ab_inputs(args.incoming_dir.resolve())
    elif args.command == "verify-ab-input":
        result = verify_ab_input_local()
    elif args.command == "freeze-ab-outcome":
        input_state = verify_ab_input_local()
        require_stage_lock("P3_AB_INPUT", input_state["commit"])
        result = freeze_ab_reviews()
    elif args.command == "verify-ab-outcome":
        result = verify_ab_local()
    elif args.command == "freeze-c-gate":
        ab_state = verify_ab_local()
        require_stage_lock("P3_AB_OUTCOME", ab_state["commit"])
        result = freeze_c_gate()
    elif args.command == "verify-c-gate":
        result = verify_c_gate_local()
    elif args.command == "ingest-c-input":
        gate_state = verify_c_gate_local()
        require_stage_lock("P4_C_GATE_OUTCOME", gate_state["commit"])
        result = ingest_c_inputs(args.incoming_dir.resolve())
    elif args.command == "verify-c-input":
        result = verify_c_input_local()
    elif args.command == "freeze-final-outcome":
        input_state = verify_c_input_local()
        require_stage_lock("P5_C_INPUT", input_state["commit"])
        result = freeze_final()
    elif args.command == "verify-final":
        result = verify_final_local()
    elif args.command == "verify-terminal-stop":
        result = verify_terminal_stop()
    elif args.command == "lock-head":
        result = create_stage_lock(args.stage)
    elif args.command == "verify-lock":
        result = verify_stage_lock(args.stage)
    else:
        raise AssertionError("unreachable command")
    print(original_pool.json_text(result), end="")


if __name__ == "__main__":
    main()
