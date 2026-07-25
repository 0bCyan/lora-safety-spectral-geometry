#!/usr/bin/env python3
"""Append-only Step3O-O4 authoring, review, and final-route stages.

This file is part of the prospective P0 protocol. It may not be changed after
an O4 author or reviewer output is opened. External session outputs are first
hashed into an immutable receipt, then validated and copied byte-for-byte into
the registered repository path. A validation failure after receipt creation
consumes attempt 003 and writes the registered terminal stop artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import step3o_o4_qualification as qualification
import step3o_o4_training_pool as pool


ROOT = pool.ROOT
BASE = pool.BASE
AUTHORS = BASE / "authors"
QUALIFICATION_PACKETS = BASE / "packets/qualification"
REVIEWS = BASE / "reviews"
PROVENANCE = BASE / "provenance"
DIAGNOSTICS = BASE / "diagnostics"
REPORTS = BASE / "reports"

AUTHOR_A_RESULT = AUTHORS / "step3o_o4_author_a.jsonl"
AUTHOR_B_RESULT = AUTHORS / "step3o_o4_author_b.jsonl"
AUTHOR_A_PROVENANCE = PROVENANCE / "step3o_o4_author_a_provenance.json"
AUTHOR_B_PROVENANCE = PROVENANCE / "step3o_o4_author_b_provenance.json"
AUTHOR_RECEIPT = REPORTS / "step3o_o4_p2_author_input_receipt.json"
AUTHOR_MANIFEST = REPORTS / "step3o_o4_p2_author_freeze_manifest.json"
AUTHOR_REPORT = REPORTS / "Step3O_O4_P2_author_freeze_report.md"

CANDIDATE_KEY = QUALIFICATION_PACKETS / "step3o_o4_candidate_hidden_key.jsonl"
REVIEWER_A_PACKET = QUALIFICATION_PACKETS / "step3o_o4_reviewer_a_packet.jsonl"
REVIEWER_B_PACKET = QUALIFICATION_PACKETS / "step3o_o4_reviewer_b_packet.jsonl"
REVIEWER_A_TEMPLATE = QUALIFICATION_PACKETS / "step3o_o4_reviewer_a_template.jsonl"
REVIEWER_B_TEMPLATE = QUALIFICATION_PACKETS / "step3o_o4_reviewer_b_template.jsonl"
REVIEWER_A_DUPLICATES = QUALIFICATION_PACKETS / "step3o_o4_reviewer_a_duplicates.json"
REVIEWER_B_DUPLICATES = QUALIFICATION_PACKETS / "step3o_o4_reviewer_b_duplicates.json"
CANDIDATE_MANIFEST = REPORTS / "step3o_o4_p3_candidate_packet_manifest.json"
CANDIDATE_REPORT = REPORTS / "Step3O_O4_P3_candidate_packet_report.md"

REVIEWER_A_RESULT = REVIEWS / "step3o_o4_reviewer_a.jsonl"
REVIEWER_B_RESULT = REVIEWS / "step3o_o4_reviewer_b.jsonl"
REVIEWER_A_PROVENANCE = PROVENANCE / "step3o_o4_reviewer_a_provenance.json"
REVIEWER_B_PROVENANCE = PROVENANCE / "step3o_o4_reviewer_b_provenance.json"
AB_RECEIPT = REPORTS / "step3o_o4_ab_review_input_receipt.json"
AB_MANIFEST = REPORTS / "step3o_o4_ab_review_freeze_manifest.json"

AB_RELIABILITY = DIAGNOSTICS / "step3o_o4_ab_reliability.json"
REVIEWER_C_PACKET = QUALIFICATION_PACKETS / "step3o_o4_reviewer_c_packet.jsonl"
REVIEWER_C_TEMPLATE = QUALIFICATION_PACKETS / "step3o_o4_reviewer_c_template.jsonl"
C_GATE_MANIFEST = REPORTS / "step3o_o4_c_gate_manifest.json"
C_GATE_REPORT = REPORTS / "Step3O_O4_C_gate_report.md"

REVIEWER_C_RESULT = REVIEWS / "step3o_o4_reviewer_c.jsonl"
REVIEWER_C_PROVENANCE = PROVENANCE / "step3o_o4_reviewer_c_provenance.json"
C_RECEIPT = REPORTS / "step3o_o4_c_review_input_receipt.json"
FINAL_LABELS = REVIEWS / "step3o_o4_final_labels.jsonl"
QUALIFIED_POOL = BASE / "machine_rubric_qualified_candidate_pool.jsonl"
QUALIFIED_POOL_KEY = BASE / "machine_rubric_qualified_candidate_pool_key.csv"
FINAL_AGREEMENT = DIAGNOSTICS / "step3o_o4_final_review_agreement.json"
FINAL_YIELD = DIAGNOSTICS / "step3o_o4_final_yield.json"
ROUTE_DECISION = REPORTS / "step3o_o4_route_decision.json"
FINAL_MANIFEST = REPORTS / "step3o_o4_final_manifest.json"
FINAL_REPORT = REPORTS / "Step3O_O4_final_route_report.md"
ATTEMPT_STOP = REPORTS / "step3o_o4_attempt_003_terminal_stop.json"


AUTHOR_STAGE_PATHS = (
    AUTHOR_RECEIPT,
    AUTHOR_A_RESULT,
    AUTHOR_B_RESULT,
    AUTHOR_A_PROVENANCE,
    AUTHOR_B_PROVENANCE,
    AUTHOR_MANIFEST,
    AUTHOR_REPORT,
)
CANDIDATE_STAGE_PATHS = (
    CANDIDATE_KEY,
    REVIEWER_A_PACKET,
    REVIEWER_B_PACKET,
    REVIEWER_A_TEMPLATE,
    REVIEWER_B_TEMPLATE,
    REVIEWER_A_DUPLICATES,
    REVIEWER_B_DUPLICATES,
    CANDIDATE_MANIFEST,
    CANDIDATE_REPORT,
)
AB_STAGE_PATHS = (
    AB_RECEIPT,
    REVIEWER_A_RESULT,
    REVIEWER_B_RESULT,
    REVIEWER_A_PROVENANCE,
    REVIEWER_B_PROVENANCE,
    AB_MANIFEST,
)
C_GATE_PASS_PATHS = (
    AB_RELIABILITY,
    REVIEWER_C_PACKET,
    REVIEWER_C_TEMPLATE,
    C_GATE_MANIFEST,
    C_GATE_REPORT,
)
C_GATE_STOP_PATHS = (
    AB_RELIABILITY,
    ROUTE_DECISION,
    C_GATE_MANIFEST,
    C_GATE_REPORT,
)
FINAL_STAGE_PATHS = (
    C_RECEIPT,
    REVIEWER_C_RESULT,
    REVIEWER_C_PROVENANCE,
    FINAL_LABELS,
    QUALIFIED_POOL,
    QUALIFIED_POOL_KEY,
    FINAL_AGREEMENT,
    FINAL_YIELD,
    ROUTE_DECISION,
    FINAL_MANIFEST,
    FINAL_REPORT,
)

STOP_STAGE_SPECS = {
    "P2_AUTHORING": {
        "route": "STOP_AUTHORING",
        "allowed_paths": AUTHOR_STAGE_PATHS,
        "required_receipt": AUTHOR_RECEIPT,
    },
    "P3_CANDIDATE_PACKETS": {
        "route": "STOP_PROTOCOL_INTEGRITY",
        "allowed_paths": CANDIDATE_STAGE_PATHS,
        "required_receipt": None,
    },
    "P3_AB_REVIEW": {
        "route": "STOP_REVIEW_INVALID",
        "allowed_paths": AB_STAGE_PATHS,
        "required_receipt": AB_RECEIPT,
    },
    "P3_AB_RELIABILITY": {
        "route": "STOP_PROTOCOL_INTEGRITY",
        "allowed_paths": tuple(dict.fromkeys((*C_GATE_PASS_PATHS, *C_GATE_STOP_PATHS))),
        "required_receipt": None,
    },
    "P3_C_REVIEW": {
        "route": "STOP_REVIEW_INVALID",
        "allowed_paths": FINAL_STAGE_PATHS,
        "required_receipt": C_RECEIPT,
    },
}


class StageValidationError(ValueError):
    """A fail-closed O4 stage error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return pool.json_text(value).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return pool.jsonl_text(rows).encode("utf-8")


def csv_bytes(rows: Iterable[dict[str, Any]], fields: list[str]) -> bytes:
    return pool.csv_text(rows, fields).encode("utf-8")


def artifact(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": pool.repo_path(path),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def read_exact_input(path: str | Path) -> bytes:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise StageValidationError(f"incoming file is absent: {candidate.name}")
    try:
        candidate.relative_to(BASE.resolve())
    except ValueError:
        pass
    else:
        raise StageValidationError("incoming files must be outside the frozen O4 result tree")
    return candidate.read_bytes()


def write_frozen_bytes(path: Path, data: bytes) -> None:
    text = qualification.decode_utf8_lf(data, context=pool.repo_path(path))
    pool.write_frozen(path, text)


def write_stage(contents: Mapping[Path, bytes], ordered_paths: Sequence[Path]) -> None:
    if set(contents) != set(ordered_paths):
        raise StageValidationError("stage content path set differs from the registered path set")
    for path in ordered_paths:
        write_frozen_bytes(path, contents[path])


def ensure_pristine(paths: Sequence[Path]) -> None:
    existing = [pool.repo_path(path) for path in paths if path.exists()]
    if existing:
        raise StageValidationError(f"registered stage path already exists: {existing}")
    if ATTEMPT_STOP.exists():
        raise StageValidationError("attempt 003 already has a terminal stop artifact")


def require_parent_head(parent: str) -> None:
    if pool.git_head() != parent:
        raise StageValidationError(f"stage must start directly on parent commit {parent}")
    try:
        pool.require_locked_canonical_tip(pool.registry(), parent)
    except ValueError as exc:
        raise StageValidationError("registered parent is not the locked canonical remote tip") from exc


def changed_paths(commit: str) -> set[str]:
    return pool.commit_changed_paths(commit)


def verify_stage_commit(
    paths: Sequence[Path], manifest: Path, parent: str, *, require_pushed: bool = True
) -> str:
    try:
        return pool.verify_exact_creation_commit(
            paths,
            manifest,
            parent,
            reg=pool.registry(),
            require_pushed=require_pushed,
        )
    except ValueError as exc:
        raise StageValidationError(str(exc)) from exc


def parse_json_bytes(data: bytes, context: str) -> dict[str, Any]:
    text = qualification.decode_utf8_lf(data, context=context)
    return qualification.parse_json(text, context=context)


def parse_packet(path: Path, fields: set[str]) -> list[dict[str, Any]]:
    rows = pool.read_jsonl(path)
    if any(set(row) != fields for row in rows):
        raise StageValidationError(f"packet schema mismatch: {pool.repo_path(path)}")
    return rows


def p1_state() -> dict[str, Any]:
    result = pool.verify_prompts()
    return {
        "protocol_commit": result["protocol_commit"],
        "beacon_round": result["beacon_round"],
        "beacon_randomness": result["beacon_randomness"],
        "p1_commit": result["prompt_freeze_creation_commit"],
    }


def expected_input_hashes(packet: Path, rubric: Path) -> dict[str, str]:
    return qualification.hash_inputs(
        {
            pool.repo_path(packet): packet.read_bytes(),
            pool.repo_path(rubric): rubric.read_bytes(),
        }
    )


def validate_provenance_bytes(
    data: bytes,
    *,
    role: str,
    stage: str,
    packet: Path,
    rubric: Path,
    row_count: int,
) -> dict[str, Any]:
    text = qualification.decode_utf8_lf(data, context=f"{role} provenance")
    return qualification.validate_provenance_json(
        text,
        expected_role=role,
        expected_stage=stage,
        expected_attempt_id=qualification.ATTEMPT_ID,
        expected_input_hashes=expected_input_hashes(packet, rubric),
        expected_row_count=row_count,
    )


def author_packet_rows(role: str) -> list[dict[str, Any]]:
    path = pool.AUTHOR_A_PACKET if role == "A" else pool.AUTHOR_B_PACKET
    rows = parse_packet(path, {"opaque_prompt_id", "prompt"})
    if len(rows) != qualification.AUTHOR_ROWS:
        raise StageValidationError("author packet row count differs from 60")
    ids = [row["opaque_prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise StageValidationError("author packet IDs are duplicated")
    return rows


def prompt_rows_from_p1() -> list[dict[str, Any]]:
    packet_a = author_packet_rows("A")
    packet_b = author_packet_rows("B")
    prompts_a = {row["opaque_prompt_id"]: row["prompt"] for row in packet_a}
    prompts_b = {row["opaque_prompt_id"]: row["prompt"] for row in packet_b}
    if prompts_a != prompts_b:
        raise StageValidationError("P1 author packets do not contain the same prompt mapping")
    key = pool.read_csv(pool.AUTHORING_KEY)
    if len(key) != qualification.PROMPT_CLUSTERS:
        raise StageValidationError("P1 authoring key must contain 60 rows")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in key:
        prompt_id = item.get("opaque_prompt_id", "")
        if prompt_id in seen or prompt_id not in prompts_a:
            raise StageValidationError("P1 hidden key prompt IDs are invalid")
        seen.add(prompt_id)
        try:
            severity = int(item["severity_level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StageValidationError("P1 hidden key severity is invalid") from exc
        rows.append(
            {
                "opaque_prompt_id": prompt_id,
                "prompt": prompts_a[prompt_id],
                "assigned_category": item.get("assigned_category", ""),
                "severity_level": severity,
            }
        )
    if seen != set(prompts_a):
        raise StageValidationError("P1 hidden key prompt set is incomplete")
    return rows


def load_author_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids_a = [row["opaque_prompt_id"] for row in author_packet_rows("A")]
    ids_b = [row["opaque_prompt_id"] for row in author_packet_rows("B")]
    rows_a = qualification.validate_author_jsonl_bytes(AUTHOR_A_RESULT.read_bytes(), ids_a)
    rows_b = qualification.validate_author_jsonl_bytes(AUTHOR_B_RESULT.read_bytes(), ids_b)
    qualification.validate_author_pair(rows_a, rows_b)
    return rows_a, rows_b


def build_candidate_package(state: Mapping[str, Any]) -> dict[str, Any]:
    rows_a, rows_b = load_author_results()
    ids_a = [row["opaque_prompt_id"] for row in author_packet_rows("A")]
    ids_b = [row["opaque_prompt_id"] for row in author_packet_rows("B")]
    return qualification.build_candidate_package(
        prompt_rows_from_p1(),
        rows_a,
        rows_b,
        ids_a,
        ids_b,
        protocol_commit=state["protocol_commit"],
        beacon_round=state["beacon_round"],
        beacon_randomness=state["beacon_randomness"],
    )


def review_template(packet: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "opaque_candidate_id": row["opaque_candidate_id"],
            "detail_retention": None,
            "safe_route": "",
            "executable_step": "",
            "prompt_binding": "",
            "naturalness": "",
            "clean_safe": "",
            "short_reason": "",
        }
        for row in packet
    ]


def load_candidate_package(state: Mapping[str, Any]) -> dict[str, Any]:
    package = build_candidate_package(state)
    qualification.validate_candidate_package(
        package,
        protocol_commit=state["protocol_commit"],
        beacon_round=state["beacon_round"],
        beacon_randomness=state["beacon_randomness"],
    )
    expected = {
        CANDIDATE_KEY: jsonl_bytes(package["primary_candidates"]),
        REVIEWER_A_PACKET: jsonl_bytes(package["reviewer_a_packet"]),
        REVIEWER_B_PACKET: jsonl_bytes(package["reviewer_b_packet"]),
        REVIEWER_A_TEMPLATE: jsonl_bytes(review_template(package["reviewer_a_packet"])),
        REVIEWER_B_TEMPLATE: jsonl_bytes(review_template(package["reviewer_b_packet"])),
        REVIEWER_A_DUPLICATES: json_bytes(package["reviewer_a_duplicates"]),
        REVIEWER_B_DUPLICATES: json_bytes(package["reviewer_b_duplicates"]),
    }
    for path, data in expected.items():
        if path.read_bytes() != data:
            raise StageValidationError(f"candidate artifact changed: {pool.repo_path(path)}")
    return package


def reviewer_rows(package: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    path = REVIEWER_A_RESULT if role == "A" else REVIEWER_B_RESULT
    packet = package["reviewer_a_packet" if role == "A" else "reviewer_b_packet"]
    ids = [row["opaque_candidate_id"] for row in packet]
    return qualification.validate_reviewer_jsonl_bytes(path.read_bytes(), ids, role=role)


def receipt(name: str, stage: str, inputs: Mapping[str, bytes]) -> dict[str, Any]:
    return {
        "name": name,
        "attempt_id": qualification.ATTEMPT_ID,
        "stage": stage,
        "inputs": [
            {"role": role, "bytes": len(data), "sha256": sha256_bytes(data)}
            for role, data in inputs.items()
        ],
        "received_once": True,
        "replacement_or_repair_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def stop_artifact(
    *,
    stage: str,
    route: str,
    parent: str,
    error: Exception,
    evidence_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = STOP_STAGE_SPECS.get(stage)
    if (
        spec is None
        or route != spec["route"]
        or route not in qualification.OFFICIAL_ROUTES
        or not route.startswith("STOP_")
    ):
        raise StageValidationError("invalid stage-specific terminal stop route")
    return {
        "name": "step3o_o4_attempt_003_terminal_stop_v1",
        "attempt_id": qualification.ATTEMPT_ID,
        "stage": stage,
        "route": route,
        "parent_commit": parent,
        "error_type": type(error).__name__,
        "error_message_sha256": sha256_bytes(str(error).encode("utf-8")),
        "evidence_artifacts": [dict(item) for item in evidence_artifacts],
        "replacement_or_retry_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def record_stop(
    *, stage: str, route: str, parent: str, error: Exception, receipt_path: Path | None
) -> None:
    if ATTEMPT_STOP.exists():
        return
    spec = STOP_STAGE_SPECS.get(stage)
    if spec is None or route != spec["route"]:
        raise StageValidationError("unregistered terminal stop stage or route")
    required_receipt = spec["required_receipt"]
    if required_receipt != receipt_path:
        raise StageValidationError("terminal stop receipt differs from stage registry")
    evidence_artifacts = [
        artifact(path, path.read_bytes())
        for path in spec["allowed_paths"]
        if path.exists()
    ]
    if required_receipt is not None and required_receipt not in {
        path for path in spec["allowed_paths"] if path.exists()
    }:
        raise StageValidationError("required external-input receipt is absent at stop")
    pool.write_frozen(
        ATTEMPT_STOP,
        pool.json_text(
            stop_artifact(
                stage=stage,
                route=route,
                parent=parent,
                error=error,
                evidence_artifacts=evidence_artifacts,
            )
        ),
    )


def author_report_text(manifest: Mapping[str, Any]) -> str:
    return f"""# Step3O-O4 P2 Author Freeze

Status: PASS. Both context-isolated author files passed the frozen byte, schema,
order, length, provenance, and cross-author non-collapse checks.

- attempt: `{qualification.ATTEMPT_ID}`;
- parent P1 commit: `{manifest['parent_commit']}`;
- Author A rows: `{manifest['author_a_rows']}`;
- Author B rows: `{manifest['author_b_rows']}`;
- distinct author session IDs: `true`;
- repair, replacement, or top-up performed: `false`;
- optimizer steps: `0`;
- training authorized: `false`.

No author identity, source response, category, severity, prior target, method
result, or aggregate result was an allowed author input.
"""


def build_author_contents(
    state: Mapping[str, Any],
    inputs: Mapping[str, bytes],
) -> dict[Path, bytes]:
    packet_a = author_packet_rows("A")
    packet_b = author_packet_rows("B")
    ids_a = [row["opaque_prompt_id"] for row in packet_a]
    ids_b = [row["opaque_prompt_id"] for row in packet_b]
    rows_a = qualification.validate_author_jsonl_bytes(inputs["author_a"], ids_a)
    rows_b = qualification.validate_author_jsonl_bytes(inputs["author_b"], ids_b)
    qualification.validate_author_pair(rows_a, rows_b)
    provenance_a = validate_provenance_bytes(
        inputs["author_a_provenance"],
        role="author_a",
        stage="authoring",
        packet=pool.AUTHOR_A_PACKET,
        rubric=pool.AUTHORING_INSTRUCTIONS,
        row_count=qualification.AUTHOR_ROWS,
    )
    provenance_b = validate_provenance_bytes(
        inputs["author_b_provenance"],
        role="author_b",
        stage="authoring",
        packet=pool.AUTHOR_B_PACKET,
        rubric=pool.AUTHORING_INSTRUCTIONS,
        row_count=qualification.AUTHOR_ROWS,
    )
    agents = qualification.validate_agent_isolation(
        [provenance_a, provenance_b], ["author_a", "author_b"]
    )
    receipt_data = json_bytes(
        receipt(
            "step3o_o4_p2_author_input_receipt_v1",
            "P2_AUTHORING",
            inputs,
        )
    )
    frozen_inputs = {
        AUTHOR_RECEIPT: receipt_data,
        AUTHOR_A_RESULT: inputs["author_a"],
        AUTHOR_B_RESULT: inputs["author_b"],
        AUTHOR_A_PROVENANCE: inputs["author_a_provenance"],
        AUTHOR_B_PROVENANCE: inputs["author_b_provenance"],
    }
    manifest = {
        "name": "step3o_o4_p2_author_freeze_manifest_v1",
        "status": "PASS_AUTHORING_FROZEN",
        "attempt_id": qualification.ATTEMPT_ID,
        "protocol_commit": state["protocol_commit"],
        "parent_commit": state["p1_commit"],
        "p1_manifest_sha256": pool.sha256(pool.PROTOCOL_MANIFEST),
        "author_a_rows": len(rows_a),
        "author_b_rows": len(rows_b),
        "author_agent_ids": agents,
        "author_packet_sha256": {
            "A": pool.sha256(pool.AUTHOR_A_PACKET),
            "B": pool.sha256(pool.AUTHOR_B_PACKET),
        },
        "authoring_rubric_sha256": pool.sha256(pool.AUTHORING_INSTRUCTIONS),
        "artifacts": [artifact(path, data) for path, data in frozen_inputs.items()],
        "same_prompt_normalized_response_collapse": False,
        "replacement_or_repair_performed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = author_report_text(manifest).encode("utf-8")
    return {
        **frozen_inputs,
        AUTHOR_MANIFEST: json_bytes(manifest),
        AUTHOR_REPORT: report,
    }


def freeze_authors(
    author_a: str,
    author_b: str,
    author_a_provenance: str,
    author_b_provenance: str,
) -> dict[str, Any]:
    inputs = {
        "author_a": read_exact_input(author_a),
        "author_b": read_exact_input(author_b),
        "author_a_provenance": read_exact_input(author_a_provenance),
        "author_b_provenance": read_exact_input(author_b_provenance),
    }
    state = p1_state()
    parent = state["p1_commit"]
    require_parent_head(parent)
    ensure_pristine(AUTHOR_STAGE_PATHS)
    receipt_data = json_bytes(
        receipt("step3o_o4_p2_author_input_receipt_v1", "P2_AUTHORING", inputs)
    )
    write_frozen_bytes(AUTHOR_RECEIPT, receipt_data)
    try:
        contents = build_author_contents(state, inputs)
        if contents[AUTHOR_RECEIPT] != receipt_data:
            raise StageValidationError("author receipt reconstruction changed")
        for path in AUTHOR_STAGE_PATHS:
            if path != AUTHOR_RECEIPT:
                write_frozen_bytes(path, contents[path])
    except Exception as exc:
        record_stop(
            stage="P2_AUTHORING",
            route="STOP_AUTHORING",
            parent=parent,
            error=exc,
            receipt_path=AUTHOR_RECEIPT,
        )
        raise
    return parse_json_bytes(contents[AUTHOR_MANIFEST], "author manifest")


def load_author_provenance() -> dict[str, dict[str, Any]]:
    return {
        "author_a": validate_provenance_bytes(
            AUTHOR_A_PROVENANCE.read_bytes(),
            role="author_a",
            stage="authoring",
            packet=pool.AUTHOR_A_PACKET,
            rubric=pool.AUTHORING_INSTRUCTIONS,
            row_count=qualification.AUTHOR_ROWS,
        ),
        "author_b": validate_provenance_bytes(
            AUTHOR_B_PROVENANCE.read_bytes(),
            role="author_b",
            stage="authoring",
            packet=pool.AUTHOR_B_PACKET,
            rubric=pool.AUTHORING_INSTRUCTIONS,
            row_count=qualification.AUTHOR_ROWS,
        ),
    }


def verify_authors() -> dict[str, Any]:
    state = p1_state()
    inputs = {
        "author_a": AUTHOR_A_RESULT.read_bytes(),
        "author_b": AUTHOR_B_RESULT.read_bytes(),
        "author_a_provenance": AUTHOR_A_PROVENANCE.read_bytes(),
        "author_b_provenance": AUTHOR_B_PROVENANCE.read_bytes(),
    }
    expected = build_author_contents(state, inputs)
    for path in AUTHOR_STAGE_PATHS:
        if path.read_bytes() != expected[path]:
            raise StageValidationError(f"author stage byte mismatch: {pool.repo_path(path)}")
    commit = verify_stage_commit(AUTHOR_STAGE_PATHS, AUTHOR_MANIFEST, state["p1_commit"])
    return {"stage": "P2_AUTHORING", "checks_pass": True, "commit": commit}


def candidate_report_text(manifest: Mapping[str, Any]) -> str:
    return f"""# Step3O-O4 P3 Candidate Packet Freeze

Status: PASS. Exactly `{manifest['primary_candidates']}` primary candidates
and `{manifest['hidden_duplicates_per_reviewer']}` hidden duplicates per A/B
reviewer were frozen before any qualification label.

- parent P2 commit: `{manifest['parent_commit']}`;
- A/B visible primary sets: identical;
- A/B packet orders: separately salted and distinct;
- author identity visible to reviewers: `false`;
- duplicate links visible to reviewers: `false`;
- optimizer steps: `0`;
- training authorized: `false`.
"""


def build_candidate_contents(
    state: Mapping[str, Any], parent: str
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    package = build_candidate_package(state)
    qualification.validate_candidate_package(
        package,
        protocol_commit=state["protocol_commit"],
        beacon_round=state["beacon_round"],
        beacon_randomness=state["beacon_randomness"],
    )
    packet_contents = {
        CANDIDATE_KEY: jsonl_bytes(package["primary_candidates"]),
        REVIEWER_A_PACKET: jsonl_bytes(package["reviewer_a_packet"]),
        REVIEWER_B_PACKET: jsonl_bytes(package["reviewer_b_packet"]),
        REVIEWER_A_TEMPLATE: jsonl_bytes(review_template(package["reviewer_a_packet"])),
        REVIEWER_B_TEMPLATE: jsonl_bytes(review_template(package["reviewer_b_packet"])),
        REVIEWER_A_DUPLICATES: json_bytes(package["reviewer_a_duplicates"]),
        REVIEWER_B_DUPLICATES: json_bytes(package["reviewer_b_duplicates"]),
    }
    manifest = {
        "name": "step3o_o4_p3_candidate_packet_manifest_v1",
        "status": "PASS_CANDIDATE_PACKETS_FROZEN_BEFORE_REVIEW",
        "attempt_id": qualification.ATTEMPT_ID,
        "protocol_commit": state["protocol_commit"],
        "parent_commit": parent,
        "author_manifest_sha256": pool.sha256(AUTHOR_MANIFEST),
        "primary_candidates": len(package["primary_candidates"]),
        "hidden_duplicates_per_reviewer": qualification.HIDDEN_DUPLICATES,
        "reviewer_a_rows": len(package["reviewer_a_packet"]),
        "reviewer_b_rows": len(package["reviewer_b_packet"]),
        "qualification_rubric_sha256": pool.sha256(pool.QUALIFICATION_RUBRIC),
        "artifacts": [artifact(path, data) for path, data in packet_contents.items()],
        "reviewer_visible_fields": list(qualification.VISIBLE_CANDIDATE_FIELDS),
        "author_identity_visible": False,
        "category_or_severity_visible": False,
        "duplicate_links_visible": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    contents = {
        **packet_contents,
        CANDIDATE_MANIFEST: json_bytes(manifest),
        CANDIDATE_REPORT: candidate_report_text(manifest).encode("utf-8"),
    }
    return contents, package


def freeze_candidates() -> dict[str, Any]:
    author_state = verify_authors()
    parent = author_state["commit"]
    require_parent_head(parent)
    ensure_pristine(CANDIDATE_STAGE_PATHS)
    state = p1_state()
    try:
        contents, _ = build_candidate_contents(state, parent)
        write_stage(contents, CANDIDATE_STAGE_PATHS)
    except Exception as exc:
        record_stop(
            stage="P3_CANDIDATE_PACKETS",
            route="STOP_PROTOCOL_INTEGRITY",
            parent=parent,
            error=exc,
            receipt_path=None,
        )
        raise
    return parse_json_bytes(contents[CANDIDATE_MANIFEST], "candidate manifest")


def verify_candidates() -> dict[str, Any]:
    author_state = verify_authors()
    state = p1_state()
    expected, _ = build_candidate_contents(state, author_state["commit"])
    for path in CANDIDATE_STAGE_PATHS:
        if path.read_bytes() != expected[path]:
            raise StageValidationError(f"candidate stage byte mismatch: {pool.repo_path(path)}")
    commit = verify_stage_commit(
        CANDIDATE_STAGE_PATHS, CANDIDATE_MANIFEST, author_state["commit"]
    )
    return {"stage": "P3_CANDIDATE_PACKETS", "checks_pass": True, "commit": commit}


def build_ab_contents(
    state: Mapping[str, Any], parent: str, inputs: Mapping[str, bytes]
) -> dict[Path, bytes]:
    package = load_candidate_package(state)
    order_a = [row["opaque_candidate_id"] for row in package["reviewer_a_packet"]]
    order_b = [row["opaque_candidate_id"] for row in package["reviewer_b_packet"]]
    rows_a = qualification.validate_reviewer_jsonl_bytes(
        inputs["reviewer_a"], order_a, role="A"
    )
    rows_b = qualification.validate_reviewer_jsonl_bytes(
        inputs["reviewer_b"], order_b, role="B"
    )
    provenance_a = validate_provenance_bytes(
        inputs["reviewer_a_provenance"],
        role="reviewer_a",
        stage="qualification",
        packet=REVIEWER_A_PACKET,
        rubric=pool.QUALIFICATION_RUBRIC,
        row_count=len(order_a),
    )
    provenance_b = validate_provenance_bytes(
        inputs["reviewer_b_provenance"],
        role="reviewer_b",
        stage="qualification",
        packet=REVIEWER_B_PACKET,
        rubric=pool.QUALIFICATION_RUBRIC,
        row_count=len(order_b),
    )
    author_provenance = load_author_provenance()
    agents = qualification.validate_agent_isolation(
        [
            author_provenance["author_a"],
            author_provenance["author_b"],
            provenance_a,
            provenance_b,
        ],
        ["author_a", "author_b", "reviewer_a", "reviewer_b"],
    )
    receipt_data = json_bytes(
        receipt("step3o_o4_ab_review_input_receipt_v1", "P3_AB_REVIEW", inputs)
    )
    frozen_inputs = {
        AB_RECEIPT: receipt_data,
        REVIEWER_A_RESULT: inputs["reviewer_a"],
        REVIEWER_B_RESULT: inputs["reviewer_b"],
        REVIEWER_A_PROVENANCE: inputs["reviewer_a_provenance"],
        REVIEWER_B_PROVENANCE: inputs["reviewer_b_provenance"],
    }
    manifest = {
        "name": "step3o_o4_ab_review_freeze_manifest_v1",
        "status": "PASS_AB_FILES_FROZEN_BEFORE_COMPARISON",
        "attempt_id": qualification.ATTEMPT_ID,
        "protocol_commit": state["protocol_commit"],
        "parent_commit": parent,
        "candidate_manifest_sha256": pool.sha256(CANDIDATE_MANIFEST),
        "reviewer_a_rows": len(rows_a),
        "reviewer_b_rows": len(rows_b),
        "session_agent_ids": agents,
        "artifacts": [artifact(path, data) for path, data in frozen_inputs.items()],
        "ab_labels_compared": False,
        "hidden_key_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return {**frozen_inputs, AB_MANIFEST: json_bytes(manifest)}


def freeze_ab_reviews(
    reviewer_a: str,
    reviewer_b: str,
    reviewer_a_provenance: str,
    reviewer_b_provenance: str,
) -> dict[str, Any]:
    inputs = {
        "reviewer_a": read_exact_input(reviewer_a),
        "reviewer_b": read_exact_input(reviewer_b),
        "reviewer_a_provenance": read_exact_input(reviewer_a_provenance),
        "reviewer_b_provenance": read_exact_input(reviewer_b_provenance),
    }
    candidate_state = verify_candidates()
    parent = candidate_state["commit"]
    require_parent_head(parent)
    ensure_pristine(AB_STAGE_PATHS)
    receipt_data = json_bytes(
        receipt("step3o_o4_ab_review_input_receipt_v1", "P3_AB_REVIEW", inputs)
    )
    write_frozen_bytes(AB_RECEIPT, receipt_data)
    try:
        contents = build_ab_contents(p1_state(), parent, inputs)
        if contents[AB_RECEIPT] != receipt_data:
            raise StageValidationError("A/B receipt reconstruction changed")
        for path in AB_STAGE_PATHS:
            if path != AB_RECEIPT:
                write_frozen_bytes(path, contents[path])
    except Exception as exc:
        record_stop(
            stage="P3_AB_REVIEW",
            route="STOP_REVIEW_INVALID",
            parent=parent,
            error=exc,
            receipt_path=AB_RECEIPT,
        )
        raise
    return parse_json_bytes(contents[AB_MANIFEST], "A/B manifest")


def load_ab_provenance() -> dict[str, dict[str, Any]]:
    return {
        "reviewer_a": validate_provenance_bytes(
            REVIEWER_A_PROVENANCE.read_bytes(),
            role="reviewer_a",
            stage="qualification",
            packet=REVIEWER_A_PACKET,
            rubric=pool.QUALIFICATION_RUBRIC,
            row_count=qualification.PRIMARY_CANDIDATES + qualification.HIDDEN_DUPLICATES,
        ),
        "reviewer_b": validate_provenance_bytes(
            REVIEWER_B_PROVENANCE.read_bytes(),
            role="reviewer_b",
            stage="qualification",
            packet=REVIEWER_B_PACKET,
            rubric=pool.QUALIFICATION_RUBRIC,
            row_count=qualification.PRIMARY_CANDIDATES + qualification.HIDDEN_DUPLICATES,
        ),
    }


def verify_ab_reviews() -> dict[str, Any]:
    candidate_state = verify_candidates()
    state = p1_state()
    inputs = {
        "reviewer_a": REVIEWER_A_RESULT.read_bytes(),
        "reviewer_b": REVIEWER_B_RESULT.read_bytes(),
        "reviewer_a_provenance": REVIEWER_A_PROVENANCE.read_bytes(),
        "reviewer_b_provenance": REVIEWER_B_PROVENANCE.read_bytes(),
    }
    expected = build_ab_contents(state, candidate_state["commit"], inputs)
    for path in AB_STAGE_PATHS:
        if path.read_bytes() != expected[path]:
            raise StageValidationError(f"A/B stage byte mismatch: {pool.repo_path(path)}")
    commit = verify_stage_commit(AB_STAGE_PATHS, AB_MANIFEST, candidate_state["commit"])
    return {"stage": "P3_AB_REVIEW", "checks_pass": True, "commit": commit}


def route_decision(
    *,
    route: str,
    stage: str,
    protocol_commit: str,
    parent_commit: str,
    reliability_pass: bool,
    target_yield_pass: bool | None,
) -> dict[str, Any]:
    if route not in qualification.OFFICIAL_ROUTES:
        raise StageValidationError("route is not registered")
    return {
        "name": "step3o_o4_route_decision_v1",
        "attempt_id": qualification.ATTEMPT_ID,
        "stage": stage,
        "route": route,
        "protocol_commit": protocol_commit,
        "parent_commit": parent_commit,
        "protocol_integrity_valid": True,
        "prompt_pool_valid": True,
        "authoring_valid": True,
        "review_inputs_valid": True,
        "reliability_pass": reliability_pass,
        "target_yield_pass": target_yield_pass,
        "output_name": "machine-rubric-qualified candidate pool",
        "human_validated": False,
        "heterogeneous_judge_validated": False,
        "training_suitability_validated": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def c_gate_report_text(
    *, route: str | None, reliability: Mapping[str, Any], c_rows: int
) -> str:
    if route == "STOP_REVIEW_UNRELIABLE":
        status = "STOP. The frozen A/B reliability conjunction failed."
        next_action = (
            "Reviewer C is not authorized, the hidden candidate strata remain closed, "
            "and attempt 003 cannot be repaired or rerun."
        )
    else:
        status = "PASS. Every frozen A/B reliability gate passed."
        next_action = (
            f"Reviewer C is authorized for exactly `{c_rows}` every-and-only "
            "A/B candidate-label disagreement rows in the frozen blinded packet."
        )
    failed = [key for key, value in reliability["gates"].items() if not value]
    return f"""# Step3O-O4 A/B Reliability and C Gate

Status: {status}

- primary candidates: `{reliability['primary_rows']}`;
- complete-vector exact agreement: `{reliability['full_vector_exact_agreement']:.6f}`;
- derived qualification raw agreement: `{reliability['candidate_qualified']['raw_agreement']}`;
- derived qualification Gwet AC1: `{reliability['candidate_qualified']['gwet_ac1']}`;
- stratified fixed-pool prompt-reweighting AC1 q0.025: `{reliability['stratified_prompt_reweighting']['q025']}`;
- failed registered gates: `{json.dumps(failed, ensure_ascii=False)}`;
- C packet rows: `{c_rows}`;
- optimizer steps: `0`;
- training authorized: `false`.

{next_action}
"""


def build_c_gate_contents(
    state: Mapping[str, Any], parent: str
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    package = load_candidate_package(state)
    rows_a = reviewer_rows(package, "A")
    rows_b = reviewer_rows(package, "B")
    reliability = qualification.compute_ab_reliability(
        package,
        rows_a,
        rows_b,
        protocol_commit=state["protocol_commit"],
        beacon_round=state["beacon_round"],
        beacon_randomness=state["beacon_randomness"],
    )
    reliability_data = json_bytes(reliability)
    if not reliability["overall_pass"]:
        route = qualification.official_route(
            protocol_integrity_valid=True,
            prompt_pool_valid=True,
            authoring_valid=True,
            review_inputs_valid=True,
            reliability_pass=False,
            target_yield_pass=False,
        )
        decision = route_decision(
            route=route,
            stage="P3_AB_RELIABILITY",
            protocol_commit=state["protocol_commit"],
            parent_commit=parent,
            reliability_pass=False,
            target_yield_pass=None,
        )
        decision_data = json_bytes(decision)
        manifest = {
            "name": "step3o_o4_c_gate_manifest_v1",
            "status": route,
            "attempt_id": qualification.ATTEMPT_ID,
            "protocol_commit": state["protocol_commit"],
            "parent_commit": parent,
            "ab_manifest_sha256": pool.sha256(AB_MANIFEST),
            "reliability_sha256": sha256_bytes(reliability_data),
            "route_decision_sha256": sha256_bytes(decision_data),
            "reviewer_c_authorized": False,
            "reviewer_c_rows": 0,
            "hidden_strata_opened": False,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        contents = {
            AB_RELIABILITY: reliability_data,
            ROUTE_DECISION: decision_data,
            C_GATE_MANIFEST: json_bytes(manifest),
            C_GATE_REPORT: c_gate_report_text(
                route=route, reliability=reliability, c_rows=0
            ).encode("utf-8"),
        }
        return contents, manifest

    c_packet = qualification.build_c_packet(
        package,
        rows_a,
        rows_b,
        protocol_commit=state["protocol_commit"],
        beacon_round=state["beacon_round"],
        beacon_randomness=state["beacon_randomness"],
    )
    c_packet_data = jsonl_bytes(c_packet)
    c_template_data = jsonl_bytes(review_template(c_packet))
    manifest = {
        "name": "step3o_o4_c_gate_manifest_v1",
        "status": "PASS_AB_RELIABILITY_C_PACKET_FROZEN",
        "attempt_id": qualification.ATTEMPT_ID,
        "protocol_commit": state["protocol_commit"],
        "parent_commit": parent,
        "ab_manifest_sha256": pool.sha256(AB_MANIFEST),
        "reliability_sha256": sha256_bytes(reliability_data),
        "reviewer_c_packet_sha256": sha256_bytes(c_packet_data),
        "reviewer_c_template_sha256": sha256_bytes(c_template_data),
        "reviewer_c_authorized": True,
        "reviewer_c_rows": len(c_packet),
        "duplicate_rows_in_c": 0,
        "ab_labels_or_reasons_visible_to_c": False,
        "disagreement_field_markers_visible_to_c": False,
        "hidden_key_visible_to_c": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    contents = {
        AB_RELIABILITY: reliability_data,
        REVIEWER_C_PACKET: c_packet_data,
        REVIEWER_C_TEMPLATE: c_template_data,
        C_GATE_MANIFEST: json_bytes(manifest),
        C_GATE_REPORT: c_gate_report_text(
            route=None, reliability=reliability, c_rows=len(c_packet)
        ).encode("utf-8"),
    }
    return contents, manifest


def freeze_c_gate() -> dict[str, Any]:
    ab_state = verify_ab_reviews()
    parent = ab_state["commit"]
    require_parent_head(parent)
    ensure_pristine(tuple(dict.fromkeys((*C_GATE_PASS_PATHS, *C_GATE_STOP_PATHS))))
    try:
        contents, manifest = build_c_gate_contents(p1_state(), parent)
        paths = (
            C_GATE_PASS_PATHS
            if manifest["status"] == "PASS_AB_RELIABILITY_C_PACKET_FROZEN"
            else C_GATE_STOP_PATHS
        )
        write_stage(contents, paths)
    except Exception as exc:
        record_stop(
            stage="P3_AB_RELIABILITY",
            route="STOP_PROTOCOL_INTEGRITY",
            parent=parent,
            error=exc,
            receipt_path=None,
        )
        raise
    return manifest


def verify_c_gate() -> dict[str, Any]:
    ab_state = verify_ab_reviews()
    contents, expected_manifest = build_c_gate_contents(p1_state(), ab_state["commit"])
    paths = (
        C_GATE_PASS_PATHS
        if expected_manifest["status"] == "PASS_AB_RELIABILITY_C_PACKET_FROZEN"
        else C_GATE_STOP_PATHS
    )
    for path in paths:
        if path.read_bytes() != contents[path]:
            raise StageValidationError(f"C-gate byte mismatch: {pool.repo_path(path)}")
    commit = verify_stage_commit(paths, C_GATE_MANIFEST, ab_state["commit"])
    return {
        "stage": "P3_AB_RELIABILITY",
        "checks_pass": True,
        "commit": commit,
        "status": expected_manifest["status"],
        "reviewer_c_rows": expected_manifest["reviewer_c_rows"],
    }


def c_packet_rows() -> list[dict[str, Any]]:
    text = qualification.decode_utf8_lf(
        REVIEWER_C_PACKET.read_bytes(), context="Reviewer C packet"
    )
    rows = qualification.parse_jsonl(
        text, context="Reviewer C packet", allow_empty=True
    )
    expected_fields = set(qualification.VISIBLE_CANDIDATE_FIELDS)
    if any(set(row) != expected_fields for row in rows):
        raise StageValidationError("Reviewer C packet schema changed")
    return rows


def expected_all_provenance_inputs() -> dict[str, dict[str, str]]:
    return {
        "author_a": expected_input_hashes(pool.AUTHOR_A_PACKET, pool.AUTHORING_INSTRUCTIONS),
        "author_b": expected_input_hashes(pool.AUTHOR_B_PACKET, pool.AUTHORING_INSTRUCTIONS),
        "reviewer_a": expected_input_hashes(REVIEWER_A_PACKET, pool.QUALIFICATION_RUBRIC),
        "reviewer_b": expected_input_hashes(REVIEWER_B_PACKET, pool.QUALIFICATION_RUBRIC),
        "reviewer_c": expected_input_hashes(REVIEWER_C_PACKET, pool.QUALIFICATION_RUBRIC),
    }


def p1_key_by_prompt() -> dict[str, dict[str, str]]:
    rows = pool.read_csv(pool.AUTHORING_KEY)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        prompt_id = row.get("opaque_prompt_id", "")
        if not prompt_id or prompt_id in result:
            raise StageValidationError("P1 key has a missing or duplicate prompt ID")
        result[prompt_id] = row
    return result


def selected_pool_rows(
    package: Mapping[str, Any], yield_summary: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {
        row["opaque_candidate_id"]: row for row in package["primary_candidates"]
    }
    p1_key = p1_key_by_prompt()
    visible: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for selected in yield_summary["selected_machine_rubric_candidate_ids"]:
        candidate = by_id[selected["opaque_candidate_id"]]
        prompt_key = p1_key[candidate["opaque_prompt_id"]]
        visible.append(
            {
                "opaque_prompt_id": candidate["opaque_prompt_id"],
                "opaque_candidate_id": candidate["opaque_candidate_id"],
                "prompt": candidate["prompt"],
                "response": candidate["response"],
            }
        )
        hidden.append(
            {
                "opaque_prompt_id": candidate["opaque_prompt_id"],
                "opaque_candidate_id": candidate["opaque_candidate_id"],
                "sample_id": prompt_key["sample_id"],
                "source_index": prompt_key["source_index"],
                "assigned_category": candidate["assigned_category"],
                "severity_level": candidate["severity_level"],
                "author_role": candidate["author_role"],
                "prompt_sha256": sha256_bytes(candidate["prompt"].encode("utf-8")),
                "response_sha256": sha256_bytes(candidate["response"].encode("utf-8")),
            }
        )
    return visible, hidden


def final_report_text(
    decision: Mapping[str, Any], yield_summary: Mapping[str, Any]
) -> str:
    return f"""# Step3O-O4 Final Route

Official route: `{decision['route']}`.

- machine-rubric-qualified prompts: `{yield_summary['best_of_two']['qualified_prompts']}/60`;
- Author A qualified candidates: `{yield_summary['author_a']['qualified']}/60`;
- Author B qualified candidates: `{yield_summary['author_b']['qualified']}/60`;
- all qualified candidates: `{yield_summary['all_candidates']['qualified']}/120`;
- category retention gates pass: `{str(yield_summary['gates']['all_category_minima_pass']).lower()}`;
- severity retention gates pass: `{str(yield_summary['gates']['all_severity_minima_pass']).lower()}`;
- human validated: `false`;
- heterogeneous-judge validated: `false`;
- training-suitability validated: `false`;
- optimizer steps: `0`;
- training authorized: `false`.

This route concerns only the fixed two-attempt, same-model,
machine-rubric-qualified candidate pool. It is not a behavioral method result,
human construct validation, mechanism evidence, or training authorization.
"""


def build_final_contents(
    state: Mapping[str, Any], parent: str, inputs: Mapping[str, bytes]
) -> dict[Path, bytes]:
    c_packet = c_packet_rows()
    c_ids = [row["opaque_candidate_id"] for row in c_packet]
    c_rows = qualification.validate_reviewer_jsonl_bytes(
        inputs["reviewer_c"], c_ids, role="C"
    )
    provenance_c = validate_provenance_bytes(
        inputs["reviewer_c_provenance"],
        role="reviewer_c",
        stage="qualification",
        packet=REVIEWER_C_PACKET,
        rubric=pool.QUALIFICATION_RUBRIC,
        row_count=len(c_ids),
    )
    package = load_candidate_package(state)
    rows_a = reviewer_rows(package, "A")
    rows_b = reviewer_rows(package, "B")
    provenance_records = {
        **load_author_provenance(),
        **load_ab_provenance(),
        "reviewer_c": provenance_c,
    }
    result = qualification.analyze_validated_qualification(
        package,
        rows_a,
        rows_b,
        c_packet,
        c_rows,
        provenance_records,
        expected_all_provenance_inputs(),
        protocol_commit=state["protocol_commit"],
        beacon_round=state["beacon_round"],
        beacon_randomness=state["beacon_randomness"],
    )
    visible_pool, hidden_pool = selected_pool_rows(package, result["yield"])
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
    receipt_data = json_bytes(
        receipt("step3o_o4_c_review_input_receipt_v1", "P3_C_REVIEW", inputs)
    )
    decision = route_decision(
        route=result["route"],
        stage="P3_FINAL",
        protocol_commit=state["protocol_commit"],
        parent_commit=parent,
        reliability_pass=True,
        target_yield_pass=result["yield"]["overall_pass"],
    )
    result_contents = {
        C_RECEIPT: receipt_data,
        REVIEWER_C_RESULT: inputs["reviewer_c"],
        REVIEWER_C_PROVENANCE: inputs["reviewer_c_provenance"],
        FINAL_LABELS: jsonl_bytes(result["final_labels"]),
        QUALIFIED_POOL: jsonl_bytes(visible_pool),
        QUALIFIED_POOL_KEY: csv_bytes(hidden_pool, hidden_fields),
        FINAL_AGREEMENT: json_bytes(result["reliability"]),
        FINAL_YIELD: json_bytes(result["yield"]),
        ROUTE_DECISION: json_bytes(decision),
        FINAL_REPORT: final_report_text(decision, result["yield"]).encode("utf-8"),
    }
    manifest = {
        "name": "step3o_o4_final_manifest_v1",
        "status": result["route"],
        "attempt_id": qualification.ATTEMPT_ID,
        "protocol_commit": state["protocol_commit"],
        "parent_commit": parent,
        "c_gate_manifest_sha256": pool.sha256(C_GATE_MANIFEST),
        "session_agent_ids": result["session_agent_ids"],
        "reviewer_c_rows": len(c_rows),
        "qualified_prompt_rows": len(visible_pool),
        "artifacts": [artifact(path, data) for path, data in result_contents.items()],
        "human_validated": False,
        "heterogeneous_judge_validated": False,
        "training_suitability_validated": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return {**result_contents, FINAL_MANIFEST: json_bytes(manifest)}


def freeze_final(reviewer_c: str, reviewer_c_provenance: str) -> dict[str, Any]:
    inputs = {
        "reviewer_c": read_exact_input(reviewer_c),
        "reviewer_c_provenance": read_exact_input(reviewer_c_provenance),
    }
    gate_state = verify_c_gate()
    if gate_state["status"] != "PASS_AB_RELIABILITY_C_PACKET_FROZEN":
        raise StageValidationError("A/B reliability stop does not authorize Reviewer C")
    parent = gate_state["commit"]
    require_parent_head(parent)
    ensure_pristine(FINAL_STAGE_PATHS)
    receipt_data = json_bytes(
        receipt("step3o_o4_c_review_input_receipt_v1", "P3_C_REVIEW", inputs)
    )
    write_frozen_bytes(C_RECEIPT, receipt_data)
    try:
        contents = build_final_contents(p1_state(), parent, inputs)
        if contents[C_RECEIPT] != receipt_data:
            raise StageValidationError("Reviewer C receipt reconstruction changed")
        for path in FINAL_STAGE_PATHS:
            if path != C_RECEIPT:
                write_frozen_bytes(path, contents[path])
    except Exception as exc:
        record_stop(
            stage="P3_C_REVIEW",
            route="STOP_REVIEW_INVALID",
            parent=parent,
            error=exc,
            receipt_path=C_RECEIPT,
        )
        raise
    return parse_json_bytes(contents[FINAL_MANIFEST], "final manifest")


def verify_final() -> dict[str, Any]:
    gate_state = verify_c_gate()
    if gate_state["status"] != "PASS_AB_RELIABILITY_C_PACKET_FROZEN":
        raise StageValidationError("attempt stopped at A/B reliability")
    inputs = {
        "reviewer_c": REVIEWER_C_RESULT.read_bytes(),
        "reviewer_c_provenance": REVIEWER_C_PROVENANCE.read_bytes(),
    }
    expected = build_final_contents(p1_state(), gate_state["commit"], inputs)
    for path in FINAL_STAGE_PATHS:
        if path.read_bytes() != expected[path]:
            raise StageValidationError(f"final stage byte mismatch: {pool.repo_path(path)}")
    commit = verify_stage_commit(FINAL_STAGE_PATHS, FINAL_MANIFEST, gate_state["commit"])
    decision = parse_json_bytes(ROUTE_DECISION.read_bytes(), "route decision")
    return {
        "stage": "P3_FINAL",
        "checks_pass": True,
        "commit": commit,
        "route": decision["route"],
        "training_authorized": False,
    }


def expected_stop_parent(stage_name: str) -> str:
    if stage_name == "P2_AUTHORING":
        return p1_state()["p1_commit"]
    if stage_name == "P3_CANDIDATE_PACKETS":
        return verify_authors()["commit"]
    if stage_name == "P3_AB_REVIEW":
        return verify_candidates()["commit"]
    if stage_name == "P3_AB_RELIABILITY":
        return verify_ab_reviews()["commit"]
    if stage_name == "P3_C_REVIEW":
        gate = verify_c_gate()
        if gate["status"] != "PASS_AB_RELIABILITY_C_PACKET_FROZEN":
            raise StageValidationError("C-review stop requires a passing C-gate parent")
        return gate["commit"]
    raise StageValidationError("unknown terminal-stop stage")


def verify_terminal_stop() -> dict[str, Any]:
    value = parse_json_bytes(ATTEMPT_STOP.read_bytes(), "terminal stop")
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
        raise StageValidationError("terminal stop artifact schema mismatch")
    stage_name = value["stage"]
    spec = STOP_STAGE_SPECS.get(stage_name)
    if (
        value["name"] != "step3o_o4_attempt_003_terminal_stop_v1"
        or
        value.get("attempt_id") != qualification.ATTEMPT_ID
        or spec is None
        or value.get("route") != spec["route"]
        or value.get("parent_commit") != expected_stop_parent(stage_name)
        or not isinstance(value.get("error_type"), str)
        or not value["error_type"]
        or not isinstance(value.get("error_message_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["error_message_sha256"])
        or value.get("replacement_or_retry_allowed") is not False
        or value.get("training_authorized") is not False
        or value.get("optimizer_steps") != 0
    ):
        raise StageValidationError("terminal stop artifact is invalid")

    evidence = value["evidence_artifacts"]
    if not isinstance(evidence, list):
        raise StageValidationError("terminal stop evidence must be a list")
    allowed_by_name = {
        pool.repo_path(path): path for path in spec["allowed_paths"]
    }
    paths: list[Path] = []
    names: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise StageValidationError("terminal stop evidence schema mismatch")
        name = item["path"]
        if name in names or name not in allowed_by_name:
            raise StageValidationError("terminal stop evidence path is duplicate or unregistered")
        names.add(name)
        path = allowed_by_name[name]
        data = path.read_bytes()
        if item != artifact(path, data):
            raise StageValidationError("terminal stop evidence hash or length mismatch")
        paths.append(path)
    expected_order = [
        pool.repo_path(path)
        for path in spec["allowed_paths"]
        if pool.repo_path(path) in names
    ]
    if [item["path"] for item in evidence] != expected_order:
        raise StageValidationError("terminal stop evidence order differs from stage registry")
    required_receipt = spec["required_receipt"]
    if required_receipt is not None and pool.repo_path(required_receipt) not in names:
        raise StageValidationError("terminal stop is missing its required input receipt")

    commit = verify_stage_commit(
        (*paths, ATTEMPT_STOP),
        ATTEMPT_STOP,
        value["parent_commit"],
    )
    return {
        **value,
        "checks_pass": True,
        "commit": commit,
    }


def add_input_arguments(parser: argparse.ArgumentParser, roles: Sequence[str]) -> None:
    for role in roles:
        parser.add_argument(f"--{role.replace('_', '-')}", required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    authors = subparsers.add_parser("freeze-authors")
    add_input_arguments(
        authors,
        ("author_a", "author_b", "author_a_provenance", "author_b_provenance"),
    )
    subparsers.add_parser("verify-authors")
    subparsers.add_parser("freeze-candidates")
    subparsers.add_parser("verify-candidates")
    ab = subparsers.add_parser("freeze-ab-reviews")
    add_input_arguments(
        ab,
        (
            "reviewer_a",
            "reviewer_b",
            "reviewer_a_provenance",
            "reviewer_b_provenance",
        ),
    )
    subparsers.add_parser("verify-ab-reviews")
    subparsers.add_parser("freeze-c-gate")
    subparsers.add_parser("verify-c-gate")
    final = subparsers.add_parser("freeze-final")
    add_input_arguments(final, ("reviewer_c", "reviewer_c_provenance"))
    subparsers.add_parser("verify-final")
    subparsers.add_parser("verify-terminal-stop")
    args = parser.parse_args()

    if args.command == "freeze-authors":
        result = freeze_authors(
            args.author_a,
            args.author_b,
            args.author_a_provenance,
            args.author_b_provenance,
        )
    elif args.command == "verify-authors":
        result = verify_authors()
    elif args.command == "freeze-candidates":
        result = freeze_candidates()
    elif args.command == "verify-candidates":
        result = verify_candidates()
    elif args.command == "freeze-ab-reviews":
        result = freeze_ab_reviews(
            args.reviewer_a,
            args.reviewer_b,
            args.reviewer_a_provenance,
            args.reviewer_b_provenance,
        )
    elif args.command == "verify-ab-reviews":
        result = verify_ab_reviews()
    elif args.command == "freeze-c-gate":
        result = freeze_c_gate()
    elif args.command == "verify-c-gate":
        result = verify_c_gate()
    elif args.command == "freeze-final":
        result = freeze_final(args.reviewer_c, args.reviewer_c_provenance)
    elif args.command == "verify-final":
        result = verify_final()
    elif args.command == "verify-terminal-stop":
        result = verify_terminal_stop()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
