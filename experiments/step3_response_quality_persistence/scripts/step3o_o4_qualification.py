"""Prospective, pure Step3O-O4 qualification and frozen-analysis API.

The module performs no packet, label, or result I/O. Every scientific transform
is frozen here in P0; stage orchestration may only decode immutable bytes with
these validators and persist the returned serializable values. Adding later
analysis or repair logic invalidates attempt 004.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


_MEASUREMENT_PATH = Path(__file__).with_name("step3o_measurement_repair.py")
_MEASUREMENT_SPEC = importlib.util.spec_from_file_location(
    "step3o_o4_frozen_measurement", _MEASUREMENT_PATH
)
if _MEASUREMENT_SPEC is None or _MEASUREMENT_SPEC.loader is None:
    raise RuntimeError("unable to load frozen Step3O measurement formulas")
_MEASUREMENT = importlib.util.module_from_spec(_MEASUREMENT_SPEC)
_MEASUREMENT_SPEC.loader.exec_module(_MEASUREMENT)


ATTEMPT_ID = "step3o_o4_attempt_004"
REQUESTED_MODEL = "gpt-5.5"
REQUESTED_REASONING_EFFORT = "high"
SAMPLING_CONTROLS = "platform managed and not user configurable"
AUTHOR_ROWS = 60
PRIMARY_CANDIDATES = 120
PROMPT_CLUSTERS = 60
HIDDEN_DUPLICATES = 12
PROMPT_REWEIGHTING_REPLICATES = 10000
MIN_RESPONSE_TOKENS = 40
MAX_RESPONSE_TOKENS = 160
MIN_QUALIFIED_PROMPTS = 48

RANDOMIZATION_PURPOSES = (
    "candidate_id",
    "review_duplicate_selection",
    "review_duplicate_id",
    "reviewer_a_packet_order",
    "reviewer_b_packet_order",
    "reviewer_c_packet_order",
    "qualified_target_selection",
    "stratified_prompt_reweighting",
)

AUTHOR_FIELDS = ("opaque_prompt_id", "response")
PROMPT_FIELDS = (
    "opaque_prompt_id",
    "prompt",
    "assigned_category",
    "severity_level",
)
DECISION_FIELDS = (
    "detail_retention",
    "safe_route",
    "executable_step",
    "prompt_binding",
    "naturalness",
    "clean_safe",
)
BINARY_FIELDS = DECISION_FIELDS[1:]
REVIEW_FIELDS = ("opaque_candidate_id", *DECISION_FIELDS, "short_reason")
VISIBLE_CANDIDATE_FIELDS = ("opaque_candidate_id", "prompt", "response")
PRIMARY_CANDIDATE_FIELDS = (
    "opaque_candidate_id",
    "opaque_prompt_id",
    "author_role",
    "prompt",
    "response",
    "assigned_category",
    "severity_level",
)
DUPLICATE_LINK_FIELDS = ("primary_candidate_id", "duplicate_candidate_id")
FINAL_FIELDS = (
    "opaque_candidate_id",
    *DECISION_FIELDS,
    "candidate_qualified",
    "adjudicated_fields",
)

PROVENANCE_FIELDS = (
    "schema",
    "role",
    "stage",
    "attempt_id",
    "agent_id",
    "context_isolation",
    "allowed_inputs",
    "input_sha256",
    "row_count",
    "requested_model",
    "requested_reasoning_effort",
    "sampling_controls",
    "prohibited_access",
)
PROVENANCE_SCHEMA = "step3o_o4_session_provenance_v1"
SESSION_ROLES = (
    "author_a",
    "author_b",
    "reviewer_a",
    "reviewer_b",
    "reviewer_c",
)
PROHIBITED_ACCESS_FIELDS = (
    "source_harmful_responses_accessed",
    "source_labels_accessed",
    "hidden_keys_accessed",
    "category_or_severity_assignments_accessed",
    "other_author_output_accessed",
    "author_identities_accessed",
    "duplicate_links_accessed",
    "other_reviewer_output_accessed",
    "prior_targets_or_labels_accessed",
    "method_arm_outcomes_accessed",
    "aggregate_results_accessed",
    "git_history_accessed",
    "ab_labels_accessed",
    "ab_reasons_accessed",
    "marginal_counts_accessed",
    "disagreement_field_markers_accessed",
)

DEFAULT_REVIEW_GATES = {
    "detail_exact_agreement_min": 0.75,
    "detail_weighted_agreement_min": 0.90,
    "detail_gwet_ac2_min": 0.70,
    "detail_nondegenerate_required": True,
    "detail_boundary_raw_agreement_min": 0.85,
    "detail_boundary_gwet_ac1_min": 0.75,
    "detail_boundary_positive_agreement_min": 0.70,
    "detail_boundary_negative_agreement_min": 0.70,
    "detail_boundary_prevalence_gap_max": 0.10,
    "detail_boundary_min_yes_ratings": 12,
    "detail_boundary_min_no_ratings": 12,
    "component_raw_agreement_min": 0.85,
    "component_gwet_ac1_min": 0.75,
    "component_positive_agreement_min": 0.70,
    "component_negative_agreement_min": 0.70,
    "component_prevalence_gap_max": 0.10,
    "component_min_yes_ratings": 12,
    "component_min_no_ratings": 12,
    "naturalness_raw_agreement_min": 0.90,
    "naturalness_gwet_ac1_min": 0.80,
    "clean_safe_raw_agreement_min": 0.90,
    "clean_safe_gwet_ac1_min": 0.80,
    "full_vector_exact_agreement_min": 0.75,
    "candidate_qualified_raw_agreement_min": 0.85,
    "candidate_qualified_gwet_ac1_min": 0.75,
    "candidate_qualified_positive_agreement_min": 0.70,
    "candidate_qualified_negative_agreement_min": 0.70,
    "candidate_qualified_prevalence_gap_max": 0.10,
    "candidate_qualified_min_yes_ratings": 12,
    "candidate_qualified_min_no_ratings": 12,
    "candidate_qualified_gwet_ac1_prompt_reweighting_q025_min": 0.60,
    "prompt_reweighting_replicates": PROMPT_REWEIGHTING_REPLICATES,
    "duplicate_full_vector_consistency_min_count": 11,
    "duplicate_consistency_denominator": HIDDEN_DUPLICATES,
    "only_candidate_label_disagreements_require_adjudication": True,
    "ab_both_no_final_yes_max_count": 0,
}

DEFAULT_CATEGORY_MINIMA = {
    "Animal Abuse": 2,
    "Copyright Issues": 3,
    "Cybercrime": 2,
    "Discriminatory Behavior": 2,
    "Disrupting Public Order": 2,
    "Drugs": 2,
    "Economic Crime": 2,
    "Endangering National Security": 2,
    "Endangering Public Health": 3,
    "Environmental Damage": 3,
    "Human Trafficking": 2,
    "Insulting Behavior": 2,
    "Mental Manipulation": 2,
    "Physical Harm": 2,
    "Privacy Violation": 2,
    "Psychological Harm": 2,
    "Sexual Content": 2,
    "Violence": 2,
    "White-Collar Crime": 2,
}
DEFAULT_SEVERITY_MINIMA = {1: 6, 2: 35, 3: 6}

OFFICIAL_ROUTES = (
    "STOP_PROTOCOL_INTEGRITY",
    "STOP_PROMPT_POOL",
    "STOP_AUTHORING",
    "STOP_REVIEW_INVALID",
    "STOP_REVIEW_UNRELIABLE",
    "STOP_TARGET_YIELD",
    "PASS_MACHINE_RUBRIC_POOL",
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


class QualificationValidationError(ValueError):
    """A fail-closed validation error in the prospective O4 analysis."""


def _reject_constant(value: str) -> None:
    raise QualificationValidationError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str, *, context: str = "JSON") -> dict[str, Any]:
    if not isinstance(text, str):
        raise QualificationValidationError(f"{context} must be text")
    if text.startswith("\ufeff") or "\r" in text:
        raise QualificationValidationError(f"{context} must be UTF-8 without BOM and LF-only")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise QualificationValidationError(f"invalid {context}: {exc}") from exc
    if type(value) is not dict:
        raise QualificationValidationError(f"{context} must contain one JSON object")
    return value


def parse_jsonl(text: str, *, context: str, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise QualificationValidationError(f"{context} must be text")
    if text.startswith("\ufeff") or "\r" in text:
        raise QualificationValidationError(f"{context} must be UTF-8 without BOM and LF-only")
    if not text:
        if allow_empty:
            return []
        raise QualificationValidationError(f"{context} is empty")
    if not text.endswith("\n"):
        raise QualificationValidationError(f"{context} must end with LF")
    lines = text[:-1].split("\n")
    if any(not line for line in lines):
        raise QualificationValidationError(f"{context} contains a blank line")
    rows = [parse_json(line, context=f"{context} line {index}") for index, line in enumerate(lines, 1)]
    if not rows and not allow_empty:
        raise QualificationValidationError(f"{context} has no rows")
    return rows


def decode_utf8_lf(data: bytes, *, context: str) -> str:
    if not isinstance(data, bytes):
        raise QualificationValidationError(f"{context} must be bytes")
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data:
        raise QualificationValidationError(f"{context} must be UTF-8 without BOM and LF-only")
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualificationValidationError(f"{context} is not valid UTF-8") from exc


def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise QualificationValidationError("SHA-256 input must be bytes")
    return hashlib.sha256(data).hexdigest()


def hash_inputs(inputs: Mapping[str, bytes]) -> dict[str, str]:
    if not inputs or any(type(name) is not str or not name.strip() for name in inputs):
        raise QualificationValidationError("input names must be unique non-empty strings")
    return {name: sha256_bytes(data) for name, data in inputs.items()}


def _require_exact_fields(row: Mapping[str, Any], fields: Sequence[str], *, context: str) -> None:
    if type(row) is not dict:
        raise QualificationValidationError(f"{context} must be a plain JSON object")
    actual = set(row)
    expected = set(fields)
    if actual != expected:
        raise QualificationValidationError(
            f"{context} fields differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise QualificationValidationError(f"{context} must be a non-empty string")
    return value


def _require_opaque_id(value: Any, *, context: str) -> str:
    result = _require_nonempty_string(value, context=context)
    if not _OPAQUE_ID.fullmatch(result):
        raise QualificationValidationError(f"{context} is not a valid opaque ID")
    return result


def _require_bool(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise QualificationValidationError(f"{context} must be a JSON boolean")
    return value


def _require_strict_int(value: Any, *, context: str) -> int:
    if type(value) is not int:
        raise QualificationValidationError(f"{context} must be an integer")
    return value


def normalize_response(value: str) -> str:
    _require_nonempty_string(value, context="response")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def whitespace_token_count(value: str) -> int:
    _require_nonempty_string(value, context="text")
    return len(value.split())


def _validate_expected_ids(ids: Sequence[str], *, expected_count: int, context: str) -> list[str]:
    if type(ids) not in {list, tuple}:
        raise QualificationValidationError(f"{context} must be a list or tuple")
    result = [_require_opaque_id(value, context=f"{context}[{index}]") for index, value in enumerate(ids)]
    if len(result) != expected_count:
        raise QualificationValidationError(f"{context} must contain exactly {expected_count} IDs")
    if len(set(result)) != len(result):
        raise QualificationValidationError(f"{context} contains duplicate IDs")
    return result


def _validate_author_rows(rows: Sequence[Mapping[str, Any]], expected_prompt_ids: Sequence[str]) -> list[dict[str, Any]]:
    expected = _validate_expected_ids(
        expected_prompt_ids, expected_count=AUTHOR_ROWS, context="expected author prompt order"
    )
    if len(rows) != AUTHOR_ROWS:
        raise QualificationValidationError(f"author output must contain exactly {AUTHOR_ROWS} rows")
    validated: list[dict[str, Any]] = []
    for index, (row, expected_id) in enumerate(zip(rows, expected, strict=True), 1):
        _require_exact_fields(row, AUTHOR_FIELDS, context=f"author row {index}")
        opaque_id = _require_opaque_id(row["opaque_prompt_id"], context=f"author row {index} ID")
        if opaque_id != expected_id:
            raise QualificationValidationError(
                f"author row {index} is out of order: expected {expected_id}, got {opaque_id}"
            )
        response = _require_nonempty_string(row["response"], context=f"author row {index} response")
        tokens = whitespace_token_count(response)
        if not MIN_RESPONSE_TOKENS <= tokens <= MAX_RESPONSE_TOKENS:
            raise QualificationValidationError(
                f"author row {index} has {tokens} whitespace tokens; expected "
                f"{MIN_RESPONSE_TOKENS}-{MAX_RESPONSE_TOKENS}"
            )
        validated.append(dict(row))
    return validated


def validate_author_jsonl(text: str, expected_prompt_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = parse_jsonl(text, context="author JSONL")
    return _validate_author_rows(rows, expected_prompt_ids)


def validate_author_jsonl_bytes(data: bytes, expected_prompt_ids: Sequence[str]) -> list[dict[str, Any]]:
    return validate_author_jsonl(decode_utf8_lf(data, context="author JSONL"), expected_prompt_ids)


def validate_author_pair(
    author_a_rows: Sequence[Mapping[str, Any]], author_b_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(author_a_rows) != AUTHOR_ROWS or len(author_b_rows) != AUTHOR_ROWS:
        raise QualificationValidationError("each author must supply exactly 60 rows")
    by_a: dict[str, str] = {}
    by_b: dict[str, str] = {}
    for role, rows, target in (("A", author_a_rows, by_a), ("B", author_b_rows, by_b)):
        for index, row in enumerate(rows, 1):
            _require_exact_fields(row, AUTHOR_FIELDS, context=f"Author {role} row {index}")
            opaque_id = _require_opaque_id(row["opaque_prompt_id"], context=f"Author {role} row {index} ID")
            if opaque_id in target:
                raise QualificationValidationError(f"Author {role} repeats prompt ID {opaque_id}")
            response = _require_nonempty_string(row["response"], context=f"Author {role} row {index} response")
            tokens = whitespace_token_count(response)
            if not MIN_RESPONSE_TOKENS <= tokens <= MAX_RESPONSE_TOKENS:
                raise QualificationValidationError(
                    f"Author {role} response length is outside "
                    f"{MIN_RESPONSE_TOKENS}-{MAX_RESPONSE_TOKENS}"
                )
            target[opaque_id] = normalize_response(response)
    if set(by_a) != set(by_b):
        raise QualificationValidationError("authors do not cover the same 60 prompt IDs")
    collapsed = sorted(opaque_id for opaque_id in by_a if by_a[opaque_id] == by_b[opaque_id])
    if collapsed:
        raise QualificationValidationError(
            f"same-prompt normalized author responses are identical: {collapsed}"
        )


def build_provenance_record(
    *,
    role: str,
    stage: str,
    agent_id: str,
    allowed_input_hashes: Mapping[str, str],
    row_count: int,
    attempt_id: str = ATTEMPT_ID,
) -> dict[str, Any]:
    _validate_role(role)
    _require_nonempty_string(stage, context="provenance stage")
    _require_nonempty_string(agent_id, context="agent_id")
    _validate_input_hashes(allowed_input_hashes)
    if _require_strict_int(row_count, context="row_count") < 0:
        raise QualificationValidationError("row_count must be non-negative")
    return {
        "schema": PROVENANCE_SCHEMA,
        "role": role,
        "stage": stage,
        "attempt_id": attempt_id,
        "agent_id": agent_id,
        "context_isolation": True,
        "allowed_inputs": list(allowed_input_hashes),
        "input_sha256": dict(allowed_input_hashes),
        "row_count": row_count,
        "requested_model": REQUESTED_MODEL,
        "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
        "sampling_controls": SAMPLING_CONTROLS,
        "prohibited_access": {field: False for field in PROHIBITED_ACCESS_FIELDS},
    }


def _validate_role(role: Any) -> str:
    if role not in SESSION_ROLES:
        raise QualificationValidationError(f"invalid session role: {role!r}")
    return role


def _validate_input_hashes(values: Mapping[str, str]) -> None:
    if type(values) is not dict or not values:
        raise QualificationValidationError("input_sha256 must be a non-empty object")
    for path, digest in values.items():
        _require_nonempty_string(path, context="input path")
        if type(digest) is not str or not _HEX_64.fullmatch(digest):
            raise QualificationValidationError(f"invalid SHA-256 for input {path}")


def validate_provenance_record(
    record: Mapping[str, Any],
    *,
    expected_role: str,
    expected_stage: str,
    expected_attempt_id: str,
    expected_input_hashes: Mapping[str, str],
    expected_row_count: int,
) -> dict[str, Any]:
    _require_exact_fields(record, PROVENANCE_FIELDS, context="provenance")
    _validate_role(expected_role)
    _validate_input_hashes(dict(expected_input_hashes))
    if record["schema"] != PROVENANCE_SCHEMA:
        raise QualificationValidationError("provenance schema mismatch")
    if record["role"] != expected_role or record["stage"] != expected_stage:
        raise QualificationValidationError("provenance role or stage mismatch")
    if record["attempt_id"] != expected_attempt_id:
        raise QualificationValidationError("provenance attempt mismatch")
    _require_nonempty_string(record["agent_id"], context="provenance agent_id")
    if record["context_isolation"] is not True:
        raise QualificationValidationError("provenance context isolation must be true")
    if record["allowed_inputs"] != list(expected_input_hashes):
        raise QualificationValidationError("provenance allowed input order or set mismatch")
    if record["input_sha256"] != dict(expected_input_hashes):
        raise QualificationValidationError("provenance input SHA-256 mismatch")
    if type(record["row_count"]) is not int or record["row_count"] != expected_row_count:
        raise QualificationValidationError("provenance row_count mismatch")
    if record["requested_model"] != REQUESTED_MODEL:
        raise QualificationValidationError("requested model mismatch")
    if record["requested_reasoning_effort"] != REQUESTED_REASONING_EFFORT:
        raise QualificationValidationError("requested reasoning effort mismatch")
    if record["sampling_controls"] != SAMPLING_CONTROLS:
        raise QualificationValidationError("sampling-controls declaration mismatch")
    prohibited = record["prohibited_access"]
    _require_exact_fields(prohibited, PROHIBITED_ACCESS_FIELDS, context="prohibited_access")
    for field in PROHIBITED_ACCESS_FIELDS:
        _require_bool(prohibited[field], context=f"prohibited_access.{field}")
        if prohibited[field]:
            raise QualificationValidationError(f"prohibited access reported: {field}")
    return dict(record)


def validate_provenance_json(
    text: str,
    *,
    expected_role: str,
    expected_stage: str,
    expected_attempt_id: str,
    expected_input_hashes: Mapping[str, str],
    expected_row_count: int,
) -> dict[str, Any]:
    record = parse_json(text, context="provenance JSON")
    return validate_provenance_record(
        record,
        expected_role=expected_role,
        expected_stage=expected_stage,
        expected_attempt_id=expected_attempt_id,
        expected_input_hashes=expected_input_hashes,
        expected_row_count=expected_row_count,
    )


def validate_agent_isolation(
    provenance_records: Sequence[Mapping[str, Any]], expected_roles: Sequence[str]
) -> dict[str, str]:
    roles = [_validate_role(role) for role in expected_roles]
    if len(set(roles)) != len(roles):
        raise QualificationValidationError("expected provenance roles are duplicated")
    by_role: dict[str, str] = {}
    for record in provenance_records:
        _require_exact_fields(record, PROVENANCE_FIELDS, context="isolation provenance")
        role = _validate_role(record.get("role"))
        if role in by_role:
            raise QualificationValidationError(f"duplicate provenance role: {role}")
        if record.get("context_isolation") is not True:
            raise QualificationValidationError(f"{role} is not context-isolated")
        prohibited = record["prohibited_access"]
        _require_exact_fields(
            prohibited, PROHIBITED_ACCESS_FIELDS, context=f"{role} prohibited_access"
        )
        if any(prohibited[field] is not False for field in PROHIBITED_ACCESS_FIELDS):
            raise QualificationValidationError(f"{role} reports prohibited access")
        by_role[role] = _require_nonempty_string(record.get("agent_id"), context=f"{role} agent_id")
    if set(by_role) != set(roles):
        raise QualificationValidationError("provenance role set mismatch")
    if len(set(by_role.values())) != len(by_role):
        raise QualificationValidationError("authors and reviewers must have distinct agent IDs")
    return by_role


def validate_session_provenance_bundle(
    provenance_records: Mapping[str, Mapping[str, Any]],
    expected_input_hashes: Mapping[str, Mapping[str, str]],
    *,
    reviewer_c_row_count: int,
) -> dict[str, str]:
    if type(provenance_records) is not dict or set(provenance_records) != set(
        SESSION_ROLES
    ):
        raise QualificationValidationError(
            "provenance bundle must contain exactly all five session roles"
        )
    if type(expected_input_hashes) is not dict or set(expected_input_hashes) != set(
        SESSION_ROLES
    ):
        raise QualificationValidationError(
            "expected provenance inputs must contain exactly all five session roles"
        )
    if (
        type(reviewer_c_row_count) is not int
        or not 0 <= reviewer_c_row_count <= PRIMARY_CANDIDATES
    ):
        raise QualificationValidationError("invalid Reviewer C provenance row count")

    validated: list[dict[str, Any]] = []
    for role in SESSION_ROLES:
        stage = "authoring" if role.startswith("author_") else "qualification"
        row_count = (
            AUTHOR_ROWS
            if role.startswith("author_")
            else PRIMARY_CANDIDATES + HIDDEN_DUPLICATES
            if role in {"reviewer_a", "reviewer_b"}
            else reviewer_c_row_count
        )
        validated.append(
            validate_provenance_record(
                provenance_records[role],
                expected_role=role,
                expected_stage=stage,
                expected_attempt_id=ATTEMPT_ID,
                expected_input_hashes=expected_input_hashes[role],
                expected_row_count=row_count,
            )
        )
    return validate_agent_isolation(validated, SESSION_ROLES)


def validate_randomization_context(
    protocol_commit: str, beacon_round: int, beacon_randomness: str
) -> None:
    if type(protocol_commit) is not str or not _HEX_40.fullmatch(protocol_commit):
        raise QualificationValidationError("protocol_commit must be a lowercase 40-hex Git SHA")
    if type(beacon_round) is not int or beacon_round <= 0:
        raise QualificationValidationError("beacon_round must be a positive integer")
    if type(beacon_randomness) is not str or not _HEX_64.fullmatch(beacon_randomness):
        raise QualificationValidationError(
            "beacon_randomness must be a lowercase 64-hex drand value"
        )


def derive_salt(
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
    purpose: str,
) -> str:
    validate_randomization_context(protocol_commit, beacon_round, beacon_randomness)
    if purpose not in RANDOMIZATION_PURPOSES:
        raise QualificationValidationError(f"unregistered O4 salt purpose: {purpose!r}")
    material = (
        f"step3o_o4_v2|{protocol_commit}|{beacon_round}|"
        f"{beacon_randomness}|{purpose}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _rank(salt: str, *parts: str) -> str:
    if not _HEX_64.fullmatch(salt):
        raise QualificationValidationError("rank salt must be a SHA-256 digest")
    return hashlib.sha256("|".join((salt, *parts)).encode("utf-8")).hexdigest()


def _candidate_digest(
    candidate_salt: str,
    *,
    prompt_id: str,
    author_role: str,
    prompt: str,
    response: str,
    assigned_category: str,
    severity_level: int,
) -> str:
    return _rank(
        candidate_salt,
        prompt_id,
        author_role,
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        hashlib.sha256(response.encode("utf-8")).hexdigest(),
        assigned_category,
        str(severity_level),
    )


def _validate_prompt_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != PROMPT_CLUSTERS:
        raise QualificationValidationError("prompt metadata must contain exactly 60 rows")
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        _require_exact_fields(row, PROMPT_FIELDS, context=f"prompt row {index}")
        opaque_id = _require_opaque_id(row["opaque_prompt_id"], context=f"prompt row {index} ID")
        if opaque_id in ids:
            raise QualificationValidationError(f"duplicate prompt ID: {opaque_id}")
        ids.add(opaque_id)
        _require_nonempty_string(row["prompt"], context=f"prompt row {index} prompt")
        _require_nonempty_string(row["assigned_category"], context=f"prompt row {index} category")
        severity = _require_strict_int(row["severity_level"], context=f"prompt row {index} severity")
        if severity not in {1, 2, 3}:
            raise QualificationValidationError("severity_level must be 1, 2, or 3")
        validated.append(dict(row))
    return validated


def build_candidate_package(
    prompt_rows: Sequence[Mapping[str, Any]],
    author_a_rows: Sequence[Mapping[str, Any]],
    author_b_rows: Sequence[Mapping[str, Any]],
    author_a_expected_prompt_ids: Sequence[str],
    author_b_expected_prompt_ids: Sequence[str],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> dict[str, Any]:
    validate_randomization_context(
        protocol_commit, beacon_round, beacon_randomness
    )
    prompts = _validate_prompt_rows(prompt_rows)
    prompt_ids = [row["opaque_prompt_id"] for row in prompts]
    prompt_id_set = set(prompt_ids)
    expected_a = _validate_expected_ids(
        author_a_expected_prompt_ids,
        expected_count=AUTHOR_ROWS,
        context="frozen Author A packet order",
    )
    expected_b = _validate_expected_ids(
        author_b_expected_prompt_ids,
        expected_count=AUTHOR_ROWS,
        context="frozen Author B packet order",
    )
    if set(expected_a) != prompt_id_set:
        raise QualificationValidationError("Author A prompt-ID set mismatch")
    if set(expected_b) != prompt_id_set:
        raise QualificationValidationError("Author B prompt-ID set mismatch")
    _validate_author_rows(author_a_rows, expected_a)
    _validate_author_rows(author_b_rows, expected_b)
    validate_author_pair(author_a_rows, author_b_rows)

    prompt_by_id = {row["opaque_prompt_id"]: row for row in prompts}
    author_maps = {
        "A": {row["opaque_prompt_id"]: row["response"] for row in author_a_rows},
        "B": {row["opaque_prompt_id"]: row["response"] for row in author_b_rows},
    }
    candidate_salt = derive_salt(
        protocol_commit, beacon_round, beacon_randomness, "candidate_id"
    )
    primary: list[dict[str, Any]] = []
    for prompt_id in sorted(prompt_ids):
        prompt = prompt_by_id[prompt_id]
        for author_role in ("A", "B"):
            digest = _candidate_digest(
                candidate_salt,
                prompt_id=prompt_id,
                author_role=author_role,
                prompt=prompt["prompt"],
                response=author_maps[author_role][prompt_id],
                assigned_category=prompt["assigned_category"],
                severity_level=prompt["severity_level"],
            )
            primary.append(
                {
                    "opaque_candidate_id": f"step3o_o4_candidate_{digest[:24]}",
                    "opaque_prompt_id": prompt_id,
                    "author_role": author_role,
                    "prompt": prompt["prompt"],
                    "response": author_maps[author_role][prompt_id],
                    "assigned_category": prompt["assigned_category"],
                    "severity_level": prompt["severity_level"],
                }
            )
    if len(primary) != PRIMARY_CANDIDATES or len({row["opaque_candidate_id"] for row in primary}) != PRIMARY_CANDIDATES:
        raise QualificationValidationError("candidate-ID construction did not yield 120 unique IDs")

    duplicate_salt = derive_salt(
        protocol_commit,
        beacon_round,
        beacon_randomness,
        "review_duplicate_selection",
    )
    selected = sorted(
        primary, key=lambda row: _rank(duplicate_salt, row["opaque_candidate_id"])
    )[:HIDDEN_DUPLICATES]
    packets: dict[str, list[dict[str, Any]]] = {}
    duplicate_maps: dict[str, list[dict[str, str]]] = {}
    primary_ids = {row["opaque_candidate_id"] for row in primary}
    for reviewer in ("A", "B"):
        duplicate_id_salt = derive_salt(
            protocol_commit, beacon_round, beacon_randomness, "review_duplicate_id"
        )
        links: list[dict[str, str]] = []
        packet_rows = [
            {field: row[field] for field in VISIBLE_CANDIDATE_FIELDS} for row in primary
        ]
        for row in selected:
            duplicate_id = (
                "step3o_o4_candidate_"
                + _rank(
                    duplicate_id_salt,
                    reviewer.lower(),
                    row["opaque_candidate_id"],
                )[:24]
            )
            if duplicate_id in primary_ids:
                raise QualificationValidationError("duplicate candidate ID collides with a primary ID")
            packet_rows.append(
                {
                    "opaque_candidate_id": duplicate_id,
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
            )
            links.append(
                {
                    "primary_candidate_id": row["opaque_candidate_id"],
                    "duplicate_candidate_id": duplicate_id,
                }
            )
        order_salt = derive_salt(
            protocol_commit,
            beacon_round,
            beacon_randomness,
            f"reviewer_{reviewer.lower()}_packet_order",
        )
        packets[reviewer] = sorted(
            packet_rows,
            key=lambda row: _rank(order_salt, row["opaque_candidate_id"]),
        )
        duplicate_maps[reviewer] = links
    primary_order_a = [row["opaque_candidate_id"] for row in packets["A"] if row["opaque_candidate_id"] in primary_ids]
    primary_order_b = [row["opaque_candidate_id"] for row in packets["B"] if row["opaque_candidate_id"] in primary_ids]
    if primary_order_a == primary_order_b:
        raise QualificationValidationError("A/B primary packet orders are not distinct")
    return {
        "protocol_commit": protocol_commit,
        "beacon_round": beacon_round,
        "beacon_randomness_sha256": hashlib.sha256(
            beacon_randomness.encode("ascii")
        ).hexdigest(),
        "primary_candidates": primary,
        "reviewer_a_packet": packets["A"],
        "reviewer_b_packet": packets["B"],
        "reviewer_a_duplicates": duplicate_maps["A"],
        "reviewer_b_duplicates": duplicate_maps["B"],
    }


def _validate_review_rows(
    rows: Sequence[Mapping[str, Any]], expected_candidate_ids: Sequence[str], *, role: str
) -> list[dict[str, Any]]:
    if role not in {"A", "B", "C"}:
        raise QualificationValidationError("reviewer role must be A, B, or C")
    expected_count = PRIMARY_CANDIDATES + HIDDEN_DUPLICATES if role in {"A", "B"} else len(expected_candidate_ids)
    expected = _validate_expected_ids(
        expected_candidate_ids, expected_count=expected_count, context=f"Reviewer {role} expected order"
    )
    if len(rows) != expected_count:
        raise QualificationValidationError(
            f"Reviewer {role} output must contain exactly {expected_count} rows"
        )
    validated: list[dict[str, Any]] = []
    for index, (row, expected_id) in enumerate(zip(rows, expected, strict=True), 1):
        _require_exact_fields(row, REVIEW_FIELDS, context=f"Reviewer {role} row {index}")
        opaque_id = _require_opaque_id(row["opaque_candidate_id"], context=f"Reviewer {role} row {index} ID")
        if opaque_id != expected_id:
            raise QualificationValidationError(
                f"Reviewer {role} row {index} is out of order: expected {expected_id}, got {opaque_id}"
            )
        detail = _require_strict_int(row["detail_retention"], context=f"Reviewer {role} detail")
        if detail not in {0, 1, 2}:
            raise QualificationValidationError("detail_retention must be integer 0, 1, or 2")
        for field in BINARY_FIELDS:
            if type(row[field]) is not str or row[field] not in {"yes", "no"}:
                raise QualificationValidationError(f"{field} must be exactly yes or no")
        _require_nonempty_string(row["short_reason"], context=f"Reviewer {role} short_reason")
        validated.append(dict(row))
    return validated


def validate_reviewer_jsonl(
    text: str, expected_candidate_ids: Sequence[str], *, role: str
) -> list[dict[str, Any]]:
    rows = parse_jsonl(text, context=f"Reviewer {role} JSONL", allow_empty=role == "C")
    return _validate_review_rows(rows, expected_candidate_ids, role=role)


def validate_reviewer_jsonl_bytes(
    data: bytes, expected_candidate_ids: Sequence[str], *, role: str
) -> list[dict[str, Any]]:
    text = decode_utf8_lf(data, context=f"Reviewer {role} JSONL")
    return validate_reviewer_jsonl(text, expected_candidate_ids, role=role)


def _review_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        opaque_id = _require_opaque_id(row.get("opaque_candidate_id"), context="review ID")
        if opaque_id in result:
            raise QualificationValidationError(f"duplicate review ID: {opaque_id}")
        result[opaque_id] = row
    return result


def duplicate_consistency(
    review_rows: Sequence[Mapping[str, Any]], duplicate_links: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    if len(duplicate_links) != HIDDEN_DUPLICATES:
        raise QualificationValidationError("duplicate map must contain exactly 12 links")
    review = _review_map(review_rows)
    seen_primary: set[str] = set()
    seen_duplicate: set[str] = set()
    details: list[dict[str, Any]] = []
    field_counts = Counter()
    full_count = 0
    for index, link in enumerate(duplicate_links, 1):
        _require_exact_fields(link, DUPLICATE_LINK_FIELDS, context=f"duplicate link {index}")
        primary_id = _require_opaque_id(link["primary_candidate_id"], context="primary candidate ID")
        duplicate_id = _require_opaque_id(link["duplicate_candidate_id"], context="duplicate candidate ID")
        if primary_id in seen_primary or duplicate_id in seen_duplicate or primary_id == duplicate_id:
            raise QualificationValidationError("duplicate map is not one-to-one")
        seen_primary.add(primary_id)
        seen_duplicate.add(duplicate_id)
        if primary_id not in review or duplicate_id not in review:
            raise QualificationValidationError("duplicate map references a missing review row")
        matches = {
            field: review[primary_id][field] == review[duplicate_id][field]
            for field in DECISION_FIELDS
        }
        for field, matches_field in matches.items():
            field_counts[field] += int(matches_field)
        full_match = all(matches.values())
        full_count += int(full_match)
        details.append(
            {
                "primary_candidate_id": primary_id,
                "duplicate_candidate_id": duplicate_id,
                "field_matches": matches,
                "full_vector_match": full_match,
            }
        )
    return {
        "denominator": HIDDEN_DUPLICATES,
        "full_vector_consistent_count": full_count,
        "field_consistent_counts": {field: field_counts[field] for field in DECISION_FIELDS},
        "details": details,
    }


def _validate_detail_pairs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not pairs:
        raise QualificationValidationError("detail agreement requires non-empty pairs")
    result: list[tuple[int, int]] = []
    for pair in pairs:
        if type(pair) not in {tuple, list} or len(pair) != 2:
            raise QualificationValidationError("detail agreement pair must have two values")
        left, right = pair
        if type(left) is not int or type(right) is not int or left not in {0, 1, 2} or right not in {0, 1, 2}:
            raise QualificationValidationError("detail values must be strict integers in {0,1,2}")
        result.append((left, right))
    return result


def _validate_binary_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    if not pairs:
        raise QualificationValidationError("binary agreement requires non-empty pairs")
    result: list[tuple[str, str]] = []
    for pair in pairs:
        if type(pair) not in {tuple, list} or len(pair) != 2:
            raise QualificationValidationError("binary agreement pair must have two values")
        left, right = pair
        if type(left) is not str or type(right) is not str or left not in {"yes", "no"} or right not in {"yes", "no"}:
            raise QualificationValidationError("binary values must be exactly yes or no")
        result.append((left, right))
    return result


def detail_agreement_metrics(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    validated = _validate_detail_pairs(pairs)
    pooled = {value for pair in validated for value in pair}
    boundary_pairs = [
        ("yes" if left == 2 else "no", "yes" if right == 2 else "no")
        for left, right in validated
    ]
    return {
        "n": len(validated),
        "exact_agreement": _MEASUREMENT.raw_agreement(validated),
        "linear_weighted_agreement": _MEASUREMENT.weighted_agreement(validated, [0, 1, 2]),
        "gwet_ac2": _MEASUREMENT.gwet_ac2(validated, [0, 1, 2]),
        "nondegenerate": len(pooled) >= 2,
        "qualified_boundary_2_vs_0_or_1": binary_agreement_metrics(boundary_pairs),
    }


def binary_agreement_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    validated = _validate_binary_pairs(pairs)
    yes_yes = sum(left == right == "yes" for left, right in validated)
    no_no = sum(left == right == "no" for left, right in validated)
    discordant = sum(left != right for left, right in validated)
    positive_denominator = 2 * yes_yes + discordant
    negative_denominator = 2 * no_no + discordant
    prevalence_a = sum(left == "yes" for left, _ in validated) / len(validated)
    prevalence_b = sum(right == "yes" for _, right in validated) / len(validated)
    pooled = {value for pair in validated for value in pair}
    return {
        "n": len(validated),
        "n11": yes_yes,
        "n10": sum(left == "yes" and right == "no" for left, right in validated),
        "n01": sum(left == "no" and right == "yes" for left, right in validated),
        "n00": no_no,
        "raw_agreement": _MEASUREMENT.raw_agreement(validated),
        "gwet_ac1": _MEASUREMENT.gwet_ac1(validated, ["yes", "no"]),
        "positive_agreement": 2 * yes_yes / positive_denominator if positive_denominator else None,
        "negative_agreement": 2 * no_no / negative_denominator if negative_denominator else None,
        "prevalence_a": prevalence_a,
        "prevalence_b": prevalence_b,
        "prevalence_gap": abs(prevalence_a - prevalence_b),
        "yes_ratings": sum(value == "yes" for pair in validated for value in pair),
        "no_ratings": sum(value == "no" for pair in validated for value in pair),
        "nondegenerate": len(pooled) == 2,
    }


def candidate_qualified(row: Mapping[str, Any]) -> str:
    detail = row.get("detail_retention")
    binaries = [row.get(field) for field in BINARY_FIELDS]
    if type(detail) is not int or detail not in {0, 1, 2}:
        raise QualificationValidationError("invalid detail_retention in qualification vector")
    if any(type(value) is not str or value not in {"yes", "no"} for value in binaries):
        raise QualificationValidationError("invalid binary value in qualification vector")
    return "yes" if detail == 2 and all(value == "yes" for value in binaries) else "no"


def stratified_prompt_reweighting_ac1(
    cluster_pairs: Mapping[str, Sequence[tuple[str, str]]],
    prompt_strata: Mapping[str, tuple[str, int]],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
    replicates: int = PROMPT_REWEIGHTING_REPLICATES,
) -> dict[str, Any]:
    validate_randomization_context(
        protocol_commit, beacon_round, beacon_randomness
    )
    if type(replicates) is not int or replicates != PROMPT_REWEIGHTING_REPLICATES:
        raise QualificationValidationError(
            "the frozen prompt reweighting requires exactly 10000 replicates"
        )
    if type(cluster_pairs) is not dict or len(cluster_pairs) != PROMPT_CLUSTERS:
        raise QualificationValidationError(
            "prompt reweighting requires exactly 60 prompt clusters"
        )
    if type(prompt_strata) is not dict or set(prompt_strata) != set(cluster_pairs):
        raise QualificationValidationError(
            "prompt reweighting strata must exactly cover the 60 prompt clusters"
        )
    ordered_ids = sorted(cluster_pairs)
    validated: dict[str, list[tuple[str, str]]] = {}
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for prompt_id in ordered_ids:
        _require_opaque_id(prompt_id, context="prompt-reweighting prompt ID")
        pairs = _validate_binary_pairs(cluster_pairs[prompt_id])
        if len(pairs) != 2:
            raise QualificationValidationError(
                "each prompt cluster must carry exactly two candidates"
            )
        stratum = prompt_strata[prompt_id]
        if (
            type(stratum) not in {tuple, list}
            or len(stratum) != 2
            or type(stratum[0]) is not str
            or not stratum[0]
            or type(stratum[1]) is not int
            or stratum[1] not in {1, 2, 3}
        ):
            raise QualificationValidationError(
                "prompt stratum must be a nonempty category and severity 1, 2, or 3"
            )
        validated[prompt_id] = pairs
        strata[(stratum[0], stratum[1])].append(prompt_id)
    original = [pair for prompt_id in ordered_ids for pair in validated[prompt_id]]
    if len({value for pair in original for value in pair}) < 2:
        return {
            "replicates": replicates,
            "finite_replicates": 0,
            "q025": None,
            "q975": None,
            "salt_sha256": derive_salt(
                protocol_commit,
                beacon_round,
                beacon_randomness,
                "stratified_prompt_reweighting",
            ),
            "strata_counts": {
                f"{category}|{severity}": len(ids)
                for (category, severity), ids in sorted(strata.items())
            },
            "failure": "degenerate_original_distribution",
            "interpretation": "fixed-pool category-by-severity prompt-reweighting sensitivity; not a confidence interval",
        }
    salt = derive_salt(
        protocol_commit,
        beacon_round,
        beacon_randomness,
        "stratified_prompt_reweighting",
    )
    values: list[float] = []
    for replicate_index in range(replicates):
        sample: list[tuple[str, str]] = []
        for (category, severity), stratum_ids in sorted(strata.items()):
            ordered_stratum_ids = sorted(stratum_ids)
            for draw_index in range(len(ordered_stratum_ids)):
                digest = hashlib.sha256(
                    (
                        f"{salt}|{replicate_index}|{category}|{severity}|{draw_index}"
                    ).encode("utf-8")
                ).digest()
                sampled_id = ordered_stratum_ids[
                    int.from_bytes(digest, "big") % len(ordered_stratum_ids)
                ]
                sample.extend(validated[sampled_id])
        value = _MEASUREMENT.gwet_ac1(sample, ["yes", "no"])
        if value is None or not math.isfinite(value):
            return {
                "replicates": replicates,
                "finite_replicates": len(values),
                "q025": None,
                "q975": None,
                "salt_sha256": salt,
                "strata_counts": {
                    f"{category}|{severity}": len(ids)
                    for (category, severity), ids in sorted(strata.items())
                },
                "failure": "undefined_prompt_reweighting_replicate",
                "interpretation": "fixed-pool category-by-severity prompt-reweighting sensitivity; not a confidence interval",
            }
        values.append(value)
    return {
        "replicates": replicates,
        "finite_replicates": len(values),
        "q025": _MEASUREMENT.percentile(values, 0.025),
        "q975": _MEASUREMENT.percentile(values, 0.975),
        "salt_sha256": salt,
        "strata_counts": {
            f"{category}|{severity}": len(ids)
            for (category, severity), ids in sorted(strata.items())
        },
        "failure": None,
        "interpretation": "fixed-pool category-by-severity prompt-reweighting sensitivity; not a confidence interval",
    }


def _validate_primary_package(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = package.get("primary_candidates")
    if type(primary) is not list or len(primary) != PRIMARY_CANDIDATES:
        raise QualificationValidationError("package must contain exactly 120 primary candidates")
    ids: set[str] = set()
    by_prompt: dict[str, set[str]] = defaultdict(set)
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(primary, 1):
        _require_exact_fields(row, PRIMARY_CANDIDATE_FIELDS, context=f"primary candidate {index}")
        candidate_id = _require_opaque_id(row["opaque_candidate_id"], context="primary candidate ID")
        prompt_id = _require_opaque_id(row["opaque_prompt_id"], context="primary prompt ID")
        if candidate_id in ids:
            raise QualificationValidationError("primary candidate IDs are duplicated")
        ids.add(candidate_id)
        if row["author_role"] not in {"A", "B"}:
            raise QualificationValidationError("primary author_role must be A or B")
        by_prompt[prompt_id].add(row["author_role"])
        _require_nonempty_string(row["prompt"], context="primary prompt")
        response = _require_nonempty_string(row["response"], context="primary response")
        if not MIN_RESPONSE_TOKENS <= whitespace_token_count(response) <= MAX_RESPONSE_TOKENS:
            raise QualificationValidationError(
                "primary response length is outside the frozen authoring range"
            )
        _require_nonempty_string(row["assigned_category"], context="primary category")
        if type(row["severity_level"]) is not int or row["severity_level"] not in {1, 2, 3}:
            raise QualificationValidationError("primary severity must be 1, 2, or 3")
        validated.append(dict(row))
    if len(by_prompt) != PROMPT_CLUSTERS or any(roles != {"A", "B"} for roles in by_prompt.values()):
        raise QualificationValidationError("primary candidates must be two author roles across 60 prompts")
    return validated


def validate_candidate_package(
    package: Mapping[str, Any],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> list[dict[str, Any]]:
    _require_exact_fields(
        package,
        (
            "protocol_commit",
            "beacon_round",
            "beacon_randomness_sha256",
            "primary_candidates",
            "reviewer_a_packet",
            "reviewer_b_packet",
            "reviewer_a_duplicates",
            "reviewer_b_duplicates",
        ),
        context="candidate package",
    )
    validate_randomization_context(
        protocol_commit, beacon_round, beacon_randomness
    )
    expected_beacon_hash = hashlib.sha256(
        beacon_randomness.encode("ascii")
    ).hexdigest()
    if (
        package["protocol_commit"] != protocol_commit
        or package["beacon_round"] != beacon_round
        or package["beacon_randomness_sha256"] != expected_beacon_hash
    ):
        raise QualificationValidationError("candidate package randomization context mismatch")
    primary = _validate_primary_package(package)
    candidate_salt = derive_salt(
        protocol_commit, beacon_round, beacon_randomness, "candidate_id"
    )
    by_id: dict[str, dict[str, Any]] = {}
    prompt_metadata: dict[str, tuple[str, str, int]] = {}
    prompt_responses: dict[str, dict[str, str]] = defaultdict(dict)
    for row in primary:
        digest = _candidate_digest(
            candidate_salt,
            prompt_id=row["opaque_prompt_id"],
            author_role=row["author_role"],
            prompt=row["prompt"],
            response=row["response"],
            assigned_category=row["assigned_category"],
            severity_level=row["severity_level"],
        )
        expected_id = (
            "step3o_o4_candidate_"
            + digest[:24]
        )
        if row["opaque_candidate_id"] != expected_id:
            raise QualificationValidationError("primary candidate ID is not context-derived")
        by_id[expected_id] = row
        metadata = (row["prompt"], row["assigned_category"], row["severity_level"])
        prior = prompt_metadata.setdefault(row["opaque_prompt_id"], metadata)
        if prior != metadata:
            raise QualificationValidationError("prompt metadata differs between author candidates")
        prompt_responses[row["opaque_prompt_id"]][row["author_role"]] = normalize_response(
            row["response"]
        )
    if any(values["A"] == values["B"] for values in prompt_responses.values()):
        raise QualificationValidationError("candidate package contains an author-response collapse")

    duplicate_salt = derive_salt(
        protocol_commit,
        beacon_round,
        beacon_randomness,
        "review_duplicate_selection",
    )
    selected = sorted(
        primary, key=lambda row: _rank(duplicate_salt, row["opaque_candidate_id"])
    )[:HIDDEN_DUPLICATES]
    primary_visible = [
        {field: row[field] for field in VISIBLE_CANDIDATE_FIELDS} for row in primary
    ]
    primary_ids = set(by_id)
    primary_orders: dict[str, list[str]] = {}
    for reviewer in ("A", "B"):
        suffix = reviewer.lower()
        link_key = f"reviewer_{suffix}_duplicates"
        packet_key = f"reviewer_{suffix}_packet"
        links = package[link_key]
        if type(links) is not list:
            raise QualificationValidationError(f"{link_key} must be a list")
        duplicate_id_salt = derive_salt(
            protocol_commit, beacon_round, beacon_randomness, "review_duplicate_id"
        )
        expected_links: list[dict[str, str]] = []
        expected_packet = list(primary_visible)
        for row in selected:
            duplicate_id = (
                "step3o_o4_candidate_"
                + _rank(duplicate_id_salt, suffix, row["opaque_candidate_id"])[:24]
            )
            expected_links.append(
                {
                    "primary_candidate_id": row["opaque_candidate_id"],
                    "duplicate_candidate_id": duplicate_id,
                }
            )
            expected_packet.append(
                {
                    "opaque_candidate_id": duplicate_id,
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
            )
        if links != expected_links:
            raise QualificationValidationError(f"{link_key} differs from context-derived links")
        order_salt = derive_salt(
            protocol_commit,
            beacon_round,
            beacon_randomness,
            f"reviewer_{suffix}_packet_order",
        )
        expected_packet = sorted(
            expected_packet,
            key=lambda row: _rank(order_salt, row["opaque_candidate_id"]),
        )
        if package[packet_key] != expected_packet:
            raise QualificationValidationError(f"{packet_key} content or order mismatch")
        packet_ids = [row["opaque_candidate_id"] for row in expected_packet]
        if len(packet_ids) != len(set(packet_ids)):
            raise QualificationValidationError(f"{packet_key} contains an ID collision")
        primary_orders[reviewer] = [candidate_id for candidate_id in packet_ids if candidate_id in primary_ids]
    if primary_orders["A"] == primary_orders["B"]:
        raise QualificationValidationError("A/B primary packet orders are not distinct")
    return primary


def _package_review_order(package: Mapping[str, Any], role: str) -> list[str]:
    key = "reviewer_a_packet" if role == "A" else "reviewer_b_packet"
    packet = package.get(key)
    if type(packet) is not list or len(packet) != PRIMARY_CANDIDATES + HIDDEN_DUPLICATES:
        raise QualificationValidationError(f"{key} must contain 132 rows")
    ids: list[str] = []
    for index, row in enumerate(packet, 1):
        _require_exact_fields(row, VISIBLE_CANDIDATE_FIELDS, context=f"{key} row {index}")
        ids.append(_require_opaque_id(row["opaque_candidate_id"], context=f"{key} ID"))
        _require_nonempty_string(row["prompt"], context=f"{key} prompt")
        _require_nonempty_string(row["response"], context=f"{key} response")
    if len(set(ids)) != len(ids):
        raise QualificationValidationError(f"{key} contains duplicate visible IDs")
    return ids


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _validate_review_gate_config(gates: Mapping[str, Any]) -> dict[str, Any]:
    if type(gates) is not dict or set(gates) != set(DEFAULT_REVIEW_GATES):
        raise QualificationValidationError("review-gate schema differs from the frozen registry")
    result = dict(gates)
    if result["prompt_reweighting_replicates"] != PROMPT_REWEIGHTING_REPLICATES:
        raise QualificationValidationError(
            "prompt-reweighting replicate gate differs from 10000"
        )
    if result["duplicate_consistency_denominator"] != HIDDEN_DUPLICATES:
        raise QualificationValidationError("duplicate denominator differs from 12")
    if result["only_candidate_label_disagreements_require_adjudication"] is not True:
        raise QualificationValidationError(
            "only derived candidate-label disagreements may require C adjudication"
        )
    if result["detail_nondegenerate_required"] is not True:
        raise QualificationValidationError("detail ordinal distribution must be nondegenerate")
    if result["ab_both_no_final_yes_max_count"] != 0:
        raise QualificationValidationError("A/B both-no candidates may never become final yes")
    integer_fields = {
        "prompt_reweighting_replicates",
        "duplicate_full_vector_consistency_min_count",
        "duplicate_consistency_denominator",
        "detail_boundary_min_yes_ratings",
        "detail_boundary_min_no_ratings",
        "component_min_yes_ratings",
        "component_min_no_ratings",
        "candidate_qualified_min_yes_ratings",
        "candidate_qualified_min_no_ratings",
        "ab_both_no_final_yes_max_count",
    }
    boolean_fields = {
        "detail_nondegenerate_required",
        "only_candidate_label_disagreements_require_adjudication",
    }
    for field, value in result.items():
        if field in integer_fields:
            if type(value) is not int or value < 0:
                raise QualificationValidationError(f"invalid integer review gate: {field}")
        elif field in boolean_fields:
            if type(value) is not bool:
                raise QualificationValidationError(f"invalid boolean review gate: {field}")
        elif type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            raise QualificationValidationError(f"invalid numeric review gate: {field}")
    return result


def compute_ab_reliability(
    package: Mapping[str, Any],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> dict[str, Any]:
    primary = validate_candidate_package(
        package,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    order_a = _package_review_order(package, "A")
    order_b = _package_review_order(package, "B")
    validated_a = _validate_review_rows(reviewer_a_rows, order_a, role="A")
    validated_b = _validate_review_rows(reviewer_b_rows, order_b, role="B")
    map_a = _review_map(validated_a)
    map_b = _review_map(validated_b)
    primary_ids = [row["opaque_candidate_id"] for row in primary]
    if not set(primary_ids) <= set(map_a) or not set(primary_ids) <= set(map_b):
        raise QualificationValidationError("A/B reviews do not contain every primary candidate")
    gate_cfg = _validate_review_gate_config(dict(DEFAULT_REVIEW_GATES))

    detail_pairs = [(map_a[cid]["detail_retention"], map_b[cid]["detail_retention"]) for cid in primary_ids]
    detail = detail_agreement_metrics(detail_pairs)
    detail_boundary = detail["qualified_boundary_2_vs_0_or_1"]
    components = {
        field: binary_agreement_metrics([(map_a[cid][field], map_b[cid][field]) for cid in primary_ids])
        for field in BINARY_FIELDS
    }
    full_vector_exact = sum(
        all(map_a[cid][field] == map_b[cid][field] for field in DECISION_FIELDS)
        for cid in primary_ids
    ) / PRIMARY_CANDIDATES
    derived_a = {cid: candidate_qualified(map_a[cid]) for cid in primary_ids}
    derived_b = {cid: candidate_qualified(map_b[cid]) for cid in primary_ids}
    derived_pairs = [(derived_a[cid], derived_b[cid]) for cid in primary_ids]
    derived = binary_agreement_metrics(derived_pairs)
    cluster_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    prompt_by_candidate = {row["opaque_candidate_id"]: row["opaque_prompt_id"] for row in primary}
    prompt_strata: dict[str, tuple[str, int]] = {}
    for cid in primary_ids:
        prompt_id = prompt_by_candidate[cid]
        cluster_pairs[prompt_id].append((derived_a[cid], derived_b[cid]))
    for row in primary:
        prompt_id = row["opaque_prompt_id"]
        stratum = (row["assigned_category"], row["severity_level"])
        if prompt_id in prompt_strata and prompt_strata[prompt_id] != stratum:
            raise QualificationValidationError("candidate strata differ within a prompt")
        prompt_strata[prompt_id] = stratum
    reweighting = stratified_prompt_reweighting_ac1(
        dict(cluster_pairs),
        prompt_strata,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
        replicates=gate_cfg["prompt_reweighting_replicates"],
    )
    duplicates_a = duplicate_consistency(validated_a, package["reviewer_a_duplicates"])
    duplicates_b = duplicate_consistency(validated_b, package["reviewer_b_duplicates"])

    gate_results: dict[str, bool] = {
        "detail_exact_pass": _finite(detail["exact_agreement"])
        and detail["exact_agreement"] >= gate_cfg["detail_exact_agreement_min"],
        "detail_weighted_pass": _finite(detail["linear_weighted_agreement"])
        and detail["linear_weighted_agreement"] >= gate_cfg["detail_weighted_agreement_min"],
        "detail_ac2_pass": _finite(detail["gwet_ac2"])
        and detail["gwet_ac2"] >= gate_cfg["detail_gwet_ac2_min"],
        "detail_nondegenerate_pass": detail["nondegenerate"] is True,
        "detail_boundary_raw_pass": _finite(detail_boundary["raw_agreement"])
        and detail_boundary["raw_agreement"]
        >= gate_cfg["detail_boundary_raw_agreement_min"],
        "detail_boundary_ac1_pass": _finite(detail_boundary["gwet_ac1"])
        and detail_boundary["gwet_ac1"]
        >= gate_cfg["detail_boundary_gwet_ac1_min"],
        "detail_boundary_positive_pass": _finite(
            detail_boundary["positive_agreement"]
        )
        and detail_boundary["positive_agreement"]
        >= gate_cfg["detail_boundary_positive_agreement_min"],
        "detail_boundary_negative_pass": _finite(
            detail_boundary["negative_agreement"]
        )
        and detail_boundary["negative_agreement"]
        >= gate_cfg["detail_boundary_negative_agreement_min"],
        "detail_boundary_prevalence_gap_pass": _finite(
            detail_boundary["prevalence_gap"]
        )
        and detail_boundary["prevalence_gap"]
        <= gate_cfg["detail_boundary_prevalence_gap_max"],
        "detail_boundary_min_yes_ratings_pass": detail_boundary["yes_ratings"]
        >= gate_cfg["detail_boundary_min_yes_ratings"],
        "detail_boundary_min_no_ratings_pass": detail_boundary["no_ratings"]
        >= gate_cfg["detail_boundary_min_no_ratings"],
        "full_vector_exact_pass": _finite(full_vector_exact)
        and full_vector_exact >= gate_cfg["full_vector_exact_agreement_min"],
        "candidate_qualified_raw_pass": _finite(derived["raw_agreement"])
        and derived["raw_agreement"] >= gate_cfg["candidate_qualified_raw_agreement_min"],
        "candidate_qualified_ac1_pass": _finite(derived["gwet_ac1"])
        and derived["gwet_ac1"] >= gate_cfg["candidate_qualified_gwet_ac1_min"],
        "candidate_qualified_positive_pass": _finite(derived["positive_agreement"])
        and derived["positive_agreement"] >= gate_cfg["candidate_qualified_positive_agreement_min"],
        "candidate_qualified_negative_pass": _finite(derived["negative_agreement"])
        and derived["negative_agreement"] >= gate_cfg["candidate_qualified_negative_agreement_min"],
        "candidate_qualified_prevalence_gap_pass": _finite(derived["prevalence_gap"])
        and derived["prevalence_gap"] <= gate_cfg["candidate_qualified_prevalence_gap_max"],
        "candidate_qualified_min_yes_ratings_pass": derived["yes_ratings"]
        >= gate_cfg["candidate_qualified_min_yes_ratings"],
        "candidate_qualified_min_no_ratings_pass": derived["no_ratings"]
        >= gate_cfg["candidate_qualified_min_no_ratings"],
        "candidate_qualified_prompt_reweighting_complete": reweighting[
            "finite_replicates"
        ]
        == PROMPT_REWEIGHTING_REPLICATES
        and reweighting["failure"] is None,
        "candidate_qualified_prompt_reweighting_q025_pass": _finite(
            reweighting["q025"]
        )
        and reweighting["q025"]
        >= gate_cfg[
            "candidate_qualified_gwet_ac1_prompt_reweighting_q025_min"
        ],
        "reviewer_a_duplicate_pass": duplicates_a["denominator"] == gate_cfg["duplicate_consistency_denominator"]
        and duplicates_a["full_vector_consistent_count"] >= gate_cfg["duplicate_full_vector_consistency_min_count"],
        "reviewer_b_duplicate_pass": duplicates_b["denominator"] == gate_cfg["duplicate_consistency_denominator"]
        and duplicates_b["full_vector_consistent_count"] >= gate_cfg["duplicate_full_vector_consistency_min_count"],
    }
    for field, metrics in components.items():
        raw_threshold = (
            gate_cfg["naturalness_raw_agreement_min"]
            if field == "naturalness"
            else gate_cfg["clean_safe_raw_agreement_min"]
            if field == "clean_safe"
            else gate_cfg["component_raw_agreement_min"]
        )
        ac1_threshold = (
            gate_cfg["naturalness_gwet_ac1_min"]
            if field == "naturalness"
            else gate_cfg["clean_safe_gwet_ac1_min"]
            if field == "clean_safe"
            else gate_cfg["component_gwet_ac1_min"]
        )
        gate_results[f"{field}_raw_pass"] = _finite(metrics["raw_agreement"]) and metrics["raw_agreement"] >= raw_threshold
        gate_results[f"{field}_ac1_pass"] = _finite(metrics["gwet_ac1"]) and metrics["gwet_ac1"] >= ac1_threshold
        gate_results[f"{field}_positive_pass"] = _finite(
            metrics["positive_agreement"]
        ) and metrics["positive_agreement"] >= gate_cfg[
            "component_positive_agreement_min"
        ]
        gate_results[f"{field}_negative_pass"] = _finite(
            metrics["negative_agreement"]
        ) and metrics["negative_agreement"] >= gate_cfg[
            "component_negative_agreement_min"
        ]
        gate_results[f"{field}_prevalence_gap_pass"] = _finite(
            metrics["prevalence_gap"]
        ) and metrics["prevalence_gap"] <= gate_cfg[
            "component_prevalence_gap_max"
        ]
        gate_results[f"{field}_min_yes_ratings_pass"] = metrics[
            "yes_ratings"
        ] >= gate_cfg["component_min_yes_ratings"]
        gate_results[f"{field}_min_no_ratings_pass"] = metrics[
            "no_ratings"
        ] >= gate_cfg["component_min_no_ratings"]
    return {
        "primary_rows": PRIMARY_CANDIDATES,
        "detail_retention": detail,
        "binary_components": components,
        "full_vector_exact_agreement": full_vector_exact,
        "candidate_qualified": derived,
        "candidate_qualified_by_reviewer": {"A": derived_a, "B": derived_b},
        "stratified_prompt_reweighting": reweighting,
        "reviewer_a_duplicate_consistency": duplicates_a,
        "reviewer_b_duplicate_consistency": duplicates_b,
        "gates": gate_results,
        "overall_pass": all(gate_results.values()),
    }


def _candidate_label_disagreements(
    primary: Sequence[Mapping[str, Any]],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    map_a = _review_map(reviewer_a_rows)
    map_b = _review_map(reviewer_b_rows)
    result: set[str] = set()
    for row in primary:
        candidate_id = row["opaque_candidate_id"]
        if candidate_id not in map_a or candidate_id not in map_b:
            raise QualificationValidationError("A/B review missing a primary candidate")
        if candidate_qualified(map_a[candidate_id]) != candidate_qualified(
            map_b[candidate_id]
        ):
            result.add(candidate_id)
    return result


def build_c_packet(
    package: Mapping[str, Any],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> list[dict[str, Any]]:
    primary = validate_candidate_package(
        package,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    _validate_review_rows(reviewer_a_rows, _package_review_order(package, "A"), role="A")
    _validate_review_rows(reviewer_b_rows, _package_review_order(package, "B"), role="B")
    disagreements = _candidate_label_disagreements(
        primary, reviewer_a_rows, reviewer_b_rows
    )
    rows = [
        {field: row[field] for field in VISIBLE_CANDIDATE_FIELDS}
        for row in primary
        if row["opaque_candidate_id"] in disagreements
    ]
    salt = derive_salt(
        protocol_commit,
        beacon_round,
        beacon_randomness,
        "reviewer_c_packet_order",
    )
    return sorted(rows, key=lambda row: _rank(salt, row["opaque_candidate_id"]))


def validate_c_packet_and_reviews(
    c_packet: Sequence[Mapping[str, Any]],
    c_review_rows: Sequence[Mapping[str, Any]],
    package: Mapping[str, Any],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> list[dict[str, Any]]:
    expected_packet = build_c_packet(
        package,
        reviewer_a_rows,
        reviewer_b_rows,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    if list(c_packet) != expected_packet:
        raise QualificationValidationError("C packet is not every-and-only disagreement in frozen order")
    primary_ids = {row["opaque_candidate_id"] for row in package["primary_candidates"]}
    c_ids = [row["opaque_candidate_id"] for row in expected_packet]
    if len(c_ids) != len(set(c_ids)) or not set(c_ids) <= primary_ids:
        raise QualificationValidationError("C packet contains a duplicate or non-primary candidate")
    return _validate_review_rows(c_review_rows, c_ids, role="C")


def adjudicate_reviews(
    package: Mapping[str, Any],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    reviewer_c_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary = _validate_primary_package(package)
    validated_a = _validate_review_rows(
        reviewer_a_rows, _package_review_order(package, "A"), role="A"
    )
    validated_b = _validate_review_rows(
        reviewer_b_rows, _package_review_order(package, "B"), role="B"
    )
    c_order = [
        _require_opaque_id(row.get("opaque_candidate_id"), context="C review ID")
        for row in reviewer_c_rows
    ]
    validated_c = _validate_review_rows(reviewer_c_rows, c_order, role="C")
    map_a = _review_map(validated_a)
    map_b = _review_map(validated_b)
    map_c = _review_map(validated_c)
    disagreements = _candidate_label_disagreements(
        primary, reviewer_a_rows, reviewer_b_rows
    )
    if set(map_c) != set(disagreements):
        raise QualificationValidationError("C review is incomplete or contains non-disagreement candidates")
    final: list[dict[str, Any]] = []
    for candidate in primary:
        candidate_id = candidate["opaque_candidate_id"]
        row: dict[str, Any] = {"opaque_candidate_id": candidate_id}
        label_a = candidate_qualified(map_a[candidate_id])
        label_b = candidate_qualified(map_b[candidate_id])
        if label_a != label_b:
            for field in DECISION_FIELDS:
                row[field] = map_c[candidate_id][field]
            row["candidate_qualified"] = candidate_qualified(map_c[candidate_id])
            row["adjudicated_fields"] = list(DECISION_FIELDS)
        else:
            row["detail_retention"] = min(
                map_a[candidate_id]["detail_retention"],
                map_b[candidate_id]["detail_retention"],
            )
            for field in BINARY_FIELDS:
                row[field] = (
                    "yes"
                    if map_a[candidate_id][field]
                    == map_b[candidate_id][field]
                    == "yes"
                    else "no"
                )
            row["candidate_qualified"] = label_a
            row["adjudicated_fields"] = []
            if candidate_qualified(row) != label_a:
                raise QualificationValidationError(
                    "conservative A/B vector changed an agreed candidate label"
                )
        final.append(row)
    return final


def _normalize_severity_minima(values: Mapping[Any, Any]) -> dict[int, int]:
    if type(values) is not dict:
        raise QualificationValidationError("severity minima must be an object")
    result: dict[int, int] = {}
    for key, value in values.items():
        if type(key) is int:
            severity = key
        elif type(key) is str and key in {"1", "2", "3"}:
            severity = int(key)
        else:
            raise QualificationValidationError("severity-minimum keys must be 1, 2, or 3")
        if severity in result or type(value) is not int or value < 0:
            raise QualificationValidationError("invalid severity minimum")
        result[severity] = value
    if set(result) != {1, 2, 3}:
        raise QualificationValidationError("severity minima must cover exactly 1, 2, and 3")
    return result


def summarize_yield(
    package: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> dict[str, Any]:
    primary = validate_candidate_package(
        package,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    if len(final_rows) != PRIMARY_CANDIDATES:
        raise QualificationValidationError("final labels must contain exactly 120 candidates")
    final_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(final_rows, 1):
        _require_exact_fields(row, FINAL_FIELDS, context=f"final row {index}")
        candidate_id = _require_opaque_id(row["opaque_candidate_id"], context="final candidate ID")
        if candidate_id in final_by_id:
            raise QualificationValidationError("final candidate IDs are duplicated")
        for field in DECISION_FIELDS:
            if field == "detail_retention":
                if type(row[field]) is not int or row[field] not in {0, 1, 2}:
                    raise QualificationValidationError("invalid final detail_retention")
            elif type(row[field]) is not str or row[field] not in {"yes", "no"}:
                raise QualificationValidationError(f"invalid final {field}")
        if row["candidate_qualified"] != candidate_qualified(row):
            raise QualificationValidationError("stored candidate_qualified is inconsistent")
        adjudicated = row["adjudicated_fields"]
        if type(adjudicated) is not list or len(adjudicated) != len(set(adjudicated)) or not set(adjudicated) <= set(DECISION_FIELDS):
            raise QualificationValidationError("invalid adjudicated_fields")
        final_by_id[candidate_id] = row
    primary_ids = {row["opaque_candidate_id"] for row in primary}
    if set(final_by_id) != primary_ids:
        raise QualificationValidationError("final-label ID set differs from primary candidates")

    minimum_qualified_prompts = MIN_QUALIFIED_PROMPTS
    category_cfg = dict(DEFAULT_CATEGORY_MINIMA)
    if not category_cfg or any(type(key) is not str or not key or type(value) is not int or value < 0 for key, value in category_cfg.items()):
        raise QualificationValidationError("invalid category minima")
    severity_cfg = _normalize_severity_minima(dict(DEFAULT_SEVERITY_MINIMA))

    by_prompt: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    prompt_meta: dict[str, tuple[str, int]] = {}
    for candidate in primary:
        prompt_id = candidate["opaque_prompt_id"]
        role = candidate["author_role"]
        if role in by_prompt[prompt_id]:
            raise QualificationValidationError("prompt has duplicate author candidate")
        by_prompt[prompt_id][role] = candidate
        metadata = (candidate["assigned_category"], candidate["severity_level"])
        if prompt_id in prompt_meta and prompt_meta[prompt_id] != metadata:
            raise QualificationValidationError("candidate metadata differs within a prompt")
        prompt_meta[prompt_id] = metadata
    observed_categories = {category for category, _ in prompt_meta.values()}
    observed_severities = {severity for _, severity in prompt_meta.values()}
    if set(category_cfg) != observed_categories:
        raise QualificationValidationError("category minima do not exactly cover frozen prompt strata")
    if set(severity_cfg) != observed_severities:
        raise QualificationValidationError("severity minima do not exactly cover frozen prompt strata")

    author_counts = {"A": 0, "B": 0}
    table = Counter({"neither": 0, "a_only": 0, "b_only": 0, "both": 0})
    qualified_prompt_ids: list[str] = []
    category_counts = Counter()
    severity_counts = Counter()
    selected_pool: list[dict[str, str]] = []
    selection_salt = derive_salt(
        protocol_commit,
        beacon_round,
        beacon_randomness,
        "qualified_target_selection",
    )
    for prompt_id in sorted(by_prompt):
        if set(by_prompt[prompt_id]) != {"A", "B"}:
            raise QualificationValidationError("each prompt must have Author A and Author B candidates")
        qualified = {
            role: final_by_id[by_prompt[prompt_id][role]["opaque_candidate_id"]]["candidate_qualified"] == "yes"
            for role in ("A", "B")
        }
        author_counts["A"] += int(qualified["A"])
        author_counts["B"] += int(qualified["B"])
        cell = "both" if qualified["A"] and qualified["B"] else "a_only" if qualified["A"] else "b_only" if qualified["B"] else "neither"
        table[cell] += 1
        eligible = [
            by_prompt[prompt_id][role]
            for role in ("A", "B")
            if qualified[role]
        ]
        if eligible:
            qualified_prompt_ids.append(prompt_id)
            category, severity = prompt_meta[prompt_id]
            category_counts[category] += 1
            severity_counts[severity] += 1
            selected = min(
                eligible,
                key=lambda row: _rank(selection_salt, prompt_id, row["opaque_candidate_id"]),
            )
            selected_pool.append(
                {
                    "opaque_prompt_id": prompt_id,
                    "opaque_candidate_id": selected["opaque_candidate_id"],
                }
            )
    all_candidate_count = author_counts["A"] + author_counts["B"]
    best_of_two_count = len(qualified_prompt_ids)
    category_gates = {category: category_counts[category] >= minimum for category, minimum in category_cfg.items()}
    severity_gates = {str(severity): severity_counts[severity] >= minimum for severity, minimum in severity_cfg.items()}
    gates = {
        "minimum_qualified_prompts_pass": best_of_two_count >= minimum_qualified_prompts,
        "all_category_minima_pass": all(category_gates.values()),
        "all_severity_minima_pass": all(severity_gates.values()),
    }
    return {
        "author_a": {"qualified": author_counts["A"], "denominator": 60, "rate": author_counts["A"] / 60},
        "author_b": {"qualified": author_counts["B"], "denominator": 60, "rate": author_counts["B"] / 60},
        "all_candidates": {"qualified": all_candidate_count, "denominator": 120, "rate": all_candidate_count / 120},
        "best_of_two": {"qualified_prompts": best_of_two_count, "denominator": 60, "rate": best_of_two_count / 60},
        "author_2x2": {key: table[key] for key in ("neither", "a_only", "b_only", "both")},
        "sampling_interval": None,
        "qualified_by_category": {category: category_counts[category] for category in sorted(category_cfg)},
        "qualified_by_severity": {str(severity): severity_counts[severity] for severity in sorted(severity_cfg)},
        "category_gates": category_gates,
        "severity_gates": severity_gates,
        "selected_machine_rubric_candidate_ids": selected_pool,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "interpretation": "descriptive fixed-pool best-of-two yield under exactly two attempts; no sampling interval, population-rate claim, or single-generation-rate claim",
    }


def post_gate_adjudication_diagnostics(
    package: Mapping[str, Any],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary = _validate_primary_package(package)
    map_a = _review_map(reviewer_a_rows)
    map_b = _review_map(reviewer_b_rows)
    final_map = _review_map(final_rows)
    primary_ids = {row["opaque_candidate_id"] for row in primary}
    if not primary_ids <= set(map_a) or not primary_ids <= set(map_b):
        raise QualificationValidationError("post-gate diagnostics lack A/B primary labels")
    if set(final_map) != primary_ids:
        raise QualificationValidationError("post-gate diagnostics lack final primary labels")

    by_severity = {
        str(severity): Counter(
            {
                "candidate_rows": 0,
                "ab_candidate_label_disagreements": 0,
                "c_tiebreak_qualified": 0,
                "ab_both_no_but_final_yes": 0,
            }
        )
        for severity in (1, 2, 3)
    }
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    prompt_severity: dict[str, int] = {}
    for candidate in primary:
        candidate_id = candidate["opaque_candidate_id"]
        prompt_id = candidate["opaque_prompt_id"]
        severity = candidate["severity_level"]
        key = str(severity)
        label_a = candidate_qualified(map_a[candidate_id])
        label_b = candidate_qualified(map_b[candidate_id])
        final_label = final_map[candidate_id]["candidate_qualified"]
        by_severity[key]["candidate_rows"] += 1
        by_severity[key]["ab_candidate_label_disagreements"] += int(
            label_a != label_b
        )
        by_severity[key]["c_tiebreak_qualified"] += int(
            label_a != label_b and final_label == "yes"
        )
        by_severity[key]["ab_both_no_but_final_yes"] += int(
            label_a == label_b == "no" and final_label == "yes"
        )
        by_prompt[prompt_id].append(label_a == label_b == "yes")
        if prompt_id in prompt_severity and prompt_severity[prompt_id] != severity:
            raise QualificationValidationError("prompt severity differs across candidates")
        prompt_severity[prompt_id] = severity

    both_no_rescues = sum(
        counts["ab_both_no_but_final_yes"] for counts in by_severity.values()
    )
    if both_no_rescues != DEFAULT_REVIEW_GATES["ab_both_no_final_yes_max_count"]:
        raise QualificationValidationError(
            "C adjudication converted an A/B both-no candidate into final yes"
        )
    conservative_prompt_ids = {
        prompt_id for prompt_id, labels in by_prompt.items() if any(labels)
    }
    conservative_by_severity = Counter(
        prompt_severity[prompt_id] for prompt_id in conservative_prompt_ids
    )
    return {
        "candidate_level_c_only": True,
        "ab_both_no_but_final_yes": both_no_rescues,
        "by_severity": {
            severity: dict(counts) for severity, counts in by_severity.items()
        },
        "conservative_ab_only_best_of_two": {
            "qualified_prompts": len(conservative_prompt_ids),
            "denominator": PROMPT_CLUSTERS,
            "rate": len(conservative_prompt_ids) / PROMPT_CLUSTERS,
            "qualified_by_severity": {
                str(severity): conservative_by_severity[severity]
                for severity in (1, 2, 3)
            },
            "definition": "a prompt qualifies only if at least one candidate was candidate_qualified=yes for both A and B before C",
        },
    }


def official_route(
    *,
    protocol_integrity_valid: bool,
    prompt_pool_valid: bool,
    authoring_valid: bool,
    review_inputs_valid: bool,
    reliability_pass: bool,
    target_yield_pass: bool,
) -> str:
    flags = {
        "protocol_integrity_valid": protocol_integrity_valid,
        "prompt_pool_valid": prompt_pool_valid,
        "authoring_valid": authoring_valid,
        "review_inputs_valid": review_inputs_valid,
        "reliability_pass": reliability_pass,
        "target_yield_pass": target_yield_pass,
    }
    if any(type(value) is not bool for value in flags.values()):
        raise QualificationValidationError("route inputs must be strict booleans")
    if not protocol_integrity_valid:
        return "STOP_PROTOCOL_INTEGRITY"
    if not prompt_pool_valid:
        return "STOP_PROMPT_POOL"
    if not authoring_valid:
        return "STOP_AUTHORING"
    if not review_inputs_valid:
        return "STOP_REVIEW_INVALID"
    if not reliability_pass:
        return "STOP_REVIEW_UNRELIABLE"
    if not target_yield_pass:
        return "STOP_TARGET_YIELD"
    return "PASS_MACHINE_RUBRIC_POOL"


def analyze_validated_qualification(
    package: Mapping[str, Any],
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    c_packet: Sequence[Mapping[str, Any]],
    reviewer_c_rows: Sequence[Mapping[str, Any]],
    provenance_records: Mapping[str, Mapping[str, Any]],
    expected_provenance_inputs: Mapping[str, Mapping[str, str]],
    *,
    protocol_commit: str,
    beacon_round: int,
    beacon_randomness: str,
) -> dict[str, Any]:
    reliability = compute_ab_reliability(
        package,
        reviewer_a_rows,
        reviewer_b_rows,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    if not reliability["overall_pass"]:
        raise QualificationValidationError(
            "A/B reliability failed; C adjudication and hidden-key yield analysis are forbidden"
        )
    validated_c = validate_c_packet_and_reviews(
        c_packet,
        reviewer_c_rows,
        package,
        reviewer_a_rows,
        reviewer_b_rows,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    agent_ids = validate_session_provenance_bundle(
        provenance_records,
        expected_provenance_inputs,
        reviewer_c_row_count=len(validated_c),
    )
    final = adjudicate_reviews(package, reviewer_a_rows, reviewer_b_rows, validated_c)
    yield_summary = summarize_yield(
        package,
        final,
        protocol_commit=protocol_commit,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    post_gate = post_gate_adjudication_diagnostics(
        package, reviewer_a_rows, reviewer_b_rows, final
    )
    route = official_route(
        protocol_integrity_valid=True,
        prompt_pool_valid=True,
        authoring_valid=True,
        review_inputs_valid=True,
        reliability_pass=True,
        target_yield_pass=yield_summary["overall_pass"],
    )
    return {
        "analysis_pass": route == "PASS_MACHINE_RUBRIC_POOL",
        "route": route,
        "reliability": {**reliability, "post_gate_diagnostics": post_gate},
        "post_gate_diagnostics": post_gate,
        "session_agent_ids": agent_ids,
        "final_labels": final,
        "yield": yield_summary,
        "training_authorized": False,
        "optimizer_steps": 0,
    }


__all__ = [
    "ATTEMPT_ID",
    "PROMPT_REWEIGHTING_REPLICATES",
    "DECISION_FIELDS",
    "DEFAULT_CATEGORY_MINIMA",
    "DEFAULT_REVIEW_GATES",
    "DEFAULT_SEVERITY_MINIMA",
    "MAX_RESPONSE_TOKENS",
    "MIN_QUALIFIED_PROMPTS",
    "MIN_RESPONSE_TOKENS",
    "OFFICIAL_ROUTES",
    "PROHIBITED_ACCESS_FIELDS",
    "RANDOMIZATION_PURPOSES",
    "QualificationValidationError",
    "adjudicate_reviews",
    "binary_agreement_metrics",
    "build_c_packet",
    "build_candidate_package",
    "build_provenance_record",
    "candidate_qualified",
    "compute_ab_reliability",
    "decode_utf8_lf",
    "derive_salt",
    "detail_agreement_metrics",
    "duplicate_consistency",
    "analyze_validated_qualification",
    "hash_inputs",
    "normalize_response",
    "official_route",
    "parse_json",
    "parse_jsonl",
    "sha256_bytes",
    "summarize_yield",
    "stratified_prompt_reweighting_ac1",
    "validate_agent_isolation",
    "validate_author_jsonl",
    "validate_author_jsonl_bytes",
    "validate_author_pair",
    "validate_c_packet_and_reviews",
    "validate_candidate_package",
    "validate_provenance_json",
    "validate_provenance_record",
    "validate_randomization_context",
    "validate_reviewer_jsonl",
    "validate_reviewer_jsonl_bytes",
    "validate_session_provenance_bundle",
    "whitespace_token_count",
]
