#!/usr/bin/env python3
"""Prepare and execute the frozen Step3Q calibrated API target-pool route."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "experiments"
    / "step3_response_quality_persistence"
    / "configs"
    / "step3q_calibrated_api_pool_registry.json"
)

COMPONENT_FIELDS = ("safe_route", "executable_step", "prompt_binding")
BINARY_FIELDS = COMPONENT_FIELDS + ("naturalness", "clean_safe")
DECISION_FIELDS = ("detail_retention",) + BINARY_FIELDS
AUTHOR_MODES = ("specific_redirect", "risk_mitigation")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class Step3QValidationError(ValueError):
    pass


class Step3QProvenanceError(Step3QValidationError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def whitespace_tokens(value: str) -> int:
    return len(value.split())


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise Step3QValidationError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Step3QValidationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise Step3QValidationError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Step3QValidationError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = read_json(path)
    if registry.get("schema") != "step3q_calibrated_api_pool_registry_v1":
        raise Step3QValidationError("unexpected Step3Q registry schema")
    if registry.get("training_authorized") is not False or registry.get("optimizer_steps") != 0:
        raise Step3QValidationError("registry crosses the frozen training boundary")
    if tuple(registry["authoring"]["modes"]) != AUTHOR_MODES:
        raise Step3QValidationError("author modes differ from the frozen implementation")
    calibration = registry["calibration"]
    role_counts = calibration.get("sentinel_role_counts")
    expected_roles = {"component_anchor", "prompt_mismatch", "clean_safety_anchor"}
    if not isinstance(role_counts, dict) or set(role_counts) != expected_roles:
        raise Step3QValidationError("sentinel role counts must explicitly cover the three frozen roles")
    if any(type(value) is not int or value <= 0 for value in role_counts.values()):
        raise Step3QValidationError("sentinel role counts must be positive integers")
    if sum(role_counts.values()) != int(calibration["sentinels_per_primary_reviewer"]):
        raise Step3QValidationError("sentinel role counts do not sum to the reviewer total")
    if role_counts["component_anchor"] < role_counts["prompt_mismatch"]:
        raise Step3QValidationError("each mismatch sentinel requires its paired component anchor")
    class_minimum = calibration.get("sentinel_component_minimum_reference_per_class")
    if type(class_minimum) is not int or class_minimum < 1:
        raise Step3QValidationError("sentinel component class minimum must be a positive integer")
    if 2 * class_minimum > role_counts["component_anchor"]:
        raise Step3QValidationError("sentinel component class minimum is impossible for the anchor quota")
    safety_counts = calibration.get("sentinel_clean_safety_reference_counts")
    if not isinstance(safety_counts, dict) or set(safety_counts) != {"yes", "no"}:
        raise Step3QValidationError("clean-safety sentinel counts must explicitly cover yes and no")
    if any(type(value) is not int or value < 1 for value in safety_counts.values()):
        raise Step3QValidationError("clean-safety sentinel counts must be positive integers")
    if sum(safety_counts.values()) != role_counts["clean_safety_anchor"]:
        raise Step3QValidationError("clean-safety class counts do not match the role quota")
    authoring = registry["authoring"]
    try:
        effective_model_pattern = re.compile(authoring["effective_model_regex"])
    except (KeyError, re.error) as exc:
        raise Step3QValidationError("invalid effective-model regular expression") from exc
    if effective_model_pattern.fullmatch(authoring["model"]) is None:
        raise Step3QValidationError("requested model does not satisfy its effective-model policy")
    if authoring.get("require_single_effective_model") is not True:
        raise Step3QValidationError("Step3Q requires one effective model across valid responses")
    attempts = int(authoring["maximum_attempts"])
    if attempts < 1 or len(authoring["retry_backoff_seconds"]) < attempts - 1:
        raise Step3QValidationError("retry schedule is shorter than maximum_attempts")
    return registry


def validate_input(name: str, spec: Mapping[str, Any]) -> Path:
    path = resolve_repo_path(str(spec["path"]))
    if not path.is_file():
        raise Step3QValidationError(f"missing frozen input {name}: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise Step3QValidationError(f"byte mismatch for frozen input {name}")
    if sha256_file(path) != spec["sha256"]:
        raise Step3QValidationError(f"SHA-256 mismatch for frozen input {name}")
    return path


def validate_frozen_inputs(registry: Mapping[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, spec in registry["inputs"].items():
        if isinstance(spec, dict) and {"path", "bytes", "sha256"} <= set(spec):
            result[name] = validate_input(name, spec)
    instructions = resolve_repo_path(registry["inputs"]["authoring_instructions_path"])
    if not instructions.is_file():
        raise Step3QValidationError("missing Step3Q authoring instructions")
    result["authoring_instructions"] = instructions
    return result


def stable_rank(namespace: str, purpose: str, *parts: Any) -> str:
    material = "\x1f".join([namespace, purpose, *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def selection_namespace(registry: Mapping[str, Any], protocol_commit: str) -> str:
    input_hashes = sorted(
        spec["sha256"]
        for spec in registry["inputs"].values()
        if isinstance(spec, dict) and "sha256" in spec
    )
    material = "\x1f".join(
        [registry["route_id"], registry["entry_commit"], protocol_commit, *input_hashes]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _strict_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise Step3QValidationError(f"{context} must be a strict boolean")
    return value


def load_source_rows(path: Path, registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(path)
    expected = int(registry["inputs"]["train_risk"]["rows"])
    if len(raw_rows) != expected:
        raise Step3QValidationError(f"unexpected train_risk rows: {len(raw_rows)} != {expected}")
    source_cfg = registry["source"]
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise Step3QValidationError(f"invalid or duplicate source sample_id at row {index}")
        seen_ids.add(sample_id)
        if row.get("family") != source_cfg["expected_family"]:
            raise Step3QValidationError(f"source family mismatch for {sample_id}")
        if row.get("source") != source_cfg["expected_source"]:
            raise Step3QValidationError(f"source name mismatch for {sample_id}")
        if _strict_bool(row.get("is_safe_source"), f"{sample_id}.is_safe_source") is not source_cfg["expected_is_safe_source"]:
            raise Step3QValidationError(f"source safety mismatch for {sample_id}")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Step3QValidationError(f"missing prompt for {sample_id}")
        severity = row.get("severity_level")
        if type(severity) is not int or severity not in {1, 2, 3}:
            raise Step3QValidationError(f"invalid severity for {sample_id}")
        harm = row.get("harm_category")
        if not isinstance(harm, dict) or not harm:
            raise Step3QValidationError(f"invalid harm categories for {sample_id}")
        categories = sorted(name for name, enabled in harm.items() if enabled is True)
        if not categories:
            raise Step3QValidationError(f"source row has no true harm category: {sample_id}")
        rows.append(
            {
                "sample_id": sample_id,
                "source_index": row.get("source_index"),
                "prompt": prompt.strip(),
                "prompt_tokens": whitespace_tokens(prompt),
                "severity_level": severity,
                "categories": categories,
            }
        )
    return rows


def load_exclusion_union(paths: Mapping[str, Path], registry: Mapping[str, Any]) -> tuple[set[str], dict[str, int]]:
    audit = read_json(paths["o4_source_audit"])
    registered = set(audit.get("excluded_source_ids", []))
    lexical = {
        row["sample_id"]
        for row in audit.get("restricted_lexical_exclusions", [])
        if isinstance(row, dict) and isinstance(row.get("sample_id"), str)
    }
    o4_rows = read_csv_rows(paths["o4_prompt_key"])
    if len(o4_rows) != int(registry["inputs"]["o4_prompt_key"]["rows"]):
        raise Step3QValidationError("O4 prompt-key row count mismatch")
    o4_prompts = {row["sample_id"] for row in o4_rows}
    if len(o4_prompts) != len(o4_rows):
        raise Step3QValidationError("O4 prompt-key sample IDs are not unique")
    step3p_rows = read_jsonl(paths["step3p_candidates"])
    if len(step3p_rows) != int(registry["inputs"]["step3p_candidates"]["rows"]):
        raise Step3QValidationError("Step3P candidate row count mismatch")
    step3p_prompts = {row.get("sample_id") for row in step3p_rows}
    if None in step3p_prompts or len(step3p_prompts) != int(
        registry["inputs"]["step3p_candidates"]["expected_unique_prompts"]
    ):
        raise Step3QValidationError("Step3P unique prompt count mismatch")
    union = registered | lexical | o4_prompts | step3p_prompts
    counts = {
        "o4_registered": len(registered),
        "o4_restricted_lexical": len(lexical),
        "o4_o4r_prompts": len(o4_prompts),
        "step3p_exposed_prompts": len(step3p_prompts),
        "union": len(union),
    }
    return union, counts


def eligible_source_rows(
    rows: Sequence[Mapping[str, Any]], exclusions: set[str], registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    minimum = int(registry["source"]["minimum_prompt_whitespace_tokens"])
    maximum = int(registry["source"]["maximum_prompt_whitespace_tokens"])
    return [
        dict(row)
        for row in rows
        if row["sample_id"] not in exclusions and minimum <= row["prompt_tokens"] <= maximum
    ]


def expand_slots(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    table = registry["selection"]["category_severity_slots"]
    for category in sorted(table):
        for severity_text in ("1", "2", "3"):
            count = int(table[category][severity_text])
            for ordinal in range(1, count + 1):
                slots.append(
                    {
                        "slot_id": f"{category}|severity={severity_text}|slot={ordinal}",
                        "category": category,
                        "severity": int(severity_text),
                    }
                )
    if len(slots) != int(registry["selection"]["prompt_count"]):
        raise Step3QValidationError("category-severity slots do not expand to prompt_count")
    observed = Counter(str(slot["severity"]) for slot in slots)
    expected = {key: int(value) for key, value in registry["selection"]["expected_severity_counts"].items()}
    if dict(observed) != expected:
        raise Step3QValidationError("expanded severity counts differ from the registry")
    return slots


def deterministic_match(
    rows: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]], namespace: str
) -> tuple[dict[str, str], dict[str, int]]:
    candidates: dict[str, list[str]] = {}
    for slot in slots:
        eligible = [
            row["sample_id"]
            for row in rows
            if slot["category"] in row["categories"] and slot["severity"] == row["severity_level"]
        ]
        eligible.sort(key=lambda sample_id: stable_rank(namespace, "prompt-match", slot["slot_id"], sample_id))
        if not eligible:
            raise Step3QValidationError(f"no fresh candidate for slot {slot['slot_id']}")
        candidates[slot["slot_id"]] = eligible
    slot_order = sorted(candidates, key=lambda slot_id: (len(candidates[slot_id]), slot_id))
    candidate_to_slot: dict[str, str] = {}
    slot_to_candidate: dict[str, str] = {}

    def augment(slot_id: str, seen: set[str]) -> bool:
        for sample_id in candidates[slot_id]:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            incumbent = candidate_to_slot.get(sample_id)
            if incumbent is None or augment(incumbent, seen):
                candidate_to_slot[sample_id] = slot_id
                slot_to_candidate[slot_id] = sample_id
                return True
        return False

    for slot_id in slot_order:
        if not augment(slot_id, set()):
            raise Step3QValidationError(f"fresh prompt matching failed at {slot_id}")
    if len(slot_to_candidate) != len(slots):
        raise Step3QValidationError("fresh prompt matching is incomplete")
    return slot_to_candidate, {slot_id: len(values) for slot_id, values in candidates.items()}


def _packet_map(rows: Sequence[Mapping[str, Any]], id_field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get(id_field)
        if not isinstance(row_id, str) or not row_id or row_id in result:
            raise Step3QValidationError(f"invalid or duplicate {id_field}")
        result[row_id] = dict(row)
    return result


def select_balanced_component_anchors(
    labels: Sequence[Mapping[str, Any]], count: int, minimum_per_class: int, namespace: str
) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in labels]
    for row in remaining:
        for field in COMPONENT_FIELDS:
            if row.get(field) not in {"yes", "no"}:
                raise Step3QValidationError(f"invalid O3 component label for {field}")
    deficits = {(field, value): minimum_per_class for field in COMPONENT_FIELDS for value in ("yes", "no")}
    selected: list[dict[str, Any]] = []
    while remaining and any(value > 0 for value in deficits.values()):
        ranked = sorted(
            remaining,
            key=lambda row: (
                -sum(deficits[(field, row[field])] > 0 for field in COMPONENT_FIELDS),
                stable_rank(namespace, "component-anchor", row["opaque_action_id"]),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        remaining.remove(chosen)
        for field in COMPONENT_FIELDS:
            key = (field, chosen[field])
            deficits[key] = max(0, deficits[key] - 1)
    if any(deficits.values()):
        raise Step3QValidationError(f"component-anchor deficits remain: {deficits}")
    remaining.sort(key=lambda row: stable_rank(namespace, "component-anchor-fill", row["opaque_action_id"]))
    selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise Step3QValidationError("component-anchor selection did not reach the fixed count")
    for field in COMPONENT_FIELDS:
        counts = Counter(row[field] for row in selected)
        if any(counts[value] < minimum_per_class for value in ("yes", "no")):
            raise Step3QValidationError(f"component-anchor class minimum failed for {field}")
    return selected


def _deterministic_category_derangement(
    bases: Sequence[Mapping[str, Any]], donors: Sequence[Mapping[str, Any]], namespace: str
) -> dict[str, str]:
    donor_candidates: dict[str, list[str]] = {}
    donor_by_id = {row["opaque_action_id"]: row for row in donors}
    for base in bases:
        base_id = base["opaque_action_id"]
        values = [
            donor_id
            for donor_id, donor in donor_by_id.items()
            if donor_id != base_id and donor["assigned_category"] != base["assigned_category"]
        ]
        values.sort(key=lambda donor_id: stable_rank(namespace, "mismatch-donor", base_id, donor_id))
        if not values:
            raise Step3QValidationError(f"no category-distinct mismatch donor for {base_id}")
        donor_candidates[base_id] = values
    donor_to_base: dict[str, str] = {}
    base_to_donor: dict[str, str] = {}

    def augment(base_id: str, seen: set[str]) -> bool:
        for donor_id in donor_candidates[base_id]:
            if donor_id in seen:
                continue
            seen.add(donor_id)
            incumbent = donor_to_base.get(donor_id)
            if incumbent is None or augment(incumbent, seen):
                donor_to_base[donor_id] = base_id
                base_to_donor[base_id] = donor_id
                return True
        return False

    for base_id in sorted(donor_candidates, key=lambda value: (len(donor_candidates[value]), value)):
        if not augment(base_id, set()):
            raise Step3QValidationError("cannot construct category-distinct mismatch derangement")
    return base_to_donor


def build_calibration_bank(
    paths: Mapping[str, Path], registry: Mapping[str, Any], namespace: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg = registry["calibration"]
    component_packet = _packet_map(read_jsonl(paths["o3_component_packet"]), "opaque_action_id")
    component_key = _packet_map(read_csv_rows(paths["o3_component_key"]), "opaque_action_id")
    component_labels = read_jsonl(paths["o3_component_labels"])
    if len(component_labels) != int(registry["inputs"]["o3_component_labels"]["rows"]):
        raise Step3QValidationError("O3 component-label row count mismatch")
    anchors = select_balanced_component_anchors(
        component_labels,
        int(cfg["component_anchor_rows"]),
        int(cfg["component_minimum_reference_per_class"]),
        namespace,
    )
    enriched: list[dict[str, Any]] = []
    for label in anchors:
        source_id = label["opaque_action_id"]
        if source_id not in component_packet or source_id not in component_key:
            raise Step3QValidationError(f"O3 component anchor lacks packet/key row: {source_id}")
        visible = component_packet[source_id]
        key = component_key[source_id]
        category = key.get("assigned_category")
        if not isinstance(category, str) or not category:
            raise Step3QValidationError(f"O3 component anchor lacks assigned category: {source_id}")
        enriched.append({**label, "prompt": visible["prompt"], "response": visible["response"], "assigned_category": category})

    anchor_id_by_source: dict[str, str] = {}
    visible_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for row in enriched:
        source_id = row["opaque_action_id"]
        opaque_id = "step3q_cal_" + stable_rank(namespace, "calibration-anchor-id", source_id)[:24]
        anchor_id_by_source[source_id] = opaque_id
        visible_rows.append({"opaque_candidate_id": opaque_id, "prompt": row["prompt"], "response": row["response"]})
        reference_rows.append(
            {
                "opaque_candidate_id": opaque_id,
                "calibration_role": "component_anchor",
                "historical_source_id": source_id,
                "expected": {field: row[field] for field in COMPONENT_FIELDS},
                "assigned_category": row["assigned_category"],
            }
        )

    mismatch_count = int(cfg["mismatch_rows"])
    bases = sorted(
        enriched,
        key=lambda row: stable_rank(namespace, "mismatch-base", row["opaque_action_id"]),
    )[:mismatch_count]
    assignment = _deterministic_category_derangement(bases, enriched, namespace)
    enriched_by_id = {row["opaque_action_id"]: row for row in enriched}
    for base in bases:
        base_id = base["opaque_action_id"]
        donor_id = assignment[base_id]
        donor = enriched_by_id[donor_id]
        opaque_id = "step3q_cal_" + stable_rank(namespace, "calibration-mismatch-id", base_id, donor_id)[:24]
        visible_rows.append({"opaque_candidate_id": opaque_id, "prompt": base["prompt"], "response": donor["response"]})
        reference_rows.append(
            {
                "opaque_candidate_id": opaque_id,
                "calibration_role": "prompt_mismatch",
                "matched_anchor_candidate_id": anchor_id_by_source[base_id],
                "base_historical_source_id": base_id,
                "donor_historical_source_id": donor_id,
                "base_assigned_category": base["assigned_category"],
                "donor_assigned_category": donor["assigned_category"],
                "expected": {"detail_retention": 0, "prompt_binding": "no"},
            }
        )

    safety_packet = _packet_map(read_jsonl(paths["o3_safety_packet"]), "opaque_safety_id")
    safety_key = _packet_map(read_csv_rows(paths["o3_safety_key"]), "opaque_safety_id")
    safety_labels = read_jsonl(paths["o3_safety_labels"])
    if len(safety_labels) != int(registry["inputs"]["o3_safety_labels"]["rows"]):
        raise Step3QValidationError("O3 clean-safety label row count mismatch")
    selected_safety: list[dict[str, Any]] = []
    for value, count_key in (("yes", "clean_safety_yes_rows"), ("no", "clean_safety_no_rows")):
        pool = [row for row in safety_labels if row.get("clean_safe") == value]
        pool.sort(key=lambda row: stable_rank(namespace, "clean-safety-anchor", value, row["opaque_safety_id"]))
        needed = int(cfg[count_key])
        if len(pool) < needed:
            raise Step3QValidationError(f"insufficient historical clean-safety {value} anchors")
        selected_safety.extend(pool[:needed])
    for label in selected_safety:
        source_id = label["opaque_safety_id"]
        if source_id not in safety_packet or source_id not in safety_key:
            raise Step3QValidationError(f"O3 clean-safety anchor lacks packet/key row: {source_id}")
        packet = safety_packet[source_id]
        opaque_id = "step3q_cal_" + stable_rank(namespace, "calibration-safety-id", source_id)[:24]
        visible_rows.append({"opaque_candidate_id": opaque_id, "prompt": packet["prompt"], "response": packet["response"]})
        reference_rows.append(
            {
                "opaque_candidate_id": opaque_id,
                "calibration_role": "clean_safety_anchor",
                "historical_source_id": source_id,
                "expected": {"clean_safe": label["clean_safe"]},
            }
        )

    if len(visible_rows) != int(cfg["bank_rows"]) or len(reference_rows) != len(visible_rows):
        raise Step3QValidationError("calibration bank does not have the frozen row count")
    if len({row["opaque_candidate_id"] for row in visible_rows}) != len(visible_rows):
        raise Step3QValidationError("calibration bank has duplicate opaque IDs")
    visible_rows.sort(key=lambda row: stable_rank(namespace, "calibration-bank-order", row["opaque_candidate_id"]))
    reference_rows.sort(key=lambda row: row["opaque_candidate_id"])
    role_counts = Counter(row["calibration_role"] for row in reference_rows)
    component_counts = {
        field: dict(Counter(row["expected"][field] for row in reference_rows if row["calibration_role"] == "component_anchor"))
        for field in COMPONENT_FIELDS
    }
    summary = {
        "rows": len(visible_rows),
        "role_counts": dict(role_counts),
        "component_reference_counts": component_counts,
        "mismatch_category_distinct": all(
            row["base_assigned_category"] != row["donor_assigned_category"]
            for row in reference_rows
            if row["calibration_role"] == "prompt_mismatch"
        ),
        "clean_safety_reference_counts": dict(
            Counter(
                row["expected"]["clean_safe"]
                for row in reference_rows
                if row["calibration_role"] == "clean_safety_anchor"
            )
        ),
    }
    return visible_rows, reference_rows, summary


def build_sentinel_allocations(
    reference_rows: Sequence[Mapping[str, Any]], registry: Mapping[str, Any], namespace: str
) -> dict[str, list[str]]:
    role_counts = registry["calibration"]["sentinel_role_counts"]
    reference_by_id = {row["opaque_candidate_id"]: dict(row) for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise Step3QValidationError("calibration references contain duplicate opaque IDs")
    allocations: dict[str, list[str]] = {}
    for reviewer in ("reviewer_a", "reviewer_b"):
        mismatch_pool = [
            row for row in reference_rows if row["calibration_role"] == "prompt_mismatch"
        ]
        selected_mismatches = select_balanced_mismatch_sentinels(
            mismatch_pool,
            reference_by_id,
            int(role_counts["prompt_mismatch"]),
            int(registry["calibration"]["sentinel_component_minimum_reference_per_class"]),
            namespace,
            reviewer,
        )
        paired_anchors = [row["matched_anchor_candidate_id"] for row in selected_mismatches]
        if len(set(paired_anchors)) != len(paired_anchors):
            raise Step3QValidationError(f"mismatch sentinels do not have unique paired anchors for {reviewer}")
        if any(
            anchor_id not in reference_by_id
            or reference_by_id[anchor_id]["calibration_role"] != "component_anchor"
            for anchor_id in paired_anchors
        ):
            raise Step3QValidationError(f"mismatch sentinel lacks a valid paired anchor for {reviewer}")

        component_needed = int(role_counts["component_anchor"])
        if len(paired_anchors) > component_needed:
            raise Step3QValidationError(f"component-anchor quota cannot carry mismatch pairs for {reviewer}")
        component_pool = [
            row["opaque_candidate_id"]
            for row in reference_rows
            if row["calibration_role"] == "component_anchor"
            and row["opaque_candidate_id"] not in set(paired_anchors)
        ]
        component_pool.sort(
            key=lambda value: stable_rank(
                namespace, "sentinel-allocation", reviewer, "component_anchor", value
            )
        )
        extra_components = component_pool[: component_needed - len(paired_anchors)]
        if len(paired_anchors) + len(extra_components) != component_needed:
            raise Step3QValidationError(f"insufficient component_anchor sentinels for {reviewer}")

        selected_safety: list[str] = []
        for value, needed_value in sorted(
            registry["calibration"]["sentinel_clean_safety_reference_counts"].items()
        ):
            safety_pool = [
                row["opaque_candidate_id"]
                for row in reference_rows
                if row["calibration_role"] == "clean_safety_anchor"
                and row["expected"]["clean_safe"] == value
            ]
            safety_pool.sort(
                key=lambda candidate_id: stable_rank(
                    namespace,
                    "sentinel-allocation",
                    reviewer,
                    f"clean_safety_anchor={value}",
                    candidate_id,
                )
            )
            needed = int(needed_value)
            if len(safety_pool) < needed:
                raise Step3QValidationError(
                    f"insufficient clean_safety_anchor={value} sentinels for {reviewer}"
                )
            selected_safety.extend(safety_pool[:needed])

        selected = paired_anchors + extra_components
        selected.extend(row["opaque_candidate_id"] for row in selected_mismatches)
        selected.extend(selected_safety)
        expected_total = int(registry["calibration"]["sentinels_per_primary_reviewer"])
        if len(selected) != expected_total or len(set(selected)) != expected_total:
            raise Step3QValidationError(f"invalid sentinel allocation for {reviewer}")
        selected.sort(key=lambda value: stable_rank(namespace, "sentinel-packet-order", reviewer, value))
        allocations[reviewer] = selected
    return allocations


def select_balanced_mismatch_sentinels(
    mismatch_rows: Sequence[Mapping[str, Any]],
    reference_by_id: Mapping[str, Mapping[str, Any]],
    count: int,
    minimum_per_class: int,
    namespace: str,
    reviewer: str,
) -> list[dict[str, Any]]:
    ordered = [dict(row) for row in mismatch_rows]
    ordered.sort(
        key=lambda row: stable_rank(
            namespace, "sentinel-allocation", reviewer, "prompt_mismatch", row["opaque_candidate_id"]
        )
    )
    if len(ordered) < count:
        raise Step3QValidationError(f"insufficient prompt_mismatch sentinels for {reviewer}")

    anchor_labels: dict[str, dict[str, str]] = {}
    for mismatch in ordered:
        anchor_id = mismatch.get("matched_anchor_candidate_id")
        anchor = reference_by_id.get(anchor_id)
        if anchor is None or anchor.get("calibration_role") != "component_anchor":
            raise Step3QValidationError(f"mismatch sentinel lacks a valid paired anchor for {reviewer}")
        expected = anchor.get("expected")
        if not isinstance(expected, dict) or any(expected.get(field) not in {"yes", "no"} for field in COMPONENT_FIELDS):
            raise Step3QValidationError(f"paired anchor has invalid component references for {reviewer}")
        anchor_labels[mismatch["opaque_candidate_id"]] = {
            field: expected[field] for field in COMPONENT_FIELDS
        }

    counts = Counter()
    selected: list[dict[str, Any]] = []

    def can_still_satisfy(start: int) -> bool:
        slots_left = count - len(selected)
        if len(ordered) - start < slots_left:
            return False
        remaining = ordered[start:]
        for field in COMPONENT_FIELDS:
            for value in ("yes", "no"):
                available = sum(
                    anchor_labels[row["opaque_candidate_id"]][field] == value for row in remaining
                )
                if counts[(field, value)] + min(slots_left, available) < minimum_per_class:
                    return False
        return True

    def search(start: int) -> bool:
        if len(selected) == count:
            return all(
                counts[(field, value)] >= minimum_per_class
                for field in COMPONENT_FIELDS
                for value in ("yes", "no")
            )
        if not can_still_satisfy(start):
            return False
        slots_left = count - len(selected)
        last_start = len(ordered) - slots_left
        for index in range(start, last_start + 1):
            row = ordered[index]
            labels = anchor_labels[row["opaque_candidate_id"]]
            selected.append(row)
            for field in COMPONENT_FIELDS:
                counts[(field, labels[field])] += 1
            if search(index + 1):
                return True
            for field in COMPONENT_FIELDS:
                counts[(field, labels[field])] -= 1
            selected.pop()
        return False

    if not search(0):
        raise Step3QValidationError(
            f"cannot allocate class-balanced paired mismatch sentinels for {reviewer}"
        )
    return selected


def summarize_sentinel_allocations(
    reference_rows: Sequence[Mapping[str, Any]], allocations: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    reference_by_id = {row["opaque_candidate_id"]: dict(row) for row in reference_rows}
    summary: dict[str, Any] = {}
    for reviewer, allocation in allocations.items():
        selected = [reference_by_id[row_id] for row_id in allocation]
        component_rows = [row for row in selected if row["calibration_role"] == "component_anchor"]
        safety_rows = [row for row in selected if row["calibration_role"] == "clean_safety_anchor"]
        selected_ids = set(allocation)
        summary[reviewer] = {
            "rows": len(selected),
            "role_counts": dict(Counter(row["calibration_role"] for row in selected)),
            "component_reference_counts": {
                field: dict(Counter(row["expected"][field] for row in component_rows))
                for field in COMPONENT_FIELDS
            },
            "clean_safety_reference_counts": dict(
                Counter(row["expected"]["clean_safe"] for row in safety_rows)
            ),
            "mismatch_pair_closure": all(
                row["matched_anchor_candidate_id"] in selected_ids
                for row in selected
                if row["calibration_role"] == "prompt_mismatch"
            ),
        }
    return summary


def build_prompt_and_request_rows(
    source_rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, str], slots: Sequence[Mapping[str, Any]], namespace: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_id = {row["sample_id"]: row for row in source_rows}
    slot_by_id = {slot["slot_id"]: slot for slot in slots}
    prompts: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for slot_id, sample_id in sorted(assignment.items()):
        row = source_by_id[sample_id]
        slot = slot_by_id[slot_id]
        prompt_id = "step3q_prompt_" + stable_rank(namespace, "opaque-prompt-id", sample_id)[:20]
        prompt_record = {
            "opaque_prompt_id": prompt_id,
            "sample_id": sample_id,
            "source_index": row["source_index"],
            "prompt": row["prompt"],
            "prompt_sha256": sha256_bytes(row["prompt"].encode("utf-8")),
            "prompt_whitespace_tokens": row["prompt_tokens"],
            "assigned_category": slot["category"],
            "severity_level": slot["severity"],
            "all_source_categories": row["categories"],
            "slot_id": slot_id,
        }
        prompts.append(prompt_record)
        for mode in AUTHOR_MODES:
            request_id = "step3q_req_" + stable_rank(namespace, "api-request-id", prompt_id, mode)[:24]
            candidate_id = "step3q_candidate_" + stable_rank(namespace, "candidate-id", prompt_id, mode)[:24]
            requests.append(
                {
                    "opaque_request_id": request_id,
                    "opaque_candidate_id": candidate_id,
                    "opaque_prompt_id": prompt_id,
                    "author_mode": mode,
                    "prompt": row["prompt"],
                }
            )
    prompts.sort(key=lambda row: stable_rank(namespace, "prompt-package-order", row["opaque_prompt_id"]))
    requests.sort(key=lambda row: stable_rank(namespace, "api-request-order", row["opaque_request_id"]))
    if len(prompts) != 60 or len(requests) != 120:
        raise Step3QValidationError("fresh prompt/request package has the wrong size")
    return prompts, requests


def preflight(registry: Mapping[str, Any], protocol_commit: str = "PRE_P0_FEASIBILITY") -> dict[str, Any]:
    paths = validate_frozen_inputs(registry)
    source_rows = load_source_rows(paths["train_risk"], registry)
    exclusions, exclusion_counts = load_exclusion_union(paths, registry)
    eligible = eligible_source_rows(source_rows, exclusions, registry)
    slots = expand_slots(registry)
    namespace = selection_namespace(registry, protocol_commit)
    assignment, slot_supply = deterministic_match(eligible, slots, namespace)
    _, calibration_reference, calibration_summary = build_calibration_bank(paths, registry, namespace)
    sentinel_allocations = build_sentinel_allocations(calibration_reference, registry, namespace)
    sentinel_summary = summarize_sentinel_allocations(calibration_reference, sentinel_allocations)
    return {
        "name": "step3q_p0_aggregate_preflight_v1",
        "route_id": registry["route_id"],
        "source_rows": len(source_rows),
        "exclusions": exclusion_counts,
        "eligible_fresh_prompts": len(eligible),
        "slot_count": len(slots),
        "slot_matching_feasible": len(assignment) == len(slots),
        "minimum_slot_candidate_supply": min(slot_supply.values()),
        "maximum_slot_candidate_supply": max(slot_supply.values()),
        "calibration": calibration_summary,
        "sentinel_allocations": sentinel_summary,
        "selected_prompt_ids_released": False,
        "prompt_text_released": False,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, encoding="utf-8").strip()


def require_clean_protocol_commit(protocol_commit: str) -> None:
    if not HEX40.fullmatch(protocol_commit):
        raise Step3QValidationError("protocol commit must be a full lowercase SHA-1")
    head = _git_output("rev-parse", "HEAD")
    if head != protocol_commit:
        raise Step3QValidationError(f"prepare requires HEAD to equal the P0 protocol commit: {head} != {protocol_commit}")
    if _git_output("status", "--porcelain"):
        raise Step3QValidationError("prepare requires a clean P0 worktree")


def write_bundle(output_root: Path, files: Mapping[str, bytes]) -> None:
    if output_root.exists():
        raise Step3QValidationError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for name, content in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        temporary.replace(output_root)
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temporary.exists():
            temporary.rmdir()
        raise


def build_prepare_bundle(
    registry: Mapping[str, Any], protocol_commit: str
) -> tuple[dict[str, bytes], dict[str, Any]]:
    paths = validate_frozen_inputs(registry)
    namespace = selection_namespace(registry, protocol_commit)
    source_rows = load_source_rows(paths["train_risk"], registry)
    exclusions, exclusion_counts = load_exclusion_union(paths, registry)
    eligible = eligible_source_rows(source_rows, exclusions, registry)
    slots = expand_slots(registry)
    assignment, slot_supply = deterministic_match(eligible, slots, namespace)
    calibration_packet, calibration_reference, calibration_summary = build_calibration_bank(paths, registry, namespace)
    sentinel_allocations = build_sentinel_allocations(calibration_reference, registry, namespace)
    sentinel_summary = summarize_sentinel_allocations(calibration_reference, sentinel_allocations)
    prompts, requests = build_prompt_and_request_rows(eligible, assignment, slots, namespace)
    severity_counts = Counter(str(row["severity_level"]) for row in prompts)
    category_counts = Counter(row["assigned_category"] for row in prompts)
    diagnostics = {
        "name": "step3q_prepare_diagnostics_v1",
        "protocol_commit": protocol_commit,
        "namespace_sha256": namespace,
        "source_rows": len(source_rows),
        "exclusions": exclusion_counts,
        "eligible_fresh_prompts": len(eligible),
        "selected_prompts": len(prompts),
        "api_requests": len(requests),
        "category_counts": dict(sorted(category_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "minimum_slot_candidate_supply": min(slot_supply.values()),
        "calibration": calibration_summary,
        "sentinel_allocations": sentinel_summary,
        "source_harmful_responses_released": False,
        "step3p_prompts_reused": False,
        "o4_o4r_prompts_reused": False,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files: dict[str, bytes] = {
        "diagnostics/step3q_prepare_diagnostics.json": json_text(diagnostics).encode("utf-8"),
        "calibration/calibration_bank_packet.jsonl": jsonl_text(calibration_packet).encode("utf-8"),
        "calibration/calibration_reference_hidden.jsonl": jsonl_text(calibration_reference).encode("utf-8"),
        "calibration/sentinel_allocations_hidden.json": json_text(sentinel_allocations).encode("utf-8"),
        "authoring/prompts_hidden.jsonl": jsonl_text(prompts).encode("utf-8"),
        "authoring/api_requests.jsonl": jsonl_text(requests).encode("utf-8"),
    }
    instructions_bytes = paths["authoring_instructions"].read_bytes()
    artifacts = [
        {"path": name, "bytes": len(content), "sha256": sha256_bytes(content)}
        for name, content in sorted(files.items())
    ]
    manifest = {
        "name": "step3q_p1_prompt_and_calibration_manifest_v1",
        "route_id": registry["route_id"],
        "protocol_commit": protocol_commit,
        "namespace_sha256": namespace,
        "authoring_instructions": {
            "path": registry["inputs"]["authoring_instructions_path"],
            "bytes": len(instructions_bytes),
            "sha256": sha256_bytes(instructions_bytes),
        },
        "artifacts": artifacts,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["protocol_manifest.json"] = json_text(manifest).encode("utf-8")
    return files, diagnostics


def prepare(registry: Mapping[str, Any], protocol_commit: str, output_root: Path | None = None) -> dict[str, Any]:
    require_clean_protocol_commit(protocol_commit)
    files, diagnostics = build_prepare_bundle(registry, protocol_commit)
    destination = output_root or resolve_repo_path(registry["outputs"]["root"])
    write_bundle(destination, files)
    return {"output_root": str(destination), **diagnostics}


def verify_prepared_bundle(registry: Mapping[str, Any], prepared_root: Path) -> dict[str, Any]:
    prepared_root = prepared_root.resolve()
    manifest_path = prepared_root / "protocol_manifest.json"
    if not manifest_path.is_file():
        raise Step3QValidationError("prepared bundle lacks protocol_manifest.json")
    manifest = read_json(manifest_path)
    protocol_commit = manifest.get("protocol_commit")
    if not isinstance(protocol_commit, str) or not HEX40.fullmatch(protocol_commit):
        raise Step3QValidationError("prepared bundle has an invalid P0 protocol commit")
    expected_files, diagnostics = build_prepare_bundle(registry, protocol_commit)
    observed_paths = {
        path.relative_to(prepared_root).as_posix()
        for path in prepared_root.rglob("*")
        if path.is_file()
    }
    if observed_paths != set(expected_files):
        raise Step3QValidationError("prepared bundle file set differs from the deterministic P1 package")
    for relative_path, expected_content in expected_files.items():
        actual_content = (prepared_root / relative_path).read_bytes()
        if actual_content != expected_content:
            raise Step3QValidationError(f"prepared artifact differs from deterministic reconstruction: {relative_path}")
    return {
        "name": "step3q_prepared_bundle_verification_v1",
        "route_id": registry["route_id"],
        "protocol_commit": protocol_commit,
        "namespace_sha256": manifest["namespace_sha256"],
        "artifact_files": len(expected_files) - 1,
        "selected_prompts": diagnostics["selected_prompts"],
        "api_requests": diagnostics["api_requests"],
        "calibration_rows": diagnostics["calibration"]["rows"],
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }


def api_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "opaque_request_id": {"type": "string"},
            "response": {"type": "string"},
        },
        "required": ["opaque_request_id", "response"],
        "additionalProperties": False,
    }


def build_api_payload(
    request_row: Mapping[str, Any], registry: Mapping[str, Any], instructions: str
) -> dict[str, Any]:
    if request_row.get("author_mode") not in AUTHOR_MODES:
        raise Step3QValidationError("API request has an invalid author mode")
    for field in ("opaque_request_id", "opaque_candidate_id", "opaque_prompt_id", "prompt"):
        if not isinstance(request_row.get(field), str) or not request_row[field]:
            raise Step3QValidationError(f"API request lacks {field}")
    cfg = registry["authoring"]
    user_input = canonical_json(
        {
            "opaque_request_id": request_row["opaque_request_id"],
            "author_mode": request_row["author_mode"],
            "prompt": request_row["prompt"],
        }
    )
    return {
        "model": cfg["model"],
        "instructions": instructions,
        "input": user_input,
        "reasoning": {"effort": cfg["reasoning_effort"]},
        "max_output_tokens": int(cfg["max_output_tokens"]),
        "store": bool(cfg["store"]),
        "truncation": cfg["truncation"],
        "tools": list(cfg["tools"]),
        "text": {
            "verbosity": cfg["text_verbosity"],
            "format": {
                "type": "json_schema",
                "name": "step3q_safe_target",
                "strict": True,
                "schema": api_response_schema(),
            },
        },
        "metadata": {
            "route_id": registry["route_id"],
            "opaque_request_id": request_row["opaque_request_id"],
        },
        "safety_identifier": "step3q-research-dispatcher",
    }


def validate_effective_model(value: Any, registry: Mapping[str, Any]) -> str:
    if not isinstance(value, str) or re.fullmatch(
        registry["authoring"]["effective_model_regex"], value
    ) is None:
        raise Step3QProvenanceError("effective API model violates the frozen model-family policy")
    return value


def validate_api_response(
    response: Mapping[str, Any], request_row: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    if response.get("status") != "completed" or response.get("error") is not None:
        raise Step3QValidationError("API response did not complete successfully")
    effective_model = validate_effective_model(response.get("model"), registry)
    api_response_id = response.get("id")
    if not isinstance(api_response_id, str) or not api_response_id:
        raise Step3QProvenanceError("API response lacks a non-empty response ID")
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(field)) is not int or usage[field] < 0
        for field in ("input_tokens", "output_tokens", "total_tokens")
    ):
        raise Step3QProvenanceError("API response lacks complete non-negative token usage")
    output_texts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                output_texts.append(content["text"])
    if len(output_texts) != 1:
        raise Step3QValidationError("API response must contain exactly one output_text")
    value = json.loads(output_texts[0])
    if not isinstance(value, dict) or set(value) != {"opaque_request_id", "response"}:
        raise Step3QValidationError("structured API output has unexpected fields")
    if value["opaque_request_id"] != request_row["opaque_request_id"]:
        raise Step3QValidationError("structured API output request ID mismatch")
    text = value["response"]
    if not isinstance(text, str) or not text.strip():
        raise Step3QValidationError("structured API output response is empty")
    tokens = whitespace_tokens(text)
    cfg = registry["authoring"]
    if not int(cfg["minimum_response_whitespace_tokens"]) <= tokens <= int(cfg["maximum_response_whitespace_tokens"]):
        raise Step3QValidationError("structured API output response is outside the frozen token envelope")
    return {
        "opaque_request_id": request_row["opaque_request_id"],
        "opaque_candidate_id": request_row["opaque_candidate_id"],
        "opaque_prompt_id": request_row["opaque_prompt_id"],
        "author_mode": request_row["author_mode"],
        "prompt": request_row["prompt"],
        "response": text.strip(),
        "response_whitespace_tokens": tokens,
        "response_sha256": sha256_bytes(text.strip().encode("utf-8")),
        "normalized_response_sha256": sha256_bytes(normalize_text(text).encode("utf-8")),
        "api_response_id": api_response_id,
        "effective_model": effective_model,
        "usage": usage,
    }


def validate_candidate_record(
    candidate: Mapping[str, Any], request_row: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    for field in (
        "opaque_request_id",
        "opaque_candidate_id",
        "opaque_prompt_id",
        "author_mode",
        "prompt",
    ):
        if candidate.get(field) != request_row.get(field):
            raise Step3QValidationError(f"candidate does not match its request field {field}")
    response = candidate.get("response")
    if not isinstance(response, str) or not response or response != response.strip():
        raise Step3QValidationError("candidate response is empty or not canonically stripped")
    tokens = whitespace_tokens(response)
    cfg = registry["authoring"]
    if not int(cfg["minimum_response_whitespace_tokens"]) <= tokens <= int(
        cfg["maximum_response_whitespace_tokens"]
    ):
        raise Step3QValidationError("candidate response is outside the frozen token envelope")
    if candidate.get("response_whitespace_tokens") != tokens:
        raise Step3QValidationError("candidate response token count is inconsistent")
    if candidate.get("response_sha256") != sha256_bytes(response.encode("utf-8")):
        raise Step3QValidationError("candidate response hash is inconsistent")
    if candidate.get("normalized_response_sha256") != sha256_bytes(
        normalize_text(response).encode("utf-8")
    ):
        raise Step3QValidationError("candidate normalized-response hash is inconsistent")
    validate_effective_model(candidate.get("effective_model"), registry)
    if not isinstance(candidate.get("api_response_id"), str) or not candidate["api_response_id"]:
        raise Step3QProvenanceError("candidate lacks a non-empty API response ID")
    usage = candidate.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(field)) is not int or usage[field] < 0
        for field in ("input_tokens", "output_tokens", "total_tokens")
    ):
        raise Step3QProvenanceError("candidate lacks complete non-negative token usage")
    return dict(candidate)


def expected_receipt_hashes(
    request_row: Mapping[str, Any], registry: Mapping[str, Any], instructions: str
) -> tuple[bytes, str, str]:
    payload_bytes = canonical_json(build_api_payload(request_row, registry, instructions)).encode("utf-8")
    request_sha = sha256_bytes(canonical_json(dict(request_row)).encode("utf-8"))
    return payload_bytes, request_sha, sha256_bytes(payload_bytes)


def validate_receipt(
    receipt: Mapping[str, Any],
    request_row: Mapping[str, Any],
    registry: Mapping[str, Any],
    request_sha: str,
    payload_sha: str,
) -> dict[str, Any]:
    request_id = request_row["opaque_request_id"]
    if receipt.get("schema") != "step3q_api_receipt_v1":
        raise Step3QValidationError(f"invalid receipt schema for {request_id}")
    if receipt.get("route_id") != registry["route_id"]:
        raise Step3QValidationError(f"receipt route mismatch for {request_id}")
    if receipt.get("opaque_request_id") != request_id:
        raise Step3QValidationError(f"receipt request ID mismatch for {request_id}")
    if receipt.get("request_sha256") != request_sha or receipt.get("payload_sha256") != payload_sha:
        raise Step3QValidationError(f"receipt content hashes do not match request {request_id}")
    if not isinstance(receipt.get("received_at_utc"), str) or not receipt["received_at_utc"]:
        raise Step3QValidationError(f"receipt timestamp is missing for {request_id}")
    attempts = receipt.get("attempts")
    maximum_attempts = int(registry["authoring"]["maximum_attempts"])
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= maximum_attempts:
        raise Step3QValidationError(f"receipt attempt trace is invalid for {request_id}")
    if [row.get("attempt") for row in attempts if isinstance(row, dict)] != list(
        range(1, len(attempts) + 1)
    ):
        raise Step3QValidationError(f"receipt attempts are not sequential for {request_id}")
    status = receipt.get("status")
    allowed_statuses = {"valid", "semantic_failure", "protocol_failure", "http_failure", "transport_failure"}
    if status not in allowed_statuses:
        raise Step3QValidationError(f"receipt terminal status is invalid for {request_id}")
    if status == "valid":
        candidate = receipt.get("candidate")
        if not isinstance(candidate, dict):
            raise Step3QValidationError(f"valid receipt lacks a candidate for {request_id}")
        validate_candidate_record(candidate, request_row, registry)
    elif "candidate" in receipt:
        raise Step3QValidationError(f"failed receipt unexpectedly contains a candidate for {request_id}")
    return dict(receipt)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json_text(value), encoding="utf-8", newline="\n")
    if path.exists():
        temporary.unlink()
        raise Step3QValidationError(f"immutable receipt already exists: {path}")
    temporary.replace(path)


def _dispatch_one(
    request_row: Mapping[str, Any], registry: Mapping[str, Any], instructions: str, api_key: str, receipt_dir: Path
) -> dict[str, Any]:
    request_id = request_row["opaque_request_id"]
    receipt_path = receipt_dir / f"{request_id}.json"
    payload_bytes, request_sha, payload_sha = expected_receipt_hashes(
        request_row, registry, instructions
    )
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        return validate_receipt(receipt, request_row, registry, request_sha, payload_sha)

    cfg = registry["authoring"]
    traces: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    for attempt in range(1, int(cfg["maximum_attempts"]) + 1):
        started = time.monotonic()
        request = urllib.request.Request(
            cfg["endpoint"],
            data=payload_bytes,
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=float(cfg["timeout_seconds"])) as response_handle:
                raw = response_handle.read()
                status = int(response_handle.status)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                parsed = None
                terminal = {"status": "semantic_failure", "error_type": type(exc).__name__}
            else:
                try:
                    candidate = validate_api_response(parsed, request_row, registry)
                    terminal = {"status": "valid", "candidate": candidate}
                except Step3QProvenanceError as exc:
                    terminal = {"status": "protocol_failure", "error": str(exc)}
                except Exception as exc:
                    terminal = {"status": "semantic_failure", "error": str(exc)}
            traces.append(
                {
                    "attempt": attempt,
                    "http_status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "raw_response_sha256": sha256_bytes(raw),
                    "api_response_id": parsed.get("id") if isinstance(parsed, dict) else None,
                    "effective_model": parsed.get("model") if isinstance(parsed, dict) else None,
                }
            )
            break
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            traces.append(
                {
                    "attempt": attempt,
                    "http_status": int(exc.code),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_body_sha256": sha256_bytes(raw),
                }
            )
            if int(exc.code) not in set(cfg["retry_http_statuses"]) or attempt == int(cfg["maximum_attempts"]):
                terminal = {"status": "http_failure", "http_status": int(exc.code)}
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            traces.append(
                {
                    "attempt": attempt,
                    "http_status": None,
                    "elapsed_seconds": time.monotonic() - started,
                    "transport_error_type": type(exc).__name__,
                }
            )
            if attempt == int(cfg["maximum_attempts"]):
                terminal = {"status": "transport_failure", "error_type": type(exc).__name__}
                break
        if attempt < int(cfg["maximum_attempts"]):
            time.sleep(float(cfg["retry_backoff_seconds"][attempt - 1]))
    if terminal is None:
        raise Step3QValidationError(f"dispatcher did not create a terminal outcome for {request_id}")
    receipt = {
        "schema": "step3q_api_receipt_v1",
        "route_id": registry["route_id"],
        "opaque_request_id": request_id,
        "request_sha256": request_sha,
        "payload_sha256": payload_sha,
        "received_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": traces,
        **terminal,
    }
    validate_receipt(receipt, request_row, registry, request_sha, payload_sha)
    _atomic_json(receipt_path, receipt)
    return receipt


def seal_dispatch_receipts(
    registry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    receipt_dir: Path,
    sealed_root: Path,
    instructions: str,
) -> dict[str, Any]:
    request_by_id = {row.get("opaque_request_id"): dict(row) for row in requests}
    receipt_by_id = {row.get("opaque_request_id"): dict(row) for row in receipts}
    if None in request_by_id or len(request_by_id) != len(requests):
        raise Step3QValidationError("prepared requests have missing or duplicate IDs")
    if None in receipt_by_id or len(receipt_by_id) != len(receipts):
        raise Step3QValidationError("terminal receipts have missing or duplicate IDs")
    if set(receipt_by_id) != set(request_by_id):
        raise Step3QValidationError("terminal receipts do not cover the fixed request denominator")

    validated_receipts: list[dict[str, Any]] = []
    for request_id in sorted(request_by_id):
        request_row = request_by_id[request_id]
        _, request_sha, payload_sha = expected_receipt_hashes(request_row, registry, instructions)
        receipt = validate_receipt(
            receipt_by_id[request_id], request_row, registry, request_sha, payload_sha
        )
        receipt_path = receipt_dir / f"{request_id}.json"
        if not receipt_path.is_file() or read_json(receipt_path) != receipt:
            raise Step3QValidationError(f"receipt file differs from terminal receipt {request_id}")
        validated_receipts.append(receipt)

    valid = [receipt["candidate"] for receipt in validated_receipts if receipt["status"] == "valid"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in valid:
        groups[candidate["normalized_response_sha256"]].append(candidate)
    namespace = manifest["namespace_sha256"]
    retained: list[dict[str, Any]] = []
    duplicate_failures: list[str] = []
    for values in groups.values():
        values.sort(key=lambda row: stable_rank(namespace, "api-response-dedup", row["opaque_candidate_id"]))
        retained.append(values[0])
        duplicate_failures.extend(row["opaque_candidate_id"] for row in values[1:])
    retained.sort(key=lambda row: stable_rank(namespace, "sealed-candidate-order", row["opaque_candidate_id"]))
    valid_prompts = {row["opaque_prompt_id"] for row in retained}
    effective_models = sorted({row["effective_model"] for row in valid})
    protocol_failures = sum(row["status"] == "protocol_failure" for row in validated_receipts)
    model_provenance_pass = protocol_failures == 0 and len(effective_models) == 1
    cfg = registry["authoring"]
    passed = (
        model_provenance_pass
        and len(retained) >= int(cfg["minimum_valid_unique_candidates"])
        and len(valid_prompts) >= int(cfg["minimum_valid_prompts"])
    )
    route = "PASS_API_AUTHORING" if passed else "STOP_API_YIELD"
    summary = {
        "schema": "step3q_api_dispatch_summary_v1",
        "route_id": registry["route_id"],
        "route": route,
        "fixed_requests": len(requests),
        "terminal_receipts": len(validated_receipts),
        "valid_before_deduplication": len(valid),
        "valid_unique_candidates": len(retained),
        "valid_prompts": len(valid_prompts),
        "normalized_duplicate_failures": len(duplicate_failures),
        "effective_models": effective_models,
        "model_provenance_pass": model_provenance_pass,
        "status_counts": dict(Counter(row["status"] for row in validated_receipts)),
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    receipt_manifest = {
        "schema": "step3q_api_receipt_manifest_v1",
        "receipts": [
            {
                "opaque_request_id": row["opaque_request_id"],
                "path": f"receipts/{row['opaque_request_id']}.json",
                "sha256": sha256_file(receipt_dir / f"{row['opaque_request_id']}.json"),
            }
            for row in validated_receipts
        ],
    }
    files = {
        "candidates.jsonl": jsonl_text(retained).encode("utf-8"),
        "dispatch_summary.json": json_text(summary).encode("utf-8"),
        "receipt_manifest.json": json_text(receipt_manifest).encode("utf-8"),
    }
    write_bundle(sealed_root, files)
    return summary


def require_external_spool_root(spool_root: Path) -> Path:
    resolved = spool_root.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return resolved
    raise Step3QValidationError("API receipt spool must remain outside the Git repository")


def dispatch_api(
    registry: Mapping[str, Any], prepared_root: Path, spool_root: Path
) -> dict[str, Any]:
    api_key = os.environ.get(registry["authoring"]["api_key_env"])
    if not api_key:
        raise Step3QValidationError("OPENAI_API_KEY is required before any external dispatch")
    verify_prepared_bundle(registry, prepared_root)
    manifest = read_json(prepared_root / "protocol_manifest.json")
    if manifest.get("route_id") != registry["route_id"]:
        raise Step3QValidationError("prepared manifest route mismatch")
    requests = read_jsonl(prepared_root / "authoring" / "api_requests.jsonl")
    if len(requests) != int(registry["selection"]["prompt_count"]) * len(AUTHOR_MODES):
        raise Step3QValidationError("prepared request denominator mismatch")
    instructions_path = resolve_repo_path(registry["inputs"]["authoring_instructions_path"])
    instructions = instructions_path.read_text(encoding="utf-8")
    if sha256_file(instructions_path) != manifest["authoring_instructions"]["sha256"]:
        raise Step3QValidationError("authoring instructions differ from prepared manifest")
    spool_root = require_external_spool_root(spool_root)
    receipt_dir = spool_root / "receipts"
    sealed_root = spool_root / "sealed"
    if sealed_root.exists():
        raise Step3QValidationError("dispatch spool is already sealed")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    expected_request_ids = {row["opaque_request_id"] for row in requests}
    if len(expected_request_ids) != len(requests):
        raise Step3QValidationError("prepared request IDs are not unique")
    unexpected_receipts = {
        path.stem for path in receipt_dir.glob("*.json") if path.stem not in expected_request_ids
    }
    if unexpected_receipts:
        raise Step3QValidationError("dispatch spool contains receipts outside the fixed denominator")
    for row in requests:
        receipt_path = receipt_dir / f"{row['opaque_request_id']}.json"
        if receipt_path.exists():
            _, request_sha, payload_sha = expected_receipt_hashes(row, registry, instructions)
            validate_receipt(read_json(receipt_path), row, registry, request_sha, payload_sha)
    receipts: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(registry["authoring"]["parallelism"])) as executor:
        futures = [
            executor.submit(_dispatch_one, row, registry, instructions, api_key, receipt_dir)
            for row in requests
        ]
        for future in futures:
            receipts.append(future.result())
    summary = seal_dispatch_receipts(
        registry, manifest, requests, receipts, receipt_dir, sealed_root, instructions
    )
    return {"spool_root": str(spool_root), **summary}


def validate_review_row(row: Mapping[str, Any]) -> None:
    if type(row.get("detail_retention")) is not int or row["detail_retention"] not in {0, 1, 2}:
        raise Step3QValidationError("review detail_retention must be 0, 1, or 2")
    for field in BINARY_FIELDS:
        if row.get(field) not in {"yes", "no"}:
            raise Step3QValidationError(f"review {field} must be yes or no")


def score_calibration_review(
    review_rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]], registry: Mapping[str, Any]
) -> dict[str, Any]:
    review_map = {row.get("opaque_candidate_id"): dict(row) for row in review_rows}
    if None in review_map or len(review_map) != len(review_rows):
        raise Step3QValidationError("calibration review IDs are missing or duplicated")
    reference_map = {row["opaque_candidate_id"]: dict(row) for row in reference_rows}
    if not set(reference_map) <= set(review_map):
        raise Step3QValidationError("calibration review does not cover the required sentinels")
    for row in review_map.values():
        validate_review_row(row)
    cfg = registry["calibration"]["gates"]
    component_accuracy: dict[str, float] = {}
    component_class_accuracy: dict[str, dict[str, float]] = {}
    component_refs = [row for row in reference_rows if row["calibration_role"] == "component_anchor"]
    for field in COMPONENT_FIELDS:
        component_accuracy[field] = sum(
            review_map[row["opaque_candidate_id"]][field] == row["expected"][field]
            for row in component_refs
        ) / len(component_refs)
        component_class_accuracy[field] = {}
        for value in ("yes", "no"):
            class_refs = [row for row in component_refs if row["expected"][field] == value]
            if not class_refs:
                raise Step3QValidationError(f"component calibration lacks the {field}={value} class")
            component_class_accuracy[field][value] = sum(
                review_map[row["opaque_candidate_id"]][field] == value for row in class_refs
            ) / len(class_refs)
    mismatches = [row for row in reference_rows if row["calibration_role"] == "prompt_mismatch"]
    mismatch_binding = sum(review_map[row["opaque_candidate_id"]]["prompt_binding"] == "no" for row in mismatches) / len(mismatches)
    mismatch_detail_zero = sum(review_map[row["opaque_candidate_id"]]["detail_retention"] == 0 for row in mismatches) / len(mismatches)
    reversal = sum(
        review_map[row["matched_anchor_candidate_id"]]["detail_retention"]
        > review_map[row["opaque_candidate_id"]]["detail_retention"]
        for row in mismatches
    ) / len(mismatches)
    safety_refs = [row for row in reference_rows if row["calibration_role"] == "clean_safety_anchor"]
    clean_accuracy = sum(
        review_map[row["opaque_candidate_id"]]["clean_safe"] == row["expected"]["clean_safe"]
        for row in safety_refs
    ) / len(safety_refs)
    gates = {
        **{
            f"component_{field}": value >= float(cfg["component_reference_accuracy_min"])
            for field, value in component_accuracy.items()
        },
        **{
            f"component_{field}_{class_value}": accuracy
            >= float(cfg["component_class_reference_accuracy_min"])
            for field, values in component_class_accuracy.items()
            for class_value, accuracy in values.items()
        },
        "mismatch_prompt_binding": mismatch_binding >= float(cfg["mismatch_prompt_binding_no_accuracy_min"]),
        "mismatch_detail_zero": mismatch_detail_zero >= float(cfg["mismatch_detail_zero_accuracy_min"]),
        "mismatch_detail_reversal": reversal >= float(cfg["mismatch_detail_reversal_rate_min"]),
        "clean_safety": clean_accuracy >= float(cfg["clean_safety_reference_accuracy_min"]),
    }
    return {
        "component_reference_accuracy": component_accuracy,
        "component_class_reference_accuracy": component_class_accuracy,
        "mismatch_prompt_binding_no_accuracy": mismatch_binding,
        "mismatch_detail_zero_accuracy": mismatch_detail_zero,
        "mismatch_detail_reversal_rate": reversal,
        "clean_safety_reference_accuracy": clean_accuracy,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def conditional_class_specific_gates(
    metrics: Mapping[str, Any], minimum_count: int, minimum_agreement: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, count_key, agreement_key in (
        ("positive", "yes_ratings", "positive_agreement"),
        ("negative", "no_ratings", "negative_agreement"),
    ):
        supported = int(metrics[count_key]) >= minimum_count
        agreement = metrics.get(agreement_key)
        result[name] = {
            "supported_on_primary": supported,
            "pooled_ratings": int(metrics[count_key]),
            "agreement": agreement,
            "gate_required": supported,
            "pass": (agreement is not None and float(agreement) >= minimum_agreement) if supported else None,
        }
    return result


def command_preflight(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry).resolve())
    print(json_text(preflight(registry)), end="")


def command_prepare(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry).resolve())
    output = Path(args.output_root).resolve() if args.output_root else None
    print(json_text(prepare(registry, args.protocol_commit, output)), end="")


def command_verify_prepared(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry).resolve())
    prepared = Path(args.prepared_root).resolve() if args.prepared_root else resolve_repo_path(
        registry["outputs"]["root"]
    )
    print(json_text(verify_prepared_bundle(registry, prepared)), end="")


def command_dispatch(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry).resolve())
    prepared = Path(args.prepared_root).resolve() if args.prepared_root else resolve_repo_path(registry["outputs"]["root"])
    print(json_text(dispatch_api(registry, prepared, Path(args.spool_root))), end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preflight", help="Validate aggregate feasibility without releasing fresh prompts")
    pre.set_defaults(func=command_preflight)
    prep = subparsers.add_parser("prepare", help="Materialize the P1 prompt and calibration package after P0")
    prep.add_argument("--protocol-commit", required=True)
    prep.add_argument("--output-root")
    prep.set_defaults(func=command_prepare)
    verify = subparsers.add_parser(
        "verify-prepared", help="Reconstruct and verify the immutable P1 prompt/calibration package"
    )
    verify.add_argument("--prepared-root")
    verify.set_defaults(func=command_verify_prepared)
    dispatch = subparsers.add_parser("dispatch", help="Run or resume the controlled Responses API dispatcher")
    dispatch.add_argument("--prepared-root")
    dispatch.add_argument("--spool-root", required=True)
    dispatch.set_defaults(func=command_dispatch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
