#!/usr/bin/env python3
"""Prepare and adjudicate the Step3N-R1 N0 tuple-qualification protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import tempfile
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/step3n_r1_n0_registry.json"
BASE = ROOT / "results/step3n_r1_n0"
PACKETS = BASE / "packets/source"
REVIEWS = BASE / "reviews/source"
REPORTS = BASE / "reports"
DIAGNOSTICS = BASE / "diagnostics"

SOURCE_PACKET = PACKETS / "step3n_r1_source_qualification_packet.jsonl"
SOURCE_KEY = PACKETS / "step3n_r1_source_qualification_hidden_key.csv"
SOURCE_DUPLICATES = PACKETS / "step3n_r1_source_qualification_duplicate_key.csv"
SOURCE_A_TEMPLATE = PACKETS / "step3n_r1_source_reviewer_a_template.jsonl"
SOURCE_B_TEMPLATE = PACKETS / "step3n_r1_source_reviewer_b_template.jsonl"
SOURCE_C_PACKET = PACKETS / "step3n_r1_source_reviewer_c_packet.jsonl"
SOURCE_PACKET_MANIFEST = PACKETS / "step3n_r1_source_packet_manifest.json"
SOURCE_C_MANIFEST = PACKETS / "step3n_r1_source_reviewer_c_packet_manifest.json"

SOURCE_A = REVIEWS / "step3n_r1_source_reviewer_a.jsonl"
SOURCE_B = REVIEWS / "step3n_r1_source_reviewer_b.jsonl"
SOURCE_C = REVIEWS / "step3n_r1_source_reviewer_c.jsonl"
SOURCE_FINAL = REVIEWS / "step3n_r1_source_final_labels.jsonl"
SOURCE_AB_FREEZE = REVIEWS / "step3n_r1_source_ab_freeze_manifest.json"

PROTOCOL_MANIFEST = REPORTS / "step3n_r1_n0_protocol_manifest.json"
PROTOCOL_REPORT = REPORTS / "Step3N_R1_N0_source_protocol_freeze_report.md"
SOURCE_AGREEMENT = REPORTS / "step3n_r1_source_review_agreement.json"
SOURCE_DECISION = REPORTS / "step3n_r1_source_route_decision.json"
SELECTED_POSITIVES = REPORTS / "step3n_r1_selected_positive_sources.jsonl"
SOURCE_REPORT = REPORTS / "Step3N_R1_N0_source_qualification_report.md"
INPUT_VALIDATION = DIAGNOSTICS / "step3n_r1_source_input_validation.json"
SYNTHETIC_TESTS = DIAGNOSTICS / "step3n_r1_n0_synthetic_tests.json"

ORDINAL_FIELDS = ["detail_retention", "safe_actionability"]
NOMINAL_FIELDS = ["naturalness", "clean_safe"]
DECISION_FIELDS = ORDINAL_FIELDS + NOMINAL_FIELDS
REVIEW_FIELDS = ["opaque_review_id", *DECISION_FIELDS, "short_reason"]
ORDINAL_VALUES = {0, 1, 2}
BOOL_VALUES = {"yes", "no"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    require_equal_if_exists: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = jsonl_text(rows)
    if require_equal_if_exists and path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to mutate frozen JSONL: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
    *,
    require_equal_if_exists: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    content = buffer.getvalue()
    if require_equal_if_exists and path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to mutate frozen CSV: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(seed_or_salt: int | str, *parts: str) -> str:
    return sha256_text("|".join([str(seed_or_salt), *parts]))


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def registry() -> dict[str, Any]:
    return read_json(REGISTRY)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_boundaries(reg: dict[str, Any]) -> None:
    required_false = [
        "heldout_opened",
        "core100_opened_for_n0",
        "proxy_dev_outcomes_opened",
        "training_authorized",
    ]
    bad = [field for field in required_false if reg.get(field) is not False]
    if bad or int(reg.get("optimizer_steps", -1)) != 0:
        raise ValueError(f"N0 boundary flags are not closed: {bad}")


def load_source_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reg = registry()
    validate_boundaries(reg)
    packet_path = resolve_repo_path(reg["source_packet"])
    key_path = resolve_repo_path(reg["source_role_key"])
    split_path = resolve_repo_path(reg["split_manifest"])
    rubric_path = resolve_repo_path(reg["review_rubric"])
    for path in (packet_path, key_path, split_path, rubric_path):
        if not path.exists():
            raise FileNotFoundError(path)

    packet = read_jsonl(packet_path)
    key = read_csv(key_path)
    packet_by_id = {row["opaque_response_id"]: row for row in packet}
    key_by_id = {row["opaque_response_id"]: row for row in key}
    if len(packet_by_id) != len(packet) or len(key_by_id) != len(key):
        raise ValueError("duplicate source opaque_response_id")
    if set(packet_by_id) != set(key_by_id):
        raise ValueError("source packet and role key IDs differ")

    split_rows = read_csv(split_path)
    safety_train = {
        row["sample_id"] for row in split_rows if row["split"] == "safety_train"
    }
    heldout_ids = {
        row["sample_id"] for row in split_rows if row["split"] == "heldout_test"
    }
    eligible_roles = set(reg["eligible_source_roles"])
    selected: list[dict[str, Any]] = []
    for source_id, hidden in key_by_id.items():
        if hidden["role"] not in eligible_roles:
            continue
        visible = packet_by_id[source_id]
        if hidden["sample_id"] not in safety_train:
            raise ValueError(f"eligible source outside safety_train: {source_id}")
        if hidden["sample_id"] in heldout_ids:
            raise ValueError(f"eligible source overlaps heldout: {source_id}")
        if sha256_text(visible["response"]) != hidden["review_response_sha256"]:
            raise ValueError(f"source response hash mismatch: {source_id}")
        selected.append(
            {
                "source_opaque_response_id": source_id,
                "sample_id": hidden["sample_id"],
                "category": hidden["category"],
                "role": hidden["role"],
                "source_response_id": hidden["source_response_id"],
                "prompt": visible["prompt"],
                "response": visible["response"],
                "prompt_sha256": sha256_text(visible["prompt"]),
                "response_sha256": sha256_text(visible["response"]),
            }
        )

    expected_responses = int(reg["expected_source_responses"])
    expected_prompts = int(reg["expected_training_prompts"])
    prompt_ids = {row["sample_id"] for row in selected}
    if len(selected) != expected_responses:
        raise ValueError(f"expected {expected_responses} source responses, got {len(selected)}")
    if prompt_ids != safety_train or len(prompt_ids) != expected_prompts:
        raise ValueError("source responses do not exactly cover frozen safety_train IDs")
    source_response_ids = [row["source_response_id"] for row in selected]
    if len(source_response_ids) != len(set(source_response_ids)):
        raise ValueError("duplicate source_response_id in eligible source pool")

    validation = {
        "name": "step3n_r1_source_input_validation_v1",
        "git_head_before_packet_freeze": git_head(),
        "source_packet": repo_path(packet_path),
        "source_packet_sha256": sha256(packet_path),
        "source_role_key": repo_path(key_path),
        "source_role_key_sha256": sha256(key_path),
        "split_manifest": repo_path(split_path),
        "split_manifest_sha256": sha256(split_path),
        "review_rubric": repo_path(rubric_path),
        "review_rubric_sha256": sha256(rubric_path),
        "source_responses": len(selected),
        "training_prompts": len(prompt_ids),
        "source_role_counts": dict(sorted(Counter(row["role"] for row in selected).items())),
        "category_prompt_counts": dict(
            sorted(
                Counter(
                    next(row["category"] for row in selected if row["sample_id"] == sample_id)
                    for sample_id in prompt_ids
                ).items()
            )
        ),
        "safety_train_ids_match_exactly": True,
        "heldout_id_overlap": 0,
        "old_quality_labels_read": False,
        "core100_content_read": False,
        "heldout_content_read": False,
        "proxy_dev_outcomes_read": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return selected, validation


def opaque_review_id(seed: int, source_id: str, instance: str) -> str:
    return f"step3n_src_{stable_rank(seed, source_id, instance)[:18]}"


def freeze_source_packet() -> dict[str, Any]:
    reg = registry()
    sources, validation = load_source_inputs()
    packet_seed = int(reg["source_packet_seed"])
    duplicate_seed = int(reg["source_duplicate_seed"])
    duplicate_count = int(reg["source_hidden_duplicates"])

    sources = sorted(
        sources,
        key=lambda row: stable_rank(packet_seed, row["source_opaque_response_id"]),
    )
    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    primary_by_source: dict[str, str] = {}
    for source in sources:
        opaque_id = opaque_review_id(
            packet_seed, source["source_opaque_response_id"], "primary"
        )
        primary_by_source[source["source_opaque_response_id"]] = opaque_id
        packet_rows.append(
            {
                "opaque_review_id": opaque_id,
                "prompt": source["prompt"],
                "response": source["response"],
            }
        )
        key_rows.append(
            {
                "opaque_review_id": opaque_id,
                "source_opaque_response_id": source["source_opaque_response_id"],
                "sample_id": source["sample_id"],
                "category": source["category"],
                "role": source["role"],
                "source_response_id": source["source_response_id"],
                "is_hidden_duplicate": False,
                "duplicate_of_review_id": "",
                "prompt_sha256": source["prompt_sha256"],
                "response_sha256": source["response_sha256"],
            }
        )

    duplicate_sources = sorted(
        sources,
        key=lambda row: stable_rank(duplicate_seed, row["source_opaque_response_id"]),
    )[:duplicate_count]
    duplicate_rows: list[dict[str, Any]] = []
    for source in duplicate_sources:
        opaque_id = opaque_review_id(
            packet_seed, source["source_opaque_response_id"], "duplicate"
        )
        primary_id = primary_by_source[source["source_opaque_response_id"]]
        packet_rows.append(
            {
                "opaque_review_id": opaque_id,
                "prompt": source["prompt"],
                "response": source["response"],
            }
        )
        key_rows.append(
            {
                "opaque_review_id": opaque_id,
                "source_opaque_response_id": source["source_opaque_response_id"],
                "sample_id": source["sample_id"],
                "category": source["category"],
                "role": source["role"],
                "source_response_id": source["source_response_id"],
                "is_hidden_duplicate": True,
                "duplicate_of_review_id": primary_id,
                "prompt_sha256": source["prompt_sha256"],
                "response_sha256": source["response_sha256"],
            }
        )
        duplicate_rows.append(
            {
                "source_opaque_response_id": source["source_opaque_response_id"],
                "primary_opaque_review_id": primary_id,
                "duplicate_opaque_review_id": opaque_id,
            }
        )

    random.Random(packet_seed).shuffle(packet_rows)
    template_rows = [
        {
            "opaque_review_id": row["opaque_review_id"],
            "detail_retention": "",
            "safe_actionability": "",
            "naturalness": "",
            "clean_safe": "",
            "short_reason": "",
        }
        for row in packet_rows
    ]
    packet_ids = [row["opaque_review_id"] for row in packet_rows]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("duplicate opaque_review_id in source packet")
    visible_fields = {"opaque_review_id", "prompt", "response"}
    if any(set(row) != visible_fields for row in packet_rows):
        raise ValueError("source packet leaks hidden fields")

    key_fields = [
        "opaque_review_id",
        "source_opaque_response_id",
        "sample_id",
        "category",
        "role",
        "source_response_id",
        "is_hidden_duplicate",
        "duplicate_of_review_id",
        "prompt_sha256",
        "response_sha256",
    ]
    write_jsonl(SOURCE_PACKET, packet_rows, require_equal_if_exists=True)
    write_csv(SOURCE_KEY, key_rows, key_fields, require_equal_if_exists=True)
    write_csv(
        SOURCE_DUPLICATES,
        duplicate_rows,
        [
            "source_opaque_response_id",
            "primary_opaque_review_id",
            "duplicate_opaque_review_id",
        ],
        require_equal_if_exists=True,
    )
    write_jsonl(SOURCE_A_TEMPLATE, template_rows, require_equal_if_exists=True)
    write_jsonl(SOURCE_B_TEMPLATE, template_rows, require_equal_if_exists=True)
    write_jsonl(SOURCE_C_PACKET, [], require_equal_if_exists=True)
    write_json(INPUT_VALIDATION, validation)

    manifest = {
        "name": "step3n_r1_source_packet_manifest_v1",
        "status": "frozen_before_fresh_source_labels",
        "git_head_before_packet_freeze": git_head(),
        "primary_rows": len(sources),
        "hidden_duplicate_rows": len(duplicate_rows),
        "packet_rows": len(packet_rows),
        "visible_fields": sorted(visible_fields),
        "source_roles_hidden": True,
        "source_response_ids_hidden": True,
        "duplicate_links_hidden": True,
        "aggregate_coverage_hidden": True,
        "packet_sha256": sha256(SOURCE_PACKET),
        "hidden_key_sha256": sha256(SOURCE_KEY),
        "duplicate_key_sha256": sha256(SOURCE_DUPLICATES),
        "reviewer_a_template_sha256": sha256(SOURCE_A_TEMPLATE),
        "reviewer_b_template_sha256": sha256(SOURCE_B_TEMPLATE),
        "review_rubric_sha256": sha256(resolve_repo_path(reg["review_rubric"])),
        "heldout_opened": False,
        "core100_opened_for_n0": False,
        "old_labels_exposed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(SOURCE_PACKET_MANIFEST, manifest)

    artifact_paths = [
        REGISTRY,
        resolve_repo_path(reg["review_rubric"]),
        SOURCE_PACKET,
        SOURCE_KEY,
        SOURCE_DUPLICATES,
        SOURCE_A_TEMPLATE,
        SOURCE_B_TEMPLATE,
        INPUT_VALIDATION,
    ]
    protocol_manifest = {
        "name": "step3n_r1_n0_protocol_manifest_v1",
        "status": "source_packet_frozen_before_any_fresh_label",
        "artifacts": [
            {"path": repo_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        ],
        "heldout_opened": False,
        "core100_opened_for_n0": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(PROTOCOL_MANIFEST, protocol_manifest)

    report = f"""# Step3N-R1 N0 Source Protocol Freeze

Status: source qualification packet frozen before fresh reviewer labels.

| item | count |
| --- | ---: |
| eligible source responses | {len(sources)} |
| safety-training prompts | {validation['training_prompts']} |
| hidden duplicate rows | {len(duplicate_rows)} |
| visible packet rows | {len(packet_rows)} |

Reviewers see only opaque IDs, prompts, and responses. Source roles, source
response IDs, duplicate links, historical quality labels, and aggregate prompt
coverage remain hidden. Reviewer A and Reviewer B must be independent
no-context sessions. The main research session may not supply labels.

This freeze authorizes only source review. It does not authorize generic
counterfactual writing, matching, optimizer construction, model scoring,
training, core100 access, proxy-dev outcome access, or heldout access.
"""
    PROTOCOL_REPORT.write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def raw_agreement(pairs: list[tuple[Any, Any]]) -> float:
    if not pairs:
        raise ValueError("agreement requires at least one pair")
    return sum(left == right for left, right in pairs) / len(pairs)


def linear_weight(left: int, right: int, categories: list[int]) -> float:
    span = max(categories) - min(categories)
    return 1.0 if span == 0 else 1.0 - abs(left - right) / span


def weighted_agreement(pairs: list[tuple[int, int]], categories: list[int]) -> float:
    return sum(linear_weight(left, right, categories) for left, right in pairs) / len(pairs)


def gwet_ac1(pairs: list[tuple[Any, Any]], labels: list[Any]) -> float | None:
    if not pairs:
        return None
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {label: pooled[label] / (2 * len(pairs)) for label in labels}
    expected = sum(prob * (1 - prob) for prob in probabilities.values()) / (len(labels) - 1)
    if math.isclose(expected, 1.0):
        return None
    return (raw_agreement(pairs) - expected) / (1.0 - expected)


def gwet_ac2(pairs: list[tuple[int, int]], categories: list[int]) -> float | None:
    if not pairs:
        return None
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {
        category: pooled[category] / (2 * len(pairs)) for category in categories
    }
    expected = 0.0
    for left in categories:
        for right in categories:
            expected += (
                linear_weight(left, right, categories)
                * probabilities[left]
                * (1.0 - probabilities[right])
                / (len(categories) - 1)
            )
    if math.isclose(expected, 1.0):
        return None
    return (weighted_agreement(pairs, categories) - expected) / (1.0 - expected)


def ordinal_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    categories = [0, 1, 2]
    return {
        "n": len(pairs),
        "exact_raw_agreement": raw_agreement(pairs),
        "linear_weighted_agreement": weighted_agreement(pairs, categories),
        "gwet_ac2": gwet_ac2(pairs, categories),
        "severe_disagreement_count": sum(abs(left - right) == 2 for left, right in pairs),
    }


def nominal_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    yes_yes = sum(left == right == "yes" for left, right in pairs)
    no_no = sum(left == right == "no" for left, right in pairs)
    discordant = sum(left != right for left, right in pairs)
    positive_denominator = 2 * yes_yes + discordant
    negative_denominator = 2 * no_no + discordant
    return {
        "n": len(pairs),
        "raw_agreement": raw_agreement(pairs),
        "gwet_ac1": gwet_ac1(pairs, ["yes", "no"]),
        "positive_agreement": (
            2 * yes_yes / positive_denominator if positive_denominator else None
        ),
        "negative_agreement": (
            2 * no_no / negative_denominator if negative_denominator else None
        ),
        "pooled_yes_prevalence": sum(
            value == "yes" for pair in pairs for value in pair
        )
        / (2 * len(pairs)),
    }


def validate_review(
    path: Path, expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != set(REVIEW_FIELDS):
            raise ValueError(
                f"unexpected fields in {path} for {row.get('opaque_review_id')}: {sorted(row)}"
            )
        opaque_id = row["opaque_review_id"]
        if opaque_id in by_id:
            raise ValueError(f"duplicate review ID in {path}: {opaque_id}")
        for field in ORDINAL_FIELDS:
            if isinstance(row[field], str) and row[field].isdigit():
                row[field] = int(row[field])
            if row[field] not in ORDINAL_VALUES:
                raise ValueError(f"bad {field} in {path}: {row[field]}")
        for field in NOMINAL_FIELDS:
            if row[field] not in BOOL_VALUES:
                raise ValueError(f"bad {field} in {path}: {row[field]}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ValueError(f"missing short_reason in {path}: {opaque_id}")
        by_id[opaque_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(
            f"review IDs differ in {path}: "
            f"missing={sorted(expected_ids - set(by_id))}, "
            f"extra={sorted(set(by_id) - expected_ids)}"
        )
    return by_id


def source_packet_by_id() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(SOURCE_PACKET)
    by_id = {row["opaque_review_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate ID in frozen source packet")
    return by_id


def freeze_source_ab() -> dict[str, Any]:
    packet = source_packet_by_id()
    reviewer_a = validate_review(SOURCE_A, set(packet))
    reviewer_b = validate_review(SOURCE_B, set(packet))
    result = {
        "name": "step3n_r1_source_ab_freeze_v1",
        "packet_rows": len(packet),
        "packet_sha256": sha256(SOURCE_PACKET),
        "hidden_key_sha256": sha256(SOURCE_KEY),
        "reviewer_a_rows": len(reviewer_a),
        "reviewer_b_rows": len(reviewer_b),
        "reviewer_a_sha256": sha256(SOURCE_A),
        "reviewer_b_sha256": sha256(SOURCE_B),
        "ab_compared_only_after_both_files_complete": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(SOURCE_AB_FREEZE, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def primary_review_ids(key_rows: list[dict[str, str]]) -> list[str]:
    return [
        row["opaque_review_id"]
        for row in key_rows
        if row["is_hidden_duplicate"] == "False"
    ]


def source_disputes(
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    primary_ids: Iterable[str],
) -> set[str]:
    return {
        opaque_id
        for opaque_id in primary_ids
        if any(
            reviewer_a[opaque_id][field] != reviewer_b[opaque_id][field]
            for field in DECISION_FIELDS
        )
    }


def prepare_source_adjudication() -> dict[str, Any]:
    if not SOURCE_AB_FREEZE.exists():
        freeze_source_ab()
    packet = source_packet_by_id()
    key_rows = read_csv(SOURCE_KEY)
    expected = set(packet)
    reviewer_a = validate_review(SOURCE_A, expected)
    reviewer_b = validate_review(SOURCE_B, expected)
    disputes = source_disputes(
        reviewer_a, reviewer_b, primary_review_ids(key_rows)
    )
    rows = [
        {
            "opaque_review_id": opaque_id,
            "prompt": packet[opaque_id]["prompt"],
            "response": packet[opaque_id]["response"],
        }
        for opaque_id in sorted(disputes)
    ]
    write_jsonl(SOURCE_C_PACKET, rows)
    result = {
        "name": "step3n_r1_source_reviewer_c_packet_manifest_v1",
        "rows": len(rows),
        "packet_sha256": sha256(SOURCE_C_PACKET),
        "reviewer_a_sha256": sha256(SOURCE_A),
        "reviewer_b_sha256": sha256(SOURCE_B),
        "reviewer_labels_exposed": False,
        "reviewer_reasons_exposed": False,
        "hidden_key_exposed": False,
        "duplicate_rows_excluded": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(SOURCE_C_MANIFEST, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def duplicate_consistency(
    review: dict[str, dict[str, Any]], key_rows: list[dict[str, str]]
) -> dict[str, Any]:
    duplicates = [row for row in key_rows if row["is_hidden_duplicate"] == "True"]
    full_vector_count = 0
    field_counts = Counter()
    details = []
    for duplicate in duplicates:
        primary_id = duplicate["duplicate_of_review_id"]
        duplicate_id = duplicate["opaque_review_id"]
        matches = {
            field: review[primary_id][field] == review[duplicate_id][field]
            for field in DECISION_FIELDS
        }
        for field, matched in matches.items():
            field_counts[field] += int(matched)
        full_match = all(matches.values())
        full_vector_count += int(full_match)
        details.append(
            {
                "primary_opaque_review_id": primary_id,
                "duplicate_opaque_review_id": duplicate_id,
                "field_matches": matches,
                "full_vector_match": full_match,
            }
        )
    return {
        "denominator": len(duplicates),
        "full_vector_consistent_count": full_vector_count,
        "field_consistent_counts": dict(field_counts),
        "details": details,
    }


def is_source_eligible(row: dict[str, Any], reg: dict[str, Any]) -> bool:
    rule = reg["source_eligibility"]
    return (
        row["detail_retention"] == int(rule["detail_retention"])
        and row["safe_actionability"] == int(rule["safe_actionability"])
        and row["naturalness"] == rule["naturalness"]
        and row["clean_safe"] == rule["clean_safe"]
    )


def finalize_source_review() -> dict[str, Any]:
    reg = registry()
    gates_cfg = reg["review_gates"]
    packet = source_packet_by_id()
    key_rows = read_csv(SOURCE_KEY)
    key_by_id = {row["opaque_review_id"]: row for row in key_rows}
    expected = set(packet)
    reviewer_a = validate_review(SOURCE_A, expected)
    reviewer_b = validate_review(SOURCE_B, expected)
    primary_ids = primary_review_ids(key_rows)
    disputes = source_disputes(reviewer_a, reviewer_b, primary_ids)
    reviewer_c = validate_review(SOURCE_C, disputes) if disputes else {}

    final_rows: list[dict[str, Any]] = []
    final_by_id: dict[str, dict[str, Any]] = {}
    for opaque_id in sorted(primary_ids):
        row: dict[str, Any] = {"opaque_review_id": opaque_id}
        adjudicated_fields: list[str] = []
        for field in DECISION_FIELDS:
            if reviewer_a[opaque_id][field] == reviewer_b[opaque_id][field]:
                row[field] = reviewer_a[opaque_id][field]
            else:
                row[field] = reviewer_c[opaque_id][field]
                adjudicated_fields.append(field)
        row["short_reason"] = (
            reviewer_c[opaque_id]["short_reason"]
            if adjudicated_fields
            else reviewer_a[opaque_id]["short_reason"]
        )
        row["adjudicated_fields"] = adjudicated_fields
        final_rows.append(row)
        final_by_id[opaque_id] = row
    write_jsonl(SOURCE_FINAL, final_rows)

    ordinal = {
        field: ordinal_metrics(
            [(reviewer_a[opaque_id][field], reviewer_b[opaque_id][field]) for opaque_id in primary_ids]
        )
        for field in ORDINAL_FIELDS
    }
    nominal = {
        field: nominal_metrics(
            [(reviewer_a[opaque_id][field], reviewer_b[opaque_id][field]) for opaque_id in primary_ids]
        )
        for field in NOMINAL_FIELDS
    }
    duplicate_a = duplicate_consistency(reviewer_a, key_rows)
    duplicate_b = duplicate_consistency(reviewer_b, key_rows)
    review_gates = {
        "detail_gwet_ac2_pass": ordinal["detail_retention"]["gwet_ac2"] is not None
        and ordinal["detail_retention"]["gwet_ac2"] >= float(gates_cfg["detail_gwet_ac2_min"]),
        "actionability_gwet_ac2_pass": ordinal["safe_actionability"]["gwet_ac2"] is not None
        and ordinal["safe_actionability"]["gwet_ac2"]
        >= float(gates_cfg["actionability_gwet_ac2_min"]),
        "naturalness_gwet_ac1_pass": nominal["naturalness"]["gwet_ac1"] is not None
        and nominal["naturalness"]["gwet_ac1"]
        >= float(gates_cfg["naturalness_gwet_ac1_min"]),
        "clean_safe_gwet_ac1_pass": nominal["clean_safe"]["gwet_ac1"] is not None
        and nominal["clean_safe"]["gwet_ac1"]
        >= float(gates_cfg["clean_safe_gwet_ac1_min"]),
        "clean_safe_raw_agreement_pass": nominal["clean_safe"]["raw_agreement"]
        >= float(gates_cfg["clean_safe_raw_agreement_min"]),
        "reviewer_a_duplicate_pass": duplicate_a["full_vector_consistent_count"]
        >= int(gates_cfg["duplicate_full_vector_consistency_min_count"]),
        "reviewer_b_duplicate_pass": duplicate_b["full_vector_consistent_count"]
        >= int(gates_cfg["duplicate_full_vector_consistency_min_count"]),
        "all_eligibility_disagreements_adjudicated": set(reviewer_c) == disputes,
    }
    review_gates["overall_review_reliability_pass"] = all(review_gates.values())

    eligible_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for opaque_id in primary_ids:
        if not is_source_eligible(final_by_id[opaque_id], reg):
            continue
        hidden = key_by_id[opaque_id]
        eligible_by_prompt.setdefault(hidden["sample_id"], []).append(hidden)

    selected_rows: list[dict[str, Any]] = []
    selection_salt = reg["positive_selection_salt"]
    for sample_id in sorted(eligible_by_prompt):
        candidates = sorted(
            eligible_by_prompt[sample_id],
            key=lambda row: stable_rank(selection_salt, row["source_response_id"]),
        )
        selected = candidates[0]
        visible = packet[selected["opaque_review_id"]]
        selected_rows.append(
            {
                "positive_id": f"step3n_pos_{stable_rank(selection_salt, sample_id, selected['source_response_id'])[:18]}",
                "sample_id": sample_id,
                "category": selected["category"],
                "source_role": selected["role"],
                "source_response_id": selected["source_response_id"],
                "source_opaque_response_id": selected["source_opaque_response_id"],
                "source_opaque_review_id": selected["opaque_review_id"],
                "prompt": visible["prompt"],
                "positive_response": visible["response"],
                "prompt_sha256": selected["prompt_sha256"],
                "positive_response_sha256": selected["response_sha256"],
                "selection_rank_sha256": stable_rank(
                    selection_salt, selected["source_response_id"]
                ),
            }
        )

    category_prompt_counts = Counter(row["category"] for row in selected_rows)
    multi_prompt_categories = sum(count >= 2 for count in category_prompt_counts.values())
    tuple_cfg = reg["tuple_gate"]
    coverage_gates = {
        "minimum_source_prompts_pass": len(selected_rows)
        >= int(tuple_cfg["minimum_complete_training_tuples"]),
        "minimum_multi_prompt_categories_pass": multi_prompt_categories
        >= int(tuple_cfg["minimum_multi_prompt_categories"]),
    }
    source_stage_pass = review_gates["overall_review_reliability_pass"] and all(
        coverage_gates.values()
    )
    write_jsonl(SELECTED_POSITIVES, selected_rows if source_stage_pass else [])

    agreement = {
        "name": "step3n_r1_source_review_agreement_v1",
        "primary_rows": len(primary_ids),
        "hidden_duplicate_rows": len(key_rows) - len(primary_ids),
        "ordinal": ordinal,
        "nominal": nominal,
        "reviewer_a_duplicate_consistency": duplicate_a,
        "reviewer_b_duplicate_consistency": duplicate_b,
        "disagreement_rows": len(disputes),
        "reviewer_c_rows": len(reviewer_c),
        "review_gates": review_gates,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(SOURCE_AGREEMENT, agreement)

    decision = {
        "name": "step3n_r1_source_route_decision_v1",
        "status": "PASS" if source_stage_pass else "STOP",
        "source_stage_pass": source_stage_pass,
        "qualified_source_prompts": len(selected_rows),
        "source_prompt_slack_over_36": len(selected_rows)
        - int(tuple_cfg["minimum_complete_training_tuples"]),
        "eligible_response_rows": sum(len(rows) for rows in eligible_by_prompt.values()),
        "category_prompt_counts": dict(sorted(category_prompt_counts.items())),
        "multi_prompt_categories": multi_prompt_categories,
        "review_gates": review_gates,
        "coverage_gates": coverage_gates,
        "selected_positive_file_written": source_stage_pass,
        "next_action": (
            "freeze_generic_writer_packet_and_pre_review_matching_inputs"
            if source_stage_pass
            else "close_pcsr_without_threshold_or_source_relaxation"
        ),
        "generic_counterfactuals_qualified": False,
        "matching_frozen": False,
        "n0_complete": False,
        "training_authorized": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(SOURCE_DECISION, decision)

    source_report = f"""# Step3N-R1 N0 Source Qualification Report

Status: **{decision['status']}** at the source-response gate.

| measure | value |
| --- | ---: |
| source responses reviewed | {len(primary_ids)} |
| A/B disagreement rows adjudicated | {len(reviewer_c)}/{len(disputes)} |
| qualified source prompts | {len(selected_rows)} |
| prompt slack over the 36 minimum | {decision['source_prompt_slack_over_36']} |
| represented multi-prompt categories | {multi_prompt_categories} |

| field | raw agreement | Gwet coefficient |
| --- | ---: | ---: |
| detail retention | {ordinal['detail_retention']['exact_raw_agreement']:.3f} | {ordinal['detail_retention']['gwet_ac2']} |
| safe actionability | {ordinal['safe_actionability']['exact_raw_agreement']:.3f} | {ordinal['safe_actionability']['gwet_ac2']} |
| naturalness | {nominal['naturalness']['raw_agreement']:.3f} | {nominal['naturalness']['gwet_ac1']} |
| clean safety | {nominal['clean_safe']['raw_agreement']:.3f} | {nominal['clean_safe']['gwet_ac1']} |

Clean-safety positive agreement is
`{nominal['clean_safe']['positive_agreement']}`; negative agreement is
`{nominal['clean_safe']['negative_agreement']}`; pooled yes prevalence is
`{nominal['clean_safe']['pooled_yes_prevalence']}`.

Reviewer A duplicate consistency:
`{duplicate_a['full_vector_consistent_count']}/{duplicate_a['denominator']}`.
Reviewer B duplicate consistency:
`{duplicate_b['full_vector_consistent_count']}/{duplicate_b['denominator']}`.

This is not an N0-R1 tuple pass. Generic-safe counterfactual review,
prompt-mismatch edge review, final cycle-cover selection, exact tokenizer
checks, and leakage diagnostics remain pending. Training is not authorized.
"""
    SOURCE_REPORT.write_text(source_report, encoding="utf-8", newline="\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


def verify_protocol() -> dict[str, Any]:
    manifest = read_json(PROTOCOL_MANIFEST)
    checks = []
    failures = []
    for artifact in manifest["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        actual = sha256(path) if path.exists() else None
        passed = actual == artifact["sha256"]
        checks.append({"path": artifact["path"], "passed": passed})
        if not passed:
            failures.append(artifact["path"])
    result = {
        "name": "step3n_r1_n0_protocol_verification_v1",
        "checked": len(checks),
        "failures": failures,
        "passed": not failures,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(DIAGNOSTICS / "step3n_r1_n0_protocol_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_synthetic_tests() -> dict[str, Any]:
    perfect_ordinal = [(0, 0), (1, 1), (2, 2)]
    zero_variance = [(2, 2)] * 8
    nominal = [("yes", "yes"), ("yes", "no"), ("no", "no")]
    assert math.isclose(gwet_ac2(perfect_ordinal, [0, 1, 2]) or 0.0, 1.0)
    assert math.isclose(gwet_ac2(zero_variance, [0, 1, 2]) or 0.0, 1.0)
    metrics = nominal_metrics(nominal)
    assert math.isclose(metrics["raw_agreement"], 2 / 3)
    assert math.isclose(metrics["positive_agreement"] or 0.0, 2 / 3)
    assert math.isclose(metrics["negative_agreement"] or 0.0, 2 / 3)

    fixture = [
        {
            "opaque_review_id": "q1",
            "detail_retention": 2,
            "safe_actionability": 2,
            "naturalness": "yes",
            "clean_safe": "yes",
            "short_reason": "fixture",
        }
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "review.jsonl"
        write_jsonl(path, fixture)
        validated = validate_review(path, {"q1"})
    assert is_source_eligible(validated["q1"], registry())

    result = {
        "name": "step3n_r1_n0_synthetic_tests_v1",
        "perfect_ordinal_ac2": gwet_ac2(perfect_ordinal, [0, 1, 2]),
        "zero_variance_ac2": gwet_ac2(zero_variance, [0, 1, 2]),
        "nominal_fixture": metrics,
        "strict_review_schema_pass": True,
        "source_eligibility_fixture_pass": True,
        "passed": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(SYNTHETIC_TESTS, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "test-fixtures",
            "freeze-source-packet",
            "verify-protocol",
            "freeze-source-ab",
            "prepare-source-adjudication",
            "finalize-source-review",
        ],
    )
    args = parser.parse_args()
    if args.command == "test-fixtures":
        run_synthetic_tests()
    elif args.command == "freeze-source-packet":
        if not SYNTHETIC_TESTS.exists() or not read_json(SYNTHETIC_TESTS).get("passed"):
            run_synthetic_tests()
        freeze_source_packet()
    elif args.command == "verify-protocol":
        verify_protocol()
    elif args.command == "freeze-source-ab":
        freeze_source_ab()
    elif args.command == "prepare-source-adjudication":
        prepare_source_adjudication()
    elif args.command == "finalize-source-review":
        finalize_source_review()


if __name__ == "__main__":
    main()
