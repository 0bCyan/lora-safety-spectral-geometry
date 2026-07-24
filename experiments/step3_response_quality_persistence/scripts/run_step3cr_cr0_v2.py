from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import prepare_step3cr_cr0_protocol as protocol


ROOT = protocol.ROOT
REPO_ROOT = Path.cwd()
REGISTRY = ROOT / "configs/step3cr_recovery_registry.json"
RUBRIC = ROOT / "review_rubrics/step3cr_prompt_aware_pairwise_rubric_v2.md"
SCRIPT = ROOT / "scripts/run_step3cr_cr0_v2.py"
V1_BASE = ROOT / "results/step3cr_cr0"
V1_PACKET = V1_BASE / "packets/step3cr_cr0_pairwise_packet.jsonl"
V1_KEY = V1_BASE / "packets/step3cr_cr0_pairwise_hidden_key.csv"
V1_ROUTE = V1_BASE / "reports/step3cr_cr0_route_decision.json"
BASE = ROOT / "results/step3cr_cr0_v2"
PACKETS = BASE / "packets"
REVIEWS = BASE / "reviews"
REPORTS = BASE / "reports"
DIAGNOSTICS = BASE / "diagnostics"

PAIR_PACKET = PACKETS / "step3cr_cr0_v2_pairwise_packet.jsonl"
PAIR_KEY = PACKETS / "step3cr_cr0_v2_pairwise_hidden_key.csv"
DUPLICATE_KEY = PACKETS / "step3cr_cr0_v2_reversed_duplicate_key.csv"
A_TEMPLATE = PACKETS / "step3cr_cr0_v2_reviewer_a_template.jsonl"
B_TEMPLATE = PACKETS / "step3cr_cr0_v2_reviewer_b_template.jsonl"
C_PACKET = PACKETS / "step3cr_cr0_v2_reviewer_c_packet.jsonl"
PACKET_MANIFEST = PACKETS / "step3cr_cr0_v2_packet_manifest.json"
C_PACKET_MANIFEST = PACKETS / "step3cr_cr0_v2_reviewer_c_packet_manifest.json"

A_REVIEW = REVIEWS / "step3cr_cr0_v2_reviewer_a.jsonl"
B_REVIEW = REVIEWS / "step3cr_cr0_v2_reviewer_b.jsonl"
C_REVIEW = REVIEWS / "step3cr_cr0_v2_reviewer_c.jsonl"
PROVENANCE = REVIEWS / "step3cr_cr0_v2_reviewer_provenance.json"
AB_FREEZE = REVIEWS / "step3cr_cr0_v2_ab_freeze_manifest.json"
FINAL_DISPLAY = REVIEWS / "step3cr_cr0_v2_final_display_labels.jsonl"
FINAL_CANONICAL = REVIEWS / "step3cr_cr0_v2_final_canonical_labels.jsonl"

PREP_VALIDATION = DIAGNOSTICS / "step3cr_cr0_v2_preparation_validation.json"
ARCHIVE_MANIFEST = REPORTS / "step3cr_cr0_v1_archive_manifest.json"
CLARIFICATION_REPORT = REPORTS / "Step3CR_CR0_v2_rubric_clarification.md"
AGREEMENT = REPORTS / "step3cr_cr0_v2_review_agreement.json"
BIAS = REPORTS / "step3cr_cr0_v2_bias_and_reversal_report.json"
CATEGORY = REPORTS / "step3cr_cr0_v2_category_summary.json"
ROUTE = REPORTS / "step3cr_cr0_v2_route_decision.json"
ROUTE_REPORT = REPORTS / "Step3CR_CR0_v2_route_report.md"
ARTIFACT_MANIFEST = REPORTS / "step3cr_cr0_v2_artifact_manifest.json"

SESSION_ID = "step3cr_cr0_review_session_v2_20260724"
RANDOMIZATION_SEED = 20260724
SOURCE_COMMIT = "1a2f407"
DECISION_FIELDS = [
    "pair_preference",
    "left_specific_safe_next_step",
    "right_specific_safe_next_step",
    "left_natural",
    "right_natural",
]
REVIEW_FIELDS = ["opaque_pair_id", *DECISION_FIELDS, "short_reason"]
KEY_FIELDS = [
    "opaque_pair_id",
    "canonical_pair_id",
    "sample_id",
    "category",
    "is_reversed_duplicate",
    "duplicate_of_opaque_pair_id",
    "construction_rule",
    "canonical_left_opaque_response_id",
    "canonical_right_opaque_response_id",
    "display_left_opaque_response_id",
    "display_right_opaque_response_id",
    "display_left_source_response_id",
    "display_right_source_response_id",
    "display_left_role",
    "display_right_role",
    "display_left_sha256",
    "display_right_sha256",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def opaque_token(namespace: str, index: int) -> str:
    payload = f"{namespace}:{RANDOMIZATION_SEED}:{index}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def by_id(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result = {str(row[field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {field}")
    return result


def reconstruct_v1_canonical() -> list[dict[str, Any]]:
    packet = by_id(protocol.read_jsonl(V1_PACKET), "opaque_pair_id")
    key_rows = [row for row in read_csv(V1_KEY) if row["is_reversed_duplicate"] == "False"]
    canonical = []
    for key in key_rows:
        shown = packet[key["opaque_pair_id"]]
        display = {
            key["display_left_opaque_response_id"]: {
                "opaque_response_id": key["display_left_opaque_response_id"],
                "source_response_id": key["display_left_source_response_id"],
                "role": key["display_left_role"],
                "response": shown["left_response"],
            },
            key["display_right_opaque_response_id"]: {
                "opaque_response_id": key["display_right_opaque_response_id"],
                "source_response_id": key["display_right_source_response_id"],
                "role": key["display_right_role"],
                "response": shown["right_response"],
            },
        }
        canonical.append({
            "source_canonical_pair_id": key["canonical_pair_id"],
            "sample_id": key["sample_id"],
            "category": key["category"],
            "construction_rule": key["construction_rule"],
            "prompt": shown["prompt"],
            "canonical_left": display[key["canonical_left_opaque_response_id"]],
            "canonical_right": display[key["canonical_right_opaque_response_id"]],
        })
    if len(canonical) != 40:
        raise ValueError(f"expected 40 v1 canonical pairs, found {len(canonical)}")
    return canonical


def make_key_row(
    pair: dict[str, Any],
    opaque_id: str,
    canonical_id: str,
    left: dict[str, str],
    right: dict[str, str],
    duplicate_of: str,
) -> dict[str, Any]:
    return {
        "opaque_pair_id": opaque_id,
        "canonical_pair_id": canonical_id,
        "sample_id": pair["sample_id"],
        "category": pair["category"],
        "is_reversed_duplicate": bool(duplicate_of),
        "duplicate_of_opaque_pair_id": duplicate_of,
        "construction_rule": pair["construction_rule"],
        "canonical_left_opaque_response_id": pair["canonical_left"]["opaque_response_id"],
        "canonical_right_opaque_response_id": pair["canonical_right"]["opaque_response_id"],
        "display_left_opaque_response_id": left["opaque_response_id"],
        "display_right_opaque_response_id": right["opaque_response_id"],
        "display_left_source_response_id": left["source_response_id"],
        "display_right_source_response_id": right["source_response_id"],
        "display_left_role": left["role"],
        "display_right_role": right["role"],
        "display_left_sha256": sha256_text(left["response"]),
        "display_right_sha256": sha256_text(right["response"]),
    }


def packet_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    sides = sorted([sha256_text(row["left_response"]), sha256_text(row["right_response"])])
    return sha256_text(row["prompt"]), sides[0], sides[1]


def archive_v1() -> dict[str, Any]:
    files = sorted(path for path in V1_BASE.rglob("*") if path.is_file())
    artifacts = [
        {"path": path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in files
    ]
    manifest = {
        "name": "step3cr_cr0_v1_archive_manifest_v1",
        "source_commit": SOURCE_COMMIT,
        "source_commit_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode == 0,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "v1_labels_reused_in_v2": False,
        "v1_packet_order_reused_in_v2": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(ARCHIVE_MANIFEST, manifest)
    return manifest


def prepare() -> dict[str, Any]:
    route = protocol.read_json(V1_ROUTE)
    if route["cr0_measurement_gate_pass"] or route["complete_revision_used"]:
        raise ValueError("v2 recovery is not authorized by the frozen v1 route")
    if not git_head().startswith(SOURCE_COMMIT):
        raise ValueError(f"prepare v2 from exact source commit {SOURCE_COMMIT}")
    for directory in [PACKETS, REVIEWS, REPORTS, DIAGNOSTICS]:
        directory.mkdir(parents=True, exist_ok=True)
    archive = archive_v1()
    source = reconstruct_v1_canonical()
    rng = random.Random(RANDOMIZATION_SEED)
    rng.shuffle(source)
    duplicate_indexes = set(rng.sample(range(len(source)), 8))
    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    serial = 1
    for index, pair in enumerate(source):
        canonical_id = f"step3cr2_cp_{opaque_token('canonical', index)}"
        flip = rng.choice([False, True])
        left = pair["canonical_right"] if flip else pair["canonical_left"]
        right = pair["canonical_left"] if flip else pair["canonical_right"]
        opaque_id = f"step3cr2_pair_{opaque_token('pair', serial)}"
        primary = {
            "opaque_pair_id": opaque_id,
            "prompt": pair["prompt"],
            "left_response": left["response"],
            "right_response": right["response"],
        }
        packet_rows.append(primary)
        key_rows.append(make_key_row(pair, opaque_id, canonical_id, left, right, ""))
        serial += 1
        if index in duplicate_indexes:
            duplicate_id = f"step3cr2_pair_{opaque_token('pair', serial)}"
            packet_rows.append({
                "opaque_pair_id": duplicate_id,
                "prompt": pair["prompt"],
                "left_response": right["response"],
                "right_response": left["response"],
            })
            key_rows.append(make_key_row(pair, duplicate_id, canonical_id, right, left, opaque_id))
            duplicate_rows.append({
                "canonical_pair_id": canonical_id,
                "primary_opaque_pair_id": opaque_id,
                "reversed_duplicate_opaque_pair_id": duplicate_id,
            })
            serial += 1
    rng.shuffle(packet_rows)
    write_jsonl(PAIR_PACKET, packet_rows)
    write_csv(PAIR_KEY, key_rows, KEY_FIELDS)
    write_csv(DUPLICATE_KEY, duplicate_rows, ["canonical_pair_id", "primary_opaque_pair_id", "reversed_duplicate_opaque_pair_id"])
    templates = [{field: row["opaque_pair_id"] if field == "opaque_pair_id" else "" for field in REVIEW_FIELDS} for row in packet_rows]
    write_jsonl(A_TEMPLATE, templates)
    write_jsonl(B_TEMPLATE, templates)
    write_jsonl(C_PACKET, [])

    v1_packet = by_id(protocol.read_jsonl(V1_PACKET), "opaque_pair_id")
    v1_primary_ids = {row["opaque_pair_id"] for row in read_csv(V1_KEY) if row["is_reversed_duplicate"] == "False"}
    v2_primary_ids = {row["opaque_pair_id"] for row in key_rows if row["is_reversed_duplicate"] is False}
    v1_signatures = sorted(packet_signature(v1_packet[opaque_id]) for opaque_id in v1_primary_ids)
    v2_by_id = by_id(packet_rows, "opaque_pair_id")
    v2_signatures = sorted(packet_signature(v2_by_id[opaque_id]) for opaque_id in v2_primary_ids)
    old_ids = set(v1_packet)
    new_ids = set(v2_by_id)
    duplicate_exact = all(
        v2_by_id[row["primary_opaque_pair_id"]]["prompt"] == v2_by_id[row["reversed_duplicate_opaque_pair_id"]]["prompt"]
        and v2_by_id[row["primary_opaque_pair_id"]]["left_response"] == v2_by_id[row["reversed_duplicate_opaque_pair_id"]]["right_response"]
        and v2_by_id[row["primary_opaque_pair_id"]]["right_response"] == v2_by_id[row["reversed_duplicate_opaque_pair_id"]]["left_response"]
        for row in duplicate_rows
    )
    validation = {
        "name": "step3cr_cr0_v2_preparation_validation_v1",
        "source_commit": SOURCE_COMMIT,
        "v1_archive_artifacts": archive["artifact_count"],
        "packet_rows": len(packet_rows),
        "canonical_pairs": len(v2_primary_ids),
        "reversed_duplicates": len(duplicate_rows),
        "canonical_content_multiset_equal_to_v1": v1_signatures == v2_signatures,
        "opaque_id_overlap_with_v1": len(old_ids & new_ids),
        "packet_order_changed": [packet_signature(row) for row in protocol.read_jsonl(V1_PACKET)] != [packet_signature(row) for row in packet_rows],
        "reversed_duplicates_exact": duplicate_exact,
        "rubric_sha256": sha256(RUBRIC),
        "old_review_labels_read_or_reused": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    validation["passed"] = (
        validation["packet_rows"] == 48
        and validation["canonical_pairs"] == 40
        and validation["reversed_duplicates"] == 8
        and validation["canonical_content_multiset_equal_to_v1"]
        and validation["opaque_id_overlap_with_v1"] == 0
        and validation["packet_order_changed"]
        and validation["reversed_duplicates_exact"]
    )
    if not validation["passed"]:
        raise ValueError(f"v2 preparation validation failed: {validation}")
    write_json(PREP_VALIDATION, validation)
    manifest = {
        "name": "step3cr_cr0_v2_packet_manifest_v1",
        "source_commit": SOURCE_COMMIT,
        "session_id": SESSION_ID,
        "randomization_seed": RANDOMIZATION_SEED,
        "packet_rows": 48,
        "canonical_pairs": 40,
        "reversed_duplicates": 8,
        "packet_sha256": sha256(PAIR_PACKET),
        "hidden_key_sha256": sha256(PAIR_KEY),
        "duplicate_key_sha256": sha256(DUPLICATE_KEY),
        "rubric_sha256": sha256(RUBRIC),
        "reviewer_a_template_sha256": sha256(A_TEMPLATE),
        "reviewer_b_template_sha256": sha256(B_TEMPLATE),
        "fresh_opaque_ids_and_order": True,
        "v1_labels_reused": False,
        "main_executor_assigned_labels": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(PACKET_MANIFEST, manifest)
    CLARIFICATION_REPORT.write_text(
        "# Step3C-R C-R0 v2 Prospective Rubric Clarification\n\n"
        f"Status: frozen after first-session commit `{SOURCE_COMMIT}` and before any v2 label.\n\n"
        "The only permitted complete clarification is now used. The v2 rubric adds an ordered independent-response check, an explicit position-swap check, a narrower tie rule, and an actionable threshold for prompt-specific safe next steps. It does not change any gate, sample, response text, or clean-safety eligibility decision.\n\n"
        "The 40 canonical content pairs are unchanged as a multiset. All opaque pair IDs, pair order, side randomization, and the eight reversed-duplicate selections were rebuilt without reading or reusing v1 review labels.\n\n"
        "Heldout remained closed. No optimizer or scheduler was created, and no training was run.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def normalize_lf(path: Path) -> None:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if raw != normalized:
        path.write_bytes(normalized)


def validate_review(path: Path, expected_order: list[str]) -> dict[str, dict[str, Any]]:
    normalize_lf(path)
    rows = protocol.read_jsonl(path)
    if [row.get("opaque_pair_id") for row in rows] != expected_order:
        raise ValueError(f"{path} does not preserve frozen packet order")
    if len(rows) != len(expected_order):
        raise ValueError(f"{path} wrong row count")
    for row in rows:
        if set(row) != set(REVIEW_FIELDS):
            raise ValueError(f"{path} bad fields for {row.get('opaque_pair_id')}")
        if row["pair_preference"] not in {"left", "right", "tie"}:
            raise ValueError(f"{path} bad preference")
        if any(row[field] not in {"yes", "no"} for field in DECISION_FIELDS[1:]):
            raise ValueError(f"{path} bad yes/no field")
        reason = str(row["short_reason"]).strip()
        if not reason or len(reason.split()) > 50:
            raise ValueError(f"{path} bad short_reason")
    return by_id(rows, "opaque_pair_id")


def packet_order() -> list[str]:
    return [row["opaque_pair_id"] for row in protocol.read_jsonl(PAIR_PACKET)]


def validate_ab() -> dict[str, Any]:
    order = packet_order()
    a = validate_review(A_REVIEW, order)
    b = validate_review(B_REVIEW, order)
    provenance = protocol.read_json(PROVENANCE)
    frozen_provenance = {**provenance, "reviewer_c": None}
    manifest = {
        "name": "step3cr_cr0_v2_ab_freeze_manifest_v1",
        "session_id": SESSION_ID,
        "packet_sha256": sha256(PAIR_PACKET),
        "rubric_sha256": sha256(RUBRIC),
        "reviewer_a_rows": len(a),
        "reviewer_b_rows": len(b),
        "reviewer_a_sha256": sha256(A_REVIEW),
        "reviewer_b_sha256": sha256(B_REVIEW),
        "provenance_before_c": frozen_provenance,
        "v1_labels_reused": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(AB_FREEZE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def require_ab_frozen() -> dict[str, Any]:
    manifest = protocol.read_json(AB_FREEZE)
    checks = {
        A_REVIEW: manifest["reviewer_a_sha256"],
        B_REVIEW: manifest["reviewer_b_sha256"],
        PAIR_PACKET: manifest["packet_sha256"],
        RUBRIC: manifest["rubric_sha256"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen artifact changed: {path}")
    return manifest


def disagreement_rows(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {
        opaque_id: [field for field in DECISION_FIELDS if a[opaque_id][field] != b[opaque_id][field]]
        for opaque_id in a
        if any(a[opaque_id][field] != b[opaque_id][field] for field in DECISION_FIELDS)
    }


def prepare_adjudication() -> dict[str, Any]:
    require_ab_frozen()
    order = packet_order()
    a = validate_review(A_REVIEW, order)
    b = validate_review(B_REVIEW, order)
    packet = by_id(protocol.read_jsonl(PAIR_PACKET), "opaque_pair_id")
    disputes = disagreement_rows(a, b)
    rows = []
    for opaque_id in order:
        if opaque_id not in disputes:
            continue
        rows.append({
            "opaque_pair_id": opaque_id,
            "prompt": packet[opaque_id]["prompt"],
            "left_response": packet[opaque_id]["left_response"],
            "right_response": packet[opaque_id]["right_response"],
            "disagreement_fields": disputes[opaque_id],
            "reviewer_a_short_reason": a[opaque_id]["short_reason"],
            "reviewer_b_short_reason": b[opaque_id]["short_reason"],
        })
    write_jsonl(C_PACKET, rows)
    manifest = {
        "name": "step3cr_cr0_v2_reviewer_c_packet_manifest_v1",
        "session_id": SESSION_ID,
        "packet_rows": len(rows),
        "packet_sha256": sha256(C_PACKET),
        "contains_other_reviewer_labels": False,
        "contains_only_disagreement_rows": True,
        "reviewer_c_must_not_read_hidden_key": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(C_PACKET_MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def load_key() -> dict[str, dict[str, str]]:
    return by_id(read_csv(PAIR_KEY), "opaque_pair_id")


def canonical_primary(reviews: dict[str, dict[str, Any]], key: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    result = {}
    for opaque_id, key_row in key.items():
        if key_row["is_reversed_duplicate"] == "True":
            continue
        left_is_canonical = protocol.display_left_is_canonical_left(key_row)
        specific = protocol.canonicalize_specific(reviews[opaque_id], left_is_canonical)
        natural = protocol.canonicalize_naturalness(reviews[opaque_id], left_is_canonical)
        result[key_row["canonical_pair_id"]] = {
            "preference": protocol.canonicalize_preference(reviews[opaque_id]["pair_preference"], left_is_canonical),
            "canonical_left_specific": specific[0],
            "canonical_right_specific": specific[1],
            "canonical_left_natural": natural[0],
            "canonical_right_natural": natural[1],
        }
    return result


def metric_pairs(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    preference = []
    specificity = []
    naturalness = []
    for canonical_id in sorted(a):
        preference.append((a[canonical_id]["preference"], b[canonical_id]["preference"]))
        for side in ["canonical_left", "canonical_right"]:
            specificity.append((a[canonical_id][f"{side}_specific"], b[canonical_id][f"{side}_specific"]))
            naturalness.append((a[canonical_id][f"{side}_natural"], b[canonical_id][f"{side}_natural"]))
    return preference, specificity, naturalness


def label_prevalence(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_preference": dict(Counter(row["pair_preference"] for row in rows.values())),
        "specificity": dict(Counter(label for row in rows.values() for label in [row["left_specific_safe_next_step"], row["right_specific_safe_next_step"]])),
        "naturalness": dict(Counter(label for row in rows.values() for label in [row["left_natural"], row["right_natural"]])),
    }


def validate_c(disputes: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    expected_order = [row["opaque_pair_id"] for row in protocol.read_jsonl(C_PACKET)]
    if not expected_order:
        if C_REVIEW.exists() and C_REVIEW.stat().st_size:
            raise ValueError("C labels exist without disagreements")
        return {}
    c = validate_review(C_REVIEW, expected_order)
    if set(c) != set(disputes):
        raise ValueError("Reviewer C coverage does not match disagreements")
    manifest = protocol.read_json(C_PACKET_MANIFEST)
    manifest["reviewer_c_rows"] = len(c)
    manifest["reviewer_c_sha256"] = sha256(C_REVIEW)
    write_json(C_PACKET_MANIFEST, manifest)
    return c


def final_labels(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    c: dict[str, dict[str, Any]],
    disputes: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    final = {}
    for opaque_id, a_row in a.items():
        row = dict(a_row)
        row["review_source"] = "A/B_agreement"
        for field in disputes.get(opaque_id, []):
            row[field] = c[opaque_id][field]
            row["review_source"] = "C_disagreement_adjudication"
        if opaque_id in disputes:
            row["adjudication_short_reason"] = c[opaque_id]["short_reason"]
        final[opaque_id] = row
    return final


def summarize() -> dict[str, Any]:
    ab_manifest = require_ab_frozen()
    order = packet_order()
    a = validate_review(A_REVIEW, order)
    b = validate_review(B_REVIEW, order)
    disputes = disagreement_rows(a, b)
    c = validate_c(disputes)
    final = final_labels(a, b, c, disputes)
    write_jsonl(FINAL_DISPLAY, [final[opaque_id] for opaque_id in order])
    key = load_key()
    canonical_rows = []
    for opaque_id, key_row in sorted(key.items(), key=lambda item: item[1]["canonical_pair_id"]):
        if key_row["is_reversed_duplicate"] == "True":
            continue
        left_is_canonical = protocol.display_left_is_canonical_left(key_row)
        specific = protocol.canonicalize_specific(final[opaque_id], left_is_canonical)
        natural = protocol.canonicalize_naturalness(final[opaque_id], left_is_canonical)
        canonical_rows.append({
            "canonical_pair_id": key_row["canonical_pair_id"],
            "opaque_pair_id": opaque_id,
            "pair_preference": protocol.canonicalize_preference(final[opaque_id]["pair_preference"], left_is_canonical),
            "canonical_left_specific_safe_next_step": specific[0],
            "canonical_right_specific_safe_next_step": specific[1],
            "canonical_left_natural": natural[0],
            "canonical_right_natural": natural[1],
            "adjudicated": "yes" if opaque_id in disputes else "no",
        })
    if len(canonical_rows) != 40:
        raise ValueError("canonical denominator is not 40")
    write_jsonl(FINAL_CANONICAL, canonical_rows)

    a_canonical = canonical_primary(a, key)
    b_canonical = canonical_primary(b, key)
    pref_pairs, spec_pairs, natural_pairs = metric_pairs(a_canonical, b_canonical)
    preference = protocol.metric_block(pref_pairs)
    specificity = protocol.metric_block(spec_pairs)
    naturalness = protocol.metric_block(natural_pairs)
    a_reverse = protocol.reversed_consistency(a, key)
    b_reverse = protocol.reversed_consistency(b, key)
    a_bias = protocol.position_bias(a, key)
    b_bias = protocol.position_bias(b, key)
    gate = protocol.read_json(REGISTRY)["cr0_measurement"]
    gates = {
        "pair_preference_agreement_pass": preference["raw_agreement"] >= float(gate["raw_agreement_min"]) and preference["gwet_ac1"] >= float(gate["gwet_ac1_min"]),
        "specificity_agreement_pass": specificity["raw_agreement"] >= float(gate["specific_safe_raw_agreement_min"]) and specificity["gwet_ac1"] >= float(gate["specific_safe_gwet_ac1_min"]),
        "reviewer_a_preference_reversal_pass": protocol.reversal_gate_pass(a_reverse["preference_consistent_count"], a_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_b_preference_reversal_pass": protocol.reversal_gate_pass(b_reverse["preference_consistent_count"], b_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_a_specificity_reversal_pass": protocol.reversal_gate_pass(a_reverse["specificity_consistent_count"], a_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_b_specificity_reversal_pass": protocol.reversal_gate_pass(b_reverse["specificity_consistent_count"], b_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_a_non_tie_pass": a_bias["non_tie_count"] >= int(gate["canonical_non_tie_count_min_per_reviewer"]),
        "reviewer_b_non_tie_pass": b_bias["non_tie_count"] >= int(gate["canonical_non_tie_count_min_per_reviewer"]),
        "reviewer_a_position_bias_pass": a_bias["absolute_left_right_share_imbalance"] is not None and a_bias["absolute_left_right_share_imbalance"] <= float(gate["absolute_position_imbalance_max"]),
        "reviewer_b_position_bias_pass": b_bias["absolute_left_right_share_imbalance"] is not None and b_bias["absolute_left_right_share_imbalance"] <= float(gate["absolute_position_imbalance_max"]),
        "all_disagreements_adjudicated": set(c) == set(disputes),
    }
    gates["overall_pass"] = all(gates.values())
    agreement = {
        "name": "step3cr_cr0_v2_review_agreement_v1",
        "session_id": SESSION_ID,
        "review_rows": 48,
        "canonical_outcome_pairs": 40,
        "duplicate_qc_rows": 8,
        "pair_preference": preference,
        "specific_safe_next_step": specificity,
        "naturalness_sensitivity": naturalness,
        "reviewer_a_label_prevalence": label_prevalence(a),
        "reviewer_b_label_prevalence": label_prevalence(b),
        "decision_field_disagreement_rows": len(disputes),
        "disagreement_type_counts": dict(Counter(field for fields in disputes.values() for field in fields)),
        "reviewer_c_rows": len(c),
        "reviewer_c_coverage": len(c) / len(disputes) if disputes else 1.0,
        "gates": gates,
        "complete_revision_used": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(AGREEMENT, agreement)
    write_json(BIAS, {
        "name": "step3cr_cr0_v2_bias_and_reversal_v1",
        "reviewer_a_reversal": a_reverse,
        "reviewer_b_reversal": b_reverse,
        "reviewer_a_position_bias": a_bias,
        "reviewer_b_position_bias": b_bias,
        "naturalness_reversal_is_descriptive_only": True,
    })
    category_by_pair = {row["canonical_pair_id"]: row["category"] for row in key.values() if row["is_reversed_duplicate"] == "False"}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in canonical_rows:
        grouped.setdefault(category_by_pair[row["canonical_pair_id"]], []).append(row)
    write_json(CATEGORY, {
        "name": "step3cr_cr0_v2_category_summary_v1",
        "descriptive_only": True,
        "categories": {
            category: {
                "pairs": len(rows),
                "preference": dict(Counter(row["pair_preference"] for row in rows)),
                "both_specific": sum(row["canonical_left_specific_safe_next_step"] == "yes" and row["canonical_right_specific_safe_next_step"] == "yes" for row in rows),
            }
            for category, rows in sorted(grouped.items())
        },
    })
    route = {
        "name": "step3cr_cr0_v2_route_decision_v1",
        "session_id": SESSION_ID,
        "cr0_measurement_gate_pass": gates["overall_pass"],
        "complete_revision_used": True,
        "next_authorized_package": "P2_C-R1_no_update_diagnostics" if gates["overall_pass"] else "C-R0_closed_after_second_complete_failure",
        "future_response_quality_endpoint_frozen": gates["overall_pass"],
        "final_display_sha256": sha256(FINAL_DISPLAY),
        "final_canonical_sha256": sha256(FINAL_CANONICAL),
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(ROUTE, route)
    ROUTE_REPORT.write_text(
        "# Step3C-R C-R0 v2 Review and Route Report\n\n"
        f"Status: {'pass' if gates['overall_pass'] else 'second-session gate failure; C-R0 closed'}.\n\n"
        f"- Reviewer A/B rows: {ab_manifest['reviewer_a_rows']} / {ab_manifest['reviewer_b_rows']}\n"
        f"- Reviewer C disagreement rows: {len(c)}\n"
        f"- Preference raw agreement / AC1 / kappa: {preference['raw_agreement']:.3f} / {preference['gwet_ac1']:.3f} / {preference['cohen_kappa']}\n"
        f"- Specificity raw agreement / AC1 / kappa: {specificity['raw_agreement']:.3f} / {specificity['gwet_ac1']:.3f} / {specificity['cohen_kappa']}\n"
        f"- Reviewer A preference/specificity reversal: {a_reverse['preference_consistent_count']}/8 / {a_reverse['specificity_consistent_count']}/8\n"
        f"- Reviewer B preference/specificity reversal: {b_reverse['preference_consistent_count']}/8 / {b_reverse['specificity_consistent_count']}/8\n"
        f"- Reviewer A non-tie / position imbalance: {a_bias['non_tie_count']} / {a_bias['absolute_left_right_share_imbalance']}\n"
        f"- Reviewer B non-tie / position imbalance: {b_bias['non_tie_count']} / {b_bias['absolute_left_right_share_imbalance']}\n"
        f"- Overall gate: {'pass' if gates['overall_pass'] else 'fail'}\n"
        f"- Next authorized package: `{route['next_authorized_package']}`\n\n"
        "This was the single permitted complete rubric revision. V1 labels were not reused. Heldout remained closed; no optimizer or scheduler was created, and no training was run.\n",
        encoding="utf-8",
        newline="\n",
    )
    artifact_paths = [
        REGISTRY, RUBRIC, SCRIPT, ARCHIVE_MANIFEST, CLARIFICATION_REPORT, PREP_VALIDATION,
        PAIR_PACKET, PAIR_KEY, DUPLICATE_KEY, A_TEMPLATE, B_TEMPLATE, C_PACKET,
        PACKET_MANIFEST, C_PACKET_MANIFEST, PROVENANCE, AB_FREEZE, A_REVIEW, B_REVIEW,
        C_REVIEW, FINAL_DISPLAY, FINAL_CANONICAL, AGREEMENT, BIAS, CATEGORY, ROUTE, ROUTE_REPORT,
    ]
    artifacts = [
        {"path": path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in artifact_paths if path.exists()
    ]
    write_json(ARTIFACT_MANIFEST, {
        "name": "step3cr_cr0_v2_artifact_manifest_v1",
        "session_id": SESSION_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "all_canonical_lf": all(b"\r" not in path.read_bytes() for path in artifact_paths if path.exists()),
        "v1_labels_reused": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    })
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "validate-ab", "prepare-adjudication", "summarize"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "validate-ab":
        validate_ab()
    elif args.command == "prepare-adjudication":
        prepare_adjudication()
    else:
        summarize()


if __name__ == "__main__":
    main()
