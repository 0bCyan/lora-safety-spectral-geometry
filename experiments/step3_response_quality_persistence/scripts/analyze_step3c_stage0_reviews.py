from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
REGISTRY = ROOT / "configs/step3c_stage0_registry.json"
BASE = ROOT / "results/step3c_stage0"
PACKETS = BASE / "packets"
REVIEWS = BASE / "reviews"
REPORTS = BASE / "reports"
DIAGNOSTICS = BASE / "diagnostics"

BEHAVIOR_PACKET = PACKETS / "step3c_stage0_behavior_packet.jsonl"
STRUCTURE_PACKET = PACKETS / "step3c_stage0_structure_packet.jsonl"
ROLE_KEY = PACKETS / "step3c_stage0_response_role_key.csv"
PACKET_MANIFEST = PACKETS / "step3c_stage0_review_packet_manifest.json"

BEHAVIOR_A = REVIEWS / "step3c_behavior_reviewer_a.jsonl"
BEHAVIOR_B = REVIEWS / "step3c_behavior_reviewer_b.jsonl"
BEHAVIOR_C = REVIEWS / "step3c_behavior_reviewer_c.jsonl"
BEHAVIOR_C_PACKET = PACKETS / "step3c_behavior_reviewer_c_packet.jsonl"
BEHAVIOR_C_MANIFEST = PACKETS / "step3c_behavior_reviewer_c_manifest.json"
BEHAVIOR_FINAL = REVIEWS / "step3c_behavior_final_labels.jsonl"

STRUCTURE_A = REVIEWS / "step3c_structure_reviewer_a.jsonl"
STRUCTURE_B = REVIEWS / "step3c_structure_reviewer_b.jsonl"
STRUCTURE_C = REVIEWS / "step3c_structure_reviewer_c.jsonl"
STRUCTURE_C_PACKET = PACKETS / "step3c_structure_reviewer_c_packet.jsonl"
STRUCTURE_C_MANIFEST = PACKETS / "step3c_structure_reviewer_c_manifest.json"
STRUCTURE_FINAL = REVIEWS / "step3c_structure_final_labels.jsonl"

TRIPLET_SCHEMA = REPORTS / "step3c_stage0_triplet_schema.json"
TRIPLET_TABLE = REPORTS / "step3c_stage0_frozen_triplets.csv"
TRIPLET_DECISION = REPORTS / "step3c_stage0_triplet_data_gate.json"
ROUTE_REPORT = REPORTS / "Step3C_Stage0_route_report.md"
ROUTE_DECISION = REPORTS / "step3c_stage0_route_decision.json"
ARTIFACT_MANIFEST = REPORTS / "step3c_stage0_artifact_manifest.json"

BEHAVIOR_FIELDS = [
    "opaque_response_id",
    "primary_label",
    "clean_safe",
    "prompt_relevant",
    "grammatical_natural",
    "harmful_procedural_leakage",
    "invalid_or_irrelevant",
    "reason_code",
    "short_reason",
]
BEHAVIOR_DISPUTE_FIELDS = [
    "primary_label",
    "clean_safe",
    "prompt_relevant",
    "grammatical_natural",
    "harmful_procedural_leakage",
    "invalid_or_irrelevant",
]
BEHAVIOR_LABELS = {"valid_clean_safe", "unsafe_or_mixed", "irrelevant_or_invalid", "needs_adjudication_boundary"}
BOOL = {"yes", "no"}

STRUCTURE_FIELDS = [
    "opaque_response_id",
    "boundary_act",
    "rationale_act",
    "redirect_act",
    "harmful_procedural_tail",
    "mechanical_template",
    "short_reason",
]
STRUCTURE_DISPUTE_FIELDS = [
    "boundary_act",
    "rationale_act",
    "redirect_act",
    "harmful_procedural_tail",
    "mechanical_template",
]
BOUNDARY = {"no_boundary", "direct_refusal", "scoped_boundary", "conditional_boundary"}
RATIONALE = {"none", "harm_risk", "legal_compliance", "support_crisis", "epistemic_uncertainty"}
REDIRECT = {
    "none",
    "benign_alternative",
    "prevention_mitigation",
    "compliance_route",
    "support_resource",
    "high_level_education",
    "clarification",
}
TAIL = {"present", "absent"}
PACKET_SEED = 20260722
REVIEW_SESSION_ID = "step3c_stage0_review_v2_20260722"
RESPONSE_NORMALIZATION = "strip_leading_empty_think_block_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_review_response(text: str) -> str:
    return re.sub(r"^\s*<think>\s*</think>\s*", "", text, count=1, flags=re.IGNORECASE).strip()


def by_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        oid = str(row.get("opaque_response_id", ""))
        if not oid:
            raise ValueError(f"{path} row without opaque_response_id")
        if oid in out:
            raise ValueError(f"{path} duplicate opaque_response_id {oid}")
        out[oid] = row
    return out


def load_packet(path: Path) -> dict[str, dict[str, Any]]:
    return by_id(read_jsonl(path), path)


def load_key() -> dict[str, dict[str, str]]:
    return by_id(read_csv(ROLE_KEY), ROLE_KEY)  # type: ignore[return-value]


def target_text_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["target_id"]: row for row in read_jsonl(Path(registry["natural_v1_targets"]))}


def category_by_sample(registry: dict[str, Any]) -> dict[str, str]:
    coverage_path = Path(registry["positive_coverage_csv"])
    return {row["sample_id"]: row["category"] for row in read_csv(coverage_path)}


def load_tokenizer(registry: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        registry["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def response_token_length(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(len(token_ids))


def prepare_packets() -> None:
    registry = read_json(REGISTRY)
    PACKETS.mkdir(parents=True, exist_ok=True)
    target_rows = target_text_by_id(registry)
    categories = category_by_sample(registry)
    key_rows = read_csv(Path(registry["step3a_target_key"]))
    positives = read_jsonl(BASE / "raw_generations/step3c_stage0_prospective_positive_candidates.jsonl")
    negatives = read_jsonl(BASE / "raw_generations/constant20_safety_train_greedy.jsonl")
    source_rows: list[dict[str, Any]] = []
    index = 1

    for key in sorted(key_rows, key=lambda row: row["opaque_target_id"]):
        target = target_rows[key["target_id"]]
        raw_response = target["response"]
        review_response = normalize_review_response(raw_response)
        oid = f"step3c_resp_{index:04d}"
        source_rows.append({
            "opaque_response_id": oid,
            "sample_id": key["sample_id"],
            "category": key["category"],
            "role": "existing_natural_v1_target",
            "source_response_id": key["target_id"],
            "prompt": target["prompt"],
            "response": review_response,
            "source_sha256": sha256_text(review_response),
            "raw_response_sha256": sha256_text(raw_response),
            "review_response_sha256": sha256_text(review_response),
            "response_normalization": RESPONSE_NORMALIZATION,
        })
        index += 1
    for row in sorted(positives, key=lambda item: item["sample_id"]):
        raw_response = row["response"]
        review_response = normalize_review_response(raw_response)
        oid = f"step3c_resp_{index:04d}"
        source_rows.append({
            "opaque_response_id": oid,
            "sample_id": row["sample_id"],
            "category": categories[row["sample_id"]],
            "role": "prospective_positive_candidate",
            "source_response_id": row["candidate_id"],
            "prompt": row["prompt"],
            "response": review_response,
            "source_sha256": sha256_text(review_response),
            "raw_response_sha256": sha256_text(raw_response),
            "review_response_sha256": sha256_text(review_response),
            "response_normalization": RESPONSE_NORMALIZATION,
        })
        index += 1
    for row in sorted(negatives, key=lambda item: item["sample_id"]):
        raw_response = row["response"]
        review_response = normalize_review_response(raw_response)
        oid = f"step3c_resp_{index:04d}"
        source_rows.append({
            "opaque_response_id": oid,
            "sample_id": row["sample_id"],
            "category": categories[row["sample_id"]],
            "role": "constant20_generated_negative_candidate",
            "source_response_id": row["generation_id"],
            "prompt": row["prompt"],
            "response": review_response,
            "source_sha256": sha256_text(review_response),
            "raw_response_sha256": sha256_text(raw_response),
            "review_response_sha256": sha256_text(review_response),
            "response_normalization": RESPONSE_NORMALIZATION,
        })
        index += 1

    if len(source_rows) != 127:
        raise ValueError(f"expected 127 response candidates, found {len(source_rows)}")
    behavior_rows = [
        {"opaque_response_id": row["opaque_response_id"], "prompt": row["prompt"], "response": row["response"]}
        for row in source_rows
    ]
    structure_rows = [
        {"opaque_response_id": row["opaque_response_id"], "response": row["response"]}
        for row in source_rows
    ]
    rng = random.Random(PACKET_SEED)
    rng.shuffle(behavior_rows)
    rng = random.Random(PACKET_SEED + 1)
    rng.shuffle(structure_rows)
    write_jsonl(BEHAVIOR_PACKET, behavior_rows)
    write_jsonl(STRUCTURE_PACKET, structure_rows)
    write_csv(
        ROLE_KEY,
        source_rows,
        [
            "opaque_response_id",
            "sample_id",
            "category",
            "role",
            "source_response_id",
            "source_sha256",
            "raw_response_sha256",
            "review_response_sha256",
            "response_normalization",
        ],
    )
    manifest = {
        "review_session_id": REVIEW_SESSION_ID,
        "response_normalization": RESPONSE_NORMALIZATION,
        "behavior_packet_rows": len(behavior_rows),
        "structure_packet_rows": len(structure_rows),
        "key_rows": len(source_rows),
        "behavior_packet_sha256": sha256(BEHAVIOR_PACKET),
        "structure_packet_sha256": sha256(STRUCTURE_PACKET),
        "role_key_sha256": sha256(ROLE_KEY),
        "behavior_packet_contains_prompt": True,
        "structure_packet_contains_prompt": False,
        "reviewers_must_not_read_key": True,
        "role_counts_hidden_from_reviewers": dict(Counter(row["role"] for row in source_rows)),
        "heldout_opened": False,
        "training_authorized": False,
    }
    PACKET_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def validate_behavior(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    rows = by_id(read_jsonl(path), path)
    if set(rows) != expected:
        raise ValueError(f"{path} ID mismatch")
    for row in rows.values():
        if set(row) != set(BEHAVIOR_FIELDS):
            raise ValueError(f"{path} bad fields for {row['opaque_response_id']}")
        if row["primary_label"] not in BEHAVIOR_LABELS:
            raise ValueError(f"{path} bad primary_label")
        for field in ["clean_safe", "prompt_relevant", "grammatical_natural", "harmful_procedural_leakage", "invalid_or_irrelevant"]:
            if row[field] not in BOOL:
                raise ValueError(f"{path} bad {field}")
    return rows


def validate_structure(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    rows = by_id(read_jsonl(path), path)
    if set(rows) != expected:
        raise ValueError(f"{path} ID mismatch")
    for row in rows.values():
        if set(row) != set(STRUCTURE_FIELDS):
            raise ValueError(f"{path} bad fields for {row['opaque_response_id']}")
        if row["boundary_act"] not in BOUNDARY:
            raise ValueError(f"{path} bad boundary_act")
        if row["rationale_act"] not in RATIONALE:
            raise ValueError(f"{path} bad rationale_act")
        if row["redirect_act"] not in REDIRECT:
            raise ValueError(f"{path} bad redirect_act")
        if row["harmful_procedural_tail"] not in TAIL:
            raise ValueError(f"{path} bad harmful_procedural_tail")
        if row["mechanical_template"] not in BOOL:
            raise ValueError(f"{path} bad mechanical_template")
    return rows


def disagreements(a: dict[str, Any], b: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if str(a.get(field, "")) != str(b.get(field, ""))]


def prepare_adjudication(kind: str) -> None:
    if kind == "behavior":
        packet = load_packet(BEHAVIOR_PACKET)
        a = validate_behavior(BEHAVIOR_A, set(packet))
        b = validate_behavior(BEHAVIOR_B, set(packet))
        fields = BEHAVIOR_DISPUTE_FIELDS
        out_packet = BEHAVIOR_C_PACKET
        out_manifest = BEHAVIOR_C_MANIFEST
    else:
        packet = load_packet(STRUCTURE_PACKET)
        a = validate_structure(STRUCTURE_A, set(packet))
        b = validate_structure(STRUCTURE_B, set(packet))
        fields = STRUCTURE_DISPUTE_FIELDS
        out_packet = STRUCTURE_C_PACKET
        out_manifest = STRUCTURE_C_MANIFEST
    rows = []
    for oid in sorted(packet):
        dispute = disagreements(a[oid], b[oid], fields)
        if not dispute:
            continue
        row = {
            "opaque_response_id": oid,
            "response": packet[oid]["response"],
            "disagreement_fields": dispute,
            "reviewer_a": {field: a[oid][field] for field in fields + ["short_reason"]},
            "reviewer_b": {field: b[oid][field] for field in fields + ["short_reason"]},
        }
        if kind == "behavior":
            row["prompt"] = packet[oid]["prompt"]
        rows.append(row)
    write_jsonl(out_packet, rows)
    manifest = {
        "reviewer": f"{kind.title()} Reviewer C",
        "packet": str(out_packet),
        "packet_rows": len(rows),
        "packet_sha256": sha256(out_packet),
        "dispute_fields": fields,
        "reviewer_c_must_not_read_key": True,
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_c(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return by_id(read_jsonl(path), path)


def final_from_reviews(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    c: dict[str, dict[str, Any]],
    fields: list[str],
) -> dict[str, dict[str, Any]]:
    final = {oid: dict(row) for oid, row in a.items()}
    for oid, brow in b.items():
        dispute = disagreements(final[oid], brow, fields)
        if not dispute:
            continue
        if oid not in c:
            raise ValueError(f"missing adjudication for {oid}: {dispute}")
        for field in fields:
            if field in c[oid]:
                final[oid][field] = c[oid][field]
        if "short_reason" in c[oid]:
            final[oid]["adjudication_short_reason"] = c[oid]["short_reason"]
    return final


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    labels = set(left for left, _ in pairs) | set(right for _, right in pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs)) for label in labels)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def agreement(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    overlap = sorted(set(a) & set(b))
    out = {"overlap_rows": len(overlap), "fields": {}}
    for field in fields:
        pairs = [(str(a[oid][field]), str(b[oid][field])) for oid in overlap]
        raw = sum(left == right for left, right in pairs) / len(pairs) if pairs else None
        kappa = cohen_kappa(pairs)
        out["fields"][field] = {
            "raw_agreement": raw,
            "cohen_kappa": kappa,
            "cohen_kappa_note": "undefined due to zero label variance" if kappa is None and pairs else "",
        }
    return out


def reliability_field_pass(metrics: dict[str, Any], raw_min: float, kappa_min: float, zero_variance_raw_min: float) -> bool:
    raw = metrics["raw_agreement"]
    kappa = metrics["cohen_kappa"]
    if raw is None:
        return False
    if kappa is None:
        return raw >= zero_variance_raw_min
    return raw >= raw_min and kappa >= kappa_min


def freeze_triplets() -> None:
    registry = read_json(REGISTRY)
    schema = read_json(TRIPLET_SCHEMA)
    behavior_packet = load_packet(BEHAVIOR_PACKET)
    packet_ids = set(behavior_packet)
    behavior_a = validate_behavior(BEHAVIOR_A, packet_ids)
    behavior_b = validate_behavior(BEHAVIOR_B, packet_ids)
    behavior_c = load_c(BEHAVIOR_C)
    behavior_agreement = agreement(behavior_a, behavior_b, ["primary_label"] + BEHAVIOR_DISPUTE_FIELDS)
    behavior_final = final_from_reviews(behavior_a, behavior_b, behavior_c, BEHAVIOR_DISPUTE_FIELDS)

    structure_ids = set(load_packet(STRUCTURE_PACKET))
    structure_a = validate_structure(STRUCTURE_A, structure_ids)
    structure_b = validate_structure(STRUCTURE_B, structure_ids)
    structure_c = load_c(STRUCTURE_C)
    structure_agreement = agreement(structure_a, structure_b, STRUCTURE_DISPUTE_FIELDS)
    structure_final = final_from_reviews(structure_a, structure_b, structure_c, STRUCTURE_DISPUTE_FIELDS)

    gates = registry["stage0_review_gates"]
    behavior_primary_pass = reliability_field_pass(
        behavior_agreement["fields"]["primary_label"],
        float(gates["safety_primary_raw_agreement_min"]),
        float(gates["safety_primary_cohen_kappa_min"]),
        float(gates["zero_variance_raw_agreement_min"]),
    )
    structure_mechanical_pass = reliability_field_pass(
        structure_agreement["fields"]["mechanical_template"],
        float(gates["mechanical_template_raw_agreement_min"]),
        float(gates["mechanical_template_cohen_kappa_min"]),
        float(gates["zero_variance_raw_agreement_min"]),
    )
    agreements = {
        "review_session_id": REVIEW_SESSION_ID,
        "behavior": behavior_agreement,
        "structure": structure_agreement,
        "adjudication_rows": {"behavior": len(behavior_c), "structure": len(structure_c)},
        "review_gate": {
            "behavior_primary_pass": behavior_primary_pass,
            "structure_mechanical_pass": structure_mechanical_pass,
            "overall_pass": behavior_primary_pass and structure_mechanical_pass,
            "one_codebook_revision_already_used": REVIEW_SESSION_ID.endswith("_v2_20260722"),
        },
    }
    (REPORTS / "step3c_stage0_review_agreement.json").write_text(
        json.dumps(agreements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not agreements["review_gate"]["overall_pass"]:
        raise ValueError("review reliability gate failed; stop Stage 0 without freezing triplets")

    write_jsonl(BEHAVIOR_FINAL, [behavior_final[oid] for oid in sorted(behavior_final)])
    write_jsonl(STRUCTURE_FINAL, [structure_final[oid] for oid in sorted(structure_final)])

    role_key = load_key()
    tokenizer = load_tokenizer(registry)
    max_response_tokens = int(registry["generation"]["max_new_tokens"])
    by_sample_role: dict[tuple[str, str], list[str]] = defaultdict(list)
    for oid, row in role_key.items():
        by_sample_role[(row["sample_id"], row["role"])].append(oid)

    triplets = []
    invalid_reasons = []
    for sample_id in sorted({row["sample_id"] for row in role_key.values()}):
        positives = []
        for role in ["existing_natural_v1_target", "prospective_positive_candidate"]:
            for oid in sorted(by_sample_role.get((sample_id, role), [])):
                b = behavior_final[oid]
                s = structure_final[oid]
                valid_positive = (
                    b["clean_safe"] == "yes"
                    and b["prompt_relevant"] == "yes"
                    and b["grammatical_natural"] == "yes"
                    and b["harmful_procedural_leakage"] == "no"
                    and b["invalid_or_irrelevant"] == "no"
                    and s["mechanical_template"] == "no"
                )
                if valid_positive:
                    positives.append((role, oid))
        negatives = []
        for oid in sorted(by_sample_role.get((sample_id, "constant20_generated_negative_candidate"), [])):
            b = behavior_final[oid]
            s = structure_final[oid]
            valid_negative = (
                b["clean_safe"] == "yes"
                and b["prompt_relevant"] == "yes"
                and b["grammatical_natural"] == "yes"
                and b["harmful_procedural_leakage"] == "no"
                and b["invalid_or_irrelevant"] == "no"
                and s["mechanical_template"] == "yes"
            )
            if valid_negative:
                negatives.append(oid)
        if not positives or not negatives:
            invalid_reasons.append({
                "sample_id": sample_id,
                "has_positive": bool(positives),
                "has_negative": bool(negatives),
            })
            continue
        positive_role, pos_oid = positives[0]
        neg_oid = negatives[0]
        pos_key = role_key[pos_oid]
        neg_key = role_key[neg_oid]
        positive_token_length = response_token_length(tokenizer, behavior_packet[pos_oid]["response"])
        negative_token_length = response_token_length(tokenizer, behavior_packet[neg_oid]["response"])
        triplets.append({
            "triplet_id": f"T3C_{len(triplets)+1:04d}",
            "sample_id": sample_id,
            "category": pos_key["category"] or neg_key["category"],
            "positive_response_id": pos_oid,
            "negative_response_id": neg_oid,
            "positive_source": positive_role,
            "negative_source": "constant20_generated_negative_candidate",
            "positive_sha256": pos_key["source_sha256"],
            "negative_sha256": neg_key["source_sha256"],
            "positive_token_length": positive_token_length,
            "negative_token_length": negative_token_length,
            "policy_review_status": "pass",
            "relevance_review_status": "pass",
            "mechanical_review_status": "pass",
            "adjudication_status": "frozen",
        })
    write_csv(TRIPLET_TABLE, triplets, schema["fields"])
    gate = {
        "valid_triplets": len(triplets),
        "required_valid_triplets": 36,
        "token_length_limit": max_response_tokens,
        "token_lengths_finite_positive_and_within_limit": all(
            0 < int(row["positive_token_length"]) <= max_response_tokens
            and 0 < int(row["negative_token_length"]) <= max_response_tokens
            for row in triplets
        ),
        "c2_data_gate_pass": len(triplets) >= 36
        and all(
            0 < int(row["positive_token_length"]) <= max_response_tokens
            and 0 < int(row["negative_token_length"]) <= max_response_tokens
            for row in triplets
        ),
        "invalid_prompt_count": len(invalid_reasons),
        "invalid_prompt_reasons": invalid_reasons,
        "behavior_final_sha256": sha256(BEHAVIOR_FINAL),
        "structure_final_sha256": sha256(STRUCTURE_FINAL),
        "triplet_table_sha256": sha256(TRIPLET_TABLE),
        "heldout_opened": False,
        "training_authorized": False,
    }
    TRIPLET_DECISION.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def tracked_artifact_paths() -> list[Path]:
    roots = [
        ROOT / "review_rubrics",
        ROOT / "configs",
        ROOT / "scripts",
        BASE / "raw_generations",
        BASE / "packets",
        BASE / "reviews",
        BASE / "diagnostics",
        BASE / "reports",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path == ARTIFACT_MANIFEST:
                continue
            paths.append(path)
    return sorted(set(paths), key=lambda path: path.as_posix())


def write_route_report() -> None:
    registry = read_json(REGISTRY)
    packet_manifest = read_json(PACKET_MANIFEST)
    agreement = read_json(REPORTS / "step3c_stage0_review_agreement.json")
    v1_failure = read_json(REPORTS / "step3c_stage0_review_v1_reliability_failure_summary.json")
    review_gate = agreement["review_gate"]
    route = {
        "name": "step3c_stage0_route_decision_v1",
        "status": "stage0_stopped_at_review_reliability_gate",
        "review_session_id": agreement["review_session_id"],
        "v1_status": v1_failure["status"],
        "v2_behavior_primary_raw_agreement": agreement["behavior"]["fields"]["primary_label"]["raw_agreement"],
        "v2_behavior_primary_cohen_kappa": agreement["behavior"]["fields"]["primary_label"]["cohen_kappa"],
        "v2_structure_mechanical_raw_agreement": agreement["structure"]["fields"]["mechanical_template"]["raw_agreement"],
        "v2_structure_mechanical_cohen_kappa": agreement["structure"]["fields"]["mechanical_template"]["cohen_kappa"],
        "behavior_primary_gate_pass": review_gate["behavior_primary_pass"],
        "structure_mechanical_gate_pass": review_gate["structure_mechanical_pass"],
        "review_gate_pass": review_gate["overall_pass"],
        "codebook_revision_used": review_gate["one_codebook_revision_already_used"],
        "c2_data_gate_pass": False,
        "c2_diagnostic_run": False,
        "c2_stage1_authorized": False,
        "c3_diagnostic_run": False,
        "c3_stage1_authorized": False,
        "stage1_training_authorized": False,
        "optimizer_steps_run": False,
        "heldout_opened": False,
        "core100_used_for_triplet_construction": False,
        "route_decision": "stop_stage0_do_not_freeze_triplets_do_not_run_no_update_diagnostics_or_training",
        "reason": "After the single allowed codebook clarification and complete v2 blinded review, the mechanical_template reliability gate failed.",
    }
    ROUTE_DECISION.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifacts = []
    for path in tracked_artifact_paths():
        artifacts.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "name": "step3c_stage0_artifact_manifest_v1",
        "route_decision_sha256": sha256(ROUTE_DECISION),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "heldout_opened": False,
        "training_authorized": False,
    }
    ARTIFACT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# Step3C Stage 0 Route Report

Status: stopped at the review reliability gate.

## Inputs

- Review session: `{agreement['review_session_id']}`
- Behavior packet rows: {packet_manifest['behavior_packet_rows']}
- Structure packet rows: {packet_manifest['structure_packet_rows']}
- Response normalization: `{packet_manifest['response_normalization']}`
- Role counts hidden from reviewers: `{json.dumps(packet_manifest['role_counts_hidden_from_reviewers'], sort_keys=True)}`

## Reliability Gate

The first review session failed the behavior reliability gate and was not used for decision-making. A single V2 codebook clarification was applied and a complete new blinded review was run.

| field | raw agreement | Cohen's kappa | gate |
| --- | ---: | ---: | --- |
| behavior primary_label | {route['v2_behavior_primary_raw_agreement']:.3f} | {route['v2_behavior_primary_cohen_kappa']:.3f} | {'pass' if route['behavior_primary_gate_pass'] else 'fail'} |
| structure mechanical_template | {route['v2_structure_mechanical_raw_agreement']:.3f} | {route['v2_structure_mechanical_cohen_kappa']:.3f} | {'pass' if route['structure_mechanical_gate_pass'] else 'fail'} |

Overall review gate: **{'pass' if route['review_gate_pass'] else 'fail'}**.

## Route Decision

Decision: **stop Stage 0**.

Triplets were not frozen, C2/C3 no-update diagnostics were not run, and Stage 1 training is not authorized. The single permitted codebook clarification has already been used, and the repeated review session still failed the frozen `mechanical_template` reliability threshold.

## Boundary Checks

- Heldout opened: `false`
- Core100 used for triplet construction: `false`
- Optimizer steps run: `false`
- Training authorized: `false`
- New checkpoints written: `false`

## Artifacts

- Route decision: `step3c_stage0_route_decision.json`
- Artifact manifest: `step3c_stage0_artifact_manifest.json`
- Review agreement: `step3c_stage0_review_agreement.json`
- V1 failure summary: `step3c_stage0_review_v1_reliability_failure_summary.json`
"""
    ROUTE_REPORT.write_text(report, encoding="utf-8", newline="\n")
    # Refresh manifest after the route report itself exists.
    artifacts = []
    for path in tracked_artifact_paths():
        artifacts.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest["artifact_count"] = len(artifacts)
    manifest["artifacts"] = artifacts
    manifest["route_report_sha256"] = sha256(ROUTE_REPORT)
    ARTIFACT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(route, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare-packets",
            "prepare-behavior-adjudication",
            "prepare-structure-adjudication",
            "freeze-triplets",
            "write-route-report",
        ],
    )
    args = parser.parse_args()
    if args.command == "prepare-packets":
        prepare_packets()
    elif args.command == "prepare-behavior-adjudication":
        prepare_adjudication("behavior")
    elif args.command == "prepare-structure-adjudication":
        prepare_adjudication("structure")
    elif args.command == "freeze-triplets":
        freeze_triplets()
    elif args.command == "write-route-report":
        write_route_report()


if __name__ == "__main__":
    main()
