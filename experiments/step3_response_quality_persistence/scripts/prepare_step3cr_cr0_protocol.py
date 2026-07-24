from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
REGISTRY = ROOT / "configs/step3cr_recovery_registry.json"
STAGE0_REGISTRY = ROOT / "configs/step3c_stage0_registry.json"
STAGE0_BASE = ROOT / "results/step3c_stage0"
BASE = ROOT / "results/step3cr_cr0"
PACKETS = BASE / "packets"
REVIEWS = BASE / "reviews"
REPORTS = BASE / "reports"
DIAGNOSTICS = BASE / "diagnostics"

RUBRIC = ROOT / "review_rubrics/step3cr_prompt_aware_pairwise_rubric.md"
SCRIPT_PATH = ROOT / "scripts/prepare_step3cr_cr0_protocol.py"
PAIR_PACKET = PACKETS / "step3cr_cr0_pairwise_packet.jsonl"
PAIR_KEY = PACKETS / "step3cr_cr0_pairwise_hidden_key.csv"
DUPLICATE_KEY = PACKETS / "step3cr_cr0_reversed_duplicate_key.csv"
REVIEWER_A_TEMPLATE = PACKETS / "step3cr_cr0_reviewer_a_template.jsonl"
REVIEWER_B_TEMPLATE = PACKETS / "step3cr_cr0_reviewer_b_template.jsonl"
REVIEWER_C_PACKET = PACKETS / "step3cr_cr0_reviewer_c_packet.jsonl"
REVIEWER_A = REVIEWS / "step3cr_cr0_reviewer_a.jsonl"
REVIEWER_B = REVIEWS / "step3cr_cr0_reviewer_b.jsonl"
REVIEWER_C = REVIEWS / "step3cr_cr0_reviewer_c.jsonl"
FINAL_LABELS = REVIEWS / "step3cr_cr0_final_pair_labels.jsonl"
PACKET_MANIFEST = PACKETS / "step3cr_cr0_packet_manifest.json"
PROTOCOL_REPORT = REPORTS / "Step3CR_CR0_protocol_freeze_report.md"
PROTOCOL_MANIFEST = REPORTS / "step3cr_cr0_protocol_manifest.json"
AGREEMENT_REPORT = REPORTS / "step3cr_cr0_review_agreement.json"
ROUTE_DECISION = REPORTS / "step3cr_cr0_route_decision.json"
INPUT_VALIDATION = DIAGNOSTICS / "step3cr_cr0_input_validation.json"
SYNTHETIC_TESTS = DIAGNOSTICS / "step3cr_cr0_synthetic_tests.json"
FRESH_VERIFY = DIAGNOSTICS / "step3cr_cr0_fresh_checkout_verification.json"
P0_EQUALITY = DIAGNOSTICS / "step3cr_p0_packet_byte_equality.json"
CANONICAL_LF_REPORT = DIAGNOSTICS / "step3cr_p0_canonical_lf_verification.json"
P0_AMENDMENT_REPORT = REPORTS / "Step3CR_P0_protocol_amendment_report.md"
BASELINE_PROTOCOL_COMMIT = "b16f5a1"

PAIR_FIELDS = [
    "opaque_pair_id",
    "pair_preference",
    "left_specific_safe_next_step",
    "right_specific_safe_next_step",
    "left_natural",
    "right_natural",
    "short_reason",
]
PREF = {"left", "right", "tie"}
BOOL = {"yes", "no"}
DECISION_FIELDS = [
    "pair_preference",
    "left_specific_safe_next_step",
    "right_specific_safe_next_step",
    "left_natural",
    "right_natural",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def reconstructed_crlf_bytes(data: bytes) -> bytes:
    return canonical_lf_bytes(data).replace(b"\n", b"\r\n")


def classify_bytes_against_hash(data: bytes, expected_sha256: str) -> str:
    if sha256_bytes(data) == expected_sha256:
        return "raw_match"
    if sha256_bytes(canonical_lf_bytes(data)) == expected_sha256:
        return "eol_only_crlf_to_lf"
    if sha256_bytes(reconstructed_crlf_bytes(data)) == expected_sha256:
        return "eol_only_lf_to_crlf"
    return "non_eol_mismatch"


def git_blob_bytes(commit: str, path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path.as_posix()}"])


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def by_field(rows: list[dict[str, Any]], field: str, path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        key = str(row.get(field, ""))
        if not key:
            raise ValueError(f"{path} row missing {field}")
        if key in out:
            raise ValueError(f"{path} duplicate {field}: {key}")
        out[key] = row
    return out


def valid_clean_safe(row: dict[str, Any]) -> bool:
    return (
        row["primary_label"] == "valid_clean_safe"
        and row["clean_safe"] == "yes"
        and row["prompt_relevant"] == "yes"
        and row["grammatical_natural"] == "yes"
        and row["harmful_procedural_leakage"] == "no"
        and row["invalid_or_irrelevant"] == "no"
    )


def load_stage0_behavior_labels() -> dict[str, dict[str, Any]]:
    a = by_field(read_jsonl(STAGE0_BASE / "reviews/step3c_behavior_reviewer_a.jsonl"), "opaque_response_id", STAGE0_BASE / "reviews/step3c_behavior_reviewer_a.jsonl")
    b = by_field(read_jsonl(STAGE0_BASE / "reviews/step3c_behavior_reviewer_b.jsonl"), "opaque_response_id", STAGE0_BASE / "reviews/step3c_behavior_reviewer_b.jsonl")
    if set(a) != set(b):
        raise ValueError("Stage0 behavior A/B ID mismatch")
    for oid in a:
        for field in ["primary_label", "clean_safe", "prompt_relevant", "grammatical_natural", "harmful_procedural_leakage", "invalid_or_irrelevant"]:
            if a[oid][field] != b[oid][field]:
                raise ValueError(f"Stage0 V2 behavior labels are not frozen-agreeing for {oid} field {field}")
    return a


def load_stage0_sources() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    packet = by_field(read_jsonl(STAGE0_BASE / "packets/step3c_stage0_behavior_packet.jsonl"), "opaque_response_id", STAGE0_BASE / "packets/step3c_stage0_behavior_packet.jsonl")
    key_rows = read_csv(STAGE0_BASE / "packets/step3c_stage0_response_role_key.csv")
    key = by_field(key_rows, "opaque_response_id", STAGE0_BASE / "packets/step3c_stage0_response_role_key.csv")
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in key_rows:
        by_sample[row["sample_id"]].append(row)
    return packet, key, by_sample


def split_overlaps(stage0_registry: dict[str, Any], sample_ids: set[str]) -> dict[str, int]:
    eval_rows = read_jsonl(Path(stage0_registry["eval_safety_path"]))
    core100 = {row["sample_id"] for row in eval_rows[:100]}
    split_ids: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(Path(stage0_registry["split_manifest"])):
        split_ids[row["split"]].add(row["sample_id"])
    overlaps = {split: len(sample_ids & ids) for split, ids in split_ids.items()}
    overlaps["core100"] = len(sample_ids & core100)
    return overlaps


def validate_stage0_artifact_manifest() -> dict[str, Any]:
    manifest_path = STAGE0_BASE / "reports/step3c_stage0_artifact_manifest.json"
    manifest = read_json(manifest_path)
    rows = []
    status_counts: Counter[str] = Counter()
    non_eol_mismatches = []
    eol_only_mismatches = []
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        exists = path.exists()
        expected = artifact["sha256"]
        status = "missing" if not exists else classify_bytes_against_hash(path.read_bytes(), expected)
        status_counts[status] += 1
        if status.startswith("eol_only_"):
            eol_only_mismatches.append({"path": str(path), "classification": status})
        elif status in {"non_eol_mismatch", "missing"}:
            non_eol_mismatches.append({"path": str(path), "classification": status})
        rows.append({"path": str(path), "exists": exists, "status": status})
    return {
        "stage0_manifest": str(manifest_path),
        "stage0_manifest_entries": len(manifest["artifacts"]),
        "expected_stage0_artifacts": 47,
        "all_stage0_artifacts_exist": all(row["exists"] for row in rows),
        "raw_match_count": status_counts["raw_match"],
        "crlf_to_lf_match_count": status_counts["eol_only_crlf_to_lf"],
        "lf_to_crlf_match_count": status_counts["eol_only_lf_to_crlf"],
        "byte_hash_mismatch_count": len(non_eol_mismatches),
        "eol_only_mismatch_count": len(eol_only_mismatches),
        "non_eol_hash_mismatches": non_eol_mismatches,
        "eol_only_mismatches": eol_only_mismatches,
        "symmetric_eol_classification": True,
    }


def validate_preconditions() -> dict[str, Any]:
    registry = read_json(REGISTRY)
    stage0_registry = read_json(STAGE0_REGISTRY)
    stage0_decision = read_json(Path(registry["stopped_stage0_decision"]))
    packet, key, by_sample = load_stage0_sources()
    labels = load_stage0_behavior_labels()
    sample_ids = set(by_sample)
    overlaps = split_overlaps(stage0_registry, sample_ids)
    stage0_artifacts = validate_stage0_artifact_manifest()
    checkpoint = Path(stage0_registry["checkpoint_dir"])
    cuda: dict[str, Any] = {"torch_available": False, "cuda_available": False}
    try:
        import torch

        cuda["torch_available"] = True
        cuda["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            cuda.update({
                "device_name": torch.cuda.get_device_name(0),
                "free_vram_bytes": int(free_bytes),
                "total_vram_bytes": int(total_bytes),
                "free_headroom_fraction": float(free_bytes / total_bytes),
                "minimum_headroom_fraction": float(registry["cr1_representation"]["minimum_vram_headroom_fraction"]),
                "headroom_pass": free_bytes / total_bytes >= float(registry["cr1_representation"]["minimum_vram_headroom_fraction"]),
            })
    except Exception as exc:
        cuda["error"] = str(exc)
    role_counts = Counter(row["role"] for row in key.values())
    valid_counts = Counter()
    for oid, row in key.items():
        if valid_clean_safe(labels[oid]):
            valid_counts[row["role"]] += 1
    validation = {
        "git_head": git_head(),
        "registry": str(REGISTRY),
        "registry_sha256": sha256(REGISTRY),
        "source_plan": registry["source_plan"],
        "stage0_decision_status": stage0_decision["status"],
        "stage0_route_decision": stage0_decision["route_decision"],
        "stage0_artifacts": stage0_artifacts,
        "stage0_packet_rows": len(packet),
        "stage0_role_key_rows": len(key),
        "stage0_behavior_label_rows": len(labels),
        "safety_train_prompt_count": len(sample_ids),
        "role_counts": dict(role_counts),
        "valid_clean_safe_by_role": dict(valid_counts),
        "split_overlaps": overlaps,
        "core100_used_for_training": False,
        "heldout_opened": False,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_adapter_exists": (checkpoint / "adapter_model.safetensors").exists(),
        "cuda": cuda,
        "training_authorized": False,
    }
    if len(sample_ids) != int(registry["cr0_measurement"]["canonical_pairs"]):
        raise ValueError("C-R0 requires exactly 40 safety_train prompts")
    if overlaps.get("core100") or overlaps.get("proxy_dev") or overlaps.get("heldout_test"):
        raise ValueError(f"unexpected split overlap: {overlaps}")
    if not stage0_artifacts["all_stage0_artifacts_exist"] or stage0_artifacts["byte_hash_mismatch_count"]:
        raise ValueError("Stage0 artifact verification failed")
    write_json(INPUT_VALIDATION, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return validation


def response_source(row: dict[str, str], packet: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "opaque_response_id": row["opaque_response_id"],
        "source_response_id": row["source_response_id"],
        "role": row["role"],
        "source_sha256": row["source_sha256"],
        "prompt": packet[row["opaque_response_id"]]["prompt"],
        "response": packet[row["opaque_response_id"]]["response"],
    }


def construct_canonical_pairs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = read_json(REGISTRY)
    packet, _, by_sample = load_stage0_sources()
    labels = load_stage0_behavior_labels()
    canonical = []
    invalid = []
    for index, sample_id in enumerate(sorted(by_sample), start=1):
        rows = by_sample[sample_id]
        existing = sorted(
            [row for row in rows if row["role"] == "existing_natural_v1_target"],
            key=lambda row: row["source_response_id"],
        )
        constant = [row for row in rows if row["role"] == "constant20_generated_negative_candidate"]
        if len(existing) != 2 or len(constant) != 1:
            raise ValueError(f"bad role counts for {sample_id}")
        if valid_clean_safe(labels[constant[0]["opaque_response_id"]]):
            selected = [existing[0], constant[0]]
            construction_rule = "constant20_valid_vs_lexicographically_smallest_existing_target"
        else:
            selected = existing
            construction_rule = "constant20_invalid_use_two_existing_targets"
        if not all(valid_clean_safe(labels[row["opaque_response_id"]]) for row in selected):
            invalid.append({
                "sample_id": sample_id,
                "selected_response_ids": [row["opaque_response_id"] for row in selected],
                "construction_rule": construction_rule,
            })
            continue
        left_source, right_source = selected
        canonical.append({
            "canonical_pair_id": f"step3cr_cp_{index:04d}",
            "sample_id": sample_id,
            "category": selected[0]["category"] or selected[1]["category"],
            "canonical_left": response_source(left_source, packet),
            "canonical_right": response_source(right_source, packet),
            "construction_rule": construction_rule,
        })
    if invalid:
        raise ValueError(f"C-R0 pair construction failed: {invalid}")
    expected = int(registry["cr0_measurement"]["canonical_pairs"])
    if len(canonical) != expected:
        raise ValueError(f"expected {expected} canonical pairs, found {len(canonical)}")
    return canonical, invalid


def packet_row(pair: dict[str, Any], opaque_pair_id: str, left: dict[str, Any], right: dict[str, Any], is_duplicate: bool) -> dict[str, Any]:
    return {
        "opaque_pair_id": opaque_pair_id,
        "prompt": left["prompt"],
        "left_response": left["response"],
        "right_response": right["response"],
    }


def build_packets() -> dict[str, Any]:
    registry = read_json(REGISTRY)
    validation = validate_preconditions()
    canonical, invalid = construct_canonical_pairs()
    seed = int(registry["cr0_measurement"]["randomization_seed"])
    rng = random.Random(seed)
    duplicate_ids = set(rng.sample([row["canonical_pair_id"] for row in canonical], int(registry["cr0_measurement"]["reversed_duplicates"])))
    packet_rows = []
    key_rows = []
    duplicate_rows = []
    template_rows = []
    serial = 1
    for pair in canonical:
        flip = rng.choice([False, True])
        left = pair["canonical_right"] if flip else pair["canonical_left"]
        right = pair["canonical_left"] if flip else pair["canonical_right"]
        opaque_id = f"step3cr_pair_{serial:04d}"
        packet_rows.append(packet_row(pair, opaque_id, left, right, False))
        key_rows.append({
            "opaque_pair_id": opaque_id,
            "canonical_pair_id": pair["canonical_pair_id"],
            "sample_id": pair["sample_id"],
            "category": pair["category"],
            "is_reversed_duplicate": False,
            "duplicate_of_opaque_pair_id": "",
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
        })
        template_rows.append({field: opaque_id if field == "opaque_pair_id" else "" for field in PAIR_FIELDS})
        serial += 1
        if pair["canonical_pair_id"] in duplicate_ids:
            duplicate_id = f"step3cr_pair_{serial:04d}"
            packet_rows.append(packet_row(pair, duplicate_id, right, left, True))
            key_rows.append({
                "opaque_pair_id": duplicate_id,
                "canonical_pair_id": pair["canonical_pair_id"],
                "sample_id": pair["sample_id"],
                "category": pair["category"],
                "is_reversed_duplicate": True,
                "duplicate_of_opaque_pair_id": opaque_id,
                "construction_rule": pair["construction_rule"],
                "canonical_left_opaque_response_id": pair["canonical_left"]["opaque_response_id"],
                "canonical_right_opaque_response_id": pair["canonical_right"]["opaque_response_id"],
                "display_left_opaque_response_id": right["opaque_response_id"],
                "display_right_opaque_response_id": left["opaque_response_id"],
                "display_left_source_response_id": right["source_response_id"],
                "display_right_source_response_id": left["source_response_id"],
                "display_left_role": right["role"],
                "display_right_role": left["role"],
                "display_left_sha256": sha256_text(right["response"]),
                "display_right_sha256": sha256_text(left["response"]),
            })
            duplicate_rows.append({
                "canonical_pair_id": pair["canonical_pair_id"],
                "primary_opaque_pair_id": opaque_id,
                "reversed_duplicate_opaque_pair_id": duplicate_id,
            })
            template_rows.append({field: duplicate_id if field == "opaque_pair_id" else "" for field in PAIR_FIELDS})
            serial += 1
    rng.shuffle(packet_rows)
    template_by_id = {row["opaque_pair_id"]: row for row in template_rows}
    template_rows = [template_by_id[row["opaque_pair_id"]] for row in packet_rows]
    write_jsonl(PAIR_PACKET, packet_rows)
    write_csv(
        PAIR_KEY,
        key_rows,
        [
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
        ],
    )
    write_csv(DUPLICATE_KEY, duplicate_rows, ["canonical_pair_id", "primary_opaque_pair_id", "reversed_duplicate_opaque_pair_id"])
    write_jsonl(REVIEWER_A_TEMPLATE, template_rows)
    write_jsonl(REVIEWER_B_TEMPLATE, template_rows)
    write_jsonl(REVIEWER_C_PACKET, [])
    manifest = {
        "name": "step3cr_cr0_packet_manifest_v1",
        "git_head": git_head(),
        "registry": str(REGISTRY),
        "registry_sha256": sha256(REGISTRY),
        "rubric": str(RUBRIC),
        "rubric_sha256": sha256(RUBRIC),
        "packet_rows": len(packet_rows),
        "canonical_pairs": len(canonical),
        "reversed_duplicates": len(duplicate_rows),
        "randomization_seed": seed,
        "duplicate_selection_sha256": sha256(DUPLICATE_KEY),
        "packet_sha256": sha256(PAIR_PACKET),
        "hidden_key_sha256": sha256(PAIR_KEY),
        "reviewer_a_template_sha256": sha256(REVIEWER_A_TEMPLATE),
        "reviewer_b_template_sha256": sha256(REVIEWER_B_TEMPLATE),
        "reviewer_c_packet_schema_ready": True,
        "reviewer_c_packet_rows_before_labels": 0,
        "stage0_input_validation_sha256": sha256(INPUT_VALIDATION),
        "construction_rule_counts": dict(Counter(row["construction_rule"] for row in key_rows if row["is_reversed_duplicate"] is False)),
        "packet_fields": sorted(packet_rows[0]),
        "packet_role_hidden": True,
        "packet_duplicate_links_hidden": True,
        "main_executor_assigned_labels": False,
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(PACKET_MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def canonicalize_preference(preference: str, display_left_is_canonical_left: bool) -> str:
    if preference == "tie":
        return "tie"
    if preference not in {"left", "right"}:
        raise ValueError(f"bad preference: {preference}")
    if display_left_is_canonical_left:
        return "canonical_left" if preference == "left" else "canonical_right"
    return "canonical_right" if preference == "left" else "canonical_left"


def canonicalize_specific(row: dict[str, Any], display_left_is_canonical_left: bool) -> tuple[str, str]:
    left = row["left_specific_safe_next_step"]
    right = row["right_specific_safe_next_step"]
    return (left, right) if display_left_is_canonical_left else (right, left)


def canonicalize_naturalness(row: dict[str, Any], display_left_is_canonical_left: bool) -> tuple[str, str]:
    left = row["left_natural"]
    right = row["right_natural"]
    return (left, right) if display_left_is_canonical_left else (right, left)


def reversal_gate_pass(consistent_count: int, denominator: int, minimum_count: int) -> bool:
    return denominator == 8 and consistent_count >= minimum_count


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


def gwet_ac1(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    labels = sorted(set(left for left, _ in pairs) | set(right for _, right in pairs))
    if len(labels) <= 1:
        return 1.0 if observed == 1.0 else None
    counts = Counter()
    for left, right in pairs:
        counts[left] += 1
        counts[right] += 1
    probs = {label: counts[label] / (2 * len(pairs)) for label in labels}
    pe = sum(prob * (1 - prob) for prob in probs.values()) / (len(labels) - 1)
    if pe >= 1.0:
        return None
    return (observed - pe) / (1.0 - pe)


def run_synthetic_tests() -> dict[str, Any]:
    pref_forward = canonicalize_preference("left", True)
    pref_reversed = canonicalize_preference("right", False)
    specific_forward = canonicalize_specific(
        {"left_specific_safe_next_step": "yes", "right_specific_safe_next_step": "no"},
        True,
    )
    specific_reversed = canonicalize_specific(
        {"left_specific_safe_next_step": "no", "right_specific_safe_next_step": "yes"},
        False,
    )
    if pref_forward != pref_reversed or pref_forward != "canonical_left":
        raise ValueError("preference canonicalization synthetic test failed")
    if specific_forward != specific_reversed or specific_forward != ("yes", "no"):
        raise ValueError("specificity canonicalization synthetic test failed")
    one_sided_specificity_failure_detected = specific_forward != canonicalize_specific(
        {"left_specific_safe_next_step": "yes", "right_specific_safe_next_step": "yes"},
        False,
    )
    if not one_sided_specificity_failure_detected:
        raise ValueError("one-sided specificity failure fixture was not detected")
    reversal_gate_failure_detected = not reversal_gate_pass(6, 8, 7)
    reversal_gate_boundary_passed = reversal_gate_pass(7, 8, 7)
    if not reversal_gate_failure_detected or not reversal_gate_boundary_passed:
        raise ValueError("reversal gate synthetic test failed")
    rows = [
        {"opaque_pair_id": "p1", "prompt": "x", "left_response": "a", "right_response": "b"},
        {"opaque_pair_id": "p2", "prompt": "x", "left_response": "b", "right_response": "a"},
    ]
    forbidden = {"canonical_pair_id", "sample_id", "category", "role", "source_response_id", "duplicate_of_opaque_pair_id"}
    if any(set(row) & forbidden for row in rows):
        raise ValueError("hidden-key isolation synthetic test failed")
    pairs = [("left", "left"), ("right", "right"), ("tie", "tie"), ("left", "right")]
    lf_fixture = b"alpha\nbeta\n"
    crlf_fixture = b"alpha\r\nbeta\r\n"
    non_eol_fixture = b"alpha\ngamma\n"
    lf_expected = sha256_bytes(lf_fixture)
    crlf_expected = sha256_bytes(crlf_fixture)
    crlf_to_lf_passed = classify_bytes_against_hash(crlf_fixture, lf_expected) == "eol_only_crlf_to_lf"
    lf_to_crlf_passed = classify_bytes_against_hash(lf_fixture, crlf_expected) == "eol_only_lf_to_crlf"
    non_eol_rejected = classify_bytes_against_hash(non_eol_fixture, lf_expected) == "non_eol_mismatch"
    if not crlf_to_lf_passed or not lf_to_crlf_passed or not non_eol_rejected:
        raise ValueError("symmetric EOL classification synthetic test failed")
    zero_variance_pairs = [("yes", "yes")] * 4
    zero_variance_kappa = cohen_kappa(zero_variance_pairs)
    zero_variance_ac1 = gwet_ac1(zero_variance_pairs)
    if zero_variance_kappa is not None or zero_variance_ac1 != 1.0:
        raise ValueError("zero-variance metric synthetic test failed")
    tests = {
        "preference_canonicalization_passed": True,
        "specificity_canonicalization_passed": True,
        "one_sided_specificity_failure_detected": one_sided_specificity_failure_detected,
        "reversal_gate_failure_detected": reversal_gate_failure_detected,
        "reversal_gate_boundary_passed": reversal_gate_boundary_passed,
        "crlf_to_lf_classification_passed": crlf_to_lf_passed,
        "lf_to_crlf_classification_passed": lf_to_crlf_passed,
        "non_eol_mutation_rejected": non_eol_rejected,
        "hidden_key_isolation_passed": True,
        "raw_agreement_fixture": sum(a == b for a, b in pairs) / len(pairs),
        "cohen_kappa_fixture": cohen_kappa(pairs),
        "gwet_ac1_fixture": gwet_ac1(pairs),
        "zero_variance_kappa": zero_variance_kappa,
        "zero_variance_gwet_ac1": zero_variance_ac1,
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(SYNTHETIC_TESTS, tests)
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    return tests


def verify_packet_hidden_fields() -> dict[str, Any]:
    rows = read_jsonl(PAIR_PACKET)
    forbidden = {"canonical_pair_id", "sample_id", "category", "role", "source_response_id", "duplicate_of_opaque_pair_id"}
    leaks = sorted({field for row in rows for field in row if field in forbidden})
    ids = [row["opaque_pair_id"] for row in rows]
    result = {
        "packet_rows": len(rows),
        "unique_opaque_pair_ids": len(set(ids)),
        "packet_fields": sorted(set().union(*(row.keys() for row in rows))),
        "forbidden_field_leaks": leaks,
        "hidden_key_exists": PAIR_KEY.exists(),
        "duplicate_key_exists": DUPLICATE_KEY.exists(),
        "passed": len(rows) == 48 and len(set(ids)) == 48 and not leaks and PAIR_KEY.exists() and DUPLICATE_KEY.exists(),
    }
    if not result["passed"]:
        raise ValueError(f"packet hidden-field verification failed: {result}")
    return result


def validate_review(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = by_field(read_jsonl(path), "opaque_pair_id", path)
    if set(rows) != expected_ids:
        raise ValueError(f"{path} opaque_pair_id mismatch")
    for row in rows.values():
        if set(row) != set(PAIR_FIELDS):
            raise ValueError(f"{path} bad fields for {row['opaque_pair_id']}")
        if row["pair_preference"] not in PREF:
            raise ValueError(f"{path} bad pair_preference")
        for field in ["left_specific_safe_next_step", "right_specific_safe_next_step", "left_natural", "right_natural"]:
            if row[field] not in BOOL:
                raise ValueError(f"{path} bad {field}")
    return rows


def load_pair_key() -> dict[str, dict[str, str]]:
    return by_field(read_csv(PAIR_KEY), "opaque_pair_id", PAIR_KEY)  # type: ignore[return-value]


def display_left_is_canonical_left(key_row: dict[str, str]) -> bool:
    return key_row["display_left_opaque_response_id"] == key_row["canonical_left_opaque_response_id"]


def prepare_adjudication() -> dict[str, Any]:
    packet = by_field(read_jsonl(PAIR_PACKET), "opaque_pair_id", PAIR_PACKET)
    expected = set(packet)
    a = validate_review(REVIEWER_A, expected)
    b = validate_review(REVIEWER_B, expected)
    rows = []
    for opaque_id in sorted(expected):
        disputes = [field for field in DECISION_FIELDS if a[opaque_id][field] != b[opaque_id][field]]
        if not disputes:
            continue
        rows.append({
            "opaque_pair_id": opaque_id,
            "prompt": packet[opaque_id]["prompt"],
            "left_response": packet[opaque_id]["left_response"],
            "right_response": packet[opaque_id]["right_response"],
            "disagreement_fields": disputes,
            "reviewer_a": {field: a[opaque_id][field] for field in DECISION_FIELDS + ["short_reason"]},
            "reviewer_b": {field: b[opaque_id][field] for field in DECISION_FIELDS + ["short_reason"]},
        })
    write_jsonl(REVIEWER_C_PACKET, rows)
    manifest = {
        "name": "step3cr_cr0_reviewer_c_packet_manifest_v1",
        "packet_rows": len(rows),
        "packet_sha256": sha256(REVIEWER_C_PACKET),
        "dispute_fields": DECISION_FIELDS,
        "reviewer_c_must_not_read_hidden_key": True,
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(PACKETS / "step3cr_cr0_reviewer_c_packet_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def load_c_labels() -> dict[str, dict[str, Any]]:
    if not REVIEWER_C.exists() or REVIEWER_C.stat().st_size == 0:
        return {}
    rows = by_field(read_jsonl(REVIEWER_C), "opaque_pair_id", REVIEWER_C)
    for row in rows.values():
        allowed = {"opaque_pair_id", *DECISION_FIELDS, "short_reason"}
        if set(row) != allowed:
            raise ValueError(f"{REVIEWER_C} bad fields for {row['opaque_pair_id']}")
        if row["pair_preference"] not in PREF:
            raise ValueError(f"{REVIEWER_C} bad pair_preference")
        for field in ["left_specific_safe_next_step", "right_specific_safe_next_step", "left_natural", "right_natural"]:
            if row[field] not in BOOL:
                raise ValueError(f"{REVIEWER_C} bad {field}")
    return rows


def final_from_reviews(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], c: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    final = {opaque_id: dict(row) for opaque_id, row in a.items()}
    for opaque_id, brow in b.items():
        disputes = [field for field in DECISION_FIELDS if final[opaque_id][field] != brow[field]]
        if not disputes:
            continue
        if opaque_id not in c:
            raise ValueError(f"missing Reviewer C adjudication for {opaque_id}: {disputes}")
        for field in DECISION_FIELDS:
            final[opaque_id][field] = c[opaque_id][field]
        final[opaque_id]["adjudication_short_reason"] = c[opaque_id]["short_reason"]
    return final


def raw_agreement(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(left == right for left, right in pairs) / len(pairs)


def metric_block(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "raw_agreement": raw_agreement(pairs),
        "gwet_ac1": gwet_ac1(pairs),
        "cohen_kappa": cohen_kappa(pairs),
    }


def canonical_primary_pairs(reviews: dict[str, dict[str, Any]], key: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    out = {}
    for opaque_id, key_row in key.items():
        if key_row["is_reversed_duplicate"] == "True":
            continue
        left_is_canonical = display_left_is_canonical_left(key_row)
        left_specific, right_specific = canonicalize_specific(reviews[opaque_id], left_is_canonical)
        out[key_row["canonical_pair_id"]] = {
            "opaque_pair_id": opaque_id,
            "preference": canonicalize_preference(reviews[opaque_id]["pair_preference"], left_is_canonical),
            "canonical_left_specific": left_specific,
            "canonical_right_specific": right_specific,
            "display_preference": reviews[opaque_id]["pair_preference"],
        }
    return out


def reversed_consistency(reviews: dict[str, dict[str, Any]], key: dict[str, dict[str, str]]) -> dict[str, Any]:
    primary_by_id = {
        opaque_id: key_row
        for opaque_id, key_row in key.items()
        if key_row["is_reversed_duplicate"] == "False"
    }
    duplicate_rows = [row for row in key.values() if row["is_reversed_duplicate"] == "True"]
    preference_consistent = 0
    specificity_consistent = 0
    naturalness_consistent = 0
    details = []
    for duplicate in duplicate_rows:
        primary_id = duplicate["duplicate_of_opaque_pair_id"]
        primary = primary_by_id[primary_id]
        p_left = display_left_is_canonical_left(primary)
        d_left = display_left_is_canonical_left(duplicate)
        p_pref = canonicalize_preference(reviews[primary_id]["pair_preference"], p_left)
        d_pref = canonicalize_preference(reviews[duplicate["opaque_pair_id"]]["pair_preference"], d_left)
        p_spec = canonicalize_specific(reviews[primary_id], p_left)
        d_spec = canonicalize_specific(reviews[duplicate["opaque_pair_id"]], d_left)
        p_natural = canonicalize_naturalness(reviews[primary_id], p_left)
        d_natural = canonicalize_naturalness(reviews[duplicate["opaque_pair_id"]], d_left)
        pref_match = p_pref == d_pref
        spec_match = p_spec == d_spec
        natural_match = p_natural == d_natural
        preference_consistent += int(pref_match)
        specificity_consistent += int(spec_match)
        naturalness_consistent += int(natural_match)
        details.append({
            "canonical_pair_id": duplicate["canonical_pair_id"],
            "primary_opaque_pair_id": primary_id,
            "duplicate_opaque_pair_id": duplicate["opaque_pair_id"],
            "preference_consistent": pref_match,
            "specificity_consistent": spec_match,
            "naturalness_consistent": natural_match,
        })
    return {
        "denominator": len(duplicate_rows),
        "preference_consistent_count": preference_consistent,
        "specificity_consistent_count": specificity_consistent,
        "naturalness_consistent_count": naturalness_consistent,
        "details": details,
    }


def position_bias(reviews: dict[str, dict[str, Any]], key: dict[str, dict[str, str]]) -> dict[str, Any]:
    primary_ids = [opaque_id for opaque_id, row in key.items() if row["is_reversed_duplicate"] == "False"]
    prefs = [reviews[opaque_id]["pair_preference"] for opaque_id in primary_ids]
    non_tie = [pref for pref in prefs if pref != "tie"]
    left = sum(pref == "left" for pref in non_tie)
    right = sum(pref == "right" for pref in non_tie)
    left_share = left / len(non_tie) if non_tie else None
    right_share = right / len(non_tie) if non_tie else None
    imbalance = abs(left_share - right_share) if left_share is not None and right_share is not None else None
    return {
        "canonical_primary_pairs": len(primary_ids),
        "non_tie_count": len(non_tie),
        "left_count": left,
        "right_count": right,
        "left_share": left_share,
        "right_share": right_share,
        "absolute_left_right_share_imbalance": imbalance,
    }


def summarize_reviews() -> dict[str, Any]:
    registry = read_json(REGISTRY)
    packet = by_field(read_jsonl(PAIR_PACKET), "opaque_pair_id", PAIR_PACKET)
    expected = set(packet)
    key = load_pair_key()
    a = validate_review(REVIEWER_A, expected)
    b = validate_review(REVIEWER_B, expected)
    c = load_c_labels()
    final = final_from_reviews(a, b, c)
    write_jsonl(FINAL_LABELS, [final[opaque_id] for opaque_id in sorted(final)])
    a_canon = canonical_primary_pairs(a, key)
    b_canon = canonical_primary_pairs(b, key)
    canonical_ids = sorted(set(a_canon) & set(b_canon))
    pref_pairs = [(a_canon[cid]["preference"], b_canon[cid]["preference"]) for cid in canonical_ids]
    specific_pairs = []
    for cid in canonical_ids:
        specific_pairs.append((a_canon[cid]["canonical_left_specific"], b_canon[cid]["canonical_left_specific"]))
        specific_pairs.append((a_canon[cid]["canonical_right_specific"], b_canon[cid]["canonical_right_specific"]))
    disputes = [opaque_id for opaque_id in expected if any(a[opaque_id][field] != b[opaque_id][field] for field in DECISION_FIELDS)]
    missing_c = [opaque_id for opaque_id in disputes if opaque_id not in c]
    a_reverse = reversed_consistency(a, key)
    b_reverse = reversed_consistency(b, key)
    a_bias = position_bias(a, key)
    b_bias = position_bias(b, key)
    gates = registry["cr0_measurement"]
    agreement = {
        "name": "step3cr_cr0_review_agreement_v1",
        "review_rows": len(expected),
        "canonical_pairs": len(canonical_ids),
        "reversed_duplicates": len([row for row in key.values() if row["is_reversed_duplicate"] == "True"]),
        "pair_preference": metric_block(pref_pairs),
        "specific_safe_next_step": metric_block(specific_pairs),
        "reviewer_a_reversed_consistency": a_reverse,
        "reviewer_b_reversed_consistency": b_reverse,
        "reviewer_a_position_bias": a_bias,
        "reviewer_b_position_bias": b_bias,
        "decision_field_disagreement_rows": len(disputes),
        "missing_reviewer_c_adjudications": missing_c,
        "gates": {
            "pair_preference_pass": metric_block(pref_pairs)["raw_agreement"] >= float(gates["raw_agreement_min"])
            and metric_block(pref_pairs)["gwet_ac1"] >= float(gates["gwet_ac1_min"]),
            "specific_safe_next_step_pass": metric_block(specific_pairs)["raw_agreement"] >= float(gates["specific_safe_raw_agreement_min"])
            and metric_block(specific_pairs)["gwet_ac1"] >= float(gates["specific_safe_gwet_ac1_min"]),
            "reviewer_a_preference_reversal_pass": reversal_gate_pass(
                a_reverse["preference_consistent_count"],
                a_reverse["denominator"],
                int(gates["reversed_consistency_min_count"]),
            ),
            "reviewer_b_preference_reversal_pass": reversal_gate_pass(
                b_reverse["preference_consistent_count"],
                b_reverse["denominator"],
                int(gates["reversed_consistency_min_count"]),
            ),
            "reviewer_a_specificity_reversal_pass": reversal_gate_pass(
                a_reverse["specificity_consistent_count"],
                a_reverse["denominator"],
                int(gates["reversed_consistency_min_count"]),
            ),
            "reviewer_b_specificity_reversal_pass": reversal_gate_pass(
                b_reverse["specificity_consistent_count"],
                b_reverse["denominator"],
                int(gates["reversed_consistency_min_count"]),
            ),
            "reviewer_a_non_tie_pass": a_bias["non_tie_count"] >= int(gates["canonical_non_tie_count_min_per_reviewer"]),
            "reviewer_b_non_tie_pass": b_bias["non_tie_count"] >= int(gates["canonical_non_tie_count_min_per_reviewer"]),
            "reviewer_a_position_bias_pass": a_bias["absolute_left_right_share_imbalance"] is not None
            and a_bias["absolute_left_right_share_imbalance"] <= float(gates["absolute_position_imbalance_max"]),
            "reviewer_b_position_bias_pass": b_bias["absolute_left_right_share_imbalance"] is not None
            and b_bias["absolute_left_right_share_imbalance"] <= float(gates["absolute_position_imbalance_max"]),
            "all_disagreements_adjudicated": not missing_c,
        },
        "heldout_opened": False,
        "training_authorized": False,
    }
    agreement["gates"]["overall_pass"] = all(agreement["gates"].values())
    write_json(AGREEMENT_REPORT, agreement)
    route = {
        "name": "step3cr_cr0_route_decision_v1",
        "cr0_measurement_gate_pass": agreement["gates"]["overall_pass"],
        "next_authorized_phase": "C-R1 no-update diagnostics" if agreement["gates"]["overall_pass"] else "stop_or_one_complete_rubric_revision_if_first_session",
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(ROUTE_DECISION, route)
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return route


def protocol_packet_paths() -> list[Path]:
    return [
        PAIR_PACKET,
        PAIR_KEY,
        DUPLICATE_KEY,
        REVIEWER_A_TEMPLATE,
        REVIEWER_B_TEMPLATE,
        REVIEWER_C_PACKET,
    ]


def packet_byte_equality_snapshot() -> dict[str, Any]:
    files = []
    for path in protocol_packet_paths():
        baseline = git_blob_bytes(BASELINE_PROTOCOL_COMMIT, path)
        current = path.read_bytes()
        files.append({
            "path": path.as_posix(),
            "baseline_commit": BASELINE_PROTOCOL_COMMIT,
            "baseline_sha256": sha256_bytes(baseline),
            "current_sha256": sha256_bytes(current),
            "byte_equal": baseline == current,
            "baseline_bytes": len(baseline),
            "current_bytes": len(current),
        })
    baseline_manifest = json.loads(git_blob_bytes(BASELINE_PROTOCOL_COMMIT, PACKET_MANIFEST).decode("utf-8"))
    current_manifest = read_json(PACKET_MANIFEST)
    frozen_fields = ["packet_rows", "canonical_pairs", "reversed_duplicates", "randomization_seed", "packet_sha256", "hidden_key_sha256", "duplicate_selection_sha256"]
    manifest_fields = {
        field: {
            "baseline": baseline_manifest[field],
            "current": current_manifest[field],
            "equal": baseline_manifest[field] == current_manifest[field],
        }
        for field in frozen_fields
    }
    result = {
        "baseline_commit": BASELINE_PROTOCOL_COMMIT,
        "files": files,
        "manifest_fields": manifest_fields,
        "all_packet_files_byte_equal": all(row["byte_equal"] for row in files),
        "all_frozen_manifest_fields_equal": all(row["equal"] for row in manifest_fields.values()),
    }
    result["passed"] = result["all_packet_files_byte_equal"] and result["all_frozen_manifest_fields_equal"]
    return result


def verify_canonical_lf_outputs(paths: list[Path]) -> dict[str, Any]:
    rows = []
    failures = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        data = path.read_bytes()
        canonical = b"\r" not in data
        rows.append({"path": path.as_posix(), "canonical_lf": canonical, "bytes": len(data)})
        if not canonical:
            failures.append(path.as_posix())
    result = {
        "checked_files": len(rows),
        "noncanonical_files": failures,
        "passed": not failures,
        "files": rows,
    }
    if failures:
        raise ValueError(f"non-canonical LF protocol outputs: {failures}")
    return result


def artifact_paths() -> list[Path]:
    paths: list[Path] = [
        REGISTRY,
        RUBRIC,
        SCRIPT_PATH,
        PAIR_PACKET,
        PAIR_KEY,
        DUPLICATE_KEY,
        REVIEWER_A_TEMPLATE,
        REVIEWER_B_TEMPLATE,
        REVIEWER_C_PACKET,
        PACKET_MANIFEST,
        PACKETS / "step3cr_cr0_reviewer_c_packet_manifest.json",
        INPUT_VALIDATION,
        SYNTHETIC_TESTS,
        P0_EQUALITY,
        CANONICAL_LF_REPORT,
        AGREEMENT_REPORT,
        ROUTE_DECISION,
        PROTOCOL_REPORT,
        P0_AMENDMENT_REPORT,
        PROTOCOL_MANIFEST,
    ]
    return [path for path in paths if path.exists()]


def verify_artifacts(write: bool = True) -> dict[str, Any]:
    manifest = read_json(PROTOCOL_MANIFEST) if PROTOCOL_MANIFEST.exists() else {"artifacts": []}
    checked = []
    status_counts: Counter[str] = Counter()
    non_eol_mismatches = []
    eol_only_mismatches = []
    for row in manifest.get("artifacts", []):
        path = Path(row["path"])
        exists = path.exists()
        status = "missing" if not exists else classify_bytes_against_hash(path.read_bytes(), row["sha256"])
        status_counts[status] += 1
        checked.append({"path": str(path), "exists": exists, "status": status})
        if status.startswith("eol_only_"):
            eol_only_mismatches.append({"path": str(path), "classification": status})
        elif status in {"missing", "non_eol_mismatch"}:
            non_eol_mismatches.append({"path": str(path), "classification": status})
    result = {
        "protocol_manifest": str(PROTOCOL_MANIFEST),
        "checked_artifacts": len(checked),
        "raw_match_count": status_counts["raw_match"],
        "crlf_to_lf_match_count": status_counts["eol_only_crlf_to_lf"],
        "lf_to_crlf_match_count": status_counts["eol_only_lf_to_crlf"],
        "eol_only_mismatch_count": len(eol_only_mismatches),
        "eol_only_mismatches": eol_only_mismatches,
        "non_eol_mismatch_count": len(non_eol_mismatches),
        "non_eol_mismatches": non_eol_mismatches,
        "passed": not non_eol_mismatches,
        "heldout_opened": False,
        "training_authorized": False,
    }
    if write:
        write_json(FRESH_VERIFY, result)
    if non_eol_mismatches:
        raise ValueError(f"artifact verification failed: {non_eol_mismatches}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def freeze_protocol() -> dict[str, Any]:
    started = time.time()
    pre_regeneration_equality = packet_byte_equality_snapshot()
    if not pre_regeneration_equality["passed"]:
        raise ValueError("P0 pre-regeneration packet equality check failed")
    validation = validate_preconditions()
    tests = run_synthetic_tests()
    packet_manifest = build_packets()
    post_regeneration_equality = packet_byte_equality_snapshot()
    equality = {
        "name": "step3cr_p0_packet_byte_equality_v1",
        "baseline_commit": BASELINE_PROTOCOL_COMMIT,
        "pre_regeneration": pre_regeneration_equality,
        "post_regeneration": post_regeneration_equality,
        "packet_mutation_detected": not (
            pre_regeneration_equality["passed"] and post_regeneration_equality["passed"]
        ),
        "passed": pre_regeneration_equality["passed"] and post_regeneration_equality["passed"],
    }
    write_json(P0_EQUALITY, equality)
    if not equality["passed"]:
        raise ValueError("P0 packet or hidden-key mutation detected")
    hidden_check = verify_packet_hidden_fields()
    reviewer_label_files = sorted(REVIEWS.glob("*.jsonl")) if REVIEWS.exists() else []
    if reviewer_label_files:
        raise ValueError(f"P0 must run before reviewer labels: {reviewer_label_files}")
    report = f"""# Step3C-R C-R0 Protocol Freeze Report

Status: hardened and re-frozen before C-R0 reviewer labeling.

## Inputs

- Git HEAD: `{git_head()}`
- Registry: `{REGISTRY}`
- Source plan: `{read_json(REGISTRY)['source_plan']}`
- Stage0 stopped decision: `{read_json(REGISTRY)['stopped_stage0_decision']}`
- Stage0 artifacts verified: {validation['stage0_artifacts']['stage0_manifest_entries']} entries, non-EOL mismatches {validation['stage0_artifacts']['byte_hash_mismatch_count']}

## Pair Construction

- Canonical pairs: {packet_manifest['canonical_pairs']}
- Reversed duplicates: {packet_manifest['reversed_duplicates']}
- Blinded packet rows: {packet_manifest['packet_rows']}
- Randomization seed: {packet_manifest['randomization_seed']}
- Construction rules: `{json.dumps(packet_manifest['construction_rule_counts'], sort_keys=True)}`
- Packet byte-equal to `{BASELINE_PROTOCOL_COMMIT}`: {equality['post_regeneration']['all_packet_files_byte_equal']}
- Frozen manifest fields unchanged: {equality['post_regeneration']['all_frozen_manifest_fields_equal']}

## Frozen Review Protocol

- Rubric: `{RUBRIC}`
- Reviewer A template: `{REVIEWER_A_TEMPLATE}`
- Reviewer B template: `{REVIEWER_B_TEMPLATE}`
- Reviewer C packet schema: `{REVIEWER_C_PACKET}`
- Hidden pair key: `{PAIR_KEY}`
- Duplicate key: `{DUPLICATE_KEY}`

Reviewers see only prompt plus left/right responses. They do not see sample IDs, source roles, model outputs, duplicate links, or hidden keys.

## Validation

- Synthetic pair-order canonicalization: {tests['preference_canonicalization_passed']}
- Synthetic specificity reversal: {tests['specificity_canonicalization_passed']}
- Synthetic reversal-gate failure: {tests['reversal_gate_failure_detected']}
- Symmetric EOL fixtures: {tests['crlf_to_lf_classification_passed'] and tests['lf_to_crlf_classification_passed']}
- Non-EOL mutation rejection: {tests['non_eol_mutation_rejected']}
- Synthetic hidden-key isolation: {tests['hidden_key_isolation_passed']}
- Packet hidden-field verification: {hidden_check['passed']}
- Reviewer label files: {len(reviewer_label_files)}
- Heldout opened: false
- Training authorized: false

## Next Authorized Step

C-R0 independent Reviewer A/B labeling may start. Main executor remains barred from assigning blind labels.
"""
    PROTOCOL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_REPORT.write_text(report, encoding="utf-8", newline="\n")
    amendment_report = f"""# Step3C-R P0 Protocol Amendment Report

Status: completed prospectively after `{BASELINE_PROTOCOL_COMMIT}` and before any C-R0 label.

## Reason

The original C-R0 implementation reported preference reversal consistency but did not make response-side specificity reversal consistency an independent gate. Its legacy artifact verifier also classified only one EOL direction. P0 corrects both implementation gaps without changing the scientific packet.

## Frozen Corrections

- Reviewer A and B each have separate `>=7/8` preference reversal gates.
- Reviewer A and B each have separate `>=7/8` specificity-vector reversal gates.
- Naturalness reversal consistency is reported descriptively and is not a promotion gate.
- Artifact verification classifies raw matches, `CRLF -> LF`, `LF -> CRLF`, and non-EOL mismatches symmetrically.
- New protocol outputs are required to use canonical LF bytes.

## Packet Preservation

| artifact | old SHA-256 | new SHA-256 | byte equal |
| --- | --- | --- | --- |
{chr(10).join(f"| `{row['path']}` | `{row['baseline_sha256']}` | `{row['current_sha256']}` | {row['byte_equal']} |" for row in equality['post_regeneration']['files'])}

Canonical pairs remain {packet_manifest['canonical_pairs']}; reversed duplicates remain {packet_manifest['reversed_duplicates']}; randomization seed remains {packet_manifest['randomization_seed']}. Response text, opaque IDs, presentation order, hidden pair mapping, and duplicate links are byte-identical to `{BASELINE_PROTOCOL_COMMIT}`.

## Integrity

- Legacy Stage0 artifacts: {validation['stage0_artifacts']['stage0_manifest_entries']}
- Raw hash matches: {validation['stage0_artifacts']['raw_match_count']}
- EOL-only classifications: {validation['stage0_artifacts']['eol_only_mismatch_count']}
- Non-EOL mismatches: {validation['stage0_artifacts']['byte_hash_mismatch_count']}
- Synthetic tests passed: true
- Reviewer labels present: {len(reviewer_label_files)}
- Optimizer steps: 0
- Heldout opened: false

## Commands

```text
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py test-fixtures
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py freeze-protocol
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py verify-artifacts
```

## Continuation

P0 passes. C-R0 independent Reviewer A/B labeling is authorized next. No model diagnostic or optimizer step is authorized before P1 passes.
"""
    P0_AMENDMENT_REPORT.write_text(amendment_report, encoding="utf-8", newline="\n")
    canonical_targets = [
        REGISTRY,
        RUBRIC,
        SCRIPT_PATH,
        *protocol_packet_paths(),
        PACKET_MANIFEST,
        INPUT_VALIDATION,
        SYNTHETIC_TESTS,
        P0_EQUALITY,
        PROTOCOL_REPORT,
        P0_AMENDMENT_REPORT,
    ]
    canonical_lf = verify_canonical_lf_outputs(canonical_targets)
    canonical_lf.update({
        "name": "step3cr_p0_canonical_lf_verification_v1",
        "heldout_opened": False,
        "training_authorized": False,
    })
    write_json(CANONICAL_LF_REPORT, canonical_lf)
    artifacts = []
    for path in artifact_paths():
        if path == PROTOCOL_MANIFEST:
            continue
        artifacts.append({"path": path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "name": "step3cr_p0_hardened_protocol_manifest_v2",
        "git_head": git_head(),
        "status": "p0_hardened_and_refrozen_before_reviewer_labels",
        "baseline_protocol_commit": BASELINE_PROTOCOL_COMMIT,
        "packet_byte_equality_pass": equality["passed"],
        "canonical_lf_pass": canonical_lf["passed"],
        "reviewer_label_file_count": len(reviewer_label_files),
        "optimizer_steps": 0,
        "wall_clock_seconds": time.time() - started,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(PROTOCOL_MANIFEST, manifest)
    verify_artifacts(write=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "validate-preconditions",
            "test-fixtures",
            "build-packets",
            "freeze-protocol",
            "verify-artifacts",
            "prepare-adjudication",
            "summarize-reviews",
        ],
    )
    args = parser.parse_args()
    if args.command == "validate-preconditions":
        validate_preconditions()
    elif args.command == "test-fixtures":
        run_synthetic_tests()
    elif args.command == "build-packets":
        build_packets()
    elif args.command == "freeze-protocol":
        freeze_protocol()
    elif args.command == "verify-artifacts":
        verify_artifacts(write=True)
    elif args.command == "prepare-adjudication":
        prepare_adjudication()
    elif args.command == "summarize-reviews":
        summarize_reviews()


if __name__ == "__main__":
    main()
