#!/usr/bin/env python3
"""Prepare and analyze the Step3C-M decomposed quality measurement study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/step3cmf_registry.json"
BASE = ROOT / "results/step3cm_protocol"
PACKETS = BASE / "packets"
REVIEWS = BASE / "reviews"
REPORTS = BASE / "reports"
DIAGNOSTICS = BASE / "diagnostics"

INPUT_VALIDATION = DIAGNOSTICS / "step3cm_input_validation.json"
LOCAL_ENVIRONMENT = DIAGNOSTICS / "step3cm_local_environment.json"
SYNTHETIC_TESTS = DIAGNOSTICS / "step3cm_synthetic_tests.json"
SPLIT_CSV = REPORTS / "step3cm_prompt_split.csv"
PROTOCOL_REPORT = REPORTS / "Step3CM_protocol_freeze_report.md"
PROTOCOL_MANIFEST = REPORTS / "step3cm_protocol_manifest.json"
ACTIVE_RUBRIC_FREEZE = REPORTS / "step3cm_active_rubric_freeze.json"

ORDINAL_FIELDS = ["detail_retention", "safe_actionability"]
DECISION_FIELDS = ORDINAL_FIELDS + ["naturalness"]
REVIEW_FIELDS = ["opaque_quality_id"] + DECISION_FIELDS + ["short_reason"]
ORDINAL_VALUES = {0, 1, 2}
BOOL_VALUES = {"yes", "no"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json_text(row) for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], require_equal_if_exists: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = jsonl_text(rows)
    if require_equal_if_exists and path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to mutate frozen packet: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str], require_equal_if_exists: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    content = buffer.getvalue()
    if require_equal_if_exists and path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to mutate frozen key: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def registry() -> dict[str, Any]:
    return read_json(REGISTRY)


def active_rubric(reg: dict[str, Any]) -> Path:
    return Path(reg["measurement"]["active_rubric"])


def packet_paths(split: str) -> dict[str, Path]:
    base = PACKETS / split
    return {
        "packet": base / f"step3cm_{split}_packet.jsonl",
        "key": base / f"step3cm_{split}_hidden_key.csv",
        "duplicates": base / f"step3cm_{split}_duplicate_key.csv",
        "reviewer_a_template": base / f"step3cm_{split}_reviewer_a_template.jsonl",
        "reviewer_b_template": base / f"step3cm_{split}_reviewer_b_template.jsonl",
        "reviewer_c_packet": base / f"step3cm_{split}_reviewer_c_packet.jsonl",
        "manifest": base / f"step3cm_{split}_packet_manifest.json",
    }


def review_paths(split: str) -> dict[str, Path]:
    base = REVIEWS / split
    return {
        "reviewer_a": base / f"step3cm_{split}_reviewer_a.jsonl",
        "reviewer_b": base / f"step3cm_{split}_reviewer_b.jsonl",
        "reviewer_c": base / f"step3cm_{split}_reviewer_c.jsonl",
        "final": base / f"step3cm_{split}_final_labels.jsonl",
        "ab_freeze": base / f"step3cm_{split}_ab_freeze_manifest.json",
    }


def result_paths(split: str) -> dict[str, Path]:
    return {
        "agreement": REPORTS / f"step3cm_{split}_review_agreement.json",
        "route": REPORTS / f"step3cm_{split}_route_decision.json",
        "report": REPORTS / f"Step3CM_{split}_route_report.md",
        "manifest": REPORTS / f"step3cm_{split}_review_artifact_manifest.json",
    }


def load_sources() -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    reg = registry()
    measurement = reg["measurement"]
    packet_path = Path(measurement["source_packet"])
    key_path = Path(measurement["source_role_key"])
    split_path = Path(measurement["split_manifest"])
    rubric = active_rubric(reg)
    closure = Path(reg["cr0_closure_summary"])
    required = [packet_path, key_path, split_path, rubric, closure]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Step3C-M inputs: {missing}")

    packet = read_jsonl(packet_path)
    key = read_csv(key_path)
    split_rows = read_csv(split_path)
    packet_by_id = {row["opaque_response_id"]: row for row in packet}
    if len(packet_by_id) != len(packet):
        raise ValueError("duplicate source packet IDs")
    if {row["opaque_response_id"] for row in key} != set(packet_by_id):
        raise ValueError("source packet/key IDs differ")

    safety_train_ids = {row["sample_id"] for row in split_rows if row["split"] == "safety_train"}
    source_sample_ids = {row["sample_id"] for row in key}
    if source_sample_ids != safety_train_ids:
        raise ValueError("Step3C-M source prompts must exactly equal the frozen safety_train split")

    validation = {
        "git_head": git_head(),
        "source_packet": str(packet_path),
        "source_packet_sha256": sha256(packet_path),
        "source_key": str(key_path),
        "source_key_sha256": sha256(key_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256(split_path),
        "active_rubric": str(rubric),
        "active_rubric_sha256": sha256(rubric),
        "closure_summary": str(closure),
        "closure_summary_sha256": sha256(closure),
        "source_responses": len(packet),
        "source_prompt_clusters": len(source_sample_ids),
        "role_counts": dict(Counter(row["role"] for row in key)),
        "category_prompt_counts": dict(Counter(next(item["category"] for item in key if item["sample_id"] == sample_id) for sample_id in sorted(source_sample_ids))),
        "all_source_ids_are_safety_train": True,
        "core100_opened_for_protocol_development": False,
        "heldout_opened": False,
        "candidate_outputs_used": False,
        "old_cr0_labels_used_as_new_labels": False,
        "optimizer_steps": 0,
    }
    expected_rows = int(measurement["expected_source_responses"])
    expected_prompts = int(measurement["prompt_clusters"])
    if len(packet) != expected_rows or len(source_sample_ids) != expected_prompts:
        raise ValueError(f"unexpected source dimensions: {len(packet)} responses, {len(source_sample_ids)} prompts")
    return packet, key, validation


def stable_rank(seed: int, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


def build_prompt_split(key: list[dict[str, str]], seed: int, development_target: int) -> dict[str, str]:
    category_to_samples: dict[str, set[str]] = defaultdict(set)
    for row in key:
        category_to_samples[row["category"]].add(row["sample_id"])

    base_quota = {category: len(samples) // 2 for category, samples in category_to_samples.items()}
    remaining = development_target - sum(base_quota.values())
    odd_categories = [category for category, samples in category_to_samples.items() if len(samples) % 2]
    odd_categories.sort(key=lambda category: stable_rank(seed, "odd-category", category))
    if remaining < 0 or remaining > len(odd_categories):
        raise ValueError("cannot construct balanced development split")
    for category in odd_categories[:remaining]:
        base_quota[category] += 1

    assignment: dict[str, str] = {}
    for category, samples in sorted(category_to_samples.items()):
        ranked = sorted(samples, key=lambda sample_id: stable_rank(seed, category, sample_id))
        dev = set(ranked[: base_quota[category]])
        for sample_id in ranked:
            assignment[sample_id] = "development" if sample_id in dev else "validation"

    counts = Counter(assignment.values())
    if counts["development"] != development_target or counts["validation"] != len(assignment) - development_target:
        raise ValueError(f"bad prompt split counts: {counts}")
    return assignment


def opaque_quality_id(seed: int, split: str, source_id: str, instance: str) -> str:
    token = stable_rank(seed, split, source_id, instance)[:16]
    return f"step3cm_{split[0]}_{token}"


def build_split_packet(
    split: str,
    source_packet: list[dict[str, Any]],
    key: list[dict[str, str]],
    assignment: dict[str, str],
    packet_seed: int,
    duplicate_seed: int,
    duplicate_count: int,
) -> dict[str, Any]:
    packet_by_id = {row["opaque_response_id"]: row for row in source_packet}
    source_rows = [row for row in key if assignment[row["sample_id"]] == split]
    source_rows.sort(key=lambda row: stable_rank(packet_seed, split, row["opaque_response_id"]))
    source_to_primary: dict[str, str] = {}
    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for source in source_rows:
        source_id = source["opaque_response_id"]
        opaque_id = opaque_quality_id(packet_seed, split, source_id, "primary")
        source_to_primary[source_id] = opaque_id
        visible = packet_by_id[source_id]
        packet_rows.append({
            "opaque_quality_id": opaque_id,
            "prompt": visible["prompt"],
            "response": visible["response"],
        })
        key_rows.append({
            "opaque_quality_id": opaque_id,
            "source_opaque_response_id": source_id,
            "sample_id": source["sample_id"],
            "category": source["category"],
            "role": source["role"],
            "source_response_id": source["source_response_id"],
            "is_hidden_duplicate": False,
            "duplicate_of_quality_id": "",
            "prompt_sha256": sha256_text(visible["prompt"]),
            "response_sha256": sha256_text(visible["response"]),
        })

    ranked_for_duplicates = sorted(source_rows, key=lambda row: stable_rank(duplicate_seed, split, row["opaque_response_id"]))
    duplicate_sources = ranked_for_duplicates[:duplicate_count]
    duplicate_rows: list[dict[str, Any]] = []
    for source in duplicate_sources:
        source_id = source["opaque_response_id"]
        opaque_id = opaque_quality_id(packet_seed, split, source_id, "duplicate")
        visible = packet_by_id[source_id]
        packet_rows.append({
            "opaque_quality_id": opaque_id,
            "prompt": visible["prompt"],
            "response": visible["response"],
        })
        key_rows.append({
            "opaque_quality_id": opaque_id,
            "source_opaque_response_id": source_id,
            "sample_id": source["sample_id"],
            "category": source["category"],
            "role": source["role"],
            "source_response_id": source["source_response_id"],
            "is_hidden_duplicate": True,
            "duplicate_of_quality_id": source_to_primary[source_id],
            "prompt_sha256": sha256_text(visible["prompt"]),
            "response_sha256": sha256_text(visible["response"]),
        })
        duplicate_rows.append({
            "source_opaque_response_id": source_id,
            "primary_opaque_quality_id": source_to_primary[source_id],
            "duplicate_opaque_quality_id": opaque_id,
        })

    rng = random.Random(packet_seed + (0 if split == "development" else 1))
    rng.shuffle(packet_rows)
    template_rows = [
        {
            "opaque_quality_id": row["opaque_quality_id"],
            "detail_retention": "",
            "safe_actionability": "",
            "naturalness": "",
            "short_reason": "",
        }
        for row in packet_rows
    ]
    if len({row["opaque_quality_id"] for row in packet_rows}) != len(packet_rows):
        raise ValueError(f"duplicate opaque quality IDs in {split}")

    visible_fields_ok = all(set(row) == {"opaque_quality_id", "prompt", "response"} for row in packet_rows)
    key_by_id = {row["opaque_quality_id"]: row for row in key_rows}
    packet_by_quality_id = {row["opaque_quality_id"]: row for row in packet_rows}
    duplicate_exact_match = all(
        packet_by_quality_id[row["primary_opaque_quality_id"]]["prompt"]
        == packet_by_quality_id[row["duplicate_opaque_quality_id"]]["prompt"]
        and packet_by_quality_id[row["primary_opaque_quality_id"]]["response"]
        == packet_by_quality_id[row["duplicate_opaque_quality_id"]]["response"]
        and key_by_id[row["primary_opaque_quality_id"]]["source_opaque_response_id"]
        == key_by_id[row["duplicate_opaque_quality_id"]]["source_opaque_response_id"]
        for row in duplicate_rows
    )
    if not visible_fields_ok or not duplicate_exact_match:
        raise ValueError(f"packet visibility or duplicate integrity failed for {split}")

    return {
        "packet_rows": packet_rows,
        "key_rows": key_rows,
        "duplicate_rows": duplicate_rows,
        "template_rows": template_rows,
        "source_response_count": len(source_rows),
        "prompt_count": len({row["sample_id"] for row in source_rows}),
        "role_counts": dict(Counter(row["role"] for row in source_rows)),
        "duplicate_role_counts": dict(Counter(row["role"] for row in duplicate_sources)),
        "visible_fields_ok": visible_fields_ok,
        "duplicate_exact_match": duplicate_exact_match,
    }


def freeze_protocol() -> dict[str, Any]:
    reg = registry()
    measurement = reg["measurement"]
    source_packet, key, validation = load_sources()
    assignment = build_prompt_split(
        key,
        int(measurement["split_seed"]),
        int(measurement["development_prompt_clusters"]),
    )

    prompt_rows = []
    sample_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in key:
        sample_to_rows[row["sample_id"]].append(row)
    for sample_id in sorted(assignment):
        rows = sample_to_rows[sample_id]
        prompt_rows.append({
            "sample_id": sample_id,
            "category": rows[0]["category"],
            "measurement_split": assignment[sample_id],
            "response_count": len(rows),
            "has_prospective_positive": any(row["role"] == "prospective_positive_candidate" for row in rows),
        })
    write_csv(SPLIT_CSV, prompt_rows, ["sample_id", "category", "measurement_split", "response_count", "has_prospective_positive"], require_equal_if_exists=True)

    split_manifests: dict[str, Any] = {}
    for split in ("development", "validation"):
        built = build_split_packet(
            split,
            source_packet,
            key,
            assignment,
            int(measurement["packet_seed"]),
            int(measurement["duplicate_seed"]),
            int(measurement["hidden_duplicates_per_split"]),
        )
        paths = packet_paths(split)
        write_jsonl(paths["packet"], built["packet_rows"], require_equal_if_exists=True)
        write_csv(
            paths["key"],
            built["key_rows"],
            [
                "opaque_quality_id",
                "source_opaque_response_id",
                "sample_id",
                "category",
                "role",
                "source_response_id",
                "is_hidden_duplicate",
                "duplicate_of_quality_id",
                "prompt_sha256",
                "response_sha256",
            ],
            require_equal_if_exists=True,
        )
        write_csv(
            paths["duplicates"],
            built["duplicate_rows"],
            ["source_opaque_response_id", "primary_opaque_quality_id", "duplicate_opaque_quality_id"],
            require_equal_if_exists=True,
        )
        write_jsonl(paths["reviewer_a_template"], built["template_rows"], require_equal_if_exists=True)
        write_jsonl(paths["reviewer_b_template"], built["template_rows"], require_equal_if_exists=True)
        if not paths["reviewer_c_packet"].exists():
            write_jsonl(paths["reviewer_c_packet"], [])
        manifest = {
            "name": f"step3cm_{split}_packet_manifest_v1",
            "git_head": git_head(),
            "measurement_split": split,
            "active_rubric": str(active_rubric(reg)),
            "active_rubric_sha256": sha256(active_rubric(reg)),
            "prompt_clusters": built["prompt_count"],
            "primary_response_rows": built["source_response_count"],
            "hidden_duplicate_rows": len(built["duplicate_rows"]),
            "packet_rows": len(built["packet_rows"]),
            "role_counts_hidden": built["role_counts"],
            "duplicate_role_counts_hidden": built["duplicate_role_counts"],
            "packet_sha256": sha256(paths["packet"]),
            "hidden_key_sha256": sha256(paths["key"]),
            "duplicate_key_sha256": sha256(paths["duplicates"]),
            "reviewer_a_template_sha256": sha256(paths["reviewer_a_template"]),
            "reviewer_b_template_sha256": sha256(paths["reviewer_b_template"]),
            "visible_fields": ["opaque_quality_id", "prompt", "response"],
            "source_roles_hidden": True,
            "duplicate_links_hidden": True,
            "visible_fields_ok": built["visible_fields_ok"],
            "duplicate_exact_match": built["duplicate_exact_match"],
            "old_cr0_labels_exposed": False,
            "candidate_outputs_used": False,
            "heldout_opened": False,
            "optimizer_steps": 0,
        }
        write_json(paths["manifest"], manifest)
        split_manifests[split] = manifest

    split_counts = Counter(row["measurement_split"] for row in prompt_rows)
    category_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in prompt_rows:
        category_split[row["category"]][row["measurement_split"]] += 1
    validation.update({
        "measurement_prompt_split_counts": dict(split_counts),
        "category_prompt_split_counts": {category: dict(counts) for category, counts in sorted(category_split.items())},
        "prompt_overlap_between_measurement_splits": 0,
        "development_packet_rows": split_manifests["development"]["packet_rows"],
        "validation_packet_rows": split_manifests["validation"]["packet_rows"],
        "development_packet_visible_fields_ok": split_manifests["development"]["visible_fields_ok"],
        "validation_packet_visible_fields_ok": split_manifests["validation"]["visible_fields_ok"],
        "development_duplicates_exact": split_manifests["development"]["duplicate_exact_match"],
        "validation_duplicates_exact": split_manifests["validation"]["duplicate_exact_match"],
    })
    write_json(INPUT_VALIDATION, validation)

    checkpoint = Path(reg["c3_no_update"]["checkpoint_dir"])
    local_environment = {
        "name": "step3cm_local_preparation_environment_v1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_importable": importlib.util.find_spec("torch") is not None,
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "checkpoint_dir_exists": checkpoint.exists(),
        "adapter_file_exists": (checkpoint / "adapter_model.safetensors").exists(),
        "gpu_diagnostics_run_locally": False,
        "optimizer_steps": 0,
        "experiment_machine_required_for_f1": True,
    }
    write_json(LOCAL_ENVIRONMENT, local_environment)

    report = f"""# Step3C-M Decomposed Measurement Protocol Freeze

Status: frozen before any Step3C-M label and before any C3 diagnostic.

## Source Boundary

- Source responses: {validation['source_responses']}
- Source prompt clusters: {validation['source_prompt_clusters']}
- Source split: frozen safety_train only
- Core100 opened for protocol development: false
- Heldout opened: false
- Candidate outputs used: false
- Old C-R0 labels reused: false

## Construct Development and Validation

| split | prompt clusters | primary responses | hidden duplicates | packet rows |
| --- | ---: | ---: | ---: | ---: |
| development | {split_manifests['development']['prompt_clusters']} | {split_manifests['development']['primary_response_rows']} | {split_manifests['development']['hidden_duplicate_rows']} | {split_manifests['development']['packet_rows']} |
| validation | {split_manifests['validation']['prompt_clusters']} | {split_manifests['validation']['primary_response_rows']} | {split_manifests['validation']['hidden_duplicate_rows']} | {split_manifests['validation']['packet_rows']} |

The split is prompt-cluster disjoint and category-balanced to within one prompt where possible. Development may use at most one complete rubric revision. Validation is opened once with fresh reviewers and permits no revision.

## Frozen Fields

- detail_retention: 0/1/2
- safe_actionability: 0/1/2
- naturalness: yes/no
- derived quality eligibility: detail >= 1, actionability >= 1, naturalness = yes

Clean safety remains a separate behavior label and is joined only after quality labels freeze. Holistic pair preference is not a promotion endpoint in this study.

## Next Authorized Action

Independent fresh-context development Reviewer A/B labeling may begin on the experiment machine. No optimizer, scheduler, C3 behavior generation, core100 evaluation, or training is authorized.
"""
    PROTOCOL_REPORT.write_text(report, encoding="utf-8", newline="\n")
    manifest_paths = [
        REGISTRY,
        active_rubric(reg),
        Path(__file__).resolve(),
        INPUT_VALIDATION,
        LOCAL_ENVIRONMENT,
        SYNTHETIC_TESTS,
        SPLIT_CSV,
        PROTOCOL_REPORT,
    ]
    for split in ("development", "validation"):
        paths = packet_paths(split)
        manifest_paths.extend(paths.values())
    artifacts = [
        {"path": path.relative_to(Path.cwd()).as_posix() if path.is_absolute() and Path.cwd() in path.parents else path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in manifest_paths
        if path.exists() and path != PROTOCOL_MANIFEST
    ]
    protocol_manifest = {
        "name": "step3cm_protocol_manifest_v1",
        "git_head": git_head(),
        "status": "frozen_before_development_labels",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(PROTOCOL_MANIFEST, protocol_manifest)
    print(json.dumps(protocol_manifest, ensure_ascii=False, indent=2))
    return protocol_manifest


def raw_agreement(pairs: list[tuple[Any, Any]]) -> float:
    if not pairs:
        raise ValueError("agreement requires at least one pair")
    return sum(left == right for left, right in pairs) / len(pairs)


def cohen_kappa(pairs: list[tuple[Any, Any]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair}, key=str)
    observed = raw_agreement(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs)) for label in labels)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def gwet_ac1(pairs: list[tuple[Any, Any]], labels: list[Any] | None = None) -> float | None:
    if not pairs:
        return None
    categories = labels or sorted({value for pair in pairs for value in pair}, key=str)
    if len(categories) < 2:
        return 1.0 if all(left == right for left, right in pairs) else None
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {label: pooled[label] / (2 * len(pairs)) for label in categories}
    expected = sum(prob * (1 - prob) for prob in probabilities.values()) / (len(categories) - 1)
    if math.isclose(expected, 1.0):
        return None
    return (raw_agreement(pairs) - expected) / (1.0 - expected)


def linear_weight(left: int, right: int, categories: list[int]) -> float:
    span = max(categories) - min(categories)
    return 1.0 if span == 0 else 1.0 - abs(left - right) / span


def weighted_agreement(pairs: list[tuple[int, int]], categories: list[int]) -> float:
    if not pairs:
        raise ValueError("weighted agreement requires pairs")
    return sum(linear_weight(left, right, categories) for left, right in pairs) / len(pairs)


def weighted_kappa(pairs: list[tuple[int, int]], categories: list[int]) -> float | None:
    if not pairs:
        return None
    observed = weighted_agreement(pairs, categories)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = 0.0
    for left in categories:
        for right in categories:
            expected += linear_weight(left, right, categories) * (left_counts[left] / len(pairs)) * (right_counts[right] / len(pairs))
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def gwet_ac2(pairs: list[tuple[int, int]], categories: list[int]) -> float | None:
    if not pairs:
        return None
    observed = weighted_agreement(pairs, categories)
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {category: pooled[category] / (2 * len(pairs)) for category in categories}
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
    return (observed - expected) / (1.0 - expected)


def ordinal_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    categories = [0, 1, 2]
    return {
        "n": len(pairs),
        "exact_raw_agreement": raw_agreement(pairs),
        "linear_weighted_agreement": weighted_agreement(pairs, categories),
        "gwet_ac2": gwet_ac2(pairs, categories),
        "linear_weighted_kappa": weighted_kappa(pairs, categories),
        "severe_disagreement_count": sum(abs(left - right) == 2 for left, right in pairs),
        "severe_disagreement_rate": sum(abs(left - right) == 2 for left, right in pairs) / len(pairs),
    }


def nominal_metrics(pairs: list[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "raw_agreement": raw_agreement(pairs),
        "gwet_ac1": gwet_ac1(pairs, labels),
        "cohen_kappa": cohen_kappa(pairs),
    }


def run_synthetic_tests() -> dict[str, Any]:
    perfect_ordinal = [(0, 0), (1, 1), (2, 2), (1, 1)]
    one_step = [(0, 1), (1, 2), (2, 1), (1, 0)]
    zero_variance = [(2, 2)] * 8
    nominal_fixture = [("yes", "yes"), ("no", "no"), ("yes", "no"), ("no", "yes")]

    assert math.isclose(weighted_agreement(perfect_ordinal, [0, 1, 2]), 1.0)
    assert math.isclose(gwet_ac2(perfect_ordinal, [0, 1, 2]) or 0.0, 1.0)
    assert math.isclose(weighted_agreement(one_step, [0, 1, 2]), 0.5)
    assert math.isclose(gwet_ac2(zero_variance, [0, 1, 2]) or 0.0, 1.0)
    assert weighted_kappa(zero_variance, [0, 1, 2]) is None
    assert math.isclose(raw_agreement(nominal_fixture), 0.5)
    assert math.isclose(gwet_ac1([("yes", "yes")] * 8, ["yes", "no"]) or 0.0, 1.0)

    fake_key = [
        {"sample_id": "a1", "category": "a"},
        {"sample_id": "a2", "category": "a"},
        {"sample_id": "b1", "category": "b"},
        {"sample_id": "b2", "category": "b"},
    ]
    split = build_prompt_split(fake_key, 7, 2)
    assert Counter(split.values()) == {"development": 2, "validation": 2}
    assert not ({sample for sample, value in split.items() if value == "development"} & {sample for sample, value in split.items() if value == "validation"})

    review_fixture = [
        {
            "opaque_quality_id": "q1",
            "detail_retention": 1,
            "safe_actionability": 2,
            "naturalness": "yes",
            "short_reason": "fixture",
        },
        {
            "opaque_quality_id": "q2",
            "detail_retention": 1,
            "safe_actionability": 2,
            "naturalness": "yes",
            "short_reason": "fixture duplicate",
        },
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        review_path = Path(temp_dir) / "review.jsonl"
        write_jsonl(review_path, review_fixture)
        validated_fixture = validate_review(review_path, {"q1", "q2"})
    duplicate_fixture = duplicate_consistency(
        validated_fixture,
        [
            {"opaque_quality_id": "q1", "is_hidden_duplicate": "False", "duplicate_of_quality_id": ""},
            {"opaque_quality_id": "q2", "is_hidden_duplicate": "True", "duplicate_of_quality_id": "q1"},
        ],
    )
    assert duplicate_fixture["full_vector_consistent_count"] == 1
    assert quality_eligible(review_fixture[0], {"detail_retention_min": 1, "safe_actionability_min": 1, "naturalness_required": "yes"}) == "yes"

    tests = {
        "name": "step3cm_synthetic_tests_v1",
        "perfect_ordinal": ordinal_metrics(perfect_ordinal),
        "one_step_ordinal": ordinal_metrics(one_step),
        "zero_variance_ac2": gwet_ac2(zero_variance, [0, 1, 2]),
        "zero_variance_weighted_kappa": weighted_kappa(zero_variance, [0, 1, 2]),
        "nominal_fixture": nominal_metrics(nominal_fixture, ["yes", "no"]),
        "prompt_cluster_split_disjoint": True,
        "hidden_field_fixture": set(["opaque_quality_id", "prompt", "response"]).isdisjoint({"role", "source_response_id", "duplicate_of_quality_id"}),
        "review_schema_fixture": True,
        "duplicate_consistency_fixture": duplicate_fixture,
        "passed": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(SYNTHETIC_TESTS, tests)
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    return tests


def validate_review(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != set(REVIEW_FIELDS):
            raise ValueError(f"{path} has unexpected fields for {row.get('opaque_quality_id')}: {sorted(row)}")
        opaque_id = row["opaque_quality_id"]
        if opaque_id in by_id:
            raise ValueError(f"duplicate review ID in {path}: {opaque_id}")
        for field in ORDINAL_FIELDS:
            if isinstance(row[field], str) and row[field].isdigit():
                row[field] = int(row[field])
            if row[field] not in ORDINAL_VALUES:
                raise ValueError(f"bad {field} in {path}: {row[field]}")
        if row["naturalness"] not in BOOL_VALUES:
            raise ValueError(f"bad naturalness in {path}: {row['naturalness']}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ValueError(f"missing short_reason in {path}: {opaque_id}")
        by_id[opaque_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(f"review IDs differ in {path}: missing={sorted(expected_ids - set(by_id))}, extra={sorted(set(by_id) - expected_ids)}")
    return by_id


def validate_ab(split: str) -> dict[str, Any]:
    paths = packet_paths(split)
    reviews = review_paths(split)
    if split == "validation":
        validate_active_rubric_freeze()
    packet_ids = {row["opaque_quality_id"] for row in read_jsonl(paths["packet"])}
    reviewer_a = validate_review(reviews["reviewer_a"], packet_ids)
    reviewer_b = validate_review(reviews["reviewer_b"], packet_ids)
    freeze = {
        "name": f"step3cm_{split}_ab_freeze_manifest_v1",
        "measurement_split": split,
        "packet_sha256": sha256(paths["packet"]),
        "rubric_sha256": sha256(active_rubric(registry())),
        "reviewer_a_rows": len(reviewer_a),
        "reviewer_b_rows": len(reviewer_b),
        "reviewer_a_sha256": sha256(reviews["reviewer_a"]),
        "reviewer_b_sha256": sha256(reviews["reviewer_b"]),
        "ab_compared_only_after_both_complete": True,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(reviews["ab_freeze"], freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    return freeze


def freeze_active_rubric() -> dict[str, Any]:
    development_route = result_paths("development")["route"]
    if not development_route.exists() or not read_json(development_route).get("overall_pass"):
        raise ValueError("active rubric may freeze only after a passing development route")
    reg = registry()
    rubric = active_rubric(reg)
    validation_paths = packet_paths("validation")
    m0_manifest = read_json(PROTOCOL_MANIFEST)
    m0_hashes = {row["path"].replace("\\", "/"): row["sha256"] for row in m0_manifest["artifacts"]}
    packet_repo_path = repo_path(validation_paths["packet"])
    key_repo_path = repo_path(validation_paths["key"])
    if packet_repo_path not in m0_hashes or key_repo_path not in m0_hashes:
        raise ValueError("validation packet/key are missing from the M0 protocol manifest")
    actual_packet_hash = sha256(validation_paths["packet"])
    actual_key_hash = sha256(validation_paths["key"])
    if actual_packet_hash != m0_hashes[packet_repo_path] or actual_key_hash != m0_hashes[key_repo_path]:
        raise ValueError("validation packet changed after M0 freeze")
    result = {
        "name": "step3cm_active_rubric_freeze_v1",
        "git_head": git_head(),
        "active_rubric": repo_path(rubric),
        "active_rubric_sha256": sha256(rubric),
        "development_route": repo_path(development_route),
        "development_route_sha256": sha256(development_route),
        "validation_packet": repo_path(validation_paths["packet"]),
        "validation_packet_sha256": actual_packet_hash,
        "validation_hidden_key_sha256": actual_key_hash,
        "m0_protocol_manifest_sha256": sha256(PROTOCOL_MANIFEST),
        "validation_packet_content_unchanged_from_m0": True,
        "validation_sessions_authorized": 1,
        "validation_rubric_revisions_authorized": 0,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(ACTIVE_RUBRIC_FREEZE, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def validate_active_rubric_freeze() -> dict[str, Any]:
    if not ACTIVE_RUBRIC_FREEZE.exists():
        raise FileNotFoundError("validation requires step3cm_active_rubric_freeze.json")
    frozen = read_json(ACTIVE_RUBRIC_FREEZE)
    rubric = active_rubric(registry())
    validation_paths = packet_paths("validation")
    checks = {
        "rubric": sha256(rubric) == frozen["active_rubric_sha256"],
        "packet": sha256(validation_paths["packet"]) == frozen["validation_packet_sha256"],
        "hidden_key": sha256(validation_paths["key"]) == frozen["validation_hidden_key_sha256"],
        "development_route": sha256(result_paths("development")["route"]) == frozen["development_route_sha256"],
        "m0_protocol_manifest": sha256(PROTOCOL_MANIFEST) == frozen["m0_protocol_manifest_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"active rubric freeze validation failed: {checks}")
    return {"passed": True, "checks": checks}


def prepare_adjudication(split: str) -> dict[str, Any]:
    paths = packet_paths(split)
    reviews = review_paths(split)
    packet = {row["opaque_quality_id"]: row for row in read_jsonl(paths["packet"])}
    expected = set(packet)
    reviewer_a = validate_review(reviews["reviewer_a"], expected)
    reviewer_b = validate_review(reviews["reviewer_b"], expected)
    rows = []
    for opaque_id in sorted(expected):
        disagreements = [field for field in DECISION_FIELDS if reviewer_a[opaque_id][field] != reviewer_b[opaque_id][field]]
        if disagreements:
            rows.append({
                "opaque_quality_id": opaque_id,
                "prompt": packet[opaque_id]["prompt"],
                "response": packet[opaque_id]["response"],
                "disagreement_fields": disagreements,
            })
    write_jsonl(paths["reviewer_c_packet"], rows)
    manifest = {
        "name": f"step3cm_{split}_reviewer_c_packet_manifest_v1",
        "measurement_split": split,
        "rows": len(rows),
        "packet_sha256": sha256(paths["reviewer_c_packet"]),
        "reviewer_labels_exposed": False,
        "reviewer_reasons_exposed": False,
        "hidden_key_exposed": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(paths["reviewer_c_packet"].with_name(paths["reviewer_c_packet"].stem + "_manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def load_reviewer_c(path: Path, expected_disputes: set[str]) -> dict[str, dict[str, Any]]:
    if not expected_disputes:
        return {}
    return validate_review(path, expected_disputes)


def duplicate_consistency(
    review: dict[str, dict[str, Any]],
    key_rows: list[dict[str, str]],
) -> dict[str, Any]:
    duplicates = [row for row in key_rows if row["is_hidden_duplicate"] == "True"]
    field_counts = Counter()
    full_vector = 0
    details = []
    for duplicate in duplicates:
        primary_id = duplicate["duplicate_of_quality_id"]
        duplicate_id = duplicate["opaque_quality_id"]
        matches = {field: review[primary_id][field] == review[duplicate_id][field] for field in DECISION_FIELDS}
        for field, matched in matches.items():
            field_counts[field] += int(matched)
        full_match = all(matches.values())
        full_vector += int(full_match)
        details.append({
            "primary_opaque_quality_id": primary_id,
            "duplicate_opaque_quality_id": duplicate_id,
            "field_matches": matches,
            "full_vector_match": full_match,
        })
    return {
        "denominator": len(duplicates),
        "full_vector_consistent_count": full_vector,
        "field_consistent_counts": dict(field_counts),
        "details": details,
    }


def quality_eligible(row: dict[str, Any], endpoint: dict[str, Any]) -> str:
    passed = (
        row["detail_retention"] >= int(endpoint["detail_retention_min"])
        and row["safe_actionability"] >= int(endpoint["safe_actionability_min"])
        and row["naturalness"] == endpoint["naturalness_required"]
    )
    return "yes" if passed else "no"


def summarize(split: str) -> dict[str, Any]:
    reg = registry()
    measurement = reg["measurement"]
    endpoint = reg["derived_endpoint"]
    paths = packet_paths(split)
    reviews = review_paths(split)
    outputs = result_paths(split)
    packet = {row["opaque_quality_id"]: row for row in read_jsonl(paths["packet"])}
    key_rows = read_csv(paths["key"])
    expected = set(packet)
    reviewer_a = validate_review(reviews["reviewer_a"], expected)
    reviewer_b = validate_review(reviews["reviewer_b"], expected)
    disputes = {
        opaque_id
        for opaque_id in expected
        if any(reviewer_a[opaque_id][field] != reviewer_b[opaque_id][field] for field in DECISION_FIELDS)
    }
    reviewer_c = load_reviewer_c(reviews["reviewer_c"], disputes)

    final: dict[str, dict[str, Any]] = {}
    for opaque_id in sorted(expected):
        row = {"opaque_quality_id": opaque_id}
        adjudicated_fields = []
        for field in DECISION_FIELDS:
            if reviewer_a[opaque_id][field] == reviewer_b[opaque_id][field]:
                row[field] = reviewer_a[opaque_id][field]
            else:
                row[field] = reviewer_c[opaque_id][field]
                adjudicated_fields.append(field)
        row["short_reason"] = reviewer_c[opaque_id]["short_reason"] if adjudicated_fields else reviewer_a[opaque_id]["short_reason"]
        row["adjudicated_fields"] = adjudicated_fields
        final[opaque_id] = row
    write_jsonl(reviews["final"], final.values())

    primary_ids = [row["opaque_quality_id"] for row in key_rows if row["is_hidden_duplicate"] == "False"]
    ordinal = {
        field: ordinal_metrics([(reviewer_a[opaque_id][field], reviewer_b[opaque_id][field]) for opaque_id in primary_ids])
        for field in ORDINAL_FIELDS
    }
    derived_pairs = [
        (quality_eligible(reviewer_a[opaque_id], endpoint), quality_eligible(reviewer_b[opaque_id], endpoint))
        for opaque_id in primary_ids
    ]
    natural_pairs = [(reviewer_a[opaque_id]["naturalness"], reviewer_b[opaque_id]["naturalness"]) for opaque_id in primary_ids]
    a_duplicate = duplicate_consistency(reviewer_a, key_rows)
    b_duplicate = duplicate_consistency(reviewer_b, key_rows)
    gates = {
        "detail_weighted_agreement_pass": ordinal["detail_retention"]["linear_weighted_agreement"] >= float(measurement["linear_weighted_agreement_min"]),
        "detail_ac2_pass": ordinal["detail_retention"]["gwet_ac2"] is not None and ordinal["detail_retention"]["gwet_ac2"] >= float(measurement["gwet_ac2_min"]),
        "detail_severe_disagreement_pass": ordinal["detail_retention"]["severe_disagreement_rate"] <= float(measurement["severe_ordinal_disagreement_rate_max"]),
        "actionability_weighted_agreement_pass": ordinal["safe_actionability"]["linear_weighted_agreement"] >= float(measurement["linear_weighted_agreement_min"]),
        "actionability_ac2_pass": ordinal["safe_actionability"]["gwet_ac2"] is not None and ordinal["safe_actionability"]["gwet_ac2"] >= float(measurement["gwet_ac2_min"]),
        "actionability_severe_disagreement_pass": ordinal["safe_actionability"]["severe_disagreement_rate"] <= float(measurement["severe_ordinal_disagreement_rate_max"]),
        "derived_quality_raw_pass": raw_agreement(derived_pairs) >= float(measurement["derived_quality_raw_agreement_min"]),
        "derived_quality_ac1_pass": gwet_ac1(derived_pairs, ["yes", "no"]) is not None and gwet_ac1(derived_pairs, ["yes", "no"]) >= float(measurement["derived_quality_gwet_ac1_min"]),
        "naturalness_raw_pass": raw_agreement(natural_pairs) >= float(measurement["naturalness_raw_agreement_min"]),
        "reviewer_a_duplicate_pass": a_duplicate["full_vector_consistent_count"] >= int(measurement["duplicate_full_vector_consistency_min_count"]),
        "reviewer_b_duplicate_pass": b_duplicate["full_vector_consistent_count"] >= int(measurement["duplicate_full_vector_consistency_min_count"]),
        "all_disagreements_adjudicated": set(reviewer_c) == disputes,
    }
    gates["overall_pass"] = all(gates.values())

    agreement = {
        "name": f"step3cm_{split}_review_agreement_v1",
        "measurement_split": split,
        "primary_response_rows": len(primary_ids),
        "hidden_duplicate_rows": len(key_rows) - len(primary_ids),
        "ordinal": ordinal,
        "derived_quality_eligibility": nominal_metrics(derived_pairs, ["yes", "no"]),
        "naturalness": nominal_metrics(natural_pairs, ["yes", "no"]),
        "reviewer_a_duplicate_consistency": a_duplicate,
        "reviewer_b_duplicate_consistency": b_duplicate,
        "disagreement_rows": len(disputes),
        "reviewer_c_rows": len(reviewer_c),
        "gates": gates,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(outputs["agreement"], agreement)

    if split == "development":
        next_action = "freeze_active_rubric_and_open_untouched_validation" if gates["overall_pass"] else "use_at_most_one_complete_development_only_revision_or_close_measurement_route"
    else:
        next_action = "authorize_joint_measurement_plus_c3_feasibility_decision" if gates["overall_pass"] else "close_decomposed_measurement_route_without_training"
    route = {
        "name": f"step3cm_{split}_route_decision_v1",
        "measurement_split": split,
        "overall_pass": gates["overall_pass"],
        "next_action": next_action,
        "training_authorized": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(outputs["route"], route)

    report = f"""# Step3C-M {split.title()} Measurement Report

Status: {'pass' if gates['overall_pass'] else 'fail'}.

| construct | exact/raw | weighted | chance-corrected | severe disagreement |
| --- | ---: | ---: | ---: | ---: |
| detail retention | {ordinal['detail_retention']['exact_raw_agreement']:.3f} | {ordinal['detail_retention']['linear_weighted_agreement']:.3f} | {ordinal['detail_retention']['gwet_ac2']} | {ordinal['detail_retention']['severe_disagreement_rate']:.3f} |
| safe actionability | {ordinal['safe_actionability']['exact_raw_agreement']:.3f} | {ordinal['safe_actionability']['linear_weighted_agreement']:.3f} | {ordinal['safe_actionability']['gwet_ac2']} | {ordinal['safe_actionability']['severe_disagreement_rate']:.3f} |
| derived quality eligibility | {agreement['derived_quality_eligibility']['raw_agreement']:.3f} | NA | {agreement['derived_quality_eligibility']['gwet_ac1']} | NA |
| naturalness | {agreement['naturalness']['raw_agreement']:.3f} | NA | {agreement['naturalness']['gwet_ac1']} | NA |

- Reviewer A duplicate full-vector consistency: {a_duplicate['full_vector_consistent_count']}/{a_duplicate['denominator']}
- Reviewer B duplicate full-vector consistency: {b_duplicate['full_vector_consistent_count']}/{b_duplicate['denominator']}
- Disagreement rows adjudicated: {len(reviewer_c)}/{len(disputes)}
- Overall gate: {'pass' if gates['overall_pass'] else 'fail'}
- Next action: `{next_action}`
- Training authorized: false
- Heldout opened: false
"""
    outputs["report"].write_text(report, encoding="utf-8", newline="\n")

    artifact_paths = [
        paths["packet"],
        paths["key"],
        paths["duplicates"],
        paths["reviewer_c_packet"],
        reviews["reviewer_a"],
        reviews["reviewer_b"],
        reviews["reviewer_c"],
        reviews["final"],
        reviews["ab_freeze"],
        outputs["agreement"],
        outputs["route"],
        outputs["report"],
    ]
    artifact_manifest = {
        "name": f"step3cm_{split}_review_artifact_manifest_v1",
        "measurement_split": split,
        "artifacts": [
            {"path": path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
            if path.exists()
        ],
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(outputs["manifest"], artifact_manifest)
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return route


def verify_protocol() -> dict[str, Any]:
    manifest = read_json(PROTOCOL_MANIFEST)
    checked = []
    mismatches = []
    for row in manifest["artifacts"]:
        path = Path(row["path"])
        exists = path.exists()
        actual = sha256(path) if exists else ""
        status = "match" if exists and actual == row["sha256"] else "missing" if not exists else "mismatch"
        checked.append({"path": str(path), "status": status})
        if status != "match":
            mismatches.append(str(path))
    result = {
        "name": "step3cm_protocol_verification_v1",
        "checked": len(checked),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(DIAGNOSTICS / "step3cm_protocol_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("test-fixtures")
    subparsers.add_parser("freeze-protocol")
    subparsers.add_parser("verify-protocol")
    subparsers.add_parser("freeze-active-rubric")
    for command in ("validate-ab", "prepare-adjudication", "summarize"):
        child = subparsers.add_parser(command)
        child.add_argument("--split", choices=["development", "validation"], required=True)
    args = parser.parse_args()

    if args.command == "test-fixtures":
        run_synthetic_tests()
    elif args.command == "freeze-protocol":
        if not SYNTHETIC_TESTS.exists() or not read_json(SYNTHETIC_TESTS).get("passed"):
            run_synthetic_tests()
        freeze_protocol()
    elif args.command == "verify-protocol":
        verify_protocol()
    elif args.command == "freeze-active-rubric":
        freeze_active_rubric()
    elif args.command == "validate-ab":
        validate_ab(args.split)
    elif args.command == "prepare-adjudication":
        prepare_adjudication(args.split)
    elif args.command == "summarize":
        summarize(args.split)


if __name__ == "__main__":
    main()
