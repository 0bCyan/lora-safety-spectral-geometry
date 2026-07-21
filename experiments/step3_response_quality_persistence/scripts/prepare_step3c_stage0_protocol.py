from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
REGISTRY = ROOT / "configs/step3c_stage0_registry.json"

TRIPLET_SCHEMA_FIELDS = [
    "triplet_id",
    "sample_id",
    "category",
    "positive_response_id",
    "negative_response_id",
    "positive_source",
    "negative_source",
    "positive_sha256",
    "negative_sha256",
    "positive_token_length",
    "negative_token_length",
    "policy_review_status",
    "relevance_review_status",
    "mechanical_review_status",
    "adjudication_status",
]

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

STRUCTURE_FIELDS = [
    "opaque_response_id",
    "boundary_act",
    "rationale_act",
    "redirect_act",
    "harmful_procedural_tail",
    "mechanical_template",
    "short_reason",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_dir(registry: dict[str, Any]) -> Path:
    return Path(registry["output_dir"])


def load_inputs(registry: dict[str, Any]) -> dict[str, Any]:
    eval_rows = read_jsonl(Path(registry["eval_safety_path"]))
    split_rows = read_csv(Path(registry["split_manifest"]))
    target_rows = read_jsonl(Path(registry["natural_v1_targets"]))
    target_key = read_csv(Path(registry["step3a_target_key"]))
    target_labels = read_jsonl(Path(registry["step3a_target_final_labels"]))
    coverage = read_csv(Path(registry["positive_coverage_csv"]))
    return {
        "eval_rows": eval_rows,
        "split_rows": split_rows,
        "target_rows": target_rows,
        "target_key": target_key,
        "target_labels": target_labels,
        "coverage": coverage,
    }


def validate_inputs(registry: dict[str, Any]) -> dict[str, Any]:
    data = load_inputs(registry)
    eval_by_id = {row["sample_id"]: row for row in data["eval_rows"]}
    split_by_name: dict[str, set[str]] = defaultdict(set)
    for row in data["split_rows"]:
        split_by_name[row["split"]].add(row["sample_id"])
    safety_train = sorted(split_by_name["safety_train"])
    core100 = {row["sample_id"] for row in data["eval_rows"][:100]}
    heldout = split_by_name["heldout_test"]
    proxy_dev = split_by_name["proxy_dev"]
    if len(safety_train) != int(registry["stage0_triplet_gates"]["safety_train_prompt_count"]):
        raise ValueError(f"expected 40 safety_train IDs, found {len(safety_train)}")
    if len(set(safety_train)) != len(safety_train):
        raise ValueError("duplicate safety_train IDs")
    if set(safety_train) & core100:
        raise ValueError("safety_train overlaps core100")
    if set(safety_train) & heldout:
        raise ValueError("safety_train overlaps heldout")
    if not set(safety_train).issubset(eval_by_id):
        raise ValueError("some safety_train IDs are missing from eval_safety")

    targets_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["target_rows"]:
        if row.get("step2g_split") == "safety_train":
            targets_by_prompt[row["sample_id"]].append(row)
    if set(targets_by_prompt) != set(safety_train):
        raise ValueError("natural_v1 targets do not exactly cover safety_train")
    bad_counts = {sid: len(rows) for sid, rows in targets_by_prompt.items() if len(rows) != 2}
    if bad_counts:
        raise ValueError(f"expected two natural_v1 targets per prompt: {bad_counts}")
    if any(row.get("reviewer_decision") != "approve" for rows in targets_by_prompt.values() for row in rows):
        raise ValueError("all natural_v1 targets must already be approved")

    key_ids = {row["opaque_target_id"] for row in data["target_key"]}
    label_ids = {row["opaque_target_id"] for row in data["target_labels"]}
    if key_ids != label_ids:
        raise ValueError("Step3A target key/final label mismatch")
    if len(key_ids) != 80:
        raise ValueError("expected 80 Step3A target labels")

    coverage_by_id = {row["sample_id"]: row for row in data["coverage"]}
    if set(coverage_by_id) != set(safety_train):
        raise ValueError("positive coverage CSV does not exactly cover safety_train")
    missing_positive = sorted(row["sample_id"] for row in data["coverage"] if row["requires_targeted_positive"] == "True")
    if len(missing_positive) != int(registry["stage0_triplet_gates"]["new_positive_candidate_count"]):
        raise ValueError(f"expected seven missing positives, found {len(missing_positive)}")
    existing_positive_count = sum(row["existing_positive_available"] == "True" for row in data["coverage"])

    category_counts = Counter(eval_by_id[sid].get("category", targets_by_prompt[sid][0]["category"]) for sid in safety_train)
    return {
        "safety_train_ids": safety_train,
        "safety_train_count": len(safety_train),
        "core100_overlap": len(set(safety_train) & core100),
        "heldout_overlap": len(set(safety_train) & heldout),
        "proxy_dev_overlap": len(set(safety_train) & proxy_dev),
        "natural_v1_target_rows": sum(len(rows) for rows in targets_by_prompt.values()),
        "targets_per_prompt": 2,
        "step3a_target_labels": len(label_ids),
        "prompts_with_existing_nonmechanical_positive": existing_positive_count,
        "prompts_requiring_targeted_positive": missing_positive,
        "category_counts": dict(sorted(category_counts.items())),
        "heldout_opened": False,
        "training_authorized": False,
    }


def write_protocol_artifacts(registry: dict[str, Any], validation: dict[str, Any]) -> None:
    out = output_dir(registry)
    reports = out / "reports"
    packets = out / "packets"
    diagnostics = out / "diagnostics"
    reports.mkdir(parents=True, exist_ok=True)
    packets.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)

    schema = {
        "name": "step3c_stage0_triplet_schema_v1",
        "fields": TRIPLET_SCHEMA_FIELDS,
        "required_fields": TRIPLET_SCHEMA_FIELDS,
        "selection_rules": {
            "positive": "prefer existing approved non-mechanical target; if two qualify, lower opaque target ID; new positive only for seven predeclared missing prompts after protocol freeze",
            "negative": "single deterministic constant20 safety_train greedy generation, if independently clean-safe prompt-relevant valid mechanical",
            "forbidden_selection_features": ["model_margin", "response_length", "core100_behavior", "heldout_content"],
        },
    }
    schema_path = reports / "step3c_stage0_triplet_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_protocol = {
        "name": "step3c_stage0_review_protocol_v1",
        "behavior_packet_schema": BEHAVIOR_FIELDS,
        "structure_packet_schema": STRUCTURE_FIELDS,
        "reviewer_a": "labels every row independently",
        "reviewer_b": "labels every row independently",
        "reviewer_c": "labels only decision-field disagreements",
        "key_join_timing": "only after A/B/C files are frozen and hashed",
        "reliability_gates": registry["stage0_review_gates"],
        "one_revision_stop_rule": "one codebook clarification and complete rerun allowed; second failure stops Stage 0",
        "main_session_labeling": "forbidden",
    }
    protocol_path = reports / "step3c_stage0_review_protocol.json"
    protocol_path.write_text(json.dumps(review_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inventory_rows = []
    data = load_inputs(registry)
    eval_by_id = {row["sample_id"]: row for row in data["eval_rows"]}
    coverage_by_id = {row["sample_id"]: row for row in data["coverage"]}
    for sample_id in validation["safety_train_ids"]:
        coverage = coverage_by_id[sample_id]
        inventory_rows.append(
            {
                "sample_id": sample_id,
                "category": eval_by_id[sample_id].get("category", coverage["category"]),
                "requires_targeted_positive": coverage["requires_targeted_positive"],
                "existing_nonmechanical_target_ids": coverage["nonmechanical_target_ids"],
                "source_role": "safety_train",
                "core100_overlap": False,
                "heldout_overlap": False,
            }
        )
    write_csv(
        reports / "step3c_stage0_source_inventory.csv",
        inventory_rows,
        [
            "sample_id",
            "category",
            "requires_targeted_positive",
            "existing_nonmechanical_target_ids",
            "source_role",
            "core100_overlap",
            "heldout_overlap",
        ],
    )

    validation_path = diagnostics / "step3c_stage0_input_boundary_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifact_paths = [
        REGISTRY,
        ROOT / "review_rubrics/step3c_triplet_behavior_rubric.md",
        ROOT / "review_rubrics/step3c_triplet_structure_codebook.md",
        schema_path,
        protocol_path,
        reports / "step3c_stage0_source_inventory.csv",
        validation_path,
    ]
    manifest = {
        "name": "step3c_stage0_protocol_freeze_manifest",
        "status": "frozen_before_generation",
        "head_record_required_before_generation": True,
        "generation_started": False,
        "new_positive_candidates_written": False,
        "heldout_opened": False,
        "training_authorized": False,
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
    }
    manifest_path = reports / "step3c_stage0_protocol_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# Step3C Stage 0A Protocol Freeze Report

Status: frozen before model generation, new positive writing, or reviewer labeling.

## Boundary Validation

| check | value |
| --- | ---: |
| safety_train IDs | {validation['safety_train_count']} |
| natural-v1 target rows | {validation['natural_v1_target_rows']} |
| targets per prompt | {validation['targets_per_prompt']} |
| core100 overlap | {validation['core100_overlap']} |
| proxy_dev overlap | {validation['proxy_dev_overlap']} |
| heldout overlap | {validation['heldout_overlap']} |
| prompts with existing non-mechanical positive | {validation['prompts_with_existing_nonmechanical_positive']} |
| prompts requiring targeted positive | {len(validation['prompts_requiring_targeted_positive'])} |

Prompts requiring exactly one prospective positive after this freeze:

```text
{', '.join(validation['prompts_requiring_targeted_positive'])}
```

## Frozen Protocol

- Triplet schema: `step3c_stage0_triplet_schema.json`
- Review protocol: `step3c_stage0_review_protocol.json`
- Behavior rubric: `review_rubrics/step3c_triplet_behavior_rubric.md`
- Structure codebook: `review_rubrics/step3c_triplet_structure_codebook.md`

No optimizer steps, model generation, target additions, core100 triplet use, heldout access, or training are authorized by this freeze.
"""
    (reports / "Step3C_Stage0A_protocol_freeze_report.md").write_text(report, encoding="utf-8", newline="\n")


def validate_triplet_row(row: dict[str, Any]) -> None:
    if set(row) != set(TRIPLET_SCHEMA_FIELDS):
        raise ValueError(f"bad triplet fields: {sorted(row)}")
    if not row["triplet_id"] or not row["sample_id"]:
        raise ValueError("triplet_id and sample_id are required")
    for field in ["positive_sha256", "negative_sha256"]:
        if len(str(row[field])) != 64:
            raise ValueError(f"{field} must be a sha256 hex digest")
    for field in ["positive_token_length", "negative_token_length"]:
        if int(row[field]) <= 0:
            raise ValueError(f"{field} must be positive")


def run_synthetic_tests() -> dict[str, Any]:
    synthetic = {
        "triplet_id": "T3C_SYN_0001",
        "sample_id": "eval_safety_999999",
        "category": "synthetic",
        "positive_response_id": "pos_syn_1",
        "negative_response_id": "neg_syn_1",
        "positive_source": "synthetic_positive",
        "negative_source": "synthetic_negative",
        "positive_sha256": sha256_text("positive"),
        "negative_sha256": sha256_text("negative"),
        "positive_token_length": 8,
        "negative_token_length": 7,
        "policy_review_status": "pending",
        "relevance_review_status": "pending",
        "mechanical_review_status": "pending",
        "adjudication_status": "pending",
    }
    validate_triplet_row(synthetic)
    bad = dict(synthetic)
    bad.pop("negative_sha256")
    failed = False
    try:
        validate_triplet_row(bad)
    except ValueError:
        failed = True
    if not failed:
        raise ValueError("synthetic negative control failed")
    return {"synthetic_valid_triplet_passed": True, "synthetic_missing_field_rejected": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate-inputs", "test-fixtures", "freeze-protocol"])
    args = parser.parse_args()
    registry = read_json(REGISTRY)
    if registry.get("heldout_opened") is not False or registry.get("training_authorized") is not False:
        raise ValueError("Stage0 registry must keep heldout_opened=false and training_authorized=false")

    if args.command == "validate-inputs":
        print(json.dumps(validate_inputs(registry), ensure_ascii=False, indent=2))
    elif args.command == "test-fixtures":
        print(json.dumps(run_synthetic_tests(), ensure_ascii=False, indent=2))
    elif args.command == "freeze-protocol":
        validation = validate_inputs(registry)
        tests = run_synthetic_tests()
        validation["synthetic_tests"] = tests
        write_protocol_artifacts(registry, validation)
        print(json.dumps({"status": "protocol_frozen", **tests}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
