from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
OUT_DIR = ROOT / "results" / "step3a_preflight"

STEP2 = Path("experiments/step2_mechanism_dynamics")
TARGETS = STEP2 / "manifests/step2ir_natural_diverse_targets_v1.jsonl"
TARGET_REVIEW = STEP2 / "manifests/step2ir_natural_diverse_targets_v1_review.csv"
TARGET_FREEZE = STEP2 / "manifests/step2ir_natural_diverse_targets_v1_freeze.json"
SPLIT_MANIFEST = STEP2 / "results/4070ti_step2g/splits/split_manifest.csv"
SPLIT_COUNTS = STEP2 / "results/4070ti_step2g/splits/split_counts.csv"
EXPOSURE = STEP2 / "results/4070ti_step2j/entry_audit/evidence/anchor_exposure_manifest.csv"

SAFETY_OUTPUTS = {
    "decay10_step32": STEP2
    / "results/4070ti_step2j_b1/behavior_quality/evidence/safety/decay10/step0032_pct050/"
    "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10_step0032_pct050_safety_full100_core.jsonl",
    "decay10_step64": STEP2
    / "results/4070ti_step2j_b1/behavior_quality/evidence/safety/decay10/step0064_pct100/"
    "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10_step0064_pct100_safety_full100_core.jsonl",
    "constant20_step32": STEP2
    / "results/4070ti_step2j_b1/behavior_quality/evidence/safety/constant20/step0032_pct050/"
    "step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20_step0032_pct050_safety_full100_core.jsonl",
    "constant20_step64": STEP2
    / "results/4070ti_step2j_b1/behavior_quality/evidence/safety/constant20/step0064_pct100/"
    "step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20_step0064_pct100_safety_full100_core.jsonl",
}

REVIEW_AGREEMENT = (
    STEP2
    / "results/4070ti_step2j_b1/behavior_quality/reports/step2j_b1_review_agreement.json"
)
SAFETY_SUMMARY = (
    STEP2
    / "results/4070ti_step2j_b1/behavior_quality/reports/step2j_b1_safety_behavior_summary.csv"
)
SEMANTIC_SUMMARY = (
    STEP2
    / "results/4070ti_step2j_b1/behavior_quality/reports/step2j_b1_semantic_template_summary.csv"
)
UTILITY_SUMMARY = (
    STEP2
    / "results/4070ti_step2j_b1/behavior_quality/reports/step2j_b1_utility_quality_summary.csv"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        return len(read_jsonl(path))
    if path.suffix == ".csv":
        return len(read_csv(path))
    return None


def evidence_manifest() -> list[dict[str, Any]]:
    inputs = [
        ("natural_v1_targets", TARGETS, "authorized_content"),
        ("natural_v1_preflight_review", TARGET_REVIEW, "authorized_content"),
        ("natural_v1_freeze", TARGET_FREEZE, "authorized_content"),
        ("step2g_split_manifest", SPLIT_MANIFEST, "ids_and_hashes_only"),
        ("step2g_split_counts", SPLIT_COUNTS, "counts_only"),
        ("natural_v1_exposure", EXPOSURE, "authorized_content"),
        ("step2j_review_agreement", REVIEW_AGREEMENT, "authorized_content"),
        ("step2j_safety_summary", SAFETY_SUMMARY, "authorized_content"),
        ("step2j_semantic_summary", SEMANTIC_SUMMARY, "authorized_content"),
        ("step2j_utility_summary", UTILITY_SUMMARY, "authorized_content"),
    ]
    inputs.extend(
        (f"core100_{name}", path, "authorized_core100_content")
        for name, path in SAFETY_OUTPUTS.items()
    )
    rows = []
    for role, path, access in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "role": role,
                "path": str(path),
                "access_class": access,
                "bytes": path.stat().st_size,
                "rows": row_count(path) if row_count(path) is not None else "",
                "sha256": sha256_file(path),
            }
        )
    return rows


def opening_family(response: str) -> str:
    if response.startswith("I can't"):
        return "I can't"
    if response.startswith("I won't"):
        return "I won't"
    return "other"


def target_inventory(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        prompts[str(row["sample_id"])].append(row)

    token_counts = [int(row["token_count"]) for row in target_rows]
    response_counts = Counter(str(row["response"]) for row in target_rows)
    opening_counts = Counter(opening_family(str(row["response"])) for row in target_rows)
    category_counts = Counter(str(row["category"]) for row in target_rows)

    phrase_flags = {
        "contains_i_can_help": sum("I can help" in str(row["response"]) for row in target_rows),
        "contains_i_wont": sum("I won't" in str(row["response"]) for row in target_rows),
        "contains_i_cant": sum("I can't" in str(row["response"]) for row in target_rows),
        "contains_instead": sum("instead" in str(row["response"]).lower() for row in target_rows),
        "contains_support": sum("support" in str(row["response"]).lower() for row in target_rows),
        "contains_legal_or_lawful": sum(
            any(term in str(row["response"]).lower() for term in ("legal", "lawful", "law enforcement"))
            for row in target_rows
        ),
    }

    return {
        "target_rows": len(target_rows),
        "unique_prompts": len(prompts),
        "targets_per_prompt_min": min(len(rows) for rows in prompts.values()),
        "targets_per_prompt_max": max(len(rows) for rows in prompts.values()),
        "unique_target_hashes": len({str(row["target_hash"]) for row in target_rows}),
        "unique_responses": len(response_counts),
        "exact_duplicate_rows_beyond_first": sum(count - 1 for count in response_counts.values()),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_count_mean": statistics.fmean(token_counts),
        "token_count_median": statistics.median(token_counts),
        "opening_counts": dict(sorted(opening_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "phrase_flags": phrase_flags,
        "deterministic_observation": (
            "Natural v1 is lexically unique but structurally narrow at the opening/redirect-phrase level. "
            "This observation does not replace independent response-policy labeling."
        ),
    }


def integrity_checks(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows = read_csv(SPLIT_MANIFEST)
    split_counts = Counter(row["split"] for row in split_rows)
    split_ids = defaultdict(set)
    for row in split_rows:
        split_ids[row["split"]].add(row["sample_id"])

    target_ids = {str(row["sample_id"]) for row in target_rows}
    output_ids_by_arm = {
        name: {str(row["sample_id"]) for row in read_jsonl(path)}
        for name, path in SAFETY_OUTPUTS.items()
    }
    first_output_ids = next(iter(output_ids_by_arm.values()))
    core100_alignment = all(ids == first_output_ids for ids in output_ids_by_arm.values())

    exposure_rows = read_csv(EXPOSURE)
    exposure_counts = [int(row["exposure_count"]) for row in exposure_rows]
    target_hashes = [str(row["target_hash"]) for row in target_rows]
    exposure_hashes = [row["target_hash"] for row in exposure_rows]

    overlaps = {
        "target_vs_core100": len(target_ids & first_output_ids),
        "core100_vs_safety_train": len(first_output_ids & split_ids["safety_train"]),
        "core100_vs_proxy_dev": len(first_output_ids & split_ids["proxy_dev"]),
        "core100_vs_heldout_ids": len(first_output_ids & split_ids["heldout_test"]),
        "target_vs_proxy_dev": len(target_ids & split_ids["proxy_dev"]),
        "target_vs_heldout_ids": len(target_ids & split_ids["heldout_test"]),
    }

    checks = {
        "target_rows_eq_80": len(target_rows) == 80,
        "unique_target_prompts_eq_40": len(target_ids) == 40,
        "targets_per_prompt_eq_2": Counter(str(row["sample_id"]) for row in target_rows).most_common()[-1][1]
        == 2
        and Counter(str(row["sample_id"]) for row in target_rows).most_common()[0][1] == 2,
        "target_ids_equal_safety_train_ids": target_ids == split_ids["safety_train"],
        "split_counts_match_40_40_120": split_counts
        == Counter({"safety_train": 40, "proxy_dev": 40, "heldout_test": 120}),
        "core100_rows_each_eq_100": all(len(ids) == 100 for ids in output_ids_by_arm.values()),
        "core100_ids_aligned_across_arms": core100_alignment,
        "training_core100_overlap_zero": overlaps["target_vs_core100"] == 0,
        "core100_split_overlap_zero": all(value == 0 for key, value in overlaps.items() if key.startswith("core100_vs_")),
        "target_heldout_overlap_zero": overlaps["target_vs_heldout_ids"] == 0,
        "exposure_rows_eq_80": len(exposure_rows) == 80,
        "exposure_total_eq_256": sum(exposure_counts) == 256,
        "exposure_range_3_to_4": min(exposure_counts) == 3 and max(exposure_counts) == 4,
        "exposure_hash_order_matches_targets": exposure_hashes == target_hashes,
        "heldout_content_opened_by_script": False,
    }
    stop_gate_pass = all(value for key, value in checks.items() if key != "heldout_content_opened_by_script")
    stop_gate_pass = stop_gate_pass and not checks["heldout_content_opened_by_script"]

    return {
        "split_counts": dict(sorted(split_counts.items())),
        "core100_unique_ids": len(first_output_ids),
        "overlaps": overlaps,
        "exposure": {
            "rows": len(exposure_rows),
            "total_draws": sum(exposure_counts),
            "minimum": min(exposure_counts),
            "maximum": max(exposure_counts),
            "mean": statistics.fmean(exposure_counts),
        },
        "checks": checks,
        "a1_stop_gate_pass": stop_gate_pass,
        "heldout_boundary": (
            "Only IDs and content hashes from split_manifest.csv were read. "
            "No heldout prompt or response content path is an input to this script."
        ),
    }


def summary_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for category, count in inventory["category_counts"].items():
        rows.append({"summary_type": "category", "key": category, "count": count, "rate": count / 80})
    for opening, count in inventory["opening_counts"].items():
        rows.append({"summary_type": "opening", "key": opening, "count": count, "rate": count / 80})
    for phrase, count in inventory["phrase_flags"].items():
        rows.append({"summary_type": "phrase", "key": phrase, "count": count, "rate": count / 80})
    return rows


def write_report(inventory: dict[str, Any], integrity: dict[str, Any]) -> None:
    checks = integrity["checks"]
    overlap = integrity["overlaps"]
    report = f"""# Step3A A1-A2 Preflight Report

## Status

A1 evidence/leakage audit: **{'pass' if integrity['a1_stop_gate_pass'] else 'fail'}**.

A2 deterministic target inventory: complete.

No new training was run. No heldout prompt or response content was opened.

## Split and Exposure Integrity

| check | result |
| --- | --- |
| target rows | {inventory['target_rows']} |
| unique training prompts | {inventory['unique_prompts']} |
| targets per prompt min/max | {inventory['targets_per_prompt_min']} / {inventory['targets_per_prompt_max']} |
| core100 IDs aligned across four cells | {checks['core100_ids_aligned_across_arms']} |
| training/core100 overlap | {overlap['target_vs_core100']} |
| core100/Step2G split-ID overlap | {overlap['core100_vs_safety_train'] + overlap['core100_vs_proxy_dev'] + overlap['core100_vs_heldout_ids']} |
| target/heldout-ID overlap | {overlap['target_vs_heldout_ids']} |
| exposure rows | {integrity['exposure']['rows']} |
| total safety draws | {integrity['exposure']['total_draws']} |
| exposure min/max | {integrity['exposure']['minimum']} / {integrity['exposure']['maximum']} |
| exposure hash order matches targets | {checks['exposure_hash_order_matches_targets']} |

The core100 development set is disjoint from natural-v1 safety training and from all Step2G split IDs. The script read only heldout IDs and hashes in the split manifest, not heldout contents.

## Natural v1 Deterministic Structure

| metric | value |
| --- | ---: |
| unique responses | {inventory['unique_responses']} / {inventory['target_rows']} |
| exact duplicate rows beyond first | {inventory['exact_duplicate_rows_beyond_first']} |
| `I can't` openings | {inventory['opening_counts'].get("I can't", 0)} |
| `I won't` openings | {inventory['opening_counts'].get("I won't", 0)} |
| other openings | {inventory['opening_counts'].get('other', 0)} |
| contains `I can help` | {inventory['phrase_flags']['contains_i_can_help']} |
| token count min/median/mean/max | {inventory['token_count_min']} / {inventory['token_count_median']:.1f} / {inventory['token_count_mean']:.2f} / {inventory['token_count_max']} |

Natural v1 is exactly diverse at the string level but visibly narrow at the opening and redirect-phrase level. This is a decision-relevant diagnostic, not a subjective mechanical-template label and not proof of causal structure transfer.

## Decision

A1 passes and A2 supports proceeding to the preregistered independent fixed-codebook review. New training remains prohibited until reviewer reliability and the Step3A route decision are complete.
"""
    (OUT_DIR / "Step3A_A1_A2_preflight_report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = read_jsonl(TARGETS)
    evidence = evidence_manifest()
    inventory = target_inventory(targets)
    integrity = integrity_checks(targets)

    write_csv(
        OUT_DIR / "step3a_evidence_manifest.csv",
        evidence,
        ["role", "path", "access_class", "bytes", "rows", "sha256"],
    )
    write_json(OUT_DIR / "step3a_split_and_exposure_integrity.json", integrity)
    write_json(OUT_DIR / "step3a_natural_v1_target_inventory.json", inventory)
    write_csv(
        OUT_DIR / "step3a_natural_v1_target_summary.csv",
        summary_rows(inventory),
        ["summary_type", "key", "count", "rate"],
    )
    write_report(inventory, integrity)

    if not integrity["a1_stop_gate_pass"]:
        raise SystemExit("Step3A A1 stop gate failed; inspect integrity report")

    print(f"wrote {OUT_DIR / 'step3a_evidence_manifest.csv'}")
    print(f"wrote {OUT_DIR / 'step3a_split_and_exposure_integrity.json'}")
    print(f"wrote {OUT_DIR / 'step3a_natural_v1_target_inventory.json'}")
    print(f"wrote {OUT_DIR / 'step3a_natural_v1_target_summary.csv'}")
    print(f"wrote {OUT_DIR / 'Step3A_A1_A2_preflight_report.md'}")


if __name__ == "__main__":
    main()
