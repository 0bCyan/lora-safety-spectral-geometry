from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
PROTOCOL = ROOT / "results/step3cm_protocol"
PROMPT_COVERAGE = ROOT / "results/step3c_preflight/step3c_prompt_positive_coverage.csv"
OUT = ROOT / "results/step3n_r1_reaudit"
SOURCE_ROLES = {"existing_natural_v1_target", "prospective_positive_candidate"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def minimum_fallback_edges(singleton_count: int) -> int:
    if singleton_count == 0:
        return 0
    if singleton_count == 1:
        return 2
    return singleton_count


def main() -> None:
    prompt_rows = read_csv(PROMPT_COVERAGE)
    category_by_prompt = {row["sample_id"]: row["category"] for row in prompt_rows}
    if len(category_by_prompt) != 40:
        raise ValueError("expected exactly 40 safety-training prompts")

    source_files = [PROMPT_COVERAGE]
    candidates: list[dict[str, Any]] = []
    for split in ("development", "validation"):
        key_path = PROTOCOL / f"packets/{split}/step3cm_{split}_hidden_key.csv"
        label_path = PROTOCOL / f"reviews/{split}/step3cm_{split}_final_labels.jsonl"
        source_files.extend([key_path, label_path])

        keys = read_csv(key_path)
        key_by_id = {row["opaque_quality_id"]: row for row in keys}
        if len(key_by_id) != len(keys):
            raise ValueError(f"duplicate hidden-key IDs in {key_path}")

        labels = read_jsonl(label_path)
        label_ids = [row["opaque_quality_id"] for row in labels]
        if len(set(label_ids)) != len(label_ids):
            raise ValueError(f"duplicate final-label IDs in {label_path}")
        unknown = sorted(set(label_ids) - set(key_by_id))
        if unknown:
            raise ValueError(f"final labels missing from hidden key in {split}: {unknown[:3]}")

        for label in labels:
            key = key_by_id[label["opaque_quality_id"]]
            if key["is_hidden_duplicate"].lower() != "false":
                continue
            if key["role"] not in SOURCE_ROLES:
                continue
            if key["sample_id"] not in category_by_prompt:
                raise ValueError(f"unexpected prompt ID: {key['sample_id']}")
            if key["category"] != category_by_prompt[key["sample_id"]]:
                raise ValueError(f"category mismatch for {key['sample_id']}")
            candidates.append(
                {
                    "sample_id": key["sample_id"],
                    "category": key["category"],
                    "role": key["role"],
                    "response_id": key["source_response_id"],
                    "detail_retention": int(label["detail_retention"]),
                    "safe_actionability": int(label["safe_actionability"]),
                    "naturalness": label["naturalness"],
                }
            )

    source_keys = [(row["sample_id"], row["response_id"]) for row in candidates]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("duplicate source response after split join")
    role_counts = Counter(row["role"] for row in candidates)
    expected_role_counts = Counter(
        {"existing_natural_v1_target": 80, "prospective_positive_candidate": 7}
    )
    if role_counts != expected_role_counts:
        raise ValueError(f"unexpected source-role counts: {dict(role_counts)}")

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        eligible = (
            row["detail_retention"] == 2
            and row["safe_actionability"] == 2
            and row["naturalness"] == "yes"
        )
        if eligible:
            by_prompt[row["sample_id"]].append(row)

    output_rows: list[dict[str, Any]] = []
    for sample_id in sorted(category_by_prompt):
        rows = sorted(by_prompt[sample_id], key=lambda row: row["response_id"])
        output_rows.append(
            {
                "sample_id": sample_id,
                "category": category_by_prompt[sample_id],
                "eligible_response_count": len(rows),
                "eligible_natural_count": sum(
                    row["role"] == "existing_natural_v1_target" for row in rows
                ),
                "eligible_prospective_count": sum(
                    row["role"] == "prospective_positive_candidate" for row in rows
                ),
                "quality_upper_bound_available": len(rows) > 0,
                "eligible_response_ids": "|".join(row["response_id"] for row in rows),
            }
        )

    eligible_prompts = [row for row in output_rows if row["quality_upper_bound_available"]]
    missing_prompts = [row["sample_id"] for row in output_rows if not row["quality_upper_bound_available"]]
    category_counts = Counter(row["category"] for row in eligible_prompts)
    exact_category_eligible = sum(count for count in category_counts.values() if count >= 2)
    singleton_categories = sorted(category for category, count in category_counts.items() if count == 1)

    summary = {
        "name": "step3n_r1_source_quality_upper_bound_v1",
        "definition": {
            "detail_retention": 2,
            "safe_actionability": 2,
            "naturalness": "yes",
            "clean_safe_included": False,
            "interpretation": "descriptive quality-only upper bound; not N0-R1 qualification",
        },
        "source_roles": sorted(SOURCE_ROLES),
        "source_role_counts": dict(sorted(role_counts.items())),
        "source_hashes": {path.as_posix(): sha256(path) for path in source_files},
        "candidate_responses_reviewed": len(candidates),
        "eligible_responses": sum(int(row["eligible_response_count"]) for row in output_rows),
        "training_prompts": len(output_rows),
        "eligible_prompts": len(eligible_prompts),
        "missing_prompts": missing_prompts,
        "category_counts": dict(sorted(category_counts.items())),
        "singleton_categories": singleton_categories,
        "exact_category_only_maximum": exact_category_eligible,
        "minimum_category_fallback_edges_for_global_derangement": minimum_fallback_edges(
            len(singleton_categories)
        ),
        "pcsr_tuple_gate": {
            "required": 36,
            "quality_upper_bound_pass": len(eligible_prompts) >= 36,
            "slack_before_fresh_clean_safe_review": len(eligible_prompts) - 36,
            "status": "pending fresh arm-blind N0-R1 qualification",
        },
        "training_authorized": False,
        "heldout_opened": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "step3n_r1_source_quality_upper_bound.csv",
        output_rows,
        [
            "sample_id",
            "category",
            "eligible_response_count",
            "eligible_natural_count",
            "eligible_prospective_count",
            "quality_upper_bound_available",
            "eligible_response_ids",
        ],
    )
    (OUT / "step3n_r1_source_quality_upper_bound.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
