from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
BASE = ROOT / "results/step3a_target_structure_review"
KEY = BASE / "keys/step3a_target_structure_key.csv"
LABELS = BASE / "reviews/step3a_target_structure_final_labels.jsonl"
OUT = ROOT / "results/step3c_preflight"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    keys = read_csv(KEY)
    labels = {row["opaque_target_id"]: row for row in read_jsonl(LABELS)}
    if set(labels) != {row["opaque_target_id"] for row in keys}:
        raise ValueError("target key and final-label IDs do not match")

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keys:
        label = labels[row["opaque_target_id"]]
        by_prompt[row["sample_id"]].append(
            {
                "target_id": row["target_id"],
                "category": row["category"],
                "mechanical_template": label["mechanical_template"],
                "redirect_act": label["redirect_act"],
            }
        )
    if len(by_prompt) != 40 or any(len(rows) != 2 for rows in by_prompt.values()):
        raise ValueError("expected 40 prompts with exactly two targets each")

    prompt_rows = []
    for sample_id, rows in sorted(by_prompt.items()):
        nonmechanical = [row for row in rows if row["mechanical_template"] == "no"]
        prompt_rows.append(
            {
                "sample_id": sample_id,
                "category": rows[0]["category"],
                "nonmechanical_target_count": len(nonmechanical),
                "existing_positive_available": len(nonmechanical) >= 1,
                "requires_targeted_positive": len(nonmechanical) == 0,
                "nonmechanical_target_ids": "|".join(row["target_id"] for row in nonmechanical),
            }
        )

    count_distribution = Counter(int(row["nonmechanical_target_count"]) for row in prompt_rows)
    usable = sum(bool(row["existing_positive_available"]) for row in prompt_rows)
    missing = sum(bool(row["requires_targeted_positive"]) for row in prompt_rows)
    summary = {
        "prompts": len(prompt_rows),
        "targets_per_prompt": 2,
        "nonmechanical_target_count_distribution": {
            str(key): count for key, count in sorted(count_distribution.items())
        },
        "prompts_with_existing_nonmechanical_positive": usable,
        "existing_positive_coverage": usable / len(prompt_rows),
        "prompts_requiring_targeted_positive": missing,
        "C2_preliminary_positive_coverage_gate": {
            "minimum": 0.80,
            "observed": usable / len(prompt_rows),
            "pass": usable / len(prompt_rows) >= 0.80,
        },
        "C2_final_triplet_gate": {
            "valid_triplets_required": 36,
            "status": "pending targeted positives and safe mechanical negatives"
        },
        "C3_existing_safe_target_coverage": 1.0,
        "heldout_opened": False,
        "training_authorized": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step3c_candidate_data_feasibility.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(
        OUT / "step3c_prompt_positive_coverage.csv",
        prompt_rows,
        [
            "sample_id",
            "category",
            "nonmechanical_target_count",
            "existing_positive_available",
            "requires_targeted_positive",
            "nonmechanical_target_ids",
        ],
    )
    report = f"""# Step3C Candidate-Data Feasibility

## Result

| metric | value |
| --- | ---: |
| safety-training prompts | {len(prompt_rows)} |
| prompts with two non-mechanical targets | {count_distribution[2]} |
| prompts with one non-mechanical target | {count_distribution[1]} |
| prompts with no non-mechanical target | {count_distribution[0]} |
| existing positive coverage | {usable / len(prompt_rows):.3f} |

The preliminary C2 positive-coverage gate (`>=0.80`) passes. Seven prompts need one targeted non-mechanical positive. This does not authorize writing those targets or training C2 until the triplet rubric and Step3B route are frozen.

C3 can use the existing independently approved safe targets for all 40 prompts, subject to representation and queue feasibility.
"""
    (OUT / "Step3C_candidate_data_feasibility_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
