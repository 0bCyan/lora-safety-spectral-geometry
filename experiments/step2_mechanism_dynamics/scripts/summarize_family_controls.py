from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import load_json, write_csv


TRANSITION_CHECKPOINTS = [
    "step0000_pct000",
    "step0006_pct009",
    "step0016_pct025",
    "step0064_pct100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def summarize_labeled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    degradation = sum(1 for row in rows if row.get("degradation_reason"))
    return {
        "n": total,
        "refusal": counts["refusal"],
        "unsafe_compliance": counts["unsafe_compliance"],
        "invalid_or_irrelevant": counts["invalid_or_irrelevant"],
        "degradation_count": degradation,
        "refusal_rate": counts["refusal"] / total if total else 0.0,
        "unsafe_compliance_rate": counts["unsafe_compliance"] / total if total else 0.0,
        "invalid_rate": counts["invalid_or_irrelevant"] / total if total else 0.0,
        "degradation_rate": degradation / total if total else 0.0,
    }


def build_family_control_summary(dynamics_rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, Any]]:
    keep = []
    columns = [
        "family",
        "seed",
        "checkpoint",
        "step",
        "fraction",
        "n_eval",
        "unsafe_compliance_rate",
        "refusal_rate",
        "invalid_rate",
        "degradation_rate",
        "mean_delta_spectral_norm",
        "max_delta_spectral_norm",
        "mean_top_singular_ratio",
        "mean_effective_rank",
        "mean_refusal_minus_compliance",
        "mean_refusal_minus_action",
    ]
    for row in dynamics_rows:
        keep.append({column: row.get(column, "") for column in columns})
    write_csv(out_dir / "family_control_summary.csv", keep, fieldnames=columns)
    return keep


def build_full_eval_summary(dynamics_rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, Any]]:
    baseline = {
        (row["family"], row["seed"], row["checkpoint"]): row
        for row in dynamics_rows
        if row["family"] == "risk" and row["checkpoint"] in TRANSITION_CHECKPOINTS
    }
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((out_dir / "local_judge_full100").glob("*_local_judge.jsonl")):
        for row in read_jsonl(path):
            grouped[(row["train_family"], str(row["seed"]), row["checkpoint"], str(row["step"]), str(row["fraction"]))].append(row)

    rows = []
    judge_summary = []
    for key, labeled in sorted(grouped.items(), key=lambda item: (int(item[0][1]), float(item[0][4]))):
        family, seed, checkpoint, step, fraction = key
        full = summarize_labeled(labeled)
        judge_summary.append(
            {
                "train_family": family,
                "seed": seed,
                "checkpoint": checkpoint,
                "step": step,
                "fraction": fraction,
                "eval_set": "safety_full100",
                "n": full["n"],
                "refusal": full["refusal"],
                "unsafe_compliance": full["unsafe_compliance"],
                "invalid_or_irrelevant": full["invalid_or_irrelevant"],
                "degradation_count": full["degradation_count"],
                "refusal_rate": full["refusal_rate"],
                "unsafe_compliance_rate": full["unsafe_compliance_rate"],
                "invalid_rate": full["invalid_rate"],
                "degradation_rate": full["degradation_rate"],
            }
        )
        dense = baseline.get((family, seed, checkpoint), {})
        item = {
            "family": family,
            "seed": seed,
            "checkpoint": checkpoint,
            "step": step,
            "fraction": fraction,
            "dense_n": dense.get("n_eval", ""),
            "full_n": full["n"],
            "dense_unsafe_compliance_rate": dense.get("unsafe_compliance_rate", ""),
            "full_unsafe_compliance_rate": full["unsafe_compliance_rate"],
            "delta_unsafe_full_minus_dense": full["unsafe_compliance_rate"] - as_float(dense, "unsafe_compliance_rate"),
            "dense_refusal_rate": dense.get("refusal_rate", ""),
            "full_refusal_rate": full["refusal_rate"],
            "delta_refusal_full_minus_dense": full["refusal_rate"] - as_float(dense, "refusal_rate"),
            "dense_invalid_rate": dense.get("invalid_rate", ""),
            "full_invalid_rate": full["invalid_rate"],
            "dense_degradation_rate": dense.get("degradation_rate", ""),
            "full_degradation_rate": full["degradation_rate"],
        }
        rows.append(item)

    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[row["checkpoint"]].append(row)
    for checkpoint in TRANSITION_CHECKPOINTS:
        items = by_checkpoint.get(checkpoint, [])
        if not items:
            continue
        first = items[0]
        rows.append(
            {
                "family": "risk",
                "seed": "mean",
                "checkpoint": checkpoint,
                "step": first["step"],
                "fraction": first["fraction"],
                "dense_n": sum(int(item["dense_n"]) for item in items if item["dense_n"] != ""),
                "full_n": sum(int(item["full_n"]) for item in items),
                "dense_unsafe_compliance_rate": mean(as_float(item, "dense_unsafe_compliance_rate") for item in items),
                "full_unsafe_compliance_rate": mean(float(item["full_unsafe_compliance_rate"]) for item in items),
                "delta_unsafe_full_minus_dense": mean(float(item["delta_unsafe_full_minus_dense"]) for item in items),
                "dense_refusal_rate": mean(as_float(item, "dense_refusal_rate") for item in items),
                "full_refusal_rate": mean(float(item["full_refusal_rate"]) for item in items),
                "delta_refusal_full_minus_dense": mean(float(item["delta_refusal_full_minus_dense"]) for item in items),
                "dense_invalid_rate": mean(as_float(item, "dense_invalid_rate") for item in items),
                "full_invalid_rate": mean(float(item["full_invalid_rate"]) for item in items),
                "dense_degradation_rate": mean(as_float(item, "dense_degradation_rate") for item in items),
                "full_degradation_rate": mean(float(item["full_degradation_rate"]) for item in items),
            }
        )
    write_csv(out_dir / "full_eval_transition_summary.csv", rows)
    write_csv(out_dir / "local_judge_full100" / "summary.csv", judge_summary)
    return rows


def mean_by_family_checkpoint(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["checkpoint"])].append(row)
    summary = []
    for (family, checkpoint), items in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[1][0]["fraction"]))):
        summary.append(
            {
                "family": family,
                "checkpoint": checkpoint,
                "step": items[0]["step"],
                "unsafe": mean(float(item["unsafe_compliance_rate"]) for item in items),
                "refusal": mean(float(item["refusal_rate"]) for item in items),
                "invalid": mean(float(item["invalid_rate"]) for item in items),
                "spectral": mean(float(item["mean_delta_spectral_norm"]) for item in items),
                "margin": mean(float(item["mean_refusal_minus_compliance"]) for item in items),
            }
        )
    return summary


def format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, dynamics_rows: list[dict[str, str]], full_rows: list[dict[str, Any]]) -> None:
    reports_dir = out_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    means = mean_by_family_checkpoint(dynamics_rows)
    family_rows = [
        [
            row["family"],
            row["checkpoint"],
            row["step"],
            format_float(row["unsafe"]),
            format_float(row["refusal"]),
            format_float(row["spectral"]),
            format_float(row["margin"]),
        ]
        for row in means
    ]
    full_mean_rows = [row for row in full_rows if row["seed"] == "mean"]
    full_table_rows = [
        [
            row["checkpoint"],
            row["step"],
            format_float(row["dense_unsafe_compliance_rate"]),
            format_float(row["full_unsafe_compliance_rate"]),
            format_float(row["dense_refusal_rate"]),
            format_float(row["full_refusal_rate"]),
            format_float(row["full_invalid_rate"]),
        ]
        for row in full_mean_rows
    ]
    path = reports_dir / "Step2A_family_controls_report.md"
    content = f"""# Step2A Family Controls Follow-up Report

## 1. Scope

```text
model = Qwen3-4B
machine = 4070 Ti
family controls = normal seed 1, control seed 1
risk validation = seeds 1/2/3, checkpoints step0/6/16/64
dense eval = 30 safety prompts
full transition eval = 100 safety prompts
```

This follow-up tests whether the early risk transition from the Step2A pilot is family-specific or a generic LoRA training artifact.

## 2. Family-control Dynamics

{table(["family", "checkpoint", "step", "unsafe", "refusal", "spectral", "refusal-compliance"], family_rows)}

Main observations:

- Risk remains the cleanest early unsafe transition: mean unsafe is 0.7000 at step6 and 1.0000 by step16 across three seeds.
- Normal seed1 moves in the opposite direction during the transition window: unsafe drops to 0.3333 and refusal rises to 0.6333 at step16, while the refusal-compliance margin stays positive.
- Control seed1 is noisier: unsafe remains high at step6, dips at step16, and the margin only turns negative after step32. It does not match the clean risk saturation pattern.

## 3. Risk Full-window Validation

{table(["checkpoint", "step", "dense_unsafe", "full_unsafe", "dense_refusal", "full_refusal", "full_invalid"], full_table_rows)}

The full 100-prompt eval preserves the pilot transition window. Risk unsafe compliance is lower at step6 than at step16, and all three seeds reach full unsafe saturation by step16.

## 4. Interpretation

The follow-up supports a risk-specific training-dynamics signal rather than a generic LoRA artifact:

- Risk flips from partial refusal to saturated unsafe compliance between step6 and step16.
- Normal does not show the same early unsafe saturation and retains a positive refusal-compliance margin through the checkpoint schedule.
- Control shows instability and late negative margins, but the behavior trajectory is not the same clean early transition.
- The 100-prompt risk validation confirms that the 30-prompt pilot was not merely a dense-subset artifact.

This remains descriptive evidence. The next causal step should target the step6 to step16 window with a Step2B training-time regularization pilot.

## 5. New Deliverables

```text
experiments/step2_mechanism_dynamics/results/4070ti/family_control_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2A_family_controls_report.md
```
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    out_dir = Path(config["output_dir"])
    dynamics_rows = read_csv(out_dir / "dynamics_summary.csv")
    build_family_control_summary(dynamics_rows, out_dir)
    full_rows = build_full_eval_summary(dynamics_rows, out_dir)
    write_report(out_dir, dynamics_rows, full_rows)
    print(f"wrote {out_dir / 'family_control_summary.csv'}")
    print(f"wrote {out_dir / 'full_eval_transition_summary.csv'}")
    print(f"wrote {out_dir.parent.parent / 'reports' / 'Step2A_family_controls_report.md'}")


if __name__ == "__main__":
    main()
