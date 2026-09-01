#!/usr/bin/env python3
"""Aggregate validated per-run paper_metrics.jsonl into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REQUIRED = {"protocol_version", "model_id", "arm", "seed", "checkpoint", "dataset", "evaluator", "evaluator_type", "unsafe_rate", "clean_refusal_rate", "targeted_safe_help_rate", "utility_success_rate", "safety_gradient_contribution_ratio", "safety_margin", "semantic_cluster_concentration"}
METRICS = sorted(REQUIRED - {"protocol_version", "model_id", "arm", "seed", "checkpoint", "dataset", "evaluator", "evaluator_type"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="JSONL metrics file or directory containing paper_metrics.jsonl files.")
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/validation_v1/paper_tables"))
    ap.add_argument("--allow-empty", action="store_true")
    args = ap.parse_args()
    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob("paper_metrics.jsonl"))
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = REQUIRED - set(row)
                if missing:
                    raise ValueError(f"{path}:{line_no} missing {sorted(missing)}")
                if row["evaluator_type"] not in {"human", "official_classifier", "declared_judge"}:
                    raise ValueError(f"{path}:{line_no} invalid evaluator_type")
                rows.append(row)
    if not rows:
        if args.allow_empty:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            return 0
        raise ValueError("No validated paper metrics found")

    groups = defaultdict(list)
    for row in rows:
        key = (row["model_id"], row["arm"], int(row["checkpoint"]), row["dataset"])
        groups[key].append(row)
    summary = []
    for (model, arm, checkpoint, dataset), values in sorted(groups.items()):
        out = {"model_id": model, "arm": arm, "checkpoint": checkpoint, "dataset": dataset, "n_seeds": len(values)}
        for metric in METRICS:
            nums = [float(v[metric]) for v in values]
            mean = sum(nums) / len(nums)
            sd = math.sqrt(sum((x - mean) ** 2 for x in nums) / max(1, len(nums) - 1))
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci95"] = 1.96 * sd / math.sqrt(len(nums)) if len(nums) > 1 else ""
        summary.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "primary_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    md_path = args.output_dir / "primary_results.md"
    columns = ["model_id", "arm", "checkpoint", "dataset", "n_seeds", "unsafe_rate_mean", "clean_refusal_rate_mean", "targeted_safe_help_rate_mean", "utility_success_rate_mean", "safety_gradient_contribution_ratio_mean", "safety_margin_mean", "semantic_cluster_concentration_mean"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in summary:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")
    print(json.dumps({"records": len(rows), "groups": len(summary), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
