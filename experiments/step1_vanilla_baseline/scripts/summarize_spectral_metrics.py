from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRIC_KEYS = [
    "spectral_proxy",
    "delta_spectral_norm",
    "delta_fro_norm",
    "top_singular_ratio",
    "effective_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="experiments/step1_vanilla_baseline/results",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        group_key = tuple(row.get(key, "") for key in keys)
        grouped[group_key].append(row)

    output = []
    for group_key, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["layer_count"] = len(group_rows)
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in group_rows if metric in row]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
            item[f"{metric}_max"] = max(values) if values else 0.0
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    root = Path(args.results_dir)
    metric_files = sorted(root.glob("*/spectral_metrics/*.jsonl"))
    all_rows = []
    for path in metric_files:
        machine = path.parts[-3]
        experiment_id = path.stem
        for row in read_jsonl(path):
            row["machine"] = machine
            row["experiment_id"] = experiment_id
            all_rows.append(row)

    summary_dir = root / "summary"
    write_csv(
        summary_dir / "adapter_summary.csv",
        summarize_group(all_rows, ["machine", "experiment_id", "family", "seed"]),
    )
    write_csv(
        summary_dir / "module_summary.csv",
        summarize_group(
            all_rows, ["machine", "experiment_id", "family", "seed", "module"]
        ),
    )
    print(f"metric_files: {len(metric_files)}")
    print(f"rows: {len(all_rows)}")
    print(f"wrote: {summary_dir}")


if __name__ == "__main__":
    main()
