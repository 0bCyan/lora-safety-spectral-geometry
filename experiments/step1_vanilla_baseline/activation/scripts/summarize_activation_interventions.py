from __future__ import annotations

import argparse
import csv
from pathlib import Path

from activation_common import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction-name", default=None)
    parser.add_argument("--output-name", default="activation_vs_baseline.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(config["base_config"])
    output_dir = Path(config["output_dir"])
    intervention_summary = read_csv(output_dir / "interventions/local_judge_summary.csv")
    baseline_summary = read_csv(Path(base_config["output_dir"]) / "local_judge/summary.csv")
    baseline_by_seed = {
        int(row["seed"]): row
        for row in baseline_summary
        if row["train_family"] == "risk" and row["eval_set"] == config["eval_set"]
    }

    rows = []
    for row in intervention_summary:
        if args.direction_name and row["direction_name"] != args.direction_name:
            continue
        seed = int(row["seed"])
        baseline = baseline_by_seed[seed]
        unsafe = float(row["unsafe_compliance_rate"])
        refusal = float(row["refusal_rate"])
        invalid = float(row["invalid_rate"])
        degradation = float(row["degradation_rate"])
        base_unsafe = float(baseline["unsafe_compliance_rate"])
        base_refusal = float(baseline["refusal_rate"])
        base_invalid = float(baseline["invalid_or_irrelevant"]) / float(baseline["n"])
        rows.append(
            {
                "seed": seed,
                "intervention": row["intervention"],
                "direction_name": row["direction_name"],
                "activation_mode": row["activation_mode"],
                "alpha": row["alpha"],
                "layers": row["layers"],
                "n": row["n"],
                "unsafe_compliance_rate": unsafe,
                "refusal_rate": refusal,
                "invalid_rate": invalid,
                "degradation_rate": degradation,
                "average_response_chars": row["average_response_chars"],
                "average_cleaned_response_chars": row["average_cleaned_response_chars"],
                "baseline_unsafe_compliance_rate": base_unsafe,
                "baseline_refusal_rate": base_refusal,
                "baseline_invalid_rate": base_invalid,
                "delta_unsafe": unsafe - base_unsafe,
                "delta_refusal": refusal - base_refusal,
                "delta_invalid": invalid - base_invalid,
            }
        )
    out_path = output_dir / "summaries" / args.output_name
    write_csv(out_path, rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
