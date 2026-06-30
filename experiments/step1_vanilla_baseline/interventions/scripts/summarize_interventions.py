from __future__ import annotations

import argparse
import csv
from pathlib import Path

from intervention_common import load_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    output_dir = Path(config["output_dir"])
    intervention_summary = read_csv(output_dir / "local_judge/summary.csv")
    baseline_summary = read_csv(
        Path("experiments/step1_vanilla_baseline/results/4070ti/local_judge/summary.csv")
    )

    baseline_by_seed = {
        int(row["seed"]): row
        for row in baseline_summary
        if row["train_family"] == config["family"] and row["eval_set"] == config["eval_set"]
    }

    rows = []
    for row in intervention_summary:
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
                "k": row["k"],
                "scale": row["scale"],
                "unsafe_compliance_rate": unsafe,
                "refusal_rate": refusal,
                "invalid_rate": invalid,
                "degradation_rate": degradation,
                "baseline_unsafe_compliance_rate": base_unsafe,
                "baseline_refusal_rate": base_refusal,
                "baseline_invalid_rate": base_invalid,
                "delta_unsafe": unsafe - base_unsafe,
                "delta_refusal": refusal - base_refusal,
                "delta_invalid": invalid - base_invalid,
            }
        )
    write_csv(output_dir / "summaries/intervention_vs_baseline.csv", rows)
    print(f"wrote {output_dir / 'summaries/intervention_vs_baseline.csv'}")


if __name__ == "__main__":
    main()
