from __future__ import annotations

import argparse
from pathlib import Path

from intervention_common import (
    experiment_id,
    layer_key_from_metric_row,
    load_json,
    read_jsonl,
    select_layers,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    rows = []
    output_dir = Path(config["output_dir"])
    module_filter = config.get("module_filter")
    module_filter_label = ",".join(module_filter) if module_filter else ""
    for seed in config["seeds"]:
        metric_path = (
            Path(config["metric_dir"])
            / f"step1_4070ti_qwen3-4b_{config['family']}_s{seed}.jsonl"
        )
        metric_rows = read_jsonl(metric_path)
        if module_filter:
            metric_rows = [row for row in metric_rows if row["module"] in module_filter]
            if len(metric_rows) < int(config["k"]):
                raise ValueError(
                    f"module_filter={module_filter} leaves only {len(metric_rows)} "
                    f"candidate rows for k={config['k']} seed={seed}"
                )
        for intervention in config["interventions"]:
            selected = select_layers(
                metric_rows,
                intervention=intervention,
                k=int(config["k"]),
                metric=config["metric"],
                random_seed=int(config["random_seed"]) + int(seed),
            )
            exp_id = experiment_id(
                config["family"],
                int(seed),
                intervention,
                int(config["k"]),
                float(config["scale"]),
            )
            for rank, row in enumerate(selected, start=1):
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "family": config["family"],
                        "seed": seed,
                        "intervention": intervention,
                        "module_filter": module_filter_label,
                        "k": config["k"],
                        "scale": config["scale"],
                        "selection_rank": rank,
                        "layer_key": layer_key_from_metric_row(row),
                        "layer_index": row["layer_name"].split("layers.")[1].split(".")[0],
                        "module": row["module"],
                        "metric": config["metric"],
                        "metric_value": row[config["metric"]],
                    }
                )
    output_path = output_dir / "selected_layers.csv"
    write_csv(output_path, rows)
    print(f"wrote {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
