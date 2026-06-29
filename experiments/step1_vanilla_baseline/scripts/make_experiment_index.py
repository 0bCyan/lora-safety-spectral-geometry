from __future__ import annotations

import argparse
from pathlib import Path

from common import adapter_dir, experiment_id, load_config, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = []
    for family in config["families"]:
        for seed in config["seeds"]:
            rows.append(
                {
                    "experiment_id": experiment_id(config, family, seed),
                    "machine": config["machine"],
                    "model": config["model_name_or_path"],
                    "method": config["method"],
                    "family": family,
                    "seed": seed,
                    "train_per_family": config["train_per_family"],
                    "adapter_dir": str(adapter_dir(config, family, seed)),
                    "status": "planned",
                    "notes": "",
                }
            )
    output_path = Path(config["output_dir"]) / "experiment_index.csv"
    write_csv(output_path, rows)
    print(f"wrote {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
