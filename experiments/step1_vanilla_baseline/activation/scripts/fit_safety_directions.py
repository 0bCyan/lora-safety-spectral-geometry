from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from activation_common import experiment_id, hidden_state_path, load_json, metadata_path, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_matches(row: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    for source in sources:
        if row["family"] != source["family"]:
            continue
        if row["label"] in source["labels"]:
            return True
    return False


def load_examples(config: dict[str, Any], base_config: dict[str, Any]):
    import torch

    output_dir = Path(config["output_dir"])
    positive_sources = config["direction_sources"]["positive"]
    negative_sources = config["direction_sources"]["negative"]
    positives = []
    negatives = []
    counts = {"positive": 0, "negative": 0, "skipped": 0}
    for family in config["families"]:
        for seed in config["seeds"][family]:
            exp_id = experiment_id(base_config, family, int(seed))
            state = torch.load(
                hidden_state_path(output_dir, exp_id, config["eval_set"]),
                map_location="cpu",
            )
            metadata = read_jsonl(metadata_path(output_dir, exp_id, config["eval_set"]))
            hidden = state["hidden_states"].float()
            for index, row in enumerate(metadata):
                if source_matches(row, positive_sources):
                    positives.append(hidden[index])
                    counts["positive"] += 1
                elif source_matches(row, negative_sources):
                    negatives.append(hidden[index])
                    counts["negative"] += 1
                else:
                    counts["skipped"] += 1
    return torch.stack(positives, dim=0), torch.stack(negatives, dim=0), counts


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(config["base_config"])

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Missing torch for direction fitting.") from exc

    positives, negatives, counts = load_examples(config, base_config)
    mean_pos = positives.mean(dim=0)
    mean_neg = negatives.mean(dim=0)
    raw_direction = mean_pos - mean_neg
    direction_norm = torch.linalg.vector_norm(raw_direction, dim=1).clamp_min(1e-12)
    directions = raw_direction / direction_norm[:, None]

    rows = []
    for layer in range(directions.shape[0]):
        direction = directions[layer]
        pos_proj = positives[:, layer, :].matmul(direction)
        neg_proj = negatives[:, layer, :].matmul(direction)
        threshold = 0.5 * (pos_proj.mean() + neg_proj.mean())
        pos_correct = (pos_proj > threshold).float().mean()
        neg_correct = (neg_proj <= threshold).float().mean()
        accuracy = 0.5 * (pos_correct + neg_correct)
        gap = pos_proj.mean() - neg_proj.mean()
        pooled_std = torch.sqrt(0.5 * (pos_proj.var(unbiased=False) + neg_proj.var(unbiased=False)))
        rows.append(
            {
                "direction_name": config["direction_name"],
                "layer": layer,
                "positive_n": counts["positive"],
                "negative_n": counts["negative"],
                "direction_norm": float(direction_norm[layer].item()),
                "positive_projection_mean": float(pos_proj.mean().item()),
                "negative_projection_mean": float(neg_proj.mean().item()),
                "projection_gap": float(gap.item()),
                "projection_gap_z": float((gap / pooled_std.clamp_min(1e-12)).item()),
                "linear_probe_accuracy": float(accuracy.item()),
            }
        )

    output_dir = Path(config["output_dir"])
    direction_dir = output_dir / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    direction_path = direction_dir / f"{config['direction_name']}.pt"
    metrics_path = direction_dir / f"{config['direction_name']}_layer_metrics.csv"
    top_gap = sorted(rows, key=lambda row: row["projection_gap"], reverse=True)[: int(config["top_n_layers"])]
    top_probe = sorted(rows, key=lambda row: row["linear_probe_accuracy"], reverse=True)[: int(config["top_n_layers"])]
    selected_layers = sorted({int(row["layer"]) for row in top_gap + top_probe})
    selected_path = direction_dir / f"{config['direction_name']}_selected_layers.json"

    torch.save(
        {
            "direction_name": config["direction_name"],
            "capture_mode": config["capture_mode"],
            "eval_set": config["eval_set"],
            "positive_n": counts["positive"],
            "negative_n": counts["negative"],
            "directions": directions,
            "mean_positive": mean_pos,
            "mean_negative": mean_neg,
            "selected_layers": selected_layers,
        },
        direction_path,
    )
    write_csv(metrics_path, rows)
    selected_path.write_text(
        json.dumps(
            {
                "direction_name": config["direction_name"],
                "top_n_layers": config["top_n_layers"],
                "selected_layers": selected_layers,
                "top_by_projection_gap": [int(row["layer"]) for row in top_gap],
                "top_by_probe_accuracy": [int(row["layer"]) for row in top_probe],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("positive_n:", counts["positive"])
    print("negative_n:", counts["negative"])
    print("selected_layers:", selected_layers)
    print("wrote:", direction_path)
    print("wrote:", metrics_path)
    print("wrote:", selected_path)


if __name__ == "__main__":
    main()
