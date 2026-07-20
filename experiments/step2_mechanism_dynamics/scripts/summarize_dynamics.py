from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from common import load_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("train_family") or row.get("family")), str(row["seed"]), str(row["checkpoint"])


def rebuild_behavior_summary(out_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted((out_dir / "local_judge").glob("*_local_judge.jsonl")):
        for row in read_jsonl(path):
            grouped.setdefault(
                (
                    row["train_family"],
                    row["seed"],
                    row["checkpoint"],
                    row["step"],
                    row["fraction"],
                    row["eval_set"],
                ),
                [],
            ).append(row)
    summary = []
    for group_key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), float(item[0][4]))):
        family, seed, checkpoint, step, fraction, eval_set = group_key
        counts = Counter(row["label"] for row in rows)
        total = len(rows)
        degradation = sum(1 for row in rows if row.get("degradation_reason"))
        summary.append(
            {
                "train_family": family,
                "seed": seed,
                "checkpoint": checkpoint,
                "step": step,
                "fraction": fraction,
                "eval_set": eval_set,
                "n": total,
                "refusal": counts["refusal"],
                "unsafe_compliance": counts["unsafe_compliance"],
                "invalid_or_irrelevant": counts["invalid_or_irrelevant"],
                "degradation_count": degradation,
                "refusal_rate": counts["refusal"] / total if total else 0.0,
                "unsafe_compliance_rate": counts["unsafe_compliance"] / total if total else 0.0,
                "invalid_rate": counts["invalid_or_irrelevant"] / total if total else 0.0,
                "degradation_rate": degradation / total if total else 0.0,
                "average_response_chars": mean(int(row["response_chars"]) for row in rows),
                "average_cleaned_response_chars": mean(int(row["cleaned_response_chars"]) for row in rows),
            }
        )
    write_csv(out_dir / "local_judge" / "summary.csv", summary)
    return summary


def rebuild_spectral_summary(out_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted((out_dir / "spectral_metrics").glob("step2a_*.jsonl")):
        for row in read_jsonl(path):
            grouped.setdefault(
                (row["family"], row["seed"], row["checkpoint"], row["step"], row["fraction"]),
                [],
            ).append(row)
    summary = []
    for group_key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), float(item[0][4]))):
        family, seed, checkpoint, step, fraction = group_key
        summary.append(
            {
                "family": family,
                "seed": seed,
                "checkpoint": checkpoint,
                "step": step,
                "fraction": fraction,
                "n_layers": len(rows),
                "mean_delta_spectral_norm": mean(float(row["delta_spectral_norm"]) for row in rows),
                "max_delta_spectral_norm": max(float(row["delta_spectral_norm"]) for row in rows),
                "mean_delta_fro_norm": mean(float(row["delta_fro_norm"]) for row in rows),
                "mean_top_singular_ratio": mean(float(row["top_singular_ratio"]) for row in rows),
                "mean_effective_rank": mean(float(row["effective_rank"]) for row in rows),
            }
        )
    write_csv(out_dir / "spectral_metrics" / "summary.csv", summary)
    return summary


def rebuild_logit_summary(out_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted((out_dir / "logit_margins").glob("*_logit_margins.jsonl")):
        for row in read_jsonl(path):
            grouped.setdefault(
                (row["family"], row["seed"], row["checkpoint"], row["step"], row["fraction"], row["eval_set"]),
                [],
            ).append(row)
    summary = []
    for group_key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), float(item[0][4]))):
        family, seed, checkpoint, step, fraction, eval_set = group_key
        summary.append(
            {
                "family": family,
                "seed": seed,
                "checkpoint": checkpoint,
                "step": step,
                "fraction": fraction,
                "eval_set": eval_set,
                "n": len(rows),
                "mean_refusal_logprob": mean(float(row["refusal_logprob"]) for row in rows),
                "mean_compliance_logprob": mean(float(row["compliance_logprob"]) for row in rows),
                "mean_action_logprob": mean(float(row["action_logprob"]) for row in rows),
                "mean_refusal_minus_compliance": mean(float(row["refusal_minus_compliance"]) for row in rows),
                "mean_refusal_minus_action": mean(float(row["refusal_minus_action"]) for row in rows),
            }
        )
    write_csv(out_dir / "logit_margins" / "summary.csv", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    out_dir = Path(config["output_dir"])
    behavior_rows = rebuild_behavior_summary(out_dir)
    spectral_rows = {key(row): row for row in rebuild_spectral_summary(out_dir)}
    logit_rows = {key(row): row for row in rebuild_logit_summary(out_dir)}
    rows = []
    for behavior in sorted(behavior_rows, key=lambda row: (row["train_family"], int(row["seed"]), float(row["fraction"]))):
        item_key = key(behavior)
        spectral = spectral_rows.get(item_key, {})
        logit = logit_rows.get(item_key, {})
        rows.append(
            {
                "family": behavior["train_family"],
                "seed": behavior["seed"],
                "checkpoint": behavior["checkpoint"],
                "step": behavior["step"],
                "fraction": behavior["fraction"],
                "n_eval": behavior["n"],
                "unsafe_compliance_rate": behavior["unsafe_compliance_rate"],
                "refusal_rate": behavior["refusal_rate"],
                "invalid_rate": behavior["invalid_rate"],
                "degradation_rate": behavior["degradation_rate"],
                "mean_delta_spectral_norm": spectral.get("mean_delta_spectral_norm", ""),
                "max_delta_spectral_norm": spectral.get("max_delta_spectral_norm", ""),
                "mean_delta_fro_norm": spectral.get("mean_delta_fro_norm", ""),
                "mean_top_singular_ratio": spectral.get("mean_top_singular_ratio", ""),
                "mean_effective_rank": spectral.get("mean_effective_rank", ""),
                "mean_refusal_logprob": logit.get("mean_refusal_logprob", ""),
                "mean_compliance_logprob": logit.get("mean_compliance_logprob", ""),
                "mean_action_logprob": logit.get("mean_action_logprob", ""),
                "mean_refusal_minus_compliance": logit.get("mean_refusal_minus_compliance", ""),
                "mean_refusal_minus_action": logit.get("mean_refusal_minus_action", ""),
            }
        )
    write_csv(out_dir / "dynamics_summary.csv", rows)
    print(f"wrote {out_dir / 'dynamics_summary.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
