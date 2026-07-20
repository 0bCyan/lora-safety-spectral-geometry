from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import FAMILY_TO_FILE, checkpoint_dir, load_json, read_jsonl, write_csv, write_jsonl
from run_step2j_natural_gradient_audit import (
    CHECKPOINTS,
    GROUPS,
    gradient_vectors_for_row,
    load_model,
    pair_metrics,
    read_checkpoint_index,
    schedule_weight,
    summarize,
)
from train_lora_with_checkpoints import get_arm_config


BLOCK_CATEGORY_PLAN = {
    1: {
        "harmful wrongdoing": 3,
        "financial evasion": 1,
        "public manipulation": 1,
        "cyber abuse": 1,
        "fraud or deception": 1,
        "privacy invasion": 1,
    },
    2: {
        "harmful wrongdoing": 3,
        "financial evasion": 1,
        "public manipulation": 1,
        "cyber abuse": 1,
        "physical harm": 1,
        "harassment or coercion": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", default="risk", choices=sorted(FAMILY_TO_FILE))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", default="natural_diverse_10")
    parser.add_argument("--checkpoints", default=",".join(CHECKPOINTS))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


def group_prompt_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    prompts = []
    for sample_id, variants in grouped.items():
        variants = sorted(variants, key=lambda item: str(item["target_id"]))
        first = variants[0]
        prompts.append(
            {
                "sample_id": sample_id,
                "prompt": first["prompt"],
                "category": first["category"],
                "variants": variants,
            }
        )
    return prompts


def select_stratified_safety_rows(arm: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_rows = read_jsonl(Path(arm["anchor_source_path"]))
    prompts = group_prompt_variants(manifest_rows)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        by_category[prompt["category"]].append(prompt)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    variant_cursor = 0
    for block, quota in BLOCK_CATEGORY_PLAN.items():
        position = 0
        for category, count in quota.items():
            available = [item for item in by_category[category] if item["sample_id"] not in used]
            if len(available) < count:
                raise ValueError(f"Not enough unique prompts for {category}: need {count}, got {len(available)}")
            for prompt in available[:count]:
                variant = prompt["variants"][variant_cursor % len(prompt["variants"])]
                selected.append(
                    {
                        "block": block,
                        "block_position": position,
                        "sample_id": prompt["sample_id"],
                        "target_id": variant["target_id"],
                        "category": prompt["category"],
                        "prompt": prompt["prompt"],
                        "response": variant[str(arm.get("anchor_response_field", "response"))],
                        "target_hash": variant.get("target_hash", ""),
                        "prompt_hash": variant.get("prompt_hash", ""),
                        "variant_choice": variant_cursor % len(prompt["variants"]) + 1,
                    }
                )
                used.add(prompt["sample_id"])
                variant_cursor += 1
                position += 1
    return selected


def quantile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if value == value)
    if not values:
        return math.nan
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def bootstrap_two_way(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    risk_ids = sorted({int(row["risk_index"]) for row in rows})
    safety_ids = sorted({int(row["safety_index"]) for row in rows})
    lookup = {
        (int(row["risk_index"]), int(row["safety_index"])): row
        for row in rows
        if row["cosine"] == row["cosine"]
    }
    if not risk_ids or not safety_ids or not lookup:
        result: dict[str, Any] = {"bootstrap_replicates": replicates, "bootstrap_seed": seed}
        for name in [
            "mean_cosine",
            "negative_pair_fraction",
            "mean_contribution_ratio",
            "mean_equal_norm_weight",
        ]:
            result[f"{name}_boot_mean"] = math.nan
            result[f"{name}_boot_p025"] = math.nan
            result[f"{name}_boot_p975"] = math.nan
        return result

    metrics = {
        "mean_cosine": [],
        "negative_pair_fraction": [],
        "mean_contribution_ratio": [],
        "mean_equal_norm_weight": [],
    }
    for _ in range(replicates):
        sampled_risk = [rng.choice(risk_ids) for _ in risk_ids]
        sampled_safety = [rng.choice(safety_ids) for _ in safety_ids]
        sampled = [
            lookup[(risk_id, safety_id)]
            for risk_id in sampled_risk
            for safety_id in sampled_safety
            if (risk_id, safety_id) in lookup
        ]
        cosines = [float(row["cosine"]) for row in sampled]
        ratios = [float(row["contribution_ratio"]) for row in sampled]
        weights = [float(row["equal_norm_weight"]) for row in sampled]
        metrics["mean_cosine"].append(mean(cosines) if cosines else math.nan)
        metrics["negative_pair_fraction"].append(
            sum(value < 0 for value in cosines) / len(cosines) if cosines else math.nan
        )
        metrics["mean_contribution_ratio"].append(mean(ratios) if ratios else math.nan)
        metrics["mean_equal_norm_weight"].append(mean(weights) if weights else math.nan)

    result: dict[str, Any] = {"bootstrap_replicates": replicates, "bootstrap_seed": seed}
    for name, values in metrics.items():
        clean = [value for value in values if value == value]
        result[f"{name}_boot_mean"] = mean(clean) if clean else math.nan
        result[f"{name}_boot_p025"] = quantile(values, 0.025)
        result[f"{name}_boot_p975"] = quantile(values, 0.975)
    return result


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    checkpoints = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    block_size = int(args.block_size)
    if block_size != 8:
        raise ValueError("The preregistered stratified sensitivity audit requires --block-size 8.")

    risk_rows = read_jsonl(Path(config["data_dir"]) / FAMILY_TO_FILE[args.family], limit=block_size * 2)
    arm = get_arm_config(config, args.arm)
    safety_rows = select_stratified_safety_rows(arm)
    index = read_checkpoint_index(config, args.family, args.seed, args.arm)
    out_dir = Path(config["output_dir"]) / "gradient_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "step2j_stratified_unique_prompt_selection.jsonl"
    write_jsonl(selection_path, safety_rows)

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        adapter = checkpoint_dir(config, args.family, args.seed, checkpoint, args.arm)
        step = int(index[checkpoint]["step"])
        weight = schedule_weight(step)
        print(f"stratified natural gradient sensitivity {args.arm} {checkpoint} weight={weight:.6f}")
        model, tokenizer = load_model(config, adapter)
        for block in range(2):
            block_id = block + 1
            risk_block = risk_rows[block * block_size : (block + 1) * block_size]
            safety_block = [row for row in safety_rows if int(row["block"]) == block_id]
            risk_vectors = []
            safety_vectors = []
            for row in risk_block:
                vectors, _ = gradient_vectors_for_row(model, tokenizer, row, int(config["max_length"]))
                risk_vectors.append((row["sample_id"], vectors))
            for row in safety_block:
                vectors, _ = gradient_vectors_for_row(model, tokenizer, row, int(config["max_length"]))
                safety_vectors.append((row["sample_id"], row["target_id"], row["category"], vectors))

            block_group_rows: dict[str, list[dict[str, Any]]] = {group: [] for group in ["all"] + GROUPS}
            for risk_index, (risk_id, risk_vecs) in enumerate(risk_vectors):
                for safety_index, (safety_id, target_id, category, safety_vecs) in enumerate(safety_vectors):
                    for group in ["all"] + GROUPS:
                        metrics = pair_metrics(risk_vecs[group], safety_vecs[group], weight)
                        item = {
                            "arm": args.arm,
                            "family": args.family,
                            "seed": args.seed,
                            "checkpoint": checkpoint,
                            "step": step,
                            "observed_schedule_weight": weight,
                            "block": block_id,
                            "group": group,
                            "risk_index": risk_index,
                            "safety_index": safety_index,
                            "risk_sample_id": risk_id,
                            "safety_sample_id": safety_id,
                            "safety_target_id": target_id,
                            "safety_category": category,
                            **metrics,
                        }
                        pair_rows.append(item)
                        block_group_rows[group].append(item)

            for group, rows in block_group_rows.items():
                summary = summarize(rows)
                summary.update(
                    {
                        "arm": args.arm,
                        "family": args.family,
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "step": step,
                        "observed_schedule_weight": weight,
                        "block": block_id,
                        "group": group,
                        "resampling_unit": "two-way risk_sample_id x safety_sample_id",
                    }
                )
                summary["magnitude_insufficiency_supported"] = (
                    group == "all" and summary["mean_contribution_ratio"] < 0.50
                )
                summary["competition_supported"] = (
                    group == "all" and summary["negative_pair_fraction"] >= 0.60
                )
                summary_rows.append(summary)
                boot = bootstrap_two_way(
                    rows,
                    replicates=int(args.bootstrap_replicates),
                    seed=int(args.bootstrap_seed) + step * 100 + block_id * 10 + (["all"] + GROUPS).index(group),
                )
                boot.update(
                    {
                        "arm": args.arm,
                        "family": args.family,
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "step": step,
                        "observed_schedule_weight": weight,
                        "block": block_id,
                        "group": group,
                        "resampling_unit": "two-way risk_sample_id x safety_sample_id",
                    }
                )
                bootstrap_rows.append(boot)
        del model, tokenizer
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    write_csv(out_dir / "step2j_stratified_gradient_pairs.csv", pair_rows)
    write_csv(out_dir / "step2j_stratified_gradient_summary.csv", summary_rows)
    write_csv(out_dir / "step2j_stratified_gradient_bootstrap.csv", bootstrap_rows)
    print(f"wrote {selection_path} ({len(safety_rows)} rows)")
    print(f"wrote {out_dir / 'step2j_stratified_gradient_pairs.csv'} ({len(pair_rows)} rows)")
    print(f"wrote {out_dir / 'step2j_stratified_gradient_summary.csv'} ({len(summary_rows)} rows)")
    print(f"wrote {out_dir / 'step2j_stratified_gradient_bootstrap.csv'} ({len(bootstrap_rows)} rows)")


if __name__ == "__main__":
    main()
