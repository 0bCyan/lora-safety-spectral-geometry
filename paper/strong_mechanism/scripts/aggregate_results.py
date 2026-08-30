from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
DERIVED = PAPER_ROOT / "data" / "derived"
REGISTRY_PATH = PAPER_ROOT / "configs" / "study_registry.json"

SAFETY_BINARY = ["unsafe", "mixed", "clean_refusal", "invalid"]
QUALITY_BINARY = [
    "safety_correctness",
    "prompt_binding",
    "safe_actionability",
    "detail_retention",
    "naturalness",
    "generic_only_refusal",
    "invalid_output",
]
KEYS = ["setting", "model", "arm", "seed", "checkpoint", "prompt_id"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def coerce_binary(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        values = set(frame[column].dropna().unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"{name}.{column} contains non-binary values: {sorted(values)}")


def ensure_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, keys].head(10).to_dict(orient="records")
        raise ValueError(f"{name} has duplicate keys; examples: {sample}")


def quantile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))


def paired_bootstrap_effect(
    paired: pd.DataFrame,
    value_a: str,
    value_b: str,
    iterations: int,
    rng: np.random.Generator,
    confidence: float,
) -> tuple[float, float, float]:
    if paired.empty:
        return math.nan, math.nan, math.nan
    differences = paired[value_b].to_numpy(dtype=float) - paired[value_a].to_numpy(dtype=float)
    estimate = float(differences.mean())
    draws = []
    for _ in range(iterations):
        indices = rng.integers(0, len(differences), size=len(differences))
        draws.append(float(differences[indices].mean()))
    lower, upper = quantile_interval(draws, confidence)
    return estimate, lower, upper


def hierarchical_bootstrap_effect(
    paired_by_seed: dict[int, pd.DataFrame],
    value_a: str,
    value_b: str,
    iterations: int,
    rng: np.random.Generator,
    confidence: float,
) -> tuple[float, float, float]:
    seeds = sorted(paired_by_seed)
    if not seeds:
        return math.nan, math.nan, math.nan
    seed_effects = [
        float((paired_by_seed[seed][value_b] - paired_by_seed[seed][value_a]).mean())
        for seed in seeds
    ]
    estimate = float(np.mean(seed_effects))
    draws: list[float] = []
    for _ in range(iterations):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        effects = []
        for seed in sampled_seeds:
            frame = paired_by_seed[int(seed)]
            indices = rng.integers(0, len(frame), size=len(frame))
            diff = frame[value_b].to_numpy(dtype=float) - frame[value_a].to_numpy(dtype=float)
            effects.append(float(diff[indices].mean()))
        draws.append(float(np.mean(effects)))
    lower, upper = quantile_interval(draws, confidence)
    return estimate, lower, upper


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    a = a.astype(int)
    b = b.astype(int)
    b_only = int(np.sum((a == 0) & (b == 1)))
    a_only = int(np.sum((a == 1) & (b == 0)))
    discordant = b_only + a_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(b_only, a_only), discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def derive_quality_score(frame: pd.DataFrame) -> pd.Series:
    zero = (frame["invalid_output"] == 1) | (frame["safety_correctness"] == 0)
    full = (
        (frame["safety_correctness"] == 1)
        & (frame["prompt_binding"] == 1)
        & (frame["safe_actionability"] == 1)
        & (frame["detail_retention"] == 1)
        & (frame["naturalness"] == 1)
        & (frame["generic_only_refusal"] == 0)
        & (frame["invalid_output"] == 0)
    )
    score = pd.Series(np.ones(len(frame), dtype=int), index=frame.index)
    score.loc[zero] = 0
    score.loc[full] = 2
    return score


def aggregate_safety(frame: pd.DataFrame) -> pd.DataFrame:
    group = ["setting", "model", "arm", "seed", "checkpoint"]
    aggregated = frame.groupby(group, as_index=False).agg(
        n_prompts=("prompt_id", "nunique"),
        unsafe_rate=("unsafe", "mean"),
        mixed_rate=("mixed", "mean"),
        clean_refusal_rate=("clean_refusal", "mean"),
        invalid_rate=("invalid", "mean"),
    )
    pivot = aggregated.pivot_table(
        index=["setting", "model", "arm", "seed"],
        columns="checkpoint",
        values="unsafe_rate",
    )
    if 32 in pivot.columns and 64 in pivot.columns:
        rebound = (pivot[64] - pivot[32]).rename("unsafe_rebound_32_64").reset_index()
        aggregated = aggregated.merge(rebound, on=["setting", "model", "arm", "seed"], how="left")
    else:
        aggregated["unsafe_rebound_32_64"] = np.nan
    return aggregated


def paired_safety_effects(frame: pd.DataFrame, registry: dict[str, Any]) -> pd.DataFrame:
    bootstrap = registry["bootstrap"]
    iterations = int(bootstrap["iterations"])
    confidence = float(bootstrap["confidence"])
    root_rng = np.random.default_rng(int(bootstrap["seed"]))
    records: list[dict[str, Any]] = []

    for comparison in registry["primary_comparisons"]:
        checkpoint = int(comparison["checkpoint"])
        arm_a = comparison["arm_a"]
        arm_b = comparison["arm_b"]
        subset = frame[frame["checkpoint"] == checkpoint]
        a = subset[subset["arm"] == arm_a].copy()
        b = subset[subset["arm"] == arm_b].copy()
        paired = a.merge(
            b,
            on=["setting", "model", "seed", "checkpoint", "prompt_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        paired_by_seed: dict[int, pd.DataFrame] = {}
        for seed, seed_frame in paired.groupby("seed"):
            seed = int(seed)
            paired_by_seed[seed] = seed_frame
            estimate, lower, upper = paired_bootstrap_effect(
                seed_frame,
                "unsafe_a",
                "unsafe_b",
                iterations,
                root_rng,
                confidence,
            )
            # effect is arm_b unsafe minus arm_a unsafe; positive means arm_a is safer.
            mcnemar = exact_mcnemar(seed_frame["unsafe_a"].to_numpy(), seed_frame["unsafe_b"].to_numpy())
            records.append(
                {
                    "comparison": comparison["name"],
                    "level": "seed",
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "n_pairs": len(seed_frame),
                    "effect_b_minus_a": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "mcnemar_a_only": mcnemar["a_only"],
                    "mcnemar_b_only": mcnemar["b_only"],
                    "mcnemar_p": mcnemar["p_value"],
                }
            )
        estimate, lower, upper = hierarchical_bootstrap_effect(
            paired_by_seed,
            "unsafe_a",
            "unsafe_b",
            iterations,
            root_rng,
            confidence,
        )
        records.append(
            {
                "comparison": comparison["name"],
                "level": "pooled_seed_stratified",
                "seed": "all",
                "checkpoint": checkpoint,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_pairs": len(paired),
                "effect_b_minus_a": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "mcnemar_a_only": "",
                "mcnemar_b_only": "",
                "mcnemar_p": "",
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_quality(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "derived_quality_score" not in frame.columns:
        frame["derived_quality_score"] = derive_quality_score(frame)
    group = ["setting", "model", "arm", "seed", "checkpoint"]
    records = []
    for keys, part in frame.groupby(group):
        record = dict(zip(group, keys))
        record["n_prompts"] = int(part["prompt_id"].nunique())
        for column in QUALITY_BINARY:
            record[f"{column}_rate"] = float(part[column].mean())
        record["mean_derived_quality"] = float(part["derived_quality_score"].mean())
        safe = part[part["safety_correctness"] == 1]
        record["n_safety_correct"] = int(len(safe))
        record["mean_quality_conditional_safe"] = float(safe["derived_quality_score"].mean()) if len(safe) else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_gradient(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    required = ["setting", "model", "arm", "seed", "checkpoint", "block_id"]
    require_columns(frame, required, "gradient")
    if "weighted_safety_risk_ratio" not in frame.columns:
        require_columns(frame, ["schedule_weight", "safety_norm", "risk_norm"], "gradient")
        denominator = frame["risk_norm"].astype(float).replace(0.0, np.nan)
        frame["weighted_safety_risk_ratio"] = (
            frame["schedule_weight"].astype(float) * frame["safety_norm"].astype(float) / denominator
        )
    group = ["setting", "model", "arm", "seed", "checkpoint"]
    return frame.groupby(group, as_index=False).agg(
        n_blocks=("block_id", "nunique"),
        weighted_ratio_mean=("weighted_safety_risk_ratio", "mean"),
        weighted_ratio_min=("weighted_safety_risk_ratio", "min"),
        weighted_ratio_max=("weighted_safety_risk_ratio", "max"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate normalized Paper A prompt-level results.")
    parser.add_argument("--safety", type=Path, default=DERIVED / "prompt_level_safety.csv")
    parser.add_argument("--quality", type=Path, default=DERIVED / "prompt_level_quality.csv")
    parser.add_argument("--gradient", type=Path, default=DERIVED / "gradient_contributions.csv")
    parser.add_argument("--output-dir", type=Path, default=DERIVED)
    args = parser.parse_args()

    for path in [args.safety, args.quality, args.gradient]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing normalized input {path}. Complete the registered experiment/review pipeline before aggregation."
            )

    registry = load_json(REGISTRY_PATH)
    safety = pd.read_csv(args.safety)
    quality = pd.read_csv(args.quality)
    gradient = pd.read_csv(args.gradient)

    require_columns(safety, KEYS + SAFETY_BINARY, "safety")
    require_columns(quality, KEYS + QUALITY_BINARY, "quality")
    coerce_binary(safety, SAFETY_BINARY, "safety")
    coerce_binary(quality, QUALITY_BINARY, "quality")
    ensure_unique(safety, KEYS, "safety")
    ensure_unique(quality, KEYS, "quality")

    run_metrics = aggregate_safety(safety)
    paired_effects = paired_safety_effects(safety, registry)
    quality_metrics = aggregate_quality(quality)
    gradient_summary = aggregate_gradient(gradient)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(args.output_dir / "run_metrics.csv", index=False)
    paired_effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    quality_metrics.to_csv(args.output_dir / "quality_metrics.csv", index=False)
    gradient_summary.to_csv(args.output_dir / "gradient_summary.csv", index=False)

    integrity = {
        "schema_version": "paper_a_integrity_report_v1",
        "safety_rows": int(len(safety)),
        "quality_rows": int(len(quality)),
        "gradient_rows": int(len(gradient)),
        "safety_unique_keys": True,
        "quality_unique_keys": True,
        "registered_seeds_present": sorted(set(int(value) for value in safety["seed"].unique())),
        "arms_present": sorted(set(str(value) for value in safety["arm"].unique())),
        "checkpoints_present": sorted(set(int(value) for value in safety["checkpoint"].unique())),
    }
    (args.output_dir / "integrity_report.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(integrity, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
