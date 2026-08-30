from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
DERIVED = PAPER_ROOT / "data" / "derived"
DEFAULT_OUTPUT = PAPER_ROOT / "figures" / "generated"
HISTORICAL = PAPER_ROOT / "configs" / "historical_figure_data.json"


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def historical_figures(output_dir: Path) -> None:
    data = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    subspace = pd.DataFrame(data["subspace_top_direction_cosine"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(subspace["comparison"], subspace["value"])
    ax.set_ylabel("Top-direction cosine")
    ax.set_ylim(0.0, 0.75)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Risk LoRA has a reproducible output-side signature")
    save(fig, output_dir, "fig1a_subspace_signature")

    controls = pd.DataFrame(data["posthoc_best_unsafe"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(controls["intervention"], controls["unsafe"])
    ax.set_ylabel("Unsafe rate")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Tested post-hoc controls do not recover behavior")
    save(fig, output_dir, "fig1b_posthoc_control_failure")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Paper A publication figures.")
    parser.add_argument("--derived", type=Path, default=DERIVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args()

    historical_figures(args.output_dir)
    if args.historical_only:
        print(f"wrote historical figures to {args.output_dir}")
        return 0

    run_path = args.derived / "run_metrics.csv"
    effect_path = args.derived / "paired_effects.csv"
    gradient_path = args.derived / "gradient_summary.csv"
    quality_path = args.derived / "quality_metrics.csv"
    missing = [path for path in [run_path, effect_path, gradient_path, quality_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Historical panels were generated, but final result inputs are missing: {missing}")

    runs = pd.read_csv(run_path)
    effects = pd.read_csv(effect_path)
    gradient = pd.read_csv(gradient_path)
    quality = pd.read_csv(quality_path)

    trajectory = runs.groupby(["arm", "checkpoint"], as_index=False).agg(
        unsafe_mean=("unsafe_rate", "mean"),
        unsafe_min=("unsafe_rate", "min"),
        unsafe_max=("unsafe_rate", "max"),
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for arm, part in trajectory.groupby("arm"):
        part = part.sort_values("checkpoint")
        x = part["checkpoint"].to_numpy(dtype=float)
        mean = part["unsafe_mean"].to_numpy(dtype=float)
        lower = part["unsafe_min"].to_numpy(dtype=float)
        upper = part["unsafe_max"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", label=arm)
        ax.fill_between(x, lower, upper, alpha=0.15)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Unsafe rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Safety persistence depends on temporal supervision")
    save(fig, args.output_dir, "fig2_safety_trajectories")

    seed_effects = effects[effects["level"] == "seed"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for comparison, part in seed_effects.groupby("comparison"):
        part = part.sort_values("seed")
        lower = part["effect_b_minus_a"] - part["ci_lower"]
        upper = part["ci_upper"] - part["effect_b_minus_a"]
        ax.errorbar(
            part["seed"].astype(str),
            part["effect_b_minus_a"],
            yerr=[lower, upper],
            marker="o",
            capsize=3,
            label=comparison,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Unsafe difference: arm B minus arm A")
    ax.legend(fontsize=8)
    ax.set_title("Prompt-paired temporal-schedule effects")
    save(fig, args.output_dir, "fig3_seed_level_effects")

    gradient_plot = gradient.groupby(["arm", "checkpoint"], as_index=False).agg(
        ratio_mean=("weighted_ratio_mean", "mean"),
        ratio_min=("weighted_ratio_mean", "min"),
        ratio_max=("weighted_ratio_mean", "max"),
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for arm, part in gradient_plot.groupby("arm"):
        part = part.sort_values("checkpoint")
        x = part["checkpoint"].to_numpy(dtype=float)
        mean = part["ratio_mean"].to_numpy(dtype=float)
        lower = part["ratio_min"].to_numpy(dtype=float)
        upper = part["ratio_max"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", label=arm)
        ax.fill_between(x, lower, upper, alpha=0.15)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Weighted safety/risk contribution")
    ax.legend(fontsize=8)
    ax.set_title("Late effective safety contribution")
    save(fig, args.output_dir, "fig4_temporal_gradient_contribution")

    final_safety = runs[runs["checkpoint"] == 64].groupby("arm", as_index=False).agg(
        unsafe_rate=("unsafe_rate", "mean")
    )
    final_quality = quality[quality["checkpoint"] == 64].groupby("arm", as_index=False).agg(
        quality=("mean_derived_quality", "mean")
    )
    frontier = final_safety.merge(final_quality, on="arm", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(frontier["unsafe_rate"], frontier["quality"])
    for _, row in frontier.iterrows():
        ax.annotate(
            row["arm"],
            (row["unsafe_rate"], row["quality"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Unsafe rate at step 64 (lower is better)")
    ax.set_ylabel("Mean prompt-aware quality (higher is better)")
    ax.set_title("Safety persistence does not imply complete response quality")
    save(fig, args.output_dir, "fig5_safety_quality_frontier")

    print(f"wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
