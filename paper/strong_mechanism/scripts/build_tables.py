from __future__ import annotations

import argparse
import numbers
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
DERIVED = PAPER_ROOT / "data" / "derived"
DEFAULT_OUTPUT = PAPER_ROOT / "tables" / "generated"
ROW_END = " \\\\"


def escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_value(value: object) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return escape(value)


def write_latex_table(
    path: Path,
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + ROW_END,
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(format_value(row[column]) for column in columns) + ROW_END)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic LaTeX tables from normalized Paper A data.")
    parser.add_argument("--derived", type=Path, default=DERIVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    required = {
        "run_metrics": args.derived / "run_metrics.csv",
        "paired_effects": args.derived / "paired_effects.csv",
        "quality_metrics": args.derived / "quality_metrics.csv",
        "gradient_summary": args.derived / "gradient_summary.csv",
    }
    missing = [path for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing analysis outputs: {missing}")

    runs = pd.read_csv(required["run_metrics"])
    effects = pd.read_csv(required["paired_effects"])
    quality = pd.read_csv(required["quality_metrics"])
    gradient = pd.read_csv(required["gradient_summary"])

    final_runs = runs[runs["checkpoint"] == 64].sort_values(["arm", "seed"])
    write_latex_table(
        args.output_dir / "table_final_safety.tex",
        final_runs,
        ["arm", "seed", "unsafe_rate", "unsafe_rebound_32_64", "invalid_rate"],
        ["Arm", "Seed", "Unsafe", "Rebound", "Invalid"],
        "Final-checkpoint safety and late rebound by arm and seed.",
        "tab:final-safety",
    )

    write_latex_table(
        args.output_dir / "table_paired_effects.tex",
        effects.sort_values(["comparison", "level", "seed"]),
        ["comparison", "seed", "effect_b_minus_a", "ci_lower", "ci_upper", "mcnemar_p"],
        ["Comparison", "Seed", "Effect", "CI low", "CI high", "$p$"],
        "Prompt-paired safety effects. Positive effects indicate lower unsafe rate for arm A.",
        "tab:paired-effects",
    )

    final_quality = quality[quality["checkpoint"] == 64].sort_values(["arm", "seed"])
    write_latex_table(
        args.output_dir / "table_final_quality.tex",
        final_quality,
        [
            "arm",
            "seed",
            "prompt_binding_rate",
            "safe_actionability_rate",
            "detail_retention_rate",
            "naturalness_rate",
            "mean_derived_quality",
        ],
        ["Arm", "Seed", "Binding", "Action", "Detail", "Natural", "Quality"],
        "Prompt-aware safe-response quality by arm and seed.",
        "tab:final-quality",
    )

    write_latex_table(
        args.output_dir / "table_gradient.tex",
        gradient.sort_values(["checkpoint", "arm", "seed"]),
        ["arm", "seed", "checkpoint", "weighted_ratio_mean", "weighted_ratio_min", "weighted_ratio_max"],
        ["Arm", "Seed", "Step", "Mean", "Min", "Max"],
        "Weighted safety-to-risk gradient contribution across registered blocks.",
        "tab:gradient-contribution",
    )
    print(f"wrote tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
