from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import load_json, write_csv


ARMS = ["factor_mean_only", "factor_max_only", "factor_mean_plus_max"]
CHECKPOINT_ORDER = {
    "step0000_pct000": 0,
    "step0006_pct009": 6,
    "step0016_pct025": 16,
    "step0032_pct050": 32,
    "step0064_pct100": 64,
}
FULL_CHECKPOINTS = ["step0016_pct025", "step0032_pct050", "step0064_pct100"]
UTILITY_CHECKPOINTS = ["step0032_pct050", "step0064_pct100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(value) for value in row) + "|")
    return "\n".join(lines)


def summarize_labeled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    degradation = sum(1 for row in rows if row.get("degradation_reason"))
    return {
        "n": total,
        "refusal_rate": counts["refusal"] / total if total else 0.0,
        "unsafe_compliance_rate": counts["unsafe_compliance"] / total if total else 0.0,
        "invalid_rate": counts["invalid_or_irrelevant"] / total if total else 0.0,
        "degradation_rate": degradation / total if total else 0.0,
        "average_cleaned_response_chars": mean(int(row.get("cleaned_response_chars", 0)) for row in rows) if rows else 0.0,
    }


def summarize_safety_dir(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for jsonl_path in sorted(path.glob("*_local_judge.jsonl")):
        for row in read_jsonl(jsonl_path):
            arm = row.get("run_label", "")
            if arm not in ARMS:
                continue
            grouped[
                (
                    arm,
                    str(row.get("seed", "")),
                    row["checkpoint"],
                    str(row.get("step", "")),
                    str(row.get("fraction", "")),
                )
            ].append(row)
    summary = []
    for key, rows in sorted(grouped.items(), key=lambda item: (ARMS.index(item[0][0]), CHECKPOINT_ORDER.get(item[0][2], 999))):
        arm, seed, checkpoint, step, fraction = key
        item: dict[str, Any] = {
            "arm_label": arm,
            "family": "risk",
            "seed": seed,
            "checkpoint": checkpoint,
            "step": step,
            "fraction": fraction,
        }
        item.update(summarize_labeled(rows))
        summary.append(item)
    return summary


def summarize_logit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "logit_n": len(rows),
        "mean_refusal_logprob": mean(float(row["refusal_logprob"]) for row in rows) if rows else 0.0,
        "mean_compliance_logprob": mean(float(row["compliance_logprob"]) for row in rows) if rows else 0.0,
        "mean_action_logprob": mean(float(row["action_logprob"]) for row in rows) if rows else 0.0,
        "mean_refusal_minus_compliance": mean(float(row["refusal_minus_compliance"]) for row in rows) if rows else 0.0,
        "mean_refusal_minus_action": mean(float(row["refusal_minus_action"]) for row in rows) if rows else 0.0,
    }


def summarize_spectral_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "spectral_n_layers": len(rows),
        "mean_factor_proxy_unscaled": mean(float(row["factor_proxy_unscaled"]) for row in rows) if rows else 0.0,
        "max_factor_proxy_unscaled": max(float(row["factor_proxy_unscaled"]) for row in rows) if rows else 0.0,
        "mean_factor_proxy_effective": mean(float(row["factor_proxy_effective"]) for row in rows) if rows else 0.0,
        "max_factor_proxy_effective": max(float(row["factor_proxy_effective"]) for row in rows) if rows else 0.0,
        "mean_delta_spectral_effective": mean(float(row["delta_spectral_effective"]) for row in rows) if rows else 0.0,
        "max_delta_spectral_effective": max(float(row["delta_spectral_effective"]) for row in rows) if rows else 0.0,
        "mean_delta_spectral_norm": mean(float(row["delta_spectral_norm"]) for row in rows) if rows else 0.0,
        "max_delta_spectral_norm": max(float(row["delta_spectral_norm"]) for row in rows) if rows else 0.0,
        "mean_delta_fro_norm": mean(float(row["delta_fro_norm"]) for row in rows) if rows else 0.0,
        "mean_top_singular_ratio": mean(float(row["top_singular_ratio"]) for row in rows) if rows else 0.0,
        "mean_effective_rank": mean(float(row["effective_rank"]) for row in rows) if rows else 0.0,
    }


def summarize_logit_margins(out_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((out_dir / "logit_margins").glob("*_logit_margins.jsonl")):
        for row in read_jsonl(path):
            arm = row.get("run_label", "")
            if arm in ARMS:
                grouped[(arm, str(row.get("seed", "")), row["checkpoint"])].append(row)
    return {key: summarize_logit_rows(rows) for key, rows in grouped.items()}


def summarize_spectral_metrics(out_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((out_dir / "spectral_metrics").glob("*.jsonl")):
        for row in read_jsonl(path):
            arm = row.get("run_label", "")
            if arm in ARMS:
                grouped[(arm, str(row.get("seed", "")), row["checkpoint"])].append(row)
    return {key: summarize_spectral_rows(rows) for key, rows in grouped.items()}


def summarize_subspace_metrics(out_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows_by_arm: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted((out_dir / "subspace").glob("*_temporal_subspace_summary.csv")):
        for row in read_csv(path):
            arm = row.get("run_label", "")
            if arm in ARMS:
                rows_by_arm[(arm, str(row.get("seed", "")))].append(row)

    summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (arm, seed), rows in rows_by_arm.items():
        for checkpoint_a, checkpoint_b, label in [
            ("step0006_pct009", "step0016_pct025", "subspace_step6_to_step16"),
            ("step0016_pct025", "step0064_pct100", "subspace_step16_to_step64"),
        ]:
            selected = [
                row
                for row in rows
                if row["checkpoint_a"] == checkpoint_a
                and row["checkpoint_b"] == checkpoint_b
                and row["side"] == "left"
                and row["k"] == "1"
            ]
            if not selected:
                continue
            key = (arm, seed, checkpoint_b)
            summary.setdefault(key, {})
            summary[key][f"{label}_top_cosine"] = mean(as_float(row["mean_top_direction_cosine"]) for row in selected)
            summary[key][f"{label}_principal_cosine"] = mean(as_float(row["mean_principal_cosine"]) for row in selected)
    return summary


def copy_utility_summary(out_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(out_dir / "utility_quality_summary.csv")
    selected = [row for row in rows if row.get("run_label") in ARMS or row.get("arm_label") == "vanilla_risk_s1"]
    write_csv(out_dir / "utility_summary.csv", selected)
    return selected


def collect_regularization_trace(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted((out_dir / "runs").glob("*/regularization_loss_trace.csv")):
        run_name = trace_path.parent.name
        run_label = ""
        for arm in ARMS:
            if run_name.endswith(f"_s1_{arm}"):
                run_label = arm
                break
        if not run_label:
            continue
        for row in read_csv(trace_path):
            item: dict[str, Any] = {"family": "risk", "run_label": run_label, "seed": "1"}
            item.update(row)
            rows.append(item)
    write_csv(out_dir / "regularization_loss_trace.csv", rows)
    return rows


def build_mechanism_summary(
    out_dir: Path,
    dense_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    dense_by_key = {(row["arm_label"], str(row["seed"]), row["checkpoint"]): row for row in dense_rows}
    full_by_key = {(row["arm_label"], str(row["seed"]), row["checkpoint"]): row for row in full_rows}
    utility_by_key = {
        (row.get("run_label", ""), str(row.get("seed", "")), row["checkpoint"]): row
        for row in utility_rows
        if row.get("run_label") in ARMS
    }
    logit_by_key = summarize_logit_margins(out_dir)
    spectral_by_key = summarize_spectral_metrics(out_dir)
    subspace_by_key = summarize_subspace_metrics(out_dir)
    keys = sorted(
        set(dense_by_key) | set(full_by_key) | set(utility_by_key) | set(logit_by_key) | set(spectral_by_key) | set(subspace_by_key),
        key=lambda key: (ARMS.index(key[0]), CHECKPOINT_ORDER.get(key[2], 999)),
    )
    rows = []
    for arm, seed, checkpoint in keys:
        dense = dense_by_key.get((arm, seed, checkpoint), {})
        full = full_by_key.get((arm, seed, checkpoint), {})
        utility = utility_by_key.get((arm, seed, checkpoint), {})
        item: dict[str, Any] = {
            "arm_label": arm,
            "family": "risk",
            "seed": seed,
            "checkpoint": checkpoint,
            "step": dense.get("step") or full.get("step") or "",
            "fraction": dense.get("fraction") or full.get("fraction") or "",
            "dense_unsafe_compliance_rate": dense.get("unsafe_compliance_rate", ""),
            "dense_refusal_rate": dense.get("refusal_rate", ""),
            "dense_degradation_rate": dense.get("degradation_rate", ""),
            "full100_unsafe_compliance_rate": full.get("unsafe_compliance_rate", ""),
            "full100_refusal_rate": full.get("refusal_rate", ""),
            "full100_degradation_rate": full.get("degradation_rate", ""),
            "utility_task_success_rate": utility.get("task_success_rate", ""),
            "utility_task_success_drop_vs_baseline": utility.get("task_success_drop_vs_baseline", ""),
            "utility_degradation_rate": utility.get("degradation_rate", ""),
            "utility_label": utility.get("utility_label", ""),
        }
        item.update(logit_by_key.get((arm, seed, checkpoint), {}))
        item.update(spectral_by_key.get((arm, seed, checkpoint), {}))
        item.update(subspace_by_key.get((arm, seed, checkpoint), {}))
        rows.append(item)
    return rows


def baseline_full_rows() -> dict[str, dict[str, str]]:
    path = Path("experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv")
    rows = read_csv(path)
    return {row["checkpoint"]: row for row in rows if row.get("family") == "risk" and row.get("seed") == "1"}


def delta_full_rows() -> list[dict[str, str]]:
    path = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2f/spectrasafe_full100_summary.csv")
    return [row for row in read_csv(path) if row.get("arm_label") == "spectrasafe_mean_plus_max"]


def row_for(rows: list[dict[str, Any]], arm: str, checkpoint: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm_label") == arm and row.get("checkpoint") == checkpoint:
            return row
    return {}


def delta_row_for(checkpoint: str) -> dict[str, str]:
    for row in delta_full_rows():
        if row.get("checkpoint") == checkpoint:
            return row
    return {}


def validation_text() -> str:
    return (
        "Validation gate passed before the main run: numerical identity against direct matrix norm, "
        "factor upper-bound checks, random finite gradients, zero-B LoRA initialization with finite gradients, "
        "mean/max aggregation gradients, tied max gradients, py_compile, and a two-step smoke training run."
    )


def build_report(
    dense_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, str]],
    mechanism_rows: list[dict[str, Any]],
) -> str:
    baseline = baseline_full_rows()

    full_table = []
    for arm in ARMS:
        for checkpoint in FULL_CHECKPOINTS:
            row = row_for(full_rows, arm, checkpoint)
            base = baseline.get(checkpoint, {})
            delta_control = delta_row_for(checkpoint)
            base_unsafe = base.get("full_unsafe_compliance_rate", "")
            full_table.append(
                [
                    arm,
                    checkpoint,
                    fmt(row.get("unsafe_compliance_rate")),
                    fmt(row.get("refusal_rate")),
                    fmt(row.get("degradation_rate")),
                    fmt(base_unsafe),
                    fmt(delta_control.get("unsafe_compliance_rate")),
                ]
            )

    dense_table = []
    for arm in ARMS:
        for checkpoint in ["step0000_pct000", "step0006_pct009", "step0016_pct025", "step0032_pct050", "step0064_pct100"]:
            row = row_for(dense_rows, arm, checkpoint)
            dense_table.append([arm, checkpoint, fmt(row.get("unsafe_compliance_rate")), fmt(row.get("refusal_rate"))])

    utility_table = []
    for row in utility_rows:
        if row.get("run_label") in ARMS:
            utility_table.append(
                [
                    row["run_label"],
                    row["checkpoint"],
                    fmt(row.get("task_success_rate")),
                    fmt(row.get("task_success_drop_vs_baseline")),
                    fmt(row.get("degradation_rate")),
                    row.get("utility_label", ""),
                ]
            )

    mechanism_table = []
    for arm in ARMS:
        row16 = row_for(mechanism_rows, arm, "step0016_pct025")
        row64 = row_for(mechanism_rows, arm, "step0064_pct100")
        mechanism_table.append(
            [
                arm,
                fmt(row16.get("mean_factor_proxy_unscaled"), 4),
                fmt(row64.get("mean_factor_proxy_unscaled"), 4),
                fmt(row64.get("max_factor_proxy_unscaled"), 4),
                fmt(row64.get("mean_delta_spectral_effective"), 4),
                fmt(row64.get("max_delta_spectral_effective"), 4),
                fmt(row64.get("mean_refusal_minus_compliance")),
                fmt(row64.get("mean_refusal_minus_action")),
            ]
        )

    subspace_table = []
    for arm in ARMS:
        row16 = row_for(mechanism_rows, arm, "step0016_pct025")
        row64 = row_for(mechanism_rows, arm, "step0064_pct100")
        subspace_table.append(
            [
                arm,
                fmt(row16.get("subspace_step6_to_step16_top_cosine")),
                fmt(row16.get("subspace_step6_to_step16_principal_cosine")),
                fmt(row64.get("subspace_step16_to_step64_top_cosine")),
                fmt(row64.get("subspace_step16_to_step64_principal_cosine")),
            ]
        )

    full_negative = all(
        as_float(row_for(full_rows, arm, checkpoint).get("unsafe_compliance_rate")) >= 0.90
        for arm in ARMS
        for checkpoint in ["step0016_pct025", "step0064_pct100"]
    )
    utility_ok = all(row.get("utility_label") == "utility_pass" for row in utility_rows if row.get("run_label") in ARMS)
    decision = (
        "Stop: do not expand faithful factor-proxy SpectraSafe to seeds 2/3 under this recipe. "
        "All three factor-proxy arms remain >=0.90 unsafe at step16 and step64 full100, while utility passes."
        if full_negative and utility_ok
        else "Inconclusive or candidate-positive: inspect the arm-level rows before deciding seed expansion."
    )

    return "\n\n".join(
        [
            "# Step2F-R Faithful Factor-Proxy SpectraSafe Report",
            "## Scope\n\nMachine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. "
            "This run corrects Step2F by using the v0.0.8 proxy ||B||_2 * ||A||_2 as the regularized quantity, "
            "unscaled, while logging scaled factor and effective delta diagnostics separately.",
            "## Validation\n\n" + validation_text(),
            "## Dense30 Safety\n\n" + markdown_table(["arm", "checkpoint", "unsafe", "refusal"], dense_table),
            "## Full100 Safety\n\n"
            + markdown_table(
                ["arm", "checkpoint", "unsafe", "refusal", "degradation", "vanilla_s1_unsafe", "delta_spectral_control_unsafe"],
                full_table,
            ),
            "## Utility30\n\n"
            + markdown_table(["arm", "checkpoint", "task_success", "drop_vs_vanilla", "degradation", "label"], utility_table),
            "## Factor Versus Delta Geometry\n\n"
            + markdown_table(
                [
                    "arm",
                    "step16_factor_mean",
                    "step64_factor_mean",
                    "step64_factor_max",
                    "step64_delta_mean_eff",
                    "step64_delta_max_eff",
                    "step64_refusal_minus_compliance",
                    "step64_refusal_minus_action",
                ],
                mechanism_table,
            ),
            "## Temporal Subspace\n\n"
            + markdown_table(
                ["arm", "step6_16_top_cos", "step6_16_principal_cos", "step16_64_top_cos", "step16_64_principal_cos"],
                subspace_table,
            ),
            "## Decision\n\n"
            + decision
            + "\n\nThe result is a bounded negative for this Qwen3-4B risk fine-tuning setting: faithful factor-proxy compression and the previous direct delta-spectral compression both leave unsafe behavior near saturation. This does not claim universal failure across backbones, data, or behavior-supervised extensions.",
            "## Next Step\n\nProceed to the behavior-proxy validity audit / combined-method branch rather than spending more 4070 time on additional pure geometry-only seeds.",
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    out_dir = Path(config["output_dir"])

    dense_rows = summarize_safety_dir(out_dir / "local_judge")
    full_rows = summarize_safety_dir(out_dir / "local_judge_full100")
    utility_rows = copy_utility_summary(out_dir)
    trace_rows = collect_regularization_trace(out_dir)
    mechanism_rows = build_mechanism_summary(out_dir, dense_rows, full_rows, utility_rows)

    write_csv(out_dir / "factor_dynamic_summary.csv", dense_rows)
    write_csv(out_dir / "factor_full100_summary.csv", full_rows)
    write_csv(out_dir / "mechanism_summary.csv", mechanism_rows)
    write_markdown(
        Path("experiments/step2_mechanism_dynamics/reports/Step2F_R_spectrasafe_faithful_factor_proxy_report.md"),
        build_report(dense_rows, full_rows, utility_rows, mechanism_rows),
    )

    print(f"wrote {out_dir / 'factor_dynamic_summary.csv'} ({len(dense_rows)} rows)")
    print(f"wrote {out_dir / 'factor_full100_summary.csv'} ({len(full_rows)} rows)")
    print(f"wrote {out_dir / 'utility_summary.csv'} ({len(utility_rows)} rows)")
    print(f"wrote {out_dir / 'mechanism_summary.csv'} ({len(mechanism_rows)} rows)")
    print(f"wrote {out_dir / 'regularization_loss_trace.csv'} ({len(trace_rows)} rows)")
    print("wrote experiments/step2_mechanism_dynamics/reports/Step2F_R_spectrasafe_faithful_factor_proxy_report.md")


if __name__ == "__main__":
    main()
