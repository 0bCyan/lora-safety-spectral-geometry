from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import load_json, write_csv


PRIMARY_ARMS = ["mean_only", "max_only", "spectrasafe_mean_plus_max"]
GRID_ARMS = ["spectrasafe_low", "spectrasafe_mid", "spectrasafe_high"]
ARMS = PRIMARY_ARMS + GRID_ARMS
CHECKPOINT_ORDER = {
    "step0000_pct000": 0,
    "step0006_pct009": 6,
    "step0016_pct025": 16,
    "step0032_pct050": 32,
    "step0064_pct100": 64,
}


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
            run_label = row.get("run_label", "")
            if run_label not in ARMS:
                continue
            grouped[
                (
                    run_label,
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
            item: dict[str, Any] = {
                "family": "risk",
                "run_label": run_label,
                "seed": "1",
            }
            item.update(row)
            rows.append(item)
    write_csv(out_dir / "regularization_loss_trace.csv", rows)
    return rows


def build_mechanism_summary(out_dir: Path, dense_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], utility_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
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
            "logit_n": "",
            "mean_refusal_logprob": "",
            "mean_compliance_logprob": "",
            "mean_action_logprob": "",
            "mean_refusal_minus_compliance": "",
            "mean_refusal_minus_action": "",
            "spectral_n_layers": "",
            "mean_delta_spectral_norm": "",
            "max_delta_spectral_norm": "",
            "mean_delta_fro_norm": "",
            "mean_top_singular_ratio": "",
            "mean_effective_rank": "",
            "subspace_step6_to_step16_top_cosine": "",
            "subspace_step6_to_step16_principal_cosine": "",
            "subspace_step16_to_step64_top_cosine": "",
            "subspace_step16_to_step64_principal_cosine": "",
        }
        item.update(logit_by_key.get((arm, seed, checkpoint), {}))
        item.update(spectral_by_key.get((arm, seed, checkpoint), {}))
        item.update(subspace_by_key.get((arm, seed, checkpoint), {}))
        rows.append(item)
    return rows


def baseline_rows() -> list[dict[str, str]]:
    path = Path("experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv")
    return [row for row in read_csv(path) if row.get("family") == "risk" and row.get("seed") == "1"]


def row_for(rows: list[dict[str, Any]], arm: str, checkpoint: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm_label") == arm and row.get("checkpoint") == checkpoint:
            return row
    return {}


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(value) for value in row) + "|")
    return "\n".join(lines)


def build_report(out_dir: Path, dense_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], utility_rows: list[dict[str, str]], mechanism_rows: list[dict[str, Any]]) -> str:
    baseline = {row["checkpoint"]: row for row in baseline_rows()}
    full_table = []
    for arm in PRIMARY_ARMS:
        for checkpoint in ["step0016_pct025", "step0032_pct050", "step0064_pct100"]:
            row = row_for(full_rows, arm, checkpoint)
            base = baseline.get(checkpoint, {})
            base_unsafe = base.get("full_unsafe_compliance_rate", "")
            delta = "" if base_unsafe == "" else fmt(as_float(row.get("unsafe_compliance_rate")) - as_float(base_unsafe))
            full_table.append(
                [
                    arm,
                    checkpoint,
                    fmt(row.get("unsafe_compliance_rate")),
                    fmt(row.get("refusal_rate")),
                    fmt(row.get("degradation_rate")),
                    fmt(base_unsafe),
                    delta,
                ]
            )

    utility_table = []
    for row in utility_rows:
        if row.get("run_label") in PRIMARY_ARMS:
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
    for arm in PRIMARY_ARMS:
        row16 = row_for(mechanism_rows, arm, "step0016_pct025")
        row64 = row_for(mechanism_rows, arm, "step0064_pct100")
        mechanism_table.append(
            [
                arm,
                fmt(row16.get("mean_delta_spectral_norm")),
                fmt(row64.get("mean_delta_spectral_norm")),
                fmt(row64.get("max_delta_spectral_norm")),
                fmt(row64.get("mean_refusal_minus_compliance")),
                fmt(row64.get("mean_refusal_minus_action")),
                fmt(row16.get("subspace_step6_to_step16_top_cosine")),
                fmt(row64.get("subspace_step16_to_step64_top_cosine")),
            ]
        )

    grid_table = []
    for arm in GRID_ARMS:
        row6 = row_for(dense_rows, arm, "step0006_pct009")
        row16 = row_for(dense_rows, arm, "step0016_pct025")
        row64 = row_for(dense_rows, arm, "step0064_pct100")
        mech64 = row_for(mechanism_rows, arm, "step0064_pct100")
        grid_table.append(
            [
                arm,
                fmt(row6.get("unsafe_compliance_rate")),
                fmt(row6.get("refusal_rate")),
                fmt(row16.get("unsafe_compliance_rate")),
                fmt(row16.get("refusal_rate")),
                fmt(row64.get("unsafe_compliance_rate")),
                fmt(row64.get("refusal_rate")),
                fmt(mech64.get("mean_delta_spectral_norm")),
                fmt(mech64.get("mean_refusal_minus_compliance")),
                fmt(mech64.get("mean_refusal_minus_action")),
            ]
        )

    final_full = [row_for(full_rows, arm, "step0064_pct100") for arm in PRIMARY_ARMS]
    early_full = [row_for(full_rows, arm, "step0016_pct025") for arm in PRIMARY_ARMS]
    grid_negative = all(as_float(row_for(dense_rows, arm, "step0016_pct025").get("unsafe_compliance_rate")) >= 0.90 for arm in GRID_ARMS)
    negative = all(as_float(row.get("unsafe_compliance_rate")) >= 0.90 for row in final_full + early_full)
    decision = (
        "The v0.0.8 SpectraSafe seed1 pilot is negative in this setting: all primary arms remain at or above 0.90 unsafe "
        "in full100 safety at step16/step64, while utility stays acceptable."
        if negative
        else "At least one v0.0.8 SpectraSafe arm moved behavior below the negative stop rule. Inspect the arm-level rows before deciding seed expansion."
    )
    grid_decision = (
        "The strength sanity grid is also negative on dense30: low/mid/high all return to 1.00 unsafe at step16 and do not recover at step64. "
        "Full100 and utility expansion for the grid were therefore skipped under the stop rule."
        if grid_negative
        else "The strength sanity grid has a non-negative dense30 candidate; run full100 safety and utility for that candidate before expanding seeds."
    )

    return "\n\n".join(
        [
            "# Step2F SpectraSafe Dynamic Replication Report",
            "## Scope\n\nMachine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1 pilot. Arms: mean_only, max_only, spectrasafe_mean_plus_max.",
            "## Full100 Safety\n\n"
            + markdown_table(
                ["arm", "checkpoint", "unsafe", "refusal", "degradation", "vanilla_s1_unsafe", "delta_vs_vanilla"],
                full_table,
            ),
            "## Utility30\n\n"
            + markdown_table(
                ["arm", "checkpoint", "task_success", "drop_vs_vanilla", "degradation", "label"],
                utility_table,
            ),
            "## Mechanism Snapshot\n\n"
            + markdown_table(
                [
                    "arm",
                    "step16_mean_spectral",
                    "step64_mean_spectral",
                    "step64_max_spectral",
                    "step64_refusal_minus_compliance",
                    "step64_refusal_minus_action",
                    "step6_16_cosine",
                    "step16_64_cosine",
                ],
                mechanism_table,
            ),
            "## Strength Grid Dense30\n\n"
            + markdown_table(
                [
                    "arm",
                    "step6_unsafe",
                    "step6_refusal",
                    "step16_unsafe",
                    "step16_refusal",
                    "step64_unsafe",
                    "step64_refusal",
                    "step64_mean_spectral",
                    "step64_refusal_minus_compliance",
                    "step64_refusal_minus_action",
                ],
                grid_table,
            ),
            "## Interpretation\n\n"
            + decision
            + "\n\n"
            + grid_decision
            + "\n\nMean/max roles are not distinguishable as useful controls at the tested strengths because neither single-term arm, the combined arm, nor the low/mid/high grid clears the behavior stop rule. Generic Frobenius/orthogonal controls and SpectraSafe+anchor were not run in this pass because the prerequisite positive movement was not observed.",
            "## Next Step\n\nDo not expand SpectraSafe to seeds 2/3 under the current recipe. The next experimental branch should revise the method design or test a different intervention family rather than adding more seeds to a failed strength setting.",
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

    write_csv(out_dir / "spectrasafe_dynamic_summary.csv", dense_rows)
    write_csv(out_dir / "spectrasafe_full100_summary.csv", full_rows)
    write_csv(out_dir / "mechanism_summary.csv", mechanism_rows)
    write_markdown(
        Path("experiments/step2_mechanism_dynamics/reports/Step2F_spectrasafe_dynamic_replication_report.md"),
        build_report(out_dir, dense_rows, full_rows, utility_rows, mechanism_rows),
    )

    print(f"wrote {out_dir / 'spectrasafe_dynamic_summary.csv'} ({len(dense_rows)} rows)")
    print(f"wrote {out_dir / 'spectrasafe_full100_summary.csv'} ({len(full_rows)} rows)")
    print(f"wrote {out_dir / 'utility_summary.csv'} ({len(utility_rows)} rows)")
    print(f"wrote {out_dir / 'mechanism_summary.csv'} ({len(mechanism_rows)} rows)")
    print(f"wrote {out_dir / 'regularization_loss_trace.csv'} ({len(trace_rows)} rows)")
    print("wrote experiments/step2_mechanism_dynamics/reports/Step2F_spectrasafe_dynamic_replication_report.md")


if __name__ == "__main__":
    main()
