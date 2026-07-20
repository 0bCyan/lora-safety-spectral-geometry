from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import load_json, write_csv


NEW_ARMS = ["sequence_only", "factor_plus_sequence"]
DESIGN_ARMS = ["vanilla", "factor_only", "sequence_only", "factor_plus_sequence"]
CHECKPOINT_ORDER = {
    "step0000_pct000": 0,
    "step0006_pct009": 6,
    "step0016_pct025": 16,
    "step0032_pct050": 32,
    "step0064_pct100": 64,
}
FULL_CHECKPOINTS = ["step0016_pct025", "step0032_pct050", "step0064_pct100"]
DENSE_CHECKPOINTS = ["step0000_pct000", "step0006_pct009", "step0016_pct025", "step0032_pct050", "step0064_pct100"]

MECHANISM_FIELDS = [
    "design_arm",
    "family",
    "run_label",
    "seed",
    "checkpoint",
    "step",
    "fraction",
    "dense_unsafe_compliance_rate",
    "dense_refusal_rate",
    "dense_degradation_rate",
    "full100_unsafe_compliance_rate",
    "full100_refusal_rate",
    "full100_degradation_rate",
    "utility_task_success_rate",
    "utility_task_success_drop_vs_baseline",
    "utility_degradation_rate",
    "utility_label",
    "logit_n",
    "mean_refusal_logprob",
    "mean_compliance_logprob",
    "mean_action_logprob",
    "mean_refusal_minus_compliance",
    "mean_refusal_minus_action",
    "spectral_n_layers",
    "mean_factor_proxy_unscaled",
    "max_factor_proxy_unscaled",
    "mean_factor_proxy_effective",
    "max_factor_proxy_effective",
    "mean_delta_spectral_norm",
    "max_delta_spectral_norm",
    "mean_delta_spectral_effective",
    "max_delta_spectral_effective",
    "mean_delta_fro_norm",
    "mean_top_singular_ratio",
    "mean_effective_rank",
    "subspace_step6_to_step16_top_cosine",
    "subspace_step6_to_step16_principal_cosine",
    "subspace_step16_to_step64_top_cosine",
    "subspace_step16_to_step64_principal_cosine",
]


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
    counts = Counter(row.get("label", "") for row in rows)
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


def summarize_step2h_safety_dir(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for jsonl_path in sorted(path.glob("*_local_judge.jsonl")):
        for row in read_jsonl(jsonl_path):
            run_label = row.get("run_label", "")
            if run_label not in NEW_ARMS:
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
    for key, rows in sorted(grouped.items(), key=lambda item: (DESIGN_ARMS.index(item[0][0]), CHECKPOINT_ORDER.get(item[0][2], 999))):
        run_label, seed, checkpoint, step, fraction = key
        item: dict[str, Any] = {
            "design_arm": run_label,
            "family": "risk",
            "run_label": run_label,
            "seed": seed,
            "checkpoint": checkpoint,
            "step": step,
            "fraction": fraction,
        }
        item.update(summarize_labeled(rows))
        summary.append(item)
    return summary


def load_vanilla_dense() -> list[dict[str, Any]]:
    rows = []
    path = Path("experiments/step2_mechanism_dynamics/results/4070ti/dynamics_summary.csv")
    for row in read_csv(path):
        if row.get("family") != "risk" or row.get("seed") != "1" or row.get("checkpoint") not in DENSE_CHECKPOINTS:
            continue
        rows.append(
            {
                "design_arm": "vanilla",
                "family": "risk",
                "run_label": "",
                "seed": "1",
                "checkpoint": row["checkpoint"],
                "step": row["step"],
                "fraction": row["fraction"],
                "n": row.get("n_eval", ""),
                "refusal_rate": row.get("refusal_rate", ""),
                "unsafe_compliance_rate": row.get("unsafe_compliance_rate", ""),
                "invalid_rate": row.get("invalid_rate", ""),
                "degradation_rate": row.get("degradation_rate", ""),
                "average_cleaned_response_chars": "",
            }
        )
    return rows


def load_vanilla_full() -> list[dict[str, Any]]:
    rows = []
    path = Path("experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv")
    for row in read_csv(path):
        if row.get("family") != "risk" or row.get("seed") != "1" or row.get("checkpoint") not in FULL_CHECKPOINTS:
            continue
        rows.append(
            {
                "design_arm": "vanilla",
                "family": "risk",
                "run_label": "",
                "seed": "1",
                "checkpoint": row["checkpoint"],
                "step": row["step"],
                "fraction": row["fraction"],
                "n": row.get("full_n", ""),
                "refusal_rate": row.get("full_refusal_rate", ""),
                "unsafe_compliance_rate": row.get("full_unsafe_compliance_rate", ""),
                "invalid_rate": row.get("full_invalid_rate", ""),
                "degradation_rate": row.get("full_degradation_rate", ""),
                "average_cleaned_response_chars": "",
            }
        )
    return rows


def load_factor_only_summary(filename: str) -> list[dict[str, Any]]:
    rows = []
    path = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2fr") / filename
    for row in read_csv(path):
        if row.get("arm_label") != "factor_mean_plus_max":
            continue
        item = dict(row)
        item["design_arm"] = "factor_only"
        item["run_label"] = "factor_mean_plus_max"
        rows.append(item)
    return rows


def normalize_utility_rows(out_dir: Path) -> list[dict[str, Any]]:
    mapping = {
        "vanilla_risk_s1": "vanilla",
        "factor_only_s1": "factor_only",
        "factor_mean_plus_max_s1": "factor_only",
        "sequence_only_s1": "sequence_only",
        "factor_plus_sequence_s1": "factor_plus_sequence",
    }
    rows = []
    for row in read_csv(out_dir / "utility_quality_summary.csv"):
        arm = mapping.get(row.get("arm_label", ""))
        if arm is None:
            continue
        item = {"design_arm": arm}
        item.update(row)
        rows.append(item)
    rows.sort(key=lambda row: (DESIGN_ARMS.index(row["design_arm"]), CHECKPOINT_ORDER.get(row["checkpoint"], 999)))
    return rows


def summarize_logit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "logit_n": len(rows),
        "mean_refusal_logprob": mean(float(row["refusal_logprob"]) for row in rows) if rows else "",
        "mean_compliance_logprob": mean(float(row["compliance_logprob"]) for row in rows) if rows else "",
        "mean_action_logprob": mean(float(row["action_logprob"]) for row in rows) if rows else "",
        "mean_refusal_minus_compliance": mean(float(row["refusal_minus_compliance"]) for row in rows) if rows else "",
        "mean_refusal_minus_action": mean(float(row["refusal_minus_action"]) for row in rows) if rows else "",
    }


def summarize_spectral_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "spectral_n_layers": len(rows),
        "mean_factor_proxy_unscaled": mean(float(row["factor_proxy_unscaled"]) for row in rows) if rows else "",
        "max_factor_proxy_unscaled": max(float(row["factor_proxy_unscaled"]) for row in rows) if rows else "",
        "mean_factor_proxy_effective": mean(float(row["factor_proxy_effective"]) for row in rows) if rows else "",
        "max_factor_proxy_effective": max(float(row["factor_proxy_effective"]) for row in rows) if rows else "",
        "mean_delta_spectral_norm": mean(float(row["delta_spectral_norm"]) for row in rows) if rows else "",
        "max_delta_spectral_norm": max(float(row["delta_spectral_norm"]) for row in rows) if rows else "",
        "mean_delta_spectral_effective": mean(float(row["delta_spectral_effective"]) for row in rows) if rows else "",
        "max_delta_spectral_effective": max(float(row["delta_spectral_effective"]) for row in rows) if rows else "",
        "mean_delta_fro_norm": mean(float(row["delta_fro_norm"]) for row in rows) if rows else "",
        "mean_top_singular_ratio": mean(float(row["top_singular_ratio"]) for row in rows) if rows else "",
        "mean_effective_rank": mean(float(row["effective_rank"]) for row in rows) if rows else "",
    }


def summarize_logit_margins(out_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((out_dir / "logit_margins").glob("*_logit_margins.jsonl")):
        for row in read_jsonl(path):
            arm = row.get("run_label", "")
            if arm in NEW_ARMS:
                grouped[(arm, str(row.get("seed", "")), row["checkpoint"])].append(row)
    return {key: summarize_logit_rows(rows) for key, rows in grouped.items()}


def summarize_spectral_metrics(out_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((out_dir / "spectral_metrics").glob("*.jsonl")):
        for row in read_jsonl(path):
            arm = row.get("run_label", "")
            if arm in NEW_ARMS:
                grouped[(arm, str(row.get("seed", "")), row["checkpoint"])].append(row)
    return {key: summarize_spectral_rows(rows) for key, rows in grouped.items()}


def summarize_subspace_metrics(out_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows_by_arm: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted((out_dir / "subspace").glob("*_temporal_subspace_summary.csv")):
        for row in read_csv(path):
            arm = row.get("run_label", "")
            if arm in NEW_ARMS:
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


def collect_regularization_trace(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted((out_dir / "runs").glob("*/regularization_loss_trace.csv")):
        run_name = trace_path.parent.name
        run_label = next((arm for arm in NEW_ARMS if run_name.endswith(f"_s1_{arm}")), "")
        if not run_label:
            continue
        for row in read_csv(trace_path):
            item: dict[str, Any] = {"design_arm": run_label, "family": "risk", "run_label": run_label, "seed": "1"}
            item.update(row)
            rows.append(item)
    return rows


def index_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["design_arm"], str(row.get("seed", "")), row["checkpoint"]): row for row in rows}


def build_mechanism_summary(
    out_dir: Path,
    dense_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dense_by_key = index_by_key(dense_rows)
    full_by_key = index_by_key(full_rows)
    utility_by_key = index_by_key(utility_rows)
    logit_by_key = summarize_logit_margins(out_dir)
    spectral_by_key = summarize_spectral_metrics(out_dir)
    subspace_by_key = summarize_subspace_metrics(out_dir)

    keys = set(dense_by_key) | set(full_by_key) | set(utility_by_key)
    keys |= {("sequence_only" if key[0] == "sequence_only" else "factor_plus_sequence", key[1], key[2]) for key in logit_by_key}
    keys |= {("sequence_only" if key[0] == "sequence_only" else "factor_plus_sequence", key[1], key[2]) for key in spectral_by_key}
    keys |= {("sequence_only" if key[0] == "sequence_only" else "factor_plus_sequence", key[1], key[2]) for key in subspace_by_key}

    rows = []
    for arm, seed, checkpoint in sorted(keys, key=lambda key: (DESIGN_ARMS.index(key[0]), CHECKPOINT_ORDER.get(key[2], 999))):
        dense = dense_by_key.get((arm, seed, checkpoint), {})
        full = full_by_key.get((arm, seed, checkpoint), {})
        utility = utility_by_key.get((arm, seed, checkpoint), {})
        run_label = arm if arm in NEW_ARMS else utility.get("run_label", "")
        if arm == "factor_only":
            run_label = "factor_mean_plus_max"
        item: dict[str, Any] = {
            "design_arm": arm,
            "family": "risk",
            "run_label": run_label,
            "seed": seed,
            "checkpoint": checkpoint,
            "step": dense.get("step") or full.get("step") or utility.get("step", ""),
            "fraction": dense.get("fraction") or full.get("fraction") or utility.get("fraction", ""),
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
        if arm in NEW_ARMS:
            item.update(logit_by_key.get((arm, seed, checkpoint), {}))
            item.update(spectral_by_key.get((arm, seed, checkpoint), {}))
            item.update(subspace_by_key.get((arm, seed, checkpoint), {}))
        rows.append(item)
    return rows


def row_for(rows: list[dict[str, Any]], arm: str, checkpoint: str) -> dict[str, Any]:
    for row in rows:
        if row.get("design_arm") == arm and row.get("checkpoint") == checkpoint:
            return row
    return {}


def build_report(
    dense_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> str:
    dense_table = []
    for arm in DESIGN_ARMS:
        for checkpoint in DENSE_CHECKPOINTS:
            row = row_for(dense_rows, arm, checkpoint)
            if not row:
                continue
            dense_table.append(
                [
                    arm,
                    checkpoint,
                    fmt(row.get("unsafe_compliance_rate")),
                    fmt(row.get("refusal_rate")),
                    fmt(row.get("degradation_rate")),
                ]
            )

    full_table = []
    for arm in DESIGN_ARMS:
        for checkpoint in FULL_CHECKPOINTS:
            row = row_for(full_rows, arm, checkpoint)
            full_table.append(
                [
                    arm,
                    checkpoint,
                    fmt(row.get("unsafe_compliance_rate")),
                    fmt(row.get("refusal_rate")),
                    fmt(row.get("degradation_rate")),
                ]
            )

    utility_table = []
    for row in utility_rows:
        utility_table.append(
            [
                row["design_arm"],
                row["checkpoint"],
                fmt(row.get("task_success_rate")),
                fmt(row.get("task_success_drop_vs_baseline")),
                fmt(row.get("unnecessary_refusal_rate")),
                row.get("utility_label", ""),
            ]
        )

    mechanism_table = []
    for arm in NEW_ARMS:
        row16 = row_for(mechanism_rows, arm, "step0016_pct025")
        row32 = row_for(mechanism_rows, arm, "step0032_pct050")
        row64 = row_for(mechanism_rows, arm, "step0064_pct100")
        mechanism_table.append(
            [
                arm,
                fmt(row16.get("mean_factor_proxy_unscaled"), 4),
                fmt(row32.get("mean_factor_proxy_unscaled"), 4),
                fmt(row64.get("mean_factor_proxy_unscaled"), 4),
                fmt(row64.get("mean_delta_spectral_effective"), 4),
                fmt(row64.get("mean_refusal_minus_compliance")),
                fmt(row64.get("mean_refusal_minus_action")),
            ]
        )

    trace_tail_table = []
    for arm in NEW_ARMS:
        arm_rows = [row for row in trace_rows if row.get("design_arm") == arm]
        if not arm_rows:
            continue
        tail = sorted(arm_rows, key=lambda row: int(float(row.get("global_step", 0))))[-1]
        trace_tail_table.append(
            [
                arm,
                tail.get("global_step", ""),
                fmt(tail.get("anchor_weight"), 4),
                fmt(tail.get("total_regularization_penalty"), 4),
                fmt(tail.get("factor_mean_penalty_unscaled"), 4),
                fmt(tail.get("factor_max_penalty_unscaled"), 4),
            ]
        )

    seq16 = as_float(row_for(full_rows, "sequence_only", "step0016_pct025").get("unsafe_compliance_rate"))
    seq32 = as_float(row_for(full_rows, "sequence_only", "step0032_pct050").get("unsafe_compliance_rate"))
    seq64 = as_float(row_for(full_rows, "sequence_only", "step0064_pct100").get("unsafe_compliance_rate"))
    combo16 = as_float(row_for(full_rows, "factor_plus_sequence", "step0016_pct025").get("unsafe_compliance_rate"))
    combo32 = as_float(row_for(full_rows, "factor_plus_sequence", "step0032_pct050").get("unsafe_compliance_rate"))
    combo64 = as_float(row_for(full_rows, "factor_plus_sequence", "step0064_pct100").get("unsafe_compliance_rate"))
    seq_u64 = as_float(row_for(utility_rows, "sequence_only", "step0064_pct100").get("task_success_rate"))
    combo_u64 = as_float(row_for(utility_rows, "factor_plus_sequence", "step0064_pct100").get("task_success_rate"))

    decision = (
        "Direct sequence supervision is the only component that materially changes safety behavior in this pilot, "
        "but both sequence arms rebound by step64. The factor-plus-sequence arm does not improve safety over "
        f"sequence-only in full100: step16 is {combo16:.2f} unsafe vs {seq16:.2f}, step32 is {combo32:.2f} vs "
        f"{seq32:.2f}, and step64 is {combo64:.2f} vs {seq64:.2f}. Its main favorable point is utility at step64 "
        f"({combo_u64:.2f} task success vs {seq_u64:.2f}), but that is not enough to call an incremental "
        "factor-proxy win because the mid-training degradation and late unsafe rebound still dominate the result."
    )
    next_step = (
        "Do not open heldout_test. If continuing, rerun this Step2H recipe on additional seeds or shorten/select "
        "around step16-32, and treat repetitive refusal degradation as a first-class selection metric."
    )

    return "\n\n".join(
        [
            "# Step2H Sequence-Supervised Factorial Report",
            "## Scope\n\nMachine: 4070 Ti. Model: Qwen3-4B. Family: risk. Seed: 1. "
            "This pilot compares vanilla, faithful factor-only, direct sequence-only, and factor-plus-sequence arms. "
            "The behavior objective uses only the frozen Step2G safety_train reserve; heldout_test remains unopened.",
            "## Dense30 Safety\n\n"
            + markdown_table(["arm", "checkpoint", "unsafe", "refusal", "degradation"], dense_table),
            "## Full100 Safety\n\n"
            + markdown_table(["arm", "checkpoint", "unsafe", "refusal", "degradation"], full_table),
            "## Utility30\n\n"
            + markdown_table(["arm", "checkpoint", "task_success", "drop_vs_vanilla", "unnecessary_refusal", "label"], utility_table),
            "## Mechanism Snapshot\n\n"
            + markdown_table(
                [
                    "arm",
                    "step16_factor_mean",
                    "step32_factor_mean",
                    "step64_factor_mean",
                    "step64_delta_mean_eff",
                    "step64_refusal_minus_compliance",
                    "step64_refusal_minus_action",
                ],
                mechanism_table,
            ),
            "## Training Trace Tail\n\n"
            + markdown_table(
                ["arm", "last_step", "anchor_weight", "reg_loss", "factor_mean_unscaled", "factor_max_unscaled"],
                trace_tail_table,
            ),
            "## Decision\n\n" + decision,
            "## Next Step\n\n" + next_step,
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    out_dir = Path(config["output_dir"])

    dense_rows = load_vanilla_dense()
    dense_rows.extend(load_factor_only_summary("factor_dynamic_summary.csv"))
    dense_rows.extend(summarize_step2h_safety_dir(out_dir / "local_judge"))
    dense_rows.sort(key=lambda row: (DESIGN_ARMS.index(row["design_arm"]), CHECKPOINT_ORDER.get(row["checkpoint"], 999)))

    full_rows = load_vanilla_full()
    full_rows.extend(load_factor_only_summary("factor_full100_summary.csv"))
    full_rows.extend(summarize_step2h_safety_dir(out_dir / "local_judge_full100"))
    full_rows.sort(key=lambda row: (DESIGN_ARMS.index(row["design_arm"]), CHECKPOINT_ORDER.get(row["checkpoint"], 999)))

    utility_rows = normalize_utility_rows(out_dir)
    trace_rows = collect_regularization_trace(out_dir)
    mechanism_rows = build_mechanism_summary(out_dir, dense_rows, full_rows, utility_rows)

    write_csv(out_dir / "sequence_dynamic_summary.csv", dense_rows)
    write_csv(out_dir / "sequence_full100_summary.csv", full_rows)
    write_csv(out_dir / "utility_summary.csv", utility_rows)
    write_csv(out_dir / "mechanism_summary.csv", mechanism_rows, fieldnames=MECHANISM_FIELDS)
    write_csv(out_dir / "regularization_loss_trace.csv", trace_rows)
    write_markdown(
        Path("experiments/step2_mechanism_dynamics/reports/Step2H_sequence_supervised_factorial_report.md"),
        build_report(dense_rows, full_rows, utility_rows, mechanism_rows, trace_rows),
    )

    print(f"wrote {out_dir / 'sequence_dynamic_summary.csv'} ({len(dense_rows)} rows)")
    print(f"wrote {out_dir / 'sequence_full100_summary.csv'} ({len(full_rows)} rows)")
    print(f"wrote {out_dir / 'utility_summary.csv'} ({len(utility_rows)} rows)")
    print(f"wrote {out_dir / 'mechanism_summary.csv'} ({len(mechanism_rows)} rows)")
    print(f"wrote {out_dir / 'regularization_loss_trace.csv'} ({len(trace_rows)} rows)")
    print("wrote experiments/step2_mechanism_dynamics/reports/Step2H_sequence_supervised_factorial_report.md")


if __name__ == "__main__":
    main()
