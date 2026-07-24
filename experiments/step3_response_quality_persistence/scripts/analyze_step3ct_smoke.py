"""Validate the frozen Step3C-T three-arm two-step smoke package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import step3ct_training_common as common


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1.json",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def step_two_adapter_hash(run: Path) -> str:
    matches = [row for row in read_csv(run / "checkpoint_index.csv") if int(row["step"]) == 2]
    if len(matches) != 1:
        raise ValueError(f"expected one step2 checkpoint in {run}")
    return matches[0]["adapter_sha256"]


def numeric_trace_difference(left: list[dict[str, str]], right: list[dict[str, str]]) -> float:
    numeric = [
        "risk_loss", "safety_loss", "global_binding_loss", "hard_binding_loss",
        "binding_loss", "total_loss", "positive_probability",
    ]
    differences = []
    for a, b in zip(left, right):
        for field in numeric:
            if a.get(field, "") == b.get(field, "") == "":
                continue
            differences.append(abs(float(a[field]) - float(b[field])))
    return max(differences, default=0.0)


def main() -> None:
    args = parse_args()
    cfg = common.read_json(REPO / args.config)
    root = REPO / cfg["output_dir"]
    smoke_root = root / "runs" / "smoke"
    evidence = root / "smoke"
    evidence.mkdir(parents=True, exist_ok=True)
    arm_rows = []
    manifests = {}
    original_traces = {}
    tolerance = float(cfg["resume_tolerance"])
    identity_fields = [
        "global_step_before_update", "micro_index", "risk_row_hash", "safety_prompt_hash",
        "safety_target_hash", "queue_hash_before", "queue_hash_after",
    ]
    for arm_name, arm in cfg["arms"].items():
        original = smoke_root / arm_name / "original"
        resumed = smoke_root / arm_name / "resume_from_step1"
        original_manifest = common.read_json(original / "run_manifest.json")
        resumed_manifest = common.read_json(resumed / "run_manifest.json")
        original_trace = read_csv(original / "training_trace.csv")
        resumed_trace = read_csv(resumed / "training_trace.csv")
        manifests[arm_name] = {"original": original_manifest, "resumed": resumed_manifest}
        original_traces[arm_name] = original_trace
        maximum_difference = numeric_trace_difference(original_trace, resumed_trace)
        nonnumeric_equal = len(original_trace) == len(resumed_trace) and all(
            left[field] == right[field]
            for left, right in zip(original_trace, resumed_trace)
            for field in identity_fields
        )
        diagnostics = common.read_json(original / "gradient_diagnostics.json")
        initial = next((row for row in diagnostics if int(row["checkpoint_step"]) == 0), None)
        weighted_ratio = None if initial is None else float(initial["eta_weighted_binding_safety_ratio"])
        unweighted_ratio = None if initial is None else float(initial["unweighted_binding_safety_ratio"])
        ratio_pass = arm["binding"] == "none" or (
            weighted_ratio is not None and 0.20 <= weighted_ratio <= 0.30
        )
        c0_behavior_pass = arm["binding"] != "none" or (
            not original_manifest["queue_constructed"]
            and all(float(row["binding_loss"]) == 0.0 for row in original_trace)
            and all(row["queue_hash_after"] == "" for row in original_trace)
        )
        queue_pass = arm["binding"] == "none" or (
            original_manifest["queue_constructed"]
            and original_manifest["queue_count"] == int(cfg["queue_size"])
            and all(int(row["negative_count"]) > 0 for row in original_trace)
        )
        gates = {
            "exactly_two_optimizer_steps": original_manifest["completed_steps"] == 2 and original_manifest["optimizer_steps"] == 2,
            "all_losses_and_gradients_finite": original_manifest["all_losses_and_gradients_finite"],
            "no_skipped_updates": original_manifest["skipped_updates"] == 0,
            "trainable_parameters_registered": original_manifest["trainable_parameter_inventory"]["passed"],
            "projection_unchanged_and_frozen": original_manifest["projection_unchanged"] and original_manifest["projection_trainable_parameters"] == 0,
            "c0_zero_binding_and_no_queue": c0_behavior_pass,
            "calibrated_initial_binding_safety_ratio_0p20_0p30": ratio_pass,
            "queue_occupancy_and_exclusion": queue_pass,
            "resume_trace_within_tolerance": maximum_difference <= tolerance and nonnumeric_equal,
            "resume_step2_adapter_hash_equal": step_two_adapter_hash(original) == step_two_adapter_hash(resumed),
            "vram_headroom": original_manifest["vram_gate_pass"],
            "core100_closed": not original_manifest["core100_opened"],
            "heldout_closed": not original_manifest["heldout_opened"],
            "no_generation": not original_manifest["candidate_outputs_generated"],
        }
        arm_rows.append(
            {
                "arm": arm_name,
                "eta": arm["eta"],
                "initial_unweighted_binding_safety_ratio": unweighted_ratio,
                "initial_eta_weighted_binding_safety_ratio": weighted_ratio,
                "resume_maximum_numeric_difference": maximum_difference,
                "resume_step2_adapter_hash_equal": gates["resume_step2_adapter_hash_equal"],
                "peak_reserved_bytes": original_manifest["peak_reserved_bytes"],
                "vram_headroom_fraction": original_manifest["vram_headroom_fraction"],
                "gates": gates,
                "overall_pass": all(gates.values()),
            }
        )

    reference = original_traces["C0R_concurrent_constant20"]
    draw_fields = ["risk_row_hash", "safety_prompt_hash", "safety_target_hash"]
    cross_arm = {
        arm_name: len(rows) == len(reference) and all(
            left[field] == right[field]
            for left, right in zip(reference, rows)
            for field in draw_fields
        )
        for arm_name, rows in original_traces.items()
    }
    overall = all(row["overall_pass"] for row in arm_rows) and all(cross_arm.values())
    summary = {
        "name": "step3ct_t2_smoke_gate_summary_v1",
        "config": args.config,
        "arms": arm_rows,
        "cross_arm_draw_trace_equal": cross_arm,
        "cross_arm_draw_gate_pass": all(cross_arm.values()),
        "overall_pass": overall,
        "next_action": "authorize_T3_full_three_arm_training" if overall else "stop_before_T3_and_report_failed_smoke",
        "core100_opened": False,
        "heldout_opened": False,
        "candidate_outputs_generated": False,
    }
    common.write_json(evidence / "step3ct_t2_smoke_gate_summary.json", summary)
    common.write_json(evidence / "step3ct_t2_smoke_run_manifests.json", manifests)
    common.write_csv(evidence / "step3ct_t2_smoke_arm_summary.csv", [
        {key: value for key, value in row.items() if key != "gates"} | {
            f"gate_{name}": passed for name, passed in row["gates"].items()
        }
        for row in arm_rows
    ])
    lines = [
        "# Step3C-T T2 Smoke Gate Report", "",
        f"- Overall pass: `{summary['overall_pass']}`",
        f"- Next action: `{summary['next_action']}`",
        f"- Cross-arm draw trace: `{summary['cross_arm_draw_gate_pass']}`",
        "- Core100/heldout opened: `false/false`", "",
    ]
    for row in arm_rows:
        lines.extend([
            f"## {row['arm']}", "",
            f"- Pass: `{row['overall_pass']}`",
            f"- Initial eta-weighted binding/safety ratio: `{row['initial_eta_weighted_binding_safety_ratio']}`",
            f"- Resume maximum numeric difference: `{row['resume_maximum_numeric_difference']}`",
            f"- VRAM headroom: `{row['vram_headroom_fraction']}`", "",
        ])
    (evidence / "Step3CT_T2_smoke_gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall_pass": summary["overall_pass"], "next_action": summary["next_action"]}))


if __name__ == "__main__":
    main()
