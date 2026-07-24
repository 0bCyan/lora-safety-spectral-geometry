"""Apply the frozen Step3C-T2R smoke requalification gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import step3ct_training_common as parent


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1_t2r.json",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_hash(root: Path, step: int) -> str:
    rows = [row for row in read_csv(root / "checkpoint_index.csv") if int(row["step"]) == step]
    if len(rows) != 1:
        raise ValueError(f"missing unique step {step} checkpoint in {root}")
    return rows[0]["adapter_sha256"]


def max_numeric_difference(left: list[dict[str, str]], right: list[dict[str, str]]) -> float:
    fields = [
        "risk_loss", "safety_loss", "global_binding_loss", "hard_binding_loss",
        "binding_loss", "total_loss", "positive_probability",
    ]
    differences = []
    for a, b in zip(left, right):
        for field in fields:
            if a.get(field, "") == b.get(field, "") == "":
                continue
            differences.append(abs(float(a[field]) - float(b[field])))
    return max(differences, default=0.0)


def main() -> None:
    args = parse_args()
    cfg = parent.read_json(REPO / args.config)
    root = REPO / cfg["output_dir"]
    runs = root / "runs" / "smoke_t2r"
    output = root / "smoke_t2r"
    output.mkdir(parents=True, exist_ok=True)
    tolerance = float(cfg["resume_tolerance"])
    identity_fields = [
        "global_step_before_update", "micro_index", "risk_row_hash",
        "safety_prompt_hash", "safety_target_hash", "queue_hash_before", "queue_hash_after",
    ]
    arm_rows = []
    manifest_package = {}
    traces = {}
    for arm_name, arm in cfg["arms"].items():
        original = runs / arm_name / "original"
        resumed = runs / arm_name / "resume_from_step1"
        original_manifest = parent.read_json(original / "run_manifest.json")
        resumed_manifest = parent.read_json(resumed / "run_manifest.json")
        original_ratio = parent.read_json(original / "optimizer_unit_gradient_ratios.json")
        resumed_ratio = parent.read_json(resumed / "optimizer_unit_gradient_ratios.json")
        original_trace = read_csv(original / "training_trace.csv")
        resumed_trace = read_csv(resumed / "training_trace.csv")
        traces[arm_name] = original_trace
        manifest_package[arm_name] = {
            "original": original_manifest,
            "resume": resumed_manifest,
            "original_ratios": original_ratio,
            "resume_ratios": resumed_ratio,
        }
        numeric_difference = max_numeric_difference(original_trace, resumed_trace)
        identities_equal = len(original_trace) == len(resumed_trace) and all(
            left[field] == right[field]
            for left, right in zip(original_trace, resumed_trace)
            for field in identity_fields
        )
        original_step2 = next(row for row in original_ratio["optimizer_steps"] if int(row["optimizer_step"]) == 2)
        resumed_step2 = next(row for row in resumed_ratio["optimizer_steps"] if int(row["optimizer_step"]) == 2)
        step_ratio_difference = abs(
            float(original_step2["eta_weighted_binding_safety_ratio"])
            - float(resumed_step2["eta_weighted_binding_safety_ratio"])
        )
        step_ratios = [float(row["eta_weighted_binding_safety_ratio"]) for row in original_ratio["optimizer_steps"]]
        aggregate_ratio = original_ratio["aggregate_eta_weighted_binding_safety_ratio"]
        c0_pass = arm["binding"] != "none" or (
            not original_manifest["queue_constructed"]
            and all(float(row["binding_loss"]) == 0 for row in original_trace)
            and aggregate_ratio is None
        )
        c3_ratio_pass = arm["binding"] == "none" or (
            len(step_ratios) == 2
            and all(0.10 <= value <= 0.50 for value in step_ratios)
            and aggregate_ratio is not None
            and 0.15 <= float(aggregate_ratio) <= 0.35
        )
        hard_count_pass = arm_name != "C3b_hierarchical_hard_negative_binding" or (
            int(original_ratio["c3b_hard_active_positions"]) == 4
        )
        gates = {
            "fresh_two_step_run": original_manifest["start_step"] == 0 and original_manifest["optimizer_steps_executed"] == 2,
            "resume_from_step1": resumed_manifest["start_step"] == 1 and resumed_manifest["optimizer_steps_executed"] == 1,
            "finite_losses_and_gradients": original_manifest["all_losses_and_gradients_finite"],
            "no_skipped_updates": original_manifest["skipped_updates"] == 0,
            "registered_trainable_parameters": original_manifest["trainable_parameter_inventory"]["passed"],
            "projection_unchanged_and_frozen": original_manifest["projection_unchanged"] and original_manifest["projection_trainable_parameters"] == 0,
            "c0_zero_binding_and_no_queue": c0_pass,
            "optimizer_unit_ratio_gates": c3_ratio_pass,
            "c3b_hard_active_exactly_4_of_8": hard_count_pass,
            "resume_trace_equal": numeric_difference <= tolerance and identities_equal,
            "resume_step2_ratio_equal": step_ratio_difference <= tolerance,
            "resume_adapter_hash_equal": checkpoint_hash(original, 2) == checkpoint_hash(resumed, 2),
            "resume_queue_hash_equal": original_manifest["queue_hash"] == resumed_manifest["queue_hash"],
            "config_hash_matches": original_manifest["config_sha256"] == parent.sha256_file(REPO / args.config),
            "vram_headroom": original_manifest["vram_gate_pass"],
            "no_generation_core100_or_heldout": (
                not original_manifest["candidate_outputs_generated"]
                and not original_manifest["core100_opened"]
                and not original_manifest["heldout_opened"]
            ),
        }
        arm_rows.append(
            {
                "arm": arm_name,
                "eta": arm["eta"],
                "step_ratios": step_ratios,
                "aggregate_eight_position_ratio": aggregate_ratio,
                "c3b_hard_active_positions": original_ratio["c3b_hard_active_positions"],
                "resume_maximum_numeric_difference": numeric_difference,
                "resume_step2_ratio_difference": step_ratio_difference,
                "vram_headroom_fraction": original_manifest["vram_headroom_fraction"],
                "gates": gates,
                "overall_pass": all(gates.values()),
            }
        )

    reference = traces["C0R_concurrent_constant20"]
    draw_fields = ["risk_row_hash", "safety_prompt_hash", "safety_target_hash"]
    cross_arm = {
        arm_name: len(rows) == len(reference) and all(
            left[field] == right[field]
            for left, right in zip(reference, rows)
            for field in draw_fields
        )
        for arm_name, rows in traces.items()
    }
    integrity_pass = all(cross_arm.values())
    candidates_pass = all(row["overall_pass"] for row in arm_rows)
    overall = integrity_pass and candidates_pass
    if not integrity_pass:
        route = "RR0_integrity_or_provenance_failure"
    elif not candidates_pass:
        route = "RR1_fixed_eta_calibration_not_stable"
    else:
        route = "RR2_three_arm_smoke_requalified"
    summary = {
        "name": "step3ct_t2r_r3_route_decision_v1",
        "route": route,
        "arms": arm_rows,
        "cross_arm_draw_trace_equal": cross_arm,
        "cross_arm_draw_gate_pass": integrity_pass,
        "overall_pass": overall,
        "next_authorized_package": "Step3C-T_T3_full_three_arm_training" if route == "RR2_three_arm_smoke_requalified" else None,
        "second_recovery_attempt_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    parent.write_json(output / "step3ct_t2r_r3_route_decision.json", summary)
    parent.write_json(output / "step3ct_t2r_smoke_manifest_package.json", manifest_package)
    parent.write_csv(output / "step3ct_t2r_smoke_gate_table.csv", [
        {key: value for key, value in row.items() if key != "gates"} | {
            f"gate_{name}": passed for name, passed in row["gates"].items()
        }
        for row in arm_rows
    ])
    report = [
        "# Step3C-T2R R3 Smoke Requalification Report", "",
        f"- Route: `{route}`",
        f"- Overall pass: `{overall}`",
        f"- Cross-arm draw trace: `{integrity_pass}`",
        f"- Next authorized package: `{summary['next_authorized_package']}`",
        "- Candidate generation/core100/heldout: `false/false/false`", "",
    ]
    for row in arm_rows:
        report.extend([
            f"## {row['arm']}", "",
            f"- Pass: `{row['overall_pass']}`",
            f"- Optimizer-step ratios: `{row['step_ratios']}`",
            f"- Eight-position ratio: `{row['aggregate_eight_position_ratio']}`",
            f"- Resume maximum difference: `{row['resume_maximum_numeric_difference']}`", "",
        ])
    (output / "Step3CT_T2R_R3_smoke_requalification_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"route": route, "overall_pass": overall}))


if __name__ == "__main__":
    main()
