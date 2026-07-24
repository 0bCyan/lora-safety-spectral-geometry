"""Run the one-shot fresh-step0 eta calibration authorized by Step3C-T2R."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import step3ct_t2r_common as recovery
import step3ct_training_common as parent
import train_step3ct_tournament as trainer


REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1.json"
RECOVERY_REGISTRY = "experiments/step3_response_quality_persistence/configs/step3ct_t2r_calibration_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)


def unweighted_safety_gradient(
    model: Any,
    feature: dict[str, list[int]],
    coefficient: float,
    fp16_scale: float,
) -> tuple[dict[str, Any], float]:
    import torch

    device = parent.model_device(model)
    zero_grad(model)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16, enabled=True):
        output = model(**parent.batch_feature(feature, device), use_cache=False, return_dict=True)
        loss = output.loss
    (loss * coefficient * fp16_scale).backward()
    scaled = parent.capture_lora_gradients(model)
    result = recovery.zeros_like_float64(scaled)
    recovery.add_in_place(result, scaled, divisor=coefficient * fp16_scale)
    zero_grad(model)
    return result, float(loss.detach().float().cpu())


def risk_forward(model: Any, feature: dict[str, list[int]]) -> float:
    import torch

    device = parent.model_device(model)
    model.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
        loss = model(**parent.batch_feature(feature, device), use_cache=False, return_dict=True).loss
    return float(loss.detach().float().cpu())


def candidate_summary(
    arm: str,
    ratios: list[float],
    eta: float,
    full_ratio: float,
    window_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    weighted = [eta * value for value in ratios]
    quantiles = {
        str(probability): float(np.quantile(ratios, probability, method="linear"))
        for probability in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    }
    weighted_quantiles = {
        str(probability): float(np.quantile(weighted, probability, method="linear"))
        for probability in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    }
    gates = {
        "eighteen_complete_windows": len(ratios) == 18 and all(len(row["positions"]) == 4 for row in window_rows),
        "no_reserved_position_in_fit": all(min(row["positions"]) >= 8 for row in window_rows),
        "finite_nonzero_window_norms_and_ratios": all(
            math.isfinite(value) and value > 0 for value in ratios
        ),
        "eta_finite_in_0p01_0p30": math.isfinite(eta) and 0.01 <= eta <= 0.30,
        "weighted_median_in_0p20_0p30": 0.20 <= float(np.median(weighted)) <= 0.30,
        "weighted_p90_lte_0p75": float(np.quantile(weighted, 0.9, method="linear")) <= 0.75,
        "weighted_max_lte_1p00": max(weighted) <= 1.0,
    }
    return {
        "arm": arm,
        "window_unweighted_ratios": ratios,
        "unweighted_ratio_quantiles": quantiles,
        "formula_eta": eta,
        "window_eta_weighted_ratios": weighted,
        "weighted_ratio_quantiles": weighted_quantiles,
        "full_72_position_unweighted_ratio": full_ratio,
        "full_72_position_eta_weighted_ratio": eta * full_ratio,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg_path = REPO / args.config
    cfg = parent.read_json(cfg_path)
    registry = parent.read_json(REPO / RECOVERY_REGISTRY)
    output = REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1_t2r/calibration"
    route_path = output / "step3ct_t2r_r2_route_decision.json"
    if route_path.exists():
        raise RuntimeError("one-shot R2 calibration already has a frozen route; rerun is prohibited")
    output.mkdir(parents=True, exist_ok=True)

    import bitsandbytes
    import peft
    import torch
    import transformers

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = trainer.load_model(cfg, None)
    device = parent.model_device(model)
    inventory = parent.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RuntimeError("unexpected trainable parameters")
    lora_hash_before = recovery.lora_state_hash(model)
    head = parent.projection_head(
        int(model.config.hidden_size), int(cfg["projection_dim"]), int(cfg["projection_seed"])
    ).to(device)
    projection_hash_before = parent.module_state_hash(head)

    risk_rows = parent.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    targets = parent.read_jsonl(REPO / cfg["safety_targets_path"])
    grouped = parent.targets_by_prompt(targets)
    order = trainer.risk_order(len(risk_rows), int(cfg["seed"]))
    risk_features = [
        parent.response_feature(tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"]))
        for row in risk_rows
    ]
    target_features = {
        str(row["target_id"]): parent.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"])
        )
        for row in targets
    }
    queue, prefill_ids = parent.prefill_queue(
        model, tokenizer, head, targets, int(cfg["queue_size"]),
        int(cfg["queue_prefill_seed"]), int(cfg["max_length"]),
    )
    metadata_audit = recovery.metadata_hard_coverage(REPO, cfg)
    if not metadata_audit["passed"]:
        raise RuntimeError("metadata hard-coverage audit changed before calibration")

    fp16_scale = 65536.0
    branch_coefficient = float(cfg["safety_mix"]) / int(cfg["gradient_accumulation_steps"])
    position_rows = []
    window_rows = []
    full_safe = full_a = full_b = None
    window_safe = window_a = window_b = None
    current_window_positions: list[int] = []
    current_window_hard: list[bool] = []

    for position, target in enumerate(targets):
        position_started = time.time()
        risk_index = order[position]
        risk_loss = risk_forward(model, risk_features[risk_index])
        safety_gradient, safety_loss = unweighted_safety_gradient(
            model, target_features[str(target["target_id"])], branch_coefficient, fp16_scale
        )
        queue_hash_before = queue.content_hash()
        a_gradient, b_gradient, binding_metadata = recovery.dual_binding_gradients(
            model, tokenizer, head, target, grouped[str(target["sample_id"])], queue,
            int(cfg["max_length"]), float(cfg["temperature"]),
            branch_coefficient * fp16_scale, update_queue=True,
        )
        hard_active = int(binding_metadata["c3b"]["hard_rows"]) > 0
        expected_metadata = metadata_audit["positions"][position]
        if hard_active != expected_metadata["hard_active"]:
            raise RuntimeError(f"hard-active metadata mismatch at position {position}")
        if queue_hash_before != expected_metadata["queue_hash_before"]:
            # Vector content differs from the metadata-only queue; ID-state equality is checked below.
            _, prompt_ids, categories, target_ids = queue.ordered()
            queue_id_hash = parent.hash_json([prompt_ids, categories, target_ids, queue.position, queue.count])
        else:
            queue_id_hash = queue_hash_before
        row = {
            "position": position,
            "fit_eligible": position >= 8,
            "risk_row_hash": parent.stable_hash(str(risk_rows[risk_index].get("sample_id", risk_index))),
            "safety_prompt_hash": parent.stable_hash(str(target["sample_id"])),
            "safety_target_hash": parent.stable_hash(str(target["target_id"])),
            "risk_loss": risk_loss,
            "safety_loss": safety_loss,
            "c3a_binding_loss": binding_metadata["c3a"]["binding_loss"],
            "c3b_binding_loss": binding_metadata["c3b"]["binding_loss"],
            "c3b_hard_binding_loss": binding_metadata["c3b"]["hard_binding_loss"],
            "hard_active": hard_active,
            "hard_negative_count": expected_metadata["hard_negative_count"],
            "queue_hash_before": queue_hash_before,
            "queue_hash_after": queue.content_hash(),
            "queue_id_state_hash": queue_id_hash,
            "queue_snapshot_equal_c3a_c3b": True,
            "safety_gradient_norm": recovery.gradient_norm(safety_gradient),
            "c3a_gradient_norm": recovery.gradient_norm(a_gradient),
            "c3b_gradient_norm": recovery.gradient_norm(b_gradient),
            "safety_gradient_hash": recovery.gradient_hash(safety_gradient),
            "c3a_gradient_hash": recovery.gradient_hash(a_gradient),
            "c3b_gradient_hash": recovery.gradient_hash(b_gradient),
            "all_gradients_finite_nonzero": all(
                recovery.gradients_finite_nonzero(value)
                for value in [safety_gradient, a_gradient, b_gradient]
            ),
            "wall_clock_seconds": time.time() - position_started,
        }
        position_rows.append(row)

        if position >= 8:
            if window_safe is None:
                window_safe = recovery.zeros_like_float64(safety_gradient)
                window_a = recovery.zeros_like_float64(a_gradient)
                window_b = recovery.zeros_like_float64(b_gradient)
            if full_safe is None:
                full_safe = recovery.zeros_like_float64(safety_gradient)
                full_a = recovery.zeros_like_float64(a_gradient)
                full_b = recovery.zeros_like_float64(b_gradient)
            recovery.add_in_place(window_safe, safety_gradient)
            recovery.add_in_place(window_a, a_gradient)
            recovery.add_in_place(window_b, b_gradient)
            recovery.add_in_place(full_safe, safety_gradient)
            recovery.add_in_place(full_a, a_gradient)
            recovery.add_in_place(full_b, b_gradient)
            current_window_positions.append(position)
            current_window_hard.append(hard_active)
            if len(current_window_positions) == 4:
                safe_norm = recovery.gradient_norm(window_safe)
                a_norm = recovery.gradient_norm(window_a)
                b_norm = recovery.gradient_norm(window_b)
                window_rows.append(
                    {
                        "window_index": len(window_rows),
                        "positions": list(current_window_positions),
                        "hard_active": any(current_window_hard),
                        "hard_active_position_count": sum(current_window_hard),
                        "safety_gradient_norm": safe_norm,
                        "c3a_gradient_norm": a_norm,
                        "c3b_gradient_norm": b_norm,
                        "c3a_unweighted_ratio": a_norm / safe_norm,
                        "c3b_unweighted_ratio": b_norm / safe_norm,
                        "safety_gradient_hash": recovery.gradient_hash(window_safe),
                        "c3a_gradient_hash": recovery.gradient_hash(window_a),
                        "c3b_gradient_hash": recovery.gradient_hash(window_b),
                        "c3a_c3b_cosine": recovery.gradient_cosine(window_a, window_b),
                        "c3b_residual_given_c3a": recovery.orthogonal_residual_ratio(window_a, window_b),
                        "c3a_energy": recovery.energy_summary(window_a),
                        "c3b_energy": recovery.energy_summary(window_b),
                    }
                )
                window_safe = window_a = window_b = None
                current_window_positions = []
                current_window_hard = []
        print(
            f"calibration position {position + 1}/80 fit={position >= 8} hard={hard_active}",
            flush=True,
        )

    if current_window_positions or len(window_rows) != 18 or full_safe is None or full_a is None or full_b is None:
        raise RuntimeError("incomplete calibration window aggregation")
    recovery.calibration_windows(position_rows)
    a_ratios = [float(row["c3a_unweighted_ratio"]) for row in window_rows]
    b_ratios = [float(row["c3b_unweighted_ratio"]) for row in window_rows]
    eta_a = 0.25 / float(np.median(a_ratios))
    eta_b = 0.25 / float(np.median(b_ratios))
    full_safe_norm = recovery.gradient_norm(full_safe)
    summary_a = candidate_summary(
        "C3a_multi_positive_global_binding", a_ratios, eta_a,
        recovery.gradient_norm(full_a) / full_safe_norm, window_rows,
    )
    summary_b = candidate_summary(
        "C3b_hierarchical_hard_negative_binding", b_ratios, eta_b,
        recovery.gradient_norm(full_b) / full_safe_norm, window_rows,
    )
    lora_hash_after = recovery.lora_state_hash(model)
    projection_hash_after = parent.module_state_hash(head)
    integrity_gates = {
        "metadata_audit_pass": metadata_audit["passed"],
        "fit_positions_exact_8_79": [row["position"] for row in position_rows if row["fit_eligible"]] == list(range(8, 80)),
        "fit_excluded_positions_absent_from_windows": all(min(row["positions"]) >= 8 for row in window_rows),
        "all_position_gradients_finite_nonzero": all(row["all_gradients_finite_nonzero"] for row in position_rows),
        "queue_snapshots_equal_between_candidates": all(row["queue_snapshot_equal_c3a_c3b"] for row in position_rows),
        "c3b_hard_active_fit_count_54": sum(row["hard_active"] for row in position_rows[8:]) == 54,
        "lora_unchanged": lora_hash_before == lora_hash_after,
        "projection_unchanged": projection_hash_before == projection_hash_after,
        "registered_trainable_parameters_only": inventory["passed"],
    }
    both_candidates_pass = summary_a["overall_pass"] and summary_b["overall_pass"]
    overall_pass = all(integrity_gates.values()) and both_candidates_pass
    hard_active_residuals = [
        row["c3b_residual_given_c3a"] for row in window_rows if row["hard_active"]
    ]
    direction = {
        "aggregate_c3a_c3b_cosine": recovery.gradient_cosine(full_a, full_b),
        "aggregate_c3b_residual_given_c3a": recovery.orthogonal_residual_ratio(full_a, full_b),
        "median_hard_active_window_residual": float(np.median(hard_active_residuals)),
        "c3a_energy": recovery.energy_summary(full_a),
        "c3b_energy": recovery.energy_summary(full_b),
    }
    direction["c3b_redundancy_warning"] = (
        direction["aggregate_c3a_c3b_cosine"] >= 0.995
        and direction["median_hard_active_window_residual"] <= 0.10
    )

    parent.write_json(output / "step3ct_t2r_position_diagnostics.json", position_rows)
    parent.write_json(output / "step3ct_t2r_window_diagnostics.json", window_rows)
    parent.write_json(output / "step3ct_t2r_direction_analysis.json", direction)
    numerical = {
        "name": "step3ct_t2r_r2_eta_calibration_v1",
        "git_head": git_head(),
        "formula": "eta=0.25/median(window_unweighted_ratio)",
        "quantile_method": "numpy_linear",
        "eta_fit_positions": [8, 79],
        "eta_fit_window_count": len(window_rows),
        "candidates": [summary_a, summary_b],
        "integrity_gates": integrity_gates,
        "direction_analysis": direction,
        "lora_hash_before": lora_hash_before,
        "lora_hash_after": lora_hash_after,
        "projection_hash_before": projection_hash_before,
        "projection_hash_after": projection_hash_after,
        "queue_prefill_target_hash": parent.hash_json(prefill_ids),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "old_uncommitted_per_microstep_traces_read": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "overall_pass": overall_pass,
    }
    parent.write_json(output / "step3ct_t2r_eta_calibration_summary.json", numerical)

    new_config_path = REPO / registry["calibration"]["new_execution_config"]
    if overall_pass:
        new_cfg = copy.deepcopy(cfg)
        new_cfg["name"] = "4070ti_qwen3_4b_step3ct_seed1_t2r_v1"
        new_cfg["recovery_registry"] = RECOVERY_REGISTRY
        new_cfg["parent_execution_config"] = args.config
        new_cfg["arms"]["C3a_multi_positive_global_binding"]["eta"] = eta_a
        new_cfg["arms"]["C3b_hierarchical_hard_negative_binding"]["eta"] = eta_b
        new_cfg["eta_freeze"] = {
            "formula": "0.25/median(window_unweighted_ratio)",
            "fit_positions": [8, 79],
            "window_count": 18,
            "quantile_method": "numpy_linear",
            "calibration_summary": str((output / "step3ct_t2r_eta_calibration_summary.json").relative_to(REPO)).replace("\\", "/"),
            "calibration_summary_sha256": parent.sha256_file(output / "step3ct_t2r_eta_calibration_summary.json"),
            "source_git_head": git_head(),
        }
        parent.write_json(new_config_path, new_cfg)
    free, total = torch.cuda.mem_get_info(device)
    environment = {
        "name": "step3ct_t2r_r2_environment_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_vram_bytes_at_end": int(free),
        "total_vram_bytes": int(total),
        "wall_clock_seconds": time.time() - started,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }
    parent.write_json(output / "step3ct_t2r_r2_environment.json", environment)
    route = {
        "name": "step3ct_t2r_r2_route_decision_v1",
        "route": "continue_to_R3_smoke_requalification" if overall_pass else "RR1_fixed_eta_calibration_not_stable",
        "calibration_pass": overall_pass,
        "formula_eta": {
            "C3a_multi_positive_global_binding": eta_a,
            "C3b_hierarchical_hard_negative_binding": eta_b,
        },
        "new_execution_config": str(new_config_path.relative_to(REPO)).replace("\\", "/") if overall_pass else None,
        "new_execution_config_sha256": parent.sha256_file(new_config_path) if overall_pass else None,
        "next_authorized_package": "R3_fresh_three_arm_smoke" if overall_pass else None,
        "second_calibration_authorized": False,
        "optimizer_steps": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    parent.write_json(route_path, route)
    report = (
        "# Step3C-T2R R2 Eta Calibration Report\n\n"
        f"- Calibration pass: `{overall_pass}`\n"
        f"- C3a eta: `{eta_a}`\n"
        f"- C3b eta: `{eta_b}`\n"
        f"- C3a weighted median/p90/max: `{summary_a['weighted_ratio_quantiles']['0.5']}` / `{summary_a['weighted_ratio_quantiles']['0.9']}` / `{summary_a['weighted_ratio_quantiles']['1.0']}`\n"
        f"- C3b weighted median/p90/max: `{summary_b['weighted_ratio_quantiles']['0.5']}` / `{summary_b['weighted_ratio_quantiles']['0.9']}` / `{summary_b['weighted_ratio_quantiles']['1.0']}`\n"
        f"- Aggregate C3a-C3b cosine: `{direction['aggregate_c3a_c3b_cosine']}`\n"
        f"- C3b redundancy warning: `{direction['c3b_redundancy_warning']}`\n"
        f"- Runtime seconds: `{environment['wall_clock_seconds']}`\n"
        "- Optimizer/scheduler objects and optimizer steps: `0/0/0`\n"
        "- Candidate generation/core100/heldout: `false/false/false`\n"
    )
    (output / "Step3CT_T2R_R2_eta_calibration_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"overall_pass": overall_pass, "eta_a": eta_a, "eta_b": eta_b, "route": route["route"]}))


if __name__ == "__main__":
    main()
