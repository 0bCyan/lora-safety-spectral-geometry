"""Run the zero-update optimizer-Jacobian common-descent mechanism gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_mcd_common as mcd
import step3_mcd_training as smoke
import step3_ojmcd_common as ojmcd
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_ramp_training as base
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_ojmcd_p0/protocol/ojmcd_p0_protocol.json"
)
CERTIFICATES = (
    "risk",
    "uniform",
    "tarc",
    "etarc",
    "continuation",
    "branch_entropy",
    "branch_negative_gap",
)
SAFETY_OBJECTIVES = (
    "uniform",
    "tarc",
    "etarc",
    "continuation",
    "branch_entropy",
    "branch_negative_gap",
)
CANDIDATES = ("raw_mcd", "jacobian_mcd", "ojmcd")


class OjmcdP0ValidationError(ValueError):
    """Raised when OJ-MCD P0 violates its prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise OjmcdP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _all_blocks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [protocol["selection"]["calibration_block"], *protocol["selection"]["validation_blocks"]]


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != "prospective_before_any_OJMCD_model_forward":
        raise OjmcdP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or etarc_p0.sha256(artifact_path) != artifact["sha256"]
        ):
            raise OjmcdP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise OjmcdP0ValidationError("protocol violates the zero-update boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise OjmcdP0ValidationError("entry commit is not an ancestor of HEAD")

    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_MCD_ETARC_SMOKE_COMMON_EFFECT_FAILED":
        raise OjmcdP0ValidationError("registered MCD precursor route differs")
    config = etarc_p0.read_json(
        resolve_repo_path(protocol["source"]["training_config_path"])
    )
    expected = protocol["optimizer_map"]
    exact_pairs = {
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "safety_mix": "safety_mix",
        "max_grad_norm": "max_grad_norm",
        "adam_beta1": "adam_beta1",
        "adam_beta2": "adam_beta2",
        "adam_epsilon": "adam_epsilon",
        "weight_decay": "weight_decay",
        "fp16": "fp16",
        "gradient_checkpointing": "gradient_checkpointing",
    }
    for config_key, protocol_key in exact_pairs.items():
        if config.get(config_key) != expected.get(protocol_key):
            raise OjmcdP0ValidationError(
                f"frozen optimizer map differs: {config_key}"
            )
    if (
        int(config["gradient_accumulation_steps"]) != 4
        or float(config["weight_decay"]) != 0.0
        or config.get("manual_gradient_composition") is not True
    ):
        raise OjmcdP0ValidationError("P0 does not match the realized smoke step")

    targets = etarc_p0.read_jsonl(
        resolve_repo_path(protocol["source"]["safety_targets_path"])
    )
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(protocol["source"]["risk_data_path"]),
        limit=int(protocol["source"]["risk_rows"]),
    )
    if len(risk_rows) != int(protocol["source"]["risk_rows"]):
        raise OjmcdP0ValidationError("risk source row count differs")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(protocol["selection"]["risk_order_seed"]))
    risk_order = torch.randperm(len(risk_rows), generator=generator).tolist()
    safety_order = list(config["training_prompt_order"])
    blocks = _all_blocks(protocol)
    if len(blocks) != 8:
        raise OjmcdP0ValidationError("optimizer-unit block count differs")
    all_positions: list[int] = []
    all_risk: list[int] = []
    all_safety: list[str] = []
    for block in blocks:
        positions = [int(value) for value in block["source_order_positions"]]
        risk_indices = [int(value) for value in block["risk_row_indices"]]
        safety_ids = [str(value) for value in block["safety_sample_ids"]]
        if (
            len(positions) != 4
            or positions != list(range(positions[0], positions[0] + 4))
            or risk_indices != [risk_order[index] for index in positions]
            or safety_ids != [safety_order[index] for index in positions]
        ):
            raise OjmcdP0ValidationError(
                f"optimizer-unit block differs: {block['block_id']}"
            )
        all_positions.extend(positions)
        all_risk.extend(risk_indices)
        all_safety.extend(safety_ids)
    if (
        len(set(all_positions)) != 32
        or len(set(all_risk)) != 32
        or len(set(all_safety)) != 32
        or set(all_positions) != set(range(8, 40))
    ):
        raise OjmcdP0ValidationError("calibration/validation blocks overlap or drift")
    return protocol, config, targets, risk_rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise OjmcdP0ValidationError("OJ-MCD P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.append(relative(path))
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    for output in protocol["outputs"].values():
        if resolve_repo_path(output).exists():
            raise OjmcdP0ValidationError(f"prospective output already exists: {output}")
    return etarc_p0.git_output("rev-parse", "HEAD")


def structural_features(
    tokenizer: Any,
    targets: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        targets,
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    expected = etarc_p0.read_json(
        resolve_repo_path(protocol["source"]["structural_audit_path"])
    )
    if audit != expected:
        raise OjmcdP0ValidationError("route structure differs from frozen MCD P0")
    return {str(pair["sample_id"]): pair for pair in pairs}, audit


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise OjmcdP0ValidationError("CUDA is unavailable")
    tokenizer = tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = structural_features(tokenizer, targets, config, protocol)
    selected = [
        sample_id
        for block in _all_blocks(protocol)
        for sample_id in block["safety_sample_ids"]
    ]
    if not all(sample_id in pair_map for sample_id in selected):
        raise OjmcdP0ValidationError("selected safety pair is absent")
    result = {
        "status": "PASS_OJMCD_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "certificates": list(CERTIFICATES),
        "candidate_directions": list(CANDIDATES),
        "optimizer_unit_blocks": 8,
        "calibration_blocks": 1,
        "validation_blocks": 7,
        "risk_microbatches_per_block": 4,
        "safety_microbatches_per_block": 4,
        "selected_risk_rows": 32,
        "selected_safety_prompt_pairs": 32,
        "calibration_validation_overlap": 0,
        "gpu": torch.cuda.get_device_name(0),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _zero_gradient(parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    return {name: torch.zeros_like(parameter) for name, parameter in parameters}


def _scale_in_place(gradient: dict[str, Any], scale: float) -> None:
    for value in gradient.values():
        value.mul_(scale)


def aggregate_optimizer_block(
    model: Any,
    parameters: list[tuple[str, Any]],
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    block: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    risk = _zero_gradient(parameters)
    safety = {name: _zero_gradient(parameters) for name in SAFETY_OBJECTIVES}
    risk_losses: list[float] = []
    losses_by_objective: dict[str, list[float]] = {
        name: [] for name in SAFETY_OBJECTIVES
    }
    response_tokens = 0
    continuation_tokens = 0
    tarc_terms = 0
    etarc_terms = 0
    device = training.model_device(model)
    microbatches = len(block["risk_row_indices"])
    for risk_index, sample_id in zip(
        block["risk_row_indices"], block["safety_sample_ids"], strict=True
    ):
        model.train()
        risk_loss = base.risk_loss(
            model,
            risk_features[int(risk_index)],
            device,
            bool(config["fp16"]),
        )
        risk_gradient = smoke.gradient_from_loss(risk_loss, parameters, "risk")
        for name in risk:
            risk[name].add_(risk_gradient[name])
        risk_losses.append(float(risk_loss.detach().float().cpu()))

        current_losses, metrics = smoke.forward_pair_objectives(
            model,
            pair_map[str(sample_id)],
            device,
            bool(config["fp16"]),
            int(config["ramp_width"]),
            float(config["softmin_temperature"]),
        )
        objective_gradients = smoke.gradients_from_objectives(
            current_losses, parameters
        )
        for objective in SAFETY_OBJECTIVES:
            for name in safety[objective]:
                safety[objective][name].add_(objective_gradients[objective][name])
            losses_by_objective[objective].append(
                float(metrics[f"{objective}_loss"])
            )
        response_tokens += int(metrics["response_tokens"])
        continuation_tokens += int(metrics["continuation_tokens"])
        tarc_terms += int(metrics["tarc_margin_terms"])
        etarc_terms += int(metrics["etarc_margin_terms"])
        del risk_loss, risk_gradient, current_losses, objective_gradients

    _scale_in_place(risk, 1.0 / microbatches)
    for gradient in safety.values():
        _scale_in_place(gradient, 1.0 / microbatches)
    gradients = {"risk": risk, **safety}
    norms = {
        name: gradient_math.gradient_norm(gradient)
        for name, gradient in gradients.items()
    }
    missing = [
        name
        for name, gradient in gradients.items()
        if set(gradient) != {parameter_name for parameter_name, _ in parameters}
    ]
    return gradients, {
        "risk_loss_mean": statistics.fmean(risk_losses),
        "safety_loss_means": {
            name: statistics.fmean(values)
            for name, values in losses_by_objective.items()
        },
        "gradient_norms": norms,
        "missing_gradient_objects": missing,
        "all_gradients_finite_nonzero": all(
            math.isfinite(value) and value > 0.0 for value in norms.values()
        ),
        "risk_microbatches": microbatches,
        "safety_microbatches": microbatches,
        "response_tokens": response_tokens,
        "continuation_tokens": continuation_tokens,
        "tarc_margin_terms": tarc_terms,
        "etarc_margin_terms": etarc_terms,
    }


def _baseline_gradient(
    gradients: dict[str, dict[str, Any]], safety_mix: float
) -> dict[str, Any]:
    return gradient_math.add_gradients(
        gradient_math.scaled_gradient(gradients["risk"], 1.0 - safety_mix),
        gradient_math.scaled_gradient(gradients["uniform"], safety_mix),
    )


def analyze_block(
    gradients: dict[str, dict[str, Any]],
    coverage: dict[str, Any],
    block: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    design = protocol["design"]
    optimizer_map = protocol["optimizer_map"]
    safety_mix = float(optimizer_map["safety_mix"])
    max_norm = float(optimizer_map["max_grad_norm"])
    adam_epsilon = float(optimizer_map["adam_epsilon"])
    clip_epsilon = float(optimizer_map["clip_epsilon"])
    certificates = {name: gradients[name] for name in CERTIFICATES}
    baseline = _baseline_gradient(gradients, safety_mix)
    budget_norm = float(design["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
        gradients["uniform"]
    )

    raw_direction, raw_dual = mcd.minimax_common_direction(
        certificates, budget_norm, float(design["dual_tolerance"])
    )
    jacobian_direction, jacobian_dual = ojmcd.optimizer_jacobian_common_direction(
        baseline,
        certificates,
        budget_norm,
        max_norm,
        adam_epsilon,
        clip_epsilon,
        float(design["dual_tolerance"]),
    )
    optimizer_direction, optimizer_dual = (
        ojmcd.baseline_aware_optimizer_jacobian_direction(
            baseline,
            certificates,
            budget_norm,
            safety_mix,
            max_norm,
            adam_epsilon,
            clip_epsilon,
            float(design["dual_tolerance"]),
        )
    )
    directions = {
        "raw_mcd": raw_direction,
        "jacobian_mcd": jacobian_direction,
        "ojmcd": optimizer_direction,
    }
    duals = {
        "raw_mcd": raw_dual,
        "jacobian_mcd": jacobian_dual,
        "ojmcd": optimizer_dual,
    }
    candidates = {
        name: ojmcd.realized_increment_metrics(
            baseline,
            direction,
            certificates,
            safety_mix,
            max_norm,
            adam_epsilon,
            clip_epsilon,
        )
        for name, direction in directions.items()
    }
    budget_errors = {
        name: abs(gradient_math.gradient_norm(direction) - budget_norm)
        / budget_norm
        for name, direction in directions.items()
    }
    return {
        "block_id": block["block_id"],
        "split": block["split"],
        "source_order_positions": block["source_order_positions"],
        "risk_row_indices": block["risk_row_indices"],
        "safety_sample_ids": block["safety_sample_ids"],
        "coverage": coverage,
        "baseline_raw_gradient_norm": gradient_math.gradient_norm(baseline),
        "uniform_gradient_norm": gradient_math.gradient_norm(gradients["uniform"]),
        "auxiliary_budget_norm": budget_norm,
        "maximum_budget_relative_error": max(budget_errors.values()),
        "candidate_budget_relative_errors": budget_errors,
        "duals": duals,
        "candidates": candidates,
    }


def _paired(values: list[float], protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "differences": values,
        "positive_rate": sum(value > 0.0 for value in values) / len(values),
        "bootstrap": etarc_p0.paired_bootstrap(
            values,
            int(protocol["statistics"]["bootstrap_samples"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        ),
        "sign_test": etarc_p0.exact_sign_test(values),
    }


def summarize_rows(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(calibration) != 1 or len(validation) != 7:
        raise OjmcdP0ValidationError("stored optimizer-unit block counts differ")

    candidate_summary: dict[str, Any] = {}
    for candidate in CANDIDATES:
        metrics = [row["candidates"][candidate] for row in validation]
        candidate_summary[candidate] = {
            "validation_blocks": len(metrics),
            "minimum_exact_candidate_score": min(
                float(item["minimum_exact_candidate_normalized_score"])
                for item in metrics
            ),
            "mean_exact_candidate_worst_score": statistics.fmean(
                float(item["minimum_exact_candidate_normalized_score"])
                for item in metrics
            ),
            "minimum_exact_increment_score": min(
                float(item["minimum_exact_increment_normalized_score"])
                for item in metrics
            ),
            "minimum_increment_norm_fraction": min(
                float(item["increment_norm_fraction_of_baseline"])
                for item in metrics
            ),
            "mean_increment_norm_fraction": statistics.fmean(
                float(item["increment_norm_fraction_of_baseline"])
                for item in metrics
            ),
            "minimum_linear_exact_cosine": min(
                float(item["linear_exact_cosine"]) for item in metrics
            ),
            "mean_linear_exact_cosine": statistics.fmean(
                float(item["linear_exact_cosine"]) for item in metrics
            ),
            "minimum_linear_exact_norm_ratio": min(
                float(item["linear_exact_norm_ratio"]) for item in metrics
            ),
            "maximum_linear_exact_norm_ratio": max(
                float(item["linear_exact_norm_ratio"]) for item in metrics
            ),
            "minimum_certificate_scores": {
                certificate: min(
                    float(item["certificates"][certificate]["exact_candidate_normalized_score"])
                    for item in metrics
                )
                for certificate in CERTIFICATES
            },
        }

    ojmcd_worst = [
        float(row["candidates"]["ojmcd"]["minimum_exact_candidate_normalized_score"])
        for row in validation
    ]
    raw_worst = [
        float(row["candidates"]["raw_mcd"]["minimum_exact_candidate_normalized_score"])
        for row in validation
    ]
    jacobian_worst = [
        float(row["candidates"]["jacobian_mcd"]["minimum_exact_candidate_normalized_score"])
        for row in validation
    ]
    baseline_worst = [
        float(row["candidates"]["ojmcd"]["minimum_baseline_normalized_score"])
        for row in validation
    ]
    comparisons = {
        "ojmcd_minus_raw_mcd": _paired(
            [left - right for left, right in zip(ojmcd_worst, raw_worst, strict=True)],
            protocol,
        ),
        "ojmcd_minus_jacobian_mcd": _paired(
            [left - right for left, right in zip(ojmcd_worst, jacobian_worst, strict=True)],
            protocol,
        ),
        "ojmcd_minus_no_auxiliary_baseline": _paired(
            [left - right for left, right in zip(ojmcd_worst, baseline_worst, strict=True)],
            protocol,
        ),
    }
    return {
        "calibration": {
            "block_id": calibration[0]["block_id"],
            "ojmcd_minimum_exact_candidate_score": calibration[0]["candidates"]["ojmcd"][
                "minimum_exact_candidate_normalized_score"
            ],
            "ojmcd_linear_exact_cosine": calibration[0]["candidates"]["ojmcd"][
                "linear_exact_cosine"
            ],
            "ojmcd_linear_exact_norm_ratio": calibration[0]["candidates"]["ojmcd"][
                "linear_exact_norm_ratio"
            ],
            "ojmcd_increment_norm_fraction": calibration[0]["candidates"]["ojmcd"][
                "increment_norm_fraction_of_baseline"
            ],
        },
        "validation": candidate_summary,
        "paired_validation": comparisons,
    }


def evaluate_gates(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    state_restored: bool,
    vram_headroom: float,
) -> dict[str, bool]:
    gates = protocol["gates"]
    all_candidates = [
        row["candidates"][candidate] for row in rows for candidate in CANDIDATES
    ]
    validation_ojmcd = summary["validation"]["ojmcd"]
    calibration = summary["calibration"]
    raw_advantage = summary["paired_validation"]["ojmcd_minus_raw_mcd"]
    baseline_advantage = summary["paired_validation"][
        "ojmcd_minus_no_auxiliary_baseline"
    ]
    return {
        "integrity": (
            len(rows) == 8
            and all(row["coverage"]["risk_microbatches"] == 4 for row in rows)
            and all(row["coverage"]["safety_microbatches"] == 4 for row in rows)
            and all(not row["coverage"]["missing_gradient_objects"] for row in rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in rows)
            and all(
                float(row["maximum_budget_relative_error"])
                <= float(gates["gradient_budget_relative_tolerance"])
                for row in rows
            )
            and state_restored
            and vram_headroom >= float(gates["minimum_vram_headroom_fraction"])
        ),
        "clipping_exercised": all(
            bool(item["baseline"]["clipping_active"])
            and bool(item["candidate"]["clipping_active"])
            for item in all_candidates
        ),
        "dual_certificate": all(
            bool(row["duals"]["ojmcd"]["solver_success"])
            and float(row["duals"]["ojmcd"]["kkt_residual"])
            <= float(gates["dual_kkt_residual_max"])
            and float(row["duals"]["ojmcd"]["primal_dual_gap"])
            <= float(gates["primal_dual_gap_max"])
            and float(row["duals"]["ojmcd"]["predicted_minimum_score_improvement"])
            > 0.0
            for row in rows
        ),
        "secant_fidelity": (
            float(calibration["ojmcd_linear_exact_cosine"])
            >= float(gates["calibration_linear_exact_cosine_min"])
            and float(calibration["ojmcd_linear_exact_norm_ratio"])
            >= float(gates["linear_exact_norm_ratio_min"])
            and float(calibration["ojmcd_linear_exact_norm_ratio"])
            <= float(gates["linear_exact_norm_ratio_max"])
            and float(validation_ojmcd["minimum_linear_exact_cosine"])
            >= float(gates["validation_linear_exact_cosine_min"])
            and float(validation_ojmcd["mean_linear_exact_cosine"])
            >= float(gates["validation_mean_linear_exact_cosine_min"])
            and float(validation_ojmcd["minimum_linear_exact_norm_ratio"])
            >= float(gates["linear_exact_norm_ratio_min"])
            and float(validation_ojmcd["maximum_linear_exact_norm_ratio"])
            <= float(gates["linear_exact_norm_ratio_max"])
        ),
        "exact_common_descent": (
            float(calibration["ojmcd_minimum_exact_candidate_score"])
            >= float(gates["calibration_minimum_exact_score_min"])
            and float(validation_ojmcd["minimum_exact_candidate_score"])
            >= float(gates["validation_minimum_exact_score_min"])
            and all(
                float(value) >= float(gates["validation_certificate_score_min"])
                for value in validation_ojmcd["minimum_certificate_scores"].values()
            )
        ),
        "optimizer_advantage": (
            float(raw_advantage["bootstrap"]["mean"])
            >= float(gates["paired_worst_score_advantage_mean_min"])
            and float(raw_advantage["bootstrap"]["ci95_lower"])
            > float(gates["paired_advantage_ci_lower_min"])
            and float(raw_advantage["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and float(baseline_advantage["bootstrap"]["mean"])
            >= float(gates["paired_worst_score_advantage_mean_min"])
            and float(baseline_advantage["bootstrap"]["ci95_lower"])
            > float(gates["paired_advantage_ci_lower_min"])
            and float(baseline_advantage["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
        ),
        "effect_not_weak": (
            float(calibration["ojmcd_increment_norm_fraction"])
            >= float(gates["increment_norm_fraction_min"])
            and float(validation_ojmcd["minimum_increment_norm_fraction"])
            >= float(gates["increment_norm_fraction_min"])
            and float(validation_ojmcd["mean_increment_norm_fraction"])
            >= float(gates["mean_increment_norm_fraction_min"])
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"]:
        return "STOP_OJMCD_P0_INVALID"
    if not gates["clipping_exercised"]:
        return "STOP_OJMCD_P0_OPTIMIZER_MAP_NOT_EXERCISED"
    if not gates["dual_certificate"]:
        return "STOP_OJMCD_P0_DUAL_CERTIFICATE_FAILED"
    if not gates["secant_fidelity"]:
        return "STOP_OJMCD_P0_SECANT_FIDELITY_FAILED"
    if not gates["exact_common_descent"]:
        return "STOP_OJMCD_P0_EXACT_COMMON_DESCENT_FAILED"
    if not gates["optimizer_advantage"]:
        return "STOP_OJMCD_P0_OPTIMIZER_ADVANTAGE_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_OJMCD_P0_EFFECT_TOO_WEAK"
    return "PASS_OJMCD_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL"


def package_artifacts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for name, value in protocol["outputs"].items():
        if name == "run_manifest":
            continue
        path = resolve_repo_path(value)
        artifacts.append(
            {
                "name": name,
                "path": relative(path),
                "sha256": etarc_p0.sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(etarc_p0.read_jsonl(path))
                if path.suffix == ".jsonl"
                else None,
            }
        )
    return artifacts


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import set_seed

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise OjmcdP0ValidationError("unexpected trainable parameter inventory")
    initial_state = model_common.trainable_state_hash(parameters)
    pair_map, structural = structural_features(tokenizer, targets, config, protocol)
    selected_risk = {
        int(index)
        for block in _all_blocks(protocol)
        for index in block["risk_row_indices"]
    }
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in selected_risk
    }

    rows = []
    for block in _all_blocks(protocol):
        block_started = time.time()
        gradients, coverage = aggregate_optimizer_block(
            model,
            parameters,
            risk_features,
            pair_map,
            block,
            config,
        )
        row = analyze_block(gradients, coverage, block, protocol)
        row["wall_seconds"] = time.time() - block_started
        rows.append(row)
        del gradients
        print(
            f"OJ-MCD {block['block_id']} exact_worst="
            f"{row['candidates']['ojmcd']['minimum_exact_candidate_normalized_score']:.6f} "
            f"secant_cos={row['candidates']['ojmcd']['linear_exact_cosine']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_state = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    summary = summarize_rows(rows, protocol)
    gates = evaluate_gates(
        rows, summary, protocol, initial_state == final_state, headroom
    )
    route = select_route(gates)
    outputs = {
        name: resolve_repo_path(value) for name, value in protocol["outputs"].items()
    }
    diagnostics = {
        "name": "step3_ojmcd_p0_optimizer_unit_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "certificates": list(CERTIFICATES),
        "candidates": list(CANDIDATES),
        "summary": summary,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "state_restored": initial_state == final_state,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }
    environment = {
        "name": "step3_ojmcd_p0_environment_v1",
        "runtime_head": runtime_head,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_ojmcd_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "summary": summary,
        "matched_smoke_protocol_authorized": route
        == "PASS_OJMCD_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    raw_pair = summary["paired_validation"]["ojmcd_minus_raw_mcd"]
    baseline_pair = summary["paired_validation"][
        "ojmcd_minus_no_auxiliary_baseline"
    ]
    validation = summary["validation"]["ojmcd"]
    report = f"""# OJ-MCD Zero-Update P0 Report

Route: `{route}`.

Eight real optimizer-unit blocks were evaluated at the unchanged initial
Qwen3-4B LoRA state. Every block aggregated four risk and four safety
microbatches before applying the exact clipping plus first-step Adam map.

Across seven prompt-disjoint validation blocks, OJ-MCD's minimum exact
normalized full-update score was
`{validation['minimum_exact_candidate_score']:.6f}`. Its mean paired
worst-score advantage over equal-budget raw seven-certificate MCD was
`{raw_pair['bootstrap']['mean']:.6f}` with 95% bootstrap interval
`[{raw_pair['bootstrap']['ci95_lower']:.6f},
{raw_pair['bootstrap']['ci95_upper']:.6f}]` and exact sign-test
`p={raw_pair['sign_test']['exact_two_sided_p']:.8g}`. The corresponding
advantage over the no-auxiliary mixed baseline was
`{baseline_pair['bootstrap']['mean']:.6f}`.

The minimum and mean linear-versus-exact secant cosines were
`{validation['minimum_linear_exact_cosine']:.6f}` and
`{validation['mean_linear_exact_cosine']:.6f}`. The exact realized auxiliary
increment occupied at least
`{validation['minimum_increment_norm_fraction']:.6%}` of the baseline update
norm over validation blocks.

This is exploratory mechanism evidence on reused disclosed development data,
not generated-behavior or confirmatory evidence. `optimizer_objects=0`,
`optimizer_steps=0`, `training_authorized=false`, and no candidate, Core100,
or heldout data was opened.
"""
    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_jsonl_frozen(outputs["block_rows"], rows)
    etarc_p0.write_json_frozen(outputs["gradient_diagnostics"], diagnostics)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_ojmcd_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_ojmcd_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_ojmcd_p0.py verify-run"
        ),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "artifacts": package_artifacts(protocol),
    }
    etarc_p0.write_json_frozen(outputs["run_manifest"], manifest)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    rows = etarc_p0.read_jsonl(resolve_repo_path(protocol["outputs"]["block_rows"]))
    diagnostics = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["gradient_diagnostics"])
    )
    environment = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["environment"])
    )
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["route"]))
    manifest = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["run_manifest"])
    )
    summary = summarize_rows(rows, protocol)
    gates = evaluate_gates(
        rows,
        summary,
        protocol,
        bool(diagnostics["state_restored"]),
        float(environment["vram_headroom_fraction"]),
    )
    route = select_route(gates)
    if (
        decision.get("route") != route
        or decision.get("gates") != gates
        or decision.get("summary") != summary
        or diagnostics.get("summary") != summary
        or manifest.get("route") != route
    ):
        raise OjmcdP0ValidationError("stored and recomputed OJ-MCD route differs")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise OjmcdP0ValidationError("P0 artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise OjmcdP0ValidationError("stored P0 boundary differs")
    result = {
        "status": "PASS_OJMCD_P0_VERIFY",
        "route": route,
        "optimizer_unit_blocks": len(rows),
        "artifacts": len(manifest["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    if args.command == "preflight":
        preflight(path)
    elif args.command == "run":
        run(path)
    else:
        verify_run(path)


if __name__ == "__main__":
    main()
