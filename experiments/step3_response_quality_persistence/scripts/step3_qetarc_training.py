"""Run the hash-bound uniform/TARC/E-TARC/Q-E-TARC matched smoke."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_qetarc_common as qetarc
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_ramp_training as base
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_qetarc/"
    "protocol/qetarc_smoke_protocol_v2.json"
)
OBJECTIVES = ("uniform", "tarc", "etarc", "continuation")
DIAGNOSTIC_LOSSES = (
    "uniform_loss",
    "tarc_loss",
    "etarc_loss",
    "continuation_loss",
    "branch_entropy_loss",
    "branch_negative_gap_loss",
)


class QetarcTrainingValidationError(ValueError):
    """Raised when matched Q-E-TARC training violates its frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    return base.resolve_repo_path(value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_bytes(payload.encode("utf-8"))


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    protocol = base.read_json(path)
    if protocol.get("status") != "prospective_before_any_matched_QETARC_smoke_training":
        raise QetarcTrainingValidationError("training protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise QetarcTrainingValidationError(f"frozen input hash mismatch: {name}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise QetarcTrainingValidationError("entry commit is not an ancestor of HEAD")

    config = base.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        config.get("smoke_training_authorized") is not True
        or protocol.get("training_authorized") is not True
        or config.get("manual_gradient_composition") is not True
        or config.get("gradient_checkpointing") is not False
    ):
        raise QetarcTrainingValidationError("dynamic smoke training is not authorized")
    if (
        config.get("development_training_authorized") is not False
        or config.get("main_training_authorized") is not False
        or config.get("candidate_generation_authorized") is not False
        or config.get("core100_opened") is not False
        or config.get("heldout_opened") is not False
    ):
        raise QetarcTrainingValidationError("config violates the sealed boundary")
    if int(protocol["optimizer_steps_per_arm"]) != int(config["smoke_steps"]):
        raise QetarcTrainingValidationError("protocol step count differs from config")
    if int(config["optimizer_steps"]) != int(config["smoke_steps"]):
        raise QetarcTrainingValidationError("optimizer schedule differs from smoke steps")
    arms = list(protocol["arm_order"])
    expected_arms = ["uniform", "tarc", "etarc", "qetarc"]
    if arms != expected_arms or list(config["arms"]) != expected_arms:
        raise QetarcTrainingValidationError("matched arm registry differs")
    if [config["arms"][name]["objective"] for name in arms] != expected_arms:
        raise QetarcTrainingValidationError("arm/objective assignment differs")

    p0_route_path = resolve_repo_path(config["p0_route"])
    p0_protocol_path = resolve_repo_path(config["p0_protocol"])
    if base.sha256(p0_route_path) != config["p0_route_sha256"]:
        raise QetarcTrainingValidationError("Q-E-TARC P0 route hash differs")
    if base.sha256(p0_protocol_path) != config["p0_protocol_sha256"]:
        raise QetarcTrainingValidationError("Q-E-TARC P0 protocol hash differs")
    p0_structural_path = resolve_repo_path(config["p0_structural_audit"])
    if base.sha256(p0_structural_path) != config["p0_structural_audit_sha256"]:
        raise QetarcTrainingValidationError("Q-E-TARC P0 structural audit hash differs")
    p0_route = base.read_json(p0_route_path)
    p0_protocol = base.read_json(p0_protocol_path)
    if (
        p0_route.get("route") != "PASS_QETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL"
        or p0_route.get("matched_smoke_protocol_authorized") is not True
    ):
        raise QetarcTrainingValidationError("Q-E-TARC P0 did not authorize smoke")
    for key in ("auxiliary_gradient_norm_ratio", "continuation_quota_cosine"):
        if not math.isclose(
            float(config[key]),
            float(p0_protocol["design"][key]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise QetarcTrainingValidationError(f"frozen Q-E-TARC design drift: {key}")

    targets_path = resolve_repo_path(config["safety_targets_path"])
    if base.sha256(targets_path) != config["safety_targets_sha256"]:
        raise QetarcTrainingValidationError("safety target hash differs")
    targets = base.read_jsonl(targets_path)
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    source_ids = list(dict.fromkeys(str(row["sample_id"]) for row in targets))
    order = list(config["training_prompt_order"])
    if len(order) != 40 or len(set(order)) != 40 or set(order) != set(source_ids):
        raise QetarcTrainingValidationError("training prompt-pair order is invalid")
    diagnostic_ids = list(config["diagnostic_sample_ids"])
    if len(diagnostic_ids) != 8 or len(set(diagnostic_ids)) != 8:
        raise QetarcTrainingValidationError("diagnostic prompt registry is invalid")
    if not set(diagnostic_ids).issubset(p0_protocol["selection"]["validation_sample_ids"]):
        raise QetarcTrainingValidationError("diagnostics are not from frozen P0 validation")
    draws = int(protocol["optimizer_steps_per_arm"]) * int(
        config["gradient_accumulation_steps"]
    )
    if int(protocol["safety_pair_draws_per_arm"]) != draws:
        raise QetarcTrainingValidationError("safety draw count differs")
    if set(order[:draws]).intersection(diagnostic_ids):
        raise QetarcTrainingValidationError("smoke draws overlap diagnostic prompts")
    return protocol, config, targets, diagnostic_ids


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if base.git_output("status", "--porcelain"):
        raise QetarcTrainingValidationError("matched stage requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([protocol["config_path"], base.relative(path)])
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    root = resolve_repo_path(protocol["output_root"])
    if root.exists():
        raise QetarcTrainingValidationError(f"prospective output root exists: {root}")
    return base.git_output("rev-parse", "HEAD")


def ordered_pair_features(
    tokenizer: Any, targets: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        targets,
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    by_id = {str(pair["sample_id"]): pair for pair in pairs}
    return [by_id[value] for value in config["training_prompt_order"]], audit


def forward_pair_objectives(
    model: Any,
    pair: dict[str, Any],
    device: Any,
    fp16: bool,
    ramp_width: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute all dynamic and diagnostic objectives from one pair forward."""
    import torch
    import torch.nn.functional as functional

    response_logits = []
    response_targets = []
    uniform_losses = []
    continuation_losses = []
    continuation_tokens = 0
    response_tokens = 0
    for feature in pair["features"]:
        ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        prefix = int(feature["prefix_tokens"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).logits
        current_logits = logits[0, prefix - 1 : ids.shape[1] - 1]
        current_targets = ids[0, prefix:]
        response_logits.append(current_logits)
        response_targets.append(current_targets)
        uniform_losses.append(
            functional.cross_entropy(current_logits.float(), current_targets)
        )
        positions = [
            int(value)
            for value in feature["semantic_positions"]
            if int(value) >= int(feature["route_depth"])
        ]
        if not positions:
            raise QetarcTrainingValidationError("target has no post-route continuation")
        position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        continuation_losses.append(
            functional.cross_entropy(
                current_logits.float().index_select(0, position_tensor),
                current_targets.index_select(0, position_tensor),
            )
        )
        continuation_tokens += len(positions)
        response_tokens += int(current_targets.numel())

    uniform = torch.stack(uniform_losses).mean()
    continuation = torch.stack(continuation_losses).mean()
    tarc_loss, tarc_metrics = etarc_p0._tarc_route_margins(
        response_logits, pair, temperature
    )
    etarc_loss, etarc_metrics = etarc_p0._etarc_path_certificate(
        response_logits, response_targets, pair, temperature
    )
    branch_entropy, entropy_metrics = etarc_p0._branch_objective(
        response_logits, pair, temperature, "branch_entropy"
    )
    branch_negative_gap, gap_metrics = etarc_p0._branch_objective(
        response_logits, pair, temperature, "branch_negative_gap"
    )
    losses = {
        "uniform": uniform,
        "tarc": tarc_loss,
        "etarc": etarc_loss,
        "continuation": continuation,
        "branch_entropy": branch_entropy,
        "branch_negative_gap": branch_negative_gap,
    }
    metrics = {
        "response_tokens": response_tokens,
        "continuation_tokens": continuation_tokens,
        "tarc_margin_terms": int(tarc_metrics["margin_terms"]),
        "etarc_margin_terms": int(etarc_metrics["margin_terms"]),
        "uniform_loss": float(uniform.detach().float().cpu()),
        "tarc_loss": float(tarc_loss.detach().float().cpu()),
        "etarc_loss": float(etarc_loss.detach().float().cpu()),
        "continuation_loss": float(continuation.detach().float().cpu()),
        "branch_entropy_loss": float(branch_entropy.detach().float().cpu()),
        "branch_negative_gap_loss": float(
            branch_negative_gap.detach().float().cpu()
        ),
        "tarc_mean_margin": float(tarc_metrics["mean_margin"]),
        "tarc_certified_node_occurrence_rate": float(
            tarc_metrics["certified_node_occurrence_rate"]
        ),
        "tarc_full_certificate": bool(tarc_metrics["full_trie_certified"]),
        "etarc_existential_path_certificate": float(
            etarc_metrics["existential_path_certificate"]
        ),
        "etarc_complete_route_certified": bool(
            etarc_metrics["complete_route_certified"]
        ),
        "etarc_winning_path_minimum_token_margin": float(
            etarc_metrics["winning_path_minimum_token_margin"]
        ),
        "branch_mean_entropy": float(entropy_metrics["mean_allowed_entropy"]),
        "branch_mean_top1_top2_gap": float(
            gap_metrics["mean_top1_top2_allowed_gap"]
        ),
        "ramp_width_registered": ramp_width,
    }
    return losses, metrics


def gradients_from_objectives(
    losses: dict[str, Any], parameters: list[tuple[str, Any]]
) -> dict[str, dict[str, Any]]:
    import torch

    values = [parameter for _, parameter in parameters]
    result: dict[str, dict[str, Any]] = {}
    for index, objective in enumerate(OBJECTIVES):
        current = torch.autograd.grad(
            losses[objective],
            values,
            retain_graph=index < len(OBJECTIVES) - 1,
            create_graph=False,
            allow_unused=True,
        )
        missing = [
            name for (name, _), gradient in zip(parameters, current, strict=True)
            if gradient is None
        ]
        if missing:
            raise QetarcTrainingValidationError(
                f"{objective} misses {len(missing)} trainable tensors"
            )
        result[objective] = {
            name: gradient.detach()
            for (name, _), gradient in zip(parameters, current, strict=True)
        }
    return result


def gradient_from_loss(
    loss: Any, parameters: list[tuple[str, Any]], label: str
) -> dict[str, Any]:
    import torch

    current = torch.autograd.grad(
        loss,
        [parameter for _, parameter in parameters],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    missing = [
        name for (name, _), gradient in zip(parameters, current, strict=True)
        if gradient is None
    ]
    if missing:
        raise QetarcTrainingValidationError(
            f"{label} misses {len(missing)} trainable tensors"
        )
    return {
        name: gradient.detach()
        for (name, _), gradient in zip(parameters, current, strict=True)
    }


def dynamic_safety_direction(
    gradients: dict[str, dict[str, Any]],
    objective: str,
    auxiliary_ratio: float,
    quota_cosine: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return g_U plus the arm's dynamically equal-budget auxiliary."""
    uniform = gradients["uniform"]
    uniform_norm = gradient_math.gradient_norm(uniform)
    raw_norms = {
        name: gradient_math.gradient_norm(gradients[name]) for name in OBJECTIVES
    }
    if not all(math.isfinite(value) and value > 0.0 for value in raw_norms.values()):
        raise QetarcTrainingValidationError("dynamic objective gradient is invalid")
    target_norm = auxiliary_ratio * uniform_norm
    record: dict[str, Any] = {
        "objective": objective,
        "uniform_gradient_norm": uniform_norm,
        "tarc_gradient_norm": raw_norms["tarc"],
        "etarc_gradient_norm": raw_norms["etarc"],
        "continuation_gradient_norm": raw_norms["continuation"],
        "raw_etarc_continuation_cosine": gradient_math.gradient_cosine(
            gradients["etarc"], gradients["continuation"]
        ),
        "target_auxiliary_norm": target_norm,
        "quota_active": None,
        "achieved_continuation_cosine": None,
        "route_retention_cosine": None,
        "continuation_first_order_dot": None,
        "continuation_dot_lower_bound": None,
    }
    if objective == "uniform":
        record.update(
            {
                "selected_auxiliary_norm": 0.0,
                "auxiliary_norm_ratio": 0.0,
                "budget_relative_error": 0.0,
            }
        )
        return {name: value.clone() for name, value in uniform.items()}, record

    if objective == "tarc":
        scale = target_norm / raw_norms["tarc"]
        auxiliary = gradient_math.scaled_gradient(gradients["tarc"], scale)
    elif objective == "etarc":
        scale = target_norm / raw_norms["etarc"]
        auxiliary = gradient_math.scaled_gradient(gradients["etarc"], scale)
    elif objective == "qetarc":
        auxiliary, quota = qetarc.quota_constrained_direction(
            gradients["etarc"],
            gradients["continuation"],
            quota_cosine,
            target_norm,
        )
        record.update(
            {
                "quota_active": bool(quota["quota_active"]),
                "achieved_continuation_cosine": float(
                    quota["achieved_continuation_cosine"]
                ),
                "route_retention_cosine": float(quota["achieved_route_cosine"]),
                "theoretical_route_retention_cosine": float(
                    quota["theoretical_route_cosine"]
                ),
                "continuation_first_order_dot": float(
                    quota["continuation_first_order_dot"]
                ),
                "continuation_dot_lower_bound": float(
                    quota["continuation_dot_lower_bound"]
                ),
            }
        )
    else:
        raise ValueError(f"unknown Q-E-TARC objective: {objective}")

    auxiliary_norm = gradient_math.gradient_norm(auxiliary)
    record.update(
        {
            "selected_auxiliary_norm": auxiliary_norm,
            "auxiliary_norm_ratio": auxiliary_norm / uniform_norm,
            "budget_relative_error": abs(auxiliary_norm - target_norm) / target_norm,
        }
    )
    if objective != "qetarc":
        record.update(
            {
                "achieved_continuation_cosine": gradient_math.gradient_cosine(
                    auxiliary, gradients["continuation"]
                ),
                "route_retention_cosine": gradient_math.gradient_cosine(
                    auxiliary, gradients["etarc"]
                ),
            }
        )
    return gradient_math.add_gradients(uniform, auxiliary), record


def evaluate_diagnostics(
    model: Any, pairs: list[dict[str, Any]], config: dict[str, Any], stage: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    device = training.model_device(model)
    previous = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for pair in pairs:
            _, metrics = forward_pair_objectives(
                model,
                pair,
                device,
                bool(config["fp16"]),
                int(config["ramp_width"]),
                float(config["softmin_temperature"]),
            )
            rows.append(
                {
                    "stage": stage,
                    "sample_id": str(pair["sample_id"]),
                    "category": str(pair["category"]),
                    **metrics,
                }
            )
    model.train(previous)
    summary = {
        "prompt_pairs": len(rows),
        "target_rows": 2 * len(rows),
        **{
            f"{key}_mean": statistics.fmean(float(row[key]) for row in rows)
            for key in DIAGNOSTIC_LOSSES
        },
        "tarc_mean_margin": statistics.fmean(
            float(row["tarc_mean_margin"]) for row in rows
        ),
        "tarc_certified_node_occurrence_rate": statistics.fmean(
            float(row["tarc_certified_node_occurrence_rate"]) for row in rows
        ),
        "tarc_full_certificate_rate": statistics.fmean(
            int(row["tarc_full_certificate"]) for row in rows
        ),
        "etarc_existential_path_certificate_mean": statistics.fmean(
            float(row["etarc_existential_path_certificate"]) for row in rows
        ),
        "etarc_complete_route_certificate_rate": statistics.fmean(
            int(row["etarc_complete_route_certified"]) for row in rows
        ),
        "branch_mean_entropy": statistics.fmean(
            float(row["branch_mean_entropy"]) for row in rows
        ),
        "branch_mean_top1_top2_gap": statistics.fmean(
            float(row["branch_mean_top1_top2_gap"]) for row in rows
        ),
    }
    return summary, rows


def diagnostic_reductions(
    initial_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    initial = {str(row["sample_id"]): row for row in initial_rows}
    final = {str(row["sample_id"]): row for row in final_rows}
    if set(initial) != set(final):
        raise QetarcTrainingValidationError("diagnostic row IDs changed")
    result = []
    for sample_id in initial:
        row = {
            "sample_id": sample_id,
            "category": str(initial[sample_id]["category"]),
        }
        for metric in DIAGNOSTIC_LOSSES:
            row[f"{metric}_reduction"] = float(initial[sample_id][metric]) - float(
                final[sample_id][metric]
            )
        result.append(row)
    return result


def geometry_summary(
    rows: list[dict[str, Any]], objective: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    gates = protocol["gates"]
    if objective == "uniform":
        return {
            "rows": len(rows),
            "selected_auxiliary_draws": 0,
            "budget_all_pass": True,
        }
    tolerance = float(gates["dynamic_budget_absolute_tolerance"])
    summary: dict[str, Any] = {
        "rows": len(rows),
        "selected_auxiliary_draws": len(rows),
        "auxiliary_norm_ratio_min": min(
            float(row["auxiliary_norm_ratio"]) for row in rows
        ),
        "auxiliary_norm_ratio_max": max(
            float(row["auxiliary_norm_ratio"]) for row in rows
        ),
        "maximum_budget_relative_error": max(
            float(row["budget_relative_error"]) for row in rows
        ),
        "budget_all_pass": all(
            abs(
                float(row["auxiliary_norm_ratio"])
                - float(protocol["design"]["auxiliary_gradient_norm_ratio"])
            )
            <= tolerance
            for row in rows
        ),
    }
    if objective == "qetarc":
        quota = float(protocol["design"]["continuation_quota_cosine"])
        quota_tolerance = float(gates["quota_cosine_absolute_tolerance"])
        bound_tolerance = float(gates["continuation_bound_numeric_tolerance"])
        active = [row for row in rows if row["quota_active"]]
        summary.update(
            {
                "quota_active_draws": len(active),
                "achieved_continuation_cosine_min": min(
                    float(row["achieved_continuation_cosine"]) for row in rows
                ),
                "route_retention_cosine_min": min(
                    float(row["route_retention_cosine"]) for row in rows
                ),
                "quota_all_pass": all(
                    float(row["achieved_continuation_cosine"])
                    + quota_tolerance
                    >= quota
                    for row in rows
                ),
                "active_boundary_all_pass": all(
                    abs(float(row["achieved_continuation_cosine"]) - quota)
                    <= quota_tolerance
                    for row in active
                ),
                "route_retention_all_pass": all(
                    float(row["route_retention_cosine"])
                    >= float(gates["route_retention_cosine_min"])
                    for row in rows
                ),
                "continuation_bound_all_pass": all(
                    float(row["continuation_first_order_dot"])
                    + bound_tolerance
                    >= float(row["continuation_dot_lower_bound"])
                    for row in rows
                ),
            }
        )
    return summary


def train_arm(
    arm_name: str,
    protocol: dict[str, Any],
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    diagnostic_ids: list[str],
    runtime_head: str,
) -> dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    started = time.time()
    output = resolve_repo_path(protocol["arm_outputs"][arm_name])
    output.mkdir(parents=True, exist_ok=False)
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    device = training.model_device(model)
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
        raise QetarcTrainingValidationError("unexpected trainable parameter inventory")
    initial_state = model_common.trainable_state_hash(parameters)

    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["train_examples"])
    )
    if len(risk_rows) != int(config["train_examples"]):
        raise QetarcTrainingValidationError("risk row count differs")
    risk_features = [
        training.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(config["max_length"])
        )
        for row in risk_rows
    ]
    safety_pairs, structural = ordered_pair_features(tokenizer, targets, config)
    p0_structural = base.read_json(
        resolve_repo_path(config["p0_structural_audit"])
    )
    if structural != p0_structural:
        raise QetarcTrainingValidationError("training route structure differs from P0")
    by_id = {str(pair["sample_id"]): pair for pair in safety_pairs}
    diagnostic_pairs = [by_id[value] for value in diagnostic_ids]
    initial_diagnostic, initial_rows = evaluate_diagnostics(
        model, diagnostic_pairs, config, "initial"
    )

    optimizer = frozen.make_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["warmup_steps"]),
        num_training_steps=int(config["optimizer_steps"]),
    )
    risk_order = frozen.risk_order(len(risk_rows), int(config["seed"]))
    steps = int(protocol["optimizer_steps_per_arm"])
    microsteps = int(config["gradient_accumulation_steps"])
    safety_mix = float(config["safety_mix"])
    risk_coefficient = (1.0 - safety_mix) / microsteps
    safety_coefficient = safety_mix / microsteps
    objective = str(config["arms"][arm_name]["objective"])
    target_position = 0
    target_exposure: Counter[str] = Counter()
    total_response_tokens = 0
    total_continuation_tokens = 0
    available_tarc_terms = 0
    available_etarc_terms = 0
    trace: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    trainable = [parameter for _, parameter in parameters]
    optimizer.zero_grad(set_to_none=True)

    for global_step in range(steps):
        step_started = time.time()
        accumulator = {
            name: torch.zeros_like(parameter)
            for name, parameter in parameters
        }
        risk_losses = []
        uniform_losses = []
        tarc_losses = []
        etarc_losses = []
        continuation_losses = []
        step_targets = []
        for micro_index in range(microsteps):
            position = global_step * microsteps + micro_index
            model.train()
            current_risk = base.risk_loss(
                model, risk_features[risk_order[position]], device, bool(config["fp16"])
            )
            risk_gradient = gradient_from_loss(current_risk, parameters, "risk")
            for name in accumulator:
                accumulator[name].add_(risk_gradient[name], alpha=risk_coefficient)
            risk_losses.append(float(current_risk.detach().float().cpu()))

            pair = safety_pairs[target_position % len(safety_pairs)]
            target_position += 1
            sample_id = str(pair["sample_id"])
            target_exposure[sample_id] += 1
            step_targets.append(sample_id)
            losses, metrics = forward_pair_objectives(
                model,
                pair,
                device,
                bool(config["fp16"]),
                int(config["ramp_width"]),
                float(config["softmin_temperature"]),
            )
            objective_gradients = gradients_from_objectives(losses, parameters)
            safety_gradient, geometry = dynamic_safety_direction(
                objective_gradients,
                objective,
                float(config["auxiliary_gradient_norm_ratio"]),
                float(config["continuation_quota_cosine"]),
            )
            for name in accumulator:
                accumulator[name].add_(safety_gradient[name], alpha=safety_coefficient)
            geometry_rows.append(
                {
                    "optimizer_step": global_step + 1,
                    "microbatch": micro_index + 1,
                    "arm": arm_name,
                    "sample_id": sample_id,
                    **geometry,
                }
            )
            uniform_losses.append(metrics["uniform_loss"])
            tarc_losses.append(metrics["tarc_loss"])
            etarc_losses.append(metrics["etarc_loss"])
            continuation_losses.append(metrics["continuation_loss"])
            total_response_tokens += int(metrics["response_tokens"])
            total_continuation_tokens += int(metrics["continuation_tokens"])
            available_tarc_terms += int(metrics["tarc_margin_terms"])
            available_etarc_terms += int(metrics["etarc_margin_terms"])
            del current_risk, risk_gradient, losses, objective_gradients, safety_gradient

        for name, parameter in parameters:
            parameter.grad = accumulator[name]
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if not finite:
            raise FloatingPointError(
                f"non-finite or missing Q-E-TARC gradient at step {global_step + 1}"
            )
        preclip = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        trace.append(
            {
                "optimizer_step": global_step + 1,
                "arm": arm_name,
                "objective": objective,
                "risk_loss_mean": statistics.fmean(risk_losses),
                "safety_uniform_loss_mean": statistics.fmean(uniform_losses),
                "tarc_loss_mean": statistics.fmean(tarc_losses),
                "etarc_loss_mean": statistics.fmean(etarc_losses),
                "continuation_loss_mean": statistics.fmean(continuation_losses),
                "gradient_preclip_norm": preclip,
                "learning_rate_after_update": float(scheduler.get_last_lr()[0]),
                "sample_ids_sha256": training.hash_json(step_targets),
                "wall_seconds": time.time() - step_started,
            }
        )
        base.write_csv(output / "training_trace.csv", trace)
        write_jsonl(output / "dynamic_gradient_geometry.jsonl", geometry_rows)
        print(
            f"Q-E-TARC {arm_name} step {global_step + 1}/{steps} "
            f"risk={trace[-1]['risk_loss_mean']:.6f} "
            f"etarc={trace[-1]['etarc_loss_mean']:.6f} "
            f"cont={trace[-1]['continuation_loss_mean']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_diagnostic, final_rows = evaluate_diagnostics(
        model, diagnostic_pairs, config, "final"
    )
    paired_reductions = diagnostic_reductions(initial_rows, final_rows)
    write_jsonl(output / "diagnostic_rows.jsonl", initial_rows + final_rows)
    final_state = model_common.trainable_state_hash(parameters)
    checkpoint = base.save_adapter(model, output, steps)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    reductions = {
        f"{metric}_reduction": float(initial_diagnostic[f"{metric}_mean"])
        - float(final_diagnostic[f"{metric}_mean"])
        for metric in DIAGNOSTIC_LOSSES
    }
    manifest = {
        "name": "step3_qetarc_matched_arm_manifest_v2",
        "experiment_mode": "exploratory_non_confirmatory_reused_development_split",
        "runtime_head": runtime_head,
        "arm": arm_name,
        "objective": objective,
        "protocol_sha256": base.sha256(resolve_repo_path(protocol["protocol_path"])),
        "config_sha256": base.sha256(resolve_repo_path(protocol["config_path"])),
        "completed_steps": steps,
        "optimizer_steps": steps,
        "skipped_updates": 0,
        "manual_gradient_composition": True,
        "all_losses_and_gradients_finite": True,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "final_adapter_sha256": checkpoint["adapter_sha256"],
        "final_checkpoint": checkpoint["checkpoint"],
        "risk_order_sha256": training.hash_json(risk_order),
        "target_order_sha256": training.hash_json(config["training_prompt_order"]),
        "target_exposure_sha256": training.hash_json(dict(sorted(target_exposure.items()))),
        "safety_pair_draws": target_position,
        "safety_target_rows_forwarded": 2 * target_position,
        "safety_response_tokens": total_response_tokens,
        "continuation_tokens": total_continuation_tokens,
        "available_tarc_margin_terms": available_tarc_terms,
        "available_etarc_margin_terms": available_etarc_terms,
        "computed_objective_gradient_draws": {
            name: target_position for name in OBJECTIVES
        },
        "selected_auxiliary": objective if objective != "uniform" else None,
        "geometry": geometry_summary(geometry_rows, objective, protocol),
        "first_step_risk_loss_mean": trace[0]["risk_loss_mean"],
        "first_step_safety_uniform_loss_mean": trace[0][
            "safety_uniform_loss_mean"
        ],
        "initial_diagnostic": initial_diagnostic,
        "final_diagnostic": final_diagnostic,
        "reductions": reductions,
        "paired_diagnostic_reductions": paired_reductions,
        "trainable_parameter_inventory": inventory,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory
        >= float(config["minimum_vram_headroom_fraction"]),
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "wall_clock_seconds": time.time() - started,
    }
    base.write_json(output / "run_manifest.json", manifest)
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def paired_advantage(
    left: dict[str, Any],
    right: dict[str, Any],
    metric: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    left_rows = {
        str(row["sample_id"]): row for row in left["paired_diagnostic_reductions"]
    }
    right_rows = {
        str(row["sample_id"]): row for row in right["paired_diagnostic_reductions"]
    }
    if set(left_rows) != set(right_rows):
        raise QetarcTrainingValidationError("paired diagnostic IDs differ between arms")
    differences = [
        float(left_rows[sample_id][metric]) - float(right_rows[sample_id][metric])
        for sample_id in left_rows
    ]
    return {
        "metric": metric,
        "left_arm": left["arm"],
        "right_arm": right["arm"],
        "differences": differences,
        "positive_rate": sum(value > 0.0 for value in differences) / len(differences),
        "bootstrap": etarc_p0.paired_bootstrap(
            differences,
            int(protocol["statistics"]["bootstrap_samples"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        ),
        "sign_test": etarc_p0.exact_sign_test(differences),
    }


def compare_manifests(
    manifests: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    arms = list(protocol["arm_order"])
    rows = [manifests[arm] for arm in arms]
    tolerance = float(protocol["gates"]["matching_tolerance"])

    def all_equal(field: str) -> bool:
        return len({json.dumps(row[field], sort_keys=True) for row in rows}) == 1

    def spread(field: str) -> float:
        values = [float(row[field]) for row in rows]
        return max(values) - min(values)

    initial_spreads = {
        field: max(float(row["initial_diagnostic"][field]) for row in rows)
        - min(float(row["initial_diagnostic"][field]) for row in rows)
        for field in rows[0]["initial_diagnostic"]
        if isinstance(rows[0]["initial_diagnostic"][field], (int, float))
    }
    route_pair = paired_advantage(
        manifests["qetarc"],
        manifests["tarc"],
        "etarc_loss_reduction",
        protocol,
    )
    continuation_tarc_pair = paired_advantage(
        manifests["qetarc"],
        manifests["tarc"],
        "continuation_loss_reduction",
        protocol,
    )
    continuation_etarc_pair = paired_advantage(
        manifests["qetarc"],
        manifests["etarc"],
        "continuation_loss_reduction",
        protocol,
    )
    q_reductions = manifests["qetarc"]["reductions"]
    e_reductions = manifests["etarc"]["reductions"]
    t_reductions = manifests["tarc"]["reductions"]
    raw_route = float(e_reductions["etarc_loss_reduction"])
    q_route = float(q_reductions["etarc_loss_reduction"])
    route_fraction = q_route / raw_route if raw_route > 0.0 else math.nan
    gates = protocol["gates"]
    assignments_correct = (
        manifests["uniform"]["selected_auxiliary"] is None
        and manifests["uniform"]["geometry"]["selected_auxiliary_draws"] == 0
        and all(
            manifests[name]["selected_auxiliary"] == name
            and manifests[name]["geometry"]["selected_auxiliary_draws"]
            == int(protocol["safety_pair_draws_per_arm"])
            for name in ("tarc", "etarc", "qetarc")
        )
    )
    dynamic_geometry = (
        all(manifests[name]["geometry"]["budget_all_pass"] for name in arms)
        and manifests["qetarc"]["geometry"]["quota_all_pass"]
        and manifests["qetarc"]["geometry"]["active_boundary_all_pass"]
        and manifests["qetarc"]["geometry"]["route_retention_all_pass"]
        and manifests["qetarc"]["geometry"]["continuation_bound_all_pass"]
    )
    route_effect = (
        q_route >= float(gates["qetarc_etarc_loss_reduction_min"])
        and route_fraction >= float(gates["route_reduction_fraction_of_raw_etarc_min"])
        and float(route_pair["bootstrap"]["mean"])
        >= float(gates["paired_advantage_mean_min"])
        and float(route_pair["bootstrap"]["ci95_lower"])
        > float(gates["paired_advantage_ci_lower_min"])
        and float(route_pair["sign_test"]["exact_two_sided_p"])
        <= float(gates["paired_sign_p_max"])
    )
    continuation_effect = all(
        float(item["bootstrap"]["mean"])
        >= float(gates["paired_advantage_mean_min"])
        and float(item["bootstrap"]["ci95_lower"])
        > float(gates["paired_advantage_ci_lower_min"])
        and float(item["sign_test"]["exact_two_sided_p"])
        <= float(gates["paired_sign_p_max"])
        for item in (continuation_tarc_pair, continuation_etarc_pair)
    )
    branch_mechanism = (
        float(q_reductions["branch_entropy_loss_reduction"])
        > float(gates["branch_objective_reduction_min"])
        and float(q_reductions["branch_negative_gap_loss_reduction"])
        > float(gates["branch_objective_reduction_min"])
    )
    return {
        "all_arms_present": set(manifests) == set(arms),
        "steps_exact": all(
            row["completed_steps"] == int(protocol["optimizer_steps_per_arm"])
            for row in rows
        ),
        "no_skipped_updates": all(row["skipped_updates"] == 0 for row in rows),
        "finite": all(row["all_losses_and_gradients_finite"] for row in rows),
        "vram": all(row["vram_gate_pass"] for row in rows),
        "initial_state_equal": all_equal("initial_trainable_state_sha256"),
        "risk_order_equal": all_equal("risk_order_sha256"),
        "target_order_equal": all_equal("target_order_sha256"),
        "target_exposure_equal": all_equal("target_exposure_sha256"),
        "safety_pair_draws_equal": all_equal("safety_pair_draws"),
        "safety_target_rows_equal": all_equal("safety_target_rows_forwarded"),
        "safety_response_tokens_equal": all_equal("safety_response_tokens"),
        "continuation_tokens_equal": all_equal("continuation_tokens"),
        "available_tarc_terms_equal": all_equal("available_tarc_margin_terms"),
        "available_etarc_terms_equal": all_equal("available_etarc_margin_terms"),
        "objective_gradient_counts_equal": all_equal(
            "computed_objective_gradient_draws"
        ),
        "first_step_risk_loss_equal": spread("first_step_risk_loss_mean")
        <= tolerance,
        "first_step_uniform_safety_loss_equal": spread(
            "first_step_safety_uniform_loss_mean"
        )
        <= tolerance,
        "initial_diagnostics_equal": max(initial_spreads.values()) <= tolerance,
        "assignment_counts_correct": assignments_correct,
        "distinct_final_adapter_hashes": len(
            {row["final_adapter_sha256"] for row in rows}
        )
        == len(rows),
        "sealed_data_untouched": all(
            not row["core100_opened"]
            and not row["heldout_opened"]
            and not row["candidate_outputs_generated"]
            and not row["main_training_authorized"]
            for row in rows
        ),
        "dynamic_geometry": dynamic_geometry,
        "qetarc_route_reduction": q_route,
        "etarc_route_reduction": raw_route,
        "tarc_route_reduction": float(t_reductions["etarc_loss_reduction"]),
        "qetarc_route_reduction_fraction_of_raw_etarc": route_fraction,
        "qetarc_continuation_reduction": float(
            q_reductions["continuation_loss_reduction"]
        ),
        "tarc_continuation_reduction": float(
            t_reductions["continuation_loss_reduction"]
        ),
        "etarc_continuation_reduction": float(
            e_reductions["continuation_loss_reduction"]
        ),
        "qetarc_branch_entropy_reduction": float(
            q_reductions["branch_entropy_loss_reduction"]
        ),
        "qetarc_branch_negative_gap_reduction": float(
            q_reductions["branch_negative_gap_loss_reduction"]
        ),
        "paired_qetarc_route_advantage_over_tarc": route_pair,
        "paired_qetarc_continuation_advantage_over_tarc": continuation_tarc_pair,
        "paired_qetarc_continuation_advantage_over_etarc": continuation_etarc_pair,
        "route_effect": route_effect,
        "continuation_effect": continuation_effect,
        "branch_mechanism": branch_mechanism,
    }


def select_smoke_route(checks: dict[str, Any]) -> str:
    matching = (
        "all_arms_present",
        "initial_state_equal",
        "risk_order_equal",
        "target_order_equal",
        "target_exposure_equal",
        "safety_pair_draws_equal",
        "safety_target_rows_equal",
        "safety_response_tokens_equal",
        "continuation_tokens_equal",
        "available_tarc_terms_equal",
        "available_etarc_terms_equal",
        "objective_gradient_counts_equal",
        "first_step_risk_loss_equal",
        "first_step_uniform_safety_loss_equal",
        "initial_diagnostics_equal",
        "sealed_data_untouched",
    )
    numerical = ("steps_exact", "no_skipped_updates", "finite", "vram")
    if not all(checks[name] for name in matching):
        return "STOP_QETARC_SMOKE_MATCHING_FAILED"
    if not all(checks[name] for name in numerical):
        return "STOP_QETARC_SMOKE_NUMERICAL_FAILED"
    if not (
        checks["assignment_counts_correct"]
        and checks["distinct_final_adapter_hashes"]
    ):
        return "STOP_QETARC_SMOKE_ASSIGNMENT_NOT_EXERCISED"
    if not checks["dynamic_geometry"]:
        return "STOP_QETARC_SMOKE_DYNAMIC_GEOMETRY_FAILED"
    if not checks["route_effect"]:
        return "STOP_QETARC_SMOKE_ROUTE_EFFECT_FAILED"
    if not checks["continuation_effect"]:
        return "STOP_QETARC_SMOKE_CONTINUATION_EFFECT_FAILED"
    if not checks["branch_mechanism"]:
        return "STOP_QETARC_SMOKE_BRANCH_MECHANISM_FAILED"
    return "PASS_QETARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT_PROTOCOL"


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, diagnostic_ids = context(path)
    if not torch.cuda.is_available():
        raise QetarcTrainingValidationError("CUDA is unavailable")
    result = {
        "status": "PASS_QETARC_MATCHED_SMOKE_PREFLIGHT",
        "protocol_sha256": base.sha256(path),
        "arms": protocol["arm_order"],
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_authorized_optimizer_steps": int(protocol["optimizer_steps_per_arm"])
        * len(protocol["arm_order"]),
        "safety_pair_draws_per_arm": protocol["safety_pair_draws_per_arm"],
        "dynamic_gradients_per_draw": list(OBJECTIVES),
        "source_rows": len(targets),
        "diagnostic_prompt_pairs": len(diagnostic_ids),
        "training_diagnostic_prompt_overlap": 0,
        "auxiliary_gradient_norm_ratio": config["auxiliary_gradient_norm_ratio"],
        "continuation_quota_cosine": config["continuation_quota_cosine"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "gpu": torch.cuda.get_device_name(0),
        "candidate_generation_authorized": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def artifact_inventory(root: Path, excluded: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": base.relative(path),
            "sha256": base.sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() != excluded.resolve()
    ]


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, config, targets, diagnostic_ids = context(path)
    runtime_head = require_clean_start(path, protocol)
    manifests = {
        arm: train_arm(
            arm, protocol, config, targets, diagnostic_ids, runtime_head
        )
        for arm in protocol["arm_order"]
    }
    checks = compare_manifests(manifests, protocol)
    route = select_smoke_route(checks)
    decision = {
        "name": "step3_qetarc_matched_smoke_decision_v2",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": base.sha256(path),
        "checks": checks,
        "arms": {
            arm: {
                "objective": manifest["objective"],
                "completed_steps": manifest["completed_steps"],
                "optimizer_steps": manifest["optimizer_steps"],
                "final_adapter_sha256": manifest["final_adapter_sha256"],
                "final_checkpoint": manifest["final_checkpoint"],
                "geometry": manifest["geometry"],
                "reductions": manifest["reductions"],
                "vram_headroom_fraction": manifest["vram_headroom_fraction"],
            }
            for arm, manifest in manifests.items()
        },
        "total_optimizer_steps": sum(
            int(manifest["optimizer_steps"]) for manifest in manifests.values()
        ),
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "next_authorized": (
            "freeze_and_push_separate_matched_QETARC_development_protocol"
            if route == "PASS_QETARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT_PROTOCOL"
            else "none"
        ),
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    reproduction_path = resolve_repo_path(
        protocol["outputs"]["reproduction_manifest"]
    )
    base.write_json(decision_path, decision)
    route_pair = checks["paired_qetarc_route_advantage_over_tarc"]
    cont_t = checks["paired_qetarc_continuation_advantage_over_tarc"]
    cont_e = checks["paired_qetarc_continuation_advantage_over_etarc"]
    report = f"""# Q-E-TARC Matched Two-Step Smoke Report

Route: `{route}`.

## Matched finite-step result

Uniform, TARC, raw E-TARC, and Q-E-TARC each completed two matched optimizer
steps. Q-E-TARC reduced held-out diagnostic E-TARC loss by
`{checks['qetarc_route_reduction']:.6f}`, retaining
`{checks['qetarc_route_reduction_fraction_of_raw_etarc']:.3%}` of raw
E-TARC's `{checks['etarc_route_reduction']:.6f}` reduction. TARC reduced the
same loss by `{checks['tarc_route_reduction']:.6f}`. The prompt-paired
Q-E-TARC advantage over TARC was `{route_pair['bootstrap']['mean']:.6f}`
with 95% bootstrap interval
`[{route_pair['bootstrap']['ci95_lower']:.6f},
{route_pair['bootstrap']['ci95_upper']:.6f}]` and exact sign-test
`p={route_pair['sign_test']['exact_two_sided_p']:.8g}`.

Q-E-TARC continuation reduction was
`{checks['qetarc_continuation_reduction']:.6f}`, versus
`{checks['tarc_continuation_reduction']:.6f}` for TARC and
`{checks['etarc_continuation_reduction']:.6f}` for raw E-TARC. Its paired
advantages were `{cont_t['bootstrap']['mean']:.6f}` over TARC and
`{cont_e['bootstrap']['mean']:.6f}` over E-TARC, with bootstrap lower bounds
`{cont_t['bootstrap']['ci95_lower']:.6f}` and
`{cont_e['bootstrap']['ci95_lower']:.6f}`.

## Mechanism trace

Every safety draw recomputed `g_U`, `g_T`, `g_E`, and `g_C`. The Q-E-TARC
minimum achieved continuation cosine was
`{manifests['qetarc']['geometry']['achieved_continuation_cosine_min']:.8f}`;
minimum route retention was
`{manifests['qetarc']['geometry']['route_retention_cosine_min']:.8f}`; and
maximum auxiliary budget relative error was
`{manifests['qetarc']['geometry']['maximum_budget_relative_error']:.3e}`.
Q-E-TARC branch-entropy and branch-negative-gap reductions were
`{checks['qetarc_branch_entropy_reduction']:.6f}` and
`{checks['qetarc_branch_negative_gap_reduction']:.6f}`.

## Evidence boundary

The eight diagnostic prompt pairs were excluded from all smoke updates, but
they came from a previously disclosed development split. This is real-model,
finite-step, teacher-forced mechanism evidence, not a fresh confirmatory or
generated-behavior result. No candidate, Core100, or heldout data was opened.
Total optimizer steps: `{decision['total_optimizer_steps']}`;
`development_training_authorized=false`; `main_training_authorized=false`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report.encode("utf-8"))
    root = resolve_repo_path(protocol["output_root"])
    reproduction = {
        "name": "step3_qetarc_matched_smoke_reproduction_manifest_v2",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": base.relative(path),
        "protocol_sha256": base.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": base.sha256(resolve_repo_path(protocol["config_path"])),
        "command": (
            "python experiments/step3_response_quality_persistence/scripts/"
            "step3_qetarc_training.py run"
        ),
        "verify_command": (
            "python experiments/step3_response_quality_persistence/scripts/"
            "step3_qetarc_training.py verify-run"
        ),
        "seed": config["seed"],
        "parameters": protocol["design"],
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "artifacts": artifact_inventory(root, reproduction_path),
    }
    base.write_json(reproduction_path, reproduction)
    print(json.dumps({"route": route, "checks": checks}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    root = resolve_repo_path(protocol["output_root"])
    manifests = {
        arm: base.read_json(resolve_repo_path(protocol["arm_outputs"][arm]) / "run_manifest.json")
        for arm in protocol["arm_order"]
    }
    checks = compare_manifests(manifests, protocol)
    route = select_smoke_route(checks)
    decision = base.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    reproduction_path = resolve_repo_path(
        protocol["outputs"]["reproduction_manifest"]
    )
    reproduction = base.read_json(reproduction_path)
    if decision.get("route") != route or decision.get("checks") != checks:
        raise QetarcTrainingValidationError("stored and recomputed decision differs")
    if reproduction.get("route") != route:
        raise QetarcTrainingValidationError("reproduction route differs")
    if reproduction.get("artifacts") != artifact_inventory(root, reproduction_path):
        raise QetarcTrainingValidationError("result artifact inventory differs")
    result = {
        "status": "PASS_QETARC_MATCHED_SMOKE_VERIFY",
        "route": route,
        "arms": list(manifests),
        "total_optimizer_steps": sum(
            int(manifest["optimizer_steps"]) for manifest in manifests.values()
        ),
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
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
