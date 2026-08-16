"""Run hash-bound matched uniform/RAMP/TARC training stages."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_ramp_p0 as ramp_p0
import step3_ramp_training as base
import step3_tarc_common as tarc
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_tarc/"
    "protocol/tarc_smoke_protocol.json"
)


class TarcTrainingValidationError(ValueError):
    """Raised when matched TARC training violates its frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    return base.resolve_repo_path(value)


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    protocol = base.read_json(path)
    if protocol.get("status") != "prospective_before_any_matched_TARC_stage_training":
        raise TarcTrainingValidationError("training protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise TarcTrainingValidationError(f"frozen input hash mismatch: {name}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise TarcTrainingValidationError("entry commit is not an ancestor of HEAD")
    config = base.read_json(resolve_repo_path(protocol["config_path"]))
    run_kind = str(protocol["run_kind"])
    if (
        config.get(f"{run_kind}_training_authorized") is not True
        or protocol.get("training_authorized") is not True
    ):
        raise TarcTrainingValidationError(f"{run_kind} training is not authorized")
    if (
        config.get("main_training_authorized") is not False
        or config.get("candidate_generation_authorized") is not False
        or config.get("core100_opened") is not False
        or config.get("heldout_opened") is not False
    ):
        raise TarcTrainingValidationError("config violates the sealed boundary")
    if int(protocol["optimizer_steps_per_arm"]) != int(config[f"{run_kind}_steps"]):
        raise TarcTrainingValidationError("protocol step count differs from config")
    arms = list(protocol["arm_order"])
    if arms != ["uniform", "ramp", "tarc"] or set(config["arms"]) != set(arms):
        raise TarcTrainingValidationError("matched arm registry differs")

    p0_route_path = resolve_repo_path(config["p0_route"])
    if base.sha256(p0_route_path) != config["p0_route_sha256"]:
        raise TarcTrainingValidationError("P0 route hash differs from config")
    p0_route = base.read_json(p0_route_path)
    if p0_route.get("route") != "PASS_TARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL":
        raise TarcTrainingValidationError("TARC-P0 did not authorize matched smoke")
    for key in ("ramp_coefficient", "tarc_coefficient"):
        if not math.isclose(
            float(config[key]),
            float(p0_route["coefficients"][key]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise TarcTrainingValidationError(f"frozen P0 coefficient drift: {key}")

    targets_path = resolve_repo_path(config["safety_targets_path"])
    if base.sha256(targets_path) != config["safety_targets_sha256"]:
        raise TarcTrainingValidationError("safety target hash differs from config")
    targets = base.read_jsonl(targets_path)
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    source_ids = list(dict.fromkeys(str(row["sample_id"]) for row in targets))
    order = list(config["training_prompt_order"])
    if len(order) != 40 or len(set(order)) != 40 or set(order) != set(source_ids):
        raise TarcTrainingValidationError("training prompt-pair order is invalid")
    diagnostic_ids = list(config["diagnostic_sample_ids"])
    p0_protocol = base.read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p0_protocol"]["path"])
    )
    if diagnostic_ids != p0_protocol["selection"]["validation_sample_ids"]:
        raise TarcTrainingValidationError("diagnostic IDs differ from TARC-P0 validation")
    draws = int(protocol["optimizer_steps_per_arm"]) * int(
        config["gradient_accumulation_steps"]
    )
    if int(protocol["safety_pair_draws_per_arm"]) != draws:
        raise TarcTrainingValidationError("safety pair draw count differs")
    if run_kind == "smoke" and set(order[:draws]).intersection(diagnostic_ids):
        raise TarcTrainingValidationError("smoke draws overlap diagnostic prompts")
    return protocol, config, targets, diagnostic_ids


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if base.git_output("status", "--porcelain"):
        raise TarcTrainingValidationError("matched stage requires a clean worktree")
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
        raise TarcTrainingValidationError(f"prospective output root exists: {root}")
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
    by_id = {pair["sample_id"]: pair for pair in pairs}
    return [by_id[value] for value in config["training_prompt_order"]], audit


def compose_safety_loss(
    uniform: Any,
    ramp_loss: Any,
    tarc_loss: Any,
    objective: str,
    ramp_coefficient: float,
    tarc_coefficient: float,
) -> Any:
    if objective == "uniform":
        return uniform
    if objective == "ramp":
        return uniform + ramp_coefficient * ramp_loss
    if objective == "tarc":
        return uniform + tarc_coefficient * tarc_loss
    raise ValueError(f"unknown TARC objective: {objective}")


def pair_losses(
    model: Any,
    pair: dict[str, Any],
    device: Any,
    fp16: bool,
    objective: str,
    ramp_width: int,
    temperature: float,
    ramp_coefficient: float,
    tarc_coefficient: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    response_logits = []
    uniform_losses = []
    ramp_losses = []
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
        uniform_losses.append(
            functional.cross_entropy(current_logits.float(), current_targets)
        )
        current_ramp, _ = ramp_p0.response_objectives(
            current_logits,
            current_targets,
            list(feature["semantic_positions"]),
            ramp_width,
            "prefix",
        )
        ramp_losses.append(current_ramp)
        response_tokens += int(current_targets.numel())
    uniform = torch.stack(uniform_losses).mean()
    ramp_loss = torch.stack(ramp_losses).mean()
    margins = []
    unique_node_margins = []
    for node in pair["nodes"]:
        node_values = []
        for path_index in node["path_indices"]:
            value = tarc.conservative_allowed_margin(
                response_logits[path_index][int(node["depth"])],
                list(node["allowed_token_ids"]),
                temperature,
            )
            margins.append(value)
            node_values.append(value)
        unique_node_margins.append(torch.stack(node_values).mean())
    margin_tensor = torch.stack(margins)
    tarc_loss = tarc.certificate_barrier(margin_tensor)
    selected = compose_safety_loss(
        uniform,
        ramp_loss,
        tarc_loss,
        objective,
        ramp_coefficient,
        tarc_coefficient,
    )
    detached = margin_tensor.detach().float()
    return {
        "selected": selected,
        "uniform": uniform,
        "ramp": ramp_loss,
        "tarc": tarc_loss,
        "response_tokens": response_tokens,
        "available_ramp_terms": 2 * ramp_width,
        "available_tarc_terms": int(margin_tensor.numel()),
        "tarc_mean_margin": float(detached.mean().cpu()),
        "tarc_certified_node_occurrence_rate": float(
            (detached > 0).float().mean().cpu()
        ),
        "tarc_full_certificate": bool(
            (torch.stack(unique_node_margins).detach() > 0).all().cpu()
        ),
    }


def evaluate_diagnostics(
    model: Any, pairs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    import torch

    device = training.model_device(model)
    previous = model.training
    model.eval()
    uniform_losses = []
    ramp_losses = []
    tarc_losses = []
    margins = []
    certified = []
    full = 0
    with torch.no_grad():
        for pair in pairs:
            losses = pair_losses(
                model,
                pair,
                device,
                bool(config["fp16"]),
                "uniform",
                int(config["ramp_width"]),
                float(config["softmin_temperature"]),
                float(config["ramp_coefficient"]),
                float(config["tarc_coefficient"]),
            )
            uniform_losses.append(float(losses["uniform"].float().cpu()))
            ramp_losses.append(float(losses["ramp"].float().cpu()))
            tarc_losses.append(float(losses["tarc"].float().cpu()))
            margins.append(float(losses["tarc_mean_margin"]))
            certified.append(float(losses["tarc_certified_node_occurrence_rate"]))
            full += int(losses["tarc_full_certificate"])
    model.train(previous)
    return {
        "prompt_pairs": len(pairs),
        "target_rows": 2 * len(pairs),
        "uniform_loss_mean": sum(uniform_losses) / len(uniform_losses),
        "ramp_margin_loss_mean": sum(ramp_losses) / len(ramp_losses),
        "tarc_margin_loss_mean": sum(tarc_losses) / len(tarc_losses),
        "tarc_mean_margin": sum(margins) / len(margins),
        "tarc_certified_node_occurrence_rate": sum(certified) / len(certified),
        "tarc_full_certificate_rate": full / len(pairs),
    }


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
        raise TarcTrainingValidationError("unexpected trainable parameter inventory")
    initial_state = model_common.trainable_state_hash(parameters)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["train_examples"])
    )
    if len(risk_rows) != int(config["train_examples"]):
        raise TarcTrainingValidationError("risk row count differs")
    risk_features = [
        training.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(config["max_length"])
        )
        for row in risk_rows
    ]
    safety_pairs, structural = ordered_pair_features(tokenizer, targets, config)
    p0_structural = base.read_json(resolve_repo_path(config["p0_route"]))[
        "structural_audit"
    ]
    if structural != p0_structural:
        raise TarcTrainingValidationError("training route structure differs from P0")
    by_id = {pair["sample_id"]: pair for pair in safety_pairs}
    diagnostic_pairs = [by_id[value] for value in diagnostic_ids]
    initial_diagnostic = evaluate_diagnostics(model, diagnostic_pairs, config)

    optimizer = frozen.make_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["warmup_steps"]),
        num_training_steps=int(config["optimizer_steps"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["fp16"]))
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
    available_ramp_terms = 0
    available_tarc_terms = 0
    active_ramp_terms = 0
    active_tarc_terms = 0
    skipped_updates = 0
    trace = []
    optimizer.zero_grad(set_to_none=True)

    for global_step in range(steps):
        step_started = time.time()
        risk_losses = []
        selected_losses = []
        uniform_losses = []
        ramp_losses = []
        tarc_losses = []
        step_targets = []
        for micro_index in range(microsteps):
            position = global_step * microsteps + micro_index
            model.train()
            current_risk = base.risk_loss(
                model, risk_features[risk_order[position]], device, bool(config["fp16"])
            )
            scaler.scale(current_risk * risk_coefficient).backward()
            risk_losses.append(float(current_risk.detach().float().cpu()))
            pair = safety_pairs[target_position % len(safety_pairs)]
            target_position += 1
            sample_id = str(pair["sample_id"])
            target_exposure[sample_id] += 1
            step_targets.append(sample_id)
            losses = pair_losses(
                model,
                pair,
                device,
                bool(config["fp16"]),
                objective,
                int(config["ramp_width"]),
                float(config["softmin_temperature"]),
                float(config["ramp_coefficient"]),
                float(config["tarc_coefficient"]),
            )
            scaler.scale(losses["selected"] * safety_coefficient).backward()
            selected_losses.append(float(losses["selected"].detach().float().cpu()))
            uniform_losses.append(float(losses["uniform"].detach().float().cpu()))
            ramp_losses.append(float(losses["ramp"].detach().float().cpu()))
            tarc_losses.append(float(losses["tarc"].detach().float().cpu()))
            total_response_tokens += int(losses["response_tokens"])
            available_ramp_terms += int(losses["available_ramp_terms"])
            available_tarc_terms += int(losses["available_tarc_terms"])
            active_ramp_terms += int(objective == "ramp") * int(
                losses["available_ramp_terms"]
            )
            active_tarc_terms += int(objective == "tarc") * int(
                losses["available_tarc_terms"]
            )

        scaler.unscale_(optimizer)
        trainable = [parameter for _, parameter in parameters]
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if not finite:
            raise FloatingPointError(
                f"non-finite or missing TARC gradient at step {global_step + 1}"
            )
        preclip = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
        )
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError(f"GradScaler skipped TARC step {global_step + 1}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        trace.append(
            {
                "optimizer_step": global_step + 1,
                "arm": arm_name,
                "objective": objective,
                "risk_loss_mean": sum(risk_losses) / len(risk_losses),
                "safety_selected_loss_mean": sum(selected_losses) / len(selected_losses),
                "safety_uniform_loss_mean": sum(uniform_losses) / len(uniform_losses),
                "ramp_margin_loss_mean": sum(ramp_losses) / len(ramp_losses),
                "tarc_margin_loss_mean": sum(tarc_losses) / len(tarc_losses),
                "gradient_preclip_norm": preclip,
                "learning_rate_after_update": float(scheduler.get_last_lr()[0]),
                "sample_ids_sha256": training.hash_json(step_targets),
                "wall_seconds": time.time() - step_started,
            }
        )
        base.write_csv(output / "training_trace.csv", trace)
        print(
            f"TARC {arm_name} step {global_step + 1}/{steps} "
            f"risk={trace[-1]['risk_loss_mean']:.6f} "
            f"safety={trace[-1]['safety_selected_loss_mean']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_diagnostic = evaluate_diagnostics(model, diagnostic_pairs, config)
    final_state = model_common.trainable_state_hash(parameters)
    checkpoint = base.save_adapter(model, output, steps)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_tarc_matched_arm_manifest_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "runtime_head": runtime_head,
        "run_kind": protocol["run_kind"],
        "arm": arm_name,
        "objective": objective,
        "protocol_sha256": base.sha256(resolve_repo_path(protocol["protocol_path"])),
        "config_sha256": base.sha256(resolve_repo_path(protocol["config_path"])),
        "completed_steps": steps,
        "optimizer_steps": steps,
        "skipped_updates": skipped_updates,
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
        "available_ramp_margin_terms": available_ramp_terms,
        "available_tarc_margin_terms": available_tarc_terms,
        "active_ramp_margin_terms": active_ramp_terms,
        "active_tarc_margin_terms": active_tarc_terms,
        "first_step_risk_loss_mean": trace[0]["risk_loss_mean"],
        "first_step_safety_uniform_loss_mean": trace[0]["safety_uniform_loss_mean"],
        "initial_diagnostic": initial_diagnostic,
        "final_diagnostic": final_diagnostic,
        "tarc_loss_reduction": initial_diagnostic["tarc_margin_loss_mean"]
        - final_diagnostic["tarc_margin_loss_mean"],
        "trainable_parameter_inventory": inventory,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory
        >= float(config["minimum_vram_headroom_fraction"]),
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "wall_clock_seconds": time.time() - started,
    }
    base.write_json(output / "run_manifest.json", manifest)
    del optimizer, scheduler, scaler, model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def compare_manifests(
    manifests: dict[str, dict[str, Any]], protocol: dict[str, Any], tolerance: float
) -> dict[str, bool | float]:
    arms = list(protocol["arm_order"])
    rows = [manifests[arm] for arm in arms]

    def all_equal(field: str) -> bool:
        return len({row[field] for row in rows}) == 1

    def spread(field: str) -> float:
        values = [float(row[field]) for row in rows]
        return max(values) - min(values)

    def diagnostic_spread(field: str) -> float:
        values = [float(row["initial_diagnostic"][field]) for row in rows]
        return max(values) - min(values)

    tarc_reduction = float(manifests["tarc"]["tarc_loss_reduction"])
    uniform_reduction = float(manifests["uniform"]["tarc_loss_reduction"])
    ramp_reduction = float(manifests["ramp"]["tarc_loss_reduction"])
    gates = protocol["gates"]
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
        "available_ramp_terms_equal": all_equal("available_ramp_margin_terms"),
        "available_tarc_terms_equal": all_equal("available_tarc_margin_terms"),
        "first_step_risk_loss_equal": spread("first_step_risk_loss_mean") <= tolerance,
        "first_step_uniform_safety_loss_equal": spread(
            "first_step_safety_uniform_loss_mean"
        )
        <= tolerance,
        "initial_uniform_diagnostic_equal": diagnostic_spread("uniform_loss_mean")
        <= tolerance,
        "initial_ramp_diagnostic_equal": diagnostic_spread("ramp_margin_loss_mean")
        <= tolerance,
        "initial_tarc_diagnostic_equal": diagnostic_spread("tarc_margin_loss_mean")
        <= tolerance,
        "assignment_counts_correct": (
            manifests["uniform"]["active_ramp_margin_terms"] == 0
            and manifests["uniform"]["active_tarc_margin_terms"] == 0
            and manifests["ramp"]["active_ramp_margin_terms"]
            == manifests["ramp"]["available_ramp_margin_terms"]
            and manifests["ramp"]["active_tarc_margin_terms"] == 0
            and manifests["tarc"]["active_tarc_margin_terms"]
            == manifests["tarc"]["available_tarc_margin_terms"]
            and manifests["tarc"]["active_ramp_margin_terms"] == 0
        ),
        "all_registered_assignments_exercised": len(
            {row["final_adapter_sha256"] for row in rows}
        )
        == len(rows),
        "sealed_data_untouched": all(
            not row["core100_opened"]
            and not row["heldout_opened"]
            and not row["candidate_outputs_generated"]
            for row in rows
        ),
        "tarc_loss_reduction": tarc_reduction,
        "uniform_tarc_loss_reduction": uniform_reduction,
        "ramp_tarc_loss_reduction": ramp_reduction,
        "tarc_advantage_over_uniform": tarc_reduction - uniform_reduction,
        "tarc_advantage_over_ramp": tarc_reduction - ramp_reduction,
        "direction_realized": (
            tarc_reduction >= float(gates["tarc_loss_reduction_min"])
            and tarc_reduction - uniform_reduction
            >= float(gates["tarc_advantage_over_each_control_min"])
            and tarc_reduction - ramp_reduction
            >= float(gates["tarc_advantage_over_each_control_min"])
        ),
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
        "available_ramp_terms_equal",
        "available_tarc_terms_equal",
        "first_step_risk_loss_equal",
        "first_step_uniform_safety_loss_equal",
        "initial_uniform_diagnostic_equal",
        "initial_ramp_diagnostic_equal",
        "initial_tarc_diagnostic_equal",
        "sealed_data_untouched",
    )
    numerical = ("steps_exact", "no_skipped_updates", "finite", "vram")
    if not all(checks[name] for name in matching):
        return "STOP_TARC_SMOKE_MATCHING_FAILED"
    if not all(checks[name] for name in numerical):
        return "STOP_TARC_SMOKE_NUMERICAL_FAILED"
    if not (
        checks["assignment_counts_correct"]
        and checks["all_registered_assignments_exercised"]
    ):
        return "STOP_TARC_SMOKE_ASSIGNMENT_NOT_EXERCISED"
    if not checks["direction_realized"]:
        return "STOP_TARC_SMOKE_DIRECTION_NOT_REALIZED"
    return "PASS_TARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, diagnostic_ids = context(path)
    if not torch.cuda.is_available():
        raise TarcTrainingValidationError("CUDA is unavailable")
    result = {
        "status": "PASS_TARC_MATCHED_SMOKE_PREFLIGHT",
        "protocol_sha256": base.sha256(path),
        "arms": protocol["arm_order"],
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_authorized_optimizer_steps": int(protocol["optimizer_steps_per_arm"])
        * len(protocol["arm_order"]),
        "safety_pair_draws_per_arm": protocol["safety_pair_draws_per_arm"],
        "safety_target_forwards_per_arm": 2 * int(protocol["safety_pair_draws_per_arm"]),
        "source_rows": len(targets),
        "diagnostic_prompt_pairs": len(diagnostic_ids),
        "training_diagnostic_prompt_overlap": 0,
        "ramp_coefficient": config["ramp_coefficient"],
        "tarc_coefficient": config["tarc_coefficient"],
        "gpu": torch.cuda.get_device_name(0),
        "candidate_generation_authorized": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def run(path: Path) -> dict[str, Any]:
    protocol, config, targets, diagnostic_ids = context(path)
    runtime_head = require_clean_start(path, protocol)
    root = resolve_repo_path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    manifests = {
        arm: train_arm(
            str(arm), protocol, config, targets, diagnostic_ids, runtime_head
        )
        for arm in protocol["arm_order"]
    }
    checks = compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_smoke_route(checks)
    decision = {
        "name": "step3_tarc_matched_smoke_decision_v1",
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
                "tarc_loss_reduction": manifest["tarc_loss_reduction"],
                "initial_diagnostic": manifest["initial_diagnostic"],
                "final_diagnostic": manifest["final_diagnostic"],
                "vram_headroom_fraction": manifest["vram_headroom_fraction"],
            }
            for arm, manifest in manifests.items()
        },
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "next_authorized": (
            "freeze_and_push_separate_matched_development16_protocol"
            if route == "PASS_TARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"
            else "stop_TARC_training"
        ),
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    base.write_json(decision_path, decision)
    report = f"""# TARC Matched Two-Step Smoke Report

Route: `{route}`.

Uniform, RAMP16, and TARC each completed two matched optimizer steps. TARC
validation-certificate loss reductions were
`{checks['uniform_tarc_loss_reduction']:.6f}` (uniform),
`{checks['ramp_tarc_loss_reduction']:.6f}` (RAMP16), and
`{checks['tarc_loss_reduction']:.6f}` (TARC). TARC advantages were
`{checks['tarc_advantage_over_uniform']:.6f}` and
`{checks['tarc_advantage_over_ramp']:.6f}`.

The diagnostic prompt pairs were disjoint from all smoke training pairs. This
teacher-forced direction test does not establish generated behavior. No
candidate output, proxy-dev, Core100, or heldout data was opened. Total
optimizer steps: `{decision['total_optimizer_steps']}`;
`main_training_authorized=false`.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_bytes(report.encode("utf-8"))
    artifacts = []
    for arm in protocol["arm_order"]:
        arm_root = resolve_repo_path(protocol["arm_outputs"][arm])
        for name in ("training_trace.csv", "checkpoint_index.csv", "run_manifest.json"):
            artifact = arm_root / name
            artifacts.append(
                {
                    "path": base.relative(artifact),
                    "sha256": base.sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )
    for artifact in (decision_path, report_path):
        artifacts.append(
            {
                "path": base.relative(artifact),
                "sha256": base.sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        )
    reproduction = {
        "name": "step3_tarc_smoke_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": base.relative(path),
        "protocol_sha256": base.sha256(path),
        "artifacts": artifacts,
        "local_ignored_checkpoints": {
            arm: manifests[arm]["final_checkpoint"] for arm in protocol["arm_order"]
        },
        "final_adapter_sha256": {
            arm: manifests[arm]["final_adapter_sha256"] for arm in protocol["arm_order"]
        },
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_tarc_training.py run"
        ),
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    base.write_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"]), reproduction
    )
    print(json.dumps({"route": route, "checks": checks}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    reproduction = base.read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if reproduction.get("protocol_sha256") != base.sha256(path):
        raise TarcTrainingValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or base.sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise TarcTrainingValidationError(f"artifact mismatch: {artifact['path']}")
    manifests = {
        arm: base.read_json(
            resolve_repo_path(protocol["arm_outputs"][arm]) / "run_manifest.json"
        )
        for arm in protocol["arm_order"]
    }
    checks = compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_smoke_route(checks)
    decision = base.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if decision.get("route") != route or reproduction.get("route") != route:
        raise TarcTrainingValidationError("stored and recomputed routes differ")
    for arm, manifest in manifests.items():
        adapter = resolve_repo_path(manifest["final_checkpoint"]) / "adapter_model.safetensors"
        if not adapter.is_file() or base.sha256(adapter) != manifest["final_adapter_sha256"]:
            raise TarcTrainingValidationError(f"local adapter mismatch: {arm}")
    result = {
        "status": "PASS_TARC_SMOKE_RESULT_VERIFICATION",
        "route": route,
        "arms": len(manifests),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "development_training_authorized": route
        == "PASS_TARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL",
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        TarcTrainingValidationError,
        tarc_p0.TarcP0ValidationError,
        base.RampTrainingValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_TARC_MATCHED_STAGE_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
