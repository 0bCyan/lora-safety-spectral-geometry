"""Run hash-bound matched uniform/RAMP/suffix training stages."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_ramp_p0 as ramp_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_ramp/"
    "protocol/ramp_smoke_protocol.json"
)


class RampTrainingValidationError(ValueError):
    """Raised when matched RAMP training violates its frozen protocol."""


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
        raise RampTrainingValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_matched_RAMP_stage_training":
        raise RampTrainingValidationError("training protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RampTrainingValidationError(f"frozen input hash mismatch: {name}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RampTrainingValidationError("entry commit is not an ancestor of HEAD")
    config = read_json(resolve_repo_path(protocol["config_path"]))
    run_kind = str(protocol["run_kind"])
    if (
        config.get(f"{run_kind}_training_authorized") is not True
        or protocol.get("training_authorized") is not True
    ):
        raise RampTrainingValidationError(f"{run_kind} training is not authorized")
    if (
        config.get("main_training_authorized") is not False
        or config.get("candidate_generation_authorized") is not False
        or config.get("core100_opened") is not False
        or config.get("heldout_opened") is not False
    ):
        raise RampTrainingValidationError("config violates the sealed boundary")
    if int(protocol["optimizer_steps_per_arm"]) != int(config[f"{run_kind}_steps"]):
        raise RampTrainingValidationError("protocol step count differs from config")
    arms = list(protocol["arm_order"])
    if arms != ["uniform", "suffix", "ramp"] or set(config["arms"]) != set(arms):
        raise RampTrainingValidationError("matched arm registry differs")

    p0_route_path = resolve_repo_path(config["p0_route"])
    if sha256(p0_route_path) != config["p0_route_sha256"]:
        raise RampTrainingValidationError("P0 route hash differs from config")
    p0_route = read_json(p0_route_path)
    if p0_route.get("route") != "PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL":
        raise RampTrainingValidationError("RAMP-P0 did not authorize matched smoke")
    for key in ("prefix_coefficient", "suffix_coefficient"):
        if not math.isclose(
            float(config[key]), float(p0_route["coefficients"][key]),
            rel_tol=0.0, abs_tol=1e-15,
        ):
            raise RampTrainingValidationError(f"frozen P0 coefficient drift: {key}")

    targets_path = resolve_repo_path(config["safety_targets_path"])
    if sha256(targets_path) != config["safety_targets_sha256"]:
        raise RampTrainingValidationError("safety target hash differs from config")
    targets = read_jsonl(targets_path)
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    by_id = {str(row["target_id"]): row for row in targets}
    diagnostic_ids = list(config["diagnostic_target_ids"])
    if len(diagnostic_ids) != 16 or len(set(diagnostic_ids)) != 16:
        raise RampTrainingValidationError("diagnostic target IDs are invalid")
    try:
        diagnostics = [by_id[target_id] for target_id in diagnostic_ids]
    except KeyError as exc:
        raise RampTrainingValidationError("diagnostic target is absent") from exc
    p0_protocol = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p0_protocol"]["path"])
    )
    if diagnostic_ids != p0_protocol["selection"]["validation_target_ids"]:
        raise RampTrainingValidationError("diagnostic IDs differ from RAMP-P0 validation")
    draws = int(protocol["optimizer_steps_per_arm"]) * int(
        config["gradient_accumulation_steps"]
    )
    train_prompt_ids = {str(row["sample_id"]) for row in targets[:draws]}
    diagnostic_prompt_ids = {str(row["sample_id"]) for row in diagnostics}
    if train_prompt_ids.intersection(diagnostic_prompt_ids):
        raise RampTrainingValidationError("smoke draws overlap diagnostic prompts")
    return protocol, config, targets, diagnostics


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise RampTrainingValidationError("matched stage requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([protocol["config_path"], relative(path)])
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
        raise RampTrainingValidationError(f"prospective output root exists: {root}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, diagnostics = context(path)
    if not torch.cuda.is_available():
        raise RampTrainingValidationError("CUDA is unavailable")
    draws = int(protocol["optimizer_steps_per_arm"]) * int(
        config["gradient_accumulation_steps"]
    )
    result = {
        "status": "PASS_RAMP_MATCHED_STAGE_PREFLIGHT",
        "run_kind": protocol["run_kind"],
        "arms": protocol["arm_order"],
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_authorized_optimizer_steps": int(protocol["optimizer_steps_per_arm"])
        * len(protocol["arm_order"]),
        "safety_target_rows": len(targets),
        "smoke_safety_draws_per_arm": draws,
        "diagnostic_rows": len(diagnostics),
        "training_diagnostic_prompt_overlap": 0,
        "margin_width": config["margin_width"],
        "prefix_coefficient": config["prefix_coefficient"],
        "suffix_coefficient": config["suffix_coefficient"],
        "risk_rows": config["train_examples"],
        "gpu": torch.cuda.get_device_name(0),
        "candidate_generation_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def safety_feature(
    tokenizer: Any, row: dict[str, Any], max_length: int, width: int
) -> dict[str, Any]:
    serialized = model_common.serialized_response(
        tokenizer, str(row["prompt"]), str(row["response"]), max_length
    )
    semantic_positions = [
        index for index, keep in enumerate(serialized["semantic_mask"]) if keep
    ]
    if semantic_positions != list(range(len(semantic_positions))):
        raise RampTrainingValidationError("semantic response tokens are not a prefix")
    if len(semantic_positions) < width:
        raise RampTrainingValidationError("safety target is shorter than margin width")
    return {
        "sample_id": str(row["sample_id"]),
        "target_id": str(row["target_id"]),
        "input_ids": list(serialized["input_ids"]),
        "attention_mask": [1] * len(serialized["input_ids"]),
        "prefix_tokens": int(serialized["prefix_tokens"]),
        "semantic_positions": semantic_positions,
        "response_tokens": len(serialized["response_token_ids"]),
    }


def risk_loss(model: Any, feature: dict[str, Any], device: Any, fp16: bool) -> Any:
    import torch

    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        return model(
            **training.batch_feature(feature, device), use_cache=False, return_dict=True
        ).loss


def compose_safety_loss(
    uniform: Any,
    prefix: Any,
    suffix: Any,
    objective: str,
    prefix_coefficient: float,
    suffix_coefficient: float,
) -> Any:
    if objective == "uniform":
        return uniform
    if objective == "prefix":
        return uniform + prefix_coefficient * prefix
    if objective == "suffix":
        return uniform + suffix_coefficient * suffix
    raise ValueError(f"unknown safety objective: {objective}")


def safety_losses(
    model: Any,
    feature: dict[str, Any],
    device: Any,
    fp16: bool,
    objective: str,
    width: int,
    prefix_coefficient: float,
    suffix_coefficient: float,
) -> dict[str, Any]:
    import torch

    ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
    attention = torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device)
    prefix_tokens = int(feature["prefix_tokens"])
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        logits = model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).logits
    response_logits = logits[0, prefix_tokens - 1 : ids.shape[1] - 1]
    response_targets = ids[0, prefix_tokens:]
    uniform, _ = ramp_p0.response_objectives(
        response_logits, response_targets, feature["semantic_positions"], width, "uniform"
    )
    prefix_loss, prefix_metrics = ramp_p0.response_objectives(
        response_logits, response_targets, feature["semantic_positions"], width, "prefix"
    )
    suffix_loss, suffix_metrics = ramp_p0.response_objectives(
        response_logits, response_targets, feature["semantic_positions"], width, "suffix"
    )
    selected = compose_safety_loss(
        uniform,
        prefix_loss,
        suffix_loss,
        objective,
        prefix_coefficient,
        suffix_coefficient,
    )
    return {
        "selected": selected,
        "uniform": uniform,
        "prefix": prefix_loss,
        "suffix": suffix_loss,
        "prefix_metrics": prefix_metrics,
        "suffix_metrics": suffix_metrics,
    }


def evaluate_diagnostics(
    model: Any,
    features: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    device = training.model_device(model)
    previous = model.training
    model.eval()
    uniform_losses = []
    prefix_losses = []
    suffix_losses = []
    prefix_token_rates = []
    prefix_full = 0
    with torch.no_grad():
        for feature in features:
            losses = safety_losses(
                model,
                feature,
                device,
                bool(config["fp16"]),
                "uniform",
                int(config["margin_width"]),
                float(config["prefix_coefficient"]),
                float(config["suffix_coefficient"]),
            )
            uniform_losses.append(float(losses["uniform"].float().cpu()))
            prefix_losses.append(float(losses["prefix"].float().cpu()))
            suffix_losses.append(float(losses["suffix"].float().cpu()))
            prefix_token_rates.append(
                float(losses["prefix_metrics"]["certified_token_rate"])
            )
            prefix_full += int(losses["prefix_metrics"]["full_region_certified"])
    model.train(previous)
    return {
        "rows": len(features),
        "uniform_loss_mean": sum(uniform_losses) / len(uniform_losses),
        "prefix_margin_loss_mean": sum(prefix_losses) / len(prefix_losses),
        "suffix_margin_loss_mean": sum(suffix_losses) / len(suffix_losses),
        "prefix_certified_token_rate": sum(prefix_token_rates) / len(prefix_token_rates),
        "prefix_full_certificate_rate": prefix_full / len(features),
    }


def save_adapter(model: Any, output: Path, step: int) -> dict[str, Any]:
    checkpoint = output / "checkpoints" / f"step{step:04d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint)
    adapter = checkpoint / "adapter_model.safetensors"
    row = {
        "step": step,
        "checkpoint": relative(checkpoint),
        "adapter_sha256": sha256(adapter),
        "adapter_bytes": adapter.stat().st_size,
    }
    write_csv(output / "checkpoint_index.csv", [row])
    return row


def train_arm(
    arm_name: str,
    protocol: dict[str, Any],
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
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
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RampTrainingValidationError("unexpected trainable parameter inventory")
    initial_state = model_common.trainable_state_hash(parameters)

    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["train_examples"])
    )
    if len(risk_rows) != int(config["train_examples"]):
        raise RampTrainingValidationError("risk row count differs")
    risk_features = [
        training.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(config["max_length"])
        )
        for row in risk_rows
    ]
    safety_features = [
        safety_feature(
            tokenizer, row, int(config["max_length"]), int(config["margin_width"])
        )
        for row in targets
    ]
    by_target = {feature["target_id"]: feature for feature in safety_features}
    diagnostic_features = [by_target[str(row["target_id"])] for row in diagnostics]
    initial_diagnostic = evaluate_diagnostics(model, diagnostic_features, config)

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
    total_available_prefix_terms = 0
    total_available_suffix_terms = 0
    active_prefix_terms = 0
    active_suffix_terms = 0
    skipped_updates = 0
    trace = []
    optimizer.zero_grad(set_to_none=True)

    for global_step in range(steps):
        step_started = time.time()
        risk_losses = []
        selected_losses = []
        uniform_losses = []
        prefix_losses = []
        suffix_losses = []
        step_targets = []
        for micro_index in range(microsteps):
            position = global_step * microsteps + micro_index
            model.train()
            current_risk = risk_loss(
                model, risk_features[risk_order[position]], device, bool(config["fp16"])
            )
            scaler.scale(current_risk * risk_coefficient).backward()
            risk_losses.append(float(current_risk.detach().float().cpu()))

            safety_index = target_position % len(safety_features)
            target_position += 1
            feature = safety_features[safety_index]
            target_id = str(feature["target_id"])
            target_exposure[target_id] += 1
            step_targets.append(target_id)
            losses = safety_losses(
                model,
                feature,
                device,
                bool(config["fp16"]),
                objective,
                int(config["margin_width"]),
                float(config["prefix_coefficient"]),
                float(config["suffix_coefficient"]),
            )
            scaler.scale(losses["selected"] * safety_coefficient).backward()
            selected_losses.append(float(losses["selected"].detach().float().cpu()))
            uniform_losses.append(float(losses["uniform"].detach().float().cpu()))
            prefix_losses.append(float(losses["prefix"].detach().float().cpu()))
            suffix_losses.append(float(losses["suffix"].detach().float().cpu()))
            total_response_tokens += int(feature["response_tokens"])
            total_available_prefix_terms += int(config["margin_width"])
            total_available_suffix_terms += int(config["margin_width"])
            active_prefix_terms += int(objective == "prefix") * int(config["margin_width"])
            active_suffix_terms += int(objective == "suffix") * int(config["margin_width"])

        scaler.unscale_(optimizer)
        trainable = [parameter for _, parameter in parameters]
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if not finite:
            raise FloatingPointError(
                f"non-finite or missing RAMP gradient at step {global_step + 1}"
            )
        preclip = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
        )
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError(f"GradScaler skipped RAMP step {global_step + 1}")
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
                "prefix_margin_loss_mean": sum(prefix_losses) / len(prefix_losses),
                "suffix_margin_loss_mean": sum(suffix_losses) / len(suffix_losses),
                "gradient_preclip_norm": preclip,
                "learning_rate_after_update": float(scheduler.get_last_lr()[0]),
                "target_ids_sha256": training.hash_json(step_targets),
                "wall_seconds": time.time() - step_started,
            }
        )
        write_csv(output / "training_trace.csv", trace)
        print(
            f"RAMP {arm_name} step {global_step + 1}/{steps} "
            f"risk={trace[-1]['risk_loss_mean']:.6f} "
            f"safety={trace[-1]['safety_selected_loss_mean']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_diagnostic = evaluate_diagnostics(model, diagnostic_features, config)
    final_state = model_common.trainable_state_hash(parameters)
    checkpoint = save_adapter(model, output, steps)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_ramp_matched_arm_manifest_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "runtime_head": runtime_head,
        "run_kind": protocol["run_kind"],
        "arm": arm_name,
        "objective": objective,
        "protocol_sha256": sha256(resolve_repo_path(protocol["protocol_path"])),
        "config_sha256": sha256(resolve_repo_path(protocol["config_path"])),
        "completed_steps": steps,
        "optimizer_steps": steps,
        "skipped_updates": skipped_updates,
        "all_losses_and_gradients_finite": True,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "final_adapter_sha256": checkpoint["adapter_sha256"],
        "final_checkpoint": checkpoint["checkpoint"],
        "risk_order_sha256": training.hash_json(risk_order),
        "target_order_sha256": training.hash_json(
            [str(row["target_id"]) for row in targets]
        ),
        "target_exposure_sha256": training.hash_json(dict(sorted(target_exposure.items()))),
        "safety_draws": target_position,
        "safety_response_tokens": total_response_tokens,
        "available_prefix_margin_terms": total_available_prefix_terms,
        "available_suffix_margin_terms": total_available_suffix_terms,
        "active_prefix_margin_terms": active_prefix_terms,
        "active_suffix_margin_terms": active_suffix_terms,
        "first_step_risk_loss_mean": trace[0]["risk_loss_mean"],
        "first_step_safety_uniform_loss_mean": trace[0]["safety_uniform_loss_mean"],
        "initial_diagnostic": initial_diagnostic,
        "final_diagnostic": final_diagnostic,
        "prefix_loss_reduction": initial_diagnostic["prefix_margin_loss_mean"]
        - final_diagnostic["prefix_margin_loss_mean"],
        "trainable_parameter_inventory": inventory,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory
        >= float(config["minimum_vram_headroom_fraction"]),
        "thinking_disabled_for_safety": True,
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "wall_clock_seconds": time.time() - started,
    }
    write_json(output / "run_manifest.json", manifest)
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

    ramp_reduction = float(manifests["ramp"]["prefix_loss_reduction"])
    uniform_reduction = float(manifests["uniform"]["prefix_loss_reduction"])
    suffix_reduction = float(manifests["suffix"]["prefix_loss_reduction"])
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
        "safety_draws_equal": all_equal("safety_draws"),
        "safety_response_tokens_equal": all_equal("safety_response_tokens"),
        "available_prefix_terms_equal": all_equal("available_prefix_margin_terms"),
        "available_suffix_terms_equal": all_equal("available_suffix_margin_terms"),
        "first_step_risk_loss_equal": spread("first_step_risk_loss_mean") <= tolerance,
        "first_step_uniform_safety_loss_equal": spread(
            "first_step_safety_uniform_loss_mean"
        )
        <= tolerance,
        "initial_uniform_diagnostic_equal": diagnostic_spread("uniform_loss_mean")
        <= tolerance,
        "initial_prefix_diagnostic_equal": diagnostic_spread("prefix_margin_loss_mean")
        <= tolerance,
        "initial_suffix_diagnostic_equal": diagnostic_spread("suffix_margin_loss_mean")
        <= tolerance,
        "assignment_counts_correct": (
            manifests["uniform"]["active_prefix_margin_terms"] == 0
            and manifests["uniform"]["active_suffix_margin_terms"] == 0
            and manifests["ramp"]["active_prefix_margin_terms"]
            == manifests["ramp"]["available_prefix_margin_terms"]
            and manifests["ramp"]["active_suffix_margin_terms"] == 0
            and manifests["suffix"]["active_suffix_margin_terms"]
            == manifests["suffix"]["available_suffix_margin_terms"]
            and manifests["suffix"]["active_prefix_margin_terms"] == 0
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
        "ramp_prefix_loss_reduction": ramp_reduction,
        "uniform_prefix_loss_reduction": uniform_reduction,
        "suffix_prefix_loss_reduction": suffix_reduction,
        "ramp_advantage_over_uniform": ramp_reduction - uniform_reduction,
        "ramp_advantage_over_suffix": ramp_reduction - suffix_reduction,
        "direction_realized": (
            ramp_reduction >= float(gates["ramp_prefix_loss_reduction_min"])
            and ramp_reduction - uniform_reduction
            >= float(gates["ramp_advantage_over_each_control_min"])
            and ramp_reduction - suffix_reduction
            >= float(gates["ramp_advantage_over_each_control_min"])
        ),
    }


def select_smoke_route(checks: dict[str, Any]) -> str:
    matching = (
        "all_arms_present",
        "initial_state_equal",
        "risk_order_equal",
        "target_order_equal",
        "target_exposure_equal",
        "safety_draws_equal",
        "safety_response_tokens_equal",
        "available_prefix_terms_equal",
        "available_suffix_terms_equal",
        "first_step_risk_loss_equal",
        "first_step_uniform_safety_loss_equal",
        "initial_uniform_diagnostic_equal",
        "initial_prefix_diagnostic_equal",
        "initial_suffix_diagnostic_equal",
        "sealed_data_untouched",
    )
    numerical = ("steps_exact", "no_skipped_updates", "finite", "vram")
    if not all(checks[name] for name in matching):
        return "STOP_RAMP_SMOKE_MATCHING_FAILED"
    if not all(checks[name] for name in numerical):
        return "STOP_RAMP_SMOKE_NUMERICAL_FAILED"
    if not (
        checks["assignment_counts_correct"]
        and checks["all_registered_assignments_exercised"]
    ):
        return "STOP_RAMP_SMOKE_ASSIGNMENT_NOT_EXERCISED"
    if not checks["direction_realized"]:
        return "STOP_RAMP_SMOKE_DIRECTION_NOT_REALIZED"
    return "PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"


def run(path: Path) -> dict[str, Any]:
    protocol, config, targets, diagnostics = context(path)
    runtime_head = require_clean_start(path, protocol)
    root = resolve_repo_path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for arm in protocol["arm_order"]:
        manifests[arm] = train_arm(
            str(arm), protocol, config, targets, diagnostics, runtime_head
        )
    checks = compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_smoke_route(checks)
    decision = {
        "name": "step3_ramp_matched_smoke_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": sha256(path),
        "checks": checks,
        "arms": {
            arm: {
                "objective": manifest["objective"],
                "completed_steps": manifest["completed_steps"],
                "optimizer_steps": manifest["optimizer_steps"],
                "final_adapter_sha256": manifest["final_adapter_sha256"],
                "prefix_loss_reduction": manifest["prefix_loss_reduction"],
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
            if route == "PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"
            else "stop_RAMP_training"
        ),
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    write_json(decision_path, decision)
    report = f"""# RAMP Matched Three-Arm Smoke Report

Route: `{route}`.

Uniform, suffix placebo, and RAMP each completed
`{protocol['optimizer_steps_per_arm']}` matched optimizer steps. All registered
input, exposure, first-step, numerical, assignment, and sealed-data checks
passed before the directional gate: `{all(value for key, value in checks.items() if isinstance(value, bool) and key != 'direction_realized')}`.

Prompt-disjoint prefix-loss reductions were
`{checks['uniform_prefix_loss_reduction']:.6f}` (uniform),
`{checks['suffix_prefix_loss_reduction']:.6f}` (suffix), and
`{checks['ramp_prefix_loss_reduction']:.6f}` (RAMP). RAMP advantages were
`{checks['ramp_advantage_over_uniform']:.6f}` and
`{checks['ramp_advantage_over_suffix']:.6f}`.

This smoke tests optimizer realization of the P0 direction, not generated
behavior. No candidate output, Core100, heldout, or reviewer data was opened.
Total optimizer steps across arms: `{decision['total_optimizer_steps']}`;
`main_training_authorized=false`.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report.encode("utf-8"))
    artifacts = []
    for arm in protocol["arm_order"]:
        arm_root = resolve_repo_path(protocol["arm_outputs"][arm])
        for name in ("training_trace.csv", "checkpoint_index.csv", "run_manifest.json"):
            artifact = arm_root / name
            artifacts.append(
                {
                    "path": relative(artifact),
                    "sha256": sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )
    for artifact in (decision_path, report_path):
        artifacts.append(
            {
                "path": relative(artifact),
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        )
    reproduction = {
        "name": "step3_ramp_matched_smoke_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": sha256(path),
        "artifacts": artifacts,
        "local_ignored_checkpoints": {
            arm: manifests[arm]["final_checkpoint"] for arm in protocol["arm_order"]
        },
        "final_adapter_sha256": {
            arm: manifests[arm]["final_adapter_sha256"] for arm in protocol["arm_order"]
        },
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_ramp_training.py run"
        ),
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    write_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"]), reproduction
    )
    print(json.dumps({"route": route, "checks": checks}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    reproduction = read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if reproduction.get("protocol_sha256") != sha256(path):
        raise RampTrainingValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise RampTrainingValidationError(f"artifact mismatch: {artifact['path']}")
    manifests = {
        arm: read_json(resolve_repo_path(protocol["arm_outputs"][arm]) / "run_manifest.json")
        for arm in protocol["arm_order"]
    }
    checks = compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_smoke_route(checks)
    decision = read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if decision.get("route") != route or reproduction.get("route") != route:
        raise RampTrainingValidationError("stored and recomputed routes differ")
    for arm, manifest in manifests.items():
        adapter = resolve_repo_path(manifest["final_checkpoint"]) / "adapter_model.safetensors"
        if not adapter.is_file() or sha256(adapter) != manifest["final_adapter_sha256"]:
            raise RampTrainingValidationError(f"local adapter mismatch: {arm}")
    result = {
        "status": "PASS_RAMP_MATCHED_STAGE_RESULT_VERIFICATION",
        "route": route,
        "arms": len(manifests),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
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
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        RampTrainingValidationError,
        ramp_p0.RampP0ValidationError,
        model_common.CitaP0ValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_RAMP_MATCHED_STAGE_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
