"""Run hash-bound matched CITA uniform/information/sham training stages."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_common as cita
import step3_cita_p0 as p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cita/"
    "protocol/cita_smoke_protocol.json"
)


class CitaTrainingValidationError(ValueError):
    """Raised when a matched CITA training stage violates its protocol."""


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
        raise CitaTrainingValidationError(f"path escapes repository: {value}") from exc
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
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_matched_CITA_stage_training":
        raise CitaTrainingValidationError("training protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise CitaTrainingValidationError(f"frozen input hash mismatch: {name}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise CitaTrainingValidationError("entry commit is not an ancestor of HEAD")
    config = read_json(resolve_repo_path(protocol["config_path"]))
    run_kind = str(protocol["run_kind"])
    authorization_key = f"{run_kind}_training_authorized"
    if config.get(authorization_key) is not True or protocol.get("training_authorized") is not True:
        raise CitaTrainingValidationError(f"{run_kind} training is not authorized")
    if config.get("main_training_authorized") is not False:
        raise CitaTrainingValidationError("main training must remain unauthorized")
    if config.get("core100_opened") is not False or config.get("heldout_opened") is not False:
        raise CitaTrainingValidationError("sealed evaluation data is open")
    expected_steps = int(config[f"{run_kind}_steps"])
    if int(protocol["optimizer_steps_per_arm"]) != expected_steps:
        raise CitaTrainingValidationError("protocol step count differs from config")
    arms = list(protocol["arm_order"])
    if arms != ["uniform", "cita", "sham"] or set(config["arms"]) != set(arms):
        raise CitaTrainingValidationError("matched arm registry differs")
    p0_route = read_json(resolve_repo_path(config["p0_route"]))
    if p0_route.get("route") != "PASS_CITA_P0_AUTHORIZE_MATCHED_SMOKE":
        raise CitaTrainingValidationError("CITA-P0 did not authorize matched training")
    packet = read_jsonl(resolve_repo_path(config["mismatch_packet_path"]))
    cache = read_jsonl(resolve_repo_path(config["weight_cache_path"]))
    p0.validate_packet_rows(packet, 74, 37, 5)
    p0.validate_weight_cache(cache, packet, {})
    if [row["target_id"] for row in packet] != [row["target_id"] for row in cache]:
        raise CitaTrainingValidationError("packet/cache target order differs")
    return protocol, config, packet, cache


def require_clean_start(protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise CitaTrainingValidationError("matched stage requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([protocol["config_path"], relative(resolve_repo_path(DEFAULT_PROTOCOL))])
    for path in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    root = resolve_repo_path(protocol["output_root"])
    if root.exists():
        raise CitaTrainingValidationError(f"prospective output root already exists: {root}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, packet, cache = context(path)
    if not torch.cuda.is_available():
        raise CitaTrainingValidationError("CUDA is unavailable")
    semantic_budget_equal = all(
        sorted(row["semantic_cita_weights"]) == sorted(row["semantic_sham_weights"])
        for row in cache
    )
    if not semantic_budget_equal:
        raise CitaTrainingValidationError("CITA/sham cache budgets differ")
    result = {
        "status": "PASS_CITA_MATCHED_STAGE_PREFLIGHT",
        "run_kind": protocol["run_kind"],
        "arms": protocol["arm_order"],
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_authorized_optimizer_steps": int(protocol["optimizer_steps_per_arm"])
        * len(protocol["arm_order"]),
        "packet_rows": len(packet),
        "cache_rows": len(cache),
        "risk_rows": int(config["train_examples"]),
        "exact_budget_sham": semantic_budget_equal,
        "gpu": torch.cuda.get_device_name(0),
        "candidate_generation_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def safety_feature(
    tokenizer: Any,
    packet_row: dict[str, Any],
    cache_row: dict[str, Any],
    max_length: int,
    weight_mode: str,
) -> dict[str, Any]:
    serialized = p0.serialized_response(
        tokenizer,
        str(packet_row["correct_prompt"]),
        str(packet_row["response"]),
        max_length,
    )
    if serialized["response_token_ids"] != cache_row["response_token_ids"]:
        raise CitaTrainingValidationError("training tokenization differs from P0 cache")
    prefix = int(serialized["prefix_tokens"])
    labels = [-100] * prefix + list(serialized["response_token_ids"])
    weights = [0.0] * prefix + list(cache_row[f"{weight_mode}_weights"])
    uniform = [0.0] * prefix + list(cache_row["uniform_weights"])
    return {
        "input_ids": list(serialized["input_ids"]),
        "attention_mask": [1] * len(serialized["input_ids"]),
        "labels": labels,
        "weights": weights,
        "uniform_weights": uniform,
        "response_tokens": len(serialized["response_token_ids"]),
        "weight_budget": sum(weights),
    }


def risk_loss(model: Any, feature: dict[str, Any], device: Any, fp16: bool) -> Any:
    import torch

    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        return model(
            **training.batch_feature(feature, device), use_cache=False, return_dict=True
        ).loss


def safety_losses(
    model: Any, feature: dict[str, Any], device: Any, fp16: bool
) -> tuple[Any, Any]:
    import torch

    ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
    attention = torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([feature["labels"]], dtype=torch.long, device=device)
    weights = torch.tensor([feature["weights"]], dtype=torch.float32, device=device)
    uniform_weights = torch.tensor(
        [feature["uniform_weights"]], dtype=torch.float32, device=device
    )
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        logits = model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).logits
    weighted = cita.weighted_token_loss(logits, labels, weights)
    uniform = cita.weighted_token_loss(logits, labels, uniform_weights)
    return weighted, uniform.detach()


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
    training.write_csv(output / "checkpoint_index.csv", [row])
    return row


def train_arm(
    arm_name: str,
    protocol: dict[str, Any],
    config: dict[str, Any],
    packet: list[dict[str, Any]],
    cache: list[dict[str, Any]],
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
        raise CitaTrainingValidationError("unexpected trainable parameter inventory")
    initial_state_hash = p0.trainable_state_hash(parameters)

    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["train_examples"])
    )
    if len(risk_rows) != int(config["train_examples"]):
        raise CitaTrainingValidationError("risk training row count differs")
    risk_features = [
        training.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(config["max_length"])
        )
        for row in risk_rows
    ]
    mode = str(config["arms"][arm_name]["weight_mode"])
    safety_features = [
        safety_feature(
            tokenizer, source, cached, int(config["max_length"]), mode
        )
        for source, cached in zip(packet, cache, strict=True)
    ]
    if any(
        not math.isclose(feature["weight_budget"], feature["response_tokens"], abs_tol=1e-9)
        for feature in safety_features
    ):
        raise CitaTrainingValidationError("per-response safety weight budget drift")

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
    target_position = 0
    target_exposure: Counter[str] = Counter()
    total_safety_weight = 0.0
    total_safety_tokens = 0
    skipped_updates = 0
    trace = []
    optimizer.zero_grad(set_to_none=True)

    for global_step in range(steps):
        step_started = time.time()
        risk_losses = []
        weighted_losses = []
        uniform_losses = []
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
            target_id = str(packet[safety_index]["target_id"])
            target_exposure[target_id] += 1
            step_targets.append(target_id)
            feature = safety_features[safety_index]
            current_safety, uniform_diagnostic = safety_losses(
                model, feature, device, bool(config["fp16"])
            )
            scaler.scale(current_safety * safety_coefficient).backward()
            weighted_losses.append(float(current_safety.detach().float().cpu()))
            uniform_losses.append(float(uniform_diagnostic.float().cpu()))
            total_safety_weight += float(feature["weight_budget"])
            total_safety_tokens += int(feature["response_tokens"])

        scaler.unscale_(optimizer)
        trainable = [parameter for _, parameter in parameters]
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if not finite:
            raise FloatingPointError(f"non-finite or missing CITA gradient at step {global_step + 1}")
        preclip = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
        )
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError(f"GradScaler skipped CITA step {global_step + 1}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        trace.append(
            {
                "optimizer_step": global_step + 1,
                "arm": arm_name,
                "weight_mode": mode,
                "risk_loss_mean": sum(risk_losses) / len(risk_losses),
                "safety_weighted_loss_mean": sum(weighted_losses) / len(weighted_losses),
                "safety_uniform_diagnostic_loss_mean": sum(uniform_losses) / len(uniform_losses),
                "gradient_preclip_norm": preclip,
                "learning_rate_after_update": float(scheduler.get_last_lr()[0]),
                "target_ids_sha256": training.hash_json(step_targets),
                "wall_seconds": time.time() - step_started,
            }
        )
        training.write_csv(output / "training_trace.csv", trace)
        print(
            f"CITA {arm_name} step {global_step + 1}/{steps} "
            f"risk={trace[-1]['risk_loss_mean']:.6f} "
            f"safety={trace[-1]['safety_weighted_loss_mean']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_state_hash = p0.trainable_state_hash(parameters)
    checkpoint = save_adapter(model, output, steps)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_cita_matched_arm_manifest_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "runtime_head": runtime_head,
        "run_kind": protocol["run_kind"],
        "arm": arm_name,
        "weight_mode": mode,
        "protocol_sha256": sha256(resolve_repo_path(protocol["protocol_path"])),
        "config_sha256": sha256(resolve_repo_path(protocol["config_path"])),
        "weight_cache_sha256": sha256(resolve_repo_path(config["weight_cache_path"])),
        "completed_steps": steps,
        "optimizer_steps": steps,
        "skipped_updates": skipped_updates,
        "all_losses_and_gradients_finite": True,
        "initial_trainable_state_sha256": initial_state_hash,
        "final_trainable_state_sha256": final_state_hash,
        "final_adapter_sha256": checkpoint["adapter_sha256"],
        "final_checkpoint": checkpoint["checkpoint"],
        "risk_order_sha256": training.hash_json(risk_order),
        "target_order_sha256": training.hash_json(
            [str(row["target_id"]) for row in packet]
        ),
        "target_exposure_sha256": training.hash_json(dict(sorted(target_exposure.items()))),
        "safety_draws": target_position,
        "safety_scored_tokens": total_safety_tokens,
        "safety_weight_budget": total_safety_weight,
        "first_step_risk_loss_mean": trace[0]["risk_loss_mean"],
        "first_step_safety_weighted_loss_mean": trace[0]["safety_weighted_loss_mean"],
        "first_step_safety_uniform_diagnostic_loss_mean": trace[0][
            "safety_uniform_diagnostic_loss_mean"
        ],
        "trainable_parameter_inventory": inventory,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory
        >= float(config["minimum_vram_headroom_fraction"]),
        "thinking_disabled_for_safety": True,
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "wall_clock_seconds": time.time() - started,
    }
    training.write_json(output / "run_manifest.json", manifest)
    del optimizer, scheduler, scaler, model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def compare_manifests(
    manifests: dict[str, dict[str, Any]], protocol: dict[str, Any], tolerance: float
) -> dict[str, bool]:
    arms = list(protocol["arm_order"])
    rows = [manifests[arm] for arm in arms]

    def all_equal(field: str) -> bool:
        return len({row[field] for row in rows}) == 1

    def spread(field: str) -> float:
        values = [float(row[field]) for row in rows]
        return max(values) - min(values)

    expected_steps = int(protocol["optimizer_steps_per_arm"])
    return {
        "all_arms_present": set(manifests) == set(arms),
        "steps_exact": all(row["completed_steps"] == expected_steps for row in rows),
        "no_skipped_updates": all(row["skipped_updates"] == 0 for row in rows),
        "finite": all(row["all_losses_and_gradients_finite"] for row in rows),
        "vram": all(row["vram_gate_pass"] for row in rows),
        "initial_state_equal": all_equal("initial_trainable_state_sha256"),
        "risk_order_equal": all_equal("risk_order_sha256"),
        "target_order_equal": all_equal("target_order_sha256"),
        "target_exposure_equal": all_equal("target_exposure_sha256"),
        "safety_draws_equal": all_equal("safety_draws"),
        "safety_tokens_equal": all_equal("safety_scored_tokens"),
        "safety_weight_budget_equal": spread("safety_weight_budget") <= tolerance,
        "first_step_risk_loss_equal": spread("first_step_risk_loss_mean") <= tolerance,
        "first_step_uniform_safety_loss_equal": spread(
            "first_step_safety_uniform_diagnostic_loss_mean"
        )
        <= tolerance,
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
    }


def select_smoke_route(checks: dict[str, bool]) -> str:
    matching = (
        "all_arms_present",
        "initial_state_equal",
        "risk_order_equal",
        "target_order_equal",
        "target_exposure_equal",
        "safety_draws_equal",
        "safety_tokens_equal",
        "safety_weight_budget_equal",
        "first_step_risk_loss_equal",
        "first_step_uniform_safety_loss_equal",
        "sealed_data_untouched",
    )
    numerical = ("steps_exact", "no_skipped_updates", "finite", "vram")
    if not all(checks[name] for name in matching):
        return "STOP_CITA_SMOKE_MATCHING_FAILED"
    if not all(checks[name] for name in numerical):
        return "STOP_CITA_SMOKE_NUMERICAL_FAILED"
    if not checks["all_registered_assignments_exercised"]:
        return "STOP_CITA_SMOKE_ASSIGNMENT_NOT_EXERCISED"
    return "PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"


def run(path: Path) -> dict[str, Any]:
    protocol, config, packet, cache = context(path)
    runtime_head = require_clean_start(protocol)
    root = resolve_repo_path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for arm in protocol["arm_order"]:
        manifests[arm] = train_arm(
            str(arm), protocol, config, packet, cache, runtime_head
        )
    checks = compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_smoke_route(checks)
    decision = {
        "name": "step3_cita_matched_smoke_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": sha256(path),
        "checks": checks,
        "arms": {
            arm: {
                "completed_steps": manifest["completed_steps"],
                "optimizer_steps": manifest["optimizer_steps"],
                "final_adapter_sha256": manifest["final_adapter_sha256"],
                "first_step_risk_loss_mean": manifest["first_step_risk_loss_mean"],
                "first_step_safety_weighted_loss_mean": manifest[
                    "first_step_safety_weighted_loss_mean"
                ],
                "first_step_safety_uniform_diagnostic_loss_mean": manifest[
                    "first_step_safety_uniform_diagnostic_loss_mean"
                ],
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
            if route == "PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"
            else "stop_CITA_training"
        ),
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    training.write_json(decision_path, decision)
    report = f"""# CITA Matched Three-Arm Smoke Report

Route: `{route}`.

All three arms were launched by one hash-bound orchestrator from fresh seed-1 LoRA states. Each arm executed `{protocol['optimizer_steps_per_arm']}` optimizer steps with the same risk order, safety target order, target exposure, response-token count, and total safety weight budget. First-step risk and uniform diagnostic safety losses matched within `{protocol['gates']['matching_tolerance']}`.

The final adapter hashes are distinct across uniform, CITA, and sham: `{checks['all_registered_assignments_exercised']}`. This confirms execution of the three assignments, not behavioral benefit. No generation or evaluator was run, and core100/heldout remained sealed. Total optimizer steps across arms: `{decision['total_optimizer_steps']}`; `main_training_authorized=false`.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(report, encoding="utf-8")
    artifacts = []
    for arm in protocol["arm_order"]:
        arm_root = resolve_repo_path(protocol["arm_outputs"][arm])
        for name in ("training_trace.csv", "checkpoint_index.csv", "run_manifest.json"):
            artifact = arm_root / name
            artifacts.append(
                {"path": relative(artifact), "sha256": sha256(artifact), "bytes": artifact.stat().st_size}
            )
    for artifact in (decision_path, report_path):
        artifacts.append(
            {"path": relative(artifact), "sha256": sha256(artifact), "bytes": artifact.stat().st_size}
        )
    reproduction = {
        "name": "step3_cita_matched_smoke_reproduction_manifest_v1",
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
            "scripts/step3_cita_training.py run"
        ),
        "optimizer_steps_per_arm": protocol["optimizer_steps_per_arm"],
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    training.write_json(
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
        raise CitaTrainingValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise CitaTrainingValidationError(f"artifact hash mismatch: {artifact['path']}")
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
        raise CitaTrainingValidationError("stored and recomputed routes differ")
    for arm, manifest in manifests.items():
        adapter = resolve_repo_path(manifest["final_checkpoint"]) / "adapter_model.safetensors"
        if not adapter.is_file() or sha256(adapter) != manifest["final_adapter_sha256"]:
            raise CitaTrainingValidationError(f"local final adapter mismatch: {arm}")
    result = {
        "status": "PASS_CITA_MATCHED_STAGE_RESULT_VERIFICATION",
        "route": route,
        "arms": len(manifests),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
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
        CitaTrainingValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_CITA_MATCHED_STAGE_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
