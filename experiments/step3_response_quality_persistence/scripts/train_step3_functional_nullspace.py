"""Train exploratory LoRA arms constrained to a safety activation nullspace."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import step3_functional_nullspace_common as fn
import step3ct_training_common as common
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-kind", choices=["smoke", "main"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", default=None)
    return parser.parse_args()


def lora_a_registry(model: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    representatives: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        key = fn.functional_key(module_name)
        entries[module_name] = {
            "module": module,
            "weight": module.lora_A["default"].weight,
            "functional_key": key,
        }
        if fn.is_capture_representative(module_name):
            if key in representatives:
                raise ValueError(f"duplicate capture representative for {key}")
            representatives[key] = module
    if not entries or not representatives:
        raise RuntimeError("no LoRA-A modules found")
    missing = sorted({entry["functional_key"] for entry in entries.values()} - set(representatives))
    if missing:
        raise ValueError(f"functional keys lack capture representatives: {missing[:5]}")
    return entries, representatives


def calibration_rows(targets: list[dict[str, Any]], cluster_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in targets:
        prompt_id = str(row["sample_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        selected.append(row)
        if len(selected) == cluster_count:
            break
    if len(selected) != cluster_count:
        raise ValueError(f"expected {cluster_count} calibration prompt clusters")
    return selected


def build_functional_bases(
    model: Any,
    tokenizer: Any,
    representatives: dict[str, Any],
    rows: list[dict[str, Any]],
    max_length: int,
    tokens_per_response: int,
    rank: int,
    epsilon: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    device = common.model_device(model)
    captures: dict[str, list[Any]] = {key: [] for key in representatives}
    state: dict[str, Any] = {"positions": None}
    handles = []

    def capture(key: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            hidden = inputs[0].detach()
            positions = state["positions"]
            if positions is None:
                raise RuntimeError("capture positions were not set")
            selected = hidden[0].index_select(0, positions.to(hidden.device))
            captures[key].append(selected.float().cpu())

        return hook

    for key, module in representatives.items():
        handles.append(module.register_forward_pre_hook(capture(key)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                feature = common.response_feature(
                    tokenizer, str(row["prompt"]), str(row["response"]), max_length
                )
                labels = torch.tensor(feature["labels"], dtype=torch.long, device=device)
                state["positions"] = fn.sampled_mask_positions(labels != -100, tokens_per_response)
                batch = common.batch_feature(feature, device, include_labels=False)
                model(**batch, use_cache=False, return_dict=True)
    finally:
        state["positions"] = None
        for handle in handles:
            handle.remove()
        model.train(was_training)

    bases: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    for key in sorted(captures):
        matrix = torch.cat(captures[key], dim=0).to(device=device, dtype=torch.float32)
        basis, metrics = fn.right_singular_basis(matrix, rank=rank, epsilon=epsilon)
        bases[key] = basis
        reports.append(
            {
                "functional_key": key,
                "basis_sha256": fn.tensor_sha256(basis),
                **metrics,
            }
        )
    return bases, reports


def aggregate_component_ratio(
    entries: dict[str, dict[str, Any]], bases: dict[str, Any], source: str
) -> float:
    import torch

    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for entry in entries.values():
        weight = entry["weight"]
        value = weight.grad if source == "gradient" else weight.data
        if value is None:
            continue
        basis = bases[entry["functional_key"]].to(device=value.device, dtype=value.dtype)
        coefficients = value.detach() @ basis.T
        numerator += coefficients.detach().double().square().sum().cpu()
        denominator += value.detach().double().square().sum().cpu()
    if not float(denominator):
        return 0.0
    return math.sqrt(float(numerator / denominator))


def project_entries_(
    entries: dict[str, dict[str, Any]], bases: dict[str, Any], source: str
) -> None:
    import torch

    with torch.no_grad():
        for entry in entries.values():
            weight = entry["weight"]
            value = weight.grad if source == "gradient" else weight.data
            if value is None:
                continue
            fn.project_rows_(value, bases[entry["functional_key"]])


def capture_entry_gradients(entries: dict[str, dict[str, Any]]) -> dict[str, Any | None]:
    return {
        name: None if entry["weight"].grad is None else entry["weight"].grad.detach().clone()
        for name, entry in entries.items()
    }


def project_new_gradient_delta_(
    entries: dict[str, dict[str, Any]], bases: dict[str, Any], baselines: dict[str, Any | None]
) -> tuple[float, float]:
    import torch

    before_numerator = torch.zeros((), dtype=torch.float64)
    after_numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for name, entry in entries.items():
            gradient = entry["weight"].grad
            if gradient is None:
                continue
            baseline = baselines[name]
            delta = gradient.detach().clone() if baseline is None else gradient.detach() - baseline
            basis = bases[entry["functional_key"]].to(device=delta.device, dtype=delta.dtype)
            before_numerator += (delta @ basis.T).double().square().sum().cpu()
            denominator += delta.double().square().sum().cpu()
            fn.project_delta_(gradient, baseline, basis)
            projected_delta = gradient.detach() if baseline is None else gradient.detach() - baseline
            after_numerator += (projected_delta @ basis.T).double().square().sum().cpu()
    if not float(denominator):
        return 0.0, 0.0
    return (
        math.sqrt(float(before_numerator / denominator)),
        math.sqrt(float(after_numerator / denominator)),
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg_path = (REPO / args.config).resolve()
    cfg = common.read_json(cfg_path)
    method = cfg["functional_nullspace"]
    if args.arm not in method["arms"]:
        raise ValueError(f"unregistered functional-nullspace arm: {args.arm}")
    arm = method["arms"][args.arm]
    projection_mode = str(arm["projection_mode"])
    if projection_mode not in {"combined_hard_nullspace", "risk_branch_only"}:
        raise ValueError(f"unsupported projection mode: {projection_mode}")
    full_steps = int(cfg["optimizer_steps"])
    max_steps = int(args.max_steps or (cfg["smoke_steps"] if args.run_kind == "smoke" else full_steps))
    if max_steps <= 0 or max_steps > full_steps:
        raise ValueError("max steps must be within the configured schedule")
    output = (REPO / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_steps = (
        {int(value) for value in args.save_steps.split(",") if value.strip()}
        if args.save_steps
        else set(int(value) for value in cfg["checkpoint_steps"])
    )
    if args.run_kind == "smoke":
        save_steps.update({0, 1, 2})

    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(int(cfg["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    risk_rows = common.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    targets = common.read_jsonl(REPO / cfg["safety_targets_path"])
    if len(risk_rows) != 256 or len(targets) != 80:
        raise ValueError("training data dimensions do not match the exploratory design")
    selected_calibration = calibration_rows(targets, int(method["calibration_prompt_clusters"]))
    model, tokenizer = frozen.load_model(cfg, resume_from=None)
    device = common.model_device(model)
    inventory = common.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RuntimeError("unexpected trainable parameter inventory")
    entries, representatives = lora_a_registry(model)
    bases, basis_reports = build_functional_bases(
        model=model,
        tokenizer=tokenizer,
        representatives=representatives,
        rows=selected_calibration,
        max_length=int(cfg["max_length"]),
        tokens_per_response=int(method["tokens_per_response"]),
        rank=int(method["basis_rank"]),
        epsilon=float(method["basis_epsilon"]),
    )
    initial_component_before = aggregate_component_ratio(entries, bases, "weight")
    project_entries_(entries, bases, "weight")
    initial_component_after = aggregate_component_ratio(entries, bases, "weight")
    if initial_component_after > 1e-5:
        raise FloatingPointError(f"initial nullspace residual too large: {initial_component_after}")

    risk_features = [
        common.response_feature(tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"]))
        for row in risk_rows
    ]
    target_features = {
        str(row["target_id"]): common.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"])
        )
        for row in targets
    }
    order = frozen.risk_order(len(risk_rows), int(cfg["seed"]))
    optimizer = frozen.make_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["warmup_steps"]), num_training_steps=full_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["fp16"]))
    trace: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    target_position = 0
    if 0 in save_steps:
        frozen.save_checkpoint(
            output, 0, model, tokenizer, optimizer, scheduler, scaler, None,
            target_position, trace, checkpoint_rows,
        )

    microsteps = int(cfg["gradient_accumulation_steps"])
    safety_mix = float(arm["safety_mix"])
    if safety_mix < 0.0 or safety_mix >= 1.0:
        raise ValueError("safety mix must be in [0, 1)")
    risk_coefficient = (1.0 - safety_mix) / microsteps
    safety_coefficient = safety_mix / microsteps
    optimizer.zero_grad(set_to_none=True)
    finite_updates = True
    skipped_updates = 0

    for global_step in range(max_steps):
        step_started = time.time()
        risk_losses: list[float] = []
        safety_losses: list[float] = []
        risk_projection_before: list[float] = []
        risk_projection_after: list[float] = []
        for micro_index in range(microsteps):
            position = global_step * microsteps + micro_index
            risk_index = order[position]
            model.train()
            gradient_baseline = (
                capture_entry_gradients(entries) if projection_mode == "risk_branch_only" else None
            )
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["fp16"])):
                risk_loss = model(
                    **common.batch_feature(risk_features[risk_index], device),
                    use_cache=False,
                    return_dict=True,
                ).loss
            scaler.scale(risk_loss * risk_coefficient).backward()
            risk_losses.append(float(risk_loss.detach().float().cpu()))
            if gradient_baseline is not None:
                ratio_before, ratio_after = project_new_gradient_delta_(
                    entries, bases, gradient_baseline
                )
                risk_projection_before.append(ratio_before)
                risk_projection_after.append(ratio_after)

            if safety_mix > 0.0:
                target_index = target_position % len(targets)
                target_position += 1
                target = targets[target_index]
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["fp16"])):
                    safety_loss = model(
                        **common.batch_feature(target_features[str(target["target_id"])], device),
                        use_cache=False,
                        return_dict=True,
                    ).loss
                scaler.scale(safety_loss * safety_coefficient).backward()
                safety_losses.append(float(safety_loss.detach().float().cpu()))

        scaler.unscale_(optimizer)
        combined_gradient_component = aggregate_component_ratio(entries, bases, "gradient")
        if projection_mode == "combined_hard_nullspace":
            gradient_component_before = combined_gradient_component
            project_entries_(entries, bases, "gradient")
            gradient_component_after = aggregate_component_ratio(entries, bases, "gradient")
        else:
            gradient_component_before = sum(risk_projection_before) / len(risk_projection_before)
            gradient_component_after = sum(risk_projection_after) / len(risk_projection_after)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
        )
        finite_updates = finite_updates and gradients_finite
        if not gradients_finite:
            raise FloatingPointError(f"non-finite gradient before step {global_step + 1}")
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"])))
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError(f"GradScaler skipped step {global_step + 1}")
        if projection_mode == "combined_hard_nullspace":
            project_entries_(entries, bases, "weight")
        weight_component_after = aggregate_component_ratio(entries, bases, "weight")
        if projection_mode == "combined_hard_nullspace" and weight_component_after > 1e-5:
            raise FloatingPointError(f"post-update nullspace residual too large: {weight_component_after}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = global_step + 1
        mean_risk = sum(risk_losses) / len(risk_losses)
        mean_safety = sum(safety_losses) / len(safety_losses) if safety_losses else None
        weighted_loss = (1.0 - safety_mix) * mean_risk + (
            safety_mix * mean_safety if mean_safety is not None else 0.0
        )
        trace.append(
            {
                "optimizer_step": completed_step,
                "projection_mode": projection_mode,
                "risk_loss_mean": mean_risk,
                "safety_loss_mean": "" if mean_safety is None else mean_safety,
                "weighted_loss": weighted_loss,
                "gradient_protected_ratio_before": gradient_component_before,
                "gradient_protected_ratio_after": gradient_component_after,
                "combined_gradient_protected_ratio_before_optimizer": combined_gradient_component,
                "weight_protected_ratio_after": weight_component_after,
                "optimizer_preclip_gradient_norm": preclip_norm,
                "learning_rate_after_update": float(scheduler.get_last_lr()[0]),
                "wall_seconds": time.time() - step_started,
            }
        )
        if completed_step in save_steps or completed_step == max_steps:
            frozen.save_checkpoint(
                output, completed_step, model, tokenizer, optimizer, scheduler, scaler, None,
                target_position, trace, checkpoint_rows,
            )
        common.write_csv(output / "training_trace.csv", trace)
        print(
            f"{args.arm} step {completed_step}/{max_steps} loss={weighted_loss:.6f} "
            f"project={gradient_component_before:.4f}->{gradient_component_after:.2e} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    basis_summary = {
        "basis_count": len(bases),
        "lora_a_module_count": len(entries),
        "requested_rank": int(method["basis_rank"]),
        "minimum_energy_fraction": min(row["top_rank_energy_fraction"] for row in basis_reports),
        "median_energy_fraction": sorted(row["top_rank_energy_fraction"] for row in basis_reports)[
            len(basis_reports) // 2
        ],
        "maximum_orthogonality_error": max(row["orthogonality_error"] for row in basis_reports),
        "basis_package_hash": common.hash_json(
            [{"key": row["functional_key"], "sha256": row["basis_sha256"]} for row in basis_reports]
        ),
    }
    common.write_json(output / "functional_basis_report.json", basis_reports)
    manifest = {
        "name": "step3_functional_nullspace_run_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "arm": args.arm,
        "run_kind": args.run_kind,
        "git_head": frozen.git_head(),
        "config_path": args.config,
        "config_sha256": common.sha256_file(cfg_path),
        "completed_steps": max_steps,
        "optimizer_steps": max_steps,
        "safety_mix": safety_mix,
        "projection_mode": projection_mode,
        "all_losses_and_gradients_finite": finite_updates,
        "skipped_updates": skipped_updates,
        "risk_order_hash": common.hash_json(order),
        "calibration_prompt_hashes": [common.stable_hash(str(row["sample_id"])) for row in selected_calibration],
        "calibration_target_hashes": [common.stable_hash(str(row["target_id"])) for row in selected_calibration],
        "calibration_forward_only": True,
        "basis_built_before_optimizer": True,
        "initial_weight_protected_ratio_before": initial_component_before,
        "initial_weight_protected_ratio_after": initial_component_after,
        "final_weight_protected_ratio": aggregate_component_ratio(entries, bases, "weight"),
        "protected_weight_interpretation": (
            "forced_nullspace_residual"
            if projection_mode == "combined_hard_nullspace"
            else "safety_branch_and_optimizer_permitted_component"
        ),
        "basis_summary": basis_summary,
        "checkpoint_steps": [int(row["step"]) for row in checkpoint_rows],
        "checkpoint_index_hash": common.hash_json(
            [{key: value for key, value in row.items() if key != "checkpoint"} for row in checkpoint_rows]
        ),
        "training_state_hash": frozen.training_state_hash(trace, checkpoint_rows),
        "trainable_parameter_inventory": inventory,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_memory,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory >= float(cfg["minimum_vram_headroom_fraction"]),
        "wall_clock_seconds": time.time() - started,
        "core100_opened": False,
        "heldout_opened": False,
        "candidate_outputs_generated": False,
    }
    common.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"arm": args.arm, "completed_steps": max_steps, "basis": basis_summary}))


if __name__ == "__main__":
    main()
