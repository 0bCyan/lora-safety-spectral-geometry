"""Run a fresh two-step T2R smoke arm with optimizer-unit diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import step3ct_t2r_common as recovery
import step3ct_training_common as parent
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg_path = REPO / args.config
    cfg = parent.read_json(cfg_path)
    if args.arm not in cfg["arms"]:
        raise ValueError("unregistered arm")
    arm = cfg["arms"][args.arm]
    output = REPO / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    resume_from = REPO / args.resume_from if args.resume_from else None

    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(int(cfg["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    risk_rows = parent.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    targets = parent.read_jsonl(REPO / cfg["safety_targets_path"])
    grouped = parent.targets_by_prompt(targets)
    order = frozen.risk_order(len(risk_rows), int(cfg["seed"]))
    model, tokenizer = frozen.load_model(cfg, resume_from)
    device = parent.model_device(model)
    inventory = parent.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RuntimeError("unexpected trainable parameter")
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
    head = parent.projection_head(
        int(model.config.hidden_size), int(cfg["projection_dim"]), int(cfg["projection_seed"])
    ).to(device)
    projection_hash = parent.module_state_hash(head)
    queue = None
    prefill_ids: list[str] = []
    if arm["binding"] != "none":
        queue, prefill_ids = parent.prefill_queue(
            model, tokenizer, head, targets, int(cfg["queue_size"]),
            int(cfg["queue_prefill_seed"]), int(cfg["max_length"]),
        )

    optimizer = frozen.make_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["warmup_steps"]),
        num_training_steps=int(cfg["optimizer_steps"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["fp16"]))
    start_step = 0
    target_position = 0
    trace: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    if resume_from is not None:
        state = torch.load(resume_from / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["global_step"])
        target_position = int(state["target_position"])
        trace = list(state["trace"])
        queue = None if state["queue"] is None else parent.DeterministicQueue.from_state_dict(state["queue"])
        frozen.restore_rng_state(state["rng"])
    optimizer.zero_grad(set_to_none=True)
    if start_step == 0:
        frozen.save_checkpoint(
            output, 0, model, tokenizer, optimizer, scheduler, scaler, queue,
            target_position, trace, checkpoint_rows,
        )

    p = float(cfg["safety_mix"])
    eta = float(arm["eta"])
    microsteps = int(cfg["gradient_accumulation_steps"])
    risk_coefficient = (1.0 - p) / microsteps
    safety_coefficient = p / microsteps
    binding_coefficient = p * eta / microsteps
    micro_rows = []
    ratio_rows = []
    aggregate_safe = None
    aggregate_bind = None
    skipped_updates = 0

    for global_step in range(start_step, 2):
        step_started = time.time()
        step_safe = None
        step_bind = None
        step_hard_active = 0
        for micro_index in range(microsteps):
            position = global_step * microsteps + micro_index
            risk_index = order[position]
            target_index = target_position % len(targets)
            target_position += 1
            risk_row = risk_rows[risk_index]
            safety_row = targets[target_index]
            prompt_id = str(safety_row["sample_id"])
            positives = grouped[prompt_id]
            scale_value = float(scaler.get_scale())

            model.train()
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["fp16"])):
                risk_loss = model(
                    **parent.batch_feature(risk_features[risk_index], device),
                    use_cache=False, return_dict=True,
                ).loss
            scaler.scale(risk_loss * risk_coefficient).backward()
            before_safety = parent.capture_lora_gradients(model)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["fp16"])):
                safety_loss = model(
                    **parent.batch_feature(target_features[str(safety_row["target_id"])], device),
                    use_cache=False, return_dict=True,
                ).loss
            scaler.scale(safety_loss * safety_coefficient).backward()
            after_safety = parent.capture_lora_gradients(model)
            safety_gradient = parent.gradient_delta(
                after_safety, before_safety, scale_value * safety_coefficient
            )
            if step_safe is None:
                step_safe = recovery.zeros_like_float64(safety_gradient)
            recovery.add_in_place(step_safe, safety_gradient)

            binding_metrics: dict[str, Any] = {
                "binding_loss": 0.0, "global_binding_loss": 0.0,
                "hard_binding_loss": 0.0, "hard_rows": 0,
                "positive_probability": "", "negative_count": 0,
                "queue_count": 0, "queue_hash_before": "", "queue_hash_after": "",
                "inserted_target_hashes": [], "evicted_target_hashes": [], "negative_id_hash": "",
            }
            binding_gradient = None
            if queue is not None:
                before_binding = parent.capture_lora_gradients(model)
                binding_metrics = parent.binding_backward(
                    model, tokenizer, head, safety_row, positives, queue,
                    int(cfg["max_length"]), float(cfg["temperature"]),
                    float(arm["hard_term_multiplier"]), binding_coefficient * scale_value,
                    update_queue=True,
                )
                after_binding = parent.capture_lora_gradients(model)
                binding_gradient = parent.gradient_delta(
                    after_binding, before_binding, scale_value * binding_coefficient
                )
                if step_bind is None:
                    step_bind = recovery.zeros_like_float64(binding_gradient)
                recovery.add_in_place(step_bind, binding_gradient)
                step_hard_active += int(int(binding_metrics["hard_rows"]) > 0)

            trace.append(
                {
                    "global_step_before_update": global_step,
                    "micro_index": micro_index,
                    "risk_row_hash": parent.stable_hash(str(risk_row.get("sample_id", risk_index))),
                    "safety_prompt_hash": parent.stable_hash(prompt_id),
                    "safety_target_hash": parent.stable_hash(str(safety_row["target_id"])),
                    "risk_loss": float(risk_loss.detach().float().cpu()),
                    "safety_loss": float(safety_loss.detach().float().cpu()),
                    "global_binding_loss": binding_metrics["global_binding_loss"],
                    "hard_binding_loss": binding_metrics["hard_binding_loss"],
                    "binding_loss": binding_metrics["binding_loss"],
                    "risk_coefficient": 1.0 - p,
                    "safety_coefficient": p,
                    "eta": eta,
                    "weighted_binding_coefficient": p * eta,
                    "total_loss": float(parent.weighted_total_loss(
                        float(risk_loss.detach().float().cpu()),
                        float(safety_loss.detach().float().cpu()),
                        float(binding_metrics["binding_loss"]), p, eta,
                    )),
                    "positive_probability": binding_metrics["positive_probability"],
                    "hard_rows": binding_metrics["hard_rows"],
                    "negative_count": binding_metrics["negative_count"],
                    "queue_count": binding_metrics["queue_count"],
                    "queue_hash_before": binding_metrics["queue_hash_before"],
                    "queue_hash_after": binding_metrics["queue_hash_after"],
                }
            )
            micro_rows.append(
                {
                    "position": position,
                    "optimizer_step": global_step + 1,
                    "micro_index": micro_index,
                    "safety_gradient_norm": recovery.gradient_norm(safety_gradient),
                    "binding_gradient_norm": 0.0 if binding_gradient is None else recovery.gradient_norm(binding_gradient),
                    "safety_gradient_hash": recovery.gradient_hash(safety_gradient),
                    "binding_gradient_hash": "" if binding_gradient is None else recovery.gradient_hash(binding_gradient),
                    "first_microstep_eta_weighted_ratio": (
                        0.0 if binding_gradient is None
                        else eta * recovery.gradient_norm(binding_gradient) / recovery.gradient_norm(safety_gradient)
                    ),
                    "hard_active": int(binding_metrics["hard_rows"]) > 0,
                }
            )

        if step_safe is None:
            raise RuntimeError("missing safety gradient sum")
        if aggregate_safe is None:
            aggregate_safe = recovery.zeros_like_float64(step_safe)
        recovery.add_in_place(aggregate_safe, step_safe)
        if step_bind is not None:
            if aggregate_bind is None:
                aggregate_bind = recovery.zeros_like_float64(step_bind)
            recovery.add_in_place(aggregate_bind, step_bind)
        safe_norm = recovery.gradient_norm(step_safe)
        bind_norm = 0.0 if step_bind is None else recovery.gradient_norm(step_bind)
        ratio_rows.append(
            {
                "optimizer_step": global_step + 1,
                "positions": list(range(global_step * 4, global_step * 4 + 4)),
                "safety_gradient_norm": safe_norm,
                "binding_gradient_norm": bind_norm,
                "unweighted_binding_safety_ratio": 0.0 if step_bind is None else bind_norm / safe_norm,
                "eta_weighted_binding_safety_ratio": 0.0 if step_bind is None else eta * bind_norm / safe_norm,
                "hard_active_positions": step_hard_active,
                "safety_gradient_hash": recovery.gradient_hash(step_safe),
                "binding_gradient_hash": "" if step_bind is None else recovery.gradient_hash(step_bind),
                "finite_nonzero": recovery.gradients_finite_nonzero(step_safe) and (
                    step_bind is None or recovery.gradients_finite_nonzero(step_bind)
                ),
            }
        )

        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise FloatingPointError("non-finite optimizer gradient")
        preclip = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"])))
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError("GradScaler skipped update")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        trace[-1]["optimizer_preclip_gradient_norm"] = preclip
        trace[-1]["learning_rate_after_update"] = float(scheduler.get_last_lr()[0])
        trace[-1]["optimizer_step_wall_seconds"] = time.time() - step_started
        frozen.save_checkpoint(
            output, global_step + 1, model, tokenizer, optimizer, scheduler, scaler, queue,
            target_position, trace, checkpoint_rows,
        )
        parent.write_csv(output / "training_trace.csv", trace)
        print(f"{args.arm} T2R step {global_step + 1}/2", flush=True)

    aggregate_ratio = None
    if aggregate_safe is not None and aggregate_bind is not None:
        aggregate_ratio = eta * recovery.gradient_norm(aggregate_bind) / recovery.gradient_norm(aggregate_safe)
    ratio_report = {
        "name": "step3ct_t2r_optimizer_unit_gradient_ratios_v1",
        "arm": args.arm,
        "eta": eta,
        "start_step": start_step,
        "optimizer_steps": ratio_rows,
        "covered_positions": [row["position"] for row in micro_rows],
        "aggregate_eta_weighted_binding_safety_ratio": aggregate_ratio,
        "c3b_hard_active_positions": sum(row["hard_active"] for row in micro_rows),
    }
    parent.write_json(output / "micro_gradient_diagnostics.json", micro_rows)
    parent.write_json(output / "optimizer_unit_gradient_ratios.json", ratio_report)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    manifest = {
        "name": "step3ct_t2r_smoke_run_manifest_v1",
        "arm": args.arm,
        "git_head": git_head(),
        "config_path": args.config,
        "config_sha256": parent.sha256_file(cfg_path),
        "start_step": start_step,
        "completed_steps": 2,
        "optimizer_steps_executed": 2 - start_step,
        "skipped_updates": skipped_updates,
        "all_losses_and_gradients_finite": all(row["finite_nonzero"] for row in ratio_rows),
        "risk_order_hash": parent.hash_json(order),
        "target_schedule_hash": parent.hash_json([str(row["target_id"]) for row in targets]),
        "target_position": target_position,
        "projection_hash_before": projection_hash,
        "projection_hash_after": parent.module_state_hash(head),
        "projection_unchanged": projection_hash == parent.module_state_hash(head),
        "projection_trainable_parameters": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
        "queue_constructed": queue is not None,
        "queue_count": 0 if queue is None else queue.count,
        "queue_hash": "" if queue is None else queue.content_hash(),
        "queue_prefill_target_hash": parent.hash_json(prefill_ids) if prefill_ids else "",
        "trainable_parameter_inventory": inventory,
        "checkpoint_steps": [int(row["step"]) for row in checkpoint_rows],
        "checkpoint_index_hash": parent.hash_json([
            {key: value for key, value in row.items() if key != "checkpoint"} for row in checkpoint_rows
        ]),
        "training_state_hash": frozen.training_state_hash(trace, checkpoint_rows),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_memory,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory >= float(cfg["minimum_vram_headroom_fraction"]),
        "wall_clock_seconds": time.time() - started,
        "core100_opened": False,
        "heldout_opened": False,
        "candidate_outputs_generated": False,
    }
    parent.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"arm": args.arm, "start_step": start_step, "aggregate_ratio": aggregate_ratio}))


if __name__ == "__main__":
    main()
