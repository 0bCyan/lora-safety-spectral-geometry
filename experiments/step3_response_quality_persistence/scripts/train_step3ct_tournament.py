"""Run one frozen Step3C-T arm with deterministic checkpoint/resume support."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import step3ct_training_common as common
import step3ct_t2r_common as recovery


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-kind", choices=["smoke", "main"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--save-steps", default=None, help="Comma-separated optimizer steps.")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def diagnostic_label_for_step(
    run_kind: str,
    global_step: int,
    max_steps: int,
    full_steps: int,
    diagnostic_steps: list[int],
) -> int | None:
    if run_kind == "smoke":
        return global_step + 1
    if global_step in diagnostic_steps:
        return global_step
    if max_steps == full_steps and global_step == full_steps - 1:
        return full_steps
    return None


def accumulate_gradient_unit(
    total: dict[str, Any] | None,
    gradients: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if gradients is None:
        return total
    if total is None:
        total = recovery.zeros_like_float64(gradients)
    recovery.add_in_place(total, gradients)
    return total


def optimizer_unit_diagnostic(
    checkpoint_step: int,
    optimizer_step: int,
    positions: list[int],
    safety_gradients: dict[str, Any],
    binding_gradients: dict[str, Any] | None,
    eta: float,
    hard_active_positions: int,
) -> dict[str, Any]:
    safety_norm = recovery.gradient_norm(safety_gradients)
    binding_norm = 0.0 if binding_gradients is None else recovery.gradient_norm(binding_gradients)
    return {
        "checkpoint_step": checkpoint_step,
        "optimizer_step": optimizer_step,
        "gradient_unit": "sum_of_four_unweighted_microstep_branch_gradients",
        "positions": positions,
        "microstep_count": len(positions),
        "safety_gradient_norm": safety_norm,
        "binding_gradient_norm": binding_norm,
        "unweighted_binding_safety_ratio": binding_norm / safety_norm if safety_norm else math.nan,
        "eta_weighted_binding_safety_ratio": eta * binding_norm / safety_norm if safety_norm else math.nan,
        "binding_safety_cosine": (
            "" if binding_gradients is None else recovery.gradient_cosine(binding_gradients, safety_gradients)
        ),
        "safety_energy": common.gradient_energy(safety_gradients),
        "binding_energy": {} if binding_gradients is None else common.gradient_energy(binding_gradients),
        "safety_gradient_hash": recovery.gradient_hash(safety_gradients),
        "binding_gradient_hash": "" if binding_gradients is None else recovery.gradient_hash(binding_gradients),
        "hard_active_positions": hard_active_positions,
        "finite_nonzero": recovery.gradients_finite_nonzero(safety_gradients)
        and (binding_gradients is None or recovery.gradients_finite_nonzero(binding_gradients)),
    }


def load_model(cfg: dict[str, Any], resume_from: Path | None) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    set_seed(int(cfg["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=bool(cfg["local_files_only"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=bool(cfg["local_files_only"]),
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
    )
    if resume_from is not None:
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True)
    else:
        set_seed(int(cfg["seed"]))
        lora = LoraConfig(
            r=int(cfg["lora_rank"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg["target_modules"]),
        )
        model = get_peft_model(base, lora)
    model.config.use_cache = False
    model.train()
    return model, tokenizer


def make_optimizer(model: Any, cfg: dict[str, Any]) -> Any:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=float(cfg["learning_rate"]),
        betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
        eps=float(cfg["adam_epsilon"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def risk_order(row_count: int, seed: int) -> list[int]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(row_count, generator=generator).tolist()


def rng_state() -> dict[str, Any]:
    import random
    import torch

    return {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import random
    import torch

    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    root: Path,
    step: int,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    queue: common.DeterministicQueue | None,
    target_position: int,
    trace: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> Path:
    import torch

    checkpoint = root / "checkpoints" / f"step{step:04d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    state = {
        "global_step": step,
        "target_position": target_position,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "queue": None if queue is None else queue.state_dict(),
        "rng": rng_state(),
        "trace": trace,
    }
    torch.save(state, checkpoint / "training_state.pt")
    adapter_path = checkpoint / "adapter_model.safetensors"
    row = {
        "step": step,
        "checkpoint": checkpoint.as_posix(),
        "adapter_sha256": common.sha256_file(adapter_path),
        "queue_hash": "" if queue is None else queue.content_hash(),
        "target_position": target_position,
    }
    existing = {int(item["step"]): item for item in checkpoint_rows}
    existing[step] = row
    checkpoint_rows[:] = [existing[key] for key in sorted(existing)]
    common.write_csv(root / "checkpoint_index.csv", checkpoint_rows)
    return checkpoint


def training_state_hash(trace: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]) -> str:
    normalized = {
        "trace": trace,
        "checkpoints": [
            {key: value for key, value in row.items() if key != "checkpoint"} for row in checkpoint_rows
        ],
    }
    return common.hash_json(normalized)


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg_path = (REPO / args.config).resolve()
    cfg = common.read_json(cfg_path)
    if args.arm not in cfg["arms"]:
        raise ValueError(f"unregistered arm: {args.arm}")
    arm = cfg["arms"][args.arm]
    full_steps = int(cfg["optimizer_steps"])
    max_steps = int(args.max_steps or (cfg["smoke_steps"] if args.run_kind == "smoke" else full_steps))
    if max_steps > full_steps or max_steps <= 0:
        raise ValueError("max steps must be within the registered 64-step schedule")
    output = (REPO / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume_from = (REPO / args.resume_from).resolve() if args.resume_from else None
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
        raise RuntimeError("CUDA is required for the registered 4-bit tournament")
    set_seed(int(cfg["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    risk_rows = common.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    targets = common.read_jsonl(REPO / cfg["safety_targets_path"])
    target_groups = common.targets_by_prompt(targets)
    if len(risk_rows) != 256 or len(targets) != 80 or len(target_groups) != 40:
        raise ValueError("registered training data dimensions do not match")
    if not all(len(rows) == 2 for rows in target_groups.values()):
        raise ValueError("every safety prompt must have exactly two approved targets")
    order = risk_order(len(risk_rows), int(cfg["seed"]))
    model, tokenizer = load_model(cfg, resume_from)
    device = common.model_device(model)
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
    inventory = common.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RuntimeError(f"unexpected trainable parameters: {inventory['unexpected_trainable']}")
    hidden_size = int(model.config.hidden_size)
    head = common.projection_head(hidden_size, int(cfg["projection_dim"]), int(cfg["projection_seed"])).to(device)
    projection_hash = common.module_state_hash(head)

    queue: common.DeterministicQueue | None = None
    prefill_ids: list[str] = []
    if arm["binding"] != "none":
        queue, prefill_ids = common.prefill_queue(
            model,
            tokenizer,
            head,
            targets,
            int(cfg["queue_size"]),
            int(cfg["queue_prefill_seed"]),
            int(cfg["max_length"]),
        )

    optimizer = make_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_steps"]),
        num_training_steps=full_steps,
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
        if arm["binding"] == "none":
            if state["queue"] is not None:
                raise ValueError("C0-R resume unexpectedly contains a queue")
            queue = None
        else:
            queue = common.DeterministicQueue.from_state_dict(state["queue"])
        restore_rng_state(state["rng"])
    optimizer.zero_grad(set_to_none=True)

    if start_step == 0 and 0 in save_steps:
        save_checkpoint(
            output, 0, model, tokenizer, optimizer, scheduler, scaler, queue,
            target_position, trace, checkpoint_rows,
        )

    diagnostic_rows: list[dict[str, Any]] = []
    microsteps = int(cfg["gradient_accumulation_steps"])
    p = float(cfg["safety_mix"])
    eta = float(arm["eta"])
    risk_coefficient = (1.0 - p) / microsteps
    safety_coefficient = p / microsteps
    binding_coefficient = p * eta / microsteps
    skipped_updates = 0
    finite_updates = True
    diagnostic_steps = [int(value) for value in cfg["diagnostic_steps"]]

    for global_step in range(start_step, max_steps):
        step_started = time.time()
        scale_value = float(scaler.get_scale())
        step_total_loss = 0.0
        diagnostic_label = diagnostic_label_for_step(
            args.run_kind, global_step, max_steps, full_steps, diagnostic_steps
        )
        step_safety_gradients: dict[str, Any] | None = None
        step_binding_gradients: dict[str, Any] | None = None
        step_hard_active_positions = 0
        for micro_index in range(microsteps):
            draw_index = global_step * microsteps + micro_index
            risk_index = order[draw_index]
            target_index = target_position % len(targets)
            target_position += 1
            risk_row = risk_rows[risk_index]
            safety_row = targets[target_index]
            prompt_id = str(safety_row["sample_id"])
            positives = target_groups[prompt_id]

            model.train()
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["fp16"])):
                risk_batch = common.batch_feature(risk_features[risk_index], device)
                risk_loss = model(**risk_batch, use_cache=False, return_dict=True).loss
            scaler.scale(risk_loss * risk_coefficient).backward()

            before_safety = common.capture_lora_gradients(model) if diagnostic_label is not None else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["fp16"])):
                safety_batch = common.batch_feature(target_features[str(safety_row["target_id"])], device)
                safety_loss = model(**safety_batch, use_cache=False, return_dict=True).loss
            scaler.scale(safety_loss * safety_coefficient).backward()
            after_safety = common.capture_lora_gradients(model) if diagnostic_label is not None else None
            safety_gradients = None
            if before_safety is not None and after_safety is not None:
                safety_gradients = common.gradient_delta(
                    after_safety, before_safety, scale_value * safety_coefficient
                )
                step_safety_gradients = accumulate_gradient_unit(step_safety_gradients, safety_gradients)

            binding_metrics: dict[str, Any] = {
                "binding_loss": 0.0,
                "global_binding_loss": 0.0,
                "hard_binding_loss": 0.0,
                "hard_rows": 0,
                "positive_probability": "",
                "negative_count": 0,
                "queue_count": 0,
                "queue_hash_before": "",
                "queue_hash_after": "",
                "inserted_target_hashes": [],
                "evicted_target_hashes": [],
                "negative_id_hash": "",
            }
            before_binding = common.capture_lora_gradients(model) if diagnostic_label is not None and queue is not None else None
            if queue is not None:
                binding_metrics = common.binding_backward(
                    model,
                    tokenizer,
                    head,
                    safety_row,
                    positives,
                    queue,
                    int(cfg["max_length"]),
                    float(cfg["temperature"]),
                    float(arm["hard_term_multiplier"]),
                    binding_coefficient * scale_value,
                    update_queue=True,
                )
            after_binding = common.capture_lora_gradients(model) if before_binding is not None else None
            binding_gradients = None
            if before_binding is not None and after_binding is not None:
                binding_gradients = common.gradient_delta(
                    after_binding, before_binding, scale_value * binding_coefficient
                )
                step_binding_gradients = accumulate_gradient_unit(step_binding_gradients, binding_gradients)
            step_hard_active_positions += int(int(binding_metrics["hard_rows"]) > 0)

            total_loss = common.weighted_total_loss(
                float(risk_loss.detach().float().cpu()),
                float(safety_loss.detach().float().cpu()),
                float(binding_metrics["binding_loss"]),
                p,
                eta,
            )
            step_total_loss += float(total_loss)
            trace.append(
                {
                    "global_step_before_update": global_step,
                    "micro_index": micro_index,
                    "risk_row_hash": common.stable_hash(str(risk_row.get("sample_id", risk_index))),
                    "safety_prompt_hash": common.stable_hash(prompt_id),
                    "safety_target_hash": common.stable_hash(str(safety_row["target_id"])),
                    "risk_loss": float(risk_loss.detach().float().cpu()),
                    "safety_loss": float(safety_loss.detach().float().cpu()),
                    "global_binding_loss": binding_metrics["global_binding_loss"],
                    "hard_binding_loss": binding_metrics["hard_binding_loss"],
                    "binding_loss": binding_metrics["binding_loss"],
                    "risk_coefficient": 1.0 - p,
                    "safety_coefficient": p,
                    "eta": eta,
                    "weighted_binding_coefficient": p * eta,
                    "total_loss": float(total_loss),
                    "positive_probability": binding_metrics["positive_probability"],
                    "hard_rows": binding_metrics["hard_rows"],
                    "negative_count": binding_metrics["negative_count"],
                    "queue_count": binding_metrics["queue_count"],
                    "queue_hash_before": binding_metrics["queue_hash_before"],
                    "queue_hash_after": binding_metrics["queue_hash_after"],
                    "inserted_target_hashes": json.dumps(binding_metrics["inserted_target_hashes"]),
                    "evicted_target_hashes": json.dumps(binding_metrics["evicted_target_hashes"]),
                    "negative_id_hash": binding_metrics["negative_id_hash"],
                }
            )

        if diagnostic_label is not None:
            if step_safety_gradients is None:
                raise RuntimeError(f"missing safety diagnostic at optimizer step {global_step + 1}")
            diagnostic_rows.append(
                optimizer_unit_diagnostic(
                    checkpoint_step=diagnostic_label,
                    optimizer_step=global_step + 1,
                    positions=list(range(global_step * microsteps, (global_step + 1) * microsteps)),
                    safety_gradients=step_safety_gradients,
                    binding_gradients=step_binding_gradients,
                    eta=eta,
                    hard_active_positions=step_hard_active_positions,
                )
            )

        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
        )
        finite_updates = finite_updates and gradients_finite and math.isfinite(step_total_loss)
        if not gradients_finite:
            raise FloatingPointError(f"non-finite gradient before optimizer step {global_step + 1}")
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"])))
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError(f"GradScaler skipped optimizer step {global_step + 1}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = global_step + 1
        trace[-1]["optimizer_preclip_gradient_norm"] = preclip_norm
        trace[-1]["learning_rate_after_update"] = float(scheduler.get_last_lr()[0])
        trace[-1]["optimizer_step_wall_seconds"] = time.time() - step_started
        if completed_step in save_steps or completed_step == max_steps:
            save_checkpoint(
                output, completed_step, model, tokenizer, optimizer, scheduler, scaler, queue,
                target_position, trace, checkpoint_rows,
            )
        common.write_csv(output / "training_trace.csv", trace)
        print(
            f"{args.arm} step {completed_step}/{max_steps} loss={step_total_loss / microsteps:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_memory, _ = torch.cuda.mem_get_info(device)
    projection_hash_after = common.module_state_hash(head)
    exposure_counts = [0] * len(targets)
    for position in range(target_position):
        exposure_counts[position % len(targets)] += 1
    exposure_rows = [
        {
            "target_hash": common.stable_hash(str(row["target_id"])),
            "prompt_hash": common.stable_hash(str(row["sample_id"])),
            "exposure_count": exposure_counts[index],
        }
        for index, row in enumerate(targets)
    ]
    common.write_csv(output / "target_exposure_manifest.csv", exposure_rows)
    common.write_json(output / "gradient_diagnostics.json", diagnostic_rows)
    manifest = {
        "name": "step3ct_arm_run_manifest_v1",
        "arm": args.arm,
        "run_kind": args.run_kind,
        "git_head": git_head(),
        "config_path": args.config,
        "config_sha256": common.sha256_file(cfg_path),
        "start_step": start_step,
        "completed_steps": max_steps,
        "optimizer_steps": max_steps - start_step,
        "skipped_updates": skipped_updates,
        "all_losses_and_gradients_finite": finite_updates,
        "risk_rows": len(risk_rows),
        "risk_order_hash": common.hash_json(order),
        "safety_targets": len(targets),
        "safety_prompt_clusters": len(target_groups),
        "target_schedule_hash": common.hash_json([str(row["target_id"]) for row in targets]),
        "target_position": target_position,
        "target_exposure_min": min(exposure_counts),
        "target_exposure_max": max(exposure_counts),
        "projection_hash_before": projection_hash,
        "projection_hash_after": projection_hash_after,
        "projection_unchanged": projection_hash == projection_hash_after,
        "projection_trainable_parameters": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
        "queue_constructed": queue is not None,
        "queue_count": 0 if queue is None else queue.count,
        "queue_hash": "" if queue is None else queue.content_hash(),
        "queue_prefill_target_hash": common.hash_json(prefill_ids) if prefill_ids else "",
        "trainable_parameter_inventory": inventory,
        "optimizer": "torch.optim.AdamW",
        "scheduler": "linear_with_warmup_0",
        "gradient_scaler": bool(cfg["fp16"]),
        "checkpoint_steps": [int(row["step"]) for row in checkpoint_rows],
        "checkpoint_index_hash": common.hash_json(
            [{key: value for key, value in row.items() if key != "checkpoint"} for row in checkpoint_rows]
        ),
        "training_state_hash": training_state_hash(trace, checkpoint_rows),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_memory,
        "free_vram_bytes_at_end": int(free_memory),
        "peak_allocated_fraction": peak_allocated / total_memory,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "minimum_vram_headroom_fraction": float(cfg["minimum_vram_headroom_fraction"]),
        "vram_gate_pass": 1.0 - peak_reserved / total_memory >= float(cfg["minimum_vram_headroom_fraction"]),
        "wall_clock_seconds": time.time() - started,
        "core100_opened": False,
        "heldout_opened": False,
        "candidate_outputs_generated": False,
        "gradient_diagnostic_definition": "optimizer_unit_sum_of_four_unweighted_microstep_branch_gradients",
        "gradient_diagnostic_labels": [int(row["checkpoint_step"]) for row in diagnostic_rows],
        "gradient_diagnostics_hash": common.hash_json(diagnostic_rows),
    }
    common.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"arm": args.arm, "completed_steps": max_steps, "manifest": str(output / 'run_manifest.json')}))


if __name__ == "__main__":
    main()
