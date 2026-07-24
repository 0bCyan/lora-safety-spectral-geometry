"""Freeze Step3C-T T1 sources and run no-update GPU preflight checks."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import step3ct_training_common as common
import train_step3ct_tournament as trainer


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "experiments/step3_response_quality_persistence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1.json",
    )
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def run_implementation_tests() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "-m",
            "py_compile",
            "experiments/step3_response_quality_persistence/scripts/step3ct_training_common.py",
            "experiments/step3_response_quality_persistence/scripts/train_step3ct_tournament.py",
            "experiments/step3_response_quality_persistence/scripts/test_step3ct_training_common.py",
            "experiments/step3_response_quality_persistence/scripts/run_step3ct_t1_preflight.py",
        ],
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "experiments/step3_response_quality_persistence/scripts",
            "-p",
            "test_step3ct_training_common.py",
            "-v",
        ],
    ]
    rows = []
    for command in commands:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        rows.append(
            {
                "command": subprocess.list2cmdline(command),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return {
        "name": "step3ct_t1_implementation_tests_v1",
        "commands": rows,
        "coverage": [
            "syntax and import",
            "left/right padding and truncation masks",
            "two-positive indexing and same-prompt exclusion",
            "global/hard-disable equivalence and singleton fallback",
            "objective parentheses",
            "queue prefill/FIFO/wraparound/exclusion/resume/hash",
            "fixed projection and CPU RNG isolation",
            "finite embedding gradients and C3a/C3b direction comparison",
            "toy optimizer/scheduler checkpoint resume determinism",
        ],
        "passed": all(row["returncode"] == 0 for row in rows),
    }


def source_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    registry = common.read_json(REPO / cfg["registry"])
    historical = common.read_json(REPO / cfg["historical_config"])
    validation = common.read_json(REPO / cfg["validation_route"])
    joint = common.read_json(REPO / cfg["joint_route"])
    targets = common.read_jsonl(REPO / cfg["safety_targets_path"])
    prompts = common.read_jsonl(REPO / cfg["safety_train_path"])
    risk = common.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    grouped = common.targets_by_prompt(targets)
    expected = {
        "model_name_or_path": registry["model_name_or_path"],
        "seed": registry["seed"],
        "optimizer_steps": registry["optimizer_steps"],
        "train_examples": registry["training"]["train_examples"],
        "learning_rate": registry["training"]["learning_rate"],
        "gradient_accumulation_steps": registry["training"]["gradient_accumulation_steps"],
        "max_length": registry["training"]["max_length"],
        "lora_rank": registry["training"]["lora_rank"],
        "lora_alpha": registry["training"]["lora_alpha"],
        "lora_dropout": registry["training"]["lora_dropout"],
        "target_modules": registry["training"]["target_modules"],
        "safety_mix": registry["training"]["safety_mix"],
        "projection_dim": registry["binding"]["projection_dim"],
        "projection_seed": registry["binding"]["projection_seed"],
        "queue_size": registry["binding"]["queue_size"],
        "temperature": registry["binding"]["temperature"],
    }
    mismatches = {
        key: {"expected": value, "actual": cfg.get(key)}
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    historical_checks = {
        "model": cfg["model_name_or_path"] == historical["model_name_or_path"],
        "train_examples": cfg["train_examples"] == historical["train_per_family"],
        "max_length": cfg["max_length"] == historical["max_length"],
        "learning_rate": cfg["learning_rate"] == historical["learning_rate"],
        "gradient_accumulation": cfg["gradient_accumulation_steps"] == historical["gradient_accumulation_steps"],
        "lora_contract": (
            cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["target_modules"]
        ) == (
            historical["lora_rank"], historical["lora_alpha"], historical["lora_dropout"], historical["target_modules"]
        ),
    }
    target_prompt_ids = set(grouped)
    safety_prompt_ids = {str(row["sample_id"]) for row in prompts}
    result = {
        "registry_name": registry["name"],
        "registry_authorizes_t1": bool(registry["authorization"]["implementation_and_no_update_preflight"]),
        "registry_mismatches": mismatches,
        "historical_contract_checks": historical_checks,
        "measurement_validation_pass": bool(validation["overall_pass"]),
        "joint_measurement_pass": bool(joint["measurement_validation_pass"]),
        "joint_c3_pass": bool(joint["c3_at_least_one_candidate_pass"]),
        "risk_rows": len(risk),
        "target_rows": len(targets),
        "target_prompt_clusters": len(grouped),
        "two_targets_per_prompt": all(len(rows) == 2 for rows in grouped.values()),
        "all_targets_approved": all(row.get("reviewer_decision") == "approve" for row in targets),
        "safety_train_rows": len(prompts),
        "target_prompt_ids_equal_safety_train": target_prompt_ids == safety_prompt_ids,
        "core100_opened": False,
        "heldout_opened": False,
    }
    result["passed"] = (
        result["registry_authorizes_t1"]
        and not mismatches
        and all(historical_checks.values())
        and result["measurement_validation_pass"]
        and result["joint_measurement_pass"]
        and result["joint_c3_pass"]
        and result["risk_rows"] == 256
        and result["target_rows"] == 80
        and result["target_prompt_clusters"] == 40
        and result["two_targets_per_prompt"]
        and result["all_targets_approved"]
        and result["safety_train_rows"] == 40
        and result["target_prompt_ids_equal_safety_train"]
    )
    return result


def zero_grad(model: Any) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def actual_rng_isolation_test(
    model: Any,
    tokenizer: Any,
    head: Any,
    cfg: dict[str, Any],
    risk_row: dict[str, Any],
    safety_row: dict[str, Any],
    positive_rows: list[dict[str, Any]],
    queue: common.DeterministicQueue,
) -> dict[str, Any]:
    import torch

    device = common.model_device(model)
    risk_feature = common.response_feature(tokenizer, risk_row["prompt"], risk_row["response"], int(cfg["max_length"]))
    safety_feature = common.response_feature(tokenizer, safety_row["prompt"], safety_row["response"], int(cfg["max_length"]))

    def run(include_zero_binding: bool) -> list[float]:
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        model.train()
        losses = []
        for iteration in range(2):
            risk_loss = model(**common.batch_feature(risk_feature, device), use_cache=False, return_dict=True).loss
            safety_loss = model(**common.batch_feature(safety_feature, device), use_cache=False, return_dict=True).loss
            losses.extend([float(risk_loss.detach().cpu()), float(safety_loss.detach().cpu())])
            if include_zero_binding and iteration == 0:
                zero_grad(model)
                common.binding_backward(
                    model, tokenizer, head, safety_row, positive_rows,
                    common.DeterministicQueue.from_state_dict(queue.state_dict()),
                    int(cfg["max_length"]), float(cfg["temperature"]), 0.0, 0.0, update_queue=False,
                )
                zero_grad(model)
        return losses

    baseline = run(False)
    toggled = run(True)
    differences = [abs(left - right) for left, right in zip(baseline, toggled)]
    return {
        "baseline_task_safety_losses": baseline,
        "zero_weight_binding_task_safety_losses": toggled,
        "maximum_absolute_difference": max(differences),
        "losses_equal": max(differences) <= 1e-6,
        "draw_trace_equal_by_construction": True,
    }


def numerical_preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = trainer.load_model(cfg, None)
    device = common.model_device(model)
    inventory = common.trainable_parameter_inventory(model)
    targets = common.read_jsonl(REPO / cfg["safety_targets_path"])
    grouped = common.targets_by_prompt(targets)
    prompt_id = sorted(grouped, key=lambda value: common.stable_hash(f"20260728:{value}"))[0]
    positives = grouped[prompt_id]
    safety_row = positives[0]
    risk_row = common.read_jsonl(REPO / cfg["risk_data_path"], limit=1)[0]
    head = common.projection_head(
        int(model.config.hidden_size), int(cfg["projection_dim"]), int(cfg["projection_seed"])
    ).to(device)
    projection_before = common.module_state_hash(head)
    queue, prefill_ids = common.prefill_queue(
        model, tokenizer, head, targets, int(cfg["queue_size"]),
        int(cfg["queue_prefill_seed"]), int(cfg["max_length"]),
    )
    queue_state = queue.state_dict()

    model.eval()
    zero_grad(model)
    safety_feature = common.response_feature(
        tokenizer, safety_row["prompt"], safety_row["response"], int(cfg["max_length"])
    )
    safety_output = model(
        **common.batch_feature(safety_feature, device), use_cache=False, return_dict=True
    )
    safety_output.loss.backward()
    safety_gradients = common.capture_lora_gradients(model)

    binding_gradients: dict[str, dict[str, Any]] = {}
    binding_metrics: dict[str, dict[str, Any]] = {}
    for arm_name in ["C3a_multi_positive_global_binding", "C3b_hierarchical_hard_negative_binding"]:
        zero_grad(model)
        arm = cfg["arms"][arm_name]
        metrics = common.binding_backward(
            model, tokenizer, head, safety_row, positives,
            common.DeterministicQueue.from_state_dict(queue_state),
            int(cfg["max_length"]), float(cfg["temperature"]),
            float(arm["hard_term_multiplier"]), 1.0, update_queue=False,
        )
        binding_metrics[arm_name] = metrics
        binding_gradients[arm_name] = common.capture_lora_gradients(model)

    safety_norm = common.gradient_norm(safety_gradients)
    gradient_rows: dict[str, Any] = {}
    for arm_name, gradients in binding_gradients.items():
        norm = common.gradient_norm(gradients)
        eta = float(cfg["arms"][arm_name]["eta"])
        gradient_rows[arm_name] = {
            "binding_loss": binding_metrics[arm_name]["binding_loss"],
            "global_binding_loss": binding_metrics[arm_name]["global_binding_loss"],
            "hard_binding_loss": binding_metrics[arm_name]["hard_binding_loss"],
            "hard_rows": binding_metrics[arm_name]["hard_rows"],
            "positive_probability": binding_metrics[arm_name]["positive_probability"],
            "binding_gradient_norm": norm,
            "safety_gradient_norm": safety_norm,
            "unweighted_binding_safety_ratio": norm / safety_norm,
            "eta": eta,
            "eta_weighted_binding_safety_ratio": eta * norm / safety_norm,
            "binding_safety_cosine": common.gradient_cosine(gradients, safety_gradients),
            "binding_energy": common.gradient_energy(gradients),
            "finite_nonzero": math.isfinite(norm) and norm > 0,
        }
    a = binding_gradients["C3a_multi_positive_global_binding"]
    b = binding_gradients["C3b_hierarchical_hard_negative_binding"]
    direct = {
        "c3a_c3b_gradient_cosine": common.gradient_cosine(a, b),
        "c3a_energy": common.gradient_energy(a),
        "c3b_energy": common.gradient_energy(b),
    }
    leakage = [
        name for name, parameter in model.named_parameters()
        if "lora_" not in name and parameter.grad is not None and float(parameter.grad.detach().float().abs().max()) > 0
    ]
    objective_test = {
        "registered": 0.8 * 2.0 + 0.2 * (3.0 + 0.1 * 5.0),
        "implemented": float(common.weighted_total_loss(2.0, 3.0, 5.0, 0.2, 0.1)),
        "incorrect_outside_parentheses": 0.8 * 2.0 + 0.2 * 3.0 + 0.1 * 5.0,
    }
    rng_test = actual_rng_isolation_test(
        model, tokenizer, head, cfg, risk_row, safety_row, positives, queue
    )
    projection_after = common.module_state_hash(head)
    free, total = torch.cuda.mem_get_info(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    headroom = 1.0 - peak_reserved / int(total)
    numerical = {
        "prompt_id_hash": common.stable_hash(prompt_id),
        "tokenizer_padding_side": tokenizer.padding_side,
        "response_mask_has_prompt_and_target": (
            any(value == -100 for value in safety_feature["labels"])
            and any(value != -100 for value in safety_feature["labels"])
        ),
        "trainable_parameter_inventory": inventory,
        "projection_hash_before": projection_before,
        "projection_hash_after": projection_after,
        "projection_unchanged": projection_before == projection_after,
        "projection_trainable_parameters": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
        "queue_capacity": queue.capacity,
        "queue_count": queue.count,
        "queue_prefill_target_hash": common.hash_json(prefill_ids),
        "queue_hash": queue.content_hash(),
        "same_prompt_exclusion_count": len(queue.ordered(prompt_id)[1]),
        "safety_gradient_norm": safety_norm,
        "candidate_gradients": gradient_rows,
        "direct_c3a_c3b_comparison": direct,
        "frozen_base_gradient_leakage": leakage,
        "objective_parentheses": objective_test,
        "objective_parentheses_pass": objective_test["registered"] == objective_test["implemented"] and objective_test["implemented"] != objective_test["incorrect_outside_parentheses"],
        "rng_isolation": rng_test,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "free_vram_bytes": int(free),
        "total_vram_bytes": int(total),
        "vram_headroom_fraction": headroom,
        "minimum_vram_headroom_fraction": float(cfg["minimum_vram_headroom_fraction"]),
        "vram_gate_pass": headroom >= float(cfg["minimum_vram_headroom_fraction"]),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    numerical["passed"] = (
        inventory["passed"]
        and numerical["response_mask_has_prompt_and_target"]
        and numerical["projection_unchanged"]
        and numerical["projection_trainable_parameters"] == 0
        and numerical["queue_capacity"] == numerical["queue_count"] == 32
        and all(row["finite_nonzero"] for row in gradient_rows.values())
        and not leakage
        and numerical["objective_parentheses_pass"]
        and rng_test["losses_equal"]
        and numerical["vram_gate_pass"]
    )
    environment = {
        "name": "step3ct_t1_environment_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "total_vram_bytes": int(total),
        "free_vram_bytes_at_end": int(free),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    return numerical, environment


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg_path = REPO / args.config
    cfg = common.read_json(cfg_path)
    output = REPO / cfg["output_dir"] / "protocol"
    output.mkdir(parents=True, exist_ok=True)
    tests = run_implementation_tests()
    common.write_json(output / "step3ct_t1_implementation_tests.json", tests)
    if not tests["passed"]:
        raise RuntimeError("T1 implementation tests failed")
    sources = source_validation(cfg)
    if not sources["passed"]:
        common.write_json(output / "step3ct_t1_source_validation.json", sources)
        raise RuntimeError("T1 source validation failed")
    numerical, environment = numerical_preflight(cfg)
    environment["wall_clock_seconds"] = time.time() - started
    common.write_json(output / "step3ct_t1_source_validation.json", sources)
    common.write_json(output / "step3ct_t1_numerical_preflight.json", numerical)
    common.write_json(output / "step3ct_t1_environment.json", environment)

    artifact_paths = [
        Path(args.config),
        Path(cfg["registry"]),
        Path(cfg["historical_config"]),
        Path(cfg["validation_route"]),
        Path(cfg["joint_route"]),
        Path(cfg["risk_data_path"]),
        Path(cfg["safety_train_path"]),
        Path(cfg["safety_targets_path"]),
        Path("experiments/step3_response_quality_persistence/scripts/step3ct_training_common.py"),
        Path("experiments/step3_response_quality_persistence/scripts/train_step3ct_tournament.py"),
        Path("experiments/step3_response_quality_persistence/scripts/test_step3ct_training_common.py"),
        Path("experiments/step3_response_quality_persistence/scripts/run_step3ct_t1_preflight.py"),
    ]
    freeze = {
        "name": "step3ct_t1_code_freeze_v1",
        "git_head_before_t1_commit": git_head(),
        "artifacts": [
            {
                "path": path.as_posix(),
                "sha256": common.sha256_file(REPO / path),
                "bytes": (REPO / path).stat().st_size,
            }
            for path in artifact_paths
        ],
        "source_validation_pass": sources["passed"],
        "implementation_tests_pass": tests["passed"],
        "numerical_preflight_pass": numerical["passed"],
        "projection_hash": numerical["projection_hash_before"],
        "queue_policy": "stable-hash prefill; detached CPU FIFO; update after binding backward; same-prompt exclusion",
        "run_order": [
            "C0R_concurrent_constant20",
            "C3a_multi_positive_global_binding",
            "C3b_hierarchical_hard_negative_binding",
        ],
        "smoke_command_template": f"{sys.executable} experiments/step3_response_quality_persistence/scripts/train_step3ct_tournament.py --config {args.config} --arm <ARM> --run-kind smoke --max-steps 2 --output-dir <OUTPUT>",
        "main_command_template": f"{sys.executable} experiments/step3_response_quality_persistence/scripts/train_step3ct_tournament.py --config {args.config} --arm <ARM> --run-kind main --max-steps 64 --output-dir <OUTPUT>",
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "overall_pass": tests["passed"] and sources["passed"] and numerical["passed"],
    }
    common.write_json(output / "step3ct_t1_code_freeze.json", freeze)
    report = (
        "# Step3C-T T1 Implementation Freeze Report\n\n"
        f"- Overall pass: `{freeze['overall_pass']}`\n"
        f"- Implementation tests: `{tests['passed']}`\n"
        f"- Source validation: `{sources['passed']}`\n"
        f"- No-update numerical preflight: `{numerical['passed']}`\n"
        f"- GPU: `{environment['gpu']}`\n"
        f"- Peak reserved VRAM: `{environment['peak_reserved_bytes']}` bytes\n"
        f"- VRAM headroom: `{numerical['vram_headroom_fraction']:.4f}`\n"
        f"- Projection unchanged and non-trainable: `{numerical['projection_unchanged']}`\n"
        f"- Queue: `{numerical['queue_count']}/{numerical['queue_capacity']}`\n"
        f"- RNG isolation: `{numerical['rng_isolation']['losses_equal']}`\n"
        f"- Frozen-base gradient leakage: `{len(numerical['frozen_base_gradient_leakage'])}`\n"
        "- Optimizer/scheduler objects and steps: `0/0/0`\n"
        "- Core100/heldout opened: `false/false`\n\n"
        "T1 authorizes only the registered three-arm two-step smoke after this package is committed and pushed.\n"
    )
    (output / "Step3CT_T1_implementation_freeze_report.md").write_text(report, encoding="utf-8")
    if not freeze["overall_pass"]:
        raise RuntimeError("T1 no-update preflight failed")
    print(json.dumps({"overall_pass": True, "output": str(output)}))


if __name__ == "__main__":
    main()
