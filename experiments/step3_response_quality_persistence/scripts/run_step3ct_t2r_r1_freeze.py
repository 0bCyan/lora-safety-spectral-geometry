"""Freeze and validate the Step3C-T2R recovery implementation before R2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import step3ct_t2r_common as recovery
import step3ct_training_common as parent
import train_step3ct_tournament as trainer


REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1.json"
REGISTRY = "experiments/step3_response_quality_persistence/configs/step3ct_t2r_calibration_registry.json"
SOURCE_COMMIT = "637b3e7ae50c2dc07b42f81dda40eb342065a87a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def run_tests() -> dict[str, Any]:
    scripts = [
        "experiments/step3_response_quality_persistence/scripts/step3ct_t2r_common.py",
        "experiments/step3_response_quality_persistence/scripts/test_step3ct_t2r_common.py",
        "experiments/step3_response_quality_persistence/scripts/run_step3ct_t2r_calibration.py",
        "experiments/step3_response_quality_persistence/scripts/train_step3ct_t2r_smoke.py",
        "experiments/step3_response_quality_persistence/scripts/analyze_step3ct_t2r_smoke.py",
        "experiments/step3_response_quality_persistence/scripts/run_step3ct_t2r_r1_freeze.py",
    ]
    commands = [
        [sys.executable, "-m", "py_compile", *scripts],
        [
            sys.executable, "-m", "unittest", "discover", "-s",
            "experiments/step3_response_quality_persistence/scripts",
            "-p", "test_step3ct_t2r_common.py", "-v",
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
    calibration_source = (REPO / scripts[2]).read_text(encoding="utf-8")
    tree = ast.parse(calibration_source)
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func).lower()
            if "optimizer" in function or "scheduler" in function:
                forbidden_calls.append(function)
    old_trace_reads = [
        line for line in calibration_source.splitlines()
        if "runs/smoke" in line or "training_trace.csv" in line
    ]
    return {
        "name": "step3ct_t2r_r1_implementation_tests_v1",
        "commands": rows,
        "calibration_forbidden_optimizer_scheduler_calls": forbidden_calls,
        "calibration_old_failed_trace_reads": old_trace_reads,
        "coverage": [
            "syntax and imports",
            "metadata queue and hard-active coverage",
            "fit-excluded-position isolation",
            "18 consecutive four-position windows",
            "CPU float64 accumulation and serialization invariance",
            "fp16 coefficient removal",
            "direction cosine and orthogonal residual",
            "zero calibration optimizer/scheduler construction",
            "no old failed-smoke trace read",
        ],
        "passed": (
            all(row["returncode"] == 0 for row in rows)
            and not forbidden_calls
            and not old_trace_reads
        ),
    }


def git_blob_sha256(commit: str, path: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)
    return hashlib.sha256(data).hexdigest()


def old_artifact_immutability() -> dict[str, Any]:
    paths = [
        DEFAULT_CONFIG,
        "experiments/step3_response_quality_persistence/scripts/step3ct_training_common.py",
        "experiments/step3_response_quality_persistence/scripts/train_step3ct_tournament.py",
        "experiments/step3_response_quality_persistence/scripts/analyze_step3ct_smoke.py",
        "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/protocol/step3ct_t1_code_freeze.json",
        "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/smoke/step3ct_t2_smoke_gate_summary.json",
        "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/smoke/step3ct_t2_execution_record.json",
    ]
    rows = []
    for path in paths:
        source_hash = git_blob_sha256(SOURCE_COMMIT, path)
        index_data = subprocess.check_output(["git", "show", f":{path}"], cwd=REPO)
        current_hash = hashlib.sha256(index_data).hexdigest()
        worktree_clean = subprocess.run(
            ["git", "diff", "--quiet", "--", path], cwd=REPO, check=False
        ).returncode == 0
        rows.append(
            {
                "path": path,
                "source_commit_sha256": source_hash,
                "current_index_sha256": current_hash,
                "worktree_clean": worktree_clean,
                "unchanged": source_hash == current_hash and worktree_clean,
            }
        )
    return {"source_commit": SOURCE_COMMIT, "artifacts": rows, "passed": all(row["unchanged"] for row in rows)}


def source_validation(cfg: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    parent_registry = parent.read_json(REPO / registry["parent_registry"])
    failed = parent.read_json(REPO / registry["failed_smoke_report"])
    immutable = registry["immutable_contract"]
    checks = {
        "source_commit_matches": registry["source_commit"] == SOURCE_COMMIT,
        "failed_smoke_stopped": not failed["overall_pass"] and failed["next_action"] == "stop_before_T3_and_report_failed_smoke",
        "model": cfg["model_name_or_path"] == immutable["model_name_or_path"],
        "seed": cfg["seed"] == immutable["seed"],
        "optimizer_steps": cfg["optimizer_steps"] == immutable["optimizer_steps"],
        "gradient_accumulation": cfg["gradient_accumulation_steps"] == immutable["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"] == immutable["learning_rate"],
        "safety_mix": cfg["safety_mix"] == immutable["safety_mix"],
        "projection": (
            cfg["projection_dim"] == immutable["projection_dim"]
            and cfg["projection_seed"] == immutable["projection_seed"]
        ),
        "queue": cfg["queue_size"] == immutable["queue_size"],
        "temperature": cfg["temperature"] == immutable["temperature"],
        "arms": list(cfg["arms"]) == immutable["arms"],
        "parent_heldout_closed": not parent_registry["heldout_opened"],
        "behavior_evidence_absent": not registry["behavior_evidence_available"],
    }
    return {
        "name": "step3ct_t2r_r1_source_validation_v1",
        "checks": checks,
        "optimizer_steps_before_recovery": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "passed": all(checks.values()),
    }


def rng_digest() -> dict[str, str]:
    import torch

    return {
        "cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda": hashlib.sha256(b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())).hexdigest(),
    }


def set_rng(state: dict[str, Any]) -> None:
    import random
    import torch

    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def numerical_repeatability(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    import run_step3ct_t2r_calibration as calibration

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    first_model, first_tokenizer = trainer.load_model(cfg, None)
    first_device = parent.model_device(first_model)
    first_lora_hash = recovery.lora_state_hash(first_model)
    first_head = parent.projection_head(
        int(first_model.config.hidden_size), int(cfg["projection_dim"]), int(cfg["projection_seed"])
    ).to(first_device)
    first_projection_hash = parent.module_state_hash(first_head)
    del first_head, first_model, first_tokenizer
    torch.cuda.empty_cache()

    model, tokenizer = trainer.load_model(cfg, None)
    device = parent.model_device(model)
    second_lora_hash = recovery.lora_state_hash(model)
    head = parent.projection_head(
        int(model.config.hidden_size), int(cfg["projection_dim"]), int(cfg["projection_seed"])
    ).to(device)
    second_projection_hash = parent.module_state_hash(head)
    targets = parent.read_jsonl(REPO / cfg["safety_targets_path"])
    queue, _ = parent.prefill_queue(
        model, tokenizer, head, targets, int(cfg["queue_size"]),
        int(cfg["queue_prefill_seed"]), int(cfg["max_length"]),
    )
    risk = parent.read_jsonl(REPO / cfg["risk_data_path"], limit=1)[0]
    safety = targets[0]
    risk_feature = parent.response_feature(tokenizer, risk["prompt"], risk["response"], int(cfg["max_length"]))
    safety_feature = parent.response_feature(tokenizer, safety["prompt"], safety["response"], int(cfg["max_length"]))
    initial_rng = trainer.rng_state()
    risk_a = calibration.risk_forward(model, risk_feature)
    _, safety_a = calibration.unweighted_safety_gradient(model, safety_feature, 0.05, 65536.0)
    rng_after_calibration = rng_digest()

    set_rng(initial_rng)
    model.zero_grad(set_to_none=True)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        risk_output = model(**parent.batch_feature(risk_feature, device), use_cache=False, return_dict=True)
    (risk_output.loss * 0.20 * 65536.0).backward()
    with torch.autocast("cuda", dtype=torch.float16):
        safety_output = model(**parent.batch_feature(safety_feature, device), use_cache=False, return_dict=True)
    (safety_output.loss * 0.05 * 65536.0).backward()
    risk_b = float(risk_output.loss.detach().float().cpu())
    safety_b = float(safety_output.loss.detach().float().cpu())
    rng_after_trainer = rng_digest()
    model.zero_grad(set_to_none=True)
    projection_after = parent.module_state_hash(head)
    lora_after = recovery.lora_state_hash(model)
    free, total = torch.cuda.mem_get_info(device)
    result = {
        "name": "step3ct_t2r_r1_numerical_repeatability_v1",
        "fresh_lora_hash_first": first_lora_hash,
        "fresh_lora_hash_second": second_lora_hash,
        "fresh_lora_repeatable": first_lora_hash == second_lora_hash,
        "projection_hash_first": first_projection_hash,
        "projection_hash_second": second_projection_hash,
        "projection_repeatable": first_projection_hash == second_projection_hash,
        "calibration_risk_safety_losses": [risk_a, safety_a],
        "trainer_risk_safety_losses": [risk_b, safety_b],
        "forward_losses_equal": max(abs(risk_a - risk_b), abs(safety_a - safety_b)) <= 1e-6,
        "calibration_rng_after": rng_after_calibration,
        "trainer_rng_after": rng_after_trainer,
        "rng_trace_equal": rng_after_calibration == rng_after_trainer,
        "lora_unchanged": second_lora_hash == lora_after,
        "projection_unchanged": second_projection_hash == projection_after,
        "queue_prefill_count": queue.count,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }
    result["passed"] = all(
        result[key] for key in [
            "fresh_lora_repeatable", "projection_repeatable", "forward_losses_equal",
            "rng_trace_equal", "lora_unchanged", "projection_unchanged",
        ]
    ) and result["queue_prefill_count"] == 32
    environment = {
        "name": "step3ct_t2r_r1_environment_v1",
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
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }
    return result, environment


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg = parent.read_json(REPO / args.config)
    registry = parent.read_json(REPO / REGISTRY)
    output = REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1_t2r/protocol"
    output.mkdir(parents=True, exist_ok=True)
    tests = run_tests()
    source = source_validation(cfg, registry)
    immutability = old_artifact_immutability()
    metadata = recovery.metadata_hard_coverage(REPO, cfg)
    numerical, environment = numerical_repeatability(cfg)
    environment["wall_clock_seconds"] = time.time() - started
    parent.write_json(output / "step3ct_t2r_r1_implementation_tests.json", tests)
    parent.write_json(output / "step3ct_t2r_r1_source_validation.json", source)
    parent.write_json(output / "step3ct_t2r_r1_old_artifact_immutability.json", immutability)
    parent.write_json(output / "step3ct_t2r_r1_metadata_audit.json", metadata)
    parent.write_json(output / "step3ct_t2r_r1_numerical_repeatability.json", numerical)
    parent.write_json(output / "step3ct_t2r_r1_environment.json", environment)
    parent.write_json(output / "step3ct_t2r_output_schema.json", recovery.scalar_schema())

    code_paths = [
        "experiments/step3_response_quality_persistence/scripts/step3ct_t2r_common.py",
        "experiments/step3_response_quality_persistence/scripts/test_step3ct_t2r_common.py",
        "experiments/step3_response_quality_persistence/scripts/run_step3ct_t2r_calibration.py",
        "experiments/step3_response_quality_persistence/scripts/train_step3ct_t2r_smoke.py",
        "experiments/step3_response_quality_persistence/scripts/analyze_step3ct_t2r_smoke.py",
        "experiments/step3_response_quality_persistence/scripts/run_step3ct_t2r_r1_freeze.py",
        args.config,
        REGISTRY,
    ]
    overall = all([
        tests["passed"], source["passed"], immutability["passed"],
        metadata["passed"], numerical["passed"],
    ])
    freeze = {
        "name": "step3ct_t2r_r1_implementation_freeze_v1",
        "git_head_before_r1_commit": git_head(),
        "artifacts": [
            {
                "path": path,
                "sha256": parent.sha256_file(REPO / path),
                "bytes": (REPO / path).stat().st_size,
            }
            for path in code_paths
        ],
        "metadata_queue_trace_hash": parent.hash_json(metadata["positions"]),
        "output_schema_hash": parent.sha256_file(output / "step3ct_t2r_output_schema.json"),
        "r2_exact_command": f"{sys.executable} experiments/step3_response_quality_persistence/scripts/run_step3ct_t2r_calibration.py --config {args.config}",
        "r3_smoke_command_template": f"{sys.executable} experiments/step3_response_quality_persistence/scripts/train_step3ct_t2r_smoke.py --config experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1_t2r.json --arm <ARM> --output-dir <NEW_OUTPUT>",
        "r3_analysis_command": f"{sys.executable} experiments/step3_response_quality_persistence/scripts/analyze_step3ct_t2r_smoke.py --config experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1_t2r.json",
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "overall_pass": overall,
    }
    parent.write_json(output / "step3ct_t2r_r1_implementation_freeze.json", freeze)
    report = (
        "# Step3C-T2R R1 Implementation Freeze Report\n\n"
        f"- Overall pass: `{overall}`\n"
        f"- Implementation tests: `{tests['passed']}`\n"
        f"- Source and parent contract: `{source['passed']}`\n"
        f"- Old T1/T2 artifacts unchanged: `{immutability['passed']}`\n"
        f"- Metadata hard-coverage audit: `{metadata['passed']}`\n"
        f"- Fresh LoRA/projection and RNG repeatability: `{numerical['passed']}`\n"
        f"- GPU: `{environment['gpu']}`\n"
        "- Optimizer/scheduler objects and optimizer steps: `0/0/0`\n"
        "- Candidate generation/core100/heldout: `false/false/false`\n\n"
        "R1 authorizes one R2 numerical calibration only after this package is committed and pushed.\n"
    )
    (output / "Step3CT_T2R_R1_implementation_freeze_report.md").write_text(report, encoding="utf-8")
    if not overall:
        raise RuntimeError("R1 implementation freeze failed")
    print(json.dumps({"overall_pass": overall, "output": str(output)}))


if __name__ == "__main__":
    main()
