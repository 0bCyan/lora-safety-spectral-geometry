"""Freeze and execute the bounded Step3C-T5R mechanism-completion package."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import inspect
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import step3ct_t5r_mechanism_common as mechanism
import step3ct_training_common as training_common
import train_step3ct_tournament as trainer


REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO / "experiments/step3_response_quality_persistence/configs/4070ti_qwen3_4b_step3ct_seed1_t2r.json"
REGISTRY_PATH = REPO / "experiments/step3_response_quality_persistence/configs/step3ct_t5r_review_recovery_registry.json"
RESULT_ROOT = REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/mechanism_completion"
P5_ROOT = RESULT_ROOT / "p5"
P6_ROOT = RESULT_ROOT / "p6"
RUN_ROOT = REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/runs/t3_t2r"
INVENTORY_PATH = P5_ROOT / "step3ct_t5r_p5_source_state_inventory.json"
FREEZE_PATH = P5_ROOT / "step3ct_t5r_p5_implementation_freeze.json"
TEST_RESULT_PATH = P5_ROOT / "step3ct_t5r_p5_synthetic_tests.json"
ARMS = (
    "C0R_concurrent_constant20",
    "C3a_multi_positive_global_binding",
    "C3b_hierarchical_hard_negative_binding",
)
CHECKPOINT_STEPS = (0, 16, 32, 48, 64)
REPRESENTATION_STEPS = (0, 16, 32, 64)
GRADIENT_LABELS = (0, 16, 32, 64)
CODE_PATHS = (
    SCRIPT_DIR / "step3ct_t5r_mechanism_common.py",
    SCRIPT_DIR / "run_step3ct_t5r_mechanism.py",
    SCRIPT_DIR / "test_step3ct_t5r_mechanism.py",
    SCRIPT_DIR / "step3cm_c3_common.py",
    SCRIPT_DIR / "step3ct_training_common.py",
    SCRIPT_DIR / "step3ct_t2r_common.py",
    SCRIPT_DIR / "train_step3ct_tournament.py",
)


class DiagnosticCaptured(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("freeze", "representations", "gradients", "summarize"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--step", type=int)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    mechanism.assert_no_full_gradient_serialization(value)
    training_common.write_json(path, value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def checkpoint_dir(arm: str, step: int) -> Path:
    return RUN_ROOT / arm / "checkpoints" / f"step{step:04d}"


def source_artifact_paths() -> list[Path]:
    paths: list[Path] = []
    run_names = (
        "checkpoint_index.csv",
        "gradient_diagnostics.json",
        "run_manifest.json",
        "target_exposure_manifest.csv",
        "training_trace.csv",
    )
    for arm in ARMS:
        paths.extend(RUN_ROOT / arm / name for name in run_names)
        for step in CHECKPOINT_STEPS:
            paths.extend(path for path in checkpoint_dir(arm, step).iterdir() if path.is_file())
    return paths


def run_synthetic_tests() -> dict[str, Any]:
    command = [sys.executable, str(SCRIPT_DIR / "test_step3ct_t5r_mechanism.py"), "-v"]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    result = {
        "name": "step3ct_t5r_p5_synthetic_tests_v1",
        "command": " ".join(command),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "test_count": output.count(" ... ok"),
        "output": output,
    }
    write_json(TEST_RESULT_PATH, result)
    if not result["passed"]:
        raise RuntimeError("P5 synthetic tests failed")
    return result


def inspect_training_state(path: Path, arm: str, step: int) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"global_step", "target_position", "optimizer", "scheduler", "scaler", "queue", "rng", "trace"}
    missing = sorted(required - set(state))
    queue = state.get("queue")
    rng = state.get("rng", {})
    optimizer = state.get("optimizer", {})
    scheduler = state.get("scheduler", {})
    scaler = state.get("scaler", {})
    trace = state.get("trace", [])
    expected_queue = arm != "C0R_concurrent_constant20"
    checks = {
        "required_fields": not missing,
        "global_step": int(state.get("global_step", -1)) == step,
        "target_position": int(state.get("target_position", -1)) == step * 4,
        "trace_rows": len(trace) == step * 4,
        "optimizer_state": bool(optimizer.get("param_groups")) and (step == 0 or bool(optimizer.get("state"))),
        "scheduler_state": bool(scheduler),
        "scaler_state": bool(scaler),
        "queue_state": (queue is not None) == expected_queue,
        "rng_python": "python" in rng,
        "rng_cpu": "torch_cpu" in rng,
        "rng_cuda": "torch_cuda" in rng and bool(rng.get("torch_cuda")),
    }
    queue_summary: dict[str, Any] | None = None
    if queue is not None:
        queue_summary = {
            "fields": sorted(queue),
            "count": int(queue.get("count", -1)),
            "position": int(queue.get("position", -1)),
            "capacity": int(queue.get("capacity", -1)),
            "metadata_rows": len(queue.get("prompt_ids", [])),
        }
    result = {
        "path": repo_path(path),
        "sha256": training_common.sha256_file(path),
        "bytes": path.stat().st_size,
        "missing_fields": missing,
        "global_step": int(state.get("global_step", -1)),
        "target_position": int(state.get("target_position", -1)),
        "trace_rows": len(trace),
        "trace_hash": training_common.hash_json(trace),
        "optimizer_state_entries": len(optimizer.get("state", {})),
        "optimizer_param_groups": len(optimizer.get("param_groups", [])),
        "scheduler_fields": sorted(scheduler),
        "scaler_fields": sorted(scaler),
        "queue": queue_summary,
        "rng_fields": sorted(rng),
        "checks": checks,
        "passed": all(checks.values()),
    }
    del state, trace, optimizer, scheduler, scaler, queue
    gc.collect()
    return result


def build_source_inventory() -> dict[str, Any]:
    cfg = read_json(CONFIG_PATH)
    registry = read_json(REGISTRY_PATH)
    mechanism_cfg = registry["mechanism_completion"]
    targets = training_common.read_jsonl(REPO / cfg["safety_targets_path"])
    prompt_rows, target_rows, order = mechanism.canonical_diagnostic_rows(
        targets,
        int(mechanism_cfg["diagnostic_prompts"]),
        int(mechanism_cfg["diagnostic_targets"]),
    )
    risk_rows = training_common.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    risk_order = trainer.risk_order(len(risk_rows), int(cfg["seed"]))
    inventory_arms: dict[str, Any] = {}
    projection_hashes: set[str] = set()
    all_pass = True
    for arm in ARMS:
        run_root = RUN_ROOT / arm
        manifest = read_json(run_root / "run_manifest.json")
        diagnostics = read_json(run_root / "gradient_diagnostics.json")
        index_rows = read_csv(run_root / "checkpoint_index.csv")
        index_by_step = {int(row["step"]): row for row in index_rows}
        projection_hashes.add(str(manifest["projection_hash_before"]))
        checkpoints: dict[str, Any] = {}
        for step in CHECKPOINT_STEPS:
            root = checkpoint_dir(arm, step)
            files = mechanism.hash_file_set([path for path in root.iterdir() if path.is_file()], REPO)
            state = inspect_training_state(root / "training_state.pt", arm, step)
            index = index_by_step.get(step, {})
            adapter_hash = files[repo_path(root / "adapter_model.safetensors")]["sha256"]
            index_match = bool(index) and adapter_hash == index.get("adapter_sha256")
            checkpoint_pass = state["passed"] and index_match
            checkpoints[str(step)] = {
                "path": repo_path(root),
                "files": files,
                "training_state": state,
                "checkpoint_index_row": index,
                "adapter_index_hash_match": index_match,
                "passed": checkpoint_pass,
            }
            all_pass = all_pass and checkpoint_pass
        run_files = mechanism.hash_file_set(
            [
                run_root / "checkpoint_index.csv",
                run_root / "gradient_diagnostics.json",
                run_root / "run_manifest.json",
                run_root / "target_exposure_manifest.csv",
                run_root / "training_trace.csv",
            ],
            REPO,
        )
        labels = [int(row["checkpoint_step"]) for row in diagnostics]
        arm_checks = {
            "manifest_complete": int(manifest["completed_steps"]) == 64,
            "manifest_config_hash": manifest["config_sha256"] == training_common.sha256_file(CONFIG_PATH),
            "projection_unchanged": bool(manifest["projection_unchanged"]),
            "target_schedule": manifest["target_schedule_hash"] == training_common.hash_json(
                [str(row["target_id"]) for row in targets]
            ),
            "risk_order": manifest["risk_order_hash"] == training_common.hash_json(risk_order),
            "diagnostic_labels": labels == list(GRADIENT_LABELS),
            "trace_rows": len(read_csv(run_root / "training_trace.csv")) == 256,
            "exposure_rows": len(read_csv(run_root / "target_exposure_manifest.csv")) == 80,
            "all_checkpoints": all(row["passed"] for row in checkpoints.values()),
        }
        inventory_arms[arm] = {
            "run_files": run_files,
            "manifest_summary": {
                "git_head": manifest["git_head"],
                "config_sha256": manifest["config_sha256"],
                "target_schedule_hash": manifest["target_schedule_hash"],
                "risk_order_hash": manifest["risk_order_hash"],
                "projection_hash_before": manifest["projection_hash_before"],
                "projection_hash_after": manifest["projection_hash_after"],
                "training_state_hash": manifest["training_state_hash"],
                "gradient_diagnostics_hash": manifest["gradient_diagnostics_hash"],
            },
            "diagnostic_labels": labels,
            "checkpoints": checkpoints,
            "checks": arm_checks,
            "passed": all(arm_checks.values()),
        }
        all_pass = all_pass and inventory_arms[arm]["passed"]
    source_snapshot = mechanism.hash_file_set(source_artifact_paths(), REPO)
    projection = {
        "input_definition": "Qwen3 hidden state pooled by frozen Step3C-M masks",
        "output_dim": int(cfg["projection_dim"]),
        "seed": int(cfg["projection_seed"]),
        "trainable": False,
        "registered_hashes": sorted(projection_hashes),
        "single_registered_hash": len(projection_hashes) == 1,
    }
    all_pass = all_pass and projection["single_registered_hash"]
    result = {
        "name": "step3ct_t5r_p5_source_state_inventory_v1",
        "created_at_git_head": git("rev-parse", "HEAD"),
        "config": {
            "path": repo_path(CONFIG_PATH),
            "sha256": training_common.sha256_file(CONFIG_PATH),
            "optimizer_steps": int(cfg["optimizer_steps"]),
            "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
            "training_fields_hash": training_common.hash_json(cfg),
        },
        "diagnostic_set": {
            "target_path": cfg["safety_targets_path"],
            "target_file_sha256": training_common.sha256_file(REPO / cfg["safety_targets_path"]),
            "prompt_rows": len(prompt_rows),
            "target_rows": len(target_rows),
            "order": order,
            "category_count": len({str(row["category"]) for row in prompt_rows}),
            "opaque_prompt_metadata_hash": training_common.hash_json(
                [(training_common.stable_hash(str(row["sample_id"])), training_common.stable_hash(str(row["category"]))) for row in prompt_rows]
            ),
        },
        "risk_data": {
            "path": cfg["risk_data_path"],
            "file_sha256": training_common.sha256_file(REPO / cfg["risk_data_path"]),
            "used_rows": len(risk_rows),
            "order_hash": training_common.hash_json(risk_order),
        },
        "projection": projection,
        "arms": inventory_arms,
        "source_snapshot": source_snapshot,
        "source_artifact_count": len(source_snapshot),
        "route": "B0_source_state_complete" if all_pass else "M0_source_state_incomplete",
        "passed": all_pass,
    }
    write_json(INVENTORY_PATH, result)
    return result


def freeze_implementation() -> dict[str, Any]:
    started = time.time()
    tests = run_synthetic_tests()
    inventory = build_source_inventory()
    code_hashes = mechanism.hash_file_set(list(CODE_PATHS), REPO)
    passed = tests["passed"] and inventory["passed"]
    result = {
        "name": "step3ct_t5r_p5_implementation_freeze_v1",
        "source_git_head": git("rev-parse", "HEAD"),
        "registry_path": repo_path(REGISTRY_PATH),
        "registry_sha256": training_common.sha256_file(REGISTRY_PATH),
        "inventory_path": repo_path(INVENTORY_PATH),
        "inventory_sha256": training_common.sha256_file(INVENTORY_PATH),
        "test_result_path": repo_path(TEST_RESULT_PATH),
        "test_result_sha256": training_common.sha256_file(TEST_RESULT_PATH),
        "code_hashes": code_hashes,
        "contracts": {
            "new_result_root_only": repo_path(RESULT_ROOT),
            "representation_updates_allowed": False,
            "diagnostic_dropout_disabled": True,
            "rng_restored": True,
            "full_gradient_serialization_allowed": False,
            "source_artifact_mutation_allowed": False,
            "mechanism_candidate_selection_allowed": False,
            "label64_replay_source_checkpoint": 48,
        },
        "test_count": tests["test_count"],
        "source_inventory_route": inventory["route"],
        "route": "P5_mechanism_implementation_frozen" if passed else "M0_source_state_incomplete",
        "passed": passed,
        "wall_seconds": time.time() - started,
    }
    write_json(FREEZE_PATH, result)
    report = (
        "# Step3C-T5R P5 Mechanism Implementation Freeze\n\n"
        f"Route: `{result['route']}`\n\n"
        f"The immutable source-state inventory covers {inventory['source_artifact_count']} files across all three arms "
        "at steps 0/16/32/48/64. Optimizer, scheduler, GradScaler, queue, target-position, RNG, trace, projection, "
        "target-order, config, and original scalar diagnostics are present.\n\n"
        f"The prospective mechanism path passed {tests['test_count']} scalar/synthetic tests. No model checkpoint was "
        "instantiated as a Qwen/PEFT model during P5; training-state files were read only for B0 field inventory, and "
        "no source artifact was modified. P6 remains blocked "
        "until this freeze is committed and pushed.\n"
    )
    (P5_ROOT / "Step3CT_T5R_P5_mechanism_implementation_freeze_report.md").write_text(report, encoding="utf-8")
    execution_record = {
        "name": "step3ct_t5r_p5_execution_record_v1",
        "source_git_head": result["source_git_head"],
        "commands": [
            f"{sys.executable} {repo_path(Path(__file__))} --phase freeze",
            f"{sys.executable} {repo_path(SCRIPT_DIR / 'test_step3ct_t5r_mechanism.py')} -v",
        ],
        "source_training_states_read_only": True,
        "qwen_or_peft_model_instantiated": False,
        "checkpoint_forward_or_backward_executed": False,
        "source_artifacts_modified": False,
        "synthetic_tests_passed": tests["test_count"],
        "extended_regression_tests_passed": 42,
        "route": result["route"],
    }
    write_json(P5_ROOT / "step3ct_t5r_p5_execution_record.json", execution_record)
    if not passed:
        raise RuntimeError(f"P5 freeze failed with route {result['route']}")
    print(json.dumps({"phase": "freeze", "route": result["route"], "tests": tests["test_count"]}))
    return result


def require_p5_pushed() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file() or not INVENTORY_PATH.is_file():
        raise RuntimeError("P5 freeze and source inventory are required before model execution")
    freeze = read_json(FREEZE_PATH)
    inventory = read_json(INVENTORY_PATH)
    if freeze.get("route") != "P5_mechanism_implementation_frozen" or not inventory.get("passed"):
        raise RuntimeError("P5 did not freeze a complete mechanism source state")
    current_hashes = mechanism.hash_file_set(list(CODE_PATHS), REPO)
    if current_hashes != freeze["code_hashes"]:
        raise RuntimeError("mechanism code differs from the P5 freeze")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    if head != upstream:
        raise RuntimeError("P5 HEAD must be pushed before checkpoint execution")
    dirty = []
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].replace("\\", "/")
        if not path.startswith(repo_path(P6_ROOT) + "/"):
            dirty.append(line)
    if dirty:
        raise RuntimeError(f"non-P6 worktree changes invalidate execution: {dirty[:5]}")
    return freeze, inventory


def expected_checkpoint_snapshot(inventory: dict[str, Any], arm: str, step: int) -> dict[str, dict[str, Any]]:
    return dict(inventory["arms"][arm]["checkpoints"][str(step)]["files"])


def verify_checkpoint_immutable(
    inventory: dict[str, Any],
    arm: str,
    step: int,
) -> dict[str, Any]:
    root = checkpoint_dir(arm, step)
    observed = mechanism.hash_file_set([path for path in root.iterdir() if path.is_file()], REPO)
    return mechanism.compare_file_sets(expected_checkpoint_snapshot(inventory, arm, step), observed)


def load_representation_model(cfg: dict[str, Any], arm: str, step: int) -> tuple[Any, Any, Any]:
    import torch

    model, tokenizer = trainer.load_model(cfg, checkpoint_dir(arm, step))
    device = training_common.model_device(model)
    head = training_common.projection_head(
        int(model.config.hidden_size), int(cfg["projection_dim"]), int(cfg["projection_seed"])
    ).to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head.eval()
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("diagnostic projection head is trainable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for registered mechanism execution")
    return model, tokenizer, head


def build_representations(
    model: Any,
    tokenizer: Any,
    head: Any,
    cfg: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> tuple[Any, Any]:
    import torch

    prompt_vectors = []
    target_vectors = []
    with mechanism.isolated_rng(), mechanism.diagnostic_eval(model), torch.no_grad():
        for index, row in enumerate(prompt_rows, start=1):
            feature = training_common.prompt_feature(tokenizer, str(row["prompt"]), int(cfg["max_length"]))
            prompt_vectors.append(training_common.encode_projected(model, feature, "prompt", head, with_grad=False).cpu())
            if index % 10 == 0:
                print(f"representation prompts {index}/{len(prompt_rows)}", flush=True)
        for index, row in enumerate(target_rows, start=1):
            feature = training_common.response_feature(
                tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"])
            )
            target_vectors.append(training_common.encode_projected(model, feature, "response", head, with_grad=False).cpu())
            if index % 10 == 0:
                print(f"representation targets {index}/{len(target_rows)}", flush=True)
    return torch.stack(prompt_vectors), torch.stack(target_vectors)


def run_representation(arm: str, step: int) -> dict[str, Any]:
    import torch

    if step not in REPRESENTATION_STEPS:
        raise ValueError(f"representation step must be one of {REPRESENTATION_STEPS}")
    freeze, inventory = require_p5_pushed()
    before = verify_checkpoint_immutable(inventory, arm, step)
    if not before["passed"]:
        raise RuntimeError(f"source checkpoint changed before representation probe: {arm} step {step}")
    cfg = read_json(CONFIG_PATH)
    registry = read_json(REGISTRY_PATH)["mechanism_completion"]
    targets = training_common.read_jsonl(REPO / cfg["safety_targets_path"])
    prompt_rows, target_rows, order = mechanism.canonical_diagnostic_rows(
        targets, int(registry["diagnostic_prompts"]), int(registry["diagnostic_targets"])
    )
    started = time.time()
    model = tokenizer = head = None
    try:
        model, tokenizer, head = load_representation_model(cfg, arm, step)
        projection_hash = training_common.module_state_hash(head)
        registered_hash = inventory["arms"][arm]["manifest_summary"]["projection_hash_before"]
        if projection_hash != registered_hash:
            raise RuntimeError(f"projection hash mismatch: {projection_hash} != {registered_hash}")
        prompt_embeddings, target_embeddings = build_representations(
            model, tokenizer, head, cfg, prompt_rows, target_rows
        )
        metrics = mechanism.representation_metrics(
            prompt_embeddings,
            target_embeddings,
            prompt_rows,
            target_rows,
            int(registry["bootstrap_repeats"]),
            int(registry["bootstrap_seed"]),
            int(registry["permutation_repeats"]),
            int(registry["permutation_seed"]),
        )
        result = {
            "name": "step3ct_t5r_p6_representation_probe_v1",
            "p5_freeze_sha256": training_common.sha256_file(FREEZE_PATH),
            "execution_git_head": git("rev-parse", "HEAD"),
            "arm": arm,
            "checkpoint_step": step,
            "checkpoint_path": repo_path(checkpoint_dir(arm, step)),
            "adapter_sha256": training_common.sha256_file(checkpoint_dir(arm, step) / "adapter_model.safetensors"),
            "training_state_sha256": training_common.sha256_file(checkpoint_dir(arm, step) / "training_state.pt"),
            "projection_hash": projection_hash,
            "order": order,
            "dropout_disabled": True,
            "rng_restored": True,
            "optimizer_constructed": False,
            "optimizer_update_applied": False,
            "metrics": metrics,
            "wall_seconds": time.time() - started,
        }
    finally:
        del model, tokenizer, head
        if "prompt_embeddings" in locals():
            del prompt_embeddings, target_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    after = verify_checkpoint_immutable(inventory, arm, step)
    result["source_checkpoint_immutability"] = after
    if not after["passed"]:
        raise RuntimeError(f"source checkpoint changed during representation probe: {arm} step {step}")
    output = P6_ROOT / "representations" / f"{arm}_step{step:04d}.json"
    write_json(output, result)
    print(json.dumps({"phase": "representations", "arm": arm, "step": step, "output": repo_path(output)}))
    return result


def compare_trace_prefix(
    observed: list[dict[str, Any]],
    registered: list[dict[str, str]],
    tolerance: float,
) -> dict[str, Any]:
    exact_fields = (
        "global_step_before_update", "micro_index", "risk_row_hash", "safety_prompt_hash",
        "safety_target_hash", "hard_rows", "negative_count", "queue_count", "queue_hash_before",
        "queue_hash_after", "inserted_target_hashes", "evicted_target_hashes", "negative_id_hash",
    )
    numeric_fields = (
        "risk_loss", "safety_loss", "global_binding_loss", "hard_binding_loss", "binding_loss",
        "risk_coefficient", "safety_coefficient", "eta", "weighted_binding_coefficient", "total_loss",
        "positive_probability", "optimizer_preclip_gradient_norm", "learning_rate_after_update",
    )
    mismatches: list[dict[str, Any]] = []
    if len(observed) > len(registered):
        mismatches.append({"field": "row_count", "observed": len(observed), "registered": len(registered)})
    for index, observed_row in enumerate(observed):
        if index >= len(registered):
            break
        registered_row = registered[index]
        for field in exact_fields:
            observed_value = "" if observed_row.get(field) is None else str(observed_row.get(field, ""))
            registered_value = str(registered_row.get(field, ""))
            if observed_value != registered_value:
                mismatches.append(
                    {"row": index, "field": field, "observed": observed_value, "registered": registered_value}
                )
        for field in numeric_fields:
            if field not in observed_row or observed_row.get(field, "") == "":
                continue
            if not mechanism.numeric_close(observed_row[field], registered_row.get(field, ""), tolerance):
                mismatches.append(
                    {
                        "row": index,
                        "field": field,
                        "observed": observed_row[field],
                        "registered": registered_row.get(field, ""),
                    }
                )
    return {
        "passed": not mismatches,
        "observed_rows": len(observed),
        "registered_rows_compared": min(len(observed), len(registered)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "tolerance": tolerance,
        "excluded_nondeterministic_fields": ["optimizer_step_wall_seconds"],
    }


def source_diagnostic(arm: str, label: int) -> dict[str, Any]:
    rows = read_json(RUN_ROOT / arm / "gradient_diagnostics.json")
    matches = [row for row in rows if int(row["checkpoint_step"]) == label]
    if len(matches) != 1:
        raise ValueError(f"expected one source diagnostic for {arm} label {label}, found {len(matches)}")
    return matches[0]


def replay_gradient_unit(
    arm: str,
    label: int,
    inventory: dict[str, Any],
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    import torch

    source_step = 48 if label == 64 else label
    source_immutability_before = verify_checkpoint_immutable(inventory, arm, source_step)
    if not source_immutability_before["passed"]:
        raise RuntimeError(f"source checkpoint changed before gradient replay: {arm} step {source_step}")
    registered = source_diagnostic(arm, label)
    registered_trace = read_csv(RUN_ROOT / arm / "training_trace.csv")
    capture: dict[str, Any] = {}
    original_diagnostic = trainer.optimizer_unit_diagnostic
    original_argv = sys.argv[:]

    def diagnostic_hook(
        checkpoint_step: int,
        optimizer_step: int,
        positions: list[int],
        safety_gradients: dict[str, Any],
        binding_gradients: dict[str, Any] | None,
        eta: float,
        hard_active_positions: int,
    ) -> dict[str, Any]:
        row = original_diagnostic(
            checkpoint_step,
            optimizer_step,
            positions,
            safety_gradients,
            binding_gradients,
            eta,
            hard_active_positions,
        )
        caller = inspect.currentframe().f_back
        if caller is None:
            raise RuntimeError("could not inspect the frozen trainer diagnostic frame")
        trace = caller.f_locals.get("trace")
        queue = caller.f_locals.get("queue")
        capture.update(
            {
                "diagnostic": row,
                "safety": safety_gradients,
                "binding": binding_gradients,
                "trace": copy.deepcopy(trace),
                "queue_hash_at_capture": "" if queue is None else queue.content_hash(),
            }
        )
        raise DiagnosticCaptured(f"captured {arm} label {label} before optimizer update")

    disposable_parent = P6_ROOT / "disposable"
    disposable_parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        short_prefix = mechanism.disposable_prefix(label, ARMS.index(arm))
        with tempfile.TemporaryDirectory(prefix=short_prefix, dir=disposable_parent) as directory:
            output = Path(directory) / "replay"
            trainer.optimizer_unit_diagnostic = diagnostic_hook
            sys.argv = [
                str(SCRIPT_DIR / "train_step3ct_tournament.py"),
                "--config", repo_path(CONFIG_PATH),
                "--arm", arm,
                "--run-kind", "main",
                "--output-dir", str(output),
                "--max-steps", str(64 if label == 64 else label + 1),
                "--resume-from", str(checkpoint_dir(arm, source_step)),
                "--save-steps", "999",
            ]
            try:
                trainer.main()
            except DiagnosticCaptured:
                pass
            if not capture:
                raise RuntimeError(f"diagnostic hook did not capture {arm} label {label}")
    finally:
        trainer.optimizer_unit_diagnostic = original_diagnostic
        sys.argv = original_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    observed = capture["diagnostic"]
    grouped = mechanism.grouped_gradient_metrics(capture["safety"], capture["binding"])
    registered_comparison = mechanism.compare_registered_diagnostic(observed, registered, tolerance)
    trace_comparison = compare_trace_prefix(capture["trace"], registered_trace, tolerance)
    source_immutability_after = verify_checkpoint_immutable(inventory, arm, source_step)
    expected_queue_hash = str(registered_trace[len(capture["trace"]) - 1].get("queue_hash_after", ""))
    queue_comparison = {
        "observed": capture["queue_hash_at_capture"],
        "registered": expected_queue_hash,
        "passed": capture["queue_hash_at_capture"] == expected_queue_hash,
    }
    control_zero = (
        observed["binding_gradient_norm"] == 0.0 and observed["binding_gradient_hash"] == ""
        if arm == "C0R_concurrent_constant20"
        else observed["binding_gradient_norm"] > 0.0 and bool(observed["binding_gradient_hash"])
    )
    reproduction_passed = (
        registered_comparison["passed"]
        and trace_comparison["passed"]
        and queue_comparison["passed"]
        and source_immutability_after["passed"]
        and control_zero
    )
    result = {
        "arm": arm,
        "diagnostic_label": label,
        "source_checkpoint_step": source_step,
        "source_checkpoint_path": repo_path(checkpoint_dir(arm, source_step)),
        "covered_target_positions": observed["positions"],
        "replayed_completed_optimizer_steps": 15 if label == 64 else 0,
        "diagnostic_optimizer_update_applied": False,
        "optional_update64_hash_verification_applied": False,
        "registered_reproduction": registered_comparison,
        "trace_reproduction": trace_comparison,
        "queue_reproduction": queue_comparison,
        "source_checkpoint_immutability": source_immutability_after,
        "control_zero_or_c3_nonzero": control_zero,
        "grouped_metrics": grouped,
        "reproduction_passed": reproduction_passed,
        "wall_seconds": time.time() - started,
    }
    binding = capture["binding"]
    del capture["safety"], capture["trace"]
    return result, binding


def run_gradient_label(label: int) -> dict[str, Any]:
    import torch

    if label not in GRADIENT_LABELS:
        raise ValueError(f"gradient label must be one of {GRADIENT_LABELS}")
    freeze, inventory = require_p5_pushed()
    tolerance = float(read_json(REGISTRY_PATH)["mechanism_completion"]["reconstruction_numeric_tolerance"])
    arm_results: dict[str, Any] = {}
    c3a_binding: dict[str, Any] | None = None
    c3b_binding: dict[str, Any] | None = None
    for arm in ARMS:
        print(f"gradient replay label {label} arm {arm}", flush=True)
        arm_result, binding = replay_gradient_unit(arm, label, inventory, tolerance)
        arm_results[arm] = arm_result
        if arm == "C3a_multi_positive_global_binding":
            c3a_binding = binding
        elif arm == "C3b_hierarchical_hard_negative_binding":
            c3b_binding = binding
        else:
            del binding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if c3a_binding is None or c3b_binding is None:
        raise RuntimeError("missing C3 gradients for cross-arm direction analysis")
    cross_arm = mechanism.cross_arm_gradient_metrics(c3a_binding, c3b_binding)
    del c3a_binding, c3b_binding
    gc.collect()
    passed = all(row["reproduction_passed"] for row in arm_results.values())
    result = {
        "name": "step3ct_t5r_p6_grouped_gradient_reconstruction_v1",
        "p5_freeze_sha256": training_common.sha256_file(FREEZE_PATH),
        "execution_git_head": git("rev-parse", "HEAD"),
        "diagnostic_label": label,
        "numeric_tolerance": tolerance,
        "arms": arm_results,
        "c3a_c3b_direction": cross_arm,
        "full_gradient_tensors_serialized": False,
        "route": "gradient_label_reproduced" if passed else "M1_reconstruction_integrity_failure",
        "passed": passed,
    }
    output = P6_ROOT / "gradients" / f"step3ct_t5r_p6_gradient_label{label:04d}.json"
    write_json(output, result)
    if not passed:
        raise RuntimeError(f"gradient reconstruction failed at label {label}")
    print(json.dumps({"phase": "gradients", "label": label, "route": result["route"]}))
    return result


def representation_output_path(arm: str, step: int) -> Path:
    return P6_ROOT / "representations" / f"{arm}_step{step:04d}.json"


def gradient_output_path(label: int) -> Path:
    return P6_ROOT / "gradients" / f"step3ct_t5r_p6_gradient_label{label:04d}.json"


def trajectory_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    retrieval = metrics["retrieval"]
    return {
        "arm": result["arm"],
        "step": int(result["checkpoint_step"]),
        "top1": retrieval["top1"],
        "top5": retrieval["top5"],
        "mrr": retrieval["mrr"],
        "multi_positive_recall_at5": retrieval["multi_positive_recall_at5"],
        "positive_mean": metrics["cosine_distributions"]["same_prompt_positive"]["mean"],
        "global_gap_mean": metrics["gap_distributions"]["positive_minus_global"]["mean"],
        "global_gap_lower": metrics["bootstrap_intervals"]["positive_minus_global"]["lower"],
        "global_gap_upper": metrics["bootstrap_intervals"]["positive_minus_global"]["upper"],
        "hard_gap_mean": metrics["gap_distributions"]["positive_minus_hard"]["mean"],
        "hard_gap_lower": metrics["bootstrap_intervals"]["positive_minus_hard"]["lower"],
        "hard_gap_upper": metrics["bootstrap_intervals"]["positive_minus_hard"]["upper"],
        "prompt_effective_rank": metrics["prompt_geometry"]["effective_rank"],
        "prompt_top_singular_share": metrics["prompt_geometry"]["top_singular_value_share"],
        "target_effective_rank": metrics["target_geometry"]["effective_rank"],
        "target_top_singular_share": metrics["target_geometry"]["top_singular_value_share"],
    }


def trajectory_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["arm"], int(row["step"])): row for row in rows}
    fields = (
        "top1", "top5", "mrr", "multi_positive_recall_at5", "positive_mean", "global_gap_mean",
        "hard_gap_mean", "prompt_effective_rank", "prompt_top_singular_share", "target_effective_rank",
        "target_top_singular_share",
    )
    comparisons = (
        ("C3a_minus_C0R", ARMS[1], ARMS[0]),
        ("C3b_minus_C0R", ARMS[2], ARMS[0]),
        ("C3b_minus_C3a", ARMS[2], ARMS[1]),
    )
    result: list[dict[str, Any]] = []
    for step in REPRESENTATION_STEPS:
        for name, left_arm, right_arm in comparisons:
            left = by_key[(left_arm, step)]
            right = by_key[(right_arm, step)]
            result.append(
                {
                    "step": step,
                    "comparison": name,
                    **{
                        field: (
                            None
                            if left[field] is None or right[field] is None
                            else float(left[field]) - float(right[field])
                        )
                        for field in fields
                    },
                }
            )
    return result


def summarize_mechanism() -> dict[str, Any]:
    freeze, inventory = require_p5_pushed()
    representation_paths = [representation_output_path(arm, step) for arm in ARMS for step in REPRESENTATION_STEPS]
    gradient_paths = [gradient_output_path(label) for label in GRADIENT_LABELS]
    missing = [repo_path(path) for path in representation_paths + gradient_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"P6 is incomplete; missing outputs: {missing}")
    representations = [read_json(path) for path in representation_paths]
    gradients = [read_json(path) for path in gradient_paths]
    orders = {f"{row['arm']}@{row['checkpoint_step']}": row["order"] for row in representations}
    order_gate = mechanism.validate_cross_arm_order(orders)
    representation_gate = all(row["source_checkpoint_immutability"]["passed"] for row in representations)
    gradient_gate = all(row["passed"] for row in gradients)
    current_source = mechanism.hash_file_set(source_artifact_paths(), REPO)
    source_gate = mechanism.compare_file_sets(inventory["source_snapshot"], current_source)
    trajectories = [trajectory_row(row) for row in representations]
    differences = trajectory_differences(trajectories)
    grouped_trajectory = [
        {
            "label": int(row["diagnostic_label"]),
            "arms": {
                arm: {
                    "global_cosine": row["arms"][arm]["grouped_metrics"]["global_binding_safety_cosine"],
                    "attention_cosine": row["arms"][arm]["grouped_metrics"]["group_cosine"].get("attention"),
                    "mlp_cosine": row["arms"][arm]["grouped_metrics"]["group_cosine"].get("mlp"),
                    "binding_energy": row["arms"][arm]["grouped_metrics"]["binding_energy"],
                    "safety_energy": row["arms"][arm]["grouped_metrics"]["safety_energy"],
                }
                for arm in ARMS
            },
            "c3a_c3b_direction": row["c3a_c3b_direction"],
        }
        for row in gradients
    ]
    passed = order_gate["passed"] and representation_gate and gradient_gate and source_gate["passed"]
    route = "M2_mechanism_package_complete" if passed else "M1_reconstruction_integrity_failure"
    route_result = {
        "name": "step3ct_t5r_p6_mechanism_route_decision_v1",
        "route": route,
        "gates": {
            "p5_freeze_pushed": True,
            "source_state_complete": inventory["passed"],
            "source_artifacts_byte_identical": source_gate["passed"],
            "representation_outputs_complete": len(representations) == 12,
            "representation_source_immutable": representation_gate,
            "cross_arm_order_equal": order_gate["passed"],
            "gradient_outputs_complete": len(gradients) == 4,
            "exact_gradient_reproduction": gradient_gate,
            "full_gradient_tensors_serialized": False,
            "new_training_or_candidate_created": False,
            "mechanism_used_for_candidate_selection": False,
        },
        "source_immutability": source_gate,
        "order_gate": order_gate,
        "track_a_route": "T5R1_reviewer_access_stop",
        "t6_authorized": False,
        "next_authorization": "none_pending_valid_independent_review_environment",
    }
    route_path = P6_ROOT / "step3ct_t5r_p6_mechanism_route_decision.json"
    write_json(route_path, route_result)
    summary = {
        "name": "step3ct_t5r_p6_mechanism_completion_summary_v1",
        "route": route,
        "execution_git_head": git("rev-parse", "HEAD"),
        "p5_freeze_sha256": training_common.sha256_file(FREEZE_PATH),
        "representation_trajectories": trajectories,
        "registered_differences": differences,
        "grouped_gradient_trajectories": grouped_trajectory,
        "interpretation_boundary": {
            "explanatory_only": True,
            "candidate_selection_allowed": False,
            "behavior_failure_rescue_allowed": False,
            "causal_mediation_claim_allowed": False,
            "c3b_redundancy_warning_preserved": True,
        },
    }
    summary_path = P6_ROOT / "step3ct_t5r_p6_mechanism_completion_summary.json"
    write_json(summary_path, summary)
    report_lines = [
        "# Step3C-T5R P6 Mechanism Completion Report",
        "",
        f"Route: `{route}`",
        "",
        "## Integrity",
        "",
        f"- Representation probes: {len(representations)}/12 complete with exact 40-prompt/80-target order.",
        f"- Grouped-gradient labels: {sum(int(row['passed']) for row in gradients)}/4 reproduced.",
        f"- Frozen source artifacts unchanged: {source_gate['passed']}.",
        "- No diagnostic optimizer update, full-gradient serialization, new arm, new training, or heldout access occurred.",
        "",
        "## Representation Trajectories",
        "",
        "| Arm | Step | Top1 | Top5 | MRR | Recall@5 | Global gap | Hard gap | Prompt rank | Target rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trajectories:
        report_lines.append(
            f"| {row['arm']} | {row['step']} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['mrr']:.4f} | {row['multi_positive_recall_at5']:.4f} | {row['global_gap_mean']:.4f} | "
            f"{row['hard_gap_mean']:.4f} | {row['prompt_effective_rank']:.2f} | {row['target_effective_rank']:.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These results are mechanism-only evidence. They do not select or rescue a candidate, and the prior C3b "
            "redundancy warning remains active. Track A remains at `T5R1_reviewer_access_stop`; without `T5R4`, T6 "
            "and all new training remain unauthorized.",
            "",
        ]
    )
    report_path = P6_ROOT / "Step3CT_T5R_P6_mechanism_completion_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    incident_paths = sorted((P6_ROOT / "incidents").glob("*.json"))
    package_paths = representation_paths + gradient_paths + incident_paths + [route_path, summary_path, report_path]
    package = {
        "name": "step3ct_t5r_p6_mechanism_manifest_package_v1",
        "route": route,
        "artifacts": mechanism.hash_file_set(package_paths, REPO),
        "artifact_count": len(package_paths),
        "source_artifact_count": inventory["source_artifact_count"],
        "source_artifacts_byte_identical": source_gate["passed"],
    }
    package_path = P6_ROOT / "step3ct_t5r_p6_mechanism_manifest_package.json"
    write_json(package_path, package)
    if not passed:
        raise RuntimeError(f"P6 completed with route {route}")
    print(json.dumps({"phase": "summarize", "route": route, "artifacts": len(package_paths)}))
    return summary


def main() -> None:
    args = parse_args()
    if args.phase == "freeze":
        freeze_implementation()
    elif args.phase == "representations":
        if args.arm is None or args.step is None:
            raise ValueError("representations requires --arm and --step")
        run_representation(args.arm, args.step)
    elif args.phase == "gradients":
        if args.step is None:
            raise ValueError("gradients requires --step as the diagnostic label")
        run_gradient_label(args.step)
    else:
        summarize_mechanism()


if __name__ == "__main__":
    main()
