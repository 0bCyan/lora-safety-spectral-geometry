"""Freeze, run, and summarize Step3C-MF C3 no-update feasibility diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import step3cm_c3_common as common


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "experiments/step3_response_quality_persistence"
CONFIG = ROOT / "configs/step3cm_c3_no_update.json"
REGISTRY = ROOT / "configs/step3cmf_registry.json"
COMMON = ROOT / "scripts/step3cm_c3_common.py"
TEST_SCRIPT = ROOT / "scripts/test_step3cm_c3_common.py"
OUTPUT = ROOT / "results/step3cm_c3_no_update"
PROTOCOL = OUTPUT / "protocol"
RAW = OUTPUT / "raw"
REPORTS = OUTPUT / "reports"
PLOTS = OUTPUT / "plots"
TEST_RESULT = PROTOCOL / "step3cm_c3_implementation_tests.json"
CODE_FREEZE = PROTOCOL / "step3cm_c3_code_freeze.json"
CODE_REPORT = PROTOCOL / "Step3CM_C3_code_freeze_report.md"
CANDIDATE_ROUTE = REPORTS / "step3cm_c3_candidate_route.json"
JOINT_ROUTE = REPORTS / "step3cm_joint_route_decision.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def config() -> dict[str, Any]:
    return read_json(CONFIG)


def registry() -> dict[str, Any]:
    return read_json(REGISTRY)


def run_tests() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import test_step3cm_c3_common

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_step3cm_c3_common)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "name": "step3cm_c3_implementation_tests_v1",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "coverage": [
            "left/right padding and truncation masks",
            "multi-positive indexing and hard-negative exclusion",
            "singleton categories",
            "queue initialization, wraparound, and determinism",
            "C3b equality to C3a with hard term disabled",
            "finite non-zero gradients and frozen base parameters",
            "temperature and eta rules",
            "projection and retrieval determinism",
            "no optimizer or scheduler construction",
        ],
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "heldout_opened": False,
    }
    write_json(TEST_RESULT, payload)
    if not payload["passed"]:
        raise RuntimeError("C3 implementation tests failed")
    return payload


def validate_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    reg = registry()["c3_no_update"]
    fields = [
        "model_name_or_path",
        "checkpoint_dir",
        "projection_dim",
        "queue_size_sequence",
        "temperature_candidates",
        "positive_probability_min",
        "positive_probability_max",
        "positive_probability_target",
        "minimum_vram_headroom_fraction",
        "permutation_repeats",
        "bootstrap_repeats",
    ]
    mismatches = {field: [cfg.get(field), reg.get(field)] for field in fields if cfg.get(field) != reg.get(field)}
    targets = read_jsonl(REPO / cfg["targets_path"])
    prompts = read_jsonl(REPO / cfg["safety_train_path"])
    checkpoint = REPO / cfg["checkpoint_dir"]
    target_counts = Counter(row["sample_id"] for row in targets)
    result = {
        "registry_mismatches": mismatches,
        "target_rows": len(targets),
        "target_prompt_clusters": len(target_counts),
        "two_targets_per_prompt": all(count == 2 for count in target_counts.values()),
        "all_targets_approved": all(row.get("reviewer_decision") == "approve" and row.get("step2g_split") == "safety_train" for row in targets),
        "safety_train_rows": len(prompts),
        "target_prompt_ids_equal_safety_train": set(target_counts) == {row["sample_id"] for row in prompts},
        "checkpoint_exists": checkpoint.is_dir(),
        "adapter_exists": (checkpoint / "adapter_model.safetensors").is_file(),
        "validation_pass": read_json(REPO / cfg["validation_route"]).get("overall_pass") is True,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    result["passed"] = (
        not mismatches
        and result["target_rows"] == 80
        and result["target_prompt_clusters"] == 40
        and result["two_targets_per_prompt"]
        and result["all_targets_approved"]
        and result["safety_train_rows"] == 40
        and result["target_prompt_ids_equal_safety_train"]
        and result["checkpoint_exists"]
        and result["adapter_exists"]
        and result["validation_pass"]
    )
    if not result["passed"]:
        raise ValueError(f"C3 source validation failed: {result}")
    return result


def freeze_code() -> dict[str, Any]:
    cfg = config()
    tests = read_json(TEST_RESULT) if TEST_RESULT.exists() else run_tests()
    if not tests.get("passed"):
        raise ValueError("tests must pass before code freeze")
    source_validation = validate_sources(cfg)
    checkpoint = REPO / cfg["checkpoint_dir"]
    artifacts = [CONFIG, REGISTRY, COMMON, TEST_SCRIPT, Path(__file__), REPO / cfg["targets_path"], REPO / cfg["safety_train_path"], checkpoint / "adapter_config.json", checkpoint / "adapter_model.safetensors", REPO / cfg["validation_route"], TEST_RESULT]
    manifest = {
        "name": "step3cm_c3_code_freeze_v1",
        "git_head_before_numerical_run": git_head(),
        "artifacts": [{"path": repo_path(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in artifacts],
        "source_validation": source_validation,
        "candidate_flags": cfg["candidates"],
        "calibration_prompt_rule": f"smallest stable hashes, n={cfg['calibration_prompt_count']}, seed={cfg['calibration_seed']}",
        "response_branch_gradient_method": "exact branchwise derivative sum with detached queue negatives",
        "f1_code_and_config_frozen_before_numerical_run": True,
        "candidate_outputs_generated": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(CODE_FREEZE, manifest)
    CODE_REPORT.write_text(
        "# Step3C-MF C3 No-Update Code Freeze\n\n"
        "Status: implementation and numerical configuration frozen before any model diagnostic.\n\n"
        f"- Entry commit: `{manifest['git_head_before_numerical_run']}`\n"
        f"- Tests: {tests['tests_run']} passed\n"
        "- Targets: 80 approved natural-v1 targets across 40 safety-train prompts\n"
        "- Candidates share one implementation and differ only by the registered hard-negative flags\n"
        "- Queue, temperature, eta, calibration, permutation, and bootstrap rules are frozen\n"
        "- Optimizer objects: 0; scheduler objects: 0; optimizer steps: 0\n"
        "- Core100 opened: false; heldout opened: false; behavior generated: false\n"
        "- Training authorized: false\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def require_code_freeze() -> dict[str, Any]:
    frozen = read_json(CODE_FREEZE)
    mismatches = []
    for row in frozen["artifacts"]:
        path = REPO / row["path"]
        if not path.exists() or sha256(path) != row["sha256"]:
            mismatches.append(row["path"])
    if mismatches:
        raise ValueError(f"code freeze mismatch: {mismatches}")
    return frozen


def format_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_chat_text(tokenizer: Any, prompt: str, response: str) -> str:
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prompt_feature(tokenizer: Any, prompt: str, max_length: int) -> dict[str, list[int]]:
    return tokenizer(prompt, truncation=True, max_length=max_length, padding=False)


def response_feature(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, list[int]]:
    prompt_text = format_prompt_text(tokenizer, prompt)
    full_text = format_chat_text(tokenizer, prompt, response)
    full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    prefix = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
    labels = list(full["input_ids"])
    labels[: min(len(prefix), len(labels))] = [-100] * min(len(prefix), len(labels))
    full["labels"] = labels
    if not any(label != -100 for label in labels):
        raise ValueError("response was fully truncated")
    return full


def load_model(cfg: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True, local_files_only=cfg["local_files_only"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        local_files_only=cfg["local_files_only"],
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, REPO / cfg["checkpoint_dir"], is_trainable=True)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def model_device(model: Any) -> Any:
    return next(parameter.device for parameter in model.parameters() if parameter.device.type == "cuda")


def encode_feature(model: Any, feature: dict[str, list[int]], mask_kind: str, head: Any, with_grad: bool) -> Any:
    import torch

    device = model_device(model)
    input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
    attention = torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device)
    if mask_kind == "prompt":
        pool_mask = attention
    else:
        labels = torch.tensor([feature["labels"]], dtype=torch.long, device=device)
        pool_mask = (labels != -100).long() * attention
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, use_cache=False, return_dict=True)
        pooled = common.masked_mean(outputs.hidden_states[-1], pool_mask)
        projected = common.l2_normalize(head(pooled.float()))
    return projected.squeeze(0)


def build_representations(model: Any, tokenizer: Any, cfg: dict[str, Any], head: Any, targets: list[dict[str, Any]]) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    target_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        target_by_prompt[row["sample_id"]].append(row)
    prompt_rows = [rows[0] for _, rows in sorted(target_by_prompt.items())]
    prompt_vectors = []
    target_vectors = []
    for index, row in enumerate(prompt_rows, start=1):
        print(f"representation prompt {index}/{len(prompt_rows)} {row['sample_id']}", flush=True)
        feature = prompt_feature(tokenizer, row["prompt"], int(cfg["max_length"]))
        prompt_vectors.append(encode_feature(model, feature, "prompt", head, False).cpu())
    target_rows = sorted(targets, key=lambda row: (row["sample_id"], row["target_id"]))
    for index, row in enumerate(target_rows, start=1):
        print(f"representation target {index}/{len(target_rows)} {row['target_id']}", flush=True)
        feature = response_feature(tokenizer, row["prompt"], row["response"], int(cfg["max_length"]))
        target_vectors.append(encode_feature(model, feature, "response", head, False).cpu())
    return torch.stack(prompt_vectors), torch.stack(target_vectors), prompt_rows, target_rows


def select_queue(model: Any, cfg: dict[str, Any]) -> tuple[int | None, list[dict[str, Any]]]:
    import torch

    trials = []
    selected = None
    device = model_device(model)
    for size in cfg["queue_size_sequence"]:
        torch.cuda.reset_peak_memory_stats(device)
        queue = torch.empty(int(size), int(cfg["projection_dim"]), dtype=torch.float32, device=device)
        queue.zero_()
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        headroom = free / total
        trials.append({
            "queue_size": int(size),
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_bytes": int(allocated),
            "reserved_bytes": int(reserved),
            "headroom_fraction": headroom,
            "pass": headroom >= float(cfg["minimum_vram_headroom_fraction"]),
        })
        del queue
        torch.cuda.empty_cache()
        if trials[-1]["pass"] and selected is None:
            selected = int(size)
    return selected, trials


def negative_indices(target_rows: list[dict[str, Any]], prompt_id: str, queue_size: int, seed: int) -> list[int]:
    candidates = [index for index, row in enumerate(target_rows) if row["sample_id"] != prompt_id]
    candidates.sort(key=lambda index: stable_hash(f"{seed}:{prompt_id}:{target_rows[index]['target_id']}"))
    return candidates[:queue_size]


def temperature_trials(prompt_embeddings: Any, target_embeddings: Any, prompt_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], cfg: dict[str, Any], queue_size: int) -> list[dict[str, Any]]:
    import torch

    trials = []
    for temperature in cfg["temperature_candidates"]:
        probabilities = []
        for query_index, prompt in enumerate(prompt_rows):
            positives = [index for index, row in enumerate(target_rows) if row["sample_id"] == prompt["sample_id"]]
            indices = positives + negative_indices(target_rows, prompt["sample_id"], queue_size, int(cfg["calibration_seed"]))
            logits = prompt_embeddings[query_index] @ target_embeddings[indices].T / float(temperature)
            probability = torch.softmax(logits, dim=0)[: len(positives)].sum()
            probabilities.append(float(probability))
        ordered = sorted(probabilities)
        median = (ordered[19] + ordered[20]) / 2
        trials.append({
            "temperature": float(temperature),
            "median_positive_probability": median,
            "minimum_positive_probability": min(probabilities),
            "maximum_positive_probability": max(probabilities),
            "probabilities": probabilities,
        })
    return trials


def trainable_lora_parameters(model: Any) -> list[tuple[str, Any]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name]


def zero_gradients(model: Any) -> None:
    for _, parameter in trainable_lora_parameters(model):
        parameter.grad = None


def capture_gradients(model: Any) -> dict[str, Any]:
    import torch

    gradients = {}
    for name, parameter in trainable_lora_parameters(model):
        gradients[name] = torch.zeros_like(parameter, dtype=torch.float32, device="cpu") if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
    return gradients


def add_gradients(total: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if total is None:
        return {name: value.clone() for name, value in current.items()}
    for name, value in current.items():
        total[name].add_(value)
    return total


def scale_gradients(gradients: dict[str, Any], divisor: float) -> dict[str, Any]:
    return {name: value / divisor for name, value in gradients.items()}


def gradient_norm(gradients: dict[str, Any]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in gradients.values()))


def gradient_cosine(left: dict[str, Any], right: dict[str, Any], names: list[str] | None = None) -> float:
    selected = names or sorted(set(left) & set(right))
    dot = sum(float((left[name] * right[name]).sum()) for name in selected)
    left_norm = math.sqrt(sum(float(left[name].square().sum()) for name in selected))
    right_norm = math.sqrt(sum(float(right[name].square().sum()) for name in selected))
    return dot / (left_norm * right_norm) if left_norm > 0 and right_norm > 0 else math.nan


def parameter_group(name: str) -> str:
    if any(token in name for token in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attention"
    if any(token in name for token in ["gate_proj", "up_proj", "down_proj"]):
        return "mlp"
    return "other"


def layer_number(name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def loss_for_query(query: Any, keys: Any, prompt_id: str, category: str, key_prompt_ids: list[str], key_categories: list[str], temperature: float, candidate: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    positives = common.positive_mask([prompt_id], key_prompt_ids, device=query.device)
    hard = common.hard_negative_mask([prompt_id], [category], key_prompt_ids, key_categories, device=query.device)
    return common.multi_positive_loss(query.unsqueeze(0), keys, positives, temperature, hard, float(candidate["hard_term_multiplier"]))


def branchwise_binding_gradient(
    model: Any,
    tokenizer: Any,
    head: Any,
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    prompt_embeddings: Any,
    target_embeddings: Any,
    prompt_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    queue_size: int,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    prompt_index = {row["sample_id"]: index for index, row in enumerate(prompt_rows)}
    total = None
    raw = []
    branch_count = 0
    device = model_device(model)
    for prompt in calibration_rows:
        prompt_id = prompt["sample_id"]
        positive_indices = [index for index, row in enumerate(target_rows) if row["sample_id"] == prompt_id]
        indices = positive_indices + negative_indices(target_rows, prompt_id, queue_size, int(cfg["calibration_seed"]))
        key_prompt_ids = [target_rows[index]["sample_id"] for index in indices]
        key_categories = [target_rows[index]["category"] for index in indices]
        fixed_keys = target_embeddings[indices].to(device)
        fixed_query = prompt_embeddings[prompt_index[prompt_id]].to(device)
        detached_loss, detail = loss_for_query(fixed_query, fixed_keys, prompt_id, prompt["category"], key_prompt_ids, key_categories, temperature, candidate)
        raw.append({
            "candidate": candidate["id"],
            "sample_id": prompt_id,
            "loss": float(detached_loss),
            "global_loss": float(detail["global_loss"]),
            "hard_loss": float(detail["hard_loss"]),
            "positive_probability": float(detail["positive_probability"][0]),
        })

        branches: list[tuple[str, int | None]] = [("prompt", None)] + [("response", index) for index in positive_indices]
        for branch, source_index in branches:
            zero_gradients(model)
            if branch == "prompt":
                feature = prompt_feature(tokenizer, prompt["prompt"], int(cfg["max_length"]))
                query = encode_feature(model, feature, "prompt", head, True)
                keys = fixed_keys
            else:
                target = target_rows[int(source_index)]
                feature = response_feature(tokenizer, target["prompt"], target["response"], int(cfg["max_length"]))
                dynamic_key = encode_feature(model, feature, "response", head, True)
                keys = fixed_keys.clone()
                local_position = indices.index(int(source_index))
                keys[local_position] = dynamic_key
                query = fixed_query
            loss, _ = loss_for_query(query, keys, prompt_id, prompt["category"], key_prompt_ids, key_categories, temperature, candidate)
            loss.backward()
            current = capture_gradients(model)
            total = add_gradients(total, current)
            branch_count += 1
            del loss, query, keys
            torch.cuda.empty_cache()
    if total is None:
        raise ValueError("no binding gradients were computed")
    return scale_gradients(total, branch_count), raw


def safety_gradient(model: Any, tokenizer: Any, cfg: dict[str, Any], calibration_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    target_by_prompt = defaultdict(list)
    for row in target_rows:
        target_by_prompt[row["sample_id"]].append(row)
    total = None
    device = model_device(model)
    for prompt in calibration_rows:
        target = sorted(target_by_prompt[prompt["sample_id"]], key=lambda row: row["target_id"])[0]
        feature = response_feature(tokenizer, target["prompt"], target["response"], int(cfg["max_length"]))
        batch = {
            "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device),
            "labels": torch.tensor([feature["labels"]], dtype=torch.long, device=device),
        }
        zero_gradients(model)
        output = model(**batch, use_cache=False, return_dict=True)
        output.loss.backward()
        total = add_gradients(total, capture_gradients(model))
        del output, batch
        torch.cuda.empty_cache()
    if total is None:
        raise ValueError("no safety gradients were computed")
    return scale_gradients(total, len(calibration_rows))


def representation_atlas(prompt_embeddings: Any, target_embeddings: Any, prompt_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    similarities = prompt_embeddings @ target_embeddings.T
    query_ids = [row["sample_id"] for row in prompt_rows]
    key_ids = [row["sample_id"] for row in target_rows]
    cosine_rows = []
    prompt_gap_rows = []
    for query_index, prompt in enumerate(prompt_rows):
        positive = []
        same_category = []
        cross_category = []
        for key_index, target in enumerate(target_rows):
            value = float(similarities[query_index, key_index])
            if target["sample_id"] == prompt["sample_id"]:
                group = "same_prompt"
                positive.append(value)
            elif target["category"] == prompt["category"]:
                group = "same_category_cross_prompt"
                same_category.append(value)
            else:
                group = "cross_category"
                cross_category.append(value)
            cosine_rows.append({"sample_id": prompt["sample_id"], "target_id": target["target_id"], "group": group, "cosine": value})
        global_negative = same_category + cross_category
        prompt_gap_rows.append({
            "sample_id": prompt["sample_id"],
            "positive_mean": sum(positive) / len(positive),
            "global_negative_mean": sum(global_negative) / len(global_negative),
            "hard_negative_mean": sum(same_category) / len(same_category) if same_category else math.nan,
            "positive_minus_global": sum(positive) / len(positive) - sum(global_negative) / len(global_negative),
            "positive_minus_hard": sum(positive) / len(positive) - sum(same_category) / len(same_category) if same_category else math.nan,
        })
    retrieval = common.retrieval_metrics(similarities, query_ids, key_ids)
    rng = random.Random(int(cfg["calibration_seed"]))
    permutations = []
    grouped_ids = sorted(set(key_ids))
    for repeat in range(int(cfg["permutation_repeats"])):
        shuffled = grouped_ids[:]
        rng.shuffle(shuffled)
        mapping = dict(zip(grouped_ids, shuffled))
        permuted_key_ids = [mapping[prompt_id] for prompt_id in key_ids]
        permutations.append({"repeat": repeat, **common.retrieval_metrics(similarities, query_ids, permuted_key_ids)})
    global_values = [row["positive_minus_global"] for row in prompt_gap_rows]
    hard_values = [row["positive_minus_hard"] for row in prompt_gap_rows if math.isfinite(row["positive_minus_hard"])]
    summary = {
        "retrieval": retrieval,
        "permutation_mean": {field: sum(row[field] for row in permutations) / len(permutations) for field in ["top1", "top5", "mrr", "multi_positive_recall_at5"]},
        "positive_minus_global_bootstrap": common.bootstrap_mean_interval(global_values, int(cfg["bootstrap_repeats"]), int(cfg["calibration_seed"])),
        "positive_minus_hard_bootstrap": common.bootstrap_mean_interval(hard_values, int(cfg["bootstrap_repeats"]), int(cfg["calibration_seed"]) + 1),
        "prompt_effective_rank": common.effective_rank(prompt_embeddings),
        "target_effective_rank": common.effective_rank(target_embeddings),
    }
    return summary, cosine_rows, permutations


def gradient_atlas(binding: dict[str, Any], safety: dict[str, Any], candidate_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = defaultdict(list)
    for name in binding:
        groups[parameter_group(name)].append(name)
    binding_total = sum(float(value.square().sum()) for value in binding.values())
    layer_rows = []
    for name in sorted(binding):
        energy = float(binding[name].square().sum())
        layer_rows.append({
            "candidate": candidate_id,
            "parameter": name,
            "layer": layer_number(name),
            "group": parameter_group(name),
            "binding_energy": energy,
            "binding_energy_share": energy / binding_total if binding_total > 0 else math.nan,
            "binding_finite": bool(binding[name].isfinite().all()),
            "binding_nonzero": bool(binding[name].count_nonzero()),
            "safety_energy": float(safety[name].square().sum()),
        })
    summary = {
        "global_cosine": gradient_cosine(binding, safety),
        "group_cosine": {group: gradient_cosine(binding, safety, names) for group, names in groups.items()},
        "binding_norm": gradient_norm(binding),
        "safety_norm": gradient_norm(safety),
        "trainable_lora_parameter_tensors": len(binding),
        "finite_tensor_fraction": sum(bool(value.isfinite().all()) for value in binding.values()) / len(binding),
        "nonzero_tensor_fraction": sum(bool(value.count_nonzero()) for value in binding.values()) / len(binding),
    }
    return summary, layer_rows


def save_plots(cosine_rows: list[dict[str, Any]], permutations: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    groups = ["same_prompt", "same_category_cross_prompt", "cross_category"]
    plt.figure(figsize=(8, 5))
    for group in groups:
        values = [row["cosine"] for row in cosine_rows if row["group"] == group]
        plt.hist(values, bins=25, alpha=0.45, label=group, density=True)
    plt.xlabel("cosine")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "step3cm_c3_cosine_distributions.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist([row["mrr"] for row in permutations], bins=20)
    plt.xlabel("shuffled MRR")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(PLOTS / "step3cm_c3_permutation_mrr.png", dpi=150)
    plt.close()

    by_layer = defaultdict(float)
    for row in layer_rows:
        by_layer[(row["candidate"], row["layer"])] += row["binding_energy_share"]
    plt.figure(figsize=(9, 4))
    for candidate in sorted({row["candidate"] for row in layer_rows}):
        layers = sorted(layer for cid, layer in by_layer if cid == candidate)
        plt.plot(layers, [by_layer[(candidate, layer)] for layer in layers], label=candidate)
    plt.xlabel("LoRA layer")
    plt.ylabel("binding gradient energy share")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "step3cm_c3_layerwise_gradient_energy.png", dpi=150)
    plt.close()


def environment_record(started: float, model: Any) -> dict[str, Any]:
    import torch

    device = model_device(model)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "name": "step3cm_c3_environment_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "total_vram_bytes": int(total),
        "free_vram_bytes_at_end": int(free),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "wall_clock_seconds": time.perf_counter() - started,
        "model_name_or_path": config()["model_name_or_path"],
        "checkpoint_dir": config()["checkpoint_dir"],
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
        "candidate_outputs_generated": False,
    }


def run_diagnostics() -> dict[str, Any]:
    import numpy as np
    import torch

    require_code_freeze()
    cfg = config()
    started = time.perf_counter()
    for directory in [RAW, REPORTS, PLOTS]:
        directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["projection_seed"]))
    random.seed(int(cfg["calibration_seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(cfg)
    device = model_device(model)
    hidden_size = int(model.config.hidden_size)
    head = common.projection_head(hidden_size, int(cfg["projection_dim"]), int(cfg["projection_seed"])).to(device)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    trainable = trainable_lora_parameters(model)
    non_lora_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" not in name]
    if not trainable or non_lora_trainable:
        raise ValueError(f"invalid trainable parameter boundary: lora={len(trainable)}, non_lora={non_lora_trainable[:5]}")
    targets = read_jsonl(REPO / cfg["targets_path"])
    prompt_embeddings, target_embeddings, prompt_rows, target_rows = build_representations(model, tokenizer, cfg, head, targets)
    np.savez_compressed(
        RAW / "step3cm_c3_embeddings.npz",
        prompt_embeddings=prompt_embeddings.numpy(),
        target_embeddings=target_embeddings.numpy(),
        prompt_ids=np.array([row["sample_id"] for row in prompt_rows]),
        target_ids=np.array([row["target_id"] for row in target_rows]),
    )
    queue_size, queue_trials = select_queue(model, cfg)
    if queue_size is None:
        raise RuntimeError("no queue size satisfies VRAM headroom")
    temperature_rows = temperature_trials(prompt_embeddings, target_embeddings, prompt_rows, target_rows, cfg, queue_size)
    selected_temperature = common.choose_temperature(
        temperature_rows,
        float(cfg["positive_probability_min"]),
        float(cfg["positive_probability_max"]),
        float(cfg["positive_probability_target"]),
    )
    atlas, cosine_rows, permutations = representation_atlas(prompt_embeddings, target_embeddings, prompt_rows, target_rows, cfg)
    write_csv(RAW / "step3cm_c3_cosine_distributions.csv", cosine_rows)
    write_csv(RAW / "step3cm_c3_permutation_controls.csv", permutations)
    write_json(RAW / "step3cm_c3_queue_trials.json", queue_trials)
    write_json(RAW / "step3cm_c3_temperature_trials.json", temperature_rows)
    write_json(REPORTS / "step3cm_c3_representation_atlas.json", atlas)

    calibration_rows = sorted(prompt_rows, key=lambda row: stable_hash(f"{cfg['calibration_seed']}:{row['sample_id']}"))[: int(cfg["calibration_prompt_count"])]
    write_json(PROTOCOL / "step3cm_c3_calibration_prompts.json", [{"sample_id": row["sample_id"], "category": row["category"]} for row in calibration_rows])
    safety = safety_gradient(model, tokenizer, cfg, calibration_rows, target_rows)
    safety_norm = gradient_norm(safety)
    candidate_summaries = []
    all_loss_rows = []
    all_layer_rows = []
    if selected_temperature is None:
        selected_temperature_value = math.nan
    else:
        selected_temperature_value = float(selected_temperature["temperature"])
    for candidate in cfg["candidates"]:
        print(f"gradient diagnostic {candidate['id']}", flush=True)
        if selected_temperature is None:
            binding = {name: value * math.nan for name, value in safety.items()}
            loss_rows = []
        else:
            binding, loss_rows = branchwise_binding_gradient(
                model, tokenizer, head, cfg, candidate, calibration_rows,
                prompt_embeddings, target_embeddings, prompt_rows, target_rows,
                queue_size, selected_temperature_value,
            )
        all_loss_rows.extend(loss_rows)
        grad_summary, layer_rows = gradient_atlas(binding, safety, candidate["id"])
        all_layer_rows.extend(layer_rows)
        eta, realized_ratio = common.calibrate_eta(
            grad_summary["binding_norm"], safety_norm, float(cfg["binding_to_safety_gradient_ratio_target"])
        )
        finite_losses = bool(loss_rows) and all(math.isfinite(row["loss"]) for row in loss_rows)
        gates = {
            "finite_losses_and_similarities": finite_losses and all(math.isfinite(row["cosine"]) for row in cosine_rows),
            "finite_lora_gradients": grad_summary["finite_tensor_fraction"] == 1.0,
            "nonzero_lora_gradients": grad_summary["binding_norm"] > 0 and grad_summary["nonzero_tensor_fraction"] > 0,
            "queue_memory_rule": any(row["queue_size"] == queue_size and row["pass"] for row in queue_trials),
            "temperature_probability_rule": selected_temperature is not None,
            "eta_finite_positive": math.isfinite(eta) and eta > 0,
            "eta_ratio_rule": float(cfg["binding_to_safety_gradient_ratio_min"]) <= realized_ratio <= float(cfg["binding_to_safety_gradient_ratio_max"]),
            "frozen_base_parameters": not non_lora_trainable,
            "implementation_tests": read_json(TEST_RESULT)["passed"],
            "optimizer_steps_zero": True,
        }
        gates["overall_pass"] = all(gates.values())
        candidate_summaries.append({
            "candidate": candidate["id"],
            "flags": candidate,
            "selected_queue_size": queue_size,
            "selected_temperature": selected_temperature_value,
            "median_positive_probability": selected_temperature["median_positive_probability"] if selected_temperature else math.nan,
            "eta": eta,
            "realized_binding_to_safety_ratio": realized_ratio,
            "gradient": grad_summary,
            "representation": atlas,
            "gates": gates,
        })
    write_csv(RAW / "step3cm_c3_calibration_losses.csv", all_loss_rows)
    write_csv(RAW / "step3cm_c3_layerwise_gradient_energy.csv", all_layer_rows)
    save_plots(cosine_rows, permutations, all_layer_rows)
    environment = environment_record(started, model)
    write_json(REPORTS / "step3cm_c3_environment.json", environment)
    route = {
        "name": "step3cm_c3_candidate_route_v1",
        "candidates": candidate_summaries,
        "passing_candidates": [row["candidate"] for row in candidate_summaries if row["gates"]["overall_pass"]],
        "at_least_one_candidate_pass": any(row["gates"]["overall_pass"] for row in candidate_summaries),
        "next_action": "joint_measurement_plus_c3_decision",
        "behavior_generated": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
        "training_authorized": False,
    }
    write_json(CANDIDATE_ROUTE, route)
    REPORTS.joinpath("Step3CM_C3_no_update_report.md").write_text(
        "# Step3C-MF C3 No-Update Feasibility Report\n\n"
        f"Status: {'at least one candidate passes' if route['at_least_one_candidate_pass'] else 'no candidate passes'}.\n\n"
        f"- Model/checkpoint: `{cfg['model_name_or_path']}` / `{cfg['checkpoint_dir']}`\n"
        f"- Selected queue: {queue_size}; temperature: {selected_temperature_value}\n"
        f"- Retrieval top1/top5/MRR: {atlas['retrieval']['top1']:.3f} / {atlas['retrieval']['top5']:.3f} / {atlas['retrieval']['mrr']:.3f}\n"
        + "\n".join(
            f"- {row['candidate']}: pass={row['gates']['overall_pass']}, eta={row['eta']}, ratio={row['realized_binding_to_safety_ratio']}, gradient cosine={row['gradient']['global_cosine']}"
            for row in candidate_summaries
        )
        + "\n\nNo optimizer or scheduler was constructed. Optimizer steps, behavior generations, core100 access, and heldout access were all zero/false. No training is authorized.\n",
        encoding="utf-8",
        newline="\n",
    )
    del model, head
    torch.cuda.empty_cache()
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return route


def write_joint_decision() -> dict[str, Any]:
    validation = read_json(REPO / config()["validation_route"])
    c3 = read_json(CANDIDATE_ROUTE)
    measurement_pass = bool(validation["overall_pass"])
    c3_pass = bool(c3["at_least_one_candidate_pass"])
    if measurement_pass and c3_pass:
        decision = "stop_and_request_prospective_seed1_training_plan"
    elif measurement_pass:
        decision = "retain_validated_measurement_and_close_C3"
    elif c3_pass:
        decision = "record_numerical_feasibility_and_prohibit_training"
    else:
        decision = "close_Step3C_MF"
    route = {
        "name": "step3cm_joint_route_decision_v1",
        "measurement_validation_pass": measurement_pass,
        "c3_at_least_one_candidate_pass": c3_pass,
        "passing_candidates": c3["passing_candidates"],
        "decision": decision,
        "training_authorized": False,
        "new_training_plan_required": measurement_pass and c3_pass,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    write_json(JOINT_ROUTE, route)
    REPORTS.joinpath("Step3CM_joint_decision_report.md").write_text(
        "# Step3C-MF Joint Decision\n\n"
        f"- Measurement validation: {'pass' if measurement_pass else 'fail'}\n"
        f"- C3 no-update feasibility: {'pass' if c3_pass else 'fail'}\n"
        f"- Passing candidates: {', '.join(c3['passing_candidates']) if c3['passing_candidates'] else 'none'}\n"
        f"- Decision: `{decision}`\n\n"
        "This decision does not authorize training. Core100 and heldout remained closed, no behavior was generated, and optimizer/scheduler objects and optimizer steps remained zero.\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "step3cm_c3_artifact_manifest.json")
    write_json(REPORTS / "step3cm_c3_artifact_manifest.json", {
        "name": "step3cm_c3_artifact_manifest_v1",
        "artifacts": [{"path": repo_path(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in artifacts],
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
    })
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "freeze", "run", "joint"])
    args = parser.parse_args()
    os.chdir(REPO)
    if args.command == "test":
        print(json.dumps(run_tests(), ensure_ascii=False, indent=2))
    elif args.command == "freeze":
        freeze_code()
    elif args.command == "run":
        run_diagnostics()
    else:
        write_joint_decision()


if __name__ == "__main__":
    main()
