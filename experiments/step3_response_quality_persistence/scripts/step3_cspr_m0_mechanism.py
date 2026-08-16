"""Prepare and run the CSPR M0 real-model zero-update mechanism pilot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import step3_cspr_common as cspr
import step3ct_training_common as training


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_m0/"
    "protocol/cspr_m0_protocol.json"
)


class M0ValidationError(ValueError):
    """Raised when an M0 artifact violates the frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise M0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: str) -> str:
    payload = "|".join((str(seed), *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


def write_frozen_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_frozen(path: Path, text: str) -> None:
    write_frozen_bytes(path, text.encode("utf-8"))


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_frozen(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_frozen_bytes(path, jsonl_bytes(rows))


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise M0ValidationError(f"frozen input hash mismatch: {name}")
    if protocol.get("optimizer_steps") != 0 or protocol.get("training_authorized") is not False:
        raise M0ValidationError("M0 violates the zero-training boundary")
    if protocol.get("core100_opened") is not False or protocol.get("heldout_opened") is not False:
        raise M0ValidationError("M0 opens a prohibited evaluation set")
    if protocol.get("human_labels_used") is not False:
        raise M0ValidationError("M0 must not claim human-qualified targets")
    p2t = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p2t_terminal_analysis"]["path"])
    )
    if p2t.get("route") != "STOP_CSPR_P2T_CALIBRATION_UNRELIABLE":
        raise M0ValidationError("M0 does not descend from the frozen P2T stop route")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise M0ValidationError("M0 entry commit is not an ancestor of HEAD")
    return protocol


def select_category_diverse(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    category_field: str,
    id_field: str,
    namespace: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row[category_field])
        grouped[category].append(row)
    if len(grouped) < count:
        raise M0ValidationError(f"{namespace} has fewer than {count} categories")
    categories = sorted(
        grouped,
        key=lambda value: stable_rank(seed, namespace, "category", value),
    )[:count]
    selected = []
    for category in categories:
        selected.append(
            min(
                grouped[category],
                key=lambda row: stable_rank(
                    seed, namespace, "row", category, str(row[id_field])
                ),
            )
        )
    return sorted(selected, key=lambda row: stable_rank(seed, namespace, "order", str(row[id_field])))


def token_shape(tokenizer: Any, prompt: str, response: str) -> dict[str, int]:
    prompt_text = training.format_prompt_text(tokenizer, prompt)
    full_text = training.format_chat_text(tokenizer, prompt, response)
    prompt_ids = tokenizer(prompt_text, truncation=False, padding=False)["input_ids"]
    full_ids = tokenizer(full_text, truncation=False, padding=False)["input_ids"]
    masked = min(len(prompt_ids), len(full_ids))
    return {
        "total_tokens": len(full_ids),
        "prompt_tokens": masked,
        "response_tokens": len(full_ids) - masked,
    }


def choose_candidate_per_pair(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    return [
        min(
            candidates,
            key=lambda row: stable_rank(seed, "candidate", pair_id, str(row["opaque_authoring_id"])),
        )
        for pair_id, candidates in sorted(grouped.items())
    ]


def find_wrong_prompt_derangement(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> list[str]:
    prompts = [str(row["prompt"]) for row in rows]
    for shift in range(1, len(rows)):
        wrong = prompts[shift:] + prompts[:shift]
        if all(
            token_shape(tokenizer, wrong_prompt, str(row["chosen_response"]))["total_tokens"]
            <= max_length
            and token_shape(tokenizer, wrong_prompt, str(row["rejected_response"]))["total_tokens"]
            <= max_length
            for row, wrong_prompt in zip(rows, wrong, strict=True)
        ):
            return wrong
    raise M0ValidationError("no length-valid wrong-prompt derangement exists")


def prepare(path: Path) -> dict[str, Any]:
    protocol = context(path)
    from transformers import AutoTokenizer

    cfg = protocol["design"]
    seed = int(cfg["selection_seed"])
    max_length = int(cfg["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_repo_path(protocol["model"]["checkpoint_dir"])),
        local_files_only=True,
        trust_remote_code=True,
    )

    source = protocol["source_paths"]
    ledger = read_jsonl(resolve_repo_path(source["candidate_ledger"]))
    author_packets = read_jsonl(resolve_repo_path(source["author_a_packet"])) + read_jsonl(
        resolve_repo_path(source["author_b_packet"])
    )
    prompt_by_author_id = {
        str(row["opaque_authoring_id"]): str(row["prompt"]) for row in author_packets
    }
    safety_packet = {
        str(row["opaque_safety_id"]): row
        for row in read_jsonl(resolve_repo_path(source["p2_safety_packet"]))
    }
    safety_key = read_jsonl(resolve_repo_path(source["p2_safety_key"]))
    rejected_by_pair = {}
    for key in safety_key:
        if key["is_hidden_duplicate"] or key["response_role"] != "rejected":
            continue
        visible = safety_packet[str(key["opaque_safety_id"])]
        rejected_by_pair[str(key["pair_id"])] = visible

    candidates = []
    for row in choose_candidate_per_pair(
        [
            row
            for row in ledger
            if row["prospective_split"] == "mechanism_dev" and row["mechanically_valid"]
        ],
        seed,
    ):
        author_id = str(row["opaque_authoring_id"])
        pair_id = str(row["pair_id"])
        rejected = rejected_by_pair.get(pair_id)
        if author_id not in prompt_by_author_id or rejected is None:
            raise M0ValidationError("preference source join is incomplete")
        prompt = prompt_by_author_id[author_id]
        if prompt != str(rejected["prompt"]):
            raise M0ValidationError("same-prompt preference join mismatch")
        chosen = str(row["response"])
        rejected_response = str(rejected["response"])
        chosen_shape = token_shape(tokenizer, prompt, chosen)
        rejected_shape = token_shape(tokenizer, prompt, rejected_response)
        if (
            chosen_shape["total_tokens"] > max_length
            or rejected_shape["total_tokens"] > max_length
            or chosen_shape["response_tokens"] <= 0
            or rejected_shape["response_tokens"] <= 0
        ):
            continue
        candidates.append(
            {
                "pair_id": pair_id,
                "candidate_id": author_id,
                "category": str(row["assigned_category"]),
                "prompt": prompt,
                "chosen_response": chosen,
                "rejected_response": rejected_response,
                "chosen_shape": chosen_shape,
                "rejected_shape": rejected_shape,
                "behaviorally_qualified": False,
            }
        )
    selected_pairs = select_category_diverse(
        candidates,
        int(cfg["preference_pairs"]),
        seed,
        "category",
        "pair_id",
        "preference",
    )
    wrong_prompts = find_wrong_prompt_derangement(selected_pairs, tokenizer, max_length)
    for row, wrong_prompt in zip(selected_pairs, wrong_prompts, strict=True):
        row["wrong_prompt"] = wrong_prompt
        row["wrong_chosen_shape"] = token_shape(
            tokenizer, wrong_prompt, str(row["chosen_response"])
        )
        row["wrong_rejected_shape"] = token_shape(
            tokenizer, wrong_prompt, str(row["rejected_response"])
        )

    safety_candidates = []
    for row in read_jsonl(resolve_repo_path(source["safety_targets"])):
        shape = token_shape(tokenizer, str(row["prompt"]), str(row["response"]))
        if shape["total_tokens"] <= max_length and shape["response_tokens"] > 0:
            safety_candidates.append(
                {
                    "target_id": str(row["target_id"]),
                    "category": str(row["category"]),
                    "prompt": str(row["prompt"]),
                    "response": str(row["response"]),
                    "shape": shape,
                }
            )
    selected_safety = select_category_diverse(
        safety_candidates,
        int(cfg["safety_rows"]),
        seed,
        "category",
        "target_id",
        "safety",
    )

    outputs = protocol["outputs"]
    pair_path = resolve_repo_path(outputs["preference_data"])
    safety_path = resolve_repo_path(outputs["safety_data"])
    write_jsonl_frozen(pair_path, selected_pairs)
    write_jsonl_frozen(safety_path, selected_safety)
    artifacts = [
        {
            "path": relative(pair_path),
            "sha256": sha256(pair_path),
            "bytes": pair_path.stat().st_size,
            "rows": len(selected_pairs),
        },
        {
            "path": relative(safety_path),
            "sha256": sha256(safety_path),
            "bytes": safety_path.stat().st_size,
            "rows": len(selected_safety),
        },
    ]
    checkpoint = resolve_repo_path(protocol["model"]["checkpoint_dir"])
    checkpoint_files = [
        {
            "path": relative(file),
            "sha256": sha256(file),
            "bytes": file.stat().st_size,
        }
        for file in sorted(checkpoint.iterdir())
        if file.is_file()
    ]
    manifest = {
        "name": "step3_cspr_m0_data_freeze_v1",
        "route": "CSPR_M0_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY",
        "entry_commit": protocol["entry_commit"],
        "artifacts": artifacts,
        "checkpoint_files": checkpoint_files,
        "preference_rows": len(selected_pairs),
        "preference_categories": sorted({row["category"] for row in selected_pairs}),
        "safety_rows": len(selected_safety),
        "safety_categories": sorted({row["category"] for row in selected_safety}),
        "all_sequences_fit_without_truncation": True,
        "human_labels_used": False,
        "behavioral_qualification_claimed": False,
        "wrong_prompt_is_deranged": all(
            row["wrong_prompt"] != row["prompt"] for row in selected_pairs
        ),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["data_manifest"]), manifest)
    report = """# CSPR M0 Data and Execution Freeze\n\nRoute: `CSPR_M0_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY`.\n\nEight category-diverse mechanism-dev preference pairs and eight category-diverse constant20 safety rows were selected by the frozen token-length and stable-hash rules. All true, swapped, wrong-prompt, and safety sequences fit the 256-token encoding without truncation. No human label is used and no candidate is claimed behaviorally qualified.\n\nThe frozen package authorizes one real-model zero-update mechanism run only after commit and push. It does not authorize an optimizer, checkpoint write, generation, target qualification, core100, heldout, or training. `optimizer_steps=0` and `training_authorized=false`.\n"""
    write_frozen(resolve_repo_path(outputs["freeze_report"]), report)
    print(json.dumps({"route": manifest["route"], "preference_rows": 8, "safety_rows": 8}, sort_keys=True))
    return manifest


def verify_prepared(protocol: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_repo_path(protocol["outputs"]["data_manifest"])
    manifest = read_json(manifest_path)
    if manifest.get("route") != "CSPR_M0_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY":
        raise M0ValidationError("M0 data-freeze route mismatch")
    for artifact in manifest["artifacts"]:
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise M0ValidationError(f"prepared artifact hash mismatch: {artifact['path']}")
        if len(read_jsonl(path)) != artifact["rows"]:
            raise M0ValidationError(f"prepared artifact row mismatch: {artifact['path']}")
    for artifact in manifest["checkpoint_files"]:
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise M0ValidationError(f"checkpoint hash mismatch: {artifact['path']}")
    if manifest.get("preference_rows") != 8 or manifest.get("safety_rows") != 8:
        raise M0ValidationError("M0 prepared row counts differ from the frozen design")
    return manifest


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_tracked_freeze(protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise M0ValidationError("M0 run requires a clean worktree")
    paths = [
        protocol["outputs"]["preference_data"],
        protocol["outputs"]["safety_data"],
        protocol["outputs"]["data_manifest"],
        relative(resolve_repo_path(DEFAULT_PROTOCOL)),
    ]
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    manifest = verify_prepared(protocol)
    result = {
        "status": "PASS_CSPR_M0_PREFLIGHT",
        "entry_commit": protocol["entry_commit"],
        "preference_rows": manifest["preference_rows"],
        "safety_rows": manifest["safety_rows"],
        "human_labels_used": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    return sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )


def flatten_gradients(parameters: list[tuple[str, Any]]) -> tuple[Any, dict[str, Any]]:
    import torch

    chunks = []
    missing = []
    finite = True
    nonzero = 0
    for name, parameter in parameters:
        if parameter.grad is None:
            missing.append(name)
            chunk = torch.zeros(parameter.numel(), dtype=torch.float64)
        else:
            chunk = parameter.grad.detach().cpu().reshape(-1).to(torch.float64)
            finite = finite and bool(torch.isfinite(chunk).all())
            nonzero += int(bool(torch.count_nonzero(chunk)))
        chunks.append(chunk)
    flat = torch.cat(chunks)
    return flat, {
        "parameter_tensors": len(parameters),
        "parameter_elements": flat.numel(),
        "missing_gradient_tensors": missing,
        "finite": finite,
        "nonzero_elements": nonzero,
    }


def gradient_hash(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def trainable_state_hash(parameters: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(protocol: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    model_cfg = protocol["model"]
    set_seed(int(protocol["design"]["model_seed"]))
    checkpoint = resolve_repo_path(model_cfg["checkpoint_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def response_feature(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, list[int]]:
    shape = token_shape(tokenizer, prompt, response)
    if shape["total_tokens"] > max_length or shape["response_tokens"] <= 0:
        raise M0ValidationError("runtime tokenization violates the frozen no-truncation rule")
    feature = training.response_feature(tokenizer, prompt, response, max_length)
    if sum(label != -100 for label in feature["labels"]) != shape["response_tokens"]:
        raise M0ValidationError("runtime response mask differs from the frozen token shape")
    return feature


def model_device(model: Any) -> Any:
    return next(parameter.device for parameter in model.parameters() if parameter.device.type == "cuda")


def sequence_log_probability(model: Any, feature: dict[str, list[int]]) -> Any:
    import torch

    batch = training.batch_feature(feature, model_device(model), include_labels=True)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    targets = labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum()


def preference_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    features: list[tuple[dict[str, list[int]], dict[str, list[int]]]],
    beta: float,
) -> tuple[Any, dict[str, Any], list[tuple[float, float]]]:
    model.zero_grad(set_to_none=True)
    baseline = []
    scale = 1.0 / len(features)
    for chosen, rejected in features:
        chosen_logp = sequence_log_probability(model, chosen)
        baseline_chosen = float(chosen_logp.detach().cpu())
        (-0.5 * beta * scale * chosen_logp).backward()
        rejected_logp = sequence_log_probability(model, rejected)
        baseline_rejected = float(rejected_logp.detach().cpu())
        (0.5 * beta * scale * rejected_logp).backward()
        baseline.append((baseline_chosen, baseline_rejected))
    gradient, coverage = flatten_gradients(parameters)
    model.zero_grad(set_to_none=True)
    return gradient, coverage, baseline


def safety_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    features: list[dict[str, list[int]]],
) -> tuple[Any, dict[str, Any], float]:
    model.zero_grad(set_to_none=True)
    losses = []
    for feature in features:
        batch = training.batch_feature(feature, model_device(model), include_labels=True)
        loss = model(**batch, use_cache=False, return_dict=True).loss
        losses.append(float(loss.detach().cpu()))
        (loss / len(features)).backward()
    gradient, coverage = flatten_gradients(parameters)
    model.zero_grad(set_to_none=True)
    return gradient, coverage, sum(losses) / len(losses)


def cosine(left: Any, right: Any) -> float:
    import torch

    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not bool(denominator > 0):
        return math.nan
    return float(torch.dot(left, right) / denominator)


def pair_projection_metrics(preference: Any, safety: Any) -> dict[str, Any]:
    import torch

    pp = float(torch.dot(preference, preference))
    ss = float(torch.dot(safety, safety))
    ps = float(torch.dot(preference, safety))
    active = ps < 0.0
    retained = 1.0 if not active else max(0.0, 1.0 - (ps * ps) / (pp * ss))
    return {
        "active": active,
        "preference_norm": math.sqrt(pp),
        "safety_cosine": ps / math.sqrt(pp * ss),
        "retained_preference_fraction": retained,
    }


def parameter_snapshots(parameters: list[tuple[str, Any]]) -> list[Any]:
    return [parameter.detach().cpu().clone() for _, parameter in parameters]


def restore_parameters(parameters: list[tuple[str, Any]], snapshots: list[Any]) -> None:
    import torch

    with torch.no_grad():
        for (_, parameter), snapshot in zip(parameters, snapshots, strict=True):
            parameter.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))


def apply_direction(
    parameters: list[tuple[str, Any]], snapshots: list[Any], direction: Any, alpha: float
) -> None:
    import torch

    offset = 0
    with torch.no_grad():
        for (_, parameter), snapshot in zip(parameters, snapshots, strict=True):
            count = parameter.numel()
            delta = direction[offset : offset + count].reshape(parameter.shape)
            parameter.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))
            parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype), alpha=alpha)
            offset += count
    if offset != direction.numel():
        raise M0ValidationError("flat direction does not match trainable parameters")


def evaluate_preference_loss(
    model: Any,
    features: list[tuple[dict[str, list[int]], dict[str, list[int]]]],
    references: list[tuple[float, float]],
    beta: float,
) -> float:
    import torch

    losses = []
    with torch.no_grad():
        for (chosen, rejected), (ref_chosen, ref_rejected) in zip(
            features, references, strict=True
        ):
            losses.append(
                float(
                    cspr.dpo_pair_loss(
                        sequence_log_probability(model, chosen),
                        sequence_log_probability(model, rejected),
                        ref_chosen,
                        ref_rejected,
                        beta,
                    ).cpu()
                )
            )
    return sum(losses) / len(losses)


def evaluate_safety_loss(model: Any, features: list[dict[str, list[int]]]) -> float:
    import torch

    losses = []
    with torch.no_grad():
        for feature in features:
            batch = training.batch_feature(feature, model_device(model), include_labels=True)
            losses.append(float(model(**batch, use_cache=False, return_dict=True).loss.cpu()))
    return sum(losses) / len(losses)


def line_search(
    model: Any,
    parameters: list[tuple[str, Any]],
    snapshots: list[Any],
    direction: Any,
    preference_features: list[tuple[dict[str, list[int]], dict[str, list[int]]]],
    safety_features: list[dict[str, list[int]]],
    references: list[tuple[float, float]],
    beta: float,
    initial_preference: float,
    initial_safety: float,
    preference_derivative: float,
    safety_derivative: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    alpha = float(config["initial_step_size"])
    trials = []
    accepted = False
    accepted_alpha = 0.0
    try:
        for trial in range(1, int(config["maximum_trials"]) + 1):
            apply_direction(parameters, snapshots, direction, alpha)
            preference = evaluate_preference_loss(
                model, preference_features, references, beta
            )
            safety = evaluate_safety_loss(model, safety_features)
            preference_bound = initial_preference + float(config["armijo_fraction"]) * alpha * preference_derivative
            safety_bound = initial_safety + float(config["armijo_fraction"]) * alpha * min(safety_derivative, 0.0)
            preference_pass = preference <= preference_bound + float(config["comparison_tolerance"])
            safety_pass = safety <= safety_bound + float(config["comparison_tolerance"])
            trials.append(
                {
                    "trial": trial,
                    "step_size": alpha,
                    "preference_loss": preference,
                    "safety_loss": safety,
                    "preference_bound": preference_bound,
                    "safety_bound": safety_bound,
                    "preference_pass": preference_pass,
                    "safety_pass": safety_pass,
                }
            )
            if preference_pass and safety_pass:
                accepted = True
                accepted_alpha = alpha
                break
            alpha *= float(config["contraction"])
    finally:
        restore_parameters(parameters, snapshots)
    return {"accepted": accepted, "accepted_step_size": accepted_alpha, "trials": trials}


def select_route(gates: dict[str, bool]) -> str:
    if (
        not gates["structural_and_gradient_valid"]
        or not gates["state_restored"]
        or not gates["zero_optimizer"]
    ):
        return "STOP_CSPR_M0_INVALID"
    if not gates["label_swap_control"]:
        return "STOP_CSPR_M0_DPO_CONTROL_FAILED"
    if not gates["pair_activation"] or not gates["aggregate_activation"]:
        return "STOP_CSPR_M0_PROJECTION_NOT_EXERCISED"
    if not gates["retained_preference"]:
        return "STOP_CSPR_M0_PROJECTED_DIRECTION_DEGENERATE"
    if not gates["directional_and_kkt"]:
        return "STOP_CSPR_M0_INVALID"
    if not gates["finite_joint_step"]:
        return "STOP_CSPR_M0_NO_FINITE_JOINT_STEP"
    return "PASS_CSPR_M0_REAL_MODEL_MECHANISM_FEASIBLE"


def run(path: Path) -> dict[str, Any]:
    import torch

    protocol = context(path)
    manifest = verify_prepared(protocol)
    runtime_head = require_clean_tracked_freeze(protocol)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(protocol)
    parameters = trainable_parameters(model)
    if not parameters:
        raise M0ValidationError("model has no trainable LoRA parameters")
    initial_hash = trainable_state_hash(parameters)
    snapshots = parameter_snapshots(parameters)

    cfg = protocol["design"]
    max_length = int(cfg["max_length"])
    beta = float(cfg["beta"])
    pair_rows = read_jsonl(resolve_repo_path(protocol["outputs"]["preference_data"]))
    safety_rows = read_jsonl(resolve_repo_path(protocol["outputs"]["safety_data"]))
    preference_features = [
        (
            response_feature(tokenizer, row["prompt"], row["chosen_response"], max_length),
            response_feature(tokenizer, row["prompt"], row["rejected_response"], max_length),
        )
        for row in pair_rows
    ]
    swapped_features = [(rejected, chosen) for chosen, rejected in preference_features]
    wrong_features = [
        (
            response_feature(tokenizer, row["wrong_prompt"], row["chosen_response"], max_length),
            response_feature(tokenizer, row["wrong_prompt"], row["rejected_response"], max_length),
        )
        for row in pair_rows
    ]
    safety_features = [
        response_feature(tokenizer, row["prompt"], row["response"], max_length)
        for row in safety_rows
    ]

    safety_grad, safety_coverage, initial_safety = safety_gradient(
        model, parameters, safety_features
    )
    aggregate_preference = torch.zeros_like(safety_grad)
    references = []
    pair_metrics = []
    for row, features in zip(pair_rows, preference_features, strict=True):
        pair_grad, coverage, baseline = preference_gradient(
            model, parameters, [features], beta
        )
        aggregate_preference += pair_grad / len(preference_features)
        references.extend(baseline)
        pair_metrics.append(
            {
                "pair_id": row["pair_id"],
                **pair_projection_metrics(pair_grad, safety_grad),
                "gradient_hash": gradient_hash(pair_grad),
                "coverage": coverage,
            }
        )
        del pair_grad
    preference_coverage = {
        "parameter_tensors": len(parameters),
        "parameter_elements": aggregate_preference.numel(),
        "missing_gradient_tensors": sorted(
            {name for row in pair_metrics for name in row["coverage"]["missing_gradient_tensors"]}
        ),
        "finite": all(row["coverage"]["finite"] for row in pair_metrics),
        "nonzero_elements": int(torch.count_nonzero(aggregate_preference)),
    }

    swapped_grad, swapped_coverage, _ = preference_gradient(
        model, parameters, swapped_features, beta
    )
    wrong_grad, wrong_coverage, _ = preference_gradient(
        model, parameters, wrong_features, beta
    )
    preference_norm = float(torch.linalg.vector_norm(aggregate_preference))
    safety_norm = float(torch.linalg.vector_norm(safety_grad))
    swap_norm = float(torch.linalg.vector_norm(swapped_grad))
    label_swap_cosine = cosine(aggregate_preference, swapped_grad)
    label_swap_norm_ratio = swap_norm / preference_norm if preference_norm else math.nan
    wrong_cosine = cosine(aggregate_preference, wrong_grad)

    diagonal = torch.ones_like(aggregate_preference)
    certificate = cspr.project_preconditioned_preference_direction(
        aggregate_preference, safety_grad, diagonal
    )
    retained = (
        -certificate.preference_directional_derivative / (preference_norm * preference_norm)
        if preference_norm
        else math.nan
    )
    initial_preference = evaluate_preference_loss(
        model, preference_features, references, beta
    )
    search = line_search(
        model,
        parameters,
        snapshots,
        certificate.direction,
        preference_features,
        safety_features,
        references,
        beta,
        initial_preference,
        initial_safety,
        certificate.preference_directional_derivative,
        certificate.safety_directional_derivative,
        protocol["line_search"],
    )
    unconstrained_at_accepted = None
    if search["accepted"]:
        alpha = float(search["accepted_step_size"])
        apply_direction(parameters, snapshots, certificate.unconstrained_direction, alpha)
        try:
            unconstrained_at_accepted = {
                "step_size": alpha,
                "preference_loss": evaluate_preference_loss(
                    model, preference_features, references, beta
                ),
                "safety_loss": evaluate_safety_loss(model, safety_features),
            }
        finally:
            restore_parameters(parameters, snapshots)
    final_hash = trainable_state_hash(parameters)

    tolerance = float(protocol["gates"]["normalized_residual_max"])
    direction_norm = float(torch.linalg.vector_norm(certificate.direction))
    safety_scale = max(safety_norm * direction_norm, 1e-30)
    preference_scale = max(preference_norm * direction_norm, 1e-30)
    normalized = {
        "positive_safety_derivative": max(certificate.safety_directional_derivative, 0.0) / safety_scale,
        "positive_preference_derivative": max(certificate.preference_directional_derivative, 0.0) / preference_scale,
        "kkt_stationarity": certificate.kkt_stationarity_norm / max(direction_norm, 1e-30),
        "kkt_complementarity": certificate.kkt_complementarity_abs / max(
            abs(certificate.multiplier) * safety_scale, 1e-30
        ),
    }
    all_coverage = [preference_coverage, safety_coverage, swapped_coverage, wrong_coverage]
    structural = (
        preference_norm > 0
        and safety_norm > 0
        and all(row["finite"] and not row["missing_gradient_tensors"] for row in all_coverage)
        and all(math.isfinite(value) for value in (
            preference_norm,
            safety_norm,
            label_swap_cosine,
            label_swap_norm_ratio,
            wrong_cosine,
            retained,
            initial_preference,
            initial_safety,
        ))
    )
    gate_cfg = protocol["gates"]
    gates = {
        "structural_and_gradient_valid": structural,
        "pair_activation": sum(row["active"] for row in pair_metrics)
        >= int(gate_cfg["minimum_active_pairs"]),
        "aggregate_activation": certificate.active,
        "retained_preference": retained >= float(gate_cfg["retained_preference_min"]),
        "directional_and_kkt": max(normalized.values()) <= tolerance,
        "label_swap_control": label_swap_cosine <= float(gate_cfg["label_swap_cosine_max"])
        and float(gate_cfg["label_swap_norm_ratio_min"])
        <= label_swap_norm_ratio
        <= float(gate_cfg["label_swap_norm_ratio_max"]),
        "finite_joint_step": bool(search["accepted"]),
        "state_restored": final_hash == initial_hash,
        "zero_optimizer": True,
    }
    route = select_route(gates)
    report = {
        "name": "step3_cspr_m0_real_model_mechanism_v1",
        "route": route,
        "runtime_head": runtime_head,
        "data_manifest_sha256": sha256(resolve_repo_path(protocol["outputs"]["data_manifest"])),
        "model": protocol["model"],
        "design": protocol["design"],
        "preference": {
            "initial_loss": initial_preference,
            "gradient_norm": preference_norm,
            "gradient_hash": gradient_hash(aggregate_preference),
            "coverage": preference_coverage,
        },
        "safety": {
            "initial_loss": initial_safety,
            "gradient_norm": safety_norm,
            "gradient_hash": gradient_hash(safety_grad),
            "coverage": safety_coverage,
        },
        "projection": {
            **certificate.scalars(),
            "preference_safety_cosine": cosine(aggregate_preference, safety_grad),
            "direction_norm": direction_norm,
            "retained_preference_fraction": retained,
            "normalized_residuals": normalized,
        },
        "pairwise_projection": pair_metrics,
        "controls": {
            "label_swap_cosine": label_swap_cosine,
            "label_swap_norm_ratio": label_swap_norm_ratio,
            "label_swap_gradient_hash": gradient_hash(swapped_grad),
            "wrong_prompt_cosine": wrong_cosine,
            "wrong_prompt_norm_ratio": float(torch.linalg.vector_norm(wrong_grad)) / preference_norm,
            "wrong_prompt_gradient_hash": gradient_hash(wrong_grad),
        },
        "line_search": search,
        "unconstrained_at_projected_accepted_step": unconstrained_at_accepted,
        "gates": gates,
        "initial_trainable_state_hash": initial_hash,
        "final_trainable_state_hash": final_hash,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_written": False,
        "candidate_outputs_generated": False,
        "human_labels_used": False,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    outputs = protocol["outputs"]
    write_json_frozen(resolve_repo_path(outputs["diagnostics"]), report)
    environment = {
        "name": "step3_cspr_m0_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_clock_seconds": time.time() - started,
        "runtime_head": runtime_head,
        "pid": os.getpid(),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["environment"]), environment)
    route_record = {
        "name": "step3_cspr_m0_route_v1",
        "route": route,
        "gates": gates,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized": (
            "consider_resuming_external_human_P2U_measurement_only"
            if route == "PASS_CSPR_M0_REAL_MODEL_MECHANISM_FEASIBLE"
            else "stop_CSPR_mechanism_route"
        ),
    }
    write_json_frozen(resolve_repo_path(outputs["route"]), route_record)
    markdown = f"""# CSPR M0 Real-Model Zero-Update Report\n\nRoute: `{route}`.\n\nThe aggregate projection was active: `{certificate.active}`. Pairwise active projections: `{sum(row['active'] for row in pair_metrics)}/8`. Retained preference fraction: `{retained:.6f}`. Label-swap cosine: `{label_swap_cosine:.6f}`. Projected dual-Armijo step accepted: `{search['accepted']}`.\n\nThis is a real-model mechanism-feasibility result on one frozen exploratory batch. It is not a behavior, safety-generalization, or target-quality result. No optimizer or scheduler was created, no checkpoint or candidate output was written, and the exact initial trainable-state hash was restored. `optimizer_steps=0` and `training_authorized=false`.\n"""
    write_frozen(resolve_repo_path(outputs["report"]), markdown)
    print(json.dumps({"route": route, "active_pairs": sum(row["active"] for row in pair_metrics), "aggregate_active": certificate.active, "retained": retained, "line_search_accepted": search["accepted"]}, sort_keys=True))
    return report


def verify_run(path: Path) -> dict[str, Any]:
    protocol = context(path)
    verify_prepared(protocol)
    report = read_json(resolve_repo_path(protocol["outputs"]["diagnostics"]))
    route = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    if report.get("route") != route.get("route"):
        raise M0ValidationError("M0 report and route disagree")
    if report.get("initial_trainable_state_hash") != report.get("final_trainable_state_hash"):
        raise M0ValidationError("M0 trainable state was not restored")
    if report.get("optimizer_steps") != 0 or report.get("training_authorized") is not False:
        raise M0ValidationError("M0 result violates zero-training boundary")
    result = {
        "status": "PASS_CSPR_M0_RESULT_VERIFICATION",
        "route": report["route"],
        "state_restored": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "prepare":
            prepare(path)
        elif args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        FileExistsError,
        KeyError,
        M0ValidationError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_M0_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
