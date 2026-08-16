"""Run the prospective CITA-P0 zero-update real-model mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import step3_cita_common as cita


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cita_p0/"
    "protocol/cita_p0_protocol.json"
)


class CitaP0ValidationError(ValueError):
    """Raised when an input or result violates the frozen CITA-P0 protocol."""


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
        raise CitaP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                json_safe(row), sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def write_frozen_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen_bytes(path, json_bytes(value))


def write_jsonl_frozen(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_frozen_bytes(path, jsonl_bytes(rows))


def write_text_frozen(path: Path, value: str) -> None:
    write_frozen_bytes(path, value.encode("utf-8"))


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_packet_rows(
    rows: list[dict[str, Any]], expected_rows: int, expected_prompts: int,
    expected_categories: int,
) -> dict[str, Any]:
    required = {
        "ranking_id",
        "sample_id",
        "target_id",
        "category",
        "correct_prompt",
        "correct_prompt_sha256",
        "wrong_prompts",
        "wrong_prompt_sha256",
        "response",
        "response_sha256",
    }
    if len(rows) != expected_rows:
        raise CitaP0ValidationError(
            f"mismatch packet has {len(rows)} rows, expected {expected_rows}"
        )
    ranking_ids: set[str] = set()
    target_ids: set[str] = set()
    for row in rows:
        if not required.issubset(row):
            raise CitaP0ValidationError("mismatch packet row lacks required fields")
        ranking_id = str(row["ranking_id"])
        target_id = str(row["target_id"])
        if ranking_id in ranking_ids or target_id in target_ids:
            raise CitaP0ValidationError("mismatch packet IDs are not unique")
        ranking_ids.add(ranking_id)
        target_ids.add(target_id)
        wrong_prompts = list(row["wrong_prompts"])
        wrong_hashes = list(row["wrong_prompt_sha256"])
        if len(wrong_prompts) != 2 or len(wrong_hashes) != 2:
            raise CitaP0ValidationError("each row must contain exactly two wrong prompts")
        if any(prompt == row["correct_prompt"] for prompt in wrong_prompts):
            raise CitaP0ValidationError("wrong prompt equals the correct prompt")
        if text_sha256(str(row["correct_prompt"])) != row["correct_prompt_sha256"]:
            raise CitaP0ValidationError("correct-prompt hash mismatch")
        if [text_sha256(str(prompt)) for prompt in wrong_prompts] != wrong_hashes:
            raise CitaP0ValidationError("wrong-prompt hash mismatch")
        if text_sha256(str(row["response"])) != row["response_sha256"]:
            raise CitaP0ValidationError("response hash mismatch")
    prompts = {str(row["sample_id"]) for row in rows}
    categories = {str(row["category"]) for row in rows}
    if len(prompts) != expected_prompts or len(categories) != expected_categories:
        raise CitaP0ValidationError("packet prompt/category counts differ from protocol")
    return {
        "rows": len(rows),
        "prompt_clusters": len(prompts),
        "categories": sorted(categories),
    }


def frozen_splits(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    selection = protocol["selection"]
    splits = cita.select_prompt_disjoint_rows(
        rows,
        int(selection["calibration_prompts"]),
        int(selection["validation_prompts"]),
        int(selection["seed"]),
    )
    for split_name in ("calibration", "validation"):
        observed = [str(row["target_id"]) for row in splits[split_name]]
        expected = list(selection[f"{split_name}_target_ids"])
        if observed != expected:
            raise CitaP0ValidationError(f"{split_name} split differs from frozen IDs")
    calibration_prompts = {str(row["sample_id"]) for row in splits["calibration"]}
    validation_prompts = {str(row["sample_id"]) for row in splits["validation"]}
    if calibration_prompts.intersection(validation_prompts):
        raise CitaP0ValidationError("calibration and validation prompts overlap")
    return splits


def context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_CITA_model_score_or_gradient":
        raise CitaP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise CitaP0ValidationError(f"frozen input hash mismatch: {name}")
    if protocol.get("optimizer_steps") != 0 or protocol.get("training_authorized") is not False:
        raise CitaP0ValidationError("CITA-P0 violates the zero-training boundary")
    if protocol.get("core100_opened") is not False or protocol.get("heldout_opened") is not False:
        raise CitaP0ValidationError("CITA-P0 opens a sealed evaluation set")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise CitaP0ValidationError("entry commit is not an ancestor of HEAD")
    packet_path = resolve_repo_path(protocol["source"]["mismatch_packet"])
    rows = read_jsonl(packet_path)
    counts = protocol["source"]["counts"]
    validate_packet_rows(
        rows,
        int(counts["rows"]),
        int(counts["prompt_clusters"]),
        int(counts["categories"]),
    )
    frozen_splits(rows, protocol)
    return protocol, rows


def require_clean_tracked_freeze(protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise CitaP0ValidationError("CITA-P0 run requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.append(DEFAULT_PROTOCOL)
    for path in tracked:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    for output in protocol["outputs"].values():
        if resolve_repo_path(output).exists():
            raise CitaP0ValidationError(f"prospective output already exists: {output}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    splits = frozen_splits(rows, protocol)
    if not torch.cuda.is_available():
        raise CitaP0ValidationError("CUDA is unavailable")
    result = {
        "status": "PASS_CITA_P0_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "packet_rows": len(rows),
        "prompt_clusters": len({row["sample_id"] for row in rows}),
        "calibration_prompts": len(splits["calibration"]),
        "validation_prompts": len(splits["validation"]),
        "gpu": torch.cuda.get_device_name(0),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def load_model(protocol: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    model_cfg = protocol["model"]
    set_seed(int(model_cfg["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        local_files_only=bool(model_cfg["local_files_only"]),
        trust_remote_code=True,
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
        model_cfg["model_name_or_path"],
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=bool(model_cfg["local_files_only"]),
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=bool(model_cfg["gradient_checkpointing"])
    )
    set_seed(int(model_cfg["seed"]))
    lora = LoraConfig(
        r=int(model_cfg["lora_rank"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        lora_dropout=float(model_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(model_cfg["target_modules"]),
    )
    model = get_peft_model(base, lora)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def model_device(model: Any) -> Any:
    return next(parameter.device for parameter in model.parameters() if parameter.device.type == "cuda")


def serialized_response(
    tokenizer: Any, prompt: str, response: str, max_length: int
) -> dict[str, Any]:
    user = [{"role": "user", "content": prompt}]
    conversation = [*user, {"role": "assistant", "content": response}]
    direct_text = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True
    )
    forced_text = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    direct = tokenizer(direct_text, add_special_tokens=False)["input_ids"]
    forced = tokenizer(forced_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full) > max_length:
        raise CitaP0ValidationError("serialized sequence exceeds max_length")
    if full[: len(forced)] != forced or len(forced) <= len(direct):
        raise CitaP0ValidationError("no-think generation prefix is not an exact prefix")
    suffix = full[len(forced) :]
    try:
        semantic_mask = cita.semantic_response_mask(suffix, tokenizer.all_special_ids)
    except ValueError as exc:
        raise CitaP0ValidationError(str(exc)) from exc
    return {
        "input_ids": full,
        "prefix_tokens": len(forced),
        "direct_prefix_tokens": len(direct),
        "response_token_ids": suffix,
        "semantic_mask": semantic_mask,
    }


def token_nll(model: Any, feature: dict[str, Any]) -> list[float]:
    import torch
    import torch.nn.functional as functional

    device = model_device(model)
    ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    prefix = int(feature["prefix_tokens"])
    with torch.no_grad():
        logits = model(
            input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True
        ).logits
        response_logits = logits[0, prefix - 1 : ids.shape[1] - 1].float()
        targets = ids[0, prefix:]
        losses = functional.cross_entropy(response_logits, targets, reduction="none")
    return [float(value) for value in losses.detach().cpu()]


def build_weight_cache(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    design = protocol["design"]
    maximum = int(design["max_length"])
    cache = []
    for position, row in enumerate(rows, start=1):
        prompts = [str(row["correct_prompt"]), *map(str, row["wrong_prompts"])]
        features = [
            serialized_response(tokenizer, prompt, str(row["response"]), maximum)
            for prompt in prompts
        ]
        suffixes = [feature["response_token_ids"] for feature in features]
        masks = [feature["semantic_mask"] for feature in features]
        if suffixes[1:] != suffixes[:1] * 2 or masks[1:] != masks[:1] * 2:
            raise CitaP0ValidationError("correct/wrong response token suffixes do not align")
        nlls = [token_nll(model, feature) for feature in features]
        weights = cita.token_weight_rows(
            nlls[0],
            nlls[1:],
            masks[0],
            float(design["minimum_weight"]),
            float(design["maximum_weight"]),
        )
        advantages = list(weights["advantages"])
        sham_scores = list(weights["sham_scores"])
        rank_correlation = cita.pearson_ranks(advantages, sham_scores)
        cache.append(
            {
                "ranking_id": row["ranking_id"],
                "sample_id": row["sample_id"],
                "target_id": row["target_id"],
                "category": row["category"],
                "response_sha256": row["response_sha256"],
                "response_token_ids": suffixes[0],
                "semantic_mask": masks[0],
                "semantic_tokens": sum(masks[0]),
                "serialized_termination_tokens": len(masks[0]) - sum(masks[0]),
                "correct_nll": nlls[0],
                "wrong_nlls": nlls[1:],
                "advantages": advantages,
                "sham_scores": sham_scores,
                "mean_token_advantage": sum(advantages) / len(advantages),
                "positive_token_fraction": sum(value > 0.0 for value in advantages)
                / len(advantages),
                "cita_sham_rank_correlation": rank_correlation,
                "uniform_weights": weights["uniform_weights"],
                "cita_weights": weights["cita_weights"],
                "sham_weights": weights["sham_weights"],
                "semantic_cita_weights": weights["semantic_cita_weights"],
                "semantic_sham_weights": weights["semantic_sham_weights"],
                "weight_multiset_equal": sorted(weights["semantic_cita_weights"])
                == sorted(weights["semantic_sham_weights"]),
                "token_alignment_exact": True,
            }
        )
        if position % 10 == 0 or position == len(rows):
            print(f"CITA-P0 reference scoring {position}/{len(rows)}", flush=True)
    return cache


def validate_weight_cache(
    cache: list[dict[str, Any]], rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    if len(cache) != len(rows):
        raise CitaP0ValidationError("weight cache row count mismatch")
    semantic_counts = []
    for source, row in zip(rows, cache, strict=True):
        if row["target_id"] != source["target_id"] or row["response_sha256"] != source["response_sha256"]:
            raise CitaP0ValidationError("weight cache source order mismatch")
        size = len(row["response_token_ids"])
        vector_keys = (
            "semantic_mask",
            "correct_nll",
            "uniform_weights",
            "cita_weights",
            "sham_weights",
        )
        if size == 0 or any(len(row[key]) != size for key in vector_keys):
            raise CitaP0ValidationError("weight cache vector lengths differ")
        if len(row["wrong_nlls"]) != 2 or any(len(values) != size for values in row["wrong_nlls"]):
            raise CitaP0ValidationError("weight cache wrong-prompt vectors differ")
        semantic = [
            weight
            for weight, keep in zip(row["cita_weights"], row["semantic_mask"], strict=True)
            if keep
        ]
        sham = [
            weight
            for weight, keep in zip(row["sham_weights"], row["semantic_mask"], strict=True)
            if keep
        ]
        if (
            len(semantic) < 2
            or sorted(semantic) != sorted(sham)
            or not math.isclose(sum(semantic) / len(semantic), 1.0, abs_tol=1e-12)
            or any(
                weight != 1.0
                for weight, keep in zip(row["cita_weights"], row["semantic_mask"], strict=True)
                if not keep
            )
            or any(
                weight != 1.0
                for weight, keep in zip(row["sham_weights"], row["semantic_mask"], strict=True)
                if not keep
            )
        ):
            raise CitaP0ValidationError("weight-cache budget invariant failed")
        numeric = [
            *row["correct_nll"],
            *row["wrong_nlls"][0],
            *row["wrong_nlls"][1],
            *row["advantages"],
            *row["sham_scores"],
            *row["cita_weights"],
            *row["sham_weights"],
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise CitaP0ValidationError("weight cache contains a non-finite value")
        semantic_counts.append(len(semantic))
    return {
        "rows": len(cache),
        "minimum_semantic_tokens": min(semantic_counts),
        "maximum_semantic_tokens": max(semantic_counts),
        "all_token_alignments_exact": all(row["token_alignment_exact"] for row in cache),
        "all_weight_multisets_equal": all(row["weight_multiset_equal"] for row in cache),
    }


def trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    return sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )


def trainable_state_hash(parameters: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_gradients(
    parameters: list[tuple[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    gradients = {}
    missing = []
    finite = True
    nonzero_tensors = 0
    nonzero_elements = 0
    for name, parameter in parameters:
        if parameter.grad is None:
            missing.append(name)
            value = torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        else:
            value = parameter.grad.detach().float().cpu().clone()
            finite = finite and bool(torch.isfinite(value).all())
            count = int(torch.count_nonzero(value))
            nonzero_elements += count
            nonzero_tensors += int(count > 0)
        gradients[name] = value
    coverage = {
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(parameter.numel() for _, parameter in parameters),
        "missing_gradient_tensors": missing,
        "finite": finite,
        "nonzero_gradient_tensors": nonzero_tensors,
        "nonzero_gradient_elements": nonzero_elements,
    }
    return gradients, coverage


def gradient_feature(
    tokenizer: Any,
    source: dict[str, Any],
    cached: dict[str, Any],
    max_length: int,
    arm: str,
) -> dict[str, list[float] | list[int]]:
    serialized = serialized_response(
        tokenizer, str(source["correct_prompt"]), str(source["response"]), max_length
    )
    if serialized["response_token_ids"] != cached["response_token_ids"]:
        raise CitaP0ValidationError("gradient tokenization differs from weight cache")
    prefix = int(serialized["prefix_tokens"])
    labels = [-100] * prefix + list(serialized["response_token_ids"])
    weights = [0.0] * prefix + list(cached[f"{arm}_weights"])
    return {
        "input_ids": list(serialized["input_ids"]),
        "labels": labels,
        "weights": weights,
    }


def aggregate_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    features: list[dict[str, Any]],
    split_name: str,
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    model.zero_grad(set_to_none=True)
    losses = []
    device = model_device(model)
    for position, feature in enumerate(features, start=1):
        ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        labels = torch.tensor([feature["labels"]], dtype=torch.long, device=device)
        weights = torch.tensor([feature["weights"]], dtype=torch.float32, device=device)
        logits = model(
            input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True
        ).logits
        loss = cita.weighted_token_loss(logits, labels, weights)
        losses.append(float(loss.detach().cpu()))
        (loss / len(features)).backward()
        del logits, loss, ids, attention, labels, weights
        if position % 4 == 0 or position == len(features):
            print(
                f"CITA-P0 gradients {split_name}/{arm} {position}/{len(features)}",
                flush=True,
            )
    gradients, coverage = capture_gradients(parameters)
    coverage.update(
        {
            "mean_loss": sum(losses) / len(losses),
            "gradient_norm": cita.gradient_norm(gradients),
            "gradient_hash": cita.gradient_hash(gradients),
        }
    )
    model.zero_grad(set_to_none=True)
    return gradients, coverage


def finite_metric(value: float) -> bool:
    return math.isfinite(value)


def gradient_comparisons(
    gradients: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, float]:
    calibration = gradients["calibration"]
    validation = gradients["validation"]
    cal_cita_delta = cita.gradient_delta(calibration["cita"], calibration["uniform"])
    cal_sham_delta = cita.gradient_delta(calibration["sham"], calibration["uniform"])
    val_cita_delta = cita.gradient_delta(validation["cita"], validation["uniform"])
    uniform_norm = cita.gradient_norm(calibration["uniform"])
    cita_delta_norm = cita.gradient_norm(cal_cita_delta)
    return {
        "calibration_cita_delta_norm": cita_delta_norm,
        "calibration_uniform_norm": uniform_norm,
        "calibration_cita_delta_to_uniform_ratio": cita_delta_norm / uniform_norm
        if uniform_norm
        else math.nan,
        "calibration_cita_uniform_cosine": cita.gradient_cosine(
            calibration["cita"], calibration["uniform"]
        ),
        "calibration_cita_sham_delta_cosine": cita.gradient_cosine(
            cal_cita_delta, cal_sham_delta
        ),
        "calibration_validation_cita_delta_cosine": cita.gradient_cosine(
            cal_cita_delta, val_cita_delta
        ),
        "sham_calibration_to_cita_validation_delta_cosine": cita.gradient_cosine(
            cal_sham_delta, val_cita_delta
        ),
    }


def package_artifacts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for name, value in protocol["outputs"].items():
        if name == "run_manifest":
            continue
        path = resolve_repo_path(value)
        artifacts.append(
            {
                "name": name,
                "path": relative(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(read_jsonl(path)) if path.suffix == ".jsonl" else None,
            }
        )
    return artifacts


def run(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    runtime_head = require_clean_tracked_freeze(protocol)
    splits = frozen_splits(rows, protocol)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(protocol)
    parameters = trainable_parameters(model)
    if not parameters or any("lora_" not in name for name, _ in parameters):
        raise CitaP0ValidationError("trainable parameter inventory is not LoRA-only")
    initial_hash = trainable_state_hash(parameters)

    cache = build_weight_cache(model, tokenizer, rows, protocol)
    cache_integrity = validate_weight_cache(cache, rows, protocol)
    cache_by_target = {str(row["target_id"]): row for row in cache}
    source_by_target = {str(row["target_id"]): row for row in rows}

    signal = {
        "row_mean_advantage_bootstrap": cita.bootstrap_mean(
            [float(row["mean_token_advantage"]) for row in cache],
            int(protocol["statistics"]["bootstrap_samples"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        ),
        "median_positive_token_fraction": cita.quantile(
            [float(row["positive_token_fraction"]) for row in cache], 0.5
        ),
        "median_absolute_cita_sham_rank_correlation": cita.quantile(
            [abs(float(row["cita_sham_rank_correlation"])) for row in cache], 0.5
        ),
    }

    gradients: dict[str, dict[str, dict[str, Any]]] = {}
    gradient_records: dict[str, dict[str, dict[str, Any]]] = {}
    max_length = int(protocol["design"]["max_length"])
    for split_name in ("calibration", "validation"):
        gradients[split_name] = {}
        gradient_records[split_name] = {}
        for arm in ("uniform", "cita", "sham"):
            features = [
                gradient_feature(
                    tokenizer,
                    source_by_target[str(row["target_id"])],
                    cache_by_target[str(row["target_id"])],
                    max_length,
                    arm,
                )
                for row in splits[split_name]
            ]
            gradient, record = aggregate_gradient(
                model, parameters, features, split_name, arm
            )
            gradients[split_name][arm] = gradient
            gradient_records[split_name][arm] = record

    comparisons = gradient_comparisons(gradients)
    final_hash = trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    gate_cfg = protocol["gates"]
    all_gradient_records = [
        gradient_records[split][arm]
        for split in ("calibration", "validation")
        for arm in ("uniform", "cita", "sham")
    ]
    gradient_valid = all(
        record["finite"]
        and not record["missing_gradient_tensors"]
        and record["gradient_norm"] > 0.0
        for record in all_gradient_records
    )
    metrics_finite = all(finite_metric(value) for value in comparisons.values())
    gates = {
        "integrity": bool(
            cache_integrity["all_token_alignments_exact"]
            and cache_integrity["all_weight_multisets_equal"]
            and headroom >= float(gate_cfg["minimum_vram_headroom_fraction"])
            and metrics_finite
        ),
        "gradient_valid": gradient_valid,
        "state_restored": final_hash == initial_hash,
        "token_signal": bool(
            signal["row_mean_advantage_bootstrap"]["ci95_lower"]
            > float(gate_cfg["row_mean_advantage_bootstrap_lower_min"])
            and signal["median_positive_token_fraction"]
            >= float(gate_cfg["median_positive_token_fraction_min"])
        ),
        "counterfactual_specificity": signal[
            "median_absolute_cita_sham_rank_correlation"
        ]
        <= float(gate_cfg["median_absolute_cita_sham_rank_correlation_max"]),
        "gradient_effect_size": bool(
            comparisons["calibration_cita_delta_to_uniform_ratio"]
            >= float(gate_cfg["cita_delta_to_uniform_norm_ratio_min"])
            and comparisons["calibration_cita_uniform_cosine"]
            <= float(gate_cfg["cita_uniform_cosine_max"])
        ),
        "sham_distinct": comparisons["calibration_cita_sham_delta_cosine"]
        <= float(gate_cfg["cita_sham_delta_cosine_max"]),
        "cross_split_replication": bool(
            comparisons["calibration_validation_cita_delta_cosine"]
            >= float(gate_cfg["calibration_validation_cita_delta_cosine_min"])
            and comparisons["calibration_validation_cita_delta_cosine"]
            - comparisons["sham_calibration_to_cita_validation_delta_cosine"]
            >= float(gate_cfg["cita_over_sham_cross_split_cosine_margin_min"])
        ),
    }
    route = cita.select_route(gates)

    split_manifest = {
        "name": "step3_cita_p0_prompt_disjoint_split_v1",
        "seed": protocol["selection"]["seed"],
        "calibration": [
            {
                "sample_id": row["sample_id"],
                "target_id": row["target_id"],
                "category": row["category"],
            }
            for row in splits["calibration"]
        ],
        "validation": [
            {
                "sample_id": row["sample_id"],
                "target_id": row["target_id"],
                "category": row["category"],
            }
            for row in splits["validation"]
        ],
        "prompt_disjoint": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    inventory = {
        "trainable_tensors": len(parameters),
        "trainable_parameters": sum(parameter.numel() for _, parameter in parameters),
        "parameters": [
            {"name": name, "numel": parameter.numel(), "dtype": str(parameter.dtype)}
            for name, parameter in parameters
        ],
    }
    diagnostics = {
        "name": "step3_cita_p0_zero_update_mechanism_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": sha256(path),
        "model": protocol["model"],
        "cache_integrity": cache_integrity,
        "signal": signal,
        "trainable_inventory": inventory,
        "gradient_records": gradient_records,
        "gradient_comparisons": comparisons,
        "gates": gates,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_written": False,
        "candidate_outputs_generated": False,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    environment = {
        "name": "step3_cita_p0_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_bytes": total_vram,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
        "runtime_head": runtime_head,
        "pid": os.getpid(),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    route_record = {
        "name": "step3_cita_p0_route_v1",
        "route": route,
        "gates": gates,
        "next_authorized": (
            "freeze_and_push_matched_three_arm_smoke_protocol"
            if route == "PASS_CITA_P0_AUTHORIZE_MATCHED_SMOKE"
            else "stop_CITA_training_and_document_negative_mechanism_result"
        ),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    markdown = f"""# CITA-P0 Zero-Update Mechanism Report

Route: `{route}`.

## Registered Evidence

- Mean token-advantage bootstrap 95% CI: `[{signal['row_mean_advantage_bootstrap']['ci95_lower']:.6f}, {signal['row_mean_advantage_bootstrap']['ci95_upper']:.6f}]`.
- Median positive-token fraction: `{signal['median_positive_token_fraction']:.6f}`.
- Median absolute CITA/sham rank correlation: `{signal['median_absolute_cita_sham_rank_correlation']:.6f}`.
- Calibration CITA-delta/uniform norm ratio: `{comparisons['calibration_cita_delta_to_uniform_ratio']:.6f}`.
- Calibration CITA/uniform cosine: `{comparisons['calibration_cita_uniform_cosine']:.6f}`.
- CITA-delta/sham-delta cosine: `{comparisons['calibration_cita_sham_delta_cosine']:.6f}`.
- Calibration/validation CITA-delta cosine: `{comparisons['calibration_validation_cita_delta_cosine']:.6f}`.
- Cross-split CITA-over-sham cosine margin: `{comparisons['calibration_validation_cita_delta_cosine'] - comparisons['sham_calibration_to_cita_validation_delta_cosine']:.6f}`.
- Peak reserved VRAM headroom: `{headroom:.6f}`.

## Interpretation Boundary

This run tests whether a fixed-budget, correct-prompt token-information ordering produces a non-degenerate and prompt-disjointly replicating LoRA gradient at the fresh seed-1 step-0 policy. It does not test generated behavior, safety, utility, or publication readiness. Fresh LoRA-A gradients may be structurally zero because LoRA-B is initialized to zero; all registered tensors must still have attached finite gradient records and the aggregate gradient must be non-zero.

No optimizer or scheduler was created, no parameter update or checkpoint was written, and the exact trainable-state hash remained unchanged. `optimizer_steps=0` and `training_authorized=false`.
"""

    outputs = protocol["outputs"]
    write_jsonl_frozen(resolve_repo_path(outputs["weight_cache"]), cache)
    write_json_frozen(resolve_repo_path(outputs["split_manifest"]), split_manifest)
    write_json_frozen(resolve_repo_path(outputs["diagnostics"]), diagnostics)
    write_json_frozen(resolve_repo_path(outputs["environment"]), environment)
    write_json_frozen(resolve_repo_path(outputs["route"]), route_record)
    write_text_frozen(resolve_repo_path(outputs["report"]), markdown)
    manifest = {
        "name": "step3_cita_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": sha256(path),
        "artifacts": package_artifacts(protocol),
        "command": (
            ".venv/Scripts/python.exe "
            "experiments/step3_response_quality_persistence/scripts/step3_cita_p0.py run"
        ),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    write_json_frozen(resolve_repo_path(outputs["run_manifest"]), manifest)
    print(
        json.dumps(
            {
                "route": route,
                "token_signal": gates["token_signal"],
                "gradient_effect_size": gates["gradient_effect_size"],
                "cross_split_replication": gates["cross_split_replication"],
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return diagnostics


def verify_run(path: Path) -> dict[str, Any]:
    protocol, rows = context(path)
    manifest_path = resolve_repo_path(protocol["outputs"]["run_manifest"])
    manifest = read_json(manifest_path)
    if manifest.get("protocol_sha256") != sha256(path):
        raise CitaP0ValidationError("result manifest protocol hash mismatch")
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise CitaP0ValidationError(f"result artifact hash mismatch: {artifact['name']}")
        if artifact["rows"] is not None and len(read_jsonl(artifact_path)) != artifact["rows"]:
            raise CitaP0ValidationError(f"result artifact row mismatch: {artifact['name']}")
    cache = read_jsonl(resolve_repo_path(protocol["outputs"]["weight_cache"]))
    validate_weight_cache(cache, rows, protocol)
    diagnostics = read_json(resolve_repo_path(protocol["outputs"]["diagnostics"]))
    route = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    if diagnostics.get("route") != route.get("route") or diagnostics.get("route") != manifest.get("route"):
        raise CitaP0ValidationError("CITA-P0 result routes disagree")
    if diagnostics.get("initial_trainable_state_sha256") != diagnostics.get("final_trainable_state_sha256"):
        raise CitaP0ValidationError("CITA-P0 trainable state changed")
    if diagnostics.get("optimizer_steps") != 0 or diagnostics.get("training_authorized") is not False:
        raise CitaP0ValidationError("CITA-P0 result violates zero-training boundary")
    result = {
        "status": "PASS_CITA_P0_RESULT_VERIFICATION",
        "route": diagnostics["route"],
        "artifacts": len(manifest["artifacts"]),
        "cache_rows": len(cache),
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
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        CitaP0ValidationError,
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_CITA_P0_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
