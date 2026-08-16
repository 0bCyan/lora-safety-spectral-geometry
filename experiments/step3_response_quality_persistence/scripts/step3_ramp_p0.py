"""Run the prospective RAMP-P0 zero-update directional mechanism gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import step3_cita_common as gradient_common
import step3_cita_p0 as model_common
import step3_ramp_common as ramp


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_ramp_p0/"
    "protocol/ramp_p0_protocol.json"
)


class RampP0ValidationError(ValueError):
    """Raised when an input or result violates the frozen RAMP-P0 protocol."""


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
        raise RampP0ValidationError(f"path escapes repository: {value}") from exc
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
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_source_rows(
    rows: list[dict[str, Any]], expected_rows: int, expected_prompts: int,
    expected_categories: int,
) -> dict[str, Any]:
    required = {
        "sample_id",
        "target_id",
        "category",
        "prompt",
        "response",
        "prompt_hash",
        "target_hash",
        "step2g_split",
        "reviewer_decision",
    }
    if len(rows) != expected_rows:
        raise RampP0ValidationError(
            f"source has {len(rows)} rows, expected {expected_rows}"
        )
    target_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not required.issubset(row):
            raise RampP0ValidationError("source row lacks required fields")
        target_id = str(row["target_id"])
        if target_id in target_ids:
            raise RampP0ValidationError("target IDs are not unique")
        target_ids.add(target_id)
        grouped[str(row["sample_id"])].append(row)
        if text_sha256(str(row["prompt"])) != row["prompt_hash"]:
            raise RampP0ValidationError("prompt hash mismatch")
        if text_sha256(str(row["response"])) != row["target_hash"]:
            raise RampP0ValidationError("target hash mismatch")
        if row["step2g_split"] != "safety_train":
            raise RampP0ValidationError("source row is not safety_train")
        if row["reviewer_decision"] != "approve":
            raise RampP0ValidationError("source target is not approved")
    if len(grouped) != expected_prompts or any(len(group) != 2 for group in grouped.values()):
        raise RampP0ValidationError("source prompt clusters are not two-target pairs")
    categories = {str(row["category"]) for row in rows}
    if len(categories) != expected_categories:
        raise RampP0ValidationError("source category count differs from protocol")
    return {
        "rows": len(rows),
        "prompt_clusters": len(grouped),
        "targets_per_prompt": 2,
        "categories": sorted(categories),
    }


def frozen_splits(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    selection = protocol["selection"]
    splits = gradient_common.select_prompt_disjoint_rows(
        rows,
        int(selection["calibration_prompts"]),
        int(selection["validation_prompts"]),
        int(selection["seed"]),
    )
    for split in ("calibration", "validation"):
        observed = [str(row["target_id"]) for row in splits[split]]
        if observed != list(selection[f"{split}_target_ids"]):
            raise RampP0ValidationError(f"{split} split differs from frozen IDs")
    if {row["sample_id"] for row in splits["calibration"]}.intersection(
        {row["sample_id"] for row in splits["validation"]}
    ):
        raise RampP0ValidationError("calibration and validation prompts overlap")
    return splits


def context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_RAMP_model_forward_or_gradient":
        raise RampP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RampP0ValidationError(f"frozen input hash mismatch: {name}")
    boundary = (
        protocol.get("optimizer_objects") == 0
        and protocol.get("scheduler_objects") == 0
        and protocol.get("optimizer_steps") == 0
        and protocol.get("training_authorized") is False
        and protocol.get("candidate_generation_authorized") is False
        and protocol.get("core100_opened") is False
        and protocol.get("heldout_opened") is False
    )
    if not boundary:
        raise RampP0ValidationError("protocol violates the zero-update boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RampP0ValidationError("entry commit is not an ancestor of HEAD")
    rows = read_jsonl(resolve_repo_path(protocol["source"]["targets_path"]))
    counts = protocol["source"]["counts"]
    validate_source_rows(
        rows,
        int(counts["rows"]),
        int(counts["prompt_clusters"]),
        int(counts["categories"]),
    )
    frozen_splits(rows, protocol)
    return protocol, rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise RampP0ValidationError("RAMP-P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.append(relative(path))
    for item in tracked:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    for output in protocol["outputs"].values():
        if resolve_repo_path(output).exists():
            raise RampP0ValidationError(f"prospective output already exists: {output}")
    return git_output("rev-parse", "HEAD")


def load_tokenizer(protocol: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model = protocol["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["model_name_or_path"],
        local_files_only=bool(model["local_files_only"]),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with",
    "this", "that", "is", "are", "be", "can", "i", "you", "your", "my",
    "how", "what", "it",
}


def content_words(value: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z][a-z'-]{2,}", value.lower())
        if word not in STOPWORDS
    }


def build_features(
    tokenizer: Any, rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    design = protocol["design"]
    width = int(design["margin_width"])
    max_length = int(design["max_length"])
    markers = tuple(str(value).lower() for value in design["safe_routing_onset_markers"])
    features = []
    prefix_templates = set()
    semantic_counts = []
    overlap_counts = []
    first_sentence = 0
    routing_onset = 0
    prompt_recalls = []
    for row in rows:
        serialized = model_common.serialized_response(
            tokenizer, str(row["prompt"]), str(row["response"]), max_length
        )
        semantic_positions = [
            index for index, keep in enumerate(serialized["semantic_mask"]) if keep
        ]
        if semantic_positions != list(range(len(semantic_positions))):
            raise RampP0ValidationError("semantic response tokens are not a prefix")
        count = len(semantic_positions)
        prefix_ordinals = ramp.selected_positions(count, width, "prefix")
        suffix_ordinals = ramp.selected_positions(count, width, "suffix")
        prefix_positions = [semantic_positions[index] for index in prefix_ordinals]
        suffix_positions = [semantic_positions[index] for index in suffix_ordinals]
        prefix_ids = [serialized["response_token_ids"][index] for index in prefix_positions]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()
        normalized = re.sub(r"\s+", " ", prefix_text.lower()).strip()
        prefix_templates.add(normalized)
        complete = bool(re.search(r"[.!?](?:\s|$)", prefix_text))
        routed = any(marker in prefix_text.lower() for marker in markers)
        first_sentence += int(complete)
        routing_onset += int(routed)
        prompt_vocabulary = content_words(str(row["prompt"]))
        prefix_vocabulary = content_words(prefix_text)
        prompt_recalls.append(
            len(prompt_vocabulary.intersection(prefix_vocabulary))
            / max(1, len(prompt_vocabulary))
        )
        overlap = len(set(prefix_positions).intersection(suffix_positions))
        overlap_counts.append(overlap)
        semantic_counts.append(count)
        features.append(
            {
                "sample_id": str(row["sample_id"]),
                "target_id": str(row["target_id"]),
                "category": str(row["category"]),
                "input_ids": list(serialized["input_ids"]),
                "response_token_ids": list(serialized["response_token_ids"]),
                "prefix_tokens": int(serialized["prefix_tokens"]),
                "semantic_positions": semantic_positions,
                "prefix_positions": prefix_positions,
                "suffix_positions": suffix_positions,
                "semantic_tokens": count,
                "prefix_suffix_overlap_tokens": overlap,
                "prefix_first_sentence_complete": complete,
                "prefix_safe_routing_onset": routed,
                "prefix_normalized_sha256": text_sha256(normalized),
                "prefix_prompt_content_recall": prompt_recalls[-1],
            }
        )
    total = len(features)
    audit = {
        "rows": total,
        "prompt_clusters": len({row["sample_id"] for row in rows}),
        "margin_width": width,
        "minimum_semantic_tokens": min(semantic_counts),
        "maximum_semantic_tokens": max(semantic_counts),
        "all_rows_support_equal_prefix_suffix_terms": all(
            len(feature["prefix_positions"]) == len(feature["suffix_positions"]) == width
            for feature in features
        ),
        "prefix_first_sentence_complete_rate": first_sentence / total,
        "prefix_safe_routing_onset_rate": routing_onset / total,
        "unique_normalized_prefix_rate": len(prefix_templates) / total,
        "mean_prompt_content_recall": sum(prompt_recalls) / total,
        "mean_prefix_suffix_overlap_tokens": sum(overlap_counts) / total,
        "maximum_prefix_suffix_overlap_tokens": max(overlap_counts),
    }
    return features, audit


def verify_structural_freeze(audit: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected = protocol["design"]["tokenizer_only_structural_audit"]
    for key, value in expected.items():
        observed = audit.get(key)
        if isinstance(value, float):
            if not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
                raise RampP0ValidationError(f"structural audit drift: {key}")
        elif observed != value:
            raise RampP0ValidationError(f"structural audit drift: {key}")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    if not torch.cuda.is_available():
        raise RampP0ValidationError("CUDA is unavailable")
    tokenizer = load_tokenizer(protocol)
    _, audit = build_features(tokenizer, rows, protocol)
    verify_structural_freeze(audit, protocol)
    result = {
        "status": "PASS_RAMP_P0_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "source_rows": len(rows),
        "prompt_clusters": len({row["sample_id"] for row in rows}),
        "calibration_prompts": protocol["selection"]["calibration_prompts"],
        "validation_prompts": protocol["selection"]["validation_prompts"],
        "structural_audit": audit,
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


def response_objectives(
    response_logits: Any,
    response_targets: Any,
    semantic_positions: list[int],
    width: int,
    objective: str,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    if response_logits.ndim != 2 or response_targets.ndim != 1:
        raise ValueError("response objective tensors have invalid ranks")
    if response_logits.shape[0] != response_targets.shape[0]:
        raise ValueError("response objective tensors do not align")
    if objective == "uniform":
        loss = functional.cross_entropy(response_logits.float(), response_targets)
        return loss, {"mean_nll": float(loss.detach().cpu())}
    ordinals = ramp.selected_positions(len(semantic_positions), width, objective)
    positions = [semantic_positions[index] for index in ordinals]
    indices = torch.tensor(positions, dtype=torch.long, device=response_logits.device)
    chosen_logits = response_logits.index_select(0, indices)
    chosen_targets = response_targets.index_select(0, indices)
    margins, competitor_ids = ramp.strongest_competitor_margin(
        chosen_logits, chosen_targets
    )
    loss = ramp.margin_barrier(margins)
    return loss, {
        "margin_terms": len(positions),
        "mean_margin": float(margins.detach().mean().cpu()),
        "minimum_margin": float(margins.detach().min().cpu()),
        "certified_token_rate": float((margins.detach() > 0).float().mean().cpu()),
        "full_region_certified": bool((margins.detach() > 0).all().cpu()),
        "competitor_ids_sha256": text_sha256(
            ",".join(str(int(value)) for value in competitor_ids.detach().cpu())
        ),
    }


def aggregate_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    features: list[dict[str, Any]],
    split_name: str,
    objective: str,
    width: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.zero_grad(set_to_none=True)
    device = model_common.model_device(model)
    losses = []
    rows = []
    for position, feature in enumerate(features, start=1):
        ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        prefix = int(feature["prefix_tokens"])
        logits = model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).logits
        response_logits = logits[0, prefix - 1 : ids.shape[1] - 1]
        response_targets = ids[0, prefix:]
        loss, metrics = response_objectives(
            response_logits,
            response_targets,
            list(feature["semantic_positions"]),
            width,
            objective,
        )
        losses.append(float(loss.detach().cpu()))
        (loss / len(features)).backward()
        rows.append(
            {
                "split": split_name,
                "objective": objective,
                "sample_id": feature["sample_id"],
                "target_id": feature["target_id"],
                **metrics,
            }
        )
        del ids, attention, logits, response_logits, response_targets, loss
        if position % 4 == 0 or position == len(features):
            print(
                f"RAMP-P0 gradients {split_name}/{objective} "
                f"{position}/{len(features)}",
                flush=True,
            )
    gradients, coverage = model_common.capture_gradients(parameters)
    coverage.update(
        {
            "mean_loss": sum(losses) / len(losses),
            "gradient_norm": ramp.gradient_norm(gradients),
            "gradient_hash": gradient_common.gradient_hash(gradients),
        }
    )
    if objective != "uniform":
        coverage.update(
            {
                "mean_margin": sum(float(row["mean_margin"]) for row in rows)
                / len(rows),
                "minimum_margin": min(float(row["minimum_margin"]) for row in rows),
                "certified_token_rate": sum(
                    float(row["certified_token_rate"]) for row in rows
                )
                / len(rows),
                "full_region_certified_rate": sum(
                    int(row["full_region_certified"]) for row in rows
                )
                / len(rows),
            }
        )
    model.zero_grad(set_to_none=True)
    return gradients, coverage, rows


def environment_record(runtime_head: str, protocol_sha: str) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "runtime_head": runtime_head,
        "protocol_sha256": protocol_sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }


def artifact_rows(path: Path) -> int | None:
    return len(read_jsonl(path)) if path.suffix == ".jsonl" else None


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
                "rows": artifact_rows(path),
            }
        )
    return artifacts


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def run(path: Path) -> dict[str, Any]:
    import torch

    protocol, source_rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
    tokenizer = load_tokenizer(protocol)
    all_features, structural = build_features(tokenizer, source_rows, protocol)
    verify_structural_freeze(structural, protocol)
    by_target = {feature["target_id"]: feature for feature in all_features}
    splits = frozen_splits(source_rows, protocol)
    split_features = {
        name: [by_target[str(row["target_id"])] for row in rows]
        for name, rows in splits.items()
    }

    started = time.time()
    model, model_tokenizer = model_common.load_model(protocol)
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RampP0ValidationError("tokenizer-only and model tokenizers differ")
    del model_tokenizer, tokenizer
    parameters = model_common.trainable_parameters(model)
    initial_state = model_common.trainable_state_hash(parameters)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    width = int(protocol["design"]["margin_width"])

    gradients: dict[str, dict[str, dict[str, Any]]] = {}
    coverage: dict[str, dict[str, dict[str, Any]]] = {}
    margin_rows = []
    for split_name in ("calibration", "validation"):
        gradients[split_name] = {}
        coverage[split_name] = {}
        for objective in ("uniform", "prefix", "suffix"):
            gradient, details, rows = aggregate_gradient(
                model,
                parameters,
                split_features[split_name],
                split_name,
                objective,
                width,
            )
            gradients[split_name][objective] = gradient
            coverage[split_name][objective] = details
            if objective != "uniform":
                margin_rows.extend(rows)

    calibration = gradients["calibration"]
    rho = float(protocol["design"]["auxiliary_gradient_norm_ratio"])
    uniform_norm = ramp.gradient_norm(calibration["uniform"])
    prefix_coefficient = ramp.calibrated_coefficient(
        uniform_norm, ramp.gradient_norm(calibration["prefix"]), rho
    )
    suffix_coefficient = ramp.calibrated_coefficient(
        uniform_norm, ramp.gradient_norm(calibration["suffix"]), rho
    )
    prefix_scaled_norm = ramp.gradient_norm(
        ramp.scaled_gradient(calibration["prefix"], prefix_coefficient)
    )
    suffix_scaled_norm = ramp.gradient_norm(
        ramp.scaled_gradient(calibration["suffix"], suffix_coefficient)
    )
    directional = ramp.directional_metrics(
        calibration["uniform"],
        calibration["prefix"],
        calibration["suffix"],
        gradients["validation"]["prefix"],
        prefix_coefficient,
        suffix_coefficient,
    )

    final_state = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    gradient_records = [
        coverage[split][objective]
        for split in ("calibration", "validation")
        for objective in ("uniform", "prefix", "suffix")
    ]
    gates_cfg = protocol["gates"]
    target_aux_norm = rho * uniform_norm
    tolerance = float(gates_cfg["gradient_budget_relative_tolerance"])
    budget_errors = [
        abs(prefix_scaled_norm - target_aux_norm) / target_aux_norm,
        abs(suffix_scaled_norm - target_aux_norm) / target_aux_norm,
        abs(prefix_scaled_norm - suffix_scaled_norm) / target_aux_norm,
    ]
    gates = {
        "integrity": (
            structural["rows"] == protocol["source"]["counts"]["rows"]
            and len(split_features["calibration"])
            == protocol["selection"]["calibration_prompts"]
            and len(split_features["validation"])
            == protocol["selection"]["validation_prompts"]
        ),
        "gradient_valid": all(
            not row["missing_gradient_tensors"]
            and row["finite"]
            and float(row["gradient_norm"]) > 0
            for row in gradient_records
        ),
        "budget_matched": (
            all(finite(value) for value in (prefix_coefficient, suffix_coefficient))
            and max(budget_errors) <= tolerance
        ),
        "vram": headroom >= float(gates_cfg["minimum_vram_headroom_fraction"]),
        "state_restored": initial_state == final_state,
        "prefix_behavioral": (
            structural["all_rows_support_equal_prefix_suffix_terms"]
            and structural["prefix_first_sentence_complete_rate"]
            >= float(gates_cfg["prefix_first_sentence_complete_rate_min"])
            and structural["prefix_safe_routing_onset_rate"]
            >= float(gates_cfg["prefix_safe_routing_onset_rate_min"])
            and structural["unique_normalized_prefix_rate"]
            >= float(gates_cfg["unique_normalized_prefix_rate_min"])
        ),
        "placebo_distinct": (
            finite(directional["calibration_prefix_suffix_cosine"])
            and directional["calibration_prefix_suffix_cosine"]
            <= float(gates_cfg["calibration_prefix_suffix_cosine_max"])
        ),
        "cross_prompt_directional": (
            finite(directional["calibration_validation_prefix_cosine"])
            and finite(directional["prefix_over_suffix_directional_cosine_margin"])
            and directional["calibration_validation_prefix_cosine"]
            >= float(gates_cfg["calibration_validation_prefix_cosine_min"])
            and directional["prefix_over_suffix_directional_cosine_margin"]
            >= float(gates_cfg["prefix_over_suffix_directional_cosine_margin_min"])
            and directional["ramp_increment_over_uniform_dot"] > 0
            and directional["ramp_validation_prefix_dot"]
            > directional["placebo_validation_prefix_dot"]
        ),
    }
    route = ramp.select_route(gates)
    coefficients = {
        "auxiliary_gradient_norm_ratio": rho,
        "uniform_calibration_gradient_norm": uniform_norm,
        "prefix_raw_gradient_norm": ramp.gradient_norm(calibration["prefix"]),
        "suffix_raw_gradient_norm": ramp.gradient_norm(calibration["suffix"]),
        "prefix_coefficient": prefix_coefficient,
        "suffix_coefficient": suffix_coefficient,
        "prefix_scaled_gradient_norm": prefix_scaled_norm,
        "suffix_scaled_gradient_norm": suffix_scaled_norm,
        "target_auxiliary_gradient_norm": target_aux_norm,
        "maximum_relative_budget_error": max(budget_errors),
    }
    split_manifest = {
        "name": "step3_ramp_p0_prompt_disjoint_split_v1",
        "selection_seed": protocol["selection"]["seed"],
        "selection_rule": protocol["selection"]["rule"],
        "calibration_target_ids": protocol["selection"]["calibration_target_ids"],
        "validation_target_ids": protocol["selection"]["validation_target_ids"],
        "prompt_overlap": 0,
    }
    diagnostics = {
        "name": "step3_ramp_p0_gradient_diagnostics_v1",
        "coverage": coverage,
        "coefficients": coefficients,
        "directional": directional,
        "gates": gates,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }
    decision = {
        "name": "step3_ramp_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": sha256(path),
        "gates": gates,
        "structural_audit": structural,
        "coefficients": coefficients,
        "directional": directional,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "training_authorized": False,
        "matched_smoke_protocol_authorized": route
        == "PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        "core100_opened": False,
        "heldout_opened": False,
    }
    report = f"""# RAMP-P0 Zero-Update Directional Mechanism Report

Route: `{route}`.

## Structural Evidence

All `{structural['rows']}` approved targets support exactly `{width}` prefix and
suffix margin terms. The 16-token prefix completes the first sentence in
`{structural['prefix_first_sentence_complete_rate']:.3f}` of rows, reaches a
registered safe-routing onset in `{structural['prefix_safe_routing_onset_rate']:.3f}`
of rows, and has normalized uniqueness
`{structural['unique_normalized_prefix_rate']:.3f}`.

## Gradient-Budget Evidence

Both auxiliary gradients were calibrated to `{rho:.3f}` of the uniform safety
gradient norm. Prefix and suffix coefficients were `{prefix_coefficient:.6g}`
and `{suffix_coefficient:.6g}`; the maximum relative budget error was
`{max(budget_errors):.3e}`. Calibration prefix/suffix gradient cosine was
`{directional['calibration_prefix_suffix_cosine']:.6f}`.

## Prompt-Disjoint Directional Evidence

Calibration-to-validation prefix cosine was
`{directional['calibration_validation_prefix_cosine']:.6f}`. The suffix-to-prefix
validation cosine was
`{directional['suffix_calibration_to_prefix_validation_cosine']:.6f}`, giving a
registered prefix-over-suffix margin of
`{directional['prefix_over_suffix_directional_cosine_margin']:.6f}`.

This P0 tests a first-order, prompt-disjoint certificate direction. It does not
test generated behavior, safety, utility, or publication readiness. No optimizer
or scheduler was created, the exact trainable-state hash was unchanged, and
`optimizer_steps=0`, `training_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
"""
    environment = environment_record(runtime_head, sha256(path))

    model_common.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["structural_audit"]), structural
    )
    model_common.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["split_manifest"]), split_manifest
    )
    model_common.write_jsonl_frozen(
        resolve_repo_path(protocol["outputs"]["margin_rows"]), margin_rows
    )
    model_common.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["gradient_diagnostics"]), diagnostics
    )
    model_common.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["environment"]), environment
    )
    model_common.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["route"]), decision
    )
    model_common.write_text_frozen(
        resolve_repo_path(protocol["outputs"]["report"]), report
    )
    reproduction = {
        "name": "step3_ramp_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": sha256(path),
        "artifacts": package_artifacts(protocol),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_ramp_p0.py run"
        ),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    model_common.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["run_manifest"]), reproduction
    )
    del gradients, calibration, parameters, model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "route": route,
                "gates": gates,
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _ = context(path)
    reproduction = read_json(resolve_repo_path(protocol["outputs"]["run_manifest"]))
    if reproduction.get("protocol_sha256") != sha256(path):
        raise RampP0ValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise RampP0ValidationError(f"artifact mismatch: {artifact['path']}")
    decision = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    recomputed = ramp.select_route(decision["gates"])
    if recomputed != decision.get("route") or recomputed != reproduction.get("route"):
        raise RampP0ValidationError("stored RAMP-P0 routes differ")
    result = {
        "status": "PASS_RAMP_P0_RESULT_VERIFICATION",
        "route": recomputed,
        "artifacts": len(reproduction["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "matched_smoke_protocol_authorized": decision[
            "matched_smoke_protocol_authorized"
        ],
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
        RampP0ValidationError,
        model_common.CitaP0ValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_RAMP_P0_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
