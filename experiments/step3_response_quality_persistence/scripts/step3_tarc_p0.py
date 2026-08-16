"""Run the prospective zero-update TARC directional mechanism gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import step3_cita_common as gradient_common
import step3_cita_p0 as model_common
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_tarc_common as tarc


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_tarc_p0/"
    "protocol/tarc_p0_protocol.json"
)
DEFAULT_SOURCE = (
    "experiments/step2_mechanism_dynamics/manifests/"
    "step2ir_natural_diverse_targets_v1.jsonl"
)


class TarcP0ValidationError(ValueError):
    """Raised when TARC-P0 violates its prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["audit-source", "preflight", "run", "verify-run"]
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--selection-seed", type=int, default=2026081720)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise TarcP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_tokenizer(model_name: str = "Qwen/Qwen3-4B") -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def stable_score(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def select_diverse_prompt_pairs(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    by_category: dict[str, list[str]] = defaultdict(list)
    for sample_id, pair in grouped.items():
        categories = {str(row["category"]) for row in pair}
        if len(pair) != 2 or len(categories) != 1:
            raise TarcP0ValidationError("source prompt pair is malformed")
        by_category[categories.pop()].append(sample_id)
    selected: dict[str, list[str]] = {"calibration": [], "validation": []}
    used: set[str] = set()
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category], key=lambda value: stable_score(seed, value)
        )
        if len(ordered) >= 2:
            selected["calibration"].append(ordered[0])
            selected["validation"].append(ordered[1])
            used.update(ordered[:2])
        else:
            split = min(selected, key=lambda name: (len(selected[name]), name))
            selected[split].append(ordered[0])
            used.add(ordered[0])

    category_by_id = {
        sample_id: category
        for category, sample_ids in by_category.items()
        for sample_id in sample_ids
    }
    while any(len(values) < 8 for values in selected.values()):
        split = min(
            (name for name in selected if len(selected[name]) < 8),
            key=lambda name: (len(selected[name]), name),
        )
        counts = Counter(category_by_id[value] for value in selected[split])
        candidates = [value for value in grouped if value not in used]
        if not candidates:
            raise TarcP0ValidationError("source cannot fill prompt-disjoint splits")
        choice = min(
            candidates,
            key=lambda value: (
                counts[category_by_id[value]],
                stable_score(seed, value),
            ),
        )
        selected[split].append(choice)
        used.add(choice)
    return selected


def route_depth(
    tokenizer: Any, response_token_ids: list[int], semantic_tokens: int, width: int
) -> tuple[int, str, str]:
    limit = min(semantic_tokens, width)
    for count in range(1, limit + 1):
        decoded = tokenizer.decode(
            response_token_ids[:count], skip_special_tokens=True
        ).strip()
        try:
            marker = tarc.refusal_route_marker(decoded)
        except ValueError:
            continue
        return count, marker, decoded
    raise TarcP0ValidationError("refusal route does not complete within width")


def build_pair_features(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    pairs = []
    route_depths = []
    markers: Counter[str] = Counter()
    branching_prompts = 0
    distinct_routes = 0
    total_nodes = 0
    maximum_branch_width = 0
    for sample_id in sorted(grouped):
        source_pair = sorted(grouped[sample_id], key=lambda row: str(row["target_id"]))
        if len(source_pair) != 2:
            raise TarcP0ValidationError("TARC requires two targets per prompt")
        features = []
        paths = []
        for row in source_pair:
            serialized = model_common.serialized_response(
                tokenizer, str(row["prompt"]), str(row["response"]), max_length
            )
            semantic_positions = [
                index
                for index, keep in enumerate(serialized["semantic_mask"])
                if keep
            ]
            if semantic_positions != list(range(len(semantic_positions))):
                raise TarcP0ValidationError("semantic response tokens are not a prefix")
            depth, marker, route_text = route_depth(
                tokenizer,
                list(serialized["response_token_ids"]),
                len(semantic_positions),
                width,
            )
            path = [int(value) for value in serialized["response_token_ids"][:depth]]
            route_depths.append(depth)
            markers[marker] += 1
            paths.append(path)
            features.append(
                {
                    "target_id": str(row["target_id"]),
                    "input_ids": list(serialized["input_ids"]),
                    "prefix_tokens": int(serialized["prefix_tokens"]),
                    "semantic_positions": semantic_positions,
                    "route_depth": depth,
                    "route_marker": marker,
                    "route_text_sha256": ramp_p0.text_sha256(route_text.lower()),
                }
            )
        nodes = tarc.trie_nodes(paths)
        branch_width = max(len(node["allowed_token_ids"]) for node in nodes)
        maximum_branch_width = max(maximum_branch_width, branch_width)
        branching_prompts += int(any(len(node["allowed_token_ids"]) > 1 for node in nodes))
        distinct_routes += int(paths[0] != paths[1])
        total_nodes += len(nodes)
        pairs.append(
            {
                "sample_id": sample_id,
                "category": str(source_pair[0]["category"]),
                "prompt_sha256": ramp_p0.text_sha256(str(source_pair[0]["prompt"])),
                "features": features,
                "paths": paths,
                "nodes": nodes,
            }
        )
    total_targets = len(route_depths)
    total_pairs = len(pairs)
    audit = {
        "rows": total_targets,
        "prompt_clusters": total_pairs,
        "categories": len({pair["category"] for pair in pairs}),
        "targets_per_prompt": 2,
        "route_width_limit": width,
        "minimum_route_tokens": min(route_depths),
        "maximum_route_tokens": max(route_depths),
        "mean_route_tokens": sum(route_depths) / total_targets,
        "all_routes_complete_within_width": max(route_depths) <= width,
        "registered_route_marker_counts": dict(sorted(markers.items())),
        "distinct_route_pair_rate": distinct_routes / total_pairs,
        "branching_prompt_rate": branching_prompts / total_pairs,
        "mean_internal_trie_nodes": total_nodes / total_pairs,
        "maximum_allowed_branch_width": maximum_branch_width,
    }
    return pairs, audit


def audit_source(source: Path, selection_seed: int) -> dict[str, Any]:
    rows = read_jsonl(source)
    ramp_p0.validate_source_rows(rows, 80, 40, 8)
    tokenizer = load_tokenizer()
    _, audit = build_pair_features(tokenizer, rows, 512, 16)
    selection = select_diverse_prompt_pairs(rows, selection_seed)
    result = {
        "status": "PASS_TARC_TOKENIZER_ONLY_SOURCE_AUDIT",
        "source_sha256": sha256(source),
        "selection_seed": selection_seed,
        "selection": selection,
        "structural_audit": audit,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_TARC_model_forward_or_gradient":
        raise TarcP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise TarcP0ValidationError(f"frozen input hash mismatch: {name}")
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
        raise TarcP0ValidationError("protocol violates the zero-update boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise TarcP0ValidationError("entry commit is not an ancestor of HEAD")
    rows = read_jsonl(resolve_repo_path(protocol["source"]["targets_path"]))
    counts = protocol["source"]["counts"]
    ramp_p0.validate_source_rows(
        rows,
        int(counts["rows"]),
        int(counts["prompt_clusters"]),
        int(counts["categories"]),
    )
    selected = select_diverse_prompt_pairs(rows, int(protocol["selection"]["seed"]))
    for split in ("calibration", "validation"):
        if selected[split] != list(protocol["selection"][f"{split}_sample_ids"]):
            raise TarcP0ValidationError(f"{split} selection differs")
    if set(selected["calibration"]).intersection(selected["validation"]):
        raise TarcP0ValidationError("calibration and validation prompts overlap")
    return protocol, rows


def verify_structural_freeze(audit: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected = protocol["design"]["tokenizer_only_structural_audit"]
    for key, value in expected.items():
        observed = audit.get(key)
        if isinstance(value, float):
            if not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
                raise TarcP0ValidationError(f"structural audit drift: {key}")
        elif observed != value:
            raise TarcP0ValidationError(f"structural audit drift: {key}")


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise TarcP0ValidationError("TARC-P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.append(relative(path))
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    for output in protocol["outputs"].values():
        if resolve_repo_path(output).exists():
            raise TarcP0ValidationError(f"prospective output already exists: {output}")
    return git_output("rev-parse", "HEAD")


def selected_pairs(
    pairs: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    by_id = {pair["sample_id"]: pair for pair in pairs}
    return {
        split: [by_id[value] for value in protocol["selection"][f"{split}_sample_ids"]]
        for split in ("calibration", "validation")
    }


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    if not torch.cuda.is_available():
        raise TarcP0ValidationError("CUDA is unavailable")
    tokenizer = load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    pairs, audit = build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    verify_structural_freeze(audit, protocol)
    splits = selected_pairs(pairs, protocol)
    result = {
        "status": "PASS_TARC_P0_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "source_rows": len(rows),
        "prompt_clusters": len(pairs),
        "calibration_prompts": len(splits["calibration"]),
        "validation_prompts": len(splits["validation"]),
        "prompt_overlap": 0,
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


def pair_objective(
    model: Any,
    pair: dict[str, Any],
    objective: str,
    ramp_width: int,
    temperature: float,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    device = model_common.model_device(model)
    response_logits = []
    response_targets = []
    uniform_losses = []
    ramp_losses = []
    for feature in pair["features"]:
        ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        prefix = int(feature["prefix_tokens"])
        logits = model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).logits
        current_logits = logits[0, prefix - 1 : ids.shape[1] - 1]
        current_targets = ids[0, prefix:]
        response_logits.append(current_logits)
        response_targets.append(current_targets)
        uniform_losses.append(
            functional.cross_entropy(current_logits.float(), current_targets)
        )
        current_ramp, _ = ramp_p0.response_objectives(
            current_logits,
            current_targets,
            list(feature["semantic_positions"]),
            ramp_width,
            "prefix",
        )
        ramp_losses.append(current_ramp)
    if objective == "uniform":
        loss = torch.stack(uniform_losses).mean()
        return loss, {"mean_uniform_loss": float(loss.detach().cpu())}
    if objective == "ramp":
        loss = torch.stack(ramp_losses).mean()
        return loss, {
            "margin_terms": 2 * ramp_width,
            "mean_margin_loss": float(loss.detach().cpu()),
        }
    if objective != "tarc":
        raise ValueError(f"unknown TARC-P0 objective: {objective}")
    margins = []
    unique_node_margins = []
    for node in pair["nodes"]:
        node_values = []
        for path_index in node["path_indices"]:
            value = tarc.conservative_allowed_margin(
                response_logits[path_index][int(node["depth"])],
                list(node["allowed_token_ids"]),
                temperature,
            )
            margins.append(value)
            node_values.append(value)
        unique_node_margins.append(torch.stack(node_values).mean())
    margin_tensor = torch.stack(margins)
    loss = tarc.certificate_barrier(margin_tensor)
    detached = margin_tensor.detach().float()
    return loss, {
        "margin_terms": int(margin_tensor.numel()),
        "unique_trie_nodes": len(pair["nodes"]),
        "mean_margin": float(detached.mean().cpu()),
        "minimum_margin": float(detached.min().cpu()),
        "certified_node_occurrence_rate": float((detached > 0).float().mean().cpu()),
        "full_trie_certified": bool(
            (torch.stack(unique_node_margins).detach() > 0).all().cpu()
        ),
    }


def aggregate_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    pairs: list[dict[str, Any]],
    split_name: str,
    objective: str,
    ramp_width: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.zero_grad(set_to_none=True)
    losses = []
    rows = []
    for position, pair in enumerate(pairs, start=1):
        loss, metrics = pair_objective(
            model, pair, objective, ramp_width, temperature
        )
        losses.append(float(loss.detach().cpu()))
        (loss / len(pairs)).backward()
        rows.append(
            {
                "split": split_name,
                "objective": objective,
                "sample_id": pair["sample_id"],
                "category": pair["category"],
                **metrics,
            }
        )
        print(
            f"TARC-P0 gradients {split_name}/{objective} {position}/{len(pairs)}",
            flush=True,
        )
    gradients, coverage = model_common.capture_gradients(parameters)
    coverage.update(
        {
            "mean_loss": sum(losses) / len(losses),
            "gradient_norm": gradient_math.gradient_norm(gradients),
            "gradient_hash": gradient_common.gradient_hash(gradients),
        }
    )
    if objective == "tarc":
        coverage.update(
            {
                "mean_margin": sum(float(row["mean_margin"]) for row in rows)
                / len(rows),
                "minimum_margin": min(float(row["minimum_margin"]) for row in rows),
                "certified_node_occurrence_rate": sum(
                    float(row["certified_node_occurrence_rate"]) for row in rows
                )
                / len(rows),
                "full_trie_certificate_rate": sum(
                    int(row["full_trie_certified"]) for row in rows
                )
                / len(rows),
            }
        )
    model.zero_grad(set_to_none=True)
    return gradients, coverage, rows


def directional_metrics(
    calibration: dict[str, dict[str, Any]],
    validation_tarc: dict[str, Any],
    ramp_coefficient: float,
    tarc_coefficient: float,
) -> dict[str, float]:
    ramp_scaled = gradient_math.scaled_gradient(
        calibration["ramp"], ramp_coefficient
    )
    tarc_scaled = gradient_math.scaled_gradient(
        calibration["tarc"], tarc_coefficient
    )
    uniform = calibration["uniform"]
    ramp_update = gradient_math.add_gradients(uniform, ramp_scaled)
    tarc_update = gradient_math.add_gradients(uniform, tarc_scaled)
    tarc_cosine = gradient_math.gradient_cosine(
        calibration["tarc"], validation_tarc
    )
    ramp_cosine = gradient_math.gradient_cosine(
        calibration["ramp"], validation_tarc
    )
    return {
        "calibration_tarc_ramp_cosine": gradient_math.gradient_cosine(
            calibration["tarc"], calibration["ramp"]
        ),
        "calibration_validation_tarc_cosine": tarc_cosine,
        "ramp_calibration_to_tarc_validation_cosine": ramp_cosine,
        "tarc_over_ramp_validation_cosine_margin": tarc_cosine - ramp_cosine,
        "uniform_tarc_validation_dot": gradient_math.gradient_dot(
            validation_tarc, uniform
        ),
        "ramp_tarc_validation_dot": gradient_math.gradient_dot(
            validation_tarc, ramp_update
        ),
        "tarc_tarc_validation_dot": gradient_math.gradient_dot(
            validation_tarc, tarc_update
        ),
        "ramp_increment_tarc_validation_dot": gradient_math.gradient_dot(
            validation_tarc, ramp_scaled
        ),
        "tarc_increment_tarc_validation_dot": gradient_math.gradient_dot(
            validation_tarc, tarc_scaled
        ),
    }


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
    runtime_head = require_clean_tracked_freeze(path, protocol)
    tokenizer = load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    all_pairs, structural = build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    verify_structural_freeze(structural, protocol)
    splits = selected_pairs(all_pairs, protocol)
    started = time.time()
    model, model_tokenizer = model_common.load_model(protocol)
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise TarcP0ValidationError("tokenizer-only and model tokenizers differ")
    del model_tokenizer, tokenizer
    parameters = model_common.trainable_parameters(model)
    initial_state = model_common.trainable_state_hash(parameters)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ramp_width = int(protocol["design"]["ramp_width"])
    temperature = float(protocol["design"]["softmin_temperature"])

    gradients: dict[str, dict[str, dict[str, Any]]] = {}
    coverage: dict[str, dict[str, dict[str, Any]]] = {}
    diagnostic_rows = []
    for split_name in ("calibration", "validation"):
        gradients[split_name] = {}
        coverage[split_name] = {}
        for objective in ("uniform", "ramp", "tarc"):
            gradient, details, objective_rows = aggregate_gradient(
                model,
                parameters,
                splits[split_name],
                split_name,
                objective,
                ramp_width,
                temperature,
            )
            gradients[split_name][objective] = gradient
            coverage[split_name][objective] = details
            if objective != "uniform":
                diagnostic_rows.extend(objective_rows)

    calibration = gradients["calibration"]
    rho = float(protocol["design"]["auxiliary_gradient_norm_ratio"])
    uniform_norm = gradient_math.gradient_norm(calibration["uniform"])
    ramp_coefficient = gradient_math.calibrated_coefficient(
        uniform_norm, gradient_math.gradient_norm(calibration["ramp"]), rho
    )
    tarc_coefficient = gradient_math.calibrated_coefficient(
        uniform_norm, gradient_math.gradient_norm(calibration["tarc"]), rho
    )
    target_norm = rho * uniform_norm
    ramp_scaled_norm = gradient_math.gradient_norm(
        gradient_math.scaled_gradient(calibration["ramp"], ramp_coefficient)
    )
    tarc_scaled_norm = gradient_math.gradient_norm(
        gradient_math.scaled_gradient(calibration["tarc"], tarc_coefficient)
    )
    budget_errors = [
        abs(ramp_scaled_norm - target_norm) / target_norm,
        abs(tarc_scaled_norm - target_norm) / target_norm,
        abs(ramp_scaled_norm - tarc_scaled_norm) / target_norm,
    ]
    directional = directional_metrics(
        calibration,
        gradients["validation"]["tarc"],
        ramp_coefficient,
        tarc_coefficient,
    )
    final_state = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    records = [
        coverage[split][objective]
        for split in ("calibration", "validation")
        for objective in ("uniform", "ramp", "tarc")
    ]
    gate_cfg = protocol["gates"]
    gates = {
        "integrity": (
            structural["rows"] == protocol["source"]["counts"]["rows"]
            and len(splits["calibration"])
            == int(protocol["selection"]["calibration_prompts"])
            and len(splits["validation"])
            == int(protocol["selection"]["validation_prompts"])
        ),
        "gradient_valid": all(
            not row["missing_gradient_tensors"]
            and row["finite"]
            and float(row["gradient_norm"]) > 0
            for row in records
        ),
        "budget_matched": (
            all(
                math.isfinite(value)
                for value in (ramp_coefficient, tarc_coefficient)
            )
            and max(budget_errors)
            <= float(gate_cfg["gradient_budget_relative_tolerance"])
        ),
        "vram": headroom >= float(gate_cfg["minimum_vram_headroom_fraction"]),
        "state_restored": initial_state == final_state,
        "route_structure": (
            structural["all_routes_complete_within_width"]
            and structural["maximum_route_tokens"]
            <= int(protocol["design"]["route_width_limit"])
            and structural["branching_prompt_rate"]
            >= float(gate_cfg["branching_prompt_rate_min"])
            and structural["distinct_route_pair_rate"]
            >= float(gate_cfg["distinct_route_pair_rate_min"])
        ),
        "objective_distinct": (
            math.isfinite(directional["calibration_tarc_ramp_cosine"])
            and directional["calibration_tarc_ramp_cosine"]
            <= float(gate_cfg["calibration_tarc_ramp_cosine_max"])
        ),
        "cross_prompt_directional": (
            math.isfinite(directional["calibration_validation_tarc_cosine"])
            and math.isfinite(
                directional["tarc_over_ramp_validation_cosine_margin"]
            )
            and directional["calibration_validation_tarc_cosine"]
            >= float(gate_cfg["calibration_validation_tarc_cosine_min"])
            and directional["tarc_over_ramp_validation_cosine_margin"]
            >= float(gate_cfg["tarc_over_ramp_validation_cosine_margin_min"])
            and directional["tarc_increment_tarc_validation_dot"] > 0
            and directional["tarc_tarc_validation_dot"]
            > directional["ramp_tarc_validation_dot"]
        ),
    }
    route = tarc.select_route(gates)
    coefficients = {
        "auxiliary_gradient_norm_ratio": rho,
        "uniform_calibration_gradient_norm": uniform_norm,
        "ramp_raw_gradient_norm": gradient_math.gradient_norm(calibration["ramp"]),
        "tarc_raw_gradient_norm": gradient_math.gradient_norm(calibration["tarc"]),
        "ramp_coefficient": ramp_coefficient,
        "tarc_coefficient": tarc_coefficient,
        "ramp_scaled_gradient_norm": ramp_scaled_norm,
        "tarc_scaled_gradient_norm": tarc_scaled_norm,
        "target_auxiliary_gradient_norm": target_norm,
        "maximum_relative_budget_error": max(budget_errors),
    }
    split_manifest = {
        "name": "step3_tarc_p0_category_balanced_prompt_pair_split_v1",
        "selection_seed": protocol["selection"]["seed"],
        "calibration_sample_ids": protocol["selection"]["calibration_sample_ids"],
        "validation_sample_ids": protocol["selection"]["validation_sample_ids"],
        "targets_per_prompt": 2,
        "prompt_overlap": 0,
    }
    diagnostics = {
        "name": "step3_tarc_p0_gradient_diagnostics_v1",
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
        "name": "step3_tarc_p0_route_decision_v1",
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
        == "PASS_TARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        "core100_opened": False,
        "heldout_opened": False,
    }
    report = f"""# TARC-P0 Zero-Update Directional Mechanism Report

Route: `{route}`.

All `{structural['rows']}` approved targets completed a registered refusal route
within `{structural['maximum_route_tokens']}` semantic tokens. Prompt tries had
branching rate `{structural['branching_prompt_rate']:.3f}` and distinct-route
pair rate `{structural['distinct_route_pair_rate']:.3f}`.

RAMP and TARC auxiliary gradients were independently calibrated to `{rho:.3f}`
of the uniform gradient norm. Coefficients were `{ramp_coefficient:.6g}` and
`{tarc_coefficient:.6g}`; maximum relative budget error was
`{max(budget_errors):.3e}`.

TARC calibration-to-validation cosine was
`{directional['calibration_validation_tarc_cosine']:.6f}` versus
`{directional['ramp_calibration_to_tarc_validation_cosine']:.6f}` for RAMP,
giving TARC's registered margin
`{directional['tarc_over_ramp_validation_cosine_margin']:.6f}`. Equal-budget
TARC and RAMP validation-gradient dots were
`{directional['tarc_tarc_validation_dot']:.6f}` and
`{directional['ramp_tarc_validation_dot']:.6f}`.

This is a first-order, prompt-disjoint initialization test, not generated
behavior or a safety/utility conclusion. No optimizer, scheduler, checkpoint,
or candidate output was created. `optimizer_steps=0`,
`training_authorized=false`, `core100_opened=false`, and
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
        resolve_repo_path(protocol["outputs"]["diagnostic_rows"]), diagnostic_rows
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
        "name": "step3_tarc_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": sha256(path),
        "artifacts": package_artifacts(protocol),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_tarc_p0.py run"
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
        raise TarcP0ValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise TarcP0ValidationError(f"artifact mismatch: {artifact['path']}")
    decision = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    recomputed = tarc.select_route(decision["gates"])
    if recomputed != decision.get("route") or recomputed != reproduction.get("route"):
        raise TarcP0ValidationError("stored TARC-P0 routes differ")
    result = {
        "status": "PASS_TARC_P0_RESULT_VERIFICATION",
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
    try:
        if args.command == "audit-source":
            audit_source(resolve_repo_path(args.source), args.selection_seed)
            return
        path = resolve_repo_path(args.protocol)
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        TarcP0ValidationError,
        ramp_p0.RampP0ValidationError,
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
        raise SystemExit(f"STOP_TARC_P0_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
