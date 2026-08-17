"""Run the zero-update CC-E-TARC mechanism and continuation gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_common as gradient_common
import step3_cita_p0 as model_common
import step3_etarc_common as etarc
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_tarc_common as tarc
import step3_tarc_p0 as tarc_p0


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_etarc_p0/"
    "protocol/etarc_p0_protocol.json"
)


class EtarcP0ValidationError(ValueError):
    """Raised when CC-E-TARC P0 violates its prospective protocol."""


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
        raise EtarcP0ValidationError(f"path escapes repository: {value}") from exc
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


def write_bytes_frozen(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_json_frozen(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    write_bytes_frozen(path, text.encode("utf-8"))


def write_jsonl_frozen(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows
    )
    write_bytes_frozen(path, text.encode("utf-8"))


def write_text_frozen(path: Path, text: str) -> None:
    write_bytes_frozen(path, text.replace("\r\n", "\n").encode("utf-8"))


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def select_calibration_and_remainder(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, list[str]]:
    base = tarc_p0.select_diverse_prompt_pairs(rows, seed)
    calibration = list(base["calibration"])
    all_ids = {str(row["sample_id"]) for row in rows}
    validation = sorted(
        all_ids.difference(calibration),
        key=lambda value: tarc_p0.stable_score(seed, value),
    )
    if len(calibration) != 8 or len(validation) != 32:
        raise EtarcP0ValidationError("E-TARC split must cover 8+32 prompt clusters")
    if set(calibration).intersection(validation):
        raise EtarcP0ValidationError("E-TARC calibration and validation overlap")
    return {"calibration": calibration, "validation": validation}


def context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_ETARC_model_forward_or_gradient":
        raise EtarcP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise EtarcP0ValidationError(f"frozen input hash mismatch: {name}")
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
        raise EtarcP0ValidationError("protocol violates the zero-update boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise EtarcP0ValidationError("entry commit is not an ancestor of HEAD")
    rows = read_jsonl(resolve_repo_path(protocol["source"]["targets_path"]))
    counts = protocol["source"]["counts"]
    ramp_p0.validate_source_rows(
        rows,
        int(counts["rows"]),
        int(counts["prompt_clusters"]),
        int(counts["categories"]),
    )
    selected = select_calibration_and_remainder(
        rows, int(protocol["selection"]["seed"])
    )
    for split in ("calibration", "validation"):
        if selected[split] != list(protocol["selection"][f"{split}_sample_ids"]):
            raise EtarcP0ValidationError(f"{split} selection differs")
    return protocol, rows


def verify_structural_freeze(audit: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected = protocol["design"]["tokenizer_only_structural_audit"]
    for key, value in expected.items():
        observed = audit.get(key)
        if isinstance(value, float):
            if not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
                raise EtarcP0ValidationError(f"structural audit drift: {key}")
        elif observed != value:
            raise EtarcP0ValidationError(f"structural audit drift: {key}")


def split_pairs(
    pairs: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    by_id = {str(pair["sample_id"]): pair for pair in pairs}
    return {
        split: [by_id[value] for value in protocol["selection"][f"{split}_sample_ids"]]
        for split in ("calibration", "validation")
    }


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise EtarcP0ValidationError("E-TARC P0 requires a clean worktree")
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
            raise EtarcP0ValidationError(f"prospective output already exists: {output}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    if not torch.cuda.is_available():
        raise EtarcP0ValidationError("CUDA is unavailable")
    tokenizer = tarc_p0.load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    verify_structural_freeze(audit, protocol)
    splits = split_pairs(pairs, protocol)
    result = {
        "status": "PASS_ETARC_P0_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "source_rows": len(rows),
        "prompt_clusters": len(pairs),
        "calibration_prompts": len(splits["calibration"]),
        "validation_prompts": len(splits["validation"]),
        "prompt_overlap": 0,
        "all_prompt_clusters_accounted_for": len(splits["calibration"])
        + len(splits["validation"])
        == len(pairs),
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


def _tarc_route_margins(
    response_logits: list[Any],
    pair: dict[str, Any],
    temperature: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    margins = []
    unique_node_margins = []
    strongest_weights = []
    weakest_weights = []
    branch_gaps = []
    for node in pair["nodes"]:
        node_values = []
        for path_index in node["path_indices"]:
            logits = response_logits[path_index][int(node["depth"])]
            allowed = list(node["allowed_token_ids"])
            value = tarc.conservative_allowed_margin(logits, allowed, temperature)
            margins.append(value)
            node_values.append(value)
            if len(allowed) > 1:
                indices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
                allowed_logits = logits.float().index_select(0, indices)
                strongest = int(torch.argmax(allowed_logits).detach())
                weights = torch.softmax(-allowed_logits / temperature, dim=0)
                strongest_weights.append(float(weights[strongest].detach().cpu()))
                weakest_weights.append(float(weights.min().detach().cpu()))
                branch_gaps.append(
                    float(torch.topk(allowed_logits.detach(), 2).values.diff().abs().cpu())
                )
        unique_node_margins.append(torch.stack(node_values).mean())
    margin_tensor = torch.stack(margins)
    detached = margin_tensor.detach().float()
    loss = tarc.certificate_barrier(margin_tensor)
    details = {
        "margin_terms": int(margin_tensor.numel()),
        "unique_trie_nodes": len(pair["nodes"]),
        "mean_margin": float(detached.mean().cpu()),
        "minimum_margin": float(detached.min().cpu()),
        "certified_node_occurrence_rate": float((detached > 0).float().mean().cpu()),
        "full_trie_certified": bool(
            (torch.stack(unique_node_margins).detach() > 0).all().cpu()
        ),
        "branch_occurrences": len(strongest_weights),
        "mean_strongest_allowed_derivative_weight": statistics.fmean(strongest_weights),
        "mean_minimum_allowed_derivative_weight": statistics.fmean(weakest_weights),
        "mean_allowed_logit_gap": statistics.fmean(branch_gaps),
    }
    return loss, details


def _etarc_path_certificate(
    response_logits: list[Any],
    response_targets: list[Any],
    pair: dict[str, Any],
    temperature: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    path_token_margins = []
    for path_index, feature in enumerate(pair["features"]):
        depth = int(feature["route_depth"])
        route_targets = response_targets[path_index][:depth]
        expected = torch.tensor(
            pair["paths"][path_index],
            dtype=torch.long,
            device=route_targets.device,
        )
        if not torch.equal(route_targets, expected):
            raise EtarcP0ValidationError("registered path and response target differ")
        margins, _ = gradient_math.strongest_competitor_margin(
            response_logits[path_index][:depth], route_targets
        )
        path_token_margins.append(margins)
    certificate, path_scores = etarc.existential_path_certificate(
        path_token_margins,
        bottleneck_temperature=temperature,
        path_temperature=temperature,
    )
    loss = etarc.path_certificate_barrier(certificate)
    detached_scores = path_scores.detach().float()
    detached_margins = [value.detach().float() for value in path_token_margins]
    path_weights = torch.softmax(detached_scores / temperature, dim=0)
    winning_path = int(torch.argmax(detached_scores))
    winning_margins = detached_margins[winning_path]
    top_scores = torch.topk(detached_scores, k=2).values
    return loss, {
        "registered_paths": len(path_token_margins),
        "margin_terms": sum(int(value.numel()) for value in path_token_margins),
        "mean_token_margin": statistics.fmean(
            float(value.mean().cpu()) for value in detached_margins
        ),
        "minimum_token_margin": min(
            float(value.min().cpu()) for value in detached_margins
        ),
        "mean_conservative_path_score": float(detached_scores.mean().cpu()),
        "maximum_conservative_path_score": float(detached_scores.max().cpu()),
        "existential_path_certificate": float(certificate.detach().float().cpu()),
        "complete_route_certified": bool(certificate.detach() > 0),
        "winning_path_index": winning_path,
        "winning_path_minimum_token_margin": float(winning_margins.min().cpu()),
        "winning_path_all_token_margins_positive": bool((winning_margins > 0).all()),
        "strongest_path_derivative_weight": float(path_weights.max().cpu()),
        "weakest_path_derivative_weight": float(path_weights.min().cpu()),
        "conservative_path_score_gap": float((top_scores[0] - top_scores[1]).cpu()),
    }


def _branch_objective(
    response_logits: list[Any], pair: dict[str, Any], temperature: float, objective: str
) -> tuple[Any, dict[str, Any]]:
    import torch

    values = []
    entropies = []
    gaps = []
    occurrence_spreads = []
    for node in pair["nodes"]:
        allowed = list(node["allowed_token_ids"])
        if len(allowed) < 2:
            continue
        occurrence_logits = []
        for path_index in node["path_indices"]:
            logits = response_logits[path_index][int(node["depth"])]
            indices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            occurrence_logits.append(logits.float().index_select(0, indices))
        stacked = torch.stack(occurrence_logits)
        mean_logits = stacked.mean(dim=0)
        occurrence_spreads.append(float((stacked.max(0).values - stacked.min(0).values).abs().max().detach().cpu()))
        probabilities = torch.softmax(mean_logits / temperature, dim=0)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
        top = torch.topk(mean_logits, k=2).values
        gap = top[0] - top[1]
        entropies.append(float(entropy.detach().cpu()))
        gaps.append(float(gap.detach().cpu()))
        values.append(entropy if objective == "branch_entropy" else -gap)
    if not values:
        raise EtarcP0ValidationError("pair has no branching trie node")
    loss = torch.stack(values).mean()
    return loss, {
        "branching_nodes": len(values),
        "mean_allowed_entropy": statistics.fmean(entropies),
        "mean_top1_top2_allowed_gap": statistics.fmean(gaps),
        "maximum_shared_history_logit_spread": max(occurrence_spreads),
        "mean_branch_metric": float(loss.detach().cpu()),
    }


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
    continuation_losses = []
    continuation_tokens = []
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
        positions = [
            int(value)
            for value in feature["semantic_positions"]
            if int(value) >= int(feature["route_depth"])
        ]
        if not positions:
            raise EtarcP0ValidationError("approved target has no post-route continuation")
        position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        continuation_losses.append(
            functional.cross_entropy(
                current_logits.float().index_select(0, position_tensor),
                current_targets.index_select(0, position_tensor),
            )
        )
        continuation_tokens.append(len(positions))
    if objective == "uniform":
        loss = torch.stack(uniform_losses).mean()
        return loss, {"mean_uniform_loss": float(loss.detach().cpu())}
    if objective == "ramp":
        loss = torch.stack(ramp_losses).mean()
        return loss, {
            "margin_terms": 2 * ramp_width,
            "mean_margin_loss": float(loss.detach().cpu()),
        }
    if objective == "continuation":
        loss = torch.stack(continuation_losses).mean()
        return loss, {
            "continuation_targets": len(continuation_losses),
            "continuation_tokens": sum(continuation_tokens),
            "mean_continuation_loss": float(loss.detach().cpu()),
        }
    if objective == "tarc":
        return _tarc_route_margins(response_logits, pair, temperature)
    if objective == "etarc":
        return _etarc_path_certificate(
            response_logits, response_targets, pair, temperature
        )
    if objective in ("branch_entropy", "branch_negative_gap"):
        return _branch_objective(response_logits, pair, temperature, objective)
    raise ValueError(f"unknown E-TARC P0 objective: {objective}")


def _coverage_from_gradient(
    gradients: dict[str, Any], parameters: list[tuple[str, Any]], missing: set[str]
) -> dict[str, Any]:
    import torch

    finite = all(bool(torch.isfinite(value).all()) for value in gradients.values())
    counts = [int(torch.count_nonzero(value)) for value in gradients.values()]
    return {
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(parameter.numel() for _, parameter in parameters),
        "missing_gradient_tensors": sorted(missing),
        "finite": finite,
        "nonzero_gradient_tensors": sum(value > 0 for value in counts),
        "nonzero_gradient_elements": sum(counts),
    }


def aggregate_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    pairs: list[dict[str, Any]],
    split_name: str,
    objective: str,
    ramp_width: int,
    temperature: float,
    reference_directions: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    capture_pairs = reference_directions is not None
    rows = []
    losses = []
    missing: set[str] = set()
    aggregate = None
    model.zero_grad(set_to_none=True)
    for position, pair in enumerate(pairs, start=1):
        if capture_pairs:
            model.zero_grad(set_to_none=True)
        loss, metrics = pair_objective(
            model, pair, objective, ramp_width, temperature
        )
        losses.append(float(loss.detach().cpu()))
        (loss if capture_pairs else loss / len(pairs)).backward()
        row = {
            "row_type": "objective_pair",
            "split": split_name,
            "objective": objective,
            "sample_id": pair["sample_id"],
            "category": pair["category"],
            **metrics,
        }
        if capture_pairs:
            pair_gradient, pair_coverage = model_common.capture_gradients(parameters)
            missing.update(pair_coverage["missing_gradient_tensors"])
            if aggregate is None:
                aggregate = {
                    name: value / len(pairs) for name, value in pair_gradient.items()
                }
            else:
                for name, value in pair_gradient.items():
                    aggregate[name].add_(value, alpha=1.0 / len(pairs))
            pair_norm = gradient_math.gradient_norm(pair_gradient)
            row["pair_gradient_norm"] = pair_norm
            row["directional"] = {
                name: {
                    "dot": gradient_math.gradient_dot(pair_gradient, direction),
                    "cosine": gradient_math.gradient_cosine(pair_gradient, direction),
                }
                for name, direction in reference_directions.items()
            }
            del pair_gradient
        rows.append(row)
        print(
            f"E-TARC P0 gradients {split_name}/{objective} {position}/{len(pairs)}",
            flush=True,
        )
    if aggregate is None:
        aggregate, captured = model_common.capture_gradients(parameters)
        missing.update(captured["missing_gradient_tensors"])
    coverage = _coverage_from_gradient(aggregate, parameters, missing)
    coverage.update(
        {
            "mean_loss": statistics.fmean(losses),
            "gradient_norm": gradient_math.gradient_norm(aggregate),
            "gradient_hash": gradient_common.gradient_hash(aggregate),
        }
    )
    if objective == "tarc":
        coverage.update(
            {
                "mean_margin": statistics.fmean(float(row["mean_margin"]) for row in rows),
                "minimum_margin": min(float(row["minimum_margin"]) for row in rows),
                "certified_node_occurrence_rate": statistics.fmean(
                    float(row["certified_node_occurrence_rate"]) for row in rows
                ),
                "full_trie_certificate_rate": statistics.fmean(
                    int(row["full_trie_certified"]) for row in rows
                ),
                "mean_strongest_allowed_derivative_weight": statistics.fmean(
                    float(row["mean_strongest_allowed_derivative_weight"])
                    for row in rows
                ),
            }
        )
    if objective == "etarc":
        coverage.update(
            {
                "mean_existential_path_certificate": statistics.fmean(
                    float(row["existential_path_certificate"]) for row in rows
                ),
                "minimum_existential_path_certificate": min(
                    float(row["existential_path_certificate"]) for row in rows
                ),
                "complete_route_certificate_rate": statistics.fmean(
                    int(row["complete_route_certified"]) for row in rows
                ),
                "mean_strongest_path_derivative_weight": statistics.fmean(
                    float(row["strongest_path_derivative_weight"]) for row in rows
                ),
                "mean_winning_path_minimum_token_margin": statistics.fmean(
                    float(row["winning_path_minimum_token_margin"]) for row in rows
                ),
            }
        )
    model.zero_grad(set_to_none=True)
    return aggregate, coverage, rows


def scaled_directions(
    calibration: dict[str, dict[str, Any]], ratio: float
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cc_raw, projection = etarc.continuation_compatible_projection(
        calibration["etarc"], calibration["continuation"]
    )
    raw = {
        "ramp": calibration["ramp"],
        "tarc": calibration["tarc"],
        "etarc": calibration["etarc"],
        "cc_etarc": cc_raw,
    }
    uniform_norm = gradient_math.gradient_norm(calibration["uniform"])
    target_norm = ratio * uniform_norm
    coefficients = {}
    directions = {}
    errors = []
    for name, gradient in raw.items():
        raw_norm = gradient_math.gradient_norm(gradient)
        coefficient = gradient_math.calibrated_coefficient(
            uniform_norm, raw_norm, ratio
        )
        scaled = gradient_math.scaled_gradient(gradient, coefficient)
        scaled_norm = gradient_math.gradient_norm(scaled)
        coefficients[name] = {
            "raw_gradient_norm": raw_norm,
            "coefficient": coefficient,
            "scaled_gradient_norm": scaled_norm,
        }
        directions[name] = scaled
        errors.append(abs(scaled_norm - target_norm) / target_norm)
    coefficient_record = {
        "auxiliary_gradient_norm_ratio": ratio,
        "uniform_calibration_gradient_norm": uniform_norm,
        "target_auxiliary_gradient_norm": target_norm,
        "directions": coefficients,
        "maximum_relative_budget_error": max(errors),
    }
    return directions, coefficient_record, projection


def directional_metrics(
    calibration: dict[str, dict[str, Any]],
    validation: dict[str, dict[str, Any]],
    directions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "calibration_etarc_tarc_cosine": gradient_math.gradient_cosine(
            calibration["etarc"], calibration["tarc"]
        ),
        "calibration_validation_etarc_cosine": gradient_math.gradient_cosine(
            calibration["etarc"], validation["etarc"]
        ),
        "calibration_validation_tarc_cosine": gradient_math.gradient_cosine(
            calibration["tarc"], validation["tarc"]
        ),
        "candidate_directions": {},
    }
    for name, direction in directions.items():
        combined = gradient_math.add_gradients(calibration["uniform"], direction)
        result["candidate_directions"][name] = {
            "validation_etarc_dot": gradient_math.gradient_dot(
                validation["etarc"], direction
            ),
            "validation_etarc_cosine": gradient_math.gradient_cosine(
                validation["etarc"], direction
            ),
            "validation_tarc_dot": gradient_math.gradient_dot(
                validation["tarc"], direction
            ),
            "validation_continuation_dot": gradient_math.gradient_dot(
                validation["continuation"], direction
            ),
            "validation_continuation_cosine": gradient_math.gradient_cosine(
                validation["continuation"], direction
            ),
            "validation_branch_entropy_dot": gradient_math.gradient_dot(
                validation["branch_entropy"], direction
            ),
            "validation_branch_negative_gap_dot": gradient_math.gradient_dot(
                validation["branch_negative_gap"], direction
            ),
            "combined_validation_etarc_dot": gradient_math.gradient_dot(
                validation["etarc"], combined
            ),
            "combined_validation_continuation_dot": gradient_math.gradient_dot(
                validation["continuation"], combined
            ),
        }
    cc = result["candidate_directions"]["cc_etarc"]
    raw = result["candidate_directions"]["etarc"]
    old = result["candidate_directions"]["tarc"]
    cc["route_dot_fraction_of_raw_etarc"] = (
        cc["validation_etarc_dot"] / raw["validation_etarc_dot"]
        if raw["validation_etarc_dot"] > 0
        else math.nan
    )
    cc["route_dot_fraction_of_tarc"] = (
        cc["validation_etarc_dot"] / old["validation_etarc_dot"]
        if old["validation_etarc_dot"] > 0
        else math.nan
    )
    cc["continuation_dot_advantage_over_tarc"] = (
        cc["validation_continuation_dot"]
        - old["validation_continuation_dot"]
    )
    return result


def paired_bootstrap(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    if not values or samples <= 0:
        raise ValueError("paired bootstrap requires values and samples")
    rng = random.Random(seed)
    size = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    lower = draws[max(0, math.floor(0.025 * samples) - 1)]
    upper = draws[min(samples - 1, math.ceil(0.975 * samples) - 1)]
    return {
        "rows": size,
        "mean": statistics.fmean(values),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def exact_sign_test(values: list[float], tolerance: float = 1e-12) -> dict[str, Any]:
    positive = sum(value > tolerance for value in values)
    negative = sum(value < -tolerance for value in values)
    ties = len(values) - positive - negative
    trials = positive + negative
    if trials == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(trials, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**trials))
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "exact_two_sided_p": p_value,
    }


def summarize_pair_directions(
    rows: list[dict[str, Any]], bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    by_objective = {
        objective: [
            row
            for row in rows
            if row["split"] == "validation" and row["objective"] == objective
        ]
        for objective in ("etarc", "continuation")
    }
    if not by_objective["etarc"] or len(by_objective["etarc"]) != len(
        by_objective["continuation"]
    ):
        raise EtarcP0ValidationError("pair-direction rows are incomplete")
    summary: dict[str, Any] = {"candidates": {}}
    for candidate in ("ramp", "tarc", "etarc", "cc_etarc"):
        route_dots = [
            float(row["directional"][candidate]["dot"])
            for row in by_objective["etarc"]
        ]
        continuation_dots = [
            float(row["directional"][candidate]["dot"])
            for row in by_objective["continuation"]
        ]
        summary["candidates"][candidate] = {
            "validation_prompts": len(route_dots),
            "route_positive_rate": sum(value > 0 for value in route_dots)
            / len(route_dots),
            "continuation_compatible_rate": sum(value >= 0 for value in continuation_dots)
            / len(continuation_dots),
            "mean_route_dot": statistics.fmean(route_dots),
            "median_route_dot": statistics.median(route_dots),
            "mean_continuation_dot": statistics.fmean(continuation_dots),
            "median_continuation_dot": statistics.median(continuation_dots),
        }
    cc_continuation = [
        float(row["directional"]["cc_etarc"]["dot"])
        for row in by_objective["continuation"]
    ]
    tarc_continuation = [
        float(row["directional"]["tarc"]["dot"])
        for row in by_objective["continuation"]
    ]
    differences = [
        left - right
        for left, right in zip(cc_continuation, tarc_continuation, strict=True)
    ]
    summary["cc_etarc_continuation_advantage_over_tarc"] = {
        "bootstrap": paired_bootstrap(
            differences, bootstrap_samples, bootstrap_seed
        ),
        "sign_test": exact_sign_test(differences),
    }
    summary["cc_etarc_compatible_rate_advantage_over_tarc"] = (
        summary["candidates"]["cc_etarc"]["continuation_compatible_rate"]
        - summary["candidates"]["tarc"]["continuation_compatible_rate"]
    )
    return summary


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
    tokenizer = tarc_p0.load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    all_pairs, structural = tarc_p0.build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    verify_structural_freeze(structural, protocol)
    splits = split_pairs(all_pairs, protocol)
    started = time.time()
    model, model_tokenizer = model_common.load_model(protocol)
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise EtarcP0ValidationError("tokenizer-only and model tokenizers differ")
    del model_tokenizer, tokenizer
    parameters = model_common.trainable_parameters(model)
    initial_state = model_common.trainable_state_hash(parameters)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ramp_width = int(protocol["design"]["ramp_width"])
    temperature = float(protocol["design"]["temperature"])

    gradients: dict[str, dict[str, dict[str, Any]]] = {
        "calibration": {},
        "validation": {},
    }
    coverage: dict[str, dict[str, dict[str, Any]]] = {
        "calibration": {},
        "validation": {},
    }
    diagnostic_rows = []
    calibration_objectives = ("uniform", "ramp", "tarc", "etarc", "continuation")
    for objective in calibration_objectives:
        gradient, details, objective_rows = aggregate_gradient(
            model,
            parameters,
            splits["calibration"],
            "calibration",
            objective,
            ramp_width,
            temperature,
        )
        gradients["calibration"][objective] = gradient
        coverage["calibration"][objective] = details
        diagnostic_rows.extend(objective_rows)

    ratio = float(protocol["design"]["auxiliary_gradient_norm_ratio"])
    directions, coefficients, projection = scaled_directions(
        gradients["calibration"], ratio
    )
    validation_objectives = (
        "uniform",
        "ramp",
        "tarc",
        "etarc",
        "continuation",
        "branch_entropy",
        "branch_negative_gap",
    )
    for objective in validation_objectives:
        pair_references = directions if objective in ("etarc", "continuation") else None
        gradient, details, objective_rows = aggregate_gradient(
            model,
            parameters,
            splits["validation"],
            "validation",
            objective,
            ramp_width,
            temperature,
            pair_references,
        )
        gradients["validation"][objective] = gradient
        coverage["validation"][objective] = details
        diagnostic_rows.extend(objective_rows)

    directional = directional_metrics(
        gradients["calibration"], gradients["validation"], directions
    )
    pair_summary = summarize_pair_directions(
        diagnostic_rows,
        int(protocol["statistics"]["bootstrap_samples"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
    final_state = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    records = [
        coverage[split][objective]
        for split in coverage
        for objective in coverage[split]
    ]
    gate_cfg = protocol["gates"]
    cc_metrics = directional["candidate_directions"]["cc_etarc"]
    pair_cc = pair_summary["candidates"]["cc_etarc"]
    bootstrap = pair_summary["cc_etarc_continuation_advantage_over_tarc"][
        "bootstrap"
    ]
    gates = {
        "integrity": (
            structural["rows"] == int(protocol["source"]["counts"]["rows"])
            and len(splits["calibration"])
            == int(protocol["selection"]["calibration_prompts"])
            and len(splits["validation"])
            == int(protocol["selection"]["validation_prompts"])
            and len(splits["calibration"]) + len(splits["validation"])
            == int(protocol["source"]["counts"]["prompt_clusters"])
        ),
        "gradient_valid": all(
            not row["missing_gradient_tensors"]
            and row["finite"]
            and float(row["gradient_norm"]) > 0
            for row in records
        ),
        "budget_matched": coefficients["maximum_relative_budget_error"]
        <= float(gate_cfg["gradient_budget_relative_tolerance"]),
        "vram": headroom >= float(gate_cfg["minimum_vram_headroom_fraction"]),
        "state_restored": initial_state == final_state,
        "route_structure": (
            structural["all_routes_complete_within_width"]
            and structural["branching_prompt_rate"]
            >= float(gate_cfg["branching_prompt_rate_min"])
            and structural["distinct_route_pair_rate"]
            >= float(gate_cfg["distinct_route_pair_rate_min"])
        ),
        "objective_distinct": (
            math.isfinite(directional["calibration_etarc_tarc_cosine"])
            and directional["calibration_etarc_tarc_cosine"]
            <= float(gate_cfg["calibration_etarc_tarc_cosine_max"])
            and projection["projected_gradient_norm"] > 0
        ),
        "route_transfer": (
            directional["calibration_validation_etarc_cosine"]
            >= float(gate_cfg["calibration_validation_etarc_cosine_min"])
            and cc_metrics["validation_etarc_dot"] > 0
        ),
        "continuation_compatibility": (
            projection["projected_continuation_dot"]
            >= -float(gate_cfg["projection_numeric_tolerance"])
            and cc_metrics["validation_continuation_dot"]
            >= float(gate_cfg["validation_continuation_dot_min"])
            and pair_cc["continuation_compatible_rate"]
            >= float(gate_cfg["continuation_compatible_pair_rate_min"])
            and pair_summary["cc_etarc_compatible_rate_advantage_over_tarc"]
            >= float(gate_cfg["compatible_rate_advantage_over_tarc_min"])
            and bootstrap["ci95_lower"]
            >= float(gate_cfg["continuation_advantage_bootstrap_lower_min"])
        ),
        "effect_not_weak": (
            pair_cc["route_positive_rate"]
            >= float(gate_cfg["route_positive_pair_rate_min"])
            and cc_metrics["route_dot_fraction_of_raw_etarc"]
            >= float(gate_cfg["route_dot_fraction_of_raw_etarc_min"])
            and cc_metrics["route_dot_fraction_of_tarc"]
            >= float(gate_cfg["route_dot_fraction_of_tarc_min"])
        ),
    }
    route = etarc.select_route(gates)
    split_manifest = {
        "name": "step3_etarc_p0_calibration8_validation32_split_v1",
        "selection_seed": protocol["selection"]["seed"],
        "calibration_sample_ids": protocol["selection"]["calibration_sample_ids"],
        "validation_sample_ids": protocol["selection"]["validation_sample_ids"],
        "targets_per_prompt": 2,
        "prompt_overlap": 0,
        "all_source_prompts_accounted_for": True,
    }
    diagnostics = {
        "name": "step3_etarc_p0_gradient_diagnostics_v1",
        "coverage": coverage,
        "coefficients": coefficients,
        "projection": projection,
        "directional": directional,
        "paired_validation": pair_summary,
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
        "name": "step3_etarc_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": sha256(path),
        "gates": gates,
        "structural_audit": structural,
        "coefficients": coefficients,
        "projection": projection,
        "directional": directional,
        "paired_validation": pair_summary,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "training_authorized": False,
        "matched_smoke_protocol_authorized": route
        == "PASS_ETARC_P0_AUTHORIZE_CC_MATCHED_SMOKE_PROTOCOL",
        "core100_opened": False,
        "heldout_opened": False,
    }
    report = f"""# CC-E-TARC P0 Zero-Update Mechanism Report

Route: `{route}`.

## Design integrity

The fixed split used `{len(splits['calibration'])}` calibration prompts and all
remaining `{len(splits['validation'])}` approved prompt clusters for disjoint
validation. All `{structural['rows']}` approved targets reached a registered
route within `{structural['maximum_route_tokens']}` semantic tokens. The model
state was restored exactly: `{initial_state == final_state}`.

All four auxiliary directions were independently normalized to `{ratio:.3f}`
of the calibration uniform-gradient norm. Maximum relative norm error was
`{coefficients['maximum_relative_budget_error']:.3e}`. The E-TARC/TARC
calibration cosine was `{directional['calibration_etarc_tarc_cosine']:.6f}`.

## Route mechanism

E-TARC calibration-to-validation cosine was
`{directional['calibration_validation_etarc_cosine']:.6f}`. Equal-budget
CC-E-TARC retained `{cc_metrics['route_dot_fraction_of_raw_etarc']:.3f}` of
raw E-TARC's validation route dot and
`{cc_metrics['route_dot_fraction_of_tarc']:.3f}` of TARC's route dot.
Its positive per-prompt route-dot rate was
`{pair_cc['route_positive_rate']:.3f}`.

For the branch diagnostics, positive entropy dot predicts entropy reduction
under gradient descent, while positive negative-gap dot predicts an increased
top1-versus-top2 legal-token gap. CC-E-TARC values were
`{cc_metrics['validation_branch_entropy_dot']:.6f}` and
`{cc_metrics['validation_branch_negative_gap_dot']:.6f}`.

## Continuation compatibility

Calibration projection active: `{projection['projection_active']}`. The raw
route/continuation dot was
`{projection['raw_route_continuation_dot']:.6f}` and the projected residual
was `{projection['projected_continuation_dot']:.3e}`. On 32 unseen prompts,
CC-E-TARC's aggregate continuation dot was
`{cc_metrics['validation_continuation_dot']:.6f}` and its compatible-prompt
rate was `{pair_cc['continuation_compatible_rate']:.3f}`, an advantage of
`{pair_summary['cc_etarc_compatible_rate_advantage_over_tarc']:.3f}` over
TARC. The paired continuation-dot advantage was
`{bootstrap['mean']:.6f}` with bootstrap 95% interval
`[{bootstrap['ci95_lower']:.6f}, {bootstrap['ci95_upper']:.6f}]`.

## Claim boundary

This is a real-model, prompt-disjoint first-order result at fresh LoRA step 0.
It is not generated behavior, semantic safety, utility, or a paper-level
effect endpoint. No optimizer, scheduler, checkpoint, or candidate output was
created. `optimizer_steps=0`, `training_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
"""
    protocol_sha = sha256(path)
    write_json_frozen(
        resolve_repo_path(protocol["outputs"]["structural_audit"]), structural
    )
    write_json_frozen(
        resolve_repo_path(protocol["outputs"]["split_manifest"]), split_manifest
    )
    write_jsonl_frozen(
        resolve_repo_path(protocol["outputs"]["diagnostic_rows"]), diagnostic_rows
    )
    write_json_frozen(
        resolve_repo_path(protocol["outputs"]["gradient_diagnostics"]), diagnostics
    )
    write_json_frozen(
        resolve_repo_path(protocol["outputs"]["environment"]),
        environment_record(runtime_head, protocol_sha),
    )
    write_json_frozen(
        resolve_repo_path(protocol["outputs"]["route"]), decision
    )
    write_text_frozen(
        resolve_repo_path(protocol["outputs"]["report"]), report
    )
    reproduction = {
        "name": "step3_etarc_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": protocol_sha,
        "artifacts": package_artifacts(protocol),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_etarc_p0.py run"
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
    write_json_frozen(
        resolve_repo_path(protocol["outputs"]["run_manifest"]), reproduction
    )
    del gradients, directions, parameters, model
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
        raise EtarcP0ValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise EtarcP0ValidationError(f"artifact mismatch: {artifact['path']}")
    decision = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    recomputed = etarc.select_route(decision["gates"])
    if recomputed != decision.get("route") or recomputed != reproduction.get("route"):
        raise EtarcP0ValidationError("stored E-TARC P0 routes differ")
    result = {
        "status": "PASS_ETARC_P0_RESULT_VERIFICATION",
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
        path = resolve_repo_path(args.protocol)
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        EtarcP0ValidationError,
        tarc_p0.TarcP0ValidationError,
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
        raise SystemExit(f"STOP_ETARC_P0_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
