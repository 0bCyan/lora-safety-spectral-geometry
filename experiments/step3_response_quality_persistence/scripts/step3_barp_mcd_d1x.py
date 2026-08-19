"""Run the BARP-MCD D1X cross-fitted zero-step route qualification."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_barp_mcd_d1 as d1
import step3_barp_mcd_d1r as d1r
import step3_barp_mcd_training as base
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_pfrsc_common as pfrsc
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_d1x/protocol/barp_mcd_d1x_protocol.json"
)
STATUS = "prospective_before_any_BARP_MCD_D1X_model_forward"
TARGET_MASKS = tuple(
    mask
    for mask in range(1, 2 ** len(rp.CERTIFICATES))
    if mask & ~sum(1 << rp.CERTIFICATES.index(name) for name in rp.TARGET_CERTIFICATES)
    == 0
)
ROUTE_FRACTIONS = rp.HOMOTOPY_FRACTIONS[1:]
STABILITY_COEFFICIENT = 2.3533634348018264


class D1XValidationError(ValueError):
    """Raised when D1X violates its prospective registry."""


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
        raise D1XValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def route_id(mask: int, fraction: float) -> str:
    return f"mask{mask:03d}_lambda{fraction:.3f}"


def route_registry() -> list[dict[str, Any]]:
    names = tuple(rp.CERTIFICATES)
    return [
        {
            "route_id": route_id(mask, fraction),
            "source_mask": mask,
            "source_subset": [
                name for index, name in enumerate(names) if mask & (1 << index)
            ],
            "homotopy_fraction": float(fraction),
        }
        for mask in TARGET_MASKS
        for fraction in ROUTE_FRACTIONS
    ]


def cluster_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise D1XValidationError("D1X requires four finite proposal clusters")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "count": len(values),
        "values": values,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "studentized_lower_score": mean - STABILITY_COEFFICIENT * standard_error,
        "population_confidence_interval_claimed": False,
        "positive": sum(value > 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
    }


def summarize_route(
    descriptor: dict[str, Any],
    proposal_rows: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    if len(proposal_rows) != 4:
        raise D1XValidationError("D1X route summary requires four proposals")
    cluster_values = [
        statistics.fmean(row["target_macro_deltas"]) for row in proposal_rows
    ]
    target = cluster_summary(cluster_values)
    protected = {}
    for name in rp.PROTECTED_CERTIFICATES:
        disadvantages = [
            -float(support["exact_delta_scores"][name])
            for row in proposal_rows
            for support in row["support_rows"]
        ]
        protected[name] = {
            "mean_disadvantage": statistics.fmean(disadvantages),
            "worst_disadvantage": max(disadvantages),
        }
    activity = sum(bool(row["route_exercised"]) for row in proposal_rows)
    minimum_retention = min(
        float(row["minimum_local_protected_retention"]) for row in proposal_rows
    )
    checks = {
        "activity": activity == int(gates["required_proposal_activity"]),
        "local_protection": minimum_retention + 1e-12
        >= float(gates["local_protected_retention_min"]),
        "target_lower": float(target["studentized_lower_score"])
        > float(gates["target_cluster_lower_min"]),
        "target_sign": int(target["positive"])
        >= int(gates["discovery_positive_clusters_min"]),
        "protected_mean": all(
            float(item["mean_disadvantage"])
            <= float(gates["protected_mean_disadvantage_max"])
            for item in protected.values()
        ),
        "protected_worst": all(
            float(item["worst_disadvantage"])
            <= float(gates["protected_worst_disadvantage_max"])
            for item in protected.values()
        ),
        "zero_optimizer_state": all(
            int(row["optimizer_completed_steps"]) == 0 for row in proposal_rows
        ),
    }
    return {
        **descriptor,
        "proposal_activity": activity,
        "minimum_local_protected_retention": minimum_retention,
        "target_clusters": target,
        "protected": protected,
        "checks": checks,
        "eligible": all(checks.values()),
        "proposal_rows": proposal_rows,
    }


def _discovery_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["target_clusters"]["studentized_lower_score"],
        row["target_clusters"]["mean"],
        min(row["target_clusters"]["values"]),
        row["minimum_local_protected_retention"],
        -row["homotopy_fraction"],
        -len(row["source_subset"]),
        -row["source_mask"],
    )


def select_discovery_route(
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [row for row in summaries if row["eligible"]]
    return max(eligible, key=_discovery_rank) if eligible else None


def qualification_checks(
    summary: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    target = summary["target_clusters"]
    protected = summary["protected"]
    return {
        "route_exercised": int(summary["proposal_activity"])
        == int(gates["required_proposal_activity"]),
        "local_protection": float(summary["minimum_local_protected_retention"])
        + 1e-12
        >= float(gates["local_protected_retention_min"]),
        "target_effect": float(target["mean"])
        >= float(gates["qualification_target_mean_min"]),
        "target_lower": float(target["studentized_lower_score"])
        > float(gates["target_cluster_lower_min"]),
        "target_sign": int(target["positive"])
        >= int(gates["qualification_positive_clusters_min"]),
        "protected_mean": all(
            float(item["mean_disadvantage"])
            <= float(gates["protected_mean_disadvantage_max"])
            for item in protected.values()
        ),
        "protected_worst": all(
            float(item["worst_disadvantage"])
            <= float(gates["protected_worst_disadvantage_max"])
            for item in protected.values()
        ),
        "zero_optimizer_state": bool(summary["checks"]["zero_optimizer_state"]),
    }


def select_route(
    discovery_selected: dict[str, Any] | None,
    qualification: dict[str, Any] | None,
    checks: dict[str, bool] | None,
) -> str:
    if discovery_selected is None:
        return "STOP_BARP_MCD_D1X_DISCOVERY_NO_STABLE_ROUTE"
    if qualification is None or checks is None:
        return "STOP_BARP_MCD_D1X_QUALIFICATION_INCOMPLETE"
    order = (
        ("zero_optimizer_state", "STOP_BARP_MCD_D1X_OPTIMIZER_STATE_DRIFT"),
        ("route_exercised", "STOP_BARP_MCD_D1X_ROUTE_NOT_EXERCISED"),
        ("local_protection", "STOP_BARP_MCD_D1X_LOCAL_PROTECTION_FAILED"),
        ("protected_mean", "STOP_BARP_MCD_D1X_TRANSFER_PROTECTION_FAILED"),
        ("protected_worst", "STOP_BARP_MCD_D1X_TRANSFER_PROTECTION_FAILED"),
        ("target_effect", "STOP_BARP_MCD_D1X_TARGET_EFFECT_TOO_WEAK"),
        ("target_lower", "STOP_BARP_MCD_D1X_TARGET_LOWER_BOUND_FAILED"),
        ("target_sign", "STOP_BARP_MCD_D1X_TARGET_SIGN_FAILED"),
    )
    for gate, route in order:
        if not checks.get(gate, False):
            return route
    return "PASS_BARP_MCD_D1X_AUTHORIZE_MATCHED_TRAINING_PROTOCOL"


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != STATUS:
        raise D1XValidationError("D1X protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise D1XValidationError(f"frozen input hash mismatch: {name}")
    if (
        int(protocol.get("optimizer_objects", -1)) != 1
        or int(protocol.get("optimizer_steps", -1)) != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("checkpoint_writes_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise D1XValidationError("D1X zero-step boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise D1XValidationError("D1X entry commit is not an ancestor")
    precursor = etarc_p0.read_json(
        resolve_repo_path(protocol["precursor"]["decision_path"])
    )
    if precursor.get("route") != "STOP_BARP_MCD_D1R_ASSIGNMENT_NOT_EXERCISED":
        raise D1XValidationError("D1X precursor route differs")
    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 6
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("model_optimizer_object_authorized") is not True
        or config.get("training_authorized") is not False
        or config.get("gradient_measurement_mode") != "eval"
        or int(config["gradient_accumulation_steps"]) != 4
        or float(config["weight_decay"]) != 0.0
    ):
        raise D1XValidationError("D1X config differs")
    design = protocol["design"]
    if (
        tuple(design["protected_certificates"]) != rp.PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != rp.TARGET_CERTIFICATES
        or tuple(design["route_source_masks"]) != TARGET_MASKS
        or tuple(float(value) for value in design["route_fractions"])
        != ROUTE_FRACTIONS
        or int(design["route_family_size"]) != 56
        or int(design["anchor_family_size"]) != 239
    ):
        raise D1XValidationError("D1X method registry differs")
    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise D1XValidationError("D1X risk source differs")
    blocks = protocol["selection"]["discovery_blocks"] + protocol["selection"][
        "qualification_blocks"
    ]
    diagnostic = protocol["selection"]["sealed_diagnostic_block"]
    training_safety = [value for block in blocks for value in block["safety_sample_ids"]]
    training_risk = [value for block in blocks for value in block["risk_row_indices"]]
    diagnostic_safety = list(diagnostic["safety_sample_ids"])
    diagnostic_risk = list(diagnostic["risk_row_indices"])
    if (
        len(blocks) != 8
        or any(len(block["safety_sample_ids"]) != 4 for block in blocks)
        or any(len(block["risk_row_indices"]) != 4 for block in blocks)
        or len(set(training_safety + diagnostic_safety)) != 40
        or len(set(training_risk + diagnostic_risk)) != 40
        or set(training_safety).intersection(diagnostic_safety)
        or set(training_risk).intersection(diagnostic_risk)
    ):
        raise D1XValidationError("D1X split overlaps or drifts")
    return protocol, config, targets, risk_rows


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise D1XValidationError("D1X requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([relative(path), protocol["config_path"]])
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    output_root = resolve_repo_path(protocol["output_root"])
    if output_root.exists():
        existing = [item for item in output_root.iterdir() if item.name != "protocol"]
        if existing:
            raise D1XValidationError("prospective D1X outputs already exist")
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise D1XValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    selected = [
        sample_id
        for key in ("discovery_blocks", "qualification_blocks")
        for block in protocol["selection"][key]
        for sample_id in block["safety_sample_ids"]
    ]
    sealed = protocol["selection"]["sealed_diagnostic_block"]["safety_sample_ids"]
    if len(set(selected + sealed)) != 40 or not all(value in pair_map for value in selected):
        raise D1XValidationError("D1X safety registry differs")
    result = {
        "status": "PASS_BARP_MCD_D1X_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "gpu": torch.cuda.get_device_name(0),
        "discovery_blocks": 4,
        "qualification_blocks": 4,
        "sealed_diagnostic_pairs": 8,
        "anchor_candidates_per_proposal": 239,
        "route_rules": 56,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _cpu_units(
    gradients: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    units, norms = frsc.normalized_certificates(gradients)
    cpu = {
        certificate: {
            name: tensor.detach().to(device="cpu", copy=True)
            for name, tensor in gradient.items()
        }
        for certificate, gradient in units.items()
    }
    return cpu, {name: float(value) for name, value in norms.items()}


def _device_units(units: dict[str, dict[str, Any]], device: Any) -> dict[str, dict[str, Any]]:
    return {
        certificate: {
            name: tensor.to(device=device, non_blocking=False)
            for name, tensor in gradient.items()
        }
        for certificate, gradient in units.items()
    }


def _raw_baseline_from_units(
    units: dict[str, dict[str, Any]], norms: dict[str, float], safety_mix: float
) -> dict[str, Any]:
    return gradient_math.add_gradients(
        gradient_math.scaled_gradient(units["risk"], norms["risk"] * (1.0 - safety_mix)),
        gradient_math.scaled_gradient(units["uniform"], norms["uniform"] * safety_mix),
    )


def _source(mask: int, names: tuple[str, ...], dual: dict[str, Any]) -> dict[str, Any]:
    return {
        "mask": mask,
        "subset": [name for index, name in enumerate(names) if mask & (1 << index)],
        "subset_size": int(mask.bit_count()),
        "dual_margin": float(dual["dual_margin"]),
        "dual_kkt_residual": float(dual["kkt_residual"]),
    }


def _fixed_route_direction(
    descriptor: dict[str, Any],
    units: dict[str, dict[str, Any]],
    budget: float,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = tuple(units)
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units.values()]
        for left in units.values()
    ]
    full_indices = list(range(len(names)))
    full, _ = frsc._subset_direction(units, gram, full_indices, budget, tolerance)
    mask = int(descriptor["source_mask"])
    indices = [index for index in range(len(names)) if mask & (1 << index)]
    endpoint, dual = frsc._subset_direction(units, gram, indices, budget, tolerance)
    fraction = float(descriptor["homotopy_fraction"])
    if fraction == 1.0:
        direction = {name: value.clone() for name, value in endpoint.items()}
    else:
        direction = pfrsc._homotopy_direction(
            full, endpoint, fraction, budget, tolerance
        )
    del full, endpoint
    return direction, _source(mask, names, dual)


def _displacement(
    raw_baseline: dict[str, Any],
    direction: dict[str, Any],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    applied = gradient_math.add_gradients(
        raw_baseline, direction, right_scale=float(config["safety_mix"])
    )
    displacement, audit = base.current_adamw_displacement(
        parameters,
        optimizer,
        applied,
        float(config["max_grad_norm"]),
        float(config["clip_epsilon"]),
    )
    del applied
    return displacement, audit


def _scores(
    units: dict[str, dict[str, Any]], displacement: dict[str, Any], reference: float
) -> dict[str, float]:
    return {
        name: float(gradient_math.gradient_dot(unit, displacement) / reference)
        for name, unit in units.items()
    }


def _proposal_anchor(
    proposal: dict[str, Any],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    units = proposal["units"]
    raw = _raw_baseline_from_units(units, proposal["norms"], float(config["safety_mix"]))
    budget = float(config["auxiliary_gradient_norm_ratio"]) * proposal["norms"]["uniform"]
    selected, anchor_direction, audit = d1r.enumerate_optimizer_anchor_homotopy(
        raw,
        units,
        parameters,
        optimizer,
        budget,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        list(rp.PROTECTED_CERTIFICATES),
        list(rp.TARGET_CERTIFICATES),
        float(protocol["design"]["local_protected_retention_floor"]),
        list(rp.HOMOTOPY_FRACTIONS),
        rp.SOURCE_TOP_K,
        float(config["clip_epsilon"]),
        float(config["dual_numeric_tolerance"]),
        4,
    )
    del selected
    anchor_displacement, map_audit = _displacement(
        raw, anchor_direction, parameters, optimizer, config
    )
    reference_displacement, baseline_audit = base.current_adamw_displacement(
        parameters,
        optimizer,
        raw,
        float(config["max_grad_norm"]),
        float(config["clip_epsilon"]),
    )
    reference = gradient_math.gradient_norm(reference_displacement)
    del reference_displacement
    return {
        "raw_baseline": raw,
        "budget": budget,
        "anchor_direction": anchor_direction,
        "anchor_displacement": anchor_displacement,
        "anchor_record": audit["raw_anchor"],
        "anchor_candidate_id": audit["raw_anchor"]["candidate_id"],
        "anchor_minimum_exact_score": audit["raw_anchor"]["minimum_exact_score"],
        "anchor_search_candidates": audit["feasible_homotopy_candidates"],
        "reference_norm": reference,
        "map_audit": map_audit,
        "baseline_audit": baseline_audit,
    }


def _evaluate_descriptor(
    descriptor: dict[str, Any],
    proposal: dict[str, Any],
    supports: list[dict[str, Any]],
    anchor: dict[str, Any],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    direction, source = _fixed_route_direction(
        descriptor,
        proposal["units"],
        float(anchor["budget"]),
        float(config["dual_numeric_tolerance"]),
    )
    candidate_displacement, map_audit = _displacement(
        anchor["raw_baseline"], direction, parameters, optimizer, config
    )
    local_scores = _scores(
        proposal["units"], candidate_displacement, float(anchor["reference_norm"])
    )
    anchor_scores = anchor["anchor_record"]["exact_scores"]
    retentions = {
        name: float(local_scores[name] / anchor_scores[name])
        for name in rp.PROTECTED_CERTIFICATES
    }
    minimum_retention = min(retentions.values())
    differs = descriptor["route_id"] != anchor["anchor_candidate_id"]
    exercised = (
        minimum_retention + 1e-12
        >= float(protocol["design"]["local_protected_retention_floor"])
        and differs
    )
    support_rows = []
    target_macros = []
    for support in supports:
        anchor_support = _scores(
            support["units"],
            anchor["anchor_displacement"],
            float(anchor["reference_norm"]),
        )
        candidate_support = (
            _scores(
                support["units"],
                candidate_displacement,
                float(anchor["reference_norm"]),
            )
            if exercised
            else dict(anchor_support)
        )
        deltas = {
            name: float(candidate_support[name] - anchor_support[name])
            for name in rp.CERTIFICATES
        }
        target_macro = statistics.fmean(deltas[name] for name in rp.TARGET_CERTIFICATES)
        target_macros.append(target_macro)
        support_rows.append(
            {
                "support_block_id": support["block_id"],
                "exact_delta_scores": deltas,
                "target_macro_delta": target_macro,
            }
        )
    row = {
        **descriptor,
        "proposal_block_id": proposal["block_id"],
        "anchor_candidate_id": anchor["anchor_candidate_id"],
        "anchor_minimum_exact_score": anchor["anchor_minimum_exact_score"],
        "anchor_search_candidates": anchor["anchor_search_candidates"],
        "route_exercised": exercised,
        "route_differs_from_anchor": differs,
        "minimum_local_protected_retention": minimum_retention,
        "local_protected_retentions": retentions,
        "local_exact_scores": local_scores,
        "source_dual_margin": source["dual_margin"],
        "source_dual_kkt_residual": source["dual_kkt_residual"],
        "target_macro_deltas": target_macros,
        "support_rows": support_rows,
        "optimizer_completed_steps": int(map_audit["completed_optimizer_steps"]),
        "candidate_clipping_active": bool(map_audit["clipping_active"]),
    }
    del direction, candidate_displacement
    return row


def _collect_partition(
    label: str,
    blocks: list[dict[str, Any]],
    model: Any,
    parameters: list[tuple[str, Any]],
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    coverage_rows = []
    for index, block in enumerate(blocks, start=1):
        started = time.time()
        gradients, coverage = rp.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(value) for value in block["risk_row_indices"]],
            [str(value) for value in block["safety_sample_ids"]],
            config,
            train_mode=False,
        )
        units, norms = _cpu_units(gradients)
        rows.append({"block_id": block["block_id"], "units": units, "norms": norms})
        coverage_rows.append(
            {
                "block_id": block["block_id"],
                "coverage": coverage,
                "wall_seconds": time.time() - started,
            }
        )
        del gradients
        print(f"BARP-D1X {label} gradients {index}/4", flush=True)
    return rows, coverage_rows


def _evaluate_partition(
    label: str,
    cpu_blocks: list[dict[str, Any]],
    descriptors: list[dict[str, Any]],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    import torch

    device = parameters[0][1].device
    blocks = [
        {
            "block_id": block["block_id"],
            "norms": block["norms"],
            "units": _device_units(block["units"], device),
        }
        for block in cpu_blocks
    ]
    by_route = {descriptor["route_id"]: [] for descriptor in descriptors}
    for index, proposal in enumerate(blocks):
        started = time.time()
        supports = [block for support_index, block in enumerate(blocks) if support_index != index]
        anchor = _proposal_anchor(
            proposal, parameters, optimizer, config, protocol
        )
        for descriptor in descriptors:
            by_route[descriptor["route_id"]].append(
                _evaluate_descriptor(
                    descriptor,
                    proposal,
                    supports,
                    anchor,
                    parameters,
                    optimizer,
                    config,
                    protocol,
                )
            )
        print(
            f"BARP-D1X {label} proposal {index + 1}/4 "
            f"routes={len(descriptors)} anchor={anchor['anchor_candidate_id']} "
            f"wall={time.time() - started:.1f}s",
            flush=True,
        )
        del anchor
        torch.cuda.empty_cache()
    del blocks
    gc.collect()
    torch.cuda.empty_cache()
    return by_route


def _artifact_inventory(root: Path, reproduction: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": relative(path),
            "sha256": etarc_p0.sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != reproduction and "protocol" not in path.parts
    ]


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import set_seed

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_start(path, protocol)
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [(name, value) for name, value in model.named_parameters() if value.requires_grad],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise D1XValidationError("unexpected D1X trainable parameter inventory")
    pair_map, structural = rp.structural_features(tokenizer, targets, config, protocol)
    opened_blocks = protocol["selection"]["discovery_blocks"] + protocol["selection"][
        "qualification_blocks"
    ]
    risk_indices = {
        int(value) for block in opened_blocks for value in block["risk_row_indices"]
    }
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in risk_indices
    }
    optimizer = frozen.make_optimizer(model, config)
    discovery_cpu, discovery_coverage = _collect_partition(
        "discovery",
        protocol["selection"]["discovery_blocks"],
        model,
        parameters,
        risk_features,
        pair_map,
        config,
    )
    registry = route_registry()
    discovery_rows = _evaluate_partition(
        "discovery",
        discovery_cpu,
        registry,
        parameters,
        optimizer,
        config,
        protocol,
    )
    discovery_summaries = [
        summarize_route(
            descriptor,
            discovery_rows[descriptor["route_id"]],
            protocol["gates"],
        )
        for descriptor in registry
    ]
    selected = select_discovery_route(discovery_summaries)
    output_root = resolve_repo_path(protocol["output_root"])
    discovery_path = resolve_repo_path(protocol["outputs"]["discovery"])
    write_json(
        discovery_path,
        {
            "name": "step3_barp_mcd_d1x_discovery_v1",
            "route_family": registry,
            "coverage": discovery_coverage,
            "route_summaries": discovery_summaries,
            "selected_route_id": selected["route_id"] if selected else None,
        },
    )
    freeze_path = resolve_repo_path(protocol["outputs"]["route_freeze"])
    freeze = {
        "name": "step3_barp_mcd_d1x_route_freeze_v1",
        "status": "frozen_before_qualification_gradients",
        "discovery_sha256": etarc_p0.sha256(discovery_path),
        "selected_route": (
            {
                key: selected[key]
                for key in (
                    "route_id",
                    "source_mask",
                    "source_subset",
                    "homotopy_fraction",
                    "target_clusters",
                    "minimum_local_protected_retention",
                )
            }
            if selected
            else None
        ),
        "qualification_gradients_opened_at_freeze": False,
        "optimizer_steps": 0,
    }
    write_json(freeze_path, freeze)

    qualification_summary = None
    checks = None
    qualification_coverage: list[dict[str, Any]] = []
    if selected is not None:
        descriptor = {
            key: selected[key]
            for key in ("route_id", "source_mask", "source_subset", "homotopy_fraction")
        }
        qualification_cpu, qualification_coverage = _collect_partition(
            "qualification",
            protocol["selection"]["qualification_blocks"],
            model,
            parameters,
            risk_features,
            pair_map,
            config,
        )
        qualification_rows = _evaluate_partition(
            "qualification",
            qualification_cpu,
            [descriptor],
            parameters,
            optimizer,
            config,
            protocol,
        )
        qualification_summary = summarize_route(
            descriptor,
            qualification_rows[descriptor["route_id"]],
            protocol["gates"],
        )
        checks = qualification_checks(qualification_summary, protocol["gates"])
        write_json(
            resolve_repo_path(protocol["outputs"]["qualification"]),
            {
                "name": "step3_barp_mcd_d1x_qualification_v1",
                "route_freeze_sha256": etarc_p0.sha256(freeze_path),
                "coverage": qualification_coverage,
                "summary": qualification_summary,
                "checks": checks,
            },
        )

    route = select_route(selected, qualification_summary, checks)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    decision = {
        "name": "step3_barp_mcd_d1x_decision_v1",
        "route": route,
        "selected_route_id": selected["route_id"] if selected else None,
        "discovery_eligible_routes": sum(row["eligible"] for row in discovery_summaries),
        "qualification": qualification_summary,
        "checks": checks,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "sealed_diagnostic_opened": False,
        "core100_opened": False,
        "heldout_opened": False,
        "matched_training_protocol_authorized": route
        == "PASS_BARP_MCD_D1X_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "wall_seconds": time.time() - started,
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    write_json(decision_path, decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    target = qualification_summary["target_clusters"] if qualification_summary else None
    report_path.write_text(
        "# BARP-MCD D1X Cross-Fitted Zero-Step Qualification\n\n"
        f"Decision: `{route}`.\n\n"
        f"Discovery selected: `{decision['selected_route_id']}` from "
        f"`{decision['discovery_eligible_routes']}/56` eligible routes.\n\n"
        + (
            f"Qualification target mean Delta: `{target['mean']:.8f}`; "
            f"Studentized stability lower score: "
            f"`{target['studentized_lower_score']:.8f}`; positive proposal clusters: "
            f"`{target['positive']}/4`.\n\n"
            if target
            else "Qualification gradients remained sealed because discovery found no route.\n\n"
        )
        + "Optimizer steps: `0`; diagnostic, Core100, and heldout opened: `false`.\n",
        encoding="utf-8",
    )
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    reproduction = {
        "name": "step3_barp_mcd_d1x_reproduction_manifest_v1",
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1x.py preflight",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1x.py run",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1x.py verify-run",
        ],
        "route": route,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
        "structural_audit_sha256": etarc_p0.sha256(
            resolve_repo_path(protocol["source"]["structural_audit_path"])
        ),
    }
    write_json(reproduction_path, reproduction)
    reproduction["artifacts"] = _artifact_inventory(output_root, reproduction_path)
    write_json(reproduction_path, reproduction)
    print(json.dumps(decision, sort_keys=True))
    del optimizer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    discovery_path = resolve_repo_path(protocol["outputs"]["discovery"])
    freeze_path = resolve_repo_path(protocol["outputs"]["route_freeze"])
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    discovery = etarc_p0.read_json(discovery_path)
    freeze = etarc_p0.read_json(freeze_path)
    stored = etarc_p0.read_json(decision_path)
    summaries = discovery["route_summaries"]
    selected = select_discovery_route(summaries)
    if discovery["selected_route_id"] != (selected["route_id"] if selected else None):
        raise D1XValidationError("D1X discovery selection does not reproduce")
    if freeze["discovery_sha256"] != etarc_p0.sha256(discovery_path):
        raise D1XValidationError("D1X route freeze does not bind discovery")
    qualification = None
    checks = None
    if selected is not None:
        qualification_path = resolve_repo_path(protocol["outputs"]["qualification"])
        qualification = etarc_p0.read_json(qualification_path)
        if qualification["route_freeze_sha256"] != etarc_p0.sha256(freeze_path):
            raise D1XValidationError("D1X qualification does not bind route freeze")
        checks = qualification_checks(qualification["summary"], protocol["gates"])
        if checks != qualification["checks"]:
            raise D1XValidationError("D1X qualification checks do not reproduce")
        qualification = qualification["summary"]
    route = select_route(selected, qualification, checks)
    if route != stored["route"] or stored.get("optimizer_steps") != 0:
        raise D1XValidationError("D1X decision does not reproduce")
    reproduction = etarc_p0.read_json(reproduction_path)
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise D1XValidationError(f"D1X artifact hash mismatch: {artifact['path']}")
    result = {
        "status": "PASS_BARP_MCD_D1X_VERIFY",
        "route": route,
        "selected_route_id": stored["selected_route_id"],
        "verified_artifacts": len(reproduction["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "sealed_diagnostic_opened": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    if args.command == "preflight":
        preflight(path)
    elif args.command == "run":
        run(path)
    else:
        verify_run(path)


if __name__ == "__main__":
    main()
