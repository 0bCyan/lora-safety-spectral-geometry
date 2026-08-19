"""Run the zero-step CDR-OPC M1 transport experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step3_barp_mcd_d1x as d1x
import step3_barp_mcd_training as optimizer_map
import step3_cdro_opc_common as common
import step3_cspr_m0_mechanism as cspr
import step3_foacpo_p0 as base
import step3_foacpo_p0s as p0s
import step3_ramp_common as gradient_math
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m1/protocol/cdro_opc_m1_protocol.json"
)
PROTECTED = ("risk", "uniform", "tarc", "continuation")


class M1ValidationError(ValueError):
    """Raised when the M1 protocol or result violates its frozen boundary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise M1ValidationError(f"path escapes repository: {value}") from exc
    return path


def protected_design(
    targets: Sequence[Mapping[str, Any]],
    risk_count: int,
    seed: int,
    historical: Mapping[str, Mapping[str, Sequence[Any]]],
    calibration_units: int = 20,
) -> dict[str, Any]:
    """Split historical FOACPO blocks from method-specific calibration units."""
    names = ("train", "calibration", "mechanism_dev")
    if set(historical) != set(names) or risk_count < calibration_units + 12:
        raise M1ValidationError("historical protected registry differs")
    historical_safety = [
        str(historical[name]["safety_sample_ids"][position])
        for position in range(4)
        for name in names
    ]
    historical_risk = [
        int(historical[name]["risk_row_indices"][position])
        for position in range(4)
        for name in names
    ]
    if len(set(historical_safety)) != 12 or len(set(historical_risk)) != 12:
        raise M1ValidationError("historical protected units are not unique")

    category_by_id: dict[str, str] = {}
    for row in targets:
        sample_id = str(row["sample_id"])
        category = str(row["category"])
        if sample_id in category_by_id and category_by_id[sample_id] != category:
            raise M1ValidationError("safety sample spans multiple categories")
        category_by_id[sample_id] = category
    remaining: dict[str, list[str]] = {}
    historical_set = set(historical_safety)
    for sample_id, category in category_by_id.items():
        if sample_id not in historical_set:
            remaining.setdefault(category, []).append(sample_id)
    categories = sorted(
        remaining, key=lambda value: base.stable_rank(seed, "m1-category", value)
    )
    for category in categories:
        remaining[category].sort(
            key=lambda value: base.stable_rank(seed, "m1-safety", value)
        )
    calibration_safety = []
    for round_index in range(max(len(values) for values in remaining.values())):
        for category in categories:
            if round_index < len(remaining[category]):
                calibration_safety.append(remaining[category][round_index])
                if len(calibration_safety) == calibration_units:
                    break
        if len(calibration_safety) == calibration_units:
            break
    if len(calibration_safety) != calibration_units:
        raise M1ValidationError("insufficient method-specific calibration safety units")
    calibration_risk = [
        index
        for index in sorted(
            range(risk_count),
            key=lambda value: base.stable_rank(seed, "m1-risk", str(value)),
        )
        if index not in set(historical_risk)
    ][:calibration_units]
    fit_blocks = [
        {
            "block_id": f"fit_{index:02d}",
            "safety_sample_ids": historical_safety[2 * index : 2 * index + 2],
            "risk_row_indices": historical_risk[2 * index : 2 * index + 2],
        }
        for index in range(6)
    ]
    calibration = [
        {
            "unit_id": f"cal_{index:02d}",
            "safety_sample_ids": [safety_id],
            "risk_row_indices": [risk_index],
        }
        for index, (safety_id, risk_index) in enumerate(
            zip(calibration_safety, calibration_risk, strict=True)
        )
    ]
    return {
        "historical_blocks": historical,
        "fit_blocks": fit_blocks,
        "calibration_units": calibration,
        "historical_safety_ids": historical_safety,
        "historical_risk_indices": historical_risk,
        "calibration_safety_disjoint_from_historical": not historical_set.intersection(
            calibration_safety
        ),
        "calibration_risk_disjoint_from_historical": not set(
            historical_risk
        ).intersection(calibration_risk),
    }


def linear_combination(
    maps: Sequence[dict[str, Any]], weights: Sequence[float]
) -> dict[str, Any]:
    if not maps or len(maps) != len(weights):
        raise M1ValidationError("map combination inputs differ")
    result = gradient_math.scaled_gradient(maps[0], float(weights[0]))
    for value, weight in zip(maps[1:], weights[1:], strict=True):
        result = gradient_math.add_gradients(
            result, gradient_math.scaled_gradient(value, float(weight))
        )
    return result


def orthonormal_update_basis(
    proposals: Mapping[str, dict[str, Any]],
    relative_eigen_floor: float,
    maximum_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = list(proposals)
    if (
        len(names) < 2
        or not 0.0 < relative_eigen_floor < 1.0
        or maximum_rank < 1
    ):
        raise M1ValidationError("basis inputs are invalid")
    maps = [proposals[name] for name in names]
    gram = np.asarray(
        [
            [gradient_math.gradient_dot(left, right) for right in maps]
            for left in maps
        ],
        dtype=np.float64,
    )
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    maximum = float(eigenvalues[order[0]])
    selected = [
        int(index)
        for index in order
        if float(eigenvalues[index]) > maximum * relative_eigen_floor
    ][:maximum_rank]
    if not selected:
        raise M1ValidationError("proposal Gram matrix has no retained direction")
    basis = [
        linear_combination(
            maps,
            eigenvectors[:, index] / math.sqrt(float(eigenvalues[index])),
        )
        for index in selected
    ]
    basis_gram = np.asarray(
        [
            [gradient_math.gradient_dot(left, right) for right in basis]
            for left in basis
        ]
    )
    orthonormality_error = float(
        np.max(np.abs(basis_gram - np.eye(len(basis), dtype=np.float64)))
    )
    if orthonormality_error > 2e-4:
        raise M1ValidationError("update basis failed orthonormality audit")
    return basis, {
        "proposal_names": names,
        "proposal_gram": gram.tolist(),
        "proposal_eigenvalues_descending": [
            float(eigenvalues[index]) for index in order
        ],
        "retained_eigen_indices": selected,
        "retained_rank": len(basis),
        "relative_eigen_floor": relative_eigen_floor,
        "maximum_rank": maximum_rank,
        "basis_gram": basis_gram.tolist(),
        "orthonormality_max_error": orthonormality_error,
    }


def coordinates(
    basis: Sequence[dict[str, Any]], value: dict[str, Any], reference_norm: float
) -> np.ndarray:
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise M1ValidationError("coordinate reference norm is invalid")
    return np.asarray(
        [
            gradient_math.gradient_dot(direction, value) / reference_norm
            for direction in basis
        ],
        dtype=np.float64,
    )


def map_from_coordinates(
    basis: Sequence[dict[str, Any]], value: Sequence[float], reference_norm: float
) -> dict[str, Any]:
    weights = reference_norm * np.asarray(value, dtype=np.float64)
    return linear_combination(basis, weights.tolist())


def unit_score_vector(
    unit: dict[str, Any], basis: Sequence[dict[str, Any]]
) -> np.ndarray:
    return np.asarray(
        [gradient_math.gradient_dot(unit, direction) for direction in basis],
        dtype=np.float64,
    )


def map_scores(
    units: Mapping[str, dict[str, Any]],
    value: dict[str, Any],
    reference_norm: float,
) -> dict[str, float]:
    return {
        name: float(gradient_math.gradient_dot(units[name], value) / reference_norm)
        for name in PROTECTED
    }


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = base.read_json(path)
    if protocol.get("status") != "prospective_before_any_CDRO_OPC_M1_model_forward":
        raise M1ValidationError("M1 protocol status differs")
    for name, item in protocol["frozen_inputs"].items():
        item_path = resolve_repo_path(item["path"])
        if not item_path.is_file() or base.sha256(item_path) != item["sha256"]:
            raise M1ValidationError(f"frozen input mismatch: {name}")
    if any(
        (
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise M1ValidationError("M1 violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise M1ValidationError("M1 entry commit is not an ancestor")
    prior = base.read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p0s_diagnostics"]["path"])
    )
    if prior.get("route") != "STOP_FOACPO_P0S_MECHANISM_TRANSFER":
        raise M1ValidationError("M1 precursor route differs")
    config = base.read_json(resolve_repo_path(protocol["config_path"]))
    targets = base.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    return protocol, config, targets, risk_rows


def frozen_design(
    protocol: dict[str, Any],
    targets: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = int(protocol["selection"]["seed"])
    historical = base.select_protected_blocks(targets, len(risk_rows), seed)
    design = protected_design(targets, len(risk_rows), seed, historical)
    expected = protocol["selection"]["protected_design_sha256"]
    actual = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual != expected:
        raise M1ValidationError("protected split differs from frozen digest")
    return design


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, risk_rows = context(path)
    if not torch.cuda.is_available():
        raise M1ValidationError("CUDA is unavailable")
    design = frozen_design(protocol, targets, risk_rows)
    pools = base.load_pairs(protocol)
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    shapes = [
        cspr.token_shape(tokenizer, row["prompt"], row[role])
        for pool in pools.values()
        for rows in pool.values()
        for row in rows
        for role in ("chosen_response", "rejected_response")
    ]
    result = {
        "status": "PASS_CDRO_OPC_M1_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "fit_blocks": len(design["fit_blocks"]),
        "calibration_units": len(design["calibration_units"]),
        "calibration_disjoint_from_historical": bool(
            design["calibration_safety_disjoint_from_historical"]
            and design["calibration_risk_disjoint_from_historical"]
        ),
        "protected_certificates": list(PROTECTED),
        "maximum_pair_tokens": max(item["total_tokens"] for item in shapes),
        "conformal_rank": 20,
        "conformal_marginal_coverage": 20 / 21,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def behavior_panel(
    model: Any,
    parameters: list[tuple[str, Any]],
    tokenizer: Any,
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    candidate: dict[str, Any],
    fallback: dict[str, Any],
    beta: float,
    max_length: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("calibration", "mechanism_dev"):
        result[split] = {}
        for pool in ("quality", "safety"):
            features = base.pair_features(tokenizer, pools[pool][split], max_length)
            gradient, coverage = base.structured_preference_gradient(
                model, parameters, features, beta
            )
            candidate_values = base.sign_summary(
                base.per_pair_scores(model, parameters, features, candidate, beta)
            )
            fallback_values = base.sign_summary(
                base.per_pair_scores(model, parameters, features, fallback, beta)
            )
            result[split][pool] = {
                "candidate_aggregate_cosine": base.cosine_score(gradient, candidate),
                "fallback_aggregate_cosine": base.cosine_score(gradient, fallback),
                "candidate": candidate_values,
                "fallback": fallback_values,
                "coverage": coverage,
            }
            del features, gradient
    return result


def route_from_gates(gates: Mapping[str, bool]) -> str:
    if not gates["structural"] or not gates["state_restored"] or not gates["zero_step"]:
        return "STOP_CDRO_OPC_M1_INVALID"
    if not gates["basis"] or not gates["solver"] or not gates["nontrivial"]:
        return "STOP_CDRO_OPC_M1_GEOMETRY"
    if not gates["conformal_transport"]:
        return "STOP_CDRO_OPC_M1_TRANSPORT"
    if not gates["behavior"]:
        return "STOP_CDRO_OPC_M1_BEHAVIOR"
    return "PASS_CDRO_OPC_M1_EXPLORATORY_TRANSPORT"


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, config, targets, risk_rows = context(path)
    runtime_head = base.require_clean_start(path, protocol)
    design = frozen_design(protocol, targets, risk_rows)
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = cspr.load_model(protocol)
    parameters = cspr.trainable_parameters(model)
    optimizer = frozen.make_optimizer(model, config)
    initial_hash = cspr.trainable_state_hash(parameters)
    pools = base.load_pairs(protocol)
    beta = float(protocol["design"]["beta"])
    max_length = int(config["max_length"])

    historical = design["historical_blocks"]
    anchor_units = {}
    anchor_records = {}
    anchor_maps = {}
    for split in ("train", "calibration"):
        units, norms, coverage = base.protected_gradients(
            model,
            parameters,
            tokenizer,
            targets,
            risk_rows,
            historical[split],
            config,
            protocol,
        )
        anchor = d1x._proposal_anchor(
            {"block_id": split, "units": units, "norms": norms},
            parameters,
            optimizer,
            config,
            protocol,
        )
        anchor_units[split] = units
        anchor_maps[split] = anchor["anchor_displacement"]
        anchor_records[split] = {
            "candidate_id": anchor["anchor_candidate_id"],
            "coverage": coverage,
            "optimizer_completed_steps": anchor["map_audit"]["completed_optimizer_steps"],
        }
        del anchor
    left_scores = {
        f"{split}:{name}": gradient_math.gradient_dot(
            anchor_units[split][name], anchor_maps["train"]
        )
        for split in ("train", "calibration")
        for name in PROTECTED
    }
    right_scores = {
        f"{split}:{name}": gradient_math.gradient_dot(
            anchor_units[split][name], anchor_maps["calibration"]
        )
        for split in ("train", "calibration")
        for name in PROTECTED
    }
    robust_solution = p0s.maximin_segment_blend(left_scores, right_scores)
    fallback_map = base.blend_maps(
        anchor_maps["train"],
        anchor_maps["calibration"],
        float(robust_solution["weight"]),
    )
    del anchor_units

    train_features = {
        pool: base.pair_features(tokenizer, pools[pool]["train_candidate"], max_length)
        for pool in ("quality", "safety")
    }
    train_gradients = {}
    preference_coverage = {}
    for pool in ("quality", "safety"):
        train_gradients[pool], preference_coverage[pool] = (
            base.structured_preference_gradient(
                model, parameters, train_features[pool], beta
            )
        )
    train_units = {
        pool: base.unit_gradient(train_gradients[pool])
        for pool in ("quality", "safety")
    }
    behavior_maps = {}
    behavior_audits = {}
    for alpha_value in protocol["design"]["alpha_registry"]:
        alpha = float(alpha_value)
        gradient = gradient_math.add_gradients(
            gradient_math.scaled_gradient(train_units["quality"], alpha),
            gradient_math.scaled_gradient(train_units["safety"], 1.0 - alpha),
        )
        behavior_maps[alpha], behavior_audits[alpha] = (
            optimizer_map.current_adamw_displacement(
                parameters,
                optimizer,
                gradient,
                float(config["max_grad_norm"]),
                float(config["clip_epsilon"]),
            )
        )
        del gradient

    proposals = {
        "anchor_train": anchor_maps["train"],
        "anchor_calibration": anchor_maps["calibration"],
        **{f"behavior_alpha_{alpha:.2f}": value for alpha, value in behavior_maps.items()},
    }
    basis, basis_audit = orthonormal_update_basis(
        proposals,
        float(protocol["design"]["relative_eigen_floor"]),
        int(protocol["design"]["maximum_basis_rank"]),
    )
    reference_norm = gradient_math.gradient_norm(fallback_map)
    fallback_coordinates = coordinates(basis, fallback_map, reference_norm)
    behavior_coordinates = coordinates(
        basis,
        behavior_maps[float(protocol["design"]["behavior_alpha"])],
        reference_norm,
    )
    fallback_reconstruction = map_from_coordinates(
        basis, fallback_coordinates, reference_norm
    )
    reconstruction_difference = gradient_math.add_gradients(
        fallback_reconstruction,
        gradient_math.scaled_gradient(fallback_map, -1.0),
    )
    reconstruction_error = (
        gradient_math.gradient_norm(reconstruction_difference) / reference_norm
    )
    if reconstruction_error > 2e-4:
        raise M1ValidationError("fallback reconstruction failed")
    quality_vector = unit_score_vector(train_units["quality"], basis)
    safety_vector = unit_score_vector(train_units["safety"], basis)
    del train_gradients, train_units, proposals, fallback_reconstruction

    fit_vectors: dict[str, list[list[float]]] = {name: [] for name in PROTECTED}
    fit_rows = []
    score_floor = float(protocol["design"]["score_scale_floor"])
    for block in design["fit_blocks"]:
        units, _, coverage = base.protected_gradients(
            model,
            parameters,
            tokenizer,
            targets,
            risk_rows,
            block,
            config,
            protocol,
        )
        row = {"block_id": block["block_id"], "coverage": coverage, "scores": {}}
        for name in PROTECTED:
            vector = unit_score_vector(units[name], basis)
            fallback_score = float(vector @ fallback_coordinates)
            scale = max(abs(fallback_score), score_floor)
            normalized = vector / scale
            fit_vectors[name].append(normalized.tolist())
            row["scores"][name] = {
                "fallback": fallback_score,
                "scale": scale,
                "normalized_vector": normalized.tolist(),
            }
        fit_rows.append(row)
        del units

    constraints = []
    moment_audits = {}
    for name in PROTECTED:
        moments = common.fit_ellipsoidal_moments(
            fit_vectors[name], ridge=float(protocol["design"]["covariance_ridge"])
        )
        moment_audits[name] = moments
        constraints.append(
            {
                "name": name,
                "mean": moments["mean"],
                "factor": moments["factor"],
                "center": fallback_coordinates.tolist(),
                "uncertainty_radius": float(
                    protocol["design"]["uncertainty_radius"]
                ),
                "floor": -float(protocol["design"]["allowed_relative_drop"]),
            }
        )
    selected_coordinates, solver_audit = common.solve_cdro_opc(
        behavior_target=behavior_coordinates,
        metric=np.eye(len(basis), dtype=np.float64),
        quality_score=quality_vector,
        safety_score=safety_vector,
        protected_constraints=constraints,
        fallback=fallback_coordinates,
        trust_radius=float(protocol["design"]["trust_radius"]),
        utility_weight=float(protocol["design"]["utility_weight"]),
        tau_regularization=float(protocol["design"]["tau_regularization"]),
        feasibility_tolerance=float(protocol["gates"]["feasibility_tolerance"]),
        kkt_tolerance=float(protocol["gates"]["kkt_tolerance"]),
    )
    candidate_digest = hashlib.sha256(
        json.dumps(selected_coordinates.tolist(), separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    candidate_map = map_from_coordinates(basis, selected_coordinates, reference_norm)
    increment = gradient_math.add_gradients(
        candidate_map, gradient_math.scaled_gradient(fallback_map, -1.0)
    )
    normalized_increment = gradient_math.gradient_norm(increment) / reference_norm

    prior = base.read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p0s_diagnostics"]["path"])
    )
    selected_prior = prior["selected"]
    if float(selected_prior["alpha"]) != float(protocol["design"]["behavior_alpha"]):
        raise M1ValidationError("FOACPO ablation alpha differs")
    foacpo_map = base.blend_maps(
        fallback_map,
        behavior_maps[float(selected_prior["alpha"])],
        float(selected_prior["lambda"]),
    )

    candidate_calibration_scores = []
    fallback_calibration_scores = []
    foacpo_calibration_scores = []
    calibration_coverage = []
    for unit in design["calibration_units"]:
        units, _, coverage = base.protected_gradients(
            model,
            parameters,
            tokenizer,
            targets,
            risk_rows,
            unit,
            config,
            protocol,
        )
        fallback_calibration_scores.append(
            map_scores(units, fallback_map, reference_norm)
        )
        candidate_calibration_scores.append(
            map_scores(units, candidate_map, reference_norm)
        )
        foacpo_calibration_scores.append(
            map_scores(units, foacpo_map, reference_norm)
        )
        calibration_coverage.append({"unit_id": unit["unit_id"], **coverage})
        del units
    allowed_drop = float(protocol["design"]["allowed_relative_drop"])
    alpha = float(protocol["design"]["conformal_alpha"])
    candidate_policy = common.conformal_standardized_change_policy(
        fallback_calibration_scores,
        candidate_calibration_scores,
        PROTECTED,
        allowed_drop,
        score_floor,
        alpha,
    )
    foacpo_policy = common.conformal_standardized_change_policy(
        fallback_calibration_scores,
        foacpo_calibration_scores,
        PROTECTED,
        allowed_drop,
        score_floor,
        alpha,
    )
    behavior = behavior_panel(
        model,
        parameters,
        tokenizer,
        pools,
        candidate_map,
        fallback_map,
        beta,
        max_length,
    )

    final_hash = cspr.trainable_state_hash(parameters)
    calibration_behavior = behavior["calibration"]
    mechanism_behavior = behavior["mechanism_dev"]
    behavior_gate = (
        all(
            calibration_behavior[pool]["candidate_aggregate_cosine"]
            > calibration_behavior[pool]["fallback_aggregate_cosine"]
            for pool in ("quality", "safety")
        )
        and calibration_behavior["quality"]["candidate"]["positive"] >= 3
        and calibration_behavior["safety"]["candidate"]["positive"] >= 6
        and all(
            mechanism_behavior[pool]["candidate_aggregate_cosine"] > 0.0
            for pool in ("quality", "safety")
        )
        and mechanism_behavior["quality"]["candidate"]["positive"] >= 4
        and mechanism_behavior["safety"]["candidate"]["positive"] >= 6
    )
    all_coverage = [row["coverage"] for row in fit_rows] + calibration_coverage
    structural = (
        all(item["all_gradients_finite_nonzero"] for item in all_coverage)
        and all(
            item["finite_nonzero"] and not item["missing_gradient_tensors"]
            for item in preference_coverage.values()
        )
    )
    zero_step = (
        all(
            int(item["optimizer_completed_steps"]) == 0
            for item in anchor_records.values()
        )
        and all(
            int(item["completed_optimizer_steps"]) == 0
            for item in behavior_audits.values()
        )
    )
    gates = {
        "structural": structural,
        "basis": basis_audit["orthonormality_max_error"] <= 2e-4
        and reconstruction_error <= 2e-4,
        "solver": bool(solver_audit["optimization_valid"]),
        "nontrivial": normalized_increment
        >= float(protocol["gates"]["minimum_normalized_increment"]),
        "conformal_transport": bool(candidate_policy["candidate_accepted"]),
        "behavior": behavior_gate,
        "state_restored": final_hash == initial_hash,
        "zero_step": zero_step,
    }
    route = route_from_gates(gates)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak = int(torch.cuda.max_memory_reserved(0))
    diagnostics = {
        "name": "step3_cdro_opc_m1_diagnostics_v1",
        "runtime_head": runtime_head,
        "protected_design": design,
        "anchor_records": anchor_records,
        "robust_fallback_solution": robust_solution,
        "basis_audit": basis_audit,
        "reference_norm": reference_norm,
        "fallback_coordinates": fallback_coordinates.tolist(),
        "behavior_coordinates": behavior_coordinates.tolist(),
        "fallback_reconstruction_relative_error": reconstruction_error,
        "fit_rows": fit_rows,
        "ellipsoidal_moments": moment_audits,
        "solver_audit": solver_audit,
        "candidate_coordinate_sha256_before_calibration": candidate_digest,
        "candidate_fixed_before_calibration": True,
        "normalized_increment": normalized_increment,
        "candidate_calibration_scores": candidate_calibration_scores,
        "fallback_calibration_scores": fallback_calibration_scores,
        "foacpo_calibration_scores": foacpo_calibration_scores,
        "candidate_conformal_policy": candidate_policy,
        "foacpo_conformal_policy": foacpo_policy,
        "calibration_coverage": calibration_coverage,
        "behavior": behavior,
        "gates": gates,
        "route": route,
        "method_status": "exploratory_method_specific_prospective_replay",
        "fresh_collection_claimed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "peak_reserved_bytes": peak,
        "vram_headroom_fraction": 1.0 - peak / total_memory,
        "wall_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_cdro_opc_m1_decision_v1",
        "route": route,
        "gates": gates,
        "basis_rank": len(basis),
        "candidate_coordinate_sha256": candidate_digest,
        "normalized_increment": normalized_increment,
        "candidate_conformal_threshold": candidate_policy["threshold"],
        "candidate_conformal_action": candidate_policy["policy_action"],
        "foacpo_conformal_threshold": foacpo_policy["threshold"],
        "foacpo_conformal_action": foacpo_policy["policy_action"],
        "finite_sample_marginal_policy_protection_lower_bound": candidate_policy[
            "finite_sample_marginal_policy_protection_lower_bound"
        ],
        "claim_scope": "exploratory_zero_step_geometry_and_transport_only",
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "freeze finite-radius virtual-step remainder study without training"
            if route == "PASS_CDRO_OPC_M1_EXPLORATORY_TRANSPORT"
            else "stop CDR-OPC training and analyze the failed gate"
        ),
    }
    base.write_json(resolve_repo_path(protocol["outputs"]["diagnostics"]), diagnostics)
    base.write_json(resolve_repo_path(protocol["outputs"]["decision"]), decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# CDR-OPC M1 Zero-Step Report\n\n"
        f"Route: `{route}`. Basis rank: `{len(basis)}`. "
        f"Normalized increment: `{normalized_increment}`.\n\n"
        f"CDR-OPC conformal threshold/action: `{candidate_policy['threshold']}` / "
        f"`{candidate_policy['policy_action']}`. FOACPO-S ablation: "
        f"`{foacpo_policy['threshold']}` / `{foacpo_policy['policy_action']}`.\n\n"
        "This is an exploratory method-specific prospective replay, not fresh-data "
        "confirmation. No optimizer step or training was authorized; Core100 and "
        "heldout remained closed.\n",
        encoding="utf-8",
    )
    base.write_json(
        resolve_repo_path(protocol["outputs"]["environment"]),
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "runtime_head": runtime_head,
        },
    )
    del model, optimizer, candidate_map, fallback_map, foacpo_map, basis
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(decision, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    decision = base.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    diagnostics = base.read_json(resolve_repo_path(protocol["outputs"]["diagnostics"]))
    if (
        decision.get("route") != diagnostics.get("route")
        or decision.get("optimizer_steps") != 0
        or decision.get("training_authorized") is not False
        or diagnostics.get("optimizer_steps") != 0
        or diagnostics.get("training_authorized") is not False
        or diagnostics.get("core100_opened") is not False
        or diagnostics.get("heldout_opened") is not False
        or diagnostics.get("candidate_fixed_before_calibration") is not True
        or diagnostics.get("fresh_collection_claimed") is not False
    ):
        raise M1ValidationError("M1 result differs or violates its claim boundary")
    policy = diagnostics["candidate_conformal_policy"]
    if (
        int(policy["calibration_units"]) != 20
        or int(policy["order_statistic_rank"]) != 20
        or policy["conditional_on_candidate_acceptance_coverage_claimed"] is not False
    ):
        raise M1ValidationError("M1 conformal certificate differs")
    result = {
        "status": "PASS_CDRO_OPC_M1_VERIFY_RUN",
        "route": decision["route"],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    {"preflight": preflight, "run": run, "verify-run": verify_run}[args.command](path)


if __name__ == "__main__":
    main()
