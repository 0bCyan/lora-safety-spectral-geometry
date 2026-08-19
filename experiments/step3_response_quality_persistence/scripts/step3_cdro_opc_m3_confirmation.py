"""Run the fresh, zero-step CDR-OPC M3 conformal-dose confirmation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step3_barp_mcd_d1x as d1x
import step3_barp_mcd_training as optimizer_map
import step3_cdro_opc_common as cdro_math
import step3_cdro_opc_dose_common as dose_math
import step3_cdro_opc_m1 as m1
import step3_cspr_m0_mechanism as cspr
import step3_foacpo_p0 as base
import step3_foacpo_p0s as p0s
import step3_frsc_common as frsc
import step3_ramp_common as gradient_math
import step3_rpfrsc_p0 as rp
import step3_tarc_p0 as tarc
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/protocol/"
    "cdro_opc_m3_confirmation_protocol.json"
)
PROTECTED = ("risk", "uniform", "tarc", "continuation")
SPLITS = ("calibration", "sealed_audit")
POOLS = ("quality", "safety")


class M3ConfirmationError(ValueError):
    """Raised when the M3 protocol, evidence, or claim boundary differs."""


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
        raise M3ConfirmationError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def generic_comparator(author_mode: str) -> str:
    values = {
        "specific_redirect": "I can't help with that.",
        "risk_mitigation": "I won't assist with that.",
    }
    try:
        return values[author_mode]
    except KeyError as exc:
        raise M3ConfirmationError("unknown author mode") from exc


def build_unit_registry(
    selected_targets: Sequence[Mapping[str, Any]],
    primary_candidates: Sequence[Mapping[str, Any]],
    cohort: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(selected_targets) != 80 or len(primary_candidates) != 120:
        raise M3ConfirmationError("M3 candidate denominators differ")
    candidate_modes = {
        str(row["opaque_candidate_id"]): str(row["author_mode"])
        for row in primary_candidates
    }
    risk_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(risk_rows):
        sample_id = str(row["sample_id"])
        if sample_id in risk_by_id:
            raise M3ConfirmationError("risk source sample ids are not unique")
        risk_by_id[sample_id] = (index, row)
    cohort_by_prompt = {
        str(row["opaque_prompt_id"]): {
            "rank": int(row["rank"]),
            "split": str(row["split"]),
        }
        for row in cohort
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_targets:
        grouped[str(row["sample_id"])].append(row)
    if len(grouped) != 40 or len(cohort_by_prompt) != 40:
        raise M3ConfirmationError("M3 prompt denominator differs")

    units = []
    seen_prompts = set()
    for sample_id, rows in grouped.items():
        if len(rows) != 2 or sample_id not in risk_by_id:
            raise M3ConfirmationError("M3 unit is incomplete")
        rows = sorted(rows, key=lambda row: str(row["target_id"]))
        prompt_ids = {str(row["opaque_prompt_id"]) for row in rows}
        splits = {str(row["split"]) for row in rows}
        prompts = {str(row["prompt"]) for row in rows}
        if len(prompt_ids) != 1 or len(splits) != 1 or len(prompts) != 1:
            raise M3ConfirmationError("M3 paired target metadata differs")
        prompt_id = prompt_ids.pop()
        split = splits.pop()
        prompt = prompts.pop()
        if prompt_id not in cohort_by_prompt or cohort_by_prompt[prompt_id]["split"] != split:
            raise M3ConfirmationError("M3 cohort split differs")
        risk_index, risk_row = risk_by_id[sample_id]
        if str(risk_row["prompt"]) != prompt:
            raise M3ConfirmationError("M3 target does not match its risk prompt")
        modes = {
            candidate_modes.get(str(row["target_id"])): str(row["target_id"])
            for row in rows
        }
        if set(modes) != {"specific_redirect", "risk_mitigation"}:
            raise M3ConfirmationError("M3 unit does not contain both frozen modes")
        rank = cohort_by_prompt[prompt_id]["rank"]
        units.append(
            {
                "unit_id": f"m3_{'cal' if split == 'calibration' else 'audit'}_{rank:02d}",
                "rank": rank,
                "split": split,
                "sample_id": sample_id,
                "opaque_prompt_id": prompt_id,
                "risk_row_index": risk_index,
                "target_ids": [str(row["target_id"]) for row in rows],
                "mode_target_ids": dict(sorted(modes.items())),
            }
        )
        seen_prompts.add(prompt_id)
    units.sort(key=lambda row: int(row["rank"]))
    if seen_prompts != set(cohort_by_prompt):
        raise M3ConfirmationError("M3 selected prompts differ from the cohort")
    counts = {split: sum(row["split"] == split for row in units) for split in SPLITS}
    if counts != {"calibration": 20, "sealed_audit": 20}:
        raise M3ConfirmationError("M3 split denominators differ")
    return units


def registry_public_projection(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in (
                "unit_id",
                "rank",
                "split",
                "sample_id",
                "opaque_prompt_id",
                "risk_row_index",
                "target_ids",
                "mode_target_ids",
            )
        }
        for row in units
    ]


def subset_targets(
    rows: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    allowed = {
        target_id
        for unit in units
        if unit["split"] == split
        for target_id in unit["target_ids"]
    }
    result = [dict(row) for row in rows if str(row["target_id"]) in allowed]
    if len(result) != 40:
        raise M3ConfirmationError(f"{split} target denominator differs")
    return result


def behavior_rows(
    selected_targets: Sequence[Mapping[str, Any]],
    primary_candidates: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    target_by_id = {str(row["target_id"]): row for row in selected_targets}
    mode_by_id = {
        str(row["opaque_candidate_id"]): str(row["author_mode"])
        for row in primary_candidates
    }
    result = {pool: [] for pool in POOLS}
    for unit in units:
        if unit["split"] != split:
            continue
        harmful = str(risk_rows[int(unit["risk_row_index"])]["response"])
        for target_id in unit["target_ids"]:
            target = target_by_id[str(target_id)]
            common = {
                "prompt": str(target["prompt"]),
                "chosen_response": str(target["response"]),
            }
            result["safety"].append(
                {
                    "pair_id": f"{unit['unit_id']}:{target_id}:safety",
                    **common,
                    "rejected_response": harmful,
                }
            )
            result["quality"].append(
                {
                    "pair_id": f"{unit['unit_id']}:{target_id}:quality",
                    **common,
                    "rejected_response": generic_comparator(mode_by_id[str(target_id)]),
                }
            )
    if any(len(rows) != 40 for rows in result.values()):
        raise M3ConfirmationError(f"{split} behavior denominator differs")
    return result


def context(
    path: Path,
) -> tuple[Any, ...]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_locked_before_M3_model_forward":
        raise M3ConfirmationError("M3 confirmation protocol status differs")
    for name, item in protocol["frozen_inputs"].items():
        item_path = resolve_repo_path(item["path"])
        if not item_path.is_file() or sha256(item_path) != item["sha256"]:
            raise M3ConfirmationError(f"frozen input mismatch: {name}")
    if any(
        (
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise M3ConfirmationError("M3 confirmation violates its zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise M3ConfirmationError("M3 entry commit is not an ancestor")

    m1_path = resolve_repo_path(protocol["frozen_inputs"]["m1_protocol"]["path"])
    m1_protocol, config, old_targets, old_risk_rows = m1.context(m1_path)
    selected_targets = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["selected_targets"]["path"])
    )
    primary_candidates = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["primary_candidates"]["path"])
    )
    cohort = read_jsonl(resolve_repo_path(protocol["frozen_inputs"]["cohort"]["path"]))
    risk_rows = read_jsonl(resolve_repo_path(protocol["frozen_inputs"]["risk_source"]["path"]))
    units = build_unit_registry(selected_targets, primary_candidates, cohort, risk_rows)
    if canonical_sha256(registry_public_projection(units)) != protocol["selection"]["unit_registry_sha256"]:
        raise M3ConfirmationError("M3 unit registry differs from the freeze")
    return (
        protocol,
        m1_protocol,
        config,
        old_targets,
        old_risk_rows,
        selected_targets,
        primary_candidates,
        risk_rows,
        units,
    )


def require_clean_start(path: Path, protocol: Mapping[str, Any]) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise M3ConfirmationError("M3 confirmation requires a clean worktree")
    tracked = [item["path"] for item in protocol["frozen_inputs"].values()]
    tracked.extend((protocol["config_path"], path.relative_to(REPO).as_posix()))
    for value in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", value],
            cwd=REPO,
            capture_output=True,
            check=True,
        )
    output_root = resolve_repo_path(protocol["output_root"])
    if output_root.exists():
        extras = [item for item in output_root.iterdir() if item.name != "protocol"]
        if extras:
            raise M3ConfirmationError("M3 confirmation outputs already exist")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def preflight(path: Path) -> dict[str, Any]:
    import torch

    (
        protocol,
        _,
        config,
        _,
        _,
        selected_targets,
        primary_candidates,
        risk_rows,
        units,
    ) = context(path)
    if not torch.cuda.is_available():
        raise M3ConfirmationError("CUDA is unavailable")
    tokenizer = tarc.load_tokenizer(str(config["model_name_or_path"]))
    structural = {}
    behavior_counts = {}
    for split in SPLITS:
        _, structural[split] = tarc.build_pair_features(
            tokenizer,
            subset_targets(selected_targets, units, split),
            int(config["max_length"]),
            int(config["route_width_limit"]),
        )
        if structural[split] != protocol["selection"]["structural_audits"][split]:
            raise M3ConfirmationError(f"{split} structural audit differs")
        behavior_counts[split] = {
            pool: len(rows)
            for pool, rows in behavior_rows(
                selected_targets, primary_candidates, units, risk_rows, split
            ).items()
        }
    result = {
        "status": "PASS_CDRO_OPC_M3_CONFIRMATION_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "calibration_units": 20,
        "sealed_audit_units": 20,
        "targets": 80,
        "behavior_pairs": behavior_counts,
        "protected_certificates": list(PROTECTED),
        "alpha": protocol["design"]["alpha"],
        "finite_sample_marginal_protection_lower_bound": 20 / 21,
        "useful_dose_floor": protocol["gates"]["useful_dose_floor"],
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def reconstruct_fixed_direction(
    m1_protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    old_targets: list[dict[str, Any]],
    old_risk_rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    design = m1.frozen_design(dict(m1_protocol), old_targets, old_risk_rows)
    historical = design["historical_blocks"]
    anchor_units = {}
    anchor_maps = {}
    anchor_records = {}
    for split in ("train", "calibration"):
        units, norms, coverage = base.protected_gradients(
            model,
            parameters,
            tokenizer,
            old_targets,
            old_risk_rows,
            historical[split],
            dict(config),
            dict(m1_protocol),
        )
        anchor = d1x._proposal_anchor(
            {"block_id": split, "units": units, "norms": norms},
            parameters,
            optimizer,
            dict(config),
            dict(m1_protocol),
        )
        anchor_units[split] = units
        anchor_maps[split] = anchor["anchor_displacement"]
        anchor_records[split] = {
            "coverage": coverage,
            "optimizer_completed_steps": anchor["map_audit"]["completed_optimizer_steps"],
        }
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

    pools = base.load_pairs(dict(m1_protocol))
    max_length = int(config["max_length"])
    beta = float(m1_protocol["design"]["beta"])
    train_features = {
        pool: base.pair_features(tokenizer, pools[pool]["train_candidate"], max_length)
        for pool in POOLS
    }
    train_gradients = {}
    preference_coverage = {}
    for pool in POOLS:
        train_gradients[pool], preference_coverage[pool] = base.structured_preference_gradient(
            model, parameters, train_features[pool], beta
        )
    train_units = {pool: base.unit_gradient(train_gradients[pool]) for pool in POOLS}
    behavior_maps = {}
    behavior_audits = {}
    for alpha_value in m1_protocol["design"]["alpha_registry"]:
        alpha = float(alpha_value)
        gradient = gradient_math.add_gradients(
            gradient_math.scaled_gradient(train_units["quality"], alpha),
            gradient_math.scaled_gradient(train_units["safety"], 1.0 - alpha),
        )
        behavior_maps[alpha], behavior_audits[alpha] = optimizer_map.current_adamw_displacement(
            parameters,
            optimizer,
            gradient,
            float(config["max_grad_norm"]),
            float(config["clip_epsilon"]),
        )
        del gradient
    proposals = {
        "anchor_train": anchor_maps["train"],
        "anchor_calibration": anchor_maps["calibration"],
        **{f"behavior_alpha_{alpha:.2f}": value for alpha, value in behavior_maps.items()},
    }
    basis, basis_audit = m1.orthonormal_update_basis(
        proposals,
        float(m1_protocol["design"]["relative_eigen_floor"]),
        int(m1_protocol["design"]["maximum_basis_rank"]),
    )
    reference_norm = gradient_math.gradient_norm(fallback_map)
    fallback_coordinates = m1.coordinates(basis, fallback_map, reference_norm)
    behavior_coordinates = m1.coordinates(
        basis,
        behavior_maps[float(m1_protocol["design"]["behavior_alpha"])],
        reference_norm,
    )
    quality_vector = m1.unit_score_vector(train_units["quality"], basis)
    safety_vector = m1.unit_score_vector(train_units["safety"], basis)
    del train_gradients, train_units, proposals

    fit_vectors: dict[str, list[list[float]]] = {name: [] for name in PROTECTED}
    fit_coverage = []
    score_floor = float(m1_protocol["design"]["score_scale_floor"])
    for block in design["fit_blocks"]:
        units, _, coverage = base.protected_gradients(
            model,
            parameters,
            tokenizer,
            old_targets,
            old_risk_rows,
            block,
            dict(config),
            dict(m1_protocol),
        )
        fit_coverage.append(coverage)
        for name in PROTECTED:
            vector = m1.unit_score_vector(units[name], basis)
            fallback_score = float(vector @ fallback_coordinates)
            fit_vectors[name].append(
                (vector / max(abs(fallback_score), score_floor)).tolist()
            )
        del units
    constraints = []
    moment_audits = {}
    for name in PROTECTED:
        moments = cdro_math.fit_ellipsoidal_moments(
            fit_vectors[name], ridge=float(m1_protocol["design"]["covariance_ridge"])
        )
        moment_audits[name] = moments
        constraints.append(
            {
                "name": name,
                "mean": moments["mean"],
                "factor": moments["factor"],
                "center": fallback_coordinates.tolist(),
                "uncertainty_radius": float(m1_protocol["design"]["uncertainty_radius"]),
                "floor": -float(m1_protocol["design"]["allowed_relative_drop"]),
            }
        )
    candidate_coordinates, solver_audit = cdro_math.solve_cdro_opc(
        behavior_target=behavior_coordinates,
        metric=np.eye(len(basis), dtype=np.float64),
        quality_score=quality_vector,
        safety_score=safety_vector,
        protected_constraints=constraints,
        fallback=fallback_coordinates,
        trust_radius=float(m1_protocol["design"]["trust_radius"]),
        utility_weight=float(m1_protocol["design"]["utility_weight"]),
        tau_regularization=float(m1_protocol["design"]["tau_regularization"]),
        feasibility_tolerance=float(m1_protocol["gates"]["feasibility_tolerance"]),
        kkt_tolerance=float(m1_protocol["gates"]["kkt_tolerance"]),
    )
    candidate_map = m1.map_from_coordinates(basis, candidate_coordinates, reference_norm)
    candidate_digest = canonical_sha256(candidate_coordinates.tolist())
    expected_fallback = np.asarray(expected["fallback_coordinates"], dtype=np.float64)
    expected_candidate = np.asarray(expected["candidate_coordinates"], dtype=np.float64)
    replay = {
        "fallback_max_abs_error": float(np.max(np.abs(fallback_coordinates - expected_fallback))),
        "candidate_max_abs_error": float(np.max(np.abs(candidate_coordinates - expected_candidate))),
        "candidate_coordinate_sha256": candidate_digest,
        "expected_candidate_coordinate_sha256": expected["candidate_coordinate_sha256"],
        "basis_rank": len(basis),
        "basis_audit": basis_audit,
        "solver_audit": solver_audit,
        "robust_fallback_solution": robust_solution,
        "anchor_records": anchor_records,
        "fit_coverage": fit_coverage,
        "preference_coverage": preference_coverage,
        "behavior_map_audits": behavior_audits,
        "ellipsoidal_moments": moment_audits,
    }
    tolerance = float(expected["coordinate_tolerance"])
    replay["fixed_direction_valid"] = bool(
        replay["fallback_max_abs_error"] <= tolerance
        and replay["candidate_max_abs_error"] <= tolerance
        and candidate_digest == expected["candidate_coordinate_sha256"]
        and solver_audit["optimization_valid"]
        and basis_audit["orthonormality_max_error"] <= 2e-4
    )
    return fallback_map, candidate_map, reference_norm, replay


def protected_score_panel(
    model: Any,
    parameters: list[tuple[str, Any]],
    tokenizer: Any,
    selected_targets: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split: str,
    fallback_map: dict[str, Any],
    endpoint_map: dict[str, Any],
    reference_norm: float,
    expected_structural: Mapping[str, Any],
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, Any]], dict[str, Any]]:
    pairs, structural = tarc.build_pair_features(
        tokenizer,
        subset_targets(selected_targets, units, split),
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    if structural != expected_structural:
        raise M3ConfirmationError(f"{split} structural audit differs during run")
    pair_map = {str(pair["sample_id"]): pair for pair in pairs}
    split_units = [unit for unit in units if unit["split"] == split]
    risk_features = {
        int(unit["risk_row_index"]): training.response_feature(
            tokenizer,
            str(risk_rows[int(unit["risk_row_index"])]["prompt"]),
            str(risk_rows[int(unit["risk_row_index"])]["response"]),
            int(config["max_length"]),
        )
        for unit in split_units
    }
    fallback_scores = []
    endpoint_scores = []
    coverage_rows = []
    for unit in split_units:
        gradients, coverage = rp.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            dict(config),
            train_mode=False,
        )
        certificates, norms = frsc.normalized_certificates(gradients)
        fallback_scores.append(m1.map_scores(certificates, fallback_map, reference_norm))
        endpoint_scores.append(m1.map_scores(certificates, endpoint_map, reference_norm))
        coverage_rows.append(
            {
                "unit_id": unit["unit_id"],
                "sample_id": unit["sample_id"],
                "gradient_norms": norms,
                **coverage,
            }
        )
        del gradients, certificates
    return fallback_scores, endpoint_scores, coverage_rows, structural


def score_at_dose(
    fallback: Mapping[str, float], endpoint: Mapping[str, float], dose: float
) -> dict[str, float]:
    return {
        name: float(fallback[name]) + dose * (float(endpoint[name]) - float(fallback[name]))
        for name in PROTECTED
    }


def panel_statistics(
    fallback_scores: Sequence[Mapping[str, float]],
    endpoint_scores: Sequence[Mapping[str, float]],
    unit_ids: Sequence[str],
    dose: float,
    allowed_drop: float,
    score_floor: float,
    tolerance: float,
) -> dict[str, Any]:
    if len(fallback_scores) != len(endpoint_scores) or len(unit_ids) != len(fallback_scores):
        raise M3ConfirmationError("protected score panel denominators differ")
    rows = []
    for unit_id, fallback, endpoint in zip(
        unit_ids, fallback_scores, endpoint_scores, strict=True
    ):
        selected = score_at_dose(fallback, endpoint, dose)
        deterioration = {
            name: (float(fallback[name]) - selected[name])
            / max(abs(float(fallback[name])), score_floor)
            for name in PROTECTED
        }
        residual = {name: deterioration[name] - allowed_drop for name in PROTECTED}
        rows.append(
            {
                "unit_id": unit_id,
                "fallback_scores": dict(fallback),
                "endpoint_scores": dict(endpoint),
                "selected_scores": selected,
                "standardized_deteriorations": deterioration,
                "residuals": residual,
                "covered": all(value <= tolerance for value in residual.values()),
            }
        )
    aggregate = {
        name: statistics.fmean(row["standardized_deteriorations"][name] for row in rows)
        for name in PROTECTED
    }
    maximum_residual = max(value for row in rows for value in row["residuals"].values())
    return {
        "rows": rows,
        "covered_units": sum(bool(row["covered"]) for row in rows),
        "total_units": len(rows),
        "maximum_residual": maximum_residual,
        "aggregate_standardized_deteriorations": aggregate,
        "aggregate_protected": all(value <= allowed_drop + tolerance for value in aggregate.values()),
    }


def behavior_panel(
    model: Any,
    parameters: list[tuple[str, Any]],
    tokenizer: Any,
    rows: Mapping[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
    beta: float,
    fallback_map: dict[str, Any],
    endpoint_map: dict[str, Any],
    selected_map: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    for pool in POOLS:
        features = base.pair_features(tokenizer, rows[pool], int(config["max_length"]))
        gradient, coverage = base.structured_preference_gradient(
            model, parameters, features, beta
        )
        fallback = base.cosine_score(gradient, fallback_map)
        endpoint = base.cosine_score(gradient, endpoint_map)
        selected = base.cosine_score(gradient, selected_map)
        result[pool] = {
            "pairs": len(rows[pool]),
            "fallback_aggregate_cosine": fallback,
            "endpoint_aggregate_cosine": endpoint,
            "selected_aggregate_cosine": selected,
            "selected_minus_fallback": selected - fallback,
            "coverage": coverage,
        }
        del features, gradient
    return result


def route_from_gates(gates: Mapping[str, bool]) -> str:
    if not gates["state_restored"] or not gates["zero_step"] or not gates["structural"]:
        return "STOP_CDRO_OPC_M3_INVALID"
    if not gates["fixed_direction"] or not gates["solver_kkt"]:
        return "STOP_CDRO_OPC_M3_DIRECTION_REPLAY"
    if not gates["calibration_arithmetic"] or not gates["useful_dose"]:
        return "STOP_CDRO_OPC_M3_DOSE"
    if not gates["sealed_unit_coverage"] or not gates["sealed_aggregate_protection"]:
        return "STOP_CDRO_OPC_M3_AUDIT_PROTECTION"
    if not gates["behavior"]:
        return "STOP_CDRO_OPC_M3_BEHAVIOR"
    return "PASS_CDRO_OPC_M3_CONFIRMATION"


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    (
        protocol,
        m1_protocol,
        config,
        old_targets,
        old_risk_rows,
        selected_targets,
        primary_candidates,
        risk_rows,
        units,
    ) = context(path)
    runtime_head = require_clean_start(path, protocol)
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = cspr.load_model(m1_protocol)
    parameters = cspr.trainable_parameters(model)
    optimizer = frozen.make_optimizer(model, config)
    initial_hash = cspr.trainable_state_hash(parameters)

    fallback_map, endpoint_map, reference_norm, direction_replay = reconstruct_fixed_direction(
        m1_protocol,
        config,
        old_targets,
        old_risk_rows,
        model,
        tokenizer,
        parameters,
        optimizer,
        protocol["fixed_direction"],
    )
    if not direction_replay["fixed_direction_valid"]:
        raise M3ConfirmationError("frozen M1 direction did not replay exactly")

    design = protocol["design"]
    calibration_units = [unit for unit in units if unit["split"] == "calibration"]
    calibration_fallback, calibration_endpoint, calibration_coverage, calibration_structural = protected_score_panel(
        model,
        parameters,
        tokenizer,
        selected_targets,
        units,
        risk_rows,
        config,
        "calibration",
        fallback_map,
        endpoint_map,
        reference_norm,
        protocol["selection"]["structural_audits"]["calibration"],
    )
    policy = dose_math.conformal_safe_dose(
        calibration_fallback,
        calibration_endpoint,
        PROTECTED,
        float(design["allowed_relative_drop"]),
        float(design["score_scale_floor"]),
        float(design["alpha"]),
        float(design["maximum_dose"]),
    )
    selected_dose = float(policy["selected_safe_dose"])
    calibration_stats = panel_statistics(
        calibration_fallback,
        calibration_endpoint,
        [str(unit["unit_id"]) for unit in calibration_units],
        selected_dose,
        float(design["allowed_relative_drop"]),
        float(design["score_scale_floor"]),
        float(protocol["gates"]["arithmetic_tolerance"]),
    )
    # The sealed split is not scored until the calibration dose is fixed above.
    sealed_units = [unit for unit in units if unit["split"] == "sealed_audit"]
    sealed_fallback, sealed_endpoint, sealed_coverage, sealed_structural = protected_score_panel(
        model,
        parameters,
        tokenizer,
        selected_targets,
        units,
        risk_rows,
        config,
        "sealed_audit",
        fallback_map,
        endpoint_map,
        reference_norm,
        protocol["selection"]["structural_audits"]["sealed_audit"],
    )
    sealed_stats = panel_statistics(
        sealed_fallback,
        sealed_endpoint,
        [str(unit["unit_id"]) for unit in sealed_units],
        selected_dose,
        float(design["allowed_relative_drop"]),
        float(design["score_scale_floor"]),
        float(protocol["gates"]["arithmetic_tolerance"]),
    )

    selected_map = base.blend_maps(fallback_map, endpoint_map, selected_dose)
    beta = float(m1_protocol["design"]["beta"])
    behavior = {
        split: behavior_panel(
            model,
            parameters,
            tokenizer,
            behavior_rows(selected_targets, primary_candidates, units, risk_rows, split),
            config,
            beta,
            fallback_map,
            endpoint_map,
            selected_map,
        )
        for split in SPLITS
    }
    final_hash = cspr.trainable_state_hash(parameters)
    all_coverage = calibration_coverage + sealed_coverage
    structural = all(
        bool(item["all_gradients_finite_nonzero"])
        and all(math.isfinite(float(value)) and float(value) > 0.0 for value in item["gradient_norms"].values())
        for item in all_coverage
    ) and all(
        bool(behavior[split][pool]["coverage"]["finite_nonzero"])
        and not behavior[split][pool]["coverage"]["missing_gradient_tensors"]
        for split in SPLITS
        for pool in POOLS
    )
    zero_step = all(
        int(item["optimizer_completed_steps"]) == 0
        for item in direction_replay["anchor_records"].values()
    ) and all(
        int(item["completed_optimizer_steps"]) == 0
        for item in direction_replay["behavior_map_audits"].values()
    )
    solver = direction_replay["solver_audit"]
    gates = {
        "fixed_direction": bool(direction_replay["fixed_direction_valid"]),
        "solver_kkt": bool(
            solver["optimization_valid"]
            and float(solver["primal_violation"]) <= float(m1_protocol["gates"]["feasibility_tolerance"])
            and float(solver["normalized_stationarity_residual"]) <= float(m1_protocol["gates"]["kkt_tolerance"])
        ),
        "calibration_arithmetic": calibration_stats["maximum_residual"]
        <= float(protocol["gates"]["arithmetic_tolerance"]),
        "useful_dose": selected_dose >= float(protocol["gates"]["useful_dose_floor"]),
        "sealed_unit_coverage": sealed_stats["covered_units"]
        >= int(protocol["gates"]["sealed_audit_required_covered_units"]),
        "sealed_aggregate_protection": bool(sealed_stats["aggregate_protected"]),
        "behavior": all(
            behavior[split][pool]["selected_minus_fallback"]
            > float(protocol["gates"]["minimum_behavior_improvement"])
            for split in SPLITS
            for pool in POOLS
        ),
        "structural": structural,
        "state_restored": final_hash == initial_hash,
        "zero_step": zero_step,
    }
    route = route_from_gates(gates)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak = int(torch.cuda.max_memory_reserved(0))
    diagnostics = {
        "name": "step3_cdro_opc_m3_confirmation_diagnostics_v1",
        "runtime_head": runtime_head,
        "route": route,
        "unit_registry_sha256": canonical_sha256(registry_public_projection(units)),
        "fixed_direction_replay": direction_replay,
        "reference_norm": reference_norm,
        "calibration": {
            "fallback_scores": calibration_fallback,
            "endpoint_scores": calibration_endpoint,
            "coverage": calibration_coverage,
            "structural_audit": calibration_structural,
            "policy": policy,
            "statistics": calibration_stats,
        },
        "sealed_audit": {
            "opened_after_selected_dose": True,
            "selected_dose_before_open": selected_dose,
            "fallback_scores": sealed_fallback,
            "endpoint_scores": sealed_endpoint,
            "coverage": sealed_coverage,
            "structural_audit": sealed_structural,
            "statistics": sealed_stats,
        },
        "behavior": behavior,
        "gates": gates,
        "claim_scope": "fresh_zero_step_first_order_conformal_dose_confirmation",
        "finite_parameter_step_claimed": False,
        "generated_response_claimed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "peak_reserved_bytes": peak,
        "vram_headroom_fraction": 1.0 - peak / total_memory,
        "wall_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_cdro_opc_m3_confirmation_decision_v1",
        "route": route,
        "selected_dose": selected_dose,
        "finite_sample_marginal_protection_lower_bound": policy[
            "finite_sample_marginal_protection_lower_bound"
        ],
        "sealed_audit_covered_units": sealed_stats["covered_units"],
        "sealed_audit_units": sealed_stats["total_units"],
        "sealed_aggregate_standardized_deteriorations": sealed_stats[
            "aggregate_standardized_deteriorations"
        ],
        "behavior_selected_minus_fallback": {
            split: {
                pool: behavior[split][pool]["selected_minus_fallback"] for pool in POOLS
            }
            for split in SPLITS
        },
        "gates": gates,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "interpret zero-step evidence without authorizing training"
            if route == "PASS_CDRO_OPC_M3_CONFIRMATION"
            else "stop and report the failed frozen gate"
        ),
    }
    write_json(resolve_repo_path(protocol["outputs"]["diagnostics"]), diagnostics)
    write_json(resolve_repo_path(protocol["outputs"]["decision"]), decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# CDR-OPC M3 Zero-Step Confirmation\n\n"
        f"Route: `{route}`. Selected dose: `{selected_dose}`. Sealed protected "
        f"units: `{sealed_stats['covered_units']}/20`.\n\n"
        "The direction was replayed from the frozen M1 construction. Twenty fresh "
        "calibration prompts selected the conformal dose before the twenty sealed "
        "audit prompts were scored. No optimizer step, training, generation, Core100, "
        "or heldout evaluation was authorized.\n",
        encoding="utf-8",
    )
    write_json(
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
    del model, optimizer, fallback_map, endpoint_map, selected_map
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(decision, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, *_ = context(path)
    diagnostics = read_json(resolve_repo_path(protocol["outputs"]["diagnostics"]))
    decision = read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if any(
        (
            diagnostics.get("optimizer_steps") != 0,
            diagnostics.get("training_authorized") is not False,
            diagnostics.get("core100_opened") is not False,
            diagnostics.get("heldout_opened") is not False,
            diagnostics.get("finite_parameter_step_claimed") is not False,
            diagnostics.get("sealed_audit", {}).get("opened_after_selected_dose") is not True,
            decision.get("optimizer_steps") != 0,
            decision.get("training_authorized") is not False,
        )
    ):
        raise M3ConfirmationError("M3 result violates its claim boundary")
    calibration = diagnostics["calibration"]
    policy = dose_math.conformal_safe_dose(
        calibration["fallback_scores"],
        calibration["endpoint_scores"],
        PROTECTED,
        float(protocol["design"]["allowed_relative_drop"]),
        float(protocol["design"]["score_scale_floor"]),
        float(protocol["design"]["alpha"]),
        float(protocol["design"]["maximum_dose"]),
    )
    selected = float(policy["selected_safe_dose"])
    if abs(selected - float(decision["selected_dose"])) > 1e-15:
        raise M3ConfirmationError("M3 selected dose does not reproduce")
    if abs(selected - float(diagnostics["sealed_audit"]["selected_dose_before_open"])) > 1e-15:
        raise M3ConfirmationError("M3 sealed split used a different dose")
    for split in SPLITS:
        panel = diagnostics[split]
        recomputed = panel_statistics(
            panel["fallback_scores"],
            panel["endpoint_scores"],
            [row["unit_id"] for row in panel["statistics"]["rows"]],
            selected,
            float(protocol["design"]["allowed_relative_drop"]),
            float(protocol["design"]["score_scale_floor"]),
            float(protocol["gates"]["arithmetic_tolerance"]),
        )
        if canonical_sha256(recomputed) != canonical_sha256(panel["statistics"]):
            raise M3ConfirmationError(f"{split} arithmetic does not reproduce")
    if decision["route"] != diagnostics["route"] or decision["route"] != route_from_gates(diagnostics["gates"]):
        raise M3ConfirmationError("M3 route does not reproduce")
    result = {
        "status": "PASS_CDRO_OPC_M3_CONFIRMATION_VERIFY_RUN",
        "route": decision["route"],
        "selected_dose": selected,
        "sealed_audit_covered_units": decision["sealed_audit_covered_units"],
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
