"""Run seed-7 nullspace-protected target MCD without optimizer steps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_jcrmcd_common as path_geometry
import step3_jcrmcd_p0 as jcr_runner
import step3_medscmcd_common as medsc
import step3_nptmcd_common as npt
import step3_ojmcd_p0 as optimizer_unit
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as model_runner
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_nptmcd_p0/protocol/nptmcd_p0_protocol.json"
)
CERTIFICATES = optimizer_unit.CERTIFICATES
SAFETY_OBJECTIVES = optimizer_unit.SAFETY_OBJECTIVES
ARMS = ("baseline", "raw_mcd", "nptmcd_mcd")
PROTECTED_CERTIFICATES = ("risk", "uniform", "tarc", "continuation")
TARGET_CERTIFICATES = ("etarc", "branch_entropy", "branch_negative_gap")
PROTECTED_RETENTION_FLOOR = 0.95
DESIGN_COSINE_FLOOR = 0.951
THETA_MAX = math.acos(DESIGN_COSINE_FLOOR)
FIT_UNITS = 12
CALIBRATION_UNITS = 20
DIAGNOSTIC_UNITS = 8
SPLIT_SALT = "nptmcd-seed7-split-v1"
CONFORMAL_ALPHA = 0.05
CONFORMAL_ACCEPTANCE_THRESHOLD = 0.0
FIT_TARGET_GAIN_FLOOR = 0.002
FIT_TARGET_SCORE_FLOOR = 0.05
PROJECTION_RCOND = 1e-10
PROJECTION_REFINEMENT_STEPS = 2
MINIMUM_RETAINED_TARGET_PROPOSAL_FRACTION = 0.01
ANGLE_GUARD = 1e-8


class NptmcdP0ValidationError(ValueError):
    """Raised when NPT-MCD P0 violates its prospective protocol."""


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
        raise NptmcdP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expected_roles(
    targets: list[dict[str, Any]], risk_order: list[int]
) -> dict[str, list[dict[str, Any]]]:
    registry = jcr_runner.prompt_category_registry(targets)
    ordered = sorted(
        registry,
        key=lambda sample_id: _stable_digest(f"{SPLIT_SALT}|{sample_id}"),
    )
    if len(ordered) != FIT_UNITS + CALIBRATION_UNITS + DIAGNOSTIC_UNITS:
        raise NptmcdP0ValidationError("prompt role count differs")
    if len(risk_order) < len(ordered):
        raise NptmcdP0ValidationError("risk role permutation is too short")

    boundaries = (
        ("fit", 0, FIT_UNITS),
        ("calibration", FIT_UNITS, FIT_UNITS + CALIBRATION_UNITS),
        (
            "diagnostic",
            FIT_UNITS + CALIBRATION_UNITS,
            FIT_UNITS + CALIBRATION_UNITS + DIAGNOSTIC_UNITS,
        ),
    )
    roles = {}
    for role, start, end in boundaries:
        roles[role] = [
            {
                "unit_id": f"{role}_{offset + 1:02d}",
                "risk_row_index": int(risk_order[index]),
                "sample_id": ordered[index],
                "category": registry[ordered[index]],
            }
            for offset, index in enumerate(range(start, end))
        ]
    return roles


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != "prospective_before_any_NPTMCD_model_forward":
        raise NptmcdP0ValidationError("NPT-MCD protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or etarc_p0.sha256(artifact_path) != artifact["sha256"]
        ):
            raise NptmcdP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("virtual_parameter_evaluations") != 24
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise NptmcdP0ValidationError("protocol violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise NptmcdP0ValidationError("entry commit is not an ancestor of HEAD")

    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_MEDSCMCD_P0_NO_MINIMUM_EFFECTIVE_DOSE":
        raise NptmcdP0ValidationError("registered MED-SC-MCD precursor route differs")
    design = protocol["design"]
    if (
        tuple(design["certificates"]) != CERTIFICATES
        or tuple(design["protected_certificates"]) != PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != TARGET_CERTIFICATES
        or float(design["protected_retention_floor"])
        != PROTECTED_RETENTION_FLOOR
        or float(design["design_cosine_floor"]) != DESIGN_COSINE_FLOOR
        or float(design["theta_max_radians"]) != THETA_MAX
        or int(design["fit_units"]) != FIT_UNITS
        or int(design["calibration_units"]) != CALIBRATION_UNITS
        or int(design["diagnostic_units"]) != DIAGNOSTIC_UNITS
        or float(design["conformal_alpha"]) != CONFORMAL_ALPHA
        or float(design["conformal_acceptance_threshold"])
        != CONFORMAL_ACCEPTANCE_THRESHOLD
        or design.get("conformal_policy")
        != "deploy_fit_candidate_if_threshold_nonpositive_else_raw_mcd"
        or design.get("conditional_on_candidate_acceptance_coverage_claimed")
        is not False
        or float(design["projection_rcond"]) != PROJECTION_RCOND
        or int(design["projection_refinement_steps"])
        != PROJECTION_REFINEMENT_STEPS
        or float(design["minimum_retained_target_proposal_fraction"])
        != MINIMUM_RETAINED_TARGET_PROPOSAL_FRACTION
        or float(design["angle_guard_radians"]) != ANGLE_GUARD
        or int(design["protected_constraints"])
        != FIT_UNITS * len(PROTECTED_CERTIFICATES)
        or design.get("candidate_update_map")
        != "direct_fixed_norm_post_Adam_map_angular_rotation_not_an_ordinary_Adam_output"
        or design.get("ordinary_Adam_candidate_claimed") is not False
        or float(design["fit_target_gain_floor"]) != FIT_TARGET_GAIN_FLOOR
        or float(design["fit_target_score_floor"]) != FIT_TARGET_SCORE_FLOOR
    ):
        raise NptmcdP0ValidationError("NPT-MCD design registry differs")

    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 7
        or config.get("required_gpu_name_substring") != "RTX 4070 Ti"
        or config.get("gradient_measurement_mode") != "eval"
        or config.get("model_optimizer_objects_authorized") is not False
        or config.get("scheduler_objects_authorized") is not False
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("transient_virtual_parameter_evaluations_authorized")
        is not True
        or config.get("training_authorized") is not False
        or float(config["weight_decay"]) != 0.0
        or int(config["gradient_accumulation_steps"]) != 1
    ):
        raise NptmcdP0ValidationError("seed-7 virtual-step config differs")

    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise NptmcdP0ValidationError("risk row count differs")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(protocol["selection"]["risk_order_seed"]))
    risk_order = torch.randperm(len(risk_rows), generator=generator).tolist()
    expected = expected_roles(targets, risk_order)
    for role in ("fit", "calibration", "diagnostic"):
        if protocol["selection"][f"{role}_units"] != expected[role]:
            raise NptmcdP0ValidationError(f"{role} role registry differs")
    all_units = [item for role in expected.values() for item in role]
    if (
        len({item["sample_id"] for item in all_units}) != 40
        or len({item["risk_row_index"] for item in all_units}) != 40
        or [item["risk_row_index"] for item in all_units] != risk_order[:40]
    ):
        raise NptmcdP0ValidationError("fit/calibration/diagnostic roles overlap")
    registry = jcr_runner.prompt_category_registry(targets)
    if dict(sorted(Counter(registry.values()).items())) != protocol["source"][
        "category_prompt_counts"
    ]:
        raise NptmcdP0ValidationError("source category counts differ")
    return protocol, config, targets, risk_rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise NptmcdP0ValidationError("NPT-MCD P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([protocol["config_path"], relative(path)])
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
            raise NptmcdP0ValidationError(f"prospective output already exists: {output}")
    return etarc_p0.git_output("rev-parse", "HEAD")


def structural_features(
    tokenizer: Any,
    targets: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        targets,
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    expected = etarc_p0.read_json(
        resolve_repo_path(protocol["source"]["structural_audit_path"])
    )
    if audit != expected:
        raise NptmcdP0ValidationError("route structure differs from frozen source")
    return {str(pair["sample_id"]): pair for pair in pairs}, audit


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise NptmcdP0ValidationError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name(0)
    if str(config["required_gpu_name_substring"]) not in gpu:
        raise NptmcdP0ValidationError(f"registered RTX 4070 Ti is unavailable: {gpu}")
    tokenizer = tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = structural_features(tokenizer, targets, config, protocol)
    selected = [
        item["sample_id"]
        for role in ("fit_units", "calibration_units", "diagnostic_units")
        for item in protocol["selection"][role]
    ]
    if not all(sample_id in pair_map for sample_id in selected):
        raise NptmcdP0ValidationError("selected safety pair is absent")
    conformal = medsc.split_conformal_accept_or_fallback(
        [-0.01] * CALIBRATION_UNITS,
        CONFORMAL_ALPHA,
        CONFORMAL_ACCEPTANCE_THRESHOLD,
        PROTECTED_RETENTION_FLOOR - 1.0,
    )
    result = {
        "status": "PASS_NPTMCD_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "seed": config["seed"],
        "gradient_measurement_mode": "eval",
        "certificates": list(CERTIFICATES),
        "arms": list(ARMS),
        "fit_units": FIT_UNITS,
        "calibration_units": CALIBRATION_UNITS,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "protected_constraints": FIT_UNITS * len(PROTECTED_CERTIFICATES),
        "theta_max_radians": THETA_MAX,
        "theta_max_degrees": math.degrees(THETA_MAX),
        "conformal_rank": conformal["order_statistic_rank"],
        "conformal_coverage_lower_bound": conformal[
            "finite_sample_marginal_coverage_lower_bound"
        ],
        "virtual_parameter_evaluations": DIAGNOSTIC_UNITS * len(ARMS),
        "gpu": gpu,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _baseline_gradient(
    gradients: dict[str, dict[str, Any]], safety_mix: float
) -> dict[str, Any]:
    return gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(gradients["risk"], 1.0 - safety_mix),
        gradients["uniform"],
        right_scale=safety_mix,
    )


def build_fit_geometry(
    fit_gradients: list[dict[str, dict[str, Any]]],
    fit_coverages: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tolerance = float(config["dual_numeric_tolerance"])
    pooled = medsc.mean_certificate_gradients(fit_gradients)
    raw_baseline = _baseline_gradient(pooled, float(config["safety_mix"]))
    budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
        pooled["uniform"]
    )
    raw_direction, target_direction, endpoint_audit = path_geometry.endpoint_directions(
        pooled, list(TARGET_CERTIFICATES), budget, tolerance
    )
    baseline_displacement, baseline_map = frsc.realized_displacement(
        raw_baseline,
        None,
        0.0,
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    raw_displacement, raw_map = frsc.realized_displacement(
        raw_baseline,
        raw_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    target_displacement, target_map = frsc.realized_displacement(
        raw_baseline,
        target_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    target_proposal = gradient_ops.add_gradients(
        target_displacement, raw_displacement, right_scale=-1.0
    )
    protected_constraints = [
        fit[name]
        for fit in fit_gradients
        for name in PROTECTED_CERTIFICATES
    ]
    projection_valid = True
    projection_error = None
    try:
        tangent, tangent_audit = npt.protected_angular_tangent(
            raw_displacement,
            target_proposal,
            protected_constraints,
            PROJECTION_RCOND,
            PROJECTION_REFINEMENT_STEPS,
            MINIMUM_RETAINED_TARGET_PROPOSAL_FRACTION,
        )
    except ValueError as exc:
        projection_valid = False
        projection_error = str(exc)
        tangent = gradient_ops.scaled_gradient(raw_displacement, 0.0)
        tangent_audit = {
            "projection_valid": False,
            "error": projection_error,
            "protected_constraints": len(protected_constraints),
        }

    panels = [pooled, *fit_gradients]
    raw_score_panels = [
        frsc.displacement_scores(panel, raw_displacement, reference_norm)["scores"]
        for panel in panels
    ]
    tangent_score_panels = [
        frsc.displacement_scores(panel, tangent, reference_norm)["scores"]
        for panel in panels
    ]
    search = npt.select_minimum_effective_angle(
        raw_score_panels[1:],
        tangent_score_panels[1:],
        list(PROTECTED_CERTIFICATES),
        list(TARGET_CERTIFICATES),
        PROTECTED_RETENTION_FLOOR,
        DESIGN_COSINE_FLOOR,
        FIT_TARGET_GAIN_FLOOR,
        FIT_TARGET_SCORE_FLOOR,
        float(protocol["gates"]["numeric_tolerance"]),
        ANGLE_GUARD,
    )
    search["projection_valid"] = projection_valid
    search["projection_error"] = projection_error
    search["calibration_scores_used_for_selection"] = False
    search["diagnostic_scores_used_for_selection"] = False
    selected_theta = float(search["selected_theta"])
    selected_displacement = (
        npt.angular_displacement(raw_displacement, tangent, selected_theta)
        if projection_valid and bool(search["eligible"])
        else {name: value.clone() for name, value in raw_displacement.items()}
    )
    selected_norm = gradient_math.gradient_norm(selected_displacement)
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    selected_map = {
        "map_kind": "direct_fixed_norm_protected_nullspace_angular_rotation",
        "raw_endpoint_from_exact_clipped_adam_map": True,
        "target_proposal_from_exact_clipped_adam_map": True,
        "source_raw_clipping_active": bool(raw_map["clipping_active"]),
        "source_target_clipping_active": bool(target_map["clipping_active"]),
        "selected_theta": selected_theta,
        "selected_norm": selected_norm,
        "raw_norm": raw_norm,
        "selected_to_raw_norm_ratio": selected_norm / raw_norm,
    }
    fit_geometry = {
        "name": "step3_nptmcd_p0_fit_geometry_v1",
        "gradient_measurement_mode": "eval",
        "unit_granularity": "one_prompt_risk_pair",
        "fit_coverages": fit_coverages,
        "pooled_gradient_norms": {
            name: gradient_math.gradient_norm(gradient)
            for name, gradient in pooled.items()
        },
        "raw_baseline_gradient_norm": gradient_math.gradient_norm(raw_baseline),
        "auxiliary_budget_norm": budget,
        "reference_displacement_norm": reference_norm,
        "baseline_optimizer_map": baseline_map,
        "raw_optimizer_map": raw_map,
        "target_optimizer_map": target_map,
        "selected_optimizer_map": selected_map,
        "raw_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(raw_direction) - budget
        )
        / budget,
        "target_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(target_direction) - budget
        )
        / budget,
        "endpoint_audit": endpoint_audit,
        "protected_constraint_labels": [
            f"fit_{fit_index + 1:02d}:{name}"
            for fit_index in range(FIT_UNITS)
            for name in PROTECTED_CERTIFICATES
        ],
        "protected_constraint_count": len(protected_constraints),
        "target_proposal_norm": gradient_math.gradient_norm(target_proposal),
        "tangent_audit": tangent_audit,
        "projection_valid": projection_valid,
        "projection_error": projection_error,
        "search": search,
        "selected_theta": selected_theta,
        "selected_theta_degrees": math.degrees(selected_theta),
        "theta_max": THETA_MAX,
        "pooled_scores": {
            "raw": raw_score_panels[0],
            "tangent": tangent_score_panels[0],
            "selected": frsc.displacement_scores(
                pooled, selected_displacement, reference_norm
            )["scores"],
        },
        "fit_score_panels": {
            f"fit_{index + 1:02d}": {
                "raw": raw_score_panels[index + 1],
                "tangent": tangent_score_panels[index + 1],
                "selected": frsc.displacement_scores(
                    fit_gradients[index], selected_displacement, reference_norm
                )["scores"],
            }
            for index in range(FIT_UNITS)
        },
        "selected_to_raw_norm_relative_error": abs(selected_norm / raw_norm - 1.0),
        "calibration_scores_used_for_selection": False,
        "diagnostic_scores_used_for_selection": False,
        "raw_gradient_tensors_serialized": False,
    }
    fixed = {
        "baseline": baseline_displacement,
        "raw_mcd": raw_displacement,
        "nptmcd_mcd": selected_displacement,
    }
    del (
        pooled,
        raw_baseline,
        raw_direction,
        target_direction,
        target_displacement,
        target_proposal,
        tangent,
        protected_constraints,
    )
    return fit_geometry, fixed


def score_fixed_displacement(
    gradients: dict[str, dict[str, Any]],
    displacement: dict[str, Any],
    parameters: list[tuple[str, Any]],
    reference_norm: float,
) -> dict[str, Any]:
    moved = jcr_runner.device_displacement(displacement, parameters)
    result = frsc.displacement_scores(gradients, moved, reference_norm)
    del moved
    return result


def summarize_results(
    rows: list[dict[str, Any]],
    fit_geometry: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    conformal: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != DIAGNOSTIC_UNITS or len(calibration_rows) != CALIBRATION_UNITS:
        raise NptmcdP0ValidationError("stored role-unit count differs")
    targets = tuple(protocol["design"]["target_certificates"])
    diagnostic: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows]
        target_minima = [
            min(float(item["diagnostic_scores"]["scores"][name]) for name in targets)
            for item in arm_rows
        ]
        diagnostic[arm] = {
            "minimum_worst_score": min(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "mean_worst_score": statistics.fmean(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "minimum_target_score": min(target_minima),
            "mean_target_minimum_score": statistics.fmean(target_minima),
            "minimum_certificate_scores": {
                certificate: min(
                    float(item["diagnostic_scores"]["scores"][certificate])
                    for item in arm_rows
                )
                for certificate in CERTIFICATES
            },
            "objectives": {},
        }
        for certificate in CERTIFICATES:
            objectives = [
                item["finite_step"]["objectives"][certificate]
                for item in arm_rows
            ]
            ratios = [
                float(item["actual_to_predicted_ratio"])
                for item in objectives
                if item["actual_to_predicted_ratio"] is not None
            ]
            diagnostic[arm]["objectives"][certificate] = {
                "mean_predicted_reduction": statistics.fmean(
                    float(item["predicted_first_order_reduction"])
                    for item in objectives
                ),
                "mean_actual_reduction": statistics.fmean(
                    float(item["actual_loss_reduction"]) for item in objectives
                ),
                "minimum_actual_reduction": min(
                    float(item["actual_loss_reduction"]) for item in objectives
                ),
                "positive_actual_rate": sum(
                    float(item["actual_loss_reduction"]) > 0.0
                    for item in objectives
                )
                / len(objectives),
                "mean_actual_to_predicted_ratio": statistics.fmean(ratios)
                if ratios
                else None,
            }

    worst = {
        arm: [
            float(row["arms"][arm]["diagnostic_scores"]["minimum_score"])
            for row in rows
        ]
        for arm in ARMS
    }
    paired: dict[str, Any] = {
        "nptmcd_minus_baseline_worst_score": jcr_runner._paired(
            [
                left - right
                for left, right in zip(
                    worst["nptmcd_mcd"], worst["baseline"], strict=True
                )
            ],
            protocol,
        ),
        "nptmcd_minus_raw_worst_score": jcr_runner._paired(
            [
                left - right
                for left, right in zip(
                    worst["nptmcd_mcd"], worst["raw_mcd"], strict=True
                )
            ],
            protocol,
        ),
        "nptmcd_minus_raw_target_minimum_score": jcr_runner._paired(
            [
                min(
                    float(row["arms"]["nptmcd_mcd"]["diagnostic_scores"]["scores"][name])
                    for name in targets
                )
                - min(
                    float(row["arms"]["raw_mcd"]["diagnostic_scores"]["scores"][name])
                    for name in targets
                )
                for row in rows
            ],
            protocol,
        ),
        "nptmcd_relative_exact_score": {},
        "nptmcd_relative_actual_reduction": {},
    }
    for certificate in CERTIFICATES:
        selected_exact = [
            float(row["arms"]["nptmcd_mcd"]["diagnostic_scores"]["scores"][certificate])
            for row in rows
        ]
        raw_exact = [
            float(row["arms"]["raw_mcd"]["diagnostic_scores"]["scores"][certificate])
            for row in rows
        ]
        selected_actual = [
            float(
                row["arms"]["nptmcd_mcd"]["finite_step"]["objectives"][certificate][
                    "actual_loss_reduction"
                ]
            )
            for row in rows
        ]
        raw_actual = [
            float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][certificate][
                    "actual_loss_reduction"
                ]
            )
            for row in rows
        ]
        paired["nptmcd_relative_exact_score"][certificate] = (
            jcr_runner._relative_paired(selected_exact, raw_exact, protocol)
        )
        paired["nptmcd_relative_actual_reduction"][certificate] = (
            jcr_runner._relative_paired(selected_actual, raw_actual, protocol)
        )

    valid_target_macro = []
    invalid_target_macro = []
    for index, row in enumerate(rows):
        raw_values = [
            float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][name][
                    "actual_loss_reduction"
                ]
            )
            for name in targets
        ]
        if any(value <= 0.0 for value in raw_values):
            invalid_target_macro.append(index)
            continue
        valid_target_macro.append(
            statistics.fmean(
                float(
                    row["arms"]["nptmcd_mcd"]["finite_step"]["objectives"][name][
                        "actual_loss_reduction"
                    ]
                )
                / raw_value
                - 1.0
                for name, raw_value in zip(targets, raw_values, strict=True)
            )
        )
    if invalid_target_macro:
        paired["target_macro_actual_relative_gain"] = {
            "defined": False,
            "invalid_denominator_indices": invalid_target_macro,
            "differences": [],
        }
    else:
        paired["target_macro_actual_relative_gain"] = {
            "defined": True,
            **jcr_runner._paired(valid_target_macro, protocol),
        }

    calibration = {
        "conformal": conformal,
        "raw_protected_scores_positive": all(
            bool(row["raw_protected_scores_positive"]) for row in calibration_rows
        ),
        "minimum_joint_protected_retention": min(
            float(row["minimum_protected_retention"]) for row in calibration_rows
        ),
        "minimum_protected_retention_by_certificate": {
            name: min(
                float(row["protected_retentions"][name])
                for row in calibration_rows
            )
            for name in PROTECTED_CERTIFICATES
        },
        "mean_target_minimum_gain_over_raw": statistics.fmean(
            float(row["candidate_target_minimum"])
            - float(row["raw_target_minimum"])
            for row in calibration_rows
        ),
        "positive_target_minimum_gain_rate": sum(
            float(row["candidate_target_minimum"])
            > float(row["raw_target_minimum"])
            for row in calibration_rows
        )
        / len(calibration_rows),
    }
    search = fit_geometry["search"]
    fit = {
        "projection_valid": bool(fit_geometry["projection_valid"]),
        "selected_theta": float(fit_geometry["selected_theta"]),
        "selected_theta_degrees": float(fit_geometry["selected_theta_degrees"]),
        "theta_max": float(fit_geometry["theta_max"]),
        "trigonometric_boundaries": len(search["trigonometric_boundaries"]),
        "selected_eligible": bool(search["eligible"]),
        "minimum_selected_protected_retention": float(
            search["minimum_protected_retention"]
        ),
        "selected_target_score_floor": float(search["selected_target_score_floor"]),
        "selected_target_gain_over_raw": float(
            search["selected_target_gain_over_raw"]
        ),
        "maximum_protected_retention_identity_error": float(
            search["maximum_protected_retention_identity_error"]
        ),
    }
    return {
        "fit": fit,
        "calibration": calibration,
        "diagnostic": diagnostic,
        "paired": paired,
    }


def _relative_gate(
    item: dict[str, Any], mean_loss: float, ci_loss: float, unit_loss: float
) -> bool:
    return bool(item.get("defined")) and (
        float(item["bootstrap"]["mean"]) >= -mean_loss
        and float(item["bootstrap"]["ci95_lower"]) > -ci_loss
        and min(float(value) for value in item["differences"]) >= -unit_loss
    )


def evaluate_gates(
    rows: list[dict[str, Any]],
    fit_geometry: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    state_restored: bool,
    vram_headroom: float,
    gpu_name: str,
) -> dict[str, bool]:
    gates = protocol["gates"]
    protected = tuple(protocol["design"]["protected_certificates"])
    targets = tuple(protocol["design"]["target_certificates"])
    selected_diag = summary["diagnostic"]["nptmcd_mcd"]
    objectives = list(selected_diag["objectives"].values())
    baseline_pair = summary["paired"]["nptmcd_minus_baseline_worst_score"]
    target_pair = summary["paired"]["nptmcd_minus_raw_target_minimum_score"]
    macro_pair = summary["paired"]["target_macro_actual_relative_gain"]
    exact_relative = summary["paired"]["nptmcd_relative_exact_score"]
    actual_relative = summary["paired"]["nptmcd_relative_actual_reduction"]
    protected_exact_ok = all(
        _relative_gate(
            exact_relative[name],
            float(gates["protected_mean_relative_loss_max"]),
            float(gates["protected_ci_relative_loss_max"]),
            float(gates["protected_single_unit_relative_loss_max"]),
        )
        for name in protected
    )
    protected_actual_ok = all(
        _relative_gate(
            actual_relative[name],
            float(gates["protected_mean_relative_loss_max"]),
            float(gates["protected_ci_relative_loss_max"]),
            float(gates["protected_single_unit_relative_loss_max"]),
        )
        for name in protected
    )
    target_floor_ok = all(
        bool(actual_relative[name].get("defined"))
        and float(actual_relative[name]["bootstrap"]["mean"])
        >= -float(gates["target_objective_mean_relative_loss_max"])
        and min(float(value) for value in actual_relative[name]["differences"])
        >= -float(gates["target_objective_single_unit_relative_loss_max"])
        for name in targets
    )
    fit_coverages = fit_geometry["fit_coverages"]
    conformal = summary["calibration"]["conformal"]
    return {
        "integrity": (
            len(fit_coverages) == FIT_UNITS
            and all(item["mode"] == "eval" for item in fit_coverages)
            and all(item["risk_microbatches"] == 1 for item in fit_coverages)
            and all(item["safety_microbatches"] == 1 for item in fit_coverages)
            and all(item["all_gradients_finite_nonzero"] for item in fit_coverages)
            and len(calibration_rows) == CALIBRATION_UNITS
            and all(row["coverage"]["mode"] == "eval" for row in calibration_rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in calibration_rows)
            and len(rows) == DIAGNOSTIC_UNITS
            and all(row["coverage"]["mode"] == "eval" for row in rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in rows)
            and int(fit_geometry["protected_constraint_count"])
            == FIT_UNITS * len(PROTECTED_CERTIFICATES)
            and len(fit_geometry["protected_constraint_labels"])
            == FIT_UNITS * len(PROTECTED_CERTIFICATES)
            and len(fit_geometry["fit_score_panels"]) == FIT_UNITS
            and bool(fit_geometry["baseline_optimizer_map"]["clipping_active"])
            and bool(fit_geometry["raw_optimizer_map"]["clipping_active"])
            and bool(fit_geometry["target_optimizer_map"]["clipping_active"])
            and fit_geometry["selected_optimizer_map"]["map_kind"]
            == "direct_fixed_norm_protected_nullspace_angular_rotation"
            and float(fit_geometry["raw_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and float(fit_geometry["target_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and float(fit_geometry["selected_to_raw_norm_relative_error"])
            <= float(gates["candidate_norm_relative_tolerance"])
            and all(
                all(
                    arm["state_restored_after_virtual_step"]
                    for arm in row["arms"].values()
                )
                for row in rows
            )
            and state_restored
            and vram_headroom >= float(gates["minimum_vram_headroom_fraction"])
            and "RTX 4070 Ti" in gpu_name
            and fit_geometry["calibration_scores_used_for_selection"] is False
            and fit_geometry["diagnostic_scores_used_for_selection"] is False
            and fit_geometry["raw_gradient_tensors_serialized"] is False
        ),
        "protected_nullspace_geometry": (
            bool(fit_geometry["projection_valid"])
            and float(
                fit_geometry["tangent_audit"]["retained_target_proposal_fraction"]
            )
            >= MINIMUM_RETAINED_TARGET_PROPOSAL_FRACTION
            and abs(float(fit_geometry["tangent_audit"]["raw_tangent_cosine"]))
            <= float(gates["projection_residual_relative_max"])
            and float(
                fit_geometry["tangent_audit"][
                    "maximum_absolute_protected_cosine"
                ]
            )
            <= float(gates["projection_residual_relative_max"])
            and abs(
                float(fit_geometry["tangent_audit"]["tangent_to_raw_norm_ratio"])
                - 1.0
            )
            <= float(gates["candidate_norm_relative_tolerance"])
        ),
        "fit_minimum_effective_angle": (
            bool(summary["fit"]["selected_eligible"])
            and float(summary["fit"]["selected_theta"])
            >= float(gates["minimum_nonzero_angle"])
            and float(summary["fit"]["selected_theta"])
            <= THETA_MAX + float(gates["numeric_tolerance"])
            and float(summary["fit"]["minimum_selected_protected_retention"])
            >= PROTECTED_RETENTION_FLOOR
            and float(summary["fit"]["selected_target_score_floor"])
            >= FIT_TARGET_SCORE_FLOOR
            and float(summary["fit"]["selected_target_gain_over_raw"])
            >= FIT_TARGET_GAIN_FLOOR
            and float(summary["fit"]["maximum_protected_retention_identity_error"])
            <= float(gates["fit_retention_identity_error_max"])
        ),
        "split_conformal_accept_or_raw_policy": (
            bool(summary["calibration"]["raw_protected_scores_positive"])
            and int(conformal["calibration_units"]) == CALIBRATION_UNITS
            and int(conformal["order_statistic_rank"]) == CALIBRATION_UNITS
            and float(conformal["finite_sample_marginal_coverage_lower_bound"])
            >= float(gates["conformal_coverage_lower_bound_min"])
            and float(
                conformal["finite_sample_marginal_policy_protection_lower_bound"]
            )
            >= float(gates["conformal_coverage_lower_bound_min"])
            and float(conformal["threshold"])
            <= float(gates["numeric_tolerance"])
            and bool(conformal["all_calibration_units_protected"])
            and bool(conformal["candidate_accepted"])
            and conformal["policy_action"] == "candidate"
            and float(conformal["fallback_nonconformity"])
            <= CONFORMAL_ACCEPTANCE_THRESHOLD
            and conformal[
                "conditional_on_candidate_acceptance_coverage_claimed"
            ]
            is False
            and fit_geometry["search"]["calibration_scores_used_for_selection"] is False
            and fit_geometry["search"]["diagnostic_scores_used_for_selection"] is False
        ),
        "diagnostic_common_transfer": (
            float(selected_diag["minimum_worst_score"])
            >= float(gates["diagnostic_exact_margin_min"])
            and all(
                float(value) >= float(gates["diagnostic_certificate_score_min"])
                for value in selected_diag["minimum_certificate_scores"].values()
            )
            and float(baseline_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_advantage_over_baseline_mean_min"])
            and float(baseline_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(baseline_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
        ),
        "finite_step_common_descent": all(
            float(item["mean_actual_reduction"])
            > float(gates["mean_actual_loss_reduction_min"])
            and float(item["positive_actual_rate"])
            >= float(gates["actual_reduction_positive_rate_min"])
            and item["mean_actual_to_predicted_ratio"] is not None
            and float(item["mean_actual_to_predicted_ratio"])
            >= float(gates["mean_actual_to_predicted_ratio_min"])
            and float(item["mean_actual_to_predicted_ratio"])
            <= float(gates["mean_actual_to_predicted_ratio_max"])
            for item in objectives
        ),
        "protected_diagnostic_noninferiority": (
            protected_exact_ok and protected_actual_ok
        ),
        "target_diagnostic_advantage": (
            float(target_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_target_gain_mean_min"])
            and float(target_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(target_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and bool(macro_pair.get("defined"))
            and float(macro_pair["bootstrap"]["mean"])
            >= float(gates["target_macro_actual_relative_gain_mean_min"])
            and float(macro_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(macro_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and target_floor_ok
        ),
        "effect_not_weak": (
            float(selected_diag["mean_target_minimum_score"])
            >= float(gates["diagnostic_mean_target_score_min"])
            and bool(macro_pair.get("defined"))
            and float(macro_pair["bootstrap"]["mean"])
            >= float(gates["target_macro_actual_relative_gain_mean_min"])
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"]:
        return "STOP_NPTMCD_P0_INVALID"
    if not gates["protected_nullspace_geometry"]:
        return "STOP_NPTMCD_P0_NULLSPACE_GEOMETRY_INVALID"
    if not gates["fit_minimum_effective_angle"]:
        return "STOP_NPTMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE"
    if not gates["split_conformal_accept_or_raw_policy"]:
        return "STOP_NPTMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED"
    if not gates["diagnostic_common_transfer"]:
        return "STOP_NPTMCD_P0_DIAGNOSTIC_TRANSFER_FAILED"
    if not gates["finite_step_common_descent"]:
        return "STOP_NPTMCD_P0_FINITE_STEP_COMMON_DESCENT_FAILED"
    if not gates["protected_diagnostic_noninferiority"]:
        return "STOP_NPTMCD_P0_PROTECTED_NONINFERIORITY_FAILED"
    if not gates["target_diagnostic_advantage"]:
        return "STOP_NPTMCD_P0_TARGET_ADVANTAGE_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_NPTMCD_P0_EFFECT_TOO_WEAK"
    return "PASS_NPTMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL"


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
                "sha256": etarc_p0.sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(etarc_p0.read_jsonl(path))
                if path.suffix == ".jsonl"
                else None,
            }
        )
    return artifacts


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import set_seed

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise NptmcdP0ValidationError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    initial_state = model_runner._snapshot(parameters)
    pair_map, structural = structural_features(tokenizer, targets, config, protocol)
    roles = {
        role: protocol["selection"][f"{role}_units"]
        for role in ("fit", "calibration", "diagnostic")
    }
    selected_risk = {
        int(item["risk_row_index"])
        for role in roles.values()
        for item in role
    }
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in selected_risk
    }

    fit_gradients = []
    fit_coverages = []
    for unit in roles["fit"]:
        gradients, coverage = model_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            config,
            train_mode=False,
        )
        fit_gradients.append(jcr_runner.cpu_certificate_copy(gradients))
        fit_coverages.append({"unit_id": unit["unit_id"], **coverage})
        if not model_runner._state_equal(parameters, initial_state):
            raise NptmcdP0ValidationError("fit gradients changed parameters")
        print(
            f"NPTMCD {unit['unit_id']} gradients=7 "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    fit_geometry, fixed_displacements = build_fit_geometry(
        fit_gradients, fit_coverages, config, protocol
    )
    fit_geometry["runtime_head"] = runtime_head
    fit_geometry["protocol_sha256"] = etarc_p0.sha256(path)
    del fit_gradients
    gc.collect()
    torch.cuda.empty_cache()

    reference_norm = float(fit_geometry["reference_displacement_norm"])
    calibration_rows = []
    for unit in roles["calibration"]:
        gradients, coverage = model_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            config,
            train_mode=False,
        )
        raw = score_fixed_displacement(
            gradients,
            fixed_displacements["raw_mcd"],
            parameters,
            reference_norm,
        )
        candidate = score_fixed_displacement(
            gradients,
            fixed_displacements["nptmcd_mcd"],
            parameters,
            reference_norm,
        )
        raw_positive = all(
            float(raw["scores"][name]) > 0.0 for name in PROTECTED_CERTIFICATES
        )
        if not raw_positive:
            raise NptmcdP0ValidationError(
                "calibration raw protected score is nonpositive"
            )
        nonconformity, residuals = medsc.joint_protected_nonconformity(
            {name: float(value) for name, value in raw["scores"].items()},
            {name: float(value) for name, value in candidate["scores"].items()},
            list(PROTECTED_CERTIFICATES),
            PROTECTED_RETENTION_FLOOR,
        )
        retentions = {
            name: float(candidate["scores"][name]) / float(raw["scores"][name])
            for name in PROTECTED_CERTIFICATES
        }
        calibration_rows.append(
            {
                "unit_id": unit["unit_id"],
                "risk_row_index": unit["risk_row_index"],
                "safety_sample_id": unit["sample_id"],
                "safety_category": unit["category"],
                "coverage": coverage,
                "raw_scores": raw,
                "candidate_scores": candidate,
                "raw_protected_scores_positive": raw_positive,
                "protected_retentions": retentions,
                "minimum_protected_retention": min(retentions.values()),
                "protected_relative_shortfalls": residuals,
                "joint_nonconformity": nonconformity,
                "raw_target_minimum": min(
                    float(raw["scores"][name]) for name in TARGET_CERTIFICATES
                ),
                "candidate_target_minimum": min(
                    float(candidate["scores"][name]) for name in TARGET_CERTIFICATES
                ),
            }
        )
        if not model_runner._state_equal(parameters, initial_state):
            raise NptmcdP0ValidationError("calibration scoring changed parameters")
        print(
            f"NPTMCD {unit['unit_id']} retention={min(retentions.values()):.6f} "
            f"q={nonconformity:.6f}",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    conformal = medsc.split_conformal_accept_or_fallback(
        [float(row["joint_nonconformity"]) for row in calibration_rows],
        CONFORMAL_ALPHA,
        CONFORMAL_ACCEPTANCE_THRESHOLD,
        PROTECTED_RETENTION_FLOOR - 1.0,
    )
    conformal["candidate_selected_before_calibration"] = True
    conformal["calibration_scores_used_for_selection"] = False
    conformal["diagnostic_scores_used_for_selection"] = False

    map_audits = {
        "baseline": fit_geometry["baseline_optimizer_map"],
        "raw_mcd": fit_geometry["raw_optimizer_map"],
        "nptmcd_mcd": fit_geometry["selected_optimizer_map"],
    }
    rows = []
    for unit in roles["diagnostic"]:
        unit_started = time.time()
        gradients, coverage = model_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            config,
            train_mode=False,
        )
        initial_losses = dict(coverage["loss_means"])
        arms = {
            arm: jcr_runner.evaluate_fixed_arm(
                arm,
                fixed_displacements[arm],
                map_audits[arm],
                gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                unit,
                config,
                reference_norm,
            )
            for arm in ARMS
        }
        raw_scores = arms["raw_mcd"]["diagnostic_scores"]["scores"]
        candidate_scores = arms["nptmcd_mcd"]["diagnostic_scores"]["scores"]
        nonconformity, residuals = medsc.joint_protected_nonconformity(
            {name: float(value) for name, value in raw_scores.items()},
            {name: float(value) for name, value in candidate_scores.items()},
            list(PROTECTED_CERTIFICATES),
            PROTECTED_RETENTION_FLOOR,
        )
        row = {
            "unit_id": unit["unit_id"],
            "risk_row_index": unit["risk_row_index"],
            "safety_sample_id": unit["sample_id"],
            "safety_category": unit["category"],
            "coverage": coverage,
            "joint_nonconformity": nonconformity,
            "protected_relative_shortfalls": residuals,
            "conformal_threshold": conformal["threshold"],
            "covered_by_conformal_threshold": nonconformity
            <= float(conformal["threshold"]),
            "arms": arms,
            "wall_seconds": time.time() - unit_started,
        }
        rows.append(row)
        print(
            f"NPTMCD {unit['unit_id']} q={nonconformity:.6f} target="
            f"{min(float(candidate_scores[name]) for name in TARGET_CERTIFICATES):.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    final_hash = model_common.trainable_state_hash(parameters)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    summary = summarize_results(
        rows, fit_geometry, calibration_rows, conformal, protocol
    )
    gates = evaluate_gates(
        rows,
        fit_geometry,
        calibration_rows,
        summary,
        protocol,
        initial_hash == final_hash,
        headroom,
        gpu_name,
    )
    route = select_route(gates)
    outputs = {
        name: resolve_repo_path(value) for name, value in protocol["outputs"].items()
    }
    diagnostics = {
        "name": "step3_nptmcd_p0_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "summary": summary,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "state_restored": initial_hash == final_hash,
        "fit_units": FIT_UNITS,
        "calibration_units": CALIBRATION_UNITS,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "raw_gradient_tensors_serialized": False,
    }
    environment = {
        "name": "step3_nptmcd_p0_environment_v1",
        "runtime_head": runtime_head,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_nptmcd_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "summary": summary,
        "registered_policy_action": (
            "nptmcd_candidate"
            if bool(fit_geometry["search"]["eligible"])
            and conformal["candidate_accepted"]
            else "raw_mcd"
        ),
        "conditional_on_candidate_acceptance_coverage_claimed": False,
        "proxy_dev_triplet_protocol_authorized": route
        == "PASS_NPTMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL",
        "matched_training_protocol_authorized": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "checkpoint_writes": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    macro = summary["paired"]["target_macro_actual_relative_gain"]
    macro_text = (
        f"{float(macro['bootstrap']['mean']):.6f}"
        if macro.get("defined")
        else "undefined because a raw target reduction was nonpositive"
    )
    retained = fit_geometry["tangent_audit"].get(
        "retained_target_proposal_fraction"
    )
    retained_text = f"{float(retained):.6f}" if retained is not None else "undefined"
    report = f"""# NPT-MCD Seed-7 Zero-Step P0 Report

Route: `{route}`.

Forty-eight unit-level protected fit gradients defined the projection
nullspace. The retained target-proposal norm fraction was `{retained_text}`.
Twelve one-pair eval-mode fit units selected the leftmost effective angular
dose, `theta={summary['fit']['selected_theta']:.8f}` radians. Its minimum fit
protected retention was `{summary['fit']['minimum_selected_protected_retention']}`
and its fit target-floor gain over raw was
`{summary['fit']['selected_target_gain_over_raw']:.6f}`.

The candidate was fixed before all 20 calibration units. The registered
accept-or-raw policy uses the candidate only when the joint split-conformal
threshold is nonpositive and otherwise falls back to raw MCD. Its rank was
`{conformal['order_statistic_rank']}`, its policy-level finite-sample marginal
protection lower bound was
`{conformal['finite_sample_marginal_policy_protection_lower_bound']:.6f}`, and
the observed threshold was `{conformal['threshold']:.6f}`. This guarantee is
for the complete policy over calibration and future draws; no coverage
conditional on candidate acceptance is claimed.

Across eight independent diagnostics, the mean candidate target minimum was
`{summary['diagnostic']['nptmcd_mcd']['mean_target_minimum_score']:.6f}` and
the macro relative finite-loss target gain over raw was `{macro_text}`.

All 24 finite-loss measurements were transient fixed-displacement applications
followed by exact state restoration. No optimizer or scheduler was created,
`optimizer_steps=0`, and no checkpoint, generated candidate, Core100 row, or
heldout row was opened. Even a pass authorizes only a separately frozen
`proxy_dev` triplet-data protocol, never training.
"""

    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_json_frozen(outputs["fit_geometry"], fit_geometry)
    etarc_p0.write_jsonl_frozen(outputs["calibration_rows"], calibration_rows)
    etarc_p0.write_jsonl_frozen(outputs["diagnostic_rows"], rows)
    etarc_p0.write_json_frozen(outputs["diagnostics"], diagnostics)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_nptmcd_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_nptmcd_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_nptmcd_p0.py verify-run"
        ),
        "seed": config["seed"],
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "artifacts": package_artifacts(protocol),
    }
    etarc_p0.write_json_frozen(outputs["run_manifest"], manifest)
    del model, fixed_displacements, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    fit_geometry = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["fit_geometry"])
    )
    calibration_rows = etarc_p0.read_jsonl(
        resolve_repo_path(protocol["outputs"]["calibration_rows"])
    )
    rows = etarc_p0.read_jsonl(resolve_repo_path(protocol["outputs"]["diagnostic_rows"]))
    diagnostics = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["diagnostics"])
    )
    environment = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["environment"])
    )
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["route"]))
    manifest = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["run_manifest"])
    )
    conformal = medsc.split_conformal_accept_or_fallback(
        [float(row["joint_nonconformity"]) for row in calibration_rows],
        CONFORMAL_ALPHA,
        CONFORMAL_ACCEPTANCE_THRESHOLD,
        PROTECTED_RETENTION_FLOOR - 1.0,
    )
    conformal["candidate_selected_before_calibration"] = True
    conformal["calibration_scores_used_for_selection"] = False
    conformal["diagnostic_scores_used_for_selection"] = False
    summary = summarize_results(
        rows, fit_geometry, calibration_rows, conformal, protocol
    )
    gates = evaluate_gates(
        rows,
        fit_geometry,
        calibration_rows,
        summary,
        protocol,
        bool(diagnostics["state_restored"]),
        float(environment["vram_headroom_fraction"]),
        str(environment["gpu"]),
    )
    route = select_route(gates)
    if (
        decision.get("route") != route
        or decision.get("gates") != gates
        or decision.get("summary") != summary
        or diagnostics.get("summary") != summary
        or manifest.get("route") != route
    ):
        raise NptmcdP0ValidationError("stored and recomputed NPT-MCD route differs")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise NptmcdP0ValidationError("NPT-MCD artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
        or decision.get("matched_training_protocol_authorized") is not False
    ):
        raise NptmcdP0ValidationError("stored NPT-MCD boundary differs")
    result = {
        "status": "PASS_NPTMCD_P0_VERIFY",
        "route": route,
        "fit_units": FIT_UNITS,
        "calibration_units": CALIBRATION_UNITS,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "conformal_rank": conformal["order_statistic_rank"],
        "conformal_threshold": conformal["threshold"],
        "virtual_parameter_evaluations": manifest["virtual_parameter_evaluations"],
        "artifacts": len(manifest["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
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
