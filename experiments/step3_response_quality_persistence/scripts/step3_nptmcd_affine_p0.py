"""Run the prospective NPT-A-MCD affine max-min zero-step experiment."""

from __future__ import annotations

import argparse
import math
from typing import Any

import step3_nptmcd_affine_common as affine
import step3_nptmcd_p0 as base
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_nptamcd_p0/protocol/nptamcd_p0_protocol.json"
)
FIXED_THETA = base.THETA_MAX - 1e-6
METHOD_NAME = "NPT-A-MCD"
_BASE_SELECT_ROUTE = base.select_route


def _target_key(unit_index: int, certificate: str) -> str:
    return f"fit_{unit_index + 1:02d}:{certificate}"


def _protected_audit(
    raw_displacement: dict[str, Any],
    tangent: dict[str, Any],
    protected_constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    tangent_norm = gradient_math.gradient_norm(tangent)
    normalized_dots = [
        gradient_math.gradient_dot(constraint, tangent)
        / gradient_math.gradient_norm(constraint)
        for constraint in protected_constraints
    ]
    raw_dot = gradient_math.gradient_dot(raw_displacement, tangent)
    return {
        "raw_displacement_norm": raw_norm,
        "tangent_norm": tangent_norm,
        "tangent_to_raw_norm_ratio": tangent_norm / raw_norm,
        "raw_tangent_cosine": raw_dot / (raw_norm * tangent_norm),
        "normalized_protected_dots": normalized_dots,
        "maximum_absolute_normalized_protected_dot": max(
            abs(value) for value in normalized_dots
        ),
        "maximum_absolute_protected_cosine": max(abs(value) for value in normalized_dots)
        / tangent_norm,
    }


def _search_record(
    raw_panels: list[dict[str, float]],
    tangent_panels: list[dict[str, float]],
    candidate_panels: list[dict[str, float]],
    solver_audit: dict[str, Any],
    projection_valid: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    protected = tuple(base.PROTECTED_CERTIFICATES)
    targets = tuple(base.TARGET_CERTIFICATES)
    raw_target_floor = min(
        float(panel[name]) for panel in raw_panels for name in targets
    )
    candidate_target_floor = min(
        float(panel[name]) for panel in candidate_panels for name in targets
    )
    required_target_floor = max(
        base.FIT_TARGET_SCORE_FLOOR, raw_target_floor + base.FIT_TARGET_GAIN_FLOOR
    )
    retentions = {
        name: min(
            float(candidate[name]) / float(raw[name])
            for raw, candidate in zip(raw_panels, candidate_panels, strict=True)
        )
        for name in protected
    }
    theta = FIXED_THETA if projection_valid else 0.0
    identity_errors = [
        abs(float(candidate[name]) / float(raw[name]) - math.cos(theta))
        for raw, candidate in zip(raw_panels, candidate_panels, strict=True)
        for name in protected
    ]
    numeric_tolerance = float(protocol["gates"]["numeric_tolerance"])
    minimum_retention = min(retentions.values())
    target_gain = candidate_target_floor - raw_target_floor
    eligible = (
        projection_valid
        and theta >= float(protocol["gates"]["minimum_nonzero_angle"])
        and theta <= base.THETA_MAX + numeric_tolerance
        and minimum_retention >= base.PROTECTED_RETENTION_FLOOR - numeric_tolerance
        and candidate_target_floor >= base.FIT_TARGET_SCORE_FLOOR - numeric_tolerance
        and target_gain >= base.FIT_TARGET_GAIN_FLOOR - numeric_tolerance
        and max(identity_errors) <= float(protocol["gates"]["fit_retention_identity_error_max"])
    )
    return {
        "eligible": eligible,
        "selected_theta": theta,
        "selected_theta_degrees": math.degrees(theta),
        "theta_max": base.THETA_MAX,
        "theta_max_degrees": math.degrees(base.THETA_MAX),
        "design_cosine_floor": base.DESIGN_COSINE_FLOOR,
        "protected_retention_floor": base.PROTECTED_RETENTION_FLOOR,
        "raw_target_score_floor": raw_target_floor,
        "required_target_score_floor": required_target_floor,
        "selected_target_score_floor": candidate_target_floor,
        "selected_target_gain_over_raw": target_gain,
        "protected_retentions": retentions,
        "minimum_protected_retention": minimum_retention,
        "maximum_protected_retention_identity_error": max(identity_errors),
        "candidate_panels": candidate_panels,
        "target_coefficients": solver_audit.get("target_names", []),
        "trigonometric_boundaries": [0.0, theta, base.THETA_MAX],
        "selection_rule": (
            "fixed_theta_max_minus_1e-6_with_fit_target_affine_max_min_"
            "simplex_dual_and_registered_fit_gates"
        ),
        "calibration_scores_used_for_selection": False,
        "diagnostic_scores_used_for_selection": False,
        "fixed_theta": FIXED_THETA,
        "fixed_theta_margin_radians": 1e-6,
        "solver_predicted_minimum_score": solver_audit.get(
            "solver_predicted_minimum_score"
        ),
        "solver_candidate_actual_minimum_score": solver_audit.get(
            "candidate_actual_minimum_score"
        ),
        "solver_candidate_actual_minus_predicted": solver_audit.get(
            "candidate_actual_minus_predicted"
        ),
    }


def build_fit_geometry(
    fit_gradients: list[dict[str, dict[str, Any]]],
    fit_coverages: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Construct the NPT-A-MCD candidate from fit data only."""
    tolerance = float(config["dual_numeric_tolerance"])
    pooled = base.medsc.mean_certificate_gradients(fit_gradients)
    raw_baseline = base._baseline_gradient(pooled, float(config["safety_mix"]))
    budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
        pooled["uniform"]
    )
    raw_direction, target_direction, endpoint_audit = base.path_geometry.endpoint_directions(
        pooled, list(base.TARGET_CERTIFICATES), budget, tolerance
    )
    baseline_displacement, baseline_map = base.frsc.realized_displacement(
        raw_baseline,
        None,
        0.0,
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    raw_displacement, raw_map = base.frsc.realized_displacement(
        raw_baseline,
        raw_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    target_displacement, target_map = base.frsc.realized_displacement(
        raw_baseline,
        target_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )

    protected_constraints = [
        fit[name]
        for fit in fit_gradients
        for name in base.PROTECTED_CERTIFICATES
    ]
    panels = [pooled, *fit_gradients]
    raw_score_panels = [
        base.frsc.displacement_scores(panel, raw_displacement, reference_norm)["scores"]
        for panel in panels
    ]
    tangent = gradient_ops.scaled_gradient(raw_displacement, 0.0)
    proposed_candidate = {name: value.clone() for name, value in raw_displacement.items()}
    solver_audit: dict[str, Any] = {"solver_valid": False}
    projection_valid = False
    projection_error = None
    try:
        target_gradients = [
            (
                _target_key(unit_index, certificate),
                fit_gradients[unit_index][certificate],
            )
            for unit_index in range(base.FIT_UNITS)
            for certificate in base.TARGET_CERTIFICATES
        ]
        raw_target_scores = {
            _target_key(unit_index, certificate): float(
                raw_score_panels[unit_index + 1][certificate]
            )
            for unit_index in range(base.FIT_UNITS)
            for certificate in base.TARGET_CERTIFICATES
        }
        tangent, proposed_candidate, solver_audit = affine.affine_maxmin_target_tangent(
            raw_displacement,
            target_gradients,
            raw_target_scores,
            protected_constraints,
            reference_norm,
            FIXED_THETA,
            float(protocol["design"]["projection_rcond"]),
            int(protocol["design"]["projection_refinement_steps"]),
            float(protocol["design"]["affine_solver_tolerance"]),
        )
        projection_valid = True
        solver_audit["solver_valid"] = True
    except (ValueError, RuntimeError) as exc:
        projection_error = str(exc)
        solver_audit = {"solver_valid": False, "error": projection_error}

    tangent_score_panels = [
        base.frsc.displacement_scores(panel, tangent, reference_norm)["scores"]
        for panel in panels
    ]
    candidate_score_panels = [
        base.frsc.displacement_scores(panel, proposed_candidate, reference_norm)["scores"]
        for panel in panels
    ]
    search = _search_record(
        raw_score_panels[1:],
        tangent_score_panels[1:],
        candidate_score_panels[1:],
        solver_audit,
        projection_valid,
        protocol,
    )
    selected_displacement = (
        proposed_candidate if bool(search["eligible"]) else raw_displacement
    )
    selected_norm = gradient_math.gradient_norm(selected_displacement)
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    protected_audit = (
        _protected_audit(raw_displacement, tangent, protected_constraints)
        if projection_valid
        else {
            "raw_displacement_norm": raw_norm,
            "tangent_norm": 0.0,
            "tangent_to_raw_norm_ratio": 0.0,
            "raw_tangent_cosine": 0.0,
            "normalized_protected_dots": [],
            "maximum_absolute_normalized_protected_dot": math.inf,
            "maximum_absolute_protected_cosine": math.inf,
        }
    )
    projected_norms = solver_audit.get("projected_target_norms", {})
    tangent_audit = {
        **protected_audit,
        "projection_valid": projection_valid,
        "projection_error": projection_error,
        "protected_constraints": len(protected_constraints),
        "retained_target_proposal_fraction": min(projected_norms.values())
        if projected_norms
        else 0.0,
        "target_covector_projection": solver_audit.get("projection"),
        "affine_solver": solver_audit.get("solver"),
        "target_covector_count": len(solver_audit.get("target_names", [])),
    }
    selected_map = {
        "map_kind": "direct_fixed_norm_protected_nullspace_angular_rotation",
        "map_variant": "fit_target_affine_max_min_tangent",
        "raw_endpoint_from_exact_clipped_adam_map": True,
        "target_endpoint_from_exact_clipped_adam_map": True,
        "source_raw_clipping_active": bool(raw_map["clipping_active"]),
        "source_target_clipping_active": bool(target_map["clipping_active"]),
        "selected_theta": float(search["selected_theta"]),
        "fixed_theta": FIXED_THETA,
        "selected_norm": selected_norm,
        "raw_norm": raw_norm,
        "selected_to_raw_norm_ratio": selected_norm / raw_norm,
        "target_endpoint_difference_used": False,
        "fit_target_affine_max_min_used": projection_valid,
    }
    fit_geometry = {
        "name": "step3_nptamcd_p0_fit_geometry_v1",
        "method": METHOD_NAME,
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
            for fit_index in range(base.FIT_UNITS)
            for name in base.PROTECTED_CERTIFICATES
        ],
        "protected_constraint_count": len(protected_constraints),
        "target_proposal_norm": gradient_math.gradient_norm(
            gradient_ops.add_gradients(target_displacement, raw_displacement, right_scale=-1.0)
        ),
        "target_endpoint_difference_used": False,
        "tangent_audit": tangent_audit,
        "projection_valid": projection_valid,
        "projection_error": projection_error,
        "search": search,
        "selected_theta": float(search["selected_theta"]),
        "selected_theta_degrees": math.degrees(float(search["selected_theta"])),
        "theta_max": base.THETA_MAX,
        "pooled_scores": {
            "raw": raw_score_panels[0],
            "tangent": tangent_score_panels[0],
            "selected": base.frsc.displacement_scores(
                pooled, selected_displacement, reference_norm
            )["scores"],
        },
        "fit_score_panels": {
            f"fit_{index:02d}": {
                "raw": raw_score_panels[index],
                "tangent": tangent_score_panels[index],
                "proposed": candidate_score_panels[index],
                "selected": base.frsc.displacement_scores(
                    fit_gradients[index - 1], selected_displacement, reference_norm
                )["scores"],
            }
            for index in range(1, base.FIT_UNITS + 1)
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
        tangent,
        protected_constraints,
    )
    return fit_geometry, fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def select_route(gates: dict[str, bool]) -> str:
    """Keep the base gate ordering while using the affine attempt namespace."""
    return _BASE_SELECT_ROUTE(gates).replace("NPTMCD", "NPTAMCD")


def main() -> None:
    args = parse_args()
    path = base.resolve_repo_path(args.protocol)
    # The old runner supplies the frozen model loading, role evaluation,
    # restoration checks, summaries, and route verifier. Only fit geometry is
    # replaced by the new prospective method.
    base.build_fit_geometry = build_fit_geometry
    base.select_route = select_route
    if args.command == "preflight":
        base.preflight(path)
    elif args.command == "run":
        base.run(path)
    else:
        base.verify_run(path)


if __name__ == "__main__":
    main()
