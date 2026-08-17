"""Run the prospective global margin-aware MCD experiment."""

from __future__ import annotations

import argparse
import math
from typing import Any

import step3_nptmcd_affine_common as affine
import step3_nptmcd_affine_p0 as affine_runner
import step3_nptmcd_p0 as base
import step3_ojmcd_common as optimizer_geometry
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rmamcd_p0/protocol/rmamcd_p0_protocol.json"
)
FIXED_THETA = base.THETA_MAX - 1e-6
METHOD_NAME = "RMA-MCD"
_BASE_SELECT_ROUTE = base.select_route
_BASE_EVALUATE_GATES = base.evaluate_gates
_BASE_CONTEXT = base.context


def context(path: Any) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol, config, targets, risk_rows = _BASE_CONTEXT(path)
    design = protocol.get("design", {})
    if (
        protocol.get("method_variant") != METHOD_NAME
        or protocol.get("predecessor", {}).get("route")
        != "STOP_NPTLRMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED"
        or design.get("target_proposal")
        != "global_pooled_protected_target_margin_affine_max_min"
        or design.get("projection_rule")
        != "shared_float64_gram_pseudoinverse_projection_of_7_normalized_global_covectors_off_raw_displacement_only"
        or int(design.get("target_covectors", -1)) != 7
        or int(design.get("protected_constraints", -1))
        != base.FIT_UNITS * len(base.PROTECTED_CERTIFICATES)
        or design.get("unit_level_protected_nullspace_claimed") is not False
        or float(design.get("minimum_retained_global_covector_fraction", -1.0))
        != 0.01
        or int(config.get("rmamcd_global_basis_count", -1)) != 7
        or float(config.get("rmamcd_minimum_retained_global_covector_fraction", -1.0))
        != 0.01
        or config.get("rmamcd_unit_level_protected_nullspace_claimed") is not False
    ):
        raise base.NptmcdP0ValidationError(
            "RMA-MCD protocol registry differs from the runner"
        )
    return protocol, config, targets, risk_rows


def _global_basis(
    fit_gradients: list[dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    if len(fit_gradients) != base.FIT_UNITS:
        raise ValueError("RMA global basis requires the frozen fit-unit count")
    normalized_units = []
    input_norms = []
    for fit in fit_gradients:
        normalized = {}
        for certificate in base.CERTIFICATES:
            norm = gradient_math.gradient_norm(fit[certificate])
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError(f"degenerate fit gradient for {certificate}")
            input_norms.append(norm)
            normalized[certificate] = gradient_ops.scaled_gradient(
                fit[certificate], 1.0 / norm
            )
        normalized_units.append(normalized)
    basis = base.medsc.mean_certificate_gradients(normalized_units)
    for certificate, gradient in basis.items():
        norm = gradient_math.gradient_norm(gradient)
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"global basis covector is degenerate for {certificate}")
    return basis, input_norms


def _margin_search(
    raw_panels: list[dict[str, float]],
    candidate_panels: list[dict[str, float]],
    solver_audit: dict[str, Any],
    projection_valid: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    protected = tuple(base.PROTECTED_CERTIFICATES)
    targets = tuple(base.TARGET_CERTIFICATES)
    if len(raw_panels) != len(candidate_panels) or not raw_panels:
        raise ValueError("RMA fit score panels differ or are empty")
    raw_target_floor = min(
        float(panel[name]) for panel in raw_panels for name in targets
    )
    candidate_target_floor = min(
        float(panel[name]) for panel in candidate_panels for name in targets
    )
    target_gain = candidate_target_floor - raw_target_floor
    if any(float(panel[name]) <= 0.0 for panel in raw_panels for name in protected):
        raise ValueError("RMA fit protected raw score is nonpositive")
    candidate_scores = {
        name: [float(panel[name]) for panel in candidate_panels]
        for name in (*protected, *targets)
    }
    raw_scores = {
        name: [float(panel[name]) for panel in raw_panels]
        for name in (*protected, *targets)
    }
    retentions = {
        name: min(
            candidate / raw
            for raw, candidate in zip(raw_scores[name], candidate_scores[name], strict=True)
        )
        for name in protected
    }
    min_retention = min(retentions.values())
    identity_errors = [
        abs(candidate / raw - math.cos(FIXED_THETA))
        for name in protected
        for raw, candidate in zip(raw_scores[name], candidate_scores[name], strict=True)
    ]
    candidate_protected_margins = [
        retention - base.PROTECTED_RETENTION_FLOOR
        for retention in retentions.values()
    ]
    candidate_target_margins = [
        score - base.FIT_TARGET_SCORE_FLOOR
        for name in targets
        for score in candidate_scores[name]
    ]
    actual_minimum_margin = min(
        *candidate_protected_margins,
        *candidate_target_margins,
    )
    gates = protocol["gates"]
    margin = solver_audit.get("solver", {}).get("primal_value")
    numeric_tolerance = float(gates["numeric_tolerance"])
    eligible = (
        projection_valid
        and solver_audit.get("solver_valid") is True
        and FIXED_THETA >= float(gates["minimum_nonzero_angle"])
        and FIXED_THETA <= base.THETA_MAX + numeric_tolerance
        and min_retention >= base.PROTECTED_RETENTION_FLOOR - numeric_tolerance
        and candidate_target_floor >= base.FIT_TARGET_SCORE_FLOOR - numeric_tolerance
        and target_gain >= base.FIT_TARGET_GAIN_FLOOR - numeric_tolerance
        and actual_minimum_margin >= -numeric_tolerance
        and float(solver_audit.get("solver_margin", -math.inf))
        >= -numeric_tolerance
    )
    return {
        "eligible": eligible,
        "selected_theta": FIXED_THETA if projection_valid else 0.0,
        "selected_theta_degrees": math.degrees(FIXED_THETA if projection_valid else 0.0),
        "theta_max": base.THETA_MAX,
        "theta_max_degrees": math.degrees(base.THETA_MAX),
        "design_cosine_floor": base.DESIGN_COSINE_FLOOR,
        "protected_retention_floor": base.PROTECTED_RETENTION_FLOOR,
        "raw_target_score_floor": raw_target_floor,
        "required_target_score_floor": max(
            base.FIT_TARGET_SCORE_FLOOR,
            raw_target_floor + base.FIT_TARGET_GAIN_FLOOR,
        ),
        "selected_target_score_floor": candidate_target_floor,
        "selected_target_gain_over_raw": target_gain,
        "protected_retentions": retentions,
        "minimum_protected_retention": min_retention,
        "maximum_protected_retention_identity_error": max(identity_errors),
        "candidate_protected_margin_floor": min(candidate_protected_margins),
        "candidate_target_margin_floor": min(candidate_target_margins),
        "candidate_actual_minimum_margin": actual_minimum_margin,
        "fit_protected_facet_count": len(raw_panels) * len(protected),
        "fit_target_facet_count": len(raw_panels) * len(targets),
        "candidate_panels": candidate_panels,
        "target_coefficients": solver_audit.get("objective_names", []),
        "trigonometric_boundaries": [0.0, FIXED_THETA, base.THETA_MAX],
        "selection_rule": "fixed_theta_joint_global_target_and_protected_margin_affine_max_min",
        "calibration_scores_used_for_selection": False,
        "diagnostic_scores_used_for_selection": False,
        "fixed_theta": FIXED_THETA,
        "fixed_theta_margin_radians": 1e-6,
        "joint_margin_primal_value": margin,
        "solver_predicted_minimum_score": margin,
        "solver_candidate_actual_minimum_score": solver_audit.get("actual_minimum_margin"),
        "solver_candidate_actual_minus_predicted": solver_audit.get("actual_minus_predicted"),
    }


def build_fit_geometry(
    fit_gradients: list[dict[str, dict[str, Any]]],
    fit_coverages: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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
    global_basis, input_norms = _global_basis(fit_gradients)
    global_scores = base.frsc.displacement_scores(
        global_basis, raw_displacement, reference_norm
    )["scores"]
    raw_target_floor = min(
        float(panel[name]) for panel in raw_score_panels[1:] for name in base.TARGET_CERTIFICATES
    )

    basis_names = [*base.PROTECTED_CERTIFICATES, *base.TARGET_CERTIFICATES]
    normalized_basis = [
        gradient_ops.scaled_gradient(
            global_basis[name], 1.0 / gradient_math.gradient_norm(global_basis[name])
        )
        for name in basis_names
    ]
    tangent = gradient_ops.scaled_gradient(raw_displacement, 0.0)
    proposed_candidate = {name: value.clone() for name, value in raw_displacement.items()}
    solver_audit: dict[str, Any] = {"solver_valid": False}
    projection_valid = False
    projection_error = None
    try:
        projected, projection = affine.project_registry_to_constraint_nullspace(
            normalized_basis,
            [raw_displacement],
            float(protocol["design"]["projection_rcond"]),
            int(protocol["design"]["projection_refinement_steps"]),
        )
        objective_names = []
        objective_gradients = {}
        offsets = {}
        for name, projected_gradient in zip(basis_names, projected, strict=True):
            if name in base.PROTECTED_CERTIFICATES:
                raw_score = float(global_scores[name])
                if raw_score <= 0.0:
                    raise ValueError("pooled protected raw score is nonpositive")
                objective_names.append(f"protected:{name}")
                objective_gradients[f"protected:{name}"] = gradient_ops.scaled_gradient(
                    projected_gradient,
                    math.sin(FIXED_THETA) / (reference_norm * raw_score),
                )
                offsets[f"protected:{name}"] = math.cos(FIXED_THETA) - base.PROTECTED_RETENTION_FLOOR
            else:
                objective_names.append(f"target:{name}")
                objective_gradients[f"target:{name}"] = gradient_ops.scaled_gradient(
                    projected_gradient, math.sin(FIXED_THETA) / reference_norm
                )
                offsets[f"target:{name}"] = (
                    float(global_scores[name]) * math.cos(FIXED_THETA)
                    - base.FIT_TARGET_SCORE_FLOOR
                )
        tangent, solver = optimizer_geometry.affine_minimax_direction(
            objective_gradients,
            offsets,
            gradient_math.gradient_norm(raw_displacement),
            float(protocol["design"]["affine_solver_tolerance"]),
        )
        proposed_candidate = base.npt.angular_displacement(
            raw_displacement, tangent, FIXED_THETA
        )
        candidate_panel_scores = [
            base.frsc.displacement_scores(panel, proposed_candidate, reference_norm)[
                "scores"
            ]
            for panel in panels[1:]
        ]
        candidate_margins = {}
        for name in objective_names:
            kind, certificate = name.split(":", 1)
            if kind == "protected":
                candidate_margins[name] = min(
                    float(candidate[certificate]) / float(raw[certificate])
                    for raw, candidate in zip(
                        raw_score_panels[1:], candidate_panel_scores, strict=True
                    )
                ) - base.PROTECTED_RETENTION_FLOOR
            else:
                candidate_margins[name] = min(
                    float(candidate[certificate]) - base.FIT_TARGET_SCORE_FLOOR
                    for candidate in candidate_panel_scores
                )
        actual_margin = min(candidate_margins.values())
        kkt_limit = float(protocol["design"]["affine_kkt_residual_max"])
        if (
            not bool(solver["solver_success"])
            or float(solver["kkt_residual"]) > kkt_limit
            or float(solver["primal_dual_gap"]) > kkt_limit
        ):
            raise ValueError("global affine simplex KKT audit exceeded frozen limit")
        projection_valid = True
        solver_audit = {
            "solver_valid": True,
            "projection": projection,
            "solver": solver,
            "objective_names": objective_names,
            "offsets": offsets,
            "global_scores": global_scores,
            "solver_margin": float(solver["primal_value"]),
            "actual_minimum_margin": actual_margin,
            "actual_minus_predicted": actual_margin - float(solver["primal_value"]),
            "kkt_residual_max": kkt_limit,
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
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
    search = _margin_search(
        raw_score_panels[1:],
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
        affine_runner._protected_audit(raw_displacement, tangent, protected_constraints)
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
    projection_audit = solver_audit.get("projection", {})
    projected_fractions = [
        float(item["projected_norm"]) / max(float(item["input_norm"]), 1e-30)
        for item in projection_audit.get("vectors", [])
    ]
    tangent_audit = {
        **protected_audit,
        "projection_valid": projection_valid,
        "projection_error": projection_error,
        "protected_constraints": len(protected_constraints),
        "retained_target_proposal_fraction": min(
            projected_fractions
        )
        if projected_fractions
        else 0.0,
        "minimum_global_covector_projection_fraction": min(projected_fractions)
        if projected_fractions
        else 0.0,
        "target_covector_projection": projection_audit,
        "affine_solver": solver_audit.get("solver"),
        "target_covector_count": len(basis_names),
        "enforcement": "global_pooled_margin_not_unit_level_nullspace",
    }
    selected_map = {
        "map_kind": "direct_fixed_norm_protected_nullspace_angular_rotation",
        "map_variant": "global_pooled_protected_target_margin_affine_max_min",
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
        "global_margin_affine_max_min_used": projection_valid,
    }
    fit_geometry = {
        "name": "step3_rmamcd_p0_fit_geometry_v1",
        "method": METHOD_NAME,
        "gradient_measurement_mode": "eval",
        "unit_granularity": "one_prompt_risk_pair",
        "fit_coverages": fit_coverages,
        "pooled_gradient_norms": {
            name: gradient_math.gradient_norm(gradient) for name, gradient in pooled.items()
        },
        "raw_baseline_gradient_norm": gradient_math.gradient_norm(raw_baseline),
        "auxiliary_budget_norm": budget,
        "reference_displacement_norm": reference_norm,
        "baseline_optimizer_map": baseline_map,
        "raw_optimizer_map": raw_map,
        "target_optimizer_map": target_map,
        "selected_optimizer_map": selected_map,
        "raw_direction_budget_relative_error": abs(gradient_math.gradient_norm(raw_direction) - budget) / budget,
        "target_direction_budget_relative_error": abs(gradient_math.gradient_norm(target_direction) - budget) / budget,
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
        "global_basis": {
            "certificates": basis_names,
            "count": len(basis_names),
            "rule": "equal_weight_mean_of_twelve_unit_normalized_fit_certificates",
            "input_gradient_norm_min": min(input_norms),
            "input_gradient_norm_max": max(input_norms),
            "raw_scores": global_scores,
            "raw_target_floor_for_fit_audit": raw_target_floor,
        },
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
            "selected": base.frsc.displacement_scores(pooled, selected_displacement, reference_norm)["scores"],
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
    fixed = {"baseline": baseline_displacement, "raw_mcd": raw_displacement, "nptmcd_mcd": selected_displacement}
    del pooled, raw_baseline, raw_direction, target_direction, target_displacement, global_basis, tangent, protected_constraints
    return fit_geometry, fixed


def _global_margin_geometry_gate(
    fit_geometry: dict[str, Any], protocol: dict[str, Any]
) -> bool:
    gates = protocol["gates"]
    design = protocol["design"]
    audit = fit_geometry.get("tangent_audit", {})
    projection = audit.get("target_covector_projection") or {}
    solver = audit.get("affine_solver") or {}
    vectors = projection.get("vectors") or []
    minimum_fraction = float(
        design.get(
            "minimum_retained_global_covector_fraction",
            design["minimum_retained_target_proposal_fraction"],
        )
    )
    return (
        bool(fit_geometry.get("projection_valid"))
        and audit.get("enforcement") == "global_pooled_margin_not_unit_level_nullspace"
        and int(audit.get("target_covector_count", -1)) == 7
        and int(projection.get("constraints", -1)) == 1
        and len(vectors) == 7
        and float(audit.get("minimum_global_covector_projection_fraction", 0.0))
        >= minimum_fraction
        and float(projection.get("maximum_relative_final_constraint_residual", math.inf))
        <= float(gates["projection_residual_relative_max"])
        and abs(float(audit.get("raw_tangent_cosine", math.inf)))
        <= float(gates["projection_residual_relative_max"])
        and abs(float(audit.get("tangent_to_raw_norm_ratio", math.inf)) - 1.0)
        <= float(gates["candidate_norm_relative_tolerance"])
        and bool(solver.get("solver_success"))
        and float(solver.get("kkt_residual", math.inf))
        <= float(design["affine_kkt_residual_max"])
        and float(solver.get("primal_dual_gap", math.inf))
        <= float(design["affine_kkt_residual_max"])
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
    gates = _BASE_EVALUATE_GATES(
        rows, fit_geometry, calibration_rows, summary, protocol,
        state_restored, vram_headroom, gpu_name,
    )
    search = fit_geometry["search"]
    gates["fit_minimum_effective_angle"] = (
        bool(search.get("eligible"))
        and float(search["selected_theta"]) >= float(protocol["gates"]["minimum_nonzero_angle"])
        and float(search["selected_theta"]) <= base.THETA_MAX + float(protocol["gates"]["numeric_tolerance"])
        and float(search["minimum_protected_retention"]) >= base.PROTECTED_RETENTION_FLOOR
        and float(search["selected_target_score_floor"]) >= base.FIT_TARGET_SCORE_FLOOR
        and float(search["selected_target_gain_over_raw"]) >= base.FIT_TARGET_GAIN_FLOOR
        and float(search["candidate_actual_minimum_margin"])
        >= -float(protocol["gates"]["numeric_tolerance"])
    )
    # This gate keeps the inherited route ordering, but its meaning is the
    # RMA global-margin geometry. Unit-level protected orthogonality is not
    # claimed by this method and is reported separately in the audit.
    gates["protected_nullspace_geometry"] = _global_margin_geometry_gate(
        fit_geometry, protocol
    )
    gates["global_margin_geometry"] = gates["protected_nullspace_geometry"]
    return gates


def select_route(gates: dict[str, bool]) -> str:
    route = _BASE_SELECT_ROUTE(gates)
    if route == "STOP_NPTMCD_P0_NULLSPACE_GEOMETRY_INVALID":
        return "STOP_RMAMCD_P0_GLOBAL_MARGIN_GEOMETRY_INVALID"
    return route.replace("NPTMCD", "RMAMCD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = base.resolve_repo_path(args.protocol)
    base.build_fit_geometry = build_fit_geometry
    base.context = context
    base.evaluate_gates = evaluate_gates
    base.select_route = select_route
    if args.command == "preflight":
        base.preflight(path)
    elif args.command == "run":
        base.run(path)
    else:
        base.verify_run(path)


if __name__ == "__main__":
    main()
