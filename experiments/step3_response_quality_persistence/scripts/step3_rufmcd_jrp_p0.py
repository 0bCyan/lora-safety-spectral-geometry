"""RUF-MCD JRP: fit-only jackknife-reflected TARC polytope P0."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import step3_nptmcd_affine_common as affine
import step3_nptmcd_p0 as base
import step3_rufmcd_p0 as v2
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math
import step3_etarc_p0 as etarc_p0
import step3_ramp_p0 as ramp_p0
import step3_jcrmcd_p0 as jcr_runner
import step3_medscmcd_common as medsc


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
METHOD_NAME = "RUF-MCD-JRP"
FIXED_THETA = v2.FIXED_THETA
ORIGINAL_FACET_COUNT = v2.FACET_COUNT
REFLECTION_VERTEX_COUNT = base.FIT_UNITS
ROBUST_EXTRA_FACET_COUNT = base.FIT_UNITS
FACET_COUNT = ORIGINAL_FACET_COUNT + ROBUST_EXTRA_FACET_COUNT


def _read_json(path: Path) -> dict[str, Any]:
    return etarc_p0.read_json(path)


def _mean_gradient(vectors: list[dict[str, Any]]) -> dict[str, Any]:
    if not vectors:
        raise ValueError("cannot average an empty gradient registry")
    result = gradient_ops.scaled_gradient(vectors[0], 1.0)
    for vector in vectors[1:]:
        result = gradient_ops.add_gradients(result, vector)
    return gradient_ops.scaled_gradient(result, 1.0 / len(vectors))


def _jackknife_reflected_tarc_specs(
    fit_gradients: list[dict[str, dict[str, Any]]],
    raw_score_panels: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a finite, fit-only uncertainty polytope in retention-score space.

    For normalized TARC covector b_i and positive raw score r_i, the relevant
    protected response is c_i=b_i/r_i.  The leave-one-out reflection is
    c_i^- = 2 mean(c_{-i}) - c_i.  The registered uncertainty set is the
    convex hull of the observed c_i and these reflected vertices.  The
    observed vertices are already present in the original fit facets, so only
    the reflected vertices are added to the affine objective.
    """
    if len(fit_gradients) != base.FIT_UNITS:
        raise ValueError("JRP requires the frozen fit-unit count")
    response_covectors = []
    raw_scores = []
    for index, (fit, panel) in enumerate(
        zip(fit_gradients, raw_score_panels, strict=True), 1
    ):
        normed, _ = v2._normalized_gradient(fit["tarc"], f"fit_{index:02d}:tarc")
        raw_score = float(panel["tarc"])
        if not math.isfinite(raw_score) or raw_score <= 0.0:
            raise ValueError("JRP requires positive fit TARC raw scores")
        response_covectors.append(
            gradient_ops.scaled_gradient(normed, 1.0 / raw_score)
        )
        raw_scores.append(raw_score)

    reflected_specs = []
    reflection_audit = []
    for index, (covector, raw_score) in enumerate(
        zip(response_covectors, raw_scores, strict=True)
    ):
        others = [
            item for offset, item in enumerate(response_covectors) if offset != index
        ]
        loo_mean = _mean_gradient(others)
        reflected = gradient_ops.add_gradients(
            gradient_ops.scaled_gradient(loo_mean, 2.0),
            covector,
            right_scale=-1.0,
        )
        reflected_norm = gradient_math.gradient_norm(reflected)
        if not math.isfinite(reflected_norm) or reflected_norm <= 0.0:
            raise ValueError("JRP reflection is degenerate")
        name = f"fit_{index + 1:02d}:tarc:jackknife_reflection"
        reflected_specs.append(
            {
                "name": name,
                "kind": "robust_protected",
                "unit_index": index,
                "certificate": "tarc",
                "gradient": reflected,
                "response_covector_norm": reflected_norm,
                "raw_score": 1.0,
                "source_raw_score": raw_score,
                "vertex_type": "leave_one_out_reflection",
            }
        )
        reflection_audit.append(
            {
                "source_unit": f"fit_{index + 1:02d}",
                "source_raw_tarc_score": raw_score,
                "leave_one_out_size": len(others),
                "reflected_response_covector_norm": reflected_norm,
            }
        )
    return reflected_specs, {
        "set_definition": "conv(observed_normalized_retention_covectors_and_leave_one_out_reflections)",
        "observed_vertex_count": base.FIT_UNITS,
        "reflected_vertex_count": REFLECTION_VERTEX_COUNT,
        "additional_affine_facet_count": ROBUST_EXTRA_FACET_COUNT,
        "vertices_used_in_objective": ROBUST_EXTRA_FACET_COUNT,
        "reflection_rule": "c_i_minus=2*mean(c_j,j!=i)-c_i with c_i=normalized_tarc_covector/raw_tarc_score_i",
        "population_coverage_claimed": False,
        "vertices": reflection_audit,
    }


def _robust_margin(
    projected_response_covector: dict[str, Any],
    tangent: dict[str, Any],
    reference_norm: float,
) -> float:
    return (
        math.cos(FIXED_THETA) - base.PROTECTED_RETENTION_FLOOR
        + math.sin(FIXED_THETA)
        * gradient_math.gradient_dot(projected_response_covector, tangent)
        / reference_norm
    )


def context(path: Path):
    protocol = _read_json(path)
    if protocol.get("status") != "prospective_before_any_RUFMCD_JRP_model_forward":
        raise base.NptmcdP0ValidationError("JRP protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = base.resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise base.NptmcdP0ValidationError(f"frozen input hash mismatch: {name}")
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
        raise base.NptmcdP0ValidationError("JRP protocol violates zero-step boundary")
    prior = _read_json(base.resolve_repo_path(protocol["prior_result"]["decision_path"]))
    if prior.get("route") != "STOP_RUFMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED":
        raise base.NptmcdP0ValidationError("registered v2 predecessor route differs")
    design = protocol["design"]
    if (
        tuple(design["certificates"]) != base.CERTIFICATES
        or tuple(design["protected_certificates"]) != base.PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != base.TARGET_CERTIFICATES
        or int(design["fit_units"]) != base.FIT_UNITS
        or int(design["calibration_units"]) != base.CALIBRATION_UNITS
        or int(design["diagnostic_units"]) != base.DIAGNOSTIC_UNITS
        or float(design["protected_retention_floor"]) != base.PROTECTED_RETENTION_FLOOR
        or float(design["design_cosine_floor"]) != base.DESIGN_COSINE_FLOOR
        or float(design["fixed_theta_radians"]) != FIXED_THETA
        or int(design["original_facet_count"]) != ORIGINAL_FACET_COUNT
        or int(design["reflected_tarc_vertex_count"]) != REFLECTION_VERTEX_COUNT
        or int(design["total_affine_facet_count"]) != FACET_COUNT
        or design.get("calibration_scores_used_for_selection") is not False
        or design.get("diagnostic_scores_used_for_selection") is not False
        or design.get("population_coverage_claimed") is not False
    ):
        raise base.NptmcdP0ValidationError("JRP design registry differs")
    config = _read_json(base.resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 7
        or config.get("required_gpu_name_substring") != "RTX 4070 Ti"
        or config.get("gradient_measurement_mode") != "eval"
        or config.get("training_authorized") is not False
        or int(config["optimizer_steps_authorized"]) != 0
        or int(config.get("rufmcd_jrp_total_facet_count", -1)) != FACET_COUNT
    ):
        raise base.NptmcdP0ValidationError("JRP config registry differs")
    targets = etarc_p0.read_jsonl(base.resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = base.training.read_jsonl(
        base.resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise base.NptmcdP0ValidationError("risk row count differs")
    generator = __import__("torch").Generator(device="cpu")
    generator.manual_seed(int(protocol["selection"]["risk_order_seed"]))
    risk_order = __import__("torch").randperm(len(risk_rows), generator=generator).tolist()
    expected = base.expected_roles(targets, risk_order)
    for role in ("fit", "calibration", "diagnostic"):
        if protocol["selection"][f"{role}_units"] != expected[role]:
            raise base.NptmcdP0ValidationError(f"{role} role registry differs")
    all_units = [item for role in expected.values() for item in role]
    if len({item["sample_id"] for item in all_units}) != 40:
        raise base.NptmcdP0ValidationError("JRP role sample IDs overlap")
    return protocol, config, targets, risk_rows


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
        raw_baseline, None, 0.0, float(config["max_grad_norm"]),
        float(config["adam_epsilon"]), float(config["clip_epsilon"]),
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    raw_displacement, raw_map = base.frsc.realized_displacement(
        raw_baseline, raw_direction, float(config["safety_mix"]),
        float(config["max_grad_norm"]), float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    target_displacement, target_map = base.frsc.realized_displacement(
        raw_baseline, target_direction, float(config["safety_mix"]),
        float(config["max_grad_norm"]), float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    panels = [pooled, *fit_gradients]
    raw_score_panels = [
        base.frsc.displacement_scores(panel, raw_displacement, reference_norm)["scores"]
        for panel in panels
    ]
    original_specs = v2._facet_specs(fit_gradients)
    robust_specs, reflection_audit = _jackknife_reflected_tarc_specs(
        fit_gradients, raw_score_panels[1:]
    )
    specs = [*original_specs, *robust_specs]
    raw_target_floor = min(
        float(panel[name])
        for panel in raw_score_panels[1:]
        for name in base.TARGET_CERTIFICATES
    )
    required_target_floor = max(
        base.FIT_TARGET_SCORE_FLOOR, raw_target_floor + base.FIT_TARGET_GAIN_FLOOR
    )
    tangent = gradient_ops.scaled_gradient(raw_displacement, 0.0)
    proposed_candidate = {name: value.clone() for name, value in raw_displacement.items()}
    solver_audit: dict[str, Any] = {"solver_valid": False}
    projection_valid = False
    projection_error = None
    projected_by_name: dict[str, dict[str, Any]] = {}
    try:
        projected, projection = affine.project_registry_to_constraint_nullspace(
            [spec["gradient"] for spec in specs], [raw_displacement],
            float(protocol["design"]["projection_rcond"]),
            int(protocol["design"]["projection_refinement_steps"]),
        )
        objective_names = []
        objective_gradients = {}
        offsets = {}
        facet_groups = {}
        for spec, projected_gradient in zip(specs, projected, strict=True):
            name = spec["name"]
            projected_by_name[name] = projected_gradient
            objective_names.append(name)
            facet_groups[name] = spec["certificate"]
            if spec["kind"] == "protected":
                scale = math.sin(FIXED_THETA) / (
                    reference_norm * float(spec["raw_score"])
                )
                offsets[name] = math.cos(FIXED_THETA) - base.PROTECTED_RETENTION_FLOOR
            elif spec["kind"] == "robust_protected":
                scale = math.sin(FIXED_THETA) / reference_norm
                offsets[name] = math.cos(FIXED_THETA) - base.PROTECTED_RETENTION_FLOOR
            else:
                scale = math.sin(FIXED_THETA) / reference_norm
                raw_score = float(raw_score_panels[spec["unit_index"] + 1][spec["certificate"]])
                offsets[name] = raw_score * math.cos(FIXED_THETA) - required_target_floor
            objective_gradients[name] = gradient_ops.scaled_gradient(
                projected_gradient, scale
            )
        if len(objective_names) != FACET_COUNT:
            raise ValueError("JRP objective facet count differs")
        kkt_limit = float(protocol["design"]["affine_kkt_residual_max"])
        tangent, solver = v2._constraint_generation_affine_minimax(
            objective_gradients, offsets, facet_groups,
            gradient_math.gradient_norm(raw_displacement),
            float(protocol["design"]["affine_solver_tolerance"]), kkt_limit,
            float(protocol["design"]["constraint_generation_full_gap_max"]),
            int(protocol["design"]["constraint_generation_max_rounds"]),
        )
        proposed_candidate = base.npt.angular_displacement(
            raw_displacement, tangent, FIXED_THETA
        )
        candidate_panel_scores = [
            base.frsc.displacement_scores(panel, proposed_candidate, reference_norm)["scores"]
            for panel in panels[1:]
        ]
        original_margins = []
        for spec in original_specs:
            panel = candidate_panel_scores[spec["unit_index"]]
            raw_panel = raw_score_panels[spec["unit_index"] + 1]
            if spec["kind"] == "protected":
                original_margins.append(
                    float(panel[spec["certificate"]])
                    / float(raw_panel[spec["certificate"]])
                    - base.PROTECTED_RETENTION_FLOOR
                )
            else:
                original_margins.append(
                    float(panel[spec["certificate"]]) - required_target_floor
                )
        robust_margins = [
            _robust_margin(
                projected_by_name[spec["name"]],
                tangent,
                reference_norm,
            )
            for spec in robust_specs
        ]
        actual_margin = min(*original_margins, *robust_margins)
        if (
            not bool(solver["solver_success"])
            or float(solver["kkt_residual"]) > kkt_limit
            or float(solver["primal_dual_gap"]) > kkt_limit
            or float(solver["full_primal_dual_gap"])
            > float(protocol["design"]["constraint_generation_full_gap_max"])
        ):
            raise ValueError("JRP affine certificate exceeded frozen limit")
        projection_valid = True
        solver_audit = {
            "solver_valid": True,
            "projection": projection,
            "solver": solver,
            "objective_names": objective_names,
            "offsets": offsets,
            "solver_margin": float(solver["full_primal_value"]),
            "actual_minimum_margin": actual_margin,
            "original_minimum_margin": min(original_margins),
            "robust_minimum_margin": min(robust_margins),
            "actual_minus_predicted": actual_margin - float(solver["full_primal_value"]),
            "kkt_residual_max": kkt_limit,
            "required_target_floor": required_target_floor,
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
    search = v2._margin_search(
        raw_score_panels[1:], candidate_score_panels[1:], solver_audit,
        projection_valid, protocol,
    )
    if projection_valid:
        robust_floor = float(solver_audit["robust_minimum_margin"])
        actual_margin = float(solver_audit["actual_minimum_margin"])
        predicted = float(solver_audit["solver_margin"])
        search["robust_protected_constraint_count"] = ROBUST_EXTRA_FACET_COUNT
        search["robust_polytope_vertex_count"] = REFLECTION_VERTEX_COUNT * 2
        search["robust_candidate_margin_floor"] = robust_floor
        search["candidate_actual_minimum_margin"] = actual_margin
        search["solver_candidate_actual_minimum_score"] = actual_margin
        search["solver_candidate_actual_minus_predicted"] = actual_margin - predicted
        search["eligible"] = bool(search["eligible"]) and robust_floor >= -float(
            protocol["gates"]["numeric_tolerance"]
        ) and abs(actual_margin - predicted) <= float(
            protocol["design"]["affine_actual_minus_predicted_max"]
        )
    else:
        search["robust_protected_constraint_count"] = ROBUST_EXTRA_FACET_COUNT
        search["robust_polytope_vertex_count"] = REFLECTION_VERTEX_COUNT * 2
        search["robust_candidate_margin_floor"] = -math.inf
        search["eligible"] = False
    selected_displacement = (
        proposed_candidate if bool(search["eligible"]) else raw_displacement
    )
    selected_norm = gradient_math.gradient_norm(selected_displacement)
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    transfer_audit = (
        v2._protected_transfer_audit(raw_displacement, tangent, fit_gradients)
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
        **transfer_audit,
        "projection_valid": projection_valid,
        "projection_error": projection_error,
        "protected_constraints": ORIGINAL_FACET_COUNT - base.FIT_UNITS * len(base.TARGET_CERTIFICATES),
        "robust_protected_constraints": ROBUST_EXTRA_FACET_COUNT,
        "target_covector_count": FACET_COUNT,
        "minimum_unit_facet_projection_fraction": min(projected_fractions) if projected_fractions else 0.0,
        "retained_target_proposal_fraction": min(projected_fractions) if projected_fractions else 0.0,
        "target_covector_projection": projection_audit,
        "affine_solver": solver_audit.get("solver"),
        "facet_count": FACET_COUNT,
        "enforcement": "unit_facet_margin_plus_jackknife_reflected_tarc_polytope",
        "unit_level_protected_nullspace_claimed": False,
    }
    selected_map = {
        "map_kind": "direct_fixed_norm_protected_nullspace_angular_rotation",
        "map_variant": "unit_facet_jackknife_reflected_tarc_polytope_margin_affine_max_min",
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
        "unit_facet_affine_max_min_used": projection_valid,
    }
    fit_geometry = {
        "name": "step3_rufmcd_jrp_p0_fit_geometry_v1",
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
        "raw_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(raw_direction) - budget
        ) / budget,
        "target_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(target_direction) - budget
        ) / budget,
        "endpoint_audit": endpoint_audit,
        "protected_constraint_labels": [
            f"fit_{fit_index + 1:02d}:{name}"
            for fit_index in range(base.FIT_UNITS)
            for name in base.PROTECTED_CERTIFICATES
        ],
        "protected_constraint_count": base.FIT_UNITS * len(base.PROTECTED_CERTIFICATES),
        "robust_protected_constraint_labels": [spec["name"] for spec in robust_specs],
        "robust_protected_constraint_count": ROBUST_EXTRA_FACET_COUNT,
        "target_facet_labels": [spec["name"] for spec in original_specs if spec["kind"] == "target"],
        "target_facet_count": base.FIT_UNITS * len(base.TARGET_CERTIFICATES),
        "facet_count": FACET_COUNT,
        "target_proposal_norm": gradient_math.gradient_norm(
            gradient_ops.add_gradients(target_displacement, raw_displacement, right_scale=-1.0)
        ),
        "target_endpoint_difference_used": False,
        "unit_facet_rule": "all_fit_unit_facets_plus_fit_only_jackknife_reflected_tarc_retention_vertices",
        "jackknife_reflection_audit": reflection_audit,
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
    return fit_geometry, fixed


def _geometry_gate(fit_geometry: dict[str, Any], protocol: dict[str, Any]) -> bool:
    design = protocol["design"]
    gates = protocol["gates"]
    audit = fit_geometry.get("tangent_audit", {})
    projection = audit.get("target_covector_projection") or {}
    solver = audit.get("affine_solver") or {}
    full_scores = solver.get("full_certificate_scores") or {}
    full_primal = float(solver.get("full_primal_value", math.inf))
    full_dual = float(solver.get("full_dual_upper_bound", -math.inf))
    full_gap = float(solver.get("full_primal_dual_gap", math.inf))
    full_min = min(full_scores.values()) if full_scores else math.inf
    return (
        bool(fit_geometry.get("projection_valid"))
        and audit.get("enforcement") == "unit_facet_margin_plus_jackknife_reflected_tarc_polytope"
        and int(audit.get("target_covector_count", -1)) == FACET_COUNT
        and int(projection.get("constraints", -1)) == 1
        and len(projection.get("vectors") or []) == FACET_COUNT
        and len(full_scores) == FACET_COUNT
        and bool(solver.get("constraint_generation_certified"))
        and int(solver.get("constraint_generation_total_facet_count", -1)) == FACET_COUNT
        and 0 < int(solver.get("constraint_generation_round_count", 0)) <= int(
            design["constraint_generation_max_rounds"]
        )
        and full_dual + float(design["constraint_generation_full_gap_max"]) >= full_primal
        and abs(full_min - full_primal) <= float(design["constraint_generation_full_gap_max"])
        and full_gap <= float(design["constraint_generation_full_gap_max"])
        and float(audit.get("minimum_unit_facet_projection_fraction", 0.0))
        >= float(design["minimum_retained_unit_facet_fraction"])
        and float(projection.get("maximum_relative_final_constraint_residual", math.inf))
        <= float(gates["projection_residual_relative_max"])
        and abs(float(audit.get("raw_tangent_cosine", math.inf)))
        <= float(gates["projection_residual_relative_max"])
        and abs(float(audit.get("tangent_to_raw_norm_ratio", math.inf)) - 1.0)
        <= float(gates["candidate_norm_relative_tolerance"])
        and bool(solver.get("solver_success"))
        and float(solver.get("kkt_residual", math.inf)) <= float(design["affine_kkt_residual_max"])
        and float(solver.get("primal_dual_gap", math.inf)) <= float(design["affine_kkt_residual_max"])
        and abs(float(fit_geometry["search"].get("solver_candidate_actual_minus_predicted", math.inf)))
        <= float(design["affine_actual_minus_predicted_max"])
    )


def evaluate_gates(*args: Any, **kwargs: Any) -> dict[str, bool]:
    rows, fit_geometry, calibration_rows, summary, protocol, state_restored, vram, gpu = args
    gates = base.evaluate_gates(
        rows, fit_geometry, calibration_rows, summary, protocol, state_restored, vram, gpu
    )
    gates["protected_nullspace_geometry"] = _geometry_gate(fit_geometry, protocol)
    gates["unit_facet_margin_geometry"] = gates["protected_nullspace_geometry"]
    search = fit_geometry["search"]
    gates["fit_minimum_effective_angle"] = bool(gates["fit_minimum_effective_angle"]) and float(
        search.get("robust_candidate_margin_floor", -math.inf)
    ) >= -float(protocol["gates"]["numeric_tolerance"])
    return gates


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"]:
        return "STOP_RUFMCD_JRP_P0_INVALID"
    if not gates["protected_nullspace_geometry"]:
        return "STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID"
    if not gates["fit_minimum_effective_angle"]:
        return "STOP_RUFMCD_JRP_P0_NO_ROBUST_EFFECTIVE_ANGLE"
    if not gates["split_conformal_accept_or_raw_policy"]:
        return "STOP_RUFMCD_JRP_P0_CONFORMAL_SELECTIVE_POLICY_FAILED"
    if not gates["diagnostic_common_transfer"]:
        return "STOP_RUFMCD_JRP_P0_DIAGNOSTIC_TRANSFER_FAILED"
    if not gates["finite_step_common_descent"]:
        return "STOP_RUFMCD_JRP_P0_FINITE_STEP_COMMON_DESCENT_FAILED"
    if not gates["protected_diagnostic_noninferiority"]:
        return "STOP_RUFMCD_JRP_P0_PROTECTED_NONINFERIORITY_FAILED"
    if not gates["target_diagnostic_advantage"]:
        return "STOP_RUFMCD_JRP_P0_TARGET_ADVANTAGE_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_RUFMCD_JRP_P0_EFFECT_TOO_WEAK"
    return "PASS_RUFMCD_JRP_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = base.resolve_repo_path(args.protocol)
    base.context = context
    base.build_fit_geometry = build_fit_geometry
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
