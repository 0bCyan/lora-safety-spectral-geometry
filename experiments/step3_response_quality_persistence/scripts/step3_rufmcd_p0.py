"""Run the robust unit-facet margin MCD zero-step experiment."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

import step3_nptmcd_affine_common as affine
import step3_nptmcd_p0 as base
import step3_ojmcd_common as optimizer_geometry
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0/protocol/rufmcd_p0_protocol.json"
)
FIXED_THETA = base.THETA_MAX - 1e-6
METHOD_NAME = "RUF-MCD"
PROTECTED_FACET_COUNT = base.FIT_UNITS * len(base.PROTECTED_CERTIFICATES)
TARGET_FACET_COUNT = base.FIT_UNITS * len(base.TARGET_CERTIFICATES)
FACET_COUNT = PROTECTED_FACET_COUNT + TARGET_FACET_COUNT
SOLVER_VARIANT = "analytic_active_set_kkt_refinement_from_slsqp_support_seed"
_BASE_CONTEXT = base.context
_BASE_EVALUATE_GATES = base.evaluate_gates
_BASE_SELECT_ROUTE = base.select_route


def context(
    path: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol, config, targets, risk_rows = _BASE_CONTEXT(path)
    design = protocol.get("design", {})
    if (
        protocol.get("method_variant") != METHOD_NAME
        or protocol.get("predecessor", {}).get("route")
        != "STOP_RMAMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE"
        or design.get("target_proposal")
        != "unit_facet_joint_protected_target_margin_affine_max_min"
        or design.get("projection_rule")
        != "shared_float64_gram_pseudoinverse_projection_of_84_normalized_unit_facets_off_raw_displacement_only"
        or int(design.get("target_covectors", -1)) != FACET_COUNT
        or int(design.get("protected_constraints", -1))
        != PROTECTED_FACET_COUNT
        or int(design.get("fit_protected_facet_count", -1))
        != PROTECTED_FACET_COUNT
        or int(design.get("fit_target_facet_count", -1)) != TARGET_FACET_COUNT
        or design.get("unit_level_protected_nullspace_claimed") is not False
        or float(design.get("minimum_retained_unit_facet_fraction", -1.0))
        != 0.01
        or int(config.get("rufmcd_facet_count", -1)) != FACET_COUNT
        or float(config.get("rufmcd_minimum_retained_unit_facet_fraction", -1.0))
        != 0.01
        or config.get("rufmcd_unit_level_protected_nullspace_claimed") is not False
        or int(config.get("rufmcd_constraint_generation_max_rounds", -1))
        != int(design.get("constraint_generation_max_rounds", -2))
        or float(config.get("rufmcd_constraint_generation_full_gap_max", -1.0))
        != float(design.get("constraint_generation_full_gap_max", -2.0))
        or float(design.get("affine_actual_minus_predicted_max", -1.0)) != 1e-6
        or design.get("affine_solver") != SOLVER_VARIANT
        or config.get("rufmcd_affine_solver_variant") != SOLVER_VARIANT
    ):
        raise base.NptmcdP0ValidationError(
            "RUF-MCD protocol registry differs from the runner"
        )
    return protocol, config, targets, risk_rows


def _normalized_gradient(
    gradient: dict[str, Any], label: str
) -> tuple[dict[str, Any], float]:
    norm = gradient_math.gradient_norm(gradient)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"degenerate gradient: {label}")
    return gradient_ops.scaled_gradient(gradient, 1.0 / norm), norm


def _facet_specs(
    fit_gradients: list[dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    if len(fit_gradients) != base.FIT_UNITS:
        raise ValueError("RUF-MCD requires the frozen fit-unit count")
    specs: list[dict[str, Any]] = []
    for index, fit in enumerate(fit_gradients, 1):
        for certificate in base.PROTECTED_CERTIFICATES:
            normalized, norm = _normalized_gradient(
                fit[certificate], f"fit_{index:02d}:{certificate}"
            )
            specs.append(
                {
                    "name": f"fit_{index:02d}:{certificate}",
                    "kind": "protected",
                    "unit_index": index - 1,
                    "certificate": certificate,
                    "gradient": normalized,
                    "input_norm": norm,
                }
            )
        for certificate in base.TARGET_CERTIFICATES:
            normalized, norm = _normalized_gradient(
                fit[certificate], f"fit_{index:02d}:{certificate}"
            )
            specs.append(
                {
                    "name": f"fit_{index:02d}:{certificate}",
                    "kind": "target",
                    "unit_index": index - 1,
                    "certificate": certificate,
                    "gradient": normalized,
                    "input_norm": norm,
                }
            )
    if len(specs) != FACET_COUNT:
        raise ValueError("RUF-MCD facet registry count differs")
    return specs


def _margin_search(
    raw_panels: list[dict[str, float]],
    candidate_panels: list[dict[str, float]],
    solver_audit: dict[str, Any],
    projection_valid: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    protected = tuple(base.PROTECTED_CERTIFICATES)
    targets = tuple(base.TARGET_CERTIFICATES)
    if len(raw_panels) != len(candidate_panels) or len(raw_panels) != base.FIT_UNITS:
        raise ValueError("RUF-MCD fit score panels differ")
    if any(
        float(panel[name]) <= 0.0
        for panel in raw_panels
        for name in protected
    ):
        raise ValueError("RUF-MCD fit protected raw score is nonpositive")
    raw_target_floor = min(
        float(panel[name]) for panel in raw_panels for name in targets
    )
    required_target_floor = max(
        base.FIT_TARGET_SCORE_FLOOR,
        raw_target_floor + base.FIT_TARGET_GAIN_FLOOR,
    )
    candidate_target_floor = min(
        float(panel[name]) for panel in candidate_panels for name in targets
    )
    target_gain = candidate_target_floor - raw_target_floor
    retentions = {
        name: min(
            float(candidate[name]) / float(raw[name])
            for raw, candidate in zip(raw_panels, candidate_panels, strict=True)
        )
        for name in protected
    }
    candidate_protected_margins = [
        retention - base.PROTECTED_RETENTION_FLOOR
        for retention in retentions.values()
    ]
    candidate_target_margins = [
        float(candidate[name]) - required_target_floor
        for candidate in candidate_panels
        for name in targets
    ]
    actual_minimum_margin = min(
        *candidate_protected_margins,
        *candidate_target_margins,
    )
    identity_errors = [
        abs(
            float(candidate[name]) / float(raw[name])
            - math.cos(FIXED_THETA)
        )
        for raw, candidate in zip(raw_panels, candidate_panels, strict=True)
        for name in protected
    ]
    gates = protocol["gates"]
    numeric_tolerance = float(gates["numeric_tolerance"])
    solver_margin = float(solver_audit.get("solver_margin", -math.inf))
    actual_minus_predicted = (
        actual_minimum_margin - solver_margin
        if math.isfinite(solver_margin)
        else None
    )
    consistency_limit = float(
        protocol["design"]["affine_actual_minus_predicted_max"]
    )
    eligible = (
        projection_valid
        and solver_audit.get("solver_valid") is True
        and FIXED_THETA >= float(gates["minimum_nonzero_angle"])
        and FIXED_THETA <= base.THETA_MAX + numeric_tolerance
        and min(retentions.values())
        >= base.PROTECTED_RETENTION_FLOOR - numeric_tolerance
        and candidate_target_floor
        >= required_target_floor - numeric_tolerance
        and target_gain >= base.FIT_TARGET_GAIN_FLOOR - numeric_tolerance
        and actual_minimum_margin >= -numeric_tolerance
        and solver_margin >= -numeric_tolerance
        and actual_minus_predicted is not None
        and abs(float(actual_minus_predicted)) <= consistency_limit
    )
    return {
        "eligible": eligible,
        "selected_theta": FIXED_THETA if projection_valid else 0.0,
        "selected_theta_degrees": math.degrees(
            FIXED_THETA if projection_valid else 0.0
        ),
        "theta_max": base.THETA_MAX,
        "theta_max_degrees": math.degrees(base.THETA_MAX),
        "design_cosine_floor": base.DESIGN_COSINE_FLOOR,
        "protected_retention_floor": base.PROTECTED_RETENTION_FLOOR,
        "raw_target_score_floor": raw_target_floor,
        "required_target_score_floor": required_target_floor,
        "selected_target_score_floor": candidate_target_floor,
        "selected_target_gain_over_raw": target_gain,
        "protected_retentions": retentions,
        "minimum_protected_retention": min(retentions.values()),
        "maximum_protected_retention_identity_error": max(identity_errors),
        "candidate_protected_margin_floor": min(candidate_protected_margins),
        "candidate_target_margin_floor": min(candidate_target_margins),
        "candidate_actual_minimum_margin": actual_minimum_margin,
        "fit_protected_facet_count": PROTECTED_FACET_COUNT,
        "fit_target_facet_count": TARGET_FACET_COUNT,
        "candidate_panels": candidate_panels,
        "target_coefficients": solver_audit.get("objective_names", []),
        "trigonometric_boundaries": [0.0, FIXED_THETA, base.THETA_MAX],
        "selection_rule": "fixed_theta_joint_unit_facet_protected_target_margin_affine_max_min",
        "calibration_scores_used_for_selection": False,
        "diagnostic_scores_used_for_selection": False,
        "fixed_theta": FIXED_THETA,
        "fixed_theta_margin_radians": 1e-6,
        "joint_margin_primal_value": solver_audit.get("solver", {}).get(
            "primal_value"
        ),
        "solver_predicted_minimum_score": solver_audit.get("solver_margin"),
        "solver_candidate_actual_minimum_score": solver_audit.get(
            "actual_minimum_margin"
        ),
        "solver_candidate_actual_minus_predicted": actual_minus_predicted,
        "affine_actual_minus_predicted_max": consistency_limit,
    }


def _protected_transfer_audit(
    raw_displacement: dict[str, Any],
    tangent: dict[str, Any],
    fit_gradients: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    tangent_norm = gradient_math.gradient_norm(tangent)
    protected_dots = [
        gradient_math.gradient_dot(fit[name], tangent)
        / gradient_math.gradient_norm(fit[name])
        for fit in fit_gradients
        for name in base.PROTECTED_CERTIFICATES
    ]
    raw_dot = gradient_math.gradient_dot(raw_displacement, tangent)
    return {
        "raw_displacement_norm": raw_norm,
        "tangent_norm": tangent_norm,
        "tangent_to_raw_norm_ratio": tangent_norm / raw_norm,
        "raw_tangent_cosine": raw_dot / (raw_norm * tangent_norm),
        "normalized_protected_dots": protected_dots,
        "maximum_absolute_normalized_protected_dot": max(
            abs(value) for value in protected_dots
        ),
        "maximum_absolute_protected_cosine": max(abs(value) for value in protected_dots)
        / tangent_norm,
    }


def _analytic_affine_minimax_direction(
    gradients: dict[str, dict[str, Any]],
    offsets: dict[str, float],
    budget_norm: float,
    tolerance: float,
    kkt_limit: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine an affine simplex solution by solving its support KKT equations.

    For a dual support A, write ``x = ||G lambda|| / B``.  The active KKT
    equations imply ``lambda_A = p + x q``; the norm identity then becomes a
    scalar quadratic in x.  Support additions/deletions enforce the remaining
    simplex inequalities.  SLSQP is used only to propose support seeds; the
    returned certificate is recomputed from the analytic KKT solution.
    """
    if (
        len(gradients) < 2
        or set(gradients) != set(offsets)
        or not math.isfinite(budget_norm)
        or budget_norm <= 0.0
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
        or not math.isfinite(kkt_limit)
        or kkt_limit <= 0.0
    ):
        raise ValueError("RUF-MCD analytic affine inputs are invalid")
    names = list(gradients)
    tensors = [gradients[name] for name in names]
    if not tensors[0] or any(set(item) != set(tensors[0]) for item in tensors[1:]):
        raise ValueError("RUF-MCD analytic affine gradient dictionaries differ")
    norms = [gradient_math.gradient_norm(item) for item in tensors]
    if any(not math.isfinite(value) or value <= 0.0 for value in norms):
        raise ValueError("RUF-MCD analytic affine gradients are invalid")
    gram = np.asarray(
        [
            [gradient_math.gradient_dot(left, right) for right in tensors]
            for left in tensors
        ],
        dtype=np.float64,
    )
    gram = 0.5 * (gram + gram.T)
    offset_values = np.asarray([offsets[name] for name in names], dtype=np.float64)
    if not np.isfinite(gram).all() or not np.isfinite(offset_values).all():
        raise ValueError("RUF-MCD analytic affine Gram matrix is invalid")

    seed_error = None
    try:
        _, seed = optimizer_geometry.affine_minimax_direction(
            gradients, offsets, budget_norm, tolerance
        )
        seed_weights = np.asarray(
            [float(seed["dual_weights"][name]) for name in names],
            dtype=np.float64,
        )
    except (ValueError, RuntimeError) as exc:
        seed = {}
        seed_weights = np.full(len(names), 1.0 / len(names), dtype=np.float64)
        seed_error = str(exc)

    support_seeds = []
    for threshold in (1e-10, 1e-9, 1e-8, 1e-7):
        support = tuple(
            index for index, weight in enumerate(seed_weights) if weight > threshold
        )
        if support and support not in support_seeds:
            support_seeds.append(support)
    all_support = tuple(range(len(names)))
    if all_support not in support_seeds:
        support_seeds.append(all_support)

    support_tolerance = 1e-10
    weight_tolerance = 1e-8
    candidates: list[dict[str, Any]] = []

    def evaluate(
        weights: np.ndarray, support: list[int], root: float, iterations: int
    ) -> dict[str, Any]:
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("RUF-MCD analytic simplex weights are invalid")
        weights = np.maximum(weights, 0.0)
        weights /= float(weights.sum())
        weighted_norm = math.sqrt(max(0.0, float(weights @ gram @ weights)))
        if not math.isfinite(weighted_norm) or weighted_norm <= tolerance:
            raise ValueError("RUF-MCD analytic affine dual has no direction")
        scores = offset_values + budget_norm * (gram @ weights) / weighted_norm
        dual_value = float(offset_values @ weights) + budget_norm * weighted_norm
        primal_value = float(np.min(scores))
        active = [index for index, value in enumerate(weights) if value > weight_tolerance]
        inactive = [index for index in range(len(names)) if index not in active]
        stationarity = max(
            [abs(float(scores[index]) - dual_value) for index in active] + [0.0]
        )
        inequality = max(
            [max(0.0, dual_value - float(scores[index])) for index in inactive]
            + [0.0]
        )
        kkt_residual = max(
            stationarity,
            inequality,
            abs(float(weights.sum()) - 1.0),
            max(0.0, -float(weights.min())),
        )
        return {
            "weights": weights,
            "scores": scores,
            "dual_value": dual_value,
            "primal_value": primal_value,
            "primal_dual_gap": abs(primal_value - dual_value),
            "kkt_residual": kkt_residual,
            "support": list(support),
            "root": root,
            "support_iterations": iterations,
            "weighted_gradient_norm": weighted_norm,
        }

    def refine(initial_support: tuple[int, ...]) -> dict[str, Any] | None:
        support = sorted(set(initial_support))
        for iteration in range(len(names) + 2):
            if not support:
                return None
            indices = np.asarray(support, dtype=np.int64)
            block = gram[np.ix_(indices, indices)]
            block_offsets = offset_values[indices]
            inverse = np.linalg.pinv(block, rcond=1e-12)
            ones = np.ones(len(indices), dtype=np.float64)
            inverse_one = inverse @ ones
            inverse_offset = inverse @ block_offsets
            denominator = float(ones @ inverse_one)
            if not math.isfinite(denominator) or denominator <= 1e-14:
                return None
            coefficient = float(ones @ inverse_offset) / denominator
            p = inverse_one / denominator
            q = coefficient * inverse_one - inverse_offset
            constant = float(p @ block @ p)
            linear = float(2.0 * p @ block @ q)
            quadratic = float(q @ block @ q) - budget_norm**2
            coefficient_scale = max(1.0, abs(constant), abs(linear), abs(quadratic))
            if abs(quadratic) <= 1e-14 * coefficient_scale:
                roots = (
                    (-constant / linear,)
                    if abs(linear) > 1e-14 * coefficient_scale
                    else ()
                )
            else:
                roots = tuple(np.roots([quadratic, linear, constant]))
            root_candidates = []
            for raw_root in roots:
                if abs(float(np.imag(raw_root))) > 1e-7:
                    continue
                root = float(np.real(raw_root))
                if not math.isfinite(root) or root <= 0.0:
                    continue
                active_weights = p + root * q
                weights = np.zeros(len(names), dtype=np.float64)
                weights[indices] = active_weights
                try:
                    root_candidates.append(
                        evaluate(weights, support, root, iteration + 1)
                    )
                except ValueError:
                    continue
            if not root_candidates:
                return None
            feasible = [
                item
                for item in root_candidates
                if min(item["weights"][indices]) >= -support_tolerance
            ]
            selected = min(
                feasible or root_candidates,
                key=lambda item: (
                    max(0.0, -float(item["weights"][indices].min())),
                    item["kkt_residual"],
                    item["primal_dual_gap"],
                ),
            )
            if min(selected["weights"][indices]) < -support_tolerance:
                drop = int(indices[np.argmin(selected["weights"][indices])])
                support.remove(drop)
                continue
            weights = selected["weights"]
            violations = [
                index
                for index in range(len(names))
                if index not in support
                and float(selected["scores"][index])
                < selected["dual_value"] - support_tolerance
            ]
            if violations:
                support.append(
                    min(
                        violations,
                        key=lambda index: (float(selected["scores"][index]), index),
                    )
                )
                support = sorted(set(support))
                continue
            return selected
        return None

    for support in support_seeds:
        candidate = refine(support)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            "RUF-MCD analytic affine support refinement found no candidate"
        )
    certified = [
        item
        for item in candidates
        if item["kkt_residual"] <= kkt_limit
        and item["primal_dual_gap"] <= kkt_limit
    ]
    selected = min(
        certified or candidates,
        key=lambda item: (
            item["dual_value"],
            item["kkt_residual"],
            item["primal_dual_gap"],
        ),
    )
    weights = selected["weights"]
    weighted = gradient_ops.scaled_gradient(tensors[0], float(weights[0]))
    for tensor, weight in zip(tensors[1:], weights[1:], strict=True):
        weighted = gradient_ops.add_gradients(
            weighted, tensor, right_scale=float(weight)
        )
    direction = gradient_ops.scaled_gradient(
        weighted, budget_norm / selected["weighted_gradient_norm"]
    )
    audit = {
        "certificate_names": names,
        "gradient_norms": dict(zip(names, norms, strict=True)),
        "offsets": dict(offsets),
        "gram_matrix": gram.tolist(),
        "dual_weights": dict(zip(names, weights.tolist(), strict=True)),
        "active_certificates": [
            names[index]
            for index, weight in enumerate(weights)
            if weight > weight_tolerance
        ],
        "active_set_size": int(sum(weight > weight_tolerance for weight in weights)),
        "dual_value": selected["dual_value"],
        "primal_value": selected["primal_value"],
        "primal_dual_gap": selected["primal_dual_gap"],
        "kkt_residual": selected["kkt_residual"],
        "certificate_scores": dict(
            zip(names, selected["scores"].tolist(), strict=True)
        ),
        "weighted_gradient_norm": selected["weighted_gradient_norm"],
        "budget_norm": budget_norm,
        "final_direction_norm": gradient_math.gradient_norm(direction),
        "solver": SOLVER_VARIANT,
        "solver_success": bool(certified),
        "solver_iterations": selected["support_iterations"],
        "feasible_solver_candidates": len(candidates),
        "analytic_support_size": len(selected["support"]),
        "analytic_support": [names[index] for index in selected["support"]],
        "analytic_root": selected["root"],
        "seed_solver": seed.get("solver"),
        "seed_solver_success": seed.get("solver_success"),
        "seed_kkt_residual": seed.get("kkt_residual"),
        "seed_primal_dual_gap": seed.get("primal_dual_gap"),
        "seed_error": seed_error,
    }
    return direction, audit


def _constraint_generation_affine_minimax(
    gradients: dict[str, dict[str, Any]],
    offsets: dict[str, float],
    groups: dict[str, str],
    budget_norm: float,
    solver_tolerance: float,
    kkt_limit: float,
    full_gap_limit: float,
    max_rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        set(gradients) != set(offsets)
        or set(gradients) != set(groups)
        or len(gradients) < 2
        or not math.isfinite(full_gap_limit)
        or full_gap_limit <= 0.0
        or max_rounds <= 0
    ):
        raise ValueError("RUF-MCD constraint-generation inputs differ")
    names = list(gradients)
    group_names = list(dict.fromkeys(groups.values()))
    active = {
        min(
            (name for name in names if groups[name] == group),
            key=lambda name: (float(offsets[name]), name),
        )
        for group in group_names
    }
    if len(active) < 2:
        raise ValueError("RUF-MCD initial active set is degenerate")

    rounds = []
    for round_index in range(1, max_rounds + 1):
        active_names = [name for name in names if name in active]
        direction, solver = _analytic_affine_minimax_direction(
            {name: gradients[name] for name in active_names},
            {name: offsets[name] for name in active_names},
            budget_norm,
            solver_tolerance,
            kkt_limit,
        )
        if (
            not bool(solver["solver_success"])
            or float(solver["kkt_residual"]) > kkt_limit
            or float(solver["primal_dual_gap"]) > kkt_limit
        ):
            raise ValueError(
                "RUF-MCD active affine solver exceeded the frozen KKT limit: "
                f"round={round_index}, active={len(active_names)}, "
                f"solver_success={solver['solver_success']}, "
                f"kkt={solver['kkt_residual']:.3e}, "
                f"gap={solver['primal_dual_gap']:.3e}"
            )
        full_scores = {
            name: float(offsets[name])
            + gradient_math.gradient_dot(gradients[name], direction)
            for name in names
        }
        full_primal = min(full_scores.values())
        dual_upper = float(solver["dual_value"])
        signed_gap = dual_upper - full_primal
        if signed_gap < -full_gap_limit:
            raise ValueError("RUF-MCD full dual bound is below the full primal")
        full_gap = max(0.0, signed_gap)
        worst_name = min(names, key=lambda name: (full_scores[name], name))
        rounds.append(
            {
                "round": round_index,
                "active_facets": len(active_names),
                "active_primal_value": float(solver["primal_value"]),
                "active_dual_value": dual_upper,
                "active_primal_dual_gap": float(solver["primal_dual_gap"]),
                "active_kkt_residual": float(solver["kkt_residual"]),
                "full_primal_value": full_primal,
                "full_primal_dual_gap": full_gap,
                "worst_full_facet": worst_name,
            }
        )
        if full_gap <= full_gap_limit:
            audit = dict(solver)
            audit.update(
                {
                    "full_certificate_scores": full_scores,
                    "full_primal_value": full_primal,
                    "full_dual_upper_bound": dual_upper,
                    "full_primal_dual_gap": full_gap,
                    "constraint_generation_rounds": rounds,
                    "constraint_generation_round_count": round_index,
                    "constraint_generation_active_facets": active_names,
                    "constraint_generation_active_facet_count": len(active_names),
                    "constraint_generation_total_facet_count": len(names),
                    "constraint_generation_full_gap_limit": full_gap_limit,
                    "constraint_generation_max_rounds": max_rounds,
                    "constraint_generation_certified": True,
                }
            )
            return direction, audit

        additions = []
        for group in group_names:
            inactive = [
                name
                for name in names
                if groups[name] == group and name not in active
            ]
            if not inactive:
                continue
            worst_group = min(
                inactive, key=lambda name: (full_scores[name], name)
            )
            if full_scores[worst_group] < dual_upper - full_gap_limit:
                additions.append(worst_group)
        if not additions:
            inactive = [name for name in names if name not in active]
            if not inactive:
                raise ValueError(
                    "RUF-MCD full active set did not close the dual gap"
                )
            additions.append(
                min(inactive, key=lambda name: (full_scores[name], name))
            )
        active.update(additions)

    raise ValueError("RUF-MCD constraint generation exceeded the frozen round limit")


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
    panels = [pooled, *fit_gradients]
    raw_score_panels = [
        base.frsc.displacement_scores(panel, raw_displacement, reference_norm)[
            "scores"
        ]
        for panel in panels
    ]
    specs = _facet_specs(fit_gradients)
    raw_target_floor = min(
        float(panel[name])
        for panel in raw_score_panels[1:]
        for name in base.TARGET_CERTIFICATES
    )
    required_target_floor = max(
        base.FIT_TARGET_SCORE_FLOOR,
        raw_target_floor + base.FIT_TARGET_GAIN_FLOOR,
    )
    normalized_facets = [spec["gradient"] for spec in specs]
    tangent = gradient_ops.scaled_gradient(raw_displacement, 0.0)
    proposed_candidate = {
        name: value.clone() for name, value in raw_displacement.items()
    }
    solver_audit: dict[str, Any] = {"solver_valid": False}
    projection_valid = False
    projection_error = None
    try:
        projected, projection = affine.project_registry_to_constraint_nullspace(
            normalized_facets,
            [raw_displacement],
            float(protocol["design"]["projection_rcond"]),
            int(protocol["design"]["projection_refinement_steps"]),
        )
        objective_names = []
        objective_gradients = {}
        offsets = {}
        facet_groups = {}
        for spec, projected_gradient in zip(specs, projected, strict=True):
            name = spec["name"]
            if spec["kind"] == "protected":
                raw_score = float(
                    raw_score_panels[spec["unit_index"] + 1][spec["certificate"]]
                )
                objective_name = f"protected:{name}"
                objective_names.append(objective_name)
                facet_groups[objective_name] = spec["certificate"]
                objective_gradients[objective_name] = gradient_ops.scaled_gradient(
                    projected_gradient,
                    math.sin(FIXED_THETA) / (reference_norm * raw_score),
                )
                offsets[objective_name] = (
                    math.cos(FIXED_THETA) - base.PROTECTED_RETENTION_FLOOR
                )
            else:
                raw_score = float(
                    raw_score_panels[spec["unit_index"] + 1][spec["certificate"]]
                )
                objective_name = f"target:{name}"
                objective_names.append(objective_name)
                facet_groups[objective_name] = spec["certificate"]
                objective_gradients[objective_name] = gradient_ops.scaled_gradient(
                    projected_gradient, math.sin(FIXED_THETA) / reference_norm
                )
                offsets[objective_name] = (
                    raw_score * math.cos(FIXED_THETA) - required_target_floor
                )
        if len(objective_names) != FACET_COUNT:
            raise ValueError("RUF-MCD objective facet count differs")
        kkt_limit = float(protocol["design"]["affine_kkt_residual_max"])
        tangent, solver = _constraint_generation_affine_minimax(
            objective_gradients,
            offsets,
            facet_groups,
            gradient_math.gradient_norm(raw_displacement),
            float(protocol["design"]["affine_solver_tolerance"]),
            kkt_limit,
            float(protocol["design"]["constraint_generation_full_gap_max"]),
            int(protocol["design"]["constraint_generation_max_rounds"]),
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
        candidate_margins = []
        for spec in specs:
            panel = candidate_panel_scores[spec["unit_index"]]
            raw_panel = raw_score_panels[spec["unit_index"] + 1]
            if spec["kind"] == "protected":
                candidate_margins.append(
                    float(panel[spec["certificate"]])
                    / float(raw_panel[spec["certificate"]])
                    - base.PROTECTED_RETENTION_FLOOR
                )
            else:
                candidate_margins.append(
                    float(panel[spec["certificate"]]) - required_target_floor
                )
        actual_margin = min(candidate_margins)
        if (
            not bool(solver["solver_success"])
            or float(solver["kkt_residual"]) > kkt_limit
            or float(solver["primal_dual_gap"]) > kkt_limit
            or float(solver["full_primal_dual_gap"])
            > float(protocol["design"]["constraint_generation_full_gap_max"])
        ):
            raise ValueError("RUF-MCD affine simplex KKT audit exceeded frozen limit")
        projection_valid = True
        solver_audit = {
            "solver_valid": True,
            "projection": projection,
            "solver": solver,
            "objective_names": objective_names,
            "offsets": offsets,
            "solver_margin": float(solver["full_primal_value"]),
            "actual_minimum_margin": actual_margin,
            "actual_minus_predicted": actual_margin
            - float(solver["full_primal_value"]),
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
        base.frsc.displacement_scores(panel, proposed_candidate, reference_norm)[
            "scores"
        ]
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
    protected_constraints = [
        fit[name]
        for fit in fit_gradients
        for name in base.PROTECTED_CERTIFICATES
    ]
    transfer_audit = (
        _protected_transfer_audit(raw_displacement, tangent, fit_gradients)
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
        "protected_constraints": PROTECTED_FACET_COUNT,
        "target_covector_count": FACET_COUNT,
        "minimum_unit_facet_projection_fraction": min(projected_fractions)
        if projected_fractions
        else 0.0,
        "retained_target_proposal_fraction": min(projected_fractions)
        if projected_fractions
        else 0.0,
        "target_covector_projection": projection_audit,
        "affine_solver": solver_audit.get("solver"),
        "facet_count": FACET_COUNT,
        "enforcement": "unit_facet_margin_not_unit_level_nullspace",
        "unit_level_protected_nullspace_claimed": False,
    }
    selected_map = {
        "map_kind": "direct_fixed_norm_protected_nullspace_angular_rotation",
        "map_variant": "unit_facet_joint_protected_target_margin_affine_max_min",
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
        "name": "step3_rufmcd_p0_fit_geometry_v1",
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
        "protected_constraint_count": PROTECTED_FACET_COUNT,
        "target_facet_labels": [spec["name"] for spec in specs if spec["kind"] == "target"],
        "target_facet_count": TARGET_FACET_COUNT,
        "facet_count": FACET_COUNT,
        "target_proposal_norm": gradient_math.gradient_norm(
            gradient_ops.add_gradients(
                target_displacement, raw_displacement, right_scale=-1.0
            )
        ),
        "target_endpoint_difference_used": False,
        "unit_facet_rule": "all_fit_units_cross_all_registered_certificates_with_protected_relative_and_target_absolute_margins",
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
        normalized_facets,
        protected_constraints,
    )
    return fit_geometry, fixed


def _unit_facet_geometry_gate(
    fit_geometry: dict[str, Any], protocol: dict[str, Any]
) -> bool:
    gates = protocol["gates"]
    design = protocol["design"]
    audit = fit_geometry.get("tangent_audit", {})
    projection = audit.get("target_covector_projection") or {}
    solver = audit.get("affine_solver") or {}
    vectors = projection.get("vectors") or []
    full_scores = solver.get("full_certificate_scores") or {}
    full_gap = float(solver.get("full_primal_dual_gap", math.inf))
    full_primal = float(solver.get("full_primal_value", math.inf))
    full_dual = float(solver.get("full_dual_upper_bound", -math.inf))
    full_score_min = min(full_scores.values()) if full_scores else math.inf
    return (
        bool(fit_geometry.get("projection_valid"))
        and audit.get("enforcement") == "unit_facet_margin_not_unit_level_nullspace"
        and int(audit.get("target_covector_count", -1)) == FACET_COUNT
        and int(projection.get("constraints", -1)) == 1
        and len(vectors) == FACET_COUNT
        and len(full_scores) == FACET_COUNT
        and bool(solver.get("constraint_generation_certified"))
        and int(solver.get("constraint_generation_total_facet_count", -1))
        == FACET_COUNT
        and 0 < int(solver.get("constraint_generation_round_count", 0))
        <= int(design["constraint_generation_max_rounds"])
        and full_dual + float(design["constraint_generation_full_gap_max"]) >= full_primal
        and abs(full_score_min - full_primal)
        <= float(design["constraint_generation_full_gap_max"])
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
        and float(solver.get("kkt_residual", math.inf))
        <= float(design["affine_kkt_residual_max"])
        and float(solver.get("primal_dual_gap", math.inf))
        <= float(design["affine_kkt_residual_max"])
        and abs(
            float(fit_geometry["search"].get("solver_candidate_actual_minus_predicted", math.inf))
        )
        <= float(design["affine_actual_minus_predicted_max"])
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
        rows,
        fit_geometry,
        calibration_rows,
        summary,
        protocol,
        state_restored,
        vram_headroom,
        gpu_name,
    )
    search = fit_geometry["search"]
    required_floor = float(search["required_target_score_floor"])
    gates["fit_minimum_effective_angle"] = (
        bool(search.get("eligible"))
        and float(search["selected_theta"])
        >= float(protocol["gates"]["minimum_nonzero_angle"])
        and float(search["selected_theta"])
        <= base.THETA_MAX + float(protocol["gates"]["numeric_tolerance"])
        and float(search["minimum_protected_retention"])
        >= base.PROTECTED_RETENTION_FLOOR
        and float(search["selected_target_score_floor"]) >= required_floor
        and float(search["selected_target_gain_over_raw"])
        >= base.FIT_TARGET_GAIN_FLOOR
        and float(search["candidate_actual_minimum_margin"])
        >= -float(protocol["gates"]["numeric_tolerance"])
    )
    gates["protected_nullspace_geometry"] = _unit_facet_geometry_gate(
        fit_geometry, protocol
    )
    gates["unit_facet_margin_geometry"] = gates["protected_nullspace_geometry"]
    return gates


def select_route(gates: dict[str, bool]) -> str:
    route = _BASE_SELECT_ROUTE(gates)
    if route == "STOP_NPTMCD_P0_NULLSPACE_GEOMETRY_INVALID":
        return "STOP_RUFMCD_P0_UNIT_FACET_GEOMETRY_INVALID"
    return route.replace("NPTMCD", "RUFMCD")


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
