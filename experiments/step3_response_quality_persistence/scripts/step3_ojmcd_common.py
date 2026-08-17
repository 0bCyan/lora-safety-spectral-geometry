"""Realized-update Jacobian geometry for optimizer-aware common descent."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import step3_mcd_common as mcd
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def _validate_pair(
    left: dict[str, Any], right: dict[str, Any], label: str
) -> None:
    if not left or set(left) != set(right):
        raise ValueError(f"{label} gradient dictionaries differ")


def _elementwise_scale(
    gradient: dict[str, Any], scales: dict[str, Any]
) -> dict[str, Any]:
    _validate_pair(gradient, scales, "elementwise scale")
    return {name: gradient[name] * scales[name] for name in gradient}


def _normalized_gradients(
    gradients: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    if len(gradients) < 2:
        raise ValueError("at least two certificate gradients are required")
    names = list(gradients)
    reference = gradients[names[0]]
    if not reference or any(set(item) != set(reference) for item in gradients.values()):
        raise ValueError("certificate gradient dictionaries differ")
    norms = {
        name: gradient_math.gradient_norm(gradient)
        for name, gradient in gradients.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise ValueError("certificate gradients must be finite and nonzero")
    return {
        name: gradient_ops.scaled_gradient(gradient, 1.0 / norms[name])
        for name, gradient in gradients.items()
    }, norms


def _weighted_gradient(
    gradients: list[dict[str, Any]], weights: list[float]
) -> dict[str, Any]:
    if not gradients or len(gradients) != len(weights):
        raise ValueError("weighted gradient inputs differ")
    result = gradient_ops.scaled_gradient(gradients[0], weights[0])
    for gradient, weight in zip(gradients[1:], weights[1:], strict=True):
        result = gradient_ops.add_gradients(result, gradient, right_scale=weight)
    return result


def _adam_slope(clipped: dict[str, Any], epsilon: float) -> dict[str, Any]:
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Adam epsilon must be finite and positive")
    return {
        name: epsilon / (value.abs() + epsilon).square()
        for name, value in clipped.items()
    }


def clip_gradient(
    gradient: dict[str, Any], max_norm: float, clip_epsilon: float = 1e-6
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    """Match torch clip_grad_norm_ scaling for a finite nonzero gradient."""
    norm = gradient_math.gradient_norm(gradient)
    if (
        not math.isfinite(norm)
        or norm <= 0.0
        or not math.isfinite(max_norm)
        or max_norm <= 0.0
        or not math.isfinite(clip_epsilon)
        or clip_epsilon < 0.0
    ):
        raise ValueError("clipping inputs are invalid")
    coefficient = min(1.0, max_norm / (norm + clip_epsilon))
    clipped = gradient_ops.scaled_gradient(gradient, coefficient)
    return clipped, {
        "raw_norm": norm,
        "clip_coefficient": coefficient,
        "clipping_active": coefficient < 1.0,
        "clipped_norm": gradient_math.gradient_norm(clipped),
    }


def clipped_adam_first_displacement(
    raw_gradient: dict[str, Any],
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    """Return the positive first-step Adam displacement without learning rate."""
    clipped, audit = clip_gradient(raw_gradient, max_norm, clip_epsilon)
    if not math.isfinite(adam_epsilon) or adam_epsilon <= 0.0:
        raise ValueError("Adam epsilon must be finite and positive")
    displacement = {
        name: value / (value.abs() + adam_epsilon)
        for name, value in clipped.items()
    }
    result = dict(audit)
    result["displacement_norm"] = gradient_math.gradient_norm(displacement)
    return displacement, result


def _clip_jvp(
    raw_gradient: dict[str, Any],
    tangent: dict[str, Any],
    max_norm: float,
    clip_epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float | bool]]:
    _validate_pair(raw_gradient, tangent, "clip JVP")
    clipped, audit = clip_gradient(raw_gradient, max_norm, clip_epsilon)
    if not audit["clipping_active"]:
        return {name: value.clone() for name, value in tangent.items()}, clipped, audit
    norm = float(audit["raw_norm"])
    coefficient = float(audit["clip_coefficient"])
    radial_dot = gradient_math.gradient_dot(raw_gradient, tangent)
    radial_scale = max_norm * radial_dot / (norm * (norm + clip_epsilon) ** 2)
    jvp = gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(tangent, coefficient),
        raw_gradient,
        right_scale=-radial_scale,
    )
    return jvp, clipped, audit


def clipped_adam_first_jvp(
    raw_gradient: dict[str, Any],
    tangent: dict[str, Any],
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    """Apply the Jacobian of clipping followed by first-step Adam."""
    clip_jvp, clipped, audit = _clip_jvp(
        raw_gradient, tangent, max_norm, clip_epsilon
    )
    result = _elementwise_scale(clip_jvp, _adam_slope(clipped, adam_epsilon))
    record = dict(audit)
    record["jvp_norm"] = gradient_math.gradient_norm(result)
    return result, record


def clipped_adam_first_vjp(
    raw_gradient: dict[str, Any],
    cotangent: dict[str, Any],
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    """Apply the transpose Jacobian to a parameter-space certificate."""
    _validate_pair(raw_gradient, cotangent, "Adam VJP")
    clipped, audit = clip_gradient(raw_gradient, max_norm, clip_epsilon)
    weighted = _elementwise_scale(cotangent, _adam_slope(clipped, adam_epsilon))
    if not audit["clipping_active"]:
        result = weighted
    else:
        norm = float(audit["raw_norm"])
        coefficient = float(audit["clip_coefficient"])
        radial_dot = gradient_math.gradient_dot(raw_gradient, weighted)
        radial_scale = max_norm * radial_dot / (norm * (norm + clip_epsilon) ** 2)
        result = gradient_ops.add_gradients(
            gradient_ops.scaled_gradient(weighted, coefficient),
            raw_gradient,
            right_scale=-radial_scale,
        )
    record = dict(audit)
    record["vjp_norm"] = gradient_math.gradient_norm(result)
    return result, record


def affine_minimax_direction(
    gradients: dict[str, dict[str, Any]],
    offsets: dict[str, float],
    budget_norm: float,
    tolerance: float = 1e-9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve max min_i(offset_i + <gradient_i, d>) for ||d|| <= B.

    The simplex dual is

        min_lambda sum_i lambda_i offset_i
                   + B ||sum_i lambda_i gradient_i||.

    The problem is only seven-dimensional in OJ-MCD. SLSQP is run from the
    simplex center and every vertex; the returned solution is accepted only
    with an explicit primal-dual and simplex KKT audit.
    """
    if (
        len(gradients) < 2
        or set(gradients) != set(offsets)
        or not math.isfinite(budget_norm)
        or budget_norm <= 0.0
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("affine minimax inputs are invalid")
    names = list(gradients)
    tensors = [gradients[name] for name in names]
    if not tensors[0] or any(set(item) != set(tensors[0]) for item in tensors[1:]):
        raise ValueError("affine minimax gradient dictionaries differ")
    norms = [gradient_math.gradient_norm(item) for item in tensors]
    offset_values = np.asarray([offsets[name] for name in names], dtype=np.float64)
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in norms)
        or not np.isfinite(offset_values).all()
    ):
        raise ValueError("affine minimax gradients or offsets are invalid")
    gram = np.asarray(
        [
            [gradient_math.gradient_dot(left, right) for right in tensors]
            for left in tensors
        ],
        dtype=np.float64,
    )
    gram = 0.5 * (gram + gram.T)
    if not np.isfinite(gram).all():
        raise ValueError("affine minimax Gram matrix is invalid")

    from scipy.optimize import minimize

    scale = max(
        1.0,
        float(np.max(np.abs(offset_values))),
        budget_norm * math.sqrt(max(0.0, float(np.max(np.diag(gram))))),
    )

    def objective(weights: np.ndarray) -> float:
        squared = max(0.0, float(weights @ gram @ weights))
        return float(offset_values @ weights + budget_norm * math.sqrt(squared)) / scale

    def jacobian(weights: np.ndarray) -> np.ndarray:
        product = gram @ weights
        norm = math.sqrt(max(0.0, float(weights @ product)))
        if norm <= tolerance:
            return offset_values / scale
        return (offset_values + budget_norm * product / norm) / scale

    size = len(names)
    starts = [np.full(size, 1.0 / size, dtype=np.float64)]
    starts.extend(np.eye(size, dtype=np.float64))
    candidates: list[tuple[float, np.ndarray, bool, int]] = []
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=jacobian,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * size,
            constraints={
                "type": "eq",
                "fun": lambda value: float(np.sum(value) - 1.0),
                "jac": lambda value: np.ones_like(value),
            },
            options={"ftol": 1e-13, "maxiter": 2000, "disp": False},
        )
        weights = np.asarray(result.x, dtype=np.float64)
        if (
            np.isfinite(weights).all()
            and float(np.min(weights)) >= -1e-7
            and abs(float(np.sum(weights)) - 1.0) <= 1e-7
        ):
            weights = np.maximum(weights, 0.0)
            weights /= float(np.sum(weights))
            candidates.append(
                (objective(weights), weights, bool(result.success), int(result.nit))
            )
    if not candidates:
        raise ValueError("affine minimax dual solver found no feasible point")
    _, weights_array, solver_success, iterations = min(
        candidates, key=lambda item: item[0]
    )
    weights = weights_array.tolist()
    weighted = _weighted_gradient(tensors, weights)
    weighted_norm = gradient_math.gradient_norm(weighted)
    if not math.isfinite(weighted_norm) or weighted_norm <= tolerance:
        raise ValueError("affine minimax dual has no controllable direction")
    direction = gradient_ops.scaled_gradient(
        weighted, budget_norm / weighted_norm
    )
    scores = {
        name: float(offsets[name])
        + gradient_math.gradient_dot(gradients[name], direction)
        for name in names
    }
    dual_value = float(offset_values @ weights_array) + budget_norm * weighted_norm
    primal_value = min(scores.values())
    active_threshold = max(1e-7, 100.0 * tolerance)
    active = [index for index, value in enumerate(weights) if value > active_threshold]
    inactive = [index for index in range(size) if index not in active]
    stationarity = max(
        [abs(scores[names[index]] - dual_value) for index in active] + [0.0]
    )
    inequality = max(
        [max(0.0, dual_value - scores[names[index]]) for index in inactive]
        + [0.0]
    )
    kkt_residual = max(
        stationarity,
        inequality,
        abs(sum(weights) - 1.0),
        max(0.0, -min(weights)),
    )
    return direction, {
        "certificate_names": names,
        "gradient_norms": dict(zip(names, norms, strict=True)),
        "offsets": dict(offsets),
        "gram_matrix": gram.tolist(),
        "dual_weights": dict(zip(names, weights, strict=True)),
        "active_certificates": [names[index] for index in active],
        "active_set_size": len(active),
        "dual_value": dual_value,
        "primal_value": primal_value,
        "primal_dual_gap": abs(primal_value - dual_value),
        "kkt_residual": kkt_residual,
        "certificate_scores": scores,
        "weighted_gradient_norm": weighted_norm,
        "budget_norm": budget_norm,
        "final_direction_norm": gradient_math.gradient_norm(direction),
        "solver": "scipy_SLSQP_multistart_with_explicit_KKT_audit",
        "solver_success": solver_success,
        "solver_iterations": iterations,
        "feasible_solver_candidates": len(candidates),
    }


def baseline_aware_optimizer_jacobian_direction(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    budget_norm: float,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optimize worst predicted full-update descent in optimizer geometry."""
    if not math.isfinite(auxiliary_scale) or auxiliary_scale <= 0.0:
        raise ValueError("auxiliary scale must be finite and positive")
    units, certificate_norms = _normalized_gradients(certificates)
    baseline_update, baseline_audit = clipped_adam_first_displacement(
        raw_baseline, max_norm, adam_epsilon, clip_epsilon
    )
    baseline_norm = gradient_math.gradient_norm(baseline_update)
    if not math.isfinite(baseline_norm) or baseline_norm <= 0.0:
        raise ValueError("baseline realized update is degenerate")
    offsets = {
        name: gradient_math.gradient_dot(unit, baseline_update) / baseline_norm
        for name, unit in units.items()
    }
    transformed: dict[str, dict[str, Any]] = {}
    vjp_audits: dict[str, dict[str, float | bool]] = {}
    for name, unit in units.items():
        current, audit = clipped_adam_first_vjp(
            raw_baseline,
            unit,
            max_norm,
            adam_epsilon,
            clip_epsilon,
        )
        transformed[name] = gradient_ops.scaled_gradient(
            current, auxiliary_scale / baseline_norm
        )
        vjp_audits[name] = audit
    direction, dual = affine_minimax_direction(
        transformed, offsets, budget_norm, dual_tolerance
    )
    first_audit = next(iter(vjp_audits.values()))
    return direction, {
        "certificate_norms": certificate_norms,
        "baseline_update_norm": baseline_norm,
        "baseline_certificate_scores": offsets,
        "baseline_minimum_score": min(offsets.values()),
        "predicted_minimum_score": float(dual["primal_value"]),
        "predicted_minimum_score_improvement": float(dual["primal_value"])
        - min(offsets.values()),
        "transformed_sensitivity_norms": {
            name: gradient_math.gradient_norm(value)
            for name, value in transformed.items()
        },
        "baseline_raw_norm": float(first_audit["raw_norm"]),
        "baseline_clip_coefficient": float(first_audit["clip_coefficient"]),
        "baseline_clipping_active": bool(first_audit["clipping_active"]),
        "auxiliary_scale": auxiliary_scale,
        **dual,
    }


def optimizer_jacobian_common_direction(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    budget_norm: float,
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve MCD over J^T g_i in raw-gradient control space."""
    transformed: dict[str, dict[str, Any]] = {}
    vjp_audits: dict[str, dict[str, float | bool]] = {}
    for name, certificate in certificates.items():
        transformed[name], vjp_audits[name] = clipped_adam_first_vjp(
            raw_baseline,
            certificate,
            max_norm,
            adam_epsilon,
            clip_epsilon,
        )
    direction, dual = mcd.minimax_common_direction(
        transformed, budget_norm, dual_tolerance
    )
    return direction, {
        "transformed_gradient_norms": {
            name: float(audit["vjp_norm"]) for name, audit in vjp_audits.items()
        },
        "baseline_raw_norm": float(next(iter(vjp_audits.values()))["raw_norm"]),
        "baseline_clip_coefficient": float(
            next(iter(vjp_audits.values()))["clip_coefficient"]
        ),
        "baseline_clipping_active": bool(
            next(iter(vjp_audits.values()))["clipping_active"]
        ),
        **dual,
    }


def realized_increment_metrics(
    raw_baseline: dict[str, Any],
    raw_auxiliary: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Compare linearized and exact first-step Adam displacement changes."""
    if not math.isfinite(auxiliary_scale) or auxiliary_scale <= 0.0:
        raise ValueError("auxiliary scale must be finite and positive")
    candidate_raw = gradient_ops.add_gradients(
        raw_baseline, raw_auxiliary, right_scale=auxiliary_scale
    )
    baseline_update, baseline_audit = clipped_adam_first_displacement(
        raw_baseline, max_norm, adam_epsilon, clip_epsilon
    )
    candidate_update, candidate_audit = clipped_adam_first_displacement(
        candidate_raw, max_norm, adam_epsilon, clip_epsilon
    )
    exact_increment = gradient_ops.add_gradients(
        candidate_update, baseline_update, right_scale=-1.0
    )
    linear_increment, _ = clipped_adam_first_jvp(
        raw_baseline,
        gradient_ops.scaled_gradient(raw_auxiliary, auxiliary_scale),
        max_norm,
        adam_epsilon,
        clip_epsilon,
    )
    exact_norm = gradient_math.gradient_norm(exact_increment)
    linear_norm = gradient_math.gradient_norm(linear_increment)
    baseline_norm = gradient_math.gradient_norm(baseline_update)
    candidate_norm = gradient_math.gradient_norm(candidate_update)
    if exact_norm <= 0.0 or linear_norm <= 0.0 or baseline_norm <= 0.0:
        raise ValueError("realized update increment is degenerate")
    units, certificate_norms = _normalized_gradients(certificates)
    certificate_metrics = {}
    for name, gradient in certificates.items():
        unit = units[name]
        certificate_metrics[name] = {
            "gradient_norm": certificate_norms[name],
            "exact_increment_dot": gradient_math.gradient_dot(
                gradient, exact_increment
            ),
            "exact_increment_cosine": gradient_math.gradient_cosine(
                gradient, exact_increment
            ),
            "linear_increment_dot": gradient_math.gradient_dot(
                gradient, linear_increment
            ),
            "baseline_update_cosine": gradient_math.gradient_cosine(
                gradient, baseline_update
            ),
            "candidate_update_cosine": gradient_math.gradient_cosine(
                gradient, candidate_update
            ),
            "candidate_update_dot": gradient_math.gradient_dot(
                gradient, candidate_update
            ),
            "baseline_normalized_score": gradient_math.gradient_dot(
                unit, baseline_update
            )
            / baseline_norm,
            "linear_candidate_normalized_score": (
                gradient_math.gradient_dot(unit, baseline_update)
                + gradient_math.gradient_dot(unit, linear_increment)
            )
            / baseline_norm,
            "exact_candidate_normalized_score": gradient_math.gradient_dot(
                unit, candidate_update
            )
            / baseline_norm,
            "exact_increment_normalized_score": gradient_math.gradient_dot(
                unit, exact_increment
            )
            / baseline_norm,
        }
    return {
        "baseline": baseline_audit,
        "candidate": candidate_audit,
        "exact_increment_norm": exact_norm,
        "linear_increment_norm": linear_norm,
        "increment_norm_fraction_of_baseline": exact_norm / baseline_norm,
        "linear_exact_norm_ratio": linear_norm / exact_norm,
        "linear_exact_relative_norm_error": abs(linear_norm - exact_norm)
        / exact_norm,
        "linear_exact_cosine": gradient_math.gradient_cosine(
            linear_increment, exact_increment
        ),
        "minimum_baseline_normalized_score": min(
            value["baseline_normalized_score"]
            for value in certificate_metrics.values()
        ),
        "minimum_linear_candidate_normalized_score": min(
            value["linear_candidate_normalized_score"]
            for value in certificate_metrics.values()
        ),
        "minimum_exact_candidate_normalized_score": min(
            value["exact_candidate_normalized_score"]
            for value in certificate_metrics.values()
        ),
        "minimum_exact_increment_normalized_score": min(
            value["exact_increment_normalized_score"]
            for value in certificate_metrics.values()
        ),
        "certificates": certificate_metrics,
        "candidate_update_norm_fraction_of_baseline": candidate_norm / baseline_norm,
    }
