"""Convex distributionally robust optimizer-space Pareto control."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

import step3_medscmcd_common as conformal


class CdroOpcMathError(ValueError):
    """Raised when a CDR-OPC problem or certificate is undefined."""


def fit_ellipsoidal_moments(
    score_vectors: Sequence[Sequence[float]], ridge: float = 0.0
) -> dict[str, Any]:
    """Fit a mean and square-root covariance for optimizer-space scores."""
    samples = np.asarray(score_vectors, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[0] < 2
        or samples.shape[1] < 1
        or not np.isfinite(samples).all()
        or not math.isfinite(ridge)
        or ridge < 0.0
    ):
        raise CdroOpcMathError("ellipsoidal moment inputs are invalid")
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / float(samples.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.T)
    if ridge:
        covariance += ridge * np.eye(samples.shape[1], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    negative_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -negative_tolerance:
        raise CdroOpcMathError("sample covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    reconstruction = factor.T @ factor
    relative_error = float(
        np.linalg.norm(reconstruction - covariance)
        / max(1.0, np.linalg.norm(covariance))
    )
    return {
        "sample_count": int(samples.shape[0]),
        "dimension": int(samples.shape[1]),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
        "factor": factor.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "ridge": ridge,
        "factorization_relative_error": relative_error,
    }


def ellipsoidal_lower_bound(
    mean: Sequence[float],
    factor: Sequence[Sequence[float]],
    point: Sequence[float],
    uncertainty_radius: float,
) -> float:
    """Return min_(||u||<=radius) (mean + factor.T u).T point."""
    mean_array = np.asarray(mean, dtype=np.float64)
    factor_array = np.asarray(factor, dtype=np.float64)
    point_array = np.asarray(point, dtype=np.float64)
    if (
        mean_array.ndim != 1
        or point_array.shape != mean_array.shape
        or factor_array.ndim != 2
        or factor_array.shape[1] != mean_array.size
        or not np.isfinite(mean_array).all()
        or not np.isfinite(factor_array).all()
        or not np.isfinite(point_array).all()
        or not math.isfinite(uncertainty_radius)
        or uncertainty_radius < 0.0
    ):
        raise CdroOpcMathError("ellipsoidal lower-bound inputs are invalid")
    return float(
        mean_array @ point_array
        - uncertainty_radius * np.linalg.norm(factor_array @ point_array)
    )


def _validated_problem(
    behavior_target: Sequence[float],
    metric: Sequence[Sequence[float]],
    quality_score: Sequence[float],
    safety_score: Sequence[float],
    protected_constraints: Sequence[Mapping[str, Any]],
    fallback: Sequence[float],
    trust_radius: float,
    utility_weight: float,
    tau_regularization: float,
) -> dict[str, Any]:
    target = np.asarray(behavior_target, dtype=np.float64)
    metric_array = np.asarray(metric, dtype=np.float64)
    quality = np.asarray(quality_score, dtype=np.float64)
    safety = np.asarray(safety_score, dtype=np.float64)
    fallback_array = np.asarray(fallback, dtype=np.float64)
    if (
        target.ndim != 1
        or target.size < 1
        or metric_array.shape != (target.size, target.size)
        or quality.shape != target.shape
        or safety.shape != target.shape
        or fallback_array.shape != target.shape
        or not all(
            np.isfinite(value).all()
            for value in [target, metric_array, quality, safety, fallback_array]
        )
        or not protected_constraints
        or not math.isfinite(trust_radius)
        or trust_radius <= 0.0
        or not math.isfinite(utility_weight)
        or utility_weight <= 0.0
        or not math.isfinite(tau_regularization)
        or tau_regularization <= 0.0
    ):
        raise CdroOpcMathError("CDR-OPC problem inputs are invalid")
    metric_array = 0.5 * (metric_array + metric_array.T)
    metric_eigenvalues = np.linalg.eigvalsh(metric_array)
    if float(np.min(metric_eigenvalues)) <= 0.0:
        raise CdroOpcMathError("proximity metric must be positive definite")

    constraints = []
    names: set[str] = set()
    for item in protected_constraints:
        name = str(item.get("name", ""))
        mean = np.asarray(item.get("mean"), dtype=np.float64)
        factor = np.asarray(item.get("factor"), dtype=np.float64)
        center = np.asarray(item.get("center", np.zeros_like(target)), dtype=np.float64)
        radius = float(item.get("uncertainty_radius", math.nan))
        floor = float(item.get("floor", math.nan))
        if (
            not name
            or name in names
            or mean.shape != target.shape
            or factor.ndim != 2
            or factor.shape[1] != target.size
            or center.shape != target.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(factor).all()
            or not np.isfinite(center).all()
            or not math.isfinite(radius)
            or radius < 0.0
            or not math.isfinite(floor)
        ):
            raise CdroOpcMathError("protected constraint is invalid")
        names.add(name)
        constraints.append(
            {
                "name": name,
                "mean": mean,
                "factor": factor,
                "center": center,
                "uncertainty_radius": radius,
                "floor": floor,
            }
        )
    return {
        "target": target,
        "metric": metric_array,
        "metric_eigenvalues": metric_eigenvalues,
        "quality": quality,
        "safety": safety,
        "protected": constraints,
        "fallback": fallback_array,
        "trust_radius": float(trust_radius),
        "utility_weight": float(utility_weight),
        "tau_regularization": float(tau_regularization),
    }


def solve_cdro_opc(
    behavior_target: Sequence[float],
    metric: Sequence[Sequence[float]],
    quality_score: Sequence[float],
    safety_score: Sequence[float],
    protected_constraints: Sequence[Mapping[str, Any]],
    fallback: Sequence[float],
    trust_radius: float,
    utility_weight: float = 1.0,
    tau_regularization: float = 1e-4,
    feasibility_tolerance: float = 1e-7,
    kkt_tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the convex CDR-OPC problem and audit a sufficient KKT system.

    The decision point is expressed in a fixed optimizer-update basis.  The
    protected constraints are exact support-function bounds for ellipsoidal
    uncertainty sets. SLSQP supplies a numerical point, but its native status
    is not treated as a proof: acceptance is determined by the independent
    primal/dual/stationarity/complementarity audit.
    """
    if (
        not math.isfinite(feasibility_tolerance)
        or feasibility_tolerance <= 0.0
        or not math.isfinite(kkt_tolerance)
        or kkt_tolerance <= 0.0
    ):
        raise CdroOpcMathError("solver tolerances must be finite and positive")
    problem = _validated_problem(
        behavior_target,
        metric,
        quality_score,
        safety_score,
        protected_constraints,
        fallback,
        trust_radius,
        utility_weight,
        tau_regularization,
    )
    target = problem["target"]
    metric_array = problem["metric"]
    quality = problem["quality"]
    safety = problem["safety"]
    protected = problem["protected"]
    fallback_array = problem["fallback"]
    radius = problem["trust_radius"]
    utility = problem["utility_weight"]
    tau_reg = problem["tau_regularization"]
    dimension = target.size

    names = ["quality_pareto", "safety_pareto"]
    names.extend(f"protected:{item['name']}" for item in protected)
    names.append("trust_region")

    def objective(value: np.ndarray) -> float:
        delta = value[:dimension] - target
        tau = float(value[-1])
        return float(
            0.5 * delta @ metric_array @ delta
            - utility * tau
            + 0.5 * tau_reg * tau * tau
        )

    def objective_gradient(value: np.ndarray) -> np.ndarray:
        delta = value[:dimension] - target
        return np.concatenate(
            [metric_array @ delta, np.asarray([-utility + tau_reg * value[-1]])]
        )

    def constraint_values(value: np.ndarray) -> np.ndarray:
        point = value[:dimension]
        tau = float(value[-1])
        values = [float(quality @ point - tau), float(safety @ point - tau)]
        for item in protected:
            delta = point - item["center"]
            values.append(
                ellipsoidal_lower_bound(
                    item["mean"],
                    item["factor"],
                    delta,
                    item["uncertainty_radius"],
                )
                - item["floor"]
            )
        values.append(float(radius * radius - point @ point))
        return np.asarray(values, dtype=np.float64)

    def constraint_jacobian(value: np.ndarray) -> np.ndarray:
        point = value[:dimension]
        rows = [
            np.concatenate([quality, np.asarray([-1.0])]),
            np.concatenate([safety, np.asarray([-1.0])]),
        ]
        for item in protected:
            delta = point - item["center"]
            transformed = item["factor"] @ delta
            transformed_norm = float(np.linalg.norm(transformed))
            if transformed_norm <= 1e-14:
                uncertainty_gradient = np.zeros(dimension, dtype=np.float64)
            else:
                uncertainty_gradient = (
                    item["factor"].T @ transformed / transformed_norm
                )
            gradient = (
                item["mean"]
                - item["uncertainty_radius"] * uncertainty_gradient
            )
            rows.append(np.concatenate([gradient, np.asarray([0.0])]))
        rows.append(np.concatenate([-2.0 * point, np.asarray([0.0])]))
        return np.asarray(rows, dtype=np.float64)

    fallback_tau = float(min(quality @ fallback_array, safety @ fallback_array))
    fallback_value = np.concatenate([fallback_array, np.asarray([fallback_tau])])
    fallback_constraints = constraint_values(fallback_value)
    protected_fallback_minimum = float(np.min(fallback_constraints[2:]))
    if protected_fallback_minimum < -feasibility_tolerance:
        raise CdroOpcMathError("registered fallback is not robustly feasible")

    def trust_project(point: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(point))
        return point if norm <= radius else point * (radius / norm)

    target_start = trust_project(target.copy())
    zero_start = np.zeros(dimension, dtype=np.float64)
    starts = [fallback_array, target_start, 0.5 * (fallback_array + target_start)]
    starts.append(zero_start)
    start_values = [
        np.concatenate(
            [point, np.asarray([min(float(quality @ point), float(safety @ point))])]
        )
        for point in starts
    ]

    from scipy.optimize import minimize, nnls

    candidates: list[dict[str, Any]] = []
    for start in start_values:
        result = minimize(
            objective,
            start,
            jac=objective_gradient,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": constraint_values,
                "jac": constraint_jacobian,
            },
            options={"ftol": 1e-12, "maxiter": 3000, "disp": False},
        )
        value = np.asarray(result.x, dtype=np.float64)
        if value.shape == (dimension + 1,) and np.isfinite(value).all():
            constraints_at_value = constraint_values(value)
            candidates.append(
                {
                    "value": value,
                    "objective": objective(value),
                    "constraints": constraints_at_value,
                    "solver_success": bool(result.success),
                    "solver_status": int(result.status),
                    "solver_message": str(result.message),
                    "solver_iterations": int(result.nit),
                }
            )
    if not candidates:
        return fallback_array.copy(), {
            "route": "fallback_no_finite_solver_candidate",
            "optimization_valid": False,
            "fallback_feasible": True,
            "fallback_constraint_values": dict(
                zip(names, fallback_constraints.tolist(), strict=True)
            ),
            "optimizer_steps": 0,
            "training_authorized": False,
        }

    candidate = min(candidates, key=lambda item: item["objective"])
    value = candidate["value"]
    constraints_at_value = candidate["constraints"]
    jacobian = constraint_jacobian(value)
    gradient = objective_gradient(value)
    active_threshold = max(10.0 * feasibility_tolerance, 1e-7)
    active_indices = [
        index
        for index, current in enumerate(constraints_at_value)
        if current <= active_threshold
    ]
    multipliers = np.zeros(len(names), dtype=np.float64)
    if active_indices:
        active_multipliers, _ = nnls(jacobian[active_indices].T, gradient)
        multipliers[active_indices] = active_multipliers
    stationarity_vector = gradient - jacobian.T @ multipliers
    stationarity = float(np.linalg.norm(stationarity_vector))
    normalized_stationarity = stationarity / max(1.0, float(np.linalg.norm(gradient)))
    primal_violation = max(0.0, -float(np.min(constraints_at_value)))
    dual_violation = max(0.0, -float(np.min(multipliers)))
    complementarity = float(np.max(np.abs(multipliers * constraints_at_value)))
    fallback_objective = objective(fallback_value)
    objective_dominates_fallback = (
        float(candidate["objective"]) <= fallback_objective + feasibility_tolerance
    )
    kkt_valid = (
        primal_violation <= feasibility_tolerance
        and dual_violation <= kkt_tolerance
        and normalized_stationarity <= kkt_tolerance
        and complementarity <= kkt_tolerance
        and objective_dominates_fallback
    )
    selected = value[:dimension].copy() if kkt_valid else fallback_array.copy()
    return selected, {
        "route": "candidate" if kkt_valid else "fallback_failed_kkt_audit",
        "optimization_valid": kkt_valid,
        "solver": "scipy_SLSQP_multistart_with_explicit_convex_KKT_audit",
        "solver_success": bool(candidate["solver_success"]),
        "solver_status": int(candidate["solver_status"]),
        "solver_message": str(candidate["solver_message"]),
        "solver_iterations": int(candidate["solver_iterations"]),
        "solver_native_status_required_for_validity": False,
        "validity_basis": "explicit_sufficient_convex_KKT_audit",
        "finite_solver_candidates": len(candidates),
        "dimension": int(dimension),
        "objective": float(candidate["objective"]),
        "fallback_objective": fallback_objective,
        "objective_dominates_fallback": objective_dominates_fallback,
        "candidate_point": value[:dimension].tolist(),
        "selected_point": selected.tolist(),
        "candidate_tau": float(value[-1]),
        "constraint_values": dict(zip(names, constraints_at_value.tolist(), strict=True)),
        "fallback_constraint_values": dict(
            zip(names, fallback_constraints.tolist(), strict=True)
        ),
        "active_constraints": [names[index] for index in active_indices],
        "multipliers": dict(zip(names, multipliers.tolist(), strict=True)),
        "primal_violation": primal_violation,
        "dual_violation": dual_violation,
        "stationarity_residual": stationarity,
        "normalized_stationarity_residual": normalized_stationarity,
        "complementarity_residual": complementarity,
        "metric_minimum_eigenvalue": float(
            np.min(problem["metric_eigenvalues"])
        ),
        "fallback_feasible": True,
        "convexity_certificate": (
            "strongly_convex_objective_over_intersection_of_SOC_superlevel_sets"
        ),
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def conformal_transport_policy(
    raw_score_units: Sequence[Mapping[str, float]],
    candidate_score_units: Sequence[Mapping[str, float]],
    protected_names: Sequence[str],
    retention_floor: float,
    alpha: float,
    fallback_nonconformity: float,
) -> dict[str, Any]:
    """Audit the fixed candidate-or-fallback policy on sealed calibration units."""
    if len(raw_score_units) != len(candidate_score_units) or not raw_score_units:
        raise CdroOpcMathError("conformal score panels differ")
    nonconformities = []
    residuals = []
    for raw, candidate in zip(raw_score_units, candidate_score_units, strict=True):
        value, current = conformal.joint_protected_nonconformity(
            dict(raw), dict(candidate), list(protected_names), retention_floor
        )
        nonconformities.append(value)
        residuals.append(current)
    policy = conformal.split_conformal_accept_or_fallback(
        nonconformities,
        alpha=alpha,
        acceptance_threshold=0.0,
        fallback_nonconformity=fallback_nonconformity,
    )
    return {
        **policy,
        "joint_nonconformities": nonconformities,
        "per_unit_residuals": residuals,
        "candidate_fixed_before_calibration_required": True,
        "calibration_used_for_optimization": False,
    }


def conformal_standardized_change_policy(
    fallback_score_units: Sequence[Mapping[str, float]],
    candidate_score_units: Sequence[Mapping[str, float]],
    protected_names: Sequence[str],
    allowed_relative_drop: float,
    score_scale_floor: float,
    alpha: float,
) -> dict[str, Any]:
    """Certify bounded standardized deterioration, including negative baselines.

    Each residual is

        (fallback - candidate) / max(abs(fallback), scale_floor) - delta.

    Unlike a retention ratio, this remains ordered correctly when the fallback
    score is negative and is bounded near zero by the fit-frozen scale floor.
    """
    protected = tuple(protected_names)
    if (
        len(fallback_score_units) != len(candidate_score_units)
        or not fallback_score_units
        or not protected
        or not math.isfinite(allowed_relative_drop)
        or allowed_relative_drop < 0.0
        or not math.isfinite(score_scale_floor)
        or score_scale_floor <= 0.0
    ):
        raise CdroOpcMathError("standardized conformal inputs are invalid")
    nonconformities = []
    residuals = []
    scales = []
    for fallback_scores, candidate_scores in zip(
        fallback_score_units, candidate_score_units, strict=True
    ):
        if (
            set(fallback_scores) != set(candidate_scores)
            or not set(protected).issubset(fallback_scores)
        ):
            raise CdroOpcMathError("standardized conformal score sets differ")
        current_residuals = {}
        current_scales = {}
        for name in protected:
            fallback_value = float(fallback_scores[name])
            candidate_value = float(candidate_scores[name])
            if not math.isfinite(fallback_value) or not math.isfinite(candidate_value):
                raise CdroOpcMathError("standardized conformal score is not finite")
            scale = max(abs(fallback_value), score_scale_floor)
            current_scales[name] = scale
            current_residuals[name] = (
                (fallback_value - candidate_value) / scale - allowed_relative_drop
            )
        residuals.append(current_residuals)
        scales.append(current_scales)
        nonconformities.append(max(current_residuals.values()))
    policy = conformal.split_conformal_accept_or_fallback(
        nonconformities,
        alpha=alpha,
        acceptance_threshold=0.0,
        fallback_nonconformity=-allowed_relative_drop,
    )
    return {
        **policy,
        "joint_nonconformities": nonconformities,
        "per_unit_residuals": residuals,
        "per_unit_scales": scales,
        "allowed_relative_drop": allowed_relative_drop,
        "score_scale_floor": score_scale_floor,
        "candidate_fixed_before_calibration_required": True,
        "calibration_used_for_optimization": False,
        "negative_fallback_scores_supported": True,
    }
