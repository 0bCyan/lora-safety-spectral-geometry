"""Fit-only affine max-min tangents inside a protected NPT-MCD nullspace."""

from __future__ import annotations

import math
from typing import Any

import step3_ojmcd_common as optimizer_geometry
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def _linear_combination(
    vectors: list[dict[str, Any]], coefficients: list[float]
) -> dict[str, Any]:
    if not vectors or len(vectors) != len(coefficients):
        raise ValueError("linear-combination inputs differ")
    result = gradient_ops.scaled_gradient(vectors[0], coefficients[0])
    for vector, coefficient in zip(vectors[1:], coefficients[1:], strict=True):
        result = gradient_ops.add_gradients(result, vector, right_scale=coefficient)
    return result


def _validate_registry(vectors: list[dict[str, Any]]) -> None:
    if not vectors or not vectors[0]:
        raise ValueError("gradient registry is empty")
    names = set(vectors[0])
    if any(set(vector) != names for vector in vectors[1:]):
        raise ValueError("gradient registries differ")


def project_registry_to_constraint_nullspace(
    vectors: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    rcond: float,
    refinement_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project many vectors using one shared float64 Gram pseudoinverse.

    Reusing the constraint factorization makes the target max-min problem
    auditable without recomputing a separate projector for every target.
    """
    import torch

    _validate_registry([*vectors, *constraints])
    if (
        not math.isfinite(rcond)
        or not 0.0 < rcond < 1.0
        or refinement_steps < 0
        or refinement_steps > 8
        or not constraints
    ):
        raise ValueError("projection controls are invalid")

    constraint_norms = [gradient_math.gradient_norm(item) for item in constraints]
    if any(not math.isfinite(value) or value <= 0.0 for value in constraint_norms):
        raise ValueError("constraint is degenerate")
    normalized_constraints = [
        gradient_ops.scaled_gradient(item, 1.0 / norm)
        for item, norm in zip(constraints, constraint_norms, strict=True)
    ]
    gram_values = [
        [
            gradient_math.gradient_dot(left, right)
            for right in normalized_constraints
        ]
        for left in normalized_constraints
    ]
    gram = torch.tensor(gram_values, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = float(eigenvalues.max())
    threshold = rcond * maximum
    keep = eigenvalues > threshold
    rank = int(keep.sum())
    if maximum <= 0.0 or rank <= 0:
        raise ValueError("constraint Gram matrix has zero numerical rank")
    kept_values = eigenvalues[keep]
    kept_vectors = eigenvectors[:, keep]
    pseudoinverse = (kept_vectors / kept_values.unsqueeze(0)) @ kept_vectors.T

    projected_vectors: list[dict[str, Any]] = []
    vector_audits = []
    for vector in vectors:
        projected = {name: value.clone() for name, value in vector.items()}
        residual_history = []
        for _ in range(refinement_steps + 1):
            residual = torch.tensor(
                [
                    gradient_math.gradient_dot(constraint, projected)
                    for constraint in normalized_constraints
                ],
                dtype=torch.float64,
            )
            residual_history.append(float(torch.linalg.vector_norm(residual)))
            coefficients = (pseudoinverse @ residual).tolist()
            correction = _linear_combination(normalized_constraints, coefficients)
            projected = gradient_ops.add_gradients(
                projected, correction, right_scale=-1.0
            )
        final_dots = [
            gradient_math.gradient_dot(constraint, projected)
            for constraint in normalized_constraints
        ]
        input_norm = gradient_math.gradient_norm(vector)
        projected_norm = gradient_math.gradient_norm(projected)
        vector_audits.append(
            {
                "input_norm": input_norm,
                "projected_norm": projected_norm,
                "retained_norm_fraction": projected_norm / max(input_norm, 1e-30),
                "residual_norm_history": residual_history,
                "maximum_absolute_final_constraint_dot": max(
                    abs(value) for value in final_dots
                ),
                "maximum_relative_final_constraint_residual": max(
                    abs(value) for value in final_dots
                )
                / max(projected_norm, 1e-30),
            }
        )
        projected_vectors.append(projected)

    positive = kept_values.tolist()
    return projected_vectors, {
        "constraints": len(constraints),
        "constraint_norms": constraint_norms,
        "gram_matrix": gram.tolist(),
        "gram_eigenvalues": eigenvalues.tolist(),
        "rcond": rcond,
        "eigenvalue_threshold": threshold,
        "numerical_rank": rank,
        "condition_number_retained": max(positive) / min(positive),
        "refinement_steps": refinement_steps,
        "vectors": vector_audits,
        "maximum_absolute_final_constraint_dot": max(
            item["maximum_absolute_final_constraint_dot"] for item in vector_audits
        ),
        "maximum_relative_final_constraint_residual": max(
            item["maximum_relative_final_constraint_residual"]
            for item in vector_audits
        ),
    }


def affine_maxmin_target_tangent(
    raw_displacement: dict[str, Any],
    target_gradients: list[tuple[str, dict[str, Any]]],
    raw_target_scores: dict[str, float],
    protected_constraints: list[dict[str, Any]],
    reference_norm: float,
    theta: float,
    rcond: float,
    refinement_steps: int,
    solver_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Solve the fixed-angle fit-target max-min problem in the nullspace.

    If ``b_j`` is a normalized target covector and ``P`` projects onto the
    protected/raw nullspace, the candidate target score is

        r_j cos(theta) + <P b_j, u> sin(theta) / reference_norm.

    The returned tangent solves the affine max-min problem over the ball
    ``||u|| <= ||raw_displacement||``. Its simplex dual and KKT residual are
    recorded by ``affine_minimax_direction``.
    """
    if (
        not target_gradients
        or len(target_gradients) < 2
        or set(raw_target_scores) != {name for name, _ in target_gradients}
        or not math.isfinite(reference_norm)
        or reference_norm <= 0.0
        or not math.isfinite(theta)
        or not 0.0 < theta < math.pi / 2.0
    ):
        raise ValueError("affine max-min inputs are invalid")
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    if raw_norm <= 0.0:
        raise ValueError("raw displacement is degenerate")

    target_names = []
    normalized_targets = []
    target_norms = {}
    for name, gradient in target_gradients:
        if name in target_names:
            raise ValueError("target names are not unique")
        norm = gradient_math.gradient_norm(gradient)
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("target gradient is degenerate")
        target_names.append(name)
        target_norms[name] = norm
        normalized_targets.append(
            gradient_ops.scaled_gradient(gradient, 1.0 / norm)
        )

    constraints = [*protected_constraints, raw_displacement]
    projected, projection = project_registry_to_constraint_nullspace(
        normalized_targets, constraints, rcond, refinement_steps
    )
    projected_norms = [gradient_math.gradient_norm(item) for item in projected]
    if any(not math.isfinite(value) or value <= solver_tolerance for value in projected_norms):
        raise ValueError("a target covector has no protected-null component")

    scale = math.sin(theta) / reference_norm
    solver_gradients = {
        name: gradient_ops.scaled_gradient(item, scale)
        for name, item in zip(target_names, projected, strict=True)
    }
    offsets = {
        name: float(raw_target_scores[name]) * math.cos(theta)
        for name in target_names
    }
    tangent, solver = optimizer_geometry.affine_minimax_direction(
        solver_gradients,
        offsets,
        raw_norm,
        solver_tolerance,
    )
    candidate = gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(raw_displacement, math.cos(theta)),
        tangent,
        right_scale=math.sin(theta),
    )
    candidate_scores = {
        name: float(raw_target_scores[name]) * math.cos(theta)
        + math.sin(theta)
        * gradient_math.gradient_dot(normalized, tangent)
        / reference_norm
        for name, normalized in zip(target_names, normalized_targets, strict=True)
    }
    solver_predicted = float(solver["primal_value"])
    actual_minimum = min(candidate_scores.values())
    return tangent, candidate, {
        "target_names": target_names,
        "target_norms": target_norms,
        "projected_target_norms": dict(
            zip(target_names, projected_norms, strict=True)
        ),
        "projection": projection,
        "solver": solver,
        "raw_target_scores": dict(raw_target_scores),
        "candidate_target_scores": candidate_scores,
        "solver_predicted_minimum_score": solver_predicted,
        "candidate_actual_minimum_score": actual_minimum,
        "candidate_actual_minus_predicted": actual_minimum - solver_predicted,
        "theta": theta,
        "raw_displacement_norm": raw_norm,
        "tangent_norm": gradient_math.gradient_norm(tangent),
        "candidate_norm": gradient_math.gradient_norm(candidate),
    }
