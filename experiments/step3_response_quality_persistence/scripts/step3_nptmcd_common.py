"""Nullspace-protected angular target geometry for MCD displacements."""

from __future__ import annotations

import math
from typing import Any

import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def _validate_registry(vectors: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    if not vectors or not vectors[0]:
        raise ValueError("at least one nonempty vector is required")
    names = set(vectors[0])
    if any(set(vector) != names for vector in vectors[1:]):
        raise ValueError("vector registries differ")


def _linear_combination(
    vectors: list[dict[str, Any]], coefficients: list[float]
) -> dict[str, Any]:
    if not vectors or len(vectors) != len(coefficients):
        raise ValueError("linear-combination inputs differ")
    result = gradient_ops.scaled_gradient(vectors[0], coefficients[0])
    for vector, coefficient in zip(vectors[1:], coefficients[1:], strict=True):
        result = gradient_ops.add_gradients(result, vector, right_scale=coefficient)
    return result


def project_to_constraint_nullspace(
    vector: dict[str, Any],
    constraints: list[dict[str, Any]],
    rcond: float,
    refinement_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project a vector off the span of normalized constraint covectors."""
    import torch

    _validate_registry([vector, *constraints])
    if (
        not math.isfinite(rcond)
        or not 0.0 < rcond < 1.0
        or refinement_steps < 0
        or refinement_steps > 8
    ):
        raise ValueError("nullspace projection controls are invalid")
    norms = [gradient_math.gradient_norm(item) for item in constraints]
    if any(not math.isfinite(value) or value <= 0.0 for value in norms):
        raise ValueError("nullspace constraint is degenerate")
    gram_values = [[0.0] * len(constraints) for _ in constraints]
    for left_index, left in enumerate(constraints):
        for right_index in range(left_index, len(constraints)):
            value = gradient_math.gradient_dot(left, constraints[right_index]) / (
                norms[left_index] * norms[right_index]
            )
            gram_values[left_index][right_index] = value
            gram_values[right_index][left_index] = value
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

    projected = {name: value.clone() for name, value in vector.items()}
    residual_history = []
    for _ in range(refinement_steps + 1):
        residual = torch.tensor(
            [
                gradient_math.gradient_dot(constraint, projected) / norm
                for constraint, norm in zip(constraints, norms, strict=True)
            ],
            dtype=torch.float64,
        )
        residual_history.append(float(torch.linalg.vector_norm(residual)))
        coefficients = (pseudoinverse @ residual).tolist()
        correction = _linear_combination(
            constraints,
            [
                coefficient / norm
                for coefficient, norm in zip(coefficients, norms, strict=True)
            ],
        )
        projected = gradient_ops.add_gradients(projected, correction, right_scale=-1.0)

    final_dots = [
        gradient_math.gradient_dot(constraint, projected) / norm
        for constraint, norm in zip(constraints, norms, strict=True)
    ]
    input_norm = gradient_math.gradient_norm(vector)
    projected_norm = gradient_math.gradient_norm(projected)
    positive = kept_values.tolist()
    audit = {
        "constraints": len(constraints),
        "constraint_norms": norms,
        "gram_matrix": gram.tolist(),
        "gram_eigenvalues": eigenvalues.tolist(),
        "rcond": rcond,
        "eigenvalue_threshold": threshold,
        "numerical_rank": rank,
        "condition_number_retained": max(positive) / min(positive),
        "refinement_steps": refinement_steps,
        "residual_norm_history": residual_history,
        "final_constraint_dots": final_dots,
        "maximum_absolute_final_constraint_dot": max(abs(value) for value in final_dots),
        "maximum_relative_final_constraint_residual": max(abs(value) for value in final_dots)
        / max(projected_norm, 1e-30),
        "input_norm": input_norm,
        "projected_norm": projected_norm,
        "retained_norm_fraction": projected_norm / input_norm,
    }
    return projected, audit


def protected_angular_tangent(
    raw_displacement: dict[str, Any],
    target_proposal: dict[str, Any],
    protected_constraints: list[dict[str, Any]],
    rcond: float,
    refinement_steps: int,
    minimum_retained_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a raw-orthogonal protected-null tangent with raw-matched norm."""
    _validate_registry([raw_displacement, target_proposal, *protected_constraints])
    if (
        not math.isfinite(minimum_retained_fraction)
        or not 0.0 < minimum_retained_fraction <= 1.0
    ):
        raise ValueError("minimum retained fraction is invalid")
    proposal_norm = gradient_math.gradient_norm(target_proposal)
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    if proposal_norm <= 0.0 or raw_norm <= 0.0:
        raise ValueError("angular tangent endpoints are degenerate")

    projected, projection = project_to_constraint_nullspace(
        target_proposal,
        [*protected_constraints, raw_displacement],
        rcond,
        refinement_steps,
    )
    projected_norm = gradient_math.gradient_norm(projected)
    retained = projected_norm / proposal_norm
    if retained < minimum_retained_fraction:
        raise ValueError("protected-null target component is too small")
    tangent = gradient_ops.scaled_gradient(projected, raw_norm / projected_norm)
    protected_dots = [
        gradient_math.gradient_dot(constraint, tangent)
        / gradient_math.gradient_norm(constraint)
        for constraint in protected_constraints
    ]
    raw_dot = gradient_math.gradient_dot(raw_displacement, tangent)
    tangent_norm = gradient_math.gradient_norm(tangent)
    return tangent, {
        "projection": projection,
        "protected_constraints": len(protected_constraints),
        "proposal_norm": proposal_norm,
        "projected_proposal_norm": projected_norm,
        "retained_target_proposal_fraction": retained,
        "raw_displacement_norm": raw_norm,
        "tangent_norm": tangent_norm,
        "tangent_to_raw_norm_ratio": tangent_norm / raw_norm,
        "raw_tangent_cosine": raw_dot / (raw_norm * tangent_norm),
        "maximum_absolute_normalized_protected_dot": max(
            abs(value) for value in protected_dots
        ),
        "maximum_absolute_protected_cosine": max(
            abs(value) for value in protected_dots
        )
        / tangent_norm,
        "normalized_protected_dots": protected_dots,
    }


def angular_displacement(
    raw_displacement: dict[str, Any], tangent: dict[str, Any], theta: float
) -> dict[str, Any]:
    _validate_registry([raw_displacement, tangent])
    if not math.isfinite(theta) or not 0.0 <= theta <= math.pi / 2.0:
        raise ValueError("angular dose is invalid")
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    tangent_norm = gradient_math.gradient_norm(tangent)
    if raw_norm <= 0.0 or abs(tangent_norm / raw_norm - 1.0) > 1e-6:
        raise ValueError("angular basis vectors do not have matched norms")
    return gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(raw_displacement, math.cos(theta)),
        tangent,
        right_scale=math.sin(theta),
    )


def _score_at_angle(raw_score: float, tangent_score: float, theta: float) -> float:
    return raw_score * math.cos(theta) + tangent_score * math.sin(theta)


def _equality_roots(
    raw_score: float, tangent_score: float, threshold: float, theta_max: float
) -> list[float]:
    amplitude = math.hypot(raw_score, tangent_score)
    if amplitude <= 0.0 or abs(threshold) > amplitude:
        return []
    phase = math.atan2(raw_score, tangent_score)
    principal = math.asin(max(-1.0, min(1.0, threshold / amplitude)))
    roots = []
    for turns in range(-2, 3):
        offset = 2.0 * math.pi * turns
        for value in (principal - phase + offset, math.pi - principal - phase + offset):
            if 0.0 <= value <= theta_max:
                roots.append(value)
    return roots


def select_minimum_effective_angle(
    raw_panels: list[dict[str, float]],
    tangent_panels: list[dict[str, float]],
    protected_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
    protected_retention_floor: float,
    design_cosine_floor: float,
    target_gain_floor: float,
    target_score_floor: float,
    numeric_tolerance: float,
    angle_guard: float,
) -> dict[str, Any]:
    """Select the leftmost exact feasible angle from trigonometric boundaries."""
    protected = tuple(protected_names)
    targets = tuple(target_names)
    if (
        len(raw_panels) < 2
        or len(raw_panels) != len(tangent_panels)
        or not protected
        or not targets
        or set(protected).intersection(targets)
        or not 0.0 < protected_retention_floor < design_cosine_floor < 1.0
        or target_gain_floor <= 0.0
        or target_score_floor <= 0.0
        or numeric_tolerance <= 0.0
        or angle_guard <= 0.0
    ):
        raise ValueError("angular selection controls are invalid")
    certificate_names = tuple(raw_panels[0])
    names = set(certificate_names)
    if set(protected).union(targets) != names or any(
        set(panel) != names for panel in [*raw_panels, *tangent_panels]
    ):
        raise ValueError("angular score panels differ")
    if any(
        not math.isfinite(float(value))
        for panel in [*raw_panels, *tangent_panels]
        for value in panel.values()
    ):
        raise ValueError("angular score panel is nonfinite")
    raw_protected_positive = all(
        float(panel[name]) > 0.0 for panel in raw_panels for name in protected
    )
    if not raw_protected_positive:
        raise ValueError("raw protected score is nonpositive")

    raw_target_floor = min(
        float(panel[name]) for panel in raw_panels for name in targets
    )
    required_target_floor = max(
        target_score_floor, raw_target_floor + target_gain_floor
    )
    theta_max = math.acos(design_cosine_floor)
    boundaries = [0.0, theta_max]
    target_coefficients = []
    for panel_index, (raw, tangent) in enumerate(
        zip(raw_panels, tangent_panels, strict=True)
    ):
        for name in targets:
            raw_score = float(raw[name])
            tangent_score = float(tangent[name])
            roots = _equality_roots(
                raw_score, tangent_score, required_target_floor, theta_max
            )
            boundaries.extend(roots)
            target_coefficients.append(
                {
                    "panel_index": panel_index,
                    "certificate": name,
                    "raw_score": raw_score,
                    "tangent_score": tangent_score,
                    "boundary_roots": roots,
                }
            )
    boundaries.sort()
    unique = []
    for value in boundaries:
        if not unique or abs(value - unique[-1]) > 1e-12:
            unique.append(value)

    def target_feasible(theta: float) -> bool:
        return all(
            _score_at_angle(
                float(raw[name]), float(tangent[name]), theta
            )
            >= required_target_floor - numeric_tolerance
            for raw, tangent in zip(raw_panels, tangent_panels, strict=True)
            for name in targets
        )

    selected_theta: float | None = None
    for index, left in enumerate(unique):
        if left > 0.0 and target_feasible(left):
            selected_theta = left
            break
        if index + 1 >= len(unique):
            continue
        right = unique[index + 1]
        midpoint = 0.5 * (left + right)
        if target_feasible(midpoint):
            selected_theta = left
            break

    if selected_theta is not None:
        selected_theta = min(theta_max, selected_theta + angle_guard)
    theta = selected_theta if selected_theta is not None else 0.0
    candidate_panels = [
        {
            name: _score_at_angle(float(raw[name]), float(tangent[name]), theta)
            for name in certificate_names
        }
        for raw, tangent in zip(raw_panels, tangent_panels, strict=True)
    ]
    retentions = {
        name: min(
            float(candidate[name]) / float(raw[name])
            for raw, candidate in zip(raw_panels, candidate_panels, strict=True)
        )
        for name in protected
    }
    target_floor = min(
        float(panel[name]) for panel in candidate_panels for name in targets
    )
    target_gain = target_floor - raw_target_floor
    identity_errors = [
        abs(float(candidate[name]) / float(raw[name]) - math.cos(theta))
        for raw, candidate in zip(raw_panels, candidate_panels, strict=True)
        for name in protected
    ]
    minimum_retention = min(retentions.values())
    eligible = (
        selected_theta is not None
        and theta > 0.0
        and theta <= theta_max + numeric_tolerance
        and minimum_retention >= protected_retention_floor - numeric_tolerance
        and target_floor >= target_score_floor - numeric_tolerance
        and target_gain >= target_gain_floor - numeric_tolerance
    )
    return {
        "eligible": eligible,
        "selected_theta": theta,
        "selected_theta_degrees": math.degrees(theta),
        "theta_max": theta_max,
        "theta_max_degrees": math.degrees(theta_max),
        "design_cosine_floor": design_cosine_floor,
        "protected_retention_floor": protected_retention_floor,
        "raw_target_score_floor": raw_target_floor,
        "required_target_score_floor": required_target_floor,
        "selected_target_score_floor": target_floor,
        "selected_target_gain_over_raw": target_gain,
        "protected_retentions": retentions,
        "minimum_protected_retention": minimum_retention,
        "maximum_protected_retention_identity_error": max(identity_errors),
        "candidate_panels": candidate_panels,
        "target_coefficients": target_coefficients,
        "trigonometric_boundaries": unique,
        "selection_rule": (
            "leftmost_exact_trigonometric_interval_meeting_joint_fit_target_"
            "floor_with_registered_angular_guard"
        ),
    }
