"""Finite-radius subset-certified common descent for the exact optimizer map."""

from __future__ import annotations

import math
from typing import Any

import step3_mcd_common as mcd
import step3_ojmcd_common as optimizer_map
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def normalized_certificates(
    certificates: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    if len(certificates) < 2:
        raise ValueError("at least two certificates are required")
    names = list(certificates)
    reference = certificates[names[0]]
    if not reference or any(
        set(gradient) != set(reference) for gradient in certificates.values()
    ):
        raise ValueError("certificate gradient dictionaries differ")
    norms = {
        name: gradient_math.gradient_norm(gradient)
        for name, gradient in certificates.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise ValueError("certificate gradients must be finite and nonzero")
    units = {
        name: gradient_ops.scaled_gradient(certificates[name], 1.0 / norms[name])
        for name in names
    }
    return units, norms


def _weighted_gradient(
    gradients: list[dict[str, Any]], weights: list[float]
) -> dict[str, Any]:
    if not gradients or len(gradients) != len(weights):
        raise ValueError("weighted gradient inputs differ")
    result = gradient_ops.scaled_gradient(gradients[0], weights[0])
    for gradient, weight in zip(gradients[1:], weights[1:], strict=True):
        result = gradient_ops.add_gradients(result, gradient, right_scale=weight)
    return result


def realized_displacement(
    raw_baseline: dict[str, Any],
    auxiliary: dict[str, Any] | None,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    if not math.isfinite(auxiliary_scale) or auxiliary_scale < 0.0:
        raise ValueError("auxiliary scale must be finite and nonnegative")
    if auxiliary is None:
        candidate = {name: value.clone() for name, value in raw_baseline.items()}
    else:
        candidate = gradient_ops.add_gradients(
            raw_baseline, auxiliary, right_scale=auxiliary_scale
        )
    return optimizer_map.clipped_adam_first_displacement(
        candidate, max_norm, adam_epsilon, clip_epsilon
    )


def displacement_scores(
    certificates: dict[str, dict[str, Any]],
    displacement: dict[str, Any],
    reference_norm: float,
) -> dict[str, Any]:
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise ValueError("score reference norm must be finite and positive")
    units, norms = normalized_certificates(certificates)
    scores = {
        name: gradient_math.gradient_dot(unit, displacement) / reference_norm
        for name, unit in units.items()
    }
    return {
        "certificate_norms": norms,
        "scores": scores,
        "minimum_score": min(scores.values()),
        "mean_score": sum(scores.values()) / len(scores),
        "displacement_norm": gradient_math.gradient_norm(displacement),
        "reference_norm": reference_norm,
    }


def _subset_direction(
    units: dict[str, dict[str, Any]],
    gram: list[list[float]],
    indices: list[int],
    budget_norm: float,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = list(units)
    if len(indices) == 1:
        direction = gradient_ops.scaled_gradient(units[names[indices[0]]], budget_norm)
        return direction, {
            "dual_margin": 1.0,
            "dual_weights": {names[indices[0]]: 1.0},
            "kkt_residual": 0.0,
            "active_set_size": 1,
        }
    subgram = [[gram[left][right] for right in indices] for left in indices]
    weights, dual = mcd.minimum_norm_simplex_weights(subgram, tolerance)
    selected = [units[names[index]] for index in indices]
    weighted = _weighted_gradient(selected, weights)
    margin = gradient_math.gradient_norm(weighted)
    if not math.isfinite(margin) or margin <= tolerance:
        raise ValueError("subset has no strict Euclidean common direction")
    direction = gradient_ops.scaled_gradient(weighted, budget_norm / margin)
    return direction, {
        "dual_margin": margin,
        "dual_weights": {
            names[index]: weight
            for index, weight in zip(indices, weights, strict=True)
        },
        **dual,
    }


def enumerate_finite_radius_subsets(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    budget_norm: float,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
    top_k: int = 8,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Exhaust all nonempty certificate subsets and maximize exact margin.

    Returns the selected direction, the full-seven-certificate control, and a
    scalar audit. The finite candidate family has exactly 2**m - 1 members
    before infeasible zero-margin subsets are removed.
    """
    if (
        not math.isfinite(budget_norm)
        or budget_norm <= 0.0
        or not math.isfinite(auxiliary_scale)
        or auxiliary_scale <= 0.0
        or top_k <= 0
    ):
        raise ValueError("finite-radius subset inputs are invalid")
    units, certificate_norms = normalized_certificates(certificates)
    names = list(units)
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units.values()]
        for left in units.values()
    ]
    baseline_displacement, baseline_audit = realized_displacement(
        raw_baseline,
        None,
        0.0,
        max_norm,
        adam_epsilon,
        clip_epsilon,
    )
    baseline_norm = gradient_math.gradient_norm(baseline_displacement)
    if not math.isfinite(baseline_norm) or baseline_norm <= 0.0:
        raise ValueError("baseline displacement is degenerate")

    candidate_records: list[dict[str, Any]] = []
    selected_direction: dict[str, Any] | None = None
    full_direction: dict[str, Any] | None = None
    selected_record: dict[str, Any] | None = None
    full_record: dict[str, Any] | None = None
    attempted = 2 ** len(names) - 1
    full_mask = attempted
    for mask in range(1, attempted + 1):
        indices = [index for index in range(len(names)) if mask & (1 << index)]
        try:
            direction, dual = _subset_direction(
                units, gram, indices, budget_norm, dual_tolerance
            )
        except ValueError:
            continue
        displacement, map_audit = realized_displacement(
            raw_baseline,
            direction,
            auxiliary_scale,
            max_norm,
            adam_epsilon,
            clip_epsilon,
        )
        score = displacement_scores(certificates, displacement, baseline_norm)
        record = {
            "mask": mask,
            "subset": [names[index] for index in indices],
            "subset_size": len(indices),
            "dual_margin": float(dual["dual_margin"]),
            "dual_kkt_residual": float(dual["kkt_residual"]),
            "minimum_exact_score": float(score["minimum_score"]),
            "mean_exact_score": float(score["mean_score"]),
            "exact_scores": score["scores"],
            "displacement_norm_fraction_of_baseline": float(
                score["displacement_norm"] / baseline_norm
            ),
            "candidate_raw_norm": float(map_audit["raw_norm"]),
            "candidate_clip_coefficient": float(map_audit["clip_coefficient"]),
            "candidate_clipping_active": bool(map_audit["clipping_active"]),
            "direction_budget_relative_error": abs(
                gradient_math.gradient_norm(direction) - budget_norm
            )
            / budget_norm,
        }
        candidate_records.append(record)
        rank = (
            record["minimum_exact_score"],
            record["mean_exact_score"],
            -record["subset_size"],
            -record["mask"],
        )
        if selected_record is None:
            better = True
        else:
            current_rank = (
                selected_record["minimum_exact_score"],
                selected_record["mean_exact_score"],
                -selected_record["subset_size"],
                -selected_record["mask"],
            )
            better = rank > current_rank
        if better:
            selected_record = record
            selected_direction = direction
        if mask == full_mask:
            full_record = record
            full_direction = {
                name: value.clone() for name, value in direction.items()
            }

    if (
        selected_direction is None
        or selected_record is None
        or full_direction is None
        or full_record is None
    ):
        raise ValueError("finite-radius subset family is incomplete")
    ordered = sorted(
        candidate_records,
        key=lambda item: (
            item["minimum_exact_score"],
            item["mean_exact_score"],
            -item["subset_size"],
            -item["mask"],
        ),
        reverse=True,
    )
    audit = {
        "certificate_names": names,
        "certificate_norms": certificate_norms,
        "gram_matrix": gram,
        "attempted_subsets": attempted,
        "feasible_subsets": len(candidate_records),
        "infeasible_subsets": attempted - len(candidate_records),
        "selection_rule": (
            "maximum_exact_minimum_score_then_mean_score_then_smaller_subset_"
            "then_lower_bitmask"
        ),
        "selected": selected_record,
        "full_certificate_control": full_record,
        "selected_margin_gain_over_full": float(
            selected_record["minimum_exact_score"]
            - full_record["minimum_exact_score"]
        ),
        "baseline": {
            **baseline_audit,
            "displacement_norm": baseline_norm,
            "scores": displacement_scores(
                certificates, baseline_displacement, baseline_norm
            )["scores"],
        },
        "top_candidates": ordered[: min(top_k, len(ordered))],
        "budget_norm": budget_norm,
        "auxiliary_scale": auxiliary_scale,
    }
    return selected_direction, full_direction, audit


def finite_step_loss_metrics(
    initial_losses: dict[str, float],
    final_losses: dict[str, float],
    diagnostic_gradients: dict[str, dict[str, Any]],
    displacement: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    if (
        set(initial_losses) != set(final_losses)
        or set(initial_losses) != set(diagnostic_gradients)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise ValueError("finite-step loss inputs differ or are invalid")
    displacement_norm = gradient_math.gradient_norm(displacement)
    if not math.isfinite(displacement_norm) or displacement_norm <= 0.0:
        raise ValueError("finite-step displacement is invalid")
    objectives = {}
    for name in initial_losses:
        predicted = learning_rate * gradient_math.gradient_dot(
            diagnostic_gradients[name], displacement
        )
        actual = float(initial_losses[name]) - float(final_losses[name])
        remainder = predicted - actual
        objectives[name] = {
            "initial_loss": float(initial_losses[name]),
            "final_loss": float(final_losses[name]),
            "predicted_first_order_reduction": predicted,
            "actual_loss_reduction": actual,
            "actual_to_predicted_ratio": actual / predicted
            if predicted != 0.0
            else None,
            "taylor_remainder": remainder,
            "empirical_positive_curvature_coefficient": (
                2.0 * max(0.0, remainder)
                / (learning_rate * learning_rate * displacement_norm**2)
            ),
        }
    return {
        "learning_rate": learning_rate,
        "displacement_norm": displacement_norm,
        "objectives": objectives,
    }
