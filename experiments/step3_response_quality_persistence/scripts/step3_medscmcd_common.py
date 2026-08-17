"""Minimum-effective-dose split-conformal MCD geometry."""

from __future__ import annotations

import math
from typing import Any

import step3_frsc_common as frsc
import step3_jcrmcd_common as path_geometry
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def mean_certificate_gradients(
    units: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if not units:
        raise ValueError("at least one gradient unit is required")
    certificates = tuple(units[0])
    if len(certificates) < 2 or any(tuple(unit) != certificates for unit in units):
        raise ValueError("gradient-unit certificate registries differ")
    result = {}
    for certificate in certificates:
        names = set(units[0][certificate])
        if not names or any(set(unit[certificate]) != names for unit in units):
            raise ValueError("gradient-unit tensor registries differ")
        result[certificate] = {}
        for name in names:
            value = units[0][certificate][name].clone()
            for unit in units[1:]:
                value.add_(unit[certificate][name])
            value.mul_(1.0 / len(units))
            result[certificate][name] = value
    return result


def score_path_panels_memory_efficient(
    raw_baseline: dict[str, Any],
    raw_direction: dict[str, Any],
    target_direction: dict[str, Any],
    certificate_panels: dict[str, dict[str, dict[str, Any]]],
    fractions: list[float] | tuple[float, ...],
    budget_norm: float,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    reference_norm: float,
    clip_epsilon: float = 1e-6,
    tolerance: float = 1e-9,
) -> dict[str, list[dict[str, Any]]]:
    """Score one shared path without materializing normalized-gradient copies."""
    values = tuple(float(value) for value in fractions)
    if (
        len(values) < 2
        or values[0] != 0.0
        or values[-1] != 1.0
        or any(left >= right for left, right in zip(values, values[1:]))
        or not certificate_panels
        or not math.isfinite(reference_norm)
        or reference_norm <= 0.0
    ):
        raise ValueError("score-path inputs are invalid")
    panel_names = list(certificate_panels)
    certificate_names = tuple(certificate_panels[panel_names[0]])
    if len(certificate_names) < 2 or any(
        tuple(panel) != certificate_names for panel in certificate_panels.values()
    ):
        raise ValueError("score-path certificate panels differ")
    norms = {}
    for panel_name, panel in certificate_panels.items():
        panel_norms = {
            name: gradient_math.gradient_norm(gradient)
            for name, gradient in panel.items()
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in panel_norms.values()):
            raise ValueError("score-path certificate gradient is degenerate")
        norms[panel_name] = panel_norms

    raw_displacement, raw_map = frsc.realized_displacement(
        raw_baseline,
        raw_direction,
        auxiliary_scale,
        max_norm,
        adam_epsilon,
        clip_epsilon,
    )
    raw_norm = gradient_math.gradient_norm(raw_displacement)
    records: dict[str, list[dict[str, Any]]] = {
        panel_name: [] for panel_name in panel_names
    }
    for fraction in values:
        direction = path_geometry.path_direction(
            raw_direction, target_direction, fraction, budget_norm, tolerance
        )
        if fraction == 0.0:
            displacement = raw_displacement
            map_audit = raw_map
        else:
            displacement, map_audit = frsc.realized_displacement(
                raw_baseline,
                direction,
                auxiliary_scale,
                max_norm,
                adam_epsilon,
                clip_epsilon,
            )
        difference = gradient_ops.add_gradients(
            displacement, raw_displacement, right_scale=-1.0
        )
        common = {
            "fraction": fraction,
            "displacement_norm_fraction": (
                gradient_math.gradient_norm(displacement) / reference_norm
            ),
            "distance_from_raw_fraction": (
                gradient_math.gradient_norm(difference) / reference_norm
            ),
            "candidate_clipping_active": bool(map_audit["clipping_active"]),
            "candidate_clip_coefficient": float(map_audit["clip_coefficient"]),
            "candidate_raw_norm": float(map_audit["raw_norm"]),
            "direction_budget_relative_error": abs(
                gradient_math.gradient_norm(direction) - budget_norm
            )
            / budget_norm,
        }
        for panel_name, panel in certificate_panels.items():
            scores = {
                name: float(
                    gradient_math.gradient_dot(gradient, displacement)
                    / norms[panel_name][name]
                    / reference_norm
                )
                for name, gradient in panel.items()
            }
            records[panel_name].append(
                {
                    **common,
                    "scores": scores,
                    "minimum_score": min(scores.values()),
                    "mean_score": sum(scores.values()) / len(scores),
                }
            )
        del direction, displacement, difference
    del raw_displacement
    if any(
        abs(panel[0]["displacement_norm_fraction"] - raw_norm / reference_norm)
        > 1e-9
        for panel in records.values()
    ):
        raise ValueError("raw path endpoint drifted")
    return records


def select_minimum_effective_candidate(
    pooled_records: list[dict[str, Any]],
    fit_records: list[list[dict[str, Any]]],
    protected_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
    retention_floor: float,
    target_gain_floor: float,
    target_score_floor: float,
    tolerance: float = 1e-12,
) -> tuple[int, dict[str, Any]]:
    protected = tuple(protected_names)
    targets = tuple(target_names)
    if (
        len(pooled_records) < 2
        or len(fit_records) < 2
        or any(len(panel) != len(pooled_records) for panel in fit_records)
        or not protected
        or not targets
        or set(protected).intersection(targets)
        or not 0.0 < retention_floor <= 1.0
        or not math.isfinite(target_gain_floor)
        or target_gain_floor <= 0.0
        or not math.isfinite(target_score_floor)
        or target_score_floor <= 0.0
        or tolerance <= 0.0
    ):
        raise ValueError("minimum-effective candidate inputs are invalid")
    names = set(pooled_records[0]["scores"])
    if set(protected).union(targets) != names:
        raise ValueError("minimum-effective certificate partition differs")
    fractions = [float(record["fraction"]) for record in pooled_records]
    if fractions[0] != 0.0 or any(
        left >= right for left, right in zip(fractions, fractions[1:])
    ):
        raise ValueError("minimum-effective path fractions differ")
    for panel in [pooled_records, *fit_records]:
        if [float(record["fraction"]) for record in panel] != fractions or any(
            set(record["scores"]) != names
            or any(not math.isfinite(float(value)) for value in record["scores"].values())
            for record in panel
        ):
            raise ValueError("minimum-effective score panels differ")

    raw_scores = {
        name: [float(panel[0]["scores"][name]) for panel in fit_records]
        for name in names
    }
    raw_scores_positive = all(
        value > 0.0 for values in raw_scores.values() for value in values
    )
    raw_target_floor = min(
        float(panel[0]["scores"][name]) for panel in fit_records for name in targets
    )
    candidates = []
    for index, pooled in enumerate(pooled_records):
        protected_retentions = {
            name: min(
                float(panel[index]["scores"][name])
                / float(panel[0]["scores"][name])
                for panel in fit_records
            )
            if raw_scores_positive
            else None
            for name in protected
        }
        target_floor = min(
            float(panel[index]["scores"][name])
            for panel in fit_records
            for name in targets
        )
        target_gain = target_floor - raw_target_floor
        minimum_retention = (
            min(float(value) for value in protected_retentions.values())
            if raw_scores_positive
            else None
        )
        eligible = (
            index > 0
            and raw_scores_positive
            and minimum_retention is not None
            and minimum_retention >= retention_floor - tolerance
            and target_floor >= target_score_floor - tolerance
            and target_gain >= target_gain_floor - tolerance
        )
        candidates.append(
            {
                "index": index,
                "fraction": float(pooled["fraction"]),
                "eligible": eligible,
                "fit_protected_retentions": protected_retentions,
                "minimum_fit_protected_retention": minimum_retention,
                "fit_target_score_floor": target_floor,
                "fit_target_gain_over_raw": target_gain,
                "pooled_target_score_floor": min(
                    float(pooled["scores"][name]) for name in targets
                ),
                "pooled_minimum_score": float(pooled["minimum_score"]),
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    selected = min(eligible, key=lambda item: item["fraction"]) if eligible else candidates[0]
    return int(selected["index"]), {
        "raw_fit_scores_positive": raw_scores_positive,
        "raw_fit_target_score_floor": raw_target_floor,
        "attempted_candidates": len(candidates),
        "eligible_candidates": len(eligible),
        "ineligible_candidates": len(candidates) - len(eligible),
        "retention_floor": retention_floor,
        "target_gain_floor": target_gain_floor,
        "target_score_floor": target_score_floor,
        "selected": selected,
        "raw_anchor": candidates[0],
        "candidates": candidates,
        "selection_rule": (
            "smallest_positive_fraction_meeting_joint_fit_protected_retention_"
            "and_fit_target_score_and_gain_floors"
        ),
        "calibration_scores_used_for_selection": False,
        "diagnostic_scores_used_for_selection": False,
    }


def joint_protected_nonconformity(
    raw_scores: dict[str, float],
    candidate_scores: dict[str, float],
    protected_names: list[str] | tuple[str, ...],
    retention_floor: float,
) -> tuple[float, dict[str, float]]:
    protected = tuple(protected_names)
    if (
        not protected
        or set(raw_scores) != set(candidate_scores)
        or not set(protected).issubset(raw_scores)
        or not 0.0 < retention_floor <= 1.0
        or any(not math.isfinite(float(value)) for value in raw_scores.values())
        or any(not math.isfinite(float(value)) for value in candidate_scores.values())
        or any(float(raw_scores[name]) <= 0.0 for name in protected)
    ):
        raise ValueError("protected nonconformity inputs are invalid")
    residuals = {
        name: retention_floor
        - float(candidate_scores[name]) / float(raw_scores[name])
        for name in protected
    }
    return max(residuals.values()), residuals


def split_conformal_upper_bound(
    nonconformities: list[float] | tuple[float, ...], alpha: float
) -> dict[str, Any]:
    values = [float(value) for value in nonconformities]
    if (
        not values
        or any(not math.isfinite(value) for value in values)
        or not math.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("split-conformal inputs are invalid")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    if rank > len(values):
        raise ValueError("calibration sample is too small for requested coverage")
    ordered = sorted(values)
    threshold = ordered[rank - 1]
    return {
        "calibration_units": len(values),
        "alpha": alpha,
        "order_statistic_rank": rank,
        "threshold": threshold,
        "finite_sample_marginal_coverage_lower_bound": rank / (len(values) + 1),
        "ordered_nonconformities": ordered,
        "all_calibration_units_protected": threshold <= 0.0,
        "assumption": (
            "candidate_fixed_before_calibration_and_calibration_plus_future_"
            "joint_nonconformities_exchangeable"
        ),
    }


def split_conformal_accept_or_fallback(
    nonconformities: list[float] | tuple[float, ...],
    alpha: float,
    acceptance_threshold: float,
    fallback_nonconformity: float,
) -> dict[str, Any]:
    """Certify a candidate-or-safe-fallback policy without a conditional claim."""
    if (
        not math.isfinite(acceptance_threshold)
        or not math.isfinite(fallback_nonconformity)
        or fallback_nonconformity > acceptance_threshold
    ):
        raise ValueError("accept-or-fallback policy inputs are invalid")
    result = split_conformal_upper_bound(nonconformities, alpha)
    candidate_accepted = float(result["threshold"]) <= acceptance_threshold
    coverage = float(result["finite_sample_marginal_coverage_lower_bound"])
    return {
        **result,
        "acceptance_threshold": acceptance_threshold,
        "fallback_nonconformity": fallback_nonconformity,
        "candidate_accepted": candidate_accepted,
        "policy_action": "candidate" if candidate_accepted else "fallback",
        "finite_sample_marginal_policy_protection_lower_bound": coverage,
        "finite_sample_marginal_policy_failure_upper_bound": 1.0 - coverage,
        "guarantee_scope": (
            "complete_accept_or_fallback_policy_over_calibration_plus_future_"
            "draws_under_exchangeability"
        ),
        "conditional_on_candidate_acceptance_coverage_claimed": False,
    }
