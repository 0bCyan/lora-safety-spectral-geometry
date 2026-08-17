"""Jackknife-calibrated convex-hull robust MCD geometry."""

from __future__ import annotations

import math
from typing import Any

import step3_frsc_common as frsc
import step3_mcd_common as mcd
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def convex_hull_distance_from_gram(
    support_gram: list[list[float]],
    target_dots: list[float],
    target_squared_norm: float = 1.0,
    tolerance: float = 1e-9,
) -> tuple[float, dict[str, Any]]:
    """Return the Euclidean distance from a target to a support convex hull."""
    size = len(support_gram)
    if (
        size == 0
        or len(target_dots) != size
        or any(len(row) != size for row in support_gram)
        or not math.isfinite(target_squared_norm)
        or target_squared_norm < 0.0
        or any(not math.isfinite(value) for row in support_gram for value in row)
        or any(not math.isfinite(value) for value in target_dots)
        or tolerance <= 0.0
    ):
        raise ValueError("convex-hull distance inputs are invalid")
    if any(
        abs(float(support_gram[left][right]) - float(support_gram[right][left]))
        > max(1e-8, 100.0 * tolerance)
        for left in range(size)
        for right in range(size)
    ):
        raise ValueError("support Gram matrix is not symmetric")
    difference_gram = [
        [
            float(support_gram[left][right])
            - float(target_dots[left])
            - float(target_dots[right])
            + float(target_squared_norm)
            for right in range(size)
        ]
        for left in range(size)
    ]
    weights, dual = mcd.minimum_norm_simplex_weights(
        difference_gram, tolerance
    )
    squared = sum(
        weights[left] * difference_gram[left][right] * weights[right]
        for left in range(size)
        for right in range(size)
    )
    squared = max(0.0, float(squared))
    return math.sqrt(squared), {
        "weights": weights,
        "squared_distance": squared,
        "difference_gram": difference_gram,
        **dual,
    }


def jackknife_convex_hull_radius(
    support_gram: list[list[float]], tolerance: float = 1e-9
) -> tuple[float, dict[str, Any]]:
    """Calibrate a radius from leave-one-support convex-hull residuals."""
    size = len(support_gram)
    if (
        size < 3
        or any(len(row) != size for row in support_gram)
        or tolerance <= 0.0
        or any(not math.isfinite(value) for row in support_gram for value in row)
    ):
        raise ValueError("jackknife calibration requires at least three supports")
    residuals = []
    for heldout in range(size):
        retained = [index for index in range(size) if index != heldout]
        subgram = [
            [support_gram[left][right] for right in retained]
            for left in retained
        ]
        target_dots = [support_gram[index][heldout] for index in retained]
        distance, audit = convex_hull_distance_from_gram(
            subgram,
            target_dots,
            float(support_gram[heldout][heldout]),
            tolerance,
        )
        residuals.append(
            {
                "heldout_support": heldout,
                "distance": distance,
                "retained_supports": retained,
                "projection_weights": audit["weights"],
                "kkt_residual": audit["kkt_residual"],
            }
        )
    radius = max(item["distance"] for item in residuals)
    return radius, {
        "radius": radius,
        "calibration_rule": (
            "maximum_leave_one_support_distance_to_remaining_convex_hull"
        ),
        "residuals": residuals,
    }


def path_direction(
    raw_direction: dict[str, Any],
    target_direction: dict[str, Any],
    fraction: float,
    budget_norm: float,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    if (
        set(raw_direction) != set(target_direction)
        or not math.isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
        or not math.isfinite(budget_norm)
        or budget_norm <= 0.0
    ):
        raise ValueError("JCR-MCD path inputs are invalid")
    blended = gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(raw_direction, 1.0 - fraction),
        target_direction,
        right_scale=fraction,
    )
    norm = gradient_math.gradient_norm(blended)
    if not math.isfinite(norm) or norm <= tolerance:
        raise ValueError("JCR-MCD path direction is degenerate")
    return gradient_ops.scaled_gradient(blended, budget_norm / norm)


def endpoint_directions(
    certificates: dict[str, dict[str, Any]],
    target_names: list[str] | tuple[str, ...],
    budget_norm: float,
    tolerance: float = 1e-9,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    targets = tuple(target_names)
    if not targets or not set(targets).issubset(certificates):
        raise ValueError("target endpoint certificate set is invalid")
    units, norms = frsc.normalized_certificates(certificates)
    names = list(units)
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units.values()]
        for left in units.values()
    ]
    raw_direction, raw_dual = mcd.minimax_common_direction(
        certificates, budget_norm, tolerance
    )
    target_direction, target_dual = mcd.minimax_common_direction(
        {name: certificates[name] for name in targets}, budget_norm, tolerance
    )
    return raw_direction, target_direction, {
        "certificate_names": names,
        "certificate_norms": norms,
        "gram_matrix": gram,
        "raw_endpoint": raw_dual,
        "target_endpoint": target_dual,
        "target_certificates": list(targets),
        "budget_norm": budget_norm,
    }


def score_path_panel(
    raw_baseline: dict[str, Any],
    raw_direction: dict[str, Any],
    target_direction: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    fractions: list[float] | tuple[float, ...],
    budget_norm: float,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    reference_norm: float,
    clip_epsilon: float = 1e-6,
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    return score_path_panels(
        raw_baseline,
        raw_direction,
        target_direction,
        {"single": certificates},
        fractions,
        budget_norm,
        auxiliary_scale,
        max_norm,
        adam_epsilon,
        reference_norm,
        clip_epsilon,
        tolerance,
    )["single"]


def score_path_panels(
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
    """Score several certificate panels on bit-identical path displacements."""
    values = tuple(float(value) for value in fractions)
    if (
        len(values) < 2
        or values[0] != 0.0
        or values[-1] != 1.0
        or any(left >= right for left, right in zip(values, values[1:]))
    ):
        raise ValueError("JCR-MCD path fractions are invalid")
    if not certificate_panels:
        raise ValueError("JCR-MCD certificate panels are empty")
    panel_names = list(certificate_panels)
    certificate_names = set(certificate_panels[panel_names[0]])
    if not certificate_names or any(
        set(panel) != certificate_names for panel in certificate_panels.values()
    ):
        raise ValueError("JCR-MCD certificate panels differ")
    normalized = {
        panel_name: frsc.normalized_certificates(panel)[0]
        for panel_name, panel in certificate_panels.items()
    }
    raw_displacement, raw_map_audit = frsc.realized_displacement(
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
        direction = path_direction(
            raw_direction, target_direction, fraction, budget_norm, tolerance
        )
        if fraction == 0.0:
            displacement = raw_displacement
            map_audit = raw_map_audit
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
        displacement_norm = gradient_math.gradient_norm(displacement)
        distance_from_raw = gradient_math.gradient_norm(difference)
        common = {
            "fraction": fraction,
            "displacement_norm_fraction": float(displacement_norm / reference_norm),
            "distance_from_raw_fraction": float(distance_from_raw / reference_norm),
            "candidate_clipping_active": bool(map_audit["clipping_active"]),
            "candidate_clip_coefficient": float(map_audit["clip_coefficient"]),
            "candidate_raw_norm": float(map_audit["raw_norm"]),
            "direction_budget_relative_error": abs(
                gradient_math.gradient_norm(direction) - budget_norm
            )
            / budget_norm,
        }
        for panel_name in panel_names:
            scores = {
                name: float(
                    gradient_math.gradient_dot(unit, displacement)
                    / reference_norm
                )
                for name, unit in normalized[panel_name].items()
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
    for panel in records.values():
        if (
            abs(panel[0]["displacement_norm_fraction"] - raw_norm / reference_norm)
            > 1e-9
        ):
            raise ValueError("raw path endpoint drifted across exact-map evaluations")
    return records


def select_robust_path_candidate(
    center_records: list[dict[str, Any]],
    support_records: list[list[dict[str, Any]]],
    radii: dict[str, float],
    protected_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
    retention_floor: float,
    tolerance: float = 1e-12,
) -> tuple[int, dict[str, Any]]:
    protected = tuple(protected_names)
    targets = tuple(target_names)
    if (
        not center_records
        or len(support_records) < 3
        or any(len(panel) != len(center_records) for panel in support_records)
        or not 0.0 < retention_floor <= 1.0
        or not protected
        or not targets
        or set(protected).intersection(targets)
    ):
        raise ValueError("robust path selection inputs are invalid")
    names = set(center_records[0]["scores"])
    if set(protected).union(targets) != names or set(radii) != names:
        raise ValueError("robust path certificate registry differs")
    if any(not math.isfinite(value) or value < 0.0 for value in radii.values()):
        raise ValueError("robust path radii are invalid")

    fractions = [float(record["fraction"]) for record in center_records]
    if fractions[0] != 0.0 or any(
        left >= right for left, right in zip(fractions, fractions[1:])
    ):
        raise ValueError("robust path center fractions are invalid")
    panels = [center_records, *support_records]
    scalar_fields = (
        "displacement_norm_fraction",
        "distance_from_raw_fraction",
        "minimum_score",
        "mean_score",
    )
    for panel in panels:
        if [float(record["fraction"]) for record in panel] != fractions:
            raise ValueError("robust path panel fractions differ")
        for record in panel:
            if set(record["scores"]) != names or any(
                not math.isfinite(float(value))
                for value in record["scores"].values()
            ):
                raise ValueError("robust path panel scores differ")
            if any(
                not math.isfinite(float(record[field]))
                for field in scalar_fields
            ) or any(
                float(record[field]) < 0.0
                for field in (
                    "displacement_norm_fraction",
                    "distance_from_raw_fraction",
                )
            ):
                raise ValueError("robust path panel metrics are invalid")

    raw = center_records[0]
    raw_support_scores = {
        name: [float(panel[0]["scores"][name]) for panel in support_records]
        for name in names
    }
    raw_lower = {
        name: min(raw_support_scores[name])
        - float(radii[name]) * float(raw["displacement_norm_fraction"])
        for name in names
    }
    if any(
        value <= 0.0
        for values in raw_support_scores.values()
        for value in values
    ):
        raise ValueError("raw support scores must be positive")
    uncertainty_valid = all(raw_lower[name] > 0.0 for name in names)
    raw_norm = float(raw["displacement_norm_fraction"])
    candidates = []
    for index, center in enumerate(center_records):
        robust_target = {
            name: min(
                float(panel[index]["scores"][name])
                for panel in support_records
            )
            - float(radii[name]) * float(center["displacement_norm_fraction"])
            for name in targets
        }
        robust_common = {
            name: min(
                float(panel[index]["scores"][name])
                for panel in support_records
            )
            - float(radii[name]) * float(center["displacement_norm_fraction"])
            for name in names
        }
        protected_margins = {}
        direct_retentions = {}
        residual_squared = (
            retention_floor
            * float(center["distance_from_raw_fraction"]) ** 2
            + (1.0 - retention_floor)
            * float(center["displacement_norm_fraction"]) ** 2
            - retention_floor * (1.0 - retention_floor) * raw_norm**2
        )
        if residual_squared < -max(tolerance, 1e-12):
            raise ValueError("retention residual norm is numerically invalid")
        retention_residual_norm = math.sqrt(max(0.0, residual_squared))
        for name in protected:
            support_residuals = [
                float(panel[index]["scores"][name])
                - retention_floor * float(panel[0]["scores"][name])
                for panel in support_records
            ]
            protected_margins[name] = (
                min(support_residuals)
                - float(radii[name]) * retention_residual_norm
            )
            direct_retentions[name] = min(
                float(panel[index]["scores"][name])
                / float(panel[0]["scores"][name])
                for panel in support_records
            )
        feasible = uncertainty_valid and all(
            value >= -tolerance for value in protected_margins.values()
        )
        target_center = {
            name: float(center["scores"][name]) for name in targets
        }
        candidates.append(
            {
                "index": index,
                "fraction": float(center["fraction"]),
                "feasible": feasible,
                "robust_target_scores": robust_target,
                "minimum_robust_target_score": min(robust_target.values()),
                "minimum_robust_common_score": min(robust_common.values()),
                "protected_robust_margins": protected_margins,
                "minimum_protected_robust_margin": min(
                    protected_margins.values()
                ),
                "retention_residual_norm_fraction": retention_residual_norm,
                "direct_support_retentions": direct_retentions,
                "minimum_direct_support_retention": min(
                    direct_retentions.values()
                ),
                "minimum_center_target_score": min(target_center.values()),
                "mean_center_target_score": sum(target_center.values())
                / len(target_center),
                "minimum_center_score": float(center["minimum_score"]),
            }
        )

    feasible = [item for item in candidates if item["feasible"]]
    if not feasible:
        selected = candidates[0]
    else:
        selected = max(
            feasible,
            key=lambda item: (
                item["minimum_robust_target_score"],
                item["minimum_robust_common_score"],
                item["minimum_center_target_score"],
                item["mean_center_target_score"],
                item["minimum_protected_robust_margin"],
                -item["fraction"],
            ),
        )
    raw_candidate = candidates[0]
    ordered = sorted(
        feasible,
        key=lambda item: (
            item["minimum_robust_target_score"],
            item["minimum_robust_common_score"],
            item["minimum_center_target_score"],
            item["mean_center_target_score"],
            item["minimum_protected_robust_margin"],
            -item["fraction"],
        ),
        reverse=True,
    )
    return int(selected["index"]), {
        "uncertainty_set_valid": uncertainty_valid,
        "raw_robust_lower_scores": raw_lower,
        "attempted_candidates": len(candidates),
        "feasible_candidates": len(feasible),
        "infeasible_candidates": len(candidates) - len(feasible),
        "retention_floor": retention_floor,
        "radii": radii,
        "raw_anchor": raw_candidate,
        "selected": selected,
        "selected_robust_target_gain_over_raw": float(
            selected["minimum_robust_target_score"]
            - raw_candidate["minimum_robust_target_score"]
        ),
        "candidates": candidates,
        "top_feasible_candidates": ordered[:8],
        "selection_rule": (
            "maximize_worst_target_lower_bound_over_jackknife_expanded_"
            "support_hull_subject_to_four_protected_retention_bounds"
        ),
    }
