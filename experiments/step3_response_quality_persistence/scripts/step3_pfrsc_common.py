"""Protected finite-radius homotopy search anchored at full-set raw MCD."""

from __future__ import annotations

import math
from typing import Any

import step3_frsc_common as frsc
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def _homotopy_direction(
    anchor: dict[str, Any],
    endpoint: dict[str, Any],
    fraction: float,
    budget_norm: float,
    tolerance: float,
) -> dict[str, Any]:
    if (
        set(anchor) != set(endpoint)
        or not math.isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
    ):
        raise ValueError("homotopy direction inputs differ or are invalid")
    blended = gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(anchor, 1.0 - fraction),
        gradient_ops.scaled_gradient(endpoint, fraction),
    )
    norm = gradient_math.gradient_norm(blended)
    if not math.isfinite(norm) or norm <= tolerance:
        raise ValueError("homotopy direction is degenerate")
    return gradient_ops.scaled_gradient(blended, budget_norm / norm)


def _score_record(
    direction: dict[str, Any],
    source: dict[str, Any],
    fraction: float,
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    protected_names: tuple[str, ...],
    target_names: tuple[str, ...],
    raw_scores: dict[str, float],
    reference_norm: float,
    budget_norm: float,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    clip_epsilon: float,
) -> dict[str, Any]:
    source_mask = int(
        source["mask"] if "mask" in source else source["source_mask"]
    )
    source_subset = list(
        source["subset"] if "subset" in source else source["source_subset"]
    )
    source_subset_size = int(
        source["subset_size"]
        if "subset_size" in source
        else source["source_subset_size"]
    )
    source_dual_margin = float(
        source["dual_margin"]
        if "dual_margin" in source
        else source["source_dual_margin"]
    )
    source_dual_kkt = float(
        source["dual_kkt_residual"]
        if "dual_kkt_residual" in source
        else source["source_dual_kkt_residual"]
    )
    displacement, map_audit = frsc.realized_displacement(
        raw_baseline,
        direction,
        auxiliary_scale,
        max_norm,
        adam_epsilon,
        clip_epsilon,
    )
    scored = frsc.displacement_scores(certificates, displacement, reference_norm)
    scores = {name: float(value) for name, value in scored["scores"].items()}
    protected_ratios = {
        name: scores[name] / raw_scores[name] for name in protected_names
    }
    target_scores = {name: scores[name] for name in target_names}
    return {
        "source_mask": source_mask,
        "source_subset": source_subset,
        "source_subset_size": source_subset_size,
        "source_dual_margin": source_dual_margin,
        "source_dual_kkt_residual": source_dual_kkt,
        "homotopy_fraction": float(fraction),
        "minimum_exact_score": float(scored["minimum_score"]),
        "mean_exact_score": float(scored["mean_score"]),
        "minimum_target_exact_score": min(target_scores.values()),
        "mean_target_exact_score": sum(target_scores.values()) / len(target_scores),
        "minimum_protected_retention": min(protected_ratios.values()),
        "protected_score_retention": protected_ratios,
        "exact_scores": scores,
        "displacement_norm_fraction_of_baseline": float(
            scored["displacement_norm"] / reference_norm
        ),
        "candidate_raw_norm": float(map_audit["raw_norm"]),
        "candidate_clip_coefficient": float(map_audit["clip_coefficient"]),
        "candidate_clipping_active": bool(map_audit["clipping_active"]),
        "direction_budget_relative_error": abs(
            gradient_math.gradient_norm(direction) - budget_norm
        )
        / budget_norm,
    }


def _source_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["minimum_target_exact_score"],
        record["minimum_exact_score"],
        record["mean_target_exact_score"],
        -record["source_subset_size"],
        -record["source_mask"],
    )


def _selection_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["minimum_target_exact_score"],
        record["minimum_exact_score"],
        record["mean_target_exact_score"],
        record["minimum_protected_retention"],
        -record["homotopy_fraction"],
        -record["source_subset_size"],
        -record["source_mask"],
    )


def enumerate_protected_homotopy(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    budget_norm: float,
    auxiliary_scale: float,
    max_norm: float,
    adam_epsilon: float,
    protected_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
    protected_retention: float,
    homotopy_fractions: list[float] | tuple[float, ...],
    source_top_k: int,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
    serialized_top_k: int = 8,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Screen subset endpoints, then optimize a protected raw-anchored family.

    The returned selected direction is the exact-map maximizer of the target
    bottleneck among protected candidates in the registered finite family.
    The full-certificate raw MCD anchor is always a member of that family.
    """
    protected = tuple(protected_names)
    targets = tuple(target_names)
    fractions = tuple(float(value) for value in homotopy_fractions)
    names = tuple(certificates)
    if (
        not math.isfinite(budget_norm)
        or budget_norm <= 0.0
        or not math.isfinite(auxiliary_scale)
        or auxiliary_scale <= 0.0
        or not 0.0 < protected_retention <= 1.0
        or source_top_k <= 0
        or serialized_top_k <= 0
        or not protected
        or not targets
        or set(protected).intersection(targets)
        or set(protected).union(targets) != set(names)
        or len(set(protected)) != len(protected)
        or len(set(targets)) != len(targets)
        or len(fractions) < 2
        or fractions[0] != 0.0
        or fractions[-1] != 1.0
        or any(not 0.0 <= value <= 1.0 for value in fractions)
        or any(left >= right for left, right in zip(fractions, fractions[1:]))
    ):
        raise ValueError("protected homotopy inputs are invalid")

    units, certificate_norms = frsc.normalized_certificates(certificates)
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units.values()]
        for left in units.values()
    ]
    full_mask = 2 ** len(names) - 1
    full_indices = list(range(len(names)))
    full_direction, full_dual = frsc._subset_direction(
        units, gram, full_indices, budget_norm, dual_tolerance
    )
    full_source = {
        "mask": full_mask,
        "subset": list(names),
        "subset_size": len(names),
        "dual_margin": float(full_dual["dual_margin"]),
        "dual_kkt_residual": float(full_dual["kkt_residual"]),
    }

    baseline_displacement, baseline_audit = frsc.realized_displacement(
        raw_baseline,
        None,
        0.0,
        max_norm,
        adam_epsilon,
        clip_epsilon,
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise ValueError("baseline displacement is degenerate")
    raw_record = _score_record(
        full_direction,
        full_source,
        0.0,
        raw_baseline,
        certificates,
        protected,
        targets,
        {name: 1.0 for name in names},
        reference_norm,
        budget_norm,
        auxiliary_scale,
        max_norm,
        adam_epsilon,
        clip_epsilon,
    )
    raw_scores = dict(raw_record["exact_scores"])
    if any(raw_scores[name] <= 0.0 for name in names):
        raise ValueError("raw anchor lacks strict common descent")
    raw_record["protected_score_retention"] = {
        name: 1.0 for name in protected
    }
    raw_record["minimum_protected_retention"] = 1.0

    source_records: list[dict[str, Any]] = []
    infeasible_subsets = 0
    for mask in range(1, full_mask):
        indices = [index for index in range(len(names)) if mask & (1 << index)]
        try:
            direction, dual = frsc._subset_direction(
                units, gram, indices, budget_norm, dual_tolerance
            )
        except ValueError:
            infeasible_subsets += 1
            continue
        source = {
            "mask": mask,
            "subset": [names[index] for index in indices],
            "subset_size": len(indices),
            "dual_margin": float(dual["dual_margin"]),
            "dual_kkt_residual": float(dual["kkt_residual"]),
        }
        endpoint = _score_record(
            direction,
            source,
            1.0,
            raw_baseline,
            certificates,
            protected,
            targets,
            raw_scores,
            reference_norm,
            budget_norm,
            auxiliary_scale,
            max_norm,
            adam_epsilon,
            clip_epsilon,
        )
        source_records.append(endpoint)
        del direction

    screened_sources = sorted(source_records, key=_source_rank, reverse=True)[
        : min(source_top_k, len(source_records))
    ]
    candidate_records = [raw_record]
    protected_candidates = [raw_record]
    interior_failures = 0
    endpoint_by_mask = {record["source_mask"]: record for record in source_records}
    for source in screened_sources:
        source_mask = int(source["source_mask"])
        source_indices = [
            index for index in range(len(names)) if source_mask & (1 << index)
        ]
        endpoint_direction, _ = frsc._subset_direction(
            units, gram, source_indices, budget_norm, dual_tolerance
        )
        for fraction in fractions[1:]:
            if fraction == 1.0:
                candidate = endpoint_by_mask[int(source["source_mask"])]
                direction = endpoint_direction
            else:
                try:
                    direction = _homotopy_direction(
                        full_direction,
                        endpoint_direction,
                        fraction,
                        budget_norm,
                        dual_tolerance,
                    )
                except ValueError:
                    interior_failures += 1
                    continue
                candidate = _score_record(
                    direction,
                    source,
                    fraction,
                    raw_baseline,
                    certificates,
                    protected,
                    targets,
                    raw_scores,
                    reference_norm,
                    budget_norm,
                    auxiliary_scale,
                    max_norm,
                    adam_epsilon,
                    clip_epsilon,
                )
            candidate_records.append(candidate)
            if candidate["minimum_protected_retention"] + 1e-12 >= protected_retention:
                protected_candidates.append(candidate)
        del endpoint_direction

    selected_record = max(protected_candidates, key=_selection_rank)
    if selected_record is raw_record:
        selected_direction = {
            name: value.clone() for name, value in full_direction.items()
        }
    else:
        selected_mask = int(selected_record["source_mask"])
        selected_indices = [
            index for index in range(len(names)) if selected_mask & (1 << index)
        ]
        endpoint, _ = frsc._subset_direction(
            units, gram, selected_indices, budget_norm, dual_tolerance
        )
        selected_direction = _homotopy_direction(
            full_direction,
            endpoint,
            float(selected_record["homotopy_fraction"]),
            budget_norm,
            dual_tolerance,
        )

    ordered_protected = sorted(
        protected_candidates, key=_selection_rank, reverse=True
    )
    audit = {
        "certificate_names": list(names),
        "protected_certificates": list(protected),
        "target_certificates": list(targets),
        "certificate_norms": certificate_norms,
        "gram_matrix": gram,
        "attempted_source_subsets": full_mask,
        "feasible_source_subsets": len(source_records) + 1,
        "infeasible_source_subsets": infeasible_subsets,
        "source_top_k": source_top_k,
        "screened_source_count": len(screened_sources),
        "homotopy_fractions": list(fractions),
        "attempted_homotopy_candidates": 1
        + len(screened_sources) * (len(fractions) - 1),
        "feasible_homotopy_candidates": len(candidate_records),
        "infeasible_homotopy_interiors": interior_failures,
        "exact_map_evaluations": len(source_records)
        + 1
        + len(screened_sources) * (len(fractions) - 2)
        - interior_failures,
        "protected_retention_floor": protected_retention,
        "protected_feasible_candidates": len(protected_candidates),
        "selection_rule": (
            "screen_endpoints_by_target_min_then_select_protected_homotopy_by_"
            "target_min_overall_min_target_mean_protected_retention_and_fixed_ties"
        ),
        "raw_anchor": raw_record,
        "selected": selected_record,
        "selected_target_gain_over_raw": float(
            selected_record["minimum_target_exact_score"]
            - raw_record["minimum_target_exact_score"]
        ),
        "selected_overall_margin_gain_over_raw": float(
            selected_record["minimum_exact_score"]
            - raw_record["minimum_exact_score"]
        ),
        "screened_source_endpoints": screened_sources,
        "top_protected_candidates": ordered_protected[
            : min(serialized_top_k, len(ordered_protected))
        ],
        "baseline": {
            **baseline_audit,
            "displacement_norm": reference_norm,
            "scores": frsc.displacement_scores(
                certificates, baseline_displacement, reference_norm
            )["scores"],
        },
        "budget_norm": budget_norm,
        "auxiliary_scale": auxiliary_scale,
    }
    return selected_direction, full_direction, audit
