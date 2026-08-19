"""Run the memory-bounded BARP-MCD Q1 qualification retry."""

from __future__ import annotations

import argparse
import math
from typing import Any

import step3_barp_mcd_training as base
import step3_frsc_common as frsc
import step3_pfrsc_common as pfrsc
import step3_rpfrsc_p0 as rp
import step3_ramp_common as gradient_math


RETRY_STATUS = "prospective_before_any_BARP_MCD_Q1R_optimizer_step"
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_q1r/protocol/barp_mcd_q1r_protocol.json"
)
BASE_SELECT_ROUTE = base.select_route


def _direction_for_mask(
    units: dict[str, dict[str, Any]],
    gram: list[list[float]],
    mask: int,
    budget_norm: float,
    tolerance: float,
) -> dict[str, Any]:
    indices = [index for index in range(len(units)) if mask & (1 << index)]
    direction, _ = frsc._subset_direction(
        units, gram, indices, budget_norm, tolerance
    )
    return direction


def enumerate_memory_bounded_homotopy(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    budget_norm: float,
    safety_mix: float,
    max_norm: float,
    protected_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
    protected_retention: float,
    homotopy_fractions: list[float] | tuple[float, ...],
    source_top_k: int,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
    serialized_top_k: int = 8,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protected = tuple(protected_names)
    targets = tuple(target_names)
    fractions = tuple(float(value) for value in homotopy_fractions)
    names = tuple(certificates)
    if (
        protected != rp.PROTECTED_CERTIFICATES
        or targets != rp.TARGET_CERTIFICATES
        or fractions != rp.HOMOTOPY_FRACTIONS
        or source_top_k != rp.SOURCE_TOP_K
        or not math.isfinite(protected_retention)
        or not 0.0 < protected_retention <= 1.0
    ):
        raise base.BarpTrainingValidationError("Q1R finite family drift")
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
    baseline_displacement, baseline_audit = base.current_adamw_displacement(
        parameters, optimizer, raw_baseline, max_norm, clip_epsilon
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise base.BarpTrainingValidationError("Q1R baseline is degenerate")
    raw_record = base._candidate_record(
        full_direction,
        full_source,
        0.0,
        raw_baseline,
        parameters,
        optimizer,
        units,
        {name: 1.0 for name in names},
        reference_norm,
        safety_mix,
        max_norm,
        clip_epsilon,
        protected,
        targets,
    )
    raw_scores = dict(raw_record["exact_scores"])
    if any(value <= 0.0 for value in raw_scores.values()):
        raise base.BarpTrainingValidationError("Q1R raw anchor lacks common descent")
    raw_record["protected_score_retention"] = {name: 1.0 for name in protected}
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
        source_records.append(
            base._candidate_record(
                direction,
                source,
                1.0,
                raw_baseline,
                parameters,
                optimizer,
                units,
                raw_scores,
                reference_norm,
                safety_mix,
                max_norm,
                clip_epsilon,
                protected,
                targets,
            )
        )
        del direction

    screened = sorted(source_records, key=base._source_rank, reverse=True)[
        : min(source_top_k, len(source_records))
    ]
    endpoint_by_mask = {record["source_mask"]: record for record in source_records}
    candidates = [raw_record]
    protected_candidates = [raw_record]
    interior_failures = 0
    for source in screened:
        endpoint = _direction_for_mask(
            units,
            gram,
            int(source["source_mask"]),
            budget_norm,
            dual_tolerance,
        )
        source_descriptor = {
            "mask": source["source_mask"],
            "subset": source["source_subset"],
            "subset_size": source["source_subset_size"],
            "dual_margin": source["source_dual_margin"],
            "dual_kkt_residual": source["source_dual_kkt_residual"],
        }
        for fraction in fractions[1:]:
            if fraction == 1.0:
                candidate = endpoint_by_mask[int(source["source_mask"])]
            else:
                try:
                    direction = pfrsc._homotopy_direction(
                        full_direction,
                        endpoint,
                        fraction,
                        budget_norm,
                        dual_tolerance,
                    )
                except ValueError:
                    interior_failures += 1
                    continue
                candidate = base._candidate_record(
                    direction,
                    source_descriptor,
                    fraction,
                    raw_baseline,
                    parameters,
                    optimizer,
                    units,
                    raw_scores,
                    reference_norm,
                    safety_mix,
                    max_norm,
                    clip_epsilon,
                    protected,
                    targets,
                )
                del direction
            candidates.append(candidate)
            if candidate["minimum_protected_retention"] + 1e-12 >= protected_retention:
                protected_candidates.append(candidate)
        del endpoint

    selected_record = max(protected_candidates, key=base._selection_rank)
    if selected_record is raw_record:
        selected_direction = {name: value.clone() for name, value in full_direction.items()}
    else:
        endpoint = _direction_for_mask(
            units,
            gram,
            int(selected_record["source_mask"]),
            budget_norm,
            dual_tolerance,
        )
        selected_direction = pfrsc._homotopy_direction(
            full_direction,
            endpoint,
            float(selected_record["homotopy_fraction"]),
            budget_norm,
            dual_tolerance,
        )
        del endpoint
    ordered = sorted(protected_candidates, key=base._selection_rank, reverse=True)
    audit = {
        "certificate_names": list(names),
        "certificate_norms": certificate_norms,
        "protected_certificates": list(protected),
        "target_certificates": list(targets),
        "attempted_source_subsets": full_mask,
        "feasible_source_subsets": len(source_records) + 1,
        "infeasible_source_subsets": infeasible_subsets,
        "source_top_k": source_top_k,
        "screened_source_count": len(screened),
        "attempted_homotopy_candidates": 1
        + len(screened) * (len(fractions) - 1),
        "feasible_homotopy_candidates": len(candidates),
        "infeasible_homotopy_interiors": interior_failures,
        "exact_map_evaluations": len(source_records)
        + 1
        + len(screened) * (len(fractions) - 2)
        - interior_failures,
        "protected_retention_floor": protected_retention,
        "protected_feasible_candidates": len(protected_candidates),
        "raw_anchor": raw_record,
        "selected": selected_record,
        "selected_target_gain_over_raw": float(
            selected_record["minimum_target_exact_score"]
            - raw_record["minimum_target_exact_score"]
        ),
        "top_protected_candidates": ordered[: min(serialized_top_k, len(ordered))],
        "screened_source_endpoints": screened,
        "baseline": {
            **baseline_audit,
            "displacement_norm": reference_norm,
            "scores": base._score_displacement(
                units, baseline_displacement, reference_norm
            )["scores"],
        },
        "budget_norm": budget_norm,
        "current_optimizer_completed_steps": int(
            baseline_audit["completed_optimizer_steps"]
        ),
        "endpoint_tensor_policy": "score_delete_then_reconstruct_screened_masks",
        "maximum_retained_endpoint_directions": 1,
    }
    del baseline_displacement
    return selected_direction, full_direction, audit


def select_retry_route(checks: dict[str, bool]) -> str:
    route = BASE_SELECT_ROUTE(checks)
    return route.replace("BARP_MCD_Q1", "BARP_MCD_Q1R")


def install_retry() -> None:
    base.STATUS = RETRY_STATUS
    base.enumerate_current_state_homotopy = enumerate_memory_bounded_homotopy
    base.select_route = select_retry_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_retry()
    path = base.resolve_repo_path(args.protocol)
    if args.command == "preflight":
        base.preflight(path)
    elif args.command == "run":
        base.run(path)
    else:
        base.verify_run(path)


if __name__ == "__main__":
    main()
