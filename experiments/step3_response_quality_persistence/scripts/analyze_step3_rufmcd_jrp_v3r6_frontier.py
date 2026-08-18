"""Prove the relaxed angular incompatibility of the frozen JRP v3r6 ray."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import analyze_step3_rufmcd_v2_angle as angle


DEFAULT_ROOT = Path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r6"
)
DEFAULT_FIT_GEOMETRY = DEFAULT_ROOT / "run/rufmcd_p0_v3r6_fit_geometry.json"
DEFAULT_CALIBRATION_ROWS = (
    DEFAULT_ROOT / "run/rufmcd_p0_v3r6_calibration_rows.jsonl"
)
DEFAULT_ROUTE = DEFAULT_ROOT / "run/rufmcd_p0_v3r6_route.json"
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "analysis/rufmcd_p0_v3r6_relaxed_angle_impossibility.json"
)
ORIGINAL_FIT_FACETS = 84
OMITTED_REFLECTED_TARC_FACETS = 12


def _range(values: list[float]) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("directional support values must be finite and nonempty")
    return [min(values), max(values)]


def directional_support_audit(
    fit_geometry_path: Path, calibration_rows_path: Path
) -> dict[str, Any]:
    """Compare TARC support-function slopes on one frozen tangent."""
    fit_geometry = angle._read_json(fit_geometry_path)
    calibration_rows = angle._read_jsonl(calibration_rows_path)
    theta = float(fit_geometry["selected_theta"])
    sine = math.sin(theta)
    cosine = math.cos(theta)
    if theta <= 0.0 or sine <= 0.0:
        raise ValueError("directional support audit requires a positive angle")

    observed = []
    for unit_id, panel in fit_geometry["fit_score_panels"].items():
        raw_score = float(panel["raw"]["tarc"])
        if raw_score <= 0.0:
            raise ValueError("fit TARC raw score must be positive")
        observed.append(
            {
                "unit_id": unit_id,
                "beta": float(panel["tangent"]["tarc"]) / raw_score,
            }
        )

    full_scores = fit_geometry["tangent_audit"]["affine_solver"][
        "full_certificate_scores"
    ]
    reflected = [
        {
            "unit_id": name,
            "beta": (
                float(margin) - (cosine - angle.PROTECTED_RETENTION_FLOOR)
            )
            / sine,
        }
        for name, margin in full_scores.items()
        if name.endswith(":tarc:jackknife_reflection")
    ]
    if len(observed) != 12 or len(reflected) != 12:
        raise ValueError("unexpected JRP TARC vertex count")

    registered_betas = [item["beta"] for item in observed + reflected]
    registered_range = _range(registered_betas)
    calibration = []
    for row in calibration_rows:
        retention = float(row["protected_retentions"]["tarc"])
        beta = (retention - cosine) / sine
        calibration.append(
            {
                "unit_id": str(row["unit_id"]),
                "beta": beta,
                "retention_at_fixed_theta": retention,
                "outside_registered_projected_hull": (
                    beta < registered_range[0] - angle.NUMERIC_TOLERANCE
                    or beta > registered_range[1] + angle.NUMERIC_TOLERANCE
                ),
            }
        )

    outside = [
        item for item in calibration if item["outside_registered_projected_hull"]
    ]
    sign_reversals = [item for item in calibration if item["beta"] < 0.0]
    return {
        "definition": (
            "beta_j=<normalized_TARC_covector_j,tangent>/reference_norm, "
            "so retention_j(theta)=cos(theta)+beta_j*sin(theta)"
        ),
        "fit_observed_beta_range": _range(
            [item["beta"] for item in observed]
        ),
        "fit_reflected_beta_range": _range(
            [item["beta"] for item in reflected]
        ),
        "registered_vertex_beta_range": registered_range,
        "calibration_beta_range": _range(
            [item["beta"] for item in calibration]
        ),
        "calibration_units_outside_registered_projected_hull": outside,
        "outside_count": len(outside),
        "calibration_negative_beta_units": sign_reversals,
        "negative_beta_count": len(sign_reversals),
        "linear_separation_conclusion": (
            "A covector whose beta lies outside the registered projected "
            "interval cannot belong to the registered convex hull, because "
            "a linear functional maps a convex hull to the convex hull of "
            "its vertex values."
        ),
    }


def decorate(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the set-inclusion proof specific to the 96-facet JRP method."""
    counts = result["facet_counts"]
    analyzed_fit_facets = int(counts["fit_protected"]) + int(
        counts["fit_target"]
    )
    if analyzed_fit_facets != ORIGINAL_FIT_FACETS:
        raise ValueError("unexpected relaxed JRP fit-facet count")

    gap = result["angular_incompatibility_gap"]
    joint_exists = bool(result["joint_feasible_angle_exists"])
    strict_separation = (
        not joint_exists and gap is not None and float(gap) > angle.NUMERIC_TOLERANCE
    )
    result["name"] = (
        "step3_rufmcd_jrp_p0_v3r6_posthoc_relaxed_angle_impossibility_v1"
    )
    result["analysis_status"] = (
        "posthoc_mechanism_proof_not_authorized_for_selection"
    )
    result["relaxed_fit_proof"] = {
        "full_registered_fit_facets": (
            ORIGINAL_FIT_FACETS + OMITTED_REFLECTED_TARC_FACETS
        ),
        "analyzed_original_fit_facets": ORIGINAL_FIT_FACETS,
        "omitted_jackknife_reflected_tarc_facets": OMITTED_REFLECTED_TARC_FACETS,
        "relaxation_direction": (
            "omitting constraints makes the analyzed fit-feasible set a "
            "superset of the complete JRP fit-feasible set"
        ),
        "strict_interval_separation": strict_separation,
        "set_inclusion_conclusion": (
            "No angle on the frozen v3r6 raw-to-tangent ray can satisfy the "
            "complete JRP fit constraints and the observed calibration "
            "protected constraints simultaneously."
            if strict_separation
            else "The relaxed audit does not prove angular impossibility."
        ),
    }
    result["interpretation"] = {
        "dose_only_repair_excluded": strict_separation,
        "direction_change_required": strict_separation,
        "calibration_use": (
            "posthoc mechanism diagnosis only; calibration remains forbidden "
            "for candidate selection"
        ),
        "fresh_confirmation_claimed": False,
    }
    return result


def analyze(
    fit_geometry_path: Path,
    calibration_rows_path: Path,
    route_path: Path,
) -> dict[str, Any]:
    result = decorate(
        angle.analyze(fit_geometry_path, calibration_rows_path, route_path)
    )
    result["tarc_directional_support_audit"] = directional_support_audit(
        fit_geometry_path, calibration_rows_path
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-geometry", type=Path, default=DEFAULT_FIT_GEOMETRY)
    parser.add_argument(
        "--calibration-rows", type=Path, default=DEFAULT_CALIBRATION_ROWS
    )
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.fit_geometry, args.calibration_rows, args.route)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "angular_incompatibility_gap": result[
                    "angular_incompatibility_gap"
                ],
                "calibration_safe_prefix_upper_theta": result[
                    "calibration_safe_prefix_upper_theta"
                ],
                "first_fit_feasible_theta": result[
                    "first_fit_feasible_theta"
                ],
                "output": args.output.as_posix(),
                "strict_interval_separation": result["relaxed_fit_proof"][
                    "strict_interval_separation"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
