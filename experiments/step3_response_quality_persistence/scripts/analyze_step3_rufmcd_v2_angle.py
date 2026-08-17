"""Audit the continuous angle frontier of the frozen RUF-MCD v2 tangent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


DEFAULT_FIT_GEOMETRY = Path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/run/rufmcd_p0_v2_fit_geometry.json"
)
DEFAULT_CALIBRATION_ROWS = Path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/run/rufmcd_p0_v2_calibration_rows.jsonl"
)
DEFAULT_ROUTE = Path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/run/rufmcd_p0_v2_route.json"
)
DEFAULT_OUTPUT = Path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/analysis/"
    "rufmcd_p0_v2_angle_sensitivity.json"
)
PROTECTED_CERTIFICATES = ("risk", "uniform", "tarc", "continuation")
TARGET_CERTIFICATES = ("etarc", "branch_entropy", "branch_negative_gap")
PROTECTED_RETENTION_FLOOR = 0.95
TARGET_SCORE_FLOOR = 0.05
TARGET_GAIN_FLOOR = 0.002
NUMERIC_TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def score_at(raw_score: float, tangent_score: float, theta: float) -> float:
    return raw_score * math.cos(theta) + tangent_score * math.sin(theta)


def trig_roots(
    cosine_coefficient: float,
    sine_coefficient: float,
    threshold: float,
    lower: float,
    upper: float,
) -> list[float]:
    """Return every root of a*cos(theta)+b*sin(theta)=c on an interval."""
    amplitude = math.hypot(cosine_coefficient, sine_coefficient)
    if amplitude == 0.0 or abs(threshold) > amplitude + NUMERIC_TOLERANCE:
        return []
    ratio = min(1.0, max(-1.0, threshold / amplitude))
    phase = math.atan2(sine_coefficient, cosine_coefficient)
    delta = math.acos(ratio)
    period = 2.0 * math.pi
    roots: list[float] = []
    for base in (phase - delta, phase + delta):
        first = math.ceil((lower - base) / period - NUMERIC_TOLERANCE)
        last = math.floor((upper - base) / period + NUMERIC_TOLERANCE)
        for cycle in range(first, last + 1):
            root = base + cycle * period
            if lower - NUMERIC_TOLERANCE <= root <= upper + NUMERIC_TOLERANCE:
                roots.append(min(upper, max(lower, root)))
    return _unique_sorted(roots)


def _unique_sorted(values: list[float], tolerance: float = 1e-11) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _feasible_intervals(
    boundaries: list[float], predicate: Callable[[float], bool]
) -> list[dict[str, float]]:
    intervals: list[dict[str, float]] = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        if upper - lower <= NUMERIC_TOLERANCE:
            continue
        midpoint = 0.5 * (lower + upper)
        if not predicate(midpoint):
            continue
        if intervals and abs(intervals[-1]["upper"] - lower) <= 1e-11:
            intervals[-1]["upper"] = upper
        else:
            intervals.append({"lower": lower, "upper": upper})
    return intervals


def _facet(
    split: str,
    unit_id: str,
    certificate: str,
    kind: str,
    raw_score: float,
    tangent_score: float,
) -> dict[str, Any]:
    if not math.isfinite(raw_score) or not math.isfinite(tangent_score):
        raise ValueError(f"nonfinite facet: {split}/{unit_id}/{certificate}")
    if kind == "protected" and raw_score <= 0.0:
        raise ValueError(f"nonpositive protected raw score: {unit_id}/{certificate}")
    return {
        "split": split,
        "unit_id": unit_id,
        "certificate": certificate,
        "kind": kind,
        "raw_score": raw_score,
        "tangent_score": tangent_score,
    }


def _build_facets(
    fit_geometry: dict[str, Any], calibration_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    selected_theta = float(fit_geometry["selected_theta"])
    if selected_theta <= 0.0 or math.sin(selected_theta) <= 0.0:
        raise ValueError("selected theta must be positive")

    fit_facets: list[dict[str, Any]] = []
    panels = fit_geometry["fit_score_panels"]
    for unit_id, panel in panels.items():
        for certificate in PROTECTED_CERTIFICATES + TARGET_CERTIFICATES:
            kind = "protected" if certificate in PROTECTED_CERTIFICATES else "target"
            fit_facets.append(
                _facet(
                    "fit",
                    unit_id,
                    certificate,
                    kind,
                    float(panel["raw"][certificate]),
                    float(panel["tangent"][certificate]),
                )
            )

    calibration_facets: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    selected_cosine = math.cos(selected_theta)
    selected_sine = math.sin(selected_theta)
    for row in calibration_rows:
        unit_id = str(row["unit_id"])
        raw_scores = row["raw_scores"]["scores"]
        candidate_scores = row["candidate_scores"]["scores"]
        for certificate in PROTECTED_CERTIFICATES + TARGET_CERTIFICATES:
            raw_score = float(raw_scores[certificate])
            candidate_score = float(candidate_scores[certificate])
            tangent_score = (
                candidate_score - raw_score * selected_cosine
            ) / selected_sine
            reconstructed = score_at(raw_score, tangent_score, selected_theta)
            reconstruction_errors.append(abs(reconstructed - candidate_score))
            kind = "protected" if certificate in PROTECTED_CERTIFICATES else "target"
            calibration_facets.append(
                _facet(
                    "calibration",
                    unit_id,
                    certificate,
                    kind,
                    raw_score,
                    tangent_score,
                )
            )
    return fit_facets, calibration_facets, max(reconstruction_errors, default=0.0)


def _minimum_with_labels(
    rows: list[tuple[float, dict[str, Any]]], tolerance: float = 1e-9
) -> tuple[float, list[dict[str, str]]]:
    minimum = min(value for value, _ in rows)
    labels = [
        {"unit_id": facet["unit_id"], "certificate": facet["certificate"]}
        for value, facet in rows
        if abs(value - minimum) <= tolerance
    ]
    return minimum, labels


def _metrics(
    theta: float,
    fit_facets: list[dict[str, Any]],
    calibration_facets: list[dict[str, Any]],
    raw_target_floor: float,
) -> dict[str, Any]:
    fit_protected = [
        (
            score_at(facet["raw_score"], facet["tangent_score"], theta)
            / facet["raw_score"],
            facet,
        )
        for facet in fit_facets
        if facet["kind"] == "protected"
    ]
    fit_targets = [
        (score_at(facet["raw_score"], facet["tangent_score"], theta), facet)
        for facet in fit_facets
        if facet["kind"] == "target"
    ]
    calibration_protected = [
        (
            score_at(facet["raw_score"], facet["tangent_score"], theta)
            / facet["raw_score"],
            facet,
        )
        for facet in calibration_facets
        if facet["kind"] == "protected"
    ]
    fit_retention, fit_retention_labels = _minimum_with_labels(fit_protected)
    fit_target_floor, fit_target_labels = _minimum_with_labels(fit_targets)
    calibration_retention, calibration_labels = _minimum_with_labels(
        calibration_protected
    )
    fit_target_gain = fit_target_floor - raw_target_floor
    fit_pass = (
        fit_retention >= PROTECTED_RETENTION_FLOOR - NUMERIC_TOLERANCE
        and fit_target_floor >= TARGET_SCORE_FLOOR - NUMERIC_TOLERANCE
        and fit_target_gain >= TARGET_GAIN_FLOOR - NUMERIC_TOLERANCE
    )
    calibration_pass = (
        calibration_retention
        >= PROTECTED_RETENTION_FLOOR - NUMERIC_TOLERANCE
    )
    return {
        "theta": theta,
        "theta_degrees": math.degrees(theta),
        "fit": {
            "minimum_protected_retention": fit_retention,
            "protected_bottlenecks": fit_retention_labels,
            "target_floor": fit_target_floor,
            "target_bottlenecks": fit_target_labels,
            "target_gain_over_raw": fit_target_gain,
            "all_gates_pass": fit_pass,
        },
        "calibration": {
            "minimum_protected_retention": calibration_retention,
            "protected_bottlenecks": calibration_labels,
            "joint_nonconformity": PROTECTED_RETENTION_FLOOR
            - calibration_retention,
            "all_protected_pass": calibration_pass,
        },
        "joint_fit_and_calibration_pass": fit_pass and calibration_pass,
    }


def analyze(
    fit_geometry_path: Path,
    calibration_rows_path: Path,
    route_path: Path,
) -> dict[str, Any]:
    fit_geometry = _read_json(fit_geometry_path)
    route = _read_json(route_path)
    calibration_rows = _read_jsonl(calibration_rows_path)
    if len(calibration_rows) != 20:
        raise ValueError("expected the frozen 20 calibration rows")
    fit_facets, calibration_facets, reconstruction_error = _build_facets(
        fit_geometry, calibration_rows
    )
    if len(fit_facets) != 84 or len(calibration_facets) != 140:
        raise ValueError("unexpected fit or calibration facet count")

    selected_theta = float(fit_geometry["selected_theta"])
    theta_max = float(fit_geometry["theta_max"])
    raw_target_floor = min(
        facet["raw_score"]
        for facet in fit_facets
        if facet["kind"] == "target"
    )
    required_target_floor = max(
        TARGET_SCORE_FLOOR, raw_target_floor + TARGET_GAIN_FLOOR
    )

    boundaries = [0.0, selected_theta, theta_max]
    root_records: list[dict[str, Any]] = []
    for facet in fit_facets + calibration_facets:
        thresholds: list[tuple[str, float]] = []
        if facet["kind"] == "protected":
            thresholds.append(
                ("protected_retention", PROTECTED_RETENTION_FLOOR * facet["raw_score"])
            )
        elif facet["split"] == "fit":
            thresholds.extend(
                [
                    ("target_score_floor", TARGET_SCORE_FLOOR),
                    ("target_gain_floor", raw_target_floor + TARGET_GAIN_FLOOR),
                ]
            )
        for gate, threshold in thresholds:
            roots = trig_roots(
                facet["raw_score"],
                facet["tangent_score"],
                threshold,
                0.0,
                theta_max,
            )
            for root in roots:
                boundaries.append(root)
                root_records.append(
                    {
                        "theta": root,
                        "split": facet["split"],
                        "unit_id": facet["unit_id"],
                        "certificate": facet["certificate"],
                        "gate": gate,
                    }
                )
    boundaries = _unique_sorted(boundaries)

    def metrics(theta: float) -> dict[str, Any]:
        return _metrics(theta, fit_facets, calibration_facets, raw_target_floor)

    fit_intervals = _feasible_intervals(
        boundaries, lambda theta: bool(metrics(theta)["fit"]["all_gates_pass"])
    )
    calibration_intervals = _feasible_intervals(
        boundaries,
        lambda theta: bool(metrics(theta)["calibration"]["all_protected_pass"]),
    )
    joint_intervals = _feasible_intervals(
        boundaries,
        lambda theta: bool(metrics(theta)["joint_fit_and_calibration_pass"]),
    )
    calibration_safe_prefix_upper = 0.0
    if calibration_intervals and calibration_intervals[0]["lower"] <= NUMERIC_TOLERANCE:
        calibration_safe_prefix_upper = calibration_intervals[0]["upper"]
    first_fit_feasible_theta = fit_intervals[0]["lower"] if fit_intervals else None

    route_summary = route["summary"]
    result = {
        "name": "step3_rufmcd_p0_v2_posthoc_angle_sensitivity_v1",
        "analysis_status": "posthoc_mechanism_audit_not_authorized_for_selection",
        "source_files": {
            "fit_geometry": {
                "path": fit_geometry_path.as_posix(),
                "sha256": _sha256(fit_geometry_path),
            },
            "calibration_rows": {
                "path": calibration_rows_path.as_posix(),
                "sha256": _sha256(calibration_rows_path),
            },
            "route": {
                "path": route_path.as_posix(),
                "sha256": _sha256(route_path),
            },
        },
        "source_route": route["route"],
        "score_formula": "raw_score*cos(theta)+tangent_score*sin(theta)",
        "calibration_tangent_reconstruction": (
            "(candidate_score-raw_score*cos(fixed_theta))/sin(fixed_theta)"
        ),
        "maximum_calibration_candidate_reconstruction_error": reconstruction_error,
        "interval_radians": [0.0, theta_max],
        "fixed_theta": selected_theta,
        "thresholds": {
            "protected_retention_floor": PROTECTED_RETENTION_FLOOR,
            "target_score_floor": TARGET_SCORE_FLOOR,
            "target_gain_floor": TARGET_GAIN_FLOOR,
            "raw_fit_target_floor": raw_target_floor,
            "effective_fit_target_floor": required_target_floor,
        },
        "facet_counts": {
            "fit_protected": 48,
            "fit_target": 36,
            "calibration_protected": 80,
            "calibration_target_descriptive_only": 60,
        },
        "fixed_theta_metrics": metrics(selected_theta),
        "fit_feasible_intervals": fit_intervals,
        "calibration_protected_feasible_intervals": calibration_intervals,
        "joint_fit_and_calibration_feasible_intervals": joint_intervals,
        "first_fit_feasible_theta": first_fit_feasible_theta,
        "calibration_safe_prefix_upper_theta": calibration_safe_prefix_upper,
        "metrics_at_first_fit_feasible_theta": (
            metrics(first_fit_feasible_theta)
            if first_fit_feasible_theta is not None
            else None
        ),
        "metrics_at_calibration_safe_prefix_upper": metrics(
            calibration_safe_prefix_upper
        ),
        "angular_incompatibility_gap": (
            first_fit_feasible_theta - calibration_safe_prefix_upper
            if first_fit_feasible_theta is not None
            else None
        ),
        "joint_feasible_angle_exists": bool(joint_intervals),
        "analytic_boundary_count": len(boundaries),
        "root_record_count": len(root_records),
        "method": (
            "analytic roots of every trigonometric gate followed by exact "
            "midpoint classification of all induced intervals"
        ),
        "selection_boundary": {
            "calibration_scores_used_for_original_selection": False,
            "diagnostic_scores_used_for_original_selection": False,
            "posthoc_angle_selection_authorized": False,
            "threshold_relaxation_authorized": False,
            "fresh_validation_required_for_successor": True,
        },
        "registered_result_consistency": {
            "route_threshold": route_summary["calibration"]["conformal"][
                "threshold"
            ],
            "fixed_theta_minimum_calibration_retention": route_summary[
                "calibration"
            ]["minimum_joint_protected_retention"],
            "fixed_theta_fit_target_floor": route_summary["fit"][
                "selected_target_score_floor"
            ],
            "optimizer_steps": route["optimizer_steps"],
            "training_authorized": route["matched_training_protocol_authorized"],
            "core100_opened": route["core100_opened"],
            "heldout_opened": route["heldout_opened"],
        },
    }
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
    print(json.dumps({
        "output": args.output.as_posix(),
        "joint_feasible_angle_exists": result["joint_feasible_angle_exists"],
        "first_fit_feasible_theta": result["first_fit_feasible_theta"],
        "calibration_safe_prefix_upper_theta": result[
            "calibration_safe_prefix_upper_theta"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
