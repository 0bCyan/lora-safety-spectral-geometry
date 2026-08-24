"""Registered M7 mirror-curvature attribution for the stopped M6 calibration.

M7 is an explanatory, development-only diagnostic.  It reuses the frozen M6
negative-radius losses and measures only the opposite, positive-radius arm.
It never opens a sealed split and never executes an optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v4 as v4
import step3_cdro_opc_m5_confirmation as m5
import step3_cdro_opc_m6_virtual_displacement as m6


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m7/protocol/"
    "cdro_opc_m7_mirror_curvature_protocol.json"
)
OUTPUT_ROOT = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m7"
)
ARMS = ("fallback", "selected")
CERTIFICATES = tuple(core.PROTECTED)
FIXED_DOSE = m6.FIXED_DOSE
VIRTUAL_SCALE_ETA = m6.VIRTUAL_SCALE_ETA
SCORE_SCALE_FLOOR = m6.SCORE_SCALE_FLOOR
EXPECTED_UNITS = 20
LOSS_REPLAY_ABSOLUTE_TOLERANCE = 1e-6
LOSS_REPLAY_RELATIVE_TOLERANCE = 1e-6
NORM_REPLAY_ABSOLUTE_TOLERANCE = 1e-8
NORM_REPLAY_RELATIVE_TOLERANCE = 1e-6
SCORE_REPLAY_ABSOLUTE_TOLERANCE = 1e-8
SCORE_REPLAY_RELATIVE_TOLERANCE = 1e-6
SOURCE_ARITHMETIC_TOLERANCE = 1e-10
DECOMPOSITION_ARITHMETIC_TOLERANCE = 1e-10
MAXIMUM_MAP_RELATIVE_L2_ERROR = m6.MAXIMUM_MAP_RELATIVE_L2_ERROR
MAXIMUM_MAP_ABSOLUTE_ERROR = m6.MAXIMUM_MAP_ABSOLUTE_ERROR
EXPECTED_M6_ROUTE = "STOP_CDRO_OPC_M6_CALIBRATION"
EXPECTED_M6_FAILED_UNITS = ("m5_cal_14", "m5_cal_18", "m5_cal_24", "m5_cal_36")
GATE_KEYS = (
    "m6_source_boundary",
    "fixed_direction",
    "solver_kkt",
    "optimizer_state_empty",
    "replay_hash_restored",
    "eval_mode",
    "zero_step",
    "source_replay",
    "gradient_coverage",
    "source_post_integrity",
    "mirror_map_precheck",
    "mirror_map_fidelity",
    "exact_restoration",
    "algebra_identity",
    "finite_values",
    "all_units_attributed",
    "final_integrity",
)
INTEGRITY_KEYS = (
    "state_hash_restored",
    "state_tensor_equal",
    "optimizer_state_empty",
    "all_gradients_none",
    "eval_mode",
    "rng_restored",
    "buffers_restored",
)
REQUIRED_FROZEN_INPUTS = {
    "runner",
    "runner_tests",
    "protocol_note",
    "m6_runner",
    "m6_protocol",
    "m6_decision",
    "m6_diagnostics",
    "m6_reproduction",
}
PROTOCOL_TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "entry_commit",
    "objective",
    "claim_scope",
    "frozen_inputs",
    "fixed_direction",
    "m6_protocol_snapshot",
    "m6_result_snapshot",
    "calibration_allowlist",
    "sealed_unit_ids_sha256",
    "m1_gate_snapshot",
    "design",
    "validity_gates",
    "output_root",
    "optimizer_objects",
    "scheduler_objects",
    "optimizer_steps",
    "training_authorized",
    "mirror_parameter_evaluations_authorized",
    "negative_arm_recomputation_authorized",
    "sealed_audit_opened",
    "checkpoint_writes_authorized",
    "candidate_generation_authorized",
    "core100_opened",
    "heldout_opened",
    "result_dependent_radius_or_dose_change_authorized",
    "result_dependent_rerun_authorized",
    "radius_grid_authorized",
    "rescue_shrink_authorized",
    "endpoint_arm_authorized",
    "behavior_panel_authorized",
}
DESIGN_KEYS = {
    "fixed_dose",
    "virtual_scale_eta",
    "score_scale_floor",
    "protected_certificates",
    "arms",
    "calibration_units",
    "new_measurement_rule",
    "negative_arm_source",
    "sealed_policy",
    "outcome_thresholds_authorized",
    "mirror_write_order",
    "map_precheck_writes",
    "positive_loss_writes",
    "negative_loss_writes",
    "sealed_loss_writes",
    "endpoint_loss_writes",
    "primary_decomposition_centering",
    "fresh_baseline_usage",
    "known_M6_failed_units_posthoc",
}
TABLE_FIELDS = (
    "unit_id",
    "sample_id",
    "certificate",
    "m6_unit_covered",
    "m6_failed_unit_posthoc",
    "m6_failed_cell_posthoc",
    "score_denominator",
    "first_order_deterioration",
    "odd_finite_radius_remainder",
    "even_deterioration",
    "central_deterioration",
    "finite_deterioration",
    "finite_minus_first_order",
    "finite_identity_residual",
    "gap_identity_residual",
    "fallback_centered_even_score",
    "selected_centered_even_score",
)


class M7MirrorCurvatureError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M7MirrorCurvatureError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def within_tolerance(
    observed: float,
    expected: float,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> bool:
    if not math.isfinite(float(observed)) or not math.isfinite(float(expected)):
        return False
    return abs(float(observed) - float(expected)) <= max(
        float(absolute_tolerance),
        float(relative_tolerance) * abs(float(expected)),
    )


def arm_decomposition(
    forward_score: float,
    mirror_score: float,
    first_order_score: float,
) -> dict[str, float]:
    """Decompose the frozen forward score into central and even contributions."""

    forward = float(forward_score)
    mirror = float(mirror_score)
    first = float(first_order_score)
    central = 0.5 * (forward + mirror)
    centered_even = 0.5 * (mirror - forward)
    even_forward = -centered_even
    central_shift = central - first
    taylor_error = forward - first
    return {
        "forward_score": forward,
        "mirror_score": mirror,
        "first_order_score": first,
        "central_score": central,
        "centered_even_score": centered_even,
        "even_forward_contribution": even_forward,
        "central_shift_from_first_order": central_shift,
        "forward_taylor_error": taylor_error,
        "forward_identity_residual": forward - (central - centered_even),
        "taylor_identity_residual": taylor_error
        - (central_shift - centered_even),
    }


def comparison_decomposition(
    fallback: Mapping[str, float],
    selected: Mapping[str, float],
    score_floor: float,
    frozen_finite_deterioration: float,
) -> dict[str, float]:
    denominator = max(abs(float(fallback["first_order_score"])), float(score_floor))
    first = (
        float(fallback["first_order_score"])
        - float(selected["first_order_score"])
    ) / denominator
    central = (
        float(fallback["central_score"]) - float(selected["central_score"])
    ) / denominator
    even = (
        float(selected["centered_even_score"])
        - float(fallback["centered_even_score"])
    ) / denominator
    finite = float(frozen_finite_deterioration)
    central_shift = central - first
    finite_minus_first = finite - first
    return {
        "score_denominator": denominator,
        "first_order_deterioration": first,
        "central_deterioration": central,
        "even_deterioration": even,
        "odd_finite_radius_remainder": central_shift,
        "finite_deterioration": finite,
        "finite_minus_first_order": finite_minus_first,
        "finite_identity_residual": finite - (central + even),
        "gap_identity_residual": finite_minus_first - (central_shift + even),
    }


def strict_gate_map(value: Mapping[str, Any]) -> bool:
    return set(value) == set(GATE_KEYS) and all(type(value[name]) is bool for name in GATE_KEYS)


def expected_route(gates: Mapping[str, Any]) -> str:
    if not strict_gate_map(gates):
        return "STOP_CDRO_OPC_M7_INVALID"
    if not gates["final_integrity"]:
        return "STOP_CDRO_OPC_M7_FINAL_INTEGRITY"
    if not all(
        gates[name]
        for name in (
            "m6_source_boundary",
            "fixed_direction",
            "solver_kkt",
            "optimizer_state_empty",
            "replay_hash_restored",
            "eval_mode",
            "zero_step",
        )
    ):
        return "STOP_CDRO_OPC_M7_INVALID_INPUT"
    if not all(
        gates[name]
        for name in ("source_replay", "gradient_coverage", "source_post_integrity")
    ):
        return "STOP_CDRO_OPC_M7_SOURCE_REPLAY"
    if not all(
        gates[name]
        for name in (
            "mirror_map_precheck",
            "mirror_map_fidelity",
            "exact_restoration",
        )
    ):
        return "STOP_CDRO_OPC_M7_MIRROR_INTEGRITY"
    if not all(
        gates[name]
        for name in ("algebra_identity", "finite_values", "all_units_attributed")
    ):
        return "STOP_CDRO_OPC_M7_ALGEBRA"
    return "PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC"


def expected_mutation_events(
    calibration_allowlist: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for arm in ARMS:
        events.append(
            {
                "sequence": len(events) + 1,
                "purpose": "map_precheck",
                "unit_id": None,
                "sample_id": None,
                "arm": arm,
                "sign": 1,
                "eta": VIRTUAL_SCALE_ETA,
                "parameter_rule": "theta0_plus_eta_times_frozen_displacement",
            }
        )
    for item in calibration_allowlist:
        for arm in ARMS:
            events.append(
                {
                    "sequence": len(events) + 1,
                    "purpose": "positive_loss",
                    "unit_id": str(item["unit_id"]),
                    "sample_id": str(item["sample_id"]),
                    "arm": arm,
                    "sign": 1,
                    "eta": VIRTUAL_SCALE_ETA,
                    "parameter_rule": "theta0_plus_eta_times_frozen_displacement",
                }
            )
    return events


def frozen_failure_snapshot(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = diagnostics["calibration"]["finite_protection"]["rows"]
    return [
        {
            "unit_id": str(row["unit_id"]),
            "failed_certificates": [
                name
                for name in CERTIFICATES
                if float(row["finite_deteriorations"][name])
                > m6.ALLOWED_RELATIVE_DROP + m6.ARITHMETIC_TOLERANCE
            ],
            "finite_deteriorations": {
                name: float(row["finite_deteriorations"][name])
                for name in CERTIFICATES
            },
        }
        for row in rows
        if not bool(row["covered"])
    ]


def context(path: Path) -> tuple[Any, ...]:
    protocol = read_json(path)
    if (
        set(protocol) != PROTOCOL_TOP_LEVEL_KEYS
        or protocol.get("objective")
        != "Decompose the frozen M6 calibration finite-radius deterioration with one registered positive mirror measurement per frozen arm and no negative-arm recomputation."
        or protocol.get("claim_scope")
        != "post_M6_fixed_radius_validity_only_nominal_mirror_decomposition_not_fresh_confirmation"
        or
        protocol.get("schema")
        != "step3_cdro_opc_m7_mirror_curvature_attribution_protocol_v1"
        or protocol.get("status")
        != "registered_after_M6_result_before_any_M7_mirror_write"
        or protocol.get("output_root") != OUTPUT_ROOT
    ):
        raise M7MirrorCurvatureError("M7 protocol identity differs")
    if any(
        (
            protocol.get("optimizer_objects") != 1,
            protocol.get("scheduler_objects") != 0,
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("mirror_parameter_evaluations_authorized") is not True,
            protocol.get("negative_arm_recomputation_authorized") is not False,
            protocol.get("sealed_audit_opened") is not False,
            protocol.get("checkpoint_writes_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
            protocol.get("result_dependent_radius_or_dose_change_authorized")
            is not False,
            protocol.get("result_dependent_rerun_authorized") is not False,
            protocol.get("radius_grid_authorized") is not False,
            protocol.get("rescue_shrink_authorized") is not False,
            protocol.get("endpoint_arm_authorized") is not False,
            protocol.get("behavior_panel_authorized") is not False,
        )
    ):
        raise M7MirrorCurvatureError("M7 authorization boundary differs")
    frozen = protocol.get("frozen_inputs", {})
    if set(frozen) != REQUIRED_FROZEN_INPUTS:
        raise M7MirrorCurvatureError("M7 frozen input registry differs")
    for name, spec in frozen.items():
        source = repo_path(str(spec["path"]))
        if not source.is_file() or file_sha256(source) != spec["sha256"]:
            raise M7MirrorCurvatureError(f"frozen input mismatch: {name}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    ).returncode:
        raise M7MirrorCurvatureError("entry commit is not an ancestor")

    design = protocol.get("design", {})
    gates = protocol.get("validity_gates", {})
    if (
        set(design) != DESIGN_KEYS
        or float(design.get("fixed_dose", math.nan)) != FIXED_DOSE
        or float(design.get("virtual_scale_eta", math.nan)) != VIRTUAL_SCALE_ETA
        or float(design.get("score_scale_floor", math.nan)) != SCORE_SCALE_FLOOR
        or tuple(design.get("protected_certificates", ())) != CERTIFICATES
        or tuple(design.get("arms", ())) != ARMS
        or int(design.get("calibration_units", -1)) != EXPECTED_UNITS
        or design.get("new_measurement_rule")
        != "measure_only_theta0_plus_eta_d_for_each_frozen_M6_arm"
        or design.get("negative_arm_source") != "frozen_M6_diagnostics_only"
        or design.get("sealed_policy") != "never_open"
        or design.get("outcome_thresholds_authorized") is not False
        or design.get("mirror_write_order") != ["fallback", "selected"]
        or int(design.get("map_precheck_writes", -1)) != len(ARMS)
        or int(design.get("positive_loss_writes", -1))
        != EXPECTED_UNITS * len(ARMS)
        or int(design.get("negative_loss_writes", -1)) != 0
        or int(design.get("sealed_loss_writes", -1)) != 0
        or int(design.get("endpoint_loss_writes", -1)) != 0
        or design.get("primary_decomposition_centering")
        != "frozen_M6_L0_Lminus_norm_and_reference_norm"
        or design.get("fresh_baseline_usage") != "source_replay_audit_only"
        or tuple(design.get("known_M6_failed_units_posthoc", ()))
        != EXPECTED_M6_FAILED_UNITS
        or set(gates)
        != {
            "loss_replay_absolute_tolerance",
            "loss_replay_relative_tolerance",
            "norm_replay_absolute_tolerance",
            "norm_replay_relative_tolerance",
            "score_replay_absolute_tolerance",
            "score_replay_relative_tolerance",
            "source_arithmetic_tolerance",
            "decomposition_arithmetic_tolerance",
            "maximum_map_relative_l2_error",
            "maximum_map_absolute_error",
        }
        or float(gates.get("loss_replay_absolute_tolerance", math.nan))
        != LOSS_REPLAY_ABSOLUTE_TOLERANCE
        or float(gates.get("loss_replay_relative_tolerance", math.nan))
        != LOSS_REPLAY_RELATIVE_TOLERANCE
        or float(gates.get("norm_replay_absolute_tolerance", math.nan))
        != NORM_REPLAY_ABSOLUTE_TOLERANCE
        or float(gates.get("norm_replay_relative_tolerance", math.nan))
        != NORM_REPLAY_RELATIVE_TOLERANCE
        or float(gates.get("score_replay_absolute_tolerance", math.nan))
        != SCORE_REPLAY_ABSOLUTE_TOLERANCE
        or float(gates.get("score_replay_relative_tolerance", math.nan))
        != SCORE_REPLAY_RELATIVE_TOLERANCE
        or float(gates.get("source_arithmetic_tolerance", math.nan))
        != SOURCE_ARITHMETIC_TOLERANCE
        or float(gates.get("decomposition_arithmetic_tolerance", math.nan))
        != DECOMPOSITION_ARITHMETIC_TOLERANCE
        or float(gates.get("maximum_map_relative_l2_error", math.nan))
        != MAXIMUM_MAP_RELATIVE_L2_ERROR
        or float(gates.get("maximum_map_absolute_error", math.nan))
        != MAXIMUM_MAP_ABSOLUTE_ERROR
    ):
        raise M7MirrorCurvatureError("M7 registered design or validity gates differ")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in gates.values()
    ):
        raise M7MirrorCurvatureError("M7 validity gate domain differs")

    m6_protocol_path = repo_path(frozen["m6_protocol"]["path"])
    m6_context = m6.context(m6_protocol_path)
    m6_protocol = m6_context[0]
    m1_protocol = m6_context[1]
    if frozen["m6_runner"] != m6_protocol["frozen_inputs"]["runner"]:
        raise M7MirrorCurvatureError("M7 inherited M6 runner differs")
    m6_decision = read_json(repo_path(frozen["m6_decision"]["path"]))
    m6_diagnostics = read_json(repo_path(frozen["m6_diagnostics"]["path"]))
    m6_reproduction = read_json(repo_path(frozen["m6_reproduction"]["path"]))
    source_route = m6.verify_artifacts(m6_protocol, m6_decision, m6_diagnostics)
    failures = frozen_failure_snapshot(m6_diagnostics)
    source_rows = m6_diagnostics["calibration"]["finite_protection"]["rows"]
    units = m6_context[-1]
    calibration_allowlist = [
        {"unit_id": str(unit["unit_id"]), "sample_id": str(unit["sample_id"])}
        for unit in units
        if unit["split"] == "calibration"
    ]
    sealed_ids = sorted(
        str(unit["unit_id"]) for unit in units if unit["split"] == "sealed_audit"
    )
    calibration_gates = m6_diagnostics["calibration"]["gates"]
    if (
        source_route != EXPECTED_M6_ROUTE
        or m6_decision.get("route") != EXPECTED_M6_ROUTE
        or m6_diagnostics.get("route") != EXPECTED_M6_ROUTE
        or m6_decision.get("calibration_covered_units") != 16
        or len(source_rows) != EXPECTED_UNITS
        or m6_diagnostics.get("sealed_audit_opened") is not False
        or m6_diagnostics.get("sealed_audit") is not None
        or m6_diagnostics.get("optimizer_steps") != 0
        or m6_diagnostics.get("training_authorized") is not False
        or m6_reproduction.get("schema")
        != "step3_cdro_opc_m6_reproduction_manifest_v1"
        or m6_reproduction.get("runtime_commit")
        != m6_diagnostics.get("runtime_head")
        or m6_reproduction.get("optimizer_steps") != 0
        or m6_reproduction.get("training_authorized") is not False
        or m6_reproduction.get("outputs_sha256", {}).get("decision")
        != file_sha256(repo_path(frozen["m6_decision"]["path"]))
        or m6_reproduction.get("outputs_sha256", {}).get("diagnostics")
        != file_sha256(repo_path(frozen["m6_diagnostics"]["path"]))
        or not all(m6_diagnostics.get("common_gates", {}).values())
        or not all(m6_diagnostics.get("final_integrity", {}).values())
        or set(calibration_gates) != set(m6.SPLIT_GATE_KEYS)
        or calibration_gates.get("protected_unit_coverage") is not False
        or not all(
            calibration_gates[name]
            for name in m6.SPLIT_GATE_KEYS
            if name != "protected_unit_coverage"
        )
        or len({str(row["unit_id"]) for row in source_rows}) != EXPECTED_UNITS
        or any(
            set(row["initial_losses"]) != set(CERTIFICATES)
            or set(row["gradient_norms"]) != set(CERTIFICATES)
            or set(row["arms"]) != set(ARMS)
            or set(row["finite_deteriorations"]) != set(CERTIFICATES)
            for row in source_rows
        )
        or tuple(item["unit_id"] for item in failures) != EXPECTED_M6_FAILED_UNITS
        or protocol.get("m6_result_snapshot")
        != {
            "route": EXPECTED_M6_ROUTE,
            "calibration_covered_units": 16,
            "calibration_total_units": EXPECTED_UNITS,
            "sealed_audit_opened": False,
            "failed_units": failures,
        }
        or protocol.get("fixed_direction") != m6_protocol.get("fixed_direction")
        or protocol.get("m6_protocol_snapshot")
        != m6_protocol.get("m5_protocol_snapshot")
        or protocol.get("calibration_allowlist") != calibration_allowlist
        or protocol.get("sealed_unit_ids_sha256") != canonical_sha256(sealed_ids)
        or protocol.get("m1_gate_snapshot")
        != {
            "feasibility_tolerance": m1_protocol["gates"][
                "feasibility_tolerance"
            ],
            "kkt_tolerance": m1_protocol["gates"]["kkt_tolerance"],
        }
    ):
        raise M7MirrorCurvatureError("frozen M6 stop boundary differs")
    return (
        protocol,
        m6_protocol,
        m6_decision,
        m6_diagnostics,
        *m6_context[1:],
    )


def require_clean_start(protocol: Mapping[str, Any]) -> str:
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise M7MirrorCurvatureError("M7 run requires a clean worktree")
    output = repo_path(str(protocol["output_root"]))
    if output.exists() and any(item.name != "protocol" for item in output.iterdir()):
        raise M7MirrorCurvatureError("M7 output already exists")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def publish_staged_directories(stage_root: Path, final_root: Path) -> None:
    names = ("run", "analysis", "figures")
    if any((final_root / name).exists() for name in names):
        raise M7MirrorCurvatureError("M7 final artifact directory already exists")
    if not all((stage_root / name).is_dir() for name in names):
        raise M7MirrorCurvatureError("M7 staged artifact directory is incomplete")
    published: list[str] = []
    try:
        for name in names:
            (stage_root / name).replace(final_root / name)
            published.append(name)
    except BaseException:
        for name in reversed(published):
            final = final_root / name
            if final.exists():
                final.replace(stage_root / name)
        raise


def source_internal_audit(
    source_row: Mapping[str, Any],
    eta: float,
    reference_norm: float,
    score_floor: float,
    tolerance: float,
) -> dict[str, Any]:
    score_residuals: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        score_residuals[arm] = {}
        for name in CERTIFICATES:
            initial = float(source_row["initial_losses"][name])
            final = float(source_row["arms"][arm]["final_losses"][name])
            norm = float(source_row["gradient_norms"][name])
            reproduced = (initial - final) / (eta * norm * reference_norm)
            frozen = float(
                source_row["arms"][arm]["normalized_actual_scores"][name]
            )
            score_residuals[arm][name] = reproduced - frozen
    deterioration_residuals = {}
    for name in CERTIFICATES:
        fallback = float(
            source_row["arms"]["fallback"]["normalized_actual_scores"][name]
        )
        selected = float(
            source_row["arms"]["selected"]["normalized_actual_scores"][name]
        )
        first_fallback = float(
            source_row["arms"]["fallback"]["first_order_scores"][name]
        )
        reproduced = (fallback - selected) / max(abs(first_fallback), score_floor)
        deterioration_residuals[name] = reproduced - float(
            source_row["finite_deteriorations"][name]
        )
    residuals = [
        abs(value)
        for arm in ARMS
        for value in score_residuals[arm].values()
    ] + [abs(value) for value in deterioration_residuals.values()]
    return {
        "normalized_score_residuals": score_residuals,
        "finite_deterioration_residuals": deterioration_residuals,
        "maximum_absolute_residual": max(residuals),
        "gate": max(residuals) <= tolerance,
    }


def coverage_structure_passes(
    fresh: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    exact_fields = (
        "all_gradients_finite_nonzero",
        "continuation_tokens",
        "etarc_margin_terms",
        "mode",
        "response_tokens",
        "risk_microbatches",
        "safety_microbatches",
        "tarc_margin_terms",
    )
    return all(fresh.get(name) == source.get(name) for name in exact_fields)


def gradient_coverage_passes(coverage: Mapping[str, Any]) -> bool:
    return bool(
        coverage.get("all_gradients_finite_nonzero")
        and int(coverage.get("risk_microbatches", 0)) == 1
        and int(coverage.get("safety_microbatches", 0)) == 1
        and int(coverage.get("response_tokens", 0)) > 0
        and int(coverage.get("continuation_tokens", 0)) > 0
        and int(coverage.get("tarc_margin_terms", 0)) > 0
        and int(coverage.get("etarc_margin_terms", 0)) > 0
        and all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in coverage.get("gradient_norms", {}).values()
        )
    )


def source_replay_row(
    unit: Mapping[str, Any],
    source_row: Mapping[str, Any],
    coverage: Mapping[str, Any],
    norms: Mapping[str, float],
    scores: Mapping[str, Mapping[str, float]],
    protocol: Mapping[str, Any],
    reference_norm: float,
) -> dict[str, Any]:
    gates = protocol["validity_gates"]
    loss_tolerance = float(gates["loss_replay_absolute_tolerance"])
    loss_relative = float(gates["loss_replay_relative_tolerance"])
    norm_absolute = float(gates["norm_replay_absolute_tolerance"])
    norm_relative = float(gates["norm_replay_relative_tolerance"])
    score_tolerance = float(gates["score_replay_absolute_tolerance"])
    score_relative = float(gates["score_replay_relative_tolerance"])
    source_coverage = source_row["gradient_coverage"]
    loss_errors = {
        name: float(coverage["loss_means"][name])
        - float(source_coverage["loss_means"][name])
        for name in source_coverage["loss_means"]
    }
    raw_norm_errors = {
        name: float(coverage["gradient_norms"][name])
        - float(source_coverage["gradient_norms"][name])
        for name in source_coverage["gradient_norms"]
    }
    certificate_norm_errors = {
        name: float(norms[name]) - float(source_row["gradient_norms"][name])
        for name in CERTIFICATES
    }
    score_errors = {
        arm: {
            name: float(scores[arm][name])
            - float(source_row["arms"][arm]["first_order_scores"][name])
            for name in CERTIFICATES
        }
        for arm in ARMS
    }
    loss_gate = all(
        within_tolerance(
            coverage["loss_means"][name],
            source_coverage["loss_means"][name],
            loss_tolerance,
            loss_relative,
        )
        for name in loss_errors
    )
    raw_norm_gate = all(
        within_tolerance(
            coverage["gradient_norms"][name],
            source_coverage["gradient_norms"][name],
            norm_absolute,
            norm_relative,
        )
        for name in raw_norm_errors
    )
    certificate_norm_gate = all(
        within_tolerance(
            norms[name],
            source_row["gradient_norms"][name],
            norm_absolute,
            norm_relative,
        )
        for name in CERTIFICATES
    )
    score_gate = all(
        within_tolerance(
            scores[arm][name],
            source_row["arms"][arm]["first_order_scores"][name],
            score_tolerance,
            score_relative,
        )
        for arm in ARMS
        for name in CERTIFICATES
    )
    structure_gate = bool(
        str(unit["unit_id"]) == str(source_row["unit_id"])
        and str(unit["sample_id"]) == str(source_row["sample_id"])
        and coverage_structure_passes(coverage, source_coverage)
    )
    internal = source_internal_audit(
        source_row,
        float(protocol["design"]["virtual_scale_eta"]),
        reference_norm,
        float(protocol["design"]["score_scale_floor"]),
        float(gates["source_arithmetic_tolerance"]),
    )
    coverage_structure_fields = (
        "all_gradients_finite_nonzero",
        "continuation_tokens",
        "etarc_margin_terms",
        "mode",
        "response_tokens",
        "risk_microbatches",
        "safety_microbatches",
        "tarc_margin_terms",
    )
    row_gate = bool(
        structure_gate
        and loss_gate
        and raw_norm_gate
        and certificate_norm_gate
        and score_gate
        and internal["gate"]
    )
    return {
        "unit_id": str(unit["unit_id"]),
        "sample_id": str(unit["sample_id"]),
        "fresh_coverage_structure": {
            name: coverage.get(name) for name in coverage_structure_fields
        },
        "fresh_loss_means": {
            name: float(value) for name, value in coverage["loss_means"].items()
        },
        "fresh_raw_gradient_norms": {
            name: float(value)
            for name, value in coverage["gradient_norms"].items()
        },
        "fresh_certificate_norms": {
            name: float(norms[name]) for name in CERTIFICATES
        },
        "fresh_first_order_scores": {
            arm: {name: float(scores[arm][name]) for name in CERTIFICATES}
            for arm in ARMS
        },
        "errors": {
            "loss_means": loss_errors,
            "raw_gradient_norms": raw_norm_errors,
            "certificate_norms": certificate_norm_errors,
            "first_order_scores": score_errors,
        },
        "checks": {
            "structure": structure_gate,
            "losses": loss_gate,
            "raw_gradient_norms": raw_norm_gate,
            "certificate_norms": certificate_norm_gate,
            "first_order_scores": score_gate,
            "source_internal_arithmetic": bool(internal["gate"]),
        },
        "source_internal_audit": internal,
        "gradient_coverage_gate": gradient_coverage_passes(coverage),
        "gate": row_gate,
    }


def prepare_calibration(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_targets: Sequence[Mapping[str, Any]],
    risk: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[Mapping[str, Any]], dict[str, Any], dict[int, Any], dict[str, Any]]:
    audit = m5.structural_audit(
        {"selection": protocol["m6_protocol_snapshot"]["selection"]},
        selected_targets,
        units,
        tokenizer,
        config,
        "calibration",
    )
    pairs, _ = core.tarc.build_pair_features(
        tokenizer,
        core.subset_targets(selected_targets, units, "calibration"),
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    pair_map = {str(pair["sample_id"]): pair for pair in pairs}
    split_units = [unit for unit in units if unit["split"] == "calibration"]
    risk_features = {
        int(unit["risk_row_index"]): core.training.response_feature(
            tokenizer,
            str(risk[int(unit["risk_row_index"])]["prompt"]),
            str(risk[int(unit["risk_row_index"])]["response"]),
            int(config["max_length"]),
        )
        for unit in split_units
    }
    if len(split_units) != EXPECTED_UNITS:
        raise M7MirrorCurvatureError("M7 calibration denominator differs")
    return split_units, pair_map, risk_features, audit


def replay_source_panel(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    split_units: Sequence[Mapping[str, Any]],
    pair_map: Mapping[str, Any],
    risk_features: Mapping[int, Any],
    fallback: Mapping[str, Any],
    selected: Mapping[str, Any],
    fresh_reference_norm: float,
    frozen_reference_norm: float,
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [str(unit["unit_id"]) for unit in split_units] != [
        str(row["unit_id"]) for row in source_rows
    ]:
        raise M7MirrorCurvatureError("M7 unit order differs from frozen M6")
    rows = []
    for unit, source_row in zip(split_units, source_rows):
        entry_gradients = m6.gradient_snapshot(parameters)
        entry_modes = m6.mode_snapshot(model)
        entry_rng = m6.rng_snapshot()
        entry_buffers = buffer_snapshot(model)
        row: dict[str, Any] | None = None
        unchanged: dict[str, bool] | None = None
        try:
            gradients, coverage = core.rp.aggregate_block_gradients(
                model,
                list(parameters),
                dict(risk_features),
                dict(pair_map),
                [int(unit["risk_row_index"])],
                [str(unit["sample_id"])],
                dict(config),
                train_mode=False,
            )
            certificates, norms = v4.normalize_registered_certificates(gradients)
            scores = {
                "fallback": core.m1.map_scores(
                    certificates, dict(fallback), fresh_reference_norm
                ),
                "selected": core.m1.map_scores(
                    certificates, dict(selected), fresh_reference_norm
                ),
            }
            row = source_replay_row(
                unit,
                source_row,
                coverage,
                norms,
                scores,
                protocol,
                frozen_reference_norm,
            )
            unchanged = {
                "state_tensor_equal": m6.state_equal(parameters, base_state),
                "gradients_unchanged": m6.gradients_equal(
                    parameters, entry_gradients
                ),
                "mode_unchanged": m6.mode_equal(model, entry_modes),
                "rng_unchanged": m6.rng_equal(entry_rng),
                "buffers_unchanged": buffers_equal(model, entry_buffers),
            }
            del gradients, certificates
        finally:
            m6.restore_parameters(parameters, base_state)
            m6.restore_gradients(parameters, entry_gradients)
            m6.restore_modes(model, entry_modes)
            m6.restore_rng(entry_rng)
            restore_buffers(model, entry_buffers)
        if row is None or unchanged is None:
            raise M7MirrorCurvatureError("source replay row did not complete")
        restored = {
            "state_tensor_equal": m6.state_equal(parameters, base_state),
            "gradients_restored": m6.gradients_equal(parameters, entry_gradients),
            "mode_restored": m6.mode_equal(model, entry_modes),
            "rng_restored": m6.rng_equal(entry_rng),
            "buffers_restored": buffers_equal(model, entry_buffers),
        }
        row["source_unit_integrity"] = {
            "unchanged_during_replay": unchanged,
            "restored_after_replay": restored,
            "gate": all(unchanged.values()) and all(restored.values()),
        }
        row["gate"] = bool(row["gate"] and row["source_unit_integrity"]["gate"])
        rows.append(row)
    error_groups = {
        "loss_means": [
            abs(value)
            for row in rows
            for value in row["errors"]["loss_means"].values()
        ],
        "raw_gradient_norms": [
            abs(value)
            for row in rows
            for value in row["errors"]["raw_gradient_norms"].values()
        ],
        "certificate_norms": [
            abs(value)
            for row in rows
            for value in row["errors"]["certificate_norms"].values()
        ],
        "first_order_scores": [
            abs(value)
            for row in rows
            for arm in ARMS
            for value in row["errors"]["first_order_scores"][arm].values()
        ],
    }
    return {
        "status": "evaluated",
        "rows": rows,
        "total_units": len(rows),
        "reproduced_units": sum(bool(row["gate"]) for row in rows),
        "maximum_absolute_errors": {
            name: max(values) for name, values in error_groups.items()
        },
        "gate": len(rows) == EXPECTED_UNITS and all(bool(row["gate"]) for row in rows),
        "gradient_coverage_gate": all(
            bool(row["gradient_coverage_gate"]) for row in rows
        ),
    }


def buffer_snapshot(model: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def buffers_equal(model: Any, state: Mapping[str, Any]) -> bool:
    import torch

    current = dict(model.named_buffers())
    return set(current) == set(state) and all(
        torch.equal(current[name].detach(), state[name]) for name in state
    )


def restore_buffers(model: Any, state: Mapping[str, Any]) -> None:
    current = dict(model.named_buffers())
    if set(current) != set(state):
        raise M7MirrorCurvatureError("model buffer registry changed")
    for name, value in state.items():
        current[name].data.copy_(value)


def integrity_audit(
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    initial_hash: str,
    optimizer: Any,
    mode_state: Mapping[str, bool],
    rng_state: Mapping[str, Any],
    buffer_state: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "state_hash_restored": core.cspr.trainable_state_hash(parameters)
        == initial_hash,
        "state_tensor_equal": m6.state_equal(parameters, base_state),
        "optimizer_state_empty": not bool(optimizer.state),
        "all_gradients_none": all(
            parameter.grad is None for _, parameter in parameters
        ),
        "eval_mode": m6.mode_equal(model, mode_state),
        "rng_restored": m6.rng_equal(rng_state),
        "buffers_restored": buffers_equal(model, buffer_state),
    }


def realized_positive_map_audit(
    parameters: Sequence[tuple[str, Any]],
    state: Mapping[str, Any],
    displacement: Mapping[str, Any],
    eta: float,
) -> dict[str, float]:
    squared_error = 0.0
    squared_reference = 0.0
    squared_delta = 0.0
    max_absolute_error = 0.0
    max_absolute_delta = 0.0
    for name, parameter in parameters:
        observed_map = (parameter.detach() - state[name]) / eta
        difference = observed_map - displacement[name]
        delta = parameter.detach() - state[name]
        squared_error += float(difference.double().square().sum().cpu())
        squared_reference += float(displacement[name].double().square().sum().cpu())
        squared_delta += float(delta.double().square().sum().cpu())
        max_absolute_error = max(
            max_absolute_error, float(difference.abs().max().cpu())
        )
        max_absolute_delta = max(max_absolute_delta, float(delta.abs().max().cpu()))
    return {
        "relative_l2_error": math.sqrt(squared_error)
        / max(math.sqrt(squared_reference), 1e-30),
        "max_absolute_error": max_absolute_error,
        "realized_parameter_l2_radius": math.sqrt(squared_delta),
        "realized_parameter_linf_radius": max_absolute_delta,
    }


def mirror_arm(
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    displacement: Mapping[str, Any],
    eta: float,
    evaluate: Any,
    write_counter: dict[str, Any],
    counter_key: str,
    event: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate theta0 + eta*d with full restoration and positive-map auditing."""

    import torch

    gradients = m6.gradient_snapshot(parameters)
    modes = m6.mode_snapshot(model)
    rng = m6.rng_snapshot()
    buffers = buffer_snapshot(model)
    value: Any = None
    applied_audit: dict[str, float] | None = None
    post_audit: dict[str, float] | None = None
    rng_unchanged = False
    mode_unchanged = False
    gradients_unchanged = False
    buffers_unchanged = False
    try:
        m6.restore_parameters(parameters, base_state)
        with torch.no_grad():
            for name, parameter in parameters:
                parameter.add_(displacement[name], alpha=eta)
        write_counter[counter_key] = int(write_counter.get(counter_key, 0)) + 1
        if event is not None:
            events = write_counter.setdefault("events", [])
            events.append(
                {
                    "sequence": len(events) + 1,
                    "purpose": str(event["purpose"]),
                    "unit_id": event.get("unit_id"),
                    "sample_id": event.get("sample_id"),
                    "arm": str(event["arm"]),
                    "sign": 1,
                    "eta": float(eta),
                    "parameter_rule": "theta0_plus_eta_times_frozen_displacement",
                }
            )
        applied_audit = realized_positive_map_audit(
            parameters, base_state, displacement, eta
        )
        value = evaluate()
        post_audit = realized_positive_map_audit(
            parameters, base_state, displacement, eta
        )
        rng_unchanged = m6.rng_equal(rng)
        mode_unchanged = m6.mode_equal(model, modes)
        gradients_unchanged = m6.gradients_equal(parameters, gradients)
        buffers_unchanged = buffers_equal(model, buffers)
    finally:
        m6.restore_parameters(parameters, base_state)
        m6.restore_gradients(parameters, gradients)
        m6.restore_modes(model, modes)
        m6.restore_rng(rng)
        restore_buffers(model, buffers)
    restored = m6.state_equal(parameters, base_state)
    gradients_restored = m6.gradients_equal(parameters, gradients)
    mode_restored = m6.mode_equal(model, modes)
    rng_restored = m6.rng_equal(rng)
    buffers_restored = buffers_equal(model, buffers)
    if applied_audit is None or post_audit is None:
        raise M7MirrorCurvatureError("positive mirror displacement was not evaluated")
    return value, {
        **applied_audit,
        "post_evaluation_relative_l2_error": post_audit["relative_l2_error"],
        "post_evaluation_max_absolute_error": post_audit["max_absolute_error"],
        "parameter_delta_sign": "positive_eta_times_frozen_displacement",
        "state_restored": restored,
        "rng_unchanged": rng_unchanged,
        "mode_unchanged": mode_unchanged,
        "gradients_unchanged": gradients_unchanged,
        "buffers_unchanged": buffers_unchanged,
        "rng_restored": rng_restored,
        "mode_restored": mode_restored,
        "gradients_restored": gradients_restored,
        "buffers_restored": buffers_restored,
    }


def mirror_audit_passes(
    audit: Mapping[str, Any],
    maximum_relative_error: float,
    maximum_absolute_error: float,
) -> bool:
    return bool(
        float(audit["relative_l2_error"]) <= maximum_relative_error
        and float(audit["max_absolute_error"]) <= maximum_absolute_error
        and float(audit["post_evaluation_relative_l2_error"])
        <= maximum_relative_error
        and float(audit["post_evaluation_max_absolute_error"])
        <= maximum_absolute_error
        and audit["parameter_delta_sign"]
        == "positive_eta_times_frozen_displacement"
        and audit["state_restored"]
        and audit["rng_unchanged"]
        and audit["mode_unchanged"]
        and audit["gradients_unchanged"]
        and audit["buffers_unchanged"]
        and audit["rng_restored"]
        and audit["mode_restored"]
        and audit["gradients_restored"]
        and audit["buffers_restored"]
    )


def mirror_map_precheck(
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    displacements: Mapping[str, Mapping[str, Any]],
    eta: float,
    maximum_relative_error: float,
    maximum_absolute_error: float,
    write_counter: dict[str, Any],
) -> dict[str, Any]:
    audits = {}
    for arm in ARMS:
        _, audits[arm] = mirror_arm(
            model,
            parameters,
            base_state,
            displacements[arm],
            eta,
            lambda: None,
            write_counter,
            "map_precheck_writes",
            {"purpose": "map_precheck", "arm": arm},
        )
    return {
        "arms": audits,
        "gate": all(
            mirror_audit_passes(
                audits[arm], maximum_relative_error, maximum_absolute_error
            )
            for arm in ARMS
        ),
    }


def mirror_map_fidelity_passes(
    audit: Mapping[str, Any],
    maximum_relative_error: float,
    maximum_absolute_error: float,
) -> bool:
    return bool(
        math.isfinite(float(audit["relative_l2_error"]))
        and math.isfinite(float(audit["max_absolute_error"]))
        and math.isfinite(float(audit["post_evaluation_relative_l2_error"]))
        and math.isfinite(float(audit["post_evaluation_max_absolute_error"]))
        and float(audit["relative_l2_error"]) <= maximum_relative_error
        and float(audit["max_absolute_error"]) <= maximum_absolute_error
        and float(audit["post_evaluation_relative_l2_error"])
        <= maximum_relative_error
        and float(audit["post_evaluation_max_absolute_error"])
        <= maximum_absolute_error
    )


def mirror_restoration_passes(audit: Mapping[str, Any]) -> bool:
    return bool(
        audit.get("parameter_delta_sign")
        == "positive_eta_times_frozen_displacement"
        and audit.get("state_restored") is True
        and audit.get("rng_unchanged") is True
        and audit.get("mode_unchanged") is True
        and audit.get("gradients_unchanged") is True
        and audit.get("buffers_unchanged") is True
        and audit.get("rng_restored") is True
        and audit.get("mode_restored") is True
        and audit.get("gradients_restored") is True
        and audit.get("buffers_restored") is True
    )


def numeric_values_finite(values: Sequence[Any]) -> bool:
    return all(
        type(value) in (int, float) and math.isfinite(float(value)) for value in values
    )


def attribution_summary_by_certificate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        name: {
            field: {
                "mean": statistics.fmean(
                    float(row["comparisons"][name][field]) for row in rows
                ),
                "maximum": max(
                    float(row["comparisons"][name][field]) for row in rows
                ),
                "minimum": min(
                    float(row["comparisons"][name][field]) for row in rows
                ),
                "positive_count": sum(
                    float(row["comparisons"][name][field]) > 0.0 for row in rows
                ),
            }
            for field in (
                "first_order_deterioration",
                "odd_finite_radius_remainder",
                "even_deterioration",
                "finite_deterioration",
            )
        }
        for name in CERTIFICATES
    }


def failed_risk_posthoc_rows(
    rows: Sequence[Mapping[str, Any]], arithmetic_tolerance: float
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["unit_id"] not in EXPECTED_M6_FAILED_UNITS:
            continue
        comparison = row["comparisons"]["risk"]
        gap = float(comparison["finite_minus_first_order"])
        result.append(
            {
                "unit_id": row["unit_id"],
                **{key: float(value) for key, value in comparison.items()},
                "signed_odd_remainder_fraction_of_nonlinear_gap": (
                    float(comparison["odd_finite_radius_remainder"]) / gap
                    if abs(gap) > arithmetic_tolerance
                    else "undefined_near_zero_nonlinear_gap"
                ),
                "signed_even_fraction_of_nonlinear_gap": (
                    float(comparison["even_deterioration"]) / gap
                    if abs(gap) > arithmetic_tolerance
                    else "undefined_near_zero_nonlinear_gap"
                ),
                "posthoc_descriptive_only": True,
            }
        )
    return result


def nominal_mirror_bound_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bounds = [
        row["arms"][arm]["nominal_mirror_map_error_bound"]
        for row in rows
        for arm in ARMS
    ]
    return {
        "maximum_midpoint_relative_l2_error_upper_bound": max(
            float(item["midpoint_relative_l2_error_upper_bound"])
            for item in bounds
        ),
        "maximum_midpoint_absolute_error_upper_bound": max(
            float(item["midpoint_max_absolute_error_upper_bound"])
            for item in bounds
        ),
        "interpretation": "nominal_mirror_within_registered_map_error_bound",
    }


def evaluate_mirror_panel(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    split_units: Sequence[Mapping[str, Any]],
    pair_map: Mapping[str, Any],
    risk_features: Mapping[int, Any],
    displacements: Mapping[str, Mapping[str, Any]],
    reference_norm: float,
    source_rows: Sequence[Mapping[str, Any]],
    sealed_unit_ids: set[str],
    write_counter: dict[str, Any],
) -> dict[str, Any]:
    eta = float(protocol["design"]["virtual_scale_eta"])
    score_floor = float(protocol["design"]["score_scale_floor"])
    arithmetic_tolerance = float(
        protocol["validity_gates"]["decomposition_arithmetic_tolerance"]
    )
    maximum_relative = float(
        protocol["validity_gates"]["maximum_map_relative_l2_error"]
    )
    maximum_absolute = float(
        protocol["validity_gates"]["maximum_map_absolute_error"]
    )
    rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    evaluated_ids: list[str] = []
    for unit, source_row in zip(split_units, source_rows):
        unit_id = str(unit["unit_id"])
        if unit_id in sealed_unit_ids:
            raise M7MirrorCurvatureError("sealed unit reached mirror evaluator")
        arm_rows: dict[str, Any] = {}
        arm_audits: dict[str, Any] = {}
        for arm in ARMS:
            plus_losses, arm_audit = mirror_arm(
                model,
                parameters,
                base_state,
                displacements[arm],
                eta,
                lambda unit=unit: {
                    name: float(value)
                    for name, value in core.rp.evaluate_block_losses(
                        model,
                        dict(risk_features),
                        dict(pair_map),
                        [int(unit["risk_row_index"])],
                        [str(unit["sample_id"])],
                        dict(config),
                    ).items()
                    if name in CERTIFICATES
                },
                write_counter,
                "positive_loss_writes",
                {
                    "purpose": "positive_loss",
                    "unit_id": unit_id,
                    "sample_id": str(unit["sample_id"]),
                    "arm": arm,
                },
            )
            decompositions = {}
            for name in CERTIFICATES:
                initial = float(source_row["initial_losses"][name])
                negative = float(source_row["arms"][arm]["final_losses"][name])
                positive = float(plus_losses[name])
                norm = float(source_row["gradient_norms"][name])
                denominator = eta * norm * reference_norm
                forward = float(
                    source_row["arms"][arm]["normalized_actual_scores"][name]
                )
                mirror = (positive - initial) / denominator
                first = float(source_row["arms"][arm]["first_order_scores"][name])
                decomposition = arm_decomposition(forward, mirror, first)
                direct_central = (positive - negative) / (2.0 * denominator)
                direct_even = (positive + negative - 2.0 * initial) / (
                    2.0 * denominator
                )
                decomposition.update(
                    {
                        "normalizing_scale": denominator,
                        "initial_loss_frozen_M6": initial,
                        "negative_loss_frozen_M6": negative,
                        "positive_loss_new_M7": positive,
                        "direct_central_score": direct_central,
                        "direct_centered_even_score": direct_even,
                        "central_formula_residual": decomposition["central_score"]
                        - direct_central,
                        "centered_even_formula_residual": decomposition[
                            "centered_even_score"
                        ]
                        - direct_even,
                        "nominal_mirror_secant_curvature_proxy": 2.0
                        * direct_even
                        / eta,
                    }
                )
                decompositions[name] = decomposition
            negative_audit = source_row["arm_audits"][arm]
            positive_relative = max(
                float(arm_audit["relative_l2_error"]),
                float(arm_audit["post_evaluation_relative_l2_error"]),
            )
            positive_absolute = max(
                float(arm_audit["max_absolute_error"]),
                float(arm_audit["post_evaluation_max_absolute_error"]),
            )
            nominal_bound = {
                "negative_relative_l2_error_frozen_M6": float(
                    negative_audit["relative_l2_error"]
                ),
                "positive_relative_l2_error_new_M7": positive_relative,
                "midpoint_relative_l2_error_upper_bound": 0.5
                * (
                    float(negative_audit["relative_l2_error"])
                    + positive_relative
                ),
                "negative_max_absolute_error_frozen_M6": float(
                    negative_audit["max_absolute_error"]
                ),
                "positive_max_absolute_error_new_M7": positive_absolute,
                "midpoint_max_absolute_error_upper_bound": 0.5
                * (
                    float(negative_audit["max_absolute_error"])
                    + positive_absolute
                ),
                "interpretation": "nominal_mirror_within_registered_map_error_bound",
            }
            arm_rows[arm] = {
                "positive_losses": {
                    name: float(plus_losses[name]) for name in CERTIFICATES
                },
                "decompositions": decompositions,
                "nominal_mirror_map_error_bound": nominal_bound,
            }
            arm_audits[arm] = arm_audit
        comparisons = {}
        for name in CERTIFICATES:
            comparison = comparison_decomposition(
                arm_rows["fallback"]["decompositions"][name],
                arm_rows["selected"]["decompositions"][name],
                score_floor,
                float(source_row["finite_deteriorations"][name]),
            )
            comparisons[name] = comparison
            table_rows.append(
                {
                    "unit_id": unit_id,
                    "sample_id": str(unit["sample_id"]),
                    "certificate": name,
                    "m6_unit_covered": bool(source_row["covered"]),
                    "m6_failed_unit_posthoc": not bool(source_row["covered"]),
                    "m6_failed_cell_posthoc": float(
                        source_row["finite_deteriorations"][name]
                    )
                    > m6.ALLOWED_RELATIVE_DROP + m6.ARITHMETIC_TOLERANCE,
                    "score_denominator": comparison["score_denominator"],
                    "first_order_deterioration": comparison[
                        "first_order_deterioration"
                    ],
                    "odd_finite_radius_remainder": comparison[
                        "odd_finite_radius_remainder"
                    ],
                    "even_deterioration": comparison["even_deterioration"],
                    "central_deterioration": comparison[
                        "central_deterioration"
                    ],
                    "finite_deterioration": comparison["finite_deterioration"],
                    "finite_minus_first_order": comparison[
                        "finite_minus_first_order"
                    ],
                    "finite_identity_residual": comparison[
                        "finite_identity_residual"
                    ],
                    "gap_identity_residual": comparison[
                        "gap_identity_residual"
                    ],
                    "fallback_centered_even_score": arm_rows["fallback"][
                        "decompositions"
                    ][name]["centered_even_score"],
                    "selected_centered_even_score": arm_rows["selected"][
                        "decompositions"
                    ][name]["centered_even_score"],
                }
            )
        rows.append(
            {
                "unit_id": unit_id,
                "sample_id": str(unit["sample_id"]),
                "m6_covered": bool(source_row["covered"]),
                "arms": arm_rows,
                "arm_audits": arm_audits,
                "comparisons": comparisons,
            }
        )
        evaluated_ids.append(unit_id)

    audits = [row["arm_audits"][arm] for row in rows for arm in ARMS]
    residuals = []
    numeric_values: list[float] = []
    for row in rows:
        for arm in ARMS:
            for name in CERTIFICATES:
                decomposition = row["arms"][arm]["decompositions"][name]
                residuals.extend(
                    abs(float(decomposition[key]))
                    for key in (
                        "forward_identity_residual",
                        "taylor_identity_residual",
                        "central_formula_residual",
                        "centered_even_formula_residual",
                    )
                )
                numeric_values.extend(
                    float(value)
                    for value in decomposition.values()
                    if type(value) in (int, float)
                )
        for name in CERTIFICATES:
            comparison = row["comparisons"][name]
            residuals.extend(
                abs(float(comparison[key]))
                for key in ("finite_identity_residual", "gap_identity_residual")
            )
            numeric_values.extend(float(value) for value in comparison.values())

    unique_ids = set(evaluated_ids)
    expected_ids = {str(unit["unit_id"]) for unit in split_units}
    all_units = bool(
        len(rows) == EXPECTED_UNITS
        and len(table_rows) == EXPECTED_UNITS * len(CERTIFICATES)
        and len(evaluated_ids) == len(unique_ids) == EXPECTED_UNITS
        and unique_ids == expected_ids
        and unique_ids.isdisjoint(sealed_unit_ids)
        and int(write_counter.get("positive_loss_writes", 0))
        == EXPECTED_UNITS * len(ARMS)
        and int(write_counter.get("negative_loss_writes", 0)) == 0
        and int(write_counter.get("sealed_loss_writes", 0)) == 0
        and int(write_counter.get("endpoint_loss_writes", 0)) == 0
        and write_counter.get("events")
        == expected_mutation_events(protocol["calibration_allowlist"])
    )
    summaries = attribution_summary_by_certificate(rows)
    failed_risk_rows = failed_risk_posthoc_rows(rows, arithmetic_tolerance)
    return {
        "status": "evaluated",
        "rows": rows,
        "table_rows": table_rows,
        "summary_by_certificate": summaries,
        "known_M6_failed_risk_rows_posthoc": failed_risk_rows,
        "new_evaluated_unit_ids": evaluated_ids,
        "sealed_unit_intersection": sorted(unique_ids.intersection(sealed_unit_ids)),
        "write_ledger": dict(write_counter),
        "nominal_mirror_map_error_bound_summary": nominal_mirror_bound_summary(
            rows
        ),
        "map_fidelity_gate": all(
            mirror_map_fidelity_passes(audit, maximum_relative, maximum_absolute)
            for audit in audits
        ),
        "exact_restoration_gate": all(
            mirror_restoration_passes(audit) for audit in audits
        ),
        "algebra_identity_gate": bool(
            residuals and max(residuals) <= arithmetic_tolerance
        ),
        "maximum_absolute_identity_residual": max(residuals),
        "finite_values_gate": numeric_values_finite(numeric_values),
        "all_units_attributed_gate": all_units,
    }


def skipped_attribution(reason: str, write_counter: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "not_evaluated",
        "reason": reason,
        "rows": [],
        "table_rows": [],
        "summary_by_certificate": {},
        "known_M6_failed_risk_rows_posthoc": [],
        "new_evaluated_unit_ids": [],
        "sealed_unit_intersection": [],
        "write_ledger": dict(write_counter),
        "nominal_mirror_map_error_bound_summary": {},
        "map_fidelity_gate": False,
        "exact_restoration_gate": False,
        "algebra_identity_gate": False,
        "maximum_absolute_identity_residual": None,
        "finite_values_gate": False,
        "all_units_attributed_gate": False,
    }


def write_table_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows and any(tuple(row) != TABLE_FIELDS for row in rows):
        raise M7MirrorCurvatureError("attribution CSV schema differs")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_risk_decomposition(
    svg_path: Path,
    png_path: Path,
    table_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    risk_rows = [row for row in table_rows if row["certificate"] == "risk"]
    if len(risk_rows) != EXPECTED_UNITS:
        raise M7MirrorCurvatureError("risk decomposition plot denominator differs")
    x_values = list(range(1, EXPECTED_UNITS + 1))
    first = [float(row["first_order_deterioration"]) for row in risk_rows]
    central = [
        float(row["first_order_deterioration"])
        + float(row["odd_finite_radius_remainder"])
        for row in risk_rows
    ]
    finite = [float(row["finite_deterioration"]) for row in risk_rows]
    figure, axis = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    for x_value, first_value, central_value, finite_value in zip(
        x_values, first, central, finite
    ):
        axis.plot(
            [x_value, x_value],
            [first_value, central_value],
            color="#2f6b9a",
            linewidth=2.1,
            solid_capstyle="round",
            zorder=1,
        )
        axis.plot(
            [x_value, x_value],
            [central_value, finite_value],
            color="#d97924",
            linewidth=2.1,
            solid_capstyle="round",
            zorder=1,
        )
    axis.scatter(
        x_values,
        first,
        marker="o",
        facecolors="white",
        edgecolors="#2f6b9a",
        s=38,
        linewidths=1.4,
        label="First-order deterioration",
        zorder=3,
    )
    axis.scatter(
        x_values,
        central,
        marker="s",
        color="#d97924",
        s=28,
        label="After odd finite-radius remainder",
        zorder=3,
    )
    axis.scatter(
        x_values,
        finite,
        marker="D",
        color="#202124",
        s=32,
        label="Frozen M6 finite deterioration",
        zorder=4,
    )
    failed_x = [
        index + 1
        for index, row in enumerate(risk_rows)
        if str(row["unit_id"]) in EXPECTED_M6_FAILED_UNITS
    ]
    failed_y = [finite[index - 1] for index in failed_x]
    axis.scatter(
        failed_x,
        failed_y,
        marker="o",
        facecolors="none",
        edgecolors="#c62828",
        s=105,
        linewidths=1.6,
        label="Known M6 failures (post-hoc)",
        zorder=5,
    )
    axis.axhline(
        m6.ALLOWED_RELATIVE_DROP,
        color="#c62828",
        linestyle="--",
        linewidth=1.2,
        label="M6 threshold = 0.04 (descriptive)",
        zorder=0,
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8, zorder=0)
    axis.set_xlim(0.35, EXPECTED_UNITS + 0.65)
    axis.set_xticks(x_values)
    axis.set_xticklabels([str(value) for value in x_values], fontsize=8)
    axis.set_xlabel("Frozen M6 calibration unit order")
    axis.set_ylabel("Selected-vs-fallback normalized deterioration")
    axis.set_title(
        "M7 risk decomposition at fixed radius: first-order + odd remainder + even term"
    )
    axis.grid(False)
    axis.legend(frameon=False, ncol=2, fontsize=8.5, loc="best")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(svg_path, format="svg")
    figure.savefig(png_path, format="png", dpi=300)
    plt.close(figure)


def interpretation_markdown(
    route: str,
    attribution: Mapping[str, Any],
) -> str:
    failed_lines = []
    for row in attribution["known_M6_failed_risk_rows_posthoc"]:
        failed_lines.append(
            "| {unit} | {first:.9f} | {odd:.9f} | {even:.9f} | {finite:.9f} |".format(
                unit=row["unit_id"],
                first=float(row["first_order_deterioration"]),
                odd=float(row["odd_finite_radius_remainder"]),
                even=float(row["even_deterioration"]),
                finite=float(row["finite_deterioration"]),
            )
        )
    return (
        "# CDR-OPC M7 Mirror Symmetric Attribution\n\n"
        f"Route: `{route}`. This route certifies diagnostic validity only; it does "
        "not certify protection efficacy.\n\n"
        "## Frozen M6 risk failures (post-hoc descriptive subset)\n\n"
        "| unit | first-order | odd finite-radius remainder | even term | frozen M6 finite |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(failed_lines)
        + "\n\n"
        "The additive identity is `finite = first-order + odd finite-radius remainder + even`. "
        "The even term is a fixed-radius centered even component, not a pure Hessian "
        "estimate and not a causal mechanism. The four rows were known before M7 and "
        "are not an independent subgroup confirmation.\n\n"
        "The nominal mirror midpoint error is bounded from the frozen M6 negative-map "
        "audit and the new M7 positive-map audit. The maximum registered relative-L2 "
        "midpoint upper bound is "
        f"`{attribution['nominal_mirror_map_error_bound_summary']['maximum_midpoint_relative_l2_error_upper_bound']}`. "
        "Therefore the secant-curvature field is a map-error-bounded proxy, not an "
        "exact geometric mirror or a pure Hessian estimate.\n\n"
        "M6 remains stopped at 16/20 calibration coverage. No negative arm was rerun, "
        "no sealed unit was opened, and this result does not authorize radius tuning, "
        "dose tuning, training, checkpointing, or generation.\n"
    )


def preflight(path: Path) -> dict[str, Any]:
    import torch

    (
        protocol,
        _,
        _,
        m6_diagnostics,
        _,
        config,
        _,
        _,
        selected_targets,
        _,
        risk,
        units,
    ) = context(path)
    if not torch.cuda.is_available():
        raise M7MirrorCurvatureError("CUDA unavailable")
    tokenizer = core.tarc.load_tokenizer(str(config["model_name_or_path"]))
    split_units, _, _, structural = prepare_calibration(
        protocol, config, selected_targets, risk, units, tokenizer
    )
    sealed_ids = {
        str(unit["unit_id"]) for unit in units if unit["split"] == "sealed_audit"
    }
    calibration_ids = {str(unit["unit_id"]) for unit in split_units}
    result = {
        "status": "PASS_CDRO_OPC_M7_MIRROR_CURVATURE_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "m6_route": m6_diagnostics["route"],
        "m6_calibration_covered_units": m6_diagnostics["calibration"][
            "finite_protection"
        ]["covered_units"],
        "calibration_units": len(split_units),
        "new_positive_loss_writes_registered": EXPECTED_UNITS * len(ARMS),
        "map_precheck_writes_registered": len(ARMS),
        "negative_loss_writes_registered": 0,
        "sealed_audit_opened": False,
        "calibration_sealed_disjoint": calibration_ids.isdisjoint(sealed_ids),
        "structural_audit": structural,
        "fixed_dose": protocol["design"]["fixed_dose"],
        "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def run(path: Path) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    (
        protocol,
        m6_protocol,
        _,
        m6_diagnostics,
        m1_protocol,
        config,
        old_targets,
        old_risk,
        selected_targets,
        _,
        risk,
        units,
    ) = context(path)
    runtime_head = require_clean_start(protocol)
    if not torch.cuda.is_available():
        raise M7MirrorCurvatureError("CUDA unavailable")
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = core.cspr.load_model(m1_protocol)
    model.eval()
    parameters = core.cspr.trainable_parameters(model)
    optimizer = core.frozen.make_optimizer(model, config)
    if optimizer.state:
        raise M7MirrorCurvatureError("optimizer state is not empty before replay")
    step_guard = {"calls": 0}

    def forbidden_step(*args: Any, **kwargs: Any) -> None:
        step_guard["calls"] += 1
        raise M7MirrorCurvatureError("optimizer.step is forbidden in M7")

    optimizer.step = forbidden_step
    initial_hash = core.cspr.trainable_state_hash(parameters)
    fallback, endpoint, reference_norm, replay = v4.reconstruct_fixed_direction(
        m1_protocol,
        config,
        old_targets,
        old_risk,
        model,
        tokenizer,
        parameters,
        optimizer,
        protocol["fixed_direction"],
    )
    if optimizer.state:
        raise M7MirrorCurvatureError("direction replay populated optimizer state")
    replay_hash = core.cspr.trainable_state_hash(parameters)
    if any(parameter.grad is not None for _, parameter in parameters):
        raise M7MirrorCurvatureError("direction replay left gradients attached")
    core.frsc.normalized_certificates = v4.normalize_registered_certificates
    selected_map = core.base.blend_maps(
        fallback, endpoint, float(protocol["design"]["fixed_dose"])
    )
    displacements = {"fallback": fallback, "selected": selected_map}
    base_state = m6.parameter_snapshot(parameters)
    mode_state = m6.mode_snapshot(model)
    base_rng = m6.rng_snapshot()
    base_buffers = buffer_snapshot(model)
    split_units, pair_map, risk_features, structural_audit = prepare_calibration(
        protocol, config, selected_targets, risk, units, tokenizer
    )
    source_rows = m6_diagnostics["calibration"]["finite_protection"]["rows"]
    frozen_reference_norm = float(m6_diagnostics["reference_norm"])
    solver = replay["solver_audit"]
    fixed_input_gates = {
        "m6_source_boundary": True,
        "fixed_direction": bool(replay["fixed_direction_valid"]),
        "solver_kkt": bool(
            solver["optimization_valid"]
            and float(solver["primal_violation"])
            <= float(m1_protocol["gates"]["feasibility_tolerance"])
            and float(solver["normalized_stationarity_residual"])
            <= float(m1_protocol["gates"]["kkt_tolerance"])
        ),
        "optimizer_state_empty": not bool(optimizer.state),
        "replay_hash_restored": replay_hash == initial_hash,
        "eval_mode": all(not value for value in mode_state.values()),
        "zero_step": bool(
            step_guard["calls"] == 0
            and all(
                int(item["optimizer_completed_steps"]) == 0
                for item in replay["anchor_records"].values()
            )
            and all(
                int(item["completed_optimizer_steps"]) == 0
                for item in replay["behavior_map_audits"].values()
            )
        ),
    }
    pre_source_ready = all(fixed_input_gates.values())
    source_replay = (
        replay_source_panel(
            protocol,
            config,
            model,
            parameters,
            base_state,
            split_units,
            pair_map,
            risk_features,
            fallback,
            selected_map,
            reference_norm,
            frozen_reference_norm,
            source_rows,
        )
        if pre_source_ready
        else {
            "status": "not_evaluated",
            "reason": "fixed_input_gate_failed",
            "rows": [],
            "total_units": 0,
            "reproduced_units": 0,
            "maximum_absolute_errors": {},
            "gate": False,
            "gradient_coverage_gate": False,
        }
    )
    reference_norm_error = float(reference_norm) - float(
        m6_diagnostics["reference_norm"]
    )
    reference_norm_gate = within_tolerance(
        reference_norm,
        m6_diagnostics["reference_norm"],
        float(protocol["validity_gates"]["norm_replay_absolute_tolerance"]),
        float(protocol["validity_gates"]["norm_replay_relative_tolerance"]),
    )
    source_replay["fresh_reference_norm"] = float(reference_norm)
    source_replay["frozen_M6_reference_norm"] = float(
        m6_diagnostics["reference_norm"]
    )
    source_replay["reference_norm_error"] = reference_norm_error
    source_replay["reference_norm_gate"] = reference_norm_gate
    source_replay["gate"] = bool(source_replay["gate"] and reference_norm_gate)
    source_post_integrity = integrity_audit(
        model,
        parameters,
        base_state,
        initial_hash,
        optimizer,
        mode_state,
        base_rng,
        base_buffers,
    )
    write_counter = {
        "map_precheck_writes": 0,
        "positive_loss_writes": 0,
        "negative_loss_writes": 0,
        "sealed_loss_writes": 0,
        "endpoint_loss_writes": 0,
        "events": [],
    }
    source_ready = bool(
        source_replay["gate"] and source_replay["gradient_coverage_gate"]
        and all(source_post_integrity.values())
    )
    map_precheck = (
        mirror_map_precheck(
            model,
            parameters,
            base_state,
            displacements,
            float(protocol["design"]["virtual_scale_eta"]),
            float(
                protocol["validity_gates"]["maximum_map_relative_l2_error"]
            ),
            float(protocol["validity_gates"]["maximum_map_absolute_error"]),
            write_counter,
        )
        if source_ready
        else {
            "status": "not_evaluated",
            "reason": "source_replay_gate_failed",
            "arms": {},
            "gate": False,
        }
    )
    if "status" not in map_precheck:
        map_precheck["status"] = "evaluated"
    map_precheck["write_count_gate"] = bool(
        int(write_counter["map_precheck_writes"]) == len(ARMS)
        if source_ready
        else int(write_counter["map_precheck_writes"]) == 0
    )
    map_precheck["gate"] = bool(
        map_precheck["gate"] and map_precheck["write_count_gate"]
    )
    sealed_ids = {
        str(unit["unit_id"]) for unit in units if unit["split"] == "sealed_audit"
    }
    attribution = (
        evaluate_mirror_panel(
            protocol,
            config,
            model,
            parameters,
            base_state,
            split_units,
            pair_map,
            risk_features,
            displacements,
            frozen_reference_norm,
            source_rows,
            sealed_ids,
            write_counter,
        )
        if map_precheck["gate"]
        else skipped_attribution(
            "source_replay_or_map_precheck_gate_failed", write_counter
        )
    )
    final_integrity = integrity_audit(
        model,
        parameters,
        base_state,
        initial_hash,
        optimizer,
        mode_state,
        base_rng,
        base_buffers,
    )
    gates = {
        **fixed_input_gates,
        "source_replay": bool(source_replay["gate"]),
        "gradient_coverage": bool(source_replay["gradient_coverage_gate"]),
        "source_post_integrity": all(source_post_integrity.values()),
        "mirror_map_precheck": bool(map_precheck["gate"]),
        "mirror_map_fidelity": bool(attribution["map_fidelity_gate"]),
        "exact_restoration": bool(attribution["exact_restoration_gate"]),
        "algebra_identity": bool(attribution["algebra_identity_gate"]),
        "finite_values": bool(attribution["finite_values_gate"]),
        "all_units_attributed": bool(attribution["all_units_attributed_gate"]),
        "final_integrity": all(final_integrity.values()),
    }
    route = expected_route(gates)
    if (
        route != "PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC"
        or attribution["status"] != "evaluated"
    ):
        raise M7MirrorCurvatureError(
            f"M7 validity stop {route}; no decision or partial result was sealed"
        )
    root = repo_path(protocol["output_root"])
    staging = tempfile.TemporaryDirectory(prefix=".m7_stage_", dir=str(root.parent))
    stage_root = Path(staging.name)
    run_root = stage_root / "run"
    analysis_root = stage_root / "analysis"
    figure_root = stage_root / "figures"
    diagnostics = {
        "schema": "step3_cdro_opc_m7_mirror_curvature_diagnostics_v1",
        "route": route,
        "runtime_head": runtime_head,
        "claim_scope": "post_M6_fixed_radius_validity_only_nominal_mirror_decomposition_not_fresh_confirmation",
        "fresh_external_confirmation_claimed": False,
        "M6_stop_preserved": True,
        "M6_route": m6_diagnostics["route"],
        "M6_calibration_covered_units": 16,
        "M6_known_failures_posthoc": list(EXPECTED_M6_FAILED_UNITS),
        "fixed_dose": protocol["design"]["fixed_dose"],
        "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
        "reference_norm": frozen_reference_norm,
        "fixed_direction_replay": replay,
        "fixed_input_audit": fixed_input_gates,
        "structural_audit": structural_audit,
        "source_replay": source_replay,
        "source_post_integrity": source_post_integrity,
        "mirror_map_precheck": map_precheck,
        "attribution": attribution,
        "validity_gates": gates,
        "final_integrity": final_integrity,
        "optimizer_objects": 1,
        "optimizer_step_guard_calls": step_guard["calls"],
        "optimizer_steps": 0,
        "training_authorized": False,
        "negative_arm_recomputed": False,
        "sealed_audit_opened": False,
        "checkpoint_writes": 0,
        "candidate_generation": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    decision = {
        "schema": "step3_cdro_opc_m7_mirror_curvature_decision_v1",
        "route": route,
        "validity_only": True,
        "M6_stop_preserved": True,
        "M6_route": EXPECTED_M6_ROUTE,
        "M6_calibration_covered_units": 16,
        "source_reproduced_units": source_replay["reproduced_units"],
        "attributed_unit_certificate_cells": len(attribution["table_rows"]),
        "write_ledger": attribution["write_ledger"],
        "sealed_audit_opened": False,
        "fresh_external_confirmation_claimed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "interpret fixed-radius decomposition only; preserve M6 stop; do not train"
            if route.startswith("PASS")
            else "stop without retry, radius tuning, dose tuning, sealed access, or training"
        ),
    }
    environment = {
        "head": runtime_head,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "gpu": torch.cuda.get_device_name(0),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "wall_seconds": time.time() - started,
    }
    diagnostics_path = run_root / "cdro_opc_m7_mirror_curvature_diagnostics.json"
    decision_path = run_root / "cdro_opc_m7_mirror_curvature_decision.json"
    environment_path = run_root / "cdro_opc_m7_mirror_curvature_environment.json"
    report_path = run_root / "CDRO_OPC_M7_MIRROR_CURVATURE_REPORT.md"
    table_path = analysis_root / "cdro_opc_m7_mirror_curvature_attribution.csv"
    interpretation_path = analysis_root / "CDRO_OPC_M7_MIRROR_CURVATURE_INTERPRETATION.md"
    svg_path = figure_root / "cdro_opc_m7_risk_decomposition.svg"
    png_path = figure_root / "cdro_opc_m7_risk_decomposition_300dpi.png"
    checklist_path = run_root / "复现清单.json"
    reproduction_path = run_root / "reproduction_manifest.json"
    write_json(diagnostics_path, diagnostics)
    write_json(decision_path, decision)
    write_json(environment_path, environment)
    write_table_csv(table_path, attribution["table_rows"])
    render_risk_decomposition(svg_path, png_path, attribution["table_rows"])
    interpretation_path.write_text(
        interpretation_markdown(route, attribution), encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# CDR-OPC M7 Mirror Curvature Attribution\n\n"
        f"Route: `{route}`. Fixed dose: `{protocol['design']['fixed_dose']}`. "
        f"Virtual scale: `{protocol['design']['virtual_scale_eta']}`. "
        f"Attributed cells: `{len(attribution['table_rows'])}/80`.\n\n"
        "This is a post-M6, validity-only fixed-radius nominal-mirror decomposition, "
        "bounded by the registered map errors, on "
        "the already opened calibration cohort. It is not fresh confirmation. "
        "M6 remains stopped at 16/20; sealed audit remained unopened. No negative "
        "arm was recomputed and no optimizer step was executed.\n",
        encoding="utf-8",
    )
    write_json(
        checklist_path,
        {
            "schema": "step3_cdro_opc_m7_chinese_reproduction_checklist_v1",
            "唯一复现命令": (
                ".\\.venv\\Scripts\\python.exe experiments/step3_response_quality_persistence/"
                "scripts/step3_cdro_opc_m7_mirror_curvature.py run"
            ),
            "校验命令": (
                ".\\.venv\\Scripts\\python.exe experiments/step3_response_quality_persistence/"
                "scripts/step3_cdro_opc_m7_mirror_curvature.py verify-run"
            ),
            "新增正向镜像损失写入": EXPECTED_UNITS * len(ARMS),
            "负向臂重算": 0,
            "sealed审计打开": False,
            "optimizer_steps": 0,
            "训练授权": False,
        },
    )
    artifact_paths = {
        "diagnostics": diagnostics_path,
        "decision": decision_path,
        "environment": environment_path,
        "report": report_path,
        "attribution_csv": table_path,
        "interpretation": interpretation_path,
        "risk_figure_svg": svg_path,
        "risk_figure_png": png_path,
        "chinese_reproduction_checklist": checklist_path,
    }
    reproduction = {
        "schema": "step3_cdro_opc_m7_reproduction_manifest_v1",
        "runtime_commit": runtime_head,
        "unique_reproduction_command": (
            ".\\.venv\\Scripts\\python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_cdro_opc_m7_mirror_curvature.py run"
        ),
        "verification_command": (
            ".\\.venv\\Scripts\\python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_cdro_opc_m7_mirror_curvature.py verify-run"
        ),
        "random_seed": int(config["seed"]),
        "parameters": {
            "fixed_dose": protocol["design"]["fixed_dose"],
            "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
            "score_scale_floor": protocol["design"]["score_scale_floor"],
            "optimizer_steps": 0,
        },
        "inputs_sha256": {
            "protocol": file_sha256(path),
            **{
                name: spec["sha256"]
                for name, spec in protocol["frozen_inputs"].items()
            },
        },
        "outputs_sha256": {
            name: file_sha256(artifact) for name, artifact in artifact_paths.items()
        },
        "environment": environment,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(reproduction_path, reproduction)
    publish_staged_directories(stage_root, root)
    staging.cleanup()
    result = {
        "route": route,
        "source_reproduced_units": source_replay["reproduced_units"],
        "attributed_cells": len(attribution["table_rows"]),
        "positive_loss_writes": attribution["write_ledger"][
            "positive_loss_writes"
        ],
        "negative_loss_writes": 0,
        "sealed_audit_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return decision


def numbers_close(left: Any, right: Any, tolerance: float) -> bool:
    if type(left) not in (int, float) or type(right) not in (int, float):
        return False
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return False
    return abs(float(left) - float(right)) <= tolerance


def deep_equal_with_tolerance(left: Any, right: Any, tolerance: float) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left == right
    if type(left) in (int, float) or type(right) in (int, float):
        return numbers_close(left, right, tolerance)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(
                deep_equal_with_tolerance(left[key], right[key], tolerance)
                for key in left
            )
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return bool(
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                deep_equal_with_tolerance(a, b, tolerance)
                for a, b in zip(left, right)
            )
        )
    return type(left) is type(right) and left == right


def verify_artifacts(
    protocol: Mapping[str, Any],
    m6_diagnostics: Mapping[str, Any],
    decision: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    if (
        decision.get("schema")
        != "step3_cdro_opc_m7_mirror_curvature_decision_v1"
        or diagnostics.get("schema")
        != "step3_cdro_opc_m7_mirror_curvature_diagnostics_v1"
        or diagnostics.get("claim_scope")
        != "post_M6_fixed_radius_validity_only_nominal_mirror_decomposition_not_fresh_confirmation"
        or decision.get("validity_only") is not True
        or decision.get("M6_stop_preserved") is not True
        or diagnostics.get("M6_stop_preserved") is not True
        or decision.get("M6_route") != EXPECTED_M6_ROUTE
        or diagnostics.get("M6_route") != EXPECTED_M6_ROUTE
        or decision.get("M6_calibration_covered_units") != 16
        or diagnostics.get("M6_calibration_covered_units") != 16
        or diagnostics.get("M6_known_failures_posthoc")
        != list(EXPECTED_M6_FAILED_UNITS)
        or diagnostics.get("fresh_external_confirmation_claimed") is not False
        or decision.get("fresh_external_confirmation_claimed") is not False
        or diagnostics.get("negative_arm_recomputed") is not False
        or diagnostics.get("sealed_audit_opened") is not False
        or decision.get("sealed_audit_opened") is not False
        or diagnostics.get("optimizer_objects") != 1
        or diagnostics.get("optimizer_step_guard_calls") != 0
        or diagnostics.get("optimizer_steps") != 0
        or decision.get("optimizer_steps") != 0
        or diagnostics.get("training_authorized") is not False
        or decision.get("training_authorized") is not False
        or diagnostics.get("checkpoint_writes") != 0
        or diagnostics.get("candidate_generation") != 0
        or diagnostics.get("core100_opened") is not False
        or diagnostics.get("heldout_opened") is not False
        or float(diagnostics.get("fixed_dose", math.nan)) != FIXED_DOSE
        or float(diagnostics.get("virtual_scale_eta", math.nan))
        != VIRTUAL_SCALE_ETA
    ):
        raise M7MirrorCurvatureError("M7 artifact boundary differs")
    final_integrity = diagnostics.get("final_integrity", {})
    if (
        set(final_integrity) != set(INTEGRITY_KEYS)
        or not all(type(final_integrity[name]) is bool for name in INTEGRITY_KEYS)
    ):
        raise M7MirrorCurvatureError("M7 final integrity registry differs")
    fixed_input = diagnostics.get("fixed_input_audit", {})
    expected_fixed_keys = {
        "m6_source_boundary",
        "fixed_direction",
        "solver_kkt",
        "optimizer_state_empty",
        "replay_hash_restored",
        "eval_mode",
        "zero_step",
    }
    if set(fixed_input) != expected_fixed_keys or not all(
        type(value) is bool for value in fixed_input.values()
    ):
        raise M7MirrorCurvatureError("M7 fixed input audit registry differs")
    replay = diagnostics["fixed_direction_replay"]
    solver = replay["solver_audit"]
    reproduced_fixed = {
        "m6_source_boundary": True,
        "fixed_direction": bool(replay["fixed_direction_valid"]),
        "solver_kkt": bool(
            solver["optimization_valid"]
            and float(solver["primal_violation"])
            <= float(protocol["m1_gate_snapshot"]["feasibility_tolerance"])
            and float(solver["normalized_stationarity_residual"])
            <= float(protocol["m1_gate_snapshot"]["kkt_tolerance"])
        ),
        "optimizer_state_empty": bool(final_integrity["optimizer_state_empty"]),
        "replay_hash_restored": bool(fixed_input["replay_hash_restored"]),
        "eval_mode": bool(fixed_input["eval_mode"]),
        "zero_step": bool(
            diagnostics["optimizer_step_guard_calls"] == 0
            and all(
                int(item["optimizer_completed_steps"]) == 0
                for item in replay["anchor_records"].values()
            )
            and all(
                int(item["completed_optimizer_steps"]) == 0
                for item in replay["behavior_map_audits"].values()
            )
        ),
    }
    if fixed_input != reproduced_fixed:
        raise M7MirrorCurvatureError("M7 fixed input audit does not reproduce")

    source_replay = diagnostics.get("source_replay", {})
    source_rows = m6_diagnostics["calibration"]["finite_protection"]["rows"]
    allowlist = protocol["calibration_allowlist"]
    expected_source_top_keys = {
        "status",
        "rows",
        "total_units",
        "reproduced_units",
        "maximum_absolute_errors",
        "gate",
        "gradient_coverage_gate",
        "fresh_reference_norm",
        "frozen_M6_reference_norm",
        "reference_norm_error",
        "reference_norm_gate",
    }
    if (
        set(source_replay) != expected_source_top_keys
        or source_replay.get("status") != "evaluated"
        or source_replay.get("total_units") != EXPECTED_UNITS
        or source_replay.get("reproduced_units") != EXPECTED_UNITS
        or source_replay.get("gate") is not True
        or source_replay.get("gradient_coverage_gate") is not True
        or source_replay.get("reference_norm_gate") is not True
        or len(source_replay.get("rows", [])) != EXPECTED_UNITS
    ):
        raise M7MirrorCurvatureError("M7 source replay registry differs")
    reproduced_source_rows = []
    for item, source, stored in zip(
        allowlist, source_rows, source_replay["rows"]
    ):
        fresh_coverage = dict(stored.get("fresh_coverage_structure", {}))
        fresh_coverage["loss_means"] = stored.get("fresh_loss_means", {})
        fresh_coverage["gradient_norms"] = stored.get(
            "fresh_raw_gradient_norms", {}
        )
        expected_source = source_replay_row(
            item,
            source,
            fresh_coverage,
            stored.get("fresh_certificate_norms", {}),
            stored.get("fresh_first_order_scores", {}),
            protocol,
            float(m6_diagnostics["reference_norm"]),
        )
        expected_unit_integrity = {
            "unchanged_during_replay": {
                "state_tensor_equal": True,
                "gradients_unchanged": True,
                "mode_unchanged": True,
                "rng_unchanged": True,
                "buffers_unchanged": True,
            },
            "restored_after_replay": {
                "state_tensor_equal": True,
                "gradients_restored": True,
                "mode_restored": True,
                "rng_restored": True,
                "buffers_restored": True,
            },
            "gate": True,
        }
        expected_source["source_unit_integrity"] = expected_unit_integrity
        expected_source["gate"] = bool(expected_source["gate"])
        if not deep_equal_with_tolerance(
            stored, expected_source, SOURCE_ARITHMETIC_TOLERANCE
        ):
            raise M7MirrorCurvatureError("M7 source replay row does not reproduce")
        reproduced_source_rows.append(expected_source)
    error_groups = {
        "loss_means": [
            abs(float(value))
            for row in reproduced_source_rows
            for value in row["errors"]["loss_means"].values()
        ],
        "raw_gradient_norms": [
            abs(float(value))
            for row in reproduced_source_rows
            for value in row["errors"]["raw_gradient_norms"].values()
        ],
        "certificate_norms": [
            abs(float(value))
            for row in reproduced_source_rows
            for value in row["errors"]["certificate_norms"].values()
        ],
        "first_order_scores": [
            abs(float(value))
            for row in reproduced_source_rows
            for arm in ARMS
            for value in row["errors"]["first_order_scores"][arm].values()
        ],
    }
    expected_maximum_errors = {
        name: max(values) for name, values in error_groups.items()
    }
    reproduced_source_unit_count = sum(
        bool(row["gate"]) for row in reproduced_source_rows
    )
    reproduced_source_rows_gate = bool(
        len(reproduced_source_rows) == EXPECTED_UNITS
        and reproduced_source_unit_count == EXPECTED_UNITS
    )
    reproduced_gradient_coverage_gate = all(
        bool(row["gradient_coverage_gate"]) for row in reproduced_source_rows
    )
    if not deep_equal_with_tolerance(
        source_replay["maximum_absolute_errors"],
        expected_maximum_errors,
        SOURCE_ARITHMETIC_TOLERANCE,
    ):
        raise M7MirrorCurvatureError("M7 source replay maxima do not reproduce")
    reference_norm = float(diagnostics["reference_norm"])
    fresh_reference_norm = source_replay.get("fresh_reference_norm")
    frozen_reference_norm = float(m6_diagnostics["reference_norm"])
    reproduced_reference_norm_gate = within_tolerance(
        fresh_reference_norm,
        frozen_reference_norm,
        float(protocol["validity_gates"]["norm_replay_absolute_tolerance"]),
        float(protocol["validity_gates"]["norm_replay_relative_tolerance"]),
    )
    reproduced_source_gate = bool(
        reproduced_source_rows_gate and reproduced_reference_norm_gate
    )
    if (
        reference_norm != frozen_reference_norm
        or source_replay.get("frozen_M6_reference_norm") != frozen_reference_norm
        or not numbers_close(
            source_replay.get("reference_norm_error"),
            float(fresh_reference_norm) - frozen_reference_norm,
            SOURCE_ARITHMETIC_TOLERANCE,
        )
        or source_replay.get("reproduced_units")
        != reproduced_source_unit_count
        or source_replay.get("gate") != reproduced_source_gate
        or source_replay.get("gradient_coverage_gate")
        != reproduced_gradient_coverage_gate
        or source_replay.get("reference_norm_gate")
        != reproduced_reference_norm_gate
    ):
        raise M7MirrorCurvatureError("M7 reference norm does not reproduce")
    source_post_integrity = diagnostics.get("source_post_integrity", {})
    if (
        set(source_post_integrity) != set(INTEGRITY_KEYS)
        or not all(source_post_integrity.values())
    ):
        raise M7MirrorCurvatureError("M7 source post-integrity differs")

    attribution = diagnostics.get("attribution", {})
    rows = attribution.get("rows", [])
    table_rows = attribution.get("table_rows", [])
    source_by_unit = {str(row["unit_id"]): row for row in source_rows}
    allowed_ids = [str(item["unit_id"]) for item in allowlist]
    if (
        set(attribution)
        != {
            "status",
            "rows",
            "table_rows",
            "summary_by_certificate",
            "known_M6_failed_risk_rows_posthoc",
            "new_evaluated_unit_ids",
            "sealed_unit_intersection",
            "write_ledger",
            "nominal_mirror_map_error_bound_summary",
            "map_fidelity_gate",
            "exact_restoration_gate",
            "algebra_identity_gate",
            "maximum_absolute_identity_residual",
            "finite_values_gate",
            "all_units_attributed_gate",
        }
        or
        attribution.get("status") != "evaluated"
        or len(rows) != EXPECTED_UNITS
        or len(table_rows) != EXPECTED_UNITS * len(CERTIFICATES)
        or [str(row.get("unit_id")) for row in rows] != allowed_ids
        or attribution.get("new_evaluated_unit_ids") != allowed_ids
        or attribution.get("sealed_unit_intersection") != []
        or len(set(allowed_ids)) != EXPECTED_UNITS
    ):
        raise M7MirrorCurvatureError("M7 attribution denominator differs")
    table_map = {
        (str(row["unit_id"]), str(row["certificate"])): row for row in table_rows
    }
    if len(table_map) != EXPECTED_UNITS * len(CERTIFICATES):
        raise M7MirrorCurvatureError("M7 attribution table has duplicate cells")
    tolerance = float(
        protocol["validity_gates"]["decomposition_arithmetic_tolerance"]
    )
    all_residuals: list[float] = []
    all_numeric: list[float] = []
    all_audits = []
    for row in rows:
        unit_id = str(row["unit_id"])
        source = source_by_unit.get(unit_id)
        if (
            source is None
            or set(row)
            != {
                "unit_id",
                "sample_id",
                "m6_covered",
                "arms",
                "arm_audits",
                "comparisons",
            }
            or str(row.get("sample_id")) != str(source["sample_id"])
            or bool(row.get("m6_covered")) != bool(source["covered"])
            or set(row.get("arms", {})) != set(ARMS)
            or set(row.get("arm_audits", {})) != set(ARMS)
            or set(row.get("comparisons", {})) != set(CERTIFICATES)
        ):
            raise M7MirrorCurvatureError("M7 attribution row identity differs")
        for arm in ARMS:
            arm_row = row["arms"][arm]
            if (
                set(arm_row)
                != {
                    "positive_losses",
                    "decompositions",
                    "nominal_mirror_map_error_bound",
                }
                or set(arm_row["positive_losses"]) != set(CERTIFICATES)
                or set(arm_row["decompositions"]) != set(CERTIFICATES)
            ):
                raise M7MirrorCurvatureError("M7 arm registry differs")
            positive_audit = row["arm_audits"][arm]
            all_audits.append(positive_audit)
            negative_audit = source["arm_audits"][arm]
            positive_relative = max(
                float(positive_audit["relative_l2_error"]),
                float(positive_audit["post_evaluation_relative_l2_error"]),
            )
            positive_absolute = max(
                float(positive_audit["max_absolute_error"]),
                float(positive_audit["post_evaluation_max_absolute_error"]),
            )
            expected_bound = {
                "negative_relative_l2_error_frozen_M6": float(
                    negative_audit["relative_l2_error"]
                ),
                "positive_relative_l2_error_new_M7": positive_relative,
                "midpoint_relative_l2_error_upper_bound": 0.5
                * (
                    float(negative_audit["relative_l2_error"])
                    + positive_relative
                ),
                "negative_max_absolute_error_frozen_M6": float(
                    negative_audit["max_absolute_error"]
                ),
                "positive_max_absolute_error_new_M7": positive_absolute,
                "midpoint_max_absolute_error_upper_bound": 0.5
                * (
                    float(negative_audit["max_absolute_error"])
                    + positive_absolute
                ),
                "interpretation": "nominal_mirror_within_registered_map_error_bound",
            }
            if not deep_equal_with_tolerance(
                arm_row["nominal_mirror_map_error_bound"],
                expected_bound,
                tolerance,
            ):
                raise M7MirrorCurvatureError(
                    "M7 nominal mirror error bound does not reproduce"
                )
            for name in CERTIFICATES:
                stored = arm_row["decompositions"][name]
                initial = float(source["initial_losses"][name])
                negative = float(source["arms"][arm]["final_losses"][name])
                positive = float(arm_row["positive_losses"][name])
                norm = float(source["gradient_norms"][name])
                denominator = VIRTUAL_SCALE_ETA * norm * reference_norm
                forward = float(
                    source["arms"][arm]["normalized_actual_scores"][name]
                )
                first = float(source["arms"][arm]["first_order_scores"][name])
                mirror = (positive - initial) / denominator
                reproduced = arm_decomposition(forward, mirror, first)
                reproduced.update(
                    {
                        "normalizing_scale": denominator,
                        "initial_loss_frozen_M6": initial,
                        "negative_loss_frozen_M6": negative,
                        "positive_loss_new_M7": positive,
                        "direct_central_score": (positive - negative)
                        / (2.0 * denominator),
                        "direct_centered_even_score": (
                            positive + negative - 2.0 * initial
                        )
                        / (2.0 * denominator),
                    }
                )
                reproduced["central_formula_residual"] = (
                    reproduced["central_score"]
                    - reproduced["direct_central_score"]
                )
                reproduced["centered_even_formula_residual"] = (
                    reproduced["centered_even_score"]
                    - reproduced["direct_centered_even_score"]
                )
                reproduced["nominal_mirror_secant_curvature_proxy"] = (
                    2.0 * reproduced["direct_centered_even_score"]
                    / VIRTUAL_SCALE_ETA
                )
                if set(stored) != set(reproduced) or not all(
                    numbers_close(stored[key], reproduced[key], tolerance)
                    for key in reproduced
                ):
                    raise M7MirrorCurvatureError(
                        "M7 arm decomposition does not reproduce"
                    )
                all_residuals.extend(
                    abs(float(stored[key]))
                    for key in (
                        "forward_identity_residual",
                        "taylor_identity_residual",
                        "central_formula_residual",
                        "centered_even_formula_residual",
                    )
                )
                all_numeric.extend(float(value) for value in stored.values())
        for name in CERTIFICATES:
            stored_comparison = row["comparisons"][name]
            reproduced_comparison = comparison_decomposition(
                row["arms"]["fallback"]["decompositions"][name],
                row["arms"]["selected"]["decompositions"][name],
                SCORE_SCALE_FLOOR,
                float(source["finite_deteriorations"][name]),
            )
            if set(stored_comparison) != set(reproduced_comparison) or not all(
                numbers_close(
                    stored_comparison[key], reproduced_comparison[key], tolerance
                )
                for key in reproduced_comparison
            ):
                raise M7MirrorCurvatureError(
                    "M7 comparison decomposition does not reproduce"
                )
            flat = table_map.get((unit_id, name))
            expected_flat = {
                "unit_id": unit_id,
                "sample_id": str(source["sample_id"]),
                "certificate": name,
                "m6_unit_covered": bool(source["covered"]),
                "m6_failed_unit_posthoc": not bool(source["covered"]),
                "m6_failed_cell_posthoc": float(
                    source["finite_deteriorations"][name]
                )
                > m6.ALLOWED_RELATIVE_DROP + m6.ARITHMETIC_TOLERANCE,
                "score_denominator": reproduced_comparison["score_denominator"],
                "first_order_deterioration": reproduced_comparison[
                    "first_order_deterioration"
                ],
                "odd_finite_radius_remainder": reproduced_comparison[
                    "odd_finite_radius_remainder"
                ],
                "even_deterioration": reproduced_comparison[
                    "even_deterioration"
                ],
                "central_deterioration": reproduced_comparison[
                    "central_deterioration"
                ],
                "finite_deterioration": reproduced_comparison[
                    "finite_deterioration"
                ],
                "finite_minus_first_order": reproduced_comparison[
                    "finite_minus_first_order"
                ],
                "finite_identity_residual": reproduced_comparison[
                    "finite_identity_residual"
                ],
                "gap_identity_residual": reproduced_comparison[
                    "gap_identity_residual"
                ],
                "fallback_centered_even_score": row["arms"]["fallback"][
                    "decompositions"
                ][name]["centered_even_score"],
                "selected_centered_even_score": row["arms"]["selected"][
                    "decompositions"
                ][name]["centered_even_score"],
            }
            if (
                flat is None
                or tuple(flat) != TABLE_FIELDS
                or not deep_equal_with_tolerance(flat, expected_flat, tolerance)
            ):
                raise M7MirrorCurvatureError("M7 flattened table does not reproduce")
            all_residuals.extend(
                abs(float(stored_comparison[key]))
                for key in ("finite_identity_residual", "gap_identity_residual")
            )
            all_numeric.extend(float(value) for value in stored_comparison.values())

    expected_summary = attribution_summary_by_certificate(rows)
    expected_posthoc = failed_risk_posthoc_rows(rows, tolerance)
    expected_bound_summary = nominal_mirror_bound_summary(rows)
    if (
        not deep_equal_with_tolerance(
            attribution.get("summary_by_certificate"),
            expected_summary,
            tolerance,
        )
        or not deep_equal_with_tolerance(
            attribution.get("known_M6_failed_risk_rows_posthoc"),
            expected_posthoc,
            tolerance,
        )
        or not deep_equal_with_tolerance(
            attribution.get("nominal_mirror_map_error_bound_summary"),
            expected_bound_summary,
            tolerance,
        )
    ):
        raise M7MirrorCurvatureError("M7 attribution summaries do not reproduce")

    maximum_relative = float(
        protocol["validity_gates"]["maximum_map_relative_l2_error"]
    )
    maximum_absolute = float(
        protocol["validity_gates"]["maximum_map_absolute_error"]
    )
    precheck = diagnostics.get("mirror_map_precheck", {})
    precheck_audits = list(precheck.get("arms", {}).values())
    write_ledger = attribution.get("write_ledger", {})
    all_units_gate = bool(
        write_ledger
        == {
            "map_precheck_writes": len(ARMS),
            "positive_loss_writes": EXPECTED_UNITS * len(ARMS),
            "negative_loss_writes": 0,
            "sealed_loss_writes": 0,
            "endpoint_loss_writes": 0,
            "events": expected_mutation_events(protocol["calibration_allowlist"]),
        }
        and decision.get("write_ledger") == write_ledger
    )
    recomputed_gates = {
        **reproduced_fixed,
        "source_replay": reproduced_source_gate,
        "gradient_coverage": reproduced_gradient_coverage_gate,
        "source_post_integrity": all(source_post_integrity.values()),
        "mirror_map_precheck": bool(
            precheck.get("status") == "evaluated"
            and precheck.get("write_count_gate") is True
            and set(precheck.get("arms", {})) == set(ARMS)
            and all(
                mirror_audit_passes(audit, maximum_relative, maximum_absolute)
                for audit in precheck_audits
            )
        ),
        "mirror_map_fidelity": all(
            mirror_map_fidelity_passes(audit, maximum_relative, maximum_absolute)
            for audit in all_audits
        ),
        "exact_restoration": all(
            mirror_restoration_passes(audit) for audit in all_audits
        ),
        "algebra_identity": bool(
            all_residuals and max(all_residuals) <= tolerance
        ),
        "finite_values": numeric_values_finite(all_numeric),
        "all_units_attributed": all_units_gate,
        "final_integrity": all(final_integrity.values()),
    }
    if (
        attribution.get("map_fidelity_gate")
        != recomputed_gates["mirror_map_fidelity"]
        or attribution.get("exact_restoration_gate")
        != recomputed_gates["exact_restoration"]
        or attribution.get("algebra_identity_gate")
        != recomputed_gates["algebra_identity"]
        or attribution.get("finite_values_gate")
        != recomputed_gates["finite_values"]
        or attribution.get("all_units_attributed_gate")
        != recomputed_gates["all_units_attributed"]
        or not numbers_close(
            attribution.get("maximum_absolute_identity_residual"),
            max(all_residuals),
            tolerance,
        )
    ):
        raise M7MirrorCurvatureError("M7 attribution gates do not reproduce")
    recorded_gates = diagnostics.get("validity_gates", {})
    if recorded_gates != recomputed_gates or not strict_gate_map(recorded_gates):
        raise M7MirrorCurvatureError("M7 validity gates do not reproduce")
    route = expected_route(recomputed_gates)
    if (
        route != "PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC"
        or diagnostics.get("route") != route
        or decision.get("route") != route
        or decision.get("source_reproduced_units") != EXPECTED_UNITS
        or decision.get("attributed_unit_certificate_cells")
        != EXPECTED_UNITS * len(CERTIFICATES)
        or decision.get("next_authorized_action")
        != "interpret fixed-radius decomposition only; preserve M6 stop; do not train"
    ):
        raise M7MirrorCurvatureError("M7 route does not reproduce")
    return route


def verify(path: Path) -> dict[str, Any]:
    protocol, _, _, m6_diagnostics, *_ = context(path)
    root = repo_path(protocol["output_root"])
    run_root = root / "run"
    analysis_root = root / "analysis"
    figure_root = root / "figures"
    artifact_paths = {
        "diagnostics": run_root / "cdro_opc_m7_mirror_curvature_diagnostics.json",
        "decision": run_root / "cdro_opc_m7_mirror_curvature_decision.json",
        "environment": run_root / "cdro_opc_m7_mirror_curvature_environment.json",
        "report": run_root / "CDRO_OPC_M7_MIRROR_CURVATURE_REPORT.md",
        "attribution_csv": analysis_root
        / "cdro_opc_m7_mirror_curvature_attribution.csv",
        "interpretation": analysis_root
        / "CDRO_OPC_M7_MIRROR_CURVATURE_INTERPRETATION.md",
        "risk_figure_svg": figure_root / "cdro_opc_m7_risk_decomposition.svg",
        "risk_figure_png": figure_root
        / "cdro_opc_m7_risk_decomposition_300dpi.png",
        "chinese_reproduction_checklist": run_root / "复现清单.json",
    }
    reproduction_path = run_root / "reproduction_manifest.json"
    if not reproduction_path.is_file() or not all(
        artifact.is_file() for artifact in artifact_paths.values()
    ):
        raise M7MirrorCurvatureError("M7 artifact set is incomplete")
    reproduction = read_json(reproduction_path)
    expected_inputs = {
        "protocol": file_sha256(path),
        **{
            name: spec["sha256"] for name, spec in protocol["frozen_inputs"].items()
        },
    }
    if (
        reproduction.get("schema")
        != "step3_cdro_opc_m7_reproduction_manifest_v1"
        or reproduction.get("inputs_sha256") != expected_inputs
        or reproduction.get("outputs_sha256")
        != {
            name: file_sha256(artifact) for name, artifact in artifact_paths.items()
        }
        or reproduction.get("optimizer_steps") != 0
        or reproduction.get("training_authorized") is not False
    ):
        raise M7MirrorCurvatureError("M7 reproduction manifest differs")
    decision = read_json(artifact_paths["decision"])
    diagnostics = read_json(artifact_paths["diagnostics"])
    environment = read_json(artifact_paths["environment"])
    if (
        environment.get("head") != diagnostics.get("runtime_head")
        or reproduction.get("runtime_commit") != diagnostics.get("runtime_head")
        or reproduction.get("environment") != environment
    ):
        raise M7MirrorCurvatureError("M7 environment record differs")
    route = verify_artifacts(protocol, m6_diagnostics, decision, diagnostics)
    with artifact_paths["attribution_csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
    expected_table_rows = diagnostics["attribution"]["table_rows"]
    if (
        tuple(reader.fieldnames or ()) != TABLE_FIELDS
        or len(csv_rows) != EXPECTED_UNITS * len(CERTIFICATES)
        or len(csv_rows) != len(expected_table_rows)
    ):
        raise M7MirrorCurvatureError("M7 attribution CSV denominator differs")
    for csv_row, expected_row in zip(csv_rows, expected_table_rows):
        for name in TABLE_FIELDS:
            expected_value = expected_row[name]
            if type(expected_value) is bool:
                valid = csv_row[name] == str(expected_value)
            elif type(expected_value) in (int, float):
                try:
                    observed_value = float(csv_row[name])
                except ValueError:
                    valid = False
                else:
                    valid = numbers_close(
                        observed_value,
                        expected_value,
                        DECOMPOSITION_ARITHMETIC_TOLERANCE,
                    )
            else:
                valid = csv_row[name] == str(expected_value)
            if not valid:
                raise M7MirrorCurvatureError(
                    f"M7 attribution CSV value differs: {name}"
                )
    result = {
        "status": "PASS_CDRO_OPC_M7_MIRROR_CURVATURE_VERIFY_RUN",
        "route": route,
        "attributed_cells": EXPECTED_UNITS * len(CERTIFICATES),
        "sealed_audit_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main() -> None:
    args = parse_args()
    path = repo_path(args.protocol)
    {"preflight": preflight, "run": run, "verify-run": verify}[args.command](path)


if __name__ == "__main__":
    main()
