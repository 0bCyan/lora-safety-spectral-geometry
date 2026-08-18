"""Prepare the realized-tensor certificate recovery for RUF-MCD-JRP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PREDECESSOR_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r5/protocol/rufmcd_p0_v3r5_protocol.json"
)
PREDECESSOR_ROUTE = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r5/run/rufmcd_p0_v3r5_route.json"
)
PREDECESSOR_ANALYSIS = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r5/analysis/"
    "rufmcd_p0_v3r5_realized_closure_analysis.json"
)
PREDECESSOR_INTERPRETATION = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r5/analysis/"
    "RUFMCD_JRP_P0_V3R5_INTERPRETATION.md"
)
ORIGINAL_V3_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3r6_realized_tensor_certificate_plan.md"
)
CONFIG = ROOT / (
    "experiments/step3_response_quality_persistence/configs/"
    "4070ti_qwen3_4b_step3_rufmcd_p0_v3r6_seed7.json"
)
RUNNER = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "step3_rufmcd_jrp_p0.py"
)
RUNNER_TESTS = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "test_step3_rufmcd_jrp_p0.py"
)
BASE_SOLVER = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/step3_rufmcd_p0.py"
)
PREPARER = Path(__file__).resolve()
OUTPUT = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r6/protocol/rufmcd_p0_v3r6_protocol.json"
)
PREDECESSOR_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "fifth_execution_repair_for_global_constraint_generation_bound"
)
RECOVERY_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "sixth_execution_repair_for_realized_tensor_certificate"
)
SOLVER_VARIANT = "direct_full_set_continuous_kkt_plus_realized_margin_bounds"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative(path), "sha256": sha256(path)}


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")

    protocol = read_json(PREDECESSOR_PROTOCOL)
    route = read_json(PREDECESSOR_ROUTE)
    analysis = read_json(PREDECESSOR_ANALYSIS)
    config = read_json(CONFIG)
    if (
        protocol["status"] != PREDECESSOR_STATUS
        or route["route"]
        != "STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID"
        or analysis["classification"]
        != "complete_active_set_realized_tensor_numeric_closure_failure_not_polytope_infeasibility"
        or analysis["geometry"]["projection_error"]
        != "RUF-MCD full active set did not close the dual gap"
        or analysis["geometry"]["all_facets_active_at_stop"] is not True
        or config["rufmcd_affine_solver_variant"] != SOLVER_VARIANT
    ):
        raise ValueError("v3r5 predecessor or v3r6 certificate registry differs")

    protocol.update(
        {
            "name": "step3_rufmcd_jrp_p0_v3r6_realized_tensor_certificate_protocol_v1",
            "status": RECOVERY_STATUS,
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "config_path": relative(CONFIG),
            "experiment_mode": (
                "deterministic_sixth_execution_recovery_of_frozen_v3_with_"
                "direct_continuous_and_realized_tensor_certificates_not_fresh_confirmation"
            ),
            "objective": (
                "Certify the unchanged 96-facet continuous optimum and the "
                "parameter-dtype realized tensor margins as separate claims."
            ),
            "execution_repair": {
                "incident_code": "V3R5_REALIZED_TENSOR_FULL_GAP_CLOSURE",
                "predecessor_protocol_path": relative(PREDECESSOR_PROTOCOL),
                "predecessor_protocol_sha256": sha256(PREDECESSOR_PROTOCOL),
                "predecessor_route_path": relative(PREDECESSOR_ROUTE),
                "predecessor_route_sha256": sha256(PREDECESSOR_ROUTE),
                "predecessor_analysis_path": relative(PREDECESSOR_ANALYSIS),
                "predecessor_analysis_sha256": sha256(PREDECESSOR_ANALYSIS),
                "original_protocol_path": relative(ORIGINAL_V3_PROTOCOL),
                "original_protocol_sha256": sha256(ORIGINAL_V3_PROTOCOL),
                "first_attempt_outputs_written": 0,
                "predecessor_outputs_written": 9,
                "code_change": (
                    "replace repeated constraint-generation prefix solves with "
                    "one equivalent complete-facet solve; retain continuous 1e-7 "
                    "KKT/gap checks and add parameter-dtype per-facet forward-error bounds"
                ),
                "old_affine_solver": (
                    "analytic_active_set_kkt_refinement_from_slsqp_support_seed_"
                    "with_constraint_generation"
                ),
                "new_affine_solver": SOLVER_VARIANT,
                "total_facets": 96,
                "direct_full_set_solve_count": 1,
                "continuous_certificate_thresholds_changed": False,
                "certificate_implementation_changed": True,
                "method_changed": False,
                "thresholds_changed": False,
                "selection_changed": False,
                "fresh_confirmation_claimed": False,
            },
        }
    )

    design = dict(protocol["design"])
    design.update(
        {
            "affine_solver": SOLVER_VARIANT,
            "continuous_affine_solver": (
                "analytic_active_set_kkt_refinement_from_slsqp_support_seed"
            ),
            "complete_facet_solver": "direct_single_solve",
            "direct_full_set_solve_count": 1,
            "continuous_kkt_residual_max": 1e-7,
            "continuous_primal_dual_gap_max": 1e-7,
            "realized_direction_rule": (
                "parameter_dtype_weighted_tensor_sum_then_realized_norm_rescale"
            ),
            "realized_direction_norm_relative_max": 1e-7,
            "realized_score_rule": (
                "all_96_unprojected_registered_parameter_dtype_scores_with_"
                "binary64_forward_error_lower_bounds"
            ),
            "realized_margin_lower_bound_min": 0.0,
            "dot_roundoff_safety_factor": 2.0,
            "constraint_generation_used": False,
            "calibration_scores_used_for_selection": False,
            "diagnostic_scores_used_for_selection": False,
        }
    )
    protocol["design"] = design

    frozen = dict(protocol["frozen_inputs"])
    frozen.update(
        {
            "method_plan": artifact(PLAN),
            "config": artifact(CONFIG),
            "runner": artifact(RUNNER),
            "runner_tests": artifact(RUNNER_TESTS),
            "base_rufmcd_solver": artifact(BASE_SOLVER),
            "protocol_preparer": artifact(PREPARER),
            "predecessor_v3r5_protocol": artifact(PREDECESSOR_PROTOCOL),
            "predecessor_v3r5_route": artifact(PREDECESSOR_ROUTE),
            "predecessor_v3r5_analysis": artifact(PREDECESSOR_ANALYSIS),
            "predecessor_v3r5_interpretation": artifact(
                PREDECESSOR_INTERPRETATION
            ),
            "original_v3_protocol": artifact(ORIGINAL_V3_PROTOCOL),
        }
    )
    protocol["frozen_inputs"] = frozen

    source = dict(protocol["source"])
    source.update(
        {
            "fresh_for_confirmation": False,
            "fit_role_opening_count_before_v3r6": 6,
            "calibration_role_opening_count_before_v3r6": 5,
            "diagnostic_role_opening_count_before_v3r6": 5,
            "recovery_result_claimed_as_fresh_confirmation": False,
        }
    )
    protocol["source"] = source

    statistics = dict(protocol["statistics"])
    statistics.update(
        {
            "execution_class": "sixth_deterministic_recovery_replay",
            "fresh_confirmation": False,
        }
    )
    protocol["statistics"] = statistics

    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_rufmcd_p0_v3r6/run/"
    )
    protocol["outputs"] = {
        "structural_audit": base + "rufmcd_p0_v3r6_structural_audit.json",
        "fit_geometry": base + "rufmcd_p0_v3r6_fit_geometry.json",
        "calibration_rows": base + "rufmcd_p0_v3r6_calibration_rows.jsonl",
        "diagnostic_rows": base + "rufmcd_p0_v3r6_diagnostic_rows.jsonl",
        "diagnostics": base + "rufmcd_p0_v3r6_diagnostics.json",
        "environment": base + "rufmcd_p0_v3r6_environment.json",
        "route": base + "rufmcd_p0_v3r6_route.json",
        "report": base + "RUFMCD_JRP_P0_V3R6_REPORT.md",
        "run_manifest": base + "rufmcd_p0_v3r6_reproduction_manifest.json",
    }
    protocol["pass_authorizes"] = (
        "mechanism_interpretation_only_pending_fresh_disjoint_confirmation"
    )
    protocol["pass_does_not_authorize"] = (
        "model_training_optimizer_steps_or_confirmatory_claims"
    )
    protocol["training_authorized"] = False
    protocol["optimizer_steps"] = 0
    protocol["core100_opened"] = False
    protocol["heldout_opened"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"protocol": relative(output), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
