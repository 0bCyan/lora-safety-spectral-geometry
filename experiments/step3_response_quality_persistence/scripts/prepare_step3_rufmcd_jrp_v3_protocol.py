"""Prepare the prospective Step3 RUF-MCD-JRP v3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
V2_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/protocol/rufmcd_p0_v2_protocol.json"
)
V3_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
V3_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3_jackknife_reflected_tarc_polytope_plan.md"
)
V3_CONFIG = ROOT / (
    "experiments/step3_response_quality_persistence/configs/"
    "4070ti_qwen3_4b_step3_rufmcd_p0_v3_seed7.json"
)
V3_RUNNER = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "step3_rufmcd_jrp_p0.py"
)
V3_TESTS = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "test_step3_rufmcd_jrp_p0.py"
)
V3_PREPARER = Path(__file__).resolve()
V2_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v2_analytic_active_set_plan.md"
)
V2_PROTOCOL_PATH = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/protocol/rufmcd_p0_v2_protocol.json"
)
V2_ROUTE = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/run/rufmcd_p0_v2_route.json"
)
V2_INTERPRETATION = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/analysis/RUFMCD_P0_V2_INTERPRETATION.md"
)
V2_ANGLE_AUDIT = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v2/analysis/rufmcd_p0_v2_angle_sensitivity.json"
)


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


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(V3_PROTOCOL))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")

    protocol = read_json(V2_PROTOCOL)
    protocol.update(
        {
            "name": "step3_rufmcd_jrp_fit_only_jackknife_reflected_tarc_polytope_p0_v3_protocol_v1",
            "status": "prospective_before_any_RUFMCD_JRP_model_forward",
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "config_path": relative(V3_CONFIG),
            "objective": (
                "Test whether a fit-only jackknife-reflected finite TARC response "
                "polytope can repair v2's unit-level TARC transfer failure while "
                "preserving the fixed angle, all original fit gates, the conformal "
                "policy, and the diagnostic gates."
            ),
            "experiment_mode": (
                "exploratory_seed7_zero_optimizer_step_RUF-MCD-JRP_v3_on_reused_disclosed_data"
            ),
            "prior_result": {
                "decision_path": relative(V2_ROUTE),
                "decision_sha256": sha256(V2_ROUTE),
                "route": read_json(V2_ROUTE)["route"],
                "registered_signal": (
                    "RUF-MCD v2 had a valid 84-facet analytic certificate and fit "
                    "target gain, but its frozen candidate failed the conformal "
                    "accept-or-raw policy on unseen calibration TARC protection."
                ),
            },
            "predecessor": {
                "route": read_json(V2_ROUTE)["route"],
                "protocol_path": relative(V2_PROTOCOL_PATH),
                "result_path": relative(V2_ROUTE),
            },
            "method_variant": "RUF-MCD-JRP",
            "method_revision": "v3-jackknife-reflected-tarc-polytope",
            "v1_aborted_schema_attempt": (
                "none; v3 protocol is prepared before any v3 model forward"
            ),
            "v1_predecessor": {
                "route": "STOP_RUFMCD_P0_UNIT_FACET_GEOMETRY_INVALID",
                "protocol_path": (
                    "experiments/step3_response_quality_persistence/results/"
                    "4070ti_step3_rufmcd_p0/protocol/rufmcd_p0_protocol.json"
                ),
                "result_path": (
                    "experiments/step3_response_quality_persistence/results/"
                    "4070ti_step3_rufmcd_p0/run/rufmcd_p0_route.json"
                ),
            },
        }
    )

    frozen = dict(protocol["frozen_inputs"])
    frozen.update(
        {
            "method_plan": artifact(V3_PLAN),
            "config": artifact(V3_CONFIG),
            "runner": artifact(V3_RUNNER),
            "runner_tests": artifact(V3_TESTS),
            "protocol_preparer": artifact(V3_PREPARER),
            "predecessor_rufmcd_v2_plan": artifact(V2_PLAN),
            "predecessor_rufmcd_v2_protocol": artifact(V2_PROTOCOL_PATH),
            "predecessor_rufmcd_v2_route": artifact(V2_ROUTE),
            "predecessor_rufmcd_v2_interpretation": artifact(V2_INTERPRETATION),
            "predecessor_rufmcd_v2_angle_audit": artifact(V2_ANGLE_AUDIT),
        }
    )
    protocol["frozen_inputs"] = frozen

    source = dict(protocol["source"])
    source.update(
        {
            "role": (
                "seed7_exploratory_fit_calibration_diagnostic_fit_only_"
                "jackknife_reflected_tarc_polytope_v3"
            ),
            "fresh_for_confirmation": False,
            "v2_calibration_used_as_selection_data": False,
            "v2_angle_audit_used_as_selection_data": False,
        }
    )
    protocol["source"] = source

    design = dict(protocol["design"])
    design.update(
        {
            "target_proposal": (
                "unit_facet_joint_protected_target_margin_affine_max_min_with_"
                "jackknife_reflected_tarc_polytope"
            ),
            "protected_constraint_rule": (
                "all_twelve_fit_units_cross_four_protected_certificate_gradients"
            ),
            "projection_rule": (
                "shared_float64_gram_pseudoinverse_projection_of_96_registered_"
                "covectors_off_raw_displacement_only"
            ),
            "target_covectors": 96,
            "original_facet_count": 84,
            "reflected_tarc_vertex_count": 12,
            "robust_protected_facet_count": 12,
            "total_affine_facet_count": 96,
            "fit_target_covector_rule": (
                "one_normalized_covector_per_fit_unit_and_certificate_plus_"
                "tarc_response_covectors_c_i_minus_from_fit_only_jackknife_reflections"
            ),
            "selection_rule": (
                "fixed_theta_max_minus_1e-6_then_joint_96_facet_protected_target_"
                "margin_affine_max_min_with_analytic_full_gap_certificate"
            ),
            "candidate_map_variant": (
                "unit_facet_joint_protected_target_margin_affine_max_min_"
                "jackknife_reflected_tarc_polytope_tangent"
            ),
            "unit_facet_margin_certificate_count": 96,
            "jackknife_response_covector_rule": (
                "c_i=normalized_tarc_covector/raw_tarc_score_i; "
                "c_i_minus=2*mean(c_j,j!=i)-c_i"
            ),
            "robust_protected_certificate": "tarc",
            "population_coverage_claimed": False,
            "calibration_scores_used_for_selection": False,
            "diagnostic_scores_used_for_selection": False,
        }
    )
    protocol["design"] = design

    statistics = dict(protocol["statistics"])
    statistics.update(
        {
            "method": "RUF-MCD-JRP-v3-fit-only-jackknife-reflected-tarc-polytope",
            "candidate_selection": design["selection_rule"],
            "finite_uncertainty_claim": (
                "robust only for the registered convex hull of observed and reflected "
                "fit TARC response covectors; no population coverage claim"
            ),
        }
    )
    protocol["statistics"] = statistics

    protocol["outputs"] = {
        "structural_audit": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_structural_audit.json"
        ),
        "fit_geometry": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_fit_geometry.json"
        ),
        "calibration_rows": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_calibration_rows.jsonl"
        ),
        "diagnostic_rows": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_diagnostic_rows.jsonl"
        ),
        "diagnostics": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_diagnostics.json"
        ),
        "environment": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_environment.json"
        ),
        "route": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_route.json"
        ),
        "report": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/RUFMCD_JRP_P0_V3_REPORT.md"
        ),
        "run_manifest": (
            "experiments/step3_response_quality_persistence/results/"
            "4070ti_step3_rufmcd_p0_v3/run/rufmcd_p0_v3_reproduction_manifest.json"
        ),
    }
    protocol["pass_route"] = "PASS_RUFMCD_JRP_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL"
    protocol["failure_routes"] = [
        "STOP_RUFMCD_JRP_P0_INVALID",
        "STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID",
        "STOP_RUFMCD_JRP_P0_NO_ROBUST_EFFECTIVE_ANGLE",
        "STOP_RUFMCD_JRP_P0_CONFORMAL_SELECTIVE_POLICY_FAILED",
        "STOP_RUFMCD_JRP_P0_DIAGNOSTIC_TRANSFER_FAILED",
        "STOP_RUFMCD_JRP_P0_FINITE_STEP_COMMON_DESCENT_FAILED",
        "STOP_RUFMCD_JRP_P0_PROTECTED_NONINFERIORITY_FAILED",
        "STOP_RUFMCD_JRP_P0_TARGET_ADVANTAGE_FAILED",
        "STOP_RUFMCD_JRP_P0_EFFECT_TOO_WEAK",
    ]
    protocol["pass_authorizes"] = (
        "freeze_and_push_a_separate_training_disjoint_proxy_dev_triplet_data_protocol_only"
    )
    protocol["pass_does_not_authorize"] = "model_training_or_optimizer_steps"
    protocol["optimizer_objects"] = 0
    protocol["scheduler_objects"] = 0
    protocol["optimizer_steps"] = 0
    protocol["virtual_parameter_evaluations"] = 24
    protocol["checkpoint_writes_authorized"] = False
    protocol["candidate_generation_authorized"] = False
    protocol["training_authorized"] = False
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
