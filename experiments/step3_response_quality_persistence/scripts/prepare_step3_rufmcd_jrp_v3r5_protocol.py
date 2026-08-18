"""Prepare the global-round-bound recovery protocol for RUF-MCD-JRP."""

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
    "4070ti_step3_rufmcd_p0_v3r4/protocol/rufmcd_p0_v3r4_protocol.json"
)
PREDECESSOR_ROUTE = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r4/run/rufmcd_p0_v3r4_route.json"
)
PREDECESSOR_INTERPRETATION = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r4/analysis/"
    "RUFMCD_JRP_P0_V3R4_INTERPRETATION.md"
)
PREDECESSOR_ANALYSIS = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r4/analysis/"
    "rufmcd_p0_v3r4_round_bound_analysis.json"
)
ORIGINAL_V3_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
RECOVERY_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3r5_global_round_bound_recovery_plan.md"
)
CONFIG = ROOT / (
    "experiments/step3_response_quality_persistence/configs/"
    "4070ti_qwen3_4b_step3_rufmcd_p0_v3r5_seed7.json"
)
RUNNER = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/step3_rufmcd_jrp_p0.py"
)
RUNNER_TESTS = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "test_step3_rufmcd_jrp_p0.py"
)
CONSTRAINT_GENERATION_TESTS = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "test_step3_rufmcd_p0.py"
)
PREPARER = Path(__file__).resolve()
OUTPUT = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r5/protocol/rufmcd_p0_v3r5_protocol.json"
)
PREDECESSOR_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "fourth_execution_recovery_after_runtime_process_loss"
)
RECOVERY_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "fifth_execution_repair_for_global_constraint_generation_bound"
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
        != "protocol_incomplete_global_constraint_generation_bound_not_method_infeasibility"
        or analysis["geometry"]["frozen_constraint_generation_max_rounds"] != 24
        or analysis["global_completeness_argument"]["complete_round_bound"] != 90
        or config["rufmcd_constraint_generation_max_rounds"] != 90
    ):
        raise ValueError("v3r4 predecessor or v3r5 global bound registry differs")

    protocol.update(
        {
            "name": "step3_rufmcd_jrp_p0_v3r5_global_round_bound_recovery_protocol_v1",
            "status": RECOVERY_STATUS,
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "config_path": relative(CONFIG),
            "experiment_mode": (
                "deterministic_fifth_execution_recovery_of_frozen_v3_with_"
                "global_constraint_generation_bound_not_fresh_confirmation"
            ),
            "objective": (
                "Certify the unchanged frozen 96-facet JRP problem using the "
                "implementation-level complete N-G+1 round bound."
            ),
            "execution_repair": {
                "incident_code": (
                    "V3R4_INCOMPLETE_GLOBAL_CONSTRAINT_GENERATION_ROUND_BOUND"
                ),
                "predecessor_protocol_path": relative(PREDECESSOR_PROTOCOL),
                "predecessor_protocol_sha256": sha256(PREDECESSOR_PROTOCOL),
                "predecessor_route_path": relative(PREDECESSOR_ROUTE),
                "predecessor_route_sha256": sha256(PREDECESSOR_ROUTE),
                "predecessor_analysis_path": relative(PREDECESSOR_ANALYSIS),
                "predecessor_analysis_sha256": sha256(PREDECESSOR_ANALYSIS),
                "original_protocol_path": relative(ORIGINAL_V3_PROTOCOL),
                "original_protocol_sha256": sha256(ORIGINAL_V3_PROTOCOL),
                "first_attempt_outputs_written": 0,
                "previous_recovery_outputs_written": 27,
                "code_change": (
                    "validate and use the N-G+1 global constraint-generation "
                    "completeness bound; solver and addition rule unchanged"
                ),
                "old_constraint_generation_max_rounds": 24,
                "new_constraint_generation_max_rounds": 90,
                "total_facets": 96,
                "initial_facet_groups": 7,
                "complete_bound_formula": "N-G+1",
                "certificate_iteration_budget_changed": True,
                "certificate_thresholds_changed": False,
                "method_changed": False,
                "thresholds_changed": False,
                "selection_changed": False,
                "fresh_confirmation_claimed": False,
            },
        }
    )

    design = dict(protocol["design"])
    design["constraint_generation_max_rounds"] = 90
    design["constraint_generation_complete_round_bound"] = 90
    design["constraint_generation_complete_bound_formula"] = "N-G+1"
    design["constraint_generation_initial_group_count"] = 7
    protocol["design"] = design

    frozen = dict(protocol["frozen_inputs"])
    frozen.update(
        {
            "method_plan": artifact(RECOVERY_PLAN),
            "config": artifact(CONFIG),
            "runner": artifact(RUNNER),
            "runner_tests": artifact(RUNNER_TESTS),
            "constraint_generation_tests": artifact(
                CONSTRAINT_GENERATION_TESTS
            ),
            "protocol_preparer": artifact(PREPARER),
            "predecessor_v3r4_protocol": artifact(PREDECESSOR_PROTOCOL),
            "predecessor_v3r4_route": artifact(PREDECESSOR_ROUTE),
            "predecessor_v3r4_interpretation": artifact(
                PREDECESSOR_INTERPRETATION
            ),
            "predecessor_v3r4_round_bound_analysis": artifact(
                PREDECESSOR_ANALYSIS
            ),
            "original_v3_protocol": artifact(ORIGINAL_V3_PROTOCOL),
        }
    )
    protocol["frozen_inputs"] = frozen

    source = dict(protocol["source"])
    source.update(
        {
            "fresh_for_confirmation": False,
            "fit_role_opening_count_before_v3r5": 5,
            "calibration_role_opening_count_before_v3r5": 4,
            "diagnostic_role_opening_count_before_v3r5": 4,
            "recovery_result_claimed_as_fresh_confirmation": False,
        }
    )
    protocol["source"] = source

    statistics = dict(protocol["statistics"])
    statistics.update(
        {
            "execution_class": "fifth_deterministic_recovery_replay",
            "fresh_confirmation": False,
        }
    )
    protocol["statistics"] = statistics

    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_rufmcd_p0_v3r5/run/"
    )
    protocol["outputs"] = {
        "structural_audit": base + "rufmcd_p0_v3r5_structural_audit.json",
        "fit_geometry": base + "rufmcd_p0_v3r5_fit_geometry.json",
        "calibration_rows": base + "rufmcd_p0_v3r5_calibration_rows.jsonl",
        "diagnostic_rows": base + "rufmcd_p0_v3r5_diagnostic_rows.jsonl",
        "diagnostics": base + "rufmcd_p0_v3r5_diagnostics.json",
        "environment": base + "rufmcd_p0_v3r5_environment.json",
        "route": base + "rufmcd_p0_v3r5_route.json",
        "report": base + "RUFMCD_JRP_P0_V3R5_REPORT.md",
        "run_manifest": base + "rufmcd_p0_v3r5_reproduction_manifest.json",
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
