"""Prepare the raw-score field-source recovery protocol for RUF-MCD-JRP."""

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
    "4070ti_step3_rufmcd_p0_v3r/protocol/rufmcd_p0_v3r_protocol.json"
)
PREDECESSOR_ROUTE = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r/run/rufmcd_p0_v3r_route.json"
)
PREDECESSOR_INTERPRETATION = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r/analysis/RUFMCD_JRP_P0_V3R_INTERPRETATION.md"
)
PREDECESSOR_ANALYSIS = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r/analysis/rufmcd_p0_v3r_failure_analysis.json"
)
ORIGINAL_V3_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
RECOVERY_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3r2_raw_score_registry_recovery_plan.md"
)
RUNNER = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/step3_rufmcd_jrp_p0.py"
)
TESTS = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "test_step3_rufmcd_jrp_p0.py"
)
PREPARER = Path(__file__).resolve()
OUTPUT = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r2/protocol/rufmcd_p0_v3r2_protocol.json"
)
RECOVERY_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "second_execution_repair_after_protected_facet_raw_score_lookup"
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
    if (
        route["route"] != "STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID"
        or analysis["classification"]
        != "implementation_invalid_before_96_facet_solver_not_method_infeasibility"
        or analysis["geometry"]["affine_solver_started"] is not False
    ):
        raise ValueError("v3r predecessor is not the registered implementation stop")

    protocol.update(
        {
            "name": "step3_rufmcd_jrp_p0_v3r2_raw_score_registry_recovery_protocol_v1",
            "status": RECOVERY_STATUS,
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "experiment_mode": (
                "deterministic_second_execution_recovery_of_frozen_v3_raw_score_"
                "registry_not_fresh_confirmation"
            ),
            "objective": (
                "Execute the unchanged frozen 96-facet JRP problem after correcting "
                "only the original facet raw-score registry lookup."
            ),
            "execution_repair": {
                "incident_code": "V3R_PROTECTED_FACET_RAW_SCORE_LOOKUP",
                "predecessor_protocol_path": relative(PREDECESSOR_PROTOCOL),
                "predecessor_protocol_sha256": sha256(PREDECESSOR_PROTOCOL),
                "predecessor_route_path": relative(PREDECESSOR_ROUTE),
                "predecessor_route_sha256": sha256(PREDECESSOR_ROUTE),
                "predecessor_analysis_path": relative(PREDECESSOR_ANALYSIS),
                "predecessor_analysis_sha256": sha256(PREDECESSOR_ANALYSIS),
                "original_protocol_path": relative(ORIGINAL_V3_PROTOCOL),
                "original_protocol_sha256": sha256(ORIGINAL_V3_PROTOCOL),
                "first_attempt_outputs_written": 0,
                "previous_recovery_outputs_written": 9,
                "code_change": (
                    "retrieve original protected facet raw score from the frozen "
                    "raw_score_panels registry instead of the facet spec"
                ),
                "invalid_source": "spec['raw_score']",
                "registered_source": (
                    "raw_score_panels[spec['unit_index'] + 1][spec['certificate']]"
                ),
                "method_changed": False,
                "thresholds_changed": False,
                "selection_changed": False,
                "fresh_confirmation_claimed": False,
            },
        }
    )

    frozen = dict(protocol["frozen_inputs"])
    frozen.update(
        {
            "method_plan": artifact(RECOVERY_PLAN),
            "runner": artifact(RUNNER),
            "runner_tests": artifact(TESTS),
            "protocol_preparer": artifact(PREPARER),
            "predecessor_v3r_protocol": artifact(PREDECESSOR_PROTOCOL),
            "predecessor_v3r_route": artifact(PREDECESSOR_ROUTE),
            "predecessor_v3r_interpretation": artifact(PREDECESSOR_INTERPRETATION),
            "predecessor_v3r_failure_analysis": artifact(PREDECESSOR_ANALYSIS),
            "original_v3_protocol": artifact(ORIGINAL_V3_PROTOCOL),
        }
    )
    protocol["frozen_inputs"] = frozen

    source = dict(protocol["source"])
    source.update(
        {
            "fresh_for_confirmation": False,
            "role_opening_count_before_v3r2": 2,
            "recovery_result_claimed_as_fresh_confirmation": False,
        }
    )
    protocol["source"] = source

    statistics = dict(protocol["statistics"])
    statistics.update(
        {
            "execution_class": "second_deterministic_recovery_replay",
            "fresh_confirmation": False,
        }
    )
    protocol["statistics"] = statistics

    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_rufmcd_p0_v3r2/run/"
    )
    protocol["outputs"] = {
        "structural_audit": base + "rufmcd_p0_v3r2_structural_audit.json",
        "fit_geometry": base + "rufmcd_p0_v3r2_fit_geometry.json",
        "calibration_rows": base + "rufmcd_p0_v3r2_calibration_rows.jsonl",
        "diagnostic_rows": base + "rufmcd_p0_v3r2_diagnostic_rows.jsonl",
        "diagnostics": base + "rufmcd_p0_v3r2_diagnostics.json",
        "environment": base + "rufmcd_p0_v3r2_environment.json",
        "route": base + "rufmcd_p0_v3r2_route.json",
        "report": base + "RUFMCD_JRP_P0_V3R2_REPORT.md",
        "run_manifest": base + "rufmcd_p0_v3r2_reproduction_manifest.json",
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
