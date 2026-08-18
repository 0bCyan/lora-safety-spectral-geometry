"""Prepare the code-only recovery protocol for Step3 RUF-MCD-JRP v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
RECOVERY_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r/protocol/rufmcd_p0_v3r_protocol.json"
)
RECOVERY_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3r_execution_recovery_plan.md"
)
ORIGINAL_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3_jackknife_reflected_tarc_polytope_plan.md"
)
RUNNER = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/step3_rufmcd_jrp_p0.py"
)
TESTS = ROOT / (
    "experiments/step3_response_quality_persistence/scripts/"
    "test_step3_rufmcd_jrp_p0.py"
)
PREPARER = Path(__file__).resolve()
INCIDENT = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/incident/rufmcd_p0_v3_execution_incident.json"
)
RECOVERY_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "execution_repair_after_post_measurement_routing_recursion"
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
    parser.add_argument("--output", default=str(RECOVERY_PROTOCOL))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")

    protocol = read_json(ORIGINAL_PROTOCOL)
    incident = read_json(INCIDENT)
    if (
        sha256(ORIGINAL_PROTOCOL) != incident["protocol_sha256"]
        or incident["outputs_written"] != 0
        or incident["optimizer_steps"] != 0
    ):
        raise ValueError("original protocol and incident registry differ")

    protocol.update(
        {
            "name": "step3_rufmcd_jrp_p0_v3r_code_only_execution_recovery_protocol_v1",
            "status": RECOVERY_STATUS,
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "experiment_mode": (
                "deterministic_execution_recovery_of_frozen_v3_after_post_measurement_"
                "routing_recursion_not_fresh_confirmation"
            ),
            "objective": (
                "Recover route computation and artifact freezing for the method "
                "already frozen in v3, changing only the gate-dispatch reference."
            ),
            "execution_repair": {
                "incident_code": incident["incident_code"],
                "incident_path": relative(INCIDENT),
                "incident_sha256": sha256(INCIDENT),
                "original_freeze_commit": incident["freeze_commit"],
                "original_protocol_path": relative(ORIGINAL_PROTOCOL),
                "original_protocol_sha256": sha256(ORIGINAL_PROTOCOL),
                "first_attempt_outputs_written": 0,
                "first_attempt_roles_opened": ["fit", "calibration", "diagnostic"],
                "code_change": (
                    "capture base.evaluate_gates before monkey patch and call the "
                    "captured reference"
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
            "original_v3_method_plan": artifact(ORIGINAL_PLAN),
            "runner": artifact(RUNNER),
            "runner_tests": artifact(TESTS),
            "protocol_preparer": artifact(PREPARER),
            "original_v3_protocol": artifact(ORIGINAL_PROTOCOL),
            "original_v3_incident": artifact(INCIDENT),
        }
    )
    protocol["frozen_inputs"] = frozen

    source = dict(protocol["source"])
    source.update(
        {
            "fresh_for_confirmation": False,
            "roles_previously_opened_by_failed_v3_process": [
                "fit",
                "calibration",
                "diagnostic",
            ],
            "recovery_result_claimed_as_fresh_confirmation": False,
        }
    )
    protocol["source"] = source

    statistics = dict(protocol["statistics"])
    statistics.update(
        {
            "execution_class": "deterministic_recovery_replay",
            "fresh_confirmation": False,
        }
    )
    protocol["statistics"] = statistics

    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_rufmcd_p0_v3r/run/"
    )
    protocol["outputs"] = {
        "structural_audit": base + "rufmcd_p0_v3r_structural_audit.json",
        "fit_geometry": base + "rufmcd_p0_v3r_fit_geometry.json",
        "calibration_rows": base + "rufmcd_p0_v3r_calibration_rows.jsonl",
        "diagnostic_rows": base + "rufmcd_p0_v3r_diagnostic_rows.jsonl",
        "diagnostics": base + "rufmcd_p0_v3r_diagnostics.json",
        "environment": base + "rufmcd_p0_v3r_environment.json",
        "route": base + "rufmcd_p0_v3r_route.json",
        "report": base + "RUFMCD_JRP_P0_V3R_REPORT.md",
        "run_manifest": base + "rufmcd_p0_v3r_reproduction_manifest.json",
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
