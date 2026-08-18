"""Prepare the process-loss recovery protocol for RUF-MCD-JRP."""

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
    "4070ti_step3_rufmcd_p0_v3r3/protocol/rufmcd_p0_v3r3_protocol.json"
)
PREDECESSOR_INCIDENT = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3r3/incident/"
    "rufmcd_p0_v3r3_process_loss_incident.json"
)
ORIGINAL_V3_PROTOCOL = ROOT / (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_rufmcd_p0_v3/protocol/rufmcd_p0_v3_protocol.json"
)
RECOVERY_PLAN = ROOT / (
    "experiments/step3_response_quality_persistence/"
    "Step3_RUFMCD_v3r4_process_loss_recovery_plan.md"
)
CONFIG = ROOT / (
    "experiments/step3_response_quality_persistence/configs/"
    "4070ti_qwen3_4b_step3_rufmcd_p0_v3r3_seed7.json"
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
    "4070ti_step3_rufmcd_p0_v3r4/protocol/rufmcd_p0_v3r4_protocol.json"
)
PREDECESSOR_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "third_execution_repair_for_24_facet_tarc_group_round_bound"
)
RECOVERY_STATUS = (
    "method_frozen_before_first_RUFMCD_JRP_model_forward_"
    "fourth_execution_recovery_after_runtime_process_loss"
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
    incident = read_json(PREDECESSOR_INCIDENT)
    config = read_json(CONFIG)
    predecessor_outputs = [ROOT / path for path in protocol["outputs"].values()]
    if (
        protocol["status"] != PREDECESSOR_STATUS
        or protocol["design"]["total_affine_facet_count"] != 96
        or protocol["design"]["constraint_generation_max_rounds"] != 24
        or config["rufmcd_constraint_generation_max_rounds"] != 24
        or incident["incident_code"]
        != "V3R3_RUNTIME_PROCESS_LOSS_BEFORE_ARTIFACT_WRITE"
        or incident["observed_state"]["outputs_written"] != 0
        or incident["optimizer_steps"] != 0
        or any(path.exists() for path in predecessor_outputs)
    ):
        raise ValueError("v3r3 process-loss predecessor registry differs")

    protocol.update(
        {
            "name": "step3_rufmcd_jrp_p0_v3r4_process_loss_recovery_protocol_v1",
            "status": RECOVERY_STATUS,
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "config_path": relative(CONFIG),
            "experiment_mode": (
                "deterministic_fourth_execution_recovery_of_frozen_v3_after_"
                "runtime_process_loss_not_fresh_confirmation"
            ),
            "objective": (
                "Re-execute the unchanged frozen 96-facet, 24-round JRP "
                "certificate after a zero-artifact runtime process loss."
            ),
            "execution_repair": {
                "incident_code": "V3R3_RUNTIME_PROCESS_LOSS_BEFORE_ARTIFACT_WRITE",
                "predecessor_protocol_path": relative(PREDECESSOR_PROTOCOL),
                "predecessor_protocol_sha256": sha256(PREDECESSOR_PROTOCOL),
                "predecessor_incident_path": relative(PREDECESSOR_INCIDENT),
                "predecessor_incident_sha256": sha256(PREDECESSOR_INCIDENT),
                "original_protocol_path": relative(ORIGINAL_V3_PROTOCOL),
                "original_protocol_sha256": sha256(ORIGINAL_V3_PROTOCOL),
                "first_attempt_outputs_written": 0,
                "interrupted_v3r3_outputs_written": 0,
                "previous_recovery_outputs_written": 18,
                "code_change": (
                    "admit only the registered provenance status and apply the "
                    "existing complete-round validation; numerical solver unchanged"
                ),
                "runtime_launch_policy": (
                    "one process with repository-external stdout/stderr; no restart "
                    "while alive"
                ),
                "runtime_launch_policy_changed": True,
                "certificate_iteration_budget_changed": False,
                "certificate_thresholds_changed": False,
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
            "config": artifact(CONFIG),
            "runner": artifact(RUNNER),
            "runner_tests": artifact(TESTS),
            "protocol_preparer": artifact(PREPARER),
            "predecessor_v3r3_protocol": artifact(PREDECESSOR_PROTOCOL),
            "predecessor_v3r3_process_loss_incident": artifact(
                PREDECESSOR_INCIDENT
            ),
            "original_v3_protocol": artifact(ORIGINAL_V3_PROTOCOL),
        }
    )
    protocol["frozen_inputs"] = frozen

    source = dict(protocol["source"])
    source.update(
        {
            "fresh_for_confirmation": False,
            "fit_role_opening_count_before_v3r4": 4,
            "calibration_role_opening_count_before_v3r4": 3,
            "diagnostic_role_opening_count_before_v3r4": 3,
            "recovery_result_claimed_as_fresh_confirmation": False,
        }
    )
    protocol["source"] = source

    statistics = dict(protocol["statistics"])
    statistics.update(
        {
            "execution_class": "fourth_deterministic_recovery_replay",
            "fresh_confirmation": False,
        }
    )
    protocol["statistics"] = statistics

    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_rufmcd_p0_v3r4/run/"
    )
    protocol["outputs"] = {
        "structural_audit": base + "rufmcd_p0_v3r4_structural_audit.json",
        "fit_geometry": base + "rufmcd_p0_v3r4_fit_geometry.json",
        "calibration_rows": base + "rufmcd_p0_v3r4_calibration_rows.jsonl",
        "diagnostic_rows": base + "rufmcd_p0_v3r4_diagnostic_rows.jsonl",
        "diagnostics": base + "rufmcd_p0_v3r4_diagnostics.json",
        "environment": base + "rufmcd_p0_v3r4_environment.json",
        "route": base + "rufmcd_p0_v3r4_route.json",
        "report": base + "RUFMCD_JRP_P0_V3R4_REPORT.md",
        "run_manifest": base + "rufmcd_p0_v3r4_reproduction_manifest.json",
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
