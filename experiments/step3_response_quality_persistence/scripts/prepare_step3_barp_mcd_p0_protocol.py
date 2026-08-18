"""Prepare the prospective BARP-MCD same-batch zero-step protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PREDECESSOR_PROTOCOL = (
    BASE / "results/4070ti_step3_rpfrsc_p0/protocol/rpfrsc_p0_protocol.json"
)
PREDECESSOR_ROUTE = BASE / "results/4070ti_step3_rpfrsc_p0/run/rpfrsc_p0_route.json"
PREDECESSOR_INTERPRETATION = (
    BASE / "results/4070ti_step3_rpfrsc_p0/analysis/RPFRSC_P0_INTERPRETATION.md"
)
V3R6_ANALYSIS = (
    BASE
    / "results/4070ti_step3_rufmcd_p0_v3r6/analysis/rufmcd_p0_v3r6_realized_certificate_analysis.json"
)
V3R6_ANGLE_AUDIT = (
    BASE
    / "results/4070ti_step3_rufmcd_p0_v3r6/analysis/rufmcd_p0_v3r6_relaxed_angle_impossibility.json"
)
V3R6_ANGLE_INTERPRETATION = (
    BASE
    / "results/4070ti_step3_rufmcd_p0_v3r6/analysis/RUFMCD_JRP_P0_V3R6_ANGLE_IMPOSSIBILITY.md"
)
PLAN = BASE / "Step3_BARP_MCD_batch_adaptive_route_protected_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_p0_seed4.json"
RUNNER = BASE / "scripts/step3_barp_mcd_p0.py"
RUNNER_TESTS = BASE / "scripts/test_step3_barp_mcd_p0.py"
PREPARER = Path(__file__).resolve()
OUTPUT = BASE / "results/4070ti_step3_barp_mcd_p0/protocol/barp_mcd_p0_protocol.json"


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

    predecessor = read_json(PREDECESSOR_PROTOCOL)
    route = read_json(PREDECESSOR_ROUTE)
    angle = read_json(V3R6_ANGLE_AUDIT)
    if (
        route["route"] != "STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED"
        or angle["relaxed_fit_proof"]["strict_interval_separation"] is not True
        or angle["interpretation"]["direction_change_required"] is not True
    ):
        raise ValueError("BARP-MCD predecessor registry differs")

    protocol = dict(predecessor)
    protocol.update(
        {
            "name": "step3_barp_mcd_batch_adaptive_route_protected_p0_protocol_v1",
            "status": "prospective_before_any_BARP_MCD_model_forward",
            "protocol_path": relative(output),
            "entry_commit": git_head(),
            "config_path": relative(CONFIG),
            "objective": (
                "Test whether recomputing the route-protected finite-family "
                "direction on each optimizer batch preserves four protected "
                "finite-step objectives while retaining non-weak three-target gain."
            ),
            "experiment_mode": (
                "prospective_algorithm_execution_on_disclosed_seed4_units_not_"
                "fresh_confirmation_zero_optimizer_step"
            ),
            "prior_result": {
                "decision_path": relative(PREDECESSOR_ROUTE),
                "decision_sha256": sha256(PREDECESSOR_ROUTE),
                "route": route["route"],
                "registered_signal": (
                    "RP-PFRSC passed target, common-descent, and effect gates but "
                    "missed the finite-step TARC bootstrap lower bound by 0.137 "
                    "percentage points under cross-batch transfer."
                ),
            },
            "predecessor": {
                "protocol_path": relative(PREDECESSOR_PROTOCOL),
                "protocol_sha256": sha256(PREDECESSOR_PROTOCOL),
                "result_path": relative(PREDECESSOR_ROUTE),
                "result_sha256": sha256(PREDECESSOR_ROUTE),
            },
            "method_basis": {
                "v3r6_analysis_path": relative(V3R6_ANALYSIS),
                "v3r6_analysis_sha256": sha256(V3R6_ANALYSIS),
                "angle_audit_path": relative(V3R6_ANGLE_AUDIT),
                "angle_audit_sha256": sha256(V3R6_ANGLE_AUDIT),
                "fixed_global_direction_closed": True,
                "batch_adaptation_is_method_change": True,
                "thresholds_changed": False,
                "finite_family_changed": False,
            },
        }
    )

    source = dict(predecessor["source"])
    source.update(
        {
            "role": "seed4_disclosed_batch_adaptive_same_batch_mechanism_p0",
            "fresh_for_confirmation": False,
            "independent_transfer_claimed": False,
            "excluded_source_block_gradients_opened": False,
            "core100_used": False,
            "heldout_used": False,
        }
    )
    protocol["source"] = source

    prior_selection = predecessor["selection"]
    protocol["selection"] = {
        "risk_order_seed": 4,
        "optimizer_block_rule": prior_selection["safety_training_order"],
        "optimizer_blocks": prior_selection["training_blocks"],
        "excluded_source_block": prior_selection["diagnostic_block"],
        "optimizer_units": 8,
        "risk_rows_per_unit": 4,
        "safety_pairs_per_unit": 4,
        "selection_scope": "same_optimizer_batch",
        "evaluation_scope": "same_optimizer_batch",
        "optimizer_excluded_overlap": 0,
    }

    design = dict(predecessor["design"])
    design.update(
        {
            "method": "BARP-MCD",
            "gradient_measurement_mode": "eval",
            "selection_scope": "same_optimizer_batch",
            "finite_step_evaluation_scope": "same_optimizer_batch",
            "independent_transfer_claimed": False,
            "candidate_recomputed_per_training_batch": True,
            "raw_anchor_in_every_finite_family": True,
            "same_batch_protection_theorem": (
                "finite_family_raw_anchor_nonemptiness_plus_coordinatewise_"
                "95_percent_protected_exact_score_retention"
            ),
        }
    )
    protocol["design"] = design

    statistics = dict(predecessor["statistics"])
    statistics.update(
        {
            "method": "BARP-MCD-seed4-same-batch-P0",
            "evidence_scope": "same_batch_optimizer_mechanism_not_transfer",
            "fresh_confirmation": False,
        }
    )
    protocol["statistics"] = statistics

    dependencies = {
        "method_plan": PLAN,
        "config": CONFIG,
        "runner": RUNNER,
        "runner_tests": RUNNER_TESTS,
        "protocol_preparer": PREPARER,
        "rpfrsc_runner": BASE / "scripts/step3_rpfrsc_p0.py",
        "rpfrsc_runner_tests": BASE / "scripts/test_step3_rpfrsc_p0.py",
        "pfrsc_common": BASE / "scripts/step3_pfrsc_common.py",
        "pfrsc_common_tests": BASE / "scripts/test_step3_pfrsc_common.py",
        "frsc_common": BASE / "scripts/step3_frsc_common.py",
        "ramp_common": BASE / "scripts/step3_ramp_common.py",
        "model_state_helper": BASE / "scripts/step3_cita_p0.py",
        "objective_helpers": BASE / "scripts/step3_mcd_training.py",
        "risk_loss_helpers": BASE / "scripts/step3_ramp_training.py",
        "training_common": BASE / "scripts/step3ct_training_common.py",
        "model_reference": BASE / "scripts/train_step3ct_tournament.py",
        "approved_targets": ROOT
        / "experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1.jsonl",
        "risk_source": ROOT / "data/processed/unified_pool/train_risk.jsonl",
        "structural_audit": ROOT / source["structural_audit_path"],
        "predecessor_protocol": PREDECESSOR_PROTOCOL,
        "predecessor_route": PREDECESSOR_ROUTE,
        "predecessor_interpretation": PREDECESSOR_INTERPRETATION,
        "v3r6_analysis": V3R6_ANALYSIS,
        "v3r6_angle_audit": V3R6_ANGLE_AUDIT,
        "v3r6_angle_interpretation": V3R6_ANGLE_INTERPRETATION,
    }
    protocol["frozen_inputs"] = {
        name: artifact(path) for name, path in dependencies.items()
    }

    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_p0/run/"
    )
    protocol["outputs"] = {
        "structural_audit": base + "barp_mcd_p0_structural_audit.json",
        "block_rows": base + "barp_mcd_p0_block_rows.jsonl",
        "gradient_diagnostics": base + "barp_mcd_p0_gradient_diagnostics.json",
        "environment": base + "barp_mcd_p0_environment.json",
        "route": base + "barp_mcd_p0_route.json",
        "report": base + "BARP_MCD_P0_REPORT.md",
        "run_manifest": base + "barp_mcd_p0_reproduction_manifest.json",
    }
    protocol["pass_authorizes"] = (
        "a_separately_frozen_matched_training_protocol_that_recomputes_BARP_"
        "MCD_on_every_training_batch"
    )
    protocol["pass_does_not_authorize"] = (
        "optimizer_steps_inside_P0_candidate_generation_core100_heldout_or_"
        "independent_transfer_claims"
    )
    protocol["optimizer_objects"] = 0
    protocol["scheduler_objects"] = 0
    protocol["optimizer_steps"] = 0
    protocol["training_authorized"] = False
    protocol["candidate_generation_authorized"] = False
    protocol["core100_opened"] = False
    protocol["heldout_opened"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": relative(output),
                "sha256": sha256(output),
                "entry_commit": protocol["entry_commit"],
                "optimizer_units": len(protocol["selection"]["optimizer_blocks"]),
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
