"""Prepare the prospective BARP-MCD two-step training qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_matched_training_qualification_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_q1_seed4.json"
RUNNER = BASE / "scripts/step3_barp_mcd_training.py"
RUNNER_TESTS = BASE / "scripts/test_step3_barp_mcd_training.py"
PREPARER = Path(__file__).resolve()
P0_PROTOCOL = (
    BASE
    / "results/4070ti_step3_barp_mcd_p0/protocol/barp_mcd_p0_protocol.json"
)
P0_ROUTE = BASE / "results/4070ti_step3_barp_mcd_p0/run/barp_mcd_p0_route.json"
P0_ANALYSIS = (
    BASE
    / "results/4070ti_step3_barp_mcd_p0/analysis/BARP_MCD_P0_INTERPRETATION.md"
)
OUTPUT = (
    BASE
    / "results/4070ti_step3_barp_mcd_q1/protocol/barp_mcd_q1_protocol.json"
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

    p0 = read_json(P0_PROTOCOL)
    route = read_json(P0_ROUTE)
    if route.get("route") != "PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL":
        raise ValueError("BARP-MCD P0 did not authorize a matched protocol")
    dependencies = {
        "qualification_plan": PLAN,
        "config": CONFIG,
        "runner": RUNNER,
        "runner_tests": RUNNER_TESTS,
        "protocol_preparer": PREPARER,
        "p0_protocol": P0_PROTOCOL,
        "p0_route": P0_ROUTE,
        "p0_interpretation": P0_ANALYSIS,
        "p0_runner": BASE / "scripts/step3_barp_mcd_p0.py",
        "rpfrsc_runner": BASE / "scripts/step3_rpfrsc_p0.py",
        "protected_search": BASE / "scripts/step3_pfrsc_common.py",
        "finite_radius_geometry": BASE / "scripts/step3_frsc_common.py",
        "optimizer_geometry": BASE / "scripts/step3_ojmcd_common.py",
        "gradient_math": BASE / "scripts/step3_ramp_common.py",
        "objective_helpers": BASE / "scripts/step3_mcd_training.py",
        "risk_helpers": BASE / "scripts/step3_ramp_training.py",
        "training_common": BASE / "scripts/step3ct_training_common.py",
        "model_loader": BASE / "scripts/train_step3ct_tournament.py",
        "model_state_helper": BASE / "scripts/step3_cita_p0.py",
        "approved_targets": ROOT
        / "experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1.jsonl",
        "risk_source": ROOT / "data/processed/unified_pool/train_risk.jsonl",
        "structural_audit": ROOT / p0["source"]["structural_audit_path"],
    }
    design = dict(p0["design"])
    design.update(
        {
            "optimizer_map": "exact_current_state_torch_AdamW",
            "optimizer_state_scored_before_each_update": True,
            "optimizer_state_mutated_during_candidate_scoring": False,
            "weight_decay": 0.0,
            "scheduler": None,
            "same_batch_finite_step_evaluated_after_real_update": True,
            "descriptive_diagnostic_selects_route": False,
        }
    )
    source = dict(p0["source"])
    source.update(
        {
            "role": "disclosed_seed4_two_step_implementation_qualification",
            "fresh_for_confirmation": False,
            "independent_transfer_claimed": False,
            "core100_used": False,
            "heldout_used": False,
        }
    )
    base = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_q1"
    )
    protocol = {
        "name": "step3_barp_mcd_q1_matched_two_step_protocol_v1",
        "status": "prospective_before_any_BARP_MCD_Q1_optimizer_step",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Qualify two consecutive matched BARP-MCD optimizer updates while "
            "scoring every candidate through the current AdamW moment state."
        ),
        "experiment_mode": (
            "exploratory_disclosed_seed4_two_step_matched_training_qualification"
        ),
        "precursor": {
            "protocol_path": relative(P0_PROTOCOL),
            "protocol_sha256": sha256(P0_PROTOCOL),
            "route_path": relative(P0_ROUTE),
            "route_sha256": sha256(P0_ROUTE),
            "route": route["route"],
        },
        "frozen_inputs": {
            name: artifact(path) for name, path in dependencies.items()
        },
        "source": source,
        "selection": {
            "optimizer_blocks": p0["selection"]["optimizer_blocks"][:2],
            "diagnostic_block": p0["selection"]["excluded_source_block"],
            "optimizer_units_per_arm": 2,
            "risk_rows_per_unit": 4,
            "safety_pairs_per_unit": 4,
            "diagnostic_selects_route": False,
        },
        "design": design,
        "arm_order": ["raw_mcd", "barp_mcd"],
        "optimizer_steps_per_arm": 2,
        "total_authorized_optimizer_steps": 4,
        "statistics": {
            "unit": "optimizer_step",
            "paired_data_order": True,
            "optimizer_steps_per_arm": 2,
            "fresh_confirmation": False,
            "diagnostic_role": "descriptive_only",
        },
        "gates": {
            "matching_tolerance": 1e-7,
            "optimizer_map_relative_error_max": 5e-6,
            "raw_anchor_score_min": 0.0,
            "minimum_nonzero_barp_choices": 1,
            "minimum_vram_headroom_fraction": 0.1,
            "all_same_batch_finite_step_reductions_positive": True,
        },
        "output_root": base,
        "arm_outputs": {
            "raw_mcd": base + "/raw_mcd",
            "barp_mcd": base + "/barp_mcd",
        },
        "outputs": {
            "decision": base + "/barp_mcd_q1_decision.json",
            "report": base + "/BARP_MCD_Q1_REPORT.md",
            "reproduction_manifest": base
            + "/barp_mcd_q1_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_Q1_AUTHORIZE_LONGER_MATCHED_PROTOCOL",
        "pass_authorizes": (
            "freeze_and_push_a_separate_longer_matched_training_protocol_only"
        ),
        "pass_does_not_authorize": (
            "candidate_generation_core100_heldout_main_training_or_behavior_claims"
        ),
        "training_authorized": True,
        "checkpoint_writes_authorized": False,
        "candidate_generation_authorized": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": relative(output),
                "sha256": sha256(output),
                "entry_commit": protocol["entry_commit"],
                "total_authorized_optimizer_steps": 4,
                "training_authorized": True,
                "checkpoint_writes_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
