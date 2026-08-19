"""Prepare the prospective optimizer-anchor BARP-MCD D1R protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_D1R_optimizer_anchor_matched_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_d1r_seed5.json"
RUNNER = BASE / "scripts/step3_barp_mcd_d1r.py"
TESTS = BASE / "scripts/test_step3_barp_mcd_d1r.py"
PREPARER = Path(__file__).resolve()
D1_PROTOCOL = (
    BASE / "results/4070ti_step3_barp_mcd_d1/protocol/barp_mcd_d1_protocol.json"
)
D1_STOP = BASE / "results/4070ti_step3_barp_mcd_d1/analysis/barp_mcd_d1_stop.json"
D1F_PROTOCOL = (
    BASE / "results/4070ti_step3_barp_mcd_d1f/protocol/barp_mcd_d1f_protocol.json"
)
D1F_DECISION = BASE / "results/4070ti_step3_barp_mcd_d1f/barp_mcd_d1f_decision.json"
D1F_ANALYSIS = (
    BASE / "results/4070ti_step3_barp_mcd_d1f/analysis/barp_mcd_d1f_analysis.json"
)
Q1R_DECISION = BASE / "results/4070ti_step3_barp_mcd_q1r/barp_mcd_q1r_decision.json"
OUTPUT = BASE / "results/4070ti_step3_barp_mcd_d1r/protocol/barp_mcd_d1r_protocol.json"
CHECKPOINT_ROOT = Path("D:/Desktop/step3-barp-mcd-d1r-seed5-checkpoints")


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
    parser.add_argument("--checkpoint-root", default=str(CHECKPOINT_ROOT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")
    if checkpoint_root.exists():
        raise SystemExit(f"prospective checkpoint root already exists: {checkpoint_root}")
    try:
        checkpoint_root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("checkpoint root must be outside the repository")

    d1_protocol = read_json(D1_PROTOCOL)
    d1_stop = read_json(D1_STOP)
    d1f = read_json(D1F_PROTOCOL)
    d1f_decision = read_json(D1F_DECISION)
    q1r_decision = read_json(Q1R_DECISION)
    if (
        d1_stop.get("route") != "STOP_BARP_MCD_D1_RAW_ANCHOR_FAILED"
        or d1f_decision.get("route")
        != "PASS_BARP_MCD_D1F_UPDATE_SPACE_ANCHOR_FEASIBLE"
        or q1r_decision.get("route")
        != "PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL"
    ):
        raise ValueError("D1R precursor registry differs")
    dependencies = {
        "d1r_plan": PLAN,
        "config": CONFIG,
        "d1r_runner": RUNNER,
        "d1r_tests": TESTS,
        "protocol_preparer": PREPARER,
        "d1_runner": BASE / "scripts/step3_barp_mcd_d1.py",
        "d1f_runner": BASE / "scripts/step3_barp_mcd_d1f.py",
        "q1_base_runner": BASE / "scripts/step3_barp_mcd_training.py",
        "q1r_memory_bounded_runner": BASE / "scripts/step3_barp_mcd_training_q1r.py",
        "rpfrsc_runner": BASE / "scripts/step3_rpfrsc_p0.py",
        "protected_search": BASE / "scripts/step3_pfrsc_common.py",
        "finite_radius_geometry": BASE / "scripts/step3_frsc_common.py",
        "optimizer_geometry": BASE / "scripts/step3_ojmcd_common.py",
        "gradient_math": BASE / "scripts/step3_ramp_common.py",
        "training_common": BASE / "scripts/step3ct_training_common.py",
        "model_loader": BASE / "scripts/train_step3ct_tournament.py",
        "d1_protocol": D1_PROTOCOL,
        "d1_stop": D1_STOP,
        "d1f_protocol": D1F_PROTOCOL,
        "d1f_decision": D1F_DECISION,
        "d1f_analysis": D1F_ANALYSIS,
        "q1r_decision": Q1R_DECISION,
        "approved_targets": ROOT / d1_protocol["source"]["safety_targets_path"],
        "risk_source": ROOT / d1_protocol["source"]["risk_data_path"],
        "structural_audit": ROOT / d1_protocol["source"]["structural_audit_path"],
    }
    design = dict(d1_protocol["design"])
    design.update(
        {
            "method": "optimizer-anchor BARP-MCD",
            "anchor_definition": "max_over_239_candidates_min_over_7_exact_current_adamw_scores",
            "anchor_margin_floor_exclusive": 1e-8,
            "anchor_endpoint_candidates": 127,
            "anchor_screened_endpoints": 16,
            "anchor_interior_candidates": 112,
            "anchor_total_exact_map_candidates": 239,
            "anchor_included_in_protected_family": True,
            "protected_scores_relative_to": "positive_optimizer_space_anchor",
            "candidate_family_changed_from_d1f": False,
            "diagnostic_thresholds_changed_from_d1": False,
            "maximum_retained_endpoint_directions": 2,
        }
    )
    gates = dict(d1_protocol["gates"])
    gates["raw_anchor_score_min"] = 1e-8
    gates["minimum_nonzero_barp_choices"] = 6
    root = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_d1r"
    )
    protocol = {
        "name": "step3_barp_mcd_d1r_optimizer_anchor_eight_step_protocol_v1",
        "status": "prospective_before_any_BARP_MCD_D1R_optimizer_step",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Compare an exact optimizer-space max-min anchor against route-protected "
            "target selection relative to that anchor over eight matched seed5 updates."
        ),
        "experiment_mode": (
            "prospective_seed5_eight_step_optimizer_anchor_vs_route_protected_matched_confirmation"
        ),
        "precursor": {
            "decision_path": relative(Q1R_DECISION),
            "decision_sha256": sha256(Q1R_DECISION),
            "route": q1r_decision["route"],
        },
        "mechanism_basis": {
            "d1_protocol_path": relative(D1_PROTOCOL),
            "d1_protocol_sha256": sha256(D1_PROTOCOL),
            "d1_stop_path": relative(D1_STOP),
            "d1_stop_sha256": sha256(D1_STOP),
            "d1_stop_route": d1_stop["route"],
            "d1f_protocol_path": relative(D1F_PROTOCOL),
            "d1f_protocol_sha256": sha256(D1F_PROTOCOL),
            "d1f_decision_path": relative(D1F_DECISION),
            "d1f_decision_sha256": sha256(D1F_DECISION),
            "d1f_route": d1f_decision["route"],
            "d1f_strict_feasible_candidates": d1f_decision["strict_feasible_candidates"],
            "d1f_total_candidates": d1f_decision["total_candidates"],
        },
        "frozen_inputs": {name: artifact(path) for name, path in dependencies.items()},
        "source": {
            **d1_protocol["source"],
            "role": "existing_approved_source_seed5_optimizer_anchor_confirmation",
        },
        "selection": d1_protocol["selection"],
        "design": design,
        "arm_order": ["raw_mcd", "barp_mcd"],
        "arm_semantics": {
            "raw_mcd": "apply_optimizer_space_maximin_anchor",
            "barp_mcd": "apply_target_best_candidate_subject_to_95_percent_anchor_protection",
        },
        "optimizer_steps_per_arm": 8,
        "total_authorized_optimizer_steps": 16,
        "statistics": d1_protocol["statistics"],
        "gates": gates,
        "checkpoint_root": checkpoint_root.as_posix(),
        "checkpoint_policy": d1_protocol["checkpoint_policy"],
        "output_root": root,
        "arm_outputs": {
            "raw_mcd": root + "/raw_mcd",
            "barp_mcd": root + "/barp_mcd",
        },
        "outputs": {
            "decision": root + "/barp_mcd_d1r_decision.json",
            "report": root + "/BARP_MCD_D1R_REPORT.md",
            "paired_diagnostic_analysis": root + "/analysis/barp_mcd_d1r_paired_analysis.json",
            "reproduction_manifest": root + "/barp_mcd_d1r_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_D1R_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL",
        "pass_authorizes": (
            "freeze_and_push_a_separate_checkpoint_hash_bound_generated_safety_utility_protocol_only"
        ),
        "pass_does_not_authorize": (
            "direct_generation_core100_heldout_additional_optimizer_steps_main_training_or_paper_claims"
        ),
        "training_authorized": True,
        "checkpoint_writes_authorized": True,
        "checkpoint_writes_exact": 2,
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
                "d1f_route": d1f_decision["route"],
                "optimizer_steps_per_arm": 8,
                "total_authorized_optimizer_steps": 16,
                "candidate_family_per_step": 239,
                "checkpoint_root": checkpoint_root.as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
