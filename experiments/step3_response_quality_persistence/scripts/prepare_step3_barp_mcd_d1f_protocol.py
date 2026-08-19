"""Prepare the prospective BARP-MCD D1 failure-state feasibility protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_D1F_update_space_anchor_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_d1f_seed5.json"
RUNNER = BASE / "scripts/step3_barp_mcd_d1f.py"
TESTS = BASE / "scripts/test_step3_barp_mcd_d1f.py"
PREPARER = Path(__file__).resolve()
D1_PROTOCOL = (
    BASE / "results/4070ti_step3_barp_mcd_d1/protocol/barp_mcd_d1_protocol.json"
)
D1_STOP = BASE / "results/4070ti_step3_barp_mcd_d1/analysis/barp_mcd_d1_stop.json"
D1_TRACE = BASE / "results/4070ti_step3_barp_mcd_d1/raw_mcd/training_trace.jsonl"
OUTPUT = BASE / "results/4070ti_step3_barp_mcd_d1f/protocol/barp_mcd_d1f_protocol.json"


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

    d1_protocol = read_json(D1_PROTOCOL)
    stop = read_json(D1_STOP)
    if (
        stop.get("route") != "STOP_BARP_MCD_D1_RAW_ANCHOR_FAILED"
        or int(stop["completed"]["raw_mcd_optimizer_steps"]) != 3
        or stop["failure"].get("next_optimizer_step") != 4
        or stop["failure"].get("optimizer_step_executed_for_failed_block") is not False
    ):
        raise ValueError("D1F precursor registry differs")
    dependencies = {
        "d1f_plan": PLAN,
        "config": CONFIG,
        "d1f_runner": RUNNER,
        "d1f_tests": TESTS,
        "protocol_preparer": PREPARER,
        "d1_runner": BASE / "scripts/step3_barp_mcd_d1.py",
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
        "d1_trace": D1_TRACE,
        "approved_targets": ROOT / d1_protocol["source"]["safety_targets_path"],
        "risk_source": ROOT / d1_protocol["source"]["risk_data_path"],
        "structural_audit": ROOT / d1_protocol["source"]["structural_audit_path"],
    }
    design = dict(d1_protocol["design"])
    design.update(
        {
            "method_variant": "optimizer_update_space_maximin_anchor",
            "anchor_endpoint_candidates": 127,
            "anchor_screened_endpoints": 16,
            "anchor_interior_candidates": 112,
            "anchor_total_exact_map_candidates": 239,
            "anchor_endpoint_screening": (
                "overall_min_then_target_min_then_target_mean_then_smaller_subset_then_lower_bitmask"
            ),
            "anchor_selection": (
                "overall_min_then_target_min_then_target_mean_then_smaller_fraction_then_smaller_subset_then_lower_bitmask"
            ),
            "anchor_constructed_in_optimizer_update_space": True,
            "relative_protection_applied_in_d1f": False,
        }
    )
    root = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_d1f"
    )
    protocol = {
        "name": "step3_barp_mcd_d1f_update_space_anchor_feasibility_protocol_v1",
        "status": "prospective_before_any_BARP_MCD_D1F_optimizer_step",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Reconstruct the D1 pre-step-four AdamW state and test whether the "
            "registered 239-candidate family contains a strict update-space common-descent anchor."
        ),
        "experiment_mode": (
            "seed5_d1_failure_state_reconstruction_and_zero_target_state_update_feasibility"
        ),
        "precursor": {
            "d1_protocol_path": relative(D1_PROTOCOL),
            "d1_protocol_sha256": sha256(D1_PROTOCOL),
            "stop_path": relative(D1_STOP),
            "stop_sha256": sha256(D1_STOP),
            "stop_route": stop["route"],
            "d1_trace_path": relative(D1_TRACE),
            "d1_trace_sha256": sha256(D1_TRACE),
            "completed_d1_raw_steps": 3,
            "failed_next_step": 4,
        },
        "frozen_inputs": {name: artifact(path) for name, path in dependencies.items()},
        "source": d1_protocol["source"],
        "selection": d1_protocol["selection"],
        "design": design,
        "replay_optimizer_steps": 3,
        "target_state_optimizer_step": 4,
        "total_authorized_optimizer_steps": 3,
        "statistics": {
            "unit": "exact_current_state_adamw_candidate",
            "endpoint_candidates": 127,
            "interior_candidates": 112,
            "total_candidates": 239,
            "primary_statistic": "maximum_over_candidates_of_minimum_over_seven_exact_scores",
            "descriptive": [
                "strict_feasible_candidate_count",
                "full_mcd_exact_score_vector",
                "selected_anchor_exact_score_vector"
            ],
        },
        "gates": {
            "replay_matching_tolerance": 1e-7,
            "optimizer_map_relative_error_max": 5e-6,
            "d1_full_anchor_margin_max": 0.0,
            "anchor_margin_min_exclusive": 1e-8,
            "minimum_vram_headroom_fraction": 0.1,
        },
        "output_root": root,
        "outputs": {
            "replay_trace": root + "/run/raw_replay_trace.jsonl",
            "feasibility": root + "/run/barp_mcd_d1f_feasibility.json",
            "decision": root + "/barp_mcd_d1f_decision.json",
            "report": root + "/BARP_MCD_D1F_REPORT.md",
            "reproduction_manifest": root + "/barp_mcd_d1f_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_D1F_UPDATE_SPACE_ANCHOR_FEASIBLE",
        "pass_authorizes": (
            "freeze_a_separate_matched_training_retry_with_update_space_anchor_selection_only"
        ),
        "pass_does_not_authorize": (
            "fourth_update_in_this_run_checkpoints_generation_core100_heldout_main_training_or_effect_claims"
        ),
        "training_authorized": True,
        "fourth_optimizer_step_authorized": False,
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
                "precursor_route": stop["route"],
                "authorized_optimizer_steps": 3,
                "target_state_optimizer_step": 4,
                "target_state_update_authorized": False,
                "exact_candidates": 239,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
