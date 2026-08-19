"""Prepare the prospective seed-5 BARP-MCD D1 matched-training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import step3_barp_mcd_d1 as d1


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_D1_seed5_matched_training_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_d1_seed5.json"
RUNNER = BASE / "scripts/step3_barp_mcd_d1.py"
RUNNER_TESTS = BASE / "scripts/test_step3_barp_mcd_d1.py"
PREPARER = Path(__file__).resolve()
Q1R_PROTOCOL = (
    BASE / "results/4070ti_step3_barp_mcd_q1r/protocol/barp_mcd_q1r_protocol.json"
)
Q1R_DECISION = BASE / "results/4070ti_step3_barp_mcd_q1r/barp_mcd_q1r_decision.json"
Q1R_ANALYSIS = (
    BASE / "results/4070ti_step3_barp_mcd_q1r/analysis/barp_mcd_q1r_analysis.json"
)
OUTPUT = BASE / "results/4070ti_step3_barp_mcd_d1/protocol/barp_mcd_d1_protocol.json"
DEFAULT_CHECKPOINT_ROOT = Path("D:/Desktop/step3-barp-mcd-d1-seed5-checkpoints")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
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

    q1r = read_json(Q1R_PROTOCOL)
    decision = read_json(Q1R_DECISION)
    analysis = read_json(Q1R_ANALYSIS)
    if (
        decision.get("route")
        != "PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL"
        or analysis.get("route")
        != "PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL"
    ):
        raise ValueError("D1 predecessor registry differs")
    config = read_json(CONFIG)
    targets_path = ROOT / config["safety_targets_path"]
    risk_path = ROOT / config["risk_data_path"]
    targets = read_jsonl(targets_path)
    safety_blocks, diagnostic_ids, categories = d1.expected_safety_selection(targets)
    risk_order = d1.expected_risk_order(int(config["seed"]), int(config["risk_rows"]))
    optimizer_blocks = [
        {
            "block_id": f"training_{index + 1:02d}",
            "risk_row_indices": risk_order[index * 4 : (index + 1) * 4],
            "safety_sample_ids": safety_blocks[index],
        }
        for index in range(8)
    ]
    diagnostic = {
        "block_id": "paired_diagnostic_8category_seed5",
        "risk_row_indices": risk_order[32:40],
        "safety_sample_ids": diagnostic_ids,
        "safety_categories": categories,
    }

    dependencies = {
        "d1_plan": PLAN,
        "config": CONFIG,
        "d1_runner": RUNNER,
        "d1_runner_tests": RUNNER_TESTS,
        "protocol_preparer": PREPARER,
        "q1_base_runner": BASE / "scripts/step3_barp_mcd_training.py",
        "q1r_memory_bounded_runner": BASE / "scripts/step3_barp_mcd_training_q1r.py",
        "q1r_protocol": Q1R_PROTOCOL,
        "q1r_decision": Q1R_DECISION,
        "q1r_analysis": Q1R_ANALYSIS,
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
        "approved_targets": targets_path,
        "risk_source": risk_path,
        "structural_audit": ROOT / q1r["source"]["structural_audit_path"],
    }
    design = dict(q1r["design"])
    design.update(
        {
            "candidate_family_changed_from_q1r": False,
            "thresholds_changed_after_seed5_forward": False,
            "optimizer_steps_per_arm": 8,
            "diagnostic_gradients_used_for_selection": False,
            "diagnostic_selection_scope": "prospective_seed5_eight_category_pair",
        }
    )
    root = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_d1"
    )
    protocol = {
        "name": "step3_barp_mcd_d1_seed5_eight_step_matched_protocol_v1",
        "status": "prospective_before_any_BARP_MCD_D1_optimizer_step",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Test BARP-MCD over eight fresh-seed matched updates and freeze two "
            "external adapters for a later checkpoint-bound generated evaluation."
        ),
        "experiment_mode": (
            "prospective_seed5_eight_step_matched_training_confirmation_on_existing_approved_source"
        ),
        "precursor": {
            "protocol_path": relative(Q1R_PROTOCOL),
            "protocol_sha256": sha256(Q1R_PROTOCOL),
            "decision_path": relative(Q1R_DECISION),
            "decision_sha256": sha256(Q1R_DECISION),
            "analysis_path": relative(Q1R_ANALYSIS),
            "analysis_sha256": sha256(Q1R_ANALYSIS),
            "route": decision["route"],
        },
        "frozen_inputs": {name: artifact(path) for name, path in dependencies.items()},
        "source": {
            **q1r["source"],
            "role": "existing_approved_source_fresh_seed5_confirmation",
            "fresh_for_confirmation": False,
            "fresh_seed_and_split": True,
            "diagnostic_selects_route": True,
        },
        "selection": {
            "optimizer_blocks": optimizer_blocks,
            "diagnostic_block": diagnostic,
            "risk_order_seed": 5,
            "risk_order_algorithm": "torch_randperm_cpu",
            "safety_diagnostic_algorithm": (
                "per_category_min_sha256_barp-d1-seed5-diagnostic_pipe_sample_id"
            ),
            "safety_training_algorithm": (
                "global_sha256_sort_barp-d1-seed5-training_pipe_sample_id_then_blocks_of_four"
            ),
            "optimizer_units_per_arm": 8,
            "risk_rows_per_unit": 4,
            "safety_pairs_per_unit": 4,
            "diagnostic_units": 8,
            "training_diagnostic_overlap": 0,
            "diagnostic_selects_route": True,
        },
        "design": design,
        "arm_order": list(d1.ARMS),
        "optimizer_steps_per_arm": 8,
        "total_authorized_optimizer_steps": 16,
        "statistics": {
            "unit": "paired_safety_category_and_risk_row",
            "paired_data_order": True,
            "diagnostic_units": 8,
            "normalized_advantage_formula": (
                "(raw_final_loss-barp_final_loss)/max(abs(shared_initial_loss),1e-8)"
            ),
            "target_macro_certificates": list(d1.TARGETS),
            "protected_disadvantage_certificates": list(d1.PROTECTED),
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 5001,
            "bootstrap_interval": "percentile_2.5_97.5_linear_interpolation",
            "sign_test": "two_sided_exact_binomial_excluding_ties",
            "sign_test_tie_tolerance": 0.0,
        },
        "gates": {
            "matching_tolerance": 1e-7,
            "optimizer_map_relative_error_max": 5e-6,
            "raw_anchor_score_min": 0.0,
            "minimum_nonzero_barp_choices": 6,
            "minimum_vram_headroom_fraction": 0.1,
            "all_same_batch_finite_step_reductions_positive": True,
            "protected_mean_disadvantage_max": 0.05,
            "protected_bootstrap_upper_max": 0.075,
            "protected_worst_unit_disadvantage_max": 0.15,
            "target_macro_mean_advantage_min": 0.002,
            "target_macro_bootstrap_lower_min": 0.0,
            "target_macro_sign_test_p_max": 0.05,
        },
        "checkpoint_root": checkpoint_root.as_posix(),
        "checkpoint_policy": {
            "directories": list(d1.ARMS),
            "exact_writes": 2,
            "required_files": ["adapter_model.safetensors", "adapter_config.json"],
            "optimizer_state_authorized": False,
            "base_model_state_authorized": False,
            "sha256_inventory_required": True,
        },
        "output_root": root,
        "arm_outputs": {
            "raw_mcd": root + "/raw_mcd",
            "barp_mcd": root + "/barp_mcd",
        },
        "outputs": {
            "decision": root + "/barp_mcd_d1_decision.json",
            "report": root + "/BARP_MCD_D1_REPORT.md",
            "paired_diagnostic_analysis": root + "/analysis/barp_mcd_d1_paired_analysis.json",
            "reproduction_manifest": root + "/barp_mcd_d1_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_D1_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL",
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
                "precursor_route": decision["route"],
                "seed": 5,
                "optimizer_steps_per_arm": 8,
                "total_authorized_optimizer_steps": 16,
                "diagnostic_units": 8,
                "checkpoint_root": checkpoint_root.as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
