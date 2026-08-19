"""Prepare the prospective single-route D1X-C1 qualification protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_D1X_C1_conditional_qualification_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_d1x_seed6.json"
RUNNER = BASE / "scripts/step3_barp_mcd_d1x_c1.py"
TESTS = BASE / "scripts/test_step3_barp_mcd_d1x_c1.py"
PREPARER = Path(__file__).resolve()
D1X_ROOT = BASE / "results/4070ti_step3_barp_mcd_d1x_r1"
D1X_PROTOCOL = D1X_ROOT / "protocol/barp_mcd_d1x_r1_protocol.json"
D1X_DECISION = D1X_ROOT / "barp_mcd_d1x_decision.json"
D1X_DISCOVERY = D1X_ROOT / "analysis/barp_mcd_d1x_discovery.json"
D1X_ANALYSIS = D1X_ROOT / "analysis/barp_mcd_d1x_analysis.json"
OUTPUT = (
    BASE / "results/4070ti_step3_barp_mcd_d1x_c1/protocol/"
    "barp_mcd_d1x_c1_protocol.json"
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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")
    d1x_protocol = read_json(D1X_PROTOCOL)
    decision = read_json(D1X_DECISION)
    analysis = read_json(D1X_ANALYSIS)
    candidate = analysis["exploratory_unique_target_sign_route"]
    if (
        decision.get("route") != "STOP_BARP_MCD_D1X_DISCOVERY_NO_STABLE_ROUTE"
        or decision.get("qualification") is not None
        or candidate.get("route_id") != "mask072_lambda0.500"
        or candidate.get("post_selection_status") != "hypothesis_only_not_a_D1X_pass"
        or analysis["boundary"].get("qualification_gradients_opened") is not False
    ):
        raise ValueError("C1 post-selection basis differs")
    dependencies = {
        "c1_plan": PLAN,
        "config": CONFIG,
        "c1_runner": RUNNER,
        "c1_tests": TESTS,
        "protocol_preparer": PREPARER,
        "d1x_runner": BASE / "scripts/step3_barp_mcd_d1x.py",
        "d1r_runner": BASE / "scripts/step3_barp_mcd_d1r.py",
        "d1f_runner": BASE / "scripts/step3_barp_mcd_d1f.py",
        "training_runner": BASE / "scripts/step3_barp_mcd_training.py",
        "rpfrsc_runner": BASE / "scripts/step3_rpfrsc_p0.py",
        "finite_radius_geometry": BASE / "scripts/step3_frsc_common.py",
        "protected_search": BASE / "scripts/step3_pfrsc_common.py",
        "gradient_math": BASE / "scripts/step3_ramp_common.py",
        "training_common": BASE / "scripts/step3ct_training_common.py",
        "model_loader": BASE / "scripts/train_step3ct_tournament.py",
        "d1x_protocol": D1X_PROTOCOL,
        "d1x_decision": D1X_DECISION,
        "d1x_discovery": D1X_DISCOVERY,
        "d1x_analysis": D1X_ANALYSIS,
        "approved_targets": ROOT / d1x_protocol["source"]["safety_targets_path"],
        "risk_source": ROOT / d1x_protocol["source"]["risk_data_path"],
        "structural_audit": ROOT / d1x_protocol["source"]["structural_audit_path"],
    }
    root = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_d1x_c1"
    )
    descriptor = {
        "route_id": "mask072_lambda0.500",
        "source_mask": 72,
        "source_subset": ["etarc", "branch_negative_gap"],
        "homotopy_fraction": 0.5,
    }
    protocol = {
        "name": "step3_barp_mcd_d1x_c1_conditional_qualification_protocol_v1",
        "status": "prospective_before_any_D1X_C1_qualification_forward",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Qualify the single post-selected mask072 lambda0.5 conditional anchor-"
            "fallback policy on the four still-sealed seed6 qualification blocks."
        ),
        "basis": {
            "d1x_protocol_path": relative(D1X_PROTOCOL),
            "d1x_protocol_sha256": sha256(D1X_PROTOCOL),
            "decision_path": relative(D1X_DECISION),
            "decision_sha256": sha256(D1X_DECISION),
            "analysis_path": relative(D1X_ANALYSIS),
            "analysis_sha256": sha256(D1X_ANALYSIS),
            "discovery_path": relative(D1X_DISCOVERY),
            "discovery_sha256": sha256(D1X_DISCOVERY),
            "post_selection_disclosed": True,
            "original_D1X_route": decision["route"],
            "original_D1X_decision_changed": False,
            "qualification_gradients_previously_opened": False,
            "only_C1_descriptor_authorized": "mask072_lambda0.500",
        },
        "frozen_inputs": {name: artifact(path) for name, path in dependencies.items()},
        "source": d1x_protocol["source"],
        "selection": {
            "qualification_blocks": d1x_protocol["selection"]["qualification_blocks"],
            "sealed_diagnostic_block": d1x_protocol["selection"]["sealed_diagnostic_block"],
            "discovery_blocks_reopened": False,
            "qualification_opened_once_for_bound_descriptor": True,
            "sealed_diagnostic_forward_authorized": False,
        },
        "design": {
            "method": "single_route_conditional_optimizer_anchor_fallback",
            "route_descriptor": descriptor,
            "anchor_definition": "exact_current_adamw_update_space_maximin",
            "anchor_family_size": 239,
            "local_protected_retention_floor": 0.95,
            "fallback": "apply_anchor_when_route_is_locally_infeasible_or_equals_anchor",
            "protected_certificates": ["risk", "uniform", "tarc", "continuation"],
            "target_certificates": ["etarc", "branch_entropy", "branch_negative_gap"],
            "optimizer_map": "exact_zero_state_torch_AdamW_with_global_clipping",
            "cross_block_estimand": d1x_protocol["design"]["cross_block_estimand"],
            "studentized_stability_coefficient": 2.3533634348018264,
            "leave_one_out_proposal_clusters_are_dependent": True,
            "stability_lower_score_is_not_a_confidence_interval": True,
            "population_inference_claimed": False,
        },
        "gates": {
            "proposal_activity_min": 3,
            "applied_local_protected_retention_min": 0.95,
            "target_macro_mean_min": 0.002,
            "target_stability_lower_min": 0.0,
            "positive_proposal_clusters_min": 3,
            "negative_proposal_clusters_max": 0,
            "target_component_mean_min": 0.0,
            "protected_mean_disadvantage_max": 0.005,
            "protected_worst_disadvantage_max": 0.02,
        },
        "output_root": root,
        "outputs": {
            "qualification": root + "/analysis/barp_mcd_d1x_c1_qualification.json",
            "decision": root + "/barp_mcd_d1x_c1_decision.json",
            "report": root + "/BARP_MCD_D1X_C1_REPORT.md",
            "reproduction_manifest": root + "/barp_mcd_d1x_c1_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_D1X_C1_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        "pass_authorizes": "freeze_a_separate_conditional_anchor_fallback_matched_training_protocol_only",
        "pass_does_not_authorize": (
            "optimizer_steps_direct_generation_diagnostic_core100_heldout_or_paper_claims"
        ),
        "failure_closes_current_BARP_target_route_line": True,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "checkpoint_writes_authorized": False,
        "candidate_generation_authorized": False,
        "sealed_diagnostic_opened": False,
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
                "route_id": descriptor["route_id"],
                "qualification_blocks": 4,
                "optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
