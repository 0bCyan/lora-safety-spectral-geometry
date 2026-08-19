"""Prepare the prospective memory-bounded BARP-MCD Q1 retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_Q1R_memory_bounded_retry.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_q1r_seed4.json"
RUNNER = BASE / "scripts/step3_barp_mcd_training_q1r.py"
RUNNER_TESTS = BASE / "scripts/test_step3_barp_mcd_training_q1r.py"
PREPARER = Path(__file__).resolve()
Q1_PROTOCOL = (
    BASE
    / "results/4070ti_step3_barp_mcd_q1/protocol/barp_mcd_q1_protocol.json"
)
Q1_DECISION = (
    BASE / "results/4070ti_step3_barp_mcd_q1/barp_mcd_q1_decision.json"
)
Q1_STOP = (
    BASE
    / "results/4070ti_step3_barp_mcd_q1/analysis/barp_mcd_q1_stop_analysis.json"
)
P0_PROTOCOL = (
    BASE
    / "results/4070ti_step3_barp_mcd_p0/protocol/barp_mcd_p0_protocol.json"
)
P0_ROUTE = BASE / "results/4070ti_step3_barp_mcd_p0/run/barp_mcd_p0_route.json"
OUTPUT = (
    BASE
    / "results/4070ti_step3_barp_mcd_q1r/protocol/barp_mcd_q1r_protocol.json"
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

    q1 = read_json(Q1_PROTOCOL)
    decision = read_json(Q1_DECISION)
    stop = read_json(Q1_STOP)
    p0 = read_json(P0_PROTOCOL)
    p0_route = read_json(P0_ROUTE)
    if (
        decision.get("route") != "STOP_BARP_MCD_Q1_VRAM_FAILED"
        or stop["root_cause"]["classification"]
        != "implementation_tensor_lifetime_not_method_geometry"
        or p0_route.get("route")
        != "PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL"
    ):
        raise ValueError("Q1R predecessor registry differs")

    dependencies = {
        "retry_plan": PLAN,
        "config": CONFIG,
        "retry_runner": RUNNER,
        "retry_runner_tests": RUNNER_TESTS,
        "protocol_preparer": PREPARER,
        "q1_base_runner": BASE / "scripts/step3_barp_mcd_training.py",
        "q1_base_runner_tests": BASE / "scripts/test_step3_barp_mcd_training.py",
        "q1_protocol": Q1_PROTOCOL,
        "q1_decision": Q1_DECISION,
        "q1_stop_analysis": Q1_STOP,
        "p0_protocol": P0_PROTOCOL,
        "p0_route": P0_ROUTE,
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
        "structural_audit": ROOT / q1["source"]["structural_audit_path"],
    }
    design = dict(q1["design"])
    design.update(
        {
            "endpoint_tensor_policy": (
                "score_delete_then_reconstruct_screened_masks"
            ),
            "maximum_retained_endpoint_directions": 1,
            "candidate_family_changed_from_q1": False,
            "thresholds_changed_from_q1": False,
            "optimizer_budget_reused_from_q1": False,
        }
    )
    source = dict(q1["source"])
    source["role"] = "disclosed_seed4_memory_bounded_q1_retry"
    root = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_q1r"
    )
    protocol = {
        "name": "step3_barp_mcd_q1r_memory_bounded_two_step_protocol_v1",
        "status": "prospective_before_any_BARP_MCD_Q1R_optimizer_step",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Repeat the frozen Q1 qualification with endpoint tensors deleted "
            "after scoring and screened masks deterministically reconstructed."
        ),
        "experiment_mode": (
            "exploratory_disclosed_seed4_memory_bounded_two_step_matched_"
            "training_qualification_retry"
        ),
        "precursor": {
            "protocol_path": relative(P0_PROTOCOL),
            "protocol_sha256": sha256(P0_PROTOCOL),
            "route_path": relative(P0_ROUTE),
            "route_sha256": sha256(P0_ROUTE),
            "route": p0_route["route"],
        },
        "retry_basis": {
            "q1_protocol_path": relative(Q1_PROTOCOL),
            "q1_protocol_sha256": sha256(Q1_PROTOCOL),
            "q1_decision_path": relative(Q1_DECISION),
            "q1_decision_sha256": sha256(Q1_DECISION),
            "q1_route": decision["route"],
            "q1_stop_analysis_path": relative(Q1_STOP),
            "q1_stop_analysis_sha256": sha256(Q1_STOP),
            "registered_estimated_avoidable_gib": stop["root_cause"][
                "estimated_retained_endpoint_gib"
            ],
        },
        "frozen_inputs": {
            name: artifact(path) for name, path in dependencies.items()
        },
        "source": source,
        "selection": q1["selection"],
        "design": design,
        "arm_order": q1["arm_order"],
        "optimizer_steps_per_arm": q1["optimizer_steps_per_arm"],
        "total_authorized_optimizer_steps": q1[
            "total_authorized_optimizer_steps"
        ],
        "statistics": q1["statistics"],
        "gates": q1["gates"],
        "output_root": root,
        "arm_outputs": {
            "raw_mcd": root + "/raw_mcd",
            "barp_mcd": root + "/barp_mcd",
        },
        "outputs": {
            "decision": root + "/barp_mcd_q1r_decision.json",
            "report": root + "/BARP_MCD_Q1R_REPORT.md",
            "reproduction_manifest": root
            + "/barp_mcd_q1r_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL",
        "pass_authorizes": q1["pass_authorizes"],
        "pass_does_not_authorize": q1["pass_does_not_authorize"],
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
                "q1_route": decision["route"],
                "maximum_retained_endpoint_directions": 1,
                "total_authorized_optimizer_steps": 4,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
