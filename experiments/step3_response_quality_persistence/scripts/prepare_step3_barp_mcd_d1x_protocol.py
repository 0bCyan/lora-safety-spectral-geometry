"""Prepare the prospective BARP-MCD D1X zero-step protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/step3_response_quality_persistence"
PLAN = BASE / "Step3_BARP_MCD_D1X_crossfit_zero_step_plan.md"
CONFIG = BASE / "configs/4070ti_qwen3_4b_step3_barp_mcd_d1x_seed6.json"
RUNNER = BASE / "scripts/step3_barp_mcd_d1x.py"
TESTS = BASE / "scripts/test_step3_barp_mcd_d1x.py"
PREPARER = Path(__file__).resolve()
D1R_DECISION = BASE / "results/4070ti_step3_barp_mcd_d1r/barp_mcd_d1r_decision.json"
D1R_ANALYSIS = (
    BASE / "results/4070ti_step3_barp_mcd_d1r/analysis/barp_mcd_d1r_analysis.json"
)
D1R_PROTOCOL = (
    BASE / "results/4070ti_step3_barp_mcd_d1r/protocol/barp_mcd_d1r_protocol.json"
)
OUTPUT = BASE / "results/4070ti_step3_barp_mcd_d1x/protocol/barp_mcd_d1x_protocol.json"


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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def seeded_digest(prefix: str, value: str) -> str:
    return hashlib.sha256((prefix + value).encode("utf-8")).hexdigest()


def split_safety(targets: list[dict[str, Any]]) -> tuple[list[list[str]], list[str], list[str]]:
    by_category: dict[str, set[str]] = {}
    for row in targets:
        by_category.setdefault(str(row["category"]), set()).add(str(row["sample_id"]))
    diagnostic = [
        min(
            by_category[category],
            key=lambda sample_id: seeded_digest(
                "barp-d1x-seed6-diagnostic|", sample_id
            ),
        )
        for category in sorted(by_category)
    ]
    all_ids = set().union(*by_category.values())
    training_ids = sorted(
        all_ids.difference(diagnostic),
        key=lambda sample_id: seeded_digest("barp-d1x-seed6-training|", sample_id),
    )
    if len(by_category) != 8 or len(all_ids) != 40 or len(training_ids) != 32:
        raise ValueError("D1X safety source differs")
    return (
        [training_ids[offset : offset + 4] for offset in range(0, 32, 4)],
        diagnostic,
        sorted(by_category),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")
    d1r = read_json(D1R_DECISION)
    if d1r.get("route") != "STOP_BARP_MCD_D1R_ASSIGNMENT_NOT_EXERCISED":
        raise ValueError("D1X precursor route differs")
    config = read_json(CONFIG)
    targets_path = ROOT / config["safety_targets_path"]
    risk_path = ROOT / config["risk_data_path"]
    targets = read_jsonl(targets_path)
    safety_blocks, diagnostic_safety, categories = split_safety(targets)

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(6)
    risk_order = torch.randperm(int(config["risk_rows"]), generator=generator).tolist()
    training_risk = risk_order[:32]
    diagnostic_risk = risk_order[32:40]
    blocks = [
        {
            "block_id": f"{('discovery' if index < 4 else 'qualification')}_{index % 4 + 1:02d}",
            "risk_row_indices": training_risk[index * 4 : (index + 1) * 4],
            "safety_sample_ids": safety_blocks[index],
        }
        for index in range(8)
    ]
    dependencies = {
        "d1x_plan": PLAN,
        "config": CONFIG,
        "d1x_runner": RUNNER,
        "d1x_tests": TESTS,
        "protocol_preparer": PREPARER,
        "d1r_runner": BASE / "scripts/step3_barp_mcd_d1r.py",
        "d1f_runner": BASE / "scripts/step3_barp_mcd_d1f.py",
        "training_runner": BASE / "scripts/step3_barp_mcd_training.py",
        "rpfrsc_runner": BASE / "scripts/step3_rpfrsc_p0.py",
        "finite_radius_geometry": BASE / "scripts/step3_frsc_common.py",
        "protected_search": BASE / "scripts/step3_pfrsc_common.py",
        "gradient_math": BASE / "scripts/step3_ramp_common.py",
        "training_common": BASE / "scripts/step3ct_training_common.py",
        "model_loader": BASE / "scripts/train_step3ct_tournament.py",
        "d1r_decision": D1R_DECISION,
        "d1r_analysis": D1R_ANALYSIS,
        "d1r_protocol": D1R_PROTOCOL,
        "approved_targets": targets_path,
        "risk_source": risk_path,
        "structural_audit": BASE
        / "results/4070ti_step3_mcd_p0/run/mcd_p0_structural_audit.json",
    }
    root = (
        "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_barp_mcd_d1x"
    )
    target_indices = [3, 5, 6]
    target_masks = [
        mask
        for mask in range(1, 128)
        if mask & ~sum(1 << index for index in target_indices) == 0
    ]
    protocol = {
        "name": "step3_barp_mcd_d1x_crossfit_zero_step_protocol_v1",
        "status": "prospective_before_any_BARP_MCD_D1X_model_forward",
        "protocol_path": relative(output),
        "entry_commit": git_head(),
        "config_path": relative(CONFIG),
        "objective": (
            "Select one target-subset optimizer route on four discovery blocks and "
            "qualify its exact AdamW transfer on four unopened blocks without updates."
        ),
        "precursor": {
            "decision_path": relative(D1R_DECISION),
            "decision_sha256": sha256(D1R_DECISION),
            "route": d1r["route"],
        },
        "frozen_inputs": {name: artifact(path) for name, path in dependencies.items()},
        "source": {
            "safety_targets_path": config["safety_targets_path"],
            "risk_data_path": config["risk_data_path"],
            "structural_audit_path": relative(dependencies["structural_audit"]),
            "safety_rows": 80,
            "safety_prompt_clusters": 40,
            "risk_rows": 256,
            "fresh_seed_and_split": True,
            "independent_source_claimed": False,
            "core100_used": False,
            "heldout_used": False,
        },
        "selection": {
            "discovery_blocks": blocks[:4],
            "qualification_blocks": blocks[4:],
            "sealed_diagnostic_block": {
                "block_id": "sealed_diagnostic_seed6_8category",
                "risk_row_indices": diagnostic_risk,
                "safety_sample_ids": diagnostic_safety,
                "safety_categories": categories,
            },
            "risk_order_seed": 6,
            "risk_order_algorithm": "torch_randperm_cpu",
            "safety_diagnostic_algorithm": (
                "per_category_min_sha256_barp-d1x-seed6-diagnostic_pipe_sample_id"
            ),
            "safety_training_algorithm": (
                "global_sha256_sort_barp-d1x-seed6-training_pipe_sample_id"
            ),
            "qualification_opened_only_after_route_freeze": True,
            "sealed_diagnostic_forward_authorized": False,
        },
        "design": {
            "method": "cross_fitted_optimizer_anchor_route",
            "certificates": [
                "risk",
                "uniform",
                "tarc",
                "etarc",
                "continuation",
                "branch_entropy",
                "branch_negative_gap",
            ],
            "protected_certificates": ["risk", "uniform", "tarc", "continuation"],
            "target_certificates": ["etarc", "branch_entropy", "branch_negative_gap"],
            "route_source_masks": target_masks,
            "route_fractions": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "route_family_size": 56,
            "anchor_family_size": 239,
            "anchor_definition": "exact_current_adamw_update_space_maximin",
            "local_protected_retention_floor": 0.95,
            "optimizer_map": "exact_zero_state_torch_AdamW_with_global_clipping",
            "cross_block_estimand": (
                "dot(normalized_support_gradient,route_minus_anchor_displacement)"
                "/proposal_baseline_displacement_norm"
            ),
            "cluster_unit": "proposal_block_mean_over_three_other_partition_blocks",
            "cluster_count": 4,
            "studentized_stability_coefficient_t_df3_one_sided_95": 2.3533634348018264,
            "leave_one_out_proposal_clusters_are_dependent": True,
            "stability_lower_score_is_not_a_confidence_interval": True,
            "population_inference_claimed": False,
        },
        "gates": {
            "required_proposal_activity": 4,
            "local_protected_retention_min": 0.95,
            "target_cluster_lower_min": 0.0,
            "discovery_positive_clusters_min": 3,
            "qualification_positive_clusters_min": 4,
            "qualification_target_mean_min": 0.002,
            "protected_mean_disadvantage_max": 0.005,
            "protected_worst_disadvantage_max": 0.02,
        },
        "output_root": root,
        "outputs": {
            "discovery": root + "/analysis/barp_mcd_d1x_discovery.json",
            "route_freeze": root + "/analysis/barp_mcd_d1x_route_freeze.json",
            "qualification": root + "/analysis/barp_mcd_d1x_qualification.json",
            "decision": root + "/barp_mcd_d1x_decision.json",
            "report": root + "/BARP_MCD_D1X_REPORT.md",
            "reproduction_manifest": root + "/barp_mcd_d1x_reproduction_manifest.json",
        },
        "pass_route": "PASS_BARP_MCD_D1X_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        "pass_authorizes": "freeze_a_separate_anchor_fallback_matched_training_protocol_only",
        "pass_does_not_authorize": (
            "optimizer_steps_direct_generation_core100_heldout_or_paper_claims"
        ),
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "checkpoint_writes_authorized": False,
        "candidate_generation_authorized": False,
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
                "discovery_blocks": 4,
                "qualification_blocks": 4,
                "route_family_size": 56,
                "optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
