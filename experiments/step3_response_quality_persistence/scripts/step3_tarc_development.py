"""Run and verify the separately authorized TARC development-16 stage."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import step3_ramp_training as artifact
import step3_tarc_training as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_tarc/"
    "protocol/tarc_development16_protocol.json"
)


class TarcDevelopmentValidationError(ValueError):
    """Raised when development-16 violates its prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    return base.resolve_repo_path(value)


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    protocol, config, targets, diagnostic_ids = base.context(path)
    if protocol.get("run_kind") != "development":
        raise TarcDevelopmentValidationError("protocol is not development")
    if int(protocol["optimizer_steps_per_arm"]) != 16:
        raise TarcDevelopmentValidationError("development must use 16 steps per arm")
    smoke = artifact.read_json(
        resolve_repo_path(protocol["frozen_inputs"]["smoke_decision"]["path"])
    )
    if smoke.get("route") != "PASS_TARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL":
        raise TarcDevelopmentValidationError("smoke did not authorize development-16")
    reproduction = artifact.read_json(
        resolve_repo_path(
            protocol["frozen_inputs"]["smoke_reproduction_manifest"]["path"]
        )
    )
    if reproduction.get("route") != smoke.get("route"):
        raise TarcDevelopmentValidationError("smoke reproduction route differs")
    draws = 16 * int(config["gradient_accumulation_steps"])
    overlap = len(
        set(config["training_prompt_order"][:draws]).intersection(diagnostic_ids)
    )
    if overlap != int(protocol["diagnostic_prompt_overlap_with_development_draws"]):
        raise TarcDevelopmentValidationError("development diagnostic overlap differs")
    return protocol, config, targets, diagnostic_ids


def select_development_route(checks: dict[str, Any]) -> str:
    matching = (
        "all_arms_present",
        "initial_state_equal",
        "risk_order_equal",
        "target_order_equal",
        "target_exposure_equal",
        "safety_pair_draws_equal",
        "safety_target_rows_equal",
        "safety_response_tokens_equal",
        "available_ramp_terms_equal",
        "available_tarc_terms_equal",
        "first_step_risk_loss_equal",
        "first_step_uniform_safety_loss_equal",
        "initial_uniform_diagnostic_equal",
        "initial_ramp_diagnostic_equal",
        "initial_tarc_diagnostic_equal",
        "sealed_data_untouched",
    )
    numerical = ("steps_exact", "no_skipped_updates", "finite", "vram")
    if not all(checks[name] for name in matching):
        return "STOP_TARC_DEVELOPMENT16_MATCHING_FAILED"
    if not all(checks[name] for name in numerical):
        return "STOP_TARC_DEVELOPMENT16_NUMERICAL_FAILED"
    if not (
        checks["assignment_counts_correct"]
        and checks["all_registered_assignments_exercised"]
    ):
        return "STOP_TARC_DEVELOPMENT16_ASSIGNMENT_NOT_EXERCISED"
    return "PASS_TARC_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE"


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, _, targets, diagnostic_ids = context(path)
    if not torch.cuda.is_available():
        raise TarcDevelopmentValidationError("CUDA is unavailable")
    result = {
        "status": "PASS_TARC_DEVELOPMENT16_PREFLIGHT",
        "protocol_sha256": artifact.sha256(path),
        "arms": protocol["arm_order"],
        "optimizer_steps_per_arm": 16,
        "total_authorized_optimizer_steps": 48,
        "safety_pair_draws_per_arm": 64,
        "safety_target_forwards_per_arm": 128,
        "source_rows": len(targets),
        "diagnostic_prompt_pairs": len(diagnostic_ids),
        "diagnostic_prompt_overlap_with_development_draws": protocol[
            "diagnostic_prompt_overlap_with_development_draws"
        ],
        "diagnostics_are_teacher_forced_descriptive": True,
        "gpu": torch.cuda.get_device_name(0),
        "candidate_generation_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def run(path: Path) -> dict[str, Any]:
    protocol, config, targets, diagnostic_ids = context(path)
    runtime_head = base.require_clean_start(path, protocol)
    root = resolve_repo_path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    manifests = {
        arm: base.train_arm(
            str(arm), protocol, config, targets, diagnostic_ids, runtime_head
        )
        for arm in protocol["arm_order"]
    }
    checks = base.compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_development_route(checks)
    decision = {
        "name": "step3_tarc_matched_development16_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": artifact.sha256(path),
        "checks": checks,
        "arms": {
            arm: {
                "objective": manifest["objective"],
                "completed_steps": manifest["completed_steps"],
                "optimizer_steps": manifest["optimizer_steps"],
                "final_adapter_sha256": manifest["final_adapter_sha256"],
                "final_checkpoint": manifest["final_checkpoint"],
                "tarc_loss_reduction": manifest["tarc_loss_reduction"],
                "final_diagnostic": manifest["final_diagnostic"],
                "vram_headroom_fraction": manifest["vram_headroom_fraction"],
            }
            for arm, manifest in manifests.items()
        },
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "generation_protocol_authorized": route
        == "PASS_TARC_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE",
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "next_authorized": (
            "freeze_and_push_separate_three_arm_prompt_disjoint_generation_gate"
            if route == "PASS_TARC_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE"
            else "stop_TARC_training"
        ),
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    artifact.write_json(decision_path, decision)
    report = f"""# TARC Matched Development-16 Integrity Report

Route: `{route}`.

Uniform, RAMP16, and TARC each completed 16 matched optimizer steps. All input,
exposure, first-step, numerical, assignment, VRAM, and sealed-data integrity
checks passed: `{route == 'PASS_TARC_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE'}`.

Teacher-forced TARC-loss reductions were
`{checks['uniform_tarc_loss_reduction']:.6f}` (uniform),
`{checks['ramp_tarc_loss_reduction']:.6f}` (RAMP16), and
`{checks['tarc_loss_reduction']:.6f}` (TARC). These in-sample diagnostics did
not enter the route and cannot establish generated behavior.

No candidate output, Core100, heldout, or reviewer data was opened. Total
optimizer steps: `{decision['total_optimizer_steps']}`;
`main_training_authorized=false`.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_bytes(report.encode("utf-8"))
    artifacts = []
    for arm in protocol["arm_order"]:
        arm_root = resolve_repo_path(protocol["arm_outputs"][arm])
        for name in ("training_trace.csv", "checkpoint_index.csv", "run_manifest.json"):
            file = arm_root / name
            artifacts.append(
                {"path": artifact.relative(file), "sha256": artifact.sha256(file), "bytes": file.stat().st_size}
            )
    for file in (decision_path, report_path):
        artifacts.append(
            {"path": artifact.relative(file), "sha256": artifact.sha256(file), "bytes": file.stat().st_size}
        )
    reproduction = {
        "name": "step3_tarc_development16_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": artifact.relative(path),
        "protocol_sha256": artifact.sha256(path),
        "artifacts": artifacts,
        "local_ignored_checkpoints": {
            arm: manifests[arm]["final_checkpoint"] for arm in protocol["arm_order"]
        },
        "final_adapter_sha256": {
            arm: manifests[arm]["final_adapter_sha256"] for arm in protocol["arm_order"]
        },
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_tarc_development.py run"
        ),
        "optimizer_steps_per_arm": 16,
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    artifact.write_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"]), reproduction
    )
    print(json.dumps({"route": route, "checks": checks}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    reproduction = artifact.read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if reproduction.get("protocol_sha256") != artifact.sha256(path):
        raise TarcDevelopmentValidationError("reproduction protocol hash mismatch")
    for row in reproduction["artifacts"]:
        file = resolve_repo_path(row["path"])
        if (
            not file.is_file()
            or artifact.sha256(file) != row["sha256"]
            or file.stat().st_size != row["bytes"]
        ):
            raise TarcDevelopmentValidationError(f"artifact mismatch: {row['path']}")
    manifests = {
        arm: artifact.read_json(
            resolve_repo_path(protocol["arm_outputs"][arm]) / "run_manifest.json"
        )
        for arm in protocol["arm_order"]
    }
    checks = base.compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_development_route(checks)
    decision = artifact.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if decision.get("route") != route or reproduction.get("route") != route:
        raise TarcDevelopmentValidationError("stored and recomputed routes differ")
    for arm, manifest in manifests.items():
        adapter = resolve_repo_path(manifest["final_checkpoint"]) / "adapter_model.safetensors"
        if not adapter.is_file() or artifact.sha256(adapter) != manifest["final_adapter_sha256"]:
            raise TarcDevelopmentValidationError(f"local adapter mismatch: {arm}")
    result = {
        "status": "PASS_TARC_DEVELOPMENT16_RESULT_VERIFICATION",
        "route": route,
        "arms": len(manifests),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "generation_protocol_authorized": decision["generation_protocol_authorized"],
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        TarcDevelopmentValidationError,
        base.TarcTrainingValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_TARC_DEVELOPMENT16_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
