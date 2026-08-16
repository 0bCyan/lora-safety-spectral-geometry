"""Run and verify the separately authorized RAMP development-16 stage."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import step3_ramp_training as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_ramp/"
    "protocol/ramp_development16_protocol.json"
)


class RampDevelopmentValidationError(ValueError):
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
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = base.read_json(path)
    if protocol.get("status") != "prospective_before_any_matched_RAMP_stage_training":
        raise RampDevelopmentValidationError("training protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or base.sha256(artifact_path) != artifact["sha256"]
        ):
            raise RampDevelopmentValidationError(
                f"frozen input hash mismatch: {name}"
            )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RampDevelopmentValidationError("entry commit is not an ancestor of HEAD")

    if protocol.get("run_kind") != "development":
        raise RampDevelopmentValidationError("protocol is not development")
    config = base.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        config.get("development_training_authorized") is not True
        or protocol.get("training_authorized") is not True
    ):
        raise RampDevelopmentValidationError("development training is not authorized")
    if (
        config.get("main_training_authorized") is not False
        or config.get("candidate_generation_authorized") is not False
        or config.get("core100_opened") is not False
        or config.get("heldout_opened") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("main_training_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise RampDevelopmentValidationError("protocol violates the sealed boundary")
    if (
        int(protocol["optimizer_steps_per_arm"]) != 16
        or int(config["development_steps"]) != 16
    ):
        raise RampDevelopmentValidationError("development must use 16 steps per arm")
    arms = list(protocol["arm_order"])
    if arms != ["uniform", "suffix", "ramp"] or set(config["arms"]) != set(arms):
        raise RampDevelopmentValidationError("matched arm registry differs")

    p0_route_path = resolve_repo_path(config["p0_route"])
    if base.sha256(p0_route_path) != config["p0_route_sha256"]:
        raise RampDevelopmentValidationError("P0 route hash differs from config")
    p0_route = base.read_json(p0_route_path)
    if p0_route.get("route") != "PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL":
        raise RampDevelopmentValidationError("RAMP-P0 did not authorize matched smoke")
    for key in ("prefix_coefficient", "suffix_coefficient"):
        if not math.isclose(
            float(config[key]),
            float(p0_route["coefficients"][key]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RampDevelopmentValidationError(f"frozen P0 coefficient drift: {key}")

    targets_path = resolve_repo_path(config["safety_targets_path"])
    if base.sha256(targets_path) != config["safety_targets_sha256"]:
        raise RampDevelopmentValidationError("safety target hash differs from config")
    targets = base.read_jsonl(targets_path)
    base.ramp_p0.validate_source_rows(targets, 80, 40, 8)
    by_id = {str(row["target_id"]): row for row in targets}
    diagnostic_ids = list(config["diagnostic_target_ids"])
    if len(diagnostic_ids) != 16 or len(set(diagnostic_ids)) != 16:
        raise RampDevelopmentValidationError("diagnostic target IDs are invalid")
    try:
        diagnostics = [by_id[target_id] for target_id in diagnostic_ids]
    except KeyError as exc:
        raise RampDevelopmentValidationError("diagnostic target is absent") from exc
    p0_protocol = base.read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p0_protocol"]["path"])
    )
    if diagnostic_ids != p0_protocol["selection"]["validation_target_ids"]:
        raise RampDevelopmentValidationError(
            "diagnostic IDs differ from RAMP-P0 validation"
        )

    smoke = base.read_json(
        resolve_repo_path(protocol["frozen_inputs"]["smoke_decision"]["path"])
    )
    if smoke.get("route") != "PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL":
        raise RampDevelopmentValidationError("smoke did not authorize development-16")
    smoke_reproduction = base.read_json(
        resolve_repo_path(
            protocol["frozen_inputs"]["smoke_reproduction_manifest"]["path"]
        )
    )
    if (
        smoke_reproduction.get("route")
        != "PASS_RAMP_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL"
    ):
        raise RampDevelopmentValidationError("smoke reproduction route differs")

    draws = int(protocol["optimizer_steps_per_arm"]) * int(
        config["gradient_accumulation_steps"]
    )
    if draws != 64 or int(protocol["safety_draws_per_arm"]) != draws:
        raise RampDevelopmentValidationError("development safety draw count differs")
    if int(protocol["diagnostic_rows"]) != len(diagnostics):
        raise RampDevelopmentValidationError("development diagnostic count differs")
    train_prompt_ids = {str(row["sample_id"]) for row in targets[:draws]}
    diagnostic_prompt_ids = {str(row["sample_id"]) for row in diagnostics}
    overlap = len(train_prompt_ids.intersection(diagnostic_prompt_ids))
    if overlap != int(protocol["diagnostic_prompt_overlap_with_development_draws"]):
        raise RampDevelopmentValidationError("development diagnostic overlap differs")
    return protocol, config, targets, diagnostics


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, diagnostics = context(path)
    if not torch.cuda.is_available():
        raise RampDevelopmentValidationError("CUDA is unavailable")
    result = {
        "status": "PASS_RAMP_DEVELOPMENT16_PREFLIGHT",
        "protocol_sha256": base.sha256(path),
        "arms": protocol["arm_order"],
        "optimizer_steps_per_arm": 16,
        "total_authorized_optimizer_steps": 48,
        "safety_draws_per_arm": 64,
        "safety_target_rows": len(targets),
        "diagnostic_rows": len(diagnostics),
        "diagnostics_are_teacher_forced_descriptive": True,
        "diagnostic_overlap_allowed_for_development": True,
        "diagnostic_prompt_overlap_with_development_draws": protocol[
            "diagnostic_prompt_overlap_with_development_draws"
        ],
        "gpu": torch.cuda.get_device_name(0),
        "candidate_generation_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def select_development_route(checks: dict[str, Any]) -> str:
    matching = (
        "all_arms_present",
        "initial_state_equal",
        "risk_order_equal",
        "target_order_equal",
        "target_exposure_equal",
        "safety_draws_equal",
        "safety_response_tokens_equal",
        "available_prefix_terms_equal",
        "available_suffix_terms_equal",
        "first_step_risk_loss_equal",
        "first_step_uniform_safety_loss_equal",
        "initial_uniform_diagnostic_equal",
        "initial_prefix_diagnostic_equal",
        "initial_suffix_diagnostic_equal",
        "sealed_data_untouched",
    )
    numerical = ("steps_exact", "no_skipped_updates", "finite", "vram")
    if not all(checks[name] for name in matching):
        return "STOP_RAMP_DEVELOPMENT16_MATCHING_FAILED"
    if not all(checks[name] for name in numerical):
        return "STOP_RAMP_DEVELOPMENT16_NUMERICAL_FAILED"
    if not (
        checks["assignment_counts_correct"]
        and checks["all_registered_assignments_exercised"]
    ):
        return "STOP_RAMP_DEVELOPMENT16_ASSIGNMENT_NOT_EXERCISED"
    return "PASS_RAMP_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE"


def run(path: Path) -> dict[str, Any]:
    protocol, config, targets, diagnostics = context(path)
    runtime_head = base.require_clean_start(path, protocol)
    root = resolve_repo_path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for arm in protocol["arm_order"]:
        manifests[arm] = base.train_arm(
            str(arm), protocol, config, targets, diagnostics, runtime_head
        )
    checks = base.compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_development_route(checks)
    decision = {
        "name": "step3_ramp_matched_development16_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": base.sha256(path),
        "checks": checks,
        "arms": {
            arm: {
                "objective": manifest["objective"],
                "completed_steps": manifest["completed_steps"],
                "optimizer_steps": manifest["optimizer_steps"],
                "final_adapter_sha256": manifest["final_adapter_sha256"],
                "final_checkpoint": manifest["final_checkpoint"],
                "prefix_loss_reduction": manifest["prefix_loss_reduction"],
                "final_diagnostic": manifest["final_diagnostic"],
                "vram_headroom_fraction": manifest["vram_headroom_fraction"],
            }
            for arm, manifest in manifests.items()
        },
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "generation_protocol_authorized": route
        == "PASS_RAMP_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE",
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "next_authorized": (
            "freeze_and_push_separate_three_arm_generation_effect_gate"
            if route == "PASS_RAMP_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE"
            else "stop_RAMP_training"
        ),
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    base.write_json(decision_path, decision)
    report = f"""# RAMP Matched Development-16 Integrity Report

Route: `{route}`.

Uniform, suffix placebo, and RAMP each completed 16 matched optimizer steps.
All registered input, exposure, first-step, numerical, assignment, VRAM, and
sealed-data integrity checks passed: `{route == 'PASS_RAMP_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE'}`.

Teacher-forced prefix-loss reductions were
`{checks['uniform_prefix_loss_reduction']:.6f}` (uniform),
`{checks['suffix_prefix_loss_reduction']:.6f}` (suffix), and
`{checks['ramp_prefix_loss_reduction']:.6f}` (RAMP). These diagnostics did not
enter the route and cannot establish generated behavior.

No candidate output, Core100, heldout, or reviewer data was opened. Total
optimizer steps: `{decision['total_optimizer_steps']}`;
`main_training_authorized=false`.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report.encode("utf-8"))
    artifacts = []
    for arm in protocol["arm_order"]:
        arm_root = resolve_repo_path(protocol["arm_outputs"][arm])
        for name in ("training_trace.csv", "checkpoint_index.csv", "run_manifest.json"):
            artifact = arm_root / name
            artifacts.append(
                {
                    "path": base.relative(artifact),
                    "sha256": base.sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )
    for artifact in (decision_path, report_path):
        artifacts.append(
            {
                "path": base.relative(artifact),
                "sha256": base.sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        )
    reproduction = {
        "name": "step3_ramp_development16_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": base.relative(path),
        "protocol_sha256": base.sha256(path),
        "artifacts": artifacts,
        "local_ignored_checkpoints": {
            arm: manifests[arm]["final_checkpoint"] for arm in protocol["arm_order"]
        },
        "final_adapter_sha256": {
            arm: manifests[arm]["final_adapter_sha256"] for arm in protocol["arm_order"]
        },
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_ramp_development.py run"
        ),
        "optimizer_steps_per_arm": 16,
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    base.write_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"]), reproduction
    )
    print(json.dumps({"route": route, "checks": checks}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    reproduction = base.read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if reproduction.get("protocol_sha256") != base.sha256(path):
        raise RampDevelopmentValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or base.sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise RampDevelopmentValidationError(f"artifact mismatch: {artifact['path']}")
    manifests = {
        arm: base.read_json(
            resolve_repo_path(protocol["arm_outputs"][arm]) / "run_manifest.json"
        )
        for arm in protocol["arm_order"]
    }
    checks = base.compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = select_development_route(checks)
    decision = base.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if decision.get("route") != route or reproduction.get("route") != route:
        raise RampDevelopmentValidationError("stored and recomputed routes differ")
    for arm, manifest in manifests.items():
        adapter = resolve_repo_path(manifest["final_checkpoint"]) / "adapter_model.safetensors"
        if not adapter.is_file() or base.sha256(adapter) != manifest["final_adapter_sha256"]:
            raise RampDevelopmentValidationError(f"local adapter mismatch: {arm}")
    result = {
        "status": "PASS_RAMP_DEVELOPMENT16_RESULT_VERIFICATION",
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
        RampDevelopmentValidationError,
        base.RampTrainingValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_RAMP_DEVELOPMENT16_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
