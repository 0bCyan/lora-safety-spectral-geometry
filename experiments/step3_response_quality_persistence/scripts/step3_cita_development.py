"""Guard and package the matched CITA 16-step development stage."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import step3_cita_training as base
import step3ct_training_common as training


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cita/"
    "protocol/cita_development16_protocol.json"
)
base.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL


class CitaDevelopmentValidationError(ValueError):
    """Raised when development authorization or packaging is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def guarded_context(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol, config, packet, cache = base.context(path)
    if protocol.get("run_kind") != "development":
        raise CitaDevelopmentValidationError("protocol is not a development stage")
    if int(protocol.get("optimizer_steps_per_arm", 0)) != 16:
        raise CitaDevelopmentValidationError("development stage must use exactly 16 steps")
    smoke = base.read_json(
        base.resolve_repo_path(protocol["frozen_inputs"]["smoke_decision"]["path"])
    )
    if smoke.get("route") != "PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL":
        raise CitaDevelopmentValidationError("smoke route did not authorize development")
    return protocol, config, packet, cache


def preflight(path: Path) -> dict[str, Any]:
    guarded_context(path)
    result = base.preflight(path)
    result["status"] = "PASS_CITA_DEVELOPMENT16_PREFLIGHT"
    print(json.dumps(result, sort_keys=True))
    return result


def development_route(checks: dict[str, bool]) -> str:
    smoke_route = base.select_smoke_route(checks)
    routes = {
        "STOP_CITA_SMOKE_MATCHING_FAILED": "STOP_CITA_DEVELOPMENT16_MATCHING_FAILED",
        "STOP_CITA_SMOKE_NUMERICAL_FAILED": "STOP_CITA_DEVELOPMENT16_NUMERICAL_FAILED",
        "STOP_CITA_SMOKE_ASSIGNMENT_NOT_EXERCISED": "STOP_CITA_DEVELOPMENT16_ASSIGNMENT_NOT_EXERCISED",
        "PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL": "PASS_CITA_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE",
    }
    return routes[smoke_route]


def final_trace(output: Path) -> dict[str, Any]:
    with (output / "training_trace.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["optimizer_step"]) != 16:
        raise CitaDevelopmentValidationError("development trace lacks step 16")
    row = rows[-1]
    return {
        "risk_loss_mean": float(row["risk_loss_mean"]),
        "safety_weighted_loss_mean": float(row["safety_weighted_loss_mean"]),
        "safety_uniform_diagnostic_loss_mean": float(
            row["safety_uniform_diagnostic_loss_mean"]
        ),
        "gradient_preclip_norm": float(row["gradient_preclip_norm"]),
        "learning_rate_after_update": float(row["learning_rate_after_update"]),
    }


def package(
    path: Path,
    protocol: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    runtime_head: str,
) -> dict[str, Any]:
    checks = base.compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = development_route(checks)
    finals = {
        arm: final_trace(base.resolve_repo_path(protocol["arm_outputs"][arm]))
        for arm in protocol["arm_order"]
    }
    decision = {
        "name": "step3_cita_matched_development16_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": base.sha256(path),
        "checks": checks,
        "arms": {
            arm: {
                "optimizer_steps": manifests[arm]["optimizer_steps"],
                "final_adapter_sha256": manifests[arm]["final_adapter_sha256"],
                "vram_headroom_fraction": manifests[arm]["vram_headroom_fraction"],
                "final_trace": finals[arm],
            }
            for arm in protocol["arm_order"]
        },
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "generation_gate_authorized": route
        == "PASS_CITA_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE",
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    decision_path = base.resolve_repo_path(protocol["outputs"]["decision"])
    training.write_json(decision_path, decision)
    report = f"""# CITA Matched Development-16 Integrity Report

Route: `{route}`.

Uniform, CITA, and exact-budget sham each completed 16 optimizer steps from matched fresh seed-1 states. All registered input, exposure, scalar-budget, first-step diagnostic, numerical, and VRAM checks passed: `{all(checks.values())}`. The three final adapter hashes are distinct.

Final uniform-diagnostic safety losses were `{finals['uniform']['safety_uniform_diagnostic_loss_mean']:.6f}` (uniform), `{finals['cita']['safety_uniform_diagnostic_loss_mean']:.6f}` (CITA), and `{finals['sham']['safety_uniform_diagnostic_loss_mean']:.6f}` (sham). These teacher-forced values are diagnostics only and do not decide the method.

No candidate output was generated and no evaluator, core100, heldout set, judge, or human reviewer was opened. A pass authorizes only a separately frozen deterministic proxy-dev generation gate. Total optimizer steps across arms: `{decision['total_optimizer_steps']}`; `main_training_authorized=false`.
"""
    report_path = base.resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(report, encoding="utf-8")
    artifacts = []
    for arm in protocol["arm_order"]:
        arm_root = base.resolve_repo_path(protocol["arm_outputs"][arm])
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
        "name": "step3_cita_development16_reproduction_manifest_v1",
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
        "optimizer_steps_per_arm": 16,
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    training.write_json(
        base.resolve_repo_path(protocol["outputs"]["reproduction_manifest"]), reproduction
    )
    print(json.dumps({"route": route, "checks": checks}, sort_keys=True))
    return decision


def run(path: Path) -> dict[str, Any]:
    protocol, config, packet, cache = guarded_context(path)
    runtime_head = base.require_clean_start(protocol)
    root = base.resolve_repo_path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    manifests = {
        arm: base.train_arm(arm, protocol, config, packet, cache, runtime_head)
        for arm in protocol["arm_order"]
    }
    return package(path, protocol, manifests, runtime_head)


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = guarded_context(path)
    reproduction = base.read_json(
        base.resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if reproduction.get("protocol_sha256") != base.sha256(path):
        raise CitaDevelopmentValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = base.resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise CitaDevelopmentValidationError(
                f"artifact hash mismatch: {artifact['path']}"
            )
    manifests = {
        arm: base.read_json(
            base.resolve_repo_path(protocol["arm_outputs"][arm]) / "run_manifest.json"
        )
        for arm in protocol["arm_order"]
    }
    checks = base.compare_manifests(
        manifests, protocol, float(protocol["gates"]["matching_tolerance"])
    )
    route = development_route(checks)
    decision = base.read_json(base.resolve_repo_path(protocol["outputs"]["decision"]))
    if decision.get("route") != route or reproduction.get("route") != route:
        raise CitaDevelopmentValidationError("stored and recomputed routes differ")
    for arm, manifest in manifests.items():
        adapter = (
            base.resolve_repo_path(manifest["final_checkpoint"])
            / "adapter_model.safetensors"
        )
        if not adapter.is_file() or base.sha256(adapter) != manifest["final_adapter_sha256"]:
            raise CitaDevelopmentValidationError(f"local adapter mismatch: {arm}")
    result = {
        "status": "PASS_CITA_DEVELOPMENT16_RESULT_VERIFICATION",
        "route": route,
        "arms": len(manifests),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in manifests.values()),
        "candidate_outputs_generated": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = base.resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        CitaDevelopmentValidationError,
        base.CitaTrainingValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_CITA_DEVELOPMENT16_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
