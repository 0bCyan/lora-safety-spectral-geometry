"""Prospective fixed-dose, zero-step confirmation on the M5 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v4 as v4


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m5/gpu_confirmation/protocol/"
    "cdro_opc_m5_confirmation_protocol.json"
)
SPLITS = ("calibration", "sealed_audit")


class M5ConfirmationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M5ConfirmationError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_units(
    selected_targets: Sequence[Mapping[str, Any]],
    primary_candidates: Sequence[Mapping[str, Any]],
    cohort: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    units = core.build_unit_registry(
        selected_targets, primary_candidates, cohort, risk_rows
    )
    for unit in units:
        prefix = "cal" if unit["split"] == "calibration" else "audit"
        unit["unit_id"] = f"m5_{prefix}_{int(unit['rank']):02d}"
    return units


def public_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return core.registry_public_projection(units)


def context(path: Path) -> tuple[Any, ...]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_locked_before_M5_model_forward":
        raise M5ConfirmationError("protocol status differs")
    if any(
        (
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise M5ConfirmationError("zero-step authorization boundary differs")
    for name, spec in protocol["frozen_inputs"].items():
        source = repo_path(str(spec["path"]))
        if not source.is_file() or file_sha256(source) != spec["sha256"]:
            raise M5ConfirmationError(f"frozen input mismatch: {name}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    ).returncode:
        raise M5ConfirmationError("entry commit is not an ancestor")

    m1_path = repo_path(protocol["frozen_inputs"]["m1_protocol"]["path"])
    m1_protocol, config, old_targets, old_risk = core.m1.context(m1_path)
    selected = read_jsonl(
        repo_path(protocol["frozen_inputs"]["selected_targets"]["path"])
    )
    candidates = read_jsonl(
        repo_path(protocol["frozen_inputs"]["primary_candidates"]["path"])
    )
    cohort = read_jsonl(repo_path(protocol["frozen_inputs"]["cohort"]["path"]))
    risk = read_jsonl(repo_path(protocol["frozen_inputs"]["risk_source"]["path"]))
    units = build_units(selected, candidates, cohort, risk)
    if canonical_sha256(public_units(units)) != protocol["selection"]["unit_registry_sha256"]:
        raise M5ConfirmationError("unit registry differs")
    return (
        protocol,
        m1_protocol,
        config,
        old_targets,
        old_risk,
        selected,
        candidates,
        risk,
        units,
    )


def structural_audit(
    protocol: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    _, audit = core.tarc.build_pair_features(
        tokenizer,
        core.subset_targets(selected, units, split),
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    if audit != protocol["selection"]["structural_audits"][split]:
        raise M5ConfirmationError(f"{split} structural audit differs")
    return audit


def calibration_authorizes_audit(gates: Mapping[str, bool]) -> bool:
    required = {
        "fixed_direction",
        "solver_kkt",
        "protected_coverage",
        "aggregate_protection",
        "behavior",
        "structural",
        "state_restored",
        "zero_step",
    }
    return set(gates) == required and all(gates.values())


def require_clean_start(protocol: Mapping[str, Any]) -> str:
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise M5ConfirmationError("run requires a clean worktree")
    output = repo_path(protocol["output_root"])
    if output.exists() and any(item.name != "protocol" for item in output.iterdir()):
        raise M5ConfirmationError("M5 confirmation output already exists")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def preflight(path: Path) -> dict[str, Any]:
    import torch

    (
        protocol,
        _,
        config,
        _,
        _,
        selected,
        candidates,
        risk,
        units,
    ) = context(path)
    if not torch.cuda.is_available():
        raise M5ConfirmationError("CUDA unavailable")
    tokenizer = core.tarc.load_tokenizer(str(config["model_name_or_path"]))
    audits = {
        split: structural_audit(protocol, selected, units, tokenizer, config, split)
        for split in SPLITS
    }
    behavior = {
        split: {
            pool: len(rows)
            for pool, rows in core.behavior_rows(
                selected, candidates, units, risk, split
            ).items()
        }
        for split in SPLITS
    }
    result = {
        "status": "PASS_CDRO_OPC_M5_CONFIRMATION_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "fixed_dose": protocol["design"]["fixed_dose"],
        "calibration_units": 20,
        "sealed_audit_units": 20,
        "targets": 80,
        "behavior_pairs": behavior,
        "structural_audits": audits,
        "audit_is_conditionally_opened": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def evaluate_split(
    protocol: Mapping[str, Any],
    m1_protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    risk: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    split: str,
    model: Any,
    tokenizer: Any,
    parameters: Sequence[tuple[str, Any]],
    fallback: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    reference: float,
    initial_hash: str,
) -> dict[str, Any]:
    audit = structural_audit(protocol, selected, units, tokenizer, config, split)
    split_targets = core.subset_targets(selected, units, split)
    fallback_scores, endpoint_scores, coverage, _ = core.protected_score_panel(
        model,
        parameters,
        tokenizer,
        split_targets,
        units,
        risk,
        config,
        split,
        fallback,
        endpoint,
        reference,
        audit,
    )
    design = protocol["design"]
    split_units = [unit for unit in units if unit["split"] == split]
    statistics = core.panel_statistics(
        fallback_scores,
        endpoint_scores,
        [str(unit["unit_id"]) for unit in split_units],
        float(design["fixed_dose"]),
        float(design["allowed_relative_drop"]),
        float(design["score_scale_floor"]),
        float(protocol["gates"]["arithmetic_tolerance"]),
    )
    selected_map = core.base.blend_maps(
        fallback, endpoint, float(design["fixed_dose"])
    )
    behavior = core.behavior_panel(
        model,
        parameters,
        tokenizer,
        core.behavior_rows(selected, candidates, units, risk, split),
        config,
        float(m1_protocol["design"]["beta"]),
        fallback,
        endpoint,
        selected_map,
    )
    structural = all(
        bool(item["all_gradients_finite_nonzero"])
        and all(
            math.isfinite(float(value)) and float(value) > 0
            for value in item["gradient_norms"].values()
        )
        for item in coverage
    ) and all(
        behavior[pool]["coverage"]["finite_nonzero"]
        and not behavior[pool]["coverage"]["missing_gradient_tensors"]
        for pool in core.POOLS
    )
    return {
        "split": split,
        "structural_audit": audit,
        "protected_coverage": coverage,
        "protected_statistics": statistics,
        "behavior": behavior,
        "structural": structural,
        "state_restored": core.cspr.trainable_state_hash(parameters) == initial_hash,
    }


def run(path: Path) -> dict[str, Any]:
    import torch

    (
        protocol,
        m1_protocol,
        config,
        old_targets,
        old_risk,
        selected,
        candidates,
        risk,
        units,
    ) = context(path)
    head = require_clean_start(protocol)
    if not torch.cuda.is_available():
        raise M5ConfirmationError("CUDA unavailable")
    start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = core.cspr.load_model(m1_protocol)
    parameters = core.cspr.trainable_parameters(model)
    optimizer = core.frozen.make_optimizer(model, config)
    initial_hash = core.cspr.trainable_state_hash(parameters)
    fallback, endpoint, reference, replay = v4.reconstruct_fixed_direction(
        m1_protocol,
        config,
        old_targets,
        old_risk,
        model,
        tokenizer,
        parameters,
        optimizer,
        protocol["fixed_direction"],
    )
    if not replay["fixed_direction_valid"]:
        raise M5ConfirmationError("fixed direction replay failed")
    core.frsc.normalized_certificates = v4.normalize_registered_certificates
    solver = replay["solver_audit"]
    zero_step = all(
        int(item["optimizer_completed_steps"]) == 0
        for item in replay["anchor_records"].values()
    ) and all(
        int(item["completed_optimizer_steps"]) == 0
        for item in replay["behavior_map_audits"].values()
    )

    calibration = evaluate_split(
        protocol,
        m1_protocol,
        config,
        selected,
        candidates,
        risk,
        units,
        "calibration",
        model,
        tokenizer,
        parameters,
        fallback,
        endpoint,
        reference,
        initial_hash,
    )
    cal_stats = calibration["protected_statistics"]
    cal_behavior = calibration["behavior"]
    common_gates = {
        "fixed_direction": bool(replay["fixed_direction_valid"]),
        "solver_kkt": bool(
            solver["optimization_valid"]
            and float(solver["primal_violation"])
            <= float(m1_protocol["gates"]["feasibility_tolerance"])
            and float(solver["normalized_stationarity_residual"])
            <= float(m1_protocol["gates"]["kkt_tolerance"])
        ),
        "zero_step": zero_step,
    }
    calibration_gates = {
        "fixed_direction": common_gates["fixed_direction"],
        "solver_kkt": common_gates["solver_kkt"],
        "protected_coverage": cal_stats["covered_units"]
        >= int(protocol["gates"]["calibration_required_covered_units"]),
        "aggregate_protection": bool(cal_stats["aggregate_protected"]),
        "behavior": all(
            cal_behavior[pool]["selected_minus_fallback"]
            > float(protocol["gates"]["minimum_behavior_improvement"])
            for pool in core.POOLS
        ),
        "structural": bool(calibration["structural"]),
        "state_restored": bool(calibration["state_restored"]),
        "zero_step": common_gates["zero_step"],
    }
    audit_authorized = calibration_authorizes_audit(calibration_gates)
    sealed = None
    sealed_gates = None
    if audit_authorized:
        sealed = evaluate_split(
            protocol,
            m1_protocol,
            config,
            selected,
            candidates,
            risk,
            units,
            "sealed_audit",
            model,
            tokenizer,
            parameters,
            fallback,
            endpoint,
            reference,
            initial_hash,
        )
        audit_stats = sealed["protected_statistics"]
        audit_behavior = sealed["behavior"]
        sealed_gates = {
            "fixed_direction": common_gates["fixed_direction"],
            "solver_kkt": common_gates["solver_kkt"],
            "protected_coverage": audit_stats["covered_units"]
            >= int(protocol["gates"]["sealed_audit_required_covered_units"]),
            "aggregate_protection": bool(audit_stats["aggregate_protected"]),
            "behavior": all(
                audit_behavior[pool]["selected_minus_fallback"]
                > float(protocol["gates"]["minimum_behavior_improvement"])
                for pool in core.POOLS
            ),
            "structural": bool(sealed["structural"]),
            "state_restored": bool(sealed["state_restored"]),
            "zero_step": common_gates["zero_step"],
        }
    if not audit_authorized:
        route = "STOP_CDRO_OPC_M5_CALIBRATION_GATE"
    elif sealed_gates is not None and all(sealed_gates.values()):
        route = "PASS_CDRO_OPC_M5_ZERO_STEP_CONFIRMATION"
    else:
        route = "STOP_CDRO_OPC_M5_SEALED_AUDIT_GATE"

    root = repo_path(protocol["output_root"]) / "run"
    diagnostics = {
        "schema": "step3_cdro_opc_m5_confirmation_diagnostics_v1",
        "route": route,
        "runtime_head": head,
        "unit_registry_sha256": canonical_sha256(public_units(units)),
        "fixed_dose": protocol["design"]["fixed_dose"],
        "fixed_direction_replay": replay,
        "calibration": calibration,
        "calibration_gates": calibration_gates,
        "sealed_audit_opened": audit_authorized,
        "sealed_audit": sealed,
        "sealed_audit_gates": sealed_gates,
        "final_state_restored": core.cspr.trainable_state_hash(parameters)
        == initial_hash,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    decision = {
        "schema": "step3_cdro_opc_m5_confirmation_decision_v1",
        "route": route,
        "fixed_dose": protocol["design"]["fixed_dose"],
        "calibration_covered_units": cal_stats["covered_units"],
        "calibration_total_units": cal_stats["total_units"],
        "calibration_gates": calibration_gates,
        "sealed_audit_opened": audit_authorized,
        "sealed_audit_covered_units": (
            sealed["protected_statistics"]["covered_units"] if sealed else None
        ),
        "sealed_audit_total_units": (
            sealed["protected_statistics"]["total_units"] if sealed else None
        ),
        "sealed_audit_gates": sealed_gates,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "interpret zero-step confirmation; do not train"
            if route.startswith("PASS")
            else "stop and interpret failed gate; do not train"
        ),
    }
    environment = {
        "head": head,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "wall_seconds": time.time() - start,
    }
    write_json(root / "cdro_opc_m5_confirmation_diagnostics.json", diagnostics)
    write_json(root / "cdro_opc_m5_confirmation_decision.json", decision)
    write_json(root / "cdro_opc_m5_confirmation_environment.json", environment)
    root.mkdir(parents=True, exist_ok=True)
    (root / "CDRO_OPC_M5_CONFIRMATION_REPORT.md").write_text(
        "# CDR-OPC M5 Fixed-Dose Zero-Step Confirmation\n\n"
        f"Route: `{route}`. Fixed dose: `{protocol['design']['fixed_dose']}`. "
        f"Calibration coverage: `{cal_stats['covered_units']}/{cal_stats['total_units']}`. "
        f"Sealed audit opened: `{str(audit_authorized).lower()}`.\n\n"
        "No optimizer step, training, generation, Core100, or heldout evaluation was authorized.\n",
        encoding="utf-8",
    )
    result = {
        "route": route,
        "fixed_dose": protocol["design"]["fixed_dose"],
        "calibration_covered_units": cal_stats["covered_units"],
        "sealed_audit_opened": audit_authorized,
        "sealed_audit_covered_units": (
            sealed["protected_statistics"]["covered_units"] if sealed else None
        ),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return decision


def verify(path: Path) -> dict[str, Any]:
    protocol, *rest = context(path)
    units = rest[-1]
    root = repo_path(protocol["output_root"]) / "run"
    diagnostics = read_json(root / "cdro_opc_m5_confirmation_diagnostics.json")
    decision = read_json(root / "cdro_opc_m5_confirmation_decision.json")
    if diagnostics["unit_registry_sha256"] != canonical_sha256(public_units(units)):
        raise M5ConfirmationError("unit registry does not reproduce")
    if abs(float(decision["fixed_dose"]) - float(protocol["design"]["fixed_dose"])) > 1e-15:
        raise M5ConfirmationError("fixed dose differs")
    if decision["optimizer_steps"] != 0 or decision["training_authorized"] is not False:
        raise M5ConfirmationError("zero-step boundary differs")
    if decision["sealed_audit_opened"] != calibration_authorizes_audit(
        decision["calibration_gates"]
    ):
        raise M5ConfirmationError("sealed-audit authorization does not reproduce")
    expected_route = (
        "STOP_CDRO_OPC_M5_CALIBRATION_GATE"
        if not decision["sealed_audit_opened"]
        else (
            "PASS_CDRO_OPC_M5_ZERO_STEP_CONFIRMATION"
            if all(decision["sealed_audit_gates"].values())
            else "STOP_CDRO_OPC_M5_SEALED_AUDIT_GATE"
        )
    )
    if decision["route"] != expected_route:
        raise M5ConfirmationError("decision route does not reproduce")
    result = {
        "status": "PASS_CDRO_OPC_M5_CONFIRMATION_VERIFY_RUN",
        "route": decision["route"],
        "sealed_audit_opened": decision["sealed_audit_opened"],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = repo_path(args.protocol)
    {"preflight": preflight, "run": run, "verify-run": verify}[args.command](path)


if __name__ == "__main__":
    main()
