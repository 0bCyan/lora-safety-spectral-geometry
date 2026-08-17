"""Run the zero-update MCD-E-TARC primal/dual mechanism gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_mcd_common as mcd
import step3_qetarc_common as qetarc
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_tarc_p0 as tarc_p0


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_mcd_p0/"
    "protocol/mcd_p0_protocol.json"
)
CALIBRATION_OBJECTIVES = (
    "uniform",
    "ramp",
    "tarc",
    "etarc",
    "continuation",
    "branch_entropy",
    "branch_negative_gap",
)
VALIDATION_OBJECTIVES = (
    "uniform",
    "tarc",
    "etarc",
    "continuation",
    "branch_entropy",
    "branch_negative_gap",
)
CERTIFICATES = (
    "uniform",
    "etarc",
    "continuation",
    "branch_entropy",
    "branch_negative_gap",
)
CANDIDATES = ("ramp", "tarc", "etarc", "qetarc", "mcd_etarc")


class McdP0ValidationError(ValueError):
    """Raised when MCD-E-TARC P0 violates its prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise McdP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != "prospective_before_any_MCD_ETARC_model_forward":
        raise McdP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise McdP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise McdP0ValidationError("protocol violates the zero-update boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise McdP0ValidationError("entry commit is not an ancestor of HEAD")
    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_QETARC_SMOKE_ROUTE_EFFECT_FAILED":
        raise McdP0ValidationError("registered Q-E-TARC precursor route differs")
    rows = etarc_p0.read_jsonl(resolve_repo_path(protocol["source"]["targets_path"]))
    counts = protocol["source"]["counts"]
    ramp_p0.validate_source_rows(
        rows,
        int(counts["rows"]),
        int(counts["prompt_clusters"]),
        int(counts["categories"]),
    )
    selected = etarc_p0.select_calibration_and_remainder(
        rows, int(protocol["selection"]["seed"])
    )
    for split in ("calibration", "validation"):
        if selected[split] != list(protocol["selection"][f"{split}_sample_ids"]):
            raise McdP0ValidationError(f"{split} selection differs")
    return protocol, rows


def split_pairs(
    pairs: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    by_id = {str(pair["sample_id"]): pair for pair in pairs}
    return {
        split: [by_id[value] for value in protocol["selection"][f"{split}_sample_ids"]]
        for split in ("calibration", "validation")
    }


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise McdP0ValidationError("MCD-E-TARC P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.append(relative(path))
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    for output in protocol["outputs"].values():
        if resolve_repo_path(output).exists():
            raise McdP0ValidationError(f"prospective output already exists: {output}")
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    if not torch.cuda.is_available():
        raise McdP0ValidationError("CUDA is unavailable")
    tokenizer = tarc_p0.load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    etarc_p0.verify_structural_freeze(audit, protocol)
    splits = split_pairs(pairs, protocol)
    result = {
        "status": "PASS_MCD_ETARC_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "source_rows": len(rows),
        "prompt_clusters": len(pairs),
        "calibration_prompts": len(splits["calibration"]),
        "validation_prompts": len(splits["validation"]),
        "prompt_overlap": 0,
        "certificates": list(CERTIFICATES),
        "candidate_directions": list(CANDIDATES),
        "auxiliary_gradient_norm_ratio": protocol["design"][
            "auxiliary_gradient_norm_ratio"
        ],
        "effective_method_perturbation_ratio": protocol["design"][
            "effective_method_perturbation_ratio"
        ],
        "gpu": torch.cuda.get_device_name(0),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def scaled_directions(
    calibration: dict[str, dict[str, Any]],
    ratio: float,
    quota_cosine: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    uniform_norm = gradient_math.gradient_norm(calibration["uniform"])
    target_norm = ratio * uniform_norm
    directions: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}
    errors = []
    for name in ("ramp", "tarc", "etarc"):
        raw_norm = gradient_math.gradient_norm(calibration[name])
        coefficient = target_norm / raw_norm
        direction = gradient_math.scaled_gradient(calibration[name], coefficient)
        direction_norm = gradient_math.gradient_norm(direction)
        directions[name] = direction
        records[name] = {
            "raw_gradient_norm": raw_norm,
            "coefficient": coefficient,
            "scaled_gradient_norm": direction_norm,
        }
        errors.append(abs(direction_norm - target_norm) / target_norm)
    quota_direction, quota = qetarc.quota_constrained_direction(
        calibration["etarc"],
        calibration["continuation"],
        quota_cosine,
        target_norm,
    )
    directions["qetarc"] = quota_direction
    records["qetarc"] = {
        "raw_gradient_norm": gradient_math.gradient_norm(calibration["etarc"]),
        "coefficient": None,
        "scaled_gradient_norm": gradient_math.gradient_norm(quota_direction),
    }
    errors.append(abs(records["qetarc"]["scaled_gradient_norm"] - target_norm) / target_norm)
    common_inputs = {name: calibration[name] for name in CERTIFICATES}
    common_direction, dual = mcd.minimax_common_direction(
        common_inputs, target_norm
    )
    directions["mcd_etarc"] = common_direction
    records["mcd_etarc"] = {
        "raw_gradient_norm": None,
        "coefficient": None,
        "scaled_gradient_norm": gradient_math.gradient_norm(common_direction),
    }
    errors.append(
        abs(records["mcd_etarc"]["scaled_gradient_norm"] - target_norm)
        / target_norm
    )
    budget = {
        "auxiliary_gradient_norm_ratio": ratio,
        "uniform_calibration_gradient_norm": uniform_norm,
        "target_auxiliary_gradient_norm": target_norm,
        "directions": records,
        "maximum_relative_budget_error": max(errors),
    }
    return directions, budget, quota, dual


def directional_metrics(
    validation: dict[str, dict[str, Any]],
    directions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate_directions": {}}
    for candidate, direction in directions.items():
        objective_metrics = {
            objective: {
                "dot": gradient_math.gradient_dot(gradient, direction),
                "cosine": gradient_math.gradient_cosine(gradient, direction),
            }
            for objective, gradient in validation.items()
        }
        result["candidate_directions"][candidate] = objective_metrics
    current = result["candidate_directions"]
    mcd_metrics = current["mcd_etarc"]
    mcd_metrics["summary"] = {
        "minimum_certificate_cosine": min(
            float(mcd_metrics[name]["cosine"]) for name in CERTIFICATES
        ),
        "route_dot_fraction_of_raw_etarc": (
            mcd_metrics["etarc"]["dot"] / current["etarc"]["etarc"]["dot"]
        ),
        "route_dot_fraction_of_tarc": (
            mcd_metrics["etarc"]["dot"] / current["tarc"]["etarc"]["dot"]
        ),
        "continuation_dot_fraction_of_ramp": (
            mcd_metrics["continuation"]["dot"]
            / current["ramp"]["continuation"]["dot"]
        ),
        "continuation_dot_fraction_of_qetarc": (
            mcd_metrics["continuation"]["dot"]
            / current["qetarc"]["continuation"]["dot"]
        ),
    }
    return result


def paired_summary(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    by_objective = {
        objective: [row for row in rows if row["objective"] == objective]
        for objective in VALIDATION_OBJECTIVES
    }
    result: dict[str, Any] = {"candidate_objectives": {}}
    for candidate in CANDIDATES:
        result["candidate_objectives"][candidate] = {}
        for objective in VALIDATION_OBJECTIVES:
            values = [
                float(row["directional"][candidate]["dot"])
                for row in by_objective[objective]
            ]
            result["candidate_objectives"][candidate][objective] = {
                "prompts": len(values),
                "mean_dot": statistics.fmean(values),
                "median_dot": statistics.median(values),
                "positive_rate": sum(value > 0.0 for value in values) / len(values),
            }

    def paired_difference(objective: str, left: str, right: str) -> dict[str, Any]:
        objective_rows = by_objective[objective]
        differences = [
            float(row["directional"][left]["dot"])
            - float(row["directional"][right]["dot"])
            for row in objective_rows
        ]
        return {
            "objective": objective,
            "left": left,
            "right": right,
            "differences": differences,
            "positive_rate": sum(value > 0.0 for value in differences)
            / len(differences),
            "bootstrap": etarc_p0.paired_bootstrap(
                differences,
                int(protocol["statistics"]["bootstrap_samples"]),
                int(protocol["statistics"]["bootstrap_seed"]),
            ),
            "sign_test": etarc_p0.exact_sign_test(differences),
        }

    result["mcd_continuation_advantage_over_qetarc"] = paired_difference(
        "continuation", "mcd_etarc", "qetarc"
    )
    result["mcd_route_advantage_over_tarc"] = paired_difference(
        "etarc", "mcd_etarc", "tarc"
    )
    return result


def package_artifacts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for name, value in protocol["outputs"].items():
        if name == "run_manifest":
            continue
        path = resolve_repo_path(value)
        artifacts.append(
            {
                "name": name,
                "path": relative(path),
                "sha256": etarc_p0.sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(etarc_p0.read_jsonl(path))
                if path.suffix == ".jsonl"
                else None,
            }
        )
    return artifacts


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
    tokenizer = tarc_p0.load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    all_pairs, structural = tarc_p0.build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    etarc_p0.verify_structural_freeze(structural, protocol)
    splits = split_pairs(all_pairs, protocol)
    started = time.time()
    model, model_tokenizer = model_common.load_model(protocol)
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise McdP0ValidationError("tokenizer-only and model tokenizers differ")
    del model_tokenizer, tokenizer
    parameters = model_common.trainable_parameters(model)
    initial_state = model_common.trainable_state_hash(parameters)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ramp_width = int(protocol["design"]["ramp_width"])
    temperature = float(protocol["design"]["temperature"])
    gradients: dict[str, dict[str, dict[str, Any]]] = {
        "calibration": {},
        "validation": {},
    }
    coverage: dict[str, dict[str, dict[str, Any]]] = {
        "calibration": {},
        "validation": {},
    }
    diagnostic_rows: list[dict[str, Any]] = []
    for objective in CALIBRATION_OBJECTIVES:
        gradient, details, objective_rows = etarc_p0.aggregate_gradient(
            model,
            parameters,
            splits["calibration"],
            "calibration",
            objective,
            ramp_width,
            temperature,
        )
        gradients["calibration"][objective] = gradient
        coverage["calibration"][objective] = details
        diagnostic_rows.extend(objective_rows)
    directions, budget, quota, dual = scaled_directions(
        gradients["calibration"],
        float(protocol["design"]["auxiliary_gradient_norm_ratio"]),
        float(protocol["design"]["qetarc_control_quota_cosine"]),
    )
    validation_rows: list[dict[str, Any]] = []
    for objective in VALIDATION_OBJECTIVES:
        gradient, details, objective_rows = etarc_p0.aggregate_gradient(
            model,
            parameters,
            splits["validation"],
            "validation",
            objective,
            ramp_width,
            temperature,
            directions,
        )
        gradients["validation"][objective] = gradient
        coverage["validation"][objective] = details
        validation_rows.extend(objective_rows)
        diagnostic_rows.extend(objective_rows)
    directional = directional_metrics(gradients["validation"], directions)
    paired = paired_summary(validation_rows, protocol)
    final_state = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    records = [
        coverage[split][objective]
        for split in coverage
        for objective in coverage[split]
    ]
    cfg = protocol["gates"]
    mcd_metrics = directional["candidate_directions"]["mcd_etarc"]
    summary = mcd_metrics["summary"]
    pair_mcd = paired["candidate_objectives"]["mcd_etarc"]
    continuation_advantage = paired["mcd_continuation_advantage_over_qetarc"]
    effective_ratio = float(protocol["design"]["safety_mix"]) * float(
        protocol["design"]["auxiliary_gradient_norm_ratio"]
    )
    gates = {
        "integrity": (
            structural["rows"] == int(protocol["source"]["counts"]["rows"])
            and len(splits["calibration"])
            == int(protocol["selection"]["calibration_prompts"])
            and len(splits["validation"])
            == int(protocol["selection"]["validation_prompts"])
        ),
        "gradient_valid": all(
            not row["missing_gradient_tensors"]
            and row["finite"]
            and float(row["gradient_norm"]) > 0.0
            for row in records
        ),
        "budget_matched": budget["maximum_relative_budget_error"]
        <= float(cfg["gradient_budget_relative_tolerance"]),
        "vram": headroom >= float(cfg["minimum_vram_headroom_fraction"]),
        "state_restored": initial_state == final_state,
        "dual_certificate": (
            float(dual["kkt_residual"]) <= float(cfg["dual_kkt_residual_max"])
            and float(dual["primal_dual_gap"])
            <= float(cfg["primal_dual_gap_max"])
            and float(dual["dual_margin"])
            >= float(cfg["calibration_common_margin_min"])
            and min(float(value) for value in dual["certificate_cosines"].values())
            + float(cfg["dual_numeric_tolerance"])
            >= float(dual["dual_margin"])
        ),
        "common_transfer": (
            float(summary["minimum_certificate_cosine"])
            >= float(cfg["validation_minimum_certificate_cosine_min"])
            and all(
                float(pair_mcd[objective]["positive_rate"])
                >= float(cfg["validation_certificate_positive_pair_rate_min"])
                for objective in CERTIFICATES
            )
        ),
        "route_effect": (
            float(summary["route_dot_fraction_of_raw_etarc"])
            >= float(cfg["route_dot_fraction_of_raw_etarc_min"])
            and float(summary["route_dot_fraction_of_tarc"])
            >= float(cfg["route_dot_fraction_of_tarc_min"])
            and float(pair_mcd["etarc"]["positive_rate"])
            >= float(cfg["route_positive_pair_rate_min"])
        ),
        "continuation_effect": (
            float(summary["continuation_dot_fraction_of_ramp"])
            >= float(cfg["continuation_dot_fraction_of_ramp_min"])
            and float(summary["continuation_dot_fraction_of_qetarc"])
            >= float(cfg["continuation_dot_fraction_of_qetarc_min"])
            and float(continuation_advantage["bootstrap"]["ci95_lower"])
            > float(cfg["paired_advantage_ci_lower_min"])
            and float(continuation_advantage["sign_test"]["exact_two_sided_p"])
            <= float(cfg["paired_sign_p_max"])
        ),
        "branch_effect": (
            float(mcd_metrics["branch_entropy"]["dot"])
            > float(cfg["branch_directional_dot_min"])
            and float(mcd_metrics["branch_negative_gap"]["dot"])
            > float(cfg["branch_directional_dot_min"])
            and float(pair_mcd["branch_entropy"]["positive_rate"])
            >= float(cfg["branch_positive_pair_rate_min"])
            and float(pair_mcd["branch_negative_gap"]["positive_rate"])
            >= float(cfg["branch_positive_pair_rate_min"])
        ),
        "effect_not_weak": abs(
            effective_ratio
            - float(protocol["design"]["effective_method_perturbation_ratio"])
        )
        <= float(cfg["effective_ratio_absolute_tolerance"]),
    }
    route = mcd.select_route(gates)
    outputs = {name: resolve_repo_path(value) for name, value in protocol["outputs"].items()}
    split_manifest = {
        "name": "step3_mcd_etarc_p0_split_manifest_v1",
        "seed": protocol["selection"]["seed"],
        "calibration_sample_ids": protocol["selection"]["calibration_sample_ids"],
        "validation_sample_ids": protocol["selection"]["validation_sample_ids"],
        "overlap": [],
        "fresh_for_confirmation": False,
    }
    diagnostics = {
        "name": "step3_mcd_etarc_p0_gradient_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "coverage": coverage,
        "budget": budget,
        "qetarc_control_geometry": quota,
        "mcd_dual_certificate": dual,
        "directional": directional,
        "paired_validation": paired,
        "effective_method_perturbation_ratio": effective_ratio,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "gates": gates,
        "optimizer_objects": 0,
        "optimizer_steps": 0,
    }
    environment = {
        "name": "step3_mcd_etarc_p0_environment_v1",
        "runtime_head": runtime_head,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_mcd_etarc_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "mcd_dual_certificate": dual,
        "directional_summary": summary,
        "paired_validation": paired,
        "matched_smoke_protocol_authorized": route
        == "PASS_MCD_ETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    report = f"""# MCD-E-TARC P0 Report

Route: `{route}`.

The five-certificate calibration common margin was
`{dual['dual_margin']:.6f}` with active certificates
`{', '.join(dual['active_certificates'])}`. The KKT residual was
`{dual['kkt_residual']:.3e}` and the primal-dual gap was
`{dual['primal_dual_gap']:.3e}`.

On 32 prompt-disjoint development-validation clusters, the minimum aggregate
certificate cosine was `{summary['minimum_certificate_cosine']:.6f}`.
MCD-E-TARC retained `{summary['route_dot_fraction_of_raw_etarc']:.3%}` of raw
E-TARC route dot and reached `{summary['route_dot_fraction_of_tarc']:.3f}x`
TARC route dot. Its continuation dot was
`{summary['continuation_dot_fraction_of_ramp']:.3%}` of RAMP and
`{summary['continuation_dot_fraction_of_qetarc']:.3f}x` Q-E-TARC.

This is a zero-update mechanism-development result on a reused disclosed
split. `optimizer_steps=0`, `training_authorized=false`, and no candidate,
Core100, or heldout data was opened.
"""
    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_json_frozen(outputs["split_manifest"], split_manifest)
    etarc_p0.write_jsonl_frozen(outputs["diagnostic_rows"], diagnostic_rows)
    etarc_p0.write_json_frozen(outputs["gradient_diagnostics"], diagnostics)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_mcd_etarc_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_mcd_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_mcd_p0.py verify-run"
        ),
        "optimizer_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "artifacts": package_artifacts(protocol),
    }
    etarc_p0.write_json_frozen(outputs["run_manifest"], manifest)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _ = context(path)
    manifest_path = resolve_repo_path(protocol["outputs"]["run_manifest"])
    manifest = etarc_p0.read_json(manifest_path)
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["route"]))
    expected_route = mcd.select_route(decision["gates"])
    if decision.get("route") != expected_route or manifest.get("route") != expected_route:
        raise McdP0ValidationError("stored and recomputed routes differ")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise McdP0ValidationError("P0 artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise McdP0ValidationError("stored P0 boundary differs")
    result = {
        "status": "PASS_MCD_ETARC_P0_VERIFY",
        "route": expected_route,
        "artifacts": len(manifest["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    if args.command == "preflight":
        preflight(path)
    elif args.command == "run":
        run(path)
    else:
        verify_run(path)


if __name__ == "__main__":
    main()
