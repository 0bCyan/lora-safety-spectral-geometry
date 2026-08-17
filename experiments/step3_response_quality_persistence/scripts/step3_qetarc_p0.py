"""Run the zero-update positive-continuation-quota E-TARC gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_qetarc_common as qetarc
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_tarc_p0 as tarc_p0


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_qetarc_p0/"
    "protocol/qetarc_p0_protocol.json"
)


class QetarcP0ValidationError(ValueError):
    """Raised when Q-E-TARC P0 violates its prospective protocol."""


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
        raise QetarcP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_QETARC_model_forward_or_gradient":
        raise QetarcP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise QetarcP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise QetarcP0ValidationError("protocol violates the zero-update boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise QetarcP0ValidationError("entry commit is not an ancestor of HEAD")
    prior = read_json(resolve_repo_path(protocol["prior_result"]["decision_path"]))
    if prior.get("route") != "STOP_ETARC_P0_CONTINUATION_COMPATIBILITY_FAILED":
        raise QetarcP0ValidationError("registered CC-E-TARC precursor route differs")
    rows = read_jsonl(resolve_repo_path(protocol["source"]["targets_path"]))
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
            raise QetarcP0ValidationError(f"{split} selection differs")
    return protocol, rows


def split_pairs(
    pairs: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    by_id = {str(pair["sample_id"]): pair for pair in pairs}
    return {
        split: [by_id[value] for value in protocol["selection"][f"{split}_sample_ids"]]
        for split in ("calibration", "validation")
    }


def verify_structural_freeze(audit: dict[str, Any], protocol: dict[str, Any]) -> None:
    etarc_p0.verify_structural_freeze(audit, protocol)


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise QetarcP0ValidationError("Q-E-TARC P0 requires a clean worktree")
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
            raise QetarcP0ValidationError(f"prospective output already exists: {output}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    if not torch.cuda.is_available():
        raise QetarcP0ValidationError("CUDA is unavailable")
    tokenizer = tarc_p0.load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    verify_structural_freeze(audit, protocol)
    splits = split_pairs(pairs, protocol)
    result = {
        "status": "PASS_QETARC_P0_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "source_rows": len(rows),
        "prompt_clusters": len(pairs),
        "calibration_prompts": len(splits["calibration"]),
        "validation_prompts": len(splits["validation"]),
        "prompt_overlap": 0,
        "gpu": torch.cuda.get_device_name(0),
        "quota_cosine": protocol["design"]["continuation_quota_cosine"],
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
    calibration: dict[str, dict[str, Any]], ratio: float, quota_cosine: float
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    uniform_norm = gradient_math.gradient_norm(calibration["uniform"])
    target_norm = ratio * uniform_norm
    directions = {}
    records = {}
    errors = []
    for name in ("ramp", "tarc", "etarc"):
        raw_norm = gradient_math.gradient_norm(calibration[name])
        coefficient = gradient_math.calibrated_coefficient(
            uniform_norm, raw_norm, ratio
        )
        direction = gradient_math.scaled_gradient(calibration[name], coefficient)
        direction_norm = gradient_math.gradient_norm(direction)
        directions[name] = direction
        records[name] = {
            "raw_gradient_norm": raw_norm,
            "coefficient": coefficient,
            "scaled_gradient_norm": direction_norm,
        }
        errors.append(abs(direction_norm - target_norm) / target_norm)
    quota_direction, geometry = qetarc.quota_constrained_direction(
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
    return directions, {
        "auxiliary_gradient_norm_ratio": ratio,
        "uniform_calibration_gradient_norm": uniform_norm,
        "target_auxiliary_gradient_norm": target_norm,
        "directions": records,
        "maximum_relative_budget_error": max(errors),
    }, geometry


def directional_metrics(
    calibration: dict[str, dict[str, Any]],
    validation: dict[str, dict[str, Any]],
    directions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "calibration_validation_etarc_cosine": gradient_math.gradient_cosine(
            calibration["etarc"], validation["etarc"]
        ),
        "candidate_directions": {},
    }
    for name, direction in directions.items():
        result["candidate_directions"][name] = {
            "validation_etarc_dot": gradient_math.gradient_dot(
                validation["etarc"], direction
            ),
            "validation_etarc_cosine": gradient_math.gradient_cosine(
                validation["etarc"], direction
            ),
            "validation_tarc_dot": gradient_math.gradient_dot(
                validation["tarc"], direction
            ),
            "validation_continuation_dot": gradient_math.gradient_dot(
                validation["continuation"], direction
            ),
            "validation_continuation_cosine": gradient_math.gradient_cosine(
                validation["continuation"], direction
            ),
            "validation_branch_entropy_dot": gradient_math.gradient_dot(
                validation["branch_entropy"], direction
            ),
            "validation_branch_negative_gap_dot": gradient_math.gradient_dot(
                validation["branch_negative_gap"], direction
            ),
        }
    quota = result["candidate_directions"]["qetarc"]
    raw = result["candidate_directions"]["etarc"]
    old = result["candidate_directions"]["tarc"]
    ramp = result["candidate_directions"]["ramp"]
    quota.update(
        {
            "route_dot_fraction_of_raw_etarc": quota["validation_etarc_dot"]
            / raw["validation_etarc_dot"]
            if raw["validation_etarc_dot"] > 0
            else math.nan,
            "route_dot_fraction_of_tarc": quota["validation_etarc_dot"]
            / old["validation_etarc_dot"]
            if old["validation_etarc_dot"] > 0
            else math.nan,
            "continuation_dot_fraction_of_ramp": quota[
                "validation_continuation_dot"
            ]
            / ramp["validation_continuation_dot"]
            if ramp["validation_continuation_dot"] > 0
            else math.nan,
            "continuation_cosine_advantage_over_tarc": quota[
                "validation_continuation_cosine"
            ]
            - old["validation_continuation_cosine"],
        }
    )
    return result


def summarize_pair_directions(
    rows: list[dict[str, Any]],
    candidate_names: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_objective = {
        objective: [
            row
            for row in rows
            if row["split"] == "validation" and row["objective"] == objective
        ]
        for objective in ("etarc", "continuation")
    }
    if len(by_objective["etarc"]) != len(by_objective["continuation"]):
        raise QetarcP0ValidationError("paired validation rows differ")
    summary: dict[str, Any] = {"candidates": {}}
    for candidate in candidate_names:
        route_dots = [
            float(row["directional"][candidate]["dot"])
            for row in by_objective["etarc"]
        ]
        continuation_dots = [
            float(row["directional"][candidate]["dot"])
            for row in by_objective["continuation"]
        ]
        summary["candidates"][candidate] = {
            "validation_prompts": len(route_dots),
            "route_positive_rate": sum(value > 0 for value in route_dots)
            / len(route_dots),
            "continuation_compatible_rate": sum(value >= 0 for value in continuation_dots)
            / len(continuation_dots),
            "mean_route_dot": statistics.fmean(route_dots),
            "median_route_dot": statistics.median(route_dots),
            "mean_continuation_dot": statistics.fmean(continuation_dots),
            "median_continuation_dot": statistics.median(continuation_dots),
        }
    quota_continuation = [
        float(row["directional"]["qetarc"]["dot"])
        for row in by_objective["continuation"]
    ]
    tarc_continuation = [
        float(row["directional"]["tarc"]["dot"])
        for row in by_objective["continuation"]
    ]
    differences = [
        left - right
        for left, right in zip(quota_continuation, tarc_continuation, strict=True)
    ]
    summary["qetarc_continuation_advantage_over_tarc"] = {
        "bootstrap": etarc_p0.paired_bootstrap(
            differences, bootstrap_samples, bootstrap_seed
        ),
        "sign_test": etarc_p0.exact_sign_test(differences),
    }
    return summary


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
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(read_jsonl(path)) if path.suffix == ".jsonl" else None,
            }
        )
    return artifacts


def run(path: Path) -> dict[str, Any]:
    import torch

    protocol, rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
    tokenizer = tarc_p0.load_tokenizer(str(protocol["model"]["model_name_or_path"]))
    all_pairs, structural = tarc_p0.build_pair_features(
        tokenizer,
        rows,
        int(protocol["design"]["max_length"]),
        int(protocol["design"]["route_width_limit"]),
    )
    verify_structural_freeze(structural, protocol)
    splits = split_pairs(all_pairs, protocol)
    started = time.time()
    model, model_tokenizer = model_common.load_model(protocol)
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise QetarcP0ValidationError("tokenizer-only and model tokenizers differ")
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
    diagnostic_rows = []
    for objective in ("uniform", "ramp", "tarc", "etarc", "continuation"):
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
    directions, coefficients, geometry = scaled_directions(
        gradients["calibration"],
        float(protocol["design"]["auxiliary_gradient_norm_ratio"]),
        float(protocol["design"]["continuation_quota_cosine"]),
    )
    for objective in (
        "tarc",
        "etarc",
        "continuation",
        "branch_entropy",
        "branch_negative_gap",
    ):
        references = directions if objective in ("etarc", "continuation") else None
        gradient, details, objective_rows = etarc_p0.aggregate_gradient(
            model,
            parameters,
            splits["validation"],
            "validation",
            objective,
            ramp_width,
            temperature,
            references,
        )
        gradients["validation"][objective] = gradient
        coverage["validation"][objective] = details
        diagnostic_rows.extend(objective_rows)
    directional = directional_metrics(
        gradients["calibration"], gradients["validation"], directions
    )
    paired = summarize_pair_directions(
        diagnostic_rows,
        tuple(directions),
        int(protocol["statistics"]["bootstrap_samples"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
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
    quota = directional["candidate_directions"]["qetarc"]
    pair_quota = paired["candidates"]["qetarc"]
    paired_advantage = paired["qetarc_continuation_advantage_over_tarc"]
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
            and float(row["gradient_norm"]) > 0
            for row in records
        ),
        "budget_matched": coefficients["maximum_relative_budget_error"]
        <= float(cfg["gradient_budget_relative_tolerance"]),
        "vram": headroom >= float(cfg["minimum_vram_headroom_fraction"]),
        "state_restored": initial_state == final_state,
        "certificate_structure": (
            structural["all_routes_complete_within_width"]
            and structural["branching_prompt_rate"]
            >= float(cfg["branching_prompt_rate_min"])
            and structural["distinct_route_pair_rate"]
            >= float(cfg["distinct_route_pair_rate_min"])
        ),
        "quota_geometry": (
            geometry["quota_active"]
            and abs(
                geometry["achieved_continuation_cosine"]
                - float(protocol["design"]["continuation_quota_cosine"])
            )
            <= float(cfg["quota_cosine_absolute_tolerance"])
            and geometry["achieved_route_cosine"]
            >= float(cfg["calibration_route_retention_cosine_min"])
            and geometry["continuation_first_order_dot"]
            + float(cfg["quota_dot_numeric_tolerance"])
            >= geometry["continuation_dot_lower_bound"]
        ),
        "route_transfer": (
            directional["calibration_validation_etarc_cosine"]
            >= float(cfg["calibration_validation_etarc_cosine_min"])
            and quota["validation_etarc_dot"] > 0
        ),
        "continuation_transfer": (
            quota["validation_continuation_cosine"]
            >= float(cfg["validation_continuation_cosine_min"])
            and quota["continuation_cosine_advantage_over_tarc"]
            >= float(cfg["continuation_cosine_advantage_over_tarc_min"])
            and quota["continuation_dot_fraction_of_ramp"]
            >= float(cfg["continuation_dot_fraction_of_ramp_min"])
            and pair_quota["continuation_compatible_rate"]
            >= float(cfg["continuation_compatible_pair_rate_min"])
            and paired_advantage["bootstrap"]["ci95_lower"]
            > float(cfg["continuation_advantage_bootstrap_lower_min"])
            and paired_advantage["sign_test"]["exact_two_sided_p"]
            <= float(cfg["continuation_advantage_sign_p_max"])
        ),
        "branch_mechanism": (
            quota["validation_branch_entropy_dot"]
            > float(cfg["branch_entropy_dot_min"])
            and quota["validation_branch_negative_gap_dot"]
            > float(cfg["branch_negative_gap_dot_min"])
        ),
        "effect_not_weak": (
            pair_quota["route_positive_rate"]
            >= float(cfg["route_positive_pair_rate_min"])
            and quota["route_dot_fraction_of_raw_etarc"]
            >= float(cfg["route_dot_fraction_of_raw_etarc_min"])
            and quota["route_dot_fraction_of_tarc"]
            >= float(cfg["route_dot_fraction_of_tarc_min"])
        ),
    }
    route = qetarc.select_route(gates)
    split_manifest = {
        "name": "step3_qetarc_p0_reused_development_split_v1",
        "selection_seed": protocol["selection"]["seed"],
        "calibration_sample_ids": protocol["selection"]["calibration_sample_ids"],
        "validation_sample_ids": protocol["selection"]["validation_sample_ids"],
        "targets_per_prompt": 2,
        "prompt_overlap": 0,
        "fresh_for_qetarc": False,
        "role": "mechanism_development_only",
    }
    diagnostics = {
        "name": "step3_qetarc_p0_gradient_diagnostics_v1",
        "coverage": coverage,
        "coefficients": coefficients,
        "quota_geometry": geometry,
        "directional": directional,
        "paired_validation": paired,
        "gates": gates,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
    }
    decision = {
        "name": "step3_qetarc_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": sha256(path),
        "gates": gates,
        "coefficients": coefficients,
        "quota_geometry": geometry,
        "directional": directional,
        "paired_validation": paired,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "training_authorized": False,
        "matched_smoke_protocol_authorized": route
        == "PASS_QETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL",
        "core100_opened": False,
        "heldout_opened": False,
    }
    bootstrap = paired_advantage["bootstrap"]
    sign_test = paired_advantage["sign_test"]
    report = f"""# Q-E-TARC P0 Positive Continuation Quota Report

Route: `{route}`.

## Geometry

The fixed quota was `{geometry['quota_cosine']:.3f}` at auxiliary norm
`{geometry['budget_norm']:.6f}`. Raw E-TARC/continuation cosine was
`{geometry['raw_route_continuation_cosine']:.6f}`; the quota was active:
`{geometry['quota_active']}`. Achieved continuation and route cosines were
`{geometry['achieved_continuation_cosine']:.6f}` and
`{geometry['achieved_route_cosine']:.6f}`. The analytic continuation-dot lower
bound was `{geometry['continuation_dot_lower_bound']:.6f}` and the observed
calibration dot was `{geometry['continuation_first_order_dot']:.6f}`.

## Prompt-disjoint route transfer

Calibration/validation E-TARC cosine was
`{directional['calibration_validation_etarc_cosine']:.6f}`. Equal-budget
Q-E-TARC retained `{quota['route_dot_fraction_of_raw_etarc']:.3f}` of raw
E-TARC's validation route dot and
`{quota['route_dot_fraction_of_tarc']:.3f}` of TARC's. Its positive route-dot
rate was `{pair_quota['route_positive_rate']:.3f}` over 32 prompts.

## Continuation transfer

Validation continuation cosine was
`{quota['validation_continuation_cosine']:.6f}`, an advantage of
`{quota['continuation_cosine_advantage_over_tarc']:.6f}` over TARC. The
continuation dot retained `{quota['continuation_dot_fraction_of_ramp']:.3f}`
of RAMP's equal-budget positive control. The paired dot advantage over TARC
was `{bootstrap['mean']:.6f}` with 95% bootstrap interval
`[{bootstrap['ci95_lower']:.6f}, {bootstrap['ci95_upper']:.6f}]`; exact sign
test `p={sign_test['exact_two_sided_p']:.6g}`.

## Branch mechanism and boundary

Q-E-TARC branch entropy and negative-gap dots were
`{quota['validation_branch_entropy_dot']:.6f}` and
`{quota['validation_branch_negative_gap_dot']:.6f}`. Positive values preserve
the strong-path signature under descent.

This reused development split is not fresh confirmation or generated
behavior. No optimizer, scheduler, checkpoint, adapter, or candidate output
was created. `optimizer_steps=0`, `training_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
"""
    protocol_sha = sha256(path)
    etarc_p0.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["structural_audit"]), structural
    )
    etarc_p0.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["split_manifest"]), split_manifest
    )
    etarc_p0.write_jsonl_frozen(
        resolve_repo_path(protocol["outputs"]["diagnostic_rows"]), diagnostic_rows
    )
    etarc_p0.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["gradient_diagnostics"]), diagnostics
    )
    etarc_p0.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["environment"]),
        etarc_p0.environment_record(runtime_head, protocol_sha),
    )
    etarc_p0.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["route"]), decision
    )
    etarc_p0.write_text_frozen(
        resolve_repo_path(protocol["outputs"]["report"]), report
    )
    reproduction = {
        "name": "step3_qetarc_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": protocol_sha,
        "artifacts": package_artifacts(protocol),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_qetarc_p0.py run"
        ),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    etarc_p0.write_json_frozen(
        resolve_repo_path(protocol["outputs"]["run_manifest"]), reproduction
    )
    del gradients, directions, parameters, model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "route": route,
                "gates": gates,
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _ = context(path)
    reproduction = read_json(resolve_repo_path(protocol["outputs"]["run_manifest"]))
    if reproduction.get("protocol_sha256") != sha256(path):
        raise QetarcP0ValidationError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise QetarcP0ValidationError(f"artifact mismatch: {artifact['path']}")
    decision = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    recomputed = qetarc.select_route(decision["gates"])
    if recomputed != decision.get("route") or recomputed != reproduction.get("route"):
        raise QetarcP0ValidationError("stored Q-E-TARC routes differ")
    result = {
        "status": "PASS_QETARC_P0_RESULT_VERIFICATION",
        "route": recomputed,
        "artifacts": len(reproduction["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "matched_smoke_protocol_authorized": decision[
            "matched_smoke_protocol_authorized"
        ],
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    try:
        path = resolve_repo_path(args.protocol)
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        QetarcP0ValidationError,
        etarc_p0.EtarcP0ValidationError,
        tarc_p0.TarcP0ValidationError,
        ramp_p0.RampP0ValidationError,
        model_common.CitaP0ValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_QETARC_P0_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
