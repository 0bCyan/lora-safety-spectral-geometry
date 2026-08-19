"""Run optimizer-anchor versus route-protected BARP-MCD D1R."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import step3_barp_mcd_d1 as d1
import step3_barp_mcd_d1f as d1f
import step3_barp_mcd_training as base
import step3_barp_mcd_training_q1r as q1r
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_pfrsc_common as pfrsc
import step3_ramp_common as gradient_math
import step3_rpfrsc_p0 as rp


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_d1r/protocol/barp_mcd_d1r_protocol.json"
)
STATUS = "prospective_before_any_BARP_MCD_D1R_optimizer_step"
ORIGINAL_SEARCH = q1r.enumerate_memory_bounded_homotopy


class D1RValidationError(ValueError):
    """Raised when D1R violates its optimizer-anchor registry."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def _anchor_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return d1f._family_rank(record)


def _route_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["minimum_target_exact_score"],
        record["minimum_exact_score"],
        record["mean_target_exact_score"],
        record["minimum_protected_retention"],
        -record["homotopy_fraction"],
        -record["source_subset_size"],
        -record["source_mask"],
    )


def _candidate_id(record: dict[str, Any]) -> str:
    return f"mask{int(record['source_mask']):03d}_lambda{float(record['homotopy_fraction']):.3f}"


def _direction_for_record(
    record: dict[str, Any],
    units: dict[str, dict[str, Any]],
    gram: list[list[float]],
    budget_norm: float,
    tolerance: float,
) -> dict[str, Any]:
    endpoint = q1r._direction_for_mask(
        units, gram, int(record["source_mask"]), budget_norm, tolerance
    )
    fraction = float(record["homotopy_fraction"])
    if fraction == 1.0:
        return endpoint
    full = q1r._direction_for_mask(
        units, gram, 2 ** len(units) - 1, budget_norm, tolerance
    )
    direction = pfrsc._homotopy_direction(
        full, endpoint, fraction, budget_norm, tolerance
    )
    del full, endpoint
    return direction


def enumerate_optimizer_anchor_homotopy(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    budget_norm: float,
    safety_mix: float,
    max_norm: float,
    protected_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
    protected_retention: float,
    homotopy_fractions: list[float] | tuple[float, ...],
    source_top_k: int,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
    serialized_top_k: int = 8,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protected = tuple(protected_names)
    targets = tuple(target_names)
    fractions = tuple(float(value) for value in homotopy_fractions)
    names = tuple(certificates)
    if (
        protected != rp.PROTECTED_CERTIFICATES
        or targets != rp.TARGET_CERTIFICATES
        or fractions != rp.HOMOTOPY_FRACTIONS
        or source_top_k != rp.SOURCE_TOP_K
        or not math.isfinite(protected_retention)
        or not 0.0 < protected_retention <= 1.0
    ):
        raise D1RValidationError("D1R finite family drift")
    units, certificate_norms = frsc.normalized_certificates(certificates)
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units.values()]
        for left in units.values()
    ]
    baseline_displacement, baseline_audit = base.current_adamw_displacement(
        parameters, optimizer, raw_baseline, max_norm, clip_epsilon
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise D1RValidationError("D1R baseline displacement is degenerate")
    full_mask = 2 ** len(names) - 1
    endpoint_records: list[dict[str, Any]] = []
    endpoint_failures: list[dict[str, Any]] = []
    full_direction: dict[str, Any] | None = None
    for mask in range(1, full_mask + 1):
        indices = [index for index in range(len(names)) if mask & (1 << index)]
        try:
            direction, dual = frsc._subset_direction(
                units, gram, indices, budget_norm, dual_tolerance
            )
        except ValueError as exc:
            endpoint_failures.append({"mask": mask, "error": str(exc)})
            continue
        source = {
            "mask": mask,
            "subset": [names[index] for index in indices],
            "subset_size": len(indices),
            "dual_margin": float(dual["dual_margin"]),
            "dual_kkt_residual": float(dual["kkt_residual"]),
        }
        record = d1f._score_direction(
            direction,
            source,
            1.0,
            raw_baseline,
            parameters,
            optimizer,
            units,
            reference_norm,
            safety_mix,
            max_norm,
            clip_epsilon,
        )
        record["candidate_id"] = _candidate_id(record)
        endpoint_records.append(record)
        if mask == full_mask:
            full_direction = {name: value.clone() for name, value in direction.items()}
        del direction
    if full_direction is None:
        raise D1RValidationError("D1R full endpoint is infeasible")
    screened = sorted(
        [record for record in endpoint_records if record["source_mask"] != full_mask],
        key=_anchor_rank,
        reverse=True,
    )[:source_top_k]
    candidates = list(endpoint_records)
    interior_failures: list[dict[str, Any]] = []
    for source_record in screened:
        endpoint = q1r._direction_for_mask(
            units,
            gram,
            int(source_record["source_mask"]),
            budget_norm,
            dual_tolerance,
        )
        descriptor = {
            "mask": source_record["source_mask"],
            "subset": source_record["source_subset"],
            "subset_size": source_record["source_subset_size"],
            "dual_margin": source_record["source_dual_margin"],
            "dual_kkt_residual": source_record["source_dual_kkt_residual"],
        }
        for fraction in fractions[1:-1]:
            try:
                direction = pfrsc._homotopy_direction(
                    full_direction,
                    endpoint,
                    fraction,
                    budget_norm,
                    dual_tolerance,
                )
            except ValueError as exc:
                interior_failures.append(
                    {
                        "mask": source_record["source_mask"],
                        "fraction": fraction,
                        "error": str(exc),
                    }
                )
                continue
            record = d1f._score_direction(
                direction,
                descriptor,
                fraction,
                raw_baseline,
                parameters,
                optimizer,
                units,
                reference_norm,
                safety_mix,
                max_norm,
                clip_epsilon,
            )
            record["candidate_id"] = _candidate_id(record)
            candidates.append(record)
            del direction
        del endpoint
    anchor = max(candidates, key=_anchor_rank)
    floor = 1e-8
    if anchor["minimum_exact_score"] <= floor:
        raise D1RValidationError(
            "D1R update-space anchor lacks strict common descent"
        )
    protected_candidates = []
    for record in candidates:
        ratios = {
            name: record["exact_scores"][name] / anchor["exact_scores"][name]
            for name in protected
        }
        record["protected_score_retention"] = ratios
        record["minimum_protected_retention"] = min(ratios.values())
        if record["minimum_protected_retention"] + 1e-12 >= protected_retention:
            protected_candidates.append(record)
    selected = max(protected_candidates, key=_route_rank)
    anchor_direction = _direction_for_record(
        anchor, units, gram, budget_norm, dual_tolerance
    )
    selected_direction = _direction_for_record(
        selected, units, gram, budget_norm, dual_tolerance
    )
    anchor = dict(anchor)
    selected = dict(selected)
    anchor["anchor_margin_floor"] = floor
    anchor["protected_score_retention"] = {name: 1.0 for name in protected}
    anchor["minimum_protected_retention"] = 1.0
    selected["route_differs_from_anchor"] = (
        selected["candidate_id"] != anchor["candidate_id"]
    )
    ordered_anchor = sorted(candidates, key=_anchor_rank, reverse=True)
    ordered_route = sorted(protected_candidates, key=_route_rank, reverse=True)
    audit = {
        "name": "step3_barp_mcd_d1r_optimizer_anchor_search_v1",
        "certificate_names": list(names),
        "certificate_norms": certificate_norms,
        "protected_certificates": list(protected),
        "target_certificates": list(targets),
        "attempted_source_subsets": full_mask,
        "feasible_source_subsets": len(endpoint_records),
        "infeasible_source_subsets": len(endpoint_failures),
        "endpoint_failures": endpoint_failures,
        "source_top_k": source_top_k,
        "screened_source_count": len(screened),
        "attempted_homotopy_candidates": full_mask
        + source_top_k * (len(fractions) - 2),
        "feasible_homotopy_candidates": len(candidates),
        "infeasible_homotopy_interiors": len(interior_failures),
        "interior_failures": interior_failures,
        "exact_map_evaluations": len(candidates),
        "strict_anchor_candidates": sum(
            record["minimum_exact_score"] > floor for record in candidates
        ),
        "anchor_margin_floor": floor,
        "protected_retention_floor": protected_retention,
        "protected_feasible_candidates": len(protected_candidates),
        "raw_anchor": anchor,
        "selected": selected,
        "selected_target_gain_over_raw": float(
            selected["minimum_target_exact_score"]
            - anchor["minimum_target_exact_score"]
        ),
        "top_anchor_candidates": ordered_anchor[: min(serialized_top_k, len(ordered_anchor))],
        "top_protected_candidates": ordered_route[: min(serialized_top_k, len(ordered_route))],
        "screened_source_endpoints": screened,
        "baseline": {
            **baseline_audit,
            "displacement_norm": reference_norm,
        },
        "budget_norm": budget_norm,
        "current_optimizer_completed_steps": int(
            baseline_audit["completed_optimizer_steps"]
        ),
        "endpoint_tensor_policy": "score_delete_then_reconstruct_selected_masks",
        "maximum_retained_endpoint_directions": 2,
    }
    del baseline_displacement, full_direction
    return selected_direction, anchor_direction, audit


def install() -> None:
    d1.STATUS = STATUS
    q1r.enumerate_memory_bounded_homotopy = enumerate_optimizer_anchor_homotopy


def select_route(checks: dict[str, bool]) -> str:
    return d1.select_route(checks).replace("BARP_MCD_D1", "BARP_MCD_D1R")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    install()
    protocol, config, targets, _ = d1.context(path)
    d1.resolve_external_root(protocol, require_absent=True)
    if not torch.cuda.is_available():
        raise D1RValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    selected = [
        sample_id
        for block in protocol["selection"]["optimizer_blocks"]
        for sample_id in block["safety_sample_ids"]
    ] + list(protocol["selection"]["diagnostic_block"]["safety_sample_ids"])
    if len(set(selected)) != 40 or not all(value in pair_map for value in selected):
        raise D1RValidationError("D1R safety selection differs")
    result = {
        "status": "PASS_BARP_MCD_D1R_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "gpu": torch.cuda.get_device_name(0),
        "arms": list(d1.ARMS),
        "optimizer_steps_per_arm": 8,
        "total_authorized_optimizer_steps": 16,
        "optimizer_anchor_candidates_per_step": 239,
        "protected_retention_floor": 0.95,
        "diagnostic_units": 8,
        "checkpoint_writes_exact": 2,
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _postprocess_manifest(
    arm: str, manifest: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    output = d1.resolve_repo_path(protocol["arm_outputs"][arm])
    trace = etarc_p0.read_jsonl(output / "training_trace.jsonl")
    nonanchor = sum(
        row["search"]["selected"]["candidate_id"]
        != row["search"]["raw_anchor"]["candidate_id"]
        for row in trace
    )
    manifest["name"] = "step3_barp_mcd_d1r_arm_manifest_v1"
    manifest["anchor_definition"] = "exact_current_adamw_update_space_maximin"
    manifest["minimum_anchor_margin"] = min(
        row["search"]["raw_anchor"]["minimum_exact_score"] for row in trace
    )
    manifest["nonanchor_route_choices"] = nonanchor if arm == "barp_mcd" else 0
    manifest["nonzero_barp_choices"] = nonanchor if arm == "barp_mcd" else 0
    d1.write_json(output / "arm_manifest.json", manifest)
    return manifest


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    install()
    protocol, config, targets, risk_rows = d1.context(path)
    runtime_head = d1.require_clean_start(path, protocol)
    checkpoint_root = d1.resolve_external_root(protocol, require_absent=True)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifests = {}
    for arm in d1.ARMS:
        manifest = d1.train_arm(
            arm,
            protocol,
            config,
            targets,
            risk_rows,
            runtime_head,
            checkpoint_root,
        )
        manifests[arm] = _postprocess_manifest(arm, manifest, protocol)
    diagnostic = d1.paired_diagnostic_analysis(manifests, protocol)
    diagnostic_path = d1.resolve_repo_path(
        protocol["outputs"]["paired_diagnostic_analysis"]
    )
    d1.write_json(diagnostic_path, diagnostic)
    checks = d1.compare_manifests(manifests, diagnostic, protocol)
    route = select_route(checks)
    decision = {
        "name": "step3_barp_mcd_d1r_decision_v1",
        "route": route,
        "checks": checks,
        "arms": {
            arm: {
                "optimizer_steps": manifest["optimizer_steps"],
                "final_trainable_state_sha256": manifest["final_trainable_state_sha256"],
                "maximum_update_map_relative_l2_error": manifest[
                    "maximum_update_map_relative_l2_error"
                ],
                "minimum_anchor_margin": manifest["minimum_anchor_margin"],
                "minimum_selected_protected_retention": manifest[
                    "minimum_selected_protected_retention"
                ],
                "minimum_selected_target_gain": manifest["minimum_selected_target_gain"],
                "nonanchor_route_choices": manifest["nonanchor_route_choices"],
                "vram_headroom_fraction": manifest["vram_headroom_fraction"],
                "checkpoint_inventory": manifest["checkpoint_inventory"],
            }
            for arm, manifest in manifests.items()
        },
        "paired_diagnostic": diagnostic,
        "total_optimizer_steps": sum(item["optimizer_steps"] for item in manifests.values()),
        "checkpoint_writes": sum(item["checkpoint_writes"] for item in manifests.values()),
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "checkpoint_bound_generation_protocol_authorized": route
        == "PASS_BARP_MCD_D1R_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL",
        "main_training_authorized": False,
        "wall_seconds": time.time() - started,
    }
    decision_path = d1.resolve_repo_path(protocol["outputs"]["decision"])
    d1.write_json(decision_path, decision)
    target_stats = diagnostic["target_macro_advantage"]
    report_path = d1.resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(
        "# BARP-MCD D1R Optimizer-Anchor Matched Confirmation\n\n"
        f"Decision: `{route}`\n\n"
        f"Both arms completed `{decision['total_optimizer_steps']}` total optimizer steps. "
        f"The treatment selected a non-anchor route on "
        f"`{manifests['barp_mcd']['nonanchor_route_choices']}/8` steps.\n\n"
        f"Paired target macro advantage: `{target_stats['mean']:.8f}`; 95% paired "
        f"bootstrap interval `[{target_stats['lower_2_5_percent']:.8f}, "
        f"{target_stats['upper_97_5_percent']:.8f}]`; exact sign-test p="
        f"`{target_stats['sign_test']['two_sided_p_value']:.8f}`.\n\n"
        "Each arm froze one external adapter. No response was generated; Core100 and "
        "heldout remained sealed.\n",
        encoding="utf-8",
    )
    reproduction_path = d1.resolve_repo_path(
        protocol["outputs"]["reproduction_manifest"]
    )
    reproduction = {
        "name": "step3_barp_mcd_d1r_reproduction_manifest_v1",
        "runtime_head": runtime_head,
        "protocol_path": d1.relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(
            d1.resolve_repo_path(protocol["config_path"])
        ),
        "checkpoint_root": checkpoint_root.as_posix(),
        "checkpoint_inventories": {
            arm: manifests[arm]["checkpoint_inventory"] for arm in d1.ARMS
        },
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1r.py preflight",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1r.py run",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1r.py verify-run",
        ],
        "route": route,
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "artifacts": d1.package_artifacts(
            d1.resolve_repo_path(protocol["output_root"]), reproduction_path
        ),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }
    d1.write_json(reproduction_path, reproduction)
    print(json.dumps(decision, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    install()
    protocol, _, _, _ = d1.context(path)
    manifests = {
        arm: etarc_p0.read_json(
            d1.resolve_repo_path(protocol["arm_outputs"][arm]) / "arm_manifest.json"
        )
        for arm in d1.ARMS
    }
    diagnostic = d1.paired_diagnostic_analysis(manifests, protocol)
    stored = etarc_p0.read_json(
        d1.resolve_repo_path(protocol["outputs"]["paired_diagnostic_analysis"])
    )
    if diagnostic != stored:
        raise D1RValidationError("D1R paired diagnostic does not reproduce")
    checks = d1.compare_manifests(manifests, diagnostic, protocol)
    route = select_route(checks)
    decision = etarc_p0.read_json(d1.resolve_repo_path(protocol["outputs"]["decision"]))
    reproduction = etarc_p0.read_json(
        d1.resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if decision["checks"] != checks or decision["route"] != route:
        raise D1RValidationError("D1R decision does not reproduce")
    for artifact in reproduction["artifacts"]:
        artifact_path = d1.resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise D1RValidationError(f"D1R artifact hash mismatch: {artifact['path']}")
    checkpoint_files = d1.verify_external_checkpoints(
        manifests, d1.resolve_external_root(protocol, require_absent=False)
    )
    result = {
        "status": "PASS_BARP_MCD_D1R_VERIFY",
        "route": route,
        "checks": checks,
        "artifacts_verified": len(reproduction["artifacts"]),
        "checkpoint_files_verified": checkpoint_files,
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "checkpoint_writes": decision["checkpoint_writes"],
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    install()
    path = d1.resolve_repo_path(args.protocol)
    if args.command == "preflight":
        preflight(path)
    elif args.command == "run":
        run(path)
    else:
        verify_run(path)


if __name__ == "__main__":
    main()
