"""Reconstruct the D1 stop state and test optimizer-space anchor feasibility."""

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

import step3_barp_mcd_d1 as d1
import step3_barp_mcd_training as base
import step3_barp_mcd_training_q1r as q1r
import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_pfrsc_common as pfrsc
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_d1f/protocol/barp_mcd_d1f_protocol.json"
)
STATUS = "prospective_before_any_BARP_MCD_D1F_optimizer_step"


class D1FFeasibilityError(ValueError):
    """Raised when D1F violates its prospective registry."""


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
        raise D1FFeasibilityError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _family_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["minimum_exact_score"],
        record["minimum_target_exact_score"],
        record["mean_target_exact_score"],
        -record["homotopy_fraction"],
        -record["source_subset_size"],
        -record["source_mask"],
    )


def _score_direction(
    direction: dict[str, Any],
    source: dict[str, Any],
    fraction: float,
    raw_baseline: dict[str, Any],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    units: dict[str, dict[str, Any]],
    reference_norm: float,
    safety_mix: float,
    max_norm: float,
    clip_epsilon: float,
) -> dict[str, Any]:
    gradient = gradient_ops.add_gradients(
        raw_baseline, direction, right_scale=safety_mix
    )
    displacement, map_audit = base.current_adamw_displacement(
        parameters, optimizer, gradient, max_norm, clip_epsilon
    )
    scores = {
        name: float(gradient_math.gradient_dot(unit, displacement) / reference_norm)
        for name, unit in units.items()
    }
    target_scores = [scores[name] for name in rp.TARGET_CERTIFICATES]
    record = {
        "source_mask": int(source["mask"]),
        "source_subset": list(source["subset"]),
        "source_subset_size": int(source["subset_size"]),
        "source_dual_margin": float(source["dual_margin"]),
        "source_dual_kkt_residual": float(source["dual_kkt_residual"]),
        "homotopy_fraction": float(fraction),
        "exact_scores": scores,
        "minimum_exact_score": min(scores.values()),
        "mean_exact_score": statistics.fmean(scores.values()),
        "minimum_target_exact_score": min(target_scores),
        "mean_target_exact_score": statistics.fmean(target_scores),
        "candidate_raw_norm": float(map_audit["raw_norm"]),
        "candidate_clip_coefficient": float(map_audit["clip_coefficient"]),
        "candidate_clipping_active": bool(map_audit["clipping_active"]),
        "current_state_completed_steps": int(map_audit["completed_optimizer_steps"]),
        "displacement_norm_fraction_of_baseline": float(
            gradient_math.gradient_norm(displacement) / reference_norm
        ),
    }
    del gradient, displacement
    return record


def enumerate_update_space_family(
    raw_baseline: dict[str, Any],
    certificates: dict[str, dict[str, Any]],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    budget_norm: float,
    safety_mix: float,
    max_norm: float,
    homotopy_fractions: list[float] | tuple[float, ...],
    source_top_k: int,
    clip_epsilon: float = 1e-6,
    dual_tolerance: float = 1e-9,
    feasibility_floor: float = 1e-8,
) -> dict[str, Any]:
    fractions = tuple(float(value) for value in homotopy_fractions)
    if fractions != rp.HOMOTOPY_FRACTIONS or source_top_k != rp.SOURCE_TOP_K:
        raise D1FFeasibilityError("D1F finite family drift")
    names = tuple(certificates)
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
        raise D1FFeasibilityError("D1F baseline displacement is degenerate")
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
        endpoint_records.append(
            _score_direction(
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
        )
        if mask == full_mask:
            full_direction = {name: value.clone() for name, value in direction.items()}
        del direction
    if full_direction is None:
        raise D1FFeasibilityError("D1F full direction is infeasible")
    screened = sorted(
        [record for record in endpoint_records if record["source_mask"] != full_mask],
        key=_family_rank,
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
            candidates.append(
                _score_direction(
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
            )
            del direction
        del endpoint
    selected = max(candidates, key=_family_rank)
    full_record = next(
        record for record in endpoint_records if record["source_mask"] == full_mask
    )
    feasible = [
        record for record in candidates if record["minimum_exact_score"] > feasibility_floor
    ]
    ordered = sorted(candidates, key=_family_rank, reverse=True)
    result = {
        "name": "step3_barp_mcd_d1f_update_space_family_v1",
        "certificate_names": list(names),
        "certificate_norms": certificate_norms,
        "current_optimizer_completed_steps": int(
            baseline_audit["completed_optimizer_steps"]
        ),
        "reference_baseline": {**baseline_audit, "displacement_norm": reference_norm},
        "budget_norm": budget_norm,
        "attempted_endpoint_candidates": full_mask,
        "feasible_endpoint_candidates": len(endpoint_records),
        "endpoint_failures": endpoint_failures,
        "screened_source_count": len(screened),
        "attempted_interior_candidates": source_top_k * (len(fractions) - 2),
        "feasible_interior_candidates": len(candidates) - len(endpoint_records),
        "interior_failures": interior_failures,
        "attempted_total_candidates": full_mask
        + source_top_k * (len(fractions) - 2),
        "feasible_total_candidates": len(candidates),
        "feasibility_floor": feasibility_floor,
        "strict_feasible_candidates": len(feasible),
        "d1_full_mcd_anchor": full_record,
        "selected_update_space_anchor": selected,
        "top_candidates": ordered[:16],
        "all_endpoint_records": sorted(
            endpoint_records, key=lambda record: record["source_mask"]
        ),
        "maximum_retained_endpoint_directions": 2,
        "fourth_optimizer_step_executed": False,
    }
    del baseline_displacement, full_direction
    return result


def context(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != STATUS:
        raise D1FFeasibilityError("D1F protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise D1FFeasibilityError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("training_authorized") is not True
        or int(protocol.get("total_authorized_optimizer_steps", -1)) != 3
        or protocol.get("fourth_optimizer_step_authorized") is not False
        or protocol.get("checkpoint_writes_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise D1FFeasibilityError("D1F authorization boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise D1FFeasibilityError("D1F entry commit is not an ancestor")
    stop = etarc_p0.read_json(resolve_repo_path(protocol["precursor"]["stop_path"]))
    if stop.get("route") != "STOP_BARP_MCD_D1_RAW_ANCHOR_FAILED":
        raise D1FFeasibilityError("D1F precursor route differs")
    d1_protocol = etarc_p0.read_json(
        resolve_repo_path(protocol["precursor"]["d1_protocol_path"])
    )
    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 5
        or int(config["optimizer_steps"]) != 3
        or int(config["optimizer_steps_authorized"]) != 3
        or config.get("fourth_optimizer_step_authorized") is not False
        or float(config["weight_decay"]) != 0.0
        or config.get("gradient_measurement_mode") != "eval"
    ):
        raise D1FFeasibilityError("D1F config differs")
    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    frozen_trace = etarc_p0.read_jsonl(
        resolve_repo_path(protocol["precursor"]["d1_trace_path"])
    )
    if len(frozen_trace) != 3:
        raise D1FFeasibilityError("D1 frozen replay trace differs")
    if protocol["selection"] != d1_protocol["selection"]:
        raise D1FFeasibilityError("D1F selection differs from D1")
    return protocol, config, d1_protocol, targets, risk_rows, frozen_trace


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise D1FFeasibilityError("D1F requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([relative(path), protocol["config_path"]])
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    output_root = resolve_repo_path(protocol["output_root"])
    if output_root.exists():
        existing = [item for item in output_root.iterdir() if item.name != "protocol"]
        if existing:
            raise D1FFeasibilityError("prospective D1F outputs already exist")
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, _, targets, _, frozen_trace = context(path)
    if not torch.cuda.is_available():
        raise D1FFeasibilityError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    selected = [
        sample_id
        for block in protocol["selection"]["optimizer_blocks"][:4]
        for sample_id in block["safety_sample_ids"]
    ]
    if not all(sample_id in pair_map for sample_id in selected):
        raise D1FFeasibilityError("D1F selected safety pair is absent")
    result = {
        "status": "PASS_BARP_MCD_D1F_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "gpu": torch.cuda.get_device_name(0),
        "replay_optimizer_steps": len(frozen_trace),
        "fourth_optimizer_step_authorized": False,
        "endpoint_candidates": 127,
        "interior_candidates": 112,
        "total_candidates": 239,
        "feasibility_floor": protocol["gates"]["anchor_margin_min_exclusive"],
        "checkpoint_writes_authorized": False,
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _nested_close(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def replay_and_measure(
    protocol: dict[str, Any],
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    frozen_trace: list[dict[str, Any]],
    runtime_head: str,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise D1FFeasibilityError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    pair_map, structural = rp.structural_features(tokenizer, targets, config, protocol)
    all_blocks = protocol["selection"]["optimizer_blocks"]
    diagnostic = protocol["selection"]["diagnostic_block"]
    risk_indices = {
        int(index) for block in all_blocks for index in block["risk_row_indices"]
    }.union(int(index) for index in diagnostic["risk_row_indices"])
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in risk_indices
    }
    initial_diagnostic = d1.evaluate_diagnostic_rows(
        model, risk_features, pair_map, diagnostic, config
    )
    optimizer = frozen.make_optimizer(model, config)
    trace: list[dict[str, Any]] = []
    trace_path = resolve_repo_path(protocol["outputs"]["replay_trace"])
    for step_index, block in enumerate(all_blocks[:3], start=1):
        block_risk = [int(value) for value in block["risk_row_indices"]]
        block_safety = [str(value) for value in block["safety_sample_ids"]]
        gradients, coverage = rp.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            block_risk,
            block_safety,
            config,
            train_mode=False,
        )
        raw_baseline = rp._baseline_gradient(gradients, float(config["safety_mix"]))
        budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
            gradients["uniform"]
        )
        _, full, search = q1r.enumerate_memory_bounded_homotopy(
            raw_baseline,
            {name: gradients[name] for name in rp.CERTIFICATES},
            parameters,
            optimizer,
            budget,
            float(config["safety_mix"]),
            float(config["max_grad_norm"]),
            list(protocol["design"]["protected_certificates"]),
            list(protocol["design"]["target_certificates"]),
            float(protocol["design"]["protected_retention_floor"]),
            list(protocol["design"]["homotopy_fractions"]),
            int(protocol["design"]["source_top_k"]),
            float(config["clip_epsilon"]),
            float(config["dual_numeric_tolerance"]),
            int(protocol["design"]["top_candidates_serialized"]),
        )
        applied_gradient = gradient_ops.add_gradients(
            raw_baseline, full, right_scale=float(config["safety_mix"])
        )
        predicted, map_audit = base.current_adamw_displacement(
            parameters,
            optimizer,
            applied_gradient,
            float(config["max_grad_norm"]),
            float(config["clip_epsilon"]),
        )
        before = {name: parameter.detach().clone() for name, parameter in parameters}
        for name, parameter in parameters:
            parameter.grad = applied_gradient[name]
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in parameters], float(config["max_grad_norm"])
            )
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        map_error = base._update_map_error(
            parameters, before, predicted, float(optimizer.param_groups[0]["lr"])
        )
        final_losses = rp.evaluate_block_losses(
            model, risk_features, pair_map, block_risk, block_safety, config
        )
        row = {
            "optimizer_step": step_index,
            "block_id": block["block_id"],
            "risk_row_indices": block_risk,
            "safety_sample_ids": block_safety,
            "coverage": coverage,
            "search": search,
            "applied_homotopy_fraction": 0.0,
            "map_audit": map_audit,
            "map_error": map_error,
            "gradient_preclip_norm": preclip,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "final_same_batch_losses": final_losses,
            "actual_same_batch_loss_reductions": {
                name: float(coverage["loss_means"][name]) - float(final_losses[name])
                for name in rp.CERTIFICATES
            },
        }
        trace.append(row)
        write_jsonl(trace_path, trace)
        print(
            f"BARP-D1F replay {step_index}/3 "
            f"raw_min={search['raw_anchor']['minimum_exact_score']:.6f} "
            f"map_error={map_error['relative_l2_error']:.3e}",
            flush=True,
        )
        del gradients, raw_baseline, full, applied_gradient, predicted, before

    fourth = all_blocks[3]
    gradients, coverage = rp.aggregate_block_gradients(
        model,
        parameters,
        risk_features,
        pair_map,
        [int(value) for value in fourth["risk_row_indices"]],
        [str(value) for value in fourth["safety_sample_ids"]],
        config,
        train_mode=False,
    )
    raw_baseline = rp._baseline_gradient(gradients, float(config["safety_mix"]))
    budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
        gradients["uniform"]
    )
    family = enumerate_update_space_family(
        raw_baseline,
        {name: gradients[name] for name in rp.CERTIFICATES},
        parameters,
        optimizer,
        budget,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        list(protocol["design"]["homotopy_fractions"]),
        int(protocol["design"]["source_top_k"]),
        float(config["clip_epsilon"]),
        float(config["dual_numeric_tolerance"]),
        float(protocol["gates"]["anchor_margin_min_exclusive"]),
    )
    tolerance = float(protocol["gates"]["replay_matching_tolerance"])
    replay_matches = all(
        _nested_close(
            {
                "block_id": observed["block_id"],
                "risk_row_indices": observed["risk_row_indices"],
                "safety_sample_ids": observed["safety_sample_ids"],
                "coverage": observed["coverage"],
                "search": observed["search"],
                "map_audit": observed["map_audit"],
                "gradient_preclip_norm": observed["gradient_preclip_norm"],
                "learning_rate": observed["learning_rate"],
                "final_same_batch_losses": observed["final_same_batch_losses"],
                "actual_same_batch_loss_reductions": observed[
                    "actual_same_batch_loss_reductions"
                ],
            },
            {
                key: frozen[key]
                for key in (
                    "block_id",
                    "risk_row_indices",
                    "safety_sample_ids",
                    "coverage",
                    "search",
                    "map_audit",
                    "gradient_preclip_norm",
                    "learning_rate",
                    "final_same_batch_losses",
                    "actual_same_batch_loss_reductions",
                )
            },
            tolerance,
        )
        for observed, frozen in zip(trace, frozen_trace, strict=True)
    )
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    result = {
        "name": "step3_barp_mcd_d1f_run_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(
            resolve_repo_path(protocol["protocol_path"])
        ),
        "initial_trainable_state_sha256": initial_hash,
        "final_reconstructed_state_sha256": model_common.trainable_state_hash(parameters),
        "replay_optimizer_steps": len(trace),
        "replay_matches_frozen_d1_trace": replay_matches,
        "maximum_update_map_relative_l2_error": max(
            row["map_error"]["relative_l2_error"] for row in trace
        ),
        "all_replay_same_batch_reductions_positive": all(
            value > 0.0
            for row in trace
            for value in row["actual_same_batch_loss_reductions"].values()
        ),
        "initial_diagnostic_sha256": training.hash_json(initial_diagnostic),
        "fourth_block_id": fourth["block_id"],
        "fourth_block_coverage": coverage,
        "family": family,
        "fourth_optimizer_step_executed": False,
        "total_optimizer_steps": len(trace),
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_memory,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "wall_seconds": time.time() - started,
        "structural_audit_sha256": training.hash_json(structural),
        "trainable_parameter_inventory": inventory,
    }
    del optimizer, model, tokenizer, gradients, raw_baseline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def checks_for(result: dict[str, Any], protocol: dict[str, Any]) -> dict[str, bool]:
    family = result["family"]
    return {
        "replay_steps_exact": result["replay_optimizer_steps"] == 3,
        "replay_matches_d1": bool(result["replay_matches_frozen_d1_trace"]),
        "optimizer_map_exact": result["maximum_update_map_relative_l2_error"]
        <= float(protocol["gates"]["optimizer_map_relative_error_max"]),
        "replay_finite_descent": bool(result["all_replay_same_batch_reductions_positive"]),
        "d1_failure_reproduced": family["d1_full_mcd_anchor"]["minimum_exact_score"]
        <= float(protocol["gates"]["d1_full_anchor_margin_max"]),
        "family_complete": family["attempted_total_candidates"] == 239
        and family["feasible_total_candidates"] == 239
        and not family["endpoint_failures"]
        and not family["interior_failures"],
        "update_space_anchor_feasible": family["selected_update_space_anchor"][
            "minimum_exact_score"
        ]
        > float(protocol["gates"]["anchor_margin_min_exclusive"]),
        "no_fourth_update": result["total_optimizer_steps"] == 3
        and not result["fourth_optimizer_step_executed"],
        "vram": result["vram_headroom_fraction"]
        >= float(protocol["gates"]["minimum_vram_headroom_fraction"]),
        "boundary": result["checkpoint_writes"] == 0
        and not result["candidate_outputs_generated"]
        and not result["core100_opened"]
        and not result["heldout_opened"],
    }


def select_route(checks: dict[str, bool]) -> str:
    order = (
        ("replay_steps_exact", "STOP_BARP_MCD_D1F_REPLAY_COUNT_FAILED"),
        ("replay_matches_d1", "STOP_BARP_MCD_D1F_REPLAY_MATCH_FAILED"),
        ("optimizer_map_exact", "STOP_BARP_MCD_D1F_OPTIMIZER_MAP_FAILED"),
        ("replay_finite_descent", "STOP_BARP_MCD_D1F_REPLAY_DESCENT_FAILED"),
        ("d1_failure_reproduced", "STOP_BARP_MCD_D1F_D1_FAILURE_NOT_REPRODUCED"),
        ("family_complete", "STOP_BARP_MCD_D1F_FAMILY_INCOMPLETE"),
        ("update_space_anchor_feasible", "STOP_BARP_MCD_D1F_NO_FEASIBLE_ANCHOR"),
        ("no_fourth_update", "STOP_BARP_MCD_D1F_BOUNDARY_FAILED"),
        ("vram", "STOP_BARP_MCD_D1F_VRAM_FAILED"),
        ("boundary", "STOP_BARP_MCD_D1F_BOUNDARY_FAILED"),
    )
    for gate, route in order:
        if not checks.get(gate, False):
            return route
    return "PASS_BARP_MCD_D1F_UPDATE_SPACE_ANCHOR_FEASIBLE"


def package_artifacts(root: Path, reproduction: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": relative(path),
            "sha256": etarc_p0.sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(etarc_p0.read_jsonl(path)) if path.suffix == ".jsonl" else None,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != reproduction and "protocol" not in path.parts
    ]


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, config, _, targets, risk_rows, frozen_trace = context(path)
    runtime_head = require_clean_start(path, protocol)
    result = replay_and_measure(
        protocol, config, targets, risk_rows, frozen_trace, runtime_head
    )
    result_path = resolve_repo_path(protocol["outputs"]["feasibility"])
    write_json(result_path, result)
    checks = checks_for(result, protocol)
    route = select_route(checks)
    family = result["family"]
    decision = {
        "name": "step3_barp_mcd_d1f_decision_v1",
        "route": route,
        "checks": checks,
        "d1_full_anchor_minimum_score": family["d1_full_mcd_anchor"][
            "minimum_exact_score"
        ],
        "d1_full_anchor_exact_scores": family["d1_full_mcd_anchor"]["exact_scores"],
        "selected_anchor": family["selected_update_space_anchor"],
        "strict_feasible_candidates": family["strict_feasible_candidates"],
        "total_candidates": family["feasible_total_candidates"],
        "total_optimizer_steps": result["total_optimizer_steps"],
        "fourth_optimizer_step_executed": False,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "matched_retry_authorized": route
        == "PASS_BARP_MCD_D1F_UPDATE_SPACE_ANCHOR_FEASIBLE",
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    write_json(decision_path, decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(
        "# BARP-MCD D1F Update-Space Anchor Feasibility\n\n"
        f"Decision: `{route}`\n\n"
        f"The frozen D1 full-MCD anchor minimum exact score was "
        f"`{decision['d1_full_anchor_minimum_score']:.10f}`. The optimizer-space "
        f"finite-family maximum minimum score was "
        f"`{decision['selected_anchor']['minimum_exact_score']:.10f}` with "
        f"`{decision['strict_feasible_candidates']}/{decision['total_candidates']}` "
        "strict feasible candidates.\n\n"
        "Exactly three raw updates reconstructed the failure state; the fourth update "
        "was not executed. No checkpoint or response was written, and Core100 and "
        "heldout remained sealed.\n",
        encoding="utf-8",
    )
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    reproduction = {
        "name": "step3_barp_mcd_d1f_reproduction_manifest_v1",
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "route": route,
        "total_optimizer_steps": result["total_optimizer_steps"],
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1f.py preflight",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1f.py run",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1f.py verify-run",
        ],
        "artifacts": package_artifacts(
            resolve_repo_path(protocol["output_root"]), reproduction_path
        ),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }
    write_json(reproduction_path, reproduction)
    print(json.dumps(decision, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _, _, _ = context(path)
    result = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["feasibility"]))
    checks = checks_for(result, protocol)
    route = select_route(checks)
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    reproduction = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if decision["checks"] != checks or decision["route"] != route:
        raise D1FFeasibilityError("D1F decision does not reproduce")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise D1FFeasibilityError(f"D1F artifact hash mismatch: {artifact['path']}")
    verified = {
        "status": "PASS_BARP_MCD_D1F_VERIFY",
        "route": route,
        "checks": checks,
        "artifacts_verified": len(reproduction["artifacts"]),
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "fourth_optimizer_step_executed": False,
        "checkpoint_writes": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(verified, sort_keys=True))
    return verified


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
