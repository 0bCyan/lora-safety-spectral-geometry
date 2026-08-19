"""Run the frozen BARP-MCD two-step matched training qualification."""

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
import step3_frsc_common as frsc
import step3_ojmcd_common as optimizer_map
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
    "4070ti_step3_barp_mcd_q1/protocol/barp_mcd_q1_protocol.json"
)
STATUS = "prospective_before_any_BARP_MCD_Q1_optimizer_step"
ARMS = ("raw_mcd", "barp_mcd")


class BarpTrainingValidationError(ValueError):
    """Raised when BARP-MCD Q1 violates its frozen protocol."""


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
        raise BarpTrainingValidationError(f"path escapes repository: {value}") from exc
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


def current_adamw_displacement(
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    raw_gradient: dict[str, Any],
    max_norm: float,
    clip_epsilon: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the positive next AdamW displacement without mutating state.

    Q1 freezes zero weight decay and one parameter group, so the returned
    displacement multiplied by the group's learning rate is the exact update.
    """
    if set(raw_gradient) != {name for name, _ in parameters}:
        raise BarpTrainingValidationError("optimizer-map gradient keys differ")
    if len(optimizer.param_groups) != 1:
        raise BarpTrainingValidationError("Q1 requires one optimizer group")
    group = optimizer.param_groups[0]
    if float(group["weight_decay"]) != 0.0 or bool(group.get("amsgrad", False)):
        raise BarpTrainingValidationError("unsupported AdamW state in Q1")
    if bool(group.get("maximize", False)):
        raise BarpTrainingValidationError("maximizing AdamW is not authorized")
    beta1, beta2 = (float(value) for value in group["betas"])
    epsilon = float(group["eps"])
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0 and epsilon > 0.0):
        raise BarpTrainingValidationError("invalid AdamW hyperparameters")

    clipped, clip_audit = optimizer_map.clip_gradient(
        raw_gradient, max_norm, clip_epsilon
    )
    displacement: dict[str, Any] = {}
    completed_steps: set[int] = set()
    for name, parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if state:
            if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
                raise BarpTrainingValidationError("incomplete AdamW moment state")
            completed = int(float(state["step"].item()))
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
        else:
            completed = 0
            exp_avg = parameter.new_zeros(parameter.shape)
            exp_avg_sq = parameter.new_zeros(parameter.shape)
        completed_steps.add(completed)
        step = completed + 1
        moment = exp_avg * beta1 + clipped[name] * (1.0 - beta1)
        variance = exp_avg_sq * beta2 + clipped[name].square() * (1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        displacement[name] = (moment / bias1) / (
            torch_sqrt(variance / bias2) + epsilon
        )
    if len(completed_steps) != 1:
        raise BarpTrainingValidationError("AdamW parameter steps are not synchronized")
    completed = next(iter(completed_steps))
    audit = {
        **clip_audit,
        "completed_optimizer_steps": completed,
        "next_optimizer_step": completed + 1,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "learning_rate": float(group["lr"]),
        "displacement_norm": gradient_math.gradient_norm(displacement),
    }
    return displacement, audit


def torch_sqrt(value: Any) -> Any:
    # Kept as a tiny seam so optimizer-map unit tests need no model imports.
    import torch

    return torch.sqrt(value)


def _score_displacement(
    certificate_units: dict[str, dict[str, Any]],
    displacement: dict[str, Any],
    reference_norm: float,
) -> dict[str, Any]:
    scores = {
        name: gradient_math.gradient_dot(unit, displacement) / reference_norm
        for name, unit in certificate_units.items()
    }
    return {
        "scores": scores,
        "minimum_score": min(scores.values()),
        "mean_score": statistics.fmean(scores.values()),
        "displacement_norm": gradient_math.gradient_norm(displacement),
    }


def _candidate_record(
    direction: dict[str, Any],
    source: dict[str, Any],
    fraction: float,
    raw_baseline: dict[str, Any],
    parameters: list[tuple[str, Any]],
    optimizer: Any,
    certificate_units: dict[str, dict[str, Any]],
    raw_scores: dict[str, float],
    reference_norm: float,
    safety_mix: float,
    max_norm: float,
    clip_epsilon: float,
    protected: tuple[str, ...],
    targets: tuple[str, ...],
) -> dict[str, Any]:
    gradient = gradient_ops.add_gradients(
        raw_baseline, direction, right_scale=safety_mix
    )
    displacement, map_audit = current_adamw_displacement(
        parameters, optimizer, gradient, max_norm, clip_epsilon
    )
    scored = _score_displacement(certificate_units, displacement, reference_norm)
    scores = {name: float(value) for name, value in scored["scores"].items()}
    protected_ratios = {
        name: scores[name] / raw_scores[name] for name in protected
    }
    target_scores = {name: scores[name] for name in targets}
    record = {
        "source_mask": int(source["mask"]),
        "source_subset": list(source["subset"]),
        "source_subset_size": int(source["subset_size"]),
        "source_dual_margin": float(source["dual_margin"]),
        "source_dual_kkt_residual": float(source["dual_kkt_residual"]),
        "homotopy_fraction": float(fraction),
        "minimum_exact_score": float(scored["minimum_score"]),
        "mean_exact_score": float(scored["mean_score"]),
        "minimum_target_exact_score": min(target_scores.values()),
        "mean_target_exact_score": statistics.fmean(target_scores.values()),
        "minimum_protected_retention": min(protected_ratios.values()),
        "protected_score_retention": protected_ratios,
        "exact_scores": scores,
        "candidate_raw_norm": float(map_audit["raw_norm"]),
        "candidate_clip_coefficient": float(map_audit["clip_coefficient"]),
        "candidate_clipping_active": bool(map_audit["clipping_active"]),
        "current_state_completed_steps": int(
            map_audit["completed_optimizer_steps"]
        ),
        "displacement_norm_fraction_of_baseline": float(
            scored["displacement_norm"] / reference_norm
        ),
    }
    del gradient, displacement
    return record


def _source_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["minimum_target_exact_score"],
        record["minimum_exact_score"],
        record["mean_target_exact_score"],
        -record["source_subset_size"],
        -record["source_mask"],
    )


def _selection_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["minimum_target_exact_score"],
        record["minimum_exact_score"],
        record["mean_target_exact_score"],
        record["minimum_protected_retention"],
        -record["homotopy_fraction"],
        -record["source_subset_size"],
        -record["source_mask"],
    )


def enumerate_current_state_homotopy(
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
    """Run the frozen finite search through the current AdamW moment state."""
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
        raise BarpTrainingValidationError("current-state finite family drift")
    units, certificate_norms = frsc.normalized_certificates(certificates)
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units.values()]
        for left in units.values()
    ]
    full_mask = 2 ** len(names) - 1
    full_indices = list(range(len(names)))
    full_direction, full_dual = frsc._subset_direction(
        units, gram, full_indices, budget_norm, dual_tolerance
    )
    full_source = {
        "mask": full_mask,
        "subset": list(names),
        "subset_size": len(names),
        "dual_margin": float(full_dual["dual_margin"]),
        "dual_kkt_residual": float(full_dual["kkt_residual"]),
    }
    baseline_displacement, baseline_audit = current_adamw_displacement(
        parameters, optimizer, raw_baseline, max_norm, clip_epsilon
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise BarpTrainingValidationError("current-state baseline is degenerate")
    raw_record = _candidate_record(
        full_direction,
        full_source,
        0.0,
        raw_baseline,
        parameters,
        optimizer,
        units,
        {name: 1.0 for name in names},
        reference_norm,
        safety_mix,
        max_norm,
        clip_epsilon,
        protected,
        targets,
    )
    raw_scores = dict(raw_record["exact_scores"])
    if any(value <= 0.0 for value in raw_scores.values()):
        raise BarpTrainingValidationError("current raw anchor lacks common descent")
    raw_record["protected_score_retention"] = {name: 1.0 for name in protected}
    raw_record["minimum_protected_retention"] = 1.0

    source_records: list[dict[str, Any]] = []
    source_directions: dict[int, dict[str, Any]] = {}
    infeasible_subsets = 0
    for mask in range(1, full_mask):
        indices = [index for index in range(len(names)) if mask & (1 << index)]
        try:
            direction, dual = frsc._subset_direction(
                units, gram, indices, budget_norm, dual_tolerance
            )
        except ValueError:
            infeasible_subsets += 1
            continue
        source = {
            "mask": mask,
            "subset": [names[index] for index in indices],
            "subset_size": len(indices),
            "dual_margin": float(dual["dual_margin"]),
            "dual_kkt_residual": float(dual["kkt_residual"]),
        }
        source_records.append(
            _candidate_record(
                direction,
                source,
                1.0,
                raw_baseline,
                parameters,
                optimizer,
                units,
                raw_scores,
                reference_norm,
                safety_mix,
                max_norm,
                clip_epsilon,
                protected,
                targets,
            )
        )
        source_directions[mask] = direction

    screened = sorted(source_records, key=_source_rank, reverse=True)[
        : min(source_top_k, len(source_records))
    ]
    endpoint_by_mask = {record["source_mask"]: record for record in source_records}
    candidates = [raw_record]
    protected_candidates = [raw_record]
    interior_failures = 0
    for source in screened:
        endpoint = source_directions[int(source["source_mask"])]
        source_descriptor = {
            "mask": source["source_mask"],
            "subset": source["source_subset"],
            "subset_size": source["source_subset_size"],
            "dual_margin": source["source_dual_margin"],
            "dual_kkt_residual": source["source_dual_kkt_residual"],
        }
        for fraction in fractions[1:]:
            if fraction == 1.0:
                candidate = endpoint_by_mask[int(source["source_mask"])]
            else:
                try:
                    direction = pfrsc._homotopy_direction(
                        full_direction,
                        endpoint,
                        fraction,
                        budget_norm,
                        dual_tolerance,
                    )
                except ValueError:
                    interior_failures += 1
                    continue
                candidate = _candidate_record(
                    direction,
                    source_descriptor,
                    fraction,
                    raw_baseline,
                    parameters,
                    optimizer,
                    units,
                    raw_scores,
                    reference_norm,
                    safety_mix,
                    max_norm,
                    clip_epsilon,
                    protected,
                    targets,
                )
                del direction
            candidates.append(candidate)
            if candidate["minimum_protected_retention"] + 1e-12 >= protected_retention:
                protected_candidates.append(candidate)

    selected_record = max(protected_candidates, key=_selection_rank)
    if selected_record is raw_record:
        selected_direction = {name: value.clone() for name, value in full_direction.items()}
    else:
        endpoint = source_directions[int(selected_record["source_mask"])]
        selected_direction = pfrsc._homotopy_direction(
            full_direction,
            endpoint,
            float(selected_record["homotopy_fraction"]),
            budget_norm,
            dual_tolerance,
        )
    ordered = sorted(protected_candidates, key=_selection_rank, reverse=True)
    audit = {
        "certificate_names": list(names),
        "certificate_norms": certificate_norms,
        "protected_certificates": list(protected),
        "target_certificates": list(targets),
        "attempted_source_subsets": full_mask,
        "feasible_source_subsets": len(source_records) + 1,
        "infeasible_source_subsets": infeasible_subsets,
        "source_top_k": source_top_k,
        "screened_source_count": len(screened),
        "attempted_homotopy_candidates": 1
        + len(screened) * (len(fractions) - 1),
        "feasible_homotopy_candidates": len(candidates),
        "infeasible_homotopy_interiors": interior_failures,
        "exact_map_evaluations": len(source_records)
        + 1
        + len(screened) * (len(fractions) - 2)
        - interior_failures,
        "protected_retention_floor": protected_retention,
        "protected_feasible_candidates": len(protected_candidates),
        "raw_anchor": raw_record,
        "selected": selected_record,
        "selected_target_gain_over_raw": float(
            selected_record["minimum_target_exact_score"]
            - raw_record["minimum_target_exact_score"]
        ),
        "top_protected_candidates": ordered[: min(serialized_top_k, len(ordered))],
        "screened_source_endpoints": screened,
        "baseline": {
            **baseline_audit,
            "displacement_norm": reference_norm,
            "scores": _score_displacement(units, baseline_displacement, reference_norm)[
                "scores"
            ],
        },
        "budget_norm": budget_norm,
        "current_optimizer_completed_steps": int(
            baseline_audit["completed_optimizer_steps"]
        ),
    }
    del baseline_displacement, source_directions
    return selected_direction, full_direction, audit


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != STATUS:
        raise BarpTrainingValidationError("BARP-MCD Q1 protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise BarpTrainingValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("training_authorized") is not True
        or int(protocol.get("optimizer_steps_per_arm", -1)) != 2
        or int(protocol.get("total_authorized_optimizer_steps", -1)) != 4
        or protocol.get("checkpoint_writes_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("development_training_authorized") is not False
        or protocol.get("main_training_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise BarpTrainingValidationError("Q1 authorization boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise BarpTrainingValidationError("entry commit is not an ancestor")
    precursor = etarc_p0.read_json(
        resolve_repo_path(protocol["precursor"]["route_path"])
    )
    if precursor.get("route") != "PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL":
        raise BarpTrainingValidationError("BARP-MCD P0 precursor route differs")

    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 4
        or int(config["optimizer_steps"]) != 2
        or int(config["optimizer_steps_authorized"]) != 4
        or int(config["gradient_accumulation_steps"]) != 4
        or config.get("gradient_measurement_mode") != "eval"
        or config.get("qualification_training_authorized") is not True
        or config.get("scheduler_objects_authorized") is not False
        or config.get("checkpoint_writes_authorized") is not False
        or float(config["weight_decay"]) != 0.0
    ):
        raise BarpTrainingValidationError("Q1 config differs")
    design = protocol["design"]
    if (
        tuple(design["protected_certificates"]) != rp.PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != rp.TARGET_CERTIFICATES
        or tuple(design["homotopy_fractions"]) != rp.HOMOTOPY_FRACTIONS
        or int(design["source_top_k"]) != rp.SOURCE_TOP_K
        or float(design["protected_retention_floor"])
        != rp.PROTECTED_RETENTION_FLOOR
        or design.get("optimizer_map") != "exact_current_state_torch_AdamW"
    ):
        raise BarpTrainingValidationError("Q1 method registry differs")

    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise BarpTrainingValidationError("risk row count differs")
    p0 = etarc_p0.read_json(resolve_repo_path(protocol["precursor"]["protocol_path"]))
    if (
        protocol["selection"]["optimizer_blocks"]
        != p0["selection"]["optimizer_blocks"][:2]
        or protocol["selection"]["diagnostic_block"]
        != p0["selection"]["excluded_source_block"]
    ):
        raise BarpTrainingValidationError("Q1 source block registry differs")
    return protocol, config, targets, risk_rows


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise BarpTrainingValidationError("Q1 requires a clean worktree")
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
            raise BarpTrainingValidationError("prospective Q1 outputs already exist")
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise BarpTrainingValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    sample_ids = [
        str(sample_id)
        for block in protocol["selection"]["optimizer_blocks"]
        for sample_id in block["safety_sample_ids"]
    ] + [
        str(value)
        for value in protocol["selection"]["diagnostic_block"]["safety_sample_ids"]
    ]
    if not all(sample_id in pair_map for sample_id in sample_ids):
        raise BarpTrainingValidationError("Q1 safety pair is absent")
    result = {
        "status": "PASS_BARP_MCD_Q1_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "gpu": torch.cuda.get_device_name(0),
        "arms": list(ARMS),
        "optimizer_steps_per_arm": 2,
        "total_authorized_optimizer_steps": 4,
        "current_state_optimizer_map": True,
        "source_subsets_per_step": 127,
        "homotopy_candidates_per_step": 129,
        "exact_map_evaluations_per_step": 239,
        "checkpoint_writes_authorized": False,
        "candidate_generation_authorized": False,
        "development_training_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _update_map_error(
    parameters: list[tuple[str, Any]],
    before: dict[str, Any],
    predicted: dict[str, Any],
    learning_rate: float,
) -> dict[str, float]:
    squared_error = 0.0
    squared_reference = 0.0
    max_absolute = 0.0
    for name, parameter in parameters:
        observed = (before[name] - parameter.detach()) / learning_rate
        difference = observed - predicted[name]
        squared_error += float(difference.double().square().sum().cpu())
        squared_reference += float(predicted[name].double().square().sum().cpu())
        max_absolute = max(max_absolute, float(difference.abs().max().cpu()))
    return {
        "relative_l2_error": math.sqrt(squared_error)
        / max(math.sqrt(squared_reference), 1e-30),
        "max_absolute_error": max_absolute,
    }


def train_arm(
    arm: str,
    protocol: dict[str, Any],
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    runtime_head: str,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    output = resolve_repo_path(protocol["arm_outputs"][arm])
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise BarpTrainingValidationError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    pair_map, structural = rp.structural_features(tokenizer, targets, config, protocol)
    blocks = protocol["selection"]["optimizer_blocks"]
    diagnostic = protocol["selection"]["diagnostic_block"]
    risk_indices = {
        int(index) for block in blocks for index in block["risk_row_indices"]
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
    initial_diagnostic = rp.evaluate_block_losses(
        model,
        risk_features,
        pair_map,
        [int(value) for value in diagnostic["risk_row_indices"]],
        [str(value) for value in diagnostic["safety_sample_ids"]],
        config,
    )
    optimizer = frozen.make_optimizer(model, config)
    trace: list[dict[str, Any]] = []
    for step_index, block in enumerate(blocks, start=1):
        step_started = time.time()
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
        selected, full, search = enumerate_current_state_homotopy(
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
        direction = full if arm == "raw_mcd" else selected
        applied_gradient = gradient_ops.add_gradients(
            raw_baseline, direction, right_scale=float(config["safety_mix"])
        )
        predicted, map_audit = current_adamw_displacement(
            parameters,
            optimizer,
            applied_gradient,
            float(config["max_grad_norm"]),
            float(config["clip_epsilon"]),
        )
        before = {name: parameter.detach().clone() for name, parameter in parameters}
        for name, parameter in parameters:
            parameter.grad = applied_gradient[name]
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for _, parameter in parameters
        )
        if not finite:
            raise FloatingPointError(f"non-finite Q1 gradient at step {step_index}")
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in parameters],
                float(config["max_grad_norm"]),
            )
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        map_error = _update_map_error(
            parameters, before, predicted, learning_rate
        )
        final_losses = rp.evaluate_block_losses(
            model,
            risk_features,
            pair_map,
            block_risk,
            block_safety,
            config,
        )
        actual_reductions = {
            name: float(coverage["loss_means"][name]) - float(final_losses[name])
            for name in rp.CERTIFICATES
        }
        row = {
            "arm": arm,
            "optimizer_step": step_index,
            "block_id": block["block_id"],
            "risk_row_indices": block_risk,
            "safety_sample_ids": block_safety,
            "coverage": coverage,
            "search": search,
            "applied": "raw_anchor" if arm == "raw_mcd" else "barp_selection",
            "applied_homotopy_fraction": 0.0
            if arm == "raw_mcd"
            else float(search["selected"]["homotopy_fraction"]),
            "map_audit": map_audit,
            "map_error": map_error,
            "gradient_preclip_norm": preclip,
            "learning_rate": learning_rate,
            "final_same_batch_losses": final_losses,
            "actual_same_batch_loss_reductions": actual_reductions,
            "wall_seconds": time.time() - step_started,
        }
        trace.append(row)
        write_jsonl(output / "training_trace.jsonl", trace)
        print(
            f"BARP-Q1 {arm} step {step_index}/2 "
            f"lambda={row['applied_homotopy_fraction']:.3f} "
            f"retention={search['selected']['minimum_protected_retention']:.6f} "
            f"target_gain={search['selected_target_gain_over_raw']:.6f} "
            f"map_error={map_error['relative_l2_error']:.3e} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del (
            gradients,
            raw_baseline,
            selected,
            full,
            direction,
            applied_gradient,
            predicted,
            before,
        )

    final_diagnostic = rp.evaluate_block_losses(
        model,
        risk_features,
        pair_map,
        [int(value) for value in diagnostic["risk_row_indices"]],
        [str(value) for value in diagnostic["safety_sample_ids"]],
        config,
    )
    diagnostic_reductions = {
        name: float(initial_diagnostic[name]) - float(final_diagnostic[name])
        for name in rp.CERTIFICATES
    }
    final_hash = model_common.trainable_state_hash(parameters)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_barp_mcd_q1_arm_manifest_v1",
        "runtime_head": runtime_head,
        "arm": arm,
        "protocol_sha256": etarc_p0.sha256(
            resolve_repo_path(protocol["protocol_path"])
        ),
        "config_sha256": etarc_p0.sha256(
            resolve_repo_path(protocol["config_path"])
        ),
        "completed_steps": len(trace),
        "optimizer_steps": len(trace),
        "skipped_updates": 0,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "block_ids": [row["block_id"] for row in trace],
        "initial_diagnostic": initial_diagnostic,
        "final_diagnostic": final_diagnostic,
        "diagnostic_reductions": diagnostic_reductions,
        "maximum_update_map_relative_l2_error": max(
            row["map_error"]["relative_l2_error"] for row in trace
        ),
        "minimum_raw_anchor_score": min(
            row["search"]["raw_anchor"]["minimum_exact_score"] for row in trace
        ),
        "minimum_selected_protected_retention": min(
            row["search"]["selected"]["minimum_protected_retention"]
            for row in trace
        ),
        "minimum_selected_target_gain": min(
            row["search"]["selected_target_gain_over_raw"] for row in trace
        ),
        "nonzero_barp_choices": sum(
            row["search"]["selected"]["homotopy_fraction"] > 0.0 for row in trace
        )
        if arm == "barp_mcd"
        else 0,
        "all_same_batch_reductions_positive": all(
            value > 0.0
            for row in trace
            for value in row["actual_same_batch_loss_reductions"].values()
        ),
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_memory,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "wall_seconds": time.time() - started,
        "checkpoint_writes": 0,
        "scheduler_objects": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "structural_audit_sha256": training.hash_json(structural),
        "trainable_parameter_inventory": inventory,
    }
    write_json(output / "arm_manifest.json", manifest)
    del optimizer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def compare_manifests(
    manifests: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, bool]:
    tolerance = float(protocol["gates"]["matching_tolerance"])
    map_tolerance = float(protocol["gates"]["optimizer_map_relative_error_max"])
    values = list(manifests.values())
    initial_diagnostic_equal = all(
        abs(
            float(values[0]["initial_diagnostic"][name])
            - float(values[1]["initial_diagnostic"][name])
        )
        <= tolerance
        for name in rp.CERTIFICATES
    )
    barp = manifests["barp_mcd"]
    return {
        "arms_complete": set(manifests) == set(ARMS),
        "steps_exact": all(item["completed_steps"] == 2 for item in values),
        "no_skipped_updates": all(item["skipped_updates"] == 0 for item in values),
        "initial_state_equal": len(
            {item["initial_trainable_state_sha256"] for item in values}
        )
        == 1,
        "block_order_equal": len({tuple(item["block_ids"]) for item in values}) == 1,
        "initial_diagnostic_equal": initial_diagnostic_equal,
        "optimizer_map_exact": all(
            item["maximum_update_map_relative_l2_error"] <= map_tolerance
            for item in values
        ),
        "raw_common_descent": all(
            item["minimum_raw_anchor_score"]
            >= float(protocol["gates"]["raw_anchor_score_min"])
            for item in values
        ),
        "barp_protection": barp["minimum_selected_protected_retention"]
        + tolerance
        >= float(protocol["design"]["protected_retention_floor"]),
        "barp_target_nondegradation": barp["minimum_selected_target_gain"]
        >= -tolerance,
        "barp_assignment_exercised": barp["nonzero_barp_choices"]
        >= int(protocol["gates"]["minimum_nonzero_barp_choices"]),
        "finite_step_common_descent": all(
            item["all_same_batch_reductions_positive"] for item in values
        ),
        "vram": all(
            item["vram_headroom_fraction"]
            >= float(protocol["gates"]["minimum_vram_headroom_fraction"])
            for item in values
        ),
        "boundary": all(
            item["checkpoint_writes"] == 0
            and item["scheduler_objects"] == 0
            and not item["candidate_outputs_generated"]
            and not item["core100_opened"]
            and not item["heldout_opened"]
            for item in values
        ),
    }


def select_route(checks: dict[str, bool]) -> str:
    order = (
        ("arms_complete", "STOP_BARP_MCD_Q1_INCOMPLETE"),
        ("steps_exact", "STOP_BARP_MCD_Q1_STEP_COUNT_FAILED"),
        ("no_skipped_updates", "STOP_BARP_MCD_Q1_NUMERICAL_FAILED"),
        ("initial_state_equal", "STOP_BARP_MCD_Q1_MATCHING_FAILED"),
        ("block_order_equal", "STOP_BARP_MCD_Q1_MATCHING_FAILED"),
        ("initial_diagnostic_equal", "STOP_BARP_MCD_Q1_MATCHING_FAILED"),
        ("optimizer_map_exact", "STOP_BARP_MCD_Q1_OPTIMIZER_MAP_FAILED"),
        ("raw_common_descent", "STOP_BARP_MCD_Q1_RAW_ANCHOR_FAILED"),
        ("barp_protection", "STOP_BARP_MCD_Q1_PROTECTION_FAILED"),
        ("barp_target_nondegradation", "STOP_BARP_MCD_Q1_TARGET_FAILED"),
        ("barp_assignment_exercised", "STOP_BARP_MCD_Q1_ASSIGNMENT_NOT_EXERCISED"),
        ("finite_step_common_descent", "STOP_BARP_MCD_Q1_FINITE_STEP_FAILED"),
        ("vram", "STOP_BARP_MCD_Q1_VRAM_FAILED"),
        ("boundary", "STOP_BARP_MCD_Q1_BOUNDARY_FAILED"),
    )
    for gate, route in order:
        if not checks.get(gate, False):
            return route
    return "PASS_BARP_MCD_Q1_AUTHORIZE_LONGER_MATCHED_PROTOCOL"


def package_artifacts(root: Path, reproduction: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == reproduction or "protocol" in path.parts:
            continue
        artifacts.append(
            {
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

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_start(path, protocol)
    started = time.time()
    manifests = {
        arm: train_arm(
            arm, protocol, config, targets, risk_rows, runtime_head
        )
        for arm in ARMS
    }
    checks = compare_manifests(manifests, protocol)
    route = select_route(checks)
    decision = {
        "name": "step3_barp_mcd_q1_decision_v1",
        "route": route,
        "checks": checks,
        "arms": {
            arm: {
                "optimizer_steps": manifest["optimizer_steps"],
                "final_trainable_state_sha256": manifest[
                    "final_trainable_state_sha256"
                ],
                "maximum_update_map_relative_l2_error": manifest[
                    "maximum_update_map_relative_l2_error"
                ],
                "minimum_raw_anchor_score": manifest["minimum_raw_anchor_score"],
                "minimum_selected_protected_retention": manifest[
                    "minimum_selected_protected_retention"
                ],
                "minimum_selected_target_gain": manifest[
                    "minimum_selected_target_gain"
                ],
                "nonzero_barp_choices": manifest["nonzero_barp_choices"],
                "diagnostic_reductions": manifest["diagnostic_reductions"],
                "vram_headroom_fraction": manifest["vram_headroom_fraction"],
            }
            for arm, manifest in manifests.items()
        },
        "total_optimizer_steps": sum(item["optimizer_steps"] for item in manifests.values()),
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "longer_training_authorized": route
        == "PASS_BARP_MCD_Q1_AUTHORIZE_LONGER_MATCHED_PROTOCOL",
        "main_training_authorized": False,
        "wall_seconds": time.time() - started,
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    write_json(decision_path, decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(
        "# BARP-MCD Q1 Matched Training Qualification\n\n"
        f"Decision: `{route}`\n\n"
        f"Total optimizer steps: `{decision['total_optimizer_steps']}`. "
        "Both arms used the same two disclosed seed-4 optimizer blocks.\n\n"
        "The current-state AdamW map was checked against every real update. "
        "The excluded eight-category diagnostic is descriptive and did not "
        "select the route.\n\n"
        "No checkpoint or candidate response was generated; Core100 and "
        "heldout remained sealed. A pass authorizes only a separately frozen "
        "longer matched-training protocol.\n",
        encoding="utf-8",
    )
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    reproduction = {
        "name": "step3_barp_mcd_q1_reproduction_manifest_v1",
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_training.py preflight",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_training.py run",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_training.py verify-run",
        ],
        "route": route,
        "total_optimizer_steps": decision["total_optimizer_steps"],
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
    protocol, _, _, _ = context(path)
    manifests = {
        arm: etarc_p0.read_json(
            resolve_repo_path(protocol["arm_outputs"][arm]) / "arm_manifest.json"
        )
        for arm in ARMS
    }
    checks = compare_manifests(manifests, protocol)
    route = select_route(checks)
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    reproduction = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if decision["checks"] != checks or decision["route"] != route:
        raise BarpTrainingValidationError("Q1 decision does not reproduce")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise BarpTrainingValidationError(
                f"Q1 artifact hash mismatch: {artifact['path']}"
            )
    result = {
        "status": "PASS_BARP_MCD_Q1_VERIFY",
        "route": route,
        "checks": checks,
        "artifacts_verified": len(reproduction["artifacts"]),
        "total_optimizer_steps": decision["total_optimizer_steps"],
        "checkpoint_writes": 0,
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
