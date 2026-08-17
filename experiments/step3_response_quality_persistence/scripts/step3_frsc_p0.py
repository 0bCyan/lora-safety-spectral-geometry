"""Run seed-2 finite-radius subset-certified MCD without optimizer steps."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_mcd_training as smoke
import step3_ojmcd_p0 as optimizer_unit
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_ramp_training as base
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_frsc_p0/protocol/frsc_p0_protocol.json"
)
CERTIFICATES = optimizer_unit.CERTIFICATES
SAFETY_OBJECTIVES = optimizer_unit.SAFETY_OBJECTIVES
ARMS = ("baseline", "raw_mcd", "frsc_mcd")


class FrscP0ValidationError(ValueError):
    """Raised when FRSC-MCD P0 violates its prospective protocol."""


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
        raise FrscP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != "prospective_before_any_FRSC_model_forward":
        raise FrscP0ValidationError("protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or etarc_p0.sha256(artifact_path) != artifact["sha256"]
        ):
            raise FrscP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise FrscP0ValidationError("protocol violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise FrscP0ValidationError("entry commit is not an ancestor of HEAD")

    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_OJMCD_P0_DUAL_CERTIFICATE_FAILED":
        raise FrscP0ValidationError("registered OJ-MCD precursor route differs")
    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 2
        or config.get("model_optimizer_objects_authorized") is not False
        or config.get("scheduler_objects_authorized") is not False
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("transient_virtual_parameter_evaluations_authorized")
        is not True
        or config.get("training_authorized") is not False
        or float(config["weight_decay"]) != 0.0
        or int(config["gradient_accumulation_steps"]) != 4
    ):
        raise FrscP0ValidationError("seed-2 virtual-step config differs")

    targets = etarc_p0.read_jsonl(
        resolve_repo_path(config["safety_targets_path"])
    )
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise FrscP0ValidationError("risk row count differs")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(protocol["selection"]["risk_order_seed"]))
    risk_order = torch.randperm(len(risk_rows), generator=generator).tolist()
    blocks = list(protocol["selection"]["training_blocks"])
    if len(blocks) != 8:
        raise FrscP0ValidationError("training block count differs")
    flat_safety = [
        str(sample_id) for block in blocks for sample_id in block["safety_sample_ids"]
    ]
    flat_risk = [
        int(index) for block in blocks for index in block["risk_row_indices"]
    ]
    if (
        any(len(block["safety_sample_ids"]) != 4 for block in blocks)
        or any(len(block["risk_row_indices"]) != 4 for block in blocks)
        or flat_risk != risk_order[:32]
    ):
        raise FrscP0ValidationError("optimizer-unit training blocks differ")
    diagnostics = protocol["selection"]["diagnostic_block"]
    diagnostic_safety = [str(value) for value in diagnostics["safety_sample_ids"]]
    diagnostic_risk = [int(value) for value in diagnostics["risk_row_indices"]]
    if (
        len(diagnostic_safety) != 8
        or len(diagnostic_risk) != 8
        or diagnostic_risk != risk_order[32:40]
        or set(flat_safety).intersection(diagnostic_safety)
        or set(flat_risk).intersection(diagnostic_risk)
        or len(set(flat_safety + diagnostic_safety)) != 40
        or len(set(flat_risk + diagnostic_risk)) != 40
    ):
        raise FrscP0ValidationError("training/diagnostic split overlaps or drifts")
    source_ids = {str(row["sample_id"]) for row in targets}
    if set(flat_safety + diagnostic_safety) != source_ids:
        raise FrscP0ValidationError("safety split does not cover the frozen source")
    by_id = {str(row["sample_id"]): str(row["category"]) for row in targets}
    categories = [by_id[sample_id] for sample_id in diagnostic_safety]
    if categories != list(diagnostics["safety_categories"]):
        raise FrscP0ValidationError("diagnostic category registry differs")
    if len(set(categories)) != 8:
        raise FrscP0ValidationError("diagnostic set does not cover eight categories")
    return protocol, config, targets, risk_rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise FrscP0ValidationError("FRSC-MCD P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([protocol["config_path"], relative(path)])
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
            raise FrscP0ValidationError(f"prospective output already exists: {output}")
    return etarc_p0.git_output("rev-parse", "HEAD")


def structural_features(
    tokenizer: Any,
    targets: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        targets,
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    expected = etarc_p0.read_json(
        resolve_repo_path(protocol["source"]["structural_audit_path"])
    )
    if audit != expected:
        raise FrscP0ValidationError("route structure differs from frozen source")
    return {str(pair["sample_id"]): pair for pair in pairs}, audit


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise FrscP0ValidationError("CUDA is unavailable")
    tokenizer = tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = structural_features(tokenizer, targets, config, protocol)
    selected = [
        sample_id
        for block in protocol["selection"]["training_blocks"]
        for sample_id in block["safety_sample_ids"]
    ] + list(protocol["selection"]["diagnostic_block"]["safety_sample_ids"])
    if not all(sample_id in pair_map for sample_id in selected):
        raise FrscP0ValidationError("selected safety pair is absent")
    result = {
        "status": "PASS_FRSC_MCD_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "seed": config["seed"],
        "certificates": list(CERTIFICATES),
        "arms": list(ARMS),
        "training_optimizer_units": 8,
        "risk_microbatches_per_unit": 4,
        "safety_microbatches_per_unit": 4,
        "diagnostic_risk_rows": 8,
        "diagnostic_safety_prompt_pairs": 8,
        "diagnostic_categories": 8,
        "training_diagnostic_overlap": 0,
        "candidate_subsets_per_unit": 127,
        "virtual_parameter_evaluations": 24,
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


def _zero_gradient(parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    return {name: torch.zeros_like(parameter) for name, parameter in parameters}


def aggregate_block_gradients(
    model: Any,
    parameters: list[tuple[str, Any]],
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    risk_indices: list[int],
    safety_ids: list[str],
    config: dict[str, Any],
    train_mode: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(risk_indices) != len(safety_ids) or not risk_indices:
        raise FrscP0ValidationError("gradient block inputs differ")
    risk = _zero_gradient(parameters)
    safety = {name: _zero_gradient(parameters) for name in SAFETY_OBJECTIVES}
    losses: dict[str, list[float]] = {name: [] for name in CERTIFICATES}
    device = training.model_device(model)
    response_tokens = 0
    continuation_tokens = 0
    tarc_terms = 0
    etarc_terms = 0
    model.train(train_mode)
    for risk_index, sample_id in zip(risk_indices, safety_ids, strict=True):
        risk_loss = base.risk_loss(
            model,
            risk_features[int(risk_index)],
            device,
            bool(config["fp16"]),
        )
        risk_gradient = smoke.gradient_from_loss(risk_loss, parameters, "risk")
        for name in risk:
            risk[name].add_(risk_gradient[name])
        losses["risk"].append(float(risk_loss.detach().float().cpu()))

        current_losses, metrics = smoke.forward_pair_objectives(
            model,
            pair_map[str(sample_id)],
            device,
            bool(config["fp16"]),
            int(config["ramp_width"]),
            float(config["softmin_temperature"]),
        )
        objective_gradients = smoke.gradients_from_objectives(
            current_losses, parameters
        )
        for objective in SAFETY_OBJECTIVES:
            for name in safety[objective]:
                safety[objective][name].add_(objective_gradients[objective][name])
            losses[objective].append(float(metrics[f"{objective}_loss"]))
        response_tokens += int(metrics["response_tokens"])
        continuation_tokens += int(metrics["continuation_tokens"])
        tarc_terms += int(metrics["tarc_margin_terms"])
        etarc_terms += int(metrics["etarc_margin_terms"])
        del risk_loss, risk_gradient, current_losses, objective_gradients

    scale = 1.0 / len(risk_indices)
    for value in risk.values():
        value.mul_(scale)
    for gradient in safety.values():
        for value in gradient.values():
            value.mul_(scale)
    gradients = {"risk": risk, **safety}
    norms = {
        name: gradient_math.gradient_norm(gradient)
        for name, gradient in gradients.items()
    }
    return gradients, {
        "mode": "train" if train_mode else "eval",
        "risk_microbatches": len(risk_indices),
        "safety_microbatches": len(safety_ids),
        "loss_means": {
            name: statistics.fmean(values) for name, values in losses.items()
        },
        "gradient_norms": norms,
        "all_gradients_finite_nonzero": all(
            math.isfinite(value) and value > 0.0 for value in norms.values()
        ),
        "response_tokens": response_tokens,
        "continuation_tokens": continuation_tokens,
        "tarc_margin_terms": tarc_terms,
        "etarc_margin_terms": etarc_terms,
    }


def evaluate_block_losses(
    model: Any,
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    risk_indices: list[int],
    safety_ids: list[str],
    config: dict[str, Any],
) -> dict[str, float]:
    import torch

    device = training.model_device(model)
    losses: dict[str, list[float]] = {name: [] for name in CERTIFICATES}
    model.eval()
    with torch.no_grad():
        for risk_index in risk_indices:
            value = base.risk_loss(
                model,
                risk_features[int(risk_index)],
                device,
                bool(config["fp16"]),
            )
            losses["risk"].append(float(value.detach().float().cpu()))
        for sample_id in safety_ids:
            _, metrics = smoke.forward_pair_objectives(
                model,
                pair_map[str(sample_id)],
                device,
                bool(config["fp16"]),
                int(config["ramp_width"]),
                float(config["softmin_temperature"]),
            )
            for objective in SAFETY_OBJECTIVES:
                losses[objective].append(float(metrics[f"{objective}_loss"]))
    return {name: statistics.fmean(values) for name, values in losses.items()}


def _snapshot(parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in parameters}


def _restore(
    parameters: list[tuple[str, Any]], state: dict[str, Any]
) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(state[name])


def _apply_virtual_step(
    parameters: list[tuple[str, Any]],
    state: dict[str, Any],
    displacement: dict[str, Any],
    learning_rate: float,
) -> None:
    import torch

    _restore(parameters, state)
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.add_(displacement[name], alpha=-learning_rate)


def _state_equal(
    parameters: list[tuple[str, Any]], state: dict[str, Any]
) -> bool:
    import torch

    return all(torch.equal(parameter.detach(), state[name]) for name, parameter in parameters)


def _baseline_gradient(
    gradients: dict[str, dict[str, Any]], safety_mix: float
) -> dict[str, Any]:
    return gradient_math.add_gradients(
        gradient_math.scaled_gradient(gradients["risk"], 1.0 - safety_mix),
        gradient_math.scaled_gradient(gradients["uniform"], safety_mix),
    )


def evaluate_arm(
    arm: str,
    raw_baseline: dict[str, Any],
    auxiliary: dict[str, Any] | None,
    diagnostic_gradients: dict[str, dict[str, Any]],
    initial_losses: dict[str, float],
    model: Any,
    parameters: list[tuple[str, Any]],
    initial_state: dict[str, Any],
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    config: dict[str, Any],
    reference_norm: float,
) -> dict[str, Any]:
    displacement, map_audit = frsc.realized_displacement(
        raw_baseline,
        auxiliary,
        float(config["safety_mix"]) if auxiliary is not None else 0.0,
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    score = frsc.displacement_scores(
        diagnostic_gradients, displacement, reference_norm
    )
    _apply_virtual_step(
        parameters,
        initial_state,
        displacement,
        float(config["learning_rate"]),
    )
    final_losses = evaluate_block_losses(
        model,
        risk_features,
        pair_map,
        [int(value) for value in diagnostic["risk_row_indices"]],
        [str(value) for value in diagnostic["safety_sample_ids"]],
        config,
    )
    finite = frsc.finite_step_loss_metrics(
        initial_losses,
        final_losses,
        diagnostic_gradients,
        displacement,
        float(config["learning_rate"]),
    )
    _restore(parameters, initial_state)
    restored = _state_equal(parameters, initial_state)
    del displacement
    return {
        "arm": arm,
        "optimizer_map": map_audit,
        "diagnostic_scores": score,
        "finite_step": finite,
        "state_restored_after_virtual_step": restored,
    }


def _paired(values: list[float], protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "differences": values,
        "positive_rate": sum(value > 0.0 for value in values) / len(values),
        "bootstrap": etarc_p0.paired_bootstrap(
            values,
            int(protocol["statistics"]["bootstrap_samples"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        ),
        "sign_test": etarc_p0.exact_sign_test(values),
    }


def summarize_rows(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    if len(rows) != 8:
        raise FrscP0ValidationError("stored training block count differs")
    search_gains = [
        float(row["search"]["selected_margin_gain_over_full"]) for row in rows
    ]
    search = {
        "minimum_selected_training_margin": min(
            float(row["search"]["selected"]["minimum_exact_score"])
            for row in rows
        ),
        "minimum_full_training_margin": min(
            float(row["search"]["full_certificate_control"]["minimum_exact_score"])
            for row in rows
        ),
        "selected_margin_gain_over_full": _paired(search_gains, protocol),
        "selected_subset_counts": dict(
            Counter("+".join(row["search"]["selected"]["subset"]) for row in rows)
        ),
    }
    diagnostic: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows]
        diagnostic[arm] = {
            "minimum_worst_score": min(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "mean_worst_score": statistics.fmean(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "minimum_certificate_scores": {
                certificate: min(
                    float(item["diagnostic_scores"]["scores"][certificate])
                    for item in arm_rows
                )
                for certificate in CERTIFICATES
            },
            "objectives": {},
        }
        for certificate in CERTIFICATES:
            objective_rows = [
                item["finite_step"]["objectives"][certificate]
                for item in arm_rows
            ]
            ratios = [
                float(item["actual_to_predicted_ratio"])
                for item in objective_rows
                if item["actual_to_predicted_ratio"] is not None
            ]
            diagnostic[arm]["objectives"][certificate] = {
                "mean_predicted_reduction": statistics.fmean(
                    float(item["predicted_first_order_reduction"])
                    for item in objective_rows
                ),
                "mean_actual_reduction": statistics.fmean(
                    float(item["actual_loss_reduction"]) for item in objective_rows
                ),
                "minimum_actual_reduction": min(
                    float(item["actual_loss_reduction"]) for item in objective_rows
                ),
                "positive_actual_rate": sum(
                    float(item["actual_loss_reduction"]) > 0.0
                    for item in objective_rows
                )
                / len(objective_rows),
                "mean_actual_to_predicted_ratio": statistics.fmean(ratios)
                if ratios
                else None,
            }

    worst = {
        arm: [
            float(row["arms"][arm]["diagnostic_scores"]["minimum_score"])
            for row in rows
        ]
        for arm in ARMS
    }
    paired = {
        "frsc_minus_baseline_worst_score": _paired(
            [left - right for left, right in zip(worst["frsc_mcd"], worst["baseline"], strict=True)],
            protocol,
        ),
        "frsc_minus_raw_mcd_worst_score": _paired(
            [left - right for left, right in zip(worst["frsc_mcd"], worst["raw_mcd"], strict=True)],
            protocol,
        ),
        "frsc_minus_raw_actual_reduction": {},
    }
    for certificate in CERTIFICATES:
        differences = [
            float(row["arms"]["frsc_mcd"]["finite_step"]["objectives"][certificate]["actual_loss_reduction"])
            - float(row["arms"]["raw_mcd"]["finite_step"]["objectives"][certificate]["actual_loss_reduction"])
            for row in rows
        ]
        paired["frsc_minus_raw_actual_reduction"][certificate] = _paired(
            differences, protocol
        )
    return {"search": search, "diagnostic": diagnostic, "paired": paired}


def evaluate_gates(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    state_restored: bool,
    vram_headroom: float,
) -> dict[str, bool]:
    gates = protocol["gates"]
    search_pair = summary["search"]["selected_margin_gain_over_full"]
    baseline_pair = summary["paired"]["frsc_minus_baseline_worst_score"]
    raw_pair = summary["paired"]["frsc_minus_raw_mcd_worst_score"]
    frsc_diag = summary["diagnostic"]["frsc_mcd"]
    objective_rows = list(frsc_diag["objectives"].values())
    actual_advantages = summary["paired"]["frsc_minus_raw_actual_reduction"]
    objectives_noninferior = sum(
        float(item["bootstrap"]["mean"])
        >= -float(gates["actual_reduction_noninferiority_tolerance"])
        for item in actual_advantages.values()
    )
    return {
        "integrity": (
            len(rows) == 8
            and all(row["coverage"]["risk_microbatches"] == 4 for row in rows)
            and all(row["coverage"]["safety_microbatches"] == 4 for row in rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in rows)
            and all(row["search"]["attempted_subsets"] == 127 for row in rows)
            and all(
                row["search"]["feasible_subsets"]
                + row["search"]["infeasible_subsets"]
                == 127
                for row in rows
            )
            and all(
                bool(row["search"]["baseline"]["clipping_active"])
                and bool(row["search"]["selected"]["candidate_clipping_active"])
                and bool(
                    row["search"]["full_certificate_control"][
                        "candidate_clipping_active"
                    ]
                )
                and float(
                    row["search"]["selected"][
                        "direction_budget_relative_error"
                    ]
                )
                <= float(gates["gradient_budget_relative_tolerance"])
                and float(
                    row["search"]["full_certificate_control"][
                        "direction_budget_relative_error"
                    ]
                )
                <= float(gates["gradient_budget_relative_tolerance"])
                for row in rows
            )
            and all(
                all(
                    arm["state_restored_after_virtual_step"]
                    for arm in row["arms"].values()
                )
                for row in rows
            )
            and state_restored
            and vram_headroom >= float(gates["minimum_vram_headroom_fraction"])
        ),
        "finite_radius_search": (
            float(summary["search"]["minimum_selected_training_margin"])
            >= float(gates["training_exact_margin_min"])
            and float(search_pair["bootstrap"]["mean"])
            >= float(gates["search_gain_over_full_mean_min"])
            and float(search_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(search_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
        ),
        "diagnostic_common_transfer": (
            float(frsc_diag["minimum_worst_score"])
            >= float(gates["diagnostic_exact_margin_min"])
            and all(
                float(value) >= float(gates["diagnostic_certificate_score_min"])
                for value in frsc_diag["minimum_certificate_scores"].values()
            )
            and float(baseline_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_advantage_over_baseline_mean_min"])
            and float(baseline_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(baseline_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
        ),
        "finite_step_common_descent": all(
            float(item["mean_actual_reduction"])
            > float(gates["mean_actual_loss_reduction_min"])
            and float(item["positive_actual_rate"])
            >= float(gates["actual_reduction_positive_rate_min"])
            and item["mean_actual_to_predicted_ratio"] is not None
            and float(item["mean_actual_to_predicted_ratio"])
            >= float(gates["mean_actual_to_predicted_ratio_min"])
            and float(item["mean_actual_to_predicted_ratio"])
            <= float(gates["mean_actual_to_predicted_ratio_max"])
            for item in objective_rows
        ),
        "raw_comparator_advantage": (
            float(raw_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_advantage_over_raw_mean_min"])
            and float(raw_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(raw_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and objectives_noninferior
            >= int(gates["minimum_noninferior_actual_objectives"])
        ),
        "effect_not_weak": (
            float(frsc_diag["mean_worst_score"])
            >= float(gates["diagnostic_mean_worst_score_min"])
            and float(baseline_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_advantage_over_baseline_mean_min"])
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"]:
        return "STOP_FRSC_P0_INVALID"
    if not gates["finite_radius_search"]:
        return "STOP_FRSC_P0_SEARCH_GAIN_FAILED"
    if not gates["diagnostic_common_transfer"]:
        return "STOP_FRSC_P0_DIAGNOSTIC_TRANSFER_FAILED"
    if not gates["finite_step_common_descent"]:
        return "STOP_FRSC_P0_FINITE_STEP_COMMON_DESCENT_FAILED"
    if not gates["raw_comparator_advantage"]:
        return "STOP_FRSC_P0_RAW_COMPARATOR_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_FRSC_P0_EFFECT_TOO_WEAK"
    return "PASS_FRSC_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL"


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
    from transformers import set_seed

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
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
        raise FrscP0ValidationError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    initial_state = _snapshot(parameters)
    pair_map, structural = structural_features(tokenizer, targets, config, protocol)
    selected_risk = {
        int(index)
        for block in protocol["selection"]["training_blocks"]
        for index in block["risk_row_indices"]
    }
    selected_risk.update(
        int(index)
        for index in protocol["selection"]["diagnostic_block"]["risk_row_indices"]
    )
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in selected_risk
    }

    diagnostic = protocol["selection"]["diagnostic_block"]
    diagnostic_gradients, diagnostic_coverage = aggregate_block_gradients(
        model,
        parameters,
        risk_features,
        pair_map,
        [int(value) for value in diagnostic["risk_row_indices"]],
        [str(value) for value in diagnostic["safety_sample_ids"]],
        config,
        train_mode=False,
    )
    initial_losses = dict(diagnostic_coverage["loss_means"])
    if not _state_equal(parameters, initial_state):
        raise FrscP0ValidationError("diagnostic gradients changed parameters")

    rows = []
    for block in protocol["selection"]["training_blocks"]:
        block_started = time.time()
        gradients, coverage = aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(value) for value in block["risk_row_indices"]],
            [str(value) for value in block["safety_sample_ids"]],
            config,
            train_mode=True,
        )
        raw_baseline = _baseline_gradient(gradients, float(config["safety_mix"]))
        budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
            gradients["uniform"]
        )
        selected_direction, full_direction, search = (
            frsc.enumerate_finite_radius_subsets(
                raw_baseline,
                {name: gradients[name] for name in CERTIFICATES},
                budget,
                float(config["safety_mix"]),
                float(config["max_grad_norm"]),
                float(config["adam_epsilon"]),
                float(config["clip_epsilon"]),
                float(config["dual_numeric_tolerance"]),
                int(protocol["design"]["top_candidates_serialized"]),
            )
        )
        baseline_displacement, _ = frsc.realized_displacement(
            raw_baseline,
            None,
            0.0,
            float(config["max_grad_norm"]),
            float(config["adam_epsilon"]),
            float(config["clip_epsilon"]),
        )
        reference_norm = gradient_math.gradient_norm(baseline_displacement)
        del baseline_displacement
        arms = {
            "baseline": evaluate_arm(
                "baseline",
                raw_baseline,
                None,
                diagnostic_gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                diagnostic,
                config,
                reference_norm,
            ),
            "raw_mcd": evaluate_arm(
                "raw_mcd",
                raw_baseline,
                full_direction,
                diagnostic_gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                diagnostic,
                config,
                reference_norm,
            ),
            "frsc_mcd": evaluate_arm(
                "frsc_mcd",
                raw_baseline,
                selected_direction,
                diagnostic_gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                diagnostic,
                config,
                reference_norm,
            ),
        }
        row = {
            "block_id": block["block_id"],
            "risk_row_indices": block["risk_row_indices"],
            "safety_sample_ids": block["safety_sample_ids"],
            "coverage": coverage,
            "auxiliary_budget_norm": budget,
            "search": search,
            "arms": arms,
            "wall_seconds": time.time() - block_started,
        }
        rows.append(row)
        del gradients, raw_baseline, selected_direction, full_direction
        print(
            f"FRSC {block['block_id']} train_margin="
            f"{search['selected']['minimum_exact_score']:.6f} "
            f"gain={search['selected_margin_gain_over_full']:.6f} "
            f"diag_margin={arms['frsc_mcd']['diagnostic_scores']['minimum_score']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_hash = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    summary = summarize_rows(rows, protocol)
    gates = evaluate_gates(
        rows, summary, protocol, initial_hash == final_hash, headroom
    )
    route = select_route(gates)
    outputs = {
        name: resolve_repo_path(value) for name, value in protocol["outputs"].items()
    }
    diagnostics_payload = {
        "name": "step3_frsc_mcd_p0_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "diagnostic_coverage": diagnostic_coverage,
        "diagnostic_sample_ids": diagnostic["safety_sample_ids"],
        "diagnostic_categories": diagnostic["safety_categories"],
        "summary": summary,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "state_restored": initial_hash == final_hash,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
    }
    environment = {
        "name": "step3_frsc_mcd_p0_environment_v1",
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
        "name": "step3_frsc_mcd_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "summary": summary,
        "matched_training_protocol_authorized": route
        == "PASS_FRSC_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "checkpoint_writes": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    search_pair = summary["search"]["selected_margin_gain_over_full"]
    raw_pair = summary["paired"]["frsc_minus_raw_mcd_worst_score"]
    report = f"""# FRSC-MCD Seed-2 Zero-Step P0 Report

Route: `{route}`.

Eight optimizer-unit training blocks independently enumerated all 127
nonempty subsets of the seven registered certificates. The minimum selected
training exact margin was
`{summary['search']['minimum_selected_training_margin']:.6f}`. The mean
selected-minus-full-seven margin gain was
`{search_pair['bootstrap']['mean']:.6f}` with 95% bootstrap interval
`[{search_pair['bootstrap']['ci95_lower']:.6f},
{search_pair['bootstrap']['ci95_upper']:.6f}]`.

The fixed eight-category diagnostic set was excluded from every direction
construction block. FRSC-MCD's minimum diagnostic exact score was
`{summary['diagnostic']['frsc_mcd']['minimum_worst_score']:.6f}`. Its mean
paired diagnostic worst-score advantage over full-seven raw MCD was
`{raw_pair['bootstrap']['mean']:.6f}` with exact sign-test
`p={raw_pair['sign_test']['exact_two_sided_p']:.8g}`.

All reported finite loss changes came from transient direct applications of
the exact first-step displacement at learning rate
`{config['learning_rate']}` followed by exact parameter restoration. No model
optimizer or scheduler was created, `optimizer_steps=0`, and no checkpoint,
candidate output, Core100 row, or heldout row was opened.
"""
    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_jsonl_frozen(outputs["block_rows"], rows)
    etarc_p0.write_json_frozen(outputs["gradient_diagnostics"], diagnostics_payload)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_frsc_mcd_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_frsc_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_frsc_p0.py verify-run"
        ),
        "seed": config["seed"],
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "artifacts": package_artifacts(protocol),
    }
    etarc_p0.write_json_frozen(outputs["run_manifest"], manifest)
    del model, diagnostic_gradients, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    rows = etarc_p0.read_jsonl(resolve_repo_path(protocol["outputs"]["block_rows"]))
    diagnostics = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["gradient_diagnostics"])
    )
    environment = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["environment"])
    )
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["route"]))
    manifest = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["run_manifest"])
    )
    summary = summarize_rows(rows, protocol)
    gates = evaluate_gates(
        rows,
        summary,
        protocol,
        bool(diagnostics["state_restored"]),
        float(environment["vram_headroom_fraction"]),
    )
    route = select_route(gates)
    if (
        decision.get("route") != route
        or decision.get("gates") != gates
        or decision.get("summary") != summary
        or diagnostics.get("summary") != summary
        or manifest.get("route") != route
    ):
        raise FrscP0ValidationError("stored and recomputed FRSC route differs")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise FrscP0ValidationError("FRSC artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise FrscP0ValidationError("stored FRSC boundary differs")
    result = {
        "status": "PASS_FRSC_MCD_P0_VERIFY",
        "route": route,
        "training_blocks": len(rows),
        "virtual_parameter_evaluations": manifest["virtual_parameter_evaluations"],
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
