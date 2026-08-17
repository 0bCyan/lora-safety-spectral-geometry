"""Run seed-6 minimum-effective split-conformal MCD without optimizer steps."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
import step3_jcrmcd_common as path_geometry
import step3_jcrmcd_p0 as jcr_runner
import step3_medscmcd_common as medsc
import step3_ojmcd_p0 as optimizer_unit
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as model_runner
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_medscmcd_p0/protocol/medscmcd_p0_protocol.json"
)
CERTIFICATES = optimizer_unit.CERTIFICATES
SAFETY_OBJECTIVES = optimizer_unit.SAFETY_OBJECTIVES
ARMS = ("baseline", "raw_mcd", "medscmcd_mcd")
PROTECTED_CERTIFICATES = ("risk", "uniform", "tarc", "continuation")
TARGET_CERTIFICATES = ("etarc", "branch_entropy", "branch_negative_gap")
PATH_FRACTIONS = tuple(index / 32.0 for index in range(33))
PROTECTED_RETENTION_FLOOR = 0.95
FIT_UNITS = 12
CALIBRATION_UNITS = 20
DIAGNOSTIC_UNITS = 8
SPLIT_SALT = "medscmcd-seed6-split-v1"
CONFORMAL_ALPHA = 0.05
CONFORMAL_ACCEPTANCE_THRESHOLD = 0.0
FIT_TARGET_GAIN_FLOOR = 0.002
FIT_TARGET_SCORE_FLOOR = 0.05


class MedscmcdP0ValidationError(ValueError):
    """Raised when MED-SC-MCD P0 violates its prospective protocol."""


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
        raise MedscmcdP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expected_roles(
    targets: list[dict[str, Any]], risk_order: list[int]
) -> dict[str, list[dict[str, Any]]]:
    registry = jcr_runner.prompt_category_registry(targets)
    ordered = sorted(
        registry,
        key=lambda sample_id: _stable_digest(f"{SPLIT_SALT}|{sample_id}"),
    )
    if len(ordered) != FIT_UNITS + CALIBRATION_UNITS + DIAGNOSTIC_UNITS:
        raise MedscmcdP0ValidationError("prompt role count differs")
    if len(risk_order) < len(ordered):
        raise MedscmcdP0ValidationError("risk role permutation is too short")

    boundaries = (
        ("fit", 0, FIT_UNITS),
        ("calibration", FIT_UNITS, FIT_UNITS + CALIBRATION_UNITS),
        (
            "diagnostic",
            FIT_UNITS + CALIBRATION_UNITS,
            FIT_UNITS + CALIBRATION_UNITS + DIAGNOSTIC_UNITS,
        ),
    )
    roles = {}
    for role, start, end in boundaries:
        roles[role] = [
            {
                "unit_id": f"{role}_{offset + 1:02d}",
                "risk_row_index": int(risk_order[index]),
                "sample_id": ordered[index],
                "category": registry[ordered[index]],
            }
            for offset, index in enumerate(range(start, end))
        ]
    return roles


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != "prospective_before_any_MEDSCMCD_model_forward":
        raise MedscmcdP0ValidationError("MED-SC-MCD protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or etarc_p0.sha256(artifact_path) != artifact["sha256"]
        ):
            raise MedscmcdP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("virtual_parameter_evaluations") != 24
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise MedscmcdP0ValidationError("protocol violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise MedscmcdP0ValidationError("entry commit is not an ancestor of HEAD")

    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_JCRMCD_P0_CONSTRUCTION_UNCERTAINTY_INVALID":
        raise MedscmcdP0ValidationError("registered JCR-MCD precursor route differs")
    design = protocol["design"]
    if (
        tuple(design["certificates"]) != CERTIFICATES
        or tuple(design["protected_certificates"]) != PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != TARGET_CERTIFICATES
        or tuple(float(value) for value in design["path_fractions"])
        != PATH_FRACTIONS
        or float(design["protected_retention_floor"])
        != PROTECTED_RETENTION_FLOOR
        or int(design["fit_units"]) != FIT_UNITS
        or int(design["calibration_units"]) != CALIBRATION_UNITS
        or int(design["diagnostic_units"]) != DIAGNOSTIC_UNITS
        or int(design["path_candidates"]) != len(PATH_FRACTIONS)
        or float(design["conformal_alpha"]) != CONFORMAL_ALPHA
        or float(design["conformal_acceptance_threshold"])
        != CONFORMAL_ACCEPTANCE_THRESHOLD
        or design.get("conformal_policy")
        != "deploy_fit_candidate_if_threshold_nonpositive_else_raw_mcd"
        or design.get("conditional_on_candidate_acceptance_coverage_claimed")
        is not False
        or float(design["fit_target_gain_floor"]) != FIT_TARGET_GAIN_FLOOR
        or float(design["fit_target_score_floor"]) != FIT_TARGET_SCORE_FLOOR
    ):
        raise MedscmcdP0ValidationError("MED-SC-MCD design registry differs")

    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 6
        or config.get("required_gpu_name_substring") != "RTX 4070 Ti"
        or config.get("gradient_measurement_mode") != "eval"
        or config.get("model_optimizer_objects_authorized") is not False
        or config.get("scheduler_objects_authorized") is not False
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("transient_virtual_parameter_evaluations_authorized")
        is not True
        or config.get("training_authorized") is not False
        or float(config["weight_decay"]) != 0.0
        or int(config["gradient_accumulation_steps"]) != 1
    ):
        raise MedscmcdP0ValidationError("seed-6 virtual-step config differs")

    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise MedscmcdP0ValidationError("risk row count differs")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(protocol["selection"]["risk_order_seed"]))
    risk_order = torch.randperm(len(risk_rows), generator=generator).tolist()
    expected = expected_roles(targets, risk_order)
    for role in ("fit", "calibration", "diagnostic"):
        if protocol["selection"][f"{role}_units"] != expected[role]:
            raise MedscmcdP0ValidationError(f"{role} role registry differs")
    all_units = [item for role in expected.values() for item in role]
    if (
        len({item["sample_id"] for item in all_units}) != 40
        or len({item["risk_row_index"] for item in all_units}) != 40
        or [item["risk_row_index"] for item in all_units] != risk_order[:40]
    ):
        raise MedscmcdP0ValidationError("fit/calibration/diagnostic roles overlap")
    registry = jcr_runner.prompt_category_registry(targets)
    if dict(sorted(Counter(registry.values()).items())) != protocol["source"][
        "category_prompt_counts"
    ]:
        raise MedscmcdP0ValidationError("source category counts differ")
    return protocol, config, targets, risk_rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise MedscmcdP0ValidationError("MED-SC-MCD P0 requires a clean worktree")
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
            raise MedscmcdP0ValidationError(f"prospective output already exists: {output}")
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
        raise MedscmcdP0ValidationError("route structure differs from frozen source")
    return {str(pair["sample_id"]): pair for pair in pairs}, audit


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise MedscmcdP0ValidationError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name(0)
    if str(config["required_gpu_name_substring"]) not in gpu:
        raise MedscmcdP0ValidationError(f"registered RTX 4070 Ti is unavailable: {gpu}")
    tokenizer = tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = structural_features(tokenizer, targets, config, protocol)
    selected = [
        item["sample_id"]
        for role in ("fit_units", "calibration_units", "diagnostic_units")
        for item in protocol["selection"][role]
    ]
    if not all(sample_id in pair_map for sample_id in selected):
        raise MedscmcdP0ValidationError("selected safety pair is absent")
    conformal = medsc.split_conformal_accept_or_fallback(
        [-0.01] * CALIBRATION_UNITS,
        CONFORMAL_ALPHA,
        CONFORMAL_ACCEPTANCE_THRESHOLD,
        PROTECTED_RETENTION_FLOOR - 1.0,
    )
    result = {
        "status": "PASS_MEDSCMCD_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "seed": config["seed"],
        "gradient_measurement_mode": "eval",
        "certificates": list(CERTIFICATES),
        "arms": list(ARMS),
        "fit_units": FIT_UNITS,
        "calibration_units": CALIBRATION_UNITS,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "path_candidates": len(PATH_FRACTIONS),
        "conformal_rank": conformal["order_statistic_rank"],
        "conformal_coverage_lower_bound": conformal[
            "finite_sample_marginal_coverage_lower_bound"
        ],
        "virtual_parameter_evaluations": DIAGNOSTIC_UNITS * len(ARMS),
        "gpu": gpu,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _baseline_gradient(
    gradients: dict[str, dict[str, Any]], safety_mix: float
) -> dict[str, Any]:
    return gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(gradients["risk"], 1.0 - safety_mix),
        gradients["uniform"],
        right_scale=safety_mix,
    )


def build_fit_geometry(
    fit_gradients: list[dict[str, dict[str, Any]]],
    fit_coverages: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tolerance = float(config["dual_numeric_tolerance"])
    pooled = medsc.mean_certificate_gradients(fit_gradients)
    raw_baseline = _baseline_gradient(pooled, float(config["safety_mix"]))
    budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
        pooled["uniform"]
    )
    raw_direction, target_direction, endpoint_audit = path_geometry.endpoint_directions(
        pooled, list(TARGET_CERTIFICATES), budget, tolerance
    )
    baseline_displacement, baseline_map = frsc.realized_displacement(
        raw_baseline,
        None,
        0.0,
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    panels = {"pooled": pooled}
    panels.update(
        {f"fit_{index + 1:02d}": gradient for index, gradient in enumerate(fit_gradients)}
    )
    scored = medsc.score_path_panels_memory_efficient(
        raw_baseline,
        raw_direction,
        target_direction,
        panels,
        list(PATH_FRACTIONS),
        budget,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        reference_norm,
        float(config["clip_epsilon"]),
        tolerance,
    )
    fit_records = [scored[f"fit_{index + 1:02d}"] for index in range(FIT_UNITS)]
    selected_index, search = medsc.select_minimum_effective_candidate(
        scored["pooled"],
        fit_records,
        list(PROTECTED_CERTIFICATES),
        list(TARGET_CERTIFICATES),
        PROTECTED_RETENTION_FLOOR,
        FIT_TARGET_GAIN_FLOOR,
        FIT_TARGET_SCORE_FLOOR,
        float(protocol["gates"]["numeric_tolerance"]),
    )
    selected_fraction = float(PATH_FRACTIONS[selected_index])
    selected_direction = path_geometry.path_direction(
        raw_direction, target_direction, selected_fraction, budget, tolerance
    )
    raw_displacement, raw_map = frsc.realized_displacement(
        raw_baseline,
        raw_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    selected_displacement, selected_map = frsc.realized_displacement(
        raw_baseline,
        selected_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    fit_geometry = {
        "name": "step3_medscmcd_p0_fit_geometry_v1",
        "gradient_measurement_mode": "eval",
        "unit_granularity": "one_prompt_risk_pair",
        "fit_coverages": fit_coverages,
        "pooled_gradient_norms": {
            name: gradient_math.gradient_norm(gradient)
            for name, gradient in pooled.items()
        },
        "raw_baseline_gradient_norm": gradient_math.gradient_norm(raw_baseline),
        "auxiliary_budget_norm": budget,
        "reference_displacement_norm": reference_norm,
        "baseline_optimizer_map": baseline_map,
        "raw_optimizer_map": raw_map,
        "selected_optimizer_map": selected_map,
        "raw_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(raw_direction) - budget
        )
        / budget,
        "target_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(target_direction) - budget
        )
        / budget,
        "selected_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(selected_direction) - budget
        )
        / budget,
        "endpoint_audit": endpoint_audit,
        "search": search,
        "selected_index": selected_index,
        "selected_fraction": selected_fraction,
        "pooled_path_records": scored["pooled"],
        "fit_path_records": {
            f"fit_{index + 1:02d}": records
            for index, records in enumerate(fit_records)
        },
        "path_candidates": len(PATH_FRACTIONS),
        "calibration_scores_used_for_selection": False,
        "diagnostic_scores_used_for_selection": False,
        "raw_gradient_tensors_serialized": False,
    }
    fixed = {
        "baseline": baseline_displacement,
        "raw_mcd": raw_displacement,
        "medscmcd_mcd": selected_displacement,
    }
    del pooled, raw_baseline, raw_direction, target_direction, selected_direction
    return fit_geometry, fixed


def score_fixed_displacement(
    gradients: dict[str, dict[str, Any]],
    displacement: dict[str, Any],
    parameters: list[tuple[str, Any]],
    reference_norm: float,
) -> dict[str, Any]:
    moved = jcr_runner.device_displacement(displacement, parameters)
    result = frsc.displacement_scores(gradients, moved, reference_norm)
    del moved
    return result


def summarize_results(
    rows: list[dict[str, Any]],
    fit_geometry: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    conformal: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != DIAGNOSTIC_UNITS or len(calibration_rows) != CALIBRATION_UNITS:
        raise MedscmcdP0ValidationError("stored role-unit count differs")
    targets = tuple(protocol["design"]["target_certificates"])
    diagnostic: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows]
        target_minima = [
            min(float(item["diagnostic_scores"]["scores"][name]) for name in targets)
            for item in arm_rows
        ]
        diagnostic[arm] = {
            "minimum_worst_score": min(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "mean_worst_score": statistics.fmean(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "minimum_target_score": min(target_minima),
            "mean_target_minimum_score": statistics.fmean(target_minima),
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
            objectives = [
                item["finite_step"]["objectives"][certificate]
                for item in arm_rows
            ]
            ratios = [
                float(item["actual_to_predicted_ratio"])
                for item in objectives
                if item["actual_to_predicted_ratio"] is not None
            ]
            diagnostic[arm]["objectives"][certificate] = {
                "mean_predicted_reduction": statistics.fmean(
                    float(item["predicted_first_order_reduction"])
                    for item in objectives
                ),
                "mean_actual_reduction": statistics.fmean(
                    float(item["actual_loss_reduction"]) for item in objectives
                ),
                "minimum_actual_reduction": min(
                    float(item["actual_loss_reduction"]) for item in objectives
                ),
                "positive_actual_rate": sum(
                    float(item["actual_loss_reduction"]) > 0.0
                    for item in objectives
                )
                / len(objectives),
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
    paired: dict[str, Any] = {
        "medscmcd_minus_baseline_worst_score": jcr_runner._paired(
            [
                left - right
                for left, right in zip(
                    worst["medscmcd_mcd"], worst["baseline"], strict=True
                )
            ],
            protocol,
        ),
        "medscmcd_minus_raw_worst_score": jcr_runner._paired(
            [
                left - right
                for left, right in zip(
                    worst["medscmcd_mcd"], worst["raw_mcd"], strict=True
                )
            ],
            protocol,
        ),
        "medscmcd_minus_raw_target_minimum_score": jcr_runner._paired(
            [
                min(
                    float(row["arms"]["medscmcd_mcd"]["diagnostic_scores"]["scores"][name])
                    for name in targets
                )
                - min(
                    float(row["arms"]["raw_mcd"]["diagnostic_scores"]["scores"][name])
                    for name in targets
                )
                for row in rows
            ],
            protocol,
        ),
        "medscmcd_relative_exact_score": {},
        "medscmcd_relative_actual_reduction": {},
    }
    for certificate in CERTIFICATES:
        selected_exact = [
            float(row["arms"]["medscmcd_mcd"]["diagnostic_scores"]["scores"][certificate])
            for row in rows
        ]
        raw_exact = [
            float(row["arms"]["raw_mcd"]["diagnostic_scores"]["scores"][certificate])
            for row in rows
        ]
        selected_actual = [
            float(
                row["arms"]["medscmcd_mcd"]["finite_step"]["objectives"][certificate][
                    "actual_loss_reduction"
                ]
            )
            for row in rows
        ]
        raw_actual = [
            float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][certificate][
                    "actual_loss_reduction"
                ]
            )
            for row in rows
        ]
        paired["medscmcd_relative_exact_score"][certificate] = (
            jcr_runner._relative_paired(selected_exact, raw_exact, protocol)
        )
        paired["medscmcd_relative_actual_reduction"][certificate] = (
            jcr_runner._relative_paired(selected_actual, raw_actual, protocol)
        )

    valid_target_macro = []
    invalid_target_macro = []
    for index, row in enumerate(rows):
        raw_values = [
            float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][name][
                    "actual_loss_reduction"
                ]
            )
            for name in targets
        ]
        if any(value <= 0.0 for value in raw_values):
            invalid_target_macro.append(index)
            continue
        valid_target_macro.append(
            statistics.fmean(
                float(
                    row["arms"]["medscmcd_mcd"]["finite_step"]["objectives"][name][
                        "actual_loss_reduction"
                    ]
                )
                / raw_value
                - 1.0
                for name, raw_value in zip(targets, raw_values, strict=True)
            )
        )
    if invalid_target_macro:
        paired["target_macro_actual_relative_gain"] = {
            "defined": False,
            "invalid_denominator_indices": invalid_target_macro,
            "differences": [],
        }
    else:
        paired["target_macro_actual_relative_gain"] = {
            "defined": True,
            **jcr_runner._paired(valid_target_macro, protocol),
        }

    calibration = {
        "conformal": conformal,
        "raw_protected_scores_positive": all(
            bool(row["raw_protected_scores_positive"]) for row in calibration_rows
        ),
        "minimum_joint_protected_retention": min(
            float(row["minimum_protected_retention"]) for row in calibration_rows
        ),
        "minimum_protected_retention_by_certificate": {
            name: min(
                float(row["protected_retentions"][name])
                for row in calibration_rows
            )
            for name in PROTECTED_CERTIFICATES
        },
        "mean_target_minimum_gain_over_raw": statistics.fmean(
            float(row["candidate_target_minimum"])
            - float(row["raw_target_minimum"])
            for row in calibration_rows
        ),
        "positive_target_minimum_gain_rate": sum(
            float(row["candidate_target_minimum"])
            > float(row["raw_target_minimum"])
            for row in calibration_rows
        )
        / len(calibration_rows),
    }
    search = fit_geometry["search"]
    fit = {
        "selected_index": int(fit_geometry["selected_index"]),
        "selected_fraction": float(fit_geometry["selected_fraction"]),
        "attempted_candidates": int(search["attempted_candidates"]),
        "eligible_candidates": int(search["eligible_candidates"]),
        "selected_eligible": bool(search["selected"]["eligible"]),
        "minimum_selected_protected_retention": search["selected"][
            "minimum_fit_protected_retention"
        ],
        "selected_target_score_floor": float(
            search["selected"]["fit_target_score_floor"]
        ),
        "selected_target_gain_over_raw": float(
            search["selected"]["fit_target_gain_over_raw"]
        ),
    }
    return {
        "fit": fit,
        "calibration": calibration,
        "diagnostic": diagnostic,
        "paired": paired,
    }


def _relative_gate(
    item: dict[str, Any], mean_loss: float, ci_loss: float, unit_loss: float
) -> bool:
    return bool(item.get("defined")) and (
        float(item["bootstrap"]["mean"]) >= -mean_loss
        and float(item["bootstrap"]["ci95_lower"]) > -ci_loss
        and min(float(value) for value in item["differences"]) >= -unit_loss
    )


def evaluate_gates(
    rows: list[dict[str, Any]],
    fit_geometry: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    state_restored: bool,
    vram_headroom: float,
    gpu_name: str,
) -> dict[str, bool]:
    gates = protocol["gates"]
    protected = tuple(protocol["design"]["protected_certificates"])
    targets = tuple(protocol["design"]["target_certificates"])
    selected_diag = summary["diagnostic"]["medscmcd_mcd"]
    objectives = list(selected_diag["objectives"].values())
    baseline_pair = summary["paired"]["medscmcd_minus_baseline_worst_score"]
    target_pair = summary["paired"]["medscmcd_minus_raw_target_minimum_score"]
    macro_pair = summary["paired"]["target_macro_actual_relative_gain"]
    exact_relative = summary["paired"]["medscmcd_relative_exact_score"]
    actual_relative = summary["paired"]["medscmcd_relative_actual_reduction"]
    protected_exact_ok = all(
        _relative_gate(
            exact_relative[name],
            float(gates["protected_mean_relative_loss_max"]),
            float(gates["protected_ci_relative_loss_max"]),
            float(gates["protected_single_unit_relative_loss_max"]),
        )
        for name in protected
    )
    protected_actual_ok = all(
        _relative_gate(
            actual_relative[name],
            float(gates["protected_mean_relative_loss_max"]),
            float(gates["protected_ci_relative_loss_max"]),
            float(gates["protected_single_unit_relative_loss_max"]),
        )
        for name in protected
    )
    target_floor_ok = all(
        bool(actual_relative[name].get("defined"))
        and float(actual_relative[name]["bootstrap"]["mean"])
        >= -float(gates["target_objective_mean_relative_loss_max"])
        and min(float(value) for value in actual_relative[name]["differences"])
        >= -float(gates["target_objective_single_unit_relative_loss_max"])
        for name in targets
    )
    fit_coverages = fit_geometry["fit_coverages"]
    conformal = summary["calibration"]["conformal"]
    return {
        "integrity": (
            len(fit_coverages) == FIT_UNITS
            and all(item["mode"] == "eval" for item in fit_coverages)
            and all(item["risk_microbatches"] == 1 for item in fit_coverages)
            and all(item["safety_microbatches"] == 1 for item in fit_coverages)
            and all(item["all_gradients_finite_nonzero"] for item in fit_coverages)
            and len(calibration_rows) == CALIBRATION_UNITS
            and all(row["coverage"]["mode"] == "eval" for row in calibration_rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in calibration_rows)
            and len(rows) == DIAGNOSTIC_UNITS
            and all(row["coverage"]["mode"] == "eval" for row in rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in rows)
            and int(fit_geometry["path_candidates"]) == len(PATH_FRACTIONS)
            and int(fit_geometry["search"]["attempted_candidates"])
            == len(PATH_FRACTIONS)
            and len(fit_geometry["pooled_path_records"]) == len(PATH_FRACTIONS)
            and len(fit_geometry["fit_path_records"]) == FIT_UNITS
            and all(
                len(panel) == len(PATH_FRACTIONS)
                for panel in fit_geometry["fit_path_records"].values()
            )
            and all(
                bool(fit_geometry[name]["clipping_active"])
                for name in (
                    "baseline_optimizer_map",
                    "raw_optimizer_map",
                    "selected_optimizer_map",
                )
            )
            and float(fit_geometry["raw_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and float(fit_geometry["target_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and float(fit_geometry["selected_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and all(
                all(
                    arm["state_restored_after_virtual_step"]
                    for arm in row["arms"].values()
                )
                for row in rows
            )
            and state_restored
            and vram_headroom >= float(gates["minimum_vram_headroom_fraction"])
            and "RTX 4070 Ti" in gpu_name
            and fit_geometry["calibration_scores_used_for_selection"] is False
            and fit_geometry["diagnostic_scores_used_for_selection"] is False
            and fit_geometry["raw_gradient_tensors_serialized"] is False
        ),
        "fit_minimum_effective_dose": (
            bool(summary["fit"]["selected_eligible"])
            and int(summary["fit"]["eligible_candidates"]) >= 1
            and float(summary["fit"]["selected_fraction"])
            >= float(gates["minimum_nonzero_path_fraction"])
            and float(summary["fit"]["minimum_selected_protected_retention"])
            >= PROTECTED_RETENTION_FLOOR
            and float(summary["fit"]["selected_target_score_floor"])
            >= FIT_TARGET_SCORE_FLOOR
            and float(summary["fit"]["selected_target_gain_over_raw"])
            >= FIT_TARGET_GAIN_FLOOR
        ),
        "split_conformal_accept_or_raw_policy": (
            bool(summary["calibration"]["raw_protected_scores_positive"])
            and int(conformal["calibration_units"]) == CALIBRATION_UNITS
            and int(conformal["order_statistic_rank"]) == CALIBRATION_UNITS
            and float(conformal["finite_sample_marginal_coverage_lower_bound"])
            >= float(gates["conformal_coverage_lower_bound_min"])
            and float(
                conformal["finite_sample_marginal_policy_protection_lower_bound"]
            )
            >= float(gates["conformal_coverage_lower_bound_min"])
            and float(conformal["threshold"])
            <= float(gates["numeric_tolerance"])
            and bool(conformal["all_calibration_units_protected"])
            and bool(conformal["candidate_accepted"])
            and conformal["policy_action"] == "candidate"
            and float(conformal["fallback_nonconformity"])
            <= CONFORMAL_ACCEPTANCE_THRESHOLD
            and conformal[
                "conditional_on_candidate_acceptance_coverage_claimed"
            ]
            is False
            and fit_geometry["search"]["calibration_scores_used_for_selection"] is False
            and fit_geometry["search"]["diagnostic_scores_used_for_selection"] is False
        ),
        "diagnostic_common_transfer": (
            float(selected_diag["minimum_worst_score"])
            >= float(gates["diagnostic_exact_margin_min"])
            and all(
                float(value) >= float(gates["diagnostic_certificate_score_min"])
                for value in selected_diag["minimum_certificate_scores"].values()
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
            for item in objectives
        ),
        "protected_diagnostic_noninferiority": (
            protected_exact_ok and protected_actual_ok
        ),
        "target_diagnostic_advantage": (
            float(target_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_target_gain_mean_min"])
            and float(target_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(target_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and bool(macro_pair.get("defined"))
            and float(macro_pair["bootstrap"]["mean"])
            >= float(gates["target_macro_actual_relative_gain_mean_min"])
            and float(macro_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(macro_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and target_floor_ok
        ),
        "effect_not_weak": (
            float(selected_diag["mean_target_minimum_score"])
            >= float(gates["diagnostic_mean_target_score_min"])
            and bool(macro_pair.get("defined"))
            and float(macro_pair["bootstrap"]["mean"])
            >= float(gates["target_macro_actual_relative_gain_mean_min"])
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"]:
        return "STOP_MEDSCMCD_P0_INVALID"
    if not gates["fit_minimum_effective_dose"]:
        return "STOP_MEDSCMCD_P0_NO_MINIMUM_EFFECTIVE_DOSE"
    if not gates["split_conformal_accept_or_raw_policy"]:
        return "STOP_MEDSCMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED"
    if not gates["diagnostic_common_transfer"]:
        return "STOP_MEDSCMCD_P0_DIAGNOSTIC_TRANSFER_FAILED"
    if not gates["finite_step_common_descent"]:
        return "STOP_MEDSCMCD_P0_FINITE_STEP_COMMON_DESCENT_FAILED"
    if not gates["protected_diagnostic_noninferiority"]:
        return "STOP_MEDSCMCD_P0_PROTECTED_NONINFERIORITY_FAILED"
    if not gates["target_diagnostic_advantage"]:
        return "STOP_MEDSCMCD_P0_TARGET_ADVANTAGE_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_MEDSCMCD_P0_EFFECT_TOO_WEAK"
    return "PASS_MEDSCMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL"


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
        raise MedscmcdP0ValidationError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    initial_state = model_runner._snapshot(parameters)
    pair_map, structural = structural_features(tokenizer, targets, config, protocol)
    roles = {
        role: protocol["selection"][f"{role}_units"]
        for role in ("fit", "calibration", "diagnostic")
    }
    selected_risk = {
        int(item["risk_row_index"])
        for role in roles.values()
        for item in role
    }
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in selected_risk
    }

    fit_gradients = []
    fit_coverages = []
    for unit in roles["fit"]:
        gradients, coverage = model_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            config,
            train_mode=False,
        )
        fit_gradients.append(jcr_runner.cpu_certificate_copy(gradients))
        fit_coverages.append({"unit_id": unit["unit_id"], **coverage})
        if not model_runner._state_equal(parameters, initial_state):
            raise MedscmcdP0ValidationError("fit gradients changed parameters")
        print(
            f"MEDSCMCD {unit['unit_id']} gradients=7 "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    fit_geometry, fixed_displacements = build_fit_geometry(
        fit_gradients, fit_coverages, config, protocol
    )
    fit_geometry["runtime_head"] = runtime_head
    fit_geometry["protocol_sha256"] = etarc_p0.sha256(path)
    del fit_gradients
    gc.collect()
    torch.cuda.empty_cache()

    reference_norm = float(fit_geometry["reference_displacement_norm"])
    calibration_rows = []
    for unit in roles["calibration"]:
        gradients, coverage = model_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            config,
            train_mode=False,
        )
        raw = score_fixed_displacement(
            gradients,
            fixed_displacements["raw_mcd"],
            parameters,
            reference_norm,
        )
        candidate = score_fixed_displacement(
            gradients,
            fixed_displacements["medscmcd_mcd"],
            parameters,
            reference_norm,
        )
        raw_positive = all(
            float(raw["scores"][name]) > 0.0 for name in PROTECTED_CERTIFICATES
        )
        if not raw_positive:
            raise MedscmcdP0ValidationError(
                "calibration raw protected score is nonpositive"
            )
        nonconformity, residuals = medsc.joint_protected_nonconformity(
            {name: float(value) for name, value in raw["scores"].items()},
            {name: float(value) for name, value in candidate["scores"].items()},
            list(PROTECTED_CERTIFICATES),
            PROTECTED_RETENTION_FLOOR,
        )
        retentions = {
            name: float(candidate["scores"][name]) / float(raw["scores"][name])
            for name in PROTECTED_CERTIFICATES
        }
        calibration_rows.append(
            {
                "unit_id": unit["unit_id"],
                "risk_row_index": unit["risk_row_index"],
                "safety_sample_id": unit["sample_id"],
                "safety_category": unit["category"],
                "coverage": coverage,
                "raw_scores": raw,
                "candidate_scores": candidate,
                "raw_protected_scores_positive": raw_positive,
                "protected_retentions": retentions,
                "minimum_protected_retention": min(retentions.values()),
                "protected_relative_shortfalls": residuals,
                "joint_nonconformity": nonconformity,
                "raw_target_minimum": min(
                    float(raw["scores"][name]) for name in TARGET_CERTIFICATES
                ),
                "candidate_target_minimum": min(
                    float(candidate["scores"][name]) for name in TARGET_CERTIFICATES
                ),
            }
        )
        if not model_runner._state_equal(parameters, initial_state):
            raise MedscmcdP0ValidationError("calibration scoring changed parameters")
        print(
            f"MEDSCMCD {unit['unit_id']} retention={min(retentions.values()):.6f} "
            f"q={nonconformity:.6f}",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    conformal = medsc.split_conformal_accept_or_fallback(
        [float(row["joint_nonconformity"]) for row in calibration_rows],
        CONFORMAL_ALPHA,
        CONFORMAL_ACCEPTANCE_THRESHOLD,
        PROTECTED_RETENTION_FLOOR - 1.0,
    )
    conformal["candidate_selected_before_calibration"] = True
    conformal["calibration_scores_used_for_selection"] = False
    conformal["diagnostic_scores_used_for_selection"] = False

    map_audits = {
        "baseline": fit_geometry["baseline_optimizer_map"],
        "raw_mcd": fit_geometry["raw_optimizer_map"],
        "medscmcd_mcd": fit_geometry["selected_optimizer_map"],
    }
    rows = []
    for unit in roles["diagnostic"]:
        unit_started = time.time()
        gradients, coverage = model_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            config,
            train_mode=False,
        )
        initial_losses = dict(coverage["loss_means"])
        arms = {
            arm: jcr_runner.evaluate_fixed_arm(
                arm,
                fixed_displacements[arm],
                map_audits[arm],
                gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                unit,
                config,
                reference_norm,
            )
            for arm in ARMS
        }
        raw_scores = arms["raw_mcd"]["diagnostic_scores"]["scores"]
        candidate_scores = arms["medscmcd_mcd"]["diagnostic_scores"]["scores"]
        nonconformity, residuals = medsc.joint_protected_nonconformity(
            {name: float(value) for name, value in raw_scores.items()},
            {name: float(value) for name, value in candidate_scores.items()},
            list(PROTECTED_CERTIFICATES),
            PROTECTED_RETENTION_FLOOR,
        )
        row = {
            "unit_id": unit["unit_id"],
            "risk_row_index": unit["risk_row_index"],
            "safety_sample_id": unit["sample_id"],
            "safety_category": unit["category"],
            "coverage": coverage,
            "joint_nonconformity": nonconformity,
            "protected_relative_shortfalls": residuals,
            "conformal_threshold": conformal["threshold"],
            "covered_by_conformal_threshold": nonconformity
            <= float(conformal["threshold"]),
            "arms": arms,
            "wall_seconds": time.time() - unit_started,
        }
        rows.append(row)
        print(
            f"MEDSCMCD {unit['unit_id']} q={nonconformity:.6f} target="
            f"{min(float(candidate_scores[name]) for name in TARGET_CERTIFICATES):.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    final_hash = model_common.trainable_state_hash(parameters)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    summary = summarize_results(
        rows, fit_geometry, calibration_rows, conformal, protocol
    )
    gates = evaluate_gates(
        rows,
        fit_geometry,
        calibration_rows,
        summary,
        protocol,
        initial_hash == final_hash,
        headroom,
        gpu_name,
    )
    route = select_route(gates)
    outputs = {
        name: resolve_repo_path(value) for name, value in protocol["outputs"].items()
    }
    diagnostics = {
        "name": "step3_medscmcd_p0_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "summary": summary,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "state_restored": initial_hash == final_hash,
        "fit_units": FIT_UNITS,
        "calibration_units": CALIBRATION_UNITS,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "raw_gradient_tensors_serialized": False,
    }
    environment = {
        "name": "step3_medscmcd_p0_environment_v1",
        "runtime_head": runtime_head,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_medscmcd_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "summary": summary,
        "registered_policy_action": (
            "fit_candidate" if conformal["candidate_accepted"] else "raw_mcd"
        ),
        "conditional_on_candidate_acceptance_coverage_claimed": False,
        "proxy_dev_triplet_protocol_authorized": route
        == "PASS_MEDSCMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL",
        "matched_training_protocol_authorized": False,
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
    macro = summary["paired"]["target_macro_actual_relative_gain"]
    macro_text = (
        f"{float(macro['bootstrap']['mean']):.6f}"
        if macro.get("defined")
        else "undefined because a raw target reduction was nonpositive"
    )
    report = f"""# MED-SC-MCD Seed-6 Zero-Step P0 Report

Route: `{route}`.

Twelve one-pair eval-mode fit units selected the smallest effective path dose,
`lambda={summary['fit']['selected_fraction']:.5f}`. Its minimum fit protected
retention was `{summary['fit']['minimum_selected_protected_retention']}` and
its fit target-floor gain over raw was
`{summary['fit']['selected_target_gain_over_raw']:.6f}`.

The candidate was fixed before all 20 calibration units. The registered
accept-or-raw policy uses the candidate only when the joint split-conformal
threshold is nonpositive and otherwise falls back to raw MCD. Its rank was
`{conformal['order_statistic_rank']}`, its policy-level finite-sample marginal
protection lower bound was
`{conformal['finite_sample_marginal_policy_protection_lower_bound']:.6f}`, and
the observed threshold was `{conformal['threshold']:.6f}`. This guarantee is
for the complete policy over calibration and future draws; no coverage
conditional on candidate acceptance is claimed.

Across eight independent diagnostics, the mean candidate target minimum was
`{summary['diagnostic']['medscmcd_mcd']['mean_target_minimum_score']:.6f}` and
the macro relative finite-loss target gain over raw was `{macro_text}`.

All 24 finite-loss measurements were transient fixed-displacement applications
followed by exact state restoration. No optimizer or scheduler was created,
`optimizer_steps=0`, and no checkpoint, generated candidate, Core100 row, or
heldout row was opened. Even a pass authorizes only a separately frozen
`proxy_dev` triplet-data protocol, never training.
"""

    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_json_frozen(outputs["fit_geometry"], fit_geometry)
    etarc_p0.write_jsonl_frozen(outputs["calibration_rows"], calibration_rows)
    etarc_p0.write_jsonl_frozen(outputs["diagnostic_rows"], rows)
    etarc_p0.write_json_frozen(outputs["diagnostics"], diagnostics)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_medscmcd_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_medscmcd_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_medscmcd_p0.py verify-run"
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
    del model, fixed_displacements, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    fit_geometry = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["fit_geometry"])
    )
    calibration_rows = etarc_p0.read_jsonl(
        resolve_repo_path(protocol["outputs"]["calibration_rows"])
    )
    rows = etarc_p0.read_jsonl(resolve_repo_path(protocol["outputs"]["diagnostic_rows"]))
    diagnostics = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["diagnostics"])
    )
    environment = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["environment"])
    )
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["route"]))
    manifest = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["run_manifest"])
    )
    conformal = medsc.split_conformal_accept_or_fallback(
        [float(row["joint_nonconformity"]) for row in calibration_rows],
        CONFORMAL_ALPHA,
        CONFORMAL_ACCEPTANCE_THRESHOLD,
        PROTECTED_RETENTION_FLOOR - 1.0,
    )
    conformal["candidate_selected_before_calibration"] = True
    conformal["calibration_scores_used_for_selection"] = False
    conformal["diagnostic_scores_used_for_selection"] = False
    summary = summarize_results(
        rows, fit_geometry, calibration_rows, conformal, protocol
    )
    gates = evaluate_gates(
        rows,
        fit_geometry,
        calibration_rows,
        summary,
        protocol,
        bool(diagnostics["state_restored"]),
        float(environment["vram_headroom_fraction"]),
        str(environment["gpu"]),
    )
    route = select_route(gates)
    if (
        decision.get("route") != route
        or decision.get("gates") != gates
        or decision.get("summary") != summary
        or diagnostics.get("summary") != summary
        or manifest.get("route") != route
    ):
        raise MedscmcdP0ValidationError("stored and recomputed MED-SC-MCD route differs")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise MedscmcdP0ValidationError("MED-SC-MCD artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
        or decision.get("matched_training_protocol_authorized") is not False
    ):
        raise MedscmcdP0ValidationError("stored MED-SC-MCD boundary differs")
    result = {
        "status": "PASS_MEDSCMCD_P0_VERIFY",
        "route": route,
        "fit_units": FIT_UNITS,
        "calibration_units": CALIBRATION_UNITS,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "conformal_rank": conformal["order_statistic_rank"],
        "conformal_threshold": conformal["threshold"],
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
