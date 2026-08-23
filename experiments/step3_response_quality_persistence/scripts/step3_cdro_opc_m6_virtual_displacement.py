"""Registered finite-radius direct virtual displacement study for CDR-OPC M6."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v4 as v4
import step3_cdro_opc_m5_confirmation as m5


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m6/protocol/cdro_opc_m6_virtual_displacement_protocol.json"
)
ARMS = ("fallback", "selected")
SPLITS = ("calibration", "sealed_audit")
FIXED_DOSE = 0.0337860296353059
VIRTUAL_SCALE_ETA = 2e-4
ALLOWED_RELATIVE_DROP = 0.04
SCORE_SCALE_FLOOR = 0.02
REQUIRED_COVERED_UNITS = 19
BEHAVIOR_RATIO_LOWER = 0.25
BEHAVIOR_RATIO_UPPER = 1.75
ARITHMETIC_TOLERANCE = 1e-10
MAXIMUM_MAP_RELATIVE_L2_ERROR = 2e-6
MAXIMUM_MAP_ABSOLUTE_ERROR = 1e-5
OUTPUT_ROOT = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m6"
)
SPLIT_GATE_KEYS = (
    "protected_unit_coverage",
    "protected_aggregate",
    "behavior",
    "gradient_coverage",
    "map_fidelity",
    "exact_restoration",
    "finite",
)
PRE_OPEN_COMMON_GATE_KEYS = (
    "fixed_direction",
    "solver_kkt",
    "optimizer_state_empty",
    "replay_hash_restored",
    "eval_mode",
    "zero_step",
    "map_precheck",
    "calibration_integrity",
)
PRE_CALIBRATION_COMMON_GATE_KEYS = PRE_OPEN_COMMON_GATE_KEYS[:-1]
COMMON_GATE_KEYS = (*PRE_OPEN_COMMON_GATE_KEYS, "final_integrity")
INTEGRITY_KEYS = (
    "state_hash_restored",
    "state_tensor_equal",
    "optimizer_state_empty",
    "all_gradients_none",
    "eval_mode",
    "rng_restored",
)
REQUIRED_FROZEN_INPUTS = {
    "runner",
    "runner_tests",
    "protocol_note",
    "m5_runner",
    "m5_protocol",
    "m5_decision",
    "m5_diagnostics",
    "m3_core_runner",
    "m3_v4_transport",
    "rpfrsc_runner",
    "frsc_common",
    "behavior_runner",
    "gradient_math",
    "model_loader",
    "tokenizer_runner",
    "training_common",
    "optimizer_factory",
    "optimizer_map",
    "protected_gradient_runner",
    "m1_protocol",
    "config",
    "selected_targets",
    "primary_candidates",
    "cohort",
    "risk_source",
}


class M6VirtualDisplacementError(ValueError):
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
        raise M6VirtualDisplacementError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def context(path: Path) -> tuple[Any, ...]:
    protocol = read_json(path)
    if (
        protocol.get("schema")
        != "step3_cdro_opc_m6_finite_radius_virtual_displacement_protocol_v1"
        or protocol.get("status") != "prospective_before_any_M6_virtual_write"
        or protocol.get("output_root") != OUTPUT_ROOT
    ):
        raise M6VirtualDisplacementError("M6 protocol identity differs")
    if any(
        (
            protocol.get("optimizer_objects") != 1,
            protocol.get("scheduler_objects") != 0,
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("transient_virtual_parameter_evaluations_authorized")
            is not True,
            protocol.get("checkpoint_writes_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
            protocol.get("result_dependent_rerun_authorized") is not False,
        )
    ):
        raise M6VirtualDisplacementError("M6 authorization boundary differs")
    if not REQUIRED_FROZEN_INPUTS.issubset(protocol.get("frozen_inputs", {})):
        raise M6VirtualDisplacementError("M6 frozen input registry is incomplete")
    for name, spec in protocol["frozen_inputs"].items():
        source = repo_path(str(spec["path"]))
        if not source.is_file() or file_sha256(source) != spec["sha256"]:
            raise M6VirtualDisplacementError(f"frozen input mismatch: {name}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    ).returncode:
        raise M6VirtualDisplacementError("entry commit is not an ancestor")
    m5_path = repo_path(protocol["frozen_inputs"]["m5_protocol"]["path"])
    m5_context = m5.context(m5_path)
    m5_protocol = m5_context[0]
    inherited_inputs = {
        "m5_runner": "runner",
        "m3_core_runner": "m3_core_runner",
        "m3_v4_transport": "m3_v4_transport",
        "m1_protocol": "m1_protocol",
        "selected_targets": "selected_targets",
        "primary_candidates": "primary_candidates",
        "cohort": "cohort",
        "risk_source": "risk_source",
    }
    if any(
        protocol["frozen_inputs"][m6_name]
        != m5_protocol["frozen_inputs"][m5_name]
        for m6_name, m5_name in inherited_inputs.items()
    ):
        raise M6VirtualDisplacementError("M6 inherited M5 input registry differs")
    m1_protocol = m5_context[1]
    if protocol["frozen_inputs"]["config"] != m1_protocol["frozen_inputs"]["config"]:
        raise M6VirtualDisplacementError("M6 inherited config registry differs")
    design = protocol.get("design", {})
    gates = protocol.get("gates", {})
    if (
        float(design.get("fixed_dose", math.nan)) != FIXED_DOSE
        or float(design.get("virtual_scale_eta", math.nan)) != VIRTUAL_SCALE_ETA
        or float(design.get("allowed_relative_drop", math.nan))
        != ALLOWED_RELATIVE_DROP
        or float(design.get("score_scale_floor", math.nan)) != SCORE_SCALE_FLOOR
        or tuple(design.get("protected_certificates", ())) != core.PROTECTED
        or tuple(design.get("arms", ())) != ("initial", *ARMS)
        or design.get("arm_order") != ["initial", "fallback", "selected"]
        or design.get("sealed_opening_rule")
        != "all_pre_open_common_and_calibration_gates"
        or set(gates) != {
            "arithmetic_tolerance",
            "required_covered_units",
            "behavior_ratio_lower",
            "behavior_ratio_upper",
            "maximum_map_relative_l2_error",
            "maximum_map_absolute_error",
        }
        or float(gates.get("arithmetic_tolerance", math.nan))
        != ARITHMETIC_TOLERANCE
        or type(gates.get("required_covered_units")) is not int
        or gates.get("required_covered_units") != REQUIRED_COVERED_UNITS
        or float(gates.get("behavior_ratio_lower", math.nan))
        != BEHAVIOR_RATIO_LOWER
        or float(gates.get("behavior_ratio_upper", math.nan))
        != BEHAVIOR_RATIO_UPPER
        or float(gates.get("maximum_map_relative_l2_error", math.nan))
        != MAXIMUM_MAP_RELATIVE_L2_ERROR
        or float(gates.get("maximum_map_absolute_error", math.nan))
        != MAXIMUM_MAP_ABSOLUTE_ERROR
    ):
        raise M6VirtualDisplacementError("M6 registered design or gates differ")
    numeric_values = (
        design["fixed_dose"],
        design["virtual_scale_eta"],
        design["allowed_relative_drop"],
        design["score_scale_floor"],
        gates["arithmetic_tolerance"],
        gates["behavior_ratio_lower"],
        gates["behavior_ratio_upper"],
        gates["maximum_map_relative_l2_error"],
        gates["maximum_map_absolute_error"],
    )
    if (
        not all(math.isfinite(float(value)) for value in numeric_values)
        or not 0.0 <= float(design["fixed_dose"]) <= 1.0
        or float(design["virtual_scale_eta"]) <= 0.0
        or float(design["allowed_relative_drop"]) < 0.0
        or float(design["score_scale_floor"]) <= 0.0
        or not 0.0
        < float(gates["behavior_ratio_lower"])
        <= float(gates["behavior_ratio_upper"])
        or float(gates["arithmetic_tolerance"]) < 0.0
        or float(gates["maximum_map_relative_l2_error"]) < 0.0
        or float(gates["maximum_map_absolute_error"]) < 0.0
    ):
        raise M6VirtualDisplacementError("M6 numeric gate domain differs")
    m5_decision = read_json(
        repo_path(protocol["frozen_inputs"]["m5_decision"]["path"])
    )
    m5_diagnostics = read_json(
        repo_path(protocol["frozen_inputs"]["m5_diagnostics"]["path"])
    )
    m5_sealed_statistics = m5_diagnostics.get("sealed_audit", {}).get(
        "protected_statistics", {}
    )
    inherited_tail = {
        "sealed_covered_units": m5_sealed_statistics.get("covered_units"),
        "sealed_total_units": m5_sealed_statistics.get("total_units"),
        "uncovered_units": [
            {
                "unit_id": row["unit_id"],
                "standardized_deteriorations": row["standardized_deteriorations"],
            }
            for row in m5_sealed_statistics.get("rows", [])
            if not row.get("covered")
        ],
    }
    if (
        m5_decision.get("route") != "PASS_CDRO_OPC_M5_ZERO_STEP_CONFIRMATION"
        or m5_diagnostics.get("route")
        != "PASS_CDRO_OPC_M5_ZERO_STEP_CONFIRMATION"
        or m5_diagnostics.get("sealed_audit_opened") is not True
        or m5_decision.get("optimizer_steps") != 0
        or m5_decision.get("training_authorized") is not False
        or float(m5_decision["fixed_dose"]) != FIXED_DOSE
        or float(m5_protocol["design"]["fixed_dose"]) != FIXED_DOSE
        or protocol.get("fixed_direction") != m5_protocol["fixed_direction"]
        or protocol.get("m5_protocol_snapshot")
        != {"selection": m5_protocol["selection"]}
        or protocol.get("predecessor_tail") != inherited_tail
    ):
        raise M6VirtualDisplacementError("M5 precursor or fixed dose differs")
    config = m5_context[2]
    if (
        float(config["learning_rate"])
        != float(protocol["design"]["virtual_scale_eta"])
        or float(config["weight_decay"]) != 0.0
        or int(config["optimizer_steps_authorized"]) != 0
    ):
        raise M6VirtualDisplacementError("frozen virtual scale or optimizer boundary differs")
    return (protocol, *m5_context[1:])


def require_clean_start(protocol: Mapping[str, Any]) -> str:
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise M6VirtualDisplacementError("M6 run requires a clean worktree")
    output = repo_path(str(protocol["output_root"]))
    if output.exists() and any(item.name != "protocol" for item in output.iterdir()):
        raise M6VirtualDisplacementError("M6 output already exists")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def parameter_snapshot(parameters: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in parameters}


def gradient_snapshot(parameters: Sequence[tuple[str, Any]]) -> dict[str, Any | None]:
    return {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in parameters
    }


def restore_parameters(
    parameters: Sequence[tuple[str, Any]], state: Mapping[str, Any]
) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(state[name])


def restore_gradients(
    parameters: Sequence[tuple[str, Any]], state: Mapping[str, Any | None]
) -> None:
    for name, parameter in parameters:
        saved = state[name]
        parameter.grad = None if saved is None else saved.detach().clone()


def state_equal(
    parameters: Sequence[tuple[str, Any]], state: Mapping[str, Any]
) -> bool:
    import torch

    return all(
        torch.equal(parameter.detach(), state[name]) for name, parameter in parameters
    )


def gradients_equal(
    parameters: Sequence[tuple[str, Any]], state: Mapping[str, Any | None]
) -> bool:
    import torch

    for name, parameter in parameters:
        saved = state[name]
        if saved is None and parameter.grad is None:
            continue
        if saved is None or parameter.grad is None or not torch.equal(parameter.grad, saved):
            return False
    return True


def mode_snapshot(model: Any) -> dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}


def mode_equal(model: Any, state: Mapping[str, bool]) -> bool:
    return all(module.training == state[name] for name, module in model.named_modules())


def restore_modes(model: Any, state: Mapping[str, bool]) -> None:
    for name, module in model.named_modules():
        module.training = state[name]


def rng_snapshot() -> dict[str, Any]:
    import torch

    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def rng_equal(state: Mapping[str, Any]) -> bool:
    import torch

    return torch.equal(torch.get_rng_state(), state["cpu"]) and all(
        torch.equal(left, right)
        for left, right in zip(torch.cuda.get_rng_state_all(), state["cuda"], strict=True)
    )


def restore_rng(state: Mapping[str, Any]) -> None:
    import torch

    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def realized_map_audit(
    parameters: Sequence[tuple[str, Any]],
    state: Mapping[str, Any],
    displacement: Mapping[str, Any],
    eta: float,
) -> dict[str, float]:
    squared_error = 0.0
    squared_reference = 0.0
    squared_delta = 0.0
    max_absolute_error = 0.0
    max_absolute_delta = 0.0
    for name, parameter in parameters:
        observed_map = (state[name] - parameter.detach()) / eta
        difference = observed_map - displacement[name]
        delta = state[name] - parameter.detach()
        squared_error += float(difference.double().square().sum().cpu())
        squared_reference += float(displacement[name].double().square().sum().cpu())
        squared_delta += float(delta.double().square().sum().cpu())
        max_absolute_error = max(
            max_absolute_error, float(difference.abs().max().cpu())
        )
        max_absolute_delta = max(max_absolute_delta, float(delta.abs().max().cpu()))
    return {
        "relative_l2_error": math.sqrt(squared_error)
        / max(math.sqrt(squared_reference), 1e-30),
        "max_absolute_error": max_absolute_error,
        "realized_parameter_l2_radius": math.sqrt(squared_delta),
        "realized_parameter_linf_radius": max_absolute_delta,
    }


def virtual_arm(
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    displacement: Mapping[str, Any],
    eta: float,
    evaluate: Callable[[], Any],
    registered_map_audit: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    gradients = gradient_snapshot(parameters)
    modes = mode_snapshot(model)
    rng = rng_snapshot()
    value: Any = None
    applied_audit: dict[str, float] | None = None
    rng_unchanged = False
    mode_unchanged = False
    gradients_unchanged = False
    try:
        restore_parameters(parameters, base_state)
        with torch.no_grad():
            for name, parameter in parameters:
                parameter.add_(displacement[name], alpha=-eta)
        applied_audit = (
            dict(registered_map_audit)
            if registered_map_audit is not None
            else realized_map_audit(parameters, base_state, displacement, eta)
        )
        value = evaluate()
        rng_unchanged = rng_equal(rng)
        mode_unchanged = mode_equal(model, modes)
        gradients_unchanged = gradients_equal(parameters, gradients)
    finally:
        restore_parameters(parameters, base_state)
        restore_gradients(parameters, gradients)
        restore_modes(model, modes)
        restore_rng(rng)
    restored = state_equal(parameters, base_state)
    gradients_restored = gradients_equal(parameters, gradients)
    mode_restored = mode_equal(model, modes)
    rng_restored = rng_equal(rng)
    if applied_audit is None:
        raise M6VirtualDisplacementError("virtual displacement was not applied")
    return value, {
        **applied_audit,
        "state_restored": restored,
        "rng_unchanged": rng_unchanged,
        "mode_unchanged": mode_unchanged,
        "gradients_unchanged": gradients_unchanged,
        "rng_restored": rng_restored,
        "mode_restored": mode_restored,
        "gradients_restored": gradients_restored,
    }


def normalized_actual_scores(
    initial: Mapping[str, float],
    final: Mapping[str, float],
    norms: Mapping[str, float],
    eta: float,
    reference_norm: float,
) -> dict[str, float]:
    return {
        name: (float(initial[name]) - float(final[name]))
        / (eta * float(norms[name]) * reference_norm)
        for name in core.PROTECTED
    }


def finite_protection_statistics(
    rows: Sequence[Mapping[str, Any]],
    allowed_drop: float,
    score_floor: float,
    tolerance: float,
) -> dict[str, Any]:
    if len(rows) != 20:
        raise M6VirtualDisplacementError("finite protection denominator differs")
    aggregate = {
        name: statistics.fmean(
            float(row["finite_deteriorations"][name]) for row in rows
        )
        for name in core.PROTECTED
    }
    return {
        "rows": list(rows),
        "covered_units": sum(bool(row["covered"]) for row in rows),
        "total_units": len(rows),
        "aggregate_finite_deteriorations": aggregate,
        "aggregate_protected": all(
            value <= allowed_drop + tolerance for value in aggregate.values()
        ),
        "allowed_relative_drop": allowed_drop,
        "score_scale_floor": score_floor,
    }


def preference_proxy_loss(model: Any, features: Sequence[Any], beta: float) -> float:
    import torch

    values = []
    model.eval()
    with torch.no_grad():
        for chosen, rejected in features:
            chosen_count = sum(value != -100 for value in chosen["labels"])
            rejected_count = sum(value != -100 for value in rejected["labels"])
            chosen_logp = core.base.cspr.sequence_log_probability(model, chosen) / chosen_count
            rejected_logp = (
                core.base.cspr.sequence_log_probability(model, rejected) / rejected_count
            )
            values.append(float((0.5 * beta * (rejected_logp - chosen_logp)).cpu()))
    return statistics.fmean(values)


def displacement_difference(
    selected: Mapping[str, Any], fallback: Mapping[str, Any]
) -> dict[str, Any]:
    return core.gradient_math.add_gradients(
        dict(selected), core.gradient_math.scaled_gradient(dict(fallback), -1.0)
    )


def arm_audit_passes(
    audit: Mapping[str, Any], maximum_relative_error: float, maximum_absolute_error: float
) -> bool:
    return bool(
        float(audit["relative_l2_error"]) <= maximum_relative_error
        and float(audit["max_absolute_error"]) <= maximum_absolute_error
        and audit["state_restored"]
        and audit["rng_unchanged"]
        and audit["mode_unchanged"]
        and audit["gradients_unchanged"]
        and audit["rng_restored"]
        and audit["mode_restored"]
        and audit["gradients_restored"]
    )


def registered_map_precheck(
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    displacements: Mapping[str, Mapping[str, Any]],
    eta: float,
    maximum_relative_error: float,
    maximum_absolute_error: float,
) -> dict[str, Any]:
    audits = {}
    for arm in ARMS:
        _, audits[arm] = virtual_arm(
            model,
            parameters,
            base_state,
            displacements[arm],
            eta,
            lambda: None,
        )
    return {
        "arms": audits,
        "gate": all(
            arm_audit_passes(
                audits[arm], maximum_relative_error, maximum_absolute_error
            )
            for arm in ARMS
        ),
    }


def behavior_advantage(
    gradient: Mapping[str, Any],
    selected_minus_fallback: Mapping[str, Any],
    eta: float,
    fallback_loss: float,
    selected_loss: float,
    ratio_lower: float,
    ratio_upper: float,
) -> dict[str, Any]:
    predicted = eta * core.gradient_math.gradient_dot(
        dict(gradient), dict(selected_minus_fallback)
    )
    actual = float(fallback_loss) - float(selected_loss)
    ratio = actual / predicted if predicted != 0.0 else None
    gate = bool(
        predicted > 0.0
        and actual > 0.0
        and ratio is not None
        and math.isfinite(predicted)
        and math.isfinite(actual)
        and math.isfinite(ratio)
        and ratio_lower <= ratio <= ratio_upper
    )
    return {
        "predicted_selected_advantage": predicted,
        "actual_selected_advantage": actual,
        "actual_to_predicted_ratio": ratio,
        "gate": gate,
    }


def behavior_finite_panel(
    model: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    tokenizer: Any,
    rows: Mapping[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
    beta: float,
    fallback: Mapping[str, Any],
    selected: Mapping[str, Any],
    eta: float,
    ratio_lower: float,
    ratio_upper: float,
    map_audits: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    result = {}
    difference = displacement_difference(selected, fallback)
    for pool in core.POOLS:
        features = core.base.pair_features(
            tokenizer, rows[pool], int(config["max_length"])
        )
        initial = preference_proxy_loss(model, features, beta)
        gradient, coverage = core.base.structured_preference_gradient(
            model, list(parameters), features, beta
        )
        fallback_loss, fallback_audit = virtual_arm(
            model,
            parameters,
            base_state,
            fallback,
            eta,
            lambda: preference_proxy_loss(model, features, beta),
            map_audits["fallback"],
        )
        selected_loss, selected_audit = virtual_arm(
            model,
            parameters,
            base_state,
            selected,
            eta,
            lambda: preference_proxy_loss(model, features, beta),
            map_audits["selected"],
        )
        advantage = behavior_advantage(
            gradient,
            difference,
            eta,
            float(fallback_loss),
            float(selected_loss),
            ratio_lower,
            ratio_upper,
        )
        result[pool] = {
            "pairs": len(features),
            "initial_loss": initial,
            "fallback_loss": fallback_loss,
            "selected_loss": selected_loss,
            **advantage,
            "gradient_coverage": coverage,
            "arm_audits": {
                "fallback": fallback_audit,
                "selected": selected_audit,
            },
        }
        del features, gradient
    return result


def evaluate_split(
    protocol: Mapping[str, Any],
    m1_protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_targets: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    risk: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    split: str,
    model: Any,
    tokenizer: Any,
    parameters: Sequence[tuple[str, Any]],
    base_state: Mapping[str, Any],
    fallback: Mapping[str, Any],
    selected: Mapping[str, Any],
    reference_norm: float,
    map_audits: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    audit = m5.structural_audit(
        protocol["m5_protocol_snapshot"],
        selected_targets,
        units,
        tokenizer,
        config,
        split,
    )
    pairs, _ = core.tarc.build_pair_features(
        tokenizer,
        core.subset_targets(selected_targets, units, split),
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    pair_map = {str(pair["sample_id"]): pair for pair in pairs}
    split_units = [unit for unit in units if unit["split"] == split]
    risk_features = {
        int(unit["risk_row_index"]): core.training.response_feature(
            tokenizer,
            str(risk[int(unit["risk_row_index"])]["prompt"]),
            str(risk[int(unit["risk_row_index"])]["response"]),
            int(config["max_length"]),
        )
        for unit in split_units
    }
    eta = float(protocol["design"]["virtual_scale_eta"])
    score_floor = float(protocol["design"]["score_scale_floor"])
    allowed = float(protocol["design"]["allowed_relative_drop"])
    tolerance = float(protocol["gates"]["arithmetic_tolerance"])
    rows = []
    for unit in split_units:
        gradients, coverage = core.rp.aggregate_block_gradients(
            model,
            list(parameters),
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            dict(config),
            train_mode=False,
        )
        certificates, norms = v4.normalize_registered_certificates(gradients)
        fallback_scores = core.m1.map_scores(certificates, dict(fallback), reference_norm)
        selected_scores = core.m1.map_scores(certificates, dict(selected), reference_norm)
        initial_losses = {
            name: float(coverage["loss_means"][name]) for name in core.PROTECTED
        }
        arm_results = {}
        arm_audits = {}
        for arm, displacement in (("fallback", fallback), ("selected", selected)):
            final_losses, arm_audit = virtual_arm(
                model,
                parameters,
                base_state,
                displacement,
                eta,
                lambda unit=unit: {
                    name: value
                    for name, value in core.rp.evaluate_block_losses(
                        model,
                        risk_features,
                        pair_map,
                        [int(unit["risk_row_index"])],
                        [str(unit["sample_id"])],
                        dict(config),
                    ).items()
                    if name in core.PROTECTED
                },
                map_audits[arm],
            )
            normalized = normalized_actual_scores(
                initial_losses, final_losses, norms, eta, reference_norm
            )
            first_order = fallback_scores if arm == "fallback" else selected_scores
            arm_results[arm] = {
                "final_losses": final_losses,
                "actual_loss_reductions": {
                    name: initial_losses[name] - float(final_losses[name])
                    for name in core.PROTECTED
                },
                "normalized_actual_scores": normalized,
                "first_order_scores": first_order,
                "taylor_errors": {
                    name: normalized[name] - float(first_order[name])
                    for name in core.PROTECTED
                },
            }
            arm_audits[arm] = arm_audit
        deterioration = {
            name: (
                arm_results["fallback"]["normalized_actual_scores"][name]
                - arm_results["selected"]["normalized_actual_scores"][name]
            )
            / max(abs(float(fallback_scores[name])), score_floor)
            for name in core.PROTECTED
        }
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "sample_id": unit["sample_id"],
                "initial_losses": initial_losses,
                "gradient_norms": norms,
                "gradient_coverage": coverage,
                "arms": arm_results,
                "arm_audits": arm_audits,
                "finite_deteriorations": deterioration,
                "covered": all(value <= allowed + tolerance for value in deterioration.values()),
            }
        )
        del gradients, certificates
    protected = finite_protection_statistics(
        rows, allowed, score_floor, tolerance
    )
    behavior = behavior_finite_panel(
        model,
        parameters,
        base_state,
        tokenizer,
        core.behavior_rows(selected_targets, candidates, units, risk, split),
        config,
        float(m1_protocol["design"]["beta"]),
        fallback,
        selected,
        eta,
        float(protocol["gates"]["behavior_ratio_lower"]),
        float(protocol["gates"]["behavior_ratio_upper"]),
        map_audits,
    )
    all_arm_audits = [
        row["arm_audits"][arm] for row in rows for arm in ARMS
    ] + [behavior[pool]["arm_audits"][arm] for pool in core.POOLS for arm in ARMS]
    fidelity = all(
        float(item["relative_l2_error"])
        <= float(protocol["gates"]["maximum_map_relative_l2_error"])
        and float(item["max_absolute_error"])
        <= float(protocol["gates"]["maximum_map_absolute_error"])
        for item in all_arm_audits
    )
    restoration = all(
        arm_audit_passes(
            item,
            math.inf,
            math.inf,
        )
        for item in all_arm_audits
    )
    protected_coverage = all(
        bool(row["gradient_coverage"]["all_gradients_finite_nonzero"])
        and all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in row["gradient_coverage"]["gradient_norms"].values()
        )
        and int(row["gradient_coverage"]["risk_microbatches"]) == 1
        and int(row["gradient_coverage"]["safety_microbatches"]) == 1
        and int(row["gradient_coverage"]["response_tokens"]) > 0
        and int(row["gradient_coverage"]["continuation_tokens"]) > 0
        and int(row["gradient_coverage"]["tarc_margin_terms"]) > 0
        and int(row["gradient_coverage"]["etarc_margin_terms"]) > 0
        for row in rows
    )
    behavior_coverage = all(
        bool(behavior[pool]["gradient_coverage"]["finite_nonzero"])
        and not behavior[pool]["gradient_coverage"]["missing_gradient_tensors"]
        and int(behavior[pool]["gradient_coverage"]["pairs"]) > 0
        and int(behavior[pool]["gradient_coverage"]["chosen_response_tokens"]) > 0
        and int(behavior[pool]["gradient_coverage"]["rejected_response_tokens"]) > 0
        for pool in core.POOLS
    )
    protected_finite = all(
        math.isfinite(float(value))
        for row in rows
        for values in (
            row["initial_losses"].values(),
            row["gradient_norms"].values(),
            row["finite_deteriorations"].values(),
            *(
                row["arms"][arm][field].values()
                for arm in ARMS
                for field in (
                    "final_losses",
                    "actual_loss_reductions",
                    "normalized_actual_scores",
                    "first_order_scores",
                    "taylor_errors",
                )
            ),
        )
        for value in values
    )
    behavior_values = [
        behavior[pool][field]
        for pool in core.POOLS
        for field in (
            "initial_loss",
            "fallback_loss",
            "selected_loss",
            "predicted_selected_advantage",
            "actual_selected_advantage",
            "actual_to_predicted_ratio",
        )
    ]
    behavior_finite = all(
        value is not None and math.isfinite(float(value)) for value in behavior_values
    )
    finite = protected_finite and behavior_finite
    gates = {
        "protected_unit_coverage": protected["covered_units"]
        >= int(protocol["gates"]["required_covered_units"]),
        "protected_aggregate": bool(protected["aggregate_protected"]),
        "behavior": all(behavior[pool]["gate"] for pool in core.POOLS),
        "gradient_coverage": protected_coverage and behavior_coverage,
        "map_fidelity": fidelity,
        "exact_restoration": restoration,
        "finite": finite,
    }
    return {
        "split": split,
        "status": "evaluated",
        "structural_audit": audit,
        "finite_protection": protected,
        "behavior": behavior,
        "gates": gates,
    }


def strict_gate_map(value: Mapping[str, Any], expected: Sequence[str]) -> bool:
    return set(value) == set(expected) and all(
        type(value[name]) is bool for name in expected
    )


def skipped_split(split: str, reason: str) -> dict[str, Any]:
    return {
        "split": split,
        "status": "not_evaluated",
        "reason": reason,
        "structural_audit": None,
        "finite_protection": {"covered_units": 0, "total_units": 20},
        "behavior": None,
        "gates": {name: False for name in SPLIT_GATE_KEYS},
    }


def route_from_results(
    common_gates: Mapping[str, bool],
    calibration: Mapping[str, Any],
    sealed: Mapping[str, Any] | None,
) -> str:
    if not strict_gate_map(common_gates, COMMON_GATE_KEYS):
        return "STOP_CDRO_OPC_M6_INVALID"
    if not all(common_gates.values()):
        return "STOP_CDRO_OPC_M6_INVALID"
    if not strict_gate_map(calibration.get("gates", {}), SPLIT_GATE_KEYS):
        return "STOP_CDRO_OPC_M6_INVALID"
    if not all(calibration["gates"].values()):
        return "STOP_CDRO_OPC_M6_CALIBRATION"
    if sealed is None or not strict_gate_map(sealed.get("gates", {}), SPLIT_GATE_KEYS):
        return "STOP_CDRO_OPC_M6_INVALID"
    if not all(sealed["gates"].values()):
        return "STOP_CDRO_OPC_M6_SEALED_AUDIT"
    return "PASS_CDRO_OPC_M6_FIXED_DOSE_FINITE_RADIUS_VIRTUAL_DISPLACEMENT"


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, m1_protocol, config, _, _, selected, candidates, risk, units = context(path)
    if not torch.cuda.is_available():
        raise M6VirtualDisplacementError("CUDA unavailable")
    tokenizer = core.tarc.load_tokenizer(str(config["model_name_or_path"]))
    audits = {
        split: m5.structural_audit(
            protocol["m5_protocol_snapshot"],
            selected,
            units,
            tokenizer,
            config,
            split,
        )
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
        "status": "PASS_CDRO_OPC_M6_VIRTUAL_DISPLACEMENT_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "arms": ["initial", *ARMS],
        "fixed_dose": protocol["design"]["fixed_dose"],
        "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
        "calibration_units": 20,
        "sealed_audit_units": 20,
        "behavior_pairs": behavior,
        "structural_audits": audits,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def run(path: Path) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    (
        protocol,
        m1_protocol,
        config,
        old_targets,
        old_risk,
        selected_targets,
        candidates,
        risk,
        units,
    ) = context(path)
    runtime_head = require_clean_start(protocol)
    if not torch.cuda.is_available():
        raise M6VirtualDisplacementError("CUDA unavailable")
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = core.cspr.load_model(m1_protocol)
    model.eval()
    parameters = core.cspr.trainable_parameters(model)
    optimizer = core.frozen.make_optimizer(model, config)
    if optimizer.state:
        raise M6VirtualDisplacementError("optimizer state is not empty before replay")
    optimizer.step = lambda *args, **kwargs: (_ for _ in ()).throw(
        M6VirtualDisplacementError("optimizer.step is forbidden in M6")
    )
    initial_hash = core.cspr.trainable_state_hash(parameters)
    fallback, endpoint, reference_norm, replay = v4.reconstruct_fixed_direction(
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
    if not replay["fixed_direction_valid"] or optimizer.state:
        raise M6VirtualDisplacementError("direction replay or empty optimizer state failed")
    replay_hash = core.cspr.trainable_state_hash(parameters)
    if replay_hash != initial_hash:
        raise M6VirtualDisplacementError("direction replay did not restore trainable state")
    if any(parameter.grad is not None for _, parameter in parameters):
        raise M6VirtualDisplacementError("direction replay left gradients attached")
    core.frsc.normalized_certificates = v4.normalize_registered_certificates
    selected_map = core.base.blend_maps(
        fallback, endpoint, float(protocol["design"]["fixed_dose"])
    )
    base_state = parameter_snapshot(parameters)
    mode_state = mode_snapshot(model)
    base_rng = rng_snapshot()
    map_precheck = registered_map_precheck(
        model,
        parameters,
        base_state,
        {"fallback": fallback, "selected": selected_map},
        float(protocol["design"]["virtual_scale_eta"]),
        float(protocol["gates"]["maximum_map_relative_l2_error"]),
        float(protocol["gates"]["maximum_map_absolute_error"]),
    )
    registered_map_audits = {
        arm: {
            name: map_precheck["arms"][arm][name]
            for name in (
                "relative_l2_error",
                "max_absolute_error",
                "realized_parameter_l2_radius",
                "realized_parameter_linf_radius",
            )
        }
        for arm in ARMS
    }
    solver = replay["solver_audit"]
    common_gates = {
        "fixed_direction": bool(replay["fixed_direction_valid"]),
        "solver_kkt": bool(
            solver["optimization_valid"]
            and float(solver["primal_violation"])
            <= float(m1_protocol["gates"]["feasibility_tolerance"])
            and float(solver["normalized_stationarity_residual"])
            <= float(m1_protocol["gates"]["kkt_tolerance"])
        ),
        "optimizer_state_empty": not bool(optimizer.state),
        "replay_hash_restored": replay_hash == initial_hash,
        "eval_mode": all(not value for value in mode_state.values()),
        "zero_step": all(
            int(item["optimizer_completed_steps"]) == 0
            for item in replay["anchor_records"].values()
        )
        and all(
            int(item["completed_optimizer_steps"]) == 0
            for item in replay["behavior_map_audits"].values()
        ),
        "map_precheck": bool(map_precheck["gate"]),
    }
    pre_calibration_ready = all(common_gates.values())
    calibration = (
        evaluate_split(
            protocol,
            m1_protocol,
            config,
            selected_targets,
            candidates,
            risk,
            units,
            "calibration",
            model,
            tokenizer,
            parameters,
            base_state,
            fallback,
            selected_map,
            reference_norm,
            registered_map_audits,
        )
        if pre_calibration_ready
        else skipped_split("calibration", "pre_calibration_common_gate_failed")
    )
    calibration_integrity = {
        "state_hash_restored": core.cspr.trainable_state_hash(parameters) == initial_hash,
        "state_tensor_equal": state_equal(parameters, base_state),
        "optimizer_state_empty": not bool(optimizer.state),
        "all_gradients_none": all(
            parameter.grad is None for _, parameter in parameters
        ),
        "eval_mode": mode_equal(model, mode_state),
        "rng_restored": rng_equal(base_rng),
    }
    common_gates["calibration_integrity"] = all(calibration_integrity.values())
    pre_open_common_gates = dict(common_gates)
    calibration_pass = all(pre_open_common_gates.values()) and all(
        calibration["gates"].values()
    )
    sealed = None
    if calibration_pass:
        sealed = evaluate_split(
            protocol,
            m1_protocol,
            config,
            selected_targets,
            candidates,
            risk,
            units,
            "sealed_audit",
            model,
            tokenizer,
            parameters,
            base_state,
            fallback,
            selected_map,
            reference_norm,
            registered_map_audits,
        )
    final_hash = core.cspr.trainable_state_hash(parameters)
    final_integrity = {
        "state_hash_restored": final_hash == initial_hash,
        "state_tensor_equal": state_equal(parameters, base_state),
        "optimizer_state_empty": not bool(optimizer.state),
        "all_gradients_none": all(parameter.grad is None for _, parameter in parameters),
        "eval_mode": mode_equal(model, mode_state),
        "rng_restored": rng_equal(base_rng),
    }
    common_gates["final_integrity"] = all(final_integrity.values())
    route = route_from_results(common_gates, calibration, sealed)
    root = repo_path(protocol["output_root"]) / "run"
    diagnostics = {
        "schema": "step3_cdro_opc_m6_virtual_displacement_diagnostics_v1",
        "route": route,
        "runtime_head": runtime_head,
        "claim_scope": "registered_post_zero_step_finite_loss_development_on_M5_cohort",
        "fresh_external_confirmation_claimed": False,
        "predecessor_tail_not_erased": protocol["predecessor_tail"],
        "direct_optimizer_space_displacement_not_standard_adamw_step": True,
        "fixed_dose": protocol["design"]["fixed_dose"],
        "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
        "reference_norm": reference_norm,
        "fixed_direction_replay": replay,
        "registered_map_precheck": map_precheck,
        "pre_calibration_ready": pre_calibration_ready,
        "pre_open_common_gates": pre_open_common_gates,
        "common_gates": common_gates,
        "calibration": calibration,
        "calibration_integrity": calibration_integrity,
        "sealed_audit_opened": calibration_pass,
        "sealed_audit": sealed,
        "final_integrity": final_integrity,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "checkpoint_writes": 0,
        "candidate_generation": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    decision = {
        "schema": "step3_cdro_opc_m6_virtual_displacement_decision_v1",
        "route": route,
        "fixed_dose": protocol["design"]["fixed_dose"],
        "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
        "calibration_covered_units": calibration["finite_protection"]["covered_units"],
        "calibration_gates": calibration["gates"],
        "sealed_audit_opened": calibration_pass,
        "sealed_audit_covered_units": (
            sealed["finite_protection"]["covered_units"] if sealed else None
        ),
        "sealed_audit_gates": sealed["gates"] if sealed else None,
        "fresh_external_confirmation_claimed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "interpret finite-loss development only; do not train"
            if route.startswith("PASS")
            else "stop without tuning radius or dose; do not train"
        ),
    }
    environment = {
        "head": runtime_head,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "gpu": torch.cuda.get_device_name(0),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "wall_seconds": time.time() - started,
    }
    diagnostics_path = root / "cdro_opc_m6_virtual_displacement_diagnostics.json"
    decision_path = root / "cdro_opc_m6_virtual_displacement_decision.json"
    environment_path = root / "cdro_opc_m6_virtual_displacement_environment.json"
    report_path = root / "CDRO_OPC_M6_VIRTUAL_DISPLACEMENT_REPORT.md"
    reproduction_path = root / "reproduction_manifest.json"
    write_json(diagnostics_path, diagnostics)
    write_json(environment_path, environment)
    root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# CDR-OPC M6 Finite-Radius Direct Virtual Displacement\n\n"
        f"Route: `{route}`. Dose: `{protocol['design']['fixed_dose']}`. "
        f"Virtual scale: `{protocol['design']['virtual_scale_eta']}`. "
        f"Calibration finite protection: `{calibration['finite_protection']['covered_units']}/20`. "
        f"Sealed finite-loss panel opened: `{str(calibration_pass).lower()}`.\n\n"
        "This reuses the already opened M5 cohort and is registered finite-loss development, "
        "not fresh external confirmation or training. No optimizer step was executed.\n"
        "The predecessor M5 tail remains explicit: 19/20 sealed units were covered; "
        "m5_audit_27 had risk deterioration 0.07003866384161442.\n",
        encoding="utf-8",
    )
    write_json(decision_path, decision)
    reproduction = {
        "schema": "step3_cdro_opc_m6_reproduction_manifest_v1",
        "runtime_commit": runtime_head,
        "unique_reproduction_command": (
            ".\\.venv\\Scripts\\python.exe "
            "experiments/step3_response_quality_persistence/scripts/"
            "step3_cdro_opc_m6_virtual_displacement.py run"
        ),
        "verification_command": (
            ".\\.venv\\Scripts\\python.exe "
            "experiments/step3_response_quality_persistence/scripts/"
            "step3_cdro_opc_m6_virtual_displacement.py verify-run"
        ),
        "random_seed": int(config["seed"]),
        "parameters": {
            "fixed_dose": protocol["design"]["fixed_dose"],
            "virtual_scale_eta": protocol["design"]["virtual_scale_eta"],
            "allowed_relative_drop": protocol["design"]["allowed_relative_drop"],
            "score_scale_floor": protocol["design"]["score_scale_floor"],
            "required_covered_units": protocol["gates"]["required_covered_units"],
            "optimizer_steps": 0,
        },
        "inputs_sha256": {
            "protocol": file_sha256(path),
            **{
                name: spec["sha256"]
                for name, spec in protocol["frozen_inputs"].items()
            },
        },
        "outputs_sha256": {
            "diagnostics": file_sha256(diagnostics_path),
            "decision": file_sha256(decision_path),
            "environment": file_sha256(environment_path),
            "report": file_sha256(report_path),
        },
        "environment": environment,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(reproduction_path, reproduction)
    result = {
        "route": route,
        "calibration_covered_units": calibration["finite_protection"]["covered_units"],
        "sealed_audit_opened": calibration_pass,
        "sealed_audit_covered_units": (
            sealed["finite_protection"]["covered_units"] if sealed else None
        ),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return decision


def verify_artifacts(
    protocol: Mapping[str, Any],
    decision: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    if (
        decision.get("schema")
        != "step3_cdro_opc_m6_virtual_displacement_decision_v1"
        or diagnostics.get("schema")
        != "step3_cdro_opc_m6_virtual_displacement_diagnostics_v1"
        or float(decision["fixed_dose"])
        != float(protocol["design"]["fixed_dose"])
        or float(decision["virtual_scale_eta"])
        != float(protocol["design"]["virtual_scale_eta"])
        or decision["optimizer_steps"] != 0
        or decision["training_authorized"] is not False
        or decision["fresh_external_confirmation_claimed"] is not False
        or float(diagnostics["fixed_dose"])
        != float(protocol["design"]["fixed_dose"])
        or float(diagnostics["virtual_scale_eta"])
        != float(protocol["design"]["virtual_scale_eta"])
        or diagnostics["optimizer_objects"] != 1
        or diagnostics["optimizer_steps"] != 0
        or diagnostics["training_authorized"] is not False
        or diagnostics["checkpoint_writes"] != 0
        or diagnostics["candidate_generation"] != 0
        or diagnostics["core100_opened"] is not False
        or diagnostics["heldout_opened"] is not False
        or diagnostics["fresh_external_confirmation_claimed"] is not False
        or diagnostics.get("predecessor_tail_not_erased")
        != protocol.get("predecessor_tail")
        or diagnostics["direct_optimizer_space_displacement_not_standard_adamw_step"]
        is not True
    ):
        raise M6VirtualDisplacementError("M6 artifact boundary differs")
    calibration = diagnostics["calibration"]
    sealed = diagnostics["sealed_audit"]
    pre_open_common = diagnostics["pre_open_common_gates"]
    common = diagnostics["common_gates"]
    calibration_integrity = diagnostics["calibration_integrity"]
    final_integrity = diagnostics["final_integrity"]
    precheck = diagnostics["registered_map_precheck"]
    if (
        not strict_gate_map(pre_open_common, PRE_OPEN_COMMON_GATE_KEYS)
        or not strict_gate_map(common, COMMON_GATE_KEYS)
        or not strict_gate_map(calibration["gates"], SPLIT_GATE_KEYS)
        or not strict_gate_map(calibration_integrity, INTEGRITY_KEYS)
        or not strict_gate_map(final_integrity, INTEGRITY_KEYS)
        or {name: common[name] for name in PRE_OPEN_COMMON_GATE_KEYS}
        != pre_open_common
        or set(precheck.get("arms", {})) != set(ARMS)
        or type(precheck.get("gate")) is not bool
    ):
        raise M6VirtualDisplacementError("M6 gate registry differs")
    reproduced_precheck = all(
        arm_audit_passes(
            precheck["arms"][arm],
            float(protocol["gates"]["maximum_map_relative_l2_error"]),
            float(protocol["gates"]["maximum_map_absolute_error"]),
        )
        for arm in ARMS
    )
    reproduced_pre_calibration_ready = all(
        pre_open_common[name] for name in PRE_CALIBRATION_COMMON_GATE_KEYS
    )
    if (
        precheck["gate"] != reproduced_precheck
        or pre_open_common["map_precheck"] != reproduced_precheck
        or diagnostics.get("pre_calibration_ready")
        != reproduced_pre_calibration_ready
        or bool(pre_open_common["calibration_integrity"])
        != all(calibration_integrity.values())
        or bool(common["final_integrity"]) != all(final_integrity.values())
    ):
        raise M6VirtualDisplacementError("M6 integrity aggregate differs")
    if (
        (
            reproduced_pre_calibration_ready
            and calibration.get("status") != "evaluated"
        )
        or (
            not reproduced_pre_calibration_ready
            and calibration.get("status") != "not_evaluated"
        )
        or (sealed is not None and sealed.get("status") != "evaluated")
        or (sealed is not None and not strict_gate_map(sealed["gates"], SPLIT_GATE_KEYS))
    ):
        raise M6VirtualDisplacementError("M6 sequential evaluation status differs")
    expected_open = all(pre_open_common.values()) and all(
        calibration["gates"].values()
    )
    if (
        diagnostics["sealed_audit_opened"] != expected_open
        or decision["sealed_audit_opened"] != expected_open
        or (sealed is not None) != expected_open
    ):
        raise M6VirtualDisplacementError("M6 sealed opening does not reproduce")
    if (
        decision["calibration_gates"] != calibration["gates"]
        or int(decision["calibration_covered_units"])
        != int(calibration["finite_protection"]["covered_units"])
        or decision["sealed_audit_gates"]
        != (sealed["gates"] if sealed is not None else None)
        or decision["sealed_audit_covered_units"]
        != (
            sealed["finite_protection"]["covered_units"]
            if sealed is not None
            else None
        )
    ):
        raise M6VirtualDisplacementError("M6 decision summary differs")
    expected = route_from_results(common, calibration, sealed)
    if decision["route"] != expected or diagnostics["route"] != expected:
        raise M6VirtualDisplacementError("M6 route does not reproduce")
    expected_action = (
        "interpret finite-loss development only; do not train"
        if expected.startswith("PASS")
        else "stop without tuning radius or dose; do not train"
    )
    if decision.get("next_authorized_action") != expected_action:
        raise M6VirtualDisplacementError("M6 next action differs")
    return expected


def verify(path: Path) -> dict[str, Any]:
    protocol, *_ = context(path)
    root = repo_path(protocol["output_root"]) / "run"
    artifact_paths = {
        "diagnostics": root / "cdro_opc_m6_virtual_displacement_diagnostics.json",
        "decision": root / "cdro_opc_m6_virtual_displacement_decision.json",
        "environment": root / "cdro_opc_m6_virtual_displacement_environment.json",
        "report": root / "CDRO_OPC_M6_VIRTUAL_DISPLACEMENT_REPORT.md",
    }
    reproduction_path = root / "reproduction_manifest.json"
    if not reproduction_path.is_file() or not all(
        artifact.is_file() for artifact in artifact_paths.values()
    ):
        raise M6VirtualDisplacementError("M6 artifact set is incomplete")
    reproduction = read_json(reproduction_path)
    expected_inputs = {
        "protocol": file_sha256(path),
        **{
            name: spec["sha256"]
            for name, spec in protocol["frozen_inputs"].items()
        },
    }
    if (
        reproduction.get("schema")
        != "step3_cdro_opc_m6_reproduction_manifest_v1"
        or reproduction.get("inputs_sha256") != expected_inputs
        or reproduction.get("outputs_sha256")
        != {
            name: file_sha256(artifact)
            for name, artifact in artifact_paths.items()
        }
        or reproduction.get("optimizer_steps") != 0
        or reproduction.get("training_authorized") is not False
    ):
        raise M6VirtualDisplacementError("M6 reproduction manifest differs")
    decision = read_json(artifact_paths["decision"])
    diagnostics = read_json(artifact_paths["diagnostics"])
    environment = read_json(artifact_paths["environment"])
    if (
        environment.get("head") != diagnostics.get("runtime_head")
        or reproduction.get("runtime_commit") != diagnostics.get("runtime_head")
        or reproduction.get("environment") != environment
    ):
        raise M6VirtualDisplacementError("M6 environment record differs")
    route = verify_artifacts(protocol, decision, diagnostics)
    result = {
        "status": "PASS_CDRO_OPC_M6_VIRTUAL_DISPLACEMENT_VERIFY_RUN",
        "route": route,
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
