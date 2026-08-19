"""Run FOACPO with an exact two-block segment-robust optimizer anchor."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import step3_barp_mcd_d1x as d1x
import step3_barp_mcd_training as optimizer_map
import step3_cspr_m0_mechanism as cspr
import step3_foacpo_common as common
import step3_foacpo_p0 as base
import step3_foacpo_p0r as p0r
import step3_ramp_common as gradient_math
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_foacpo_p0s/protocol/foacpo_p0s_protocol.json"
)


class P0SValidationError(ValueError):
    """Raised when the segment-robust protocol is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise P0SValidationError(f"path escapes repository: {value}") from exc
    return path


def maximin_segment_blend(
    left_scores: Mapping[str, float], right_scores: Mapping[str, float]
) -> dict[str, Any]:
    """Globally maximize the minimum affine score on one closed segment."""
    if set(left_scores) != set(right_scores) or not left_scores:
        raise P0SValidationError("segment score sets differ")
    names = sorted(left_scores)
    values = [float(left_scores[name]) for name in names] + [
        float(right_scores[name]) for name in names
    ]
    if not all(math.isfinite(value) for value in values):
        raise P0SValidationError("segment scores must be finite")
    candidates = {0.0, 1.0}
    for left_index, left_name in enumerate(names):
        left_intercept = float(left_scores[left_name])
        left_slope = float(right_scores[left_name]) - left_intercept
        for right_name in names[left_index + 1 :]:
            right_intercept = float(left_scores[right_name])
            right_slope = float(right_scores[right_name]) - right_intercept
            denominator = left_slope - right_slope
            if abs(denominator) <= 1e-15:
                continue
            weight = (right_intercept - left_intercept) / denominator
            if 0.0 <= weight <= 1.0:
                candidates.add(weight)

    def scores(weight: float) -> dict[str, float]:
        return {
            name: (1.0 - weight) * float(left_scores[name])
            + weight * float(right_scores[name])
            for name in names
        }

    rows = [
        {
            "weight": weight,
            "scores": scores(weight),
        }
        for weight in sorted(candidates)
    ]
    for row in rows:
        row["minimum_score"] = min(row["scores"].values())
    selected = max(
        rows,
        key=lambda row: (
            row["minimum_score"],
            sum(row["scores"].values()),
            -abs(row["weight"] - 0.5),
        ),
    )
    return {
        **selected,
        "candidate_weights": len(rows),
        "active_certificates": [
            name
            for name, value in selected["scores"].items()
            if abs(value - selected["minimum_score"]) <= 1e-12
        ],
    }


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = base.read_json(path)
    if protocol.get("status") != "prospective_before_any_P0S_model_forward_or_robust_anchor_score":
        raise P0SValidationError("P0S status differs")
    for name, item in protocol["frozen_inputs"].items():
        item_path = resolve_repo_path(item["path"])
        if not item_path.is_file() or base.sha256(item_path) != item["sha256"]:
            raise P0SValidationError(f"frozen input mismatch: {name}")
    if any(
        (
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise P0SValidationError("P0S violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise P0SValidationError("P0S entry commit is not an ancestor")
    prior = base.read_json(resolve_repo_path(protocol["frozen_inputs"]["p0r_decision"]["path"]))
    if prior.get("route") != "STOP_FOACPO_P0R_NO_ANCHOR_RELATIVE_PARETO_ROUTE":
        raise P0SValidationError("P0S precursor route differs")
    config = base.read_json(resolve_repo_path(protocol["config_path"]))
    targets = base.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    return protocol, config, targets, risk_rows


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, risk_rows = context(path)
    if not torch.cuda.is_available():
        raise P0SValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    blocks = base.select_protected_blocks(
        targets, len(risk_rows), int(protocol["selection"]["seed"])
    )
    if any(
        sample_id not in pair_map
        for block in blocks.values()
        for sample_id in block["safety_sample_ids"]
    ):
        raise P0SValidationError("protected block differs")
    pools = base.load_pairs(protocol)
    shapes = [
        cspr.token_shape(tokenizer, row["prompt"], row[role])
        for split in pools.values()
        for rows in split.values()
        for row in rows
        for role in ("chosen_response", "rejected_response")
    ]
    result = {
        "status": "PASS_FOACPO_P0S_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "maximum_pair_tokens": max(value["total_tokens"] for value in shapes),
        "robust_anchor_blocks": ["train", "calibration"],
        "robust_certificates": 8,
        "segment_solver": "endpoints plus all pairwise affine intersections",
        "protected_retention_floor": protocol["design"]["protected_retention_floor"],
        "mechanism_dev_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def raw_scores(
    units: dict[str, dict[str, Any]], displacement: dict[str, Any], prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}:{name}": float(gradient_math.gradient_dot(units[name], displacement))
        for name in rp.PROTECTED_CERTIFICATES
    }


def route_from_gates(gates: dict[str, bool]) -> str:
    if not gates["structural"] or not gates["state_restored"] or not gates["zero_optimizer"]:
        return "STOP_FOACPO_P0S_INVALID"
    if not gates["robust_anchor"]:
        return "STOP_FOACPO_P0S_NO_ROBUST_ANCHOR"
    if not gates["candidate_selected"]:
        return "STOP_FOACPO_P0S_NO_PARETO_ROUTE"
    if not gates["calibration"]:
        return "STOP_FOACPO_P0S_CALIBRATION_TRANSFER"
    if not gates["mechanism"]:
        return "STOP_FOACPO_P0S_MECHANISM_TRANSFER"
    return "PASS_FOACPO_P0S_ROBUST_ANCHOR_TRANSFER"


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, config, targets, risk_rows = context(path)
    runtime_head = base.require_clean_start(path, protocol)
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = cspr.load_model(protocol)
    parameters = cspr.trainable_parameters(model)
    optimizer = frozen.make_optimizer(model, config)
    initial_hash = cspr.trainable_state_hash(parameters)
    blocks = base.select_protected_blocks(
        targets, len(risk_rows), int(protocol["selection"]["seed"])
    )
    pools = base.load_pairs(protocol)
    max_length = int(config["max_length"])
    beta = float(protocol["design"]["beta"])

    protected: dict[str, dict[str, Any]] = {}
    protected_norms: dict[str, dict[str, float]] = {}
    protected_coverage: dict[str, dict[str, Any]] = {}
    local_anchors: dict[str, dict[str, Any]] = {}
    for split in ("train", "calibration"):
        protected[split], protected_norms[split], protected_coverage[split] = base.protected_gradients(
            model, parameters, tokenizer, targets, risk_rows, blocks[split], config, protocol
        )
        local_anchors[split] = d1x._proposal_anchor(
            {
                "block_id": split,
                "units": protected[split],
                "norms": protected_norms[split],
            },
            parameters,
            optimizer,
            config,
            protocol,
        )

    left_scores = {
        **raw_scores(protected["train"], local_anchors["train"]["anchor_displacement"], "train"),
        **raw_scores(
            protected["calibration"],
            local_anchors["train"]["anchor_displacement"],
            "calibration",
        ),
    }
    right_scores = {
        **raw_scores(
            protected["train"], local_anchors["calibration"]["anchor_displacement"], "train"
        ),
        **raw_scores(
            protected["calibration"],
            local_anchors["calibration"]["anchor_displacement"],
            "calibration",
        ),
    }
    robust_solution = maximin_segment_blend(left_scores, right_scores)
    robust_anchor = base.blend_maps(
        local_anchors["train"]["anchor_displacement"],
        local_anchors["calibration"]["anchor_displacement"],
        float(robust_solution["weight"]),
    )
    robust_valid = float(robust_solution["minimum_score"]) > float(
        protocol["gates"]["robust_anchor_strict_score_min"]
    )

    features: dict[str, dict[str, list[Any]]] = {"train_candidate": {}, "calibration": {}}
    gradients: dict[str, dict[str, dict[str, Any]]] = {"train_candidate": {}, "calibration": {}}
    coverage: dict[str, dict[str, dict[str, Any]]] = {"train_candidate": {}, "calibration": {}}
    for split in ("train_candidate", "calibration"):
        for pool in ("quality", "safety"):
            features[split][pool] = base.pair_features(
                tokenizer, pools[pool][split], max_length
            )
            gradients[split][pool], coverage[split][pool] = base.structured_preference_gradient(
                model, parameters, features[split][pool], beta
            )
    train_quality = base.unit_gradient(gradients["train_candidate"]["quality"])
    train_safety = base.unit_gradient(gradients["train_candidate"]["safety"])
    anchor_quality = base.cosine_score(gradients["calibration"]["quality"], robust_anchor)
    anchor_safety = base.cosine_score(gradients["calibration"]["safety"], robust_anchor)

    registry = []
    candidate_maps: dict[float, dict[str, Any]] = {}
    for alpha_value in protocol["design"]["alpha_registry"]:
        alpha = float(alpha_value)
        behavior_gradient = gradient_math.add_gradients(
            gradient_math.scaled_gradient(train_quality, alpha),
            gradient_math.scaled_gradient(train_safety, 1.0 - alpha),
        )
        behavior, audit = optimizer_map.current_adamw_displacement(
            parameters,
            optimizer,
            behavior_gradient,
            float(config["max_grad_norm"]),
            float(config["clip_epsilon"]),
        )
        behavior_scores = {
            **raw_scores(protected["train"], behavior, "train"),
            **raw_scores(protected["calibration"], behavior, "calibration"),
        }
        certificate = common.maximal_protected_blend(
            robust_solution["scores"],
            behavior_scores,
            float(protocol["design"]["protected_retention_floor"]),
        )
        candidate = base.blend_maps(robust_anchor, behavior, float(certificate["lambda"]))
        increment = gradient_math.add_gradients(
            candidate, gradient_math.scaled_gradient(robust_anchor, -1.0)
        )
        increment_ratio = gradient_math.gradient_norm(increment) / gradient_math.gradient_norm(
            robust_anchor
        )
        quality = base.cosine_score(gradients["calibration"]["quality"], candidate)
        safety = base.cosine_score(gradients["calibration"]["safety"], candidate)
        quality_delta = quality - anchor_quality
        safety_delta = safety - anchor_safety
        eligible = (
            robust_valid
            and increment_ratio > float(protocol["gates"]["normalized_increment_numeric_min"])
            and quality > 0.0
            and safety > 0.0
            and quality_delta > 0.0
            and safety_delta > 0.0
            and float(certificate["minimum_retention"]) + 1e-12
            >= float(protocol["design"]["protected_retention_floor"])
            and int(audit["completed_optimizer_steps"]) == 0
        )
        registry.append(
            {
                "alpha": alpha,
                "lambda": certificate["lambda"],
                "normalized_increment": increment_ratio,
                "minimum_train_calibration_protected_retention": certificate[
                    "minimum_retention"
                ],
                "limiting_certificates": certificate["limiting_certificates"],
                "anchor_quality_cosine": anchor_quality,
                "anchor_safety_cosine": anchor_safety,
                "candidate_quality_cosine": quality,
                "candidate_safety_cosine": safety,
                "quality_delta_over_anchor": quality_delta,
                "safety_delta_over_anchor": safety_delta,
                "minimum_delta_over_anchor": min(quality_delta, safety_delta),
                "optimizer_completed_steps": audit["completed_optimizer_steps"],
                "eligible": eligible,
            }
        )
        candidate_maps[alpha] = candidate
        del behavior_gradient, behavior, increment
    eligible = [row for row in registry if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["minimum_delta_over_anchor"],
            row["quality_delta_over_anchor"] + row["safety_delta_over_anchor"],
            -abs(row["alpha"] - 0.5),
        ),
        default=None,
    )

    calibration = None
    mechanism = None
    selected_map = candidate_maps[selected["alpha"]] if selected else None
    if selected is not None:
        cal_quality = base.sign_summary(
            base.per_pair_scores(
                model, parameters, features["calibration"]["quality"], selected_map, beta
            )
        )
        cal_safety = base.sign_summary(
            base.per_pair_scores(
                model, parameters, features["calibration"]["safety"], selected_map, beta
            )
        )
        calibration = {"quality": cal_quality, "safety": cal_safety}
        calibration["passed"] = (
            cal_quality["positive"]
            >= int(protocol["gates"]["calibration_quality_positive_min"])
            and cal_safety["positive"]
            >= int(protocol["gates"]["calibration_safety_positive_min"])
        )
        if calibration["passed"]:
            features["mechanism_dev"] = {}
            gradients["mechanism_dev"] = {}
            coverage["mechanism_dev"] = {}
            for pool in ("quality", "safety"):
                features["mechanism_dev"][pool] = base.pair_features(
                    tokenizer, pools[pool]["mechanism_dev"], max_length
                )
                gradients["mechanism_dev"][pool], coverage["mechanism_dev"][pool] = base.structured_preference_gradient(
                    model, parameters, features["mechanism_dev"][pool], beta
                )
            mech_units, _, mech_coverage = base.protected_gradients(
                model,
                parameters,
                tokenizer,
                targets,
                risk_rows,
                blocks["mechanism_dev"],
                config,
                protocol,
            )
            robust_record = {
                "anchor_displacement": robust_anchor,
                "reference_norm": local_anchors["train"]["reference_norm"],
            }
            protection = base.support_protection(
                mech_units,
                robust_record,
                selected_map,
                float(protocol["design"]["protected_retention_floor"]),
            )
            mech_quality = base.sign_summary(
                base.per_pair_scores(
                    model, parameters, features["mechanism_dev"]["quality"], selected_map, beta
                )
            )
            mech_safety = base.sign_summary(
                base.per_pair_scores(
                    model, parameters, features["mechanism_dev"]["safety"], selected_map, beta
                )
            )
            mechanism = {
                "aggregate_quality_cosine": base.cosine_score(
                    gradients["mechanism_dev"]["quality"], selected_map
                ),
                "aggregate_safety_cosine": base.cosine_score(
                    gradients["mechanism_dev"]["safety"], selected_map
                ),
                "quality": mech_quality,
                "safety": mech_safety,
                "protection": protection,
                "protected_coverage": mech_coverage,
            }
            mechanism["passed"] = (
                mechanism["aggregate_quality_cosine"] > 0.0
                and mechanism["aggregate_safety_cosine"] > 0.0
                and mech_quality["positive"]
                >= int(protocol["gates"]["mechanism_quality_positive_min"])
                and mech_safety["positive"]
                >= int(protocol["gates"]["mechanism_safety_positive_min"])
                and protection["passed"]
            )
            del mech_units

    final_hash = cspr.trainable_state_hash(parameters)
    preference_coverage = [item for split in coverage.values() for item in split.values()]
    structural = (
        all(item["all_gradients_finite_nonzero"] for item in protected_coverage.values())
        and all(item["finite_nonzero"] and not item["missing_gradient_tensors"] for item in preference_coverage)
    )
    gates = {
        "structural": structural,
        "robust_anchor": robust_valid,
        "candidate_selected": selected is not None,
        "calibration": bool(calibration and calibration["passed"]),
        "mechanism": bool(mechanism and mechanism["passed"]),
        "state_restored": final_hash == initial_hash,
        "zero_optimizer": all(int(row["optimizer_completed_steps"]) == 0 for row in registry),
    }
    route = route_from_gates(gates)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak = int(torch.cuda.max_memory_reserved(0))
    diagnostics = {
        "name": "step3_foacpo_p0s_segment_robust_diagnostics_v1",
        "runtime_head": runtime_head,
        "protected_blocks": blocks,
        "local_anchors": {
            split: {
                "candidate_id": value["anchor_candidate_id"],
                "exact_scores": value["anchor_record"]["exact_scores"],
            }
            for split, value in local_anchors.items()
        },
        "robust_anchor_solution": robust_solution,
        "route_registry": registry,
        "selected": selected,
        "calibration": calibration,
        "mechanism": mechanism,
        "gradient_coverage": coverage,
        "gates": gates,
        "route": route,
        "mechanism_dev_opened": mechanism is not None,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "peak_reserved_bytes": peak,
        "vram_headroom_fraction": 1.0 - peak / total_memory,
        "wall_seconds": time.time() - started,
    }
    base.write_json(resolve_repo_path(protocol["outputs"]["diagnostics"]), diagnostics)
    decision = {
        "name": "step3_foacpo_p0s_decision_v1",
        "route": route,
        "selected_alpha": selected["alpha"] if selected else None,
        "selected_lambda": selected["lambda"] if selected else None,
        "robust_anchor_weight": robust_solution["weight"],
        "robust_anchor_minimum_score": robust_solution["minimum_score"],
        "gates": gates,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "freeze finite-radius one-step FOACPO verification"
            if route == "PASS_FOACPO_P0S_ROBUST_ANCHOR_TRANSFER"
            else "stop FOACPO without training"
        ),
    }
    base.write_json(resolve_repo_path(protocol["outputs"]["decision"]), decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# FOACPO P0S Segment-Robust Report\n\n"
        f"Route: `{route}`. Robust anchor weight/minimum: "
        f"`{robust_solution['weight']}` / `{robust_solution['minimum_score']}`.\n\n"
        f"Selected alpha/lambda: `{decision['selected_alpha']}` / `{decision['selected_lambda']}`. "
        f"Mechanism-dev opened: `{mechanism is not None}`.\n\n"
        "No optimizer step or training was authorized; Core100 and heldout remained closed.\n",
        encoding="utf-8",
    )
    base.write_json(
        resolve_repo_path(protocol["outputs"]["environment"]),
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "runtime_head": runtime_head,
        },
    )
    del model, optimizer, candidate_maps, selected_map
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(decision, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    decision = base.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    diagnostics = base.read_json(resolve_repo_path(protocol["outputs"]["diagnostics"]))
    if (
        decision.get("route") != diagnostics.get("route")
        or decision.get("optimizer_steps") != 0
        or decision.get("training_authorized") is not False
        or diagnostics.get("optimizer_steps") != 0
        or diagnostics.get("training_authorized") is not False
        or diagnostics.get("core100_opened") is not False
        or diagnostics.get("heldout_opened") is not False
    ):
        raise P0SValidationError("P0S result differs or violates boundary")
    result = {
        "status": "PASS_FOACPO_P0S_VERIFY_RUN",
        "route": decision["route"],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    {"preflight": preflight, "run": run, "verify-run": verify_run}[args.command](path)


if __name__ == "__main__":
    main()
