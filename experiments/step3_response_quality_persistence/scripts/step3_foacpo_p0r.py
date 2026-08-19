"""Run the anchor-relative FOACPO zero-step retry."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_barp_mcd_d1x as d1x
import step3_barp_mcd_training as optimizer_map
import step3_cspr_m0_mechanism as cspr
import step3_foacpo_common as common
import step3_foacpo_p0 as base
import step3_ramp_common as gradient_math
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_foacpo_p0r/protocol/foacpo_p0r_protocol.json"
)


class P0RValidationError(ValueError):
    """Raised when P0R violates its prospective registry."""


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
        raise P0RValidationError(f"path escapes repository: {value}") from exc
    return path


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = base.read_json(path)
    if protocol.get("status") != "prospective_before_any_P0R_model_forward_or_anchor_relative_score":
        raise P0RValidationError("P0R status differs")
    for name, item in protocol["frozen_inputs"].items():
        item_path = resolve_repo_path(item["path"])
        if not item_path.is_file() or base.sha256(item_path) != item["sha256"]:
            raise P0RValidationError(f"frozen input mismatch: {name}")
    if any(
        (
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise P0RValidationError("P0R violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise P0RValidationError("P0R entry commit is not an ancestor of HEAD")
    prior = base.read_json(resolve_repo_path(protocol["frozen_inputs"]["p0_decision"]["path"]))
    if prior.get("route") != "STOP_FOACPO_P0_NO_CALIBRATION_ROUTE":
        raise P0RValidationError("P0R precursor route differs")
    config = base.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        float(config["weight_decay"]) != 0.0
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("training_authorized") is not False
    ):
        raise P0RValidationError("P0R optimizer configuration differs")
    targets = base.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    return protocol, config, targets, risk_rows


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, risk_rows = context(path)
    if not torch.cuda.is_available():
        raise P0RValidationError("CUDA is unavailable")
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
        raise P0RValidationError("protected selection differs")
    pools = base.load_pairs(protocol)
    shapes = [
        cspr.token_shape(tokenizer, row["prompt"], row[role])
        for split in pools.values()
        for rows in split.values()
        for row in rows
        for role in ("chosen_response", "rejected_response")
    ]
    if max(value["total_tokens"] for value in shapes) > int(config["max_length"]):
        raise P0RValidationError("P0R pair violates no-truncation rule")
    result = {
        "status": "PASS_FOACPO_P0R_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "maximum_pair_tokens": max(value["total_tokens"] for value in shapes),
        "alpha_registry": protocol["design"]["alpha_registry"],
        "activity_gate": "nonzero normalized displacement plus strict anchor-relative quality/safety Pareto improvement",
        "protected_blocks": blocks,
        "mechanism_dev_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def route_from_gates(gates: dict[str, bool]) -> str:
    if not gates["structural"] or not gates["state_restored"] or not gates["zero_optimizer"]:
        return "STOP_FOACPO_P0R_INVALID"
    if not gates["candidate_selected"]:
        return "STOP_FOACPO_P0R_NO_ANCHOR_RELATIVE_PARETO_ROUTE"
    if not gates["calibration"]:
        return "STOP_FOACPO_P0R_CALIBRATION_TRANSFER"
    if not gates["mechanism"]:
        return "STOP_FOACPO_P0R_MECHANISM_TRANSFER"
    return "PASS_FOACPO_P0R_ANCHOR_RELATIVE_TRANSFER"


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

    train_units, train_norms, train_protected_coverage = base.protected_gradients(
        model, parameters, tokenizer, targets, risk_rows, blocks["train"], config, protocol
    )
    proposal = {"block_id": "train", "units": train_units, "norms": train_norms}
    anchor = d1x._proposal_anchor(proposal, parameters, optimizer, config, protocol)
    anchor_scores = {
        name: float(anchor["anchor_record"]["exact_scores"][name])
        for name in rp.PROTECTED_CERTIFICATES
    }
    if not all(value > 0.0 for value in anchor_scores.values()):
        raise P0RValidationError("P0R anchor is not strictly protected")

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
    anchor_quality = base.cosine_score(
        gradients["calibration"]["quality"], anchor["anchor_displacement"]
    )
    anchor_safety = base.cosine_score(
        gradients["calibration"]["safety"], anchor["anchor_displacement"]
    )
    cal_units, _, cal_protected_coverage = base.protected_gradients(
        model,
        parameters,
        tokenizer,
        targets,
        risk_rows,
        blocks["calibration"],
        config,
        protocol,
    )

    registry = []
    candidate_maps: dict[float, dict[str, Any]] = {}
    for alpha_value in protocol["design"]["alpha_registry"]:
        alpha = float(alpha_value)
        behavior_gradient = gradient_math.add_gradients(
            gradient_math.scaled_gradient(train_quality, alpha),
            gradient_math.scaled_gradient(train_safety, 1.0 - alpha),
        )
        behavior_displacement, audit = optimizer_map.current_adamw_displacement(
            parameters,
            optimizer,
            behavior_gradient,
            float(config["max_grad_norm"]),
            float(config["clip_epsilon"]),
        )
        behavior_scores = d1x._scores(
            train_units, behavior_displacement, float(anchor["reference_norm"])
        )
        certificate = common.maximal_protected_blend(
            anchor_scores,
            {name: behavior_scores[name] for name in rp.PROTECTED_CERTIFICATES},
            float(protocol["design"]["protected_retention_floor"]),
        )
        candidate = base.blend_maps(
            anchor["anchor_displacement"],
            behavior_displacement,
            float(certificate["lambda"]),
        )
        increment = gradient_math.add_gradients(
            candidate, gradient_math.scaled_gradient(anchor["anchor_displacement"], -1.0)
        )
        increment_ratio = gradient_math.gradient_norm(increment) / gradient_math.gradient_norm(
            anchor["anchor_displacement"]
        )
        quality = base.cosine_score(gradients["calibration"]["quality"], candidate)
        safety = base.cosine_score(gradients["calibration"]["safety"], candidate)
        quality_delta = quality - anchor_quality
        safety_delta = safety - anchor_safety
        support = base.support_protection(
            cal_units,
            anchor,
            candidate,
            float(protocol["design"]["protected_retention_floor"]),
        )
        eligible = (
            increment_ratio > float(protocol["gates"]["normalized_increment_numeric_min"])
            and quality > 0.0
            and safety > 0.0
            and quality_delta > 0.0
            and safety_delta > 0.0
            and support["passed"]
            and int(audit["completed_optimizer_steps"]) == 0
        )
        registry.append(
            {
                "alpha": alpha,
                "lambda": certificate["lambda"],
                "normalized_increment": increment_ratio,
                "train_minimum_protected_retention": certificate["minimum_retention"],
                "anchor_quality_cosine": anchor_quality,
                "anchor_safety_cosine": anchor_safety,
                "candidate_quality_cosine": quality,
                "candidate_safety_cosine": safety,
                "quality_delta_over_anchor": quality_delta,
                "safety_delta_over_anchor": safety_delta,
                "minimum_delta_over_anchor": min(quality_delta, safety_delta),
                "calibration_protection": support,
                "optimizer_completed_steps": audit["completed_optimizer_steps"],
                "eligible": eligible,
            }
        )
        candidate_maps[alpha] = candidate
        del behavior_gradient, behavior_displacement, increment
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
        calibration = {
            "quality": cal_quality,
            "safety": cal_safety,
            "protection": selected["calibration_protection"],
            "protected_coverage": cal_protected_coverage,
        }
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
            mech_units, _, mech_protected_coverage = base.protected_gradients(
                model,
                parameters,
                tokenizer,
                targets,
                risk_rows,
                blocks["mechanism_dev"],
                config,
                protocol,
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
            mech_protection = base.support_protection(
                mech_units,
                anchor,
                selected_map,
                float(protocol["design"]["protected_retention_floor"]),
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
                "protection": mech_protection,
                "protected_coverage": mech_protected_coverage,
            }
            mechanism["passed"] = (
                mechanism["aggregate_quality_cosine"] > 0.0
                and mechanism["aggregate_safety_cosine"] > 0.0
                and mech_quality["positive"]
                >= int(protocol["gates"]["mechanism_quality_positive_min"])
                and mech_safety["positive"]
                >= int(protocol["gates"]["mechanism_safety_positive_min"])
                and mech_protection["passed"]
            )
            del mech_units

    final_hash = cspr.trainable_state_hash(parameters)
    preference_coverage = [item for split in coverage.values() for item in split.values()]
    structural = (
        train_protected_coverage["all_gradients_finite_nonzero"]
        and all(item["finite_nonzero"] and not item["missing_gradient_tensors"] for item in preference_coverage)
    )
    gates = {
        "structural": structural,
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
        "name": "step3_foacpo_p0r_anchor_relative_diagnostics_v1",
        "runtime_head": runtime_head,
        "anchor": {
            "candidate_id": anchor["anchor_candidate_id"],
            "exact_scores": anchor["anchor_record"]["exact_scores"],
            "reference_norm": anchor["reference_norm"],
        },
        "protected_blocks": blocks,
        "anchor_relative_registry": registry,
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
        "name": "step3_foacpo_p0r_decision_v1",
        "route": route,
        "selected_alpha": selected["alpha"] if selected else None,
        "selected_lambda": selected["lambda"] if selected else None,
        "gates": gates,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "freeze finite-radius one-step FOACPO verification"
            if route == "PASS_FOACPO_P0R_ANCHOR_RELATIVE_TRANSFER"
            else "stop FOACPO without training"
        ),
    }
    base.write_json(resolve_repo_path(protocol["outputs"]["decision"]), decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# FOACPO P0R Anchor-Relative Report\n\n"
        f"Route: `{route}`. Selected alpha/lambda: `{decision['selected_alpha']}` / "
        f"`{decision['selected_lambda']}`.\n\n"
        f"Calibration passed: `{gates['calibration']}`; mechanism-dev opened: "
        f"`{mechanism is not None}`; mechanism passed: `{gates['mechanism']}`.\n\n"
        "No optimizer step or training was authorized. Core100 and heldout remained closed.\n",
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
        raise P0RValidationError("P0R result differs or violates boundary")
    result = {
        "status": "PASS_FOACPO_P0R_VERIFY_RUN",
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
