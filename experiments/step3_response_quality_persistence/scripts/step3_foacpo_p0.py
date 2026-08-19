"""Run the FOACPO factorized optimizer-space zero-step transfer study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import step3_barp_mcd_d1x as d1x
import step3_barp_mcd_training as optimizer_map
import step3_cspr_m0_mechanism as cspr
import step3_etarc_p0 as artifact
import step3_foacpo_common as common
import step3_frsc_common as frsc
import step3_ramp_common as gradient_math
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_foacpo_p0/protocol/foacpo_p0_protocol.json"
)


class P0ValidationError(ValueError):
    """Raised when P0 violates its prospective zero-step registry."""


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
        raise P0ValidationError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


def select_protected_blocks(
    targets: list[dict[str, Any]], risk_count: int, seed: int
) -> dict[str, dict[str, list[Any]]]:
    category_by_id: dict[str, str] = {}
    for row in targets:
        sample_id = str(row["sample_id"])
        category = str(row["category"])
        if sample_id in category_by_id and category_by_id[sample_id] != category:
            raise P0ValidationError("safety pair spans multiple categories")
        category_by_id[sample_id] = category
    buckets: dict[str, list[str]] = defaultdict(list)
    for sample_id, category in category_by_id.items():
        buckets[category].append(sample_id)
    categories = sorted(buckets, key=lambda value: stable_rank(seed, "category", value))
    if len(categories) != 8:
        raise P0ValidationError("protected source must contain eight categories")
    for category in categories:
        buckets[category].sort(key=lambda value: stable_rank(seed, "safety", value))
    names = ("train", "calibration", "mechanism_dev")
    selected: list[tuple[str, str]] = []
    for round_index in range(max(len(values) for values in buckets.values())):
        for category in categories:
            if round_index < len(buckets[category]):
                selected.append((buckets[category][round_index], category))
                if len(selected) == 12:
                    break
        if len(selected) == 12:
            break
    if len(selected) != 12:
        raise P0ValidationError("insufficient protected safety pairs")
    assigned: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    for sample_id, category in selected:
        eligible = [name for name in names if len(assigned[name]) < 4]
        destination = min(
            eligible,
            key=lambda name: (
                category in {value[1] for value in assigned[name]},
                len(assigned[name]),
                names.index(name),
            ),
        )
        assigned[destination].append((sample_id, category))
    safety_blocks = {
        name: [sample_id for sample_id, _ in assigned[name]] for name in names
    }
    risk_order = sorted(
        range(risk_count), key=lambda value: stable_rank(seed, "risk", str(value))
    )[:12]
    return {
        name: {
            "risk_row_indices": risk_order[index * 4 : (index + 1) * 4],
            "safety_sample_ids": safety_blocks[name],
        }
        for index, name in enumerate(names)
    }


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_FOACPO_P0_model_forward":
        raise P0ValidationError("P0 protocol status differs")
    for name, item in protocol["frozen_inputs"].items():
        item_path = resolve_repo_path(item["path"])
        if not item_path.is_file() or sha256(item_path) != item["sha256"]:
            raise P0ValidationError(f"frozen input mismatch: {name}")
    if any(
        (
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise P0ValidationError("P0 protocol violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise P0ValidationError("P0 entry commit is not an ancestor of HEAD")
    config = read_json(resolve_repo_path(protocol["config_path"]))
    if (
        float(config["weight_decay"]) != 0.0
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("training_authorized") is not False
        or config.get("model_optimizer_object_authorized") is not True
    ):
        raise P0ValidationError("P0 optimizer configuration differs")
    manifest = read_json(resolve_repo_path(protocol["frozen_inputs"]["d3_manifest"]["path"]))
    if manifest.get("route") != "PASS_FOACPO_D3_FACTORIZED_POOL_FREEZE":
        raise P0ValidationError("D3 factorized pool is not frozen")
    targets = read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise P0ValidationError("risk source row count differs")
    return protocol, config, targets, risk_rows


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise P0ValidationError("P0 requires a clean worktree")
    tracked = [item["path"] for item in protocol["frozen_inputs"].values()]
    tracked.extend((protocol["config_path"], path.resolve().relative_to(REPO).as_posix()))
    for value in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", value],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    output_root = resolve_repo_path(protocol["output_root"])
    if output_root.exists():
        extras = [item for item in output_root.iterdir() if item.name != "protocol"]
        if extras:
            raise P0ValidationError("P0 output root already contains run artifacts")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_pairs(protocol: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pool in ("quality", "safety"):
        rows = read_jsonl(resolve_repo_path(protocol["data"][f"{pool}_pairs"]))
        grouped = {
            split: [row for row in rows if row["prospective_split"] == split]
            for split in ("train_candidate", "calibration", "mechanism_dev")
        }
        if [len(grouped[name]) for name in grouped] != protocol["data"][f"{pool}_split_counts"]:
            raise P0ValidationError(f"{pool} split counts differ")
        result[pool] = grouped
    return result


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, risk_rows = context(path)
    if not torch.cuda.is_available():
        raise P0ValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    blocks = select_protected_blocks(targets, len(risk_rows), int(protocol["selection"]["seed"]))
    if any(sample_id not in pair_map for block in blocks.values() for sample_id in block["safety_sample_ids"]):
        raise P0ValidationError("protected selection contains an unknown safety pair")
    pairs = load_pairs(protocol)
    shapes = []
    for pools in pairs.values():
        for rows in pools.values():
            for row in rows:
                for role in ("chosen_response", "rejected_response"):
                    shapes.append(cspr.token_shape(tokenizer, row["prompt"], row[role]))
    if any(shape["total_tokens"] > int(config["max_length"]) for shape in shapes):
        raise P0ValidationError("factorized pair violates no-truncation rule")
    result = {
        "status": "PASS_FOACPO_P0_PREFLIGHT",
        "gpu": torch.cuda.get_device_name(0),
        "quality_split_counts": protocol["data"]["quality_split_counts"],
        "safety_split_counts": protocol["data"]["safety_split_counts"],
        "maximum_pair_tokens": max(shape["total_tokens"] for shape in shapes),
        "alpha_registry": protocol["design"]["alpha_registry"],
        "protected_blocks": blocks,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def structured_preference_gradient(
    model: Any,
    parameters: list[tuple[str, Any]],
    features: list[tuple[dict[str, list[int]], dict[str, list[int]]]],
    beta: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    model.zero_grad(set_to_none=True)
    scale = 1.0 / len(features)
    token_counts = []
    for chosen, rejected in features:
        chosen_count = sum(value != -100 for value in chosen["labels"])
        rejected_count = sum(value != -100 for value in rejected["labels"])
        chosen_logp = cspr.sequence_log_probability(model, chosen) / chosen_count
        (-0.5 * beta * scale * chosen_logp).backward()
        rejected_logp = cspr.sequence_log_probability(model, rejected) / rejected_count
        (0.5 * beta * scale * rejected_logp).backward()
        token_counts.append((chosen_count, rejected_count))
    gradient = {
        name: (
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
        )
        for name, parameter in parameters
    }
    missing = [name for name, parameter in parameters if parameter.grad is None]
    model.zero_grad(set_to_none=True)
    norm = gradient_math.gradient_norm(gradient)
    return gradient, {
        "pairs": len(features),
        "gradient_norm": norm,
        "finite_nonzero": math.isfinite(norm) and norm > 0.0,
        "missing_gradient_tensors": missing,
        "chosen_response_tokens": sum(value[0] for value in token_counts),
        "rejected_response_tokens": sum(value[1] for value in token_counts),
    }


def pair_features(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> list[tuple[dict[str, list[int]], dict[str, list[int]]]]:
    return [
        (
            cspr.response_feature(tokenizer, row["prompt"], row["chosen_response"], max_length),
            cspr.response_feature(tokenizer, row["prompt"], row["rejected_response"], max_length),
        )
        for row in rows
    ]


def unit_gradient(value: dict[str, Any]) -> dict[str, Any]:
    norm = gradient_math.gradient_norm(value)
    if not math.isfinite(norm) or norm <= 0.0:
        raise P0ValidationError("preference gradient is not finite and nonzero")
    return gradient_math.scaled_gradient(value, 1.0 / norm)


def blend_maps(left: dict[str, Any], right: dict[str, Any], weight: float) -> dict[str, Any]:
    return gradient_math.add_gradients(
        gradient_math.scaled_gradient(left, 1.0 - weight),
        gradient_math.scaled_gradient(right, weight),
    )


def cosine_score(gradient: dict[str, Any], displacement: dict[str, Any]) -> float:
    denominator = gradient_math.gradient_norm(gradient) * gradient_math.gradient_norm(displacement)
    if denominator <= 0.0:
        return math.nan
    return float(gradient_math.gradient_dot(gradient, displacement) / denominator)


def protected_gradients(
    model: Any,
    parameters: list[tuple[str, Any]],
    tokenizer: Any,
    targets: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    block: dict[str, list[Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, Any]]:
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    risk_features = {
        int(index): training.response_feature(
            tokenizer,
            str(risk_rows[int(index)]["prompt"]),
            str(risk_rows[int(index)]["response"]),
            int(config["max_length"]),
        )
        for index in block["risk_row_indices"]
    }
    gradients, coverage = rp.aggregate_block_gradients(
        model,
        parameters,
        risk_features,
        pair_map,
        [int(value) for value in block["risk_row_indices"]],
        [str(value) for value in block["safety_sample_ids"]],
        config,
        train_mode=False,
    )
    units, norms = frsc.normalized_certificates(gradients)
    del gradients
    return units, {name: float(value) for name, value in norms.items()}, coverage


def support_protection(
    units: dict[str, dict[str, Any]],
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    floor: float,
) -> dict[str, Any]:
    reference = float(anchor["reference_norm"])
    anchor_scores = d1x._scores(units, anchor["anchor_displacement"], reference)
    candidate_scores = d1x._scores(units, candidate, reference)
    checks = {}
    retentions: dict[str, float | None] = {}
    for name in rp.PROTECTED_CERTIFICATES:
        base_score = anchor_scores[name]
        score = candidate_scores[name]
        if base_score > 0.0:
            retentions[name] = score / base_score
            checks[name] = score + 1e-12 >= floor * base_score
        else:
            retentions[name] = None
            checks[name] = score + 1e-12 >= base_score
    return {
        "anchor_scores": anchor_scores,
        "candidate_scores": candidate_scores,
        "retentions": retentions,
        "checks": checks,
        "passed": all(checks.values()),
    }


def per_pair_scores(
    model: Any,
    parameters: list[tuple[str, Any]],
    features: list[tuple[dict[str, list[int]], dict[str, list[int]]]],
    displacement: dict[str, Any],
    beta: float,
) -> list[float]:
    values = []
    for feature in features:
        gradient, _ = structured_preference_gradient(model, parameters, [feature], beta)
        values.append(cosine_score(gradient, displacement))
        del gradient
    return values


def sign_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "positive": sum(value > 0.0 for value in values),
        "nonpositive": sum(value <= 0.0 for value in values),
    }


def route_from_gates(gates: dict[str, bool]) -> str:
    if not gates["structural"] or not gates["state_restored"] or not gates["zero_optimizer"]:
        return "STOP_FOACPO_P0_INVALID"
    if not gates["candidate_selected"]:
        return "STOP_FOACPO_P0_NO_CALIBRATION_ROUTE"
    if not gates["calibration"]:
        return "STOP_FOACPO_P0_CALIBRATION_TRANSFER"
    if not gates["mechanism"]:
        return "STOP_FOACPO_P0_MECHANISM_TRANSFER"
    return "PASS_FOACPO_P0_ZERO_STEP_TRANSFER"


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_start(path, protocol)
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = cspr.load_model(protocol)
    parameters = cspr.trainable_parameters(model)
    optimizer = frozen.make_optimizer(model, config)
    initial_hash = cspr.trainable_state_hash(parameters)
    blocks = select_protected_blocks(targets, len(risk_rows), int(protocol["selection"]["seed"]))
    pools = load_pairs(protocol)
    max_length = int(config["max_length"])
    beta = float(protocol["design"]["beta"])

    train_units, train_norms, train_protected_coverage = protected_gradients(
        model, parameters, tokenizer, targets, risk_rows, blocks["train"], config, protocol
    )
    proposal = {"block_id": "train", "units": train_units, "norms": train_norms}
    anchor = d1x._proposal_anchor(proposal, parameters, optimizer, config, protocol)
    anchor_scores = {
        name: float(anchor["anchor_record"]["exact_scores"][name])
        for name in rp.PROTECTED_CERTIFICATES
    }
    if not all(value > 0.0 for value in anchor_scores.values()):
        raise P0ValidationError("optimizer-space anchor is not strictly protected")

    gradients: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    coverage: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    features: dict[str, dict[str, list[Any]]] = defaultdict(dict)
    for split in ("train_candidate", "calibration"):
        for pool in ("quality", "safety"):
            features[split][pool] = pair_features(
                tokenizer, pools[pool][split], max_length
            )
            gradients[split][pool], coverage[split][pool] = structured_preference_gradient(
                model, parameters, features[split][pool], beta
            )
    train_quality = unit_gradient(gradients["train_candidate"]["quality"])
    train_safety = unit_gradient(gradients["train_candidate"]["safety"])
    calibration_rows = []
    candidate_maps: dict[float, dict[str, Any]] = {}
    for alpha in protocol["design"]["alpha_registry"]:
        alpha = float(alpha)
        behavior_gradient = gradient_math.add_gradients(
            gradient_math.scaled_gradient(train_quality, alpha),
            gradient_math.scaled_gradient(train_safety, 1.0 - alpha),
        )
        behavior_displacement, behavior_audit = optimizer_map.current_adamw_displacement(
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
        candidate = blend_maps(
            anchor["anchor_displacement"], behavior_displacement, float(certificate["lambda"])
        )
        quality_score = cosine_score(gradients["calibration"]["quality"], candidate)
        safety_score = cosine_score(gradients["calibration"]["safety"], candidate)
        eligible = (
            float(certificate["lambda"]) + 1e-12
            >= float(protocol["gates"]["minimum_behavior_blend"])
            and quality_score > 0.0
            and safety_score > 0.0
            and int(behavior_audit["completed_optimizer_steps"]) == 0
        )
        calibration_rows.append(
            {
                "alpha": alpha,
                "lambda": certificate["lambda"],
                "limiting_certificates": certificate["limiting_certificates"],
                "train_minimum_protected_retention": certificate["minimum_retention"],
                "calibration_quality_cosine": quality_score,
                "calibration_safety_cosine": safety_score,
                "calibration_minimum_cosine": min(quality_score, safety_score),
                "eligible": eligible,
                "optimizer_completed_steps": behavior_audit["completed_optimizer_steps"],
            }
        )
        candidate_maps[alpha] = candidate
        del behavior_gradient, behavior_displacement
    eligible = [row for row in calibration_rows if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["calibration_minimum_cosine"],
            row["calibration_quality_cosine"] + row["calibration_safety_cosine"],
            -abs(row["alpha"] - 0.5),
        ),
        default=None,
    )

    calibration = None
    mechanism = None
    selected_map = candidate_maps[selected["alpha"]] if selected else None
    if selected is not None:
        cal_quality_values = per_pair_scores(
            model, parameters, features["calibration"]["quality"], selected_map, beta
        )
        cal_safety_values = per_pair_scores(
            model, parameters, features["calibration"]["safety"], selected_map, beta
        )
        cal_units, _, cal_protected_coverage = protected_gradients(
            model,
            parameters,
            tokenizer,
            targets,
            risk_rows,
            blocks["calibration"],
            config,
            protocol,
        )
        cal_protection = support_protection(
            cal_units,
            anchor,
            selected_map,
            float(protocol["design"]["protected_retention_floor"]),
        )
        calibration = {
            "quality": sign_summary(cal_quality_values),
            "safety": sign_summary(cal_safety_values),
            "protection": cal_protection,
            "protected_coverage": cal_protected_coverage,
        }
        calibration["passed"] = (
            calibration["quality"]["positive"]
            >= int(protocol["gates"]["calibration_quality_positive_min"])
            and calibration["safety"]["positive"]
            >= int(protocol["gates"]["calibration_safety_positive_min"])
            and cal_protection["passed"]
        )
        del cal_units

        if calibration["passed"]:
            for pool in ("quality", "safety"):
                features["mechanism_dev"][pool] = pair_features(
                    tokenizer, pools[pool]["mechanism_dev"], max_length
                )
                gradients["mechanism_dev"][pool], coverage["mechanism_dev"][pool] = structured_preference_gradient(
                    model, parameters, features["mechanism_dev"][pool], beta
                )
            mech_quality_values = per_pair_scores(
                model, parameters, features["mechanism_dev"]["quality"], selected_map, beta
            )
            mech_safety_values = per_pair_scores(
                model, parameters, features["mechanism_dev"]["safety"], selected_map, beta
            )
            mech_units, _, mech_protected_coverage = protected_gradients(
                model,
                parameters,
                tokenizer,
                targets,
                risk_rows,
                blocks["mechanism_dev"],
                config,
                protocol,
            )
            mech_protection = support_protection(
                mech_units,
                anchor,
                selected_map,
                float(protocol["design"]["protected_retention_floor"]),
            )
            mechanism = {
                "aggregate_quality_cosine": cosine_score(
                    gradients["mechanism_dev"]["quality"], selected_map
                ),
                "aggregate_safety_cosine": cosine_score(
                    gradients["mechanism_dev"]["safety"], selected_map
                ),
                "quality": sign_summary(mech_quality_values),
                "safety": sign_summary(mech_safety_values),
                "protection": mech_protection,
                "protected_coverage": mech_protected_coverage,
            }
            mechanism["passed"] = (
                mechanism["aggregate_quality_cosine"] > 0.0
                and mechanism["aggregate_safety_cosine"] > 0.0
                and mechanism["quality"]["positive"]
                >= int(protocol["gates"]["mechanism_quality_positive_min"])
                and mechanism["safety"]["positive"]
                >= int(protocol["gates"]["mechanism_safety_positive_min"])
                and mech_protection["passed"]
            )
            del mech_units

    final_hash = cspr.trainable_state_hash(parameters)
    all_coverage = [
        item for split in coverage.values() for item in split.values()
    ] + [train_protected_coverage]
    structural = all(
        item.get("finite_nonzero", item.get("all_gradients_finite_nonzero", False))
        and not item.get("missing_gradient_tensors", [])
        for item in all_coverage
    )
    gates = {
        "structural": structural,
        "candidate_selected": selected is not None,
        "calibration": bool(calibration and calibration["passed"]),
        "mechanism": bool(mechanism and mechanism["passed"]),
        "state_restored": final_hash == initial_hash,
        "zero_optimizer": all(
            int(row["optimizer_completed_steps"]) == 0 for row in calibration_rows
        ),
    }
    route = route_from_gates(gates)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_memory = int(torch.cuda.max_memory_reserved(0))
    diagnostics = {
        "name": "step3_foacpo_p0_zero_step_diagnostics_v1",
        "runtime_head": runtime_head,
        "protected_blocks": blocks,
        "anchor": {
            "candidate_id": anchor["anchor_candidate_id"],
            "minimum_exact_score": anchor["anchor_minimum_exact_score"],
            "exact_scores": anchor["anchor_record"]["exact_scores"],
            "reference_norm": anchor["reference_norm"],
            "optimizer_completed_steps": anchor["map_audit"]["completed_optimizer_steps"],
        },
        "gradient_coverage": coverage,
        "calibration_registry": calibration_rows,
        "selected": selected,
        "calibration": calibration,
        "mechanism": mechanism,
        "gates": gates,
        "route": route,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "peak_reserved_bytes": peak_memory,
        "vram_headroom_fraction": 1.0 - peak_memory / total_memory,
        "wall_seconds": time.time() - started,
    }
    write_json(resolve_repo_path(protocol["outputs"]["diagnostics"]), diagnostics)
    decision = {
        "name": "step3_foacpo_p0_decision_v1",
        "route": route,
        "selected_alpha": selected["alpha"] if selected else None,
        "selected_lambda": selected["lambda"] if selected else None,
        "gates": gates,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized_action": (
            "freeze a finite-radius one-step verification protocol"
            if route == "PASS_FOACPO_P0_ZERO_STEP_TRANSFER"
            else "stop this FOACPO P0 route without training"
        ),
    }
    write_json(resolve_repo_path(protocol["outputs"]["decision"]), decision)
    report = (
        "# FOACPO P0 Zero-Step Transfer Report\n\n"
        f"Route: `{route}`.\n\n"
        f"Selected alpha/lambda: `{decision['selected_alpha']}` / `{decision['selected_lambda']}`.\n\n"
        f"Calibration passed: `{gates['calibration']}`; mechanism-dev passed: `{gates['mechanism']}`.\n\n"
        "This study evaluates exact first-order AdamW displacement geometry only. "
        "It performs no update and does not authorize training. "
        "`optimizer_steps=0`, `training_authorized=false`; Core100 and heldout remain closed.\n"
    )
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "runtime_head": runtime_head,
    }
    write_json(resolve_repo_path(protocol["outputs"]["environment"]), environment)
    del model, optimizer, candidate_maps, selected_map
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(decision, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    diagnostics_path = resolve_repo_path(protocol["outputs"]["diagnostics"])
    if not decision_path.is_file() or not diagnostics_path.is_file():
        raise P0ValidationError("P0 result files are incomplete")
    decision = read_json(decision_path)
    diagnostics = read_json(diagnostics_path)
    if (
        decision.get("route") != diagnostics.get("route")
        or decision.get("optimizer_steps") != 0
        or decision.get("training_authorized") is not False
        or diagnostics.get("optimizer_steps") != 0
        or diagnostics.get("training_authorized") is not False
        or diagnostics.get("core100_opened") is not False
        or diagnostics.get("heldout_opened") is not False
    ):
        raise P0ValidationError("P0 result boundary or route differs")
    result = {
        "status": "PASS_FOACPO_P0_VERIFY_RUN",
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
