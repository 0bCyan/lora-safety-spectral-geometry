"""Run the prospective seed-5 BARP-MCD eight-step matched confirmation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_barp_mcd_training as base
import step3_barp_mcd_training_q1r as q1r
import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_ramp_common as gradient_math
import step3_qetarc_common as gradient_ops
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_d1/protocol/barp_mcd_d1_protocol.json"
)
STATUS = "prospective_before_any_BARP_MCD_D1_optimizer_step"
ARMS = ("raw_mcd", "barp_mcd")
TARGETS = rp.TARGET_CERTIFICATES
PROTECTED = rp.PROTECTED_CERTIFICATES


class BarpD1ValidationError(ValueError):
    """Raised when BARP-MCD D1 violates its prospective protocol."""


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
        raise BarpD1ValidationError(f"path escapes repository: {value}") from exc
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


def seeded_digest(prefix: str, value: str) -> str:
    return hashlib.sha256((prefix + value).encode("utf-8")).hexdigest()


def expected_safety_selection(
    targets: list[dict[str, Any]],
) -> tuple[list[list[str]], list[str], list[str]]:
    by_category: dict[str, set[str]] = {}
    for row in targets:
        by_category.setdefault(str(row["category"]), set()).add(str(row["sample_id"]))
    all_ids = set().union(*by_category.values()) if by_category else set()
    if len(by_category) != 8 or len(all_ids) != 40 or any(
        not values for values in by_category.values()
    ):
        raise BarpD1ValidationError("D1 safety source is not 40 pairs across eight categories")
    diagnostic: list[str] = []
    categories = sorted(by_category)
    for category in categories:
        diagnostic.append(
            min(
                by_category[category],
                key=lambda sample_id: seeded_digest(
                    "barp-d1-seed5-diagnostic|", sample_id
                ),
            )
        )
    training_ids = sorted(
        set().union(*by_category.values()).difference(diagnostic),
        key=lambda sample_id: seeded_digest("barp-d1-seed5-training|", sample_id),
    )
    if len(training_ids) != 32:
        raise BarpD1ValidationError("D1 safety training split differs")
    return (
        [training_ids[offset : offset + 4] for offset in range(0, 32, 4)],
        diagnostic,
        categories,
    )


def expected_risk_order(seed: int, count: int) -> list[int]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator).tolist()


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise BarpD1ValidationError("invalid percentile input")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean(
    values: list[float], resamples: int, seed: int
) -> dict[str, float | int]:
    if not values or resamples <= 0:
        raise BarpD1ValidationError("invalid bootstrap registry")
    rng = random.Random(seed)
    count = len(values)
    draws = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return {
        "count": count,
        "resamples": resamples,
        "seed": seed,
        "mean": statistics.fmean(values),
        "lower_2_5_percent": percentile(draws, 0.025),
        "upper_97_5_percent": percentile(draws, 0.975),
        "minimum": min(values),
        "maximum": max(values),
    }


def exact_two_sided_sign_test(values: list[float], tie_tolerance: float = 0.0) -> dict[str, Any]:
    positive = sum(value > tie_tolerance for value in values)
    negative = sum(value < -tie_tolerance for value in values)
    ties = len(values) - positive - negative
    non_ties = positive + negative
    if non_ties == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(non_ties, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**non_ties))
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "non_ties": non_ties,
        "two_sided_p_value": p_value,
    }


def checkpoint_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        raise BarpD1ValidationError(f"checkpoint directory is absent: {path}")
    forbidden = {
        "optimizer.pt",
        "optimizer.bin",
        "training_state.pt",
        "scheduler.pt",
        "pytorch_model.bin",
        "model.safetensors",
    }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    names = {item.name for item in files}
    if not {"adapter_model.safetensors", "adapter_config.json"}.issubset(names):
        raise BarpD1ValidationError("adapter checkpoint is incomplete")
    if names.intersection(forbidden) or any(name.startswith("model-") for name in names):
        raise BarpD1ValidationError("checkpoint contains optimizer or base-model state")
    if any(item.is_symlink() for item in files):
        raise BarpD1ValidationError("checkpoint contains a symbolic link")
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": etarc_p0.sha256(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    ]


def resolve_external_root(protocol: dict[str, Any], require_absent: bool) -> Path:
    root = Path(protocol["checkpoint_root"])
    if not root.is_absolute():
        raise BarpD1ValidationError("D1 checkpoint root must be absolute")
    root = root.resolve()
    try:
        root.relative_to(REPO)
    except ValueError:
        pass
    else:
        raise BarpD1ValidationError("D1 checkpoint root must be outside the repository")
    if require_absent and root.exists():
        raise BarpD1ValidationError("prospective D1 checkpoint root already exists")
    if not require_absent and not root.is_dir():
        raise BarpD1ValidationError("frozen D1 checkpoint root is absent")
    return root


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != STATUS:
        raise BarpD1ValidationError("BARP-MCD D1 protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise BarpD1ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("training_authorized") is not True
        or int(protocol.get("optimizer_steps_per_arm", -1)) != 8
        or int(protocol.get("total_authorized_optimizer_steps", -1)) != 16
        or protocol.get("checkpoint_writes_authorized") is not True
        or int(protocol.get("checkpoint_writes_exact", -1)) != 2
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise BarpD1ValidationError("D1 authorization boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise BarpD1ValidationError("D1 entry commit is not an ancestor")
    precursor = etarc_p0.read_json(resolve_repo_path(protocol["precursor"]["decision_path"]))
    if precursor.get("route") != "PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL":
        raise BarpD1ValidationError("D1 precursor route differs")

    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 5
        or int(config["optimizer_steps"]) != 8
        or int(config["optimizer_steps_authorized"]) != 16
        or int(config["gradient_accumulation_steps"]) != 4
        or config.get("gradient_measurement_mode") != "eval"
        or config.get("training_authorized") is not True
        or config.get("scheduler_objects_authorized") is not False
        or config.get("checkpoint_writes_authorized") is not True
        or int(config.get("checkpoint_writes_exact", -1)) != 2
        or float(config["weight_decay"]) != 0.0
    ):
        raise BarpD1ValidationError("D1 config differs")
    design = protocol["design"]
    if (
        tuple(design["protected_certificates"]) != PROTECTED
        or tuple(design["target_certificates"]) != TARGETS
        or tuple(design["homotopy_fractions"]) != rp.HOMOTOPY_FRACTIONS
        or int(design["source_top_k"]) != rp.SOURCE_TOP_K
        or float(design["protected_retention_floor"]) != rp.PROTECTED_RETENTION_FLOOR
        or design.get("optimizer_map") != "exact_current_state_torch_AdamW"
        or design.get("endpoint_tensor_policy")
        != "score_delete_then_reconstruct_screened_masks"
    ):
        raise BarpD1ValidationError("D1 method registry differs")

    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise BarpD1ValidationError("D1 risk row count differs")
    safety_blocks, diagnostic_ids, categories = expected_safety_selection(targets)
    risk_order = expected_risk_order(int(config["seed"]), len(risk_rows))
    expected_blocks = [
        {
            "block_id": f"training_{index + 1:02d}",
            "risk_row_indices": risk_order[index * 4 : (index + 1) * 4],
            "safety_sample_ids": safety_blocks[index],
        }
        for index in range(8)
    ]
    expected_diagnostic = {
        "block_id": "paired_diagnostic_8category_seed5",
        "risk_row_indices": risk_order[32:40],
        "safety_sample_ids": diagnostic_ids,
        "safety_categories": categories,
    }
    if (
        protocol["selection"]["optimizer_blocks"] != expected_blocks
        or protocol["selection"]["diagnostic_block"] != expected_diagnostic
    ):
        raise BarpD1ValidationError("D1 prospective split differs")
    return protocol, config, targets, risk_rows


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise BarpD1ValidationError("D1 requires a clean worktree")
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
            raise BarpD1ValidationError("prospective D1 outputs already exist")
    resolve_external_root(protocol, require_absent=True)
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    resolve_external_root(protocol, require_absent=True)
    if not torch.cuda.is_available():
        raise BarpD1ValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    selected = [
        str(sample_id)
        for block in protocol["selection"]["optimizer_blocks"]
        for sample_id in block["safety_sample_ids"]
    ] + [
        str(value)
        for value in protocol["selection"]["diagnostic_block"]["safety_sample_ids"]
    ]
    if len(selected) != 40 or len(set(selected)) != 40 or not all(
        sample_id in pair_map for sample_id in selected
    ):
        raise BarpD1ValidationError("D1 safety selection is absent or overlapping")
    result = {
        "status": "PASS_BARP_MCD_D1_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "gpu": torch.cuda.get_device_name(0),
        "arms": list(ARMS),
        "optimizer_steps_per_arm": 8,
        "total_authorized_optimizer_steps": 16,
        "diagnostic_units": 8,
        "bootstrap_resamples": int(protocol["statistics"]["bootstrap_resamples"]),
        "source_subsets_per_step": 127,
        "homotopy_candidates_per_step": 129,
        "maximum_retained_endpoint_directions": 1,
        "checkpoint_writes_exact": 2,
        "checkpoint_root_absent": True,
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def evaluate_diagnostic_rows(
    model: Any,
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for risk_index, sample_id, category in zip(
        diagnostic["risk_row_indices"],
        diagnostic["safety_sample_ids"],
        diagnostic["safety_categories"],
        strict=True,
    ):
        losses = rp.evaluate_block_losses(
            model,
            risk_features,
            pair_map,
            [int(risk_index)],
            [str(sample_id)],
            config,
        )
        rows.append(
            {
                "sample_id": str(sample_id),
                "category": str(category),
                "risk_row_index": int(risk_index),
                "losses": {name: float(losses[name]) for name in rp.CERTIFICATES},
            }
        )
    return rows


def save_adapter(model: Any, checkpoint_root: Path, arm: str) -> list[dict[str, Any]]:
    checkpoint = checkpoint_root / arm
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    return checkpoint_inventory(checkpoint)


def train_arm(
    arm: str,
    protocol: dict[str, Any],
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    runtime_head: str,
    checkpoint_root: Path,
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
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise BarpD1ValidationError("unexpected trainable parameter inventory")
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
    initial_diagnostic = evaluate_diagnostic_rows(
        model, risk_features, pair_map, diagnostic, config
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
        selected, full, search = q1r.enumerate_memory_bounded_homotopy(
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
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for _, parameter in parameters
        ):
            raise FloatingPointError(f"non-finite D1 gradient at step {step_index}")
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in parameters],
                float(config["max_grad_norm"]),
            )
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        map_error = base._update_map_error(parameters, before, predicted, learning_rate)
        final_losses = rp.evaluate_block_losses(
            model, risk_features, pair_map, block_risk, block_safety, config
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
            f"BARP-D1 {arm} step {step_index}/8 "
            f"lambda={row['applied_homotopy_fraction']:.3f} "
            f"retention={search['selected']['minimum_protected_retention']:.6f} "
            f"target_gain={search['selected_target_gain_over_raw']:.6f} "
            f"map_error={map_error['relative_l2_error']:.3e} "
            f"vram={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients, raw_baseline, selected, full, direction, applied_gradient, predicted, before

    final_diagnostic = evaluate_diagnostic_rows(
        model, risk_features, pair_map, diagnostic, config
    )
    final_hash = model_common.trainable_state_hash(parameters)
    checkpoint_files = save_adapter(model, checkpoint_root, arm)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_barp_mcd_d1_arm_manifest_v1",
        "runtime_head": runtime_head,
        "arm": arm,
        "protocol_sha256": etarc_p0.sha256(resolve_repo_path(protocol["protocol_path"])),
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "completed_steps": len(trace),
        "optimizer_steps": len(trace),
        "skipped_updates": 0,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "block_ids": [row["block_id"] for row in trace],
        "initial_diagnostic_rows": initial_diagnostic,
        "final_diagnostic_rows": final_diagnostic,
        "maximum_update_map_relative_l2_error": max(
            row["map_error"]["relative_l2_error"] for row in trace
        ),
        "minimum_raw_anchor_score": min(
            row["search"]["raw_anchor"]["minimum_exact_score"] for row in trace
        ),
        "minimum_selected_protected_retention": min(
            row["search"]["selected"]["minimum_protected_retention"] for row in trace
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
        "checkpoint_writes": 1,
        "checkpoint_relative_path": arm,
        "checkpoint_inventory": checkpoint_files,
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


def paired_diagnostic_analysis(
    manifests: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    raw = manifests["raw_mcd"]
    barp = manifests["barp_mcd"]
    raw_initial = {row["sample_id"]: row for row in raw["initial_diagnostic_rows"]}
    barp_initial = {row["sample_id"]: row for row in barp["initial_diagnostic_rows"]}
    raw_final = {row["sample_id"]: row for row in raw["final_diagnostic_rows"]}
    barp_final = {row["sample_id"]: row for row in barp["final_diagnostic_rows"]}
    if not (
        set(raw_initial) == set(barp_initial) == set(raw_final) == set(barp_final)
        and len(raw_initial) == 8
    ):
        raise BarpD1ValidationError("D1 diagnostic unit registry differs")
    unit_rows: list[dict[str, Any]] = []
    for sample_id in protocol["selection"]["diagnostic_block"]["safety_sample_ids"]:
        initial = raw_initial[sample_id]
        advantages = {
            name: (
                float(raw_final[sample_id]["losses"][name])
                - float(barp_final[sample_id]["losses"][name])
            )
            / max(abs(float(initial["losses"][name])), 1e-8)
            for name in rp.CERTIFICATES
        }
        unit_rows.append(
            {
                "sample_id": sample_id,
                "category": initial["category"],
                "risk_row_index": initial["risk_row_index"],
                "normalized_barp_advantage": advantages,
                "target_macro_advantage": statistics.fmean(advantages[name] for name in TARGETS),
                "protected_disadvantage": {name: -advantages[name] for name in PROTECTED},
            }
        )
    stats = protocol["statistics"]
    resamples = int(stats["bootstrap_resamples"])
    seed = int(stats["bootstrap_seed"])
    protected = {
        name: bootstrap_mean(
            [row["protected_disadvantage"][name] for row in unit_rows],
            resamples,
            seed + index,
        )
        for index, name in enumerate(PROTECTED)
    }
    target_values = [row["target_macro_advantage"] for row in unit_rows]
    return {
        "name": "step3_barp_mcd_d1_paired_diagnostic_analysis_v1",
        "formula": "(raw_final_loss-barp_final_loss)/max(abs(shared_initial_loss),1e-8)",
        "unit_rows": unit_rows,
        "protected_disadvantage": protected,
        "target_macro_advantage": {
            **bootstrap_mean(target_values, resamples, seed + 100),
            "sign_test": exact_two_sided_sign_test(
                target_values, float(stats["sign_test_tie_tolerance"])
            ),
        },
    }


def compare_manifests(
    manifests: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, bool]:
    tolerance = float(protocol["gates"]["matching_tolerance"])
    map_tolerance = float(protocol["gates"]["optimizer_map_relative_error_max"])
    values = list(manifests.values())
    raw_initial = {
        row["sample_id"]: row["losses"]
        for row in manifests["raw_mcd"]["initial_diagnostic_rows"]
    }
    barp_initial = {
        row["sample_id"]: row["losses"]
        for row in manifests["barp_mcd"]["initial_diagnostic_rows"]
    }
    initial_diagnostic_equal = raw_initial.keys() == barp_initial.keys() and all(
        abs(float(raw_initial[sample_id][name]) - float(barp_initial[sample_id][name]))
        <= tolerance
        for sample_id in raw_initial
        for name in rp.CERTIFICATES
    )
    barp = manifests["barp_mcd"]
    protected_stats = diagnostic["protected_disadvantage"]
    target_stats = diagnostic["target_macro_advantage"]
    return {
        "arms_complete": set(manifests) == set(ARMS),
        "steps_exact": all(item["completed_steps"] == 8 for item in values),
        "no_skipped_updates": all(item["skipped_updates"] == 0 for item in values),
        "initial_state_equal": len({item["initial_trainable_state_sha256"] for item in values}) == 1,
        "block_order_equal": len({tuple(item["block_ids"]) for item in values}) == 1,
        "initial_diagnostic_equal": initial_diagnostic_equal,
        "optimizer_map_exact": all(
            item["maximum_update_map_relative_l2_error"] <= map_tolerance for item in values
        ),
        "raw_common_descent": all(
            item["minimum_raw_anchor_score"] >= float(protocol["gates"]["raw_anchor_score_min"])
            for item in values
        ),
        "barp_protection": barp["minimum_selected_protected_retention"] + tolerance
        >= float(protocol["design"]["protected_retention_floor"]),
        "barp_target_nondegradation": barp["minimum_selected_target_gain"] >= -tolerance,
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
        "checkpoints": all(
            item["checkpoint_writes"] == 1
            and any(row["path"] == "adapter_model.safetensors" for row in item["checkpoint_inventory"])
            and any(row["path"] == "adapter_config.json" for row in item["checkpoint_inventory"])
            for item in values
        ),
        "protected_diagnostic": all(
            float(item["mean"]) <= float(protocol["gates"]["protected_mean_disadvantage_max"])
            and float(item["upper_97_5_percent"])
            <= float(protocol["gates"]["protected_bootstrap_upper_max"])
            and float(item["maximum"])
            <= float(protocol["gates"]["protected_worst_unit_disadvantage_max"])
            for item in protected_stats.values()
        ),
        "target_diagnostic_mean": float(target_stats["mean"])
        >= float(protocol["gates"]["target_macro_mean_advantage_min"]),
        "target_diagnostic_bootstrap": float(target_stats["lower_2_5_percent"])
        > float(protocol["gates"]["target_macro_bootstrap_lower_min"]),
        "target_diagnostic_sign": float(target_stats["sign_test"]["two_sided_p_value"])
        <= float(protocol["gates"]["target_macro_sign_test_p_max"]),
        "boundary": all(
            item["scheduler_objects"] == 0
            and not item["candidate_outputs_generated"]
            and not item["core100_opened"]
            and not item["heldout_opened"]
            for item in values
        ),
    }


def select_route(checks: dict[str, bool]) -> str:
    order = (
        ("arms_complete", "STOP_BARP_MCD_D1_INCOMPLETE"),
        ("steps_exact", "STOP_BARP_MCD_D1_STEP_COUNT_FAILED"),
        ("no_skipped_updates", "STOP_BARP_MCD_D1_NUMERICAL_FAILED"),
        ("initial_state_equal", "STOP_BARP_MCD_D1_MATCHING_FAILED"),
        ("block_order_equal", "STOP_BARP_MCD_D1_MATCHING_FAILED"),
        ("initial_diagnostic_equal", "STOP_BARP_MCD_D1_MATCHING_FAILED"),
        ("optimizer_map_exact", "STOP_BARP_MCD_D1_OPTIMIZER_MAP_FAILED"),
        ("raw_common_descent", "STOP_BARP_MCD_D1_RAW_ANCHOR_FAILED"),
        ("barp_protection", "STOP_BARP_MCD_D1_PROTECTION_FAILED"),
        ("barp_target_nondegradation", "STOP_BARP_MCD_D1_TARGET_FAILED"),
        ("barp_assignment_exercised", "STOP_BARP_MCD_D1_ASSIGNMENT_NOT_EXERCISED"),
        ("finite_step_common_descent", "STOP_BARP_MCD_D1_FINITE_STEP_FAILED"),
        ("vram", "STOP_BARP_MCD_D1_VRAM_FAILED"),
        ("checkpoints", "STOP_BARP_MCD_D1_CHECKPOINT_FAILED"),
        ("protected_diagnostic", "STOP_BARP_MCD_D1_DIAGNOSTIC_PROTECTION_FAILED"),
        ("target_diagnostic_mean", "STOP_BARP_MCD_D1_DIAGNOSTIC_TARGET_MEAN_FAILED"),
        ("target_diagnostic_bootstrap", "STOP_BARP_MCD_D1_DIAGNOSTIC_TARGET_BOOTSTRAP_FAILED"),
        ("target_diagnostic_sign", "STOP_BARP_MCD_D1_DIAGNOSTIC_TARGET_SIGN_FAILED"),
        ("boundary", "STOP_BARP_MCD_D1_BOUNDARY_FAILED"),
    )
    for gate, route in order:
        if not checks.get(gate, False):
            return route
    return "PASS_BARP_MCD_D1_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL"


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
                "rows": len(etarc_p0.read_jsonl(path)) if path.suffix == ".jsonl" else None,
            }
        )
    return artifacts


def verify_external_checkpoints(
    manifests: dict[str, dict[str, Any]], checkpoint_root: Path
) -> int:
    verified = 0
    expected_directories = set(ARMS)
    actual_directories = {item.name for item in checkpoint_root.iterdir() if item.is_dir()}
    if actual_directories != expected_directories or any(
        item.is_file() for item in checkpoint_root.iterdir()
    ):
        raise BarpD1ValidationError("external checkpoint root contains unexpected entries")
    for arm in ARMS:
        actual = checkpoint_inventory(checkpoint_root / arm)
        if actual != manifests[arm]["checkpoint_inventory"]:
            raise BarpD1ValidationError(f"external checkpoint hash mismatch: {arm}")
        verified += len(actual)
    return verified


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_start(path, protocol)
    checkpoint_root = resolve_external_root(protocol, require_absent=True)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifests = {
        arm: train_arm(
            arm, protocol, config, targets, risk_rows, runtime_head, checkpoint_root
        )
        for arm in ARMS
    }
    diagnostic = paired_diagnostic_analysis(manifests, protocol)
    diagnostic_path = resolve_repo_path(protocol["outputs"]["paired_diagnostic_analysis"])
    write_json(diagnostic_path, diagnostic)
    checks = compare_manifests(manifests, diagnostic, protocol)
    route = select_route(checks)
    decision = {
        "name": "step3_barp_mcd_d1_decision_v1",
        "route": route,
        "checks": checks,
        "arms": {
            arm: {
                "optimizer_steps": manifest["optimizer_steps"],
                "final_trainable_state_sha256": manifest["final_trainable_state_sha256"],
                "maximum_update_map_relative_l2_error": manifest[
                    "maximum_update_map_relative_l2_error"
                ],
                "minimum_raw_anchor_score": manifest["minimum_raw_anchor_score"],
                "minimum_selected_protected_retention": manifest[
                    "minimum_selected_protected_retention"
                ],
                "minimum_selected_target_gain": manifest["minimum_selected_target_gain"],
                "nonzero_barp_choices": manifest["nonzero_barp_choices"],
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
        == "PASS_BARP_MCD_D1_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL",
        "main_training_authorized": False,
        "wall_seconds": time.time() - started,
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    write_json(decision_path, decision)
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    target_stats = diagnostic["target_macro_advantage"]
    report_path.write_text(
        "# BARP-MCD D1 Seed-5 Matched Confirmation\n\n"
        f"Decision: `{route}`\n\n"
        f"Both arms completed `{decision['total_optimizer_steps']}` total optimizer steps. "
        f"BARP selected a nonzero candidate on `{manifests['barp_mcd']['nonzero_barp_choices']}/8` steps.\n\n"
        f"Paired target macro advantage: `{target_stats['mean']:.8f}`; "
        f"95% paired bootstrap interval "
        f"`[{target_stats['lower_2_5_percent']:.8f}, {target_stats['upper_97_5_percent']:.8f}]`; "
        f"two-sided exact sign-test p=`{target_stats['sign_test']['two_sided_p_value']:.8f}`.\n\n"
        "Exactly one external LoRA adapter per arm was frozen and SHA-256 inventoried. "
        "No response was generated; Core100 and heldout remained sealed. A pass authorizes "
        "only a separately frozen, checkpoint-hash-bound generated evaluation protocol.\n",
        encoding="utf-8",
    )
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    reproduction = {
        "name": "step3_barp_mcd_d1_reproduction_manifest_v1",
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "checkpoint_root": checkpoint_root.as_posix(),
        "checkpoint_inventories": {
            arm: manifests[arm]["checkpoint_inventory"] for arm in ARMS
        },
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1.py preflight",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1.py run",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1.py verify-run",
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
    diagnostic = paired_diagnostic_analysis(manifests, protocol)
    stored_diagnostic = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["paired_diagnostic_analysis"])
    )
    if stored_diagnostic != diagnostic:
        raise BarpD1ValidationError("D1 paired diagnostic does not reproduce")
    checks = compare_manifests(manifests, diagnostic, protocol)
    route = select_route(checks)
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    reproduction = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if decision["checks"] != checks or decision["route"] != route:
        raise BarpD1ValidationError("D1 decision does not reproduce")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise BarpD1ValidationError(f"D1 artifact hash mismatch: {artifact['path']}")
    checkpoint_files_verified = verify_external_checkpoints(
        manifests, resolve_external_root(protocol, require_absent=False)
    )
    result = {
        "status": "PASS_BARP_MCD_D1_VERIFY",
        "route": route,
        "checks": checks,
        "artifacts_verified": len(reproduction["artifacts"]),
        "checkpoint_files_verified": checkpoint_files_verified,
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
    path = resolve_repo_path(args.protocol)
    if args.command == "preflight":
        preflight(path)
    elif args.command == "run":
        run(path)
    else:
        verify_run(path)


if __name__ == "__main__":
    main()
