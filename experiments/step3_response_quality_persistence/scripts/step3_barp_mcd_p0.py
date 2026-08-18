"""Run batch-adaptive route-protected common descent without optimizer steps."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_pfrsc_common as pfrsc
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_p0/protocol/barp_mcd_p0_protocol.json"
)
STATUS = "prospective_before_any_BARP_MCD_model_forward"
METHOD_NAME = "BARP-MCD"
ARMS = ("baseline", "raw_mcd", "barp_mcd")
EXPECTED_BLOCKS = 8


class BarpP0ValidationError(ValueError):
    """Raised when BARP-MCD P0 violates its frozen protocol."""


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
        raise BarpP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != STATUS:
        raise BarpP0ValidationError("BARP-MCD protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or etarc_p0.sha256(artifact_path) != artifact["sha256"]
        ):
            raise BarpP0ValidationError(f"frozen input hash mismatch: {name}")
    if any(
        (
            protocol.get("optimizer_objects") != 0,
            protocol.get("scheduler_objects") != 0,
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("candidate_generation_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise BarpP0ValidationError("protocol violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise BarpP0ValidationError("entry commit is not an ancestor of HEAD")

    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED":
        raise BarpP0ValidationError("registered RP-PFRSC precursor route differs")
    angle_audit = etarc_p0.read_json(
        resolve_repo_path(protocol["method_basis"]["angle_audit_path"])
    )
    if (
        angle_audit["relaxed_fit_proof"]["strict_interval_separation"]
        is not True
        or angle_audit["interpretation"]["direction_change_required"] is not True
    ):
        raise BarpP0ValidationError("registered fixed-direction closure differs")

    design = protocol["design"]
    if (
        tuple(design["protected_certificates"]) != rp.PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != rp.TARGET_CERTIFICATES
        or tuple(float(value) for value in design["homotopy_fractions"])
        != rp.HOMOTOPY_FRACTIONS
        or float(design["protected_retention_floor"])
        != rp.PROTECTED_RETENTION_FLOOR
        or int(design["source_top_k"]) != rp.SOURCE_TOP_K
        or int(design["exact_map_evaluations_per_unit"])
        != rp.EXACT_MAP_EVALUATIONS_PER_UNIT
        or design.get("gradient_measurement_mode") != "eval"
        or design.get("selection_scope") != "same_optimizer_batch"
    ):
        raise BarpP0ValidationError("BARP-MCD finite-family registry differs")

    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 4
        or config.get("gradient_measurement_mode") != "eval"
        or config.get("model_optimizer_objects_authorized") is not False
        or config.get("scheduler_objects_authorized") is not False
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("transient_virtual_parameter_evaluations_authorized")
        is not True
        or config.get("training_authorized") is not False
        or int(config["gradient_accumulation_steps"]) != 4
    ):
        raise BarpP0ValidationError("BARP-MCD zero-step config differs")

    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise BarpP0ValidationError("risk row count differs")

    selection = protocol["selection"]
    blocks = list(selection["optimizer_blocks"])
    excluded = selection["excluded_source_block"]
    if len(blocks) != EXPECTED_BLOCKS:
        raise BarpP0ValidationError("optimizer block count differs")
    flat_safety = [
        str(sample_id) for block in blocks for sample_id in block["safety_sample_ids"]
    ]
    flat_risk = [
        int(index) for block in blocks for index in block["risk_row_indices"]
    ]
    excluded_safety = [str(value) for value in excluded["safety_sample_ids"]]
    excluded_risk = [int(value) for value in excluded["risk_row_indices"]]
    if (
        any(len(block["safety_sample_ids"]) != 4 for block in blocks)
        or any(len(block["risk_row_indices"]) != 4 for block in blocks)
        or len(set(flat_safety)) != 32
        or len(set(flat_risk)) != 32
        or set(flat_safety).intersection(excluded_safety)
        or set(flat_risk).intersection(excluded_risk)
        or len(set(flat_safety + excluded_safety)) != 40
        or len(set(flat_risk + excluded_risk)) != 40
    ):
        raise BarpP0ValidationError("optimizer/excluded source blocks drift")

    predecessor_protocol = etarc_p0.read_json(
        resolve_repo_path(protocol["predecessor"]["protocol_path"])
    )
    prior_selection = predecessor_protocol["selection"]
    if (
        blocks != prior_selection["training_blocks"]
        or excluded != prior_selection["diagnostic_block"]
    ):
        raise BarpP0ValidationError("seed-4 source block registry differs")
    return protocol, config, targets, risk_rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise BarpP0ValidationError("BARP-MCD P0 requires a clean worktree")
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
            raise BarpP0ValidationError(f"prospective output already exists: {output}")
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise BarpP0ValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    selected = [
        str(sample_id)
        for block in protocol["selection"]["optimizer_blocks"]
        for sample_id in block["safety_sample_ids"]
    ]
    if not all(sample_id in pair_map for sample_id in selected):
        raise BarpP0ValidationError("selected safety pair is absent")
    result = {
        "status": "PASS_BARP_MCD_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "seed": config["seed"],
        "certificates": list(rp.CERTIFICATES),
        "arms": list(ARMS),
        "optimizer_units": EXPECTED_BLOCKS,
        "risk_microbatches_per_unit": 4,
        "safety_microbatches_per_unit": 4,
        "source_subsets_per_unit": 127,
        "screened_sources_per_unit": rp.SOURCE_TOP_K,
        "homotopy_candidates_per_unit": 129,
        "exact_map_evaluations_per_unit": rp.EXACT_MAP_EVALUATIONS_PER_UNIT,
        "virtual_parameter_evaluations": EXPECTED_BLOCKS * len(ARMS),
        "gradient_measurement_mode": "eval",
        "selection_and_evaluation_scope": "same_optimizer_batch",
        "excluded_source_prompts": 8,
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


def _replace_keys(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {
            str(key).replace(old, new): _replace_keys(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_keys(item, old, new) for item in value]
    return value


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _replace_keys(rows, "rpfrsc", "barp")


def legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _replace_keys(rows, "barp", "rpfrsc")


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = _replace_keys(summary, "rpfrsc", "barp")
    result["same_batch"] = result.pop("diagnostic")
    return result


def evaluate_gates(
    rows: list[dict[str, Any]],
    legacy_summary: dict[str, Any],
    protocol: dict[str, Any],
    state_restored: bool,
    vram_headroom: float,
) -> dict[str, bool]:
    inherited = rp.evaluate_gates(
        rows, legacy_summary, protocol, state_restored, vram_headroom
    )
    return {
        "integrity": inherited["integrity"],
        "protected_finite_radius_search": inherited[
            "protected_finite_radius_search"
        ],
        "same_batch_common_descent": inherited["diagnostic_common_transfer"],
        "finite_step_common_descent": inherited["finite_step_common_descent"],
        "protected_same_batch_noninferiority": inherited[
            "protected_diagnostic_noninferiority"
        ],
        "target_same_batch_advantage": inherited["target_diagnostic_advantage"],
        "effect_not_weak": inherited["effect_not_weak"],
    }


def select_route(gates: dict[str, bool]) -> str:
    order = (
        ("integrity", "STOP_BARP_MCD_P0_INVALID"),
        ("protected_finite_radius_search", "STOP_BARP_MCD_P0_PROTECTED_SEARCH_FAILED"),
        ("same_batch_common_descent", "STOP_BARP_MCD_P0_SAME_BATCH_COMMON_DESCENT_FAILED"),
        ("finite_step_common_descent", "STOP_BARP_MCD_P0_FINITE_STEP_COMMON_DESCENT_FAILED"),
        ("protected_same_batch_noninferiority", "STOP_BARP_MCD_P0_PROTECTED_NONINFERIORITY_FAILED"),
        ("target_same_batch_advantage", "STOP_BARP_MCD_P0_TARGET_ADVANTAGE_FAILED"),
        ("effect_not_weak", "STOP_BARP_MCD_P0_EFFECT_TOO_WEAK"),
    )
    for gate, route in order:
        if not gates[gate]:
            return route
    return "PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL"


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
        raise BarpP0ValidationError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    initial_state = rp._snapshot(parameters)
    pair_map, structural = rp.structural_features(
        tokenizer, targets, config, protocol
    )
    blocks = protocol["selection"]["optimizer_blocks"]
    selected_risk = {
        int(index) for block in blocks for index in block["risk_row_indices"]
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

    rows: list[dict[str, Any]] = []
    for block in blocks:
        block_started = time.time()
        risk_indices = [int(value) for value in block["risk_row_indices"]]
        safety_ids = [str(value) for value in block["safety_sample_ids"]]
        gradients, coverage = rp.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            risk_indices,
            safety_ids,
            config,
            train_mode=False,
        )
        initial_losses = dict(coverage["loss_means"])
        if not rp._state_equal(parameters, initial_state):
            raise BarpP0ValidationError("batch gradients changed parameters")
        raw_baseline = rp._baseline_gradient(
            gradients, float(config["safety_mix"])
        )
        budget = float(config["auxiliary_gradient_norm_ratio"]) * (
            gradient_math.gradient_norm(gradients["uniform"])
        )
        selected_direction, full_direction, search = (
            pfrsc.enumerate_protected_homotopy(
                raw_baseline,
                {name: gradients[name] for name in rp.CERTIFICATES},
                budget,
                float(config["safety_mix"]),
                float(config["max_grad_norm"]),
                float(config["adam_epsilon"]),
                list(protocol["design"]["protected_certificates"]),
                list(protocol["design"]["target_certificates"]),
                float(protocol["design"]["protected_retention_floor"]),
                list(protocol["design"]["homotopy_fractions"]),
                int(protocol["design"]["source_top_k"]),
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
            "baseline": rp.evaluate_arm(
                "baseline",
                raw_baseline,
                None,
                gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                block,
                config,
                reference_norm,
            ),
            "raw_mcd": rp.evaluate_arm(
                "raw_mcd",
                raw_baseline,
                full_direction,
                gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                block,
                config,
                reference_norm,
            ),
            "rpfrsc_mcd": rp.evaluate_arm(
                "barp_mcd",
                raw_baseline,
                selected_direction,
                gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                block,
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
            "selection_scope": "same_optimizer_batch",
            "evaluation_scope": "same_optimizer_batch",
            "search": search,
            "arms": arms,
            "wall_seconds": time.time() - block_started,
        }
        rows.append(row)
        del gradients, raw_baseline, selected_direction, full_direction
        print(
            f"BARP {block['block_id']} target="
            f"{search['selected']['minimum_target_exact_score']:.6f} "
            f"gain={search['selected_target_gain_over_raw']:.6f} "
            f"retention={search['selected']['minimum_protected_retention']:.6f} "
            f"lambda={search['selected']['homotopy_fraction']:.3f} "
            f"batch_margin={arms['rpfrsc_mcd']['diagnostic_scores']['minimum_score']:.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    final_hash = model_common.trainable_state_hash(parameters)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    legacy = rp.summarize_rows(rows, protocol)
    summary = public_summary(legacy)
    gates = evaluate_gates(
        rows, legacy, protocol, initial_hash == final_hash, headroom
    )
    route = select_route(gates)
    outputs = {
        name: resolve_repo_path(value) for name, value in protocol["outputs"].items()
    }
    stored_rows = public_rows(rows)
    diagnostics_payload = {
        "name": "step3_barp_mcd_p0_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "summary": summary,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "state_restored": initial_hash == final_hash,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "same_batch_only": True,
        "independent_transfer_claimed": False,
    }
    environment = {
        "name": "step3_barp_mcd_p0_environment_v1",
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
        "name": "step3_barp_mcd_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "summary": summary,
        "matched_training_protocol_authorized": route
        == "PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        "independent_transfer_claimed": False,
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
    search_pair = summary["search"]["selected_target_gain_over_raw"]
    target_pair = summary["paired"]["barp_minus_raw_target_minimum_score"]
    macro_pair = summary["paired"]["target_macro_actual_relative_gain"]
    report = f"""# BARP-MCD Seed-4 Same-Batch Zero-Step P0 Report

Route: `{route}`.

Eight optimizer batches each recomputed the 127 subset endpoints, fixed Top-16
screen, and 129 raw-anchored route-protected homotopy candidates. Selection
and transient finite-step evaluation used the same batch by algorithm design;
no independent transfer claim is made.

The minimum protected exact-score retention was
`{summary['search']['minimum_selected_protected_retention']:.6f}`. Mean
same-batch target-bottleneck construction gain over raw MCD was
`{search_pair['bootstrap']['mean']:.6f}`, with 95% bootstrap interval
`[{search_pair['bootstrap']['ci95_lower']:.6f},
{search_pair['bootstrap']['ci95_upper']:.6f}]`.

BARP-MCD's minimum same-batch exact score was
`{summary['same_batch']['barp_mcd']['minimum_worst_score']:.6f}`. Mean paired
same-batch target-bottleneck advantage over raw MCD was
`{target_pair['bootstrap']['mean']:.6f}`, with exact sign-test
`p={target_pair['sign_test']['exact_two_sided_p']:.8g}`. Macro-average
finite-loss relative gain over the three target objectives was
`{macro_pair['bootstrap']['mean']:.6f}`.

All finite losses came from transient direct applications of the exact
first-step displacement at learning rate `{config['learning_rate']}`, followed
by exact restoration. No optimizer or scheduler was created,
`optimizer_steps=0`, and no checkpoint, generated response, Core100 row, or
heldout row was opened.
"""
    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_jsonl_frozen(outputs["block_rows"], stored_rows)
    etarc_p0.write_json_frozen(outputs["gradient_diagnostics"], diagnostics_payload)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_barp_mcd_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_barp_mcd_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_barp_mcd_p0.py verify-run"
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
    del model, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    stored_rows = etarc_p0.read_jsonl(
        resolve_repo_path(protocol["outputs"]["block_rows"])
    )
    rows = legacy_rows(stored_rows)
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
    legacy = rp.summarize_rows(rows, protocol)
    summary = public_summary(legacy)
    gates = evaluate_gates(
        rows,
        legacy,
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
        raise BarpP0ValidationError("stored and recomputed BARP-MCD route differs")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise BarpP0ValidationError("BARP-MCD artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise BarpP0ValidationError("stored BARP-MCD boundary differs")
    result = {
        "status": "PASS_BARP_MCD_P0_VERIFY",
        "route": route,
        "optimizer_units": len(rows),
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
