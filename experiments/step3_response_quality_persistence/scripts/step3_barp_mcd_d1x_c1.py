"""Run the single-route BARP-MCD D1X-C1 zero-step qualification."""

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

import step3_barp_mcd_d1x as d1x
import step3_etarc_p0 as etarc_p0
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as rp
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_barp_mcd_d1x_c1/protocol/barp_mcd_d1x_c1_protocol.json"
)
STATUS = "prospective_before_any_D1X_C1_qualification_forward"


class C1ValidationError(ValueError):
    """Raised when C1 violates its single-route qualification registry."""


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
        raise C1ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def summarize_conditional(
    descriptor: dict[str, Any], proposal_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(proposal_rows) != 4:
        raise C1ValidationError("C1 requires four qualification proposals")
    target_clusters = d1x.cluster_summary(
        [statistics.fmean(row["target_macro_deltas"]) for row in proposal_rows]
    )
    protected = {}
    for name in rp.PROTECTED_CERTIFICATES:
        disadvantages = [
            -float(support["exact_delta_scores"][name])
            for row in proposal_rows
            for support in row["support_rows"]
        ]
        protected[name] = {
            "mean_disadvantage": statistics.fmean(disadvantages),
            "worst_disadvantage": max(disadvantages),
        }
    components = {}
    for name in rp.TARGET_CERTIFICATES:
        values = [
            float(support["exact_delta_scores"][name])
            for row in proposal_rows
            for support in row["support_rows"]
        ]
        components[name] = {
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "positive": sum(value > 0.0 for value in values),
            "negative": sum(value < 0.0 for value in values),
            "fallback_ties": sum(value == 0.0 for value in values),
        }
    applied_retentions = [
        float(row["minimum_local_protected_retention"])
        if row["route_exercised"]
        else 1.0
        for row in proposal_rows
    ]
    return {
        **descriptor,
        "proposal_activity": sum(bool(row["route_exercised"]) for row in proposal_rows),
        "minimum_candidate_local_protected_retention": min(
            float(row["minimum_local_protected_retention"]) for row in proposal_rows
        ),
        "minimum_applied_local_protected_retention": min(applied_retentions),
        "target_clusters": target_clusters,
        "target_components": components,
        "protected": protected,
        "optimizer_completed_steps": [
            int(row["optimizer_completed_steps"]) for row in proposal_rows
        ],
        "proposal_rows": proposal_rows,
    }


def qualification_checks(
    summary: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    target = summary["target_clusters"]
    return {
        "route_activity": int(summary["proposal_activity"])
        >= int(gates["proposal_activity_min"]),
        "applied_local_protection": float(
            summary["minimum_applied_local_protected_retention"]
        )
        + 1e-12
        >= float(gates["applied_local_protected_retention_min"]),
        "target_effect": float(target["mean"])
        >= float(gates["target_macro_mean_min"]),
        "target_stability": float(target["studentized_lower_score"])
        > float(gates["target_stability_lower_min"]),
        "target_cluster_sign": int(target["positive"])
        >= int(gates["positive_proposal_clusters_min"])
        and int(target["negative"]) <= int(gates["negative_proposal_clusters_max"]),
        "target_components": all(
            float(item["mean"]) > float(gates["target_component_mean_min"])
            for item in summary["target_components"].values()
        ),
        "protected_mean": all(
            float(item["mean_disadvantage"])
            <= float(gates["protected_mean_disadvantage_max"])
            for item in summary["protected"].values()
        ),
        "protected_worst": all(
            float(item["worst_disadvantage"])
            <= float(gates["protected_worst_disadvantage_max"])
            for item in summary["protected"].values()
        ),
        "zero_optimizer_state": all(
            int(value) == 0 for value in summary["optimizer_completed_steps"]
        ),
    }


def select_route(checks: dict[str, bool]) -> str:
    order = (
        ("zero_optimizer_state", "STOP_BARP_MCD_D1X_C1_OPTIMIZER_STATE_DRIFT"),
        ("route_activity", "STOP_BARP_MCD_D1X_C1_ROUTE_ACTIVITY_FAILED"),
        ("applied_local_protection", "STOP_BARP_MCD_D1X_C1_LOCAL_PROTECTION_FAILED"),
        ("protected_mean", "STOP_BARP_MCD_D1X_C1_TRANSFER_PROTECTION_FAILED"),
        ("protected_worst", "STOP_BARP_MCD_D1X_C1_TRANSFER_PROTECTION_FAILED"),
        ("target_effect", "STOP_BARP_MCD_D1X_C1_TARGET_EFFECT_TOO_WEAK"),
        ("target_stability", "STOP_BARP_MCD_D1X_C1_TARGET_STABILITY_FAILED"),
        ("target_cluster_sign", "STOP_BARP_MCD_D1X_C1_TARGET_SIGN_FAILED"),
        ("target_components", "STOP_BARP_MCD_D1X_C1_TARGET_COMPONENT_FAILED"),
    )
    for gate, route in order:
        if not checks.get(gate, False):
            return route
    return "PASS_BARP_MCD_D1X_C1_AUTHORIZE_MATCHED_TRAINING_PROTOCOL"


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != STATUS:
        raise C1ValidationError("C1 protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise C1ValidationError(f"C1 frozen input hash mismatch: {name}")
    if (
        int(protocol.get("optimizer_objects", -1)) != 1
        or int(protocol.get("optimizer_steps", -1)) != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("checkpoint_writes_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("sealed_diagnostic_opened") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise C1ValidationError("C1 zero-step boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise C1ValidationError("C1 entry commit is not an ancestor")
    prior = etarc_p0.read_json(resolve_repo_path(protocol["basis"]["decision_path"]))
    analysis = etarc_p0.read_json(resolve_repo_path(protocol["basis"]["analysis_path"]))
    if (
        prior.get("route") != "STOP_BARP_MCD_D1X_DISCOVERY_NO_STABLE_ROUTE"
        or analysis.get("exploratory_unique_target_sign_route", {}).get("route_id")
        != "mask072_lambda0.500"
        or analysis.get("boundary", {}).get("qualification_gradients_opened") is not False
    ):
        raise C1ValidationError("C1 post-selection basis differs")
    descriptor = protocol["design"]["route_descriptor"]
    if (
        descriptor.get("route_id") != "mask072_lambda0.500"
        or int(descriptor.get("source_mask", -1)) != 72
        or descriptor.get("source_subset") != ["etarc", "branch_negative_gap"]
        or float(descriptor.get("homotopy_fraction", -1.0)) != 0.5
        or float(protocol["design"]["local_protected_retention_floor"]) != 0.95
        or int(protocol["design"]["anchor_family_size"]) != 239
        or protocol["design"].get("population_inference_claimed") is not False
    ):
        raise C1ValidationError("C1 route registry differs")
    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 6
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("training_authorized") is not False
        or config.get("gradient_measurement_mode") != "eval"
    ):
        raise C1ValidationError("C1 config differs")
    source_protocol = etarc_p0.read_json(
        resolve_repo_path(protocol["basis"]["d1x_protocol_path"])
    )
    if (
        protocol["selection"]["qualification_blocks"]
        != source_protocol["selection"]["qualification_blocks"]
        or protocol["selection"]["sealed_diagnostic_block"]
        != source_protocol["selection"]["sealed_diagnostic_block"]
    ):
        raise C1ValidationError("C1 qualification split differs")
    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise C1ValidationError("C1 risk source differs")
    return protocol, config, targets, risk_rows


def require_clean_start(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise C1ValidationError("C1 requires a clean worktree")
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
            raise C1ValidationError("prospective C1 outputs already exist")
    return etarc_p0.git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise C1ValidationError("CUDA is unavailable")
    tokenizer = rp.tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    qualification_ids = [
        sample_id
        for block in protocol["selection"]["qualification_blocks"]
        for sample_id in block["safety_sample_ids"]
    ]
    sealed_ids = protocol["selection"]["sealed_diagnostic_block"]["safety_sample_ids"]
    if (
        len(qualification_ids) != 16
        or set(qualification_ids).intersection(sealed_ids)
        or not all(value in pair_map for value in qualification_ids)
    ):
        raise C1ValidationError("C1 qualification registry differs")
    result = {
        "status": "PASS_BARP_MCD_D1X_C1_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "gpu": torch.cuda.get_device_name(0),
        "route_id": protocol["design"]["route_descriptor"]["route_id"],
        "qualification_blocks": 4,
        "anchor_candidates_per_proposal": 239,
        "route_descriptors_evaluated": 1,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "sealed_diagnostic_opened": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _artifact_inventory(root: Path, reproduction: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": relative(path),
            "sha256": etarc_p0.sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != reproduction and "protocol" not in path.parts
    ]


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import set_seed

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_start(path, protocol)
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [(name, value) for name, value in model.named_parameters() if value.requires_grad],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise C1ValidationError("unexpected C1 trainable parameter inventory")
    pair_map, _ = rp.structural_features(tokenizer, targets, config, protocol)
    blocks = protocol["selection"]["qualification_blocks"]
    risk_indices = {int(value) for block in blocks for value in block["risk_row_indices"]}
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in risk_indices
    }
    optimizer = frozen.make_optimizer(model, config)
    cpu_blocks, coverage = d1x._collect_partition(
        "C1 qualification",
        blocks,
        model,
        parameters,
        risk_features,
        pair_map,
        config,
    )
    descriptor = dict(protocol["design"]["route_descriptor"])
    rows = d1x._evaluate_partition(
        "C1 qualification",
        cpu_blocks,
        [descriptor],
        parameters,
        optimizer,
        config,
        protocol,
    )[descriptor["route_id"]]
    summary = summarize_conditional(descriptor, rows)
    checks = qualification_checks(summary, protocol["gates"])
    route = select_route(checks)
    qualification_path = resolve_repo_path(protocol["outputs"]["qualification"])
    write_json(
        qualification_path,
        {
            "name": "step3_barp_mcd_d1x_c1_qualification_v1",
            "post_selection_disclosed": True,
            "coverage": coverage,
            "summary": summary,
            "checks": checks,
        },
    )
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    decision = {
        "name": "step3_barp_mcd_d1x_c1_decision_v1",
        "route": route,
        "route_id": descriptor["route_id"],
        "post_selection_hypothesis": True,
        "checks": checks,
        "qualification": summary,
        "optimizer_objects": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
        "checkpoint_writes": 0,
        "candidate_outputs_generated": False,
        "sealed_diagnostic_opened": False,
        "core100_opened": False,
        "heldout_opened": False,
        "matched_training_protocol_authorized": route
        == "PASS_BARP_MCD_D1X_C1_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "wall_seconds": time.time() - started,
    }
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    write_json(decision_path, decision)
    target = summary["target_clusters"]
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(
        "# BARP-MCD D1X-C1 Conditional Qualification\n\n"
        f"Decision: `{route}`.\n\n"
        f"Frozen route: `{descriptor['route_id']}`; activity: "
        f"`{summary['proposal_activity']}/4`.\n\n"
        f"Target mean Delta: `{target['mean']:.8f}`; Studentized stability "
        f"lower score: `{target['studentized_lower_score']:.8f}`; positive/negative/tie "
        f"proposal clusters: `{target['positive']}/{target['negative']}/{target['ties']}`.\n\n"
        "This is a post-selection finite-sample qualification, not a confidence "
        "interval or population claim. Optimizer steps: `0`; diagnostic, Core100, "
        "and heldout opened: `false`.\n",
        encoding="utf-8",
    )
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    reproduction = {
        "name": "step3_barp_mcd_d1x_c1_reproduction_manifest_v1",
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1x_c1.py preflight",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1x_c1.py run",
            "python experiments/step3_response_quality_persistence/scripts/step3_barp_mcd_d1x_c1.py verify-run",
        ],
        "route": route,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }
    write_json(reproduction_path, reproduction)
    reproduction["artifacts"] = _artifact_inventory(
        resolve_repo_path(protocol["output_root"]), reproduction_path
    )
    write_json(reproduction_path, reproduction)
    print(json.dumps(decision, sort_keys=True))
    del optimizer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    qualification_path = resolve_repo_path(protocol["outputs"]["qualification"])
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    reproduction_path = resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    qualification = etarc_p0.read_json(qualification_path)
    stored = etarc_p0.read_json(decision_path)
    summary = summarize_conditional(
        dict(protocol["design"]["route_descriptor"]),
        qualification["summary"]["proposal_rows"],
    )
    if summary != qualification["summary"]:
        raise C1ValidationError("C1 summary does not reproduce")
    checks = qualification_checks(summary, protocol["gates"])
    route = select_route(checks)
    if checks != qualification["checks"] or route != stored["route"]:
        raise C1ValidationError("C1 decision does not reproduce")
    if stored.get("optimizer_steps") != 0 or stored.get("post_selection_hypothesis") is not True:
        raise C1ValidationError("C1 boundary does not reproduce")
    reproduction = etarc_p0.read_json(reproduction_path)
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or etarc_p0.sha256(artifact_path) != artifact["sha256"]:
            raise C1ValidationError(f"C1 artifact hash mismatch: {artifact['path']}")
    result = {
        "status": "PASS_BARP_MCD_D1X_C1_VERIFY",
        "route": route,
        "route_id": stored["route_id"],
        "verified_artifacts": len(reproduction["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "sealed_diagnostic_opened": False,
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
