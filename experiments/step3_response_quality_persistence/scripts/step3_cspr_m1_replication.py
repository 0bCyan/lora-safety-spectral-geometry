"""Prepare and run the CSPR M1 disjoint-batch zero-update replication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import step3_cspr_m0_mechanism as m0


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_m1/"
    "protocol/cspr_m1_protocol.json"
)


class M1ValidationError(ValueError):
    """Raised when an M1 artifact violates the frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise M1ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: str) -> str:
    payload = "|".join((str(seed), *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


def write_frozen_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_frozen(path: Path, text: str) -> None:
    write_frozen_bytes(path, text.encode("utf-8"))


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_frozen(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_frozen_bytes(path, jsonl_bytes(rows))


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise M1ValidationError(f"frozen input hash mismatch: {name}")
    if protocol.get("optimizer_steps") != 0 or protocol.get("training_authorized") is not False:
        raise M1ValidationError("M1 violates the zero-training boundary")
    if protocol.get("core100_opened") is not False or protocol.get("heldout_opened") is not False:
        raise M1ValidationError("M1 opens a prohibited evaluation set")
    if protocol.get("human_labels_used") is not False:
        raise M1ValidationError("M1 must not claim human-qualified targets")
    m0_result = read_json(resolve_repo_path(protocol["frozen_inputs"]["m0_diagnostics"]["path"]))
    if m0_result.get("route") != "PASS_CSPR_M0_REAL_MODEL_MECHANISM_FEASIBLE":
        raise M1ValidationError("M1 does not descend from the frozen M0 pass route")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise M1ValidationError("M1 entry commit is not an ancestor of HEAD")
    return protocol


def assign_preference_batches(
    rows: list[dict[str, Any]],
    seed: int,
    batch_count: int,
    repeated_categories: int,
    singleton_categories: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    repeated_pool = [category for category, values in grouped.items() if len(values) >= 2]
    if len(repeated_pool) < repeated_categories:
        raise M1ValidationError("insufficient repeated preference categories")
    repeated = sorted(
        repeated_pool,
        key=lambda category: stable_rank(seed, "m1", "preference", "repeated", category),
    )[:repeated_categories]
    singleton_pool = [category for category in grouped if category not in repeated]
    if len(singleton_pool) < singleton_categories:
        raise M1ValidationError("insufficient singleton preference categories")
    singletons = sorted(
        singleton_pool,
        key=lambda category: stable_rank(seed, "m1", "preference", "singleton", category),
    )[:singleton_categories]

    batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    for position, category in enumerate(repeated):
        candidates = sorted(
            grouped[category],
            key=lambda row: stable_rank(
                seed, "m1", "preference", "row", category, str(row["pair_id"])
            ),
        )[:2]
        destination_batches = [index for index in range(batch_count) if index != position % batch_count]
        if len(destination_batches) != len(candidates):
            raise M1ValidationError("repeated-category design requires exactly three batches")
        for batch_index, row in zip(destination_batches, candidates, strict=True):
            batches[batch_index].append(dict(row))
    for position, category in enumerate(singletons):
        row = min(
            grouped[category],
            key=lambda value: stable_rank(
                seed, "m1", "preference", "row", category, str(value["pair_id"])
            ),
        )
        batches[position % batch_count].append(dict(row))

    expected_rows = (2 * repeated_categories + singleton_categories) // batch_count
    for batch in batches:
        if len(batch) != expected_rows or len({row["category"] for row in batch}) != len(batch):
            raise M1ValidationError("preference batches are not balanced and category-diverse")
        batch.sort(
            key=lambda row: stable_rank(seed, "m1", "preference", "order", str(row["pair_id"]))
        )
    pair_ids = [str(row["pair_id"]) for batch in batches for row in batch]
    if len(pair_ids) != len(set(pair_ids)):
        raise M1ValidationError("preference batches reuse a pair ID")
    return batches


def assign_safety_batches(
    rows: list[dict[str, Any]],
    seed: int,
    batch_count: int,
    shared_categories: int,
    rare_categories: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    common_pool = [category for category, values in grouped.items() if len(values) >= batch_count]
    if len(common_pool) < shared_categories:
        raise M1ValidationError("insufficient shared safety categories")
    common = sorted(
        common_pool,
        key=lambda category: stable_rank(seed, "m1", "safety", "shared", category),
    )[:shared_categories]
    rare_pool = [category for category in grouped if category not in common and grouped[category]]
    if len(rare_pool) < rare_categories:
        raise M1ValidationError("insufficient rare safety categories")
    rare = sorted(
        rare_pool,
        key=lambda category: stable_rank(seed, "m1", "safety", "rare", category),
    )[:rare_categories]

    batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    for category in common:
        candidates = sorted(
            grouped[category],
            key=lambda row: stable_rank(
                seed, "m1", "safety", "row", category, str(row["target_id"])
            ),
        )[:batch_count]
        for batch_index, row in enumerate(candidates):
            batches[batch_index].append(dict(row))
    for position, category in enumerate(rare):
        row = min(
            grouped[category],
            key=lambda value: stable_rank(
                seed, "m1", "safety", "row", category, str(value["target_id"])
            ),
        )
        batches[position % batch_count].append(dict(row))

    expected_rows = shared_categories + rare_categories // batch_count
    for batch in batches:
        if len(batch) != expected_rows or len({row["category"] for row in batch}) != len(batch):
            raise M1ValidationError("safety batches are not balanced and category-diverse")
        batch.sort(
            key=lambda row: stable_rank(seed, "m1", "safety", "order", str(row["target_id"]))
        )
    target_ids = [str(row["target_id"]) for batch in batches for row in batch]
    if len(target_ids) != len(set(target_ids)):
        raise M1ValidationError("safety batches reuse a target ID")
    return batches


def prepare(path: Path) -> dict[str, Any]:
    protocol = context(path)
    from transformers import AutoTokenizer

    cfg = protocol["design"]
    seed = int(cfg["selection_seed"])
    max_length = int(cfg["max_length"])
    batch_count = int(cfg["batch_count"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_repo_path(protocol["model"]["checkpoint_dir"])),
        local_files_only=True,
        trust_remote_code=True,
    )
    source = protocol["source_paths"]
    m0_pairs = read_jsonl(resolve_repo_path(source["m0_preference_data"]))
    m0_safety = read_jsonl(resolve_repo_path(source["m0_safety_data"]))
    excluded_pair_ids = {str(row["pair_id"]) for row in m0_pairs}
    excluded_target_ids = {str(row["target_id"]) for row in m0_safety}

    ledger = read_jsonl(resolve_repo_path(source["candidate_ledger"]))
    author_packets = read_jsonl(resolve_repo_path(source["author_a_packet"])) + read_jsonl(
        resolve_repo_path(source["author_b_packet"])
    )
    prompt_by_author_id = {
        str(row["opaque_authoring_id"]): str(row["prompt"]) for row in author_packets
    }
    safety_packet = {
        str(row["opaque_safety_id"]): row
        for row in read_jsonl(resolve_repo_path(source["p2_safety_packet"]))
    }
    rejected_by_pair: dict[str, dict[str, Any]] = {}
    for key in read_jsonl(resolve_repo_path(source["p2_safety_key"])):
        if key["is_hidden_duplicate"] or key["response_role"] != "rejected":
            continue
        rejected_by_pair[str(key["pair_id"])] = safety_packet[str(key["opaque_safety_id"])]

    candidate_rows = [
        row
        for row in ledger
        if row["prospective_split"] == "mechanism_dev"
        and row["mechanically_valid"]
        and str(row["pair_id"]) not in excluded_pair_ids
    ]
    preference_candidates = []
    for row in m0.choose_candidate_per_pair(candidate_rows, seed):
        author_id = str(row["opaque_authoring_id"])
        pair_id = str(row["pair_id"])
        rejected = rejected_by_pair.get(pair_id)
        if author_id not in prompt_by_author_id or rejected is None:
            raise M1ValidationError("preference source join is incomplete")
        prompt = prompt_by_author_id[author_id]
        if prompt != str(rejected["prompt"]):
            raise M1ValidationError("same-prompt preference join mismatch")
        chosen_response = str(row["response"])
        rejected_response = str(rejected["response"])
        chosen_shape = m0.token_shape(tokenizer, prompt, chosen_response)
        rejected_shape = m0.token_shape(tokenizer, prompt, rejected_response)
        if (
            chosen_shape["total_tokens"] > max_length
            or rejected_shape["total_tokens"] > max_length
            or chosen_shape["response_tokens"] <= 0
            or rejected_shape["response_tokens"] <= 0
        ):
            continue
        preference_candidates.append(
            {
                "pair_id": pair_id,
                "candidate_id": author_id,
                "category": str(row["assigned_category"]),
                "prompt": prompt,
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
                "chosen_shape": chosen_shape,
                "rejected_shape": rejected_shape,
                "behaviorally_qualified": False,
            }
        )
    preference_batches = assign_preference_batches(
        preference_candidates,
        seed,
        batch_count,
        int(cfg["repeated_preference_categories"]),
        int(cfg["singleton_preference_categories"]),
    )
    preference_rows = []
    for batch_index, batch in enumerate(preference_batches, start=1):
        wrong_prompts = m0.find_wrong_prompt_derangement(batch, tokenizer, max_length)
        for row, wrong_prompt in zip(batch, wrong_prompts, strict=True):
            row["batch_id"] = f"m1_batch_{batch_index}"
            row["wrong_prompt"] = wrong_prompt
            row["wrong_chosen_shape"] = m0.token_shape(
                tokenizer, wrong_prompt, str(row["chosen_response"])
            )
            row["wrong_rejected_shape"] = m0.token_shape(
                tokenizer, wrong_prompt, str(row["rejected_response"])
            )
            preference_rows.append(row)

    safety_candidates = []
    for row in read_jsonl(resolve_repo_path(source["safety_targets"])):
        if str(row["target_id"]) in excluded_target_ids:
            continue
        shape = m0.token_shape(tokenizer, str(row["prompt"]), str(row["response"]))
        if shape["total_tokens"] <= max_length and shape["response_tokens"] > 0:
            safety_candidates.append(
                {
                    "target_id": str(row["target_id"]),
                    "category": str(row["category"]),
                    "prompt": str(row["prompt"]),
                    "response": str(row["response"]),
                    "shape": shape,
                }
            )
    safety_batches = assign_safety_batches(
        safety_candidates,
        seed,
        batch_count,
        int(cfg["shared_safety_categories"]),
        int(cfg["rare_safety_categories"]),
    )
    safety_rows = []
    for batch_index, batch in enumerate(safety_batches, start=1):
        for row in batch:
            row["batch_id"] = f"m1_batch_{batch_index}"
            safety_rows.append(row)

    outputs = protocol["outputs"]
    preference_path = resolve_repo_path(outputs["preference_data"])
    safety_path = resolve_repo_path(outputs["safety_data"])
    write_jsonl_frozen(preference_path, preference_rows)
    write_jsonl_frozen(safety_path, safety_rows)
    artifacts = [
        {
            "path": relative(preference_path),
            "sha256": sha256(preference_path),
            "bytes": preference_path.stat().st_size,
            "rows": len(preference_rows),
        },
        {
            "path": relative(safety_path),
            "sha256": sha256(safety_path),
            "bytes": safety_path.stat().st_size,
            "rows": len(safety_rows),
        },
    ]
    checkpoint = resolve_repo_path(protocol["model"]["checkpoint_dir"])
    checkpoint_files = [
        {"path": relative(file), "sha256": sha256(file), "bytes": file.stat().st_size}
        for file in sorted(checkpoint.iterdir())
        if file.is_file()
    ]
    batch_manifest = []
    for batch_index in range(1, batch_count + 1):
        batch_id = f"m1_batch_{batch_index}"
        pref = [row for row in preference_rows if row["batch_id"] == batch_id]
        safety = [row for row in safety_rows if row["batch_id"] == batch_id]
        batch_manifest.append(
            {
                "batch_id": batch_id,
                "preference_pair_ids": [row["pair_id"] for row in pref],
                "preference_categories": sorted({row["category"] for row in pref}),
                "safety_target_ids": [row["target_id"] for row in safety],
                "safety_categories": sorted({row["category"] for row in safety}),
            }
        )
    manifest = {
        "name": "step3_cspr_m1_disjoint_batch_data_freeze_v1",
        "route": "CSPR_M1_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY",
        "entry_commit": protocol["entry_commit"],
        "artifacts": artifacts,
        "checkpoint_files": checkpoint_files,
        "batches": batch_manifest,
        "batch_count": batch_count,
        "preference_rows": len(preference_rows),
        "safety_rows": len(safety_rows),
        "m0_preference_disjoint": not excluded_pair_ids.intersection(
            {str(row["pair_id"]) for row in preference_rows}
        ),
        "m0_safety_disjoint": not excluded_target_ids.intersection(
            {str(row["target_id"]) for row in safety_rows}
        ),
        "all_sequences_fit_without_truncation": True,
        "wrong_prompt_is_deranged": all(
            row["wrong_prompt"] != row["prompt"] for row in preference_rows
        ),
        "human_labels_used": False,
        "behavioral_qualification_claimed": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["data_manifest"]), manifest)
    report = """# CSPR M1 Data and Execution Freeze

Route: `CSPR_M1_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY`.

Three mutually disjoint batches were frozen. Each contains six category-diverse M0-disjoint mechanism-dev preference pairs and six category-diverse M0-disjoint constant20 safety rows. All true, swapped, wrong-prompt, and safety sequences fit the 256-token encoding without truncation. No human label is used and no candidate is claimed behaviorally qualified.

The package authorizes one three-batch real-model zero-update replication only after commit and push. It does not authorize an optimizer, checkpoint write, generation, target qualification, core100, heldout, or training. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(outputs["freeze_report"]), report)
    print(
        json.dumps(
            {
                "route": manifest["route"],
                "batches": batch_count,
                "preference_rows": len(preference_rows),
                "safety_rows": len(safety_rows),
            },
            sort_keys=True,
        )
    )
    return manifest


def verify_prepared(protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(resolve_repo_path(protocol["outputs"]["data_manifest"]))
    if manifest.get("route") != "CSPR_M1_DATA_AND_EXECUTION_FROZEN_ZERO_UPDATE_ONLY":
        raise M1ValidationError("M1 data-freeze route mismatch")
    for artifact in manifest["artifacts"]:
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise M1ValidationError(f"prepared artifact hash mismatch: {artifact['path']}")
        if len(read_jsonl(path)) != artifact["rows"]:
            raise M1ValidationError(f"prepared artifact row mismatch: {artifact['path']}")
    for artifact in manifest["checkpoint_files"]:
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise M1ValidationError(f"checkpoint hash mismatch: {artifact['path']}")
    cfg = protocol["design"]
    batch_count = int(cfg["batch_count"])
    expected_rows = batch_count * int(cfg["rows_per_objective_per_batch"])
    if (
        manifest.get("batch_count") != batch_count
        or manifest.get("preference_rows") != expected_rows
        or manifest.get("safety_rows") != expected_rows
    ):
        raise M1ValidationError("M1 prepared row counts differ from the frozen design")
    preference_rows = read_jsonl(resolve_repo_path(protocol["outputs"]["preference_data"]))
    safety_rows = read_jsonl(resolve_repo_path(protocol["outputs"]["safety_data"]))
    preference_ids = [str(row["pair_id"]) for row in preference_rows]
    safety_ids = [str(row["target_id"]) for row in safety_rows]
    if len(preference_ids) != len(set(preference_ids)) or len(safety_ids) != len(set(safety_ids)):
        raise M1ValidationError("M1 batches are not mutually disjoint")
    m0_preference_ids = {
        str(row["pair_id"])
        for row in read_jsonl(resolve_repo_path(protocol["source_paths"]["m0_preference_data"]))
    }
    m0_safety_ids = {
        str(row["target_id"])
        for row in read_jsonl(resolve_repo_path(protocol["source_paths"]["m0_safety_data"]))
    }
    if m0_preference_ids.intersection(preference_ids) or m0_safety_ids.intersection(safety_ids):
        raise M1ValidationError("M1 data overlaps M0")
    rows_per_batch = int(cfg["rows_per_objective_per_batch"])
    for batch_index in range(1, batch_count + 1):
        batch_id = f"m1_batch_{batch_index}"
        pref = [row for row in preference_rows if row["batch_id"] == batch_id]
        safety = [row for row in safety_rows if row["batch_id"] == batch_id]
        if len(pref) != rows_per_batch or len({row["category"] for row in pref}) != rows_per_batch:
            raise M1ValidationError(f"invalid preference composition: {batch_id}")
        if len(safety) != rows_per_batch or len({row["category"] for row in safety}) != rows_per_batch:
            raise M1ValidationError(f"invalid safety composition: {batch_id}")
        if any(row["wrong_prompt"] == row["prompt"] for row in pref):
            raise M1ValidationError(f"invalid wrong-prompt derangement: {batch_id}")
    if not manifest.get("m0_preference_disjoint") or not manifest.get("m0_safety_disjoint"):
        raise M1ValidationError("M1 manifest disjointness claim is false")
    return manifest


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_tracked_freeze(protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise M1ValidationError("M1 run requires a clean worktree")
    paths = [
        protocol["outputs"]["preference_data"],
        protocol["outputs"]["safety_data"],
        protocol["outputs"]["data_manifest"],
        relative(resolve_repo_path(DEFAULT_PROTOCOL)),
    ]
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    manifest = verify_prepared(protocol)
    result = {
        "status": "PASS_CSPR_M1_PREFLIGHT",
        "entry_commit": protocol["entry_commit"],
        "batch_count": manifest["batch_count"],
        "preference_rows": manifest["preference_rows"],
        "safety_rows": manifest["safety_rows"],
        "m0_disjoint": True,
        "human_labels_used": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def normalize_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameter_tensors": int(coverage["parameter_tensors"]),
        "parameter_elements": int(coverage["parameter_elements"]),
        "missing_gradient_tensors": list(coverage["missing_gradient_tensors"]),
        "finite": bool(coverage["finite"]),
        "nonzero_gradient_tensors": int(coverage["nonzero_elements"]),
    }


def coverage_valid(coverage: dict[str, Any]) -> bool:
    return (
        coverage["finite"]
        and not coverage["missing_gradient_tensors"]
        and coverage["nonzero_gradient_tensors"] == coverage["parameter_tensors"]
    )


def same_step_safety_rescue(
    active: bool,
    line_search: dict[str, Any],
    unconstrained: dict[str, Any] | None,
    initial_safety: float,
    tolerance: float,
) -> bool:
    return bool(
        active
        and line_search["accepted"]
        and unconstrained is not None
        and unconstrained["safety_loss"] > initial_safety + tolerance
    )


def select_route(gates: dict[str, bool]) -> str:
    if (
        not gates["structural_and_gradient_valid"]
        or not gates["directional_and_kkt"]
        or not gates["state_restored"]
        or not gates["zero_optimizer"]
    ):
        return "STOP_CSPR_M1_INVALID"
    if not gates["label_swap_control"]:
        return "STOP_CSPR_M1_DPO_CONTROL_FAILED"
    if not gates["pair_activation"] or not gates["aggregate_activation"]:
        return "STOP_CSPR_M1_PROJECTION_NOT_REPLICATED"
    if not gates["retained_preference"]:
        return "STOP_CSPR_M1_PROJECTED_DIRECTION_DEGENERATE"
    if not gates["finite_joint_step"]:
        return "STOP_CSPR_M1_FINITE_STEP_NOT_REPLICATED"
    if not gates["replicated_safety_rescue"]:
        return "STOP_CSPR_M1_NO_REPLICATED_SAFETY_RESCUE"
    return "PASS_CSPR_M1_BATCH_ROBUST_MECHANISM"


def run_batch(
    model: Any,
    tokenizer: Any,
    parameters: list[tuple[str, Any]],
    snapshots: list[Any],
    pair_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    global_initial_hash: str,
) -> dict[str, Any]:
    import torch

    cfg = protocol["design"]
    gate_cfg = protocol["gates"]
    max_length = int(cfg["max_length"])
    beta = float(cfg["beta"])
    batch_id = str(pair_rows[0]["batch_id"])
    if any(row["batch_id"] != batch_id for row in pair_rows + safety_rows):
        raise M1ValidationError("mixed batch IDs passed to run_batch")
    initial_hash = m0.trainable_state_hash(parameters)
    if initial_hash != global_initial_hash:
        raise M1ValidationError(f"initial model state drift before {batch_id}")

    preference_features = [
        (
            m0.response_feature(tokenizer, row["prompt"], row["chosen_response"], max_length),
            m0.response_feature(tokenizer, row["prompt"], row["rejected_response"], max_length),
        )
        for row in pair_rows
    ]
    swapped_features = [(rejected, chosen) for chosen, rejected in preference_features]
    wrong_features = [
        (
            m0.response_feature(
                tokenizer, row["wrong_prompt"], row["chosen_response"], max_length
            ),
            m0.response_feature(
                tokenizer, row["wrong_prompt"], row["rejected_response"], max_length
            ),
        )
        for row in pair_rows
    ]
    safety_features = [
        m0.response_feature(tokenizer, row["prompt"], row["response"], max_length)
        for row in safety_rows
    ]

    safety_grad, raw_safety_coverage, initial_safety = m0.safety_gradient(
        model, parameters, safety_features
    )
    safety_coverage = normalize_coverage(raw_safety_coverage)
    aggregate_preference = torch.zeros_like(safety_grad)
    references = []
    pair_metrics = []
    pair_coverages = []
    for row, features in zip(pair_rows, preference_features, strict=True):
        pair_grad, raw_coverage, baseline = m0.preference_gradient(
            model, parameters, [features], beta
        )
        coverage = normalize_coverage(raw_coverage)
        pair_coverages.append(coverage)
        aggregate_preference += pair_grad / len(preference_features)
        references.extend(baseline)
        pair_metrics.append(
            {
                "pair_id": row["pair_id"],
                **m0.pair_projection_metrics(pair_grad, safety_grad),
                "gradient_hash": m0.gradient_hash(pair_grad),
                "coverage": coverage,
            }
        )
        del pair_grad
    preference_coverage = {
        "parameter_tensors": len(parameters),
        "parameter_elements": aggregate_preference.numel(),
        "missing_gradient_tensors": sorted(
            {
                name
                for coverage in pair_coverages
                for name in coverage["missing_gradient_tensors"]
            }
        ),
        "finite": all(coverage["finite"] for coverage in pair_coverages),
        "minimum_nonzero_gradient_tensors_across_pairs": min(
            coverage["nonzero_gradient_tensors"] for coverage in pair_coverages
        ),
        "aggregate_nonzero_parameter_elements": int(torch.count_nonzero(aggregate_preference)),
    }

    swapped_grad, raw_swapped_coverage, _ = m0.preference_gradient(
        model, parameters, swapped_features, beta
    )
    wrong_grad, raw_wrong_coverage, _ = m0.preference_gradient(
        model, parameters, wrong_features, beta
    )
    swapped_coverage = normalize_coverage(raw_swapped_coverage)
    wrong_coverage = normalize_coverage(raw_wrong_coverage)
    preference_norm = float(torch.linalg.vector_norm(aggregate_preference))
    safety_norm = float(torch.linalg.vector_norm(safety_grad))
    swap_norm = float(torch.linalg.vector_norm(swapped_grad))
    wrong_norm = float(torch.linalg.vector_norm(wrong_grad))
    label_swap_cosine = m0.cosine(aggregate_preference, swapped_grad)
    label_swap_norm_ratio = swap_norm / preference_norm if preference_norm else math.nan
    wrong_cosine = m0.cosine(aggregate_preference, wrong_grad)

    certificate = m0.cspr.project_preconditioned_preference_direction(
        aggregate_preference, safety_grad, torch.ones_like(aggregate_preference)
    )
    retained = (
        -certificate.preference_directional_derivative / (preference_norm * preference_norm)
        if preference_norm
        else math.nan
    )
    initial_preference = m0.evaluate_preference_loss(
        model, preference_features, references, beta
    )
    search = m0.line_search(
        model,
        parameters,
        snapshots,
        certificate.direction,
        preference_features,
        safety_features,
        references,
        beta,
        initial_preference,
        initial_safety,
        certificate.preference_directional_derivative,
        certificate.safety_directional_derivative,
        protocol["line_search"],
    )
    unconstrained_at_accepted = None
    if search["accepted"]:
        alpha = float(search["accepted_step_size"])
        m0.apply_direction(parameters, snapshots, certificate.unconstrained_direction, alpha)
        try:
            unconstrained_at_accepted = {
                "step_size": alpha,
                "preference_loss": m0.evaluate_preference_loss(
                    model, preference_features, references, beta
                ),
                "safety_loss": m0.evaluate_safety_loss(model, safety_features),
            }
        finally:
            m0.restore_parameters(parameters, snapshots)
    final_hash = m0.trainable_state_hash(parameters)

    direction_norm = float(torch.linalg.vector_norm(certificate.direction))
    safety_scale = max(safety_norm * direction_norm, 1e-30)
    preference_scale = max(preference_norm * direction_norm, 1e-30)
    normalized = {
        "positive_safety_derivative": max(certificate.safety_directional_derivative, 0.0)
        / safety_scale,
        "positive_preference_derivative": max(certificate.preference_directional_derivative, 0.0)
        / preference_scale,
        "kkt_stationarity": certificate.kkt_stationarity_norm / max(direction_norm, 1e-30),
        "kkt_complementarity": certificate.kkt_complementarity_abs
        / max(abs(certificate.multiplier) * safety_scale, 1e-30),
    }
    all_finite_values = [
        preference_norm,
        safety_norm,
        swap_norm,
        wrong_norm,
        label_swap_cosine,
        label_swap_norm_ratio,
        wrong_cosine,
        retained,
        initial_preference,
        initial_safety,
        direction_norm,
        *normalized.values(),
    ]
    structural = (
        preference_norm > 0
        and safety_norm > 0
        and preference_coverage["finite"]
        and not preference_coverage["missing_gradient_tensors"]
        and preference_coverage["minimum_nonzero_gradient_tensors_across_pairs"]
        == len(parameters)
        and preference_coverage["aggregate_nonzero_parameter_elements"] > 0
        and coverage_valid(safety_coverage)
        and coverage_valid(swapped_coverage)
        and coverage_valid(wrong_coverage)
        and all(math.isfinite(value) for value in all_finite_values)
    )
    batch_gates = {
        "structural_and_gradient_valid": structural,
        "label_swap_control": label_swap_cosine <= float(gate_cfg["label_swap_cosine_max"])
        and float(gate_cfg["label_swap_norm_ratio_min"])
        <= label_swap_norm_ratio
        <= float(gate_cfg["label_swap_norm_ratio_max"]),
        "aggregate_activation": bool(certificate.active),
        "retained_preference": (not certificate.active)
        or retained >= float(gate_cfg["retained_preference_min"]),
        "directional_and_kkt": max(normalized.values())
        <= float(gate_cfg["normalized_residual_max"]),
        "finite_joint_step": bool(search["accepted"]),
        "same_step_safety_rescue": same_step_safety_rescue(
            bool(certificate.active),
            search,
            unconstrained_at_accepted,
            initial_safety,
            float(protocol["line_search"]["comparison_tolerance"]),
        ),
        "state_restored": final_hash == global_initial_hash,
    }
    result = {
        "batch_id": batch_id,
        "preference_pair_ids": [row["pair_id"] for row in pair_rows],
        "safety_target_ids": [row["target_id"] for row in safety_rows],
        "preference": {
            "initial_loss": initial_preference,
            "gradient_norm": preference_norm,
            "gradient_hash": m0.gradient_hash(aggregate_preference),
            "coverage": preference_coverage,
        },
        "safety": {
            "initial_loss": initial_safety,
            "gradient_norm": safety_norm,
            "gradient_hash": m0.gradient_hash(safety_grad),
            "coverage": safety_coverage,
        },
        "projection": {
            **certificate.scalars(),
            "preference_safety_cosine": m0.cosine(aggregate_preference, safety_grad),
            "direction_norm": direction_norm,
            "retained_preference_fraction": retained,
            "normalized_residuals": normalized,
        },
        "pairwise_projection": pair_metrics,
        "controls": {
            "label_swap_cosine": label_swap_cosine,
            "label_swap_norm_ratio": label_swap_norm_ratio,
            "label_swap_gradient_hash": m0.gradient_hash(swapped_grad),
            "label_swap_coverage": swapped_coverage,
            "wrong_prompt_cosine": wrong_cosine,
            "wrong_prompt_norm_ratio": wrong_norm / preference_norm,
            "wrong_prompt_gradient_hash": m0.gradient_hash(wrong_grad),
            "wrong_prompt_coverage": wrong_coverage,
        },
        "line_search": search,
        "unconstrained_at_projected_accepted_step": unconstrained_at_accepted,
        "gates": batch_gates,
        "initial_trainable_state_hash": initial_hash,
        "final_trainable_state_hash": final_hash,
    }
    del aggregate_preference, safety_grad, swapped_grad, wrong_grad, certificate
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(path: Path) -> dict[str, Any]:
    import torch

    protocol = context(path)
    manifest = verify_prepared(protocol)
    runtime_head = require_clean_tracked_freeze(protocol)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = m0.load_model(protocol)
    parameters = m0.trainable_parameters(model)
    if not parameters:
        raise M1ValidationError("model has no trainable LoRA parameters")
    global_initial_hash = m0.trainable_state_hash(parameters)
    snapshots = m0.parameter_snapshots(parameters)
    preference_rows = read_jsonl(resolve_repo_path(protocol["outputs"]["preference_data"]))
    safety_rows = read_jsonl(resolve_repo_path(protocol["outputs"]["safety_data"]))
    batch_results = []
    for batch_index in range(1, int(protocol["design"]["batch_count"]) + 1):
        batch_id = f"m1_batch_{batch_index}"
        batch_results.append(
            run_batch(
                model,
                tokenizer,
                parameters,
                snapshots,
                [row for row in preference_rows if row["batch_id"] == batch_id],
                [row for row in safety_rows if row["batch_id"] == batch_id],
                protocol,
                global_initial_hash,
            )
        )
    global_final_hash = m0.trainable_state_hash(parameters)

    gate_cfg = protocol["gates"]
    active_pairs = sum(
        int(row["active"])
        for batch in batch_results
        for row in batch["pairwise_projection"]
    )
    active_batches = sum(int(batch["gates"]["aggregate_activation"]) for batch in batch_results)
    active_joint_steps = sum(
        int(batch["gates"]["aggregate_activation"] and batch["gates"]["finite_joint_step"])
        for batch in batch_results
    )
    safety_rescues = sum(int(batch["gates"]["same_step_safety_rescue"]) for batch in batch_results)
    gates = {
        "structural_and_gradient_valid": all(
            batch["gates"]["structural_and_gradient_valid"] for batch in batch_results
        ),
        "label_swap_control": all(
            batch["gates"]["label_swap_control"] for batch in batch_results
        ),
        "pair_activation": active_pairs >= int(gate_cfg["minimum_active_pairs_total"]),
        "aggregate_activation": active_batches
        >= int(gate_cfg["minimum_active_aggregate_batches"]),
        "retained_preference": all(
            batch["gates"]["retained_preference"] for batch in batch_results
        ),
        "directional_and_kkt": all(
            batch["gates"]["directional_and_kkt"] for batch in batch_results
        ),
        "finite_joint_step": active_joint_steps
        >= int(gate_cfg["minimum_active_batches_with_joint_step"]),
        "replicated_safety_rescue": safety_rescues
        >= int(gate_cfg["minimum_same_step_safety_rescues"]),
        "state_restored": global_final_hash == global_initial_hash
        and all(batch["gates"]["state_restored"] for batch in batch_results),
        "zero_optimizer": True,
    }
    route = select_route(gates)
    report = {
        "name": "step3_cspr_m1_disjoint_batch_replication_v1",
        "route": route,
        "runtime_head": runtime_head,
        "data_manifest_sha256": sha256(resolve_repo_path(protocol["outputs"]["data_manifest"])),
        "model": protocol["model"],
        "design": protocol["design"],
        "batch_results": batch_results,
        "summary": {
            "active_pairwise_projections": active_pairs,
            "total_pairwise_projections": sum(
                len(batch["pairwise_projection"]) for batch in batch_results
            ),
            "active_aggregate_batches": active_batches,
            "active_batches_with_joint_step": active_joint_steps,
            "same_step_safety_rescues": safety_rescues,
        },
        "gates": gates,
        "initial_trainable_state_hash": global_initial_hash,
        "final_trainable_state_hash": global_final_hash,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "checkpoint_written": False,
        "candidate_outputs_generated": False,
        "human_labels_used": False,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    outputs = protocol["outputs"]
    write_json_frozen(resolve_repo_path(outputs["diagnostics"]), report)
    environment = {
        "name": "step3_cspr_m1_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_clock_seconds": time.time() - started,
        "runtime_head": runtime_head,
        "pid": os.getpid(),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["environment"]), environment)
    route_record = {
        "name": "step3_cspr_m1_route_v1",
        "route": route,
        "summary": report["summary"],
        "gates": gates,
        "optimizer_steps": 0,
        "training_authorized": False,
        "next_authorized": (
            "resume_external_human_P2U_measurement_only"
            if route == "PASS_CSPR_M1_BATCH_ROBUST_MECHANISM"
            else "stop_CSPR_mechanism_promotion"
        ),
    }
    write_json_frozen(resolve_repo_path(outputs["route"]), route_record)
    markdown = f"""# CSPR M1 Disjoint-Batch Replication Report

Route: `{route}`.

Active pairwise projections: `{active_pairs}/18`. Active aggregate batches: `{active_batches}/3`. Active batches with a projected dual-Armijo step: `{active_joint_steps}/3`. Registered same-step safety rescues: `{safety_rescues}`.

This is a real-model zero-update mechanism replication over three frozen disjoint batches. It is not a behavior, safety-generalization, target-quality, or training-benefit result. No optimizer or scheduler was created, no checkpoint or candidate output was written, and the exact initial trainable-state hash was restored after every batch. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(outputs["report"]), markdown)
    print(json.dumps({"route": route, **report["summary"]}, sort_keys=True))
    return report


def verify_run(path: Path) -> dict[str, Any]:
    protocol = context(path)
    verify_prepared(protocol)
    report = read_json(resolve_repo_path(protocol["outputs"]["diagnostics"]))
    route = read_json(resolve_repo_path(protocol["outputs"]["route"]))
    if report.get("route") != route.get("route"):
        raise M1ValidationError("M1 report and route disagree")
    if report.get("initial_trainable_state_hash") != report.get("final_trainable_state_hash"):
        raise M1ValidationError("M1 trainable state was not restored")
    if len(report.get("batch_results", [])) != int(protocol["design"]["batch_count"]):
        raise M1ValidationError("M1 result has the wrong batch count")
    if not all(batch["gates"]["state_restored"] for batch in report["batch_results"]):
        raise M1ValidationError("an M1 batch did not restore the trainable state")
    if report.get("optimizer_steps") != 0 or report.get("training_authorized") is not False:
        raise M1ValidationError("M1 result violates zero-training boundary")
    result = {
        "status": "PASS_CSPR_M1_RESULT_VERIFICATION",
        "route": report["route"],
        "batch_count": len(report["batch_results"]),
        "state_restored": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "prepare":
            prepare(path)
        elif args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        FileExistsError,
        KeyError,
        M1ValidationError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_CSPR_M1_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
