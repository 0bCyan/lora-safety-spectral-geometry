"""Generate and evaluate the prospective TARC proxy-dev effect gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import evaluate_step3_tarc_proxy_dev as evaluator
import generate_step3_tarc_proxy_dev as generation
import step3ct_training_common as training


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_tarc/"
    "protocol/tarc_proxy_dev_effect_protocol.json"
)


class TarcProxyRunError(ValueError):
    """Raised when TARC proxy generation violates the frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    return evaluator.resolve_repo_path(value)


def read_json(path: Path) -> Any:
    return evaluator.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return evaluator.read_jsonl(path)


def sha256(path: Path) -> str:
    return evaluator.sha256(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def prompt_overlap_counts(
    split: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> dict[str, int]:
    split_ids = {str(row["sample_id"]) for row in split}
    target_ids = {str(row["sample_id"]) for row in targets}
    split_prompts = {str(row["prompt"]) for row in split}
    target_prompts = {str(row["prompt"]) for row in targets}
    return {
        "sample_id_overlap": len(split_ids.intersection(target_ids)),
        "exact_prompt_overlap": len(split_prompts.intersection(target_prompts)),
        "split_prompt_clusters": len(split_ids),
        "training_prompt_clusters": len(target_ids),
    }


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_TARC_proxy_dev_generation":
        raise TarcProxyRunError("proxy protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise TarcProxyRunError(f"frozen input hash mismatch: {name}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise TarcProxyRunError("entry commit is not an ancestor of HEAD")
    development = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["development_decision"]["path"])
    )
    if (
        development.get("route")
        != "PASS_TARC_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE"
    ):
        raise TarcProxyRunError("development route did not authorize generation freeze")
    if (
        protocol.get("proxy_dev_generation_authorized") is not True
        or protocol.get("blind_review_protocol_authorized") is not False
        or protocol.get("main_training_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise TarcProxyRunError("proxy authorization or sealed boundary differs")
    if protocol.get("historical_proxy_reuse_disclosed") is not True:
        raise TarcProxyRunError("historical proxy reuse is not disclosed")

    split = read_jsonl(resolve_repo_path(protocol["split_path"]))
    if len(split) != int(protocol["split_rows"]) or any(
        row.get("step2g_split") != "proxy_dev" for row in split
    ):
        raise TarcProxyRunError("frozen proxy-dev split is invalid")
    targets = read_jsonl(resolve_repo_path(protocol["targets_path"]))
    overlap = prompt_overlap_counts(split, targets)
    if overlap["sample_id_overlap"] or overlap["exact_prompt_overlap"]:
        raise TarcProxyRunError("proxy-dev prompts overlap approved training targets")
    if (
        overlap["training_prompt_clusters"]
        != int(protocol["training_prompt_clusters"])
        or overlap["split_prompt_clusters"] != int(protocol["split_rows"])
        or protocol.get("training_prompt_overlap") != 0
    ):
        raise TarcProxyRunError("prompt-disjoint registry differs")

    if list(protocol["generation_order"]) != ["uniform", "ramp", "tarc"]:
        raise TarcProxyRunError("generation order differs")
    if protocol.get("candidate_arm") != "tarc" or list(
        protocol["control_arms"]
    ) != ["uniform", "ramp"]:
        raise TarcProxyRunError("candidate/control registry differs")
    for arm in protocol["generation_order"]:
        adapter = (
            resolve_repo_path(protocol["arms"][arm]["checkpoint_dir"])
            / "adapter_model.safetensors"
        )
        if (
            not adapter.is_file()
            or sha256(adapter) != protocol["arms"][arm]["adapter_sha256"]
        ):
            raise TarcProxyRunError(f"local adapter hash mismatch: {arm}")

    registry = evaluator.load_route_registry(protocol)
    structure = {
        "paths": len(registry["paths"]),
        "global_nodes": len(registry["global_nodes"]),
        "prompt_pair_tries": len(registry["pair_nodes"]),
        "pair_margin_terms": int(registry["pair_margin_terms"]),
        "route_width": int(registry["route_width"]),
    }
    if structure != protocol["route_registry_structure"]:
        raise TarcProxyRunError("route registry structure differs")
    return protocol


def require_clean_start(protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise TarcProxyRunError("proxy-dev generation requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.append(protocol["protocol_path"])
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    root = resolve_repo_path(protocol["generation_root"])
    if root.exists():
        raise TarcProxyRunError(f"prospective generation root already exists: {root}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    result = {
        "status": "PASS_TARC_PROXY_DEV_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "generation_order": protocol["generation_order"],
        "split_rows": protocol["split_rows"],
        "training_prompt_overlap": 0,
        "historical_proxy_reuse_disclosed": True,
        "route_registry_sha256": protocol["route_registry_sha256"],
        "route_registry_structure": protocol["route_registry_structure"],
        "decoding": protocol["decoding"],
        "effect_gate": protocol["effect_gate"],
        "optimizer_steps": 0,
        "blind_review_protocol_authorized": False,
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def generate_arm(protocol: dict[str, Any], arm: str, root: Path) -> None:
    script = resolve_repo_path(protocol["generation_script"])
    output = root / arm
    command = [
        sys.executable,
        str(script),
        "--config",
        protocol["config_path"],
        "--checkpoint-dir",
        protocol["arms"][arm]["checkpoint_dir"],
        "--arm",
        arm,
        "--split-path",
        protocol["split_path"],
        "--targets-path",
        protocol["targets_path"],
        "--output-dir",
        relative(output),
        "--max-new-tokens",
        str(protocol["decoding"]["max_new_tokens"]),
        "--route-width",
        str(protocol["decoding"]["route_width"]),
        "--prefix-width",
        str(protocol["decoding"]["prefix_width"]),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def write_package(
    protocol: dict[str, Any], path: Path, decision: dict[str, Any], runtime_head: str
) -> None:
    root = resolve_repo_path(protocol["generation_root"])
    decision["runtime_head"] = runtime_head
    decision["optimizer_steps"] = 0
    decision_path = resolve_repo_path(protocol["outputs"]["decision"])
    training.write_json(decision_path, decision)
    comparisons = decision["tarc_comparisons"]
    metrics = decision["arm_metrics"]
    report = f"""# TARC Prompt-Disjoint Proxy-Dev Effect Report

Route: `{decision['route']}`.

Exact registered four-token route rates were
`{metrics['uniform']['registered_route4_rate']:.6f}` (uniform),
`{metrics['ramp']['registered_route4_rate']:.6f}` (RAMP), and
`{metrics['tarc']['registered_route4_rate']:.6f}` (TARC). TARC paired deltas
were `{comparisons['uniform']['deltas']['registered_route4_rate']:.6f}` and
`{comparisons['ramp']['deltas']['registered_route4_rate']:.6f}`; exact
two-sided McNemar p-values were
`{comparisons['uniform']['paired_route_effect']['mcnemar_exact_two_sided_p']:.6f}`
and `{comparisons['ramp']['paired_route_effect']['mcnemar_exact_two_sided_p']:.6f}`.

Mean unseen-prompt frozen-pair TARC barriers were
`{metrics['uniform']['mean_route_pair_barrier']:.6f}` (uniform),
`{metrics['ramp']['mean_route_pair_barrier']:.6f}` (RAMP), and
`{metrics['tarc']['mean_route_pair_barrier']:.6f}` (TARC). Positive paired
barrier reductions for TARC were
`{comparisons['uniform']['paired_barrier_reduction']['mean']:.6f}` and
`{comparisons['ramp']['paired_barrier_reduction']['mean']:.6f}`.

Both control comparisons independently passed every frozen non-weak check:
`{decision['passed_both_controls']}`.

The 40 prompts had zero overlap with all TARC training prompt clusters. This
proxy split was used by an earlier RAMP screen and is therefore exploratory,
not fresh confirmation. Exact route completion and trie barriers are
mechanism-linked deterministic proxies, not semantic safety or utility labels.
Core100 and heldout remained sealed; generation used `optimizer_steps=0`.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_bytes(report.encode("utf-8"))
    artifacts = []
    for arm in protocol["generation_order"]:
        for name in ("responses.jsonl", "manifest.json"):
            artifact = root / arm / name
            artifacts.append(
                {
                    "path": relative(artifact),
                    "sha256": sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )
    for artifact in (decision_path, report_path):
        artifacts.append(
            {
                "path": relative(artifact),
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        )
    reproduction = {
        "name": "step3_tarc_proxy_dev_reproduction_manifest_v1",
        "route": decision["route"],
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": sha256(path),
        "artifacts": artifacts,
        "generation_order": protocol["generation_order"],
        "training_prompt_overlap": 0,
        "historical_proxy_reuse_disclosed": True,
        "route_registry_sha256": protocol["route_registry_sha256"],
        "optimizer_steps": 0,
        "blind_review_protocol_authorized": decision[
            "blind_review_protocol_authorized"
        ],
        "utility_review_protocol_authorized": decision[
            "utility_review_protocol_authorized"
        ],
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    training.write_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"]),
        reproduction,
    )


def run(path: Path) -> dict[str, Any]:
    protocol = context(path)
    runtime_head = require_clean_start(protocol)
    root = resolve_repo_path(protocol["generation_root"])
    root.mkdir(parents=True, exist_ok=False)
    for arm in protocol["generation_order"]:
        generate_arm(protocol, arm, root)
    decision = evaluator.evaluate_outputs(protocol, path, root)
    write_package(protocol, path, decision, runtime_head)
    print(
        json.dumps(
            {
                "route": decision["route"],
                "passed_both_controls": decision["passed_both_controls"],
                "optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol = context(path)
    reproduction = read_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"])
    )
    if reproduction.get("protocol_sha256") != sha256(path):
        raise TarcProxyRunError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != artifact["sha256"]
            or artifact_path.stat().st_size != artifact["bytes"]
        ):
            raise TarcProxyRunError(f"artifact mismatch: {artifact['path']}")
    recomputed = evaluator.evaluate_outputs(
        protocol, path, resolve_repo_path(protocol["generation_root"])
    )
    stored = read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if (
        recomputed["route"] != stored.get("route")
        or recomputed["route"] != reproduction.get("route")
        or recomputed["tarc_comparisons"] != stored.get("tarc_comparisons")
        or recomputed["arm_metrics"] != stored.get("arm_metrics")
    ):
        raise TarcProxyRunError("stored and recomputed proxy decisions differ")
    result = {
        "status": "PASS_TARC_PROXY_DEV_RESULT_VERIFICATION",
        "route": recomputed["route"],
        "generation_arms": len(protocol["generation_order"]),
        "rows_per_arm": protocol["split_rows"],
        "optimizer_steps": 0,
        "blind_review_protocol_authorized": recomputed[
            "blind_review_protocol_authorized"
        ],
        "utility_review_protocol_authorized": recomputed[
            "utility_review_protocol_authorized"
        ],
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "run":
            run(path)
        else:
            verify_run(path)
    except (
        TarcProxyRunError,
        evaluator.TarcProxyValidationError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_TARC_PROXY_DEV_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
