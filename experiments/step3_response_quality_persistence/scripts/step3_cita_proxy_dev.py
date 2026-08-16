"""Generate and evaluate the prospective CITA proxy-dev effect gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import evaluate_step3_cita_proxy_dev as evaluator
import step3ct_training_common as training


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cita/"
    "protocol/cita_proxy_dev_effect_protocol.json"
)


class CitaProxyRunError(ValueError):
    """Raised when proxy-dev generation violates the frozen protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    return evaluator.resolve_repo_path(value)


def read_json(path: Path) -> Any:
    return evaluator.read_json(path)


def sha256(path: Path) -> str:
    return evaluator.sha256(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if protocol.get("status") != "prospective_before_any_CITA_proxy_dev_generation":
        raise CitaProxyRunError("proxy protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise CitaProxyRunError(f"frozen input hash mismatch: {name}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise CitaProxyRunError("entry commit is not an ancestor of HEAD")
    development = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["development_decision"]["path"])
    )
    if development.get("route") != "PASS_CITA_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE":
        raise CitaProxyRunError("development route did not authorize generation freeze")
    if protocol.get("proxy_dev_generation_authorized") is not True:
        raise CitaProxyRunError("proxy-dev generation is not authorized")
    if protocol.get("main_training_authorized") is not False:
        raise CitaProxyRunError("main training must remain unauthorized")
    split = evaluator.read_jsonl(resolve_repo_path(protocol["split_path"]))
    if len(split) != int(protocol["split_rows"]) or any(
        row.get("step2g_split") != "proxy_dev" for row in split
    ):
        raise CitaProxyRunError("frozen proxy-dev split is invalid")
    for arm in protocol["generation_order"]:
        adapter = (
            resolve_repo_path(protocol["arms"][arm]["checkpoint_dir"])
            / "adapter_model.safetensors"
        )
        if not adapter.is_file() or sha256(adapter) != protocol["arms"][arm]["adapter_sha256"]:
            raise CitaProxyRunError(f"local adapter hash mismatch: {arm}")
    return protocol


def require_clean_start(protocol: dict[str, Any]) -> str:
    if git_output("status", "--porcelain"):
        raise CitaProxyRunError("proxy-dev generation requires a clean worktree")
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
        raise CitaProxyRunError(f"prospective generation root already exists: {root}")
    return git_output("rev-parse", "HEAD")


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    result = {
        "status": "PASS_CITA_PROXY_DEV_PREFLIGHT",
        "protocol_sha256": sha256(path),
        "generation_order": protocol["generation_order"],
        "split_rows": protocol["split_rows"],
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
        "--output-dir",
        relative(output),
        "--max-new-tokens",
        str(protocol["decoding"]["max_new_tokens"]),
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
    comparisons = decision["cita_comparisons"]
    report = f"""# CITA Proxy-Dev Generated-Behavior Effect Report

Route: `{decision['route']}`.

CITA qualified-safe proxy deltas were `{comparisons['uniform']['deltas']['qualified_safe_proxy_rate']:.6f}` versus uniform and `{comparisons['sham']['deltas']['qualified_safe_proxy_rate']:.6f}` versus exact-budget sham. The paired bootstrap lower bounds were `{comparisons['uniform']['paired_qualified_effect']['ci95_lower']:.6f}` and `{comparisons['sham']['paired_qualified_effect']['ci95_lower']:.6f}`. Both controls passed all registered non-weak checks: `{decision['passed_both_controls']}`.

This deterministic marker gate is development triage, not a human safety judgment. It cannot establish publication-level safety or utility. Core100 and heldout remained sealed, and generation executed `optimizer_steps=0`. A pass authorizes only a separately frozen blinded three-arm review protocol; a failure stops CITA training.
"""
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    report_path.write_text(report, encoding="utf-8")
    artifacts = []
    for arm in protocol["generation_order"]:
        for name in ("responses.jsonl", "manifest.json"):
            artifact = root / arm / name
            artifacts.append(
                {"path": relative(artifact), "sha256": sha256(artifact), "bytes": artifact.stat().st_size}
            )
    for artifact in (decision_path, report_path):
        artifacts.append(
            {"path": relative(artifact), "sha256": sha256(artifact), "bytes": artifact.stat().st_size}
        )
    reproduction = {
        "name": "step3_cita_proxy_dev_reproduction_manifest_v1",
        "route": decision["route"],
        "runtime_head": runtime_head,
        "protocol": relative(path),
        "protocol_sha256": sha256(path),
        "artifacts": artifacts,
        "generation_order": protocol["generation_order"],
        "optimizer_steps": 0,
        "blind_review_protocol_authorized": decision["blind_review_protocol_authorized"],
        "main_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    training.write_json(
        resolve_repo_path(protocol["outputs"]["reproduction_manifest"]), reproduction
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
    reproduction = read_json(resolve_repo_path(protocol["outputs"]["reproduction_manifest"]))
    if reproduction.get("protocol_sha256") != sha256(path):
        raise CitaProxyRunError("reproduction protocol hash mismatch")
    for artifact in reproduction["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise CitaProxyRunError(f"artifact hash mismatch: {artifact['path']}")
    recomputed = evaluator.evaluate_outputs(
        protocol, path, resolve_repo_path(protocol["generation_root"])
    )
    stored = read_json(resolve_repo_path(protocol["outputs"]["decision"]))
    if (
        recomputed["route"] != stored.get("route")
        or recomputed["route"] != reproduction.get("route")
        or recomputed["cita_comparisons"] != stored.get("cita_comparisons")
    ):
        raise CitaProxyRunError("stored and recomputed proxy decisions differ")
    result = {
        "status": "PASS_CITA_PROXY_DEV_RESULT_VERIFICATION",
        "route": recomputed["route"],
        "generation_arms": len(protocol["generation_order"]),
        "rows_per_arm": protocol["split_rows"],
        "optimizer_steps": 0,
        "blind_review_protocol_authorized": recomputed[
            "blind_review_protocol_authorized"
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
        CitaProxyRunError,
        evaluator.CitaProxyValidationError,
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"STOP_CITA_PROXY_DEV_INVALID: {exc}") from exc


if __name__ == "__main__":
    main()
