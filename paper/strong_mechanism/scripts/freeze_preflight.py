from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def pinned_files(registry: dict[str, Any]) -> list[Path]:
    relative = [
        "paper/strong_mechanism/CLAIMS.md",
        "paper/strong_mechanism/EVIDENCE_MAP.md",
        "paper/strong_mechanism/configs/study_registry.json",
        "paper/strong_mechanism/configs/qwen3_4b_equal_dose_temporal_schedules.json",
        "paper/strong_mechanism/configs/qwen3_4b_context_replication.json",
        "paper/strong_mechanism/configs/safety_behavior_rubric_v1.json",
        "paper/strong_mechanism/configs/quality_rubric_v1.json",
        "paper/strong_mechanism/protocols/P0_claim_freeze.md",
        "paper/strong_mechanism/protocols/P1_equal_dose_temporal_distribution.md",
        "paper/strong_mechanism/protocols/P2_multiseed_replication.md",
        "paper/strong_mechanism/scripts/run_pinned_training.py",
        "paper/strong_mechanism/scripts/run_registered_gradient_audit.py",
        "experiments/step2_mechanism_dynamics/scripts/train_lora_with_checkpoints.py",
        "experiments/step2_mechanism_dynamics/scripts/evaluate_checkpoint_dynamics.py",
    ]
    relative.extend(item["path"] for item in registry["formal_configs"])
    return sorted({REPO_ROOT / value for value in relative})


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the immutable Paper A M1 preflight lock.")
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=PAPER_ROOT / "data" / "derived" / "run_commands.json",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=PAPER_ROOT / "data" / "derived" / "evaluation_commands.json",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=PAPER_ROOT / "data" / "runs" / "runtime_configs" / "runtime_manifest.json",
    )
    parser.add_argument(
        "--dry-run-report",
        type=Path,
        default=PAPER_ROOT / "data" / "runs" / "preflight" / "dry_run_report.json",
    )
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_ROOT / "preflight" / "M1_preflight_lock.json",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Preflight lock already exists; use a versioned amendment: {args.output}")
    inputs = [args.run_manifest, args.evaluation_manifest, args.runtime_manifest, args.dry_run_report, args.smoke_report]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing preflight input: {path}")

    status = git("status", "--short")
    if status:
        raise RuntimeError("Worktree must be clean before creating the preflight lock")
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")

    registry = load_json(PAPER_ROOT / "configs" / "study_registry.json")
    runs = load_json(args.run_manifest)
    evaluations = load_json(args.evaluation_manifest)
    runtimes = load_json(args.runtime_manifest)
    dry_runs = load_json(args.dry_run_report)
    smoke = load_json(args.smoke_report)

    if len(runs["commands"]) != int(registry["expected_formal_runs"]):
        raise RuntimeError("Training command count does not match registry")
    if len(evaluations["jobs"]) != int(registry["expected_evaluation_jobs"]):
        raise RuntimeError("Evaluation job count does not match registry")
    if len(runtimes["runtime_configs"]) != int(registry["expected_formal_runs"]):
        raise RuntimeError("Runtime config count does not match registry")
    if runtimes.get("environment_checked") is not True:
        raise RuntimeError("Strict environment was not verified")
    if dry_runs.get("passed") is not True or dry_runs.get("job_count") != 60:
        raise RuntimeError("All 60 dry-run jobs must pass")

    required_smoke = [
        "model_revision_verified",
        "environment_lock_verified",
        "model_load_pass",
        "one_optimizer_step_pass",
        "checkpoint_write_pass",
        "evaluator_load_pass",
        "state_cleanup_pass",
    ]
    if smoke.get("status") != "pass" or not all(smoke.get(key) is True for key in required_smoke):
        raise RuntimeError("Non-formal smoke report does not satisfy every required gate")
    if smoke.get("formal_seed_used") is not False or smoke.get("formal_output_directory_used") is not False:
        raise RuntimeError("Smoke job contaminated the formal seed set or formal output namespace")

    files = pinned_files(registry)
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing pinned file: {path}")
    document = {
        "schema_version": "paper_a_m1_preflight_lock_v1",
        "branch": branch,
        "commit": commit,
        "model_repository": registry["model_repository"],
        "model_revision": registry["model_revision"],
        "formal_training_runs": len(runs["commands"]),
        "formal_evaluation_jobs": len(evaluations["jobs"]),
        "dry_run_jobs": dry_runs["job_count"],
        "inputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "pinned_files": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pinned_files": len(files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
