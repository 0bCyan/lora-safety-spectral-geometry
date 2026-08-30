from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def argv_from_command(command: str) -> list[str]:
    argv = shlex.split(command, posix=True)
    if not argv or argv[0] != "python":
        raise ValueError(f"Dry-run command must begin with python: {command}")
    argv[0] = sys.executable
    return argv


def run_command(command: str, log_prefix: Path) -> dict[str, Any]:
    argv = argv_from_command(command)
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path = log_prefix.with_suffix(".stdout.txt")
    stderr_path = log_prefix.with_suffix(".stderr.txt")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "argv": argv,
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout_path.relative_to(REPO_ROOT)),
        "stdout_sha256": sha256(stdout_path),
        "stderr": str(stderr_path.relative_to(REPO_ROOT)),
        "stderr_sha256": sha256(stderr_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute and freeze all Paper A training/evaluation dry-runs.")
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
        "--output",
        type=Path,
        default=PAPER_ROOT / "data" / "runs" / "preflight" / "dry_run_report.json",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PAPER_ROOT / "data" / "runs" / "preflight" / "dry_run_logs",
    )
    args = parser.parse_args()

    for path in [args.run_manifest, args.evaluation_manifest, args.runtime_manifest]:
        if not path.exists():
            raise FileNotFoundError(f"Missing preflight input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Dry-run report already exists; refusing overwrite: {args.output}")

    run_manifest = load_json(args.run_manifest)
    evaluation_manifest = load_json(args.evaluation_manifest)
    runtime_manifest = load_json(args.runtime_manifest)
    if runtime_manifest.get("environment_checked") is not True:
        raise RuntimeError("Runtime manifest was created without strict environment validation")

    jobs = []
    for index, record in enumerate(run_manifest["commands"], start=1):
        jobs.append((f"train_{index:02d}", "training", record["dry_run_command"]))
    for index, record in enumerate(evaluation_manifest["jobs"], start=1):
        jobs.append((f"eval_{index:03d}", record["kind"], record["dry_run_command"]))

    args.log_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    results = []
    for name, kind, command in jobs:
        result = run_command(command, args.log_dir / name)
        result["name"] = name
        result["kind"] = kind
        results.append(result)
        print(f"{name}: returncode={result['returncode']}")
    finished = now()

    passed = all(result["returncode"] == 0 for result in results)
    report = {
        "schema_version": "paper_a_dry_run_report_v1",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "run_manifest": str(args.run_manifest.relative_to(REPO_ROOT)),
        "run_manifest_sha256": sha256(args.run_manifest),
        "evaluation_manifest": str(args.evaluation_manifest.relative_to(REPO_ROOT)),
        "evaluation_manifest_sha256": sha256(args.evaluation_manifest),
        "runtime_manifest": str(args.runtime_manifest.relative_to(REPO_ROOT)),
        "runtime_manifest_sha256": sha256(args.runtime_manifest),
        "jobs": results,
        "job_count": len(results),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"job_count": len(results), "passed": passed}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
