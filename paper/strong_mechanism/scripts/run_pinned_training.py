from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from prepare_runtime_config import REPO_ROOT, load_json, prepare_runtime_config, validate_paper_config


PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
CANONICAL_SCRIPTS = REPO_ROOT / "experiments" / "step2_mechanism_dynamics" / "scripts"
CANONICAL_TRAINER = CANONICAL_SCRIPTS / "train_lora_with_checkpoints.py"
sys.path.insert(0, str(CANONICAL_SCRIPTS))
from common import run_dir  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_path(source: Path, run_label: str, seed: int) -> Path:
    label = "".join(character if character.isalnum() or character in "-_" else "_" for character in run_label)
    return PAPER_ROOT / "data" / "runs" / "runtime_configs" / f"{source.stem}__{label}__seed{seed}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Paper A formal training job with model and environment pins.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    source = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_json(source)
    validate_paper_config(config)

    base_command = [
        sys.executable,
        str(CANONICAL_TRAINER),
        "--config",
        str(source),
        "--family",
        args.family,
        "--seed",
        str(args.seed),
        "--run-label",
        args.run_label,
    ]
    if args.dry_run:
        completed = subprocess.run([*base_command, "--dry-run"], cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
        print(json.dumps({
            "paper_pin_valid": True,
            "machine": config["machine"],
            "model_repository": config["model_name_or_path"],
            "model_revision": config["model_revision"],
            "environment_lock": config["environment_lock"],
        }, ensure_ascii=False))
        return 0

    runtime_config_path = runtime_path(source, args.run_label, args.seed)
    runtime, provenance = prepare_runtime_config(
        source,
        runtime_config_path,
        offline=args.offline,
        verify_machine=True,
    )
    root = run_dir(runtime, args.family, args.seed, args.run_label)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Formal run directory is not empty; refusing overwrite or silent retry: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)

    formal_command = [
        sys.executable,
        str(CANONICAL_TRAINER),
        "--config",
        str(runtime_config_path),
        "--family",
        args.family,
        "--seed",
        str(args.seed),
        "--run-label",
        args.run_label,
    ]
    run_record = {
        **provenance,
        "schema_version": "paper_a_formal_run_provenance_v1",
        "family": args.family,
        "seed": args.seed,
        "run_label": args.run_label,
        "canonical_trainer": str(CANONICAL_TRAINER.relative_to(REPO_ROOT)),
        "canonical_trainer_sha256": sha256(CANONICAL_TRAINER),
        "formal_command": formal_command,
        "formal_run_dir": str(root),
        "started_at_utc": now(),
        "status": "started_before_model_load",
    }
    record_path = root / "paper_formal_run_provenance.json"
    write_json(record_path, run_record)

    completed = subprocess.run(formal_command, cwd=REPO_ROOT, check=False)
    run_record["finished_at_utc"] = now()
    run_record["returncode"] = completed.returncode
    run_record["status"] = "completed" if completed.returncode == 0 else "failed_preserved_no_retry"
    write_json(record_path, run_record)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
