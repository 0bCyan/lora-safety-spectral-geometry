from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
REGISTRY_PATH = PAPER_ROOT / "configs" / "study_registry.json"
TRAINER = "experiments/step2_mechanism_dynamics/scripts/train_lora_with_checkpoints.py"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def ordering_key(run_order_seed: int, record: dict[str, Any]) -> str:
    text = "|".join(
        [
            str(run_order_seed),
            record["role"],
            record["config"],
            record["arm"],
            str(record["seed"]),
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render deterministic Paper A formal run commands.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=PAPER_ROOT / "data" / "derived" / "run_commands.txt")
    parser.add_argument("--manifest", type=Path, default=PAPER_ROOT / "data" / "derived" / "run_commands.json")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    if not args.skip_validation:
        completed = subprocess.run(
            [sys.executable, str(PAPER_ROOT / "scripts" / "validate_study.py")],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit("Study validation failed; commands were not rendered.")

    registry_path = args.registry if args.registry.is_absolute() else REPO_ROOT / args.registry
    registry = load_json(registry_path)
    records: list[dict[str, Any]] = []
    for formal in registry["formal_configs"]:
        config_path = REPO_ROOT / formal["path"]
        config_hash = sha256(config_path)
        for arm in formal["arms"]:
            for seed in formal["seeds"]:
                record = {
                    "role": formal["role"],
                    "config": formal["path"],
                    "config_sha256": config_hash,
                    "arm": arm,
                    "seed": int(seed),
                    "family": "risk",
                }
                record["order_key"] = ordering_key(int(registry["run_order_seed"]), record)
                base = [
                    "python",
                    TRAINER,
                    "--config",
                    record["config"],
                    "--family",
                    record["family"],
                    "--seed",
                    str(record["seed"]),
                    "--run-label",
                    record["arm"],
                ]
                record["command"] = shell_join(base)
                record["dry_run_command"] = shell_join(base + ["--dry-run"])
                records.append(record)

    records.sort(key=lambda row: row["order_key"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper A frozen formal run commands",
        f"# registry={registry_path.relative_to(REPO_ROOT)}",
        f"# run_order_seed={registry['run_order_seed']}",
        f"# command_count={len(records)}",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index:02d} {record['role']} arm={record['arm']} seed={record['seed']}",
                f"# order_key={record['order_key']}",
                f"# config_sha256={record['config_sha256']}",
                f"DRY_RUN: {record['dry_run_command']}",
                f"FORMAL:  {record['command']}",
                "",
            ]
        )
    args.output.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "schema_version": "paper_a_run_command_manifest_v1",
        "registry": str(registry_path.relative_to(REPO_ROOT)),
        "registry_sha256": sha256(registry_path),
        "run_order_seed": registry["run_order_seed"],
        "commands": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} commands to {args.output}")
    print(f"wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
