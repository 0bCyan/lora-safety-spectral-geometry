#!/usr/bin/env python3
"""Register and optionally execute the frozen validation-v1 job matrix.

The script intentionally separates job registration from execution. Training and
evaluation entrypoints are supplied as templates so the same protocol can wrap
the repository's existing scripts or a cluster launcher without changing the
experimental matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def render(template: str, values: dict[str, object]) -> str:
    return template.format(**{k: str(v) for k, v in values.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/validation_v1/runs"))
    ap.add_argument("--train-template", default="", help="Optional command template with {model_id} {arm} {seed} {run_label}.")
    ap.add_argument("--eval-template", default="", help="Optional command template with {model_id} {arm} {seed} {checkpoint} {dataset}.")
    ap.add_argument("--execute", action="store_true", help="Execute commands sequentially after writing the manifest.")
    ap.add_argument("--dry-run", action="store_true", help="Only write the matrix and commands.")
    args = ap.parse_args()
    if args.execute and not args.train_template:
        ap.error("--execute requires --train-template")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = cfg.get("eval_datasets", ["beavertails", "sorry_bench", "harmbench", "utility"])
    jobs = []
    commands = []
    for model in cfg["models"]:
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                job_id = f"{model['id']}__{arm['id']}__s{seed}"
                values = {"model_id": model["id"], "model_path": model["model_name_or_path"], "arm": arm["id"], "run_label": arm.get("run_label") or "", "seed": seed, "job_id": job_id}
                train_cmd = render(args.train_template, values) if args.train_template else ""
                jobs.append({"job_id": job_id, "model": model, "arm": arm, "seed": seed, "train_command": train_cmd})
                if train_cmd:
                    commands.append(train_cmd)
                for checkpoint in cfg["checkpoints"]:
                    for dataset in datasets:
                        eval_values = {**values, "checkpoint": checkpoint, "dataset": dataset}
                        eval_cmd = render(args.eval_template, eval_values) if args.eval_template else ""
                        jobs.append({"job_id": job_id, "checkpoint": checkpoint, "dataset": dataset, "eval_command": eval_cmd})
                        if eval_cmd:
                            commands.append(eval_cmd)

    manifest = {"protocol_version": cfg["protocol_version"], "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(args.config), "jobs": jobs}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    command_file = args.output_dir / "run_commands.sh"
    command_file.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    command_file.chmod(0o755)
    print(json.dumps({"jobs": len(jobs), "commands": len(commands), "manifest": str(args.output_dir / 'run_manifest.json'), "commands_file": str(command_file)}, indent=2))
    if args.execute and not args.dry_run:
        for command in commands:
            subprocess.run(command, shell=True, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
