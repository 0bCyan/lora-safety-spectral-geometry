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
CANONICAL_EVALUATOR = "experiments/step2_mechanism_dynamics/scripts/evaluate_checkpoint_dynamics.py"
GRADIENT_AUDIT = "paper/strong_mechanism/scripts/run_registered_gradient_audit.py"

CANONICAL_SCRIPTS = REPO_ROOT / "experiments" / "step2_mechanism_dynamics" / "scripts"
sys.path.insert(0, str(CANONICAL_SCRIPTS))
from common import EVAL_TO_FILE, checkpoint_tag, run_id  # noqa: E402

sys.path.insert(0, str(PAPER_ROOT / "scripts"))
from run_pinned_training import runtime_path  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def order_key(seed: int, record: dict[str, Any]) -> str:
    payload = "|".join(
        [str(seed), record["kind"], record["config"], record["arm"], str(record["seed"])]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generation_path(config: dict[str, Any], arm: str, seed: int, checkpoint: str, eval_set: str, output_tag: str) -> Path:
    identifier = run_id(config, "risk", seed, arm)
    return (
        Path(config["output_dir"])
        / f"generations_{output_tag}"
        / f"{identifier}_{checkpoint}_{eval_set}_{output_tag}.jsonl"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render deterministic Paper A generation and gradient-audit commands.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=PAPER_ROOT / "data" / "derived" / "evaluation_commands.txt")
    parser.add_argument("--manifest", type=Path, default=PAPER_ROOT / "data" / "derived" / "evaluation_commands.json")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    if not args.skip_validation:
        result = subprocess.run(
            [sys.executable, str(PAPER_ROOT / "scripts" / "validate_study.py")],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit("Study validation failed; evaluation commands were not rendered.")

    registry_path = args.registry if args.registry.is_absolute() else REPO_ROOT / args.registry
    registry = load_json(registry_path)
    total_steps = int(registry["expected_total_steps"])
    checkpoint_tags = [checkpoint_tag(int(step), total_steps) for step in registry["registered_checkpoints"]]
    final_checkpoint = checkpoint_tags[-1]
    jobs: list[dict[str, Any]] = []

    for formal in registry["formal_configs"]:
        source = REPO_ROOT / formal["path"]
        config = load_json(source)
        for arm in formal["arms"]:
            for seed in formal["seeds"]:
                runtime = runtime_path(source, arm, int(seed))
                common = {
                    "role": formal["role"],
                    "config": formal["path"],
                    "config_sha256": sha256(source),
                    "runtime_config": str(runtime.relative_to(REPO_ROOT)),
                    "arm": arm,
                    "seed": int(seed),
                    "family": "risk",
                    "model": registry["model_repository"],
                    "model_revision": registry["model_revision"],
                }

                safety_tag = "paperA_safety100"
                safety_base = [
                    "python",
                    CANONICAL_EVALUATOR,
                    "--config",
                    common["runtime_config"],
                    "--family",
                    "risk",
                    "--seed",
                    str(seed),
                    "--run-label",
                    arm,
                    "--eval-set",
                    "safety",
                    "--limit",
                    "100",
                    "--checkpoints",
                    ",".join(checkpoint_tags),
                    "--output-tag",
                    safety_tag,
                ]
                safety_job = {
                    **common,
                    "kind": "safety_generation",
                    "eval_set": "safety",
                    "limit": 100,
                    "checkpoints": checkpoint_tags,
                    "canonical_eval_path": str(Path(config["data_dir"]) / EVAL_TO_FILE["safety"]),
                    "output_tag": safety_tag,
                    "expected_generation_paths": [
                        {
                            "checkpoint": int(step),
                            "checkpoint_tag": tag,
                            "path": str(generation_path(config, arm, int(seed), tag, "safety", safety_tag)),
                        }
                        for step, tag in zip(registry["registered_checkpoints"], checkpoint_tags)
                    ],
                    "command": shell_join(safety_base),
                    "dry_run_command": shell_join([*safety_base, "--dry-run"]),
                }
                safety_job["order_key"] = order_key(int(registry["evaluation_order_seed"]), safety_job)
                jobs.append(safety_job)

                utility_tag = "paperA_utility100"
                utility_base = [
                    "python",
                    CANONICAL_EVALUATOR,
                    "--config",
                    common["runtime_config"],
                    "--family",
                    "risk",
                    "--seed",
                    str(seed),
                    "--run-label",
                    arm,
                    "--eval-set",
                    "utility",
                    "--limit",
                    "100",
                    "--checkpoints",
                    final_checkpoint,
                    "--output-tag",
                    utility_tag,
                ]
                utility_job = {
                    **common,
                    "kind": "utility_generation",
                    "eval_set": "utility",
                    "limit": 100,
                    "checkpoints": [final_checkpoint],
                    "canonical_eval_path": str(Path(config["data_dir"]) / EVAL_TO_FILE["utility"]),
                    "output_tag": utility_tag,
                    "expected_generation_paths": [
                        {
                            "checkpoint": int(registry["registered_checkpoints"][-1]),
                            "checkpoint_tag": final_checkpoint,
                            "path": str(generation_path(config, arm, int(seed), final_checkpoint, "utility", utility_tag)),
                        }
                    ],
                    "command": shell_join(utility_base),
                    "dry_run_command": shell_join([*utility_base, "--dry-run"]),
                }
                utility_job["order_key"] = order_key(int(registry["evaluation_order_seed"]), utility_job)
                jobs.append(utility_job)

                gradient_base = [
                    "python",
                    GRADIENT_AUDIT,
                    "--config",
                    common["runtime_config"],
                    "--family",
                    "risk",
                    "--seed",
                    str(seed),
                    "--arm",
                    arm,
                ]
                gradient_job = {
                    **common,
                    "kind": "gradient_audit",
                    "checkpoints": [checkpoint_tags[3], checkpoint_tags[4], checkpoint_tags[5]],
                    "command": shell_join(gradient_base),
                    "dry_run_command": shell_join([*gradient_base, "--dry-run"]),
                }
                gradient_job["order_key"] = order_key(int(registry["evaluation_order_seed"]), gradient_job)
                jobs.append(gradient_job)

    jobs.sort(key=lambda item: item["order_key"])
    expected = int(registry["expected_evaluation_jobs"])
    if len(jobs) != expected:
        raise RuntimeError(f"Expected {expected} evaluation jobs, rendered {len(jobs)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper A frozen evaluation and registered gradient-audit commands",
        f"# registry={registry_path.relative_to(REPO_ROOT)}",
        f"# command_count={len(jobs)}",
        f"# model={registry['model_repository']}@{registry['model_revision']}",
        "# Local heuristic judge files produced by the canonical evaluator are not Paper A primary labels.",
        "",
    ]
    for index, job in enumerate(jobs, start=1):
        lines.extend(
            [
                f"## {index:03d} kind={job['kind']} arm={job['arm']} seed={job['seed']}",
                f"# order_key={job['order_key']}",
                f"DRY_RUN: {job['dry_run_command']}",
                f"FORMAL:  {job['command']}",
                "",
            ]
        )
    args.output.write_text("\n".join(lines), encoding="utf-8")

    document = {
        "schema_version": "paper_a_evaluation_command_manifest_v1",
        "registry": str(registry_path.relative_to(REPO_ROOT)),
        "registry_sha256": sha256(registry_path),
        "model": registry["model_repository"],
        "model_revision": registry["model_revision"],
        "jobs": jobs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(jobs)} evaluation jobs to {args.output}")
    print(f"wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
