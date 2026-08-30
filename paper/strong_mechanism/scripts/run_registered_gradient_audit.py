from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
CANONICAL_SCRIPTS = REPO_ROOT / "experiments" / "step2_mechanism_dynamics" / "scripts"
sys.path.insert(0, str(CANONICAL_SCRIPTS))

from common import FAMILY_TO_FILE, checkpoint_dir, load_json, read_jsonl, run_id, write_csv  # noqa: E402
from train_lora_with_checkpoints import get_arm_config  # noqa: E402
import run_step2j_natural_gradient_audit as legacy  # noqa: E402


DEFAULT_CHECKPOINTS = ["step0032_pct050", "step0048_pct075", "step0064_pct100"]
GROUPS = ["all", *legacy.GROUPS]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def trainer_schedule_weight(arm: dict[str, Any], completed_steps: int) -> float:
    for window in arm.get("anchor_schedule") or []:
        start = int(window.get("start_step", 0))
        end = window.get("end_step")
        if completed_steps >= start and (end is None or completed_steps < int(end)):
            if "anchor_mix_start" in window and "anchor_mix_end" in window:
                if end is None or int(end) <= start:
                    return float(window["anchor_mix_end"])
                progress = (completed_steps - start) / max(1, int(end) - start)
                start_mix = float(window["anchor_mix_start"])
                end_mix = float(window["anchor_mix_end"])
                return start_mix + progress * (end_mix - start_mix)
            return float(window.get("anchor_mix", 0.0))
    return 0.0


def checkpoint_schedule_weight(arm: dict[str, Any], checkpoint_step: int, total_steps: int) -> tuple[float, int]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    # A checkpoint before the end uses the weight that the next optimizer update
    # would receive. The final checkpoint has no next update, so it uses the last
    # weight actually applied by the trainer at completed_steps=total_steps-1.
    schedule_step = checkpoint_step if checkpoint_step < total_steps else total_steps - 1
    schedule_step = max(0, schedule_step)
    return trainer_schedule_weight(arm, schedule_step), schedule_step


def schedule_vector(arm: dict[str, Any], total_steps: int) -> list[float]:
    return [trainer_schedule_weight(arm, step) for step in range(total_steps)]


def read_checkpoint_index(config: dict[str, Any], family: str, seed: int, arm: str) -> dict[str, dict[str, str]]:
    path = Path(config["output_dir"]) / "runs" / run_id(config, family, seed, arm) / "checkpoint_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {path}")
    return legacy.read_checkpoint_index(config, family, seed, arm)


def ids_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(str(row["sample_id"]) for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the registered schedule-aware Paper A gradient contribution audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", default="risk", choices=sorted(FAMILY_TO_FILE))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoints", default=",".join(DEFAULT_CHECKPOINTS))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--setting", default="qwen3_4b_paper_a")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    arm = get_arm_config(config, args.arm)
    total_steps = int(config.get("paper_design", {}).get("total_steps", 64))
    checkpoints = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    vector = schedule_vector(arm, total_steps)
    schedule_hash = sha256_text(json.dumps(vector, separators=(",", ":")))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": args.config,
                    "family": args.family,
                    "seed": args.seed,
                    "arm": args.arm,
                    "checkpoints": checkpoints,
                    "total_steps": total_steps,
                    "schedule_sum": sum(vector),
                    "schedule_hash": schedule_hash,
                    "checkpoint_weights": {
                        checkpoint: checkpoint_schedule_weight(
                            arm,
                            int(checkpoint.split("_")[0].replace("step", "")),
                            total_steps,
                        )[0]
                        for checkpoint in checkpoints
                    },
                },
                ensure_ascii=False,
            )
        )
        return 0

    block_size = int(args.block_size)
    risk_rows = read_jsonl(Path(config["data_dir"]) / FAMILY_TO_FILE[args.family], limit=block_size * 2)
    safety_rows = legacy.fixed_safety_rows(arm, block_size * 2)
    if len(risk_rows) != block_size * 2 or len(safety_rows) != block_size * 2:
        raise ValueError("Registered gradient audit requires exactly two complete risk and safety blocks")

    index = read_checkpoint_index(config, args.family, args.seed, args.arm)
    out_dir = (
        Path(config["output_dir"])
        / "paper_gradient_audit"
        / run_id(config, args.family, args.seed, args.arm)
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Gradient audit output is not empty; refusing overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        if checkpoint not in index:
            raise KeyError(f"Checkpoint {checkpoint} is not present in the run index")
        adapter = checkpoint_dir(config, args.family, args.seed, checkpoint, args.arm)
        step = int(index[checkpoint]["step"])
        weight, schedule_step = checkpoint_schedule_weight(arm, step, total_steps)
        print(
            f"registered gradient audit arm={args.arm} seed={args.seed} "
            f"checkpoint={checkpoint} checkpoint_step={step} schedule_step={schedule_step} weight={weight:.9f}"
        )
        model, tokenizer = legacy.load_model(config, adapter)
        for block in range(2):
            risk_block = risk_rows[block * block_size : (block + 1) * block_size]
            safety_block = safety_rows[block * block_size : (block + 1) * block_size]
            risk_vectors = []
            safety_vectors = []
            for row in risk_block:
                vectors, loss = legacy.gradient_vectors_for_row(model, tokenizer, row, int(config["max_length"]))
                risk_vectors.append((row["sample_id"], vectors, loss))
            for row in safety_block:
                vectors, loss = legacy.gradient_vectors_for_row(model, tokenizer, row, int(config["max_length"]))
                safety_vectors.append((row["sample_id"], vectors, loss))

            grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
            for risk_index, (risk_id, risk_vecs, risk_loss) in enumerate(risk_vectors):
                for safety_index, (safety_id, safety_vecs, safety_loss) in enumerate(safety_vectors):
                    for group in GROUPS:
                        metrics = legacy.pair_metrics(risk_vecs[group], safety_vecs[group], weight)
                        item = {
                            "setting": args.setting,
                            "model": config.get("model_repository", config["model_name_or_path"]),
                            "model_revision": config.get("resolved_model_revision", config.get("model_revision", "")),
                            "arm": args.arm,
                            "family": args.family,
                            "seed": args.seed,
                            "checkpoint": checkpoint,
                            "checkpoint_step": step,
                            "schedule_step": schedule_step,
                            "schedule_weight": weight,
                            "schedule_hash": schedule_hash,
                            "block_id": block + 1,
                            "group": group,
                            "risk_index": risk_index,
                            "safety_index": safety_index,
                            "risk_sample_id": risk_id,
                            "safety_sample_id": safety_id,
                            "risk_loss": risk_loss,
                            "safety_loss": safety_loss,
                            **metrics,
                        }
                        pair_rows.append(item)
                        grouped[group].append(item)

            for group, rows in grouped.items():
                summary = legacy.summarize(rows)
                summary.update(
                    {
                        "setting": args.setting,
                        "model": config.get("model_repository", config["model_name_or_path"]),
                        "model_revision": config.get("resolved_model_revision", config.get("model_revision", "")),
                        "arm": args.arm,
                        "family": args.family,
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "checkpoint_step": step,
                        "schedule_step": schedule_step,
                        "schedule_weight": weight,
                        "schedule_hash": schedule_hash,
                        "block_id": block + 1,
                        "group": group,
                    }
                )
                summary_rows.append(summary)
                if group == "all":
                    normalized_rows.append(
                        {
                            "setting": args.setting,
                            "model": summary["model"],
                            "model_revision": summary["model_revision"],
                            "arm": args.arm,
                            "seed": args.seed,
                            "checkpoint": step,
                            "checkpoint_tag": checkpoint,
                            "block_id": block + 1,
                            "schedule_step": schedule_step,
                            "schedule_weight": weight,
                            "schedule_hash": schedule_hash,
                            "safety_norm": summary["mean_safety_gradient_norm"],
                            "risk_norm": summary["mean_risk_gradient_norm"],
                            "weighted_safety_risk_ratio": summary["mean_contribution_ratio"],
                            "mean_cosine": summary["mean_cosine"],
                            "negative_pair_fraction": summary["negative_pair_fraction"],
                        }
                    )

        del model, tokenizer
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    write_csv(out_dir / "gradient_pairs.csv", pair_rows)
    write_csv(out_dir / "gradient_summary.csv", summary_rows)
    write_csv(out_dir / "normalized_gradient_contributions.csv", normalized_rows)
    integrity = {
        "schema_version": "paper_a_gradient_audit_integrity_v1",
        "config": args.config,
        "family": args.family,
        "seed": args.seed,
        "arm": args.arm,
        "checkpoints": checkpoints,
        "total_steps": total_steps,
        "schedule_sum": sum(vector),
        "schedule_hash": schedule_hash,
        "risk_ids_hash": ids_hash(risk_rows),
        "safety_ids_hash": ids_hash(safety_rows),
        "pair_rows": len(pair_rows),
        "summary_rows": len(summary_rows),
        "normalized_rows": len(normalized_rows),
    }
    (out_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(integrity, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
