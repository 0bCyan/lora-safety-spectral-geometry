from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FAMILY_TO_FILE = {
    "normal": "train_normal.jsonl",
    "risk": "train_risk.jsonl",
    "control": "train_control.jsonl",
}

EVAL_TO_FILE = {
    "safety": "eval_safety.jsonl",
    "utility": "eval_utility.jsonl",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def model_tag(config: dict[str, Any]) -> str:
    return config["model_name_or_path"].split("/")[-1].lower()


def run_id(config: dict[str, Any], family: str, seed: int) -> str:
    return f"step2a_{config['machine']}_{model_tag(config)}_{family}_s{seed}"


def checkpoint_tag(step: int, total_steps: int) -> str:
    percent = 0 if total_steps <= 0 else round(100 * step / total_steps)
    return f"step{step:04d}_pct{percent:03d}"


def run_dir(config: dict[str, Any], family: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "runs" / run_id(config, family, seed)


def checkpoint_dir(config: dict[str, Any], family: str, seed: int, checkpoint: str) -> Path:
    return run_dir(config, family, seed) / "checkpoints" / checkpoint


def generation_path(
    config: dict[str, Any],
    family: str,
    seed: int,
    checkpoint: str,
    eval_set: str,
) -> Path:
    return (
        Path(config["output_dir"])
        / "generations"
        / f"{run_id(config, family, seed)}_{checkpoint}_{eval_set}.jsonl"
    )


def local_judge_path(
    config: dict[str, Any],
    family: str,
    seed: int,
    checkpoint: str,
    eval_set: str,
) -> Path:
    return (
        Path(config["output_dir"])
        / "local_judge"
        / f"{run_id(config, family, seed)}_{checkpoint}_{eval_set}_local_judge.jsonl"
    )


def spectral_path(config: dict[str, Any], family: str, seed: int, checkpoint: str) -> Path:
    return (
        Path(config["output_dir"])
        / "spectral_metrics"
        / f"{run_id(config, family, seed)}_{checkpoint}.jsonl"
    )


def logit_margin_path(config: dict[str, Any], family: str, seed: int, checkpoint: str) -> Path:
    return (
        Path(config["output_dir"])
        / "logit_margins"
        / f"{run_id(config, family, seed)}_{checkpoint}_logit_margins.jsonl"
    )
