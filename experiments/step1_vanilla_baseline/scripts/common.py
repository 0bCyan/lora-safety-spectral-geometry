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


def load_config(path: str | Path) -> dict[str, Any]:
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


def experiment_id(config: dict[str, Any], family: str, seed: int) -> str:
    model = config["model_name_or_path"].split("/")[-1].lower()
    return f"step1_{config['machine']}_{model}_{family}_s{seed}"


def adapter_dir(config: dict[str, Any], family: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "adapters" / experiment_id(config, family, seed)


def log_dir(config: dict[str, Any], family: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "logs" / experiment_id(config, family, seed)


def metrics_path(config: dict[str, Any], family: str, seed: int) -> Path:
    return (
        Path(config["output_dir"])
        / "spectral_metrics"
        / f"{experiment_id(config, family, seed)}.jsonl"
    )


def generation_path(
    config: dict[str, Any], family: str, seed: int, eval_set: str
) -> Path:
    return (
        Path(config["output_dir"])
        / "generations"
        / f"{experiment_id(config, family, seed)}_{eval_set}.jsonl"
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
