"""Helpers for prompt-conditional response mean/worst ranking."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_mismatch_rows(targets: list[dict[str, Any]], minimum_category_prompts: int) -> list[dict[str, Any]]:
    prompt_rows: dict[str, dict[str, Any]] = {}
    category_prompts: dict[str, list[str]] = {}
    for row in targets:
        sample_id = str(row["sample_id"])
        prompt_rows.setdefault(sample_id, row)
    for sample_id, row in prompt_rows.items():
        category_prompts.setdefault(str(row["category"]), []).append(sample_id)
    for sample_ids in category_prompts.values():
        sample_ids.sort(key=lambda value: stable_hash(value))

    output = []
    for source_position, row in enumerate(targets):
        sample_id = str(row["sample_id"])
        category = str(row["category"])
        candidates = [value for value in category_prompts[category] if value != sample_id]
        if len(category_prompts[category]) < minimum_category_prompts:
            continue
        wrong_ids = sorted(
            candidates, key=lambda value: stable_hash(str(row["target_id"]), value)
        )[:2]
        if len(wrong_ids) != 2 or len(set(wrong_ids)) != 2:
            raise ValueError(f"target lacks two distinct mismatches: {row['target_id']}")
        output.append(
            {
                "ranking_id": f"pcr_rank_{len(output) + 1:03d}",
                "source_position": source_position,
                "sample_id": sample_id,
                "target_id": row["target_id"],
                "category": category,
                "correct_prompt": row["prompt"],
                "response": row["response"],
                "wrong_sample_ids": wrong_ids,
                "wrong_prompts": [prompt_rows[value]["prompt"] for value in wrong_ids],
                "correct_prompt_sha256": stable_hash(str(row["prompt"])),
                "response_sha256": stable_hash(str(row["response"])),
                "wrong_prompt_sha256": [
                    stable_hash(str(prompt_rows[value]["prompt"])) for value in wrong_ids
                ],
            }
        )
    return output


def ranking_losses(
    correct_nll: Any,
    wrong_nlls: Any,
    reference_margins: Any,
    worst_temperature: float,
) -> dict[str, Any]:
    import torch

    if wrong_nlls.ndim != 1 or wrong_nlls.numel() != 2:
        raise ValueError("PCR requires exactly two wrong-prompt NLL values")
    if reference_margins.shape != wrong_nlls.shape:
        raise ValueError("reference margins must match wrong-prompt scores")
    if worst_temperature <= 0:
        raise ValueError("worst temperature must be positive")
    current_margins = wrong_nlls - correct_nll
    centered_margins = current_margins - reference_margins
    violations = torch.nn.functional.softplus(-centered_margins)
    mean_loss = violations.mean()
    worst_loss = worst_temperature * (
        torch.logsumexp(violations / worst_temperature, dim=0) - math.log(violations.numel())
    )
    return {
        "mean_loss": mean_loss,
        "worst_loss": worst_loss,
        "mean_worst_loss": mean_loss + worst_loss,
        "current_margins": current_margins,
        "centered_margins": centered_margins,
        "violations": violations,
    }


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
