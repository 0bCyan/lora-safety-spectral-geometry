"""Shared deterministic I/O and packet helpers for Step3C-T4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[3]
THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG = re.compile(r"</?think>", flags=re.IGNORECASE)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def stable_rank(seed: int, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


def selected_sample_id_hash(rows: list[dict[str, Any]]) -> str:
    return canonical_hash([str(row["sample_id"]) for row in rows])


def clean_thinking_output(text: str) -> tuple[str, dict[str, Any]]:
    raw_hash = sha256_text(text)
    complete_blocks = len(THINK_BLOCK.findall(text))
    cleaned = THINK_BLOCK.sub("", text)
    unmatched_open_removed = False
    open_match = re.search(r"<think>", cleaned, flags=re.IGNORECASE)
    if open_match:
        cleaned = cleaned[: open_match.start()]
        unmatched_open_removed = True
    residual_tags = len(THINK_TAG.findall(cleaned))
    cleaned = THINK_TAG.sub("", cleaned).strip()
    return cleaned, {
        "raw_response_sha256": raw_hash,
        "complete_thinking_blocks_removed": complete_blocks,
        "unmatched_open_thinking_removed_to_end": unmatched_open_removed,
        "residual_thinking_tags_removed": residual_tags,
        "removed_character_count": len(text) - len(cleaned),
        "thinking_content_persisted": False,
    }


def duplicate_count(primary_rows: int, fraction: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, math.ceil(primary_rows * fraction)))


def opaque_id(domain: str, seed: int, source_key: str, instance: str) -> str:
    token = stable_rank(seed, domain, source_key, instance)[:20]
    prefix = "quality" if domain == "quality" else domain
    return f"step3ct_{prefix}_{token}"


def balanced_duplicate_sources(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    group_fields: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    if not groups or count % len(groups):
        raise ValueError("duplicate count must divide evenly across registered groups")
    per_group = count // len(groups)
    selected = []
    for key, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda row: stable_rank(seed, *key, str(row["source_key"])))
        if len(ranked) < per_group:
            raise ValueError(f"insufficient rows in duplicate group {key}")
        selected.extend(ranked[:per_group])
    return selected


def shuffled(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result
