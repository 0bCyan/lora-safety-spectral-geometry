#!/usr/bin/env python3
"""Verify normalized mechanism-v1 JSONL files and detect split leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = (
    "train_risk.jsonl",
    "train_safety.jsonl",
    "train_utility.jsonl",
    "eval_beavertails.jsonl",
    "eval_sorry_bench.jsonl",
    "eval_harmbench.jsonl",
)


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("sample_id", "family", "source", "prompt"):
                if not row.get(key):
                    raise ValueError(f"{path}:{line_no} missing {key}")
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    args = ap.parse_args()
    tables = {}
    for name in EXPECTED:
        path = args.dataset_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        tables[name] = load(path)
        prompts = [row["prompt"] for row in tables[name]]
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"duplicate prompts in {name}")

    train_prompts = set().union(*(set(row["prompt"] for row in tables[n]) for n in EXPECTED[:3]))
    for name in EXPECTED[3:]:
        overlap = train_prompts & {row["prompt"] for row in tables[name]}
        if overlap:
            raise ValueError(f"train/eval prompt leakage: {name} has {len(overlap)} overlaps")

    counts = {name: len(rows) for name, rows in tables.items()}
    print(json.dumps({"status": "ok", "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
