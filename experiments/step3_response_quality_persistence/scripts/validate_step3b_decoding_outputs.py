from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = read_json(config_path)
    output_dir = Path(config["output_dir"])
    expected_reps = int(config["samples_per_prompt"])
    expected_prompts = int(config["eval_limit"])

    with Path(config["split_manifest"]).open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    split_ids = defaultdict(set)
    for row in split_rows:
        split_ids[row["split"]].add(row["sample_id"])

    condition_summaries = []
    evidence = []
    reference_ids: set[str] | None = None
    all_seeds = set()
    for condition in config["conditions"]:
        name = condition["name"]
        path = output_dir / "generations" / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        expected_rows = expected_prompts * expected_reps
        if len(rows) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, found {len(rows)}")
        generation_ids = [row["generation_id"] for row in rows]
        seeds = [int(row["generation_seed"]) for row in rows]
        if len(generation_ids) != len(set(generation_ids)):
            raise ValueError(f"{name}: duplicate generation_id")
        if len(seeds) != len(set(seeds)) or all_seeds.intersection(seeds):
            raise ValueError(f"{name}: duplicate generation seed")
        all_seeds.update(seeds)

        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["condition"] != name:
                raise ValueError(f"{name}: wrong condition field")
            by_prompt[row["sample_id"]].append(row)
        if len(by_prompt) != expected_prompts:
            raise ValueError(f"{name}: expected {expected_prompts} prompts, found {len(by_prompt)}")
        for sample_id, prompt_rows in by_prompt.items():
            reps = sorted(int(row["replicate"]) for row in prompt_rows)
            if reps != list(range(expected_reps)):
                raise ValueError(f"{name}/{sample_id}: replicate set {reps}")

        ids = set(by_prompt)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{name}: prompt IDs differ across conditions")
        overlaps = {split: len(ids & values) for split, values in split_ids.items()}
        if any(overlaps.values()):
            raise ValueError(f"{name}: overlaps Step2G split IDs {overlaps}")

        unique_counts = []
        empty_count = 0
        max_token_limit_hit_count = 0
        for prompt_rows in by_prompt.values():
            cleaned = [clean_response(str(row["response"])) for row in prompt_rows]
            unique_counts.append(len(set(cleaned)))
            empty_count += sum(not response for response in cleaned)
            max_token_limit_hit_count += sum(
                int(row["generated_tokens"]) >= int(config["max_new_tokens"])
                for row in prompt_rows
            )
        condition_summaries.append(
            {
                "condition": name,
                "rows": len(rows),
                "prompts": len(by_prompt),
                "samples_per_prompt": expected_reps,
                "mean_exact_unique_per_prompt": sum(unique_counts) / len(unique_counts),
                "all_identical_prompt_count": sum(count == 1 for count in unique_counts),
                "all_identical_prompt_rate": sum(count == 1 for count in unique_counts) / len(unique_counts),
                "empty_response_count": empty_count,
                "max_token_limit_hit_count": max_token_limit_hit_count,
                "safety_label_status": "pending_independent_review",
                "conditional_mechanical_status": "pending_after_safety_freeze",
            }
        )
        evidence.append(
            {"condition": name, "path": str(path), "rows": len(rows), "sha256": sha256(path)}
        )

    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = reports / "step3b_surface_diversity_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_summaries[0]))
        writer.writeheader()
        writer.writerows(condition_summaries)
    integrity = {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "conditions": condition_summaries,
        "evidence": evidence,
        "all_generation_seeds_unique": True,
        "prompt_ids_aligned": True,
        "Step2G_split_overlap_zero": True,
        "heldout_content_opened": False,
        "new_training_run": False,
        "route_decision_status": "pending independent safety and clean-safe mechanical review",
    }
    integrity_path = reports / "step3b_decoding_integrity.json"
    integrity_path.write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {summary_path}")
    print(f"wrote {integrity_path}")


if __name__ == "__main__":
    main()
