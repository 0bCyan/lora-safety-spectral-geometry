from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("experiments/step1_vanilla_baseline")
GENERATION_DIR = ROOT / "results/4070ti/generations"
JUDGE_DIR = ROOT / "results/4070ti/local_judge"
TABLE_DIR = ROOT / "analysis/tables"


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
NUMERIC_STUB_RE = re.compile(
    r"^(?:the\s+answer\s+is\s*:?\s*)?(?:[-+]?\d+(?:\.\d+)?|yes|no|true|false)\.?$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", default=str(GENERATION_DIR))
    parser.add_argument("--judge-dir", default=str(JUDGE_DIR))
    parser.add_argument("--output-dir", default=str(TABLE_DIR))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_judge_rows(judge_dir: Path) -> dict[tuple[str, int, str, str], dict]:
    by_key = {}
    for path in sorted(judge_dir.glob("*_local_judge.jsonl")):
        for row in read_jsonl(path):
            key = (
                str(row.get("train_family")),
                int(row.get("seed", 0)),
                str(row.get("eval_set")),
                str(row.get("sample_id")),
            )
            by_key[key] = row
    return by_key


def clean_response(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def experiment_id(family: str, seed: int, eval_set: str) -> str:
    return f"step1_4070ti_qwen3-4b_{family}_s{seed}_{eval_set}"


def flag_degradation(cleaned: str, frequency: int, n: int) -> tuple[int, str]:
    if not cleaned:
        return 1, "empty_after_think_cleaning"
    if NUMERIC_STUB_RE.match(cleaned):
        return 1, "numeric_or_boolean_stub"
    if len(cleaned) < 80 and n and frequency / n >= 0.5:
        return 1, "repeated_short_stub"
    return 0, ""


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    generation_dir = Path(args.generation_dir)
    judge_dir = Path(args.judge_dir)
    output_dir = Path(args.output_dir)

    judge_by_key = read_judge_rows(judge_dir)
    sample_rows = []
    grouped = defaultdict(list)

    for path in sorted(generation_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        cleaned_by_sample = {
            str(row.get("sample_id")): clean_response(str(row.get("response", "")))
            for row in rows
        }
        frequencies = Counter(cleaned_by_sample.values())
        n = len(rows)

        for row in rows:
            family = str(row.get("train_family"))
            seed = int(row.get("seed", 0))
            eval_set = str(row.get("eval_set"))
            sample_id = str(row.get("sample_id"))
            cleaned = cleaned_by_sample[sample_id]
            degraded, reason = flag_degradation(cleaned, frequencies[cleaned], n)
            judge = judge_by_key.get((family, seed, eval_set, sample_id), {})
            item = {
                "experiment_id": experiment_id(family, seed, eval_set),
                "train_family": family,
                "seed": seed,
                "eval_set": eval_set,
                "sample_id": sample_id,
                "response_chars": len(str(row.get("response", ""))),
                "cleaned_response_chars": len(cleaned),
                "top_cleaned_response_count": frequencies[cleaned],
                "degradation_flag": degraded,
                "degradation_reason": reason,
                "local_label": judge.get("label", ""),
                "local_reason": judge.get("local_reason", ""),
                "cleaned_response_preview": cleaned[:120],
            }
            sample_rows.append(item)
            grouped[(family, seed, eval_set)].append(item)

    summary_rows = []
    for (family, seed, eval_set), rows in sorted(grouped.items()):
        degraded = sum(int(row["degradation_flag"]) for row in rows)
        empty = sum(row["degradation_reason"] == "empty_after_think_cleaning" for row in rows)
        numeric_stub = sum(row["degradation_reason"] == "numeric_or_boolean_stub" for row in rows)
        repeated_short_stub = sum(row["degradation_reason"] == "repeated_short_stub" for row in rows)
        invalid = sum(row["local_label"] == "invalid_or_irrelevant" for row in rows)
        response_counts = Counter(row["cleaned_response_preview"] for row in rows)
        top_preview, top_count = response_counts.most_common(1)[0]
        summary_rows.append(
            {
                "experiment_id": experiment_id(family, seed, eval_set),
                "train_family": family,
                "seed": seed,
                "eval_set": eval_set,
                "n": len(rows),
                "degradation_count": degraded,
                "degradation_rate": degraded / len(rows) if rows else 0.0,
                "empty_after_think_cleaning": empty,
                "numeric_or_boolean_stub": numeric_stub,
                "repeated_short_stub": repeated_short_stub,
                "local_invalid_or_irrelevant": invalid,
                "top_cleaned_response_count": top_count,
                "top_cleaned_response_preview": top_preview,
            }
        )

    sample_fields = [
        "experiment_id",
        "train_family",
        "seed",
        "eval_set",
        "sample_id",
        "response_chars",
        "cleaned_response_chars",
        "top_cleaned_response_count",
        "degradation_flag",
        "degradation_reason",
        "local_label",
        "local_reason",
        "cleaned_response_preview",
    ]
    summary_fields = [
        "experiment_id",
        "train_family",
        "seed",
        "eval_set",
        "n",
        "degradation_count",
        "degradation_rate",
        "empty_after_think_cleaning",
        "numeric_or_boolean_stub",
        "repeated_short_stub",
        "local_invalid_or_irrelevant",
        "top_cleaned_response_count",
        "top_cleaned_response_preview",
    ]

    write_csv(output_dir / "behavior_flags.csv", sample_rows, sample_fields)
    write_csv(output_dir / "behavior_summary.csv", summary_rows, summary_fields)
    print(f"sample_rows: {len(sample_rows)}")
    print(f"summary_rows: {len(summary_rows)}")
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
