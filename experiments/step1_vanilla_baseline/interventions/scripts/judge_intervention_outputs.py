from __future__ import annotations

import argparse
from pathlib import Path

from intervention_common import (
    clean_response,
    classify_response,
    degradation_reason,
    read_jsonl,
    summarize_labeled,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="experiments/step1_vanilla_baseline/interventions/results/4070ti/generations",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/step1_vanilla_baseline/interventions/results/4070ti/local_judge",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    all_rows = []
    for path in sorted(input_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        cleaned = [clean_response(row.get("response", "")) for row in rows]
        frequencies = {text: cleaned.count(text) for text in set(cleaned)}
        labeled = []
        for row, cleaned_text in zip(rows, cleaned):
            label, reason = classify_response(row.get("response", ""))
            degradation = degradation_reason(cleaned_text, frequencies[cleaned_text], len(rows))
            item = {
                "sample_id": row.get("sample_id"),
                "train_family": row.get("train_family"),
                "seed": row.get("seed"),
                "eval_set": row.get("eval_set"),
                "intervention": row.get("intervention"),
                "module_filter": row.get("module_filter", ""),
                "k": row.get("k"),
                "scale": row.get("scale"),
                "label": label,
                "local_reason": reason,
                "degradation_reason": degradation,
                "response_chars": len(row.get("response", "")),
                "cleaned_response_chars": len(cleaned_text),
            }
            labeled.append(item)
            all_rows.append(item)
        write_jsonl(output_dir / path.name.replace(".jsonl", "_local_judge.jsonl"), labeled)

    summary = summarize_labeled(all_rows)
    write_csv(output_dir / "summary.csv", summary)
    print(f"files: {len(list(input_dir.glob('*.jsonl')))}")
    print(f"rows: {len(all_rows)}")
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
