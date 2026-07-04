from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INTERVENTION_SCRIPTS = SCRIPT_DIR.parents[1] / "interventions" / "scripts"
sys.path.insert(0, str(INTERVENTION_SCRIPTS))

from activation_common import load_json, write_jsonl  # noqa: E402
from intervention_common import clean_response, classify_response, degradation_reason, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    root = Path(args.input_dir) if args.input_dir else Path(config["output_dir"]) / "interventions"
    paths = sorted(root.glob("*/generations/*.jsonl"))
    all_rows = []
    for path in paths:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    import json

                    rows.append(json.loads(line))
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
                "direction_name": row.get("direction_name"),
                "activation_mode": row.get("activation_mode"),
                "alpha": row.get("alpha"),
                "layers": ",".join(str(layer) for layer in row.get("layers", [])),
                "label": label,
                "local_reason": reason,
                "degradation_reason": degradation,
                "response_chars": len(row.get("response", "")),
                "cleaned_response_chars": len(cleaned_text),
            }
            labeled.append(item)
            all_rows.append(item)
        out_dir = path.parents[1] / "local_judge"
        write_jsonl(out_dir / path.name.replace(".jsonl", "_local_judge.jsonl"), labeled)

    grouped = {}
    for row in all_rows:
        key = (
            row["train_family"],
            row["seed"],
            row["eval_set"],
            row["intervention"],
            row["direction_name"],
            row["activation_mode"],
            row["alpha"],
            row["layers"],
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, group_rows in sorted(grouped.items()):
        total = len(group_rows)
        refusal = sum(1 for row in group_rows if row["label"] == "refusal")
        unsafe = sum(1 for row in group_rows if row["label"] == "unsafe_compliance")
        invalid = sum(1 for row in group_rows if row["label"] == "invalid_or_irrelevant")
        degradation = sum(1 for row in group_rows if row["degradation_reason"])
        response_chars = sum(int(row["response_chars"]) for row in group_rows) / total
        cleaned_chars = sum(int(row["cleaned_response_chars"]) for row in group_rows) / total
        (
            family,
            seed,
            eval_set,
            intervention,
            direction_name,
            activation_mode,
            alpha,
            layers,
        ) = key
        summary.append(
            {
                "train_family": family,
                "seed": seed,
                "eval_set": eval_set,
                "intervention": intervention,
                "direction_name": direction_name,
                "activation_mode": activation_mode,
                "alpha": alpha,
                "layers": layers,
                "n": total,
                "refusal": refusal,
                "unsafe_compliance": unsafe,
                "invalid_or_irrelevant": invalid,
                "degradation_count": degradation,
                "refusal_rate": refusal / total if total else 0.0,
                "unsafe_compliance_rate": unsafe / total if total else 0.0,
                "invalid_rate": invalid / total if total else 0.0,
                "degradation_rate": degradation / total if total else 0.0,
                "average_response_chars": response_chars,
                "average_cleaned_response_chars": cleaned_chars,
            }
        )
    write_csv(root / "local_judge_summary.csv", summary)
    print(f"files: {len(paths)}")
    print(f"rows: {len(all_rows)}")
    print(f"wrote: {root / 'local_judge_summary.csv'}")


if __name__ == "__main__":
    main()
