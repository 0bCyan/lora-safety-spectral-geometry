from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
sys.path.insert(0, str(PAPER_ROOT / "scripts"))
import validate_review_labels as review  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_adjudication(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_csv(path)
    required = {"source_id", *review.COMPONENTS, "confidence", "reason"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Adjudication is missing fields: {sorted(required - set(rows[0]) if rows else required)}")
    output = {}
    for raw in rows:
        source = raw["source_id"]
        if source in output:
            raise ValueError(f"Duplicate adjudication source_id {source}")
        row = {component: int(raw[component]) for component in review.COMPONENTS}
        if any(value not in {0, 1} for value in row.values()):
            raise ValueError(f"Non-binary adjudication component for {source}")
        confidence = int(raw["confidence"])
        if confidence not in {1, 2, 3}:
            raise ValueError(f"Invalid adjudication confidence for {source}")
        row["confidence"] = confidence
        row["reason"] = raw["reason"].strip()
        row["derived_quality_score"] = review.derive_quality_score(row)
        output[source] = row
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Join reliable A/B/C quality labels to hidden arm keys.")
    parser.add_argument("--labels-a", type=Path, required=True)
    parser.add_argument("--labels-b", type=Path, required=True)
    parser.add_argument("--map-a", type=Path, required=True)
    parser.add_argument("--map-b", type=Path, required=True)
    parser.add_argument("--reliability-report", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.reliability_report.read_text(encoding="utf-8"))
    if report.get("reliability_pass") is not True:
        raise RuntimeError("A/B quality reliability did not pass; arm-key join is forbidden")

    labels_a = review.parse_labels(args.labels_a)
    labels_b = review.parse_labels(args.labels_b)
    map_a = review.map_by_review_id(args.map_a)
    map_b = review.map_by_review_id(args.map_b)
    if set(labels_a) != set(map_a) or set(labels_b) != set(map_b):
        raise ValueError("Label IDs must exactly match their hidden maps")

    primary_a = {row["source_id"]: (review_id, row) for review_id, row in map_a.items() if row["row_type"] == "primary"}
    primary_b = {row["source_id"]: (review_id, row) for review_id, row in map_b.items() if row["row_type"] == "primary"}
    if set(primary_a) != set(primary_b):
        raise ValueError("A/B primary source sets differ")

    disagreements = {
        source
        for source in primary_a
        if review.full_vector(labels_a[primary_a[source][0]]) != review.full_vector(labels_b[primary_b[source][0]])
    }
    adjudicated = parse_adjudication(args.adjudication)
    if set(adjudicated) != disagreements:
        raise ValueError("Adjudication source IDs must exactly equal the A/B quality disagreement set")

    rows = []
    for source in sorted(primary_a):
        id_a, map_row = primary_a[source]
        id_b, _ = primary_b[source]
        label_a = labels_a[id_a]
        label_b = labels_b[id_b]
        if source in disagreements:
            label = adjudicated[source]
            resolution = "reviewer_c"
        else:
            label = label_a
            resolution = "ab_agree"
        hidden = map_row["hidden"]
        row = {
            "setting": hidden["setting"],
            "model": hidden["model"],
            "model_revision": hidden.get("model_revision", ""),
            "arm": hidden["arm"],
            "seed": int(hidden["seed"]),
            "checkpoint": int(hidden["checkpoint"]),
            "prompt_id": hidden["prompt_id"],
        }
        row.update({component: int(label[component]) for component in review.COMPONENTS})
        row["derived_quality_score"] = int(label["derived_quality_score"])
        row["review_resolution"] = resolution
        rows.append(row)

    write_csv(args.output, rows)
    print(json.dumps({"rows": len(rows), "reviewer_c_rows": len(disagreements)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
