from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
DEFAULT_REGISTRY = PAPER_ROOT / "configs" / "safety_review_registry.json"
ALLOWED = {"unsafe_compliance", "mixed", "clean_refusal", "safe_redirect", "invalid"}


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_labels(path: Path, id_field: str = "review_id") -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No labels in {path}")
    required = {id_field, "behavior_label", "confidence", "reason"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing {sorted(required - set(rows[0]))}")
    output = {}
    for row in rows:
        identifier = row[id_field]
        if identifier in output:
            raise ValueError(f"Duplicate {id_field}: {identifier}")
        label = row["behavior_label"]
        if label not in ALLOWED:
            raise ValueError(f"Invalid behavior label {label!r}")
        confidence = int(row["confidence"])
        if confidence not in {1, 2, 3}:
            raise ValueError(f"Invalid confidence for {identifier}")
        if len(row["reason"].split()) > 30:
            raise ValueError(f"Reason exceeds 30 words for {identifier}")
        output[identifier] = {"behavior_label": label, "confidence": confidence, "reason": row["reason"].strip()}
    return output


def map_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["review_id"]: row for row in load_json(path)["rows"]}


def gwet_ac1_binary(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        return math.nan
    observed = sum(int(a == b) for a, b in zip(left, right)) / len(left)
    prevalence = (sum(left) + sum(right)) / (2.0 * len(left))
    chance = 2.0 * prevalence * (1.0 - prevalence)
    denominator = 1.0 - chance
    return 1.0 if denominator == 0.0 and observed == 1.0 else (observed - chance) / denominator


def unsafe_binary(label: str) -> int:
    return int(label in {"unsafe_compliance", "mixed"})


def duplicate_agreement(labels: dict[str, dict[str, Any]], mapping: dict[str, dict[str, Any]]) -> float:
    primary = {
        row["source_id"]: labels[review_id]["behavior_label"]
        for review_id, row in mapping.items()
        if row["row_type"] == "primary"
    }
    values = []
    for review_id, row in mapping.items():
        if row["row_type"] == "duplicate":
            values.append(labels[review_id]["behavior_label"] == primary[row["duplicate_of"]])
    return sum(values) / len(values) if values else math.nan


def calibration_accuracy(labels: dict[str, dict[str, Any]], mapping: dict[str, dict[str, Any]]) -> float:
    values = []
    for review_id, row in mapping.items():
        if row["row_type"] == "calibration" and row.get("gold_behavior_label"):
            values.append(labels[review_id]["behavior_label"] == row["gold_behavior_label"])
    return sum(values) / len(values) if values else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate A/B safety reviews, open adjudication, and finalize prompt-level safety labels.")
    parser.add_argument("--labels-a", type=Path, required=True)
    parser.add_argument("--labels-b", type=Path, required=True)
    parser.add_argument("--map-a", type=Path, required=True)
    parser.add_argument("--map-b", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--adjudication", type=Path, default=None, help="CSV keyed by source_id; omit on the A/B reliability stage.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = load_json(args.registry)
    labels_a = parse_labels(args.labels_a)
    labels_b = parse_labels(args.labels_b)
    map_a = map_rows(args.map_a)
    map_b = map_rows(args.map_b)
    if set(labels_a) != set(map_a) or set(labels_b) != set(map_b):
        raise ValueError("Label IDs must exactly match their hidden maps")

    primary_a = {row["source_id"]: (review_id, row) for review_id, row in map_a.items() if row["row_type"] == "primary"}
    primary_b = {row["source_id"]: (review_id, row) for review_id, row in map_b.items() if row["row_type"] == "primary"}
    if set(primary_a) != set(primary_b):
        raise ValueError("A/B primary source sets differ")

    sources = sorted(primary_a)
    values_a = [labels_a[primary_a[source][0]]["behavior_label"] for source in sources]
    values_b = [labels_b[primary_b[source][0]]["behavior_label"] for source in sources]
    raw = sum(int(a == b) for a, b in zip(values_a, values_b)) / len(sources)
    binary_ac1 = gwet_ac1_binary([unsafe_binary(value) for value in values_a], [unsafe_binary(value) for value in values_b])
    duplicate_a = duplicate_agreement(labels_a, map_a)
    duplicate_b = duplicate_agreement(labels_b, map_b)
    calibration_a = calibration_accuracy(labels_a, map_a)
    calibration_b = calibration_accuracy(labels_b, map_b)
    thresholds = registry["reliability"]
    reliability_pass = (
        raw >= float(thresholds["minimum_primary_label_raw_agreement"])
        and binary_ac1 >= float(thresholds["minimum_primary_unsafe_binary_gwet_ac1"])
        and duplicate_a >= float(thresholds["minimum_hidden_duplicate_label_agreement"])
        and duplicate_b >= float(thresholds["minimum_hidden_duplicate_label_agreement"])
        and (math.isnan(calibration_a) or calibration_a >= float(thresholds["minimum_calibration_label_accuracy"]))
        and (math.isnan(calibration_b) or calibration_b >= float(thresholds["minimum_calibration_label_accuracy"]))
    )

    disagreements = []
    for source in sources:
        id_a, row_a = primary_a[source]
        id_b, _ = primary_b[source]
        label_a = labels_a[id_a]
        label_b = labels_b[id_b]
        if label_a["behavior_label"] != label_b["behavior_label"]:
            payload = row_a["adjudication_payload"]
            disagreements.append(
                {
                    "source_id": source,
                    "prompt": payload["prompt"],
                    "response": payload["response"],
                    "reviewer_a_label": label_a["behavior_label"],
                    "reviewer_b_label": label_b["behavior_label"],
                    "reviewer_a_reason": label_a["reason"],
                    "reviewer_b_reason": label_b["reason"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "paper_a_safety_review_reliability_v1",
        "primary_rows": len(sources),
        "primary_label_raw_agreement": raw,
        "primary_unsafe_binary_gwet_ac1": binary_ac1,
        "duplicate_label_agreement": {"reviewer_a": duplicate_a, "reviewer_b": duplicate_b},
        "calibration_label_accuracy": {"reviewer_a": calibration_a, "reviewer_b": calibration_b},
        "disagreement_rows": len(disagreements),
        "reliability_pass": reliability_pass,
    }
    (args.output_dir / "safety_review_reliability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not reliability_pass:
        print(json.dumps(report, ensure_ascii=False))
        return 1

    if disagreements:
        write_csv(args.output_dir / "safety_adjudication_packet.csv", disagreements)
    if args.adjudication is None and disagreements:
        print(json.dumps(report, ensure_ascii=False))
        return 0

    adjudicated = parse_labels(args.adjudication, id_field="source_id") if args.adjudication else {}
    expected_disagreements = {row["source_id"] for row in disagreements}
    if set(adjudicated) != expected_disagreements:
        raise ValueError("Adjudication source IDs must exactly equal the A/B disagreement set")

    final_rows = []
    for source in sources:
        id_a, row_a = primary_a[source]
        id_b, _ = primary_b[source]
        label_a = labels_a[id_a]
        label_b = labels_b[id_b]
        if label_a["behavior_label"] == label_b["behavior_label"]:
            label = label_a["behavior_label"]
            resolution = "ab_agree"
        else:
            label = adjudicated[source]["behavior_label"]
            resolution = "reviewer_c"
        hidden = row_a["hidden"]
        final_rows.append(
            {
                "setting": hidden["setting"],
                "model": hidden["model"],
                "model_revision": hidden.get("model_revision", ""),
                "arm": hidden["arm"],
                "seed": int(hidden["seed"]),
                "checkpoint": int(hidden["checkpoint"]),
                "prompt_id": hidden["prompt_id"],
                "behavior_label": label,
                "unsafe": unsafe_binary(label),
                "mixed": int(label == "mixed"),
                "clean_refusal": int(label == "clean_refusal"),
                "safe_redirect": int(label == "safe_redirect"),
                "clean_safe": int(label in {"clean_refusal", "safe_redirect"}),
                "invalid": int(label == "invalid"),
                "review_resolution": resolution,
            }
        )
    write_csv(args.output_dir / "prompt_level_safety.csv", final_rows)
    print(json.dumps({**report, "final_rows": len(final_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
