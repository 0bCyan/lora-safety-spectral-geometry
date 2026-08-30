from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_LABELS = {"unsafe_compliance", "mixed", "clean_refusal", "safe_redirect", "invalid"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic arm-blind Paper A safety review packets.")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--reviewers", nargs="+", default=["reviewer_a", "reviewer_b"])
    parser.add_argument("--duplicates", type=int, default=18)
    args = parser.parse_args()

    targets = read_csv(args.targets)
    required = {"sample_id", "prompt", "response"}
    if not targets or not required.issubset(targets[0]):
        raise ValueError("Targets must contain sample_id,prompt,response")
    if len({row["sample_id"] for row in targets}) != len(targets):
        raise ValueError("Target sample_id values must be unique")

    calibration: list[dict[str, str]] = []
    if args.calibration:
        calibration = read_csv(args.calibration)
        calibration_required = {"sample_id", "prompt", "response", "behavior_label"}
        if not calibration or not calibration_required.issubset(calibration[0]):
            raise ValueError("Calibration must contain sample_id,prompt,response,behavior_label")
        invalid = {row["behavior_label"] for row in calibration} - ALLOWED_LABELS
        if invalid:
            raise ValueError(f"Invalid calibration labels: {sorted(invalid)}")

    duplicate_sources = sorted(
        targets,
        key=lambda row: stable_hash(args.seed, "duplicate", row["sample_id"]),
    )[: min(args.duplicates, len(targets))]
    hidden_fields = sorted(set(targets[0]) - {"prompt", "response"})
    manifests = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for reviewer in args.reviewers:
        packet_rows: list[dict[str, str]] = []
        map_rows: list[dict[str, Any]] = []

        def add(source: dict[str, str], row_type: str, duplicate_of: str | None = None) -> None:
            source_id = source["sample_id"]
            occurrence = sum(1 for row in map_rows if row["source_id"] == source_id)
            review_id = f"pas_{stable_hash(args.seed, reviewer, row_type, source_id, occurrence)[:24]}"
            packet_rows.append({"review_id": review_id, "prompt": source["prompt"], "response": source["response"]})
            map_rows.append(
                {
                    "review_id": review_id,
                    "source_id": source_id,
                    "row_type": row_type,
                    "duplicate_of": duplicate_of,
                    "hidden": {field: source.get(field, "") for field in hidden_fields},
                    "gold_behavior_label": source.get("behavior_label") if row_type == "calibration" else None,
                    "adjudication_payload": {"prompt": source["prompt"], "response": source["response"]},
                }
            )

        for row in targets:
            add(row, "primary")
        for row in calibration:
            add(row, "calibration")
        for row in duplicate_sources:
            add(row, "duplicate", duplicate_of=row["sample_id"])

        order = sorted(
            range(len(packet_rows)),
            key=lambda index: stable_hash(args.seed, reviewer, "order", packet_rows[index]["review_id"]),
        )
        packet_rows = [packet_rows[index] for index in order]
        map_rows = [map_rows[index] for index in order]
        packet_path = args.output_dir / f"{reviewer}_packet.csv"
        map_path = args.output_dir / f"{reviewer}_hidden_map.json"
        write_csv(packet_path, packet_rows, ["review_id", "prompt", "response"])
        map_path.write_text(
            json.dumps(
                {"schema_version": "paper_a_safety_hidden_map_v1", "reviewer": reviewer, "seed": args.seed, "rows": map_rows},
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        manifests.append(
            {
                "reviewer": reviewer,
                "packet": str(packet_path),
                "packet_sha256": sha256(packet_path),
                "hidden_map": str(map_path),
                "hidden_map_sha256": sha256(map_path),
                "primary_rows": len(targets),
                "calibration_rows": len(calibration),
                "duplicate_rows": len(duplicate_sources),
            }
        )

    (args.output_dir / "packet_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper_a_safety_packet_manifest_v1",
                "target_source": str(args.targets),
                "target_source_sha256": sha256(args.targets),
                "seed": args.seed,
                "reviewers": manifests,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifests)} safety reviewer packets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
