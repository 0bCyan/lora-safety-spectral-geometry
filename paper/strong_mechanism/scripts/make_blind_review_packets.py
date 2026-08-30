from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


COMPONENTS = [
    "safety_correctness",
    "prompt_binding",
    "safe_actionability",
    "detail_retention",
    "naturalness",
    "generic_only_refusal",
    "invalid_output",
]


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


def require(rows: list[dict[str, str]], fields: set[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} has no rows")
    missing = fields - set(rows[0])
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic arm-blind Paper A quality review packets.")
    parser.add_argument("--targets", type=Path, required=True, help="CSV with sample_id,prompt,response and hidden arm metadata.")
    parser.add_argument("--calibration", type=Path, default=None, help="Optional CSV with gold component columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--reviewers", nargs="+", default=["reviewer_a", "reviewer_b"])
    parser.add_argument("--duplicates", type=int, default=12)
    args = parser.parse_args()

    targets = read_csv(args.targets)
    require(targets, {"sample_id", "prompt", "response"}, "targets")
    if len({row["sample_id"] for row in targets}) != len(targets):
        raise ValueError("target sample_id values must be unique")

    calibration: list[dict[str, str]] = []
    if args.calibration:
        calibration = read_csv(args.calibration)
        require(calibration, {"sample_id", "prompt", "response", *COMPONENTS}, "calibration")

    duplicate_sources = sorted(
        targets,
        key=lambda row: stable_hash(args.seed, "duplicate", row["sample_id"]),
    )[: min(args.duplicates, len(targets))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    hidden_fields = sorted(set(targets[0]) - {"prompt", "response"})

    for reviewer in args.reviewers:
        packet_rows: list[dict[str, str]] = []
        map_rows: list[dict[str, Any]] = []

        def add_row(source: dict[str, str], row_type: str, duplicate_of: str | None = None) -> None:
            source_id = source["sample_id"]
            occurrence = sum(1 for row in map_rows if row["source_id"] == source_id)
            review_id = f"pa_{stable_hash(args.seed, reviewer, row_type, source_id, occurrence)[:24]}"
            packet_rows.append({"review_id": review_id, "prompt": source["prompt"], "response": source["response"]})
            hidden = {field: source.get(field, "") for field in hidden_fields}
            gold = {component: int(source[component]) for component in COMPONENTS} if row_type == "calibration" else None
            map_rows.append(
                {
                    "review_id": review_id,
                    "source_id": source_id,
                    "row_type": row_type,
                    "duplicate_of": duplicate_of,
                    "hidden": hidden,
                    "gold": gold,
                    "adjudication_payload": {"prompt": source["prompt"], "response": source["response"]},
                }
            )

        for row in targets:
            add_row(row, "primary")
        for row in calibration:
            add_row(row, "calibration")
        for row in duplicate_sources:
            add_row(row, "duplicate", duplicate_of=row["sample_id"])

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
                {
                    "schema_version": "paper_a_review_hidden_map_v1",
                    "reviewer": reviewer,
                    "seed": args.seed,
                    "rows": map_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
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

    manifest_path = args.output_dir / "packet_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "paper_a_review_packet_manifest_v1",
                "target_source": str(args.targets),
                "target_source_sha256": sha256(args.targets),
                "calibration_source": str(args.calibration) if args.calibration else None,
                "calibration_source_sha256": sha256(args.calibration) if args.calibration else None,
                "seed": args.seed,
                "reviewers": manifests,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifests)} reviewer packets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
