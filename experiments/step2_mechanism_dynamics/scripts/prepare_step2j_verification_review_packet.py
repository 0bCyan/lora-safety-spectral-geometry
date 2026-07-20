from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl


ROOT = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2j/entry_audit")
BLIND_PACKET = ROOT / "step2j_natural_v1_independent_target_review_blinded.jsonl"
REVIEW_DIR = ROOT / "independent_target_review"
PRIMARY_REVIEW = REVIEW_DIR / "primary_review.jsonl"
VERIFICATION_SAMPLE = REVIEW_DIR / "verification_sample.json"
OUT_PACKET = REVIEW_DIR / "verification_review_packet.jsonl"


def read_review_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-primary-rejects", action="store_true", default=True)
    args = parser.parse_args()

    packet_by_id = {row["target_id"]: row for row in read_jsonl(BLIND_PACKET)}
    sample = json.loads(VERIFICATION_SAMPLE.read_text(encoding="utf-8"))
    target_ids = {row["target_id"] for row in sample["rows"]}

    primary_rows = read_review_jsonl(PRIMARY_REVIEW)
    reject_or_uncertain = []
    for row in primary_rows:
        reason_code = str(row.get("reason_code", "")).lower()
        if row.get("overall_decision") == "reject" or reason_code.startswith("borderline_") or "uncertain" in reason_code:
            reject_or_uncertain.append(row["target_id"])
    if args.include_primary_rejects:
        target_ids.update(reject_or_uncertain)

    rows = []
    for target_id in sorted(target_ids):
        if target_id not in packet_by_id:
            raise ValueError(f"Target id missing from blind packet: {target_id}")
        rows.append(packet_by_id[target_id])
    write_jsonl(OUT_PACKET, rows)
    print(f"wrote {OUT_PACKET} ({len(rows)} rows)")
    print(f"primary reject/borderline rows: {len(reject_or_uncertain)}")


if __name__ == "__main__":
    main()
