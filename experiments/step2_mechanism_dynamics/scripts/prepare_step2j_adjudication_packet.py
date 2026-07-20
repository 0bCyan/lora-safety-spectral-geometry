from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl


ROOT = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2j/entry_audit")
REVIEW_DIR = ROOT / "independent_target_review"
BLIND_PACKET = ROOT / "step2j_natural_v1_independent_target_review_blinded.jsonl"
PRIMARY_REVIEW = REVIEW_DIR / "primary_review.jsonl"
VERIFICATION_REVIEW = REVIEW_DIR / "verification_review.jsonl"
OUT_PACKET = REVIEW_DIR / "adjudication_packet.jsonl"

LABEL_FIELDS = [
    "explicit_refusal_validity",
    "harmful_procedural_leakage",
    "prompt_relevant_redirect",
    "grammatical_naturalness",
    "semantic_coherence",
    "keyword_stuffing",
    "mechanical_filler",
    "overall_decision",
]


def read_review_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["target_id"]: row for row in rows}


def main() -> None:
    packet = by_id(read_jsonl(BLIND_PACKET))
    primary = by_id(read_review_jsonl(PRIMARY_REVIEW))
    verification = by_id(read_review_jsonl(VERIFICATION_REVIEW))
    rows = []
    for target_id in sorted(set(primary) & set(verification)):
        disagreed = [field for field in LABEL_FIELDS if primary[target_id][field] != verification[target_id][field]]
        if not disagreed:
            continue
        source = packet[target_id]
        rows.append(
            {
                "target_id": target_id,
                "prompt": source["prompt"],
                "target_response": source["target_response"],
                "disagreed_fields": disagreed,
                "primary_short_reason": primary[target_id].get("short_reason", ""),
                "verification_short_reason": verification[target_id].get("short_reason", ""),
            }
        )
    write_jsonl(OUT_PACKET, rows)
    print(f"wrote {OUT_PACKET} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
