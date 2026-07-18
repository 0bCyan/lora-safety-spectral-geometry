from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl


PACKET = Path(
    "experiments/step2_mechanism_dynamics/results/4070ti_step2j/entry_audit/"
    "step2j_natural_v1_independent_target_review_blinded.jsonl"
)
MANIFEST = Path("experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1.jsonl")
OUT_DIR = Path(
    "experiments/step2_mechanism_dynamics/results/4070ti_step2j/entry_audit/"
    "independent_target_review"
)
RUBRIC_VERSION = "step2j_b0_independent_target_review_rubric_v1"
REVIEW_SESSION_ID = "step2j_b0_natural_v1_independent_review_20260719"

SAMPLE_CATEGORY_PLAN = {
    "harmful wrongdoing": 6,
    "financial evasion": 2,
    "public manipulation": 2,
    "cyber abuse": 2,
    "fraud or deception": 1,
    "physical harm": 1,
    "privacy invasion": 1,
    "harassment or coercion": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def packet_by_target_id() -> dict[str, dict[str, Any]]:
    return {row["target_id"]: row for row in read_jsonl(PACKET)}


def manifest_unique_prompts() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(MANIFEST):
        grouped[str(row["sample_id"])].append(row)
    return {sample_id: sorted(rows, key=lambda item: item["target_id"]) for sample_id, rows in grouped.items()}


def select_verification_sample() -> list[dict[str, Any]]:
    packet_rows = packet_by_target_id()
    by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for sample_id, variants in manifest_unique_prompts().items():
        by_category[str(variants[0]["category"])].append((sample_id, variants))

    selected: list[dict[str, Any]] = []
    variant_cursor = 0
    used_prompts: set[str] = set()
    for category, count in SAMPLE_CATEGORY_PLAN.items():
        available = [(sample_id, rows) for sample_id, rows in by_category[category] if sample_id not in used_prompts]
        if len(available) < count:
            raise ValueError(f"Not enough prompts for {category}: need {count}, got {len(available)}")
        for sample_id, variants in available[:count]:
            variant = variants[variant_cursor % len(variants)]
            target_id = variant["target_id"]
            if target_id not in packet_rows:
                raise ValueError(f"Selected target missing from blinded packet: {target_id}")
            selected.append(
                {
                    "target_id": target_id,
                    "sample_id": sample_id,
                    "category": category,
                    "variant_choice": variant_cursor % len(variants) + 1,
                }
            )
            used_prompts.add(sample_id)
            variant_cursor += 1

    if len({row["sample_id"] for row in selected}) != 16:
        raise ValueError("Verification sample must contain 16 unique prompts.")
    if Counter(row["variant_choice"] for row in selected) != {1: 8, 2: 8}:
        raise ValueError("Verification sample must balance target1 and target2 variants.")
    return selected


def build_review_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": row["target_id"],
            "explicit_refusal_validity": "",
            "harmful_procedural_leakage": "",
            "prompt_relevant_redirect": "",
            "grammatical_naturalness": "",
            "semantic_coherence": "",
            "keyword_stuffing": "",
            "mechanical_filler": "",
            "overall_decision": "",
            "reason_code": "",
            "short_reason": "",
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-type", default="pending_independent_agent_or_human")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    packet_rows = read_jsonl(PACKET)
    if len(packet_rows) != 80:
        raise ValueError(f"Expected 80 blinded packet rows, got {len(packet_rows)}")

    sample = select_verification_sample()
    sample_ids_text = "\n".join(row["target_id"] for row in sample) + "\n"
    input_manifest = {
        "review_session_id": REVIEW_SESSION_ID,
        "rubric_version": RUBRIC_VERSION,
        "reviewer_type": args.reviewer_type,
        "status": "prepared_pending_independent_review",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_packet": str(PACKET),
        "input_packet_rows": len(packet_rows),
        "input_packet_sha256": sha256_file(PACKET),
        "input_packet_immutable_after_hashing": True,
        "primary_reviewer_blinding": [
            "no training results",
            "no arm identities beyond packet fields",
            "no local preflight decisions",
            "no gradient results",
            "no target-writing script",
            "no previous target-quality analyses",
        ],
        "verification_sample_path": str(OUT_DIR / "verification_sample.json"),
        "verification_sample_rows": len(sample),
        "verification_sample_target_id_sha256": sha256_text(sample_ids_text),
    }
    write_json(OUT_DIR / "review_input_manifest.json", input_manifest)
    write_json(
        OUT_DIR / "verification_sample.json",
        {
            "review_session_id": REVIEW_SESSION_ID,
            "rubric_version": RUBRIC_VERSION,
            "selected_before_primary_decisions": True,
            "target_id_sha256": input_manifest["verification_sample_target_id_sha256"],
            "category_counts": dict(Counter(row["category"] for row in sample)),
            "variant_counts": dict(Counter(str(row["variant_choice"]) for row in sample)),
            "rows": sample,
        },
    )
    write_jsonl(OUT_DIR / "primary_review_template.jsonl", build_review_template(packet_rows))
    verification_rows = [packet_by_target_id()[row["target_id"]] for row in sample]
    write_jsonl(OUT_DIR / "verification_review_template.jsonl", build_review_template(verification_rows))

    print(f"wrote {OUT_DIR / 'review_input_manifest.json'}")
    print(f"wrote {OUT_DIR / 'verification_sample.json'}")
    print(f"wrote {OUT_DIR / 'primary_review_template.jsonl'}")
    print(f"wrote {OUT_DIR / 'verification_review_template.jsonl'}")


if __name__ == "__main__":
    main()
