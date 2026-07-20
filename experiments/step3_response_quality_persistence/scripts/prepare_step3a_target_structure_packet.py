from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
BASE = ROOT / "results/step3a_target_structure_review"
PACKETS = BASE / "packets"
KEYS = BASE / "keys"

TARGETS = Path(
    "experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1.jsonl"
)
SEED = 20260720
REVIEW_SESSION = "step3a_natural_v1_response_only_20260720"
RUBRIC_VERSION = "step3a_target_structure_response_only_v1"

FIELDS = [
    "opaque_target_id",
    "boundary_act",
    "rationale_act",
    "redirect_act",
    "harmful_procedural_tail",
    "mechanical_template",
    "short_reason",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = read_jsonl(TARGETS)
    if len(rows) != 80:
        raise ValueError(f"expected 80 natural-v1 targets, found {len(rows)}")
    random.Random(SEED).shuffle(rows)

    packet = []
    key = []
    for index, row in enumerate(rows, start=1):
        opaque_id = f"T3S_{index:04d}"
        packet.append({"opaque_target_id": opaque_id, "response": row["response"]})
        key.append(
            {
                "opaque_target_id": opaque_id,
                "target_id": row["target_id"],
                "sample_id": row["sample_id"],
                "category": row["category"],
                "target_hash": row["target_hash"],
            }
        )
    template = [
        {field: row["opaque_target_id"] if field == "opaque_target_id" else "" for field in FIELDS}
        for row in packet
    ]

    packet_path = PACKETS / "step3a_target_structure_response_only_packet.jsonl"
    template_a = PACKETS / "step3a_target_structure_reviewer_a_template.jsonl"
    template_b = PACKETS / "step3a_target_structure_reviewer_b_template.jsonl"
    key_path = KEYS / "step3a_target_structure_key.csv"
    write_jsonl(packet_path, packet)
    write_jsonl(template_a, template)
    write_jsonl(template_b, template)
    write_csv(key_path, key)

    manifest = {
        "review_session": REVIEW_SESSION,
        "rubric_version": RUBRIC_VERSION,
        "rows": len(packet),
        "packet_contains_prompt": False,
        "packet_contains_safe_target_response_only": True,
        "heldout_content_included": False,
        "reviewers_must_not_read_key": True,
        "artifacts": {
            "packet": {"path": str(packet_path), "sha256": sha256(packet_path)},
            "template_a": {"path": str(template_a), "sha256": sha256(template_a)},
            "template_b": {"path": str(template_b), "sha256": sha256(template_b)},
            "key": {"path": str(key_path), "sha256": sha256(key_path)},
        },
    }
    manifest_path = PACKETS / "step3a_target_structure_packet_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {packet_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
