from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl


MANIFEST = Path("experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1.jsonl")
FREEZE = Path("experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1_freeze.json")
PREFLIGHT_REVIEW = Path("experiments/step2_mechanism_dynamics/manifests/step2ir_natural_diverse_targets_v1_review.csv")
RUN_DIR = Path(
    "experiments/step2_mechanism_dynamics/results/4070ti_step2ir_natural/runs/"
    "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10"
)
OUT_DIR = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2j/entry_audit")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_blind_review_packet(rows: list[dict[str, Any]]) -> Path:
    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "target_id": row["target_id"],
                "prompt": row["prompt"],
                "target_response": row["response"],
                "explicit_refusal_validity": "",
                "harmful_procedural_leakage": "",
                "prompt_relevant_redirect": "",
                "grammatical_naturalness": "",
                "semantic_coherence": "",
                "keyword_stuffing": "",
                "mechanical_filler": "",
                "overall_decision": "",
                "short_reason": "",
            }
        )
    path = OUT_DIR / "step2j_natural_v1_independent_target_review_blinded.jsonl"
    write_jsonl(path, review_rows)
    return path


def exposure_integrity(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exposure_rows = read_csv(RUN_DIR / "anchor_exposure_manifest.csv")
    draw_rows = read_csv(RUN_DIR / "anchor_draw_trace.csv")
    checkpoint_rows = read_csv(RUN_DIR / "checkpoint_index.csv")
    manifest_hashes = [row["target_hash"] for row in manifest_rows]
    exposure_hashes = [row["target_hash"] for row in exposure_rows]
    exposure_counts = [int(row["exposure_count"]) for row in exposure_rows]
    return {
        "run_dir": str(RUN_DIR),
        "manifest_rows": len(manifest_rows),
        "exposure_rows": len(exposure_rows),
        "anchor_draw_rows": len(draw_rows),
        "checkpoint_rows": len(checkpoint_rows),
        "manifest_sha256": sha256_text_lf(MANIFEST),
        "preflight_review_sha256": sha256_text_lf(PREFLIGHT_REVIEW),
        "manifest_raw_sha256": sha256_file(MANIFEST),
        "preflight_review_raw_sha256": sha256_file(PREFLIGHT_REVIEW),
        "text_hash_normalization": "utf-8 with CRLF normalized to LF for freeze comparisons",
        "anchor_draw_trace_sha256": sha256_file(RUN_DIR / "anchor_draw_trace.csv"),
        "anchor_exposure_manifest_sha256": sha256_file(RUN_DIR / "anchor_exposure_manifest.csv"),
        "checkpoint_index_sha256": sha256_file(RUN_DIR / "checkpoint_index.csv"),
        "regularization_loss_trace_sha256": sha256_file(RUN_DIR / "regularization_loss_trace.csv"),
        "target_hash_order_matches_manifest": exposure_hashes == manifest_hashes,
        "exposure_min": min(exposure_counts),
        "exposure_max": max(exposure_counts),
        "exposure_mean": sum(exposure_counts) / len(exposure_counts),
        "exposure_count_3": sum(count == 3 for count in exposure_counts),
        "exposure_count_4": sum(count == 4 for count in exposure_counts),
        "total_anchor_draws": sum(exposure_counts),
        "checkpoints": [row["checkpoint"] for row in checkpoint_rows],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(MANIFEST)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    review_packet = build_blind_review_packet(manifest_rows)
    integrity = exposure_integrity(manifest_rows)
    integrity["freeze_manifest_sha256"] = freeze["manifest_sha256"]
    integrity["freeze_review_sha256"] = freeze["review_sha256"]
    integrity["manifest_hash_matches_freeze"] = integrity["manifest_sha256"] == freeze["manifest_sha256"]
    integrity["review_hash_matches_freeze"] = integrity["preflight_review_sha256"] == freeze["review_sha256"]
    integrity["formal_control_usable"] = (
        integrity["manifest_rows"] == 80
        and integrity["exposure_rows"] == 80
        and integrity["total_anchor_draws"] == 256
        and integrity["exposure_min"] == 3
        and integrity["exposure_max"] == 4
        and integrity["target_hash_order_matches_manifest"]
        and integrity["manifest_hash_matches_freeze"]
        and integrity["review_hash_matches_freeze"]
    )
    write_json(OUT_DIR / "step2j_natural_v1_evidence_integrity.json", integrity)
    (OUT_DIR / "step2j_entry_audit_status.md").write_text(
        f"""# Step2J-A Entry Audit Status

## Independent Target Review Packet

Prepared blinded review packet:

`{review_packet}`

Rows: {len(manifest_rows)}

Status: pending external independent agent or human review. The existing v1 review remains a local preflight, not an independent approval.

## Evidence Integrity

| check | result |
| --- | --- |
| manifest rows | {integrity['manifest_rows']} |
| exposure rows | {integrity['exposure_rows']} |
| total anchor draws | {integrity['total_anchor_draws']} |
| exposure min/max | {integrity['exposure_min']} / {integrity['exposure_max']} |
| exposure mean | {integrity['exposure_mean']:.3f} |
| target hash order matches manifest | {integrity['target_hash_order_matches_manifest']} |
| manifest hash matches freeze | {integrity['manifest_hash_matches_freeze']} |
| review hash matches freeze | {integrity['review_hash_matches_freeze']} |
| formal decay-to-10 control usable | {integrity['formal_control_usable']} |

Text manifest and preflight-review hashes are computed after normalizing CRLF to LF for freeze comparisons.
Small reproducibility artifact hashes are recorded in `step2j_natural_v1_evidence_integrity.json`.
""",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {review_packet}")
    print(f"wrote {OUT_DIR / 'step2j_natural_v1_evidence_integrity.json'}")
    print(f"wrote {OUT_DIR / 'step2j_entry_audit_status.md'}")


if __name__ == "__main__":
    main()
