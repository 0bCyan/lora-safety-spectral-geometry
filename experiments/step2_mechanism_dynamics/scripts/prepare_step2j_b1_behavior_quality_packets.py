from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from common import read_jsonl, write_csv, write_jsonl


DECAY_ROOT = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2ir_natural")
CONSTANT_ROOT = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2j_b1")
OUT_ROOT = CONSTANT_ROOT / "behavior_quality"
EVIDENCE_DIR = OUT_ROOT / "evidence"
PACKET_DIR = OUT_ROOT / "packets"
REVIEW_DIR = OUT_ROOT / "reviews"
SAFETY_SEED = 20260720
UTILITY_SEED = 20260720
RUBRIC_VERSION = "step2j_b1_behavior_quality_adjudication_rubric_v1"
REVIEW_SESSION_ID = "step2j_b1_behavior_quality_20260720"


SAFETY_SOURCES = [
    ("decay10", "step0032_pct050", "generation", DECAY_ROOT / "generations_full100_core" / "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10_step0032_pct050_safety_full100_core.jsonl"),
    ("decay10", "step0064_pct100", "generation", DECAY_ROOT / "generations_full100_core" / "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10_step0064_pct100_safety_full100_core.jsonl"),
    ("decay10", "step0032_pct050", "local_judge", DECAY_ROOT / "local_judge_full100_core" / "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10_step0032_pct050_safety_full100_core_local_judge.jsonl"),
    ("decay10", "step0064_pct100", "local_judge", DECAY_ROOT / "local_judge_full100_core" / "step2irn_4070ti_qwen3-4b_risk_s1_natural_diverse_10_step0064_pct100_safety_full100_core_local_judge.jsonl"),
    ("constant20", "step0032_pct050", "generation", CONSTANT_ROOT / "generations_full100_core" / "step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20_step0032_pct050_safety_full100_core.jsonl"),
    ("constant20", "step0064_pct100", "generation", CONSTANT_ROOT / "generations_full100_core" / "step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20_step0064_pct100_safety_full100_core.jsonl"),
    ("constant20", "step0032_pct050", "local_judge", CONSTANT_ROOT / "local_judge_full100_core" / "step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20_step0032_pct050_safety_full100_core_local_judge.jsonl"),
    ("constant20", "step0064_pct100", "local_judge", CONSTANT_ROOT / "local_judge_full100_core" / "step2jb1_4070ti_qwen3-4b_risk_s1_natural_constant20_step0064_pct100_safety_full100_core_local_judge.jsonl"),
]

UTILITY_SOURCES = [
    ("constant20", "step0064_pct100", "generation", CONSTANT_ROOT / "utility_generations" / "natural_constant20_s1_step0064_pct100_utility100.jsonl"),
    ("constant20", "step0064_pct100", "quality", CONSTANT_ROOT / "utility_quality" / "natural_constant20_s1_step0064_pct100_utility100_quality.jsonl"),
    ("vanilla", "step0064_pct100", "generation", CONSTANT_ROOT / "utility_generations" / "vanilla_risk_s1_step0064_pct100_utility100.jsonl"),
    ("vanilla", "step0064_pct100", "quality", CONSTANT_ROOT / "utility_quality" / "vanilla_risk_s1_step0064_pct100_utility100_quality.jsonl"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def copy_evidence() -> list[dict[str, Any]]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, sources in [("safety", SAFETY_SOURCES), ("utility", UTILITY_SOURCES)]:
        for arm, checkpoint, kind, source in sources:
            if not source.exists():
                raise FileNotFoundError(source)
            target = EVIDENCE_DIR / group / arm / checkpoint / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hash = sha256_file(source)
            target_hash = sha256_file(target)
            rows.append(
                {
                    "group": group,
                    "arm": arm,
                    "checkpoint": checkpoint,
                    "kind": kind,
                    "source_path": str(source),
                    "tracked_path": str(target),
                    "row_count": row_count_jsonl(target),
                    "byte_count": target.stat().st_size,
                    "sha256": target_hash,
                    "source_sha256": source_hash,
                    "source_tracked_hash_equal": source_hash == target_hash,
                }
            )
    write_csv(OUT_ROOT / "raw_evidence_manifest.csv", rows)
    return rows


def safety_generation(arm: str, checkpoint: str) -> list[dict[str, Any]]:
    source = next(path for a, c, kind, path in SAFETY_SOURCES if a == arm and c == checkpoint and kind == "generation")
    return read_jsonl(source)


def safety_judge(arm: str, checkpoint: str) -> dict[str, dict[str, Any]]:
    source = next(path for a, c, kind, path in SAFETY_SOURCES if a == arm and c == checkpoint and kind == "local_judge")
    return {row["sample_id"]: row for row in read_jsonl(source)}


def utility_generation(arm: str) -> list[dict[str, Any]]:
    source = next(path for a, _, kind, path in UTILITY_SOURCES if a == arm and kind == "generation")
    return read_jsonl(source)


def utility_quality(arm: str) -> dict[str, dict[str, Any]]:
    source = next(path for a, _, kind, path in UTILITY_SOURCES if a == arm and kind == "quality")
    return {row["sample_id"]: row for row in read_jsonl(source)}


def validate_alignment() -> dict[str, Any]:
    safety_ids = {}
    for arm in ["decay10", "constant20"]:
        for checkpoint in ["step0032_pct050", "step0064_pct100"]:
            gen_ids = [row["sample_id"] for row in safety_generation(arm, checkpoint)]
            judge_ids = list(safety_judge(arm, checkpoint))
            if len(gen_ids) != 100 or len(set(gen_ids)) != 100:
                raise ValueError(f"{arm} {checkpoint} safety generation must contain 100 unique sample IDs")
            if gen_ids != judge_ids:
                raise ValueError(f"{arm} {checkpoint} safety generation/local-judge IDs differ")
            safety_ids[(arm, checkpoint)] = gen_ids
    reference = safety_ids[("decay10", "step0032_pct050")]
    for key, ids in safety_ids.items():
        if ids != reference:
            raise ValueError(f"Safety prompt alignment differs for {key}")

    utility_ids = {}
    for arm in ["vanilla", "constant20"]:
        gen_ids = [row["sample_id"] for row in utility_generation(arm)]
        quality_ids = list(utility_quality(arm))
        if len(gen_ids) != 100 or len(set(gen_ids)) != 100:
            raise ValueError(f"{arm} utility generation must contain 100 unique sample IDs")
        if gen_ids != quality_ids:
            raise ValueError(f"{arm} utility generation/quality IDs differ")
        utility_ids[arm] = gen_ids
    if utility_ids["vanilla"] != utility_ids["constant20"]:
        raise ValueError("Utility prompt alignment differs between vanilla and constant20")
    return {
        "safety_shared_sample_ids": len(reference),
        "utility_shared_sample_ids": len(utility_ids["vanilla"]),
        "safety_prompt_alignment": True,
        "utility_prompt_alignment": True,
    }


def opaque_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def freeze_safety_packet() -> dict[str, Any]:
    packet_rows = []
    key_rows = []
    index = 0
    for arm in ["decay10", "constant20"]:
        for checkpoint in ["step0032_pct050", "step0064_pct100"]:
            judge = safety_judge(arm, checkpoint)
            for row in safety_generation(arm, checkpoint):
                index += 1
                response_id = opaque_id("safety", index)
                packet_rows.append(
                    {
                        "opaque_response_id": response_id,
                        "prompt": row["prompt"],
                        "response": row["response"],
                    }
                )
                local = judge[row["sample_id"]]
                key_rows.append(
                    {
                        "opaque_response_id": response_id,
                        "arm": arm,
                        "checkpoint": checkpoint,
                        "sample_id": row["sample_id"],
                        "local_heuristic_label": local["label"],
                        "local_heuristic_reason": local.get("local_reason", ""),
                    }
                )
    rng = random.Random(SAFETY_SEED)
    rng.shuffle(packet_rows)
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    packet_path = PACKET_DIR / "step2j_b1_safety_behavior_packet.jsonl"
    key_path = PACKET_DIR / "step2j_b1_safety_behavior_key.csv"
    write_jsonl(packet_path, packet_rows)
    write_csv(key_path, key_rows)
    verification_ids = []
    by_cell = {(row["arm"], row["checkpoint"]): [] for row in key_rows}
    for row in key_rows:
        by_cell[(row["arm"], row["checkpoint"])].append(row["opaque_response_id"])
    sample_rng = random.Random(SAFETY_SEED + 1)
    for cell in sorted(by_cell):
        verification_ids.extend(sample_rng.sample(sorted(by_cell[cell]), 25))
    verification_path = PACKET_DIR / "step2j_b1_safety_verification_sample.json"
    write_json(
        verification_path,
        {
            "sample_rows": len(verification_ids),
            "rows_per_arm_checkpoint_cell": 25,
            "sample_seed": SAFETY_SEED + 1,
            "opaque_response_ids": verification_ids,
            "target_id_sha256": hashlib.sha256(("\n".join(verification_ids) + "\n").encode("utf-8")).hexdigest(),
        },
    )
    return {
        "packet": str(packet_path),
        "key": str(key_path),
        "verification_sample": str(verification_path),
        "packet_rows": len(packet_rows),
        "key_rows": len(key_rows),
        "packet_sha256": sha256_file(packet_path),
        "key_sha256": sha256_file(key_path),
        "verification_sample_sha256": sha256_file(verification_path),
        "row_order_seed": SAFETY_SEED,
        "rubric_version": RUBRIC_VERSION,
        "review_session_id": REVIEW_SESSION_ID,
    }


def freeze_utility_packet() -> dict[str, Any]:
    packet_rows = []
    key_rows = []
    index = 0
    for arm in ["vanilla", "constant20"]:
        quality = utility_quality(arm)
        for row in utility_generation(arm):
            index += 1
            response_id = opaque_id("utility", index)
            packet_rows.append(
                {
                    "opaque_response_id": response_id,
                    "prompt": row["prompt"],
                    "reference_response": row.get("reference_response", ""),
                    "response": row["response"],
                }
            )
            local = quality[row["sample_id"]]
            key_rows.append(
                {
                    "opaque_response_id": response_id,
                    "arm": arm,
                    "checkpoint": "step0064_pct100",
                    "sample_id": row["sample_id"],
                    "local_quality_label": local["label"],
                    "local_task_success": local["task_success"],
                    "local_unnecessary_refusal": local["unnecessary_refusal"],
                    "local_invalid_or_irrelevant": local["invalid_or_irrelevant"],
                    "local_degradation": local["degradation"],
                }
            )
    rng = random.Random(UTILITY_SEED)
    rng.shuffle(packet_rows)
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    packet_path = PACKET_DIR / "step2j_b1_utility100_packet.jsonl"
    key_path = PACKET_DIR / "step2j_b1_utility100_key.csv"
    write_jsonl(packet_path, packet_rows)
    write_csv(key_path, key_rows)
    verification_ids = []
    by_arm = {"vanilla": [], "constant20": []}
    for row in key_rows:
        by_arm[row["arm"]].append(row["opaque_response_id"])
    sample_rng = random.Random(UTILITY_SEED + 2)
    for arm, ids in sorted(by_arm.items()):
        verification_ids.extend(sample_rng.sample(sorted(ids), 20))
    verification_path = PACKET_DIR / "step2j_b1_utility_verification_sample.json"
    write_json(
        verification_path,
        {
            "sample_rows": len(verification_ids),
            "rows_per_arm": 20,
            "sample_seed": UTILITY_SEED + 2,
            "opaque_response_ids": verification_ids,
            "target_id_sha256": hashlib.sha256(("\n".join(verification_ids) + "\n").encode("utf-8")).hexdigest(),
        },
    )
    return {
        "packet": str(packet_path),
        "key": str(key_path),
        "verification_sample": str(verification_path),
        "packet_rows": len(packet_rows),
        "key_rows": len(key_rows),
        "packet_sha256": sha256_file(packet_path),
        "key_sha256": sha256_file(key_path),
        "verification_sample_sha256": sha256_file(verification_path),
        "row_order_seed": UTILITY_SEED,
        "rubric_version": RUBRIC_VERSION,
        "review_session_id": REVIEW_SESSION_ID,
    }


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    evidence = copy_evidence()
    if not all(row["source_tracked_hash_equal"] for row in evidence):
        raise ValueError("Evidence source/tracked hash mismatch")
    alignment = validate_alignment()
    safety = freeze_safety_packet()
    utility = freeze_utility_packet()
    manifest = {
        "review_session_id": REVIEW_SESSION_ID,
        "rubric_version": RUBRIC_VERSION,
        "bootstrap_seed": SAFETY_SEED,
        "raw_evidence_manifest": str(OUT_ROOT / "raw_evidence_manifest.csv"),
        "alignment": alignment,
        "safety": safety,
        "utility": utility,
        "status": "packets_frozen_pending_independent_adjudication",
        "prohibited_until_route_decision": [
            "constant30",
            "adaptive_norm",
            "risk_seeds_2_3",
            "heldout_test",
            "SpectraSafe",
            "additional_safety_weight_scans",
            "target_rewrites",
        ],
    }
    write_json(PACKET_DIR / "step2j_b1_behavior_quality_packet_manifest.json", manifest)
    print(f"wrote {OUT_ROOT / 'raw_evidence_manifest.csv'} ({len(evidence)} rows)")
    print(f"wrote {PACKET_DIR / 'step2j_b1_safety_behavior_packet.jsonl'} ({safety['packet_rows']} rows)")
    print(f"wrote {PACKET_DIR / 'step2j_b1_utility100_packet.jsonl'} ({utility['packet_rows']} rows)")
    print(f"wrote {PACKET_DIR / 'step2j_b1_behavior_quality_packet_manifest.json'}")


if __name__ == "__main__":
    main()
