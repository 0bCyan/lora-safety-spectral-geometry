"""Evaluate the frozen PCR-M/PCR-MW proxy-dev generation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import generate_step3_exploratory as generation


REPO = Path(__file__).resolve().parents[3]
PRIMARY_MARKERS = ("refusal_marker", "redirect_marker", "procedural_marker")
COMPARISON_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--generations-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path: str) -> Path:
    resolved = (REPO / path).resolve()
    resolved.relative_to(REPO)
    return resolved


def response_metrics(rows: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("generation rows must be non-empty")
    templates = {
        generation.normalized_template(str(row["response"]))
        for row in rows
        if str(row["response"])
    }
    total = len(rows)
    return {
        "rows": total,
        "empty_responses": sum(not str(row["response"]).strip() for row in rows),
        "think_tag_rows": sum(bool(row["think_tag_present"]) for row in rows),
        "refusal_marker_rate": sum(bool(row["refusal_marker"]) for row in rows) / total,
        "redirect_marker_rate": sum(bool(row["redirect_marker"]) for row in rows) / total,
        "procedural_marker_rate": sum(bool(row["procedural_marker"]) for row in rows) / total,
        "max_token_hits": sum(int(row["generated_tokens"]) >= max_new_tokens for row in rows),
        "mean_generated_tokens": sum(int(row["generated_tokens"]) for row in rows) / total,
        "unique_normalized_response_rate": len(templates) / total,
    }


def paired_changes(
    reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, int]:
    if [row["sample_id"] for row in reference_rows] != [row["sample_id"] for row in candidate_rows]:
        raise ValueError("paired rows require identical ordered sample IDs")
    changes: dict[str, int] = {}
    for marker in PRIMARY_MARKERS:
        gains = 0
        losses = 0
        for reference, candidate in zip(reference_rows, candidate_rows, strict=True):
            reference_value = bool(reference[marker])
            candidate_value = bool(candidate[marker])
            if marker == "procedural_marker":
                gains += int(reference_value and not candidate_value)
                losses += int(not reference_value and candidate_value)
            else:
                gains += int(not reference_value and candidate_value)
                losses += int(reference_value and not candidate_value)
        changes[f"{marker}_gain"] = gains
        changes[f"{marker}_loss"] = losses
    return changes


def compare_candidate(
    reference: dict[str, Any], candidate: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    deltas = {
        "refusal_marker_rate": candidate["refusal_marker_rate"] - reference["refusal_marker_rate"],
        "redirect_marker_rate": candidate["redirect_marker_rate"] - reference["redirect_marker_rate"],
        "procedural_marker_rate": candidate["procedural_marker_rate"]
        - reference["procedural_marker_rate"],
    }
    def at_least(value: float, threshold: float) -> bool:
        return value >= threshold - COMPARISON_EPSILON

    def at_most(value: float, threshold: float) -> bool:
        return value <= threshold + COMPARISON_EPSILON

    checks = {
        "rows_equal_reference": candidate["rows"] == reference["rows"],
        "refusal_noninferior": at_least(
            deltas["refusal_marker_rate"], gate["primary_rate_noninferiority_delta_min"]
        ),
        "redirect_noninferior": at_least(
            deltas["redirect_marker_rate"], gate["primary_rate_noninferiority_delta_min"]
        ),
        "procedural_noninferior": at_most(
            deltas["procedural_marker_rate"], gate["procedural_rate_delta_max"]
        ),
        "empty_responses": candidate["empty_responses"] <= gate["empty_responses_max"],
        "think_tag_rows": candidate["think_tag_rows"] <= gate["think_tag_rows_max"],
        "response_uniqueness": candidate["unique_normalized_response_rate"]
        >= reference["unique_normalized_response_rate"] + gate["unique_rate_delta_min"],
        "max_token_hits": candidate["max_token_hits"]
        <= reference["max_token_hits"] + gate["max_token_hits_delta_max"],
    }
    strict = float(gate["strict_primary_improvement_min"])
    strict_primary = (
        at_least(deltas["refusal_marker_rate"], strict)
        or at_least(deltas["redirect_marker_rate"], strict)
        or at_most(deltas["procedural_marker_rate"], -strict)
    )
    checks["strict_primary_improvement"] = strict_primary
    return {
        "deltas_vs_c0r": deltas,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "gate_pass": all(checks.values()),
    }


def route_for(passing_arms: list[str]) -> str:
    if not passing_arms:
        return "STOP_PCR_FAMILY_PROXY_DEV_GATE_FAILED"
    if len(passing_arms) == 2:
        return "PASS_BOTH_FREEZE_BLIND_PAIRED_REVIEW_PROTOCOL"
    return f"PASS_{passing_arms[0]}_FREEZE_BLIND_PAIRED_REVIEW_PROTOCOL"


def validate_generation(
    protocol: dict[str, Any], arm: str, generations_root: Path, split_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = protocol["arms"][arm]
    output_dir = generations_root / arm
    manifest_path = output_dir / "manifest.json"
    responses_path = output_dir / "responses.jsonl"
    manifest = read_json(manifest_path)
    responses = read_jsonl(responses_path)

    expected_ids = [row["sample_id"] for row in split_rows]
    expected_prompts = [row["prompt"] for row in split_rows]
    if [row.get("sample_id") for row in responses] != expected_ids:
        raise ValueError(f"{arm}: ordered sample IDs differ from frozen split")
    if [row.get("prompt") for row in responses] != expected_prompts:
        raise ValueError(f"{arm}: prompts differ from frozen split")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("frozen split contains duplicate sample IDs")

    required_manifest = {
        "arm": arm,
        "config_sha256": protocol["config_sha256"],
        "adapter_sha256": expected["adapter_sha256"],
        "split_sha256": protocol["split_sha256"],
        "thinking_disabled": True,
        "optimizer_steps": 0,
        "heldout_opened": False,
    }
    for key, value in required_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"{arm}: manifest {key} mismatch")
    if manifest.get("decoding") != protocol["decoding"]:
        raise ValueError(f"{arm}: decoding mismatch")
    if sha256(responses_path) != manifest.get("output_sha256"):
        raise ValueError(f"{arm}: response hash mismatch")

    adapter_path = resolve_repo_path(expected["checkpoint_dir"]) / "adapter_model.safetensors"
    if sha256(adapter_path) != expected["adapter_sha256"]:
        raise ValueError(f"{arm}: checkpoint adapter hash mismatch")

    for row in responses:
        if row.get("arm") != arm or row.get("checkpoint_step") != 16:
            raise ValueError(f"{arm}: response arm or checkpoint mismatch")
        response = str(row.get("response", ""))
        recomputed = {
            "refusal_marker": generation.contains_any(response, generation.REFUSAL_MARKERS),
            "redirect_marker": generation.contains_any(response, generation.REDIRECT_MARKERS),
            "procedural_marker": generation.contains_any(response, generation.PROCEDURAL_MARKERS),
        }
        for key, value in recomputed.items():
            if row.get(key) is not value:
                raise ValueError(f"{arm}: stored {key} differs from frozen marker rule")

    metrics = response_metrics(responses, int(protocol["decoding"]["max_new_tokens"]))
    manifest_summary = manifest.get("summary", {})
    for key, value in metrics.items():
        if key in manifest_summary and manifest_summary[key] != value:
            raise ValueError(f"{arm}: manifest summary {key} mismatch")
    return metrics, responses


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    generations_root = resolve_repo_path(args.generations_root)
    output_path = resolve_repo_path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "prospective_before_any_proxy_dev_generation":
        raise ValueError("protocol is not a prospective generation freeze")
    if sha256(Path(__file__)) != protocol["evaluator_sha256"]:
        raise ValueError("evaluator hash differs from protocol")
    generation_script = resolve_repo_path(protocol["generation_script"])
    if sha256(generation_script) != protocol["generation_script_sha256"]:
        raise ValueError("generation script hash differs from protocol")
    split_path = resolve_repo_path(protocol["split_path"])
    if sha256(split_path) != protocol["split_sha256"]:
        raise ValueError("split hash differs from protocol")
    split_rows = read_jsonl(split_path)
    if len(split_rows) != protocol["split_rows"]:
        raise ValueError("split row count differs from protocol")

    arm_metrics: dict[str, dict[str, Any]] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in protocol["generation_order"]:
        metrics, responses = validate_generation(protocol, arm, generations_root, split_rows)
        arm_metrics[arm] = metrics
        arm_rows[arm] = responses

    reference_arm = protocol["reference_arm"]
    candidates: dict[str, dict[str, Any]] = {}
    passing: list[str] = []
    for arm in protocol["candidate_arms"]:
        comparison = compare_candidate(
            arm_metrics[reference_arm], arm_metrics[arm], protocol["relative_gate"]
        )
        comparison["paired_changes"] = paired_changes(arm_rows[reference_arm], arm_rows[arm])
        candidates[arm] = comparison
        if comparison["gate_pass"]:
            passing.append(arm)

    decision = {
        "name": "step3_pcr_mw_proxy_dev_gate_decision_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "protocol": args.protocol,
        "protocol_sha256": sha256(protocol_path),
        "route": route_for(passing),
        "reference_arm": reference_arm,
        "arm_metrics": arm_metrics,
        "candidate_comparisons": candidates,
        "passing_candidate_arms": passing,
        "interpretation_boundary": protocol["interpretation_boundary"],
        "blind_review_protocol_authorized": bool(passing),
        "main64_training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"route": decision["route"], "passing": passing}))


if __name__ == "__main__":
    main()
