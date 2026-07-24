from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import prepare_step3cr_cr0_protocol as protocol


BASE = protocol.BASE
PACKETS = protocol.PACKETS
REVIEWS = protocol.REVIEWS
REPORTS = protocol.REPORTS
DIAGNOSTICS = protocol.DIAGNOSTICS

AB_FREEZE = REVIEWS / "step3cr_cr0_ab_freeze_manifest.json"
PROVENANCE = REVIEWS / "step3cr_cr0_reviewer_provenance.json"
C_PACKET_MANIFEST = PACKETS / "step3cr_cr0_reviewer_c_packet_manifest.json"
FINAL_DISPLAY = REVIEWS / "step3cr_cr0_final_display_labels.jsonl"
FINAL_CANONICAL = REVIEWS / "step3cr_cr0_final_canonical_labels.jsonl"
AGREEMENT = REPORTS / "step3cr_cr0_review_agreement.json"
BIAS = REPORTS / "step3cr_cr0_bias_and_reversal_report.json"
CATEGORY = REPORTS / "step3cr_cr0_category_summary.json"
ROUTE = REPORTS / "step3cr_cr0_route_decision.json"
ROUTE_REPORT = REPORTS / "Step3CR_CR0_route_report.md"
ARTIFACT_MANIFEST = REPORTS / "step3cr_cr0_review_artifact_manifest.json"
SCRIPT_PATH = protocol.ROOT / "scripts/analyze_step3cr_cr0_reviews.py"
SESSION_ID = "step3cr_cr0_review_session_v1_20260724"


def normalize_lf(path: Path) -> dict[str, Any]:
    before = path.read_bytes()
    after = protocol.canonical_lf_bytes(before)
    if before != after:
        path.write_bytes(after)
    return {
        "path": path.as_posix(),
        "before_sha256": protocol.sha256_bytes(before),
        "canonical_sha256": protocol.sha256_bytes(after),
        "eol_normalized": before != after,
    }


def packet_rows() -> list[dict[str, Any]]:
    return protocol.read_jsonl(protocol.PAIR_PACKET)


def validate_reason(row: dict[str, Any], path: Path) -> None:
    reason = str(row.get("short_reason", "")).strip()
    if not reason:
        raise ValueError(f"{path} empty short_reason for {row['opaque_pair_id']}")
    if len(reason.split()) > 40:
        raise ValueError(f"{path} non-concise short_reason for {row['opaque_pair_id']}")


def validate_review_with_order(path: Path) -> dict[str, dict[str, Any]]:
    packet = packet_rows()
    expected_order = [row["opaque_pair_id"] for row in packet]
    rows = protocol.read_jsonl(path)
    actual_order = [row.get("opaque_pair_id", "") for row in rows]
    if actual_order != expected_order:
        raise ValueError(f"{path} row order does not match frozen packet")
    validated = protocol.validate_review(path, set(expected_order))
    for row in validated.values():
        validate_reason(row, path)
    return validated


def validate_ab() -> dict[str, Any]:
    REVIEWS.mkdir(parents=True, exist_ok=True)
    normalization = [normalize_lf(protocol.REVIEWER_A), normalize_lf(protocol.REVIEWER_B)]
    a = validate_review_with_order(protocol.REVIEWER_A)
    b = validate_review_with_order(protocol.REVIEWER_B)
    provenance = protocol.read_json(PROVENANCE) if PROVENANCE.exists() else {}
    frozen_provenance = {
        **provenance,
        "reviewer_c": None,
    }
    manifest = {
        "name": "step3cr_cr0_ab_freeze_manifest_v1",
        "session_id": SESSION_ID,
        "packet_sha256": protocol.sha256(protocol.PAIR_PACKET),
        "rubric_sha256": protocol.sha256(protocol.RUBRIC),
        "reviewer_a_rows": len(a),
        "reviewer_b_rows": len(b),
        "reviewer_a_sha256": protocol.sha256(protocol.REVIEWER_A),
        "reviewer_b_sha256": protocol.sha256(protocol.REVIEWER_B),
        "canonical_lf_normalization": normalization,
        "provenance": frozen_provenance,
        "ab_labels_frozen_before_c_packet": True,
        "main_executor_assigned_labels": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    protocol.write_json(AB_FREEZE, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def require_ab_frozen() -> dict[str, Any]:
    manifest = protocol.read_json(AB_FREEZE)
    if protocol.sha256(protocol.REVIEWER_A) != manifest["reviewer_a_sha256"]:
        raise ValueError("Reviewer A file changed after A/B freeze")
    if protocol.sha256(protocol.REVIEWER_B) != manifest["reviewer_b_sha256"]:
        raise ValueError("Reviewer B file changed after A/B freeze")
    if protocol.sha256(protocol.PAIR_PACKET) != manifest["packet_sha256"]:
        raise ValueError("pair packet changed after A/B freeze")
    return manifest


def prepare_adjudication() -> dict[str, Any]:
    require_ab_frozen()
    manifest = protocol.prepare_adjudication()
    packet = protocol.read_jsonl(protocol.REVIEWER_C_PACKET)
    allowed = {
        "opaque_pair_id",
        "prompt",
        "left_response",
        "right_response",
        "disagreement_fields",
        "reviewer_a",
        "reviewer_b",
    }
    if any(set(row) != allowed for row in packet):
        raise ValueError("Reviewer C packet has unexpected fields")
    if any(set(row["disagreement_fields"]) - set(protocol.DECISION_FIELDS) for row in packet):
        raise ValueError("Reviewer C packet has unexpected disagreement field")
    return manifest


def canonical_primary(reviews: dict[str, dict[str, Any]], key: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    out = {}
    for opaque_id, key_row in key.items():
        if key_row["is_reversed_duplicate"] == "True":
            continue
        left_is_canonical = protocol.display_left_is_canonical_left(key_row)
        left_specific, right_specific = protocol.canonicalize_specific(reviews[opaque_id], left_is_canonical)
        left_natural, right_natural = protocol.canonicalize_naturalness(reviews[opaque_id], left_is_canonical)
        out[key_row["canonical_pair_id"]] = {
            "opaque_pair_id": opaque_id,
            "preference": protocol.canonicalize_preference(reviews[opaque_id]["pair_preference"], left_is_canonical),
            "canonical_left_specific": left_specific,
            "canonical_right_specific": right_specific,
            "canonical_left_natural": left_natural,
            "canonical_right_natural": right_natural,
            "display_preference": reviews[opaque_id]["pair_preference"],
        }
    return out


def disagreement_rows(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {
        opaque_id: [field for field in protocol.DECISION_FIELDS if a[opaque_id][field] != b[opaque_id][field]]
        for opaque_id in a
        if any(a[opaque_id][field] != b[opaque_id][field] for field in protocol.DECISION_FIELDS)
    }


def validate_c(disputes: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    if not disputes:
        if protocol.REVIEWER_C.exists() and protocol.REVIEWER_C.stat().st_size:
            raise ValueError("Reviewer C labels exist despite zero disagreements")
        return {}
    normalization = normalize_lf(protocol.REVIEWER_C)
    rows = protocol.load_c_labels()
    if set(rows) != set(disputes):
        raise ValueError("Reviewer C IDs do not exactly match disagreement rows")
    for row in rows.values():
        validate_reason(row, protocol.REVIEWER_C)
    manifest = protocol.read_json(C_PACKET_MANIFEST)
    manifest["reviewer_c_rows"] = len(rows)
    manifest["reviewer_c_sha256"] = protocol.sha256(protocol.REVIEWER_C)
    manifest["canonical_lf_normalization"] = normalization
    protocol.write_json(C_PACKET_MANIFEST, manifest)
    return rows


def metric_pairs(
    a_canon: dict[str, dict[str, Any]],
    b_canon: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    canonical_ids = sorted(set(a_canon) & set(b_canon))
    preference = [(a_canon[cid]["preference"], b_canon[cid]["preference"]) for cid in canonical_ids]
    specificity = []
    naturalness = []
    for cid in canonical_ids:
        for side in ["canonical_left", "canonical_right"]:
            specificity.append((a_canon[cid][f"{side}_specific"], b_canon[cid][f"{side}_specific"]))
            naturalness.append((a_canon[cid][f"{side}_natural"], b_canon[cid][f"{side}_natural"]))
    return preference, specificity, naturalness


def label_prevalence(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_preference": dict(Counter(row["pair_preference"] for row in rows.values())),
        "specificity": dict(Counter(
            label
            for row in rows.values()
            for label in [row["left_specific_safe_next_step"], row["right_specific_safe_next_step"]]
        )),
        "naturalness": dict(Counter(
            label
            for row in rows.values()
            for label in [row["left_natural"], row["right_natural"]]
        )),
    }


def canonical_final_rows(
    final: dict[str, dict[str, Any]],
    key: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for opaque_id, key_row in sorted(key.items(), key=lambda item: item[1]["canonical_pair_id"]):
        if key_row["is_reversed_duplicate"] == "True":
            continue
        left_is_canonical = protocol.display_left_is_canonical_left(key_row)
        left_specific, right_specific = protocol.canonicalize_specific(final[opaque_id], left_is_canonical)
        left_natural, right_natural = protocol.canonicalize_naturalness(final[opaque_id], left_is_canonical)
        rows.append({
            "canonical_pair_id": key_row["canonical_pair_id"],
            "opaque_pair_id": opaque_id,
            "pair_preference": protocol.canonicalize_preference(final[opaque_id]["pair_preference"], left_is_canonical),
            "canonical_left_specific_safe_next_step": left_specific,
            "canonical_right_specific_safe_next_step": right_specific,
            "canonical_left_natural": left_natural,
            "canonical_right_natural": right_natural,
            "adjudicated": "yes" if "adjudication_short_reason" in final[opaque_id] else "no",
        })
    return rows


def category_summary(canonical: list[dict[str, Any]], key: dict[str, dict[str, str]]) -> dict[str, Any]:
    category_by_pair = {
        row["canonical_pair_id"]: row["category"]
        for row in key.values()
        if row["is_reversed_duplicate"] == "False"
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in canonical:
        grouped.setdefault(category_by_pair[row["canonical_pair_id"]], []).append(row)
    return {
        category: {
            "pairs": len(rows),
            "preference": dict(Counter(row["pair_preference"] for row in rows)),
            "both_specific": sum(
                row["canonical_left_specific_safe_next_step"] == "yes"
                and row["canonical_right_specific_safe_next_step"] == "yes"
                for row in rows
            ),
        }
        for category, rows in sorted(grouped.items())
    }


def review_artifact_paths() -> list[Path]:
    paths = [
        protocol.REGISTRY,
        protocol.RUBRIC,
        protocol.PAIR_PACKET,
        protocol.PAIR_KEY,
        protocol.DUPLICATE_KEY,
        protocol.PROTOCOL_MANIFEST,
        SCRIPT_PATH,
        PROVENANCE,
        AB_FREEZE,
        protocol.REVIEWER_A,
        protocol.REVIEWER_B,
        protocol.REVIEWER_C_PACKET,
        C_PACKET_MANIFEST,
        protocol.REVIEWER_C,
        FINAL_DISPLAY,
        FINAL_CANONICAL,
        AGREEMENT,
        BIAS,
        CATEGORY,
        ROUTE,
        ROUTE_REPORT,
    ]
    return [path for path in paths if path.exists() and path != ARTIFACT_MANIFEST]


def summarize() -> dict[str, Any]:
    ab_manifest = require_ab_frozen()
    packet = packet_rows()
    expected = {row["opaque_pair_id"] for row in packet}
    a = validate_review_with_order(protocol.REVIEWER_A)
    b = validate_review_with_order(protocol.REVIEWER_B)
    disputes = disagreement_rows(a, b)
    c = validate_c(disputes)
    final = protocol.final_from_reviews(a, b, c)

    protocol.write_jsonl(FINAL_DISPLAY, [final[row["opaque_pair_id"]] for row in packet])
    final_display_sha = protocol.sha256(FINAL_DISPLAY)
    key = protocol.load_pair_key()
    canonical = canonical_final_rows(final, key)
    if len(canonical) != 40:
        raise ValueError("final canonical outcome denominator must be 40")
    protocol.write_jsonl(FINAL_CANONICAL, canonical)
    final_canonical_sha = protocol.sha256(FINAL_CANONICAL)

    a_canon = canonical_primary(a, key)
    b_canon = canonical_primary(b, key)
    preference_pairs, specificity_pairs, naturalness_pairs = metric_pairs(a_canon, b_canon)
    a_reverse = protocol.reversed_consistency(a, key)
    b_reverse = protocol.reversed_consistency(b, key)
    a_bias = protocol.position_bias(a, key)
    b_bias = protocol.position_bias(b, key)
    registry = protocol.read_json(protocol.REGISTRY)
    gate = registry["cr0_measurement"]
    preference_metrics = protocol.metric_block(preference_pairs)
    specificity_metrics = protocol.metric_block(specificity_pairs)
    naturalness_metrics = protocol.metric_block(naturalness_pairs)
    disagreement_types = Counter(field for fields in disputes.values() for field in fields)
    gates = {
        "pair_preference_agreement_pass": preference_metrics["raw_agreement"] >= float(gate["raw_agreement_min"])
        and preference_metrics["gwet_ac1"] >= float(gate["gwet_ac1_min"]),
        "specificity_agreement_pass": specificity_metrics["raw_agreement"] >= float(gate["specific_safe_raw_agreement_min"])
        and specificity_metrics["gwet_ac1"] >= float(gate["specific_safe_gwet_ac1_min"]),
        "reviewer_a_preference_reversal_pass": protocol.reversal_gate_pass(a_reverse["preference_consistent_count"], a_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_b_preference_reversal_pass": protocol.reversal_gate_pass(b_reverse["preference_consistent_count"], b_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_a_specificity_reversal_pass": protocol.reversal_gate_pass(a_reverse["specificity_consistent_count"], a_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_b_specificity_reversal_pass": protocol.reversal_gate_pass(b_reverse["specificity_consistent_count"], b_reverse["denominator"], int(gate["reversed_consistency_min_count"])),
        "reviewer_a_non_tie_pass": a_bias["non_tie_count"] >= int(gate["canonical_non_tie_count_min_per_reviewer"]),
        "reviewer_b_non_tie_pass": b_bias["non_tie_count"] >= int(gate["canonical_non_tie_count_min_per_reviewer"]),
        "reviewer_a_position_bias_pass": a_bias["absolute_left_right_share_imbalance"] is not None
        and a_bias["absolute_left_right_share_imbalance"] <= float(gate["absolute_position_imbalance_max"]),
        "reviewer_b_position_bias_pass": b_bias["absolute_left_right_share_imbalance"] is not None
        and b_bias["absolute_left_right_share_imbalance"] <= float(gate["absolute_position_imbalance_max"]),
        "all_disagreements_adjudicated": set(c) == set(disputes),
    }
    gates["overall_pass"] = all(gates.values())
    agreement = {
        "name": "step3cr_cr0_review_agreement_v1",
        "session_id": SESSION_ID,
        "review_rows": len(packet),
        "canonical_outcome_pairs": 40,
        "duplicate_qc_rows": 8,
        "pair_preference": preference_metrics,
        "specific_safe_next_step": specificity_metrics,
        "naturalness_sensitivity": naturalness_metrics,
        "reviewer_a_label_prevalence": label_prevalence(a),
        "reviewer_b_label_prevalence": label_prevalence(b),
        "decision_field_disagreement_rows": len(disputes),
        "disagreement_type_counts": dict(disagreement_types),
        "reviewer_c_rows": len(c),
        "reviewer_c_coverage": len(c) / len(disputes) if disputes else 1.0,
        "gates": gates,
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    protocol.write_json(AGREEMENT, agreement)
    bias = {
        "name": "step3cr_cr0_bias_and_reversal_v1",
        "reviewer_a_reversal": a_reverse,
        "reviewer_b_reversal": b_reverse,
        "reviewer_a_position_bias": a_bias,
        "reviewer_b_position_bias": b_bias,
        "naturalness_reversal_is_descriptive_only": True,
    }
    protocol.write_json(BIAS, bias)
    categories = {
        "name": "step3cr_cr0_category_summary_v1",
        "descriptive_only": True,
        "categories": category_summary(canonical, key),
    }
    protocol.write_json(CATEGORY, categories)
    route = {
        "name": "step3cr_cr0_route_decision_v1",
        "session_id": SESSION_ID,
        "cr0_measurement_gate_pass": gates["overall_pass"],
        "complete_revision_used": False,
        "next_authorized_package": "P2_C-R1_no_update_diagnostics" if gates["overall_pass"] else "one_complete_rubric_clarification_and_fresh_review_session",
        "final_display_sha256": final_display_sha,
        "final_canonical_sha256": final_canonical_sha,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    protocol.write_json(ROUTE, route)
    report = f"""# Step3C-R C-R0 Review and Route Report

Status: {'pass' if gates['overall_pass'] else 'first-session gate failure'}.

## Review Isolation

- Session: `{SESSION_ID}`
- Reviewer A rows: {ab_manifest['reviewer_a_rows']}
- Reviewer B rows: {ab_manifest['reviewer_b_rows']}
- Reviewer C disagreement rows: {len(c)}
- Canonical outcome denominator: 40
- Reversed duplicate QC rows: 8
- Main executor assigned labels: false

## Reliability

| construct | n | raw agreement | Gwet AC1 | Cohen kappa | pass |
| --- | ---: | ---: | ---: | ---: | --- |
| pair preference | {preference_metrics['n']} | {preference_metrics['raw_agreement']:.3f} | {preference_metrics['gwet_ac1']:.3f} | {preference_metrics['cohen_kappa'] if preference_metrics['cohen_kappa'] is not None else 'undefined'} | {gates['pair_preference_agreement_pass']} |
| response-side specificity | {specificity_metrics['n']} | {specificity_metrics['raw_agreement']:.3f} | {specificity_metrics['gwet_ac1']:.3f} | {specificity_metrics['cohen_kappa'] if specificity_metrics['cohen_kappa'] is not None else 'undefined'} | {gates['specificity_agreement_pass']} |

## Reversal and Position Gates

- Reviewer A preference/specificity consistency: {a_reverse['preference_consistent_count']}/8, {a_reverse['specificity_consistent_count']}/8
- Reviewer B preference/specificity consistency: {b_reverse['preference_consistent_count']}/8, {b_reverse['specificity_consistent_count']}/8
- Reviewer A non-tie count and position imbalance: {a_bias['non_tie_count']}, {a_bias['absolute_left_right_share_imbalance']}
- Reviewer B non-tie count and position imbalance: {b_bias['non_tie_count']}, {b_bias['absolute_left_right_share_imbalance']}
- All disagreement rows adjudicated: {gates['all_disagreements_adjudicated']}

## Route

Overall C-R0 gate: **{'pass' if gates['overall_pass'] else 'fail'}**.

Next authorized package: `{route['next_authorized_package']}`.

Heldout remained closed. No optimizer or scheduler was created, and no training was run.
"""
    ROUTE_REPORT.write_text(report, encoding="utf-8", newline="\n")

    artifacts = [
        {"path": path.as_posix(), "sha256": protocol.sha256(path), "bytes": path.stat().st_size}
        for path in review_artifact_paths()
    ]
    manifest = {
        "name": "step3cr_cr0_review_artifact_manifest_v1",
        "session_id": SESSION_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "all_canonical_lf": all(b"\r" not in path.read_bytes() for path in review_artifact_paths()),
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    protocol.write_json(ARTIFACT_MANIFEST, manifest)
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate-ab", "prepare-adjudication", "summarize"])
    args = parser.parse_args()
    if args.command == "validate-ab":
        validate_ab()
    elif args.command == "prepare-adjudication":
        prepare_adjudication()
    elif args.command == "summarize":
        summarize()


if __name__ == "__main__":
    main()
