"""Run Step3C-T5R review freeze, disagreement adjudication, and reliability."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import step3ct_t5r_common as common


REGISTRY_PATH = common.REPO / "experiments/step3_response_quality_persistence/configs/step3ct_t5r_review_recovery_registry.json"
T4_ROOT = common.REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t4"
T5_ROOT = common.REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5"
T5R_ROOT = common.REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5r"
DOMAINS = ("safety", "quality", "utility")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze-safety-ab", "freeze-quality-utility-ab", "prepare-c", "finalize", "record-access-stop"))
    parser.add_argument("--reviewer-role")
    parser.add_argument("--reviewer-id")
    parser.add_argument("--session-id")
    parser.add_argument("--reason-code")
    parser.add_argument("--partial-path")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.REPO, text=True).strip()


def registry() -> dict[str, Any]:
    return common.read_json(REGISTRY_PATH)


def packet_spec(domain: str) -> dict[str, Any]:
    return registry()["packets"][domain]


def packet_path(domain: str) -> Path:
    return common.REPO / packet_spec(domain)["packet_path"]


def t4_template_path(domain: str, reviewer: str) -> Path:
    return packet_path(domain).parent / f"step3ct_t4_{domain}_reviewer_{reviewer}_template.jsonl"


def rubric_path(domain: str) -> Path:
    return common.REPO / packet_spec(domain)["rubric_path"]


def review_root(domain: str) -> Path:
    return T5R_ROOT / "reviews" / domain


def label_path(domain: str, reviewer: str) -> Path:
    return review_root(domain) / f"step3ct_t5r_{domain}_reviewer_{reviewer}_labels.jsonl"


def provenance_path(domain: str, reviewer: str) -> Path:
    return review_root(domain) / f"step3ct_t5r_{domain}_reviewer_{reviewer}_provenance.json"


def c_packet_path(domain: str) -> Path:
    return T5R_ROOT / "packets" / domain / f"step3ct_t5r_{domain}_reviewer_c_packet.jsonl"


def c_template_path(domain: str) -> Path:
    return T5R_ROOT / "packets" / domain / f"step3ct_t5r_{domain}_reviewer_c_template.jsonl"


def failed_t5_session_ids() -> set[str]:
    incident = common.read_json(T5_ROOT / "step3ct_t5_safety_gateway_incident.json")
    ids = {str(incident["failed_required_session"]["agent_id"])}
    ids.update(str(row["agent_id"]) for row in incident["stopped_sessions"])
    return ids


def validate_reviewer(domain: str, reviewer: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    template = t4_template_path(domain, reviewer) if reviewer in {"a", "b"} else c_template_path(domain)
    output = label_path(domain, reviewer)
    rows = common.read_jsonl(output)
    common.validate_review_rows(domain, rows, common.read_jsonl(template))
    provenance = common.read_json(provenance_path(domain, reviewer))
    common.validate_provenance(
        provenance,
        domain=domain,
        reviewer_role=f"{domain}_{reviewer}",
        packet_path=packet_path(domain) if reviewer in {"a", "b"} else c_packet_path(domain),
        template_path=template,
        rubric_path=rubric_path(domain),
        output_path=output,
        output_rows=len(rows),
        prohibited_session_ids=failed_t5_session_ids(),
    )
    return rows, provenance


def reviewer_record(domain: str, reviewer: str, rows: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewer_role": f"{domain}_{reviewer}",
        "anonymized_reviewer_id": provenance["anonymized_reviewer_id"],
        "reviewer_type": provenance["reviewer_type"],
        "model_provider": provenance.get("model_provider"),
        "model_id_version": provenance.get("model_id_version"),
        "session_id": provenance["session_id"],
        "labels_path": label_path(domain, reviewer).relative_to(common.REPO).as_posix(),
        "labels_sha256": common.sha256_file(label_path(domain, reviewer)),
        "provenance_path": provenance_path(domain, reviewer).relative_to(common.REPO).as_posix(),
        "provenance_sha256": common.sha256_file(provenance_path(domain, reviewer)),
        "rows": len(rows),
    }


def ensure_independent(records: list[dict[str, Any]]) -> None:
    reviewer_ids = [str(row["anonymized_reviewer_id"]) for row in records]
    session_ids = [str(row["session_id"]) for row in records]
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValueError("reviewer identities are not independent")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("reviewer sessions are not independent")


def write_freeze(path: Path, name: str, domains: dict[str, Any], source: dict[str, Any] | None = None) -> None:
    records = [reviewer for value in domains.values() for reviewer in value["reviewers"].values()]
    ensure_independent(records)
    common.write_json(path, {
        "name": name,
        "execution_git_head": git_head(),
        "registry_sha256": common.sha256_file(REGISTRY_PATH),
        "source_freeze": source,
        "domains": domains,
        "labels_compared_before_freeze": False,
        "outcome_key_joined": False,
        "main_executor_assigned_or_edited_labels": False,
        "heldout_opened": False,
    })


def freeze_safety_ab() -> None:
    domain = "safety"
    reviewers = {}
    for reviewer in ("a", "b"):
        rows, provenance = validate_reviewer(domain, reviewer)
        reviewers[reviewer] = reviewer_record(domain, reviewer, rows, provenance)
    output = T5R_ROOT / "freezes" / "step3ct_t5r_safety_ab_freeze.json"
    write_freeze(output, "step3ct_t5r_safety_ab_freeze_v1", {domain: {"reviewers": reviewers}})
    print(json.dumps({"route": "P2_safety_ab_complete_frozen_authorize_P3", "rows": {key: value["rows"] for key, value in reviewers.items()}}, indent=2))


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = common.read_json(path)
    for domain, domain_record in freeze["domains"].items():
        for reviewer, record in domain_record["reviewers"].items():
            if common.sha256_file(label_path(domain, reviewer)) != record["labels_sha256"]:
                raise ValueError(f"{domain} reviewer {reviewer} labels changed after freeze")
            if common.sha256_file(provenance_path(domain, reviewer)) != record["provenance_sha256"]:
                raise ValueError(f"{domain} reviewer {reviewer} provenance changed after freeze")
    return freeze


def freeze_quality_utility_ab() -> None:
    safety_path = T5R_ROOT / "freezes" / "step3ct_t5r_safety_ab_freeze.json"
    safety = verify_freeze(safety_path)
    domains = dict(safety["domains"])
    for domain in ("quality", "utility"):
        reviewers = {}
        for reviewer in ("a", "b"):
            rows, provenance = validate_reviewer(domain, reviewer)
            reviewers[reviewer] = reviewer_record(domain, reviewer, rows, provenance)
        domains[domain] = {"reviewers": reviewers}
    output = T5R_ROOT / "freezes" / "step3ct_t5r_all_ab_freeze.json"
    write_freeze(
        output,
        "step3ct_t5r_all_ab_freeze_v1",
        domains,
        {"path": safety_path.relative_to(common.REPO).as_posix(), "sha256": common.sha256_file(safety_path)},
    )
    print(json.dumps({domain: {reviewer: domains[domain]["reviewers"][reviewer]["rows"] for reviewer in ("a", "b")} for domain in DOMAINS}, indent=2))


def prepare_c() -> None:
    ab_path = T5R_ROOT / "freezes" / "step3ct_t5r_all_ab_freeze.json"
    ab_freeze = verify_freeze(ab_path)
    summary: dict[str, Any] = {}
    for domain in DOMAINS:
        spec = common.DOMAIN_SPECS[domain]
        id_field = spec["id_field"]
        packet_rows = common.read_jsonl(packet_path(domain))
        a_rows, _ = validate_reviewer(domain, "a")
        b_rows, _ = validate_reviewer(domain, "b")
        c_packet, c_template = common.build_disagreement_packet(domain, packet_rows, a_rows, b_rows)
        common.write_jsonl(c_packet_path(domain), c_packet)
        common.write_jsonl(c_template_path(domain), c_template)
        manifest = {
            "name": f"step3ct_t5r_{domain}_reviewer_c_packet_v1",
            "domain": domain,
            "source_ab_freeze_sha256": common.sha256_file(ab_path),
            "packet_rows": len(c_packet),
            "decision_field_disagreements": sum(len(row["disagreement_fields"]) for row in c_packet),
            "packet_sha256": common.sha256_file(c_packet_path(domain)),
            "template_sha256": common.sha256_file(c_template_path(domain)),
            "a_b_labels_reasons_identities_or_votes_visible": False,
            "arm_checkpoint_sample_or_duplicate_link_visible": False,
            "heldout_opened": False,
        }
        manifest_path = c_packet_path(domain).with_name(f"step3ct_t5r_{domain}_reviewer_c_packet_manifest.json")
        common.write_json(manifest_path, manifest)
        summary[domain] = manifest
    freeze_path = T5R_ROOT / "freezes" / "step3ct_t5r_c_packet_freeze.json"
    common.write_json(freeze_path, {
        "name": "step3ct_t5r_c_packet_freeze_v1",
        "execution_git_head": git_head(),
        "source_ab_freeze_sha256": common.sha256_file(ab_path),
        "domains": summary,
        "a_b_labels_or_reasons_visible": False,
        "outcome_key_joined": False,
        "heldout_opened": False,
    })
    print(json.dumps({domain: summary[domain]["packet_rows"] for domain in DOMAINS}, indent=2))


def primary_ids(domain: str, packet_rows: list[dict[str, Any]]) -> list[str]:
    id_field = common.DOMAIN_SPECS[domain]["id_field"]
    duplicate_rows = common.read_duplicate_pairs(packet_path(domain).parent / f"step3ct_t4_{domain}_duplicate_map.csv")
    repeated_ids = {row["duplicate_opaque_id"] for row in duplicate_rows}
    return [str(row[id_field]) for row in packet_rows if str(row[id_field]) not in repeated_ids]


def safety_reliability(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], ids: list[str], gates: dict[str, Any]) -> dict[str, Any]:
    transforms = {
        "unsafe_with_mixed": lambda row: "yes" if row["primary_label"] in {"unsafe_compliance", "mixed_refusal_compliance"} else "no",
        "mixed": lambda row: row["mixed_response"],
        "invalid": lambda row: "yes" if row["primary_label"] == "invalid_or_irrelevant" else "no",
        "prompt_relevant_redirect": lambda row: row["prompt_relevant_safe_redirect"],
    }
    result = {}
    for name, transform in transforms.items():
        metric = common.nominal_metrics([(transform(a[opaque_id]), transform(b[opaque_id])) for opaque_id in ids], ["yes", "no"])
        if name == "unsafe_with_mixed":
            passed = common.nominal_gate(metric, gates["unsafe_with_mixed_raw_agreement_min"], gates["unsafe_with_mixed_gwet_ac1_min"], registry()["reliability_gates"]["zero_variance"]["raw_agreement_min"])
        else:
            passed = common.nominal_gate(metric, gates["secondary_field_raw_agreement_min"], gates["secondary_field_gwet_ac1_min"], registry()["reliability_gates"]["zero_variance"]["raw_agreement_min"])
        result[name] = {**metric, "pass": passed}
    return result


def quality_reliability(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], ids: list[str], gates: dict[str, Any]) -> dict[str, Any]:
    zero_raw = registry()["reliability_gates"]["zero_variance"]["raw_agreement_min"]
    ordinal = {}
    for field in ("detail_retention", "safe_actionability"):
        metric = common.ordinal_metrics([(a[opaque_id][field], b[opaque_id][field]) for opaque_id in ids])
        chance_pass = metric["exact_raw_agreement"] >= zero_raw if metric["zero_variance"] else metric["gwet_ac2"] is not None and metric["gwet_ac2"] >= gates["ordinal_gwet_ac2_min"]
        metric["pass"] = metric["linear_weighted_agreement"] >= gates["ordinal_linear_weighted_agreement_min"] and metric["two_level_disagreement_rate"] <= gates["two_level_disagreement_max"] and chance_pass
        ordinal[field] = metric
    derived = common.nominal_metrics([(common.quality_eligible(a[opaque_id]), common.quality_eligible(b[opaque_id])) for opaque_id in ids], ["yes", "no"])
    derived["pass"] = common.nominal_gate(derived, gates["derived_eligibility_raw_agreement_min"], gates["derived_eligibility_gwet_ac1_min"], zero_raw)
    naturalness = common.nominal_metrics([(a[opaque_id]["naturalness"], b[opaque_id]["naturalness"]) for opaque_id in ids], ["yes", "no"])
    naturalness["pass"] = naturalness["raw_agreement"] >= (zero_raw if naturalness["zero_variance"] else gates["naturalness_raw_agreement_min"])
    return {"ordinal": ordinal, "derived_eligibility": derived, "naturalness": naturalness}


def utility_reliability(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], ids: list[str], gates: dict[str, Any]) -> dict[str, Any]:
    zero_raw = registry()["reliability_gates"]["zero_variance"]["raw_agreement_min"]
    result = {}
    for field in ("task_success", "unnecessary_refusal", "invalid_or_irrelevant", "material_degradation", "format_failure"):
        metric = common.nominal_metrics([(a[opaque_id][field], b[opaque_id][field]) for opaque_id in ids], ["yes", "no"])
        metric["pass"] = common.nominal_gate(metric, gates["per_field_raw_agreement_min"], gates["per_field_gwet_ac1_min"], zero_raw)
        metric["registered_gate_field"] = field != "format_failure"
        result[field] = metric
    return result


def finalize() -> None:
    ab_path = T5R_ROOT / "freezes" / "step3ct_t5r_all_ab_freeze.json"
    c_freeze_path = T5R_ROOT / "freezes" / "step3ct_t5r_c_packet_freeze.json"
    ab_freeze = verify_freeze(ab_path)
    c_freeze = common.read_json(c_freeze_path)
    reports = {}
    all_pass = True
    adjudication_sources: dict[str, dict[str, Any]] = {}
    ab_records = [reviewer for domain in ab_freeze["domains"].values() for reviewer in domain["reviewers"].values()]
    for domain in DOMAINS:
        spec = common.DOMAIN_SPECS[domain]
        id_field = spec["id_field"]
        packet_rows = common.read_jsonl(packet_path(domain))
        ids = primary_ids(domain, packet_rows)
        a_rows, _ = validate_reviewer(domain, "a")
        b_rows, _ = validate_reviewer(domain, "b")
        a_map = common.by_id(domain, a_rows)
        b_map = common.by_id(domain, b_rows)
        c_packet = common.read_jsonl(c_packet_path(domain))
        if common.sha256_file(c_packet_path(domain)) != c_freeze["domains"][domain]["packet_sha256"]:
            raise ValueError(f"{domain} Reviewer C packet changed after freeze")
        if c_packet:
            c_rows, c_provenance = validate_reviewer(domain, "c")
            c_record = reviewer_record(domain, "c", c_rows, c_provenance)
            if c_record["session_id"] in {record["session_id"] for record in ab_records} or c_record["anonymized_reviewer_id"] in {record["anonymized_reviewer_id"] for record in ab_records}:
                raise ValueError(f"{domain} Reviewer C is not independent from A/B")
        else:
            c_rows, c_record = [], None
        c_map = common.by_id(domain, c_rows)
        expected_c = {str(row[id_field]) for row in c_packet}
        if set(c_map) != expected_c:
            raise ValueError(f"{domain} Reviewer C coverage mismatch")
        duplicate_pairs = common.read_duplicate_pairs(packet_path(domain).parent / f"step3ct_t4_{domain}_duplicate_map.csv")
        a_duplicate = common.duplicate_consistency(domain, a_rows, duplicate_pairs)
        b_duplicate = common.duplicate_consistency(domain, b_rows, duplicate_pairs)
        domain_gates = registry()["reliability_gates"][domain]
        duplicate_pass = a_duplicate["full_vector_consistency"] >= domain_gates["duplicate_full_vector_consistency_min"] and b_duplicate["full_vector_consistency"] >= domain_gates["duplicate_full_vector_consistency_min"]
        if domain == "safety":
            metrics = safety_reliability(a_map, b_map, ids, domain_gates)
            metric_pass = all(value["pass"] for value in metrics.values())
        elif domain == "quality":
            metrics = quality_reliability(a_map, b_map, ids, domain_gates)
            metric_pass = all(value["pass"] for value in metrics["ordinal"].values()) and metrics["derived_eligibility"]["pass"] and metrics["naturalness"]["pass"]
        else:
            metrics = utility_reliability(a_map, b_map, ids, domain_gates)
            metric_pass = all(value["pass"] for value in metrics.values() if value["registered_gate_field"])
        coverage_pass = len(c_rows) == len(expected_c)
        domain_pass = metric_pass and duplicate_pass and coverage_pass
        all_pass &= domain_pass
        adjudication_sources[domain] = {
            "packet_rows": packet_rows,
            "a_map": a_map,
            "b_map": b_map,
            "c_map": c_map,
        }
        reports[domain] = {
            "primary_rows": len(ids),
            "hidden_duplicate_pairs": len(duplicate_pairs),
            "metrics": metrics,
            "reviewer_a_duplicate_consistency": a_duplicate,
            "reviewer_b_duplicate_consistency": b_duplicate,
            "disagreement_rows": len(expected_c),
            "reviewer_c_rows": len(c_rows),
            "reviewer_c_record": c_record,
            "all_decision_disagreements_adjudicated": coverage_pass,
            "final_labels_path": None,
            "final_labels_sha256": None,
            "overall_pass": domain_pass,
        }
    if all_pass:
        for domain in DOMAINS:
            spec = common.DOMAIN_SPECS[domain]
            id_field = spec["id_field"]
            source = adjudication_sources[domain]
            final_rows = []
            for packet in source["packet_rows"]:
                opaque_id = str(packet[id_field])
                final = {id_field: opaque_id}
                disputed = []
                for field in spec["decision_fields"]:
                    if source["a_map"][opaque_id][field] == source["b_map"][opaque_id][field]:
                        final[field] = source["a_map"][opaque_id][field]
                    else:
                        final[field] = source["c_map"][opaque_id][field]
                        disputed.append(field)
                reason_source = source["c_map"][opaque_id] if disputed else source["a_map"][opaque_id]
                for field in spec["reason_fields"]:
                    final[field] = reason_source[field]
                final["adjudicated_fields"] = disputed
                final_rows.append(final)
            final_path = review_root(domain) / f"step3ct_t5r_{domain}_final_labels.jsonl"
            common.write_jsonl(final_path, final_rows)
            reports[domain]["final_labels_path"] = final_path.relative_to(common.REPO).as_posix()
            reports[domain]["final_labels_sha256"] = common.sha256_file(final_path)
    route = "T5R4_complete_labels_authorize_T6" if all_pass else "T5R3_review_reliability_stop"
    report_path = T5R_ROOT / "reports" / "step3ct_t5r_review_reliability.json"
    common.write_json(report_path, {
        "name": "step3ct_t5r_review_reliability_v1",
        "execution_git_head": git_head(),
        "source_ab_freeze_sha256": common.sha256_file(ab_path),
        "source_c_freeze_sha256": common.sha256_file(c_freeze_path),
        "domains": reports,
        "overall_pass": all_pass,
        "route": route,
        "reliability_computed_before_adjudication_and_key_join": True,
        "adjudication_written_only_after_all_reliability_gates_pass": all_pass,
        "outcome_key_joined": False,
        "main_executor_assigned_or_edited_labels": False,
        "heldout_opened": False,
        "new_training_authorized": False,
    })
    common.write_json(T5R_ROOT / "step3ct_t5r_route_decision.json", {
        "name": "step3ct_t5r_route_decision_v1",
        "route": route,
        "overall_pass": all_pass,
        "next_authorized_package": "Step3C_T6_statistics_and_route_decision" if all_pass else "stop_before_outcome_key_join",
        "outcome_key_join_authorized": all_pass,
        "new_training_authorized": False,
        "heldout_opened": False,
    })
    print(json.dumps({"route": route, "domains": {domain: reports[domain]["overall_pass"] for domain in DOMAINS}}, indent=2))


def record_access_stop(args: argparse.Namespace) -> None:
    required = [args.reviewer_role, args.reviewer_id, args.session_id, args.reason_code]
    if any(not value for value in required):
        raise ValueError("record-access-stop requires reviewer role/id, session id, and reason code")
    partial = None
    if args.partial_path:
        path = common.REPO / args.partial_path
        if path.exists():
            partial = {
                "path": args.partial_path,
                "sha256": common.sha256_file(path),
                "bytes": path.stat().st_size,
                "accepted": False,
                "content_read_by_executor": False,
            }
    incident = {
        "name": "step3ct_t5r_reviewer_access_stop_v1",
        "execution_git_head": git_head(),
        "route": "T5R1_reviewer_access_stop",
        "reviewer_role": args.reviewer_role,
        "anonymized_reviewer_id": args.reviewer_id,
        "session_id": args.session_id,
        "reason_code": args.reason_code,
        "partial_output": partial,
        "accepted_label_files": 0,
        "accepted_rows": 0,
        "quality_or_utility_started": False,
        "gateway_circumvention": False,
        "packet_transformed": False,
        "local_heuristic_substituted": False,
        "outcome_key_joined": False,
        "heldout_opened": False,
        "new_training_authorized": False,
        "next_authorized_action": "stop_track_a_preserve_t4_and_continue_only_independent_track_b_if_desired",
    }
    common.write_json(T5R_ROOT / "incidents" / "step3ct_t5r_reviewer_access_stop.json", incident)
    print(json.dumps({"route": incident["route"], "accepted_rows": 0}, indent=2))


def main() -> None:
    args = parse_args()
    if args.action == "freeze-safety-ab":
        freeze_safety_ab()
    elif args.action == "freeze-quality-utility-ab":
        freeze_quality_utility_ab()
    elif args.action == "prepare-c":
        prepare_c()
    elif args.action == "finalize":
        finalize()
    else:
        record_access_stop(args)


if __name__ == "__main__":
    main()
