"""Freeze three isolated Reviewer C sessions before disagreement-packet access."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import step3ct_t5r_common as common


ROOT = common.REPO / "experiments/step3_response_quality_persistence"
CONFIG_PATH = ROOT / "configs/step3ct_t5h_reviewer_c_model_registry.json"
T5_INCIDENT_PATH = ROOT / "results/4070ti_step3ct_seed1/t5/step3ct_t5_safety_gateway_incident.json"
H4Q_ROOT = ROOT / "results/4070ti_step3ct_seed1/t5h/h4q"
QUALIFICATION_PATH = H4Q_ROOT / "step3ct_t5h_h4q_model_reviewer_qualification.json"
ROUTE_PATH = H4Q_ROOT / "step3ct_t5h_h4q_route_decision.json"
EXECUTION_PATH = H4Q_ROOT / "step3ct_t5h_h4q_execution_record.json"
REPORT_PATH = H4Q_ROOT / "Step3CT_T5H_H4Q_model_reviewer_qualification_report.md"
PACKAGE_PATH = H4Q_ROOT / "step3ct_t5h_h4q_package_manifest.json"

H3_ROUTE = "H3A1_all_ab_frozen_authorize_disagreement_c"
QUALIFIED_ROUTE = "H4Q1_three_model_sessions_qualified_for_disagreement_c"
FAILURE_ROUTE = "T5R2_review_completion_or_schema_stop"
DOMAINS = ("safety", "quality", "utility")
ROLES = tuple(f"{domain}_c" for domain in DOMAINS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.relative_to(common.REPO).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": common.sha256_file(path), "bytes": path.stat().st_size}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=common.REPO, text=True).strip()


def nonblank_line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def prohibited_identities(incident: dict[str, Any], ab_freeze: dict[str, Any]) -> tuple[set[str], set[str]]:
    sessions = {str(incident["failed_required_session"]["agent_id"])}
    sessions.update(str(row["agent_id"]) for row in incident["stopped_sessions"])
    reviewer_ids: set[str] = set()
    for domain in DOMAINS:
        for record in ab_freeze["domains"][domain]["reviewers"].values():
            sessions.add(str(record["session_id"]))
            reviewer_ids.add(str(record["anonymized_reviewer_id"]))
    return sessions, reviewer_ids


def validate_reviewer(
    role: str,
    reviewer: dict[str, Any],
    prohibited_sessions: set[str],
    prohibited_reviewers: set[str],
) -> None:
    if reviewer.get("reviewer_role") != role:
        raise ValueError(f"reviewer {role} role mismatch")
    if reviewer.get("reviewer_type") != "policy_compliant_model_session":
        raise ValueError(f"reviewer {role} type mismatch")
    required_text = (
        "anonymized_reviewer_id",
        "session_id",
        "model_provider",
        "runtime_requested_model_id",
        "self_reported_model_id",
        "model_id_version_for_final_provenance",
    )
    if any(not reviewer.get(field) for field in required_text):
        raise ValueError(f"reviewer {role} missing identity or model provenance")
    if reviewer.get("runtime_requested_model_id") != "gpt-5.6-terra":
        raise ValueError(f"reviewer {role} runtime selection mismatch")
    if reviewer.get("fork_context") is not False:
        raise ValueError(f"reviewer {role} inherited context")
    if str(reviewer["session_id"]) in prohibited_sessions:
        raise ValueError(f"reviewer {role} reuses a prohibited session")
    if str(reviewer["anonymized_reviewer_id"]) in prohibited_reviewers:
        raise ValueError(f"reviewer {role} reuses a prohibited reviewer identity")
    confirmations = reviewer.get("qualification_confirmations", {})
    required_false = ("prior_study_involvement", "repository_access_so_far", "other_reviewer_contact")
    required_true = (
        "agrees_independent_complete_review",
        "agrees_no_partial_output",
        "agrees_no_external_retrieval",
    )
    if any(confirmations.get(field) is not False for field in required_false):
        raise ValueError(f"reviewer {role} failed zero-access independence qualification")
    if confirmations.get("study_rows_accessed") != 0:
        raise ValueError(f"reviewer {role} accessed study rows before qualification freeze")
    if any(confirmations.get(field) is not True for field in required_true):
        raise ValueError(f"reviewer {role} did not accept the complete-or-abort contract")


def validate_materials(
    domain: str,
    material: dict[str, Any],
    c_freeze: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    frozen = c_freeze["domains"][domain]
    expected = {
        "packet_sha256": frozen["packet_sha256"],
        "packet_rows": frozen["packet_rows"],
        "decision_field_disagreements": frozen["decision_field_disagreements"],
        "template_sha256": frozen["template_sha256"],
    }
    for field, value in expected.items():
        if material.get(field) != value:
            raise ValueError(f"{domain} material {field} differs from C packet freeze")

    paths = {
        "packet": common.REPO / material["packet_path"],
        "template": common.REPO / material["template_path"],
        "manifest": common.REPO / material["manifest_path"],
        "rubric": common.REPO / material["rubric_path"],
    }
    expected_hashes = {
        "packet": material["packet_sha256"],
        "template": material["template_sha256"],
        "manifest": material["manifest_sha256"],
        "rubric": material["rubric_sha256"],
    }
    records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file() or common.sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{domain} frozen material mismatch: {name}")
        records[name] = {"path": relative(path), **artifact_record(path)}
    if nonblank_line_count(paths["packet"]) != int(material["packet_rows"]):
        raise ValueError(f"{domain} packet row count mismatch")
    if nonblank_line_count(paths["template"]) != int(material["packet_rows"]):
        raise ValueError(f"{domain} template row count mismatch")
    manifest = read_json(paths["manifest"])
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"{domain} manifest {field} mismatch")
    if manifest.get("a_b_labels_reasons_identities_or_votes_visible") is not False:
        raise ValueError(f"{domain} manifest exposes A/B information")
    if manifest.get("arm_checkpoint_sample_or_duplicate_link_visible") is not False:
        raise ValueError(f"{domain} manifest exposes hidden identities")
    if manifest.get("heldout_opened") is not False:
        raise ValueError(f"{domain} manifest opened heldout")
    return records


def validate_entry() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    h3_path = common.REPO / config["parent_h3_route"]["path"]
    ab_path = common.REPO / config["source_ab_freeze"]["path"]
    c_freeze_path = common.REPO / config["c_packet_freeze"]["path"]
    h3 = read_json(h3_path)
    ab_freeze = read_json(ab_path)
    c_freeze = read_json(c_freeze_path)
    incident = read_json(T5_INCIDENT_PATH)

    if h3.get("route") != H3_ROUTE or h3.get("disagreement_c_authorized") is not True:
        raise ValueError("H4Q requires H3 disagreement-only C authorization")
    if config["parent_h3_route"]["route"] != H3_ROUTE:
        raise ValueError("registered H3 route mismatch")
    for record, path in (
        (config["parent_h3_route"], h3_path),
        (config["source_ab_freeze"], ab_path),
        (config["c_packet_freeze"], c_freeze_path),
    ):
        if common.sha256_file(path) != record["sha256"]:
            raise ValueError(f"registered source hash mismatch: {relative(path)}")
    if c_freeze.get("a_b_labels_or_reasons_visible") is not False:
        raise ValueError("C packet freeze exposes A/B decisions")
    if c_freeze.get("outcome_key_joined") is not False or c_freeze.get("heldout_opened") is not False:
        raise ValueError("C packet freeze opened prohibited identities or heldout")
    if config.get("qualification_route") != QUALIFIED_ROUTE or config.get("failure_route") != FAILURE_ROUTE:
        raise ValueError("H4Q route mismatch")

    reviewers = config.get("reviewers", {})
    if set(reviewers) != set(ROLES):
        raise ValueError("H4Q requires three named Reviewer C sessions")
    prohibited_sessions, prohibited_reviewers = prohibited_identities(incident, ab_freeze)
    for role in ROLES:
        validate_reviewer(role, reviewers[role], prohibited_sessions, prohibited_reviewers)
    reviewer_ids = [str(reviewers[role]["anonymized_reviewer_id"]) for role in ROLES]
    session_ids = [str(reviewers[role]["session_id"]) for role in ROLES]
    if len(set(reviewer_ids)) != 3 or len(set(session_ids)) != 3:
        raise ValueError("H4Q reviewer or session identities are not distinct")

    materials = {
        domain: validate_materials(domain, config["material_freeze"][domain], c_freeze)
        for domain in DOMAINS
    }
    return config, reviewers, materials, c_freeze


def run() -> None:
    config, reviewers, materials, c_freeze = validate_entry()
    source_head = git("rev-parse", "HEAD")
    if source_head != config["entry_git_head"]:
        raise ValueError("H4Q must execute from its registered entry HEAD")

    packet_rows = {domain: int(c_freeze["domains"][domain]["packet_rows"]) for domain in DOMAINS}
    disagreement_fields = {
        domain: int(c_freeze["domains"][domain]["decision_field_disagreements"])
        for domain in DOMAINS
    }
    qualification = {
        "name": "step3ct_t5h_h4q_model_reviewer_qualification_v1",
        "source_git_head": source_head,
        "source_registry": relative(CONFIG_PATH),
        "source_registry_sha256": common.sha256_file(CONFIG_PATH),
        "route": QUALIFIED_ROUTE,
        "reviewers": reviewers,
        "reviewer_count": 3,
        "distinct_reviewer_ids": True,
        "distinct_session_ids": True,
        "separate_from_all_ab_and_failed_t5_sessions": True,
        "reviewer_repository_access_before_freeze": False,
        "reviewer_study_rows_accessed_before_freeze": 0,
        "coordinator_material_hash_validation_only": True,
        "coordinator_semantic_rows_viewed": 0,
        "packet_rows": packet_rows,
        "decision_field_disagreements": disagreement_fields,
        "materials": materials,
        "authorization": config["authorization"],
        "acceptance_condition": "each domain reviewer must produce one complete validated output and provenance",
    }
    common.write_json(QUALIFICATION_PATH, qualification)

    route = {
        "name": "step3ct_t5h_h4q_route_decision_v1",
        "source_git_head": source_head,
        "route": QUALIFIED_ROUTE,
        "gates": {
            "h3_authorization_verified": True,
            "c_packet_freeze_verified": True,
            "three_distinct_no_context_sessions": True,
            "sessions_distinct_from_all_ab_and_failed_t5_sessions": True,
            "runtime_and_self_reported_model_metadata_frozen": True,
            "zero_reviewer_repository_access_before_freeze": True,
            "zero_reviewer_study_row_access_before_freeze": True,
            "complete_or_abort_contract_frozen": True,
            "packet_template_manifest_and_rubric_hashes_match": True,
            "hidden_key_and_heldout_sealed": True,
        },
        "reviewer_c_attempts_authorized": True,
        "label_acceptance_authorized": False,
        "reliability_authorized": False,
        "hidden_key_join_authorized": False,
        "t6_authorized": False,
        "failure_route": FAILURE_ROUTE,
        "next_action": "push_h4q_then_launch_one_complete_c_attempt_per_domain",
    }
    common.write_json(ROUTE_PATH, route)

    execution = {
        "name": "step3ct_t5h_h4q_execution_record_v1",
        "source_git_head": source_head,
        "command": "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_reviewer_c_qualification.py run",
        "reviewer_sessions_launched_for_qualification": 3,
        "reviewer_repository_access": 0,
        "reviewer_study_rows_accessed": 0,
        "coordinator_semantic_rows_viewed": 0,
        "coordinator_actions": [
            "read zero-access qualification JSON responses",
            "validate C packet, template, manifest, and rubric hashes and row counts",
        ],
        "labels_created": 0,
        "ab_labels_opened_by_coordinator": False,
        "hidden_key_opened": False,
        "heldout_opened": False,
        "route": QUALIFIED_ROUTE,
    }
    common.write_json(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H4Q Reviewer C Qualification Report",
            "",
            f"Route: `{QUALIFIED_ROUTE}`",
            "",
            "Three distinct no-context model sessions completed prospective qualification without repository "
            "or study-row access. Safety C, Quality C, and Utility C are separate from one another, all six "
            "frozen A/B sessions, and all failed T5 sessions.",
            "",
            "The frozen disagreement-only workloads are Safety 11 rows / 37 disputed fields, Quality 69 / 76, "
            "and Utility 33 / 43. Packet, blank-template, manifest, and rubric hashes match the C packet freeze. "
            "The coordinator performed mechanical metadata validation and did not inspect response or label semantics.",
            "",
            "After this package is committed and pushed, each Reviewer C session is authorized for one complete "
            "unchanged attempt in its assigned domain. Any partial, blocked, malformed, or unprovenanced output "
            f"is rejected in full and selects `{FAILURE_ROUTE}`. No label is accepted and no reliability metric "
            "is computed by this qualification package.",
            "",
        ]
    )
    common.write_text(REPORT_PATH, report)

    package_paths = [QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH]
    package = {
        "name": "step3ct_t5h_h4q_package_manifest_v1",
        "route": QUALIFIED_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    common.write_json(PACKAGE_PATH, package)
    print(json.dumps({"route": QUALIFIED_ROUTE, "reviewers": 3, "study_rows_accessed": 0}))


def verify() -> None:
    validate_entry()
    package = read_json(PACKAGE_PATH)
    qualification = read_json(QUALIFICATION_PATH)
    route = read_json(ROUTE_PATH)
    failures: list[str] = []
    if package.get("route") != QUALIFIED_ROUTE or package.get("artifact_count") != 4:
        failures.append("package_route_or_count")
    for path_text, expected in package.get("artifacts", {}).items():
        path = common.REPO / path_text
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(path_text)
    if qualification.get("reviewer_count") != 3 or qualification.get("reviewer_study_rows_accessed_before_freeze") != 0:
        failures.append("reviewer_count_or_access")
    if route.get("route") != QUALIFIED_ROUTE or route.get("label_acceptance_authorized") is not False:
        failures.append("route_or_label_acceptance")
    for path in (QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH, PACKAGE_PATH):
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{relative(path)}")
    if failures:
        raise RuntimeError(f"H4Q verification failed: {failures}")
    print(json.dumps({"route": QUALIFIED_ROUTE, "package_artifacts": 4, "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
