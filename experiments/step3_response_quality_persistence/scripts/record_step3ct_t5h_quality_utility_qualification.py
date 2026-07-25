"""Freeze four isolated model reviewers before Quality/Utility packet access."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import step3ct_t5r_common as common


ROOT = common.REPO / "experiments/step3_response_quality_persistence"
CONFIG_PATH = ROOT / "configs/step3ct_t5h_quality_utility_model_reviewer_registry.json"
H2_ROUTE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5h/h2/step3ct_t5h_h2_route_decision.json"
T5R_REGISTRY_PATH = ROOT / "configs/step3ct_t5r_review_recovery_registry.json"
T5_INCIDENT_PATH = ROOT / "results/4070ti_step3ct_seed1/t5/step3ct_t5_safety_gateway_incident.json"
SAFETY_FREEZE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_safety_ab_freeze.json"
H3Q_ROOT = ROOT / "results/4070ti_step3ct_seed1/t5h/h3q"
QUALIFICATION_PATH = H3Q_ROOT / "step3ct_t5h_h3q_model_reviewer_qualification.json"
ROUTE_PATH = H3Q_ROOT / "step3ct_t5h_h3q_route_decision.json"
EXECUTION_PATH = H3Q_ROOT / "step3ct_t5h_h3q_execution_record.json"
REPORT_PATH = H3Q_ROOT / "Step3CT_T5H_H3Q_model_reviewer_qualification_report.md"
PACKAGE_PATH = H3Q_ROOT / "step3ct_t5h_h3q_package_manifest.json"

H2_ROUTE = "T5H2_safety_ab_frozen_authorize_quality_utility"
QUALIFIED_ROUTE = "H3Q1_four_model_sessions_qualified_for_quality_utility_attempts"
FAILURE_ROUTE = "T5R2_review_completion_or_schema_stop"
ROLES = ("quality_a", "quality_b", "utility_a", "utility_b")


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


def failed_session_ids(incident: dict[str, Any], safety_freeze: dict[str, Any]) -> set[str]:
    result = {str(incident["failed_required_session"]["agent_id"])}
    result.update(str(row["agent_id"]) for row in incident["stopped_sessions"])
    safety = safety_freeze["domains"]["safety"]["reviewers"]
    result.update(str(safety[role]["session_id"]) for role in ("a", "b"))
    return result


def validate_reviewer(role: str, reviewer: dict[str, Any], prohibited: set[str]) -> None:
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
    if reviewer.get("fork_context") is not False:
        raise ValueError(f"reviewer {role} inherited context")
    if str(reviewer["session_id"]) in prohibited:
        raise ValueError(f"reviewer {role} reuses a prohibited session")
    confirmations = reviewer.get("qualification_confirmations", {})
    required_false = ("prior_study_involvement", "repository_access_so_far", "other_reviewer_contact")
    required_true = (
        "agrees_independent_complete_330_review",
        "agrees_no_partial_output",
        "agrees_no_external_retrieval",
    )
    if any(confirmations.get(field) is not False for field in required_false):
        raise ValueError(f"reviewer {role} failed zero-access independence qualification")
    if confirmations.get("study_rows_accessed") != 0:
        raise ValueError(f"reviewer {role} accessed study rows before qualification freeze")
    if any(confirmations.get(field) is not True for field in required_true):
        raise ValueError(f"reviewer {role} did not accept the complete-or-abort contract")


def validate_domain_materials(
    domain: str,
    material: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    spec = registry["packets"][domain]
    expected = {
        "packet_path": spec["packet_path"],
        "packet_sha256": spec["packet_sha256"],
        "packet_rows": spec["packet_rows"],
        "rubric_path": spec["rubric_path"],
        "rubric_sha256": spec["rubric_sha256"],
        "template_sha256": spec["reviewer_template_sha256"],
    }
    for field, value in expected.items():
        if material.get(field) != value:
            raise ValueError(f"{domain} material {field} differs from T5R registry")

    paths = {
        "handoff": common.REPO / material["handoff_path"],
        "packet": common.REPO / material["packet_path"],
        "rubric": common.REPO / material["rubric_path"],
        "template_a": common.REPO / material["templates"]["a"],
        "template_b": common.REPO / material["templates"]["b"],
    }
    expected_hashes = {
        "handoff": material["handoff_sha256"],
        "packet": material["packet_sha256"],
        "rubric": material["rubric_sha256"],
        "template_a": material["template_sha256"],
        "template_b": material["template_sha256"],
    }
    records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file() or common.sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"{domain} frozen material mismatch: {name}")
        records[name] = {"path": relative(path), **artifact_record(path)}
    with paths["packet"].open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != int(material["packet_rows"]):
        raise ValueError(f"{domain} packet row count mismatch: {rows}")
    return records


def validate_entry() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    h2_path = common.REPO / config["parent_h2_route"]["path"]
    h2 = read_json(h2_path)
    registry = read_json(T5R_REGISTRY_PATH)
    incident = read_json(T5_INCIDENT_PATH)
    safety_freeze = read_json(SAFETY_FREEZE_PATH)

    if h2.get("route") != H2_ROUTE or h2.get("quality_utility_review_authorized") is not True:
        raise ValueError("H3Q requires H2 Quality/Utility authorization")
    if config["parent_h2_route"]["route"] != H2_ROUTE:
        raise ValueError("registered H2 route mismatch")
    if common.sha256_file(h2_path) != config["parent_h2_route"]["sha256"]:
        raise ValueError("parent H2 route hash mismatch")
    if config.get("qualification_route") != QUALIFIED_ROUTE or config.get("failure_route") != FAILURE_ROUTE:
        raise ValueError("H3Q route mismatch")

    reviewers = config.get("reviewers", {})
    if set(reviewers) != set(ROLES):
        raise ValueError("H3Q requires four named reviewers")
    prohibited = failed_session_ids(incident, safety_freeze)
    for role in ROLES:
        validate_reviewer(role, reviewers[role], prohibited)
    reviewer_ids = [str(reviewers[role]["anonymized_reviewer_id"]) for role in ROLES]
    session_ids = [str(reviewers[role]["session_id"]) for role in ROLES]
    if len(set(reviewer_ids)) != 4 or len(set(session_ids)) != 4:
        raise ValueError("H3Q reviewer or session identities are not distinct")

    materials = {
        domain: validate_domain_materials(domain, config["material_freeze"][domain], registry)
        for domain in ("quality", "utility")
    }
    return config, reviewers, materials


def run() -> None:
    config, reviewers, materials = validate_entry()
    source_head = git("rev-parse", "HEAD")
    if source_head != config["entry_git_head"]:
        raise ValueError("H3Q must execute from its registered entry HEAD")

    qualification = {
        "name": "step3ct_t5h_h3q_model_reviewer_qualification_v1",
        "source_git_head": source_head,
        "source_registry": relative(CONFIG_PATH),
        "source_registry_sha256": common.sha256_file(CONFIG_PATH),
        "route": QUALIFIED_ROUTE,
        "reviewers": reviewers,
        "reviewer_count": 4,
        "distinct_reviewer_ids": True,
        "distinct_session_ids": True,
        "separate_from_safety_sessions": True,
        "reviewer_repository_access_before_freeze": False,
        "reviewer_study_rows_accessed_before_freeze": 0,
        "coordinator_material_hash_validation_only": True,
        "coordinator_semantic_rows_viewed": 0,
        "materials": materials,
        "authorization": config["authorization"],
        "acceptance_condition": "each reviewer must produce one complete validated 330-row output and provenance",
    }
    common.write_json(QUALIFICATION_PATH, qualification)

    route = {
        "name": "step3ct_t5h_h3q_route_decision_v1",
        "source_git_head": source_head,
        "route": QUALIFIED_ROUTE,
        "gates": {
            "h2_authorization_verified": True,
            "four_distinct_no_context_sessions": True,
            "sessions_distinct_from_safety_reviewers": True,
            "runtime_and_self_reported_model_metadata_frozen": True,
            "zero_reviewer_repository_access_before_freeze": True,
            "zero_reviewer_study_row_access_before_freeze": True,
            "complete_or_abort_contract_frozen": True,
            "quality_and_utility_material_hashes_match": True,
        },
        "quality_utility_attempts_authorized": True,
        "label_acceptance_authorized": False,
        "ab_comparison_authorized": False,
        "reviewer_c_authorized": False,
        "t6_authorized": False,
        "failure_route": FAILURE_ROUTE,
        "next_action": "push_h3q_then_launch_one_complete_attempt_per_reviewer",
    }
    common.write_json(ROUTE_PATH, route)

    execution = {
        "name": "step3ct_t5h_h3q_execution_record_v1",
        "source_git_head": source_head,
        "command": "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_quality_utility_qualification.py run",
        "reviewer_sessions_launched_for_qualification": 4,
        "reviewer_repository_access": 0,
        "reviewer_study_rows_accessed": 0,
        "coordinator_semantic_rows_viewed": 0,
        "coordinator_actions": ["read qualification JSON responses", "validate material hashes and row counts"],
        "labels_created": 0,
        "safety_labels_opened": False,
        "hidden_key_opened": False,
        "heldout_opened": False,
        "route": QUALIFIED_ROUTE,
    }
    common.write_json(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H3Q Model Reviewer Qualification Report",
            "",
            f"Route: `{QUALIFIED_ROUTE}`",
            "",
            "Four distinct no-context model sessions completed qualification without repository or study-row access. "
            "Quality A/B and Utility A/B are separate from one another and from both frozen Safety reviewers.",
            "",
            "All registered Quality and Utility packet, role-specific template, rubric, and handoff hashes match. "
            "The coordinator performed hash and row-count validation only and did not inspect response semantics.",
            "",
            "After this package is committed and pushed, each reviewer is authorized for one complete unchanged "
            "330-row attempt in its assigned domain. Any partial, blocked, malformed, or unprovenanced output is "
            f"rejected in full and selects `{FAILURE_ROUTE}`. No label is accepted by this qualification package.",
            "",
        ]
    )
    common.write_text(REPORT_PATH, report)

    package_paths = [QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH]
    package = {
        "name": "step3ct_t5h_h3q_package_manifest_v1",
        "route": QUALIFIED_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    common.write_json(PACKAGE_PATH, package)
    print(json.dumps({"route": QUALIFIED_ROUTE, "reviewers": 4, "study_rows_accessed": 0}))


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
    if qualification.get("reviewer_count") != 4 or qualification.get("reviewer_study_rows_accessed_before_freeze") != 0:
        failures.append("reviewer_count_or_access")
    if route.get("route") != QUALIFIED_ROUTE or route.get("label_acceptance_authorized") is not False:
        failures.append("route_or_label_acceptance")
    for path in (QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH, PACKAGE_PATH):
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{relative(path)}")
    if failures:
        raise RuntimeError(f"H3Q verification failed: {failures}")
    print(json.dumps({"route": QUALIFIED_ROUTE, "package_artifacts": 4, "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
