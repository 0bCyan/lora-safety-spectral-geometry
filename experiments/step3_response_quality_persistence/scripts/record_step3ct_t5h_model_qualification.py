"""Freeze two isolated model reviewers before Safety packet access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "experiments/step3_response_quality_persistence"
CONFIG_PATH = ROOT / "configs/step3ct_t5h_model_reviewer_launch_registry.json"
P6A_ROUTE_PATH = (
    ROOT
    / "results/4070ti_step3ct_seed1/mechanism_completion/p6a/step3ct_t5r_p6a_route_decision.json"
)
T5R_REGISTRY_PATH = ROOT / "configs/step3ct_t5r_review_recovery_registry.json"
T5_INCIDENT_PATH = ROOT / "results/4070ti_step3ct_seed1/t5/step3ct_t5_safety_gateway_incident.json"
H1Q_ROOT = ROOT / "results/4070ti_step3ct_seed1/t5h/h1q"
QUALIFICATION_PATH = H1Q_ROOT / "step3ct_t5h_h1q_model_reviewer_qualification.json"
ROUTE_PATH = H1Q_ROOT / "step3ct_t5h_h1q_route_decision.json"
EXECUTION_PATH = H1Q_ROOT / "step3ct_t5h_h1q_execution_record.json"
REPORT_PATH = H1Q_ROOT / "Step3CT_T5H_H1Q_model_reviewer_qualification_report.md"
PACKAGE_PATH = H1Q_ROOT / "step3ct_t5h_h1q_package_manifest.json"

SUCCESS_P6A_ROUTE = "P6A1_eol_reconciled_content_unchanged"
WAITING_ROUTE = "T5H0_waiting_for_qualified_reviewers"
QUALIFIED_ROUTE = "H1Q1_two_model_sessions_qualified_for_complete_safety_attempt"
FAILURE_ROUTE = "T5H1_reviewer_access_or_provenance_stop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def write_text_lf(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json_lf(path: Path, value: Any) -> None:
    write_text_lf(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def failed_session_ids(incident: dict[str, Any]) -> set[str]:
    result = {str(incident["failed_required_session"]["agent_id"])}
    result.update(str(row["agent_id"]) for row in incident["stopped_sessions"])
    return result


def validate_reviewer(role: str, reviewer: dict[str, Any], prohibited: set[str]) -> None:
    if reviewer.get("reviewer_role") != f"safety_{role}":
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
        raise ValueError(f"reviewer {role} reuses a failed T5 session")
    confirmations = reviewer.get("qualification_confirmations", {})
    required_false = ("prior_study_involvement", "repository_access_so_far", "other_reviewer_contact")
    required_true = (
        "agrees_independent_complete_630_review",
        "agrees_no_partial_output",
        "agrees_no_external_retrieval",
    )
    if any(confirmations.get(field) is not False for field in required_false):
        raise ValueError(f"reviewer {role} failed zero-access independence qualification")
    if confirmations.get("study_rows_accessed") != 0:
        raise ValueError(f"reviewer {role} accessed study rows before qualification freeze")
    if any(confirmations.get(field) is not True for field in required_true):
        raise ValueError(f"reviewer {role} did not accept the complete-or-abort contract")


def validate_materials(config: dict[str, Any], registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    material = config["material_freeze"]
    safety = registry["packets"]["safety"]
    expected = {
        "packet_path": safety["packet_path"],
        "packet_sha256": safety["packet_sha256"],
        "packet_rows": safety["packet_rows"],
        "rubric_path": safety["rubric_path"],
        "rubric_sha256": safety["rubric_sha256"],
        "template_sha256": safety["reviewer_template_sha256"],
    }
    for field, value in expected.items():
        if material.get(field) != value:
            raise ValueError(f"material freeze {field} differs from T5R registry")

    paths = {
        "handoff": REPO / material["handoff_path"],
        "packet": REPO / material["packet_path"],
        "rubric": REPO / material["rubric_path"],
        "template_a": REPO / material["templates"]["a"],
        "template_b": REPO / material["templates"]["b"],
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
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"frozen material mismatch: {name}")
        records[name] = {"path": relative(path), **artifact_record(path)}
    with paths["packet"].open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != int(material["packet_rows"]):
        raise ValueError(f"packet row count mismatch: {rows}")
    return records


def validate_entry() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    config = read_json(CONFIG_PATH)
    p6a = read_json(P6A_ROUTE_PATH)
    registry = read_json(T5R_REGISTRY_PATH)
    incident = read_json(T5_INCIDENT_PATH)
    waiting_path = REPO / config["parent_waiting_route"]["path"]
    waiting = read_json(waiting_path)

    if p6a.get("route") != SUCCESS_P6A_ROUTE:
        raise ValueError("model review requires P6A1")
    if waiting.get("route") != WAITING_ROUTE or config["parent_waiting_route"]["route"] != WAITING_ROUTE:
        raise ValueError("model review must resume from T5H0")
    if sha256_file(waiting_path) != config["parent_waiting_route"]["sha256"]:
        raise ValueError("parent T5H0 route hash mismatch")
    if config.get("qualification_route") != QUALIFIED_ROUTE or config.get("failure_route") != FAILURE_ROUTE:
        raise ValueError("qualification route mismatch")

    reviewers = config.get("reviewers", {})
    if set(reviewers) != {"a", "b"}:
        raise ValueError("qualification requires reviewers a and b")
    prohibited = failed_session_ids(incident)
    for role in ("a", "b"):
        validate_reviewer(role, reviewers[role], prohibited)
    reviewer_ids = [reviewers[role]["anonymized_reviewer_id"] for role in ("a", "b")]
    session_ids = [reviewers[role]["session_id"] for role in ("a", "b")]
    if len(set(reviewer_ids)) != 2 or len(set(session_ids)) != 2:
        raise ValueError("reviewer or session identities are not distinct")
    materials = validate_materials(config, registry)
    return config, reviewers, materials


def run() -> None:
    config, reviewers, materials = validate_entry()
    source_head = git("rev-parse", "HEAD")
    if source_head != config["entry_git_head"]:
        raise ValueError("qualification must execute from its registered entry HEAD")

    qualification = {
        "name": "step3ct_t5h_h1q_model_reviewer_qualification_v1",
        "source_git_head": source_head,
        "source_registry": relative(CONFIG_PATH),
        "source_registry_sha256": sha256_file(CONFIG_PATH),
        "route": QUALIFIED_ROUTE,
        "reviewers": reviewers,
        "reviewer_count": 2,
        "distinct_reviewer_ids": True,
        "distinct_session_ids": True,
        "reviewer_repository_access_before_freeze": False,
        "reviewer_study_rows_accessed_before_freeze": 0,
        "coordinator_material_hash_validation_only": True,
        "coordinator_semantic_rows_viewed": 0,
        "materials": materials,
        "authorization": config["authorization"],
        "acceptance_condition": "each reviewer must produce one complete validated 630-row output and provenance",
    }
    write_json_lf(QUALIFICATION_PATH, qualification)

    route = {
        "name": "step3ct_t5h_h1q_route_decision_v1",
        "source_git_head": source_head,
        "route": QUALIFIED_ROUTE,
        "gates": {
            "p6a1_verified": True,
            "parent_t5h0_verified": True,
            "two_distinct_no_context_sessions": True,
            "exact_runtime_and_self_reported_model_metadata_frozen": True,
            "zero_reviewer_repository_access_before_freeze": True,
            "zero_reviewer_study_row_access_before_freeze": True,
            "complete_or_abort_contract_frozen": True,
            "material_hashes_match": True,
        },
        "safety_attempt_authorized": True,
        "label_acceptance_authorized": False,
        "quality_utility_authorized": False,
        "failure_route": FAILURE_ROUTE,
        "next_action": "push_h1q_then_launch_one_complete_safety_attempt_per_reviewer",
    }
    write_json_lf(ROUTE_PATH, route)

    execution = {
        "name": "step3ct_t5h_h1q_execution_record_v1",
        "source_git_head": source_head,
        "command": "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_model_qualification.py run",
        "reviewer_sessions_launched_for_qualification": 2,
        "reviewer_repository_access": 0,
        "reviewer_study_rows_accessed": 0,
        "coordinator_semantic_rows_viewed": 0,
        "coordinator_actions": ["read qualification JSON responses", "validate registered material hashes and row count"],
        "labels_created": 0,
        "heldout_opened": False,
        "hidden_key_opened": False,
        "route": QUALIFIED_ROUTE,
    }
    write_json_lf(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H1Q Model Reviewer Qualification Report",
            "",
            f"Route: `{QUALIFIED_ROUTE}`",
            "",
            "Two distinct no-context model sessions completed qualification without repository or study-row access. "
            "Reviewer A freezes the runtime selection `gpt-5.5` and self-reported family `gpt-5`; Reviewer B freezes "
            "`gpt-5.6-sol` for both fields.",
            "",
            "All registered Safety packet, A/B template, rubric, and handoff hashes match. The coordinator performed "
            "hash and row-count validation only and did not inspect response semantics.",
            "",
            "After this package is committed and pushed, each reviewer is authorized for one complete unchanged "
            "630-row Safety attempt. A partial, blocked, malformed, or unprovenanced output is rejected in full and "
            f"selects `{FAILURE_ROUTE}`. No label is accepted by this qualification package.",
            "",
        ]
    )
    write_text_lf(REPORT_PATH, report)

    package_paths = [QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH]
    package = {
        "name": "step3ct_t5h_h1q_package_manifest_v1",
        "route": QUALIFIED_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    write_json_lf(PACKAGE_PATH, package)
    print(json.dumps({"route": QUALIFIED_ROUTE, "reviewers": 2, "study_rows_accessed": 0}))


def verify() -> None:
    validate_entry()
    package = read_json(PACKAGE_PATH)
    qualification = read_json(QUALIFICATION_PATH)
    route = read_json(ROUTE_PATH)
    failures: list[str] = []
    if package.get("route") != QUALIFIED_ROUTE or package.get("artifact_count") != 4:
        failures.append("package_route_or_count")
    for path_text, expected in package.get("artifacts", {}).items():
        path = REPO / path_text
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(path_text)
    if qualification.get("reviewer_count") != 2:
        failures.append("reviewer_count")
    if qualification.get("reviewer_study_rows_accessed_before_freeze") != 0:
        failures.append("reviewer_study_access")
    if route.get("route") != QUALIFIED_ROUTE or route.get("label_acceptance_authorized") is not False:
        failures.append("route_or_label_acceptance")
    for path in (QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH, PACKAGE_PATH):
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{relative(path)}")
    if failures:
        raise RuntimeError(f"H1Q verification failed: {failures}")
    print(json.dumps({"route": QUALIFIED_ROUTE, "package_artifacts": 4, "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
