"""Record the zero-access T5H reviewer-qualification waiting state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO / "experiments/step3_response_quality_persistence"
REGISTRY_PATH = EXPERIMENT_ROOT / "configs/step3ct_t5h_external_review_registry.json"
P6A_ROUTE_PATH = (
    EXPERIMENT_ROOT
    / "results/4070ti_step3ct_seed1/mechanism_completion/p6a/step3ct_t5r_p6a_route_decision.json"
)
H1_ROOT = EXPERIMENT_ROOT / "results/4070ti_step3ct_seed1/t5h/h1"
QUALIFICATION_PATH = H1_ROOT / "step3ct_t5h_h1_reviewer_qualification.json"
ROUTE_PATH = H1_ROOT / "step3ct_t5h_h1_route_decision.json"
EXECUTION_PATH = H1_ROOT / "step3ct_t5h_h1_execution_record.json"
REPORT_PATH = H1_ROOT / "Step3CT_T5H_H1_reviewer_qualification_report.md"
PACKAGE_PATH = H1_ROOT / "step3ct_t5h_h1_package_manifest.json"

SUCCESS_P6A_ROUTE = "P6A1_eol_reconciled_content_unchanged"
WAITING_ROUTE = "T5H0_waiting_for_qualified_reviewers"
RUN_COMMAND = (
    "python experiments/step3_response_quality_persistence/scripts/"
    "record_step3ct_t5h_waiting.py run"
)
VERIFY_COMMAND = (
    "python experiments/step3_response_quality_persistence/scripts/"
    "record_step3ct_t5h_waiting.py verify"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_lf(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json_lf(path: Path, value: Any) -> None:
    write_text_lf(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_entry(p6a_route: dict[str, Any], registry: dict[str, Any]) -> int:
    if p6a_route.get("route") != SUCCESS_P6A_ROUTE:
        raise RuntimeError("H1 requires a verified P6A1 route")
    required = int(registry["external_review"]["required_primary_reviewers"])
    if required != 2:
        raise RuntimeError(f"H1 requires exactly two primary reviewers, found {required}")
    if registry["external_review"]["first_domain"] != "safety":
        raise RuntimeError("H1 qualification must begin with safety")
    return required


def zero_access_record() -> dict[str, Any]:
    return {
        "safety_packet_opened": False,
        "safety_packet_rows_accessed": 0,
        "quality_packet_opened": False,
        "quality_packet_rows_accessed": 0,
        "utility_packet_opened": False,
        "utility_packet_rows_accessed": 0,
        "hidden_key_opened": False,
        "heldout_opened": False,
        "gateway_probe_using_study_rows": False,
        "packet_transformed_or_sanitized": False,
        "main_executor_assigned_labels": False,
        "heuristic_or_embedding_labels_substituted": False,
    }


def zero_output_counts() -> dict[str, int]:
    return {
        "accepted_files": 0,
        "accepted_rows": 0,
        "rejected_files": 0,
        "rejected_rows": 0,
        "partial_files": 0,
        "partial_rows": 0,
    }


def handoff_records(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for domain in ("safety", "quality", "utility"):
        path = REPO / registry["handoffs"][domain]
        records[domain] = {"path": relative(path), **artifact_record(path)}
    return records


def build_qualification(
    entry_git_head: str,
    required_reviewers: int,
    registry_sha256: str,
    p6a_route_sha256: str,
    handoffs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": "step3ct_t5h_h1_reviewer_qualification_v1",
        "entry_git_head": entry_git_head,
        "result_commit_reference": "the Git commit containing this H1 package",
        "p6a_route": SUCCESS_P6A_ROUTE,
        "p6a_route_path": relative(P6A_ROUTE_PATH),
        "p6a_route_sha256": p6a_route_sha256,
        "registry_path": relative(REGISTRY_PATH),
        "registry_sha256": registry_sha256,
        "required_domain": "safety",
        "preferred_reviewer_type": "external_human",
        "required_reviewers": required_reviewers,
        "supplied_reviewer_records": 0,
        "qualified_reviewers": 0,
        "reviewer_ids_registered": 0,
        "session_ids_registered": 0,
        "reviewer_sessions_launched": 0,
        "reviewer_records": [],
        "packet_template_rubric_freeze_status": "not_started_no_qualified_reviewers",
        "handoff_artifacts": handoffs,
        "study_access": zero_access_record(),
        "outputs": zero_output_counts(),
        "qualification_pass": False,
        "route": WAITING_ROUTE,
        "method_result": False,
        "reason_code": "research_coordinator_has_not_supplied_two_qualified_independent_safety_reviewers",
        "next_required_resource": "two_qualified_independent_safety_reviewers_from_research_coordinator",
    }


def run() -> None:
    p6a_route = read_json(P6A_ROUTE_PATH)
    registry = read_json(REGISTRY_PATH)
    required_reviewers = validate_entry(p6a_route, registry)
    entry_git_head = git("rev-parse", "HEAD")
    handoffs = handoff_records(registry)

    qualification = build_qualification(
        entry_git_head=entry_git_head,
        required_reviewers=required_reviewers,
        registry_sha256=sha256_file(REGISTRY_PATH),
        p6a_route_sha256=sha256_file(P6A_ROUTE_PATH),
        handoffs=handoffs,
    )
    write_json_lf(QUALIFICATION_PATH, qualification)

    route_record = {
        "name": "step3ct_t5h_h1_route_decision_v1",
        "entry_git_head": entry_git_head,
        "result_commit_reference": "the Git commit containing this H1 package",
        "route": WAITING_ROUTE,
        "gates": {
            "p6a1_verified": True,
            "two_reviewer_records_supplied": False,
            "two_reviewers_qualified": False,
            "zero_study_row_access": True,
            "zero_reviewer_sessions_launched": True,
            "zero_label_outputs": True,
            "hidden_key_sealed": True,
            "heldout_sealed": True,
            "new_training_or_generation_started": False,
        },
        "counts": {
            "required_reviewers": required_reviewers,
            "supplied_reviewers": 0,
            "qualified_reviewers": 0,
            **zero_output_counts(),
        },
        "method_result": False,
        "next_authorized_action": "wait_for_research_coordinator_to_supply_two_qualified_reviewers",
        "safety_review_authorized": False,
        "quality_utility_review_authorized": False,
        "training_or_generation_authorized": False,
    }
    write_json_lf(ROUTE_PATH, route_record)

    execution = {
        "name": "step3ct_t5h_h1_execution_record_v1",
        "entry_git_head": entry_git_head,
        "result_commit_reference": "the Git commit containing this H1 package",
        "commands": [RUN_COMMAND, VERIFY_COMMAND],
        "inputs_read": {
            relative(P6A_ROUTE_PATH): artifact_record(P6A_ROUTE_PATH),
            relative(REGISTRY_PATH): artifact_record(REGISTRY_PATH),
            **{record["path"]: {"sha256": record["sha256"], "bytes": record["bytes"]} for record in handoffs.values()},
        },
        "study_packet_files_read": 0,
        "study_packet_rows_read": 0,
        "reviewer_sessions_launched": 0,
        "outputs": zero_output_counts(),
        "route": WAITING_ROUTE,
        "heldout_opened": False,
        "hidden_key_opened": False,
        "gateway_probe_using_study_rows": False,
        "new_training_or_generation": False,
    }
    write_json_lf(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H1 Reviewer Qualification Report",
            "",
            f"Route: `{WAITING_ROUTE}`",
            "",
            "## Decision",
            "",
            f"P6A is verified as `{SUCCESS_P6A_ROUTE}`. H1 requires two independent safety reviewers, but the research "
            "coordinator has supplied zero reviewer provenance records. No reviewer or session ID was registered, and "
            "no review session was launched.",
            "",
            "This is a zero-access waiting state, not a method result or a reviewer-access failure.",
            "",
            "## Counts",
            "",
            "| Item | Count |",
            "| --- | ---: |",
            f"| required safety reviewers | {required_reviewers} |",
            "| supplied reviewer records | 0 |",
            "| qualified reviewers | 0 |",
            "| launched reviewer sessions | 0 |",
            "| accepted files / rows | 0 / 0 |",
            "| rejected files / rows | 0 / 0 |",
            "| partial files / rows | 0 / 0 |",
            "| safety / quality / utility rows accessed | 0 / 0 / 0 |",
            "",
            "## Boundary",
            "",
            "Safety, quality, and utility packets were not opened. The hidden key and heldout set remain sealed; no "
            "gateway probe, heuristic label, training, or generation was attempted. The only authorized next action is "
            "to wait for the research coordinator to supply two qualified independent safety reviewers.",
            "",
        ]
    )
    write_text_lf(REPORT_PATH, report)

    package_paths = [QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH]
    package = {
        "name": "step3ct_t5h_h1_package_manifest_v1",
        "entry_git_head": entry_git_head,
        "result_commit_reference": "the Git commit containing this H1 package",
        "route": WAITING_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    write_json_lf(PACKAGE_PATH, package)
    print(
        json.dumps(
            {
                "route": WAITING_ROUTE,
                "required_reviewers": required_reviewers,
                "qualified_reviewers": 0,
                "study_rows_accessed": 0,
            }
        )
    )


def verify() -> None:
    package = read_json(PACKAGE_PATH)
    qualification = read_json(QUALIFICATION_PATH)
    route_record = read_json(ROUTE_PATH)
    execution = read_json(EXECUTION_PATH)
    failures: list[str] = []

    if package.get("route") != WAITING_ROUTE or package.get("artifact_count") != 4:
        failures.append("package_route_or_count")
    for relative_path, expected in package.get("artifacts", {}).items():
        path = REPO / relative_path
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(relative_path)
    for relative_path, expected in execution.get("inputs_read", {}).items():
        path = REPO / relative_path
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(f"input:{relative_path}")

    access = qualification.get("study_access", {})
    counts = qualification.get("outputs", {})
    if any(value is True for value in access.values()) or any(
        isinstance(value, int) and not isinstance(value, bool) and value != 0 for value in access.values()
    ):
        failures.append("nonzero_study_access")
    if any(value != 0 for value in counts.values()):
        failures.append("nonzero_output_counts")
    if qualification.get("qualified_reviewers") != 0 or qualification.get("reviewer_sessions_launched") != 0:
        failures.append("reviewer_session_or_qualification_present")
    if route_record.get("route") != WAITING_ROUTE or route_record.get("method_result") is not False:
        failures.append("invalid_route_record")
    if execution.get("study_packet_files_read") != 0 or execution.get("study_packet_rows_read") != 0:
        failures.append("execution_records_study_access")

    for path in (QUALIFICATION_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH, PACKAGE_PATH):
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{relative(path)}")
    if failures:
        raise RuntimeError(f"H1 verification failed: {failures}")
    print(
        json.dumps(
            {
                "route": WAITING_ROUTE,
                "package_artifacts": package["artifact_count"],
                "study_rows_accessed": 0,
                "passed": True,
            }
        )
    )


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
