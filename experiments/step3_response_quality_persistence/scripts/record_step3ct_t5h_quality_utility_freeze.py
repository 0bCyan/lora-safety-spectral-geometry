"""Record the H3 route after complete Quality/Utility A/B freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_step3ct_t5r_review as review
import step3ct_t5r_common as common


ROOT = common.REPO / "experiments/step3_response_quality_persistence"
H3Q_ROUTE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5h/h3q/step3ct_t5h_h3q_route_decision.json"
SAFETY_FREEZE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_safety_ab_freeze.json"
ALL_AB_FREEZE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_all_ab_freeze.json"
H3_ROOT = ROOT / "results/4070ti_step3ct_seed1/t5h/h3"
ROUTE_PATH = H3_ROOT / "step3ct_t5h_h3_route_decision.json"
EXECUTION_PATH = H3_ROOT / "step3ct_t5h_h3_execution_record.json"
REPORT_PATH = H3_ROOT / "Step3CT_T5H_H3_quality_utility_ab_freeze_report.md"
PACKAGE_PATH = H3_ROOT / "step3ct_t5h_h3_package_manifest.json"

H3Q_ROUTE = "H3Q1_four_model_sessions_qualified_for_quality_utility_attempts"
H3_ROUTE = "H3A1_all_ab_frozen_authorize_disagreement_c"
FAILURE_ROUTE = "T5R2_review_completion_or_schema_stop"
DOMAIN_ROWS = {"safety": 630, "quality": 330, "utility": 330}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.relative_to(common.REPO).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": common.sha256_file(path), "bytes": path.stat().st_size}


def validate_all_records(domains: dict[str, Any]) -> None:
    if set(domains) != set(DOMAIN_ROWS):
        raise ValueError("all-A/B freeze requires safety, quality, and utility")
    records: list[dict[str, Any]] = []
    for domain, expected_rows in DOMAIN_ROWS.items():
        reviewers = domains[domain].get("reviewers", {})
        if set(reviewers) != {"a", "b"}:
            raise ValueError(f"{domain} requires reviewers a and b")
        for role in ("a", "b"):
            record = reviewers[role]
            if int(record.get("rows", -1)) != expected_rows:
                raise ValueError(f"{domain}_{role} row count mismatch")
            records.append(record)
    reviewer_ids = [str(record["anonymized_reviewer_id"]) for record in records]
    session_ids = [str(record["session_id"]) for record in records]
    if len(set(reviewer_ids)) != 6 or len(set(session_ids)) != 6:
        raise ValueError("reviewer or session identities are reused across domains")


def validate_entry() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    h3q = common.read_json(H3Q_ROUTE_PATH)
    if h3q.get("route") != H3Q_ROUTE or h3q.get("quality_utility_attempts_authorized") is not True:
        raise ValueError("H3 requires a valid H3Q launch authorization")

    frozen = review.verify_freeze(ALL_AB_FREEZE_PATH)
    domains = frozen.get("domains", {})
    validate_all_records(domains)
    source = frozen.get("source_freeze", {})
    if source.get("path") != relative(SAFETY_FREEZE_PATH):
        raise ValueError("all-A/B freeze does not reference the Safety freeze")
    if source.get("sha256") != common.sha256_file(SAFETY_FREEZE_PATH):
        raise ValueError("Safety freeze hash changed")
    if frozen.get("labels_compared_before_freeze") is not False:
        raise ValueError("A/B labels were compared before freeze")
    if frozen.get("outcome_key_joined") is not False:
        raise ValueError("outcome key was joined before reliability")
    if frozen.get("main_executor_assigned_or_edited_labels") is not False:
        raise ValueError("main executor edited labels")
    if frozen.get("heldout_opened") is not False:
        raise ValueError("heldout was opened")

    validated: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ROWS:
        validated[domain] = {}
        for role in ("a", "b"):
            rows, provenance = review.validate_reviewer(domain, role)
            current = review.reviewer_record(domain, role, rows, provenance)
            if current != domains[domain]["reviewers"][role]:
                raise ValueError(f"{domain}_{role} differs from all-A/B freeze")
            validated[domain][role] = current
    return frozen, validated


def run() -> None:
    _, reviewers = validate_entry()
    source_head = review.git_head()
    source_freeze = {"path": relative(ALL_AB_FREEZE_PATH), **artifact_record(ALL_AB_FREEZE_PATH)}

    route = {
        "name": "step3ct_t5h_h3_route_decision_v1",
        "source_git_head": source_head,
        "route": H3_ROUTE,
        "failure_route": FAILURE_ROUTE,
        "gates": {
            "h3q_launch_authorization_verified": True,
            "safety_a_b_complete_630_each": True,
            "quality_a_b_complete_330_each": True,
            "utility_a_b_complete_330_each": True,
            "six_reviewer_and_session_ids_distinct": True,
            "schemas_ids_hashes_and_provenance_valid": True,
            "labels_compared_before_freeze": False,
            "outcome_key_joined": False,
            "main_executor_assigned_or_edited_labels": False,
            "hidden_key_opened": False,
            "heldout_opened": False,
        },
        "counts": {
            "accepted_label_files": 6,
            "safety_rows_per_reviewer": 630,
            "quality_rows_per_reviewer": 330,
            "utility_rows_per_reviewer": 330,
            "accepted_rows_total": 2580,
            "rejected_files": 0,
            "partial_files": 0,
        },
        "source_freeze": source_freeze,
        "behavior_result_available": False,
        "review_reliability_available": False,
        "disagreement_c_authorized": True,
        "hidden_key_join_authorized": False,
        "t6_authorized": False,
        "new_training_or_generation_authorized": False,
        "next_action": "prepare_blinded_disagreement_only_c_packets",
    }
    common.write_json(ROUTE_PATH, route)

    execution = {
        "name": "step3ct_t5h_h3_execution_record_v1",
        "source_git_head": source_head,
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py freeze-quality-utility-ab",
            "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_quality_utility_freeze.py run",
            "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_quality_utility_freeze.py verify",
        ],
        "reviewers": reviewers,
        "labels_compared": False,
        "agreement_or_reliability_computed": False,
        "main_executor_semantic_label_review": False,
        "hidden_key_opened": False,
        "heldout_opened": False,
        "route": H3_ROUTE,
    }
    common.write_json(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H3 Quality and Utility A/B Freeze Report",
            "",
            f"Route: `{H3_ROUTE}`",
            "",
            "Four fresh no-context model reviewers returned complete 330-row Quality A/B and Utility A/B files "
            "with valid provenance. Together with the frozen Safety A/B pair, the all-A/B package contains six "
            "distinct reviewer and session identities.",
            "",
            "The existing T5R validator passed schema, opaque-ID order, value domain, hashes, completion, and "
            "independence checks for every file. A/B decisions were not compared before the all-A/B freeze.",
            "",
            "This route authorizes blinded disagreement-only Reviewer C packet construction. It does not provide "
            "agreement, reliability, behavior, candidate-selection, or causal conclusions.",
            "",
        ]
    )
    common.write_text(REPORT_PATH, report)

    package_paths: list[Path] = []
    for domain in ("quality", "utility"):
        for role in ("a", "b"):
            package_paths.extend([review.label_path(domain, role), review.provenance_path(domain, role)])
    package_paths.extend([ALL_AB_FREEZE_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH])
    package = {
        "name": "step3ct_t5h_h3_package_manifest_v1",
        "route": H3_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    common.write_json(PACKAGE_PATH, package)
    print(json.dumps({"route": H3_ROUTE, "reviewers": 6, "quality_utility_rows_each": 330}))


def verify() -> None:
    validate_entry()
    package = common.read_json(PACKAGE_PATH)
    route = common.read_json(ROUTE_PATH)
    execution = common.read_json(EXECUTION_PATH)
    failures: list[str] = []
    if package.get("route") != H3_ROUTE or package.get("artifact_count") != 12:
        failures.append("package_route_or_count")
    for path_text, expected in package.get("artifacts", {}).items():
        path = common.REPO / path_text
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(path_text)
    if route.get("route") != H3_ROUTE or route.get("behavior_result_available") is not False:
        failures.append("route_or_behavior_claim")
    if route.get("disagreement_c_authorized") is not True or route.get("t6_authorized") is not False:
        failures.append("authorization")
    if execution.get("labels_compared") is not False or execution.get("agreement_or_reliability_computed") is not False:
        failures.append("premature_label_analysis")
    for path_text in package.get("artifacts", {}):
        path = common.REPO / path_text
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{path_text}")
    if b"\r" in PACKAGE_PATH.read_bytes():
        failures.append(f"non_lf:{relative(PACKAGE_PATH)}")
    if failures:
        raise RuntimeError(f"H3 verification failed: {failures}")
    print(json.dumps({"route": H3_ROUTE, "package_artifacts": 12, "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
