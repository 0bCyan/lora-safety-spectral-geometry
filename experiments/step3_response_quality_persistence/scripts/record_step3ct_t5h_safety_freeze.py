"""Record the T5H2 route after complete Safety A/B validation and freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_step3ct_t5r_review as review
import step3ct_t5r_common as common


ROOT = common.REPO / "experiments/step3_response_quality_persistence"
H1Q_ROUTE_PATH = (
    ROOT
    / "results/4070ti_step3ct_seed1/t5h/h1q/step3ct_t5h_h1q_route_decision.json"
)
SAFETY_FREEZE_PATH = (
    ROOT
    / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_safety_ab_freeze.json"
)
H2_ROOT = ROOT / "results/4070ti_step3ct_seed1/t5h/h2"
ROUTE_PATH = H2_ROOT / "step3ct_t5h_h2_route_decision.json"
EXECUTION_PATH = H2_ROOT / "step3ct_t5h_h2_execution_record.json"
REPORT_PATH = H2_ROOT / "Step3CT_T5H_H2_safety_ab_freeze_report.md"
PACKAGE_PATH = H2_ROOT / "step3ct_t5h_h2_package_manifest.json"

H1Q_ROUTE = "H1Q1_two_model_sessions_qualified_for_complete_safety_attempt"
H2_ROUTE = "T5H2_safety_ab_frozen_authorize_quality_utility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.relative_to(common.REPO).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": common.sha256_file(path), "bytes": path.stat().st_size}


def validate_pair_records(records: dict[str, dict[str, Any]]) -> None:
    if set(records) != {"a", "b"}:
        raise ValueError("Safety freeze requires reviewers a and b")
    reviewer_ids = [str(records[role]["anonymized_reviewer_id"]) for role in ("a", "b")]
    session_ids = [str(records[role]["session_id"]) for role in ("a", "b")]
    if len(set(reviewer_ids)) != 2 or len(set(session_ids)) != 2:
        raise ValueError("Safety reviewer or session identities are not distinct")
    if any(int(records[role]["rows"]) != 630 for role in ("a", "b")):
        raise ValueError("Safety reviewer output is not exactly 630 rows")


def validate_entry() -> tuple[dict[str, Any], dict[str, Any]]:
    h1q = common.read_json(H1Q_ROUTE_PATH)
    if h1q.get("route") != H1Q_ROUTE or h1q.get("safety_attempt_authorized") is not True:
        raise ValueError("H2 requires a valid H1Q launch authorization")

    frozen = review.verify_freeze(SAFETY_FREEZE_PATH)
    records = frozen.get("domains", {}).get("safety", {}).get("reviewers", {})
    validate_pair_records(records)
    if frozen.get("labels_compared_before_freeze") is not False:
        raise ValueError("Safety labels were compared before freeze")
    if frozen.get("outcome_key_joined") is not False:
        raise ValueError("Outcome key was joined before review reliability")
    if frozen.get("main_executor_assigned_or_edited_labels") is not False:
        raise ValueError("Main executor edited labels")
    if frozen.get("heldout_opened") is not False:
        raise ValueError("Heldout was opened")

    validated: dict[str, Any] = {}
    for role in ("a", "b"):
        rows, provenance = review.validate_reviewer("safety", role)
        current = review.reviewer_record("safety", role, rows, provenance)
        if current != records[role]:
            raise ValueError(f"Safety reviewer {role} differs from freeze")
        validated[role] = current
    return frozen, validated


def run() -> None:
    frozen, reviewers = validate_entry()
    source_head = review.git_head()
    source_freeze = {"path": relative(SAFETY_FREEZE_PATH), **artifact_record(SAFETY_FREEZE_PATH)}

    route = {
        "name": "step3ct_t5h_h2_route_decision_v1",
        "source_git_head": source_head,
        "route": H2_ROUTE,
        "gates": {
            "h1q_launch_authorization_verified": True,
            "reviewer_a_complete_630": True,
            "reviewer_b_complete_630": True,
            "reviewer_and_session_ids_distinct": True,
            "schemas_ids_hashes_and_provenance_valid": True,
            "labels_compared_before_freeze": False,
            "outcome_key_joined": False,
            "main_executor_assigned_or_edited_labels": False,
            "quality_utility_opened": False,
            "hidden_key_opened": False,
            "heldout_opened": False,
        },
        "counts": {
            "accepted_label_files": 2,
            "accepted_rows_per_reviewer": 630,
            "accepted_rows_total": 1260,
            "rejected_files": 0,
            "partial_files": 0,
        },
        "source_freeze": source_freeze,
        "behavior_result_available": False,
        "safety_ab_reliability_available": False,
        "quality_utility_review_authorized": True,
        "t6_authorized": False,
        "new_training_or_generation_authorized": False,
        "next_action": "launch_separate_quality_and_utility_ab_sessions",
    }
    common.write_json(ROUTE_PATH, route)

    execution = {
        "name": "step3ct_t5h_h2_execution_record_v1",
        "source_git_head": source_head,
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py freeze-safety-ab",
            "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_safety_freeze.py run",
            "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_safety_freeze.py verify",
        ],
        "reviewers": reviewers,
        "labels_compared": False,
        "agreement_or_reliability_computed": False,
        "main_executor_semantic_label_review": False,
        "quality_packet_opened": False,
        "utility_packet_opened": False,
        "hidden_key_opened": False,
        "heldout_opened": False,
        "route": H2_ROUTE,
    }
    common.write_json(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H2 Safety A/B Freeze Report",
            "",
            f"Route: `{H2_ROUTE}`",
            "",
            "Two distinct no-context model reviewers each returned one complete 630-row Safety label file and valid "
            "provenance. The existing T5R validator passed schema, opaque-ID order, value domain, input/output hash, "
            "session, model, completion, and independence checks for both files.",
            "",
            "A/B decisions were not compared before freeze. No agreement metric, hidden-key join, behavior endpoint, "
            "candidate selection, training, generation, or heldout evaluation was performed.",
            "",
            "This route authorizes separate Quality A/B and Utility A/B review sessions. It is a review-completion "
            "milestone, not evidence that C3 is behaviorally better or worse than C0-R.",
            "",
        ]
    )
    common.write_text(REPORT_PATH, report)

    package_paths = [
        review.label_path("safety", "a"),
        review.provenance_path("safety", "a"),
        review.label_path("safety", "b"),
        review.provenance_path("safety", "b"),
        SAFETY_FREEZE_PATH,
        ROUTE_PATH,
        EXECUTION_PATH,
        REPORT_PATH,
    ]
    package = {
        "name": "step3ct_t5h_h2_package_manifest_v1",
        "route": H2_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    common.write_json(PACKAGE_PATH, package)
    print(json.dumps({"route": H2_ROUTE, "reviewers": 2, "rows_each": 630}))


def verify() -> None:
    validate_entry()
    package = common.read_json(PACKAGE_PATH)
    route = common.read_json(ROUTE_PATH)
    execution = common.read_json(EXECUTION_PATH)
    failures: list[str] = []
    if package.get("route") != H2_ROUTE or package.get("artifact_count") != 8:
        failures.append("package_route_or_count")
    for path_text, expected in package.get("artifacts", {}).items():
        path = common.REPO / path_text
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(path_text)
    if route.get("route") != H2_ROUTE or route.get("behavior_result_available") is not False:
        failures.append("route_or_behavior_claim")
    if route.get("quality_utility_review_authorized") is not True or route.get("t6_authorized") is not False:
        failures.append("authorization")
    if execution.get("labels_compared") is not False or execution.get("agreement_or_reliability_computed") is not False:
        failures.append("premature_label_analysis")
    for path in (SAFETY_FREEZE_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH, PACKAGE_PATH):
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{relative(path)}")
    if failures:
        raise RuntimeError(f"H2 verification failed: {failures}")
    print(json.dumps({"route": H2_ROUTE, "package_artifacts": 8, "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
