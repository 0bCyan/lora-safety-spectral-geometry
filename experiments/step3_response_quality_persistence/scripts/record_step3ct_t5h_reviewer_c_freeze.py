"""Freeze complete Reviewer C outputs before reliability analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import run_step3ct_t5r_review as review
import step3ct_t5r_common as common


ROOT = common.REPO / "experiments/step3_response_quality_persistence"
CONFIG_PATH = ROOT / "configs/step3ct_t5h_reviewer_c_model_registry.json"
H4Q_PACKAGE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5h/h4q/step3ct_t5h_h4q_package_manifest.json"
H4Q_ROUTE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5h/h4q/step3ct_t5h_h4q_route_decision.json"
ALL_AB_FREEZE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_all_ab_freeze.json"
C_PACKET_FREEZE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_c_packet_freeze.json"
ALL_C_FREEZE_PATH = ROOT / "results/4070ti_step3ct_seed1/t5r/freezes/step3ct_t5r_all_c_freeze.json"
H5_ROOT = ROOT / "results/4070ti_step3ct_seed1/t5h/h5"
ROUTE_PATH = H5_ROOT / "step3ct_t5h_h5_route_decision.json"
EXECUTION_PATH = H5_ROOT / "step3ct_t5h_h5_execution_record.json"
REPORT_PATH = H5_ROOT / "Step3CT_T5H_H5_reviewer_c_freeze_report.md"
PACKAGE_PATH = H5_ROOT / "step3ct_t5h_h5_package_manifest.json"

H4Q_COMMIT = "3dbf39753090a4457e82351e68c8237d663ebe70"
H4Q_ROUTE = "H4Q1_three_model_sessions_qualified_for_disagreement_c"
H5_ROUTE = "H5A1_all_c_frozen_authorize_reliability"
FAILURE_ROUTE = "T5R2_review_completion_or_schema_stop"
DOMAINS = ("safety", "quality", "utility")


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


def validate_h4q_package() -> tuple[dict[str, Any], dict[str, Any]]:
    package = read_json(H4Q_PACKAGE_PATH)
    route = read_json(H4Q_ROUTE_PATH)
    if package.get("route") != H4Q_ROUTE or package.get("artifact_count") != 4:
        raise ValueError("invalid H4Q package route or artifact count")
    for path_text, expected in package.get("artifacts", {}).items():
        path = common.REPO / path_text
        if not path.is_file() or artifact_record(path) != expected:
            raise ValueError(f"H4Q artifact mismatch: {path_text}")
    if route.get("route") != H4Q_ROUTE or route.get("reviewer_c_attempts_authorized") is not True:
        raise ValueError("H4Q does not authorize Reviewer C attempts")
    if route.get("label_acceptance_authorized") is not False or route.get("reliability_authorized") is not False:
        raise ValueError("H4Q contains premature label or reliability authorization")
    return package, route


def validate_sources(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ab_freeze = read_json(ALL_AB_FREEZE_PATH)
    c_freeze = read_json(C_PACKET_FREEZE_PATH)
    if common.sha256_file(ALL_AB_FREEZE_PATH) != config["source_ab_freeze"]["sha256"]:
        raise ValueError("all-A/B freeze hash changed")
    if common.sha256_file(C_PACKET_FREEZE_PATH) != config["c_packet_freeze"]["sha256"]:
        raise ValueError("C packet freeze hash changed")
    if ab_freeze.get("outcome_key_joined") is not False or ab_freeze.get("heldout_opened") is not False:
        raise ValueError("all-A/B freeze opened a prohibited identity or heldout")
    if c_freeze.get("a_b_labels_or_reasons_visible") is not False:
        raise ValueError("C packet freeze exposes A/B decisions")
    if c_freeze.get("outcome_key_joined") is not False or c_freeze.get("heldout_opened") is not False:
        raise ValueError("C packet freeze opened a prohibited identity or heldout")
    return ab_freeze, c_freeze


def validate_reviewer_identity(
    domain: str,
    provenance: dict[str, Any],
    registered: dict[str, Any],
) -> None:
    exact = {
        "domain": domain,
        "reviewer_role": f"{domain}_c",
        "anonymized_reviewer_id": registered["anonymized_reviewer_id"],
        "reviewer_type": registered["reviewer_type"],
        "model_provider": registered["model_provider"],
        "model_id_version": registered["model_id_version_for_final_provenance"],
        "session_id": registered["session_id"],
    }
    for field, expected in exact.items():
        if provenance.get(field) != expected:
            raise ValueError(f"{domain}_c provenance {field} differs from H4Q")


def validate_complete_c(
    config: dict[str, Any],
    ab_freeze: dict[str, Any],
    c_freeze: dict[str, Any],
) -> dict[str, Any]:
    ab_records = [
        record
        for domain in DOMAINS
        for record in ab_freeze["domains"][domain]["reviewers"].values()
    ]
    prohibited_sessions = {str(record["session_id"]) for record in ab_records}
    prohibited_reviewers = {str(record["anonymized_reviewer_id"]) for record in ab_records}
    c_records: dict[str, Any] = {}
    for domain in DOMAINS:
        rows, provenance = review.validate_reviewer(domain, "c")
        registered = config["reviewers"][f"{domain}_c"]
        material = config["material_freeze"][domain]
        validate_reviewer_identity(domain, provenance, registered)
        if len(rows) != int(material["packet_rows"]):
            raise ValueError(f"{domain}_c row count differs from H4Q")
        if len(rows) != int(c_freeze["domains"][domain]["packet_rows"]):
            raise ValueError(f"{domain}_c row count differs from C packet freeze")
        if str(provenance["session_id"]) in prohibited_sessions:
            raise ValueError(f"{domain}_c reuses an A/B session")
        if str(provenance["anonymized_reviewer_id"]) in prohibited_reviewers:
            raise ValueError(f"{domain}_c reuses an A/B reviewer identity")
        record = review.reviewer_record(domain, "c", rows, provenance)
        c_records[domain] = {
            "packet_rows": int(material["packet_rows"]),
            "decision_field_disagreements": int(material["decision_field_disagreements"]),
            "reviewer": record,
        }

    session_ids = [str(item["reviewer"]["session_id"]) for item in c_records.values()]
    reviewer_ids = [str(item["reviewer"]["anonymized_reviewer_id"]) for item in c_records.values()]
    if len(set(session_ids)) != 3 or len(set(reviewer_ids)) != 3:
        raise ValueError("Reviewer C identities are not distinct across domains")
    return c_records


def validate_entry(require_pushed_boundary: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_h4q_package()
    config = read_json(CONFIG_PATH)
    if config.get("qualification_route") != H4Q_ROUTE:
        raise ValueError("Reviewer C registry route mismatch")
    if require_pushed_boundary:
        if git("rev-parse", "HEAD") != H4Q_COMMIT:
            raise ValueError("H5 must execute directly from the frozen H4Q commit")
        if git("rev-parse", "@{upstream}") != H4Q_COMMIT:
            raise ValueError("H4Q must be pushed before H5 execution")
    ab_freeze, c_freeze = validate_sources(config)
    records = validate_complete_c(config, ab_freeze, c_freeze)
    return config, records


def write_all_c_freeze(records: dict[str, Any], source_head: str) -> None:
    common.write_json(
        ALL_C_FREEZE_PATH,
        {
            "name": "step3ct_t5r_all_c_freeze_v1",
            "execution_git_head": source_head,
            "source_h4q": {"path": relative(H4Q_PACKAGE_PATH), **artifact_record(H4Q_PACKAGE_PATH)},
            "source_c_packet_freeze": {
                "path": relative(C_PACKET_FREEZE_PATH),
                **artifact_record(C_PACKET_FREEZE_PATH),
            },
            "domains": records,
            "counts": {
                "accepted_label_files": 3,
                "accepted_provenance_files": 3,
                "accepted_rows_total": sum(int(item["packet_rows"]) for item in records.values()),
                "decision_field_disagreements_total": sum(
                    int(item["decision_field_disagreements"]) for item in records.values()
                ),
                "rejected_files": 0,
                "partial_files": 0,
            },
            "coordinator_semantic_label_review": False,
            "reliability_computed_before_freeze": False,
            "outcome_key_joined": False,
            "heldout_opened": False,
        },
    )


def run() -> None:
    _, records = validate_entry(require_pushed_boundary=True)
    source_head = git("rev-parse", "HEAD")
    write_all_c_freeze(records, source_head)

    route = {
        "name": "step3ct_t5h_h5_route_decision_v1",
        "source_git_head": source_head,
        "route": H5_ROUTE,
        "failure_route": FAILURE_ROUTE,
        "gates": {
            "h4q_launch_authorization_verified": True,
            "safety_c_complete_11": True,
            "quality_c_complete_69": True,
            "utility_c_complete_33": True,
            "three_c_reviewer_and_session_ids_distinct": True,
            "c_sessions_distinct_from_all_ab_and_failed_t5_sessions": True,
            "schemas_ids_hashes_and_provenance_valid": True,
            "all_c_outputs_frozen_before_reliability": True,
            "coordinator_semantic_label_review": False,
            "outcome_key_joined": False,
            "hidden_key_opened": False,
            "heldout_opened": False,
        },
        "counts": {
            "accepted_label_files": 3,
            "accepted_rows_total": 113,
            "decision_field_disagreements_total": 156,
            "rejected_files": 0,
            "partial_files": 0,
        },
        "source_freeze": {"path": relative(ALL_C_FREEZE_PATH), **artifact_record(ALL_C_FREEZE_PATH)},
        "behavior_result_available": False,
        "review_reliability_available": False,
        "reliability_authorized_after_push": True,
        "hidden_key_join_authorized": False,
        "t6_authorized": False,
        "new_training_or_generation_authorized": False,
        "next_action": "push_h5_then_run_frozen_t5r_finalize",
    }
    common.write_json(ROUTE_PATH, route)

    execution = {
        "name": "step3ct_t5h_h5_execution_record_v1",
        "source_git_head": source_head,
        "commands": [
            "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_reviewer_c_freeze.py run",
            "python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_reviewer_c_freeze.py verify",
        ],
        "reviewers": records,
        "labels_compared_or_summarized": False,
        "agreement_or_reliability_computed": False,
        "main_executor_semantic_label_review": False,
        "hidden_key_opened": False,
        "heldout_opened": False,
        "route": H5_ROUTE,
    }
    common.write_json(EXECUTION_PATH, execution)

    report = "\n".join(
        [
            "# Step3C-T5H H5 Reviewer C Freeze Report",
            "",
            f"Route: `{H5_ROUTE}`",
            "",
            "Three prospectively qualified no-context Reviewer C sessions returned complete Safety, Quality, "
            "and Utility adjudication files with 11, 69, and 33 rows. All three label files and provenance "
            "records passed the existing T5R schema, opaque-ID order, field-domain, hash, completion, and "
            "independence validators.",
            "",
            "The six Reviewer C artifacts are now frozen before reliability analysis. The coordinator performed "
            "mechanical validation only and did not inspect, edit, summarize, or compare label semantics. Hidden "
            "keys and heldout remain sealed.",
            "",
            "After this package is committed and pushed, the unchanged T5R finalization script may compute the "
            "prespecified A/B reliability gates and C coverage. H5 itself is not a behavior result and does not "
            "authorize hidden-key join, T6, or new training.",
            "",
        ]
    )
    common.write_text(REPORT_PATH, report)

    package_paths: list[Path] = []
    for domain in DOMAINS:
        package_paths.extend([review.label_path(domain, "c"), review.provenance_path(domain, "c")])
    package_paths.extend([ALL_C_FREEZE_PATH, ROUTE_PATH, EXECUTION_PATH, REPORT_PATH])
    package = {
        "name": "step3ct_t5h_h5_package_manifest_v1",
        "route": H5_ROUTE,
        "canonical_lf": True,
        "artifact_count": len(package_paths),
        "artifacts": {relative(path): artifact_record(path) for path in package_paths},
    }
    common.write_json(PACKAGE_PATH, package)
    print(json.dumps({"route": H5_ROUTE, "reviewers": 3, "accepted_rows": 113}))


def verify() -> None:
    _, records = validate_entry(require_pushed_boundary=False)
    freeze = read_json(ALL_C_FREEZE_PATH)
    package = read_json(PACKAGE_PATH)
    route = read_json(ROUTE_PATH)
    execution = read_json(EXECUTION_PATH)
    failures: list[str] = []
    if package.get("route") != H5_ROUTE or package.get("artifact_count") != 10:
        failures.append("package_route_or_count")
    for path_text, expected in package.get("artifacts", {}).items():
        path = common.REPO / path_text
        if not path.is_file() or artifact_record(path) != expected:
            failures.append(path_text)
    if route.get("route") != H5_ROUTE or route.get("behavior_result_available") is not False:
        failures.append("route_or_behavior_claim")
    if route.get("reliability_authorized_after_push") is not True or route.get("t6_authorized") is not False:
        failures.append("authorization")
    if execution.get("labels_compared_or_summarized") is not False:
        failures.append("premature_label_analysis")
    if execution.get("agreement_or_reliability_computed") is not False:
        failures.append("premature_reliability")
    if freeze.get("domains") != records:
        failures.append("frozen_reviewer_records")
    if freeze.get("outcome_key_joined") is not False or freeze.get("heldout_opened") is not False:
        failures.append("identity_or_heldout")
    for path_text in package.get("artifacts", {}):
        path = common.REPO / path_text
        if b"\r" in path.read_bytes():
            failures.append(f"non_lf:{path_text}")
    if b"\r" in PACKAGE_PATH.read_bytes():
        failures.append(f"non_lf:{relative(PACKAGE_PATH)}")
    if failures:
        raise RuntimeError(f"H5 verification failed: {failures}")
    print(json.dumps({"route": H5_ROUTE, "package_artifacts": 10, "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
