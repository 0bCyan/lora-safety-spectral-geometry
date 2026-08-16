"""Validate the prospectively qualified CSPR P2T Safety A/B review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import step3_cspr_p2s_review as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTION_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2t/"
    "protocol/cspr_p2t_safety_review_execution_protocol.json"
)


class SafetyReviewValidationError(ValueError):
    """Raised when the qualified P2T Safety review is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "freeze-safety", "record-inconclusive"])
    parser.add_argument("--execution-protocol", default=DEFAULT_EXECUTION_PROTOCOL)
    parser.add_argument("--reviewer-role", choices=["a", "b"])
    parser.add_argument("--agent-id")
    parser.add_argument("--reason")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise SafetyReviewValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = read_json(path)
    for name, artifact in execution["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise SafetyReviewValidationError(f"frozen execution input hash mismatch: {name}")
    review_protocol_path = resolve_repo_path(
        execution["frozen_inputs"]["review_protocol"]["path"]
    )
    manifest_path = resolve_repo_path(
        execution["frozen_inputs"]["packet_manifest"]["path"]
    )
    protocol = read_json(review_protocol_path)
    manifest = read_json(manifest_path)
    if manifest.get("protocol_sha256") != base.sha256(review_protocol_path):
        raise SafetyReviewValidationError("packet manifest does not bind the review protocol")
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise SafetyReviewValidationError(f"packet artifact hash mismatch: {artifact['path']}")
    qualification = read_json(
        resolve_repo_path(execution["frozen_inputs"]["qualification_registry"]["path"])
    )
    if qualification.get("route") != "P2T1_TWO_SAFETY_REVIEWERS_QUALIFIED":
        raise SafetyReviewValidationError("Safety reviewers are not prospectively qualified")
    if execution.get("optimizer_steps") != 0 or execution.get("training_authorized") is not False:
        raise SafetyReviewValidationError("Safety execution violates the zero-training boundary")
    return execution, protocol, manifest


def validate_identity(
    provenances: list[dict[str, Any]], execution: dict[str, Any]
) -> None:
    if len(provenances) != 2:
        raise SafetyReviewValidationError("exactly two Safety provenances are required")
    by_role = {row.get("reviewer_role"): row for row in provenances}
    if set(by_role) != {"a", "b"}:
        raise SafetyReviewValidationError("Safety provenance roles are incomplete")
    for role in ("a", "b"):
        provenance = by_role[role]
        expected = execution["qualified_reviewers"][role]
        checks = {
            "agent_id": expected["agent_id"],
            "session_id": expected["session_id"],
            "model_provider": "OpenAI",
            "runtime_requested_model_id": expected["runtime_requested_model_id"],
            "self_reported_model_id": expected["self_reported_model_id"],
            "qualification_attestation_sha256": expected["qualification_attestation_sha256"],
            "complete_attempt_index": 1,
            "packet_access_after_qualification_push": True,
        }
        for field, value in checks.items():
            if provenance.get(field) != value:
                raise SafetyReviewValidationError(f"Safety {role} identity mismatch: {field}")
    try:
        base.ensure_reviewer_isolation(
            provenances, execution["forbidden_prior_agent_ids"]
        )
    except base.ReviewValidationError as exc:
        raise SafetyReviewValidationError(str(exc)) from exc
    if len({row["runtime_requested_model_id"] for row in provenances}) != 2:
        raise SafetyReviewValidationError("Safety A/B cross-model diversity is missing")


def review_spec(
    role: str,
    execution: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return base.review_spec("safety", role, execution, protocol)


def preflight(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    for role in ("a", "b"):
        spec = review_spec(role, execution, protocol)
        base.validate_packet_template(spec, manifest)
        if spec["output"].exists() or spec["provenance"].exists():
            raise SafetyReviewValidationError(f"Safety {role} output exists before launch")
    result = {
        "status": "PASS_CSPR_P2T_SAFETY_REVIEW_PREFLIGHT",
        "reviewer_a_model": execution["qualified_reviewers"]["a"][
            "runtime_requested_model_id"
        ],
        "reviewer_b_model": execution["qualified_reviewers"]["b"][
            "runtime_requested_model_id"
        ],
        "rows_per_reviewer": 424,
        "complete_attempts_per_reviewer": 1,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def freeze_safety(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    provenances = []
    artifacts = []
    for role in ("a", "b"):
        spec = review_spec(role, execution, protocol)
        rows, provenance = base.validate_review(spec, manifest)
        if len(rows) != 424:
            raise SafetyReviewValidationError(f"Safety {role} row count mismatch")
        provenances.append(provenance)
        artifacts.extend(
            [
                {
                    "path": relative(spec["output"]),
                    "rows": len(rows),
                    "sha256": base.sha256(spec["output"]),
                },
                {
                    "path": relative(spec["provenance"]),
                    "sha256": base.sha256(spec["provenance"]),
                },
            ]
        )
    validate_identity(provenances, execution)
    result = {
        "name": "step3_cspr_p2t_safety_ab_freeze_v1",
        "route": "P2T2_SAFETY_AB_FROZEN_AUTHORIZE_ACTIONABILITY_QUALIFICATION",
        "reviewers": [
            {
                "reviewer_role": row["reviewer_role"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "runtime_requested_model_id": row["runtime_requested_model_id"],
                "self_reported_model_id": row["self_reported_model_id"],
            }
            for row in provenances
        ],
        "artifacts": artifacts,
        "labels_compared": False,
        "agreement_computed": False,
        "hidden_key_joined": False,
        "actionability_qualification_authorized_after_push": True,
        "reviewer_c_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    output_path = resolve_repo_path(execution["freeze_outputs"]["manifest"])
    write_json_frozen(output_path, result)
    report = """# CSPR P2T Safety A/B Freeze

Route: `P2T2_SAFETY_AB_FROZEN_AUTHORIZE_ACTIONABILITY_QUALIFICATION`.

Two prospectively qualified cross-model reviewers each returned one complete 424-row Safety file with valid role, agent, session, model, qualification, schema, order, and hash provenance. A/B labels were not compared and no hidden key was opened.

After this freeze is committed and pushed, zero-access Actionability reviewer qualification is authorized. Agreement, Reviewer C, gradients, and training remain unauthorized. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(execution["freeze_outputs"]["report"]), report)
    print(json.dumps({"route": result["route"], "reviewers": 2}, sort_keys=True))
    return result


def record_inconclusive(
    path: Path, role: str | None, agent_id: str | None, reason: str | None
) -> dict[str, Any]:
    execution, _, _ = context(path)
    if not role or not agent_id or not reason:
        raise SafetyReviewValidationError("inconclusive record requires role, agent ID, and reason")
    result = {
        "name": "step3_cspr_p2t_safety_review_inconclusive_v1",
        "route": "STOP_CSPR_P2T_REVIEW_INCONCLUSIVE",
        "reviewer_role": role,
        "agent_id": agent_id,
        "reason": reason,
        "partial_output_accepted": False,
        "replacement_reviewer_started": False,
        "heuristic_substitution_used": False,
        "automated_recovery_attempts_remaining": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(execution["freeze_outputs"]["incident"]), result)
    print(json.dumps({"route": result["route"]}, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.execution_protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "freeze-safety":
            freeze_safety(path)
        else:
            record_inconclusive(path, args.reviewer_role, args.agent_id, args.reason)
    except (
        SafetyReviewValidationError,
        base.ReviewValidationError,
        FileExistsError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2T_SAFETY_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
