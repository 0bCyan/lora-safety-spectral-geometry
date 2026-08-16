"""Prospective zero-access qualification for CSPR P2T Actionability reviewers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import step3_cspr_p2s_review as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2t/"
    "protocol/cspr_p2t_actionability_qualification_protocol.json"
)


class ActionQualificationError(ValueError):
    """Raised when an Actionability qualification artifact is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "qualify", "record-stop"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--reviewer-role", choices=["actionability_a", "actionability_b"])
    parser.add_argument("--agent-id")
    parser.add_argument("--reason")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise ActionQualificationError(f"path escapes repository: {value}") from exc
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


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise ActionQualificationError(f"frozen input hash mismatch: {name}")
    safety_freeze = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["safety_freeze"]["path"])
    )
    if safety_freeze.get("route") != (
        "P2T2_SAFETY_AB_FROZEN_AUTHORIZE_ACTIONABILITY_QUALIFICATION"
    ):
        raise ActionQualificationError("Safety A/B freeze does not authorize Actionability")
    if safety_freeze.get("labels_compared") is not False:
        raise ActionQualificationError("Safety labels were compared before Actionability qualification")
    if protocol.get("optimizer_steps") != 0 or protocol.get("training_authorized") is not False:
        raise ActionQualificationError("Actionability qualification violates zero training")
    return protocol


def validate_attestation(
    value: dict[str, Any], role: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "name",
        "reviewer_role",
        "agent_id",
        "session_id",
        "model_provider",
        "runtime_requested_model_id",
        "self_reported_model_id",
        "fork_context",
        "prior_study_involvement",
        "repository_material_opened",
        "study_rows_accessed",
        "other_reviewer_contact",
        "agrees_independent_complete_review",
        "agrees_no_partial_output",
        "agrees_no_external_retrieval",
        "agrees_forbidden_material_not_opened",
        "committed_packet_rows",
        "completion_status",
    }
    if set(value) != expected_fields:
        raise ActionQualificationError(f"{role} qualification schema mismatch")
    if value["name"] != "step3_cspr_p2t_zero_access_reviewer_attestation_v1":
        raise ActionQualificationError(f"{role} qualification name mismatch")
    if value["reviewer_role"] != role:
        raise ActionQualificationError(f"{role} qualification role mismatch")
    expected = protocol["prospective_reviewers"][role]
    if value["model_provider"] != "OpenAI":
        raise ActionQualificationError(f"{role} provider mismatch")
    if value["runtime_requested_model_id"] != expected["runtime_requested_model_id"]:
        raise ActionQualificationError(f"{role} requested model mismatch")
    for field in ("agent_id", "session_id", "self_reported_model_id"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ActionQualificationError(f"{role} missing {field}")
    for field in (
        "fork_context",
        "prior_study_involvement",
        "repository_material_opened",
        "other_reviewer_contact",
    ):
        if value[field] is not False:
            raise ActionQualificationError(f"{role} zero-access or independence claim failed")
    for field in (
        "agrees_independent_complete_review",
        "agrees_no_partial_output",
        "agrees_no_external_retrieval",
        "agrees_forbidden_material_not_opened",
    ):
        if value[field] is not True:
            raise ActionQualificationError(f"{role} completion contract was not accepted")
    if value["study_rows_accessed"] != 0:
        raise ActionQualificationError(f"{role} accessed study rows before qualification")
    if value["committed_packet_rows"] != 312:
        raise ActionQualificationError(f"{role} packet commitment mismatch")
    if value["completion_status"] != "qualified_zero_access":
        raise ActionQualificationError(f"{role} qualification is incomplete")
    if value["agent_id"] in set(protocol["forbidden_prior_agent_ids"]):
        raise ActionQualificationError(f"{role} reuses a prior author or reviewer")
    return value


def validate_pair(attestations: list[dict[str, Any]]) -> None:
    if len(attestations) != 2:
        raise ActionQualificationError("exactly two Actionability attestations are required")
    if len({row["agent_id"] for row in attestations}) != 2:
        raise ActionQualificationError("Actionability reviewer agent reuse is forbidden")
    if len({row["session_id"] for row in attestations}) != 2:
        raise ActionQualificationError("Actionability reviewer session reuse is forbidden")
    if len({row["runtime_requested_model_id"] for row in attestations}) != 2:
        raise ActionQualificationError("Actionability qualification requires cross-model diversity")


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    for role in ("actionability_a", "actionability_b"):
        attestation = resolve_repo_path(protocol["outputs"][f"{role}_attestation"])
        if attestation.exists():
            raise ActionQualificationError(f"{role} attestation exists before launch")
    result = {
        "status": "PASS_CSPR_P2T_ACTIONABILITY_QUALIFICATION_PREFLIGHT",
        "reviewers_authorized_for_zero_access_qualification": 2,
        "study_rows_accessed": 0,
        "packet_access_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def qualify(path: Path) -> dict[str, Any]:
    protocol = context(path)
    attestations = []
    artifacts = []
    for role in ("actionability_a", "actionability_b"):
        attestation_path = resolve_repo_path(protocol["outputs"][f"{role}_attestation"])
        row = validate_attestation(read_json(attestation_path), role, protocol)
        attestations.append(row)
        artifacts.append(
            {"path": relative(attestation_path), "sha256": base.sha256(attestation_path)}
        )
    validate_pair(attestations)
    result = {
        "name": "step3_cspr_p2t_actionability_reviewer_qualification_v1",
        "route": "P2T3_TWO_ACTIONABILITY_REVIEWERS_QUALIFIED",
        "reviewers": {
            row["reviewer_role"]: {
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "runtime_requested_model_id": row["runtime_requested_model_id"],
                "self_reported_model_id": row["self_reported_model_id"],
            }
            for row in attestations
        },
        "attestation_artifacts": artifacts,
        "study_rows_accessed_before_qualification": 0,
        "one_complete_attempt_per_reviewer": True,
        "packet_access_authorized_after_qualification_and_execution_push": True,
        "reviewer_c_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(protocol["outputs"]["registry"]), result)
    report = """# CSPR P2T Actionability Reviewer Qualification

Route: `P2T3_TWO_ACTIONABILITY_REVIEWERS_QUALIFIED`.

Two distinct cross-model, zero-access sessions accepted one complete-or-abort 312-row Actionability review. No repository material or study row was opened during qualification.

Packet access requires a separate hash-bound execution protocol to be committed and pushed. Reviewer C, hidden-key analysis, gradients, and training remain unauthorized. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(protocol["outputs"]["report"]), report)
    print(json.dumps({"route": result["route"], "reviewers": 2}, sort_keys=True))
    return result


def record_stop(
    path: Path, role: str | None, agent_id: str | None, reason: str | None
) -> dict[str, Any]:
    protocol = context(path)
    if not role or not agent_id or not reason:
        raise ActionQualificationError("qualification stop requires role, agent ID, and reason")
    result = {
        "name": "step3_cspr_p2t_actionability_qualification_stop_v1",
        "route": "STOP_CSPR_P2T_ACTIONABILITY_QUALIFICATION",
        "reviewer_role": role,
        "agent_id": agent_id,
        "reason": reason,
        "study_rows_accessed": 0,
        "replacement_reviewer_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(protocol["outputs"]["stop_incident"]), result)
    print(json.dumps({"route": result["route"]}, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "qualify":
            qualify(path)
        else:
            record_stop(path, args.reviewer_role, args.agent_id, args.reason)
    except (
        ActionQualificationError,
        FileExistsError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2T_ACTIONABILITY_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
