"""Prospective qualification gates for CSPR P2T cross-model review recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2t/"
    "protocol/cspr_p2t_cross_model_review_recovery_protocol.json"
)


class RecoveryValidationError(ValueError):
    """Raised when a P2T recovery artifact violates the prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["preflight", "qualify-safety", "record-qualification-stop"]
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--reviewer-role", choices=["safety_a", "safety_b"])
    parser.add_argument("--agent-id")
    parser.add_argument("--reason")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise RecoveryValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def git_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RecoveryValidationError(f"frozen input hash mismatch: {name}")

    incident = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p2s_inconclusive_incident"]["path"])
    )
    if incident.get("route") != "STOP_CSPR_P2S_REVIEW_INCONCLUSIVE":
        raise RecoveryValidationError("P2T does not descend from the frozen P2S stop route")
    if incident.get("heuristic_substitution_used") is not False:
        raise RecoveryValidationError("P2S incident used a forbidden heuristic substitution")

    review_protocol_path = resolve_repo_path(
        protocol["frozen_inputs"]["p2s_review_protocol"]["path"]
    )
    manifest = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p2s_packet_manifest"]["path"])
    )
    if manifest.get("protocol_sha256") != sha256(review_protocol_path):
        raise RecoveryValidationError("packet manifest does not bind the unchanged review protocol")
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RecoveryValidationError(f"packet artifact hash mismatch: {artifact['path']}")

    if not git_is_ancestor(protocol["entry_commit"]):
        raise RecoveryValidationError("P2T entry commit is not an ancestor of HEAD")
    if protocol.get("optimizer_steps") != 0 or protocol.get("training_authorized") is not False:
        raise RecoveryValidationError("P2T violates the zero-training boundary")
    if protocol.get("prior_p2s_labels_eligible") is not False:
        raise RecoveryValidationError("prior P2S labels must be excluded")
    excluded = {
        artifact["path"]: artifact["sha256"]
        for artifact in protocol["excluded_prior_label_artifacts"]
    }
    for key in ("prior_action_a_labels", "prior_action_a_provenance", "prior_action_b_partial"):
        artifact = protocol["frozen_inputs"][key]
        if excluded.get(artifact["path"]) != artifact["sha256"]:
            raise RecoveryValidationError(f"prior evidence is not explicitly excluded: {key}")
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
        raise RecoveryValidationError(f"{role} qualification schema mismatch")
    if value["name"] != "step3_cspr_p2t_zero_access_reviewer_attestation_v1":
        raise RecoveryValidationError(f"{role} qualification name mismatch")
    if value["reviewer_role"] != role:
        raise RecoveryValidationError(f"{role} qualification role mismatch")
    expected = protocol["prospective_reviewers"][role]
    if value["model_provider"] != "OpenAI":
        raise RecoveryValidationError(f"{role} provider mismatch")
    if value["runtime_requested_model_id"] != expected["runtime_requested_model_id"]:
        raise RecoveryValidationError(f"{role} requested model mismatch")
    for field in ("agent_id", "session_id", "self_reported_model_id"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise RecoveryValidationError(f"{role} missing {field}")
    false_fields = (
        "fork_context",
        "prior_study_involvement",
        "repository_material_opened",
        "other_reviewer_contact",
    )
    if any(value[field] is not False for field in false_fields):
        raise RecoveryValidationError(f"{role} zero-access or independence claim failed")
    true_fields = (
        "agrees_independent_complete_review",
        "agrees_no_partial_output",
        "agrees_no_external_retrieval",
        "agrees_forbidden_material_not_opened",
    )
    if any(value[field] is not True for field in true_fields):
        raise RecoveryValidationError(f"{role} completion contract was not accepted")
    if value["study_rows_accessed"] != 0:
        raise RecoveryValidationError(f"{role} accessed study rows before qualification freeze")
    if value["committed_packet_rows"] != 424:
        raise RecoveryValidationError(f"{role} packet commitment mismatch")
    if value["completion_status"] != "qualified_zero_access":
        raise RecoveryValidationError(f"{role} qualification is incomplete")
    if value["agent_id"] in set(protocol["forbidden_prior_agent_ids"]):
        raise RecoveryValidationError(f"{role} reuses a forbidden prior agent")
    return value


def validate_pair(attestations: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    if len(attestations) != 2:
        raise RecoveryValidationError("exactly two Safety attestations are required")
    if len({row["agent_id"] for row in attestations}) != 2:
        raise RecoveryValidationError("Safety reviewer agent reuse is forbidden")
    if len({row["session_id"] for row in attestations}) != 2:
        raise RecoveryValidationError("Safety reviewer session reuse is forbidden")
    if len({row["runtime_requested_model_id"] for row in attestations}) != 2:
        raise RecoveryValidationError("P2T Safety qualification requires cross-model diversity")


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    result = {
        "status": "PASS_CSPR_P2T_P0_PREFLIGHT",
        "entry_commit": protocol["entry_commit"],
        "safety_reviewers_authorized": False,
        "study_rows_accessed": 0,
        "prior_p2s_labels_eligible": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def qualify_safety(path: Path) -> dict[str, Any]:
    protocol = context(path)
    attestations = []
    artifacts = []
    for role in ("safety_a", "safety_b"):
        attestation_path = resolve_repo_path(
            protocol["qualification_outputs"][f"{role}_attestation"]
        )
        value = validate_attestation(read_json(attestation_path), role, protocol)
        attestations.append(value)
        artifacts.append(
            {"path": relative(attestation_path), "sha256": sha256(attestation_path)}
        )
    validate_pair(attestations, protocol)
    registry = {
        "name": "step3_cspr_p2t_safety_reviewer_qualification_v1",
        "route": "P2T1_TWO_SAFETY_REVIEWERS_QUALIFIED",
        "entry_commit": protocol["entry_commit"],
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
        "one_complete_attempt_per_reviewer": True,
        "study_rows_accessed_before_qualification": 0,
        "prior_p2s_labels_eligible": False,
        "safety_review_authorized_after_qualification_push": True,
        "actionability_review_authorized": False,
        "reviewer_c_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    registry_path = resolve_repo_path(protocol["qualification_outputs"]["registry"])
    write_json_frozen(registry_path, registry)
    report = f"""# CSPR P2T Safety Reviewer Qualification

Route: `P2T1_TWO_SAFETY_REVIEWERS_QUALIFIED`.

Two distinct zero-access, no-context model sessions accepted the complete-or-abort contract for the unchanged 424-row Safety packets. Requested runtimes are `{attestations[0]['runtime_requested_model_id']}` and `{attestations[1]['runtime_requested_model_id']}`. No repository material or study row was opened during qualification.

Safety review is authorized only after this qualification package and a hash-bound execution protocol are committed and pushed. Actionability, Reviewer C, hidden-key analysis, gradients, and training remain unauthorized. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(protocol["qualification_outputs"]["report"]), report)
    print(json.dumps({"route": registry["route"], "reviewers": 2}, sort_keys=True))
    return registry


def record_qualification_stop(
    path: Path, role: str | None, agent_id: str | None, reason: str | None
) -> dict[str, Any]:
    protocol = context(path)
    if not role or not agent_id or not reason:
        raise RecoveryValidationError("qualification stop requires role, agent ID, and reason")
    result = {
        "name": "step3_cspr_p2t_qualification_stop_v1",
        "route": "STOP_CSPR_P2T_QUALIFICATION",
        "reviewer_role": role,
        "agent_id": agent_id,
        "reason": reason,
        "study_rows_accessed": 0,
        "replacement_reviewer_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(
        resolve_repo_path(protocol["qualification_outputs"]["stop_incident"]), result
    )
    print(json.dumps({"route": result["route"]}, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "qualify-safety":
            qualify_safety(path)
        else:
            record_qualification_stop(path, args.reviewer_role, args.agent_id, args.reason)
    except (
        RecoveryValidationError,
        FileExistsError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2T_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
