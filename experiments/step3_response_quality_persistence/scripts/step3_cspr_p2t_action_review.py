"""Validate the prospectively qualified CSPR P2T Actionability A/B review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import step3_cspr_p2s_review as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTION_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2t/"
    "protocol/cspr_p2t_actionability_review_execution_protocol.json"
)


class ActionReviewValidationError(ValueError):
    """Raised when a P2T Actionability review artifact is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "freeze-action", "record-inconclusive"])
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
        raise ActionReviewValidationError(f"path escapes repository: {value}") from exc
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


def require_canonical_lf(path: Path) -> None:
    data = path.read_bytes()
    if b"\r" in data:
        raise ActionReviewValidationError(f"non-canonical line ending: {path}")


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = read_json(path)
    for name, artifact in execution["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise ActionReviewValidationError(f"frozen input hash mismatch: {name}")
    protocol_path = resolve_repo_path(execution["frozen_inputs"]["review_protocol"]["path"])
    manifest_path = resolve_repo_path(execution["frozen_inputs"]["packet_manifest"]["path"])
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    if manifest.get("protocol_sha256") != base.sha256(protocol_path):
        raise ActionReviewValidationError("packet manifest does not bind review protocol")
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or base.sha256(artifact_path) != artifact["sha256"]:
            raise ActionReviewValidationError(f"packet artifact hash mismatch: {artifact['path']}")
    qualification = read_json(
        resolve_repo_path(execution["frozen_inputs"]["qualification_registry"]["path"])
    )
    if qualification.get("route") != "P2T3_TWO_ACTIONABILITY_REVIEWERS_QUALIFIED":
        raise ActionReviewValidationError("Actionability reviewers are not qualified")
    if execution.get("optimizer_steps") != 0 or execution.get("training_authorized") is not False:
        raise ActionReviewValidationError("Actionability execution violates zero training")
    return execution, protocol, manifest


def validate_identity(provenances: list[dict[str, Any]], execution: dict[str, Any]) -> None:
    if len(provenances) != 2:
        raise ActionReviewValidationError("exactly two Actionability provenances are required")
    by_role = {row.get("reviewer_role"): row for row in provenances}
    if set(by_role) != {"a", "b"}:
        raise ActionReviewValidationError("Actionability provenance roles are incomplete")
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
            "canonical_line_endings": "LF",
        }
        for field, value in checks.items():
            if provenance.get(field) != value:
                raise ActionReviewValidationError(
                    f"Actionability {role} identity mismatch: {field}"
                )
    try:
        base.ensure_reviewer_isolation(
            provenances, execution["forbidden_prior_agent_ids"]
        )
    except base.ReviewValidationError as exc:
        raise ActionReviewValidationError(str(exc)) from exc
    if len({row["runtime_requested_model_id"] for row in provenances}) != 2:
        raise ActionReviewValidationError("Actionability cross-model diversity is missing")


def review_spec(
    role: str, execution: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    return base.review_spec("actionability", role, execution, protocol)


def preflight(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    for role in ("a", "b"):
        spec = review_spec(role, execution, protocol)
        base.validate_packet_template(spec, manifest)
        if spec["output"].exists() or spec["provenance"].exists():
            raise ActionReviewValidationError(f"Actionability {role} output exists before launch")
    result = {
        "status": "PASS_CSPR_P2T_ACTIONABILITY_REVIEW_PREFLIGHT",
        "reviewer_a_model": execution["qualified_reviewers"]["a"][
            "runtime_requested_model_id"
        ],
        "reviewer_b_model": execution["qualified_reviewers"]["b"][
            "runtime_requested_model_id"
        ],
        "rows_per_reviewer": 312,
        "canonical_line_endings": "LF",
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def freeze_action(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    provenances = []
    artifacts = []
    for role in ("a", "b"):
        spec = review_spec(role, execution, protocol)
        require_canonical_lf(spec["output"])
        require_canonical_lf(spec["provenance"])
        rows, provenance = base.validate_review(spec, manifest)
        if len(rows) != 312:
            raise ActionReviewValidationError(f"Actionability {role} row count mismatch")
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
        "name": "step3_cspr_p2t_actionability_ab_freeze_v1",
        "route": "P2T4_ALL_AB_FROZEN_AUTHORIZE_ANALYSIS_PROTOCOL_ONLY",
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
        "analysis_protocol_authorized_after_push": True,
        "reviewer_c_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(execution["freeze_outputs"]["manifest"]), result)
    report = """# CSPR P2T Actionability A/B Freeze

Route: `P2T4_ALL_AB_FROZEN_AUTHORIZE_ANALYSIS_PROTOCOL_ONLY`.

Two prospectively qualified cross-model reviewers each returned one complete canonical-LF 312-row Actionability file with valid identity, model, qualification, schema, order, and hash provenance. Together with the committed Safety A/B freeze, all four P2T A/B outputs now exist.

A/B labels have not been compared and no hidden key has been opened. This route authorizes only a separately frozen analysis execution protocol. Reviewer C, gradients, and training remain unauthorized. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(execution["freeze_outputs"]["report"]), report)
    print(json.dumps({"route": result["route"], "reviewers": 2}, sort_keys=True))
    return result


def record_inconclusive(
    path: Path, role: str | None, agent_id: str | None, reason: str | None
) -> dict[str, Any]:
    execution, _, _ = context(path)
    if not role or not agent_id or not reason:
        raise ActionReviewValidationError("inconclusive record requires role, agent ID, and reason")
    result = {
        "name": "step3_cspr_p2t_actionability_review_inconclusive_v1",
        "route": "STOP_CSPR_P2T_REVIEW_INCONCLUSIVE",
        "reviewer_role": role,
        "agent_id": agent_id,
        "reason": reason,
        "partial_output_accepted": False,
        "replacement_reviewer_started": False,
        "heuristic_substitution_used": False,
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
        elif args.command == "freeze-action":
            freeze_action(path)
        else:
            record_inconclusive(path, args.reviewer_role, args.agent_id, args.reason)
    except (
        ActionReviewValidationError,
        base.ReviewValidationError,
        FileExistsError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2T_ACTIONABILITY_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
