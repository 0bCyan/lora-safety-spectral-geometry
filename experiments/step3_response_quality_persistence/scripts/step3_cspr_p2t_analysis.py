"""Compute frozen P2T A/B reliability and target-disagreement C packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import step3_cspr_p2s_review as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTION_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2t/"
    "protocol/cspr_p2t_ab_analysis_execution_protocol.json"
)


class AnalysisValidationError(ValueError):
    """Raised when a frozen P2T analysis input is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "analyze-ab"])
    parser.add_argument("--execution-protocol", default=DEFAULT_EXECUTION_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise AnalysisValidationError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        execution, protocol, manifest = base.context(path)
    except base.ReviewValidationError as exc:
        raise AnalysisValidationError(str(exc)) from exc
    safety_freeze = read_json(
        resolve_repo_path(execution["frozen_inputs"]["safety_freeze"]["path"])
    )
    action_freeze = read_json(
        resolve_repo_path(execution["frozen_inputs"]["actionability_freeze"]["path"])
    )
    if safety_freeze.get("route") != (
        "P2T2_SAFETY_AB_FROZEN_AUTHORIZE_ACTIONABILITY_QUALIFICATION"
    ):
        raise AnalysisValidationError("Safety A/B freeze route mismatch")
    if action_freeze.get("route") != "P2T4_ALL_AB_FROZEN_AUTHORIZE_ANALYSIS_PROTOCOL_ONLY":
        raise AnalysisValidationError("Actionability A/B freeze route mismatch")
    if safety_freeze.get("labels_compared") is not False or action_freeze.get(
        "labels_compared"
    ) is not False:
        raise AnalysisValidationError("A/B labels were compared before analysis freeze")
    return execution, protocol, manifest


def select_route(calibration_pass: bool, target_pass: bool, duplicate_pass: bool) -> str:
    if not calibration_pass:
        return "STOP_CSPR_P2T_CALIBRATION_UNRELIABLE"
    if not duplicate_pass:
        return "STOP_CSPR_P2T_DUPLICATE_UNRELIABLE"
    if not target_pass:
        return "STOP_CSPR_P2T_TARGET_UNRELIABLE"
    return "AUTHORIZE_CSPR_P2T_REVIEWER_C_PROTOCOL_ONLY"


def preflight(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    reviews, provenances = base.load_ab(execution, protocol, manifest)
    if set(reviews) != {"safety", "actionability"} or len(provenances) != 4:
        raise AnalysisValidationError("four complete P2T A/B reviews are required")
    result = {
        "status": "PASS_CSPR_P2T_AB_ANALYSIS_PREFLIGHT",
        "complete_reviewers": 4,
        "safety_rows_per_reviewer": 424,
        "actionability_rows_per_reviewer": 312,
        "labels_compared": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def analyze_ab(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    reviews, provenances = base.load_ab(execution, protocol, manifest)
    safety_key = base.read_jsonl(
        resolve_repo_path(protocol["packet_outputs"]["safety_key"])
    )
    action_key = base.read_jsonl(
        resolve_repo_path(protocol["packet_outputs"]["action_key"])
    )
    safety = base.safety_analysis(
        reviews["safety"]["a"], reviews["safety"]["b"], safety_key, protocol
    )
    action = base.action_analysis(
        reviews["actionability"]["a"],
        reviews["actionability"]["b"],
        action_key,
        protocol,
    )
    calibration_pass = safety["calibration_pass"] and action["calibration_pass"]
    target_pass = safety["target_pass"] and action["target_pass"]
    duplicate_pass = safety["duplicate_pass"] and action["duplicate_pass"]
    route = select_route(calibration_pass, target_pass, duplicate_pass)
    overall = calibration_pass and target_pass and duplicate_pass

    c_artifacts = []
    if overall:
        for construct, block in (("safety", safety), ("actionability", action)):
            packet, template = base.build_c_packet(
                construct, block["target_disagreement_ids"], execution, protocol
            )
            files = execution["review_files"][construct]
            packet_path = resolve_repo_path(files["reviewer_c_packet"])
            template_path = resolve_repo_path(files["reviewer_c_template"])
            base.write_jsonl_frozen(packet_path, packet)
            base.write_jsonl_frozen(template_path, template)
            c_artifacts.extend(
                [
                    {
                        "path": base.relative(packet_path),
                        "rows": len(packet),
                        "sha256": base.sha256(packet_path),
                    },
                    {
                        "path": base.relative(template_path),
                        "rows": len(template),
                        "sha256": base.sha256(template_path),
                    },
                ]
            )

    report = {
        "name": "step3_cspr_p2t_ab_analysis_v1",
        "route": route,
        "overall_pass": overall,
        "calibration_pass": calibration_pass,
        "target_pass": target_pass,
        "duplicate_pass": duplicate_pass,
        "safety": safety,
        "actionability": action,
        "reviewer_provenance": [
            {
                field: row[field]
                for field in (
                    "construct",
                    "reviewer_role",
                    "agent_id",
                    "session_id",
                    "output_sha256",
                )
            }
            for row in provenances
        ],
        "reviewer_c_artifacts": c_artifacts,
        "target_prevalence_used_as_reviewer_gate": False,
        "pair_qualification_run": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    base.write_json_frozen(
        resolve_repo_path(execution["analysis_outputs"]["ab_analysis"]), report
    )
    markdown = f"""# CSPR P2T A/B Review Analysis

Route: `{route}`.

Calibration gates passed: `{calibration_pass}`. Target reliability gates passed: `{target_pass}`. Hidden duplicate gates passed: `{duplicate_pass}`. Safety has `{safety['target_disagreement_rows']}` target disagreements and Actionability has `{action['target_disagreement_rows']}`. Target score prevalence was reported but not used as a reviewer gate.

Reviewer C is authorized only through a separate prospective qualification and execution protocol when every gate above passes. Pair qualification, gradients, and training were not run. `optimizer_steps=0` and `training_authorized=false`.
"""
    base.write_frozen(
        resolve_repo_path(execution["analysis_outputs"]["ab_report"]), markdown
    )
    print(
        json.dumps(
            {
                "route": route,
                "safety_disagreements": safety["target_disagreement_rows"],
                "actionability_disagreements": action["target_disagreement_rows"],
            },
            sort_keys=True,
        )
    )
    return report


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.execution_protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        else:
            analyze_ab(path)
    except (
        AnalysisValidationError,
        base.ReviewValidationError,
        FileExistsError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2T_ANALYSIS_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
