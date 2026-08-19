"""Replay opened M1 evidence through the conformal-dose development rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3_cdro_opc_dose_common as dose_math


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m2/protocol/cdro_opc_m2_development_protocol.json"
)


class M2DevelopmentError(ValueError):
    """Raised when the retrospective development boundary is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise M2DevelopmentError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(path)
    if protocol.get("status") != "retrospective_development_after_M1_opened":
        raise M2DevelopmentError("M2 development status differs")
    for name, item in protocol["frozen_inputs"].items():
        item_path = resolve_repo_path(item["path"])
        if not item_path.is_file() or sha256(item_path) != item["sha256"]:
            raise M2DevelopmentError(f"frozen input mismatch: {name}")
    if (
        protocol.get("optimizer_steps") != 0
        or protocol.get("training_authorized") is not False
        or protocol.get("fresh_confirmation_claimed") is not False
    ):
        raise M2DevelopmentError("M2 development claim boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise M2DevelopmentError("M2 entry commit is not an ancestor")
    diagnostics = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["m1_diagnostics"]["path"])
    )
    if diagnostics.get("route") != "STOP_CDRO_OPC_M1_TRANSPORT":
        raise M2DevelopmentError("M1 route differs")
    return protocol, diagnostics


def behavior_at_dose(
    behavior: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fallback_coordinates: Sequence[float],
    endpoint_coordinates: Sequence[float],
    selected_dose: float,
) -> dict[str, Any]:
    result = {}
    for split, pools in behavior.items():
        result[split] = {}
        for pool, record in pools.items():
            aggregate = dose_math.interpolate_cosine(
                float(record["fallback_aggregate_cosine"]),
                float(record["candidate_aggregate_cosine"]),
                fallback_coordinates,
                endpoint_coordinates,
                selected_dose,
            )
            values = [
                dose_math.interpolate_cosine(
                    float(fallback),
                    float(endpoint),
                    fallback_coordinates,
                    endpoint_coordinates,
                    selected_dose,
                )
                for fallback, endpoint in zip(
                    record["fallback"]["values"],
                    record["candidate"]["values"],
                    strict=True,
                )
            ]
            result[split][pool] = {
                "aggregate_cosine": aggregate,
                "per_pair_cosines": values,
                "positive": sum(value > 0.0 for value in values),
                "nonpositive": sum(value <= 0.0 for value in values),
                "count": len(values),
            }
    return result


def direct_residuals(
    fallback_units: Sequence[Mapping[str, float]],
    endpoint_units: Sequence[Mapping[str, float]],
    protected_names: Sequence[str],
    selected_dose: float,
    allowed_relative_drop: float,
    score_scale_floor: float,
) -> list[dict[str, float]]:
    rows = []
    for fallback, endpoint in zip(fallback_units, endpoint_units, strict=True):
        rows.append(
            {
                name: (
                    float(fallback[name])
                    - (
                        float(fallback[name])
                        + selected_dose
                        * (float(endpoint[name]) - float(fallback[name]))
                    )
                )
                / max(abs(float(fallback[name])), score_scale_floor)
                - allowed_relative_drop
                for name in protected_names
            }
        )
    return rows


def run(path: Path) -> dict[str, Any]:
    protocol, diagnostics = context(path)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise M2DevelopmentError("M2 development run requires a clean worktree")
    output_root = resolve_repo_path(protocol["output_root"])
    extras = [item for item in output_root.iterdir() if item.name != "protocol"]
    if extras:
        raise M2DevelopmentError("M2 development outputs already exist")

    design = protocol["design"]
    protected = list(design["protected_certificates"])
    fallback_scores = diagnostics["fallback_calibration_scores"]
    endpoint_scores = diagnostics["candidate_calibration_scores"]
    policy = dose_math.conformal_safe_dose(
        fallback_scores,
        endpoint_scores,
        protected,
        float(design["allowed_relative_drop"]),
        float(design["score_scale_floor"]),
        float(design["alpha"]),
        float(design["maximum_dose"]),
    )
    selected = float(policy["selected_safe_dose"])
    fallback_coordinates = diagnostics["fallback_coordinates"]
    endpoint_coordinates = diagnostics["solver_audit"]["candidate_point"]
    behavior = behavior_at_dose(
        diagnostics["behavior"],
        fallback_coordinates,
        endpoint_coordinates,
        selected,
    )
    residuals = direct_residuals(
        fallback_scores,
        endpoint_scores,
        protected,
        selected,
        float(design["allowed_relative_drop"]),
        float(design["score_scale_floor"]),
    )
    maximum_residual = max(value for row in residuals for value in row.values())
    behavior_gate = (
        behavior["calibration"]["quality"]["aggregate_cosine"] > 0.0
        and behavior["calibration"]["safety"]["aggregate_cosine"] > 0.0
        and behavior["mechanism_dev"]["quality"]["aggregate_cosine"] > 0.0
        and behavior["mechanism_dev"]["safety"]["aggregate_cosine"] > 0.0
        and behavior["calibration"]["quality"]["positive"] >= 3
        and behavior["calibration"]["safety"]["positive"] >= 6
        and behavior["mechanism_dev"]["quality"]["positive"] >= 4
        and behavior["mechanism_dev"]["safety"]["positive"] >= 6
    )
    development_ready = (
        selected > 0.0 and maximum_residual <= 1e-12 and behavior_gate
    )
    route = (
        "DEVELOP_CDRO_OPC_M2_DOSE_READY_FOR_FRESH_CONFIRMATION"
        if development_ready
        else "STOP_CDRO_OPC_M2_DOSE_DEVELOPMENT"
    )
    result = {
        "name": "step3_cdro_opc_m2_retrospective_development_v1",
        "route": route,
        "policy": policy,
        "selected_dose": selected,
        "direct_calibration_residuals": residuals,
        "maximum_direct_calibration_residual": maximum_residual,
        "behavior_at_selected_dose": behavior,
        "gates": {
            "positive_dose": selected > 0.0,
            "direct_calibration_protection": maximum_residual <= 1e-12,
            "behavior": behavior_gate,
        },
        "claim_status": "retrospective_development_only",
        "m1_units_may_not_be_reused_for_confirmation": True,
        "fresh_confirmation_claimed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(resolve_repo_path(protocol["outputs"]["development"]), result)
    report = resolve_repo_path(protocol["outputs"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# CDR-OPC M2 Conformal-Dose Development\n\n"
        f"Route: `{route}`. Selected dose: `{selected}`. Maximum direct "
        f"calibration residual: `{maximum_residual}`.\n\n"
        "This replay uses the already-open M1 panel and is development evidence, "
        "not fresh confirmation. It cannot authorize training.\n",
        encoding="utf-8",
    )
    print(json.dumps({"route": route, "selected_dose": selected}, sort_keys=True))
    return result


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _ = context(path)
    result = read_json(resolve_repo_path(protocol["outputs"]["development"]))
    if (
        result.get("claim_status") != "retrospective_development_only"
        or result.get("fresh_confirmation_claimed") is not False
        or result.get("m1_units_may_not_be_reused_for_confirmation") is not True
        or result.get("optimizer_steps") != 0
        or result.get("training_authorized") is not False
        or result["policy"]["direction_fixed_before_calibration_required"] is not True
        or result["policy"]["conditional_on_selected_dose_coverage_claimed"] is not False
    ):
        raise M2DevelopmentError("M2 development result violates its claim boundary")
    verified = {
        "status": "PASS_CDRO_OPC_M2_DEVELOPMENT_VERIFY_RUN",
        "route": result["route"],
        "selected_dose": result["selected_dose"],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(verified, sort_keys=True))
    return verified


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    {"run": run, "verify-run": verify_run}[args.command](path)


if __name__ == "__main__":
    main()
