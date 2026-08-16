"""Run deterministic numerical checks for the CSPR P0 proof obligations."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import step3_cspr_common as cspr


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
    return path


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def main() -> None:
    args = parse_args()
    protocol_path = resolve(args.protocol)
    output_path = resolve(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for key in ("common", "tests", "runner", "proof"):
        path = resolve(protocol["artifacts"][key]["path"])
        if sha256(path) != protocol["artifacts"][key]["sha256"]:
            raise ValueError(f"frozen artifact hash mismatch: {key}")

    import torch

    generator = torch.Generator().manual_seed(int(protocol["numerics"]["seed"]))
    cases = int(protocol["numerics"]["random_projection_cases"])
    dimension = int(protocol["numerics"]["dimension"])
    active = 0
    inactive = 0
    maximum_safety_violation = 0.0
    maximum_preference_violation = 0.0
    maximum_stationarity = 0.0
    maximum_complementarity = 0.0
    for _ in range(cases):
        g_p = torch.randn(dimension, generator=generator, dtype=torch.float64)
        g_s = torch.randn(dimension, generator=generator, dtype=torch.float64)
        diagonal = torch.exp(
            0.5 * torch.randn(dimension, generator=generator, dtype=torch.float64)
        )
        result = cspr.project_preconditioned_preference_direction(g_p, g_s, diagonal)
        active += int(result.active)
        inactive += int(not result.active)
        maximum_safety_violation = max(
            maximum_safety_violation, result.safety_directional_derivative
        )
        maximum_preference_violation = max(
            maximum_preference_violation, result.preference_directional_derivative
        )
        maximum_stationarity = max(maximum_stationarity, result.kkt_stationarity_norm)
        maximum_complementarity = max(
            maximum_complementarity, result.kkt_complementarity_abs
        )

    point = torch.tensor([0.2, -0.3], dtype=torch.float64)
    preference_target = torch.tensor([1.0, 0.5], dtype=torch.float64)
    safety_target = torch.tensor([-0.5, 0.75], dtype=torch.float64)

    def preference(value: Any) -> Any:
        return 0.5 * ((value - preference_target) ** 2).sum()

    def safety(value: Any) -> Any:
        return 0.5 * ((value - safety_target) ** 2).sum()

    certificate = cspr.project_preconditioned_preference_direction(
        point - preference_target,
        point - safety_target,
        torch.tensor([1.25, 0.75], dtype=torch.float64),
    )
    epsilon = float(protocol["numerics"]["finite_difference_epsilon"])
    preference_fd = cspr.finite_difference_directional_derivative(
        preference, point, certificate.direction, epsilon
    )
    safety_fd = cspr.finite_difference_directional_derivative(
        safety, point, certificate.direction, epsilon
    )
    finite_difference_error = max(
        abs(preference_fd - certificate.preference_directional_derivative),
        abs(safety_fd - certificate.safety_directional_derivative),
    )

    beta = 0.3
    gap = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    dpo_loss = cspr.dpo_pair_loss(gap, 0.0, 0.0, 0.0, beta)
    dpo_loss.backward()
    closed_form = -beta / (1.0 + math.exp(beta * float(gap.detach())))
    dpo_derivative_error = abs(float(gap.grad) - closed_form)

    accepted_point = torch.zeros(2, dtype=torch.float64)
    p_target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    s_target = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def accepted_preference(value: Any) -> Any:
        return 0.5 * ((value - p_target) ** 2).sum()

    def accepted_safety(value: Any) -> Any:
        return 0.5 * ((value - s_target) ** 2).sum()

    accepted_projection = cspr.project_preconditioned_preference_direction(
        accepted_point - p_target, accepted_point - s_target, torch.ones(2)
    )
    accepted_line_search = cspr.dual_objective_backtracking(
        accepted_point,
        accepted_projection.direction,
        accepted_preference,
        accepted_safety,
        accepted_projection.preference_directional_derivative,
        accepted_projection.safety_directional_derivative,
    )

    def tangent_preference(value: Any) -> Any:
        return 0.5 * (value[0] - 1.0) ** 2

    def tangent_safety(value: Any) -> Any:
        return value[0] ** 2 + value[1] ** 2

    tangent_line_search = cspr.dual_objective_backtracking(
        torch.zeros(2),
        torch.tensor([1.0, 0.0]),
        tangent_preference,
        tangent_safety,
        -1.0,
        0.0,
        maximum_trials=12,
    )

    tolerance = protocol["numerics"]["tolerances"]
    checks = {
        "both_projection_branches_exercised": active > 0 and inactive > 0,
        "safety_first_order_nonincrease": maximum_safety_violation
        <= float(tolerance["directional_derivative"]),
        "preference_first_order_nonincrease": maximum_preference_violation
        <= float(tolerance["directional_derivative"]),
        "kkt_stationarity": maximum_stationarity <= float(tolerance["kkt"]),
        "kkt_complementarity": maximum_complementarity <= float(tolerance["kkt"]),
        "finite_difference": finite_difference_error
        <= float(tolerance["finite_difference"]),
        "dpo_closed_form_derivative": dpo_derivative_error
        <= float(tolerance["dpo_derivative"]),
        "joint_descent_step_accepted": accepted_line_search.accepted
        and accepted_line_search.final_preference_loss
        < accepted_line_search.initial_preference_loss
        and accepted_line_search.final_safety_loss < accepted_line_search.initial_safety_loss,
        "tangent_second_order_counterexample_rejected": not tangent_line_search.accepted,
    }
    result = {
        "name": "step3_cspr_p0_numerical_verification_v1",
        "experiment_mode": "proof_and_zero_training_validation",
        "git_head": git_head(),
        "protocol": args.protocol,
        "protocol_sha256": sha256(protocol_path),
        "random_projection": {
            "cases": cases,
            "dimension": dimension,
            "active": active,
            "inactive": inactive,
            "maximum_safety_directional_derivative": maximum_safety_violation,
            "maximum_preference_directional_derivative": maximum_preference_violation,
            "maximum_kkt_stationarity_norm": maximum_stationarity,
            "maximum_kkt_complementarity_abs": maximum_complementarity,
        },
        "finite_difference_error": finite_difference_error,
        "dpo_derivative_error": dpo_derivative_error,
        "accepted_line_search": accepted_line_search.scalars(),
        "tangent_counterexample_line_search": tangent_line_search.scalars(),
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": protocol["claim_boundary"],
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": checks}))
    if not result["passed"]:
        raise RuntimeError("CSPR P0 numerical verification failed")


if __name__ == "__main__":
    main()
