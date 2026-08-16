"""Numerical primitives for constrained same-prompt preference replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ProjectionCertificate:
    direction: Any
    unconstrained_direction: Any
    active: bool
    multiplier: float
    preference_directional_derivative: float
    safety_directional_derivative: float
    denominator: float
    kkt_stationarity_norm: float
    kkt_complementarity_abs: float

    def scalars(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("direction")
        value.pop("unconstrained_direction")
        return value


@dataclass(frozen=True)
class BacktrackingCertificate:
    point: Any
    accepted: bool
    step_size: float
    trials: int
    initial_preference_loss: float
    initial_safety_loss: float
    final_preference_loss: float
    final_safety_loss: float
    preference_bound: float
    safety_bound: float

    def scalars(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("point")
        return value


def _torch() -> Any:
    import torch

    return torch


def _flat_float64(value: Any) -> Any:
    torch = _torch()
    result = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("input contains a non-finite value")
    return result


def project_preconditioned_preference_direction(
    preference_gradient: Any,
    safety_gradient: Any,
    diagonal_preconditioner: Any,
    activation_tolerance: float = 0.0,
) -> ProjectionCertificate:
    """Project -P g_p onto {d: g_s^T d <= 0} in the P^-1 metric."""

    torch = _torch()
    g_p = _flat_float64(preference_gradient)
    g_s = _flat_float64(safety_gradient)
    diagonal = _flat_float64(diagonal_preconditioner)
    if g_p.shape != g_s.shape or g_p.shape != diagonal.shape:
        raise ValueError("gradients and diagonal preconditioner must have the same shape")
    if not bool((diagonal > 0).all()):
        raise ValueError("diagonal preconditioner must be strictly positive")
    if activation_tolerance < 0:
        raise ValueError("activation tolerance must be non-negative")

    d0 = -diagonal * g_p
    violation = torch.dot(g_s, d0)
    denominator = torch.dot(g_s, diagonal * g_s)
    active = bool(violation > activation_tolerance)
    multiplier = 0.0
    if active:
        if not bool(denominator > 0):
            raise ValueError("active projection has a zero denominator")
        multiplier = float(violation / denominator)
        direction = d0 - multiplier * diagonal * g_s
    else:
        direction = d0.clone()

    stationarity = (direction - d0) / diagonal + multiplier * g_s
    safety_derivative = torch.dot(g_s, direction)
    preference_derivative = torch.dot(g_p, direction)
    complementarity = multiplier * float(safety_derivative)
    values = torch.stack(
        [
            safety_derivative,
            preference_derivative,
            denominator,
            torch.linalg.vector_norm(stationarity),
        ]
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("projection certificate contains a non-finite value")
    return ProjectionCertificate(
        direction=direction,
        unconstrained_direction=d0,
        active=active,
        multiplier=multiplier,
        preference_directional_derivative=float(preference_derivative),
        safety_directional_derivative=float(safety_derivative),
        denominator=float(denominator),
        kkt_stationarity_norm=float(torch.linalg.vector_norm(stationarity)),
        kkt_complementarity_abs=abs(complementarity),
    )


def dpo_pair_loss(
    chosen_log_probability: Any,
    rejected_log_probability: Any,
    reference_chosen_log_probability: Any,
    reference_rejected_log_probability: Any,
    beta: float,
) -> Any:
    torch = _torch()
    if beta <= 0:
        raise ValueError("beta must be positive")
    gap = (chosen_log_probability - reference_chosen_log_probability) - (
        rejected_log_probability - reference_rejected_log_probability
    )
    return torch.nn.functional.softplus(-beta * gap)


def finite_difference_directional_derivative(
    function: Callable[[Any], Any], point: Any, direction: Any, epsilon: float
) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    x = _flat_float64(point)
    d = _flat_float64(direction)
    if x.shape != d.shape:
        raise ValueError("point and direction must have the same shape")
    positive = float(function(x + epsilon * d))
    negative = float(function(x - epsilon * d))
    return (positive - negative) / (2.0 * epsilon)


def dual_objective_backtracking(
    point: Any,
    direction: Any,
    preference_loss: Callable[[Any], Any],
    safety_loss: Callable[[Any], Any],
    preference_directional_derivative: float,
    safety_directional_derivative: float,
    initial_step_size: float = 1.0,
    contraction: float = 0.5,
    armijo_fraction: float = 0.1,
    maximum_trials: int = 20,
    comparison_tolerance: float = 0.0,
) -> BacktrackingCertificate:
    x = _flat_float64(point)
    d = _flat_float64(direction)
    if x.shape != d.shape:
        raise ValueError("point and direction must have the same shape")
    if preference_directional_derivative >= 0:
        raise ValueError("preference direction must be strict descent")
    if safety_directional_derivative > comparison_tolerance:
        raise ValueError("safety direction violates the first-order half-space")
    if initial_step_size <= 0 or not 0 < contraction < 1:
        raise ValueError("invalid backtracking schedule")
    if not 0 < armijo_fraction < 1 or maximum_trials <= 0:
        raise ValueError("invalid Armijo configuration")
    if comparison_tolerance < 0:
        raise ValueError("comparison tolerance must be non-negative")

    initial_preference = float(preference_loss(x))
    initial_safety = float(safety_loss(x))
    step_size = initial_step_size
    last_preference = initial_preference
    last_safety = initial_safety
    preference_bound = initial_preference
    safety_bound = initial_safety
    for trial in range(1, maximum_trials + 1):
        candidate = x + step_size * d
        last_preference = float(preference_loss(candidate))
        last_safety = float(safety_loss(candidate))
        preference_bound = (
            initial_preference
            + armijo_fraction * step_size * preference_directional_derivative
        )
        safety_bound = initial_safety + armijo_fraction * step_size * min(
            safety_directional_derivative, 0.0
        )
        if (
            last_preference <= preference_bound + comparison_tolerance
            and last_safety <= safety_bound + comparison_tolerance
        ):
            return BacktrackingCertificate(
                point=candidate,
                accepted=True,
                step_size=step_size,
                trials=trial,
                initial_preference_loss=initial_preference,
                initial_safety_loss=initial_safety,
                final_preference_loss=last_preference,
                final_safety_loss=last_safety,
                preference_bound=preference_bound,
                safety_bound=safety_bound,
            )
        step_size *= contraction
    return BacktrackingCertificate(
        point=x.clone(),
        accepted=False,
        step_size=0.0,
        trials=maximum_trials,
        initial_preference_loss=initial_preference,
        initial_safety_loss=initial_safety,
        final_preference_loss=initial_preference,
        final_safety_loss=initial_safety,
        preference_bound=preference_bound,
        safety_bound=safety_bound,
    )
