"""Pure mathematics for continuation-compatible existential TARC."""

from __future__ import annotations

import math
from typing import Any


def conservative_path_score(token_margins: Any, temperature: float = 1.0) -> Any:
    """Return a smooth lower bound on the weakest token margin of one path."""
    import torch

    if (
        token_margins.ndim != 1
        or token_margins.numel() == 0
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("E-TARC path margins or temperature are invalid")
    return -temperature * torch.logsumexp(-token_margins.float() / temperature, dim=0)


def existential_path_score(path_scores: Any, temperature: float = 1.0) -> Any:
    """Return a smooth lower bound on the strongest complete-path score."""
    import torch

    if (
        path_scores.ndim != 1
        or path_scores.numel() < 2
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("E-TARC requires at least two valid path scores")
    return temperature * (
        torch.logsumexp(path_scores.float() / temperature, dim=0)
        - math.log(int(path_scores.numel()))
    )


def existential_path_certificate(
    path_token_margins: list[Any],
    bottleneck_temperature: float = 1.0,
    path_temperature: float = 1.0,
) -> tuple[Any, Any]:
    """Return the complete-route certificate and conservative path scores."""
    import torch

    if len(path_token_margins) < 2:
        raise ValueError("E-TARC requires at least two registered paths")
    path_scores = torch.stack(
        [
            conservative_path_score(margins, bottleneck_temperature)
            for margins in path_token_margins
        ]
    )
    return existential_path_score(path_scores, path_temperature), path_scores


def path_certificate_barrier(certificate: Any) -> Any:
    import torch.nn.functional as functional

    if certificate.ndim != 0:
        raise ValueError("E-TARC path certificate must be scalar")
    return functional.softplus(-certificate)


def gradient_dot(left: dict[str, Any], right: dict[str, Any]) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("gradient dictionaries differ or are empty")
    return sum(
        float((left[name].double() * right[name].double()).sum())
        for name in sorted(left)
    )


def gradient_norm(gradients: dict[str, Any]) -> float:
    return math.sqrt(max(0.0, gradient_dot(gradients, gradients)))


def continuation_compatible_projection(
    route_gradient: dict[str, Any], continuation_gradient: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project a route gradient onto the continuation non-conflict halfspace."""
    if set(route_gradient) != set(continuation_gradient) or not route_gradient:
        raise ValueError("projection gradients differ or are empty")
    route_norm = gradient_norm(route_gradient)
    continuation_norm = gradient_norm(continuation_gradient)
    if route_norm <= 0 or continuation_norm <= 0:
        raise ValueError("projection gradients must be nonzero")
    cross = gradient_dot(route_gradient, continuation_gradient)
    continuation_norm_sq = continuation_norm * continuation_norm
    coefficient = max(0.0, -cross / continuation_norm_sq)
    projected = {
        name: route_gradient[name] + coefficient * continuation_gradient[name]
        for name in route_gradient
    }
    projected_norm = gradient_norm(projected)
    residual = gradient_dot(projected, continuation_gradient)
    route_retained = gradient_dot(projected, route_gradient)
    return projected, {
        "projection_active": coefficient > 0.0,
        "continuation_addition_coefficient": coefficient,
        "raw_route_continuation_dot": cross,
        "projected_continuation_dot": residual,
        "raw_route_gradient_norm": route_norm,
        "continuation_gradient_norm": continuation_norm,
        "projected_gradient_norm": projected_norm,
        "projected_route_dot": route_retained,
        "projected_route_retention_fraction": route_retained / (route_norm * route_norm),
        "route_direction_cosine": route_retained / (route_norm * projected_norm),
    }


def select_route(gates: dict[str, bool]) -> str:
    required_integrity = (
        "integrity",
        "gradient_valid",
        "budget_matched",
        "vram",
        "state_restored",
    )
    if not all(gates[name] for name in required_integrity):
        return "STOP_ETARC_P0_INVALID"
    if not gates["route_structure"]:
        return "STOP_ETARC_P0_ROUTE_STRUCTURE_FAILED"
    if not gates["objective_distinct"]:
        return "STOP_ETARC_P0_OBJECTIVE_NOT_DISTINCT"
    if not gates["route_transfer"]:
        return "STOP_ETARC_P0_ROUTE_DIRECTION_NOT_REPLICATED"
    if not gates["continuation_compatibility"]:
        return "STOP_ETARC_P0_CONTINUATION_COMPATIBILITY_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_ETARC_P0_ROUTE_EFFECT_TOO_WEAK"
    return "PASS_ETARC_P0_AUTHORIZE_CC_MATCHED_SMOKE_PROTOCOL"
