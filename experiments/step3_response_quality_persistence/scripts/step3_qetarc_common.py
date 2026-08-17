"""Fixed-budget spherical-cap geometry for quota-constrained E-TARC."""

from __future__ import annotations

import math
from typing import Any

import step3_etarc_common as etarc_geometry


def scaled_gradient(gradient: dict[str, Any], scale: float) -> dict[str, Any]:
    if not math.isfinite(scale):
        raise ValueError("gradient scale must be finite")
    return {name: value * scale for name, value in gradient.items()}


def add_gradients(
    left: dict[str, Any], right: dict[str, Any], right_scale: float = 1.0
) -> dict[str, Any]:
    if set(left) != set(right) or not left or not math.isfinite(right_scale):
        raise ValueError("gradient dictionaries differ or scale is invalid")
    return {
        name: left[name] + right_scale * right[name]
        for name in left
    }


def quota_constrained_direction(
    route_gradient: dict[str, Any],
    continuation_gradient: dict[str, Any],
    quota_cosine: float,
    budget_norm: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the fixed-norm route-optimal direction in a continuation cap."""
    if set(route_gradient) != set(continuation_gradient) or not route_gradient:
        raise ValueError("quota gradients differ or are empty")
    if (
        not math.isfinite(quota_cosine)
        or not 0.0 < quota_cosine < 1.0
        or not math.isfinite(budget_norm)
        or budget_norm <= 0.0
    ):
        raise ValueError("quota cosine and budget must be finite and positive")
    route_norm = etarc_geometry.gradient_norm(route_gradient)
    continuation_norm = etarc_geometry.gradient_norm(continuation_gradient)
    if route_norm <= 0.0 or continuation_norm <= 0.0:
        raise ValueError("quota gradients must be nonzero")
    route_unit = scaled_gradient(route_gradient, 1.0 / route_norm)
    continuation_unit = scaled_gradient(
        continuation_gradient, 1.0 / continuation_norm
    )
    raw_cosine = etarc_geometry.gradient_dot(route_unit, continuation_unit)
    raw_cosine = min(1.0, max(-1.0, raw_cosine))
    active = raw_cosine < quota_cosine
    if active:
        orthogonal_norm = math.sqrt(max(0.0, 1.0 - raw_cosine * raw_cosine))
        if orthogonal_norm <= 1e-12:
            raise ValueError("anti-collinear gradients do not define a unique cap optimum")
        route_orthogonal = add_gradients(
            route_unit,
            continuation_unit,
            right_scale=-raw_cosine,
        )
        route_orthogonal = scaled_gradient(route_orthogonal, 1.0 / orthogonal_norm)
        unit_direction = add_gradients(
            scaled_gradient(continuation_unit, quota_cosine),
            route_orthogonal,
            right_scale=math.sqrt(1.0 - quota_cosine * quota_cosine),
        )
    else:
        unit_direction = route_unit
    pre_normalization_norm = etarc_geometry.gradient_norm(unit_direction)
    if pre_normalization_norm <= 0.0:
        raise ValueError("quota direction is zero")
    direction = scaled_gradient(
        unit_direction, budget_norm / pre_normalization_norm
    )
    final_norm = etarc_geometry.gradient_norm(direction)
    continuation_cosine = (
        etarc_geometry.gradient_dot(direction, continuation_gradient)
        / (final_norm * continuation_norm)
    )
    route_cosine = (
        etarc_geometry.gradient_dot(direction, route_gradient)
        / (final_norm * route_norm)
    )
    theoretical_route_cosine = (
        raw_cosine
        if not active
        else raw_cosine * quota_cosine
        + math.sqrt(1.0 - raw_cosine * raw_cosine)
        * math.sqrt(1.0 - quota_cosine * quota_cosine)
    )
    return direction, {
        "quota_active": active,
        "quota_cosine": quota_cosine,
        "budget_norm": budget_norm,
        "raw_route_continuation_cosine": raw_cosine,
        "achieved_continuation_cosine": continuation_cosine,
        "achieved_route_cosine": route_cosine,
        "theoretical_route_cosine": theoretical_route_cosine,
        "quota_slack": continuation_cosine - quota_cosine,
        "pre_normalization_norm": pre_normalization_norm,
        "final_direction_norm": final_norm,
        "continuation_first_order_dot": etarc_geometry.gradient_dot(
            direction, continuation_gradient
        ),
        "continuation_dot_lower_bound": budget_norm
        * continuation_norm
        * quota_cosine,
        "route_first_order_dot": etarc_geometry.gradient_dot(
            direction, route_gradient
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    integrity = (
        "integrity",
        "gradient_valid",
        "budget_matched",
        "vram",
        "state_restored",
    )
    if not all(gates[name] for name in integrity):
        return "STOP_QETARC_P0_INVALID"
    if not gates["certificate_structure"]:
        return "STOP_QETARC_P0_CERTIFICATE_STRUCTURE_FAILED"
    if not gates["quota_geometry"]:
        return "STOP_QETARC_P0_QUOTA_GEOMETRY_FAILED"
    if not gates["route_transfer"]:
        return "STOP_QETARC_P0_ROUTE_DIRECTION_NOT_REPLICATED"
    if not gates["continuation_transfer"]:
        return "STOP_QETARC_P0_CONTINUATION_QUOTA_NOT_REPLICATED"
    if not gates["branch_mechanism"]:
        return "STOP_QETARC_P0_BRANCH_MECHANISM_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_QETARC_P0_EFFECT_TOO_WEAK"
    return "PASS_QETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL"
