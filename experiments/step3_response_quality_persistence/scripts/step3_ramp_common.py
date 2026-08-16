"""Pure mathematical helpers for RAMP prefix-margin experiments."""

from __future__ import annotations

import math
from typing import Any


def selected_positions(semantic_tokens: int, width: int, region: str) -> list[int]:
    if width <= 0 or semantic_tokens < width:
        raise ValueError("margin region does not contain the registered width")
    if region == "prefix":
        return list(range(width))
    if region == "suffix":
        return list(range(semantic_tokens - width, semantic_tokens))
    raise ValueError(f"unknown margin region: {region}")


def strongest_competitor_margin(logits: Any, targets: Any) -> tuple[Any, Any]:
    """Return target-minus-best-other logits and the competitor token IDs."""
    import torch

    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("margin logits and targets do not align")
    if logits.shape[1] < 2:
        raise ValueError("margin requires at least two vocabulary entries")
    top_values, top_indices = torch.topk(logits.float(), k=2, dim=-1)
    target_values = logits.float().gather(1, targets[:, None]).squeeze(1)
    target_is_top = top_indices[:, 0] == targets
    competitor_values = torch.where(target_is_top, top_values[:, 1], top_values[:, 0])
    competitor_ids = torch.where(target_is_top, top_indices[:, 1], top_indices[:, 0])
    return target_values - competitor_values, competitor_ids


def margin_barrier(margins: Any) -> Any:
    import torch.nn.functional as functional

    if margins.ndim != 1 or margins.numel() == 0:
        raise ValueError("margin barrier requires a non-empty vector")
    return functional.softplus(-margins).mean()


def gradient_dot(left: dict[str, Any], right: dict[str, Any]) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("gradient dictionaries differ or are empty")
    return sum(
        float((left[name].double() * right[name].double()).sum()) for name in sorted(left)
    )


def gradient_norm(gradients: dict[str, Any]) -> float:
    value = gradient_dot(gradients, gradients)
    return math.sqrt(max(0.0, value))


def gradient_cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
    denominator = gradient_norm(left) * gradient_norm(right)
    return gradient_dot(left, right) / denominator if denominator else math.nan


def scaled_gradient(gradients: dict[str, Any], scale: float) -> dict[str, Any]:
    if not math.isfinite(scale):
        raise ValueError("gradient scale must be finite")
    return {name: value * scale for name, value in gradients.items()}


def add_gradients(*gradients: dict[str, Any]) -> dict[str, Any]:
    if not gradients or any(set(item) != set(gradients[0]) for item in gradients[1:]):
        raise ValueError("gradient dictionaries differ or are empty")
    return {
        name: sum((item[name] for item in gradients[1:]), gradients[0][name].clone())
        for name in gradients[0]
    }


def calibrated_coefficient(uniform_norm: float, auxiliary_norm: float, ratio: float) -> float:
    if not all(math.isfinite(value) for value in (uniform_norm, auxiliary_norm, ratio)):
        raise ValueError("gradient calibration values must be finite")
    if uniform_norm <= 0 or auxiliary_norm <= 0 or ratio <= 0:
        raise ValueError("gradient calibration values must be positive")
    return ratio * uniform_norm / auxiliary_norm


def directional_metrics(
    calibration_uniform: dict[str, Any],
    calibration_prefix: dict[str, Any],
    calibration_suffix: dict[str, Any],
    validation_prefix: dict[str, Any],
    prefix_coefficient: float,
    suffix_coefficient: float,
) -> dict[str, float]:
    prefix_scaled = scaled_gradient(calibration_prefix, prefix_coefficient)
    suffix_scaled = scaled_gradient(calibration_suffix, suffix_coefficient)
    ramp = add_gradients(calibration_uniform, prefix_scaled)
    placebo = add_gradients(calibration_uniform, suffix_scaled)
    prefix_cosine = gradient_cosine(calibration_prefix, validation_prefix)
    suffix_cosine = gradient_cosine(calibration_suffix, validation_prefix)
    return {
        "calibration_prefix_suffix_cosine": gradient_cosine(
            calibration_prefix, calibration_suffix
        ),
        "calibration_validation_prefix_cosine": prefix_cosine,
        "suffix_calibration_to_prefix_validation_cosine": suffix_cosine,
        "prefix_over_suffix_directional_cosine_margin": prefix_cosine - suffix_cosine,
        "uniform_validation_prefix_dot": gradient_dot(
            validation_prefix, calibration_uniform
        ),
        "ramp_validation_prefix_dot": gradient_dot(validation_prefix, ramp),
        "placebo_validation_prefix_dot": gradient_dot(validation_prefix, placebo),
        "ramp_increment_over_uniform_dot": gradient_dot(
            validation_prefix, prefix_scaled
        ),
        "placebo_increment_over_uniform_dot": gradient_dot(
            validation_prefix, suffix_scaled
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    invalid = (
        "integrity",
        "gradient_valid",
        "budget_matched",
        "vram",
        "state_restored",
    )
    if not all(gates[name] for name in invalid):
        return "STOP_RAMP_P0_INVALID"
    if not gates["prefix_behavioral"]:
        return "STOP_RAMP_P0_PREFIX_NOT_BEHAVIORAL"
    if not gates["placebo_distinct"]:
        return "STOP_RAMP_P0_PLACEBO_NOT_DISTINCT"
    if not gates["cross_prompt_directional"]:
        return "STOP_RAMP_P0_DIRECTION_NOT_REPLICATED"
    return "PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL"
