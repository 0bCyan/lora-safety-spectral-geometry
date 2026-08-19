"""Finite-sample conformal dose control for a fixed optimizer-space direction."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class ConformalDoseError(ValueError):
    """Raised when a conformal safe-dose certificate is undefined."""


def unit_safe_dose(
    fallback_scores: Mapping[str, float],
    endpoint_scores: Mapping[str, float],
    protected_names: Sequence[str],
    allowed_relative_drop: float,
    score_scale_floor: float,
    maximum_dose: float = 1.0,
) -> dict[str, Any]:
    """Return the exact safe dose on one fallback-to-endpoint line."""
    protected = tuple(protected_names)
    if (
        set(fallback_scores) != set(endpoint_scores)
        or not protected
        or not set(protected).issubset(fallback_scores)
        or not math.isfinite(allowed_relative_drop)
        or allowed_relative_drop < 0.0
        or not math.isfinite(score_scale_floor)
        or score_scale_floor <= 0.0
        or not math.isfinite(maximum_dose)
        or maximum_dose <= 0.0
    ):
        raise ConformalDoseError("safe-dose inputs are invalid")
    limits = {}
    standardized_deteriorations = {}
    for name in protected:
        fallback = float(fallback_scores[name])
        endpoint = float(endpoint_scores[name])
        if not math.isfinite(fallback) or not math.isfinite(endpoint):
            raise ConformalDoseError("safe-dose score is not finite")
        scale = max(abs(fallback), score_scale_floor)
        deterioration = (fallback - endpoint) / scale
        standardized_deteriorations[name] = deterioration
        if deterioration <= 0.0:
            limits[name] = maximum_dose
        else:
            limits[name] = min(
                maximum_dose, allowed_relative_drop / deterioration
            )
    dose = min(limits.values())
    return {
        "safe_dose": dose,
        "limiting_certificates": [
            name for name, value in limits.items() if abs(value - dose) <= 1e-15
        ],
        "per_certificate_safe_doses": limits,
        "standardized_endpoint_deteriorations": standardized_deteriorations,
        "allowed_relative_drop": allowed_relative_drop,
        "score_scale_floor": score_scale_floor,
        "maximum_dose": maximum_dose,
    }


def conformal_safe_dose(
    fallback_score_units: Sequence[Mapping[str, float]],
    endpoint_score_units: Sequence[Mapping[str, float]],
    protected_names: Sequence[str],
    allowed_relative_drop: float,
    score_scale_floor: float,
    alpha: float,
    maximum_dose: float = 1.0,
) -> dict[str, Any]:
    """Calibrate a marginally protected dose for a fit-fixed direction.

    If L_i is the maximum safe dose for calibration unit i, exchangeability of
    calibration and future L values implies

        P(L_future >= L_(k)) >= 1 - k/(n+1),

    where k=floor((n+1) alpha). The direction and all non-calibration
    hyperparameters must be fixed before these units are opened.
    """
    if (
        len(fallback_score_units) != len(endpoint_score_units)
        or not fallback_score_units
        or not math.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ConformalDoseError("conformal dose panels are invalid")
    unit_records = [
        unit_safe_dose(
            fallback,
            endpoint,
            protected_names,
            allowed_relative_drop,
            score_scale_floor,
            maximum_dose,
        )
        for fallback, endpoint in zip(
            fallback_score_units, endpoint_score_units, strict=True
        )
    ]
    sample_count = len(unit_records)
    rank = math.floor((sample_count + 1) * alpha)
    if rank < 1:
        raise ConformalDoseError("calibration sample is too small for requested alpha")
    ordered = sorted(float(item["safe_dose"]) for item in unit_records)
    selected = ordered[rank - 1]
    coverage = 1.0 - rank / (sample_count + 1)
    return {
        "calibration_units": sample_count,
        "alpha": alpha,
        "lower_order_statistic_rank": rank,
        "selected_safe_dose": selected,
        "ordered_unit_safe_doses": ordered,
        "unit_records": unit_records,
        "finite_sample_marginal_protection_lower_bound": coverage,
        "finite_sample_marginal_failure_upper_bound": 1.0 - coverage,
        "direction_fixed_before_calibration_required": True,
        "calibration_selects_dose": True,
        "conditional_on_selected_dose_coverage_claimed": False,
        "guarantee_scope": (
            "future_standardized_protected_change_on_the_fixed_direction_under_"
            "exchangeability"
        ),
        "fallback_closed": selected >= 0.0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def interpolate_cosine(
    fallback_cosine: float,
    endpoint_cosine: float,
    fallback_coordinates: Sequence[float],
    endpoint_coordinates: Sequence[float],
    dose: float,
) -> float:
    """Interpolate an exact cosine when endpoint coordinates share an ON basis."""
    fallback = [float(value) for value in fallback_coordinates]
    endpoint = [float(value) for value in endpoint_coordinates]
    if (
        len(fallback) != len(endpoint)
        or not fallback
        or not all(math.isfinite(value) for value in [*fallback, *endpoint])
        or not math.isfinite(fallback_cosine)
        or not math.isfinite(endpoint_cosine)
        or not math.isfinite(dose)
        or not 0.0 <= dose <= 1.0
    ):
        raise ConformalDoseError("cosine interpolation inputs are invalid")
    fallback_norm = math.sqrt(sum(value * value for value in fallback))
    endpoint_norm = math.sqrt(sum(value * value for value in endpoint))
    mixed = [
        (1.0 - dose) * left + dose * right
        for left, right in zip(fallback, endpoint, strict=True)
    ]
    mixed_norm = math.sqrt(sum(value * value for value in mixed))
    if fallback_norm <= 0.0 or endpoint_norm <= 0.0 or mixed_norm <= 0.0:
        raise ConformalDoseError("cosine interpolation has a zero norm")
    normalized_dot = (
        (1.0 - dose) * fallback_cosine * fallback_norm
        + dose * endpoint_cosine * endpoint_norm
    )
    return normalized_dot / mixed_norm
