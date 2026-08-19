"""Pure numerical helpers for factorized optimizer-anchored CPO."""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


class FoacpoMathError(ValueError):
    """Raised when a factorized objective or certificate is undefined."""


def factorized_objective(
    quality_losses: Sequence[float], safety_losses: Sequence[float], alpha: float
) -> float:
    """Return alpha E[Lq] + (1-alpha) E[Ls] from independent samples."""
    if not quality_losses or not safety_losses:
        raise FoacpoMathError("both factorized pools must be nonempty")
    if not 0.0 <= alpha <= 1.0:
        raise FoacpoMathError("alpha must lie in [0, 1]")
    values = [*quality_losses, *safety_losses, alpha]
    if not all(math.isfinite(float(value)) for value in values):
        raise FoacpoMathError("factorized objective inputs must be finite")
    return alpha * statistics.fmean(quality_losses) + (
        1.0 - alpha
    ) * statistics.fmean(safety_losses)


def maximal_protected_blend(
    anchor_scores: Mapping[str, float],
    behavior_scores: Mapping[str, float],
    retention_floor: float,
) -> dict[str, object]:
    """Find the largest anchor-to-behavior interpolation with certified retention.

    Scores are linear descent certificates evaluated on displacements. For
    d(lambda)=(1-lambda)d_anchor+lambda*d_behavior, each score is affine.
    """
    if set(anchor_scores) != set(behavior_scores) or not anchor_scores:
        raise FoacpoMathError("anchor and behavior certificate sets must match")
    if not 0.0 <= retention_floor <= 1.0:
        raise FoacpoMathError("retention floor must lie in [0, 1]")
    if not all(
        math.isfinite(float(value))
        for value in [*anchor_scores.values(), *behavior_scores.values()]
    ):
        raise FoacpoMathError("certificate scores must be finite")
    if not all(float(value) > 0.0 for value in anchor_scores.values()):
        raise FoacpoMathError("anchor certificates must be strictly positive")

    upper = 1.0
    limiting: list[str] = []
    bounds: dict[str, float] = {}
    for name in sorted(anchor_scores):
        anchor = float(anchor_scores[name])
        behavior = float(behavior_scores[name])
        if behavior >= retention_floor * anchor:
            bound = 1.0
        else:
            bound = ((1.0 - retention_floor) * anchor) / (anchor - behavior)
        bound = min(1.0, max(0.0, bound))
        bounds[name] = bound
        if bound < upper - 1e-15:
            upper = bound
            limiting = [name]
        elif abs(bound - upper) <= 1e-15:
            limiting.append(name)

    blended = {
        name: (1.0 - upper) * float(anchor_scores[name])
        + upper * float(behavior_scores[name])
        for name in sorted(anchor_scores)
    }
    retentions = {
        name: blended[name] / float(anchor_scores[name]) for name in blended
    }
    return {
        "lambda": upper,
        "limiting_certificates": limiting,
        "per_certificate_upper_bounds": bounds,
        "blended_scores": blended,
        "retentions": retentions,
        "minimum_retention": min(retentions.values()),
    }
