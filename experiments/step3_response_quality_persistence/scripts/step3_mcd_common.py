"""Minimax multi-certificate common-descent geometry."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math


def minimum_norm_simplex_weights(
    gram: list[list[float]], tolerance: float = 1e-9
) -> tuple[list[float], dict[str, Any]]:
    """Solve min lambda^T K lambda on the simplex by support enumeration."""
    matrix = np.asarray(gram, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.isfinite(matrix).all()
        or tolerance <= 0.0
    ):
        raise ValueError("simplex Gram matrix is invalid")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-8):
        raise ValueError("simplex Gram matrix is not symmetric")
    size = matrix.shape[0]
    best: tuple[float, np.ndarray, tuple[int, ...], float] | None = None
    candidates = 0
    for support_size in range(1, size + 1):
        for support in itertools.combinations(range(size), support_size):
            indices = np.asarray(support, dtype=np.int64)
            block = matrix[np.ix_(indices, indices)]
            ones = np.ones(support_size, dtype=np.float64)
            kkt = np.block(
                [
                    [2.0 * block, -ones[:, None]],
                    [ones[None, :], np.zeros((1, 1), dtype=np.float64)],
                ]
            )
            right = np.concatenate(
                [np.zeros(support_size, dtype=np.float64), np.ones(1)]
            )
            solution, _, _, _ = np.linalg.lstsq(kkt, right, rcond=1e-12)
            if not np.isfinite(solution).all() or np.max(
                np.abs(kkt @ solution - right)
            ) > max(1e-7, 100.0 * tolerance):
                continue
            active = solution[:support_size]
            if np.min(active) < -tolerance:
                continue
            active = np.maximum(active, 0.0)
            active /= active.sum()
            weights = np.zeros(size, dtype=np.float64)
            weights[indices] = active
            product = matrix @ weights
            objective = float(weights @ product)
            stationarity = max(
                [abs(float(product[index]) - objective) for index in support]
                + [0.0]
            )
            inactive = [index for index in range(size) if index not in support]
            inequality = max(
                [max(0.0, objective - float(product[index])) for index in inactive]
                + [0.0]
            )
            residual = max(stationarity, inequality, abs(float(weights.sum()) - 1.0))
            if residual > max(1e-7, 100.0 * tolerance):
                continue
            candidates += 1
            if best is None or objective < best[0] - tolerance:
                best = objective, weights, support, residual
    if best is None:
        raise ValueError("simplex active-set enumeration found no KKT point")
    objective, weights, support, residual = best
    return weights.tolist(), {
        "dual_squared_margin": max(0.0, objective),
        "active_indices": list(support),
        "active_set_size": len(support),
        "kkt_residual": residual,
        "accepted_kkt_candidates": candidates,
    }


def minimax_common_direction(
    gradients: dict[str, dict[str, Any]],
    budget_norm: float,
    tolerance: float = 1e-9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the fixed-budget direction maximizing worst normalized descent."""
    if len(gradients) < 2 or not math.isfinite(budget_norm) or budget_norm <= 0.0:
        raise ValueError("common-descent inputs or budget are invalid")
    names = list(gradients)
    tensors = [gradients[name] for name in names]
    if any(set(item) != set(tensors[0]) for item in tensors[1:]):
        raise ValueError("common-descent gradient dictionaries differ")
    norms = [gradient_math.gradient_norm(item) for item in tensors]
    if any(not math.isfinite(value) or value <= 0.0 for value in norms):
        raise ValueError("common-descent gradients must be finite and nonzero")
    units = [
        gradient_ops.scaled_gradient(item, 1.0 / norm)
        for item, norm in zip(tensors, norms, strict=True)
    ]
    gram = [
        [gradient_math.gradient_dot(left, right) for right in units]
        for left in units
    ]
    weights, dual = minimum_norm_simplex_weights(gram, tolerance)
    weighted = gradient_ops.scaled_gradient(units[0], weights[0])
    for weight, unit in zip(weights[1:], units[1:], strict=True):
        weighted = gradient_ops.add_gradients(weighted, unit, right_scale=weight)
    dual_margin = gradient_math.gradient_norm(weighted)
    if dual_margin <= tolerance:
        raise ValueError("registered certificates have no strict common descent")
    unit_direction = gradient_ops.scaled_gradient(weighted, 1.0 / dual_margin)
    direction = gradient_ops.scaled_gradient(unit_direction, budget_norm)
    cosines = {
        name: gradient_math.gradient_cosine(gradient, direction)
        for name, gradient in gradients.items()
    }
    primal_margin = min(cosines.values())
    return direction, {
        "certificate_names": names,
        "gradient_norms": dict(zip(names, norms, strict=True)),
        "gram_matrix": gram,
        "dual_weights": dict(zip(names, weights, strict=True)),
        "active_certificates": [
            name for name, weight in zip(names, weights, strict=True)
            if weight > tolerance
        ],
        "dual_margin": dual_margin,
        "primal_margin": primal_margin,
        "primal_dual_gap": abs(primal_margin - dual_margin),
        "certificate_cosines": cosines,
        "budget_norm": budget_norm,
        "final_direction_norm": gradient_math.gradient_norm(direction),
        **dual,
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
        return "STOP_MCD_ETARC_P0_INVALID"
    if not gates["dual_certificate"]:
        return "STOP_MCD_ETARC_P0_DUAL_CERTIFICATE_FAILED"
    if not gates["common_transfer"]:
        return "STOP_MCD_ETARC_P0_COMMON_DESCENT_NOT_REPLICATED"
    if not gates["route_effect"]:
        return "STOP_MCD_ETARC_P0_ROUTE_EFFECT_FAILED"
    if not gates["continuation_effect"]:
        return "STOP_MCD_ETARC_P0_CONTINUATION_EFFECT_FAILED"
    if not gates["branch_effect"]:
        return "STOP_MCD_ETARC_P0_BRANCH_EFFECT_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_MCD_ETARC_P0_EFFECT_TOO_WEAK"
    return "PASS_MCD_ETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL"
