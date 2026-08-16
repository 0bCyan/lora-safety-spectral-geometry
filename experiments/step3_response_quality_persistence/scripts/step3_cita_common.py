"""Deterministic helpers for Counterfactual Information Token Alignment."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Iterable


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def linear_rank_weights(
    scores: list[float], minimum_weight: float = 0.25, maximum_weight: float = 1.75
) -> list[float]:
    """Assign a fixed mean-one weight multiset by stable score/position rank."""
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError("rank weights require finite non-empty scores")
    if minimum_weight <= 0 or maximum_weight <= minimum_weight:
        raise ValueError("invalid rank-weight bounds")
    if not math.isclose((minimum_weight + maximum_weight) / 2.0, 1.0):
        raise ValueError("rank-weight bounds must have mean one")
    if len(scores) == 1:
        return [1.0]
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    weights = [0.0] * len(scores)
    width = maximum_weight - minimum_weight
    for rank, index in enumerate(order):
        weights[index] = minimum_weight + width * rank / (len(scores) - 1)
    if not math.isclose(sum(weights) / len(weights), 1.0, abs_tol=1e-12):
        raise AssertionError("rank-weight mean drift")
    return weights


def token_weight_rows(
    correct_nll: list[float],
    wrong_nlls: list[list[float]],
    semantic_mask: list[bool],
    minimum_weight: float = 0.25,
    maximum_weight: float = 1.75,
) -> dict[str, Any]:
    """Build CITA and exact-budget sham weights for one free-response suffix."""
    if len(wrong_nlls) != 2:
        raise ValueError("CITA requires exactly two wrong prompts")
    size = len(correct_nll)
    if size == 0 or any(len(values) != size for values in wrong_nlls):
        raise ValueError("token NLL vectors must have equal positive length")
    if len(semantic_mask) != size or sum(semantic_mask) < 2:
        raise ValueError("CITA requires at least two semantic response tokens")
    if not all(
        math.isfinite(value)
        for values in [correct_nll, *wrong_nlls]
        for value in values
    ):
        raise ValueError("token NLL vectors must be finite")
    semantic_indices = [index for index, keep in enumerate(semantic_mask) if keep]
    advantages = [
        (wrong_nlls[0][index] + wrong_nlls[1][index]) / 2.0 - correct_nll[index]
        for index in semantic_indices
    ]
    sham_scores = [
        wrong_nlls[1][index] - wrong_nlls[0][index] for index in semantic_indices
    ]
    ranked = linear_rank_weights(advantages, minimum_weight, maximum_weight)
    sham_ranked = linear_rank_weights(sham_scores, minimum_weight, maximum_weight)
    cita = [1.0] * size
    sham = [1.0] * size
    for index, weight, sham_weight in zip(
        semantic_indices, ranked, sham_ranked, strict=True
    ):
        cita[index] = weight
        sham[index] = sham_weight
    if sorted(ranked) != sorted(sham_ranked):
        raise AssertionError("CITA and sham weight multisets differ")
    return {
        "advantages": advantages,
        "sham_scores": sham_scores,
        "semantic_indices": semantic_indices,
        "uniform_weights": [1.0] * size,
        "cita_weights": cita,
        "sham_weights": sham,
        "semantic_cita_weights": ranked,
        "semantic_sham_weights": sham_ranked,
    }


def semantic_response_mask(
    response_token_ids: list[int], special_token_ids: Iterable[int]
) -> list[bool]:
    """Keep response content before the first serialized termination token."""
    if not response_token_ids:
        raise ValueError("response suffix is empty")
    special = {int(token_id) for token_id in special_token_ids}
    termination = next(
        (
            index
            for index, token_id in enumerate(response_token_ids)
            if int(token_id) in special
        ),
        None,
    )
    if termination is None:
        raise ValueError("response suffix has no special termination token")
    mask = [index < termination for index in range(len(response_token_ids))]
    if sum(mask) < 2:
        raise ValueError("response suffix has fewer than two semantic tokens")
    return mask


def pearson_ranks(left: list[float], right: list[float]) -> float:
    """Spearman-style correlation with stable positional tie breaking."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation requires equal vectors of length at least two")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        output = [0.0] * len(values)
        for rank, index in enumerate(order):
            output[index] = float(rank)
        return output

    x = ranks(left)
    y = ranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_scale = math.sqrt(sum((value - x_mean) ** 2 for value in x))
    y_scale = math.sqrt(sum((value - y_mean) ** 2 for value in y))
    return numerator / (x_scale * y_scale)


def bootstrap_mean(
    values: list[float], samples: int, seed: int
) -> dict[str, float | int]:
    if not values or samples < 100:
        raise ValueError("bootstrap requires values and at least 100 draws")
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return {
        "rows": len(values),
        "mean": sum(values) / len(values),
        "ci95_lower": draws[int(0.025 * samples)],
        "ci95_upper": draws[min(samples - 1, int(0.975 * samples))],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid quantile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def select_prompt_disjoint_rows(
    rows: list[dict[str, Any]],
    calibration_prompts: int,
    validation_prompts: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Select one target per prompt, then form category-balanced disjoint splits."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    representatives = []
    for sample_id, candidates in grouped.items():
        representatives.append(
            min(
                candidates,
                key=lambda row: stable_hash(
                    str(seed), "target", sample_id, str(row["target_id"])
                ),
            )
        )
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        by_category[str(row["category"])].append(row)
    for category in by_category:
        by_category[category].sort(
            key=lambda row: stable_hash(
                str(seed), "prompt", category, str(row["sample_id"])
            )
        )
    ordered = []
    categories = sorted(
        by_category,
        key=lambda value: stable_hash(str(seed), "category", value),
    )
    position = 0
    while any(by_category.values()):
        category = categories[position % len(categories)]
        if by_category[category]:
            ordered.append(by_category[category].pop(0))
        position += 1
    required = calibration_prompts + validation_prompts
    if len(ordered) < required:
        raise ValueError(f"only {len(ordered)} prompt clusters for {required} required")
    calibration = ordered[:calibration_prompts]
    validation = ordered[calibration_prompts:required]
    if {row["sample_id"] for row in calibration}.intersection(
        {row["sample_id"] for row in validation}
    ):
        raise AssertionError("calibration and validation prompts overlap")
    return {"calibration": calibration, "validation": validation}


def weighted_token_loss(logits: Any, labels: Any, weights: Any) -> Any:
    """Response-token cross entropy with a fixed per-token weight budget."""
    import torch
    import torch.nn.functional as functional

    if labels.shape != weights.shape or logits.shape[:2] != labels.shape:
        raise ValueError("weighted loss tensor shapes do not align")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    shifted_weights = weights[:, 1:].float()
    mask = shifted_labels != -100
    effective = shifted_weights * mask
    if not bool((effective.sum(dim=1) > 0).all()):
        raise ValueError("weighted loss has no response tokens")
    targets = shifted_labels.masked_fill(~mask, 0)
    losses = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return (losses * effective).sum() / effective.sum()


def gradient_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    if set(after) != set(before):
        raise ValueError("gradient dictionaries have different parameters")
    return {name: after[name] - before[name] for name in sorted(after)}


def gradient_norm(gradients: dict[str, Any]) -> float:
    if not gradients:
        raise ValueError("gradient dictionary is empty")
    return math.sqrt(sum(float(value.double().square().sum()) for value in gradients.values()))


def gradient_cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("gradient dictionaries have different parameters or are empty")
    names = sorted(left)
    dot = sum(float((left[name].double() * right[name].double()).sum()) for name in names)
    left_norm = gradient_norm(left)
    right_norm = gradient_norm(right)
    return dot / (left_norm * right_norm) if left_norm and right_norm else math.nan


def gradient_hash(gradients: dict[str, Any]) -> str:
    if not gradients:
        raise ValueError("gradient dictionary is empty")
    digest = hashlib.sha256()
    for name in sorted(gradients):
        value = gradients[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"] or not gates["gradient_valid"] or not gates["state_restored"]:
        return "STOP_CITA_P0_INVALID"
    if not gates["token_signal"] or not gates["counterfactual_specificity"]:
        return "STOP_CITA_P0_TOKEN_SIGNAL_WEAK"
    if not gates["gradient_effect_size"] or not gates["sham_distinct"]:
        return "STOP_CITA_P0_GRADIENT_EFFECT_WEAK"
    if not gates["cross_split_replication"]:
        return "STOP_CITA_P0_ALIGNMENT_NOT_REPLICATED"
    return "PASS_CITA_P0_AUTHORIZE_MATCHED_SMOKE"
