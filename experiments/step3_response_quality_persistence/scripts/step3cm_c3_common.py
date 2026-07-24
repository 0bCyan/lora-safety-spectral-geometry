"""Shared no-update representation and loss utilities for Step3C-MF C3."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable


def masked_mean(hidden: Any, mask: Any) -> Any:
    import torch

    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError("hidden/mask shapes do not align")
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * weights).sum(dim=1) / counts
    if not torch.isfinite(pooled).all():
        raise FloatingPointError("non-finite pooled representation")
    return pooled


def l2_normalize(value: Any) -> Any:
    import torch

    return torch.nn.functional.normalize(value, p=2, dim=-1, eps=1e-12)


def projection_head(input_dim: int, output_dim: int, seed: int) -> Any:
    import torch

    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    head = torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.GELU(),
        torch.nn.Linear(output_dim, output_dim),
    )
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    torch.random.set_rng_state(state)
    return head


def positive_mask(query_prompt_ids: list[str], key_prompt_ids: list[str], device: Any = None) -> Any:
    import torch

    return torch.tensor(
        [[query_id == key_id for key_id in key_prompt_ids] for query_id in query_prompt_ids],
        dtype=torch.bool,
        device=device,
    )


def hard_negative_mask(
    query_prompt_ids: list[str],
    query_categories: list[str],
    key_prompt_ids: list[str],
    key_categories: list[str],
    device: Any = None,
) -> Any:
    import torch

    return torch.tensor(
        [
            [query_categories[i] == key_categories[j] and query_prompt_ids[i] != key_prompt_ids[j] for j in range(len(key_prompt_ids))]
            for i in range(len(query_prompt_ids))
        ],
        dtype=torch.bool,
        device=device,
    )


def multi_positive_loss(
    queries: Any,
    keys: Any,
    positives: Any,
    temperature: float,
    hard_negatives: Any | None = None,
    hard_multiplier: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if positives.shape != (queries.shape[0], keys.shape[0]):
        raise ValueError("positive mask shape mismatch")
    if not positives.any(dim=1).all():
        raise ValueError("every query requires at least one positive")
    logits = queries @ keys.T / temperature
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    global_loss = -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
    hard_loss = torch.zeros((), dtype=global_loss.dtype, device=global_loss.device)
    hard_rows = 0
    if hard_negatives is not None and hard_multiplier > 0:
        row_has_hard = hard_negatives.any(dim=1)
        hard_rows = int(row_has_hard.sum().item())
        if hard_rows:
            hard_logits = logits.masked_fill(~hard_negatives, -torch.inf)
            positive_mass = torch.logsumexp(positive_logits[row_has_hard], dim=1)
            hard_mass = torch.logsumexp(hard_logits[row_has_hard], dim=1)
            hard_loss = -(positive_mass - torch.logaddexp(positive_mass, hard_mass)).mean()
    total = global_loss + hard_multiplier * hard_loss
    positive_probability = torch.exp(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1))
    return total, {
        "global_loss": global_loss,
        "hard_loss": hard_loss,
        "hard_rows": hard_rows,
        "positive_probability": positive_probability,
        "logits": logits,
    }


@dataclass
class DeterministicQueue:
    capacity: int
    dimension: int

    def __post_init__(self) -> None:
        import torch

        if self.capacity <= 0 or self.dimension <= 0:
            raise ValueError("queue dimensions must be positive")
        self.values = torch.zeros(self.capacity, self.dimension, dtype=torch.float32)
        self.prompt_ids: list[str | None] = [None] * self.capacity
        self.categories: list[str | None] = [None] * self.capacity
        self.position = 0
        self.count = 0

    def enqueue(self, values: Any, prompt_ids: list[str], categories: list[str]) -> None:
        if len(values) != len(prompt_ids) or len(values) != len(categories):
            raise ValueError("queue metadata length mismatch")
        for value, prompt_id, category in zip(values.detach().cpu().float(), prompt_ids, categories):
            self.values[self.position].copy_(value)
            self.prompt_ids[self.position] = prompt_id
            self.categories[self.position] = category
            self.position = (self.position + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def ordered(self) -> tuple[Any, list[str], list[str]]:
        if self.count < self.capacity:
            indices = list(range(self.count))
        else:
            indices = list(range(self.position, self.capacity)) + list(range(0, self.position))
        return (
            self.values[indices].clone(),
            [str(self.prompt_ids[index]) for index in indices],
            [str(self.categories[index]) for index in indices],
        )


def choose_temperature(
    rows: list[dict[str, float]],
    minimum: float,
    maximum: float,
    target: float,
) -> dict[str, float] | None:
    eligible = [row for row in rows if minimum <= row["median_positive_probability"] <= maximum]
    if not eligible:
        return None
    tie_preference = {0.1: 0, 0.2: 1, 0.05: 2}
    return min(
        eligible,
        key=lambda row: (
            round(abs(row["median_positive_probability"] - target), 12),
            tie_preference.get(round(row["temperature"], 5), 3),
            -row["temperature"],
        ),
    )


def calibrate_eta(binding_norm: float, safety_norm: float, target_ratio: float) -> tuple[float, float]:
    if not math.isfinite(binding_norm) or not math.isfinite(safety_norm) or binding_norm <= 0 or safety_norm <= 0:
        return math.nan, math.nan
    eta = target_ratio * safety_norm / binding_norm
    return eta, eta * binding_norm / safety_norm


def cosine(left: Any, right: Any) -> float:
    import torch

    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return math.nan
    return float(torch.dot(left, right) / denominator)


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_mean_interval(values: list[float], repeats: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "lower": math.nan, "upper": math.nan}
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(repeats)]
    return {"mean": sum(values) / len(values), "lower": quantile(means, 0.025), "upper": quantile(means, 0.975)}


def retrieval_metrics(similarities: Any, query_prompt_ids: list[str], key_prompt_ids: list[str]) -> dict[str, float]:
    import torch

    ranks = torch.argsort(similarities, dim=1, descending=True)
    reciprocal = []
    top1 = 0
    top5 = 0
    multi_recall = []
    for row_index, query_id in enumerate(query_prompt_ids):
        ordered = [key_prompt_ids[index] for index in ranks[row_index].tolist()]
        positive_positions = [index + 1 for index, prompt_id in enumerate(ordered) if prompt_id == query_id]
        if not positive_positions:
            reciprocal.append(0.0)
            multi_recall.append(0.0)
            continue
        top1 += int(min(positive_positions) <= 1)
        top5 += int(min(positive_positions) <= 5)
        reciprocal.append(1.0 / min(positive_positions))
        positive_total = sum(prompt_id == query_id for prompt_id in key_prompt_ids)
        multi_recall.append(sum(position <= 5 for position in positive_positions) / positive_total)
    count = len(query_prompt_ids)
    return {
        "top1": top1 / count,
        "top5": top5 / count,
        "mrr": sum(reciprocal) / count,
        "multi_positive_recall_at5": sum(multi_recall) / count,
    }


def effective_rank(embeddings: Any) -> dict[str, float]:
    import torch

    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    total = variance.sum()
    if float(total) <= 0:
        return {"effective_rank": 0.0, "top_singular_value_share": 1.0}
    probabilities = variance / total
    entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
    return {
        "effective_rank": float(torch.exp(entropy)),
        "top_singular_value_share": float(variance.max() / total),
    }
