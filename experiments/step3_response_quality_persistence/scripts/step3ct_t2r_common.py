"""Shared scalar-only diagnostics for the Step3C-T2R recovery."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import step3ct_training_common as parent


def ordered_gradient_names(gradients: dict[str, Any]) -> list[str]:
    return sorted(gradients)


def gradient_hash(gradients: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ordered_gradient_names(gradients):
        value = gradients[name].detach().cpu().to(dtype=__import__("torch").float64).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def zeros_like_float64(gradients: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        name: torch.zeros_like(value, dtype=torch.float64, device="cpu")
        for name, value in sorted(gradients.items())
    }


def add_in_place(total: dict[str, Any], gradients: dict[str, Any], divisor: float = 1.0) -> None:
    if set(total) != set(gradients):
        raise ValueError("gradient parameter sets differ")
    for name in sorted(total):
        total[name].add_(gradients[name].detach().cpu().to(dtype=total[name].dtype), alpha=1.0 / divisor)


def sum_gradients(gradients: list[dict[str, Any]]) -> dict[str, Any]:
    if not gradients:
        raise ValueError("cannot sum an empty gradient list")
    total = zeros_like_float64(gradients[0])
    for row in gradients:
        add_in_place(total, row)
    return total


def gradient_norm(gradients: dict[str, Any]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in gradients.values()))


def gradient_dot(left: dict[str, Any], right: dict[str, Any]) -> float:
    names = sorted(set(left) & set(right))
    return sum(float((left[name] * right[name]).sum()) for name in names)


def gradient_cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
    denominator = gradient_norm(left) * gradient_norm(right)
    return gradient_dot(left, right) / denominator if denominator else math.nan


def orthogonal_residual_ratio(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    reference_sq = gradient_dot(reference, reference)
    candidate_norm = gradient_norm(candidate)
    if reference_sq <= 0 or candidate_norm <= 0:
        return math.nan
    coefficient = gradient_dot(candidate, reference) / reference_sq
    residual_sq = 0.0
    for name in sorted(candidate):
        residual = candidate[name] - coefficient * reference[name]
        residual_sq += float(residual.square().sum())
    return math.sqrt(max(0.0, residual_sq)) / candidate_norm


def gradients_finite_nonzero(gradients: dict[str, Any]) -> bool:
    import torch

    return bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients.values()) and gradient_norm(gradients) > 0


def parameter_group(name: str) -> str:
    if any(token in name for token in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(token in name for token in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    return "other"


def layer_number(name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def energy_summary(gradients: dict[str, Any]) -> dict[str, Any]:
    group_energy: dict[str, float] = defaultdict(float)
    layer_energy: dict[int, float] = defaultdict(float)
    for name, value in gradients.items():
        energy = float(value.square().sum())
        group_energy[parameter_group(name)] += energy
        layer_energy[layer_number(name)] += energy
    total = sum(group_energy.values())
    return {
        "group_share": {key: value / total if total else 0.0 for key, value in sorted(group_energy.items())},
        "layer_share": {str(key): value / total if total else 0.0 for key, value in sorted(layer_energy.items())},
    }


def lora_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    rows = [
        (name, parameter) for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    for name, parameter in sorted(rows):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metadata_hard_coverage(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    targets = parent.read_jsonl(repo / cfg["safety_targets_path"])
    grouped = parent.targets_by_prompt(targets)
    prefill = parent.deterministic_prefill_rows(
        targets, int(cfg["queue_size"]), int(cfg["queue_prefill_seed"])
    )
    queue = parent.DeterministicQueue(int(cfg["queue_size"]), 1)
    queue.enqueue(
        torch.zeros(len(prefill), 1),
        [str(row["sample_id"]) for row in prefill],
        [str(row["category"]) for row in prefill],
        [str(row["target_id"]) for row in prefill],
    )
    positions = []
    for position, target in enumerate(targets):
        prompt_id = str(target["sample_id"])
        category = str(target["category"])
        queue_hash_before = queue.content_hash()
        _, prompt_ids, categories, target_ids = queue.ordered(prompt_id)
        hard_count = sum(category == candidate for candidate in categories)
        positives = grouped[prompt_id]
        queue.enqueue(
            torch.zeros(2, 1),
            [prompt_id, prompt_id],
            [category, category],
            [str(row["target_id"]) for row in positives],
        )
        positions.append(
            {
                "position": position,
                "fit_eligible": position >= 8,
                "prompt_id_hash": parent.stable_hash(prompt_id),
                "target_id_hash": parent.stable_hash(str(target["target_id"])),
                "hard_negative_count": hard_count,
                "hard_active": hard_count > 0,
                "queue_count": queue.count,
                "queue_hash_before": queue_hash_before,
                "queue_hash_after": queue.content_hash(),
                "negative_target_hash": parent.hash_json([parent.stable_hash(value) for value in target_ids]),
                "same_prompt_excluded": all(value != prompt_id for value in prompt_ids),
            }
        )
    windows = []
    for window_index, start in enumerate(range(8, 80, 4)):
        members = positions[start : start + 4]
        windows.append(
            {
                "window_index": window_index,
                "positions": [row["position"] for row in members],
                "hard_active": any(row["hard_active"] for row in members),
                "hard_active_position_count": sum(row["hard_active"] for row in members),
            }
        )
    result = {
        "name": "step3ct_t2r_metadata_hard_coverage_v1",
        "target_positions": len(positions),
        "hard_active_positions_all": sum(row["hard_active"] for row in positions),
        "eta_fit_excluded_positions": [row["position"] for row in positions if not row["fit_eligible"]],
        "hard_active_positions_eta_fit_excluded": sum(row["hard_active"] for row in positions[:8]),
        "calibration_positions": sum(row["fit_eligible"] for row in positions),
        "hard_active_positions_calibration": sum(row["hard_active"] for row in positions[8:]),
        "calibration_windows": len(windows),
        "hard_active_calibration_windows": sum(row["hard_active"] for row in windows),
        "first_position_hard_negative_count": positions[0]["hard_negative_count"],
        "all_same_prompt_exclusions_pass": all(row["same_prompt_excluded"] for row in positions),
        "positions": positions,
        "windows": windows,
    }
    result["passed"] = (
        result["target_positions"] == 80
        and result["hard_active_positions_all"] == 58
        and result["eta_fit_excluded_positions"] == list(range(8))
        and result["hard_active_positions_eta_fit_excluded"] == 4
        and result["calibration_positions"] == 72
        and result["hard_active_positions_calibration"] == 54
        and result["calibration_windows"] == 18
        and result["hard_active_calibration_windows"] == 17
        and result["first_position_hard_negative_count"] == 0
        and result["all_same_prompt_exclusions_pass"]
    )
    return result


def calibration_windows(position_rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    all_positions = [int(row["position"]) for row in position_rows]
    if all_positions != list(range(80)):
        raise ValueError("calibration trace must contain positions 0-79 exactly once")
    eligible = [row for row in position_rows if int(row["position"]) >= 8]
    positions = [int(row["position"]) for row in eligible]
    if positions != list(range(8, 80)):
        raise ValueError("eta-fit positions must be exactly 8-79 once each")
    return [eligible[index : index + 4] for index in range(0, len(eligible), 4)]


def scalar_schema() -> dict[str, Any]:
    return {
        "position_fields": [
            "position", "fit_eligible", "risk_row_hash", "safety_target_hash",
            "queue_hash_before", "queue_hash_after", "hard_active", "hard_negative_count",
            "safety_gradient_norm", "c3a_gradient_norm", "c3b_gradient_norm",
            "safety_gradient_hash", "c3a_gradient_hash", "c3b_gradient_hash",
        ],
        "window_fields": [
            "window_index", "positions", "hard_active", "safety_gradient_norm",
            "c3a_gradient_norm", "c3b_gradient_norm", "c3a_unweighted_ratio",
            "c3b_unweighted_ratio", "c3a_c3b_cosine", "c3b_residual_given_c3a",
        ],
        "full_gradient_tensors_committed": False,
    }


def dual_binding_gradients(
    model: Any,
    tokenizer: Any,
    head: Any,
    prompt_row: dict[str, Any],
    positive_rows: list[dict[str, Any]],
    queue: Any,
    max_length: int,
    temperature: float,
    coefficient: float,
    update_queue: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute C3a/C3b gradients from identical embeddings and queue state."""
    import torch

    if coefficient <= 0:
        raise ValueError("coefficient must be positive")
    prompt_id = str(prompt_row["sample_id"])
    category = str(prompt_row["category"])
    prompt = str(prompt_row["prompt"])
    with parent.binding_mode(model):
        prompt_input = parent.prompt_feature(tokenizer, prompt, max_length)
        response_inputs = [
            parent.response_feature(tokenizer, prompt, str(row["response"]), max_length)
            for row in positive_rows
        ]
        query = parent.encode_projected(model, prompt_input, "prompt", head, with_grad=False)
        positive_values = torch.stack(
            [parent.encode_projected(model, feature, "response", head, with_grad=False) for feature in response_inputs]
        )
        negative_values, negative_prompt_ids, negative_categories, negative_target_ids = queue.ordered(prompt_id)
        if not negative_prompt_ids:
            raise ValueError("queue has no cross-prompt negatives")
        negative_values = negative_values.to(query.device)
        vector_a, metrics_a = parent.binding_embedding_gradients(
            query, positive_values, negative_values, prompt_id, category,
            negative_prompt_ids, negative_categories, temperature, 0.0,
        )
        vector_b, metrics_b = parent.binding_embedding_gradients(
            query, positive_values, negative_values, prompt_id, category,
            negative_prompt_ids, negative_categories, temperature, 1.0,
        )
        branches = [(prompt_input, "prompt"), *[(feature, "response") for feature in response_inputs]]
        total_a: dict[str, Any] | None = None
        total_b: dict[str, Any] | None = None
        for (feature, kind), grad_a, grad_b in zip(branches, vector_a, vector_b):
            projected = parent.encode_projected(model, feature, kind, head, with_grad=True)
            model.zero_grad(set_to_none=True)
            torch.autograd.backward(projected, grad_a.to(projected.device) * coefficient, retain_graph=True)
            current_a = parent.capture_lora_gradients(model)
            if total_a is None:
                total_a = zeros_like_float64(current_a)
            add_in_place(total_a, current_a, divisor=coefficient)
            model.zero_grad(set_to_none=True)
            torch.autograd.backward(projected, grad_b.to(projected.device) * coefficient)
            current_b = parent.capture_lora_gradients(model)
            if total_b is None:
                total_b = zeros_like_float64(current_b)
            add_in_place(total_b, current_b, divisor=coefficient)
        if total_a is None or total_b is None:
            raise RuntimeError("no binding gradients captured")
        queue_hash_before = queue.content_hash()
        evicted: list[str] = []
        if update_queue:
            evicted = queue.enqueue(
                positive_values,
                [prompt_id, prompt_id],
                [category, category],
                [str(row["target_id"]) for row in positive_rows],
            )
        metadata = {
            "c3a": metrics_a,
            "c3b": metrics_b,
            "queue_hash_before": queue_hash_before,
            "queue_hash_after": queue.content_hash(),
            "queue_count": queue.count,
            "negative_count": len(negative_prompt_ids),
            "negative_target_hash": parent.hash_json([parent.stable_hash(value) for value in negative_target_ids]),
            "inserted_target_hashes": [parent.stable_hash(str(row["target_id"])) for row in positive_rows],
            "evicted_target_hashes": [parent.stable_hash(value) for value in evicted],
        }
        model.zero_grad(set_to_none=True)
        return total_a, total_b, metadata
