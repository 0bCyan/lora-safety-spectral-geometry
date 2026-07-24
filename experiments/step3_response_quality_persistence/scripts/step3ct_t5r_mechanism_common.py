"""Scalar-only helpers for the frozen Step3C-T5R mechanism completion."""

from __future__ import annotations

import contextlib
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import step3cm_c3_common as representation_common
import step3ct_t2r_common as gradient_common
import step3ct_training_common as training_common


def disposable_prefix(label: int, arm_index: int) -> str:
    if label < 0 or arm_index < 0:
        raise ValueError("disposable identifiers must be non-negative")
    prefix = f"g{label:02d}_{arm_index}_"
    if len(prefix) > 12:
        raise ValueError("disposable prefix exceeds the Windows-safe bound")
    return prefix


def hash_file_set(paths: list[Path], root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Hash immutable artifacts without reading model tensors into Python."""
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        key = path.relative_to(root.resolve()).as_posix() if root is not None else path.as_posix()
        result[key] = {"sha256": training_common.sha256_file(path), "bytes": path.stat().st_size}
    return result


def compare_file_sets(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    return {"passed": not changed, "changed_paths": changed, "artifact_count": len(before)}


def canonical_diagnostic_rows(
    targets: list[dict[str, Any]],
    expected_prompts: int = 40,
    expected_targets: int = 80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(targets) != expected_targets:
        raise ValueError(f"expected {expected_targets} diagnostic targets, found {len(targets)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_ids: set[str] = set()
    for row in targets:
        required = {"sample_id", "target_id", "prompt", "response", "category"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"target row is missing fields: {sorted(missing)}")
        target_id = str(row["target_id"])
        if target_id in target_ids:
            raise ValueError(f"duplicate target_id: {target_id}")
        target_ids.add(target_id)
        grouped[str(row["sample_id"])].append(row)
    if len(grouped) != expected_prompts:
        raise ValueError(f"expected {expected_prompts} diagnostic prompts, found {len(grouped)}")
    for prompt_id, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"prompt {prompt_id} does not have exactly two targets")
        prompt_values = {(str(row["prompt"]), str(row["category"])) for row in rows}
        if len(prompt_values) != 1:
            raise ValueError(f"prompt/category metadata differs within prompt {prompt_id}")
    prompt_rows = [sorted(grouped[prompt_id], key=lambda row: str(row["target_id"]))[0] for prompt_id in sorted(grouped)]
    target_rows = sorted(targets, key=lambda row: (str(row["sample_id"]), str(row["target_id"])))
    order = {
        "prompt_count": len(prompt_rows),
        "target_count": len(target_rows),
        "prompt_order_hash": training_common.hash_json([str(row["sample_id"]) for row in prompt_rows]),
        "target_order_hash": training_common.hash_json([str(row["target_id"]) for row in target_rows]),
        "prompt_category_hash": training_common.hash_json(
            [(str(row["sample_id"]), str(row["category"])) for row in prompt_rows]
        ),
    }
    return prompt_rows, target_rows, order


def validate_cross_arm_order(orders: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not orders:
        raise ValueError("no arm orders were supplied")
    fields = ("prompt_count", "target_count", "prompt_order_hash", "target_order_hash", "prompt_category_hash")
    reference_arm = sorted(orders)[0]
    reference = orders[reference_arm]
    mismatches = {
        arm: [field for field in fields if order.get(field) != reference.get(field)]
        for arm, order in sorted(orders.items())
        if any(order.get(field) != reference.get(field) for field in fields)
    }
    return {"passed": not mismatches, "reference_arm": reference_arm, "mismatches": mismatches}


@contextlib.contextmanager
def isolated_rng() -> Iterator[None]:
    import torch

    python_state = random.getstate()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


@contextlib.contextmanager
def diagnostic_eval(model: Any) -> Iterator[None]:
    """Disable all dropout for diagnostic forwards and restore every module mode."""
    states = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    if any(module.training for module in model.modules() if module.__class__.__name__.lower().startswith("dropout")):
        raise RuntimeError("diagnostic dropout remained enabled")
    try:
        yield
    finally:
        for module, state in states:
            module.training = state


def distribution_summary(values: list[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "mean": None, "median": None, "q025": None, "q25": None, "q75": None, "q975": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": representation_common.quantile(finite, 0.5),
        "q025": representation_common.quantile(finite, 0.025),
        "q25": representation_common.quantile(finite, 0.25),
        "q75": representation_common.quantile(finite, 0.75),
        "q975": representation_common.quantile(finite, 0.975),
    }


def bootstrap_prompt_mean(
    rows: list[dict[str, Any]],
    field: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    if not values:
        return {"clusters": 0, "repeats": repeats, "seed": seed, "mean": None, "lower": None, "upper": None}
    interval = representation_common.bootstrap_mean_interval(values, repeats, seed)
    return {"clusters": len(values), "repeats": repeats, "seed": seed, **interval}


def permutation_controls(
    similarities: Any,
    query_ids: list[str],
    key_ids: list[str],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped_ids = sorted(set(key_ids))
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        shuffled = grouped_ids[:]
        rng.shuffle(shuffled)
        mapping = dict(zip(grouped_ids, shuffled))
        permuted_ids = [mapping[prompt_id] for prompt_id in key_ids]
        rows.append({"repeat": repeat, **representation_common.retrieval_metrics(similarities, query_ids, permuted_ids)})
    return rows


def representation_metrics(
    prompt_embeddings: Any,
    target_embeddings: Any,
    prompt_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    bootstrap_repeats: int,
    bootstrap_seed: int,
    permutation_repeats: int,
    permutation_seed: int,
) -> dict[str, Any]:
    import torch

    if prompt_embeddings.shape[0] != len(prompt_rows) or target_embeddings.shape[0] != len(target_rows):
        raise ValueError("embedding and metadata dimensions differ")
    if not bool(torch.isfinite(prompt_embeddings).all()) or not bool(torch.isfinite(target_embeddings).all()):
        raise FloatingPointError("non-finite diagnostic embeddings")
    prompt_norms = torch.linalg.vector_norm(prompt_embeddings.float(), dim=1)
    target_norms = torch.linalg.vector_norm(target_embeddings.float(), dim=1)
    if not torch.allclose(prompt_norms, torch.ones_like(prompt_norms), atol=1e-5, rtol=0):
        raise ValueError("prompt embeddings are not L2 normalized")
    if not torch.allclose(target_norms, torch.ones_like(target_norms), atol=1e-5, rtol=0):
        raise ValueError("target embeddings are not L2 normalized")

    similarities = prompt_embeddings.float() @ target_embeddings.float().T
    query_ids = [str(row["sample_id"]) for row in prompt_rows]
    key_ids = [str(row["sample_id"]) for row in target_rows]
    cosine_values: dict[str, list[float]] = defaultdict(list)
    per_prompt: list[dict[str, Any]] = []
    for query_index, prompt in enumerate(prompt_rows):
        prompt_id = str(prompt["sample_id"])
        category = str(prompt["category"])
        positive: list[float] = []
        hard: list[float] = []
        cross: list[float] = []
        for target_index, target in enumerate(target_rows):
            value = float(similarities[query_index, target_index])
            if str(target["sample_id"]) == prompt_id:
                positive.append(value)
                cosine_values["same_prompt_positive"].append(value)
            elif str(target["category"]) == category:
                hard.append(value)
                cosine_values["same_category_cross_prompt"].append(value)
            else:
                cross.append(value)
                cosine_values["cross_category_negative"].append(value)
        global_negative = hard + cross
        cosine_values["global_negative"].extend(global_negative)
        row = {
            "prompt_hash": training_common.stable_hash(prompt_id),
            "category_hash": training_common.stable_hash(category),
            "positive_mean": sum(positive) / len(positive),
            "global_negative_mean": sum(global_negative) / len(global_negative),
            "hard_negative_mean": sum(hard) / len(hard) if hard else math.nan,
            "positive_minus_global": sum(positive) / len(positive) - sum(global_negative) / len(global_negative),
            "positive_minus_hard": sum(positive) / len(positive) - sum(hard) / len(hard) if hard else math.nan,
        }
        per_prompt.append(row)

    result = {
        "retrieval": representation_common.retrieval_metrics(similarities, query_ids, key_ids),
        "cosine_distributions": {key: distribution_summary(value) for key, value in sorted(cosine_values.items())},
        "gap_distributions": {
            "positive_minus_global": distribution_summary([row["positive_minus_global"] for row in per_prompt]),
            "positive_minus_hard": distribution_summary([row["positive_minus_hard"] for row in per_prompt]),
        },
        "bootstrap_intervals": {
            "positive_minus_global": bootstrap_prompt_mean(
                per_prompt, "positive_minus_global", bootstrap_repeats, bootstrap_seed
            ),
            "positive_minus_hard": bootstrap_prompt_mean(
                per_prompt, "positive_minus_hard", bootstrap_repeats, bootstrap_seed
            ),
        },
        "permutation_controls": permutation_controls(
            similarities, query_ids, key_ids, permutation_repeats, permutation_seed
        ),
        "prompt_geometry": representation_common.effective_rank(prompt_embeddings),
        "target_geometry": training_common.effective_rank(target_embeddings),
        "per_prompt": per_prompt,
    }
    assert_no_full_gradient_serialization(result)
    return result


def _select_gradients(gradients: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {name: gradients[name] for name in names}


def grouped_gradient_metrics(
    safety: dict[str, Any],
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    names = sorted(safety)
    groups: dict[str, list[str]] = defaultdict(list)
    layers: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name in names:
        group = gradient_common.parameter_group(name)
        groups[group].append(name)
        layers[(gradient_common.layer_number(name), group)].append(name)
    result: dict[str, Any] = {
        "safety_gradient_hash": gradient_common.gradient_hash(safety),
        "safety_gradient_norm": gradient_common.gradient_norm(safety),
        "safety_energy": gradient_common.energy_summary(safety),
        "binding_gradient_hash": "",
        "binding_gradient_norm": 0.0,
        "global_binding_safety_cosine": None,
        "group_cosine": {group: None for group in sorted(groups)},
        "binding_energy": {"group_share": {}, "layer_share": {}},
        "per_layer_group": [],
    }
    if binding is None:
        return result
    if set(binding) != set(safety):
        raise ValueError("binding and safety gradient parameter sets differ")
    result.update(
        {
            "binding_gradient_hash": gradient_common.gradient_hash(binding),
            "binding_gradient_norm": gradient_common.gradient_norm(binding),
            "global_binding_safety_cosine": gradient_common.gradient_cosine(binding, safety),
            "group_cosine": {
                group: gradient_common.gradient_cosine(
                    _select_gradients(binding, selected), _select_gradients(safety, selected)
                )
                for group, selected in sorted(groups.items())
            },
            "binding_energy": gradient_common.energy_summary(binding),
            "per_layer_group": [
                {
                    "layer": layer,
                    "group": group,
                    "parameter_tensors": len(selected),
                    "binding_norm": gradient_common.gradient_norm(_select_gradients(binding, selected)),
                    "safety_norm": gradient_common.gradient_norm(_select_gradients(safety, selected)),
                    "cosine": gradient_common.gradient_cosine(
                        _select_gradients(binding, selected), _select_gradients(safety, selected)
                    ),
                }
                for (layer, group), selected in sorted(layers.items())
            ],
        }
    )
    assert_no_full_gradient_serialization(result)
    return result


def cross_arm_gradient_metrics(
    c3a: dict[str, Any],
    c3b: dict[str, Any],
) -> dict[str, Any]:
    if set(c3a) != set(c3b):
        raise ValueError("C3a and C3b gradient parameter sets differ")
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in sorted(c3a):
        grouped[gradient_common.parameter_group(name)].append(name)
    result = {
        "direction": "C3b_relative_to_C3a",
        "global_cosine": gradient_common.gradient_cosine(c3a, c3b),
        "global_orthogonal_residual_ratio": gradient_common.orthogonal_residual_ratio(c3a, c3b),
        "parameter_groups": {
            group: {
                "cosine": gradient_common.gradient_cosine(
                    _select_gradients(c3a, names), _select_gradients(c3b, names)
                ),
                "orthogonal_residual_ratio": gradient_common.orthogonal_residual_ratio(
                    _select_gradients(c3a, names), _select_gradients(c3b, names)
                ),
            }
            for group, names in sorted(grouped.items())
        },
    }
    assert_no_full_gradient_serialization(result)
    return result


def assert_no_full_gradient_serialization(value: Any, path: str = "root") -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        raise TypeError(f"tensor serialization is forbidden at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"gradients", "gradient_tensors", "full_gradient"}:
                raise ValueError(f"full-gradient field is forbidden at {path}.{key}")
            assert_no_full_gradient_serialization(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_full_gradient_serialization(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"non-scalar payload type {type(value).__name__} at {path}")


def redact_failed_grouped_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Remove approximate grouped outcomes when exact reconstruction did not pass."""
    if bool(result.get("passed")):
        return result
    for arm_result in result.get("arms", {}).values():
        if not bool(arm_result.get("reproduction_passed")):
            arm_result["grouped_metrics"] = {
                "registered": False,
                "withheld_reason": "exact registered diagnostic reconstruction failed",
            }
    result["c3a_c3b_direction"] = {
        "registered": False,
        "withheld_reason": "at least one arm failed exact registered diagnostic reconstruction",
    }
    result["approximate_grouped_metrics_reported"] = False
    assert_no_full_gradient_serialization(result)
    return result


def numeric_close(left: Any, right: Any, tolerance: float) -> bool:
    if left == "" or right == "":
        return left == right
    if left is None or right is None:
        return left is right
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    return math.isclose(left_value, right_value, rel_tol=tolerance, abs_tol=tolerance)


def compare_registered_diagnostic(
    observed: dict[str, Any],
    registered: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    exact_fields = (
        "checkpoint_step", "optimizer_step", "gradient_unit", "positions", "microstep_count",
        "safety_gradient_hash", "binding_gradient_hash", "hard_active_positions", "finite_nonzero",
    )
    numeric_fields = (
        "safety_gradient_norm", "binding_gradient_norm", "unweighted_binding_safety_ratio",
        "eta_weighted_binding_safety_ratio", "binding_safety_cosine",
    )
    mismatches: dict[str, Any] = {}
    for field in exact_fields:
        if observed.get(field) != registered.get(field):
            mismatches[field] = {"observed": observed.get(field), "registered": registered.get(field)}
    for field in numeric_fields:
        if not numeric_close(observed.get(field), registered.get(field), tolerance):
            mismatches[field] = {"observed": observed.get(field), "registered": registered.get(field)}
    for energy_field in ("safety_energy", "binding_energy"):
        keys = sorted(set(observed.get(energy_field, {})) | set(registered.get(energy_field, {})))
        for key in keys:
            if not numeric_close(
                observed.get(energy_field, {}).get(key), registered.get(energy_field, {}).get(key), tolerance
            ):
                mismatches[f"{energy_field}.{key}"] = {
                    "observed": observed.get(energy_field, {}).get(key),
                    "registered": registered.get(energy_field, {}).get(key),
                }
    return {"passed": not mismatches, "tolerance": tolerance, "mismatches": mismatches}
