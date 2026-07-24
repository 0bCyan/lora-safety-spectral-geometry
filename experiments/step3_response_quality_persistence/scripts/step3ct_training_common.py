"""Shared deterministic training utilities for the Step3C-T tournament."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return stable_hash(payload)


def format_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_chat_text(tokenizer: Any, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prompt_feature(tokenizer: Any, prompt: str, max_length: int) -> dict[str, list[int]]:
    # This deliberately matches the frozen Step3C-M prompt encoding.
    return tokenizer(prompt, truncation=True, max_length=max_length, padding=False)


def response_feature(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, list[int]]:
    prompt_text = format_prompt_text(tokenizer, prompt)
    full_text = format_chat_text(tokenizer, prompt, response)
    full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    prefix = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
    labels = list(full["input_ids"])
    labels[: min(len(prefix), len(labels))] = [-100] * min(len(prefix), len(labels))
    if not any(label != -100 for label in labels):
        raise ValueError("response was fully truncated")
    full["labels"] = labels
    return full


def batch_feature(feature: dict[str, list[int]], device: Any, include_labels: bool = True) -> dict[str, Any]:
    import torch

    result = {
        "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device),
    }
    if include_labels:
        result["labels"] = torch.tensor([feature["labels"]], dtype=torch.long, device=device)
    return result


def masked_mean(hidden: Any, mask: Any) -> Any:
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * weights).sum(dim=1) / counts
    if not pooled.isfinite().all():
        raise FloatingPointError("non-finite pooled representation")
    return pooled


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
    head.requires_grad_(False)
    head.eval()
    return head


def module_state_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def isolated_rng() -> Iterator[None]:
    import torch

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        random.setstate(python_state)


@contextlib.contextmanager
def binding_mode(model: Any) -> Iterator[None]:
    was_training = model.training
    model.eval()
    with isolated_rng():
        try:
            yield
        finally:
            model.train(was_training)


def model_device(model: Any) -> Any:
    return next(parameter.device for parameter in model.parameters() if parameter.device.type == "cuda")


def encode_projected(
    model: Any,
    feature: dict[str, list[int]],
    mask_kind: str,
    head: Any,
    with_grad: bool,
) -> Any:
    import torch

    device = model_device(model)
    batch = batch_feature(feature, device, include_labels=mask_kind == "response")
    if mask_kind == "prompt":
        pool_mask = batch["attention_mask"]
    else:
        pool_mask = (batch["labels"] != -100).long() * batch["attention_mask"]
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        pooled = masked_mean(outputs.hidden_states[-1], pool_mask)
        projected = torch.nn.functional.normalize(head(pooled.float()), p=2, dim=-1, eps=1e-12)
    return projected.squeeze(0)


def positive_mask(query_prompt_ids: list[str], key_prompt_ids: list[str], device: Any) -> Any:
    import torch

    return torch.tensor(
        [[query == key for key in key_prompt_ids] for query in query_prompt_ids],
        dtype=torch.bool,
        device=device,
    )


def hard_negative_mask(
    query_prompt_ids: list[str],
    query_categories: list[str],
    key_prompt_ids: list[str],
    key_categories: list[str],
    device: Any,
) -> Any:
    import torch

    return torch.tensor(
        [
            [
                query_categories[i] == key_categories[j] and query_prompt_ids[i] != key_prompt_ids[j]
                for j in range(len(key_prompt_ids))
            ]
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
    if not positives.any(dim=1).all():
        raise ValueError("every query requires at least one positive")
    logits = queries @ keys.T / temperature
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    positive_mass = torch.logsumexp(positive_logits, dim=1)
    global_loss = -(positive_mass - torch.logsumexp(logits, dim=1)).mean()
    hard_loss = torch.zeros((), dtype=global_loss.dtype, device=global_loss.device)
    hard_rows = 0
    if hard_negatives is not None and hard_multiplier > 0:
        rows = hard_negatives.any(dim=1)
        hard_rows = int(rows.sum().item())
        if hard_rows:
            hard_logits = logits.masked_fill(~hard_negatives, -torch.inf)
            hard_mass = torch.logsumexp(hard_logits[rows], dim=1)
            hard_loss = -(positive_mass[rows] - torch.logaddexp(positive_mass[rows], hard_mass)).mean()
    total = global_loss + hard_multiplier * hard_loss
    probability = torch.exp(positive_mass - torch.logsumexp(logits, dim=1))
    return total, {
        "global_loss": global_loss,
        "hard_loss": hard_loss,
        "hard_rows": hard_rows,
        "positive_probability": probability,
    }


@dataclass
class DeterministicQueue:
    capacity: int
    dimension: int

    def __post_init__(self) -> None:
        import torch

        self.values = torch.zeros(self.capacity, self.dimension, dtype=torch.float32)
        self.prompt_ids: list[str | None] = [None] * self.capacity
        self.categories: list[str | None] = [None] * self.capacity
        self.target_ids: list[str | None] = [None] * self.capacity
        self.position = 0
        self.count = 0
        self.insertions = 0

    def enqueue(
        self,
        values: Any,
        prompt_ids: list[str],
        categories: list[str],
        target_ids: list[str],
    ) -> list[str]:
        if not (len(values) == len(prompt_ids) == len(categories) == len(target_ids)):
            raise ValueError("queue metadata length mismatch")
        evicted: list[str] = []
        for value, prompt_id, category, target_id in zip(
            values.detach().cpu().float(), prompt_ids, categories, target_ids
        ):
            if self.count == self.capacity and self.target_ids[self.position] is not None:
                evicted.append(str(self.target_ids[self.position]))
            self.values[self.position].copy_(value)
            self.prompt_ids[self.position] = prompt_id
            self.categories[self.position] = category
            self.target_ids[self.position] = target_id
            self.position = (self.position + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)
            self.insertions += 1
        return evicted

    def ordered(self, exclude_prompt_id: str | None = None) -> tuple[Any, list[str], list[str], list[str]]:
        if self.count < self.capacity:
            indices = list(range(self.count))
        else:
            indices = list(range(self.position, self.capacity)) + list(range(self.position))
        if exclude_prompt_id is not None:
            indices = [index for index in indices if self.prompt_ids[index] != exclude_prompt_id]
        return (
            self.values[indices].clone(),
            [str(self.prompt_ids[index]) for index in indices],
            [str(self.categories[index]) for index in indices],
            [str(self.target_ids[index]) for index in indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "dimension": self.dimension,
            "values": self.values.clone(),
            "prompt_ids": list(self.prompt_ids),
            "categories": list(self.categories),
            "target_ids": list(self.target_ids),
            "position": self.position,
            "count": self.count,
            "insertions": self.insertions,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DeterministicQueue":
        queue = cls(int(state["capacity"]), int(state["dimension"]))
        queue.values.copy_(state["values"].float().cpu())
        queue.prompt_ids = list(state["prompt_ids"])
        queue.categories = list(state["categories"])
        queue.target_ids = list(state["target_ids"])
        queue.position = int(state["position"])
        queue.count = int(state["count"])
        queue.insertions = int(state["insertions"])
        return queue

    def content_hash(self) -> str:
        digest = hashlib.sha256(self.values[: self.count].numpy().tobytes())
        digest.update(json.dumps(self.prompt_ids, ensure_ascii=False).encode("utf-8"))
        digest.update(json.dumps(self.categories, ensure_ascii=False).encode("utf-8"))
        digest.update(json.dumps(self.target_ids, ensure_ascii=False).encode("utf-8"))
        digest.update(f"{self.position}:{self.count}:{self.insertions}".encode("ascii"))
        return digest.hexdigest()


def deterministic_prefill_rows(targets: list[dict[str, Any]], capacity: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(targets, key=lambda row: stable_hash(f"{seed}:{row['target_id']}"))
    return ordered[:capacity]


def targets_by_prompt(targets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        grouped[str(row["sample_id"])].append(row)
    for prompt_id in grouped:
        grouped[prompt_id].sort(key=lambda row: str(row["target_id"]))
    return dict(grouped)


def weighted_total_loss(risk: Any, safety: Any, binding: Any, p: float, eta: float) -> Any:
    return (1.0 - p) * risk + p * (safety + eta * binding)


def binding_embedding_gradients(
    query: Any,
    positives: Any,
    negatives: Any,
    prompt_id: str,
    category: str,
    negative_prompt_ids: list[str],
    negative_categories: list[str],
    temperature: float,
    hard_multiplier: float,
) -> tuple[list[Any], dict[str, float | int]]:
    import torch

    query_leaf = query.detach().float().requires_grad_(True)
    positive_leaf = positives.detach().float().requires_grad_(True)
    negative_leaf = negatives.detach().float()
    keys = torch.cat([positive_leaf, negative_leaf], dim=0)
    key_prompt_ids = [prompt_id] * len(positive_leaf) + negative_prompt_ids
    key_categories = [category] * len(positive_leaf) + negative_categories
    positives_mask = positive_mask([prompt_id], key_prompt_ids, query.device)
    hard_mask = hard_negative_mask([prompt_id], [category], key_prompt_ids, key_categories, query.device)
    loss, detail = multi_positive_loss(
        query_leaf.unsqueeze(0),
        keys,
        positives_mask,
        temperature,
        hard_mask,
        hard_multiplier,
    )
    query_grad, positive_grad = torch.autograd.grad(loss, [query_leaf, positive_leaf])
    return [query_grad, *positive_grad.unbind(0)], {
        "binding_loss": float(loss.detach().cpu()),
        "global_binding_loss": float(detail["global_loss"].detach().cpu()),
        "hard_binding_loss": float(detail["hard_loss"].detach().cpu()),
        "hard_rows": int(detail["hard_rows"]),
        "positive_probability": float(detail["positive_probability"][0].detach().cpu()),
        "negative_count": len(negative_prompt_ids),
    }


def binding_backward(
    model: Any,
    tokenizer: Any,
    head: Any,
    prompt_row: dict[str, Any],
    positive_rows: list[dict[str, Any]],
    queue: DeterministicQueue,
    max_length: int,
    temperature: float,
    hard_multiplier: float,
    coefficient: float,
    update_queue: bool = True,
) -> dict[str, Any]:
    import torch

    if len(positive_rows) != 2:
        raise ValueError("binding requires exactly two approved positives")
    prompt_id = str(prompt_row["sample_id"])
    category = str(prompt_row["category"])
    prompt = str(prompt_row["prompt"])
    with binding_mode(model):
        prompt_input = prompt_feature(tokenizer, prompt, max_length)
        response_inputs = [
            response_feature(tokenizer, prompt, str(row["response"]), max_length) for row in positive_rows
        ]
        query = encode_projected(model, prompt_input, "prompt", head, with_grad=False)
        positive_values = torch.stack(
            [encode_projected(model, feature, "response", head, with_grad=False) for feature in response_inputs]
        )
        negative_values, negative_prompt_ids, negative_categories, negative_target_ids = queue.ordered(prompt_id)
        if not negative_prompt_ids:
            raise ValueError("queue has no cross-prompt negatives")
        negative_values = negative_values.to(query.device)
        vector_grads, metrics = binding_embedding_gradients(
            query,
            positive_values,
            negative_values,
            prompt_id,
            category,
            negative_prompt_ids,
            negative_categories,
            temperature,
            hard_multiplier,
        )
        branches = [(prompt_input, "prompt"), *[(feature, "response") for feature in response_inputs]]
        for (feature, kind), vector_grad in zip(branches, vector_grads):
            projected = encode_projected(model, feature, kind, head, with_grad=True)
            torch.autograd.backward(projected, vector_grad.to(projected.device) * coefficient)
        queue_hash_before = queue.content_hash()
        evicted: list[str] = []
        if update_queue:
            evicted = queue.enqueue(
                positive_values,
                [prompt_id, prompt_id],
                [category, category],
                [str(row["target_id"]) for row in positive_rows],
            )
        metrics.update(
            {
                "queue_count": queue.count,
                "queue_hash_before": queue_hash_before,
                "queue_hash_after": queue.content_hash(),
                "inserted_target_hashes": [stable_hash(str(row["target_id"])) for row in positive_rows],
                "evicted_target_hashes": [stable_hash(value) for value in evicted],
                "negative_id_hash": hash_json([stable_hash(value) for value in negative_target_ids]),
            }
        )
        return metrics


def prefill_queue(
    model: Any,
    tokenizer: Any,
    head: Any,
    targets: list[dict[str, Any]],
    capacity: int,
    seed: int,
    max_length: int,
) -> tuple[DeterministicQueue, list[str]]:
    import torch

    rows = deterministic_prefill_rows(targets, capacity, seed)
    queue = DeterministicQueue(capacity, int(head[-1].out_features))
    values = []
    with binding_mode(model):
        for row in rows:
            feature = response_feature(tokenizer, str(row["prompt"]), str(row["response"]), max_length)
            values.append(encode_projected(model, feature, "response", head, with_grad=False).cpu())
    queue.enqueue(
        torch.stack(values),
        [str(row["sample_id"]) for row in rows],
        [str(row["category"]) for row in rows],
        [str(row["target_id"]) for row in rows],
    )
    return queue, [str(row["target_id"]) for row in rows]


def trainable_parameter_inventory(model: Any) -> dict[str, Any]:
    rows = []
    unexpected = []
    total = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            count = parameter.numel()
            total += count
            rows.append({"name": name, "numel": count, "dtype": str(parameter.dtype)})
            if "lora_" not in name:
                unexpected.append(name)
    return {
        "trainable_tensors": len(rows),
        "trainable_parameters": total,
        "unexpected_trainable": unexpected,
        "parameters": rows,
        "passed": bool(rows) and not unexpected,
    }


def capture_lora_gradients(model: Any) -> dict[str, Any]:
    import torch

    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name:
            result[name] = (
                torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
                if parameter.grad is None
                else parameter.grad.detach().float().cpu().clone()
            )
    return result


def gradient_delta(after: dict[str, Any], before: dict[str, Any], divisor: float = 1.0) -> dict[str, Any]:
    return {name: (after[name] - before[name]) / divisor for name in after}


def gradient_norm(gradients: dict[str, Any]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in gradients.values()))


def gradient_cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
    names = sorted(set(left) & set(right))
    dot = sum(float((left[name] * right[name]).sum()) for name in names)
    left_norm = math.sqrt(sum(float(left[name].square().sum()) for name in names))
    right_norm = math.sqrt(sum(float(right[name].square().sum()) for name in names))
    return dot / (left_norm * right_norm) if left_norm and right_norm else math.nan


def parameter_group(name: str) -> str:
    if any(token in name for token in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(token in name for token in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    return "other"


def gradient_energy(gradients: dict[str, Any]) -> dict[str, float]:
    energy: dict[str, float] = defaultdict(float)
    for name, value in gradients.items():
        energy[parameter_group(name)] += float(value.square().sum())
    total = sum(energy.values())
    return {key: value / total if total else 0.0 for key, value in sorted(energy.items())}


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
