"""Deterministic data and numerical helpers for the TRIM-CS0 preflight."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def stable_hash(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def transition_split(response: str, marker: str) -> tuple[str, str]:
    index = response.lower().find(marker.lower())
    if index < 0:
        raise ValueError(f"transition marker not found: {marker}")
    split_index = len(response[:index].rstrip())
    prefix = response[:split_index]
    suffix = response[split_index:]
    if not prefix.strip() or not suffix.strip():
        raise ValueError("transition split produced an empty side")
    return prefix, suffix


def _target_by_sample(rows: list[dict[str, Any]], marker: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        response = str(row["response"])
        if not response.lower().startswith("i can't") or marker.lower() not in response.lower():
            continue
        sample_id = str(row["sample_id"])
        if sample_id in selected:
            raise ValueError(f"multiple eligible targets for {sample_id}")
        selected[sample_id] = row
    return selected


def build_pair_rows(
    targets: list[dict[str, Any]], category_quotas: dict[str, int], marker: str
) -> list[dict[str, Any]]:
    selected = _target_by_sample(targets, marker)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in selected.values():
        by_category.setdefault(str(row["category"]), []).append(row)
    for rows in by_category.values():
        rows.sort(key=lambda row: stable_hash(str(row["sample_id"])))

    all_rows = sorted(selected.values(), key=lambda row: stable_hash(str(row["sample_id"])))
    pair_specs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for category, quota in category_quotas.items():
        candidates = by_category.get(category, [])
        if len(candidates) < 2 or quota <= 0 or quota > len(candidates):
            raise ValueError(f"invalid pair quota for {category}: {quota}/{len(candidates)}")
        for index in range(quota):
            left = candidates[index]
            right = candidates[(index + 1) % len(candidates)]
            pair_specs.append((category, left, right))

    output: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str]] = set()
    for pair_index, (category, left, right) in enumerate(pair_specs, start=1):
        key = tuple(sorted((str(left["sample_id"]), str(right["sample_id"]))))
        if key in pair_keys:
            raise ValueError(f"duplicate undirected pair: {key}")
        pair_keys.add(key)
        pair_id = f"trim_pair_{pair_index:03d}"
        for direction, donor, recipient in (("a_to_b", left, right), ("b_to_a", right, left)):
            cross_candidates = [
                row
                for row in all_rows
                if str(row["category"]) != category
                and str(row["sample_id"])
                not in {str(donor["sample_id"]), str(recipient["sample_id"])}
            ]
            cross = min(
                cross_candidates,
                key=lambda row: stable_hash(pair_id, direction, str(row["sample_id"])),
            )
            forced_prefix, target_suffix = transition_split(str(donor["response"]), marker)
            output.append(
                {
                    "instance_id": f"{pair_id}_{direction}",
                    "pair_id": pair_id,
                    "direction": direction,
                    "category": category,
                    "correct_donor_sample_id": donor["sample_id"],
                    "recipient_sample_id": recipient["sample_id"],
                    "paired_wrong_sample_id": recipient["sample_id"],
                    "cross_wrong_sample_id": cross["sample_id"],
                    "correct_donor_prompt": donor["prompt"],
                    "recipient_prompt": recipient["prompt"],
                    "paired_wrong_prompt": recipient["prompt"],
                    "cross_wrong_prompt": cross["prompt"],
                    "forced_prefix": forced_prefix,
                    "target_suffix": target_suffix,
                    "transition_marker": marker,
                    "correct_donor_prompt_sha256": stable_hash(str(donor["prompt"])),
                    "recipient_prompt_sha256": stable_hash(str(recipient["prompt"])),
                    "forced_prefix_sha256": stable_hash(forced_prefix),
                    "target_suffix_sha256": stable_hash(target_suffix),
                }
            )
    if len(output) != 2 * sum(category_quotas.values()):
        raise AssertionError("directional pair count mismatch")
    return output


def patch_layer_output(output: Any, positions: list[int], replacement: Any) -> Any:
    hidden = output[0] if isinstance(output, tuple) else output
    if replacement.shape != hidden[0, positions, :].shape:
        raise ValueError("replacement state shape does not match patch window")
    patched = hidden.clone()
    patched[0, positions, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def paired_bootstrap_mean_ci(
    values_by_pair: dict[str, list[float]], seed: int, samples: int = 10000
) -> dict[str, float | int]:
    if not values_by_pair:
        raise ValueError("bootstrap requires at least one pair")
    pair_means = [sum(values) / len(values) for _, values in sorted(values_by_pair.items())]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [pair_means[rng.randrange(len(pair_means))] for _ in pair_means]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    lower = draws[int(0.025 * samples)]
    upper = draws[min(samples - 1, int(0.975 * samples))]
    return {
        "pairs": len(pair_means),
        "mean": sum(pair_means) / len(pair_means),
        "bootstrap_samples": samples,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }
