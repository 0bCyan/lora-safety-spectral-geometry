"""Audit and freeze source-labeled same-prompt pairs for CSPR P1."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger", required=True)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
    return path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def stable_rank(seed: int, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


def token_shingles(value: str, maximum_width: int) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    features: set[tuple[str, ...]] = set()
    for width in range(1, min(maximum_width, len(tokens)) + 1):
        for index in range(len(tokens) - width + 1):
            features.add((str(width), *tokens[index : index + width]))
    return frozenset(features)


def longest_shingles(value: str, maximum_width: int) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    if not tokens:
        return frozenset()
    width = min(maximum_width, len(tokens))
    return frozenset(
        (str(width), *tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def lexical_similarity(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return max(intersection / len(left | right), intersection / min(len(left), len(right)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    for label, artifact in protocol["artifacts"].items():
        path = resolve_repo_path(artifact["path"])
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact hash mismatch: {label}")
    for reference in protocol["exclusion_references"]:
        path = resolve_repo_path(reference["path"])
        if sha256(path) != reference["sha256"]:
            raise ValueError(f"frozen reference hash mismatch: {reference['name']}")


def load_reference_state(
    protocol: dict[str, Any]
) -> tuple[set[str], set[str], list[str], dict[str, int]]:
    exact_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    prompt_texts: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for spec in protocol["exclusion_references"]:
        path = resolve_repo_path(spec["path"])
        mode = spec["mode"]
        if mode == "jsonl_prompt":
            rows = read_jsonl(path)
            field = spec["field"]
            for row in rows:
                prompt = row.get(field)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"missing prompt in {spec['name']}")
                prompt = prompt.strip()
                exact_hashes.add(sha256_text(prompt))
                normalized = normalize_text(prompt)
                normalized_hashes.add(sha256_text(normalized))
                prompt_texts.setdefault(sha256_text(normalized), prompt)
        elif mode in {"csv_normalized_hash", "csv_exact_hash"}:
            rows = read_csv(path)
            field = spec["field"]
            for row in rows:
                digest = row.get(field, "").strip().casefold()
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(f"invalid hash in {spec['name']}")
                target = normalized_hashes if mode == "csv_normalized_hash" else exact_hashes
                target.add(digest)
        else:
            raise ValueError(f"unsupported reference mode: {mode}")
        expected = int(spec["expected_rows"])
        if len(rows) != expected:
            raise ValueError(f"unexpected reference row count: {spec['name']}")
        row_counts[spec["name"]] = len(rows)
    return exact_hashes, normalized_hashes, list(prompt_texts.values()), row_counts


def load_beaver_groups(
    source_path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_cfg = protocol["source"]
    expected_categories = set(source_cfg["expected_categories"])
    groups: dict[str, dict[str, Any]] = {}
    row_count = 0
    with gzip.open(source_path, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            source = json.loads(line)
            if not {"prompt", "response", "is_safe", "category"} <= set(source):
                raise ValueError(f"source schema mismatch at row {source_index}")
            prompt = source["prompt"]
            response = source["response"]
            categories = source["category"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"empty prompt at row {source_index}")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"empty response at row {source_index}")
            if not isinstance(source["is_safe"], bool):
                raise ValueError(f"non-boolean safety label at row {source_index}")
            if not isinstance(categories, dict) or set(categories) != expected_categories:
                raise ValueError(f"category schema mismatch at row {source_index}")
            if any(type(value) is not bool for value in categories.values()):
                raise ValueError(f"non-boolean category label at row {source_index}")

            prompt = prompt.strip()
            response = response.strip()
            normalized_prompt = normalize_text(prompt)
            normalized_prompt_hash = sha256_text(normalized_prompt)
            group = groups.setdefault(
                normalized_prompt_hash,
                {
                    "prompt": prompt,
                    "prompt_exact_sha256": sha256_text(prompt),
                    "prompt_normalized_sha256": normalized_prompt_hash,
                    "safe": {},
                    "unsafe": {},
                    "unsafe_categories": set(),
                },
            )
            normalized_response_hash = sha256_text(normalize_text(response))
            response_record = {
                "source_index": source_index,
                "response_sha256": sha256_text(response),
                "response_normalized_sha256": normalized_response_hash,
            }
            target = group["safe"] if source["is_safe"] else group["unsafe"]
            target.setdefault(normalized_response_hash, response_record)
            if not source["is_safe"]:
                group["unsafe_categories"].update(
                    category for category, present in categories.items() if present
                )

    if row_count != int(source_cfg["expected_rows"]):
        raise ValueError(f"unexpected source row count: {row_count}")
    raw_mixed = [
        group
        for group in groups.values()
        if group["safe"] and group["unsafe"] and group["unsafe_categories"]
    ]
    raw_safe_response_count = sum(len(group["safe"]) for group in raw_mixed)
    raw_unsafe_response_count = sum(len(group["unsafe"]) for group in raw_mixed)
    candidates = []
    label_conflict_responses = 0
    for group in groups.values():
        conflicts = set(group["safe"]) & set(group["unsafe"])
        label_conflict_responses += len(conflicts)
        group["unsafe"] = {
            key: value for key, value in group["unsafe"].items() if key not in conflicts
        }
        if group["safe"] and group["unsafe"] and group["unsafe_categories"]:
            candidates.append(group)
    inventory = {
        "rows": row_count,
        "unique_normalized_prompts": len(groups),
        "mixed_safe_unsafe_prompts": len(raw_mixed),
        "safe_responses_on_mixed_prompts": raw_safe_response_count,
        "unsafe_responses_on_mixed_prompts": raw_unsafe_response_count,
        "cross_label_identical_responses_removed": label_conflict_responses,
        "pair_eligible_prompts_after_response_conflict_removal": len(candidates),
    }
    return candidates, inventory


def build_feature_index(
    texts: Iterable[str], maximum_width: int
) -> tuple[list[frozenset[tuple[str, ...]]], dict[tuple[str, ...], set[int]], dict[tuple[str, ...], set[int]]]:
    features: list[frozenset[tuple[str, ...]]] = []
    all_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    longest_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for text in texts:
        position = len(features)
        values = token_shingles(text, maximum_width)
        features.append(values)
        for value in values:
            all_index[value].add(position)
        for value in longest_shingles(text, maximum_width):
            longest_index[value].add(position)
    return features, all_index, longest_index


def maximum_indexed_similarity(
    text: str,
    features: list[frozenset[tuple[str, ...]]],
    all_index: dict[tuple[str, ...], set[int]],
    longest_index: dict[tuple[str, ...], set[int]],
    maximum_width: int,
) -> float:
    values = token_shingles(text, maximum_width)
    positions: set[int] = set()
    for value in longest_shingles(text, maximum_width):
        positions.update(all_index.get(value, set()))
    for value in values:
        positions.update(longest_index.get(value, set()))
    return max((lexical_similarity(values, features[p]) for p in positions), default=0.0)


def filter_candidates(
    candidates: list[dict[str, Any]],
    exact_hashes: set[str],
    normalized_hashes: set[str],
    reference_texts: list[str],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = protocol["freshness_filter"]
    maximum_width = int(cfg["maximum_shingle_width"])
    threshold = float(cfg["similarity_fail_threshold"])
    reference_features, reference_index, reference_longest = build_feature_index(
        reference_texts, maximum_width
    )
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    maximum_similarity = 0.0
    for row in candidates:
        if row["prompt_exact_sha256"] in exact_hashes:
            reasons["exact_prior_overlap"] += 1
            continue
        if row["prompt_normalized_sha256"] in normalized_hashes:
            reasons["normalized_prior_overlap"] += 1
            continue
        similarity = maximum_indexed_similarity(
            row["prompt"],
            reference_features,
            reference_index,
            reference_longest,
            maximum_width,
        )
        maximum_similarity = max(maximum_similarity, similarity)
        if similarity >= threshold:
            reasons["near_prior_overlap"] += 1
            continue
        kept.append(row)
    return kept, {
        "before": len(candidates),
        "after": len(kept),
        "excluded": sum(reasons.values()),
        "exclusion_counts": dict(sorted(reasons.items())),
        "maximum_similarity_among_nonexact_comparisons": maximum_similarity,
        "threshold": threshold,
    }


def category_targets(total: int, categories: list[str], seed: int) -> dict[str, int]:
    base, remainder = divmod(total, len(categories))
    ranked = sorted(categories, key=lambda value: stable_rank(seed, "category-remainder", value))
    return {category: base + int(category in ranked[:remainder]) for category in categories}


def select_pairs(
    candidates: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    cfg = protocol["selection"]
    seed = int(cfg["seed"])
    categories = list(protocol["source"]["expected_categories"])
    targets = category_targets(int(cfg["prompt_count"]), categories, seed)
    supply = {
        category: sum(category in row["unsafe_categories"] for row in candidates)
        for category in categories
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for category in sorted(categories, key=lambda value: (supply[value], value)):
        pool = [
            row
            for row in candidates
            if category in row["unsafe_categories"]
            and row["prompt_normalized_sha256"] not in used
        ]
        pool.sort(
            key=lambda row: stable_rank(
                seed, "prompt", category, row["prompt_normalized_sha256"]
            )
        )
        if len(pool) < targets[category]:
            raise ValueError(f"insufficient prompt supply for {category}")
        for row in pool[: targets[category]]:
            safe = min(
                row["safe"].values(),
                key=lambda value: stable_rank(
                    seed,
                    "safe-response",
                    row["prompt_normalized_sha256"],
                    value["response_normalized_sha256"],
                ),
            )
            unsafe = min(
                row["unsafe"].values(),
                key=lambda value: stable_rank(
                    seed,
                    "unsafe-response",
                    row["prompt_normalized_sha256"],
                    value["response_normalized_sha256"],
                ),
            )
            selected.append(
                {
                    "pair_id": "cspr_pair_"
                    + stable_rank(seed, "pair", row["prompt_normalized_sha256"])[:20],
                    "assigned_category": category,
                    "prompt_normalized_sha256": row["prompt_normalized_sha256"],
                    "prompt_exact_sha256": row["prompt_exact_sha256"],
                    "chosen_source_index": safe["source_index"],
                    "chosen_response_sha256": safe["response_sha256"],
                    "chosen_response_normalized_sha256": safe["response_normalized_sha256"],
                    "rejected_source_index": unsafe["source_index"],
                    "rejected_response_sha256": unsafe["response_sha256"],
                    "rejected_response_normalized_sha256": unsafe[
                        "response_normalized_sha256"
                    ],
                }
            )
            used.add(row["prompt_normalized_sha256"])

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_category[row["assigned_category"]].append(row)
    for category, rows in by_category.items():
        rows.sort(key=lambda row: stable_rank(seed, "split", row["pair_id"]))
        calibration = int(cfg["calibration_per_category"])
        development = int(cfg["development_per_category"])
        for index, row in enumerate(rows):
            if index < calibration:
                row["prospective_split"] = "calibration"
            elif index < calibration + development:
                row["prospective_split"] = "mechanism_dev"
            else:
                row["prospective_split"] = "train_candidate"
    return sorted(selected, key=lambda row: row["pair_id"])


def write_frozen(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    output_path = resolve_repo_path(args.output)
    ledger_path = resolve_repo_path(args.ledger)
    protocol = read_json(protocol_path)
    verify_frozen_inputs(protocol)

    source_path = resolve_repo_path(protocol["source"]["path"])
    if sha256(source_path) != protocol["source"]["sha256"]:
        raise ValueError("frozen BeaverTails source hash mismatch")
    exact_hashes, normalized_hashes, reference_texts, reference_counts = load_reference_state(
        protocol
    )
    candidates, inventory = load_beaver_groups(source_path, protocol)
    filtered, freshness = filter_candidates(
        candidates,
        exact_hashes,
        normalized_hashes,
        reference_texts,
        protocol,
    )
    selected = select_pairs(filtered, protocol)

    source_cfg = protocol["source"]
    gate_cfg = protocol["gates"]
    category_supply = dict(
        sorted(
            (
                category,
                sum(category in row["unsafe_categories"] for row in filtered),
            )
            for category in source_cfg["expected_categories"]
        )
    )
    selected_categories = Counter(row["assigned_category"] for row in selected)
    splits = Counter(row["prospective_split"] for row in selected)
    checks = {
        "source_row_count": inventory["rows"] == int(source_cfg["expected_rows"]),
        "source_mixed_prompt_count": inventory["mixed_safe_unsafe_prompts"]
        == int(source_cfg["expected_mixed_prompts"]),
        "source_safe_candidate_count": inventory["safe_responses_on_mixed_prompts"]
        == int(source_cfg["expected_safe_responses_on_mixed_prompts"]),
        "eligible_prompt_minimum": len(filtered) >= int(gate_cfg["minimum_eligible_prompts"]),
        "category_supply_minimum": min(category_supply.values())
        >= int(gate_cfg["minimum_eligible_prompts_per_category"]),
        "selected_prompt_count": len(selected) == int(protocol["selection"]["prompt_count"]),
        "selected_prompts_unique": len(
            {row["prompt_normalized_sha256"] for row in selected}
        )
        == len(selected),
        "selected_responses_distinct": all(
            row["chosen_response_normalized_sha256"]
            != row["rejected_response_normalized_sha256"]
            for row in selected
        ),
        "split_counts": dict(splits) == protocol["selection"]["expected_split_counts"],
        "category_targets": dict(selected_categories)
        == category_targets(
            int(protocol["selection"]["prompt_count"]),
            list(source_cfg["expected_categories"]),
            int(protocol["selection"]["seed"]),
        ),
    }
    passed = all(checks.values())
    result = {
        "name": "step3_cspr_p1_public_source_feasibility_audit_v1",
        "protocol": args.protocol,
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "dataset": source_cfg["dataset"],
            "revision": source_cfg["revision"],
            "sha256": source_cfg["sha256"],
        },
        "reference_rows": reference_counts,
        "source_inventory": inventory,
        "freshness_filter": freshness,
        "eligible_category_supply": category_supply,
        "selected_category_counts": dict(sorted(selected_categories.items())),
        "prospective_split_counts": dict(sorted(splits.items())),
        "selected_pair_ledger": args.ledger,
        "selected_pair_ledger_sha256": sha256_bytes(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode("utf-8")
        ),
        "checks": checks,
        "passed": passed,
        "route": "PASS_CSPR_P1_SOURCE_FEASIBILITY"
        if passed
        else "STOP_CSPR_P1_DATA_INSUFFICIENT",
        "claim_boundary": protocol["claim_boundary"],
        "source_labels_accepted_as_final": False,
        "model_gradient_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_content_opened": False,
    }
    ledger_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    write_frozen(ledger_path, ledger_text)
    write_frozen(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "route": result["route"], "checks": checks}))
    if not passed:
        raise RuntimeError(result["route"])


if __name__ == "__main__":
    main()
