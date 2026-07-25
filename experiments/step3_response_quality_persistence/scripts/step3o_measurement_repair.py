#!/usr/bin/env python3
"""Freeze and analyze the Step3O external measurement-repair protocol."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/step3o_measurement_repair_registry.json"
BASE = ROOT / "results/step3o_measurement_repair"
PACKETS = BASE / "packets"
REVIEWS = BASE / "reviews"
REPORTS = BASE / "reports"
DIAGNOSTICS = BASE / "diagnostics"

SOURCE_INVENTORY = DIAGNOSTICS / "step3o_source_inventory.json"
CONTAMINATION_AUDIT = DIAGNOSTICS / "step3o_contamination_audit.json"
O0_MANIFEST = REPORTS / "step3o_o0_protocol_manifest.json"
O0_REPORT = REPORTS / "Step3O_O0_protocol_freeze_report.md"

ACTION_FIELDS = ["safe_route", "executable_step", "prompt_binding"]
BOOL_VALUES = {"yes", "no"}
DNA_MODELS = [
    "GPT4",
    "ChatGPT",
    "Claude",
    "ChatGLM2",
    "llama2-7b-chat",
    "vicuna-7b",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_json_frozen(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to mutate frozen JSON: {repo_path(path)}")
    path.write_text(content, encoding="utf-8", newline="\n")


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    require_equal_if_exists: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = jsonl_text(rows)
    if require_equal_if_exists and path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise ValueError(f"refusing to mutate frozen JSONL: {repo_path(path)}")
    path.write_text(content, encoding="utf-8", newline="\n")


def csv_text(rows: Iterable[dict[str, Any]], fields: list[str]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue()


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
    *,
    require_equal_if_exists: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = csv_text(rows, fields)
    if require_equal_if_exists and path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise ValueError(f"refusing to mutate frozen CSV: {repo_path(path)}")
    path.write_text(content, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(seed: int | str, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def require_git_frozen(paths: Iterable[Path]) -> None:
    for path in paths:
        relative = repo_path(path)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=REPO_ROOT,
            check=False,
        )
        if tracked.returncode != 0 or unchanged.returncode != 0:
            raise ValueError(f"required committed freeze is absent or dirty: {relative}")


def registry() -> dict[str, Any]:
    return read_json(REGISTRY)


def validate_boundaries(reg: dict[str, Any]) -> None:
    boundaries = reg["boundaries"]
    required_false = [
        "old_87_rows_used_for_anchor_writing",
        "old_37_prompts_released",
        "old_n0_labels_reopened",
        "core100_content_read",
        "heldout_content_read",
        "proxy_dev_outcomes_read",
        "training_authorized",
        "pcsr_r2_authorized",
    ]
    bad = [field for field in required_false if boundaries.get(field) is not False]
    if bad or int(boundaries.get("optimizer_steps", -1)) != 0:
        raise ValueError(f"Step3O boundary flags are open: {bad}")


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def token_shingles(
    value: str, maximum_width: int, minimum_width: int = 1
) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    if not tokens:
        return frozenset()
    features: set[tuple[str, ...]] = set()
    for width in range(minimum_width, min(maximum_width, len(tokens)) + 1):
        for index in range(len(tokens) - width + 1):
            features.add((str(width), *tokens[index : index + width]))
    return frozenset(features)


def longest_token_shingles(value: str, maximum_width: int) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    if not tokens:
        return frozenset()
    width = min(maximum_width, len(tokens))
    return frozenset(
        (str(width), *tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap_coefficient(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def lexical_similarity(
    left: frozenset[Any], right: frozenset[Any]
) -> tuple[float, float, float]:
    jac = jaccard(left, right)
    overlap = overlap_coefficient(left, right)
    return jac, overlap, max(jac, overlap)


def source_spec(name: str) -> dict[str, Any]:
    return registry()["sources"][name]


def check_raw_source(name: str) -> Path:
    spec = source_spec(name)
    path = resolve_repo_path(spec["raw_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"missing ignored raw source {repo_path(path)}; download frozen URL first"
        )
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(
            f"raw source hash mismatch for {name}: expected {spec['sha256']}, got {actual}"
        )
    return path


def load_dna_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = check_raw_source("development")
    reg = registry()
    allowed_actions = {int(value) for value in reg["development"]["source_action_values"]}
    required_harmful = str(reg["development"]["source_harmful_value_required"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(source_spec("development")["expected_prompt_rows"])
    if len(rows) != expected:
        raise ValueError(f"unexpected Do-Not-Answer row count: {len(rows)} != {expected}")

    candidates: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["question"].strip()
        if not prompt:
            continue
        for model in DNA_MODELS:
            if row[f"{model}_harmful"] != required_harmful:
                continue
            action = int(row[f"{model}_action"])
            if action not in allowed_actions:
                continue
            response = row[f"{model}_response"].strip()
            if not response:
                continue
            candidates.append(
                {
                    "source_dataset": "LibrAI/do-not-answer",
                    "source_prompt_id": str(row["id"]),
                    "source_response_id": f"{row['id']}:{model}",
                    "source_model": model,
                    "source_action": action,
                    "source_harmful": int(row[f"{model}_harmful"]),
                    "risk_area": row["risk_area"],
                    "prompt": prompt,
                    "response": response,
                    "prompt_sha256": sha256_text(normalize_text(prompt)),
                    "response_sha256": sha256_text(normalize_text(response)),
                }
            )
    inventory = {
        "dataset_rows": len(rows),
        "eligible_response_rows": len(candidates),
        "eligible_unique_prompts": len({row["source_prompt_id"] for row in candidates}),
        "eligible_by_source_action": dict(
            sorted(Counter(str(row["source_action"]) for row in candidates).items())
        ),
        "risk_area_prompt_counts": dict(
            sorted(Counter(row["risk_area"] for row in rows).items())
        ),
    }
    return candidates, inventory


def select_dna_development(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = registry()["development"]
    seed = int(cfg["packet_seed"])
    quota = int(cfg["primary_rows_per_source_action"])
    actions = [int(value) for value in cfg["source_action_values"]]
    by_action = {action: [row for row in candidates if row["source_action"] == action] for action in actions}
    action_order = sorted(
        actions,
        key=lambda action: (
            len({row["source_prompt_id"] for row in by_action[action]}),
            action,
        ),
    )

    selected: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    global_risk = Counter()
    global_model = Counter()
    for action in action_order:
        action_risk = Counter()
        action_model = Counter()
        for _ in range(quota):
            pool = [row for row in by_action[action] if row["source_prompt_id"] not in used_prompts]
            if not pool:
                raise ValueError(f"insufficient unique development prompts for source action {action}")
            chosen = min(
                pool,
                key=lambda row: (
                    action_risk[row["risk_area"]],
                    action_model[row["source_model"]],
                    global_risk[row["risk_area"]],
                    global_model[row["source_model"]],
                    stable_rank(seed, "dna", str(action), row["source_response_id"]),
                ),
            )
            selected.append(chosen)
            used_prompts.add(chosen["source_prompt_id"])
            action_risk[chosen["risk_area"]] += 1
            action_model[chosen["source_model"]] += 1
            global_risk[chosen["risk_area"]] += 1
            global_model[chosen["source_model"]] += 1

    expected = int(cfg["primary_rows"])
    if len(selected) != expected or len(used_prompts) != expected:
        raise ValueError("development selection is not one row per prompt")
    if Counter(row["source_action"] for row in selected) != Counter({action: quota for action in actions}):
        raise ValueError("development source-action quotas were not met")
    return sorted(selected, key=lambda row: stable_rank(seed, "selected", row["source_response_id"]))


def load_beaver_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = check_raw_source("transport")
    groups: dict[str, dict[str, Any]] = {}
    row_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            row_count += 1
            prompt = source["prompt"].strip()
            response = source["response"].strip()
            norm = normalize_text(prompt)
            group = groups.setdefault(
                norm,
                {
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(norm),
                    "safe_responses": {},
                    "unsafe_response_count": 0,
                    "unsafe_categories": set(),
                },
            )
            if bool(source["is_safe"]):
                if response:
                    group["safe_responses"].setdefault(
                        sha256_text(normalize_text(response)), response
                    )
            else:
                group["unsafe_response_count"] += 1
                group["unsafe_categories"].update(
                    key for key, value in source["category"].items() if value
                )
    expected = int(source_spec("transport")["expected_rows"])
    if row_count != expected:
        raise ValueError(f"unexpected BeaverTails row count: {row_count} != {expected}")

    candidates: list[dict[str, Any]] = []
    for group in groups.values():
        if not group["safe_responses"] or not group["unsafe_response_count"]:
            continue
        if not group["unsafe_categories"]:
            continue
        candidates.append(group)
    inventory = {
        "dataset_rows": row_count,
        "unique_normalized_prompts": len(groups),
        "mixed_safe_unsafe_prompt_candidates": len(candidates),
        "safe_response_candidates_on_mixed_prompts": sum(
            len(row["safe_responses"]) for row in candidates
        ),
        "mixed_prompt_category_coverage": dict(
            sorted(
                Counter(
                    category
                    for row in candidates
                    for category in row["unsafe_categories"]
                ).items()
            )
        ),
    }
    return candidates, inventory


def category_targets(total: int, categories: list[str], seed: int, label: str) -> dict[str, int]:
    base, remainder = divmod(total, len(categories))
    ranked = sorted(categories, key=lambda category: stable_rank(seed, label, category))
    return {
        category: base + int(category in set(ranked[:remainder]))
        for category in categories
    }


def choose_length_stratified_response(
    group: dict[str, Any], seed: int
) -> tuple[str, str, str]:
    ranked = sorted(
        group["safe_responses"].items(),
        key=lambda item: (len(item[1]), stable_rank(seed, "response", item[0])),
    )
    selector = int(stable_rank(seed, "length-stratum", group["prompt_sha256"]), 16) % 3
    stratum = ["short", "middle", "long"][selector]
    index = 0 if selector == 0 else (len(ranked) - 1) // 2 if selector == 1 else len(ranked) - 1
    response_sha, response = ranked[index]
    return response_sha, response, stratum


def select_beaver_safety(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = registry()["transport_safety"]
    seed = int(cfg["packet_seed"])
    total = int(cfg["candidate_primary_rows"])
    categories = sorted({category for row in candidates for category in row["unsafe_categories"]})
    targets = category_targets(total, categories, seed, "safety-category-target")
    category_supply = {
        category: sum(category in row["unsafe_categories"] for row in candidates)
        for category in categories
    }
    category_order = sorted(categories, key=lambda category: (category_supply[category], category))
    selected: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    for category in category_order:
        pool = [
            row
            for row in candidates
            if category in row["unsafe_categories"] and row["prompt_sha256"] not in used_prompts
        ]
        ranked = sorted(
            pool,
            key=lambda row: stable_rank(seed, "beaver", category, row["prompt_sha256"]),
        )
        needed = targets[category]
        if len(ranked) < needed:
            raise ValueError(f"insufficient BeaverTails prompts for {category}: {len(ranked)} < {needed}")
        for group in ranked[:needed]:
            response_sha, response, length_stratum = choose_length_stratified_response(group, seed)
            selected.append(
                {
                    "source_dataset": "PKU-Alignment/BeaverTails",
                    "source_prompt_id": group["prompt_sha256"],
                    "source_response_id": response_sha,
                    "assigned_category": category,
                    "source_categories": sorted(group["unsafe_categories"]),
                    "source_is_safe": True,
                    "response_length_stratum": length_stratum,
                    "prompt": group["prompt"],
                    "response": response,
                    "prompt_sha256": group["prompt_sha256"],
                    "response_sha256": response_sha,
                }
            )
            used_prompts.add(group["prompt_sha256"])
    if len(selected) != total or len(used_prompts) != total:
        raise ValueError("transport-safety selection is not one row per prompt")
    if Counter(row["assigned_category"] for row in selected) != Counter(targets):
        raise ValueError("transport-safety category targets were not met")
    return sorted(selected, key=lambda row: stable_rank(seed, "selected", row["source_prompt_id"]))


def forbidden_prompts() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value in registry()["permitted_contamination_audit_inputs"]:
        path = resolve_repo_path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        for index, row in enumerate(read_jsonl(path)):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"missing prompt in contamination input {value}:{index}")
            opaque = str(row.get("sample_id") or row.get("opaque_review_id") or index)
            rows.append(
                {
                    "source": value,
                    "opaque_id": opaque,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(normalize_text(prompt)),
                }
            )
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        unique.setdefault(row["prompt_sha256"], row)
    return list(unique.values())


def compare_prompt_sets(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    cfg = registry()["near_duplicate_audit"]
    minimum_width = int(cfg["token_ngram_min"])
    maximum_width = int(cfg["token_ngram_max"])
    threshold = float(cfg["similarity_fail_threshold"])
    right_exact = defaultdict(list)
    right_features: list[tuple[dict[str, Any], frozenset[Any]]] = []
    right_index: dict[Any, set[int]] = defaultdict(set)
    right_longest_index: dict[Any, set[int]] = defaultdict(set)
    for row in right:
        right_exact[row["prompt_sha256"]].append(row)
        features = token_shingles(row["prompt"], maximum_width, minimum_width)
        right_features.append((row, features))
        right_position = len(right_features) - 1
        for feature in features:
            right_index[feature].add(right_position)
        for feature in longest_token_shingles(row["prompt"], maximum_width):
            right_longest_index[feature].add(right_position)

    exact_hits: list[dict[str, str]] = []
    near_hits: list[dict[str, Any]] = []
    maximum = 0.0
    maximum_pair: dict[str, Any] | None = None
    for left_row in left:
        for right_row in right_exact.get(left_row["prompt_sha256"], []):
            exact_hits.append(
                {
                    "left_id": str(left_row["source_prompt_id"]),
                    "right_id": str(right_row.get("opaque_id") or right_row.get("source_prompt_id")),
                    "right_source": str(right_row.get("source", right_row.get("source_dataset", ""))),
                }
            )
        left_features = token_shingles(
            left_row["prompt"], maximum_width, minimum_width
        )
        candidate_positions: set[int] = set()
        for feature in longest_token_shingles(left_row["prompt"], maximum_width):
            candidate_positions.update(right_index.get(feature, set()))
        for feature in left_features:
            candidate_positions.update(right_longest_index.get(feature, set()))
        for position in candidate_positions:
            right_row, features = right_features[position]
            jac, overlap, score = lexical_similarity(left_features, features)
            if score > maximum:
                maximum = score
                maximum_pair = {
                    "left_id": str(left_row["source_prompt_id"]),
                    "right_id": str(right_row.get("opaque_id") or right_row.get("source_prompt_id")),
                    "right_source": str(right_row.get("source", right_row.get("source_dataset", ""))),
                    "jaccard": jac,
                    "overlap_coefficient": overlap,
                    "maximum_similarity": score,
                }
            if score >= threshold and left_row["prompt_sha256"] != right_row["prompt_sha256"]:
                near_hits.append(
                    {
                        "left_id": str(left_row["source_prompt_id"]),
                        "right_id": str(right_row.get("opaque_id") or right_row.get("source_prompt_id")),
                        "right_source": str(right_row.get("source", right_row.get("source_dataset", ""))),
                        "jaccard": jac,
                        "overlap_coefficient": overlap,
                        "maximum_similarity": score,
                    }
                )
    return {
        "label": label,
        "left_rows": len(left),
        "right_rows": len(right),
        "exact_overlap_count": len(exact_hits),
        "near_overlap_count": len(near_hits),
        "similarity_fail_threshold": threshold,
        "maximum_similarity": maximum,
        "maximum_pair": maximum_pair,
        "exact_hits": exact_hits,
        "near_hits": near_hits,
        "pass": not exact_hits and not near_hits,
    }


def exclude_contaminated_candidates(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = registry()["near_duplicate_audit"]
    minimum_width = int(cfg["token_ngram_min"])
    maximum_width = int(cfg["token_ngram_max"])
    threshold = float(cfg["similarity_fail_threshold"])
    reference_exact = {row["prompt_sha256"] for row in references}
    reference_features = [
        token_shingles(row["prompt"], maximum_width, minimum_width)
        for row in references
    ]
    reference_index: dict[Any, set[int]] = defaultdict(set)
    reference_longest_index: dict[Any, set[int]] = defaultdict(set)
    for position, features in enumerate(reference_features):
        for feature in features:
            reference_index[feature].add(position)
        for feature in longest_token_shingles(
            references[position]["prompt"], maximum_width
        ):
            reference_longest_index[feature].add(position)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in candidates:
        source_id = str(row.get("source_prompt_id") or row["prompt_sha256"])
        if row["prompt_sha256"] in reference_exact:
            excluded.append(
                {
                    "source_prompt_id": source_id,
                    "reason": "exact_normalized_overlap",
                    "maximum_similarity": 1.0,
                }
            )
            continue
        features = token_shingles(row["prompt"], maximum_width, minimum_width)
        candidate_positions: set[int] = set()
        for feature in longest_token_shingles(row["prompt"], maximum_width):
            candidate_positions.update(reference_index.get(feature, set()))
        for feature in features:
            candidate_positions.update(reference_longest_index.get(feature, set()))
        maximum = max(
            (
                lexical_similarity(features, reference_features[position])[2]
                for position in candidate_positions
            ),
            default=0.0,
        )
        if maximum >= threshold:
            excluded.append(
                {
                    "source_prompt_id": source_id,
                    "reason": "near_duplicate",
                    "maximum_similarity": maximum,
                }
            )
            continue
        kept.append(row)
    return kept, {
        "label": label,
        "candidate_rows_before": len(candidates),
        "candidate_rows_after": len(kept),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "threshold": threshold,
        "performed_before_step3o_labels": True,
    }


def packet_paths(stage: str) -> dict[str, Path]:
    base = PACKETS / stage
    return {
        "packet": base / f"step3o_{stage}_packet.jsonl",
        "key": base / f"step3o_{stage}_hidden_key.csv",
        "duplicates": base / f"step3o_{stage}_duplicate_key.csv",
        "template_a": base / f"step3o_{stage}_reviewer_a_template.jsonl",
        "template_b": base / f"step3o_{stage}_reviewer_b_template.jsonl",
        "c_packet": base / f"step3o_{stage}_reviewer_c_packet.jsonl",
        "c_template": base / f"step3o_{stage}_reviewer_c_template.jsonl",
        "manifest": base / f"step3o_{stage}_packet_manifest.json",
        "c_manifest": base / f"step3o_{stage}_reviewer_c_packet_manifest.json",
    }


def action_template(opaque_id: str) -> dict[str, str]:
    return {
        "opaque_action_id": opaque_id,
        "safe_route": "",
        "executable_step": "",
        "prompt_binding": "",
        "short_reason": "",
    }


def safety_template(opaque_id: str) -> dict[str, str]:
    return {"opaque_safety_id": opaque_id, "clean_safe": "", "short_reason": ""}


def build_development_packet(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    cfg = registry()["development"]
    packet_seed = int(cfg["packet_seed"])
    duplicate_seed = int(cfg["duplicate_seed"])
    primary_by_source: dict[str, str] = {}
    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in selected:
        opaque_id = f"step3o_dev_{stable_rank(packet_seed, row['source_response_id'], 'primary')[:18]}"
        primary_by_source[row["source_response_id"]] = opaque_id
        packet.append(
            {"opaque_action_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        key.append(
            {
                "opaque_action_id": opaque_id,
                "source_dataset": row["source_dataset"],
                "source_prompt_id": row["source_prompt_id"],
                "source_response_id": row["source_response_id"],
                "source_model": row["source_model"],
                "source_action": row["source_action"],
                "risk_area": row["risk_area"],
                "is_hidden_duplicate": False,
                "duplicate_of_action_id": "",
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
    duplicate_sources = sorted(
        selected,
        key=lambda row: stable_rank(duplicate_seed, row["source_response_id"]),
    )[: int(cfg["hidden_duplicates"])]
    duplicate_map: list[dict[str, str]] = []
    for row in duplicate_sources:
        opaque_id = f"step3o_dev_{stable_rank(packet_seed, row['source_response_id'], 'duplicate')[:18]}"
        primary_id = primary_by_source[row["source_response_id"]]
        packet.append(
            {"opaque_action_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        key.append(
            {
                "opaque_action_id": opaque_id,
                "source_dataset": row["source_dataset"],
                "source_prompt_id": row["source_prompt_id"],
                "source_response_id": row["source_response_id"],
                "source_model": row["source_model"],
                "source_action": row["source_action"],
                "risk_area": row["risk_area"],
                "is_hidden_duplicate": True,
                "duplicate_of_action_id": primary_id,
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
        duplicate_map.append(
            {
                "primary_opaque_action_id": primary_id,
                "duplicate_opaque_action_id": opaque_id,
            }
        )
    random.Random(packet_seed).shuffle(packet)
    return packet, key, duplicate_map


def build_safety_packet(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    cfg = registry()["transport_safety"]
    packet_seed = int(cfg["packet_seed"])
    duplicate_seed = int(cfg["duplicate_seed"])
    primary_by_source: dict[str, str] = {}
    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in selected:
        opaque_id = f"step3o_safe_{stable_rank(packet_seed, row['source_prompt_id'], row['source_response_id'], 'primary')[:18]}"
        primary_by_source[row["source_prompt_id"]] = opaque_id
        packet.append(
            {"opaque_safety_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        key.append(
            {
                "opaque_safety_id": opaque_id,
                "source_dataset": row["source_dataset"],
                "source_prompt_id": row["source_prompt_id"],
                "source_response_id": row["source_response_id"],
                "assigned_category": row["assigned_category"],
                "source_categories": ";".join(row["source_categories"]),
                "response_length_stratum": row["response_length_stratum"],
                "source_is_safe": row["source_is_safe"],
                "is_hidden_duplicate": False,
                "duplicate_of_safety_id": "",
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
    duplicate_sources = sorted(
        selected,
        key=lambda row: stable_rank(duplicate_seed, row["source_prompt_id"], row["source_response_id"]),
    )[: int(cfg["hidden_duplicates"])]
    duplicate_map: list[dict[str, str]] = []
    for row in duplicate_sources:
        opaque_id = f"step3o_safe_{stable_rank(packet_seed, row['source_prompt_id'], row['source_response_id'], 'duplicate')[:18]}"
        primary_id = primary_by_source[row["source_prompt_id"]]
        packet.append(
            {"opaque_safety_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        key.append(
            {
                "opaque_safety_id": opaque_id,
                "source_dataset": row["source_dataset"],
                "source_prompt_id": row["source_prompt_id"],
                "source_response_id": row["source_response_id"],
                "assigned_category": row["assigned_category"],
                "source_categories": ";".join(row["source_categories"]),
                "response_length_stratum": row["response_length_stratum"],
                "source_is_safe": row["source_is_safe"],
                "is_hidden_duplicate": True,
                "duplicate_of_safety_id": primary_id,
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
        duplicate_map.append(
            {
                "primary_opaque_safety_id": primary_id,
                "duplicate_opaque_safety_id": opaque_id,
            }
        )
    random.Random(packet_seed).shuffle(packet)
    return packet, key, duplicate_map


DEV_KEY_FIELDS = [
    "opaque_action_id",
    "source_dataset",
    "source_prompt_id",
    "source_response_id",
    "source_model",
    "source_action",
    "risk_area",
    "is_hidden_duplicate",
    "duplicate_of_action_id",
    "prompt_sha256",
    "response_sha256",
]

SAFETY_KEY_FIELDS = [
    "opaque_safety_id",
    "source_dataset",
    "source_prompt_id",
    "source_response_id",
    "assigned_category",
    "source_categories",
    "response_length_stratum",
    "source_is_safe",
    "is_hidden_duplicate",
    "duplicate_of_safety_id",
    "prompt_sha256",
    "response_sha256",
]


def artifact(path: Path) -> dict[str, Any]:
    return {"path": repo_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def freeze_o0() -> dict[str, Any]:
    reg = registry()
    validate_boundaries(reg)
    forbidden = forbidden_prompts()
    dna_candidates, dna_inventory = load_dna_candidates()
    dna_candidates, dna_exclusions = exclude_contaminated_candidates(
        dna_candidates,
        forbidden,
        label="development_candidates_vs_permitted_prior_inputs",
    )
    development = select_dna_development(dna_candidates)
    beaver_candidates, beaver_inventory = load_beaver_candidates()
    for row in beaver_candidates:
        row["source_prompt_id"] = row["prompt_sha256"]
    beaver_references = [*forbidden, *development]
    beaver_candidates, beaver_exclusions = exclude_contaminated_candidates(
        beaver_candidates,
        beaver_references,
        label="transport_candidates_vs_prior_and_selected_development",
    )
    safety = select_beaver_safety(beaver_candidates)

    comparisons = [
        compare_prompt_sets(development, forbidden, label="development_vs_permitted_prior_inputs"),
        compare_prompt_sets(safety, forbidden, label="transport_vs_permitted_prior_inputs"),
        compare_prompt_sets(development, safety, label="development_vs_transport"),
    ]
    contamination = {
        "name": "step3o_contamination_audit_v1",
        "comparisons": comparisons,
        "pre_sampling_exclusions": [dna_exclusions, beaver_exclusions],
        "all_pass": all(item["pass"] for item in comparisons),
        "permitted_prior_prompt_count": len(forbidden),
        "core100_content_read": False,
        "heldout_content_read": False,
        "old_n0_labels_read": False,
        "old_n0_review_reasons_read": False,
    }
    if not contamination["all_pass"]:
        raise ValueError("Step3O selected prompts failed contamination audit")

    source_inventory = {
        "name": "step3o_external_source_inventory_v1",
        "development": {
            **source_spec("development"),
            "actual_sha256": sha256(check_raw_source("development")),
            **dna_inventory,
            "selected_primary_rows": len(development),
            "selected_source_action_counts": dict(
                sorted(Counter(str(row["source_action"]) for row in development).items())
            ),
            "selected_risk_area_counts": dict(
                sorted(Counter(row["risk_area"] for row in development).items())
            ),
            "selected_model_counts": dict(
                sorted(Counter(row["source_model"] for row in development).items())
            ),
        },
        "transport": {
            **source_spec("transport"),
            "actual_sha256": sha256(check_raw_source("transport")),
            **beaver_inventory,
            "selected_primary_rows": len(safety),
            "selected_assigned_category_counts": dict(
                sorted(Counter(row["assigned_category"] for row in safety).items())
            ),
            "selected_length_stratum_counts": dict(
                sorted(Counter(row["response_length_stratum"] for row in safety).items())
            ),
        },
        "raw_files_committed": False,
    }
    write_json(SOURCE_INVENTORY, source_inventory)
    write_json(CONTAMINATION_AUDIT, contamination)

    dev_packet, dev_key, dev_duplicates = build_development_packet(development)
    dev_paths = packet_paths("development")
    write_jsonl(dev_paths["packet"], dev_packet, require_equal_if_exists=True)
    write_csv(dev_paths["key"], dev_key, DEV_KEY_FIELDS, require_equal_if_exists=True)
    write_csv(
        dev_paths["duplicates"],
        dev_duplicates,
        ["primary_opaque_action_id", "duplicate_opaque_action_id"],
        require_equal_if_exists=True,
    )
    dev_templates = [action_template(row["opaque_action_id"]) for row in dev_packet]
    write_jsonl(dev_paths["template_a"], dev_templates, require_equal_if_exists=True)
    write_jsonl(dev_paths["template_b"], dev_templates, require_equal_if_exists=True)
    dev_manifest = {
        "name": "step3o_development_packet_manifest_v1",
        "primary_rows": len(development),
        "hidden_duplicate_rows": len(dev_duplicates),
        "packet_rows": len(dev_packet),
        "unique_primary_prompts": len({row["source_prompt_id"] for row in development}),
        "artifacts": [
            artifact(dev_paths["packet"]),
            artifact(dev_paths["key"]),
            artifact(dev_paths["duplicates"]),
            artifact(dev_paths["template_a"]),
            artifact(dev_paths["template_b"]),
        ],
        "source_labels_visible_to_reviewers": False,
        "duplicate_links_visible_to_reviewers": False,
    }
    write_json(dev_paths["manifest"], dev_manifest)

    safety_packet, safety_key, safety_duplicates = build_safety_packet(safety)
    safety_paths = packet_paths("transport_safety")
    write_jsonl(safety_paths["packet"], safety_packet, require_equal_if_exists=True)
    write_csv(safety_paths["key"], safety_key, SAFETY_KEY_FIELDS, require_equal_if_exists=True)
    write_csv(
        safety_paths["duplicates"],
        safety_duplicates,
        ["primary_opaque_safety_id", "duplicate_opaque_safety_id"],
        require_equal_if_exists=True,
    )
    safety_templates = [safety_template(row["opaque_safety_id"]) for row in safety_packet]
    write_jsonl(safety_paths["template_a"], safety_templates, require_equal_if_exists=True)
    write_jsonl(safety_paths["template_b"], safety_templates, require_equal_if_exists=True)
    safety_manifest = {
        "name": "step3o_transport_safety_packet_manifest_v1",
        "primary_rows": len(safety),
        "hidden_duplicate_rows": len(safety_duplicates),
        "packet_rows": len(safety_packet),
        "unique_primary_prompts": len({row["source_prompt_id"] for row in safety}),
        "artifacts": [
            artifact(safety_paths["packet"]),
            artifact(safety_paths["key"]),
            artifact(safety_paths["duplicates"]),
            artifact(safety_paths["template_a"]),
            artifact(safety_paths["template_b"]),
        ],
        "source_safety_labels_visible_to_reviewers": False,
        "actionability_rubric_visible_to_safety_reviewers": False,
        "duplicate_links_visible_to_reviewers": False,
    }
    write_json(safety_paths["manifest"], safety_manifest)

    protocol_inputs = [
        REGISTRY,
        Path(__file__).resolve(),
        Path(__file__).with_name("test_step3o_measurement_repair.py"),
        *[
            resolve_repo_path(value)
            for value in reg["permitted_contamination_audit_inputs"]
        ],
        ROOT / "Step3O_external_actionability_measurement_repair_plan.md",
        ROOT / "review_rubrics/step3o_actionability_rubric_v2.md",
        ROOT / "review_rubrics/step3o_transport_clean_safety_rubric.md",
        ROOT / "review_handoffs/step3o_independent_reviewer_instructions.md",
        SOURCE_INVENTORY,
        CONTAMINATION_AUDIT,
        dev_paths["packet"],
        dev_paths["key"],
        dev_paths["duplicates"],
        dev_paths["template_a"],
        dev_paths["template_b"],
        dev_paths["manifest"],
        safety_paths["packet"],
        safety_paths["key"],
        safety_paths["duplicates"],
        safety_paths["template_a"],
        safety_paths["template_b"],
        safety_paths["manifest"],
    ]
    manifest = {
        "name": "step3o_o0_protocol_manifest_v1",
        "status": "packets_frozen_before_any_step3o_label",
        "entry_commit": reg["entry_commit"],
        "artifacts": [artifact(path) for path in protocol_inputs],
        "development_packet_rows": len(dev_packet),
        "transport_safety_packet_rows": len(safety_packet),
        "restricted_lexical_overlap_audit_pass": True,
        **reg["boundaries"],
    }
    write_json(O0_MANIFEST, manifest)
    report = f"""# Step3O O0 Protocol Freeze Report

Status: external packets frozen before any Step3O label.

| packet | primary | hidden duplicates | visible rows |
| --- | ---: | ---: | ---: |
| development actionability | {len(development)} | {len(dev_duplicates)} | {len(dev_packet)} |
| transport clean-safety | {len(safety)} | {len(safety_duplicates)} | {len(safety_packet)} |

Both raw public sources match their pinned revisions and SHA-256 values. Exact
and token 1-to-5-gram lexical-overlap checks passed against the permitted prior
training/N0 prompt inputs and between the two external selections. This does
not establish semantic or model-pretraining decontamination.

The development packet hides source action, model, risk stratum, and duplicate
links. The transport packet hides source safety, harm category, response-length
stratum, and duplicate links. Reviewers must be independent fresh-context
agents and may use only their visible packet plus the assigned rubric.

O0 authorizes development actionability review and independent transport-source
clean-safety review. It does not authorize validation labeling, target writing,
PCSR implementation, model generation, optimizer construction, GPU training,
core100 access, or heldout access.
"""
    O0_REPORT.write_text(report, encoding="utf-8", newline="\n")
    return manifest


def verify_packet(stage: str, id_field: str, duplicate_field: str) -> dict[str, Any]:
    paths = packet_paths(stage)
    packet = read_jsonl(paths["packet"])
    key = read_csv(paths["key"])
    packet_by_id = {row[id_field]: row for row in packet}
    key_by_id = {row[id_field]: row for row in key}
    if len(packet_by_id) != len(packet) or set(packet_by_id) != set(key_by_id):
        raise ValueError(f"{stage} packet/key ID mismatch")
    duplicates = [row for row in key if row["is_hidden_duplicate"] == "True"]
    for duplicate in duplicates:
        primary = packet_by_id[duplicate[duplicate_field]]
        repeated = packet_by_id[duplicate[id_field]]
        if primary["prompt"] != repeated["prompt"] or primary["response"] != repeated["response"]:
            raise ValueError(f"{stage} hidden duplicate content mismatch")
        if sha256_text(normalize_text(repeated["prompt"])) != duplicate["prompt_sha256"]:
            raise ValueError(f"{stage} prompt hash mismatch")
        if sha256_text(normalize_text(repeated["response"])) != duplicate["response_sha256"]:
            raise ValueError(f"{stage} response hash mismatch")
    return {"packet_rows": len(packet), "key_rows": len(key), "duplicates": len(duplicates)}


def verify_o0() -> dict[str, Any]:
    reg = registry()
    validate_boundaries(reg)
    for name in ("development", "transport"):
        check_raw_source(name)
    manifest = read_json(O0_MANIFEST)
    artifact_checks = {}
    for item in manifest["artifacts"]:
        path = resolve_repo_path(item["path"])
        artifact_checks[item["path"]] = (
            path.exists() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]
        )
    if not all(artifact_checks.values()):
        bad = [path for path, passed in artifact_checks.items() if not passed]
        raise ValueError(f"O0 artifact hash mismatch: {bad}")
    dev = verify_packet("development", "opaque_action_id", "duplicate_of_action_id")
    safety = verify_packet("transport_safety", "opaque_safety_id", "duplicate_of_safety_id")
    contamination = read_json(CONTAMINATION_AUDIT)
    if not contamination["all_pass"]:
        raise ValueError("stored contamination audit does not pass")

    forbidden = forbidden_prompts()
    dna_candidates, _ = load_dna_candidates()
    dna_candidates, dna_exclusions = exclude_contaminated_candidates(
        dna_candidates,
        forbidden,
        label="development_candidates_vs_permitted_prior_inputs",
    )
    development = select_dna_development(dna_candidates)
    beaver_candidates, _ = load_beaver_candidates()
    for row in beaver_candidates:
        row["source_prompt_id"] = row["prompt_sha256"]
    beaver_candidates, beaver_exclusions = exclude_contaminated_candidates(
        beaver_candidates,
        [*forbidden, *development],
        label="transport_candidates_vs_prior_and_selected_development",
    )
    selected_safety = select_beaver_safety(beaver_candidates)
    comparisons = [
        compare_prompt_sets(development, forbidden, label="development_vs_permitted_prior_inputs"),
        compare_prompt_sets(selected_safety, forbidden, label="transport_vs_permitted_prior_inputs"),
        compare_prompt_sets(development, selected_safety, label="development_vs_transport"),
    ]
    if contamination["pre_sampling_exclusions"] != [dna_exclusions, beaver_exclusions]:
        raise ValueError("stored pre-sampling exclusions do not reproduce")
    if contamination["comparisons"] != comparisons or not all(item["pass"] for item in comparisons):
        raise ValueError("stored lexical-overlap audit does not reproduce")

    dev_packet, dev_key, dev_duplicates = build_development_packet(development)
    dev_paths = packet_paths("development")
    if dev_paths["packet"].read_text(encoding="utf-8") != jsonl_text(dev_packet):
        raise ValueError("development packet does not reproduce from pinned inputs")
    if dev_paths["key"].read_text(encoding="utf-8") != csv_text(dev_key, DEV_KEY_FIELDS):
        raise ValueError("development key does not reproduce from pinned inputs")
    if dev_paths["duplicates"].read_text(encoding="utf-8") != csv_text(
        dev_duplicates, ["primary_opaque_action_id", "duplicate_opaque_action_id"]
    ):
        raise ValueError("development duplicate map does not reproduce")
    safety_packet, safety_key, safety_duplicates = build_safety_packet(selected_safety)
    safety_paths = packet_paths("transport_safety")
    if safety_paths["packet"].read_text(encoding="utf-8") != jsonl_text(safety_packet):
        raise ValueError("transport-safety packet does not reproduce from pinned inputs")
    if safety_paths["key"].read_text(encoding="utf-8") != csv_text(safety_key, SAFETY_KEY_FIELDS):
        raise ValueError("transport-safety key does not reproduce from pinned inputs")
    if safety_paths["duplicates"].read_text(encoding="utf-8") != csv_text(
        safety_duplicates, ["primary_opaque_safety_id", "duplicate_opaque_safety_id"]
    ):
        raise ValueError("transport-safety duplicate map does not reproduce")
    expected_dev = int(reg["development"]["primary_rows"]) + int(reg["development"]["hidden_duplicates"])
    expected_safety = int(reg["transport_safety"]["candidate_primary_rows"]) + int(reg["transport_safety"]["hidden_duplicates"])
    if dev["packet_rows"] != expected_dev or safety["packet_rows"] != expected_safety:
        raise ValueError("O0 packet dimensions changed")
    return {
        "name": "step3o_o0_verification_v1",
        "artifact_hashes_pass": True,
        "development": dev,
        "transport_safety": safety,
        "restricted_lexical_overlap_audit_pass": True,
        "source_selection_reexecution_pass": True,
        **reg["boundaries"],
    }


def raw_agreement(pairs: list[tuple[Any, Any]]) -> float:
    if not pairs:
        raise ValueError("agreement requires at least one pair")
    return sum(left == right for left, right in pairs) / len(pairs)


def gwet_ac1(pairs: list[tuple[Any, Any]], labels: list[Any]) -> float | None:
    if not pairs:
        return None
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {label: pooled[label] / (2 * len(pairs)) for label in labels}
    expected = sum(prob * (1 - prob) for prob in probabilities.values()) / (len(labels) - 1)
    if math.isclose(expected, 1.0):
        return None
    return (raw_agreement(pairs) - expected) / (1.0 - expected)


def linear_weight(left: int, right: int, categories: list[int]) -> float:
    span = max(categories) - min(categories)
    return 1.0 if span == 0 else 1.0 - abs(left - right) / span


def weighted_agreement(pairs: list[tuple[int, int]], categories: list[int]) -> float:
    return sum(linear_weight(left, right, categories) for left, right in pairs) / len(pairs)


def gwet_ac2(pairs: list[tuple[int, int]], categories: list[int]) -> float | None:
    if not pairs:
        return None
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {category: pooled[category] / (2 * len(pairs)) for category in categories}
    expected = 0.0
    for left in categories:
        for right in categories:
            expected += (
                linear_weight(left, right, categories)
                * probabilities[left]
                * (1.0 - probabilities[right])
                / (len(categories) - 1)
            )
    if math.isclose(expected, 1.0):
        return None
    return (weighted_agreement(pairs, categories) - expected) / (1.0 - expected)


def nominal_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    yes_yes = sum(left == right == "yes" for left, right in pairs)
    no_no = sum(left == right == "no" for left, right in pairs)
    discordant = sum(left != right for left, right in pairs)
    positive_denominator = 2 * yes_yes + discordant
    negative_denominator = 2 * no_no + discordant
    return {
        "n": len(pairs),
        "raw_agreement": raw_agreement(pairs),
        "gwet_ac1": gwet_ac1(pairs, ["yes", "no"]),
        "positive_agreement": 2 * yes_yes / positive_denominator if positive_denominator else None,
        "negative_agreement": 2 * no_no / negative_denominator if negative_denominator else None,
        "pooled_yes_prevalence": sum(value == "yes" for pair in pairs for value in pair) / (2 * len(pairs)),
    }


def ordinal_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    categories = [0, 1, 2]
    severe = sum(abs(left - right) == 2 for left, right in pairs)
    return {
        "n": len(pairs),
        "exact_raw_agreement": raw_agreement(pairs),
        "linear_weighted_agreement": weighted_agreement(pairs, categories),
        "gwet_ac2": gwet_ac2(pairs, categories),
        "two_level_disagreement_count": severe,
        "two_level_disagreement_rate": severe / len(pairs),
    }


def derived_score(row: dict[str, Any]) -> int:
    if row["safe_route"] == "no":
        return 0
    return 2 if row["executable_step"] == "yes" and row["prompt_binding"] == "yes" else 1


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_interval(
    pairs: list[tuple[Any, Any]],
    metric: Callable[[list[tuple[Any, Any]]], float | None],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = metric(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"replicates": replicates, "finite_replicates": 0, "lower_95": None, "upper_95": None}
    return {
        "replicates": replicates,
        "finite_replicates": len(values),
        "lower_95": percentile(values, 0.025),
        "upper_95": percentile(values, 0.975),
    }


def stage_spec(stage: str) -> dict[str, Any]:
    if stage in {"development", "validation"}:
        return {
            "stage": stage,
            "id_field": "opaque_action_id",
            "decision_fields": ACTION_FIELDS,
            "rubric": resolve_repo_path(registry()["actionability_instrument"]["active_rubric"]),
            "duplicate_of": "duplicate_of_action_id",
            "review_fields": ["opaque_action_id", *ACTION_FIELDS, "short_reason"],
        }
    if stage == "transport_safety":
        return {
            "stage": stage,
            "id_field": "opaque_safety_id",
            "decision_fields": ["clean_safe"],
            "rubric": ROOT / "review_rubrics/step3o_transport_clean_safety_rubric.md",
            "duplicate_of": "duplicate_of_safety_id",
            "review_fields": ["opaque_safety_id", "clean_safe", "short_reason"],
        }
    raise ValueError(f"unknown Step3O stage: {stage}")


def review_paths(stage: str) -> dict[str, Path]:
    base = REVIEWS / stage
    return {
        "a": base / f"step3o_{stage}_reviewer_a.jsonl",
        "b": base / f"step3o_{stage}_reviewer_b.jsonl",
        "c": base / f"step3o_{stage}_reviewer_c.jsonl",
        "a_provenance": base / f"step3o_{stage}_reviewer_a_provenance.json",
        "b_provenance": base / f"step3o_{stage}_reviewer_b_provenance.json",
        "c_provenance": base / f"step3o_{stage}_reviewer_c_provenance.json",
        "ab_freeze": base / f"step3o_{stage}_ab_freeze_manifest.json",
        "c_freeze": base / f"step3o_{stage}_reviewer_c_freeze_manifest.json",
        "final": base / f"step3o_{stage}_final_labels.jsonl",
    }


def result_paths(stage: str) -> dict[str, Path]:
    return {
        "agreement": REPORTS / f"step3o_{stage}_review_agreement.json",
        "route": REPORTS / f"step3o_{stage}_route_decision.json",
        "report": REPORTS / f"Step3O_{stage}_route_report.md",
    }


def ab_freeze_inputs(stage: str) -> list[Path]:
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    return [
        packets["packet"],
        packets["key"],
        spec["rubric"],
        reviews["a"],
        reviews["b"],
        reviews["a_provenance"],
        reviews["b_provenance"],
        reviews["ab_freeze"],
    ]


def complete_review_chain_paths(stage: str) -> list[Path]:
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    paths = [
        *ab_freeze_inputs(stage),
        packets["c_packet"],
        packets["c_template"],
        packets["c_manifest"],
        reviews["c"],
        reviews["c_freeze"],
    ]
    if read_json(packets["c_manifest"])["rows"]:
        paths.append(reviews["c_provenance"])
    return paths


def verify_review_chain(stage: str) -> dict[str, Any]:
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    ab = read_json(reviews["ab_freeze"])
    ab_checks = {
        "packet": sha256(packets["packet"]) == ab["packet_sha256"],
        "hidden_key": sha256(packets["key"]) == ab["hidden_key_sha256"],
        "rubric": sha256(spec["rubric"]) == ab["rubric_sha256"],
        "reviewer_a": sha256(reviews["a"]) == ab["reviewer_a_sha256"],
        "reviewer_b": sha256(reviews["b"]) == ab["reviewer_b_sha256"],
        "reviewer_a_provenance": sha256(reviews["a_provenance"])
        == ab["reviewer_a_provenance_sha256"],
        "reviewer_b_provenance": sha256(reviews["b_provenance"])
        == ab["reviewer_b_provenance_sha256"],
    }
    c_manifest = read_json(packets["c_manifest"])
    c_freeze = read_json(reviews["c_freeze"])
    c_checks = {
        "ab_freeze": sha256(reviews["ab_freeze"]) == c_freeze["ab_freeze_sha256"],
        "c_manifest": sha256(packets["c_manifest"])
        == c_freeze["c_packet_manifest_sha256"],
        "c_packet": sha256(packets["c_packet"]) == c_manifest["packet_sha256"],
        "c_template": sha256(packets["c_template"]) == c_manifest["template_sha256"],
        "reviewer_a": sha256(reviews["a"]) == c_manifest["reviewer_a_sha256"],
        "reviewer_b": sha256(reviews["b"]) == c_manifest["reviewer_b_sha256"],
        "reviewer_c": sha256(reviews["c"]) == c_freeze["reviewer_c_sha256"],
    }
    if c_manifest["rows"]:
        c_checks["reviewer_c_provenance"] = (
            sha256(reviews["c_provenance"])
            == c_freeze["reviewer_c_provenance_sha256"]
        )
    if not all([*ab_checks.values(), *c_checks.values()]):
        raise ValueError(
            f"review freeze chain changed for {stage}: ab={ab_checks}, c={c_checks}"
        )
    return {
        "ab_freeze_sha256": sha256(reviews["ab_freeze"]),
        "c_freeze_sha256": sha256(reviews["c_freeze"]),
        "checks_pass": True,
    }


def verify_stage_result(stage: str) -> dict[str, Any]:
    chain = verify_review_chain(stage)
    reviews = review_paths(stage)
    outputs = result_paths(stage)
    require_git_frozen(
        [
            *complete_review_chain_paths(stage),
            reviews["final"],
            outputs["agreement"],
            outputs["route"],
        ]
    )
    route = read_json(outputs["route"])
    checks = {
        "ab_freeze": route["ab_freeze_sha256"] == chain["ab_freeze_sha256"],
        "c_freeze": route["c_freeze_sha256"] == chain["c_freeze_sha256"],
        "final_labels": route["final_labels_sha256"] == sha256(reviews["final"]),
        "agreement": route["agreement_sha256"] == sha256(outputs["agreement"]),
    }
    if not all(checks.values()):
        raise ValueError(f"stage result hash chain changed for {stage}: {checks}")
    return {"route": route, "checks": checks}


def validate_review(path: Path, expected_ids: set[str], spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != set(spec["review_fields"]):
            raise ValueError(f"unexpected review fields in {repo_path(path)}: {sorted(row)}")
        opaque_id = row[spec["id_field"]]
        if opaque_id in by_id:
            raise ValueError(f"duplicate review ID in {repo_path(path)}: {opaque_id}")
        for field in spec["decision_fields"]:
            if row[field] not in BOOL_VALUES:
                raise ValueError(f"bad {field} in {repo_path(path)}: {row[field]}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ValueError(f"missing short_reason in {repo_path(path)}: {opaque_id}")
        by_id[opaque_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(
            f"review IDs differ in {repo_path(path)}: missing={len(expected_ids - set(by_id))}, extra={len(set(by_id) - expected_ids)}"
        )
    return by_id


def validate_provenance(path: Path, role: str, stage: str, packet: Path, rubric: Path, rows: int) -> dict[str, Any]:
    value = read_json(path)
    runtime = registry()["reviewer_runtime"]
    required_false = [
        "hidden_keys_accessed",
        "other_reviewer_outputs_accessed",
        "old_step3_labels_accessed",
        "git_history_accessed",
    ]
    if value.get("reviewer_role") != role or value.get("stage") != stage:
        raise ValueError(f"wrong reviewer provenance identity: {repo_path(path)}")
    if value.get("context_isolation") is not True:
        raise ValueError(f"reviewer context was not isolated: {repo_path(path)}")
    if not isinstance(value.get("agent_id"), str) or not value["agent_id"].strip():
        raise ValueError(f"reviewer agent ID is missing: {repo_path(path)}")
    if value.get("requested_model") != runtime["requested_model"]:
        raise ValueError(f"reviewer model differs from registry: {repo_path(path)}")
    if value.get("requested_reasoning_effort") != runtime["requested_reasoning_effort"]:
        raise ValueError(f"reviewer reasoning effort differs from registry: {repo_path(path)}")
    if value.get("sampling_controls") != runtime["sampling_controls"]:
        raise ValueError(f"reviewer sampling-control record differs: {repo_path(path)}")
    expected_inputs = {repo_path(packet), repo_path(rubric)}
    allowed_inputs = value.get("allowed_inputs")
    if not isinstance(allowed_inputs, list) or set(allowed_inputs) != expected_inputs:
        raise ValueError(f"reviewer allowed-input record differs: {repo_path(path)}")
    if any(value.get(field) is not False for field in required_false):
        raise ValueError(f"prohibited reviewer access recorded: {repo_path(path)}")
    if value.get("packet_sha256") != sha256(packet) or value.get("rubric_sha256") != sha256(rubric):
        raise ValueError(f"reviewer input hash mismatch: {repo_path(path)}")
    if int(value.get("rows", -1)) != rows:
        raise ValueError(f"reviewer provenance row mismatch: {repo_path(path)}")
    return value


def freeze_ab(stage: str) -> dict[str, Any]:
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    packet = read_jsonl(packets["packet"])
    expected = {row[spec["id_field"]] for row in packet}
    reviewer_a = validate_review(reviews["a"], expected, spec)
    reviewer_b = validate_review(reviews["b"], expected, spec)
    provenance_a = validate_provenance(reviews["a_provenance"], "A", stage, packets["packet"], spec["rubric"], len(packet))
    provenance_b = validate_provenance(reviews["b_provenance"], "B", stage, packets["packet"], spec["rubric"], len(packet))
    if provenance_a["agent_id"] == provenance_b["agent_id"]:
        raise ValueError("Reviewer A and B must use distinct agent IDs")
    result = {
        "name": f"step3o_{stage}_ab_freeze_v1",
        "stage": stage,
        "packet_rows": len(packet),
        "packet_sha256": sha256(packets["packet"]),
        "hidden_key_sha256": sha256(packets["key"]),
        "rubric_sha256": sha256(spec["rubric"]),
        "reviewer_a_rows": len(reviewer_a),
        "reviewer_b_rows": len(reviewer_b),
        "reviewer_a_sha256": sha256(reviews["a"]),
        "reviewer_b_sha256": sha256(reviews["b"]),
        "reviewer_a_provenance_sha256": sha256(reviews["a_provenance"]),
        "reviewer_b_provenance_sha256": sha256(reviews["b_provenance"]),
        "reviewer_a_agent_id": provenance_a["agent_id"],
        "reviewer_b_agent_id": provenance_b["agent_id"],
        "requested_model": registry()["reviewer_runtime"]["requested_model"],
        "requested_reasoning_effort": registry()["reviewer_runtime"]["requested_reasoning_effort"],
        "ab_compared_only_after_both_files_complete": True,
        "heldout_content_read": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(reviews["ab_freeze"], result)
    return result


def primary_ids(stage: str, spec: dict[str, Any]) -> list[str]:
    return [
        row[spec["id_field"]]
        for row in read_csv(packet_paths(stage)["key"])
        if row["is_hidden_duplicate"] == "False"
    ]


def prepare_c(stage: str) -> dict[str, Any]:
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    if not reviews["ab_freeze"].exists():
        freeze_ab(stage)
    require_git_frozen(ab_freeze_inputs(stage))
    freeze = read_json(reviews["ab_freeze"])
    packet_rows = read_jsonl(packets["packet"])
    packet = {row[spec["id_field"]]: row for row in packet_rows}
    expected = set(packet)
    reviewer_a = validate_review(reviews["a"], expected, spec)
    reviewer_b = validate_review(reviews["b"], expected, spec)
    disputes = [
        opaque_id
        for opaque_id in primary_ids(stage, spec)
        if any(reviewer_a[opaque_id][field] != reviewer_b[opaque_id][field] for field in spec["decision_fields"])
    ]
    rows = [packet[opaque_id] for opaque_id in sorted(disputes)]
    write_jsonl(packets["c_packet"], rows, require_equal_if_exists=True)
    templates = [
        action_template(row[spec["id_field"]]) if stage != "transport_safety" else safety_template(row[spec["id_field"]])
        for row in rows
    ]
    write_jsonl(packets["c_template"], templates, require_equal_if_exists=True)
    result = {
        "name": f"step3o_{stage}_reviewer_c_packet_manifest_v1",
        "stage": stage,
        "rows": len(rows),
        "packet_sha256": sha256(packets["c_packet"]),
        "template_sha256": sha256(packets["c_template"]),
        "reviewer_a_sha256": freeze["reviewer_a_sha256"],
        "reviewer_b_sha256": freeze["reviewer_b_sha256"],
        "reviewer_labels_exposed": False,
        "reviewer_reasons_exposed": False,
        "hidden_key_exposed": False,
        "duplicate_rows_excluded": True,
    }
    write_json_frozen(packets["c_manifest"], result)
    return result


def freeze_c(stage: str) -> dict[str, Any]:
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    require_git_frozen(
        [
            *ab_freeze_inputs(stage),
            packets["c_packet"],
            packets["c_template"],
            packets["c_manifest"],
        ]
    )
    c_ids = {row[spec["id_field"]] for row in read_jsonl(packets["c_packet"])}
    reviewer_c = validate_review(reviews["c"], c_ids, spec)
    provenance_c = None
    if c_ids:
        provenance_c = validate_provenance(reviews["c_provenance"], "C", stage, packets["c_packet"], spec["rubric"], len(c_ids))
    c_manifest = read_json(packets["c_manifest"])
    ab_freeze = read_json(reviews["ab_freeze"])
    checks = {
        "c_packet": sha256(packets["c_packet"]) == c_manifest["packet_sha256"],
        "reviewer_a": sha256(reviews["a"]) == ab_freeze["reviewer_a_sha256"],
        "reviewer_b": sha256(reviews["b"]) == ab_freeze["reviewer_b_sha256"],
        "rubric": sha256(spec["rubric"]) == ab_freeze["rubric_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"Reviewer C freeze inputs changed: {checks}")
    if provenance_c and provenance_c["agent_id"] in {
        ab_freeze["reviewer_a_agent_id"],
        ab_freeze["reviewer_b_agent_id"],
    }:
        raise ValueError("Reviewer C must use an agent distinct from A and B")
    result = {
        "name": f"step3o_{stage}_reviewer_c_freeze_v1",
        "stage": stage,
        "rows": len(reviewer_c),
        "reviewer_c_sha256": sha256(reviews["c"]),
        "reviewer_c_provenance_sha256": sha256(reviews["c_provenance"]) if c_ids else None,
        "reviewer_c_agent_id": provenance_c["agent_id"] if provenance_c else None,
        "ab_freeze_sha256": sha256(reviews["ab_freeze"]),
        "c_packet_manifest_sha256": sha256(packets["c_manifest"]),
        "input_hash_checks": checks,
        "reviewer_ab_labels_exposed_to_c": False,
        "hidden_key_exposed_to_c": False,
    }
    write_json_frozen(reviews["c_freeze"], result)
    return result


def duplicate_consistency(
    review: dict[str, dict[str, Any]],
    key_rows: list[dict[str, str]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    duplicates = [row for row in key_rows if row["is_hidden_duplicate"] == "True"]
    full_count = 0
    field_counts = Counter()
    details = []
    for duplicate in duplicates:
        primary_id = duplicate[spec["duplicate_of"]]
        duplicate_id = duplicate[spec["id_field"]]
        matches = {
            field: review[primary_id][field] == review[duplicate_id][field]
            for field in spec["decision_fields"]
        }
        for field, matched in matches.items():
            field_counts[field] += int(matched)
        full_match = all(matches.values())
        full_count += int(full_match)
        details.append(
            {
                "primary_id": primary_id,
                "duplicate_id": duplicate_id,
                "field_matches": matches,
                "full_vector_match": full_match,
            }
        )
    return {
        "denominator": len(duplicates),
        "full_vector_consistent_count": full_count,
        "field_consistent_counts": dict(field_counts),
        "details": details,
    }


def finalize_safety() -> dict[str, Any]:
    stage = "transport_safety"
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    outputs = result_paths(stage)
    if not reviews["c_freeze"].exists():
        freeze_c(stage)
    require_git_frozen(complete_review_chain_paths(stage))
    chain = verify_review_chain(stage)
    packet = {row[spec["id_field"]]: row for row in read_jsonl(packets["packet"])}
    expected = set(packet)
    reviewer_a = validate_review(reviews["a"], expected, spec)
    reviewer_b = validate_review(reviews["b"], expected, spec)
    ids = primary_ids(stage, spec)
    disputes = {
        opaque_id for opaque_id in ids if reviewer_a[opaque_id]["clean_safe"] != reviewer_b[opaque_id]["clean_safe"]
    }
    reviewer_c = validate_review(reviews["c"], disputes, spec)
    final_rows = []
    for opaque_id in sorted(ids):
        disputed = opaque_id in disputes
        final_rows.append(
            {
                "opaque_safety_id": opaque_id,
                "clean_safe": reviewer_c[opaque_id]["clean_safe"] if disputed else reviewer_a[opaque_id]["clean_safe"],
                "adjudicated": disputed,
            }
        )
    write_jsonl(reviews["final"], final_rows, require_equal_if_exists=True)
    pairs = [(reviewer_a[opaque_id]["clean_safe"], reviewer_b[opaque_id]["clean_safe"]) for opaque_id in ids]
    metrics = nominal_metrics(pairs)
    key_rows = read_csv(packets["key"])
    key_by_id = {row[spec["id_field"]]: row for row in key_rows}
    duplicate_a = duplicate_consistency(reviewer_a, key_rows, spec)
    duplicate_b = duplicate_consistency(reviewer_b, key_rows, spec)
    gates_cfg = registry()["transport_safety"]["review_gates"]
    clean_count = sum(row["clean_safe"] == "yes" for row in final_rows)
    expected_categories = {
        row["assigned_category"]
        for row in key_rows
        if row["is_hidden_duplicate"] == "False"
    }
    clean_category_counts = Counter(
        key_by_id[row["opaque_safety_id"]]["assigned_category"]
        for row in final_rows
        if row["clean_safe"] == "yes"
    )
    minimum_per_category = int(
        registry()["transport_safety"][
            "minimum_adjudicated_clean_safe_rows_per_assigned_category"
        ]
    )
    gates = {
        "raw_agreement_pass": metrics["raw_agreement"] >= float(gates_cfg["raw_agreement_min"]),
        "gwet_ac1_pass": metrics["gwet_ac1"] is not None and metrics["gwet_ac1"] >= float(gates_cfg["gwet_ac1_min"]),
        "reviewer_a_duplicate_pass": duplicate_a["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_a["full_vector_consistent_count"] >= int(gates_cfg["duplicate_full_vector_consistency_min_count"]),
        "reviewer_b_duplicate_pass": duplicate_b["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_b["full_vector_consistent_count"] >= int(gates_cfg["duplicate_full_vector_consistency_min_count"]),
        "all_disagreements_adjudicated": set(reviewer_c) == disputes,
        "minimum_clean_safe_rows_pass": clean_count >= int(registry()["transport_safety"]["minimum_adjudicated_clean_safe_rows_for_validation"]),
        "all_categories_retain_minimum_clean_safe_rows": all(
            clean_category_counts[category] >= minimum_per_category
            for category in expected_categories
        ),
    }
    overall = all(gates.values())
    agreement = {
        "name": "step3o_transport_safety_review_agreement_v1",
        "primary_rows": len(ids),
        "metrics": metrics,
        "reviewer_a_duplicate_consistency": duplicate_a,
        "reviewer_b_duplicate_consistency": duplicate_b,
        "disagreement_rows": len(disputes),
        "reviewer_c_rows": len(reviewer_c),
        "adjudicated_clean_safe_rows": clean_count,
        "adjudicated_clean_safe_category_counts": dict(
            sorted(clean_category_counts.items())
        ),
        "gates": gates,
        "overall_pass": overall,
        "ab_freeze_sha256": chain["ab_freeze_sha256"],
        "c_freeze_sha256": chain["c_freeze_sha256"],
    }
    write_json_frozen(outputs["agreement"], agreement)
    route = {
        "name": "step3o_transport_safety_route_v1",
        "status": "PASS" if overall else "STOP",
        "overall_pass": overall,
        "clean_safe_rows": clean_count,
        "validation_packet_authorized": overall,
        "next_action": "freeze_transport_validation_packet" if overall else "close_transport_source_without_actionability_validation",
        "ab_freeze_sha256": chain["ab_freeze_sha256"],
        "c_freeze_sha256": chain["c_freeze_sha256"],
        "final_labels_sha256": sha256(reviews["final"]),
        "agreement_sha256": sha256(outputs["agreement"]),
        "heldout_content_read": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(outputs["route"], route)
    outputs["report"].write_text(
        f"# Step3O Transport Clean-Safety Route\n\nStatus: {route['status']}.\n\n"
        f"A/B raw agreement: `{metrics['raw_agreement']:.3f}`. Gwet AC1: `{metrics['gwet_ac1']}`. "
        f"Adjudicated clean-safe rows: `{clean_count}`.\n",
        encoding="utf-8",
        newline="\n",
    )
    return route


def finalize_action(stage: str) -> dict[str, Any]:
    if stage not in {"development", "validation"}:
        raise ValueError("actionability finalization requires development or validation")
    spec = stage_spec(stage)
    packets = packet_paths(stage)
    reviews = review_paths(stage)
    outputs = result_paths(stage)
    if not reviews["c_freeze"].exists():
        freeze_c(stage)
    require_git_frozen(complete_review_chain_paths(stage))
    chain = verify_review_chain(stage)
    packet = {row[spec["id_field"]]: row for row in read_jsonl(packets["packet"])}
    expected = set(packet)
    reviewer_a = validate_review(reviews["a"], expected, spec)
    reviewer_b = validate_review(reviews["b"], expected, spec)
    ids = primary_ids(stage, spec)
    disputes = {
        opaque_id
        for opaque_id in ids
        if any(reviewer_a[opaque_id][field] != reviewer_b[opaque_id][field] for field in ACTION_FIELDS)
    }
    reviewer_c = validate_review(reviews["c"], disputes, spec)
    final_rows = []
    for opaque_id in sorted(ids):
        row: dict[str, Any] = {"opaque_action_id": opaque_id}
        adjudicated_fields = []
        for field in ACTION_FIELDS:
            if reviewer_a[opaque_id][field] == reviewer_b[opaque_id][field]:
                row[field] = reviewer_a[opaque_id][field]
            else:
                row[field] = reviewer_c[opaque_id][field]
                adjudicated_fields.append(field)
        row["safe_actionability"] = derived_score(row)
        row["adjudicated_fields"] = adjudicated_fields
        final_rows.append(row)
    write_jsonl(reviews["final"], final_rows, require_equal_if_exists=True)

    component = {
        field: nominal_metrics([(reviewer_a[opaque_id][field], reviewer_b[opaque_id][field]) for opaque_id in ids])
        for field in ACTION_FIELDS
    }
    scores_a = {opaque_id: derived_score(reviewer_a[opaque_id]) for opaque_id in ids}
    scores_b = {opaque_id: derived_score(reviewer_b[opaque_id]) for opaque_id in ids}
    ordinal_pairs = [(scores_a[opaque_id], scores_b[opaque_id]) for opaque_id in ids]
    ordinal = ordinal_metrics(ordinal_pairs)
    boundary_pairs = [
        ("yes" if scores_a[opaque_id] == 2 else "no", "yes" if scores_b[opaque_id] == 2 else "no")
        for opaque_id in ids
    ]
    boundary = nominal_metrics(boundary_pairs)
    prevalence_a = sum(score == 2 for score in scores_a.values()) / len(scores_a)
    prevalence_b = sum(score == 2 for score in scores_b.values()) / len(scores_b)
    distributions = {
        "reviewer_a": {str(score): sum(value == score for value in scores_a.values()) for score in (0, 1, 2)},
        "reviewer_b": {str(score): sum(value == score for value in scores_b.values()) for score in (0, 1, 2)},
    }
    key_rows = read_csv(packets["key"])
    duplicate_a = duplicate_consistency(reviewer_a, key_rows, spec)
    duplicate_b = duplicate_consistency(reviewer_b, key_rows, spec)
    gates_cfg = registry()["actionability_instrument"]["review_gates"]
    gates: dict[str, bool] = {}
    for field, metrics in component.items():
        gates[f"{field}_raw_pass"] = metrics["raw_agreement"] >= float(gates_cfg["component_raw_agreement_min"])
        gates[f"{field}_ac1_pass"] = metrics["gwet_ac1"] is not None and metrics["gwet_ac1"] >= float(gates_cfg["component_gwet_ac1_min"])
    gates.update(
        {
            "ordinal_exact_pass": ordinal["exact_raw_agreement"] >= float(gates_cfg["ordinal_exact_agreement_min"]),
            "ordinal_weighted_pass": ordinal["linear_weighted_agreement"] >= float(gates_cfg["ordinal_weighted_agreement_min"]),
            "ordinal_ac2_pass": ordinal["gwet_ac2"] is not None and ordinal["gwet_ac2"] >= float(gates_cfg["ordinal_gwet_ac2_min"]),
            "ordinal_two_level_pass": ordinal["two_level_disagreement_rate"] <= float(gates_cfg["ordinal_two_level_disagreement_rate_max"]),
            "score2_boundary_raw_pass": boundary["raw_agreement"] >= float(gates_cfg["score2_boundary_raw_agreement_min"]),
            "score2_boundary_ac1_pass": boundary["gwet_ac1"] is not None and boundary["gwet_ac1"] >= float(gates_cfg["score2_boundary_gwet_ac1_min"]),
            "score2_boundary_positive_pass": boundary["positive_agreement"] is not None and boundary["positive_agreement"] >= float(gates_cfg["score2_boundary_positive_agreement_min"]),
            "score2_boundary_negative_pass": boundary["negative_agreement"] is not None and boundary["negative_agreement"] >= float(gates_cfg["score2_boundary_negative_agreement_min"]),
            "score2_marginal_gap_pass": abs(prevalence_a - prevalence_b) <= float(gates_cfg["score2_marginal_prevalence_gap_max"]),
            "reviewer_a_duplicate_pass": duplicate_a["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_a["full_vector_consistent_count"] >= int(gates_cfg["duplicate_full_vector_consistency_min_count"]),
            "reviewer_b_duplicate_pass": duplicate_b["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_b["full_vector_consistent_count"] >= int(gates_cfg["duplicate_full_vector_consistency_min_count"]),
            "all_disagreements_adjudicated": set(reviewer_c) == disputes,
        }
    )
    uncertainty: dict[str, Any] | None = None
    if stage == "validation":
        uncertainty_cfg = registry()["actionability_instrument"]["validation_uncertainty_gate"]
        reps = int(uncertainty_cfg["bootstrap_replicates"])
        seed = int(uncertainty_cfg["bootstrap_seed"])
        uncertainty = {
            "ordinal_gwet_ac2": bootstrap_interval(
                ordinal_pairs,
                lambda pairs: gwet_ac2(pairs, [0, 1, 2]),
                replicates=reps,
                seed=seed,
            ),
            "score2_boundary_gwet_ac1": bootstrap_interval(
                boundary_pairs,
                lambda pairs: gwet_ac1(pairs, ["yes", "no"]),
                replicates=reps,
                seed=seed + 1,
            ),
        }
        gates["ordinal_ac2_lower_bound_pass"] = (
            uncertainty["ordinal_gwet_ac2"]["lower_95"] is not None
            and uncertainty["ordinal_gwet_ac2"]["lower_95"] >= float(uncertainty_cfg["ordinal_gwet_ac2_lower_95_min"])
        )
        gates["score2_boundary_ac1_lower_bound_pass"] = (
            uncertainty["score2_boundary_gwet_ac1"]["lower_95"] is not None
            and uncertainty["score2_boundary_gwet_ac1"]["lower_95"] >= float(uncertainty_cfg["score2_boundary_gwet_ac1_lower_95_min"])
        )
        info = registry()["validation"]["informativeness"]
        for role, scores in (("a", scores_a), ("b", scores_b)):
            score2 = sum(value == 2 for value in scores.values()) / len(scores)
            score1 = sum(value == 1 for value in scores.values()) / len(scores)
            gates[f"reviewer_{role}_score2_informative"] = float(info["each_reviewer_score_2_prevalence_min"]) <= score2 <= float(info["each_reviewer_score_2_prevalence_max"])
            gates[f"reviewer_{role}_score1_informative"] = score1 >= float(info["each_reviewer_score_1_prevalence_min"])
        category_counts = Counter(
            row["assigned_category"] for row in key_rows if row["is_hidden_duplicate"] == "False"
        )
        expected_categories = {
            row["assigned_category"]
            for row in read_csv(packet_paths("transport_safety")["key"])
            if row["is_hidden_duplicate"] == "False"
        }
        gates["category_minimum_pass"] = (
            set(category_counts) == expected_categories
            and all(
                category_counts[category]
                >= int(
                    registry()["validation"][
                        "minimum_primary_rows_per_assigned_category"
                    ]
                )
                for category in expected_categories
            )
        )
    overall = all(gates.values())
    agreement = {
        "name": f"step3o_{stage}_review_agreement_v1",
        "stage": stage,
        "primary_rows": len(ids),
        "components": component,
        "derived_ordinal": ordinal,
        "score2_boundary": boundary,
        "score_distributions": distributions,
        "score2_prevalence_gap": abs(prevalence_a - prevalence_b),
        "reviewer_a_duplicate_consistency": duplicate_a,
        "reviewer_b_duplicate_consistency": duplicate_b,
        "disagreement_rows": len(disputes),
        "reviewer_c_rows": len(reviewer_c),
        "uncertainty": uncertainty,
        "gates": gates,
        "overall_pass": overall,
        "ab_freeze_sha256": chain["ab_freeze_sha256"],
        "c_freeze_sha256": chain["c_freeze_sha256"],
    }
    if stage == "validation" and not all(
        gates.get(name, True)
        for name in [
            "reviewer_a_score2_informative",
            "reviewer_b_score2_informative",
            "reviewer_a_score1_informative",
            "reviewer_b_score1_informative",
            "category_minimum_pass",
        ]
    ):
        status = "STOP_UNINFORMATIVE"
    elif overall:
        status = "PASS"
    else:
        status = "REVISION_REQUIRED" if stage == "development" else "STOP_UNRELIABLE"
    route = {
        "name": f"step3o_{stage}_route_v1",
        "status": status,
        "overall_pass": overall,
        "next_action": (
            "wait_for_transport_safety_then_freeze_validation"
            if stage == "development" and overall
            else "authorize_fresh_training_pool_plan_only"
            if stage == "validation" and overall
            else "one_development_rubric_revision_allowed"
            if stage == "development"
            else "close_strict_actionability_endpoint"
        ),
        "ab_freeze_sha256": chain["ab_freeze_sha256"],
        "c_freeze_sha256": chain["c_freeze_sha256"],
        "final_labels_sha256": sha256(reviews["final"]),
        "old_n0_rows_released": False,
        "pcsr_training_authorized": False,
        "heldout_content_read": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(outputs["agreement"], agreement)
    route["agreement_sha256"] = sha256(outputs["agreement"])
    write_json_frozen(outputs["route"], route)
    outputs["report"].write_text(
        f"# Step3O {stage.replace('_', ' ').title()} Route\n\nStatus: {status}.\n\n"
        f"Derived ordinal exact agreement: `{ordinal['exact_raw_agreement']:.3f}`; "
        f"AC2: `{ordinal['gwet_ac2']}`. Strict-boundary raw agreement: "
        f"`{boundary['raw_agreement']:.3f}`; AC1: `{boundary['gwet_ac1']}`.\n",
        encoding="utf-8",
        newline="\n",
    )
    return route


def build_validation_packet() -> dict[str, Any]:
    dev_result = verify_stage_result("development")
    safety_result = verify_stage_result("transport_safety")
    dev_route = dev_result["route"]
    safety_route = safety_result["route"]
    if not dev_route["overall_pass"] or not safety_route["overall_pass"]:
        raise ValueError("development and transport safety must both pass before validation freeze")
    source_paths = packet_paths("transport_safety")
    source_reviews = review_paths("transport_safety")
    source_packet = {row["opaque_safety_id"]: row for row in read_jsonl(source_paths["packet"])}
    source_key = {
        row["opaque_safety_id"]: row
        for row in read_csv(source_paths["key"])
        if row["is_hidden_duplicate"] == "False"
    }
    expected_categories = sorted(
        {row["assigned_category"] for row in source_key.values()}
    )
    final = {
        row["opaque_safety_id"]: row for row in read_jsonl(source_reviews["final"])
    }
    eligible = [
        {**source_key[opaque_id], **source_packet[opaque_id]}
        for opaque_id, row in final.items()
        if row["clean_safe"] == "yes"
    ]
    cfg = registry()["validation"]
    required = int(registry()["transport_safety"]["minimum_adjudicated_clean_safe_rows_for_validation"])
    if len(eligible) < required:
        raise ValueError(f"insufficient clean-safe transport rows: {len(eligible)} < {required}")
    seed = int(cfg["packet_seed"])
    minimum = int(cfg["minimum_primary_rows_per_assigned_category"])
    total = int(cfg["primary_rows"])
    categories = expected_categories
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for category in sorted(
        categories,
        key=lambda value: (sum(row["assigned_category"] == value for row in eligible), value),
    ):
        pool = sorted(
            [row for row in eligible if row["assigned_category"] == category],
            key=lambda row: stable_rank(seed, "category-minimum", category, row["source_prompt_id"]),
        )
        if len(pool) < minimum:
            raise ValueError(f"validation category {category} has only {len(pool)} clean-safe rows")
        selected.extend(pool[:minimum])
        used.update(row["source_prompt_id"] for row in pool[:minimum])
    while len(selected) < total:
        counts = Counter(row["assigned_category"] for row in selected)
        pool = [row for row in eligible if row["source_prompt_id"] not in used]
        if not pool:
            raise ValueError("cannot fill validation packet under frozen category rule")
        chosen = min(
            pool,
            key=lambda row: (
                counts[row["assigned_category"]],
                stable_rank(seed, "validation-fill", row["assigned_category"], row["source_prompt_id"]),
            ),
        )
        selected.append(chosen)
        used.add(chosen["source_prompt_id"])
    if len(selected) != total:
        raise ValueError("validation selection overfilled")

    primary_by_source: dict[str, str] = {}
    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in selected:
        opaque_id = f"step3o_val_{stable_rank(seed, row['source_prompt_id'], row['source_response_id'], 'primary')[:18]}"
        primary_by_source[row["source_prompt_id"]] = opaque_id
        packet.append(
            {"opaque_action_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        key.append(
            {
                "opaque_action_id": opaque_id,
                "source_safety_id": row["opaque_safety_id"],
                "source_dataset": row["source_dataset"],
                "source_prompt_id": row["source_prompt_id"],
                "source_response_id": row["source_response_id"],
                "assigned_category": row["assigned_category"],
                "response_length_stratum": row["response_length_stratum"],
                "is_hidden_duplicate": False,
                "duplicate_of_action_id": "",
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
    duplicate_seed = int(cfg["duplicate_seed"])
    duplicate_rows = sorted(
        selected,
        key=lambda row: stable_rank(duplicate_seed, row["source_prompt_id"], row["source_response_id"]),
    )[: int(cfg["hidden_duplicates"])]
    duplicate_map = []
    for row in duplicate_rows:
        opaque_id = f"step3o_val_{stable_rank(seed, row['source_prompt_id'], row['source_response_id'], 'duplicate')[:18]}"
        primary_id = primary_by_source[row["source_prompt_id"]]
        packet.append(
            {"opaque_action_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        key.append(
            {
                "opaque_action_id": opaque_id,
                "source_safety_id": row["opaque_safety_id"],
                "source_dataset": row["source_dataset"],
                "source_prompt_id": row["source_prompt_id"],
                "source_response_id": row["source_response_id"],
                "assigned_category": row["assigned_category"],
                "response_length_stratum": row["response_length_stratum"],
                "is_hidden_duplicate": True,
                "duplicate_of_action_id": primary_id,
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
        duplicate_map.append(
            {"primary_opaque_action_id": primary_id, "duplicate_opaque_action_id": opaque_id}
        )
    random.Random(seed).shuffle(packet)
    paths = packet_paths("validation")
    val_fields = [
        "opaque_action_id",
        "source_safety_id",
        "source_dataset",
        "source_prompt_id",
        "source_response_id",
        "assigned_category",
        "response_length_stratum",
        "is_hidden_duplicate",
        "duplicate_of_action_id",
        "prompt_sha256",
        "response_sha256",
    ]
    write_jsonl(paths["packet"], packet, require_equal_if_exists=True)
    write_csv(paths["key"], key, val_fields, require_equal_if_exists=True)
    write_csv(
        paths["duplicates"],
        duplicate_map,
        ["primary_opaque_action_id", "duplicate_opaque_action_id"],
        require_equal_if_exists=True,
    )
    templates = [action_template(row["opaque_action_id"]) for row in packet]
    write_jsonl(paths["template_a"], templates, require_equal_if_exists=True)
    write_jsonl(paths["template_b"], templates, require_equal_if_exists=True)
    rubric = stage_spec("validation")["rubric"]
    manifest = {
        "name": "step3o_validation_packet_manifest_v1",
        "status": "untouched_transport_packet_frozen_before_actionability_labels",
        "primary_rows": len(selected),
        "hidden_duplicate_rows": len(duplicate_map),
        "packet_rows": len(packet),
        "assigned_category_counts": dict(
            sorted(Counter(row["assigned_category"] for row in selected).items())
        ),
        "active_rubric": repo_path(rubric),
        "active_rubric_sha256": sha256(rubric),
        "development_route_sha256": sha256(result_paths("development")["route"]),
        "transport_safety_route_sha256": sha256(result_paths("transport_safety")["route"]),
        "transport_safety_final_sha256": sha256(source_reviews["final"]),
        "artifacts": [
            artifact(paths["packet"]),
            artifact(paths["key"]),
            artifact(paths["duplicates"]),
            artifact(paths["template_a"]),
            artifact(paths["template_b"]),
        ],
        "validation_sessions_authorized": 1,
        "rubric_revisions_authorized": 0,
        "old_n0_rows_used": False,
        "heldout_content_read": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(paths["manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-o0")
    subparsers.add_parser("verify-o0")
    for name in ("freeze-ab", "prepare-c", "freeze-c", "finalize"):
        command = subparsers.add_parser(name)
        command.add_argument("--stage", choices=["development", "transport_safety", "validation"], required=True)
    subparsers.add_parser("freeze-validation")
    args = parser.parse_args()

    if args.command == "freeze-o0":
        result = freeze_o0()
    elif args.command == "verify-o0":
        result = verify_o0()
    elif args.command == "freeze-ab":
        result = freeze_ab(args.stage)
    elif args.command == "prepare-c":
        result = prepare_c(args.stage)
    elif args.command == "freeze-c":
        result = freeze_c(args.stage)
    elif args.command == "finalize":
        result = finalize_safety() if args.stage == "transport_safety" else finalize_action(args.stage)
    elif args.command == "freeze-validation":
        result = build_validation_packet()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
