#!/usr/bin/env python3
"""Build and qualify the Step3P pinned public paired-response target pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import step3o_o4_qualification as o4q


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = (
    ROOT
    / "experiments"
    / "step3_response_quality_persistence"
    / "configs"
    / "step3p_public_paired_pool_registry.json"
)

PUBLIC_FIELDS = {
    "prompt",
    "response",
    "prompt_source",
    "response_source",
    "is_safe",
    "harm_category",
    "severity_level",
    "sha256",
}
LOCAL_FIELDS = {
    "sample_id",
    "family",
    "source",
    "prompt",
    "response",
    "is_safe_source",
    "severity_level",
    "harm_category",
    "source_index",
    "content_hash",
}
VISIBLE_FIELDS = ("opaque_candidate_id", "prompt", "response")
DECISION_FIELDS = tuple(o4q.DECISION_FIELDS)
BINARY_FIELDS = tuple(o4q.BINARY_FIELDS)
REVIEW_FIELDS = ("opaque_candidate_id", *DECISION_FIELDS, "short_reason")
PROVENANCE_FIELDS = {
    "schema",
    "route_id",
    "role",
    "reviewer_id",
    "model",
    "context_isolated",
    "packet_sha256",
    "rubric_sha256",
    "review_sha256",
    "row_count",
    "prohibited_access",
}
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


class Step3PValidationError(ValueError):
    """Raised when a frozen Step3P boundary is violated."""


def _reject_constant(value: str) -> None:
    raise Step3PValidationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Step3PValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Step3PValidationError(f"invalid {context}: {exc}") from exc


def read_json_value(path: Path) -> Any:
    return parse_json(path.read_text(encoding="utf-8"), str(path))


def read_json(path: Path) -> dict[str, Any]:
    value = read_json_value(path)
    if type(value) is not dict:
        raise Step3PValidationError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n"):
                raise Step3PValidationError(f"{path}:{line_number} lacks LF termination")
            if raw.endswith("\r\n"):
                raise Step3PValidationError(f"{path}:{line_number} uses CRLF")
            value = parse_json(raw[:-1], f"{path}:{line_number}")
            if type(value) is not dict:
                raise Step3PValidationError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=False) + "\n" for row in rows
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise Step3PValidationError(f"missing frozen input: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise Step3PValidationError(f"byte-size mismatch for {path}")
    if sha256_file(path) != expected_sha256:
        raise Step3PValidationError(f"SHA-256 mismatch for {path}")


def resolve_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise Step3PValidationError(f"path escapes repository: {value}") from exc
    return path


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def whitespace_tokens(value: str) -> int:
    return len(value.split())


def stable_rank(namespace: str, *parts: object) -> str:
    material = "\x1f".join([namespace, *(str(part) for part in parts)])
    return sha256_text(material)


def selection_namespace(reg: Mapping[str, Any]) -> str:
    expected = sha256_text(
        "\x1f".join(
            [reg["route_id"], reg["entry_commit"], reg["source"]["public_sha256"]]
        )
    )
    randomization = reg.get("randomization")
    if (
        type(randomization) is not dict
        or randomization.get("derivation")
        != "sha256(route_id + U+001F + entry_commit + U+001F + public_source_sha256)"
        or randomization.get("fixed_salt_sha256") != expected
    ):
        raise Step3PValidationError("registry randomization salt is not reproducible")
    return expected


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise Step3PValidationError("quantile requires values")
    ordered = sorted(values)
    index = math.floor(probability * (len(ordered) - 1))
    return ordered[index]


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip().lower()
    if not HEX40.fullmatch(value):
        raise Step3PValidationError("git HEAD is not a full SHA-1")
    return value


def git_status_porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        raise Step3PValidationError("git ancestry check failed")
    return result.returncode == 0


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    reg = read_json(path)
    if reg.get("schema") != "step3p_public_paired_pool_registry_v1":
        raise Step3PValidationError("wrong Step3P registry schema")
    if reg.get("training_authorized") is not False or reg.get("optimizer_steps") != 0:
        raise Step3PValidationError("Step3P registry must keep training closed")
    categories = reg.get("selection", {}).get("categories")
    if type(categories) is not list or len(categories) != 19 or len(set(categories)) != 19:
        raise Step3PValidationError("registry must freeze 19 unique categories")
    severity = reg["selection"].get("severity_minima")
    if type(severity) is not dict or set(severity) != {"1", "2", "3"}:
        raise Step3PValidationError("severity minima must cover 1, 2, and 3")
    target = reg["selection"].get("target_count")
    mandatory = sum(int(value) for value in severity.values())
    if type(target) is not int or target <= mandatory:
        raise Step3PValidationError("target count must exceed mandatory severity capacity")
    if target <= len(categories) * int(reg["selection"]["category_minimum_each"]):
        raise Step3PValidationError("target count must exceed mandatory category capacity")
    full_distribution = reg["review"].get("full_distribution_fields")
    agreement_only = reg["review"].get("agreement_only_fields")
    if (
        type(full_distribution) is not list
        or type(agreement_only) is not list
        or set(full_distribution) | set(agreement_only) != set(BINARY_FIELDS)
        or set(full_distribution) & set(agreement_only)
    ):
        raise Step3PValidationError("registry binary field gate partition is invalid")
    selection_namespace(reg)
    return reg


def _validate_local_rows(reg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    cfg = reg["source"]
    path = resolve_repo_path(cfg["local_risk_path"])
    check_file(path, cfg["local_risk_sha256"])
    rows = read_jsonl(path)
    if len(rows) != int(cfg["local_risk_rows"]):
        raise Step3PValidationError("local risk row count changed")
    categories = set(reg["selection"]["categories"])
    by_prompt: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    ids: set[str] = set()
    for row in rows:
        if set(row) != LOCAL_FIELDS:
            raise Step3PValidationError("local risk schema changed")
        sample_id = row["sample_id"]
        prompt = row["prompt"]
        response = row["response"]
        source_index = row["source_index"]
        harm = row["harm_category"]
        if (
            type(sample_id) is not str
            or type(prompt) is not str
            or not prompt.strip()
            or type(response) is not str
            or not response.strip()
            or type(source_index) is not int
            or source_index <= 0
            or type(harm) is not dict
            or set(harm) != categories
            or any(type(value) is not bool for value in harm.values())
            or type(row["severity_level"]) is not int
            or row["severity_level"] not in {1, 2, 3}
            or row["is_safe_source"] is not False
        ):
            raise Step3PValidationError(f"invalid local risk row: {sample_id}")
        if row["content_hash"] != sha256_text(prompt + "\n" + response):
            raise Step3PValidationError(f"local content hash mismatch: {sample_id}")
        if sample_id in ids or prompt in by_prompt or source_index in by_index:
            raise Step3PValidationError("local risk IDs, prompts, or source indexes repeat")
        ids.add(sample_id)
        by_prompt[prompt] = row
        by_index[source_index] = row
    return rows, by_prompt, by_index


def _load_exclusions(reg: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    cfg = reg["frozen_inputs"]
    audit_path = resolve_repo_path(cfg["o4_source_audit_path"])
    hidden_path = resolve_repo_path(cfg["o4_hidden_key_path"])
    terminal_path = resolve_repo_path(cfg["o4r_terminal_path"])
    check_file(audit_path, cfg["o4_source_audit_sha256"])
    check_file(hidden_path, cfg["o4_hidden_key_sha256"])
    check_file(terminal_path, cfg["o4r_terminal_sha256"])

    terminal = read_json(terminal_path)
    if (
        terminal.get("route") != "STOP_AUTHORING"
        or terminal.get("training_authorized") is not False
        or terminal.get("optimizer_steps") != 0
    ):
        raise Step3PValidationError("O4R terminal boundary changed")
    audit = read_json(audit_path)
    excluded = set(audit.get("excluded_source_ids", []))
    lexical = audit.get("restricted_lexical_exclusions")
    if type(lexical) is not list:
        raise Step3PValidationError("O4 lexical exclusions are malformed")
    excluded.update(row["sample_id"] for row in lexical)

    with hidden_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        hidden_rows = list(reader)
    if len(hidden_rows) != int(cfg["o4_hidden_key_rows"]):
        raise Step3PValidationError("O4 hidden key row count changed")
    if any("sample_id" not in row or not row["sample_id"] for row in hidden_rows):
        raise Step3PValidationError("O4 hidden key lacks sample IDs")
    old_prompt_ids = {row["sample_id"] for row in hidden_rows}
    if len(old_prompt_ids) != len(hidden_rows):
        raise Step3PValidationError("O4 hidden key repeats source IDs")
    excluded.update(old_prompt_ids)
    return excluded, {
        "o4_registered_exclusions": len(audit["excluded_source_ids"]),
        "o4_lexical_exclusions": len(lexical),
        "o4_o4r_prompt_exclusions": len(old_prompt_ids),
        "exclusion_union": len(excluded),
    }


def _validate_public_row(row: Mapping[str, Any], line_number: int, categories: set[str]) -> None:
    if set(row) != PUBLIC_FIELDS:
        raise Step3PValidationError(f"public source schema mismatch at line {line_number}")
    if (
        type(row["prompt"]) is not str
        or not row["prompt"].strip()
        or type(row["response"]) is not str
        or type(row["prompt_source"]) is not str
        or type(row["response_source"]) is not str
        or type(row["is_safe"]) is not bool
        or type(row["harm_category"]) is not dict
        or set(row["harm_category"]) != categories
        or any(type(value) is not bool for value in row["harm_category"].values())
        or type(row["severity_level"]) is not int
        or row["severity_level"] not in {0, 1, 2, 3}
        or type(row["sha256"]) is not str
        or not HEX64.fullmatch(row["sha256"])
    ):
        raise Step3PValidationError(f"invalid public source row at line {line_number}")


def collect_public_candidates(
    reg: Mapping[str, Any], public_source: Path, protocol_commit: str
) -> dict[str, Any]:
    if not HEX40.fullmatch(protocol_commit):
        raise Step3PValidationError("protocol commit must be a full lowercase SHA-1")
    source_cfg = reg["source"]
    check_file(
        public_source,
        source_cfg["public_sha256"],
        int(source_cfg["public_bytes"]),
    )
    local_rows, local_by_prompt, local_by_index = _validate_local_rows(reg)
    excluded, exclusion_summary = _load_exclusions(reg)
    categories = set(reg["selection"]["categories"])
    minimum_prompt_tokens = int(reg["eligibility"]["minimum_prompt_whitespace_tokens"])
    maximum_prompt_tokens = int(reg["eligibility"]["maximum_prompt_whitespace_tokens"])
    minimum_response_tokens = int(reg["eligibility"]["minimum_response_whitespace_tokens"])
    maximum_response_tokens = int(reg["eligibility"]["maximum_response_whitespace_tokens"])

    eligible_local = {
        prompt: row
        for prompt, row in local_by_prompt.items()
        if row["sample_id"] not in excluded
        and minimum_prompt_tokens <= whitespace_tokens(prompt) <= maximum_prompt_tokens
    }
    raw_candidates: list[dict[str, Any]] = []
    matched_safe_rows = 0
    empty_safe_rows = 0
    response_length_exclusions = 0
    aligned_indexes: set[int] = set()
    line_count = 0
    with public_source.open("r", encoding="utf-8", newline="") as handle:
        for line_count, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or raw.endswith("\r\n"):
                raise Step3PValidationError(f"public source line ending changed at {line_count}")
            row = parse_json(raw[:-1], f"public source line {line_count}")
            if type(row) is not dict:
                raise Step3PValidationError(f"public source line {line_count} is not an object")
            _validate_public_row(row, line_count, categories)
            local = local_by_index.get(line_count)
            if local is not None:
                if (
                    row["prompt"] != local["prompt"]
                    or row["response"].strip() != local["response"]
                    or row["is_safe"] is not False
                    or row["severity_level"] != local["severity_level"]
                    or row["harm_category"] != local["harm_category"]
                ):
                    raise Step3PValidationError(
                        f"local/public source alignment failed at public line {line_count}"
                    )
                aligned_indexes.add(line_count)

            local = eligible_local.get(row["prompt"])
            if local is None or row["is_safe"] is not True:
                continue
            if row["response_source"] != source_cfg["expected_response_source"]:
                raise Step3PValidationError(
                    f"safe candidate has unexpected response source at line {line_count}"
                )
            response = row["response"].strip()
            matched_safe_rows += 1
            if not response:
                empty_safe_rows += 1
                continue
            response_tokens = whitespace_tokens(response)
            if not minimum_response_tokens <= response_tokens <= maximum_response_tokens:
                response_length_exclusions += 1
                continue
            normalized = normalize_text(response)
            if not normalized:
                raise Step3PValidationError(f"non-empty response normalized empty at line {line_count}")
            raw_candidates.append(
                {
                    "sample_id": local["sample_id"],
                    "source_index": local["source_index"],
                    "public_source_line": line_count,
                    "prompt": local["prompt"],
                    "response": response,
                    "severity_level": local["severity_level"],
                    "categories": sorted(
                        category for category, present in local["harm_category"].items() if present
                    ),
                    "prompt_sha256": sha256_text(local["prompt"]),
                    "response_sha256": sha256_text(response),
                    "normalized_response_sha256": sha256_text(normalized),
                    "prompt_whitespace_tokens": whitespace_tokens(local["prompt"]),
                    "response_whitespace_tokens": response_tokens,
                }
            )

    if line_count != int(source_cfg["public_rows"]):
        raise Step3PValidationError("public source row count changed")
    if aligned_indexes != set(local_by_index):
        raise Step3PValidationError("not every local source index was aligned")

    namespace = selection_namespace(reg)
    by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in raw_candidates:
        by_response[candidate["normalized_response_sha256"]].append(candidate)
    globally_unique: list[dict[str, Any]] = []
    for response_hash, group in sorted(by_response.items()):
        globally_unique.append(
            min(
                group,
                key=lambda row: stable_rank(
                    namespace,
                    "global_response_dedup",
                    response_hash,
                    row["sample_id"],
                    row["public_source_line"],
                ),
            )
        )

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in globally_unique:
        by_prompt[candidate["sample_id"]].append(candidate)
    retained: list[dict[str, Any]] = []
    maximum_per_prompt = int(reg["eligibility"]["maximum_candidates_per_prompt"])
    for sample_id, group in sorted(by_prompt.items()):
        retained.extend(
            sorted(
                group,
                key=lambda row: stable_rank(
                    namespace,
                    "within_prompt_candidate",
                    sample_id,
                    row["public_source_line"],
                    row["response_sha256"],
                ),
            )[:maximum_per_prompt]
        )

    primary: list[dict[str, Any]] = []
    prompt_ids: dict[str, str] = {}
    for candidate in sorted(retained, key=lambda row: (row["sample_id"], row["public_source_line"])):
        sample_id = candidate["sample_id"]
        prompt_id = prompt_ids.setdefault(
            sample_id,
            "step3p_prompt_"
            + stable_rank(namespace, "prompt_id", sample_id, candidate["prompt_sha256"])[:20],
        )
        candidate_id = "step3p_candidate_" + stable_rank(
            namespace,
            "candidate_id",
            sample_id,
            candidate["public_source_line"],
            candidate["response_sha256"],
        )[:24]
        primary.append(
            {
                "opaque_candidate_id": candidate_id,
                "opaque_prompt_id": prompt_id,
                **candidate,
            }
        )

    if len(primary) < int(reg["eligibility"]["minimum_primary_candidates"]):
        raise Step3PValidationError("too few deterministic primary candidates")
    if len({row["opaque_prompt_id"] for row in primary}) < int(
        reg["eligibility"]["minimum_unique_candidate_prompts"]
    ):
        raise Step3PValidationError("too few deterministic candidate prompts")
    if len({row["opaque_candidate_id"] for row in primary}) != len(primary):
        raise Step3PValidationError("candidate IDs collide")
    if len({row["normalized_response_sha256"] for row in primary}) != len(primary):
        raise Step3PValidationError("normalized response deduplication failed")

    preflight_selection = select_targets(primary, reg, namespace=namespace)
    if not preflight_selection["pass"]:
        raise Step3PValidationError("pre-review constrained target flow is infeasible")

    duplicate_count = int(reg["review"]["hidden_duplicate_count"])
    if duplicate_count >= len(primary):
        raise Step3PValidationError("hidden duplicate count is too large")
    duplicate_source = sorted(
        primary,
        key=lambda row: stable_rank(namespace, "duplicate_selection", row["opaque_candidate_id"]),
    )[:duplicate_count]
    packets: dict[str, list[dict[str, Any]]] = {}
    duplicate_links: dict[str, list[dict[str, str]]] = {}
    primary_ids = {row["opaque_candidate_id"] for row in primary}
    for role in ("A", "B"):
        visible = [{field: row[field] for field in VISIBLE_FIELDS} for row in primary]
        links: list[dict[str, str]] = []
        for row in duplicate_source:
            duplicate_id = "step3p_candidate_" + stable_rank(
                namespace, "duplicate_id", role, row["opaque_candidate_id"]
            )[:24]
            if duplicate_id in primary_ids:
                raise Step3PValidationError("duplicate ID collides with primary ID")
            visible.append(
                {
                    "opaque_candidate_id": duplicate_id,
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
            )
            links.append(
                {
                    "primary_candidate_id": row["opaque_candidate_id"],
                    "duplicate_candidate_id": duplicate_id,
                }
            )
        packets[role] = sorted(
            visible,
            key=lambda row: stable_rank(
                namespace, f"reviewer_{role.lower()}_packet_order", row["opaque_candidate_id"]
            ),
        )
        duplicate_links[role] = links
    primary_order_a = [row["opaque_candidate_id"] for row in packets["A"] if row["opaque_candidate_id"] in primary_ids]
    primary_order_b = [row["opaque_candidate_id"] for row in packets["B"] if row["opaque_candidate_id"] in primary_ids]
    if primary_order_a == primary_order_b:
        raise Step3PValidationError("A/B primary packet orders unexpectedly match")

    category_counts = {
        category: sum(category in row["categories"] for row in primary)
        for category in reg["selection"]["categories"]
    }
    severity_counts = {
        str(severity): sum(row["severity_level"] == severity for row in primary)
        for severity in (1, 2, 3)
    }
    token_lengths = sorted(row["response_whitespace_tokens"] for row in primary)
    diagnostics = {
        "name": "step3p_public_source_preflight_v1",
        "route_id": reg["route_id"],
        "protocol_commit": protocol_commit,
        "randomization_salt_sha256": namespace,
        "public_source": {
            "revision": source_cfg["public_revision"],
            "bytes": public_source.stat().st_size,
            "rows": line_count,
            "sha256": source_cfg["public_sha256"],
        },
        "local_source_rows_aligned": len(aligned_indexes),
        "eligible_local_prompts_before_public_safe_match": len(eligible_local),
        "matched_public_safe_rows": matched_safe_rows,
        "empty_public_safe_rows": empty_safe_rows,
        "response_length_exclusions": response_length_exclusions,
        "length_qualified_public_safe_candidate_rows": len(raw_candidates),
        "normalized_unique_candidate_rows": len(globally_unique),
        "retained_primary_candidates": len(primary),
        "retained_unique_prompts": len({row["opaque_prompt_id"] for row in primary}),
        "response_tokens": {
            "minimum": token_lengths[0],
            "median": token_lengths[len(token_lengths) // 2],
            "maximum": token_lengths[-1],
        },
        "candidate_category_counts": category_counts,
        "candidate_severity_counts": severity_counts,
        "pre_review_flow": preflight_selection["summary"],
        "exclusions": exclusion_summary,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    return {
        "namespace": namespace,
        "primary": primary,
        "packets": packets,
        "duplicate_links": duplicate_links,
        "diagnostics": diagnostics,
    }


def _add_edge(graph: dict[str, list[str]], capacity: dict[tuple[str, str], int], left: str, right: str, amount: int) -> None:
    if amount < 0:
        raise Step3PValidationError("negative flow capacity")
    if (left, right) in capacity:
        raise Step3PValidationError(f"duplicate flow edge: {left} -> {right}")
    graph[left].append(right)
    graph[right].append(left)
    capacity[(left, right)] = amount
    capacity[(right, left)] = 0


def _maximum_flow(
    graph: dict[str, list[str]], capacity: dict[tuple[str, str], int], source: str, sink: str
) -> tuple[int, dict[tuple[str, str], int]]:
    residual = dict(capacity)
    total = 0
    while True:
        parent: dict[str, str | None] = {source: None}
        queue: deque[str] = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in parent and residual[(node, neighbor)] > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
        if sink not in parent:
            break
        amount = 1 << 30
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            assert previous is not None
            amount = min(amount, residual[(previous, node)])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            assert previous is not None
            residual[(previous, node)] -= amount
            residual[(node, previous)] += amount
            node = previous
        total += amount
    return total, residual


def select_targets(
    candidates: Sequence[Mapping[str, Any]], reg: Mapping[str, Any], *, namespace: str
) -> dict[str, Any]:
    target_count = int(reg["selection"]["target_count"])
    severity_minima = {
        int(key): int(value) for key, value in reg["selection"]["severity_minima"].items()
    }
    categories = list(reg["selection"]["categories"])
    category_minimum = int(reg["selection"]["category_minimum_each"])
    severity_any = target_count - sum(severity_minima.values())
    category_any = target_count - category_minimum * len(categories)
    if severity_any < 0 or category_any < 0:
        raise Step3PValidationError("selection capacities exceed target count")

    candidate_rows = sorted(
        (dict(row) for row in candidates),
        key=lambda row: stable_rank(namespace, "flow_candidate_order", row["opaque_candidate_id"]),
    )
    if len({row["opaque_candidate_id"] for row in candidate_rows}) != len(candidate_rows):
        raise Step3PValidationError("flow candidates repeat IDs")
    if len({row["normalized_response_sha256"] for row in candidate_rows}) != len(candidate_rows):
        raise Step3PValidationError("flow candidates repeat normalized responses")

    graph: dict[str, list[str]] = defaultdict(list)
    capacity: dict[tuple[str, str], int] = {}
    source = "flow_source"
    sink = "flow_sink"
    for severity in (1, 2, 3):
        _add_edge(
            graph,
            capacity,
            source,
            f"severity_mandatory_{severity}",
            severity_minima[severity],
        )
    _add_edge(graph, capacity, source, "severity_any", severity_any)

    prompt_meta: dict[str, tuple[int, tuple[str, ...]]] = {}
    for row in candidate_rows:
        candidate_id = row["opaque_candidate_id"]
        prompt_id = row["opaque_prompt_id"]
        severity = row["severity_level"]
        row_categories = tuple(row["categories"])
        if severity not in {1, 2, 3} or not row_categories or not set(row_categories) <= set(categories):
            raise Step3PValidationError(f"invalid flow metadata for {candidate_id}")
        prior = prompt_meta.setdefault(prompt_id, (severity, row_categories))
        if prior != (severity, row_categories):
            raise Step3PValidationError("candidate metadata differs within a prompt")
        _add_edge(graph, capacity, f"severity_mandatory_{severity}", f"candidate::{candidate_id}", 1)
        _add_edge(graph, capacity, "severity_any", f"candidate::{candidate_id}", 1)
        _add_edge(graph, capacity, f"candidate::{candidate_id}", f"prompt_in::{prompt_id}", 1)

    for prompt_id, (_, row_categories) in sorted(prompt_meta.items()):
        _add_edge(graph, capacity, f"prompt_in::{prompt_id}", f"prompt_out::{prompt_id}", 1)
        for category in sorted(
            row_categories,
            key=lambda value: stable_rank(namespace, "flow_category_order", prompt_id, value),
        ):
            _add_edge(
                graph,
                capacity,
                f"prompt_out::{prompt_id}",
                f"category_mandatory::{category}",
                1,
            )
        _add_edge(graph, capacity, f"prompt_out::{prompt_id}", "category_any", 1)
    for category in categories:
        _add_edge(
            graph,
            capacity,
            f"category_mandatory::{category}",
            sink,
            category_minimum,
        )
    _add_edge(graph, capacity, "category_any", sink, category_any)
    for node in graph:
        graph[node].sort(key=lambda value: stable_rank(namespace, "flow_edge_order", node, value))

    flow, residual = _maximum_flow(graph, capacity, source, sink)
    selected: list[dict[str, Any]] = []
    assigned_categories: dict[str, str | None] = {}
    if flow == target_count:
        row_by_id = {row["opaque_candidate_id"]: row for row in candidate_rows}
        for candidate_id, row in row_by_id.items():
            edge = (f"candidate::{candidate_id}", f"prompt_in::{row['opaque_prompt_id']}")
            if capacity[edge] == 1 and residual[edge] == 0:
                selected.append(row)
        if len(selected) != target_count:
            raise Step3PValidationError("flow accounting did not recover target count")
        for row in selected:
            prompt_out = f"prompt_out::{row['opaque_prompt_id']}"
            assigned: str | None = None
            for category in row["categories"]:
                edge = (prompt_out, f"category_mandatory::{category}")
                if edge in capacity and capacity[edge] == 1 and residual[edge] == 0:
                    assigned = category
                    break
            assigned_categories[row["opaque_candidate_id"]] = assigned

    severity_counts = {
        str(severity): sum(row["severity_level"] == severity for row in selected)
        for severity in (1, 2, 3)
    }
    mandatory_category_counts = {
        category: sum(assigned_categories.get(row["opaque_candidate_id"]) == category for row in selected)
        for category in categories
    }
    passed = (
        flow == target_count
        and len(selected) == target_count
        and len({row["opaque_prompt_id"] for row in selected}) == target_count
        and len({row["normalized_response_sha256"] for row in selected}) == target_count
        and all(severity_counts[str(key)] >= value for key, value in severity_minima.items())
        and all(value >= category_minimum for value in mandatory_category_counts.values())
    )
    return {
        "pass": passed,
        "selected": selected,
        "assigned_categories": assigned_categories,
        "summary": {
            "maximum_flow": flow,
            "target_count": target_count,
            "selected_count": len(selected),
            "unique_prompts": len({row["opaque_prompt_id"] for row in selected}),
            "unique_normalized_responses": len(
                {row["normalized_response_sha256"] for row in selected}
            ),
            "severity_counts": severity_counts,
            "mandatory_category_counts": mandatory_category_counts,
            "pass": passed,
        },
    }


def _template_rows(packet: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "opaque_candidate_id": row["opaque_candidate_id"],
            "detail_retention": None,
            "safe_route": "",
            "executable_step": "",
            "prompt_binding": "",
            "naturalness": "",
            "clean_safe": "",
            "short_reason": "",
        }
        for row in packet
    ]


def _provenance_template(
    reg: Mapping[str, Any], role: str, packet_bytes: bytes, rubric_bytes: bytes, row_count: int
) -> dict[str, Any]:
    return {
        "schema": "step3p_reviewer_provenance_v1",
        "route_id": reg["route_id"],
        "role": role,
        "reviewer_id": "",
        "model": "",
        "context_isolated": True,
        "packet_sha256": sha256_bytes(packet_bytes),
        "rubric_sha256": sha256_bytes(rubric_bytes),
        "review_sha256": "",
        "row_count": row_count,
        "prohibited_access": False,
    }


def build_prepare_files(package: Mapping[str, Any], reg: Mapping[str, Any]) -> dict[str, bytes]:
    primary = package["primary"]
    packet_a = jsonl_text(package["packets"]["A"]).encode("utf-8")
    packet_b = jsonl_text(package["packets"]["B"]).encode("utf-8")
    rubric_path = resolve_repo_path(reg["frozen_inputs"]["qualification_rubric_path"])
    rubric_bytes = rubric_path.read_bytes()
    files: dict[str, bytes] = {
        "diagnostics/step3p_source_preflight.json": json_text(package["diagnostics"]).encode("utf-8"),
        "packets/primary_candidates_hidden.jsonl": jsonl_text(primary).encode("utf-8"),
        "packets/reviewer_a_packet.jsonl": packet_a,
        "packets/reviewer_b_packet.jsonl": packet_b,
        "packets/reviewer_a_duplicate_links.json": json_text(package["duplicate_links"]["A"]).encode("utf-8"),
        "packets/reviewer_b_duplicate_links.json": json_text(package["duplicate_links"]["B"]).encode("utf-8"),
        "templates/reviewer_a_review.jsonl": jsonl_text(_template_rows(package["packets"]["A"])).encode("utf-8"),
        "templates/reviewer_b_review.jsonl": jsonl_text(_template_rows(package["packets"]["B"])).encode("utf-8"),
    }
    files["templates/reviewer_a_provenance.json"] = json_text(
        _provenance_template(reg, "reviewer_a", packet_a, rubric_bytes, len(package["packets"]["A"]))
    ).encode("utf-8")
    files["templates/reviewer_b_provenance.json"] = json_text(
        _provenance_template(reg, "reviewer_b", packet_b, rubric_bytes, len(package["packets"]["B"]))
    ).encode("utf-8")
    manifest = {
        "name": "step3p_public_paired_packet_manifest_v1",
        "route_id": reg["route_id"],
        "protocol_commit": package["diagnostics"]["protocol_commit"],
        "randomization_salt_sha256": package["diagnostics"]["randomization_salt_sha256"],
        "primary_candidates": len(primary),
        "unique_prompts": len({row["opaque_prompt_id"] for row in primary}),
        "hidden_duplicates_per_reviewer": int(reg["review"]["hidden_duplicate_count"]),
        "artifacts": [
            {"path": name, "bytes": len(value), "sha256": sha256_bytes(value)}
            for name, value in sorted(files.items())
        ],
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["packet_manifest.json"] = json_text(manifest).encode("utf-8")
    return files


def write_bundle(output_root: Path, files: Mapping[str, bytes]) -> None:
    if output_root.exists():
        raise Step3PValidationError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".step3p-", dir=output_root.parent))
    try:
        for relative, content in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        staging.replace(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_review_rows(rows: Sequence[Mapping[str, Any]], packet: Sequence[Mapping[str, Any]], role: str) -> list[dict[str, Any]]:
    expected_ids = [row["opaque_candidate_id"] for row in packet]
    if len(rows) != len(expected_ids):
        raise Step3PValidationError(f"{role} review row count mismatch")
    validated: list[dict[str, Any]] = []
    for index, (row, expected_id) in enumerate(zip(rows, expected_ids), 1):
        if set(row) != set(REVIEW_FIELDS) or row["opaque_candidate_id"] != expected_id:
            raise Step3PValidationError(f"{role} review schema or order mismatch at row {index}")
        if type(row["detail_retention"]) is not int or row["detail_retention"] not in {0, 1, 2}:
            raise Step3PValidationError(f"{role} invalid detail label at row {index}")
        for field in BINARY_FIELDS:
            if type(row[field]) is not str or row[field] not in {"yes", "no"}:
                raise Step3PValidationError(f"{role} invalid {field} at row {index}")
        if type(row["short_reason"]) is not str or not row["short_reason"].strip():
            raise Step3PValidationError(f"{role} lacks a short reason at row {index}")
        validated.append(dict(row))
    return validated


def validate_provenance(
    value: Mapping[str, Any], *, reg: Mapping[str, Any], role: str, packet_bytes: bytes, rubric_bytes: bytes, review_bytes: bytes, row_count: int
) -> None:
    if set(value) != PROVENANCE_FIELDS:
        raise Step3PValidationError(f"{role} provenance schema mismatch")
    if (
        value["schema"] != "step3p_reviewer_provenance_v1"
        or value["route_id"] != reg["route_id"]
        or value["role"] != role
        or type(value["reviewer_id"]) is not str
        or not value["reviewer_id"].strip()
        or type(value["model"]) is not str
        or not value["model"].strip()
        or value["context_isolated"] is not True
        or value["packet_sha256"] != sha256_bytes(packet_bytes)
        or value["rubric_sha256"] != sha256_bytes(rubric_bytes)
        or value["review_sha256"] != sha256_bytes(review_bytes)
        or value["row_count"] != row_count
        or value["prohibited_access"] is not False
    ):
        raise Step3PValidationError(f"{role} provenance validation failed")


def _metric_pass(value: Any, threshold: float, *, upper: bool = False) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and (
        float(value) <= threshold if upper else float(value) >= threshold
    )


def _prompt_bootstrap_ac1(
    primary: Sequence[Mapping[str, Any]], review_a: Mapping[str, Mapping[str, Any]], review_b: Mapping[str, Mapping[str, Any]], reg: Mapping[str, Any], namespace: str
) -> dict[str, Any]:
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for row in primary:
        by_prompt[row["opaque_prompt_id"]].append(row["opaque_candidate_id"])
    prompt_ids = sorted(by_prompt)
    seed = int(stable_rank(namespace, "prompt_reweighting_seed")[:16], 16)
    rng = random.Random(seed)
    values: list[float] = []
    replicates = int(reg["review"]["prompt_reweighting_replicates"])
    for _ in range(replicates):
        pairs: list[tuple[str, str]] = []
        for _ in prompt_ids:
            sampled = prompt_ids[rng.randrange(len(prompt_ids))]
            for candidate_id in by_prompt[sampled]:
                pairs.append(
                    (
                        o4q.candidate_qualified(review_a[candidate_id]),
                        o4q.candidate_qualified(review_b[candidate_id]),
                    )
                )
        value = o4q.binary_agreement_metrics(pairs)["gwet_ac1"]
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if len(values) != replicates:
        raise Step3PValidationError("prompt bootstrap produced a non-finite replicate")
    return {
        "replicates": replicates,
        "seed_sha256": sha256_text(str(seed)),
        "q025": quantile(values, 0.025),
        "median": quantile(values, 0.5),
        "q975": quantile(values, 0.975),
    }


def compute_ab_reliability(
    primary: Sequence[Mapping[str, Any]], packet_a: Sequence[Mapping[str, Any]], packet_b: Sequence[Mapping[str, Any]], links_a: Sequence[Mapping[str, str]], links_b: Sequence[Mapping[str, str]], rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], reg: Mapping[str, Any], namespace: str
) -> dict[str, Any]:
    map_a = {row["opaque_candidate_id"]: row for row in rows_a}
    map_b = {row["opaque_candidate_id"]: row for row in rows_b}
    primary_ids = [row["opaque_candidate_id"] for row in primary]
    if not set(primary_ids) <= set(map_a) or not set(primary_ids) <= set(map_b):
        raise Step3PValidationError("A/B reviews lack primary candidates")

    duplicate_consistency: dict[str, Any] = {}
    for role, review_map, links in (("A", map_a, links_a), ("B", map_b, links_b)):
        counts = []
        for link in links:
            primary_row = review_map[link["primary_candidate_id"]]
            duplicate_row = review_map[link["duplicate_candidate_id"]]
            counts.append(all(primary_row[field] == duplicate_row[field] for field in DECISION_FIELDS))
        duplicate_consistency[role] = {
            "consistent": sum(counts),
            "denominator": len(counts),
        }

    detail = o4q.detail_agreement_metrics(
        [(map_a[cid]["detail_retention"], map_b[cid]["detail_retention"]) for cid in primary_ids]
    )
    components = {
        field: o4q.binary_agreement_metrics([(map_a[cid][field], map_b[cid][field]) for cid in primary_ids])
        for field in BINARY_FIELDS
    }
    qualified = o4q.binary_agreement_metrics(
        [
            (o4q.candidate_qualified(map_a[cid]), o4q.candidate_qualified(map_b[cid]))
            for cid in primary_ids
        ]
    )
    full_vector_exact = sum(
        all(map_a[cid][field] == map_b[cid][field] for field in DECISION_FIELDS)
        for cid in primary_ids
    ) / len(primary_ids)
    bootstrap = _prompt_bootstrap_ac1(primary, map_a, map_b, reg, namespace)
    gates_cfg = reg["review"]
    detail_boundary = detail["qualified_boundary_2_vs_0_or_1"]
    gates: dict[str, bool] = {
        "detail_exact": _metric_pass(detail["exact_agreement"], gates_cfg["detail_exact_agreement_min"]),
        "detail_weighted": _metric_pass(detail["linear_weighted_agreement"], gates_cfg["detail_weighted_agreement_min"]),
        "detail_ac2": _metric_pass(detail["gwet_ac2"], gates_cfg["detail_gwet_ac2_min"]),
        "detail_nondegenerate": detail["nondegenerate"] is True,
        "detail_boundary_raw": _metric_pass(detail_boundary["raw_agreement"], gates_cfg["detail_boundary_raw_agreement_min"]),
        "detail_boundary_ac1": _metric_pass(detail_boundary["gwet_ac1"], gates_cfg["detail_boundary_gwet_ac1_min"]),
        "detail_boundary_positive": _metric_pass(detail_boundary["positive_agreement"], gates_cfg["detail_boundary_positive_agreement_min"]),
        "detail_boundary_negative": _metric_pass(detail_boundary["negative_agreement"], gates_cfg["detail_boundary_negative_agreement_min"]),
        "detail_boundary_prevalence_gap": _metric_pass(detail_boundary["prevalence_gap"], gates_cfg["detail_boundary_prevalence_gap_max"], upper=True),
        "detail_boundary_yes_count": detail_boundary["yes_ratings"] >= gates_cfg["detail_boundary_min_yes_ratings"],
        "detail_boundary_no_count": detail_boundary["no_ratings"] >= gates_cfg["detail_boundary_min_no_ratings"],
        "full_vector_exact": _metric_pass(full_vector_exact, gates_cfg["full_vector_exact_agreement_min"]),
        "qualified_raw": _metric_pass(qualified["raw_agreement"], gates_cfg["candidate_qualified_raw_agreement_min"]),
        "qualified_ac1": _metric_pass(qualified["gwet_ac1"], gates_cfg["candidate_qualified_gwet_ac1_min"]),
        "qualified_positive": _metric_pass(qualified["positive_agreement"], gates_cfg["candidate_qualified_positive_agreement_min"]),
        "qualified_negative": _metric_pass(qualified["negative_agreement"], gates_cfg["candidate_qualified_negative_agreement_min"]),
        "qualified_prevalence_gap": _metric_pass(qualified["prevalence_gap"], gates_cfg["candidate_qualified_prevalence_gap_max"], upper=True),
        "qualified_yes_count": qualified["yes_ratings"] >= gates_cfg["candidate_qualified_min_yes_ratings"],
        "qualified_no_count": qualified["no_ratings"] >= gates_cfg["candidate_qualified_min_no_ratings"],
        "qualified_prompt_bootstrap_q025": _metric_pass(bootstrap["q025"], gates_cfg["candidate_qualified_prompt_reweighting_q025_min"]),
    }
    full_distribution_fields = set(gates_cfg["full_distribution_fields"])
    agreement_only_fields = set(gates_cfg["agreement_only_fields"])
    if full_distribution_fields | agreement_only_fields != set(BINARY_FIELDS) or full_distribution_fields & agreement_only_fields:
        raise Step3PValidationError("binary field gate partition is invalid")
    for field, metrics in components.items():
        raw_min = gates_cfg["naturalness_raw_agreement_min"] if field == "naturalness" else gates_cfg["clean_safe_raw_agreement_min"] if field == "clean_safe" else gates_cfg["component_raw_agreement_min"]
        ac1_min = gates_cfg["naturalness_gwet_ac1_min"] if field == "naturalness" else gates_cfg["clean_safe_gwet_ac1_min"] if field == "clean_safe" else gates_cfg["component_gwet_ac1_min"]
        gates[f"{field}_raw"] = _metric_pass(metrics["raw_agreement"], raw_min)
        gates[f"{field}_ac1"] = _metric_pass(metrics["gwet_ac1"], ac1_min)
        gates[f"{field}_prevalence_gap"] = _metric_pass(metrics["prevalence_gap"], gates_cfg["component_prevalence_gap_max"], upper=True)
        if field in full_distribution_fields:
            gates[f"{field}_positive"] = _metric_pass(metrics["positive_agreement"], gates_cfg["component_positive_agreement_min"])
            gates[f"{field}_negative"] = _metric_pass(metrics["negative_agreement"], gates_cfg["component_negative_agreement_min"])
            gates[f"{field}_yes_count"] = metrics["yes_ratings"] >= gates_cfg["component_min_yes_ratings"]
            gates[f"{field}_no_count"] = metrics["no_ratings"] >= gates_cfg["component_min_no_ratings"]
    duplicate_min = int(gates_cfg["duplicate_full_vector_consistency_min_count"])
    gates["reviewer_a_duplicates"] = duplicate_consistency["A"]["consistent"] >= duplicate_min
    gates["reviewer_b_duplicates"] = duplicate_consistency["B"]["consistent"] >= duplicate_min
    return {
        "primary_candidates": len(primary_ids),
        "detail": detail,
        "components": components,
        "candidate_qualified": qualified,
        "full_vector_exact_agreement": full_vector_exact,
        "prompt_cluster_bootstrap": bootstrap,
        "hidden_duplicate_consistency": duplicate_consistency,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def load_prepare_bundle(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "packet_manifest.json"
    manifest = read_json(manifest_path)
    for artifact in manifest["artifacts"]:
        path = output_root / artifact["path"]
        check_file(path, artifact["sha256"], int(artifact["bytes"]))
    return {
        "manifest": manifest,
        "primary": read_jsonl(output_root / "packets" / "primary_candidates_hidden.jsonl"),
        "packet_a": read_jsonl(output_root / "packets" / "reviewer_a_packet.jsonl"),
        "packet_b": read_jsonl(output_root / "packets" / "reviewer_b_packet.jsonl"),
        "links_a": read_json_value(output_root / "packets" / "reviewer_a_duplicate_links.json"),
        "links_b": read_json_value(output_root / "packets" / "reviewer_b_duplicate_links.json"),
    }


def _load_and_validate_review(
    review_path: Path, provenance_path: Path, packet_path: Path, packet: Sequence[Mapping[str, Any]], reg: Mapping[str, Any], role: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    review_bytes = review_path.read_bytes()
    packet_bytes = packet_path.read_bytes()
    rubric_bytes = resolve_repo_path(reg["frozen_inputs"]["qualification_rubric_path"]).read_bytes()
    rows = _validate_review_rows(read_jsonl(review_path), packet, role)
    provenance = read_json(provenance_path)
    validate_provenance(
        provenance,
        reg=reg,
        role=role,
        packet_bytes=packet_bytes,
        rubric_bytes=rubric_bytes,
        review_bytes=review_bytes,
        row_count=len(rows),
    )
    return rows, provenance


def validate_reviewer_separation(*records: Mapping[str, Any]) -> None:
    reviewer_ids = [record["reviewer_id"] for record in records]
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise Step3PValidationError("reviewer session IDs must be distinct across roles")


def build_c_files(
    bundle: Mapping[str, Any], rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], reliability: Mapping[str, Any], reg: Mapping[str, Any]
) -> dict[str, bytes]:
    primary = bundle["primary"]
    map_a = {row["opaque_candidate_id"]: row for row in rows_a}
    map_b = {row["opaque_candidate_id"]: row for row in rows_b}
    namespace = selection_namespace(reg)
    visible_by_id = {
        row["opaque_candidate_id"]: {field: row[field] for field in VISIBLE_FIELDS}
        for row in primary
    }
    disagreement_ids = [
        row["opaque_candidate_id"]
        for row in primary
        if o4q.candidate_qualified(map_a[row["opaque_candidate_id"]])
        != o4q.candidate_qualified(map_b[row["opaque_candidate_id"]])
    ]
    packet = sorted(
        (visible_by_id[candidate_id] for candidate_id in disagreement_ids),
        key=lambda row: stable_rank(namespace, "reviewer_c_packet_order", row["opaque_candidate_id"]),
    )
    packet_bytes = jsonl_text(packet).encode("utf-8")
    rubric_bytes = resolve_repo_path(reg["frozen_inputs"]["qualification_rubric_path"]).read_bytes()
    files = {
        "ab/ab_reliability.json": json_text(reliability).encode("utf-8"),
        "ab/reviewer_c_packet.jsonl": packet_bytes,
        "ab/reviewer_c_template.jsonl": jsonl_text(_template_rows(packet)).encode("utf-8"),
        "ab/reviewer_c_provenance_template.json": json_text(
            _provenance_template(reg, "reviewer_c", packet_bytes, rubric_bytes, len(packet))
        ).encode("utf-8"),
    }
    route = {
        "route": "OPEN_REVIEWER_C" if reliability["overall_pass"] else "STOP_REVIEW_UNRELIABLE",
        "reviewer_c_rows": len(packet) if reliability["overall_pass"] else 0,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["ab/ab_route.json"] = json_text(route).encode("utf-8")
    if not reliability["overall_pass"]:
        files.pop("ab/reviewer_c_packet.jsonl")
        files.pop("ab/reviewer_c_template.jsonl")
        files.pop("ab/reviewer_c_provenance_template.json")
    return files


def _adjudicate(
    primary: Sequence[Mapping[str, Any]], rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], rows_c: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    map_a = {row["opaque_candidate_id"]: row for row in rows_a}
    map_b = {row["opaque_candidate_id"]: row for row in rows_b}
    map_c = {row["opaque_candidate_id"]: row for row in rows_c}
    final: list[dict[str, Any]] = []
    expected_c: set[str] = set()
    for candidate in primary:
        candidate_id = candidate["opaque_candidate_id"]
        label_a = o4q.candidate_qualified(map_a[candidate_id])
        label_b = o4q.candidate_qualified(map_b[candidate_id])
        if label_a != label_b:
            expected_c.add(candidate_id)
            if candidate_id not in map_c:
                raise Step3PValidationError("Reviewer C lacks a disagreement candidate")
            row = {field: map_c[candidate_id][field] for field in DECISION_FIELDS}
            adjudicated_fields = list(DECISION_FIELDS)
        else:
            row = {"detail_retention": min(map_a[candidate_id]["detail_retention"], map_b[candidate_id]["detail_retention"])}
            row.update(
                {
                    field: "yes" if map_a[candidate_id][field] == map_b[candidate_id][field] == "yes" else "no"
                    for field in BINARY_FIELDS
                }
            )
            adjudicated_fields = []
        final.append(
            {
                "opaque_candidate_id": candidate_id,
                **row,
                "candidate_qualified": o4q.candidate_qualified(row),
                "adjudicated_fields": adjudicated_fields,
            }
        )
    if set(map_c) != expected_c:
        raise Step3PValidationError("Reviewer C row set differs from A/B label disagreements")
    return final


def command_prepare(args: argparse.Namespace) -> None:
    reg = load_registry(Path(args.registry).resolve())
    current_head = git_head()
    protocol_commit = args.protocol_commit.lower() if args.protocol_commit else current_head
    if protocol_commit != current_head:
        raise Step3PValidationError("protocol commit must equal the current HEAD")
    if not git_is_ancestor(reg["entry_commit"], protocol_commit):
        raise Step3PValidationError("Step3P protocol commit does not descend from the frozen entry")
    if not args.dry_run and git_status_porcelain():
        raise Step3PValidationError("formal packet preparation requires a clean P0 worktree")
    package = collect_public_candidates(reg, Path(args.public_source).resolve(), protocol_commit)
    files = build_prepare_files(package, reg)
    if args.dry_run:
        print(json_text(package["diagnostics"]), end="")
        return
    registered_output = resolve_repo_path(reg["outputs"]["root"])
    output_root = Path(args.output_root).resolve() if args.output_root else registered_output
    if output_root != registered_output:
        raise Step3PValidationError("formal packet output must use the registered repository path")
    write_bundle(output_root, files)
    print(json_text({"output_root": str(output_root), **package["diagnostics"]}), end="")


def command_analyze_ab(args: argparse.Namespace) -> None:
    reg = load_registry(Path(args.registry).resolve())
    output_root = Path(args.output_root).resolve() if args.output_root else resolve_repo_path(reg["outputs"]["root"])
    bundle = load_prepare_bundle(output_root)
    rows_a, provenance_a = _load_and_validate_review(
        Path(args.review_a).resolve(),
        Path(args.provenance_a).resolve(),
        output_root / "packets" / "reviewer_a_packet.jsonl",
        bundle["packet_a"],
        reg,
        "reviewer_a",
    )
    rows_b, provenance_b = _load_and_validate_review(
        Path(args.review_b).resolve(),
        Path(args.provenance_b).resolve(),
        output_root / "packets" / "reviewer_b_packet.jsonl",
        bundle["packet_b"],
        reg,
        "reviewer_b",
    )
    validate_reviewer_separation(provenance_a, provenance_b)
    namespace = selection_namespace(reg)
    reliability = compute_ab_reliability(
        bundle["primary"],
        bundle["packet_a"],
        bundle["packet_b"],
        bundle["links_a"],
        bundle["links_b"],
        rows_a,
        rows_b,
        reg,
        namespace,
    )
    files = build_c_files(bundle, rows_a, rows_b, reliability, reg)
    files["ab/reviewer_a_review.jsonl"] = Path(args.review_a).resolve().read_bytes()
    files["ab/reviewer_a_provenance.json"] = Path(args.provenance_a).resolve().read_bytes()
    files["ab/reviewer_b_review.jsonl"] = Path(args.review_b).resolve().read_bytes()
    files["ab/reviewer_b_provenance.json"] = Path(args.provenance_b).resolve().read_bytes()
    files["ab/ab_input_manifest.json"] = json_text(
        {
            "route_id": reg["route_id"],
            "reviewers": [
                {"role": provenance_a["role"], "reviewer_id": provenance_a["reviewer_id"], "model": provenance_a["model"]},
                {"role": provenance_b["role"], "reviewer_id": provenance_b["reviewer_id"], "model": provenance_b["model"]},
            ],
            "artifacts": [
                {
                    "path": name,
                    "bytes": len(value),
                    "sha256": sha256_bytes(value),
                }
                for name, value in sorted(files.items())
                if name.startswith("ab/reviewer_")
            ],
            "training_authorized": False,
            "optimizer_steps": 0,
        }
    ).encode("utf-8")
    review_root = output_root / "reviews"
    write_bundle(review_root, files)
    print(json_text({"review_root": str(review_root), **reliability}), end="")


def command_finalize(args: argparse.Namespace) -> None:
    reg = load_registry(Path(args.registry).resolve())
    output_root = Path(args.output_root).resolve() if args.output_root else resolve_repo_path(reg["outputs"]["root"])
    bundle = load_prepare_bundle(output_root)
    frozen_ab = output_root / "reviews" / "ab"
    frozen_inputs = {
        Path(args.review_a).resolve(): frozen_ab / "reviewer_a_review.jsonl",
        Path(args.provenance_a).resolve(): frozen_ab / "reviewer_a_provenance.json",
        Path(args.review_b).resolve(): frozen_ab / "reviewer_b_review.jsonl",
        Path(args.provenance_b).resolve(): frozen_ab / "reviewer_b_provenance.json",
    }
    for supplied, frozen in frozen_inputs.items():
        if not frozen.is_file() or sha256_file(supplied) != sha256_file(frozen):
            raise Step3PValidationError(
                "finalization A/B input differs from the frozen C-packet input"
            )
    rows_a, provenance_a = _load_and_validate_review(
        Path(args.review_a).resolve(), Path(args.provenance_a).resolve(), output_root / "packets" / "reviewer_a_packet.jsonl", bundle["packet_a"], reg, "reviewer_a"
    )
    rows_b, provenance_b = _load_and_validate_review(
        Path(args.review_b).resolve(), Path(args.provenance_b).resolve(), output_root / "packets" / "reviewer_b_packet.jsonl", bundle["packet_b"], reg, "reviewer_b"
    )
    validate_reviewer_separation(provenance_a, provenance_b)
    namespace = selection_namespace(reg)
    reliability = compute_ab_reliability(
        bundle["primary"], bundle["packet_a"], bundle["packet_b"], bundle["links_a"], bundle["links_b"], rows_a, rows_b, reg, namespace
    )
    if not reliability["overall_pass"]:
        raise Step3PValidationError("A/B reliability failed; finalization is forbidden")
    c_packet_path = output_root / "reviews" / "ab" / "reviewer_c_packet.jsonl"
    c_packet = read_jsonl(c_packet_path)
    if c_packet:
        if not args.review_c or not args.provenance_c:
            raise Step3PValidationError("Reviewer C review and provenance are required")
        rows_c, provenance_c = _load_and_validate_review(
            Path(args.review_c).resolve(), Path(args.provenance_c).resolve(), c_packet_path, c_packet, reg, "reviewer_c"
        )
        validate_reviewer_separation(provenance_a, provenance_b, provenance_c)
    else:
        rows_c = []
        provenance_c = None
    final = _adjudicate(bundle["primary"], rows_a, rows_b, rows_c)
    final_by_id = {row["opaque_candidate_id"]: row for row in final}
    qualified = [
        row
        for row in bundle["primary"]
        if final_by_id[row["opaque_candidate_id"]]["candidate_qualified"] == "yes"
    ]
    selection = select_targets(qualified, reg, namespace=namespace)
    route = "PASS_PUBLIC_MACHINE_RUBRIC_POOL" if selection["pass"] else "STOP_TARGET_YIELD"
    files: dict[str, bytes] = {
        "final/final_labels.jsonl": jsonl_text(final).encode("utf-8"),
        "final/final_report.json": json_text(
            {
                "route": route,
                "qualified_candidates": len(qualified),
                "qualified_prompts": len({row["opaque_prompt_id"] for row in qualified}),
                "selection": selection["summary"],
                "training_authorized": False,
                "optimizer_steps": 0,
            }
        ).encode("utf-8"),
    }
    if provenance_c is not None:
        files["final/reviewer_c_review.jsonl"] = Path(args.review_c).resolve().read_bytes()
        files["final/reviewer_c_provenance.json"] = Path(args.provenance_c).resolve().read_bytes()
    if selection["pass"]:
        selected_rows = []
        for row in sorted(
            selection["selected"],
            key=lambda value: stable_rank(namespace, "final_target_order", value["opaque_candidate_id"]),
        ):
            selected_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "opaque_prompt_id": row["opaque_prompt_id"],
                    "opaque_candidate_id": row["opaque_candidate_id"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "assigned_coverage_category": selection["assigned_categories"][row["opaque_candidate_id"]],
                    "all_categories": row["categories"],
                    "severity_level": row["severity_level"],
                    "prompt_sha256": row["prompt_sha256"],
                    "response_sha256": row["response_sha256"],
                    "normalized_response_sha256": row["normalized_response_sha256"],
                    "public_source_revision": reg["source"]["public_revision"],
                    "public_source_line": row["public_source_line"],
                }
            )
        files["final/machine_rubric_candidate_pool.jsonl"] = jsonl_text(selected_rows).encode("utf-8")
    final_root = output_root / "finalization"
    write_bundle(final_root, files)
    print(json_text({"final_root": str(final_root), "route": route, **selection["summary"]}), end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Validate the pinned source and freeze blind packets")
    prepare.add_argument("--public-source", required=True)
    prepare.add_argument("--protocol-commit")
    prepare.add_argument("--output-root")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=command_prepare)

    ab = sub.add_parser("analyze-ab", help="Validate A/B reviews and open C only after reliability passes")
    ab.add_argument("--output-root")
    ab.add_argument("--review-a", required=True)
    ab.add_argument("--provenance-a", required=True)
    ab.add_argument("--review-b", required=True)
    ab.add_argument("--provenance-b", required=True)
    ab.set_defaults(func=command_analyze_ab)

    final = sub.add_parser("finalize", help="Adjudicate valid reviews and apply the fixed 48-target flow")
    final.add_argument("--output-root")
    final.add_argument("--review-a", required=True)
    final.add_argument("--provenance-a", required=True)
    final.add_argument("--review-b", required=True)
    final.add_argument("--provenance-b", required=True)
    final.add_argument("--review-c")
    final.add_argument("--provenance-c")
    final.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
