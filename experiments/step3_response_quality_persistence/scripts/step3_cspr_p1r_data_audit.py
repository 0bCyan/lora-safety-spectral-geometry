"""Run the conservative empty-response repair for the CSPR P1 audit."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cspr_p1_data_audit as base


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger", required=True)
    return parser.parse_args()


def verify_repair_protocol(repair: dict[str, Any]) -> dict[str, Any]:
    for label, artifact in repair["artifacts"].items():
        path = base.resolve_repo_path(artifact["path"])
        if base.sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen repair artifact hash mismatch: {label}")
    base_spec = repair["base_protocol"]
    base_path = base.resolve_repo_path(base_spec["path"])
    if base.sha256(base_path) != base_spec["sha256"]:
        raise ValueError("frozen P1 v1 protocol hash mismatch")
    inherited = base.read_json(base_path)
    base.verify_frozen_inputs(inherited)
    if repair["unchanged_constraints"] != {
        "source_revision": inherited["source"]["revision"],
        "source_sha256": inherited["source"]["sha256"],
        "freshness_filter": inherited["freshness_filter"],
        "selection": inherited["selection"],
        "gates": inherited["gates"],
        "exclusion_reference_count": len(inherited["exclusion_references"]),
    }:
        raise ValueError("P1R changed a frozen P1 v1 constraint")
    return inherited


def load_beaver_groups(
    source_path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_cfg = protocol["source"]
    expected_categories = set(source_cfg["expected_categories"])
    groups: dict[str, dict[str, Any]] = {}
    row_count = 0
    empty_response_rows = 0
    empty_safe_response_rows = 0
    empty_unsafe_response_rows = 0
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
            if not isinstance(response, str):
                raise ValueError(f"non-string response at row {source_index}")
            if not isinstance(source["is_safe"], bool):
                raise ValueError(f"non-boolean safety label at row {source_index}")
            if not isinstance(categories, dict) or set(categories) != expected_categories:
                raise ValueError(f"category schema mismatch at row {source_index}")
            if any(type(value) is not bool for value in categories.values()):
                raise ValueError(f"non-boolean category label at row {source_index}")

            prompt = prompt.strip()
            response = response.strip()
            if not response:
                empty_response_rows += 1
                empty_safe_response_rows += int(source["is_safe"])
                empty_unsafe_response_rows += int(not source["is_safe"])
                continue

            normalized_prompt = base.normalize_text(prompt)
            normalized_prompt_hash = base.sha256_text(normalized_prompt)
            group = groups.setdefault(
                normalized_prompt_hash,
                {
                    "prompt": prompt,
                    "prompt_exact_sha256": base.sha256_text(prompt),
                    "prompt_normalized_sha256": normalized_prompt_hash,
                    "safe": {},
                    "unsafe": {},
                    "unsafe_categories": set(),
                },
            )
            normalized_response_hash = base.sha256_text(base.normalize_text(response))
            response_record = {
                "source_index": source_index,
                "response_sha256": base.sha256_text(response),
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
        "nonempty_response_rows": row_count - empty_response_rows,
        "empty_response_rows_excluded": empty_response_rows,
        "empty_safe_response_rows_excluded": empty_safe_response_rows,
        "empty_unsafe_response_rows_excluded": empty_unsafe_response_rows,
        "unique_normalized_prompts_with_nonempty_response": len(groups),
        "mixed_safe_unsafe_prompts": len(raw_mixed),
        "safe_responses_on_mixed_prompts": raw_safe_response_count,
        "unsafe_responses_on_mixed_prompts": raw_unsafe_response_count,
        "cross_label_identical_responses_removed": label_conflict_responses,
        "pair_eligible_prompts_after_response_conflict_removal": len(candidates),
    }
    return candidates, inventory


def main() -> None:
    args = parse_args()
    repair_path = base.resolve_repo_path(args.protocol)
    output_path = base.resolve_repo_path(args.output)
    ledger_path = base.resolve_repo_path(args.ledger)
    repair = base.read_json(repair_path)
    protocol = verify_repair_protocol(repair)

    source_path = base.resolve_repo_path(protocol["source"]["path"])
    if base.sha256(source_path) != protocol["source"]["sha256"]:
        raise ValueError("frozen BeaverTails source hash mismatch")
    exact_hashes, normalized_hashes, reference_texts, reference_counts = (
        base.load_reference_state(protocol)
    )
    candidates, inventory = load_beaver_groups(source_path, protocol)
    filtered, freshness = base.filter_candidates(
        candidates,
        exact_hashes,
        normalized_hashes,
        reference_texts,
        protocol,
    )
    selected = base.select_pairs(filtered, protocol)

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
        "empty_responses_excluded_from_pair_eligibility": inventory[
            "empty_response_rows_excluded"
        ]
        >= 1,
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
        == base.category_targets(
            int(protocol["selection"]["prompt_count"]),
            list(source_cfg["expected_categories"]),
            int(protocol["selection"]["seed"]),
        ),
    }
    passed = all(checks.values())
    ledger_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    result = {
        "name": "step3_cspr_p1r_public_source_feasibility_audit_v1",
        "repair_protocol": args.protocol,
        "repair_protocol_sha256": base.sha256(repair_path),
        "base_protocol": repair["base_protocol"]["path"],
        "base_protocol_sha256": repair["base_protocol"]["sha256"],
        "closed_v1_stop": repair["closed_v1_stop"],
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
        "selected_pair_ledger_sha256": base.sha256_bytes(ledger_text.encode("utf-8")),
        "checks": checks,
        "passed": passed,
        "route": "PASS_CSPR_P1R_SOURCE_FEASIBILITY"
        if passed
        else "STOP_CSPR_P1R_DATA_INSUFFICIENT",
        "claim_boundary": repair["claim_boundary"],
        "source_labels_accepted_as_final": False,
        "model_gradient_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_content_opened": False,
    }
    base.write_frozen(ledger_path, ledger_text)
    base.write_frozen(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "route": result["route"], "checks": checks}))
    if not passed:
        raise RuntimeError(result["route"])


if __name__ == "__main__":
    main()
