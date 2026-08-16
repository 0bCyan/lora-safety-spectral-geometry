"""Certify the best-case CSPR P2 fixed-pool yield under every legal C ruling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2/"
    "protocol/cspr_p2_resource_closure_protocol.json"
)
ACTION_FIELDS = ("safe_route", "executable_step", "prompt_binding")


class ClosureValidationError(ValueError):
    """Raised when a frozen closure input is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise ClosureValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_frozen(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def by_id(rows: list[dict[str, Any]], id_field: str) -> dict[str, dict[str, Any]]:
    mapped = {str(row[id_field]): row for row in rows}
    if len(mapped) != len(rows):
        raise ClosureValidationError(f"duplicate {id_field}")
    return mapped


def validate_action_reviews(
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    mapped_a = by_id(reviewer_a, "opaque_action_id")
    mapped_b = by_id(reviewer_b, "opaque_action_id")
    if set(mapped_a) != set(mapped_b):
        raise ClosureValidationError("reviewer A/B actionability IDs differ")
    for role, mapped in (("a", mapped_a), ("b", mapped_b)):
        for opaque_id, row in mapped.items():
            for field in ACTION_FIELDS:
                if row.get(field) not in {"yes", "no"}:
                    raise ClosureValidationError(f"reviewer {role} invalid {field}: {opaque_id}")
    primary = [row for row in key_rows if not row["is_hidden_duplicate"]]
    duplicate_ids = {str(row["opaque_action_id"]) for row in key_rows if row["is_hidden_duplicate"]}
    primary_ids = {str(row["opaque_action_id"]) for row in primary}
    if primary_ids | duplicate_ids != set(mapped_a) or primary_ids & duplicate_ids:
        raise ClosureValidationError("hidden key does not partition the reviewed IDs")
    return mapped_a, mapped_b, primary


def compute_bounds(
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    primary_key_rows: list[dict[str, Any]],
    yield_gates: dict[str, Any],
) -> dict[str, Any]:
    consensus_score2 = 0
    best_case_score2 = 0
    disputed_best_case = 0
    agreed_no_locked = 0
    split_lower = Counter()
    split_upper = Counter()
    category_lower = Counter()
    category_upper = Counter()
    field_lock_counts = Counter()

    for key_row in primary_key_rows:
        opaque_id = str(key_row["opaque_action_id"])
        left = reviewer_a[opaque_id]
        right = reviewer_b[opaque_id]
        agreed_no_fields = [
            field for field in ACTION_FIELDS if left[field] == right[field] == "no"
        ]
        disagreement_fields = [
            field for field in ACTION_FIELDS if left[field] != right[field]
        ]
        is_consensus_score2 = all(
            left[field] == right[field] == "yes" for field in ACTION_FIELDS
        )
        can_be_score2 = not agreed_no_fields
        split = str(key_row["prospective_split"])
        category = str(key_row["assigned_category"])
        field_lock_counts.update(agreed_no_fields)
        if is_consensus_score2:
            consensus_score2 += 1
            split_lower[split] += 1
            category_lower[category] += 1
        if can_be_score2:
            best_case_score2 += 1
            split_upper[split] += 1
            category_upper[category] += 1
            if disagreement_fields:
                disputed_best_case += 1
        else:
            agreed_no_locked += 1

    expected_splits = {
        "train_candidate": (
            int(yield_gates["minimum_train_candidate_pairs"]),
            int(yield_gates["train_candidate_denominator"]),
        ),
        "calibration": (
            int(yield_gates["minimum_calibration_pairs"]),
            int(yield_gates["calibration_denominator"]),
        ),
        "mechanism_dev": (
            int(yield_gates["minimum_mechanism_dev_pairs"]),
            int(yield_gates["mechanism_dev_denominator"]),
        ),
    }
    split_bounds = {
        split: {
            "denominator": denominator,
            "minimum_required": minimum,
            "consensus_score2_lower_bound": split_lower[split],
            "adjudication_best_case_score2_upper_bound": split_upper[split],
            "yield_mathematically_reachable": split_upper[split] >= minimum,
        }
        for split, (minimum, denominator) in expected_splits.items()
    }
    category_minimum = int(yield_gates["minimum_categories_with_at_least_two_pairs"])
    category_upper_with_two = sum(value >= 2 for value in category_upper.values())
    total_minimum = int(yield_gates["minimum_total_pairs"])
    total_reachable = best_case_score2 >= total_minimum
    return {
        "primary_candidate_rows": len(primary_key_rows),
        "consensus_score2_lower_bound": consensus_score2,
        "disputed_rows_that_could_reach_score2": disputed_best_case,
        "adjudication_best_case_score2_upper_bound": best_case_score2,
        "rows_locked_below_score2_by_at_least_one_ab_agreed_no": agreed_no_locked,
        "agreed_no_field_lock_counts": dict(sorted(field_lock_counts.items())),
        "split_bounds": split_bounds,
        "category_bounds": {
            "category_denominator": int(yield_gates["category_denominator"]),
            "minimum_categories_with_at_least_two_pairs": category_minimum,
            "consensus_categories_with_at_least_two_score2_rows": sum(
                value >= 2 for value in category_lower.values()
            ),
            "best_case_categories_with_at_least_two_score2_rows": category_upper_with_two,
            "diversity_mathematically_reachable": category_upper_with_two >= category_minimum,
        },
        "total_bound": {
            "minimum_qualified_pairs_required": total_minimum,
            "adjudication_best_case_score2_upper_bound": best_case_score2,
            "qualification_upper_bound": best_case_score2,
            "total_yield_mathematically_reachable": total_reachable,
            "shortfall_even_under_best_case_adjudication": max(0, total_minimum - best_case_score2),
        },
        "proof_obligations": {
            "agreed_fields_are_not_adjudicated": True,
            "score2_requires_all_three_actionability_fields_yes": True,
            "pair_qualification_requires_candidate_score2": True,
            "qualified_set_is_subset_of_potential_score2_set": True,
        },
        "fixed_pool_mathematically_unreachable": not total_reachable,
    }


def verify_inputs(protocol: dict[str, Any]) -> None:
    for name, artifact in protocol["frozen_inputs"].items():
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ClosureValidationError(f"frozen input hash mismatch: {name}")


def output_values(protocol_path: Path, protocol: dict[str, Any]) -> tuple[dict[str, Any], str]:
    verify_inputs(protocol)
    inputs = protocol["frozen_inputs"]
    base = read_json(resolve_repo_path(inputs["base_protocol"]["path"]))
    ab_analysis = read_json(resolve_repo_path(inputs["ab_analysis"]["path"]))
    if ab_analysis.get("route") != "STOP_CSPR_P2_REVIEW_UNRELIABLE":
        raise ClosureValidationError("closure requires the frozen unreliable A/B stop route")
    if ab_analysis.get("reviewer_c_authorized") is not False:
        raise ClosureValidationError("closure requires Reviewer C to remain unauthorized")
    if ab_analysis.get("optimizer_steps") != 0 or ab_analysis.get("training_authorized") is not False:
        raise ClosureValidationError("A/B analysis violates the zero-training boundary")

    reviewer_a_path = resolve_repo_path(inputs["actionability_reviewer_a"]["path"])
    reviewer_b_path = resolve_repo_path(inputs["actionability_reviewer_b"]["path"])
    key_path = resolve_repo_path(inputs["actionability_hidden_key"]["path"])
    reviewer_hashes = {
        (row["construct"], row["reviewer_role"]): row["output_sha256"]
        for row in ab_analysis["reviewer_provenance"]
    }
    if reviewer_hashes.get(("actionability", "a")) != sha256(reviewer_a_path):
        raise ClosureValidationError("Reviewer A output is not bound to the A/B analysis")
    if reviewer_hashes.get(("actionability", "b")) != sha256(reviewer_b_path):
        raise ClosureValidationError("Reviewer B output is not bound to the A/B analysis")

    reviewer_a, reviewer_b, primary = validate_action_reviews(
        read_jsonl(reviewer_a_path), read_jsonl(reviewer_b_path), read_jsonl(key_path)
    )
    bounds = compute_bounds(
        reviewer_a, reviewer_b, primary, base["fixed_pool_yield_gates"]
    )
    certificate = {
        "name": "step3_cspr_p2_resource_impossibility_certificate_v1",
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "route": "CLOSE_CSPR_P2_FIXED_POOL_MATHEMATICALLY_UNREACHABLE",
        "theorem": "For every legal disagreement-only Reviewer C labeling, the qualified pair count is at most the count of primary candidate rows without an A/B-agreed no on any actionability field.",
        "bounds": bounds,
        "reviewer_c_started": False,
        "candidate_responses_opened_by_closure": False,
        "source_labels_used": False,
        "replacement_pairs_used": False,
        "thresholds_changed": False,
        "model_gradient_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    total = bounds["total_bound"]
    split_lines = "\n".join(
        f"| {name} | {row['minimum_required']} | {row['adjudication_best_case_score2_upper_bound']} | {row['yield_mathematically_reachable']} |"
        for name, row in bounds["split_bounds"].items()
    )
    report = f"""# CSPR P2 Fixed-Pool Impossibility Certificate

Route: `CLOSE_CSPR_P2_FIXED_POOL_MATHEMATICALLY_UNREACHABLE`.

Let `U` contain each primary candidate row for which no actionability field has an A/B-agreed `no`. Reviewer C may decide only disputed fields, so every row outside `U` remains below score 2 under every legal C labeling. Pair qualification requires all three candidate actionability fields to be `yes`; therefore the qualified set is a subset of `U`.

The frozen reviews contain `{bounds['consensus_score2_lower_bound']}` consensus score-2 rows and only `{bounds['disputed_rows_that_could_reach_score2']}` additional rows that could reach score 2 under best-case adjudication. Thus `|U|={total['qualification_upper_bound']}` and

`qualified_pairs <= {total['qualification_upper_bound']} < {total['minimum_qualified_pairs_required']}`.

The total fixed-pool gate is short by at least `{total['shortfall_even_under_best_case_adjudication']}` pairs before applying either safety requirement, so safety adjudication cannot repair the deficit.

| split | required | best-case action score-2 upper bound | reachable |
| --- | ---: | ---: | --- |
{split_lines}

This is a deterministic closure bound, not a prevalence estimate. It does not weaken a threshold, inspect candidate text, launch Reviewer C, replace a pair, compute a model gradient, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
"""
    return certificate, report


def run(protocol_path: Path, protocol: dict[str, Any], verify_only: bool) -> dict[str, Any]:
    certificate, report = output_values(protocol_path, protocol)
    certificate_path = resolve_repo_path(protocol["outputs"]["certificate"])
    report_path = resolve_repo_path(protocol["outputs"]["report"])
    if verify_only:
        if not certificate_path.is_file() or read_json(certificate_path) != certificate:
            raise ClosureValidationError("certificate does not reproduce")
        if not report_path.is_file() or report_path.read_text(encoding="utf-8") != report:
            raise ClosureValidationError("closure report does not reproduce")
        status = "PASS_CSPR_P2_RESOURCE_CLOSURE_VERIFY"
    else:
        write_json_frozen(certificate_path, certificate)
        write_frozen(report_path, report)
        status = "PASS_CSPR_P2_RESOURCE_CLOSURE"
    result = {
        "status": status,
        "route": certificate["route"],
        "qualification_upper_bound": certificate["bounds"]["total_bound"]["qualification_upper_bound"],
        "minimum_required": certificate["bounds"]["total_bound"]["minimum_qualified_pairs_required"],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    protocol = read_json(protocol_path)
    try:
        run(protocol_path, protocol, args.command == "verify")
    except (ClosureValidationError, FileExistsError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2_RESOURCE_CLOSURE_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
