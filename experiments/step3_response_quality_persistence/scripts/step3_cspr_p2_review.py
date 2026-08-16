"""Validate, adjudicate, and summarize frozen CSPR P2 blinded reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTION_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2/"
    "protocol/cspr_p2_review_execution_protocol.json"
)
YES_NO = {"yes", "no"}
ACTION_FIELDS = ("safe_route", "executable_step", "prompt_binding")


class ReviewValidationError(ValueError):
    """Raised when a frozen review input violates the prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["preflight", "analyze-ab", "finalize", "record-inconclusive"]
    )
    parser.add_argument("--execution-protocol", default=DEFAULT_EXECUTION_PROTOCOL)
    parser.add_argument("--construct", choices=["safety", "actionability"])
    parser.add_argument("--reviewer-role", choices=["a", "b", "c"])
    parser.add_argument("--agent-id")
    parser.add_argument("--reason")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise ReviewValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReviewValidationError(f"JSONL row {number} is not an object: {path}")
        rows.append(value)
    return rows


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_frozen(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_frozen(path: Path, rows: list[dict[str, Any]]) -> None:
    write_frozen(path, jsonl_text(rows))


def execution_context(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = read_json(path)
    for name, artifact in execution["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise ReviewValidationError(f"frozen input hash mismatch: {name}")
    base_path = resolve_repo_path(execution["frozen_inputs"]["base_protocol"]["path"])
    manifest_path = resolve_repo_path(execution["frozen_inputs"]["packet_manifest"]["path"])
    base = read_json(base_path)
    manifest = read_json(manifest_path)
    if manifest.get("protocol_sha256") != sha256(base_path):
        raise ReviewValidationError("packet manifest does not bind the frozen base protocol")
    for name, artifact in base["artifacts"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise ReviewValidationError(f"base protocol artifact hash mismatch: {name}")
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise ReviewValidationError(
                f"packet manifest artifact hash mismatch: {artifact['path']}"
            )
    if base.get("optimizer_steps") != 0 or base.get("training_authorized") is not False:
        raise ReviewValidationError("base protocol does not preserve the zero-training boundary")
    if execution.get("optimizer_steps") != 0 or execution.get("training_authorized") is not False:
        raise ReviewValidationError("execution protocol does not preserve the zero-training boundary")
    return execution, base, manifest


def artifact_hash(manifest: dict[str, Any], path: Path) -> str:
    wanted = relative(path)
    matches = [row for row in manifest["artifacts"] if row["path"] == wanted]
    if len(matches) != 1:
        raise ReviewValidationError(f"packet artifact not uniquely registered: {wanted}")
    return str(matches[0]["sha256"])


def construct_spec(
    construct: str,
    role: str,
    execution: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    cfg = execution["review_files"][construct]
    if construct == "safety":
        return {
            "construct": construct,
            "role": role,
            "id_field": "opaque_safety_id",
            "decision_fields": ("clean_safe",),
            "packet": resolve_repo_path(base["outputs"][f"safety_{role}_packet"]),
            "template": resolve_repo_path(base["outputs"][f"safety_{role}_template"]),
            "rubric": resolve_repo_path(base["artifacts"]["safety_rubric"]["path"]),
            "output": resolve_repo_path(cfg[f"reviewer_{role}_labels"]),
            "provenance": resolve_repo_path(cfg[f"reviewer_{role}_provenance"]),
        }
    if construct == "actionability":
        return {
            "construct": construct,
            "role": role,
            "id_field": "opaque_action_id",
            "decision_fields": ACTION_FIELDS,
            "packet": resolve_repo_path(base["outputs"][f"action_{role}_packet"]),
            "template": resolve_repo_path(base["outputs"][f"action_{role}_template"]),
            "rubric": resolve_repo_path(base["artifacts"]["actionability_rubric"]["path"]),
            "output": resolve_repo_path(cfg[f"reviewer_{role}_labels"]),
            "provenance": resolve_repo_path(cfg[f"reviewer_{role}_provenance"]),
        }
    raise ReviewValidationError(f"unknown construct: {construct}")


def validate_packet_template(spec: dict[str, Any], manifest: dict[str, Any]) -> None:
    packet = read_jsonl(spec["packet"])
    template = read_jsonl(spec["template"])
    if sha256(spec["packet"]) != artifact_hash(manifest, spec["packet"]):
        raise ReviewValidationError(f"{spec['construct']} {spec['role']} packet hash mismatch")
    if sha256(spec["template"]) != artifact_hash(manifest, spec["template"]):
        raise ReviewValidationError(f"{spec['construct']} {spec['role']} template hash mismatch")
    if sha256(spec["rubric"]) == "" or not spec["rubric"].is_file():
        raise ReviewValidationError("rubric missing")
    id_field = spec["id_field"]
    packet_ids = [row.get(id_field) for row in packet]
    template_ids = [row.get(id_field) for row in template]
    if packet_ids != template_ids or len(set(packet_ids)) != len(packet_ids):
        raise ReviewValidationError(f"{spec['construct']} {spec['role']} packet/template order mismatch")
    expected_fields = {id_field, *spec["decision_fields"], "short_reason"}
    for row in template:
        if set(row) != expected_fields:
            raise ReviewValidationError(f"{spec['construct']} template schema mismatch")
        if any(row[field] != "" for field in spec["decision_fields"] + ("short_reason",)):
            raise ReviewValidationError(f"{spec['construct']} template is not blank")


def validate_review(
    spec: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validate_packet_template(spec, manifest)
    template = read_jsonl(spec["template"])
    rows = read_jsonl(spec["output"])
    id_field = spec["id_field"]
    expected_fields = {id_field, *spec["decision_fields"], "short_reason"}
    if len(rows) != len(template):
        raise ReviewValidationError(f"{spec['construct']} {spec['role']} incomplete review")
    mapped: dict[str, dict[str, Any]] = {}
    for index, (row, blank) in enumerate(zip(rows, template, strict=True), start=1):
        opaque_id = row.get(id_field)
        if set(row) != expected_fields:
            raise ReviewValidationError(f"review schema mismatch at row {index}")
        if opaque_id != blank[id_field] or opaque_id in mapped:
            raise ReviewValidationError(f"review ID/order mismatch at row {index}")
        for field in spec["decision_fields"]:
            if row[field] not in YES_NO:
                raise ReviewValidationError(f"invalid {field} at {opaque_id}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ReviewValidationError(f"blank short_reason at {opaque_id}")
        mapped[str(opaque_id)] = row

    provenance = read_json(spec["provenance"])
    expected = {
        "construct": spec["construct"],
        "reviewer_role": spec["role"],
        "packet_path": relative(spec["packet"]),
        "packet_sha256": sha256(spec["packet"]),
        "template_path": relative(spec["template"]),
        "template_sha256": sha256(spec["template"]),
        "rubric_path": relative(spec["rubric"]),
        "rubric_sha256": sha256(spec["rubric"]),
        "output_path": relative(spec["output"]),
        "output_sha256": sha256(spec["output"]),
        "row_count": len(rows),
        "fresh_context": True,
        "forbidden_material_not_opened": True,
        "completion_status": "complete",
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ReviewValidationError(
                f"{spec['construct']} {spec['role']} provenance mismatch: {field}"
            )
    if not isinstance(provenance.get("agent_id"), str) or not provenance["agent_id"].strip():
        raise ReviewValidationError("missing reviewer agent_id")
    if not isinstance(provenance.get("session_id"), str) or not provenance["session_id"].strip():
        raise ReviewValidationError("missing reviewer session_id")
    if not isinstance(provenance.get("runtime"), str) or not provenance["runtime"].strip():
        raise ReviewValidationError("missing reviewer runtime")
    return mapped, provenance


def raw_agreement(pairs: list[tuple[Any, Any]]) -> float:
    if not pairs:
        raise ReviewValidationError("agreement requires at least one pair")
    return sum(left == right for left, right in pairs) / len(pairs)


def gwet_ac1(pairs: list[tuple[Any, Any]], labels: list[Any]) -> float | None:
    if not pairs:
        return None
    pooled = Counter(value for pair in pairs for value in pair)
    probabilities = {label: pooled[label] / (2 * len(pairs)) for label in labels}
    expected = sum(prob * (1 - prob) for prob in probabilities.values()) / (len(labels) - 1)
    if math.isclose(expected, 1.0):
        return None
    return (raw_agreement(pairs) - expected) / (1 - expected)


def linear_weight(left: int, right: int, categories: list[int]) -> float:
    span = max(categories) - min(categories)
    return 1.0 if span == 0 else 1.0 - abs(left - right) / span


def weighted_agreement(pairs: list[tuple[int, int]], categories: list[int]) -> float:
    if not pairs:
        raise ReviewValidationError("agreement requires at least one pair")
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
                * (1 - probabilities[right])
                / (len(categories) - 1)
            )
    if math.isclose(expected, 1.0):
        return None
    return (weighted_agreement(pairs, categories) - expected) / (1 - expected)


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
        "pooled_yes_prevalence": sum(value == "yes" for pair in pairs for value in pair)
        / (2 * len(pairs)),
    }


def derived_score(row: dict[str, Any]) -> int:
    if row["safe_route"] == "no":
        return 0
    return 2 if row["executable_step"] == "yes" and row["prompt_binding"] == "yes" else 1


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


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
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
    values = []
    for _ in range(replicates):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = metric(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    return {
        "replicates": replicates,
        "finite_replicates": len(values),
        "lower_95": percentile(values, 0.025) if values else None,
        "upper_95": percentile(values, 0.975) if values else None,
    }


def primary_ids_and_key(
    construct: str, base: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    output_name = "safety_key" if construct == "safety" else "action_key"
    id_field = "opaque_safety_id" if construct == "safety" else "opaque_action_id"
    key_rows = read_jsonl(resolve_repo_path(base["outputs"][output_name]))
    primary = sorted(str(row[id_field]) for row in key_rows if not row["is_hidden_duplicate"])
    return primary, key_rows


def duplicate_consistency(
    review: dict[str, dict[str, Any]],
    key_rows: list[dict[str, Any]],
    id_field: str,
    decision_fields: tuple[str, ...],
) -> dict[str, Any]:
    details = []
    field_counts = Counter()
    full_count = 0
    duplicates = [row for row in key_rows if row["is_hidden_duplicate"]]
    for duplicate in duplicates:
        duplicate_id = str(duplicate[id_field])
        primary_id = str(duplicate["duplicate_of"])
        matches = {
            field: review[primary_id][field] == review[duplicate_id][field]
            for field in decision_fields
        }
        field_counts.update(field for field, matched in matches.items() if matched)
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


def safety_ab_analysis(
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    primary_ids: list[str],
    key_rows: list[dict[str, Any]],
    gates_cfg: dict[str, Any],
) -> dict[str, Any]:
    pairs = [(reviewer_a[row_id]["clean_safe"], reviewer_b[row_id]["clean_safe"]) for row_id in primary_ids]
    metrics = nominal_metrics(pairs)
    duplicate_a = duplicate_consistency(reviewer_a, key_rows, "opaque_safety_id", ("clean_safe",))
    duplicate_b = duplicate_consistency(reviewer_b, key_rows, "opaque_safety_id", ("clean_safe",))
    counts = {
        role: Counter(review[row_id]["clean_safe"] for row_id in primary_ids)
        for role, review in (("reviewer_a", reviewer_a), ("reviewer_b", reviewer_b))
    }
    gates = {
        "raw_agreement_pass": metrics["raw_agreement"] >= float(gates_cfg["raw_agreement_min"]),
        "gwet_ac1_pass": metrics["gwet_ac1"] is not None and metrics["gwet_ac1"] >= float(gates_cfg["gwet_ac1_min"]),
        "positive_agreement_pass": metrics["positive_agreement"] is not None and metrics["positive_agreement"] >= float(gates_cfg["positive_agreement_min"]),
        "negative_agreement_pass": metrics["negative_agreement"] is not None and metrics["negative_agreement"] >= float(gates_cfg["negative_agreement_min"]),
        "reviewer_a_yes_count_pass": counts["reviewer_a"]["yes"] >= int(gates_cfg["minimum_yes_ratings_per_reviewer"]),
        "reviewer_a_no_count_pass": counts["reviewer_a"]["no"] >= int(gates_cfg["minimum_no_ratings_per_reviewer"]),
        "reviewer_b_yes_count_pass": counts["reviewer_b"]["yes"] >= int(gates_cfg["minimum_yes_ratings_per_reviewer"]),
        "reviewer_b_no_count_pass": counts["reviewer_b"]["no"] >= int(gates_cfg["minimum_no_ratings_per_reviewer"]),
        "reviewer_a_duplicate_pass": duplicate_a["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_a["full_vector_consistent_count"] >= int(gates_cfg["duplicate_consistency_min_count"]),
        "reviewer_b_duplicate_pass": duplicate_b["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_b["full_vector_consistent_count"] >= int(gates_cfg["duplicate_consistency_min_count"]),
    }
    disagreements = [row_id for row_id in primary_ids if reviewer_a[row_id]["clean_safe"] != reviewer_b[row_id]["clean_safe"]]
    return {
        "primary_rows": len(primary_ids),
        "metrics": metrics,
        "primary_label_counts": {name: dict(value) for name, value in counts.items()},
        "reviewer_a_duplicate_consistency": duplicate_a,
        "reviewer_b_duplicate_consistency": duplicate_b,
        "disagreement_ids": disagreements,
        "disagreement_rows": len(disagreements),
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def actionability_ab_analysis(
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    primary_ids: list[str],
    key_rows: list[dict[str, Any]],
    gates_cfg: dict[str, Any],
) -> dict[str, Any]:
    component = {
        field: nominal_metrics([(reviewer_a[row_id][field], reviewer_b[row_id][field]) for row_id in primary_ids])
        for field in ACTION_FIELDS
    }
    scores_a = {row_id: derived_score(reviewer_a[row_id]) for row_id in primary_ids}
    scores_b = {row_id: derived_score(reviewer_b[row_id]) for row_id in primary_ids}
    ordinal_pairs = [(scores_a[row_id], scores_b[row_id]) for row_id in primary_ids]
    ordinal = ordinal_metrics(ordinal_pairs)
    boundary_pairs = [
        ("yes" if scores_a[row_id] == 2 else "no", "yes" if scores_b[row_id] == 2 else "no")
        for row_id in primary_ids
    ]
    boundary = nominal_metrics(boundary_pairs)
    prevalence = {
        "reviewer_a_score2": sum(value == 2 for value in scores_a.values()) / len(scores_a),
        "reviewer_b_score2": sum(value == 2 for value in scores_b.values()) / len(scores_b),
        "reviewer_a_score1": sum(value == 1 for value in scores_a.values()) / len(scores_a),
        "reviewer_b_score1": sum(value == 1 for value in scores_b.values()) / len(scores_b),
    }
    duplicate_a = duplicate_consistency(reviewer_a, key_rows, "opaque_action_id", ACTION_FIELDS)
    duplicate_b = duplicate_consistency(reviewer_b, key_rows, "opaque_action_id", ACTION_FIELDS)
    uncertainty = {
        "ordinal_gwet_ac2": bootstrap_interval(
            ordinal_pairs,
            lambda pairs: gwet_ac2(pairs, [0, 1, 2]),
            replicates=int(gates_cfg["bootstrap_replicates"]),
            seed=int(gates_cfg["bootstrap_seed"]),
        ),
        "score2_boundary_gwet_ac1": bootstrap_interval(
            boundary_pairs,
            lambda pairs: gwet_ac1(pairs, ["yes", "no"]),
            replicates=int(gates_cfg["bootstrap_replicates"]),
            seed=int(gates_cfg["bootstrap_seed"]) + 1,
        ),
    }
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
            "score2_marginal_gap_pass": abs(prevalence["reviewer_a_score2"] - prevalence["reviewer_b_score2"]) <= float(gates_cfg["score2_marginal_prevalence_gap_max"]),
            "reviewer_a_score2_prevalence_pass": float(gates_cfg["reviewer_score2_prevalence_min"]) <= prevalence["reviewer_a_score2"] <= float(gates_cfg["reviewer_score2_prevalence_max"]),
            "reviewer_b_score2_prevalence_pass": float(gates_cfg["reviewer_score2_prevalence_min"]) <= prevalence["reviewer_b_score2"] <= float(gates_cfg["reviewer_score2_prevalence_max"]),
            "reviewer_a_score1_prevalence_pass": prevalence["reviewer_a_score1"] >= float(gates_cfg["reviewer_score1_prevalence_min"]),
            "reviewer_b_score1_prevalence_pass": prevalence["reviewer_b_score1"] >= float(gates_cfg["reviewer_score1_prevalence_min"]),
            "reviewer_a_duplicate_pass": duplicate_a["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_a["full_vector_consistent_count"] >= int(gates_cfg["duplicate_consistency_min_count"]),
            "reviewer_b_duplicate_pass": duplicate_b["denominator"] == int(gates_cfg["duplicate_consistency_denominator"]) and duplicate_b["full_vector_consistent_count"] >= int(gates_cfg["duplicate_consistency_min_count"]),
            "ordinal_ac2_lower_bound_pass": uncertainty["ordinal_gwet_ac2"]["lower_95"] is not None and uncertainty["ordinal_gwet_ac2"]["lower_95"] >= float(gates_cfg["ordinal_gwet_ac2_lower_95_min"]),
            "score2_boundary_ac1_lower_bound_pass": uncertainty["score2_boundary_gwet_ac1"]["lower_95"] is not None and uncertainty["score2_boundary_gwet_ac1"]["lower_95"] >= float(gates_cfg["score2_boundary_gwet_ac1_lower_95_min"]),
        }
    )
    disagreements = [
        row_id
        for row_id in primary_ids
        if any(reviewer_a[row_id][field] != reviewer_b[row_id][field] for field in ACTION_FIELDS)
    ]
    return {
        "primary_rows": len(primary_ids),
        "component_metrics": component,
        "ordinal_metrics": ordinal,
        "score2_boundary_metrics": boundary,
        "score_prevalence": prevalence,
        "score_distributions": {
            "reviewer_a": dict(Counter(str(value) for value in scores_a.values())),
            "reviewer_b": dict(Counter(str(value) for value in scores_b.values())),
        },
        "reviewer_a_duplicate_consistency": duplicate_a,
        "reviewer_b_duplicate_consistency": duplicate_b,
        "bootstrap_intervals": uncertainty,
        "disagreement_ids": disagreements,
        "disagreement_rows": len(disagreements),
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def ensure_unique_sessions(provenances: list[dict[str, Any]]) -> None:
    agent_ids = [row["agent_id"] for row in provenances]
    session_ids = [row["session_id"] for row in provenances]
    if len(set(agent_ids)) != len(agent_ids) or len(set(session_ids)) != len(session_ids):
        raise ReviewValidationError("reviewer agent/session reuse is forbidden")


def load_ab(
    execution: dict[str, Any], base: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    reviews: dict[str, dict[str, dict[str, Any]]] = {}
    provenances = []
    for construct in ("safety", "actionability"):
        reviews[construct] = {}
        for role in ("a", "b"):
            mapped, provenance = validate_review(
                construct_spec(construct, role, execution, base), manifest
            )
            reviews[construct][role] = mapped
            provenances.append(provenance)
    ensure_unique_sessions(provenances)
    return reviews, provenances


def build_c_packet(
    construct: str,
    disagreement_ids: list[str],
    execution: dict[str, Any],
    base: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = construct_spec(construct, "a", execution, base)
    packet_by_id = {str(row[spec["id_field"]]): row for row in read_jsonl(spec["packet"])}
    packet = [packet_by_id[row_id] for row_id in sorted(disagreement_ids)]
    template = []
    for row in packet:
        template.append(
            {
                spec["id_field"]: row[spec["id_field"]],
                **{field: "" for field in spec["decision_fields"]},
                "short_reason": "",
            }
        )
    forbidden = {"pair_id", "response_role", "prospective_split", "assigned_category", "source_index", "duplicate_of", "is_hidden_duplicate", "reviewer_a", "reviewer_b", "disagreement_fields"}
    if any(forbidden & set(row) for row in packet):
        raise ReviewValidationError(f"private field leaked into {construct} Reviewer C packet")
    return packet, template


def preflight(execution_path: Path) -> dict[str, Any]:
    execution, base, manifest = execution_context(execution_path)
    for construct in ("safety", "actionability"):
        for role in ("a", "b"):
            validate_packet_template(construct_spec(construct, role, execution, base), manifest)
    result = {
        "status": "PASS_CSPR_P2_REVIEW_PREFLIGHT",
        "safety_rows_per_reviewer": int(base["packet_design"]["safety_packet_rows_per_reviewer"]),
        "actionability_rows_per_reviewer": int(base["packet_design"]["actionability_packet_rows_per_reviewer"]),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def analyze_ab(execution_path: Path) -> dict[str, Any]:
    execution, base, manifest = execution_context(execution_path)
    reviews, provenances = load_ab(execution, base, manifest)
    safety_ids, safety_key = primary_ids_and_key("safety", base)
    action_ids, action_key = primary_ids_and_key("actionability", base)
    safety = safety_ab_analysis(reviews["safety"]["a"], reviews["safety"]["b"], safety_ids, safety_key, base["safety_review_gates"])
    action = actionability_ab_analysis(reviews["actionability"]["a"], reviews["actionability"]["b"], action_ids, action_key, base["actionability_review_gates"])
    passed = safety["overall_pass"] and action["overall_pass"]
    route = "AUTHORIZE_CSPR_P2_REVIEWER_C_ONLY" if passed else "STOP_CSPR_P2_REVIEW_UNRELIABLE"
    c_artifacts: list[dict[str, Any]] = []
    if passed:
        for construct, analysis in (("safety", safety), ("actionability", action)):
            packet, template = build_c_packet(construct, analysis["disagreement_ids"], execution, base)
            cfg = execution["review_files"][construct]
            packet_path = resolve_repo_path(cfg["reviewer_c_packet"])
            template_path = resolve_repo_path(cfg["reviewer_c_template"])
            write_jsonl_frozen(packet_path, packet)
            write_jsonl_frozen(template_path, template)
            c_artifacts.extend(
                [
                    {"path": relative(packet_path), "rows": len(packet), "sha256": sha256(packet_path)},
                    {"path": relative(template_path), "rows": len(template), "sha256": sha256(template_path)},
                ]
            )
    report = {
        "name": "step3_cspr_p2_ab_review_analysis_v1",
        "route": route,
        "overall_pass": passed,
        "safety": safety,
        "actionability": action,
        "reviewer_provenance": [
            {
                "construct": row["construct"],
                "reviewer_role": row["reviewer_role"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "output_sha256": row["output_sha256"],
            }
            for row in provenances
        ],
        "reviewer_c_artifacts": c_artifacts,
        "reviewer_c_authorized": passed and bool(safety["disagreement_rows"] + action["disagreement_rows"]),
        "source_labels_used": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report_path = resolve_repo_path(execution["analysis_outputs"]["ab_analysis"])
    write_json_frozen(report_path, report)
    markdown = f"""# CSPR P2 A/B Review Analysis

Route: `{route}`.

Safety A/B primary agreement was `{safety['metrics']['raw_agreement']:.3f}` with Gwet AC1 `{safety['metrics']['gwet_ac1']}` and `{safety['disagreement_rows']}` disagreement rows. Actionability produced `{action['disagreement_rows']}` rows with at least one component disagreement. All prospective gates passed: `{passed}`.

Reviewer C is authorized only for the frozen disagreement packets when this route passes. Source labels and hidden pair roles were not used in A/B reliability. No model gradient or optimizer step was computed; `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(execution["analysis_outputs"]["ab_report"]), markdown)
    print(json.dumps({"route": route, "safety_disagreements": safety["disagreement_rows"], "actionability_disagreements": action["disagreement_rows"]}, sort_keys=True))
    return report


def validate_c_review(
    construct: str,
    execution: dict[str, Any],
    base: dict[str, Any],
    expected_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cfg = execution["review_files"][construct]
    id_field = "opaque_safety_id" if construct == "safety" else "opaque_action_id"
    fields = ("clean_safe",) if construct == "safety" else ACTION_FIELDS
    packet_path = resolve_repo_path(cfg["reviewer_c_packet"])
    template_path = resolve_repo_path(cfg["reviewer_c_template"])
    packet = read_jsonl(packet_path)
    template = read_jsonl(template_path)
    packet_ids = [str(row[id_field]) for row in packet]
    if packet_ids != sorted(expected_ids) or [str(row[id_field]) for row in template] != packet_ids:
        raise ReviewValidationError(f"{construct} Reviewer C packet does not match A/B disagreements")
    spec = {
        "construct": construct,
        "role": "c",
        "id_field": id_field,
        "decision_fields": fields,
        "packet": packet_path,
        "template": template_path,
        "rubric": resolve_repo_path(base["artifacts"]["safety_rubric" if construct == "safety" else "actionability_rubric"]["path"]),
        "output": resolve_repo_path(cfg["reviewer_c_labels"]),
        "provenance": resolve_repo_path(cfg["reviewer_c_provenance"]),
    }
    # C packets are created after the packet manifest, so validate their structure directly.
    rows = read_jsonl(spec["output"])
    if len(rows) != len(template):
        raise ReviewValidationError(f"{construct} Reviewer C incomplete review")
    mapped = {}
    expected_fields = {id_field, *fields, "short_reason"}
    for index, (row, blank) in enumerate(zip(rows, template, strict=True), start=1):
        if set(row) != expected_fields or row.get(id_field) != blank.get(id_field):
            raise ReviewValidationError(f"{construct} Reviewer C schema/order mismatch at row {index}")
        if any(row[field] not in YES_NO for field in fields):
            raise ReviewValidationError(f"{construct} Reviewer C invalid label at row {index}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ReviewValidationError(f"{construct} Reviewer C blank reason at row {index}")
        mapped[str(row[id_field])] = row
    provenance = read_json(spec["provenance"])
    expected = {
        "construct": construct,
        "reviewer_role": "c",
        "packet_path": relative(packet_path),
        "packet_sha256": sha256(packet_path),
        "template_path": relative(template_path),
        "template_sha256": sha256(template_path),
        "rubric_path": relative(spec["rubric"]),
        "rubric_sha256": sha256(spec["rubric"]),
        "output_path": relative(spec["output"]),
        "output_sha256": sha256(spec["output"]),
        "row_count": len(rows),
        "fresh_context": True,
        "forbidden_material_not_opened": True,
        "completion_status": "complete",
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ReviewValidationError(f"{construct} Reviewer C provenance mismatch: {field}")
    for field in ("agent_id", "session_id", "runtime"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ReviewValidationError(f"{construct} Reviewer C missing {field}")
    return mapped, provenance


def final_labels(
    construct: str,
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    reviewer_c: dict[str, dict[str, Any]],
    primary_ids: list[str],
) -> list[dict[str, Any]]:
    fields = ("clean_safe",) if construct == "safety" else ACTION_FIELDS
    id_field = "opaque_safety_id" if construct == "safety" else "opaque_action_id"
    output = []
    for row_id in primary_ids:
        row: dict[str, Any] = {id_field: row_id}
        adjudicated = []
        for field in fields:
            if reviewer_a[row_id][field] == reviewer_b[row_id][field]:
                row[field] = reviewer_a[row_id][field]
            else:
                row[field] = reviewer_c[row_id][field]
                adjudicated.append(field)
        row["adjudicated_fields"] = adjudicated
        if construct == "actionability":
            row["derived_score"] = derived_score(row)
        output.append(row)
    return output


def qualify_pairs(
    safety_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    safety_key: list[dict[str, Any]],
    action_key: list[dict[str, Any]],
    gates_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    safety_labels = {row["opaque_safety_id"]: row["clean_safe"] for row in safety_rows}
    action_labels = {row["opaque_action_id"]: row for row in action_rows}
    safety_primary = [row for row in safety_key if not row["is_hidden_duplicate"]]
    action_primary = [row for row in action_key if not row["is_hidden_duplicate"]]
    safety_by_pair_role = {(row["pair_id"], row["response_role"]): row for row in safety_primary}
    pool = []
    for action_key_row in sorted(action_primary, key=lambda row: row["pair_id"]):
        pair_id = action_key_row["pair_id"]
        chosen = safety_by_pair_role[(pair_id, "chosen")]
        rejected = safety_by_pair_role[(pair_id, "rejected")]
        action = action_labels[action_key_row["opaque_action_id"]]
        checks = {
            "candidate_clean_safe": safety_labels[chosen["opaque_safety_id"]] == "yes",
            "candidate_safe_route": action["safe_route"] == "yes",
            "candidate_executable_step": action["executable_step"] == "yes",
            "candidate_prompt_binding": action["prompt_binding"] == "yes",
            "rejected_clean_safe": safety_labels[rejected["opaque_safety_id"]] == "no",
        }
        pool.append(
            {
                "pair_id": pair_id,
                "prospective_split": action_key_row["prospective_split"],
                "assigned_category": action_key_row["assigned_category"],
                "candidate_safety_id": chosen["opaque_safety_id"],
                "rejected_safety_id": rejected["opaque_safety_id"],
                "actionability_id": action_key_row["opaque_action_id"],
                "qualification_checks": checks,
                "qualified": all(checks.values()),
            }
        )
    qualified = [row for row in pool if row["qualified"]]
    split_counts = Counter(row["prospective_split"] for row in qualified)
    category_counts = Counter(row["assigned_category"] for row in qualified)
    categories_with_two = sum(count >= 2 for count in category_counts.values())
    gates = {
        "train_candidate_yield_pass": split_counts["train_candidate"] >= int(gates_cfg["minimum_train_candidate_pairs"]),
        "calibration_yield_pass": split_counts["calibration"] >= int(gates_cfg["minimum_calibration_pairs"]),
        "mechanism_dev_yield_pass": split_counts["mechanism_dev"] >= int(gates_cfg["minimum_mechanism_dev_pairs"]),
        "total_yield_pass": len(qualified) >= int(gates_cfg["minimum_total_pairs"]),
        "category_diversity_pass": categories_with_two >= int(gates_cfg["minimum_categories_with_at_least_two_pairs"]),
    }
    return pool, {
        "total_pairs": len(pool),
        "qualified_pairs": len(qualified),
        "qualified_split_counts": dict(sorted(split_counts.items())),
        "qualified_category_counts": dict(sorted(category_counts.items())),
        "categories_with_at_least_two_pairs": categories_with_two,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def finalize(execution_path: Path) -> dict[str, Any]:
    execution, base, manifest = execution_context(execution_path)
    ab_path = resolve_repo_path(execution["analysis_outputs"]["ab_analysis"])
    ab = read_json(ab_path)
    if ab.get("route") != "AUTHORIZE_CSPR_P2_REVIEWER_C_ONLY" or not ab.get("overall_pass"):
        raise ReviewValidationError("A/B reliability did not authorize finalization")
    reviews, ab_provenances = load_ab(execution, base, manifest)
    safety_ids, safety_key = primary_ids_and_key("safety", base)
    action_ids, action_key = primary_ids_and_key("actionability", base)
    current_safety = safety_ab_analysis(reviews["safety"]["a"], reviews["safety"]["b"], safety_ids, safety_key, base["safety_review_gates"])
    current_action = actionability_ab_analysis(reviews["actionability"]["a"], reviews["actionability"]["b"], action_ids, action_key, base["actionability_review_gates"])
    if current_safety != ab["safety"] or current_action != ab["actionability"]:
        raise ReviewValidationError("A/B labels changed after their analysis freeze")

    c_provenances = []
    c_reviews: dict[str, dict[str, dict[str, Any]]] = {}
    for construct, disagreements in (("safety", current_safety["disagreement_ids"]), ("actionability", current_action["disagreement_ids"])):
        if disagreements:
            c_reviews[construct], provenance = validate_c_review(construct, execution, base, disagreements)
            c_provenances.append(provenance)
        else:
            c_reviews[construct] = {}
    ensure_unique_sessions(ab_provenances + c_provenances)

    safety_final = final_labels("safety", reviews["safety"]["a"], reviews["safety"]["b"], c_reviews["safety"], safety_ids)
    action_final = final_labels("actionability", reviews["actionability"]["a"], reviews["actionability"]["b"], c_reviews["actionability"], action_ids)
    pool, yield_analysis = qualify_pairs(safety_final, action_final, safety_key, action_key, base["fixed_pool_yield_gates"])
    route = "AUTHORIZE_CSPR_P3_ZERO_UPDATE_MECHANISM_PROTOCOL_ONLY" if yield_analysis["overall_pass"] else "STOP_CSPR_P2_PAIR_YIELD"
    outputs = execution["final_outputs"]
    safety_path = resolve_repo_path(outputs["safety_final_labels"])
    action_path = resolve_repo_path(outputs["actionability_final_labels"])
    pool_path = resolve_repo_path(outputs["qualified_pair_pool"])
    write_jsonl_frozen(safety_path, safety_final)
    write_jsonl_frozen(action_path, action_final)
    write_jsonl_frozen(pool_path, pool)
    analysis = {
        "name": "step3_cspr_p2_final_analysis_v1",
        "route": route,
        "safety_ab_analysis_sha256": sha256_text(json.dumps(current_safety, sort_keys=True)),
        "actionability_ab_analysis_sha256": sha256_text(json.dumps(current_action, sort_keys=True)),
        "safety_final_labels": {"path": relative(safety_path), "rows": len(safety_final), "sha256": sha256(safety_path)},
        "actionability_final_labels": {"path": relative(action_path), "rows": len(action_final), "sha256": sha256(action_path)},
        "qualified_pair_pool": {"path": relative(pool_path), "rows": len(pool), "sha256": sha256(pool_path)},
        "reviewer_c_rows": {"safety": len(c_reviews["safety"]), "actionability": len(c_reviews["actionability"])},
        "fixed_pool_yield": yield_analysis,
        "source_labels_used_as_final": False,
        "holistic_pair_preference_used": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    analysis_path = resolve_repo_path(outputs["final_analysis"])
    write_json_frozen(analysis_path, analysis)
    decision = {
        "name": "step3_cspr_p2_route_decision_v1",
        "route": route,
        "overall_pass": yield_analysis["overall_pass"],
        "qualified_pairs": yield_analysis["qualified_pairs"],
        "next_authorized_action": "freeze_a_separate_CSPR_P3_zero_update_mechanism_protocol" if yield_analysis["overall_pass"] else "stop_without_training_or_replacement_pairs",
        "new_training_authorized": False,
        "model_gradient_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_content_opened": False,
    }
    decision_path = resolve_repo_path(outputs["route_decision"])
    write_json_frozen(decision_path, decision)
    markdown = f"""# CSPR P2 Final Review Decision

Route: `{route}`.

After complete independent A/B review and disagreement-only C adjudication, `{yield_analysis['qualified_pairs']}` of `{yield_analysis['total_pairs']}` frozen pairs qualified. Qualified split counts were `{json.dumps(yield_analysis['qualified_split_counts'], sort_keys=True)}`; `{yield_analysis['categories_with_at_least_two_pairs']}` categories retained at least two pairs. Fixed-pool yield and diversity gates passed: `{yield_analysis['overall_pass']}`.

This result is a fixed-resource gate, not a population-rate estimate and not evidence that CSPR training works. It authorizes at most a separately prospective zero-update mechanism protocol. No gradient, checkpoint experiment, or optimizer step was run; `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(outputs["final_report"]), markdown)
    print(json.dumps({"route": route, "qualified_pairs": yield_analysis["qualified_pairs"]}, sort_keys=True))
    return decision


def record_inconclusive(
    execution_path: Path, construct: str | None, role: str | None, agent_id: str | None, reason: str | None
) -> dict[str, Any]:
    execution, _, _ = execution_context(execution_path)
    if not construct or not role or not agent_id or not reason:
        raise ReviewValidationError("inconclusive records require construct, role, agent ID, and reason")
    record = {
        "name": "step3_cspr_p2_review_inconclusive_v1",
        "route": "STOP_CSPR_P2_REVIEW_INCONCLUSIVE",
        "construct": construct,
        "reviewer_role": role,
        "agent_id": agent_id,
        "reason": reason,
        "heuristic_substitution_used": False,
        "replacement_reviewer_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(execution["analysis_outputs"]["inconclusive_incident"]), record)
    print(json.dumps({"route": record["route"], "construct": construct, "reviewer_role": role}, sort_keys=True))
    return record


def main() -> None:
    args = parse_args()
    execution_path = resolve_repo_path(args.execution_protocol)
    try:
        if args.command == "preflight":
            preflight(execution_path)
        elif args.command == "analyze-ab":
            analyze_ab(execution_path)
        elif args.command == "finalize":
            finalize(execution_path)
        else:
            record_inconclusive(
                execution_path, args.construct, args.reviewer_role, args.agent_id, args.reason
            )
    except (ReviewValidationError, FileExistsError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2_REVIEW_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
