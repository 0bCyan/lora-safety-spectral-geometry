"""Validate and analyze frozen OACPO D1 decomposed A/B reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cspr_p2_review as metrics


REPO = Path(__file__).resolve().parents[3]
YES_NO = {"yes", "no"}


class AnalysisError(ValueError):
    """Raised when a frozen D1 input is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "analyze-ab"])
    parser.add_argument("--protocol", required=True)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
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


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_frozen(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise AnalysisError(f"refusing to overwrite different frozen artifact: {relative(path)}")
    path.write_text(text, encoding="utf-8", newline="\n")


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    for label, artifact in protocol["frozen_inputs"].items():
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise AnalysisError(f"frozen input mismatch: {label}")


def construct_spec(protocol: dict[str, Any], construct: str, role: str) -> dict[str, Any]:
    review = protocol["review_files"][construct]
    if construct == "safety":
        return {
            "construct": construct,
            "role": role,
            "id_field": "opaque_safety_id",
            "decision_fields": ("clean_safe",),
            "packet": resolve_repo_path(review[f"reviewer_{role}_packet"]),
            "template": resolve_repo_path(review[f"reviewer_{role}_template"]),
            "rubric": resolve_repo_path(protocol["rubrics"][construct]),
            "output": resolve_repo_path(review[f"reviewer_{role}_labels"]),
            "provenance": resolve_repo_path(review[f"reviewer_{role}_provenance"]),
        }
    if construct == "quality":
        return {
            "construct": construct,
            "role": role,
            "id_field": "opaque_quality_id",
            "decision_fields": ("detail_retention", "safe_actionability", "naturalness"),
            "packet": resolve_repo_path(review[f"reviewer_{role}_packet"]),
            "template": resolve_repo_path(review[f"reviewer_{role}_template"]),
            "rubric": resolve_repo_path(protocol["rubrics"][construct]),
            "output": resolve_repo_path(review[f"reviewer_{role}_labels"]),
            "provenance": resolve_repo_path(review[f"reviewer_{role}_provenance"]),
        }
    if construct == "structure":
        return {
            "construct": construct,
            "role": role,
            "id_field": "opaque_structure_id",
            "decision_fields": ("mechanical_template",),
            "packet": resolve_repo_path(review[f"reviewer_{role}_packet"]),
            "template": resolve_repo_path(review[f"reviewer_{role}_template"]),
            "rubric": resolve_repo_path(protocol["rubrics"][construct]),
            "output": resolve_repo_path(review[f"reviewer_{role}_labels"]),
            "provenance": resolve_repo_path(review[f"reviewer_{role}_provenance"]),
        }
    raise AnalysisError(f"unknown construct: {construct}")


def validate_review(spec: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    packet = read_jsonl(spec["packet"])
    template = read_jsonl(spec["template"])
    rows = read_jsonl(spec["output"])
    id_field = spec["id_field"]
    expected_fields = {id_field, *spec["decision_fields"], "short_reason"}
    if len(packet) != len(template) or len(rows) != len(template):
        raise AnalysisError(f"incomplete {spec['construct']} reviewer {spec['role']}")
    packet_ids = [row.get(id_field) for row in packet]
    template_ids = [row.get(id_field) for row in template]
    output_ids = [row.get(id_field) for row in rows]
    if packet_ids != template_ids or template_ids != output_ids or len(set(output_ids)) != len(output_ids):
        raise AnalysisError(f"ID/order mismatch: {spec['construct']} reviewer {spec['role']}")

    mapped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if set(row) != expected_fields:
            raise AnalysisError(f"schema mismatch at {spec['construct']} row {index}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise AnalysisError(f"blank short_reason at {spec['construct']} row {index}")
        if spec["construct"] == "safety" and row["clean_safe"] not in YES_NO:
            raise AnalysisError(f"invalid clean_safe at row {index}")
        if spec["construct"] == "quality":
            if row["detail_retention"] not in (0, 1, 2) or row["safe_actionability"] not in (0, 1, 2):
                raise AnalysisError(f"invalid quality ordinal at row {index}")
            if row["naturalness"] not in YES_NO:
                raise AnalysisError(f"invalid naturalness at row {index}")
        if spec["construct"] == "structure" and row["mechanical_template"] not in YES_NO:
            raise AnalysisError(f"invalid mechanical_template at row {index}")
        mapped[str(row[id_field])] = row

    provenance = read_json(spec["provenance"])
    expected_provenance = {
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
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise AnalysisError(f"provenance mismatch: {spec['construct']} {spec['role']} {field}")
    if not isinstance(provenance.get("agent_id"), str) or not provenance["agent_id"]:
        raise AnalysisError("missing agent_id")
    return mapped, provenance


def duplicate_consistency(
    review: dict[str, dict[str, Any]],
    key_rows: list[dict[str, Any]],
    id_field: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    details = []
    full = 0
    for key in key_rows:
        if not key["is_hidden_duplicate"]:
            continue
        duplicate_id = str(key[id_field])
        primary_id = str(key["duplicate_of"])
        matches = {field: review[duplicate_id][field] == review[primary_id][field] for field in fields}
        is_full = all(matches.values())
        full += int(is_full)
        details.append(
            {
                "primary_id": primary_id,
                "duplicate_id": duplicate_id,
                "field_matches": matches,
                "full_vector_match": is_full,
            }
        )
    return {"denominator": len(details), "full_vector_consistent_count": full, "details": details}


def at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def analyze_constructs(
    protocol: dict[str, Any], reviews: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    base_protocol = read_json(resolve_repo_path(protocol["frozen_inputs"]["base_protocol"]["path"]))
    results: dict[str, Any] = {}
    disagreements: list[dict[str, Any]] = []

    # Safety
    safety_key = read_jsonl(resolve_repo_path(base_protocol["outputs"]["safety_key"]))
    safety_ids = sorted(str(row["opaque_safety_id"]) for row in safety_key if not row["is_hidden_duplicate"])
    safety_pairs = [
        (reviews["safety"]["a"][row_id]["clean_safe"], reviews["safety"]["b"][row_id]["clean_safe"])
        for row_id in safety_ids
    ]
    safety_metrics = metrics.nominal_metrics(safety_pairs)
    safety_duplicates = {
        role: duplicate_consistency(reviews["safety"][role], safety_key, "opaque_safety_id", ("clean_safe",))
        for role in ("a", "b")
    }
    safety_cfg = base_protocol["safety_reliability_gates"]
    safety_counts = {
        role: Counter(reviews["safety"][role][row_id]["clean_safe"] for row_id in safety_ids)
        for role in ("a", "b")
    }
    safety_gates = {
        "raw_agreement": at_least(safety_metrics["raw_agreement"], safety_cfg["primary_raw_agreement_min"]),
        "gwet_ac1": at_least(safety_metrics["gwet_ac1"], safety_cfg["primary_gwet_ac1_min"]),
        "positive_agreement": at_least(safety_metrics["positive_agreement"], safety_cfg["positive_agreement_min"]),
        "negative_agreement": at_least(safety_metrics["negative_agreement"], safety_cfg["negative_agreement_min"]),
        "prevalence": all(
            safety_counts[role]["yes"] >= safety_cfg["minimum_yes_ratings_per_reviewer"]
            and safety_counts[role]["no"] >= safety_cfg["minimum_no_ratings_per_reviewer"]
            for role in ("a", "b")
        ),
        "duplicates": all(
            safety_duplicates[role]["full_vector_consistent_count"]
            >= safety_cfg["full_vector_duplicate_consistency_min"]
            for role in ("a", "b")
        ),
    }
    for row_id in safety_ids:
        if reviews["safety"]["a"][row_id]["clean_safe"] != reviews["safety"]["b"][row_id]["clean_safe"]:
            disagreements.append({"construct": "safety", "opaque_id": row_id, "disputed_fields": ["clean_safe"]})
    results["safety"] = {
        "metrics": safety_metrics,
        "primary_counts": {role: dict(counts) for role, counts in safety_counts.items()},
        "duplicates": safety_duplicates,
        "gates": safety_gates,
        "pass": all(safety_gates.values()),
    }

    # Quality
    quality_key = read_jsonl(resolve_repo_path(base_protocol["outputs"]["quality_key"]))
    quality_ids = sorted(str(row["opaque_quality_id"]) for row in quality_key if not row["is_hidden_duplicate"])
    quality_cfg = base_protocol["quality_reliability_gates"]
    ordinal_results = {}
    ordinal_gates = {}
    for field in ("detail_retention", "safe_actionability"):
        pairs = [(reviews["quality"]["a"][row_id][field], reviews["quality"]["b"][row_id][field]) for row_id in quality_ids]
        result = metrics.ordinal_metrics(pairs)
        ordinal_results[field] = result
        ordinal_gates[field] = {
            "exact": at_least(result["exact_raw_agreement"], quality_cfg["per_field_exact_agreement_min"]),
            "weighted": at_least(result["linear_weighted_agreement"], quality_cfg["per_field_linear_weighted_agreement_min"]),
            "gwet_ac2": at_least(result["gwet_ac2"], quality_cfg["per_field_gwet_ac2_min"]),
            "two_level": result["two_level_disagreement_rate"]
            <= quality_cfg["per_field_two_level_disagreement_rate_max"],
        }
    natural_pairs = [
        (reviews["quality"]["a"][row_id]["naturalness"], reviews["quality"]["b"][row_id]["naturalness"])
        for row_id in quality_ids
    ]
    natural_metrics = metrics.nominal_metrics(natural_pairs)
    quality_duplicates = {
        role: duplicate_consistency(
            reviews["quality"][role],
            quality_key,
            "opaque_quality_id",
            ("detail_retention", "safe_actionability", "naturalness"),
        )
        for role in ("a", "b")
    }
    quality_gates = {
        "ordinal_fields": all(all(gates.values()) for gates in ordinal_gates.values()),
        "naturalness": at_least(
            natural_metrics["raw_agreement"], quality_cfg["naturalness_raw_agreement_min"]
        ),
        "duplicates": all(
            quality_duplicates[role]["full_vector_consistent_count"]
            >= quality_cfg["full_vector_duplicate_consistency_min"]
            for role in ("a", "b")
        ),
    }
    for row_id in quality_ids:
        fields = [
            field
            for field in ("detail_retention", "safe_actionability", "naturalness")
            if reviews["quality"]["a"][row_id][field] != reviews["quality"]["b"][row_id][field]
        ]
        if fields:
            disagreements.append({"construct": "quality", "opaque_id": row_id, "disputed_fields": fields})
    results["quality"] = {
        "ordinal_metrics": ordinal_results,
        "ordinal_gates": ordinal_gates,
        "naturalness_metrics": natural_metrics,
        "duplicates": quality_duplicates,
        "gates": quality_gates,
        "pass": all(quality_gates.values()),
    }

    # Structure
    structure_key = read_jsonl(resolve_repo_path(base_protocol["outputs"]["structure_key"]))
    structure_ids = sorted(
        str(row["opaque_structure_id"]) for row in structure_key if not row["is_hidden_duplicate"]
    )
    structure_pairs = [
        (
            reviews["structure"]["a"][row_id]["mechanical_template"],
            reviews["structure"]["b"][row_id]["mechanical_template"],
        )
        for row_id in structure_ids
    ]
    structure_metrics = metrics.nominal_metrics(structure_pairs)
    structure_duplicates = {
        role: duplicate_consistency(
            reviews["structure"][role], structure_key, "opaque_structure_id", ("mechanical_template",)
        )
        for role in ("a", "b")
    }
    structure_cfg = base_protocol["structure_reliability_gates"]
    structure_counts = {
        role: Counter(
            reviews["structure"][role][row_id]["mechanical_template"] for row_id in structure_ids
        )
        for role in ("a", "b")
    }
    structure_gates = {
        "raw_agreement": at_least(
            structure_metrics["raw_agreement"], structure_cfg["mechanical_template_raw_agreement_min"]
        ),
        "gwet_ac1": at_least(
            structure_metrics["gwet_ac1"], structure_cfg["mechanical_template_gwet_ac1_min"]
        ),
        "positive_agreement": at_least(
            structure_metrics["positive_agreement"], structure_cfg["positive_agreement_min"]
        ),
        "negative_agreement": at_least(
            structure_metrics["negative_agreement"], structure_cfg["negative_agreement_min"]
        ),
        "prevalence": all(
            structure_counts[role]["yes"] >= structure_cfg["minimum_yes_ratings_per_reviewer"]
            and structure_counts[role]["no"] >= structure_cfg["minimum_no_ratings_per_reviewer"]
            for role in ("a", "b")
        ),
        "duplicates": all(
            structure_duplicates[role]["full_vector_consistent_count"]
            >= structure_cfg["full_vector_duplicate_consistency_min"]
            for role in ("a", "b")
        ),
    }
    for row_id in structure_ids:
        if (
            reviews["structure"]["a"][row_id]["mechanical_template"]
            != reviews["structure"]["b"][row_id]["mechanical_template"]
        ):
            disagreements.append(
                {"construct": "structure", "opaque_id": row_id, "disputed_fields": ["mechanical_template"]}
            )
    results["structure"] = {
        "metrics": structure_metrics,
        "primary_counts": {role: dict(counts) for role, counts in structure_counts.items()},
        "duplicates": structure_duplicates,
        "gates": structure_gates,
        "pass": all(structure_gates.values()),
    }
    return {"constructs": results, "disagreements": disagreements}


def route_for(constructs: dict[str, Any], disagreement_count: int) -> str:
    if not all(block["pass"] for block in constructs.values()):
        return "STOP_OACPO_D1_REVIEW_UNRELIABLE"
    if disagreement_count:
        return "AUTHORIZE_OACPO_D1_REVIEWER_C_PROTOCOL_ONLY"
    return "AUTHORIZE_OACPO_D2_TRIPLET_QUALIFICATION_PROTOCOL_ONLY"


def main() -> None:
    args = parse_args()
    protocol = read_json(resolve_repo_path(args.protocol))
    if protocol["self_path"] != args.protocol.replace("\\", "/"):
        raise AnalysisError("protocol self path mismatch")
    verify_frozen_inputs(protocol)
    if args.command == "preflight":
        print("PASS_OACPO_D1_ANALYSIS_PREFLIGHT")
        return

    reviews: dict[str, dict[str, dict[str, Any]]] = {}
    provenances = []
    for construct in ("safety", "quality", "structure"):
        reviews[construct] = {}
        for role in ("a", "b"):
            review, provenance = validate_review(construct_spec(protocol, construct, role))
            reviews[construct][role] = review
            provenances.append(provenance)
    analyzed = analyze_constructs(protocol, reviews)
    route = route_for(analyzed["constructs"], len(analyzed["disagreements"]))
    result = {
        "name": "step3_oacpo_d1_ab_reliability_analysis_v1",
        "route": route,
        "all_constructs_pass": all(block["pass"] for block in analyzed["constructs"].values()),
        "constructs": analyzed["constructs"],
        "primary_disagreement_count": len(analyzed["disagreements"]),
        "primary_disagreement_counts_by_construct": dict(
            Counter(row["construct"] for row in analyzed["disagreements"])
        ),
        "reviewer_provenance": [
            {
                "construct": row["construct"],
                "reviewer_role": row["reviewer_role"],
                "agent_id": row["agent_id"],
                "output_sha256": row["output_sha256"],
            }
            for row in provenances
        ],
        "reviewer_c_started": False,
        "triplet_qualification_run": False,
        "source_labels_used": False,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_frozen(resolve_repo_path(protocol["outputs"]["analysis"]), json_text(result))
    write_frozen(
        resolve_repo_path(protocol["outputs"]["disagreement_ledger"]),
        jsonl_text(analyzed["disagreements"]),
    )
    report = f"""# OACPO D1 A/B Reliability Analysis

Route: `{route}`.

Safety pass: `{analyzed['constructs']['safety']['pass']}`. Quality pass: `{analyzed['constructs']['quality']['pass']}`. Structure pass: `{analyzed['constructs']['structure']['pass']}`. Primary disagreement rows: `{len(analyzed['disagreements'])}`.

This stage validates only reviewer reliability and hidden-duplicate consistency. It does not adjudicate disagreements, qualify triplets, compute a gradient, open a checkpoint, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(protocol["outputs"]["report"]), report)
    print(json.dumps({"route": route, "disagreements": len(analyzed["disagreements"])}, sort_keys=True))


if __name__ == "__main__":
    main()
