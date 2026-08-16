"""Validate and adjudicate calibration-separated CSPR P2S blinded reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import step3_cspr_p2_review as metrics
import step3_cspr_p2s_packets as packet_builder


REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTION_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2s/"
    "protocol/cspr_p2s_review_execution_protocol.json"
)
ACTION_FIELDS = ("safe_route", "executable_step", "prompt_binding")
YES_NO = {"yes", "no"}


class ReviewValidationError(ValueError):
    """Raised when a frozen P2S review input is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "analyze-ab", "finalize", "record-inconclusive"])
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = read_json(path)
    for name, artifact in execution["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise ReviewValidationError(f"frozen execution input hash mismatch: {name}")
    protocol_path = resolve_repo_path(execution["frozen_inputs"]["review_protocol"]["path"])
    manifest_path = resolve_repo_path(execution["frozen_inputs"]["packet_manifest"]["path"])
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    if manifest.get("protocol_sha256") != sha256(protocol_path):
        raise ReviewValidationError("packet manifest does not bind the review protocol")
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise ReviewValidationError(f"packet artifact hash mismatch: {artifact['path']}")
    if execution.get("optimizer_steps") != 0 or execution.get("training_authorized") is not False:
        raise ReviewValidationError("execution protocol violates zero-training boundary")
    return execution, protocol, manifest


def manifest_hash(manifest: dict[str, Any], path: Path) -> str:
    matches = [row["sha256"] for row in manifest["artifacts"] if row["path"] == relative(path)]
    if len(matches) != 1:
        raise ReviewValidationError(f"packet artifact is not uniquely registered: {path}")
    return str(matches[0])


def review_spec(
    construct: str, role: str, execution: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    files = execution["review_files"][construct]
    if construct == "safety":
        return {
            "construct": construct,
            "role": role,
            "id_field": "opaque_safety_id",
            "fields": ("clean_safe",),
            "packet": resolve_repo_path(protocol["packet_outputs"][f"safety_{role}_packet"]),
            "template": resolve_repo_path(protocol["packet_outputs"][f"safety_{role}_template"]),
            "rubric": resolve_repo_path(protocol["frozen_inputs"]["safety_rubric"]["path"]),
            "output": resolve_repo_path(files[f"reviewer_{role}_labels"]),
            "provenance": resolve_repo_path(files[f"reviewer_{role}_provenance"]),
        }
    return {
        "construct": construct,
        "role": role,
        "id_field": "opaque_action_id",
        "fields": ACTION_FIELDS,
        "packet": resolve_repo_path(protocol["packet_outputs"][f"action_{role}_packet"]),
        "template": resolve_repo_path(protocol["packet_outputs"][f"action_{role}_template"]),
        "rubric": resolve_repo_path(protocol["frozen_inputs"]["action_rubric"]["path"]),
        "output": resolve_repo_path(files[f"reviewer_{role}_labels"]),
        "provenance": resolve_repo_path(files[f"reviewer_{role}_provenance"]),
    }


def validate_packet_template(spec: dict[str, Any], manifest: dict[str, Any]) -> None:
    packet = read_jsonl(spec["packet"])
    template = read_jsonl(spec["template"])
    if sha256(spec["packet"]) != manifest_hash(manifest, spec["packet"]):
        raise ReviewValidationError("packet hash mismatch")
    if sha256(spec["template"]) != manifest_hash(manifest, spec["template"]):
        raise ReviewValidationError("template hash mismatch")
    id_field = spec["id_field"]
    if [row[id_field] for row in packet] != [row[id_field] for row in template]:
        raise ReviewValidationError(f"{spec['construct']} packet/template order mismatch")
    expected = {id_field, *spec["fields"], "short_reason"}
    for row in template:
        if set(row) != expected or any(row[field] != "" for field in spec["fields"] + ("short_reason",)):
            raise ReviewValidationError(f"{spec['construct']} template schema or blank mismatch")


def validate_review(
    spec: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validate_packet_template(spec, manifest)
    template = read_jsonl(spec["template"])
    rows = read_jsonl(spec["output"])
    if len(rows) != len(template):
        raise ReviewValidationError(f"{spec['construct']} {spec['role']} incomplete review")
    expected = {spec["id_field"], *spec["fields"], "short_reason"}
    mapped = {}
    for index, (row, blank) in enumerate(zip(rows, template, strict=True), start=1):
        opaque_id = row.get(spec["id_field"])
        if set(row) != expected or opaque_id != blank[spec["id_field"]] or opaque_id in mapped:
            raise ReviewValidationError(f"{spec['construct']} {spec['role']} schema/order mismatch at {index}")
        if any(row[field] not in YES_NO for field in spec["fields"]):
            raise ReviewValidationError(f"invalid review label at {opaque_id}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ReviewValidationError(f"blank review reason at {opaque_id}")
        mapped[str(opaque_id)] = row
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
    for field, value in expected_provenance.items():
        if provenance.get(field) != value:
            raise ReviewValidationError(f"{spec['construct']} {spec['role']} provenance mismatch: {field}")
    for field in ("agent_id", "session_id", "runtime"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ReviewValidationError(f"missing provenance {field}")
    return mapped, provenance


def ensure_reviewer_isolation(
    provenances: list[dict[str, Any]], forbidden_agent_ids: list[str] | set[str]
) -> None:
    for provenance in provenances:
        for field in ("agent_id", "session_id"):
            if not isinstance(provenance.get(field), str) or not provenance[field].strip():
                raise ReviewValidationError(f"missing reviewer provenance {field}")
    if len({row["agent_id"] for row in provenances}) != len(provenances):
        raise ReviewValidationError("reviewer agent reuse is forbidden")
    if len({row["session_id"] for row in provenances}) != len(provenances):
        raise ReviewValidationError("reviewer session reuse is forbidden")
    forbidden = set(forbidden_agent_ids)
    if any(row["agent_id"] in forbidden for row in provenances):
        raise ReviewValidationError("P2S reviewer reuses an author or old P2 reviewer")


def load_ab(
    execution: dict[str, Any], protocol: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    reviews = {}
    provenances = []
    for construct in ("safety", "actionability"):
        reviews[construct] = {}
        for role in ("a", "b"):
            reviews[construct][role], provenance = validate_review(
                review_spec(construct, role, execution, protocol), manifest
            )
            provenances.append(provenance)
    ensure_reviewer_isolation(provenances, execution["forbidden_prior_agent_ids"])
    return reviews, provenances


def duplicate_consistency(
    review: dict[str, dict[str, Any]], key_rows: list[dict[str, Any]], id_field: str, fields: tuple[str, ...]
) -> dict[str, Any]:
    details = []
    field_counts = Counter()
    full_count = 0
    duplicates = [row for row in key_rows if row["is_hidden_duplicate"]]
    for row in duplicates:
        matches = {
            field: review[str(row[id_field])][field] == review[str(row["duplicate_of"])][field]
            for field in fields
        }
        field_counts.update(field for field, matched in matches.items() if matched)
        full_count += int(all(matches.values()))
        details.append(
            {
                "duplicate_id": row[id_field],
                "primary_id": row["duplicate_of"],
                "field_matches": matches,
                "full_vector_match": all(matches.values()),
            }
        )
    return {
        "denominator": len(duplicates),
        "full_vector_consistent_count": full_count,
        "field_consistent_counts": dict(field_counts),
        "details": details,
    }


def action_metric_block(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], ids: list[str]
) -> dict[str, Any]:
    components = {
        field: metrics.nominal_metrics([(left[row_id][field], right[row_id][field]) for row_id in ids])
        for field in ACTION_FIELDS
    }
    scores_left = {row_id: metrics.derived_score(left[row_id]) for row_id in ids}
    scores_right = {row_id: metrics.derived_score(right[row_id]) for row_id in ids}
    ordinal_pairs = [(scores_left[row_id], scores_right[row_id]) for row_id in ids]
    boundary_pairs = [
        ("yes" if scores_left[row_id] == 2 else "no", "yes" if scores_right[row_id] == 2 else "no")
        for row_id in ids
    ]
    return {
        "components": components,
        "ordinal": metrics.ordinal_metrics(ordinal_pairs),
        "boundary": metrics.nominal_metrics(boundary_pairs),
        "ordinal_pairs": ordinal_pairs,
        "boundary_pairs": boundary_pairs,
        "score_prevalence": {
            "left_score2": sum(value == 2 for value in scores_left.values()) / len(ids),
            "right_score2": sum(value == 2 for value in scores_right.values()) / len(ids),
            "left_score1": sum(value == 1 for value in scores_left.values()) / len(ids),
            "right_score1": sum(value == 1 for value in scores_right.values()) / len(ids),
        },
        "score_distributions": {
            "left": dict(Counter(str(value) for value in scores_left.values())),
            "right": dict(Counter(str(value) for value in scores_right.values())),
        },
    }


def safety_analysis(
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    key_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    calibration = sorted(
        str(row["opaque_safety_id"])
        for row in key_rows
        if not row["is_hidden_duplicate"] and row["item_type"] == "calibration_anchor"
    )
    target = sorted(
        str(row["opaque_safety_id"])
        for row in key_rows
        if not row["is_hidden_duplicate"] and row["item_type"].startswith("target_")
    )
    reference = {
        str(row["opaque_safety_id"]): {"clean_safe": row["reference_clean_safe"]}
        for row in key_rows
        if not row["is_hidden_duplicate"] and row["item_type"] == "calibration_anchor"
    }
    calibration_ab = metrics.nominal_metrics(
        [(reviewer_a[row_id]["clean_safe"], reviewer_b[row_id]["clean_safe"]) for row_id in calibration]
    )
    target_ab = metrics.nominal_metrics(
        [(reviewer_a[row_id]["clean_safe"], reviewer_b[row_id]["clean_safe"]) for row_id in target]
    )
    reference_metrics = {
        role: metrics.nominal_metrics(
            [(review[row_id]["clean_safe"], reference[row_id]["clean_safe"]) for row_id in calibration]
        )
        for role, review in (("a", reviewer_a), ("b", reviewer_b))
    }
    duplicate = {
        "a": duplicate_consistency(reviewer_a, key_rows, "opaque_safety_id", ("clean_safe",)),
        "b": duplicate_consistency(reviewer_b, key_rows, "opaque_safety_id", ("clean_safe",)),
    }
    cal_cfg = protocol["safety_calibration_gates"]
    target_cfg = protocol["target_reliability_gates"]
    dup_cfg = protocol["duplicate_gates"]
    calibration_gates = {
        "ab_raw": calibration_ab["raw_agreement"] >= cal_cfg["ab_raw_agreement_min"],
        "ab_ac1": calibration_ab["gwet_ac1"] is not None and calibration_ab["gwet_ac1"] >= cal_cfg["ab_gwet_ac1_min"],
        "ab_positive": calibration_ab["positive_agreement"] is not None and calibration_ab["positive_agreement"] >= cal_cfg["ab_positive_agreement_min"],
        "ab_negative": calibration_ab["negative_agreement"] is not None and calibration_ab["negative_agreement"] >= cal_cfg["ab_negative_agreement_min"],
    }
    for role, block in reference_metrics.items():
        calibration_gates[f"{role}_reference_raw"] = block["raw_agreement"] >= cal_cfg["per_reviewer_reference_raw_min"]
        calibration_gates[f"{role}_reference_ac1"] = block["gwet_ac1"] is not None and block["gwet_ac1"] >= cal_cfg["per_reviewer_reference_gwet_ac1_min"]
        calibration_gates[f"{role}_reference_positive"] = block["positive_agreement"] is not None and block["positive_agreement"] >= cal_cfg["per_reviewer_reference_positive_min"]
        calibration_gates[f"{role}_reference_negative"] = block["negative_agreement"] is not None and block["negative_agreement"] >= cal_cfg["per_reviewer_reference_negative_min"]
    target_gates = {
        "raw": target_ab["raw_agreement"] >= target_cfg["safety_raw_agreement_min"],
        "ac1": target_ab["gwet_ac1"] is not None and target_ab["gwet_ac1"] >= target_cfg["safety_gwet_ac1_min"],
        "positive": target_ab["positive_agreement"] is not None and target_ab["positive_agreement"] >= target_cfg["safety_positive_agreement_min"],
        "negative": target_ab["negative_agreement"] is not None and target_ab["negative_agreement"] >= target_cfg["safety_negative_agreement_min"],
    }
    duplicate_gates = {
        role: block["denominator"] == dup_cfg["safety_denominator"]
        and block["full_vector_consistent_count"] >= dup_cfg["safety_full_vector_consistency_min_count"]
        for role, block in duplicate.items()
    }
    disagreements = [
        row_id for row_id in target if reviewer_a[row_id]["clean_safe"] != reviewer_b[row_id]["clean_safe"]
    ]
    return {
        "calibration_rows": len(calibration),
        "target_rows": len(target),
        "calibration_ab": calibration_ab,
        "calibration_reference": reference_metrics,
        "target_ab": target_ab,
        "duplicate_consistency": duplicate,
        "calibration_gates": calibration_gates,
        "target_gates": target_gates,
        "duplicate_gates": duplicate_gates,
        "calibration_pass": all(calibration_gates.values()),
        "target_pass": all(target_gates.values()),
        "duplicate_pass": all(duplicate_gates.values()),
        "target_disagreement_ids": disagreements,
        "target_disagreement_rows": len(disagreements),
    }


def action_analysis(
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    key_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    calibration = sorted(
        str(row["opaque_action_id"])
        for row in key_rows
        if not row["is_hidden_duplicate"] and row["item_type"] == "calibration_anchor"
    )
    target = sorted(
        str(row["opaque_action_id"])
        for row in key_rows
        if not row["is_hidden_duplicate"] and row["item_type"] == "target_candidate"
    )
    reference = {
        str(row["opaque_action_id"]): {
            "safe_route": row["reference_safe_route"],
            "executable_step": row["reference_executable_step"],
            "prompt_binding": row["reference_prompt_binding"],
        }
        for row in key_rows
        if not row["is_hidden_duplicate"] and row["item_type"] == "calibration_anchor"
    }
    calibration_ab = action_metric_block(reviewer_a, reviewer_b, calibration)
    target_ab = action_metric_block(reviewer_a, reviewer_b, target)
    reference_metrics = {
        role: action_metric_block(review, reference, calibration)
        for role, review in (("a", reviewer_a), ("b", reviewer_b))
    }
    duplicate = {
        "a": duplicate_consistency(reviewer_a, key_rows, "opaque_action_id", ACTION_FIELDS),
        "b": duplicate_consistency(reviewer_b, key_rows, "opaque_action_id", ACTION_FIELDS),
    }
    cfg = protocol["action_calibration_gates"]
    target_cfg = protocol["target_reliability_gates"]
    duplicate_cfg = protocol["duplicate_gates"]
    uncertainty = {
        "ordinal_gwet_ac2": metrics.bootstrap_interval(
            calibration_ab["ordinal_pairs"],
            lambda rows: metrics.gwet_ac2(rows, [0, 1, 2]),
            replicates=int(cfg["bootstrap_replicates"]),
            seed=int(cfg["bootstrap_seed"]),
        ),
        "score2_boundary_gwet_ac1": metrics.bootstrap_interval(
            calibration_ab["boundary_pairs"],
            lambda rows: metrics.gwet_ac1(rows, ["yes", "no"]),
            replicates=int(cfg["bootstrap_replicates"]),
            seed=int(cfg["bootstrap_seed"]) + 1,
        ),
    }
    calibration_gates = {}
    for field, block in calibration_ab["components"].items():
        calibration_gates[f"ab_{field}_raw"] = block["raw_agreement"] >= cfg["component_ab_raw_min"]
        calibration_gates[f"ab_{field}_ac1"] = block["gwet_ac1"] is not None and block["gwet_ac1"] >= cfg["component_ab_gwet_ac1_min"]
    calibration_gates.update(
        {
            "ab_ordinal_exact": calibration_ab["ordinal"]["exact_raw_agreement"] >= cfg["ordinal_ab_exact_min"],
            "ab_ordinal_weighted": calibration_ab["ordinal"]["linear_weighted_agreement"] >= cfg["ordinal_ab_weighted_min"],
            "ab_ordinal_ac2": calibration_ab["ordinal"]["gwet_ac2"] is not None and calibration_ab["ordinal"]["gwet_ac2"] >= cfg["ordinal_ab_gwet_ac2_min"],
            "ab_two_level": calibration_ab["ordinal"]["two_level_disagreement_rate"] <= cfg["ordinal_two_level_disagreement_rate_max"],
            "ab_boundary_raw": calibration_ab["boundary"]["raw_agreement"] >= cfg["score2_boundary_ab_raw_min"],
            "ab_boundary_ac1": calibration_ab["boundary"]["gwet_ac1"] is not None and calibration_ab["boundary"]["gwet_ac1"] >= cfg["score2_boundary_ab_gwet_ac1_min"],
            "ab_boundary_positive": calibration_ab["boundary"]["positive_agreement"] is not None and calibration_ab["boundary"]["positive_agreement"] >= cfg["score2_boundary_ab_positive_min"],
            "ab_boundary_negative": calibration_ab["boundary"]["negative_agreement"] is not None and calibration_ab["boundary"]["negative_agreement"] >= cfg["score2_boundary_ab_negative_min"],
            "ab_score2_gap": abs(calibration_ab["score_prevalence"]["left_score2"] - calibration_ab["score_prevalence"]["right_score2"]) <= cfg["score2_marginal_prevalence_gap_max"],
            "a_score2_informative": cfg["reviewer_score2_prevalence_min"] <= calibration_ab["score_prevalence"]["left_score2"] <= cfg["reviewer_score2_prevalence_max"],
            "b_score2_informative": cfg["reviewer_score2_prevalence_min"] <= calibration_ab["score_prevalence"]["right_score2"] <= cfg["reviewer_score2_prevalence_max"],
            "a_score1_informative": calibration_ab["score_prevalence"]["left_score1"] >= cfg["reviewer_score1_prevalence_min"],
            "b_score1_informative": calibration_ab["score_prevalence"]["right_score1"] >= cfg["reviewer_score1_prevalence_min"],
            "bootstrap_ordinal_lower": uncertainty["ordinal_gwet_ac2"]["lower_95"] is not None and uncertainty["ordinal_gwet_ac2"]["lower_95"] >= cfg["ordinal_gwet_ac2_lower_95_min"],
            "bootstrap_boundary_lower": uncertainty["score2_boundary_gwet_ac1"]["lower_95"] is not None and uncertainty["score2_boundary_gwet_ac1"]["lower_95"] >= cfg["score2_boundary_gwet_ac1_lower_95_min"],
        }
    )
    for role, block in reference_metrics.items():
        for field, component in block["components"].items():
            calibration_gates[f"{role}_reference_{field}_raw"] = component["raw_agreement"] >= cfg["per_reviewer_reference_component_raw_min"]
            calibration_gates[f"{role}_reference_{field}_ac1"] = component["gwet_ac1"] is not None and component["gwet_ac1"] >= cfg["per_reviewer_reference_component_gwet_ac1_min"]
        calibration_gates[f"{role}_reference_ordinal_exact"] = block["ordinal"]["exact_raw_agreement"] >= cfg["per_reviewer_reference_ordinal_exact_min"]
        calibration_gates[f"{role}_reference_ordinal_weighted"] = block["ordinal"]["linear_weighted_agreement"] >= cfg["per_reviewer_reference_ordinal_weighted_min"]
        calibration_gates[f"{role}_reference_boundary_raw"] = block["boundary"]["raw_agreement"] >= cfg["per_reviewer_reference_score2_boundary_raw_min"]
        calibration_gates[f"{role}_reference_boundary_ac1"] = block["boundary"]["gwet_ac1"] is not None and block["boundary"]["gwet_ac1"] >= cfg["per_reviewer_reference_score2_boundary_gwet_ac1_min"]
        calibration_gates[f"{role}_reference_boundary_positive"] = block["boundary"]["positive_agreement"] is not None and block["boundary"]["positive_agreement"] >= cfg["per_reviewer_reference_score2_boundary_positive_min"]
        calibration_gates[f"{role}_reference_boundary_negative"] = block["boundary"]["negative_agreement"] is not None and block["boundary"]["negative_agreement"] >= cfg["per_reviewer_reference_score2_boundary_negative_min"]
    target_gates = {}
    for field, block in target_ab["components"].items():
        target_gates[f"{field}_raw"] = block["raw_agreement"] >= target_cfg["action_component_raw_min"]
        target_gates[f"{field}_ac1"] = block["gwet_ac1"] is not None and block["gwet_ac1"] >= target_cfg["action_component_gwet_ac1_min"]
    target_gates.update(
        {
            "ordinal_exact": target_ab["ordinal"]["exact_raw_agreement"] >= target_cfg["action_ordinal_exact_min"],
            "ordinal_weighted": target_ab["ordinal"]["linear_weighted_agreement"] >= target_cfg["action_ordinal_weighted_min"],
            "ordinal_ac2": target_ab["ordinal"]["gwet_ac2"] is not None and target_ab["ordinal"]["gwet_ac2"] >= target_cfg["action_ordinal_gwet_ac2_min"],
            "two_level": target_ab["ordinal"]["two_level_disagreement_rate"] <= target_cfg["action_two_level_disagreement_rate_max"],
            "boundary_raw": target_ab["boundary"]["raw_agreement"] >= target_cfg["action_score2_boundary_raw_min"],
            "boundary_ac1": target_ab["boundary"]["gwet_ac1"] is not None and target_ab["boundary"]["gwet_ac1"] >= target_cfg["action_score2_boundary_gwet_ac1_min"],
            "boundary_positive": target_ab["boundary"]["positive_agreement"] is not None and target_ab["boundary"]["positive_agreement"] >= target_cfg["action_score2_boundary_positive_min"],
            "boundary_negative": target_ab["boundary"]["negative_agreement"] is not None and target_ab["boundary"]["negative_agreement"] >= target_cfg["action_score2_boundary_negative_min"],
        }
    )
    duplicate_gates = {
        role: block["denominator"] == duplicate_cfg["action_denominator"]
        and block["full_vector_consistent_count"] >= duplicate_cfg["action_full_vector_consistency_min_count"]
        for role, block in duplicate.items()
    }
    disagreements = [
        row_id
        for row_id in target
        if any(reviewer_a[row_id][field] != reviewer_b[row_id][field] for field in ACTION_FIELDS)
    ]
    for block in (calibration_ab, target_ab, *reference_metrics.values()):
        block.pop("ordinal_pairs", None)
        block.pop("boundary_pairs", None)
    return {
        "calibration_rows": len(calibration),
        "target_rows": len(target),
        "calibration_ab": calibration_ab,
        "calibration_reference": reference_metrics,
        "calibration_bootstrap": uncertainty,
        "target_ab": target_ab,
        "duplicate_consistency": duplicate,
        "calibration_gates": calibration_gates,
        "target_gates": target_gates,
        "duplicate_gates": duplicate_gates,
        "calibration_pass": all(calibration_gates.values()),
        "target_pass": all(target_gates.values()),
        "duplicate_pass": all(duplicate_gates.values()),
        "target_disagreement_ids": disagreements,
        "target_disagreement_rows": len(disagreements),
        "target_score_prevalence_used_as_gate": False,
    }


def build_c_packet(
    construct: str,
    disagreement_ids: list[str],
    execution: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = review_spec(construct, "a", execution, protocol)
    packet_by_id = {str(row[spec["id_field"]]): row for row in read_jsonl(spec["packet"])}
    packet = [packet_by_id[row_id] for row_id in sorted(disagreement_ids)]
    template = [
        {
            spec["id_field"]: row[spec["id_field"]],
            **{field: "" for field in spec["fields"]},
            "short_reason": "",
        }
        for row in packet
    ]
    return packet, template


def preflight(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    for construct in ("safety", "actionability"):
        for role in ("a", "b"):
            validate_packet_template(review_spec(construct, role, execution, protocol), manifest)
    result = {
        "status": "PASS_CSPR_P2S_REVIEW_PREFLIGHT",
        "action_rows_per_reviewer": manifest["action_packet_rows_per_reviewer"],
        "safety_rows_per_reviewer": manifest["safety_packet_rows_per_reviewer"],
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def analyze_ab(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    reviews, provenances = load_ab(execution, protocol, manifest)
    safety_key = read_jsonl(resolve_repo_path(protocol["packet_outputs"]["safety_key"]))
    action_key = read_jsonl(resolve_repo_path(protocol["packet_outputs"]["action_key"]))
    safety = safety_analysis(reviews["safety"]["a"], reviews["safety"]["b"], safety_key, protocol)
    action = action_analysis(reviews["actionability"]["a"], reviews["actionability"]["b"], action_key, protocol)
    calibration_pass = safety["calibration_pass"] and action["calibration_pass"]
    target_pass = safety["target_pass"] and action["target_pass"]
    duplicate_pass = safety["duplicate_pass"] and action["duplicate_pass"]
    overall = calibration_pass and target_pass and duplicate_pass
    if not calibration_pass or not duplicate_pass:
        route = "STOP_CSPR_P2S_CALIBRATION_UNRELIABLE"
    elif not target_pass:
        route = "STOP_CSPR_P2S_TARGET_UNRELIABLE"
    else:
        route = "AUTHORIZE_CSPR_P2S_REVIEWER_C_ONLY"
    c_artifacts = []
    if overall:
        for construct, block in (("safety", safety), ("actionability", action)):
            packet, template = build_c_packet(
                construct, block["target_disagreement_ids"], execution, protocol
            )
            files = execution["review_files"][construct]
            packet_path = resolve_repo_path(files["reviewer_c_packet"])
            template_path = resolve_repo_path(files["reviewer_c_template"])
            write_jsonl_frozen(packet_path, packet)
            write_jsonl_frozen(template_path, template)
            c_artifacts.extend(
                [
                    {"path": relative(packet_path), "rows": len(packet), "sha256": sha256(packet_path)},
                    {"path": relative(template_path), "rows": len(template), "sha256": sha256(template_path)},
                ]
            )
    report = {
        "name": "step3_cspr_p2s_ab_analysis_v1",
        "route": route,
        "overall_pass": overall,
        "calibration_pass": calibration_pass,
        "target_pass": target_pass,
        "duplicate_pass": duplicate_pass,
        "safety": safety,
        "actionability": action,
        "reviewer_provenance": [
            {field: row[field] for field in ("construct", "reviewer_role", "agent_id", "session_id", "output_sha256")}
            for row in provenances
        ],
        "reviewer_c_artifacts": c_artifacts,
        "target_prevalence_used_as_reviewer_gate": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(execution["analysis_outputs"]["ab_analysis"]), report)
    markdown = f"""# CSPR P2S A/B Review Analysis

Route: `{route}`.

Calibration gates passed: `{calibration_pass}`. Target reliability gates passed: `{target_pass}`. Hidden duplicate gates passed: `{duplicate_pass}`. Safety has `{safety['target_disagreement_rows']}` target disagreements and actionability has `{action['target_disagreement_rows']}`. Target score prevalence was reported but not used as a reviewer gate.

Reviewer C is authorized only when every gate above passes and only for the frozen target disagreement packets. No pair was selected and no gradient or optimizer step was computed. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(execution["analysis_outputs"]["ab_report"]), markdown)
    print(json.dumps({"route": route, "safety_disagreements": safety["target_disagreement_rows"], "actionability_disagreements": action["target_disagreement_rows"]}, sort_keys=True))
    return report


def validate_c(
    construct: str,
    execution: dict[str, Any],
    protocol: dict[str, Any],
    ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files = execution["review_files"][construct]
    id_field = "opaque_safety_id" if construct == "safety" else "opaque_action_id"
    fields = ("clean_safe",) if construct == "safety" else ACTION_FIELDS
    packet_path = resolve_repo_path(files["reviewer_c_packet"])
    template_path = resolve_repo_path(files["reviewer_c_template"])
    output_path = resolve_repo_path(files["reviewer_c_labels"])
    provenance_path = resolve_repo_path(files["reviewer_c_provenance"])
    rubric_path = resolve_repo_path(protocol["frozen_inputs"][f"{'safety' if construct == 'safety' else 'action'}_rubric"]["path"])
    packet = read_jsonl(packet_path)
    template = read_jsonl(template_path)
    if [str(row[id_field]) for row in packet] != sorted(ids):
        raise ReviewValidationError(f"{construct} C packet is not the target disagreement set")
    rows = read_jsonl(output_path)
    if len(rows) != len(template):
        raise ReviewValidationError(f"{construct} C review incomplete")
    mapped = {}
    expected = {id_field, *fields, "short_reason"}
    for row, blank in zip(rows, template, strict=True):
        if set(row) != expected or row[id_field] != blank[id_field]:
            raise ReviewValidationError(f"{construct} C schema/order mismatch")
        if any(row[field] not in YES_NO for field in fields):
            raise ReviewValidationError(f"{construct} C invalid label")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise ReviewValidationError(f"{construct} C blank reason")
        mapped[str(row[id_field])] = row
    provenance = read_json(provenance_path)
    expected_prov = {
        "construct": construct,
        "reviewer_role": "c",
        "packet_path": relative(packet_path),
        "packet_sha256": sha256(packet_path),
        "template_path": relative(template_path),
        "template_sha256": sha256(template_path),
        "rubric_path": relative(rubric_path),
        "rubric_sha256": sha256(rubric_path),
        "output_path": relative(output_path),
        "output_sha256": sha256(output_path),
        "row_count": len(rows),
        "fresh_context": True,
        "forbidden_material_not_opened": True,
        "completion_status": "complete",
    }
    for field, value in expected_prov.items():
        if provenance.get(field) != value:
            raise ReviewValidationError(f"{construct} C provenance mismatch: {field}")
    for field in ("agent_id", "session_id", "runtime"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ReviewValidationError(f"{construct} C missing provenance {field}")
    return mapped, provenance


def final_target_labels(
    construct: str,
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    reviewer_c: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    id_field = "opaque_safety_id" if construct == "safety" else "opaque_action_id"
    fields = ("clean_safe",) if construct == "safety" else ACTION_FIELDS
    output = []
    for row_id in sorted(ids):
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
            row["derived_score"] = metrics.derived_score(row)
        output.append(row)
    return output


def qualify_pairs(
    safety_final: list[dict[str, Any]],
    action_final: list[dict[str, Any]],
    safety_key: list[dict[str, Any]],
    action_key: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    safety_labels = {row["opaque_safety_id"]: row["clean_safe"] for row in safety_final}
    action_labels = {row["opaque_action_id"]: row for row in action_final}
    safety_candidate = {
        row["source_authoring_id"]: row
        for row in safety_key
        if not row["is_hidden_duplicate"] and row["item_type"] == "target_candidate"
    }
    safety_rejected = {
        row["pair_id"]: row
        for row in safety_key
        if not row["is_hidden_duplicate"] and row["item_type"] == "target_rejected"
    }
    action_candidate = {
        row["source_authoring_id"]: row
        for row in action_key
        if not row["is_hidden_duplicate"] and row["item_type"] == "target_candidate"
    }
    by_pair: dict[str, list[str]] = defaultdict(list)
    for authoring_id, row in action_candidate.items():
        by_pair[str(row["pair_id"])].append(str(authoring_id))
    seed = int(protocol["packet_design"]["seed"])
    pool = []
    for pair_id in sorted(by_pair):
        rejected = safety_rejected[pair_id]
        candidate_rows = []
        for authoring_id in sorted(by_pair[pair_id]):
            safety = safety_candidate[authoring_id]
            action = action_candidate[authoring_id]
            action_label = action_labels[action["opaque_action_id"]]
            checks = {
                "candidate_clean_safe": safety_labels[safety["opaque_safety_id"]] == "yes",
                "candidate_safe_route": action_label["safe_route"] == "yes",
                "candidate_executable_step": action_label["executable_step"] == "yes",
                "candidate_prompt_binding": action_label["prompt_binding"] == "yes",
                "rejected_clean_safe": safety_labels[rejected["opaque_safety_id"]] == "no",
            }
            candidate_rows.append(
                {
                    "source_authoring_id": authoring_id,
                    "author_role": action["author_role"],
                    "checks": checks,
                    "qualified": all(checks.values()),
                }
            )
        qualified_candidates = [row for row in candidate_rows if row["qualified"]]
        selected = (
            min(
                qualified_candidates,
                key=lambda row: packet_builder.stable_rank(
                    seed, "qualified-candidate-selection", row["source_authoring_id"]
                ),
            )
            if qualified_candidates
            else None
        )
        reference = action_candidate[by_pair[pair_id][0]]
        pool.append(
            {
                "pair_id": pair_id,
                "prospective_split": reference["prospective_split"],
                "assigned_category": reference["assigned_category"],
                "candidate_results": candidate_rows,
                "selected_authoring_id": selected["source_authoring_id"] if selected else None,
                "selected_author_role": selected["author_role"] if selected else None,
                "qualified": selected is not None,
            }
        )
    qualified = [row for row in pool if row["qualified"]]
    split_counts = Counter(row["prospective_split"] for row in qualified)
    category_counts = Counter(row["assigned_category"] for row in qualified)
    cfg = protocol["fixed_pool_yield_gates"]
    categories_with_two = sum(value >= 2 for value in category_counts.values())
    gates = {
        "train": split_counts["train_candidate"] >= cfg["minimum_train_candidate_pairs"],
        "calibration": split_counts["calibration"] >= cfg["minimum_calibration_pairs"],
        "mechanism_dev": split_counts["mechanism_dev"] >= cfg["minimum_mechanism_dev_pairs"],
        "total": len(qualified) >= cfg["minimum_total_pairs"],
        "category": categories_with_two >= cfg["minimum_categories_with_at_least_two_pairs"],
    }
    return pool, {
        "qualified_pairs": len(qualified),
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "categories_with_at_least_two_pairs": categories_with_two,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def finalize(path: Path) -> dict[str, Any]:
    execution, protocol, manifest = context(path)
    ab = read_json(resolve_repo_path(execution["analysis_outputs"]["ab_analysis"]))
    if ab.get("route") != "AUTHORIZE_CSPR_P2S_REVIEWER_C_ONLY" or not ab.get("overall_pass"):
        raise ReviewValidationError("A/B analysis did not authorize P2S finalization")
    reviews, ab_provenances = load_ab(execution, protocol, manifest)
    safety_key = read_jsonl(resolve_repo_path(protocol["packet_outputs"]["safety_key"]))
    action_key = read_jsonl(resolve_repo_path(protocol["packet_outputs"]["action_key"]))
    safety = safety_analysis(reviews["safety"]["a"], reviews["safety"]["b"], safety_key, protocol)
    action = action_analysis(reviews["actionability"]["a"], reviews["actionability"]["b"], action_key, protocol)
    if safety != ab["safety"] or action != ab["actionability"]:
        raise ReviewValidationError("A/B labels changed after analysis freeze")
    c_reviews = {}
    c_provenances = []
    for construct, block in (("safety", safety), ("actionability", action)):
        ids = block["target_disagreement_ids"]
        if ids:
            c_reviews[construct], provenance = validate_c(construct, execution, protocol, ids)
            c_provenances.append(provenance)
        else:
            c_reviews[construct] = {}
    all_provenances = ab_provenances + c_provenances
    ensure_reviewer_isolation(all_provenances, execution["forbidden_prior_agent_ids"])
    safety_ids = sorted(
        str(row["opaque_safety_id"])
        for row in safety_key
        if not row["is_hidden_duplicate"] and row["item_type"].startswith("target_")
    )
    action_ids = sorted(
        str(row["opaque_action_id"])
        for row in action_key
        if not row["is_hidden_duplicate"] and row["item_type"] == "target_candidate"
    )
    safety_final = final_target_labels(
        "safety", reviews["safety"]["a"], reviews["safety"]["b"], c_reviews["safety"], safety_ids
    )
    action_final = final_target_labels(
        "actionability", reviews["actionability"]["a"], reviews["actionability"]["b"], c_reviews["actionability"], action_ids
    )
    pool, yield_analysis = qualify_pairs(safety_final, action_final, safety_key, action_key, protocol)
    route = (
        "AUTHORIZE_CSPR_P3_ZERO_UPDATE_MECHANISM_PROTOCOL_ONLY"
        if yield_analysis["overall_pass"]
        else "STOP_CSPR_P2S_PAIR_YIELD"
    )
    outputs = execution["final_outputs"]
    safety_path = resolve_repo_path(outputs["safety_final_labels"])
    action_path = resolve_repo_path(outputs["action_final_labels"])
    pool_path = resolve_repo_path(outputs["qualified_pair_pool"])
    write_jsonl_frozen(safety_path, safety_final)
    write_jsonl_frozen(action_path, action_final)
    write_jsonl_frozen(pool_path, pool)
    analysis = {
        "name": "step3_cspr_p2s_final_analysis_v1",
        "route": route,
        "safety_final": {"path": relative(safety_path), "rows": len(safety_final), "sha256": sha256(safety_path)},
        "action_final": {"path": relative(action_path), "rows": len(action_final), "sha256": sha256(action_path)},
        "qualified_pool": {"path": relative(pool_path), "rows": len(pool), "sha256": sha256(pool_path)},
        "yield": yield_analysis,
        "target_prevalence_used_as_reviewer_gate": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["final_analysis"]), analysis)
    decision = {
        "name": "step3_cspr_p2s_route_decision_v1",
        "route": route,
        "overall_pass": yield_analysis["overall_pass"],
        "qualified_pairs": yield_analysis["qualified_pairs"],
        "next_authorized_action": "freeze_separate_CSPR_P3_zero_update_mechanism_protocol" if yield_analysis["overall_pass"] else "stop_without_training_or_replacement",
        "optimizer_steps": 0,
        "training_authorized": False,
        "model_gradient_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["route_decision"]), decision)
    markdown = f"""# CSPR P2S Final Decision

Route: `{route}`.

After calibration-qualified A/B review and target-only C adjudication, `{yield_analysis['qualified_pairs']}` of 120 prompts retained at least one qualified authored candidate. Split counts were `{json.dumps(yield_analysis['split_counts'], sort_keys=True)}` and `{yield_analysis['categories_with_at_least_two_pairs']}` categories retained at least two pairs. Fixed-pool gates passed: `{yield_analysis['overall_pass']}`.

This can authorize only a separately prospective zero-update mechanism protocol. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(outputs["final_report"]), markdown)
    print(json.dumps({"route": route, "qualified_pairs": yield_analysis["qualified_pairs"]}, sort_keys=True))
    return decision


def record_inconclusive(
    path: Path, construct: str | None, role: str | None, agent_id: str | None, reason: str | None
) -> dict[str, Any]:
    execution, _, _ = context(path)
    if not construct or not role or not agent_id or not reason:
        raise ReviewValidationError("inconclusive record requires construct, role, agent ID, and reason")
    result = {
        "name": "step3_cspr_p2s_review_inconclusive_v1",
        "route": "STOP_CSPR_P2S_REVIEW_INCONCLUSIVE",
        "construct": construct,
        "reviewer_role": role,
        "agent_id": agent_id,
        "reason": reason,
        "heuristic_substitution_used": False,
        "replacement_reviewer_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(execution["analysis_outputs"]["inconclusive_incident"]), result)
    print(json.dumps({"route": result["route"]}, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.execution_protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        elif args.command == "analyze-ab":
            analyze_ab(path)
        elif args.command == "finalize":
            finalize(path)
        else:
            record_inconclusive(path, args.construct, args.reviewer_role, args.agent_id, args.reason)
    except (ReviewValidationError, FileExistsError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2S_REVIEW_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
