"""Finalize OACPO D1 labels and compute the frozen D2 triplet-yield route."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
YES_NO = {"yes", "no"}


class FinalizeError(ValueError):
    """Raised when a frozen adjudication input is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "finalize", "verify"])
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


def verify_frozen_inputs(protocol: dict[str, Any]) -> None:
    for label, artifact in protocol["frozen_inputs"].items():
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise FinalizeError(f"frozen input mismatch: {label}")


def validate_c_review(
    construct: str, protocol: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files = protocol["reviewer_c_files"][construct]
    packet_path = resolve_repo_path(files["packet"])
    template_path = resolve_repo_path(files["template"])
    output_path = resolve_repo_path(files["labels"])
    provenance_path = resolve_repo_path(files["provenance"])
    packet = read_jsonl(packet_path)
    template = read_jsonl(template_path)
    output = read_jsonl(output_path)
    id_field = {
        "safety": "opaque_safety_id",
        "quality": "opaque_quality_id",
        "structure": "opaque_structure_id",
    }[construct]
    if len(packet) != len(template) or len(output) != len(template):
        raise FinalizeError(f"incomplete Reviewer C output: {construct}")
    if [row[id_field] for row in packet] != [row[id_field] for row in template]:
        raise FinalizeError(f"packet/template order mismatch: {construct}")
    if [row.get(id_field) for row in output] != [row[id_field] for row in template]:
        raise FinalizeError(f"output order mismatch: {construct}")

    mapped = {}
    for index, (visible, blank, row) in enumerate(zip(packet, template, output, strict=True), start=1):
        if set(row) != set(blank):
            raise FinalizeError(f"Reviewer C schema mismatch: {construct} row {index}")
        if not isinstance(row["short_reason"], str) or not row["short_reason"].strip():
            raise FinalizeError(f"blank Reviewer C reason: {construct} row {index}")
        disputed = set(visible["disputed_fields"])
        if construct == "safety":
            if row["clean_safe"] not in YES_NO:
                raise FinalizeError("invalid C safety label")
        elif construct == "structure":
            if row["mechanical_template"] not in YES_NO:
                raise FinalizeError("invalid C structure label")
        else:
            for field in ("detail_retention", "safe_actionability", "naturalness"):
                value = row[field]
                if field not in disputed:
                    if value is not None:
                        raise FinalizeError(f"C changed agreed quality field: {field}")
                elif field in ("detail_retention", "safe_actionability"):
                    if value not in (0, 1, 2):
                        raise FinalizeError(f"invalid C quality ordinal: {field}")
                elif value not in YES_NO:
                    raise FinalizeError("invalid C naturalness")
        mapped[str(row[id_field])] = row

    provenance = read_json(provenance_path)
    expected = {
        "construct": construct,
        "reviewer_role": "c",
        "packet_path": relative(packet_path),
        "packet_sha256": sha256(packet_path),
        "template_path": relative(template_path),
        "template_sha256": sha256(template_path),
        "output_path": relative(output_path),
        "output_sha256": sha256(output_path),
        "row_count": len(output),
        "fresh_context": True,
        "forbidden_material_not_opened": True,
        "completion_status": "complete",
    }
    for field, expected_value in expected.items():
        if provenance.get(field) != expected_value:
            raise FinalizeError(f"Reviewer C provenance mismatch: {construct}/{field}")
    if not isinstance(provenance.get("agent_id"), str) or not provenance["agent_id"]:
        raise FinalizeError("Reviewer C agent_id missing")
    return mapped, provenance


def adjudicate_field(
    row_id: str,
    field: str,
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    reviewer_c: dict[str, dict[str, Any]],
) -> tuple[Any, bool]:
    left = reviewer_a[row_id][field]
    right = reviewer_b[row_id][field]
    if left == right:
        return left, False
    if row_id not in reviewer_c or reviewer_c[row_id][field] is None:
        raise FinalizeError(f"missing Reviewer C decision: {row_id}/{field}")
    return reviewer_c[row_id][field], True


def final_labels_for_construct(
    construct: str,
    protocol: dict[str, Any],
    reviewer_c: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    files = protocol["ab_review_files"][construct]
    reviewer_a_rows = read_jsonl(resolve_repo_path(files["reviewer_a_labels"]))
    reviewer_b_rows = read_jsonl(resolve_repo_path(files["reviewer_b_labels"]))
    id_field = {
        "safety": "opaque_safety_id",
        "quality": "opaque_quality_id",
        "structure": "opaque_structure_id",
    }[construct]
    fields = {
        "safety": ("clean_safe",),
        "quality": ("detail_retention", "safe_actionability", "naturalness"),
        "structure": ("mechanical_template",),
    }[construct]
    reviewer_a = {str(row[id_field]): row for row in reviewer_a_rows}
    reviewer_b = {str(row[id_field]): row for row in reviewer_b_rows}
    key_rows = read_jsonl(resolve_repo_path(files["hidden_key"]))
    primary_ids = sorted(str(row[id_field]) for row in key_rows if not row["is_hidden_duplicate"])
    final = []
    for row_id in primary_ids:
        row = {id_field: row_id}
        adjudicated = []
        for field in fields:
            value, used_c = adjudicate_field(row_id, field, reviewer_a, reviewer_b, reviewer_c)
            row[field] = value
            if used_c:
                adjudicated.append(field)
        row["adjudicated_fields"] = adjudicated
        final.append(row)
    return final


def qualify_triplets(
    protocol: dict[str, Any], final_labels: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = read_json(resolve_repo_path(protocol["frozen_inputs"]["base_protocol"]["path"]))
    specs = {
        "safety": ("opaque_safety_id", base["outputs"]["safety_key"]),
        "quality": ("opaque_quality_id", base["outputs"]["quality_key"]),
        "structure": ("opaque_structure_id", base["outputs"]["structure_key"]),
    }
    by_pair_role: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for construct, (id_field, key_path) in specs.items():
        labels = {row[id_field]: row for row in final_labels[construct]}
        joined = {}
        for key in read_jsonl(resolve_repo_path(key_path)):
            if key["is_hidden_duplicate"]:
                continue
            joined[(key["pair_id"], key["response_role"])] = {**key, **labels[key[id_field]]}
        by_pair_role[construct] = joined

    pair_ids = sorted(
        {pair_id for pair_id, role in by_pair_role["safety"] if role == "authored"}
    )
    ledger = []
    for pair_id in pair_ids:
        authored_safety = by_pair_role["safety"][(pair_id, "authored")]
        source_safe_safety = by_pair_role["safety"][(pair_id, "source_safe")]
        source_unsafe_safety = by_pair_role["safety"][(pair_id, "source_unsafe")]
        authored_quality = by_pair_role["quality"][(pair_id, "authored")]
        source_safe_quality = by_pair_role["quality"][(pair_id, "source_safe")]
        authored_structure = by_pair_role["structure"][(pair_id, "authored")]
        source_safe_structure = by_pair_role["structure"][(pair_id, "source_safe")]

        authored_positive = (
            authored_safety["clean_safe"] == "yes"
            and authored_quality["detail_retention"] == 2
            and authored_quality["safe_actionability"] == 2
            and authored_quality["naturalness"] == "yes"
            and authored_structure["mechanical_template"] == "no"
        )
        safe_quality_negative = (
            source_safe_safety["clean_safe"] == "yes"
            and (
                source_safe_quality["detail_retention"] < 2
                or source_safe_quality["safe_actionability"] < 2
                or source_safe_structure["mechanical_template"] == "yes"
            )
        )
        unsafe_negative = source_unsafe_safety["clean_safe"] == "no"
        qualified = authored_positive and safe_quality_negative and unsafe_negative
        ledger.append(
            {
                "pair_id": pair_id,
                "prospective_split": authored_safety["prospective_split"],
                "assigned_category": authored_safety["assigned_category"],
                "authored_positive": authored_positive,
                "safe_quality_negative": safe_quality_negative,
                "unsafe_negative": unsafe_negative,
                "qualified_triplet": qualified,
                "authored_response_normalized_sha256": authored_safety[
                    "response_normalized_sha256"
                ],
                "source_safe_response_normalized_sha256": source_safe_safety[
                    "response_normalized_sha256"
                ],
                "source_unsafe_response_normalized_sha256": source_unsafe_safety[
                    "response_normalized_sha256"
                ],
            }
        )

    qualified = [row for row in ledger if row["qualified_triplet"]]
    split_counts = Counter(row["prospective_split"] for row in qualified)
    category_counts = Counter(row["assigned_category"] for row in qualified)
    gates = protocol["triplet_yield_gates"]
    gate_results = {
        "total": len(qualified) >= gates["minimum_total_triplets"],
        "train_candidate": split_counts["train_candidate"] >= gates["minimum_train_triplets"],
        "calibration": split_counts["calibration"] >= gates["minimum_calibration_triplets"],
        "mechanism_dev": split_counts["mechanism_dev"] >= gates["minimum_mechanism_dev_triplets"],
        "category_coverage": len(category_counts) >= gates["minimum_categories_with_triplets"],
    }
    summary = {
        "qualified_triplets": len(qualified),
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "component_pass_counts": {
            "authored_positive": sum(row["authored_positive"] for row in ledger),
            "safe_quality_negative": sum(row["safe_quality_negative"] for row in ledger),
            "unsafe_negative": sum(row["unsafe_negative"] for row in ledger),
        },
        "gates": gate_results,
        "pass": all(gate_results.values()),
    }
    return ledger, summary


def rendered_outputs(protocol: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    c_reviews = {}
    provenances = []
    for construct in ("safety", "quality", "structure"):
        c_reviews[construct], provenance = validate_c_review(construct, protocol)
        provenances.append(provenance)
    final_labels = {
        construct: final_labels_for_construct(construct, protocol, c_reviews[construct])
        for construct in ("safety", "quality", "structure")
    }
    ledger, summary = qualify_triplets(protocol, final_labels)
    route = (
        "AUTHORIZE_OACPO_P0_ZERO_UPDATE_PROTOCOL_ONLY"
        if summary["pass"]
        else "STOP_OACPO_D2_TRIPLET_YIELD"
    )
    result = {
        "name": "step3_oacpo_d2_triplet_qualification_v1",
        "route": route,
        "triplet_summary": summary,
        "final_label_counts": {name: len(rows) for name, rows in final_labels.items()},
        "reviewer_c_provenance": [
            {
                "construct": row["construct"],
                "agent_id": row["agent_id"],
                "output_sha256": row["output_sha256"],
            }
            for row in provenances
        ],
        "source_labels_used": False,
        "gradient_computed": False,
        "model_checkpoint_opened": False,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    report = f"""# OACPO D2 Triplet Qualification Report

Route: `{route}`.

Qualified triplets: `{summary['qualified_triplets']}`. Split counts: `{json.dumps(summary['split_counts'], sort_keys=True)}`. Categories represented: `{len(summary['category_counts'])}`. Gate pass: `{summary['pass']}`.

Component pass counts are recorded in the machine-readable result. Qualification uses only reliable A/B plus disagreement-only C labels; source labels are not used. This result authorizes at most a separately frozen zero-update OACPO P0 protocol. It does not compute a gradient, open a checkpoint, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
"""
    rendered = {
        "safety_final_labels": jsonl_text(final_labels["safety"]),
        "quality_final_labels": jsonl_text(final_labels["quality"]),
        "structure_final_labels": jsonl_text(final_labels["structure"]),
        "triplet_ledger": jsonl_text(ledger),
        "result": json_text(result),
        "report": report,
    }
    return rendered, result


def output_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {name: resolve_repo_path(path) for name, path in protocol["outputs"].items()}


def write_or_verify(protocol: dict[str, Any], rendered: dict[str, str], verify: bool) -> None:
    for name, path in output_paths(protocol).items():
        if verify:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered[name]:
                raise FinalizeError(f"final artifact mismatch: {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != rendered[name]:
            raise FinalizeError(f"refusing to overwrite different final artifact: {name}")
        path.write_text(rendered[name], encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    protocol = read_json(resolve_repo_path(args.protocol))
    if protocol["self_path"] != args.protocol.replace("\\", "/"):
        raise FinalizeError("protocol self path mismatch")
    verify_frozen_inputs(protocol)
    if args.command == "preflight":
        print("PASS_OACPO_D2_FINALIZE_PREFLIGHT")
        return
    rendered, result = rendered_outputs(protocol)
    write_or_verify(protocol, rendered, args.command == "verify")
    print(json.dumps({"route": result["route"], **result["triplet_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
