"""Freeze Step3C-T4 arm-blind packets without assigning outcome labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import generate_step3ct_t4 as generation
import step3ct_t4_common as common


LABEL_FIELDS = {
    "safety": [
        "opaque_response_id", "primary_label", "actionable_harmful_content", "explicit_refusal",
        "prompt_relevant_safe_redirect", "generic_refusal_only", "mixed_response", "reason_code", "short_reason",
    ],
    "quality": ["opaque_quality_id", "detail_retention", "safe_actionability", "naturalness", "short_reason"],
    "utility": [
        "opaque_response_id", "task_success", "unnecessary_refusal", "invalid_or_irrelevant",
        "format_failure", "material_degradation", "reason_code", "short_reason",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    return parser.parse_args()


def generation_sources(registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safety = []
    utility = []
    for arm in registry["arms"]:
        for checkpoint in registry["allowed_checkpoints"]["safety"]:
            path, manifest_path = generation.output_paths(registry, arm, checkpoint, "safety")
            if not path.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"missing frozen generation: {path}")
            manifest = common.read_json(manifest_path)
            rows = common.read_jsonl(path)
            if len(rows) != 100 or manifest["output_sha256"] != common.sha256_file(path):
                raise ValueError(f"generation manifest mismatch: {path}")
            safety.extend(rows)
        path, manifest_path = generation.output_paths(registry, arm, 64, "utility")
        if not path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"missing frozen generation: {path}")
        manifest = common.read_json(manifest_path)
        rows = common.read_jsonl(path)
        if len(rows) != 100 or manifest["output_sha256"] != common.sha256_file(path):
            raise ValueError(f"generation manifest mismatch: {path}")
        utility.extend(rows)
    return safety, utility


def packet_sources(domain: str, safety: list[dict[str, Any]], utility: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if domain == "safety":
        selected = [row for row in safety if int(row["checkpoint_step"]) in {32, 64}]
    elif domain == "quality":
        selected = [row for row in safety if int(row["checkpoint_step"]) == 64]
    elif domain == "utility":
        selected = list(utility)
    else:
        raise ValueError(domain)
    result = []
    for row in selected:
        source_key = "|".join([str(row["arm"]), str(row["checkpoint_step"]), str(row["sample_id"]), domain])
        result.append({**row, "source_key": source_key})
    return result


def visible_row(domain: str, opaque: str, source: dict[str, Any]) -> dict[str, Any]:
    id_field = "opaque_quality_id" if domain == "quality" else "opaque_response_id"
    result = {id_field: opaque, "prompt": source["prompt"], "response": source["response"]}
    if domain == "utility":
        result["reference_response"] = source["reference_response"]
    return result


def build_packet(domain: str, sources: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    spec = registry["packets"][domain]
    if len(sources) != int(spec["primary_rows"]):
        raise ValueError(f"{domain} primary source count mismatch")
    packet_seed = int(registry["packets"]["packet_seed"])
    duplicate_seed = int(registry["packets"]["duplicate_seed"])
    count = common.duplicate_count(
        len(sources),
        float(registry["packets"]["duplicate_fraction"]),
        int(registry["packets"]["duplicate_min"]),
        int(registry["packets"]["duplicate_max"]),
    )
    if count != int(spec["hidden_duplicates"]):
        raise ValueError(f"{domain} duplicate count mismatch")
    group_fields = ["arm", "checkpoint_step"] if domain == "safety" else ["arm"]
    duplicate_sources = common.balanced_duplicate_sources(sources, count, duplicate_seed, group_fields)

    packet_rows = []
    key_rows = []
    primary_by_source: dict[str, str] = {}
    for source in sources:
        opaque = common.opaque_id(domain, packet_seed, source["source_key"], "primary")
        primary_by_source[source["source_key"]] = opaque
        packet_rows.append(visible_row(domain, opaque, source))
        key_rows.append({
            "opaque_id": opaque,
            "domain": domain,
            "arm": source["arm"],
            "checkpoint_step": source["checkpoint_step"],
            "sample_id": source["sample_id"],
            "generation_id": source["generation_id"],
            "is_hidden_duplicate": "false",
            "duplicate_of_opaque_id": "",
            "prompt_sha256": source["prompt_sha256"],
            "response_sha256": source["response_sha256"],
        })

    duplicate_rows = []
    for source in duplicate_sources:
        opaque = common.opaque_id(domain, packet_seed, source["source_key"], "duplicate")
        primary = primary_by_source[source["source_key"]]
        packet_rows.append(visible_row(domain, opaque, source))
        key_rows.append({
            "opaque_id": opaque,
            "domain": domain,
            "arm": source["arm"],
            "checkpoint_step": source["checkpoint_step"],
            "sample_id": source["sample_id"],
            "generation_id": source["generation_id"],
            "is_hidden_duplicate": "true",
            "duplicate_of_opaque_id": primary,
            "prompt_sha256": source["prompt_sha256"],
            "response_sha256": source["response_sha256"],
        })
        duplicate_rows.append({
            "domain": domain,
            "primary_opaque_id": primary,
            "duplicate_opaque_id": opaque,
            "arm": source["arm"],
            "checkpoint_step": source["checkpoint_step"],
            "sample_id": source["sample_id"],
        })

    packet_rows = common.shuffled(packet_rows, packet_seed + {"safety": 1, "quality": 2, "utility": 3}[domain])
    id_field = "opaque_quality_id" if domain == "quality" else "opaque_response_id"
    packet_by_id = {row[id_field]: row for row in packet_rows}
    if len(packet_by_id) != len(packet_rows):
        raise ValueError(f"duplicate opaque IDs in {domain}")
    exact = all(
        {k: v for k, v in packet_by_id[row["primary_opaque_id"]].items() if k != id_field}
        == {k: v for k, v in packet_by_id[row["duplicate_opaque_id"]].items() if k != id_field}
        for row in duplicate_rows
    )
    if not exact:
        raise ValueError(f"hidden duplicate visible content mismatch in {domain}")
    template_rows = [
        {field: (row[id_field] if field == id_field else "") for field in LABEL_FIELDS[domain]}
        for row in packet_rows
    ]
    return {
        "packet_rows": packet_rows,
        "key_rows": key_rows,
        "duplicate_rows": duplicate_rows,
        "template_rows": template_rows,
        "id_field": id_field,
        "primary_rows": len(sources),
        "duplicate_rows_count": len(duplicate_rows),
        "cell_counts": dict(Counter(
            f"{row['arm']}|{row['checkpoint_step']}" for row in key_rows if row["is_hidden_duplicate"] == "false"
        )),
        "duplicate_cell_counts": dict(Counter(
            f"{row['arm']}|{row['checkpoint_step']}" for row in key_rows if row["is_hidden_duplicate"] == "true"
        )),
        "duplicate_exact_match": exact,
    }


def freeze_domain(domain: str, built: dict[str, Any], registry: dict[str, Any], root: Path) -> dict[str, Any]:
    domain_root = root / "packets" / domain
    packet_path = domain_root / f"step3ct_t4_{domain}_packet.jsonl"
    key_path = domain_root / f"step3ct_t4_{domain}_hidden_key.csv"
    duplicate_path = domain_root / f"step3ct_t4_{domain}_duplicate_map.csv"
    template_a = domain_root / f"step3ct_t4_{domain}_reviewer_a_template.jsonl"
    template_b = domain_root / f"step3ct_t4_{domain}_reviewer_b_template.jsonl"
    common.write_jsonl(packet_path, built["packet_rows"])
    common.write_csv(key_path, built["key_rows"])
    common.write_csv(duplicate_path, built["duplicate_rows"])
    common.write_jsonl(template_a, built["template_rows"])
    common.write_jsonl(template_b, built["template_rows"])
    manifest = {
        "name": f"step3ct_t4_{domain}_packet_manifest_v1",
        "domain": domain,
        "rubric": registry["packets"][domain]["rubric"],
        "rubric_sha256": registry["packets"][domain]["rubric_sha256"],
        "packet_rows": len(built["packet_rows"]),
        "primary_rows": built["primary_rows"],
        "hidden_duplicates": built["duplicate_rows_count"],
        "cell_counts": built["cell_counts"],
        "duplicate_cell_counts": built["duplicate_cell_counts"],
        "duplicate_exact_match": built["duplicate_exact_match"],
        "visible_fields": list(built["packet_rows"][0]),
        "packet_sha256": common.sha256_file(packet_path),
        "hidden_key_sha256": common.sha256_file(key_path),
        "duplicate_map_sha256": common.sha256_file(duplicate_path),
        "reviewer_a_template_sha256": common.sha256_file(template_a),
        "reviewer_b_template_sha256": common.sha256_file(template_b),
        "outcome_labels_present": False,
        "arm_or_checkpoint_visible": False,
    }
    manifest_path = domain_root / f"step3ct_t4_{domain}_packet_manifest.json"
    common.write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": manifest_path.relative_to(common.REPO).as_posix(), "manifest_sha256": common.sha256_file(manifest_path)}


def main() -> None:
    args = parse_args()
    registry_path = (common.REPO / args.registry).resolve()
    registry = common.read_json(registry_path)
    if common.sha256_file(common.REPO / registry["t3_route"]) != registry["t3_route_sha256"]:
        raise ValueError("T3 route hash mismatch")
    safety, utility = generation_sources(registry)
    expected_safety_ids = [row["sample_id"] for row in safety[:100]]
    if any([row["sample_id"] for row in safety[i:i + 100]] != expected_safety_ids for i in range(0, len(safety), 100)):
        raise ValueError("safety prompt order differs across generation cells")
    expected_utility_ids = [row["sample_id"] for row in utility[:100]]
    if any([row["sample_id"] for row in utility[i:i + 100]] != expected_utility_ids for i in range(0, len(utility), 100)):
        raise ValueError("utility prompt order differs across generation cells")

    root = common.REPO / registry["packet_root"]
    domain_manifests = {}
    for domain in ["safety", "quality", "utility"]:
        sources = packet_sources(domain, safety, utility)
        built = build_packet(domain, sources, registry)
        domain_manifests[domain] = freeze_domain(domain, built, registry, root)

    source_manifests = []
    generation_root = common.REPO / registry["generation_root"]
    for path in sorted(generation_root.glob("*/*_manifest.json")):
        manifest = common.read_json(path)
        source_manifests.append({
            "arm": manifest["arm"],
            "checkpoint_step": manifest["checkpoint_step"],
            "eval_set": manifest["eval_set"],
            "rows": manifest["rows"],
            "output_path": manifest["output_path"],
            "output_sha256": manifest["output_sha256"],
            "manifest_path": path.relative_to(common.REPO).as_posix(),
            "manifest_sha256": common.sha256_file(path),
            "empty_response_count": manifest["empty_response_count"],
            "thinking_blocks_removed": manifest["thinking_blocks_removed"],
        })
    if len(source_manifests) != 12:
        raise ValueError("expected exactly 12 generation manifests")
    freeze = {
        "name": "step3ct_t4_generation_and_packet_freeze_v1",
        "registry_path": args.registry,
        "registry_sha256": common.sha256_file(registry_path),
        "generation_cells": source_manifests,
        "generation_cell_count": len(source_manifests),
        "safety_generation_rows": len(safety),
        "utility_generation_rows": len(utility),
        "prompt_alignment": {"safety": True, "utility": True},
        "packets": domain_manifests,
        "outcome_labels_present": False,
        "candidate_outputs_generated": True,
        "core100_opened": True,
        "utility100_opened": True,
        "heldout_opened": False,
        "next_authorized_package": "Step3C_T5_independent_blind_review_after_T4_commit_push",
    }
    common.write_json(root / "step3ct_t4_generation_and_packet_freeze_manifest.json", freeze)
    print({domain: manifest["packet_rows"] for domain, manifest in domain_manifests.items()})


if __name__ == "__main__":
    main()
