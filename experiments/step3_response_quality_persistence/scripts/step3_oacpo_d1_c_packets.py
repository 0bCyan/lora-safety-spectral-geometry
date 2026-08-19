"""Freeze and verify OACPO D1 disagreement-only Reviewer C packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "freeze", "verify"])
    parser.add_argument("--protocol", required=True)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(seed: int, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


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
            raise ValueError(f"frozen input mismatch: {label}")


def build_packets(protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    analysis = read_json(resolve_repo_path(protocol["frozen_inputs"]["ab_analysis"]["path"]))
    if analysis["route"] != "AUTHORIZE_OACPO_D1_REVIEWER_C_PROTOCOL_ONLY":
        raise ValueError("A/B route does not authorize Reviewer C")
    disagreements = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["disagreement_ledger"]["path"])
    )
    if len(disagreements) != int(analysis["primary_disagreement_count"]):
        raise ValueError("disagreement count mismatch")

    base = read_json(resolve_repo_path(protocol["frozen_inputs"]["base_protocol"]["path"]))
    specs = {
        "safety": ("opaque_safety_id", base["outputs"]["safety_a_packet"]),
        "quality": ("opaque_quality_id", base["outputs"]["quality_a_packet"]),
        "structure": ("opaque_structure_id", base["outputs"]["structure_a_packet"]),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    seed = int(protocol["packet_design"]["seed"])
    for construct, (id_field, packet_path) in specs.items():
        visible_by_id = {row[id_field]: row for row in read_jsonl(resolve_repo_path(packet_path))}
        rows = [row for row in disagreements if row["construct"] == construct]
        packet = []
        template = []
        for disagreement in rows:
            opaque_id = disagreement["opaque_id"]
            visible = visible_by_id.get(opaque_id)
            if visible is None:
                raise ValueError(f"missing visible disagreement row: {opaque_id}")
            disputed = disagreement["disputed_fields"]
            packet.append({**visible, "disputed_fields": disputed})
            if construct == "safety":
                template.append(
                    {id_field: opaque_id, "clean_safe": "", "short_reason": ""}
                )
            elif construct == "quality":
                template.append(
                    {
                        id_field: opaque_id,
                        "detail_retention": "" if "detail_retention" in disputed else None,
                        "safe_actionability": "" if "safe_actionability" in disputed else None,
                        "naturalness": "" if "naturalness" in disputed else None,
                        "short_reason": "",
                    }
                )
            else:
                template.append(
                    {id_field: opaque_id, "mechanical_template": "", "short_reason": ""}
                )
        order = sorted(
            range(len(packet)),
            key=lambda index: stable_rank(seed, construct, str(packet[index][id_field])),
        )
        result[f"{construct}_c_packet"] = [packet[index] for index in order]
        result[f"{construct}_c_template"] = [template[index] for index in order]
    return result


def validate_packets(packets: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]) -> None:
    expected_counts = protocol["packet_design"]["expected_rows"]
    for construct, expected in expected_counts.items():
        packet = packets[f"{construct}_c_packet"]
        template = packets[f"{construct}_c_template"]
        if len(packet) != expected or len(template) != expected:
            raise ValueError(f"wrong Reviewer C row count: {construct}")
        id_field = {
            "safety": "opaque_safety_id",
            "quality": "opaque_quality_id",
            "structure": "opaque_structure_id",
        }[construct]
        if [row[id_field] for row in packet] != [row[id_field] for row in template]:
            raise ValueError(f"Reviewer C packet/template order mismatch: {construct}")
        forbidden = {"pair_id", "response_role", "prospective_split", "assigned_category"}
        if any(forbidden.intersection(row) for row in packet):
            raise ValueError(f"private field leaked into Reviewer C packet: {construct}")
        if any(set(row["disputed_fields"]) - set(protocol["allowed_fields"][construct]) for row in packet):
            raise ValueError(f"invalid disputed field: {construct}")
        if construct == "quality":
            for packet_row, template_row in zip(packet, template, strict=True):
                for field in protocol["allowed_fields"]["quality"]:
                    expected_value = "" if field in packet_row["disputed_fields"] else None
                    if template_row[field] != expected_value:
                        raise ValueError("quality C template mutability mismatch")


def rendered_outputs(protocol: dict[str, Any]) -> dict[str, str]:
    packets = build_packets(protocol)
    validate_packets(packets, protocol)
    rendered = {name: jsonl_text(rows) for name, rows in packets.items()}
    manifest = {
        "name": "step3_oacpo_d1_reviewer_c_packet_manifest_v1",
        "protocol": protocol["self_path"],
        "protocol_sha256": sha256(resolve_repo_path(protocol["self_path"])),
        "route": "OACPO_D1_REVIEWER_C_PACKETS_FROZEN",
        "row_counts": {name: len(rows) for name, rows in packets.items()},
        "artifact_sha256": {name: sha256_text(text) for name, text in rendered.items()},
        "a_b_labels_exposed": False,
        "source_roles_exposed": False,
        "triplet_qualification_run": False,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    rendered["manifest"] = json_text(manifest)
    return rendered


def output_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {name: resolve_repo_path(path) for name, path in protocol["outputs"].items()}


def write_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    for name, path in output_paths(protocol).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != rendered[name]:
            raise ValueError(f"refusing to overwrite different C artifact: {name}")
        path.write_text(rendered[name], encoding="utf-8", newline="\n")


def verify_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    for name, path in output_paths(protocol).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered[name]:
            raise ValueError(f"Reviewer C artifact mismatch: {name}")


def main() -> None:
    args = parse_args()
    protocol = read_json(resolve_repo_path(args.protocol))
    if protocol["self_path"] != args.protocol.replace("\\", "/"):
        raise ValueError("protocol self path mismatch")
    verify_frozen_inputs(protocol)
    rendered = rendered_outputs(protocol)
    if args.command == "preflight":
        print("PASS_OACPO_D1_REVIEWER_C_PREFLIGHT")
    elif args.command == "freeze":
        write_outputs(protocol, rendered)
        print("OACPO_D1_REVIEWER_C_PACKETS_FROZEN")
    else:
        verify_outputs(protocol, rendered)
        print("PASS_OACPO_D1_REVIEWER_C_PACKET_VERIFY")


if __name__ == "__main__":
    main()
