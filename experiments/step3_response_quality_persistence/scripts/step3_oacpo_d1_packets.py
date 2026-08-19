"""Freeze and verify decomposed blinded OACPO D1 review packets."""

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


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


def load_triplet_sources(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = protocol["frozen_inputs"]
    selected = read_jsonl(resolve_repo_path(inputs["selected_source_ledger"]["path"]))
    author_map = read_jsonl(resolve_repo_path(inputs["author_private_map"]["path"]))
    authored = read_jsonl(resolve_repo_path(inputs["frozen_author_output"]["path"]))
    source_packet = read_jsonl(resolve_repo_path(inputs["p2_safety_packet"]["path"]))
    source_key = read_jsonl(resolve_repo_path(inputs["p2_safety_hidden_key"]["path"]))

    author_by_id = {row["author_id"]: row["authored_response"] for row in authored}
    author_id_by_pair = {row["pair_id"]: row["author_id"] for row in author_map}
    source_visible = {row["opaque_safety_id"]: row for row in source_packet}
    source_by_pair_role: dict[tuple[str, str], dict[str, str]] = {}
    for key in source_key:
        if key["is_hidden_duplicate"]:
            continue
        visible = source_visible.get(key["opaque_safety_id"])
        if visible is None:
            raise ValueError("missing source packet row")
        source_by_pair_role[(key["pair_id"], key["response_role"])] = visible

    triplets: list[dict[str, Any]] = []
    for row in selected:
        pair_id = row["pair_id"]
        author_id = author_id_by_pair.get(pair_id)
        safe = source_by_pair_role.get((pair_id, "chosen"))
        unsafe = source_by_pair_role.get((pair_id, "rejected"))
        if author_id is None or author_id not in author_by_id or safe is None or unsafe is None:
            raise ValueError(f"incomplete triplet source: {pair_id}")
        if normalize_text(safe["prompt"]) != normalize_text(unsafe["prompt"]):
            raise ValueError(f"source prompt mismatch: {pair_id}")
        if sha256_text(normalize_text(safe["prompt"])) != row["prompt_normalized_sha256"]:
            raise ValueError(f"source prompt hash mismatch: {pair_id}")
        if sha256_text(normalize_text(safe["response"])) != row["chosen_response_normalized_sha256"]:
            raise ValueError(f"source safe hash mismatch: {pair_id}")
        if sha256_text(normalize_text(unsafe["response"])) != row["rejected_response_normalized_sha256"]:
            raise ValueError(f"source unsafe hash mismatch: {pair_id}")
        triplets.append(
            {
                "pair_id": pair_id,
                "prompt": safe["prompt"],
                "prospective_split": row["prospective_split"],
                "assigned_category": row["assigned_category"],
                "authored": author_by_id[author_id],
                "source_safe": safe["response"],
                "source_unsafe": unsafe["response"],
            }
        )
    expected = int(protocol["packet_design"]["primary_triplets"])
    if len(triplets) != expected or len({row["pair_id"] for row in triplets}) != expected:
        raise ValueError("triplet source count mismatch")
    return sorted(triplets, key=lambda row: row["pair_id"])


def packet_order(rows: list[dict[str, Any]], seed: int, label: str, id_field: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_rank(seed, label, str(row[id_field])))


def add_duplicates(
    primary_packet: list[dict[str, Any]],
    primary_key: list[dict[str, Any]],
    roles: tuple[str, ...],
    per_role: int,
    seed: int,
    construct: str,
    id_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packet_by_id = {row[id_field]: row for row in primary_packet}
    duplicate_packet: list[dict[str, Any]] = []
    duplicate_key: list[dict[str, Any]] = []
    for role in roles:
        eligible = [row for row in primary_key if row["response_role"] == role]
        chosen = sorted(
            eligible,
            key=lambda row: stable_rank(seed, construct, "duplicate", role, row["pair_id"]),
        )[:per_role]
        if len(chosen) != per_role:
            raise ValueError(f"insufficient duplicate supply: {construct}/{role}")
        for key in chosen:
            primary = packet_by_id[key[id_field]]
            duplicate_id = f"oacpo_d1_{construct}_" + stable_rank(
                seed, construct, "duplicate-id", role, key["pair_id"]
            )[:20]
            duplicate_packet.append({**primary, id_field: duplicate_id})
            duplicate_key.append(
                {
                    **key,
                    id_field: duplicate_id,
                    "is_hidden_duplicate": True,
                    "duplicate_of": key[id_field],
                }
            )
    return primary_packet + duplicate_packet, primary_key + duplicate_key


def build_packets(triplets: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    design = protocol["packet_design"]
    seed = int(design["seed"])
    safety_packet: list[dict[str, Any]] = []
    safety_key: list[dict[str, Any]] = []
    quality_packet: list[dict[str, Any]] = []
    quality_key: list[dict[str, Any]] = []
    structure_packet: list[dict[str, Any]] = []
    structure_key: list[dict[str, Any]] = []

    for triplet in triplets:
        for role in ("authored", "source_safe", "source_unsafe"):
            response = triplet[role]
            safety_id = "oacpo_d1_safety_" + stable_rank(seed, "safety", triplet["pair_id"], role)[:20]
            safety_packet.append(
                {"opaque_safety_id": safety_id, "prompt": triplet["prompt"], "response": response}
            )
            safety_key.append(
                {
                    "opaque_safety_id": safety_id,
                    "pair_id": triplet["pair_id"],
                    "response_role": role,
                    "prospective_split": triplet["prospective_split"],
                    "assigned_category": triplet["assigned_category"],
                    "response_normalized_sha256": sha256_text(normalize_text(response)),
                    "is_hidden_duplicate": False,
                    "duplicate_of": None,
                }
            )

        for role in ("authored", "source_safe"):
            response = triplet[role]
            quality_id = "oacpo_d1_quality_" + stable_rank(seed, "quality", triplet["pair_id"], role)[:20]
            quality_packet.append(
                {"opaque_quality_id": quality_id, "prompt": triplet["prompt"], "response": response}
            )
            quality_key.append(
                {
                    "opaque_quality_id": quality_id,
                    "pair_id": triplet["pair_id"],
                    "response_role": role,
                    "prospective_split": triplet["prospective_split"],
                    "assigned_category": triplet["assigned_category"],
                    "response_normalized_sha256": sha256_text(normalize_text(response)),
                    "is_hidden_duplicate": False,
                    "duplicate_of": None,
                }
            )
            structure_id = "oacpo_d1_structure_" + stable_rank(
                seed, "structure", triplet["pair_id"], role
            )[:20]
            structure_packet.append({"opaque_structure_id": structure_id, "response": response})
            structure_key.append(
                {
                    "opaque_structure_id": structure_id,
                    "pair_id": triplet["pair_id"],
                    "response_role": role,
                    "prospective_split": triplet["prospective_split"],
                    "assigned_category": triplet["assigned_category"],
                    "response_normalized_sha256": sha256_text(normalize_text(response)),
                    "is_hidden_duplicate": False,
                    "duplicate_of": None,
                }
            )

    safety_packet, safety_key = add_duplicates(
        safety_packet,
        safety_key,
        ("authored", "source_safe", "source_unsafe"),
        int(design["safety_duplicates_per_role"]),
        seed,
        "safety",
        "opaque_safety_id",
    )
    quality_packet, quality_key = add_duplicates(
        quality_packet,
        quality_key,
        ("authored", "source_safe"),
        int(design["quality_duplicates_per_role"]),
        seed,
        "quality",
        "opaque_quality_id",
    )
    structure_packet, structure_key = add_duplicates(
        structure_packet,
        structure_key,
        ("authored", "source_safe"),
        int(design["structure_duplicates_per_role"]),
        seed,
        "structure",
        "opaque_structure_id",
    )

    def reviewer_views(rows: list[dict[str, Any]], construct: str, id_field: str) -> tuple[list, list]:
        a = packet_order(rows, seed, f"{construct}-a", id_field)
        b = packet_order(rows, seed, f"{construct}-b", id_field)
        if [row[id_field] for row in a] == [row[id_field] for row in b]:
            b.reverse()
        return a, b

    safety_a, safety_b = reviewer_views(safety_packet, "safety", "opaque_safety_id")
    quality_a, quality_b = reviewer_views(quality_packet, "quality", "opaque_quality_id")
    structure_a, structure_b = reviewer_views(structure_packet, "structure", "opaque_structure_id")

    def safety_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"opaque_safety_id": row["opaque_safety_id"], "clean_safe": "", "short_reason": ""}
            for row in rows
        ]

    def quality_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "opaque_quality_id": row["opaque_quality_id"],
                "detail_retention": "",
                "safe_actionability": "",
                "naturalness": "",
                "short_reason": "",
            }
            for row in rows
        ]

    def structure_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "opaque_structure_id": row["opaque_structure_id"],
                "mechanical_template": "",
                "short_reason": "",
            }
            for row in rows
        ]

    return {
        "safety_a_packet": safety_a,
        "safety_b_packet": safety_b,
        "safety_a_template": safety_template(safety_a),
        "safety_b_template": safety_template(safety_b),
        "safety_key": sorted(safety_key, key=lambda row: row["opaque_safety_id"]),
        "quality_a_packet": quality_a,
        "quality_b_packet": quality_b,
        "quality_a_template": quality_template(quality_a),
        "quality_b_template": quality_template(quality_b),
        "quality_key": sorted(quality_key, key=lambda row: row["opaque_quality_id"]),
        "structure_a_packet": structure_a,
        "structure_b_packet": structure_b,
        "structure_a_template": structure_template(structure_a),
        "structure_b_template": structure_template(structure_b),
        "structure_key": sorted(structure_key, key=lambda row: row["opaque_structure_id"]),
    }


def validate_packets(packets: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]) -> None:
    design = protocol["packet_design"]
    n = int(design["primary_triplets"])
    expected = {
        "safety": 3 * n + 3 * int(design["safety_duplicates_per_role"]),
        "quality": 2 * n + 2 * int(design["quality_duplicates_per_role"]),
        "structure": 2 * n + 2 * int(design["structure_duplicates_per_role"]),
    }
    for construct in expected:
        names = [name for name in packets if name.startswith(construct + "_")]
        if any(len(packets[name]) != expected[construct] for name in names):
            raise ValueError(f"wrong {construct} packet row count")
    if any(set(row) != {"opaque_safety_id", "prompt", "response"} for row in packets["safety_a_packet"]):
        raise ValueError("private field leaked into safety packet")
    if any(set(row) != {"opaque_quality_id", "prompt", "response"} for row in packets["quality_a_packet"]):
        raise ValueError("private field leaked into quality packet")
    if any(set(row) != {"opaque_structure_id", "response"} for row in packets["structure_a_packet"]):
        raise ValueError("private field leaked into structure packet")
    for construct, id_field in (
        ("safety", "opaque_safety_id"),
        ("quality", "opaque_quality_id"),
        ("structure", "opaque_structure_id"),
    ):
        a = packets[f"{construct}_a_packet"]
        b = packets[f"{construct}_b_packet"]
        if {row[id_field] for row in a} != {row[id_field] for row in b}:
            raise ValueError(f"{construct} reviewer row sets differ")
        if [row[id_field] for row in a] == [row[id_field] for row in b]:
            raise ValueError(f"{construct} reviewer orders do not differ")
        key = packets[f"{construct}_key"]
        duplicates = [row for row in key if row["is_hidden_duplicate"]]
        if len(duplicates) != expected[construct] - ({"safety": 3, "quality": 2, "structure": 2}[construct] * n):
            raise ValueError(f"{construct} duplicate count mismatch")


def rendered_outputs(protocol: dict[str, Any]) -> dict[str, str]:
    triplets = load_triplet_sources(protocol)
    packets = build_packets(triplets, protocol)
    validate_packets(packets, protocol)
    rendered = {name: jsonl_text(rows) for name, rows in packets.items()}
    manifest = {
        "name": "step3_oacpo_d1_blind_packet_manifest_v1",
        "protocol": protocol["self_path"],
        "protocol_sha256": sha256(resolve_repo_path(protocol["self_path"])),
        "route": "OACPO_D1_PACKETS_FROZEN_AUTHORIZE_INDEPENDENT_A_B_ONLY",
        "primary_triplets": len(triplets),
        "split_counts": dict(Counter(row["prospective_split"] for row in triplets)),
        "category_counts": dict(Counter(row["assigned_category"] for row in triplets)),
        "row_counts": {name: len(rows) for name, rows in packets.items()},
        "artifact_sha256": {name: sha256_text(text) for name, text in rendered.items()},
        "roles_hidden": True,
        "source_labels_used_as_final": False,
        "review_labels_created": 0,
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
            raise ValueError(f"refusing to overwrite different frozen packet: {name}")
        path.write_text(rendered[name], encoding="utf-8", newline="\n")


def verify_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    for name, path in output_paths(protocol).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered[name]:
            raise ValueError(f"frozen packet mismatch: {name}")


def main() -> None:
    args = parse_args()
    protocol = read_json(resolve_repo_path(args.protocol))
    if protocol["self_path"] != args.protocol.replace("\\", "/"):
        raise ValueError("protocol self path mismatch")
    verify_frozen_inputs(protocol)
    rendered = rendered_outputs(protocol)
    if args.command == "preflight":
        print("PASS_OACPO_D1_PACKET_PREFLIGHT")
    elif args.command == "freeze":
        write_outputs(protocol, rendered)
        print("OACPO_D1_PACKETS_FROZEN_AUTHORIZE_INDEPENDENT_A_B_ONLY")
    else:
        verify_outputs(protocol, rendered)
        print("PASS_OACPO_D1_PACKET_VERIFY")


if __name__ == "__main__":
    main()
