"""Freeze and verify the OACPO D0 safe-response authoring packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "freeze-author", "verify-author"])
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
        if not path.is_file():
            raise ValueError(f"missing frozen input: {label}")
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {label}")


def load_source_pairs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = protocol["frozen_inputs"]
    ledger = read_jsonl(resolve_repo_path(inputs["selected_pair_ledger"]["path"]))
    packet = read_jsonl(resolve_repo_path(inputs["p2_safety_packet"]["path"]))
    hidden_key = read_jsonl(resolve_repo_path(inputs["p2_safety_hidden_key"]["path"]))
    packet_by_id = {row["opaque_safety_id"]: row for row in packet}

    primary_by_pair_role: dict[tuple[str, str], dict[str, Any]] = {}
    for key in hidden_key:
        if key["is_hidden_duplicate"]:
            continue
        opaque_id = key["opaque_safety_id"]
        visible = packet_by_id.get(opaque_id)
        if visible is None:
            raise ValueError(f"missing visible P2 row: {opaque_id}")
        joined = {**key, "prompt": visible["prompt"], "response": visible["response"]}
        pair_role = (key["pair_id"], key["response_role"])
        if pair_role in primary_by_pair_role:
            raise ValueError(f"duplicate P2 primary role: {pair_role}")
        primary_by_pair_role[pair_role] = joined

    pairs: list[dict[str, Any]] = []
    for row in ledger:
        pair_id = row["pair_id"]
        chosen = primary_by_pair_role.get((pair_id, "chosen"))
        rejected = primary_by_pair_role.get((pair_id, "rejected"))
        if chosen is None or rejected is None:
            raise ValueError(f"missing P2 pair roles: {pair_id}")
        if normalize_text(chosen["prompt"]) != normalize_text(rejected["prompt"]):
            raise ValueError(f"same-prompt binding failed: {pair_id}")
        if sha256_text(normalize_text(chosen["prompt"])) != row["prompt_normalized_sha256"]:
            raise ValueError(f"prompt hash mismatch: {pair_id}")
        if chosen["response_normalized_sha256"] != row["chosen_response_normalized_sha256"]:
            raise ValueError(f"chosen response hash mismatch: {pair_id}")
        if rejected["response_normalized_sha256"] != row["rejected_response_normalized_sha256"]:
            raise ValueError(f"rejected response hash mismatch: {pair_id}")
        if chosen["prospective_split"] != row["prospective_split"]:
            raise ValueError(f"split mismatch: {pair_id}")
        if chosen["assigned_category"] != row["assigned_category"]:
            raise ValueError(f"category mismatch: {pair_id}")
        pairs.append(
            {
                **row,
                "prompt": chosen["prompt"],
                "source_chosen_response": chosen["response"],
                "source_rejected_response": rejected["response"],
            }
        )
    if len(pairs) != int(protocol["source_design"]["source_pair_count"]):
        raise ValueError("unexpected source pair count")
    return pairs


def select_pairs(pairs: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    design = protocol["selection_design"]
    seed = int(design["seed"])
    categories = sorted({row["assigned_category"] for row in pairs})
    if len(categories) != int(design["category_count"]):
        raise ValueError("unexpected category count")

    selected: list[dict[str, Any]] = []
    for split, per_category in design["per_split_per_category"].items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pairs:
            if row["prospective_split"] == split:
                grouped[row["assigned_category"]].append(row)
        if set(grouped) != set(categories):
            raise ValueError(f"category coverage mismatch in split: {split}")
        for category in categories:
            ranked = sorted(
                grouped[category],
                key=lambda row: stable_rank(seed, split, category, row["pair_id"]),
            )
            count = int(per_category)
            if len(ranked) < count:
                raise ValueError(f"insufficient rows for {split}/{category}")
            selected.extend(ranked[:count])

    expected = int(design["selected_pair_count"])
    if len(selected) != expected or len({row["pair_id"] for row in selected}) != expected:
        raise ValueError("selected pair count or uniqueness mismatch")
    return sorted(selected, key=lambda row: row["pair_id"])


def build_author_artifacts(
    selected: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    seed = int(protocol["author_packet_design"]["seed"])
    visible: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in selected:
        author_id = "oacpo_d0_author_" + stable_rank(seed, row["pair_id"])[:20]
        visible.append({"author_id": author_id, "prompt": row["prompt"]})
        template.append({"author_id": author_id, "authored_response": ""})
        private.append(
            {
                "author_id": author_id,
                "pair_id": row["pair_id"],
                "prospective_split": row["prospective_split"],
                "assigned_category": row["assigned_category"],
                "prompt_normalized_sha256": row["prompt_normalized_sha256"],
                "source_chosen_response_normalized_sha256": row[
                    "chosen_response_normalized_sha256"
                ],
                "source_rejected_response_normalized_sha256": row[
                    "rejected_response_normalized_sha256"
                ],
            }
        )
        ledger.append(
            {
                "pair_id": row["pair_id"],
                "prospective_split": row["prospective_split"],
                "assigned_category": row["assigned_category"],
                "prompt_normalized_sha256": row["prompt_normalized_sha256"],
                "chosen_source_index": row["chosen_source_index"],
                "rejected_source_index": row["rejected_source_index"],
                "chosen_response_normalized_sha256": row[
                    "chosen_response_normalized_sha256"
                ],
                "rejected_response_normalized_sha256": row[
                    "rejected_response_normalized_sha256"
                ],
            }
        )

    visible = sorted(visible, key=lambda row: stable_rank(seed, "visible", row["author_id"]))
    template_by_id = {row["author_id"]: row for row in template}
    template = [template_by_id[row["author_id"]] for row in visible]
    return {
        "author_packet": visible,
        "author_template": template,
        "author_private_map": sorted(private, key=lambda row: row["author_id"]),
        "selected_source_ledger": sorted(ledger, key=lambda row: row["pair_id"]),
    }


def validate_author_artifacts(
    artifacts: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]
) -> None:
    expected = int(protocol["selection_design"]["selected_pair_count"])
    for rows in artifacts.values():
        if len(rows) != expected:
            raise ValueError("author artifact row count mismatch")
    packet = artifacts["author_packet"]
    template = artifacts["author_template"]
    if any(set(row) != {"author_id", "prompt"} for row in packet):
        raise ValueError("private field leaked into author packet")
    if any(set(row) != {"author_id", "authored_response"} for row in template):
        raise ValueError("author template schema mismatch")
    if any(row["authored_response"] for row in template):
        raise ValueError("author template is not blank")
    if [row["author_id"] for row in packet] != [row["author_id"] for row in template]:
        raise ValueError("author packet/template order mismatch")
    if len({row["author_id"] for row in packet}) != expected:
        raise ValueError("duplicate author ID")

    private = artifacts["author_private_map"]
    split_counts = Counter(row["prospective_split"] for row in private)
    category_counts = Counter(row["assigned_category"] for row in private)
    if dict(split_counts) != protocol["selection_design"]["expected_split_counts"]:
        raise ValueError("selected split counts mismatch")
    if set(category_counts.values()) != {int(protocol["selection_design"]["per_category_total"])}:
        raise ValueError("selected category balance mismatch")


def output_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {key: resolve_repo_path(value) for key, value in protocol["outputs"].items()}


def rendered_artifacts(protocol: dict[str, Any]) -> dict[str, str]:
    selected = select_pairs(load_source_pairs(protocol), protocol)
    artifacts = build_author_artifacts(selected, protocol)
    validate_author_artifacts(artifacts, protocol)
    rendered = {name: jsonl_text(rows) for name, rows in artifacts.items()}
    manifest = {
        "name": "step3_oacpo_d0_author_packet_manifest_v1",
        "protocol": protocol["self_path"],
        "protocol_sha256": sha256(resolve_repo_path(protocol["self_path"])),
        "route": "OACPO_D0_AUTHOR_PACKET_FROZEN",
        "row_count": len(selected),
        "split_counts": dict(Counter(row["prospective_split"] for row in selected)),
        "category_counts": dict(Counter(row["assigned_category"] for row in selected)),
        "artifact_sha256": {name: sha256_text(text) for name, text in rendered.items()},
        "source_labels_accepted_as_final": False,
        "candidate_responses_opened_by_author": False,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    rendered["manifest"] = json_text(manifest)
    return rendered


def write_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    paths = output_paths(protocol)
    for name, text in rendered.items():
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite different frozen artifact: {name}")
        path.write_text(text, encoding="utf-8", newline="\n")


def verify_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    paths = output_paths(protocol)
    for name, expected in rendered.items():
        path = paths[name]
        if not path.is_file():
            raise ValueError(f"missing output: {name}")
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"output mismatch: {name}")


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    protocol = read_json(protocol_path)
    if protocol["self_path"] != args.protocol.replace("\\", "/"):
        raise ValueError("protocol self path mismatch")
    verify_frozen_inputs(protocol)
    rendered = rendered_artifacts(protocol)
    if args.command == "preflight":
        print("PASS_OACPO_D0_PREFLIGHT")
        return
    if args.command == "freeze-author":
        write_outputs(protocol, rendered)
        print("OACPO_D0_AUTHOR_PACKET_FROZEN")
        return
    verify_outputs(protocol, rendered)
    print("PASS_OACPO_D0_AUTHOR_PACKET_VERIFY")


if __name__ == "__main__":
    main()
