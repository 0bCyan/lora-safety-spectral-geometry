"""Freeze and verify blinded CSPR P2 safety and actionability packets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "verify"])
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


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def verify_protocol(protocol: dict[str, Any]) -> None:
    for label, artifact in protocol["artifacts"].items():
        path = resolve_repo_path(artifact["path"])
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact hash mismatch: {label}")


def load_selected_source_rows(
    source_path: Path, ledger: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    wanted = {
        int(row[field])
        for row in ledger
        for field in ("chosen_source_index", "rejected_source_index")
    }
    found: dict[int, dict[str, Any]] = {}
    with gzip.open(source_path, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle, start=1):
            if source_index not in wanted:
                continue
            source = json.loads(line)
            prompt = source.get("prompt")
            response = source.get("response")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"invalid selected prompt at source row {source_index}")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"invalid selected response at source row {source_index}")
            if not isinstance(source.get("is_safe"), bool):
                raise ValueError(f"invalid selected safety label at source row {source_index}")
            found[source_index] = {
                "prompt": prompt.strip(),
                "response": response.strip(),
                "is_safe": source["is_safe"],
            }
    if set(found) != wanted:
        raise ValueError("selected source rows are missing")
    return found


def validate_and_expand_pairs(
    ledger: list[dict[str, Any]], source_rows: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    seen_prompts: set[str] = set()
    for row in ledger:
        pair_id = row["pair_id"]
        prompt_hash = row["prompt_normalized_sha256"]
        if pair_id in seen_pairs or prompt_hash in seen_prompts:
            raise ValueError("duplicate pair or prompt in ledger")
        chosen = source_rows[int(row["chosen_source_index"])]
        rejected = source_rows[int(row["rejected_source_index"])]
        if not chosen["is_safe"] or rejected["is_safe"]:
            raise ValueError(f"source roles changed for {pair_id}")
        chosen_prompt_hash = sha256_text(normalize_text(chosen["prompt"]))
        rejected_prompt_hash = sha256_text(normalize_text(rejected["prompt"]))
        if chosen_prompt_hash != prompt_hash or rejected_prompt_hash != prompt_hash:
            raise ValueError(f"same-prompt binding failed for {pair_id}")
        if sha256_text(chosen["prompt"]) != row["prompt_exact_sha256"]:
            raise ValueError(f"prompt exact hash mismatch for {pair_id}")
        for role, source, exact_field, normalized_field in (
            (
                "chosen",
                chosen,
                "chosen_response_sha256",
                "chosen_response_normalized_sha256",
            ),
            (
                "rejected",
                rejected,
                "rejected_response_sha256",
                "rejected_response_normalized_sha256",
            ),
        ):
            if sha256_text(source["response"]) != row[exact_field]:
                raise ValueError(f"{role} exact response hash mismatch for {pair_id}")
            if sha256_text(normalize_text(source["response"])) != row[normalized_field]:
                raise ValueError(f"{role} normalized response hash mismatch for {pair_id}")
        pairs.append(
            {
                **row,
                "prompt": chosen["prompt"],
                "chosen_response": chosen["response"],
                "rejected_response": rejected["response"],
            }
        )
        seen_pairs.add(pair_id)
        seen_prompts.add(prompt_hash)
    return pairs


def ordered(rows: list[dict[str, Any]], seed: int, label: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_rank(seed, label, next(iter(row.values()))))


def packet_order(
    rows: list[dict[str, Any]], seed: int, label: str, id_field: str
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_rank(seed, label, str(row[id_field])))


def build_packets(
    pairs: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    cfg = protocol["packet_design"]
    seed = int(cfg["seed"])
    safety_primary: list[dict[str, Any]] = []
    safety_key: list[dict[str, Any]] = []
    action_primary: list[dict[str, Any]] = []
    action_key: list[dict[str, Any]] = []

    for pair in pairs:
        for role in ("chosen", "rejected"):
            opaque_id = "cspr_p2_safe_" + stable_rank(
                seed, "safety-primary", pair["pair_id"], role
            )[:20]
            response = pair[f"{role}_response"]
            safety_primary.append(
                {"opaque_safety_id": opaque_id, "prompt": pair["prompt"], "response": response}
            )
            safety_key.append(
                {
                    "opaque_safety_id": opaque_id,
                    "pair_id": pair["pair_id"],
                    "response_role": role,
                    "prospective_split": pair["prospective_split"],
                    "assigned_category": pair["assigned_category"],
                    "source_index": pair[f"{role}_source_index"],
                    "prompt_normalized_sha256": pair["prompt_normalized_sha256"],
                    "response_normalized_sha256": pair[
                        f"{role}_response_normalized_sha256"
                    ],
                    "is_hidden_duplicate": False,
                    "duplicate_of": None,
                }
            )

        action_id = "cspr_p2_action_" + stable_rank(
            seed, "action-primary", pair["pair_id"]
        )[:20]
        action_primary.append(
            {
                "opaque_action_id": action_id,
                "prompt": pair["prompt"],
                "response": pair["chosen_response"],
            }
        )
        action_key.append(
            {
                "opaque_action_id": action_id,
                "pair_id": pair["pair_id"],
                "prospective_split": pair["prospective_split"],
                "assigned_category": pair["assigned_category"],
                "source_index": pair["chosen_source_index"],
                "prompt_normalized_sha256": pair["prompt_normalized_sha256"],
                "response_normalized_sha256": pair[
                    "chosen_response_normalized_sha256"
                ],
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )

    safety_by_pair_role = {
        (row["pair_id"], row["response_role"]): row for row in safety_key
    }
    safety_packet_by_id = {row["opaque_safety_id"]: row for row in safety_primary}
    safety_duplicate_rows: list[dict[str, Any]] = []
    safety_duplicate_keys: list[dict[str, Any]] = []
    per_role = int(cfg["safety_duplicates_per_role"])
    for role in ("chosen", "rejected"):
        selected = sorted(
            pairs,
            key=lambda pair: stable_rank(seed, "safety-duplicate", role, pair["pair_id"]),
        )[:per_role]
        for pair in selected:
            primary_key = safety_by_pair_role[(pair["pair_id"], role)]
            primary = safety_packet_by_id[primary_key["opaque_safety_id"]]
            duplicate_id = "cspr_p2_safe_" + stable_rank(
                seed, "safety-duplicate-id", pair["pair_id"], role
            )[:20]
            safety_duplicate_rows.append(
                {
                    "opaque_safety_id": duplicate_id,
                    "prompt": primary["prompt"],
                    "response": primary["response"],
                }
            )
            safety_duplicate_keys.append(
                {
                    **primary_key,
                    "opaque_safety_id": duplicate_id,
                    "is_hidden_duplicate": True,
                    "duplicate_of": primary["opaque_safety_id"],
                }
            )

    action_packet_by_id = {row["opaque_action_id"]: row for row in action_primary}
    action_key_by_pair = {row["pair_id"]: row for row in action_key}
    action_duplicate_rows: list[dict[str, Any]] = []
    action_duplicate_keys: list[dict[str, Any]] = []
    selected_actions = sorted(
        pairs, key=lambda pair: stable_rank(seed, "action-duplicate", pair["pair_id"])
    )[: int(cfg["actionability_duplicates"])]
    for pair in selected_actions:
        primary_key = action_key_by_pair[pair["pair_id"]]
        primary = action_packet_by_id[primary_key["opaque_action_id"]]
        duplicate_id = "cspr_p2_action_" + stable_rank(
            seed, "action-duplicate-id", pair["pair_id"]
        )[:20]
        action_duplicate_rows.append(
            {
                "opaque_action_id": duplicate_id,
                "prompt": primary["prompt"],
                "response": primary["response"],
            }
        )
        action_duplicate_keys.append(
            {
                **primary_key,
                "opaque_action_id": duplicate_id,
                "is_hidden_duplicate": True,
                "duplicate_of": primary["opaque_action_id"],
            }
        )

    safety_all = safety_primary + safety_duplicate_rows
    action_all = action_primary + action_duplicate_rows
    safety_a = packet_order(safety_all, seed, "safety-a-order", "opaque_safety_id")
    safety_b = packet_order(safety_all, seed, "safety-b-order", "opaque_safety_id")
    action_a = packet_order(action_all, seed, "action-a-order", "opaque_action_id")
    action_b = packet_order(action_all, seed, "action-b-order", "opaque_action_id")
    if [row["opaque_safety_id"] for row in safety_a] == [
        row["opaque_safety_id"] for row in safety_b
    ]:
        safety_b.reverse()
    if [row["opaque_action_id"] for row in action_a] == [
        row["opaque_action_id"] for row in action_b
    ]:
        action_b.reverse()

    return {
        "safety_a_packet": safety_a,
        "safety_b_packet": safety_b,
        "safety_a_template": [
            {"opaque_safety_id": row["opaque_safety_id"], "clean_safe": "", "short_reason": ""}
            for row in safety_a
        ],
        "safety_b_template": [
            {"opaque_safety_id": row["opaque_safety_id"], "clean_safe": "", "short_reason": ""}
            for row in safety_b
        ],
        "safety_key": sorted(
            safety_key + safety_duplicate_keys, key=lambda row: row["opaque_safety_id"]
        ),
        "action_a_packet": action_a,
        "action_b_packet": action_b,
        "action_a_template": [
            {
                "opaque_action_id": row["opaque_action_id"],
                "safe_route": "",
                "executable_step": "",
                "prompt_binding": "",
                "short_reason": "",
            }
            for row in action_a
        ],
        "action_b_template": [
            {
                "opaque_action_id": row["opaque_action_id"],
                "safe_route": "",
                "executable_step": "",
                "prompt_binding": "",
                "short_reason": "",
            }
            for row in action_b
        ],
        "action_key": sorted(
            action_key + action_duplicate_keys, key=lambda row: row["opaque_action_id"]
        ),
    }


def validate_packet_structure(
    packets: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]
) -> None:
    cfg = protocol["packet_design"]
    primary = int(cfg["primary_pairs"])
    safety_expected = 2 * primary + 2 * int(cfg["safety_duplicates_per_role"])
    action_expected = primary + int(cfg["actionability_duplicates"])
    for name in ("safety_a_packet", "safety_b_packet", "safety_a_template", "safety_b_template", "safety_key"):
        if len(packets[name]) != safety_expected:
            raise ValueError(f"wrong safety row count: {name}")
    for name in ("action_a_packet", "action_b_packet", "action_a_template", "action_b_template", "action_key"):
        if len(packets[name]) != action_expected:
            raise ValueError(f"wrong actionability row count: {name}")
    forbidden = {
        "pair_id",
        "response_role",
        "prospective_split",
        "assigned_category",
        "source_index",
        "is_safe",
        "duplicate_of",
        "is_hidden_duplicate",
    }
    for name in ("safety_a_packet", "safety_b_packet", "action_a_packet", "action_b_packet"):
        if any(forbidden & set(row) for row in packets[name]):
            raise ValueError(f"private field leaked into {name}")
    safety_roles = Counter(
        row["response_role"] for row in packets["safety_key"] if row["is_hidden_duplicate"]
    )
    if safety_roles != Counter(
        {"chosen": int(cfg["safety_duplicates_per_role"]), "rejected": int(cfg["safety_duplicates_per_role"])}
    ):
        raise ValueError("safety duplicate roles are not balanced")


def output_texts(
    protocol_path: Path, protocol: dict[str, Any]
) -> dict[Path, str]:
    ledger_path = resolve_repo_path(protocol["inputs"]["ledger"]["path"])
    source_path = resolve_repo_path(protocol["inputs"]["source"]["path"])
    ledger = read_jsonl(ledger_path)
    if len(ledger) != int(protocol["packet_design"]["primary_pairs"]):
        raise ValueError("unexpected P1R ledger row count")
    if sha256(source_path) != protocol["inputs"]["source"]["sha256"]:
        raise ValueError("raw source hash mismatch")
    source_rows = load_selected_source_rows(source_path, ledger)
    pairs = validate_and_expand_pairs(ledger, source_rows)
    packets = build_packets(pairs, protocol)
    validate_packet_structure(packets, protocol)

    outputs = protocol["outputs"]
    text_by_path = {
        resolve_repo_path(outputs[name]): jsonl_text(packets[name])
        for name in (
            "safety_a_packet",
            "safety_b_packet",
            "safety_a_template",
            "safety_b_template",
            "safety_key",
            "action_a_packet",
            "action_b_packet",
            "action_a_template",
            "action_b_template",
            "action_key",
        )
    }
    manifest = {
        "name": "step3_cspr_p2_packet_manifest_v1",
        "protocol": str(protocol_path.relative_to(REPO)).replace("\\", "/"),
        "protocol_sha256": sha256(protocol_path),
        "source_sha256": protocol["inputs"]["source"]["sha256"],
        "ledger_sha256": protocol["inputs"]["ledger"]["sha256"],
        "primary_pairs": len(pairs),
        "safety_primary_rows": 2 * len(pairs),
        "safety_packet_rows_per_reviewer": len(packets["safety_a_packet"]),
        "actionability_primary_rows": len(pairs),
        "actionability_packet_rows_per_reviewer": len(packets["action_a_packet"]),
        "artifacts": [
            {
                "path": str(path.relative_to(REPO)).replace("\\", "/"),
                "bytes": len(text.encode("utf-8")),
                "sha256": sha256_text(text),
            }
            for path, text in sorted(text_by_path.items(), key=lambda item: str(item[0]))
        ],
        "source_role_visible_to_reviewers": False,
        "prospective_split_visible_to_reviewers": False,
        "source_label_visible_to_reviewers": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    manifest_path = resolve_repo_path(outputs["manifest"])
    text_by_path[manifest_path] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    report = f"""# CSPR P2 Packet Freeze Report

The exact P1R ledger produced {len(pairs)} same-prompt candidate pairs. Safety reviewers A and B each receive {len(packets['safety_a_packet'])} independently ordered rows: {2 * len(pairs)} primary responses with chosen/rejected roles hidden plus {2 * int(protocol['packet_design']['safety_duplicates_per_role'])} hidden duplicates balanced across roles. Actionability reviewers A and B each receive {len(packets['action_a_packet'])} independently ordered rows: {len(pairs)} primary candidate responses plus {int(protocol['packet_design']['actionability_duplicates'])} hidden duplicates.

Reviewer-visible packets contain only opaque ID, prompt, and response. They do not expose pair ID, source role, source label, category, prospective split, duplicate links, or training information. Hidden keys and reviewer templates are frozen separately.

This package authorizes only independent blinded A/B review under the two unchanged Step3O rubrics. It does not validate any pair, open Reviewer C, compute a model gradient, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
"""
    text_by_path[resolve_repo_path(outputs["report"])] = report
    return text_by_path


def freeze(protocol_path: Path, protocol: dict[str, Any]) -> None:
    texts = output_texts(protocol_path, protocol)
    existing = [path for path in texts if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    for path, text in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS_CSPR_P2_PACKET_FREEZE", "files": len(texts)}))


def verify(protocol_path: Path, protocol: dict[str, Any]) -> None:
    texts = output_texts(protocol_path, protocol)
    for path, expected in texts.items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"packet artifact does not reproduce: {path}")
    print(json.dumps({"status": "PASS_CSPR_P2_PACKET_VERIFY", "files": len(texts)}))


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    protocol = read_json(protocol_path)
    verify_protocol(protocol)
    if args.command == "freeze":
        freeze(protocol_path, protocol)
    else:
        verify(protocol_path, protocol)


if __name__ == "__main__":
    main()
