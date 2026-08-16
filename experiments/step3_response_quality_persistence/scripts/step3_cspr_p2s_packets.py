"""Freeze blinded CSPR P2S calibration-interleaved review packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2s/"
    "protocol/cspr_p2s_blinded_review_protocol.json"
)


class PacketValidationError(ValueError):
    """Raised when a frozen P2S packet input or output is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "verify"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise PacketValidationError(f"path escapes repository: {value}") from exc
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


def stable_rank(seed: int, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_frozen(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def verify_inputs(protocol: dict[str, Any]) -> None:
    for name, artifact in protocol["frozen_inputs"].items():
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise PacketValidationError(f"frozen input hash mismatch: {name}")


def select_category_diverse(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    groups: list[Any],
    per_group: int,
    id_field: str,
    category_field: str,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    selected = []
    for group in groups:
        available = [row for row in rows if row[group_field] == group]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in available:
            by_category[str(row[category_field])].append(row)
        first_pass = []
        for category, category_rows in by_category.items():
            first_pass.append(
                min(
                    category_rows,
                    key=lambda row: stable_rank(
                        seed, namespace, str(group), category, str(row[id_field])
                    ),
                )
            )
        first_pass.sort(
            key=lambda row: stable_rank(
                seed, namespace, str(group), "category-first", str(row[id_field])
            )
        )
        group_selected = first_pass[:per_group]
        selected_ids = {str(row[id_field]) for row in group_selected}
        remainder = sorted(
            [row for row in available if str(row[id_field]) not in selected_ids],
            key=lambda row: stable_rank(
                seed, namespace, str(group), "fill", str(row[id_field])
            ),
        )
        group_selected.extend(remainder[: per_group - len(group_selected)])
        if len(group_selected) != per_group:
            raise PacketValidationError(f"insufficient calibration supply for group {group}")
        selected.extend(group_selected)
    return selected


def load_candidates(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    ledger = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["candidate_ledger"]["path"])
    )
    candidates = [row for row in ledger if row["mechanically_valid"]]
    expected = int(protocol["packet_design"]["target_candidate_rows"])
    if len(candidates) != expected:
        raise PacketValidationError("unexpected mechanically valid candidate count")
    prompt_by_authoring_id = {}
    for name in ("author_a_packet", "author_b_packet"):
        for row in read_jsonl(resolve_repo_path(protocol["frozen_inputs"][name]["path"])):
            prompt_by_authoring_id[str(row["opaque_authoring_id"])] = str(row["prompt"])
    if any(str(row["opaque_authoring_id"]) not in prompt_by_authoring_id for row in candidates):
        raise PacketValidationError("candidate prompt is missing")
    return candidates, prompt_by_authoring_id


def load_rejected(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    packet = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["p2_safety_packet"]["path"])
    )
    key_rows = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["p2_safety_hidden_key"]["path"])
    )
    packet_by_id = {str(row["opaque_safety_id"]): row for row in packet}
    rejected = []
    for key in key_rows:
        if key["is_hidden_duplicate"] or key["response_role"] != "rejected":
            continue
        row = packet_by_id[str(key["opaque_safety_id"])]
        rejected.append(
            {
                "pair_id": key["pair_id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "prospective_split": key["prospective_split"],
                "assigned_category": key["assigned_category"],
                "source_safety_id": key["opaque_safety_id"],
            }
        )
    expected = int(protocol["packet_design"]["target_rejected_rows"])
    if len(rejected) != expected or len({row["pair_id"] for row in rejected}) != expected:
        raise PacketValidationError("unexpected rejected target count")
    return rejected


def load_action_calibration(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = protocol["frozen_inputs"]
    packet = read_jsonl(resolve_repo_path(inputs["action_calibration_packet"]["path"]))
    packet_by_id = {str(row["opaque_action_id"]): row for row in packet}
    key_rows = read_csv(resolve_repo_path(inputs["action_calibration_hidden_key"]["path"]))
    key_by_id = {str(row["opaque_action_id"]): row for row in key_rows}
    labels = read_jsonl(resolve_repo_path(inputs["action_calibration_final_labels"]["path"]))
    rows = []
    for label in labels:
        opaque_id = str(label["opaque_action_id"])
        packet_row = packet_by_id[opaque_id]
        key = key_by_id[opaque_id]
        rows.append(
            {
                "source_id": opaque_id,
                "prompt": packet_row["prompt"],
                "response": packet_row["response"],
                "assigned_category": key["assigned_category"],
                "reference_safe_route": label["safe_route"],
                "reference_executable_step": label["executable_step"],
                "reference_prompt_binding": label["prompt_binding"],
                "reference_score": int(label["safe_actionability"]),
            }
        )
    cfg = protocol["packet_design"]
    return select_category_diverse(
        rows,
        group_field="reference_score",
        groups=[0, 1, 2],
        per_group=int(cfg["action_calibration_rows_per_score"]),
        id_field="source_id",
        category_field="assigned_category",
        seed=int(cfg["seed"]),
        namespace="action-calibration",
    )


def load_safety_calibration(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = protocol["frozen_inputs"]
    packet = read_jsonl(resolve_repo_path(inputs["safety_calibration_packet"]["path"]))
    packet_by_id = {str(row["opaque_safety_id"]): row for row in packet}
    key_rows = read_csv(resolve_repo_path(inputs["safety_calibration_hidden_key"]["path"]))
    key_by_id = {str(row["opaque_safety_id"]): row for row in key_rows}
    labels = read_jsonl(resolve_repo_path(inputs["safety_calibration_final_labels"]["path"]))
    rows = []
    for label in labels:
        opaque_id = str(label["opaque_safety_id"])
        packet_row = packet_by_id[opaque_id]
        key = key_by_id[opaque_id]
        rows.append(
            {
                "source_id": opaque_id,
                "prompt": packet_row["prompt"],
                "response": packet_row["response"],
                "assigned_category": key["assigned_category"],
                "reference_clean_safe": label["clean_safe"],
            }
        )
    cfg = protocol["packet_design"]
    return select_category_diverse(
        rows,
        group_field="reference_clean_safe",
        groups=["yes", "no"],
        per_group=int(cfg["safety_calibration_rows_per_label"]),
        id_field="source_id",
        category_field="assigned_category",
        seed=int(cfg["seed"]),
        namespace="safety-calibration",
    )


def add_duplicate(
    packet_row: dict[str, Any], key_row: dict[str, Any], new_id: str, id_field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_id = packet_row[id_field]
    return (
        {**packet_row, id_field: new_id},
        {
            **key_row,
            id_field: new_id,
            "is_hidden_duplicate": True,
            "duplicate_of": original_id,
        },
    )


def build_packets(
    candidates: list[dict[str, Any]],
    prompts: dict[str, str],
    rejected: list[dict[str, Any]],
    action_calibration: list[dict[str, Any]],
    safety_calibration: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    cfg = protocol["packet_design"]
    seed = int(cfg["seed"])
    action_packet = []
    action_key = []
    safety_packet = []
    safety_key = []

    for candidate in candidates:
        authoring_id = str(candidate["opaque_authoring_id"])
        action_id = "cspr_p2s_action_" + stable_rank(seed, "action-target", authoring_id)[:20]
        safety_id = "cspr_p2s_safe_" + stable_rank(seed, "safety-candidate", authoring_id)[:20]
        visible = {"prompt": prompts[authoring_id], "response": candidate["response"]}
        action_packet.append({"opaque_action_id": action_id, **visible})
        action_key.append(
            {
                "opaque_action_id": action_id,
                "item_type": "target_candidate",
                "pair_id": candidate["pair_id"],
                "source_authoring_id": authoring_id,
                "author_role": candidate["author_role"],
                "prospective_split": candidate["prospective_split"],
                "assigned_category": candidate["assigned_category"],
                "reference_safe_route": None,
                "reference_executable_step": None,
                "reference_prompt_binding": None,
                "reference_score": None,
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )
        safety_packet.append({"opaque_safety_id": safety_id, **visible})
        safety_key.append(
            {
                "opaque_safety_id": safety_id,
                "item_type": "target_candidate",
                "pair_id": candidate["pair_id"],
                "source_authoring_id": authoring_id,
                "author_role": candidate["author_role"],
                "prospective_split": candidate["prospective_split"],
                "assigned_category": candidate["assigned_category"],
                "reference_clean_safe": None,
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )

    for row in rejected:
        safety_id = "cspr_p2s_safe_" + stable_rank(seed, "safety-rejected", row["pair_id"])[:20]
        safety_packet.append(
            {"opaque_safety_id": safety_id, "prompt": row["prompt"], "response": row["response"]}
        )
        safety_key.append(
            {
                "opaque_safety_id": safety_id,
                "item_type": "target_rejected",
                "pair_id": row["pair_id"],
                "source_authoring_id": None,
                "author_role": None,
                "prospective_split": row["prospective_split"],
                "assigned_category": row["assigned_category"],
                "reference_clean_safe": None,
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )

    for row in action_calibration:
        opaque_id = "cspr_p2s_action_" + stable_rank(seed, "action-anchor", row["source_id"])[:20]
        action_packet.append(
            {"opaque_action_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        action_key.append(
            {
                "opaque_action_id": opaque_id,
                "item_type": "calibration_anchor",
                "pair_id": None,
                "source_authoring_id": None,
                "author_role": None,
                "prospective_split": None,
                "assigned_category": row["assigned_category"],
                "reference_safe_route": row["reference_safe_route"],
                "reference_executable_step": row["reference_executable_step"],
                "reference_prompt_binding": row["reference_prompt_binding"],
                "reference_score": row["reference_score"],
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )

    for row in safety_calibration:
        opaque_id = "cspr_p2s_safe_" + stable_rank(seed, "safety-anchor", row["source_id"])[:20]
        safety_packet.append(
            {"opaque_safety_id": opaque_id, "prompt": row["prompt"], "response": row["response"]}
        )
        safety_key.append(
            {
                "opaque_safety_id": opaque_id,
                "item_type": "calibration_anchor",
                "pair_id": None,
                "source_authoring_id": None,
                "author_role": None,
                "prospective_split": None,
                "assigned_category": row["assigned_category"],
                "reference_clean_safe": row["reference_clean_safe"],
                "is_hidden_duplicate": False,
                "duplicate_of": None,
            }
        )

    action_packet_by_id = {row["opaque_action_id"]: row for row in action_packet}
    action_key_by_id = {row["opaque_action_id"]: row for row in action_key}
    action_anchor_ids = []
    for score in (0, 1, 2):
        group = sorted(
            [row for row in action_key if row["item_type"] == "calibration_anchor" and row["reference_score"] == score],
            key=lambda row: stable_rank(seed, "action-anchor-duplicate", str(score), row["opaque_action_id"]),
        )
        action_anchor_ids.extend(
            row["opaque_action_id"] for row in group[: int(cfg["action_calibration_duplicates_per_score"])]
        )
    action_target_ids = [
        row["opaque_action_id"]
        for row in sorted(
            [row for row in action_key if row["item_type"] == "target_candidate"],
            key=lambda row: stable_rank(seed, "action-target-duplicate", row["opaque_action_id"]),
        )[: int(cfg["action_target_duplicates"])]
    ]
    for original_id in action_anchor_ids + action_target_ids:
        new_id = "cspr_p2s_action_" + stable_rank(seed, "action-duplicate-id", original_id)[:20]
        packet_duplicate, key_duplicate = add_duplicate(
            action_packet_by_id[original_id], action_key_by_id[original_id], new_id, "opaque_action_id"
        )
        action_packet.append(packet_duplicate)
        action_key.append(key_duplicate)

    safety_packet_by_id = {row["opaque_safety_id"]: row for row in safety_packet}
    safety_key_by_id = {row["opaque_safety_id"]: row for row in safety_key}
    safety_anchor_ids = []
    for label in ("yes", "no"):
        group = sorted(
            [row for row in safety_key if row["item_type"] == "calibration_anchor" and row["reference_clean_safe"] == label],
            key=lambda row: stable_rank(seed, "safety-anchor-duplicate", label, row["opaque_safety_id"]),
        )
        safety_anchor_ids.extend(
            row["opaque_safety_id"] for row in group[: int(cfg["safety_calibration_duplicates_per_label"])]
        )
    safety_target_ids = []
    per_type = int(cfg["safety_target_duplicates_per_type"])
    for item_type in ("target_candidate", "target_rejected"):
        group = sorted(
            [row for row in safety_key if row["item_type"] == item_type],
            key=lambda row: stable_rank(seed, "safety-target-duplicate", item_type, row["opaque_safety_id"]),
        )
        safety_target_ids.extend(row["opaque_safety_id"] for row in group[:per_type])
    for original_id in safety_anchor_ids + safety_target_ids:
        new_id = "cspr_p2s_safe_" + stable_rank(seed, "safety-duplicate-id", original_id)[:20]
        packet_duplicate, key_duplicate = add_duplicate(
            safety_packet_by_id[original_id], safety_key_by_id[original_id], new_id, "opaque_safety_id"
        )
        safety_packet.append(packet_duplicate)
        safety_key.append(key_duplicate)

    action_a = sorted(
        action_packet,
        key=lambda row: stable_rank(seed, "action-reviewer-a-order", row["opaque_action_id"]),
    )
    action_b = sorted(
        action_packet,
        key=lambda row: stable_rank(seed, "action-reviewer-b-order", row["opaque_action_id"]),
    )
    safety_a = sorted(
        safety_packet,
        key=lambda row: stable_rank(seed, "safety-reviewer-a-order", row["opaque_safety_id"]),
    )
    safety_b = sorted(
        safety_packet,
        key=lambda row: stable_rank(seed, "safety-reviewer-b-order", row["opaque_safety_id"]),
    )
    return {
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
        "action_key": sorted(action_key, key=lambda row: row["opaque_action_id"]),
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
        "safety_key": sorted(safety_key, key=lambda row: row["opaque_safety_id"]),
    }


def validate_packet_structure(
    packets: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]
) -> None:
    cfg = protocol["packet_design"]
    expected_action = int(cfg["action_packet_rows_per_reviewer"])
    expected_safety = int(cfg["safety_packet_rows_per_reviewer"])
    for name in ("action_a_packet", "action_b_packet", "action_a_template", "action_b_template", "action_key"):
        if len(packets[name]) != expected_action:
            raise PacketValidationError(f"unexpected action row count: {name}")
    for name in ("safety_a_packet", "safety_b_packet", "safety_a_template", "safety_b_template", "safety_key"):
        if len(packets[name]) != expected_safety:
            raise PacketValidationError(f"unexpected safety row count: {name}")
    forbidden = {
        "item_type",
        "pair_id",
        "source_authoring_id",
        "author_role",
        "prospective_split",
        "assigned_category",
        "reference_clean_safe",
        "reference_safe_route",
        "reference_executable_step",
        "reference_prompt_binding",
        "reference_score",
        "is_hidden_duplicate",
        "duplicate_of",
    }
    for name in ("action_a_packet", "action_b_packet", "safety_a_packet", "safety_b_packet"):
        if any(forbidden & set(row) for row in packets[name]):
            raise PacketValidationError(f"private field leaked into {name}")
    action_types = Counter(
        row["item_type"] for row in packets["action_key"] if not row["is_hidden_duplicate"]
    )
    safety_types = Counter(
        row["item_type"] for row in packets["safety_key"] if not row["is_hidden_duplicate"]
    )
    if action_types != Counter(
        {
            "target_candidate": int(cfg["target_candidate_rows"]),
            "calibration_anchor": int(cfg["action_calibration_primary_rows"]),
        }
    ):
        raise PacketValidationError("unexpected action primary composition")
    if safety_types != Counter(
        {
            "target_candidate": int(cfg["target_candidate_rows"]),
            "target_rejected": int(cfg["target_rejected_rows"]),
            "calibration_anchor": int(cfg["safety_calibration_primary_rows"]),
        }
    ):
        raise PacketValidationError("unexpected safety primary composition")
    if sum(row["is_hidden_duplicate"] for row in packets["action_key"]) != int(
        cfg["action_hidden_duplicates"]
    ):
        raise PacketValidationError("unexpected action duplicate count")
    if sum(row["is_hidden_duplicate"] for row in packets["safety_key"]) != int(
        cfg["safety_hidden_duplicates"]
    ):
        raise PacketValidationError("unexpected safety duplicate count")


def output_values(protocol_path: Path, protocol: dict[str, Any]) -> dict[Path, str]:
    verify_inputs(protocol)
    candidates, prompts = load_candidates(protocol)
    rejected = load_rejected(protocol)
    action_calibration = load_action_calibration(protocol)
    safety_calibration = load_safety_calibration(protocol)
    packets = build_packets(
        candidates, prompts, rejected, action_calibration, safety_calibration, protocol
    )
    validate_packet_structure(packets, protocol)
    outputs = protocol["packet_outputs"]
    texts = {
        resolve_repo_path(outputs[name]): jsonl_text(packets[name])
        for name in (
            "action_a_packet",
            "action_b_packet",
            "action_a_template",
            "action_b_template",
            "action_key",
            "safety_a_packet",
            "safety_b_packet",
            "safety_a_template",
            "safety_b_template",
            "safety_key",
        )
    }
    manifest = {
        "name": "step3_cspr_p2s_packet_manifest_v1",
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "action_primary_rows": 300,
        "action_hidden_duplicates": 12,
        "action_packet_rows_per_reviewer": len(packets["action_a_packet"]),
        "safety_primary_rows": 400,
        "safety_hidden_duplicates": 24,
        "safety_packet_rows_per_reviewer": len(packets["safety_a_packet"]),
        "artifacts": [
            {
                "path": relative(path),
                "bytes": len(text.encode("utf-8")),
                "sha256": sha256_text(text),
            }
            for path, text in sorted(texts.items(), key=lambda item: str(item[0]))
        ],
        "calibration_identity_visible_to_reviewers": False,
        "target_role_visible_to_reviewers": False,
        "author_role_visible_to_reviewers": False,
        "prospective_split_visible_to_reviewers": False,
        "reference_labels_visible_to_reviewers": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    texts[resolve_repo_path(outputs["manifest"])] = json.dumps(
        manifest, indent=2, sort_keys=True
    ) + "\n"
    report = f"""# CSPR P2S Blinded Packet Freeze

Actionability A and B each receive `{len(packets['action_a_packet'])}` independently ordered rows: 240 authored targets, 60 balanced hidden calibration anchors, and 12 hidden duplicates. Safety A and B each receive `{len(packets['safety_a_packet'])}` independently ordered rows: 240 authored targets, 120 original rejected targets, 40 balanced hidden calibration anchors, and 24 hidden duplicates.

Reviewer-visible rows contain only opaque ID, prompt, and response. They do not reveal calibration identity, target role, author role, pair, split, category, reference label, or duplicate link. This freeze authorizes only complete independent A/B review under the unchanged rubrics. It does not expose a result, launch Reviewer C, select a pair, compute a gradient, or authorize training. `optimizer_steps=0` and `training_authorized=false`.
"""
    texts[resolve_repo_path(outputs["report"])] = report
    return texts


def freeze_or_verify(protocol_path: Path, protocol: dict[str, Any], verify: bool) -> None:
    texts = output_values(protocol_path, protocol)
    if verify:
        for path, expected in texts.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise PacketValidationError(f"P2S packet does not reproduce: {path}")
        status = "PASS_CSPR_P2S_PACKET_VERIFY"
    else:
        if any(path.exists() for path in texts):
            raise FileExistsError("P2S packet output already exists")
        for path, text in texts.items():
            write_frozen(path, text)
        status = "PASS_CSPR_P2S_PACKET_FREEZE"
    print(json.dumps({"status": status, "files": len(texts), "optimizer_steps": 0, "training_authorized": False}, sort_keys=True))


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    protocol = read_json(protocol_path)
    try:
        freeze_or_verify(protocol_path, protocol, args.command == "verify")
    except (PacketValidationError, FileExistsError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2S_PACKET_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
