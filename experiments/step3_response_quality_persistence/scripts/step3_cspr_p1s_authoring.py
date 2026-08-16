"""Freeze and validate blinded CSPR P1S authored-candidate source repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p1s/"
    "protocol/cspr_p1s_authored_candidate_repair_protocol.json"
)
MODES = ("specific_redirect", "risk_mitigation")


class AuthoringValidationError(ValueError):
    """Raised when a frozen P1S authoring artifact is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "verify", "validate-authors"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise AuthoringValidationError(f"path escapes repository: {value}") from exc
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


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def stable_rank(seed: int, *parts: str) -> str:
    return sha256_text("|".join([str(seed), *parts]))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise AuthoringValidationError(f"JSONL row is not an object at {path}:{number}")
        rows.append(row)
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


def verify_inputs(protocol: dict[str, Any]) -> None:
    for name, artifact in protocol["frozen_inputs"].items():
        path = resolve_repo_path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise AuthoringValidationError(f"frozen input hash mismatch: {name}")


def source_prompt_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = protocol["frozen_inputs"]
    packet = read_jsonl(resolve_repo_path(inputs["p2_action_packet"]["path"]))
    key_rows = read_jsonl(resolve_repo_path(inputs["p2_action_hidden_key"]["path"]))
    packet_by_id = {str(row["opaque_action_id"]): row for row in packet}
    if len(packet_by_id) != len(packet):
        raise AuthoringValidationError("duplicate IDs in source action packet")
    source = []
    for key in key_rows:
        if key["is_hidden_duplicate"]:
            continue
        opaque_id = str(key["opaque_action_id"])
        packet_row = packet_by_id[opaque_id]
        prompt = packet_row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise AuthoringValidationError(f"blank source prompt: {opaque_id}")
        if sha256_text(normalize_text(prompt)) != key["prompt_normalized_sha256"]:
            raise AuthoringValidationError(f"source prompt hash mismatch: {opaque_id}")
        source.append(
            {
                "pair_id": key["pair_id"],
                "source_action_id": opaque_id,
                "prompt": prompt,
                "prompt_normalized_sha256": key["prompt_normalized_sha256"],
                "prospective_split": key["prospective_split"],
                "assigned_category": key["assigned_category"],
            }
        )
    if len(source) != int(protocol["design"]["prompt_rows"]):
        raise AuthoringValidationError("unexpected primary prompt count")
    if len({row["pair_id"] for row in source}) != len(source):
        raise AuthoringValidationError("source pair IDs are not unique")
    if len({row["prompt_normalized_sha256"] for row in source}) != len(source):
        raise AuthoringValidationError("source prompts are not unique")
    return source


def build_author_packets(
    source: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    seed = int(protocol["design"]["seed"])
    packets: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    keys = []
    for row in source:
        mode_index = int(stable_rank(seed, "mode", row["pair_id"]), 16) % 2
        for role in ("a", "b"):
            role_index = 0 if role == "a" else 1
            mode = MODES[(mode_index + role_index) % 2]
            opaque_id = "cspr_p1s_author_" + stable_rank(
                seed, "authoring-id", role, row["pair_id"]
            )[:20]
            packets[role].append(
                {
                    "opaque_authoring_id": opaque_id,
                    "prompt": row["prompt"],
                    "authoring_mode": mode,
                }
            )
            keys.append(
                {
                    "opaque_authoring_id": opaque_id,
                    "author_role": role,
                    "authoring_mode": mode,
                    "pair_id": row["pair_id"],
                    "source_action_id": row["source_action_id"],
                    "prompt_normalized_sha256": row["prompt_normalized_sha256"],
                    "prospective_split": row["prospective_split"],
                    "assigned_category": row["assigned_category"],
                }
            )
    for role in ("a", "b"):
        packets[role] = sorted(
            packets[role],
            key=lambda row: stable_rank(seed, f"author-{role}-order", row["opaque_authoring_id"]),
        )
    return {
        "author_a_packet": packets["a"],
        "author_b_packet": packets["b"],
        "author_a_template": [
            {"opaque_authoring_id": row["opaque_authoring_id"], "response": ""}
            for row in packets["a"]
        ],
        "author_b_template": [
            {"opaque_authoring_id": row["opaque_authoring_id"], "response": ""}
            for row in packets["b"]
        ],
        "hidden_key": sorted(keys, key=lambda row: row["opaque_authoring_id"]),
    }


def validate_packet_structure(
    packets: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]
) -> None:
    expected = int(protocol["design"]["prompt_rows"])
    for name, rows in packets.items():
        multiplier = 2 if name == "hidden_key" else 1
        if len(rows) != expected * multiplier:
            raise AuthoringValidationError(f"unexpected authoring row count: {name}")
    forbidden = {
        "pair_id",
        "source_action_id",
        "prospective_split",
        "assigned_category",
        "author_role",
        "source_label",
        "old_response",
        "rejected_response",
    }
    for name in ("author_a_packet", "author_b_packet"):
        if any(forbidden & set(row) for row in packets[name]):
            raise AuthoringValidationError(f"private field leaked into {name}")
    key_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packets["hidden_key"]:
        key_by_pair[row["pair_id"]].append(row)
    if any(len(rows) != 2 for rows in key_by_pair.values()):
        raise AuthoringValidationError("each prompt must have two authoring assignments")
    for rows in key_by_pair.values():
        if {row["author_role"] for row in rows} != {"a", "b"}:
            raise AuthoringValidationError("author roles are not paired")
        if {row["authoring_mode"] for row in rows} != set(MODES):
            raise AuthoringValidationError("authoring modes are not complementary")


def packet_output_values(
    protocol_path: Path, protocol: dict[str, Any]
) -> dict[Path, str]:
    verify_inputs(protocol)
    source = source_prompt_rows(protocol)
    packets = build_author_packets(source, protocol)
    validate_packet_structure(packets, protocol)
    outputs = protocol["packet_outputs"]
    texts = {
        resolve_repo_path(outputs[name]): jsonl_text(packets[name])
        for name in (
            "author_a_packet",
            "author_b_packet",
            "author_a_template",
            "author_b_template",
            "hidden_key",
        )
    }
    manifest = {
        "name": "step3_cspr_p1s_authoring_packet_manifest_v1",
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "prompt_rows": len(source),
        "author_rows": 2 * len(source),
        "artifacts": [
            {
                "path": relative(path),
                "bytes": len(text.encode("utf-8")),
                "sha256": sha256_text(text),
            }
            for path, text in sorted(texts.items(), key=lambda item: str(item[0]))
        ],
        "old_candidate_visible_to_authors": False,
        "rejected_response_visible_to_authors": False,
        "prospective_split_visible_to_authors": False,
        "source_label_visible_to_authors": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    texts[resolve_repo_path(outputs["manifest"])] = json.dumps(
        manifest, indent=2, sort_keys=True
    ) + "\n"
    report = f"""# CSPR P1S Authoring Packet Freeze

Two independent author roles each receive `{len(source)}` prompts in distinct deterministic orders. Every prompt receives complementary `specific_redirect` and `risk_mitigation` assignments. Reviewer-visible authoring packets expose only opaque ID, prompt, and mode; old candidates, rejected responses, pair IDs, source labels, categories, and prospective splits remain hidden.

This packet freeze authorizes one complete output per author role. It does not label, rank, repair, retry, top up, or select a candidate, and it does not authorize review or training. `optimizer_steps=0` and `training_authorized=false`.
"""
    texts[resolve_repo_path(outputs["report"])] = report
    return texts


def freeze_or_verify(protocol_path: Path, protocol: dict[str, Any], verify: bool) -> dict[str, Any]:
    texts = packet_output_values(protocol_path, protocol)
    if verify:
        for path, expected in texts.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise AuthoringValidationError(f"authoring packet does not reproduce: {path}")
        status = "PASS_CSPR_P1S_AUTHORING_PACKET_VERIFY"
    else:
        if any(path.exists() for path in texts):
            raise FileExistsError("authoring packet output already exists")
        for path, text in texts.items():
            write_frozen(path, text)
        status = "PASS_CSPR_P1S_AUTHORING_PACKET_FREEZE"
    result = {"status": status, "files": len(texts), "optimizer_steps": 0, "training_authorized": False}
    print(json.dumps(result, sort_keys=True))
    return result


def validate_provenance(
    provenance: dict[str, Any],
    role: str,
    packet_path: Path,
    template_path: Path,
    output_path: Path,
    rubric_path: Path,
    row_count: int,
) -> None:
    expected = {
        "author_role": role,
        "packet_path": relative(packet_path),
        "packet_sha256": sha256(packet_path),
        "template_path": relative(template_path),
        "template_sha256": sha256(template_path),
        "instructions_path": relative(rubric_path),
        "instructions_sha256": sha256(rubric_path),
        "output_path": relative(output_path),
        "output_sha256": sha256(output_path),
        "row_count": row_count,
        "fresh_context": True,
        "forbidden_material_not_opened": True,
        "completion_status": "complete",
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise AuthoringValidationError(f"author {role} provenance mismatch: {field}")
    for field in ("agent_id", "session_id", "runtime"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise AuthoringValidationError(f"author {role} missing {field}")


def load_author_output(
    role: str, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_path = resolve_repo_path(protocol["packet_outputs"][f"author_{role}_packet"])
    template_path = resolve_repo_path(protocol["packet_outputs"][f"author_{role}_template"])
    output_path = resolve_repo_path(protocol["author_outputs"][f"author_{role}_responses"])
    provenance_path = resolve_repo_path(protocol["author_outputs"][f"author_{role}_provenance"])
    instructions_path = resolve_repo_path(protocol["frozen_inputs"]["authoring_instructions"]["path"])
    template = read_jsonl(template_path)
    output = read_jsonl(output_path)
    if len(output) != len(template):
        raise AuthoringValidationError(f"author {role} output is incomplete")
    seen = set()
    for index, (row, blank) in enumerate(zip(output, template, strict=True), start=1):
        if set(row) != {"opaque_authoring_id", "response"}:
            raise AuthoringValidationError(f"author {role} output schema mismatch at row {index}")
        opaque_id = row["opaque_authoring_id"]
        if opaque_id != blank["opaque_authoring_id"] or opaque_id in seen:
            raise AuthoringValidationError(f"author {role} ID/order mismatch at row {index}")
        if not isinstance(row["response"], str) or not row["response"].strip():
            raise AuthoringValidationError(f"author {role} blank response at row {index}")
        seen.add(opaque_id)
    provenance = read_json(provenance_path)
    validate_provenance(
        provenance,
        role,
        packet_path,
        template_path,
        output_path,
        instructions_path,
        len(output),
    )
    return output, provenance


def prior_response_hashes(protocol: dict[str, Any]) -> set[str]:
    packet = read_jsonl(
        resolve_repo_path(protocol["frozen_inputs"]["p2_safety_packet"]["path"])
    )
    return {
        sha256_text(normalize_text(str(row["response"])))
        for row in packet
        if isinstance(row.get("response"), str) and row["response"].strip()
    }


def candidate_ledger(
    authors: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_rows = read_jsonl(resolve_repo_path(protocol["packet_outputs"]["hidden_key"]))
    key_by_id = {str(row["opaque_authoring_id"]): row for row in key_rows}
    prior_hashes = prior_response_hashes(protocol)
    mechanical = protocol["mechanical_validation"]
    minimum_words = int(mechanical["minimum_words"])
    maximum_words = int(mechanical["maximum_words"])
    seed = int(protocol["design"]["seed"])
    raw_rows = []
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for role, rows in authors.items():
        for row in rows:
            opaque_id = str(row["opaque_authoring_id"])
            response = row["response"].strip()
            normalized_hash = sha256_text(normalize_text(response))
            hash_groups[normalized_hash].append(opaque_id)
            word_count = len(response.split())
            reasons = []
            if not minimum_words <= word_count <= maximum_words:
                reasons.append("word_count_out_of_range")
            if "<think>" in response.casefold() or "</think>" in response.casefold():
                reasons.append("thinking_tag_present")
            if normalized_hash in prior_hashes:
                reasons.append("normalized_exact_prior_response_match")
            raw_rows.append(
                {
                    **key_by_id[opaque_id],
                    "response": response,
                    "response_sha256": sha256_text(response),
                    "response_normalized_sha256": normalized_hash,
                    "word_count": word_count,
                    "mechanical_exclusion_reasons": reasons,
                    "duplicate_of_authoring_id": None,
                }
            )
    representative: dict[str, str] = {}
    for normalized_hash, ids in hash_groups.items():
        representative[normalized_hash] = min(
            ids, key=lambda opaque_id: stable_rank(seed, "dedup-representative", opaque_id)
        )
    for row in raw_rows:
        group = hash_groups[row["response_normalized_sha256"]]
        if len(group) > 1 and row["opaque_authoring_id"] != representative[row["response_normalized_sha256"]]:
            row["mechanical_exclusion_reasons"].append("normalized_exact_authored_duplicate")
            row["duplicate_of_authoring_id"] = representative[row["response_normalized_sha256"]]
        row["mechanically_valid"] = not row["mechanical_exclusion_reasons"]

    split_prompts: dict[str, set[str]] = defaultdict(set)
    category_prompts: dict[str, set[str]] = defaultdict(set)
    valid_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row["mechanically_valid"]:
            valid_by_pair[row["pair_id"]].append(row)
            split_prompts[row["prospective_split"]].add(row["pair_id"])
            category_prompts[row["assigned_category"]].add(row["pair_id"])
    gates_cfg = protocol["review_supply_gates"]
    categories_with_two = sum(len(pair_ids) >= 2 for pair_ids in category_prompts.values())
    gates = {
        "train_candidate_prompt_supply_pass": len(split_prompts["train_candidate"]) >= int(gates_cfg["minimum_train_candidate_prompts"]),
        "calibration_prompt_supply_pass": len(split_prompts["calibration"]) >= int(gates_cfg["minimum_calibration_prompts"]),
        "mechanism_dev_prompt_supply_pass": len(split_prompts["mechanism_dev"]) >= int(gates_cfg["minimum_mechanism_dev_prompts"]),
        "total_prompt_supply_pass": len(valid_by_pair) >= int(gates_cfg["minimum_total_prompts"]),
        "category_supply_pass": categories_with_two >= int(gates_cfg["minimum_categories_with_at_least_two_prompts"]),
    }
    counts = Counter(
        reason for row in raw_rows for reason in row["mechanical_exclusion_reasons"]
    )
    analysis = {
        "author_rows": len(raw_rows),
        "mechanically_valid_candidates": sum(row["mechanically_valid"] for row in raw_rows),
        "mechanically_excluded_candidates": sum(not row["mechanically_valid"] for row in raw_rows),
        "mechanical_exclusion_reason_counts": dict(sorted(counts.items())),
        "prompts_with_at_least_one_valid_candidate": len(valid_by_pair),
        "valid_prompt_split_counts": {
            split: len(pair_ids) for split, pair_ids in sorted(split_prompts.items())
        },
        "valid_prompt_category_counts": {
            category: len(pair_ids) for category, pair_ids in sorted(category_prompts.items())
        },
        "categories_with_at_least_two_valid_prompts": categories_with_two,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    return sorted(raw_rows, key=lambda row: row["opaque_authoring_id"]), analysis


def validate_authors(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    verify_inputs(protocol)
    authors = {}
    provenances = []
    for role in ("a", "b"):
        authors[role], provenance = load_author_output(role, protocol)
        provenances.append(provenance)
    if len({row["agent_id"] for row in provenances}) != 2 or len(
        {row["session_id"] for row in provenances}
    ) != 2:
        raise AuthoringValidationError("author agent/session reuse is forbidden")
    ledger, analysis = candidate_ledger(authors, protocol)
    route = (
        "AUTHORIZE_CSPR_P2S_REVIEW_PACKET_PROTOCOL_ONLY"
        if analysis["overall_pass"]
        else "STOP_CSPR_P1S_AUTHORED_SOURCE_YIELD"
    )
    outputs = protocol["validation_outputs"]
    ledger_path = resolve_repo_path(outputs["candidate_ledger"])
    write_jsonl_frozen(ledger_path, ledger)
    result = {
        "name": "step3_cspr_p1s_authoring_validation_v1",
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "route": route,
        "analysis": analysis,
        "candidate_ledger": {
            "path": relative(ledger_path),
            "rows": len(ledger),
            "sha256": sha256(ledger_path),
        },
        "author_provenance": [
            {
                "author_role": row["author_role"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "output_sha256": row["output_sha256"],
            }
            for row in provenances
        ],
        "semantic_or_safety_labels_assigned": False,
        "review_authorized": analysis["overall_pass"],
        "model_gradient_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json_frozen(resolve_repo_path(outputs["validation"]), result)
    report = f"""# CSPR P1S Authoring Validation

Route: `{route}`.

The two frozen author outputs contain `{analysis['author_rows']}` rows. Mechanical validation retained `{analysis['mechanically_valid_candidates']}` candidates covering `{analysis['prompts_with_at_least_one_valid_candidate']}` prompts. Exclusions were `{json.dumps(analysis['mechanical_exclusion_reason_counts'], sort_keys=True)}`. Review-supply gates passed: `{analysis['overall_pass']}`.

Mechanical validation checks only completeness, word count, thinking tags, and normalized exact duplication against prior or authored responses. It does not assign clean-safety or actionability labels. Passing authoring can authorize only a separately frozen P2S review-packet protocol. `optimizer_steps=0` and `training_authorized=false`.
"""
    write_frozen(resolve_repo_path(outputs["report"]), report)
    print(json.dumps({"route": route, "valid_candidates": analysis["mechanically_valid_candidates"], "valid_prompts": analysis["prompts_with_at_least_one_valid_candidate"]}, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    protocol_path = resolve_repo_path(args.protocol)
    protocol = read_json(protocol_path)
    try:
        if args.command == "freeze":
            freeze_or_verify(protocol_path, protocol, False)
        elif args.command == "verify":
            freeze_or_verify(protocol_path, protocol, True)
        else:
            validate_authors(protocol_path, protocol)
    except (AuthoringValidationError, FileExistsError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "STOP_CSPR_P1S_AUTHORING_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
