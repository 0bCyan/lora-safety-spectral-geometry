"""Validate and seal the immutable M8-A attempt-002 submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import prepare_step3_cdro_opc_m8_authoring as base
import prepare_step3_cdro_opc_m8_authoring_attempt_002 as attempt2


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = attempt2.DEFAULT_PROTOCOL
OUTPUT = (
    REPO
    / "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m8/authoring"
)


class M8AuthoringFinalizationError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise M8AuthoringFinalizationError(f"expected JSONL objects: {path}")
    return rows


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def validate_response_row(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    authoring: Mapping[str, Any],
    normalized_hashes: set[str],
) -> dict[str, Any]:
    expected_keys = set(authoring["responses_schema"])
    if set(actual) != expected_keys:
        raise M8AuthoringFinalizationError("response schema differs")
    for key in ("request_id", "opaque_prompt_id", "sample_id", "author_mode"):
        if actual.get(key) != expected.get(key):
            raise M8AuthoringFinalizationError(f"response metadata differs: {key}")
    response = actual.get("response")
    if not isinstance(response, str) or not response or response != response.strip():
        raise M8AuthoringFinalizationError("response text is empty or not stripped")
    if authoring["single_line_response"] and ("\n" in response or "\r" in response):
        raise M8AuthoringFinalizationError("response is not single-line")
    tokens = len(response.split())
    if not (
        int(authoring["minimum_response_whitespace_tokens"])
        <= tokens
        <= int(authoring["maximum_response_whitespace_tokens"])
    ):
        raise M8AuthoringFinalizationError("response token envelope differs")
    opening = authoring["required_openings"][str(actual["author_mode"])]
    if not response.startswith(opening):
        raise M8AuthoringFinalizationError("response opening differs")
    if "<think" in response.casefold() or "</think" in response.casefold():
        raise M8AuthoringFinalizationError("response contains thinking tag")
    response_sha256 = hashlib.sha256(response.encode("utf-8")).hexdigest()
    normalized_sha256 = hashlib.sha256(
        base.normalize_prompt(response).encode("utf-8")
    ).hexdigest()
    if normalized_sha256 in normalized_hashes:
        raise M8AuthoringFinalizationError("global normalized response duplicate")
    normalized_hashes.add(normalized_sha256)
    return {
        "response": response,
        "response_whitespace_tokens": tokens,
        "response_sha256": response_sha256,
        "normalized_response_sha256": normalized_sha256,
    }


def validate_external_spool(
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    protocol, master, registry, _ = attempt2.load_context(protocol_path)
    spool = Path(protocol["authoring"]["spool_root"]).resolve()
    transport_path = spool / "transport_manifest.json"
    registry_path = spool / "prompt_registry.json"
    if not transport_path.is_file() or not registry_path.is_file():
        raise M8AuthoringFinalizationError("attempt-002 base spool is incomplete")
    transport = attempt2.read_json(transport_path)
    if (
        transport.get("schema")
        != "step3_cdro_opc_m8_authoring_attempt_002_transport_manifest_v1"
        or transport.get("protocol_sha256") != attempt2.sha256_file(protocol_path)
        or transport.get("prompt_registry_canonical_sha256")
        != base.canonical_sha256(registry)
        or transport.get("prompt_registry_file_sha256")
        != attempt2.sha256_file(registry_path)
        or transport.get("responses_committed") != 0
    ):
        raise M8AuthoringFinalizationError("attempt-002 transport manifest differs")
    if attempt2.read_json(registry_path) != registry:
        raise M8AuthoringFinalizationError("attempt-002 prompt registry bytes differ")

    candidates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    normalized_hashes: set[str] = set()
    external_hashes = {
        "prompt_registry.json": attempt2.sha256_file(registry_path),
        "transport_manifest.json": attempt2.sha256_file(transport_path),
    }
    agent_ids: set[str] = set()
    for shard_index in range(int(protocol["authoring"]["shards"])):
        shard = f"{shard_index:02d}"
        packet_path = spool / "packets" / f"shard_{shard}.jsonl"
        launch_path = spool / "launches" / f"shard_{shard}.json"
        response_path = spool / "responses" / f"shard_{shard}.jsonl"
        if not packet_path.is_file() or not launch_path.is_file() or not response_path.is_file():
            raise M8AuthoringFinalizationError(f"attempt-002 shard incomplete: {shard}")
        packet = read_jsonl(packet_path)
        deterministic = base.packet_rows(master, registry, shard_index)
        responses = read_jsonl(response_path)
        launch = attempt2.read_json(launch_path)
        expected_launch = {
            "shard": shard,
            "agent_id": launch.get("agent_id"),
            "fork_turns": "none",
            "packet_path": str(packet_path),
            "response_path": str(response_path),
        }
        agent_id = launch.get("agent_id")
        if (
            packet != deterministic
            or len(packet) != int(protocol["authoring"]["rows_per_shard"])
            or len(responses) != len(packet)
            or launch != expected_launch
            or not isinstance(agent_id, str)
            or not agent_id.startswith("/root/")
            or agent_id in agent_ids
        ):
            raise M8AuthoringFinalizationError(f"attempt-002 shard provenance differs: {shard}")
        agent_ids.add(agent_id)
        if transport["packet_sha256"].get(packet_path.name) != attempt2.sha256_file(
            packet_path
        ):
            raise M8AuthoringFinalizationError(f"attempt-002 packet hash differs: {shard}")
        external_hashes[f"packets/{packet_path.name}"] = attempt2.sha256_file(packet_path)
        external_hashes[f"launches/{launch_path.name}"] = attempt2.sha256_file(launch_path)
        external_hashes[f"responses/{response_path.name}"] = attempt2.sha256_file(response_path)
        for expected, actual in zip(packet, responses, strict=True):
            validated = validate_response_row(
                expected, actual, protocol["authoring"], normalized_hashes
            )
            request_id = str(actual["request_id"])
            candidate_id = (
                "m8_candidate_"
                + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:24]
            )
            public = {
                "request_id": request_id,
                "opaque_candidate_id": candidate_id,
                "opaque_prompt_id": actual["opaque_prompt_id"],
                "sample_id": actual["sample_id"],
                "author_mode": actual["author_mode"],
                **{key: value for key, value in validated.items() if key != "response"},
                "authoring_shard": shard,
                "authoring_agent_id": agent_id,
            }
            records.append(public)
            candidates.append(
                {
                    **public,
                    "prompt": expected["prompt"],
                    "response": validated["response"],
                    "authoring_source": "current_model_no_context_subagents",
                    "model_provenance": "inherited_parent_model_unversioned",
                }
            )
    if (
        len(candidates) != 120
        or len({row["opaque_prompt_id"] for row in candidates}) != 60
        or len(normalized_hashes) != 120
        or len(agent_ids) != 12
    ):
        raise M8AuthoringFinalizationError("attempt-002 final denominator differs")
    return candidates, records, external_hashes


def finalize(protocol_path: Path) -> dict[str, Any]:
    if OUTPUT.exists():
        raise M8AuthoringFinalizationError("M8 authoring output already exists")
    candidates, records, external_hashes = validate_external_spool(protocol_path)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temporary.exists():
        raise M8AuthoringFinalizationError("M8 temporary authoring output already exists")
    try:
        private_path = temporary / "private" / "primary_candidates.jsonl"
        private_path.parent.mkdir(parents=True)
        private_path.write_bytes(jsonl_bytes(candidates))
        (temporary / "record_manifest.jsonl").write_bytes(jsonl_bytes(records))
        summary = {
            "schema": "step3_cdro_opc_m8_authoring_summary_v1",
            "route": "PASS_CDRO_OPC_M8_AUTHORING",
            "attempt": "attempt-002",
            "fixed_requests": 120,
            "valid_candidates": 120,
            "valid_prompts": 60,
            "unique_normalized_responses": 120,
            "subagent_sessions": 12,
            "context_isolated": True,
            "attempt_001_responses_reused": False,
            "retry_or_reassignment_used": False,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        (temporary / "authoring_summary.json").write_bytes(json_bytes(summary))
        (temporary / "CDRO_OPC_M8_AUTHORING_REPORT.md").write_text(
            "# CDR-OPC M8 Independent Authoring\n\n"
            "M8-A attempt-002 passed the frozen prompt-disjoint authoring gate. "
            "Twelve no-context agents completed 120 fixed requests over 60 prompts "
            "through the permission-checked transport boundary. Attempt-001 "
            "responses were not recovered or reused. The main session did not author, "
            "label, revise, or replace candidate text.\n\n"
            "route=PASS_CDRO_OPC_M8_AUTHORING\n"
            "optimizer_steps=0\ntraining_authorized=false\n",
            encoding="utf-8",
        )
        internal_hashes = {
            path.relative_to(temporary).as_posix(): attempt2.sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": "step3_cdro_opc_m8_authoring_manifest_v1",
            "route": summary["route"],
            "attempt": "attempt-002",
            "protocol_sha256": attempt2.sha256_file(protocol_path),
            "external_spool_sha256": external_hashes,
            "artifacts": internal_hashes,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        temporary.replace(OUTPUT)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    finalize(base.repo_path(args.protocol))


if __name__ == "__main__":
    main()
