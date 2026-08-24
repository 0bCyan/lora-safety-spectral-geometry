"""Prepare M8-A attempt-002 with a permission-checked transport boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import prepare_step3_cdro_opc_m8_authoring as base


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m8/protocol/"
    "cdro_opc_m8_authoring_attempt_002_protocol.json"
)
SCHEMA = "step3_cdro_opc_m8_authoring_attempt_002_protocol_v1"
STATUS = "prospective_locked_after_attempt_001_transport_stop_before_attempt_002_packet_materialization"


class M8AuthoringAttempt002Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M8AuthoringAttempt002Error(f"expected JSON object: {path}")
    return value


def frozen_path(spec: Mapping[str, Any], name: str) -> Path:
    scope = spec.get("scope", "repository")
    if scope == "repository":
        path = base.repo_path(str(spec.get("path", "")))
    elif scope == "absolute":
        path = Path(str(spec.get("path", ""))).resolve()
    else:
        raise M8AuthoringAttempt002Error(f"invalid frozen input scope: {name}")
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        raise M8AuthoringAttempt002Error(f"frozen input mismatch: {name}")
    return path


def validate_attempt_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise M8AuthoringAttempt002Error("attempt-002 protocol identity differs")
    if protocol.get("transport_only_change") != {
        "attempt_001_responses_reused": False,
        "authoring_rules_changed": False,
        "blind_review_or_gpu_rules_changed": False,
        "prompt_registry_changed": False,
        "spool_permission_boundary_changed": True,
    }:
        raise M8AuthoringAttempt002Error("attempt-002 change boundary differs")
    authoring = protocol.get("authoring", {})
    if authoring != {
        "attempt": "attempt-002",
        "fork_turns": "none",
        "immutable_submission": True,
        "modes": ["specific_redirect", "risk_mitigation"],
        "maximum_response_whitespace_tokens": 160,
        "minimum_response_whitespace_tokens": 40,
        "prompt_replacement_or_top_up_allowed": False,
        "required_openings": {
            "risk_mitigation": "I won't assist",
            "specific_redirect": "I can't help",
        },
        "responses_schema": [
            "request_id",
            "opaque_prompt_id",
            "sample_id",
            "author_mode",
            "response",
        ],
        "retry_or_reassignment_allowed": False,
        "rows_per_shard": 10,
        "shards": 12,
        "single_line_response": True,
        "source": "current_model_no_context_subagents",
        "spool_root": (
            "C:/Users/1/Documents/Codex/2026-08-23/"
            "apiidparam-input-192-id-invalid-id/work/"
            "step3-cdro-opc-m8-authoring-attempt-002"
        ),
    }:
        raise M8AuthoringAttempt002Error("attempt-002 authoring contract differs")
    if protocol.get("authorization_boundary") != {
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }:
        raise M8AuthoringAttempt002Error("attempt-002 authorization boundary differs")


def load_context(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    protocol = read_json(protocol_path)
    validate_attempt_protocol(protocol)
    required_inputs = {
        "master_protocol",
        "attempt_001_stop",
        "permission_probe",
        "base_preparer",
        "attempt_002_preparer",
        "attempt_002_finalizer",
        "attempt_002_tests",
    }
    inputs = protocol.get("frozen_inputs", {})
    if set(inputs) != required_inputs:
        raise M8AuthoringAttempt002Error("attempt-002 frozen input registry differs")
    paths = {name: frozen_path(spec, name) for name, spec in inputs.items()}
    if paths["permission_probe"].read_bytes() != b"M8A2_WRITE_OK\n":
        raise M8AuthoringAttempt002Error("permission probe bytes differ")
    stop = read_json(paths["attempt_001_stop"])
    if (
        stop.get("route") != "STOP_CDRO_OPC_M8_AUTHORING"
        or stop.get("committed_response_rows") != 0
        or stop.get("attempt_001_responses_reused_by_future_attempt") is not False
    ):
        raise M8AuthoringAttempt002Error("attempt-001 stop boundary differs")
    master = base.read_json(paths["master_protocol"])
    base.validate_protocol(master, allow_pending=False)
    registry, counts = base.build_registry(master)
    registry_hash = base.canonical_sha256(registry)
    if (
        registry_hash != master.get("prompt_registry_sha256")
        or registry_hash != protocol.get("prompt_registry_sha256")
        or counts
        != {
            "risk_rows": 1000,
            "excluded_historical_samples": 120,
            "eligible_distinct_prompt_rows": 880,
            "selected_prompts": 60,
            "requests": 120,
        }
    ):
        raise M8AuthoringAttempt002Error("attempt-002 frozen registry differs")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(protocol["entry_commit"]), "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise M8AuthoringAttempt002Error("attempt-002 entry commit is not an ancestor")
    return protocol, master, registry, counts


def prepare(protocol_path: Path) -> dict[str, Any]:
    protocol, master, registry, counts = load_context(protocol_path)
    spool = Path(protocol["authoring"]["spool_root"]).resolve()
    if spool.exists():
        raise M8AuthoringAttempt002Error(f"attempt-002 spool already exists: {spool}")
    packets = spool / "packets"
    packets.mkdir(parents=True)
    packet_hashes: dict[str, str] = {}
    for shard_index in range(int(protocol["authoring"]["shards"])):
        rows = base.packet_rows(master, registry, shard_index)
        if len(rows) != int(protocol["authoring"]["rows_per_shard"]):
            raise M8AuthoringAttempt002Error(f"shard denominator differs: {shard_index:02d}")
        path = packets / f"shard_{shard_index:02d}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        packet_hashes[path.name] = sha256_file(path)
    registry_path = spool / "prompt_registry.json"
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "step3_cdro_opc_m8_authoring_attempt_002_transport_manifest_v1",
        "attempt": "attempt-002",
        "protocol_sha256": sha256_file(protocol_path),
        "prompt_registry_canonical_sha256": base.canonical_sha256(registry),
        "prompt_registry_file_sha256": sha256_file(registry_path),
        "packet_sha256": packet_hashes,
        "responses_committed": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    (spool / "transport_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "PASS_CDRO_OPC_M8_AUTHORING_ATTEMPT_002_PACKETS_PREPARED",
        **counts,
        "spool_root": str(spool),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = base.repo_path(args.protocol)
    if args.command == "prepare":
        prepare(protocol_path)
        return
    _, _, _, counts = load_context(protocol_path)
    print(
        json.dumps(
            {
                "status": "PASS_CDRO_OPC_M8_AUTHORING_ATTEMPT_002_PREFLIGHT",
                **counts,
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
