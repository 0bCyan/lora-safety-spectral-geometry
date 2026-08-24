"""Prepare the prospectively frozen CDR-OPC M8 authoring packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m8/protocol/cdro_opc_m8_master_protocol.json"
)
SCHEMA = "step3_cdro_opc_m8_master_protocol_v1"
STATUS = "prospective_locked_before_m8_prompt_packet_materialization"
REGISTRY_SCHEMA = "step3_cdro_opc_m8_prompt_registry_v1"
PENDING_HASH = "PENDING_DERIVATION_BEFORE_COMMIT"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class M8AuthoringError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def normalize_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M8AuthoringError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise M8AuthoringError(f"expected JSONL objects: {path}")
    return rows


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M8AuthoringError(f"path escapes repository: {value}") from exc
    return path


def verify_frozen(spec: Mapping[str, Any], name: str) -> Path:
    path = repo_path(str(spec.get("path", "")))
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        raise M8AuthoringError(f"frozen input mismatch: {name}")
    return path


def validate_protocol(protocol: Mapping[str, Any], *, allow_pending: bool) -> None:
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise M8AuthoringError("M8 protocol identity differs")
    boundary = protocol.get("authorization_boundary", {})
    if boundary != {
        "candidate_generation_authorized": False,
        "checkpoint_writes_authorized": False,
        "core100_opened": False,
        "evaluated_model_generation_authorized": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "result_dependent_rerun_authorized": False,
        "training_authorized": False,
        "transient_virtual_parameter_evaluations_authorized": False,
    }:
        raise M8AuthoringError("M8 authorization boundary differs")
    authoring = protocol.get("authoring", {})
    selection = protocol.get("selection", {})
    if any(
        (
            selection.get("prompts") != 60,
            selection.get("historical_unique_samples_expected") != 120,
            selection.get("risk_rows_expected") != 1000,
            authoring.get("requests") != 120,
            authoring.get("shards") != 12,
            authoring.get("rows_per_shard") != 10,
            authoring.get("modes") != ["specific_redirect", "risk_mitigation"],
            authoring.get("minimum_response_whitespace_tokens") != 40,
            authoring.get("maximum_response_whitespace_tokens") != 160,
            authoring.get("required_openings")
            != {
                "specific_redirect": "I can't help",
                "risk_mitigation": "I won't assist",
            },
            authoring.get("immutable_submission") is not True,
            authoring.get("retry_or_reassignment_allowed") is not False,
            authoring.get("prompt_replacement_or_top_up_allowed") is not False,
        )
    ):
        raise M8AuthoringError("M8 selection or authoring denominator differs")
    registry_hash = protocol.get("prompt_registry_sha256")
    if allow_pending and registry_hash == PENDING_HASH:
        return
    if not isinstance(registry_hash, str) or HASH_RE.fullmatch(registry_hash) is None:
        raise M8AuthoringError("M8 prompt registry hash is not locked")


def build_registry(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    inputs = protocol.get("frozen_inputs", {})
    risk_path = verify_frozen(inputs.get("risk_source", {}), "risk_source")
    historical_specs = inputs.get("historical_primary_candidates", {})
    if set(historical_specs) != {"m3", "m5"}:
        raise M8AuthoringError("historical candidate registry differs")
    historical = {
        name: read_jsonl(verify_frozen(spec, f"historical_primary_candidates.{name}"))
        for name, spec in historical_specs.items()
    }
    verify_frozen(inputs.get("authoring_instructions", {}), "authoring_instructions")
    verify_frozen(inputs.get("m6_protocol", {}), "m6_protocol")
    verify_frozen(inputs.get("m6_decision", {}), "m6_decision")
    for name in ("authoring_preparer", "authoring_preparer_tests", "protocol_note"):
        verify_frozen(inputs.get(name, {}), name)

    risk_rows = read_jsonl(risk_path)
    risk_by_sample = {str(row.get("sample_id")): row for row in risk_rows}
    if len(risk_rows) != protocol["selection"]["risk_rows_expected"] or len(
        risk_by_sample
    ) != len(risk_rows):
        raise M8AuthoringError("risk source denominator or sample uniqueness differs")

    source_counts: dict[str, int] = {}
    source_unique_samples: dict[str, int] = {}
    excluded_samples: set[str] = set()
    excluded_prompt_hashes: set[str] = set()
    for name, rows in historical.items():
        samples = {str(row.get("sample_id")) for row in rows}
        prompts = {normalize_prompt(str(row.get("prompt", ""))) for row in rows}
        if len(rows) != 120 or len(samples) != 60 or len(prompts) != 60:
            raise M8AuthoringError(f"historical denominator differs: {name}")
        for row in rows:
            sample_id = str(row.get("sample_id"))
            source = risk_by_sample.get(sample_id)
            if source is None or normalize_prompt(str(source.get("prompt", ""))) != normalize_prompt(
                str(row.get("prompt", ""))
            ):
                raise M8AuthoringError(f"historical source binding differs: {name}")
        source_counts[name] = len(rows)
        source_unique_samples[name] = len(samples)
        if excluded_samples.intersection(samples) or excluded_prompt_hashes.intersection(
            sha256_bytes(prompt.encode("utf-8")) for prompt in prompts
        ):
            raise M8AuthoringError("historical cohorts are not prompt-disjoint")
        excluded_samples.update(samples)
        excluded_prompt_hashes.update(
            sha256_bytes(prompt.encode("utf-8")) for prompt in prompts
        )
    if len(excluded_samples) != protocol["selection"]["historical_unique_samples_expected"]:
        raise M8AuthoringError("historical exclusion denominator differs")

    salt = str(protocol["selection"]["ordering_salt"])
    ordered = sorted(
        risk_rows,
        key=lambda row: sha256_bytes(
            f"{salt}|{row['sample_id']}".encode("utf-8")
        ),
    )
    eligible: list[dict[str, Any]] = []
    seen_prompt_hashes = set(excluded_prompt_hashes)
    for row in ordered:
        sample_id = str(row["sample_id"])
        prompt = str(row["prompt"])
        prompt_hash = sha256_bytes(normalize_prompt(prompt).encode("utf-8"))
        if sample_id in excluded_samples or prompt_hash in seen_prompt_hashes:
            continue
        seen_prompt_hashes.add(prompt_hash)
        eligible.append(row)
    selected = eligible[: int(protocol["selection"]["prompts"])]
    if len(selected) != int(protocol["selection"]["prompts"]):
        raise M8AuthoringError("insufficient prompt-disjoint risk rows")

    prompts = []
    for rank, row in enumerate(selected, 1):
        sample_id = str(row["sample_id"])
        prompts.append(
            {
                "opaque_prompt_id": (
                    f"m8_prompt_{rank:03d}_"
                    f"{sha256_bytes(sample_id.encode('utf-8'))[:12]}"
                ),
                "sample_id": sample_id,
                "prompt": str(row["prompt"]),
                "rank": rank,
            }
        )
    registry = {
        "schema": REGISTRY_SCHEMA,
        "selection_salt": salt,
        "risk_rows": len(risk_rows),
        "excluded_candidate_rows": source_counts,
        "excluded_unique_samples": source_unique_samples,
        "excluded_historical_sample_count": len(excluded_samples),
        "eligible_distinct_prompt_rows": len(eligible),
        "selected_prompt_count": len(prompts),
        "prompts": prompts,
    }
    counts = {
        "risk_rows": len(risk_rows),
        "excluded_historical_samples": len(excluded_samples),
        "eligible_distinct_prompt_rows": len(eligible),
        "selected_prompts": len(prompts),
        "requests": len(prompts) * len(protocol["authoring"]["modes"]),
    }
    return registry, counts


def packet_rows(
    protocol: Mapping[str, Any], registry: Mapping[str, Any], shard_index: int
) -> list[dict[str, Any]]:
    rows = []
    modes = protocol["authoring"]["modes"]
    shards = int(protocol["authoring"]["shards"])
    for prompt_index, prompt in enumerate(registry["prompts"]):
        for mode_index, mode in enumerate(modes):
            request_index = prompt_index * len(modes) + mode_index
            if request_index % shards != shard_index:
                continue
            rows.append(
                {
                    "request_id": f"m8_req_{request_index:03d}",
                    "opaque_prompt_id": prompt["opaque_prompt_id"],
                    "sample_id": prompt["sample_id"],
                    "author_mode": mode,
                    "prompt": prompt["prompt"],
                    "rank": prompt["rank"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("derive-registry-hash", "preflight", "prepare"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    validate_protocol(protocol, allow_pending=args.command == "derive-registry-hash")
    registry, counts = build_registry(protocol)
    registry_hash = canonical_sha256(registry)
    if args.command == "derive-registry-hash":
        if protocol.get("prompt_registry_sha256") != PENDING_HASH:
            raise M8AuthoringError("hash derivation is allowed only before protocol commit")
        print(json.dumps({"prompt_registry_sha256": registry_hash}, sort_keys=True))
        return
    if registry_hash != protocol["prompt_registry_sha256"]:
        raise M8AuthoringError("prompt registry differs from frozen hash")
    output = {
        "status": "PASS_CDRO_OPC_M8_AUTHORING_PREFLIGHT",
        **counts,
        "prompt_registry_sha256": registry_hash,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(output, sort_keys=True))
    if args.command != "prepare":
        return
    spool = Path(str(protocol["authoring"]["spool_root"])).resolve()
    try:
        spool.relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        raise M8AuthoringError("authoring spool must remain outside the repository")
    if spool.exists():
        raise M8AuthoringError(f"authoring spool already exists: {spool}")
    packets = spool / "packets"
    packets.mkdir(parents=True)
    for shard_index in range(int(protocol["authoring"]["shards"])):
        rows = packet_rows(protocol, registry, shard_index)
        if len(rows) != int(protocol["authoring"]["rows_per_shard"]):
            raise M8AuthoringError(f"shard denominator differs: {shard_index:02d}")
        (packets / f"shard_{shard_index:02d}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    (spool / "prompt_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS_CDRO_OPC_M8_PROMPT_PACKETS_PREPARED",
                "spool_root": str(spool),
                "shards": protocol["authoring"]["shards"],
                "requests": counts["requests"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
