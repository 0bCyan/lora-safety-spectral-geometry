from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m5/protocol/cdro_opc_m5_authoring_protocol.json"
)
AGENTS = {
    "00": "01a02078-d517-7b60-a6f1-7514b6846313",
    "01": "01a02078-d67d-72c0-b91c-c24d1c9b033f",
    "02": "01a02078-d85b-7ed3-8043-694d70703e28",
    "03": "01a02078-da0e-7e70-a22c-1008ef46f59f",
    "04": "01a02078-dba2-7801-8de6-2a1f3568d9b4",
    "05": "01a02078-dd42-7c02-88ac-a471aac0d369",
    "06": "01a0208e-9739-7d82-accd-6e5a51b7a131",
    "07": "01a0208e-98a5-79c3-95a9-1c90279f2433",
    "08": "01a0208e-9a7a-7de1-b377-0b6a86ba83e4",
    "09": "01a0208e-9c26-7982-9e91-b3a99e57209d",
    "10": "01a0208e-9e14-7423-801e-780b421a9dd2",
    "11": "01a0208e-a0ad-7d80-90a7-f551dad63c7d",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_and_collect(
    protocol: dict[str, Any], spool: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    normalized_hashes: set[str] = set()
    for shard_index in range(12):
        shard = f"{shard_index:02d}"
        packet = read_jsonl(spool / "packets" / f"shard_{shard}.jsonl")
        responses = read_jsonl(spool / "responses" / f"shard_{shard}.jsonl")
        if len(packet) != 10 or len(responses) != 10:
            raise RuntimeError(f"shard {shard} row count mismatch")
        for row_index, (expected, actual) in enumerate(
            zip(packet, responses, strict=True)
        ):
            for key in (
                "request_id",
                "opaque_prompt_id",
                "sample_id",
                "author_mode",
            ):
                if actual.get(key) != expected.get(key):
                    raise RuntimeError(
                        f"shard {shard} row {row_index} metadata mismatch: {key}"
                    )
            response = actual.get("response")
            if (
                not isinstance(response, str)
                or not response
                or response != response.strip()
            ):
                raise RuntimeError(f"shard {shard} row {row_index} invalid response")
            tokens = len(response.split())
            minimum = int(protocol["authoring"]["minimum_response_whitespace_tokens"])
            maximum = int(protocol["authoring"]["maximum_response_whitespace_tokens"])
            if not minimum <= tokens <= maximum:
                raise RuntimeError(f"shard {shard} row {row_index} token envelope")
            opening = protocol["authoring"]["required_openings"][
                actual["author_mode"]
            ]
            if not response.startswith(opening):
                raise RuntimeError(f"shard {shard} row {row_index} opening")
            if "<think" in response.casefold() or "</think" in response.casefold():
                raise RuntimeError(f"shard {shard} row {row_index} thinking tag")
            response_hash = sha256_bytes(response.encode("utf-8"))
            normalized_hash = sha256_bytes(normalize(response).encode("utf-8"))
            if normalized_hash in normalized_hashes:
                raise RuntimeError("global normalized duplicate")
            normalized_hashes.add(normalized_hash)
            candidate_id = (
                "m5_candidate_"
                + hashlib.sha256(actual["request_id"].encode()).hexdigest()[:24]
            )
            public = {
                "request_id": actual["request_id"],
                "opaque_candidate_id": candidate_id,
                "opaque_prompt_id": actual["opaque_prompt_id"],
                "sample_id": actual["sample_id"],
                "author_mode": actual["author_mode"],
                "response_whitespace_tokens": tokens,
                "response_sha256": response_hash,
                "normalized_response_sha256": normalized_hash,
                "authoring_shard": shard,
                "authoring_agent_id": AGENTS[shard],
            }
            candidates.append(
                {
                    **public,
                    "prompt": expected["prompt"],
                    "response": response,
                    "authoring_source": "current_model_no_context_subagents",
                    "model_provenance": "inherited_parent_model_unversioned",
                }
            )
            records.append(public)
    return candidates, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = read_json(REPO / args.protocol)
    spool = Path(protocol["spool_root"])
    output = (
        REPO
        / "experiments/step3_response_quality_persistence/results/"
        "4070ti_step3_cdro_opc_m5/authoring"
    )
    if output.exists():
        raise RuntimeError("M5 authoring output already exists")
    candidates, records = validate_and_collect(protocol, spool)
    prompts = {row["opaque_prompt_id"] for row in candidates}
    normalized = {row["normalized_response_sha256"] for row in candidates}
    if len(candidates) != 120 or len(prompts) != 60 or len(normalized) != 120:
        raise RuntimeError("M5 denominator mismatch")
    summary = {
        "schema": "step3_cdro_opc_m5_authoring_summary_v1",
        "route": "PASS_CDRO_OPC_M5_AUTHORING",
        "fixed_requests": 120,
        "valid_candidates": 120,
        "valid_prompts": 60,
        "unique_normalized_responses": 120,
        "subagent_sessions": 12,
        "context_isolated": True,
        "retry_or_reassignment_used": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    private_path = output / "private" / "primary_candidates.jsonl"
    private_path.parent.mkdir(parents=True)
    private_path.write_bytes(jsonl_bytes(candidates))
    (output / "record_manifest.jsonl").write_bytes(jsonl_bytes(records))
    (output / "authoring_summary.json").write_bytes(json_bytes(summary))
    report = (
        "# CDR-OPC M5 Independent Authoring\n\n"
        "M5 passed the frozen prompt-disjoint authoring gate. Twelve no-context "
        "sub-agents completed 120 fixed requests over 60 prompts. All candidates "
        "passed schema, order, length, opening, thinking-tag, and normalized "
        "deduplication checks. The main session did not label or rewrite responses."
        "\n\nroute=PASS_CDRO_OPC_M5_AUTHORING\n"
        "valid_candidates=120\nvalid_prompts=60\nsubagent_sessions=12\n"
        "optimizer_steps=0\ntraining_authorized=false\n"
    )
    (output / "CDRO_OPC_M5_AUTHORING_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    artifacts = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": "step3_cdro_opc_m5_authoring_manifest_v1",
        "route": summary["route"],
        "spool_root": str(spool),
        "artifacts": artifacts,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

