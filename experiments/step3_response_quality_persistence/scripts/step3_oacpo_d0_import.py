"""Validate and freeze an OACPO D0 author output without judging content quality."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "import-author", "verify-import"])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--input")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
    return path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


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


def validate_author_output(
    packet: list[dict[str, Any]], output: list[dict[str, Any]], gates: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = int(gates["required_rows"])
    if len(packet) != required or len(output) != required:
        raise ValueError(f"author row count mismatch: expected {required}, got {len(output)}")
    expected_ids = [row["author_id"] for row in packet]
    actual_ids = [row.get("author_id") for row in output]
    if actual_ids != expected_ids:
        raise ValueError("author ID set or order mismatch")

    allowed = set(gates["schema_exact"])
    minimum = int(gates["minimum_words"])
    maximum = int(gates["maximum_words"])
    forbidden = [str(value).casefold() for value in gates["forbidden_substrings"]]
    canonical: list[dict[str, Any]] = []
    word_counts: list[int] = []
    normalized: list[str] = []
    for index, row in enumerate(output):
        if set(row) != allowed:
            raise ValueError(f"author schema mismatch at row {index + 1}")
        response = row["authored_response"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"empty author response at row {index + 1}")
        response = response.strip()
        count = len(response.split())
        if count < minimum or count > maximum:
            raise ValueError(f"author word count out of range at row {index + 1}: {count}")
        folded = response.casefold()
        if any(marker in folded for marker in forbidden):
            raise ValueError(f"forbidden substring in author response at row {index + 1}")
        normalized_response = normalize_text(response)
        normalized.append(normalized_response)
        word_counts.append(count)
        canonical.append({"author_id": row["author_id"], "authored_response": response})
    if len(set(normalized)) != required:
        raise ValueError("normalized author responses are not all distinct")

    diagnostics = {
        "row_count": required,
        "minimum_word_count": min(word_counts),
        "maximum_word_count": max(word_counts),
        "mean_word_count": sum(word_counts) / len(word_counts),
        "distinct_normalized_response_count": len(set(normalized)),
    }
    return canonical, diagnostics


def load_and_validate(protocol: dict[str, Any], input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = read_json(resolve_repo_path(protocol["base_protocol"]["path"]))
    packet = read_jsonl(resolve_repo_path(base["outputs"]["author_packet"]))
    output = read_jsonl(input_path)
    return validate_author_output(packet, output, base["author_import_gates"])


def expected_outputs(
    protocol: dict[str, Any], input_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    canonical, diagnostics = load_and_validate(protocol, input_path)
    frozen_text = jsonl_text(canonical)
    manifest = {
        "name": "step3_oacpo_d0_author_import_manifest_v1",
        "protocol": protocol["self_path"],
        "protocol_sha256": sha256(resolve_repo_path(protocol["self_path"])),
        "base_protocol_sha256": protocol["base_protocol"]["sha256"],
        "input_path": str(input_path.relative_to(REPO)).replace("\\", "/"),
        "input_sha256": sha256(input_path),
        "frozen_output_sha256": sha256_bytes(frozen_text.encode("utf-8")),
        "mechanical_diagnostics": diagnostics,
        "route": "PASS_OACPO_D0_AUTHOR_IMPORT_AUTHORIZE_D1_PROTOCOL_ONLY",
        "semantic_quality_claimed": False,
        "safety_quality_claimed": False,
        "review_labels_created": 0,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return {"frozen_author_output": frozen_text, "import_manifest": json_text(manifest)}, diagnostics


def output_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {name: resolve_repo_path(path) for name, path in protocol["outputs"].items()}


def write_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    for name, path in output_paths(protocol).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != rendered[name]:
            raise ValueError(f"refusing to overwrite different frozen output: {name}")
        path.write_text(rendered[name], encoding="utf-8", newline="\n")


def verify_outputs(protocol: dict[str, Any], rendered: dict[str, str]) -> None:
    for name, path in output_paths(protocol).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered[name]:
            raise ValueError(f"frozen import output mismatch: {name}")


def main() -> None:
    args = parse_args()
    protocol = read_json(resolve_repo_path(args.protocol))
    if protocol["self_path"] != args.protocol.replace("\\", "/"):
        raise ValueError("protocol self path mismatch")
    verify_frozen_inputs(protocol)
    if args.command == "preflight":
        print("PASS_OACPO_D0_IMPORT_PREFLIGHT")
        return
    if not args.input:
        raise ValueError("--input is required")
    input_path = resolve_repo_path(args.input)
    if str(input_path.relative_to(REPO)).replace("\\", "/") != protocol["author_output_input"]:
        raise ValueError("author output input path mismatch")
    rendered, diagnostics = expected_outputs(protocol, input_path)
    if args.command == "import-author":
        write_outputs(protocol, rendered)
        print("PASS_OACPO_D0_AUTHOR_IMPORT_AUTHORIZE_D1_PROTOCOL_ONLY")
    else:
        verify_outputs(protocol, rendered)
        print("PASS_OACPO_D0_AUTHOR_IMPORT_VERIFY")
    print(json.dumps(diagnostics, sort_keys=True))


if __name__ == "__main__":
    main()
