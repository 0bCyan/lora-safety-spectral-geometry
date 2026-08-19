"""Build the immutable factorized OACPO quality and safety pair pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import step3_oacpo_d1_packets as d1


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_foacpo_d3/protocol/foacpo_d3_factorization_protocol.json"
)


class D3ValidationError(ValueError):
    """Raised when the frozen D3 construction is inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "build", "verify"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise D3ValidationError(f"path escapes repository: {value}") from exc
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def write_frozen(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise D3ValidationError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if protocol.get("status") != "frozen_after_D2_labels_before_any_FOACPO_gradient":
        raise D3ValidationError("D3 protocol status differs")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise D3ValidationError(f"frozen input mismatch: {name}")
    if any(
        (
            protocol.get("gradient_computed") is not False,
            protocol.get("model_checkpoint_opened") is not False,
            protocol.get("optimizer_steps") != 0,
            protocol.get("training_authorized") is not False,
            protocol.get("core100_opened") is not False,
            protocol.get("heldout_opened") is not False,
        )
    ):
        raise D3ValidationError("D3 protocol violates the zero-step boundary")
    d2 = read_json(resolve_repo_path(protocol["frozen_inputs"]["d2_summary"]["path"]))
    if d2.get("route") != "STOP_OACPO_D2_TRIPLET_YIELD":
        raise D3ValidationError("D3 does not descend from the frozen D2 stop")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise D3ValidationError("D3 entry commit is not an ancestor of HEAD")
    return protocol


def build_rows(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d1_protocol = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["d1_protocol"]["path"])
    )
    d1.verify_frozen_inputs(d1_protocol)
    sources = {row["pair_id"]: row for row in d1.load_triplet_sources(d1_protocol)}
    ledger = read_jsonl(resolve_repo_path(protocol["frozen_inputs"]["d2_ledger"]["path"]))
    if len(ledger) != 56 or set(sources) != {row["pair_id"] for row in ledger}:
        raise D3ValidationError("D2 ledger and source triplets differ")

    quality: list[dict[str, Any]] = []
    safety: list[dict[str, Any]] = []
    for label in sorted(ledger, key=lambda row: row["pair_id"]):
        source = sources[label["pair_id"]]
        hashes = {
            "authored_response_normalized_sha256": text_hash(source["authored"]),
            "source_safe_response_normalized_sha256": text_hash(source["source_safe"]),
            "source_unsafe_response_normalized_sha256": text_hash(source["source_unsafe"]),
        }
        if any(hashes[name] != label[name] for name in hashes):
            raise D3ValidationError(f"D2 response hash mismatch: {label['pair_id']}")
        common = {
            "pair_id": label["pair_id"],
            "prospective_split": label["prospective_split"],
            "assigned_category": label["assigned_category"],
            "prompt": source["prompt"],
            "chosen_response": source["authored"],
            "prompt_normalized_sha256": text_hash(source["prompt"]),
            "chosen_response_normalized_sha256": hashes[
                "authored_response_normalized_sha256"
            ],
        }
        if label["authored_positive"] and label["safe_quality_negative"]:
            quality.append(
                {
                    **common,
                    "pool": "quality",
                    "rejected_response": source["source_safe"],
                    "rejected_response_normalized_sha256": hashes[
                        "source_safe_response_normalized_sha256"
                    ],
                }
            )
        if label["authored_positive"] and label["unsafe_negative"]:
            safety.append(
                {
                    **common,
                    "pool": "safety",
                    "rejected_response": source["source_unsafe"],
                    "rejected_response_normalized_sha256": hashes[
                        "source_unsafe_response_normalized_sha256"
                    ],
                }
            )
    return quality, safety


def pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row["prospective_split"] for row in rows).items())),
        "category_count": len({row["assigned_category"] for row in rows}),
        "category_counts": dict(sorted(Counter(row["assigned_category"] for row in rows).items())),
    }


def expected_manifest(protocol: dict[str, Any], quality: list[dict[str, Any]], safety: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = set(row["pair_id"] for row in quality) & set(row["pair_id"] for row in safety)
    return {
        "name": "step3_foacpo_d3_factorized_pool_manifest_v1",
        "route": "PASS_FOACPO_D3_FACTORIZED_POOL_FREEZE",
        "quality_pool": pool_summary(quality),
        "safety_pool": pool_summary(safety),
        "cross_pool_pair_overlap": len(overlap),
        "full_triplet_intersection_required": False,
        "source_labels_used": False,
        "gradient_computed": False,
        "model_checkpoint_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }


def preflight(path: Path) -> dict[str, Any]:
    protocol = context(path)
    quality, safety = build_rows(protocol)
    result = expected_manifest(protocol, quality, safety)
    result["status"] = "PASS_FOACPO_D3_PREFLIGHT"
    print(json.dumps(result, sort_keys=True))
    return result


def build(path: Path) -> dict[str, Any]:
    protocol = context(path)
    quality, safety = build_rows(protocol)
    outputs = protocol["outputs"]
    write_frozen(resolve_repo_path(outputs["quality_pairs"]), jsonl_bytes(quality))
    write_frozen(resolve_repo_path(outputs["safety_pairs"]), jsonl_bytes(safety))
    manifest = expected_manifest(protocol, quality, safety)
    write_frozen(
        resolve_repo_path(outputs["manifest"]),
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def verify(path: Path) -> dict[str, Any]:
    protocol = context(path)
    quality, safety = build_rows(protocol)
    outputs = protocol["outputs"]
    expected = {
        "quality_pairs": jsonl_bytes(quality),
        "safety_pairs": jsonl_bytes(safety),
        "manifest": (
            json.dumps(expected_manifest(protocol, quality, safety), indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    }
    for name, data in expected.items():
        output = resolve_repo_path(outputs[name])
        if not output.is_file() or output.read_bytes() != data:
            raise D3ValidationError(f"frozen output differs: {name}")
    result = {"status": "PASS_FOACPO_D3_VERIFY", "optimizer_steps": 0, "training_authorized": False}
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    {"preflight": preflight, "build": build, "verify": verify}[args.command](path)


if __name__ == "__main__":
    main()
