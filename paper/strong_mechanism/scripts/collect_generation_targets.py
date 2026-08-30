from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
DEFAULT_MANIFEST = PAPER_ROOT / "data" / "derived" / "evaluation_commands.json"
DEFAULT_OUTPUT = PAPER_ROOT / "data" / "raw" / "review_targets"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def opaque_id(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    return "pa_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:28]


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def validate_generation_rows(
    path: Path,
    generation_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    job: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    if len(generation_rows) != int(job["limit"]):
        raise ValueError(f"{path}: expected {job['limit']} rows, got {len(generation_rows)}")
    if len(canonical_rows) < len(generation_rows):
        raise ValueError(f"{path}: canonical evaluation source is shorter than generated output")
    seen = set()
    for index, (generated, canonical) in enumerate(zip(generation_rows, canonical_rows), start=1):
        for field in ["sample_id", "prompt", "response"]:
            if field not in generated:
                raise ValueError(f"{path}:{index}: missing {field}")
        if generated["sample_id"] != canonical["sample_id"]:
            raise ValueError(f"{path}:{index}: sample_id differs from canonical evaluation order")
        if generated["prompt"] != canonical["prompt"]:
            raise ValueError(f"{path}:{index}: prompt bytes differ from canonical evaluation source")
        if generated["sample_id"] in seen:
            raise ValueError(f"{path}:{index}: duplicate sample_id {generated['sample_id']}")
        seen.add(generated["sample_id"])
        if str(generated.get("checkpoint")) != str(checkpoint["checkpoint_tag"]):
            raise ValueError(f"{path}:{index}: checkpoint tag mismatch")
        if int(generated.get("seed")) != int(job["seed"]):
            raise ValueError(f"{path}:{index}: seed mismatch")
        if str(generated.get("run_label", "")) != str(job["arm"]):
            raise ValueError(f"{path}:{index}: arm/run_label mismatch")


def row_from_generation(
    packet_seed: int,
    job: dict[str, Any],
    checkpoint: dict[str, Any],
    generated: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    prompt_id = str(generated["sample_id"])
    response = str(generated.get("response", ""))
    sample_id = opaque_id(packet_seed, job["kind"], job["arm"], job["seed"], checkpoint["checkpoint"], prompt_id)
    return {
        "sample_id": sample_id,
        "prompt": str(generated["prompt"]),
        "response": response,
        "setting": "qwen3_4b_paper_a",
        "model": job["model"],
        "model_revision": job["model_revision"],
        "arm": job["arm"],
        "seed": int(job["seed"]),
        "checkpoint": int(checkpoint["checkpoint"]),
        "checkpoint_tag": checkpoint["checkpoint_tag"],
        "prompt_id": prompt_id,
        "eval_set": job["eval_set"],
        "response_sha256": sha256_bytes(response.encode("utf-8")),
        "generation_file": str(source_path),
        "generation_file_sha256": sha256_file(source_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate formal generations and materialize arm-blind review target tables.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true", help="Testing only: do not require the full registered job matrix.")
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    manifest = load_json(manifest_path)
    registry_path = resolve(manifest["registry"])
    registry = load_json(registry_path)
    packet_seed = int(registry["review_packet_seed"])

    jobs = [job for job in manifest["jobs"] if job["kind"] in {"safety_generation", "utility_generation"}]
    if not args.allow_partial:
        expected = int(registry["expected_formal_runs"]) * 2
        if len(jobs) != expected:
            raise ValueError(f"Expected {expected} generation jobs, found {len(jobs)}")

    canonical_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
    safety_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    for job in jobs:
        canonical_key = (job["canonical_eval_path"], int(job["limit"]))
        if canonical_key not in canonical_cache:
            canonical_cache[canonical_key] = read_jsonl(resolve(job["canonical_eval_path"]))[: int(job["limit"])]
        canonical_rows = canonical_cache[canonical_key]
        for checkpoint in job["expected_generation_paths"]:
            path = resolve(checkpoint["path"])
            if not path.exists():
                raise FileNotFoundError(f"Missing registered generation file: {path}")
            generated = read_jsonl(path)
            validate_generation_rows(path, generated, canonical_rows, job, checkpoint)
            materialized = [
                row_from_generation(packet_seed, job, checkpoint, row, path)
                for row in generated
            ]
            if job["kind"] == "safety_generation":
                safety_rows.extend(materialized)
                if int(checkpoint["checkpoint"]) == int(registry["registered_checkpoints"][-1]):
                    quality_rows.extend(materialized)
            else:
                utility_rows.extend(materialized)
            source_files.append(
                {
                    "kind": job["kind"],
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "checkpoint": checkpoint["checkpoint"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": len(generated),
                }
            )

    for name, rows in [
        ("safety", safety_rows),
        ("quality", quality_rows),
        ("utility", utility_rows),
    ]:
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Opaque IDs are not unique in {name} targets")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safety_path = args.output_dir / "safety_review_targets.csv"
    quality_path = args.output_dir / "quality_review_targets.csv"
    utility_path = args.output_dir / "utility_review_targets.csv"
    write_csv(safety_path, safety_rows)
    write_csv(quality_path, quality_rows)
    write_csv(utility_path, utility_rows)

    integrity = {
        "schema_version": "paper_a_generation_target_integrity_v1",
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "safety_rows": len(safety_rows),
        "quality_rows": len(quality_rows),
        "utility_rows": len(utility_rows),
        "safety_target_sha256": sha256_file(safety_path),
        "quality_target_sha256": sha256_file(quality_path),
        "utility_target_sha256": sha256_file(utility_path),
        "source_files": source_files,
    }
    (args.output_dir / "generation_target_integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: integrity[key] for key in ["safety_rows", "quality_rows", "utility_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
