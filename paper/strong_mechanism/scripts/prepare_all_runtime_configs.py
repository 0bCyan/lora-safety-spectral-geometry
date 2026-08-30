from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
sys.path.insert(0, str(PAPER_ROOT / "scripts"))
from prepare_runtime_config import load_json, prepare_runtime_config  # noqa: E402
from run_pinned_training import runtime_path  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize every machine-local pinned runtime config before formal training.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=PAPER_ROOT / "configs" / "study_registry.json",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-environment-check", action="store_true", help="Testing only; forbidden for the formal M1 lock.")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PAPER_ROOT / "data" / "runs" / "runtime_configs" / "runtime_manifest.json",
    )
    args = parser.parse_args()

    registry_path = args.registry if args.registry.is_absolute() else REPO_ROOT / args.registry
    registry = load_json(registry_path)
    records = []
    first = True
    for formal in registry["formal_configs"]:
        source = REPO_ROOT / formal["path"]
        for arm in formal["arms"]:
            for seed in formal["seeds"]:
                output = runtime_path(source, arm, int(seed))
                runtime, provenance = prepare_runtime_config(
                    source,
                    output,
                    offline=args.offline,
                    verify_machine=(not args.skip_environment_check),
                )
                records.append(
                    {
                        "role": formal["role"],
                        "source_config": formal["path"],
                        "arm": arm,
                        "seed": int(seed),
                        "runtime_config": str(output.relative_to(REPO_ROOT)),
                        "runtime_config_sha256": sha256(output),
                        "provenance": str(output.with_suffix(".provenance.json").relative_to(REPO_ROOT)),
                        "provenance_sha256": sha256(output.with_suffix(".provenance.json")),
                        "resolved_model_revision": runtime["resolved_model_revision"],
                        "resolved_snapshot_path": provenance["resolved_snapshot_path"],
                    }
                )
                first = False

    expected = int(registry["expected_formal_runs"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} runtime configs, created {len(records)}")
    revisions = {record["resolved_model_revision"] for record in records}
    snapshots = {record["resolved_snapshot_path"] for record in records}
    if revisions != {registry["model_revision"]}:
        raise RuntimeError(f"Runtime revisions differ from registry: {sorted(revisions)}")
    if len(snapshots) != 1:
        raise RuntimeError(f"Formal runs resolved to multiple snapshot paths: {sorted(snapshots)}")

    document = {
        "schema_version": "paper_a_runtime_manifest_v1",
        "registry": str(registry_path.relative_to(REPO_ROOT)),
        "registry_sha256": sha256(registry_path),
        "model_repository": registry["model_repository"],
        "model_revision": registry["model_revision"],
        "environment_checked": not args.skip_environment_check,
        "offline_resolution": bool(args.offline),
        "runtime_configs": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.output_manifest.exists():
        existing = json.loads(args.output_manifest.read_text(encoding="utf-8"))
        if existing != document:
            raise FileExistsError(f"Runtime manifest exists with different content: {args.output_manifest}")
    args.output_manifest.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_configs": len(records), "snapshot_paths": len(snapshots)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
