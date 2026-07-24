"""Reconcile the frozen P6 manifest with Git-canonical LF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
P6_ROOT = REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/mechanism_completion/p6"
P6A_ROOT = REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/mechanism_completion/p6a"
LEGACY_MANIFEST = P6_ROOT / "step3ct_t5r_p6_mechanism_manifest_package.json"
CANONICAL_MANIFEST = P6A_ROOT / "step3ct_t5r_p6a_canonical_manifest.json"
ROUTE_PATH = P6A_ROOT / "step3ct_t5r_p6a_route_decision.json"
REPORT_PATH = P6A_ROOT / "Step3CT_T5R_P6A_canonical_hash_reconciliation_report.md"
EXECUTION_PATH = P6A_ROOT / "step3ct_t5r_p6a_execution_record.json"
PACKAGE_PATH = P6A_ROOT / "step3ct_t5r_p6a_package_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob_oid(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def canonical_lf_bytes(value: bytes) -> bytes:
    text = value.decode("utf-8")
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        raise ValueError("lone CR is not allowed in a canonical text artifact")
    return text.encode("utf-8")


def crlf_bytes_from_lf(value: bytes) -> bytes:
    canonical = canonical_lf_bytes(value)
    return canonical.replace(b"\n", b"\r\n")


def legacy_match_mode(current: bytes, expected_sha256: str, expected_bytes: int) -> str | None:
    if len(current) == expected_bytes and sha256_bytes(current) == expected_sha256:
        return "raw_committed_bytes"
    crlf = crlf_bytes_from_lf(current)
    if len(crlf) == expected_bytes and sha256_bytes(crlf) == expected_sha256:
        return "canonical_lf_reexpanded_to_crlf"
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_lf(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json_lf(path: Path, value: Any) -> None:
    write_text_lf(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def audit_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    legacy = read_json(LEGACY_MANIFEST)
    rows: list[dict[str, Any]] = []
    for relative_path, expected in legacy["artifacts"].items():
        path = REPO / relative_path
        current = path.read_bytes()
        canonical = canonical_lf_bytes(current)
        mode = legacy_match_mode(current, str(expected["sha256"]), int(expected["bytes"]))
        head_blob = git("rev-parse", f"HEAD:{relative_path}")
        row = {
            "path": relative_path,
            "legacy_sha256": str(expected["sha256"]),
            "legacy_bytes": int(expected["bytes"]),
            "legacy_match_mode": mode,
            "committed_lf_sha256": sha256_bytes(canonical),
            "committed_lf_bytes": len(canonical),
            "working_tree_is_canonical_lf": current == canonical,
            "computed_git_blob_oid": git_blob_oid(canonical),
            "head_git_blob_oid": head_blob,
            "head_git_blob_match": git_blob_oid(canonical) == head_blob,
        }
        if path.suffix == ".json":
            json.loads(canonical.decode("utf-8"))
            row["json_parse_pass"] = True
        rows.append(row)
    return legacy, rows


def run() -> None:
    legacy, rows = audit_rows()
    raw_matches = sum(row["legacy_match_mode"] == "raw_committed_bytes" for row in rows)
    crlf_matches = sum(row["legacy_match_mode"] == "canonical_lf_reexpanded_to_crlf" for row in rows)
    unmatched = [row["path"] for row in rows if row["legacy_match_mode"] is None]
    blob_failures = [row["path"] for row in rows if not row["head_git_blob_match"]]
    passed = not unmatched and not blob_failures and len(rows) == int(legacy["artifact_count"])
    route = "P6A1_eol_reconciled_content_unchanged" if passed else "P6A0_non_eol_content_or_git_blob_mismatch"

    canonical_manifest = {
        "name": "step3ct_t5r_p6a_canonical_manifest_v1",
        "route": route,
        "source_git_head": git("rev-parse", "HEAD"),
        "legacy_manifest_path": LEGACY_MANIFEST.relative_to(REPO).as_posix(),
        "legacy_manifest_sha256": sha256_file(LEGACY_MANIFEST),
        "canonicalization": "UTF-8 with LF line endings; SHA-256 over committed LF bytes",
        "git_blob_definition": "SHA-1 over blob <byte_length> NUL plus committed LF bytes",
        "artifact_count": len(rows),
        "legacy_raw_matches": raw_matches,
        "legacy_crlf_reconstruction_matches": crlf_matches,
        "unmatched_paths": unmatched,
        "head_blob_mismatches": blob_failures,
        "artifacts": rows,
    }
    write_json_lf(CANONICAL_MANIFEST, canonical_manifest)

    route_record = {
        "name": "step3ct_t5r_p6a_route_decision_v1",
        "route": route,
        "gates": {
            "legacy_artifact_count_complete": len(rows) == int(legacy["artifact_count"]),
            "every_legacy_hash_explained_by_raw_or_crlf_only": not unmatched,
            "every_committed_lf_blob_matches_head": not blob_failures,
            "p6_files_modified": False,
            "mechanism_route_revised": False,
            "behavior_or_candidate_claim_added": False,
        },
        "legacy_route_preserved": legacy["route"],
        "canonical_manifest_sha256": sha256_file(CANONICAL_MANIFEST),
        "next_authorization": "external_independent_safety_review_only",
    }
    write_json_lf(ROUTE_PATH, route_record)

    report = "\n".join(
        [
            "# Step3C-T5R P6A Canonical Hash Reconciliation Report",
            "",
            f"Route: `{route}`",
            "",
            "## Result",
            "",
            f"The legacy P6 manifest tracked {len(rows)} artifacts. {raw_matches} already match committed LF bytes, "
            f"and {crlf_matches} match exactly after canonical LF bytes are re-expanded to CRLF. "
            f"Unexplained artifacts: {len(unmatched)}. HEAD blob mismatches: {len(blob_failures)}.",
            "",
            "The discrepancy is therefore an end-of-line freeze defect rather than a numerical or semantic content change. "
            "This addendum records SHA-256 over committed UTF-8/LF bytes and the corresponding Git blob OID.",
            "",
            "## Boundary",
            "",
            "No P6 file was overwritten. The mechanism route remains `M1_reconstruction_integrity_failure`; label64 "
            "grouped metrics remain unregistered, Track A remains without behavior labels, and no candidate or new "
            "training is authorized.",
            "",
        ]
    )
    write_text_lf(REPORT_PATH, report)

    execution = {
        "name": "step3ct_t5r_p6a_execution_record_v1",
        "command": "python experiments/step3_response_quality_persistence/scripts/audit_step3ct_t5r_p6_eol.py run",
        "git_head": git("rev-parse", "HEAD"),
        "legacy_artifacts": len(rows),
        "raw_matches": raw_matches,
        "crlf_reconstruction_matches": crlf_matches,
        "unmatched": len(unmatched),
        "head_blob_mismatches": len(blob_failures),
        "p6_files_modified": False,
        "heldout_opened": False,
    }
    write_json_lf(EXECUTION_PATH, execution)

    package_paths = [CANONICAL_MANIFEST, ROUTE_PATH, REPORT_PATH, EXECUTION_PATH]
    package = {
        "name": "step3ct_t5r_p6a_package_manifest_v1",
        "route": route,
        "canonical_lf": True,
        "artifacts": {
            path.relative_to(REPO).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in package_paths
        },
    }
    write_json_lf(PACKAGE_PATH, package)
    if not passed:
        raise RuntimeError(route)
    print(json.dumps({"route": route, "raw": raw_matches, "crlf": crlf_matches, "artifacts": len(rows)}))


def verify() -> None:
    package = read_json(PACKAGE_PATH)
    failures = []
    for relative_path, expected in package["artifacts"].items():
        path = REPO / relative_path
        if sha256_file(path) != expected["sha256"] or path.stat().st_size != int(expected["bytes"]):
            failures.append(relative_path)
    canonical = read_json(CANONICAL_MANIFEST)
    if canonical["unmatched_paths"] or canonical["head_blob_mismatches"]:
        failures.append(CANONICAL_MANIFEST.relative_to(REPO).as_posix())
    if failures:
        raise RuntimeError(f"P6A verification failed: {failures}")
    print(json.dumps({"route": package["route"], "package_artifacts": len(package["artifacts"]), "passed": True}))


def main() -> None:
    args = parse_args()
    if args.action == "run":
        run()
    else:
        verify()


if __name__ == "__main__":
    main()
