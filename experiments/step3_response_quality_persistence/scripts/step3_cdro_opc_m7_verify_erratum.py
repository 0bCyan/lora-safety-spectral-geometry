"""Read-only verifier erratum for the frozen M7 artifact set.

VE1 corrects one representation-layer defect: JSON object member order is not
schema, while the frozen M7 verifier compared that order with TABLE_FIELDS.
The adapter validates the exact key set, reconstructs an in-memory mapping in
TABLE_FIELDS order, and delegates every other check to the frozen verifier.
No original artifact is written or reserialized.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3_cdro_opc_m7_mirror_curvature as m7


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m7_verifier_erratum/protocol/"
    "cdro_opc_m7_verifier_erratum_protocol.json"
)
OUTPUT_ROOT = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m7_verifier_erratum"
)
ORIGINAL_M7_ROOT = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m7"
)
ATTEMPT_CLAIM_PATH = f"{OUTPUT_ROOT}/official_attempt_01_claim.json"
RECEIPT_PATH = f"{OUTPUT_ROOT}/run/cdro_opc_m7_ve1_verification_receipt.json"
REPORT_PATH = f"{OUTPUT_ROOT}/run/CDRO_OPC_M7_VE1_REPORT.md"
REPRODUCTION_PATH = f"{OUTPUT_ROOT}/run/reproduction_manifest.json"
EXPECTED_OUTPUTS = {
    "attempt_claim": ATTEMPT_CLAIM_PATH,
    "receipt": RECEIPT_PATH,
    "report": REPORT_PATH,
    "reproduction": REPRODUCTION_PATH,
}
ENTRY_COMMIT = "0cb5788f0ca9eba13f1be94deabe1e1bfb4690fc"
SOURCE_RESULT_COMMIT = "ab270c07fee396cb39d3c1593f45acdf56155976"
M7_RUNTIME_COMMIT = "892768286718f0dd54b7fb8df47e6e33b7a5364c"
REGISTERED_STATUS = "FROZEN_POST_RESULT_CDRO_OPC_M7_VERIFIER_ONLY_ERRATUM_V1"
ERRATUM_ROUTE = "PASS_CDRO_OPC_M7_VE1_UNCHANGED_ARTIFACTS_VERIFIED"
TARGET_DRIFT_ROUTE = "STOP_CDRO_OPC_M7_VE1_TARGET_DRIFT"
SCOPE_INTEGRITY_ROUTE = "STOP_CDRO_OPC_M7_VE1_SCOPE_INTEGRITY"
FULL_VERIFY_ROUTE = "STOP_CDRO_OPC_M7_VE1_FULL_VERIFY"
UNDERLYING_M7_ROUTE = "PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC"
ORIGINAL_VERIFY_STATUS = "ERROR_CDRO_OPC_M7_VERIFY_RUN_IMPLEMENTATION_DEFECT"
ORIGINAL_ERROR = "M7 flattened table does not reproduce"
UNIQUE_REPRODUCTION_COMMAND = (
    ".\\.venv\\Scripts\\python.exe experiments/"
    "step3_response_quality_persistence/scripts/"
    "step3_cdro_opc_m7_verify_erratum.py run"
)
VERIFICATION_COMMAND = (
    ".\\.venv\\Scripts\\python.exe experiments/"
    "step3_response_quality_persistence/scripts/"
    "step3_cdro_opc_m7_verify_erratum.py verify-run"
)
EXPECTED_ROWS = 80
ATTEMPT_CLAIM_KEYS = {
    "schema",
    "status",
    "official_attempt_index",
    "runtime_commit",
    "protocol_path",
    "protocol_sha256",
    "runner_sha256",
    "worktree_clean_before_claim",
    "overwrite_allowed",
    "result_dependent_patch_or_rerun_allowed",
}
REPRODUCTION_KEYS = {
    "schema",
    "runtime_commit",
    "environment",
    "inputs_sha256",
    "outputs_sha256",
    "source_sha256_before",
    "source_sha256_after",
    "unique_reproduction_command",
    "verification_command",
    "official_attempt_index",
    "official_attempt_status",
    "erratum_route",
    "input_validation_complete",
    "original_artifact_writes",
    "fresh_model_measurements",
    "optimizer_steps",
    "sealed_audit_opened",
    "training_authorized",
    "scientific_claim_expanded",
}
STOP_RECEIPT_KEYS = {
    "schema",
    "erratum_route",
    "underlying_m7_route",
    "official_attempt_index",
    "official_attempt_closed",
    "input_validation_complete",
    "failure",
    "claim_scope",
    "M6_stop_preserved",
    "original_artifact_writes",
    "fresh_model_measurements",
    "loss_writes",
    "gradient_evaluations",
    "optimizer_steps",
    "parameter_writes",
    "checkpoint_writes",
    "candidate_generation",
    "sealed_audit_opened",
    "training_authorized",
    "scientific_claim_expanded",
}
EXPECTED_SOURCE_KEYS = (
    "m7_runner",
    "m7_runner_tests",
    "m7_protocol_note",
    "m7_protocol",
    "m7_decision",
    "m7_diagnostics",
    "m7_environment",
    "m7_report",
    "m7_attribution_csv",
    "m7_interpretation",
    "m7_figure_svg",
    "m7_figure_png",
    "m7_chinese_checklist",
    "m7_reproduction",
    "m6_diagnostics",
)
EXPECTED_FROZEN_KEYS = {
    "erratum_runner",
    "erratum_tests",
    "erratum_note",
    "source_evidence",
    *EXPECTED_SOURCE_KEYS,
}
EXPECTED_FROZEN_PATHS = {
    "erratum_runner": "experiments/step3_response_quality_persistence/scripts/step3_cdro_opc_m7_verify_erratum.py",
    "erratum_tests": "experiments/step3_response_quality_persistence/scripts/test_step3_cdro_opc_m7_verify_erratum.py",
    "erratum_note": "experiments/step3_response_quality_persistence/Step3_CDR_OPC_M7_verifier_erratum_plan.md",
    "source_evidence": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7_verifier_erratum/source_evidence/m7_pre_erratum_failure_and_exploratory_record.json",
    "m7_runner": "experiments/step3_response_quality_persistence/scripts/step3_cdro_opc_m7_mirror_curvature.py",
    "m7_runner_tests": "experiments/step3_response_quality_persistence/scripts/test_step3_cdro_opc_m7_mirror_curvature.py",
    "m7_protocol_note": "experiments/step3_response_quality_persistence/Step3_CDR_OPC_M7_mirror_curvature_attribution_plan.md",
    "m7_protocol": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/protocol/cdro_opc_m7_mirror_curvature_protocol.json",
    "m7_decision": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/run/cdro_opc_m7_mirror_curvature_decision.json",
    "m7_diagnostics": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/run/cdro_opc_m7_mirror_curvature_diagnostics.json",
    "m7_environment": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/run/cdro_opc_m7_mirror_curvature_environment.json",
    "m7_report": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/run/CDRO_OPC_M7_MIRROR_CURVATURE_REPORT.md",
    "m7_attribution_csv": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/analysis/cdro_opc_m7_mirror_curvature_attribution.csv",
    "m7_interpretation": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/analysis/CDRO_OPC_M7_MIRROR_CURVATURE_INTERPRETATION.md",
    "m7_figure_svg": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/figures/cdro_opc_m7_risk_decomposition.svg",
    "m7_figure_png": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/figures/cdro_opc_m7_risk_decomposition_300dpi.png",
    "m7_chinese_checklist": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/run/复现清单.json",
    "m7_reproduction": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m7/run/reproduction_manifest.json",
    "m6_diagnostics": "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m6/run/cdro_opc_m6_virtual_displacement_diagnostics.json",
}
M7_RUNTIME_BOUND_KEYS = {
    "m7_runner",
    "m7_runner_tests",
    "m7_protocol_note",
    "m7_protocol",
    "m6_diagnostics",
}
VE1_RUNTIME_BOUND_KEYS = {"erratum_runner", "erratum_tests", "erratum_note"}
PROTOCOL_KEYS = {
    "schema",
    "status",
    "entry_commit",
    "source_result_commit",
    "m7_runtime_commit",
    "objective",
    "claim_scope",
    "known_before_registration",
    "frozen_inputs",
    "source_artifact_keys",
    "defect_fingerprint",
    "correction",
    "expected_outcomes",
    "output_root",
    "outputs",
    "attempt_policy",
    "authorization_boundary",
}


class M7VerifierErratumError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M7VerifierErratumError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M7VerifierErratumError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise M7VerifierErratumError("VE1 official attempt is already claimed") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_commit_exists(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def git_commit_blob_oid(commit: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_filtered_worktree_blob_oid(path: str) -> str | None:
    completed = subprocess.run(
        ["git", "hash-object", "--path", path, path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_file_matches_commit(commit: str, path: str) -> bool:
    return bool(
        repo_path(path).is_file()
        and git_commit_blob_oid(commit, path)
        == git_filtered_worktree_blob_oid(path)
    )


def expected_input_hashes(
    protocol_path: Path, protocol: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "protocol": file_sha256(protocol_path),
        **{
            name: str(spec["sha256"])
            for name, spec in protocol["frozen_inputs"].items()
        },
    }


def validate_source_git_bindings(protocol: Mapping[str, Any]) -> None:
    for name, expected_path in EXPECTED_FROZEN_PATHS.items():
        if protocol["frozen_inputs"][name]["path"] != expected_path:
            raise M7VerifierErratumError(
                f"VE1 frozen input path differs: {name}"
            )
    for name in EXPECTED_SOURCE_KEYS:
        if not git_file_matches_commit(
            SOURCE_RESULT_COMMIT, EXPECTED_FROZEN_PATHS[name]
        ):
            raise M7VerifierErratumError(
                f"VE1 source-result Git blob differs: {name}"
            )
    for name in M7_RUNTIME_BOUND_KEYS:
        if not git_file_matches_commit(
            M7_RUNTIME_COMMIT, EXPECTED_FROZEN_PATHS[name]
        ):
            raise M7VerifierErratumError(
                f"VE1 M7-runtime Git blob differs: {name}"
            )
    if not git_file_matches_commit(
        ENTRY_COMMIT, EXPECTED_FROZEN_PATHS["source_evidence"]
    ):
        raise M7VerifierErratumError("VE1 entry evidence Git blob differs")


def validate_ve1_runtime_git_bindings(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    runtime_commit: str,
) -> None:
    if (
        not git_commit_exists(runtime_commit)
        or not git_is_ancestor(ENTRY_COMMIT, runtime_commit)
        or not git_file_matches_commit(runtime_commit, DEFAULT_PROTOCOL)
    ):
        raise M7VerifierErratumError("VE1 runtime protocol Git blob differs")
    for name in VE1_RUNTIME_BOUND_KEYS:
        if not git_file_matches_commit(
            runtime_commit, EXPECTED_FROZEN_PATHS[name]
        ):
            raise M7VerifierErratumError(
                f"VE1 runtime Git blob differs: {name}"
            )
    if file_sha256(protocol_path) != file_sha256(repo_path(DEFAULT_PROTOCOL)):
        raise M7VerifierErratumError("VE1 runtime protocol path differs")
    for name in VE1_RUNTIME_BOUND_KEYS:
        if file_sha256(repo_path(EXPECTED_FROZEN_PATHS[name])) != protocol[
            "frozen_inputs"
        ][name]["sha256"]:
            raise M7VerifierErratumError(
                f"VE1 runtime frozen SHA differs: {name}"
            )


def load_context(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if set(protocol) != PROTOCOL_KEYS:
        raise M7VerifierErratumError("VE1 protocol top-level schema differs")
    if (
        protocol.get("schema")
        != "step3_cdro_opc_m7_verifier_erratum_protocol_v1"
        or protocol.get("status") != REGISTERED_STATUS
        or protocol.get("entry_commit") != ENTRY_COMMIT
        or protocol.get("source_result_commit") != SOURCE_RESULT_COMMIT
        or protocol.get("m7_runtime_commit") != M7_RUNTIME_COMMIT
        or protocol.get("claim_scope")
        != "retrospective_post_result_verifier_only_representation_erratum_no_scientific_claim_expansion"
        or protocol.get("output_root") != OUTPUT_ROOT
        or tuple(protocol.get("source_artifact_keys", ()))
        != EXPECTED_SOURCE_KEYS
        or set(protocol.get("frozen_inputs", {})) != EXPECTED_FROZEN_KEYS
    ):
        raise M7VerifierErratumError("VE1 protocol identity differs")
    output_root = repo_path(str(protocol["output_root"]))
    original_root = repo_path(ORIGINAL_M7_ROOT)
    if (
        output_root == original_root
        or output_root.is_relative_to(original_root)
        or original_root.is_relative_to(output_root)
    ):
        raise M7VerifierErratumError("VE1 output root overlaps original M7 root")
    if (
        not all(
            git_commit_exists(commit)
            for commit in (ENTRY_COMMIT, SOURCE_RESULT_COMMIT, M7_RUNTIME_COMMIT)
        )
        or not git_is_ancestor(M7_RUNTIME_COMMIT, SOURCE_RESULT_COMMIT)
        or not git_is_ancestor(SOURCE_RESULT_COMMIT, ENTRY_COMMIT)
    ):
        raise M7VerifierErratumError("VE1 registered commit provenance differs")
    for name, spec in protocol["frozen_inputs"].items():
        if set(spec) != {"path", "sha256"}:
            raise M7VerifierErratumError(f"VE1 frozen input schema differs: {name}")
        frozen_path = repo_path(str(spec["path"]))
        if not frozen_path.is_file() or file_sha256(frozen_path) != spec["sha256"]:
            raise M7VerifierErratumError(f"VE1 frozen input differs: {name}")
    validate_source_git_bindings(protocol)
    if protocol.get("known_before_registration") != {
        "original_verify_failure_observed": True,
        "in_memory_normalized_full_verify_pass_observed": True,
        "prospective_or_blinded": False,
    }:
        raise M7VerifierErratumError("VE1 prior-knowledge declaration differs")
    if protocol.get("correction") != {
        "authorized_change_count": 1,
        "json_object_schema_predicate": "exact_member_name_set_equality",
        "in_memory_normalization_order": "frozen_M7_TABLE_FIELDS",
        "mapping_equality_must_be_unchanged": True,
        "canonical_sha256_must_be_unchanged": True,
        "delegate_frozen_full_M7_verifier": True,
        "csv_header_order_predicate": "unchanged_exact_TABLE_FIELDS_tuple",
    }:
        raise M7VerifierErratumError("VE1 correction scope differs")
    expected_outcomes = {
        "original_verify_status": ORIGINAL_VERIFY_STATUS,
        "original_error": ORIGINAL_ERROR,
        "erratum_route": ERRATUM_ROUTE,
        "target_drift_route": TARGET_DRIFT_ROUTE,
        "scope_integrity_route": SCOPE_INTEGRITY_ROUTE,
        "full_verify_route": FULL_VERIFY_ROUTE,
        "underlying_m7_route": UNDERLYING_M7_ROUTE,
    }
    if protocol.get("expected_outcomes") != expected_outcomes:
        raise M7VerifierErratumError("VE1 expected outcomes differ")
    if protocol.get("attempt_policy") != {
        "official_attempts": 1,
        "claim_before_official_audit": True,
        "failed_attempt_published_as_stop": True,
        "hard_crash_claim_blocks_rerun": True,
        "overwrite_allowed": False,
        "result_dependent_patch_or_rerun_allowed": False,
    }:
        raise M7VerifierErratumError("VE1 attempt policy differs")
    if protocol.get("authorization_boundary") != {
        "cpu_read_only": True,
        "original_artifact_writes": 0,
        "fresh_model_measurements": 0,
        "loss_writes": 0,
        "gradient_evaluations": 0,
        "optimizer_steps": 0,
        "parameter_writes": 0,
        "checkpoint_writes": 0,
        "candidate_generation": 0,
        "sealed_audit_opened": False,
        "training_authorized": False,
        "scientific_claim_expanded": False,
    }:
        raise M7VerifierErratumError("VE1 authorization boundary differs")
    if protocol.get("outputs") != EXPECTED_OUTPUTS:
        raise M7VerifierErratumError("VE1 output registry differs")
    expected_run_root = repo_path(RECEIPT_PATH).parent
    if expected_run_root != output_root / "run":
        raise M7VerifierErratumError("VE1 run root differs")
    output_paths = [repo_path(value) for value in EXPECTED_OUTPUTS.values()]
    frozen_paths = {
        repo_path(str(spec["path"]))
        for spec in protocol["frozen_inputs"].values()
    }
    if any(
        path == original_root
        or path.is_relative_to(original_root)
        or path in frozen_paths
        or not path.is_relative_to(output_root)
        for path in output_paths
    ):
        raise M7VerifierErratumError("VE1 output path violates isolation")
    m7_environment = read_json(
        repo_path(str(protocol["frozen_inputs"]["m7_environment"]["path"]))
    )
    m7_reproduction = read_json(
        repo_path(str(protocol["frozen_inputs"]["m7_reproduction"]["path"]))
    )
    if (
        m7_environment.get("head") != M7_RUNTIME_COMMIT
        or m7_reproduction.get("runtime_commit") != M7_RUNTIME_COMMIT
        or m7_reproduction.get("environment") != m7_environment
    ):
        raise M7VerifierErratumError("VE1 frozen M7 runtime provenance differs")
    return protocol


def source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: file_sha256(
            repo_path(str(protocol["frozen_inputs"][name]["path"]))
        )
        for name in protocol["source_artifact_keys"]
    }


def normalize_table_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise M7VerifierErratumError("M7 flattened table row is not an object")
    if set(row) != set(m7.TABLE_FIELDS):
        missing = sorted(set(m7.TABLE_FIELDS) - set(row))
        extra = sorted(set(row) - set(m7.TABLE_FIELDS))
        raise M7VerifierErratumError(
            f"M7 flattened table exact key set differs; missing={missing}, extra={extra}"
        )
    return {field: row[field] for field in m7.TABLE_FIELDS}


def normalize_diagnostics(
    diagnostics: Mapping[str, Any], m7_protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_serialization = json.dumps(
        diagnostics, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    rows = diagnostics.get("attribution", {}).get("table_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise M7VerifierErratumError("M7 flattened table row denominator differs")
    allowlist = m7_protocol.get("calibration_allowlist", [])
    expected_cells = [
        (str(unit["unit_id"]), str(unit["sample_id"]), certificate)
        for unit in allowlist
        for certificate in m7.CERTIFICATES
    ]
    observed_cells: list[tuple[str, str, str]] = []
    normalized_rows: list[dict[str, Any]] = []
    order_defect_rows = 0
    for row in rows:
        normalized = normalize_table_row(row)
        normalized_rows.append(normalized)
        observed_cells.append(
            (
                str(row["unit_id"]),
                str(row["sample_id"]),
                str(row["certificate"]),
            )
        )
        order_defect_rows += tuple(row) != m7.TABLE_FIELDS
    if observed_cells != expected_cells or len(set(observed_cells)) != EXPECTED_ROWS:
        raise M7VerifierErratumError("M7 flattened table cell registry differs")
    normalized_diagnostics = copy.deepcopy(diagnostics)
    normalized_diagnostics["attribution"]["table_rows"] = normalized_rows
    if rows != normalized_rows:
        raise M7VerifierErratumError("VE1 normalization changed mapping values")
    original_canonical = canonical_sha256(rows)
    normalized_canonical = canonical_sha256(normalized_rows)
    if original_canonical != normalized_canonical:
        raise M7VerifierErratumError("VE1 normalization changed canonical SHA-256")
    if json.dumps(
        diagnostics, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) != original_serialization:
        raise M7VerifierErratumError("VE1 mutated the source diagnostics object")
    first = rows[0]
    audit = {
        "rows": len(rows),
        "unique_cells": len(set(observed_cells)),
        "exact_key_set_rows": sum(
            set(row) == set(m7.TABLE_FIELDS) for row in rows
        ),
        "order_defect_rows": order_defect_rows,
        "first_unit_id": str(first["unit_id"]),
        "first_certificate": str(first["certificate"]),
        "first_actual_key": next(iter(first)),
        "expected_first_key": m7.TABLE_FIELDS[0],
        "table_rows_canonical_sha256": original_canonical,
        "cell_sequence_canonical_sha256": canonical_sha256(observed_cells),
        "mapping_equal_after_normalization": rows == normalized_rows,
        "canonical_sha256_equal_after_normalization": (
            original_canonical == normalized_canonical
        ),
        "source_object_not_mutated": True,
    }
    return normalized_diagnostics, audit


def validate_defect_fingerprint(
    protocol: Mapping[str, Any], audit: Mapping[str, Any]
) -> None:
    if audit != protocol.get("defect_fingerprint"):
        raise M7VerifierErratumError("VE1 registered defect fingerprint differs")


def reproduce_original_failure(m7_protocol_path: Path) -> dict[str, Any]:
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            m7.verify(m7_protocol_path)
    except m7.M7MirrorCurvatureError as exc:
        if str(exc) != ORIGINAL_ERROR:
            raise M7VerifierErratumError(
                f"original M7 verifier failed differently: {exc}"
            ) from exc
        return {
            "status": ORIGINAL_VERIFY_STATUS,
            "error_class": type(exc).__name__,
            "error_message": str(exc),
            "captured_stdout": stdout.getvalue().strip(),
        }
    raise M7VerifierErratumError("original M7 verifier no longer reproduces its error")


def corrected_full_verify(
    m7_protocol_path: Path,
    diagnostics_path: Path,
    m7_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    original_read_json = m7.read_json
    audits: list[dict[str, Any]] = []

    def adapted_read_json(path: Path) -> dict[str, Any]:
        value = original_read_json(path)
        if Path(path).resolve() == diagnostics_path.resolve():
            value, audit = normalize_diagnostics(value, m7_protocol)
            audits.append(audit)
        return value

    stdout = io.StringIO()
    m7.read_json = adapted_read_json
    try:
        with contextlib.redirect_stdout(stdout):
            result = m7.verify(m7_protocol_path)
    finally:
        m7.read_json = original_read_json
    if len(audits) != 1:
        raise M7VerifierErratumError("VE1 diagnostics adapter call count differs")
    if result != {
        "status": "PASS_CDRO_OPC_M7_MIRROR_CURVATURE_VERIFY_RUN",
        "route": UNDERLYING_M7_ROUTE,
        "attributed_cells": EXPECTED_ROWS,
        "sealed_audit_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }:
        raise M7VerifierErratumError("adapted frozen M7 full verifier did not pass")
    return result, audits[0], stdout.getvalue().strip()


def perform_read_only_audit(
    protocol_path: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    before = source_hashes(protocol)
    m7_protocol_path = repo_path(
        str(protocol["frozen_inputs"]["m7_protocol"]["path"])
    )
    diagnostics_path = repo_path(
        str(protocol["frozen_inputs"]["m7_diagnostics"]["path"])
    )
    m7_protocol = read_json(m7_protocol_path)
    _, direct_audit = normalize_diagnostics(
        read_json(diagnostics_path), m7_protocol
    )
    validate_defect_fingerprint(protocol, direct_audit)
    original_failure = reproduce_original_failure(m7_protocol_path)
    corrected, adapted_audit, delegated_stdout = corrected_full_verify(
        m7_protocol_path, diagnostics_path, m7_protocol
    )
    if adapted_audit != direct_audit:
        raise M7VerifierErratumError("VE1 adapter audit differs from direct audit")
    after = source_hashes(protocol)
    if before != after:
        raise M7VerifierErratumError("VE1 changed a frozen source artifact")
    return {
        "schema": "step3_cdro_opc_m7_verifier_erratum_receipt_v1",
        "erratum_route": ERRATUM_ROUTE,
        "underlying_m7_route": UNDERLYING_M7_ROUTE,
        "original_verify": original_failure,
        "representation_audit": direct_audit,
        "adapted_frozen_full_verify": corrected,
        "adapted_frozen_full_verify_stdout": delegated_stdout,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "official_attempt_index": 1,
        "known_pass_before_registration": True,
        "claim_scope": protocol["claim_scope"],
        "M6_stop_preserved": True,
        "original_artifact_writes": 0,
        "fresh_model_measurements": 0,
        "loss_writes": 0,
        "gradient_evaluations": 0,
        "optimizer_steps": 0,
        "parameter_writes": 0,
        "checkpoint_writes": 0,
        "candidate_generation": 0,
        "sealed_audit_opened": False,
        "training_authorized": False,
        "scientific_claim_expanded": False,
    }


def report_text(receipt: Mapping[str, Any]) -> str:
    audit = receipt["representation_audit"]
    return (
        "# CDR-OPC M7 Verifier Erratum VE1 Report\n\n"
        f"VE1 route: `{receipt['erratum_route']}`. Underlying frozen M7 route: "
        f"`{receipt['underlying_m7_route']}`.\n\n"
        "VE1 is retrospective, post-result, and non-blinded. The original "
        "failure and an exploratory in-memory normalized PASS were both known "
        "before VE1 registration; the exploratory PASS was not an official "
        "erratum attempt.\n\n"
        "The original M7 verifier outcome remains "
        f"`{receipt['original_verify']['status']}` with "
        f"`{receipt['original_verify']['error_message']}`. VE1 required exact "
        f"member-name sets for {audit['rows']} JSON rows, rebuilt only their "
        "in-memory mapping order, and delegated every remaining predicate to "
        "the frozen full M7 verifier. Mapping equality and canonical SHA-256 "
        "were unchanged; the exact CSV header-order predicate was retained.\n\n"
        "All frozen source SHA-256 values were identical before and after. "
        "VE1 performed zero original-artifact writes, model measurements, loss "
        "writes, gradient evaluations, optimizer steps, parameter writes, "
        "checkpoints, generations, or sealed-audit access. It does not expand "
        "the M7 scientific claim, and the M6 16/20 calibration STOP remains in "
        "force.\n"
    )


def runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "execution": "CPU_read_only_no_model_load",
    }


def claim_official_attempt(protocol_path: Path) -> dict[str, Any]:
    if protocol_path.resolve() != repo_path(DEFAULT_PROTOCOL):
        raise M7VerifierErratumError("VE1 official run requires the registered protocol")
    claim_path = repo_path(ATTEMPT_CLAIM_PATH)
    run_root = repo_path(RECEIPT_PATH).parent
    if claim_path.exists() or run_root.exists():
        raise M7VerifierErratumError("VE1 official attempt is already claimed")
    if git_output("status", "--porcelain"):
        raise M7VerifierErratumError("VE1 official run requires a clean worktree")
    runtime_commit = git_output("rev-parse", "HEAD")
    if (
        not git_commit_exists(runtime_commit)
        or not git_is_ancestor(ENTRY_COMMIT, runtime_commit)
    ):
        raise M7VerifierErratumError("VE1 runtime commit provenance differs")
    claim = {
        "schema": "step3_cdro_opc_m7_ve1_official_attempt_claim_v1",
        "status": "CLAIMED_BEFORE_ANY_OFFICIAL_AUDIT",
        "official_attempt_index": 1,
        "runtime_commit": runtime_commit,
        "protocol_path": DEFAULT_PROTOCOL,
        "protocol_sha256": file_sha256(protocol_path),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "worktree_clean_before_claim": True,
        "overwrite_allowed": False,
        "result_dependent_patch_or_rerun_allowed": False,
    }
    write_json_exclusive(claim_path, claim)
    return claim


def stop_route(stage: str, error: Exception) -> str:
    message = str(error)
    if (
        "frozen input differs" in message
        or "frozen M7 runtime provenance differs" in message
        or "changed a frozen source artifact" in message
        or "source-result Git blob differs" in message
        or "M7-runtime Git blob differs" in message
        or "entry evidence Git blob differs" in message
    ):
        return TARGET_DRIFT_ROUTE
    if stage == "full_verify":
        return FULL_VERIFY_ROUTE
    return SCOPE_INTEGRITY_ROUTE


def stop_report_text(receipt: Mapping[str, Any]) -> str:
    return (
        "# CDR-OPC M7 Verifier Erratum VE1 STOP Report\n\n"
        f"VE1 route: `{receipt['erratum_route']}`. The one registered official "
        "attempt was claimed before audit and is permanently closed.\n\n"
        f"Failure stage: `{receipt['failure']['stage']}`. Error: "
        f"`{receipt['failure']['error_class']}: "
        f"{receipt['failure']['error_message']}`.\n\n"
        "No original M7 artifact, model state, loss, gradient, optimizer state, "
        "checkpoint, generation, or sealed data was written or opened. The M6 "
        "STOP and original M7 claim boundary remain unchanged.\n"
    )


def publish_attempt(
    protocol_path: Path,
    protocol: Mapping[str, Any] | None,
    claim: Mapping[str, Any],
    receipt: Mapping[str, Any],
    report: str,
    input_validation_complete: bool,
) -> None:
    claim_path = repo_path(ATTEMPT_CLAIM_PATH)
    receipt_path = repo_path(RECEIPT_PATH)
    report_path = repo_path(REPORT_PATH)
    reproduction_path = repo_path(REPRODUCTION_PATH)
    run_root = receipt_path.parent
    if (
        not claim_path.is_file()
        or report_path.parent != run_root
        or reproduction_path.parent != run_root
        or run_root != repo_path(OUTPUT_ROOT) / "run"
        or run_root.exists()
        or input_validation_complete != (protocol is not None)
    ):
        raise M7VerifierErratumError("VE1 publication boundary differs")
    inputs_sha256 = (
        expected_input_hashes(protocol_path, protocol)
        if protocol is not None
        else {
            "protocol": file_sha256(protocol_path),
            "runner": file_sha256(Path(__file__).resolve()),
        }
    )
    root = repo_path(OUTPUT_ROOT)
    with tempfile.TemporaryDirectory(prefix=".ve1-stage-", dir=root) as stage:
        staged_run = Path(stage) / "run"
        staged_run.mkdir()
        staged_receipt = staged_run / receipt_path.name
        staged_report = staged_run / report_path.name
        staged_reproduction = staged_run / reproduction_path.name
        write_json(staged_receipt, receipt)
        staged_report.write_text(report, encoding="utf-8")
        reproduction = {
            "schema": "step3_cdro_opc_m7_verifier_erratum_reproduction_manifest_v1",
            "runtime_commit": claim["runtime_commit"],
            "environment": runtime_environment(),
            "inputs_sha256": inputs_sha256,
            "outputs_sha256": {
                "attempt_claim": file_sha256(claim_path),
                "receipt": file_sha256(staged_receipt),
                "report": file_sha256(staged_report),
            },
            "source_sha256_before": receipt.get("source_sha256_before", {}),
            "source_sha256_after": receipt.get("source_sha256_after", {}),
            "unique_reproduction_command": UNIQUE_REPRODUCTION_COMMAND,
            "verification_command": VERIFICATION_COMMAND,
            "official_attempt_index": 1,
            "official_attempt_status": (
                "completed_pass"
                if receipt["erratum_route"] == ERRATUM_ROUTE
                else "completed_stop"
            ),
            "erratum_route": receipt["erratum_route"],
            "input_validation_complete": input_validation_complete,
            "original_artifact_writes": 0,
            "fresh_model_measurements": 0,
            "optimizer_steps": 0,
            "sealed_audit_opened": False,
            "training_authorized": False,
            "scientific_claim_expanded": False,
        }
        write_json(staged_reproduction, reproduction)
        staged_run.replace(run_root)


def preflight(path: Path) -> dict[str, Any]:
    protocol = load_context(path)
    run_root = repo_path(RECEIPT_PATH).parent
    if run_root.exists() or repo_path(ATTEMPT_CLAIM_PATH).exists():
        raise M7VerifierErratumError("VE1 official output already exists")
    before = source_hashes(protocol)
    m7_protocol_path = repo_path(
        str(protocol["frozen_inputs"]["m7_protocol"]["path"])
    )
    diagnostics_path = repo_path(
        str(protocol["frozen_inputs"]["m7_diagnostics"]["path"])
    )
    m7_protocol = read_json(m7_protocol_path)
    _, audit = normalize_diagnostics(read_json(diagnostics_path), m7_protocol)
    validate_defect_fingerprint(protocol, audit)
    failure = reproduce_original_failure(m7_protocol_path)
    if source_hashes(protocol) != before:
        raise M7VerifierErratumError("VE1 preflight changed a source artifact")
    result = {
        "status": "PASS_CDRO_OPC_M7_VE1_PREFLIGHT",
        "original_verify_status": failure["status"],
        "rows": audit["rows"],
        "order_defect_rows": audit["order_defect_rows"],
        "source_artifact_writes": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def run(path: Path) -> dict[str, Any]:
    claim = claim_official_attempt(path)
    protocol: dict[str, Any] | None = None
    input_validation_complete = False
    stage = "scope_and_target_preflight"
    try:
        protocol = load_context(path)
        input_validation_complete = True
        validate_ve1_runtime_git_bindings(
            path, protocol, str(claim["runtime_commit"])
        )
        stage = "full_verify"
        receipt = perform_read_only_audit(path, protocol)
        publish_attempt(
            path,
            protocol,
            claim,
            receipt,
            report_text(receipt),
            input_validation_complete=True,
        )
    except Exception as exc:
        route = stop_route(stage, exc)
        receipt = {
            "schema": "step3_cdro_opc_m7_verifier_erratum_stop_receipt_v1",
            "erratum_route": route,
            "underlying_m7_route": UNDERLYING_M7_ROUTE,
            "official_attempt_index": 1,
            "official_attempt_closed": True,
            "input_validation_complete": input_validation_complete,
            "failure": {
                "stage": stage,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
            },
            "claim_scope": (
                protocol["claim_scope"]
                if protocol is not None
                else "scope_validation_incomplete"
            ),
            "M6_stop_preserved": True,
            "original_artifact_writes": 0,
            "fresh_model_measurements": 0,
            "loss_writes": 0,
            "gradient_evaluations": 0,
            "optimizer_steps": 0,
            "parameter_writes": 0,
            "checkpoint_writes": 0,
            "candidate_generation": 0,
            "sealed_audit_opened": False,
            "training_authorized": False,
            "scientific_claim_expanded": False,
        }
        publish_attempt(
            path,
            protocol,
            claim,
            receipt,
            stop_report_text(receipt),
            input_validation_complete=input_validation_complete,
        )
    print(
        json.dumps(
            {
                "erratum_route": receipt["erratum_route"],
                "underlying_m7_route": UNDERLYING_M7_ROUTE,
                "official_attempt_index": 1,
                "source_artifact_writes": 0,
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return receipt


def replay_recorded_stop_failure(
    path: Path,
    claim: Mapping[str, Any],
    stored_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    recorded_failure = stored_receipt["failure"]
    recorded_stage = str(recorded_failure["stage"])
    observed_protocol: dict[str, Any] | None = None
    input_validation_complete = False
    observed_error: Exception | None = None
    try:
        observed_protocol = load_context(path)
        input_validation_complete = True
        validate_ve1_runtime_git_bindings(
            path, observed_protocol, str(claim["runtime_commit"])
        )
        if recorded_stage == "full_verify":
            perform_read_only_audit(path, observed_protocol)
    except Exception as exc:
        observed_error = exc
    if observed_error is None:
        raise M7VerifierErratumError("VE1 recorded STOP failure no longer reproduces")
    if (
        type(observed_error).__name__ != recorded_failure["error_class"]
        or str(observed_error) != recorded_failure["error_message"]
        or input_validation_complete
        is not stored_receipt["input_validation_complete"]
    ):
        raise M7VerifierErratumError("VE1 recorded STOP failure differs")
    return observed_protocol, input_validation_complete


def verify_run(path: Path) -> dict[str, Any]:
    if path.resolve() != repo_path(DEFAULT_PROTOCOL):
        raise M7VerifierErratumError("VE1 verify-run requires the registered protocol")
    claim_path = repo_path(ATTEMPT_CLAIM_PATH)
    receipt_path = repo_path(RECEIPT_PATH)
    report_path = repo_path(REPORT_PATH)
    reproduction_path = repo_path(REPRODUCTION_PATH)
    if not all(
        item.is_file()
        for item in (claim_path, receipt_path, report_path, reproduction_path)
    ):
        raise M7VerifierErratumError("VE1 artifact set is incomplete")
    claim = read_json(claim_path)
    if (
        set(claim) != ATTEMPT_CLAIM_KEYS
        or claim.get("schema")
        != "step3_cdro_opc_m7_ve1_official_attempt_claim_v1"
        or claim.get("status") != "CLAIMED_BEFORE_ANY_OFFICIAL_AUDIT"
        or claim.get("official_attempt_index") != 1
        or claim.get("protocol_path") != DEFAULT_PROTOCOL
        or claim.get("protocol_sha256") != file_sha256(path)
        or claim.get("runner_sha256") != file_sha256(Path(__file__).resolve())
        or claim.get("worktree_clean_before_claim") is not True
        or claim.get("overwrite_allowed") is not False
        or claim.get("result_dependent_patch_or_rerun_allowed") is not False
        or not isinstance(claim.get("runtime_commit"), str)
        or not git_commit_exists(str(claim.get("runtime_commit")))
        or not git_is_ancestor(ENTRY_COMMIT, str(claim.get("runtime_commit")))
    ):
        raise M7VerifierErratumError("VE1 official attempt claim differs")
    stored_receipt = read_json(receipt_path)
    route = stored_receipt.get("erratum_route")
    terminal_stop_routes = {
        TARGET_DRIFT_ROUTE,
        SCOPE_INTEGRITY_ROUTE,
        FULL_VERIFY_ROUTE,
    }
    if route != ERRATUM_ROUTE and route not in terminal_stop_routes:
        raise M7VerifierErratumError("VE1 terminal route differs")
    reproduction = read_json(reproduction_path)
    if (
        set(reproduction) != REPRODUCTION_KEYS
        or reproduction.get("schema")
        != "step3_cdro_opc_m7_verifier_erratum_reproduction_manifest_v1"
        or reproduction.get("runtime_commit") != claim["runtime_commit"]
        or reproduction.get("environment") != runtime_environment()
        or reproduction.get("outputs_sha256")
        != {
            "attempt_claim": file_sha256(claim_path),
            "receipt": file_sha256(receipt_path),
            "report": file_sha256(report_path),
        }
        or reproduction.get("unique_reproduction_command")
        != UNIQUE_REPRODUCTION_COMMAND
        or reproduction.get("verification_command") != VERIFICATION_COMMAND
        or reproduction.get("official_attempt_index") != 1
        or reproduction.get("official_attempt_status")
        != ("completed_pass" if route == ERRATUM_ROUTE else "completed_stop")
        or reproduction.get("erratum_route") != route
        or reproduction.get("original_artifact_writes") != 0
        or reproduction.get("fresh_model_measurements") != 0
        or reproduction.get("optimizer_steps") != 0
        or reproduction.get("sealed_audit_opened") is not False
        or reproduction.get("training_authorized") is not False
        or reproduction.get("scientific_claim_expanded") is not False
    ):
        raise M7VerifierErratumError("VE1 reproduction manifest differs")
    if route == ERRATUM_ROUTE:
        protocol = load_context(path)
        validate_ve1_runtime_git_bindings(
            path, protocol, str(claim["runtime_commit"])
        )
        if (
            reproduction.get("input_validation_complete") is not True
            or reproduction.get("inputs_sha256")
            != expected_input_hashes(path, protocol)
        ):
            raise M7VerifierErratumError("VE1 PASS input provenance differs")
        expected_receipt = perform_read_only_audit(path, protocol)
        if stored_receipt != expected_receipt:
            raise M7VerifierErratumError("VE1 receipt does not reproduce")
        if report_path.read_text(encoding="utf-8") != report_text(expected_receipt):
            raise M7VerifierErratumError("VE1 report does not reproduce")
        if (
            reproduction.get("source_sha256_before")
            != expected_receipt["source_sha256_before"]
            or reproduction.get("source_sha256_after")
            != expected_receipt["source_sha256_after"]
        ):
            raise M7VerifierErratumError("VE1 source hash audit differs")
        result = {
            "status": "PASS_CDRO_OPC_M7_VE1_VERIFY_RUN",
            "erratum_route": ERRATUM_ROUTE,
            "underlying_m7_route": UNDERLYING_M7_ROUTE,
            "original_verify_status": ORIGINAL_VERIFY_STATUS,
            "source_artifact_writes": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
    else:
        failure = stored_receipt.get("failure")
        if (
            set(stored_receipt) != STOP_RECEIPT_KEYS
            or stored_receipt.get("schema")
            != "step3_cdro_opc_m7_verifier_erratum_stop_receipt_v1"
            or stored_receipt.get("underlying_m7_route") != UNDERLYING_M7_ROUTE
            or stored_receipt.get("official_attempt_index") != 1
            or stored_receipt.get("official_attempt_closed") is not True
            or not isinstance(failure, Mapping)
            or set(failure) != {"stage", "error_class", "error_message"}
            or not all(isinstance(failure[name], str) for name in failure)
            or failure["stage"]
            not in {"scope_and_target_preflight", "full_verify"}
            or type(stored_receipt.get("input_validation_complete")) is not bool
            or route
            != stop_route(
                str(failure["stage"]),
                M7VerifierErratumError(str(failure["error_message"])),
            )
            or stored_receipt.get("M6_stop_preserved") is not True
            or stored_receipt.get("original_artifact_writes") != 0
            or stored_receipt.get("fresh_model_measurements") != 0
            or stored_receipt.get("loss_writes") != 0
            or stored_receipt.get("gradient_evaluations") != 0
            or stored_receipt.get("optimizer_steps") != 0
            or stored_receipt.get("parameter_writes") != 0
            or stored_receipt.get("checkpoint_writes") != 0
            or stored_receipt.get("candidate_generation") != 0
            or stored_receipt.get("sealed_audit_opened") is not False
            or stored_receipt.get("training_authorized") is not False
            or stored_receipt.get("scientific_claim_expanded") is not False
        ):
            raise M7VerifierErratumError("VE1 STOP receipt differs")
        input_validation_complete = bool(
            stored_receipt["input_validation_complete"]
        )
        if reproduction.get("input_validation_complete") is not input_validation_complete:
            raise M7VerifierErratumError("VE1 STOP input validation status differs")
        observed_protocol, replay_input_validation = replay_recorded_stop_failure(
            path, claim, stored_receipt
        )
        if replay_input_validation is not input_validation_complete:
            raise M7VerifierErratumError("VE1 STOP replay boundary differs")
        if input_validation_complete:
            if observed_protocol is None:
                raise M7VerifierErratumError("VE1 STOP protocol replay differs")
            expected_inputs = expected_input_hashes(path, observed_protocol)
            expected_scope = observed_protocol.get("claim_scope")
        else:
            expected_inputs = {
                "protocol": file_sha256(path),
                "runner": file_sha256(Path(__file__).resolve()),
            }
            expected_scope = "scope_validation_incomplete"
        if (
            reproduction.get("inputs_sha256") != expected_inputs
            or reproduction.get("source_sha256_before") != {}
            or reproduction.get("source_sha256_after") != {}
            or stored_receipt.get("claim_scope") != expected_scope
            or report_path.read_text(encoding="utf-8")
            != stop_report_text(stored_receipt)
        ):
            raise M7VerifierErratumError("VE1 STOP artifacts do not reproduce")
        result = {
            "status": "PASS_CDRO_OPC_M7_VE1_TERMINAL_STOP_VERIFY_RUN",
            "erratum_route": route,
            "underlying_m7_route": UNDERLYING_M7_ROUTE,
            "official_attempt_index": 1,
            "source_artifact_writes": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main() -> None:
    args = parse_args()
    path = repo_path(args.protocol)
    {"preflight": preflight, "run": run, "verify-run": verify_run}[
        args.command
    ](path)


if __name__ == "__main__":
    main()
