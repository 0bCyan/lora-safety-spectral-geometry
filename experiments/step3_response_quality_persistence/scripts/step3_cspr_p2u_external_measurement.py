"""Freeze calibration-only views for prospective CSPR P2U human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/4070ti_step3_cspr_p2u/"
    "protocol/cspr_p2u_external_human_measurement_protocol.json"
)


class P2UValidationError(ValueError):
    """Raised when a P2U artifact violates the prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "prepare-calibration"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise P2UValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_raw(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    data = path.read_bytes()
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise P2UValidationError(f"JSONL is not canonical LF: {relative(path)}")
    rows = []
    for index, raw in enumerate(data.splitlines(keepends=True), start=1):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise P2UValidationError(
                f"invalid JSONL row {index}: {relative(path)}"
            ) from exc
        if not isinstance(value, dict):
            raise P2UValidationError(f"non-object JSONL row: {relative(path)}")
        rows.append((raw, value))
    return rows


def write_frozen_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_frozen(path: Path, text: str) -> None:
    write_frozen_bytes(path, text.encode("utf-8"))


def write_json_frozen(path: Path, value: Any) -> None:
    write_frozen(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def unique_map(rows: list[tuple[bytes, dict[str, Any]]], id_field: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for _, row in rows:
        row_id = row.get(id_field)
        if not isinstance(row_id, str) or not row_id or row_id in mapped:
            raise P2UValidationError(f"invalid or duplicate {id_field}")
        mapped[row_id] = row
    return mapped


def calibration_subsequence(
    packet_rows: list[tuple[bytes, dict[str, Any]]],
    template_rows: list[tuple[bytes, dict[str, Any]]],
    key_rows: list[tuple[bytes, dict[str, Any]]],
    id_field: str,
    decision_fields: tuple[str, ...],
    expected_primary: int,
    expected_duplicates: int,
) -> tuple[bytes, bytes, dict[str, int]]:
    packet = unique_map(packet_rows, id_field)
    template = unique_map(template_rows, id_field)
    key = unique_map(key_rows, id_field)
    if set(packet) != set(template) or set(packet) != set(key):
        raise P2UValidationError("packet, template, and hidden-key ID sets differ")

    packet_order = [row[id_field] for _, row in packet_rows]
    template_order = [row[id_field] for _, row in template_rows]
    if packet_order != template_order:
        raise P2UValidationError("packet/template order mismatch")

    packet_schema = {id_field, "prompt", "response"}
    template_schema = {id_field, *decision_fields, "short_reason"}
    for row in packet.values():
        if set(row) != packet_schema:
            raise P2UValidationError("packet schema mismatch")
    for row in template.values():
        if set(row) != template_schema:
            raise P2UValidationError("template schema mismatch")
        if any(row[field] != "" for field in (*decision_fields, "short_reason")):
            raise P2UValidationError("template decision fields are not blank")

    selected = {
        row_id
        for row_id, row in key.items()
        if row.get("item_type") == "calibration_anchor"
    }
    primary = sum(not bool(key[row_id].get("is_hidden_duplicate")) for row_id in selected)
    duplicates = sum(bool(key[row_id].get("is_hidden_duplicate")) for row_id in selected)
    if primary != expected_primary or duplicates != expected_duplicates:
        raise P2UValidationError("calibration primary or duplicate count mismatch")

    packet_output = b"".join(raw for raw, row in packet_rows if row[id_field] in selected)
    template_output = b"".join(raw for raw, row in template_rows if row[id_field] in selected)
    selected_packet_ids = [
        row[id_field] for _, row in packet_rows if row[id_field] in selected
    ]
    selected_template_ids = [
        row[id_field] for _, row in template_rows if row[id_field] in selected
    ]
    if selected_packet_ids != selected_template_ids:
        raise P2UValidationError("calibration packet/template order mismatch")
    if len(selected_packet_ids) != expected_primary + expected_duplicates:
        raise P2UValidationError("calibration output row count mismatch")
    return packet_output, template_output, {
        "primary_rows": primary,
        "hidden_duplicate_rows": duplicates,
        "total_rows": len(selected_packet_ids),
    }


def context(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(path)
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise P2UValidationError(f"frozen input hash mismatch: {name}")

    p2t = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p2t_analysis"]["path"])
    )
    if p2t.get("route") != "STOP_CSPR_P2T_CALIBRATION_UNRELIABLE":
        raise P2UValidationError("P2U does not descend from the terminal P2T route")
    if p2t.get("reviewer_c_artifacts") != []:
        raise P2UValidationError("P2T Reviewer C artifacts unexpectedly exist")
    if p2t.get("optimizer_steps") != 0 or p2t.get("training_authorized") is not False:
        raise P2UValidationError("P2T terminal record violates zero-training boundary")

    manifest = read_json(
        resolve_repo_path(protocol["frozen_inputs"]["p2s_packet_manifest"]["path"])
    )
    for artifact in manifest["artifacts"]:
        artifact_path = resolve_repo_path(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise P2UValidationError(f"packet artifact hash mismatch: {artifact['path']}")

    if not git_is_ancestor(protocol["entry_commit"]):
        raise P2UValidationError("P2U entry commit is not an ancestor of HEAD")
    if protocol.get("reviewer_channel") != "external_human_only":
        raise P2UValidationError("P2U reviewer channel is not external-human-only")
    if protocol.get("automated_model_review_authorized") is not False:
        raise P2UValidationError("automated model review must remain unauthorized")
    if protocol.get("optimizer_steps") != 0 or protocol.get("training_authorized") is not False:
        raise P2UValidationError("P2U violates the zero-training boundary")
    return protocol, manifest


def preflight(path: Path) -> dict[str, Any]:
    protocol, manifest = context(path)
    expected = protocol["calibration_design"]
    result = {
        "status": "PASS_CSPR_P2U_P0_PREFLIGHT",
        "entry_commit": protocol["entry_commit"],
        "safety_parent_rows": manifest["safety_packet_rows_per_reviewer"],
        "actionability_parent_rows": manifest["action_packet_rows_per_reviewer"],
        "safety_calibration_rows_per_reviewer": expected["safety"]["total_rows"],
        "actionability_calibration_rows_per_reviewer": expected["actionability"]["total_rows"],
        "target_packets_written": False,
        "human_reviewers_qualified": False,
        "automated_model_review_authorized": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def prepare_calibration(path: Path) -> dict[str, Any]:
    protocol, _ = context(path)
    artifacts = []
    construct_counts: dict[str, dict[str, int]] = {}
    for construct in ("safety", "actionability"):
        spec = protocol["calibration_views"][construct]
        design = protocol["calibration_design"][construct]
        key_rows = read_jsonl_raw(resolve_repo_path(spec["hidden_key"]))
        for role in ("a", "b"):
            role_spec = spec[f"reviewer_{role}"]
            source_packet = resolve_repo_path(role_spec["source_packet"])
            source_template = resolve_repo_path(role_spec["source_template"])
            output_packet = resolve_repo_path(role_spec["output_packet"])
            output_template = resolve_repo_path(role_spec["output_template"])
            packet_bytes, template_bytes, counts = calibration_subsequence(
                read_jsonl_raw(source_packet),
                read_jsonl_raw(source_template),
                key_rows,
                spec["id_field"],
                tuple(spec["decision_fields"]),
                int(design["primary_rows"]),
                int(design["hidden_duplicate_rows"]),
            )
            write_frozen_bytes(output_packet, packet_bytes)
            write_frozen_bytes(output_template, template_bytes)
            construct_counts.setdefault(construct, counts)
            for kind, output, source in (
                ("packet", output_packet, source_packet),
                ("template", output_template, source_template),
            ):
                artifacts.append(
                    {
                        "construct": construct,
                        "reviewer_role": role,
                        "kind": kind,
                        "path": relative(output),
                        "sha256": sha256(output),
                        "bytes": output.stat().st_size,
                        "rows": counts["total_rows"],
                        "source_path": relative(source),
                        "source_sha256": sha256(source),
                        "exact_ordered_byte_subsequence": True,
                    }
                )

    manifest = {
        "name": "step3_cspr_p2u_calibration_view_manifest_v1",
        "route": "P2U0_CALIBRATION_VIEWS_FROZEN_REVIEWER_QUALIFICATION_ONLY",
        "entry_commit": protocol["entry_commit"],
        "construct_counts": construct_counts,
        "artifacts": artifacts,
        "calibration_rows_only": True,
        "target_packets_written": False,
        "response_text_transformed": False,
        "human_reviewer_access_authorized": False,
        "automated_model_review_authorized": False,
        "hidden_keys_reviewer_visible": False,
        "old_p2t_labels_eligible": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    manifest_path = resolve_repo_path(protocol["outputs"]["calibration_manifest"])
    write_json_frozen(manifest_path, manifest)
    report = """# CSPR P2U P0 Calibration View Freeze\n\nRoute: `P2U0_CALIBRATION_VIEWS_FROZEN_REVIEWER_QUALIFICATION_ONLY`.\n\nThe reviewer-specific Safety and Actionability calibration views were written as exact ordered byte subsequences of the frozen P2S packets and blank templates. They contain only the 40+4 Safety and 60+6 Actionability calibration rows. No target packet, label, or Reviewer C artifact was created.\n\nOnly zero-access external-human qualification is authorized after this package is committed and pushed. Automated model review, target access, gradients, and training remain unauthorized. `optimizer_steps=0` and `training_authorized=false`.\n"""
    write_frozen(resolve_repo_path(protocol["outputs"]["freeze_report"]), report)
    print(
        json.dumps(
            {
                "route": manifest["route"],
                "artifacts": len(artifacts),
                "target_packets_written": False,
            },
            sort_keys=True,
        )
    )
    return manifest


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    try:
        if args.command == "preflight":
            preflight(path)
        else:
            prepare_calibration(path)
    except (
        FileExistsError,
        KeyError,
        OSError,
        P2UValidationError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "STOP_CSPR_P2U_P0_INVALID", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
