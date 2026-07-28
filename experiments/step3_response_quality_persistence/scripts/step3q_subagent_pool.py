"""Freeze, seal, and import Step3Q Attempt 002 subagent authoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3q_calibrated_api_pool as step3q
import step3q_p2_review as p2


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments/step3_response_quality_persistence"
REGISTRY_PATH = EXPERIMENT_ROOT / "configs/step3q_subagent_pool_attempt_002_registry.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_FIELDS = ("opaque_request_id", "response")


class Step3QSubagentError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if b"\r" in data or (data and not data.endswith(b"\n")):
        raise Step3QSubagentError(f"JSONL must be LF-terminated: {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line:
            raise Step3QSubagentError(f"blank JSONL row: {path}:{index}")
        value = json.loads(line)
        if type(value) is not dict:
            raise Step3QSubagentError(f"non-object JSONL row: {path}:{index}")
        rows.append(value)
    return rows


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise Step3QSubagentError(f"path escapes repository: {value}") from exc
    return path


def check_file(spec: Mapping[str, Any]) -> Path:
    path = resolve_repo_path(str(spec["path"]))
    if path.is_symlink() or not path.is_file():
        raise Step3QSubagentError(f"missing frozen input: {path}")
    if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
        raise Step3QSubagentError(f"frozen input changed: {path}")
    return path


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    reg = read_json(path)
    if reg.get("schema") != "step3q_subagent_pool_registry_v1":
        raise Step3QSubagentError("unexpected Attempt 002 registry schema")
    if reg.get("status") != "prospective_before_subagent_launch_or_candidate_authoring":
        raise Step3QSubagentError("Attempt 002 registry is not prospective")
    if reg.get("external_api_called") is not False or reg.get("training_authorized") is not False:
        raise Step3QSubagentError("Attempt 002 crosses an authorization boundary")
    if reg.get("optimizer_steps") != 0:
        raise Step3QSubagentError("Attempt 002 optimizer boundary changed")
    for spec in reg.get("frozen_inputs", {}).values():
        if (
            type(spec) is not dict
            or set(spec) != {"path", "bytes", "sha256"}
            or type(spec["bytes"]) is not int
            or spec["bytes"] <= 0
            or not isinstance(spec["sha256"], str)
            or HEX64.fullmatch(spec["sha256"]) is None
        ):
            raise Step3QSubagentError("invalid frozen-input specification")
        check_file(spec)
    authoring = reg.get("authoring", {})
    roles = authoring.get("worker_roles")
    if (
        authoring.get("source") != "codex_multi_agent_v1"
        or authoring.get("model_provenance") != "inherited_parent_model_unversioned"
        or authoring.get("exact_provider_snapshot_available") is not False
        or type(roles) is not dict
        or set(roles) != {"specific_1", "specific_2", "mitigation_1", "mitigation_2"}
        or sum(int(value.get("rows", -1)) for value in roles.values()) != 120
        or authoring.get("fixed_requests") != 120
        or authoring.get("immutable_submission") is not True
        or authoring.get("retry_or_reassignment_allowed") is not False
    ):
        raise Step3QSubagentError("invalid subagent authoring contract")
    return reg


def source_bundle(reg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    p0 = step3q.load_registry(check_file(reg["frozen_inputs"]["p0_registry"]))
    p1_manifest_path = check_file(reg["frozen_inputs"]["p1_manifest"])
    p1_root = p1_manifest_path.parent
    step3q.verify_prepared_bundle(p0, p1_root)
    requests_path = check_file(reg["frozen_inputs"]["requests"])
    check_file(reg["frozen_inputs"]["prompts"])
    requests = read_jsonl(requests_path)
    if len(requests) != int(reg["authoring"]["fixed_requests"]):
        raise Step3QSubagentError("frozen request denominator changed")
    return p0, read_json(p1_manifest_path), requests


def assignment_rows(reg: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in requests:
        row = dict(raw)
        mode = row.get("author_mode")
        if mode not in {"specific_redirect", "risk_mitigation"}:
            raise Step3QSubagentError("frozen request has an invalid author mode")
        by_mode[mode].append(row)
    role_pairs = {
        "specific_redirect": ("specific_1", "specific_2"),
        "risk_mitigation": ("mitigation_1", "mitigation_2"),
    }
    assignments = {role: [] for role in reg["authoring"]["worker_roles"]}
    for mode, roles in role_pairs.items():
        ordered = sorted(by_mode[mode], key=lambda row: row["opaque_request_id"])
        for index, row in enumerate(ordered):
            assignments[roles[index % 2]].append(
                {
                    "opaque_request_id": row["opaque_request_id"],
                    "author_mode": row["author_mode"],
                    "prompt": row["prompt"],
                }
            )
    for role, rows in assignments.items():
        expected = reg["authoring"]["worker_roles"][role]
        if len(rows) != int(expected["rows"]) or any(
            row["author_mode"] != expected["author_mode"] for row in rows
        ):
            raise Step3QSubagentError(f"invalid deterministic assignment: {role}")
    return assignments


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, encoding="utf-8").strip()


def require_locked_clean(reg: Mapping[str, Any]) -> str:
    commit = reg.get("protocol_commit_sha1")
    if reg.get("protocol_lock_status") != "locked_to_protocol_commit" or not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise Step3QSubagentError("Attempt 002 protocol is not locked")
    if git("rev-parse", f"{reg['protocol_tag']}^{{commit}}") != commit:
        raise Step3QSubagentError("Attempt 002 protocol tag differs from registry lock")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT).returncode != 0:
        raise Step3QSubagentError("Attempt 002 protocol tag is not an ancestor of HEAD")
    if git("status", "--porcelain"):
        raise Step3QSubagentError("Attempt 002 execution requires a clean checkout")
    changed = git("diff", "--name-only", commit, "HEAD").splitlines()
    expected = [REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()]
    if changed != expected:
        raise Step3QSubagentError(f"protocol-to-lock diff changed: {changed}")
    return commit


def external_spool(path: Path, *, must_not_exist: bool = False) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise Step3QSubagentError("subagent spool must remain outside the repository")
    if must_not_exist and resolved.exists():
        raise Step3QSubagentError("subagent spool already exists")
    return resolved


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def preflight(reg: Mapping[str, Any]) -> dict[str, Any]:
    _, _, requests = source_bundle(reg)
    assignments = assignment_rows(reg, requests)
    return {
        "schema": "step3q_subagent_preflight_v1",
        "route_id": reg["route_id"],
        "fixed_requests": len(requests),
        "worker_rows": {role: len(rows) for role, rows in assignments.items()},
        "subagent_sessions_launched": 0,
        "prompt_text_released_by_preflight": False,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }


def prepare_spool(reg: Mapping[str, Any], spool_root: Path) -> dict[str, Any]:
    protocol_commit = require_locked_clean(reg)
    spool = external_spool(spool_root, must_not_exist=True)
    _, p1_manifest, requests = source_bundle(reg)
    assignments = assignment_rows(reg, requests)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spool.name}.", dir=spool.parent))
    try:
        artifacts: list[dict[str, Any]] = []
        instructions = check_file(reg["frozen_inputs"]["authoring_instructions"])
        destination = temporary / "instructions.md"
        destination.write_bytes(instructions.read_bytes())
        artifacts.append({"path": "instructions.md", "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
        for role, rows in sorted(assignments.items()):
            path = temporary / "assignments" / f"{role}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(jsonl_bytes(rows))
            artifacts.append({"path": f"assignments/{role}.jsonl", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "schema": "step3q_subagent_spool_manifest_v1",
            "route_id": reg["route_id"],
            "protocol_commit": protocol_commit,
            "lock_commit": git("rev-parse", "HEAD"),
            "p1_namespace_sha256": p1_manifest["namespace_sha256"],
            "artifacts": artifacts,
            "fixed_requests": len(requests),
            "worker_count": len(assignments),
            "external_api_called": False,
            "training_authorized": False,
            "optimizer_steps": 0,
        }
        (temporary / "spool_manifest.json").write_bytes(json_bytes(manifest))
        temporary.replace(spool)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {"route": "READY_SUBAGENT_AUTHORING", "spool_root": str(spool), "fixed_requests": 120, "workers": 4}


def verify_base_spool(reg: Mapping[str, Any], spool_root: Path) -> tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    spool = external_spool(spool_root)
    manifest = read_json(spool / "spool_manifest.json")
    if manifest.get("schema") != "step3q_subagent_spool_manifest_v1" or manifest.get("route_id") != reg["route_id"]:
        raise Step3QSubagentError("invalid subagent spool manifest")
    _, _, requests = source_bundle(reg)
    expected = assignment_rows(reg, requests)
    observed_paths = set()
    for artifact in manifest.get("artifacts", []):
        path = spool / artifact["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise Step3QSubagentError("subagent spool base artifact changed")
        observed_paths.add(artifact["path"])
    expected_paths = {"instructions.md", *(f"assignments/{role}.jsonl" for role in expected)}
    if observed_paths != expected_paths:
        raise Step3QSubagentError("subagent spool base artifact set changed")
    for role, rows in expected.items():
        if read_jsonl(spool / "assignments" / f"{role}.jsonl") != rows:
            raise Step3QSubagentError(f"subagent assignment changed: {role}")
    return spool, manifest, expected


def record_launch(reg: Mapping[str, Any], spool_root: Path, role: str, agent_id: str, nickname: str) -> dict[str, Any]:
    spool, _, assignments = verify_base_spool(reg, spool_root)
    if role not in assignments or not agent_id.strip():
        raise Step3QSubagentError("invalid subagent launch record")
    value = {
        "schema": "step3q_subagent_launch_v1",
        "route_id": reg["route_id"],
        "worker_role": role,
        "agent_id": agent_id,
        "nickname": nickname,
        "fork_parent_context": False,
        "model_provenance": reg["authoring"]["model_provenance"],
        "exact_provider_snapshot_available": False,
        "assignment_sha256": sha256_file(spool / "assignments" / f"{role}.jsonl"),
    }
    write_new(spool / "launches" / f"{role}.json", json_bytes(value))
    return {"worker_role": role, "agent_id": agent_id, "recorded": True}


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_worker_output(reg: Mapping[str, Any], assignment: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != len(assignment):
        raise Step3QSubagentError("worker output row count mismatch")
    result: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, (raw, expected) in enumerate(zip(rows, assignment), 1):
        if tuple(raw) != OUTPUT_FIELDS or raw.get("opaque_request_id") != expected["opaque_request_id"]:
            raise Step3QSubagentError(f"worker output schema or order mismatch at row {index}")
        response = raw.get("response")
        if type(response) is not str or not response or response != response.strip():
            raise Step3QSubagentError(f"worker response is empty or unstripped at row {index}")
        tokens = len(response.split())
        if not int(reg["authoring"]["minimum_response_whitespace_tokens"]) <= tokens <= int(reg["authoring"]["maximum_response_whitespace_tokens"]):
            raise Step3QSubagentError(f"worker response token envelope failure at row {index}")
        normalized_hash = sha256_bytes(normalize_text(response).encode("utf-8"))
        if normalized_hash in seen_hashes:
            raise Step3QSubagentError("worker output repeats a normalized response")
        seen_hashes.add(normalized_hash)
        result.append({"opaque_request_id": raw["opaque_request_id"], "response": response})
    return result


def submit_worker(reg: Mapping[str, Any], spool_root: Path, role: str, agent_id: str, input_path: Path) -> dict[str, Any]:
    spool, _, assignments = verify_base_spool(reg, spool_root)
    if role not in assignments:
        raise Step3QSubagentError("unknown worker role")
    launch = read_json(spool / "launches" / f"{role}.json")
    if launch.get("agent_id") != agent_id:
        raise Step3QSubagentError("worker agent ID differs from launch record")
    rows = validate_worker_output(reg, assignments[role], read_jsonl(input_path))
    output_bytes = jsonl_bytes(rows)
    write_new(spool / "workers" / f"{role}.jsonl", output_bytes)
    manifest = {
        "schema": "step3q_subagent_worker_submission_v1",
        "route_id": reg["route_id"],
        "worker_role": role,
        "agent_id": agent_id,
        "model_provenance": reg["authoring"]["model_provenance"],
        "assignment_sha256": launch["assignment_sha256"],
        "output_sha256": sha256_bytes(output_bytes),
        "rows": len(rows),
        "immutable_submission": True,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    write_new(spool / "workers" / f"{role}_manifest.json", json_bytes(manifest))
    return {"route": "WORKER_SUBMISSION_ACCEPTED", "worker_role": role, "rows": len(rows), "output_sha256": manifest["output_sha256"]}


def collect_candidates(reg: Mapping[str, Any], spool_root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    spool, spool_manifest, assignments = verify_base_spool(reg, spool_root)
    _, p1_manifest, requests = source_bundle(reg)
    request_by_id = {row["opaque_request_id"]: row for row in requests}
    launches: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for role in sorted(assignments):
        launch = read_json(spool / "launches" / f"{role}.json")
        submission = read_json(spool / "workers" / f"{role}_manifest.json")
        output_path = spool / "workers" / f"{role}.jsonl"
        rows = validate_worker_output(reg, assignments[role], read_jsonl(output_path))
        if (
            launch.get("worker_role") != role
            or submission.get("worker_role") != role
            or submission.get("agent_id") != launch.get("agent_id")
            or submission.get("output_sha256") != sha256_file(output_path)
            or submission.get("rows") != len(rows)
        ):
            raise Step3QSubagentError(f"worker provenance mismatch: {role}")
        launches.append(launch)
        for row in rows:
            request = request_by_id[row["opaque_request_id"]]
            response = row["response"]
            candidates.append(
                {
                    "opaque_request_id": request["opaque_request_id"],
                    "opaque_candidate_id": request["opaque_candidate_id"],
                    "opaque_prompt_id": request["opaque_prompt_id"],
                    "author_mode": request["author_mode"],
                    "prompt": request["prompt"],
                    "response": response,
                    "response_whitespace_tokens": len(response.split()),
                    "response_sha256": sha256_bytes(response.encode("utf-8")),
                    "normalized_response_sha256": sha256_bytes(normalize_text(response).encode("utf-8")),
                    "authoring_source": reg["authoring"]["source"],
                    "authoring_worker": role,
                    "authoring_agent_id": launch["agent_id"],
                    "model_provenance": reg["authoring"]["model_provenance"],
                }
            )
    if len({row["agent_id"] for row in launches}) != len(assignments):
        raise Step3QSubagentError("subagent launch IDs are not distinct")
    if len(candidates) != len(requests) or {row["opaque_request_id"] for row in candidates} != set(request_by_id):
        raise Step3QSubagentError("worker submissions do not cover the fixed request denominator")
    return spool, p1_manifest, candidates


def sealed_files(reg: Mapping[str, Any], p1_manifest: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, bytes], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["normalized_response_sha256"]].append(dict(row))
    retained: list[dict[str, Any]] = []
    for values in groups.values():
        values.sort(key=lambda row: row["opaque_request_id"])
        retained.append(values[0])
    retained.sort(key=lambda row: row["opaque_request_id"])
    valid_prompts = {row["opaque_prompt_id"] for row in retained}
    passed = len(retained) >= int(reg["authoring"]["minimum_valid_unique_candidates"]) and len(valid_prompts) >= int(reg["authoring"]["minimum_valid_prompts"])
    summary = {
        "schema": "step3q_subagent_authoring_summary_v1",
        "route_id": reg["route_id"],
        "route": "PASS_SUBAGENT_AUTHORING" if passed else "STOP_SUBAGENT_YIELD",
        "fixed_requests": len(candidates),
        "terminal_authoring_records": len(candidates),
        "valid_before_deduplication": len(candidates),
        "valid_unique_candidates": len(retained),
        "valid_prompts": len(valid_prompts),
        "normalized_duplicate_failures": len(candidates) - len(retained),
        "subagent_sessions": len({row["authoring_agent_id"] for row in candidates}),
        "model_provenance": [reg["authoring"]["model_provenance"]],
        "exact_provider_snapshot_available": False,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    record_manifest = {
        "schema": "step3q_subagent_record_manifest_v1",
        "p1_namespace_sha256": p1_manifest["namespace_sha256"],
        "records": [
            {
                "opaque_request_id": row["opaque_request_id"],
                "response_sha256": row["response_sha256"],
                "authoring_worker": row["authoring_worker"],
                "authoring_agent_id": row["authoring_agent_id"],
            }
            for row in sorted(candidates, key=lambda value: value["opaque_request_id"])
        ],
    }
    return {
        "candidates.jsonl": jsonl_bytes(retained),
        "authoring_summary.json": json_bytes(summary),
        "record_manifest.json": json_bytes(record_manifest),
    }, summary


def seal_spool(reg: Mapping[str, Any], spool_root: Path) -> dict[str, Any]:
    spool, p1_manifest, candidates = collect_candidates(reg, spool_root)
    files, summary = sealed_files(reg, p1_manifest, candidates)
    sealed = spool / "sealed"
    if sealed.exists():
        raise Step3QSubagentError("subagent spool is already sealed")
    staging = Path(tempfile.mkdtemp(prefix=".step3q-subagent-seal-", dir=spool))
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        staging.replace(sealed)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def verify_spool(reg: Mapping[str, Any], spool_root: Path) -> dict[str, Any]:
    spool, p1_manifest, candidates = collect_candidates(reg, spool_root)
    expected, summary = sealed_files(reg, p1_manifest, candidates)
    sealed = spool / "sealed"
    if set(path.name for path in sealed.iterdir() if path.is_file()) != set(expected):
        raise Step3QSubagentError("sealed subagent file set changed")
    for name, data in expected.items():
        if (sealed / name).read_bytes() != data:
            raise Step3QSubagentError(f"sealed subagent artifact changed: {name}")
    return {**summary, "verified": True}


def build_import_files(reg: Mapping[str, Any], spool_root: Path, protocol_commit: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    spool, p1_manifest, candidates = collect_candidates(reg, spool_root)
    sealed, summary = sealed_files(reg, p1_manifest, candidates)
    if summary["route"] != "PASS_SUBAGENT_AUTHORING":
        raise Step3QSubagentError("subagent authoring did not pass")
    for name, data in sealed.items():
        if (spool / "sealed" / name).read_bytes() != data:
            raise Step3QSubagentError("subagent spool differs from reconstruction")
    p0 = step3q.load_registry(check_file(reg["frozen_inputs"]["p0_registry"]))
    sources = p2._load_p1_review_sources(reg)
    primary = p2.enrich_primary_candidates(json.loads("[" + ",".join((sealed["candidates.jsonl"].decode("utf-8").splitlines())) + "]"), sources, p0)
    namespace = p2.p2_namespace(reg, p1_manifest, protocol_commit, REGISTRY_PATH)
    duplicate_count = int(reg["review_packets"]["hidden_primary_duplicates_per_reviewer"])
    packages = {
        reviewer: p2.build_reviewer_packet(primary, sources, reviewer, namespace, duplicate_count)
        for reviewer in p2.REVIEWERS_AB
    }
    rubric_bytes = check_file(reg["frozen_inputs"]["review_rubric"]).read_bytes()
    files: dict[str, bytes] = {f"import/sealed/{name}": data for name, data in sealed.items()}
    files["private/primary_candidates_hidden.jsonl"] = p2.jsonl_text(primary).encode("utf-8")
    for reviewer in p2.REVIEWERS_AB:
        packet = packages[reviewer]["packet"]
        packet_bytes = p2.jsonl_text(packet).encode("utf-8")
        files[f"packets/{reviewer}_packet.jsonl"] = packet_bytes
        files[f"private/{reviewer}_hidden_map.jsonl"] = p2.jsonl_text(packages[reviewer]["hidden_map"]).encode("utf-8")
        files[f"templates/{reviewer}_review.jsonl"] = p2.jsonl_text(p2._review_template(packet)).encode("utf-8")
        files[f"templates/{reviewer}_provenance.json"] = p2.json_text(
            p2._provenance_template(reg, reviewer, packet_bytes, rubric_bytes, len(packet))
        ).encode("utf-8")
    import_summary = {
        "schema": "step3q_subagent_p2_import_summary_v1",
        "route_id": reg["route_id"],
        "protocol_commit": protocol_commit,
        "namespace_sha256": namespace,
        "authoring_route": summary["route"],
        "authoring_records_verified": summary["terminal_authoring_records"],
        "subagent_sessions_verified": summary["subagent_sessions"],
        "primary_candidates": len(primary),
        "primary_prompts": len({row["opaque_prompt_id"] for row in primary}),
        "model_provenance": summary["model_provenance"],
        "exact_provider_snapshot_available": False,
        "review_rows_per_reviewer": len(packages["reviewer_a"]["packet"]),
        "sentinels_per_reviewer": 24,
        "duplicates_per_reviewer": duplicate_count,
        "candidate_text_inspected_by_main_executor": False,
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["import/import_summary.json"] = json_bytes(import_summary)
    manifest = {
        "schema": "step3q_subagent_p2_import_packet_manifest_v1",
        "route_id": reg["route_id"],
        "stage_id": reg["stage_id"],
        "entry_commit": reg["entry_commit"],
        "protocol_commit": protocol_commit,
        "namespace_sha256": namespace,
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "artifacts": p2._manifest_artifacts(files),
        "external_api_called": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["packet_manifest.json"] = json_bytes(manifest)
    return files, import_summary


def import_packets(reg: Mapping[str, Any], spool_root: Path) -> dict[str, Any]:
    protocol_commit = require_locked_clean(reg)
    verify_spool(reg, spool_root)
    files, summary = build_import_files(reg, spool_root, protocol_commit)
    output_root = resolve_repo_path(reg["outputs"]["root"])
    p2.write_bundle(output_root, files)
    return {"output_root": str(output_root), **summary}


def verify_import(reg: Mapping[str, Any], spool_root: Path) -> dict[str, Any]:
    protocol_commit = reg["protocol_commit_sha1"]
    files, summary = build_import_files(reg, spool_root, protocol_commit)
    output_root = resolve_repo_path(reg["outputs"]["root"])
    actual = {path.relative_to(output_root).as_posix() for path in output_root.rglob("*") if path.is_file()}
    if actual != set(files):
        raise Step3QSubagentError("Attempt 002 import file set changed")
    for name, data in files.items():
        if (output_root / name).read_bytes() != data:
            raise Step3QSubagentError(f"Attempt 002 import artifact changed: {name}")
    return {**summary, "artifact_files": len(files), "verified": True}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    for name in ("prepare-spool", "seal", "verify-spool", "import-packets", "verify-import"):
        command = sub.add_parser(name)
        command.add_argument("--spool-root", required=True)
    launch = sub.add_parser("record-launch")
    launch.add_argument("--spool-root", required=True)
    launch.add_argument("--worker", required=True)
    launch.add_argument("--agent-id", required=True)
    launch.add_argument("--nickname", default="")
    submit = sub.add_parser("submit-worker")
    submit.add_argument("--spool-root", required=True)
    submit.add_argument("--worker", required=True)
    submit.add_argument("--agent-id", required=True)
    submit.add_argument("--input", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    reg = load_registry()
    if args.command == "preflight":
        result = preflight(reg)
    elif args.command == "prepare-spool":
        result = prepare_spool(reg, Path(args.spool_root))
    elif args.command == "record-launch":
        result = record_launch(reg, Path(args.spool_root), args.worker, args.agent_id, args.nickname)
    elif args.command == "submit-worker":
        result = submit_worker(reg, Path(args.spool_root), args.worker, args.agent_id, Path(args.input))
    elif args.command == "seal":
        result = seal_spool(reg, Path(args.spool_root))
    elif args.command == "verify-spool":
        result = verify_spool(reg, Path(args.spool_root))
    elif args.command == "import-packets":
        result = import_packets(reg, Path(args.spool_root))
    else:
        result = verify_import(reg, Path(args.spool_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
