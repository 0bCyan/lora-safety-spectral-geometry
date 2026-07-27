#!/usr/bin/env python3
"""Import, blind-review, and select the frozen Step3Q API target pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import step3o_o4_qualification as o4q
import step3p_public_paired_pool as step3p
import step3q_calibrated_api_pool as step3q


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "experiments"
    / "step3_response_quality_persistence"
    / "configs"
    / "step3q_p2_review_registry.json"
)

REVIEWERS_AB = ("reviewer_a", "reviewer_b")
VISIBLE_FIELDS = ("opaque_candidate_id", "prompt", "response")
DECISION_FIELDS = tuple(o4q.DECISION_FIELDS)
BINARY_FIELDS = tuple(o4q.BINARY_FIELDS)
COMPONENT_FIELDS = tuple(step3q.COMPONENT_FIELDS)
REVIEW_FIELDS = ("opaque_candidate_id", *DECISION_FIELDS, "short_reason")
HIDDEN_MAP_FIELDS = (
    "opaque_candidate_id",
    "item_role",
    "canonical_candidate_id",
    "duplicate_parent_id",
)
PROVENANCE_FIELDS = {
    "schema",
    "route_id",
    "stage_id",
    "role",
    "reviewer_id",
    "model",
    "context_isolated",
    "packet_sha256",
    "rubric_sha256",
    "review_sha256",
    "row_count",
    "prohibited_access",
    "saw_hidden_map",
    "saw_api_mode",
    "saw_calibration_roles",
    "saw_other_reviews",
}
REVIEW_ID = re.compile(r"^step3q_review_[0-9a-f]{24}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Step3QP2ValidationError(ValueError):
    """Raised when a frozen Step3Q P2 boundary is violated."""


def _reject_constant(value: str) -> None:
    raise Step3QP2ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Step3QP2ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Step3QP2ValidationError(f"invalid {context}: {exc}") from exc


def read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Step3QP2ValidationError(f"cannot read JSON object {path}: {exc}") from exc
    value = parse_json(text, str(path))
    if type(value) is not dict:
        raise Step3QP2ValidationError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Step3QP2ValidationError(f"cannot read JSONL {path}: {exc}") from exc
    if b"\r\n" in data or b"\r" in data:
        raise Step3QP2ValidationError(f"{path} must use LF line endings")
    if data and not data.endswith(b"\n"):
        raise Step3QP2ValidationError(f"{path} lacks final LF termination")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise Step3QP2ValidationError(f"blank JSONL row at {path}:{line_number}")
        value = parse_json(line, f"{path}:{line_number}")
        if type(value) is not dict:
            raise Step3QP2ValidationError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(namespace: str, purpose: str, *parts: object) -> str:
    material = "\x1f".join([namespace, purpose, *(str(part) for part in parts)])
    return sha256_bytes(material.encode("utf-8"))


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise Step3QP2ValidationError(f"path escapes repository: {value}") from exc
    return path


def check_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise Step3QP2ValidationError(f"missing frozen input: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise Step3QP2ValidationError(f"byte-size mismatch for frozen input: {path}")
    if sha256_file(path) != expected_sha256:
        raise Step3QP2ValidationError(f"SHA-256 mismatch for frozen input: {path}")


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def git_head() -> str:
    value = _git_output("rev-parse", "HEAD")
    if not HEX40.fullmatch(value):
        raise Step3QP2ValidationError("git HEAD is not a full lowercase SHA-1")
    return value


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        raise Step3QP2ValidationError("git ancestry check failed")
    return result.returncode == 0


def require_protocol_head(registry_path: Path, registry: Mapping[str, Any]) -> str:
    head = git_head()
    status = _git_output("status", "--porcelain")
    if status:
        raise Step3QP2ValidationError("P2 import requires a clean protocol checkout")
    if not git_is_ancestor(str(registry["entry_commit"]), head):
        raise Step3QP2ValidationError("P2 protocol HEAD does not descend from the frozen P1 entry")
    tag_commit = protocol_tag_commit(registry)
    if not git_is_ancestor(tag_commit, head):
        raise Step3QP2ValidationError("P2 import checkout does not descend from the frozen protocol commit")
    try:
        relative = registry_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise Step3QP2ValidationError("P2 registry must be inside the repository") from exc
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed != registry_path.read_bytes():
        raise Step3QP2ValidationError("P2 registry bytes are not frozen in protocol HEAD")
    revision_count = int(_git_output("rev-list", "--count", f"{tag_commit}..{head}"))
    parents = _git_output("rev-list", "--parents", "-n", "1", head).split()
    if revision_count != 1 or parents != [head, tag_commit]:
        raise Step3QP2ValidationError(
            "P2 registry lock must be the sole direct child of the protocol commit"
        )
    changed = subprocess.run(
        ["git", "diff", "--name-only", tag_commit, head],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed != [relative]:
        raise Step3QP2ValidationError(
            "P2 import checkout may differ from the protocol tag only by the locked registry"
        )
    tagged_registry_bytes = subprocess.run(
        ["git", "show", f"{tag_commit}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    try:
        tagged_registry_value = parse_json(
            tagged_registry_bytes.decode("utf-8"), "tagged pending P2 registry"
        )
    except UnicodeDecodeError as exc:
        raise Step3QP2ValidationError("tagged pending P2 registry is not UTF-8") from exc
    if type(tagged_registry_value) is not dict:
        raise Step3QP2ValidationError("tagged pending P2 registry is not an object")
    expected_locked_registry = dict(tagged_registry_value)
    if (
        expected_locked_registry.get("protocol_commit_sha1")
        != "TO_BE_LOCKED_AFTER_PROTOCOL_COMMIT"
        or expected_locked_registry.get("protocol_lock_status")
        != "pending_protocol_commit"
    ):
        raise Step3QP2ValidationError("tagged P2 registry was not in the pending lock state")
    expected_locked_registry["protocol_commit_sha1"] = tag_commit
    expected_locked_registry["protocol_lock_status"] = "locked_to_protocol_commit"
    if dict(registry) != expected_locked_registry:
        raise Step3QP2ValidationError(
            "P2 lock commit changed registry fields beyond the two lock fields"
        )
    return tag_commit


def protocol_tag_commit(registry: Mapping[str, Any]) -> str:
    tag = registry.get("protocol_tag")
    if type(tag) is not str or tag != "step3q-calibrated-api-attempt-001-p2-protocol":
        raise Step3QP2ValidationError("P2 protocol tag changed")
    expected = registry.get("protocol_commit_sha1")
    if (
        registry.get("protocol_lock_status") != "locked_to_protocol_commit"
        or type(expected) is not str
        or not HEX40.fullmatch(expected)
    ):
        raise Step3QP2ValidationError("P2 protocol commit has not been locked in the registry")
    try:
        value = _git_output("rev-parse", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise Step3QP2ValidationError("P2 protocol tag is missing") from exc
    if value != expected:
        raise Step3QP2ValidationError("P2 protocol tag moved away from the registry-locked commit")
    return value


def require_frozen_stage_checkout(
    registry_path: Path,
    registry: Mapping[str, Any],
    protocol_commit: str,
) -> None:
    if _git_output("status", "--porcelain"):
        raise Step3QP2ValidationError("P2 downstream stages require a clean checkout")
    if registry_path.is_symlink():
        raise Step3QP2ValidationError("P2 registry cannot be a symlink")
    try:
        registry_relative = registry_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise Step3QP2ValidationError("P2 registry escapes the repository") from exc
    registry_blob = subprocess.run(
        ["git", "show", f"HEAD:{registry_relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if registry_blob != registry_path.read_bytes():
        raise Step3QP2ValidationError("P2 locked registry bytes differ from HEAD")
    tag_commit = protocol_tag_commit(registry)
    if protocol_commit != tag_commit or not git_is_ancestor(tag_commit, git_head()):
        raise Step3QP2ValidationError("P2 downstream stage is not anchored to the protocol tag")
    critical_paths = [
        Path(__file__).resolve(),
        Path(step3q.__file__).resolve(),
        Path(step3p.__file__).resolve(),
        Path(o4q.__file__).resolve(),
    ]
    relative_paths: list[str] = []
    for path in critical_paths:
        try:
            relative_paths.append(path.relative_to(REPO_ROOT.resolve()).as_posix())
        except ValueError as exc:
            raise Step3QP2ValidationError("critical P2 implementation path escapes repository") from exc
    result = subprocess.run(
        ["git", "diff", "--quiet", registry["protocol_tag"], "--", *relative_paths],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise Step3QP2ValidationError("P2 implementation differs from the frozen protocol tag")


def require_bundle_committed(root: Path, manifest: Mapping[str, Any], manifest_name: str) -> None:
    try:
        relative_root = root.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise Step3QP2ValidationError("P2 stage bundle must be committed inside the repository") from exc
    paths = [manifest_name, *(artifact["path"] for artifact in manifest["artifacts"])]
    for path in paths:
        local_path = root / path
        if local_path.is_symlink() or not local_path.is_file():
            raise Step3QP2ValidationError(f"P2 stage artifact is not a regular file: {path}")
        relative = (relative_root / Path(path)).as_posix()
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise Step3QP2ValidationError(f"P2 stage artifact is not committed at HEAD: {relative}")
        if result.stdout != local_path.read_bytes():
            raise Step3QP2ValidationError(f"P2 stage artifact bytes differ from HEAD: {relative}")


def load_registry(path: Path = DEFAULT_REGISTRY) -> tuple[dict[str, Any], dict[str, Any]]:
    registry_path = path.resolve()
    if path.is_symlink():
        raise Step3QP2ValidationError("P2 registry cannot be a symlink")
    reg = read_json(registry_path)
    if reg.get("schema") != "step3q_p2_review_registry_v1":
        raise Step3QP2ValidationError("unexpected Step3Q P2 registry schema")
    if reg.get("status") != "prospective_before_external_receipts_or_candidate_labels_are_opened":
        raise Step3QP2ValidationError("P2 registry is not prospective")
    if reg.get("training_authorized") is not False or reg.get("optimizer_steps") != 0:
        raise Step3QP2ValidationError("P2 registry crosses the frozen training boundary")
    if not isinstance(reg.get("entry_commit"), str) or not HEX40.fullmatch(reg["entry_commit"]):
        raise Step3QP2ValidationError("P2 registry has an invalid P1 entry commit")
    if reg.get("protocol_tag") != "step3q-calibrated-api-attempt-001-p2-protocol":
        raise Step3QP2ValidationError("P2 registry has an invalid protocol tag")
    protocol_commit = reg.get("protocol_commit_sha1")
    protocol_lock_status = reg.get("protocol_lock_status")
    pending_lock = (
        protocol_commit == "TO_BE_LOCKED_AFTER_PROTOCOL_COMMIT"
        and protocol_lock_status == "pending_protocol_commit"
    )
    completed_lock = (
        type(protocol_commit) is str
        and HEX40.fullmatch(protocol_commit) is not None
        and protocol_lock_status == "locked_to_protocol_commit"
    )
    if not pending_lock and not completed_lock:
        raise Step3QP2ValidationError("P2 protocol lock fields are inconsistent")

    expected_inputs = {"p0_registry", "p1_manifest", "review_rubric"}
    inputs = reg.get("frozen_inputs")
    if type(inputs) is not dict or set(inputs) != expected_inputs:
        raise Step3QP2ValidationError("P2 frozen-input registry is incomplete")
    for name, spec in inputs.items():
        if type(spec) is not dict or set(spec) != {"path", "bytes", "sha256"}:
            raise Step3QP2ValidationError(f"invalid frozen-input specification: {name}")
        if type(spec["bytes"]) is not int or spec["bytes"] <= 0 or not HEX64.fullmatch(spec["sha256"]):
            raise Step3QP2ValidationError(f"invalid frozen-input hash specification: {name}")
        check_file(resolve_repo_path(spec["path"]), spec["sha256"], spec["bytes"])

    dispatch = reg.get("dispatch_import")
    if (
        type(dispatch) is not dict
        or dispatch.get("required_route") != "PASS_API_AUTHORING"
        or dispatch.get("fixed_requests") != 120
        or dispatch.get("minimum_valid_unique_candidates") != 96
        or dispatch.get("minimum_valid_prompts") != 54
        or dispatch.get("sealed_files")
        != ["candidates.jsonl", "dispatch_summary.json", "receipt_manifest.json"]
        or dispatch.get("spool_must_be_outside_repository") is not True
        or dispatch.get("copy_raw_receipts_into_repository") is not False
    ):
        raise Step3QP2ValidationError("P2 dispatch-import contract changed")

    packets = reg.get("review_packets")
    if (
        type(packets) is not dict
        or packets.get("roles") != ["reviewer_a", "reviewer_b"]
        or packets.get("visible_fields") != list(VISIBLE_FIELDS)
        or packets.get("hidden_primary_duplicates_per_reviewer") != 12
        or packets.get("hidden_sentinels_per_reviewer") != 24
        or packets.get("uniform_packet_id_prefix") != "step3q_review_"
        or any(
            packets.get(name) is not True
            for name in (
                "reviewer_specific_id_remapping",
                "reviewer_specific_packet_order",
                "sentinel_and_duplicate_roles_hidden",
                "api_mode_hidden",
            )
        )
    ):
        raise Step3QP2ValidationError("P2 review-packet contract changed")

    provenance = reg.get("review_provenance")
    if type(provenance) is not dict or provenance.get("distinct_reviewer_ids_required") is not True:
        raise Step3QP2ValidationError("P2 reviewer-provenance contract is incomplete")
    for name in (
        "context_isolated_required",
        "prohibited_access_required_false",
        "saw_hidden_map_required_false",
        "saw_api_mode_required_false",
        "saw_calibration_roles_required_false",
        "saw_other_reviews_required_false",
    ):
        if provenance.get(name) is not True:
            raise Step3QP2ValidationError(f"P2 provenance condition changed: {name}")
    expected_stage_order = {
        "calibration_before_primary_metrics",
        "primary_metrics_forbidden_on_calibration_failure",
        "reviewer_c_only_after_ab_pass",
        "reviewer_c_only_for_derived_qualification_disagreements",
        "target_flow_only_after_valid_adjudication",
    }
    if (
        type(reg.get("stage_order")) is not dict
        or set(reg["stage_order"]) != expected_stage_order
        or not all(value is True for value in reg["stage_order"].values())
    ):
        raise Step3QP2ValidationError("P2 stage-order gates must all remain enabled")
    outputs = reg.get("outputs")
    if (
        type(outputs) is not dict
        or set(outputs) != {"root"}
        or outputs["root"]
        != "experiments/step3_response_quality_persistence/results/step3q_calibrated_api_pool_attempt_001_p2_review"
    ):
        raise Step3QP2ValidationError("P2 output root changed")
    resolve_repo_path(outputs["root"])

    p0_path = resolve_repo_path(inputs["p0_registry"]["path"])
    p0 = step3q.load_registry(p0_path)
    if p0.get("route_id") != reg.get("route_id"):
        raise Step3QP2ValidationError("P0 and P2 route IDs differ")
    if (
        dispatch["minimum_valid_unique_candidates"]
        != p0["authoring"]["minimum_valid_unique_candidates"]
        or dispatch["minimum_valid_prompts"] != p0["authoring"]["minimum_valid_prompts"]
        or packets["hidden_primary_duplicates_per_reviewer"]
        != p0["calibration"]["hidden_primary_duplicates_per_reviewer"]
        or packets["hidden_sentinels_per_reviewer"]
        != p0["calibration"]["sentinels_per_primary_reviewer"]
        or inputs["review_rubric"]["path"] != p0["inputs"]["review_rubric"]["path"]
        or inputs["review_rubric"]["sha256"] != p0["inputs"]["review_rubric"]["sha256"]
    ):
        raise Step3QP2ValidationError("P2 denominators differ from the frozen P0 registry")
    prepared_root = resolve_repo_path(inputs["p1_manifest"]["path"]).parent
    verification = step3q.verify_prepared_bundle(p0, prepared_root)
    if verification.get("api_requests") != dispatch["fixed_requests"]:
        raise Step3QP2ValidationError("P1 request denominator differs from P2")
    return reg, p0


def prepared_root(reg: Mapping[str, Any]) -> Path:
    return resolve_repo_path(reg["frozen_inputs"]["p1_manifest"]["path"]).parent


def p2_namespace(
    reg: Mapping[str, Any], p1_manifest: Mapping[str, Any], protocol_commit: str, registry_path: Path
) -> str:
    if not HEX40.fullmatch(protocol_commit):
        raise Step3QP2ValidationError("P2 protocol commit must be a full lowercase SHA-1")
    material = "\x1f".join(
        [
            reg["route_id"],
            reg["stage_id"],
            reg["entry_commit"],
            protocol_commit,
            p1_manifest["namespace_sha256"],
            sha256_file(registry_path),
            *(reg["frozen_inputs"][name]["sha256"] for name in sorted(reg["frozen_inputs"])),
        ]
    )
    return sha256_bytes(material.encode("utf-8"))


def write_bundle(output_root: Path, files: Mapping[str, bytes]) -> None:
    if output_root.exists():
        raise Step3QP2ValidationError(f"immutable output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".step3q-p2-", dir=output_root.parent))
    try:
        for relative, content in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or relative.replace("\\", "/") != path.as_posix():
                raise Step3QP2ValidationError(f"invalid bundle path: {relative}")
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        staging.replace(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _manifest_artifacts(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": len(content), "sha256": sha256_bytes(content)}
        for name, content in sorted(files.items())
    ]


def _verify_manifest_bundle(
    root: Path,
    manifest_name: str,
    expected_schema: str,
    *,
    ignored_top_level: Sequence[str] = (),
) -> tuple[dict[str, Any], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise Step3QP2ValidationError(f"immutable bundle root is not a regular directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Step3QP2ValidationError(f"immutable bundle contains a symlink: {path}")
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise Step3QP2ValidationError(f"missing immutable manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != expected_schema:
        raise Step3QP2ValidationError(f"unexpected immutable manifest schema: {manifest_path}")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list:
        raise Step3QP2ValidationError("immutable manifest lacks artifacts")
    observed: set[str] = set()
    for artifact in artifacts:
        artifact_path = Path(artifact["path"]) if type(artifact) is dict and type(artifact.get("path")) is str else None
        if (
            type(artifact) is not dict
            or set(artifact) != {"path", "bytes", "sha256"}
            or type(artifact["path"]) is not str
            or artifact_path is None
            or artifact_path.is_absolute()
            or ".." in artifact_path.parts
            or artifact["path"].replace("\\", "/") != artifact_path.as_posix()
            or artifact["path"] in observed
            or type(artifact["bytes"]) is not int
            or artifact["bytes"] < 0
            or not isinstance(artifact["sha256"], str)
            or not HEX64.fullmatch(artifact["sha256"])
        ):
            raise Step3QP2ValidationError("invalid artifact record in immutable manifest")
        observed.add(artifact["path"])
        check_file(root / artifact["path"], artifact["sha256"], artifact["bytes"])
    ignored = set(ignored_top_level)
    actual = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in ignored:
            continue
        actual.add(relative.as_posix())
    if actual != observed | {manifest_name}:
        raise Step3QP2ValidationError("immutable bundle file set differs from its manifest")
    return manifest, observed


def verify_external_dispatch(
    reg: Mapping[str, Any], p0: Mapping[str, Any], spool_root: Path
) -> dict[str, Any]:
    spool = step3q.require_external_spool_root(spool_root)
    p1_root = prepared_root(reg)
    step3q.verify_prepared_bundle(p0, p1_root)
    p1_manifest = read_json(p1_root / "protocol_manifest.json")
    requests = read_jsonl(p1_root / "authoring" / "api_requests.jsonl")
    if len(requests) != int(reg["dispatch_import"]["fixed_requests"]):
        raise Step3QP2ValidationError("P1 request denominator changed")
    request_ids = [row.get("opaque_request_id") for row in requests]
    if any(type(value) is not str or not value for value in request_ids) or len(set(request_ids)) != len(request_ids):
        raise Step3QP2ValidationError("P1 request IDs are missing or duplicated")

    receipt_dir = spool / "receipts"
    sealed_root = spool / "sealed"
    if (
        receipt_dir.is_symlink()
        or sealed_root.is_symlink()
        or not receipt_dir.is_dir()
        or not sealed_root.is_dir()
    ):
        raise Step3QP2ValidationError("external spool lacks receipts or sealed output")
    receipt_entries = list(receipt_dir.iterdir())
    expected_receipt_names = {f"{request_id}.json" for request_id in request_ids}
    if any(path.is_symlink() or not path.is_file() for path in receipt_entries) or {path.name for path in receipt_entries} != expected_receipt_names:
        raise Step3QP2ValidationError("external receipt directory differs from the fixed denominator")
    expected_sealed = set(reg["dispatch_import"]["sealed_files"])
    sealed_entries = list(sealed_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in sealed_entries) or {path.name for path in sealed_entries} != expected_sealed:
        raise Step3QP2ValidationError("external sealed file set changed")

    instructions_path = resolve_repo_path(p0["inputs"]["authoring_instructions_path"])
    instructions = instructions_path.read_text(encoding="utf-8")
    if sha256_file(instructions_path) != p1_manifest["authoring_instructions"]["sha256"]:
        raise Step3QP2ValidationError("authoring instructions differ from P1 manifest")
    receipts: list[dict[str, Any]] = []
    for request in requests:
        path = receipt_dir / f"{request['opaque_request_id']}.json"
        receipt = read_json(path)
        _, request_sha, payload_sha = step3q.expected_receipt_hashes(request, p0, instructions)
        try:
            validated = step3q.validate_receipt(receipt, request, p0, request_sha, payload_sha)
        except step3q.Step3QValidationError as exc:
            raise Step3QP2ValidationError(str(exc)) from exc
        receipts.append(validated)

    external_bytes = {name: (sealed_root / name).read_bytes() for name in expected_sealed}
    with tempfile.TemporaryDirectory(prefix="step3q-p2-recompute-") as temporary:
        recomputed_root = Path(temporary) / "sealed"
        try:
            recomputed_summary = step3q.seal_dispatch_receipts(
                p0,
                p1_manifest,
                requests,
                receipts,
                receipt_dir,
                recomputed_root,
                instructions,
            )
        except step3q.Step3QValidationError as exc:
            raise Step3QP2ValidationError(str(exc)) from exc
        for name in expected_sealed:
            if external_bytes[name] != (recomputed_root / name).read_bytes():
                raise Step3QP2ValidationError(f"external sealed artifact differs from recomputation: {name}")

    summary = parse_json(external_bytes["dispatch_summary.json"].decode("utf-8"), "dispatch summary")
    if type(summary) is not dict or summary != recomputed_summary:
        raise Step3QP2ValidationError("dispatch summary differs from recomputed result")
    if (
        summary.get("route") != reg["dispatch_import"]["required_route"]
        or summary.get("fixed_requests") != reg["dispatch_import"]["fixed_requests"]
        or summary.get("terminal_receipts") != reg["dispatch_import"]["fixed_requests"]
        or summary.get("valid_unique_candidates", 0)
        < reg["dispatch_import"]["minimum_valid_unique_candidates"]
        or summary.get("valid_prompts", 0) < reg["dispatch_import"]["minimum_valid_prompts"]
        or summary.get("model_provenance_pass") is not True
        or type(summary.get("effective_models")) is not list
        or len(summary["effective_models"]) != 1
        or summary.get("status_counts", {}).get("protocol_failure", 0) != 0
    ):
        raise Step3QP2ValidationError("external dispatch does not satisfy PASS_API_AUTHORING")

    candidates = parse_jsonl_bytes(external_bytes["candidates.jsonl"], "sealed candidates")
    receipt_manifest = parse_json(
        external_bytes["receipt_manifest.json"].decode("utf-8"), "receipt manifest"
    )
    if type(receipt_manifest) is not dict:
        raise Step3QP2ValidationError("receipt manifest is not an object")
    return {
        "p1_manifest": p1_manifest,
        "requests": requests,
        "candidates": candidates,
        "summary": summary,
        "receipt_manifest": receipt_manifest,
        "sealed_bytes": external_bytes,
    }


def parse_jsonl_bytes(data: bytes, context: str) -> list[dict[str, Any]]:
    if b"\r" in data or (data and not data.endswith(b"\n")):
        raise Step3QP2ValidationError(f"{context} must be LF-terminated JSONL")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Step3QP2ValidationError(f"invalid UTF-8 in {context}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line:
            raise Step3QP2ValidationError(f"blank row in {context}:{index}")
        value = parse_json(line, f"{context}:{index}")
        if type(value) is not dict:
            raise Step3QP2ValidationError(f"non-object row in {context}:{index}")
        rows.append(value)
    return rows


def _load_p1_review_sources(reg: Mapping[str, Any]) -> dict[str, Any]:
    root = prepared_root(reg)
    prompts = read_jsonl(root / "authoring" / "prompts_hidden.jsonl")
    calibration_packet = read_jsonl(root / "calibration" / "calibration_bank_packet.jsonl")
    calibration_reference = read_jsonl(root / "calibration" / "calibration_reference_hidden.jsonl")
    allocations = read_json(root / "calibration" / "sentinel_allocations_hidden.json")
    prompt_by_id = {row.get("opaque_prompt_id"): row for row in prompts}
    packet_by_id = {row.get("opaque_candidate_id"): row for row in calibration_packet}
    reference_by_id = {row.get("opaque_candidate_id"): row for row in calibration_reference}
    if (
        None in prompt_by_id
        or len(prompt_by_id) != len(prompts)
        or None in packet_by_id
        or len(packet_by_id) != len(calibration_packet)
        or None in reference_by_id
        or len(reference_by_id) != len(calibration_reference)
        or set(packet_by_id) != set(reference_by_id)
        or set(allocations) != set(REVIEWERS_AB)
    ):
        raise Step3QP2ValidationError("P1 prompt or calibration identifiers are invalid")
    for role in REVIEWERS_AB:
        selected = allocations[role]
        if type(selected) is not list or len(selected) != 24 or len(set(selected)) != 24:
            raise Step3QP2ValidationError(f"invalid P1 sentinel allocation for {role}")
        if not set(selected) <= set(reference_by_id):
            raise Step3QP2ValidationError(f"P1 sentinel allocation is out of bank for {role}")
        for sentinel_id in selected:
            reference = reference_by_id[sentinel_id]
            if reference.get("calibration_role") == "prompt_mismatch" and reference.get(
                "matched_anchor_candidate_id"
            ) not in selected:
                raise Step3QP2ValidationError(f"P1 mismatch pair is not closed for {role}")
    return {
        "prompts": prompts,
        "prompt_by_id": prompt_by_id,
        "calibration_packet": calibration_packet,
        "calibration_packet_by_id": packet_by_id,
        "calibration_reference": calibration_reference,
        "calibration_reference_by_id": reference_by_id,
        "allocations": allocations,
    }


def enrich_primary_candidates(
    candidates: Sequence[Mapping[str, Any]], p1_sources: Mapping[str, Any], p0: Mapping[str, Any]
) -> list[dict[str, Any]]:
    prompt_by_id = p1_sources["prompt_by_id"]
    categories = set(p0["selection"]["category_severity_slots"])
    primary: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    for index, raw in enumerate(candidates, 1):
        row = dict(raw)
        candidate_id = row.get("opaque_candidate_id")
        prompt_id = row.get("opaque_prompt_id")
        metadata = prompt_by_id.get(prompt_id)
        if (
            type(candidate_id) is not str
            or not candidate_id
            or candidate_id in candidate_ids
            or type(metadata) is not dict
            or row.get("prompt") != metadata.get("prompt")
        ):
            raise Step3QP2ValidationError(f"invalid sealed candidate identity at row {index}")
        row_categories = metadata.get("all_source_categories")
        assigned = metadata.get("assigned_category")
        severity = metadata.get("severity_level")
        normalized_hash = row.get("normalized_response_sha256")
        if (
            type(row_categories) is not list
            or not row_categories
            or len(set(row_categories)) != len(row_categories)
            or not set(row_categories) <= categories
            or assigned not in row_categories
            or type(severity) is not int
            or severity not in {1, 2, 3}
            or type(normalized_hash) is not str
            or not HEX64.fullmatch(normalized_hash)
            or normalized_hash in normalized_hashes
        ):
            raise Step3QP2ValidationError(f"invalid candidate coverage metadata at row {index}")
        candidate_ids.add(candidate_id)
        normalized_hashes.add(normalized_hash)
        primary.append(
            {
                **row,
                "sample_id": metadata["sample_id"],
                "source_index": metadata["source_index"],
                "severity_level": severity,
                "categories": [assigned],
                "all_source_categories": sorted(row_categories),
                "assigned_category": assigned,
                "prompt_sha256": metadata["prompt_sha256"],
            }
        )
    if len(primary) < int(p0["authoring"]["minimum_valid_unique_candidates"]):
        raise Step3QP2ValidationError("sealed primary candidate count is below the frozen minimum")
    if len({row["opaque_prompt_id"] for row in primary}) < int(
        p0["authoring"]["minimum_valid_prompts"]
    ):
        raise Step3QP2ValidationError("sealed primary prompt count is below the frozen minimum")
    return primary


def _review_id(namespace: str, reviewer: str, purpose: str, *parts: object) -> str:
    return "step3q_review_" + stable_rank(namespace, purpose, reviewer, *parts)[:24]


def build_reviewer_packet(
    primary: Sequence[Mapping[str, Any]],
    p1_sources: Mapping[str, Any],
    reviewer: str,
    namespace: str,
    duplicate_count: int,
) -> dict[str, list[dict[str, Any]]]:
    if reviewer not in REVIEWERS_AB:
        raise Step3QP2ValidationError(f"invalid A/B reviewer role: {reviewer}")
    visible: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    primary_review_id: dict[str, str] = {}
    used_ids: set[str] = set()

    def add_item(
        *,
        item_role: str,
        canonical_id: str,
        prompt: str,
        response: str,
        purpose: str,
        duplicate_parent_id: str | None = None,
    ) -> str:
        review_id = _review_id(namespace, reviewer, purpose, canonical_id)
        if review_id in used_ids or not REVIEW_ID.fullmatch(review_id):
            raise Step3QP2ValidationError("review ID collision or schema violation")
        used_ids.add(review_id)
        visible.append(
            {"opaque_candidate_id": review_id, "prompt": prompt, "response": response}
        )
        hidden.append(
            {
                "opaque_candidate_id": review_id,
                "item_role": item_role,
                "canonical_candidate_id": canonical_id,
                "duplicate_parent_id": duplicate_parent_id,
            }
        )
        return review_id

    for row in primary:
        canonical_id = row["opaque_candidate_id"]
        primary_review_id[canonical_id] = add_item(
            item_role="primary",
            canonical_id=canonical_id,
            prompt=row["prompt"],
            response=row["response"],
            purpose="primary-review-id",
        )

    packet_by_id = p1_sources["calibration_packet_by_id"]
    for canonical_id in p1_sources["allocations"][reviewer]:
        row = packet_by_id[canonical_id]
        add_item(
            item_role="sentinel",
            canonical_id=canonical_id,
            prompt=row["prompt"],
            response=row["response"],
            purpose="sentinel-review-id",
        )

    duplicate_sources = sorted(
        (row["opaque_candidate_id"] for row in primary),
        key=lambda value: stable_rank(namespace, "duplicate-source", reviewer, value),
    )[:duplicate_count]
    if len(duplicate_sources) != duplicate_count:
        raise Step3QP2ValidationError(f"insufficient primary rows for hidden duplicates: {reviewer}")
    primary_by_id = {row["opaque_candidate_id"]: row for row in primary}
    for canonical_id in duplicate_sources:
        row = primary_by_id[canonical_id]
        add_item(
            item_role="duplicate",
            canonical_id=canonical_id,
            prompt=row["prompt"],
            response=row["response"],
            purpose="duplicate-review-id",
            duplicate_parent_id=primary_review_id[canonical_id],
        )

    order = sorted(
        range(len(visible)),
        key=lambda index: stable_rank(
            namespace, "packet-order", reviewer, visible[index]["opaque_candidate_id"]
        ),
    )
    visible = [visible[index] for index in order]
    hidden = [hidden[index] for index in order]
    packet_bytes = jsonl_text(visible).encode("utf-8")
    for forbidden in (
        b"step3q_cal_",
        b"step3q_candidate_",
        b"specific_redirect",
        b"risk_mitigation",
    ):
        if forbidden in packet_bytes:
            raise Step3QP2ValidationError("review packet leaks a hidden construction role")
    if any(tuple(row) != VISIBLE_FIELDS for row in visible):
        raise Step3QP2ValidationError("review packet visible schema changed")
    return {"packet": visible, "hidden_map": hidden}


def _review_template(packet: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "opaque_candidate_id": row["opaque_candidate_id"],
            "detail_retention": None,
            "safe_route": "",
            "executable_step": "",
            "prompt_binding": "",
            "naturalness": "",
            "clean_safe": "",
            "short_reason": "",
        }
        for row in packet
    ]


def _provenance_template(
    reg: Mapping[str, Any], role: str, packet_bytes: bytes, rubric_bytes: bytes, row_count: int
) -> dict[str, Any]:
    rubric_sha256 = sha256_bytes(rubric_bytes)
    if rubric_sha256 != reg["frozen_inputs"]["review_rubric"]["sha256"]:
        raise Step3QP2ValidationError("review rubric bytes differ from the P2 registry")
    return {
        "schema": "step3q_p2_reviewer_provenance_v1",
        "route_id": reg["route_id"],
        "stage_id": reg["stage_id"],
        "role": role,
        "reviewer_id": "",
        "model": "",
        "context_isolated": True,
        "packet_sha256": sha256_bytes(packet_bytes),
        "rubric_sha256": rubric_sha256,
        "review_sha256": "",
        "row_count": row_count,
        "prohibited_access": False,
        "saw_hidden_map": False,
        "saw_api_mode": False,
        "saw_calibration_roles": False,
        "saw_other_reviews": False,
    }


def build_import_files(
    reg: Mapping[str, Any],
    p0: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    protocol_commit: str,
    registry_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    p1_sources = _load_p1_review_sources(reg)
    primary = enrich_primary_candidates(dispatch["candidates"], p1_sources, p0)
    namespace = p2_namespace(reg, dispatch["p1_manifest"], protocol_commit, registry_path)
    duplicate_count = int(reg["review_packets"]["hidden_primary_duplicates_per_reviewer"])
    packages = {
        reviewer: build_reviewer_packet(
            primary, p1_sources, reviewer, namespace, duplicate_count
        )
        for reviewer in REVIEWERS_AB
    }
    map_a = {
        row["canonical_candidate_id"]: row["opaque_candidate_id"]
        for row in packages["reviewer_a"]["hidden_map"]
        if row["item_role"] == "primary"
    }
    map_b = {
        row["canonical_candidate_id"]: row["opaque_candidate_id"]
        for row in packages["reviewer_b"]["hidden_map"]
        if row["item_role"] == "primary"
    }
    if set(map_a) != set(map_b) or any(map_a[key] == map_b[key] for key in map_a):
        raise Step3QP2ValidationError("A/B primary IDs are not reviewer-specific")

    rubric_bytes = resolve_repo_path(reg["frozen_inputs"]["review_rubric"]["path"]).read_bytes()
    files: dict[str, bytes] = {
        f"import/sealed/{name}": content
        for name, content in dispatch["sealed_bytes"].items()
    }
    files["private/primary_candidates_hidden.jsonl"] = jsonl_text(primary).encode("utf-8")
    for reviewer in REVIEWERS_AB:
        packet = packages[reviewer]["packet"]
        hidden_map = packages[reviewer]["hidden_map"]
        packet_bytes = jsonl_text(packet).encode("utf-8")
        files[f"packets/{reviewer}_packet.jsonl"] = packet_bytes
        files[f"private/{reviewer}_hidden_map.jsonl"] = jsonl_text(hidden_map).encode("utf-8")
        files[f"templates/{reviewer}_review.jsonl"] = jsonl_text(
            _review_template(packet)
        ).encode("utf-8")
        files[f"templates/{reviewer}_provenance.json"] = json_text(
            _provenance_template(reg, reviewer, packet_bytes, rubric_bytes, len(packet))
        ).encode("utf-8")

    import_summary = {
        "schema": "step3q_p2_dispatch_import_summary_v1",
        "route_id": reg["route_id"],
        "protocol_commit": protocol_commit,
        "namespace_sha256": namespace,
        "dispatch_route": dispatch["summary"]["route"],
        "terminal_receipts_verified": dispatch["summary"]["terminal_receipts"],
        "primary_candidates": len(primary),
        "primary_prompts": len({row["opaque_prompt_id"] for row in primary}),
        "effective_models": dispatch["summary"]["effective_models"],
        "review_rows_per_reviewer": len(packages["reviewer_a"]["packet"]),
        "sentinels_per_reviewer": 24,
        "duplicates_per_reviewer": duplicate_count,
        "raw_receipts_committed": False,
        "external_api_called": True,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["import/import_summary.json"] = json_text(import_summary).encode("utf-8")
    manifest = {
        "schema": "step3q_p2_import_packet_manifest_v1",
        "route_id": reg["route_id"],
        "stage_id": reg["stage_id"],
        "entry_commit": reg["entry_commit"],
        "protocol_commit": protocol_commit,
        "namespace_sha256": namespace,
        "p2_registry_sha256": sha256_file(registry_path),
        "artifacts": _manifest_artifacts(files),
        "external_api_called": True,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["packet_manifest.json"] = json_text(manifest).encode("utf-8")
    return files, import_summary


def load_import_bundle(output_root: Path, reg: Mapping[str, Any]) -> dict[str, Any]:
    manifest, _ = _verify_manifest_bundle(
        output_root,
        "packet_manifest.json",
        "step3q_p2_import_packet_manifest_v1",
        ignored_top_level=("ab_outcome", "finalization"),
    )
    if (
        manifest.get("route_id") != reg["route_id"]
        or manifest.get("entry_commit") != reg["entry_commit"]
        or manifest.get("training_authorized") is not False
        or manifest.get("optimizer_steps") != 0
    ):
        raise Step3QP2ValidationError("P2 import manifest violates the frozen route")
    primary = read_jsonl(output_root / "private" / "primary_candidates_hidden.jsonl")
    result: dict[str, Any] = {
        "manifest": manifest,
        "manifest_sha256": sha256_file(output_root / "packet_manifest.json"),
        "primary": primary,
    }
    for reviewer in REVIEWERS_AB:
        result[f"packet_{reviewer}"] = read_jsonl(
            output_root / "packets" / f"{reviewer}_packet.jsonl"
        )
        result[f"hidden_{reviewer}"] = read_jsonl(
            output_root / "private" / f"{reviewer}_hidden_map.jsonl"
        )
    return result


def verify_import_reconstruction(
    output_root: Path,
    bundle: Mapping[str, Any],
    reg: Mapping[str, Any],
    p0: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    registry_path: Path,
) -> None:
    protocol_commit = bundle["manifest"].get("protocol_commit")
    if type(protocol_commit) is not str or not HEX40.fullmatch(protocol_commit):
        raise Step3QP2ValidationError("import manifest has an invalid protocol commit")
    expected, _ = build_import_files(
        reg, p0, dispatch, protocol_commit, registry_path
    )
    actual_paths = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
        and path.relative_to(output_root).parts[0] not in {"ab_outcome", "finalization"}
    }
    if actual_paths != set(expected):
        raise Step3QP2ValidationError("committed import bundle differs from receipt reconstruction")
    for relative, content in expected.items():
        if (output_root / relative).read_bytes() != content:
            raise Step3QP2ValidationError(
                f"committed import artifact differs from receipt reconstruction: {relative}"
            )


def validate_review_rows(
    rows: Sequence[Mapping[str, Any]], packet: Sequence[Mapping[str, Any]], role: str
) -> list[dict[str, Any]]:
    expected_ids = [row["opaque_candidate_id"] for row in packet]
    if len(rows) != len(expected_ids):
        raise Step3QP2ValidationError(f"{role} review row count mismatch")
    validated: list[dict[str, Any]] = []
    for index, (raw, expected_id) in enumerate(zip(rows, expected_ids), 1):
        row = dict(raw)
        if tuple(row) != REVIEW_FIELDS or row.get("opaque_candidate_id") != expected_id:
            raise Step3QP2ValidationError(f"{role} review schema or order mismatch at row {index}")
        try:
            step3q.validate_review_row(row)
        except step3q.Step3QValidationError as exc:
            raise Step3QP2ValidationError(f"{role}: {exc}") from exc
        if type(row["short_reason"]) is not str or not row["short_reason"].strip():
            raise Step3QP2ValidationError(f"{role} lacks a short reason at row {index}")
        validated.append(row)
    return validated


def validate_hidden_map(
    hidden: Sequence[Mapping[str, Any]], packet: Sequence[Mapping[str, Any]], role: str
) -> list[dict[str, Any]]:
    if len(hidden) != len(packet):
        raise Step3QP2ValidationError(f"{role} hidden-map row count mismatch")
    expected_ids = [row["opaque_candidate_id"] for row in packet]
    result: list[dict[str, Any]] = []
    for index, (raw, expected_id) in enumerate(zip(hidden, expected_ids), 1):
        row = dict(raw)
        if (
            tuple(row) != HIDDEN_MAP_FIELDS
            or row.get("opaque_candidate_id") != expected_id
            or row.get("item_role") not in {"primary", "sentinel", "duplicate", "primary_disagreement"}
            or type(row.get("canonical_candidate_id")) is not str
            or not row["canonical_candidate_id"]
            or (
                row["item_role"] == "duplicate"
                and (type(row.get("duplicate_parent_id")) is not str or not REVIEW_ID.fullmatch(row["duplicate_parent_id"]))
            )
            or (row["item_role"] != "duplicate" and row.get("duplicate_parent_id") is not None)
        ):
            raise Step3QP2ValidationError(f"{role} hidden-map schema mismatch at row {index}")
        result.append(row)
    if len({row["opaque_candidate_id"] for row in result}) != len(result):
        raise Step3QP2ValidationError(f"{role} hidden map repeats review IDs")
    return result


def validate_provenance(
    value: Mapping[str, Any],
    *,
    reg: Mapping[str, Any],
    role: str,
    packet_bytes: bytes,
    review_bytes: bytes,
    row_count: int,
) -> dict[str, Any]:
    if set(value) != PROVENANCE_FIELDS:
        raise Step3QP2ValidationError(f"{role} provenance schema mismatch")
    if (
        value.get("schema") != "step3q_p2_reviewer_provenance_v1"
        or value.get("route_id") != reg["route_id"]
        or value.get("stage_id") != reg["stage_id"]
        or value.get("role") != role
        or type(value.get("reviewer_id")) is not str
        or not value["reviewer_id"].strip()
        or type(value.get("model")) is not str
        or not value["model"].strip()
        or value.get("context_isolated") is not True
        or value.get("packet_sha256") != sha256_bytes(packet_bytes)
        or value.get("rubric_sha256") != reg["frozen_inputs"]["review_rubric"]["sha256"]
        or value.get("review_sha256") != sha256_bytes(review_bytes)
        or value.get("row_count") != row_count
        or any(
            value.get(name) is not False
            for name in (
                "prohibited_access",
                "saw_hidden_map",
                "saw_api_mode",
                "saw_calibration_roles",
                "saw_other_reviews",
            )
        )
    ):
        raise Step3QP2ValidationError(f"{role} provenance validation failed")
    return dict(value)


def load_and_validate_review(
    review_path: Path,
    provenance_path: Path,
    packet_path: Path,
    packet: Sequence[Mapping[str, Any]],
    reg: Mapping[str, Any],
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes, bytes]:
    review_bytes = review_path.read_bytes()
    packet_bytes = packet_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    rows = validate_review_rows(
        parse_jsonl_bytes(review_bytes, f"{role} review"), packet, role
    )
    try:
        provenance_value = parse_json(
            provenance_bytes.decode("utf-8"), f"{role} provenance"
        )
    except UnicodeDecodeError as exc:
        raise Step3QP2ValidationError(f"invalid UTF-8 in {role} provenance") from exc
    if type(provenance_value) is not dict:
        raise Step3QP2ValidationError(f"{role} provenance must be a JSON object")
    provenance = validate_provenance(
        provenance_value,
        reg=reg,
        role=role,
        packet_bytes=packet_bytes,
        review_bytes=review_bytes,
        row_count=len(rows),
    )
    return rows, provenance, review_bytes, provenance_bytes


def validate_reviewer_separation(*records: Mapping[str, Any]) -> None:
    ids = [record["reviewer_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise Step3QP2ValidationError("reviewer IDs must be distinct across roles")


def _review_by_visible_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {row["opaque_candidate_id"]: dict(row) for row in rows}
    if len(result) != len(rows):
        raise Step3QP2ValidationError("review rows repeat visible IDs")
    return result


def score_reviewer_calibration(
    rows: Sequence[Mapping[str, Any]],
    hidden: Sequence[Mapping[str, Any]],
    p1_sources: Mapping[str, Any],
    p0: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    review = _review_by_visible_id(rows)
    hidden_rows = validate_hidden_map(hidden, [{"opaque_candidate_id": row["opaque_candidate_id"]} for row in hidden], reviewer)
    sentinel_rows: list[dict[str, Any]] = []
    selected_reference: list[dict[str, Any]] = []
    reference_by_id = p1_sources["calibration_reference_by_id"]
    primary_visible: set[str] = set()
    duplicate_links: list[tuple[str, str]] = []
    for mapping in hidden_rows:
        visible_id = mapping["opaque_candidate_id"]
        if visible_id not in review:
            raise Step3QP2ValidationError(f"{reviewer} hidden map references a missing review")
        if mapping["item_role"] == "sentinel":
            canonical_id = mapping["canonical_candidate_id"]
            if canonical_id not in reference_by_id:
                raise Step3QP2ValidationError(f"{reviewer} sentinel is outside the calibration bank")
            sentinel_rows.append({**review[visible_id], "opaque_candidate_id": canonical_id})
            selected_reference.append(reference_by_id[canonical_id])
        elif mapping["item_role"] == "primary":
            primary_visible.add(visible_id)
        elif mapping["item_role"] == "duplicate":
            duplicate_links.append((mapping["duplicate_parent_id"], visible_id))
    if len(sentinel_rows) != 24 or len(duplicate_links) != 12:
        raise Step3QP2ValidationError(f"{reviewer} calibration denominator changed")
    if any(parent not in primary_visible for parent, _ in duplicate_links):
        raise Step3QP2ValidationError(f"{reviewer} duplicate parent is not a primary row")
    try:
        sentinel_score = step3q.score_calibration_review(
            sentinel_rows, selected_reference, p0
        )
    except step3q.Step3QValidationError as exc:
        raise Step3QP2ValidationError(f"{reviewer}: {exc}") from exc
    full_vector_count = sum(
        all(review[parent][field] == review[duplicate][field] for field in DECISION_FIELDS)
        for parent, duplicate in duplicate_links
    )
    duplicate_min = int(p0["calibration"]["gates"]["primary_duplicate_full_vector_consistency_min_count"])
    duplicate_pass = full_vector_count >= duplicate_min
    return {
        "reviewer": reviewer,
        "sentinel": sentinel_score,
        "hidden_primary_duplicate_consistency": {
            "consistent": full_vector_count,
            "denominator": len(duplicate_links),
            "minimum": duplicate_min,
            "pass": duplicate_pass,
        },
        "overall_pass": sentinel_score["overall_pass"] is True and duplicate_pass,
    }


def canonical_primary_reviews(
    rows: Sequence[Mapping[str, Any]],
    hidden: Sequence[Mapping[str, Any]],
    primary_ids: set[str],
    reviewer: str,
) -> dict[str, dict[str, Any]]:
    review = _review_by_visible_id(rows)
    result: dict[str, dict[str, Any]] = {}
    for mapping in hidden:
        if mapping["item_role"] != "primary":
            continue
        canonical_id = mapping["canonical_candidate_id"]
        if canonical_id not in primary_ids or canonical_id in result:
            raise Step3QP2ValidationError(f"{reviewer} primary hidden map is invalid")
        result[canonical_id] = {
            **review[mapping["opaque_candidate_id"]],
            "opaque_candidate_id": canonical_id,
        }
    if set(result) != primary_ids:
        raise Step3QP2ValidationError(f"{reviewer} primary reviews do not cover the imported candidates")
    return result


def _metric_pass(value: Any, threshold: float, *, upper: bool = False) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and (
        float(value) <= float(threshold) if upper else float(value) >= float(threshold)
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise Step3QP2ValidationError("quantile requires non-empty values")
    ordered = sorted(values)
    return ordered[math.floor(probability * (len(ordered) - 1))]


def prompt_cluster_bootstrap(
    primary: Sequence[Mapping[str, Any]],
    review_a: Mapping[str, Mapping[str, Any]],
    review_b: Mapping[str, Mapping[str, Any]],
    p0: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for row in primary:
        by_prompt[row["opaque_prompt_id"]].append(row["opaque_candidate_id"])
    prompt_ids = sorted(by_prompt)
    if len(prompt_ids) < int(p0["authoring"]["minimum_valid_prompts"]):
        raise Step3QP2ValidationError("prompt bootstrap has too few prompt clusters")
    replicates = int(p0["review"]["prompt_bootstrap_replicates"])
    seed = int(stable_rank(namespace, "prompt-cluster-bootstrap-seed")[:16], 16)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        pairs: list[tuple[str, str]] = []
        for _ in prompt_ids:
            sampled = prompt_ids[rng.randrange(len(prompt_ids))]
            for candidate_id in by_prompt[sampled]:
                pairs.append(
                    (
                        o4q.candidate_qualified(review_a[candidate_id]),
                        o4q.candidate_qualified(review_b[candidate_id]),
                    )
                )
        value = o4q.binary_agreement_metrics(pairs)["gwet_ac1"]
        if value is None or not math.isfinite(float(value)):
            raise Step3QP2ValidationError("prompt bootstrap produced a non-finite replicate")
        values.append(float(value))
    return {
        "replicates": replicates,
        "seed_sha256": sha256_bytes(str(seed).encode("utf-8")),
        "q025": _quantile(values, 0.025),
        "median": _quantile(values, 0.5),
        "q975": _quantile(values, 0.975),
    }


def compute_primary_reliability(
    primary: Sequence[Mapping[str, Any]],
    review_a: Mapping[str, Mapping[str, Any]],
    review_b: Mapping[str, Mapping[str, Any]],
    p0: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    primary_ids = [row["opaque_candidate_id"] for row in primary]
    if set(primary_ids) != set(review_a) or set(primary_ids) != set(review_b):
        raise Step3QP2ValidationError("A/B canonical primary review sets differ")
    detail = o4q.detail_agreement_metrics(
        [(review_a[cid]["detail_retention"], review_b[cid]["detail_retention"]) for cid in primary_ids]
    )
    components = {
        field: o4q.binary_agreement_metrics(
            [(review_a[cid][field], review_b[cid][field]) for cid in primary_ids]
        )
        for field in BINARY_FIELDS
    }
    qualified = o4q.binary_agreement_metrics(
        [
            (
                o4q.candidate_qualified(review_a[cid]),
                o4q.candidate_qualified(review_b[cid]),
            )
            for cid in primary_ids
        ]
    )
    full_vector = sum(
        all(review_a[cid][field] == review_b[cid][field] for field in DECISION_FIELDS)
        for cid in primary_ids
    ) / len(primary_ids)
    bootstrap = prompt_cluster_bootstrap(primary, review_a, review_b, p0, namespace)
    cfg = p0["review"]
    conditional_min = int(cfg["class_specific_minimum_pooled_ratings"])
    conditional_agreement = float(cfg["class_specific_agreement_min"])
    detail_classes = step3q.conditional_class_specific_gates(
        detail["qualified_boundary_2_vs_0_or_1"], conditional_min, conditional_agreement
    )
    component_classes = {
        field: step3q.conditional_class_specific_gates(
            components[field], conditional_min, conditional_agreement
        )
        for field in COMPONENT_FIELDS
    }
    gates: dict[str, bool] = {
        "full_vector_exact": _metric_pass(full_vector, cfg["full_vector_exact_agreement_min"]),
        "detail_exact": _metric_pass(detail["exact_agreement"], cfg["detail_exact_agreement_min"]),
        "detail_weighted": _metric_pass(
            detail["linear_weighted_agreement"], cfg["detail_weighted_agreement_min"]
        ),
        "detail_ac2": _metric_pass(detail["gwet_ac2"], cfg["detail_gwet_ac2_min"]),
        "candidate_qualified_raw": _metric_pass(
            qualified["raw_agreement"], cfg["candidate_qualified_raw_agreement_min"]
        ),
        "candidate_qualified_ac1": _metric_pass(
            qualified["gwet_ac1"], cfg["candidate_qualified_gwet_ac1_min"]
        ),
        "candidate_qualified_positive": _metric_pass(
            qualified["positive_agreement"], cfg["candidate_qualified_positive_agreement_min"]
        ),
        "candidate_qualified_negative": _metric_pass(
            qualified["negative_agreement"], cfg["candidate_qualified_negative_agreement_min"]
        ),
        "candidate_qualified_prevalence_gap": _metric_pass(
            qualified["prevalence_gap"], cfg["candidate_qualified_prevalence_gap_max"], upper=True
        ),
        "candidate_qualified_yes_count": qualified["yes_ratings"]
        >= int(cfg["candidate_qualified_minimum_pooled_ratings_per_class"]),
        "candidate_qualified_no_count": qualified["no_ratings"]
        >= int(cfg["candidate_qualified_minimum_pooled_ratings_per_class"]),
        "candidate_qualified_bootstrap_q025": _metric_pass(
            bootstrap["q025"], cfg["candidate_qualified_prompt_bootstrap_q025_min"]
        ),
    }
    for class_name, value in detail_classes.items():
        gates[f"detail_{class_name}_if_supported"] = value["pass"] is not False
    for field in COMPONENT_FIELDS:
        metrics = components[field]
        gates[f"{field}_raw"] = _metric_pass(
            metrics["raw_agreement"], cfg["component_raw_agreement_min"]
        )
        gates[f"{field}_ac1"] = _metric_pass(
            metrics["gwet_ac1"], cfg["component_gwet_ac1_min"]
        )
        gates[f"{field}_prevalence_gap"] = _metric_pass(
            metrics["prevalence_gap"], cfg["component_prevalence_gap_max"], upper=True
        )
        for class_name, value in component_classes[field].items():
            gates[f"{field}_{class_name}_if_supported"] = value["pass"] is not False
    for field, raw_key, ac1_key in (
        ("naturalness", "naturalness_raw_agreement_min", "naturalness_gwet_ac1_min"),
        ("clean_safe", "clean_safe_raw_agreement_min", "clean_safe_gwet_ac1_min"),
    ):
        gates[f"{field}_raw"] = _metric_pass(components[field]["raw_agreement"], cfg[raw_key])
        gates[f"{field}_ac1"] = _metric_pass(components[field]["gwet_ac1"], cfg[ac1_key])
    return {
        "primary_candidates": len(primary_ids),
        "primary_prompts": len({row["opaque_prompt_id"] for row in primary}),
        "detail": detail,
        "detail_class_specific": detail_classes,
        "components": components,
        "component_class_specific": component_classes,
        "candidate_qualified": qualified,
        "full_vector_exact_agreement": full_vector,
        "prompt_cluster_bootstrap": bootstrap,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def build_c_packet(
    primary: Sequence[Mapping[str, Any]],
    review_a: Mapping[str, Mapping[str, Any]],
    review_b: Mapping[str, Mapping[str, Any]],
    namespace: str,
) -> dict[str, list[dict[str, Any]]]:
    disagreements = [
        row
        for row in primary
        if o4q.candidate_qualified(review_a[row["opaque_candidate_id"]])
        != o4q.candidate_qualified(review_b[row["opaque_candidate_id"]])
    ]
    packet: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for row in disagreements:
        canonical_id = row["opaque_candidate_id"]
        visible_id = _review_id(namespace, "reviewer_c", "primary-disagreement", canonical_id)
        packet.append(
            {"opaque_candidate_id": visible_id, "prompt": row["prompt"], "response": row["response"]}
        )
        hidden.append(
            {
                "opaque_candidate_id": visible_id,
                "item_role": "primary_disagreement",
                "canonical_candidate_id": canonical_id,
                "duplicate_parent_id": None,
            }
        )
    order = sorted(
        range(len(packet)),
        key=lambda index: stable_rank(
            namespace, "reviewer-c-order", packet[index]["opaque_candidate_id"]
        ),
    )
    packet = [packet[index] for index in order]
    hidden = [hidden[index] for index in order]
    packet_bytes = jsonl_text(packet).encode("utf-8")
    if any(
        forbidden in packet_bytes
        for forbidden in (b"step3q_candidate_", b"specific_redirect", b"risk_mitigation")
    ):
        raise Step3QP2ValidationError("Reviewer C packet leaks hidden candidate roles")
    return {"packet": packet, "hidden_map": hidden}


def build_ab_outcome_files(
    bundle: Mapping[str, Any],
    reg: Mapping[str, Any],
    p0: Mapping[str, Any],
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    provenance_a: Mapping[str, Any],
    provenance_b: Mapping[str, Any],
    review_bytes_a: bytes,
    review_bytes_b: bytes,
    provenance_bytes_a: bytes,
    provenance_bytes_b: bytes,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    p1_sources = _load_p1_review_sources(reg)
    calibration_a = score_reviewer_calibration(
        rows_a, bundle["hidden_reviewer_a"], p1_sources, p0, "reviewer_a"
    )
    calibration_b = score_reviewer_calibration(
        rows_b, bundle["hidden_reviewer_b"], p1_sources, p0, "reviewer_b"
    )
    calibration_pass = calibration_a["overall_pass"] and calibration_b["overall_pass"]
    files: dict[str, bytes] = {
        "inputs/reviewer_a_review.jsonl": review_bytes_a,
        "inputs/reviewer_a_provenance.json": provenance_bytes_a,
        "inputs/reviewer_b_review.jsonl": review_bytes_b,
        "inputs/reviewer_b_provenance.json": provenance_bytes_b,
        "calibration/reviewer_a_calibration.json": json_text(calibration_a).encode("utf-8"),
        "calibration/reviewer_b_calibration.json": json_text(calibration_b).encode("utf-8"),
    }
    route = "STOP_REVIEW_CALIBRATION"
    reliability: dict[str, Any] | None = None
    c_rows = 0
    if calibration_pass:
        primary_ids = {row["opaque_candidate_id"] for row in bundle["primary"]}
        canonical_a = canonical_primary_reviews(
            rows_a, bundle["hidden_reviewer_a"], primary_ids, "reviewer_a"
        )
        canonical_b = canonical_primary_reviews(
            rows_b, bundle["hidden_reviewer_b"], primary_ids, "reviewer_b"
        )
        reliability = compute_primary_reliability(
            bundle["primary"],
            canonical_a,
            canonical_b,
            p0,
            bundle["manifest"]["namespace_sha256"],
        )
        files["primary/ab_reliability.json"] = json_text(reliability).encode("utf-8")
        route = "PASS_AB_REVIEW" if reliability["overall_pass"] else "STOP_REVIEW_UNRELIABLE"
        if reliability["overall_pass"]:
            c_package = build_c_packet(
                bundle["primary"],
                canonical_a,
                canonical_b,
                bundle["manifest"]["namespace_sha256"],
            )
            c_packet_bytes = jsonl_text(c_package["packet"]).encode("utf-8")
            rubric_bytes = resolve_repo_path(reg["frozen_inputs"]["review_rubric"]["path"]).read_bytes()
            files["reviewer_c/reviewer_c_packet.jsonl"] = c_packet_bytes
            files["private/reviewer_c_hidden_map.jsonl"] = jsonl_text(
                c_package["hidden_map"]
            ).encode("utf-8")
            files["reviewer_c/reviewer_c_review_template.jsonl"] = jsonl_text(
                _review_template(c_package["packet"])
            ).encode("utf-8")
            files["reviewer_c/reviewer_c_provenance_template.json"] = json_text(
                _provenance_template(
                    reg, "reviewer_c", c_packet_bytes, rubric_bytes, len(c_package["packet"])
                )
            ).encode("utf-8")
            c_rows = len(c_package["packet"])

    report = {
        "schema": "step3q_p2_ab_route_v1",
        "route_id": reg["route_id"],
        "route": route,
        "calibration_pass": calibration_pass,
        "primary_metrics_created": calibration_pass,
        "primary_reliability_pass": reliability["overall_pass"] if reliability is not None else None,
        "reviewer_c_rows": c_rows,
        "reviewer_c_opened": route == "PASS_AB_REVIEW" and c_rows > 0,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["ab_route.json"] = json_text(report).encode("utf-8")
    manifest = {
        "schema": "step3q_p2_ab_outcome_manifest_v1",
        "route_id": reg["route_id"],
        "import_manifest_sha256": bundle["manifest_sha256"],
        "reviewer_ids": [provenance_a["reviewer_id"], provenance_b["reviewer_id"]],
        "artifacts": _manifest_artifacts(files),
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["outcome_manifest.json"] = json_text(manifest).encode("utf-8")
    return files, report


def load_ab_outcome(
    output_root: Path, reg: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    root = output_root / "ab_outcome"
    manifest, _ = _verify_manifest_bundle(
        root, "outcome_manifest.json", "step3q_p2_ab_outcome_manifest_v1"
    )
    route = read_json(root / "ab_route.json")
    if (
        manifest.get("route_id") != reg["route_id"]
        or route.get("route_id") != reg["route_id"]
        or manifest.get("import_manifest_sha256") != bundle["manifest_sha256"]
    ):
        raise Step3QP2ValidationError("A/B outcome route mismatch")
    return {"root": root, "manifest": manifest, "route": route}


def verify_ab_outcome_reconstruction(
    output_root: Path,
    bundle: Mapping[str, Any],
    ab: Mapping[str, Any],
    reg: Mapping[str, Any],
    p0: Mapping[str, Any],
) -> dict[str, Any]:
    stored: dict[str, Any] = {}
    for reviewer in REVIEWERS_AB:
        stored[reviewer] = load_and_validate_review(
            ab["root"] / "inputs" / f"{reviewer}_review.jsonl",
            ab["root"] / "inputs" / f"{reviewer}_provenance.json",
            output_root / "packets" / f"{reviewer}_packet.jsonl",
            bundle[f"packet_{reviewer}"],
            reg,
            reviewer,
        )
    validate_reviewer_separation(stored["reviewer_a"][1], stored["reviewer_b"][1])
    expected, report = build_ab_outcome_files(
        bundle,
        reg,
        p0,
        stored["reviewer_a"][0],
        stored["reviewer_b"][0],
        stored["reviewer_a"][1],
        stored["reviewer_b"][1],
        stored["reviewer_a"][2],
        stored["reviewer_b"][2],
        stored["reviewer_a"][3],
        stored["reviewer_b"][3],
    )
    actual_paths = {
        path.relative_to(ab["root"]).as_posix()
        for path in ab["root"].rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        raise Step3QP2ValidationError("committed A/B outcome differs from deterministic reconstruction")
    for relative, content in expected.items():
        if (ab["root"] / relative).read_bytes() != content:
            raise Step3QP2ValidationError(
                f"committed A/B artifact differs from deterministic reconstruction: {relative}"
            )
    return {"stored": stored, "report": report}


def canonical_c_reviews(
    rows: Sequence[Mapping[str, Any]], hidden: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    review = _review_by_visible_id(rows)
    result: dict[str, dict[str, Any]] = {}
    for mapping in hidden:
        if mapping["item_role"] != "primary_disagreement":
            raise Step3QP2ValidationError("Reviewer C hidden map contains a non-disagreement row")
        canonical_id = mapping["canonical_candidate_id"]
        if canonical_id in result:
            raise Step3QP2ValidationError("Reviewer C hidden map repeats a canonical candidate")
        result[canonical_id] = {
            **review[mapping["opaque_candidate_id"]],
            "opaque_candidate_id": canonical_id,
        }
    if len(result) != len(rows):
        raise Step3QP2ValidationError("Reviewer C mapping does not cover its packet")
    return result


def adjudicate(
    primary: Sequence[Mapping[str, Any]],
    review_a: Mapping[str, Mapping[str, Any]],
    review_b: Mapping[str, Mapping[str, Any]],
    review_c: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    disagreement_ids = {
        row["opaque_candidate_id"]
        for row in primary
        if o4q.candidate_qualified(review_a[row["opaque_candidate_id"]])
        != o4q.candidate_qualified(review_b[row["opaque_candidate_id"]])
    }
    if set(review_c) != disagreement_ids:
        raise Step3QP2ValidationError("Reviewer C covers anything other than derived A/B disagreements")
    final: list[dict[str, Any]] = []
    for row in primary:
        candidate_id = row["opaque_candidate_id"]
        label_a = o4q.candidate_qualified(review_a[candidate_id])
        label_b = o4q.candidate_qualified(review_b[candidate_id])
        if label_a == label_b:
            vector: dict[str, Any] = {
                "detail_retention": min(
                    review_a[candidate_id]["detail_retention"],
                    review_b[candidate_id]["detail_retention"],
                )
            }
            vector.update(
                {
                    field: (
                        "yes"
                        if review_a[candidate_id][field]
                        == review_b[candidate_id][field]
                        == "yes"
                        else "no"
                    )
                    for field in BINARY_FIELDS
                }
            )
            label = o4q.candidate_qualified(vector)
            source = "ab_derived_agreement"
            label_c = None
            adjudicated_fields: list[str] = []
        else:
            vector = {field: review_c[candidate_id][field] for field in DECISION_FIELDS}
            label_c = o4q.candidate_qualified(vector)
            label = label_c
            source = "reviewer_c_disagreement_only"
            adjudicated_fields = list(DECISION_FIELDS)
        final.append(
            {
                "opaque_candidate_id": candidate_id,
                **vector,
                "reviewer_a_candidate_qualified": label_a,
                "reviewer_b_candidate_qualified": label_b,
                "reviewer_c_candidate_qualified": label_c,
                "candidate_qualified": label,
                "decision_source": source,
                "adjudicated_fields": adjudicated_fields,
            }
        )
    return final


def select_final_targets(
    qualified: Sequence[Mapping[str, Any]], p0: Mapping[str, Any], namespace: str
) -> dict[str, Any]:
    selection_reg = {
        "selection": {
            "target_count": int(p0["final_pool"]["target_count"]),
            "severity_minima": dict(p0["final_pool"]["severity_minima"]),
            "categories": sorted(p0["selection"]["category_severity_slots"]),
            "category_minimum_each": int(p0["final_pool"]["category_minimum_each"]),
        }
    }
    canonical_by_flow_id: dict[str, dict[str, Any]] = {}
    projected: list[dict[str, Any]] = []
    for row in qualified:
        flow_id = "step3q_flow_" + stable_rank(
            namespace,
            "mode-blind-flow-id",
            row["opaque_prompt_id"],
            row["normalized_response_sha256"],
        )[:24]
        if flow_id in canonical_by_flow_id:
            raise Step3QP2ValidationError("mode-blind final-flow ID collision")
        canonical_by_flow_id[flow_id] = dict(row)
        projected.append(
            {
                "opaque_candidate_id": flow_id,
                "opaque_prompt_id": row["opaque_prompt_id"],
                "normalized_response_sha256": row["normalized_response_sha256"],
                "severity_level": row["severity_level"],
                "categories": list(row["categories"]),
            }
        )
    try:
        selection = step3p.select_targets(projected, selection_reg, namespace=namespace)
    except step3p.Step3PValidationError as exc:
        raise Step3QP2ValidationError(str(exc)) from exc
    selected_rows = [
        canonical_by_flow_id[row["opaque_candidate_id"]] for row in selection["selected"]
    ]
    assigned_categories = {
        canonical_by_flow_id[flow_id]["opaque_candidate_id"]: category
        for flow_id, category in selection["assigned_categories"].items()
    }
    return {
        **selection,
        "selected": selected_rows,
        "assigned_categories": assigned_categories,
        "mode_blind_projection_fields": [
            "opaque_prompt_id",
            "normalized_response_sha256",
            "severity_level",
            "categories",
        ],
    }


def build_final_files(
    bundle: Mapping[str, Any],
    reg: Mapping[str, Any],
    p0: Mapping[str, Any],
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    rows_c: Sequence[Mapping[str, Any]],
    hidden_c: Sequence[Mapping[str, Any]],
    c_review_bytes: bytes | None,
    c_provenance_bytes: bytes | None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    primary_ids = {row["opaque_candidate_id"] for row in bundle["primary"]}
    review_a = canonical_primary_reviews(
        rows_a, bundle["hidden_reviewer_a"], primary_ids, "reviewer_a"
    )
    review_b = canonical_primary_reviews(
        rows_b, bundle["hidden_reviewer_b"], primary_ids, "reviewer_b"
    )
    review_c = canonical_c_reviews(rows_c, hidden_c) if rows_c else {}
    final_labels = adjudicate(bundle["primary"], review_a, review_b, review_c)
    final_by_id = {row["opaque_candidate_id"]: row for row in final_labels}
    qualified = [
        row
        for row in bundle["primary"]
        if final_by_id[row["opaque_candidate_id"]]["candidate_qualified"] == "yes"
    ]
    selection = select_final_targets(
        qualified, p0, bundle["manifest"]["namespace_sha256"]
    )
    route = "PASS_CALIBRATED_API_POOL" if selection["pass"] else "STOP_TARGET_YIELD"
    report = {
        "schema": "step3q_p2_final_report_v1",
        "route_id": reg["route_id"],
        "route": route,
        "qualified_candidates": len(qualified),
        "qualified_prompts": len({row["opaque_prompt_id"] for row in qualified}),
        "selection": selection["summary"],
        "no_api_mode_quota_or_preference": True,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files: dict[str, bytes] = {
        "final_labels.jsonl": jsonl_text(final_labels).encode("utf-8"),
        "final_report.json": json_text(report).encode("utf-8"),
    }
    if c_review_bytes is not None and c_provenance_bytes is not None:
        files["reviewer_c_review.jsonl"] = c_review_bytes
        files["reviewer_c_provenance.json"] = c_provenance_bytes
    if selection["pass"]:
        selected_rows: list[dict[str, Any]] = []
        for row in sorted(
            selection["selected"],
            key=lambda value: stable_rank(
                bundle["manifest"]["namespace_sha256"],
                "final-target-order",
                value["opaque_candidate_id"],
            ),
        ):
            selected_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "opaque_prompt_id": row["opaque_prompt_id"],
                    "opaque_candidate_id": row["opaque_candidate_id"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "author_mode": row["author_mode"],
                    "assigned_coverage_category": selection["assigned_categories"][
                        row["opaque_candidate_id"]
                    ],
                    "all_categories": row["all_source_categories"],
                    "severity_level": row["severity_level"],
                    "prompt_sha256": row["prompt_sha256"],
                    "response_sha256": row["response_sha256"],
                    "normalized_response_sha256": row["normalized_response_sha256"],
                    "effective_model": row["effective_model"],
                }
            )
        files["machine_rubric_candidate_pool.jsonl"] = jsonl_text(selected_rows).encode("utf-8")
    manifest = {
        "schema": "step3q_p2_final_manifest_v1",
        "route_id": reg["route_id"],
        "route": route,
        "artifacts": _manifest_artifacts(files),
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    files["final_manifest.json"] = json_text(manifest).encode("utf-8")
    return files, report


def command_preflight(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).resolve()
    reg, p0 = load_registry(registry_path)
    verification = step3q.verify_prepared_bundle(p0, prepared_root(reg))
    result = {
        "schema": "step3q_p2_preflight_v1",
        "route_id": reg["route_id"],
        "entry_commit": reg["entry_commit"],
        "protocol_tag": reg["protocol_tag"],
        "protocol_commit_sha1": reg["protocol_commit_sha1"],
        "protocol_lock_status": reg["protocol_lock_status"],
        "p1_verification": verification,
        "fixed_requests": reg["dispatch_import"]["fixed_requests"],
        "reviewers": reg["review_packets"]["roles"],
        "calibration_before_primary_metrics": True,
        "output_exists": resolve_repo_path(reg["outputs"]["root"]).exists(),
        "external_api_called_by_preflight": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    print(json_text(result), end="")


def command_import_dispatch(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).resolve()
    reg, p0 = load_registry(registry_path)
    protocol_commit = require_protocol_head(registry_path, reg)
    dispatch = verify_external_dispatch(reg, p0, Path(args.spool_root))
    files, summary = build_import_files(reg, p0, dispatch, protocol_commit, registry_path)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else resolve_repo_path(reg["outputs"]["root"])
    )
    write_bundle(output_root, files)
    print(json_text({"output_root": str(output_root), **summary}), end="")


def command_analyze_ab(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).resolve()
    reg, p0 = load_registry(registry_path)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else resolve_repo_path(reg["outputs"]["root"])
    )
    bundle = load_import_bundle(output_root, reg)
    require_frozen_stage_checkout(
        registry_path, reg, bundle["manifest"]["protocol_commit"]
    )
    require_bundle_committed(output_root, bundle["manifest"], "packet_manifest.json")
    dispatch = verify_external_dispatch(reg, p0, Path(args.spool_root))
    verify_import_reconstruction(
        output_root, bundle, reg, p0, dispatch, registry_path
    )
    reviews: dict[str, Any] = {}
    for reviewer, review_arg, provenance_arg in (
        ("reviewer_a", args.review_a, args.provenance_a),
        ("reviewer_b", args.review_b, args.provenance_b),
    ):
        packet_path = output_root / "packets" / f"{reviewer}_packet.jsonl"
        reviews[reviewer] = load_and_validate_review(
            Path(review_arg).resolve(),
            Path(provenance_arg).resolve(),
            packet_path,
            bundle[f"packet_{reviewer}"],
            reg,
            reviewer,
        )
    validate_reviewer_separation(reviews["reviewer_a"][1], reviews["reviewer_b"][1])
    files, report = build_ab_outcome_files(
        bundle,
        reg,
        p0,
        reviews["reviewer_a"][0],
        reviews["reviewer_b"][0],
        reviews["reviewer_a"][1],
        reviews["reviewer_b"][1],
        reviews["reviewer_a"][2],
        reviews["reviewer_b"][2],
        reviews["reviewer_a"][3],
        reviews["reviewer_b"][3],
    )
    destination = output_root / "ab_outcome"
    write_bundle(destination, files)
    print(json_text({"output_root": str(destination), **report}), end="")


def command_finalize(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).resolve()
    reg, p0 = load_registry(registry_path)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else resolve_repo_path(reg["outputs"]["root"])
    )
    bundle = load_import_bundle(output_root, reg)
    require_frozen_stage_checkout(
        registry_path, reg, bundle["manifest"]["protocol_commit"]
    )
    require_bundle_committed(output_root, bundle["manifest"], "packet_manifest.json")
    dispatch = verify_external_dispatch(reg, p0, Path(args.spool_root))
    verify_import_reconstruction(
        output_root, bundle, reg, p0, dispatch, registry_path
    )
    ab = load_ab_outcome(output_root, reg, bundle)
    require_bundle_committed(ab["root"], ab["manifest"], "outcome_manifest.json")
    reconstructed_ab = verify_ab_outcome_reconstruction(
        output_root, bundle, ab, reg, p0
    )
    if ab["route"].get("route") != "PASS_AB_REVIEW":
        raise Step3QP2ValidationError("finalization is forbidden unless A/B review passed")

    stored = reconstructed_ab["stored"]
    c_packet_path = ab["root"] / "reviewer_c" / "reviewer_c_packet.jsonl"
    c_hidden_path = ab["root"] / "private" / "reviewer_c_hidden_map.jsonl"
    if not c_packet_path.is_file() or not c_hidden_path.is_file():
        raise Step3QP2ValidationError("A/B pass outcome lacks the frozen Reviewer C gate")
    c_packet = read_jsonl(c_packet_path)
    c_hidden = read_jsonl(c_hidden_path)
    validate_hidden_map(c_hidden, c_packet, "reviewer_c")
    if c_packet:
        if not args.review_c or not args.provenance_c:
            raise Step3QP2ValidationError("Reviewer C review and provenance are required")
        c_review = load_and_validate_review(
            Path(args.review_c).resolve(),
            Path(args.provenance_c).resolve(),
            c_packet_path,
            c_packet,
            reg,
            "reviewer_c",
        )
        validate_reviewer_separation(stored["reviewer_a"][1], stored["reviewer_b"][1], c_review[1])
        rows_c = c_review[0]
        c_review_bytes = c_review[2]
        c_provenance_bytes = c_review[3]
    else:
        if args.review_c or args.provenance_c:
            raise Step3QP2ValidationError("Reviewer C input is forbidden for an empty disagreement packet")
        rows_c = []
        c_review_bytes = None
        c_provenance_bytes = None
    files, report = build_final_files(
        bundle,
        reg,
        p0,
        stored["reviewer_a"][0],
        stored["reviewer_b"][0],
        rows_c,
        c_hidden,
        c_review_bytes,
        c_provenance_bytes,
    )
    destination = output_root / "finalization"
    write_bundle(destination, files)
    print(json_text({"output_root": str(destination), **report}), end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="Verify the prospective P2 boundary without opening receipts")
    preflight.set_defaults(func=command_preflight)

    import_dispatch = sub.add_parser(
        "import-dispatch", help="Recompute an external PASS_API_AUTHORING spool and freeze blind packets"
    )
    import_dispatch.add_argument("--spool-root", required=True)
    import_dispatch.add_argument("--output-root")
    import_dispatch.set_defaults(func=command_import_dispatch)

    analyze = sub.add_parser(
        "analyze-ab", help="Apply calibration first, then primary reliability, then optionally open C"
    )
    analyze.add_argument("--output-root")
    analyze.add_argument("--spool-root", required=True)
    analyze.add_argument("--review-a", required=True)
    analyze.add_argument("--provenance-a", required=True)
    analyze.add_argument("--review-b", required=True)
    analyze.add_argument("--provenance-b", required=True)
    analyze.set_defaults(func=command_analyze_ab)

    finalize = sub.add_parser(
        "finalize", help="Adjudicate derived A/B disagreements and apply the fixed 48-target flow"
    )
    finalize.add_argument("--output-root")
    finalize.add_argument("--spool-root", required=True)
    finalize.add_argument("--review-c")
    finalize.add_argument("--provenance-c")
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
