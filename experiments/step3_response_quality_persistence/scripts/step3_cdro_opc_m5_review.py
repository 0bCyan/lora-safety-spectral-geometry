"""Run the CDR-OPC M5 independent blind review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import finalize_step3_cdro_opc_m5_authoring as m5_authoring
import step3_cdro_opc_m3_review_v3 as m3
import step3o_o4_qualification as qualification
import step3q_calibrated_api_pool as step3q
import step3q_p2_review as p2


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "experiments/step3_response_quality_persistence"
REGISTRY = ROOT / "configs/4070ti_step3_cdro_opc_m5_review_registry.json"
REVIEWERS_AB = ("reviewer_a", "reviewer_b")
REVIEWER_C = "reviewer_c"
DECISION_FIELDS = m3.DECISION_FIELDS
ANCHOR_FAMILIES = m3.ANCHOR_FAMILIES
REQUEST_ID_RE = re.compile(r"^m5_req_[0-9]{3}$")


class M5ReviewError(RuntimeError):
    pass


sha256_bytes = m3.sha256_bytes
sha256_file = m3.sha256_file
stable = m3.stable
json_bytes = m3.json_bytes
jsonl_bytes = m3.jsonl_bytes
read_json = m3.read_json
read_jsonl = m3.read_jsonl


def canonical_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(data)


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M5ReviewError(f"path escapes repository: {value}") from exc
    return path


def check_input(spec: Mapping[str, Any]) -> Path:
    path = repo_path(str(spec["path"]))
    if not path.is_file() or path.is_symlink() or sha256_file(path) != spec["sha256"]:
        raise M5ReviewError(f"frozen input changed: {path}")
    return path


def load_registry() -> dict[str, Any]:
    reg = read_json(REGISTRY)
    if reg.get("schema") != "step3_cdro_opc_m5_review_registry_v1":
        raise M5ReviewError("unexpected M5 review registry schema")
    if reg.get("status") != "prospective_before_packet_construction":
        raise M5ReviewError("M5 review registry is not prospective")
    if (
        type(reg.get("protocol_tag")) is not str
        or not reg["protocol_tag"]
        or type(reg.get("lock_tag")) is not str
        or not reg["lock_tag"]
        or type(reg.get("review_spool_root")) is not str
        or not reg["review_spool_root"]
    ):
        raise M5ReviewError("M5 lock or spool identity is missing")
    if (
        reg.get("optimizer_steps") != 0
        or reg.get("training_authorized") is not False
        or reg.get("core100_opened") is not False
        or reg.get("heldout_opened") is not False
    ):
        raise M5ReviewError("M5 authorization boundary changed")
    expected_boundary = {
        "main_session_labels_allowed": False,
        "reviewer_context_isolated": True,
        "reviewers_may_read_repository": False,
        "reviewers_may_read_hidden_maps": False,
        "reviewers_may_read_other_reviews": False,
        "reviewer_c_may_read_ab_short_reasons": True,
        "reviewers_may_read_training_or_gradient_results": False,
        "reviewer_replacement_allowed": False,
        "post_submission_revision_allowed": False,
        "heuristic_label_substitution_allowed": False,
    }
    if reg.get("review_boundary") != expected_boundary:
        raise M5ReviewError("M5 blind-review boundary changed")
    for spec in reg["authoring_seal"].values():
        if isinstance(spec, dict) and "path" in spec:
            check_input(spec)
    for spec in reg["frozen_inputs"].values():
        check_input(spec)
    packet = reg["packet"]
    if (
        packet.get("reviewer_roles") != list(REVIEWERS_AB)
        or packet.get("primary_candidates") != 120
        or sum(
            packet.get(name, -1)
            for name in (
                "positive_anchors",
                "generic_anchors",
                "mismatch_anchors",
                "clean_safety_negative_anchors",
            )
        )
        != 48
        or packet.get("hidden_primary_duplicates") != 12
        or packet.get("rows_per_reviewer") != 180
        or packet.get("immutable_submission") is not True
    ):
        raise M5ReviewError("M5 packet denominator changed")
    selection = reg["selection"]
    if (
        selection.get("eligible_prompts_required") != 40
        or selection.get("calibration_units") != 20
        or selection.get("sealed_audit_units") != 20
    ):
        raise M5ReviewError("M5 selection denominator changed")
    return reg


def git(*args: str) -> str:
    return m3.git(*args)


def git_json(commit: str, relative: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(raw)
    if type(value) is not dict:
        raise M5ReviewError(f"git registry is not an object: {commit}")
    return value


def require_locked_clean(reg: Mapping[str, Any]) -> str:
    commit = str(reg.get("protocol_commit_sha1"))
    if reg.get("protocol_lock_status") != "locked_to_protocol_commit":
        raise M5ReviewError("M5 review protocol is not locked")
    lock_tag = str(reg.get("lock_tag"))
    if not lock_tag:
        raise M5ReviewError("M5 review lock tag is missing")
    if git("rev-parse", f"{reg['protocol_tag']}^{{commit}}") != commit:
        raise M5ReviewError("M5 protocol tag differs from registry lock")
    head = git("rev-parse", "HEAD")
    if git("rev-parse", f"{lock_tag}^{{commit}}") != head:
        raise M5ReviewError("M5 checkout is not at the locked review commit")
    if git("rev-list", "--count", f"{commit}..HEAD") != "1":
        raise M5ReviewError("M5 lock must be exactly one commit after protocol freeze")
    if git("rev-parse", "HEAD^") != commit:
        raise M5ReviewError("M5 lock commit parent is not the protocol commit")
    if git("status", "--porcelain"):
        raise M5ReviewError("M5 review requires a clean checkout")
    changed = git("diff", "--name-only", commit, "HEAD").splitlines()
    expected = [REGISTRY.relative_to(REPO).as_posix()]
    if changed != expected:
        raise M5ReviewError(f"M5 protocol-to-lock diff changed: {changed}")
    base = git_json(commit, REGISTRY.relative_to(REPO).as_posix())
    current = read_json(REGISTRY)
    allowed = {"protocol_commit_sha1", "protocol_lock_status"}
    base_norm = {key: value for key, value in base.items() if key not in allowed}
    current_norm = {key: value for key, value in current.items() if key not in allowed}
    if base_norm != current_norm:
        raise M5ReviewError("M5 lock commit changed fields beyond the lock transition")
    if base.get("protocol_commit_sha1") != "PENDING_PROTOCOL_COMMIT" or base.get("protocol_lock_status") != "pending":
        raise M5ReviewError("M5 protocol base commit is not the pending registry")
    if current.get("protocol_commit_sha1") != commit:
        raise M5ReviewError("M5 registry does not name the protocol commit")
    return commit


def external(path: Path, *, new: bool = False) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        raise M5ReviewError("M5 review spool must remain outside the repository")
    if new and resolved.exists():
        raise M5ReviewError("M5 review spool already exists")
    return resolved


def require_review_root(reg: Mapping[str, Any], review_root: Path, *, new: bool = False) -> Path:
    target = external(review_root, new=new)
    expected = external(Path(str(reg["review_spool_root"])))
    if target != expected:
        raise M5ReviewError(f"M5 review spool must be the frozen root: {expected}")
    return target


def write_verified_files(root: Path, files: Mapping[str, bytes]) -> None:
    resolved_root = root.resolve()
    targets: list[tuple[Path, bytes]] = []
    for relative, data in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise M5ReviewError(f"invalid phase artifact path: {relative}")
        target = (root / relative_path).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise M5ReviewError(f"phase artifact escapes root: {relative}") from exc
        if target.exists():
            if not target.is_file() or target.is_symlink() or target.read_bytes() != data:
                raise M5ReviewError(f"immutable phase artifact changed: {relative}")
        else:
            targets.append((target, data))
    for target, data in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if target.exists():
                if target.read_bytes() != data:
                    raise M5ReviewError(f"phase artifact appeared with different bytes: {target}")
                temporary.unlink()
            else:
                temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()


def load_primary(
    reg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    seal = reg["authoring_seal"]
    protocol = read_json(check_input(seal["protocol"]))
    summary = read_json(check_input(seal["summary"]))
    manifest = read_json(check_input(seal["manifest"]))
    records = read_jsonl(check_input(seal["record_manifest"]))
    candidates = read_jsonl(check_input(seal["primary_candidates"]))
    if (
        protocol.get("schema") != "step3_cdro_opc_m5_independent_authoring_protocol_v1"
        or summary.get("route") != seal["required_route"]
        or manifest.get("route") != seal["required_route"]
        or summary.get("valid_candidates") != seal["candidates"]
        or summary.get("valid_prompts") != seal["prompts"]
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise M5ReviewError("M5 authoring seal metadata changed")
    expected_artifacts = {
        "authoring_summary.json": seal["summary"]["sha256"],
        "record_manifest.jsonl": seal["record_manifest"]["sha256"],
        "private/primary_candidates.jsonl": seal["primary_candidates"]["sha256"],
    }
    for path, expected_hash in expected_artifacts.items():
        if manifest.get("artifacts", {}).get(path) != expected_hash:
            raise M5ReviewError(f"M5 authoring manifest changed: {path}")
    if len(records) != 120 or len(candidates) != 120:
        raise M5ReviewError("M5 authoring candidate denominator changed")

    authoring_inputs = protocol.get("frozen_inputs")
    if not isinstance(authoring_inputs, dict):
        raise M5ReviewError("M5 authoring protocol lacks frozen inputs")
    for spec in authoring_inputs.values():
        if isinstance(spec, dict) and "path" in spec:
            check_input(spec)
    risk_rows = read_jsonl(check_input(reg["frozen_inputs"]["risk_source"]))
    risk_by_id = {str(row["sample_id"]): row for row in risk_rows}
    if len(risk_by_id) != len(risk_rows):
        raise M5ReviewError("risk source repeats sample IDs")

    old_rows = read_jsonl(check_input(authoring_inputs["m3_primary_candidates"]))
    excluded_samples = {str(row["sample_id"]) for row in old_rows}
    eligible_rows = [
        row for row in risk_rows if str(row["sample_id"]) not in excluded_samples
    ]
    eligible_rows.sort(
        key=lambda row: hashlib.sha256(
            f"cdro-opc-m5-authoring|{row['sample_id']}".encode("utf-8")
        ).hexdigest()
    )
    selected_rows = eligible_rows[: int(protocol["selection"]["prompts"])]
    if len(selected_rows) != 60:
        raise M5ReviewError("M5 prompt selection denominator changed")
    prompt_registry = {
        "schema": "step3_cdro_opc_m5_prompt_registry_v1",
        "excluded_m3_sample_count": len(excluded_samples),
        "selected_prompt_count": len(selected_rows),
        "prompts": [
            {
                "opaque_prompt_id": (
                    f"m5_prompt_{rank:03d}_"
                    f"{hashlib.sha256(str(row['sample_id']).encode('utf-8')).hexdigest()[:12]}"
                ),
                "sample_id": str(row["sample_id"]),
                "prompt": str(row["prompt"]),
                "rank": rank,
            }
            for rank, row in enumerate(selected_rows, 1)
        ],
    }
    if (
        canonical_sha256(prompt_registry) != protocol.get("prompt_registry_sha256")
    ):
        raise M5ReviewError("M5 prompt registry hash changed")
    expected_prompts = {
        row["opaque_prompt_id"]: row for row in prompt_registry["prompts"]
    }
    record_by_id = {str(row["opaque_candidate_id"]): row for row in records}
    if len(record_by_id) != len(records):
        raise M5ReviewError("M5 record manifest repeats candidate IDs")

    primary: list[dict[str, Any]] = []
    prompt_metadata: dict[str, dict[str, Any]] = {}
    prompt_modes: dict[str, set[str]] = defaultdict(set)
    candidate_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    for row in candidates:
        candidate_id = str(row.get("opaque_candidate_id"))
        prompt_id = str(row.get("opaque_prompt_id"))
        sample_id = str(row.get("sample_id"))
        source = risk_by_id.get(sample_id)
        record = record_by_id.get(candidate_id)
        if source is None or record is None:
            raise M5ReviewError("M5 candidate lacks frozen source or record")
        expected_prompt = expected_prompts.get(prompt_id)
        if (
            expected_prompt is None
            or expected_prompt["sample_id"] != sample_id
            or expected_prompt["prompt"] != row.get("prompt")
        ):
            raise M5ReviewError("M5 candidate uses an unregistered prompt")
        request_id = str(row.get("request_id"))
        if not REQUEST_ID_RE.fullmatch(request_id):
            raise M5ReviewError("M5 request ID format changed")
        mode = str(row.get("author_mode"))
        modes = list(protocol["authoring"]["modes"])
        if mode not in modes:
            raise M5ReviewError("M5 author mode changed")
        request_index = (int(expected_prompt["rank"]) - 1) * len(modes) + modes.index(mode)
        expected_request_id = f"m5_req_{request_index:03d}"
        expected_shard = f"{request_index % int(protocol['authoring']['shards']):02d}"
        if (
            request_id != expected_request_id
            or row.get("authoring_shard") != expected_shard
            or row.get("authoring_agent_id") != m5_authoring.AGENTS[expected_shard]
        ):
            raise M5ReviewError("M5 request assignment changed")
        expected_candidate_id = (
            "m5_candidate_"
            + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:24]
        )
        if candidate_id != expected_candidate_id:
            raise M5ReviewError("M5 candidate ID derivation changed")
        public = {key: row.get(key) for key in record}
        if public != record:
            raise M5ReviewError("M5 candidate public record changed")
        prompt = row.get("prompt")
        response = row.get("response")
        response_tokens = len(response.split()) if isinstance(response, str) else -1
        required_opening = protocol["authoring"]["required_openings"][mode]
        if (
            prompt != source.get("prompt")
            or source.get("family") != "risk"
            or source.get("source") != "PKU-SafeRLHF-QA/Alpaca3-8B"
            or source.get("is_safe_source") is not False
            or not isinstance(response, str)
            or not response
            or response != response.strip()
            or response_tokens != row.get("response_whitespace_tokens")
            or not int(protocol["authoring"]["minimum_response_whitespace_tokens"])
            <= response_tokens
            <= int(protocol["authoring"]["maximum_response_whitespace_tokens"])
            or not response.startswith(required_opening)
            or sha256_bytes(response.encode("utf-8")) != row.get("response_sha256")
            or sha256_bytes(m5_authoring.normalize(response).encode("utf-8"))
            != row.get("normalized_response_sha256")
            or "<think" in response.casefold()
            or "</think" in response.casefold()
            or row.get("authoring_source") != "current_model_no_context_subagents"
            or row.get("model_provenance") != "inherited_parent_model_unversioned"
        ):
            raise M5ReviewError("M5 candidate content seal changed")
        if candidate_id in candidate_ids or row["normalized_response_sha256"] in normalized_hashes:
            raise M5ReviewError("M5 candidate identity or response repeated")
        candidate_ids.add(candidate_id)
        normalized_hashes.add(row["normalized_response_sha256"])
        prompt_modes[prompt_id].add(mode)
        metadata = {
            "opaque_prompt_id": prompt_id,
            "sample_id": sample_id,
            "source_index": source["source_index"],
            "assigned_category": source["harm_category"],
            "severity_level": source["severity_level"],
            "prompt": prompt,
        }
        previous = prompt_metadata.setdefault(prompt_id, metadata)
        if previous != metadata:
            raise M5ReviewError("M5 prompt metadata changed across candidates")
        primary.append(
            {
                **row,
                "source_index": source["source_index"],
                "assigned_category": source["harm_category"],
                "severity_level": source["severity_level"],
            }
        )
    expected_modes = set(protocol["authoring"]["modes"])
    if (
        len(prompt_metadata) != 60
        or set(prompt_metadata) != set(expected_prompts)
        or {str(row["sample_id"]) for row in primary}
        != {str(row["sample_id"]) for row in selected_rows}
        or any(modes != expected_modes for modes in prompt_modes.values())
        or any(sum(row["opaque_prompt_id"] == prompt_id for row in primary) != 2 for prompt_id in prompt_metadata)
    ):
        raise M5ReviewError("M5 prompt pairing changed")
    prompts = sorted(prompt_metadata.values(), key=lambda row: row["opaque_prompt_id"])
    p0 = step3q.load_registry(check_input(reg["frozen_inputs"]["step3q_p0_registry"]))
    return p0, prompts, primary


def build_anchors(
    reg: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return m3.build_anchors(reg, prompts, primary)


def namespace(reg: Mapping[str, Any], protocol_commit: str) -> str:
    return stable(
        reg["route_id"],
        "namespace",
        protocol_commit,
        reg["authoring_seal"]["primary_candidates"]["sha256"],
        sha256_file(REGISTRY),
    )


def build_packets(reg: Mapping[str, Any], protocol_commit: str) -> dict[str, Any]:
    p0, prompts, primary = load_primary(reg)
    anchors = build_anchors(reg, prompts, primary)
    ns = namespace(reg, protocol_commit)
    packages: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role in REVIEWERS_AB:
        visible: list[dict[str, Any]] = []
        hidden: list[dict[str, Any]] = []
        primary_visible: dict[str, str] = {}

        def add(
            item_role: str,
            canonical_id: str,
            prompt: str,
            response: str,
            expected: Mapping[str, Any] | None,
            purpose: str,
            duplicate_parent: str | None = None,
        ) -> str:
            visible_id = "m5_review_" + stable(ns, purpose, role, canonical_id)[:24]
            visible.append(
                {"opaque_candidate_id": visible_id, "prompt": prompt, "response": response}
            )
            hidden.append(
                {
                    "opaque_candidate_id": visible_id,
                    "item_role": item_role,
                    "canonical_candidate_id": canonical_id,
                    "duplicate_parent_id": duplicate_parent,
                    "expected": dict(expected) if expected is not None else None,
                }
            )
            return visible_id

        for row in primary:
            candidate_id = row["opaque_candidate_id"]
            primary_visible[candidate_id] = add(
                "primary", candidate_id, row["prompt"], row["response"], None, "primary"
            )
        for anchor in anchors:
            add(
                anchor["item_role"],
                anchor["canonical_id"],
                anchor["prompt"],
                anchor["response"],
                anchor["expected"],
                anchor["item_role"],
            )
        duplicate_ids = sorted(
            primary_visible,
            key=lambda value: stable(ns, "duplicate-source", role, value),
        )[:12]
        primary_by_id = {row["opaque_candidate_id"]: row for row in primary}
        for candidate_id in duplicate_ids:
            row = primary_by_id[candidate_id]
            add(
                "duplicate",
                candidate_id,
                row["prompt"],
                row["response"],
                None,
                "duplicate",
                primary_visible[candidate_id],
            )
        order = sorted(
            range(len(visible)),
            key=lambda index: stable(
                ns, "packet-order", role, visible[index]["opaque_candidate_id"]
            ),
        )
        visible = [visible[index] for index in order]
        hidden = [hidden[index] for index in order]
        if len(visible) != 180 or len({row["opaque_candidate_id"] for row in visible}) != 180:
            raise M5ReviewError(f"M5 packet denominator or ID collision: {role}")
        packages[role] = {"packet": visible, "hidden_map": hidden}
    return {
        "p0": p0,
        "prompts": prompts,
        "primary": primary,
        "anchors": anchors,
        "namespace": ns,
        "packages": packages,
    }


def expected_base_files(
    reg: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, bytes]:
    files = {
        "rubric.md": check_input(reg["frozen_inputs"]["review_rubric"]).read_bytes(),
        "finalize_review.py": check_input(
            reg["frozen_inputs"]["review_finalizer"]
        ).read_bytes(),
        "private/primary_candidates.jsonl": jsonl_bytes(bundle["primary"]),
        "private/anchors.jsonl": jsonl_bytes(bundle["anchors"]),
    }
    for role in REVIEWERS_AB:
        files[f"packets/{role}.jsonl"] = jsonl_bytes(
            bundle["packages"][role]["packet"]
        )
        files[f"private/{role}_hidden_map.jsonl"] = jsonl_bytes(
            bundle["packages"][role]["hidden_map"]
        )
    return dict(sorted(files.items()))


def artifact_records(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": path, "bytes": len(data), "sha256": sha256_bytes(data)}
        for path, data in sorted(files.items())
    ]


def prepare_review(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    protocol_commit = require_locked_clean(reg)
    target = require_review_root(reg, review_root, new=True)
    bundle = build_packets(reg, protocol_commit)
    files = expected_base_files(reg, bundle)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        def emit(relative: str, data: bytes) -> None:
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        for relative, data in files.items():
            emit(relative, data)
        manifest = {
            "schema": "step3_cdro_opc_m5_review_spool_manifest_v1",
            "route_id": reg["route_id"],
            "protocol_commit": protocol_commit,
            "lock_commit": git("rev-parse", "HEAD"),
            "registry_sha256": sha256_file(REGISTRY),
            "namespace_sha256": bundle["namespace"],
            "artifacts": artifact_records(files),
            "reviewers_launched": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
            "core100_opened": False,
            "heldout_opened": False,
        }
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "route": "READY_CDRO_OPC_M5_REVIEW",
        "review_root": str(target),
        "rows_per_reviewer": 180,
        "primary_candidates": 120,
        "anchors": 48,
        "hidden_duplicates": 12,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def verify_base(
    reg: Mapping[str, Any], review_root: Path
) -> tuple[Path, dict[str, Any]]:
    protocol_commit = require_locked_clean(reg)
    root = require_review_root(reg, review_root)
    bundle = build_packets(reg, protocol_commit)
    expected_files = expected_base_files(reg, bundle)
    expected_artifacts = artifact_records(expected_files)
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("schema") != "step3_cdro_opc_m5_review_spool_manifest_v1"
        or manifest.get("route_id") != reg["route_id"]
        or manifest.get("protocol_commit") != protocol_commit
        or manifest.get("lock_commit") != git("rev-parse", "HEAD")
        or manifest.get("registry_sha256") != sha256_file(REGISTRY)
        or manifest.get("namespace_sha256") != bundle["namespace"]
        or manifest.get("artifacts") != expected_artifacts
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise M5ReviewError("invalid M5 review spool manifest")
    expected_artifact_paths = set(expected_files)
    if {artifact.get("path") for artifact in manifest.get("artifacts", [])} != expected_artifact_paths:
        raise M5ReviewError("M5 spool base artifact set changed")
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise M5ReviewError(f"M5 spool artifact changed: {artifact['path']}")
        if path.read_bytes() != expected_files[artifact["path"]]:
            raise M5ReviewError(f"M5 spool artifact content differs from deterministic packet: {artifact['path']}")
    allowed_paths = expected_artifact_paths | {
        "manifest.json",
        *(f"launches/{role}.json" for role in (*REVIEWERS_AB, REVIEWER_C)),
        *(f"reviews/{role}.jsonl" for role in (*REVIEWERS_AB, REVIEWER_C)),
        *(f"reviews/{role}_manifest.json" for role in (*REVIEWERS_AB, REVIEWER_C)),
        "outcomes/ab_outcome.json",
        "outcomes/final_decision.json",
        "packets/reviewer_c.jsonl",
        "private/ab_primary_reviews.json",
        "private/ab_agreement_labels.json",
        "private/reviewer_c_hidden_map.jsonl",
        "private/final_labels.json",
        "private/cohort.jsonl",
        "private/selected_targets.jsonl",
    }
    observed_paths = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise M5ReviewError(f"M5 spool contains a symlink: {path}")
        if path.is_file():
            observed_paths.add(path.relative_to(root).as_posix())
    unexpected = observed_paths - allowed_paths
    if unexpected:
        raise M5ReviewError(f"M5 spool contains unregistered files: {sorted(unexpected)}")
    return root, manifest


def packet_for(root: Path, role: str) -> Path:
    path = root / "packets" / f"{role}.jsonl"
    if not path.is_file():
        raise M5ReviewError(f"missing packet for {role}")
    return path


def record_launch(
    reg: Mapping[str, Any], review_root: Path, role: str, agent_id: str
) -> dict[str, Any]:
    root, manifest = verify_base(reg, review_root)
    allowed = set(REVIEWERS_AB) | {REVIEWER_C}
    if role not in allowed or not agent_id.strip():
        raise M5ReviewError("invalid M5 reviewer launch")
    for existing_role in (*REVIEWERS_AB, REVIEWER_C):
        path = root / "launches" / f"{existing_role}.json"
        if path.is_file() and load_launch(root, existing_role)["agent_id"] == agent_id:
            raise M5ReviewError("M5 reviewer agent identity is already in use")
    if role == REVIEWER_C:
        outcome = verify_ab_state(reg, root, manifest)["outcome"]
        if (
            outcome.get("route") != "PASS_CDRO_OPC_M5_REVIEW_AB"
            or outcome.get("reviewer_c_required") is not True
        ):
            raise M5ReviewError("Reviewer C is not authorized by M5 A/B outcome")
        ab_agents = {
            read_json(root / "launches" / f"{ab_role}.json")["agent_id"]
            for ab_role in REVIEWERS_AB
        }
        if agent_id in ab_agents:
            raise M5ReviewError("Reviewer C must be a new independent agent")
    value = {
        "schema": "step3_cdro_opc_m5_review_launch_v1",
        "role": role,
        "agent_id": agent_id,
        "context_isolated": True,
        "packet_sha256": sha256_file(packet_for(root, role)),
        "rubric_sha256": sha256_file(root / "rubric.md"),
        "saw_repository": False,
        "saw_hidden_map": False,
        "saw_other_full_reviews": False,
        "saw_ab_short_reasons": role == REVIEWER_C,
        "saw_training_or_gradient_results": False,
    }
    write_verified_files(root, {f"launches/{role}.json": json_bytes(value)})
    return {"role": role, "agent_id": agent_id, "recorded": True}


def load_launch(root: Path, role: str) -> dict[str, Any]:
    path = root / "launches" / f"{role}.json"
    launch = read_json(path)
    expected = {
        "schema": "step3_cdro_opc_m5_review_launch_v1",
        "role": role,
        "context_isolated": True,
        "packet_sha256": sha256_file(packet_for(root, role)),
        "rubric_sha256": sha256_file(root / "rubric.md"),
        "saw_repository": False,
        "saw_hidden_map": False,
        "saw_other_full_reviews": False,
        "saw_ab_short_reasons": role == REVIEWER_C,
        "saw_training_or_gradient_results": False,
    }
    for key, value in expected.items():
        if launch.get(key) != value:
            raise M5ReviewError(f"M5 launch provenance changed: {role}.{key}")
    if type(launch.get("agent_id")) is not str or not launch["agent_id"].strip():
        raise M5ReviewError(f"M5 launch agent identity is invalid: {role}")
    return launch


def validate_review(
    root: Path, role: str, path: Path
) -> tuple[list[dict[str, Any]], bytes]:
    packet = read_jsonl(packet_for(root, role))
    data = path.read_bytes()
    rows = p2.parse_jsonl_bytes(data, f"{role} review")
    return p2.validate_review_rows(rows, packet, role), data


def submit_review(
    reg: Mapping[str, Any],
    review_root: Path,
    role: str,
    agent_id: str,
    input_path: Path,
) -> dict[str, Any]:
    root, _ = verify_base(reg, review_root)
    launch = load_launch(root, role)
    if launch.get("agent_id") != agent_id:
        raise M5ReviewError("M5 reviewer ID differs from launch record")
    rows, data = validate_review(root, role, input_path)
    record = {
        "schema": "step3_cdro_opc_m5_review_submission_v1",
        "role": role,
        "agent_id": agent_id,
        "packet_sha256": sha256_file(packet_for(root, role)),
        "rubric_sha256": sha256_file(root / "rubric.md"),
        "rows": len(rows),
        "review_sha256": sha256_bytes(data),
        "immutable_submission": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_verified_files(
        root,
        {
            f"reviews/{role}.jsonl": data,
            f"reviews/{role}_manifest.json": json_bytes(record),
        },
    )
    return {"route": "CDRO_OPC_M5_REVIEW_ACCEPTED", **record}


def stored_review(root: Path, role: str) -> list[dict[str, Any]]:
    review_path = root / "reviews" / f"{role}.jsonl"
    manifest = read_json(root / "reviews" / f"{role}_manifest.json")
    launch = load_launch(root, role)
    if (
        manifest.get("schema") != "step3_cdro_opc_m5_review_submission_v1"
        or manifest.get("role") != role
        or manifest.get("agent_id") != launch.get("agent_id")
        or manifest.get("packet_sha256") != launch.get("packet_sha256")
        or manifest.get("rubric_sha256") != launch.get("rubric_sha256")
        or manifest.get("review_sha256") != sha256_file(review_path)
        or manifest.get("immutable_submission") is not True
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise M5ReviewError(f"stored M5 review provenance changed: {role}")
    rows, _ = validate_review(root, role, review_path)
    if manifest.get("rows") != len(rows):
        raise M5ReviewError(f"stored M5 review denominator changed: {role}")
    return rows


def review_by_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {row["opaque_candidate_id"]: dict(row) for row in rows}
    if len(result) != len(rows):
        raise M5ReviewError("review repeats visible IDs")
    return result


def derive_ab(
    reg: Mapping[str, Any], root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    p0, _, primary_rows = load_primary(reg)
    rows: dict[str, list[dict[str, Any]]] = {}
    hidden: dict[str, list[dict[str, Any]]] = {}
    launches = []
    scores = {}
    canonical: dict[str, dict[str, dict[str, Any]]] = {}
    for role in REVIEWERS_AB:
        rows[role] = stored_review(root, role)
        hidden[role] = read_jsonl(root / "private" / f"{role}_hidden_map.jsonl")
        launches.append(load_launch(root, role))
        scores[role], canonical[role] = m3.score_one_reviewer(
            reg, rows[role], hidden[role], role
        )
    if len({value["agent_id"] for value in launches}) != 2:
        raise M5ReviewError("M5 reviewer identities are not independent")
    reliability = m3.primary_reliability(
        reg,
        primary_rows,
        canonical[REVIEWERS_AB[0]],
        canonical[REVIEWERS_AB[1]],
        p0,
        manifest["namespace_sha256"],
    )
    disagreements = []
    disagreement_sources = []
    agreement_labels = {}
    for row in primary_rows:
        candidate_id = row["opaque_candidate_id"]
        label_a = qualification.candidate_qualified(canonical[REVIEWERS_AB[0]][candidate_id])
        label_b = qualification.candidate_qualified(canonical[REVIEWERS_AB[1]][candidate_id])
        if label_a == label_b:
            agreement_labels[candidate_id] = label_a
        else:
            disagreements.append(
                {
                    "opaque_candidate_id": "m5_review_"
                    + stable(manifest["namespace_sha256"], "reviewer-c", candidate_id)[:24],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "reviewer_a_short_reason": canonical[REVIEWERS_AB[0]][candidate_id][
                        "short_reason"
                    ],
                    "reviewer_b_short_reason": canonical[REVIEWERS_AB[1]][candidate_id][
                        "short_reason"
                    ],
                }
            )
            disagreement_sources.append(candidate_id)
    passed = all(score["overall_pass"] for score in scores.values()) and reliability[
        "overall_pass"
    ]
    route = (
        "PASS_CDRO_OPC_M5_REVIEW_AB"
        if passed
        else "STOP_CDRO_OPC_M5_REVIEW_UNRELIABLE"
    )
    outcome = {
        "schema": "step3_cdro_opc_m5_review_ab_outcome_v1",
        "route": route,
        "reviewer_scores": scores,
        "primary_reliability": reliability,
        "derived_qualification_agreements": len(agreement_labels),
        "derived_qualification_disagreements": len(disagreements),
        "reviewer_c_required": passed and bool(disagreements),
        "reviewer_c_forbidden": not passed or not disagreements,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    files = {
        "outcomes/ab_outcome.json": json_bytes(outcome),
        "private/ab_primary_reviews.json": json_bytes(canonical),
        "private/ab_agreement_labels.json": json_bytes(agreement_labels),
    }
    if passed and disagreements:
        hidden_c = [
            {
                "opaque_candidate_id": visible["opaque_candidate_id"],
                "canonical_candidate_id": candidate_id,
            }
            for visible, candidate_id in zip(disagreements, disagreement_sources, strict=True)
        ]
        order = sorted(
            range(len(disagreements)),
            key=lambda index: stable(
                manifest["namespace_sha256"],
                "reviewer-c-order",
                disagreements[index]["opaque_candidate_id"],
            ),
        )
        files["packets/reviewer_c.jsonl"] = jsonl_bytes(
            [disagreements[index] for index in order]
        )
        files["private/reviewer_c_hidden_map.jsonl"] = jsonl_bytes(
            [hidden_c[index] for index in order]
        )
    return {
        "outcome": outcome,
        "canonical": canonical,
        "agreement_labels": agreement_labels,
        "files": files,
    }


def verify_ab_state(
    reg: Mapping[str, Any], root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    derived = derive_ab(reg, root, manifest)
    for relative, expected in derived["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.read_bytes() != expected:
            raise M5ReviewError(f"stored M5 A/B artifact differs from derivation: {relative}")
    outcome = derived["outcome"]
    c_paths = [
        root / "packets/reviewer_c.jsonl",
        root / "private/reviewer_c_hidden_map.jsonl",
        root / "launches/reviewer_c.json",
        root / "reviews/reviewer_c.jsonl",
        root / "reviews/reviewer_c_manifest.json",
    ]
    if not outcome["reviewer_c_required"] and any(path.exists() for path in c_paths):
        raise M5ReviewError("Reviewer C artifacts exist when C is forbidden")
    return derived


def analyze_ab(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    root, manifest = verify_base(reg, review_root)
    derived = derive_ab(reg, root, manifest)
    write_verified_files(root, derived["files"])
    verify_ab_state(reg, root, manifest)
    return derived["outcome"]


def derive_final(
    reg: Mapping[str, Any], root: Path, ab_state: Mapping[str, Any]
) -> dict[str, Any]:
    outcome = ab_state["outcome"]
    if outcome.get("route") != "PASS_CDRO_OPC_M5_REVIEW_AB":
        raise M5ReviewError("M5 A/B review did not authorize finalization")
    labels = dict(ab_state["agreement_labels"])
    if outcome["reviewer_c_required"]:
        rows_c = stored_review(root, REVIEWER_C)
        hidden_c = read_jsonl(root / "private" / "reviewer_c_hidden_map.jsonl")
        review_c = review_by_id(rows_c)
        for mapping in hidden_c:
            labels[mapping["canonical_candidate_id"]] = qualification.candidate_qualified(
                review_c[mapping["opaque_candidate_id"]]
            )
    _, _, primary = load_primary(reg)
    if set(labels) != {row["opaque_candidate_id"] for row in primary}:
        raise M5ReviewError("M5 final labels do not cover all primary candidates")
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        by_prompt[row["opaque_prompt_id"]].append(row)
    eligible = [
        prompt_id
        for prompt_id, values in by_prompt.items()
        if len(values) == 2 and all(labels[row["opaque_candidate_id"]] == "yes" for row in values)
    ]
    eligible.sort(key=lambda value: stable(reg["selection"]["seed"], "cohort", value))
    required = int(reg["selection"]["eligible_prompts_required"])
    route = (
        "PASS_CDRO_OPC_M5_TARGET_COHORT"
        if len(eligible) >= required
        else "STOP_CDRO_OPC_M5_TARGET_YIELD"
    )
    selected = eligible[:required] if route.startswith("PASS") else []
    cohort = []
    selected_targets = []
    for rank, prompt_id in enumerate(selected, 1):
        split = "calibration" if rank % 2 == 0 else "sealed_audit"
        cohort.append({"rank": rank, "opaque_prompt_id": prompt_id, "split": split})
        for row in sorted(by_prompt[prompt_id], key=lambda value: value["opaque_candidate_id"]):
            selected_targets.append(
                {
                    "sample_id": row["sample_id"],
                    "opaque_prompt_id": prompt_id,
                    "target_id": row["opaque_candidate_id"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "category": row["assigned_category"],
                    "severity_level": row["severity_level"],
                    "split": split,
                }
            )
    decision = {
        "schema": "step3_cdro_opc_m5_review_final_decision_v1",
        "route": route,
        "final_qualified_candidates": sum(value == "yes" for value in labels.values()),
        "eligible_prompts": len(eligible),
        "selected_prompts": len(selected),
        "selected_targets": len(selected_targets),
        "calibration_units": sum(row["split"] == "calibration" for row in cohort),
        "sealed_audit_units": sum(row["split"] == "sealed_audit" for row in cohort),
        "reviewer_c_used": bool(outcome["reviewer_c_required"]),
        "future_fixed_dose": reg["future_fixed_dose"],
        "gpu_protocol_authorized": route.startswith("PASS"),
        "training_authorized": False,
        "optimizer_steps": 0,
        "core100_opened": False,
        "heldout_opened": False,
    }
    return {
        "decision": decision,
        "labels": labels,
        "cohort": cohort,
        "selected_targets": selected_targets,
        "files": {
            "outcomes/final_decision.json": json_bytes(decision),
            "private/final_labels.json": json_bytes(labels),
            "private/cohort.jsonl": jsonl_bytes(cohort),
            "private/selected_targets.jsonl": jsonl_bytes(selected_targets),
        },
    }


def verify_final_state(
    reg: Mapping[str, Any], root: Path, ab_state: Mapping[str, Any]
) -> dict[str, Any]:
    derived = derive_final(reg, root, ab_state)
    for relative, expected in derived["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.read_bytes() != expected:
            raise M5ReviewError(f"stored M5 final artifact differs from derivation: {relative}")
    return derived


def finalize(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    root, manifest = verify_base(reg, review_root)
    ab_state = verify_ab_state(reg, root, manifest)
    derived = derive_final(reg, root, ab_state)
    write_verified_files(root, derived["files"])
    verify_final_state(reg, root, ab_state)
    return derived["decision"]


def import_results(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    root, manifest = verify_base(reg, review_root)
    ab_state = verify_ab_state(reg, root, manifest)
    outcome = ab_state["outcome"]
    if outcome.get("route") == "PASS_CDRO_OPC_M5_REVIEW_AB":
        final_state = verify_final_state(reg, root, ab_state)
        decision = final_state["decision"]
    else:
        final_paths = [
            root / "outcomes/final_decision.json",
            root / "private/final_labels.json",
            root / "private/cohort.jsonl",
            root / "private/selected_targets.jsonl",
        ]
        if any(path.exists() for path in final_paths):
            raise M5ReviewError("stopped M5 review unexpectedly has final artifacts")
        decision = {
            "route": outcome["route"],
            "gpu_protocol_authorized": False,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
    output = repo_path(str(reg["repository_output"]))
    if output.exists():
        raise M5ReviewError("M5 repository review output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    report = (
        "# CDR-OPC M5 Independent Blind Review\n\n"
        f"A/B route: `{outcome['route']}`. Final route: `{decision['route']}`.\n\n"
        f"Primary candidates: {outcome['primary_reliability']['primary_candidates']}  \n"
        f"Primary prompts: {outcome['primary_reliability']['primary_prompts']}  \n"
        f"Full-vector agreement: {outcome['primary_reliability']['full_vector_exact_agreement']:.6f}  \n"
        f"Qualification agreement: {outcome['primary_reliability']['candidate_qualified']['raw_agreement']:.6f}  \n"
        f"Reviewer C used: {str(bool(decision.get('reviewer_c_used', False))).lower()}  \n"
        f"Eligible prompts: {decision.get('eligible_prompts', 0)}  \n"
        f"Selected prompts: {decision.get('selected_prompts', 0)}\n\n"
        "This stage qualifies a prompt-disjoint target cohort. It does not show a GPU effect, "
        "authorize training, or open Core100 or heldout data.\n\n"
        "```text\noptimizer_steps=0\ntraining_authorized=false\n"
        "core100_opened=false\nheldout_opened=false\n```\n"
    )
    try:
        shutil.copytree(root, temporary, dirs_exist_ok=True)
        (temporary / "CDRO_OPC_M5_REVIEW_REPORT.md").write_text(
            report, encoding="utf-8"
        )
        copied_artifacts = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        summary = {
            "schema": "step3_cdro_opc_m5_review_import_summary_v1",
            "route": decision["route"],
            "protocol_commit": manifest["protocol_commit"],
            "lock_commit": manifest["lock_commit"],
            "reviewer_a_agent_id": load_launch(root, "reviewer_a")["agent_id"],
            "reviewer_b_agent_id": load_launch(root, "reviewer_b")["agent_id"],
            "reviewer_c_used": bool(decision.get("reviewer_c_used", False)),
            "artifacts_before_summary": copied_artifacts,
            "optimizer_steps": 0,
            "training_authorized": False,
            "core100_opened": False,
            "heldout_opened": False,
        }
        (temporary / "review_import_summary.json").write_bytes(json_bytes(summary))
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "route": decision["route"],
        "output": str(output),
        "files": len(copied_artifacts) + 1,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    prepare = sub.add_parser("prepare-review")
    prepare.add_argument("--review-root", required=True)
    launch = sub.add_parser("record-launch")
    launch.add_argument("--review-root", required=True)
    launch.add_argument("--role", required=True)
    launch.add_argument("--agent-id", required=True)
    submit = sub.add_parser("submit-review")
    submit.add_argument("--review-root", required=True)
    submit.add_argument("--role", required=True)
    submit.add_argument("--agent-id", required=True)
    submit.add_argument("--input", required=True)
    analyze = sub.add_parser("analyze-ab")
    analyze.add_argument("--review-root", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--review-root", required=True)
    imported = sub.add_parser("import-results")
    imported.add_argument("--review-root", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    reg = load_registry()
    if args.command == "preflight":
        p0, prompts, primary = load_primary(reg)
        anchors = build_anchors(reg, prompts, primary)
        result = {
            "route": "PASS_CDRO_OPC_M5_REVIEW_PREFLIGHT",
            "primary_candidates": len(primary),
            "primary_prompts": len(prompts),
            "anchor_counts": {
                family: sum(row["item_role"] == family for row in anchors)
                for family in ANCHOR_FAMILIES
            },
            "prompt_bootstrap_replicates": p0["review"]["prompt_bootstrap_replicates"],
            "optimizer_steps": 0,
            "training_authorized": False,
            "core100_opened": False,
            "heldout_opened": False,
        }
    elif args.command == "prepare-review":
        result = prepare_review(reg, Path(args.review_root))
    elif args.command == "record-launch":
        result = record_launch(reg, Path(args.review_root), args.role, args.agent_id)
    elif args.command == "submit-review":
        result = submit_review(
            reg, Path(args.review_root), args.role, args.agent_id, Path(args.input)
        )
    elif args.command == "analyze-ab":
        result = analyze_ab(reg, Path(args.review_root))
    elif args.command == "finalize":
        result = finalize(reg, Path(args.review_root))
    else:
        result = import_results(reg, Path(args.review_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
