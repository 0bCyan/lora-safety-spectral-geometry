"""Run CDR-OPC M3 blind-review transport repair V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3o_o4_qualification as qualification
import step3q_calibrated_api_pool as step3q
import step3q_p2_review as p2


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "experiments/step3_response_quality_persistence"
REGISTRY = ROOT / "configs/4070ti_step3_cdro_opc_m3_review_v3_registry.json"
REVIEWERS_AB = ("reviewer_f", "reviewer_g")
REVIEWER_C = "reviewer_c"
DECISION_FIELDS = (
    "detail_retention",
    "safe_route",
    "executable_step",
    "prompt_binding",
    "naturalness",
    "clean_safe",
)
BINARY_FIELDS = DECISION_FIELDS[1:]
ANCHOR_FAMILIES = (
    "positive_anchor",
    "generic_anchor",
    "mismatch_anchor",
    "clean_safety_anchor",
)


class M3ReviewV2Error(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable(seed: object, purpose: str, *parts: object) -> str:
    material = "\x1f".join(str(value) for value in (seed, purpose, *parts))
    return sha256_bytes(material.encode("utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise M3ReviewV2Error(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p2.parse_jsonl_bytes(path.read_bytes(), str(path))


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M3ReviewV2Error(f"path escapes repository: {value}") from exc
    return path


def check_input(spec: Mapping[str, Any]) -> Path:
    path = repo_path(str(spec["path"]))
    if not path.is_file() or path.is_symlink() or sha256_file(path) != spec["sha256"]:
        raise M3ReviewV2Error(f"frozen input changed: {path}")
    return path


def load_registry() -> dict[str, Any]:
    reg = read_json(REGISTRY)
    if reg.get("schema") != "step3_cdro_opc_m3_review_v3_registry_v1":
        raise M3ReviewV2Error("unexpected M3 review V3 registry schema")
    if reg.get("status") != "prospective_before_v3_packet_construction":
        raise M3ReviewV2Error("M3 review V3 registry is not prospective")
    if reg.get("optimizer_steps") != 0 or reg.get("training_authorized") is not False:
        raise M3ReviewV2Error("M3 review V2 authorization boundary changed")
    for name, value in reg["prior_review_boundary"].items():
        expected = name == "aggregate_stop_reports_only"
        if value is not expected:
            raise M3ReviewV2Error("prior row-level review data became readable")
    for spec in reg["frozen_inputs"].values():
        check_input(spec)
    packet = reg["packet"]
    if (
        packet.get("reviewer_roles") != list(REVIEWERS_AB)
        or packet.get("primary_candidates") != 120
        or sum(packet.get(name, -1) for name in (
            "positive_anchors",
            "generic_anchors",
            "mismatch_anchors",
            "clean_safety_negative_anchors",
        )) != 48
        or packet.get("hidden_primary_duplicates") != 12
        or packet.get("rows_per_reviewer") != 180
        or packet.get("post_submission_revision_allowed") is not False
        or packet.get("heuristic_label_substitution_allowed") is not False
    ):
        raise M3ReviewV2Error("M3 review V2 packet denominator changed")
    return reg


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def require_locked_clean(reg: Mapping[str, Any]) -> str:
    commit = str(reg.get("protocol_commit_sha1"))
    if reg.get("protocol_lock_status") != "locked_to_protocol_commit":
        raise M3ReviewV2Error("M3 review V2 protocol is not locked")
    if git("rev-parse", f"{reg['protocol_tag']}^{{commit}}") != commit:
        raise M3ReviewV2Error("M3 review V2 protocol tag differs from registry lock")
    if git("status", "--porcelain"):
        raise M3ReviewV2Error("M3 review V2 requires a clean checkout")
    changed = git("diff", "--name-only", commit, "HEAD").splitlines()
    expected = [REGISTRY.relative_to(REPO).as_posix()]
    if changed != expected:
        raise M3ReviewV2Error(f"V2 protocol-to-lock diff changed: {changed}")
    return commit


def external(path: Path, *, new: bool = False) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        raise M3ReviewV2Error("V2 review spool must remain outside the repository")
    if new and resolved.exists():
        raise M3ReviewV2Error("V2 review spool already exists")
    return resolved


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load_primary(reg: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    seal = reg["authoring_seal"]
    sealed = external(Path(seal["spool_root"])) / "sealed"
    candidates_path = sealed / "candidates.jsonl"
    summary_path = sealed / "authoring_summary.json"
    if (
        sha256_file(candidates_path) != seal["candidates_sha256"]
        or sha256_file(summary_path) != seal["summary_sha256"]
    ):
        raise M3ReviewV2Error("M3 authoring seal changed")
    summary = read_json(summary_path)
    if (
        summary.get("route") != seal["required_route"]
        or summary.get("valid_unique_candidates") != 120
        or summary.get("valid_prompts") != 60
    ):
        raise M3ReviewV2Error("M3 authoring denominator changed")
    prompts = read_jsonl(check_input(reg["frozen_inputs"]["prompts_hidden"]))
    prompt_by_id = {row["opaque_prompt_id"]: row for row in prompts}
    if len(prompt_by_id) != 60:
        raise M3ReviewV2Error("M3 hidden prompt denominator changed")
    candidates = read_jsonl(candidates_path)
    primary = []
    for row in candidates:
        metadata = prompt_by_id.get(row.get("opaque_prompt_id"))
        if metadata is None or row.get("prompt") != metadata.get("prompt"):
            raise M3ReviewV2Error("candidate/prompt hidden mapping changed")
        primary.append(
            {
                **row,
                "sample_id": metadata["sample_id"],
                "source_index": metadata["source_index"],
                "assigned_category": metadata["assigned_category"],
                "severity_level": metadata["severity_level"],
            }
        )
    if (
        len(primary) != 120
        or len({row["opaque_candidate_id"] for row in primary}) != 120
        or len({row["opaque_prompt_id"] for row in primary}) != 60
    ):
        raise M3ReviewV2Error("M3 primary candidate denominator changed")
    p0 = step3q.load_registry(check_input(reg["frozen_inputs"]["step3q_p0_registry"]))
    return p0, prompts, primary


def build_anchors(reg: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]], primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seed = reg["packet"]["anchor_seed"]
    calibration_packet = {
        row["opaque_candidate_id"]: row
        for row in read_jsonl(check_input(reg["frozen_inputs"]["calibration_packet"]))
    }
    references = read_jsonl(check_input(reg["frozen_inputs"]["calibration_reference"]))
    positive_ids = [
        row["opaque_candidate_id"]
        for row in references
        if row.get("calibration_role") == "component_anchor"
        and row.get("expected")
        == {"safe_route": "yes", "executable_step": "yes", "prompt_binding": "yes"}
    ]
    positive_ids = sorted(positive_ids, key=lambda value: stable(seed, "positive", value))[:12]
    if len(positive_ids) != 12:
        raise M3ReviewV2Error("insufficient positive anchors")
    anchors: list[dict[str, Any]] = [
        {
            "canonical_id": f"positive::{candidate_id}",
            "item_role": "positive_anchor",
            "prompt": calibration_packet[candidate_id]["prompt"],
            "response": calibration_packet[candidate_id]["response"],
            "expected": {"safe_route": "yes", "executable_step": "yes", "prompt_binding": "yes"},
        }
        for candidate_id in positive_ids
    ]

    ordered_prompts = sorted(prompts, key=lambda row: stable(seed, "generic", row["opaque_prompt_id"]))
    for row in ordered_prompts[:12]:
        anchors.append(
            {
                "canonical_id": f"generic::{row['opaque_prompt_id']}",
                "item_role": "generic_anchor",
                "prompt": row["prompt"],
                "response": reg["packet"]["generic_response"],
                "expected": {
                    "detail_retention": 0,
                    "safe_route": "no",
                    "executable_step": "no",
                    "prompt_binding": "no",
                    "clean_safe": "yes",
                },
            }
        )

    source_candidates = sorted(primary, key=lambda row: stable(seed, "mismatch-source", row["opaque_candidate_id"]))
    used_targets: set[str] = set()
    mismatch_count = 0
    for source in source_candidates:
        targets = [
            row
            for row in prompts
            if row["opaque_prompt_id"] != source["opaque_prompt_id"]
            and row["assigned_category"] != source["assigned_category"]
            and row["opaque_prompt_id"] not in used_targets
        ]
        if not targets:
            continue
        target = min(
            targets,
            key=lambda row: stable(seed, "mismatch-target", source["opaque_candidate_id"], row["opaque_prompt_id"]),
        )
        used_targets.add(target["opaque_prompt_id"])
        anchors.append(
            {
                "canonical_id": f"mismatch::{source['opaque_candidate_id']}::{target['opaque_prompt_id']}",
                "item_role": "mismatch_anchor",
                "prompt": target["prompt"],
                "response": source["response"],
                "expected": {"detail_retention": 0, "prompt_binding": "no"},
            }
        )
        mismatch_count += 1
        if mismatch_count == 12:
            break
    if mismatch_count != 12:
        raise M3ReviewV2Error("could not construct twelve category-mismatched anchors")

    clean_negative_ids = [
        row["opaque_candidate_id"]
        for row in references
        if row.get("calibration_role") == "clean_safety_anchor"
        and row.get("expected") == {"clean_safe": "no"}
    ]
    clean_negative_ids = sorted(
        clean_negative_ids,
        key=lambda value: stable(seed, "clean-safety-negative", value),
    )[:12]
    if len(clean_negative_ids) != 12:
        raise M3ReviewV2Error("insufficient clean-safety negative anchors")
    for candidate_id in clean_negative_ids:
        source = calibration_packet[candidate_id]
        anchors.append(
            {
                "canonical_id": f"clean-safety::{candidate_id}",
                "item_role": "clean_safety_anchor",
                "prompt": source["prompt"],
                "response": source["response"],
                "expected": {"clean_safe": "no"},
            }
        )
    counts = {family: sum(row["item_role"] == family for row in anchors) for family in ANCHOR_FAMILIES}
    if set(counts.values()) != {12}:
        raise M3ReviewV2Error(f"V3 anchor denominator changed: {counts}")
    return anchors


def namespace(reg: Mapping[str, Any], protocol_commit: str) -> str:
    return stable(
        reg["route_id"],
        "namespace",
        protocol_commit,
        reg["authoring_seal"]["candidates_sha256"],
        sha256_file(REGISTRY),
    )


def build_packets(reg: Mapping[str, Any], protocol_commit: str) -> dict[str, Any]:
    p0, prompts, primary = load_primary(reg)
    anchors = build_anchors(reg, prompts, primary)
    ns = namespace(reg, protocol_commit)
    packages = {}
    for role in REVIEWERS_AB:
        visible: list[dict[str, Any]] = []
        hidden: list[dict[str, Any]] = []
        primary_visible: dict[str, str] = {}

        def add(item_role: str, canonical_id: str, prompt: str, response: str, expected: Mapping[str, Any] | None, purpose: str, duplicate_parent: str | None = None) -> str:
            visible_id = "step3q_review_" + stable(ns, purpose, role, canonical_id)[:24]
            visible.append({"opaque_candidate_id": visible_id, "prompt": prompt, "response": response})
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
            key=lambda index: stable(ns, "packet-order", role, visible[index]["opaque_candidate_id"]),
        )
        visible = [visible[index] for index in order]
        hidden = [hidden[index] for index in order]
        if len(visible) != 180 or len({row["opaque_candidate_id"] for row in visible}) != 180:
            raise M3ReviewV2Error(f"V2 packet denominator or ID collision: {role}")
        packages[role] = {"packet": visible, "hidden_map": hidden}
    return {"p0": p0, "prompts": prompts, "primary": primary, "anchors": anchors, "namespace": ns, "packages": packages}


def prepare_review(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    protocol_commit = require_locked_clean(reg)
    target = external(review_root, new=True)
    bundle = build_packets(reg, protocol_commit)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        artifacts: list[dict[str, Any]] = []

        def emit(relative: str, data: bytes) -> None:
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            artifacts.append({"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)})

        emit("rubric.md", check_input(reg["frozen_inputs"]["review_rubric"]).read_bytes())
        emit("finalize_review.py", check_input(reg["frozen_inputs"]["review_finalizer"]).read_bytes())
        emit("private/primary_candidates.jsonl", jsonl_bytes(bundle["primary"]))
        emit("private/anchors.jsonl", jsonl_bytes(bundle["anchors"]))
        for role in REVIEWERS_AB:
            emit(f"packets/{role}.jsonl", jsonl_bytes(bundle["packages"][role]["packet"]))
            emit(f"private/{role}_hidden_map.jsonl", jsonl_bytes(bundle["packages"][role]["hidden_map"]))
        manifest = {
            "schema": "step3_cdro_opc_m3_review_v3_spool_manifest_v1",
            "route_id": reg["route_id"],
            "protocol_commit": protocol_commit,
            "lock_commit": git("rev-parse", "HEAD"),
            "namespace_sha256": bundle["namespace"],
            "artifacts": artifacts,
            "attempt_001_row_level_inputs": 0,
            "reviewers_launched": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "route": "READY_M3_REVIEW_V3",
        "review_root": str(target),
        "rows_per_reviewer": 180,
        "primary_candidates": 120,
        "anchors": 48,
        "hidden_duplicates": 12,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def verify_base(reg: Mapping[str, Any], review_root: Path) -> tuple[Path, dict[str, Any]]:
    root = external(review_root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != "step3_cdro_opc_m3_review_v3_spool_manifest_v1":
        raise M3ReviewV2Error("invalid V3 spool manifest")
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise M3ReviewV2Error(f"V2 spool artifact changed: {artifact['path']}")
    return root, manifest


def packet_for(root: Path, role: str) -> Path:
    path = root / "packets" / f"{role}.jsonl"
    if not path.is_file():
        raise M3ReviewV2Error(f"missing packet for {role}")
    return path


def record_launch(reg: Mapping[str, Any], review_root: Path, role: str, agent_id: str) -> dict[str, Any]:
    root, _ = verify_base(reg, review_root)
    allowed = set(REVIEWERS_AB) | {REVIEWER_C}
    if role not in allowed or not agent_id.strip():
        raise M3ReviewV2Error("invalid V2 reviewer launch")
    if role == REVIEWER_C:
        outcome = read_json(root / "outcomes" / "ab_outcome.json")
        if outcome.get("route") != "PASS_M3_REVIEW_V3_AB" or outcome.get("reviewer_c_required") is not True:
            raise M3ReviewV2Error("Reviewer C is not authorized by V3 A/B outcome")
    value = {
        "schema": "step3_cdro_opc_m3_review_v3_launch_v1",
        "role": role,
        "agent_id": agent_id,
        "context_isolated": True,
        "packet_sha256": sha256_file(packet_for(root, role)),
        "rubric_sha256": sha256_file(root / "rubric.md"),
        "saw_attempt_001_row_labels": False,
        "saw_hidden_map": False,
        "saw_other_reviews": False,
        "saw_training_or_gradient_results": False,
    }
    write_new(root / "launches" / f"{role}.json", json_bytes(value))
    return {"role": role, "agent_id": agent_id, "recorded": True}


def validate_review(root: Path, role: str, path: Path) -> tuple[list[dict[str, Any]], bytes]:
    packet = read_jsonl(packet_for(root, role))
    data = path.read_bytes()
    rows = p2.parse_jsonl_bytes(data, f"{role} review")
    return p2.validate_review_rows(rows, packet, role), data


def submit_review(reg: Mapping[str, Any], review_root: Path, role: str, agent_id: str, input_path: Path) -> dict[str, Any]:
    root, _ = verify_base(reg, review_root)
    launch = read_json(root / "launches" / f"{role}.json")
    if launch.get("agent_id") != agent_id:
        raise M3ReviewV2Error("V2 reviewer ID differs from launch record")
    rows, data = validate_review(root, role, input_path)
    write_new(root / "reviews" / f"{role}.jsonl", data)
    record = {
        "schema": "step3_cdro_opc_m3_review_v3_submission_v1",
        "role": role,
        "agent_id": agent_id,
        "rows": len(rows),
        "review_sha256": sha256_bytes(data),
        "immutable_submission": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_new(root / "reviews" / f"{role}_manifest.json", json_bytes(record))
    return {"route": "M3_REVIEW_V3_ACCEPTED", **record}


def stored_review(root: Path, role: str) -> list[dict[str, Any]]:
    review_path = root / "reviews" / f"{role}.jsonl"
    manifest = read_json(root / "reviews" / f"{role}_manifest.json")
    launch = read_json(root / "launches" / f"{role}.json")
    if (
        manifest.get("schema") != "step3_cdro_opc_m3_review_v3_submission_v1"
        or manifest.get("role") != role
        or manifest.get("agent_id") != launch.get("agent_id")
        or manifest.get("review_sha256") != sha256_file(review_path)
        or manifest.get("immutable_submission") is not True
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise M3ReviewV2Error(f"stored V2 review provenance changed: {role}")
    rows, _ = validate_review(root, role, review_path)
    if manifest.get("rows") != len(rows):
        raise M3ReviewV2Error(f"stored V2 review denominator changed: {role}")
    return rows


def review_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {row["opaque_candidate_id"]: dict(row) for row in rows}
    if len(result) != len(rows):
        raise M3ReviewV2Error("review repeats visible IDs")
    return result


def score_one_reviewer(reg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], hidden: Sequence[Mapping[str, Any]], role: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    review = review_by_id(rows)
    primary: dict[str, dict[str, Any]] = {}
    duplicate_pairs = []
    families: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for mapping in hidden:
        visible_id = mapping["opaque_candidate_id"]
        judged = review.get(visible_id)
        if judged is None:
            raise M3ReviewV2Error(f"{role} hidden map references missing review")
        item_role = mapping["item_role"]
        if item_role == "primary":
            canonical_id = mapping["canonical_candidate_id"]
            if canonical_id in primary:
                raise M3ReviewV2Error(f"{role} repeats canonical primary")
            primary[canonical_id] = judged
        elif item_role == "duplicate":
            duplicate_pairs.append((mapping["duplicate_parent_id"], visible_id))
        elif item_role in ANCHOR_FAMILIES:
            families[item_role].append((judged, mapping["expected"]))
        else:
            raise M3ReviewV2Error(f"unknown hidden item role: {item_role}")
    if len(primary) != 120 or len(duplicate_pairs) != 12 or len(hidden) != 180:
        raise M3ReviewV2Error(f"{role} hidden packet denominator changed")
    duplicate_exact = sum(
        all(review[parent][field] == review[duplicate][field] for field in DECISION_FIELDS)
        for parent, duplicate in duplicate_pairs
    )
    anchor_scores = {}
    gates = {
        "hidden_duplicate_full_vector": duplicate_exact >= int(reg["gates"]["duplicate_full_vector_min_count"])
    }
    for family in ANCHOR_FAMILIES:
        values = families[family]
        if len(values) != 12:
            raise M3ReviewV2Error(f"{role} anchor denominator changed: {family}")
        fields = sorted({field for _, expected in values for field in expected})
        field_correct = {
            field: sum(expected.get(field) == judged.get(field) for judged, expected in values if field in expected)
            for field in fields
        }
        field_denominators = {
            field: sum(field in expected for _, expected in values) for field in fields
        }
        exact = sum(all(judged.get(field) == value for field, value in expected.items()) for judged, expected in values)
        anchor_scores[family] = {
            "rows": len(values),
            "field_correct": field_correct,
            "field_denominators": field_denominators,
            "exact_expected_subvector": exact,
        }
        gates[f"{family}_exact"] = exact >= int(reg["gates"]["anchor_exact_subvector_min_count"])
        for field in fields:
            gates[f"{family}_{field}"] = field_correct[field] >= int(reg["gates"]["anchor_field_correct_min_count"])
    score = {
        "reviewer": role,
        "hidden_duplicate_full_vector": {
            "correct": duplicate_exact,
            "denominator": len(duplicate_pairs),
            "minimum": reg["gates"]["duplicate_full_vector_min_count"],
        },
        "anchors": anchor_scores,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    return score, primary


def metric_pass(value: Any, threshold: float) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and float(value) >= threshold


def primary_reliability(reg: Mapping[str, Any], primary_rows: Sequence[Mapping[str, Any]], a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]], p0: Mapping[str, Any], ns: str) -> dict[str, Any]:
    ids = [row["opaque_candidate_id"] for row in primary_rows]
    if set(ids) != set(a) or set(ids) != set(b):
        raise M3ReviewV2Error("V2 canonical primary review sets differ")
    detail = qualification.detail_agreement_metrics(
        [(a[cid]["detail_retention"], b[cid]["detail_retention"]) for cid in ids]
    )
    components = {
        field: qualification.binary_agreement_metrics([(a[cid][field], b[cid][field]) for cid in ids])
        for field in BINARY_FIELDS
    }
    qualified = qualification.binary_agreement_metrics(
        [(qualification.candidate_qualified(a[cid]), qualification.candidate_qualified(b[cid])) for cid in ids]
    )
    full_vector = sum(
        all(a[cid][field] == b[cid][field] for field in DECISION_FIELDS) for cid in ids
    ) / len(ids)
    bootstrap = p2.prompt_cluster_bootstrap(primary_rows, a, b, p0, ns)
    cfg = reg["gates"]
    gates = {
        "full_vector_exact": metric_pass(full_vector, cfg["primary_full_vector_exact_min"]),
        "detail_exact": metric_pass(detail["exact_agreement"], cfg["detail_exact_min"]),
        "detail_weighted": metric_pass(detail["linear_weighted_agreement"], cfg["detail_weighted_min"]),
        "detail_gwet_ac2": metric_pass(detail["gwet_ac2"], cfg["detail_gwet_ac2_min"]),
        "qualification_raw": metric_pass(qualified["raw_agreement"], cfg["qualification_raw_min"]),
        "qualification_gwet_ac1": metric_pass(qualified["gwet_ac1"], cfg["qualification_gwet_ac1_min"]),
        "qualification_positive": metric_pass(qualified["positive_agreement"], cfg["qualification_positive_agreement_min"]),
        "prompt_bootstrap_q025": metric_pass(bootstrap["q025"], cfg["prompt_bootstrap_q025_min"]),
    }
    for field, metrics in components.items():
        gates[f"{field}_raw"] = metric_pass(metrics["raw_agreement"], cfg["component_raw_min"])
        gates[f"{field}_gwet_ac1"] = metric_pass(metrics["gwet_ac1"], cfg["component_gwet_ac1_min"])
    return {
        "primary_candidates": len(ids),
        "primary_prompts": len({row["opaque_prompt_id"] for row in primary_rows}),
        "full_vector_exact_agreement": full_vector,
        "detail": detail,
        "components": components,
        "candidate_qualified": qualified,
        "prompt_cluster_bootstrap": bootstrap,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "negative_class_reported_not_gated": True,
    }


def analyze_ab(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    root, manifest = verify_base(reg, review_root)
    p0, _, primary_rows = load_primary(reg)
    rows = {}
    hidden = {}
    launches = []
    scores = {}
    canonical = {}
    for role in REVIEWERS_AB:
        rows[role] = stored_review(root, role)
        hidden[role] = read_jsonl(root / "private" / f"{role}_hidden_map.jsonl")
        launches.append(read_json(root / "launches" / f"{role}.json"))
        scores[role], canonical[role] = score_one_reviewer(reg, rows[role], hidden[role], role)
    if len({value["agent_id"] for value in launches}) != 2:
        raise M3ReviewV2Error("V2 reviewer identities are not independent")
    reliability = primary_reliability(
        reg,
        primary_rows,
        canonical[REVIEWERS_AB[0]],
        canonical[REVIEWERS_AB[1]],
        p0,
        manifest["namespace_sha256"],
    )
    disagreements = []
    agreement_labels = {}
    primary_by_id = {row["opaque_candidate_id"]: row for row in primary_rows}
    for candidate_id, row in primary_by_id.items():
        label_a = qualification.candidate_qualified(canonical[REVIEWERS_AB[0]][candidate_id])
        label_b = qualification.candidate_qualified(canonical[REVIEWERS_AB[1]][candidate_id])
        if label_a == label_b:
            agreement_labels[candidate_id] = label_a
        else:
            disagreements.append(
                {
                    "opaque_candidate_id": "step3q_review_"
                    + stable(manifest["namespace_sha256"], "reviewer-c", candidate_id)[:24],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "reviewer_f_short_reason": canonical[REVIEWERS_AB[0]][candidate_id]["short_reason"],
                    "reviewer_g_short_reason": canonical[REVIEWERS_AB[1]][candidate_id]["short_reason"],
                }
            )
    passed = all(score["overall_pass"] for score in scores.values()) and reliability["overall_pass"]
    route = "PASS_M3_REVIEW_V3_AB" if passed else "STOP_M3_REVIEW_V3_UNRELIABLE"
    outcome = {
        "schema": "step3_cdro_opc_m3_review_v3_ab_outcome_v1",
        "route": route,
        "reviewer_scores": scores,
        "primary_reliability": reliability,
        "derived_qualification_agreements": len(agreement_labels),
        "derived_qualification_disagreements": len(disagreements),
        "reviewer_c_required": passed and bool(disagreements),
        "reviewer_c_forbidden": not passed or not disagreements,
        "attempt_001_row_level_inputs": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_new(root / "outcomes" / "ab_outcome.json", json_bytes(outcome))
    write_new(root / "private" / "ab_primary_reviews.json", json_bytes(canonical))
    write_new(root / "private" / "ab_agreement_labels.json", json_bytes(agreement_labels))
    if passed and disagreements:
        hidden_c = []
        for visible, source in zip(disagreements, [
            row for row in primary_rows
            if qualification.candidate_qualified(canonical[REVIEWERS_AB[0]][row["opaque_candidate_id"]])
            != qualification.candidate_qualified(canonical[REVIEWERS_AB[1]][row["opaque_candidate_id"]])
        ]):
            hidden_c.append(
                {
                    "opaque_candidate_id": visible["opaque_candidate_id"],
                    "canonical_candidate_id": source["opaque_candidate_id"],
                }
            )
        order = sorted(
            range(len(disagreements)),
            key=lambda index: stable(manifest["namespace_sha256"], "reviewer-c-order", disagreements[index]["opaque_candidate_id"]),
        )
        write_new(root / "packets" / "reviewer_c.jsonl", jsonl_bytes([disagreements[index] for index in order]))
        write_new(root / "private" / "reviewer_c_hidden_map.jsonl", jsonl_bytes([hidden_c[index] for index in order]))
    return outcome


def finalize(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    root, _ = verify_base(reg, review_root)
    outcome = read_json(root / "outcomes" / "ab_outcome.json")
    if outcome.get("route") != "PASS_M3_REVIEW_V3_AB":
        raise M3ReviewV2Error("V3 A/B review did not authorize finalization")
    labels = read_json(root / "private" / "ab_agreement_labels.json")
    if outcome["reviewer_c_required"]:
        rows_c = stored_review(root, REVIEWER_C)
        hidden_c = read_jsonl(root / "private" / "reviewer_c_hidden_map.jsonl")
        review_c = review_by_id(rows_c)
        for mapping in hidden_c:
            labels[mapping["canonical_candidate_id"]] = qualification.candidate_qualified(
                review_c[mapping["opaque_candidate_id"]]
            )
    elif (root / "packets" / "reviewer_c.jsonl").exists():
        raise M3ReviewV2Error("Reviewer C packet exists when C is forbidden")
    _, _, primary = load_primary(reg)
    if set(labels) != {row["opaque_candidate_id"] for row in primary}:
        raise M3ReviewV2Error("V2 final labels do not cover all primary candidates")
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
    route = "PASS_M3_TARGET_COHORT" if len(eligible) >= required else "STOP_M3_TARGET_YIELD"
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
        "schema": "step3_cdro_opc_m3_review_v3_final_decision_v1",
        "route": route,
        "final_qualified_candidates": sum(value == "yes" for value in labels.values()),
        "eligible_prompts": len(eligible),
        "selected_prompts": len(selected),
        "selected_targets": len(selected_targets),
        "calibration_units": sum(row["split"] == "calibration" for row in cohort),
        "sealed_audit_units": sum(row["split"] == "sealed_audit" for row in cohort),
        "reviewer_c_used": bool(outcome["reviewer_c_required"]),
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    write_new(root / "outcomes" / "final_decision.json", json_bytes(decision))
    write_new(root / "private" / "final_labels.json", json_bytes(labels))
    write_new(root / "private" / "cohort.jsonl", jsonl_bytes(cohort))
    write_new(root / "private" / "selected_targets.jsonl", jsonl_bytes(selected_targets))
    return decision


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
    return result


def main() -> None:
    args = parser().parse_args()
    reg = load_registry()
    if args.command == "preflight":
        p0, prompts, primary = load_primary(reg)
        anchors = build_anchors(reg, prompts, primary)
        result = {
            "route": "PASS_M3_REVIEW_V3_PREFLIGHT",
            "primary_candidates": len(primary),
            "primary_prompts": len(prompts),
            "anchor_counts": {family: sum(row["item_role"] == family for row in anchors) for family in ANCHOR_FAMILIES},
            "prompt_bootstrap_replicates": p0["review"]["prompt_bootstrap_replicates"],
            "attempt_001_row_level_inputs": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
    elif args.command == "prepare-review":
        result = prepare_review(reg, Path(args.review_root))
    elif args.command == "record-launch":
        result = record_launch(reg, Path(args.review_root), args.role, args.agent_id)
    elif args.command == "submit-review":
        result = submit_review(reg, Path(args.review_root), args.role, args.agent_id, Path(args.input))
    elif args.command == "analyze-ab":
        result = analyze_ab(reg, Path(args.review_root))
    else:
        result = finalize(reg, Path(args.review_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
