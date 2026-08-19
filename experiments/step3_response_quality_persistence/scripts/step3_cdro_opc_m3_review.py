"""Build and evaluate blinded CDR-OPC M3 target-qualification packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import step3o_o4_qualification as qualification
import step3q_calibrated_api_pool as step3q
import step3q_p2_review as p2


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "experiments/step3_response_quality_persistence"
REGISTRY = ROOT / "configs/4070ti_step3_cdro_opc_m3_review_registry.json"
ROLES = ("reviewer_a", "reviewer_b")


class M3ReviewError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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
        raise M3ReviewError(f"expected JSON object: {path}")
    return value


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise M3ReviewError(f"path escapes repository: {value}") from exc
    return path


def check_input(spec: Mapping[str, Any]) -> Path:
    path = repo_path(str(spec["path"]))
    if not path.is_file() or path.is_symlink() or sha256_file(path) != spec["sha256"]:
        raise M3ReviewError(f"frozen input changed: {path}")
    return path


def load_registry() -> dict[str, Any]:
    reg = read_json(REGISTRY)
    if reg.get("schema") != "step3_cdro_opc_m3_review_registry_v1":
        raise M3ReviewError("unexpected M3 review registry schema")
    if reg.get("status") != "prospective_before_reviewer_packet_construction":
        raise M3ReviewError("M3 review registry is not prospective")
    if reg.get("optimizer_steps") != 0 or reg.get("training_authorized") is not False:
        raise M3ReviewError("M3 review authorization boundary changed")
    for spec in reg["frozen_inputs"].values():
        check_input(spec)
    review = reg["review"]
    if (
        review.get("roles") != list(ROLES)
        or review.get("primary_candidates") != 120
        or review.get("hidden_sentinels_per_reviewer") != 24
        or review.get("hidden_primary_duplicates_per_reviewer") != 12
        or review.get("rows_per_reviewer") != 156
        or review.get("post_submission_revision_allowed") is not False
        or review.get("heuristic_label_substitution_allowed") is not False
    ):
        raise M3ReviewError("M3 review denominator or boundary changed")
    return reg


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def require_locked_clean(reg: Mapping[str, Any]) -> str:
    commit = str(reg.get("protocol_commit_sha1"))
    if reg.get("protocol_lock_status") != "locked_to_protocol_commit":
        raise M3ReviewError("M3 review protocol is not locked")
    if git("rev-parse", f"{reg['protocol_tag']}^{{commit}}") != commit:
        raise M3ReviewError("M3 review protocol tag differs from registry lock")
    if git("status", "--porcelain"):
        raise M3ReviewError("M3 review execution requires a clean checkout")
    changed = git("diff", "--name-only", commit, "HEAD").splitlines()
    expected = [REGISTRY.relative_to(REPO).as_posix()]
    if changed != expected:
        raise M3ReviewError(f"review protocol-to-lock diff changed: {changed}")
    return commit


def external(path: Path, *, new: bool = False) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        raise M3ReviewError("review spool must remain outside the repository")
    if new and resolved.exists():
        raise M3ReviewError("review spool already exists")
    return resolved


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load_sources(reg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    seal = reg["authoring_seal"]
    root = external(Path(seal["spool_root"])) / "sealed"
    candidate_path = root / "candidates.jsonl"
    summary_path = root / "authoring_summary.json"
    if (
        sha256_file(candidate_path) != seal["candidates_sha256"]
        or sha256_file(summary_path) != seal["summary_sha256"]
    ):
        raise M3ReviewError("M3 authoring seal changed")
    summary = read_json(summary_path)
    if (
        summary.get("route") != seal["required_route"]
        or summary.get("valid_unique_candidates") != seal["candidates"]
        or summary.get("valid_prompts") != seal["prompts"]
    ):
        raise M3ReviewError("M3 authoring route or denominator changed")
    candidates = p2.parse_jsonl_bytes(candidate_path.read_bytes(), "M3 sealed candidates")
    p0_path = check_input(reg["frozen_inputs"]["step3q_p0_registry"])
    p0 = step3q.load_registry(p0_path)
    locator = read_json(check_input(reg["frozen_inputs"]["step3q_p1_locator_registry"]))
    sources = p2._load_p1_review_sources(locator)
    primary = p2.enrich_primary_candidates(candidates, sources, p0)
    if len(primary) != 120 or len({row["opaque_prompt_id"] for row in primary}) != 60:
        raise M3ReviewError("M3 primary enrichment denominator changed")
    return p0, sources, primary


def namespace(reg: Mapping[str, Any], protocol_commit: str) -> str:
    material = "\x1f".join(
        [
            reg["route_id"],
            protocol_commit,
            reg["authoring_seal"]["candidates_sha256"],
            sha256_file(REGISTRY),
        ]
    )
    return sha256_bytes(material.encode("utf-8"))


def packet_bundle(reg: Mapping[str, Any], protocol_commit: str) -> dict[str, Any]:
    p0, sources, primary = load_sources(reg)
    ns = namespace(reg, protocol_commit)
    packages = {
        role: p2.build_reviewer_packet(
            primary,
            sources,
            role,
            ns,
            int(reg["review"]["hidden_primary_duplicates_per_reviewer"]),
        )
        for role in ROLES
    }
    return {"p0": p0, "sources": sources, "primary": primary, "namespace": ns, "packages": packages}


def prepare_review(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    protocol_commit = require_locked_clean(reg)
    target = external(review_root, new=True)
    bundle = packet_bundle(reg, protocol_commit)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        artifacts: list[dict[str, Any]] = []

        def emit(relative: str, data: bytes) -> None:
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            artifacts.append({"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)})

        rubric = check_input(reg["frozen_inputs"]["review_rubric"]).read_bytes()
        finalizer = check_input(reg["frozen_inputs"]["review_finalizer"]).read_bytes()
        emit("rubric.md", rubric)
        emit("finalize_review.py", finalizer)
        emit("private/primary_candidates.jsonl", jsonl_bytes(bundle["primary"]))
        for role in ROLES:
            package = bundle["packages"][role]
            packet = jsonl_bytes(package["packet"])
            if len(package["packet"]) != int(reg["review"]["rows_per_reviewer"]):
                raise M3ReviewError(f"review packet denominator changed: {role}")
            emit(f"packets/{role}.jsonl", packet)
            emit(f"private/{role}_hidden_map.jsonl", jsonl_bytes(package["hidden_map"]))
        manifest = {
            "schema": "step3_cdro_opc_m3_review_spool_manifest_v1",
            "route_id": reg["route_id"],
            "protocol_commit": protocol_commit,
            "lock_commit": git("rev-parse", "HEAD"),
            "namespace_sha256": bundle["namespace"],
            "artifacts": artifacts,
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
        "route": "READY_M3_BLIND_REVIEW",
        "review_root": str(target),
        "rows_per_reviewer": reg["review"]["rows_per_reviewer"],
        "primary_candidates": 120,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def verify_base(reg: Mapping[str, Any], review_root: Path) -> tuple[Path, dict[str, Any]]:
    root = external(review_root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != "step3_cdro_opc_m3_review_spool_manifest_v1":
        raise M3ReviewError("invalid M3 review spool manifest")
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise M3ReviewError(f"review spool artifact changed: {artifact['path']}")
    return root, manifest


def record_launch(reg: Mapping[str, Any], review_root: Path, role: str, agent_id: str) -> dict[str, Any]:
    root, _ = verify_base(reg, review_root)
    if role not in ROLES or not agent_id.strip():
        raise M3ReviewError("invalid reviewer launch")
    value = {
        "schema": "step3_cdro_opc_m3_reviewer_launch_v1",
        "role": role,
        "agent_id": agent_id,
        "context_isolated": True,
        "packet_sha256": sha256_file(root / "packets" / f"{role}.jsonl"),
        "rubric_sha256": sha256_file(root / "rubric.md"),
        "saw_hidden_map": False,
        "saw_other_reviews": False,
        "saw_training_or_gradient_results": False,
    }
    write_new(root / "launches" / f"{role}.json", json_bytes(value))
    return {"role": role, "agent_id": agent_id, "recorded": True}


def validated_review(root: Path, role: str, input_path: Path) -> tuple[list[dict[str, Any]], bytes]:
    packet_path = root / "packets" / f"{role}.jsonl"
    packet = p2.parse_jsonl_bytes(packet_path.read_bytes(), f"{role} packet")
    data = input_path.read_bytes()
    rows = p2.parse_jsonl_bytes(data, f"{role} review")
    return p2.validate_review_rows(rows, packet, role), data


def submit_review(
    reg: Mapping[str, Any], review_root: Path, role: str, agent_id: str, input_path: Path
) -> dict[str, Any]:
    root, _ = verify_base(reg, review_root)
    launch = read_json(root / "launches" / f"{role}.json")
    if launch.get("agent_id") != agent_id:
        raise M3ReviewError("reviewer ID differs from launch record")
    rows, data = validated_review(root, role, input_path)
    output = root / "reviews" / f"{role}.jsonl"
    write_new(output, data)
    record = {
        "schema": "step3_cdro_opc_m3_reviewer_submission_v1",
        "role": role,
        "agent_id": agent_id,
        "rows": len(rows),
        "review_sha256": sha256_bytes(data),
        "immutable_submission": True,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_new(root / "reviews" / f"{role}_manifest.json", json_bytes(record))
    return {"route": "M3_REVIEW_ACCEPTED", **record}


def analyze_ab(reg: Mapping[str, Any], review_root: Path) -> dict[str, Any]:
    root, manifest = verify_base(reg, review_root)
    p0, sources, primary = load_sources(reg)
    rows: dict[str, list[dict[str, Any]]] = {}
    hidden: dict[str, list[dict[str, Any]]] = {}
    launches = []
    for role in ROLES:
        review_path = root / "reviews" / f"{role}.jsonl"
        rows[role], _ = validated_review(root, role, review_path)
        hidden[role] = p2.parse_jsonl_bytes(
            (root / "private" / f"{role}_hidden_map.jsonl").read_bytes(), f"{role} hidden map"
        )
        launches.append(read_json(root / "launches" / f"{role}.json"))
    if len({item["agent_id"] for item in launches}) != 2:
        raise M3ReviewError("reviewer A/B identities are not independent")
    calibrations = {
        role: p2.score_reviewer_calibration(rows[role], hidden[role], sources, p0, role)
        for role in ROLES
    }
    primary_ids = {row["opaque_candidate_id"] for row in primary}
    canonical = {
        role: p2.canonical_primary_reviews(rows[role], hidden[role], primary_ids, role)
        for role in ROLES
    }
    reliability = p2.compute_primary_reliability(
        primary, canonical["reviewer_a"], canonical["reviewer_b"], p0, manifest["namespace_sha256"]
    )
    disagreements = []
    agreement_labels: dict[str, str] = {}
    for item in primary:
        candidate_id = item["opaque_candidate_id"]
        label_a = qualification.candidate_qualified(canonical["reviewer_a"][candidate_id])
        label_b = qualification.candidate_qualified(canonical["reviewer_b"][candidate_id])
        if label_a == label_b:
            agreement_labels[candidate_id] = label_a
        else:
            disagreements.append(
                {
                    "opaque_candidate_id": candidate_id,
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "reviewer_a_short_reason": canonical["reviewer_a"][candidate_id]["short_reason"],
                    "reviewer_b_short_reason": canonical["reviewer_b"][candidate_id]["short_reason"],
                }
            )
    overall = all(value["overall_pass"] for value in calibrations.values()) and reliability["overall_pass"]
    route = "PASS_M3_AB_REVIEW" if overall else "STOP_M3_REVIEW_UNRELIABLE"
    outcome = {
        "schema": "step3_cdro_opc_m3_ab_review_outcome_v1",
        "route": route,
        "reviewer_calibration": calibrations,
        "primary_reliability": reliability,
        "derived_qualification_agreements": len(agreement_labels),
        "derived_qualification_disagreements": len(disagreements),
        "reviewer_c_required": overall and bool(disagreements),
        "reviewer_c_forbidden": not overall or not disagreements,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_new(root / "outcomes" / "ab_outcome.json", json_bytes(outcome))
    write_new(root / "private" / "ab_primary_reviews.json", json_bytes(canonical))
    write_new(root / "private" / "ab_agreement_labels.json", json_bytes(agreement_labels))
    if overall and disagreements:
        write_new(root / "packets" / "reviewer_c.jsonl", jsonl_bytes(disagreements))
    return outcome


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
    return result


def main() -> None:
    args = parser().parse_args()
    reg = load_registry()
    if args.command == "preflight":
        p0, sources, primary = load_sources(reg)
        result = {
            "route": "PASS_M3_REVIEW_PREFLIGHT",
            "primary_candidates": len(primary),
            "primary_prompts": len({row["opaque_prompt_id"] for row in primary}),
            "calibration_bank": len(sources["calibration_packet"]),
            "optimizer_steps": 0,
            "training_authorized": False,
        }
    elif args.command == "prepare-review":
        result = prepare_review(reg, Path(args.review_root))
    elif args.command == "record-launch":
        result = record_launch(reg, Path(args.review_root), args.role, args.agent_id)
    elif args.command == "submit-review":
        result = submit_review(
            reg, Path(args.review_root), args.role, args.agent_id, Path(args.input)
        )
    else:
        result = analyze_ab(reg, Path(args.review_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
