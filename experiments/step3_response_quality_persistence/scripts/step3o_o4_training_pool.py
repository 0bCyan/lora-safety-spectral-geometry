#!/usr/bin/env python3
"""Freeze and verify the prospective Step3O-O4 prompt pool."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import subprocess
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/step3o_o4_training_pool_registry.json"
PLAN = ROOT / "Step3O_O4_fresh_training_pool_plan.md"
AUTHORING_INSTRUCTIONS = ROOT / "review_rubrics/step3o_o4_target_authoring_instructions.md"
QUALIFICATION_RUBRIC = ROOT / "review_rubrics/step3o_o4_target_qualification_rubric.md"
HANDOFF = ROOT / "review_handoffs/step3o_o4_independent_author_and_reviewer_instructions.md"
TEST_SCRIPT = ROOT / "scripts/test_step3o_o4_training_pool.py"
RANDOMNESS_SCRIPT = ROOT / "scripts/step3o_o4_randomness.py"
RANDOMNESS_TEST = ROOT / "scripts/test_step3o_o4_randomness.py"
QUALIFICATION_SCRIPT = ROOT / "scripts/step3o_o4_qualification.py"
QUALIFICATION_TEST = ROOT / "scripts/test_step3o_o4_qualification.py"
STAGE_SCRIPT = ROOT / "scripts/step3o_o4_stage.py"
STAGE_TEST = ROOT / "scripts/test_step3o_o4_stage.py"
MEASUREMENT_SCRIPT = ROOT / "scripts/step3o_measurement_repair.py"
README = ROOT / "README.md"
EXPERIMENT_TASK = ROOT / "EXPERIMENT_MACHINE_TASK.md"
GIT_ATTRIBUTES = REPO_ROOT / ".gitattributes"
GIT_IGNORE = REPO_ROOT / ".gitignore"
AGENT_DOC = REPO_ROOT / "agent.md"
RANDOMNESS_WORKFLOW = REPO_ROOT / ".github/workflows/step3o-o4-randomness-request.yml"
DRAND_VERIFIER_ROOT = ROOT / "tools/step3o_o4_drand_verifier"
DRAND_VERIFIER_PACKAGE = DRAND_VERIFIER_ROOT / "package.json"
DRAND_VERIFIER_LOCK = DRAND_VERIFIER_ROOT / "package-lock.json"
DRAND_VERIFIER_SCRIPT = DRAND_VERIFIER_ROOT / "verify.mjs"

BASE = ROOT / "results/step3o_o4_training_pool"
PACKETS = BASE / "packets/authoring"
DIAGNOSTICS = BASE / "diagnostics"
REPORTS = BASE / "reports"

AUTHOR_A_PACKET = PACKETS / "step3o_o4_author_a_packet.jsonl"
AUTHOR_B_PACKET = PACKETS / "step3o_o4_author_b_packet.jsonl"
AUTHORING_KEY = PACKETS / "step3o_o4_authoring_hidden_key.csv"
AUTHOR_A_TEMPLATE = PACKETS / "step3o_o4_author_a_template.jsonl"
AUTHOR_B_TEMPLATE = PACKETS / "step3o_o4_author_b_template.jsonl"
PACKET_MANIFEST = PACKETS / "step3o_o4_authoring_packet_manifest.json"
SOURCE_AUDIT = DIAGNOSTICS / "step3o_o4_source_and_exclusion_audit.json"
SELECTION_AUDIT = DIAGNOSTICS / "step3o_o4_prompt_selection_audit.json"
ATTEMPT_LEDGER = REPORTS / "step3o_o4_attempt_001_ledger.json"
ATTEMPT_STOP = REPORTS / "step3o_o4_attempt_001_stop.json"
BEACON_REQUEST = REPORTS / "step3o_o4_drand_request.json"
BEACON_PULSE = REPORTS / "step3o_o4_drand_target_pulse.json"
PROTOCOL_MANIFEST = REPORTS / "step3o_o4_p1_prompt_freeze_manifest.json"
PROTOCOL_REPORT = REPORTS / "Step3O_O4_P1_prompt_freeze_report.md"

RISK_ID_PATTERN = re.compile(r"risk_pkuqa_[0-9]{6}")
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_FIELDS = {
    "sample_id",
    "family",
    "source",
    "prompt",
    "response",
    "is_safe_source",
    "severity_level",
    "harm_category",
    "source_index",
    "content_hash",
}
EVAL_FIELDS = {
    "sample_id",
    "family",
    "source",
    "prompt",
    "reference_response",
    "content_hash",
}
SPLIT_FIELDS = {"split", "sample_id", "content_hash", "hash_key"}
OLD37_FIELDS = {
    "sample_id",
    "category",
    "eligible_response_count",
    "eligible_natural_count",
    "eligible_prospective_count",
    "quality_upper_bound_available",
    "eligible_response_ids",
}
P0_ADDED_PATHS = [
    REGISTRY,
    PLAN,
    AUTHORING_INSTRUCTIONS,
    QUALIFICATION_RUBRIC,
    HANDOFF,
    Path(__file__).resolve(),
    TEST_SCRIPT,
    RANDOMNESS_SCRIPT,
    RANDOMNESS_TEST,
    QUALIFICATION_SCRIPT,
    QUALIFICATION_TEST,
    STAGE_SCRIPT,
    STAGE_TEST,
    RANDOMNESS_WORKFLOW,
    DRAND_VERIFIER_PACKAGE,
    DRAND_VERIFIER_LOCK,
    DRAND_VERIFIER_SCRIPT,
]
P0_COMMIT_PATHS = [
    *P0_ADDED_PATHS,
    GIT_ATTRIBUTES,
    GIT_IGNORE,
    AGENT_DOC,
    README,
    EXPERIMENT_TASK,
]

# Every evidence-bearing stage is locked by a fixed-name GitHub immutable
# release before the next stage can open. Fixed names prevent parallel
# replacement histories for attempt 001; immutable releases prevent tag reuse.
REMOTE_LOCK_MARKERS = {
    "p0": REGISTRY,
    "r0-request": BEACON_REQUEST,
    "r1-pulse": BEACON_PULSE,
    "p1-prompts": PROTOCOL_MANIFEST,
    "p1-stop": ATTEMPT_STOP,
    "p2-authors": BASE / "reports/step3o_o4_p2_author_freeze_manifest.json",
    "p3-candidates": BASE / "reports/step3o_o4_p3_candidate_packet_manifest.json",
    "p3-ab": BASE / "reports/step3o_o4_ab_review_freeze_manifest.json",
    "p3-c-gate": BASE / "reports/step3o_o4_c_gate_manifest.json",
    "p3-final": BASE / "reports/step3o_o4_final_manifest.json",
    "terminal-stop": BASE / "reports/step3o_o4_attempt_001_terminal_stop.json",
}

_REMOTE_TIP_CACHE: dict[tuple[str, str, str], str] = {}
_REQUEST_CHECK_CACHE: dict[str, dict[str, Any]] = {}
_VERIFIED_DRAND_PULSES: set[tuple[int, str, str, str]] = set()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def parse_json_text(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {context}: {exc}") from exc


def parse_tracked_artifact_json_text(text: str, context: str) -> Any:
    """Parse legacy tracked artifacts without treating NaN as prompt text."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: None,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid tracked JSON at {context}: {exc}") from exc


def read_json(path: Path) -> Any:
    return parse_json_text(path.read_text(encoding="utf-8-sig"), str(path))


def parse_frozen_json_bytes(data: bytes, context: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 at {context}") from exc
    if text.startswith("\ufeff") or "\r" in text or not text.endswith("\n"):
        raise ValueError(f"frozen JSON must be UTF-8 without BOM and LF-terminated: {context}")
    return parse_json_text(text, context)


def fetch_public_json(url: str) -> dict[str, Any]:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("public evidence URL must use HTTPS")
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "step3o-o4-protocol/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"failed to retrieve public evidence endpoint: {url}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"public evidence endpoint was not UTF-8: {url}") from exc
    value = parse_json_text(text, url)
    if not isinstance(value, dict):
        raise ValueError(f"public evidence endpoint did not return an object: {url}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return parse_jsonl_bytes(path.read_bytes(), str(path), reject_blank=True)


def parse_jsonl_bytes(
    data: bytes, context: str, *, reject_blank: bool, strict_json: bool = True
) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 at {context}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            if reject_blank:
                raise ValueError(f"blank JSONL row at {context}:{line_number}")
            continue
        parser = parse_json_text if strict_json else parse_tracked_artifact_json_text
        value = parser(line, f"{context}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {context}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL input: {context}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    return parse_csv_bytes(path.read_bytes(), str(path))


def parse_csv_bytes(data: bytes, context: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 at {context}: {exc}") from exc
    reader = csv.DictReader(StringIO(text, newline=""))
    fields = reader.fieldnames
    if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
        raise ValueError(f"invalid or duplicate CSV header at {context}")
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed CSV row at {context}:{line_number}")
        rows.append(dict(row))
    return rows


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def csv_text(rows: Iterable[dict[str, Any]], fields: list[str]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue()


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError(
            f"attempt 001 artifact already exists: {repo_path(path)}"
        ) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_rank(salt: str, *parts: str) -> str:
    return sha256_text("|".join([salt, *parts]))


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def registry() -> dict[str, Any]:
    value = read_json(REGISTRY)
    if not isinstance(value, dict):
        raise ValueError("O4 registry must be a JSON object")
    return value


def _git(args: list[str], *, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=text, check=False
    )
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise ValueError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout


def git_head() -> str:
    return str(_git(["rev-parse", "HEAD"])).strip()


def git_blob_bytes(commit: str, path: Path) -> bytes:
    relative = repo_path(path)
    return bytes(_git(["show", f"{commit}:{relative}"], text=False))


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def commit_changed_paths(commit: str) -> set[str]:
    output = str(
        _git(
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "--root",
                commit,
            ]
        )
    )
    return {line for line in output.splitlines() if line}


def added_commits(path: Path) -> list[str]:
    output = str(
        _git(["log", "--format=%H", "--diff-filter=A", "--", repo_path(path)])
    )
    return [line for line in output.splitlines() if line]


def path_commits_after(commit: str, path: Path) -> list[str]:
    output = str(
        _git(["log", "--format=%H", f"{commit}..HEAD", "--", repo_path(path)])
    )
    return [line for line in output.splitlines() if line]


def require_immutable_since(paths: Iterable[Path], commit: str) -> None:
    require_git_frozen(paths)
    for path in paths:
        relative = repo_path(path)
        require_exact_blob_bytes(
            path.read_bytes(), git_blob_bytes(commit, path), f"{commit}:{relative}"
        )
        later = path_commits_after(commit, path)
        if later:
            raise ValueError(
                f"frozen path was touched after {commit}: {relative}; commits={later}"
            )


def validate_repository_boundary(reg: dict[str, Any]) -> dict[str, Any]:
    value = reg.get("repository_boundary")
    expected_fields = {
        "remote_name",
        "remote_url",
        "branch",
        "upstream_ref",
        "github_repository",
        "github_repository_id",
        "github_token_env",
        "github_token_required",
        "remote_lock_mode",
        "immutable_releases_required",
        "release_attestation_required",
        "release_attestation_verifier",
        "gh_cli_version",
        "stage_lock_tag_prefix",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("repository boundary schema mismatch")
    if (
        value["remote_name"] != "origin"
        or value["remote_url"]
        != "https://github.com/0bCyan/lora-safety-spectral-geometry.git"
        or value["branch"] != "exp/step3-response-quality-persistence"
        or value["upstream_ref"] != "origin/exp/step3-response-quality-persistence"
        or value["github_repository"]
        != "0bCyan/lora-safety-spectral-geometry"
        or value["github_repository_id"] != 1284243466
        or value["github_token_env"] != "STEP3O_O4_GITHUB_TOKEN"
        or value["github_token_required"] is not True
        or value["remote_lock_mode"] != "github_immutable_release_v1"
        or value["immutable_releases_required"] is not True
        or value["release_attestation_required"] is not True
        or value["release_attestation_verifier"]
        != "gh release verify --format json"
        or value["gh_cli_version"] != "2.95.0"
        or value["stage_lock_tag_prefix"] != "step3o-o4-attempt-001"
    ):
        raise ValueError("repository boundary differs from the P0 canonical remote")
    return value


def canonical_remote_tip(reg: dict[str, Any], *, refresh: bool = False) -> str:
    cfg = validate_repository_boundary(reg)
    cache_key = (cfg["remote_name"], cfg["branch"], cfg["remote_url"])
    if not refresh and cache_key in _REMOTE_TIP_CACHE:
        return _REMOTE_TIP_CACHE[cache_key]

    branch = str(_git(["rev-parse", "--abbrev-ref", "HEAD"])).strip()
    remote_url = str(_git(["remote", "get-url", cfg["remote_name"]])).strip()
    upstream = str(
        _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    ).strip()
    if branch != cfg["branch"] or remote_url != cfg["remote_url"] or upstream != cfg["upstream_ref"]:
        raise ValueError("local branch, remote URL, or upstream differs from the P0 registry")

    _git(
        [
            "fetch",
            "--no-tags",
            cfg["remote_name"],
            f"+refs/heads/{cfg['branch']}:refs/remotes/{cfg['upstream_ref']}",
        ]
    )
    fetched_tip = str(_git(["rev-parse", cfg["upstream_ref"]])).strip()
    encoded_branch = urllib.parse.quote(cfg["branch"], safe="")
    branch_state = github_api_json(
        f"repos/{cfg['github_repository']}/branches/{encoded_branch}",
        "GitHub canonical branch API",
        reg=reg,
    )
    commit = branch_state.get("commit")
    api_tip = commit.get("sha") if isinstance(commit, dict) else None
    if (
        branch_state.get("name") != cfg["branch"]
        or not isinstance(api_tip, str)
        or not FULL_SHA_PATTERN.fullmatch(api_tip)
        or api_tip != fetched_tip
    ):
        raise ValueError("fetched canonical branch differs from GitHub branch state")

    repository = github_api_json(
        f"repos/{cfg['github_repository']}",
        "GitHub canonical repository API",
        reg=reg,
    )
    if (
        repository.get("id") != cfg["github_repository_id"]
        or repository.get("full_name") != cfg["github_repository"]
        or repository.get("private") is not True
    ):
        raise ValueError("canonical GitHub repository identity mismatch")

    immutability = github_api_json(
        f"repos/{cfg['github_repository']}/immutable-releases",
        "GitHub immutable-releases API",
        reg=reg,
    )
    if immutability.get("enabled") is not True:
        raise ValueError("canonical repository must enable immutable releases")
    _REMOTE_TIP_CACHE[cache_key] = api_tip
    return api_tip


def remote_lock_stage(commit: str) -> str:
    changed = commit_changed_paths(commit)
    matches = [
        stage
        for stage, marker in REMOTE_LOCK_MARKERS.items()
        if repo_path(marker) in changed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"commit does not identify exactly one registered remote-lock stage: {commit}"
        )
    return matches[0]


def remote_lock_tag(reg: dict[str, Any], stage: str) -> str:
    cfg = validate_repository_boundary(reg)
    if stage not in REMOTE_LOCK_MARKERS:
        raise ValueError(f"unregistered remote-lock stage: {stage}")
    return f"{cfg['stage_lock_tag_prefix']}-{stage}"


def verify_release_attestation(
    reg: dict[str, Any], tag: str, commit: str
) -> dict[str, Any]:
    cfg = validate_repository_boundary(reg)
    token = os.environ.get(cfg["github_token_env"], "").strip()
    if cfg["github_token_required"] and not token:
        raise ValueError(
            f"authenticated release verification requires {cfg['github_token_env']}"
        )
    version = subprocess.run(
        ["gh", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    expected_version = f"gh version {cfg['gh_cli_version']} "
    if version.returncode != 0 or not version.stdout.startswith(expected_version):
        raise ValueError("GitHub CLI version differs from the P0 attestation verifier")
    child_env = os.environ.copy()
    child_env["GH_TOKEN"] = token
    result = subprocess.run(
        [
            "gh",
            "release",
            "verify",
            tag,
            "--repo",
            cfg["github_repository"],
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode != 0:
        error_hash = sha256_bytes(result.stderr)
        raise ValueError(
            f"GitHub release attestation verification failed: stderr_sha256={error_hash}"
        )
    try:
        output_text = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("GitHub release attestation output was not UTF-8") from exc
    output = parse_json_text(output_text, f"gh release verify {tag}")
    if not isinstance(output, (dict, list)) or not output:
        raise ValueError("GitHub release attestation verification returned no evidence")
    return {
        "verifier": cfg["release_attestation_verifier"],
        "gh_cli_version": cfg["gh_cli_version"],
        "repository_id": cfg["github_repository_id"],
        "tag": tag,
        "commit": commit,
        "verification_output_sha256": sha256_bytes(result.stdout),
    }


def validate_immutable_release_lock(
    reg: dict[str, Any], commit: str, stage: str, release: dict[str, Any]
) -> dict[str, Any]:
    cfg = validate_repository_boundary(reg)
    tag = remote_lock_tag(reg, stage)
    if (
        release.get("tag_name") != tag
        or release.get("target_commitish") != commit
        or release.get("immutable") is not True
        or release.get("draft") is not False
        or release.get("prerelease") is not True
        or not isinstance(release.get("id"), int)
        or not isinstance(release.get("published_at"), str)
    ):
        raise ValueError(f"GitHub immutable release lock mismatch for stage {stage}")
    ref = github_api_json(
        f"repos/{cfg['github_repository']}/git/ref/tags/{urllib.parse.quote(tag, safe='')}",
        f"GitHub immutable tag API for {stage}",
        reg=reg,
    )
    target = ref.get("object")
    if (
        ref.get("ref") != f"refs/tags/{tag}"
        or not isinstance(target, dict)
        or target.get("type") != "commit"
        or target.get("sha") != commit
    ):
        raise ValueError(f"GitHub immutable tag target mismatch for stage {stage}")
    attestation = verify_release_attestation(reg, tag, commit)
    return {
        "stage": stage,
        "tag": tag,
        "commit": commit,
        "release_id": release["id"],
        "published_at": release["published_at"],
        "immutable": True,
        "release_attestation_required": cfg["release_attestation_required"],
        "release_attestation": attestation,
    }


def verify_remote_lock(reg: dict[str, Any], commit: str) -> dict[str, Any]:
    cfg = validate_repository_boundary(reg)
    stage = remote_lock_stage(commit)
    tag = remote_lock_tag(reg, stage)
    release = github_api_json(
        f"repos/{cfg['github_repository']}/releases/tags/{urllib.parse.quote(tag, safe='')}",
        f"GitHub immutable release API for {stage}",
        reg=reg,
    )
    return validate_immutable_release_lock(reg, commit, stage, release)


def create_remote_lock() -> dict[str, Any]:
    reg = registry()
    validate_boundaries(reg)
    cfg = validate_repository_boundary(reg)
    head = git_head()
    remote_tip = canonical_remote_tip(reg, refresh=True)
    if head != remote_tip:
        raise ValueError("remote lock requires local HEAD to equal the canonical remote tip")
    stage = remote_lock_stage(head)
    tag = remote_lock_tag(reg, stage)
    release_endpoint = (
        f"repos/{cfg['github_repository']}/releases/tags/"
        f"{urllib.parse.quote(tag, safe='')}"
    )
    existing = github_api_optional_json(
        release_endpoint,
        f"existing GitHub immutable release for {stage}",
        reg=reg,
    )
    if existing is not None:
        recovered = validate_immutable_release_lock(reg, head, stage, existing)
        recovered["recovered_existing_release"] = True
        return recovered
    payload = {
        "tag_name": tag,
        "target_commitish": head,
        "name": f"Step3O-O4 attempt 001 immutable lock: {stage}",
        "body": (
            "Prospective Step3O-O4 stage lock.\n\n"
            f"attempt: step3o_o4_attempt_001\n"
            f"stage: {stage}\n"
            f"commit: {head}\n"
            "training_authorized: false\n"
        ),
        "draft": False,
        "prerelease": True,
        "make_latest": "false",
    }
    release = github_api_mutation_json(
        "POST",
        f"repos/{cfg['github_repository']}/releases",
        f"GitHub immutable release creation for {stage}",
        payload,
        reg=reg,
    )
    result = validate_immutable_release_lock(reg, head, stage, release)
    result["recovered_existing_release"] = False
    return result


def require_remote_commits(reg: dict[str, Any], commits: Iterable[str]) -> str:
    remote_tip = canonical_remote_tip(reg)
    for commit in commits:
        if not isinstance(commit, str) or not FULL_SHA_PATTERN.fullmatch(commit):
            raise ValueError("remote ancestry check received an invalid commit")
        if not git_is_ancestor(commit, remote_tip):
            raise ValueError(f"commit is absent from canonical remote branch: {commit}")
        verify_remote_lock(reg, commit)
    return remote_tip


def resolve_protocol_commit(reg: dict[str, Any], *, require_pushed: bool) -> str:
    commits = added_commits(REGISTRY)
    if len(commits) != 1:
        raise ValueError(
            "attempt 001 requires exactly one immutable commit that first adds the O4 registry"
        )
    protocol_commit = commits[0]
    for path in P0_ADDED_PATHS:
        if added_commits(path) != [protocol_commit]:
            raise ValueError(
                f"P0 file was not first added by protocol commit {protocol_commit}: {repo_path(path)}"
            )
    parent_line = str(_git(["rev-list", "--parents", "-n", "1", protocol_commit])).split()
    if len(parent_line) != 2 or parent_line[1] != reg["entry_commit"]:
        raise ValueError("P0 protocol commit must have the registered entry commit as its sole parent")
    expected_paths = {repo_path(path) for path in P0_COMMIT_PATHS}
    observed_paths = commit_changed_paths(protocol_commit)
    if observed_paths != expected_paths:
        raise ValueError(
            "P0 protocol commit changed an unexpected path set: "
            f"missing={sorted(expected_paths-observed_paths)}, "
            f"extra={sorted(observed_paths-expected_paths)}"
        )
    if not git_is_ancestor(protocol_commit, "HEAD"):
        raise ValueError("P0 protocol commit is not an ancestor of HEAD")
    require_immutable_since(P0_COMMIT_PATHS, protocol_commit)
    if require_pushed:
        require_remote_commits(reg, [protocol_commit])
    return protocol_commit


def require_git_frozen(paths: Iterable[Path], *, commit: str = "HEAD") -> None:
    for path in paths:
        relative = repo_path(path)
        if not path.is_file():
            raise ValueError(f"required committed freeze is absent: {relative}")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode != 0:
            raise ValueError(f"required committed freeze is untracked: {relative}")
        committed = git_blob_bytes(commit, path)
        require_exact_blob_bytes(path.read_bytes(), committed, f"{commit}:{relative}")


def require_exact_blob_bytes(working: bytes, committed: bytes, context: str) -> None:
    if working != committed:
        raise ValueError(f"working bytes differ from committed blob (including EOL): {context}")


def require_protocol_inputs_frozen(reg: dict[str, Any], protocol_commit: str) -> None:
    paths = protocol_inputs(reg)
    require_immutable_since(paths, protocol_commit)


def validate_boundaries(reg: dict[str, Any]) -> None:
    repository_boundary = validate_repository_boundary(reg)
    attempt = reg["attempt"]
    if (
        attempt.get("attempt_id") != "step3o_o4_attempt_001"
        or attempt.get("attempt_number") != 1
        or attempt.get("prior_o4_attempts") != 0
        or attempt.get("overwrite_or_silent_restart_allowed") is not False
        or attempt.get("failure_is_terminal_for_attempt") is not True
        or attempt.get("ledger") != repo_path(ATTEMPT_LEDGER)
        or attempt.get("stop_event") != repo_path(ATTEMPT_STOP)
    ):
        raise ValueError("attempt 001 registry is not fail-closed")
    boundary = reg["boundaries"]
    false_fields = [
        "source_harmful_responses_released",
        "old37_prompts_released",
        "method_arm_outcomes_opened_to_authors_or_reviewers",
        "heldout_content_opened",
        "training_authorized",
        "pcsr_r2_authorized",
    ]
    bad = [field for field in false_fields if boundary.get(field) is not False]
    if bad or type(boundary.get("optimizer_steps")) is not int or boundary["optimizer_steps"] != 0:
        raise ValueError(f"O4 boundaries are open: {bad}")
    beacon = reg["randomness_beacon"]
    if (
        beacon.get("github_repository")
        != repository_boundary["github_repository"]
        or
        beacon.get("request_ci_app_slug") != "github-actions"
        or beacon.get("request_ci_event") != "push"
        or beacon.get("request_ci_run_attempt") != 1
        or beacon.get("target_round_offset") != 60
        or beacon.get("request_commit_must_precede_target_round") is not True
        or beacon.get("successful_request_ci_required_before_pulse") is not True
        or beacon.get("live_chain_info_and_round_refetch_required_before_selection")
        is not True
        or beacon.get("cryptographic_signature_verification_required") is not True
        or beacon.get("verifier_package") != "drand-client@1.4.2"
        or beacon.get("verifier_script") != repo_path(DRAND_VERIFIER_SCRIPT)
    ):
        raise ValueError("O4 public-randomness evidence boundary changed")


def validate_protocol_commit_format(protocol_commit: str) -> None:
    if not isinstance(protocol_commit, str) or not FULL_SHA_PATTERN.fullmatch(protocol_commit):
        raise ValueError("protocol commit must be a full lowercase 40-character SHA")
    if not git_is_ancestor(protocol_commit, protocol_commit):
        raise ValueError(f"unknown protocol commit: {protocol_commit}")


def derive_salt(
    reg: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
    purpose: str,
) -> str:
    randomness = reg["selection"]["randomness"]
    if purpose not in randomness["purposes"]:
        raise ValueError(f"unregistered O4 salt purpose: {purpose}")
    beacon_round = randomness_context.get("beacon_round")
    beacon_randomness = randomness_context.get("beacon_randomness")
    if type(beacon_round) is not int or beacon_round <= 0:
        raise ValueError("invalid drand beacon round in randomness context")
    if (
        not isinstance(beacon_randomness, str)
        or not re.fullmatch(r"[0-9a-f]{64}", beacon_randomness)
    ):
        raise ValueError("invalid drand beacon randomness in randomness context")
    return sha256_text(
        f"step3o_o4_v2|{protocol_commit}|{beacon_round}|{beacon_randomness}|{purpose}"
    )


def drand_round_time(reg: dict[str, Any], round_number: int) -> int:
    cfg = reg["randomness_beacon"]
    if type(round_number) is not int or round_number <= 0:
        raise ValueError("drand round must be a positive integer")
    return int(cfg["genesis_time_unix"]) + (round_number - 1) * int(
        cfg["period_seconds"]
    )


def validate_drand_info(value: dict[str, Any], reg: dict[str, Any]) -> None:
    cfg = reg["randomness_beacon"]
    required = {
        "public_key": cfg["public_key"],
        "period": int(cfg["period_seconds"]),
        "genesis_time": int(cfg["genesis_time_unix"]),
        "hash": cfg["chain_hash"],
        "schemeID": cfg["scheme_id"],
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("drand chain info differs from the P0 registry")


def validate_drand_pulse(
    value: dict[str, Any], *, expected_round: int | None = None
) -> dict[str, Any]:
    required_fields = {"round", "randomness", "signature", "previous_signature"}
    if not isinstance(value, dict) or set(value) != required_fields:
        raise ValueError("drand pulse schema mismatch")
    round_number = value["round"]
    if type(round_number) is not int or round_number <= 0:
        raise ValueError("invalid drand pulse round")
    if expected_round is not None and round_number != expected_round:
        raise ValueError("drand pulse round differs from the requested future round")
    for field, width in (("randomness", 64), ("signature", 192), ("previous_signature", 192)):
        item = value[field]
        if not isinstance(item, str) or not re.fullmatch(rf"[0-9a-f]{{{width}}}", item):
            raise ValueError(f"invalid drand pulse {field}")
    derived = sha256_bytes(bytes.fromhex(value["signature"]))
    if derived != value["randomness"]:
        raise ValueError("drand randomness is not SHA-256(signature)")
    return value


def verify_drand_pulse_cryptographically(
    reg: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    pulse = validate_drand_pulse(value)
    cfg = reg["randomness_beacon"]
    cache_key = (
        pulse["round"],
        pulse["randomness"],
        pulse["signature"],
        pulse["previous_signature"],
    )
    if cache_key in _VERIFIED_DRAND_PULSES:
        return pulse
    if (
        cfg.get("cryptographic_signature_verification_required") is not True
        or cfg.get("verifier_package") != "drand-client@1.4.2"
        or cfg.get("verifier_script") != repo_path(DRAND_VERIFIER_SCRIPT)
    ):
        raise ValueError("drand cryptographic verifier boundary changed")
    for path in (
        DRAND_VERIFIER_PACKAGE,
        DRAND_VERIFIER_LOCK,
        DRAND_VERIFIER_SCRIPT,
    ):
        if not path.is_file():
            raise ValueError(f"drand verifier file is absent: {repo_path(path)}")

    verifier_input = {
        "base_url": cfg["verifier_base_url"],
        "chain_hash": cfg["chain_hash"],
        "expected_pulse": pulse,
        "public_key": cfg["public_key"],
    }
    with TemporaryDirectory(prefix="step3o_o4_drand_") as directory:
        input_path = Path(directory) / "pulse.json"
        input_path.write_text(json_text(verifier_input), encoding="utf-8", newline="\n")
        try:
            result = subprocess.run(
                ["node", str(DRAND_VERIFIER_SCRIPT), str(input_path)],
                cwd=DRAND_VERIFIER_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("failed to execute pinned drand cryptographic verifier") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ValueError(f"pinned drand cryptographic verifier rejected pulse: {stderr}")
    verified = parse_json_text(result.stdout, "drand verifier stdout")
    if verified != pulse:
        raise ValueError("drand verifier returned a different pulse")
    _VERIFIED_DRAND_PULSES.add(cache_key)
    return pulse


def validate_randomness_request(
    value: dict[str, Any], reg: dict[str, Any], protocol_commit: str
) -> dict[str, Any]:
    expected_fields = {
        "name",
        "status",
        "provider",
        "protocol_commit",
        "chain_info",
        "observed_pulse",
        "observed_round_time_unix",
        "target_round",
        "target_round_time_unix",
        "target_round_offset",
        "selection_started",
        "optimizer_steps",
        "training_authorized",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("drand request artifact schema mismatch")
    cfg = reg["randomness_beacon"]
    if (
        value["name"] != "step3o_o4_drand_request_v1"
        or value["status"] != "FROZEN_REQUEST_BEFORE_TARGET_PULSE"
        or value["provider"] != cfg["provider"]
        or value["protocol_commit"] != protocol_commit
        or value["selection_started"] is not False
        or value["optimizer_steps"] != 0
        or value["training_authorized"] is not False
    ):
        raise ValueError("drand request artifact boundary mismatch")
    if set(value["chain_info"]) != {
        "public_key",
        "period",
        "genesis_time",
        "hash",
        "schemeID",
    }:
        raise ValueError("drand request chain-info schema mismatch")
    validate_drand_info(value["chain_info"], reg)
    observed = validate_drand_pulse(value["observed_pulse"])
    offset = int(cfg["target_round_offset"])
    if type(value["target_round_offset"]) is not int or value["target_round_offset"] != offset:
        raise ValueError("drand target offset changed")
    if value["target_round"] != observed["round"] + offset:
        raise ValueError("drand target round is not the registered future offset")
    if value["observed_round_time_unix"] != drand_round_time(reg, observed["round"]):
        raise ValueError("drand observed round time mismatch")
    if value["target_round_time_unix"] != drand_round_time(reg, value["target_round"]):
        raise ValueError("drand target round time mismatch")
    return value


def validate_randomness_pulse_artifact(
    value: dict[str, Any],
    reg: dict[str, Any],
    protocol_commit: str,
    request_commit: str,
    request_sha256: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "name",
        "status",
        "provider",
        "protocol_commit",
        "request_commit",
        "request_sha256",
        "target_round",
        "target_round_time_unix",
        "target_pulse",
        "request_ci",
        "selection_started",
        "optimizer_steps",
        "training_authorized",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("drand target-pulse artifact schema mismatch")
    cfg = reg["randomness_beacon"]
    if (
        value["name"] != "step3o_o4_drand_target_pulse_v1"
        or value["status"] != "FROZEN_PUBLIC_RANDOMNESS_BEFORE_SELECTION"
        or value["provider"] != cfg["provider"]
        or value["protocol_commit"] != protocol_commit
        or value["request_commit"] != request_commit
        or value["request_sha256"] != request_sha256
        or value["target_round"] != request["target_round"]
        or value["target_round_time_unix"] != request["target_round_time_unix"]
        or value["selection_started"] is not False
        or value["optimizer_steps"] != 0
        or value["training_authorized"] is not False
    ):
        raise ValueError("drand target-pulse artifact boundary mismatch")
    validate_drand_pulse(value["target_pulse"], expected_round=request["target_round"])
    validate_request_ci_evidence(
        value["request_ci"],
        reg,
        request_commit,
        target_round_time_unix=request["target_round_time_unix"],
    )
    return value


def github_timestamp_unix(value: Any, context: str) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{context} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{context} must be UTC")
    return parsed.timestamp()


def validate_request_ci_evidence(
    value: dict[str, Any],
    reg: dict[str, Any],
    request_commit: str,
    *,
    target_round_time_unix: int | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "id",
        "name",
        "head_sha",
        "status",
        "conclusion",
        "html_url",
        "details_url",
        "started_at",
        "completed_at",
        "app_slug",
        "check_suite_id",
        "workflow_run_id",
        "workflow_run_attempt",
        "workflow_path",
        "event",
        "job_id",
    }
    cfg = reg["randomness_beacon"]
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("GitHub Actions request-check evidence schema mismatch")
    if (
        type(value["id"]) is not int
        or value["id"] <= 0
        or value["name"] != cfg["request_ci_check_name"]
        or value["head_sha"] != request_commit
        or value["status"] != "completed"
        or value["conclusion"] != "success"
        or value["app_slug"] != cfg["request_ci_app_slug"]
        or type(value["check_suite_id"]) is not int
        or value["check_suite_id"] <= 0
        or type(value["workflow_run_id"]) is not int
        or value["workflow_run_id"] <= 0
        or type(value["workflow_run_attempt"]) is not int
        or value["workflow_run_attempt"] != cfg["request_ci_run_attempt"]
        or value["workflow_path"] != cfg["request_ci_workflow"]
        or value["event"] != cfg["request_ci_event"]
        or type(value["job_id"]) is not int
        or value["job_id"] <= 0
        or not isinstance(value["html_url"], str)
        or not isinstance(value["details_url"], str)
        or not isinstance(value["started_at"], str)
        or not value["started_at"]
        or not isinstance(value["completed_at"], str)
        or not value["completed_at"]
    ):
        raise ValueError("GitHub Actions request check did not pass exactly")
    repository = cfg["github_repository"]
    expected_job_url = (
        f"https://github.com/{repository}/actions/runs/"
        f"{value['workflow_run_id']}/job/{value['job_id']}"
    )
    expected_check_url = f"https://github.com/{repository}/runs/{value['id']}"
    valid_html_url = (
        value["html_url"] == expected_job_url
        or value["html_url"] == expected_check_url
        or value["html_url"].startswith(expected_check_url + "?")
    )
    if not valid_html_url or value["details_url"] != expected_job_url:
        raise ValueError("GitHub Actions request check URL identity mismatch")
    started = github_timestamp_unix(value["started_at"], "check started_at")
    completed = github_timestamp_unix(value["completed_at"], "check completed_at")
    if completed < started:
        raise ValueError("GitHub Actions request check completed before it started")
    if target_round_time_unix is not None:
        if type(target_round_time_unix) is not int or target_round_time_unix <= 0:
            raise ValueError("invalid drand target-round publication time")
        if completed >= target_round_time_unix:
            raise ValueError("GitHub Actions request check did not complete before target round")
    return value


def github_api_json(
    endpoint: str, context: str, *, reg: dict[str, Any] | None = None
) -> dict[str, Any]:
    decoded_endpoint = urllib.parse.unquote(endpoint) if isinstance(endpoint, str) else ""
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint.startswith("/")
        or ".." in decoded_endpoint
        or not re.fullmatch(r"(?:[A-Za-z0-9_./-]|%[0-9A-Fa-f]{2})+", endpoint)
    ):
        raise ValueError("invalid relative GitHub API endpoint")
    active_registry = registry() if reg is None else reg
    boundary = validate_repository_boundary(active_registry)
    token = os.environ.get(boundary["github_token_env"], "").strip()
    if boundary["github_token_required"] and not token:
        raise ValueError(
            f"authenticated GitHub API requires {boundary['github_token_env']}"
        )
    url = f"https://api.github.com/{endpoint}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "step3o-o4-protocol/1",
        "Authorization": f"Bearer {token}",
    }
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"failed to query authenticated {context}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"authenticated {context} was not UTF-8") from exc
    payload = parse_json_text(text, context)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} did not return an object")
    return payload


def github_api_optional_json(
    endpoint: str, context: str, *, reg: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    decoded_endpoint = urllib.parse.unquote(endpoint) if isinstance(endpoint, str) else ""
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint.startswith("/")
        or ".." in decoded_endpoint
        or not re.fullmatch(r"(?:[A-Za-z0-9_./-]|%[0-9A-Fa-f]{2})+", endpoint)
    ):
        raise ValueError("invalid relative GitHub API endpoint")
    active_registry = registry() if reg is None else reg
    boundary = validate_repository_boundary(active_registry)
    token = os.environ.get(boundary["github_token_env"], "").strip()
    if boundary["github_token_required"] and not token:
        raise ValueError(
            f"authenticated GitHub API requires {boundary['github_token_env']}"
        )
    request = urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "step3o-o4-protocol/1",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ValueError(f"failed to query authenticated {context}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"failed to query authenticated {context}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"authenticated {context} was not UTF-8") from exc
    payload = parse_json_text(text, context)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} did not return an object")
    return payload


def github_api_mutation_json(
    method: str,
    endpoint: str,
    context: str,
    payload: dict[str, Any],
    *,
    reg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if method not in {"POST", "PUT"}:
        raise ValueError("GitHub mutation method must be POST or PUT")
    decoded_endpoint = urllib.parse.unquote(endpoint) if isinstance(endpoint, str) else ""
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint.startswith("/")
        or ".." in decoded_endpoint
        or not re.fullmatch(r"(?:[A-Za-z0-9_./-]|%[0-9A-Fa-f]{2})+", endpoint)
    ):
        raise ValueError("invalid relative GitHub API endpoint")
    if not isinstance(payload, dict):
        raise ValueError("GitHub mutation payload must be an object")
    active_registry = registry() if reg is None else reg
    boundary = validate_repository_boundary(active_registry)
    token = os.environ.get(boundary["github_token_env"], "").strip()
    if boundary["github_token_required"] and not token:
        raise ValueError(
            f"authenticated GitHub API requires {boundary['github_token_env']}"
        )
    request = urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        data=json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "step3o-o4-protocol/1",
            "Authorization": f"Bearer {token}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"failed to mutate authenticated {context}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"authenticated {context} was not UTF-8") from exc
    result = parse_json_text(text, context)
    if not isinstance(result, dict):
        raise ValueError(f"{context} did not return an object")
    return result


def github_request_check_evidence(
    reg: dict[str, Any], request_commit: str
) -> dict[str, Any]:
    cached = _REQUEST_CHECK_CACHE.get(request_commit)
    if cached is not None:
        return dict(cached)
    repository = reg["randomness_beacon"]["github_repository"]
    payload = github_api_json(
        f"repos/{repository}/commits/{request_commit}/check-runs",
        "GitHub check-runs API",
        reg=reg,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("check_runs"), list):
        raise ValueError("GitHub check-runs API schema mismatch")
    check_name = reg["randomness_beacon"]["request_ci_check_name"]
    matching = [
        item
        for item in payload["check_runs"]
        if isinstance(item, dict)
        and item.get("name") == check_name
        and item.get("head_sha") == request_commit
        and isinstance(item.get("app"), dict)
        and item["app"].get("slug")
        == reg["randomness_beacon"]["request_ci_app_slug"]
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected exactly one non-rerun request check, found {len(matching)}"
        )
    item = matching[0]
    details_url = item.get("details_url")
    match = (
        re.fullmatch(
            rf"https://github\.com/{re.escape(repository)}/actions/runs/(\d+)/job/(\d+)",
            details_url,
        )
        if isinstance(details_url, str)
        else None
    )
    if match is None:
        raise ValueError("GitHub Actions request check has an unexpected job URL")
    run_id = int(match.group(1))
    job_id = int(match.group(2))
    run = github_api_json(
        f"repos/{repository}/actions/runs/{run_id}",
        "GitHub workflow-run API",
        reg=reg,
    )
    evidence = {
        "id": item.get("id"),
        "name": item.get("name"),
        "head_sha": item.get("head_sha"),
        "status": item.get("status"),
        "conclusion": item.get("conclusion"),
        "html_url": item.get("html_url"),
        "details_url": details_url,
        "started_at": item.get("started_at"),
        "completed_at": item.get("completed_at"),
        "app_slug": item["app"].get("slug"),
        "check_suite_id": item.get("check_suite", {}).get("id")
        if isinstance(item.get("check_suite"), dict)
        else None,
        "workflow_run_id": run.get("id"),
        "workflow_run_attempt": run.get("run_attempt"),
        "workflow_path": run.get("path"),
        "event": run.get("event"),
        "job_id": job_id,
    }
    validated = validate_request_ci_evidence(evidence, reg, request_commit)
    _REQUEST_CHECK_CACHE[request_commit] = dict(validated)
    return dict(validated)


def validate_live_drand_chain(
    reg: dict[str, Any], request: dict[str, Any], pulse: dict[str, Any]
) -> None:
    cfg = reg["randomness_beacon"]
    info = fetch_public_json(cfg["info_url"])
    validate_drand_info(info, reg)
    for label, expected in (
        ("observed", request["observed_pulse"]),
        ("target", pulse["target_pulse"]),
    ):
        try:
            verify_drand_pulse_cryptographically(reg, expected)
        except ValueError as exc:
            raise ValueError(
                f"frozen {label} drand pulse failed live cryptographic verification"
            ) from exc


def verify_randomness_chain(
    reg: dict[str, Any], protocol_commit: str, *, require_pushed: bool
) -> dict[str, Any]:
    cfg = reg["randomness_beacon"]
    if cfg["request_artifact"] != repo_path(BEACON_REQUEST) or cfg[
        "pulse_artifact"
    ] != repo_path(BEACON_PULSE):
        raise ValueError("drand artifact paths differ from the registry")
    request_commit = verify_exact_creation_commit(
        [BEACON_REQUEST],
        BEACON_REQUEST,
        protocol_commit,
        reg=reg,
        require_pushed=require_pushed,
    )
    request_bytes = git_blob_bytes(request_commit, BEACON_REQUEST)
    request = validate_randomness_request(
        parse_frozen_json_bytes(request_bytes, "R0 request commit blob"),
        reg,
        protocol_commit,
    )
    request_sha256 = sha256_bytes(request_bytes)
    pulse_commit = verify_exact_creation_commit(
        [BEACON_PULSE],
        BEACON_PULSE,
        request_commit,
        reg=reg,
        require_pushed=require_pushed,
    )
    pulse_bytes = git_blob_bytes(pulse_commit, BEACON_PULSE)
    pulse = validate_randomness_pulse_artifact(
        parse_frozen_json_bytes(pulse_bytes, "R1 pulse commit blob"),
        reg,
        protocol_commit,
        request_commit,
        request_sha256,
        request,
    )
    if require_pushed:
        live_ci = validate_request_ci_evidence(
            github_request_check_evidence(reg, request_commit),
            reg,
            request_commit,
            target_round_time_unix=request["target_round_time_unix"],
        )
        if live_ci != pulse["request_ci"]:
            raise ValueError("frozen request-check evidence differs from GitHub")
        validate_live_drand_chain(reg, request, pulse)
    target = pulse["target_pulse"]
    return {
        "protocol_commit": protocol_commit,
        "request_commit": request_commit,
        "pulse_commit": pulse_commit,
        "selection_parent_commit": pulse_commit,
        "beacon_round": target["round"],
        "beacon_randomness": target["randomness"],
        "request_sha256": request_sha256,
        "pulse_sha256": sha256_bytes(pulse_bytes),
    }


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def unicode_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", normalize_text(value), flags=re.UNICODE))


def token_shingles(value: str, maximum_width: int) -> frozenset[tuple[str, ...]]:
    tokens = unicode_tokens(value)
    features: set[tuple[str, ...]] = set()
    for width in range(1, min(maximum_width, len(tokens)) + 1):
        for index in range(len(tokens) - width + 1):
            features.add((str(width), *tokens[index : index + width]))
    return frozenset(features)


def lexical_similarity(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
) -> tuple[float, float, float]:
    if not left and not right:
        return 1.0, 1.0, 1.0
    if not left or not right:
        return 0.0, 0.0, 0.0
    intersection = len(left & right)
    jaccard = intersection / len(left | right)
    overlap = intersection / min(len(left), len(right))
    return jaccard, overlap, max(jaccard, overlap)


def sequence_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None, normalize_text(left), normalize_text(right), autojunk=False
    ).ratio()


def check_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"hash mismatch for {repo_path(path)}: expected {expected}, got {actual}"
        )


def validate_entry_route(reg: dict[str, Any]) -> dict[str, Any]:
    route_path = resolve_repo_path(reg["entry_route"])
    check_hash(route_path, reg["entry_route_sha256"])
    route = read_json(route_path)
    required = {
        "status": "PASS",
        "overall_pass": True,
        "next_action": "authorize_fresh_training_pool_plan_only",
        "pcsr_training_authorized": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }
    bad = {
        key: (route.get(key), expected)
        for key, expected in required.items()
        if route.get(key) != expected
    }
    if bad:
        raise ValueError(f"O3 route does not authorize O4 planning: {bad}")
    return route


def tracked_risk_ids(reg: dict[str, Any], commit: str) -> set[str]:
    exclusion = reg["exclusion_references"]
    args = ["grep", "-h", "-o", "-E", RISK_ID_PATTERN.pattern, commit, "--"]
    args.extend(exclusion["tracked_scan_roots"])
    args.extend(
        [
            f":(exclude){repo_path(REGISTRY)}",
            f":(exclude){exclusion['tracked_scan_excluded_prefix']}**",
        ]
    )
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode not in (0, 1):
        raise ValueError(f"git exposure scan failed: {result.stderr.strip()}")
    found = set(RISK_ID_PATTERN.findall(result.stdout))
    expected = set(exclusion["expected_tracked_risk_ids"])
    if found != expected:
        raise ValueError(
            f"tracked risk-ID exposure set changed: missing={sorted(expected-found)}, extra={sorted(found-expected)}"
        )
    return found


def git_structured_blobs(
    reg: dict[str, Any], commit: str
) -> list[tuple[str, str]]:
    exclusion = reg["exclusion_references"]
    output = bytes(
        _git(
            ["ls-tree", "-r", "-z", commit, "--", *exclusion["tracked_scan_roots"]],
            text=False,
        )
    )
    extensions = set(exclusion["tracked_structured_extensions"])
    excluded_prefix = exclusion["tracked_scan_excluded_prefix"]
    blobs: list[tuple[str, str]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        del mode
        path = raw_path.decode("utf-8", "surrogateescape")
        if (
            object_type == "blob"
            and Path(path).suffix.casefold() in extensions
            and not path.startswith(excluded_prefix)
        ):
            blobs.append((object_id, path))
    return sorted(blobs, key=lambda item: item[1])


def _read_exact(handle: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ValueError("git cat-file ended before the declared blob size")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_git_blobs(blobs: Iterable[tuple[str, str]]) -> Iterator[tuple[str, bytes]]:
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for object_id, path in blobs:
            process.stdin.write((object_id + "\n").encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii", "strict").strip().split()
            if len(header) != 3 or header[0] != object_id or header[1] != "blob":
                raise ValueError(f"unexpected git cat-file header for {path}: {header}")
            data = _read_exact(process.stdout, int(header[2]))
            if process.stdout.read(1) != b"\n":
                raise ValueError(f"missing git cat-file terminator for {path}")
            yield path, data
    finally:
        process.stdin.close()
        return_code = process.wait()
        process.stdout.close()
        if return_code != 0:
            assert process.stderr is not None
            error = process.stderr.read().decode("utf-8", "replace").strip()
            process.stderr.close()
            raise ValueError(f"git cat-file failed: {error}")
        assert process.stderr is not None
        process.stderr.close()


def _walk_prompt_fields(
    value: Any, field_names: set[str], location: str = "$"
) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key.casefold() in field_names and isinstance(child, str) and child.strip():
                yield child_location, child
            yield from _walk_prompt_fields(child, field_names, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_prompt_fields(child, field_names, f"{location}[{index}]")


def structured_prompt_values(
    path: str, data: bytes, field_names: set[str]
) -> Iterator[tuple[str, str]]:
    suffix = Path(path).suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        if not data.strip():
            return
        if suffix == ".json":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 in tracked artifact {path}: {exc}") from exc
            value = parse_tracked_artifact_json_text(text, f"{path}@git")
            yield from _walk_prompt_fields(value, field_names)
        else:
            rows = parse_jsonl_bytes(
                data,
                f"{path}@git",
                reject_blank=False,
                strict_json=False,
            )
            for index, row in enumerate(rows):
                yield from _walk_prompt_fields(row, field_names, f"$[{index}]")
    elif suffix == ".csv":
        rows = parse_csv_bytes(data, f"{path}@git")
        if not rows:
            return
        matching_fields = [field for field in rows[0] if field.casefold() in field_names]
        for index, row in enumerate(rows, 2):
            for field in matching_fields:
                value = row[field]
                if value.strip():
                    yield f"row={index}.{field}", value


def tracked_prompt_fingerprint_exposures(
    source_rows: list[dict[str, Any]], reg: dict[str, Any], commit: str
) -> dict[str, Any]:
    cfg = reg["exclusion_references"]
    field_names = {value.casefold() for value in cfg["tracked_prompt_field_names"]}
    normalized_to_ids: dict[str, list[str]] = {}
    for row in source_rows:
        digest = sha256_text(normalize_text(row["prompt"]))
        normalized_to_ids.setdefault(digest, []).append(row["sample_id"])

    blobs = git_structured_blobs(reg, commit)
    occurrences: dict[str, list[dict[str, str]]] = {}
    parsed_files = 0
    for path, data in read_git_blobs(blobs):
        found_in_file = False
        for field_path, value in structured_prompt_values(path, data, field_names):
            digest = sha256_text(normalize_text(value))
            for sample_id in normalized_to_ids.get(digest, []):
                occurrence = {
                    "path": path,
                    "field": field_path,
                    "normalized_prompt_sha256": digest,
                }
                if occurrence not in occurrences.setdefault(sample_id, []):
                    occurrences[sample_id].append(occurrence)
                found_in_file = True
        if found_in_file:
            parsed_files += 1
    return {
        "structured_files_scanned": len(blobs),
        "files_with_source_prompt_match": parsed_files,
        "source_ids": sorted(occurrences),
        "occurrences": [
            {"sample_id": sample_id, "matches": sorted(occurrences[sample_id], key=lambda x: (x["path"], x["field"]))}
            for sample_id in sorted(occurrences)
        ],
    }


def load_eval_rows(
    path: Path, *, expected_rows: int, expected_family: str, expected_source: str
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {repo_path(path)}, got {len(rows)}")
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for row in rows:
        if set(row) != EVAL_FIELDS:
            raise ValueError(f"evaluation schema mismatch in {repo_path(path)}")
        sample_id = row["sample_id"]
        prompt = row["prompt"]
        reference_response = row["reference_response"]
        if (
            not isinstance(sample_id, str)
            or not re.fullmatch(rf"eval_{re.escape(expected_family)}_[0-9]{{6}}", sample_id)
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(reference_response, str)
            or not reference_response.strip()
        ):
            raise ValueError(f"invalid evaluation text fields in {repo_path(path)}")
        if row["family"] != expected_family or row["source"] != expected_source:
            raise ValueError(f"evaluation provenance mismatch: {sample_id}")
        if sample_id in ids:
            raise ValueError(f"duplicate evaluation ID: {sample_id}")
        prompt_hash = sha256_text(prompt)
        if prompt_hash in prompt_hashes:
            raise ValueError(f"duplicate evaluation prompt in {repo_path(path)}")
        if row["content_hash"] != prompt_hash:
            raise ValueError(f"evaluation content hash mismatch: {sample_id}")
        ids.add(sample_id)
        prompt_hashes.add(prompt_hash)
    return rows


def validate_source_row(
    row: dict[str, Any], source_cfg: dict[str, Any], expected_categories: set[str]
) -> None:
    if set(row) != SOURCE_FIELDS:
        raise ValueError("source row has an unexpected top-level schema")
    sample_id = row["sample_id"]
    if not isinstance(sample_id, str) or not RISK_ID_PATTERN.fullmatch(sample_id):
        raise ValueError(f"invalid source sample ID: {sample_id}")
    if row["family"] != source_cfg["expected_family"]:
        raise ValueError(f"wrong source family: {sample_id}")
    if row["source"] != source_cfg["expected_source"]:
        raise ValueError(f"wrong source name: {sample_id}")
    if row["is_safe_source"] is not source_cfg["expected_is_safe_source"]:
        raise ValueError(f"wrong source safety flag: {sample_id}")
    prompt = row["prompt"]
    response = row["response"]
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or not isinstance(response, str)
        or not response.strip()
    ):
        raise ValueError(f"missing source text: {sample_id}")
    if row["content_hash"] != sha256_text(prompt + "\n" + response):
        raise ValueError(f"source content hash mismatch: {sample_id}")
    if type(row["source_index"]) is not int or row["source_index"] <= 0:
        raise ValueError(f"invalid source_index: {sample_id}")
    if type(row["severity_level"]) is not int or row["severity_level"] not in {1, 2, 3}:
        raise ValueError(f"invalid severity_level: {sample_id}")
    categories = row["harm_category"]
    if not isinstance(categories, dict) or set(categories) != expected_categories:
        raise ValueError(f"source category schema mismatch: {sample_id}")
    if any(type(value) is not bool for value in categories.values()):
        raise ValueError(f"non-boolean source category: {sample_id}")


def load_source_rows(reg: dict[str, Any]) -> list[dict[str, Any]]:
    source_cfg = reg["source"]
    path = resolve_repo_path(source_cfg["train_risk"])
    check_hash(path, source_cfg["train_risk_sha256"])
    rows = read_jsonl(path)
    if len(rows) != int(source_cfg["expected_rows"]):
        raise ValueError(f"expected {source_cfg['expected_rows']} source rows, got {len(rows)}")

    expected_categories = set(reg["selection"]["category_severity_slots"])
    ids: set[str] = set()
    source_indexes: set[int] = set()
    prompt_hashes: set[str] = set()
    normalized_prompt_hashes: set[str] = set()
    for row in rows:
        validate_source_row(row, source_cfg, expected_categories)
        sample_id = row["sample_id"]
        if sample_id in ids:
            raise ValueError(f"duplicate source sample ID: {sample_id}")
        source_index = row["source_index"]
        if source_index in source_indexes:
            raise ValueError(f"duplicate source_index: {source_index}")
        prompt_hash = sha256_text(row["prompt"])
        normalized_hash = sha256_text(normalize_text(row["prompt"]))
        if prompt_hash in prompt_hashes or normalized_hash in normalized_prompt_hashes:
            raise ValueError(f"duplicate source prompt: {sample_id}")
        ids.add(sample_id)
        source_indexes.add(source_index)
        prompt_hashes.add(prompt_hash)
        normalized_prompt_hashes.add(normalized_hash)
    return rows


def load_exclusion_state(
    reg: dict[str, Any], source_rows: list[dict[str, Any]], protocol_commit: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = reg["exclusion_references"]
    eval_safety_path = resolve_repo_path(cfg["eval_safety"])
    eval_utility_path = resolve_repo_path(cfg["eval_utility"])
    split_path = resolve_repo_path(cfg["prior_split"])
    old37_path = resolve_repo_path(cfg["old37_reference"])
    check_hash(eval_safety_path, cfg["eval_safety_sha256"])
    check_hash(eval_utility_path, cfg["eval_utility_sha256"])
    check_hash(split_path, cfg["prior_split_sha256"])
    check_hash(old37_path, cfg["old37_reference_sha256"])

    eval_safety = load_eval_rows(
        eval_safety_path,
        expected_rows=int(cfg["expected_eval_safety_rows"]),
        expected_family=cfg["expected_eval_safety_family"],
        expected_source=cfg["expected_eval_safety_source"],
    )
    eval_utility = load_eval_rows(
        eval_utility_path,
        expected_rows=int(cfg["expected_eval_utility_rows"]),
        expected_family=cfg["expected_eval_utility_family"],
        expected_source=cfg["expected_eval_utility_source"],
    )
    eval_rows = eval_safety + eval_utility
    eval_by_id = {row["sample_id"]: row for row in eval_rows}
    if len(eval_by_id) != len(eval_rows):
        raise ValueError("evaluation IDs overlap across safety and utility files")

    split_rows = read_csv(split_path)
    if len(split_rows) != int(cfg["expected_prior_split_rows"]):
        raise ValueError("prior split row count changed")
    if any(set(row) != SPLIT_FIELDS for row in split_rows):
        raise ValueError("prior split schema changed")
    split_ids = {row["sample_id"] for row in split_rows}
    if len(split_ids) != len(split_rows) or not split_ids <= set(eval_by_id):
        raise ValueError("prior split IDs are duplicated or outside evaluation sources")
    for row in split_rows:
        if row["content_hash"] != eval_by_id[row["sample_id"]]["content_hash"]:
            raise ValueError(f"prior split content hash mismatch: {row['sample_id']}")
    safety_train = {row["sample_id"] for row in split_rows if row["split"] == "safety_train"}
    if len(safety_train) != int(cfg["expected_safety_train_rows"]):
        raise ValueError("prior safety-train count changed")

    old37_rows = read_csv(old37_path)
    if len(old37_rows) != int(cfg["expected_old37_reference_rows"]):
        raise ValueError("old37 reference row count changed")
    if any(set(row) != OLD37_FIELDS for row in old37_rows):
        raise ValueError("old37 reference schema changed")
    old37_ids = {
        row["sample_id"]
        for row in old37_rows
        if row["quality_upper_bound_available"].casefold() == "true"
    }
    if len(old37_ids) != int(cfg["expected_old37_rows"]) or not old37_ids <= safety_train:
        raise ValueError("old37 reference changed or is not a safety-train subset")

    tracked = tracked_risk_ids(reg, protocol_commit)
    prompt_scan = tracked_prompt_fingerprint_exposures(source_rows, reg, protocol_commit)
    fingerprint_ids = set(prompt_scan["source_ids"])
    incidental = set(cfg["incidental_main_session_prompt_ids"])
    exposed_source_ids = tracked | fingerprint_ids | incidental
    source_by_id = {row["sample_id"]: row for row in source_rows}
    if not exposed_source_ids <= set(source_by_id):
        raise ValueError("registered source exposure ID is absent from train_risk")

    reference_rows = [
        {"reference_id": row["sample_id"], "prompt": row["prompt"], "kind": "evaluation"}
        for row in eval_rows
    ]
    reference_rows.extend(
        {
            "reference_id": sample_id,
            "prompt": source_by_id[sample_id]["prompt"],
            "kind": "prior_or_incidental_exposure",
        }
        for sample_id in sorted(exposed_source_ids)
    )
    exposure_ledger = [
        {
            "sample_id": sample_id,
            "tracked_risk_id": sample_id in tracked,
            "tracked_prompt_fingerprint": sample_id in fingerprint_ids,
            "incidental_main_session": sample_id in incidental,
        }
        for sample_id in sorted(exposed_source_ids)
    ]
    audit = {
        "evaluation_rows": len(eval_rows),
        "prior_split_rows": len(split_rows),
        "prior_safety_train_rows": len(safety_train),
        "old37_rows": len(old37_ids),
        "tracked_risk_ids": sorted(tracked),
        "tracked_prompt_fingerprint_scan": prompt_scan,
        "incidental_main_session_prompt_ids": sorted(incidental),
        "exposure_ledger": exposure_ledger,
        "excluded_source_ids": sorted(exposed_source_ids),
        "source_harmful_responses_released": False,
        "old37_prompts_released": False,
        "heldout_content_opened": False,
    }
    return reference_rows, audit


def lexical_filter(
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    reg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligibility = reg["prompt_eligibility"]
    shingle_threshold = float(eligibility["restricted_lexical_overlap_threshold"])
    sequence_threshold = float(
        eligibility["normalized_character_sequence_similarity_threshold"]
    )
    max_width = int(eligibility["maximum_shingle_size"])
    prepared = [
        {
            **row,
            "normalized": normalize_text(row["prompt"]),
            "features": token_shingles(row["prompt"], max_width),
            "sequence_matcher": difflib.SequenceMatcher(
                None, "", normalize_text(row["prompt"]), autojunk=False
            ),
        }
        for row in references
    ]
    reference_exact_hashes = {sha256_text(row["prompt"]) for row in references}
    reference_normalized_hashes = {
        sha256_text(normalize_text(row["prompt"])) for row in references
    }

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["prompt"]
        normalized = normalize_text(prompt)
        exact_hash = sha256_text(prompt)
        normalized_hash = sha256_text(normalized)
        if exact_hash in reference_exact_hashes or normalized_hash in reference_normalized_hashes:
            excluded.append({"sample_id": row["sample_id"], "reason": "exact_reference_overlap"})
            continue

        features = token_shingles(prompt, max_width)
        best: dict[str, Any] | None = None
        best_scaled = -1.0
        for reference in prepared:
            jaccard, overlap, shingle_max = lexical_similarity(features, reference["features"])
            left_len = len(normalized)
            right_len = len(reference["normalized"])
            length_upper = (
                2 * min(left_len, right_len) / (left_len + right_len)
                if left_len + right_len
                else 1.0
            )
            sequence = 0.0
            if length_upper >= sequence_threshold:
                matcher = reference["sequence_matcher"]
                matcher.set_seq1(normalized)
                # quick_ratio is an upper bound on ratio, so this skips only
                # pairs that cannot reach the preregistered threshold.
                if matcher.quick_ratio() >= sequence_threshold:
                    sequence = matcher.ratio()
            scaled = max(
                shingle_max / shingle_threshold,
                sequence / sequence_threshold,
            )
            if scaled > best_scaled:
                best_scaled = scaled
                best = {
                    "reference_id": reference["reference_id"],
                    "reference_kind": reference["kind"],
                    "jaccard": jaccard,
                    "overlap_coefficient": overlap,
                    "maximum_shingle_similarity": shingle_max,
                    "sequence_similarity": sequence,
                }
        assert best is not None
        if (
            best["maximum_shingle_similarity"] >= shingle_threshold
            or best["sequence_similarity"] >= sequence_threshold
        ):
            excluded.append(
                {"sample_id": row["sample_id"], "reason": "restricted_lexical_overlap", **best}
            )
            continue
        kept.append(row)
    return kept, excluded


def eligible_source_rows(
    source_rows: list[dict[str, Any]], reg: dict[str, Any], excluded_ids: set[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligibility = reg["prompt_eligibility"]
    minimum = int(eligibility["minimum_whitespace_tokens"])
    maximum = int(eligibility["maximum_whitespace_tokens"])
    severities = set(eligibility["allowed_severity_levels"])
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        sample_id = row["sample_id"]
        if sample_id in excluded_ids:
            counters["registered_or_scanned_exposure_id"] += 1
            continue
        token_count = len(row["prompt"].split())
        if token_count < minimum or token_count > maximum:
            counters["prompt_length"] += 1
            continue
        severity = row["severity_level"]
        if severity not in severities:
            raise ValueError(f"source severity escaped strict validation: {sample_id}")
        true_categories = sorted(
            category for category, present in row["harm_category"].items() if present
        )
        if eligibility["require_at_least_one_harm_category"] and not true_categories:
            counters["no_harm_category"] += 1
            continue
        rows.append(
            {
                **row,
                "prompt_token_count": token_count,
                "true_categories": true_categories,
                "prompt_sha256": sha256_text(row["prompt"]),
                "normalized_prompt_sha256": sha256_text(normalize_text(row["prompt"])),
            }
        )
    return rows, counters


def expand_slots(reg: dict[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    table = reg["selection"]["category_severity_slots"]
    for category in sorted(table):
        for severity in (1, 2, 3):
            count = int(table[category].get(str(severity), 0))
            for ordinal in range(1, count + 1):
                slots.append(
                    {
                        "slot_id": f"{category}|severity={severity}|slot={ordinal}",
                        "category": category,
                        "severity": severity,
                        "ordinal": ordinal,
                    }
                )
    expected = int(reg["selection"]["prompt_count"])
    if len(slots) != expected:
        raise ValueError(f"registry expands to {len(slots)} slots, expected {expected}")
    return slots


def deterministic_match(
    rows: list[dict[str, Any]], slots: list[dict[str, Any]], salt: str
) -> dict[str, str]:
    row_by_id = {row["sample_id"]: row for row in rows}
    candidates: dict[str, list[str]] = {}
    for slot in slots:
        eligible = [
            row["sample_id"]
            for row in rows
            if slot["category"] in row["true_categories"]
            and row["severity_level"] == slot["severity"]
        ]
        eligible.sort(key=lambda sample_id: stable_rank(salt, slot["slot_id"], sample_id))
        if not eligible:
            raise ValueError(f"no eligible candidate for slot {slot['slot_id']}")
        candidates[slot["slot_id"]] = eligible

    slot_order = sorted(
        (slot["slot_id"] for slot in slots),
        key=lambda slot_id: (len(candidates[slot_id]), slot_id),
    )
    candidate_to_slot: dict[str, str] = {}
    slot_to_candidate: dict[str, str] = {}

    def augment(slot_id: str, seen: set[str]) -> bool:
        for sample_id in candidates[slot_id]:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            incumbent = candidate_to_slot.get(sample_id)
            if incumbent is None or augment(incumbent, seen):
                candidate_to_slot[sample_id] = slot_id
                slot_to_candidate[slot_id] = sample_id
                return True
        return False

    for slot_id in slot_order:
        if not augment(slot_id, set()):
            raise ValueError(f"deterministic matching failed at slot {slot_id}")
    if len(slot_to_candidate) != len(slots) or len(set(slot_to_candidate.values())) != len(slots):
        raise ValueError("matching did not produce one unique prompt per slot")
    if not set(slot_to_candidate.values()) <= set(row_by_id):
        raise AssertionError("matching returned an unknown source row")
    return slot_to_candidate


def build_prompt_package(
    protocol_commit: str, randomness_context: dict[str, Any]
) -> dict[str, Any]:
    validate_protocol_commit_format(protocol_commit)
    reg = registry()
    validate_boundaries(reg)
    validate_entry_route(reg)
    source_rows = load_source_rows(reg)
    references, exclusion_audit = load_exclusion_state(reg, source_rows, protocol_commit)
    excluded_ids = set(exclusion_audit["excluded_source_ids"])
    structurally_eligible, structural_exclusions = eligible_source_rows(
        source_rows, reg, excluded_ids
    )
    lexically_eligible, lexical_exclusions = lexical_filter(
        structurally_eligible, references, reg
    )
    slots = expand_slots(reg)
    salts = {
        purpose: derive_salt(reg, protocol_commit, randomness_context, purpose)
        for purpose in reg["selection"]["randomness"]["purposes"]
    }
    assignment = deterministic_match(lexically_eligible, slots, salts["prompt_matching"])
    slot_by_id = {slot["slot_id"]: slot for slot in slots}
    row_by_id = {row["sample_id"]: row for row in lexically_eligible}

    selected: list[dict[str, Any]] = []
    for slot_id, sample_id in sorted(assignment.items()):
        slot = slot_by_id[slot_id]
        row = row_by_id[sample_id]
        selected.append(
            {
                "sample_id": sample_id,
                "source_index": row["source_index"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "normalized_prompt_sha256": row["normalized_prompt_sha256"],
                "prompt_token_count": row["prompt_token_count"],
                "assigned_category": slot["category"],
                "severity_level": slot["severity"],
                "all_source_categories": row["true_categories"],
                "slot_id": slot_id,
                "selection_rank_sha256": stable_rank(
                    salts["prompt_matching"], slot_id, sample_id
                ),
                "opaque_prompt_id": "step3o_o4_prompt_"
                + stable_rank(salts["opaque_prompt_id"], sample_id)[:18],
            }
        )

    canonical = sorted(selected, key=lambda row: row["opaque_prompt_id"])
    author_a_rows = sorted(
        selected,
        key=lambda row: stable_rank(salts["author_a_packet_order"], row["opaque_prompt_id"]),
    )
    author_b_rows = sorted(
        selected,
        key=lambda row: stable_rank(salts["author_b_packet_order"], row["opaque_prompt_id"]),
    )
    author_a_packet = [
        {"opaque_prompt_id": row["opaque_prompt_id"], "prompt": row["prompt"]}
        for row in author_a_rows
    ]
    author_b_packet = [
        {"opaque_prompt_id": row["opaque_prompt_id"], "prompt": row["prompt"]}
        for row in author_b_rows
    ]
    key_fields = [
        "opaque_prompt_id",
        "sample_id",
        "source_index",
        "assigned_category",
        "severity_level",
        "all_source_categories",
        "prompt_sha256",
        "normalized_prompt_sha256",
        "prompt_token_count",
        "slot_id",
        "selection_rank_sha256",
    ]
    key = [
        {**row, "all_source_categories": "|".join(row["all_source_categories"])}
        for row in canonical
    ]
    author_a_template = [
        {"opaque_prompt_id": row["opaque_prompt_id"], "response": ""}
        for row in author_a_rows
    ]
    author_b_template = [
        {"opaque_prompt_id": row["opaque_prompt_id"], "response": ""}
        for row in author_b_rows
    ]

    selection_cfg = reg["selection"]
    category_counts = Counter(row["assigned_category"] for row in selected)
    severity_counts = Counter(str(row["severity_level"]) for row in selected)
    expected_category = {
        category: sum(int(value) for value in severities.values())
        for category, severities in selection_cfg["category_severity_slots"].items()
    }
    expected_severity = {
        str(key): int(value)
        for key, value in selection_cfg["expected_severity_counts"].items()
    }
    selected_ids = {row["sample_id"] for row in selected}
    selected_prompt_hashes = {row["prompt_sha256"] for row in selected}
    selected_normalized_hashes = {row["normalized_prompt_sha256"] for row in selected}
    reference_prompt_hashes = {sha256_text(row["prompt"]) for row in references}
    reference_normalized_hashes = {
        sha256_text(normalize_text(row["prompt"])) for row in references
    }
    order_a = [row["opaque_prompt_id"] for row in author_a_rows]
    order_b = [row["opaque_prompt_id"] for row in author_b_rows]
    gates = {
        "exact_prompt_count": len(selected) == int(selection_cfg["prompt_count"]),
        "unique_source_ids": len(selected_ids) == len(selected),
        "unique_prompt_hashes": len(selected_prompt_hashes) == len(selected),
        "category_counts_match": dict(sorted(category_counts.items()))
        == dict(sorted(expected_category.items())),
        "severity_counts_match": dict(sorted(severity_counts.items()))
        == dict(sorted(expected_severity.items())),
        "registered_or_scanned_exposure_id_overlap_zero": not (selected_ids & excluded_ids),
        "evaluation_or_exposure_prompt_hash_overlap_zero": not (
            selected_prompt_hashes & reference_prompt_hashes
        ),
        "evaluation_or_exposure_normalized_prompt_hash_overlap_zero": not (
            selected_normalized_hashes & reference_normalized_hashes
        ),
        "author_packet_id_sets_identical": set(order_a) == set(order_b),
        "author_packet_orders_distinct": order_a != order_b,
        "source_response_fields_absent": not any(
            "response" in row for row in author_a_packet + author_b_packet + key
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"O4 prompt selection gate failed: {gates}")

    source_audit = {
        "name": "step3o_o4_source_and_exclusion_audit_v2",
        "protocol_commit": protocol_commit,
        "beacon_round": randomness_context["beacon_round"],
        "beacon_randomness_sha256": sha256_text(
            randomness_context["beacon_randomness"]
        ),
        "source_rows": len(source_rows),
        "structurally_eligible_before_lexical_audit": len(structurally_eligible),
        "lexically_eligible": len(lexically_eligible),
        "structural_exclusion_counts": dict(sorted(structural_exclusions.items())),
        "restricted_lexical_exclusion_count": len(lexical_exclusions),
        "restricted_lexical_exclusions": lexical_exclusions,
        "restricted_lexical_overlap_threshold": reg["prompt_eligibility"][
            "restricted_lexical_overlap_threshold"
        ],
        "normalized_character_sequence_similarity_threshold": reg[
            "prompt_eligibility"
        ][
            "normalized_character_sequence_similarity_threshold"
        ],
        "normalization": "Unicode NFKC, casefold, collapsed whitespace",
        "audit_scope": "tracked structured prompt fingerprint plus restricted Unicode token-shingle and character-sequence audit; not semantic or pretraining decontamination",
        **exclusion_audit,
    }
    selection_audit = {
        "name": "step3o_o4_prompt_selection_audit_v2",
        "attempt_id": reg["attempt"]["attempt_id"],
        "protocol_commit": protocol_commit,
        "randomness_request_commit": randomness_context.get("request_commit"),
        "randomness_pulse_commit": randomness_context.get("pulse_commit"),
        "beacon_round": randomness_context["beacon_round"],
        "beacon_randomness": randomness_context["beacon_randomness"],
        "salt_derivation": reg["selection"]["randomness"]["derivation"],
        "derived_salts_sha256": salts,
        "selected_prompts": len(selected),
        "category_counts": dict(sorted(category_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "prompt_token_count_min": min(row["prompt_token_count"] for row in selected),
        "prompt_token_count_max": max(row["prompt_token_count"] for row in selected),
        "slot_candidate_counts": {
            slot["slot_id"]: sum(
                slot["category"] in row["true_categories"]
                and row["severity_level"] == slot["severity"]
                for row in lexically_eligible
            )
            for slot in slots
        },
        "gates": gates,
        "source_harmful_responses_released": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return {
        "author_a_packet": author_a_packet,
        "author_b_packet": author_b_packet,
        "key": key,
        "key_fields": key_fields,
        "author_a_template": author_a_template,
        "author_b_template": author_b_template,
        "source_audit": source_audit,
        "selection_audit": selection_audit,
    }


def artifact_from_bytes(path: Path, data: bytes) -> dict[str, Any]:
    return {"path": repo_path(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def artifact_from_text(path: Path, content: str) -> dict[str, Any]:
    return artifact_from_bytes(path, content.encode("utf-8"))


def protocol_inputs(reg: dict[str, Any]) -> list[Path]:
    source = reg["source"]
    exclusions = reg["exclusion_references"]
    return [
        GIT_ATTRIBUTES,
        AGENT_DOC,
        RANDOMNESS_WORKFLOW,
        DRAND_VERIFIER_PACKAGE,
        DRAND_VERIFIER_LOCK,
        DRAND_VERIFIER_SCRIPT,
        REGISTRY,
        PLAN,
        AUTHORING_INSTRUCTIONS,
        QUALIFICATION_RUBRIC,
        HANDOFF,
        Path(__file__).resolve(),
        TEST_SCRIPT,
        RANDOMNESS_SCRIPT,
        RANDOMNESS_TEST,
        QUALIFICATION_SCRIPT,
        QUALIFICATION_TEST,
        STAGE_SCRIPT,
        STAGE_TEST,
        MEASUREMENT_SCRIPT,
        README,
        EXPERIMENT_TASK,
        resolve_repo_path(reg["entry_route"]),
        resolve_repo_path(source["train_risk"]),
        resolve_repo_path(exclusions["eval_safety"]),
        resolve_repo_path(exclusions["eval_utility"]),
        resolve_repo_path(exclusions["prior_split"]),
        resolve_repo_path(exclusions["old37_reference"]),
    ]


def build_attempt_ledger(
    reg: dict[str, Any], protocol_commit: str, randomness_context: dict[str, Any]
) -> dict[str, Any]:
    return {
        "name": "step3o_o4_attempt_ledger_v1",
        "attempt_id": reg["attempt"]["attempt_id"],
        "attempt_number": 1,
        "status": "P1_MATERIALIZATION_STARTED",
        "protocol_commit": protocol_commit,
        "protocol_commit_parent": reg["entry_commit"],
        "randomness_request_commit": randomness_context["request_commit"],
        "randomness_pulse_commit": randomness_context["pulse_commit"],
        "beacon_round": randomness_context["beacon_round"],
        "beacon_randomness": randomness_context["beacon_randomness"],
        "replacement_or_retry_allowed": False,
        "failure_is_terminal_for_attempt": True,
        "prompt_replacement_allowed": False,
        "extra_authoring_rounds_allowed": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def protocol_report_text(
    package: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
) -> str:
    return f"""# Step3O-O4 P1 Prompt Freeze

Status: PASS. Exactly `{len(package['author_a_packet'])}` new PKU training
prompts were selected before target authoring.

- attempt: `step3o_o4_attempt_001`;
- P0 protocol commit: `{protocol_commit}`;
- drand request commit: `{randomness_context['request_commit']}`;
- drand pulse commit: `{randomness_context['pulse_commit']}`;
- drand target round: `{randomness_context['beacon_round']}`;
- category slots: `19` categories, exact registered counts;
- severity counts: `{json.dumps(package['selection_audit']['severity_counts'], sort_keys=True)}`;
- Author A/B prompt sets: identical;
- Author A/B packet orders: separately salted and distinct;
- source harmful responses released: `false`;
- old 37 prompts released: `false`;
- target authoring started: `false`;
- optimizer steps: `0`;
- training authorized: `false`.

The contamination audit combines tracked structured prompt fingerprints with
Unicode-NFKC token 1-5 gram and character-sequence checks at the frozen
thresholds. It is not semantic or pretraining decontamination. Any in-memory
selection made before the P0 commit is non-protocol and void.
"""


def build_packet_manifest(
    package: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
    contents: dict[Path, str],
) -> dict[str, Any]:
    artifact_paths = [
        ATTEMPT_LEDGER,
        AUTHOR_A_PACKET,
        AUTHOR_B_PACKET,
        AUTHORING_KEY,
        AUTHOR_A_TEMPLATE,
        AUTHOR_B_TEMPLATE,
        SOURCE_AUDIT,
        SELECTION_AUDIT,
    ]
    return {
        "name": "step3o_o4_authoring_packet_manifest_v2",
        "status": "frozen_before_target_authoring",
        "attempt_id": "step3o_o4_attempt_001",
        "protocol_commit": protocol_commit,
        "randomness_request_commit": randomness_context["request_commit"],
        "randomness_pulse_commit": randomness_context["pulse_commit"],
        "beacon_round": randomness_context["beacon_round"],
        "prompt_rows_per_author": len(package["author_a_packet"]),
        "visible_fields": ["opaque_prompt_id", "prompt"],
        "author_packet_id_sets_identical": True,
        "author_packet_orders_separately_salted": True,
        "source_response_fields_released": False,
        "authoring_attempts_per_prompt": 2,
        "extra_authoring_rounds_allowed": 0,
        "artifacts": [artifact_from_text(path, contents[path]) for path in artifact_paths],
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def protocol_input_artifacts(reg: dict[str, Any], protocol_commit: str) -> list[dict[str, Any]]:
    return [
        artifact_from_bytes(path, git_blob_bytes(protocol_commit, path))
        for path in protocol_inputs(reg)
    ]


def build_protocol_manifest(
    reg: dict[str, Any],
    package: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
    contents: dict[Path, str],
) -> dict[str, Any]:
    output_paths = [
        ATTEMPT_LEDGER,
        AUTHOR_A_PACKET,
        AUTHOR_B_PACKET,
        AUTHORING_KEY,
        AUTHOR_A_TEMPLATE,
        AUTHOR_B_TEMPLATE,
        PACKET_MANIFEST,
        SOURCE_AUDIT,
        SELECTION_AUDIT,
        PROTOCOL_REPORT,
    ]
    return {
        "name": "step3o_o4_p1_prompt_freeze_manifest_v2",
        "attempt_id": reg["attempt"]["attempt_id"],
        "protocol_commit": protocol_commit,
        "protocol_commit_parent": reg["entry_commit"],
        "randomness_request_commit": randomness_context["request_commit"],
        "randomness_pulse_commit": randomness_context["pulse_commit"],
        "beacon_round": randomness_context["beacon_round"],
        "beacon_randomness": randomness_context["beacon_randomness"],
        "prompt_freeze_parent_commit": randomness_context[
            "selection_parent_commit"
        ],
        "entry_commit": reg["entry_commit"],
        "entry_route_sha256": reg["entry_route_sha256"],
        "protocol_inputs": protocol_input_artifacts(reg, protocol_commit),
        "randomness_inputs": [
            artifact_from_bytes(BEACON_REQUEST, BEACON_REQUEST.read_bytes()),
            artifact_from_bytes(BEACON_PULSE, BEACON_PULSE.read_bytes()),
        ],
        "outputs": [artifact_from_text(path, contents[path]) for path in output_paths],
        "author_a_prompt_rows": len(package["author_a_packet"]),
        "author_b_prompt_rows": len(package["author_b_packet"]),
        "source_harmful_responses_released": False,
        "old37_prompts_released": False,
        "heldout_content_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "pcsr_r2_authorized": False,
    }


def build_freeze_contents(
    reg: dict[str, Any],
    package: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
) -> dict[Path, str]:
    contents = {
        ATTEMPT_LEDGER: json_text(
            build_attempt_ledger(reg, protocol_commit, randomness_context)
        ),
        AUTHOR_A_PACKET: jsonl_text(package["author_a_packet"]),
        AUTHOR_B_PACKET: jsonl_text(package["author_b_packet"]),
        AUTHORING_KEY: csv_text(package["key"], package["key_fields"]),
        AUTHOR_A_TEMPLATE: jsonl_text(package["author_a_template"]),
        AUTHOR_B_TEMPLATE: jsonl_text(package["author_b_template"]),
        SOURCE_AUDIT: json_text(package["source_audit"]),
        SELECTION_AUDIT: json_text(package["selection_audit"]),
        PROTOCOL_REPORT: protocol_report_text(
            package, protocol_commit, randomness_context
        ),
    }
    contents[PACKET_MANIFEST] = json_text(
        build_packet_manifest(package, protocol_commit, randomness_context, contents)
    )
    contents[PROTOCOL_MANIFEST] = json_text(
        build_protocol_manifest(
            reg, package, protocol_commit, randomness_context, contents
        )
    )
    return contents


def expected_output_paths() -> list[Path]:
    return [
        ATTEMPT_LEDGER,
        AUTHOR_A_PACKET,
        AUTHOR_B_PACKET,
        AUTHORING_KEY,
        AUTHOR_A_TEMPLATE,
        AUTHOR_B_TEMPLATE,
        PACKET_MANIFEST,
        SOURCE_AUDIT,
        SELECTION_AUDIT,
        PROTOCOL_REPORT,
        PROTOCOL_MANIFEST,
    ]


def verify_creation_parent(path: Path, expected_parent: str) -> str:
    commits = added_commits(path)
    if len(commits) != 1:
        raise ValueError(f"expected exactly one creation commit for {repo_path(path)}")
    creation_commit = commits[0]
    parent_line = str(_git(["rev-list", "--parents", "-n", "1", creation_commit])).split()
    if len(parent_line) != 2 or parent_line[1] != expected_parent:
        raise ValueError(
            f"{repo_path(path)} creation parent differs from {expected_parent}"
        )
    if not git_is_ancestor(creation_commit, "HEAD"):
        raise ValueError(f"creation commit is not an ancestor of HEAD: {creation_commit}")
    return creation_commit


def verify_exact_creation_commit(
    paths: Iterable[Path],
    marker: Path,
    expected_parent: str,
    *,
    reg: dict[str, Any],
    require_pushed: bool,
) -> str:
    frozen_paths = list(paths)
    if not frozen_paths or marker not in frozen_paths or len(frozen_paths) != len(set(frozen_paths)):
        raise ValueError("exact creation boundary requires unique paths including its marker")
    creation_commit = verify_creation_parent(marker, expected_parent)
    for path in frozen_paths:
        if added_commits(path) != [creation_commit]:
            raise ValueError(
                f"stage path was not first created in {creation_commit}: {repo_path(path)}"
            )
    expected = {repo_path(path) for path in frozen_paths}
    observed = commit_changed_paths(creation_commit)
    if observed != expected:
        raise ValueError(
            "creation commit changed an unexpected path set: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    require_immutable_since(frozen_paths, creation_commit)
    if require_pushed:
        require_remote_commits(reg, [creation_commit])
    return creation_commit


def verify_manifest_exact(actual_text: str, expected_text: str) -> None:
    actual = parse_json_text(actual_text, "actual protocol manifest")
    expected = parse_json_text(expected_text, "expected protocol manifest")
    if actual != expected or actual_text.encode("utf-8") != expected_text.encode("utf-8"):
        raise ValueError("protocol manifest is missing, hollow, reordered, or tampered")


def build_attempt_stop_event(
    reg: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
    phase: str,
    error: Exception,
    evidence_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": "step3o_o4_attempt_stop_v1",
        "attempt_id": reg["attempt"]["attempt_id"],
        "status": "STOP_PROMPT_POOL",
        "phase": phase,
        "protocol_commit": protocol_commit,
        "parent_commit": randomness_context["pulse_commit"],
        "beacon_round": randomness_context["beacon_round"],
        "error_type": type(error).__name__,
        "error_message_sha256": sha256_text(str(error)),
        "evidence_artifacts": evidence_artifacts,
        "replacement_or_retry_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def _write_stop_after_failure(
    reg: dict[str, Any],
    protocol_commit: str,
    randomness_context: dict[str, Any],
    phase: str,
    error: Exception,
) -> None:
    if ATTEMPT_STOP.exists():
        return
    evidence_artifacts = [
        artifact_from_bytes(path, path.read_bytes())
        for path in expected_output_paths()
        if path.exists()
    ]
    write_frozen(
        ATTEMPT_STOP,
        json_text(
            build_attempt_stop_event(
                reg,
                protocol_commit,
                randomness_context,
                phase,
                error,
                evidence_artifacts,
            )
        ),
    )


def freeze_prompts() -> dict[str, Any]:
    reg = registry()
    validate_boundaries(reg)
    protocol_commit = resolve_protocol_commit(reg, require_pushed=True)
    require_protocol_inputs_frozen(reg, protocol_commit)
    randomness_context = verify_randomness_chain(
        reg, protocol_commit, require_pushed=True
    )
    if git_head() != randomness_context["selection_parent_commit"]:
        raise ValueError(
            "P1 prompt materialization must start directly on the pushed drand pulse commit"
        )
    allowed_prefreeze = {repo_path(BEACON_REQUEST), repo_path(BEACON_PULSE)}
    existing = {
        repo_path(path) for path in BASE.rglob("*") if path.is_file()
    }
    if existing != allowed_prefreeze:
        raise ValueError(
            f"attempt 001 prefreeze path set is not pristine: {sorted(existing)}"
        )

    ledger_text = json_text(
        build_attempt_ledger(reg, protocol_commit, randomness_context)
    )
    write_frozen(ATTEMPT_LEDGER, ledger_text)
    phase = "build_prompt_package"
    try:
        package = build_prompt_package(protocol_commit, randomness_context)
        phase = "build_freeze_contents"
        contents = build_freeze_contents(
            reg, package, protocol_commit, randomness_context
        )
        if contents[ATTEMPT_LEDGER] != ledger_text:
            raise ValueError("attempt ledger reconstruction changed after selection")
        phase = "write_p1_outputs"
        for path in expected_output_paths():
            if path != ATTEMPT_LEDGER:
                write_frozen(path, contents[path])
    except Exception as exc:
        _write_stop_after_failure(
            reg, protocol_commit, randomness_context, phase, exc
        )
        raise
    return parse_json_text(contents[PROTOCOL_MANIFEST], "generated P1 manifest")


def verify_prompt_stop() -> dict[str, Any]:
    reg = registry()
    validate_boundaries(reg)
    protocol_commit = resolve_protocol_commit(reg, require_pushed=True)
    require_protocol_inputs_frozen(reg, protocol_commit)
    randomness = verify_randomness_chain(reg, protocol_commit, require_pushed=True)
    value = parse_frozen_json_bytes(ATTEMPT_STOP.read_bytes(), "P1 terminal stop")
    expected_fields = {
        "name",
        "attempt_id",
        "status",
        "phase",
        "protocol_commit",
        "parent_commit",
        "beacon_round",
        "error_type",
        "error_message_sha256",
        "evidence_artifacts",
        "replacement_or_retry_allowed",
        "optimizer_steps",
        "training_authorized",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("P1 terminal-stop schema mismatch")
    if (
        value["name"] != "step3o_o4_attempt_stop_v1"
        or value["attempt_id"] != reg["attempt"]["attempt_id"]
        or value["status"] != "STOP_PROMPT_POOL"
        or not isinstance(value["phase"], str)
        or not value["phase"]
        or value["protocol_commit"] != protocol_commit
        or value["parent_commit"] != randomness["pulse_commit"]
        or value["beacon_round"] != randomness["beacon_round"]
        or not isinstance(value["error_type"], str)
        or not value["error_type"]
        or not isinstance(value["error_message_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["error_message_sha256"])
        or value["replacement_or_retry_allowed"] is not False
        or value["optimizer_steps"] != 0
        or value["training_authorized"] is not False
    ):
        raise ValueError("P1 terminal-stop boundary mismatch")

    artifacts = value["evidence_artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("P1 terminal-stop evidence must be a list")
    allowed = {repo_path(path): path for path in expected_output_paths()}
    observed_paths: list[Path] = []
    observed_names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"}:
            raise ValueError("P1 terminal-stop evidence artifact schema mismatch")
        path_name = artifact["path"]
        if path_name in observed_names or path_name not in allowed:
            raise ValueError("P1 terminal-stop evidence path is duplicate or unregistered")
        observed_names.add(path_name)
        path = allowed[path_name]
        data = path.read_bytes()
        if artifact != artifact_from_bytes(path, data):
            raise ValueError("P1 terminal-stop evidence hash or length mismatch")
        observed_paths.append(path)
    expected_artifact_order = [
        repo_path(path) for path in expected_output_paths() if repo_path(path) in observed_names
    ]
    if [artifact["path"] for artifact in artifacts] != expected_artifact_order:
        raise ValueError("P1 terminal-stop evidence order differs from the registry")

    commit = verify_exact_creation_commit(
        [*observed_paths, ATTEMPT_STOP],
        ATTEMPT_STOP,
        randomness["pulse_commit"],
        reg=reg,
        require_pushed=True,
    )
    tracked_at_stop = {
        line
        for line in str(
            _git(
                [
                    "ls-tree",
                    "-r",
                    "--name-only",
                    commit,
                    "--",
                    repo_path(BASE),
                ]
            )
        ).splitlines()
        if line
    }
    expected_tree = {
        repo_path(BEACON_REQUEST),
        repo_path(BEACON_PULSE),
        repo_path(ATTEMPT_STOP),
        *observed_names,
    }
    if tracked_at_stop != expected_tree:
        raise ValueError("P1 terminal-stop result tree contains unregistered artifacts")
    return {
        "name": "step3o_o4_p1_terminal_stop_verification_v1",
        "checks_pass": True,
        "commit": commit,
        "status": "STOP_PROMPT_POOL",
        "training_authorized": False,
    }


def verify_prompts() -> dict[str, Any]:
    reg = registry()
    validate_boundaries(reg)
    protocol_commit = resolve_protocol_commit(reg, require_pushed=True)
    require_protocol_inputs_frozen(reg, protocol_commit)
    randomness_context = verify_randomness_chain(
        reg, protocol_commit, require_pushed=True
    )
    outputs = expected_output_paths()
    creation_commit = verify_exact_creation_commit(
        outputs,
        PROTOCOL_MANIFEST,
        randomness_context["selection_parent_commit"],
        reg=reg,
        require_pushed=True,
    )

    tracked_at_creation = set(
        line
        for line in str(
            _git(
                [
                    "ls-tree",
                    "-r",
                    "--name-only",
                    creation_commit,
                    "--",
                    repo_path(BASE),
                ]
            )
        ).splitlines()
        if line
    )
    expected_paths = {
        repo_path(path) for path in [BEACON_REQUEST, BEACON_PULSE, *outputs]
    }
    if tracked_at_creation != expected_paths:
        raise ValueError(
            "P1 creation tree mismatch: "
            f"missing={sorted(expected_paths-tracked_at_creation)}, "
            f"extra={sorted(tracked_at_creation-expected_paths)}"
        )

    tracked_output = set(
        line
        for line in str(_git(["ls-files", "--", repo_path(BASE)])).splitlines()
        if line
    )
    if not expected_paths <= tracked_output:
        raise ValueError(
            f"P1 tracked output path set is incomplete: missing={sorted(expected_paths-tracked_output)}"
        )
    filesystem_output = {
        repo_path(path) for path in BASE.rglob("*") if path.is_file()
    }
    if not expected_paths <= filesystem_output:
        raise ValueError(
            f"P1 filesystem output path set is incomplete: missing={sorted(expected_paths-filesystem_output)}"
        )

    package = build_prompt_package(protocol_commit, randomness_context)
    contents = build_freeze_contents(
        reg, package, protocol_commit, randomness_context
    )
    failed = [
        repo_path(path)
        for path in outputs
        if path.read_bytes() != contents[path].encode("utf-8")
    ]
    if failed:
        raise ValueError(f"O4 prompt freeze byte verification failed: {failed}")
    verify_manifest_exact(
        PROTOCOL_MANIFEST.read_text(encoding="utf-8"), contents[PROTOCOL_MANIFEST]
    )
    return {
        "name": "step3o_o4_p1_prompt_freeze_verification_v2",
        "checks_pass": True,
        "attempt_id": reg["attempt"]["attempt_id"],
        "protocol_commit": protocol_commit,
        "prompt_freeze_creation_commit": creation_commit,
        "randomness_request_commit": randomness_context["request_commit"],
        "randomness_pulse_commit": randomness_context["pulse_commit"],
        "beacon_round": randomness_context["beacon_round"],
        "beacon_randomness": randomness_context["beacon_randomness"],
        "author_a_prompt_rows": len(package["author_a_packet"]),
        "author_b_prompt_rows": len(package["author_b_packet"]),
        "category_counts": package["selection_audit"]["category_counts"],
        "severity_counts": package["selection_audit"]["severity_counts"],
        "source_harmful_responses_released": False,
        "old37_prompts_released": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "lock-head",
            "freeze-prompts",
            "verify-prompts",
            "verify-prompt-stop",
        ],
    )
    args = parser.parse_args()
    if args.command == "lock-head":
        result = create_remote_lock()
    elif args.command == "freeze-prompts":
        result = freeze_prompts()
    elif args.command == "verify-prompts":
        result = verify_prompts()
    else:
        result = verify_prompt_stop()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
