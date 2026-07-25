#!/usr/bin/env python3
"""Tests for the Step3O-O4 prospective public-randomness freeze."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import step3o_o4_randomness as beacon
import step3o_o4_training_pool as pool


class FakeResponse:
    def __init__(self, value: dict) -> None:
        self.data = json.dumps(value).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


class Step3OO4RandomnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = pool.registry()
        cls.protocol_commit = pool.git_head()
        cls.signature = "00" * 96
        cls.pulse = {
            "round": 100,
            "randomness": hashlib.sha256(bytes.fromhex(cls.signature)).hexdigest(),
            "signature": cls.signature,
            "previous_signature": "11" * 96,
        }
        cfg = cls.reg["randomness_beacon"]
        cls.info = {
            "public_key": cfg["public_key"],
            "period": cfg["period_seconds"],
            "genesis_time": cfg["genesis_time_unix"],
            "hash": cfg["chain_hash"],
            "schemeID": cfg["scheme_id"],
        }

    def test_round_time_uses_registered_genesis_and_period(self) -> None:
        cfg = self.reg["randomness_beacon"]
        expected = cfg["genesis_time_unix"] + 99 * cfg["period_seconds"]
        self.assertEqual(pool.drand_round_time(self.reg, 100), expected)

    def test_request_workflow_is_push_and_path_bound(self) -> None:
        workflow = pool.RANDOMNESS_WORKFLOW.read_text(encoding="utf-8")
        artifact = self.reg["randomness_beacon"]["request_artifact"]
        self.assertIn('"on":\n  push:', workflow)
        self.assertIn("- exp/step3-response-quality-persistence", workflow)
        self.assertIn(f"- {artifact}", workflow)
        self.assertIn("step3o_o4_randomness.py", workflow)
        self.assertIn("ci-verify-request-future", workflow)
        self.assertIn("npm ci", workflow)
        self.assertNotIn("workflow_dispatch", workflow)

    def test_authenticated_github_api_uses_frozen_token_environment(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["url"] = request.full_url
            observed["accept"] = request.headers["Accept"]
            observed["version"] = request.headers["X-github-api-version"]
            observed["authorization"] = request.headers["Authorization"]
            observed["timeout"] = timeout
            return FakeResponse({"check_runs": []})

        with patch.dict(os.environ, {"STEP3O_O4_GITHUB_TOKEN": "test-token"}):
            original = pool.urllib.request.urlopen
            try:
                pool.urllib.request.urlopen = fake_urlopen
                value = pool.github_api_json(
                    "repos/0bCyan/lora-safety-spectral-geometry/commits/abc/check-runs",
                    "synthetic GitHub API",
                    reg=self.reg,
                )
            finally:
                pool.urllib.request.urlopen = original
        self.assertEqual(value, {"check_runs": []})
        self.assertEqual(
            observed["url"],
            "https://api.github.com/repos/0bCyan/lora-safety-spectral-geometry/commits/abc/check-runs",
        )
        self.assertEqual(observed["accept"], "application/vnd.github+json")
        self.assertEqual(observed["version"], "2022-11-28")
        self.assertEqual(observed["authorization"], "Bearer test-token")
        self.assertEqual(observed["timeout"], 30)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "STEP3O_O4_GITHUB_TOKEN"):
                pool.github_api_json(
                    "repos/0bCyan/lora-safety-spectral-geometry/commits/abc/check-runs",
                    "synthetic GitHub API",
                    reg=self.reg,
                )

    def test_canonical_remote_tip_binds_branch_url_api_tip_and_immutable_releases(self) -> None:
        cfg = self.reg["repository_boundary"]
        remote_tip = "a" * 40

        def invoke(
            *,
            local_branch: str = cfg["branch"],
            remote_url: str = cfg["remote_url"],
            api_tip: str = remote_tip,
            immutable_releases: bool = True,
            repository_id: int = cfg["github_repository_id"],
        ) -> str:
            observed_git: list[tuple[str, ...]] = []

            def fake_git(args: list[str], *, text: bool = True) -> str:
                del text
                key = tuple(args)
                observed_git.append(key)
                values = {
                    ("rev-parse", "--abbrev-ref", "HEAD"): local_branch,
                    ("remote", "get-url", cfg["remote_name"]): remote_url,
                    (
                        "rev-parse",
                        "--abbrev-ref",
                        "--symbolic-full-name",
                        "@{u}",
                    ): cfg["upstream_ref"],
                    ("rev-parse", cfg["upstream_ref"]): remote_tip,
                }
                if args[0] == "fetch":
                    return ""
                if key not in values:
                    raise AssertionError(f"unexpected git call: {args}")
                return values[key]

            def fake_api(
                endpoint: str, context: str, *, reg: dict | None = None
            ) -> dict:
                del context, reg
                if endpoint.endswith("/immutable-releases"):
                    return {"enabled": immutable_releases, "enforced_by_owner": False}
                if endpoint == f"repos/{cfg['github_repository']}":
                    return {
                        "id": repository_id,
                        "full_name": cfg["github_repository"],
                        "private": True,
                    }
                return {
                    "name": cfg["branch"],
                    "protected": False,
                    "commit": {"sha": api_tip},
                }

            pool._REMOTE_TIP_CACHE.clear()
            with (
                patch.object(pool, "_git", side_effect=fake_git),
                patch.object(pool, "github_api_json", side_effect=fake_api),
            ):
                result = pool.canonical_remote_tip(self.reg, refresh=True)
            self.assertIn(
                (
                    "fetch",
                    "--no-tags",
                    cfg["remote_name"],
                    "+refs/heads/exp/step3-response-quality-persistence:refs/remotes/origin/exp/step3-response-quality-persistence",
                ),
                observed_git,
            )
            return result

        self.assertEqual(invoke(), remote_tip)
        for kwargs, message in (
            ({"local_branch": "main"}, "local branch"),
            ({"remote_url": "https://example.invalid/repo.git"}, "remote URL"),
            ({"api_tip": "b" * 40}, "GitHub branch state"),
            ({"repository_id": 1}, "repository identity"),
            ({"immutable_releases": False}, "enable immutable releases"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    invoke(**kwargs)

    def test_remote_lock_stage_and_immutable_release_are_commit_bound(self) -> None:
        commit = "a" * 40
        marker = pool.REMOTE_LOCK_MARKERS["r0-request"]
        tag = pool.remote_lock_tag(self.reg, "r0-request")
        release = {
            "id": 123,
            "tag_name": tag,
            "target_commitish": commit,
            "immutable": True,
            "draft": False,
            "prerelease": True,
            "published_at": "2026-07-26T00:00:00Z",
        }

        def fake_api(endpoint: str, context: str, *, reg: dict | None = None) -> dict:
            del context, reg
            self.assertTrue(endpoint.endswith(tag))
            return {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "commit", "sha": commit},
            }

        with (
            patch.object(
                pool,
                "commit_changed_paths",
                return_value={pool.repo_path(marker)},
            ),
            patch.object(pool, "github_api_json", side_effect=fake_api),
            patch.object(
                pool,
                "verify_release_attestation",
                return_value={"verification_output_sha256": "f" * 64},
            ),
        ):
            self.assertEqual(pool.remote_lock_stage(commit), "r0-request")
            result = pool.validate_immutable_release_lock(
                self.reg, commit, "r0-request", release
            )
        self.assertEqual(result["tag"], tag)
        self.assertTrue(result["immutable"])
        self.assertEqual(
            result["release_attestation"]["verification_output_sha256"],
            "f" * 64,
        )

        bad = dict(release)
        bad["immutable"] = False
        with self.assertRaisesRegex(ValueError, "immutable release lock mismatch"):
            pool.validate_immutable_release_lock(
                self.reg, commit, "r0-request", bad
            )

        with patch.object(
            pool,
            "commit_changed_paths",
            return_value={
                pool.repo_path(pool.REMOTE_LOCK_MARKERS["r0-request"]),
                pool.repo_path(pool.REMOTE_LOCK_MARKERS["r1-pulse"]),
            },
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                pool.remote_lock_stage(commit)

    def test_github_release_creation_is_authenticated_and_exact(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["method"] = request.method
            observed["url"] = request.full_url
            observed["body"] = json.loads(request.data.decode("utf-8"))
            observed["content_type"] = request.headers["Content-type"]
            observed["timeout"] = timeout
            return FakeResponse({"id": 1})

        payload = {"tag_name": "fixed", "target_commitish": "a" * 40}
        with patch.dict(os.environ, {"STEP3O_O4_GITHUB_TOKEN": "test-token"}):
            original = pool.urllib.request.urlopen
            try:
                pool.urllib.request.urlopen = fake_urlopen
                result = pool.github_api_mutation_json(
                    "POST",
                    "repos/0bCyan/lora-safety-spectral-geometry/releases",
                    "synthetic immutable release",
                    payload,
                    reg=self.reg,
                )
            finally:
                pool.urllib.request.urlopen = original
        self.assertEqual(result, {"id": 1})
        self.assertEqual(observed["method"], "POST")
        self.assertEqual(observed["body"], payload)
        self.assertEqual(observed["content_type"], "application/json")
        self.assertEqual(observed["timeout"], 30)

    def test_release_attestation_uses_pinned_gh_verifier(self) -> None:
        cfg = self.reg["repository_boundary"]
        tag = pool.remote_lock_tag(self.reg, "p0")
        commit = "a" * 40
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if args == ["gh", "--version"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout=f"gh version {cfg['gh_cli_version']} (test)\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=b'{"verification":"success"}\n',
                stderr=b"",
            )

        with (
            patch.dict(os.environ, {"STEP3O_O4_GITHUB_TOKEN": "test-token"}),
            patch.object(pool.subprocess, "run", side_effect=fake_run),
        ):
            result = pool.verify_release_attestation(self.reg, tag, commit)
        self.assertEqual(result["repository_id"], cfg["github_repository_id"])
        self.assertEqual(result["tag"], tag)
        self.assertEqual(result["commit"], commit)
        verify_args = calls[1][0]
        self.assertEqual(
            verify_args,
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
        )
        self.assertEqual(calls[1][1]["env"]["GH_TOKEN"], "test-token")

    def test_lock_head_recovers_only_an_exact_existing_release(self) -> None:
        commit = "a" * 40
        exact = {"id": 7}
        verified = {"stage": "p0", "commit": commit}
        with (
            patch.object(pool, "registry", return_value=self.reg),
            patch.object(pool, "validate_boundaries"),
            patch.object(pool, "git_head", return_value=commit),
            patch.object(pool, "canonical_remote_tip", return_value=commit),
            patch.object(pool, "remote_lock_stage", return_value="p0"),
            patch.object(pool, "github_api_optional_json", return_value=exact),
            patch.object(
                pool,
                "validate_immutable_release_lock",
                return_value=verified,
            ) as validate,
            patch.object(pool, "github_api_mutation_json") as mutate,
        ):
            result = pool.create_remote_lock()
        validate.assert_called_once_with(self.reg, commit, "p0", exact)
        mutate.assert_not_called()
        self.assertTrue(result["recovered_existing_release"])

    def test_pulse_randomness_is_bound_to_signature(self) -> None:
        self.assertEqual(pool.validate_drand_pulse(self.pulse), self.pulse)
        bad = copy.deepcopy(self.pulse)
        bad["randomness"] = "f" * 64
        with self.assertRaises(ValueError):
            pool.validate_drand_pulse(bad)

    def test_pinned_official_verifier_cannot_disable_bls_checks(self) -> None:
        package = json.loads(pool.DRAND_VERIFIER_PACKAGE.read_text(encoding="utf-8"))
        lock = json.loads(pool.DRAND_VERIFIER_LOCK.read_text(encoding="utf-8"))
        script = pool.DRAND_VERIFIER_SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(package["dependencies"], {"drand-client": "1.4.2"})
        self.assertEqual(lock["packages"]["node_modules/drand-client"]["version"], "1.4.2")
        self.assertIn("disableBeaconVerification: false", script)
        self.assertIn("chainVerificationParams", script)

    def test_creation_blob_cannot_be_changed_by_a_later_commit(self) -> None:
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                result = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
                return result.stdout.strip()

            git("init")
            git("config", "user.email", "o4-test@example.invalid")
            git("config", "user.name", "O4 Test")
            git("config", "core.autocrlf", "false")
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            git("add", "base.txt")
            git("commit", "-m", "base")
            parent = git("rev-parse", "HEAD")
            request_path = root / "request.json"
            request_path.write_text('{"round": 1}\n', encoding="utf-8")
            git("add", "request.json")
            git("commit", "-m", "request")
            creation = git("rev-parse", "HEAD")
            with patch.object(pool, "REPO_ROOT", root):
                self.assertEqual(
                    pool.verify_exact_creation_commit(
                        [request_path],
                        request_path,
                        parent,
                        reg=self.reg,
                        require_pushed=False,
                    ),
                    creation,
                )
            request_path.write_text('{"round": 2}\n', encoding="utf-8")
            git("add", "request.json")
            git("commit", "-m", "rewrite request")
            with patch.object(pool, "REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "committed blob|touched after"):
                    pool.verify_exact_creation_commit(
                        [request_path],
                        request_path,
                        parent,
                        reg=self.reg,
                        require_pushed=False,
                    )

    def test_r0_and_r1_reject_extra_paths_and_request_rewrites(self) -> None:
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)

        def git(root: Path, *args: str) -> str:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            return result.stdout.strip()

        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            git(root, "init")
            git(root, "config", "user.email", "o4-test@example.invalid")
            git(root, "config", "user.name", "O4 Test")
            git(root, "config", "core.autocrlf", "false")
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            git(root, "add", "base.txt")
            git(root, "commit", "-m", "base")
            parent = git(root, "rev-parse", "HEAD")
            request = root / "request.json"
            extra = root / "extra.txt"
            request.write_text('{"round": 1}\n', encoding="utf-8")
            extra.write_text("smuggled\n", encoding="utf-8")
            git(root, "add", "request.json", "extra.txt")
            git(root, "commit", "-m", "invalid R0")
            with patch.object(pool, "REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "unexpected path set"):
                    pool.verify_exact_creation_commit(
                        [request],
                        request,
                        parent,
                        reg=self.reg,
                        require_pushed=False,
                    )

        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            git(root, "init")
            git(root, "config", "user.email", "o4-test@example.invalid")
            git(root, "config", "user.name", "O4 Test")
            git(root, "config", "core.autocrlf", "false")
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            git(root, "add", "base.txt")
            git(root, "commit", "-m", "base")
            request = root / "request.json"
            request.write_text('{"round": 1}\n', encoding="utf-8")
            git(root, "add", "request.json")
            git(root, "commit", "-m", "valid R0")
            request_commit = git(root, "rev-parse", "HEAD")
            pulse = root / "pulse.json"
            request.write_text('{"round": 2}\n', encoding="utf-8")
            pulse.write_text('{"round": 2}\n', encoding="utf-8")
            git(root, "add", "request.json", "pulse.json")
            git(root, "commit", "-m", "invalid R1 rewrite")
            with patch.object(pool, "REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "unexpected path set"):
                    pool.verify_exact_creation_commit(
                        [pulse],
                        pulse,
                        request_commit,
                        reg=self.reg,
                        require_pushed=False,
                    )

    def test_request_selects_a_future_unpublished_round(self) -> None:
        request = beacon.build_request_artifact(
            self.reg, self.protocol_commit, self.info, self.pulse
        )
        offset = self.reg["randomness_beacon"]["target_round_offset"]
        self.assertEqual(request["target_round"], self.pulse["round"] + offset)
        self.assertGreater(
            request["target_round_time_unix"], request["observed_round_time_unix"]
        )
        pool.validate_randomness_request(
            request, self.reg, self.protocol_commit
        )

    def test_request_rejects_wrong_protocol_commit(self) -> None:
        request = beacon.build_request_artifact(
            self.reg, self.protocol_commit, self.info, self.pulse
        )
        with self.assertRaises(ValueError):
            pool.validate_randomness_request(request, self.reg, "f" * 40)

    def test_live_drand_chain_must_match_both_frozen_rounds(self) -> None:
        request = beacon.build_request_artifact(
            self.reg, self.protocol_commit, self.info, self.pulse
        )
        target_signature = "22" * 96
        target = {
            "round": request["target_round"],
            "randomness": hashlib.sha256(bytes.fromhex(target_signature)).hexdigest(),
            "signature": target_signature,
            "previous_signature": "33" * 96,
        }
        pulse = {"target_pulse": target}
        cfg = self.reg["randomness_beacon"]
        values = {cfg["info_url"]: copy.deepcopy(self.info)}
        original = pool.fetch_public_json
        original_verify = pool.verify_drand_pulse_cryptographically
        try:
            pool.fetch_public_json = lambda url: copy.deepcopy(values[url])
            pool.verify_drand_pulse_cryptographically = (
                lambda _reg, pulse: copy.deepcopy(pulse)
            )
            pool.validate_live_drand_chain(self.reg, request, pulse)
            pool.verify_drand_pulse_cryptographically = (
                lambda _reg, _pulse: (_ for _ in ()).throw(ValueError("invalid proof"))
            )
            with self.assertRaises(ValueError):
                pool.validate_live_drand_chain(self.reg, request, pulse)
        finally:
            pool.fetch_public_json = original
            pool.verify_drand_pulse_cryptographically = original_verify

    def test_salt_depends_on_public_pulse(self) -> None:
        first = {"beacon_round": 100, "beacon_randomness": self.pulse["randomness"]}
        second = {"beacon_round": 101, "beacon_randomness": self.pulse["randomness"]}
        self.assertNotEqual(
            pool.derive_salt(self.reg, self.protocol_commit, first, "prompt_matching"),
            pool.derive_salt(self.reg, self.protocol_commit, second, "prompt_matching"),
        )

    def test_pulse_artifact_binds_request_hash_and_round(self) -> None:
        request = beacon.build_request_artifact(
            self.reg, self.protocol_commit, self.info, self.pulse
        )
        target_signature = "22" * 96
        target = {
            "round": request["target_round"],
            "randomness": hashlib.sha256(bytes.fromhex(target_signature)).hexdigest(),
            "signature": target_signature,
            "previous_signature": "33" * 96,
        }
        try:
            run_id = 2
            job_id = 3
            completed = datetime.fromtimestamp(
                request["target_round_time_unix"] - 1, tz=timezone.utc
            )
            started = datetime.fromtimestamp(
                request["target_round_time_unix"] - 2, tz=timezone.utc
            )
            job_url = (
                "https://github.com/0bCyan/lora-safety-spectral-geometry/"
                f"actions/runs/{run_id}/job/{job_id}"
            )
            request_ci = {
                "id": 1,
                "name": self.reg["randomness_beacon"]["request_ci_check_name"],
                "head_sha": "b" * 40,
                "status": "completed",
                "conclusion": "success",
                "html_url": job_url,
                "details_url": job_url,
                "started_at": started.isoformat().replace("+00:00", "Z"),
                "completed_at": completed.isoformat().replace("+00:00", "Z"),
                "app_slug": "github-actions",
                "check_suite_id": 4,
                "workflow_run_id": run_id,
                "workflow_run_attempt": 1,
                "workflow_path": self.reg["randomness_beacon"]["request_ci_workflow"],
                "event": "push",
                "job_id": job_id,
            }
            artifact = beacon.build_pulse_artifact(
                self.reg,
                self.protocol_commit,
                "b" * 40,
                "a" * 64,
                request,
                target,
                request_ci,
            )
            spoofed = copy.deepcopy(request_ci)
            spoofed["app_slug"] = "third-party"
            with self.assertRaises(ValueError):
                beacon.build_pulse_artifact(
                    self.reg,
                    self.protocol_commit,
                    "b" * 40,
                    "a" * 64,
                    request,
                    target,
                    spoofed,
                )
            rerun = copy.deepcopy(request_ci)
            rerun["workflow_run_attempt"] = 2
            with self.assertRaises(ValueError):
                beacon.build_pulse_artifact(
                    self.reg,
                    self.protocol_commit,
                    "b" * 40,
                    "a" * 64,
                    request,
                    target,
                    rerun,
                )
            late = copy.deepcopy(request_ci)
            late["completed_at"] = datetime.fromtimestamp(
                request["target_round_time_unix"], tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")
            with self.assertRaises(ValueError):
                beacon.build_pulse_artifact(
                    self.reg,
                    self.protocol_commit,
                    "b" * 40,
                    "a" * 64,
                    request,
                    target,
                    late,
                )
        finally:
            pass
        self.assertEqual(artifact["target_round"], request["target_round"])
        self.assertEqual(artifact["request_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
