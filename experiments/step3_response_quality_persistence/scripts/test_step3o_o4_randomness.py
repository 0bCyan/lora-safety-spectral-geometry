#!/usr/bin/env python3
"""Tests for the Step3O-O4 prospective public-randomness freeze."""

from __future__ import annotations

import copy
import hashlib
import http.server
import json
import os
import subprocess
import threading
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

    def test_drand_mirror_order_and_transport_policy_are_frozen(self) -> None:
        cfg = self.reg["randomness_beacon"]
        self.assertEqual(
            cfg["mirror_base_urls"],
            [
                "https://drand.cloudflare.com",
                "https://api2.drand.sh",
                "https://api3.drand.sh",
                "https://api.drand.sh",
            ],
        )
        self.assertEqual(pool.validate_drand_transport_config(self.reg), cfg)
        changed = copy.deepcopy(self.reg)
        changed["randomness_beacon"]["mirror_base_urls"].reverse()
        with self.assertRaisesRegex(ValueError, "transport boundary"):
            pool.validate_drand_transport_config(changed)

    def test_drand_fetch_fails_over_once_in_registered_order(self) -> None:
        calls: list[tuple[str, int]] = []
        trace: list[dict] = []

        def fake_fetch(url: str, *, timeout_seconds: int) -> dict:
            calls.append((url, timeout_seconds))
            if len(calls) == 1:
                raise pool.PublicEvidenceTransportError("synthetic timeout")
            return copy.deepcopy(self.info)

        with patch.object(pool, "fetch_public_json", side_effect=fake_fetch):
            value = pool.fetch_drand_json(
                self.reg,
                "info",
                validator=lambda item: pool.validate_drand_info(item, self.reg),
                trace=trace,
                trace_operation="test_info",
            )
        self.assertEqual(value, self.info)
        self.assertEqual(
            [url for url, _ in calls],
            [
                "https://drand.cloudflare.com/info",
                "https://api2.drand.sh/info",
            ],
        )
        self.assertEqual(
            {timeout for _, timeout in calls},
            {self.reg["randomness_beacon"]["per_mirror_timeout_seconds"]},
        )
        self.assertEqual(
            [event["outcome"] for event in trace],
            ["TRANSPORT_FAILURE", "ACCEPTED_CONTENT"],
        )
        self.assertEqual([event["mirror_index"] for event in trace], [1, 2])

    def test_public_fetch_uses_process_deadline_and_response_limit(self) -> None:
        timeout = self.reg["randomness_beacon"]["per_mirror_timeout_seconds"]
        with patch.object(
            pool.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["python"], timeout),
        ):
            with self.assertRaisesRegex(
                pool.PublicEvidenceTransportError, "wall-clock deadline"
            ):
                pool.fetch_public_json("https://example.invalid/info", timeout_seconds=timeout)

        oversized = subprocess.CompletedProcess(
            ["python"],
            0,
            stdout=b"x" * (pool.DRAND_MAXIMUM_RESPONSE_BYTES + 1),
            stderr=b"",
        )
        with patch.object(pool.subprocess, "run", return_value=oversized):
            with self.assertRaisesRegex(
                pool.PublicEvidenceTransportError, "response byte limit"
            ):
                pool.fetch_public_json("https://example.invalid/info", timeout_seconds=timeout)

        observed: dict[str, object] = {}

        def success(args, **kwargs):
            observed["args"] = args
            observed["timeout"] = kwargs["timeout"]
            return subprocess.CompletedProcess(
                args, 0, stdout=b'{"ok":true}', stderr=b""
            )

        with patch.object(pool.subprocess, "run", side_effect=success):
            self.assertEqual(
                pool.fetch_public_json(
                    "https://example.invalid/info", timeout_seconds=timeout
                ),
                {"ok": True},
            )
        self.assertEqual(observed["timeout"], timeout)
        self.assertIn(str(pool.DRAND_MAXIMUM_RESPONSE_BYTES), observed["args"])

    def test_drand_fetch_stops_after_all_registered_transport_failures(self) -> None:
        calls: list[str] = []

        def fake_fetch(url: str, *, timeout_seconds: int) -> dict:
            del timeout_seconds
            calls.append(url)
            raise pool.PublicEvidenceTransportError("synthetic transport failure")

        with patch.object(pool, "fetch_public_json", side_effect=fake_fetch):
            with self.assertRaisesRegex(
                pool.PublicEvidenceTransportError, "all registered drand mirrors"
            ):
                pool.fetch_drand_json(self.reg, "latest")
        self.assertEqual(
            calls,
            [f"{base}/public/latest" for base in pool.DRAND_MIRROR_BASE_URLS],
        )

    def test_returned_drand_chain_mismatch_is_terminal_without_failover(self) -> None:
        calls: list[str] = []
        wrong = copy.deepcopy(self.info)
        wrong["hash"] = "f" * 64

        def fake_fetch(url: str, *, timeout_seconds: int) -> dict:
            del timeout_seconds
            calls.append(url)
            return copy.deepcopy(wrong)

        with patch.object(pool, "fetch_public_json", side_effect=fake_fetch):
            with self.assertRaisesRegex(ValueError, "differs from the P0 registry"):
                pool.fetch_drand_json(
                    self.reg,
                    "info",
                    validator=lambda item: pool.validate_drand_info(item, self.reg),
                )
        self.assertEqual(calls, ["https://drand.cloudflare.com/info"])

    def test_request_workflow_is_push_and_path_bound(self) -> None:
        workflow = pool.RANDOMNESS_WORKFLOW.read_text(encoding="utf-8")
        artifact = self.reg["randomness_beacon"]["request_artifact"]
        self.assertIn('"on":\n  push:', workflow)
        self.assertIn("- exp/step3o-o4-attempt3", workflow)
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
        self.assertEqual(
            observed["timeout"],
            self.reg["repository_boundary"]["github_read_timeout_seconds"],
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "STEP3O_O4_GITHUB_TOKEN"):
                pool.github_api_json(
                    "repos/0bCyan/lora-safety-spectral-geometry/commits/abc/check-runs",
                    "synthetic GitHub API",
                    reg=self.reg,
                )

    def test_github_read_retries_only_transport_failures_in_fixed_order(self) -> None:
        calls = 0

        def flaky_urlopen(_request, timeout):
            nonlocal calls
            calls += 1
            self.assertEqual(
                timeout,
                self.reg["repository_boundary"]["github_read_timeout_seconds"],
            )
            if calls < 4:
                raise pool.urllib.error.URLError("synthetic proxy EOF")
            return FakeResponse({"ok": True})

        with (
            patch.dict(os.environ, {"STEP3O_O4_GITHUB_TOKEN": "test-token"}),
            patch.object(pool.urllib.request, "urlopen", side_effect=flaky_urlopen),
            patch.object(pool.time, "sleep") as sleep,
        ):
            value = pool.github_api_json(
                "repos/0bCyan/lora-safety-spectral-geometry",
                "synthetic retry API",
                reg=self.reg,
            )
        self.assertEqual(value, {"ok": True})
        self.assertEqual(calls, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            self.reg["repository_boundary"]["github_read_backoff_seconds"],
        )

        http_error = pool.urllib.error.HTTPError(
            "https://api.github.com/repos/example",
            500,
            "synthetic server response",
            None,
            None,
        )
        with (
            patch.dict(os.environ, {"STEP3O_O4_GITHUB_TOKEN": "test-token"}),
            patch.object(
                pool.urllib.request, "urlopen", side_effect=http_error
            ) as urlopen,
            patch.object(pool.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(ValueError, "failed to query authenticated"):
                pool.github_api_json(
                    "repos/0bCyan/lora-safety-spectral-geometry",
                    "synthetic HTTP failure",
                    reg=self.reg,
                )
        urlopen.assert_called_once()
        sleep.assert_not_called()

        class ReadFailureResponse(FakeResponse):
            def read(self) -> bytes:
                raise OSError("synthetic body read failure")

        with (
            patch.dict(os.environ, {"STEP3O_O4_GITHUB_TOKEN": "test-token"}),
            patch.object(
                pool.urllib.request,
                "urlopen",
                return_value=ReadFailureResponse({"unused": True}),
            ) as urlopen,
            patch.object(pool.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(ValueError, "after HTTP response"):
                pool.github_api_json(
                    "repos/0bCyan/lora-safety-spectral-geometry",
                    "synthetic response-read failure",
                    reg=self.reg,
                )
        urlopen.assert_called_once()
        sleep.assert_not_called()

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
                    "+refs/heads/exp/step3o-o4-attempt3:refs/remotes/origin/exp/step3o-o4-attempt3",
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
        marker = pool.REMOTE_LOCK_MARKERS["r0-outcome"]
        tag = pool.remote_lock_tag(self.reg, "r0-outcome")
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
            self.assertEqual(pool.remote_lock_stage(commit), "r0-outcome")
            result = pool.validate_immutable_release_lock(
                self.reg, commit, "r0-outcome", release
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
                self.reg, commit, "r0-outcome", bad
            )

        with patch.object(
            pool,
            "commit_changed_paths",
            return_value={
                pool.repo_path(pool.REMOTE_LOCK_MARKERS["r0-outcome"]),
                pool.repo_path(pool.REMOTE_LOCK_MARKERS["r1-outcome"]),
            },
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                pool.remote_lock_stage(commit)

        terminal = {
            "name": "step3o_o4_randomness_terminal_stop_v1",
            "stage": "R0_REQUEST",
        }
        with (
            patch.object(
                pool,
                "commit_changed_paths",
                return_value={pool.repo_path(pool.ATTEMPT_TERMINAL_STOP)},
            ),
            patch.object(
                pool,
                "git_blob_bytes",
                return_value=pool.json_text(terminal).encode("utf-8"),
            ),
        ):
            self.assertEqual(pool.remote_lock_stage(commit), "r0-outcome")

    def test_stage_opening_requires_fresh_exact_canonical_tip(self) -> None:
        expected = "a" * 40
        with (
            patch.object(pool, "git_head", return_value=expected),
            patch.object(pool, "canonical_remote_tip", return_value="b" * 40) as tip,
        ):
            with self.assertRaisesRegex(ValueError, "canonical remote tip"):
                pool.require_locked_canonical_tip(self.reg, expected)
        tip.assert_called_once_with(self.reg, refresh=True)

        lock = {"commit": expected, "immutable": True}
        with (
            patch.object(pool, "git_head", return_value=expected),
            patch.object(pool, "canonical_remote_tip", return_value=expected),
            patch.object(pool, "verify_remote_lock", return_value=lock),
        ):
            self.assertEqual(
                pool.require_locked_canonical_tip(self.reg, expected), lock
            )

    def test_prepublication_locks_reject_target_time_or_later(self) -> None:
        deadline = self.reg["randomness_beacon"]["fixed_target_round_time_unix"]
        at_deadline = datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        with patch.object(
            pool,
            "verify_remote_lock",
            return_value={"published_at": at_deadline},
        ):
            with self.assertRaisesRegex(ValueError, "after the target round"):
                pool.require_lock_before_unix(
                    self.reg, "a" * 40, deadline, "R1 start"
                )

        before = datetime.fromtimestamp(deadline - 1, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        evidence = {"published_at": before}
        with patch.object(pool, "verify_remote_lock", return_value=evidence):
            self.assertEqual(
                pool.require_lock_before_unix(
                    self.reg, "a" * 40, deadline, "R1 start"
                ),
                evidence,
            )

    def test_late_r0_start_is_rejected_before_any_drand_fetch(self) -> None:
        protocol_commit = "a" * 40
        start_commit = "b" * 40
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            start = root / "start.json"
            start.write_text("{}\n", encoding="utf-8", newline="\n")
            with (
                patch.object(pool, "R0_START", start),
                patch.object(pool, "registry", return_value=copy.deepcopy(self.reg)),
                patch.object(pool, "validate_boundaries"),
                patch.object(
                    pool, "resolve_protocol_commit", return_value=protocol_commit
                ),
                patch.object(pool, "require_protocol_inputs_frozen"),
                patch.object(
                    pool,
                    "verify_randomness_start_commit",
                    return_value=(start_commit, b"{}\n"),
                ),
                patch.object(pool, "git_head", return_value=start_commit),
                patch.object(pool, "protocol_artifact_files", return_value=[start]),
                patch.object(
                    pool,
                    "require_lock_before_unix",
                    side_effect=[
                        {"published_at": "2026-07-26T05:59:58Z"},
                        ValueError("R0 start immutable lock was published after the target round"),
                    ],
                ) as lock_gate,
                patch.object(beacon, "fetch_json") as fetch,
                patch.object(beacon, "write_randomness_stop"),
            ):
                with self.assertRaisesRegex(ValueError, "after the target round"):
                    beacon._create_request_once()
            self.assertEqual(lock_gate.call_count, 2)
            fetch.assert_not_called()

    def test_late_r0_terminal_outcome_is_rejected(self) -> None:
        protocol_commit = "a" * 40
        start_commit = "b" * 40
        stop_commit = "c" * 40
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            start = root / "start.json"
            trace = root / "trace.json"
            terminal = root / "terminal.json"
            for path in (start, trace):
                path.write_text("{}\n", encoding="utf-8", newline="\n")
            value = {
                "name": "step3o_o4_randomness_terminal_stop_v1",
                "attempt_id": self.reg["attempt"]["attempt_id"],
                "stage": "R0_REQUEST",
                "route": "STOP_PROTOCOL_INTEGRITY",
                "protocol_commit": protocol_commit,
                "parent_commit": start_commit,
                "error_type": "RuntimeError",
                "error_message_sha256": "d" * 64,
                "evidence_artifacts": [
                    pool.artifact_from_bytes(start, start.read_bytes()),
                    pool.artifact_from_bytes(trace, trace.read_bytes()),
                ],
                "replacement_or_retry_allowed": False,
                "optimizer_steps": 0,
                "training_authorized": False,
            }
            terminal.write_text(
                pool.json_text(value), encoding="utf-8", newline="\n"
            )
            with (
                patch.object(pool, "ATTEMPT_TERMINAL_STOP", terminal),
                patch.object(pool, "registry", return_value=copy.deepcopy(self.reg)),
                patch.object(pool, "validate_boundaries"),
                patch.object(
                    pool, "resolve_protocol_commit", return_value=protocol_commit
                ),
                patch.object(
                    beacon,
                    "verify_request_start",
                    return_value={"start_commit": start_commit},
                ),
                patch.object(
                    beacon,
                    "RANDOMNESS_STAGE_PATHS",
                    {"R0_REQUEST": (start, trace, root / "request.json")},
                ),
                patch.object(
                    beacon, "RANDOMNESS_START_PATHS", {"R0_REQUEST": start}
                ),
                patch.object(
                    beacon, "RANDOMNESS_TRACE_PATHS", {"R0_REQUEST": trace}
                ),
                patch.object(pool, "validate_drand_transport_trace"),
                patch.object(
                    pool,
                    "verify_exact_creation_commit",
                    return_value=stop_commit,
                ),
                patch.object(
                    pool,
                    "require_lock_before_unix",
                    side_effect=[
                        {"published_at": "2026-07-26T05:59:57Z"},
                        {"published_at": "2026-07-26T05:59:58Z"},
                        ValueError("R0 outcome immutable lock was published after the target round"),
                    ],
                ) as lock_gate,
            ):
                with self.assertRaisesRegex(ValueError, "after the target round"):
                    beacon.verify_randomness_stop(require_pushed=True)
            self.assertEqual(lock_gate.call_count, 3)

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
        policy = pool.DRAND_VERIFIER_RETRY_POLICY.read_text(encoding="utf-8")
        self.assertEqual(package["dependencies"], {"drand-client": "1.4.2"})
        self.assertEqual(lock["packages"]["node_modules/drand-client"]["version"], "1.4.2")
        self.assertIn("disableBeaconVerification: false", script)
        self.assertIn("chainVerificationParams", script)
        self.assertIn("isRetryableRemoteFailure", script)
        self.assertIn('message === "fetch failed"', policy)
        self.assertIn("drand cryptographic verification failed", script)

    def test_node_retry_policy_executes_fail_closed_classification(self) -> None:
        module_uri = pool.DRAND_VERIFIER_RETRY_POLICY.as_uri()
        source = (
            f'import {{ isRetryableRemoteFailure }} from {json.dumps(module_uri)}; '
            "process.exit(isRetryableRemoteFailure({message: process.argv[1]}) ? 75 : 1);"
        )
        cases = {
            "fetch failed": 75,
            "Error response fetching https://api.drand.sh/info - got 503": 75,
            "The beacon retrieved was not valid!": 1,
            "The chain info retrieved from an endpoint did not match the verification params!": 1,
            "Unexpected token '<' while parsing JSON": 1,
        }
        for message, expected_code in cases.items():
            with self.subTest(message=message):
                result = subprocess.run(
                    ["node", "--input-type=module", "-e", source, message],
                    cwd=pool.DRAND_VERIFIER_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, expected_code, result.stderr)

    def test_real_node_verifier_classifies_controlled_http_boundaries(self) -> None:
        info = {
            "genesis_time": 1595431050,
            "groupHash": "176f93498eac9ca337150b46d21dd58673ea4e3581185f869672e59fa4cb390a",
            "hash": self.reg["randomness_beacon"]["chain_hash"],
            "metadata": {"beaconID": "default"},
            "period": 30,
            "public_key": self.reg["randomness_beacon"]["public_key"],
            "schemeID": "pedersen-bls-chained",
        }
        pulse = {
            "previous_signature": "976efbf800823148417ec4e2385ed1db09d640dcd07c56d10829954b791ef8491dfce68468e430d14238b3dc68199b6b09db106e92d1a93cc86754d103366d7f8d250ea0f3d033522542604fb9907b3eae94c6cf6fe8af3ea2e4f9a115f48185",
            "randomness": "7c866abe75c92986e4fde6be622d7bab0c1198e14ac6bd6de33309a589da5405",
            "round": 6320122,
            "signature": "9388410454c16a018528f898b0bdc85601db9e0afe01088ddcf35cfb7522b362da32e8fc4bcc9cb413540366c3d328af125844be7f2ef8ab9856c353627a47b30ee940922975ff2bd3c2cecb98bc36976aab1786799eec139801550211d47449",
        }
        state = {"mode": "valid", "pulse": pulse}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                return None

            def do_GET(self) -> None:
                if state["mode"] == "http_error":
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if state["mode"] == "malformed_json":
                    self.wfile.write(b"{")
                elif self.path.startswith("/info"):
                    value = copy.deepcopy(info)
                    if state["mode"] == "chain_mismatch":
                        value["hash"] = "f" * 64
                    self.wfile.write(json.dumps(value).encode("utf-8"))
                else:
                    self.wfile.write(json.dumps(state["pulse"]).encode("utf-8"))

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def invoke(mode: str, expected: dict) -> subprocess.CompletedProcess:
                state["mode"] = mode
                state["pulse"] = expected
                with TemporaryDirectory() as directory:
                    input_path = Path(directory) / "input.json"
                    input_path.write_text(
                        pool.json_text(
                            {
                                "base_url": f"http://127.0.0.1:{server.server_port}",
                                "chain_hash": info["hash"],
                                "expected_pulse": expected,
                                "public_key": info["public_key"],
                            }
                        ),
                        encoding="utf-8",
                        newline="\n",
                    )
                    return subprocess.run(
                        ["node", str(pool.DRAND_VERIFIER_SCRIPT), str(input_path)],
                        cwd=pool.DRAND_VERIFIER_ROOT,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                        timeout=15,
                    )

            valid = invoke("valid", pulse)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout), pulse)
            self.assertEqual(invoke("http_error", pulse).returncode, 75)
            self.assertEqual(invoke("malformed_json", pulse).returncode, 1)
            self.assertEqual(invoke("chain_mismatch", pulse).returncode, 1)
            invalid = copy.deepcopy(pulse)
            invalid["signature"] = "00" * 96
            invalid["randomness"] = hashlib.sha256(bytes.fromhex(invalid["signature"])).hexdigest()
            self.assertEqual(invoke("valid", invalid).returncode, 1)
            wrong_round = copy.deepcopy(pulse)
            wrong_round["round"] += 1
            self.assertEqual(invoke("valid", wrong_round).returncode, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_r0_failure_uses_precommitted_start_and_freezes_trace_stop(self) -> None:
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            base = Path(directory)
            reports = base / "reports"
            start = reports / "start.json"
            trace = reports / "trace.json"
            output = reports / "request.json"
            terminal = reports / "terminal.json"
            stage_paths = {"R0_REQUEST": (start, trace, output)}
            start_paths = {"R0_REQUEST": start}
            trace_paths = {"R0_REQUEST": trace}
            output_paths = {"R0_REQUEST": output}
            start_commit = "a" * 40
            with (
                patch.object(pool, "BASE", base),
                patch.object(pool, "R0_START", start),
                patch.object(pool, "R0_TRACE", trace),
                patch.object(pool, "BEACON_REQUEST", output),
                patch.object(pool, "ATTEMPT_TERMINAL_STOP", terminal),
                patch.object(pool, "git_head", return_value=start_commit),
                patch.object(pool, "require_locked_canonical_tip"),
                patch.object(beacon, "RANDOMNESS_STAGE_PATHS", stage_paths),
                patch.object(beacon, "RANDOMNESS_START_PATHS", start_paths),
                patch.object(beacon, "RANDOMNESS_TRACE_PATHS", trace_paths),
                patch.object(beacon, "RANDOMNESS_OUTPUT_PATHS", output_paths),
                patch.object(beacon, "verify_request_start") as verify_start,
                patch.object(
                    beacon,
                    "fetch_json",
                    side_effect=pool.PublicEvidenceTransportError("synthetic timeout"),
                ) as fetch,
            ):
                start_value = beacon.build_start_receipt(
                    self.reg,
                    stage="R0_REQUEST",
                    protocol_commit=self.protocol_commit,
                    parent_commit=self.protocol_commit,
                )
                pool.write_frozen(start, pool.json_text(start_value))
                verify_start.return_value = {
                    "reg": self.reg,
                    "protocol_commit": self.protocol_commit,
                    "start_commit": start_commit,
                    "start_sha256": pool.sha256(start),
                }
                with self.assertRaises(pool.PublicEvidenceTransportError):
                    beacon._create_request_once()
                self.assertTrue(start.is_file())
                self.assertTrue(trace.is_file())
                self.assertTrue(terminal.is_file())
                self.assertFalse(output.exists())
                with self.assertRaisesRegex(ValueError, "not pristine"):
                    beacon._create_request_once()
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(verify_start.call_count, 3)

    def test_r1_failure_uses_precommitted_start_and_freezes_trace_stop(self) -> None:
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            base = Path(directory)
            reports = base / "reports"
            r0_start = reports / "r0_start.json"
            r0_trace = reports / "r0_trace.json"
            request = reports / "request.json"
            r1_start = reports / "r1_start.json"
            r1_trace = reports / "r1_trace.json"
            pulse = reports / "pulse.json"
            terminal = reports / "terminal.json"
            for path in (r0_start, r0_trace, request):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8", newline="\n")
            request_commit = "b" * 40
            start_commit = "c" * 40
            state = {
                "reg": self.reg,
                "protocol_commit": self.protocol_commit,
                "request_commit": request_commit,
                "request_sha256": "a" * 64,
                "request": {"target_round": 200},
                "pulse_start_commit": start_commit,
            }

            with (
                patch.object(pool, "BASE", base),
                patch.object(pool, "R0_START", r0_start),
                patch.object(pool, "R0_TRACE", r0_trace),
                patch.object(pool, "BEACON_REQUEST", request),
                patch.object(pool, "R1_START", r1_start),
                patch.object(pool, "R1_TRACE", r1_trace),
                patch.object(pool, "BEACON_PULSE", pulse),
                patch.object(pool, "ATTEMPT_TERMINAL_STOP", terminal),
                patch.object(pool, "git_head", return_value=start_commit),
                patch.object(pool, "require_locked_canonical_tip"),
                patch.object(beacon, "verify_pulse_start", return_value=state) as verify_start,
                patch.object(
                    beacon,
                    "verify_request",
                    side_effect=pool.PublicEvidenceTransportError(
                        "synthetic CI query timeout"
                    ),
                ) as verify_request,
                patch.object(
                    beacon,
                    "RANDOMNESS_STAGE_PATHS",
                    {"R1_PULSE": (r1_start, r1_trace, pulse)},
                ),
                patch.object(
                    beacon, "RANDOMNESS_START_PATHS", {"R1_PULSE": r1_start}
                ),
                patch.object(
                    beacon, "RANDOMNESS_TRACE_PATHS", {"R1_PULSE": r1_trace}
                ),
                patch.object(
                    beacon, "RANDOMNESS_OUTPUT_PATHS", {"R1_PULSE": pulse}
                ),
            ):
                start_value = beacon.build_start_receipt(
                    self.reg,
                    stage="R1_PULSE",
                    protocol_commit=self.protocol_commit,
                    parent_commit=request_commit,
                )
                pool.write_frozen(r1_start, pool.json_text(start_value))
                state["pulse_start_sha256"] = pool.sha256(r1_start)
                with self.assertRaises(pool.PublicEvidenceTransportError):
                    beacon._acquire_pulse_once()
                self.assertTrue(r1_start.is_file())
                self.assertTrue(r1_trace.is_file())
                self.assertTrue(terminal.is_file())
                self.assertFalse(pulse.exists())
                with self.assertRaisesRegex(ValueError, "not pristine"):
                    beacon._acquire_pulse_once()
                self.assertEqual(verify_start.call_count, 3)
                self.assertEqual(verify_request.call_count, 1)

    def test_randomness_command_timeout_seals_interrupted_invocation(self) -> None:
        timeout = 7
        class FakeReceive:
            def poll(self) -> bool:
                return False

            def close(self) -> None:
                return None

        class FakeSend:
            def close(self) -> None:
                return None

        class FakeProcess:
            def __init__(self, **_kwargs) -> None:
                self.alive = True
                self.exitcode = None

            def start(self) -> None:
                return None

            def join(self, _timeout=None) -> None:
                return None

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                self.alive = False

            def kill(self) -> None:
                self.alive = False

        class FakeContext:
            def Pipe(self, *, duplex: bool):
                self.duplex = duplex
                return FakeReceive(), FakeSend()

            def Process(self, **kwargs):
                self.process_kwargs = kwargs
                return FakeProcess(**kwargs)

        context = FakeContext()
        with (
            patch.object(beacon.multiprocessing, "get_context", return_value=context),
            patch.object(beacon, "seal_interrupted_stop") as seal,
        ):
            with self.assertRaisesRegex(ValueError, "wall-clock budget"):
                beacon.supervise_once("request", "R0_REQUEST", timeout)
        seal.assert_called_once()
        self.assertEqual(seal.call_args.args[0], "R0_REQUEST")
        self.assertFalse(context.duplex)

    def test_randomness_pipe_failure_seals_interrupted_invocation(self) -> None:
        class FakeReceive:
            def poll(self) -> bool:
                raise BrokenPipeError("synthetic closed child pipe")

            def close(self) -> None:
                return None

        class FakeSend:
            def close(self) -> None:
                return None

        class FakeProcess:
            exitcode = 0

            def start(self) -> None:
                return None

            def join(self, _timeout=None) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        class FakeContext:
            def Pipe(self, *, duplex: bool):
                self.duplex = duplex
                return FakeReceive(), FakeSend()

            def Process(self, **_kwargs):
                return FakeProcess()

        context = FakeContext()
        with (
            patch.object(beacon.multiprocessing, "get_context", return_value=context),
            patch.object(beacon, "seal_interrupted_stop") as seal,
        ):
            with self.assertRaisesRegex(ValueError, "supervised worker failed"):
                beacon.supervise_once("request", "R0_REQUEST", 7)
        seal.assert_called_once()
        self.assertEqual(seal.call_args.args[0], "R0_REQUEST")

    def test_formal_cli_exposes_only_supervised_randomness_workers(self) -> None:
        source = Path(beacon.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"_request-once"', source)
        self.assertNotIn('"_acquire-pulse-once"', source)
        self.assertIn('"start-request"', source)
        self.assertIn('"start-pulse"', source)
        self.assertEqual(
            self.reg["randomness_beacon"]["stage_commit_sequence"],
            ["P0", "R0_START", "R0_REQUEST", "R1_START", "R1_PULSE", "P1"],
        )
        self.assertIs(pool.REMOTE_LOCK_MARKERS["r0-start"], pool.R0_START)
        self.assertIs(pool.REMOTE_LOCK_MARKERS["r1-start"], pool.R1_START)

    def test_complete_trace_is_order_url_failover_and_payload_bound(self) -> None:
        protocol_commit = "a" * 40
        start_commit = "b" * 40
        base = self.reg["randomness_beacon"]["mirror_base_urls"][0]
        info = pool.canonical_drand_info(self.info, self.reg)
        payloads = {"info": info, "observed": self.pulse}
        events = [
            {
                "operation": "r0_fetch_info",
                "mirror_index": 1,
                "url": f"{base}/info",
                "outcome": "ACCEPTED_CONTENT",
                "elapsed_milliseconds": 1,
                "parsed_object_sha256": pool.canonical_json_sha256(info),
                "failure_class": None,
            },
            {
                "operation": "r0_fetch_latest",
                "mirror_index": 1,
                "url": f"{base}/public/latest",
                "outcome": "ACCEPTED_CONTENT",
                "elapsed_milliseconds": 1,
                "parsed_object_sha256": pool.canonical_json_sha256(self.pulse),
                "failure_class": None,
            },
            {
                "operation": "r0_verify_observed",
                "mirror_index": 1,
                "url": base,
                "outcome": "CRYPTOGRAPHICALLY_VERIFIED",
                "elapsed_milliseconds": 1,
                "parsed_object_sha256": pool.canonical_json_sha256(self.pulse),
                "failure_class": None,
            },
        ]

        def validate(candidate: list[dict]) -> dict:
            artifact = {
                "name": "step3o_o4_drand_transport_trace_v1",
                "attempt_id": self.reg["attempt"]["attempt_id"],
                "stage": "R0_REQUEST",
                "status": "COMPLETE_SUCCESS",
                "protocol_commit": protocol_commit,
                "parent_commit": start_commit,
                "events": candidate,
                "optimizer_steps": 0,
                "training_authorized": False,
            }
            return pool.validate_drand_transport_trace(
                artifact,
                self.reg,
                stage="R0_REQUEST",
                protocol_commit=protocol_commit,
                parent_commit=start_commit,
                required_success_operations=beacon.RANDOMNESS_REQUIRED_OPERATIONS[
                    "R0_REQUEST"
                ],
                expected_payloads=payloads,
            )

        self.assertEqual(validate(copy.deepcopy(events))["status"], "COMPLETE_SUCCESS")
        changed = copy.deepcopy(events)
        changed[1]["mirror_index"] = 2
        changed[1]["url"] = self.reg["randomness_beacon"]["mirror_base_urls"][1] + "/public/latest"
        with self.assertRaisesRegex(ValueError, "increment exactly"):
            validate(changed)
        changed = copy.deepcopy(events)
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            validate(changed)
        changed = copy.deepcopy(events)
        changed[0]["url"] += "/suffix"
        with self.assertRaisesRegex(ValueError, "exact operation endpoint"):
            validate(changed)
        changed = copy.deepcopy(events)
        changed[1]["parsed_object_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "frozen payload"):
            validate(changed)
        changed = copy.deepcopy(events)
        changed.insert(1, copy.deepcopy(changed[0]))
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            validate(changed)

        failover = copy.deepcopy(events)
        failover.insert(
            0,
            {
                "operation": "r0_fetch_info",
                "mirror_index": 1,
                "url": f"{base}/info",
                "outcome": "TRANSPORT_FAILURE",
                "elapsed_milliseconds": 1,
                "parsed_object_sha256": None,
                "failure_class": "PUBLIC_EVIDENCE_TRANSPORT",
            },
        )
        failover[1]["mirror_index"] = 2
        failover[1]["url"] = self.reg["randomness_beacon"]["mirror_base_urls"][1] + "/info"
        self.assertEqual(validate(failover)["status"], "COMPLETE_SUCCESS")

        wrong_fetch_class = copy.deepcopy(failover)
        wrong_fetch_class[0]["failure_class"] = "VERIFIER_TIMEOUT"
        with self.assertRaisesRegex(ValueError, "illegal for its operation and outcome"):
            validate(wrong_fetch_class)

        verifier_failover = copy.deepcopy(events)
        verifier_failover.insert(
            2,
            {
                "operation": "r0_verify_observed",
                "mirror_index": 1,
                "url": base,
                "outcome": "REMOTE_VERIFIER_FAILURE",
                "elapsed_milliseconds": 1,
                "parsed_object_sha256": None,
                "failure_class": "VERIFIER_TIMEOUT",
            },
        )
        verifier_failover[3]["mirror_index"] = 2
        verifier_failover[3]["url"] = self.reg["randomness_beacon"][
            "mirror_base_urls"
        ][1]
        self.assertEqual(
            validate(verifier_failover)["status"], "COMPLETE_SUCCESS"
        )
        wrong_verifier_class = copy.deepcopy(verifier_failover)
        wrong_verifier_class[2]["failure_class"] = "PUBLIC_EVIDENCE_TRANSPORT"
        with self.assertRaisesRegex(ValueError, "illegal for its operation and outcome"):
            validate(wrong_verifier_class)

    def test_randomness_terminal_stop_requires_trace_even_with_no_events(self) -> None:
        temp_root = pool.REPO_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as directory:
            reports = Path(directory)
            start = reports / "start.json"
            trace = reports / "trace.json"
            output = reports / "request.json"
            terminal = reports / "terminal.json"
            start.write_text("{}\n", encoding="utf-8", newline="\n")
            start_commit = "a" * 40
            stop = beacon.build_randomness_stop(
                self.reg,
                stage="R0_REQUEST",
                protocol_commit=self.protocol_commit,
                parent_commit=start_commit,
                error=RuntimeError("synthetic"),
                evidence_artifacts=[pool.artifact_from_bytes(start, start.read_bytes())],
            )
            terminal.write_text(pool.json_text(stop), encoding="utf-8", newline="\n")
            with (
                patch.object(pool, "R0_START", start),
                patch.object(pool, "R0_TRACE", trace),
                patch.object(pool, "BEACON_REQUEST", output),
                patch.object(pool, "ATTEMPT_TERMINAL_STOP", terminal),
                patch.object(pool, "validate_boundaries"),
                patch.object(
                    pool,
                    "resolve_protocol_commit",
                    return_value=self.protocol_commit,
                ),
                patch.object(
                    beacon,
                    "RANDOMNESS_STAGE_PATHS",
                    {"R0_REQUEST": (start, trace, output)},
                ),
                patch.object(
                    beacon,
                    "RANDOMNESS_START_PATHS",
                    {"R0_REQUEST": start},
                ),
                patch.object(
                    beacon,
                    "RANDOMNESS_TRACE_PATHS",
                    {"R0_REQUEST": trace},
                ),
                patch.object(
                    beacon,
                    "verify_request_start",
                    return_value={"start_commit": start_commit},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "mandatory transport trace"):
                    beacon.verify_randomness_stop(require_pushed=False)

    def test_pinned_verifier_uses_fixed_failover_and_stops_on_integrity_error(self) -> None:
        signature = "44" * 96
        pulse = {
            "round": 104,
            "randomness": hashlib.sha256(bytes.fromhex(signature)).hexdigest(),
            "signature": signature,
            "previous_signature": "55" * 96,
        }
        cache_key = (
            self.reg["randomness_beacon"]["chain_hash"],
            self.reg["randomness_beacon"]["public_key"],
            pulse["round"],
            pulse["randomness"],
            pulse["signature"],
            pulse["previous_signature"],
        )
        pool._VERIFIED_DRAND_PULSES.discard(cache_key)
        bases: list[str] = []

        def fail_then_pass(args, **kwargs):
            del kwargs
            payload = json.loads(Path(args[2]).read_text(encoding="utf-8"))
            bases.append(payload["base_url"])
            if len(bases) == 1:
                return subprocess.CompletedProcess(
                    args, 75, stdout="", stderr="synthetic timeout"
                )
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps(pulse) + "\n", stderr=""
            )

        with patch.object(pool.subprocess, "run", side_effect=fail_then_pass):
            self.assertEqual(
                pool.verify_drand_pulse_cryptographically(self.reg, pulse), pulse
            )
        self.assertEqual(bases, list(pool.DRAND_MIRROR_BASE_URLS[:2]))

        pool._VERIFIED_DRAND_PULSES.discard(cache_key)
        bases.clear()

        def integrity_failure(args, **kwargs):
            del kwargs
            payload = json.loads(Path(args[2]).read_text(encoding="utf-8"))
            bases.append(payload["base_url"])
            return subprocess.CompletedProcess(
                args, 1, stdout="", stderr="synthetic invalid proof"
            )

        with patch.object(pool.subprocess, "run", side_effect=integrity_failure):
            with self.assertRaisesRegex(ValueError, "rejected pulse"):
                pool.verify_drand_pulse_cryptographically(self.reg, pulse)
        self.assertEqual(bases, [pool.DRAND_MIRROR_BASE_URLS[0]])

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
            self.reg,
            self.protocol_commit,
            "c" * 40,
            self.info,
            self.pulse,
            "a" * 64,
            "b" * 64,
        )
        self.assertEqual(
            request["target_round"],
            self.reg["randomness_beacon"]["fixed_target_round"],
        )
        self.assertGreater(
            request["target_round_time_unix"], request["observed_round_time_unix"]
        )
        pool.validate_randomness_request(
            request, self.reg, self.protocol_commit
        )

    def test_request_rejects_wrong_protocol_commit(self) -> None:
        request = beacon.build_request_artifact(
            self.reg,
            self.protocol_commit,
            "c" * 40,
            self.info,
            self.pulse,
            "a" * 64,
            "b" * 64,
        )
        with self.assertRaises(ValueError):
            pool.validate_randomness_request(request, self.reg, "f" * 40)

    def test_live_drand_chain_must_match_both_frozen_rounds(self) -> None:
        request = beacon.build_request_artifact(
            self.reg,
            self.protocol_commit,
            "c" * 40,
            self.info,
            self.pulse,
            "a" * 64,
            "b" * 64,
        )
        target_signature = "22" * 96
        target = {
            "round": request["target_round"],
            "randomness": hashlib.sha256(bytes.fromhex(target_signature)).hexdigest(),
            "signature": target_signature,
            "previous_signature": "33" * 96,
        }
        pulse = {"target_pulse": target}
        pulses = {self.pulse["round"]: self.pulse, target["round"]: target}

        def fake_drand_fetch(
            _reg: dict,
            endpoint: str,
            *,
            round_number: int | None = None,
            validator=None,
        ) -> dict:
            value = self.info if endpoint == "info" else pulses[round_number]
            value = copy.deepcopy(value)
            if validator is not None:
                validator(value)
            return value

        with (
            patch.object(pool, "fetch_drand_json", side_effect=fake_drand_fetch),
            patch.object(
                pool,
                "verify_drand_pulse_cryptographically",
                side_effect=lambda _reg, value: copy.deepcopy(value),
            ) as verifier,
        ):
            pool.validate_live_drand_chain(self.reg, request, pulse)
            self.assertEqual(verifier.call_count, 2)
        with (
            patch.object(pool, "fetch_drand_json", side_effect=fake_drand_fetch),
            patch.object(
                pool,
                "verify_drand_pulse_cryptographically",
                side_effect=ValueError("invalid proof"),
            ),
        ):
            with self.assertRaises(ValueError):
                pool.validate_live_drand_chain(self.reg, request, pulse)

    def test_salt_depends_on_public_pulse(self) -> None:
        first = {"beacon_round": 100, "beacon_randomness": self.pulse["randomness"]}
        second = {"beacon_round": 101, "beacon_randomness": self.pulse["randomness"]}
        self.assertNotEqual(
            pool.derive_salt(self.reg, self.protocol_commit, first, "prompt_matching"),
            pool.derive_salt(self.reg, self.protocol_commit, second, "prompt_matching"),
        )

    def test_pulse_artifact_binds_request_hash_and_round(self) -> None:
        request = beacon.build_request_artifact(
            self.reg,
            self.protocol_commit,
            "c" * 40,
            self.info,
            self.pulse,
            "a" * 64,
            "b" * 64,
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
                "d" * 40,
                "b" * 40,
                "a" * 64,
                request,
                target,
                request_ci,
                "c" * 64,
                "d" * 64,
            )
            spoofed = copy.deepcopy(request_ci)
            spoofed["app_slug"] = "third-party"
            with self.assertRaises(ValueError):
                beacon.build_pulse_artifact(
                    self.reg,
                    self.protocol_commit,
                    "d" * 40,
                    "b" * 40,
                    "a" * 64,
                    request,
                    target,
                    spoofed,
                    "c" * 64,
                    "d" * 64,
                )
            rerun = copy.deepcopy(request_ci)
            rerun["workflow_run_attempt"] = 2
            with self.assertRaises(ValueError):
                beacon.build_pulse_artifact(
                    self.reg,
                    self.protocol_commit,
                    "d" * 40,
                    "b" * 40,
                    "a" * 64,
                    request,
                    target,
                    rerun,
                    "c" * 64,
                    "d" * 64,
                )
            late = copy.deepcopy(request_ci)
            late["completed_at"] = datetime.fromtimestamp(
                request["target_round_time_unix"], tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")
            with self.assertRaises(ValueError):
                beacon.build_pulse_artifact(
                    self.reg,
                    self.protocol_commit,
                    "d" * 40,
                    "b" * 40,
                    "a" * 64,
                    request,
                    target,
                    late,
                    "c" * 64,
                    "d" * 64,
                )
        finally:
            pass
        self.assertEqual(artifact["target_round"], request["target_round"])
        self.assertEqual(artifact["request_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
