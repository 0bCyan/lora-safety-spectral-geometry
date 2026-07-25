#!/usr/bin/env python3
"""Freeze a future public drand pulse before O4 prompt selection."""

from __future__ import annotations

import argparse
import json
from typing import Any

import step3o_o4_training_pool as pool


def fetch_json(url: str) -> dict[str, Any]:
    return pool.fetch_public_json(url)


def canonical_chain_info(value: dict[str, Any], reg: dict[str, Any]) -> dict[str, Any]:
    pool.validate_drand_info(value, reg)
    return {
        "public_key": value["public_key"],
        "period": value["period"],
        "genesis_time": value["genesis_time"],
        "hash": value["hash"],
        "schemeID": value["schemeID"],
    }


def build_request_artifact(
    reg: dict[str, Any],
    protocol_commit: str,
    info: dict[str, Any],
    observed_pulse: dict[str, Any],
) -> dict[str, Any]:
    cfg = reg["randomness_beacon"]
    observed = pool.validate_drand_pulse(observed_pulse)
    offset = int(cfg["target_round_offset"])
    target_round = observed["round"] + offset
    return {
        "name": "step3o_o4_drand_request_v1",
        "status": "FROZEN_REQUEST_BEFORE_TARGET_PULSE",
        "provider": cfg["provider"],
        "protocol_commit": protocol_commit,
        "chain_info": canonical_chain_info(info, reg),
        "observed_pulse": observed,
        "observed_round_time_unix": pool.drand_round_time(reg, observed["round"]),
        "target_round": target_round,
        "target_round_time_unix": pool.drand_round_time(reg, target_round),
        "target_round_offset": offset,
        "selection_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def verify_request(*, require_pushed: bool) -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=require_pushed)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    request_commit = pool.verify_exact_creation_commit(
        [pool.BEACON_REQUEST],
        pool.BEACON_REQUEST,
        protocol_commit,
        reg=reg,
        require_pushed=require_pushed,
    )
    request_bytes = pool.git_blob_bytes(request_commit, pool.BEACON_REQUEST)
    value = pool.validate_randomness_request(
        pool.parse_frozen_json_bytes(request_bytes, "R0 request commit blob"),
        reg,
        protocol_commit,
    )
    live_info = fetch_json(reg["randomness_beacon"]["info_url"])
    pool.validate_drand_info(live_info, reg)
    pool.verify_drand_pulse_cryptographically(reg, value["observed_pulse"])
    return {
        "request": value,
        "protocol_commit": protocol_commit,
        "request_commit": request_commit,
        "request_sha256": pool.sha256_bytes(request_bytes),
    }


def create_request() -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=True)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    if pool.git_head() != protocol_commit:
        raise ValueError("drand request must be created directly on the pushed P0 commit")
    existing = {
        pool.repo_path(path)
        for path in pool.BASE.rglob("*")
        if path.is_file()
    } if pool.BASE.exists() else set()
    if existing:
        raise ValueError(f"O4 randomness request path is not pristine: {sorted(existing)}")
    cfg = reg["randomness_beacon"]
    info = fetch_json(cfg["info_url"])
    latest = fetch_json(cfg["latest_url"])
    pool.verify_drand_pulse_cryptographically(reg, latest)
    artifact = build_request_artifact(reg, protocol_commit, info, latest)
    pool.write_frozen(pool.BEACON_REQUEST, pool.json_text(artifact))
    return artifact


def build_pulse_artifact(
    reg: dict[str, Any],
    protocol_commit: str,
    request_commit: str,
    request_sha256: str,
    request: dict[str, Any],
    target_pulse: dict[str, Any],
    request_ci: dict[str, Any],
) -> dict[str, Any]:
    target = pool.validate_drand_pulse(
        target_pulse, expected_round=request["target_round"]
    )
    return {
        "name": "step3o_o4_drand_target_pulse_v1",
        "status": "FROZEN_PUBLIC_RANDOMNESS_BEFORE_SELECTION",
        "provider": reg["randomness_beacon"]["provider"],
        "protocol_commit": protocol_commit,
        "request_commit": request_commit,
        "request_sha256": request_sha256,
        "target_round": request["target_round"],
        "target_round_time_unix": request["target_round_time_unix"],
        "target_pulse": target,
        "request_ci": pool.validate_request_ci_evidence(
            request_ci,
            reg,
            request_commit,
            target_round_time_unix=request["target_round_time_unix"],
        ),
        "selection_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def acquire_pulse() -> dict[str, Any]:
    state = verify_request(require_pushed=True)
    reg = pool.registry()
    if pool.git_head() != state["request_commit"]:
        raise ValueError(
            "target pulse must be acquired directly on the pushed request commit"
        )
    if pool.BEACON_PULSE.exists():
        raise ValueError("target pulse artifact already exists")
    request_ci = pool.github_request_check_evidence(
        reg, state["request_commit"]
    )
    url = reg["randomness_beacon"]["round_url_template"].format(
        round=state["request"]["target_round"]
    )
    target = fetch_json(url)
    pool.verify_drand_pulse_cryptographically(reg, target)
    artifact = build_pulse_artifact(
        reg,
        state["protocol_commit"],
        state["request_commit"],
        state["request_sha256"],
        state["request"],
        target,
        request_ci,
    )
    pool.write_frozen(pool.BEACON_PULSE, pool.json_text(artifact))
    return artifact


def verify_pulse(*, require_pushed: bool) -> dict[str, Any]:
    reg = pool.registry()
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=require_pushed)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    return pool.verify_randomness_chain(
        reg, protocol_commit, require_pushed=require_pushed
    )


def ci_verify_request_future() -> dict[str, Any]:
    state = verify_request(require_pushed=False)
    if pool.git_head() != state["request_commit"]:
        raise ValueError("CI must evaluate the exact drand request commit")
    reg = pool.registry()
    latest = pool.validate_drand_pulse(
        fetch_json(reg["randomness_beacon"]["latest_url"])
    )
    target_round = state["request"]["target_round"]
    if latest["round"] >= target_round:
        raise ValueError(
            "target drand pulse was already published when request CI ran"
        )
    return {
        "name": "step3o_o4_drand_request_ci_verification_v1",
        "status": "PASS_TARGET_ROUND_STILL_FUTURE",
        "request_commit": state["request_commit"],
        "latest_round_at_ci": latest["round"],
        "target_round": target_round,
        "unpublished_round_margin": target_round - latest["round"],
        "selection_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "request",
            "verify-request",
            "acquire-pulse",
            "verify-pulse",
            "ci-verify-request-future",
        ],
    )
    args = parser.parse_args()
    if args.command == "request":
        result = create_request()
    elif args.command == "verify-request":
        result = verify_request(require_pushed=True)
    elif args.command == "acquire-pulse":
        result = acquire_pulse()
    elif args.command == "verify-pulse":
        result = verify_pulse(require_pushed=True)
    else:
        result = ci_verify_request_future()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
