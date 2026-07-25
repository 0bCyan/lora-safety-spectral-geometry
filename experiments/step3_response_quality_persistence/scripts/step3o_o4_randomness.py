#!/usr/bin/env python3
"""Freeze a future public drand pulse before O4 prompt selection."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import Any

import step3o_o4_training_pool as pool


RANDOMNESS_STAGE_PATHS = {
    "R0_REQUEST": (pool.R0_START, pool.R0_TRACE, pool.BEACON_REQUEST),
    "R1_PULSE": (pool.R1_START, pool.R1_TRACE, pool.BEACON_PULSE),
}
RANDOMNESS_START_PATHS = {
    "R0_REQUEST": pool.R0_START,
    "R1_PULSE": pool.R1_START,
}
RANDOMNESS_TRACE_PATHS = {
    "R0_REQUEST": pool.R0_TRACE,
    "R1_PULSE": pool.R1_TRACE,
}
RANDOMNESS_OUTPUT_PATHS = {
    "R0_REQUEST": pool.BEACON_REQUEST,
    "R1_PULSE": pool.BEACON_PULSE,
}
RANDOMNESS_REQUIRED_OPERATIONS = {
    "R0_REQUEST": {"r0_fetch_info", "r0_fetch_latest", "r0_verify_observed"},
    "R1_PULSE": {
        "r1_refetch_info",
        "r1_verify_observed",
        "r1_fetch_target",
        "r1_verify_target",
    },
}


def fetch_json(
    reg: dict[str, Any],
    endpoint: str,
    *,
    round_number: int | None = None,
    trace: list[dict[str, Any]] | None = None,
    trace_operation: str | None = None,
) -> dict[str, Any]:
    if endpoint == "info":
        validator = lambda value: pool.validate_drand_info(value, reg)
    elif endpoint in {"latest", "round"}:
        validator = lambda value: pool.validate_drand_pulse(
            value, expected_round=round_number if endpoint == "round" else None
        )
    else:
        raise ValueError(f"unknown drand endpoint: {endpoint}")
    return pool.fetch_drand_json(
        reg,
        endpoint,
        round_number=round_number,
        validator=validator,
        trace=trace,
        trace_operation=trace_operation,
    )


def canonical_chain_info(value: dict[str, Any], reg: dict[str, Any]) -> dict[str, Any]:
    return pool.canonical_drand_info(value, reg)


def build_start_receipt(
    reg: dict[str, Any],
    *,
    stage: str,
    protocol_commit: str,
    parent_commit: str,
) -> dict[str, Any]:
    value = {
        "name": "step3o_o4_randomness_start_v2",
        "attempt_id": reg["attempt"]["attempt_id"],
        "stage": stage,
        "status": "AUTHORIZED_ONE_ACCEPTED_OUTCOME_FIXED_TARGET",
        "protocol_commit": protocol_commit,
        "parent_commit": parent_commit,
        "fixed_target_round": reg["randomness_beacon"]["fixed_target_round"],
        "expected_output": pool.repo_path(RANDOMNESS_OUTPUT_PATHS[stage]),
        "hidden_local_invocation_absence_claimed": False,
        "replacement_or_retry_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return pool.validate_randomness_start(
        value,
        reg,
        stage=stage,
        protocol_commit=protocol_commit,
        parent_commit=parent_commit,
        expected_output=RANDOMNESS_OUTPUT_PATHS[stage],
    )


def build_trace_artifact(
    reg: dict[str, Any],
    *,
    stage: str,
    protocol_commit: str,
    parent_commit: str,
    status: str,
    events: list[dict[str, Any]],
    expected_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = {
        "name": "step3o_o4_drand_transport_trace_v1",
        "attempt_id": reg["attempt"]["attempt_id"],
        "stage": stage,
        "status": status,
        "protocol_commit": protocol_commit,
        "parent_commit": parent_commit,
        "events": events,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    return pool.validate_drand_transport_trace(
        value,
        reg,
        stage=stage,
        protocol_commit=protocol_commit,
        parent_commit=parent_commit,
        required_success_operations=(
            RANDOMNESS_REQUIRED_OPERATIONS[stage]
            if status == "COMPLETE_SUCCESS"
            else None
        ),
        expected_payloads=expected_payloads,
    )


def write_trace_once(
    reg: dict[str, Any],
    *,
    stage: str,
    protocol_commit: str,
    parent_commit: str,
    status: str,
    events: list[dict[str, Any]],
    expected_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = build_trace_artifact(
        reg,
        stage=stage,
        protocol_commit=protocol_commit,
        parent_commit=parent_commit,
        status=status,
        events=events,
        expected_payloads=expected_payloads,
    )
    pool.write_frozen(RANDOMNESS_TRACE_PATHS[stage], pool.json_text(value))
    return value


def build_randomness_stop(
    reg: dict[str, Any],
    *,
    stage: str,
    protocol_commit: str,
    parent_commit: str,
    error: Exception,
    evidence_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": "step3o_o4_randomness_terminal_stop_v1",
        "attempt_id": reg["attempt"]["attempt_id"],
        "stage": stage,
        "route": "STOP_PROTOCOL_INTEGRITY",
        "protocol_commit": protocol_commit,
        "parent_commit": parent_commit,
        "error_type": type(error).__name__,
        "error_message_sha256": pool.sha256_text(str(error)),
        "evidence_artifacts": evidence_artifacts,
        "replacement_or_retry_allowed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def write_randomness_stop(
    reg: dict[str, Any],
    *,
    stage: str,
    protocol_commit: str,
    parent_commit: str,
    error: Exception,
    events: list[dict[str, Any]],
) -> None:
    if pool.ATTEMPT_TERMINAL_STOP.exists():
        return
    trace_path = RANDOMNESS_TRACE_PATHS[stage]
    if not trace_path.exists():
        write_trace_once(
            reg,
            stage=stage,
            protocol_commit=protocol_commit,
            parent_commit=parent_commit,
            status="TERMINAL_FAILURE",
            events=events,
        )
    evidence = [
        pool.artifact_from_bytes(path, path.read_bytes())
        for path in RANDOMNESS_STAGE_PATHS[stage]
        if path.exists()
    ]
    pool.write_frozen(
        pool.ATTEMPT_TERMINAL_STOP,
        pool.json_text(
            build_randomness_stop(
                reg,
                stage=stage,
                protocol_commit=protocol_commit,
                parent_commit=parent_commit,
                error=error,
                evidence_artifacts=evidence,
            )
        ),
    )


def build_request_artifact(
    reg: dict[str, Any],
    protocol_commit: str,
    start_commit: str,
    info: dict[str, Any],
    observed_pulse: dict[str, Any],
    start_receipt_sha256: str,
    transport_trace_sha256: str,
) -> dict[str, Any]:
    cfg = reg["randomness_beacon"]
    observed = pool.validate_drand_pulse(observed_pulse)
    target_round = int(cfg["fixed_target_round"])
    return {
        "name": "step3o_o4_drand_request_v1",
        "status": "FROZEN_REQUEST_BEFORE_TARGET_PULSE",
        "provider": cfg["provider"],
        "protocol_commit": protocol_commit,
        "start_commit": start_commit,
        "start_receipt_sha256": start_receipt_sha256,
        "transport_trace_sha256": transport_trace_sha256,
        "chain_info": canonical_chain_info(info, reg),
        "observed_pulse": observed,
        "observed_round_time_unix": pool.drand_round_time(reg, observed["round"]),
        "target_round": target_round,
        "target_round_time_unix": pool.drand_round_time(reg, target_round),
        "selection_started": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def verify_request(
    *,
    require_pushed: bool,
    live_refetch: bool = True,
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=require_pushed)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    start_commit, start_bytes = pool.verify_randomness_start_commit(
        reg,
        path=pool.R0_START,
        stage="R0_REQUEST",
        expected_output=pool.BEACON_REQUEST,
        protocol_commit=protocol_commit,
        parent_commit=protocol_commit,
        require_pushed=require_pushed,
    )
    request_commit = pool.verify_exact_creation_commit(
        [pool.R0_TRACE, pool.BEACON_REQUEST],
        pool.BEACON_REQUEST,
        start_commit,
        reg=reg,
        require_pushed=require_pushed,
    )
    trace_bytes = pool.git_blob_bytes(request_commit, pool.R0_TRACE)
    request_bytes = pool.git_blob_bytes(request_commit, pool.BEACON_REQUEST)
    value = pool.validate_randomness_request(
        pool.parse_frozen_json_bytes(request_bytes, "R0 request commit blob"),
        reg,
        protocol_commit,
    )
    if value["start_commit"] != start_commit:
        raise ValueError("drand request does not bind its separately locked start commit")
    pool.validate_drand_transport_trace(
        pool.parse_frozen_json_bytes(trace_bytes, "R0 transport-trace commit blob"),
        reg,
        stage="R0_REQUEST",
        protocol_commit=protocol_commit,
        parent_commit=start_commit,
        required_success_operations=RANDOMNESS_REQUIRED_OPERATIONS["R0_REQUEST"],
        expected_payloads={
            "info": value["chain_info"],
            "observed": value["observed_pulse"],
        },
    )
    if (
        value["start_receipt_sha256"] != pool.sha256_bytes(start_bytes)
        or value["transport_trace_sha256"] != pool.sha256_bytes(trace_bytes)
    ):
        raise ValueError("drand request does not bind its start receipt and trace")
    if require_pushed:
        deadline = value["target_round_time_unix"]
        for commit, context in (
            (protocol_commit, "P0"),
            (start_commit, "R0 start"),
            (request_commit, "R0 outcome"),
        ):
            pool.require_lock_before_unix(reg, commit, deadline, context)
    if live_refetch:
        live_info = fetch_json(
            reg,
            "info",
            trace=trace,
            trace_operation="r1_refetch_info" if trace is not None else None,
        )
        pool.validate_drand_info(live_info, reg)
        pool.verify_drand_pulse_cryptographically(
            reg,
            value["observed_pulse"],
            trace=trace,
            trace_operation="r1_verify_observed",
        )
    return {
        "request": value,
        "protocol_commit": protocol_commit,
        "start_commit": start_commit,
        "start_sha256": pool.sha256_bytes(start_bytes),
        "request_commit": request_commit,
        "request_sha256": pool.sha256_bytes(request_bytes),
    }


def verify_request_start(
    *, require_pushed: bool, enforce_predeadline: bool = True
) -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=require_pushed)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    start_commit, start_bytes = pool.verify_randomness_start_commit(
        reg,
        path=pool.R0_START,
        stage="R0_REQUEST",
        expected_output=pool.BEACON_REQUEST,
        protocol_commit=protocol_commit,
        parent_commit=protocol_commit,
        require_pushed=require_pushed,
    )
    if require_pushed and enforce_predeadline:
        deadline = reg["randomness_beacon"]["fixed_target_round_time_unix"]
        for commit, context in (
            (protocol_commit, "P0"),
            (start_commit, "R0 start"),
        ):
            pool.require_lock_before_unix(reg, commit, deadline, context)
    return {
        "reg": reg,
        "protocol_commit": protocol_commit,
        "start_commit": start_commit,
        "start_sha256": pool.sha256_bytes(start_bytes),
    }


def freeze_request_start() -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    deadline = reg["randomness_beacon"]["fixed_target_round_time_unix"]
    pool.require_start_window_open(deadline, "R0 start receipt")
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=True)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    pool.require_locked_canonical_tip(reg, protocol_commit)
    pool.require_lock_before_unix(reg, protocol_commit, deadline, "P0")
    if pool.git_head() != protocol_commit:
        raise ValueError("R0 start receipt must be created directly on the P0 commit")
    existing = {
        pool.repo_path(path) for path in pool.protocol_artifact_files(pool.BASE)
    } if pool.BASE.exists() else set()
    if existing:
        raise ValueError(f"O4 R0 start path is not pristine: {sorted(existing)}")
    start = build_start_receipt(
        reg,
        stage="R0_REQUEST",
        protocol_commit=protocol_commit,
        parent_commit=protocol_commit,
    )
    pool.write_start_receipt_before_unix(
        pool.R0_START,
        pool.json_text(start),
        deadline,
        "R0 start receipt",
    )
    return start


def _create_request_once() -> dict[str, Any]:
    local = verify_request_start(require_pushed=False)
    reg = local["reg"]
    protocol_commit = local["protocol_commit"]
    start_commit = local["start_commit"]
    if pool.git_head() != start_commit:
        raise ValueError("drand request must run directly on the R0 start commit")
    existing = {
        pool.repo_path(path) for path in pool.protocol_artifact_files(pool.BASE)
    }
    expected_existing = {pool.repo_path(pool.R0_START)}
    if existing != expected_existing:
        raise ValueError(f"R0 result path set is not pristine: {sorted(existing)}")
    events: list[dict[str, Any]] = []
    try:
        confirmed = verify_request_start(require_pushed=True)
        if confirmed["start_commit"] != start_commit:
            raise ValueError("remote R0 start commit differs from the local start commit")
        pool.require_locked_canonical_tip(reg, start_commit)
        info = fetch_json(
            reg, "info", trace=events, trace_operation="r0_fetch_info"
        )
        latest = fetch_json(
            reg, "latest", trace=events, trace_operation="r0_fetch_latest"
        )
        pool.verify_drand_pulse_cryptographically(
            reg,
            latest,
            trace=events,
            trace_operation="r0_verify_observed",
        )
        write_trace_once(
            reg,
            stage="R0_REQUEST",
            protocol_commit=protocol_commit,
            parent_commit=start_commit,
            status="COMPLETE_SUCCESS",
            events=events,
            expected_payloads={
                "info": canonical_chain_info(info, reg),
                "observed": latest,
            },
        )
        artifact = build_request_artifact(
            reg,
            protocol_commit,
            start_commit,
            info,
            latest,
            local["start_sha256"],
            pool.sha256_bytes(pool.R0_TRACE.read_bytes()),
        )
        pool.validate_randomness_request(artifact, reg, protocol_commit)
        pool.write_frozen(pool.BEACON_REQUEST, pool.json_text(artifact))
        return artifact
    except Exception as exc:
        write_randomness_stop(
            reg,
            stage="R0_REQUEST",
            protocol_commit=protocol_commit,
            parent_commit=start_commit,
            error=exc,
            events=events,
        )
        raise


def build_pulse_artifact(
    reg: dict[str, Any],
    protocol_commit: str,
    start_commit: str,
    request_commit: str,
    request_sha256: str,
    request: dict[str, Any],
    target_pulse: dict[str, Any],
    request_ci: dict[str, Any],
    start_receipt_sha256: str,
    transport_trace_sha256: str,
) -> dict[str, Any]:
    target = pool.validate_drand_pulse(
        target_pulse, expected_round=request["target_round"]
    )
    return {
        "name": "step3o_o4_drand_target_pulse_v1",
        "status": "FROZEN_PUBLIC_RANDOMNESS_BEFORE_SELECTION",
        "provider": reg["randomness_beacon"]["provider"],
        "protocol_commit": protocol_commit,
        "start_commit": start_commit,
        "request_commit": request_commit,
        "request_sha256": request_sha256,
        "start_receipt_sha256": start_receipt_sha256,
        "transport_trace_sha256": transport_trace_sha256,
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


def verify_pulse_start(
    *, require_pushed: bool, enforce_predeadline: bool = True
) -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    state = verify_request(require_pushed=require_pushed, live_refetch=False)
    start_commit, start_bytes = pool.verify_randomness_start_commit(
        reg,
        path=pool.R1_START,
        stage="R1_PULSE",
        expected_output=pool.BEACON_PULSE,
        protocol_commit=state["protocol_commit"],
        parent_commit=state["request_commit"],
        require_pushed=require_pushed,
    )
    if require_pushed and enforce_predeadline:
        pool.require_lock_before_unix(
            reg,
            start_commit,
            state["request"]["target_round_time_unix"],
            "R1 start",
        )
    return {
        **state,
        "reg": reg,
        "pulse_start_commit": start_commit,
        "pulse_start_sha256": pool.sha256_bytes(start_bytes),
    }


def freeze_pulse_start() -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    deadline = reg["randomness_beacon"]["fixed_target_round_time_unix"]
    pool.require_start_window_open(deadline, "R1 start receipt")
    state = verify_request(require_pushed=True, live_refetch=False)
    pool.require_locked_canonical_tip(reg, state["request_commit"])
    if pool.git_head() != state["request_commit"]:
        raise ValueError("R1 start receipt must be created directly on the request commit")
    expected_existing = {
        pool.repo_path(pool.R0_START),
        pool.repo_path(pool.R0_TRACE),
        pool.repo_path(pool.BEACON_REQUEST),
    }
    existing = {
        pool.repo_path(path) for path in pool.protocol_artifact_files(pool.BASE)
    }
    if existing != expected_existing:
        raise ValueError(f"R1 pre-start path set is not pristine: {sorted(existing)}")
    start = build_start_receipt(
        reg,
        stage="R1_PULSE",
        protocol_commit=state["protocol_commit"],
        parent_commit=state["request_commit"],
    )
    pool.write_start_receipt_before_unix(
        pool.R1_START,
        pool.json_text(start),
        deadline,
        "R1 start receipt",
    )
    return start


def _acquire_pulse_once() -> dict[str, Any]:
    local = verify_pulse_start(require_pushed=False)
    reg = local["reg"]
    start_commit = local["pulse_start_commit"]
    if pool.git_head() != start_commit:
        raise ValueError("target pulse must run directly on the R1 start commit")
    expected_existing = {
        pool.repo_path(pool.R0_START),
        pool.repo_path(pool.R0_TRACE),
        pool.repo_path(pool.BEACON_REQUEST),
        pool.repo_path(pool.R1_START),
    }
    existing = {
        pool.repo_path(path) for path in pool.protocol_artifact_files(pool.BASE)
    }
    if existing != expected_existing:
        raise ValueError(f"R1 result path set is not pristine: {sorted(existing)}")
    events: list[dict[str, Any]] = []
    try:
        confirmed_start = verify_pulse_start(require_pushed=True)
        if confirmed_start["pulse_start_commit"] != start_commit:
            raise ValueError("remote R1 start commit differs from the local start commit")
        pool.require_locked_canonical_tip(reg, start_commit)
        state = verify_request(require_pushed=True, trace=events)
        if state["request_commit"] != local["request_commit"]:
            raise ValueError("remote R0 request differs after the R1 start lock")
        request_ci = pool.github_request_check_evidence(reg, state["request_commit"])
        target = fetch_json(
            reg,
            "round",
            round_number=state["request"]["target_round"],
            trace=events,
            trace_operation="r1_fetch_target",
        )
        pool.verify_drand_pulse_cryptographically(
            reg,
            target,
            trace=events,
            trace_operation="r1_verify_target",
        )
        write_trace_once(
            reg,
            stage="R1_PULSE",
            protocol_commit=state["protocol_commit"],
            parent_commit=start_commit,
            status="COMPLETE_SUCCESS",
            events=events,
            expected_payloads={
                "info": state["request"]["chain_info"],
                "observed": state["request"]["observed_pulse"],
                "target": target,
            },
        )
        artifact = build_pulse_artifact(
            reg,
            state["protocol_commit"],
            start_commit,
            state["request_commit"],
            state["request_sha256"],
            state["request"],
            target,
            request_ci,
            local["pulse_start_sha256"],
            pool.sha256_bytes(pool.R1_TRACE.read_bytes()),
        )
        pool.validate_randomness_pulse_artifact(
            artifact,
            reg,
            state["protocol_commit"],
            state["request_commit"],
            state["request_sha256"],
            state["request"],
        )
        pool.write_frozen(pool.BEACON_PULSE, pool.json_text(artifact))
        return artifact
    except Exception as exc:
        write_randomness_stop(
            reg,
            stage="R1_PULSE",
            protocol_commit=local["protocol_commit"],
            parent_commit=start_commit,
            error=exc,
            events=events,
        )
        raise


def seal_interrupted_stop(stage: str, error: Exception) -> None:
    start_path = RANDOMNESS_START_PATHS[stage]
    if not start_path.exists() or pool.ATTEMPT_TERMINAL_STOP.exists():
        return
    reg = pool.registry()
    start = pool.read_json(start_path)
    protocol_commit = start.get("protocol_commit")
    start_parent = start.get("parent_commit")
    pool.validate_randomness_start(
        start,
        reg,
        stage=stage,
        protocol_commit=protocol_commit,
        parent_commit=start_parent,
        expected_output=RANDOMNESS_OUTPUT_PATHS[stage],
    )
    start_commit = pool.verify_exact_creation_commit(
        [start_path],
        start_path,
        start_parent,
        reg=reg,
        require_pushed=False,
    )
    write_randomness_stop(
        reg,
        stage=stage,
        protocol_commit=protocol_commit,
        parent_commit=start_commit,
        error=error,
        events=[],
    )


def _supervised_worker_entry(worker_name: str, connection: Any) -> None:
    workers = {
        "request": _create_request_once,
        "acquire-pulse": _acquire_pulse_once,
    }
    try:
        worker = workers[worker_name]
        value = worker()
        payload = {"ok": True, "value": value}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    try:
        connection.send(payload)
    finally:
        connection.close()


def supervise_once(worker_name: str, stage: str, timeout_seconds: int) -> dict[str, Any]:
    if worker_name not in {"request", "acquire-pulse"}:
        raise ValueError("unregistered randomness worker")
    receive_connection = None
    send_connection = None
    try:
        context = multiprocessing.get_context("spawn")
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_supervised_worker_entry,
            args=(worker_name, send_connection),
            name=f"step3o-o4-{worker_name}",
        )
        process.start()
    except BaseException as exc:
        if send_connection is not None:
            send_connection.close()
        if receive_connection is not None:
            receive_connection.close()
        error = RuntimeError(
            f"{stage} supervisor failed to start; error_type={type(exc).__name__}"
        )
        seal_interrupted_stop(stage, error)
        raise ValueError(str(error)) from exc
    send_connection.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        receive_connection.close()
        error = TimeoutError(
            f"{stage} exceeded its frozen {timeout_seconds}-second wall-clock budget"
        )
        seal_interrupted_stop(stage, error)
        raise ValueError(str(error))

    pipe_error_type = None
    try:
        payload = receive_connection.recv() if receive_connection.poll() else None
    except (EOFError, OSError) as exc:
        payload = None
        pipe_error_type = type(exc).__name__
    finally:
        receive_connection.close()
    if (
        process.exitcode != 0
        or not isinstance(payload, dict)
        or payload.get("ok") is not True
        or set(payload) != {"ok", "value"}
    ):
        error_type = (
            payload.get("error_type")
            if isinstance(payload, dict)
            else pipe_error_type or "WorkerExit"
        )
        error_message = (
            payload.get("error_message") if isinstance(payload, dict) else f"exitcode={process.exitcode}"
        )
        error = RuntimeError(
            f"{stage} supervised worker failed; error_type={error_type}; "
            f"error_message_sha256={pool.sha256_text(str(error_message))}"
        )
        seal_interrupted_stop(stage, error)
        raise ValueError(str(error))
    value = payload["value"]
    if not isinstance(value, dict):
        error = RuntimeError(f"{stage} supervised worker returned a non-object")
        seal_interrupted_stop(stage, error)
        raise ValueError(str(error))
    return value


def verify_pulse(*, require_pushed: bool) -> dict[str, Any]:
    reg = pool.registry()
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=require_pushed)
    pool.require_protocol_inputs_frozen(reg, protocol_commit)
    return pool.verify_randomness_chain(
        reg, protocol_commit, require_pushed=require_pushed
    )


def validate_randomness_stop(value: dict[str, Any], reg: dict[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "name",
        "attempt_id",
        "stage",
        "route",
        "protocol_commit",
        "parent_commit",
        "error_type",
        "error_message_sha256",
        "evidence_artifacts",
        "replacement_or_retry_allowed",
        "optimizer_steps",
        "training_authorized",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("randomness terminal-stop schema mismatch")
    stage = value["stage"]
    if (
        value["name"] != "step3o_o4_randomness_terminal_stop_v1"
        or value["attempt_id"] != reg["attempt"]["attempt_id"]
        or stage not in RANDOMNESS_STAGE_PATHS
        or value["route"] != "STOP_PROTOCOL_INTEGRITY"
        or not isinstance(value["protocol_commit"], str)
        or not isinstance(value["parent_commit"], str)
        or not isinstance(value["error_type"], str)
        or not value["error_type"]
        or not isinstance(value["error_message_sha256"], str)
        or len(value["error_message_sha256"]) != 64
        or value["replacement_or_retry_allowed"] is not False
        or value["optimizer_steps"] != 0
        or value["training_authorized"] is not False
        or not isinstance(value["evidence_artifacts"], list)
    ):
        raise ValueError("randomness terminal-stop boundary mismatch")
    return value


def verify_randomness_stop(*, require_pushed: bool) -> dict[str, Any]:
    reg = pool.registry()
    pool.validate_boundaries(reg)
    value = validate_randomness_stop(
        pool.parse_frozen_json_bytes(
            pool.ATTEMPT_TERMINAL_STOP.read_bytes(), "randomness terminal stop"
        ),
        reg,
    )
    stage = value["stage"]
    protocol_commit = pool.resolve_protocol_commit(reg, require_pushed=require_pushed)
    if value["protocol_commit"] != protocol_commit:
        raise ValueError("randomness stop protocol commit mismatch")
    if stage == "R0_REQUEST":
        start_state = verify_request_start(
            require_pushed=require_pushed, enforce_predeadline=False
        )
        expected_parent = start_state["start_commit"]
    else:
        start_state = verify_pulse_start(
            require_pushed=require_pushed, enforce_predeadline=False
        )
        expected_parent = start_state["pulse_start_commit"]
    if value["parent_commit"] != expected_parent:
        raise ValueError("randomness stop parent mismatch")
    allowed = {pool.repo_path(path): path for path in RANDOMNESS_STAGE_PATHS[stage]}
    observed_paths: list[Path] = []
    for item in value["evidence_artifacts"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("randomness stop evidence schema mismatch")
        path = allowed.get(item["path"])
        if path is None or path in observed_paths or not path.exists():
            raise ValueError("randomness stop evidence path mismatch")
        if item != pool.artifact_from_bytes(path, path.read_bytes()):
            raise ValueError("randomness stop evidence hash mismatch")
        observed_paths.append(path)
    start_path = RANDOMNESS_START_PATHS[stage]
    trace_path = RANDOMNESS_TRACE_PATHS[stage]
    if start_path not in observed_paths:
        raise ValueError("randomness stop lacks its one-shot start receipt")
    if trace_path not in observed_paths:
        raise ValueError("randomness stop lacks its mandatory transport trace")
    trace = pool.validate_drand_transport_trace(
        pool.read_json(trace_path),
        reg,
        stage=stage,
        protocol_commit=protocol_commit,
        parent_commit=expected_parent,
    )
    changed_evidence_paths = [path for path in observed_paths if path != start_path]
    stop_commit = pool.verify_exact_creation_commit(
        [*changed_evidence_paths, pool.ATTEMPT_TERMINAL_STOP],
        pool.ATTEMPT_TERMINAL_STOP,
        expected_parent,
        reg=reg,
        require_pushed=require_pushed,
    )
    late_start_context = None
    if require_pushed:
        deadline = reg["randomness_beacon"]["fixed_target_round_time_unix"]
        start_context = "R0 start" if stage == "R0_REQUEST" else "R1 start"
        start_checks = (
            ((protocol_commit, "P0"), (expected_parent, start_context))
            if stage == "R0_REQUEST"
            else ((expected_parent, start_context),)
        )
        for commit, context in start_checks:
            _, before_deadline = pool.remote_lock_before_unix(
                reg, commit, deadline, context
            )
            if not before_deadline:
                late_start_context = context
                break
        if late_start_context is not None:
            expected_error = (
                f"{late_start_context} immutable lock was published after the target round"
            )
            if (
                value["error_type"] != "ValueError"
                or value["error_message_sha256"] != pool.sha256_text(expected_error)
                or trace["events"]
            ):
                raise ValueError(
                    "late-start terminal stop does not prove a pre-network deadline failure"
                )
        elif stage == "R0_REQUEST":
            for commit, context in (
                (protocol_commit, "P0"),
                (expected_parent, "R0 start"),
                (stop_commit, "R0 outcome"),
            ):
                pool.require_lock_before_unix(reg, commit, deadline, context)
    return {
        "name": "step3o_o4_randomness_terminal_stop_verification_v1",
        "status": "PASS_TERMINAL_STOP_VERIFIED",
        "stage": stage,
        "route": value["route"],
        "stop_commit": stop_commit,
        "terminal_cause": "LATE_START_LOCK" if late_start_context else "EXECUTION_FAILURE",
        "late_start_context": late_start_context,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def ci_verify_request_future() -> dict[str, Any]:
    state = verify_request(require_pushed=False)
    if pool.git_head() != state["request_commit"]:
        raise ValueError("CI must evaluate the exact drand request commit")
    reg = pool.registry()
    latest = pool.validate_drand_pulse(fetch_json(reg, "latest"))
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
            "start-request",
            "verify-request-start",
            "request",
            "verify-request",
            "start-pulse",
            "verify-pulse-start",
            "acquire-pulse",
            "verify-pulse",
            "verify-stop",
            "ci-verify-request-future",
        ],
    )
    args = parser.parse_args()
    reg = pool.registry()
    if args.command == "start-request":
        result = freeze_request_start()
    elif args.command == "verify-request-start":
        result = verify_request_start(require_pushed=True)
        result.pop("reg", None)
    elif args.command == "request":
        result = supervise_once(
            "request",
            "R0_REQUEST",
            reg["randomness_beacon"]["local_request_command_timeout_seconds"],
        )
    elif args.command == "verify-request":
        result = verify_request(require_pushed=True)
    elif args.command == "start-pulse":
        result = freeze_pulse_start()
    elif args.command == "verify-pulse-start":
        result = verify_pulse_start(require_pushed=True)
        result.pop("reg", None)
    elif args.command == "acquire-pulse":
        result = supervise_once(
            "acquire-pulse",
            "R1_PULSE",
            reg["randomness_beacon"]["local_pulse_command_timeout_seconds"],
        )
    elif args.command == "verify-pulse":
        result = verify_pulse(require_pushed=True)
    elif args.command == "verify-stop":
        result = verify_randomness_stop(require_pushed=True)
    else:
        result = ci_verify_request_future()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
