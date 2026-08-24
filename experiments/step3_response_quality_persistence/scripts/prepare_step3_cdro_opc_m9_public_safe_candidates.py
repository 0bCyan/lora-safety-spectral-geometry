"""Select and seal the prospective M9 public-safe candidate cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import prepare_step3_cdro_opc_m8_authoring as m8


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m9/protocol/cdro_opc_m9_master_protocol.json"
)
SCHEMA = "step3_cdro_opc_m9_public_safe_single_target_master_protocol_v1"
STATUS = "prospective_locked_before_m9_candidate_materialization"
PENDING_HASH = "PENDING_DERIVATION_BEFORE_COMMIT"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT = (
    REPO
    / "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m9/candidates"
)


class M9CandidateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M9CandidateError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise M9CandidateError(f"expected JSONL objects: {path}")
    return rows


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def check_input(spec: Mapping[str, Any], name: str) -> Path:
    path = m8.repo_path(str(spec.get("path", "")))
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        raise M9CandidateError(f"frozen input mismatch: {name}")
    return path


def validate_protocol(protocol: Mapping[str, Any], *, allow_pending: bool) -> None:
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise M9CandidateError("M9 protocol identity differs")
    selection = protocol.get("selection", {})
    if selection != {
        "candidate_ordering": "SHA256(candidate_ordering_salt|source_candidate_id)",
        "candidate_ordering_salt": "cdro-opc-m9-public-candidate-v1",
        "exclude_by": [
            "sample_id",
            "NFKC_casefold_whitespace_normalized_prompt_sha256",
        ],
        "forbidden_selection_inputs": [
            "prior_review_labels",
            "M6_M7_row_metrics",
            "category",
            "severity",
            "response_content_score",
        ],
        "fresh_public_safe_prompts_expected": 77,
        "prompt_ordering": "SHA256(prompt_ordering_salt|sample_id)",
        "prompt_ordering_salt": "cdro-opc-m9-public-prompt-v1",
        "prompts": 60,
        "responses_per_prompt": 1,
    }:
        raise M9CandidateError("M9 selection contract differs")
    if protocol.get("authorization_boundary") != {
        "candidate_generation_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }:
        raise M9CandidateError("M9 authorization boundary differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(protocol.get("entry_commit", ""))):
        raise M9CandidateError("M9 entry commit is not a full commit ID")
    if protocol.get("review") != {
        "primary_candidates": 60,
        "positive_anchors": 12,
        "generic_anchors": 12,
        "mismatch_anchors": 12,
        "clean_safety_negative_anchors": 12,
        "hidden_primary_duplicates": 12,
        "rows_per_reviewer": 120,
        "reliability_and_qualification_gates": "identical_to_M5",
        "eligible_rule": "single_candidate_qualified",
        "eligible_prompts_required": 40,
        "cohort_ordering": "SHA256(590031|cohort|opaque_prompt_id)",
        "calibration_units": 20,
        "sealed_audit_units": 20,
        "split_rule": "alternating_zero_based_selected_rank_even_calibration_odd_sealed",
    }:
        raise M9CandidateError("M9 review contract differs")
    if protocol.get("gpu_validation") != {
        "targets_per_unit": 1,
        "fixed_dose": 0.0337860296353059,
        "virtual_scale_eta": 0.0002,
        "protected_certificates": ["risk", "uniform", "tarc", "continuation"],
        "allowed_relative_drop": 0.04,
        "score_scale_floor": 0.02,
        "required_covered_units_per_split": 19,
        "required_total_units_per_split": 20,
        "certificate_mean_deterioration_max": 0.04,
        "minimum_behavior_prediction": 0.0,
        "minimum_behavior_actual": 0.0,
        "behavior_ratio_lower": 0.25,
        "behavior_ratio_upper": 1.75,
        "maximum_map_relative_l2_error": 0.000002,
        "maximum_map_absolute_error": 0.00001,
        "sealed_opening_rule": "all_common_and_calibration_gates_pass",
        "two_route_branching_claim_authorized": False,
        "positive_mirror_authorized": False,
        "radius_grid_authorized": False,
        "dose_tuning_authorized": False,
        "result_dependent_rerun_authorized": False,
    }:
        raise M9CandidateError("M9 GPU validation contract differs")
    registry_hash = protocol.get("candidate_registry_sha256")
    if allow_pending and registry_hash == PENDING_HASH:
        return
    if not isinstance(registry_hash, str) or not HASH_RE.fullmatch(registry_hash):
        raise M9CandidateError("M9 candidate registry hash is not frozen")


def load_context(
    protocol_path: Path, *, allow_pending: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    protocol = read_json(protocol_path)
    validate_protocol(protocol, allow_pending=allow_pending)
    inputs = protocol.get("frozen_inputs", {})
    required = {
        "risk_source",
        "m3_primary_candidates",
        "m5_primary_candidates",
        "m8_master_protocol",
        "m8_attempt_002_stop",
        "step3p_registry",
        "step3p_source_preflight",
        "step3p_packet_manifest",
        "step3p_primary_candidates",
        "preparer",
        "preparer_tests",
        "protocol_note",
    }
    if set(inputs) != required:
        raise M9CandidateError("M9 frozen input registry differs")
    paths = {name: check_input(spec, name) for name, spec in inputs.items()}

    risk_rows = read_jsonl(paths["risk_source"])
    risk_by_id = {str(row["sample_id"]): row for row in risk_rows}
    if len(risk_rows) != 1000 or len(risk_by_id) != 1000:
        raise M9CandidateError("M9 risk source denominator differs")
    m3_rows = read_jsonl(paths["m3_primary_candidates"])
    m5_rows = read_jsonl(paths["m5_primary_candidates"])
    if (
        len(m3_rows) != 120
        or len({str(row.get("sample_id")) for row in m3_rows}) != 60
        or len(m5_rows) != 120
        or len({str(row.get("sample_id")) for row in m5_rows}) != 60
    ):
        raise M9CandidateError("M3 or M5 exclusion denominator differs")
    m8_master = read_json(paths["m8_master_protocol"])
    m8.validate_protocol(m8_master, allow_pending=False)
    m8_registry, m8_counts = m8.build_registry(m8_master)
    if m8_counts.get("selected_prompts") != 60:
        raise M9CandidateError("M8 exclusion registry differs")
    m8_stop = read_json(paths["m8_attempt_002_stop"])
    if (
        m8_stop.get("route") != "STOP_CDRO_OPC_M8_AUTHORING"
        or m8_stop.get("attempt_002_responses_reused_by_future_attempt") is not False
    ):
        raise M9CandidateError("M8 stop boundary differs")

    step3p_registry = read_json(paths["step3p_registry"])
    preflight = read_json(paths["step3p_source_preflight"])
    packet_manifest = read_json(paths["step3p_packet_manifest"])
    source_rows = read_jsonl(paths["step3p_primary_candidates"])
    if (
        step3p_registry.get("schema") != "step3p_public_paired_pool_registry_v1"
        or step3p_registry.get("source", {}).get("public_revision")
        != "a4200bf784bad53d73fa0536ab4306e475644143"
        or step3p_registry.get("source", {}).get("public_sha256")
        != "9e62bfadf6925beebc6964b17e8dd6b6854ec14230a75b4ff106c8f9fecb7479"
        or step3p_registry.get("source", {}).get("expected_response_source")
        != "Alpaca3-8B"
        or preflight.get("retained_primary_candidates") != 92
        or preflight.get("retained_unique_prompts") != 87
        or packet_manifest.get("primary_candidates") != 92
        or packet_manifest.get("unique_prompts") != 87
        or len(source_rows) != 92
        or not any(
            row.get("path") == "packets/primary_candidates_hidden.jsonl"
            and row.get("sha256") == inputs["step3p_primary_candidates"]["sha256"]
            for row in packet_manifest.get("artifacts", [])
        )
    ):
        raise M9CandidateError("Step3P public-safe provenance differs")

    excluded_samples = {
        str(row["sample_id"]) for row in [*m3_rows, *m5_rows]
    } | {str(row["sample_id"]) for row in m8_registry["prompts"]}
    excluded_prompt_hashes = {
        hashlib.sha256(m8.normalize_prompt(str(row["prompt"])).encode("utf-8")).hexdigest()
        for row in [*m3_rows, *m5_rows, *m8_registry["prompts"]]
    }
    if len(excluded_samples) != 180 or len(excluded_prompt_hashes) != 180:
        raise M9CandidateError("M9 historical exclusion union differs")
    source_candidate_ids = [str(row.get("opaque_candidate_id")) for row in source_rows]
    if len(set(source_candidate_ids)) != len(source_candidate_ids):
        raise M9CandidateError("Step3P source candidate IDs are not unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        sample_id = str(row.get("sample_id"))
        source = risk_by_id.get(sample_id)
        response = row.get("response")
        prompt = row.get("prompt")
        if (
            source is None
            or prompt != source.get("prompt")
            or row.get("source_index") != source.get("source_index")
            or row.get("severity_level") != source.get("severity_level")
            or sorted(row.get("categories", []))
            != sorted(
                key
                for key, enabled in source.get("harm_category", {}).items()
                if enabled
            )
            or sample_id in excluded_samples
            or hashlib.sha256(m8.normalize_prompt(str(prompt)).encode("utf-8")).hexdigest()
            in excluded_prompt_hashes
        ):
            continue
        if (
            not isinstance(response, str)
            or not response
            or row.get("prompt_sha256")
            != hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
            or row.get("response_sha256")
            != hashlib.sha256(response.encode("utf-8")).hexdigest()
            or row.get("normalized_response_sha256")
            != hashlib.sha256(m8.normalize_prompt(response).encode("utf-8")).hexdigest()
            or row.get("response_whitespace_tokens") != len(response.split())
            or not 40 <= len(response.split()) <= 160
        ):
            raise M9CandidateError("Step3P candidate content seal differs")
        grouped[sample_id].append(row)
    if len(grouped) != int(protocol["selection"]["fresh_public_safe_prompts_expected"]):
        raise M9CandidateError("M9 fresh public-safe denominator differs")

    candidate_salt = str(protocol["selection"]["candidate_ordering_salt"])
    one_per_prompt = []
    for sample_id, rows in grouped.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{candidate_salt}|{row['opaque_candidate_id']}".encode("utf-8")
            ).hexdigest()
        )
        one_per_prompt.append(rows[0])
    prompt_salt = str(protocol["selection"]["prompt_ordering_salt"])
    one_per_prompt.sort(
        key=lambda row: hashlib.sha256(
            f"{prompt_salt}|{row['sample_id']}".encode("utf-8")
        ).hexdigest()
    )
    selected = one_per_prompt[: int(protocol["selection"]["prompts"])]
    if len(selected) != 60 or len({str(row["sample_id"]) for row in selected}) != 60:
        raise M9CandidateError("M9 selected candidate denominator differs")
    candidates = []
    for rank, row in enumerate(selected, 1):
        sample_id = str(row["sample_id"])
        candidate_id = (
            "m9_candidate_"
            + hashlib.sha256(
                f"{sample_id}|{row['opaque_candidate_id']}".encode("utf-8")
            ).hexdigest()[:24]
        )
        prompt_id = (
            f"m9_prompt_{rank:03d}_"
            f"{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:12]}"
        )
        candidates.append(
            {
                "rank": rank,
                "opaque_prompt_id": prompt_id,
                "opaque_candidate_id": candidate_id,
                "sample_id": sample_id,
                "source_public_candidate_id": row["opaque_candidate_id"],
                "source_public_line": row["public_source_line"],
                "source_index": row["source_index"],
                "prompt": row["prompt"],
                "response": row["response"],
                "severity_level": row["severity_level"],
                "categories": row["categories"],
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
                "normalized_response_sha256": row["normalized_response_sha256"],
                "response_whitespace_tokens": row["response_whitespace_tokens"],
                "candidate_source": "frozen_Step3P_public_is_safe_true_Alpaca3_8B",
            }
        )
    registry = {
        "schema": "step3_cdro_opc_m9_public_safe_candidate_registry_v1",
        "excluded_prompt_release_count": len(excluded_samples),
        "fresh_public_safe_prompts": len(grouped),
        "selected_prompts": len(candidates),
        "responses_per_prompt": 1,
        "candidates": candidates,
    }
    counts = {
        "source_candidates": len(source_rows),
        "source_prompts": 87,
        "excluded_prompt_releases": len(excluded_samples),
        "fresh_public_safe_prompts": len(grouped),
        "selected_candidates": len(candidates),
        "selected_prompts": len(candidates),
    }
    return registry, counts


def prepare(protocol_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    registry, counts = load_context(protocol_path)
    registry_hash = m8.canonical_sha256(registry)
    if registry_hash != protocol["candidate_registry_sha256"]:
        raise M9CandidateError("M9 candidate registry differs from frozen hash")
    if OUTPUT.exists():
        raise M9CandidateError("M9 candidate output already exists")
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temporary.exists():
        raise M9CandidateError("M9 temporary candidate output already exists")
    try:
        candidates = registry["candidates"]
        private_path = temporary / "private" / "primary_candidates.jsonl"
        private_path.parent.mkdir(parents=True)
        private_path.write_bytes(jsonl_bytes(candidates))
        records = [
            {key: value for key, value in row.items() if key not in {"prompt", "response"}}
            for row in candidates
        ]
        (temporary / "record_manifest.jsonl").write_bytes(jsonl_bytes(records))
        summary = {
            "schema": "step3_cdro_opc_m9_candidate_summary_v1",
            "route": "PASS_CDRO_OPC_M9_PUBLIC_SAFE_CANDIDATE_SELECTION",
            **counts,
            "candidate_registry_sha256": registry_hash,
            "candidate_generation": 0,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        (temporary / "candidate_summary.json").write_bytes(json_bytes(summary))
        (temporary / "CDRO_OPC_M9_CANDIDATE_REPORT.md").write_text(
            "# CDR-OPC M9 Public-Safe Candidate Selection\n\n"
            "M9 selected 60 single-target candidates from 77 public-safe prompts "
            "that are prompt-disjoint from M3, M5, and the full M8 registry. No "
            "candidate was generated, rewritten, or selected by a prior review or "
            "M6/M7 measurement. New blind review is required.\n\n"
            "route=PASS_CDRO_OPC_M9_PUBLIC_SAFE_CANDIDATE_SELECTION\n"
            "optimizer_steps=0\ntraining_authorized=false\n",
            encoding="utf-8",
        )
        artifacts = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": "step3_cdro_opc_m9_candidate_manifest_v1",
            "route": summary["route"],
            "protocol_sha256": sha256_file(protocol_path),
            "artifacts": artifacts,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        temporary.replace(OUTPUT)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("derive-registry-hash", "preflight", "prepare"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = m8.repo_path(args.protocol)
    protocol = read_json(protocol_path)
    registry, counts = load_context(
        protocol_path, allow_pending=args.command == "derive-registry-hash"
    )
    registry_hash = m8.canonical_sha256(registry)
    if args.command == "derive-registry-hash":
        if protocol.get("candidate_registry_sha256") != PENDING_HASH:
            raise M9CandidateError("M9 hash derivation is no longer pending")
        print(json.dumps({"candidate_registry_sha256": registry_hash}, sort_keys=True))
        return
    if registry_hash != protocol["candidate_registry_sha256"]:
        raise M9CandidateError("M9 candidate registry differs from frozen hash")
    if args.command == "prepare":
        prepare(protocol_path)
        return
    print(
        json.dumps(
            {
                "status": "PASS_CDRO_OPC_M9_CANDIDATE_PREFLIGHT",
                **counts,
                "candidate_registry_sha256": registry_hash,
                "optimizer_steps": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
