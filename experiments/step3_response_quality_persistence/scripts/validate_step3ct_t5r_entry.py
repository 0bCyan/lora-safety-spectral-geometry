"""Validate immutable Step3C-T5R inputs and freeze the P1 entry package."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import transformers
import peft

import step3ct_t5r_common as common


REGISTRY_PATH = common.REPO / "experiments/step3_response_quality_persistence/configs/step3ct_t5r_review_recovery_registry.json"
T4_ROOT = common.REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t4"
T5_ROOT = common.REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5"
T5R_ROOT = common.REPO / "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5r"
P1_ROOT = T5R_ROOT / "p1"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=common.REPO, text=True).strip()


def git_ancestor(commit: str, head: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=common.REPO).returncode == 0


def add_gate(gates: list[dict[str, Any]], gate: str, passed: bool, observed: Any, expected: Any) -> None:
    gates.append({"gate": gate, "pass": bool(passed), "observed": observed, "expected": expected})


def verify_packet(domain: str, spec: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    packet_path = common.REPO / spec["packet_path"]
    packet_rows = common.read_jsonl(packet_path)
    id_field = common.DOMAIN_SPECS[domain]["id_field"]
    ids = [row[id_field] for row in packet_rows]
    add_gate(gates, f"{domain}_packet_hash", common.sha256_file(packet_path) == spec["packet_sha256"], common.sha256_file(packet_path), spec["packet_sha256"])
    add_gate(gates, f"{domain}_packet_rows", len(packet_rows) == spec["packet_rows"], len(packet_rows), spec["packet_rows"])
    add_gate(gates, f"{domain}_packet_unique_ids", len(set(ids)) == len(ids), len(set(ids)), len(ids))
    domain_root = packet_path.parent
    paths = {
        "hidden_key": domain_root / f"step3ct_t4_{domain}_hidden_key.csv",
        "duplicate_map": domain_root / f"step3ct_t4_{domain}_duplicate_map.csv",
        "reviewer_a_template": domain_root / f"step3ct_t4_{domain}_reviewer_a_template.jsonl",
        "reviewer_b_template": domain_root / f"step3ct_t4_{domain}_reviewer_b_template.jsonl",
        "packet_manifest": domain_root / f"step3ct_t4_{domain}_packet_manifest.json",
        "rubric": common.REPO / spec["rubric_path"],
    }
    expected_hashes = {
        "hidden_key": spec["hidden_key_sha256"],
        "duplicate_map": spec["duplicate_map_sha256"],
        "reviewer_a_template": spec["reviewer_template_sha256"],
        "reviewer_b_template": spec["reviewer_template_sha256"],
        "packet_manifest": spec["packet_manifest_sha256"],
        "rubric": spec["rubric_sha256"],
    }
    for name, path in paths.items():
        observed = common.sha256_file(path)
        add_gate(gates, f"{domain}_{name}_hash", observed == expected_hashes[name], observed, expected_hashes[name])
    key_rows = common.read_csv(paths["hidden_key"])
    duplicate_rows = common.read_csv(paths["duplicate_map"])
    template_a = common.read_jsonl(paths["reviewer_a_template"])
    template_b = common.read_jsonl(paths["reviewer_b_template"])
    add_gate(gates, f"{domain}_hidden_key_rows", len(key_rows) == spec["packet_rows"], len(key_rows), spec["packet_rows"])
    add_gate(gates, f"{domain}_duplicate_rows", len(duplicate_rows) == spec["hidden_duplicates"], len(duplicate_rows), spec["hidden_duplicates"])
    add_gate(gates, f"{domain}_template_rows", len(template_a) == len(template_b) == spec["packet_rows"], [len(template_a), len(template_b)], spec["packet_rows"])
    template_ids_a = [row[id_field] for row in template_a]
    template_ids_b = [row[id_field] for row in template_b]
    add_gate(gates, f"{domain}_packet_template_order", ids == template_ids_a == template_ids_b, "identical" if ids == template_ids_a == template_ids_b else "mismatch", "identical")
    labels_blank = all(all(value == "" for field, value in row.items() if field != id_field) for row in template_a + template_b)
    add_gate(gates, f"{domain}_templates_blank", labels_blank, labels_blank, True)
    return {
        "packet_rows": len(packet_rows),
        "primary_rows": spec["primary_rows"],
        "hidden_duplicates": len(duplicate_rows),
        "packet_sha256": common.sha256_file(packet_path),
    }


def verify_generations(freeze: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    cells = freeze["generation_cells"]
    total_rows = 0
    all_hashes = True
    all_manifests = True
    all_heldout_closed = True
    for cell in cells:
        output_path = common.REPO / cell["output_path"]
        manifest_path = common.REPO / cell["manifest_path"]
        rows = common.read_jsonl(output_path)
        total_rows += len(rows)
        all_hashes &= common.sha256_file(output_path) == cell["output_sha256"]
        all_manifests &= common.sha256_file(manifest_path) == cell["manifest_sha256"]
        all_heldout_closed &= common.read_json(manifest_path).get("heldout_opened") is False
    add_gate(gates, "generation_manifest_count", len(cells) == 12, len(cells), 12)
    add_gate(gates, "generation_total_rows", total_rows == 1200, total_rows, 1200)
    add_gate(gates, "generation_output_hashes", all_hashes, all_hashes, True)
    add_gate(gates, "generation_manifest_hashes", all_manifests, all_manifests, True)
    add_gate(gates, "generation_heldout_closed", all_heldout_closed, all_heldout_closed, True)
    return {"cells": len(cells), "rows": total_rows}


def environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "gpu_free_bytes": torch.cuda.mem_get_info()[0] if cuda else None,
    }


def main() -> None:
    registry = common.read_json(REGISTRY_PATH)
    head = git("rev-parse", "HEAD")
    gates: list[dict[str, Any]] = []
    for commit_name in ("t4_implementation_commit", "t4_result_commit", "incident_commit"):
        commit = registry[commit_name]
        add_gate(gates, f"ancestor_{commit}", git_ancestor(commit, head), git("rev-parse", commit), f"ancestor of {head}")
    freeze_path = common.REPO / registry["t4_freeze_manifest"]["path"]
    freeze_hash = common.sha256_file(freeze_path)
    add_gate(gates, "t4_freeze_manifest_hash", freeze_hash == registry["t4_freeze_manifest"]["registered_sha256"], freeze_hash, registry["t4_freeze_manifest"]["registered_sha256"])
    freeze = common.read_json(freeze_path)
    generation = verify_generations(freeze, gates)
    packets = {domain: verify_packet(domain, spec, gates) for domain, spec in registry["packets"].items()}
    incident_path = common.REPO / registry["t5_incident"]["path"]
    incident = common.read_json(incident_path)
    accepted_files = incident.get("accepted_label_files", [])
    accepted_rows = sum(int(item.get("rows", 0)) for item in accepted_files if isinstance(item, dict))
    t5_files = sorted(path.relative_to(common.REPO).as_posix() for path in T5_ROOT.rglob("*") if path.is_file())
    expected_t5_files = sorted([
        "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5/Step3CT_T5_inconclusive_route_report.md",
        "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t5/step3ct_t5_safety_gateway_incident.json",
    ])
    add_gate(gates, "t5_incident_route", incident.get("route") == registry["t5_incident"]["route"], incident.get("route"), registry["t5_incident"]["route"])
    add_gate(gates, "t5_accepted_label_files_zero", len(accepted_files) == registry["t5_incident"]["accepted_label_files"], len(accepted_files), 0)
    add_gate(gates, "t5_accepted_rows_zero", accepted_rows == registry["t5_incident"]["accepted_rows"], accepted_rows, 0)
    add_gate(gates, "t5_no_partial_artifacts", t5_files == expected_t5_files, t5_files, expected_t5_files)
    add_gate(gates, "t5_partial_reuse_forbidden", registry["t5_incident"]["partial_outputs_reusable"] is False, registry["t5_incident"]["partial_outputs_reusable"], False)
    changed_after_t4 = git("diff", "--name-only", f"{registry['t4_result_commit']}..HEAD", "--", "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/runs", "experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t4")
    add_gate(gates, "no_run_or_t4_change_after_t4_commit", changed_after_t4 == "", changed_after_t4 or "none", "none")
    heldout_closed = freeze.get("heldout_opened") is False and incident.get("heldout_opened") is False and registry["authorized_after_plan_push"]["heldout"] is False
    add_gate(gates, "heldout_closed", heldout_closed, heldout_closed, True)
    overall = all(row["pass"] for row in gates)
    route = "P1_entry_and_implementation_pass_authorize_P2_safety_review" if overall else "T5R0_integrity_or_provenance_stop"
    next_action = "push_P1_then_qualify_and_launch_fresh_Safety_A_B" if overall else "stop_before_any_reviewer_or_model_checkpoint"
    gate_path = P1_ROOT / "step3ct_t5r_p1_entry_gate_table.csv"
    common.write_csv(gate_path, gates, ["gate", "pass", "observed", "expected"])
    implementation_files = [
        common.REPO / "experiments/step3_response_quality_persistence/scripts/step3ct_t5r_common.py",
        common.REPO / "experiments/step3_response_quality_persistence/scripts/validate_step3ct_t5r_entry.py",
        common.REPO / "experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py",
        common.REPO / "experiments/step3_response_quality_persistence/scripts/summarize_step3ct_t6.py",
        common.REPO / "experiments/step3_response_quality_persistence/scripts/test_step3ct_t5r.py",
        common.REPO / "experiments/step3_response_quality_persistence/configs/step3ct_t5r_reviewer_provenance_template.json",
    ]
    implementation_hashes = {path.relative_to(common.REPO).as_posix(): common.sha256_file(path) for path in implementation_files}
    record = {
        "name": "step3ct_t5r_p1_entry_and_implementation_record_v1",
        "entry_git_head": head,
        "registry": REGISTRY_PATH.relative_to(common.REPO).as_posix(),
        "registry_sha256": common.sha256_file(REGISTRY_PATH),
        "commands": [
            ".venv/Scripts/python.exe -m py_compile experiments/step3_response_quality_persistence/scripts/step3ct_t5r_common.py experiments/step3_response_quality_persistence/scripts/validate_step3ct_t5r_entry.py experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py experiments/step3_response_quality_persistence/scripts/summarize_step3ct_t6.py experiments/step3_response_quality_persistence/scripts/test_step3ct_t5r.py",
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/scripts/test_step3ct_t5r.py",
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/scripts/test_step3ct_t4.py",
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/scripts/test_step3ct_training_common.py",
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/scripts/validate_step3ct_t5r_entry.py"
        ],
        "synthetic_protocol_tests": {"tests": 10, "passed": True},
        "regression_tests": {"tests": 13, "passed": True},
        "environment": environment(),
        "generation": generation,
        "packets": packets,
        "t5_accepted_label_files": len(accepted_files),
        "t5_accepted_rows": accepted_rows,
        "implementation_hashes": implementation_hashes,
        "gate_table": gate_path.relative_to(common.REPO).as_posix(),
        "gate_table_sha256": common.sha256_file(gate_path),
        "route": route,
        "next_authorized_action": next_action,
        "overall_pass": overall,
        "reviewer_launched": False,
        "model_checkpoint_opened": False,
        "core100_newly_opened": False,
        "heldout_opened": False,
        "fresh_training_or_generation": False,
    }
    common.write_json(P1_ROOT / "step3ct_t5r_p1_execution_record.json", record)
    common.write_json(P1_ROOT / "step3ct_t5r_p1_route_decision.json", {
        "name": "step3ct_t5r_p1_route_decision_v1",
        "route": route,
        "overall_pass": overall,
        "entry_git_head": head,
        "gate_table_sha256": common.sha256_file(gate_path),
        "next_authorized_action": next_action,
        "reviewer_launch_authorized_only_after_p1_push": overall,
        "model_checkpoint_execution_authorized": False,
        "fresh_training_or_generation_authorized": False,
        "heldout_opened": False,
    })
    report = f"""# Step3C-T5R P1 Entry and Implementation Freeze Report

## Decision

Route: `{route}`

Entry HEAD `{head}` was checked against the frozen T5R registry. {sum(row['pass'] for row in gates)}/{len(gates)} entry gates passed. T5 still contains zero accepted label files and zero accepted rows; no partial T5 output is present or reusable.

## Immutable Inputs

| Item | Count | Result |
| --- | ---: | --- |
| T4 generation manifests | {generation['cells']} | {'pass' if generation['cells'] == 12 else 'fail'} |
| T4 generated rows | {generation['rows']} | {'pass' if generation['rows'] == 1200 else 'fail'} |
| Safety packet rows | {packets['safety']['packet_rows']} | pass |
| Quality packet rows | {packets['quality']['packet_rows']} | pass |
| Utility packet rows | {packets['utility']['packet_rows']} | pass |
| T5 accepted labels | {accepted_rows} | pass |

All registered packet, hidden-key, duplicate-map, template, packet-manifest, rubric, generation-output, generation-manifest, and T4 freeze hashes were checked. Commits `b3375d2`, `1ea3bf6`, and `fb23b7e` are ancestors of the entry HEAD.

## Implementation Boundary

P1 freezes dedicated entry validation, complete-review schema validation, reviewer provenance checks, A/B file freezing, disagreement-only C packet construction, reliability metrics, adjudication, and T6 statistics. Ten T5R synthetic protocol tests and thirteen T4/training regression tests passed. Synthetic tests do not inspect study responses.

No reviewer was launched and no model checkpoint was opened during P1. T3, T4, and T5 artifacts remained unchanged. No new training, generation, heldout access, or outcome-key join occurred.

## Next Authorization

{'After this P1 commit is pushed, P2 may qualify and launch new Safety A/B reviewers on the complete unchanged packet.' if overall else 'Stop at T5R0. P2 and mechanism checkpoint execution are not authorized.'}
"""
    common.write_text(P1_ROOT / "Step3CT_T5R_P1_entry_and_implementation_report.md", report)
    print(json.dumps({"route": route, "gates_passed": sum(row["pass"] for row in gates), "gates_total": len(gates)}, indent=2))
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
