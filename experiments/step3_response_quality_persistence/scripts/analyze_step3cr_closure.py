#!/usr/bin/env python3
"""Freeze a reproducible synthesis of the two completed C-R0 sessions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
V1 = RESULTS / "step3cr_cr0"
V2 = RESULTS / "step3cr_cr0_v2"
OUT = RESULTS / "step3cr_closure" / "reports"

V1_AGREEMENT = V1 / "reports/step3cr_cr0_review_agreement.json"
V2_AGREEMENT = V2 / "reports/step3cr_cr0_v2_review_agreement.json"
V1_KEY = V1 / "packets/step3cr_cr0_pairwise_hidden_key.csv"
V2_KEY = V2 / "packets/step3cr_cr0_v2_pairwise_hidden_key.csv"
V1_LABELS = V1 / "reviews/step3cr_cr0_final_canonical_labels.jsonl"
V2_LABELS = V2 / "reviews/step3cr_cr0_v2_final_canonical_labels.jsonl"
V1_DECISION = V1 / "reports/step3cr_cr0_route_decision.json"
V2_DECISION = V2 / "reports/step3cr_cr0_v2_route_decision.json"

SUMMARY = OUT / "step3cr_closure_summary.json"
REPORT = OUT / "Step3CR_closure_synthesis.md"
MANIFEST = OUT / "step3cr_closure_manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def canonical_roles(key_row: dict[str, str]) -> tuple[str, str]:
    left_is_canonical = key_row["display_left_opaque_response_id"] == key_row["canonical_left_opaque_response_id"]
    if left_is_canonical:
        return key_row["display_left_role"], key_row["display_right_role"]
    return key_row["display_right_role"], key_row["display_left_role"]


def session_rows(key_path: Path, labels_path: Path) -> dict[str, dict[str, Any]]:
    labels = {row["canonical_pair_id"]: row for row in read_jsonl(labels_path)}
    output: dict[str, dict[str, Any]] = {}
    for key_row in read_csv(key_path):
        if key_row["is_reversed_duplicate"] != "False":
            continue
        label = labels[key_row["canonical_pair_id"]]
        left_role, right_role = canonical_roles(key_row)
        preference = label["pair_preference"]
        if preference == "canonical_left":
            winning_response_id = key_row["canonical_left_opaque_response_id"]
            winning_role = left_role
        elif preference == "canonical_right":
            winning_response_id = key_row["canonical_right_opaque_response_id"]
            winning_role = right_role
        else:
            winning_response_id = "tie"
            winning_role = "tie"
        output[key_row["sample_id"]] = {
            "sample_id": key_row["sample_id"],
            "category": key_row["category"],
            "construction_rule": key_row["construction_rule"],
            "canonical_left_role": left_role,
            "canonical_right_role": right_role,
            "canonical_left_response_id": key_row["canonical_left_opaque_response_id"],
            "canonical_right_response_id": key_row["canonical_right_opaque_response_id"],
            "winning_response_id": winning_response_id,
            "winning_role": winning_role,
            "left_specific": label["canonical_left_specific_safe_next_step"],
            "right_specific": label["canonical_right_specific_safe_next_step"],
            "adjudicated": label["adjudicated"],
        }
    if len(output) != 40:
        raise ValueError(f"expected 40 canonical sample rows in {key_path}, found {len(output)}")
    return output


def role_specificity(row: dict[str, Any], role: str) -> str | None:
    if row["canonical_left_role"] == role:
        return row["left_specific"]
    if row["canonical_right_role"] == role:
        return row["right_specific"]
    return None


def summarize_session(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows.values() if row["construction_rule"] == "constant20_valid_vs_lexicographically_smallest_existing_target"]
    role_counts: dict[str, Counter[str]] = {}
    for row in rows.values():
        for side, role in (("left", row["canonical_left_role"]), ("right", row["canonical_right_role"])):
            role_counts.setdefault(role, Counter())[row[f"{side}_specific"]] += 1
    return {
        "canonical_pairs": len(rows),
        "constant20_vs_natural_pairs": len(paired),
        "constant20_wins": sum(row["winning_role"] == "constant20_generated_negative_candidate" for row in paired),
        "natural_target_wins": sum(row["winning_role"] == "existing_natural_v1_target" for row in paired),
        "ties": sum(row["winning_role"] == "tie" for row in paired),
        "constant20_specific_yes_in_paired_clean_safe": sum(
            role_specificity(row, "constant20_generated_negative_candidate") == "yes" for row in paired
        ),
        "natural_specific_yes_in_paired_clean_safe": sum(
            role_specificity(row, "existing_natural_v1_target") == "yes" for row in paired
        ),
        "specificity_by_role": {
            role: {"yes": counts["yes"], "no": counts["no"], "total": sum(counts.values())}
            for role, counts in sorted(role_counts.items())
        },
        "adjudicated_canonical_pairs": sum(row["adjudicated"] == "yes" for row in rows.values()),
    }


def main() -> None:
    inputs = [V1_AGREEMENT, V2_AGREEMENT, V1_KEY, V2_KEY, V1_LABELS, V2_LABELS, V1_DECISION, V2_DECISION]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing closure inputs: {missing}")

    v1_rows = session_rows(V1_KEY, V1_LABELS)
    v2_rows = session_rows(V2_KEY, V2_LABELS)
    if set(v1_rows) != set(v2_rows):
        raise ValueError("v1/v2 sample sets differ")
    exact_matches = sum(v1_rows[sample_id]["winning_response_id"] == v2_rows[sample_id]["winning_response_id"] for sample_id in v1_rows)

    v1_agreement = read_json(V1_AGREEMENT)
    v2_agreement = read_json(V2_AGREEMENT)
    summary = {
        "name": "step3cr_closure_summary_v1",
        "git_head": git_head(),
        "status": "cr0_closed_after_two_complete_measurement_gate_failures",
        "session_1": {
            "pair_preference": v1_agreement["pair_preference"],
            "specific_safe_next_step": v1_agreement["specific_safe_next_step"],
            "overall_pass": v1_agreement["gates"]["overall_pass"],
            "descriptive_role_join": summarize_session(v1_rows),
        },
        "session_2": {
            "pair_preference": v2_agreement["pair_preference"],
            "specific_safe_next_step": v2_agreement["specific_safe_next_step"],
            "overall_pass": v2_agreement["gates"]["overall_pass"],
            "descriptive_role_join": summarize_session(v2_rows),
        },
        "cross_session_final_winning_response": {
            "matches": exact_matches,
            "n": len(v1_rows),
            "agreement": exact_matches / len(v1_rows),
            "post_adjudication_descriptive_only": True,
        },
        "claim_boundary": {
            "measurement_validated": False,
            "c3_diagnostics_run": False,
            "c3_training_run": False,
            "c3_method_falsified": False,
            "constant20_quality_gap_descriptive_only": True,
            "heldout_opened": False,
            "optimizer_steps": 0,
        },
        "inputs": [{"path": repo_path(path), "sha256": sha256(path)} for path in inputs],
    }
    write_json(SUMMARY, summary)

    v1 = summary["session_1"]["descriptive_role_join"]
    v2 = summary["session_2"]["descriptive_role_join"]
    report = f"""# Step3C-R Closure Synthesis

Status: C-R0 closed after two complete prompt-aware pair-preference reliability failures.

## Confirmatory Route Result

| session | preference raw | preference AC1 | specificity raw | specificity AC1 | overall |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | {v1_agreement['pair_preference']['raw_agreement']:.3f} | {v1_agreement['pair_preference']['gwet_ac1']:.3f} | {v1_agreement['specific_safe_next_step']['raw_agreement']:.3f} | {v1_agreement['specific_safe_next_step']['gwet_ac1']:.3f} | fail |
| v2 | {v2_agreement['pair_preference']['raw_agreement']:.3f} | {v2_agreement['pair_preference']['gwet_ac1']:.3f} | {v2_agreement['specific_safe_next_step']['raw_agreement']:.3f} | {v2_agreement['specific_safe_next_step']['gwet_ac1']:.3f} | fail |

The frozen preference raw-agreement gate was `>=0.85`. The complete revision was used before v2, so no third C-R0 session is authorized. Specificity was reproducible, but the holistic preference construct was not validated under the registered conjunction of gates.

## Descriptive Signal After Role Join

Among the 29 pairs containing a clean-safe constant20 response and a natural target:

| session | natural wins | constant20 wins | ties | constant20 specific | natural specific |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | {v1['natural_target_wins']} | {v1['constant20_wins']} | {v1['ties']} | {v1['constant20_specific_yes_in_paired_clean_safe']}/29 | {v1['natural_specific_yes_in_paired_clean_safe']}/29 |
| v2 | {v2['natural_target_wins']} | {v2['constant20_wins']} | {v2['ties']} | {v2['constant20_specific_yes_in_paired_clean_safe']}/29 | {v2['natural_specific_yes_in_paired_clean_safe']}/29 |

The exact post-adjudication winning response matched across sessions for {exact_matches}/40 pairs ({exact_matches / 40:.3f}). These joined outcomes are exploratory because the primary measurement gate failed.

## Interpretation

The stopped experiment supports three bounded conclusions:

1. A decomposed specific-safe judgment is more reproducible than a single holistic pair preference in this packet.
2. Constant20 leaves descriptive response-quality headroom relative to approved natural targets.
3. C3 was never diagnosed or trained, so this result neither supports nor falsifies prompt-response contrastive binding.

The next study must use a new prospectively frozen construct-development and development-untouched validation design. It may not lower the old threshold, reuse C-R0 labels as new validation labels, or call a third C-R0 review a replication.

## Boundaries

- Heldout opened: false
- Optimizer steps: 0
- C3 diagnostics run: false
- C3 training run: false
- Future response-quality endpoint frozen: false
"""
    write_text(REPORT, report)

    manifest = {
        "name": "step3cr_closure_manifest_v1",
        "git_head": git_head(),
        "artifacts": [
            {"path": repo_path(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()), "bytes": Path(__file__).resolve().stat().st_size},
            {"path": repo_path(SUMMARY), "sha256": sha256(SUMMARY), "bytes": SUMMARY.stat().st_size},
            {"path": repo_path(REPORT), "sha256": sha256(REPORT), "bytes": REPORT.stat().st_size},
        ],
        "heldout_opened": False,
        "optimizer_steps": 0,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
