from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl


ROOT = Path("experiments/step2_mechanism_dynamics/results/4070ti_step2ir")
VALIDATION_DIR = ROOT / "validation"
TARGET_MANIFEST = Path("experiments/step2_mechanism_dynamics/manifests/step2ir_diverse_matched_targets.jsonl")
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def stable_id(prefix: str, parts: list[str]) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def write_behavior_adjudication_inputs() -> tuple[Path, Path, int]:
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "generations_full100_core").glob("*.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            response_id = stable_id(
                "step2ir_resp",
                [
                    str(row.get("run_label", "")),
                    str(row.get("checkpoint", "")),
                    str(row.get("sample_id", "")),
                    str(row.get("response", "")),
                ],
            )
            blinded_rows.append(
                {
                    "response_id": response_id,
                    "prompt": row["prompt"],
                    "response": row.get("response", ""),
                    "judge_label": "",
                    "prompt_relevant_safe_redirect": "",
                    "generic_refusal_only": "",
                    "actionable_harmful_content": "",
                    "reason_code": "",
                }
            )
            key_rows.append(
                {
                    "response_id": response_id,
                    "arm": row.get("run_label", ""),
                    "checkpoint": row.get("checkpoint", ""),
                    "sample_id": row.get("sample_id", ""),
                    "seed": row.get("seed", ""),
                    "source_file": path.name,
                }
            )
    blinded_rows.sort(key=lambda row: row["response_id"])
    key_rows.sort(key=lambda row: row["response_id"])
    blinded_path = VALIDATION_DIR / "step2ir_behavior_adjudication_blinded.jsonl"
    key_path = VALIDATION_DIR / "step2ir_behavior_adjudication_key.jsonl"
    write_jsonl(blinded_path, blinded_rows)
    write_jsonl(key_path, key_rows)
    return blinded_path, key_path, len(blinded_rows)


def write_target_review_inputs() -> tuple[Path, Path, int]:
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for row in read_jsonl(TARGET_MANIFEST):
        review_id = stable_id("step2ir_tgt", [row["target_id"], row["prompt"], row["response"]])
        blinded_rows.append(
            {
                "target_review_id": review_id,
                "prompt": row["prompt"],
                "target_response": row["response"],
                "explicit_refusal_validity": "",
                "no_harmful_procedural_leakage": "",
                "prompt_relevant_redirect": "",
                "grammatical_naturalness": "",
                "semantic_coherence": "",
                "keyword_stuffing": "",
                "mechanical_filler": "",
                "overall_decision": "",
                "review_reason": "",
            }
        )
        key_rows.append(
            {
                "target_review_id": review_id,
                "target_id": row["target_id"],
                "sample_id": row["sample_id"],
                "category": row.get("category", ""),
                "target_hash": row.get("target_hash", ""),
            }
        )
    blinded_rows.sort(key=lambda row: row["target_review_id"])
    key_rows.sort(key=lambda row: row["target_review_id"])
    blinded_path = VALIDATION_DIR / "step2ir_target_quality_review_blinded.jsonl"
    key_path = VALIDATION_DIR / "step2ir_target_quality_review_key.jsonl"
    write_jsonl(blinded_path, blinded_rows)
    write_jsonl(key_path, key_rows)
    return blinded_path, key_path, len(blinded_rows)


def semantic_dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "embedding_model": SEMANTIC_MODEL,
        "status": "pending_unavailable",
        "sentence_transformers_importable": False,
        "model_loadable_offline": False,
        "note": "Per plan, no replacement embedding model is selected after viewing Step2I-R arm results.",
    }
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        status["sentence_transformers_importable"] = True
        try:
            SentenceTransformer(SEMANTIC_MODEL, local_files_only=True)
            status["model_loadable_offline"] = True
            status["status"] = "ready"
        except Exception as exc:
            status["model_load_error"] = repr(exc)
    except Exception as exc:
        status["import_error"] = repr(exc)
    return status


def write_status_report(
    behavior_count: int,
    target_count: int,
    semantic_status: dict[str, Any],
) -> Path:
    report = VALIDATION_DIR / "step2ir_candidate_validation_status.md"
    report.write_text(
        f"""# Step2I-R Candidate Validation Status

## Prepared Inputs

| artifact | rows | status |
| --- | ---: | --- |
| behavior adjudication blinded table | {behavior_count} | ready for independent judge |
| target-quality review blinded table | {target_count} | ready for independent judge/human review |
| semantic-template audit | 0 | {semantic_status['status']} |

## Semantic Audit Dependency

```json
{json.dumps(semantic_status, indent=2, ensure_ascii=False)}
```

## Notes

Behavior adjudication inputs hide arm, schedule, checkpoint, current heuristic labels, and prior report decisions. Separate key files are stored for later analysis after adjudication is complete.

Target-quality review inputs hide model outcomes and contain only prompt plus target response. The current local preflight decisions remain non-independent and are not treated as approval evidence.
""",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    behavior_path, behavior_key, behavior_count = write_behavior_adjudication_inputs()
    target_path, target_key, target_count = write_target_review_inputs()
    semantic_status = semantic_dependency_status()
    semantic_path = VALIDATION_DIR / "step2ir_semantic_template_dependency_status.json"
    semantic_path.write_text(json.dumps(semantic_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = write_status_report(behavior_count, target_count, semantic_status)
    print(f"wrote {behavior_path} ({behavior_count} rows)")
    print(f"wrote {behavior_key}")
    print(f"wrote {target_path} ({target_count} rows)")
    print(f"wrote {target_key}")
    print(f"wrote {semantic_path}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
