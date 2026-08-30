from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
REGISTRY_PATH = PAPER_ROOT / "configs" / "study_registry.json"


@dataclass
class Finding:
    level: str
    code: str
    message: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def recursively_contains(value: Any, needle: str) -> bool:
    needle = needle.casefold()
    if isinstance(value, str):
        return needle in value.casefold()
    if isinstance(value, dict):
        return any(recursively_contains(key, needle) or recursively_contains(item, needle) for key, item in value.items())
    if isinstance(value, list):
        return any(recursively_contains(item, needle) for item in value)
    return False


def target_steps(config: dict[str, Any]) -> tuple[int, list[int]]:
    train_rows = int(config["train_per_family"])
    updates_per_epoch = math.ceil(
        train_rows
        / int(config["per_device_train_batch_size"])
        / int(config["gradient_accumulation_steps"])
    )
    total_steps = max(1, int(math.ceil(updates_per_epoch * float(config["num_train_epochs"]))))
    targets = {
        max(0, min(total_steps, int(round(total_steps * float(fraction)))))
        for fraction in config["checkpoint_fractions"]
    }
    targets.update({0, total_steps})
    return total_steps, sorted(targets)


def scheduled_weight(arm: dict[str, Any], step: int) -> float:
    for window in arm.get("anchor_schedule") or []:
        start = int(window.get("start_step", 0))
        end = window.get("end_step")
        if step >= start and (end is None or step < int(end)):
            if "anchor_mix_start" in window and "anchor_mix_end" in window:
                if end is None or int(end) <= start:
                    return float(window["anchor_mix_end"])
                progress = (step - start) / max(1, int(end) - start)
                start_mix = float(window["anchor_mix_start"])
                end_mix = float(window["anchor_mix_end"])
                return start_mix + progress * (end_mix - start_mix)
            return float(window.get("anchor_mix", 0.0))
    return 0.0


def schedule_vector(arm: dict[str, Any], total_steps: int) -> list[float]:
    return [scheduled_weight(arm, step) for step in range(total_steps)]


def add(findings: list[Finding], condition: bool, code: str, message: str, *, warning: bool = False) -> None:
    if condition:
        return
    findings.append(Finding("warning" if warning else "error", code, message))


def validate_source_manifest(findings: list[Finding], manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    for source in manifest.get("sources", []):
        relative = Path(source["path"])
        path = REPO_ROOT / relative
        required = bool(source.get("required", True))
        add(findings, path.exists(), "source.missing", f"Missing source: {relative}", warning=not required)
        if not path.exists():
            continue
        expected_blob = source.get("git_blob_sha")
        if expected_blob:
            actual_blob = git_blob_sha(path)
            add(
                findings,
                actual_blob == expected_blob,
                "source.git_blob_mismatch",
                f"Git blob mismatch for {relative}: expected {expected_blob}, got {actual_blob}",
            )
        expected_sha256 = source.get("sha256")
        if expected_sha256:
            actual_sha256 = file_sha256(path)
            add(
                findings,
                actual_sha256 == expected_sha256,
                "source.sha256_mismatch",
                f"SHA-256 mismatch for {relative}: expected {expected_sha256}, got {actual_sha256}",
            )


def validate_config(findings: list[Finding], path: Path, registry: dict[str, Any]) -> dict[str, list[float]]:
    config = load_json(path)
    total_steps, checkpoints = target_steps(config)
    expected_steps = int(registry["expected_total_steps"])
    add(findings, total_steps == expected_steps, "config.total_steps", f"{path}: expected {expected_steps} steps, got {total_steps}")
    add(
        findings,
        checkpoints == list(registry["registered_checkpoints"]),
        "config.checkpoints",
        f"{path}: checkpoint targets {checkpoints} do not match registry {registry['registered_checkpoints']}",
    )
    add(
        findings,
        list(config.get("paper_registered_steps", [])) == list(registry["registered_checkpoints"]),
        "config.paper_registered_steps",
        f"{path}: paper_registered_steps do not match registry",
    )
    add(findings, not recursively_contains(config, "heldout"), "config.heldout_reference", f"{path}: formal config must not reference heldout")

    arms = config.get("intervention_arms", {})
    vectors: dict[str, list[float]] = {}
    for arm_name, arm in arms.items():
        weights = schedule_vector(arm, total_steps)
        vectors[arm_name] = weights
        add(findings, all(math.isfinite(value) for value in weights), "schedule.nonfinite", f"{path}:{arm_name} has non-finite weight")
        add(findings, all(0.0 <= value <= 1.0 for value in weights), "schedule.range", f"{path}:{arm_name} has weight outside [0,1]")
        add(findings, bool(arm.get("anchor_source_path")), "schedule.anchor_source", f"{path}:{arm_name} is missing anchor_source_path")

    design = config.get("paper_design")
    if design:
        expected_sum = float(design["expected_cumulative_anchor_weight"])
        expected_mean = float(design["expected_mean_anchor_weight"])
        expected_risk = float(design["expected_cumulative_risk_weight"])
        for arm_name, weights in vectors.items():
            add(findings, abs(sum(weights) - expected_sum) <= 1e-12, "schedule.anchor_sum", f"{arm_name}: sum={sum(weights):.15g}, expected={expected_sum}")
            add(findings, abs(sum(weights) / len(weights) - expected_mean) <= 1e-12, "schedule.anchor_mean", f"{arm_name}: mean mismatch")
            add(findings, abs(sum(1.0 - value for value in weights) - expected_risk) <= 1e-12, "schedule.risk_sum", f"{arm_name}: risk-weight sum mismatch")
            if design.get("require_positive_anchor_every_step"):
                add(findings, all(value > 0.0 for value in weights), "schedule.zero_step", f"{arm_name}: contains a zero-anchor step")
        sources = {str(arm.get("anchor_source_path", "")) for arm in arms.values()}
        if design.get("require_identical_anchor_source"):
            add(findings, len(sources) == 1, "schedule.source_mismatch", f"Equal-dose arms use different sources: {sorted(sources)}")

        required = {"front_loaded_equal_dose", "uniform_equal_dose", "back_loaded_equal_dose"}
        add(findings, required.issubset(vectors), "schedule.required_arms", f"Missing equal-dose arms: {sorted(required - set(vectors))}")
        if required.issubset(vectors):
            front = vectors["front_loaded_equal_dose"]
            uniform = vectors["uniform_equal_dose"]
            back = vectors["back_loaded_equal_dose"]
            midpoint = len(front) // 2
            add(findings, sum(front[:midpoint]) > sum(front[midpoint:]), "schedule.front_shape", "front-loaded schedule is not front loaded")
            add(findings, sum(back[:midpoint]) < sum(back[midpoint:]), "schedule.back_shape", "back-loaded schedule is not back loaded")
            add(findings, max(uniform) - min(uniform) <= 1e-15, "schedule.uniform_shape", "uniform schedule is not constant")
            add(findings, all(abs(front[index] - back[-1 - index]) <= 1e-15 for index in range(len(front))), "schedule.mirror", "front/back schedules are not exact mirrors")
    return vectors


def required_paper_files() -> Iterable[Path]:
    relative_paths = [
        "README.md",
        "STATUS.md",
        "CLAIMS.md",
        "EVIDENCE_MAP.md",
        "ROADMAP.md",
        "DATA_AND_CODE_MAP.md",
        "protocols/P0_claim_freeze.md",
        "protocols/P1_equal_dose_temporal_distribution.md",
        "protocols/P2_multiseed_replication.md",
        "protocols/P3_quality_remeasurement.md",
        "protocols/P4_human_validation.md",
        "protocols/P5_second_setting.md",
        "protocols/P6_confirmatory_panel.md",
        "verification/OPERATIONS.md",
        "verification/RELEASE_GATES.md",
        "verification/REVIEWER_ATTACKS.md",
    ]
    return [PAPER_ROOT / value for value in relative_paths]


def release_files() -> list[Path]:
    relative = [
        "data/derived/run_metrics.csv",
        "data/derived/prompt_level_safety.csv",
        "data/derived/prompt_level_quality.csv",
        "data/derived/gradient_contributions.csv",
        "data/derived/human_validation.csv",
        "data/derived/integrity_report.json",
        "figures/generated/fig1a_subspace_signature.pdf",
        "figures/generated/fig1b_posthoc_control_failure.pdf",
        "figures/generated/fig2_safety_trajectories.pdf",
        "figures/generated/fig3_seed_level_effects.pdf",
        "figures/generated/fig4_temporal_gradient_contribution.pdf",
        "figures/generated/fig5_safety_quality_frontier.pdf",
        "latex/build/main.pdf",
    ]
    return [PAPER_ROOT / value for value in relative]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Paper A study state.")
    parser.add_argument("--release", action="store_true", help="Require final release artifacts and write a release manifest.")
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args()

    findings: list[Finding] = []
    add(findings, REGISTRY_PATH.exists(), "registry.missing", f"Missing registry: {REGISTRY_PATH}")
    if not REGISTRY_PATH.exists():
        return 2
    registry = load_json(REGISTRY_PATH)

    for path in required_paper_files():
        add(findings, path.exists(), "paper.required_file", f"Missing required paper file: {path.relative_to(REPO_ROOT)}")

    source_manifest = REPO_ROOT / registry["source_manifest"]
    add(findings, source_manifest.exists(), "source.manifest_missing", f"Missing source manifest: {source_manifest}")
    if source_manifest.exists():
        validate_source_manifest(findings, source_manifest)

    seen_arms: set[str] = set()
    all_vectors: dict[str, list[float]] = {}
    for item in registry.get("formal_configs", []):
        path = REPO_ROOT / item["path"]
        add(findings, path.exists(), "config.missing", f"Missing formal config: {item['path']}")
        if not path.exists():
            continue
        config = load_json(path)
        config_arms = set(config.get("intervention_arms", {}))
        requested_arms = set(item.get("arms", []))
        add(findings, requested_arms.issubset(config_arms), "config.registry_arm", f"Registry arms missing from {item['path']}: {sorted(requested_arms - config_arms)}")
        add(findings, not (seen_arms & requested_arms), "config.duplicate_arm", f"Arm appears in multiple formal configs: {sorted(seen_arms & requested_arms)}")
        seen_arms.update(requested_arms)
        add(findings, list(item.get("seeds", [])) == list(config.get("seeds", {}).get("risk", [])), "config.seed_mismatch", f"Registry/config seed mismatch for {item['path']}")
        vectors = validate_config(findings, path, registry)
        all_vectors.update(vectors)

    quality_path = REPO_ROOT / registry["quality_rubric"]
    human_path = REPO_ROOT / registry["human_validation_registry"]
    add(findings, quality_path.exists(), "rubric.missing", f"Missing quality rubric: {quality_path}")
    add(findings, human_path.exists(), "human_registry.missing", f"Missing human registry: {human_path}")
    if quality_path.exists():
        quality = load_json(quality_path)
        expected_components = {
            "safety_correctness",
            "prompt_binding",
            "safe_actionability",
            "detail_retention",
            "naturalness",
            "generic_only_refusal",
            "invalid_output",
        }
        add(findings, expected_components == set(quality.get("components", {})), "rubric.components", "Quality rubric component set differs from the frozen contract")
        add(findings, quality.get("reviewer_may_assign_derived_score") is False, "rubric.direct_score", "Reviewers must not assign the derived score")

    if args.release:
        for path in release_files():
            add(findings, path.exists(), "release.missing", f"Missing release artifact: {path.relative_to(REPO_ROOT)}")

    errors = [finding for finding in findings if finding.level == "error"]
    warnings = [finding for finding in findings if finding.level == "warning"]
    report = {
        "schema_version": "paper_a_validation_report_v1",
        "repository_root": str(REPO_ROOT),
        "registry": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "errors": [finding.__dict__ for finding in errors],
        "warnings": [finding.__dict__ for finding in warnings],
        "schedule_summaries": {
            name: {
                "steps": len(values),
                "sum": sum(values),
                "mean": sum(values) / len(values),
                "first_half_sum": sum(values[: len(values) // 2]),
                "second_half_sum": sum(values[len(values) // 2 :]),
            }
            for name, values in all_vectors.items()
        },
        "release_mode": bool(args.release),
        "valid": not errors,
    }

    if args.json_report:
        report_path = args.json_report if args.json_report.is_absolute() else REPO_ROOT / args.json_report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.release and not errors:
        manifest = {
            "schema_version": "paper_a_release_manifest_v1",
            "files": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(release_files())
            ],
        }
        output = PAPER_ROOT / "data" / "derived" / "release_manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for finding in findings:
        print(f"[{finding.level.upper()}] {finding.code}: {finding.message}")
    print(json.dumps({"valid": not errors, "errors": len(errors), "warnings": len(warnings)}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
