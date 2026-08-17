"""Run seed-5 jackknife convex-hull robust MCD without optimizer steps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_cita_p0 as model_common
import step3_etarc_p0 as etarc_p0
import step3_frsc_common as frsc
import step3_jcrmcd_common as jcrmcd
import step3_ojmcd_p0 as optimizer_unit
import step3_qetarc_common as gradient_ops
import step3_ramp_common as gradient_math
import step3_ramp_p0 as ramp_p0
import step3_rpfrsc_p0 as prior_runner
import step3_tarc_p0 as tarc_p0
import step3ct_training_common as training
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_jcrmcd_p0/protocol/jcrmcd_p0_protocol.json"
)
CERTIFICATES = optimizer_unit.CERTIFICATES
SAFETY_OBJECTIVES = optimizer_unit.SAFETY_OBJECTIVES
ARMS = ("baseline", "raw_mcd", "jcrmcd_mcd")
PROTECTED_CERTIFICATES = ("risk", "uniform", "tarc", "continuation")
TARGET_CERTIFICATES = (
    "etarc",
    "branch_entropy",
    "branch_negative_gap",
)
PATH_FRACTIONS = tuple(index / 32.0 for index in range(33))
PROTECTED_RETENTION_FLOOR = 0.95
SUPPORT_COUNT = 4
SUPPORT_SIZE = 8
DIAGNOSTIC_UNITS = 8
DIAGNOSTIC_QUOTAS = (
    ("cyber abuse", 1),
    ("financial evasion", 2),
    ("fraud or deception", 1),
    ("harmful wrongdoing", 2),
    ("public manipulation", 2),
)
DIAGNOSTIC_SALT = "jcrmcd-seed5-diagnostic-v1"
DIAGNOSTIC_UNIT_SALT = "jcrmcd-seed5-diagnostic-unit-v1"
SUPPORT_SALT = "jcrmcd-seed5-support-v1"


class JcrmcdP0ValidationError(ValueError):
    """Raised when JCR-MCD P0 violates its prospective protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "verify-run"])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise JcrmcdP0ValidationError(f"path escapes repository: {value}") from exc
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_category_registry(targets: list[dict[str, Any]]) -> dict[str, str]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in targets:
        rows_by_id.setdefault(str(row["sample_id"]), []).append(row)
    if len(rows_by_id) != 40 or any(len(rows) != 2 for rows in rows_by_id.values()):
        raise JcrmcdP0ValidationError("natural target prompt-pair registry differs")
    registry = {}
    for sample_id, rows in rows_by_id.items():
        categories = {str(row["category"]) for row in rows}
        if len(categories) != 1:
            raise JcrmcdP0ValidationError("target-pair category differs")
        registry[sample_id] = next(iter(categories))
    return registry


def expected_safety_partition(
    targets: list[dict[str, Any]],
) -> tuple[list[list[str]], list[dict[str, str]]]:
    registry = prompt_category_registry(targets)
    by_category: dict[str, list[str]] = {}
    for sample_id, category in registry.items():
        by_category.setdefault(category, []).append(sample_id)

    diagnostic_ids: list[str] = []
    for category, quota in DIAGNOSTIC_QUOTAS:
        available = sorted(
            by_category.get(category, []),
            key=lambda sample_id: _stable_digest(
                f"{DIAGNOSTIC_SALT}|{category}|{sample_id}"
            ),
        )
        if len(available) < quota:
            raise JcrmcdP0ValidationError("supported diagnostic quota is impossible")
        diagnostic_ids.extend(available[:quota])
    if len(set(diagnostic_ids)) != DIAGNOSTIC_UNITS:
        raise JcrmcdP0ValidationError("diagnostic identity selection differs")
    diagnostic_ids.sort(
        key=lambda sample_id: _stable_digest(
            f"{DIAGNOSTIC_UNIT_SALT}|{registry[sample_id]}|{sample_id}"
        )
    )

    diagnostic_set = set(diagnostic_ids)
    construction_ids = sorted(
        (sample_id for sample_id in registry if sample_id not in diagnostic_set),
        key=lambda sample_id: _stable_digest(
            f"{SUPPORT_SALT}|{registry[sample_id]}|{sample_id}"
        ),
    )
    if len(construction_ids) != SUPPORT_COUNT * SUPPORT_SIZE:
        raise JcrmcdP0ValidationError("construction prompt count differs")
    supports = [[] for _ in range(SUPPORT_COUNT)]
    category_counts = [Counter() for _ in range(SUPPORT_COUNT)]
    for sample_id in construction_ids:
        category = registry[sample_id]
        available = [
            index for index, support in enumerate(supports) if len(support) < SUPPORT_SIZE
        ]
        if not available:
            raise JcrmcdP0ValidationError("construction support capacity exhausted")
        selected = min(
            available,
            key=lambda index: (
                category_counts[index][category],
                len(supports[index]),
                index,
            ),
        )
        supports[selected].append(sample_id)
        category_counts[selected][category] += 1
    if any(len(support) != SUPPORT_SIZE for support in supports):
        raise JcrmcdP0ValidationError("construction support sizes differ")
    diagnostics = [
        {"sample_id": sample_id, "category": registry[sample_id]}
        for sample_id in diagnostic_ids
    ]
    return supports, diagnostics


def expected_selection(
    targets: list[dict[str, Any]], risk_order: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supports, diagnostics = expected_safety_partition(targets)
    if len(risk_order) < SUPPORT_COUNT * SUPPORT_SIZE + DIAGNOSTIC_UNITS:
        raise JcrmcdP0ValidationError("risk permutation is too short")
    support_rows = [
        {
            "support_id": f"support_{index + 1:02d}",
            "risk_row_indices": risk_order[
                index * SUPPORT_SIZE : (index + 1) * SUPPORT_SIZE
            ],
            "safety_sample_ids": sample_ids,
        }
        for index, sample_ids in enumerate(supports)
    ]
    offset = SUPPORT_COUNT * SUPPORT_SIZE
    diagnostic_rows = [
        {
            "unit_id": f"diagnostic_{index + 1:02d}",
            "risk_row_index": risk_order[offset + index],
            **diagnostic,
        }
        for index, diagnostic in enumerate(diagnostics)
    ]
    return support_rows, diagnostic_rows


def context(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = etarc_p0.read_json(path)
    if protocol.get("status") != "prospective_before_any_JCRMCD_model_forward":
        raise JcrmcdP0ValidationError("JCR-MCD protocol is not prospective")
    for name, artifact in protocol["frozen_inputs"].items():
        artifact_path = resolve_repo_path(artifact["path"])
        if (
            not artifact_path.is_file()
            or etarc_p0.sha256(artifact_path) != artifact["sha256"]
        ):
            raise JcrmcdP0ValidationError(f"frozen input hash mismatch: {name}")
    if (
        protocol.get("optimizer_objects") != 0
        or protocol.get("scheduler_objects") != 0
        or protocol.get("optimizer_steps") != 0
        or protocol.get("virtual_parameter_evaluations") != 24
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_generation_authorized") is not False
        or protocol.get("core100_opened") is not False
        or protocol.get("heldout_opened") is not False
    ):
        raise JcrmcdP0ValidationError("protocol violates the zero-step boundary")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["entry_commit"], "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise JcrmcdP0ValidationError("entry commit is not an ancestor of HEAD")

    prior = etarc_p0.read_json(
        resolve_repo_path(protocol["prior_result"]["decision_path"])
    )
    if prior.get("route") != "STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED":
        raise JcrmcdP0ValidationError("registered RP-PFRSC precursor route differs")
    design = protocol["design"]
    if (
        tuple(design["certificates"]) != CERTIFICATES
        or tuple(design["protected_certificates"]) != PROTECTED_CERTIFICATES
        or tuple(design["target_certificates"]) != TARGET_CERTIFICATES
        or tuple(float(value) for value in design["path_fractions"])
        != PATH_FRACTIONS
        or float(design["protected_retention_floor"])
        != PROTECTED_RETENTION_FLOOR
        or int(design["construction_supports"]) != SUPPORT_COUNT
        or int(design["pairs_per_support"]) != SUPPORT_SIZE
        or int(design["diagnostic_units"]) != DIAGNOSTIC_UNITS
        or int(design["path_candidates"]) != len(PATH_FRACTIONS)
    ):
        raise JcrmcdP0ValidationError("JCR-MCD design registry differs")

    config = etarc_p0.read_json(resolve_repo_path(protocol["config_path"]))
    if (
        int(config["seed"]) != 5
        or config.get("required_gpu_name_substring") != "RTX 4070 Ti"
        or config.get("model_optimizer_objects_authorized") is not False
        or config.get("scheduler_objects_authorized") is not False
        or int(config["optimizer_steps_authorized"]) != 0
        or config.get("transient_virtual_parameter_evaluations_authorized")
        is not True
        or config.get("training_authorized") is not False
        or float(config["weight_decay"]) != 0.0
        or int(config["gradient_accumulation_steps"]) != SUPPORT_SIZE
    ):
        raise JcrmcdP0ValidationError("seed-5 virtual-step config differs")

    targets = etarc_p0.read_jsonl(resolve_repo_path(config["safety_targets_path"]))
    ramp_p0.validate_source_rows(targets, 80, 40, 8)
    risk_rows = training.read_jsonl(
        resolve_repo_path(config["risk_data_path"]), limit=int(config["risk_rows"])
    )
    if len(risk_rows) != int(config["risk_rows"]):
        raise JcrmcdP0ValidationError("risk row count differs")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(protocol["selection"]["risk_order_seed"]))
    risk_order = torch.randperm(len(risk_rows), generator=generator).tolist()
    expected_supports, expected_diagnostics = expected_selection(targets, risk_order)
    if protocol["selection"]["construction_supports"] != expected_supports:
        raise JcrmcdP0ValidationError("construction support registry differs")
    if protocol["selection"]["diagnostic_units"] != expected_diagnostics:
        raise JcrmcdP0ValidationError("diagnostic unit registry differs")

    registry = prompt_category_registry(targets)
    category_counts = dict(sorted(Counter(registry.values()).items()))
    if category_counts != protocol["source"]["category_prompt_counts"]:
        raise JcrmcdP0ValidationError("source category counts differ")
    support_ids = [
        sample_id
        for support in expected_supports
        for sample_id in support["safety_sample_ids"]
    ]
    diagnostic_ids = [item["sample_id"] for item in expected_diagnostics]
    support_risk = [
        index
        for support in expected_supports
        for index in support["risk_row_indices"]
    ]
    diagnostic_risk = [item["risk_row_index"] for item in expected_diagnostics]
    if (
        set(support_ids).intersection(diagnostic_ids)
        or set(support_risk).intersection(diagnostic_risk)
        or set(support_ids + diagnostic_ids) != set(registry)
        or len(set(support_risk + diagnostic_risk)) != 40
        or support_risk + diagnostic_risk != risk_order[:40]
    ):
        raise JcrmcdP0ValidationError("construction/diagnostic split overlaps or drifts")
    return protocol, config, targets, risk_rows


def require_clean_tracked_freeze(path: Path, protocol: dict[str, Any]) -> str:
    if etarc_p0.git_output("status", "--porcelain"):
        raise JcrmcdP0ValidationError("JCR-MCD P0 requires a clean worktree")
    tracked = [artifact["path"] for artifact in protocol["frozen_inputs"].values()]
    tracked.extend([protocol["config_path"], relative(path)])
    for item in set(tracked):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    for output in protocol["outputs"].values():
        if resolve_repo_path(output).exists():
            raise JcrmcdP0ValidationError(f"prospective output already exists: {output}")
    return etarc_p0.git_output("rev-parse", "HEAD")


def structural_features(
    tokenizer: Any,
    targets: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pairs, audit = tarc_p0.build_pair_features(
        tokenizer,
        targets,
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    expected = etarc_p0.read_json(
        resolve_repo_path(protocol["source"]["structural_audit_path"])
    )
    if audit != expected:
        raise JcrmcdP0ValidationError("route structure differs from frozen source")
    return {str(pair["sample_id"]): pair for pair in pairs}, audit


def preflight(path: Path) -> dict[str, Any]:
    import torch

    protocol, config, targets, _ = context(path)
    if not torch.cuda.is_available():
        raise JcrmcdP0ValidationError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name(0)
    if str(config["required_gpu_name_substring"]) not in gpu:
        raise JcrmcdP0ValidationError(f"registered RTX 4070 Ti is unavailable: {gpu}")
    tokenizer = tarc_p0.load_tokenizer(str(config["model_name_or_path"]))
    pair_map, _ = structural_features(tokenizer, targets, config, protocol)
    selected = [
        sample_id
        for support in protocol["selection"]["construction_supports"]
        for sample_id in support["safety_sample_ids"]
    ] + [item["sample_id"] for item in protocol["selection"]["diagnostic_units"]]
    if not all(sample_id in pair_map for sample_id in selected):
        raise JcrmcdP0ValidationError("selected safety pair is absent")
    result = {
        "status": "PASS_JCRMCD_P0_PREFLIGHT",
        "protocol_sha256": etarc_p0.sha256(path),
        "seed": config["seed"],
        "certificates": list(CERTIFICATES),
        "protected_certificates": list(PROTECTED_CERTIFICATES),
        "target_certificates": list(TARGET_CERTIFICATES),
        "arms": list(ARMS),
        "construction_supports": SUPPORT_COUNT,
        "pairs_per_support": SUPPORT_SIZE,
        "diagnostic_units": DIAGNOSTIC_UNITS,
        "diagnostic_hull_checks": DIAGNOSTIC_UNITS * len(CERTIFICATES),
        "path_candidates": len(PATH_FRACTIONS),
        "virtual_parameter_evaluations": DIAGNOSTIC_UNITS * len(ARMS),
        "gpu": gpu,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def cpu_gradient_copy(gradient: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in gradient.items()
    }


def cpu_certificate_copy(
    gradients: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {name: cpu_gradient_copy(gradient) for name, gradient in gradients.items()}


def mean_certificate_gradients(
    supports: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if len(supports) != SUPPORT_COUNT or any(
        tuple(support) != CERTIFICATES for support in supports
    ):
        raise JcrmcdP0ValidationError("support certificate registry differs")
    result: dict[str, dict[str, Any]] = {}
    for certificate in CERTIFICATES:
        names = set(supports[0][certificate])
        if any(set(support[certificate]) != names for support in supports):
            raise JcrmcdP0ValidationError("support gradient dictionaries differ")
        result[certificate] = {
            name: sum(
                (support[certificate][name] for support in supports[1:]),
                supports[0][certificate][name].clone(),
            )
            / len(supports)
            for name in names
        }
    return result


def _baseline_gradient(
    gradients: dict[str, dict[str, Any]], safety_mix: float
) -> dict[str, Any]:
    return gradient_ops.add_gradients(
        gradient_ops.scaled_gradient(gradients["risk"], 1.0 - safety_mix),
        gradients["uniform"],
        right_scale=safety_mix,
    )


def support_uncertainty(
    supports: list[dict[str, dict[str, Any]]], tolerance: float
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    normalized_supports = []
    norm_rows = []
    for support in supports:
        units, norms = frsc.normalized_certificates(support)
        normalized_supports.append(units)
        norm_rows.append(norms)
    units_by_certificate = {
        certificate: [support[certificate] for support in normalized_supports]
        for certificate in CERTIFICATES
    }
    certificates = {}
    for certificate, units in units_by_certificate.items():
        gram = [
            [gradient_math.gradient_dot(left, right) for right in units]
            for left in units
        ]
        radius, calibration = jcrmcd.jackknife_convex_hull_radius(gram, tolerance)
        certificates[certificate] = {
            "support_gram_matrix": gram,
            "radius": radius,
            "calibration": calibration,
        }
    return units_by_certificate, {
        "calibration_rule": (
            "maximum_leave_one_support_distance_to_remaining_convex_hull"
        ),
        "support_gradient_norms": norm_rows,
        "certificates": certificates,
    }


def build_construction_geometry(
    supports: list[dict[str, dict[str, Any]]],
    support_coverages: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    tolerance = float(config["dual_numeric_tolerance"])
    pooled = mean_certificate_gradients(supports)
    raw_baseline = _baseline_gradient(pooled, float(config["safety_mix"]))
    budget = float(config["auxiliary_gradient_norm_ratio"]) * gradient_math.gradient_norm(
        pooled["uniform"]
    )
    raw_direction, target_direction, endpoint_audit = jcrmcd.endpoint_directions(
        pooled,
        list(protocol["design"]["target_certificates"]),
        budget,
        tolerance,
    )
    baseline_displacement, baseline_map = frsc.realized_displacement(
        raw_baseline,
        None,
        0.0,
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    reference_norm = gradient_math.gradient_norm(baseline_displacement)
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise JcrmcdP0ValidationError("pooled baseline displacement is degenerate")

    support_units, uncertainty = support_uncertainty(supports, tolerance)
    panels = {"pooled": pooled}
    panels.update(
        {f"support_{index + 1:02d}": support for index, support in enumerate(supports)}
    )
    scored = jcrmcd.score_path_panels(
        raw_baseline,
        raw_direction,
        target_direction,
        panels,
        list(protocol["design"]["path_fractions"]),
        budget,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        reference_norm,
        float(config["clip_epsilon"]),
        tolerance,
    )
    radii = {
        certificate: float(uncertainty["certificates"][certificate]["radius"])
        for certificate in CERTIFICATES
    }
    support_records = [scored[f"support_{index + 1:02d}"] for index in range(4)]
    selected_index, search = jcrmcd.select_robust_path_candidate(
        scored["pooled"],
        support_records,
        radii,
        list(protocol["design"]["protected_certificates"]),
        list(protocol["design"]["target_certificates"]),
        float(protocol["design"]["protected_retention_floor"]),
        tolerance=float(protocol["gates"]["robust_numeric_tolerance"]),
    )
    selected_fraction = float(PATH_FRACTIONS[selected_index])
    selected_direction = jcrmcd.path_direction(
        raw_direction, target_direction, selected_fraction, budget, tolerance
    )
    raw_displacement, raw_map = frsc.realized_displacement(
        raw_baseline,
        raw_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    selected_displacement, selected_map = frsc.realized_displacement(
        raw_baseline,
        selected_direction,
        float(config["safety_mix"]),
        float(config["max_grad_norm"]),
        float(config["adam_epsilon"]),
        float(config["clip_epsilon"]),
    )
    selected_record = scored["pooled"][selected_index]
    if (
        abs(
            gradient_math.gradient_norm(raw_displacement) / reference_norm
            - float(scored["pooled"][0]["displacement_norm_fraction"])
        )
        > 1e-9
        or abs(
            gradient_math.gradient_norm(selected_displacement) / reference_norm
            - float(selected_record["displacement_norm_fraction"])
        )
        > 1e-9
    ):
        raise JcrmcdP0ValidationError("fixed displacement drifted from path panel")

    construction = {
        "name": "step3_jcrmcd_p0_construction_geometry_v1",
        "compute_device": "cpu_float_preserving_gradient_copies",
        "support_coverages": support_coverages,
        "pooled_gradient_norms": {
            name: gradient_math.gradient_norm(gradient)
            for name, gradient in pooled.items()
        },
        "raw_baseline_gradient_norm": gradient_math.gradient_norm(raw_baseline),
        "auxiliary_budget_norm": budget,
        "reference_displacement_norm": reference_norm,
        "baseline_optimizer_map": baseline_map,
        "raw_optimizer_map": raw_map,
        "selected_optimizer_map": selected_map,
        "raw_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(raw_direction) - budget
        )
        / budget,
        "target_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(target_direction) - budget
        )
        / budget,
        "selected_direction_budget_relative_error": abs(
            gradient_math.gradient_norm(selected_direction) - budget
        )
        / budget,
        "endpoint_audit": endpoint_audit,
        "uncertainty": uncertainty,
        "search": search,
        "selected_index": selected_index,
        "selected_fraction": selected_fraction,
        "pooled_path_records": scored["pooled"],
        "support_path_records": {
            f"support_{index + 1:02d}": records
            for index, records in enumerate(support_records)
        },
        "path_candidates": len(PATH_FRACTIONS),
        "diagnostic_gradients_used_for_selection": False,
        "raw_gradient_tensors_serialized": False,
    }
    fixed_displacements = {
        "baseline": baseline_displacement,
        "raw_mcd": raw_displacement,
        "jcrmcd_mcd": selected_displacement,
    }
    del pooled, raw_baseline, raw_direction, target_direction, selected_direction
    return construction, fixed_displacements, support_units


def diagnostic_hull_audit(
    diagnostic_gradients: dict[str, dict[str, Any]],
    support_units: dict[str, list[dict[str, Any]]],
    construction: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    records = {}
    for certificate in CERTIFICATES:
        diagnostic_cpu = cpu_gradient_copy(diagnostic_gradients[certificate])
        norm = gradient_math.gradient_norm(diagnostic_cpu)
        if not math.isfinite(norm) or norm <= 0.0:
            raise JcrmcdP0ValidationError("diagnostic gradient is degenerate")
        unit = gradient_ops.scaled_gradient(diagnostic_cpu, 1.0 / norm)
        dots = [
            gradient_math.gradient_dot(support, unit)
            for support in support_units[certificate]
        ]
        gram = construction["uncertainty"]["certificates"][certificate][
            "support_gram_matrix"
        ]
        distance, projection = jcrmcd.convex_hull_distance_from_gram(
            gram, dots, 1.0, tolerance
        )
        radius = float(
            construction["uncertainty"]["certificates"][certificate]["radius"]
        )
        records[certificate] = {
            "diagnostic_gradient_norm": norm,
            "support_dot_products": dots,
            "distance_to_construction_hull": distance,
            "registered_radius": radius,
            "radius_excess": distance - radius,
            "covered": distance <= radius + tolerance,
            "projection_weights": projection["weights"],
            "projection_kkt_residual": projection["kkt_residual"],
        }
        del diagnostic_cpu, unit
    return {
        "certificates": records,
        "covered_certificates": sum(item["covered"] for item in records.values()),
        "all_certificates_covered": all(item["covered"] for item in records.values()),
    }


def device_displacement(
    displacement: dict[str, Any], parameters: list[tuple[str, Any]]
) -> dict[str, Any]:
    if set(displacement) != {name for name, _ in parameters}:
        raise JcrmcdP0ValidationError("fixed displacement registry differs")
    return {
        name: displacement[name].to(device=parameter.device, dtype=parameter.dtype)
        for name, parameter in parameters
    }


def evaluate_fixed_arm(
    arm: str,
    displacement: dict[str, Any],
    map_audit: dict[str, Any],
    diagnostic_gradients: dict[str, dict[str, Any]],
    initial_losses: dict[str, float],
    model: Any,
    parameters: list[tuple[str, Any]],
    initial_state: dict[str, Any],
    risk_features: dict[int, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
    diagnostic: dict[str, Any],
    config: dict[str, Any],
    reference_norm: float,
) -> dict[str, Any]:
    moved = device_displacement(displacement, parameters)
    score = frsc.displacement_scores(diagnostic_gradients, moved, reference_norm)
    prior_runner._apply_virtual_step(
        parameters, initial_state, moved, float(config["learning_rate"])
    )
    final_losses = prior_runner.evaluate_block_losses(
        model,
        risk_features,
        pair_map,
        [int(diagnostic["risk_row_index"])],
        [str(diagnostic["sample_id"])],
        config,
    )
    finite = frsc.finite_step_loss_metrics(
        initial_losses,
        final_losses,
        diagnostic_gradients,
        moved,
        float(config["learning_rate"]),
    )
    prior_runner._restore(parameters, initial_state)
    restored = prior_runner._state_equal(parameters, initial_state)
    del moved
    return {
        "arm": arm,
        "optimizer_map": map_audit,
        "diagnostic_scores": score,
        "finite_step": finite,
        "state_restored_after_virtual_step": restored,
    }


def _paired(values: list[float], protocol: dict[str, Any]) -> dict[str, Any]:
    if len(values) != DIAGNOSTIC_UNITS or any(
        not math.isfinite(value) for value in values
    ):
        raise JcrmcdP0ValidationError("paired diagnostic values are invalid")
    return {
        "differences": values,
        "positive_rate": sum(value > 0.0 for value in values) / len(values),
        "bootstrap": etarc_p0.paired_bootstrap(
            values,
            int(protocol["statistics"]["bootstrap_samples"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        ),
        "sign_test": etarc_p0.exact_sign_test(values),
    }


def _relative_paired(
    numerators: list[float],
    denominators: list[float],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if len(numerators) != DIAGNOSTIC_UNITS or len(denominators) != len(numerators):
        raise JcrmcdP0ValidationError("relative diagnostic values differ")
    invalid = [
        index
        for index, denominator in enumerate(denominators)
        if not math.isfinite(denominator) or denominator <= 0.0
    ]
    if invalid:
        return {
            "defined": False,
            "invalid_denominator_indices": invalid,
            "differences": [],
        }
    values = [
        float(numerator / denominator - 1.0)
        for numerator, denominator in zip(numerators, denominators, strict=True)
    ]
    return {"defined": True, **_paired(values, protocol)}


def summarize_diagnostics(
    rows: list[dict[str, Any]],
    construction: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != DIAGNOSTIC_UNITS:
        raise JcrmcdP0ValidationError("stored diagnostic unit count differs")
    targets = tuple(protocol["design"]["target_certificates"])
    diagnostic: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows]
        target_minima = [
            min(float(item["diagnostic_scores"]["scores"][name]) for name in targets)
            for item in arm_rows
        ]
        diagnostic[arm] = {
            "minimum_worst_score": min(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "mean_worst_score": statistics.fmean(
                float(item["diagnostic_scores"]["minimum_score"])
                for item in arm_rows
            ),
            "minimum_target_score": min(target_minima),
            "mean_target_minimum_score": statistics.fmean(target_minima),
            "minimum_certificate_scores": {
                certificate: min(
                    float(item["diagnostic_scores"]["scores"][certificate])
                    for item in arm_rows
                )
                for certificate in CERTIFICATES
            },
            "objectives": {},
        }
        for certificate in CERTIFICATES:
            objective_rows = [
                item["finite_step"]["objectives"][certificate]
                for item in arm_rows
            ]
            ratios = [
                float(item["actual_to_predicted_ratio"])
                for item in objective_rows
                if item["actual_to_predicted_ratio"] is not None
            ]
            diagnostic[arm]["objectives"][certificate] = {
                "mean_predicted_reduction": statistics.fmean(
                    float(item["predicted_first_order_reduction"])
                    for item in objective_rows
                ),
                "mean_actual_reduction": statistics.fmean(
                    float(item["actual_loss_reduction"]) for item in objective_rows
                ),
                "minimum_actual_reduction": min(
                    float(item["actual_loss_reduction"]) for item in objective_rows
                ),
                "positive_actual_rate": sum(
                    float(item["actual_loss_reduction"]) > 0.0
                    for item in objective_rows
                )
                / len(objective_rows),
                "mean_actual_to_predicted_ratio": statistics.fmean(ratios)
                if ratios
                else None,
            }

    coverage_by_certificate = {}
    for certificate in CERTIFICATES:
        items = [row["hull_audit"]["certificates"][certificate] for row in rows]
        coverage_by_certificate[certificate] = {
            "registered_radius": float(items[0]["registered_radius"]),
            "covered_units": sum(bool(item["covered"]) for item in items),
            "maximum_distance": max(
                float(item["distance_to_construction_hull"]) for item in items
            ),
            "maximum_radius_excess": max(float(item["radius_excess"]) for item in items),
        }
    coverage_checks = DIAGNOSTIC_UNITS * len(CERTIFICATES)
    covered_checks = sum(
        bool(row["hull_audit"]["certificates"][certificate]["covered"])
        for row in rows
        for certificate in CERTIFICATES
    )
    protected_slacks = [
        float(value)
        for row in rows
        for value in row["theorem_audit"]["protected_residual_slacks"].values()
    ]
    target_slacks = [
        float(value)
        for row in rows
        for value in row["theorem_audit"]["target_lower_bound_slacks"].values()
    ]
    coverage = {
        "checks": coverage_checks,
        "covered_checks": covered_checks,
        "coverage_rate": covered_checks / coverage_checks,
        "all_covered": covered_checks == coverage_checks,
        "by_certificate": coverage_by_certificate,
        "minimum_protected_residual_slack": min(protected_slacks),
        "minimum_target_lower_bound_slack": min(target_slacks),
    }

    worst = {
        arm: [
            float(row["arms"][arm]["diagnostic_scores"]["minimum_score"])
            for row in rows
        ]
        for arm in ARMS
    }
    paired: dict[str, Any] = {
        "jcrmcd_minus_baseline_worst_score": _paired(
            [
                left - right
                for left, right in zip(
                    worst["jcrmcd_mcd"], worst["baseline"], strict=True
                )
            ],
            protocol,
        ),
        "jcrmcd_minus_raw_worst_score": _paired(
            [
                left - right
                for left, right in zip(
                    worst["jcrmcd_mcd"], worst["raw_mcd"], strict=True
                )
            ],
            protocol,
        ),
        "jcrmcd_minus_raw_target_minimum_score": _paired(
            [
                min(
                    float(row["arms"]["jcrmcd_mcd"]["diagnostic_scores"]["scores"][name])
                    for name in targets
                )
                - min(
                    float(row["arms"]["raw_mcd"]["diagnostic_scores"]["scores"][name])
                    for name in targets
                )
                for row in rows
            ],
            protocol,
        ),
        "jcrmcd_relative_exact_score": {},
        "jcrmcd_relative_actual_reduction": {},
    }
    for certificate in CERTIFICATES:
        selected_exact = [
            float(row["arms"]["jcrmcd_mcd"]["diagnostic_scores"]["scores"][certificate])
            for row in rows
        ]
        raw_exact = [
            float(row["arms"]["raw_mcd"]["diagnostic_scores"]["scores"][certificate])
            for row in rows
        ]
        selected_actual = [
            float(
                row["arms"]["jcrmcd_mcd"]["finite_step"]["objectives"][certificate][
                    "actual_loss_reduction"
                ]
            )
            for row in rows
        ]
        raw_actual = [
            float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][certificate][
                    "actual_loss_reduction"
                ]
            )
            for row in rows
        ]
        paired["jcrmcd_relative_exact_score"][certificate] = _relative_paired(
            selected_exact, raw_exact, protocol
        )
        paired["jcrmcd_relative_actual_reduction"][certificate] = _relative_paired(
            selected_actual, raw_actual, protocol
        )

    target_numerators = [
        statistics.fmean(
            float(
                row["arms"]["jcrmcd_mcd"]["finite_step"]["objectives"][name][
                    "actual_loss_reduction"
                ]
            )
            / float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][name][
                    "actual_loss_reduction"
                ]
            )
            - 1.0
            for name in targets
        )
        for row in rows
        if all(
            float(
                row["arms"]["raw_mcd"]["finite_step"]["objectives"][name][
                    "actual_loss_reduction"
                ]
            )
            > 0.0
            for name in targets
        )
    ]
    if len(target_numerators) == DIAGNOSTIC_UNITS:
        paired["target_macro_actual_relative_gain"] = {
            "defined": True,
            **_paired(target_numerators, protocol),
        }
    else:
        paired["target_macro_actual_relative_gain"] = {
            "defined": False,
            "valid_units": len(target_numerators),
            "differences": [],
        }

    construction_summary = {
        "selected_fraction": float(construction["selected_fraction"]),
        "selected_index": int(construction["selected_index"]),
        "attempted_candidates": int(construction["search"]["attempted_candidates"]),
        "feasible_candidates": int(construction["search"]["feasible_candidates"]),
        "uncertainty_set_valid": bool(
            construction["search"]["uncertainty_set_valid"]
        ),
        "minimum_raw_robust_lower_score": min(
            float(value)
            for value in construction["search"]["raw_robust_lower_scores"].values()
        ),
        "minimum_selected_protected_robust_margin": float(
            construction["search"]["selected"]["minimum_protected_robust_margin"]
        ),
        "minimum_selected_robust_target_score": float(
            construction["search"]["selected"]["minimum_robust_target_score"]
        ),
        "selected_robust_target_gain_over_raw": float(
            construction["search"]["selected_robust_target_gain_over_raw"]
        ),
    }
    return {
        "construction": construction_summary,
        "coverage": coverage,
        "diagnostic": diagnostic,
        "paired": paired,
    }


def _relative_gate(
    item: dict[str, Any], mean_loss: float, ci_loss: float, unit_loss: float
) -> bool:
    return bool(item.get("defined")) and (
        float(item["bootstrap"]["mean"]) >= -mean_loss
        and float(item["bootstrap"]["ci95_lower"]) > -ci_loss
        and min(float(value) for value in item["differences"]) >= -unit_loss
    )


def evaluate_gates(
    rows: list[dict[str, Any]],
    construction: dict[str, Any],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    state_restored: bool,
    vram_headroom: float,
    gpu_name: str,
) -> dict[str, bool]:
    gates = protocol["gates"]
    protected = tuple(protocol["design"]["protected_certificates"])
    targets = tuple(protocol["design"]["target_certificates"])
    baseline_pair = summary["paired"]["jcrmcd_minus_baseline_worst_score"]
    target_pair = summary["paired"]["jcrmcd_minus_raw_target_minimum_score"]
    macro_pair = summary["paired"]["target_macro_actual_relative_gain"]
    selected_diag = summary["diagnostic"]["jcrmcd_mcd"]
    objective_rows = list(selected_diag["objectives"].values())
    exact_relative = summary["paired"]["jcrmcd_relative_exact_score"]
    actual_relative = summary["paired"]["jcrmcd_relative_actual_reduction"]
    protected_exact_ok = all(
        _relative_gate(
            exact_relative[name],
            float(gates["protected_mean_relative_loss_max"]),
            float(gates["protected_ci_relative_loss_max"]),
            float(gates["protected_single_unit_relative_loss_max"]),
        )
        for name in protected
    )
    protected_actual_ok = all(
        _relative_gate(
            actual_relative[name],
            float(gates["protected_mean_relative_loss_max"]),
            float(gates["protected_ci_relative_loss_max"]),
            float(gates["protected_single_unit_relative_loss_max"]),
        )
        for name in protected
    )
    target_floor_ok = all(
        bool(actual_relative[name].get("defined"))
        and float(actual_relative[name]["bootstrap"]["mean"])
        >= -float(gates["target_objective_mean_relative_loss_max"])
        and min(float(value) for value in actual_relative[name]["differences"])
        >= -float(gates["target_objective_single_unit_relative_loss_max"])
        for name in targets
    )
    uncertainty = construction["uncertainty"]["certificates"]
    radius_calibration_valid = all(
        len(uncertainty[name]["calibration"]["residuals"]) == SUPPORT_COUNT
        and abs(
            float(uncertainty[name]["radius"])
            - max(
                float(item["distance"])
                for item in uncertainty[name]["calibration"]["residuals"]
            )
        )
        <= float(gates["robust_numeric_tolerance"])
        for name in CERTIFICATES
    )
    support_coverages = construction["support_coverages"]
    return {
        "integrity": (
            len(support_coverages) == SUPPORT_COUNT
            and all(item["risk_microbatches"] == SUPPORT_SIZE for item in support_coverages)
            and all(item["safety_microbatches"] == SUPPORT_SIZE for item in support_coverages)
            and all(item["all_gradients_finite_nonzero"] for item in support_coverages)
            and len(rows) == DIAGNOSTIC_UNITS
            and all(row["coverage"]["risk_microbatches"] == 1 for row in rows)
            and all(row["coverage"]["safety_microbatches"] == 1 for row in rows)
            and all(row["coverage"]["all_gradients_finite_nonzero"] for row in rows)
            and int(construction["path_candidates"]) == len(PATH_FRACTIONS)
            and int(construction["search"]["attempted_candidates"])
            == len(PATH_FRACTIONS)
            and len(construction["pooled_path_records"]) == len(PATH_FRACTIONS)
            and all(
                len(panel) == len(PATH_FRACTIONS)
                for panel in construction["support_path_records"].values()
            )
            and all(
                bool(construction[name]["clipping_active"])
                for name in (
                    "baseline_optimizer_map",
                    "raw_optimizer_map",
                    "selected_optimizer_map",
                )
            )
            and float(construction["raw_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and float(construction["target_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and float(construction["selected_direction_budget_relative_error"])
            <= float(gates["gradient_budget_relative_tolerance"])
            and all(
                all(
                    arm["state_restored_after_virtual_step"]
                    for arm in row["arms"].values()
                )
                for row in rows
            )
            and state_restored
            and vram_headroom >= float(gates["minimum_vram_headroom_fraction"])
            and "RTX 4070 Ti" in gpu_name
            and construction["diagnostic_gradients_used_for_selection"] is False
            and construction["raw_gradient_tensors_serialized"] is False
        ),
        "construction_uncertainty_valid": (
            radius_calibration_valid
            and bool(summary["construction"]["uncertainty_set_valid"])
            and float(summary["construction"]["minimum_raw_robust_lower_score"]) > 0.0
            and bool(construction["search"]["raw_anchor"]["feasible"])
            and bool(construction["search"]["selected"]["feasible"])
            and float(
                summary["construction"]["minimum_selected_protected_robust_margin"]
            )
            >= -float(gates["robust_numeric_tolerance"])
        ),
        "robust_target_gain": (
            float(summary["construction"]["selected_fraction"])
            >= float(gates["minimum_nonzero_path_fraction"])
            and float(summary["construction"]["minimum_selected_robust_target_score"])
            >= float(gates["construction_robust_target_score_min"])
            and float(summary["construction"]["selected_robust_target_gain_over_raw"])
            >= float(gates["construction_robust_target_gain_min"])
        ),
        "diagnostic_hull_coverage": (
            bool(summary["coverage"]["all_covered"])
            and int(summary["coverage"]["covered_checks"])
            == DIAGNOSTIC_UNITS * len(CERTIFICATES)
            and float(summary["coverage"]["minimum_protected_residual_slack"])
            >= -float(gates["theorem_numeric_tolerance"])
            and float(summary["coverage"]["minimum_target_lower_bound_slack"])
            >= -float(gates["theorem_numeric_tolerance"])
        ),
        "diagnostic_common_transfer": (
            float(selected_diag["minimum_worst_score"])
            >= float(gates["diagnostic_exact_margin_min"])
            and all(
                float(value) >= float(gates["diagnostic_certificate_score_min"])
                for value in selected_diag["minimum_certificate_scores"].values()
            )
            and float(baseline_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_advantage_over_baseline_mean_min"])
            and float(baseline_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(baseline_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
        ),
        "finite_step_common_descent": all(
            float(item["mean_actual_reduction"])
            > float(gates["mean_actual_loss_reduction_min"])
            and float(item["positive_actual_rate"])
            >= float(gates["actual_reduction_positive_rate_min"])
            and item["mean_actual_to_predicted_ratio"] is not None
            and float(item["mean_actual_to_predicted_ratio"])
            >= float(gates["mean_actual_to_predicted_ratio_min"])
            and float(item["mean_actual_to_predicted_ratio"])
            <= float(gates["mean_actual_to_predicted_ratio_max"])
            for item in objective_rows
        ),
        "protected_diagnostic_noninferiority": (
            protected_exact_ok and protected_actual_ok
        ),
        "target_diagnostic_advantage": (
            float(target_pair["bootstrap"]["mean"])
            >= float(gates["diagnostic_target_gain_mean_min"])
            and float(target_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(target_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and bool(macro_pair.get("defined"))
            and float(macro_pair["bootstrap"]["mean"])
            >= float(gates["target_macro_actual_relative_gain_mean_min"])
            and float(macro_pair["bootstrap"]["ci95_lower"])
            > float(gates["paired_ci_lower_min"])
            and float(macro_pair["sign_test"]["exact_two_sided_p"])
            <= float(gates["paired_sign_p_max"])
            and target_floor_ok
        ),
        "effect_not_weak": (
            float(selected_diag["mean_target_minimum_score"])
            >= float(gates["diagnostic_mean_target_score_min"])
            and bool(macro_pair.get("defined"))
            and float(macro_pair["bootstrap"]["mean"])
            >= float(gates["target_macro_actual_relative_gain_mean_min"])
        ),
    }


def select_route(gates: dict[str, bool]) -> str:
    if not gates["integrity"]:
        return "STOP_JCRMCD_P0_INVALID"
    if not gates["construction_uncertainty_valid"]:
        return "STOP_JCRMCD_P0_CONSTRUCTION_UNCERTAINTY_INVALID"
    if not gates["robust_target_gain"]:
        return "STOP_JCRMCD_P0_ROBUST_TARGET_GAIN_FAILED"
    if not gates["diagnostic_hull_coverage"]:
        return "STOP_JCRMCD_P0_DIAGNOSTIC_HULL_COVERAGE_FAILED"
    if not gates["diagnostic_common_transfer"]:
        return "STOP_JCRMCD_P0_DIAGNOSTIC_TRANSFER_FAILED"
    if not gates["finite_step_common_descent"]:
        return "STOP_JCRMCD_P0_FINITE_STEP_COMMON_DESCENT_FAILED"
    if not gates["protected_diagnostic_noninferiority"]:
        return "STOP_JCRMCD_P0_PROTECTED_NONINFERIORITY_FAILED"
    if not gates["target_diagnostic_advantage"]:
        return "STOP_JCRMCD_P0_TARGET_ADVANTAGE_FAILED"
    if not gates["effect_not_weak"]:
        return "STOP_JCRMCD_P0_EFFECT_TOO_WEAK"
    return "PASS_JCRMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL"


def package_artifacts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for name, value in protocol["outputs"].items():
        if name == "run_manifest":
            continue
        path = resolve_repo_path(value)
        artifacts.append(
            {
                "name": name,
                "path": relative(path),
                "sha256": etarc_p0.sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(etarc_p0.read_jsonl(path))
                if path.suffix == ".jsonl"
                else None,
            }
        )
    return artifacts


def run(path: Path) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import set_seed

    protocol, config, targets, risk_rows = context(path)
    runtime_head = require_clean_tracked_freeze(path, protocol)
    started = time.time()
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = frozen.load_model(config, resume_from=None)
    parameters = sorted(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        key=lambda item: item[0],
    )
    inventory = training.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise JcrmcdP0ValidationError("unexpected trainable parameter inventory")
    initial_hash = model_common.trainable_state_hash(parameters)
    initial_state = prior_runner._snapshot(parameters)
    pair_map, structural = structural_features(tokenizer, targets, config, protocol)

    support_registry = protocol["selection"]["construction_supports"]
    diagnostic_registry = protocol["selection"]["diagnostic_units"]
    selected_risk = {
        int(index)
        for support in support_registry
        for index in support["risk_row_indices"]
    }
    selected_risk.update(int(item["risk_row_index"]) for item in diagnostic_registry)
    risk_features = {
        index: training.response_feature(
            tokenizer,
            str(risk_rows[index]["prompt"]),
            str(risk_rows[index]["response"]),
            int(config["max_length"]),
        )
        for index in selected_risk
    }

    support_gradients = []
    support_coverages = []
    for support in support_registry:
        gradients, coverage = prior_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(value) for value in support["risk_row_indices"]],
            [str(value) for value in support["safety_sample_ids"]],
            config,
            train_mode=True,
        )
        support_gradients.append(cpu_certificate_copy(gradients))
        support_coverages.append({"support_id": support["support_id"], **coverage})
        if not prior_runner._state_equal(parameters, initial_state):
            raise JcrmcdP0ValidationError("support gradients changed parameters")
        print(
            f"JCRMCD {support['support_id']} gradients=7 "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    construction, fixed_displacements, support_units = build_construction_geometry(
        support_gradients, support_coverages, config, protocol
    )
    construction["runtime_head"] = runtime_head
    construction["protocol_sha256"] = etarc_p0.sha256(path)
    del support_gradients
    gc.collect()
    torch.cuda.empty_cache()

    reference_norm = float(construction["reference_displacement_norm"])
    map_audits = {
        "baseline": construction["baseline_optimizer_map"],
        "raw_mcd": construction["raw_optimizer_map"],
        "jcrmcd_mcd": construction["selected_optimizer_map"],
    }
    rows = []
    for diagnostic in diagnostic_registry:
        unit_started = time.time()
        gradients, coverage = prior_runner.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(diagnostic["risk_row_index"])],
            [str(diagnostic["sample_id"])],
            config,
            train_mode=False,
        )
        if not prior_runner._state_equal(parameters, initial_state):
            raise JcrmcdP0ValidationError("diagnostic gradients changed parameters")
        hull = diagnostic_hull_audit(
            gradients,
            support_units,
            construction,
            float(config["dual_numeric_tolerance"]),
        )
        initial_losses = dict(coverage["loss_means"])
        arms = {
            arm: evaluate_fixed_arm(
                arm,
                fixed_displacements[arm],
                map_audits[arm],
                gradients,
                initial_losses,
                model,
                parameters,
                initial_state,
                risk_features,
                pair_map,
                diagnostic,
                config,
                reference_norm,
            )
            for arm in ARMS
        }
        search_selected = construction["search"]["selected"]
        rho = float(protocol["design"]["protected_retention_floor"])
        protected_observed = {
            name: float(arms["jcrmcd_mcd"]["diagnostic_scores"]["scores"][name])
            - rho * float(arms["raw_mcd"]["diagnostic_scores"]["scores"][name])
            for name in PROTECTED_CERTIFICATES
        }
        protected_lower = {
            name: float(search_selected["protected_robust_margins"][name])
            for name in PROTECTED_CERTIFICATES
        }
        target_observed = {
            name: float(arms["jcrmcd_mcd"]["diagnostic_scores"]["scores"][name])
            for name in TARGET_CERTIFICATES
        }
        target_lower = {
            name: float(search_selected["robust_target_scores"][name])
            for name in TARGET_CERTIFICATES
        }
        theorem_audit = {
            "protected_observed_residuals": protected_observed,
            "protected_robust_lower_margins": protected_lower,
            "protected_residual_slacks": {
                name: protected_observed[name] - protected_lower[name]
                for name in PROTECTED_CERTIFICATES
            },
            "target_observed_scores": target_observed,
            "target_robust_lower_scores": target_lower,
            "target_lower_bound_slacks": {
                name: target_observed[name] - target_lower[name]
                for name in TARGET_CERTIFICATES
            },
        }
        row = {
            "unit_id": diagnostic["unit_id"],
            "risk_row_index": diagnostic["risk_row_index"],
            "safety_sample_id": diagnostic["sample_id"],
            "safety_category": diagnostic["category"],
            "coverage": coverage,
            "hull_audit": hull,
            "theorem_audit": theorem_audit,
            "arms": arms,
            "wall_seconds": time.time() - unit_started,
        }
        rows.append(row)
        print(
            f"JCRMCD {diagnostic['unit_id']} covered="
            f"{hull['covered_certificates']}/7 target="
            f"{min(target_observed.values()):.6f} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        del gradients
        gc.collect()
        torch.cuda.empty_cache()

    final_hash = model_common.trainable_state_hash(parameters)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_vram
    summary = summarize_diagnostics(rows, construction, protocol)
    gates = evaluate_gates(
        rows,
        construction,
        summary,
        protocol,
        initial_hash == final_hash,
        headroom,
        gpu_name,
    )
    route = select_route(gates)
    outputs = {
        name: resolve_repo_path(value) for name, value in protocol["outputs"].items()
    }
    diagnostics_payload = {
        "name": "step3_jcrmcd_p0_gradient_diagnostics_v1",
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "summary": summary,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": final_hash,
        "state_restored": initial_hash == final_hash,
        "diagnostic_units": len(rows),
        "diagnostic_hull_checks": len(rows) * len(CERTIFICATES),
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "raw_gradient_tensors_serialized": False,
    }
    environment = {
        "name": "step3_jcrmcd_p0_environment_v1",
        "runtime_head": runtime_head,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
    }
    decision = {
        "name": "step3_jcrmcd_p0_route_decision_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_sha256": etarc_p0.sha256(path),
        "gates": gates,
        "summary": summary,
        "proxy_dev_triplet_protocol_authorized": route
        == "PASS_JCRMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL",
        "matched_training_protocol_authorized": False,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "checkpoint_writes": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    macro = summary["paired"]["target_macro_actual_relative_gain"]
    macro_text = (
        f"{float(macro['bootstrap']['mean']):.6f}"
        if macro.get("defined")
        else "undefined because a raw target reduction was nonpositive"
    )
    report = f"""# JCR-MCD Seed-5 Zero-Step P0 Report

Route: `{route}`.

Four disjoint eight-pair construction supports calibrated seven jackknife
convex-hull radii without diagnostic gradients. The fixed 33-point path chose
`lambda={summary['construction']['selected_fraction']:.5f}`. Its robust target
minimum was `{summary['construction']['minimum_selected_robust_target_score']:.6f}`
and its robust gain over the raw anchor was
`{summary['construction']['selected_robust_target_gain_over_raw']:.6f}`.

The independent supported-domain panel covered
`{summary['coverage']['covered_checks']}/{summary['coverage']['checks']}`
certificate vectors under the prospectively calibrated uncertainty sets. The
minimum observed-minus-certified protected residual slack was
`{summary['coverage']['minimum_protected_residual_slack']:.6f}`. Coverage is a
hard gate; favorable outcomes outside the set do not repair a coverage failure.

Across eight independent diagnostic units, JCR-MCD's mean target minimum was
`{summary['diagnostic']['jcrmcd_mcd']['mean_target_minimum_score']:.6f}`. The
macro relative finite-loss target gain over raw MCD was `{macro_text}`.

All 24 finite-loss measurements used transient direct applications of fixed
first-step displacements followed by exact state restoration. No optimizer or
scheduler was created, `optimizer_steps=0`, and no checkpoint, generated
candidate, Core100 row, or heldout row was opened. Even a pass authorizes only
a separately frozen `proxy_dev` triplet-data protocol, never training.
"""

    etarc_p0.write_json_frozen(outputs["structural_audit"], structural)
    etarc_p0.write_json_frozen(outputs["construction_geometry"], construction)
    etarc_p0.write_jsonl_frozen(outputs["diagnostic_rows"], rows)
    etarc_p0.write_json_frozen(outputs["gradient_diagnostics"], diagnostics_payload)
    etarc_p0.write_json_frozen(outputs["environment"], environment)
    etarc_p0.write_json_frozen(outputs["route"], decision)
    etarc_p0.write_text_frozen(outputs["report"], report)
    manifest = {
        "name": "step3_jcrmcd_p0_reproduction_manifest_v1",
        "route": route,
        "runtime_head": runtime_head,
        "protocol_path": relative(path),
        "protocol_sha256": etarc_p0.sha256(path),
        "config_path": protocol["config_path"],
        "config_sha256": etarc_p0.sha256(resolve_repo_path(protocol["config_path"])),
        "command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_jcrmcd_p0.py run"
        ),
        "verify_command": (
            ".venv/Scripts/python.exe experiments/step3_response_quality_persistence/"
            "scripts/step3_jcrmcd_p0.py verify-run"
        ),
        "seed": config["seed"],
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "optimizer_steps": 0,
        "virtual_parameter_evaluations": len(rows) * len(ARMS),
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "artifacts": package_artifacts(protocol),
    }
    etarc_p0.write_json_frozen(outputs["run_manifest"], manifest)
    del model, fixed_displacements, support_units, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"route": route, "gates": gates}, sort_keys=True))
    return decision


def verify_run(path: Path) -> dict[str, Any]:
    protocol, _, _, _ = context(path)
    construction = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["construction_geometry"])
    )
    rows = etarc_p0.read_jsonl(
        resolve_repo_path(protocol["outputs"]["diagnostic_rows"])
    )
    diagnostics = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["gradient_diagnostics"])
    )
    environment = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["environment"])
    )
    decision = etarc_p0.read_json(resolve_repo_path(protocol["outputs"]["route"]))
    manifest = etarc_p0.read_json(
        resolve_repo_path(protocol["outputs"]["run_manifest"])
    )
    summary = summarize_diagnostics(rows, construction, protocol)
    gates = evaluate_gates(
        rows,
        construction,
        summary,
        protocol,
        bool(diagnostics["state_restored"]),
        float(environment["vram_headroom_fraction"]),
        str(environment["gpu"]),
    )
    route = select_route(gates)
    if (
        decision.get("route") != route
        or decision.get("gates") != gates
        or decision.get("summary") != summary
        or diagnostics.get("summary") != summary
        or manifest.get("route") != route
    ):
        raise JcrmcdP0ValidationError("stored and recomputed JCR-MCD route differs")
    if manifest.get("artifacts") != package_artifacts(protocol):
        raise JcrmcdP0ValidationError("JCR-MCD artifact inventory differs")
    if (
        manifest.get("optimizer_objects") != 0
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_authorized") is not False
        or decision.get("matched_training_protocol_authorized") is not False
    ):
        raise JcrmcdP0ValidationError("stored JCR-MCD boundary differs")
    result = {
        "status": "PASS_JCRMCD_P0_VERIFY",
        "route": route,
        "construction_supports": SUPPORT_COUNT,
        "diagnostic_units": len(rows),
        "diagnostic_hull_checks": summary["coverage"]["checks"],
        "covered_hull_checks": summary["coverage"]["covered_checks"],
        "virtual_parameter_evaluations": manifest["virtual_parameter_evaluations"],
        "artifacts": len(manifest["artifacts"]),
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    path = resolve_repo_path(args.protocol)
    if args.command == "preflight":
        preflight(path)
    elif args.command == "run":
        run(path)
    else:
        verify_run(path)


if __name__ == "__main__":
    main()
