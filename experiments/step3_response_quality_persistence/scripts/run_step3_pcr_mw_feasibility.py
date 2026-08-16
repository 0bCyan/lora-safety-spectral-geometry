"""Freeze and run the zero-update PCR-M/PCR-MW gradient feasibility stage."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_pcr_mw_common as pcr
import step3ct_t2r_common as gradient_common
import step3ct_training_common as training_common
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze-packet", "preflight", "run"])
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return (REPO / value).resolve()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def freeze_packet(registry: dict[str, Any], registry_path: Path) -> None:
    source = resolve_path(registry["source_targets_path"])
    packet = resolve_path(registry["mismatch_packet_path"])
    manifest_path = packet.with_name("pcr_mw_mismatch_packet_manifest.json")
    if packet.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite PCR mismatch packet")
    if pcr.sha256_file(source) != registry["source_targets_sha256"]:
        raise ValueError("source target hash mismatch")
    rows = pcr.build_mismatch_rows(
        pcr.read_jsonl(source), int(registry["minimum_category_prompts"])
    )
    pcr.write_jsonl(packet, rows)
    manifest = {
        "name": "step3_pcr_mw_mismatch_packet_manifest_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "git_head": git_head(),
        "registry_path": registry_path.relative_to(REPO).as_posix(),
        "registry_sha256_before_packet_hash_freeze": pcr.sha256_file(registry_path),
        "source_targets_sha256": pcr.sha256_file(source),
        "packet_path": packet.relative_to(REPO).as_posix(),
        "packet_sha256": pcr.sha256_file(packet),
        "target_rows": len(rows),
        "prompt_clusters": len({row["sample_id"] for row in rows}),
        "categories": sorted({str(row["category"]) for row in rows}),
        "distinct_wrong_prompts_per_row": all(
            len(set(row["wrong_sample_ids"])) == 2
            and row["sample_id"] not in row["wrong_sample_ids"]
            for row in rows
        ),
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    pcr.write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def validate_preflight(registry: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    source = resolve_path(registry["source_targets_path"])
    packet = resolve_path(registry["mismatch_packet_path"])
    source_rows = pcr.read_jsonl(source)
    category_by_sample = {
        str(row["sample_id"]): str(row["category"]) for row in source_rows
    }
    rows = pcr.read_jsonl(packet)
    checks = {
        "source_hash": pcr.sha256_file(source) == registry["source_targets_sha256"],
        "packet_hash": pcr.sha256_file(packet) == registry["mismatch_packet_sha256"],
        "target_rows": len(rows) == int(registry["expected_target_rows"]),
        "prompt_clusters": len({row["sample_id"] for row in rows})
        == int(registry["expected_prompt_clusters"]),
        "two_distinct_wrong_prompts": all(
            len(set(row["wrong_sample_ids"])) == 2
            and row["sample_id"] not in row["wrong_sample_ids"]
            for row in rows
        ),
        "same_category_wrong_prompts": all(
            len(row["wrong_prompts"]) == int(registry["wrong_prompts_per_target"])
            and all(
                category_by_sample[str(wrong_id)] == str(row["category"])
                for wrong_id in row["wrong_sample_ids"]
            )
            for row in rows
        ),
        "source_order_preserved": [int(row["source_position"]) for row in rows]
        == sorted(int(row["source_position"]) for row in rows),
        "optimizer_steps_zero": int(registry["optimizer_steps"]) == 0,
        "training_not_authorized": registry["training_authorized"] is False,
        "core100_sealed": registry["core100_opened"] is False,
        "heldout_sealed": registry["heldout_opened"] is False,
    }
    result = {
        "name": "step3_pcr_mw_feasibility_preflight_v1",
        "registry_path": registry_path.relative_to(REPO).as_posix(),
        "registry_sha256": pcr.sha256_file(registry_path),
        "checks": checks,
        "passed": all(checks.values()),
        "target_rows": len(rows),
        "prompt_clusters": len({row["sample_id"] for row in rows}),
        "packet_sha256": pcr.sha256_file(packet),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"PCR preflight failed: {[key for key, value in checks.items() if not value]}")
    return result


def _gradient(model: Any, loss: Any, scale: float, retain_graph: bool = False) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    (loss * scale).backward(retain_graph=retain_graph)
    captured = training_common.capture_lora_gradients(model)
    model.zero_grad(set_to_none=True)
    return {name: value / scale for name, value in captured.items()}


def _sequence_nll(model: Any, feature: dict[str, Any], device: Any, fp16: bool) -> Any:
    import torch

    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        return model(
            **training_common.batch_feature(feature, device),
            use_cache=False,
            return_dict=True,
        ).loss


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "q1": pcr.quantile(values, 0.25),
        "median": pcr.quantile(values, 0.5),
        "q3": pcr.quantile(values, 0.75),
        "maximum": max(values),
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def run(registry: dict[str, Any], registry_path: Path, output_dir: Path) -> None:
    import torch

    preflight = validate_preflight(registry, registry_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = resolve_path(registry["parent_training_config"])
    cfg = training_common.read_json(cfg_path)
    packet = pcr.read_jsonl(resolve_path(registry["mismatch_packet_path"]))
    calibration_rows = packet[: int(registry["calibration_rows"])]
    if len(calibration_rows) != int(registry["calibration_rows"]):
        raise ValueError("incomplete calibration window")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model, tokenizer = frozen.load_model(cfg, resume_from=None)
    device = training_common.model_device(model)
    inventory = training_common.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RuntimeError("unexpected trainable parameter inventory")
    fp16 = bool(cfg["fp16"])
    scale = float(registry["fp16_gradient_scale"])
    temperature = float(registry["worst_temperature"])
    max_length = int(cfg["max_length"])
    result_rows: list[dict[str, Any]] = []

    for index, row in enumerate(calibration_rows, start=1):
        correct_feature = training_common.response_feature(
            tokenizer, str(row["correct_prompt"]), str(row["response"]), max_length
        )
        wrong_features = [
            training_common.response_feature(tokenizer, str(prompt), str(row["response"]), max_length)
            for prompt in row["wrong_prompts"]
        ]
        model.train()
        safety_nll = _sequence_nll(model, correct_feature, device, fp16)
        safety_gradient = _gradient(model, safety_nll, scale)

        with training_common.binding_mode(model):
            correct_nll = _sequence_nll(model, correct_feature, device, fp16)
            wrong_nlls = torch.stack(
                [_sequence_nll(model, feature, device, fp16) for feature in wrong_features]
            )
            reference_margins = (wrong_nlls - correct_nll).detach()
            detail = pcr.ranking_losses(
                correct_nll, wrong_nlls, reference_margins, temperature
            )
            mean_gradient = _gradient(model, detail["mean_loss"], scale, retain_graph=True)
            mean_worst_gradient = _gradient(model, detail["mean_worst_loss"], scale)

        safety_norm = gradient_common.gradient_norm(safety_gradient)
        mean_norm = gradient_common.gradient_norm(mean_gradient)
        mean_worst_norm = gradient_common.gradient_norm(mean_worst_gradient)
        mean_ratio = mean_norm / safety_norm if safety_norm else math.nan
        mean_worst_ratio = mean_worst_norm / safety_norm if safety_norm else math.nan
        finite_nonzero = all(
            gradient_common.gradients_finite_nonzero(value)
            for value in (safety_gradient, mean_gradient, mean_worst_gradient)
        )
        result_rows.append(
            {
                "calibration_index": index - 1,
                "ranking_id": row["ranking_id"],
                "source_position": row["source_position"],
                "sample_id_hash": pcr.stable_hash(str(row["sample_id"])),
                "target_id_hash": pcr.stable_hash(str(row["target_id"])),
                "category": row["category"],
                "correct_response_tokens": sum(
                    int(value != -100) for value in correct_feature["labels"]
                ),
                "safety_nll": float(safety_nll.detach().float().cpu()),
                "reference_margins": [float(value) for value in reference_margins.float().cpu()],
                "mean_loss_at_reference": float(detail["mean_loss"].detach().float().cpu()),
                "mean_worst_loss_at_reference": float(
                    detail["mean_worst_loss"].detach().float().cpu()
                ),
                "safety_gradient_norm": _finite_or_none(safety_norm),
                "mean_gradient_norm": _finite_or_none(mean_norm),
                "mean_worst_gradient_norm": _finite_or_none(mean_worst_norm),
                "mean_aux_to_safety_ratio": _finite_or_none(mean_ratio),
                "mean_worst_aux_to_safety_ratio": _finite_or_none(mean_worst_ratio),
                "mean_mean_worst_gradient_cosine": _finite_or_none(
                    gradient_common.gradient_cosine(mean_gradient, mean_worst_gradient)
                ),
                "mean_safety_gradient_cosine": _finite_or_none(
                    gradient_common.gradient_cosine(mean_gradient, safety_gradient)
                ),
                "mean_worst_safety_gradient_cosine": _finite_or_none(
                    gradient_common.gradient_cosine(mean_worst_gradient, safety_gradient)
                ),
                "finite_nonzero": finite_nonzero,
            }
        )
        print(f"PCR-MW feasibility {index}/{len(calibration_rows)} finite={finite_nonzero}", flush=True)

    pcr.write_jsonl(output_dir / "calibration_rows.jsonl", result_rows)
    mean_ratios = [
        float(row["mean_aux_to_safety_ratio"])
        for row in result_rows
        if row["mean_aux_to_safety_ratio"] is not None
        and float(row["mean_aux_to_safety_ratio"]) > 0
    ]
    mw_ratios = [
        float(row["mean_worst_aux_to_safety_ratio"])
        for row in result_rows
        if row["mean_worst_aux_to_safety_ratio"] is not None
        and float(row["mean_worst_aux_to_safety_ratio"]) > 0
    ]
    target_ratio = float(registry["target_weighted_aux_to_safety_gradient_ratio"])
    all_ratios_valid = len(mean_ratios) == len(result_rows) and len(mw_ratios) == len(result_rows)
    mean_summary = _summary(mean_ratios) if mean_ratios else {}
    mw_summary = _summary(mw_ratios) if mw_ratios else {}
    eta_mean = target_ratio / mean_summary["median"] if all_ratios_valid else None
    eta_mw = target_ratio / mw_summary["median"] if all_ratios_valid else None
    eta_mean_values = [target_ratio / value for value in mean_ratios]
    eta_mw_values = [target_ratio / value for value in mw_ratios]
    eta_mean_summary = _summary(eta_mean_values) if eta_mean_values else {}
    eta_mw_summary = _summary(eta_mw_values) if eta_mw_values else {}
    thresholds = registry["feasibility_gates"]
    finite_rows = sum(bool(row["finite_nonzero"]) for row in result_rows)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    headroom = 1.0 - peak_reserved / total_memory
    gates = {
        "eligible_prompt_clusters": preflight["prompt_clusters"]
        >= int(thresholds["eligible_prompt_clusters_min"]),
        "eligible_target_rows": preflight["target_rows"]
        >= int(thresholds["eligible_target_rows_min"]),
        "finite_nonzero_gradient_rows": finite_rows
        == int(thresholds["finite_nonzero_gradient_rows"]),
        "vram_headroom": headroom >= float(thresholds["vram_headroom_fraction_min"]),
        "all_auxiliary_ratios_finite_positive": all_ratios_valid,
        "mean_raw_ratio_range": all_ratios_valid
        and float(thresholds["raw_aux_to_safety_ratio_median_min"])
        <= mean_summary["median"]
        <= float(thresholds["raw_aux_to_safety_ratio_median_max"]),
        "mean_worst_raw_ratio_range": all_ratios_valid
        and float(
            thresholds["raw_aux_to_safety_ratio_median_min"]
        )
        <= mw_summary["median"]
        <= float(thresholds["raw_aux_to_safety_ratio_median_max"]),
        "mean_eta_iqr_stability": all_ratios_valid
        and eta_mean_summary["q3"]
        / max(eta_mean_summary["q1"], 1e-30)
        <= float(thresholds["eta_interquartile_ratio_max"]),
        "mean_worst_eta_iqr_stability": all_ratios_valid
        and eta_mw_summary["q3"]
        / max(eta_mw_summary["q1"], 1e-30)
        <= float(thresholds["eta_interquartile_ratio_max"]),
    }
    route = "PASS_FREEZE_ETA_AND_SMOKE_PROTOCOL" if all(gates.values()) else "STOP_PCR_MW_FEASIBILITY_FAIL"
    summary = {
        "name": "step3_pcr_mw_zero_update_feasibility_result_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "git_head": git_head(),
        "registry_path": registry_path.relative_to(REPO).as_posix(),
        "registry_sha256": pcr.sha256_file(registry_path),
        "parent_config_sha256": pcr.sha256_file(cfg_path),
        "preflight": preflight,
        "calibration_rows": len(result_rows),
        "finite_nonzero_gradient_rows": finite_rows,
        "mean_aux_to_safety_ratio": mean_summary,
        "mean_worst_aux_to_safety_ratio": mw_summary,
        "eta_pcr_m": eta_mean,
        "eta_pcr_mw": eta_mw,
        "eta_pcr_m_distribution": eta_mean_summary,
        "eta_pcr_mw_distribution": eta_mw_summary,
        "mean_mean_worst_gradient_cosine_median": (
            pcr.quantile(
                [
                    float(row["mean_mean_worst_gradient_cosine"])
                    for row in result_rows
                    if row["mean_mean_worst_gradient_cosine"] is not None
                ],
                0.5,
            )
            if any(row["mean_mean_worst_gradient_cosine"] is not None for row in result_rows)
            else None
        ),
        "gates": gates,
        "route": route,
        "calibration_rows_sha256": pcr.sha256_file(output_dir / "calibration_rows.jsonl"),
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": headroom,
        "wall_clock_seconds": time.time() - started,
        "optimizer_steps": 0,
        "training_authorized": False,
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    pcr.write_json(output_dir / "feasibility_summary.json", summary)
    print(json.dumps({"route": route, "eta_pcr_m": eta_mean, "eta_pcr_mw": eta_mw, "gates": gates}))


def main() -> None:
    args = parse_args()
    registry_path = resolve_path(args.registry)
    registry = pcr.read_json(registry_path)
    if args.command == "freeze-packet":
        freeze_packet(registry, registry_path)
    elif args.command == "preflight":
        print(json.dumps(validate_preflight(registry, registry_path), ensure_ascii=False))
    else:
        if not args.output_dir:
            raise ValueError("run requires --output-dir")
        run(registry, registry_path, resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
