"""Train the PCR-M and PCR-MW prompt-conditional sequence-ranking arms."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import step3_pcr_mw_common as pcr
import step3ct_training_common as common
import train_step3ct_tournament as frozen


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-kind", choices=["smoke", "development", "main"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-steps")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _sequence_nll(model: Any, feature: dict[str, Any], device: Any, fp16: bool) -> Any:
    import torch

    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        return model(
            **common.batch_feature(feature, device), use_cache=False, return_dict=True
        ).loss


def _ranking_features(
    tokenizer: Any, packet: list[dict[str, Any]], max_length: int
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in packet:
        result[str(row["target_id"])] = {
            "row": row,
            "correct": common.response_feature(
                tokenizer, str(row["correct_prompt"]), str(row["response"]), max_length
            ),
            "wrong": [
                common.response_feature(tokenizer, str(prompt), str(row["response"]), max_length)
                for prompt in row["wrong_prompts"]
            ],
        }
    return result


def build_reference_cache(
    model: Any,
    features: dict[str, dict[str, Any]],
    device: Any,
    fp16: bool,
) -> list[dict[str, Any]]:
    import torch

    rows = []
    with common.binding_mode(model), torch.no_grad():
        for target_id in sorted(features):
            entry = features[target_id]
            correct_nll = _sequence_nll(model, entry["correct"], device, fp16)
            wrong_nlls = torch.stack(
                [_sequence_nll(model, feature, device, fp16) for feature in entry["wrong"]]
            )
            margins = wrong_nlls - correct_nll
            rows.append(
                {
                    "target_id_hash": pcr.stable_hash(target_id),
                    "target_id": target_id,
                    "reference_correct_nll": float(correct_nll.float().cpu()),
                    "reference_wrong_nlls": [float(value) for value in wrong_nlls.float().cpu()],
                    "reference_margins": [float(value) for value in margins.float().cpu()],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cfg_path = (REPO / args.config).resolve()
    cfg = common.read_json(cfg_path)
    if args.arm not in cfg["arms"]:
        raise ValueError(f"unregistered PCR arm: {args.arm}")
    if args.run_kind == "main" and cfg.get("main_training_authorized") is not True:
        raise PermissionError("main PCR training is not authorized")
    if args.run_kind == "development" and cfg.get("development_training_authorized") is not True:
        raise PermissionError("development PCR training is not authorized")
    arm = cfg["arms"][args.arm]
    if arm["ranking_mode"] not in {"none", "mean", "mean_worst"}:
        raise ValueError("unsupported PCR ranking mode")
    full_steps = int(cfg["optimizer_steps"])
    default_steps = (
        int(cfg["smoke_steps"])
        if args.run_kind == "smoke"
        else int(cfg["development_steps"])
        if args.run_kind == "development"
        else full_steps
    )
    max_steps = int(args.max_steps or default_steps)
    if max_steps <= 0 or max_steps > full_steps:
        raise ValueError("max steps outside configured schedule")
    if args.run_kind == "smoke" and max_steps != int(cfg["smoke_steps"]):
        raise ValueError("PCR smoke must use exactly the frozen two steps")
    if args.run_kind == "development" and max_steps != int(cfg["development_steps"]):
        raise ValueError("PCR development run must use exactly the frozen 16 steps")
    output = (REPO / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(int(cfg["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    packet_path = (REPO / cfg["mismatch_packet_path"]).resolve()
    if pcr.sha256_file(packet_path) != cfg["mismatch_packet_sha256"]:
        raise ValueError("PCR mismatch packet hash mismatch")
    packet = pcr.read_jsonl(packet_path)
    if len(packet) != 74:
        raise ValueError("PCR mismatch packet must contain 74 rows")
    ranking_by_target = {str(row["target_id"]): row for row in packet}

    risk_rows = common.read_jsonl(REPO / cfg["risk_data_path"], limit=int(cfg["train_examples"]))
    targets = common.read_jsonl(REPO / cfg["safety_targets_path"])
    if len(risk_rows) != 256 or len(targets) != 80:
        raise ValueError("PCR training data dimensions mismatch")
    model, tokenizer = frozen.load_model(cfg, resume_from=None)
    device = common.model_device(model)
    inventory = common.trainable_parameter_inventory(model)
    if not inventory["passed"]:
        raise RuntimeError("unexpected trainable parameter inventory")
    risk_features = [
        common.response_feature(tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"]))
        for row in risk_rows
    ]
    target_features = {
        str(row["target_id"]): common.response_feature(
            tokenizer, str(row["prompt"]), str(row["response"]), int(cfg["max_length"])
        )
        for row in targets
    }
    ranking_features = _ranking_features(tokenizer, packet, int(cfg["max_length"]))
    reference_rows = build_reference_cache(model, ranking_features, device, bool(cfg["fp16"]))
    reference_by_target = {
        row["target_id"]: row["reference_margins"] for row in reference_rows
    }
    pcr.write_jsonl(output / "reference_margin_cache.jsonl", reference_rows)
    reference_cache_hash = pcr.sha256_file(output / "reference_margin_cache.jsonl")

    optimizer = frozen.make_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["warmup_steps"]), num_training_steps=full_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["fp16"]))
    order = frozen.risk_order(len(risk_rows), int(cfg["seed"]))
    save_steps = (
        {int(value) for value in args.save_steps.split(",") if value.strip()}
        if args.save_steps
        else set(int(value) for value in cfg["checkpoint_steps"])
    )
    if args.run_kind == "smoke":
        save_steps.update({0, 1, 2})
    trace: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    target_position = 0
    if 0 in save_steps:
        frozen.save_checkpoint(
            output, 0, model, tokenizer, optimizer, scheduler, scaler, None,
            target_position, trace, checkpoint_rows,
        )

    microsteps = int(cfg["gradient_accumulation_steps"])
    p = float(cfg["safety_mix"])
    eta = float(arm["eta"])
    risk_coefficient = (1.0 - p) / microsteps
    safety_coefficient = p / microsteps
    ranking_coefficient = p * eta / microsteps
    optimizer.zero_grad(set_to_none=True)
    skipped_updates = 0
    target_exposure: Counter[str] = Counter()

    for global_step in range(max_steps):
        step_started = time.time()
        risk_losses = []
        safety_losses = []
        ranking_losses = []
        margin_changes = []
        ranking_positions = 0
        for micro_index in range(microsteps):
            position = global_step * microsteps + micro_index
            model.train()
            risk_loss = _sequence_nll(model, risk_features[order[position]], device, bool(cfg["fp16"]))
            scaler.scale(risk_loss * risk_coefficient).backward()
            risk_losses.append(float(risk_loss.detach().float().cpu()))

            target = targets[target_position % len(targets)]
            target_position += 1
            target_id = str(target["target_id"])
            target_exposure[target_id] += 1
            safety_loss = _sequence_nll(model, target_features[target_id], device, bool(cfg["fp16"]))
            scaler.scale(safety_loss * safety_coefficient).backward()
            safety_losses.append(float(safety_loss.detach().float().cpu()))

            if target_id in ranking_by_target and arm["ranking_mode"] != "none":
                entry = ranking_features[target_id]
                with common.binding_mode(model):
                    correct_nll = _sequence_nll(model, entry["correct"], device, bool(cfg["fp16"]))
                    wrong_nlls = torch.stack(
                        [_sequence_nll(model, feature, device, bool(cfg["fp16"])) for feature in entry["wrong"]]
                    )
                    reference = torch.tensor(
                        reference_by_target[target_id], dtype=wrong_nlls.dtype, device=wrong_nlls.device
                    )
                    detail = pcr.ranking_losses(
                        correct_nll, wrong_nlls, reference, float(cfg["worst_temperature"])
                    )
                    ranking_loss = detail[
                        "mean_loss" if arm["ranking_mode"] == "mean" else "mean_worst_loss"
                    ]
                    scaler.scale(ranking_loss * ranking_coefficient).backward()
                ranking_losses.append(float(ranking_loss.detach().float().cpu()))
                margin_changes.extend(float(value) for value in detail["centered_margins"].detach().float().cpu())
                ranking_positions += 1

        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
        )
        if not finite:
            raise FloatingPointError(f"non-finite PCR gradient before step {global_step + 1}")
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"])))
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            skipped_updates += 1
            raise FloatingPointError(f"GradScaler skipped PCR step {global_step + 1}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = global_step + 1
        trace.append(
            {
                "optimizer_step": completed_step,
                "arm": args.arm,
                "ranking_mode": arm["ranking_mode"],
                "eta": eta,
                "risk_loss_mean": sum(risk_losses) / len(risk_losses),
                "safety_loss_mean": sum(safety_losses) / len(safety_losses),
                "ranking_loss_mean": sum(ranking_losses) / len(ranking_losses) if ranking_losses else "",
                "ranking_positions": ranking_positions,
                "centered_margin_change_mean": sum(margin_changes) / len(margin_changes) if margin_changes else "",
                "gradient_preclip_norm": preclip_norm,
                "learning_rate_after_update": float(scheduler.get_last_lr()[0]),
                "wall_seconds": time.time() - step_started,
            }
        )
        common.write_csv(output / "training_trace.csv", trace)
        if completed_step in save_steps or completed_step == max_steps:
            frozen.save_checkpoint(
                output, completed_step, model, tokenizer, optimizer, scheduler, scaler, None,
                target_position, trace, checkpoint_rows,
            )
        rank_display = trace[-1]["ranking_loss_mean"]
        margin_display = trace[-1]["centered_margin_change_mean"]
        print(
            f"{args.arm} step {completed_step}/{max_steps} "
            f"rank={'none' if rank_display == '' else f'{float(rank_display):.6f}'} "
            f"margin={'none' if margin_display == '' else f'{float(margin_display):.6f}'} "
            f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )

    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_pcr_mw_arm_run_manifest_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "git_head": git_head(),
        "arm": args.arm,
        "run_kind": args.run_kind,
        "ranking_mode": arm["ranking_mode"],
        "eta": eta,
        "config_path": args.config,
        "config_sha256": pcr.sha256_file(cfg_path),
        "mismatch_packet_sha256": pcr.sha256_file(packet_path),
        "reference_cache_rows": len(reference_rows),
        "reference_cache_sha256": reference_cache_hash,
        "completed_steps": max_steps,
        "optimizer_steps": max_steps,
        "skipped_updates": skipped_updates,
        "all_losses_and_gradients_finite": True,
        "safety_draws": target_position,
        "ranking_draws": sum(int(row["ranking_positions"]) for row in trace),
        "target_exposure_hash": common.hash_json(dict(sorted(target_exposure.items()))),
        "risk_order_hash": common.hash_json(order),
        "trainable_parameter_inventory": inventory,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_memory,
        "vram_gate_pass": 1.0 - peak_reserved / total_memory
        >= float(cfg["minimum_vram_headroom_fraction"]),
        "candidate_outputs_generated": False,
        "core100_opened": False,
        "heldout_opened": False,
        "main_training_authorized": cfg["main_training_authorized"],
        "wall_clock_seconds": time.time() - started,
    }
    common.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"arm": args.arm, "completed_steps": max_steps, "reference_cache_sha256": reference_cache_hash}))


if __name__ == "__main__":
    main()
