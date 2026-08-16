"""Audit whether PCR conditionality transfer is concentrated in a forced no-think prefix."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import step3_pcr_mw_common as pcr


REPO = Path(__file__).resolve().parents[3]
COMPONENTS = ("legacy_combined", "forced_preamble", "response_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze-packet", "preflight", "run"])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
    return path


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def build_mismatch_packet(rows: list[dict[str, Any]], wrong_prompts: int) -> list[dict[str, Any]]:
    if wrong_prompts != 2:
        raise ValueError("serialization audit requires exactly two wrong prompts")
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows) or len(rows) < 3:
        raise ValueError("split must contain at least three unique sample IDs")
    output = []
    for position, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        candidates = [value for value in by_id if value != sample_id]
        wrong_ids = sorted(candidates, key=lambda value: pcr.stable_hash(sample_id, value))[:2]
        output.append(
            {
                "position": position,
                "sample_id": sample_id,
                "correct_prompt": row["prompt"],
                "wrong_sample_ids": wrong_ids,
                "wrong_prompts": [by_id[value]["prompt"] for value in wrong_ids],
                "correct_prompt_sha256": pcr.stable_hash(str(row["prompt"])),
                "wrong_prompt_sha256": [
                    pcr.stable_hash(str(by_id[value]["prompt"])) for value in wrong_ids
                ],
            }
        )
    return output


def freeze_packet(protocol: dict[str, Any]) -> None:
    split_path = resolve_path(protocol["split_path"])
    packet_path = resolve_path(protocol["mismatch_packet_path"])
    if packet_path.exists():
        raise FileExistsError(packet_path)
    if pcr.sha256_file(split_path) != protocol["split_sha256"]:
        raise ValueError("split hash mismatch")
    rows = pcr.read_jsonl(split_path)
    packet = build_mismatch_packet(rows, int(protocol["wrong_prompts_per_row"]))
    pcr.write_jsonl(packet_path, packet)
    print(json.dumps({"rows": len(packet), "packet_sha256": pcr.sha256_file(packet_path)}))


def _response_maps(protocol: dict[str, Any], split_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    expected_ids = [str(row["sample_id"]) for row in split_rows]
    sources = {"reference": {str(row["sample_id"]): str(row["reference_response"]) for row in split_rows}}
    for source, spec in protocol["generated_response_sources"].items():
        path = resolve_path(spec["path"])
        if pcr.sha256_file(path) != spec["sha256"]:
            raise ValueError(f"response source hash mismatch: {source}")
        rows = pcr.read_jsonl(path)
        if [str(row["sample_id"]) for row in rows] != expected_ids:
            raise ValueError(f"response source IDs mismatch: {source}")
        if any(str(row["arm"]) != spec["arm"] for row in rows):
            raise ValueError(f"response source arm mismatch: {source}")
        sources[source] = {str(row["sample_id"]): str(row["response"]) for row in rows}
    return sources


def validate_preflight(protocol: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    split_path = resolve_path(protocol["split_path"])
    packet_path = resolve_path(protocol["mismatch_packet_path"])
    script_path = Path(__file__).resolve()
    split_rows = pcr.read_jsonl(split_path)
    packet = pcr.read_jsonl(packet_path)
    response_maps = _response_maps(protocol, split_rows)
    packet_ids = [str(row["sample_id"]) for row in packet]
    split_ids = [str(row["sample_id"]) for row in split_rows]
    checks = {
        "status_prospective": protocol["status"] == "prospective_before_serialization_scores",
        "script_hash": pcr.sha256_file(script_path) == protocol["script_sha256"],
        "split_hash": pcr.sha256_file(split_path) == protocol["split_sha256"],
        "packet_hash": pcr.sha256_file(packet_path) == protocol["mismatch_packet_sha256"],
        "row_count": len(packet) == int(protocol["expected_rows"]),
        "ordered_ids": packet_ids == split_ids,
        "two_distinct_wrong_prompts": all(
            len(row["wrong_sample_ids"]) == 2
            and len(set(row["wrong_sample_ids"])) == 2
            and str(row["sample_id"]) not in row["wrong_sample_ids"]
            for row in packet
        ),
        "four_response_sources": set(response_maps)
        == {"reference", "C0R_reproduction", "PCR_M_mean", "PCR_MW_mean_worst"},
        "adapter_hashes": all(
            pcr.sha256_file(resolve_path(spec["checkpoint_dir"]) / "adapter_model.safetensors")
            == spec["adapter_sha256"]
            for spec in protocol["models"].values()
        ),
        "optimizer_steps_zero": int(protocol["optimizer_steps"]) == 0,
        "training_not_authorized": protocol["training_authorized"] is False,
        "core100_sealed": protocol["core100_opened"] is False,
        "heldout_sealed": protocol["heldout_opened"] is False,
    }
    result = {
        "name": "step3_pcr_serialization_transfer_preflight_v1",
        "protocol": protocol_path.relative_to(REPO).as_posix(),
        "protocol_sha256": pcr.sha256_file(protocol_path),
        "checks": checks,
        "passed": all(checks.values()),
        "rows": len(packet),
        "response_sources": sorted(response_maps),
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"preflight failed: {[key for key, value in checks.items() if not value]}")
    return result


def component_means(token_losses: list[float], direct_prefix: int, forced_prefix: int) -> dict[str, Any]:
    # token_losses[k] is the loss for token index k + 1 in the full sequence.
    indexed = [(index + 1, value) for index, value in enumerate(token_losses)]
    groups = {
        "legacy_combined": [value for index, value in indexed if index >= direct_prefix],
        "forced_preamble": [
            value for index, value in indexed if direct_prefix <= index < forced_prefix
        ],
        "response_only": [value for index, value in indexed if index >= forced_prefix],
    }
    if any(not values for values in groups.values()):
        raise ValueError("serialization component has no scored tokens")
    return {
        component: {"nll": sum(values) / len(values), "tokens": len(values)}
        for component, values in groups.items()
    }


def tokenize_entry(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, Any]:
    user = [{"role": "user", "content": prompt}]
    conversation = [*user, {"role": "assistant", "content": response}]
    direct_text = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    forced_text = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    direct = tokenizer(direct_text, add_special_tokens=False)["input_ids"]
    forced = tokenizer(forced_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    if full[: len(direct)] != direct or full[: len(forced)] != forced:
        raise ValueError("chat-template prefixes are not exact full-sequence prefixes")
    if len(forced) <= len(direct) or len(full) <= len(forced):
        raise ValueError("unexpected no-think serialization lengths")
    return {
        "input_ids": full,
        "direct_prefix_tokens": len(direct),
        "forced_prefix_tokens": len(forced),
    }


def score_batch(model: Any, entries: list[dict[str, Any]], pad_token_id: int, fp16: bool) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    maximum = max(len(entry["input_ids"]) for entry in entries)
    input_ids = []
    attention_masks = []
    for entry in entries:
        ids = list(entry["input_ids"])
        padding = maximum - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        attention_masks.append([1] * len(ids) + [0] * padding)
    device = next(model.parameters()).device
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        logits = model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False).logits
    results = []
    for index, entry in enumerate(entries):
        length = len(entry["input_ids"])
        losses = functional.cross_entropy(
            logits[index, : length - 1].float(),
            input_tensor[index, 1:length],
            reduction="none",
        )
        values = [float(value) for value in losses.cpu()]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("non-finite token NLL")
        results.append(
            component_means(
                values, int(entry["direct_prefix_tokens"]), int(entry["forced_prefix_tokens"])
            )
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def margin_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(row["model"], row["response_source"], row["sample_id"])].append(row)
    output = []
    for (model, source, sample_id), rows in sorted(grouped.items()):
        by_role = {row["prompt_role"]: row for row in rows}
        if set(by_role) != {"correct", "wrong_1", "wrong_2"}:
            raise ValueError("incomplete prompt-role triplet")
        result: dict[str, Any] = {
            "model": model,
            "response_source": source,
            "sample_id": sample_id,
        }
        for component in COMPONENTS:
            correct = float(by_role["correct"][f"{component}_nll"])
            wrong = [float(by_role[role][f"{component}_nll"]) for role in ("wrong_1", "wrong_2")]
            result[f"{component}_margin"] = sum(wrong) / 2.0 - correct
            result[f"{component}_correct_rank1"] = int(correct < min(wrong))
        output.append(result)
    return output


def bootstrap(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return {
        "rows": len(values),
        "mean": sum(values) / len(values),
        "ci95_lower": draws[int(0.025 * samples)],
        "ci95_upper": draws[min(samples - 1, int(0.975 * samples))],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def paired_transfer(
    margins: list[dict[str, Any]], candidate: str, source: str, samples: int, seed: int
) -> dict[str, Any]:
    maps: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in margins:
        if row["response_source"] == source and row["model"] in {"C0R_reproduction", candidate}:
            maps[row["model"]][row["sample_id"]] = row
    if set(maps["C0R_reproduction"]) != set(maps[candidate]):
        raise ValueError("paired transfer IDs differ")
    output = {}
    per_component: dict[str, list[float]] = {}
    for offset, component in enumerate(COMPONENTS):
        values = [
            float(maps[candidate][sample_id][f"{component}_margin"])
            - float(maps["C0R_reproduction"][sample_id][f"{component}_margin"])
            for sample_id in sorted(maps[candidate])
        ]
        per_component[component] = values
        output[component] = bootstrap(values, samples, seed + offset)
    leakage = [
        preamble - response
        for preamble, response in zip(
            per_component["forced_preamble"], per_component["response_only"], strict=True
        )
    ]
    output["preamble_minus_response_transfer"] = bootstrap(leakage, samples, seed + 3)
    return output


def candidate_route(transfer: dict[str, Any]) -> str:
    legacy = transfer["legacy_combined"]
    response = transfer["response_only"]
    leakage = transfer["preamble_minus_response_transfer"]
    if legacy["ci95_lower"] > 0 and response["ci95_lower"] > 0:
        return "TRANSFER_SURVIVES_RESPONSE_ONLY"
    if (
        legacy["ci95_lower"] > 0
        and response["ci95_lower"] <= 0
        and leakage["ci95_lower"] > 0
    ):
        return "PREFIX_LEAKAGE_SUPPORTED"
    if legacy["ci95_lower"] <= 0 and response["ci95_lower"] <= 0:
        return "NO_OUT_OF_SAMPLE_CONDITIONALITY_TRANSFER"
    return "INCONCLUSIVE_SERIALIZATION_TRANSFER"


def overall_route(candidate_routes: dict[str, str]) -> str:
    routes = set(candidate_routes.values())
    if "PREFIX_LEAKAGE_SUPPORTED" in routes:
        return "FREEZE_NEW_SERIALIZATION_ALIGNED_ZERO_UPDATE_FEASIBILITY"
    if "TRANSFER_SURVIVES_RESPONSE_ONLY" in routes:
        return "SERIALIZATION_NOT_SUFFICIENT_EXPLANATION_CLOSE_PCR"
    if routes == {"NO_OUT_OF_SAMPLE_CONDITIONALITY_TRANSFER"}:
        return "NO_TRANSFER_CLOSE_PCR"
    return "INCONCLUSIVE_NO_NEW_TRAINING"


def run(protocol: dict[str, Any], protocol_path: Path, output_dir: Path) -> None:
    preflight = validate_preflight(protocol, protocol_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    set_seed(int(protocol["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model_name_or_path"], trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    split_rows = pcr.read_jsonl(resolve_path(protocol["split_path"]))
    packet = pcr.read_jsonl(resolve_path(protocol["mismatch_packet_path"]))
    response_maps = _response_maps(protocol, split_rows)

    entries = []
    for packet_row in packet:
        sample_id = str(packet_row["sample_id"])
        prompts = [packet_row["correct_prompt"], *packet_row["wrong_prompts"]]
        for source, responses in response_maps.items():
            response = responses[sample_id]
            for prompt_index, prompt in enumerate(prompts):
                feature = tokenize_entry(
                    tokenizer, str(prompt), response, int(protocol["max_length"])
                )
                feature.update(
                    {
                        "response_source": source,
                        "sample_id": sample_id,
                        "prompt_role": "correct" if prompt_index == 0 else f"wrong_{prompt_index}",
                    }
                )
                entries.append(feature)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        protocol["model_name_or_path"],
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = None
    for model_index, (model_name, spec) in enumerate(protocol["models"].items()):
        checkpoint = resolve_path(spec["checkpoint_dir"])
        if model is None:
            model = PeftModel.from_pretrained(
                base, checkpoint, adapter_name=model_name, is_trainable=False, local_files_only=True
            )
        else:
            model.load_adapter(
                checkpoint, adapter_name=model_name, is_trainable=False, low_cpu_mem_usage=True
            )
        if model_index == 0:
            model.eval()
    if model is None:
        raise ValueError("protocol contains no models")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    score_rows: list[dict[str, Any]] = []
    batch_size = int(protocol["batch_size"])
    for model_name in protocol["models"]:
        model.set_adapter(model_name)
        model.eval()
        for start in range(0, len(entries), batch_size):
            batch = entries[start : start + batch_size]
            scored = score_batch(model, batch, int(tokenizer.pad_token_id), bool(protocol["fp16"]))
            for entry, components in zip(batch, scored, strict=True):
                row = {
                    "model": model_name,
                    "response_source": entry["response_source"],
                    "sample_id": entry["sample_id"],
                    "prompt_role": entry["prompt_role"],
                    "direct_prefix_tokens": entry["direct_prefix_tokens"],
                    "forced_prefix_tokens": entry["forced_prefix_tokens"],
                }
                for component in COMPONENTS:
                    row[f"{component}_nll"] = components[component]["nll"]
                    row[f"{component}_tokens"] = components[component]["tokens"]
                score_rows.append(row)
            if (start // batch_size + 1) % 30 == 0:
                print(f"{model_name} batches {start // batch_size + 1}/{math.ceil(len(entries) / batch_size)}", flush=True)

    del model
    del base
    gc.collect()
    torch.cuda.empty_cache()

    scores_path = output_dir / "serialization_component_scores.csv"
    margins_path = output_dir / "serialization_margins.csv"
    write_csv(scores_path, score_rows)
    margins = margin_rows(score_rows)
    write_csv(margins_path, margins)

    samples = int(protocol["bootstrap_samples"])
    seed = int(protocol["bootstrap_seed"])
    transfers = {}
    routes = {}
    candidates = list(protocol["candidate_models"])
    for index, candidate in enumerate(candidates):
        primary = paired_transfer(margins, candidate, "reference", samples, seed + index * 100)
        generated = paired_transfer(margins, candidate, candidate, samples, seed + 50 + index * 100)
        transfers[candidate] = {
            "shared_reference_response": primary,
            "candidate_generated_response": generated,
        }
        routes[candidate] = candidate_route(primary)

    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    decision = {
        "name": "step3_pcr_serialization_transfer_decision_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "git_head": git_head(),
        "protocol": protocol_path.relative_to(REPO).as_posix(),
        "protocol_sha256": pcr.sha256_file(protocol_path),
        "preflight": preflight,
        "route": overall_route(routes),
        "candidate_routes": routes,
        "paired_transfer": transfers,
        "score_rows": len(score_rows),
        "margin_rows": len(margins),
        "scores_sha256": pcr.sha256_file(scores_path),
        "margins_sha256": pcr.sha256_file(margins_path),
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_vram,
        "wall_clock_seconds": time.time() - started,
        "interpretation_boundary": protocol["interpretation_boundary"],
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    pcr.write_json(output_dir / "serialization_transfer_decision.json", decision)
    print(json.dumps({"route": decision["route"], "candidate_routes": routes}))


def main() -> None:
    args = parse_args()
    protocol_path = resolve_path(args.protocol)
    protocol = pcr.read_json(protocol_path)
    if args.command == "freeze-packet":
        freeze_packet(protocol)
        return
    if args.command == "preflight":
        print(json.dumps(validate_preflight(protocol, protocol_path), ensure_ascii=False))
        return
    if not args.output_dir:
        raise ValueError("run requires --output-dir")
    run(protocol, protocol_path, resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
