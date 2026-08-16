"""Freeze and run the zero-training TRIM-CS0 objective causal preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import step3_trim_cs0_common as trim


REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze-pairs", "preflight", "run-objective"])
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def resolve_path(value: str) -> Path:
    return (REPO / value).resolve()


def freeze_pairs(registry: dict[str, Any], registry_path: Path) -> None:
    target_path = resolve_path(registry["source_targets_path"])
    packet_path = resolve_path(registry["pair_packet_path"])
    manifest_path = packet_path.with_name("trim_cs0_pairs_manifest.json")
    if packet_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite frozen TRIM pair artifacts")
    if trim.sha256_file(target_path) != registry["source_targets_sha256"]:
        raise ValueError("source target hash mismatch")
    rows = trim.build_pair_rows(
        trim.read_jsonl(target_path),
        {str(key): int(value) for key, value in registry["category_pair_quotas"].items()},
        str(registry["transition_marker"]),
    )
    trim.write_jsonl(packet_path, rows)
    manifest = {
        "name": "step3_trim_cs0_pair_packet_manifest_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "git_head": git_head(),
        "registry_path": registry_path.relative_to(REPO).as_posix(),
        "registry_sha256_before_packet_hash_freeze": trim.sha256_file(registry_path),
        "source_targets_sha256": trim.sha256_file(target_path),
        "pair_packet_path": packet_path.relative_to(REPO).as_posix(),
        "pair_packet_sha256": trim.sha256_file(packet_path),
        "pair_count": len({row["pair_id"] for row in rows}),
        "directional_instance_count": len(rows),
        "category_counts": {
            category: len({row["pair_id"] for row in rows if row["category"] == category})
            for category in sorted({str(row["category"]) for row in rows})
        },
        "unique_instance_ids": len({row["instance_id"] for row in rows}),
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    trim.write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def validate_preflight(registry: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    target_path = resolve_path(registry["source_targets_path"])
    packet_path = resolve_path(registry["pair_packet_path"])
    checkpoint = resolve_path(registry["checkpoint_dir"])
    adapter_path = checkpoint / "adapter_model.safetensors"
    rows = trim.read_jsonl(packet_path)
    checks = {
        "source_target_hash": trim.sha256_file(target_path) == registry["source_targets_sha256"],
        "pair_packet_hash": trim.sha256_file(packet_path) == registry["pair_packet_sha256"],
        "adapter_hash": trim.sha256_file(adapter_path) == registry["adapter_sha256"],
        "pair_count": len({row["pair_id"] for row in rows}) == int(registry["pair_count"]),
        "directional_instance_count": len(rows) == int(registry["directional_instance_count"]),
        "instance_ids_unique": len({row["instance_id"] for row in rows}) == len(rows),
        "both_directions": all(
            {row["direction"] for row in rows if row["pair_id"] == pair_id}
            == {"a_to_b", "b_to_a"}
            for pair_id in {row["pair_id"] for row in rows}
        ),
        "five_categories": len({row["category"] for row in rows}) == 5,
        "optimizer_steps_zero": int(registry["optimizer_steps"]) == 0,
        "training_not_authorized": registry["training_authorized"] is False,
        "core100_sealed": registry["core100_opened"] is False,
        "heldout_sealed": registry["heldout_opened"] is False,
    }
    token_alignment_rows = 0
    token_length_max = 0
    if all(checks.values()):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            registry["model_name_or_path"], trust_remote_code=True, local_files_only=True
        )
        try:
            for row in rows:
                context = _tokenized_context(
                    tokenizer,
                    str(row["recipient_prompt"]),
                    str(row["forced_prefix"]),
                    int(registry["max_length"]),
                )
                full = _full_inputs(
                    tokenizer, context, str(row["target_suffix"]), int(registry["max_length"])
                )
                token_alignment_rows += 1
                token_length_max = max(token_length_max, len(full["input_ids"]))
        except (ValueError, IndexError):
            pass
    checks["token_alignment_all_rows"] = token_alignment_rows == len(rows)
    checks["token_length_within_limit"] = token_length_max <= int(registry["max_length"])
    result = {
        "name": "step3_trim_cs0_preflight_v1",
        "registry_path": registry_path.relative_to(REPO).as_posix(),
        "registry_sha256": trim.sha256_file(registry_path),
        "checks": checks,
        "passed": all(checks.values()),
        "pair_packet_sha256": trim.sha256_file(packet_path),
        "adapter_sha256": trim.sha256_file(adapter_path),
        "rows": len(rows),
        "token_alignment_rows": token_alignment_rows,
        "maximum_full_token_length": token_length_max,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    if not result["passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"TRIM preflight failed: {failed}")
    return result


def _layer_module(model: Any, index: int) -> Any:
    base = model.get_base_model()
    backbone = getattr(base, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise AttributeError("could not locate decoder layers")
    return layers[index], len(layers)


def _tokenized_context(tokenizer: Any, prompt: str, forced_prefix: str, max_length: int) -> dict[str, Any]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = prompt_text + forced_prefix
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt_start = prompt_text.find(prompt)
    if prompt_start < 0:
        raise ValueError("prompt span not found in chat template")
    prompt_span = (prompt_start, prompt_start + len(prompt))
    prefix_span = (len(prompt_text), len(text))

    def overlapping(span: tuple[int, int]) -> list[int]:
        start, end = span
        return [
            index
            for index, (left, right) in enumerate(encoded["offset_mapping"])
            if right > start and left < end
        ]

    prompt_positions = overlapping(prompt_span)
    prefix_positions = overlapping(prefix_span)
    if len(prompt_positions) < 2 or len(prefix_positions) < 2:
        raise ValueError("prompt or transition window has fewer than two tokens")
    return {
        "text": text,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "transition_positions": prefix_positions[-2:],
        "prompt_positions": prompt_positions[-2:],
    }


def _full_inputs(tokenizer: Any, context: dict[str, Any], suffix: str, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        context["text"] + suffix,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    context_ids = context["input_ids"]
    if encoded["input_ids"][: len(context_ids)] != context_ids:
        raise ValueError("suffix tokenization changed the frozen context tokenization")
    if len(encoded["input_ids"]) <= len(context_ids):
        raise ValueError("target suffix was fully truncated")
    return {**encoded, "suffix_start": len(context_ids)}


def _tensors(feature: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device),
    }


def _forward(
    model: Any,
    layer: Any,
    feature: dict[str, Any],
    capture_positions: list[int] | None = None,
    patch_positions: list[int] | None = None,
    replacement: Any | None = None,
) -> tuple[Any, Any | None]:
    import torch

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if capture_positions is not None:
            captured["state"] = hidden[0, capture_positions, :].detach().clone()
        if patch_positions is not None:
            if replacement is None:
                raise ValueError("patch replacement is missing")
            return trim.patch_layer_output(output, patch_positions, replacement)
        return output

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(**_tensors(feature, model.device), use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return output.logits, captured.get("state")


def _suffix_logits(logits: Any, suffix_start: int, full_length: int) -> Any:
    return logits[:, suffix_start - 1 : full_length - 1, :]


def _suffix_nll(logits: Any, feature: dict[str, Any]) -> float:
    import torch

    selected = _suffix_logits(logits, int(feature["suffix_start"]), len(feature["input_ids"]))
    targets = torch.tensor(
        feature["input_ids"][int(feature["suffix_start"]) :],
        dtype=torch.long,
        device=selected.device,
    )
    return float(
        torch.nn.functional.cross_entropy(
            selected.reshape(-1, selected.shape[-1]).float(), targets.reshape(-1), reduction="mean"
        )
    )


def _capture_context_state(model: Any, layer: Any, context: dict[str, Any], key: str) -> Any:
    feature = {
        "input_ids": context["input_ids"],
        "attention_mask": context["attention_mask"],
    }
    _, state = _forward(model, layer, feature, capture_positions=context[key])
    if state is None:
        raise RuntimeError("activation capture produced no state")
    return state


def _patched_nll(
    model: Any,
    layer: Any,
    feature: dict[str, Any],
    positions: list[int],
    replacement: Any,
) -> tuple[float, Any]:
    logits, _ = _forward(
        model, layer, feature, patch_positions=positions, replacement=replacement
    )
    return _suffix_nll(logits, feature), logits


def _load_model(registry: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(int(registry["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        registry["model_name_or_path"], trust_remote_code=True, local_files_only=True
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        registry["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base, resolve_path(registry["checkpoint_dir"]), is_trainable=False, local_files_only=True
    )
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def _bootstrap(rows: list[dict[str, Any]], field: str, registry: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["valid"]:
            grouped.setdefault(str(row["pair_id"]), []).append(float(row[field]))
    return trim.paired_bootstrap_mean_ci(
        grouped, seed=int(registry["bootstrap_seed"]), samples=int(registry["bootstrap_samples"])
    )


def run_objective(
    registry: dict[str, Any], registry_path: Path, output_dir: Path, limit: int | None
) -> None:
    import torch

    preflight = validate_preflight(registry, registry_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = trim.read_jsonl(resolve_path(registry["pair_packet_path"]))
    rows = all_rows[:limit] if limit is not None else all_rows
    if limit is not None and (limit <= 0 or limit >= len(all_rows)):
        raise ValueError("smoke limit must be positive and smaller than the full packet")
    model, tokenizer = _load_model(registry)
    layer, layer_count = _layer_module(model, int(registry["patch_layer_zero_based"]))
    if layer_count != int(registry["model_layer_count"]):
        raise ValueError("model layer count mismatch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        result: dict[str, Any] = {
            "instance_id": row["instance_id"],
            "pair_id": row["pair_id"],
            "direction": row["direction"],
            "category": row["category"],
            "valid": False,
        }
        try:
            prefix = str(row["forced_prefix"])
            suffix = str(row["target_suffix"])
            max_length = int(registry["max_length"])
            correct_context = _tokenized_context(
                tokenizer, str(row["correct_donor_prompt"]), prefix, max_length
            )
            recipient_context = _tokenized_context(
                tokenizer, str(row["recipient_prompt"]), prefix, max_length
            )
            cross_context = _tokenized_context(
                tokenizer, str(row["cross_wrong_prompt"]), prefix, max_length
            )
            recipient_full = _full_inputs(tokenizer, recipient_context, suffix, max_length)
            correct_full = _full_inputs(tokenizer, correct_context, suffix, max_length)

            correct_transition = _capture_context_state(
                model, layer, correct_context, "transition_positions"
            )
            paired_transition = _capture_context_state(
                model, layer, recipient_context, "transition_positions"
            )
            cross_transition = _capture_context_state(
                model, layer, cross_context, "transition_positions"
            )
            correct_placebo = _capture_context_state(model, layer, correct_context, "prompt_positions")
            paired_placebo = _capture_context_state(model, layer, recipient_context, "prompt_positions")
            cross_placebo = _capture_context_state(model, layer, cross_context, "prompt_positions")

            baseline_logits, noop_state = _forward(
                model,
                layer,
                recipient_full,
                capture_positions=recipient_context["transition_positions"],
            )
            if noop_state is None:
                raise RuntimeError("no-op state capture failed")
            no_patch_nll = _suffix_nll(baseline_logits, recipient_full)
            noop_nll, noop_logits = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["transition_positions"],
                noop_state,
            )
            base_suffix_logits = _suffix_logits(
                baseline_logits, recipient_full["suffix_start"], len(recipient_full["input_ids"])
            )
            noop_suffix_logits = _suffix_logits(
                noop_logits, recipient_full["suffix_start"], len(recipient_full["input_ids"])
            )
            noop_error = float((base_suffix_logits.float() - noop_suffix_logits.float()).abs().max())

            correct_nll, _ = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["transition_positions"],
                correct_transition,
            )
            paired_wrong_nll, _ = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["transition_positions"],
                paired_transition,
            )
            cross_wrong_nll, _ = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["transition_positions"],
                cross_transition,
            )
            placebo_correct_nll, _ = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["prompt_positions"],
                correct_placebo,
            )
            placebo_paired_nll, _ = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["prompt_positions"],
                paired_placebo,
            )
            placebo_cross_nll, _ = _patched_nll(
                model,
                layer,
                recipient_full,
                recipient_context["prompt_positions"],
                cross_placebo,
            )

            natural_logits, _ = _forward(model, layer, correct_full)
            natural_nll = _suffix_nll(natural_logits, correct_full)
            corruption_nll, _ = _patched_nll(
                model,
                layer,
                correct_full,
                correct_context["transition_positions"],
                paired_transition,
            )
            restoration_nll, _ = _patched_nll(
                model,
                layer,
                correct_full,
                correct_context["transition_positions"],
                correct_transition,
            )
            result.update(
                {
                    "valid": True,
                    "context_tokens": len(recipient_context["input_ids"]),
                    "suffix_tokens": len(recipient_full["input_ids"])
                    - int(recipient_full["suffix_start"]),
                    "no_patch_nll": no_patch_nll,
                    "noop_patch_nll": noop_nll,
                    "noop_suffix_max_logit_error": noop_error,
                    "correct_donor_nll": correct_nll,
                    "paired_wrong_donor_nll": paired_wrong_nll,
                    "cross_wrong_donor_nll": cross_wrong_nll,
                    "correct_vs_mean_wrong_nll_gain": (
                        (paired_wrong_nll + cross_wrong_nll) / 2.0 - correct_nll
                    ),
                    "placebo_correct_nll": placebo_correct_nll,
                    "placebo_paired_wrong_nll": placebo_paired_nll,
                    "placebo_cross_wrong_nll": placebo_cross_nll,
                    "placebo_correct_vs_mean_wrong_nll_gain": (
                        (placebo_paired_nll + placebo_cross_nll) / 2.0
                        - placebo_correct_nll
                    ),
                    "natural_correct_nll": natural_nll,
                    "corruption_nll": corruption_nll,
                    "restoration_nll": restoration_nll,
                    "necessity_nll_increase": corruption_nll - natural_nll,
                    "restoration_nll_gain": corruption_nll - restoration_nll,
                }
            )
        except (ValueError, RuntimeError, IndexError) as error:
            result["failure_type"] = type(error).__name__
            result["failure_reason"] = str(error)
        results.append(result)
        print(f"TRIM-CS0 objective {index}/{len(rows)} valid={result['valid']}", flush=True)

    trim.write_jsonl(output_dir / "objective_rows.jsonl", results)
    valid_rows = [row for row in results if row["valid"]]
    valid_rate = len(valid_rows) / len(results)
    summary: dict[str, Any] = {
        "valid_alignment_rate": valid_rate,
        "noop_suffix_max_logit_error": max(
            (float(row["noop_suffix_max_logit_error"]) for row in valid_rows), default=float("inf")
        ),
    }
    if valid_rows:
        summary["correct_effect"] = _bootstrap(
            valid_rows, "correct_vs_mean_wrong_nll_gain", registry
        )
        summary["placebo_effect"] = _bootstrap(
            valid_rows, "placebo_correct_vs_mean_wrong_nll_gain", registry
        )
        summary["necessity_effect"] = _bootstrap(valid_rows, "necessity_nll_increase", registry)
        summary["restoration_effect"] = _bootstrap(valid_rows, "restoration_nll_gain", registry)
        summary["transition_to_placebo_effect_ratio"] = float(
            summary["correct_effect"]["mean"]
        ) / max(abs(float(summary["placebo_effect"]["mean"])), 1e-12)

    smoke_only = limit is not None
    gates: dict[str, bool] = {}
    if not smoke_only and valid_rows:
        thresholds = registry["objective_gates"]
        gates = {
            "valid_alignment_rate": valid_rate >= thresholds["valid_alignment_rate_min"],
            "noop_suffix_max_logit_error": summary["noop_suffix_max_logit_error"]
            <= thresholds["noop_suffix_max_logit_error_max"],
            "correct_nll_gain": summary["correct_effect"]["mean"]
            >= thresholds["correct_vs_mean_wrong_nll_gain_min"],
            "correct_bootstrap_lower": summary["correct_effect"]["ci95_lower"]
            > thresholds["correct_vs_mean_wrong_bootstrap_lower_gt"],
            "transition_vs_placebo": summary["transition_to_placebo_effect_ratio"]
            >= thresholds["transition_to_placebo_effect_ratio_min"],
            "necessity_nll_increase": summary["necessity_effect"]["mean"]
            >= thresholds["necessity_nll_increase_min"],
            "necessity_bootstrap_lower": summary["necessity_effect"]["ci95_lower"]
            > thresholds["necessity_bootstrap_lower_gt"],
            "restoration_nll_gain": summary["restoration_effect"]["mean"]
            >= thresholds["restoration_nll_gain_min"],
            "restoration_bootstrap_lower": summary["restoration_effect"]["ci95_lower"]
            > thresholds["restoration_bootstrap_lower_gt"],
        }
    route = "SMOKE_ONLY" if smoke_only else (
        "PASS_OBJECTIVE_AUTHORIZE_BLIND_BEHAVIOR_REVIEW"
        if gates and all(gates.values())
        else "STOP_TRIM_CS0_OBJECTIVE_FAIL"
    )
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3_trim_cs0_objective_result_v1",
        "experiment_mode": "exploratory_non_confirmatory",
        "git_head": git_head(),
        "registry_path": registry_path.relative_to(REPO).as_posix(),
        "registry_sha256": trim.sha256_file(registry_path),
        "preflight": preflight,
        "smoke_only": smoke_only,
        "requested_limit": limit,
        "rows": len(results),
        "valid_rows": len(valid_rows),
        "summary": summary,
        "gates": gates,
        "route": route,
        "objective_rows_sha256": trim.sha256_file(output_dir / "objective_rows.jsonl"),
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_vram,
        "wall_clock_seconds": time.time() - started,
        "candidate_outputs_generated": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    trim.write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"route": route, "summary": summary, "gates": gates}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    registry_path = resolve_path(args.registry)
    registry = trim.read_json(registry_path)
    if args.command == "freeze-pairs":
        freeze_pairs(registry, registry_path)
    elif args.command == "preflight":
        print(json.dumps(validate_preflight(registry, registry_path), ensure_ascii=False))
    else:
        if not args.output_dir:
            raise ValueError("run-objective requires --output-dir")
        run_objective(registry, registry_path, resolve_path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()
