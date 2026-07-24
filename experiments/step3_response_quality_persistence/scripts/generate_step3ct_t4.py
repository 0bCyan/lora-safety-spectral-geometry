"""Generate one frozen Step3C-T4 arm/checkpoint/evaluation cell."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import step3ct_t4_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--eval-set", required=True, choices=["safety", "utility"])
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.REPO, text=True).strip()


def validate_inputs(registry_path: Path, arm: str, checkpoint: int, eval_set: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path, str]:
    registry = common.read_json(registry_path)
    for field in ["t3_route", "t3_manifest_package", "execution_config"]:
        path = common.REPO / registry[field]
        expected = registry[f"{field}_sha256"]
        if common.sha256_file(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {field}")
    route = common.read_json(common.REPO / registry["t3_route"])
    if route["route"] != "T3_integrity_pass_runs_frozen_authorize_T4" or not route["overall_pass"]:
        raise ValueError("T3 route does not authorize T4")
    if arm not in registry["arms"]:
        raise ValueError(f"unregistered arm: {arm}")
    if checkpoint not in registry["allowed_checkpoints"][eval_set]:
        raise ValueError(f"unregistered checkpoint for {eval_set}: {checkpoint}")

    input_spec = registry["inputs"][eval_set]
    input_path = common.REPO / input_spec["path"]
    if common.sha256_file(input_path) != input_spec["file_sha256"]:
        raise ValueError(f"{eval_set} input file hash mismatch")
    rows = common.read_jsonl(input_path)[: int(input_spec["selected_rows"])]
    if len(rows) != int(input_spec["selected_rows"]) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"bad {eval_set} selected row count or duplicate sample IDs")
    if common.selected_sample_id_hash(rows) != input_spec["selected_sample_id_sha256"]:
        raise ValueError(f"{eval_set} selected sample order mismatch")

    run_root = Path(common.read_json(common.REPO / registry["execution_config"])["output_dir"])
    checkpoint_dir = common.REPO / run_root / "runs" / "t3_t2r" / arm / "checkpoints" / f"step{checkpoint:04d}"
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    expected_adapter = registry["adapter_sha256"][arm][str(checkpoint)]
    if not adapter_path.exists() or common.sha256_file(adapter_path) != expected_adapter:
        raise ValueError(f"adapter hash mismatch: {arm} step{checkpoint}")
    return registry, rows, checkpoint_dir, expected_adapter


def output_paths(registry: dict[str, Any], arm: str, checkpoint: int, eval_set: str) -> tuple[Path, Path]:
    root = common.REPO / registry["generation_root"] / arm
    stem = f"{arm}_step{checkpoint:04d}_{eval_set}100"
    return root / f"{stem}.jsonl", root / f"{stem}_manifest.json"


def main() -> None:
    args = parse_args()
    registry_path = (common.REPO / args.registry).resolve()
    registry, eval_rows, checkpoint_dir, adapter_sha = validate_inputs(
        registry_path, args.arm, args.checkpoint_step, args.eval_set
    )
    output_path, manifest_path = output_paths(registry, args.arm, args.checkpoint_step, args.eval_set)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen generation cell: {output_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    generation = registry["generation"]
    set_seed(int(generation["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        registry["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=bool(registry["local_files_only"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        registry["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=bool(registry["local_files_only"]),
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True

    started = time.time()
    generated_rows = []
    removed_blocks = 0
    removed_chars = 0
    for index, row in enumerate(eval_rows, start=1):
        prompt = str(row["prompt"])
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(generation["enable_thinking"]),
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(generation["max_input_length"]),
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        response, cleaning = common.clean_thinking_output(raw_response)
        if "<think>" in response.lower() or "</think>" in response.lower():
            raise ValueError(f"residual thinking tag in {row['sample_id']}")
        removed_blocks += int(cleaning["complete_thinking_blocks_removed"])
        removed_chars += int(cleaning["removed_character_count"])
        generated = {
            "generation_id": common.stable_rank(
                2026072403, args.arm, str(args.checkpoint_step), args.eval_set, str(row["sample_id"])
            )[:24],
            "sample_id": row["sample_id"],
            "arm": args.arm,
            "checkpoint_step": args.checkpoint_step,
            "eval_set": args.eval_set,
            "prompt": prompt,
            "response": response,
            "prompt_sha256": common.sha256_text(prompt),
            "response_sha256": common.sha256_text(response),
            **cleaning,
        }
        if args.eval_set == "utility":
            generated["reference_response"] = str(row.get("reference_response", ""))
            generated["reference_response_sha256"] = common.sha256_text(generated["reference_response"])
        generated_rows.append(generated)
        if index % 10 == 0:
            print(f"{args.arm} step{args.checkpoint_step} {args.eval_set} {index}/100", flush=True)

    common.write_jsonl(output_path, generated_rows)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    manifest = {
        "name": "step3ct_t4_generation_cell_manifest_v1",
        "git_head": git_head(),
        "registry_path": args.registry,
        "registry_sha256": common.sha256_file(registry_path),
        "arm": args.arm,
        "checkpoint_step": args.checkpoint_step,
        "adapter_sha256": adapter_sha,
        "eval_set": args.eval_set,
        "rows": len(generated_rows),
        "unique_sample_ids": len({row["sample_id"] for row in generated_rows}),
        "selected_sample_id_sha256": common.selected_sample_id_hash(generated_rows),
        "output_path": output_path.relative_to(common.REPO).as_posix(),
        "output_sha256": common.sha256_file(output_path),
        "output_canonical_hash": common.canonical_hash(generated_rows),
        "generation": generation,
        "thinking_blocks_removed": removed_blocks,
        "thinking_characters_removed": removed_chars,
        "thinking_content_persisted": False,
        "empty_response_count": sum(not row["response"].strip() for row in generated_rows),
        "wall_clock_seconds": time.time() - started,
        "peak_reserved_bytes": peak_reserved,
        "vram_headroom_fraction": 1.0 - peak_reserved / total_vram,
        "candidate_outputs_generated": True,
        "core100_opened": args.eval_set == "safety",
        "utility100_opened": args.eval_set == "utility",
        "heldout_opened": False,
    }
    common.write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output_path), "manifest": str(manifest_path), "rows": len(generated_rows)}))


if __name__ == "__main__":
    main()
