from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/step3_response_quality_persistence")
REGISTRY = ROOT / "configs/step3c_stage0_registry.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def paths(registry: dict[str, Any]) -> dict[str, Path]:
    out = Path(registry["output_dir"])
    return {
        "raw": out / "raw_generations",
        "reports": out / "reports",
        "diagnostics": out / "diagnostics",
        "generation": out / "raw_generations/constant20_safety_train_greedy.jsonl",
        "manifest": out / "raw_generations/constant20_safety_train_greedy_manifest.json",
        "environment": out / "raw_generations/constant20_safety_train_environment.json",
        "integrity": out / "diagnostics/step3c_stage0_generation_integrity.json",
        "positive_candidates": out / "raw_generations/step3c_stage0_prospective_positive_candidates.jsonl",
    }


def safety_train_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    eval_rows = {row["sample_id"]: row for row in read_jsonl(Path(registry["eval_safety_path"]))}
    split_rows = read_csv(Path(registry["split_manifest"]))
    ids = sorted(row["sample_id"] for row in split_rows if row["split"] == "safety_train")
    rows = [eval_rows[sample_id] for sample_id in ids]
    if len(rows) != int(registry["stage0_triplet_gates"]["safety_train_prompt_count"]):
        raise ValueError("expected exactly 40 safety_train prompts")
    return rows


def split_overlaps(registry: dict[str, Any], sample_ids: set[str]) -> dict[str, int]:
    eval_rows = read_jsonl(Path(registry["eval_safety_path"]))
    core100 = {row["sample_id"] for row in eval_rows[:100]}
    split_rows = read_csv(Path(registry["split_manifest"]))
    split_ids: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        split_ids[row["split"]].add(row["sample_id"])
    overlaps = {split: len(sample_ids & ids) for split, ids in split_ids.items()}
    overlaps["core100"] = len(sample_ids & core100)
    return overlaps


def dry_run(registry: dict[str, Any]) -> dict[str, Any]:
    rows = safety_train_rows(registry)
    sample_ids = {row["sample_id"] for row in rows}
    overlaps = split_overlaps(registry, sample_ids)
    if overlaps.get("core100") or overlaps.get("heldout_test") or overlaps.get("proxy_dev"):
        raise ValueError(f"unexpected overlap: {overlaps}")
    checkpoint = Path(registry["checkpoint_dir"])
    return {
        "registry": str(REGISTRY),
        "registry_sha256": sha256(REGISTRY),
        "head_before_generation": git_head(),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "safety_train_rows": len(rows),
        "unique_sample_ids": len(sample_ids),
        "split_overlaps": overlaps,
        "generation": registry["generation"],
        "heldout_opened": False,
        "training_authorized": False,
    }


def generate(registry: dict[str, Any], overwrite: bool) -> None:
    planned = dry_run(registry)
    if not planned["checkpoint_exists"]:
        raise FileNotFoundError(planned["checkpoint"])
    p = paths(registry)
    p["raw"].mkdir(parents=True, exist_ok=True)
    if p["generation"].exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {p['generation']}; pass --overwrite")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("missing generation dependencies") from exc

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        registry["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        registry["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, registry["checkpoint_dir"], local_files_only=True)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(registry["generation"]["generation_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(registry["generation"]["generation_seed"]))

    generated = []
    for row in safety_train_rows(registry):
        messages = [{"role": "user", "content": row["prompt"]}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"User: {row['prompt']}\nAssistant:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(registry["generation"]["max_new_tokens"]),
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][prompt_tokens:]
        generated.append(
            {
                "generation_id": f"constant20_safety_train_greedy|{row['sample_id']}|r0",
                "condition": registry["generation"]["condition"],
                "sample_id": row["sample_id"],
                "family": row.get("family", "safety"),
                "source": row.get("source", ""),
                "prompt": row["prompt"],
                "response": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                "generation_seed": int(registry["generation"]["generation_seed"]),
                "temperature": registry["generation"]["temperature"],
                "top_p": registry["generation"]["top_p"],
                "max_new_tokens": registry["generation"]["max_new_tokens"],
                "generated_tokens": int(new_tokens.shape[-1]),
            }
        )
    write_jsonl(p["generation"], generated)
    end = time.time()

    cuda = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory)
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        cuda = {
            "device_name": torch.cuda.get_device_name(0),
            "total_vram_bytes": total,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "reserved_headroom_fraction": (total - peak_reserved) / total,
            "required_headroom_fraction": 0.10,
            "headroom_pass": (total - peak_reserved) / total >= 0.10,
        }
    manifest = {
        **planned,
        "generation_path": str(p["generation"]),
        "generation_rows": len(generated),
        "generation_sha256": sha256(p["generation"]),
        "checkpoint_adapter_model_sha256": sha256(Path(registry["checkpoint_dir"]) / "adapter_model.safetensors"),
        "wall_clock_seconds": end - start,
        "new_training_run": False,
        "optimizer_created": False,
        "checkpoint_modified": False,
        "heldout_content_opened": False,
        "cuda": cuda,
    }
    p["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "python": os.sys.version,
        "git_head": git_head(),
        "cuda": cuda,
    }
    p["environment"].write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def validate(registry: dict[str, Any]) -> dict[str, Any]:
    p = paths(registry)
    rows = read_jsonl(p["generation"])
    prompt_rows = safety_train_rows(registry)
    expected_ids = {row["sample_id"] for row in prompt_rows}
    sample_ids = [row["sample_id"] for row in rows]
    generation_ids = [row["generation_id"] for row in rows]
    if len(rows) != 40:
        raise ValueError(f"expected 40 generation rows, found {len(rows)}")
    if set(sample_ids) != expected_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("generation sample IDs do not exactly match safety_train")
    if len(generation_ids) != len(set(generation_ids)):
        raise ValueError("duplicate generation_id")
    if any(row["condition"] != registry["generation"]["condition"] for row in rows):
        raise ValueError("wrong condition in generation rows")
    empty = sum(not str(row["response"]).strip() for row in rows)
    max_hits = sum(int(row["generated_tokens"]) >= int(registry["generation"]["max_new_tokens"]) for row in rows)
    overlaps = split_overlaps(registry, set(sample_ids))
    if overlaps.get("core100") or overlaps.get("heldout_test") or overlaps.get("proxy_dev"):
        raise ValueError(f"unexpected overlap: {overlaps}")

    positives = read_jsonl(p["positive_candidates"])
    positive_ids = [row["sample_id"] for row in positives]
    expected_positive_ids = {
        row["sample_id"]
        for row in read_csv(Path(registry["positive_coverage_csv"]))
        if row["requires_targeted_positive"] == "True"
    }
    if len(positives) != int(registry["stage0_triplet_gates"]["new_positive_candidate_count"]):
        raise ValueError("expected exactly seven prospective positives")
    if set(positive_ids) != expected_positive_ids or len(positive_ids) != len(set(positive_ids)):
        raise ValueError("prospective positives do not exactly match predeclared missing IDs")
    if any(not str(row["response"]).strip() for row in positives):
        raise ValueError("empty prospective positive")

    integrity = {
        "generation_rows": len(rows),
        "unique_generation_ids": len(set(generation_ids)),
        "unique_sample_ids": len(set(sample_ids)),
        "expected_safety_train_ids": len(expected_ids),
        "split_overlaps": overlaps,
        "empty_response_count": empty,
        "max_token_limit_hit_count": max_hits,
        "generation_sha256": sha256(p["generation"]),
        "manifest_sha256": sha256(p["manifest"]),
        "environment_sha256": sha256(p["environment"]),
        "prospective_positive_rows": len(positives),
        "prospective_positive_sha256": sha256(p["positive_candidates"]),
        "heldout_content_opened": False,
        "new_training_run": False,
        "validator_status": "passed",
    }
    p["diagnostics"].mkdir(parents=True, exist_ok=True)
    p["integrity"].write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(integrity, ensure_ascii=False, indent=2))
    return integrity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["dry-run", "generate", "validate"])
    parser.add_argument("--registry", default=str(REGISTRY))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    registry = read_json(Path(args.registry))
    if registry.get("heldout_opened") is not False or registry.get("training_authorized") is not False:
        raise ValueError("registry must keep heldout_opened=false and training_authorized=false")
    if args.command == "dry-run":
        print(json.dumps(dry_run(registry), ensure_ascii=False, indent=2))
    elif args.command == "generate":
        generate(registry, overwrite=args.overwrite)
    elif args.command == "validate":
        validate(registry)


if __name__ == "__main__":
    main()
