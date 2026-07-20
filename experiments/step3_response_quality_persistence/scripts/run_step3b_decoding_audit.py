from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_seed(base: int, condition: str, sample_id: str, replicate: int) -> int:
    material = f"{base}|{condition}|{sample_id}|{replicate}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16) % (2**31 - 1)


def validate_config(config: dict[str, Any]) -> None:
    conditions = config["conditions"]
    names = [condition["name"] for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    if int(config["eval_limit"]) != 100 or int(config["samples_per_prompt"]) != 5:
        raise ValueError("frozen design requires 100 prompts and 5 samples per prompt")
    for condition in conditions:
        if float(condition["temperature"]) <= 0:
            raise ValueError("sampling condition temperature must be positive")
        if not 0 < float(condition["top_p"]) <= 1:
            raise ValueError("top_p must be in (0, 1]")
    if config.get("heldout_test_opened") is not False:
        raise ValueError("heldout_test_opened must remain false")
    if config.get("new_training_authorized") is not False:
        raise ValueError("this script does not authorize training")


def check_id_boundaries(config: dict[str, Any], eval_rows: list[dict[str, Any]]) -> dict[str, int]:
    with Path(config["split_manifest"]).open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    core_ids = {row["sample_id"] for row in eval_rows}
    split_ids: dict[str, set[str]] = {}
    for row in split_rows:
        split_ids.setdefault(row["split"], set()).add(row["sample_id"])
    overlaps = {split: len(core_ids & ids) for split, ids in split_ids.items()}
    if any(overlaps.values()):
        raise ValueError(f"core100 overlaps Step2G split IDs: {overlaps}")
    return overlaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = read_json(config_path)
    validate_config(config)
    eval_path = Path(config["eval_path"])
    eval_rows = read_jsonl(eval_path, limit=int(config["eval_limit"]))
    if len(eval_rows) != int(config["eval_limit"]):
        raise ValueError(f"expected {config['eval_limit']} eval rows, found {len(eval_rows)}")
    overlaps = check_id_boundaries(config, eval_rows)

    requested = set(args.condition)
    conditions = [
        condition
        for condition in config["conditions"]
        if not requested or condition["name"] in requested
    ]
    if requested - {condition["name"] for condition in conditions}:
        raise ValueError(f"unknown conditions: {sorted(requested - {c['name'] for c in conditions})}")

    checkpoint = Path(config["checkpoint_dir"])
    output_dir = Path(config["output_dir"])
    print("config:", config_path)
    print("checkpoint:", checkpoint)
    print("checkpoint_exists:", checkpoint.exists())
    print("eval_rows:", len(eval_rows))
    print("Step2G split overlaps:", overlaps)
    print("conditions:", [condition["name"] for condition in conditions])
    print("planned_rows:", len(conditions) * len(eval_rows) * int(config["samples_per_prompt"]))
    if args.dry_run:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found on this machine: {checkpoint}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("Missing torch/transformers/peft/bitsandbytes generation dependencies") from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if config.get("load_in_4bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    for condition in conditions:
        output_path = output_dir / "generations" / f"{condition['name']}.jsonl"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite explicitly")
        generated = []
        for row in eval_rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = f"User: {row['prompt']}\nAssistant:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            for replicate in range(int(config["samples_per_prompt"])):
                seed = generation_seed(
                    int(config["generation_seed_base"]),
                    condition["name"],
                    row["sample_id"],
                    replicate,
                )
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=int(config["max_new_tokens"]),
                        do_sample=True,
                        temperature=float(condition["temperature"]),
                        top_p=float(condition["top_p"]),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0][prompt_tokens:]
                generated.append(
                    {
                        "generation_id": f"{condition['name']}|{row['sample_id']}|r{replicate}",
                        "condition": condition["name"],
                        "temperature": condition["temperature"],
                        "top_p": condition["top_p"],
                        "generation_seed": seed,
                        "replicate": replicate,
                        "sample_id": row["sample_id"],
                        "checkpoint": config["checkpoint"],
                        "family": config["family"],
                        "model_seed": config["seed"],
                        "prompt": row["prompt"],
                        "response": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                        "generated_tokens": int(new_tokens.shape[-1]),
                    }
                )
        write_jsonl(output_path, generated)
        run_records.append(
            {
                "condition": condition["name"],
                "rows": len(generated),
                "path": str(output_path),
                "sha256": sha256(output_path),
            }
        )
        print(f"wrote {output_path} ({len(generated)} rows)")

    manifest = {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_rows": len(eval_rows),
        "split_overlaps": overlaps,
        "heldout_content_opened": False,
        "new_training_run": False,
        "outputs": run_records,
    }
    manifest_path = output_dir / "step3b_decoding_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
