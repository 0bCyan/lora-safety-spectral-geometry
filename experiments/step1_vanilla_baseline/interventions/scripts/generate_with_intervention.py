from __future__ import annotations

import argparse
import csv
from pathlib import Path

from intervention_common import (
    EVAL_TO_FILE,
    experiment_id,
    load_json,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--intervention", required=True, choices=["top", "bottom", "random"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_selected_layer_keys(path: Path, exp_id: str) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {
            row["layer_key"]
            for row in csv.DictReader(handle)
            if row["experiment_id"] == exp_id
        }


def load_module_filter(path: Path, exp_id: str) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["experiment_id"] == exp_id:
                return row.get("module_filter", "")
    return ""


def adapter_dir(base_config: dict, family: str, seed: int) -> Path:
    model = base_config["model_name_or_path"].split("/")[-1].lower()
    exp_id = f"step1_{base_config['machine']}_{model}_{family}_s{seed}"
    return Path(base_config["output_dir"]) / "adapters" / exp_id


def output_path(config: dict, seed: int, intervention: str) -> Path:
    exp_id = experiment_id(
        config["family"],
        seed,
        intervention,
        int(config["k"]),
        float(config["scale"]),
    )
    return Path(config["output_dir"]) / "generations" / f"{exp_id}_{config['eval_set']}.jsonl"


def maybe_scale_lora_layers(model, selected_keys: set[str], scale: float) -> list[str]:
    scaled = []
    for name, module in model.named_modules():
        if name not in selected_keys:
            continue
        if hasattr(module, "lora_B"):
            for adapter_name, lora_b in module.lora_B.items():
                lora_b.weight.data.mul_(scale)
                scaled.append(f"{name}.lora_B.{adapter_name}")
    return scaled


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(config["base_config"])
    exp_id = experiment_id(
        config["family"],
        args.seed,
        args.intervention,
        int(config["k"]),
        float(config["scale"]),
    )
    selected_path = Path(config["output_dir"]) / "selected_layers.csv"
    selected_keys = load_selected_layer_keys(selected_path, exp_id)
    module_filter = load_module_filter(selected_path, exp_id)
    eval_path = Path(base_config["data_dir"]) / EVAL_TO_FILE[config["eval_set"]]
    eval_rows = read_jsonl(eval_path, limit=base_config["eval_per_set"])
    adapter = adapter_dir(base_config, config["family"], args.seed)
    out_path = output_path(config, args.seed, args.intervention)

    print("experiment_id:", exp_id)
    print("adapter:", adapter)
    print("selected_layers:", len(selected_keys))
    print("eval_rows:", len(eval_rows))
    print("output:", out_path)
    if args.dry_run:
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "Missing generation dependencies. Install torch, transformers, peft and bitsandbytes."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        base_config["model_name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if base_config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if base_config.get("bf16") else torch.float16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_config["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    scaled = maybe_scale_lora_layers(model, selected_keys, float(config["scale"]))
    print("scaled_lora_B:", len(scaled))
    model.eval()

    generated = []
    for row in eval_rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"User: {row['prompt']}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=base_config["generation"]["max_new_tokens"],
                do_sample=base_config["generation"]["temperature"] > 0,
                temperature=max(base_config["generation"]["temperature"], 1e-5),
                top_p=base_config["generation"]["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated.append(
            {
                "sample_id": row["sample_id"],
                "eval_set": config["eval_set"],
                "train_family": config["family"],
                "seed": args.seed,
                "intervention": args.intervention,
                "module_filter": module_filter,
                "k": config["k"],
                "scale": config["scale"],
                "prompt": row["prompt"],
                "response": response,
            }
        )
    write_jsonl(out_path, generated)
    print(f"wrote {out_path} ({len(generated)} rows)")


if __name__ == "__main__":
    main()
