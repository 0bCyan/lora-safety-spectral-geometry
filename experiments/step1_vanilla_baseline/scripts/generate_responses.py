from __future__ import annotations

import argparse
from pathlib import Path

from common import EVAL_TO_FILE, adapter_dir, generation_path, load_config, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--eval-set", required=True, choices=sorted(EVAL_TO_FILE))
    parser.add_argument(
        "--base",
        action="store_true",
        help="Generate with the base model only. Use --family base --seed 0.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_path = Path(config["data_dir"]) / EVAL_TO_FILE[args.eval_set]
    rows = read_jsonl(eval_path, limit=config["eval_per_set"])
    adapter = None if args.base else adapter_dir(config, args.family, args.seed)
    output = generation_path(config, args.family, args.seed, args.eval_set)

    print("model:", config["model_name_or_path"])
    print("adapter:", adapter if adapter is not None else "base model only")
    print("eval_rows:", len(rows))
    print("output:", output)

    if args.dry_run:
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not args.base:
            from peft import PeftModel
    except Exception as exc:
        raise RuntimeError(
            "Missing generation dependencies. Install torch, transformers, peft "
            "and bitsandbytes on the GPU machine."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.get("load_in_4bit", False):
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
    model = base_model if args.base else PeftModel.from_pretrained(base_model, adapter)
    model.eval()

    generated_rows = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"用户：{row['prompt']}\n助手："
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                do_sample=config["generation"]["temperature"] > 0,
                temperature=max(config["generation"]["temperature"], 1e-5),
                top_p=config["generation"]["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_rows.append(
            {
                "sample_id": row["sample_id"],
                "eval_set": args.eval_set,
                "train_family": args.family,
                "seed": args.seed,
                "prompt": row["prompt"],
                "response": response,
            }
        )

    write_jsonl(output, generated_rows)
    print(f"wrote {output} ({len(generated_rows)} rows)")


if __name__ == "__main__":
    main()
