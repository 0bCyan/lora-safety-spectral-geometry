from __future__ import annotations

import argparse
from pathlib import Path

from common import FAMILY_TO_FILE, adapter_dir, load_config, log_dir, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_TO_FILE))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def format_prompt_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"用户：{prompt}\n助手："


def format_chat_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"{format_prompt_text(tokenizer, prompt)}{response}"


def make_tokenized_row(tokenizer, row: dict, max_length: int) -> dict:
    prompt_text = format_prompt_text(tokenizer, row["prompt"])
    full_text = format_chat_text(tokenizer, row["prompt"], row["response"])
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    labels = list(full["input_ids"])
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len
    full["labels"] = labels
    return full


def collate_batch(tokenizer, features: list[dict]) -> dict:
    import torch

    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_mask = []
    labels = []
    pad_id = tokenizer.pad_token_id
    for feature in features:
        length = len(feature["input_ids"])
        pad_len = max_len - length
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_path = Path(config["data_dir"]) / FAMILY_TO_FILE[args.family]
    rows = read_jsonl(data_path, limit=config["train_per_family"])
    out_dir = adapter_dir(config, args.family, args.seed)
    logs = log_dir(config, args.family, args.seed)

    print("config:", args.config)
    print("family:", args.family)
    print("seed:", args.seed)
    print("model:", config["model_name_or_path"])
    print("train_rows:", len(rows))
    print("adapter_dir:", out_dir)

    if args.dry_run:
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, transformers, peft, "
            "datasets, accelerate and bitsandbytes on the GPU machine."
        ) from exc

    set_seed(args.seed)

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

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_rows = [
        make_tokenized_row(tokenizer, row, config["max_length"]) for row in rows
    ]
    tokenized = Dataset.from_list(tokenized_rows)

    training_args = TrainingArguments(
        output_dir=str(logs),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_strategy="no" if config.get("save_steps", 0) == 0 else "steps",
        save_steps=config.get("save_steps", 0) or 500,
        bf16=config.get("bf16", False),
        fp16=config.get("fp16", False),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=lambda features: collate_batch(tokenizer, features),
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved adapter to {out_dir}")


if __name__ == "__main__":
    main()
