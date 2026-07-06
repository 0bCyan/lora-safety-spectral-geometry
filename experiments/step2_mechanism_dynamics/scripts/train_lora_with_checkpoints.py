from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import (
    FAMILY_TO_FILE,
    checkpoint_dir,
    checkpoint_tag,
    load_json,
    read_jsonl,
    run_dir,
    run_id,
    write_csv,
)


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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"用户：{prompt}\n助手："


def format_chat_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{format_prompt_text(tokenizer, prompt)}{response}"


def make_tokenized_row(tokenizer, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt_text = format_prompt_text(tokenizer, row["prompt"])
    full_text = format_chat_text(tokenizer, row["prompt"], row["response"])
    full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
    labels = list(full["input_ids"])
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len
    full["labels"] = labels
    return full


def collate_batch(tokenizer, features: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_mask = []
    labels = []
    pad_id = tokenizer.pad_token_id
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def target_steps(config: dict[str, Any], train_rows: int) -> tuple[int, list[int]]:
    updates_per_epoch = math.ceil(
        train_rows
        / int(config["per_device_train_batch_size"])
        / int(config["gradient_accumulation_steps"])
    )
    total_steps = max(1, int(math.ceil(updates_per_epoch * float(config["num_train_epochs"]))))
    targets = {
        max(0, min(total_steps, int(round(total_steps * float(fraction)))))
        for fraction in config["checkpoint_fractions"]
    }
    targets.add(0)
    targets.add(total_steps)
    return total_steps, sorted(targets)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    data_path = Path(config["data_dir"]) / FAMILY_TO_FILE[args.family]
    rows = read_jsonl(data_path, limit=int(config["train_per_family"]))
    total_steps, targets = target_steps(config, len(rows))
    rid = run_id(config, args.family, args.seed)
    root = run_dir(config, args.family, args.seed)
    index_path = root / "checkpoint_index.csv"

    print("config:", args.config)
    print("run_id:", rid)
    print("family:", args.family)
    print("seed:", args.seed)
    print("model:", config["model_name_or_path"])
    print("train_rows:", len(rows))
    print("total_steps:", total_steps)
    print("target_steps:", targets)
    print("run_dir:", root)
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
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, transformers, peft, datasets, accelerate and bitsandbytes."
        ) from exc

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
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

    tokenized = Dataset.from_list(
        [make_tokenized_row(tokenizer, row, int(config["max_length"])) for row in rows]
    )

    checkpoint_rows: list[dict[str, Any]] = []
    saved_steps: set[int] = set()

    def save_adapter(step: int) -> None:
        tag = checkpoint_tag(step, total_steps)
        out_dir = checkpoint_dir(config, args.family, args.seed, tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        if step not in saved_steps:
            checkpoint_rows.append(
                {
                    "run_id": rid,
                    "family": args.family,
                    "seed": args.seed,
                    "checkpoint": tag,
                    "step": step,
                    "total_steps": total_steps,
                    "fraction": step / total_steps if total_steps else 0.0,
                    "adapter_dir": str(out_dir),
                }
            )
            saved_steps.add(step)
            write_csv(index_path, sorted(checkpoint_rows, key=lambda row: int(row["step"])))
        print(f"saved checkpoint {tag} to {out_dir}")

    class SaveTargetStepsCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            step = int(state.global_step)
            if step in targets and step not in saved_steps:
                save_adapter(step)
            return control

        def on_train_end(self, args_, state, control, **kwargs):
            step = int(state.global_step)
            if total_steps not in saved_steps:
                save_adapter(total_steps)
            elif step not in saved_steps and step in targets:
                save_adapter(step)
            return control

    training_args = TrainingArguments(
        output_dir=str(root / "trainer_logs"),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_strategy="no",
        bf16=config.get("bf16", False),
        fp16=config.get("fp16", False),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    save_adapter(0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=lambda features: collate_batch(tokenizer, features),
        callbacks=[SaveTargetStepsCallback()],
    )
    trainer.train()
    write_csv(index_path, sorted(checkpoint_rows, key=lambda row: int(row["step"])))
    print(f"wrote {index_path}")


if __name__ == "__main__":
    main()
