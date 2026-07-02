from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from activation_common import (
    adapter_dir,
    experiment_id,
    generation_path,
    hidden_state_path,
    load_json,
    local_judge_path,
    metadata_path,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=["base", "normal", "risk", "control"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_generation_and_labels(base_config: dict[str, Any], family: str, seed: int, eval_set: str) -> list[dict[str, Any]]:
    generations = read_jsonl(generation_path(base_config, family, seed, eval_set))
    labels = {
        row["sample_id"]: row
        for row in read_jsonl(local_judge_path(base_config, family, seed, eval_set))
    }
    rows = []
    for row in generations:
        label = labels[row["sample_id"]]
        rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt": row["prompt"],
                "family": family,
                "seed": seed,
                "eval_set": eval_set,
                "label": label["label"],
                "local_reason": label.get("local_reason", ""),
                "response_chars": label.get("response_chars", len(row.get("response", ""))),
                "cleaned_response_chars": label.get("cleaned_response_chars", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(config["base_config"])
    eval_set = config["eval_set"]
    output_dir = Path(config["output_dir"])
    exp_id = experiment_id(base_config, args.family, args.seed)
    rows = load_generation_and_labels(base_config, args.family, args.seed, eval_set)
    state_out = hidden_state_path(output_dir, exp_id, eval_set)
    meta_out = metadata_path(output_dir, exp_id, eval_set)

    print("experiment_id:", exp_id)
    print("family:", args.family)
    print("seed:", args.seed)
    print("rows:", len(rows))
    print("hidden_states:", state_out)
    print("metadata:", meta_out)
    if args.dry_run:
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if args.family != "base":
            from peft import PeftModel
    except Exception as exc:
        raise RuntimeError(
            "Missing activation dependencies. Install torch, transformers, peft and bitsandbytes."
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
    if args.family == "base":
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, adapter_dir(base_config, args.family, args.seed))
    model.eval()

    captured = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {row['prompt']}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        layer_states = [
            hidden[0, -1, :].detach().float().cpu()
            for hidden in outputs.hidden_states[1:]
        ]
        captured.append(torch.stack(layer_states, dim=0))

    state_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": exp_id,
            "family": args.family,
            "seed": args.seed,
            "eval_set": eval_set,
            "capture_mode": config["capture_mode"],
            "hidden_states": torch.stack(captured, dim=0),
            "sample_ids": [row["sample_id"] for row in rows],
        },
        state_out,
    )
    write_jsonl(meta_out, rows)
    print(f"wrote {state_out}")
    print(f"wrote {meta_out}")


if __name__ == "__main__":
    main()
