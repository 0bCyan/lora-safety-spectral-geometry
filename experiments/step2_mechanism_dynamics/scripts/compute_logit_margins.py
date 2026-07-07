from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from statistics import mean
from typing import Any

from common import (
    EVAL_TO_FILE,
    checkpoint_dir,
    load_json,
    logit_margin_path,
    read_jsonl,
    run_dir,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=["normal", "risk", "control"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-label", default=None, help="Optional run label used for regularized Step2B arms.")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_checkpoint_index(config: dict[str, Any], family: str, seed: int, run_label: str | None) -> list[dict[str, str]]:
    with (run_dir(config, family, seed, run_label) / "checkpoint_index.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def token_ids_for_strings(tokenizer, strings: list[str]) -> list[int]:
    ids: list[int] = []
    for text in strings:
        for variant in [text, " " + text]:
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if encoded:
                ids.append(int(encoded[-1]))
    return sorted(set(ids))


def set_logprob(log_probs, token_ids: list[int]) -> float:
    import torch

    if not token_ids:
        return float("nan")
    values = log_probs[token_ids]
    return float(torch.logsumexp(values, dim=0).detach().float().cpu().item())


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    margin_config = config["logit_margin"]
    eval_set = margin_config["eval_set"]
    eval_rows = read_jsonl(
        Path(config["data_dir"]) / EVAL_TO_FILE[eval_set],
        limit=int(margin_config["limit"]),
    )
    index_rows = read_checkpoint_index(config, args.family, args.seed, args.run_label)
    if args.checkpoints:
        wanted = {item.strip() for item in args.checkpoints.split(",") if item.strip()}
        index_rows = [row for row in index_rows if row["checkpoint"] in wanted]

    print("family:", args.family)
    print("seed:", args.seed)
    print("run_label:", args.run_label or "")
    print("eval_rows:", len(eval_rows))
    print("checkpoints:", [row["checkpoint"] for row in index_rows])
    if args.dry_run:
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("Missing logit margin dependencies.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_sets = {
        "refusal": token_ids_for_strings(tokenizer, margin_config["refusal_strings"]),
        "compliance": token_ids_for_strings(tokenizer, margin_config["compliance_strings"]),
        "action": token_ids_for_strings(tokenizer, margin_config["action_strings"]),
    }
    print("token_sets:", token_sets)

    quantization_config = None
    if config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
        )
    summary_rows = []
    for index_row in index_rows:
        checkpoint = index_row["checkpoint"]
        adapter = checkpoint_dir(config, args.family, args.seed, checkpoint, args.run_label)
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter)
        model.eval()
        rows = []
        for row in eval_rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = f"用户：{row['prompt']}\n助手："
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            refusal_lp = set_logprob(log_probs, token_sets["refusal"])
            compliance_lp = set_logprob(log_probs, token_sets["compliance"])
            action_lp = set_logprob(log_probs, token_sets["action"])
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "family": args.family,
                    "run_label": args.run_label or "",
                    "seed": args.seed,
                    "checkpoint": checkpoint,
                    "step": index_row["step"],
                    "fraction": index_row["fraction"],
                    "eval_set": eval_set,
                    "refusal_logprob": refusal_lp,
                    "compliance_logprob": compliance_lp,
                    "action_logprob": action_lp,
                    "refusal_minus_compliance": refusal_lp - compliance_lp,
                    "refusal_minus_action": refusal_lp - action_lp,
                }
            )
        write_jsonl(logit_margin_path(config, args.family, args.seed, checkpoint, args.run_label), rows)
        summary_rows.append(
            {
                "family": args.family,
                "run_label": args.run_label or "",
                "seed": args.seed,
                "checkpoint": checkpoint,
                "step": index_row["step"],
                "fraction": index_row["fraction"],
                "eval_set": eval_set,
                "n": len(rows),
                "mean_refusal_logprob": mean(row["refusal_logprob"] for row in rows),
                "mean_compliance_logprob": mean(row["compliance_logprob"] for row in rows),
                "mean_action_logprob": mean(row["action_logprob"] for row in rows),
                "mean_refusal_minus_compliance": mean(row["refusal_minus_compliance"] for row in rows),
                "mean_refusal_minus_action": mean(row["refusal_minus_action"] for row in rows),
            }
        )
        print(f"wrote {logit_margin_path(config, args.family, args.seed, checkpoint, args.run_label)} ({len(rows)} rows)")
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(Path(config["output_dir"]) / "logit_margins" / "summary.csv", summary_rows)
    print(f"wrote {Path(config['output_dir']) / 'logit_margins' / 'summary.csv'}")


if __name__ == "__main__":
    main()
