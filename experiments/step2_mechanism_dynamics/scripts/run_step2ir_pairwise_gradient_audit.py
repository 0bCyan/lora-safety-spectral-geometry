from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from common import FAMILY_TO_FILE, checkpoint_dir, load_json, read_jsonl, run_id, write_csv
from train_lora_with_checkpoints import get_arm_config, make_tokenized_row


ARMS = ["fixed_matched_5", "fixed_matched_10", "diverse_matched_5", "diverse_matched_10"]
CHECKPOINTS = ["step0000_pct000", "step0016_pct025", "step0032_pct050", "step0064_pct100"]
GROUPS = ["attention", "mlp", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", default="risk", choices=sorted(FAMILY_TO_FILE))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--checkpoints", default=",".join(CHECKPOINTS))
    parser.add_argument("--block-size", type=int, default=8)
    return parser.parse_args()


def read_checkpoint_index(config: dict[str, Any], family: str, seed: int, run_label: str) -> dict[str, dict[str, str]]:
    path = Path(config["output_dir"]) / "runs" / run_id(config, family, seed, run_label) / "checkpoint_index.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["checkpoint"]: row for row in csv.DictReader(handle)}


def fixed_safety_rows(config: dict[str, Any], arm: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(arm["anchor_source_path"]), limit=limit)
    response_field = str(arm.get("anchor_response_field", "response"))
    return [
        {
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "response": row[response_field],
        }
        for row in rows
    ]


def load_model(config: dict[str, Any], adapter_dir: Path):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    base = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False
    if config.get("load_in_4bit", False):
        base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.eval()
    return model, tokenizer


def collate_one(tokenizer: Any, feature: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([feature["attention_mask"]], dtype=torch.long),
        "labels": torch.tensor([feature["labels"]], dtype=torch.long),
    }


def parameter_group(name: str) -> str:
    if any(token in name for token in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attention"
    if any(token in name for token in ["gate_proj", "up_proj", "down_proj"]):
        return "mlp"
    return "other"


def gradient_vectors_for_row(model: Any, tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[dict[str, Any], float]:
    import torch

    model.zero_grad(set_to_none=True)
    feature = make_tokenized_row(tokenizer, row, max_length)
    batch = collate_one(tokenizer, feature)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    outputs = model(**batch)
    outputs.loss.backward()
    pieces: dict[str, list[Any]] = {group: [] for group in GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_group(name)
        grad = parameter.grad
        if grad is None:
            pieces[group].append(torch.zeros(parameter.numel(), dtype=torch.float32, device="cpu"))
        else:
            pieces[group].append(grad.detach().float().reshape(-1).cpu())
    model.zero_grad(set_to_none=True)
    vectors = {
        group: torch.cat(items) if items else torch.empty(0, dtype=torch.float32)
        for group, items in pieces.items()
    }
    vectors["all"] = torch.cat([vectors[group] for group in GROUPS if vectors[group].numel() > 0])
    return vectors, float(outputs.loss.detach().float().cpu())


def cosine(left: Any, right: Any) -> float:
    import torch

    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return math.nan
    return float(torch.dot(left, right) / (left_norm * right_norm))


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["cosine"] == row["cosine"]]
    values = [float(row["cosine"]) for row in valid_rows]
    negative = [value for value in values if value < 0]
    risk_norms = [float(row["risk_gradient_norm"]) for row in valid_rows]
    safety_norms = [float(row["safety_gradient_norm"]) for row in valid_rows]
    return {
        "pair_count": len(values),
        "mean_cosine": mean(values) if values else math.nan,
        "median_cosine": median(values) if values else math.nan,
        "negative_pair_fraction": len(negative) / len(values) if values else math.nan,
        "p10_cosine": quantile(values, 0.10),
        "p90_cosine": quantile(values, 0.90),
        "mean_negative_cosine_magnitude": mean([-value for value in negative]) if negative else 0.0,
        "mean_risk_gradient_norm": mean(risk_norms) if risk_norms else math.nan,
        "mean_safety_gradient_norm": mean(safety_norms) if safety_norms else math.nan,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    checkpoints = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    block_size = int(args.block_size)
    risk_rows = read_jsonl(Path(config["data_dir"]) / FAMILY_TO_FILE[args.family], limit=block_size * 2)
    out_dir = Path(config["output_dir"]) / "gradient_pairwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []

    for arm_label in arms:
        arm = get_arm_config(config, arm_label)
        safety_rows = fixed_safety_rows(config, arm, block_size * 2)
        index = read_checkpoint_index(config, args.family, args.seed, arm_label)
        for checkpoint in checkpoints:
            adapter = checkpoint_dir(config, args.family, args.seed, checkpoint, arm_label)
            print(f"pairwise gradient audit {arm_label} {checkpoint}")
            model, tokenizer = load_model(config, adapter)
            for block in range(2):
                risk_block = risk_rows[block * block_size : (block + 1) * block_size]
                safety_block = safety_rows[block * block_size : (block + 1) * block_size]
                risk_vectors = []
                safety_vectors = []
                risk_losses = []
                safety_losses = []
                for row in risk_block:
                    vectors, loss = gradient_vectors_for_row(model, tokenizer, row, int(config["max_length"]))
                    risk_vectors.append((row["sample_id"], vectors))
                    risk_losses.append(loss)
                for row in safety_block:
                    vectors, loss = gradient_vectors_for_row(model, tokenizer, row, int(config["max_length"]))
                    safety_vectors.append((row["sample_id"], vectors))
                    safety_losses.append(loss)
                block_pair_rows: list[dict[str, Any]] = []
                block_group_rows: dict[str, list[dict[str, Any]]] = {group: [] for group in ["all"] + GROUPS}
                for risk_index, (risk_id, risk_vecs) in enumerate(risk_vectors):
                    for safety_index, (safety_id, safety_vecs) in enumerate(safety_vectors):
                        for group in ["all"] + GROUPS:
                            value = cosine(risk_vecs[group], safety_vecs[group])
                            item = {
                                "arm": arm_label,
                                "family": args.family,
                                "seed": args.seed,
                                "checkpoint": checkpoint,
                                "step": index[checkpoint]["step"],
                                "block": block + 1,
                                "group": group,
                                "risk_index": risk_index,
                                "safety_index": safety_index,
                                "risk_sample_id": risk_id,
                                "safety_sample_id": safety_id,
                                "cosine": value,
                                "risk_gradient_norm": float(__import__("torch").linalg.vector_norm(risk_vecs[group])),
                                "safety_gradient_norm": float(__import__("torch").linalg.vector_norm(safety_vecs[group])),
                            }
                            pair_rows.append(item)
                            block_group_rows[group].append(item)
                            if group == "all":
                                block_pair_rows.append(item)
                summary = summarize_pairs(block_pair_rows)
                summary.update(
                    {
                        "arm": arm_label,
                        "family": args.family,
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "step": index[checkpoint]["step"],
                        "block": block + 1,
                        "mean_risk_nll": mean(risk_losses),
                        "mean_safety_nll": mean(safety_losses),
                    }
                )
                summary_rows.append(summary)
                for group, rows in block_group_rows.items():
                    group_summary = summarize_pairs([row for row in rows if row["group"] == group])
                    group_summary.update(
                        {
                            "arm": arm_label,
                            "family": args.family,
                            "seed": args.seed,
                            "checkpoint": checkpoint,
                            "step": index[checkpoint]["step"],
                            "block": block + 1,
                            "group": group,
                        }
                    )
                    group_summary_rows.append(group_summary)
            del model, tokenizer
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    write_csv(out_dir / "step2ir_pairwise_gradient_pairs.csv", pair_rows)
    write_csv(out_dir / "step2ir_pairwise_gradient_summary.csv", summary_rows)
    write_csv(out_dir / "step2ir_pairwise_gradient_group_summary.csv", group_summary_rows)
    print(f"wrote {out_dir / 'step2ir_pairwise_gradient_pairs.csv'} ({len(pair_rows)} rows)")
    print(f"wrote {out_dir / 'step2ir_pairwise_gradient_summary.csv'} ({len(summary_rows)} rows)")
    print(f"wrote {out_dir / 'step2ir_pairwise_gradient_group_summary.csv'} ({len(group_summary_rows)} rows)")


if __name__ == "__main__":
    main()
