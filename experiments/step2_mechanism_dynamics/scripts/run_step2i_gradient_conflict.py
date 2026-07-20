from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

from common import FAMILY_TO_FILE, checkpoint_dir, load_json, read_jsonl, run_id, write_csv
from train_lora_with_checkpoints import (
    collate_batch,
    format_chat_text,
    get_arm_config,
    make_tokenized_row,
)


CHECKPOINT_ORDER = {
    "step0000_pct000": 0,
    "step0006_pct009": 6,
    "step0016_pct025": 16,
    "step0032_pct050": 32,
    "step0048_pct075": 48,
    "step0064_pct100": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_TO_FILE))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-labels", required=True, help="Comma-separated Step2I arm labels.")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint tags. Defaults to index CSV.")
    parser.add_argument("--risk-limit", type=int, default=16)
    parser.add_argument("--safety-limit", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_checkpoint_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_rows(config: dict[str, Any], family: str, seed: int, run_label: str, requested: str | None) -> list[dict[str, str]]:
    index_path = Path(config["output_dir"]) / "runs" / run_id(config, family, seed, run_label) / "checkpoint_index.csv"
    rows = read_checkpoint_index(index_path)
    if requested:
        wanted = {item.strip() for item in requested.split(",") if item.strip()}
        rows = [row for row in rows if row["checkpoint"] in wanted]
        missing = wanted - {row["checkpoint"] for row in rows}
        if missing:
            raise ValueError(f"Missing checkpoints for {run_label}: {sorted(missing)}")
    rows.sort(key=lambda row: CHECKPOINT_ORDER.get(row["checkpoint"], int(row.get("step", 999999))))
    return rows


def safety_prompt_rows(config: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    source = Path(config["source_step2g_safety_train"])
    return read_jsonl(source, limit=limit)


def diverse_targets_by_prompt(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(Path(config["diverse_target_manifest"]))
    selected: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in selected:
            selected[sample_id] = row
    return dict(selected)


def diagnostic_safety_rows(config: dict[str, Any], arm: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    prompts = safety_prompt_rows(config, limit)
    response_field = str(arm.get("anchor_response_field", ""))
    if response_field:
        targets = diverse_targets_by_prompt(config)
        rows = []
        for row in prompts:
            target = targets[row["sample_id"]]
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "target_id": target.get("target_id", row["sample_id"]),
                    "prompt": row["prompt"],
                    "response": target[response_field],
                }
            )
        return rows
    fixed_response = str(arm["anchor_response"])
    return [
        {
            "sample_id": row["sample_id"],
            "target_id": f"{row['sample_id']}_fixed",
            "prompt": row["prompt"],
            "response": fixed_response,
        }
        for row in prompts
    ]


def write_manifest(
    path: Path,
    config: dict[str, Any],
    family: str,
    seed: int,
    risk_rows: list[dict[str, Any]],
    safety_rows_by_arm: dict[str, list[dict[str, Any]]],
) -> None:
    payload = {
        "config": config["name"],
        "family": family,
        "seed": seed,
        "risk_rows": [{"sample_id": row["sample_id"], "content_hash": row.get("content_hash", "")} for row in risk_rows],
        "safety_rows_by_arm": {
            arm: [
                {
                    "sample_id": row["sample_id"],
                    "target_id": row.get("target_id", ""),
                    "response_chars": len(row["response"]),
                }
                for row in rows
            ]
            for arm, rows in safety_rows_by_arm.items()
        },
        "definition": {
            "gradient_cosine": "dot(risk_grad, safety_grad) / (norm(risk_grad) * norm(safety_grad))",
            "projected_conflicting_gradient_fraction": "max(0, -gradient_cosine)",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def trainable_gradient_vector(model):
    import torch

    pieces = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            grad = parameter.grad
            if grad is None:
                pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32, device="cpu"))
            else:
                pieces.append(grad.detach().float().reshape(-1).cpu())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces)


def gradient_for_rows(model, tokenizer, rows: list[dict[str, Any]], max_length: int, micro_batch_size: int):
    import torch

    model.zero_grad(set_to_none=True)
    tokenized = [make_tokenized_row(tokenizer, row, max_length) for row in rows]
    losses = []
    batches = chunked(tokenized, micro_batch_size)
    for batch_rows in batches:
        batch = collate_batch(tokenizer, batch_rows)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / max(1, len(batches))
        loss.backward()
        losses.append(float(outputs.loss.detach().float().cpu()))
    vector = trainable_gradient_vector(model)
    model.zero_grad(set_to_none=True)
    return vector, sum(losses) / len(losses) if losses else math.nan


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
    model.train()
    return model, tokenizer


def cosine_and_norms(risk_grad, safety_grad) -> tuple[float, float, float, float]:
    import torch

    risk_norm = float(torch.linalg.vector_norm(risk_grad).item())
    safety_norm = float(torch.linalg.vector_norm(safety_grad).item())
    if risk_norm == 0.0 or safety_norm == 0.0:
        return math.nan, risk_norm, safety_norm, math.nan
    cosine = float(torch.dot(risk_grad, safety_grad).item() / (risk_norm * safety_norm))
    return cosine, risk_norm, safety_norm, max(0.0, -cosine)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    run_labels = [item.strip() for item in args.run_labels.split(",") if item.strip()]
    risk_rows = read_jsonl(Path(config["data_dir"]) / FAMILY_TO_FILE[args.family], limit=args.risk_limit)
    safety_by_arm = {
        run_label: diagnostic_safety_rows(config, get_arm_config(config, run_label), args.safety_limit)
        for run_label in run_labels
    }
    out_dir = Path(config["output_dir"]) / "gradient_conflict"
    output = Path(args.output) if args.output else out_dir / "step2i_gradient_conflict_seed1.csv"
    write_manifest(out_dir / "step2i_gradient_conflict_manifest.json", config, args.family, args.seed, risk_rows, safety_by_arm)

    rows: list[dict[str, Any]] = []
    for run_label in run_labels:
        for index_row in checkpoint_rows(config, args.family, args.seed, run_label, args.checkpoints):
            checkpoint = index_row["checkpoint"]
            adapter = checkpoint_dir(config, args.family, args.seed, checkpoint, run_label)
            print(f"diagnosing {run_label} {checkpoint}: {adapter}")
            model, tokenizer = load_model(config, adapter)
            risk_grad, risk_nll = gradient_for_rows(
                model, tokenizer, risk_rows, int(config["max_length"]), args.micro_batch_size
            )
            safety_grad, safety_nll = gradient_for_rows(
                model, tokenizer, safety_by_arm[run_label], int(config["max_length"]), args.micro_batch_size
            )
            cosine, risk_norm, safety_norm, conflict = cosine_and_norms(risk_grad, safety_grad)
            rows.append(
                {
                    "arm": run_label,
                    "family": args.family,
                    "seed": args.seed,
                    "checkpoint": checkpoint,
                    "step": index_row.get("step", CHECKPOINT_ORDER.get(checkpoint, "")),
                    "risk_n": len(risk_rows),
                    "safety_n": len(safety_by_arm[run_label]),
                    "risk_nll": risk_nll,
                    "safety_target_nll": safety_nll,
                    "gradient_cosine": cosine,
                    "risk_gradient_norm": risk_norm,
                    "safety_gradient_norm": safety_norm,
                    "projected_conflicting_gradient_fraction": conflict,
                }
            )
            del model, tokenizer, risk_grad, safety_grad
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    rows.sort(key=lambda row: (run_labels.index(row["arm"]), int(row["step"])))
    write_csv(output, rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
